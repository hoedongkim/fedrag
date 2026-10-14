"""Tests for grounded answer generation with a fake retriever and a fake chat model (no OpenAI)."""

import asyncio
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI
from pydantic import Field

from app.core.config import Settings
from app.rag.answer import (
    LLM_NOT_CONFIGURED,
    LLMNotConfiguredError,
    answer_question,
    get_llm,
    stream_answer,
    trim_snippet,
)
from app.rag.types import Chunk, RetrievedChunk
from app.schemas.ask import (
    AskRequest,
    AskResponse,
    StreamCitation,
    StreamDelta,
    StreamStatus,
)
from app.schemas.common import DocType

STATEMENT = "FOMC Statement, September 18, 2024"
MINUTES = "Minutes of the FOMC Meeting, September 17-18, 2024"
CUT = (
    "The Committee decided to lower the target range for the federal funds rate by 1/2 "
    "percentage point to 4-3/4 to 5 percent."
)
RISKS = "The Committee judges that the risks to achieving its goals are roughly in balance."
LABOR = "Job gains have slowed, and the unemployment rate has moved up but remains low."
MINUTES_VIEW = (
    "Some participants noted that they would have preferred a 25 basis point reduction of the "
    "target range at this meeting."
)


def hit(
    index: int,
    text: str,
    *,
    doc_type: DocType = "statement",
    released: date = date(2024, 9, 18),
    section: str | None = None,
    page_updated: date | None = None,
    score: float = 0.5,
) -> RetrievedChunk:
    doc_id = f"{doc_type}-2024-09-18"
    title = MINUTES if doc_type == "minutes" else STATEMENT
    chunk = Chunk(
        chunk_id=f"{doc_id}#{index:04d}",
        doc_id=doc_id,
        doc_type=doc_type,
        meeting_date=date(2024, 9, 18),
        release_date=released,
        page_updated=page_updated,
        title=title,
        url=f"https://www.federalreserve.gov/{doc_id}.htm",
        section=section,
        index=index,
        text=text,
        context=title if section is None else f"{title} | {section}",
    )
    return RetrievedChunk(chunk=chunk, score=score)


PASSAGES = [
    hit(0, CUT, score=0.91),
    hit(1, RISKS, score=0.72),
    hit(2, LABOR, score=0.55),
    hit(
        7,
        MINUTES_VIEW,
        doc_type="minutes",
        released=date(2024, 10, 9),
        section="Committee Policy Actions",
        page_updated=date(2024, 10, 9),
        score=0.4,
    ),
]


@dataclass
class FakeRetriever:
    """Returns its hits for any query and records how it was called."""

    hits: Sequence[RetrievedChunk]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        as_of: date | None = None,
        doc_types: Sequence[DocType] | None = None,
    ) -> list[RetrievedChunk]:
        self.calls.append({"query": query, "top_k": top_k, "as_of": as_of, "doc_types": doc_types})
        return list(self.hits[:top_k])


class FakeChatModel(GenericFakeChatModel):
    """Replies with canned answers, streamed in pieces split at whitespace, and records each
    prompt."""

    prompts: list[list[BaseMessage]] = Field(default_factory=list)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.prompts.append(messages)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def fake_llm(*replies: str) -> FakeChatModel:
    return FakeChatModel(messages=iter(replies))


def ask(
    answer: str,
    passages: Sequence[RetrievedChunk] = PASSAGES,
    **request: Any,
) -> tuple[Any, FakeRetriever, FakeChatModel]:
    retriever = FakeRetriever(passages)
    llm = fake_llm(answer)
    request = {"question": "What did the Committee decide?", "top_k": 5, **request}
    response = answer_question(
        AskRequest(**request), retriever=retriever, llm=llm, model_name="fake-model"
    )
    return response, retriever, llm


def chunk_ids(response: Any) -> list[str]:
    return [citation.chunk_id for citation in response.citations]


def test_citations_follow_first_appearance_and_markers_are_renumbered() -> None:
    response, _, _ = ask(
        "The Committee cut rates by 1/2 percentage point [3][1]. Risks were balanced [2]. "
        "Some participants preferred a smaller cut [4], citing the labor market [3].",
    )

    assert response.answer == (
        "The Committee cut rates by 1/2 percentage point [1][2]. Risks were balanced [3]. "
        "Some participants preferred a smaller cut [4], citing the labor market [1]."
    )
    assert chunk_ids(response) == [
        "statement-2024-09-18#0002",
        "statement-2024-09-18#0000",
        "statement-2024-09-18#0001",
        "minutes-2024-09-18#0007",
    ]


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("Rates fell [1, 3].", "Rates fell [1][2]."),
        ("Rates fell [1,3] [3].", "Rates fell [1][2]."),
        ("Rates fell [1-3].", "Rates fell [1][2][3]."),
        ("Rates fell [1] [3]; risks [2].", "Rates fell [1][2]; risks [3]."),
    ],
    ids=["comma-list", "duplicate", "range", "adjacent"],
)
def test_marker_groups_are_normalized(answer: str, expected: str) -> None:
    response, _, _ = ask(answer)

    assert response.answer == expected


def test_markers_naming_no_passage_are_dropped() -> None:
    response, _, _ = ask(
        "Rates fell by 1/2 percentage point [9]. Risks were balanced [2][7]. Jobs slowed [0].",
        passages=PASSAGES[:2],
    )

    assert response.answer == (
        "Rates fell by 1/2 percentage point. Risks were balanced [1]. Jobs slowed."
    )
    assert chunk_ids(response) == ["statement-2024-09-18#0001"]


def test_only_passages_cited_inline_are_citations() -> None:
    response, _, _ = ask("Rates fell [2].")

    assert response.answer == "Rates fell [1]."
    assert chunk_ids(response) == ["statement-2024-09-18#0001"]


def test_answer_without_markers_has_no_citations() -> None:
    response, _, _ = ask("The Committee lowered the target range.")

    assert response.answer == "The Committee lowered the target range."
    assert response.citations == []


def test_unsupported_answer_has_no_citations() -> None:
    answer = "The provided FOMC passages do not say how the balance sheet runoff will change."

    response, _, llm = ask(answer)

    assert response.answer == answer
    assert response.citations == []
    assert len(llm.prompts) == 1


def test_request_filters_reach_the_retriever() -> None:
    _, retriever, _ = ask(
        "Rates fell [1].",
        question="  How did forward guidance change?  ",
        as_of="2024-10-15",
        doc_types=["minutes", "statement"],
        top_k=3,
        retrieval="bm25",
    )

    assert retriever.calls == [
        {
            "query": "How did forward guidance change?",
            "top_k": 3,
            "as_of": date(2024, 10, 15),
            "doc_types": ["minutes", "statement"],
        }
    ]


def test_prompt_numbers_passages_and_states_the_as_of_date() -> None:
    _, _, llm = ask("Rates fell [1].", as_of="2024-10-15")

    [system, human] = llm.prompts[0]
    assert system.type == "system"
    assert "Answer as of 2024-10-15" in system.content
    assert "Use only the passages" in system.content
    assert "only if a passage says so" in system.content  # no inferred start dates or firsts
    assert "later decisions may be missing" in system.content  # stale passages are flagged
    assert human.content.startswith("As of: 2024-10-15\n\nPassages:\n\n[1] ")
    assert f"[1] {STATEMENT} | released 2024-09-18\n{CUT}" in human.content
    assert (
        f"[4] {MINUTES} | Committee Policy Actions | released 2024-10-09\n{MINUTES_VIEW}"
        in human.content
    )
    assert human.content.endswith("Question: What did the Committee decide?")


def test_prompt_without_as_of_has_no_cutoff() -> None:
    _, _, llm = ask("Rates fell [1].")

    [system, human] = llm.prompts[0]
    assert "There is no date cutoff" in system.content
    assert human.content.startswith("As of: no cutoff\n")


def test_answer_is_the_model_text_trimmed() -> None:
    response, _, llm = ask("\n  Rates fell [1].\n\nRisks were balanced [2].  \n")

    assert response.answer == "Rates fell [1].\n\nRisks were balanced [2]."
    assert "cited_passages" not in llm.prompts[0][0].content  # plain text, no JSON fields


def test_passages_outside_the_filters_never_reach_the_llm(caplog: pytest.LogCaptureFixture) -> None:
    # A retriever that ignored as_of and doc_types: the minutes came out after as_of.
    with caplog.at_level(logging.ERROR):
        response, _, llm = ask(
            "Rates fell [1]. Some preferred less [2].",
            as_of="2024-09-30",
            doc_types=["statement"],
            passages=[PASSAGES[0], PASSAGES[3]],
        )

    assert "Some participants" not in llm.prompts[0][1].content
    assert response.answer == "Rates fell [1]. Some preferred less."
    assert chunk_ids(response) == ["statement-2024-09-18#0000"]
    assert "Dropped 1 retrieved passages" in caplog.text


@pytest.mark.parametrize(
    ("request_fields", "message"),
    [
        (
            {"as_of": "1990-01-01", "doc_types": ["statement"]},
            "No indexed FOMC passages match the filters (released on or before 1990-01-01; "
            "of type statement), so there is nothing to answer from.",
        ),
        ({}, "No indexed FOMC passages were found, so there is nothing to answer from."),
    ],
    ids=["filters", "no-filters"],
)
def test_no_passages_skips_the_llm(request_fields: dict[str, Any], message: str) -> None:
    retriever = FakeRetriever([])
    llm = fake_llm()

    response = answer_question(
        AskRequest(question="What happened in 1990?", **request_fields),
        retriever=retriever,
        llm=llm,
        model_name="fake-model",
    )

    assert response.answer == message
    assert response.citations == []
    assert llm.prompts == []


def test_citation_carries_document_section_and_score() -> None:
    response, _, _ = ask("Some participants preferred a 25 basis point cut [4].")

    [citation] = response.citations
    assert citation.model_dump(mode="json") == {
        "document": {
            "doc_id": "minutes-2024-09-18",
            "doc_type": "minutes",
            "meeting_date": "2024-09-18",
            "release_date": "2024-10-09",
            "page_updated": "2024-10-09",
            "title": MINUTES,
            "url": "https://www.federalreserve.gov/minutes-2024-09-18.htm",
        },
        "chunk_id": "minutes-2024-09-18#0007",
        "section": "Committee Policy Actions",
        "snippet": MINUTES_VIEW,
        "passage": None,  # live answers: only recorded ones carry the whole passage
        "score": 0.4,
    }


def test_revised_page_date_flows_into_the_source_document() -> None:
    revised = hit(3, LABOR, page_updated=date(2026, 10, 9))

    response, _, _ = ask("Jobs slowed [1].", passages=[revised])

    assert response.citations[0].document.page_updated == date(2026, 10, 9)
    assert response.citations[0].document.release_date == date(2024, 9, 18)


def test_an_old_newest_citation_gets_a_release_date_note() -> None:
    response, _, _ = ask("Rates fell [1].", as_of="2025-06-01")

    assert response.answer == "Rates fell [1]."  # only the model's text
    assert response.notes == [
        "The most recent passage cited was released Sep 18, 2024, 256 days before Jun 1, 2025; "
        "later decisions may not be reflected."
    ]
    # The newest citation counts: the October minutes are recent enough.
    assert ask("Rates fell [1], some preferred less [4].", as_of="2024-12-31")[0].notes == []
    # A recent enough citation (27 days), no as_of, or no citation: no note.
    for fields in ({"as_of": "2024-10-15"}, {}):
        assert ask("Rates fell [1].", **fields)[0].notes == []
    refusal = "The provided FOMC passages do not say."
    assert ask(refusal, as_of="2025-06-01")[0].notes == []


def test_response_reports_method_cutoff_model_and_latency() -> None:
    response, _, _ = ask("Rates fell [1].", retrieval="dense", as_of="2024-10-15")

    assert response.retrieval == "dense"
    assert response.as_of == date(2024, 10, 15)
    assert response.model == "fake-model"
    assert response.latency_ms >= 0


def test_long_passage_snippet_is_trimmed() -> None:
    long_text = " ".join([CUT, RISKS, LABOR] * 6)

    response, _, _ = ask("Rates fell [1].", passages=[hit(0, long_text)])

    snippet = response.citations[0].snippet
    assert len(long_text) > 600 >= len(snippet)
    assert long_text.startswith(snippet)
    assert snippet.endswith(".")


SENTENCE = "Participants noted that financial conditions had eased somewhat over the period."


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("  Short passage.  ", "Short passage."),
        (
            f"{SENTENCE} {SENTENCE} {SENTENCE} Yields on U.S. Treasury securities rose "
            + "steadily " * 60,
            f"{SENTENCE} {SENTENCE} {SENTENCE}",
        ),
        (
            f"{SENTENCE} {SENTENCE} {SENTENCE} Voting for this action were Jerome H. Powell, "
            + "Chair, " * 90,
            f"{SENTENCE} {SENTENCE} {SENTENCE}",
        ),
        (
            f"{SENTENCE} Yields rose " + "steadily " * 80,
            f"{SENTENCE} Yields rose " + "steadily " * 55 + "steadily…",
        ),
    ],
    ids=[
        "short",
        "abbreviation-is-not-a-sentence-end",
        "initial-is-not-a-sentence-end",
        "word-cut-when-sentences-too-short",
    ],
)
def test_trim_snippet(text: str, expected: str) -> None:
    assert trim_snippet(text) == expected
    assert len(trim_snippet(text)) <= 600


def test_get_llm_requires_a_key() -> None:
    with pytest.raises(LLMNotConfiguredError, match="set OPENAI_API_KEY"):
        get_llm(Settings(_env_file=None))
    assert LLM_NOT_CONFIGURED == "LLM not configured: set OPENAI_API_KEY in backend/.env."


def test_get_llm_uses_low_reasoning_effort_and_no_temperature() -> None:
    settings = Settings(_env_file=None, openai_api_key="sk-test")

    llm = get_llm(settings)
    payload = llm._get_request_payload([("human", "Hi")])

    assert isinstance(llm, ChatOpenAI)
    assert llm is get_llm(Settings(_env_file=None, openai_api_key="sk-test"))
    assert payload["model"] == settings.llm_model
    assert payload["reasoning_effort"] == "low"
    assert "temperature" not in payload


def stream(
    answer: str, passages: Sequence[RetrievedChunk] = PASSAGES, **request: Any
) -> tuple[list[Any], AskResponse]:
    """The events stream_answer yields for `answer`, and what answer_question returns for it."""
    fields = {"question": "What did the Committee decide?", **request}

    async def collect() -> list[Any]:
        events = stream_answer(
            AskRequest(**fields),
            retriever=FakeRetriever(passages),
            llm=fake_llm(answer),
            model_name="fake-model",
        )
        return [event async for event in events]

    whole = answer_question(
        AskRequest(**fields),
        retriever=FakeRetriever(passages),
        llm=fake_llm(answer),
        model_name="fake-model",
    )
    return asyncio.run(collect()), whole


def without_latency(response: AskResponse) -> dict[str, Any]:
    return response.model_dump(exclude={"latency_ms"})


def test_stream_yields_stages_deltas_citations_then_the_answer() -> None:
    events, whole = stream("Rates fell [3][1]. Risks were balanced [2]. Jobs slowed [3].")

    assert events[:2] == [StreamStatus(stage="retrieving"), StreamStatus(stage="generating")]
    *body, done = events[2:]
    assert all(isinstance(event, StreamCitation | StreamDelta) for event in body)
    assert "".join(e.text for e in body if isinstance(e, StreamDelta)) == done.answer
    assert done.answer == "Rates fell [1][2]. Risks were balanced [3]. Jobs slowed [1]."
    assert without_latency(done) == without_latency(whole)
    citations = [e for e in body if isinstance(e, StreamCitation)]
    assert [c.n for c in citations] == [1, 2, 3]
    assert [c.citation for c in citations] == done.citations


def test_stream_sends_each_citation_before_the_delta_that_cites_it() -> None:
    events, _ = stream("Rates fell [2]. Risks were balanced [1]. More [2].")

    shown: set[int] = set()
    for event in events:
        if isinstance(event, StreamCitation):
            shown.add(event.n)
        elif isinstance(event, StreamDelta):
            cited = {int(n) for n in re.findall(r"\[(\d+)\]", event.text)}
            assert cited <= shown, event.text


def test_stream_applies_the_doc_types_guard_and_adds_notes() -> None:
    # A retriever that ignored doc_types: the minutes are not a statement.
    events, whole = stream(
        "Rates fell [1]. Some preferred less [2].",
        passages=[PASSAGES[0], PASSAGES[3]],
        as_of="2025-06-01",
        doc_types=["statement"],
    )

    done = events[-1]
    assert done.answer == "Rates fell [1]. Some preferred less."
    assert [c.chunk_id for c in done.citations] == ["statement-2024-09-18#0000"]
    assert done.notes == whole.notes and len(done.notes) == 1


def test_stream_applies_the_as_of_guard() -> None:
    # A retriever that ignored as_of: the minutes came out after it.
    events, whole = stream(
        "Rates fell [1]. Some preferred less [2].",
        passages=[PASSAGES[0], PASSAGES[3]],
        as_of="2024-09-30",
    )

    done = events[-1]
    assert done.answer == "Rates fell [1]. Some preferred less."
    assert [c.chunk_id for c in done.citations] == ["statement-2024-09-18#0000"]
    assert done.notes == whole.notes == []


def test_stream_without_passages_skips_the_llm() -> None:
    events, whole = stream("unused", passages=[], as_of="1990-01-01")

    [delta] = [e for e in events if isinstance(e, StreamDelta)]
    assert delta.text == whole.answer == events[-1].answer
    assert whole.answer.startswith("No indexed FOMC passages match the filters")
