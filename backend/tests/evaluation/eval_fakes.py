"""Fakes shared by the evaluation tests: a small chunk corpus and question set, a retriever, and
chat models for answers and the judge (no OpenAI)."""

import re
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import Field

from app.evaluation.judge_prompts import (
    AnswerJudgment,
    CitationVerdict,
    ClaimVerdict,
    FactVerdict,
)
from app.rag.types import Chunk, RetrievedChunk
from app.schemas.common import DocType

USAGE = {
    "input_tokens": 1_000,
    "output_tokens": 200,
    "total_tokens": 1_200,
    "input_token_details": {"cache_read": 100},
    "output_token_details": {"reasoning": 50},
}


def make_chunk(
    chunk_id: str,
    text: str = "",
    *,
    released: date | None = None,
    section: str | None = None,
) -> Chunk:
    doc_id, index = chunk_id.split("#")
    doc_type: DocType = "statement" if doc_id.startswith("statement") else "minutes"
    meeting = date.fromisoformat(doc_id.split("-", 1)[1])
    title = (
        f"FOMC Statement, {meeting:%B %d, %Y}"
        if doc_type == "statement"
        else f"Minutes of the FOMC Meeting, {meeting:%B %d, %Y}"
    )
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        doc_type=doc_type,
        meeting_date=meeting,
        release_date=released or meeting,
        page_updated=None,
        title=title,
        url=f"https://www.federalreserve.gov/{doc_id}.htm",
        section=section,
        index=int(index),
        text=text or f"Text of {chunk_id}.",
        context=title,
    )


CHUNKS = {
    c.chunk_id: c
    for c in [
        make_chunk(
            "statement-2024-09-18#0000",
            "The Committee decided to lower the target range for the federal funds rate by 1/2 "
            "percentage point to 4-3/4 to 5 percent. Voting against this action was Michelle W. "
            "Bowman.",
        ),
        make_chunk(
            "minutes-2024-09-18#0001",
            "Some participants noted that they would have preferred a 25 basis point reduction.",
            released=date(2024, 10, 9),
            section="Committee Policy Actions",
        ),
        make_chunk("minutes-2024-09-18#0002", released=date(2024, 10, 9), section="Staff Outlook"),
        make_chunk("statement-2024-07-31#0000", "The Committee decided to maintain the range."),
        make_chunk("minutes-2024-07-31#0003", released=date(2024, 8, 21), section="Outlook"),
        make_chunk("statement-2015-12-16#0000", "The Committee decided to raise the target range."),
    ]
}

QUESTIONS = [
    {
        "id": "d1",
        "question": "How much did the FOMC cut rates in September 2024, and who dissented?",
        "as_of": "2024-10-01",
        "type": "date_specific",
        "answerable": True,
        "relevant_chunks": [{"chunk_id": "statement-2024-09-18#0000", "role": "primary"}],
        "relevant_doc_ids": ["statement-2024-09-18"],
        "reference_answer": "By 1/2 percentage point to 4-3/4 to 5 percent; Bowman dissented.",
        "required_facts": ["cut by 1/2 percentage point", "Michelle W. Bowman dissented"],
        "notes": "",
        "split": "dev",
    },
    {
        "id": "d2",
        "question": "What did some participants prefer at the September 2024 meeting?",
        "as_of": None,
        "type": "minutes_fact",
        "answerable": True,
        "relevant_chunks": [
            {"chunk_id": "minutes-2024-09-18#0001", "role": "primary"},
            {"chunk_id": "minutes-2024-09-18#0002", "role": "supporting"},
        ],
        "relevant_doc_ids": ["minutes-2024-09-18"],
        "reference_answer": "A 25 basis point reduction.",
        "required_facts": ["some participants preferred a 25 basis point reduction"],
        "notes": None,
        "split": "dev",
    },
    {
        "id": "d3",
        "question": "What did Paul Volcker say at the October 1979 FOMC meeting?",
        "as_of": None,
        "type": "unanswerable",
        "answerable": False,
        "relevant_chunks": [],
        "relevant_doc_ids": [],
        "reference_answer": "The corpus starts in 2015.",
        "required_facts": [],
        "notes": "",
        "split": "dev",
    },
    {
        "id": "t1",
        "question": "When did the FOMC first raise rates after the financial crisis?",
        "as_of": None,
        "type": "statement_fact",
        "answerable": True,
        "relevant_chunks": [{"chunk_id": "statement-2015-12-16#0000", "role": "primary"}],
        "relevant_doc_ids": ["statement-2015-12-16"],
        "reference_answer": "December 2015.",
        "required_facts": ["December 16, 2015"],
        "notes": "",
        "split": "test",
    },
]

# What the fake retriever returns per question (chunk ids, best first, with scores).
RANKINGS: dict[str, list[tuple[str, float]]] = {
    QUESTIONS[0]["question"]: [
        ("minutes-2024-07-31#0003", 0.9),
        ("statement-2024-09-18#0000", 0.8),
        ("minutes-2024-09-18#0001", 0.7),  # released after d1's as_of: the fake filters it
        ("statement-2024-07-31#0000", 0.6),
    ],
    QUESTIONS[1]["question"]: [
        ("minutes-2024-09-18#0001", 0.9),
        ("statement-2024-09-18#0000", 0.5),
        ("minutes-2024-09-18#0002", 0.5),
    ],
    QUESTIONS[2]["question"]: [("statement-2015-12-16#0000", 0.1)],
    QUESTIONS[3]["question"]: [
        ("statement-2024-07-31#0000", 0.4),
        ("statement-2015-12-16#0000", 0.3),
    ],
}


@dataclass
class FakeRetriever:
    """Returns RANKINGS for a question, filtered by as_of like a real retriever, and records how
    it was called."""

    rankings: dict[str, list[tuple[str, float]]] = field(default_factory=lambda: dict(RANKINGS))
    chunks: dict[str, Chunk] = field(default_factory=lambda: dict(CHUNKS))
    calls: list[dict[str, Any]] = field(default_factory=list)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        as_of: date | None = None,
        doc_types: Sequence[DocType] | None = None,
    ) -> list[RetrievedChunk]:
        self.calls.append({"query": query, "top_k": top_k, "as_of": as_of})
        hits = [
            RetrievedChunk(self.chunks[cid], score)
            for cid, score in self.rankings.get(query, [])
            if as_of is None or self.chunks[cid].release_date <= as_of
        ]
        return hits[:top_k]


class FakeAnswerLLM(BaseChatModel):
    """Answers every prompt with `reply`, with usage metadata; counts calls."""

    reply: str = "The Committee lowered the target range by 1/2 percentage point [1]."
    calls: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-answer"

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        self.calls.append(messages)
        message = AIMessage(
            content=self.reply, usage_metadata=USAGE, response_metadata={"model_name": "fake"}
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


def numbered(messages: list[BaseMessage], prefix: str) -> list[int]:
    """The numbers of the lines starting with prefix + number + ":" in the human message."""
    text = str(messages[-1].content)
    return [int(m) for m in re.findall(rf"^{prefix}(\d+):", text, flags=re.MULTILINE)]


def auto_judgment(messages: list[BaseMessage]) -> AnswerJudgment:
    """A well-formed judgment of the prompt: one supported claim per sentence, every citation
    supports, every fact covered."""
    sentences = numbered(messages, "S")
    return AnswerJudgment(
        claims=[
            ClaimVerdict(sentence=n, claim=f"claim {n}", reason="stated", verdict="supported")
            for n in sentences
        ],
        citations=[
            CitationVerdict(id=n, reason="stated", verdict="supports")
            for n in numbered(messages, "C")
        ],
        facts=[
            FactVerdict(id=n, reason="stated", verdict="covered") for n in numbered(messages, "F")
        ],
        response_type="answers",
    )


Reply = Callable[[list[BaseMessage]], AnswerJudgment | None]


class FakeJudgeLLM(BaseChatModel):
    """Answers each structured call with reply(messages), wrapped as
    with_structured_output(include_raw=True) wraps a reply; None stands for an unparseable one."""

    reply: Reply = auto_judgment
    calls: list[list[BaseMessage]] = Field(default_factory=list)
    lock: Any = Field(default_factory=threading.Lock)

    @property
    def _llm_type(self) -> str:
        return "fake-judge"

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        raise NotImplementedError("only with_structured_output is used")

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable[Any, Any]:
        assert schema is AnswerJudgment
        assert kwargs == {"method": "json_schema", "strict": True, "include_raw": True}

        def run(messages: list[BaseMessage]) -> dict[str, Any]:
            with self.lock:
                self.calls.append(messages)
            parsed = self.reply(messages)
            error = None if parsed is not None else ValueError("not valid JSON")
            raw = AIMessage(content="{}", usage_metadata=USAGE)
            return {"raw": raw, "parsed": parsed, "parsing_error": error}

        return RunnableLambda(run)
