"""Live checks of plain-text answers from ChatOpenAI with the key in backend/.env.

Two short calls on three hand-written passages (about 1-2k tokens each, well under a cent); run
with RUN_LIVE_TESTS=1 pytest tests/rag/test_answer_live.py -s to see the token usage.
"""

import re
from collections.abc import Sequence
from datetime import date

import pytest
from langchain_core.callbacks import get_usage_metadata_callback

from app.core.config import Settings
from app.rag.answer import answer_question, get_llm
from app.rag.types import Chunk, RetrievedChunk
from app.schemas.ask import AskRequest, AskResponse
from app.schemas.common import DocType

pytestmark = pytest.mark.live

PASSAGES = [
    (
        "statement",
        None,
        "The Committee decided to lower the target range for the federal funds rate by 1/2 "
        "percentage point to 4-3/4 to 5 percent.",
    ),
    (
        "statement",
        None,
        "Voting against this action was Michelle W. Bowman, who preferred to lower the target "
        "range for the federal funds rate by 1/4 percentage point at this meeting.",
    ),
    (
        "minutes",
        "Committee Policy Actions",
        "Some participants observed that a 25 basis point reduction would be in line with a "
        "gradual path of policy normalization.",
    ),
]


class ListRetriever:
    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        as_of: date | None = None,
        doc_types: Sequence[DocType] | None = None,
    ) -> list[RetrievedChunk]:
        hits = []
        for index, (doc_type, section, text) in enumerate(PASSAGES):
            minutes = doc_type == "minutes"
            chunk = Chunk(
                chunk_id=f"{doc_type}-2024-09-18#{index:04d}",
                doc_id=f"{doc_type}-2024-09-18",
                doc_type=doc_type,
                meeting_date=date(2024, 9, 18),
                release_date=date(2024, 10, 9) if minutes else date(2024, 9, 18),
                page_updated=None,
                title=(
                    "Minutes of the FOMC Meeting, September 17-18, 2024"
                    if minutes
                    else "FOMC Statement, September 18, 2024"
                ),
                url=f"https://www.federalreserve.gov/{doc_type}-2024-09-18.htm",
                section=section,
                index=index,
                text=text,
                context="",
            )
            hits.append(RetrievedChunk(chunk, score=1.0 / (index + 1)))
        return hits[:top_k]


def _ask(question: str) -> AskResponse:
    settings = Settings()
    if not settings.llm_configured:
        pytest.skip("OPENAI_API_KEY is not set in backend/.env")
    request = AskRequest(question=question, as_of=date(2024, 10, 15), top_k=3)
    with get_usage_metadata_callback() as usage:
        response = answer_question(
            request, retriever=ListRetriever(), llm=get_llm(settings), model_name=settings.llm_model
        )
    print(f"\n{response.model_dump_json(indent=2)}\ntoken usage: {usage.usage_metadata}")
    return response


def test_live_answer_cites_the_passages() -> None:
    response = _ask("What did the FOMC decide in September 2024, and was the vote unanimous?")

    markers = {int(n) for n in re.findall(r"\[(\d+)\]", response.answer)}
    assert response.citations
    assert markers == set(range(1, len(response.citations) + 1))
    assert "4-3/4 to 5 percent" in response.answer


def test_live_answer_says_when_the_passages_do_not_answer() -> None:
    response = _ask("What did the Committee say about the Bank of Japan's yield curve control?")

    assert response.citations == []
    assert "[" not in response.answer
