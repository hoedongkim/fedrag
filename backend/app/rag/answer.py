"""Grounded answers: retrieve passages, have the LLM cite them as [n], map markers to citations.

answer_question returns a whole answer and stream_answer yields one as the LLM writes it. Both run
the same two LCEL steps, retrieval then generation, and build the final AskResponse the same way.
langchain_openai (and with it the openai client) is imported by get_llm, when a chat model is
first needed, not with this module.
"""

import logging
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import aclosing
from datetime import date
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from fastapi.concurrency import run_in_threadpool
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda, RunnablePassthrough
from pydantic import SecretStr

from app.core.config import Settings
from app.rag.chunking import sentence_starts
from app.rag.filters import is_eligible
from app.rag.markers import Renumbered, StreamRenumberer, renumber_markers
from app.rag.prompts import ANSWER_PROMPT, prompt_inputs
from app.rag.types import RetrievedChunk, Retriever
from app.schemas.ask import (
    AskRequest,
    AskResponse,
    Citation,
    StreamCitation,
    StreamDelta,
    StreamStatus,
)
from app.schemas.common import SourceDocument

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

LLM_NOT_CONFIGURED = "LLM not configured: set OPENAI_API_KEY in backend/.env."

# GPT-5.x models reason before answering and accept reasoning_effort, not temperature (LangChain
# drops temperature for them, so sampling can't be pinned). "low" keeps a short reasoning pass,
# enough to tie each claim to a passage and to notice when the passages don't answer, at a
# fraction of the latency and reasoning-token cost of "medium" for this short, extractive task.
REASONING_EFFORT = "low"
MAX_OUTPUT_TOKENS = 4096  # reasoning plus answer: a cost ceiling far above a normal answer
TIMEOUT_S = 60.0
SNIPPET_CHARS = 600
# FOMC statements are at most ~8 weeks apart: when the newest passage an as_of answer cites is
# older than this, a later meeting is missing from it, and a note says so (rule 8, in code).
STALE_AFTER_DAYS = 90

AnswerEvent = StreamStatus | StreamCitation | StreamDelta | AskResponse


class LLMNotConfiguredError(RuntimeError):
    """Raised when an answer needs the LLM but OPENAI_API_KEY is not set."""

    def __init__(self) -> None:
        super().__init__(LLM_NOT_CONFIGURED)


def get_llm(settings: Settings) -> "ChatOpenAI":
    """The chat model for answers, cached per model and key; LLMNotConfiguredError without a key."""
    if settings.openai_api_key is None:
        raise LLMNotConfiguredError
    return _chat_model(settings.llm_model, settings.openai_api_key)


@lru_cache(maxsize=4)
def _chat_model(model: str, api_key: SecretStr) -> "ChatOpenAI":
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        reasoning_effort=REASONING_EFFORT,
        max_tokens=MAX_OUTPUT_TOKENS,
        timeout=TIMEOUT_S,
        max_retries=2,
    )


def retrieval_step(retriever: Retriever) -> Runnable[dict[str, Any], list[RetrievedChunk]]:
    """{"request"} -> the passages to answer from, filtered again by the request's as_of and
    doc_types."""
    return RunnableLambda(lambda inputs: _retrieve(retriever, inputs["request"]), name="retrieve")


def generation_step(llm: BaseChatModel) -> Runnable[dict[str, Any], str]:
    """{"request", "passages"} -> the answer as plain text with [n] markers, streamable token by
    token. The LLM is skipped when no passage passed the filters: there is nothing to ground on.
    """
    generate = (
        RunnableLambda(_prompt_inputs) | ANSWER_PROMPT | llm | StrOutputParser()
    ).with_config(run_name="generate")
    no_passages = RunnableLambda(_no_passages, name="no_passages")
    return RunnableBranch((_has_no_passages, no_passages), generate)


def build_answer_chain(retriever: Retriever, llm: BaseChatModel) -> Runnable[Any, dict[str, Any]]:
    """LCEL chain: {"request"} -> retrieve -> numbered passages -> prompt -> plain-text answer.

    The output dict holds the request, its "passages" and the "generated" answer text, whose
    markers still number the passages in prompt order.
    """
    return RunnablePassthrough.assign(passages=retrieval_step(retriever)) | (
        RunnablePassthrough.assign(generated=generation_step(llm))
    )


def answer_question(
    request: AskRequest, *, retriever: Retriever, llm: BaseChatModel, model_name: str
) -> AskResponse:
    """Answer from the passages `retriever` finds, with citations. Blocking: run it in a thread."""
    started = time.perf_counter()
    result = build_answer_chain(retriever, llm).invoke({"request": request})
    return build_response(
        request, result["passages"], result["generated"], model_name=model_name, started=started
    )


async def stream_answer(
    request: AskRequest, *, retriever: Retriever, llm: BaseChatModel, model_name: str
) -> AsyncIterator[AnswerEvent]:
    """Answer like answer_question, yielding the answer as the LLM writes it.

    Yields StreamStatus("retrieving"), StreamStatus("generating"), then StreamDelta pieces of the
    answer with their markers already renumbered (StreamRenumberer), each passage's
    StreamCitation just before the first delta that cites it, and last the AskResponse that
    answer_question would return: built by resolve_citations from the whole generated text, so its
    answer equals the joined deltas. Errors propagate. Closing the iteration early (a client
    disconnect) closes the LLM's stream, and with it the OpenAI request: no further tokens.
    """
    started = time.perf_counter()
    inputs: dict[str, Any] = {"request": request}
    yield StreamStatus(stage="retrieving")
    # Retrieval blocks (BM25, Chroma, the cross-encoder, the query embedding): a worker thread.
    passages = await run_in_threadpool(retrieval_step(retriever).invoke, inputs)
    yield StreamStatus(stage="generating")
    renumberer = StreamRenumberer(len(passages))
    generated: list[str] = []
    # aclosing: if this generator is closed at a yield, the LLM stream is closed with it, at once.
    async with aclosing(generation_step(llm).astream({**inputs, "passages": passages})) as chunks:
        async for chunk in chunks:
            generated.append(chunk)
            for event in _stream_events(renumberer.feed(chunk), passages):
                yield event
    for event in _stream_events(renumberer.flush(), passages):
        yield event
    yield build_response(
        request, passages, "".join(generated), model_name=model_name, started=started
    )


def build_response(
    request: AskRequest,
    passages: Sequence[RetrievedChunk],
    generated: str,
    *,
    model_name: str,
    started: float,
) -> AskResponse:
    """The AskResponse for `generated`, the LLM's text over `passages`; `started` is the
    time.perf_counter() reading taken when answering began."""
    answer, citations = resolve_citations(generated, passages)
    return AskResponse(
        answer=answer,
        citations=citations,
        notes=answer_notes(request, citations),
        retrieval=request.retrieval,
        as_of=request.as_of,
        model=model_name,
        latency_ms=round((time.perf_counter() - started) * 1000, 1),
    )


def resolve_citations(
    answer: str, passages: Sequence[RetrievedChunk]
) -> tuple[str, list[Citation]]:
    """Map the answer's [n] markers onto `passages` and renumber them to the citation order.

    The final numbering of every answer, whole or streamed. Citations are exactly the passages
    that valid inline markers name, in order of first appearance, so [k] in the returned answer is
    citations[k - 1]. Markers naming no passage are dropped and adjacent ones are deduplicated and
    sorted; an answer without markers has no citations.
    """
    text, cited = renumber_markers(answer, len(passages))
    return text, [_citation(passages[ref - 1]) for ref in cited]


def answer_notes(request: AskRequest, citations: Sequence[Citation]) -> list[str]:
    """What the answer's reader should know that the model may not say: when the newest passage
    an as_of answer cites is more than STALE_AFTER_DAYS older than as_of, later decisions may be
    missing from it (prompt rule 8, enforced in code)."""
    if request.as_of is None or not citations:
        return []
    newest = max(c.document.release_date for c in citations)
    gap = (request.as_of - newest).days
    if gap <= STALE_AFTER_DAYS:
        return []
    return [
        f"The most recent passage cited was released {_display_date(newest)}, {gap:,} days before "
        f"{_display_date(request.as_of)}; later decisions may not be reflected."
    ]


def trim_snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    """`text` if it fits in `limit` characters, else its leading whole sentences that do (when
    they fill at least a third of the limit), else its leading words with an ellipsis."""
    text = text.strip()
    if len(text) <= limit:
        return text
    ends = (len(text[:start].rstrip()) for start in sentence_starts(text)[1:])
    cut = max((end for end in ends if end <= limit), default=0)
    if cut >= limit // 3:
        return text[:cut]
    return text[: limit - 1].rsplit(maxsplit=1)[0].rstrip(",;:") + "…"


def _retrieve(retriever: Retriever, request: AskRequest) -> list[RetrievedChunk]:
    hits = retriever.retrieve(
        request.question, top_k=request.top_k, as_of=request.as_of, doc_types=request.doc_types
    )
    # Retrievers filter before ranking; this re-check only stops a retriever bug from putting a
    # later (or excluded) document in front of the LLM.
    eligible = [hit for hit in hits if is_eligible(hit.chunk, request.as_of, request.doc_types)]
    if len(eligible) < len(hits):
        logger.error(
            "Dropped %d retrieved passages outside as_of/doc_types", len(hits) - len(eligible)
        )
    return eligible[: request.top_k]


def _prompt_inputs(inputs: dict[str, Any]) -> dict[str, str]:
    request: AskRequest = inputs["request"]
    return prompt_inputs(request.question, request.as_of, inputs["passages"])


def _has_no_passages(inputs: dict[str, Any]) -> bool:
    return not inputs["passages"]


def _no_passages(inputs: dict[str, Any]) -> str:
    """The answer when no passage passed the filters, in place of the LLM's."""
    request: AskRequest = inputs["request"]
    filters = [
        f"released on or before {request.as_of.isoformat()}" if request.as_of else "",
        f"of type {', '.join(request.doc_types)}" if request.doc_types else "",
    ]
    scope = "; ".join(filter(None, filters))
    if scope:
        return (
            f"No indexed FOMC passages match the filters ({scope}), so there is nothing to "
            "answer from."
        )
    return "No indexed FOMC passages were found, so there is nothing to answer from."


def _stream_events(
    piece: Renumbered, passages: Sequence[RetrievedChunk]
) -> list[StreamCitation | StreamDelta]:
    """The events for a renumbered piece of a streamed answer: its new citations, then its text."""
    events: list[StreamCitation | StreamDelta] = [
        StreamCitation(n=n, citation=_citation(passages[ref - 1])) for n, ref in piece.cited.items()
    ]
    if piece.text:
        events.append(StreamDelta(text=piece.text))
    return events


def _citation(hit: RetrievedChunk) -> Citation:
    chunk = hit.chunk
    document = SourceDocument(
        doc_id=chunk.doc_id,
        doc_type=chunk.doc_type,
        meeting_date=chunk.meeting_date,
        release_date=chunk.release_date,
        page_updated=chunk.page_updated,
        title=chunk.title,
        url=chunk.url,
    )
    return Citation(
        document=document,
        chunk_id=chunk.chunk_id,
        section=chunk.section,
        snippet=trim_snippet(chunk.text),
        score=hit.score,
    )


def _display_date(day: date) -> str:
    """`day` as the page shows dates, e.g. "Sep 18, 2024" (no %-d, which is not portable)."""
    return f"{day:%b} {day.day}, {day.year}"
