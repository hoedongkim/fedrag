"""POST /api/ask, POST /api/ask/stream and GET /api/ask/examples."""

import logging
from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel

from app.api.ask_examples import EXAMPLES_RESPONSES, ExamplesUnavailableError, load_ask_examples
from app.api.demo import ASK_DISABLED_RESPONSES, AskBody
from app.api.rag import (
    ASK_RESPONSES,
    STREAM_UNAVAILABLE_RESPONSES,
    ChatModel,
    Retrievers,
    answer_error,
    is_answer_error,
    stream_error_responses,
)
from app.core.config import Settings, get_settings
from app.rag.answer import LLM_NOT_CONFIGURED, AnswerEvent, answer_question, stream_answer
from app.rag.index import IndexNotBuiltError
from app.rag.retrievers import RerankerUnavailableError, preload
from app.schemas.ask import (
    AskExample,
    AskResponse,
    AskStreamEvent,
    StreamCitation,
    StreamDelta,
    StreamError,
    StreamStatus,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ask"])

STREAM_FAILED = "Internal error while streaming the answer."

# The SSE event name of each payload type.
EVENT_NAMES: dict[type[BaseModel], str] = {
    StreamStatus: "status",
    StreamDelta: "delta",
    StreamCitation: "citation",
    AskResponse: "done",
    StreamError: "error",
}


def _event_schema(name: str, payload: type[BaseModel]) -> dict[str, Any]:
    """One event of the stream in OpenAPI 3.2 itemSchema form: its name and its JSON data."""
    return {
        "properties": {
            "event": {"const": name},
            "data": {
                "contentMediaType": "application/json",
                "contentSchema": {"$ref": f"#/components/schemas/{payload.__name__}"},
            },
        }
    }


STREAM_RESPONSE: dict[str, Any] = {
    # The union of the event payloads: puts every payload model in the components.
    "model": AskStreamEvent,
    "description": (
        "A text/event-stream; each event is `event: <name>` and `data: <JSON>`. In order: "
        "`status` (StreamStatus) when retrieval starts and again when generation starts; then "
        "`delta` (StreamDelta), the next piece of the answer with final citation numbers, each "
        "passage's first citation preceded by `citation` (StreamCitation); last `done` "
        "(AskResponse), the final answer that replaces the streamed text. A failure after the "
        "stream started ends it with `error` (StreamError) instead of `done`. Comment lines "
        "(`: ping`) keep an idle connection open. The schema is the union of the data payloads."
    ),
    "content": {
        "text/event-stream": {
            "itemSchema": {
                "required": ["event", "data"],
                "oneOf": [_event_schema(name, model) for model, name in EVENT_NAMES.items()],
            }
        }
    },
}


@router.post(
    "/ask",
    operation_id="ask",
    summary="Ask a question about FOMC communications",
    responses={**ASK_RESPONSES, **ASK_DISABLED_RESPONSES},
)
def ask(
    request: AskBody,
    settings: Annotated[Settings, Depends(get_settings)],
    llm: ChatModel,
    retrievers: Retrievers,
) -> AskResponse:
    """Answer a question from retrieved FOMC passages, citing the ones the answer relies on.

    With `as_of` only documents released by then are searched, and the model is told to answer as
    of that date. Each `[n]` marker in the answer refers to `citations[n - 1]`. Disabled (403) on
    the public demo, after request validation.
    """
    # A plain def: FastAPI runs it in its threadpool, so the blocking retrieval and LLM call never
    # hold up the event loop. AskBody comes first: on the public demo its 403 is raised before
    # the chat model and retrievers are built.
    if llm is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, LLM_NOT_CONFIGURED)
    try:
        retriever = retrievers(request.retrieval)
        return answer_question(request, retriever=retriever, llm=llm, model_name=settings.llm_model)
    except Exception as exc:
        if not is_answer_error(exc):
            raise
        raise HTTPException(*answer_error(exc)) from None


async def answer_stream(
    request: AskBody,
    settings: Annotated[Settings, Depends(get_settings)],
    llm: ChatModel,
    retrievers: Retrievers,
) -> AsyncIterator[AnswerEvent]:
    """The events of a streamed answer, once every check POST /api/ask makes before answering
    has passed.

    A dependency, so the checks run before the response starts: once the endpoint's generator
    runs, the 200 and its headers are sent, and a failure can only be an error event. The body is
    declared here (through AskBody) and only here, so validation still comes first: an invalid
    body skips this function and is a 422, without a duplicate error; on the public demo a valid
    one is a 403.
    """
    if llm is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, LLM_NOT_CONFIGURED)
    try:
        # Blocking: the index check, and loading the cross-encoder unless warm-up already did.
        retriever = await run_in_threadpool(retrievers, request.retrieval)
        await run_in_threadpool(preload, retriever)
    except (IndexNotBuiltError, RerankerUnavailableError) as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from None
    return stream_answer(request, retriever=retriever, llm=llm, model_name=settings.llm_model)


@router.post(
    "/ask/stream",
    operation_id="askStream",
    summary="Ask a question and stream the answer as it is written",
    response_class=EventSourceResponse,
    responses={
        status.HTTP_200_OK: STREAM_RESPONSE,
        **STREAM_UNAVAILABLE_RESPONSES,
        **stream_error_responses(ASK_DISABLED_RESPONSES),
    },
)
async def ask_stream(
    events: Annotated[AsyncIterator[AnswerEvent], Depends(answer_stream)],
) -> AsyncIterator[ServerSentEvent]:
    """Answer like POST /api/ask, streaming the answer as server-sent events.

    A missing key, index or reranker is an HTTP error before the stream starts, as for /api/ask,
    and so is the public demo's 403. The `done` event carries the same AskResponse that /api/ask
    returns. If the client disconnects, generation stops.
    """
    try:
        async with aclosing(events):
            async for event in events:
                yield ServerSentEvent(event=EVENT_NAMES[type(event)], data=event)
    except Exception as exc:  # the 200 is already sent: only an event can report it
        if is_answer_error(exc):
            yield _error_event(*answer_error(exc))
        else:
            logger.exception("Streaming an answer failed")
            yield _error_event(status.HTTP_500_INTERNAL_SERVER_ERROR, STREAM_FAILED)


@router.get(
    "/ask/examples",
    operation_id="listAskExamples",
    summary="Recorded example answers",
    responses=EXAMPLES_RESPONSES,
)
def list_ask_examples(settings: Annotated[Settings, Depends(get_settings)]) -> list[AskExample]:
    """Real answers of the production configuration (hybrid + rerank under the tuned retriever
    preset), recorded during the RAG evaluation, each with its citations, the question's
    evaluation id and the evaluation judge's verdicts.

    Read from data/eval/ask_examples.json (make ask-examples); serving them calls no LLM, so they
    are available in every mode, including the public demo, where they stand in for live
    answers.
    """
    try:
        return load_ask_examples(settings)
    except ExamplesUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from None


def _error_event(status_code: int, detail: str) -> ServerSentEvent:
    return ServerSentEvent(event="error", data=StreamError(status=status_code, detail=detail))
