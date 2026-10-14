"""RAG for routes: the chat-model and retriever dependencies, their documented errors, and the
retrieval warm-up at startup.

Nothing here imports the openai package, Chroma or torch: they load with the first chat model,
retriever or error that needs them, so the public demo (PUBLIC_DEMO=true), which never answers,
never loads them."""

import logging
import threading
import time
from collections.abc import Callable
from typing import Annotated, Any

from fastapi import Depends, status
from langchain_core.language_models import BaseChatModel

from app.core.config import Settings, get_settings
from app.rag.answer import get_llm
from app.rag.config import production_config
from app.rag.index import IndexNotBuiltError, index_ready, require_index
from app.rag.retrievers import RerankerUnavailableError, get_retriever, warm_up
from app.rag.types import Retriever
from app.schemas.common import ErrorResponse, RetrievalMethod

logger = logging.getLogger(__name__)

OPENAI_FAILED = "OpenAI request failed ({error}).{advice}"
# OpenAI's 429 for an account with no credit left (the error's code, or its type): unlike a rate
# limit, retrying does not help until someone adds credit.
NO_CREDIT_CODES = frozenset({"insufficient_quota", "credit_balance_exhausted"})
NO_CREDIT = (
    "The OpenAI account behind this server has no credit left; add credit at "
    "platform.openai.com/billing."
)

RetrieverFactory = Callable[[RetrievalMethod], Retriever]

UNAVAILABLE_RESPONSES: dict[int | str, dict[str, Any]] = {
    status.HTTP_503_SERVICE_UNAVAILABLE: {
        "model": ErrorResponse,
        "description": (
            "LLM not configured (OPENAI_API_KEY is not set), the search index not built yet "
            "(run make index), or the reranker model unavailable."
        ),
    },
}


def stream_error_responses(
    responses: dict[int | str, dict[str, Any]],
) -> dict[int | str, dict[str, Any]]:
    """The same error responses for a text/event-stream route: FastAPI documents a response
    "model" under the route's media type, but the error body is JSON (ErrorResponse)."""
    return {
        code: {
            "description": response["description"],
            "content": {
                "application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}
            },
        }
        for code, response in responses.items()
    }


STREAM_UNAVAILABLE_RESPONSES = stream_error_responses(UNAVAILABLE_RESPONSES)
ASK_RESPONSES: dict[int | str, dict[str, Any]] = {
    status.HTTP_502_BAD_GATEWAY: {
        "model": ErrorResponse,
        "description": (
            "The OpenAI request failed: network error, rate limit, no credit left on the "
            "OpenAI account, or bad key."
        ),
    },
    **UNAVAILABLE_RESPONSES,
}


def is_answer_error(exc: BaseException) -> bool:
    """Whether answering a question raised `exc` for a reason answer_error maps to a status, not
    a 500: the index missing or out of sync, the reranker unavailable, or an OpenAI failure."""
    if isinstance(exc, (IndexNotBuiltError, RerankerUnavailableError)):
        return True
    import openai  # already loaded by whatever raised an OpenAI error

    return isinstance(exc, openai.OpenAIError)


def answer_error(exc: Exception) -> tuple[int, str]:
    """The status and detail for an error is_answer_error accepts.

    502 when an OpenAI request failed (the answer, or the query embedding of dense retrieval),
    naming only the error's class: OpenAI messages can quote part of the API key. 503 when the
    index is missing or out of sync with Chroma, or the reranker cannot be downloaded.
    """
    import openai

    if isinstance(exc, openai.OpenAIError):
        reason = ": no credit left" if out_of_credit(exc) else ""
        logger.warning("OpenAI request failed (%s%s)", type(exc).__name__, reason)
        return status.HTTP_502_BAD_GATEWAY, openai_failed(exc)
    return status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)


def out_of_credit(exc: Exception) -> bool:
    """Whether `exc` is OpenAI's 429 for an account with no credit left (code or type in
    NO_CREDIT_CODES), not a transient rate limit."""
    import openai

    return isinstance(exc, openai.RateLimitError) and bool({exc.code, exc.type} & NO_CREDIT_CODES)


def openai_failed(exc: Exception) -> str:
    """The 502 detail: the error's class only (OpenAI messages can quote part of the API key),
    and what to do about it."""
    import openai

    if out_of_credit(exc):
        advice = f" {NO_CREDIT}"  # fixed text: OpenAI's own message is never echoed
    elif isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
        advice = " Check OPENAI_API_KEY in backend/.env."
    elif isinstance(
        exc, (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError)
    ):
        advice = " Try again shortly."  # APITimeoutError subclasses APIConnectionError
    else:
        advice = ""
    return OPENAI_FAILED.format(error=type(exc).__name__, advice=advice)


def get_chat_model(settings: Annotated[Settings, Depends(get_settings)]) -> BaseChatModel | None:
    """The chat model, or None while OPENAI_API_KEY is not set and always on the public demo;
    override this dependency in tests.

    It never raises, so request validation comes first: a bad body is a 422 even without a key.
    FastAPI still resolves it for a request whose body is invalid, hence the demo check here too.
    """
    return get_llm(settings) if settings.llm_configured and not settings.public_demo else None


def get_retriever_factory(
    settings: Annotated[Settings, Depends(get_settings)],
) -> RetrieverFactory:
    """A factory of retrievers by method, under the production retriever config (the preset
    RETRIEVER_PRESET names: app.rag.config.production_config); override this dependency in tests.

    The factory, not the dependency, checks the index and raises IndexNotBuiltError, so the check
    runs only for a valid request that has an LLM to answer it.
    """

    def factory(method: RetrievalMethod) -> Retriever:
        require_index(settings)
        return get_retriever(method, settings, production_config(settings))

    return factory


def start_warm_up(settings: Settings) -> threading.Thread:
    """Warm retrieval up (app.rag.retrievers.warm_up) in a daemon thread, so startup never waits
    for it. It is skipped while the index is not built, and a failure is only logged: the first
    request that needs the missing part reports it anyway."""
    thread = threading.Thread(target=_warm_up, args=(settings,), name="warm-up", daemon=True)
    thread.start()
    return thread


def _warm_up(settings: Settings) -> None:
    started = time.perf_counter()
    try:
        if not index_ready(settings):
            logger.info("Retrieval warm-up skipped: the search index is not built.")
            return
        warm_up(settings)
    except Exception as exc:  # a background thread: there is no request to report it to
        logger.warning("Retrieval warm-up failed (%s): %s", type(exc).__name__, exc)
        return
    logger.info("Retrieval warmed up in %.1f s.", time.perf_counter() - started)


ChatModel = Annotated[BaseChatModel | None, Depends(get_chat_model)]
Retrievers = Annotated[RetrieverFactory, Depends(get_retriever_factory)]
