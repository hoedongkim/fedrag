"""Tests for POST /api/ask/stream with the chat model faked out: no OpenAI calls, no real index."""

import asyncio
import json
import re
from collections.abc import AsyncIterator, Callable
from datetime import date
from typing import Any

import anyio
import httpx
import openai
import pytest
from fastapi.testclient import TestClient
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field

from app.api.rag import get_chat_model, get_retriever_factory
from app.core.config import Settings
from app.rag.answer import LLM_NOT_CONFIGURED
from app.rag.index import NOT_BUILT, IndexNotBuiltError
from app.rag.retrievers import CrossEncoderScorer, RerankRetriever
from app.rag.types import Chunk, RetrievedChunk
from app.schemas.common import DocType

MakeClient = Callable[..., TestClient]

QUESTION = {"question": "What did the Committee decide in September 2024?"}
CUT = (
    "The Committee decided to lower the target range for the federal funds rate by 1/2 "
    "percentage point to 4-3/4 to 5 percent."
)
VIEW = "Some participants would have preferred a 25 basis point reduction."


def chunk(doc_type: DocType, released: date, text: str) -> Chunk:
    doc_id = f"{doc_type}-2024-09-18"
    return Chunk(
        chunk_id=f"{doc_id}#0001",
        doc_id=doc_id,
        doc_type=doc_type,
        meeting_date=date(2024, 9, 18),
        release_date=released,
        page_updated=None,
        title=f"FOMC {doc_type}, September 2024",
        url=f"https://www.federalreserve.gov/{doc_id}.htm",
        section=None,
        index=1,
        text=text,
        context="",
    )


STATEMENT = RetrievedChunk(chunk("statement", date(2024, 9, 18), CUT), score=0.9)
MINUTES = RetrievedChunk(chunk("minutes", date(2024, 10, 9), VIEW), score=0.4)
# A marker split across chunks, a run across chunks, and a marker naming no passage.
CHUNKS = [
    "The Committee lowered the range to 4-3/4 to 5 percent [",
    "2",
    "]",
    " [1",
    "]. Some participants preferred a smaller cut [",
    "1",
    "]",
    " [7].",
]


class FakeRetriever:
    def __init__(
        self, hits: list[RetrievedChunk] | None = None, error: Exception | None = None
    ) -> None:
        self.hits = [MINUTES, STATEMENT] if hits is None else hits
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def retrieve(self, query: str, **filters: Any) -> list[RetrievedChunk]:
        self.calls.append({"query": query, **filters})
        if self.error:
            raise self.error
        return self.hits


class StreamingChatModel(BaseChatModel):
    """Stands in for ChatOpenAI: replies with `chunks`, streamed one by one `delay` seconds
    apart (whole for invoke); with `error`, raises it after `fail_after` chunks. Records each
    prompt, how many chunks it streamed, and whether its stream was closed."""

    chunks: list[str] = Field(default_factory=lambda: list(CHUNKS))
    error: Exception | None = None
    fail_after: int = 0
    delay: float = 0.0
    prompts: list[list[BaseMessage]] = Field(default_factory=list)
    streamed: int = 0
    closed: bool = False

    @property
    def _llm_type(self) -> str:
        return "fake-streaming"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.prompts.append(messages)
        if self.error:
            raise self.error
        message = AIMessage(content="".join(self.chunks))
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        self.prompts.append(messages)
        try:
            for number, text in enumerate(self.chunks):
                if self.error and number == self.fail_after:
                    raise self.error
                await asyncio.sleep(self.delay)
                self.streamed += 1
                yield ChatGenerationChunk(message=AIMessageChunk(content=text))
        finally:
            self.closed = True


def install(
    client: TestClient,
    *,
    llm: StreamingChatModel | None = None,
    retriever: Any = None,
) -> tuple[StreamingChatModel, Any]:
    llm = llm or StreamingChatModel()
    retriever = retriever or FakeRetriever()
    client.app.dependency_overrides[get_chat_model] = lambda: llm
    client.app.dependency_overrides[get_retriever_factory] = lambda: lambda method: retriever
    return llm, retriever


def parse_sse(body: str) -> list[tuple[str, Any]]:
    """(event name, parsed data) for each event of a text/event-stream body; comments skipped."""
    events = []
    for block in body.split("\n\n"):
        fields = dict(
            line.split(": ", 1) for line in block.splitlines() if not line.startswith(":")
        )
        if fields:
            events.append((fields["event"], json.loads(fields["data"])))
    return events


def stream(client: TestClient, body: dict[str, Any]) -> tuple[httpx.Response, list[Any]]:
    response = client.post("/api/ask/stream", json=body)
    return response, parse_sse(response.text) if response.status_code == 200 else []


def test_events_arrive_in_order(client: TestClient) -> None:
    install(client)

    response, events = stream(client, QUESTION)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    # no-transform: the Next.js proxy would otherwise gzip the stream and hold it back to the end.
    assert response.headers.get_list("cache-control") == ["no-cache, no-transform"]
    assert response.headers["x-accel-buffering"] == "no"
    names = [name for name, _ in events]
    assert names[:2] == ["status", "status"]
    assert [data for _, data in events[:2]] == [{"stage": "retrieving"}, {"stage": "generating"}]
    assert set(names[2:-1]) == {"delta", "citation"}
    assert names[-1] == "done"


def test_deltas_carry_final_numbers_and_join_to_the_answer(client: TestClient) -> None:
    install(client)

    _, events = stream(client, QUESTION)

    done = events[-1][1]
    deltas = [data["text"] for name, data in events if name == "delta"]
    assert (
        "".join(deltas)
        == done["answer"]
        == (
            "The Committee lowered the range to 4-3/4 to 5 percent [1][2]. "
            "Some participants preferred a smaller cut [2]."
        )
    )
    citations = [data for name, data in events if name == "citation"]
    assert [c["n"] for c in citations] == [1, 2]
    assert [c["citation"] for c in citations] == done["citations"]
    assert [c["chunk_id"] for c in done["citations"]] == [
        "statement-2024-09-18#0001",
        "minutes-2024-09-18#0001",
    ]
    shown: set[int] = set()
    for name, data in events:  # each citation comes before the first delta that uses it
        if name == "citation":
            shown.add(data["n"])
        elif name == "delta":
            assert {int(n) for n in re.findall(r"\[(\d+)\]", data["text"])} <= shown


def test_done_is_what_post_ask_returns(client: TestClient) -> None:
    install(client)
    body = {**QUESTION, "as_of": "2024-12-31", "retrieval": "bm25", "top_k": 4}

    whole = client.post("/api/ask", json=body).json()
    _, events = stream(client, body)

    done = events[-1][1]
    assert isinstance(done.pop("latency_ms"), float)
    whole.pop("latency_ms")
    assert done == whole


def test_stale_sources_get_a_note_outside_the_answer(client: TestClient) -> None:
    install(client, retriever=FakeRetriever([STATEMENT]))

    _, events = stream(client, {**QUESTION, "as_of": "2025-06-01"})

    done = events[-1][1]
    assert done["notes"] == [
        "The most recent passage cited was released Sep 18, 2024, 256 days before Jun 1, 2025; "
        "later decisions may not be reflected."
    ]
    assert "later decisions" not in "".join(d["text"] for n, d in events if n == "delta")


def test_passages_released_after_as_of_never_reach_the_llm(client: TestClient) -> None:
    # A retriever that ignored as_of: the minutes came out after it.
    llm, retriever = install(client)

    _, events = stream(client, {**QUESTION, "as_of": "2024-09-30"})

    assert retriever.calls[0]["as_of"] == date(2024, 9, 30)
    prompt = "\n".join(str(message.content) for message in llm.prompts[0])
    assert CUT in prompt and VIEW not in prompt
    done = events[-1][1]
    assert [c["chunk_id"] for c in done["citations"]] == ["statement-2024-09-18#0001"]


@pytest.mark.parametrize(
    "body",
    [{"question": "Hi"}, {"question": "Rates?", "top_k": 0}, {"question": "Rates?", "x": 1}],
    ids=["short-question", "top-k-zero", "unknown-field"],
)
def test_invalid_request_is_a_plain_422(client: TestClient, body: dict[str, Any]) -> None:
    llm, retriever = install(client)

    response = client.post("/api/ask/stream", json=body)

    assert response.status_code == 422
    assert response.headers["content-type"] == "application/json"
    assert "cache-control" not in response.headers  # only event streams get no-transform
    assert len(response.json()["detail"]) == 1  # one error, not one per declaration of the body
    assert retriever.calls == [] and llm.prompts == []


def test_503_without_api_key(
    client: TestClient, index_ready: Callable[[bool], list[Settings]]
) -> None:
    seen = index_ready(True)

    response = client.post("/api/ask/stream", json=QUESTION)

    assert response.status_code == 503
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {"detail": LLM_NOT_CONFIGURED}
    assert seen == []


def test_503_without_index(
    make_client: MakeClient, index_ready: Callable[[bool], list[Settings]]
) -> None:
    client = make_client(openai_api_key="sk-test")
    index_ready(False)

    response = client.post("/api/ask/stream", json=QUESTION)

    assert response.status_code == 503
    assert response.json() == {"detail": NOT_BUILT}
    assert client.post("/api/ask", json=QUESTION).json() == response.json()


def test_503_when_the_reranker_cannot_be_loaded(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def not_cached(*args: Any, **kwargs: Any) -> None:
        raise OSError("not in the cache and no network")

    monkeypatch.setattr("sentence_transformers.CrossEncoder", not_cached)
    llm, _ = install(
        client, retriever=RerankRetriever(FakeRetriever(), CrossEncoderScorer("no/such-model"))
    )

    response = client.post("/api/ask/stream", json=QUESTION)

    assert response.status_code == 503  # decided before the stream: no 200, no events
    assert response.headers["content-type"] == "application/json"
    assert "no/such-model" in response.json()["detail"]
    assert llm.prompts == []


def _response(status_code: int) -> httpx.Response:
    return httpx.Response(status_code, request=httpx.Request("POST", "https://api.openai.com"))


NO_CREDIT_DETAIL = (
    "OpenAI request failed (RateLimitError). The OpenAI account behind this server has no credit "
    "left; add credit at platform.openai.com/billing."
)


def _no_credit(code: str) -> openai.RateLimitError:
    """A fake OpenAI 429 for an account with no credit; its message (never echoed) quotes part
    of the key."""
    message = f"You exceeded your current quota (sk-abc...wxyz): {code}."
    body = {"message": message, "type": code, "param": None, "code": code}
    return openai.RateLimitError(message, response=_response(429), body=body)


@pytest.mark.parametrize(
    ("llm_error", "retriever_error", "status", "detail"),
    [
        (
            openai.AuthenticationError(
                "Incorrect API key provided: sk-abc...wxyz", response=_response(401), body=None
            ),
            None,
            502,
            "OpenAI request failed (AuthenticationError). Check OPENAI_API_KEY in backend/.env.",
        ),
        (
            openai.APIConnectionError(
                message="Connection to sk-abc...wxyz failed",
                request=httpx.Request("POST", "https://api.openai.com"),
            ),
            None,
            502,
            "OpenAI request failed (APIConnectionError). Try again shortly.",
        ),
        (
            None,
            openai.RateLimitError(
                "Rate limit for sk-abc...wxyz", response=_response(429), body=None
            ),
            502,
            "OpenAI request failed (RateLimitError). Try again shortly.",
        ),
        (_no_credit("insufficient_quota"), None, 502, NO_CREDIT_DETAIL),
        (None, _no_credit("credit_balance_exhausted"), 502, NO_CREDIT_DETAIL),
        (None, IndexNotBuiltError("Chroma is out of sync."), 503, "Chroma is out of sync."),
        (
            ValueError("unexpected: sk-abc...wxyz"),
            None,
            500,
            "Internal error while streaming the answer.",
        ),
    ],
    ids=[
        "bad-key",
        "llm-unreachable",
        "embedding-rate-limited",
        "llm-no-credit",
        "embedding-no-credit",
        "index-broke",
        "bug",
    ],
)
def test_failure_after_the_stream_started_is_an_error_event(
    client: TestClient,
    llm_error: Exception | None,
    retriever_error: Exception | None,
    status: int,
    detail: str,
) -> None:
    llm = StreamingChatModel(error=llm_error, fail_after=3)
    install(client, llm=llm, retriever=FakeRetriever(error=retriever_error))

    response, events = stream(client, QUESTION)

    assert response.status_code == 200
    assert events[0] == ("status", {"stage": "retrieving"})
    assert events[-1] == ("error", {"status": status, "detail": detail})
    assert "done" not in [name for name, _ in events]
    assert "sk-abc" not in response.text
    if llm_error:  # what was generated before the failure was streamed
        assert "".join(d["text"] for n, d in events if n == "delta").startswith("The Committee")


def test_client_disconnect_stops_generation(client: TestClient) -> None:
    """Drive the ASGI app as uvicorn does (ASGI spec 2.3: disconnects arrive through receive)
    and hang up after the first delta: the model's stream is closed long before its end."""
    llm, _ = install(client, llm=StreamingChatModel(chunks=["word "] * 200, delay=0.01))
    sent: list[bytes] = []

    async def run() -> None:
        hung_up = anyio.Event()
        requests = [{"type": "http.request", "body": json.dumps(QUESTION).encode()}]

        async def receive() -> dict[str, Any]:
            if requests:
                return requests.pop()
            await hung_up.wait()
            return {"type": "http.disconnect"}

        async def send(message: dict[str, Any]) -> None:
            sent.append(message.get("body", b""))
            if b"event: delta" in sent[-1]:
                hung_up.set()

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/api/ask/stream",
            "raw_path": b"/api/ask/stream",
            "query_string": b"",
            "root_path": "",
            "headers": [(b"host", b"testserver"), (b"content-type", b"application/json")],
            "client": ("127.0.0.1", 50000),
            "server": ("testserver", 80),
        }
        with anyio.fail_after(1.5):  # streaming all 200 chunks would take 2 s
            await client.app(scope, receive, send)

    anyio.run(run)

    assert any(b"event: delta" in body for body in sent)
    assert not any(b"event: done" in body for body in sent)
    assert llm.closed
    assert llm.streamed < 20
