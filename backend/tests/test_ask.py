"""Tests for POST /api/ask with the chat model faked out: no OpenAI calls, no real index."""

import hashlib
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import openai
import pytest
from fastapi.testclient import TestClient
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from app.api.rag import get_chat_model, get_retriever_factory
from app.core.config import Settings, get_settings
from app.db.load import DocumentRow, ParagraphRow
from app.rag.answer import LLM_NOT_CONFIGURED
from app.rag.index import NOT_BUILT, IndexNotBuiltError, build_index
from app.rag.retrievers import CrossEncoderScorer, RerankRetriever, clear_retriever_cache
from app.rag.types import Chunk, RetrievedChunk

MakeClient = Callable[..., TestClient]

QUESTION = {"question": "What did the Committee decide in September 2024?"}
CUT = (
    "The Committee decided to lower the target range for the federal funds rate by 1/2 "
    "percentage point to 4-3/4 to 5 percent."
)
CHUNK = Chunk(
    chunk_id="statement-2024-09-18#0001",
    doc_id="statement-2024-09-18",
    doc_type="statement",
    meeting_date=date(2024, 9, 18),
    release_date=date(2024, 9, 18),
    page_updated=None,
    title="FOMC Statement, September 18, 2024",
    url="https://www.federalreserve.gov/newsevents/pressreleases/monetary20240918a.htm",
    section=None,
    index=1,
    text=CUT,
    context="FOMC Statement, September 18, 2024",
)


class FakeRetriever:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def retrieve(self, query: str, **filters: Any) -> list[RetrievedChunk]:
        self.calls.append({"query": query, **filters})
        if self.error:
            raise self.error
        return [RetrievedChunk(CHUNK, score=0.031)]


class FakeChatModel(BaseChatModel):
    """Stands in for ChatOpenAI: replies with `reply`, or raises `error`; records each prompt."""

    reply: str = ""
    error: Exception | None = None
    prompts: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake"

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
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.reply))])


class Fakes:
    """The fakes installed on a client, and the retrieval methods the route asked for."""

    def __init__(self, client: TestClient, llm: FakeChatModel, retriever: FakeRetriever) -> None:
        self.llm = llm
        self.retriever = retriever
        self.methods: list[str] = []

        def factory(method: str) -> FakeRetriever:
            self.methods.append(method)
            return retriever

        client.app.dependency_overrides[get_chat_model] = lambda: llm
        client.app.dependency_overrides[get_retriever_factory] = lambda: factory


REPLY = "The Committee lowered the target range to 4-3/4 to 5 percent [1]."


def install(
    client: TestClient, *, llm: FakeChatModel | None = None, retriever: FakeRetriever | None = None
) -> Fakes:
    return Fakes(client, llm or FakeChatModel(reply=REPLY), retriever or FakeRetriever())


def test_answer_shape(client: TestClient) -> None:
    fakes = install(client)

    response = client.post("/api/ask", json={**QUESTION, "as_of": "2024-10-01", "top_k": 3})

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body.pop("latency_ms"), float)
    assert body == {
        "answer": "The Committee lowered the target range to 4-3/4 to 5 percent [1].",
        "citations": [
            {
                "document": {
                    "doc_id": "statement-2024-09-18",
                    "doc_type": "statement",
                    "meeting_date": "2024-09-18",
                    "release_date": "2024-09-18",
                    "page_updated": None,
                    "title": "FOMC Statement, September 18, 2024",
                    "url": CHUNK.url,
                },
                "chunk_id": "statement-2024-09-18#0001",
                "section": None,
                "snippet": CUT,
                "passage": None,  # only recorded answers carry the whole passage
                "score": 0.031,
            }
        ],
        "notes": [],
        "retrieval": "hybrid_rerank",
        "as_of": "2024-10-01",
        "model": Settings(_env_file=None).llm_model,
    }
    assert fakes.methods == ["hybrid_rerank"]
    assert fakes.retriever.calls == [
        {"query": QUESTION["question"], "top_k": 3, "as_of": date(2024, 10, 1), "doc_types": None}
    ]


def test_request_picks_the_retrieval_method(client: TestClient) -> None:
    fakes = install(client)

    response = client.post(
        "/api/ask", json={**QUESTION, "retrieval": "bm25", "doc_types": ["minutes", "minutes"]}
    )

    assert response.status_code == 200
    assert response.json()["retrieval"] == "bm25"
    assert fakes.methods == ["bm25"]
    assert fakes.retriever.calls[0]["doc_types"] == ["minutes"]  # duplicates dropped


@pytest.mark.parametrize(
    "body",
    [
        QUESTION,
        {
            "question": "How did forward guidance change?",
            "as_of": "2024-09-18",
            "retrieval": "bm25",
            "top_k": 20,
            "doc_types": ["statement", "minutes"],
        },
    ],
    ids=["defaults", "every-option"],
)
def test_503_without_api_key(
    client: TestClient,
    index_ready: Callable[[bool], list[Settings]],
    body: dict[str, Any],
) -> None:
    seen = index_ready(True)

    response = client.post("/api/ask", json=body)

    assert response.status_code == 503
    assert response.json() == {"detail": LLM_NOT_CONFIGURED}
    assert LLM_NOT_CONFIGURED == "LLM not configured: set OPENAI_API_KEY in backend/.env."
    assert seen == []


def test_503_without_index(
    make_client: MakeClient, index_ready: Callable[[bool], list[Settings]]
) -> None:
    client = make_client(openai_api_key="sk-test")
    seen = index_ready(False)

    response = client.post("/api/ask", json=QUESTION)

    assert response.status_code == 503
    assert response.json() == {"detail": NOT_BUILT}
    assert NOT_BUILT == "Search index not built: run make index."
    assert seen == [client.app.dependency_overrides[get_settings]()]


def test_503_when_the_index_breaks_mid_request(client: TestClient) -> None:
    install(client, retriever=FakeRetriever(IndexNotBuiltError("Chroma is out of sync.")))

    response = client.post("/api/ask", json=QUESTION)

    assert response.status_code == 503
    assert response.json() == {"detail": "Chroma is out of sync."}


@pytest.mark.parametrize(
    "body",
    [
        {"question": "Hi"},
        {"question": "Rates?", "top_k": 0},
        {"question": "Rates?", "retrieval": "tfidf"},
        {"question": "Rates?", "as_of": "2024-02-30"},
        {"question": "Rates?", "doc_types": []},
        {"question": "Rates?", "doc_types": ["minutes"] * 5},
        {"question": "Rates?", "asof": "2024-10-01"},
    ],
    ids=[
        "short-question",
        "top-k-zero",
        "unknown-method",
        "bad-date",
        "empty-doc-types",
        "too-many-doc-types",
        "typo",
    ],
)
def test_validation_comes_before_retrieval_and_llm(
    client: TestClient, body: dict[str, Any]
) -> None:
    fakes = install(client)

    assert client.post("/api/ask", json=body).status_code == 422
    assert fakes.methods == []
    assert fakes.llm.prompts == []


def _response(status_code: int) -> httpx.Response:
    return httpx.Response(status_code, request=httpx.Request("POST", "https://api.openai.com"))


NO_CREDIT_DETAIL = (
    "OpenAI request failed (RateLimitError). The OpenAI account behind this server has no credit "
    "left; add credit at platform.openai.com/billing."
)


def _rate_limit(code: str, *, as_type: bool = False) -> openai.RateLimitError:
    """A fake OpenAI 429 whose error body carries `code` (or, with as_type, only its type), and a
    message quoting part of the key, which must never be echoed."""
    message = f"You exceeded your current quota (sk-abc...wxyz): {code}."
    body = {"message": message, "type": code if as_type else "requests", "param": None}
    if not as_type:
        body["code"] = code
    return openai.RateLimitError(message, response=_response(429), body=body)


@pytest.mark.parametrize(
    ("llm_error", "retriever_error", "detail"),
    [
        (
            openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.com")),
            None,
            "OpenAI request failed (APIConnectionError). Try again shortly.",
        ),
        (
            None,
            openai.RateLimitError("Rate limit reached", response=_response(429), body=None),
            "OpenAI request failed (RateLimitError). Try again shortly.",
        ),
        (
            openai.AuthenticationError(
                "Incorrect API key provided: sk-abc...wxyz", response=_response(401), body=None
            ),
            None,
            "OpenAI request failed (AuthenticationError). Check OPENAI_API_KEY in backend/.env.",
        ),
        (_rate_limit("insufficient_quota"), None, NO_CREDIT_DETAIL),
        (None, _rate_limit("credit_balance_exhausted"), NO_CREDIT_DETAIL),
        (_rate_limit("insufficient_quota", as_type=True), None, NO_CREDIT_DETAIL),
        (
            _rate_limit("rate_limit_exceeded"),
            None,
            "OpenAI request failed (RateLimitError). Try again shortly.",
        ),
    ],
    ids=[
        "llm-unreachable",
        "embedding-rate-limited",
        "bad-key",
        "llm-no-credit",
        "embedding-no-credit",
        "no-credit-by-type",
        "llm-rate-limited",
    ],
)
def test_openai_failure_is_502(
    client: TestClient,
    llm_error: Exception | None,
    retriever_error: Exception | None,
    detail: str,
) -> None:
    install(client, llm=FakeChatModel(error=llm_error), retriever=FakeRetriever(retriever_error))

    response = client.post("/api/ask", json=QUESTION)

    assert response.status_code == 502
    assert response.json() == {"detail": detail}
    assert "sk-abc" not in response.text
    assert "exceeded your current quota" not in response.text  # OpenAI's message is not echoed


def test_503_when_the_reranker_cannot_be_loaded(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def not_cached(*args: Any, **kwargs: Any) -> None:
        raise OSError("not in the cache and no network")

    monkeypatch.setattr("sentence_transformers.CrossEncoder", not_cached)
    install(client, retriever=RerankRetriever(FakeRetriever(), CrossEncoderScorer("no/such-model")))

    response = client.post("/api/ask", json=QUESTION)

    assert response.status_code == 503
    assert "no/such-model" in response.json()["detail"]


def _statement(meeting: date, text: str) -> DocumentRow:
    return DocumentRow(
        doc_id=f"statement-{meeting.isoformat()}",
        doc_type="statement",
        meeting_date=meeting,
        release_date=meeting,
        release_date_source="test",
        page_updated=None,
        title=f"FOMC Statement, {meeting:%B %d, %Y}",
        url=f"https://www.federalreserve.gov/test/{meeting:%Y%m%d}.htm",
        paragraphs=[ParagraphRow(section=None, text=text)],
        text=text,
        word_count=len(text.split()),
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        parser_version=1,
    )


def test_answers_from_the_built_index(make_client: MakeClient, tmp_path: Path) -> None:
    """The real factory and index_ready over an index built with fake embeddings; BM25 needs no
    query embedding, so nothing calls OpenAI. as_of keeps the 2024 statement out of the prompt."""
    client = make_client(openai_api_key="sk-test")
    settings = client.app.dependency_overrides[get_settings]()
    assert settings.index_dir == tmp_path / "index"
    documents = [
        _statement(
            date(2023, 7, 26),
            "The Committee decided to raise the target range to 5-1/4 to 5-1/2 percent.",
        ),
        _statement(date(2024, 9, 18), CUT),
    ]
    fake = DeterministicFakeEmbedding(size=8)
    build_index(settings, documents=documents, embeddings=fake, log=lambda _: None)
    clear_retriever_cache()
    llm = FakeChatModel(reply="It raised the range [1].")
    client.app.dependency_overrides[get_chat_model] = lambda: llm

    assert client.get("/api/health").json()["vector_store_ready"] is True
    response = client.post(
        "/api/ask",
        json={"question": "target range decision", "as_of": "2024-01-01", "retrieval": "bm25"},
    )

    assert response.status_code == 200
    [citation] = response.json()["citations"]
    assert citation["chunk_id"] == "statement-2023-07-26#0000"
    assert citation["document"]["release_date"] == "2023-07-26"
    prompt = "\n".join(str(message.content) for message in llm.prompts[0])
    assert "July 26, 2023" in prompt and "September 18, 2024" not in prompt


def test_503_names_an_embedding_model_mismatch(make_client: MakeClient) -> None:
    """An index built with another EMBEDDING_MODEL says to rebuild, not to build, for any method."""
    client = make_client(openai_api_key="sk-test", embedding_model="text-embedding-3-large")
    settings = client.app.dependency_overrides[get_settings]()
    built_with = settings.model_copy(update={"embedding_model": "text-embedding-3-small"})
    documents = [_statement(date(2024, 9, 18), CUT)]
    build_index(
        built_with,
        documents=documents,
        embeddings=DeterministicFakeEmbedding(size=8),
        log=lambda _: None,
    )
    clear_retriever_cache()
    client.app.dependency_overrides[get_chat_model] = lambda: FakeChatModel(reply=REPLY)

    response = client.post("/api/ask", json={**QUESTION, "retrieval": "bm25"})

    assert response.status_code == 503
    assert "text-embedding-3-small" in response.json()["detail"]
    assert "--rebuild" in response.json()["detail"]
