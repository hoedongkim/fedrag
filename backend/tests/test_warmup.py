"""Tests for the retrieval warm-up the app starts in the background at startup."""

import hashlib
import logging
import threading
from collections.abc import Callable
from datetime import date
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding

from app.core.config import Settings, get_settings
from app.db.load import DocumentRow, ParagraphRow
from app.rag import retrievers
from app.rag.config import production_config
from app.rag.index import build_index
from app.rag.retrievers import (
    CrossEncoderScorer,
    RerankerUnavailableError,
    clear_retriever_cache,
    get_retriever,
)

MakeClient = Callable[..., TestClient]

TEXT = "The Committee decided to lower the target range to 4-3/4 to 5 percent."


@pytest.fixture
def loads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Stand in for loading the cross-encoder; records the model names it was asked for."""
    loaded: list[str] = []
    monkeypatch.setattr(CrossEncoderScorer, "load", lambda self: loaded.append(self.model_name))
    return loaded


@pytest.fixture
def bm25_builds(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Counts the BM25 indexes built (one per process and index build when cached)."""
    built: list[int] = []
    original = retrievers.BM25Retriever

    def counting(*args: Any, **kwargs: Any) -> Any:
        built.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(retrievers, "BM25Retriever", counting)
    clear_retriever_cache()
    return built


def build_small_index(settings: Settings) -> None:
    document = DocumentRow(
        doc_id="statement-2024-09-18",
        doc_type="statement",
        meeting_date=date(2024, 9, 18),
        release_date=date(2024, 9, 18),
        release_date_source="test",
        page_updated=None,
        title="FOMC Statement, September 18, 2024",
        url="https://www.federalreserve.gov/test/20240918.htm",
        paragraphs=[ParagraphRow(section=None, text=TEXT)],
        text=TEXT,
        word_count=len(TEXT.split()),
        sha256=hashlib.sha256(TEXT.encode("utf-8")).hexdigest(),
        parser_version=1,
    )
    embeddings = DeterministicFakeEmbedding(size=8)
    build_index(settings, documents=[document], embeddings=embeddings, log=lambda _: None)


def started(client: TestClient) -> threading.Thread:
    """The warm-up thread of a client whose lifespan is running, once it has finished."""
    thread = client.app.state.warm_up
    assert isinstance(thread, threading.Thread)
    thread.join(timeout=10)
    assert not thread.is_alive()
    return thread


def test_warmup_is_on_by_default_and_off_in_tests(client: TestClient, loads: list[str]) -> None:
    assert Settings(_env_file=None).warmup is True

    with client:
        assert client.app.state.warm_up is None
    assert loads == []


def test_warmup_builds_bm25_and_loads_the_reranker_once(
    make_client: MakeClient, loads: list[str], bm25_builds: list[int]
) -> None:
    client = make_client(warmup=True)
    settings: Settings = client.app.dependency_overrides[get_settings]()
    build_small_index(settings)

    with client:
        started(client)

    assert loads == [settings.reranker_model]
    assert bm25_builds == [1]
    assert get_retriever("bm25", settings).retrieve("lower the target range", top_k=1)
    assert bm25_builds == [1]  # the first query found it built


def test_warmup_loads_the_presets_pinned_reranker(
    make_client: MakeClient, loads: list[str], bm25_builds: list[int]
) -> None:
    # The settings name another model; the tuned preset pins its own, which production uses.
    client = make_client(warmup=True, reranker_model="other/model")
    settings: Settings = client.app.dependency_overrides[get_settings]()
    build_small_index(settings)

    with client:
        started(client)

    pinned = production_config(settings).reranker(settings)[0]
    assert pinned == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert loads == [pinned] and "other/model" not in loads


def test_warmup_never_delays_startup(
    make_client: MakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = threading.Event()
    monkeypatch.setattr(CrossEncoderScorer, "load", lambda self: release.wait(timeout=10))
    client = make_client(warmup=True)
    build_small_index(client.app.dependency_overrides[get_settings]())
    clear_retriever_cache()

    with client:  # startup returns while the reranker is still "loading"
        assert client.get("/api/health").json()["vector_store_ready"] is True
        assert client.app.state.warm_up.is_alive()
        release.set()
        started(client)


def test_warmup_without_an_index_is_skipped(
    make_client: MakeClient, loads: list[str], caplog: pytest.LogCaptureFixture
) -> None:
    client = make_client(warmup=True)

    with caplog.at_level(logging.INFO, logger="app.api.rag"), client:
        started(client)
        assert client.get("/api/health").status_code == 200

    assert loads == []
    assert "warm-up skipped: the search index is not built" in caplog.text


def test_warmup_failure_is_only_logged(
    make_client: MakeClient, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def unavailable(self: CrossEncoderScorer) -> None:
        raise RerankerUnavailableError("Reranker no/such-model is not downloaded")

    monkeypatch.setattr(CrossEncoderScorer, "load", unavailable)
    client = make_client(warmup=True)
    build_small_index(client.app.dependency_overrides[get_settings]())
    clear_retriever_cache()

    with caplog.at_level(logging.WARNING, logger="app.api.rag"), client:
        started(client)
        assert client.get("/api/health").status_code == 200

    assert "Retrieval warm-up failed (RerankerUnavailableError)" in caplog.text
