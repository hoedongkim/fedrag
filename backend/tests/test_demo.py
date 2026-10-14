"""The public demo (PUBLIC_DEMO=true): nothing that would call OpenAI runs, validation still comes
first, the recorded examples are served, and startup needs no key, index or model and imports no
torch, OpenAI client or vector store (checked in a fresh interpreter: tests/demo_import_probe.py).
The comparison's demo status is tested in tests/compare."""

import json
import logging
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.ask_examples import ASK_EXAMPLES_FILE, EXAMPLES_MISSING
from app.api.demo import ASK_DISABLED
from app.core.config import REPO_ROOT, Settings
from app.rag.index import CHUNKS_FILE, MANIFEST_FILE
from app.schemas.ask import AskExample

MakeClient = Callable[..., TestClient]

BACKEND = Path(__file__).resolve().parents[1]
PROBE = Path(__file__).with_name("demo_import_probe.py")
ASK = {"question": "What did the FOMC decide in September 2024?"}
ASK_PATHS = ["/api/ask", "/api/ask/stream"]
INVALID_BODIES = {
    "too-short": {"question": "Hi"},
    "top-k-zero": {"question": "Rates?", "top_k": 0},
    "unknown-field": {"question": "Rates?", "asof": "2020-03-01"},
    "not-json": None,
}


@pytest.fixture
def built(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Records every chat model and retriever the ask dependencies build (none should be)."""
    calls: list[str] = []
    monkeypatch.setattr("app.api.rag.get_llm", lambda settings: calls.append("llm"))
    monkeypatch.setattr(
        "app.api.rag.get_retriever", lambda *args, **kwargs: calls.append("retriever")
    )
    return calls


@pytest.fixture
def demo(make_client: MakeClient) -> TestClient:
    """The public demo, with a key set to show that the demo does not use it."""
    return make_client(public_demo=True, openai_api_key="sk-test")


@pytest.mark.parametrize("path", ASK_PATHS)
def test_ask_is_403_on_the_demo(demo: TestClient, built: list[str], path: str) -> None:
    response = demo.post(path, json=ASK)

    assert response.status_code == 403
    assert response.json() == {"detail": ASK_DISABLED}
    assert response.headers["content-type"] == "application/json"  # before any stream starts
    assert built == []


@pytest.mark.parametrize("body", INVALID_BODIES.values(), ids=INVALID_BODIES.keys())
@pytest.mark.parametrize("path", ASK_PATHS)
def test_validation_comes_before_the_demo_403(
    demo: TestClient, built: list[str], path: str, body: dict | None
) -> None:
    response = demo.post(path, json=body) if body else demo.post(path, content=b"{")

    assert response.status_code == 422
    assert built == []  # FastAPI resolves the chat model even then; on the demo it builds none


@pytest.mark.parametrize("path", ASK_PATHS)
def test_off_the_demo_the_same_request_reaches_the_setup_checks(
    make_client: MakeClient, path: str
) -> None:
    response = make_client().post(path, json=ASK)  # no key: the usual 503, not the demo's 403

    assert response.status_code == 503


def test_the_demo_skips_the_warm_up(
    make_client: MakeClient, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    started: list[Settings] = []
    monkeypatch.setattr("app.main.start_warm_up", started.append)
    client = make_client(public_demo=True, warmup=True)

    with caplog.at_level(logging.INFO, logger="app.main"), client:
        assert client.get("/api/health").json()["demo_mode"] is True
        assert client.app.state.warm_up is None
    assert started == []
    assert "Public demo" in caplog.text


# --- GET /api/ask/examples ------------------------------------------------------------------------


@pytest.mark.parametrize("demo_mode", [False, True], ids=["local", "public-demo"])
def test_examples_serve_the_tracked_file_in_every_mode(
    make_client: MakeClient, demo_mode: bool
) -> None:
    client = make_client(public_demo=demo_mode, data_dir=REPO_ROOT / "data")

    response = client.get("/api/ask/examples")

    assert response.status_code == 200
    examples = [AskExample.model_validate(e) for e in response.json()]
    assert len(examples) == 8
    assert len({e.id for e in examples}) == 8
    assert {e.retrieval for e in examples} == {"hybrid_rerank"}
    assert {e.retriever_preset for e in examples} == {"tuned"}
    assert any(e.as_of is not None for e in examples)  # a point-in-time question
    assert any(e.question_type == "unanswerable" for e in examples)
    for example in examples:
        assert example.judge is not None
        assert all(c.snippet and c.document.url for c in example.citations)
    # The fields of an AskResponse come first, so the page can render one as an answer.
    assert list(response.json()[0])[:7] == [
        "answer",
        "citations",
        "notes",
        "retrieval",
        "as_of",
        "model",
        "latency_ms",
    ]


def test_examples_are_503_until_exported(client: TestClient) -> None:
    response = client.get("/api/ask/examples")

    assert response.status_code == 503
    assert response.json() == {"detail": EXAMPLES_MISSING}


def test_an_invalid_examples_file_is_503_naming_the_problem(
    make_client: MakeClient, tmp_path: Path
) -> None:
    client = make_client()
    path = tmp_path / ASK_EXAMPLES_FILE
    path.parent.mkdir(parents=True)
    tracked = json.loads((REPO_ROOT / "data" / ASK_EXAMPLES_FILE).read_text(encoding="utf-8"))
    path.write_text(json.dumps(tracked | {"examples": []}), encoding="utf-8")

    response = client.get("/api/ask/examples")

    assert response.status_code == 503
    assert response.json()["detail"].startswith(
        "Example answers file data/eval/ask_examples.json is invalid: examples:"
    )

    path.write_text(json.dumps(tracked), encoding="utf-8")  # fixed on disk: served at once
    assert client.get("/api/ask/examples").status_code == 200


# --- Startup in a fresh interpreter ---------------------------------------------------------------


def test_the_demo_runs_without_key_index_or_heavy_imports(tmp_path: Path) -> None:
    """Startup and every demo endpoint, in a process of their own: no OPENAI_API_KEY, no Chroma,
    a search index of chunks.jsonl and its manifest only, as the Space ships it. torch, the
    openai package and Chroma are never imported, and no warm-up thread starts."""
    index_dir, chroma_dir = tmp_path / "index", tmp_path / "chroma"
    index_dir.mkdir()
    chunk = {
        "chunk_id": "statement-2017-12-13#0000",
        "doc_id": "statement-2017-12-13",
        "doc_type": "statement",
        "meeting_date": "2017-12-13",
        "release_date": "2017-12-13",
        "page_updated": None,
        "title": "FOMC Statement, December 13, 2017",
        "url": "https://www.federalreserve.gov/newsevents/pressreleases/monetary20171213a.htm",
        "section": None,
        "index": 0,
        "text": "Voting against the action were Charles L. Evans and Neel Kashkari.",
        "context": "FOMC Statement, December 13, 2017",
    }
    (index_dir / CHUNKS_FILE).write_text(json.dumps(chunk) + "\n", encoding="utf-8")
    manifest = {
        "embedding_model": "text-embedding-3-small",
        "collection": "fedrag_chunks",
        "chunker_version": 2,
        "chunk_count": 1,
        "document_count": 1,
        "corpus_sha256": "0" * 64,
        "chunks_sha256": "0" * 64,
        "source": "processed",
        "built_at": "2026-10-13T00:00:00Z",
    }
    (index_dir / MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")
    settings_env = {name.upper() for name in Settings.model_fields}
    env = {k: v for k, v in os.environ.items() if k not in settings_env}  # no OPENAI_API_KEY

    run = subprocess.run(
        [sys.executable, str(PROBE), str(REPO_ROOT / "data"), str(index_dir), str(chroma_dir)],
        cwd=BACKEND,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )

    assert run.returncode == 0, run.stderr
    result = json.loads(run.stdout.strip().splitlines()[-1])
    calls = result["calls"]
    assert calls["health"] | {"content_type": None} == {
        "status": 200,
        "content_type": None,
        "demo_mode": True,
        "llm_configured": False,
    }
    for name in ("ask", "ask_stream"):
        assert (calls[name]["status"], calls[name]["content_type"]) == (403, "application/json")
    assert calls["ask"]["detail"] == ASK_DISABLED
    assert calls["ask_invalid"]["status"] == 422
    assert (calls["ask_examples"]["status"], calls["ask_examples"]["count"]) == (200, 8)
    assert calls["evaluation"]["status"] == 200
    assert calls["evaluation_question"] | {"content_type": None} == {
        "status": 200,
        "content_type": None,
        "passages_available": True,
    }
    for name in ("compare_cached_only", "compare_generate"):
        assert (calls[name]["status"], calls[name]["analysis_status"]) == (200, "disabled_on_demo")
    assert result["imported"] == []
    assert result["warm_up"] is False and "warm-up" not in result["threads"]
    assert not chroma_dir.exists()
