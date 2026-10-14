"""Live check of the OpenAI embeddings + Chroma path on three real statements (under 2,000
tokens, a fraction of a cent): run with RUN_LIVE_TESTS=1 after make ingest."""

from datetime import date
from pathlib import Path

import pytest

from app.core.config import Settings
from app.db.load import DOCUMENTS_FILE, DocumentRow, read_jsonl
from app.rag.index import build_index, index_ready
from app.rag.retrievers import clear_retriever_cache, get_retriever

pytestmark = pytest.mark.live

STATEMENTS = ["statement-2022-03-16", "statement-2023-07-26", "statement-2024-09-18"]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """The developer's .env (for OPENAI_API_KEY), with the index and Chroma in tmp_path."""
    configured = Settings(chroma_dir=tmp_path / "chroma", index_dir=tmp_path / "index")
    if not configured.llm_configured:
        pytest.skip("OPENAI_API_KEY is not set in backend/.env")
    clear_retriever_cache()
    return configured


@pytest.fixture
def statements(settings: Settings) -> list[DocumentRow]:
    path = settings.data_dir / "processed" / DOCUMENTS_FILE
    if not path.is_file():
        pytest.skip(f"{path} not found: run make ingest")
    return [d for d in read_jsonl(path, DocumentRow) if d.doc_id in STATEMENTS]


def test_live_embed_and_search(settings: Settings, statements: list[DocumentRow]) -> None:
    report = build_index(settings, documents=statements, max_tokens=5_000)

    assert report.to_embed == len(STATEMENTS) and index_ready(settings)
    assert build_index(settings, documents=statements).to_embed == 0  # nothing re-embedded

    dense = get_retriever("dense", settings)
    hits = dense.retrieve("When did the Fed start cutting rates in 2024?", top_k=3)
    print([(h.chunk.doc_id, round(h.score, 3)) for h in hits])
    assert hits[0].chunk.doc_id == "statement-2024-09-18"
    assert all(0.0 < h.score <= 1.0 for h in hits)

    early = dense.retrieve("raise the target range", top_k=3, as_of=date(2023, 1, 1))
    assert [h.chunk.doc_id for h in early] == ["statement-2022-03-16"]
