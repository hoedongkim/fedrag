"""Tests for the index build: incremental embedding, deletions, model guard, manifest, readiness."""

import hashlib
import threading
import time
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path

import httpx2
import openai
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings

from app.core.config import Settings
from app.db.load import DocumentRow, ParagraphRow
from app.rag.chunking import chunk_documents, count_tokens
from app.rag.index import (
    CHUNKS_FILE,
    COLLECTION,
    EMBED_TIMEOUT_S,
    MANIFEST_FILE,
    EmbeddingModelMismatchError,
    EmbeddingsNotConfiguredError,
    IndexBuildError,
    IndexNotBuiltError,
    IndexReport,
    build_index,
    chroma_client,
    index_ready,
    load_chunks,
    make_embeddings,
    read_manifest,
)


class CountingEmbeddings(Embeddings):
    """Deterministic fake vectors that record every text sent to be embedded."""

    def __init__(self, failures: list[Exception] | None = None) -> None:
        self._fake = DeterministicFakeEmbedding(size=16)
        self._failures = list(failures or [])
        self.texts: list[str] = []
        self.calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._failures:
            raise self._failures.pop(0)
        self.calls += 1
        self.texts.extend(texts)
        return self._fake.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._fake.embed_query(text)


def make_document(
    doc_type: str, meeting: date, texts: list[tuple[str | None, str]], **overrides: object
) -> DocumentRow:
    text = "\n\n".join(t for _, t in texts)
    fields: dict[str, object] = {
        "doc_id": f"{doc_type}-{meeting.isoformat()}",
        "doc_type": doc_type,
        "meeting_date": meeting,
        "release_date": meeting + timedelta(days=21 if doc_type == "minutes" else 0),
        "release_date_source": "test",
        "page_updated": None,
        "title": f"{'Minutes of the FOMC Meeting' if doc_type == 'minutes' else 'FOMC Statement'}"
        f", {meeting:%B %d, %Y}",
        "url": f"https://www.federalreserve.gov/test/{doc_type}{meeting:%Y%m%d}.htm",
        "paragraphs": [ParagraphRow(section=s, text=t) for s, t in texts],
        "text": text,
        "word_count": len(text.split()),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "parser_version": 1,
    }
    return DocumentRow.model_validate(fields | overrides)


def statement(meeting: date, decision: str) -> DocumentRow:
    return make_document("statement", meeting, [(None, f"The Committee decided {decision}.")])


def minutes(meeting: date) -> DocumentRow:
    sentence = "Participants judged that inflation pressures in sector {n} were easing gradually."
    body = " ".join(sentence.format(n=n) for n in range(30))
    return make_document(
        "minutes",
        meeting,
        [
            ("Preamble", "A meeting of the Federal Open Market Committee was held."),
            ("Participants' Views", body),
            ("Participants' Views", body.replace("easing", "building")),
            ("Attendance", "Jerome H. Powell, Chair"),
            ("Footnotes", "Attended Tuesday's session only."),
        ],
    )


CORPUS = [
    statement(date(2023, 7, 26), "to raise the target range to 5-1/4 to 5-1/2 percent"),
    minutes(date(2023, 7, 26)),
    statement(date(2024, 9, 18), "to lower the target range to 4-3/4 to 5 percent"),
    minutes(date(2024, 9, 18)),
]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        environment="test",
        openai_api_key=None,
        database_url=None,
        embedding_model="text-embedding-3-small",
        data_dir=tmp_path / "data",
        chroma_dir=tmp_path / "chroma",
        index_dir=tmp_path / "index",
    )


Build = Callable[..., IndexReport]


@pytest.fixture
def build(settings: Settings) -> Build:
    """build_index over in-memory documents, quiet, with no real sleeping."""

    def run(documents: list[DocumentRow], **kwargs: object) -> IndexReport:
        kwargs.setdefault("settings", settings)
        return build_index(
            source="processed",
            documents=documents,
            log=lambda _: None,
            sleep=lambda _: None,
            **kwargs,  # type: ignore[arg-type]
        )

    return run


def stored(settings: Settings) -> dict[str, dict[str, object]]:
    result = chroma_client(settings.chroma_dir).get_collection(COLLECTION).get()
    return dict(zip(result["ids"], result["metadatas"], strict=True))


def test_first_build_embeds_every_chunk(settings: Settings, build: Build) -> None:
    embeddings = CountingEmbeddings()
    chunks = chunk_documents(sorted(CORPUS, key=lambda d: (d.release_date, d.doc_id)))

    report = build(CORPUS, embeddings=embeddings)

    assert report.to_embed == len(chunks) == report.chunks and report.to_delete == 0
    assert report.tokens == sum(count_tokens(c.search_text) for c in chunks)
    assert report.estimated_usd == pytest.approx(report.tokens / 1e6 * 0.02)
    assert embeddings.texts == [c.search_text for c in chunks]
    assert load_chunks(settings) == chunks
    manifest = read_manifest(settings)
    assert manifest is not None
    assert (manifest.chunk_count, manifest.document_count) == (len(chunks), len(CORPUS))
    assert manifest.embedding_model == "text-embedding-3-small"
    assert index_ready(settings)

    metadata = stored(settings)
    first = metadata["statement-2023-07-26#0000"]
    assert first["release_ord"] == 20230726 and first["doc_type"] == "statement"
    assert "section" not in first and "page_updated" not in first  # Chroma cannot store None
    assert metadata["minutes-2023-07-26#0001"]["section"] == "Participants' Views"
    assert metadata["minutes-2023-07-26#0001"]["release_ord"] == 20230816


def test_a_build_with_no_changes_makes_no_embedding_calls(settings: Settings, build: Build) -> None:
    build(CORPUS, embeddings=CountingEmbeddings())
    files = {
        name: (settings.index_dir / name).read_bytes() for name in (CHUNKS_FILE, MANIFEST_FILE)
    }
    embeddings = CountingEmbeddings()

    report = build(CORPUS, embeddings=embeddings)
    build(CORPUS)  # no key is needed when nothing has to be embedded

    assert embeddings.calls == 0 and report.to_embed == 0 and report.unchanged == report.chunks
    assert files == {name: (settings.index_dir / name).read_bytes() for name in files}
    assert index_ready(settings)

    (settings.index_dir / CHUNKS_FILE).unlink()
    assert not index_ready(settings)
    build(CORPUS)  # restores chunks.jsonl, still without embedding anything
    assert files == {name: (settings.index_dir / name).read_bytes() for name in files}
    assert index_ready(settings)


def test_changed_text_re_embeds_only_that_chunk(settings: Settings, build: Build) -> None:
    build(CORPUS, embeddings=CountingEmbeddings())
    revised = statement(date(2024, 9, 18), "to lower the target range by 1/2 percentage point")
    embeddings = CountingEmbeddings()

    report = build([*CORPUS[:2], revised, CORPUS[3]], embeddings=embeddings)

    [chunk] = chunk_documents([revised])
    assert embeddings.texts == [chunk.search_text]
    assert (report.to_embed, report.to_delete) == (1, 0)
    assert chunk in load_chunks(settings)
    assert index_ready(settings)


def test_a_changed_release_date_re_embeds_with_new_metadata(
    settings: Settings, build: Build
) -> None:
    build(CORPUS, embeddings=CountingEmbeddings())
    moved = make_document(
        "statement",
        date(2024, 9, 18),
        [(None, "The Committee decided to lower the target range to 4-3/4 to 5 percent.")],
        release_date=date(2024, 9, 19),
    )
    embeddings = CountingEmbeddings()

    build([*CORPUS[:2], moved, CORPUS[3]], embeddings=embeddings)

    assert embeddings.calls == 1
    assert stored(settings)["statement-2024-09-18#0000"]["release_ord"] == 20240919


def test_a_removed_document_loses_its_chunks(settings: Settings, build: Build) -> None:
    build(CORPUS, embeddings=CountingEmbeddings())
    embeddings = CountingEmbeddings()

    report = build(CORPUS[:3], embeddings=embeddings)

    remaining = {c.chunk_id for c in chunk_documents(CORPUS[:3])}
    assert embeddings.calls == 0 and report.to_delete > 0
    assert set(stored(settings)) == remaining
    assert {c.chunk_id for c in load_chunks(settings)} == remaining
    assert index_ready(settings)


def test_a_different_embedding_model_needs_a_rebuild(settings: Settings, build: Build) -> None:
    build(CORPUS, embeddings=CountingEmbeddings())
    large = settings.model_copy(update={"embedding_model": "text-embedding-3-large"})
    embeddings = CountingEmbeddings()

    with pytest.raises(EmbeddingModelMismatchError, match="--rebuild"):
        build(CORPUS, settings=large, embeddings=embeddings)
    assert embeddings.calls == 0
    assert index_ready(settings) and not index_ready(large)

    report = build(CORPUS, settings=large, embeddings=embeddings, rebuild=True)

    assert report.to_embed == report.chunks == len(embeddings.texts)
    collection = chroma_client(settings.chroma_dir).get_collection(COLLECTION)
    assert collection.metadata["embedding_model"] == "text-embedding-3-large"
    assert index_ready(large) and not index_ready(settings)


def test_dry_run_estimates_and_changes_nothing(settings: Settings, build: Build) -> None:
    embeddings = CountingEmbeddings()

    report = build(CORPUS, embeddings=embeddings, dry_run=True)

    assert report.dry_run and report.to_embed == report.chunks > 0 and report.tokens > 0
    assert embeddings.calls == 0
    assert not settings.chroma_dir.exists() and not settings.index_dir.exists()


def test_the_token_cap_refuses_before_embedding(settings: Settings, build: Build) -> None:
    embeddings = CountingEmbeddings()

    with pytest.raises(IndexBuildError, match="safety cap"):
        build(CORPUS, embeddings=embeddings, max_tokens=100)
    assert embeddings.calls == 0 and not settings.chroma_dir.exists()


def test_without_a_key_a_build_fails_before_changing_anything(
    settings: Settings, build: Build
) -> None:
    build(CORPUS, embeddings=CountingEmbeddings())

    with pytest.raises(EmbeddingsNotConfiguredError):
        build([*CORPUS, statement(date(2024, 11, 7), "to lower the target range")])
    assert index_ready(settings)


def test_rate_limits_are_retried(settings: Settings) -> None:
    request = httpx2.Request("POST", "https://api.openai.com/v1/embeddings")
    limited = openai.RateLimitError(
        "rate limited", response=httpx2.Response(429, request=request), body=None
    )
    embeddings = CountingEmbeddings(failures=[limited, limited])
    sleeps: list[float] = []

    build_index(
        settings, documents=CORPUS, embeddings=embeddings, log=lambda _: None, sleep=sleeps.append
    )

    assert sleeps == [2.0, 4.0]
    assert index_ready(settings)


def test_build_reads_processed_documents(settings: Settings) -> None:
    processed = settings.data_dir / "processed"
    processed.mkdir(parents=True)
    (processed / "documents.jsonl").write_text(
        "".join(d.model_dump_json() + "\n" for d in CORPUS), encoding="utf-8"
    )

    report = build_index(settings, "processed", embeddings=CountingEmbeddings(), log=lambda _: None)

    manifest = read_manifest(settings)
    assert manifest is not None and manifest.source == "processed"
    assert report.documents == len(CORPUS) and manifest.chunk_count == report.chunks


def test_index_ready_needs_a_complete_index(settings: Settings, build: Build) -> None:
    assert not index_ready(settings)
    assert not settings.chroma_dir.exists()  # checking readiness creates nothing
    with pytest.raises(IndexNotBuiltError, match="make index"):
        load_chunks(settings)

    build(CORPUS, embeddings=CountingEmbeddings())
    assert index_ready(settings)

    chroma_client(settings.chroma_dir).get_collection(COLLECTION).delete(
        ids=["statement-2023-07-26#0000"]
    )
    assert not index_ready(settings)  # Chroma no longer holds the indexed chunk count

    (settings.index_dir / MANIFEST_FILE).unlink()
    assert not index_ready(settings)


def test_chroma_clients_are_created_one_at_a_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The startup warm-up and GET /api/health can open the store at once; Chroma breaks when two
    threads create the first client for a directory together, so creation is serialized."""
    import chromadb

    inside = 0
    most_inside = 0
    count_lock = threading.Lock()

    def slow_client(path: str, settings: object) -> object:
        nonlocal inside, most_inside
        with count_lock:
            inside += 1
            most_inside = max(most_inside, inside)
        time.sleep(0.05)
        with count_lock:
            inside -= 1
        return object()

    monkeypatch.setattr(chromadb, "PersistentClient", slow_client)
    threads = [
        threading.Thread(target=chroma_client, args=(tmp_path / "chroma",)) for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert most_inside == 1


def test_embedding_requests_time_out() -> None:
    embeddings = make_embeddings(Settings(_env_file=None, openai_api_key="sk-test"))

    assert embeddings.request_timeout == EMBED_TIMEOUT_S  # a stalled connection gives up
