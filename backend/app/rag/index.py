"""Build and open the search index: chunks.jsonl, manifest.json and the Chroma vector store.

Chroma, LangChain's Chroma store, OpenAI embeddings and the openai package are imported where
they are used, not with this module: the API imports it for the manifest and chunks.jsonl
(health, the evaluation's passages), and the public demo (PUBLIC_DEMO=true) serves those without
ever loading a vector store or an OpenAI client.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from functools import lru_cache, partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.core.config import Settings
from app.db.load import DOCUMENTS_FILE, DocumentRow, read_jsonl
from app.ingest.records import write_text_atomic
from app.rag.chunking import CHUNKER_VERSION, chunk_documents, count_tokens
from app.rag.filters import release_ord
from app.rag.types import Chunk

if TYPE_CHECKING:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    from langchain_chroma import Chroma
    from langchain_core.embeddings import Embeddings
    from langchain_openai import OpenAIEmbeddings

Source = Literal["db", "processed"]
Log = Callable[[str], None]
T = TypeVar("T")

COLLECTION = "fedrag_chunks"
CHUNKS_FILE = "chunks.jsonl"
MANIFEST_FILE = "manifest.json"
CHROMA_DB_FILE = "chroma.sqlite3"  # present once Chroma has initialized a directory
# Chroma shares one system per directory between clients, and two threads that create the first
# client for a directory at once can break it (one sees the system before it has started, then
# stops it under the other: AttributeError 'bindings', then KeyError). The API does exactly that
# when GET /api/health arrives during the startup warm-up, so clients are created one at a time.
_CHROMA_CLIENT_LOCK = threading.Lock()
EMBED_BATCH_SIZE = 256  # chunks per embeddings request: about 100k tokens, under OpenAI's cap
MAX_EMBED_TOKENS = 1_500_000  # refuse a build that would embed more (the corpus is about 1.1M)
RETRY_ATTEMPTS = 5
EMBED_TIMEOUT_S = 60.0  # per request, like the chat model's; a 256-chunk batch takes about 2 s
USD_PER_MILLION_TOKENS = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-ada-002": 0.10,
}
NOT_BUILT = "Search index not built: run make index."
NO_OPENAI_KEY = "OpenAI not configured: set OPENAI_API_KEY in backend/.env to embed chunks."

SELECT_DOCUMENTS = """
SELECT doc_id, doc_type, meeting_date, release_date, release_date_source, page_updated, title, url,
       paragraphs, text, word_count, sha256, parser_version
FROM fedrag.documents
ORDER BY doc_id
"""


@lru_cache(maxsize=1)
def _chroma_settings() -> ChromaSettings:
    """One settings object for every client: Chroma keeps one client per path and refuses a
    second one with different settings. Telemetry off, so building and searching never phone
    home."""
    from chromadb.config import Settings as ChromaSettings

    return ChromaSettings(anonymized_telemetry=False)


def _retryable() -> tuple[type[Exception], ...]:
    """The OpenAI errors an embedding batch is retried on: rate limits and transient failures."""
    import openai

    return (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError)


class IndexNotBuiltError(RuntimeError):
    """The index is missing, incomplete or built for another embedding model."""

    def __init__(self, detail: str = NOT_BUILT) -> None:
        super().__init__(detail)


class IndexBuildError(RuntimeError):
    """A build was refused before it changed anything."""


class EmbeddingModelMismatchError(IndexBuildError):
    """The Chroma collection holds vectors from a different embedding model."""


class EmbeddingsNotConfiguredError(RuntimeError):
    """Embedding needs OPENAI_API_KEY, which is not set."""

    def __init__(self) -> None:
        super().__init__(NO_OPENAI_KEY)


class IndexManifest(BaseModel):
    """index_dir/manifest.json, written last: its presence means the build completed."""

    model_config = ConfigDict(frozen=True)

    embedding_model: str = Field(description="Model that embedded every chunk in the collection.")
    collection: str = Field(description="Chroma collection name.")
    chunker_version: int = Field(description="CHUNKER_VERSION the chunks were cut with.")
    chunk_count: int = Field(description="Chunks in chunks.jsonl and in the collection.")
    document_count: int = Field(description="Source documents the chunks come from.")
    corpus_sha256: str = Field(description="Hash of the source documents' ids, texts and dates.")
    chunks_sha256: str = Field(description="Hash of chunks.jsonl.")
    source: Source = Field(description="Where the documents were read: Postgres or processed/.")
    built_at: datetime = Field(description="When the index last changed (UTC).")


@dataclass(frozen=True)
class IndexReport:
    """What a build did, or with dry_run would do."""

    documents: int
    chunks: int
    to_embed: int  # new or changed chunks
    to_delete: int  # chunks in Chroma that the corpus no longer has
    unchanged: int
    tokens: int  # tokens of the chunks to embed
    estimated_usd: float | None  # None for a model without a known price
    dry_run: bool


def make_embeddings(settings: Settings) -> OpenAIEmbeddings:
    """OpenAI embeddings for settings.embedding_model; the client retries 429s and 5xx itself, and
    gives up on a request after EMBED_TIMEOUT_S (without it a stalled connection never returns)."""
    if settings.openai_api_key is None:
        raise EmbeddingsNotConfiguredError
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
        chunk_size=EMBED_BATCH_SIZE,
        max_retries=3,
        timeout=EMBED_TIMEOUT_S,
    )


def chroma_client(chroma_dir: Path) -> chromadb.ClientAPI:
    """A persistent Chroma client for chroma_dir (created if missing), telemetry off. Safe to call
    from several threads at once (_CHROMA_CLIENT_LOCK)."""
    import chromadb

    chroma_dir.mkdir(parents=True, exist_ok=True)
    with _CHROMA_CLIENT_LOCK:
        return chromadb.PersistentClient(path=str(chroma_dir), settings=_chroma_settings())


def open_vectorstore(
    settings: Settings, embeddings: Embeddings | None, *, create: bool = False
) -> Chroma:
    """The LangChain Chroma store over the chunk collection (cosine distance).

    `embeddings` embeds added texts and queries (make_embeddings(settings), or a fake in tests);
    None opens the store for reads and deletes only. Without `create`, a missing collection raises
    IndexNotBuiltError.
    """
    if not create and not (settings.chroma_dir / CHROMA_DB_FILE).is_file():
        raise IndexNotBuiltError
    from chromadb.errors import NotFoundError
    from langchain_chroma import Chroma

    try:
        return Chroma(
            collection_name=COLLECTION,
            embedding_function=embeddings,
            client=chroma_client(settings.chroma_dir),
            collection_metadata={"embedding_model": settings.embedding_model},
            collection_configuration={"hnsw": {"space": "cosine"}},
            create_collection_if_not_exists=create,
        )
    except NotFoundError:
        raise IndexNotBuiltError from None


def load_documents(settings: Settings, source: Source = "db") -> list[DocumentRow]:
    """Documents from Postgres (a read-only transaction) or data/processed/documents.jsonl."""
    if source == "processed":
        return read_jsonl(settings.data_dir / "processed" / DOCUMENTS_FILE, DocumentRow)
    from app.db.connection import connect  # only the db source needs the database layer

    with connect(settings) as conn:
        conn.read_only = True
        rows = conn.execute(SELECT_DOCUMENTS).fetchall()
    try:
        return [DocumentRow.model_validate(row) for row in rows]
    except ValidationError as exc:
        raise IndexBuildError(
            f"fedrag.documents has a row that breaks the contract: {exc}"
        ) from None


def build_index(
    settings: Settings,
    source: Source = "db",
    *,
    documents: Sequence[DocumentRow] | None = None,
    rebuild: bool = False,
    dry_run: bool = False,
    embeddings: Embeddings | None = None,
    max_tokens: int = MAX_EMBED_TOKENS,
    log: Log = print,
    sleep: Callable[[float], None] = time.sleep,
) -> IndexReport:
    """Chunk the documents and bring the index up to date with them, incrementally.

    Each chunk's metadata carries a hash of its text and metadata, so only new or changed chunks
    are embedded and chunks that disappeared are deleted: a build with no changes makes no
    embedding calls. The collection records its embedding model, and a different
    settings.embedding_model is refused unless `rebuild`, which re-embeds everything. The token
    count and cost are logged before any embedding; dry_run stops there and changes nothing.
    `documents` (for tests) replaces reading from `source`.
    """
    docs = sorted(
        documents if documents is not None else load_documents(settings, source),
        key=lambda d: (d.release_date, d.doc_id),
    )
    chunks = chunk_documents(docs)
    metadata = {chunk.chunk_id: chunk_metadata(chunk) for chunk in chunks}
    if len(metadata) != len(chunks):
        raise IndexBuildError("duplicate chunk ids: the documents repeat a doc_id")
    by_type = ", ".join(f"{n} {t}" for t, n in Counter(c.doc_type for c in chunks).items())
    log(f"Chunked {len(docs)} documents into {len(chunks)} chunks ({by_type or 'none'}).")

    stored = {} if rebuild else _stored_hashes(settings)
    to_embed = [c for c in chunks if stored.get(c.chunk_id) != metadata[c.chunk_id]["content_hash"]]
    to_delete = sorted(set(stored) - set(metadata))
    tokens = sum(count_tokens(c.search_text) for c in to_embed)
    price = USD_PER_MILLION_TOKENS.get(settings.embedding_model)
    usd = None if price is None else tokens / 1_000_000 * price
    report = IndexReport(
        documents=len(docs),
        chunks=len(chunks),
        to_embed=len(to_embed),
        to_delete=len(to_delete),
        unchanged=len(chunks) - len(to_embed),
        tokens=tokens,
        estimated_usd=usd,
        dry_run=dry_run,
    )
    cost = (
        "unknown price" if usd is None else f"about ${usd:.4f}" if usd >= 0.001 else "under $0.001"
    )
    log(
        f"{'Rebuild' if rebuild else 'Update'}: {len(to_embed)} chunks to embed "
        f"({tokens:,} tokens with {settings.embedding_model}, {cost}), {len(to_delete)} to delete, "
        f"{report.unchanged} unchanged."
    )
    if tokens > max_tokens:
        raise IndexBuildError(
            f"refused: {tokens:,} tokens to embed is over the {max_tokens:,}-token safety cap."
        )
    if dry_run:
        return report

    if to_embed and embeddings is None:
        embeddings = make_embeddings(settings)  # raises before anything changes without a key
    if to_embed or to_delete or rebuild:  # not ready again until this build completes
        (settings.index_dir / MANIFEST_FILE).unlink(missing_ok=True)
    if rebuild:
        _delete_collection(settings)
    store = open_vectorstore(settings, embeddings, create=True)
    for ids in _batches(to_delete, EMBED_BATCH_SIZE):
        store.delete(ids=ids)
    batches = list(_batches(to_embed, EMBED_BATCH_SIZE))
    for number, batch in enumerate(batches, start=1):
        log(f"  embedding batch {number}/{len(batches)}: {len(batch)} chunks")
        add = partial(
            store.add_texts,
            [c.search_text for c in batch],
            metadatas=[metadata[c.chunk_id] for c in batch],
            ids=[c.chunk_id for c in batch],
        )
        _with_retries(add, sleep=sleep, log=log)

    count = chroma_client(settings.chroma_dir).get_collection(COLLECTION).count()
    if count != len(chunks):
        raise IndexBuildError(
            f"Chroma holds {count} chunks after the build, expected {len(chunks)}"
        )
    _write_index_files(settings, docs, chunks, source=source)
    log(f"Index ready: {len(chunks)} chunks in {settings.chroma_dir} (collection {COLLECTION}).")
    return report


def chunk_metadata(chunk: Chunk) -> dict[str, str | int]:
    """Chroma metadata for a chunk: what filters and lookups need, plus a content hash.

    Chroma cannot store None, so page_updated and section are left out when unset.
    """
    metadata: dict[str, str | int] = {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "doc_type": chunk.doc_type,
        "meeting_date": chunk.meeting_date.isoformat(),
        "release_date": chunk.release_date.isoformat(),
        "release_ord": release_ord(chunk.release_date),
        "title": chunk.title,
        "url": chunk.url,
        "index": chunk.index,
    }
    if chunk.page_updated is not None:
        metadata["page_updated"] = chunk.page_updated.isoformat()
    if chunk.section is not None:
        metadata["section"] = chunk.section
    payload = json.dumps([chunk.search_text, metadata], sort_keys=True, ensure_ascii=False)
    metadata["content_hash"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return metadata


def read_manifest(settings: Settings) -> IndexManifest | None:
    """The manifest of the last completed build, or None if there is none (or it is unreadable)."""
    path = settings.index_dir / MANIFEST_FILE
    try:
        return IndexManifest.model_validate_json(path.read_bytes())
    except (OSError, ValidationError):
        return None


def load_chunks(settings: Settings) -> list[Chunk]:
    """Every indexed chunk, in index order; raises IndexNotBuiltError if there is no index."""
    path = settings.index_dir / CHUNKS_FILE
    if read_manifest(settings) is None or not path.is_file():
        raise IndexNotBuiltError
    try:
        return [
            _chunk_from_json(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
        ]
    except (ValueError, TypeError, KeyError) as exc:
        raise IndexNotBuiltError(f"{path} is unreadable ({exc}): run make index.") from None


def index_ready(settings: Settings) -> bool:
    """True iff a completed build's manifest matches settings.embedding_model, chunks.jsonl exists
    and Chroma holds the manifest's chunk count. Opens nothing that does not already exist."""
    manifest = read_manifest(settings)
    if (
        manifest is None
        or manifest.embedding_model != settings.embedding_model
        or not (settings.index_dir / CHUNKS_FILE).is_file()
        or not (settings.chroma_dir / CHROMA_DB_FILE).is_file()
    ):
        return False
    from chromadb.errors import ChromaError

    try:
        count = chroma_client(settings.chroma_dir).get_collection(COLLECTION).count()
    except (ChromaError, ValueError, OSError):
        return False
    return count == manifest.chunk_count


def require_index(settings: Settings) -> None:
    """Raise IndexNotBuiltError, naming the cause, unless index_ready(settings)."""
    manifest = read_manifest(settings)
    if manifest is not None and manifest.embedding_model != settings.embedding_model:
        raise IndexNotBuiltError(
            f"The index was embedded with {manifest.embedding_model}, but EMBEDDING_MODEL is "
            f"{settings.embedding_model}: run make index with --rebuild."
        )
    if not index_ready(settings):
        raise IndexNotBuiltError


def _stored_hashes(settings: Settings) -> dict[str, str]:
    """chunk_id -> content_hash of what Chroma holds; refuses a collection from another model."""
    if not (settings.chroma_dir / CHROMA_DB_FILE).is_file():
        return {}
    from chromadb.errors import NotFoundError

    try:
        collection = chroma_client(settings.chroma_dir).get_collection(COLLECTION)
    except NotFoundError:
        return {}
    stored_model = (collection.metadata or {}).get("embedding_model")
    if stored_model != settings.embedding_model:
        raise EmbeddingModelMismatchError(
            f"refused: the index was embedded with {stored_model}, but EMBEDDING_MODEL is "
            f"{settings.embedding_model}. Vectors from two models cannot be mixed: rebuild with "
            "`python -m app.rag index --rebuild`."
        )
    result = collection.get(include=["metadatas"])
    return {
        chunk_id: str((meta or {}).get("content_hash", ""))
        for chunk_id, meta in zip(result["ids"], result["metadatas"] or [], strict=True)
    }


def _delete_collection(settings: Settings) -> None:
    if (settings.chroma_dir / CHROMA_DB_FILE).is_file():
        from chromadb.errors import NotFoundError

        with suppress(NotFoundError):
            chroma_client(settings.chroma_dir).delete_collection(COLLECTION)


def _write_index_files(
    settings: Settings, docs: Sequence[DocumentRow], chunks: Sequence[Chunk], *, source: Source
) -> None:
    """Write chunks.jsonl, then the manifest; an unchanged index keeps its manifest's built_at."""
    lines = "".join(json.dumps(_chunk_json(c), ensure_ascii=False) + "\n" for c in chunks)
    corpus = "\n".join(
        f"{d.doc_id} {d.sha256} {d.release_date} {d.page_updated}"
        for d in sorted(docs, key=lambda d: d.doc_id)
    )
    fields: dict[str, Any] = {
        "embedding_model": settings.embedding_model,
        "collection": COLLECTION,
        "chunker_version": CHUNKER_VERSION,
        "chunk_count": len(chunks),
        "document_count": len(docs),
        "corpus_sha256": hashlib.sha256(corpus.encode("utf-8")).hexdigest(),
        "chunks_sha256": hashlib.sha256(lines.encode("utf-8")).hexdigest(),
        "source": source,
    }
    previous = read_manifest(settings)
    same = previous is not None and previous.model_dump(exclude={"built_at"}) == fields
    chunks_path = settings.index_dir / CHUNKS_FILE
    if not same or _file_sha256(chunks_path) != fields["chunks_sha256"]:
        write_text_atomic(chunks_path, lines)  # also restores a deleted or edited chunks.jsonl
    if same:
        return
    manifest = IndexManifest(**fields, built_at=datetime.now(UTC).replace(microsecond=0))
    write_text_atomic(settings.index_dir / MANIFEST_FILE, manifest.model_dump_json(indent=2) + "\n")


def _file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _chunk_json(chunk: Chunk) -> dict[str, Any]:
    return {
        key: value.isoformat() if isinstance(value, date) else value
        for key, value in asdict(chunk).items()
    }


def _chunk_from_json(row: dict[str, Any]) -> Chunk:
    dates = {
        key: date.fromisoformat(row[key]) if row[key] else None
        for key in ("meeting_date", "release_date", "page_updated")
    }
    return Chunk(**(row | dates))


def _with_retries(call: Callable[[], T], *, sleep: Callable[[float], None], log: Log) -> T:
    """Run `call`, retrying rate limits and transient OpenAI errors with exponential backoff."""
    retryable = _retryable()
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            return call()
        except retryable as exc:
            if attempt == RETRY_ATTEMPTS:
                raise
            delay = min(60.0, 2.0**attempt)
            log(
                f"  {type(exc).__name__}: retrying in {delay:.0f}s ({attempt}/{RETRY_ATTEMPTS - 1})"
            )
            sleep(delay)
    raise AssertionError("unreachable")


def _batches(items: Sequence[T], size: int) -> Iterator[list[T]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])
