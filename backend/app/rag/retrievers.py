"""Dense, hybrid (reciprocal rank fusion) and reranked retrieval, and a cached retriever factory
that builds any method under a RetrieverConfig (app/rag/config.py).

Chroma (through app.rag.index), torch and sentence-transformers are imported when a retriever
first needs them, so importing this module loads none of them."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from functools import lru_cache, partial
from typing import TYPE_CHECKING, Any, TypeVar

from langchain_core.embeddings import Embeddings

from app.core.config import Settings, get_settings
from app.rag.bm25 import BM25Retriever
from app.rag.config import MIN_CANDIDATES, RRF_K, RetrieverConfig, production_config
from app.rag.filters import ChunkFactor, MetadataPrior, chroma_where
from app.rag.index import (
    IndexManifest,
    IndexNotBuiltError,
    load_chunks,
    make_embeddings,
    open_vectorstore,
    read_manifest,
)
from app.rag.types import Chunk, RetrievedChunk, Retriever
from app.schemas.common import DocType, RetrievalMethod

if TYPE_CHECKING:
    from langchain_chroma import Chroma

T = TypeVar("T")
Scorer = Callable[[str, Sequence[str]], Sequence[float]]  # (query, passages) -> one score each


def candidate_count(top_k: int, minimum: int = MIN_CANDIDATES) -> int:
    """Candidates fetched per ranked list before fusing, reranking or applying a prior:
    max(minimum, 4 * top_k)."""
    return max(minimum, 4 * top_k)


class DenseRetriever:
    """Chroma similarity search over embeddings of Chunk.search_text, filtered inside Chroma.

    Scores are cosine similarity, 1 - Chroma's cosine distance: higher is better, at most 1.
    Query vectors are cached (LRU), so methods that share this retriever embed a question once.
    """

    def __init__(
        self, store: Chroma, chunks: Mapping[str, Chunk], *, query_cache_size: int = 256
    ) -> None:
        if store.embeddings is None:
            raise ValueError("DenseRetriever needs a Chroma store with an embedding function")
        self._store = store
        self._chunks = chunks
        self._embed_query = lru_cache(maxsize=query_cache_size)(store.embeddings.embed_query)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        as_of: date | None = None,
        doc_types: Sequence[DocType] | None = None,
    ) -> list[RetrievedChunk]:
        if top_k <= 0 or (doc_types is not None and not doc_types):
            return []
        hits = self._store.similarity_search_by_vector_with_relevance_scores(
            self._embed_query(query), k=top_k, filter=chroma_where(as_of, doc_types)
        )
        # LangChain's method name notwithstanding, Chroma returns distances: lower is closer.
        return [RetrievedChunk(self._chunk(doc.id), 1.0 - distance) for doc, distance in hits]

    def _chunk(self, chunk_id: str | None) -> Chunk:
        if chunk_id is None or chunk_id not in self._chunks:
            raise IndexNotBuiltError(
                f"Chroma returned {chunk_id}, which is not in chunks.jsonl: "
                "the index is out of sync, run make index."
            )
        return self._chunks[chunk_id]


class HybridRetriever:
    """Reciprocal rank fusion of a sparse and a dense ranking: score = sum of weight / (k + rank).

    Only sparse hits that match a query term are fused: BM25 pads its ranking with score-0
    chunks in corpus order, which would otherwise get full RRF weight. Dense still returns every
    eligible chunk up to the candidate count, so a cutoff never leaves fewer than top_k. A prior
    multiplies the fused scores before the cutoff.

    Custom: LangChain's EnsembleRetriever lives in langchain-classic (not installed) and cannot
    pass per-query as_of and doc_types filters down to the retrievers it fuses.
    """

    def __init__(
        self,
        sparse: Retriever,
        dense: Retriever,
        *,
        k: int = RRF_K,
        weights: tuple[float, float] = (1.0, 1.0),
        minimum: int = MIN_CANDIDATES,
        prior: MetadataPrior | None = None,
    ) -> None:
        self._retrievers = (sparse, dense)
        self._k = k
        self._weights = weights
        self._minimum = minimum
        self._prior = prior

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        as_of: date | None = None,
        doc_types: Sequence[DocType] | None = None,
    ) -> list[RetrievedChunk]:
        if top_k <= 0:
            return []
        n = candidate_count(top_k, self._minimum)
        sparse, dense = self._retrievers
        rankings = [
            [
                hit
                for hit in sparse.retrieve(query, top_k=n, as_of=as_of, doc_types=doc_types)
                if hit.score > 0  # BM25 pads with score-0 non-matches
            ],
            dense.retrieve(query, top_k=n, as_of=as_of, doc_types=doc_types),
        ]
        fused = reciprocal_rank_fusion(rankings, k=self._k, weights=self._weights)
        return apply_prior(fused, self._prior, query, as_of)[:top_k]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RetrievedChunk]],
    *,
    k: int = RRF_K,
    weights: Sequence[float] | None = None,
) -> list[RetrievedChunk]:
    """Fuse rankings by RRF (Cormack et al., 2009), each rank adding weight / (k + rank) (weights
    default to 1); ties keep the chunk first reached by rank."""
    weights = [1.0] * len(rankings) if weights is None else list(weights)
    if len(weights) != len(rankings):
        raise ValueError(f"{len(weights)} weights for {len(rankings)} rankings")
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for rank in range(max((len(ranking) for ranking in rankings), default=0)):
        # Interleaved, so insertion order is best-rank order.
        for ranking, weight in zip(rankings, weights, strict=True):
            if rank < len(ranking):
                chunk = ranking[rank].chunk
                scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + weight / (k + rank + 1)
                chunks.setdefault(chunk.chunk_id, chunk)
    order = sorted(scores, key=lambda chunk_id: -scores[chunk_id])  # stable sort
    return [RetrievedChunk(chunks[chunk_id], scores[chunk_id]) for chunk_id in order]


def boosted(score: float, factor: float) -> float:
    """score times factor, or divided by it for a negative score (a cosine similarity can be),
    so a factor above 1 always raises a score and one below 1 always lowers it."""
    return score * factor if score >= 0 else score / factor


def apply_prior(
    hits: list[RetrievedChunk], prior: MetadataPrior | None, query: str, as_of: date | None
) -> list[RetrievedChunk]:
    """The hits re-scored by the prior's factors and re-sorted (stable: ties keep their order),
    or unchanged if no factor applies to this query."""
    factor = None if prior is None else prior.for_query(query, as_of)
    return hits if factor is None else _rescored(hits, factor)


def _rescored(hits: Sequence[RetrievedChunk], factor: ChunkFactor) -> list[RetrievedChunk]:
    rescored = [RetrievedChunk(hit.chunk, boosted(hit.score, factor(hit.chunk))) for hit in hits]
    return sorted(rescored, key=lambda hit: -hit.score)


class PriorRetriever:
    """Applies a metadata prior to another retriever's ranking: fetches candidate_count(top_k)
    candidates (filters still apply inside), multiplies their scores, returns the best top_k.
    A query the prior does not apply to is passed through unchanged."""

    def __init__(
        self, inner: Retriever, prior: MetadataPrior, *, minimum: int = MIN_CANDIDATES
    ) -> None:
        self._inner = inner
        self._prior = prior
        self._minimum = minimum

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        as_of: date | None = None,
        doc_types: Sequence[DocType] | None = None,
    ) -> list[RetrievedChunk]:
        if top_k <= 0:
            return []
        factor = self._prior.for_query(query, as_of)
        if factor is None:
            return self._inner.retrieve(query, top_k=top_k, as_of=as_of, doc_types=doc_types)
        pool = self._inner.retrieve(
            query, top_k=candidate_count(top_k, self._minimum), as_of=as_of, doc_types=doc_types
        )
        return _rescored(pool, factor)[:top_k]


class RerankRetriever:
    """Reranks the top candidate_count(top_k) candidates of another retriever with a scorer that
    reads query and passage together (a cross-encoder); scores are the scorer's, higher is better,
    times the prior's factors when there is one.
    """

    def __init__(
        self,
        candidates: Retriever,
        scorer: Scorer,
        *,
        minimum: int = MIN_CANDIDATES,
        prior: MetadataPrior | None = None,
    ) -> None:
        self._candidates = candidates
        self._scorer = scorer
        self._minimum = minimum
        self._prior = prior

    @property
    def scorer(self) -> Scorer:
        return self._scorer

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        as_of: date | None = None,
        doc_types: Sequence[DocType] | None = None,
    ) -> list[RetrievedChunk]:
        if top_k <= 0:
            return []
        pool = self._candidates.retrieve(
            query, top_k=candidate_count(top_k, self._minimum), as_of=as_of, doc_types=doc_types
        )
        if not pool:
            return []
        scores = [float(s) for s in self._scorer(query, [hit.chunk.search_text for hit in pool])]
        if len(scores) != len(pool):
            raise ValueError(f"the scorer returned {len(scores)} scores for {len(pool)} passages")
        if self._prior is not None and (factor := self._prior.for_query(query, as_of)):
            scores = [boosted(s, factor(hit.chunk)) for s, hit in zip(scores, pool, strict=True)]
        order = sorted(range(len(pool)), key=lambda i: -scores[i])  # stable: ties keep fused order
        return [RetrievedChunk(pool[i].chunk, scores[i]) for i in order[:top_k]]


class RerankerUnavailableError(RuntimeError):
    """The cross-encoder is not cached and could not be downloaded."""


class CrossEncoderScorer:
    """A sentence-transformers CrossEncoder, loaded on first use on MPS, CUDA or CPU.

    Scores are sigmoid(logit) in (0, 1). Calls are serialized: one model, one device. `revision`
    pins the Hugging Face commit, so a fresh download scores like the cached copy.
    """

    def __init__(
        self,
        model_name: str,
        *,
        revision: str | None = None,
        device: str | None = None,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.revision = revision
        self._device = device
        self._batch_size = batch_size
        self._model: Any = None
        self._lock = threading.Lock()

    def __call__(self, query: str, passages: Sequence[str]) -> list[float]:
        import torch  # deferred: torch and the model are loaded only when reranking first runs

        with self._lock:
            scores = self._loaded(torch).predict(
                [(query, passage) for passage in passages],
                batch_size=self._batch_size,
                activation_fn=torch.nn.Sigmoid(),
                show_progress_bar=False,
            )
        return [float(score) for score in scores]

    def load(self) -> None:
        """Load the model now instead of on the first call (a no-op once loaded); raises
        RerankerUnavailableError if it is not cached and cannot be downloaded."""
        import torch

        with self._lock:
            self._loaded(torch)

    def _loaded(self, torch: Any) -> Any:
        """The model, loaded on first use; the caller holds the lock."""
        if self._model is None:
            self._model = self._load(torch)
        return self._model

    def _load(self, torch: Any) -> Any:
        from sentence_transformers import CrossEncoder
        from transformers.utils import logging as transformers_logging

        transformers_logging.disable_progress_bar()  # no "Loading weights" bar in CLI or API logs

        device = self._device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        load = partial(
            CrossEncoder,
            self.model_name,
            revision=self.revision,
            device=device,
            trust_remote_code=False,
        )
        try:  # the cached copy, without asking the Hugging Face Hub for updates
            return load(local_files_only=True)
        except OSError:  # not downloaded yet
            try:
                return load()
            except OSError as exc:
                raise RerankerUnavailableError(
                    f"Reranker {self.model_name} is not downloaded and could not be fetched from "
                    "Hugging Face: run once with network access, or use retrieval bm25, dense or "
                    "hybrid."
                ) from exc


def get_retriever(
    method: RetrievalMethod,
    settings: Settings | None = None,
    config: RetrieverConfig | None = None,
    *,
    embeddings: Embeddings | None = None,
    scorer: Scorer | None = None,
) -> Retriever:
    """The retriever for `method` over the built index, under `config` (default: the production
    config, production_config(settings): the preset RETRIEVER_PRESET names, tuned unless set).

    Parts are built once per process and index build: the chunk list, the BM25 index (one per
    stemming setting), the Chroma store (with its query cache) and each CrossEncoder. A rebuild
    changes the manifest, and the next call picks the new index up. `embeddings` and `scorer`
    replace OpenAI and the CrossEncoder (tests); a part built from them is not cached. Raises
    IndexNotBuiltError without an index.
    """
    settings = settings or get_settings()
    config = config or production_config(settings)
    manifest = read_manifest(settings)
    if manifest is None:
        raise IndexNotBuiltError
    if method not in ("bm25", "dense", "hybrid", "hybrid_rerank"):
        raise ValueError(f"unknown retrieval method: {method}")
    prior = config.prior()
    sparse = _bm25(settings, manifest, stem=config.bm25_stemming)
    if config.query_expansion:
        sparse = sparse.with_query_expansion(config.expansion_weight)
    if method == "bm25":
        return _with_prior(sparse, prior, config)
    dense = _dense(settings, manifest, embeddings)
    if method == "dense":
        return _with_prior(dense, prior, config)
    hybrid = HybridRetriever(
        sparse,
        dense,
        k=config.rrf_k,
        weights=config.rrf_weights,
        minimum=config.min_candidates,
        prior=prior,
    )
    if method == "hybrid":
        return hybrid
    return RerankRetriever(
        hybrid,
        scorer or _cross_encoder(*config.reranker(settings)),
        minimum=config.min_candidates,
        prior=prior,
    )


def _with_prior(
    retriever: Retriever, prior: MetadataPrior | None, config: RetrieverConfig
) -> Retriever:
    if prior is None:
        return retriever
    return PriorRetriever(retriever, prior, minimum=config.min_candidates)


def preload(retriever: Retriever) -> None:
    """Load the model `retriever` would otherwise load on its first query, the cross-encoder of
    hybrid_rerank, so that a reranker that cannot be loaded fails now, before any answer starts
    (RerankerUnavailableError). Blocking, and a no-op for other retrievers or a loaded model."""
    if isinstance(retriever, RerankRetriever) and isinstance(retriever.scorer, CrossEncoderScorer):
        retriever.scorer.load()


def warm_up(settings: Settings) -> None:
    """Build what the first hybrid_rerank query of the production config would otherwise build,
    about 5 s together: the chunk list and BM25 index, the Chroma store (with OPENAI_API_KEY set;
    nothing is embedded) and the cross-encoder. Makes no OpenAI request. Raises
    IndexNotBuiltError without an index and RerankerUnavailableError if the reranker can be
    neither loaded nor downloaded."""
    config = production_config(settings)
    manifest = read_manifest(settings)
    if manifest is None:
        raise IndexNotBuiltError
    _bm25(settings, manifest, stem=config.bm25_stemming)
    if settings.llm_configured:
        _dense(settings, manifest, None)
    _cross_encoder(*config.reranker(settings)).load()


def clear_retriever_cache() -> None:
    """Forget every cached part (tests; or to free the models)."""
    with _CACHE_LOCK:
        _CACHE.clear()


# kind -> (signature, value): one live value per kind, replaced when its signature changes.
_CACHE: dict[str, tuple[tuple[Any, ...], Any]] = {}
_CACHE_LOCK = threading.RLock()  # re-entrant: building one part may build another


def _cached(kind: str, signature: tuple[Any, ...], build: Callable[[], T]) -> T:
    with _CACHE_LOCK:
        hit = _CACHE.get(kind)
        if hit is None or hit[0] != signature:
            hit = (signature, build())
            _CACHE[kind] = hit
        return hit[1]


def _index_signature(settings: Settings, manifest: IndexManifest) -> tuple[Any, ...]:
    return (str(settings.index_dir), manifest.chunks_sha256, manifest.built_at)


def _chunks(settings: Settings, manifest: IndexManifest) -> list[Chunk]:
    return _cached("chunks", _index_signature(settings, manifest), lambda: load_chunks(settings))


def _bm25(settings: Settings, manifest: IndexManifest, *, stem: bool = False) -> BM25Retriever:
    if stem:
        return _cached(
            "bm25:stemmed",
            _index_signature(settings, manifest),
            lambda: BM25Retriever(_chunks(settings, manifest), stem=True),
        )
    return _cached(
        "bm25",
        _index_signature(settings, manifest),
        lambda: BM25Retriever(_chunks(settings, manifest)),
    )


def _cross_encoder(model_name: str, revision: str | None) -> CrossEncoderScorer:
    """One cached CrossEncoder per model name (a new revision replaces the old one)."""
    return _cached(
        f"scorer:{model_name}",
        (model_name, revision),
        lambda: CrossEncoderScorer(model_name, revision=revision),
    )


def _dense(
    settings: Settings, manifest: IndexManifest, embeddings: Embeddings | None
) -> DenseRetriever:
    if manifest.embedding_model != settings.embedding_model:
        raise IndexNotBuiltError(
            f"The index was embedded with {manifest.embedding_model}, but EMBEDDING_MODEL is "
            f"{settings.embedding_model}: run make index with --rebuild."
        )

    def build(embedder: Embeddings) -> DenseRetriever:
        chunks = {chunk.chunk_id: chunk for chunk in _chunks(settings, manifest)}
        return DenseRetriever(open_vectorstore(settings, embedder), chunks)

    if embeddings is not None:
        return build(embeddings)
    signature = (*_index_signature(settings, manifest), str(settings.chroma_dir))
    return _cached("dense", signature, lambda: build(make_embeddings(settings)))
