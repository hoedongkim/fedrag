"""Retrieval and answer runs over the evaluation set, cached on disk under data/eval_cache.

Retrieval (run_retrieval) asks each method for the top DEPTH (10) chunks of every question, with
the question's as_of exactly as POST /api/ask passes it, and times each call after an untimed
warm-up query (the cross-encoder loaded, Chroma opened). Every query is embedded before timing
(CachedQueryEmbeddings: one batched OpenAI request for the questions not embedded before, the
vectors kept on disk), so the latencies exclude the query-embedding round trip that dense, hybrid
and hybrid_rerank pay once per question in the API. With top_k 10 the hybrid and rerank candidate
pools are the ones top_k 5 gets (max(50, 4 * top_k)), so the first 5 hits are what Ask the Fed
shows. Each run overwrites its rows in retrieval.jsonl.

Answers (AnswerRunner) come from the production chain of app.rag.answer: its retrieval step
(top_k 5, as_of, the eligibility re-check), then its generation step and build_response, the
functions answer_question chains. They are cached in answers.jsonl, keyed by (question, as_of,
method, retriever config hash, top_k, answer prompt version, model), so a rerun makes no call.
Retrieval for every uncached answer runs first, so estimate() prices the exact prompts with
tiktoken before any LLM call; Usage (app.stance.llm) counts calls and tokens from each reply's
usage_metadata and refuses calls past max_calls.

The retriever configuration under test (RetrieverSetup) is an app.rag.config.RetrieverConfig:
"default" (the production retriever, app.rag.config.production_config: the preset
RETRIEVER_PRESET names, tuned since Phase 7), a preset name ("phase3", or one from
retriever_presets.json), or a JSON object of RetrieverConfig fields. It is passed to
app.rag.retrievers.get_retriever as its `config` argument. Cached results are keyed by
a hash of the config's behaviour (reranker resolved; name and description left out), the index
(chunks.jsonl digest) and the embedding model, so results from another index or reranker are
never reused and two presets that rank alike share them.
"""

import hashlib
import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar, get_args

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from app.core.config import Settings, get_settings
from app.evaluation.questions import EvalQuestion
from app.evaluation.retrieval_metrics import DEPTH
from app.rag import answer as rag_answer
from app.rag import prompts as rag_prompts
from app.rag.types import RetrievedChunk, Retriever
from app.schemas.ask import AskRequest
from app.schemas.common import RetrievalMethod
from app.stance.llm import (
    MESSAGE_OVERHEAD_TOKENS,
    CostEstimate,
    PredictionCache,
    Usage,
    cache_key,
    cost_usd,
    count_tokens,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")

METHODS: tuple[RetrievalMethod, ...] = get_args(RetrievalMethod)
CACHE_DIR = "eval_cache"  # under settings.data_dir; git-ignored
RETRIEVAL_CACHE = "retrieval.jsonl"
ANSWER_CACHE = "answers.jsonl"
EMBEDDING_CACHE = "query_embeddings.jsonl"
DEFAULT_CONFIG = "default"
ANSWER_TOP_K = 5  # what Ask the Fed uses by default
# Bump when the answer prompt's rendering changes outside the templates that answer_prompt_label
# hashes: app.rag.prompts.time_rule, format_passages or prompt_inputs (pinned by
# tests/evaluation/test_answer_rendering.py). Cached answers are keyed by the label, so a bump
# regenerates them; 0 leaves the Phase 7 label unchanged.
ANSWER_PROMPT_REVISION = 0
# Output tokens per answer that estimate() assumes (reasoning, billed as output, plus the text):
# about twice the Phase 3 and 4 smoke runs (2.5k for 12 answers; 990 for 6); the Phase 7
# validation calls used 174 and 58.
ANSWER_OUTPUT_TOKENS_PER_CALL = 400
USAGE_KIND = "answer"

RetrieverFactory = Callable[["RetrieverSetup", RetrievalMethod], Retriever]


def cache_dir(settings: Settings) -> Path:
    return settings.data_dir / CACHE_DIR


def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


# --- Retriever configuration ---------------------------------------------------------------------


@dataclass(frozen=True)
class RetrieverSetup:
    """A retriever configuration under evaluation."""

    name: str  # the config's name (the default is the production preset, e.g. "tuned")
    value: Any  # the app.rag.config.RetrieverConfig passed to get_retriever
    description: Any  # JSON-serializable form of value
    hash: str  # 12 hex digits over what changes rankings: the config's behaviour, the index


def describe_config(value: Any) -> Any:
    """A JSON-serializable description of a config object (pydantic model, dataclass, mapping,
    string; anything else by repr)."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return json.loads(json.dumps(asdict(value), default=str))
    if isinstance(value, Mapping):
        return json.loads(json.dumps(dict(value), default=str))
    return repr(value)


def lookup_config(name: str | None, settings: Settings | None = None) -> Any:
    """The RetrieverConfig of app.rag.config for `name`: None or "default" is the production
    config (production_config(settings): the preset RETRIEVER_PRESET names), a JSON object
    (inline, or a path ending in .json) is validated as a RetrieverConfig, anything else is a
    preset (get_preset; ValueError names the known presets). A JSON object without a name is
    named custom-<6 hex digits of its fields>. A file that cannot be read is a ValueError."""
    from app.rag import config as rag_config

    if name is None or name == DEFAULT_CONFIG:
        return rag_config.production_config(settings or get_settings())
    text = name.strip()
    if text.startswith("{") or text.endswith(".json"):
        if text.startswith("{"):
            raw = text
        else:
            try:
                raw = Path(text).read_text(encoding="utf-8")
            except OSError as exc:
                raise ValueError(f"--config {name}: cannot read the file ({exc.strerror})") from exc
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError(f"--config {name}: expected a JSON object of RetrieverConfig fields")
        digest = hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()
        return rag_config.RetrieverConfig.model_validate({"name": f"custom-{digest[:6]}", **data})
    return rag_config.get_preset(text)


def resolve_config(name: str | None, settings: Settings) -> RetrieverSetup:
    """The RetrieverSetup for `name` (None or "default": the production retriever)."""
    value = lookup_config(name, settings)
    label = str(
        getattr(value, "name", None) or (value if isinstance(value, str) else DEFAULT_CONFIG)
    )
    return RetrieverSetup(
        name=label,
        value=value,
        description=describe_config(value),
        hash=config_hash(value, settings),
    )


def config_hash(value: Any, settings: Settings) -> str:
    """Hash of what changes rankings: the config's behaviour with its reranker resolved against
    `settings` (RetrieverConfig.resolved; name and description left out, so two presets that
    rank alike share cached results), the index's chunks digest and the embedding model."""
    from app.rag.index import read_manifest

    resolved = value.resolved(settings) if hasattr(value, "resolved") else value
    behaviour = getattr(resolved, "behaviour", None)
    manifest = read_manifest(settings)
    payload = {
        "config": behaviour if behaviour is not None else describe_config(resolved),
        "chunks_sha256": manifest.chunks_sha256 if manifest else None,
        "embedding_model": settings.embedding_model,
        "reranker": [settings.reranker_model, settings.reranker_revision],
    }
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def make_retriever_factory(
    settings: Settings, embeddings: Embeddings | None = None
) -> RetrieverFactory:
    """(setup, method) -> the retriever from app.rag.retrievers.get_retriever, built once each;
    `embeddings` replaces OpenAI's for the query vectors (CachedQueryEmbeddings)."""
    from app.rag.retrievers import get_retriever

    built: dict[tuple[str, RetrievalMethod], Retriever] = {}

    def factory(setup: RetrieverSetup, method: RetrievalMethod) -> Retriever:
        key = (setup.hash, method)
        if key not in built:
            kwargs: dict[str, Any] = {}
            if embeddings is not None and method != "bm25":
                kwargs["embeddings"] = embeddings
            built[key] = get_retriever(method, settings, config=setup.value, **kwargs)
        return built[key]

    return factory


class CachedQueryEmbeddings(Embeddings):
    """Query vectors from a disk cache (data/eval_cache/query_embeddings.jsonl), embedding the
    missing ones in one batched request (prefetch). OpenAIEmbeddings.embed_query is
    embed_documents([text])[0], so a batched vector is the vector the API would use."""

    def __init__(
        self, model: str, cache: PredictionCache, embedder: Callable[[], Embeddings]
    ) -> None:
        self.model = model
        self.cache = cache
        self._embedder = embedder
        self._base: Embeddings | None = None
        self.requests = 0
        self.embedded = 0
        self.tokens = 0

    def key(self, text: str) -> str:
        return cache_key("query_embedding", self.model, text)

    def missing(self, texts: Sequence[str]) -> list[str]:
        return [t for t in dict.fromkeys(texts) if self.cache.get(self.key(t)) is None]

    def prefetch(self, texts: Sequence[str]) -> int:
        """Embed every text not cached yet, in one call; returns how many were embedded."""
        todo = self.missing(texts)
        if not todo:
            return 0
        if self._base is None:
            self._base = self._embedder()
        vectors = self._base.embed_documents(todo)
        self.requests += 1
        self.embedded += len(todo)
        self.tokens += sum(_embedding_tokens(t) for t in todo)
        for text, vector in zip(todo, vectors, strict=True):
            self.cache.put(self.key(text), {"vector": [float(x) for x in vector]})
        return len(todo)

    def embed_query(self, text: str) -> list[float]:
        hit = self.cache.get(self.key(text))
        if hit is None:
            self.prefetch([text])
            hit = self.cache.get(self.key(text))
        assert hit is not None
        return list(hit["vector"])

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


def _embedding_tokens(text: str) -> int:
    import tiktoken

    return len(tiktoken.get_encoding("cl100k_base").encode(text, disallowed_special=()))


def query_embeddings(settings: Settings) -> CachedQueryEmbeddings:
    from app.rag.index import make_embeddings

    cache = PredictionCache(cache_dir(settings) / EMBEDDING_CACHE)
    return CachedQueryEmbeddings(settings.embedding_model, cache, lambda: make_embeddings(settings))


# --- Retrieval ------------------------------------------------------------------------------------


def retrieval_key(question: EvalQuestion, method: RetrievalMethod, setup: RetrieverSetup) -> str:
    as_of = question.as_of.isoformat() if question.as_of else ""
    return cache_key("retrieval", question.question, as_of, method, setup.hash, str(DEPTH))


def hit_json(hit: RetrievedChunk) -> dict[str, Any]:
    chunk = hit.chunk
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "score": hit.score,
        "release_date": chunk.release_date.isoformat(),
    }


def run_retrieval(
    questions: Sequence[EvalQuestion],
    methods: Sequence[RetrievalMethod],
    setup: RetrieverSetup,
    retrievers: RetrieverFactory,
    cache: PredictionCache,
    *,
    depth: int = DEPTH,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[RetrievalMethod, list[dict[str, Any]]]:
    """Retrieve the top `depth` chunks for each question with each method, store each row in
    `cache` (replacing an earlier run's) and return the rows per method, in question order."""
    from app.rag.retrievers import preload

    rows: dict[RetrievalMethod, list[dict[str, Any]]] = {}
    for method in methods:
        retriever = retrievers(setup, method)
        preload(retriever)  # the cross-encoder of hybrid_rerank, untimed
        if questions:  # untimed warm-up: first Chroma query, first cross-encoder batch
            first = questions[0]
            retriever.retrieve(first.question, top_k=depth, as_of=first.as_of)
        rows[method] = []
        for q in questions:
            started = clock()
            hits = retriever.retrieve(q.question, top_k=depth, as_of=q.as_of)
            elapsed = (clock() - started) * 1000
            leaks = sum(q.as_of is not None and h.chunk.release_date > q.as_of for h in hits)
            if leaks:
                logger.error("%s %s: %d hits released after as_of %s", method, q.id, leaks, q.as_of)
            row = {
                "question_id": q.id,
                "fingerprint": q.fingerprint,
                "method": method,
                "config": setup.name,
                "config_hash": setup.hash,
                "depth": depth,
                "hits": [hit_json(h) for h in hits],
                "latency_ms": round(elapsed, 2),
                "leaks": leaks,
                "run_at": now_iso(),
            }
            cache.put(retrieval_key(q, method, setup), row)
            rows[method].append(row)
    return rows


# --- Answers --------------------------------------------------------------------------------------


def answer_prompt_label() -> str:
    """Version of the answer prompt: a hash of its templates and generation settings (the RAG
    prompt has no version number of its own), e.g. "answer.3fa2c1d0", plus
    ANSWER_PROMPT_REVISION once it is bumped for a change in how the prompt is rendered."""
    payload = json.dumps(
        [
            rag_prompts.SYSTEM,
            rag_prompts.HUMAN,
            rag_answer.REASONING_EFFORT,
            rag_answer.MAX_OUTPUT_TOKENS,
            *([ANSWER_PROMPT_REVISION] if ANSWER_PROMPT_REVISION else []),
        ]
    )
    return "answer." + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


@dataclass(frozen=True)
class AnswerItem:
    question: EvalQuestion
    method: RetrievalMethod
    setup: RetrieverSetup


@dataclass(frozen=True)
class PendingAnswer:
    """An uncached answer whose passages are retrieved: what generation needs, and its cost."""

    item: AnswerItem
    key: str
    request: AskRequest
    passages: list[RetrievedChunk]
    retrieval_ms: float
    input_tokens: int  # 0 when there are no passages (no LLM call)


class _UsageCapture(BaseCallbackHandler):
    """Collects the usage_metadata of the chat replies in one chain run."""

    def __init__(self) -> None:
        self.metadata: dict[str, Any] = {}

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        for generations in response.generations:
            for generation in generations:
                message = generation.message if isinstance(generation, ChatGeneration) else None
                if isinstance(message, AIMessage) and message.usage_metadata:
                    self.metadata = _add_usage(self.metadata, dict(message.usage_metadata))


def _add_usage(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    if not a:
        return dict(b)
    total = {
        name: int(a.get(name) or 0) + int(b.get(name) or 0)
        for name in ("input_tokens", "output_tokens")
    }
    for group, name in (
        ("input_token_details", "cache_read"),
        ("output_token_details", "reasoning"),
    ):
        value = int((a.get(group) or {}).get(name) or 0) + int((b.get(group) or {}).get(name) or 0)
        total[group] = {name: value}  # type: ignore[assignment]
    return total


def usage_row(metadata: Mapping[str, Any] | None) -> dict[str, int]:
    """Token counts of one reply's usage_metadata (missing counts as 0), as stored per record."""
    metadata = metadata or {}
    return {
        "input_tokens": int(metadata.get("input_tokens") or 0),
        "cached_input_tokens": int(
            (metadata.get("input_token_details") or {}).get("cache_read") or 0
        ),
        "output_tokens": int(metadata.get("output_tokens") or 0),
        "reasoning_tokens": int((metadata.get("output_token_details") or {}).get("reasoning") or 0),
    }


def answer_prompt_tokens(question: EvalQuestion, passages: Sequence[RetrievedChunk]) -> int:
    """Input tokens of the answer prompt for these passages (o200k_base)."""
    messages = rag_prompts.ANSWER_PROMPT.format_messages(
        **rag_prompts.prompt_inputs(question.question, question.as_of, passages)
    )
    return sum(count_tokens(str(m.content)) + MESSAGE_OVERHEAD_TOKENS for m in messages) + 3


class AnswerRunner:
    """Answers questions with the production chain and caches each answer (see the module
    docstring). `llm` replaces the OpenAI chat model (tests); it is created on the first uncached
    answer, so a fully cached run needs no API key."""

    def __init__(
        self,
        settings: Settings,
        retrievers: RetrieverFactory,
        *,
        llm: BaseChatModel | None = None,
        cache: PredictionCache | None = None,
        usage: Usage | None = None,
        top_k: int = ANSWER_TOP_K,
        max_concurrency: int = 1,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.settings = settings
        self.retrievers = retrievers
        self.cache = cache or PredictionCache(cache_dir(settings) / ANSWER_CACHE)
        self.usage = usage if usage is not None else Usage()
        self.top_k = top_k
        self.max_concurrency = max_concurrency
        self.prompt_version = answer_prompt_label()
        self._llm = llm
        self._clock = clock
        self._warm: set[tuple[str, str]] = set()  # (config hash, method) already warmed up

    @property
    def model(self) -> str:
        return self.settings.llm_model

    def key(self, item: AnswerItem) -> str:
        q = item.question
        return cache_key(
            "answer",
            q.question,
            q.as_of.isoformat() if q.as_of else "",
            item.method,
            item.setup.hash,
            str(self.top_k),
            self.prompt_version,
            self.model,
        )

    def cached(self, item: AnswerItem) -> dict[str, Any] | None:
        return self.cache.get(self.key(item))

    def prepare(self, items: Sequence[AnswerItem]) -> list[PendingAnswer]:
        """Retrieve the passages of every uncached item (no LLM call; dense methods need the
        query vectors, which CachedQueryEmbeddings serves)."""
        pending = []
        for item in items:
            key = self.key(item)
            if self.cache.get(key) is not None:
                continue
            q = item.question
            request = AskRequest(
                question=q.question, as_of=q.as_of, retrieval=item.method, top_k=self.top_k
            )
            retriever = self.retrievers(item.setup, item.method)
            self._warm_up(item, retriever, request)
            started = self._clock()
            passages = rag_answer.retrieval_step(retriever).invoke({"request": request})
            elapsed = (self._clock() - started) * 1000
            tokens = answer_prompt_tokens(q, passages) if passages else 0
            pending.append(PendingAnswer(item, key, request, passages, elapsed, tokens))
        return pending

    def _warm_up(self, item: AnswerItem, retriever: Retriever, request: AskRequest) -> None:
        """Before the first timed retrieval of a run: load the cross-encoder and send one untimed
        query, so model loading does not count as the first answer's latency."""
        from app.rag.retrievers import preload

        key = (item.setup.hash, item.method)
        if key not in self._warm:
            preload(retriever)
            retriever.retrieve(request.question, top_k=request.top_k, as_of=request.as_of)
            self._warm.add(key)

    def estimate(self, pending: Sequence[PendingAnswer], items: int) -> CostEstimate:
        """What generating `pending` would cost (of `items` asked for); no calls."""
        calls = sum(bool(p.passages) for p in pending)
        input_tokens = sum(p.input_tokens for p in pending)
        output_tokens = calls * ANSWER_OUTPUT_TOKENS_PER_CALL
        return CostEstimate(
            items=items,
            cached=items - len(pending),
            calls=calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            usd=cost_usd(self.model, input_tokens, 0, output_tokens),
        )

    def generate(self, pending: Sequence[PendingAnswer]) -> list[dict[str, Any]]:
        """Generate and cache each pending answer (max_concurrency at a time); the records."""
        if pending and any(p.passages for p in pending):
            self.usage.require(sum(bool(p.passages) for p in pending))
            self._chat_model()  # LLMNotConfiguredError before the first call
        return _map(self._generate, pending, self.max_concurrency)

    def _chat_model(self) -> BaseChatModel:
        if self._llm is None:
            self._llm = rag_answer.get_llm(self.settings)
        return self._llm

    def _generate(self, p: PendingAnswer) -> dict[str, Any]:
        capture = _UsageCapture()
        inputs = {"request": p.request, "passages": p.passages}
        if p.passages:
            self.usage.reserve()
        started = self._clock()
        generated = rag_answer.generation_step(self._chat_model()).invoke(
            inputs, config={"callbacks": [capture]}
        )
        elapsed = (self._clock() - started) * 1000
        if p.passages:
            self.usage.record(USAGE_KIND, 1, capture.metadata)  # type: ignore[arg-type]
        response = rag_answer.build_response(
            p.request, p.passages, generated, model_name=self.model, started=started
        )
        q = p.item.question
        record = {
            "question_id": q.id,
            "fingerprint": q.fingerprint,
            "method": p.item.method,
            "config": p.item.setup.name,
            "config_hash": p.item.setup.hash,
            "top_k": self.top_k,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "answer": response.answer,
            "generated": generated,
            "citations": [c.chunk_id for c in response.citations],
            "passages": [hit_json(hit) for hit in p.passages],
            "notes": response.notes,
            "llm_called": bool(p.passages),
            "retrieval_ms": round(p.retrieval_ms, 2),
            "generation_ms": round(elapsed, 2),
            "latency_ms": round(p.retrieval_ms + elapsed, 2),
            "usage": usage_row(capture.metadata),
            "created_at": now_iso(),
        }
        self.cache.put(p.key, record)
        return record


def answer_usd(record: Mapping[str, Any]) -> float:
    """Dollar cost of one cached answer (0 without an LLM call; NaN for an unpriced model)."""
    usage = record.get("usage") or {}
    return cost_usd(
        record["model"],
        int(usage.get("input_tokens") or 0),
        int(usage.get("cached_input_tokens") or 0),
        int(usage.get("output_tokens") or 0),
    )


def _map(function: Callable[[T], R], items: Sequence[T], concurrency: int) -> list[R]:
    """function over items, `concurrency` at a time, in order; the first error propagates."""
    if concurrency <= 1 or len(items) <= 1:
        return [function(item) for item in items]
    with ThreadPoolExecutor(max_workers=min(concurrency, len(items))) as pool:
        return list(pool.map(function, items))
