"""LLM stance classifiers (zero-shot and few-shot): sentence batches and whole statements.

LLMSentenceClassifier labels sentences BATCH_SIZE to a call through a strict JSON schema, checks
that the reply labels every number exactly once and retries the batch when it does not.
LLMStatementClassifier judges one whole statement: stance, score, confidence, key factors and
evidence, keeping only evidence that is verbatim in the statement. Both cache every prediction on
disk (data/stance/cache), keyed by method, model, prompt fingerprint, example set and input text,
so a rerun makes no calls; both count calls and tokens from each reply's usage_metadata and refuse
to go past max_calls. estimate() prices a run with tiktoken before any call is made.

Few-shot examples are a fixed stratified sample of one TDW seed's train split, never test
sentences and never gold-labelled statements. A cached sentence label is reused whatever batch it
was first labelled in, although the batch around a sentence can sway its label slightly.
"""

import hashlib
import json
import logging
import math
import random
import re
import threading
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, TypeVar

import tiktoken
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, SecretStr, ValidationError

from app.core.config import Settings, get_settings
from app.rag.answer import LLMNotConfiguredError
from app.schemas.common import Stance, StanceClassifier
from app.stance import tdw
from app.stance.prompts import (
    OUTPUT_SCHEMAS,
    Kind,
    SentenceLabels,
    StatementJudgment,
    TaskDefinition,
    example_set_digest,
    fingerprint,
    load_task_definition,
    one_line,
    prompt_label,
    sentence_messages,
    statement_messages,
)
from app.stance.tdw import LabeledSentence
from app.stance.types import STANCES, SentencePrediction, StatementPrediction

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

Mode = Literal["zero_shot", "few_shot"]
T = TypeVar("T")
R = TypeVar("R")

BATCH_SIZE = 20
MAX_ATTEMPTS = 3  # calls per batch or statement: the first, plus two retries on a malformed reply
FEW_SHOT_PER_CLASS = 4
EXAMPLE_WORDS = (8, 40)  # few-shot examples have this many words (inclusive): short, cheap shots
MAX_KEY_FACTORS = 3
MAX_EVIDENCE = 2
MIN_EVIDENCE_WORDS = 3  # e.g. "Inflation remains elevated."; shorter quotes are fragments
CACHE_FILES: dict[Kind, str] = {
    "sentence": "llm_sentences.jsonl",
    "statement": "llm_statements.jsonl",
}
ENCODING = "o200k_base"  # the GPT-5.x tokenizer
MESSAGE_OVERHEAD_TOKENS = 4  # per chat message, plus 3 per request, in OpenAI's token accounting
# The stance classifiers' own generation settings, not Ask the Fed's (app.rag.answer). They are
# part of what the cache key stands for: changing either requires bumping prompts.PROMPT_VERSION.
REASONING_EFFORT = "low"
MAX_OUTPUT_TOKENS = 4096
TIMEOUT_S = 60.0

# Output tokens per call that estimate() assumes: reasoning (billed as output) plus the reply,
# rounded up from the validation calls (reasoning_effort="low", prompt stance-v1/v2): 20-sentence
# batches used 572-722 output tokens (384-521 of them reasoning), statements 134-164 (13-28).
SENTENCE_OUTPUT_TOKENS_PER_CALL = 600
SENTENCE_OUTPUT_TOKENS_PER_SENTENCE = 15
STATEMENT_OUTPUT_TOKENS_PER_CALL = 400


@dataclass(frozen=True)
class Price:
    """USD per million tokens of the standard (not Batch) API; reasoning is billed as output."""

    input: float
    cached_input: float
    output: float


# From third-party price lists as of October 2026 (OpenAI's page could not be checked here);
# verify at https://developers.openai.com/api/docs/pricing before relying on the dollar figures.
PRICES: dict[str, Price] = {"gpt-5.4-mini": Price(input=0.75, cached_input=0.075, output=4.50)}


class CallLimitError(RuntimeError):
    """Raised before a call that would take the calls past max_calls."""


class LLMReplyError(RuntimeError):
    """Raised when MAX_ATTEMPTS replies in a row could not be used."""


def price_for(model: str) -> Price | None:
    """The price of `model` or of its base model (the name without a -YYYY-MM-DD snapshot)."""
    return PRICES.get(model) or PRICES.get(re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model))


def cost_usd(model: str, input_tokens: int, cached_input_tokens: int, output_tokens: int) -> float:
    """Dollar cost of the tokens; NaN for a model without a known price."""
    price = price_for(model)
    if price is None:
        return math.nan
    uncached = input_tokens - cached_input_tokens
    total = uncached * price.input + cached_input_tokens * price.cached_input
    return (total + output_tokens * price.output) / 1_000_000


@dataclass
class Usage:
    """Calls and tokens spent by one or more classifiers (share one to budget a whole run), read
    from each reply's usage_metadata. max_calls (None: no limit) is a hard ceiling on calls,
    retries included; a call is counted when it is made, even if it fails. Thread-safe."""

    max_calls: int | None = None
    calls: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    log: list[dict[str, Any]] = field(default_factory=list)  # one row per reply
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    @property
    def remaining(self) -> int | None:
        return None if self.max_calls is None else max(self.max_calls - self.calls, 0)

    def require(self, calls: int) -> None:
        """Refuse up front a run that needs more calls than remain."""
        remaining = self.remaining
        if remaining is not None and calls > remaining:
            raise CallLimitError(
                f"refused: {calls} calls needed but only {remaining} of max_calls="
                f"{self.max_calls} remain; nothing was called"
            )

    def reserve(self) -> None:
        """Count one call about to be made; CallLimitError if it would pass max_calls."""
        with self._lock:
            if self.max_calls is not None and self.calls >= self.max_calls:
                raise CallLimitError(f"max_calls={self.max_calls} reached")
            self.calls += 1

    def record(self, kind: Kind, items: int, metadata: Mapping[str, Any] | None) -> None:
        """Add one reply's usage_metadata (missing counts as zero tokens)."""
        metadata = metadata or {}
        row = {
            "kind": kind,
            "items": items,
            "input_tokens": int(metadata.get("input_tokens") or 0),
            "cached_input_tokens": int(
                (metadata.get("input_token_details") or {}).get("cache_read") or 0
            ),
            "output_tokens": int(metadata.get("output_tokens") or 0),
            "reasoning_tokens": int(
                (metadata.get("output_token_details") or {}).get("reasoning") or 0
            ),
        }
        with self._lock:
            self.log.append(row)
            self.input_tokens += row["input_tokens"]
            self.cached_input_tokens += row["cached_input_tokens"]
            self.output_tokens += row["output_tokens"]
            self.reasoning_tokens += row["reasoning_tokens"]

    def usd(self, model: str) -> float:
        return cost_usd(model, self.input_tokens, self.cached_input_tokens, self.output_tokens)

    def summary(self, model: str) -> dict[str, Any]:
        """Totals for a results file: calls, tokens and the dollar cost (null if unknown)."""
        usd = self.usd(model)
        return {
            "model": model,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "usd": None if math.isnan(usd) else round(usd, 4),
        }


@dataclass(frozen=True)
class CostEstimate:
    """What a run would cost before it is made: uncached items only, no retries, no prompt-cache
    discount on input; output tokens are an assumption (see SENTENCE_OUTPUT_TOKENS_PER_CALL)."""

    items: int  # distinct inputs asked for
    cached: int  # already in the cache: no call
    calls: int
    input_tokens: int
    output_tokens: int
    usd: float  # NaN for a model without a known price

    def __str__(self) -> str:
        cost = "unknown price" if math.isnan(self.usd) else f"about ${self.usd:.3f}"
        return (
            f"{self.calls} calls for {self.items - self.cached} uncached of {self.items} items "
            f"({self.input_tokens:,} input + ~{self.output_tokens:,} output tokens, {cost})"
        )


class PredictionCache:
    """Append-only JSONL file mapping a cache key to a JSON object, read once and appended to on
    every put, so an interrupted run keeps each prediction it finished. Unreadable lines (a write
    torn by a crash) are skipped. Thread-safe."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._entries: dict[str, dict[str, Any]] | None = None
        self._torn_tail = False
        self._lock = threading.Lock()

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            return self._load().get(key)

    def put(self, key: str, value: dict[str, Any]) -> None:
        line = json.dumps({"key": key, "value": value}, ensure_ascii=False)
        with self._lock:
            self._load()[key] = value
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as file:
                file.write(("\n" if self._torn_tail else "") + line + "\n")
            self._torn_tail = False

    def __len__(self) -> int:
        with self._lock:
            return len(self._load())

    def _load(self) -> dict[str, dict[str, Any]]:
        if self._entries is None:
            self._entries = {}
            text = self.path.read_text(encoding="utf-8") if self.path.is_file() else ""
            self._torn_tail = bool(text) and not text.endswith("\n")
            for number, line in enumerate(text.splitlines(), start=1):
                try:
                    row = json.loads(line)
                    self._entries[row["key"]] = dict(row["value"])
                except (ValueError, KeyError, TypeError):
                    logger.warning("%s:%d: skipped an unreadable line", self.path.name, number)
        return self._entries


def cache_key(*parts: str) -> str:
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode("utf-8")).hexdigest()


def get_llm(settings: Settings) -> "ChatOpenAI":
    """The chat model for stance labels, with this module's REASONING_EFFORT and
    MAX_OUTPUT_TOKENS; cached per model and key. LLMNotConfiguredError without a key."""
    if settings.openai_api_key is None:
        raise LLMNotConfiguredError
    return _chat_model(settings.llm_model, settings.openai_api_key)


@lru_cache(maxsize=4)
def _chat_model(model: str, api_key: SecretStr) -> "ChatOpenAI":
    from langchain_openai import ChatOpenAI  # here, so importing this module loads no OpenAI client

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        reasoning_effort=REASONING_EFFORT,
        max_tokens=MAX_OUTPUT_TOKENS,
        timeout=TIMEOUT_S,
        max_retries=2,
    )


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(ENCODING)


def count_tokens(text: str) -> int:
    """Tokens in `text` under o200k_base (special-token text counts as plain text)."""
    return len(_encoding().encode(text, disallowed_special=()))


def prompt_tokens(messages: Sequence[BaseMessage], kind: Kind) -> int:
    """Input tokens of one request: the messages plus the output schema it sends."""
    schema = json.dumps(OUTPUT_SCHEMAS[kind].model_json_schema())
    content = sum(count_tokens(str(m.content)) + MESSAGE_OVERHEAD_TOKENS for m in messages)
    return content + 3 + count_tokens(schema)


def few_shot_examples(
    train: Sequence[LabeledSentence],
    *,
    seed: int,
    per_class: int = FEW_SHOT_PER_CLASS,
    words: tuple[int, int] = EXAMPLE_WORDS,
) -> tuple[LabeledSentence, ...]:
    """A fixed stratified sample of a TRAIN split: per_class sentences of each stance, drawn with
    random.Random(seed) among distinct texts of `words` words that carry one label throughout
    `train`, then shuffled into a fixed order so no label is always last."""
    labels: dict[str, set[Stance]] = defaultdict(set)
    for sentence in train:
        labels[sentence.sentence].add(sentence.stance)
    rng = random.Random(seed)
    chosen: list[LabeledSentence] = []
    for stance in STANCES:
        pool = {
            s.sentence: s
            for s in train
            if labels[s.sentence] == {stance} and words[0] <= len(s.sentence.split()) <= words[1]
        }
        if len(pool) < per_class:
            raise ValueError(f"only {len(pool)} {stance} candidates for {per_class} examples")
        chosen += rng.sample(sorted(pool.values(), key=lambda s: s.sentence), per_class)
    rng.shuffle(chosen)
    return tuple(chosen)


def aligned_labels(reply: SentenceLabels | None, count: int) -> list[Stance] | None:
    """The labels for sentences 1..count in order, or None unless the reply has each id exactly
    once and no other: a missing, repeated or extra id means the labels may be misaligned."""
    if reply is None:
        return None
    by_id = {label.id: label.stance for label in reply.labels}
    if len(reply.labels) != count or sorted(by_id) != list(range(1, count + 1)):
        return None
    return [by_id[number] for number in range(1, count + 1)]


def verbatim_evidence(candidates: Sequence[str], text: str) -> tuple[str, ...]:
    """The candidates that occur verbatim in `text` between word boundaries (whitespace runs
    aside; quotes around a candidate removed) and have at least MIN_EVIDENCE_WORDS words,
    deduplicated, in order."""
    haystack = one_line(text)
    kept: list[str] = []
    for candidate in candidates:
        quote = one_line(candidate)
        if not _occurs(quote, haystack):
            quote = quote.strip("\"'“”‘’")
        if len(quote.split()) >= MIN_EVIDENCE_WORDS and _occurs(quote, haystack):
            kept.append(quote)
    return tuple(dict.fromkeys(kept))


def _occurs(quote: str, text: str) -> bool:
    """Whether `quote` is in `text` and starts and ends at word boundaries (not "ation rose")."""
    return bool(quote) and re.search(rf"(?<!\w){re.escape(quote)}(?!\w)", text) is not None


def clip(value: float, low: float, high: float) -> float:
    """`value` clamped to [low, high]; NaN becomes the midpoint."""
    if math.isnan(value):
        return (low + high) / 2
    return min(max(value, low), high)


class StatementText(Protocol):
    """A statement to classify, such as app.ingest.records.Document."""

    @property
    def doc_id(self) -> str: ...
    @property
    def meeting_date(self) -> date: ...
    @property
    def text(self) -> str: ...


@dataclass(frozen=True)
class Statement:
    """A StatementText built from its fields."""

    doc_id: str
    meeting_date: date
    text: str


class _LLMClassifier:
    """What both classifiers share: the model, the prompt identity, the cache and the budget."""

    kind: ClassVar[Kind]

    def __init__(
        self,
        mode: Mode,
        examples: Sequence[LabeledSentence],
        *,
        llm: BaseChatModel | None,
        settings: Settings,
        cache_dir: Path | None,
        usage: Usage | None,
        max_calls: int | None,
        max_concurrency: int,
        task: TaskDefinition | None,
    ) -> None:
        if mode not in ("zero_shot", "few_shot"):
            raise ValueError(f"unknown mode {mode!r}")
        if usage is not None and max_calls is not None:
            raise ValueError("pass max_calls or a shared Usage (with its own max_calls), not both")
        self.mode = mode
        self.name: StanceClassifier = "llm_zero_shot" if mode == "zero_shot" else "llm_few_shot"
        self.settings = settings
        self.model = settings.llm_model
        self.task = task or load_task_definition()
        self.examples = tuple(examples)
        self.usage = usage if usage is not None else Usage(max_calls=max_calls)
        self.max_concurrency = max_concurrency
        directory = cache_dir or settings.data_dir / "stance" / "cache"
        self.cache = PredictionCache(directory / CACHE_FILES[self.kind])
        self.prompt_fingerprint = fingerprint(self.kind, self.task)
        self.examples_digest = example_set_digest(self.examples)
        parts = [self.model, prompt_label(self.kind, self.task)]
        if self.examples:
            parts.append(f"examples.{self.examples_digest[:8]}")
        self.model_version = "+".join(parts)
        self._llm = llm
        self._structured: Runnable[Any, dict[str, Any]] | None = None

    def key(self, text: str) -> str:
        """Cache key of one input under this classifier's method, model, prompt and examples."""
        return cache_key(
            self.kind, self.mode, self.model, self.prompt_fingerprint, self.examples_digest, text
        )

    def _call(self, messages: list[BaseMessage], items: int) -> BaseModel | None:
        """One request; the parsed reply, or None when it could not be parsed."""
        self.usage.reserve()
        reply = self._structured_llm().invoke(messages)
        self.usage.record(self.kind, items, getattr(reply.get("raw"), "usage_metadata", None))
        if reply.get("parsing_error") is not None or reply.get("parsed") is None:
            logger.warning("Unusable %s reply: %s", self.kind, reply.get("parsing_error"))
            return None
        return reply["parsed"]

    def _structured_llm(self) -> Runnable[Any, dict[str, Any]]:
        """The chat model with the strict output schema, created on the first uncached input (so a
        fully cached run needs no API key)."""
        if self._structured is None:
            llm = self._llm if self._llm is not None else get_llm(self.settings)
            self._structured = llm.with_structured_output(
                OUTPUT_SCHEMAS[self.kind], method="json_schema", strict=True, include_raw=True
            )
        return self._structured

    def _map(self, function: Callable[[T], R], items: Sequence[T]) -> list[R]:
        """function over items, max_concurrency calls at a time; the first error propagates."""
        if self.max_concurrency <= 1 or len(items) <= 1:
            return [function(item) for item in items]
        with ThreadPoolExecutor(max_workers=min(self.max_concurrency, len(items))) as pool:
            return list(pool.map(function, items))


class LLMSentenceClassifier(_LLMClassifier):
    """Labels sentences with the LLM, BATCH_SIZE numbered sentences per call (SentenceClassifier).

    few_shot needs `seed`: its examples are few_shot_examples() of that seed's TDW train split
    (`train` replaces loading it). Predictions carry a label only, no probabilities.
    """

    kind = "sentence"

    def __init__(
        self,
        mode: Mode,
        seed: int | None = None,
        *,
        train: Sequence[LabeledSentence] | None = None,
        batch_size: int = BATCH_SIZE,
        llm: BaseChatModel | None = None,
        settings: Settings | None = None,
        cache_dir: Path | None = None,
        usage: Usage | None = None,
        max_calls: int | None = None,
        max_concurrency: int = 1,
        task: TaskDefinition | None = None,
    ) -> None:
        settings = settings or get_settings()
        examples: tuple[LabeledSentence, ...] = ()
        if mode == "few_shot":
            if seed is None:
                raise ValueError("few_shot needs the seed whose train split supplies examples")
            split = train if train is not None else tdw.load("train", seed, settings)
            examples = few_shot_examples(split, seed=seed)
        super().__init__(
            mode,
            examples,
            llm=llm,
            settings=settings,
            cache_dir=cache_dir,
            usage=usage,
            max_calls=max_calls,
            max_concurrency=max_concurrency,
            task=task,
        )
        self.seed = seed
        self.batch_size = batch_size

    def predict_sentences(self, sentences: Sequence[str]) -> list[SentencePrediction]:
        """One prediction per sentence, in order; cached sentences make no call."""
        pending = [s for s in dict.fromkeys(sentences) if self._cached(s) is None]
        batches = [
            pending[i : i + self.batch_size] for i in range(0, len(pending), self.batch_size)
        ]
        self.usage.require(len(batches))
        self._map(self._label_batch, batches)
        return [SentencePrediction(stance=self._cached_or_fail(s)) for s in sentences]

    def estimate(self, sentences: Sequence[str]) -> CostEstimate:
        """The cost of predict_sentences(sentences) from the prompts it would send; no calls."""
        unique = list(dict.fromkeys(sentences))
        pending = [s for s in unique if self._cached(s) is None]
        batches = [
            pending[i : i + self.batch_size] for i in range(0, len(pending), self.batch_size)
        ]
        messages = (sentence_messages(batch, self.task, self.examples) for batch in batches)
        input_tokens = sum(prompt_tokens(m, self.kind) for m in messages)
        output_tokens = len(batches) * SENTENCE_OUTPUT_TOKENS_PER_CALL
        output_tokens += len(pending) * SENTENCE_OUTPUT_TOKENS_PER_SENTENCE
        return CostEstimate(
            items=len(unique),
            cached=len(unique) - len(pending),
            calls=len(batches),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            usd=cost_usd(self.model, input_tokens, 0, output_tokens),
        )

    def _label_batch(self, batch: list[str]) -> None:
        messages = sentence_messages(batch, self.task, self.examples)
        for attempt in range(1, MAX_ATTEMPTS + 1):
            reply = self._call(messages, len(batch))
            stances = aligned_labels(
                reply if isinstance(reply, SentenceLabels) else None, len(batch)
            )
            if stances is not None:
                for sentence, stance in zip(batch, stances, strict=True):
                    self.cache.put(self.key(sentence), {"stance": stance})
                return
            logger.warning(
                "Reply %d/%d did not label sentences 1-%d exactly once; retrying",
                attempt,
                MAX_ATTEMPTS,
                len(batch),
            )
        raise LLMReplyError(
            f"{MAX_ATTEMPTS} replies in a row did not label a {len(batch)}-sentence batch"
        )

    def _cached(self, sentence: str) -> Stance | None:
        hit = self.cache.get(self.key(sentence))
        stance = hit.get("stance") if hit else None
        return stance if stance in STANCES else None

    def _cached_or_fail(self, sentence: str) -> Stance:
        stance = self._cached(sentence)
        if stance is None:  # unreachable unless the cache file is changed during the run
            raise LLMReplyError("a labelled sentence is missing from the cache")
        return stance


def llm_factory(
    mode: Mode,
    settings: Settings | None = None,
    *,
    usage: Usage | None = None,
    max_concurrency: int = 1,
) -> Callable[[int], LLMSentenceClassifier]:
    """seed -> an LLM sentence classifier whose few-shot examples come from that seed's train
    split; pass one Usage to budget and count the calls of every seed together."""
    return lambda seed: LLMSentenceClassifier(
        mode, seed, settings=settings, usage=usage, max_concurrency=max_concurrency
    )


class LLMStatementClassifier(_LLMClassifier):
    """Judges whole statements with the LLM, one call per statement.

    few_shot uses the few_shot_examples() of TDW seed `example_seed`'s train split (`train`
    replaces loading it): labelled sentences that illustrate the labels, never gold statements.
    score and confidence are the model's own numbers clipped to [-1, 1] and [0, 1]: not
    calibrated. Evidence the statement does not contain verbatim is dropped.
    """

    kind = "statement"

    def __init__(
        self,
        mode: Mode,
        *,
        example_seed: int = tdw.SEEDS[0],
        train: Sequence[LabeledSentence] | None = None,
        llm: BaseChatModel | None = None,
        settings: Settings | None = None,
        cache_dir: Path | None = None,
        usage: Usage | None = None,
        max_calls: int | None = None,
        max_concurrency: int = 1,
        task: TaskDefinition | None = None,
    ) -> None:
        settings = settings or get_settings()
        examples: tuple[LabeledSentence, ...] = ()
        if mode == "few_shot":
            split = train if train is not None else tdw.load("train", example_seed, settings)
            examples = few_shot_examples(split, seed=example_seed)
        super().__init__(
            mode,
            examples,
            llm=llm,
            settings=settings,
            cache_dir=cache_dir,
            usage=usage,
            max_calls=max_calls,
            max_concurrency=max_concurrency,
            task=task,
        )
        self.example_seed = example_seed if mode == "few_shot" else None

    def predict_statement(self, doc_id: str, meeting_date: date, text: str) -> StatementPrediction:
        return self.predict_statements([Statement(doc_id, meeting_date, text)])[0]

    def predict_statements(self, statements: Sequence[StatementText]) -> list[StatementPrediction]:
        """One prediction per statement, in order; cached statements make no call."""
        pending = [t for t in dict.fromkeys(s.text for s in statements) if self._cached(t) is None]
        self.usage.require(len(pending))
        self._map(self._judge, pending)
        return [self._prediction(statement) for statement in statements]

    def estimate(self, statements: Sequence[StatementText]) -> CostEstimate:
        """The cost of predict_statements(statements) from the prompts it would send; no calls."""
        unique = list(dict.fromkeys(s.text for s in statements))
        pending = [text for text in unique if self._cached(text) is None]
        input_tokens = sum(
            prompt_tokens(statement_messages(text, self.task, self.examples), self.kind)
            for text in pending
        )
        output_tokens = len(pending) * STATEMENT_OUTPUT_TOKENS_PER_CALL
        return CostEstimate(
            items=len(unique),
            cached=len(unique) - len(pending),
            calls=len(pending),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            usd=cost_usd(self.model, input_tokens, 0, output_tokens),
        )

    def _judge(self, text: str) -> None:
        messages = statement_messages(text, self.task, self.examples)
        for attempt in range(1, MAX_ATTEMPTS + 1):
            reply = self._call(messages, 1)
            if isinstance(reply, StatementJudgment):
                self.cache.put(self.key(text), reply.model_dump())
                return
            logger.warning("Statement reply %d/%d unusable; retrying", attempt, MAX_ATTEMPTS)
        raise LLMReplyError(f"{MAX_ATTEMPTS} replies in a row could not be parsed")

    def _cached(self, text: str) -> StatementJudgment | None:
        hit = self.cache.get(self.key(text))
        if hit is None:
            return None
        try:
            return StatementJudgment.model_validate(hit)
        except ValidationError:
            logger.warning("Ignoring an invalid cached statement judgment")
            return None

    def _prediction(self, statement: StatementText) -> StatementPrediction:
        judgment = self._cached(statement.text)
        if judgment is None:  # unreachable unless the cache file is changed during the run
            raise LLMReplyError(f"{statement.doc_id}: judgment missing from the cache")
        return to_prediction(
            judgment, statement, classifier=self.name, model_version=self.model_version
        )


def to_prediction(
    judgment: StatementJudgment,
    statement: StatementText,
    *,
    classifier: StanceClassifier,
    model_version: str,
) -> StatementPrediction:
    """The StatementPrediction for a model's judgment: score and confidence clipped, at most
    MAX_KEY_FACTORS distinct key factors, at most MAX_EVIDENCE verbatim evidence sentences."""
    evidence = verbatim_evidence(judgment.evidence, statement.text)
    if len(evidence) < len(judgment.evidence):
        logger.warning(
            "%s: dropped %d evidence quotes that are not verbatim sentences of the statement",
            statement.doc_id,
            len(judgment.evidence) - len(evidence),
        )
    score = clip(judgment.score, -1.0, 1.0)
    expected_sign = {"hawkish": 1, "dovish": -1, "neutral": 0}[judgment.stance]
    if expected_sign and score * expected_sign < 0:
        logger.warning("%s: %s with score %.2f", statement.doc_id, judgment.stance, score)
    factors = (one_line(factor) for factor in judgment.key_factors)
    return StatementPrediction(
        doc_id=statement.doc_id,
        meeting_date=statement.meeting_date,
        classifier=classifier,
        stance=judgment.stance,
        score=score,
        confidence=clip(judgment.confidence, 0.0, 1.0),
        key_factors=tuple(dict.fromkeys(f for f in factors if f))[:MAX_KEY_FACTORS],
        evidence=evidence[:MAX_EVIDENCE],
        model_version=model_version,
    )
