"""LLM reading of a statement comparison: what the changed wording signals, checked in code.

LLMAnalyzer (get_analyzer) is the Analyzer of app.compare.types. It sends one strict
structured-output request per pair of statements (prompts.comparison_messages: both dates, the
changed items only and the full target statement, read as of the target's release date) and
cleans the reply before it is stored or shown:

- change ids that are not changed items of this diff are removed, repeats too, and a key change
  left without ids is dropped; key changes past MAX_KEY_CHANGES are cut;
- text in double quotes (straight or “ ”) that is not verbatim in either statement (whitespace
  runs, case and apostrophe style aside, edge punctuation trimmed) loses its quotation marks, so a
  paraphrase is never shown as a quote;
- confidence, the model's own number and not calibrated, is clipped to [0, 1].

What cleaning removed is counted (AnalysisChecks) and stored in the run's usage under "checks".
A pair without changed items makes no call: identical statements are "unchanged" by construction.
Calls and tokens come from each reply's usage_metadata (app.stance.llm.Usage, whose max_calls is
the budget guard), and estimate() prices a set of pairs with tiktoken before any call.
"""

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from pydantic import SecretStr

from app.compare.prompts import (
    MAX_KEY_CHANGES,
    ComparisonJudgment,
    comparison_messages,
    prompt_label,
)
from app.compare.types import AnalysisRun, ComparisonInput
from app.core.config import Settings, get_settings
from app.rag.answer import LLMNotConfiguredError
from app.schemas.compare import ComparisonAnalysis, KeyChange
from app.stance.llm import (
    MESSAGE_OVERHEAD_TOKENS,
    CostEstimate,
    LLMReplyError,
    Usage,
    clip,
    cost_usd,
    count_tokens,
)
from app.stance.prompts import one_line

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# The analysis's own generation settings. They are part of what a cached analysis stands for:
# changing either requires bumping prompts.PROMPT_VERSION.
REASONING_EFFORT = "low"
MAX_OUTPUT_TOKENS = 4096
TIMEOUT_S = 90.0
MAX_ATTEMPTS = 2  # calls per pair: the first, plus one retry on an unparseable reply
USAGE_KIND = "comparison"  # the "kind" of each row in Usage.log
# Output tokens per call that estimate() assumes (reasoning, billed as output, plus the reply),
# rounded up from the validation calls (prompts compare-v1/v2, reasoning_effort="low"): 213-747
# output tokens per call, of which 4-118 reasoning (7 calls, 14,854 input tokens, $0.028).
OUTPUT_TOKENS_PER_CALL = 800

IDENTICAL_SUMMARY = (
    "The two statements have the same sentences, word for word (whitespace aside), so the "
    "wording signals no change in the policy stance."
)
QUOTED = re.compile(r'"([^"\n]+)"|“([^”\n]+)”')
_EDGE = " .,;:!?…"  # trimmed from a quote's ends when it does not match as written
_STRAIGHT = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})


@dataclass(frozen=True)
class AnalysisChecks:
    """What cleaning one reply removed; all zero for a clean reply or a pair without a call."""

    key_changes_returned: int = 0  # key changes in the reply, before cleaning
    key_changes_dropped: int = 0  # left without a valid change id
    key_changes_cut: int = 0  # past MAX_KEY_CHANGES
    invalid_ids: int = 0  # cited ids that are not changed items of this diff (repeats aside)
    quotes: int = 0  # double-quoted spans in the kept text
    quotes_unquoted: int = 0  # not verbatim in either statement: quotation marks removed
    quotes_outside_cited: int = 0  # verbatim, but not in the key change's cited items
    confidence_clipped: bool = False


def get_llm(settings: Settings) -> "ChatOpenAI":
    """The chat model for comparisons, with this module's REASONING_EFFORT and MAX_OUTPUT_TOKENS;
    cached per model and key. LLMNotConfiguredError without a key."""
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


def _plain(text: str) -> str:
    """One line with typographic quotes and apostrophes made straight: the statements are stored
    with straight ones, and a model asked to quote between “ and ” often writes “Committee’s”."""
    return one_line(text).translate(_STRAIGHT)


def is_verbatim(quote: str, text: str) -> bool:
    """Whether `quote` occurs in `text` between word boundaries, whitespace runs, case and the
    style of quotes and apostrophes aside; a quote that does not is tried again with edge
    punctuation and ellipses trimmed."""
    candidate = _plain(quote)
    for form in dict.fromkeys([candidate, candidate.strip(_EDGE)]):
        pattern = rf"(?<!\w){re.escape(form)}(?!\w)"
        if form and re.search(pattern, _plain(text), re.IGNORECASE):
            return True
    return False


def check_quotes(text: str, source: str, cited: str | None = None) -> tuple[str, int, int, int]:
    """`text` with the quotation marks removed around every double-quoted span that is not
    verbatim in `source`, plus the number of quotes, of quotes unquoted, and of verbatim quotes
    not found in `cited` (when given)."""
    counts = [0, 0, 0]

    def replace(match: re.Match[str]) -> str:
        span = match.group(1) or match.group(2)
        counts[0] += 1
        if not is_verbatim(span, source):
            counts[1] += 1
            return span
        if cited is not None and not is_verbatim(span, cited):
            counts[2] += 1
        return match.group(0)

    return QUOTED.sub(replace, text), *counts


def clean(
    judgment: ComparisonJudgment, comparison: ComparisonInput
) -> tuple[ComparisonAnalysis, AnalysisChecks]:
    """The ComparisonAnalysis for a model's judgment of `comparison`, and what cleaning removed
    (see the module docstring)."""
    by_id = {change.id: change for change in comparison.changed}
    source = f"{comparison.base.text}\n{comparison.target.text}"
    kept, invalid, dropped = [], 0, 0
    for item in judgment.key_changes:
        invalid += len({i for i in item.change_ids if i not in by_id})
        ids = sorted({i for i in item.change_ids if i in by_id})
        if ids:
            kept.append((ids, item))
        else:
            dropped += 1
    cut = max(len(kept) - MAX_KEY_CHANGES, 0)
    quotes = unquoted = outside = 0
    key_changes = []
    for ids, item in kept[:MAX_KEY_CHANGES]:
        cited = "\n".join(
            text for i in ids for text in (by_id[i].base, by_id[i].target) if text is not None
        )
        fields = {}
        for name in ("what_changed", "interpretation"):
            value, found, removed, elsewhere = check_quotes(
                one_line(getattr(item, name)), source, cited
            )
            fields[name] = value
            quotes, unquoted, outside = quotes + found, unquoted + removed, outside + elsewhere
        key_changes.append(KeyChange(change_ids=ids, direction=item.direction, **fields))
    summary, found, removed, _ = check_quotes(one_line(judgment.summary), source)
    confidence = clip(judgment.confidence, 0.0, 1.0)
    analysis = ComparisonAnalysis(
        summary=summary,
        key_changes=key_changes,
        stance_shift=judgment.stance_shift,
        confidence=confidence,
    )
    checks = AnalysisChecks(
        key_changes_returned=len(judgment.key_changes),
        key_changes_dropped=dropped,
        key_changes_cut=cut,
        invalid_ids=invalid,
        quotes=quotes + found,
        quotes_unquoted=unquoted + removed,
        quotes_outside_cited=outside,
        confidence_clipped=confidence != judgment.confidence,
    )
    if dropped or invalid or cut or checks.quotes_unquoted:
        logger.warning(
            "%s -> %s: cleaned the analysis (%s)",
            comparison.base.doc_id,
            comparison.target.doc_id,
            json.dumps(asdict(checks)),
        )
    return analysis, checks


def identical_analysis() -> ComparisonAnalysis:
    """The analysis of a pair without changed items, written without a call. Its confidence is 1
    because "unchanged" follows from the text, not from a model."""
    return ComparisonAnalysis(
        summary=IDENTICAL_SUMMARY, key_changes=[], stance_shift="unchanged", confidence=1.0
    )


def prompt_tokens(messages: Sequence[BaseMessage]) -> int:
    """Input tokens of one request: the messages plus the output schema it sends."""
    schema = json.dumps(ComparisonJudgment.model_json_schema())
    content = sum(count_tokens(str(m.content)) + MESSAGE_OVERHEAD_TOKENS for m in messages)
    return content + 3 + count_tokens(schema)


class LLMAnalyzer:
    """Writes the analysis of one comparison with one LLM call (two after an unparseable reply).

    `usage` (or a new one with `max_calls`) counts every call and token across the analyzer's
    runs and refuses calls past its max_calls; each AnalysisRun's usage covers that run alone.
    The chat model is created on the first call, so building an analyzer (for its model and
    prompt_version, which the cache key needs) requires no API key.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        llm: BaseChatModel | None = None,
        usage: Usage | None = None,
        max_calls: int | None = None,
    ) -> None:
        if usage is not None and max_calls is not None:
            raise ValueError("pass max_calls or a shared Usage (with its own max_calls), not both")
        self.settings = settings or get_settings()
        self.usage = usage if usage is not None else Usage(max_calls=max_calls)
        self._llm = llm
        self._structured: Runnable[Any, dict[str, Any]] | None = None
        self._prompt_version = prompt_label()

    @property
    def model(self) -> str:
        return self.settings.llm_model

    @property
    def prompt_version(self) -> str:
        return self._prompt_version

    def __call__(self, comparison: ComparisonInput) -> AnalysisRun:
        """The cleaned analysis of `comparison` and its usage (with "checks"). LLMReplyError
        after MAX_ATTEMPTS unparseable replies; CallLimitError past max_calls."""
        run = Usage()
        if not comparison.changed:
            return AnalysisRun(identical_analysis(), self._usage_json(run, AnalysisChecks()))
        messages = comparison_messages(comparison)
        for attempt in range(1, MAX_ATTEMPTS + 1):
            judgment = self._call(messages, run)
            if judgment is not None:
                analysis, checks = clean(judgment, comparison)
                return AnalysisRun(analysis, self._usage_json(run, checks))
            logger.warning("Comparison reply %d/%d unusable", attempt, MAX_ATTEMPTS)
        raise LLMReplyError(f"{MAX_ATTEMPTS} replies in a row could not be parsed")

    def estimate(self, comparisons: Sequence[ComparisonInput]) -> CostEstimate:
        """What analysing `comparisons` would cost, from the prompts it would send; no calls.
        Pairs without changed items need no call and count as cached; the caller leaves out the
        pairs already stored."""
        pending = [c for c in comparisons if c.changed]
        input_tokens = sum(prompt_tokens(comparison_messages(c)) for c in pending)
        output_tokens = len(pending) * OUTPUT_TOKENS_PER_CALL
        return CostEstimate(
            items=len(comparisons),
            cached=len(comparisons) - len(pending),
            calls=len(pending),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            usd=cost_usd(self.model, input_tokens, 0, output_tokens),
        )

    def _call(self, messages: list[BaseMessage], run: Usage) -> ComparisonJudgment | None:
        """One request, counted in the analyzer's usage and the run's; the parsed reply, or None
        when it could not be parsed."""
        structured = self._structured_llm()  # LLMNotConfiguredError before any call is counted
        self.usage.reserve()
        run.reserve()
        reply = structured.invoke(messages)
        metadata = getattr(reply.get("raw"), "usage_metadata", None)
        for usage in (self.usage, run):
            usage.record(USAGE_KIND, 1, metadata)  # type: ignore[arg-type]
        parsed = reply.get("parsed")
        if reply.get("parsing_error") is not None or not isinstance(parsed, ComparisonJudgment):
            logger.warning("Unusable comparison reply: %s", reply.get("parsing_error"))
            return None
        return parsed

    def _structured_llm(self) -> Runnable[Any, dict[str, Any]]:
        if self._structured is None:
            llm = self._llm if self._llm is not None else get_llm(self.settings)
            self._structured = llm.with_structured_output(
                ComparisonJudgment, method="json_schema", strict=True, include_raw=True
            )
        return self._structured

    def _usage_json(self, run: Usage, checks: AnalysisChecks) -> dict[str, Any]:
        """The run's usage as stored next to the analysis: calls, tokens, USD and the checks."""
        return {**run.summary(self.model), "checks": asdict(checks)}


def get_analyzer(settings: Settings | None = None, **kwargs: Any) -> LLMAnalyzer:
    """The Analyzer the comparison service uses; keyword arguments go to LLMAnalyzer."""
    return LLMAnalyzer(settings, **kwargs)
