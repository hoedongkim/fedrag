"""Statement Comparison: both statements looked up by meeting date, diffed, with the cached (or a
newly written) LLM analysis and the stored stance of each.

No connection is held during an LLM call (the API pool is small and a call takes seconds): the
statements and the cache are read on one, the analysis is written, then stored on another. A
per-key lock keeps concurrent requests for the same pair in this process from paying twice.
"""

import logging
import threading
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from datetime import date
from typing import Any, Protocol

import psycopg
from pydantic import ValidationError

from app.compare.diff import diff_statements
from app.compare.types import (
    Analyzer,
    ComparisonInput,
    StatementText,
    cache_key,
    restrict_to_changes,
)
from app.db import compare as compare_db
from app.rag.answer import LLMNotConfiguredError
from app.schemas.common import StanceClassifier
from app.schemas.compare import (
    AnalysisMode,
    AnalysisStatus,
    CompareRequest,
    CompareResponse,
    ComparisonAnalysis,
    ComparisonStance,
    SentenceChange,
)

logger = logging.getLogger(__name__)

# The stance shown next to a comparison: the zero-shot LLM classifier's, as stored by make stance.
STANCE_CLASSIFIER: StanceClassifier = "llm_zero_shot"


class ConnectionSource(Protocol):
    """Lends a connection, like app.db.connection.Database (the API pool)."""

    def connection(self) -> AbstractContextManager[psycopg.Connection[Any]]: ...


class StatementNotFoundError(LookupError):
    """No statement for a requested meeting date; the message names the nearest statement dates."""


@dataclass(frozen=True)
class AnalysisOutcome:
    """The analysis of a comparison, or why there is none. model_version is "model/prompt"."""

    status: AnalysisStatus
    analysis: ComparisonAnalysis | None = None
    model_version: str | None = None


# One lock per cache key ever generated in this process: at most one per pair of statements and
# prompt version, a few thousand.
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def comparison_input(base: StatementText, target: StatementText) -> ComparisonInput:
    """Diff two statements into what the analysis reads."""
    return ComparisonInput(base, target, diff_statements(base, target).changes)


def compare_statements(
    db: ConnectionSource,
    request: CompareRequest,
    analyzer: Analyzer,
    *,
    llm_configured: bool,
    demo: bool = False,
) -> CompareResponse:
    """The diff of the statements of request.base_date and request.target_date, the analysis as
    request.analysis asks (analysis_for; `demo` for the public demo), and both stored stances.
    Raises StatementNotFoundError."""
    with db.connection() as conn:
        found = compare_db.statements_on(conn, [request.base_date, request.target_date])
        missing = [day for day in (request.base_date, request.target_date) if day not in found]
        if missing:
            raise StatementNotFoundError(" ".join(_not_found(conn, day) for day in missing))
        base, target = found[request.base_date], found[request.target_date]
        doc_ids = [base.source.doc_id, target.source.doc_id]
        stances = compare_db.stances_of(conn, doc_ids, STANCE_CLASSIFIER)

    diff = diff_statements(base.statement, target.statement)
    comparison = ComparisonInput(base.statement, target.statement, diff.changes)
    outcome = analysis_for(
        db, comparison, analyzer, mode=request.analysis, llm_configured=llm_configured, demo=demo
    )
    return CompareResponse(
        base=base.source,
        target=target.source,
        changes=[SentenceChange.model_validate(c, from_attributes=True) for c in diff.changes],
        stats=diff.stats,
        analysis=outcome.analysis,
        analysis_status=outcome.status,
        analysis_model_version=outcome.model_version,
        stance=ComparisonStance(base=stances.get(doc_ids[0]), target=stances.get(doc_ids[1])),
    )


def analysis_for(
    db: ConnectionSource,
    comparison: ComparisonInput,
    analyzer: Analyzer,
    *,
    mode: AnalysisMode,
    llm_configured: bool = True,
    demo: bool = False,
) -> AnalysisOutcome:
    """The stored analysis of `comparison` under the analyzer's model and prompt version; with
    mode "generate" and none stored, a new one (one paid call, two when the first reply cannot be
    parsed), stored before it is returned.

    On the public demo (`demo`) nothing is ever written: without a stored analysis the status is
    disabled_on_demo whatever the mode, and the analyzer is never called. Without an LLM
    (llm_configured false) it is llm_not_configured whatever the mode. Either way a client knows
    asking to generate cannot help. A failed call is "failed", logged with the
    error's class only (OpenAI messages can quote part of the key); a failed store is logged and
    the analysis still returned.
    """
    key = cache_key(comparison, model=analyzer.model, prompt_version=analyzer.prompt_version)
    with db.connection() as conn:
        cached = _cached(conn, key, comparison)
    if cached is not None:
        return cached
    if demo:
        return AnalysisOutcome("disabled_on_demo")
    if not llm_configured:
        return AnalysisOutcome("llm_not_configured")
    if mode == "cached_only":
        return AnalysisOutcome("not_cached")

    with _lock(key):
        with db.connection() as conn:  # a request that held the lock may have just stored it
            cached = _cached(conn, key, comparison)
        if cached is not None:
            return cached
        try:
            run = analyzer(comparison)
        except LLMNotConfiguredError:
            return AnalysisOutcome("llm_not_configured")
        except Exception as exc:
            import openai  # loaded with the analyzer's chat model, not with this module

            if isinstance(exc, openai.OpenAIError):
                logger.warning(
                    "Comparison analysis failed: OpenAI request failed (%s)", type(exc).__name__
                )
            else:
                logger.exception("Comparison analysis failed")
            return AnalysisOutcome("failed")
        analysis = restrict_to_changes(run.analysis, comparison.changes)
        try:
            with db.connection() as conn:
                compare_db.save_comparison(
                    conn,
                    cache_key=key,
                    base_doc_id=comparison.base.doc_id,
                    target_doc_id=comparison.target.doc_id,
                    model=analyzer.model,
                    prompt_version=analyzer.prompt_version,
                    analysis=analysis,
                    usage=run.usage,
                )
        except psycopg.Error as exc:
            logger.warning("Comparison analysis not stored (%s)", type(exc).__name__)
    return AnalysisOutcome("ready", analysis, f"{analyzer.model}/{analyzer.prompt_version}")


def _cached(
    conn: psycopg.Connection[Any], key: str, comparison: ComparisonInput
) -> AnalysisOutcome | None:
    stored = compare_db.read_comparison(conn, key)
    if stored is None:
        return None
    try:
        analysis = ComparisonAnalysis.model_validate(stored.analysis)
    except ValidationError:
        logger.warning("Stored comparison analysis %s is invalid; treating it as not cached", key)
        return None
    return AnalysisOutcome(
        "ready",
        restrict_to_changes(analysis, comparison.changes),
        f"{stored.model}/{stored.prompt_version}",
    )


def _not_found(conn: psycopg.Connection[Any], day: date) -> str:
    nearest = [d.isoformat() for d in compare_db.nearest_statement_dates(conn, day) if d]
    if not nearest:
        return f"No FOMC statement for a meeting on {day}: no statements are loaded."
    label = "date" if len(nearest) == 1 else "dates"
    dates = ", ".join(nearest)
    return f"No FOMC statement for a meeting on {day}; nearest statement {label}: {dates}."


@contextmanager
def _lock(key: str) -> Iterator[None]:
    with _locks_guard:
        lock = _locks.setdefault(key, threading.Lock())
    with lock:
        yield
