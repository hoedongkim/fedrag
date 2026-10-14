"""The comparison's internal types and the interface between the diff and the LLM analysis.

The diff (app.compare.diff) turns two statements into a ComparisonInput; an Analyzer (built by
app.compare.analysis.get_analyzer(settings)) reads one and returns an AnalysisRun; the service
(app.compare.service) caches each run in fedrag.statement_comparisons under cache_key().
"""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from functools import cached_property
from typing import Any, Protocol

from app.schemas.compare import ChangeKind, ComparisonAnalysis, SegmentOp

# Bump when anything that decides the change ids or procedural flags changes (sentence splitting
# and is_procedural in app.stance.statements, tokens, the threshold, the alignment): stored
# analyses cite changes by id, so the cache key includes it. It is bumped by hand: the key does
# not hash the diff itself.
DIFF_VERSION = "diff-v1"


@dataclass(frozen=True)
class Segment:
    """A run of a modified pair's word diff: "equal" (both), "delete" (base) or "insert"
    (target), with its text verbatim."""

    op: SegmentOp
    text: str


@dataclass(frozen=True)
class Change:
    """One item of the alignment, in document order; id is its index in the alignment.

    base / target are verbatim sentences (None for an added / removed one). similarity is the
    word-level ratio of a pair (None for added / removed). segments is the word diff of a
    modified pair (None otherwise). procedural flags a roll call or similar note on either side.
    """

    id: int
    kind: ChangeKind
    base: str | None
    target: str | None
    similarity: float | None
    procedural: bool
    segments: tuple[Segment, ...] | None = None


@dataclass(frozen=True)
class StatementText:
    """An FOMC statement as the comparison reads it: its paragraphs, verbatim."""

    doc_id: str
    meeting_date: date
    release_date: date
    paragraphs: tuple[str, ...]

    @cached_property
    def text(self) -> str:
        """The whole statement as stored in fedrag.documents.text (paragraphs joined by blank
        lines)."""
        return "\n\n".join(self.paragraphs)

    @cached_property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ComparisonInput:
    """What the analysis reads: both statements (ids, dates, texts) and the whole alignment.

    The prompt shows only `changed` (every item that is not unchanged, ids kept) plus the full
    target text, and reads it as of `as_of`, the target statement's release date.
    """

    base: StatementText
    target: StatementText
    changes: tuple[Change, ...]

    @property
    def changed(self) -> tuple[Change, ...]:
        return tuple(change for change in self.changes if change.kind != "unchanged")

    @property
    def as_of(self) -> date:
        return self.target.release_date


@dataclass(frozen=True)
class AnalysisRun:
    """An analysis and what it cost. usage is stored as JSON next to it, e.g. {"calls": 1,
    "input_tokens": ..., "cached_input_tokens": ..., "output_tokens": ..., "usd": ...}."""

    analysis: ComparisonAnalysis
    usage: dict[str, Any] = field(default_factory=dict)


class Analyzer(Protocol):
    """Writes the LLM analysis of one comparison (app.compare.analysis provides it).

    model and prompt_version identify what wrote an analysis: both go into the cache key and are
    stored with it. Calling it makes the paid LLM call(s); it raises on failure (openai errors
    included) rather than returning a partial analysis. Its key_changes should cite only ids of
    `comparison.changed`: the service drops any other id on read (restrict_to_changes).
    """

    @property
    def model(self) -> str: ...

    @property
    def prompt_version(self) -> str: ...

    def __call__(self, comparison: ComparisonInput) -> AnalysisRun: ...


def cache_key(comparison: ComparisonInput, *, model: str, prompt_version: str) -> str:
    """SHA-256 of both doc ids, both texts' SHA-256, the diff version, the prompt version and the
    model: a revised statement text, or a bumped DIFF_VERSION or prompt version, gives a new key
    (the release dates are not part of it)."""
    parts = [
        comparison.base.doc_id,
        comparison.target.doc_id,
        comparison.base.sha256,
        comparison.target.sha256,
        DIFF_VERSION,
        prompt_version,
        model,
    ]
    return hashlib.sha256(json.dumps(parts).encode("utf-8")).hexdigest()


def restrict_to_changes(
    analysis: ComparisonAnalysis, changes: Sequence[Change]
) -> ComparisonAnalysis:
    """The analysis with each key change citing only ids of changed items (unknown ids, ids of
    unchanged sentences and repeats removed); a key change left without ids is dropped."""
    valid = {change.id for change in changes if change.kind != "unchanged"}
    key_changes = []
    for key_change in analysis.key_changes:
        ids = list(dict.fromkeys(i for i in key_change.change_ids if i in valid))
        if ids:
            key_changes.append(key_change.model_copy(update={"change_ids": ids}))
    return analysis.model_copy(update={"key_changes": key_changes})
