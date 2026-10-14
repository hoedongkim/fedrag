"""Statement Comparison: sentence-level diff of two FOMC statements and an LLM reading of it."""

from datetime import date
from typing import Literal, Self

from pydantic import Field, model_validator

from app.schemas.common import (
    ApiModel,
    RequestModel,
    SourceDocument,
    Stance,
    StanceClassifier,
)

ChangeKind = Literal["unchanged", "modified", "added", "removed"]
SegmentOp = Literal["equal", "insert", "delete"]
StanceShift = Literal["more_hawkish", "unchanged", "more_dovish"]
AnalysisMode = Literal["cached_only", "generate"]
AnalysisStatus = Literal["ready", "not_cached", "llm_not_configured", "disabled_on_demo", "failed"]


class CompareRequest(RequestModel):
    """Two meetings whose FOMC statements should be compared, earlier one first."""

    base_date: date = Field(description="Meeting date of the earlier (base) statement.")
    target_date: date = Field(description="Meeting date of the later (target) statement.")
    analysis: AnalysisMode = Field(
        default="cached_only",
        description=(
            'LLM analysis of the changes. "cached_only" returns a stored analysis if there is '
            'one and never calls the LLM; "generate" also writes and stores one when none is '
            "stored (normally one paid LLM call per pair, two if the first reply cannot be "
            "parsed; stored once it succeeds). The public demo never writes one: there a pair "
            'without a stored analysis has analysis_status "disabled_on_demo".'
        ),
    )

    @model_validator(mode="after")
    def _base_before_target(self) -> Self:
        if self.base_date >= self.target_date:
            raise ValueError("base_date must be earlier than target_date")
        return self


class DiffSegment(ApiModel):
    """A run of words that is common to both sentences, or only in one of them."""

    op: SegmentOp = Field(
        description='"equal" (in both), "delete" (only in base) or "insert" (only in target).'
    )
    text: str = Field(description="The words, with the whitespace between them, verbatim.")


class SentenceChange(ApiModel):
    """One item of the sentence alignment: a base/target sentence pair, or a lone sentence."""

    id: int = Field(
        ge=0,
        description=(
            "Index of the item in `changes`; the analysis cites changes by it. Stable for a "
            "given pair of statement texts."
        ),
    )
    kind: ChangeKind = Field(
        description=(
            '"unchanged" (identical up to whitespace), "modified" (aligned with a similarity '
            'at or above the threshold), "added" (only in target) or "removed" (only in base).'
        )
    )
    base: str | None = Field(description="Sentence of the base statement; null if added.")
    target: str | None = Field(description="Sentence of the target statement; null if removed.")
    similarity: float | None = Field(
        ge=0,
        le=1,
        description=(
            "Word-level similarity of the pair (difflib ratio on lower-cased word tokens); null "
            "if added or removed."
        ),
    )
    procedural: bool = Field(
        description="A roll call, vote tally or similar note rather than policy content."
    )
    segments: list[DiffSegment] | None = Field(
        description=(
            "Word-level diff of a modified pair: the equal and delete segments rebuild the base "
            "sentence exactly, the equal and insert segments the target one. Null otherwise."
        )
    )


class ComparisonStats(ApiModel):
    """Counts of the aligned items by kind, and how similar the two statements are overall."""

    unchanged: int = Field(ge=0, description="Sentences identical up to whitespace.")
    modified: int = Field(ge=0, description="Sentence pairs that were reworded.")
    added: int = Field(ge=0, description="Sentences only in the target statement.")
    removed: int = Field(ge=0, description="Sentences only in the base statement.")
    similarity: float = Field(
        ge=0,
        le=1,
        description=(
            "Word-level similarity of the two whole statements (difflib ratio on lower-cased "
            "word tokens): 1 means the same words in the same order."
        ),
    )


class KeyChange(ApiModel):
    """One policy-relevant change in wording and what it signals."""

    change_ids: list[int] = Field(
        min_length=1, description="Ids of the changes (items of `changes`) this point is about."
    )
    what_changed: str = Field(description="What changed, quoting the changed words.")
    interpretation: str = Field(description="What the change signals for monetary policy.")
    direction: Stance = Field(description="Policy lean of this change.")


class ComparisonAnalysis(ApiModel):
    """An LLM reading of the changes, as of the target statement's release date."""

    summary: str = Field(description="What the changes signal overall, in 2 to 4 sentences.")
    key_changes: list[KeyChange] = Field(
        description="The policy-relevant changes, most important first."
    )
    stance_shift: StanceShift = Field(
        description="Whether the target statement leans more hawkish or more dovish than the base."
    )
    confidence: float = Field(
        ge=0,
        le=1,
        description="Model-reported confidence in stance_shift. NOT a calibrated probability.",
    )


class StanceSnapshot(ApiModel):
    """A statement's stance according to one classifier (GET /api/stance has the details)."""

    classifier: StanceClassifier = Field(description="Classifier that produced the stance.")
    stance: Stance = Field(description="Predicted policy stance.")
    score: float = Field(
        ge=-1,
        le=1,
        description=(
            "Model-reported stance score: +1 most hawkish, -1 most dovish. "
            "NOT a calibrated probability."
        ),
    )


class ComparisonStance(ApiModel):
    """The stored stance of both statements, for context next to the analysis."""

    base: StanceSnapshot | None = Field(description="Base statement; null if not classified.")
    target: StanceSnapshot | None = Field(description="Target statement; null if not classified.")


class CompareResponse(ApiModel):
    """Sentence-level diff of two FOMC statements plus an LLM reading of what changed."""

    base: SourceDocument = Field(description="Earlier (base) statement.")
    target: SourceDocument = Field(description="Later (target) statement.")
    changes: list[SentenceChange] = Field(
        description="The sentence alignment in document order, unchanged sentences included."
    )
    stats: ComparisonStats = Field(description="Summary of the alignment.")
    analysis: ComparisonAnalysis | None = Field(
        description="LLM reading of the changes; null unless analysis_status is ready."
    )
    analysis_status: AnalysisStatus = Field(
        description=(
            '"ready": analysis holds a stored or newly written analysis. "not_cached": none is '
            'stored for this pair; request analysis="generate" to write one. '
            '"llm_not_configured": none is stored and OPENAI_API_KEY is not set, so none can be '
            'written. "disabled_on_demo": none is stored and this is the public demo, where no '
            'analysis is written (whatever the analysis mode), to avoid API costs. "failed": '
            "writing one failed; the diff is still returned."
        )
    )
    analysis_model_version: str | None = Field(
        description=(
            "Model and prompt version that wrote the analysis, e.g. "
            '"gpt-5.4-mini-2026-03-17/compare-v3.edfeb8da" (the prompt version ends in a '
            "fingerprint of its templates); null without an analysis."
        )
    )
    stance: ComparisonStance = Field(
        description=(
            "Stance of each statement according to the llm_zero_shot classifier, as stored by "
            "the stance pipeline (not part of the analysis)."
        )
    )
