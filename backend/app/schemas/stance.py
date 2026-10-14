"""Policy Stance: hawkish/neutral/dovish classification per meeting, and its benchmarks."""

import datetime as dt
from typing import Literal, Self

from pydantic import Field, model_validator

from app.schemas.common import ApiModel, RequestModel, Stance, StanceClassifier

ReferenceSource = Literal["llm_adjudicated", "human"]
BenchmarkReference = Literal["llm_adjudicated", "mixed", "human"]
BenchmarkSplit = Literal["official", "deduplicated"]
StatementVariant = Literal["pre_registered", "post_hoc_rubric_ordered"]
StatementSubset = Literal["all", "human_subset"]
SentenceBaselineName = Literal["majority_class"]
StatementBaselineName = Literal["majority_class", "rate_move_rule"]

CLASSIFIER_QUERY = (
    "Classifier to report. The default, llm_few_shot, was fixed before any results and is not "
    "chosen by score; GET /api/stance/benchmark compares the classifiers."
)


class ReferenceLabel(ApiModel):
    """The reference stance of a statement that the classifiers are scored against."""

    label: Stance = Field(description="Reference stance of the statement.")
    source: ReferenceSource = Field(
        description=(
            '"human" when a human reviewed the statement blind and set the label; otherwise '
            '"llm_adjudicated": two LLM annotations under the labelling rubric, with '
            "disagreements adjudicated by a third pass."
        )
    )
    confidence: float | str | None = Field(
        default=None,
        description=(
            "Confidence of the LLM-adjudicated label as the labelling run recorded it (a number "
            'in [0, 1] or a level such as "high"); null for a human label. NOT a calibrated '
            "probability."
        ),
    )


class StanceResult(ApiModel):
    """Stance of one FOMC meeting according to one classifier."""

    meeting_date: dt.date = Field(description="Date of the FOMC meeting.")
    doc_id: str = Field(description="Document that was classified.")
    classifier: StanceClassifier = Field(description="Classifier that produced this result.")
    stance: Stance = Field(description="Predicted policy stance.")
    score: float = Field(
        ge=-1,
        le=1,
        description=(
            "Model-reported stance score: +1 most hawkish, -1 most dovish. "
            "NOT a calibrated probability."
        ),
    )
    confidence: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description=(
            "Model-reported confidence (LLM classifiers) or |score| (sentence classifiers). "
            "NOT a calibrated probability."
        ),
    )
    key_factors: list[str] = Field(
        default_factory=list, description="Short phrases summarising what drove the call."
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="Sentences quoted verbatim from the document that drove the call.",
    )
    model_version: str = Field(
        description="Model, prompt or lexicon version that produced the result."
    )
    url: str = Field(description="Canonical source URL of the classified document.")
    reference: ReferenceLabel | None = Field(
        default=None,
        description=(
            "Reference label of the statement (data/eval/stance_gold.jsonl); null when there is "
            "none."
        ),
    )


class FedFundsTarget(ApiModel):
    """The federal funds target range from one change point until the next (FRED DFEDTARL and
    DFEDTARU)."""

    date: dt.date = Field(description="Day the range took effect.")
    lower: float = Field(ge=0, description="Lower bound of the target range, in percent.")
    upper: float = Field(ge=0, description="Upper bound of the target range, in percent.")

    @model_validator(mode="after")
    def _lower_not_above_upper(self) -> Self:
        if self.lower > self.upper:
            raise ValueError("lower must not be above upper")
        return self


class StanceTimeline(ApiModel):
    """Stance results for a range of meetings, in meeting order, with the policy rate."""

    classifier: StanceClassifier = Field(description="Classifier that produced the results.")
    points: list[StanceResult] = Field(description="One result per meeting.")
    fed_funds: list[FedFundsTarget] = Field(
        description=(
            "Change points of the federal funds target range in date order: first the range in "
            "force on `start` (its date can be earlier than `start`), then every change up to "
            "`end`. Without `start`, every change point up to `end`. Empty when not loaded."
        )
    )


class StanceTimelineQuery(RequestModel):
    """Query parameters for the stance timeline."""

    classifier: StanceClassifier = Field(default="llm_few_shot", description=CLASSIFIER_QUERY)
    start: dt.date | None = Field(default=None, description="Earliest meeting date (inclusive).")
    end: dt.date | None = Field(default=None, description="Latest meeting date (inclusive).")

    @model_validator(mode="after")
    def _start_not_after_end(self) -> Self:
        if self.start and self.end and self.start > self.end:
            raise ValueError("start must be on or before end")
        return self


class StanceF1(ApiModel):
    """F1 score of each stance class."""

    dovish: float = Field(ge=0, le=1, description="F1 of the dovish class.")
    neutral: float = Field(ge=0, le=1, description="F1 of the neutral class.")
    hawkish: float = Field(ge=0, le=1, description="F1 of the hawkish class.")


class SentenceBenchmark(ApiModel):
    """Sentence-level scores of one classifier on one test split of Trillion Dollar Words, as
    the mean and standard deviation over the dataset's three train/test seeds."""

    classifier: StanceClassifier = Field(description="Classifier being scored.")
    split: BenchmarkSplit = Field(
        description=(
            '"official": the published test split. "deduplicated": the test split minus '
            "sentences whose text also appears in that seed's train split."
        )
    )
    macro_f1_mean: float = Field(ge=0, le=1, description="Macro-F1, mean over the seeds.")
    macro_f1_sd: float | None = Field(
        ge=0, description="Macro-F1, sample standard deviation over the seeds; null for one seed."
    )
    accuracy_mean: float = Field(ge=0, le=1, description="Accuracy, mean over the seeds.")
    accuracy_sd: float | None = Field(
        ge=0, description="Accuracy, sample standard deviation over the seeds; null for one seed."
    )
    per_class_f1_mean: StanceF1 = Field(description="F1 of each class, mean over the seeds.")
    n_seeds: int = Field(ge=1, description="Number of train/test seeds averaged.")
    n_test_mean: float = Field(ge=0, description="Test sentences per seed, on average.")


class ConfidenceInterval(ApiModel):
    """A bootstrap confidence interval (StanceBenchmark.statement_bootstrap says which)."""

    low: float = Field(description="Lower bound.")
    high: float = Field(description="Upper bound.")

    @model_validator(mode="after")
    def _low_not_above_high(self) -> Self:
        if self.low > self.high:
            raise ValueError("low must not be above high")
        return self


_ACCURACY_CI = (
    "95% percentile bootstrap interval of the accuracy, resampling statements "
    "(StanceBenchmark.statement_bootstrap); null when not computed."
)
_MACRO_F1_CI = (
    "95% percentile bootstrap interval of the macro-F1, resampling statements "
    "(StanceBenchmark.statement_bootstrap); null when not computed."
)


class StatementScores(ApiModel):
    """Agreement of one classifier with the reference labels on a subset of the statements."""

    n: int = Field(ge=0, description="Statements in the subset.")
    accuracy: float = Field(
        ge=0, le=1, description="Share of statements labelled as the reference."
    )
    macro_f1: float = Field(ge=0, le=1, description="Macro-F1 over the three stances.")
    kappa: float | None = Field(
        ge=-1, le=1, description="Cohen's kappa with the reference; null when undefined."
    )
    accuracy_ci: ConfidenceInterval | None = Field(default=None, description=_ACCURACY_CI)
    macro_f1_ci: ConfidenceInterval | None = Field(default=None, description=_MACRO_F1_CI)


class StatementBenchmark(ApiModel):
    """Statement-level agreement of one classifier with the reference labels."""

    classifier: StanceClassifier = Field(description="Classifier being scored.")
    variant: StatementVariant = Field(
        default="pre_registered",
        description=(
            'How a statement\'s stance was obtained. "pre_registered": the method fixed before '
            "any reference label existed (the LLM judges the whole statement; a sentence "
            "classifier's sentence scores are averaged and cut at +-0.15); every timeline result "
            'is this variant. "post_hoc_rubric_ordered" (sentence classifiers only): the '
            "rate-action sentence's label decides, else the mean score of the rate-guidance "
            "sentences, else neutral; designed after the pre-registered results were seen."
        ),
    )
    n: int = Field(ge=0, description="Statements that have both a prediction and a reference.")
    accuracy: float = Field(
        ge=0, le=1, description="Share of statements labelled as the reference."
    )
    macro_f1: float = Field(ge=0, le=1, description="Macro-F1 over the three stances.")
    kappa: float | None = Field(
        ge=-1, le=1, description="Cohen's kappa with the reference; null when undefined."
    )
    reference: BenchmarkReference = Field(
        description=(
            'Reference labels the scores were computed against: "llm_adjudicated" (no human '
            'review yet), "mixed" (some statements human-reviewed) or "human".'
        )
    )
    spearman: float | None = Field(
        default=None,
        ge=-1,
        le=1,
        description=(
            "Spearman rank correlation of the classifier's score with the reference stance "
            "(dovish -1, neutral 0, hawkish +1): agreement in ordering, free of any label "
            "threshold; null when undefined."
        ),
    )
    accuracy_ci: ConfidenceInterval | None = Field(default=None, description=_ACCURACY_CI)
    macro_f1_ci: ConfidenceInterval | None = Field(default=None, description=_MACRO_F1_CI)
    human_subset: StatementScores | None = Field(
        default=None,
        description=(
            "The same scores on only the statements whose reference label a human set in a "
            "blind review; null before any review."
        ),
    )


class SentenceBaseline(ApiModel):
    """A reference point that uses no classifier, on one TDW test split (mean and sd over the
    seeds): the floor a sentence classifier has to clear."""

    name: SentenceBaselineName = Field(
        description='"majority_class": each seed\'s most frequent training label, always.'
    )
    label: Stance | None = Field(
        description="The stance it always predicts; null when the seeds' majority classes differ."
    )
    split: BenchmarkSplit = Field(description="Test split, as in SentenceBenchmark.")
    macro_f1_mean: float = Field(ge=0, le=1, description="Macro-F1, mean over the seeds.")
    macro_f1_sd: float | None = Field(
        ge=0, description="Macro-F1, sample standard deviation over the seeds; null for one seed."
    )
    accuracy_mean: float = Field(ge=0, le=1, description="Accuracy, mean over the seeds.")
    accuracy_sd: float | None = Field(
        ge=0, description="Accuracy, sample standard deviation over the seeds; null for one seed."
    )
    n_seeds: int = Field(ge=1, description="Number of train/test seeds averaged.")
    n_test_mean: float = Field(ge=0, description="Test sentences per seed, on average.")


class StatementBaseline(ApiModel):
    """A reference point that uses no classifier, scored against the reference labels like a
    statement-level row."""

    name: StatementBaselineName = Field(
        description=(
            '"majority_class": always the most frequent reference label. "rate_move_rule": '
            "the verb of the statement's rate-action sentence decides a move (raise or increase "
            "hawkish, lower, reduce or cut dovish) and anything else is neutral; a post-hoc "
            "control for the post-hoc rows, showing what detecting the move alone scores."
        )
    )
    label: Stance | None = Field(
        description="The stance it always predicts (majority_class); null otherwise."
    )
    post_hoc: bool = Field(description="Designed after the pre-registered results were seen.")
    n: int = Field(ge=0, description="Statements that have a reference label.")
    accuracy: float = Field(
        ge=0, le=1, description="Share of statements labelled as the reference."
    )
    macro_f1: float = Field(ge=0, le=1, description="Macro-F1 over the three stances.")
    kappa: float | None = Field(
        ge=-1, le=1, description="Cohen's kappa with the reference; null when undefined."
    )
    spearman: float | None = Field(
        default=None,
        ge=-1,
        le=1,
        description=(
            "Spearman rank correlation of its call (dovish -1, neutral 0, hawkish +1) with the "
            "reference stance; null when undefined (a constant call)."
        ),
    )
    accuracy_ci: ConfidenceInterval | None = Field(default=None, description=_ACCURACY_CI)
    macro_f1_ci: ConfidenceInterval | None = Field(default=None, description=_MACRO_F1_CI)
    human_subset: StatementScores | None = Field(
        default=None,
        description="The same scores on the human-reviewed statements; null before any review.",
    )


class StatementRowKey(ApiModel):
    """Which statement-level row a comparison refers to."""

    classifier: StanceClassifier = Field(description="Classifier of the row.")
    variant: StatementVariant = Field(description="Aggregation variant of the row.")


class MetricDifference(ApiModel):
    """One metric's paired difference between two rows on the same statements."""

    difference: float = Field(
        ge=-1, le=1, description="Observed difference: minuend minus subtrahend."
    )
    ci: ConfidenceInterval = Field(
        description="95% percentile bootstrap interval of the difference (paired resamples)."
    )


class PairedDifference(ApiModel):
    """A paired bootstrap comparison of two statement-level rows: both are scored on the same
    resampled statements, so the interval reflects their disagreements, not two separate
    spreads."""

    minuend: StatementRowKey = Field(description="The row the other is subtracted from.")
    subtrahend: StatementRowKey = Field(description="The row subtracted.")
    subset: StatementSubset = Field(
        description='"all" statements with a reference label, or only the "human_subset".'
    )
    n: int = Field(ge=0, description="Statements compared.")
    accuracy: MetricDifference = Field(description="Accuracy difference.")
    macro_f1: MetricDifference = Field(description="Macro-F1 difference.")


class BaselineDifference(ApiModel):
    """A statement-level row minus a baseline (StanceBenchmark.statement_baselines), both scored
    on the same resampled statements."""

    row: StatementRowKey = Field(description="The row the baseline is subtracted from.")
    baseline: StatementBaselineName = Field(description="The baseline subtracted.")
    subset: StatementSubset = Field(
        description='"all" statements with a reference label, or only the "human_subset".'
    )
    n: int = Field(ge=0, description="Statements compared.")
    accuracy: MetricDifference = Field(description="Accuracy difference.")
    macro_f1: MetricDifference = Field(description="Macro-F1 difference.")


class StatementBootstrap(ApiModel):
    """How the statement-level intervals were computed, and the paired comparisons."""

    method: str = Field(description="The resampling method, in words.")
    resamples: int = Field(ge=1, description="Bootstrap resamples.")
    seed: int = Field(description="Seed of the random number generator.")
    level: float = Field(gt=0, lt=1, description="Confidence level of the intervals.")
    paired_differences: list[PairedDifference] = Field(
        description="Paired comparisons of selected rows."
    )
    baseline_differences: list[BaselineDifference] = Field(
        default_factory=list, description="Paired comparisons of selected rows with a baseline."
    )


class BenchmarkDataset(ApiModel):
    """The public sentence-level dataset the classifiers are benchmarked on."""

    name: str = Field(description="Dataset name.")
    citation: str = Field(description="How to cite the dataset.")
    license: str = Field(description="Dataset license.")
    url: str = Field(description="Where the dataset is published.")
    commit: str = Field(description="Commit of the dataset repository that was used.")


class ReferenceLabelSummary(ApiModel):
    """The statement reference labels as they stand now (data/eval/stance_gold.jsonl)."""

    n_labels: int = Field(ge=0, description="Statements with a reference label.")
    n_human_reviewed: int = Field(
        ge=0, description="Statements whose reference label was set by a human reviewer."
    )


class StanceBenchmark(ApiModel):
    """How well each stance classifier does, on sentences and on whole statements."""

    generated_at: dt.datetime = Field(description="When the benchmark was computed.")
    dataset: BenchmarkDataset = Field(description="Sentence-level benchmark dataset.")
    sentence_level: list[SentenceBenchmark] = Field(
        description="One entry per classifier and test split."
    )
    statement_level: list[StatementBenchmark] = Field(
        description=(
            "One entry per classifier and aggregation variant (pre-registered first); empty "
            "until reference labels exist."
        )
    )
    sentence_baselines: list[SentenceBaseline] = Field(
        default_factory=list,
        description="Reference points with no classifier on the same test splits.",
    )
    statement_baselines: list[StatementBaseline] = Field(
        default_factory=list,
        description="Reference points with no classifier on the same statements.",
    )
    statement_bootstrap: StatementBootstrap | None = Field(
        default=None,
        description=(
            "The bootstrap behind the statement-level intervals, with paired comparisons; null "
            "when not computed."
        ),
    )
    reference_labels: ReferenceLabelSummary | None = Field(
        default=None,
        description=(
            "The reference labels as they stand now, which can be newer than the statement-level "
            "scores; null when there are none."
        ),
    )
