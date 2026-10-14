"""RAG evaluation: retrieval and answer-quality benchmarks per retrieval method and retriever
config, with per-question drilldowns.

`make rag-eval` writes data/eval/rag_benchmark.json, an EvaluationFile: the report fields plus one
QuestionRecord per question. GET /api/evaluation returns the report with each question summarised
(EvaluationReport); GET /api/evaluation/questions/{id} returns one question in full
(QuestionDetail), with the passages' text looked up in the search index by chunk id.
"""

import datetime as dt
from collections.abc import Callable, Iterable
from typing import Annotated, Any, Literal, Self

from pydantic import AliasChoices, Field, StringConstraints, model_validator

from app.schemas.common import ApiModel, RetrievalMethod, SourceDocument

QuestionType = Literal[
    "statement_fact",
    "minutes_fact",
    "point_in_time",
    "date_specific",
    "multi_doc",
    "paraphrase",
    "unanswerable",
]
QuestionSplit = Literal["dev", "test"]
ReportSplit = Literal["dev", "test", "all"]
ChunkRole = Literal["primary", "supporting"]
CostStage = Literal["generation", "judge", "embedding"]
JudgeCheck = Literal["claim_support", "citation_support", "fact_coverage", "refusal"]
ValidationStatus = Literal["not_started", "pending", "complete"]
EvalMetric = Literal[
    "recall_at_5",
    "recall_at_10",
    "mrr_at_10",
    "ndcg_at_10",
    "doc_recall_at_5",
    "doc_recall_at_10",
    "faithfulness",
    "citation_accuracy",
    "answer_correctness",
    "refusal_accuracy",
    "false_refusal_rate",
]

CONFIG_ID_PATTERN = r"^[a-z0-9][a-z0-9_-]{0,31}$"
QUESTION_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$"
MAX_RETRIEVED = 20  # passages kept per run; the report scores the top 10

ConfigId = Annotated[str, StringConstraints(pattern=CONFIG_ID_PATTERN)]
QuestionId = Annotated[str, StringConstraints(pattern=QUESTION_ID_PATTERN)]

_SPLIT = (
    'Questions the row is computed on: "dev" (used to choose settings), "test" (held out until '
    'the settings were fixed) or "all".'
)
_CONFIG = "Retriever config id (EvaluationReport.configs), e.g. phase3 or tuned."
_ESTIMATE = "{what}, with its bootstrap interval over questions."
_PER_QUESTION = "{what} of this question; null when it does not apply (e.g. unanswerable)."


# --- Shared pieces ------------------------------------------------------------------------------


class MetricInterval(ApiModel):
    """A percentile bootstrap interval, resampling questions (EvaluationReport.bootstrap)."""

    low: float = Field(description="Lower bound.")
    high: float = Field(description="Upper bound.")

    @model_validator(mode="before")
    @classmethod
    def _from_pair(cls, value: Any) -> Any:
        """Also accept [low, high]."""
        if isinstance(value, list | tuple) and len(value) == 2:
            return {"low": value[0], "high": value[1]}
        return value

    @model_validator(mode="after")
    def _low_not_above_high(self) -> Self:
        if self.low > self.high:
            raise ValueError("low must not be above high")
        return self


class Estimate(ApiModel):
    """A mean over questions, with its interval and the number of questions behind it."""

    value: float | None = Field(
        ge=0, le=1, description="Mean over the questions; null when no question qualifies."
    )
    ci: MetricInterval | None = Field(
        default=None, description="95% bootstrap interval of the mean; null when not computed."
    )
    n: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Questions behind the mean (e.g. only unanswerable ones for refusal accuracy); null "
            "when not recorded."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _from_number(cls, value: Any) -> Any:
        """Also accept a bare number (or null) as a value without an interval, and "mean" for
        "value" ({"mean", "ci", "n"}, as app.evaluation's mean_ci writes it)."""
        if value is None or (isinstance(value, int | float) and not isinstance(value, bool)):
            return {"value": value}
        if isinstance(value, dict) and "value" not in value and "mean" in value:
            return {**value, "value": value["mean"]}
        return value


def _flatten_latency(value: Any) -> Any:
    """{"latency_ms": {"p50", "p95", ...}} (a latency summary) as latency_ms_p50/_p95."""
    if isinstance(value, dict) and isinstance(latency := value.get("latency_ms"), dict):
        return {
            "latency_ms_p50": latency.get("p50"),
            "latency_ms_p95": latency.get("p95"),
            **{k: v for k, v in value.items() if k != "latency_ms"},
        }
    return value


class RunKey(ApiModel):
    """One retrieval method under one retriever config."""

    method: RetrievalMethod = Field(description="Retrieval method.")
    config: ConfigId = Field(description=_CONFIG)


# --- Report: setup ----------------------------------------------------------------------------


class RetrieverConfig(ApiModel):
    """A set of retriever settings every method was run under."""

    id: ConfigId = Field(description="Short id the rows refer to, e.g. phase3 or tuned.")
    label: str = Field(description="Display name, e.g. 'Phase 3 settings'.")
    description: str = Field(description="What the config changes, in a sentence or two.")
    hash: str | None = Field(
        default=None,
        description="Hash of the settings: part of the answer and judge cache keys.",
    )
    params: dict[str, str | int | float | bool | None] = Field(
        default_factory=dict, description="The settings themselves, by name."
    )


class GeneratorInfo(ApiModel):
    """How the evaluated answers were generated: the production chain of POST /api/ask."""

    model: str = Field(description="LLM that wrote the answers.")
    reasoning_effort: str | None = Field(default=None, description="Its reasoning effort.")
    prompt_version: str = Field(description="Version of the answer prompt.")
    top_k: int = Field(ge=1, description="Passages given to the LLM per question.")


_CHECK = (
    '"claim_support": is a claim supported by the passages it cites; '
    '"citation_support": does a cited passage support its sentence; "fact_coverage": '
    'does the answer state a required fact; "refusal": does the answer decline.'
)
_PASS_FAIL = (
    "Share of verdicts the two agree on, with the labels collapsed to pass / fail (supported; "
    "supports or partially supports; covered; answers or partial)."
)
_EXACT = (
    "Share with the same label before collapsing (e.g. supports vs partially_supports); null "
    "when not recorded."
)


class JudgeAgreement(ApiModel):
    """How often the judge's verdicts match an annotator's on one kind of check."""

    check: JudgeCheck = Field(description=_CHECK)
    annotator: str | None = Field(
        default=None,
        description="Id of the annotator compared with (JudgeValidation.annotators); null when "
        "the file names none.",
    )
    n: int = Field(ge=0, description="Verdicts compared.")
    agreement: float = Field(ge=0, le=1, description=_PASS_FAIL)
    exact_agreement: float | None = Field(default=None, ge=0, le=1, description=_EXACT)
    kappa: float | None = Field(
        default=None, ge=-1, le=1, description="Cohen's kappa on pass / fail; null when undefined."
    )


class AnnotatorInfo(ApiModel):
    """One annotator of the judge-validation sample."""

    id: str = Field(min_length=1, description="Short id the agreement rows use, e.g. A.")
    description: str | None = Field(
        default=None, description="Who or what annotated, and how (blind to the judge, ...)."
    )
    isolated: bool = Field(
        default=False,
        description="True when the annotator worked from a copy of the sample alone and saved "
        "its labels before seeing the judge's verdicts or the other annotators' labels; two "
        "annotators that are not isolated may not be independent of each other.",
    )


class SampledRun(RunKey):
    """How many of the validation sample's answers come from one run."""

    n: int = Field(ge=0, description="Sampled answers of this run.")


class AnnotatorAgreement(ApiModel):
    """How often two annotators agree on one kind of check: how reproducible the labels are, when
    the two worked independently (AnnotatorInfo.isolated)."""

    check: JudgeCheck = Field(description=_CHECK)
    annotators: list[str] = Field(
        min_length=2, max_length=2, description="Ids of the two annotators compared."
    )
    n: int = Field(ge=0, description="Labels compared.")
    agreement: float = Field(ge=0, le=1, description=_PASS_FAIL)
    exact_agreement: float | None = Field(default=None, ge=0, le=1, description=_EXACT)
    kappa: float | None = Field(
        default=None, ge=-1, le=1, description="Cohen's kappa on pass / fail; null when undefined."
    )


class JudgeValidation(ApiModel):
    """The check of the LLM judge against annotators on a blind sample of answers."""

    status: ValidationStatus = Field(
        description=(
            '"not_started": no sample exported; "pending": a sample was exported and awaits '
            'annotation; "complete": annotated, and agreement is reported.'
        )
    )
    n_sampled: int = Field(ge=0, description="Answers in the exported sample.")
    n_annotated: int = Field(ge=0, description="Answers the annotator has labelled.")
    annotator: str | None = Field(
        default=None, description="Who annotated, e.g. 'one human annotator, blind to verdicts'."
    )
    annotators: list[AnnotatorInfo] = Field(
        default_factory=list, description="Each annotator, when the file names them."
    )
    sample_file: str | None = Field(
        default=None, description="Repository path of the exported sample."
    )
    sampled_runs: list[SampledRun] = Field(
        default_factory=list,
        description="The runs the sampled answers come from: the judge was checked on these only.",
    )
    agreement: list[JudgeAgreement] = Field(
        default_factory=list,
        description="Judge vs annotator, per annotator and kind of check; empty until annotated.",
    )
    inter_annotator: list[AnnotatorAgreement] = Field(
        default_factory=list,
        description="Annotator vs annotator per kind of check; empty with fewer than two.",
    )
    note: str | None = Field(default=None, description="Anything else worth knowing.")


class JudgeInfo(ApiModel):
    """The LLM judge that scored the answers."""

    model: str = Field(
        description="Judge model: larger than the generator, to limit self-preference."
    )
    reasoning_effort: str | None = Field(default=None, description="Its reasoning effort.")
    prompt_version: str = Field(description="Version of the judge prompt.")
    n_answers: int = Field(ge=0, description="Answers judged.")
    validation: JudgeValidation | None = Field(
        default=None, description="Agreement with annotators on a blind sample; null before any."
    )


class BootstrapInfo(ApiModel):
    """How the intervals were computed."""

    method: str = Field(description="The resampling method, in words.")
    resamples: int = Field(ge=1, description="Bootstrap resamples.")
    seed: int = Field(description="Seed of the random number generator.")
    level: float = Field(gt=0, lt=1, description="Confidence level of the intervals.")


class SplitCount(ApiModel):
    """Questions in one split."""

    split: QuestionSplit = Field(description="dev or test.")
    n: int = Field(ge=0, description="Questions.")
    n_answerable: int = Field(ge=0, description="Questions the corpus can answer.")


class TypeCount(ApiModel):
    """Questions of one type."""

    type: QuestionType = Field(description="Question type.")
    n: int = Field(ge=0, description="Questions.")
    n_answerable: int = Field(ge=0, description="Questions the corpus can answer.")


class EvalDataset(ApiModel):
    """The evaluation question set."""

    file: str = Field(description="Repository path of the question set.")
    sha256: str | None = Field(default=None, description="Hash of the file that was evaluated.")
    n_questions: int = Field(ge=0, description="Questions.")
    n_answerable: int = Field(ge=0, description="Questions the corpus can answer.")
    n_with_as_of: int = Field(ge=0, description="Questions with a point-in-time cutoff.")
    n_human_reviewed: int = Field(
        ge=0, description="Questions whose labels a human reviewed and corrected where needed."
    )
    splits: list[SplitCount] = Field(description="Questions per split.")
    types: list[TypeCount] = Field(description="Questions per type.")
    construction: str | None = Field(
        default=None, description="How the questions and their labels were made, in words."
    )


# --- Report: results ----------------------------------------------------------------------------


class RetrievalRow(ApiModel):
    """Retrieval scores of one method and config on one split, over the answerable questions
    (point-in-time cutoffs applied as in /api/ask)."""

    @model_validator(mode="before")
    @classmethod
    def _latency_summary(cls, value: Any) -> Any:
        return _flatten_latency(value)

    method: RetrievalMethod = Field(description="Retrieval method.")
    config: ConfigId = Field(description=_CONFIG)
    split: ReportSplit = Field(description=_SPLIT)
    n_questions: int = Field(ge=0, description="Answerable questions scored.")
    recall_at_5: Estimate = Field(
        description=_ESTIMATE.format(what="Share of a question's gold chunks in the top 5")
    )
    recall_at_10: Estimate = Field(
        description=_ESTIMATE.format(what="Share of a question's gold chunks in the top 10")
    )
    mrr_at_10: Estimate = Field(
        description=_ESTIMATE.format(what="Reciprocal rank of the first gold chunk in the top 10")
    )
    ndcg_at_10: Estimate = Field(
        description=_ESTIMATE.format(
            what="nDCG@10 with graded gains (primary chunk 2, supporting chunk 1)"
        )
    )
    doc_recall_at_5: Estimate = Field(
        description=_ESTIMATE.format(what="Share of a question's relevant documents in the top 5")
    )
    doc_recall_at_10: Estimate = Field(
        description=_ESTIMATE.format(what="Share of a question's relevant documents in the top 10")
    )
    latency_ms_p50: float | None = Field(
        default=None, ge=0, description="Median retrieval latency per question, ms."
    )
    latency_ms_p95: float | None = Field(
        default=None, ge=0, description="95th-percentile retrieval latency per question, ms."
    )
    as_of_leaks: int = Field(
        default=0,
        ge=0,
        description="Retrieved passages released after their question's as_of; must be 0.",
    )


class AnswerRow(ApiModel):
    """LLM-judged quality of the answers one method and config produced, on one split."""

    @model_validator(mode="before")
    @classmethod
    def _latency_summary(cls, value: Any) -> Any:
        return _flatten_latency(value)

    method: RetrievalMethod = Field(description="Retrieval method that fed the answers.")
    config: ConfigId = Field(description=_CONFIG)
    split: ReportSplit = Field(description=_SPLIT)
    n_questions: int = Field(ge=0, description="Answers judged.")
    faithfulness: Estimate = Field(
        description=_ESTIMATE.format(
            what="Share of an answer's claims supported by the passages it cites"
        )
    )
    citation_accuracy: Estimate = Field(
        description=_ESTIMATE.format(
            what="Share of an answer's citations whose passage supports the cited sentence"
        )
    )
    answer_correctness: Estimate = Field(
        description=_ESTIMATE.format(what="Share of a question's required facts the answer states")
    )
    refusal_accuracy: Estimate = Field(
        description=_ESTIMATE.format(
            what="Share of unanswerable questions declined without fabricating an answer"
        )
    )
    false_refusal_rate: Estimate = Field(
        description=_ESTIMATE.format(
            what="Share of answerable questions declined (lower is better)"
        )
    )
    latency_ms_p50: float | None = Field(
        default=None, ge=0, description="Median end-to-end latency (retrieval and answer), ms."
    )
    latency_ms_p95: float | None = Field(
        default=None, ge=0, description="95th-percentile end-to-end latency, ms."
    )
    cost_per_query_usd: float | None = Field(
        default=None,
        ge=0,
        description="Mean OpenAI cost of the answer call for one question (the query embedding, "
        "about $0.000001, is excluded), at the prices in code.",
    )
    judge_cost_per_query_usd: float | None = Field(
        default=None, ge=0, description="Mean OpenAI cost of judging one answer."
    )


class TypeRow(ApiModel):
    """Scores of one method and config on the questions of one type."""

    method: RetrievalMethod = Field(description="Retrieval method.")
    config: ConfigId = Field(description=_CONFIG)
    split: ReportSplit = Field(description=_SPLIT)
    type: QuestionType = Field(description="Question type.")
    n_questions: int = Field(ge=0, description="Questions of this type in the split.")
    recall_at_5: Estimate | None = Field(default=None, description="As RetrievalRow.")
    recall_at_10: Estimate | None = Field(default=None, description="As RetrievalRow.")
    mrr_at_10: Estimate | None = Field(default=None, description="As RetrievalRow.")
    ndcg_at_10: Estimate | None = Field(default=None, description="As RetrievalRow.")
    doc_recall_at_5: Estimate | None = Field(default=None, description="As RetrievalRow.")
    doc_recall_at_10: Estimate | None = Field(default=None, description="As RetrievalRow.")
    faithfulness: Estimate | None = Field(default=None, description="As AnswerRow.")
    citation_accuracy: Estimate | None = Field(default=None, description="As AnswerRow.")
    answer_correctness: Estimate | None = Field(default=None, description="As AnswerRow.")
    refusal_accuracy: Estimate | None = Field(default=None, description="As AnswerRow.")
    false_refusal_rate: Estimate | None = Field(default=None, description="As AnswerRow.")


class EvalDifference(ApiModel):
    """A paired bootstrap comparison of two runs on the same questions: both are scored on each
    resample, so the interval reflects where they differ, not two separate spreads."""

    metric: EvalMetric = Field(description="Metric compared.")
    split: ReportSplit = Field(description=_SPLIT)
    minuend: RunKey = Field(description="The run the other is subtracted from.")
    subtrahend: RunKey = Field(description="The run subtracted.")
    n: int = Field(ge=0, description="Questions compared.")
    difference: float = Field(ge=-1, le=1, description="Observed difference: minuend - subtrahend.")
    ci: MetricInterval = Field(description="95% percentile bootstrap interval of the difference.")


class ConfigSelection(ApiModel):
    """The run chosen, and how: the best dev run of a metric, or a rule decided before the test
    split was scored (e.g. the default-switch rule)."""

    method: RetrievalMethod = Field(description="Chosen retrieval method.")
    config: ConfigId = Field(description=_CONFIG)
    metric: EvalMetric = Field(
        description="Metric the choice maximised, or the rule's main metric."
    )
    split: ReportSplit = Field(
        description="Split the choice was made on: dev for a best-of-dev choice, test for a "
        "rule decided before the test run and applied once to the held-out split."
    )
    rule: str = Field(description="The selection rule, in words (including tie-breaks).")


class CostLine(ApiModel):
    """OpenAI usage of one stage of the evaluation."""

    stage: CostStage = Field(description="generation, judge or embedding (query vectors).")
    model: str = Field(description="Model called.")
    calls: int = Field(
        ge=0,
        description="API calls behind the results, cached ones included; for the embedding "
        "stage, the texts embedded (they go out in batched requests).",
    )
    input_tokens: int = Field(ge=0, description="Input tokens.")
    cached_input_tokens: int = Field(default=0, ge=0, description="Input tokens OpenAI cached.")
    output_tokens: int = Field(ge=0, description="Output tokens, reasoning included.")
    usd: float = Field(ge=0, description="Cost in USD at the recorded prices.")


class CostSummary(ApiModel):
    """What the results cost, counting answers and verdicts reused from the on-disk cache at the
    cost they had when first made."""

    total_usd: float = Field(ge=0, description="Total cost in USD.")
    max_usd: float | None = Field(
        default=None,
        ge=0,
        description="Spending cap per command (--max-usd); null when the report does not know "
        "it (a report rebuilt from the caches).",
    )
    lines: list[CostLine] = Field(default_factory=list, description="Usage per stage and model.")


# --- Per question -------------------------------------------------------------------------------


class RunScores(ApiModel):
    """One question's retrieval scores under one run."""

    first_relevant_rank: int | None = Field(
        default=None, ge=1, description="Rank of the first gold chunk in the top 10; null if none."
    )
    recall_at_5: float | None = Field(
        default=None, ge=0, le=1, description=_PER_QUESTION.format(what="Recall@5")
    )
    recall_at_10: float | None = Field(
        default=None, ge=0, le=1, description=_PER_QUESTION.format(what="Recall@10")
    )
    mrr_at_10: float | None = Field(
        default=None, ge=0, le=1, description=_PER_QUESTION.format(what="Reciprocal rank")
    )
    ndcg_at_10: float | None = Field(
        default=None, ge=0, le=1, description=_PER_QUESTION.format(what="nDCG@10")
    )
    doc_recall_at_5: float | None = Field(
        default=None, ge=0, le=1, description=_PER_QUESTION.format(what="Document recall@5")
    )
    doc_recall_at_10: float | None = Field(
        default=None, ge=0, le=1, description=_PER_QUESTION.format(what="Document recall@10")
    )


class ClaimVerdict(ApiModel):
    """The judge's verdict on one claim extracted from an answer."""

    claim: str = Field(description="The claim, as the judge extracted it.")
    citations: list[int] = Field(
        default_factory=list, description="Citation numbers [n] attached to the claim."
    )
    supported: bool = Field(description="Supported by the passages the claim cites.")
    reason: str | None = Field(default=None, description="The judge's short justification.")


class CitationVerdict(ApiModel):
    """The judge's verdict on one citation marker."""

    n: int = Field(ge=1, description="Citation number: [n] refers to the answer's citations[n-1].")
    sentence: str = Field(description="The sentence the marker is attached to.")
    supported: bool = Field(description="The cited passage supports that sentence.")
    reason: str | None = Field(default=None, description="The judge's short justification.")


class FactVerdict(ApiModel):
    """The judge's verdict on one required fact of the question."""

    fact: str = Field(description="The required fact.")
    covered: bool = Field(description="The answer states it (judged against the reference).")
    reason: str | None = Field(default=None, description="The judge's short justification.")


class RefusalVerdict(ApiModel):
    """Whether the answer declines, and whether it fabricates."""

    declined: bool = Field(description="The answer says the passages do not answer the question.")
    fabricated: bool = Field(
        default=False, description="It asserts facts the passages do not contain."
    )
    reason: str | None = Field(default=None, description="The judge's short justification.")


class JudgeVerdicts(ApiModel):
    """Everything the judge said about one answer, and the per-answer scores derived from it."""

    faithfulness: float | None = Field(
        ge=0, le=1, description="Supported claims / claims; null for an answer without claims."
    )
    citation_accuracy: float | None = Field(
        ge=0, le=1, description="Supporting citations / citations; null without citations."
    )
    answer_correctness: float | None = Field(
        ge=0, le=1, description="Covered facts / required facts; null for unanswerable questions."
    )
    refusal_correct: bool | None = Field(
        description="Unanswerable question: declined without fabricating. Null if answerable."
    )
    false_refusal: bool | None = Field(
        description="Answerable question: declined anyway. Null if unanswerable."
    )
    claims: list[ClaimVerdict] = Field(default_factory=list, description="Per-claim verdicts.")
    citations: list[CitationVerdict] = Field(
        default_factory=list, description="Per-citation verdicts."
    )
    facts: list[FactVerdict] = Field(default_factory=list, description="Per-fact verdicts.")
    refusal: RefusalVerdict | None = Field(default=None, description="The refusal verdict.")


class RelevantChunk(ApiModel):
    """A gold chunk of a question."""

    chunk_id: str = Field(description="Chunk id in the search index ({doc_id}#{index:04d}).")
    role: ChunkRole = Field(
        description='"primary": answers the question (gain 2); "supporting": helps (gain 1).'
    )


class RetrievedRecord(ApiModel):
    """One retrieved chunk, in rank order (file form)."""

    chunk_id: str = Field(description="Chunk id in the search index.")
    score: float | None = Field(default=None, description="Retrieval score, higher is better.")


class AnswerRecord(ApiModel):
    """The answer the production chain gave under one run (file form)."""

    text: str = Field(description="The answer with [n] markers: [n] is citations[n-1].")
    citations: list[str] = Field(description="Chunk ids of the cited passages, in [n] order.")
    notes: list[str] = Field(default_factory=list, description="System notes, as in AskResponse.")
    latency_ms: float | None = Field(
        default=None, ge=0, description="End-to-end latency, ms (when first generated)."
    )
    cost_usd: float | None = Field(default=None, ge=0, description="OpenAI cost of the answer.")
    judge: JudgeVerdicts | None = Field(default=None, description="Verdicts; null if unjudged.")


class RunRecord(ApiModel):
    """One question under one run (file form)."""

    method: RetrievalMethod = Field(description="Retrieval method.")
    config: ConfigId = Field(description=_CONFIG)
    latency_ms: float | None = Field(default=None, ge=0, description="Retrieval latency, ms.")
    scores: RunScores = Field(description="Retrieval scores.")
    retrieved: list[RetrievedRecord] = Field(
        max_length=MAX_RETRIEVED,
        validation_alias=AliasChoices("retrieved", "hits"),
        description="Retrieved chunks in rank order (the top 10).",
    )
    answer: AnswerRecord | None = Field(
        default=None, description="The generated answer; null when this run was not answered."
    )


class _QuestionFields(ApiModel):
    id: QuestionId = Field(description="Question id.")
    question: str = Field(description="The question, as a user would ask it.")
    as_of: dt.date | None = Field(default=None, description="Point-in-time cutoff, if any.")
    type: QuestionType = Field(description="Question type.")
    split: QuestionSplit = Field(description="dev or test.")
    answerable: bool = Field(description="The corpus can answer it.")
    human_reviewed: bool = Field(default=False, description="A human reviewed its labels.")


class _QuestionLabels(_QuestionFields):
    reference_answer: str | None = Field(default=None, description="Reference answer.")
    required_facts: list[str] = Field(
        default_factory=list, description="Facts a correct answer states."
    )
    relevant_doc_ids: list[str] = Field(
        default_factory=list, description="Documents that answer it."
    )
    notes: str | None = Field(default=None, description="Labelling notes.")


class QuestionRecord(_QuestionLabels):
    """One question and every run on it (file form)."""

    relevant_chunks: list[RelevantChunk] = Field(
        default_factory=list, description="Gold chunks; empty for unanswerable questions."
    )
    runs: list[RunRecord] = Field(description="One entry per method and config.")

    @model_validator(mode="after")
    def _runs_unique(self) -> Self:
        _require_unique(self.runs, lambda r: (r.method, r.config), f"run of {self.id}")
        return self


class RunSummary(ApiModel):
    """One question's outcome under one run, for the question list."""

    method: RetrievalMethod = Field(description="Retrieval method.")
    config: ConfigId = Field(description=_CONFIG)
    first_relevant_rank: int | None = Field(description="As RunScores.")
    recall_at_5: float | None = Field(description="As RunScores.")
    ndcg_at_10: float | None = Field(description="As RunScores.")
    answered: bool = Field(description="An answer was generated under this run.")
    faithfulness: float | None = Field(description="As JudgeVerdicts; null if not judged.")
    citation_accuracy: float | None = Field(description="As JudgeVerdicts.")
    answer_correctness: float | None = Field(description="As JudgeVerdicts.")
    refusal_correct: bool | None = Field(description="As JudgeVerdicts.")
    false_refusal: bool | None = Field(description="As JudgeVerdicts.")
    declined: bool | None = Field(
        default=None,
        description="The judge read the answer as declining (RefusalVerdict.declined); null if "
        "not judged. Tells a decline the judge still faults (an unsupported claim beside it) "
        "from an answer to an unanswerable question.",
    )


class QuestionSummary(_QuestionFields):
    """A question and its outcome under every run, without passages or answers."""

    runs: list[RunSummary] = Field(description="One entry per method and config.")


class EvalPassage(ApiModel):
    """A chunk of the search index, as the reader sees it."""

    chunk_id: str = Field(description="Chunk id.")
    document: SourceDocument = Field(description="Document the chunk belongs to.")
    section: str | None = Field(description="Minutes section; null for statements.")
    text: str = Field(description="The passage text.")


class RetrievedHit(ApiModel):
    """A retrieved chunk with its hit marker."""

    rank: int = Field(ge=1, description="1-based rank.")
    chunk_id: str = Field(description="Chunk id; its text is QuestionDetail.passages[chunk_id].")
    doc_id: str = Field(description="Document id (the part of the chunk id before '#').")
    score: float | None = Field(description="Retrieval score, comparable only within a method.")
    role: ChunkRole | None = Field(
        description="Gold role of the chunk; null when it is not a gold chunk."
    )
    doc_match: bool = Field(description="The chunk is from one of the relevant documents.")


class AnswerCitation(ApiModel):
    """A passage an answer cites."""

    n: int = Field(ge=1, description="Citation number: [n] in the answer.")
    chunk_id: str = Field(description="Chunk id; its text is QuestionDetail.passages[chunk_id].")
    rank: int | None = Field(description="Its rank in the run's retrieved list; null if absent.")
    role: ChunkRole | None = Field(description="Gold role; null when not a gold chunk.")
    doc_match: bool = Field(description="From one of the relevant documents.")


class AnswerDetail(ApiModel):
    """An answer with its citations resolved and the judge's verdicts."""

    text: str = Field(description="The answer with [n] markers.")
    citations: list[AnswerCitation] = Field(description="Cited passages in [n] order.")
    notes: list[str] = Field(description="System notes, as in AskResponse.")
    latency_ms: float | None = Field(description="End-to-end latency, ms.")
    cost_usd: float | None = Field(description="OpenAI cost of the answer.")
    judge: JudgeVerdicts | None = Field(description="The judge's verdicts; null if unjudged.")


class RunDetail(ApiModel):
    """One question under one run: what was retrieved and what was answered."""

    method: RetrievalMethod = Field(description="Retrieval method.")
    config: ConfigId = Field(description=_CONFIG)
    latency_ms: float | None = Field(description="Retrieval latency, ms.")
    scores: RunScores = Field(description="Retrieval scores.")
    retrieved: list[RetrievedHit] = Field(description="Retrieved chunks in rank order.")
    answer: AnswerDetail | None = Field(description="The answer; null when not generated.")


class QuestionDetail(_QuestionLabels):
    """One evaluation question in full: gold chunks, each run's retrieval and answer, and the
    text of every chunk they refer to."""

    gold: list[RelevantChunk] = Field(description="Gold chunks, primary first.")
    runs: list[RunDetail] = Field(description="One entry per method and config.")
    passages: dict[str, EvalPassage] = Field(
        description=(
            "Every gold, retrieved or cited chunk by chunk id, each once; a chunk the local "
            "search index does not have is missing."
        )
    )
    passages_available: bool = Field(
        description="The search index could be read here (make index); without it, passages is "
        "empty."
    )


# --- The report and the file -------------------------------------------------------------------


class _ReportFields(ApiModel):
    generated_at: dt.datetime = Field(
        description="When the newest result behind the report was made (the latest cached "
        "retrieval, answer or judgment), so a rebuild with nothing new keeps it."
    )
    dataset: EvalDataset = Field(description="The question set.")
    configs: list[RetrieverConfig] = Field(
        min_length=1, description="Retriever configs compared, in display order."
    )
    generator: GeneratorInfo | None = Field(
        default=None, description="How answers were generated; null without answers."
    )
    judge: JudgeInfo | None = Field(default=None, description="The judge; null without answers.")
    bootstrap: BootstrapInfo | None = Field(
        default=None, description="How intervals were computed; null without intervals."
    )
    selection: ConfigSelection | None = Field(
        default=None, description="The run chosen and the rule; null when none was chosen."
    )
    retrieval: list[RetrievalRow] = Field(description="One row per method, config and split.")
    answers: list[AnswerRow] = Field(
        default_factory=list, description="One row per answered method, config and split."
    )
    per_type: list[TypeRow] = Field(
        default_factory=list, description="One row per method, config, split and question type."
    )
    differences: list[EvalDifference] = Field(
        default_factory=list, description="Paired bootstrap differences between runs."
    )
    cost: CostSummary | None = Field(default=None, description="OpenAI usage and cost.")
    notes: list[str] = Field(default_factory=list, description="Caveats the run recorded.")

    @model_validator(mode="after")
    def _rows_consistent(self) -> Self:
        _require_unique(self.configs, lambda c: c.id, "config")
        _require_unique(self.retrieval, lambda r: (r.method, r.config, r.split), "retrieval row")
        _require_unique(self.answers, lambda r: (r.method, r.config, r.split), "answer row")
        _require_unique(
            self.per_type, lambda r: (r.method, r.config, r.split, r.type), "per-type row"
        )
        known = {c.id for c in self.configs}
        keys: list[RunKey | RetrievalRow | AnswerRow | TypeRow | ConfigSelection] = [
            *self.retrieval,
            *self.answers,
            *self.per_type,
            *(key for d in self.differences for key in (d.minuend, d.subtrahend)),
            *([self.selection] if self.selection else []),
        ]
        _require_known_configs((k.config for k in keys), known)
        return self


class EvaluationReport(_ReportFields):
    """Results of the RAG evaluation: retrieval and answer quality per retrieval method and
    retriever config, with bootstrap intervals, per question type and per split."""

    questions: list[QuestionSummary] = Field(description="Every question, summarised.")


class EvaluationFile(_ReportFields):
    """data/eval/rag_benchmark.json: the report fields plus every question in full. Not served
    as is: the API summarises the questions and serves each one on its own."""

    questions: list[QuestionRecord] = Field(description="Every question with every run.")

    @model_validator(mode="after")
    def _questions_consistent(self) -> Self:
        _require_unique(self.questions, lambda q: q.id, "question id")
        known = {c.id for c in self.configs}
        _require_known_configs((r.config for q in self.questions for r in q.runs), known)
        return self


def _require_unique(items: Iterable[Any], key: Callable[[Any], object], what: str) -> None:
    seen: set[object] = set()
    for item in items:
        k = key(item)
        if k in seen:
            raise ValueError(f"duplicate {what}: {k}")
        seen.add(k)


def _require_known_configs(configs: Iterable[str], known: set[str]) -> None:
    unknown = sorted(set(configs) - known)
    if unknown:
        raise ValueError(f"unknown config(s) {', '.join(unknown)}; configs lists {sorted(known)}")
