"""The RAG benchmark file (data/eval/rag_benchmark.json) and the evaluation CLI's tables.

The file is an app.schemas.evaluation.EvaluationFile, the contract GET /api/evaluation serves
from; write_report() checks it with app.api.evaluation.evaluation_file_from_json before writing.
It is tracked in git: metrics, configuration, costs and per-question rows (answer texts, chunk
ids, the judge's verdicts; the corpus is public), never a secret. Floats keep 5 significant
digits. `report` rebuilds it from the caches under data/eval_cache without any API call.

How build_report() fills it, for every split (dev, test, all), retriever config and method:

- retrieval: one RetrievalRow when every answerable question of the split has a cached ranking:
  the metrics of app.evaluation.retrieval_metrics (tie-aware) as Estimates (mean, 95% bootstrap
  interval over questions, n where defined), retrieval latency p50/p95 and as_of leaks;
- answers: one AnswerRow when every question of the split has a cached answer and judgment:
  faithfulness, citation_accuracy, answer_correctness, refusal_accuracy (the mean of
  refusal_correct over unanswerable questions) and false_refusal_rate (of false_refusal over
  answerable ones), end-to-end latency, and the mean generation and judge cost per question;
  citation_accuracy_strict (full support only) is written alongside;
- per_type: the same Estimates per question type, where the split's sections exist;
- differences: paired bootstrap differences for every pair of methods within a config (the later
  of bm25, dense, hybrid, hybrid_rerank minus the earlier) and for every config against the first,
  on every metric both runs have;
- questions: every question with its labels and, per run, the top 10 retrieved chunks, the
  per-question retrieval scores, the answer and the judge's verdicts (claims with the citation
  numbers of their sentence, citation checks with the text each marker is attached to, required
  facts, refusal), each verdict also carrying the judge's three-way label;
- cost: generation, judge and query-embedding usage of the cached calls behind the results, each
  answer and judgment counted once (a rerun served from the cache costs nothing more);
- selection: with --select METRIC, the run with the highest dev mean of that metric; with
  --switch-rule, the default-switch rule decided before the TEST run (switch_selection) applied
  to the test split's paired differences of the first two configs.

A row appears only once its split is complete; coverage() says how far each run got.
"""

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.evaluation.judge import (
    ANSWER_METRICS,
    annotator_agreement,
    annotator_sets,
    compare_labels,
    human_labels,
    judge_labels,
    model_usd,
)
from app.evaluation.questions import EvalQuestion, QuestionSet, SplitChoice
from app.evaluation.retrieval_metrics import (
    RETRIEVAL_METRICS,
    RankedHit,
    bootstrap_method,
    latency_summary,
    mean_ci,
    paired_difference,
    question_metrics,
)
from app.evaluation.runner import METHODS, RetrieverSetup, answer_usd
from app.ingest.records import write_text_atomic
from app.stance.metrics import DEFAULT_BOOTSTRAP, Bootstrap
from app.stance.report import rounded

BENCHMARK_FILE = Path("eval") / "rag_benchmark.json"  # under settings.data_dir
SPLIT_ORDER: tuple[SplitChoice, ...] = ("dev", "test", "all")
TOP_RETRIEVED = 10  # retrieved chunks kept per run in the file
USAGE_FIELDS = ("input_tokens", "cached_input_tokens", "output_tokens")
EMBEDDING_USD_PER_MILLION = {"text-embedding-3-small": 0.02, "text-embedding-3-large": 0.13}
# Per-answer metric -> the name of its mean in AnswerRow, TypeRow and EvalDifference.
ROW_NAMES = {
    "faithfulness": "faithfulness",
    "citation_accuracy": "citation_accuracy",
    "citation_accuracy_strict": "citation_accuracy_strict",
    "answer_correctness": "answer_correctness",
    "refusal_correct": "refusal_accuracy",
    "false_refusal": "false_refusal_rate",
}
COMPARED = (*RETRIEVAL_METRICS, *(m for m in ANSWER_METRICS if m != "citation_accuracy_strict"))
TIE_NOTE = (
    "Retrieval metrics are expected values over the orders of tied scores (McSherry and Najork, "
    "2008): fused rankings often tie, and the fusion's tie-break is arbitrary."
)
LATENCY_NOTE = (
    "Retrieval latency is measured per question after a warm-up query, with the query embedding "
    "served from a local cache: the OpenAI embedding round trip that dense, hybrid and "
    "hybrid_rerank pay in the API is not included. Answer latency adds the LLM call."
)
JUDGE_NOTE = (
    "Answer metrics come from an LLM judge, not from people. The judge check compares its "
    "verdicts with annotators' labels on a blind sample and says which annotators worked in "
    "isolation."
)
# Verdicts found wrong by reading tuned/hybrid_rerank TEST answers against their passages
# (docs/technical-report.md, "TEST results: answers"). The metrics keep the judge's verdicts; the
# note is added while the report holds those judgments (the same prompt version, the same questions
# judged).
KNOWN_ERRORS_PROMPT = "rag-judge-v1.f9315670"
KNOWN_ERRORS_RUN = {"method": "hybrid_rerank", "config": "tuned"}
KNOWN_ERRORS_IDS = ("ua-03", "pp-03", "sf-05", "pp-01")
KNOWN_ERRORS_NOTE = (
    "Known judge errors, found by reading tuned answers against their passages; the metrics keep "
    "the judge's verdicts. ua-03: a correct decline marked as fabricated (with it corrected, "
    "tuned refusal accuracy is 5/5 and faithfulness 0.956). pp-03: an answer about the wrong "
    "meeting credited 0.5 correctness (should be 0). sf-05: two citations marked as not "
    "supporting a sentence they support together. pp-01 is under review: its required facts "
    "bundle the step schedule, so an answer with the asked caps scores 0."
)
CONSTRUCTION_NOTE = "converted on the fly from data/eval/smoke.jsonl: document labels only"
# The default-switch rule, decided before the TEST run (self-recorded: no timestamped copy
# predates it; see docs/rag-tuning.md): switch the production default from the incumbent config
# to the candidate only if, on TEST, (a) the paired 95% CI of SWITCH_METHOD's nDCG@10 (candidate
# minus incumbent) lies entirely above 0 and (b) its answer_correctness difference has a CI
# lower bound above SWITCH_CORRECTNESS_MARGIN.
SWITCH_METHOD = "hybrid_rerank"
SWITCH_SPLIT: SplitChoice = "test"
SWITCH_CORRECTNESS_MARGIN = -0.05


@dataclass
class RunData:
    """The cached results of one retriever configuration and method, by question id."""

    setup: RetrieverSetup
    method: str
    retrieval: dict[str, dict[str, Any]] = field(default_factory=dict)  # rows of runner
    answers: dict[str, dict[str, Any]] = field(default_factory=dict)  # records of AnswerRunner
    # {"scores": judge.score_judgment(...), "verdicts": judge.verdicts(...), "usage", "model",
    #  "key": the judgment's cache key}
    judged: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.setup.name}/{self.method}"

    def key(self) -> dict[str, str]:
        return {"method": self.method, "config": self.setup.name}


def hits_of(row: Mapping[str, Any]) -> list[RankedHit]:
    return [RankedHit(h["chunk_id"], h["doc_id"], float(h["score"])) for h in row["hits"]]


def first_relevant_rank(hits: Sequence[Mapping[str, Any]], question: EvalQuestion) -> int | None:
    """1-based rank of the first relevant chunk in the stored order (top 10), or None."""
    grades = question.grades
    top = hits[:TOP_RETRIEVED]
    return next((rank for rank, h in enumerate(top, start=1) if h["chunk_id"] in grades), None)


def question_values(run: RunData, question: EvalQuestion) -> dict[str, Any]:
    """Every metric of `question` under `run` that its cached data defines (missing data: the
    metric is absent, not None)."""
    values: dict[str, Any] = {}
    row = run.retrieval.get(question.id)
    if row is not None and question.answerable:
        values |= question_metrics(hits_of(row), question)
    judged = run.judged.get(question.id)
    if judged is not None:
        values |= {name: judged["scores"][name] for name in ANSWER_METRICS}
    return values


def split_questions(questions: Sequence[EvalQuestion], split: SplitChoice) -> list[EvalQuestion]:
    return [q for q in questions if split == "all" or q.split == split]


def coverage(run: RunData, questions: Sequence[EvalQuestion]) -> dict[str, int]:
    """How many of `questions` have each kind of cached result under `run`."""
    answerable = [q for q in questions if q.answerable]
    return {
        "questions": len(questions),
        "answerable": len(answerable),
        "retrieval": sum(q.id in run.retrieval for q in answerable),
        "answers": sum(q.id in run.answers for q in questions),
        "judged": sum(q.id in run.judged for q in questions),
    }


def retrieval_complete(run: RunData, questions: Sequence[EvalQuestion]) -> bool:
    answerable = [q for q in questions if q.answerable]
    return bool(answerable) and all(q.id in run.retrieval for q in answerable)


def answers_complete(run: RunData, questions: Sequence[EvalQuestion]) -> bool:
    return bool(questions) and all(q.id in run.judged for q in questions)


def retrieval_row(
    run: RunData,
    questions: Sequence[EvalQuestion],
    split: SplitChoice,
    values: Mapping[str, Mapping[str, Any]],
    bootstrap: Bootstrap = DEFAULT_BOOTSTRAP,
) -> dict[str, Any] | None:
    """The RetrievalRow of `run` on `questions` (a split), or None until it is complete."""
    if not retrieval_complete(run, questions):
        return None
    answerable = [q for q in questions if q.answerable]
    rows = [run.retrieval[q.id] for q in answerable]
    latency = latency_summary([r["latency_ms"] for r in rows]) or {}
    return {
        **run.key(),
        "split": split,
        "n_questions": len(answerable),
        **{m: mean_ci([values[q.id][m] for q in answerable], bootstrap) for m in RETRIEVAL_METRICS},
        "latency_ms_p50": latency.get("p50"),
        "latency_ms_p95": latency.get("p95"),
        "as_of_leaks": sum(
            int(run.retrieval[q.id].get("leaks") or 0) for q in questions if q.id in run.retrieval
        ),
    }


def answer_row(
    run: RunData,
    questions: Sequence[EvalQuestion],
    split: SplitChoice,
    values: Mapping[str, Mapping[str, Any]],
    bootstrap: Bootstrap = DEFAULT_BOOTSTRAP,
) -> dict[str, Any] | None:
    """The AnswerRow of `run` on `questions` (a split), or None until it is complete."""
    if not answers_complete(run, questions):
        return None
    records = [run.answers[q.id] for q in questions]
    judged = [run.judged[q.id] for q in questions]
    latency = latency_summary([r["latency_ms"] for r in records]) or {}
    judge_costs = [_usage_usd(j.get("usage"), str(j.get("model", ""))) for j in judged]
    return {
        **run.key(),
        "split": split,
        "n_questions": len(questions),
        **{
            ROW_NAMES[m]: mean_ci([values[q.id].get(m) for q in questions], bootstrap)
            for m in ANSWER_METRICS
        },
        "latency_ms_p50": latency.get("p50"),
        "latency_ms_p95": latency.get("p95"),
        "cost_per_query_usd": _mean_finite([answer_usd(r) for r in records]),
        "judge_cost_per_query_usd": _mean_finite(judge_costs),
        "response_types": dict(Counter(j["scores"]["response_type"] for j in judged)),
        "passages_differ_from_retrieval": sum(
            _passages_differ(run.retrieval.get(q.id), r)
            for q, r in zip(questions, records, strict=True)
        ),
    }


def stale_answers(run: RunData) -> int:
    """How many of the run's cached answers were generated from passages that are not the first
    hits of its cached retrieval (the retrieval was re-run since, or the answer key missed a
    change): their verdicts describe other passages than the ones the report shows."""
    return sum(_passages_differ(run.retrieval.get(qid), r) for qid, r in run.answers.items())


def _passages_differ(row: Mapping[str, Any] | None, record: Mapping[str, Any]) -> bool:
    """Whether the answer's passages are not the first hits of the retrieval run (both come from
    the same retriever, so they should be)."""
    if row is None:
        return False
    passages = [p["chunk_id"] for p in record["passages"]]
    return [h["chunk_id"] for h in row["hits"][: len(passages)]] != passages


def type_rows(
    run: RunData,
    questions: Sequence[EvalQuestion],
    split: SplitChoice,
    values: Mapping[str, Mapping[str, Any]],
    bootstrap: Bootstrap = DEFAULT_BOOTSTRAP,
) -> list[dict[str, Any]]:
    """One TypeRow per question type, with the metrics of the sections that are complete."""
    retrieval = retrieval_complete(run, questions)
    answers = answers_complete(run, questions)
    if not retrieval and not answers:
        return []
    rows = []
    for kind in dict.fromkeys(q.type for q in questions):
        mine = [q for q in questions if q.type == kind]
        row: dict[str, Any] = {**run.key(), "split": split, "type": kind, "n_questions": len(mine)}
        if retrieval:
            for m in RETRIEVAL_METRICS:
                row[m] = mean_ci([values[q.id].get(m) for q in mine], bootstrap)
        if answers:
            for m in ANSWER_METRICS:
                row[ROW_NAMES[m]] = mean_ci([values[q.id].get(m) for q in mine], bootstrap)
        rows.append(row)
    return rows


def run_pairs(runs: Sequence[RunData]) -> list[tuple[RunData, RunData]]:
    """(minuend, subtrahend) pairs: within each config, the later method minus the earlier; then
    each config's run minus the first config's run of the same method."""
    order = {method: i for i, method in enumerate(METHODS)}
    configs = list(dict.fromkeys(run.setup.name for run in runs))
    pairs: list[tuple[RunData, RunData]] = []
    for config in configs:
        mine = sorted(
            (r for r in runs if r.setup.name == config), key=lambda r: order.get(r.method, 99)
        )
        pairs += [(mine[j], mine[i]) for i in range(len(mine)) for j in range(i + 1, len(mine))]
    for run in runs:
        if run.setup.name != configs[0]:
            base = next(
                (r for r in runs if r.setup.name == configs[0] and r.method == run.method), None
            )
            if base is not None:
                pairs.append((run, base))
    return pairs


def differences(
    runs: Sequence[RunData],
    questions: Sequence[EvalQuestion],
    splits: Sequence[SplitChoice],
    values: Mapping[str, Mapping[str, Mapping[str, Any]]],
    bootstrap: Bootstrap = DEFAULT_BOOTSTRAP,
) -> list[dict[str, Any]]:
    """EvalDifference rows (see the module docstring) where both runs' sections are complete."""
    rows = []
    for split in splits:
        mine = split_questions(questions, split)
        if not mine:
            continue
        for a, b in run_pairs(runs):
            parts = {
                "retrieval": retrieval_complete(a, mine) and retrieval_complete(b, mine),
                "answers": answers_complete(a, mine) and answers_complete(b, mine),
            }
            for metric in COMPARED:
                if not parts["retrieval" if metric in RETRIEVAL_METRICS else "answers"]:
                    continue
                diff = paired_difference(
                    [values[a.label][q.id].get(metric) for q in mine],
                    [values[b.label][q.id].get(metric) for q in mine],
                    bootstrap,
                )
                if diff is not None:
                    rows.append(
                        {
                            "metric": ROW_NAMES.get(metric, metric),
                            "split": split,
                            "minuend": a.key(),
                            "subtrahend": b.key(),
                            **diff,
                        }
                    )
    return rows


def select_run(
    retrieval: Sequence[Mapping[str, Any]],
    answers: Sequence[Mapping[str, Any]],
    metric: str,
    runs: Sequence[RunData],
) -> dict[str, Any] | None:
    """The run with the highest dev mean of `metric` (ties: the earlier config, then the earlier
    method in the bm25, dense, hybrid, hybrid_rerank order); None without dev rows."""
    rows = [r for r in [*retrieval, *answers] if r["split"] == "dev" and metric in r]
    scored = [(r, (r[metric] or {}).get("mean")) for r in rows]
    scored = [(r, v) for r, v in scored if v is not None]
    if not scored:
        return None
    configs = list(dict.fromkeys(run.setup.name for run in runs))
    order = {method: i for i, method in enumerate(METHODS)}
    best, _ = max(
        scored,
        key=lambda rv: (rv[1], -configs.index(rv[0]["config"]), -order[rv[0]["method"]]),
    )
    return {
        "method": best["method"],
        "config": best["config"],
        "metric": metric,
        "split": "dev",
        "rule": (
            f"Highest mean {metric} on the dev split among the runs evaluated; ties go to the "
            "earlier config, then to the earlier method in bm25, dense, hybrid, hybrid_rerank "
            "order. Chosen before the test split was read."
        ),
    }


def _difference(
    rows: Sequence[Mapping[str, Any]], metric: str, minuend: Mapping, subtrahend: Mapping
) -> Mapping[str, Any] | None:
    return next(
        (
            d
            for d in rows
            if d["split"] == SWITCH_SPLIT
            and d["metric"] == metric
            and d["minuend"] == minuend
            and d["subtrahend"] == subtrahend
        ),
        None,
    )


def _shown(d: Mapping[str, Any]) -> str:
    return f"{d['difference']:+.3f} [{d['ci']['low']:+.3f}, {d['ci']['high']:+.3f}], n = {d['n']}"


def switch_selection(
    differences: Sequence[Mapping[str, Any]], incumbent: str, candidate: str
) -> dict[str, Any] | None:
    """The default-switch rule (see SWITCH_METHOD) on the report's paired
    differences: the candidate config if both conditions hold, else the incumbent, with the rule
    and its numbers in words. None when either difference is missing (no complete test rows)."""
    minuend = {"method": SWITCH_METHOD, "config": candidate}
    subtrahend = {"method": SWITCH_METHOD, "config": incumbent}
    ndcg = _difference(differences, "ndcg_at_10", minuend, subtrahend)
    correct = _difference(differences, "answer_correctness", minuend, subtrahend)
    if ndcg is None or correct is None:
        return None
    holds_a = ndcg["ci"]["low"] > 0
    holds_b = correct["ci"]["low"] > SWITCH_CORRECTNESS_MARGIN
    switch = holds_a and holds_b
    verdict = (
        f"Both hold, so {candidate} is chosen."
        if switch
        else f"The rule is not met, so {incumbent} stays the default."
    )
    rule = (
        "Default-switch rule, decided before the TEST run (self-recorded; no timestamped copy "
        "predates it): make "
        f"{candidate} the production default instead of {incumbent} only if, on TEST, (a) the "
        f"paired-bootstrap 95% CI of {SWITCH_METHOD} nDCG@10, {candidate} minus {incumbent}, "
        f"lies entirely above 0, and (b) {SWITCH_METHOD} answer_correctness is not worse "
        f"(paired difference, {candidate} minus {incumbent}, CI lower bound above "
        f"{SWITCH_CORRECTNESS_MARGIN:.2f}). TEST: (a) {_shown(ndcg)}: "
        f"{'holds' if holds_a else 'fails'}; (b) {_shown(correct)}: "
        f"{'holds' if holds_b else 'fails'}. {verdict} The {candidate} preset itself was frozen "
        "on DEV only (docs/rag-tuning.md); TEST only decided the switch."
    )
    return {
        "method": SWITCH_METHOD,
        "config": candidate if switch else incumbent,
        "metric": "ndcg_at_10",
        "split": SWITCH_SPLIT,
        "rule": rule,
    }


def unjudged_notes(runs: Sequence[RunData], questions: Sequence[EvalQuestion]) -> list[str]:
    """One note per run with answers that were generated but never judged: it has no answer row,
    and its answers show without verdicts."""
    notes = []
    for run in runs:
        answered = [q for q in questions if q.id in run.answers]
        unjudged = [q for q in answered if q.id not in run.judged]
        if unjudged:
            splits = "/".join(sorted({q.split for q in unjudged}))
            notes.append(
                f"{run.label}: {len(unjudged)} of {len(answered)} cached answers ({splits}) were "
                "not judged, so this run has no answer row there and its answers carry no "
                "verdicts."
            )
    return notes


def dataset(question_set: QuestionSet, questions: Sequence[EvalQuestion], label: str) -> dict:
    """The EvalDataset of `questions` (the questions the report covers)."""
    return {
        "file": label,
        "sha256": question_set.sha256,
        "n_questions": len(questions),
        "n_answerable": sum(q.answerable for q in questions),
        "n_with_as_of": sum(q.as_of is not None for q in questions),
        "n_human_reviewed": sum(_human_reviewed(q) for q in questions),
        "splits": [
            {
                "split": split,
                "n": sum(q.split == split for q in questions),
                "n_answerable": sum(q.split == split and q.answerable for q in questions),
            }
            for split in ("dev", "test")
        ],
        "types": [
            {
                "type": kind,
                "n": sum(q.type == kind for q in questions),
                "n_answerable": sum(q.type == kind and q.answerable for q in questions),
            }
            for kind in dict.fromkeys(q.type for q in questions)
        ],
        "construction": CONSTRUCTION_NOTE if question_set.format == "smoke" else None,
    }


def _human_reviewed(question: EvalQuestion) -> bool:
    return question.human_reviewed


def config_entry(setup: RetrieverSetup) -> dict[str, Any]:
    """The RetrieverConfig entry (API schema) of a setup: params are the config's scalar knobs
    (lists joined with ", ")."""
    description = setup.description if isinstance(setup.description, Mapping) else {}
    params = {
        key: ", ".join(map(str, value)) if isinstance(value, list | tuple) else value
        for key, value in description.items()
        if key not in ("name", "description")
    }
    return {
        "id": setup.name,
        "label": setup.name,
        "description": str(description.get("description") or ""),
        "hash": setup.hash,
        "params": params,
    }


def newest_result(runs: Sequence[RunData]) -> datetime | None:
    """When the newest cached result in `runs` was made (retrieval run_at, answer and judgment
    created_at), or None without any: the report's generated_at, so a rebuild from the same
    caches writes the same file."""
    stamps = [
        *(row.get("run_at") for run in runs for row in run.retrieval.values()),
        *(record.get("created_at") for run in runs for record in run.answers.values()),
        *(entry.get("created_at") for run in runs for entry in run.judged.values()),
    ]
    return max((datetime.fromisoformat(s) for s in stamps if s), default=None)


def build_report(
    question_set: QuestionSet,
    runs: Sequence[RunData],
    *,
    file_label: str,
    generator: Mapping[str, Any],
    judge: Mapping[str, Any],
    splits: Sequence[SplitChoice] = SPLIT_ORDER,
    questions: Sequence[EvalQuestion] | None = None,
    select: str | None = None,
    switch_rule: bool = False,
    embedding: Mapping[str, Any] | None = None,
    bootstrap: Bootstrap = DEFAULT_BOOTSTRAP,
    now: datetime | None = None,
) -> dict[str, Any]:
    """The EvaluationFile (see the module docstring) from the cached results in `runs`; pure.
    `questions` (default: the whole set) are the questions it covers; `embedding` is
    {"model", "texts", "tokens"} of the query vectors behind it. `select` (a metric) picks the
    best dev run; `switch_rule` applies the default-switch rule to the first two configs."""
    covered = list(question_set.questions if questions is None else questions)
    values = {run.label: {q.id: question_values(run, q) for q in covered} for run in runs}
    retrieval, answers, per_type = [], [], []
    for split in splits:
        mine = split_questions(covered, split)
        if not mine:
            continue
        for run in runs:
            v = values[run.label]
            if (row := retrieval_row(run, mine, split, v, bootstrap)) is not None:
                retrieval.append(row)
            if (row := answer_row(run, mine, split, v, bootstrap)) is not None:
                answers.append(row)
            per_type += type_rows(run, mine, split, v, bootstrap)
    configs = list({run.setup.name: run.setup for run in runs}.values())
    judged = {f"{run.label}/{qid}" for run in runs for qid in run.judged}
    stamp = (now or newest_result(runs) or datetime.now(UTC)).replace(microsecond=0).isoformat()
    compared = differences(runs, covered, splits, values, bootstrap)
    selection = select_run(retrieval, answers, select, runs) if select else None
    notes = [TIE_NOTE, LATENCY_NOTE, *([JUDGE_NOTE] if judged else [])]
    if judge.get("prompt_version") == KNOWN_ERRORS_PROMPT and any(
        run.key() == KNOWN_ERRORS_RUN and all(qid in run.judged for qid in KNOWN_ERRORS_IDS)
        for run in runs
    ):
        notes.append(KNOWN_ERRORS_NOTE)
    if switch_rule:
        if len(configs) < 2:
            raise ValueError("--switch-rule needs two configs: the incumbent, then the candidate")
        selection = switch_selection(compared, configs[0].name, configs[1].name)
        if selection is None:
            raise ValueError(
                f"--switch-rule needs complete {SWITCH_SPLIT} retrieval and judged answers of "
                f"{SWITCH_METHOD} under {configs[0].name} and {configs[1].name}"
            )
        notes.append(
            "Selection: the default-switch rule, decided before the TEST run and applied once to "
            "TEST (the rule and its numbers are recorded with the selection)."
        )
    notes += unjudged_notes(runs, covered)
    return {
        "generated_at": stamp,
        "dataset": dataset(question_set, covered, file_label),
        "configs": [config_entry(setup) for setup in configs],
        "generator": dict(generator) if any(run.answers for run in runs) else None,
        "judge": {**judge, "n_answers": len(judged)} if judged else None,
        "bootstrap": {
            "method": bootstrap_method(bootstrap),
            "resamples": bootstrap.resamples,
            "seed": bootstrap.seed,
            "level": bootstrap.level,
        },
        "selection": selection,
        "retrieval": retrieval,
        "answers": answers,
        "per_type": per_type,
        "differences": compared,
        "cost": cost_summary(runs, generator, judge, embedding),
        "notes": notes,
        "questions": [_question_record(q, runs, values) for q in covered],
    }


def _question_record(
    question: EvalQuestion,
    runs: Sequence[RunData],
    values: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    entries = []
    for run in runs:
        row = run.retrieval.get(question.id)
        record = run.answers.get(question.id)
        if row is None and record is None:
            continue
        hits = row["hits"] if row is not None else record["passages"]  # type: ignore[index]
        mine = values[run.label][question.id]
        judged = run.judged.get(question.id)
        answer = None
        if record is not None:
            usd = answer_usd(record)
            answer = {
                "text": record["answer"],
                "citations": record["citations"],
                "notes": record.get("notes") or [],
                "latency_ms": record["latency_ms"],
                "cost_usd": None if math.isnan(usd) else usd,
                "judge": judged["verdicts"] if judged else None,
            }
        entries.append(
            {
                **run.key(),
                "latency_ms": row["latency_ms"] if row is not None else None,
                "scores": {
                    "first_relevant_rank": first_relevant_rank(hits, question),
                    **{m: mine.get(m) for m in RETRIEVAL_METRICS},
                },
                "retrieved": [
                    {"chunk_id": h["chunk_id"], "score": h.get("score")}
                    for h in hits[:TOP_RETRIEVED]
                ],
                "answer": answer,
            }
        )
    return {
        "id": question.id,
        "question": question.question,
        "as_of": question.as_of.isoformat() if question.as_of else None,
        "type": question.type,
        "split": question.split,
        "answerable": question.answerable,
        "human_reviewed": _human_reviewed(question),
        "reference_answer": question.reference_answer or None,
        "required_facts": list(question.required_facts),
        "relevant_doc_ids": list(question.relevant_doc_ids),
        "notes": question.notes,
        "relevant_chunks": [
            c.model_dump(include={"chunk_id", "role"}) for c in question.relevant_chunks
        ],
        "runs": entries,
    }


def _usage_usd(usage: Mapping[str, Any] | None, model: str) -> float:
    usage = usage or {}
    return model_usd(
        model,
        int(usage.get("input_tokens") or 0),
        int(usage.get("cached_input_tokens") or 0),
        int(usage.get("output_tokens") or 0),
    )


def _cost_line(stage: str, model: str, usages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = {name: sum(int(u.get(name) or 0) for u in usages) for name in USAGE_FIELDS}
    usd = model_usd(
        model, total["input_tokens"], total["cached_input_tokens"], total["output_tokens"]
    )
    return {
        "stage": stage,
        "model": model,
        "calls": sum(int(u.get("calls") or 1) for u in usages),
        **total,
        "usd": 0.0 if math.isnan(usd) else usd,
    }


def cost_summary(
    runs: Sequence[RunData],
    generator: Mapping[str, Any],
    judge: Mapping[str, Any],
    embedding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """CostSummary of the cached calls behind the results, each answer and judgment once."""
    answers: dict[str, Mapping[str, Any]] = {}
    judged: dict[str, Mapping[str, Any]] = {}
    for run in runs:
        for qid, record in run.answers.items():
            if record.get("llm_called", True):
                answers[f"{run.label}/{qid}"] = record["usage"]
        for qid, entry in run.judged.items():
            if entry.get("usage") is not None:  # one judgment can serve identical answers
                judged[entry.get("key") or f"{run.label}/{qid}"] = entry["usage"]
    lines = []
    if answers:
        lines.append(
            _cost_line("generation", str(generator.get("model", "")), list(answers.values()))
        )
    if judged:
        lines.append(_cost_line("judge", str(judge.get("model", "")), list(judged.values())))
    if embedding and embedding.get("texts"):
        model = str(embedding["model"])
        tokens = int(embedding.get("tokens") or 0)
        lines.append(
            {
                "stage": "embedding",
                "model": model,
                "calls": int(embedding["texts"]),
                "input_tokens": tokens,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "usd": tokens * EMBEDDING_USD_PER_MILLION.get(model, 0.0) / 1_000_000,
            }
        )
    return {"total_usd": sum(line["usd"] for line in lines), "max_usd": None, "lines": lines}


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _mean_finite(values: Sequence[float]) -> float | None:
    if any(math.isnan(v) for v in values):
        return None
    return _mean(values)


CHECKS = {
    "claims": "claim_support",
    "citations": "citation_support",
    "facts": "fact_coverage",
    "response_type": "refusal",
}
VALIDATION_NOTE = (
    "Agreement is on pass/fail verdicts (claims supported; citations supporting fully or in part; "
    "facts covered; answer declines or not), exact agreement on the three-way labels; kappa is "
    "Cohen's, on pass/fail."
)


def _annotated(items: Sequence[Mapping[str, Any]]) -> set[str]:
    """Ids of the items with at least one label filled in."""
    labels = human_labels(items)
    return {item_id for by_unit in labels.values() for item_id, _ in by_unit}


def _rows(result: Mapping[str, Any] | None, **extra: Any) -> list[dict[str, Any]]:
    return [
        {
            "check": check,
            **extra,
            "n": r["n"],
            "agreement": r["pass_fail_agreement"],
            "exact_agreement": r["exact_agreement"],
            "kappa": r["kappa"],
        }
        for kind, check in CHECKS.items()
        if (r := (result or {}).get(kind))
    ]


def judge_validation(
    sample: Mapping[str, Any] | None,
    annotations: Mapping[str, Any] | None,
    key: Mapping[str, Any] | None,
    sample_file: str | None,
) -> dict[str, Any]:
    """The JudgeValidation entry (API schema) from the exported sample, the annotations file (one
    annotator or several: judge.annotator_sets) and the key with the judge's verdicts: judge vs
    each annotator, and each pair of annotators against each other, per kind of check, in the
    file's order (the main annotator first)."""
    if sample is None:
        return {"status": "not_started", "n_sampled": 0, "n_annotated": 0, "agreement": []}
    sets = annotator_sets(annotations)
    sampled = {str(item.get("item_id")) for item in sample.get("items", [])}
    done = [_annotated(s["items"]) & sampled for s in sets]
    several = len(sets) > 1
    rows: list[dict[str, Any]] = []
    if key is not None:
        judge = judge_labels(key)
        for entry in sets:
            result = compare_labels(judge, human_labels(entry["items"]))
            rows += _rows(result, annotator=entry["id"] if several else None)
    pairs = []
    for i, first in enumerate(sets):
        for second in sets[i + 1 :]:
            result = annotator_agreement(first["items"], second["items"])
            pairs += _rows(result, annotators=[first["id"], second["id"]])
    summary = (annotations or {}).get("annotator")
    if summary is None and sets:
        summary = "; ".join(
            f"{s['id']}: {s['annotator']}" if several else str(s["annotator"])
            for s in sets
            if s["annotator"]
        )
    runs = Counter(
        str(item.get("item_id", "")).rpartition("/")[0] for item in sample.get("items", [])
    )
    covered = ", ".join(f"{r} {n}" for r, n in sorted(runs.items()) if r)
    extra = (annotations or {}).get("note")
    return {
        "status": "complete" if any(done) else "pending",
        "n_sampled": len(sample.get("items", [])),
        "n_annotated": min(len(d) for d in done) if done else 0,
        "annotator": summary or None,
        "annotators": [
            {"id": s["id"], "description": s["annotator"], "isolated": s["isolated"]}
            for s in sets
            if several
        ],
        "sample_file": sample_file,
        "sampled_runs": _sampled_runs(runs),
        "agreement": rows,
        "inter_annotator": pairs,
        "note": " ".join(
            filter(
                None,
                [
                    VALIDATION_NOTE if rows or pairs else None,
                    f"Sampled answers by run: {covered}." if covered else None,
                    extra,
                ],
            )
        )
        or None,
    }


def _sampled_runs(runs: Mapping[str, int]) -> list[dict[str, Any]]:
    """{"method", "config", "n"} for each "config/method" run label of the sampled items."""
    rows = []
    for label, n in sorted(runs.items()):
        config, _, method = label.partition("/")
        if config and method in METHODS:
            rows.append({"method": method, "config": config, "n": n})
    return rows


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    """Write the file atomically after checking it with the API's own contract check
    (app.api.evaluation.evaluation_file_from_json)."""
    from app.api.evaluation import evaluation_file_from_json

    content = rounded(report)
    evaluation_file_from_json(content)
    write_text_atomic(path, json.dumps(content, indent=2, ensure_ascii=False) + "\n")


# --- Console tables -------------------------------------------------------------------------------


def _cell(stat: Mapping[str, Any] | None, *, ci: bool = True) -> str:
    if stat is None:
        return "-"
    if not ci:
        return f"{stat['mean']:.3f}"
    return f"{stat['mean']:.3f} [{stat['ci']['low']:.2f}, {stat['ci']['high']:.2f}]"


def _names(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Row labels: the method, with its config when the table has several configs."""
    several = len({r["config"] for r in rows}) > 1
    return [f"{r['config']}/{r['method']}" if several else r["method"] for r in rows]


def retrieval_table(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| Method | n | recall@5 | recall@10 | MRR@10 | nDCG@10 | doc recall@5 | doc recall@10 "
        "| p50 ms | p95 ms | leaks |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row, name in zip(rows, _names(rows), strict=True):
        cells = " | ".join(_cell(row[m]) for m in RETRIEVAL_METRICS)
        lines.append(
            f"| {name} | {row['n_questions']} | {cells} | {row['latency_ms_p50'] or 0:.0f} | "
            f"{row['latency_ms_p95'] or 0:.0f} | {row['as_of_leaks']} |"
        )
    return "\n".join(lines)


def answers_table(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| Method | n | faithfulness | citation acc. | correctness | refusal acc. | false refusal "
        "| p50 ms | $/query |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row, name in zip(rows, _names(rows), strict=True):
        cost = row["cost_per_query_usd"]
        lines.append(
            f"| {name} | {row['n_questions']} | {_cell(row['faithfulness'])} | "
            f"{_cell(row['citation_accuracy'])} | {_cell(row['answer_correctness'])} | "
            f"{_cell(row['refusal_accuracy'], ci=False)} | "
            f"{_cell(row['false_refusal_rate'], ci=False)} | {row['latency_ms_p50'] or 0:.0f} | "
            + ("-" if cost is None else f"{cost:.4f}")
            + " |"
        )
    return "\n".join(lines)


def comparison_table(rows: Sequence[Mapping[str, Any]], metrics: Sequence[str]) -> str:
    lines = ["| a - b | metric | difference [95% CI] | n |", "| --- | --- | ---: | ---: |"]
    configs = {run["config"] for row in rows for run in (row["minuend"], row["subtrahend"])}
    for row in rows:
        if row["metric"] not in metrics:
            continue
        a, b = row["minuend"], row["subtrahend"]
        within = f"{a['method']} - {b['method']}"
        names = (
            (f"{a['config']}: {within}" if len(configs) > 1 else within)
            if a["config"] == b["config"]
            else f"{a['config']} - {b['config']} ({a['method']})"
        )
        lines.append(
            f"| {names} | {row['metric']} | {row['difference']:+.3f} "
            f"[{row['ci']['low']:+.2f}, {row['ci']['high']:+.2f}] | {row['n']} |"
        )
    return "\n".join(lines)


def coverage_lines(runs: Sequence[RunData], questions: Sequence[EvalQuestion]) -> list[str]:
    """One line per run whose results are incomplete for `questions`."""
    lines = []
    for run in runs:
        c = coverage(run, questions)
        if c["retrieval"] < c["answerable"] or c["judged"] < c["questions"]:
            lines.append(
                f"{run.label}: retrieval {c['retrieval']}/{c['answerable']} answerable, answers "
                f"{c['answers']}/{c['questions']}, judged {c['judged']}/{c['questions']}"
            )
    return lines


def response_counts(scores: Sequence[Mapping[str, Any]]) -> str:
    counts = Counter(s["response_type"] for s in scores)
    return ", ".join(f"{kind} {counts[kind]}" for kind in ("answers", "partial", "declines"))


def validation_table(validation: Mapping[str, Any]) -> str:
    """Judge vs each annotator and annotator vs annotator, one line per kind of check."""
    lines = [
        "| compared | check | n | pass/fail agreement | exact agreement | kappa |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    isolated = {a["id"] for a in validation.get("annotators", []) if a.get("isolated")}
    rows = [
        (f"judge vs {r.get('annotator') or 'annotator'}", r)
        for r in validation.get("agreement", [])
    ] + [
        (" vs ".join(r["annotators"]) + _not_independent(r["annotators"], isolated), r)
        for r in validation.get("inter_annotator", [])
    ]
    for name, r in rows:
        exact = "-" if r.get("exact_agreement") is None else f"{r['exact_agreement']:.3f}"
        kappa = "-" if r.get("kappa") is None else f"{r['kappa']:.3f}"
        lines.append(
            f"| {name} | {r['check']} | {r['n']} | {r['agreement']:.3f} | {exact} | {kappa} |"
        )
    return "\n".join(lines)


def _not_independent(pair: Sequence[str], isolated: set[str]) -> str:
    """The label suffix " (not independent)" for a pair of annotators neither of which worked in
    isolation, when the file marks some annotator isolated; "" otherwise."""
    return " (not independent)" if isolated and not isolated & set(pair) else ""
