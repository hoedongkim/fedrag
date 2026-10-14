"""Command line for policy stance: python -m app.stance {fetch,train,benchmark,predict,...}."""

import argparse
import math
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.core.config import Settings, get_settings
from app.schemas.common import Stance, StanceClassifier
from app.schemas.stance import StatementBaselineName, StatementVariant
from app.stance.types import CLASSIFIERS, GOLD_FILE

if TYPE_CHECKING:
    from app.stance.llm import CostEstimate, Mode, Usage
    from app.stance.metrics import BaselineCall, Bootstrap, Reference, StatementCall
    from app.stance.types import SentenceClassifier, StatementPrediction

LLM_MODES: dict[str, "Mode"] = {"llm_zero_shot": "zero_shot", "llm_few_shot": "few_shot"}
DEFAULT_MAX_USD = 1.0
DEFAULT_CONCURRENCY = 4
RETRY_MARGIN = 0.2  # calls allowed beyond the estimate, for retries and partial batches
FED_FUNDS_FILE = Path("stance") / "fed_funds_target.csv"
COST_NOTE = (
    "OpenAI usage of the calls that filled data/stance/cache (git-ignored), added up per step; a "
    "rerun served from the cache adds nothing. USD at $0.75 input / $0.075 cached input / $4.50 "
    "output per million tokens (third-party price lists, October 2026). Excludes the 8 "
    "development calls (about $0.025) made while writing the prompts."
)
DESCRIPTIONS: dict[StanceClassifier, str] = {
    "keyword": (
        "Hawkish and dovish term lists learned from each seed's TDW train split (log-odds "
        "z-scores, top_k by grouped 5-fold CV on train), with negation; +1/0/-1 per sentence."
    ),
    "finbert": (
        "ProsusAI/finbert fine-tuned on each seed's TDW train split (10% of train held out for "
        "early stopping); p(hawkish) - p(dovish) per sentence."
    ),
    "llm_zero_shot": (
        "LLM with the labelling rubric as the task definition; sentences in numbered batches, "
        "statements judged whole."
    ),
    "llm_few_shot": (
        "As llm_zero_shot plus 4 labelled sentences per class from a TDW train split (never "
        "test sentences or gold statements)."
    ),
}
STATEMENT_METHOD = {
    "keyword": "statement split into sentences; procedural ones dropped; the 3 seed lexicons vote",
    "finbert": "statement split into sentences; procedural ones dropped; mean probs of 3 seeds",
    "llm_zero_shot": "whole statement in one call",
    "llm_few_shot": "whole statement in one call; examples from TDW seed 5768's train split",
}
POST_HOC_METHOD = (
    "post hoc, designed after the pre-registered results were seen: the same sentence labels, "
    "aggregated in the rubric's decision order (the rate-move sentence's label if hawkish or "
    "dovish, else the mean score of the rate-guidance sentences cut at +-0.15, else neutral; "
    "balance-sheet and risk-language steps left out)"
)
# Paired bootstrap comparisons: (minuend, subtrahend) as (classifier, variant).
PAIRED_COMPARISONS: tuple[tuple[tuple[StanceClassifier, StatementVariant], ...], ...] = (
    (("llm_zero_shot", "pre_registered"), ("llm_few_shot", "pre_registered")),
    (("finbert", "post_hoc_rubric_ordered"), ("finbert", "pre_registered")),
)
BASELINE_NOTES: dict[StatementBaselineName, str] = {
    "majority_class": "Always the most frequent reference label; no classifier.",
    "rate_move_rule": (
        "Post hoc, no classifier: the verb of the statement's rate-action sentence decides a "
        "move (raise or increase hawkish, lower, reduce or cut dovish), anything else is "
        "neutral. What the post-hoc rows get from detecting the move alone."
    ),
}
# Paired bootstrap comparisons of a row, as (classifier, variant), with a baseline.
BASELINE_COMPARISONS: tuple[
    tuple[tuple[StanceClassifier, StatementVariant], StatementBaselineName], ...
] = (
    (("finbert", "pre_registered"), "majority_class"),
    (("finbert", "post_hoc_rubric_ordered"), "rate_move_rule"),
    (("keyword", "post_hoc_rubric_ordered"), "rate_move_rule"),
)
VARIANT_NOTES: dict[StatementVariant, str] = {
    "pre_registered": (
        "The method fixed before any reference label existed. Every timeline, database and API "
        "default result is this variant."
    ),
    "post_hoc_rubric_ordered": POST_HOC_METHOD + ".",
}
COMMANDS = {
    "fetch": "download the TDW sentence splits and the FRED fed funds target range",
    "train": "fine-tune FinBERT on each TDW seed's train split (skips seeds already trained)",
    "benchmark": "sentence-level benchmark on TDW test splits (LLM classifiers call OpenAI)",
    "predict": "classify all FOMC statements (LLM classifiers call OpenAI)",
    "evaluate": "score the statement predictions against data/eval/stance_gold.jsonl",
    "load": "upsert the statement predictions and the fed funds range into Postgres",
}


class BudgetError(RuntimeError):
    """The estimated OpenAI cost is over --max-usd: nothing was called."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.stance",
        description="Classify FOMC statements hawkish / neutral / dovish and benchmark the "
        "classifiers.",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")
    subs = {
        name: commands.add_parser(name, help=text, description=text)
        for name, text in COMMANDS.items()
    }
    subs["train"].add_argument(
        "--retrain", action="store_true", help="fine-tune again even where a model is saved"
    )
    for name in ("benchmark", "predict"):
        subs[name].add_argument(
            "--classifiers",
            nargs="+",
            choices=CLASSIFIERS,
            default=list(CLASSIFIERS),
            help="classifiers to run (default: all four); results of the others are kept",
        )
        subs[name].add_argument(
            "--max-usd",
            type=float,
            default=DEFAULT_MAX_USD,
            help="refuse to start when the estimated OpenAI cost of uncached calls is above this "
            f"(default: {DEFAULT_MAX_USD})",
        )
        subs[name].add_argument(
            "--concurrency",
            type=int,
            default=DEFAULT_CONCURRENCY,
            help=f"OpenAI calls in flight at once (default: {DEFAULT_CONCURRENCY})",
        )
    subs["evaluate"].add_argument(
        "--gold", type=Path, help="reference labels (default: data/eval/stance_gold.jsonl)"
    )
    return parser


def main(argv: Sequence[str] | None = None, *, settings: Settings | None = None) -> int:
    """Run one command; returns the process exit code."""
    args = build_parser().parse_args(argv)
    settings = settings or get_settings()
    # Imported here so --help stays fast; these pull in torch, LangChain and the DB driver.
    import httpx2
    import openai
    import psycopg

    from app.db.connection import DatabaseNotConfiguredError, DatabaseUrlError
    from app.db.load import LoadError
    from app.rag.answer import LLMNotConfiguredError
    from app.stance.llm import CallLimitError, LLMReplyError

    commands: dict[str, Callable[[Settings, argparse.Namespace], int]] = {
        "fetch": run_fetch,
        "train": run_train,
        "benchmark": run_benchmark,
        "predict": run_predict,
        "evaluate": run_evaluate,
        "load": run_load,
    }
    try:
        return commands[args.command](settings, args)
    except (
        BudgetError,
        CallLimitError,
        LLMReplyError,
        LLMNotConfiguredError,
        LoadError,
        FileNotFoundError,
        DatabaseNotConfiguredError,
        DatabaseUrlError,
        psycopg.Error,
        httpx2.HTTPError,
        openai.OpenAIError,
        ValueError,  # a malformed input file; also pydantic's ValidationError
    ) as exc:
        message = str(exc)
        if isinstance(exc, openai.OpenAIError):  # its message can quote part of the API key
            detail = type(exc).__name__
            if (status := getattr(exc, "status_code", None)) is not None:
                detail += f", HTTP {status}"
            message = f"OpenAI request failed ({detail})"
        print(f"{args.command}: {message}", file=sys.stderr)
        return 1


def run_fetch(settings: Settings, args: argparse.Namespace) -> int:
    from app.stance import fred, tdw

    folder = tdw.download(settings)
    print(
        f"TDW: {len(tdw.SHA256)} split files verified in {folder} (commit {tdw.COMMIT[:7]}, "
        "CC BY-NC 4.0: git-ignored, never committed)"
    )
    points = fred.fetch_target_range()
    path = settings.data_dir / FED_FUNDS_FILE
    fred.write_change_points(path, points)
    print(
        f"FRED: {len(points)} target-range change points, {points[0].date} to "
        f"{points[-1].date} -> {path} (source: {fred.SOURCE})"
    )
    return 0


def run_train(settings: Settings, args: argparse.Namespace) -> int:
    from app.stance import finbert

    for report in finbert.train_seeds(settings=settings, force=args.retrain):
        best = report.val_macro_f1[report.best_epoch - 1]
        print(
            f"seed {report.seed}: best epoch {report.best_epoch} of {report.epochs_run}, "
            f"validation macro-F1 {best:.3f}, trained in {report.train_seconds:.0f} s on "
            f"{report.device}"
        )
    return 0


def run_benchmark(settings: Settings, args: argparse.Namespace) -> int:
    """TDW sentence benchmark of the chosen classifiers -> sentence_level of the results file."""
    from app.stance import metrics, report
    from app.stance.llm import Usage

    chosen = _ordered(args.classifiers)
    usage = Usage()
    llm_names = [name for name in chosen if name in LLM_MODES]
    if llm_names:
        print("OpenAI estimate (uncached sentences only):")
        calls = check_budget(sentence_estimates(settings, llm_names), args.max_usd)
        _require_llm(settings, calls)
        usage.max_calls = call_ceiling(calls)
    path = report.benchmark_path(settings)
    results = report.read_results(path)
    models = results.setdefault("models", {})
    entries = []
    for name in chosen:
        described: dict[str, Any] = {}
        factory = _described(sentence_factory(name, settings, usage, args.concurrency), described)
        started = time.perf_counter()
        entries.append(metrics.evaluate_sentences(factory, settings=settings))
        print(f"{name}: 3 seeds labelled in {time.perf_counter() - started:.1f} s")
        models[name] = models.get(name, {}) | {
            "description": DESCRIPTIONS[name],
            "sentence": described,
        }
    results["sentence_level"] = report.merge_section(results.get("sentence_level", []), entries)
    results["sentence_baselines"] = [metrics.majority_class_sentences(settings=settings)]
    report.record_cost(
        results, "sentence_benchmark", usage.summary(settings.llm_model), note=COST_NOTE
    )
    report.write_results(path, results)
    print("\nTDW sentence benchmark (mean +- sd over seeds; n = test sentences per seed):")
    print(report.sentence_table(results["sentence_level"]))
    print(report.sentence_baseline_table(results["sentence_baselines"]))
    _print_usage(usage, settings)
    print(f"\nWrote {path}")
    return 0


def run_predict(settings: Settings, args: argparse.Namespace) -> int:
    """Statement predictions of the chosen classifiers -> data/stance/predictions.jsonl."""
    from app.stance import report
    from app.stance import statements as st
    from app.stance.llm import LLMStatementClassifier, Usage

    chosen = _ordered(args.classifiers)
    documents = st.load_statements(settings)
    usage = Usage()
    judges = {
        name: LLMStatementClassifier(
            LLM_MODES[name], settings=settings, usage=usage, max_concurrency=args.concurrency
        )
        for name in chosen
        if name in LLM_MODES
    }
    if judges:
        print(f"OpenAI estimate ({len(documents)} statements, uncached only):")
        estimates = [(name, judge.estimate(documents)) for name, judge in judges.items()]
        calls = check_budget(estimates, args.max_usd)
        _require_llm(settings, calls)
        usage.max_calls = call_ceiling(calls)
    predictions: list[StatementPrediction] = []
    post_hoc: list[StatementPrediction] = []
    decided_by: dict[str, dict[str, int]] = {}
    for name in chosen:
        started = time.perf_counter()
        if name in judges:
            mine = judges[name].predict_statements(documents)
        else:  # one pass of sentence labels, aggregated both ways
            classifier = statement_classifier(name, settings)
            labelled = st.label_statements(classifier, documents)
            mine = st.aggregate_statements(classifier, labelled)
            post_hoc += st.aggregate_statements(
                classifier, labelled, variant="post_hoc_rubric_ordered"
            )
            steps = Counter(st.rubric_decision(i.sentences, i.predictions).step for i in labelled)
            decided_by[name] = {step: steps[step] for step in ("rate_action", "guidance", "none")}
        predictions += mine
        print(
            f"{name}: {len(mine)} statements in {time.perf_counter() - started:.1f} s "
            f"({_stance_counts(mine)})"
        )
        if name in decided_by:
            extra = [p for p in post_hoc if p.classifier == name]
            by_step = ", ".join(f"{step} {count}" for step, count in decided_by[name].items())
            print(f"  post hoc, rubric-ordered: {_stance_counts(extra)} (decided by {by_step})")
    written = _write_merged(st.predictions_path(settings), predictions, chosen)
    print(f"Wrote {written} predictions to {st.predictions_path(settings)}")
    if post_hoc:
        path = st.predictions_path(settings, "post_hoc_rubric_ordered")
        written = _write_merged(path, post_hoc, list(decided_by))
        print(f"Wrote {written} post-hoc predictions to {path}")

    results_path = report.benchmark_path(settings)
    results = report.read_results(results_path)
    models = results.setdefault("models", {})
    for name in chosen:
        version = next(p.model_version for p in predictions if p.classifier == name)
        models[name] = models.get(name, {}) | {
            "description": DESCRIPTIONS[name],
            "statement": {"method": STATEMENT_METHOD[name], "model_version": version},
        }
        if name in decided_by:
            models[name]["statement_post_hoc"] = {
                "method": POST_HOC_METHOD,
                "model_version": next(p.model_version for p in post_hoc if p.classifier == name),
                "decided_by": decided_by[name],
            }
    report.record_cost(
        results, "statement_predictions", usage.summary(settings.llm_model), note=COST_NOTE
    )
    report.write_results(results_path, results)
    _print_usage(usage, settings)
    return 0


def run_evaluate(settings: Settings, args: argparse.Namespace) -> int:
    """Statement predictions against the reference labels -> statement_level of the results."""
    import hashlib

    from app.stance import metrics, report
    from app.stance import statements as st

    predictions = _read_predictions(settings)
    gold_path = args.gold or settings.data_dir / GOLD_FILE
    results_path = report.benchmark_path(settings)
    results = report.read_results(results_path)
    if not gold_path.is_file():
        print(
            f"{gold_path} not found: statement-level evaluation skipped. Run it again once the "
            "reference labels exist."
        )
        results["statement_level"], results["references"] = [], None
        for key in ("statement_bootstrap", "statement_variants", "statement_baselines"):
            results.pop(key, None)
        report.write_results(results_path, results)
        return 0
    rows = metrics.read_gold(gold_path)
    refs = metrics.references(rows)
    bootstrap = metrics.Bootstrap()
    post_hoc_path = st.predictions_path(settings, "post_hoc_rubric_ordered")
    by_variant: dict[StatementVariant, list[StatementPrediction]] = {
        "pre_registered": predictions,
        "post_hoc_rubric_ordered": (
            st.read_predictions(post_hoc_path) if post_hoc_path.is_file() else []
        ),
    }
    entries = []
    rows_by_key: dict[tuple[str, str], list[StatementPrediction]] = {}
    for variant, variant_predictions in by_variant.items():
        for name in CLASSIFIERS:
            mine = [p for p in variant_predictions if p.classifier == name]
            if not mine:
                continue
            rows_by_key[(name, variant)] = mine
            entry = metrics.evaluate_statements(mine, refs, bootstrap)
            scored = {p.doc_id: refs[p.doc_id] for p in mine if p.doc_id in refs}
            entry["variant"] = variant
            entry["reference"] = report.reference_status(scored)
            entry["predicted"] = _counts(p.stance for p in mine)
            entry["model_version"] = mine[0].model_version
            entries.append(entry)
    paired: list[dict[str, Any]] = []
    paired_years: list[dict[str, Any]] = []
    for minuend, subtrahend in PAIRED_COMPARISONS:
        if minuend not in rows_by_key or subtrahend not in rows_by_key:
            continue
        keys = {"minuend": _row_key(minuend), "subtrahend": _row_key(subtrahend)}
        pair = rows_by_key[minuend], rows_by_key[subtrahend]
        paired += _paired(*pair, refs, bootstrap, keys)
        paired_years += _paired(*pair, refs, bootstrap, keys, year_blocks=True)
    calls = baseline_calls(settings, predictions, refs)
    baselines = [
        {
            "name": name,
            "label": mine[0].stance if name == "majority_class" else None,
            "post_hoc": name == "rate_move_rule",
            "description": BASELINE_NOTES[name],
            "predicted": _counts(c.stance for c in mine),
            **metrics.score_statements(mine, refs, bootstrap),
        }
        for name, mine in calls.items()
    ]
    against_baselines: list[dict[str, Any]] = []
    for row, name in BASELINE_COMPARISONS:
        if row in rows_by_key and name in calls:
            keys = {"row": _row_key(row), "baseline": name}
            against_baselines += _paired(rows_by_key[row], calls[name], refs, bootstrap, keys)
    n_human = sum(ref.source == "human" for ref in refs.values())
    agreement = metrics.gold_agreement(rows)
    results["statement_level"] = entries
    results["statement_baselines"] = baselines
    results["statement_variants"] = {
        variant: VARIANT_NOTES[variant] for variant, found in by_variant.items() if found
    }
    results["statement_bootstrap"] = {
        "method": bootstrap.method,
        "resamples": bootstrap.resamples,
        "seed": bootstrap.seed,
        "level": bootstrap.level,
        "paired_differences": paired,
        "baseline_differences": against_baselines,
        "year_blocks": {
            "method": bootstrap.year_blocks_method,
            "years": len({p.meeting_date.year for p in predictions if p.doc_id in refs}),
            "paired_differences": paired_years,
        },
    }
    results["references"] = {
        "file": _shown(gold_path, settings),
        "sha256": hashlib.sha256(gold_path.read_bytes()).hexdigest(),
        "n_labels": len(refs),
        "n_human": n_human,
        "effective_counts": _counts(ref.label for ref in refs.values()),
        "llm_vs_human": agreement
        and {key: agreement[key] for key in ("n", "accuracy", "kappa", "confusion_matrix")},
    }
    report.write_results(results_path, results)

    print(
        f"Statement level: {len(refs)} reference labels ({n_human} human, "
        f"{len(refs) - n_human} LLM-adjudicated); thresholds fixed at +-{st.THRESHOLD}, never "
        "tuned on them."
    )
    print(report.statement_table(entries, baselines))
    if paired or against_baselines:
        print(f"\nPaired bootstrap differences ({bootstrap.level:.0%} intervals):")
        print(report.paired_table([*paired, *against_baselines]))
    print(
        "\nSensitivity: whole calendar years resampled (consecutive statements are "
        "near-duplicates):"
    )
    print(report.year_block_table([*report.statement_rows(entries), *baselines]))
    if paired_years:
        print(report.paired_table(paired_years))
    for entry in entries:
        print(f"\n{entry['classifier']}, {entry['variant']} (all {entry['all']['n']} statements):")
        print(report.confusion_table(entry["all"]["confusion_matrix"]))
    if agreement:
        print(
            f"\nLLM-adjudicated label vs blind human label on {agreement['n']} statements: "
            f"accuracy {agreement['accuracy']:.3f}, kappa {agreement['kappa']:.3f}"
        )
        print(report.confusion_table(agreement["confusion_matrix"], "human / LLM-adjudicated"))
    print(f"\nWrote {results_path}")
    return 0


def run_load(settings: Settings, args: argparse.Namespace) -> int:
    from app.db.connection import connect
    from app.db.stance import load_stance
    from app.schemas.stance import FedFundsTarget
    from app.stance import fred

    predictions = _read_predictions(settings)
    path = settings.data_dir / FED_FUNDS_FILE
    if path.is_file():
        points = fred.read_change_points(path)
    else:
        points = fred.fetch_target_range()
        fred.write_change_points(path, points)
        print(f"Downloaded {len(points)} fed funds change points from FRED -> {path}")
    series = [
        FedFundsTarget(date=p.date, lower=float(p.lower), upper=float(p.upper)) for p in points
    ]
    with connect(settings) as conn:
        counts = load_stance(conn, predictions, series)
    loaded = Counter(p.classifier for p in predictions)
    print(
        f"Loaded {counts['stance_results']} stance results ("
        + ", ".join(f"{name} {loaded[name]}" for name in CLASSIFIERS if loaded[name])
        + f") and {counts['fed_funds_target']} fed funds change points into schema fedrag."
    )
    return 0


def baseline_calls(
    settings: Settings,
    predictions: Sequence["StatementPrediction"],
    refs: dict[str, "Reference"],
) -> dict[StatementBaselineName, list["BaselineCall"]]:
    """The statement baselines' calls: majority_class (the most frequent reference label) on
    every predicted statement, and rate_move_rule on every statement in data/processed (left out
    when that file is missing)."""
    from app.db.load import DOCUMENTS_FILE
    from app.stance import metrics
    from app.stance import statements as st

    def call(doc_id: str, day: date, stance: Stance) -> "BaselineCall":
        return metrics.BaselineCall(doc_id, day, stance, float(metrics.ORDINAL[stance]))

    majority = metrics.majority_label([ref.label for ref in refs.values()])
    dates = {p.doc_id: p.meeting_date for p in predictions}
    calls: dict[StatementBaselineName, list[BaselineCall]] = {
        "majority_class": [call(doc_id, day, majority) for doc_id, day in sorted(dates.items())]
    }
    if (settings.data_dir / "processed" / DOCUMENTS_FILE).is_file():
        calls["rate_move_rule"] = [
            call(s.doc_id, s.meeting_date, st.rate_move_rule(s.sentences()))
            for s in st.load_statements(settings)
        ]
    return calls


def _row_key(key: tuple[StanceClassifier, StatementVariant]) -> dict[str, str]:
    return {"classifier": key[0], "variant": key[1]}


def _paired(
    minuend: Sequence["StatementCall"],
    subtrahend: Sequence["StatementCall"],
    refs: dict[str, "Reference"],
    bootstrap: "Bootstrap",
    keys: dict[str, Any],
    *,
    year_blocks: bool = False,
) -> list[dict[str, Any]]:
    """The paired bootstrap of minuend minus subtrahend as results-file rows (all statements,
    then the human subset when there is one), each starting with `keys`."""
    from app.stance import metrics

    compared = metrics.compare_statements(
        minuend, subtrahend, refs, bootstrap, year_blocks=year_blocks
    )
    return [
        {**keys, "subset": subset, **compared[subset]}
        for subset in ("all", "human_subset")
        if compared[subset]
    ]


def sentence_factory(
    name: StanceClassifier, settings: Settings, usage: "Usage", concurrency: int
) -> Callable[[int], "SentenceClassifier"]:
    """seed -> the classifier for that seed's TDW test split, built from its train split only."""
    if name == "keyword":
        from app.stance.keyword import keyword_factory

        return keyword_factory(settings)
    if name == "finbert":
        from app.stance.finbert import finbert_factory

        _quiet_transformers()
        return finbert_factory(settings)
    from app.stance.llm import llm_factory

    return llm_factory(LLM_MODES[name], settings, usage=usage, max_concurrency=concurrency)


def statement_classifier(name: StanceClassifier, settings: Settings) -> "SentenceClassifier":
    """The sentence classifier used on statements: the three seeds together."""
    if name == "keyword":
        from app.stance.keyword import ensemble as keyword_ensemble

        return keyword_ensemble(settings=settings)
    from app.stance.finbert import ensemble as finbert_ensemble

    _quiet_transformers()
    return finbert_ensemble(settings=settings)


def _quiet_transformers() -> None:
    """No "Loading weights" progress bar for each of the saved models."""
    from transformers.utils import logging as transformers_logging

    transformers_logging.disable_progress_bar()


def sentence_estimates(
    settings: Settings, names: Sequence[StanceClassifier]
) -> list[tuple[str, "CostEstimate"]]:
    """What labelling the test splits would cost, per LLM classifier (and seed for few-shot:
    each seed has its own examples, so its own cache entries)."""
    from app.stance import tdw
    from app.stance.llm import LLMSentenceClassifier

    tests = {seed: [s.sentence for s in tdw.load("test", seed, settings)] for seed in tdw.SEEDS}
    estimates = []
    for name in names:
        mode = LLM_MODES[name]
        if mode == "zero_shot":  # no examples: one cache entry per text, shared by the seeds
            union = list(dict.fromkeys(text for seed in tdw.SEEDS for text in tests[seed]))
            classifier = LLMSentenceClassifier(mode, settings=settings)
            estimates.append((f"{name}, test splits of all seeds", classifier.estimate(union)))
            continue
        for seed in tdw.SEEDS:
            classifier = LLMSentenceClassifier(mode, seed, settings=settings)
            estimates.append((f"{name}, seed {seed}", classifier.estimate(tests[seed])))
    return estimates


def check_budget(estimates: Sequence[tuple[str, "CostEstimate"]], max_usd: float) -> int:
    """Print the estimates; raise BudgetError when their total is over max_usd (or unknown while
    calls are needed). Returns the estimated number of calls."""
    for label, estimate in estimates:
        print(f"  {label}: {estimate}")
    calls = sum(estimate.calls for _, estimate in estimates)
    usd = sum(estimate.usd for _, estimate in estimates if estimate.calls)
    print(f"  total: {calls} calls, about ${usd:.3f} (limit --max-usd {max_usd:.2f})")
    if calls and (math.isnan(usd) or usd > max_usd):
        raise BudgetError(
            f"refused: the estimate (${usd:.3f}) is over --max-usd {max_usd:.2f}; nothing was "
            "called"
        )
    return calls


def _require_llm(settings: Settings, calls: int) -> None:
    """Fail before any work when uncached LLM predictions are needed but there is no key."""
    from app.rag.answer import LLMNotConfiguredError

    if calls and not settings.llm_configured:
        raise LLMNotConfiguredError


def call_ceiling(calls: int) -> int:
    return calls + max(3, math.ceil(calls * RETRY_MARGIN))


def _ordered(names: Sequence[str]) -> list[StanceClassifier]:
    return [name for name in CLASSIFIERS if name in names]


def _described(
    factory: Callable[[int], "SentenceClassifier"], into: dict[str, Any]
) -> Callable[[int], "SentenceClassifier"]:
    """The factory, recording what each seed's classifier is (no TDW text) into `into`."""

    def build(seed: int) -> "SentenceClassifier":
        classifier = factory(seed)
        into[str(seed)] = describe(classifier)
        return classifier

    return build


def describe(classifier: object) -> dict[str, Any]:
    """Configuration of one seed's classifier for the results file: lexicon terms, FinBERT
    training report, or the LLM prompt identity (examples only as a digest)."""
    lexicons = getattr(classifier, "lexicons", None)
    if lexicons:
        return lexicons[0].to_json()
    reports = getattr(classifier, "reports", None)
    if reports:
        return asdict(reports[0])
    info = {"model_version": getattr(classifier, "model_version", "")}
    if getattr(classifier, "examples", ()):
        info["examples_sha256"] = getattr(classifier, "examples_digest", "")
    return info


def _write_merged(
    path: Path, predictions: Sequence["StatementPrediction"], replaced: Sequence[str]
) -> int:
    """Write `predictions` to the file, keeping its rows of the classifiers not in `replaced`."""
    from app.stance import statements as st

    kept = (
        [p for p in st.read_predictions(path) if p.classifier not in replaced]
        if path.is_file()
        else []
    )
    merged = sorted(
        [*kept, *predictions], key=lambda p: (p.meeting_date, CLASSIFIERS.index(p.classifier))
    )
    return st.write_predictions(path, merged)


def _stance_counts(predictions: Sequence["StatementPrediction"]) -> str:
    counts = Counter(p.stance for p in predictions)
    return ", ".join(f"{stance} {counts[stance]}" for stance in ("hawkish", "neutral", "dovish"))


def _read_predictions(settings: Settings) -> list["StatementPrediction"]:
    from app.stance import statements as st

    path = st.predictions_path(settings)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: run `python -m app.stance predict` first.")
    return st.read_predictions(path)


def _shown(path: Path, settings: Settings) -> str:
    """`path` relative to the repository (data/...) when it is inside it, else as given."""
    root = settings.data_dir.parent
    return path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)


def _counts(stances: Any) -> dict[str, int]:
    counts = Counter(stances)
    return {stance: counts[stance] for stance in ("dovish", "neutral", "hawkish")}


def _print_usage(usage: "Usage", settings: Settings) -> None:
    summary = usage.summary(settings.llm_model)
    if summary["calls"]:
        print(
            f"\nOpenAI: {summary['calls']} calls, {summary['input_tokens']:,} input "
            f"({summary['cached_input_tokens']:,} cached) + {summary['output_tokens']:,} output "
            f"tokens ({summary['reasoning_tokens']:,} reasoning), about ${summary['usd']}"
        )
    else:
        print("\nOpenAI: no calls (everything was cached)")


if __name__ == "__main__":
    sys.exit(main())
