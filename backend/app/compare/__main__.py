"""Command line for Statement Comparison: python -m app.compare {precompute,evaluate,show}."""

import argparse
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.core.config import Settings, get_settings

if TYPE_CHECKING:
    from app.schemas.compare import CompareResponse, ComparisonAnalysis, SentenceChange

DEFAULT_MAX_USD = 1.0
DEFAULT_CONCURRENCY = 4
COMMANDS = {
    "precompute": "write and store the analysis of every consecutive pair of statements (calls "
    "OpenAI once per pair not stored yet)",
    "evaluate": "score the stored analyses of consecutive pairs against the gold-implied shifts "
    "(no LLM calls) and write data/eval/compare_benchmark.json",
    "show": "print the sentence diff of two statements and their stored analysis",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.compare",
        description="Compare FOMC statements: sentence diff, LLM analysis and its evaluation.",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")
    subs = {
        name: commands.add_parser(name, help=text, description=text)
        for name, text in COMMANDS.items()
    }
    subs["precompute"].add_argument(
        "--max-usd",
        type=float,
        default=DEFAULT_MAX_USD,
        help="refuse to start when the estimated OpenAI cost of the pairs not stored yet is above "
        f"this (default: {DEFAULT_MAX_USD})",
    )
    subs["precompute"].add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"OpenAI calls in flight at once (default: {DEFAULT_CONCURRENCY})",
    )
    subs["precompute"].add_argument(
        "--dry-run", action="store_true", help="print the estimate only: no calls, nothing written"
    )
    subs["evaluate"].add_argument(
        "--out", type=Path, help="results file (default: data/eval/compare_benchmark.json)"
    )
    show = subs["show"]
    show.add_argument("base", type=date.fromisoformat, help="base meeting date, YYYY-MM-DD")
    show.add_argument("target", type=date.fromisoformat, help="target meeting date, YYYY-MM-DD")
    show.add_argument(
        "--generate",
        action="store_true",
        help="write and store the analysis when none is stored (one OpenAI call)",
    )
    show.add_argument("--all", action="store_true", help="also print the unchanged sentences")
    return parser


def main(argv: Sequence[str] | None = None, *, settings: Settings | None = None) -> int:
    """Run one command; returns the process exit code."""
    args = build_parser().parse_args(argv)
    settings = settings or get_settings()
    # Imported here so --help stays fast; these pull in LangChain and the DB driver.
    import openai
    import psycopg

    from app.compare.service import StatementNotFoundError
    from app.db.connection import DatabaseNotConfiguredError, DatabaseUrlError
    from app.rag.answer import LLMNotConfiguredError
    from app.stance.__main__ import BudgetError
    from app.stance.llm import CallLimitError

    commands: dict[str, Callable[[Settings, argparse.Namespace], int]] = {
        "precompute": run_precompute,
        "evaluate": run_evaluate,
        "show": run_show,
    }
    try:
        return commands[args.command](settings, args)
    except (
        BudgetError,
        CallLimitError,
        LLMNotConfiguredError,
        StatementNotFoundError,
        DatabaseNotConfiguredError,
        DatabaseUrlError,
        psycopg.Error,
        openai.OpenAIError,
        OSError,
        ValueError,  # nothing stored to evaluate, a malformed gold file, base not before target
    ) as exc:
        message = str(exc)
        if isinstance(exc, openai.OpenAIError):  # its message can quote part of the API key
            message = f"OpenAI request failed ({type(exc).__name__})"
        print(f"{args.command}: {message}", file=sys.stderr)
        return 1


def run_precompute(settings: Settings, args: argparse.Namespace) -> int:
    """Store the analysis of every consecutive pair of statements, estimate and budget first.

    Each pair goes through the API's own path (service.analysis_for with mode "generate"), so a
    stored pair is skipped, and what is stored is exactly what POST /api/compare serves.
    """
    from concurrent.futures import ThreadPoolExecutor

    from app.compare.analysis import get_analyzer
    from app.compare.service import AnalysisOutcome, analysis_for, comparison_input
    from app.compare.types import ComparisonInput, cache_key
    from app.db.compare import all_statements, read_comparison
    from app.db.connection import Database, apply_schema, connect
    from app.rag.answer import LLMNotConfiguredError
    from app.stance.__main__ import call_ceiling, check_budget
    from app.stance.llm import Usage

    usage = Usage()
    analyzer = get_analyzer(settings, usage=usage)
    version = f"{analyzer.model}/{analyzer.prompt_version}"
    with connect(settings) as conn:
        if not args.dry_run:
            with conn.transaction():
                apply_schema(conn)  # creates fedrag.statement_comparisons if it is missing
        statements = all_statements(conn)
        pairs = [
            comparison_input(base.statement, target.statement)
            for base, target in zip(statements, statements[1:], strict=False)
        ]
        keys = [
            cache_key(pair, model=analyzer.model, prompt_version=analyzer.prompt_version)
            for pair in pairs
        ]
        pending = [
            (pair, key)
            for pair, key in zip(pairs, keys, strict=True)
            if read_comparison(conn, key) is None
        ]
    print(
        f"{len(statements)} statements, {len(pairs)} consecutive pairs; "
        f"{len(pairs) - len(pending)} already stored under {version}"
    )
    print("OpenAI estimate (pairs not stored yet):")
    estimate = analyzer.estimate([pair for pair, _ in pending])
    calls = check_budget([("consecutive pairs", estimate)], args.max_usd)
    if args.dry_run or not pending:
        return 0
    if calls and not settings.llm_configured:
        raise LLMNotConfiguredError
    usage.max_calls = call_ceiling(calls)

    def analyse(item: tuple[ComparisonInput, str]) -> AnalysisOutcome:
        return analysis_for(db, item[0], analyzer, mode="generate")

    db = Database(settings.database_url)
    try:
        with ThreadPoolExecutor(max_workers=max(args.concurrency, 1)) as pool:
            outcomes = list(pool.map(analyse, pending))
    finally:
        db.close()
    # analysis_for returns a written analysis even when storing it failed (the API still shows
    # it), so what was stored is read back.
    with connect(settings) as conn:
        stored = sum(read_comparison(conn, key) is not None for _, key in pending)
    shifts = Counter(o.analysis.stance_shift for o in outcomes if o.analysis is not None)
    print(
        f"Wrote {sum(o.status == 'ready' for o in outcomes)} analyses ("
        + ", ".join(f"{shift} {shifts[shift]}" for shift in sorted(shifts))
        + f"); {stored} of {len(pending)} pairs now stored"
    )
    if failed := len(pending) - stored:
        statuses = dict(Counter(o.status for o in outcomes))
        print(f"{failed} not stored ({statuses}); run again to retry them", file=sys.stderr)
    _print_usage(usage.summary(analyzer.model))
    return 1 if failed else 0


def run_evaluate(settings: Settings, args: argparse.Namespace) -> int:
    from app.compare import evaluate

    path = args.out or evaluate.benchmark_path(settings)
    results = evaluate.run(settings, path)
    coverage = results["coverage"]
    print(
        f"{coverage['analysed']} of {coverage['pairs']} consecutive pairs analysed "
        f"({coverage['with_gold']} with gold labels on both statements, "
        f"{coverage['with_gold_both_human']} with both human)"
    )
    agreement, baselines = results["agreement"], results["baselines"]
    current = _prompt_name(results["analyses"])
    rows = [
        (f"LLM ({current}) vs gold-implied shift", agreement["llm_vs_gold"]),
        *(
            (
                f"LLM ({_prompt_name(run['analyses'])}) vs gold-implied shift",
                run["agreement"]["llm_vs_gold"],
            )
            for run in results["earlier_prompts"]
        ),
        ("always unchanged vs gold", baselines["always_unchanged_vs_gold"]),
        ("zero-shot delta vs gold", baselines["zero_shot_delta_vs_gold"]),
        ("rate-decision rule vs gold (post hoc)", baselines["rate_decision_vs_gold"]),
        ("LLM vs zero-shot delta (same family)", agreement["llm_vs_zero_shot_delta"]),
    ]
    print("\nAccuracy [95% bootstrap CI] (n): all pairs | reference changed | reference same")
    for label, subsets in rows:
        cells = [
            _accuracy(subsets[name]) for name in ("all", "reference_changed", "reference_unchanged")
        ]
        print(f"  {label}: " + " | ".join(cells))
    human = agreement["llm_vs_gold_both_human"]
    print(f"  LLM ({current}) vs gold-implied shift, both labels human: {_accuracy(human)}")
    for run in results["earlier_prompts"]:
        print(f"  {run['analyses']}: {run['prompt']}")
    counts = results["shift_counts"]
    print("\nShifts (more_dovish / unchanged / more_hawkish):")
    for name, tally in counts.items():
        print(f"  {name}: " + " / ".join(str(tally[s]) for s in evaluate.SHIFTS))
    structural = results["structural"]
    before = structural["before_cleaning"]
    print(
        "\nStructural: key changes present on "
        f"{_rate(structural['key_changes_present_when_changed'])} changed pairs; cited ids not "
        f"changed items {structural['cited_ids_not_changed_items']['count']}; shift backed by a "
        f"key change of its direction {_rate(structural['shift_backed_by_key_change'])}; key "
        f"changes without a quote {_rate(structural['key_changes_without_quote'])}; before "
        f"cleaning: {before['invalid_ids']} invalid ids, {before['key_changes_dropped']} key "
        f"changes dropped, quotes verbatim {_rate(before['quotes_verbatim'])}"
    )
    if cost := results["cost"]:
        _print_usage(cost)
    print(f"\nWrote {path}")
    return 0


def run_show(settings: Settings, args: argparse.Namespace) -> int:
    from app.compare import service
    from app.compare.analysis import get_analyzer
    from app.db.connection import Database
    from app.schemas.compare import CompareRequest

    if args.base >= args.target:
        raise ValueError("the base meeting date must be earlier than the target")
    request = CompareRequest(
        base_date=args.base,
        target_date=args.target,
        analysis="generate" if args.generate else "cached_only",
    )
    db = Database(settings.database_url)
    try:
        response = service.compare_statements(
            db, request, get_analyzer(settings), llm_configured=settings.llm_configured
        )
    finally:
        db.close()
    print(format_response(response, show_unchanged=args.all))
    return 0


def format_response(response: "CompareResponse", *, show_unchanged: bool = False) -> str:
    """The comparison as plain text: both statements, the stats, each change (deleted words in
    [-...-], inserted in {+...+}), the analysis or its status, and the stored stances."""
    stats = response.stats
    lines = [
        f"Base:   {response.base.title} ({response.base.doc_id}, released "
        f"{response.base.release_date})",
        f"Target: {response.target.title} ({response.target.doc_id}, released "
        f"{response.target.release_date})",
        f"{len(response.changes)} aligned sentences: {stats.modified} modified, {stats.added} "
        f"added, {stats.removed} removed, {stats.unchanged} unchanged; overall similarity "
        f"{stats.similarity:.3f}",
        "",
    ]
    for change in response.changes:
        if change.kind == "unchanged" and not show_unchanged:
            continue
        lines += [*_change_lines(change), ""]
    analysis = response.analysis
    if analysis is None:
        lines.append(f"Analysis: {response.analysis_status}")
    else:
        lines += _analysis_lines(analysis, response.analysis_model_version or "")
    stance = response.stance
    if stance.base and stance.target:
        lines += [
            "",
            f"Stance ({stance.base.classifier}, each statement scored alone): "
            f"{stance.base.stance} {stance.base.score:+.2f} -> {stance.target.stance} "
            f"{stance.target.score:+.2f}",
        ]
    return "\n".join(lines)


def _change_lines(change: "SentenceChange") -> list[str]:
    head = f"[{change.id}] {change.kind}"
    if change.similarity is not None and change.kind == "modified":
        head += f" ({change.similarity:.2f})"
    if change.procedural:
        head += ", procedural"
    if change.kind == "modified" and change.segments:
        marks = {"equal": "{}", "delete": "[-{}-]", "insert": "{{+{}+}}"}
        return [head, "  " + "".join(marks[s.op].format(s.text) for s in change.segments)]
    if change.kind == "added":
        return [head, f"  + {change.target}"]
    if change.kind == "removed":
        return [head, f"  - {change.base}"]
    return [head, f"  {change.target}"]


def _analysis_lines(analysis: "ComparisonAnalysis", version: str) -> list[str]:
    lines = [
        f"Analysis ({version}): {analysis.stance_shift}, confidence {analysis.confidence:.2f} "
        "(model-reported, not calibrated)",
        f"  {analysis.summary}",
        "  Key changes:",
    ]
    for n, item in enumerate(analysis.key_changes, 1):
        ids = ", ".join(f"#{i}" for i in item.change_ids)
        lines += [
            f"  {n}. [{item.direction}] {ids}: {item.what_changed}",
            f"     {item.interpretation}",
        ]
    return lines


def _prompt_name(analyses: str) -> str:
    """ "model/compare-v3.1a2b3c4d" -> "compare-v3"."""
    return analyses.rsplit("/", 1)[-1].split(".")[0]


def _accuracy(report: dict[str, Any] | None) -> str:
    if not report:
        return "n/a"
    ci = report.get("accuracy_ci") or {}
    interval = f" [{ci['low']:.3f}, {ci['high']:.3f}]" if ci else ""
    return f"{report['accuracy']:.3f}{interval} (n={report['n']})"


def _rate(rate: dict[str, Any]) -> str:
    return f"{rate['count']}/{rate['of']}"


def _print_usage(summary: dict[str, Any]) -> None:
    if summary.get("calls"):
        print(
            f"\nOpenAI: {summary['calls']} calls, {summary['input_tokens']:,} input "
            f"({summary['cached_input_tokens']:,} cached) + {summary['output_tokens']:,} output "
            f"tokens ({summary['reasoning_tokens']:,} reasoning), about ${summary['usd']}"
        )
    else:
        print("\nOpenAI: no calls (every pair was stored)")


if __name__ == "__main__":
    sys.exit(main())
