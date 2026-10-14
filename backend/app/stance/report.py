"""The stance results file (data/eval/stance_benchmark.json) and the stance CLI's console tables.

The file is tracked in git, so it holds metrics, configuration and costs only: never a Trillion
Dollar Words sentence (CC BY-NC 4.0) or a few-shot example, which appear only as digests. Each CLI
step rewrites its own sections and keeps the others: `benchmark` writes sentence_level and
sentence_baselines, `evaluate` writes statement_level (pre-registered and post-hoc rows, with
bootstrap intervals), statement_baselines, statement_variants, statement_bootstrap and
references, `benchmark` and `predict` add the OpenAI usage of their uncached calls to cost, and
both describe the classifiers they ran under models.
"""

import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, get_args

from app.core.config import Settings, get_settings
from app.ingest.records import write_text_atomic
from app.schemas.stance import StatementVariant
from app.stance import tdw
from app.stance.llm import cost_usd
from app.stance.metrics import Reference
from app.stance.types import BENCHMARK_FILE, CLASSIFIERS, STANCES

VARIANTS: tuple[StatementVariant, ...] = get_args(StatementVariant)
DIGITS = 5  # floats keep this many significant digits in the file
USAGE_FIELDS = (
    "calls",
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_tokens",
)
DATASET: dict[str, str] = {
    "name": "Trillion Dollar Words",
    "citation": tdw.CITATION,
    "license": "CC BY-NC 4.0",
    "url": f"https://github.com/{tdw.REPO}",
    "commit": tdw.COMMIT,
}


def benchmark_path(settings: Settings | None = None) -> Path:
    return (settings or get_settings()).data_dir / BENCHMARK_FILE


def read_results(path: Path) -> dict[str, Any]:
    """The results file's contents; {} when it does not exist yet."""
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def rounded(value: Any, digits: int = DIGITS) -> Any:
    """`value` with every float cut to `digits` significant digits (NaN and infinity become
    null), recursively."""
    if isinstance(value, float):
        return float(f"{value:.{digits}g}") if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {key: rounded(item, digits) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [rounded(item, digits) for item in value]
    return value


def write_results(path: Path, results: Mapping[str, Any], *, now: datetime | None = None) -> None:
    """Write the file atomically, stamped with generated_at and the dataset, after checking that
    the API can serve it (app.api.stance.benchmark_from_json)."""
    from app.api.stance import benchmark_from_json  # the API's own contract check

    stamp = (now or datetime.now(UTC)).replace(microsecond=0).isoformat()
    ordered = {"generated_at": stamp, "dataset": DATASET}
    ordered |= {key: value for key, value in results.items() if key not in ordered}
    ordered.setdefault("sentence_level", [])
    ordered.setdefault("statement_level", [])
    content = rounded(ordered)
    benchmark_from_json(content)
    write_text_atomic(path, json.dumps(content, indent=2, ensure_ascii=False) + "\n")


def by_classifier(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Entries in CLASSIFIERS order (one per classifier; the last one wins)."""
    latest = {entry["classifier"]: dict(entry) for entry in entries}
    return [latest[name] for name in CLASSIFIERS if name in latest]


def merge_section(
    old: Sequence[Mapping[str, Any]], new: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """The old entries with those of the classifiers in `new` replaced."""
    return by_classifier([*old, *new])


def add_usage(old: Mapping[str, Any] | None, new: Mapping[str, Any], model: str) -> dict[str, Any]:
    """Two usage summaries (llm.Usage.summary) added together, the dollar cost recomputed."""
    total = {name: int((old or {}).get(name) or 0) + int(new[name]) for name in USAGE_FIELDS}
    usd = cost_usd(
        model, total["input_tokens"], total["cached_input_tokens"], total["output_tokens"]
    )
    return {"model": model, **total, "usd": None if math.isnan(usd) else round(usd, 4)}


def record_cost(
    results: dict[str, Any], step: str, usage: Mapping[str, Any], *, note: str = ""
) -> None:
    """Add one step's OpenAI usage to results["cost"] (cumulative, so a rerun served from the
    cache adds nothing) and update the total."""
    if not usage["calls"]:
        return
    model = usage["model"]
    cost = results.setdefault("cost", {})
    steps = cost.setdefault("steps", {})
    steps[step] = add_usage(steps.get(step), usage, model)
    total: dict[str, Any] | None = None
    for summary in steps.values():
        total = add_usage(total, summary, model)
    cost["total"] = total
    if note:
        cost["note"] = note


def reference_status(refs: Mapping[str, Reference]) -> str:
    """ "human" when every reference label is human, "mixed" when some are, else
    "llm_adjudicated" (the StatementBenchmark.reference values)."""
    human = sum(ref.source == "human" for ref in refs.values())
    return "human" if refs and human == len(refs) else "mixed" if human else "llm_adjudicated"


def _num(value: float | None, width: int = 6) -> str:
    return f"{value:{width}.3f}" if value is not None else f"{'n/a':>{width}}"


def _mean_sd(stat: Mapping[str, float | None]) -> str:
    sd = stat.get("sd")
    return f"{_num(stat['mean'], 5)} +- {_num(sd, 5) if sd is not None else '  n/a'}"


def sentence_table(entries: Sequence[Mapping[str, Any]]) -> str:
    """Macro-F1 and accuracy (mean +- sd over seeds) per classifier and test split."""
    lines = [
        f"{'classifier':<14} {'split':<13} {'macro-F1':>15} {'accuracy':>15} {'kappa':>15}  n",
    ]
    for entry in by_classifier(entries):
        for split in ("official", "deduplicated"):
            report = entry[split]
            n = "/".join(str(v) for v in report["n"])
            lines.append(
                f"{entry['classifier']:<14} {split:<13} {_mean_sd(report['macro_f1'])} "
                f"{_mean_sd(report['accuracy'])} {_mean_sd(report['kappa'])}  {n}"
            )
    return "\n".join(lines)


def sentence_baseline_table(entries: Sequence[Mapping[str, Any]]) -> str:
    """The sentence baselines (metrics.majority_class_sentences) in sentence_table's columns."""
    lines = []
    for entry in entries:
        name = f"{entry['name']} ({entry.get('label') or 'per seed'})"
        for split in ("official", "deduplicated"):
            report = entry[split]
            lines.append(
                f"{name:<14} {split:<13} {_mean_sd(report['macro_f1'])} "
                f"{_mean_sd(report['accuracy'])} {_mean_sd(report['kappa'])}"
            )
    return "\n".join(lines)


def statement_rows(entries: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Statement-level entries with pre-registered ones first, each variant in CLASSIFIERS order
    (an entry without a variant is pre-registered)."""

    def key(entry: Mapping[str, Any]) -> tuple[int, int]:
        variant = entry.get("variant", "pre_registered")
        return VARIANTS.index(variant), CLASSIFIERS.index(entry["classifier"])

    return sorted(entries, key=key)


def _ci(report: Mapping[str, Any], name: str, suffix: str = "") -> str:
    interval = report.get(f"{name}_ci{suffix}")
    if not interval:
        return f"{'':>15}"
    return f"[{interval['low']:.3f}, {interval['high']:.3f}]"


def _row_label(entry: Mapping[str, Any]) -> tuple[str, str]:
    """(name, variant column) of a statement row or a baseline (named, no classifier)."""
    if "classifier" not in entry:
        return entry["name"], "post-hoc base" if entry.get("post_hoc") else "baseline"
    post_hoc = entry.get("variant") == "post_hoc_rubric_ordered"
    return entry["classifier"], "post hoc" if post_hoc else "pre-reg."


def statement_table(
    entries: Sequence[Mapping[str, Any]], baselines: Sequence[Mapping[str, Any]] = ()
) -> str:
    """Accuracy and macro-F1 (with bootstrap intervals when present), kappa and Spearman per
    classifier and variant, then per baseline: all statements, then the human subset."""
    lines = [
        f"{'classifier':<14} {'variant':<14} {'subset':<13} {'n':>3} {'accuracy':>8} "
        f"{'95% CI':<15} {'macro-F1':>8} {'95% CI':<15} {'kappa':>6} {'spearman':>8}"
    ]
    for entry in [*statement_rows(entries), *baselines]:
        name, variant = _row_label(entry)
        for subset in ("all", "human_subset"):
            report = entry.get(subset)
            if not report:
                continue
            lines.append(
                f"{name:<14} {variant:<14} {subset:<13} {report['n']:>3} "
                f"{_num(report['accuracy'], 8)} {_ci(report, 'accuracy')} "
                f"{_num(report['macro_f1'], 8)} {_ci(report, 'macro_f1')} "
                f"{_num(report['kappa'])} {_num(report['spearman'], 8)}"
            )
    return "\n".join(lines)


def year_block_table(entries: Sequence[Mapping[str, Any]]) -> str:
    """Accuracy intervals of each row on all statements: single statements resampled, then
    whole calendar years (the sensitivity check)."""
    lines = [f"{'classifier':<14} {'variant':<14} {'accuracy':>8} {'95% CI':<15} {'years':<15}"]
    for entry in entries:
        name, variant = _row_label(entry)
        report = entry["all"]
        lines.append(
            f"{name:<14} {variant:<14} {_num(report['accuracy'], 8)} "
            f"{_ci(report, 'accuracy')} {_ci(report, 'accuracy', '_year_blocks')}"
        )
    return "\n".join(lines)


def paired_table(differences: Sequence[Mapping[str, Any]]) -> str:
    """Paired bootstrap differences of accuracy and macro-F1: minuend minus subtrahend, or a
    row minus a baseline."""

    def row_name(key: Mapping[str, str] | str) -> str:
        if isinstance(key, str):  # a baseline
            return key
        suffix = " (post hoc)" if key["variant"] == "post_hoc_rubric_ordered" else ""
        return key["classifier"] + suffix

    lines = []
    for diff in differences:
        parts = [
            f"{name} {diff[name]['difference']:+.3f} "
            f"[{diff[name]['ci']['low']:+.3f}, {diff[name]['ci']['high']:+.3f}]"
            for name in ("accuracy", "macro_f1")
        ]
        left, right = (
            (diff["minuend"], diff["subtrahend"])
            if "minuend" in diff
            else (diff["row"], diff["baseline"])
        )
        lines.append(
            f"  {row_name(left)} - {row_name(right)}, "
            f"{diff['subset']} (n={diff['n']}): " + "; ".join(parts)
        )
    return "\n".join(lines)


def confusion_table(
    matrix: Sequence[Sequence[int]], corner: str = "reference / predicted", indent: str = "  "
) -> str:
    """A 3x3 confusion matrix: reference labels as rows, predictions as columns."""
    width = max(22, len(corner) + 2)
    header = f"{indent}{corner:<{width}}" + "".join(f"{s:>9}" for s in STANCES)
    rows = [
        f"{indent}{stance:<{width}}" + "".join(f"{count:>9}" for count in row)
        for stance, row in zip(STANCES, matrix, strict=True)
    ]
    return "\n".join([header, *rows])
