"""Benchmark of the stored comparison analyses on consecutive FOMC statements (metrics only).

For every pair of consecutive statements (meeting-date order: 96 pairs for 97 statements) the
stored analysis's stance_shift is compared with two references:

(a) The shift implied by the effective gold labels (data/eval/stance_gold.jsonl: human_label,
    else label), on the ordinal scale dovish < neutral < hawkish: a higher target label is
    more_hawkish, a lower one more_dovish, the same label unchanged. The same label does not
    mean no shift: each statement was labelled on its own text, without the other, and two
    statements with one label can still differ in how far they lean (a hike then a hold with
    tightening guidance are both hawkish, and the prompt counts a hold after a hike as a shift).
    Agreement is therefore reported on all pairs, on the pairs whose gold label changed (where
    the reference shift is least ambiguous) and on those whose label did not.
(b) The sign of the llm_zero_shot statement-score delta (target minus base), unchanged when
    |delta| <= DELTA_THRESHOLD, fixed at 0.2 before any analysis was scored. Those scores come
    from the same model family judging each statement alone, prompted with the rubric the
    reference annotators followed: a consistency cross-check, not an independent reference.

Agreement is 3-class accuracy (and macro-F1 on all pairs) with percentile bootstrap intervals
from app.stance.metrics, each shift mapped to the stance in the same position (more_dovish to
dovish, unchanged to neutral, more_hawkish to hawkish) so those helpers apply unchanged, plus
intervals that resample whole calendar years (by target meeting) as a sensitivity check.
Baselines on the same pairs: always "unchanged"; the zero-shot score delta itself scored against
the gold-implied shift (an upper-bound cross-check: it shares the reference annotators' rubric);
and, post hoc (added after the analyses had been scored), the rate-decision rule: each
statement's decision on the target range from its first rate-action sentence
(app.stance.statements.rate_action), compared as decisions the way the prompt's rule 9 compares
them (a hike after a hold is more_hawkish, a hold after a hike more_dovish, a hike after a hike
unchanged; a statement that states no decision gives unchanged). The same rule splits the
same-label pairs into those whose rate decision changed, stayed the same, or is not stated, with
how often the LLM and the zero-shot delta call a shift on each. Structural checks read the counts
each analysis recorded before its reply was cleaned (invalid ids, dropped key changes, quotes
that were not verbatim) and check the stored analysis against the current diff: every cited id
is a changed item, key changes are present whenever something changed, and a non-unchanged
shift is backed by a key change with that direction (a label match, not a check of the
reasoning), and every key change quotes the wording it is about. The results file holds
metrics, labels and doc ids only, never statement text.

The file scores the analyses of the current prompt and, under earlier_prompts, those of the prompt
versions in EARLIER_PROMPTS that are still stored: compare-v2 was fixed before any analysis was
scored against a reference (pre-registered); compare-v3 was written after v2's results were seen
(post hoc), so its agreement is not an independent estimate.
"""

import json
import re
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import psycopg
from pydantic import ValidationError

from app.core.config import Settings, get_settings
from app.ingest.records import write_text_atomic
from app.schemas.common import Stance
from app.schemas.compare import ComparisonAnalysis, StanceShift
from app.stance.metrics import (
    DEFAULT_BOOTSTRAP,
    ORDINAL,
    Bootstrap,
    Reference,
    bootstrap_ci,
    classification_metrics,
)
from app.stance.report import USAGE_FIELDS, add_usage, rounded
from app.stance.statements import MOVE_DIRECTIONS, RATE_MOVE, rate_action, split_sentences
from app.stance.types import STANCES

SHIFTS: tuple[StanceShift, ...] = ("more_dovish", "unchanged", "more_hawkish")
AS_STANCE: dict[StanceShift, Stance] = dict(zip(SHIFTS, STANCES, strict=True))
DIRECTION: dict[StanceShift, Stance] = {"more_dovish": "dovish", "more_hawkish": "hawkish"}
DELTA_THRESHOLD = 0.2
DELTA_DIGITS = 9  # the delta is rounded first, so 0.5 - 0.3 counts as exactly 0.2
BENCHMARK_FILE = Path("eval") / "compare_benchmark.json"  # under Settings.data_dir
ZERO_SHOT = "llm_zero_shot"
CHECK_FIELDS = (
    "key_changes_returned",
    "key_changes_dropped",
    "key_changes_cut",
    "invalid_ids",
    "quotes",
    "quotes_unquoted",
    "quotes_outside_cited",
)
# Stored prompt versions scored next to the current one, and what each of them is.
EARLIER_PROMPTS: dict[str, str] = {
    "compare-v2.3b53da07": (
        "pre-registered: the prompt was fixed before any analysis was scored against a reference"
    ),
}
PROMPT_NOTES: dict[str, str] = {
    "compare-v3": (
        "post hoc: written after the compare-v2 run had been scored against these references, to "
        "fix quotes that broke the JSON output and rate moves compared by level; its agreement is "
        "not an independent estimate"
    ),
    **{label.split(".")[0]: note for label, note in EARLIER_PROMPTS.items()},
}
QUOTED = re.compile(r'"[^"\n]+"|“[^”\n]+”')  # as app.compare.analysis.QUOTED
NOTES = (
    "A same gold label does not mean no shift: each statement was labelled on its own text, so "
    "two hawkish statements can differ in how hawkish they are (e.g. a hike followed by a hold: "
    "both hawkish, and the prompt counts it as a shift), and an analysis that reads such a pair "
    "as a shift is not necessarily wrong.",
    "The zero-shot score delta comes from the same model family (llm_zero_shot judging each "
    "statement alone): agreement with it is a consistency cross-check, not independent "
    "validation. llm_zero_shot is also prompted with the rubric the reference annotators "
    "followed, so its delta agrees with the gold-implied shift largely by construction: its "
    "baseline rows are an upper-bound cross-check, not a baseline the comparison should beat.",
    "The rate-decision rule (rate_decision_vs_gold, same_label_by_rate_decision) is post hoc: "
    "added after the analyses had been scored. It reads each statement's decision on the target "
    "range with app.stance.statements.rate_action and compares the two decisions as the prompt's "
    "rule 9 does; a statement without a rate decision (2019-10-04, 2020-03-23) gives unchanged "
    "in the baseline.",
    "Pairs share statements (each statement is the target of one pair and the base of the "
    "next), and the bootstrap resamples pairs as if independent, so the intervals understate the "
    "uncertainty; the year-block intervals are a sensitivity check.",
    "confidence is model-reported, never a calibrated probability.",
)


@dataclass(frozen=True)
class PairRecord:
    """One consecutive pair: the ids of the current diff's changed items, the stored analysis
    and usage (None when no analysis is stored for the pair's current cache key), and the base's
    and target's rate decisions (rate_decision)."""

    base_doc_id: str
    target_doc_id: str
    target_date: date
    changed_ids: frozenset[int]
    analysis: Mapping[str, Any] | None
    usage: Mapping[str, Any] | None = None
    rate_decisions: tuple[int | None, int | None] = (None, None)


def gold_shift(base: Stance, target: Stance) -> StanceShift:
    """The shift implied by two stance labels on the ordinal dovish < neutral < hawkish."""
    step = ORDINAL[target] - ORDINAL[base]
    return "more_hawkish" if step > 0 else "more_dovish" if step < 0 else "unchanged"


def delta_shift(
    base_score: float, target_score: float, threshold: float = DELTA_THRESHOLD
) -> StanceShift:
    """The shift implied by two stance scores: unchanged when |target - base| <= threshold."""
    delta = round(target_score - base_score, DELTA_DIGITS)
    if abs(delta) <= threshold:
        return "unchanged"
    return "more_hawkish" if delta > 0 else "more_dovish"


def rate_decision(paragraphs: Sequence[str]) -> int | None:
    """The statement's decision on the target range, from its first sentence that states one
    (app.stance.statements.rate_action): 1 a raise, 0 a hold, -1 a cut; None when no sentence
    does (the 2019-10-04 reserve-management and 2020-03-23 purchases-only statements)."""
    for paragraph in paragraphs:
        for sentence in split_sentences(paragraph):
            if move := RATE_MOVE.search(sentence):  # a move wins over a hold, as in rate_action
                return 1 if MOVE_DIRECTIONS[move.group("verb").lower()] == "hawkish" else -1
            if rate_action(sentence):
                return 0
    return None


def decision_shift(base: int | None, target: int | None) -> StanceShift | None:
    """The shift implied by two rate decisions, compared as decisions (None when either statement
    states none): a hike after a hold or a hold after a cut is more_hawkish, a hold after a hike
    or a cut after a hold more_dovish, and the same decision (a hike after a hike) unchanged."""
    if base is None or target is None:
        return None
    return "more_hawkish" if target > base else "more_dovish" if target < base else "unchanged"


def counts(shifts: Sequence[StanceShift]) -> dict[str, int]:
    tally = Counter(shifts)
    return {shift: tally[shift] for shift in SHIFTS}


def agreement(
    reference: Sequence[StanceShift],
    predicted: Sequence[StanceShift],
    years: Sequence[int],
    bootstrap: Bootstrap | None = DEFAULT_BOOTSTRAP,
    *,
    full: bool = True,
    macro_f1: bool = True,
) -> dict[str, Any] | None:
    """Agreement of `predicted` with `reference` shifts (None without pairs): accuracy with its
    bootstrap intervals (pairs, then whole years), and with `full` macro-F1 with its intervals
    (unless `macro_f1` is off), Cohen's kappa, per-class scores and the confusion matrix
    (reference rows, predicted columns, both in SHIFTS order); then the predicted counts. Leave
    `full` off for a subset whose reference holds one class, where macro-F1 says nothing, and
    `macro_f1` off for one whose reference lacks a class (the pairs whose reference changed have
    no "unchanged"), where macro-F1 would average in that class's score of 0."""
    if not reference:
        return None
    gold = [AS_STANCE[shift] for shift in reference]
    pred = [AS_STANCE[shift] for shift in predicted]
    metrics = classification_metrics(gold, pred)
    intervals: dict[str, dict[str, Any]] = {"accuracy": {}, "macro_f1": {}}
    if bootstrap is not None:
        pairs = bootstrap_ci(gold, pred, bootstrap)
        blocks = bootstrap_ci(gold, pred, bootstrap, years)
        for name, values in intervals.items():
            values |= {f"{name}_ci": pairs[name], f"{name}_ci_year_blocks": blocks[name]}
    report: dict[str, Any] = {"n": len(gold), "accuracy": metrics["accuracy"]}
    report |= intervals["accuracy"]
    if full:
        if macro_f1:
            report |= {"macro_f1": metrics["macro_f1"], **intervals["macro_f1"]}
        report |= {
            "kappa": metrics["kappa"],
            "per_class": {
                shift: metrics["per_class"][stance] for shift, stance in AS_STANCE.items()
            },
            "confusion_matrix": metrics["confusion_matrix"],
        }
    return report | {"predicted": counts(predicted)}


@dataclass(frozen=True)
class _Scored:
    """A pair with a valid stored analysis and whatever references it has."""

    record: PairRecord
    analysis: ComparisonAnalysis
    gold: StanceShift | None
    delta: StanceShift | None
    score_delta: float | None
    decision: StanceShift | None = None  # decision_shift of the pair's rate decisions


def evaluate(
    pairs: Sequence[PairRecord],
    refs: Mapping[str, Reference],
    zero_shot_scores: Mapping[str, float],
    *,
    bootstrap: Bootstrap | None = DEFAULT_BOOTSTRAP,
    threshold: float = DELTA_THRESHOLD,
) -> dict[str, Any]:
    """The benchmark of the stored analyses of `pairs` (see the module docstring): coverage,
    shift counts, agreement with both references, baselines, the same-label pairs split by rate
    decision, structural checks, cost and one row per pair. ValueError when no pair has a valid
    stored analysis."""
    ordered = sorted(pairs, key=lambda p: (p.target_date, p.target_doc_id))
    scored: list[_Scored] = []
    missing: list[list[str]] = []
    invalid: list[list[str]] = []
    for record in ordered:
        ids = [record.base_doc_id, record.target_doc_id]
        if record.analysis is None:
            missing.append(ids)
            continue
        try:
            analysis = ComparisonAnalysis.model_validate(record.analysis)
        except ValidationError:
            invalid.append(ids)
            continue
        base, target = refs.get(record.base_doc_id), refs.get(record.target_doc_id)
        gold = gold_shift(base.label, target.label) if base and target else None
        scores = [zero_shot_scores.get(doc) for doc in ids]
        score_delta = delta = None
        if scores[0] is not None and scores[1] is not None:
            score_delta = scores[1] - scores[0]
            delta = delta_shift(scores[0], scores[1], threshold)
        decision = decision_shift(*record.rate_decisions)
        scored.append(_Scored(record, analysis, gold, delta, score_delta, decision))
    if not scored:
        raise ValueError(f"no stored analysis for any of the {len(pairs)} pairs")

    with_gold = [s for s in scored if s.gold is not None]
    with_delta = [s for s in scored if s.delta is not None]

    def compare(
        items: Sequence[_Scored],
        reference: str,
        predicted: str,
        *,
        full: bool = True,
        macro_f1: bool = True,
    ) -> dict[str, Any] | None:
        return agreement(
            [getattr(s, reference) for s in items],
            [_shift(s, predicted) for s in items],
            [s.record.target_date.year for s in items],
            bootstrap,
            full=full,
            macro_f1=macro_f1,
        )

    def subsets(items: Sequence[_Scored], reference: str, predicted: str) -> dict[str, Any]:
        moved = [s for s in items if getattr(s, reference) != "unchanged"]
        still = [s for s in items if getattr(s, reference) == "unchanged"]
        return {
            "all": compare(items, reference, predicted),
            "reference_changed": compare(moved, reference, predicted, macro_f1=False),
            "reference_unchanged": compare(still, reference, predicted, full=False),
        }

    human = [
        s
        for s in with_gold
        if refs[s.record.base_doc_id].source == refs[s.record.target_doc_id].source == "human"
    ]
    return {
        "threshold": threshold,
        "notes": list(NOTES),
        "coverage": {
            "pairs": len(pairs),
            "analysed": len(scored),
            "missing": missing,
            "invalid_stored": invalid,
            "with_gold": len(with_gold),
            "with_gold_both_human": len(human),
            "with_zero_shot": len(with_delta),
        },
        "shift_counts": {
            "llm": counts([s.analysis.stance_shift for s in scored]),
            "gold_implied": counts([s.gold for s in with_gold if s.gold]),
            "zero_shot_delta": counts([s.delta for s in with_delta if s.delta]),
        },
        "agreement": {
            "llm_vs_gold": subsets(with_gold, "gold", "llm"),
            "llm_vs_gold_both_human": compare(human, "gold", "llm", full=False),
            "llm_vs_zero_shot_delta": subsets(with_delta, "delta", "llm"),
        },
        "baselines": {
            "always_unchanged_vs_gold": subsets(with_gold, "gold", "unchanged"),
            "always_unchanged_vs_gold_both_human": compare(human, "gold", "unchanged", full=False),
            "zero_shot_delta_vs_gold": subsets(
                [s for s in with_gold if s.delta is not None], "gold", "delta"
            ),
            "zero_shot_delta_vs_gold_both_human": compare(
                [s for s in human if s.delta is not None], "gold", "delta", full=False
            ),
            "rate_decision_vs_gold": subsets(with_gold, "gold", "decision"),
            "rate_decision_vs_gold_both_human": compare(human, "gold", "decision", full=False),
        },
        "same_label_by_rate_decision": same_label_by_decision(with_gold),
        "structural": structural_checks(scored),
        "cost": total_usage([s.record.usage for s in scored]),
        "pairs": [_pair_row(s) for s in scored],
    }


def _shift(item: _Scored, source: str) -> StanceShift:
    """The shift `source` ("llm", "gold", "delta", "decision" or a constant shift) gives `item`;
    "decision" is unchanged when a statement of the pair states no rate decision."""
    if source == "llm":
        return item.analysis.stance_shift
    if source == "decision":
        return item.decision or "unchanged"
    if source in ("gold", "delta"):
        shift = getattr(item, source)
        if shift is None:
            raise ValueError(f"{item.record.target_doc_id}: no {source} shift")
        return shift
    if source in SHIFTS:
        return source  # type: ignore[return-value]
    raise ValueError(f"unknown shift source {source!r}")


def same_label_by_decision(items: Sequence[_Scored]) -> dict[str, Any]:
    """The pairs whose gold-implied shift is unchanged, split by their rate-decision shift
    (changed, the same, or not stated by a statement): how many of each the LLM and the zero-shot
    delta call a shift, and, where the decision changed, how many LLM calls follow its
    direction."""
    same = [s for s in items if s.gold == "unchanged"]
    groups = {
        "decision_changed": [s for s in same if s.decision in DIRECTION],
        "decision_same": [s for s in same if s.decision == "unchanged"],
        "no_decision": [s for s in same if s.decision is None],
    }
    report: dict[str, Any] = {}
    for name, group in groups.items():
        deltas = [s.delta for s in group if s.delta is not None]
        report[name] = {
            "pairs": len(group),
            "llm_shift": _rate(
                sum(1 for s in group if s.analysis.stance_shift != "unchanged"), len(group)
            ),
            "zero_shot_delta_shift": _rate(sum(1 for d in deltas if d != "unchanged"), len(deltas)),
        }
    changed = groups["decision_changed"]
    report["decision_changed"]["llm_in_decision_direction"] = _rate(
        sum(1 for s in changed if s.analysis.stance_shift == s.decision), len(changed)
    )
    return report


def structural_checks(scored: Sequence[_Scored]) -> dict[str, Any]:
    """Checks of the stored analyses against the current diff, and the counts recorded when each
    reply was cleaned (summed over the analyses that recorded them)."""
    changed = [s for s in scored if s.record.changed_ids]
    present = sum(1 for s in changed if s.analysis.key_changes)
    cited = [i for s in scored for k in s.analysis.key_changes for i in k.change_ids]
    stale = sum(
        1
        for s in scored
        for k in s.analysis.key_changes
        for i in k.change_ids
        if i not in s.record.changed_ids
    )
    shifted = [s for s in scored if s.analysis.stance_shift != "unchanged"]
    backed = sum(
        1
        for s in shifted
        if any(k.direction == DIRECTION[s.analysis.stance_shift] for k in s.analysis.key_changes)
    )
    recorded = [
        s.record.usage["checks"]
        for s in scored
        if s.record.usage and isinstance(s.record.usage.get("checks"), Mapping)
    ]
    before = {name: sum(int(c.get(name) or 0) for c in recorded) for name in CHECK_FIELDS}
    before["confidence_clipped"] = sum(1 for c in recorded if c.get("confidence_clipped"))
    quotes = before["quotes"]
    key_changes = [k for s in scored for k in s.analysis.key_changes]
    unquoted = sum(1 for k in key_changes if not QUOTED.search(k.what_changed))
    per_pair = [len(s.analysis.key_changes) for s in changed]
    confidences = [s.analysis.confidence for s in scored]
    return {
        "pairs_with_changes": len(changed),
        "pairs_without_changes": len(scored) - len(changed),
        "key_changes_present_when_changed": _rate(present, len(changed)),
        "key_changes_per_changed_pair": {
            "mean": statistics.fmean(per_pair) if per_pair else None,
            "max": max(per_pair, default=None),
        },
        "cited_ids_not_changed_items": {"count": stale, "of": len(cited)},
        "shift_backed_by_key_change": _rate(backed, len(shifted)),
        "key_changes_without_quote": _rate(unquoted, len(key_changes)),
        "before_cleaning": {
            "analyses_with_checks": len(recorded),
            **before,
            "quotes_verbatim": _rate(quotes - before["quotes_unquoted"], quotes),
        },
        "confidence": {
            "mean": statistics.fmean(confidences),
            "min": min(confidences),
            "max": max(confidences),
        },
    }


def _rate(count: int, total: int) -> dict[str, Any]:
    return {"count": count, "of": total, "rate": count / total if total else None}


def total_usage(usages: Sequence[Mapping[str, Any] | None]) -> dict[str, Any] | None:
    """Calls, tokens and USD of writing the stored analyses (None when none recorded usage)."""
    rows = [u for u in usages if u and all(name in u for name in USAGE_FIELDS)]
    if not rows:
        return None
    total: dict[str, Any] | None = None
    for row in rows:
        total = add_usage(total, row, str(row.get("model") or ""))
    return total


def _pair_row(item: _Scored) -> dict[str, Any]:
    analysis = item.analysis
    return {
        "base": item.record.base_doc_id,
        "target": item.record.target_doc_id,
        "changed": len(item.record.changed_ids),
        "llm": analysis.stance_shift,
        "confidence": analysis.confidence,
        "key_changes": len(analysis.key_changes),
        "gold": item.gold,
        "zero_shot_delta": item.score_delta,
        "zero_shot_shift": item.delta,
        "rate_decision_shift": item.decision,
    }


# --- Loading from Postgres and writing the results file --------------------------------------


def load_pairs(
    conn: psycopg.Connection[Any], settings: Settings, prompt_version: str | None = None
) -> tuple[list[PairRecord], dict[str, float], str]:
    """Every consecutive pair of stored statements with the analysis stored under its current
    cache key (the model of get_analyzer(settings) and its prompt version, or `prompt_version`,
    an earlier prompt's label), the llm_zero_shot score of every statement, and
    "model/prompt_version"."""
    from app.compare.analysis import get_analyzer
    from app.compare.diff import diff_statements
    from app.compare.types import ComparisonInput, cache_key
    from app.db.compare import all_statements, read_comparison, stances_of

    analyzer = get_analyzer(settings)
    version = prompt_version or analyzer.prompt_version
    statements = all_statements(conn)
    records: list[PairRecord] = []
    decisions = {s.source.doc_id: rate_decision(s.statement.paragraphs) for s in statements}
    for base, target in zip(statements, statements[1:], strict=False):
        diff = diff_statements(base.statement, target.statement)
        comparison = ComparisonInput(base.statement, target.statement, diff.changes)
        key = cache_key(comparison, model=analyzer.model, prompt_version=version)
        stored = read_comparison(conn, key)
        records.append(
            PairRecord(
                base_doc_id=base.source.doc_id,
                target_doc_id=target.source.doc_id,
                target_date=target.source.meeting_date,
                changed_ids=frozenset(change.id for change in comparison.changed),
                analysis=stored.analysis if stored else None,
                usage=stored.usage if stored else None,
                rate_decisions=(decisions[base.source.doc_id], decisions[target.source.doc_id]),
            )
        )
    doc_ids = [s.source.doc_id for s in statements]
    scores = {doc: snap.score for doc, snap in stances_of(conn, doc_ids, ZERO_SHOT).items()}
    return records, scores, f"{analyzer.model}/{version}"


def benchmark_path(settings: Settings | None = None) -> Path:
    return (settings or get_settings()).data_dir / BENCHMARK_FILE


def prompt_note(analyses: str) -> str | None:
    """What the prompt of "model/prompt_version" is (pre-registered or post hoc), if noted."""
    return PROMPT_NOTES.get(analyses.rsplit("/", 1)[-1].split(".")[0])


def write_benchmark(
    path: Path, results: Mapping[str, Any], *, analyses: str, now: datetime | None = None
) -> None:
    """Write the results file atomically, stamped with generated_at and what wrote the
    analyses, floats cut to 5 significant digits."""
    stamp = (now or datetime.now(UTC)).replace(microsecond=0).isoformat()
    head = {"generated_at": stamp, "analyses": analyses, "prompt": prompt_note(analyses)}
    content = rounded({**head, **results})
    write_text_atomic(path, json.dumps(content, indent=2, ensure_ascii=False) + "\n")


def run(settings: Settings | None = None, out: Path | None = None) -> dict[str, Any]:
    """Score the analyses stored in Postgres against data/eval/stance_gold.jsonl and the stored
    llm_zero_shot scores, and write data/eval/compare_benchmark.json (python -m app.compare
    evaluate), returning what it wrote. Makes no LLM call; ValueError when no pair of the current
    prompt has a stored analysis."""
    from app.db.connection import connect
    from app.stance.metrics import read_gold, references
    from app.stance.types import GOLD_FILE

    settings = settings or get_settings()
    refs = references(read_gold(settings.data_dir / GOLD_FILE))
    earlier: list[dict[str, Any]] = []
    with connect(settings) as conn:
        pairs, scores, analyses = load_pairs(conn, settings)
        for version in EARLIER_PROMPTS:
            if version == analyses.rsplit("/", 1)[-1]:
                continue
            old_pairs, old_scores, label = load_pairs(conn, settings, version)
            if any(p.analysis is not None for p in old_pairs):
                scored = evaluate(old_pairs, refs, old_scores)
                del scored["notes"]
                earlier.append({"analyses": label, "prompt": prompt_note(label), **scored})
    results = evaluate(pairs, refs, scores) | {"earlier_prompts": earlier}
    write_benchmark(out or benchmark_path(settings), results, analyses=analyses)
    return {"analyses": analyses, "prompt": prompt_note(analyses), **results}
