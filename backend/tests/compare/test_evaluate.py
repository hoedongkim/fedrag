"""Tests for the comparison benchmark on hand-made pairs (no database, no LLM)."""

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from app.compare.evaluate import (
    PairRecord,
    agreement,
    decision_shift,
    delta_shift,
    evaluate,
    gold_shift,
    prompt_note,
    rate_decision,
    write_benchmark,
)
from app.schemas.common import Stance
from app.stance.metrics import Bootstrap, Reference

LABELS: dict[str, Stance] = {
    "s1": "neutral",
    "s2": "hawkish",
    "s3": "hawkish",
    "s4": "dovish",
    "s5": "dovish",
    "s6": "neutral",
}
HUMAN = {"s3", "s4"}
REFS = {
    doc: Reference(doc, label, "human" if doc in HUMAN else "llm_adjudicated", None)
    for doc, label in LABELS.items()
}
SCORES = {"s1": 0.0, "s2": 0.6, "s3": 0.5, "s4": -0.5, "s5": -0.3}


def analysis(shift: str, *key_changes: tuple[list[int], Stance], confidence: float = 0.9) -> Any:
    return {
        "summary": "A reading.",
        "key_changes": [
            {
                "change_ids": ids,
                "what_changed": "Words changed.",
                "interpretation": "A signal.",
                "direction": direction,
            }
            for ids, direction in key_changes
        ],
        "stance_shift": shift,
        "confidence": confidence,
    }


def usage(calls: int = 1, **checks: int) -> dict[str, Any]:
    return {
        "model": "gpt-5.4-mini-2026-03-17",
        "calls": calls,
        "input_tokens": 1_000 * calls,
        "cached_input_tokens": 0,
        "output_tokens": 500 * calls,
        "reasoning_tokens": 10 * calls,
        "usd": 0.003 * calls,
        "checks": checks,
    }


def pair(
    n: int,
    changed: set[int],
    stored: Any,
    used: Any = None,
    *,
    year: int = 2020,
    decisions: tuple[int | None, int | None] = (None, None),
) -> PairRecord:
    return PairRecord(
        base_doc_id=f"s{n}",
        target_doc_id=f"s{n + 1}",
        target_date=date(year, 1, n + 1),
        changed_ids=frozenset(changed),
        analysis=stored,
        usage=used,
        rate_decisions=decisions,
    )


PAIRS = [
    # Listed out of order on purpose: evaluate() sorts by target date.
    pair(3, {1}, analysis("more_dovish", ([1, 7], "dovish")), None, year=2021),
    pair(
        1,
        {0, 1, 2},
        analysis("more_hawkish", ([0, 1], "hawkish")),
        usage(quotes=4, quotes_unquoted=1, invalid_ids=2),
    ),
    pair(2, {3}, analysis("more_dovish", ([3], "neutral")), usage(quotes=2)),
    pair(4, set(), analysis("unchanged", confidence=1.0), usage(calls=0), year=2021),
    pair(5, {0}, None, year=2021),  # nothing stored
    pair(6, {0}, analysis("unchanged", confidence=2.0), usage(), year=2021),  # fails validation
]


def test_gold_shift_is_ordinal() -> None:
    assert gold_shift("neutral", "hawkish") == "more_hawkish"
    assert gold_shift("hawkish", "dovish") == "more_dovish"
    assert gold_shift("dovish", "neutral") == "more_hawkish"
    assert gold_shift("neutral", "neutral") == "unchanged"


def test_delta_shift_threshold_is_inclusive() -> None:
    assert delta_shift(0.3, 0.5) == "unchanged"  # 0.2 exactly, despite float rounding
    assert delta_shift(-0.5, -0.3) == "unchanged"
    assert delta_shift(0.3, 0.51) == "more_hawkish"
    assert delta_shift(0.1, -0.15) == "more_dovish"
    assert delta_shift(0.0, 0.3, threshold=0.5) == "unchanged"


def test_agreement_on_all_pairs_and_on_the_subsets() -> None:
    results = evaluate(PAIRS, REFS, SCORES, bootstrap=None)
    gold = results["agreement"]["llm_vs_gold"]

    assert (gold["all"]["n"], gold["all"]["accuracy"]) == (4, 0.75)
    assert gold["all"]["confusion_matrix"] == [[1, 0, 0], [1, 1, 0], [0, 0, 1]]
    assert gold["all"]["predicted"] == {"more_dovish": 2, "unchanged": 1, "more_hawkish": 1}
    assert gold["all"]["per_class"]["more_hawkish"]["f1"] == 1.0
    assert (gold["reference_changed"]["n"], gold["reference_changed"]["accuracy"]) == (2, 1.0)
    # no "unchanged" reference there: macro-F1 would average in that class's 0
    assert "macro_f1" not in gold["reference_changed"]
    assert gold["reference_changed"]["per_class"]["more_hawkish"]["f1"] == 1.0
    unchanged = gold["reference_unchanged"]
    assert (unchanged["n"], unchanged["accuracy"]) == (2, 0.5)
    assert "macro_f1" not in unchanged  # one reference class: macro-F1 says nothing
    assert results["agreement"]["llm_vs_gold_both_human"]["n"] == 1  # s3 -> s4

    delta = results["agreement"]["llm_vs_zero_shot_delta"]["all"]
    assert (delta["n"], delta["accuracy"]) == (4, 0.75)
    baselines = results["baselines"]
    assert baselines["always_unchanged_vs_gold"]["all"]["accuracy"] == 0.5
    assert baselines["always_unchanged_vs_gold"]["reference_changed"]["accuracy"] == 0.0
    assert baselines["zero_shot_delta_vs_gold"]["all"]["accuracy"] == 1.0
    assert "macro_f1" not in baselines["zero_shot_delta_vs_gold"]["reference_changed"]
    for name in ("always_unchanged", "zero_shot_delta", "rate_decision"):
        assert baselines[f"{name}_vs_gold_both_human"]["n"] == 1  # s3 -> s4
    assert baselines["always_unchanged_vs_gold_both_human"]["accuracy"] == 0.0
    assert baselines["zero_shot_delta_vs_gold_both_human"]["accuracy"] == 1.0
    # without rate decisions the rule says "unchanged" everywhere
    assert baselines["rate_decision_vs_gold"] == baselines["always_unchanged_vs_gold"]
    assert results["shift_counts"]["gold_implied"] == {
        "more_dovish": 1,
        "unchanged": 2,
        "more_hawkish": 1,
    }
    assert results["shift_counts"]["zero_shot_delta"]["unchanged"] == 2


def test_rate_decision_reads_the_first_rate_action_sentence() -> None:
    hike = "The Committee decided to raise the target range to 1 to 1-1/4 percent."
    hold = "The Committee decided to maintain the target range at 1 to 1-1/4 percent."
    cut = "The Committee decided to lower the target range to 3/4 to 1 percent."
    purchases = "The Committee will increase its holdings of Treasury securities."

    assert rate_decision([f"Job gains slowed. {hike}", hold]) == 1  # the first one counts
    assert rate_decision([hold, cut]) == 0
    assert rate_decision([purchases, cut]) == -1
    assert rate_decision([purchases]) is None


def test_decision_shift_compares_decisions_not_levels() -> None:
    assert decision_shift(0, 1) == "more_hawkish"  # a hike after a hold
    assert decision_shift(1, 0) == "more_dovish"  # a hold after a hike
    assert decision_shift(0, -1) == "more_dovish"  # a cut after a hold
    assert decision_shift(-1, 0) == "more_hawkish"  # a hold after a cut
    assert decision_shift(1, 1) == "unchanged"  # a hike after a hike
    assert decision_shift(None, 0) is None and decision_shift(1, None) is None


def test_rate_decision_baseline_and_same_label_split() -> None:
    pairs = [
        # gold more_hawkish, decision hold -> hike
        pair(1, {0}, analysis("more_hawkish", ([0], "hawkish")), decisions=(0, 1)),
        # gold unchanged (hawkish -> hawkish), decision hike -> hold
        pair(2, {0}, analysis("more_dovish", ([0], "dovish")), decisions=(1, 0)),
        # gold more_dovish, no decision in the target: the rule says unchanged
        pair(3, {0}, analysis("more_dovish", ([0], "dovish")), decisions=(0, None)),
        # gold unchanged (dovish -> dovish), decision hold -> hold
        pair(4, {0}, analysis("more_dovish", ([0], "dovish")), decisions=(0, 0)),
    ]

    results = evaluate(pairs, REFS, SCORES, bootstrap=None)

    rule = results["baselines"]["rate_decision_vs_gold"]
    assert (rule["all"]["n"], rule["all"]["accuracy"]) == (4, 0.5)
    assert rule["all"]["predicted"] == {"more_dovish": 1, "unchanged": 2, "more_hawkish": 1}
    assert rule["reference_changed"]["accuracy"] == 0.5
    assert rule["reference_unchanged"]["accuracy"] == 0.5
    assert results["baselines"]["rate_decision_vs_gold_both_human"]["accuracy"] == 0.0
    split = results["same_label_by_rate_decision"]
    assert split["decision_changed"] == {
        "pairs": 1,
        "llm_shift": {"count": 1, "of": 1, "rate": 1.0},
        "zero_shot_delta_shift": {"count": 0, "of": 1, "rate": 0.0},
        "llm_in_decision_direction": {"count": 1, "of": 1, "rate": 1.0},
    }
    assert split["decision_same"]["pairs"] == 1
    assert split["decision_same"]["llm_shift"]["count"] == 1
    assert split["no_decision"]["llm_shift"] == {"count": 0, "of": 0, "rate": None}


def test_coverage_rows_and_cost() -> None:
    results = evaluate(PAIRS, REFS, SCORES, bootstrap=None)

    assert results["coverage"] == {
        "pairs": 6,
        "analysed": 4,
        "missing": [["s5", "s6"]],
        "invalid_stored": [["s6", "s7"]],
        "with_gold": 4,
        "with_gold_both_human": 1,
        "with_zero_shot": 4,
    }
    assert [row["target"] for row in results["pairs"]] == ["s2", "s3", "s4", "s5"]
    assert results["pairs"][3] | {"zero_shot_delta": None} == {
        "base": "s4",
        "target": "s5",
        "changed": 0,
        "llm": "unchanged",
        "confidence": 1.0,
        "key_changes": 0,
        "gold": "unchanged",
        "zero_shot_delta": None,
        "zero_shot_shift": "unchanged",
        "rate_decision_shift": None,
    }
    cost = results["cost"]
    assert (cost["calls"], cost["input_tokens"], cost["output_tokens"]) == (2, 2_000, 1_000)
    # 2,000 input at $0.75/M plus 1,000 output at $4.50/M
    assert cost["usd"] == pytest.approx(0.006, abs=1e-4)


def test_structural_checks() -> None:
    checks = evaluate(PAIRS, REFS, SCORES, bootstrap=None)["structural"]

    assert (checks["pairs_with_changes"], checks["pairs_without_changes"]) == (3, 1)
    assert checks["key_changes_present_when_changed"] == {"count": 3, "of": 3, "rate": 1.0}
    assert checks["cited_ids_not_changed_items"] == {"count": 1, "of": 5}  # id 7 of s3 -> s4
    assert checks["shift_backed_by_key_change"]["count"] == 2  # not s2 -> s3 (only neutral)
    assert checks["key_changes_without_quote"] == {"count": 3, "of": 3, "rate": 1.0}
    before = checks["before_cleaning"]
    assert before["analyses_with_checks"] == 3  # s3 -> s4 has no usage
    assert (before["quotes"], before["quotes_unquoted"], before["invalid_ids"]) == (6, 1, 2)
    assert before["quotes_verbatim"]["rate"] == pytest.approx(5 / 6)
    assert checks["confidence"]["max"] == 1.0


def test_key_changes_without_a_quote_are_counted() -> None:
    stored = analysis("more_dovish", ([0], "dovish"), ([0], "dovish"), ([0], "neutral"))
    for key_change, text in zip(
        stored["key_changes"],
        [
            "“slowed” replaced “moderated”.",
            '"slowed" replaced "moderated".',
            "The Committee said it",
        ],
        strict=True,
    ):
        key_change["what_changed"] = text

    checks = evaluate([pair(1, {0}, stored)], REFS, SCORES, bootstrap=None)["structural"]

    assert checks["key_changes_without_quote"] == {"count": 1, "of": 3, "rate": 1 / 3}


def test_prompt_notes_say_which_prompt_is_post_hoc() -> None:
    assert prompt_note("m/compare-v2.3b53da07").startswith("pre-registered")
    assert prompt_note("m/compare-v3.0123abcd").startswith("post hoc")
    assert prompt_note("m/compare-v1.0123abcd") is None


def test_bootstrap_intervals_bracket_the_estimate() -> None:
    reference = ["more_hawkish", "unchanged", "more_dovish", "unchanged"] * 5
    predicted = ["more_hawkish", "more_dovish", "more_dovish", "unchanged"] * 5
    years = [2019 + i % 4 for i in range(20)]

    report = agreement(reference, predicted, years, Bootstrap(resamples=500))  # type: ignore[arg-type]

    assert report is not None and report["accuracy"] == 0.75
    for name in ("accuracy_ci", "accuracy_ci_year_blocks", "macro_f1_ci"):
        assert report[name]["low"] <= report[name.split("_ci")[0]] <= report[name]["high"]
    assert agreement([], [], [], None) is None


def test_no_stored_analysis_is_an_error() -> None:
    with pytest.raises(ValueError, match="no stored analysis"):
        evaluate([pair(5, {0}, None)], REFS, SCORES)


def test_results_file_holds_metrics_only(tmp_path: Path) -> None:
    results = evaluate(PAIRS, REFS, SCORES, bootstrap=Bootstrap(resamples=200))
    path = tmp_path / "compare_benchmark.json"

    write_benchmark(
        path, results, analyses="m/compare-v2.x", now=datetime(2026, 10, 13, tzinfo=UTC)
    )

    content = json.loads(path.read_text())
    assert content["generated_at"] == "2026-10-13T00:00:00+00:00"
    assert content["analyses"] == "m/compare-v2.x" and content["threshold"] == 0.2
    assert content["prompt"].startswith("pre-registered")
    assert "Words changed." not in path.read_text() and "A reading." not in path.read_text()
    verbatim = content["structural"]["before_cleaning"]["quotes_verbatim"]
    assert verbatim["rate"] == 0.83333  # 5 / 6, cut to 5 significant digits
    assert set(content["agreement"]["llm_vs_gold"]["all"]["accuracy_ci"]) == {"low", "high"}


def test_load_pairs_reads_each_pair_under_its_current_cache_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.compare import evaluate as module
    from app.compare.analysis import get_analyzer
    from app.compare.diff import diff_statements
    from app.compare.types import ComparisonInput, StatementText, cache_key
    from app.core.config import Settings
    from app.db import compare as db
    from app.schemas.common import SourceDocument
    from app.schemas.compare import StanceSnapshot

    def record(doc_id: str, day: date, text: str) -> db.StatementRecord:
        source = SourceDocument(
            doc_id=doc_id,
            doc_type="statement",
            meeting_date=day,
            release_date=day,
            page_updated=None,
            title=doc_id,
            url=f"https://example.test/{doc_id}",
        )
        return db.StatementRecord(source, StatementText(doc_id, day, day, (text,)))

    first = record("statement-2024-01-31", date(2024, 1, 31), "Rates held. Job gains moderated.")
    second = record("statement-2024-03-20", date(2024, 3, 20), "Rates held. Job gains strong.")
    third = record("statement-2024-05-01", date(2024, 5, 1), "Rates held. Job gains strong.")
    settings = Settings(_env_file=None, openai_api_key=None)
    analyzer = get_analyzer(settings)
    diff = diff_statements(first.statement, second.statement)
    stored_key = cache_key(
        ComparisonInput(first.statement, second.statement, diff.changes),
        model=analyzer.model,
        prompt_version=analyzer.prompt_version,
    )
    asked: list[str] = []

    def read(conn: Any, key: str) -> db.StoredComparison | None:
        asked.append(key)
        if key != stored_key:
            return None
        row = analysis("unchanged", ([1], "hawkish"))
        return db.StoredComparison(
            key, first.source.doc_id, second.source.doc_id, "m", "p", row, usage(), datetime.now()
        )

    snapshot = StanceSnapshot(classifier="llm_zero_shot", stance="neutral", score=0.1)
    monkeypatch.setattr(db, "all_statements", lambda conn: [first, second, third])
    monkeypatch.setattr(db, "read_comparison", read)
    monkeypatch.setattr(db, "stances_of", lambda conn, ids, name: {ids[0]: snapshot})

    pairs, scores, analyses = module.load_pairs(None, settings)  # type: ignore[arg-type]

    assert [(p.base_doc_id, p.target_doc_id) for p in pairs] == [
        ("statement-2024-01-31", "statement-2024-03-20"),
        ("statement-2024-03-20", "statement-2024-05-01"),
    ]
    assert len(asked) == 2 and asked[0] == stored_key
    assert pairs[0].analysis is not None and pairs[1].analysis is None
    assert pairs[0].changed_ids == {1} and pairs[1].changed_ids == frozenset()
    assert scores == {"statement-2024-01-31": 0.1}
    assert analyses == f"{analyzer.model}/{analyzer.prompt_version}"

    asked.clear()
    earlier, _, label = module.load_pairs(None, settings, "compare-v0.00000000")  # type: ignore[arg-type]
    assert label == f"{analyzer.model}/compare-v0.00000000"
    assert all(p.analysis is None for p in earlier) and stored_key not in asked
