"""Ranking metrics against hand-computed values (graded relevance, ties) and the bootstrap."""

import itertools
import math
import random
from collections.abc import Callable, Sequence

import pytest

from app.evaluation.questions import EvalQuestion
from app.evaluation.retrieval_metrics import (
    RankedHit,
    dcg_at_k,
    doc_recall_at_k,
    latency_summary,
    mean_ci,
    ndcg_at_k,
    paired_difference,
    question_metrics,
    recall_at_k,
    reciprocal_rank,
    tied_runs,
)
from app.stance.metrics import Bootstrap


def ranking(*items: tuple[str, float]) -> list[RankedHit]:
    """Hits named "<doc>/<chunk>" (the doc id is the part before the slash)."""
    return [RankedHit(name, name.split("/")[0], score) for name, score in items]


def log2(rank: int) -> float:
    return math.log2(rank + 1)


# No ties: a (grade 2), x, b (grade 1), y, c (grade 2); d (grade 1) is never retrieved.
ORDERED = ranking(("A/a", 0.9), ("X/x", 0.8), ("B/b", 0.7), ("Y/y", 0.6), ("C/c", 0.5))
GRADES = {"A/a": 2, "B/b": 1, "C/c": 2, "D/d": 1}


def test_recall_counts_every_relevant_chunk_against_all_of_them() -> None:
    assert recall_at_k(ORDERED, GRADES, 5) == 3 / 4
    assert recall_at_k(ORDERED, GRADES, 2) == 1 / 4
    assert recall_at_k(ORDERED, GRADES, 10) == 3 / 4  # d is never retrieved


def test_mrr_is_one_over_the_rank_of_the_first_relevant_chunk() -> None:
    assert reciprocal_rank(ORDERED, GRADES, 10) == 1.0
    assert reciprocal_rank(ORDERED[1:], GRADES, 10) == 1 / 2  # b is now second
    assert reciprocal_rank(ORDERED[1:], {"C/c": 2}, 10) == 1 / 4
    assert reciprocal_rank(ORDERED[1:], {"C/c": 2}, 3) == 0.0  # beyond the cutoff


def test_ndcg_uses_linear_gain_and_the_ideal_order_of_all_graded_chunks() -> None:
    dcg = 2 / log2(1) + 1 / log2(3) + 2 / log2(5)
    ideal = 2 / log2(1) + 2 / log2(2) + 1 / log2(3) + 1 / log2(4)
    assert dcg_at_k(ORDERED, GRADES, 10) == pytest.approx(dcg)
    assert ndcg_at_k(ORDERED, GRADES, 10) == pytest.approx(dcg / ideal)
    assert ndcg_at_k(ORDERED, GRADES, 10) == pytest.approx(0.78085, abs=1e-5)
    # The cutoff applies to both: top 2 holds a only; the ideal top 2 holds a and c.
    assert ndcg_at_k(ORDERED, GRADES, 2) == pytest.approx(2 / (2 + 2 / log2(2)))
    perfect = ranking(("A/a", 4), ("C/c", 3), ("B/b", 2), ("D/d", 1))
    assert ndcg_at_k(perfect, GRADES, 10) == pytest.approx(1.0)


# Ties: x and a share 0.5; y, b, z and w share 0.3.
TIED = ranking(("X/x", 0.5), ("A/a", 0.5), ("Y/y", 0.3), ("B/b", 0.3), ("Z/z", 0.3), ("W/w", 0.3))


def test_tied_runs_group_equal_consecutive_scores() -> None:
    assert tied_runs(TIED) == [(0, 2), (2, 6)]
    assert tied_runs(TIED, ties=False) == [(i, i + 1) for i in range(6)]
    assert tied_runs([]) == []


def test_recall_with_ties_counts_a_straddling_run_in_proportion() -> None:
    relevant = {"A/a", "B/b"}
    # k=1: one of the 2 slots of {x, a} is in the top 1 -> a with probability 1/2.
    assert recall_at_k(TIED, relevant, 1) == pytest.approx(0.5 / 2)
    # k=4: {x, a} in full; 2 of the 4 slots of {y, b, z, w} -> b with probability 1/2.
    assert recall_at_k(TIED, relevant, 4) == pytest.approx(1.5 / 2)
    assert recall_at_k(TIED, relevant, 6) == 1.0
    assert recall_at_k(TIED, relevant, 1, ties=False) == 0.0  # x first, as listed


def test_mrr_with_ties_is_the_expected_reciprocal_rank() -> None:
    # a is first or second with probability 1/2 each.
    assert reciprocal_rank(TIED, {"A/a"}, 10) == pytest.approx(0.5 / 1 + 0.5 / 2)
    assert reciprocal_rank(TIED, {"A/a"}, 1) == pytest.approx(0.5)
    assert reciprocal_rank(TIED, {"A/a"}, 10, ties=False) == 0.5
    # Two relevant in a run of three after one miss: the first relevant is at rank 2 with
    # probability C(2,1)/C(3,2) = 2/3, else at rank 3.
    run = ranking(("X/x", 0.9), ("Y/y", 0.5), ("A/a", 0.5), ("B/b", 0.5))
    assert reciprocal_rank(run, {"A/a", "B/b"}, 10) == pytest.approx((2 / 3) / 2 + (1 / 3) / 3)
    assert reciprocal_rank(run, {"A/a", "B/b"}, 2) == pytest.approx((2 / 3) / 2)


def test_dcg_with_ties_gives_each_position_the_runs_mean_grade() -> None:
    grades = {"A/a": 2, "B/b": 1}
    expected = 1.0 * (1 / log2(1) + 1 / log2(2)) + 0.25 * (1 / log2(3) + 1 / log2(4))
    assert dcg_at_k(TIED, grades, 4) == pytest.approx(expected)
    assert ndcg_at_k(TIED, grades, 4) == pytest.approx(expected / (2 + 1 / log2(2)))


def test_doc_recall_with_ties_is_the_chance_a_chunk_of_the_document_makes_the_cut() -> None:
    two = ranking(("D1/x", 0.5), ("D2/y", 0.5))
    assert doc_recall_at_k(two, {"D2"}, 1) == pytest.approx(0.5)
    assert doc_recall_at_k(two, {"D2"}, 2) == 1.0
    # Two of D3's chunks in a run of four with two slots: 1 - C(2,2)/C(4,2) = 5/6.
    four = ranking(("D1/a", 0.9), ("D3/b", 0.4), ("D3/c", 0.4), ("D2/d", 0.4), ("D2/e", 0.4))
    assert doc_recall_at_k(four, {"D3"}, 3) == pytest.approx(5 / 6)
    assert doc_recall_at_k(four, {"D1", "D3"}, 3) == pytest.approx((1 + 5 / 6) / 2)
    assert doc_recall_at_k(ORDERED, {"A", "C", "Q"}, 5) == pytest.approx(2 / 3)


def _brute_force(
    metric: Callable[..., float | None], hits: Sequence[RankedHit], *args: object
) -> float:
    """The metric averaged over every order of every tied run (list order otherwise)."""
    runs = tied_runs(hits)
    orders = [itertools.permutations(hits[a:b]) for a, b in runs]
    values = []
    for combo in itertools.product(*orders):
        flat = [hit for run in combo for hit in run]
        values.append(metric(flat, *args, ties=False))
    return sum(values) / len(values)


@pytest.mark.parametrize("seed", range(40))
def test_tie_aware_metrics_equal_the_average_over_all_tie_orders(seed: int) -> None:
    rng = random.Random(seed)
    size = rng.randint(1, 7)
    scores = sorted((rng.choice([0.1, 0.2, 0.3]) for _ in range(size)), reverse=True)
    hits = ranking(*((f"D{rng.randint(1, 3)}/c{i}", s) for i, s in enumerate(scores)))
    grades = {h.chunk_id: rng.choice([1, 2]) for h in hits if rng.random() < 0.5} or {"none": 1}
    docs = {f"D{rng.randint(1, 3)}"}
    k = rng.randint(1, 6)
    assert recall_at_k(hits, grades, k) == pytest.approx(_brute_force(recall_at_k, hits, grades, k))
    assert reciprocal_rank(hits, grades, k) == pytest.approx(
        _brute_force(reciprocal_rank, hits, grades, k)
    )
    assert dcg_at_k(hits, grades, k) == pytest.approx(_brute_force(dcg_at_k, hits, grades, k))
    assert doc_recall_at_k(hits, docs, k) == pytest.approx(
        _brute_force(doc_recall_at_k, hits, docs, k)
    )


def test_a_chunk_listed_twice_counts_once_at_its_first_position() -> None:
    hits = ranking(("A/a", 0.9), ("A/a", 0.8), ("B/b", 0.7))
    assert recall_at_k(hits, {"A/a", "B/b"}, 2) == 1.0  # b moves up to second
    assert dcg_at_k(hits, {"A/a": 2}, 10) == 2.0


def test_metrics_are_undefined_without_labels() -> None:
    assert recall_at_k(ORDERED, set(), 5) is None
    assert reciprocal_rank(ORDERED, set(), 10) is None
    assert ndcg_at_k(ORDERED, {}, 10) is None
    assert doc_recall_at_k(ORDERED, (), 5) is None
    assert recall_at_k([], {"A/a"}, 5) == 0.0
    assert reciprocal_rank([], {"A/a"}, 5) == 0.0


def test_question_metrics_read_the_questions_labels() -> None:
    question = EvalQuestion(
        id="q",
        question="What happened?",
        type="multi_doc",
        answerable=True,
        relevant_chunks=[
            {"chunk_id": "A/a", "role": "primary"},
            {"chunk_id": "B/b", "role": "supporting"},
        ],
        relevant_doc_ids=["A", "B", "Q"],
        split="dev",
    )
    metrics = question_metrics(ORDERED, question)
    assert metrics == {
        "recall_at_5": 1.0,
        "recall_at_10": 1.0,
        "mrr_at_10": 1.0,
        "ndcg_at_10": pytest.approx((2 + 1 / log2(3)) / (2 + 1 / log2(2))),
        "doc_recall_at_5": pytest.approx(2 / 3),
        "doc_recall_at_10": pytest.approx(2 / 3),
    }


def test_latency_summary_uses_linear_interpolation() -> None:
    summary = latency_summary([float(v) for v in range(1, 101)])
    assert summary == {"p50": 50.5, "p95": pytest.approx(95.05), "mean": 50.5, "n": 100}
    assert latency_summary([]) is None


FAST = Bootstrap(resamples=2_000, seed=7)


def test_mean_ci_skips_undefined_values_and_is_reproducible() -> None:
    values = [0.0, 1.0, None, 1.0, 0.0, 1.0]
    first = mean_ci(values, FAST)
    assert first == mean_ci(values, FAST)
    assert first is not None
    assert first["mean"] == pytest.approx(0.6) and first["n"] == 5
    assert 0.0 <= first["ci"]["low"] < 0.6 < first["ci"]["high"] <= 1.0
    assert mean_ci([None, None], FAST) is None
    assert mean_ci([0.25], FAST) == {"mean": 0.25, "ci": {"low": 0.25, "high": 0.25}, "n": 1}


def test_paired_difference_resamples_questions_once_for_both_runs() -> None:
    a = [0.5, 0.7, 0.2, 0.9]
    assert paired_difference(a, a, FAST) == {
        "difference": 0.0,
        "ci": {"low": 0.0, "high": 0.0},
        "n": 4,
    }
    shifted = paired_difference([v + 0.1 for v in a], a, FAST)
    assert shifted is not None
    assert shifted["difference"] == pytest.approx(0.1)
    assert shifted["ci"]["low"] == pytest.approx(0.1) and shifted["ci"]["high"] == pytest.approx(
        0.1
    )
    # Pairs with an undefined side are left out.
    partial = paired_difference([1.0, None, 0.0], [0.0, 1.0, None], FAST)
    assert partial == {"difference": 1.0, "ci": {"low": 1.0, "high": 1.0}, "n": 1}
    assert paired_difference([None], [1.0], FAST) is None
    with pytest.raises(ValueError, match="not aligned"):
        paired_difference([1.0], [1.0, 0.0], FAST)
