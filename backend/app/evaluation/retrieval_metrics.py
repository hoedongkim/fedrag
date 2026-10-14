"""Ranking metrics for one question, latency percentiles, and question-level bootstrap intervals.

Chunk level, over a question's relevant chunks (primary grade 2, supporting grade 1):

- recall@k: relevant chunks in the top k / relevant chunks (every role counts the same);
- MRR@10: 1 / rank of the first relevant chunk, 0 when none is in the top 10;
- nDCG@10: DCG@10 / ideal DCG@10 with linear gain, DCG = sum of grade / log2(rank + 1); the
  ideal ranking puts every relevant chunk first, highest grade first, so a relevant chunk the
  retriever never returns still counts against it.

Document level: doc recall@k is the share of relevant documents with at least one chunk in the
top k.

Ties. When consecutive hits have equal scores their order is arbitrary (common after reciprocal
rank fusion: a chunk ranked r by BM25 alone and one ranked r by dense alone get the same fused
score, and the fusion puts BM25's first). Every metric is therefore the expected value over all
orders of each run of tied hits (McSherry and Najork, ECIR 2008): a run that straddles the
cutoff k contributes its relevant items in proportion to the slots it has in the top k, and
MRR takes the expected reciprocal rank of the first relevant item within the run. A tied run that
reaches the end of the list is treated as ending there (the list is all a retriever returns).
With ties=False the list order is used as is. A chunk id listed twice counts once, at its first
position.

A metric is None for a question that has no labels of its kind (no relevant chunk, or no relevant
document); means and intervals are taken over the questions where it is defined, and say so in n.

Intervals: percentile bootstrap over questions, reusing app.stance.metrics (Bootstrap settings and
draws): 10,000 resamples of the questions with replacement, numpy default_rng seed 20261013,
95%. Every metric on the same number of questions sees the same resamples, and a paired
difference of two methods resamples the questions once and scores both methods on each resample.
"""

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from app.stance.metrics import DEFAULT_BOOTSTRAP, Bootstrap, _draws, _interval

K_VALUES = (5, 10)
DEPTH = max(K_VALUES)  # hits retrieved per question for the metrics
RETRIEVAL_METRICS = (
    "recall_at_5",
    "recall_at_10",
    "mrr_at_10",
    "ndcg_at_10",
    "doc_recall_at_5",
    "doc_recall_at_10",
)
_REL_TOL = 1e-12  # scores this close are tied (RRF sums of the same terms are exactly equal)


class Hit(Protocol):
    """A retrieved passage as the metrics see it."""

    @property
    def chunk_id(self) -> str: ...
    @property
    def doc_id(self) -> str: ...
    @property
    def score(self) -> float: ...


@dataclass(frozen=True)
class RankedHit:
    chunk_id: str
    doc_id: str
    score: float


class Labels(Protocol):
    """What the metrics need from a question (app.evaluation.questions.EvalQuestion)."""

    @property
    def grades(self) -> dict[str, int]: ...
    @property
    def relevant_doc_ids(self) -> Sequence[str]: ...


def unique(ranking: Sequence[Hit]) -> list[Hit]:
    """The ranking with repeated chunk ids dropped after their first position."""
    seen: set[str] = set()
    kept = []
    for hit in ranking:
        if hit.chunk_id not in seen:
            seen.add(hit.chunk_id)
            kept.append(hit)
    return kept


def tied_runs(ranking: Sequence[Hit], *, ties: bool = True) -> list[tuple[int, int]]:
    """[start, end) of each maximal run of consecutive hits with equal scores, in order (every
    hit its own run with ties=False)."""
    runs: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(ranking) + 1):
        if i == len(ranking) or not ties or not _tied(ranking[i - 1].score, ranking[i].score):
            runs.append((start, i))
            start = i
    return runs


def _tied(a: float, b: float) -> bool:
    return a == b or math.isclose(a, b, rel_tol=_REL_TOL, abs_tol=0.0)


def recall_at_k(
    ranking: Sequence[Hit], relevant: Collection[str], k: int, *, ties: bool = True
) -> float | None:
    """Expected share of `relevant` chunk ids in the top k; None without relevant chunks."""
    if not relevant:
        return None
    ranking = unique(ranking)
    found = 0.0
    for start, end in tied_runs(ranking, ties=ties):
        if start >= k:
            break
        hits = sum(hit.chunk_id in relevant for hit in ranking[start:end])
        found += hits * (min(end, k) - start) / (end - start)
    return found / len(relevant)


def reciprocal_rank(
    ranking: Sequence[Hit], relevant: Collection[str], k: int, *, ties: bool = True
) -> float | None:
    """Expected 1 / rank of the first relevant chunk in the top k (0 if none is there); None
    without relevant chunks."""
    if not relevant:
        return None
    ranking = unique(ranking)
    for start, end in tied_runs(ranking, ties=ties):
        if start >= k:
            return 0.0
        size = end - start
        hits = sum(hit.chunk_id in relevant for hit in ranking[start:end])
        if not hits:
            continue
        # P(the first relevant item of the run sits at offset j) = C(size - j, hits - 1) /
        # C(size, hits), for j = 1 .. size - hits + 1.
        expected = 0.0
        for offset in range(1, size - hits + 2):
            rank = start + offset
            if rank > k:
                break
            expected += math.comb(size - offset, hits - 1) / math.comb(size, hits) / rank
        return expected
    return 0.0


def dcg_at_k(
    ranking: Sequence[Hit], grades: Mapping[str, int], k: int, *, ties: bool = True
) -> float:
    """Expected DCG@k with linear gain: each position of a tied run gets the run's mean grade."""
    ranking = unique(ranking)
    total = 0.0
    for start, end in tied_runs(ranking, ties=ties):
        if start >= k:
            break
        mean_grade = sum(grades.get(hit.chunk_id, 0) for hit in ranking[start:end]) / (end - start)
        total += mean_grade * sum(
            1 / math.log2(rank + 1) for rank in range(start + 1, min(end, k) + 1)
        )
    return total


def ndcg_at_k(
    ranking: Sequence[Hit], grades: Mapping[str, int], k: int, *, ties: bool = True
) -> float | None:
    """DCG@k / ideal DCG@k; None without graded chunks."""
    ideal = sorted((g for g in grades.values() if g > 0), reverse=True)[:k]
    best = sum(grade / math.log2(rank + 1) for rank, grade in enumerate(ideal, start=1))
    if best <= 0:
        return None
    return dcg_at_k(ranking, grades, k, ties=ties) / best


def doc_recall_at_k(
    ranking: Sequence[Hit], relevant_docs: Collection[str], k: int, *, ties: bool = True
) -> float | None:
    """Expected share of `relevant_docs` with at least one chunk in the top k; None without
    relevant documents."""
    if not relevant_docs:
        return None
    ranking = unique(ranking)
    runs = tied_runs(ranking, ties=ties)
    found = 0.0
    for doc_id in set(relevant_docs):
        missed = 1.0  # probability that no chunk of the document is in the top k
        for start, end in runs:
            if start >= k:
                break
            mine = sum(hit.doc_id == doc_id for hit in ranking[start:end])
            slots, size = min(end, k) - start, end - start
            if mine:
                missed *= math.comb(size - mine, slots) / math.comb(size, slots)
        found += 1.0 - missed
    return found / len(set(relevant_docs))


def question_metrics(
    ranking: Sequence[Hit], question: Labels, *, ties: bool = True
) -> dict[str, float | None]:
    """Every metric of RETRIEVAL_METRICS for one question's ranking."""
    grades = question.grades
    docs = question.relevant_doc_ids
    return {
        "recall_at_5": recall_at_k(ranking, grades, 5, ties=ties),
        "recall_at_10": recall_at_k(ranking, grades, 10, ties=ties),
        "mrr_at_10": reciprocal_rank(ranking, grades, 10, ties=ties),
        "ndcg_at_10": ndcg_at_k(ranking, grades, 10, ties=ties),
        "doc_recall_at_5": doc_recall_at_k(ranking, docs, 5, ties=ties),
        "doc_recall_at_10": doc_recall_at_k(ranking, docs, 10, ties=ties),
    }


def latency_summary(milliseconds: Sequence[float]) -> dict[str, Any] | None:
    """p50, p95 (numpy's linear interpolation) and mean of the timings; None without any."""
    if not milliseconds:
        return None
    values = np.asarray(milliseconds, dtype=float)
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "mean": float(values.mean()),
        "n": len(values),
    }


def mean_ci(
    values: Sequence[float | None], config: Bootstrap = DEFAULT_BOOTSTRAP
) -> dict[str, Any] | None:
    """Mean of the defined values and its percentile bootstrap interval over questions
    ({"mean", "ci": {"low", "high"}, "n"}); None when no value is defined."""
    present = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(present):
        return None
    resampled = present[_draws(len(present), config)].mean(axis=1)
    return {
        "mean": float(present.mean()),
        "ci": _interval(resampled, config.level),
        "n": len(present),
    }


def paired_difference(
    a: Sequence[float | None],
    b: Sequence[float | None],
    config: Bootstrap = DEFAULT_BOOTSTRAP,
) -> dict[str, Any] | None:
    """Mean of a minus mean of b on the questions where both are defined (`a` and `b` aligned by
    question), with the percentile bootstrap interval of the difference ({"difference",
    "ci": {"low", "high"}, "n"}); None when no question has both."""
    if len(a) != len(b):
        raise ValueError(f"{len(a)} and {len(b)} values: not aligned by question")
    pairs = [(x, y) for x, y in zip(a, b, strict=True) if x is not None and y is not None]
    if not pairs:
        return None
    diffs = np.asarray([x - y for x, y in pairs], dtype=float)
    resampled = diffs[_draws(len(diffs), config)].mean(axis=1)
    return {
        "difference": float(diffs.mean()),
        "ci": _interval(resampled, config.level),
        "n": len(diffs),
    }


def bootstrap_method(config: Bootstrap = DEFAULT_BOOTSTRAP) -> str:
    """The interval method, as the results file states it."""
    return (
        f"Percentile bootstrap over questions: {config.resamples:,} resamples drawn with "
        f"replacement (numpy default_rng seed {config.seed}), {config.level:.0%} intervals. Each "
        "metric uses the questions where it is defined (n). A paired difference resamples the "
        "questions once and scores both runs on each resample. Questions are treated as "
        "independent draws from the kinds of questions the set covers."
    )
