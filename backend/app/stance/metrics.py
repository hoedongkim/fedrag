"""Stance classification metrics and the two benchmarks: TDW sentences and gold statement labels.

Every report uses the three stances in STANCES order (dovish, neutral, hawkish): macro-F1 averages
the three per-class F1 scores even when a class is absent, confusion matrices have gold labels as
rows and predictions as columns, and a precision with no predictions of its class is 0. Summaries
over seeds give the mean and the sample standard deviation (ddof=1) of each metric.

Statement-level intervals are percentile bootstrap intervals: the scored statements (sorted by
doc_id) are resampled with replacement Bootstrap.resamples times from a generator seeded with
Bootstrap.seed, accuracy and macro-F1 (same conventions as above) are recomputed on each resample,
and the interval runs between the (1 - level) / 2 and (1 + level) / 2 quantiles. Every row on the
same statements therefore sees the same resamples, and a paired comparison of two rows scores both
on each resample and takes the quantiles of the difference.

That resampling treats the statements as independent, but consecutive statements are
near-duplicates, so the intervals understate the uncertainty. As a sensitivity check the same
intervals are also computed with whole calendar years resampled (`groups`: a year's statements
are drawn together, as many draws as there are years).
"""

import json
import math
import statistics
import warnings
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from scipy.stats import spearmanr
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from app.core.config import Settings
from app.schemas.common import Stance
from app.schemas.stance import ReferenceSource
from app.stance import tdw
from app.stance.types import ORDINAL, STANCES, SentenceClassifier, StatementPrediction

SplitLoader = Callable[[tdw.Split, int], Sequence[tdw.LabeledSentence]]
Scores = Callable[[np.ndarray, np.ndarray], dict[str, np.ndarray]]
Labeler = Callable[[Sequence[tdw.LabeledSentence]], list[Stance]]

SCALARS = ("macro_f1", "accuracy", "kappa")
CLASS_METRICS = ("precision", "recall", "f1")
BOOTSTRAP_METRICS = ("accuracy", "macro_f1")


def _finite(value: float) -> float | None:
    """A float JSON can hold: NaN (e.g. kappa when only one label occurs) becomes None."""
    return None if math.isnan(value) else float(value)


def classification_metrics(gold: Sequence[Stance], pred: Sequence[Stance]) -> dict[str, Any]:
    """Macro-F1, accuracy, Cohen's kappa, per-class precision/recall/F1/support and the confusion
    matrix of predictions against gold labels."""
    if len(gold) != len(pred):
        raise ValueError(f"{len(gold)} gold labels but {len(pred)} predictions")
    if not gold:
        raise ValueError("no labels to evaluate")
    labels = list(STANCES)
    precision, recall, f1, support = precision_recall_fscore_support(
        gold, pred, labels=labels, zero_division=0
    )
    with warnings.catch_warnings():  # undefined kappa (one label in common) is reported as None
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        kappa = cohen_kappa_score(gold, pred, labels=labels)
    return {
        "n": len(gold),
        "macro_f1": float(f1.mean()),
        "accuracy": float(accuracy_score(gold, pred)),
        "kappa": _finite(kappa),
        "per_class": {
            stance: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, stance in enumerate(STANCES)
        },
        "confusion_matrix": confusion_matrix(gold, pred, labels=labels).tolist(),
    }


@dataclass(frozen=True)
class Bootstrap:
    """Settings of the statement-level percentile bootstrap."""

    resamples: int = 10_000
    seed: int = 20261013  # fixed before any interval was computed
    level: float = 0.95

    @property
    def method(self) -> str:
        return (
            f"Percentile bootstrap: {self.resamples:,} resamples of the scored statements "
            f"(sorted by doc_id, drawn with replacement; numpy default_rng seed {self.seed}), "
            f"accuracy and macro-F1 recomputed on each; {self.level:.0%} intervals. Every row on "
            "the same statements sees the same resamples; a paired difference scores both rows "
            "on each resample. Statements are resampled as if independent, but consecutive "
            "statements are near-duplicates, so these intervals understate the uncertainty."
        )

    @property
    def year_blocks_method(self) -> str:
        return (
            f"Sensitivity check: the same {self.resamples:,} resamples, but drawing whole calendar "
            "years of statements with replacement (as many draws as there are years) instead of "
            f"single statements; numpy default_rng seed {self.seed}; {self.level:.0%} intervals."
        )


DEFAULT_BOOTSTRAP = Bootstrap()


def _codes(labels: Sequence[Stance]) -> np.ndarray:
    index = {stance: i for i, stance in enumerate(STANCES)}
    return np.array([index[label] for label in labels], dtype=np.int8)


def _resampled_scores(
    gold: np.ndarray, pred: np.ndarray, draws: np.ndarray
) -> dict[str, np.ndarray]:
    """Accuracy and macro-F1 of each resample (a row of item indices in `draws`), with the
    conventions of classification_metrics: a class with no gold and no predicted item has F1 0."""
    g, p = gold[draws], pred[draws]
    f1 = np.zeros(draws.shape[0])
    for c in range(len(STANCES)):
        tp = ((g == c) & (p == c)).sum(axis=1)
        wrong = ((g == c) != (p == c)).sum(axis=1)  # false positives plus false negatives
        f1 += np.divide(2 * tp, 2 * tp + wrong, out=np.zeros(len(tp)), where=2 * tp + wrong > 0)
    return {"accuracy": (g == p).mean(axis=1), "macro_f1": f1 / len(STANCES)}


def _draws(n: int, config: Bootstrap) -> np.ndarray:
    if n < 1 or config.resamples < 1 or not 0 < config.level < 1:
        raise ValueError(f"cannot bootstrap {n} items with {config}")
    return np.random.default_rng(config.seed).integers(0, n, size=(config.resamples, n))


def _weighted_scores(
    gold: np.ndarray, pred: np.ndarray, weights: np.ndarray
) -> dict[str, np.ndarray]:
    """As _resampled_scores, with each resample given as item multiplicities (a row of
    `weights`) instead of item indices."""
    w = weights.astype(float)
    f1 = np.zeros(w.shape[0])
    for c in range(len(STANCES)):
        tp = w @ ((gold == c) & (pred == c))
        wrong = w @ ((gold == c) != (pred == c))
        f1 += np.divide(2 * tp, 2 * tp + wrong, out=np.zeros(len(tp)), where=2 * tp + wrong > 0)
    return {"accuracy": (w @ (gold == pred)) / w.sum(axis=1), "macro_f1": f1 / len(STANCES)}


def _block_weights(groups: Sequence[int], config: Bootstrap) -> np.ndarray:
    """Item multiplicities, one row per resample, when whole groups (e.g. the statements of one
    calendar year) are drawn with replacement, as many draws as there are groups."""
    keys = sorted(set(groups))
    if not keys or config.resamples < 1 or not 0 < config.level < 1:
        raise ValueError(f"cannot bootstrap {len(groups)} items with {config}")
    rows = np.random.default_rng(config.seed).integers(
        0, len(keys), size=(config.resamples, len(keys))
    )
    counts = np.zeros(rows.shape, dtype=np.int64)
    np.add.at(counts, (np.arange(config.resamples)[:, np.newaxis], rows), 1)
    return counts[:, [keys.index(group) for group in groups]]


def _resampler(n: int, config: Bootstrap, groups: Sequence[int] | None) -> Scores:
    """Scores of (gold codes, predicted codes) on each resample of n items: single items, or
    whole groups when `groups` (one per item) is given. Every call sees the same resamples."""
    if groups is None:
        draws = _draws(n, config)
        return lambda gold, pred: _resampled_scores(gold, pred, draws)
    if len(groups) != n:
        raise ValueError(f"{len(groups)} groups for {n} items")
    weights = _block_weights(groups, config)
    return lambda gold, pred: _weighted_scores(gold, pred, weights)


def _interval(values: np.ndarray, level: float) -> dict[str, float]:
    low, high = np.quantile(values, [(1 - level) / 2, (1 + level) / 2])
    return {"low": float(low), "high": float(high)}


def bootstrap_ci(
    gold: Sequence[Stance],
    pred: Sequence[Stance],
    config: Bootstrap = DEFAULT_BOOTSTRAP,
    groups: Sequence[int] | None = None,
) -> dict[str, dict[str, float]]:
    """Percentile bootstrap intervals of accuracy and macro-F1 ({"accuracy": {"low", "high"},
    "macro_f1": {...}}), resampling the items with replacement (whole groups with `groups`)."""
    if len(gold) != len(pred):
        raise ValueError(f"{len(gold)} gold labels but {len(pred)} predictions")
    scores = _resampler(len(gold), config, groups)(_codes(gold), _codes(pred))
    return {name: _interval(scores[name], config.level) for name in BOOTSTRAP_METRICS}


def paired_bootstrap(
    gold: Sequence[Stance],
    pred_a: Sequence[Stance],
    pred_b: Sequence[Stance],
    config: Bootstrap = DEFAULT_BOOTSTRAP,
    groups: Sequence[int] | None = None,
) -> dict[str, dict[str, Any]]:
    """Accuracy and macro-F1 of a minus those of b on the same items: the observed difference and
    its percentile interval, both scored on each resample ({"accuracy": {"difference", "ci"},
    ...}); whole groups are resampled with `groups`."""
    if not len(gold) == len(pred_a) == len(pred_b):
        raise ValueError(f"{len(gold)} gold labels, {len(pred_a)} and {len(pred_b)} predictions")
    g, a, b = _codes(gold), _codes(pred_a), _codes(pred_b)
    scores = _resampler(len(gold), config, groups)
    resampled_a, resampled_b = scores(g, a), scores(g, b)
    whole = np.arange(len(gold))[np.newaxis, :]
    observed_a, observed_b = _resampled_scores(g, a, whole), _resampled_scores(g, b, whole)
    return {
        name: {
            "difference": float(observed_a[name][0] - observed_b[name][0]),
            "ci": _interval(resampled_a[name] - resampled_b[name], config.level),
        }
        for name in BOOTSTRAP_METRICS
    }


def mean_sd(values: Sequence[float | None]) -> dict[str, float | None]:
    """Mean and sample standard deviation of per-seed values (None ignored); sd is None for fewer
    than two values."""
    present = [v for v in values if v is not None]
    return {
        "mean": statistics.fmean(present) if present else None,
        "sd": statistics.stdev(present) if len(present) > 1 else None,
    }


def summarize(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Mean +- sd over seeds of classification_metrics reports; confusion matrices are summed."""
    if not reports:
        raise ValueError("no reports to summarize")
    matrix = [
        [sum(r["confusion_matrix"][i][j] for r in reports) for j in range(3)] for i in range(3)
    ]
    return {
        "n": [r["n"] for r in reports],
        **{name: mean_sd([r[name] for r in reports]) for name in SCALARS},
        "per_class": {
            stance: {
                name: mean_sd([r["per_class"][stance][name] for r in reports])
                for name in CLASS_METRICS
            }
            | {"support": [r["per_class"][stance]["support"] for r in reports]}
            for stance in STANCES
        },
        "confusion_matrix_total": matrix,
    }


def evaluate_sentences(
    classifier_factory: Callable[[int], SentenceClassifier],
    seeds: Sequence[int] = tdw.SEEDS,
    *,
    load: SplitLoader | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """The TDW sentence benchmark: for each seed, build classifier_factory(seed) (it may only learn
    from that seed's train split), label the seed's test split once, and score it on the official
    test split and on the deduplicated one (test minus sentences whose text is in that seed's
    train split, as tdw.deduplicated_test). `load(split, seed)` defaults to tdw.load."""
    name = ""

    def prepare(seed: int) -> Labeler:
        nonlocal name
        classifier = classifier_factory(seed)
        name = classifier.name

        def label(test: Sequence[tdw.LabeledSentence]) -> list[Stance]:
            predictions = classifier.predict_sentences([s.sentence for s in test])
            if len(predictions) != len(test):
                raise ValueError(
                    f"{name} returned {len(predictions)} labels for {len(test)} sentences"
                )
            return [p.stance for p in predictions]

        return label

    report = _sentence_benchmark(prepare, seeds, load or _tdw_loader(settings))
    return {"classifier": name, **report}


def majority_label(labels: Sequence[Stance]) -> Stance:
    """The most frequent label; a tie goes to the first in STANCES order."""
    if not labels:
        raise ValueError("no labels")
    counts = Counter(labels)
    return max(STANCES, key=lambda stance: counts[stance])


def majority_class_sentences(
    seeds: Sequence[int] = tdw.SEEDS,
    *,
    load: SplitLoader | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """The majority-class baseline of the TDW sentence benchmark, scored like evaluate_sentences:
    each seed's most frequent train label, predicted for every test sentence. label is that class
    when it is the same for every seed, else None."""
    load = load or _tdw_loader(settings)
    labels: dict[str, Stance] = {}

    def prepare(seed: int) -> Labeler:
        majority = labels[str(seed)] = majority_label([s.stance for s in load("train", seed)])
        return lambda test: [majority] * len(test)

    report = _sentence_benchmark(prepare, seeds, load)
    distinct = set(labels.values())
    return {
        "name": "majority_class",
        "label": distinct.pop() if len(distinct) == 1 else None,
        "labels": labels,
        **report,
    }


def _tdw_loader(settings: Settings | None) -> SplitLoader:
    return lambda split, seed: tdw.load(split, seed, settings)


def _sentence_benchmark(
    prepare: Callable[[int], Labeler], seeds: Sequence[int], load: SplitLoader
) -> dict[str, Any]:
    """Per seed, prepare(seed) builds the labeller (from the train split only) before any test
    sentence is read; it labels the test split once, scored on the official and the deduplicated
    test split, then summarized over the seeds."""
    per_seed: dict[str, list[dict[str, Any]]] = {"official": [], "deduplicated": []}
    for seed in seeds:
        label = prepare(seed)
        train_texts = {s.sentence for s in load("train", seed)}
        test = list(load("test", seed))
        pred = label(test)
        gold = [s.stance for s in test]
        keep = [i for i, s in enumerate(test) if s.sentence not in train_texts]
        per_seed["official"].append({"seed": seed, **classification_metrics(gold, pred)})
        per_seed["deduplicated"].append(
            {
                "seed": seed,
                **classification_metrics([gold[i] for i in keep], [pred[i] for i in keep]),
            }
        )
    return {
        "seeds": list(seeds),
        **{split: {**summarize(rows), "per_seed": rows} for split, rows in per_seed.items()},
    }


@dataclass(frozen=True)
class Reference:
    """A statement's reference stance: the human label when the statement was re-labelled blind,
    otherwise the LLM-adjudicated label of the gold file."""

    doc_id: str
    label: Stance
    source: ReferenceSource
    confidence: str | None  # "high" / "medium" / "low" from the gold file; None for human labels


def read_gold(path: Path) -> list[dict[str, Any]]:
    """The rows of data/eval/stance_gold.jsonl (one JSON object per line)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _human_label(row: Mapping[str, Any]) -> Any:
    """The row's human_label; None when it is missing or blank, as the API reads the file."""
    human = row.get("human_label")
    return None if isinstance(human, str) and not human.strip() else human


def references(rows: Sequence[Mapping[str, Any]]) -> dict[str, Reference]:
    """Effective reference label per doc_id: human_label if set, else label."""
    result: dict[str, Reference] = {}
    for row in rows:
        human = _human_label(row)
        label = human if human is not None else row["label"]
        if label not in STANCES:
            raise ValueError(f"{row['doc_id']}: unknown stance {label!r}")
        result[row["doc_id"]] = Reference(
            doc_id=row["doc_id"],
            label=label,
            source="human" if human is not None else "llm_adjudicated",
            confidence=None if human is not None else row.get("confidence"),
        )
    return result


def gold_agreement(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """How often the LLM-adjudicated label matches the blind human label, on the statements a
    human re-labelled; None until there are any."""
    pairs = [(human, row["label"]) for row in rows if (human := _human_label(row)) is not None]
    if not pairs:
        return None
    human, llm = zip(*pairs, strict=True)
    return classification_metrics(list(human), list(llm))


class StatementCall(Protocol):
    """What scoring needs from a statement-level call: a StatementPrediction or a BaselineCall."""

    @property
    def doc_id(self) -> str: ...
    @property
    def meeting_date(self) -> date: ...
    @property
    def stance(self) -> Stance: ...
    @property
    def score(self) -> float: ...


@dataclass(frozen=True)
class BaselineCall:
    """A statement's call by a baseline that uses no classifier; score is ORDINAL[stance]."""

    doc_id: str
    meeting_date: date
    stance: Stance
    score: float


def evaluate_statements(
    predictions: Sequence[StatementPrediction],
    refs: Mapping[str, Reference],
    bootstrap: Bootstrap | None = None,
) -> dict[str, Any]:
    """Score one classifier's statement predictions against the reference labels: on every
    statement with a reference, and on the human-labelled subset alone. spearman is the rank
    correlation of score with the reference stance (-1, 0, +1), free of the label thresholds.
    With `bootstrap`, each subset also gets accuracy_ci and macro_f1_ci, and the year-block
    sensitivity intervals accuracy_ci_year_blocks and macro_f1_ci_year_blocks."""
    scored = score_statements(predictions, refs, bootstrap)
    return {
        "classifier": next(p for p in predictions if p.doc_id in refs).classifier,
        **scored,
        "without_reference": sorted(p.doc_id for p in predictions if p.doc_id not in refs),
    }


def score_statements(
    calls: Sequence[StatementCall],
    refs: Mapping[str, Reference],
    bootstrap: Bootstrap | None = None,
) -> dict[str, Any]:
    """The scores of evaluate_statements ({"all": ..., "human_subset": ...}) for any calls."""
    scored = [c for c in calls if c.doc_id in refs]
    if not scored:
        raise ValueError("no predictions have a reference label")
    human = [c for c in scored if refs[c.doc_id].source == "human"]
    return {
        "all": _statement_report(scored, refs, bootstrap),
        "human_subset": _statement_report(human, refs, bootstrap) if human else None,
    }


def _statement_report(
    calls: Sequence[StatementCall],
    refs: Mapping[str, Reference],
    bootstrap: Bootstrap | None = None,
) -> dict[str, Any]:
    gold = [refs[c.doc_id].label for c in calls]
    report = classification_metrics(gold, [c.stance for c in calls])
    scores, ordinals = [c.score for c in calls], [ORDINAL[g] for g in gold]
    rho = None
    if len(set(scores)) > 1 and len(set(ordinals)) > 1:  # undefined for a constant input
        rho = _finite(float(spearmanr(scores, ordinals).statistic))
    report["spearman"] = rho
    if bootstrap:
        ordered = sorted(calls, key=lambda c: c.doc_id)
        labels = [refs[c.doc_id].label for c in ordered]
        stances = [c.stance for c in ordered]
        cis = bootstrap_ci(labels, stances, bootstrap)
        years = bootstrap_ci(labels, stances, bootstrap, [c.meeting_date.year for c in ordered])
        report |= {f"{name}_ci": cis[name] for name in BOOTSTRAP_METRICS}
        report |= {f"{name}_ci_year_blocks": years[name] for name in BOOTSTRAP_METRICS}
    return report


def compare_statements(
    minuend: Sequence[StatementCall],
    subtrahend: Sequence[StatementCall],
    refs: Mapping[str, Reference],
    bootstrap: Bootstrap = DEFAULT_BOOTSTRAP,
    *,
    year_blocks: bool = False,
) -> dict[str, Any]:
    """Paired bootstrap of two sets of statement calls (minuend minus subtrahend) on the
    statements both called that have a reference label: on all of them, and on the human
    subset alone (None when there is none). year_blocks resamples whole calendar years."""
    a = {c.doc_id: c.stance for c in minuend}
    b = {c.doc_id: c.stance for c in subtrahend}
    years = {c.doc_id: c.meeting_date.year for c in minuend}
    common = sorted(set(a) & set(b) & set(refs))
    if not common:
        raise ValueError("no statement has both predictions and a reference label")

    def compare(ids: Sequence[str]) -> dict[str, Any]:
        gold = [refs[i].label for i in ids]
        groups = [years[i] for i in ids] if year_blocks else None
        diffs = paired_bootstrap(gold, [a[i] for i in ids], [b[i] for i in ids], bootstrap, groups)
        return {"n": len(ids), **diffs}

    human = [i for i in common if refs[i].source == "human"]
    return {"all": compare(common), "human_subset": compare(human) if human else None}
