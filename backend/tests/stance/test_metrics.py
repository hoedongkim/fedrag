"""Tests for the stance metrics, the TDW sentence-benchmark runner and statement evaluation."""

import math
import random
from collections.abc import Sequence
from datetime import date

import numpy as np
import pytest

from app.schemas.common import Stance
from app.stance.metrics import (
    BaselineCall,
    Bootstrap,
    _block_weights,
    _codes,
    _resampled_scores,
    _weighted_scores,
    bootstrap_ci,
    classification_metrics,
    compare_statements,
    evaluate_sentences,
    evaluate_statements,
    gold_agreement,
    majority_class_sentences,
    majority_label,
    mean_sd,
    paired_bootstrap,
    references,
    score_statements,
    summarize,
)
from app.stance.tdw import LabeledSentence, Split
from app.stance.types import SentencePrediction, StatementPrediction

GOLD: list[Stance] = ["dovish", "dovish", "neutral", "neutral", "hawkish", "hawkish"]
PRED: list[Stance] = ["dovish", "neutral", "neutral", "neutral", "hawkish", "dovish"]


def test_metrics_match_hand_computed_values() -> None:
    report = classification_metrics(GOLD, PRED)
    # dovish P 1/2 R 1/2 F1 1/2; neutral P 2/3 R 1 F1 4/5; hawkish P 1 R 1/2 F1 2/3
    assert report["per_class"]["dovish"] == pytest.approx(
        {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 2}
    )
    assert report["per_class"]["neutral"] == pytest.approx(
        {"precision": 2 / 3, "recall": 1.0, "f1": 0.8, "support": 2}
    )
    assert report["per_class"]["hawkish"] == pytest.approx(
        {"precision": 1.0, "recall": 0.5, "f1": 2 / 3, "support": 2}
    )
    assert report["macro_f1"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)
    assert report["accuracy"] == pytest.approx(4 / 6)
    # p_o = 2/3, p_e = (2*2 + 2*3 + 2*1) / 36 = 1/3, kappa = (2/3 - 1/3) / (1 - 1/3)
    assert report["kappa"] == pytest.approx(0.5)
    assert report["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert report["n"] == 6


def test_absent_classes_count_as_zero_and_undefined_kappa_is_none() -> None:
    report = classification_metrics(["neutral", "neutral"], ["neutral", "neutral"])
    assert report["macro_f1"] == pytest.approx(1 / 3)
    assert report["per_class"]["hawkish"]["precision"] == 0.0
    assert report["kappa"] is None


@pytest.mark.parametrize(("gold", "pred"), [(["neutral"], []), ([], [])])
def test_rejects_mismatched_or_empty_labels(gold: list[Stance], pred: list[Stance]) -> None:
    with pytest.raises(ValueError):
        classification_metrics(gold, pred)


def test_mean_sd_uses_sample_sd_and_skips_missing() -> None:
    assert mean_sd([0.5, 0.7, 0.9]) == pytest.approx({"mean": 0.7, "sd": 0.2})
    assert mean_sd([0.4, None]) == {"mean": 0.4, "sd": None}
    assert mean_sd([None]) == {"mean": None, "sd": None}


def test_summarize_averages_seeds_and_sums_confusion_matrices() -> None:
    first = classification_metrics(GOLD, PRED)
    second = classification_metrics(GOLD, GOLD)
    summary = summarize([first, second])
    assert summary["n"] == [6, 6]
    assert summary["macro_f1"]["mean"] == pytest.approx((first["macro_f1"] + 1.0) / 2)
    assert summary["per_class"]["hawkish"]["recall"]["mean"] == pytest.approx(0.75)
    assert summary["per_class"]["hawkish"]["support"] == [2, 2]
    assert summary["confusion_matrix_total"] == [[3, 1, 0], [0, 4, 0], [1, 0, 3]]


class Lookup:
    """A fake sentence classifier that knows every sentence's label and records its calls."""

    name = "keyword"

    def __init__(self, labels: dict[str, Stance]) -> None:
        self.labels = labels
        self.calls: list[list[str]] = []

    def predict_sentences(self, sentences: Sequence[str]) -> list[SentencePrediction]:
        self.calls.append(list(sentences))
        return [SentencePrediction(self.labels[s]) for s in sentences]


def fake_splits() -> dict[tuple[Split, int], list[LabeledSentence]]:
    def rows(items: list[tuple[str, Stance]]) -> list[LabeledSentence]:
        return [LabeledSentence(i, text, 2020, stance) for i, (text, stance) in enumerate(items)]

    train = rows([("shared", "neutral"), ("train only", "hawkish")])
    test = rows([("shared", "neutral"), ("a", "hawkish"), ("b", "dovish"), ("c", "neutral")])
    return {("train", 1): train, ("test", 1): test, ("train", 2): train, ("test", 2): test}


def test_evaluate_sentences_scores_official_and_deduplicated_splits() -> None:
    splits = fake_splits()
    answers: dict[str, Stance] = {
        "shared": "neutral",
        "a": "hawkish",
        "b": "neutral",
        "c": "neutral",
    }
    built: list[int] = []

    def factory(seed: int) -> Lookup:
        built.append(seed)
        return Lookup(answers)

    result = evaluate_sentences(factory, (1, 2), load=lambda split, seed: splits[(split, seed)])
    assert built == [1, 2]
    assert result["classifier"] == "keyword"
    assert result["seeds"] == [1, 2]
    assert result["official"]["n"] == [4, 4]
    assert result["deduplicated"]["n"] == [3, 3]  # "shared" is in the train split
    assert result["official"]["accuracy"] == {"mean": pytest.approx(0.75), "sd": 0.0}
    assert result["deduplicated"]["accuracy"]["mean"] == pytest.approx(2 / 3)
    assert [row["seed"] for row in result["official"]["per_seed"]] == [1, 2]


def test_evaluate_sentences_labels_each_test_split_once() -> None:
    splits = fake_splits()
    classifier = Lookup({"shared": "neutral", "a": "hawkish", "b": "dovish", "c": "neutral"})
    evaluate_sentences(lambda seed: classifier, (1,), load=lambda s, seed: splits[(s, seed)])
    assert classifier.calls == [["shared", "a", "b", "c"]]


def test_evaluate_sentences_rejects_a_short_answer() -> None:
    splits = fake_splits()

    class Short(Lookup):
        def predict_sentences(self, sentences: Sequence[str]) -> list[SentencePrediction]:
            return super().predict_sentences(sentences)[:-1]

    classifier = Short({"shared": "neutral", "a": "hawkish", "b": "dovish", "c": "neutral"})
    with pytest.raises(ValueError, match="3 labels for 4 sentences"):
        evaluate_sentences(lambda seed: classifier, (1,), load=lambda s, seed: splits[(s, seed)])


def test_majority_label_breaks_ties_in_stance_order() -> None:
    assert majority_label(["hawkish", "neutral", "hawkish"]) == "hawkish"
    assert majority_label(["hawkish", "neutral"]) == "neutral"  # tie: dovish, neutral, hawkish
    with pytest.raises(ValueError):
        majority_label([])


def test_majority_class_sentences_predicts_each_seeds_train_majority() -> None:
    splits = fake_splits()
    read: list[tuple[str, int]] = []

    def load(split: Split, seed: int) -> list[LabeledSentence]:
        read.append((split, seed))
        return splits[(split, seed)]

    result = majority_class_sentences((1, 2), load=load)
    assert read[0] == ("train", 1)  # the majority is fixed before any test sentence is read
    assert (result["name"], result["label"]) == ("majority_class", "neutral")  # a 1-1 tie
    assert result["labels"] == {"1": "neutral", "2": "neutral"}
    assert result["official"]["accuracy"] == {"mean": 0.5, "sd": 0.0}  # 2 of 4 are neutral
    assert result["deduplicated"]["accuracy"]["mean"] == pytest.approx(1 / 3)
    assert result["official"]["kappa"]["mean"] == 0.0


GOLD_ROWS = [
    {"doc_id": "statement-a", "label": "hawkish", "confidence": "high", "human_label": None},
    {"doc_id": "statement-b", "label": "neutral", "confidence": "low", "human_label": "dovish"},
    {"doc_id": "statement-c", "label": "dovish", "confidence": "medium", "human_label": "dovish"},
    {"doc_id": "statement-d", "label": "neutral", "confidence": "high"},
]


def test_references_prefer_the_human_label() -> None:
    refs = references(GOLD_ROWS)
    assert (refs["statement-a"].label, refs["statement-a"].source) == ("hawkish", "llm_adjudicated")
    assert refs["statement-a"].confidence == "high"
    assert (refs["statement-b"].label, refs["statement-b"].source) == ("dovish", "human")
    assert refs["statement-b"].confidence is None
    assert refs["statement-d"].source == "llm_adjudicated"


def test_blank_human_labels_count_as_missing() -> None:
    rows = [{"doc_id": "x", "label": "neutral", "confidence": "high", "human_label": "  "}]
    ref = references(rows)["x"]
    assert (ref.label, ref.source, ref.confidence) == ("neutral", "llm_adjudicated", "high")
    assert gold_agreement(rows) is None  # as the API reads the gold file


def test_references_reject_unknown_labels() -> None:
    with pytest.raises(ValueError, match="unknown stance"):
        references([{"doc_id": "x", "label": "very hawkish"}])


def test_gold_agreement_compares_llm_and_human_labels() -> None:
    agreement = gold_agreement(GOLD_ROWS)
    assert agreement is not None
    assert agreement["n"] == 2
    assert agreement["accuracy"] == pytest.approx(0.5)
    assert gold_agreement(GOLD_ROWS[:1]) is None


def prediction(doc_id: str, stance: Stance, score: float) -> StatementPrediction:
    return StatementPrediction(doc_id, date(2020, 1, 1), "finbert", stance, score, abs(score))


def test_evaluate_statements_reports_all_and_human_subsets() -> None:
    predictions = [
        prediction("statement-a", "hawkish", 0.4),
        prediction("statement-b", "neutral", -0.1),
        prediction("statement-c", "dovish", -0.5),
        prediction("statement-d", "neutral", 0.0),
        prediction("statement-z", "neutral", 0.0),
    ]
    result = evaluate_statements(predictions, references(GOLD_ROWS))
    assert result["classifier"] == "finbert"
    assert result["all"]["n"] == 4
    assert result["all"]["accuracy"] == pytest.approx(0.75)
    # score ranks (4, 2, 1, 3) against reference ranks (4, 1.5, 1.5, 3): r = 4.5 / sqrt(5 * 4.5)
    assert result["all"]["spearman"] == pytest.approx(4.5 / math.sqrt(22.5))
    assert result["human_subset"]["n"] == 2
    assert result["without_reference"] == ["statement-z"]


def test_spearman_is_none_for_constant_scores() -> None:
    predictions = [
        prediction("statement-a", "neutral", 0.0),
        prediction("statement-c", "neutral", 0.0),
    ]
    assert evaluate_statements(predictions, references(GOLD_ROWS))["all"]["spearman"] is None


# --- Bootstrap -------------------------------------------------------------------------------

SMALL = Bootstrap(resamples=2_000, seed=7)


def test_resampled_scores_match_classification_metrics() -> None:
    rng = random.Random(3)
    stances: list[Stance] = ["dovish", "neutral", "hawkish"]
    gold = [rng.choice(stances) for _ in range(40)]
    pred = [rng.choice(stances) for _ in range(40)]
    draws = np.array(
        [
            list(range(40)),
            [rng.randrange(40) for _ in range(40)],
            [i for i in range(40) if gold[i] != "dovish"][:10] * 4,  # a class with no gold item
            [0] * 40,  # a single item repeated
        ]
    )
    scores = _resampled_scores(_codes(gold), _codes(pred), draws)
    for row, draw in enumerate(draws):
        expected = classification_metrics([gold[i] for i in draw], [pred[i] for i in draw])
        assert scores["accuracy"][row] == pytest.approx(expected["accuracy"])
        assert scores["macro_f1"][row] == pytest.approx(expected["macro_f1"])


def test_bootstrap_ci_brackets_the_estimate_and_is_reproducible() -> None:
    gold, pred = GOLD * 10, PRED * 10
    report = classification_metrics(gold, pred)
    ci = bootstrap_ci(gold, pred, SMALL)
    for name in ("accuracy", "macro_f1"):
        assert ci[name]["low"] < report[name] < ci[name]["high"]
    assert bootstrap_ci(gold, pred, SMALL) == ci
    assert bootstrap_ci(gold, pred, Bootstrap(resamples=2_000, seed=8)) != ci
    narrower = bootstrap_ci(gold, pred, Bootstrap(resamples=2_000, seed=7, level=0.5))
    assert ci["accuracy"]["low"] <= narrower["accuracy"]["low"]
    assert narrower["accuracy"]["high"] <= ci["accuracy"]["high"]


def test_perfect_predictions_have_a_degenerate_interval() -> None:
    ci = bootstrap_ci(GOLD * 5, GOLD * 5, SMALL)
    assert ci["accuracy"] == {"low": 1.0, "high": 1.0}
    assert ci["macro_f1"]["high"] == 1.0  # below 1 only when a resample loses a class


@pytest.mark.parametrize(
    ("gold", "pred", "config"),
    [
        (GOLD, PRED[:-1], SMALL),
        ([], [], SMALL),
        (GOLD, PRED, Bootstrap(resamples=0)),
        (GOLD, PRED, Bootstrap(level=1.0)),
    ],
    ids=["mismatched", "empty", "no-resamples", "level-1"],
)
def test_bootstrap_rejects_bad_input(
    gold: list[Stance], pred: list[Stance], config: Bootstrap
) -> None:
    with pytest.raises(ValueError):
        bootstrap_ci(gold, pred, config)


def test_paired_bootstrap_differences() -> None:
    gold, pred = GOLD * 10, PRED * 10
    same = paired_bootstrap(gold, pred, pred, SMALL)
    assert same["accuracy"] == {"difference": 0.0, "ci": {"low": 0.0, "high": 0.0}}
    better = paired_bootstrap(gold, gold, pred, SMALL)
    report = classification_metrics(gold, pred)
    assert better["accuracy"]["difference"] == pytest.approx(1 - report["accuracy"])
    assert better["macro_f1"]["difference"] == pytest.approx(1 - report["macro_f1"])
    assert 0 < better["accuracy"]["ci"]["low"] < better["accuracy"]["ci"]["high"]
    worse = paired_bootstrap(gold, pred, gold, SMALL)
    assert worse["accuracy"]["ci"] == {
        "low": pytest.approx(-better["accuracy"]["ci"]["high"]),
        "high": pytest.approx(-better["accuracy"]["ci"]["low"]),
    }
    with pytest.raises(ValueError):
        paired_bootstrap(gold, pred, pred[:-1], SMALL)


def test_evaluate_statements_adds_intervals_independent_of_order() -> None:
    predictions = [
        prediction("statement-a", "hawkish", 0.4),
        prediction("statement-b", "neutral", -0.1),
        prediction("statement-c", "dovish", -0.5),
        prediction("statement-d", "neutral", 0.0),
    ]
    refs = references(GOLD_ROWS)
    result = evaluate_statements(predictions, refs, SMALL)
    for subset in ("all", "human_subset"):
        report = result[subset]
        for name in ("accuracy", "macro_f1"):
            assert report[f"{name}_ci"]["low"] <= report[name] <= report[f"{name}_ci"]["high"]
    assert evaluate_statements(predictions[::-1], refs, SMALL) == result
    assert "accuracy_ci" not in evaluate_statements(predictions, refs)["all"]


def test_weighted_scores_match_classification_metrics_on_repeated_items() -> None:
    rng = random.Random(5)
    stances: list[Stance] = ["dovish", "neutral", "hawkish"]
    gold = [rng.choice(stances) for _ in range(12)]
    pred = [rng.choice(stances) for _ in range(12)]
    weights = np.array([[1] * 12, [rng.randrange(4) for _ in range(12)], [0] * 11 + [3]])
    scores = _weighted_scores(_codes(gold), _codes(pred), weights)
    for row, counts in enumerate(weights):
        items = [i for i, count in enumerate(counts) for _ in range(count)]
        expected = classification_metrics([gold[i] for i in items], [pred[i] for i in items])
        assert scores["accuracy"][row] == pytest.approx(expected["accuracy"])
        assert scores["macro_f1"][row] == pytest.approx(expected["macro_f1"])


def test_block_bootstrap_draws_whole_groups() -> None:
    groups = [2015, 2015, 2016, 2017, 2017, 2017]
    weights = _block_weights(groups, SMALL)
    assert weights.shape == (SMALL.resamples, len(groups))
    assert (weights[:, 0] == weights[:, 1]).all()  # the two 2015 items always travel together
    assert (weights[:, 3] == weights[:, 5]).all()
    # As many group draws as there are groups: 2015 and 2017 count 2 and 3 items per draw.
    assert set((weights[:, 0] + weights[:, 2] + weights[:, 3]).tolist()) == {3}
    gold, pred = GOLD * 3, PRED * 3
    years = [2015 + i // 3 for i in range(len(gold))]
    ci = bootstrap_ci(gold, pred, SMALL, years)
    assert ci == bootstrap_ci(gold, pred, SMALL, years)  # reproducible from the seed
    assert ci != bootstrap_ci(gold, pred, SMALL)  # not the single-statement resampling
    report = classification_metrics(gold, pred)
    assert ci["accuracy"]["low"] <= report["accuracy"] <= ci["accuracy"]["high"]
    one_year = bootstrap_ci(gold, pred, SMALL, [2020] * len(gold))
    assert one_year["accuracy"] == {"low": report["accuracy"], "high": report["accuracy"]}
    same = paired_bootstrap(gold, pred, pred, SMALL, years)
    assert same["accuracy"] == {"difference": 0.0, "ci": {"low": 0.0, "high": 0.0}}
    with pytest.raises(ValueError):
        bootstrap_ci(gold, pred, SMALL, years[:-1])


def test_score_statements_takes_baseline_calls_and_adds_year_blocks() -> None:
    refs = references(GOLD_ROWS)
    calls = [
        BaselineCall("statement-a", date(2019, 1, 30), "hawkish", 1.0),
        BaselineCall("statement-b", date(2019, 3, 20), "hawkish", 1.0),
        BaselineCall("statement-c", date(2020, 1, 29), "hawkish", 1.0),
        BaselineCall("statement-d", date(2021, 1, 27), "hawkish", 1.0),
        BaselineCall("statement-z", date(2021, 3, 17), "hawkish", 1.0),  # no reference
    ]
    result = score_statements(calls, refs, SMALL)
    assert result["all"]["n"] == 4 and result["all"]["accuracy"] == 0.25
    assert result["all"]["spearman"] is None  # a constant call
    assert result["human_subset"]["n"] == 2
    for name in ("accuracy", "macro_f1"):
        interval = result["all"][f"{name}_ci_year_blocks"]
        assert interval["low"] <= result["all"][name] <= interval["high"]
    years = compare_statements(calls, calls[:1], refs, SMALL, year_blocks=True)
    assert years["all"]["n"] == 1


def test_compare_statements_pairs_by_doc_id() -> None:
    refs = references(GOLD_ROWS)
    right = [
        prediction("statement-d", "neutral", 0.0),
        prediction("statement-a", "hawkish", 0.4),
        prediction("statement-b", "dovish", -0.3),
        prediction("statement-c", "dovish", -0.5),
    ]
    wrong = [
        prediction("statement-a", "dovish", -0.4),
        prediction("statement-b", "dovish", -0.3),
        prediction("statement-c", "dovish", -0.5),
        prediction("statement-z", "neutral", 0.0),  # no reference: ignored
    ]
    result = compare_statements(right, wrong, refs, SMALL)
    assert result["all"]["n"] == 3  # statement-d has no prediction in `wrong`
    assert result["all"]["accuracy"]["difference"] == pytest.approx(1 / 3)
    assert result["human_subset"]["n"] == 2
    assert result["human_subset"]["accuracy"]["difference"] == 0.0
    with pytest.raises(ValueError, match="no statement"):
        compare_statements(right[:1], wrong[:1], refs, SMALL)
