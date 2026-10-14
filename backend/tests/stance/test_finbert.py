"""Tests for FinBERT fine-tuning and prediction on a tiny random BERT (nothing is downloaded)."""

import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

from app.core.config import Settings
from app.schemas.common import Stance
from app.stance import finbert, tdw
from app.stance.finbert import FinbertClassifier, fine_tune, read_report, validation_split

CPU = torch.device("cpu")
WORDS = ["rates", "rise", "fall", "hold", "steady", "inflation", "growth", "jobs", "."]


def tiny_bert(path: Path, seed: int = 0, labels: tuple[str, ...] = ("a", "b", "c")) -> Path:
    """Save a 1-layer, 16-wide BERT and its tokenizer to `path`."""
    vocab = {w: i for i, w in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS])}
    torch.manual_seed(seed)
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=160,
        num_labels=3,
        id2label=dict(enumerate(labels)),
        label2id={label: i for i, label in enumerate(labels)},
    )
    BertForSequenceClassification(config).save_pretrained(path)
    BertTokenizer(vocab=vocab).save_pretrained(path)
    return path


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tiny_bert(tmp_path_factory.mktemp("tiny") / "base")


def toy_train(copies: int = 1) -> list[tdw.LabeledSentence]:
    rows: list[tdw.LabeledSentence] = []
    templates: list[tuple[str, Stance]] = [
        ("rates rise", "hawkish"),
        ("rates fall", "dovish"),
        ("rates hold steady", "neutral"),
    ]
    for i in range(20):
        for text, stance in templates:
            sentence = f"{text} {WORDS[5 + i % 3]} " + "." * (i // 3)
            rows += [tdw.LabeledSentence(len(rows), sentence, 2020, stance)] * copies
    return rows


def test_validation_split_is_stratified_and_grouped_by_text() -> None:
    train = toy_train(copies=2)  # every sentence twice
    fit, held = validation_split(train, 0.1, seed=7)
    assert len(fit) + len(held) == len(train)
    assert not {s.sentence for s in fit} & {s.sentence for s in held}
    assert Counter(s.stance for s in held) == {"hawkish": 4, "dovish": 4, "neutral": 4}
    assert validation_split(train, 0.1, seed=7) == (fit, held)


def test_fine_tune_saves_the_model_and_its_report(base: Path, tmp_path: Path) -> None:
    out = tmp_path / "finbert-tdw-1"
    report = fine_tune(
        toy_train(), 1, out, base=base, revision=None, device=CPU, max_epochs=2, log=lambda _: None
    )
    assert (report.seed, report.device, report.n_train, report.n_validation) == (1, "cpu", 54, 6)
    assert 1 <= report.best_epoch <= report.epochs_run <= 2
    assert len(report.val_macro_f1) == len(report.train_loss) == report.epochs_run
    assert report.hyperparameters["batch_size"] == finbert.BATCH_SIZE
    assert report.train_seconds >= 0
    assert {"config.json", "model.safetensors", "training.json"} <= {p.name for p in out.iterdir()}
    assert not out.with_name("finbert-tdw-1.part").exists()
    assert read_report(out) == report
    config = json.loads((out / "config.json").read_text())
    assert config["id2label"] == {"0": "dovish", "1": "neutral", "2": "hawkish"}


def test_fine_tune_stops_at_the_first_epoch_without_improvement(
    base: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scores: Iterator[float] = iter([0.5, 0.6, 0.6, 0.9])
    monkeypatch.setattr(finbert, "_evaluate", lambda *args: (next(scores), 1.0))
    report = fine_tune(
        toy_train(), 1, tmp_path / "m", base=base, revision=None, device=CPU, log=lambda _: None
    )
    assert report.val_macro_f1 == [0.5, 0.6, 0.6]
    assert (report.epochs_run, report.best_epoch) == (3, 2)


def test_classifier_returns_probabilities_and_averages_an_ensemble(
    base: Path, tmp_path: Path
) -> None:
    dirs = []
    for seed in (1, 2):
        dirs.append(tmp_path / f"finbert-tdw-{seed}")
        fine_tune(
            toy_train(),
            seed,
            dirs[-1],
            base=base,
            revision=None,
            device=CPU,
            max_epochs=1,
            log=lambda _: None,
        )
    sentences = ["rates rise", "rates fall growth", "jobs"]
    singles = [FinbertClassifier(d, device=CPU).probabilities(sentences) for d in dirs]
    ensemble = FinbertClassifier(dirs, device=CPU)
    assert np.allclose(ensemble.probabilities(sentences), np.mean(singles, axis=0))
    predictions = ensemble.predict_sentences(sentences)
    assert ensemble.name == "finbert"
    assert ensemble.version.endswith("+ TDW fine-tune (seeds 1/2; mean probs)")
    for prediction in predictions:
        assert set(prediction.probs) == {"dovish", "neutral", "hawkish"}
        assert sum(prediction.probs.values()) == pytest.approx(1.0)
        assert prediction.stance == max(prediction.probs, key=prediction.probs.__getitem__)
    assert ensemble.predict_sentences([]) == []


def test_classifier_requires_a_finished_model(base: Path, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        FinbertClassifier(tmp_path / "missing", device=CPU)
    with pytest.raises(FileNotFoundError):  # weights but no training.json: training never finished
        FinbertClassifier(base, device=CPU)


def test_classifier_rejects_models_with_other_labels(base: Path, tmp_path: Path) -> None:
    out = tmp_path / "finbert-tdw-1"
    fine_tune(
        toy_train(), 1, out, base=base, revision=None, device=CPU, max_epochs=1, log=lambda _: None
    )
    config = json.loads((out / "config.json").read_text())
    config["id2label"] = {"0": "positive", "1": "negative", "2": "neutral"}
    (out / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="are not"):
        FinbertClassifier(out, device=CPU)


def test_train_seeds_skips_seeds_already_trained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path)
    trained: list[int] = []

    def fake_fine_tune(train: object, seed: int, out_dir: Path, **kwargs: object) -> object:
        trained.append(seed)
        report = finbert.TrainingReport(
            seed, "base", "rev", "cpu", 1, 1, 1, 1, [0.5], [1.0], [1.0], 0.1, {}
        )
        out_dir.mkdir(parents=True)
        (out_dir / finbert.TRAINING_FILE).write_text(json.dumps(report.__dict__))
        return report

    monkeypatch.setattr(finbert, "fine_tune", fake_fine_tune)
    monkeypatch.setattr(tdw, "load", lambda split, seed, settings=None: [])
    first = finbert.train_seeds((1, 2), settings, log=lambda _: None)
    second = finbert.train_seeds((1, 2), settings, log=lambda _: None)
    assert trained == [1, 2]
    assert first == second
    assert finbert.model_dir(1, settings) == tmp_path / "models" / "finbert-tdw-1"
