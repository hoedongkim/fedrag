"""FinBERT (ProsusAI/finbert) fine-tuned on a TDW train split: one model per seed.

Each seed's train split is cut into a fine-tuning part and a 10% validation part (stratified by
label, grouped by sentence text so a duplicated sentence never sits on both sides); the test split
is never touched. The classification head is re-initialized for dovish / neutral / hawkish (the
checkpoint's head predicts sentiment). Training is a plain PyTorch loop: AdamW, linear warmup and
decay, batch 16, max length 128, at most 4 epochs, keeping the epoch with the best validation
macro-F1 and stopping after the first epoch that does not improve it. Python, NumPy and torch are
seeded, but MPS kernels are not bit-for-bit deterministic, so a rerun can differ slightly.

Models are saved to data/models/finbert-tdw-<seed>/ (git-ignored: derived from FinBERT, CC BY-NC-SA
4.0) with training.json, written last, recording the run. Statement-level use averages the class
probabilities of the three seed models.
"""

import json
import math
import random
import shutil
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel

from app.core.config import Settings, get_settings
from app.schemas.common import Stance, StanceClassifier
from app.stance import tdw
from app.stance.metrics import classification_metrics
from app.stance.types import STANCES, SentencePrediction

BASE_MODEL = "ProsusAI/finbert"
BASE_REVISION = "0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031"  # commit of BASE_MODEL on the Hub
MAX_LENGTH = 128
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.1
BATCH_SIZE = 16
MAX_EPOCHS = 4
VALIDATION_FRACTION = 0.1
PREDICT_BATCH_SIZE = 64
TRAINING_FILE = "training.json"
LABEL2ID: dict[Stance, int] = {stance: i for i, stance in enumerate(STANCES)}

Log = Callable[[str], None]


@dataclass(frozen=True)
class TrainingReport:
    """One fine-tuning run, saved as training.json next to the model."""

    seed: int
    base_model: str
    base_revision: str
    device: str
    n_train: int
    n_validation: int
    epochs_run: int
    best_epoch: int  # the epoch whose weights were saved
    val_macro_f1: list[float]  # per epoch
    val_loss: list[float]
    train_loss: list[float]
    train_seconds: float  # wall clock of all epochs, validation and checkpoints included
    hyperparameters: dict[str, float | int]


def model_dir(seed: int, settings: Settings | None = None) -> Path:
    return (settings or get_settings()).data_dir / "models" / f"finbert-tdw-{seed}"


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def validation_split(
    sentences: Sequence[tdw.LabeledSentence], fraction: float, seed: int
) -> tuple[list[tdw.LabeledSentence], list[tdw.LabeledSentence]]:
    """(fine-tuning, validation) parts of a train split: stratified by label and grouped by text,
    so every copy of a duplicated sentence lands on the same side."""
    by_text: dict[str, list[tdw.LabeledSentence]] = {}
    for s in sentences:
        by_text.setdefault(s.sentence, []).append(s)
    texts = sorted(by_text)
    labels = [Counter(s.stance for s in by_text[t]).most_common(1)[0][0] for t in texts]
    fit_texts, held_texts = train_test_split(
        texts, test_size=fraction, stratify=labels, random_state=seed % 2**32
    )
    held = set(held_texts)
    return (
        [s for s in sentences if s.sentence not in held],
        [s for s in sentences if s.sentence in held],
    )


def _batches(n: int, size: int, generator: torch.Generator | None = None) -> list[list[int]]:
    order = torch.randperm(n, generator=generator).tolist() if generator else list(range(n))
    return [order[i : i + size] for i in range(0, n, size)]


def _encode(tokenizer: Any, texts: Sequence[str], device: torch.device) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        list(texts), padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
    )
    return {name: tensor.to(device) for name, tensor in encoded.items()}


@torch.no_grad()
def _probabilities(
    model: PreTrainedModel,
    tokenizer: Any,
    texts: Sequence[str],
    device: torch.device,
    batch_size: int = PREDICT_BATCH_SIZE,
) -> np.ndarray:
    """Softmax class probabilities, columns in STANCES order."""
    model.eval()
    out = np.zeros((len(texts), len(STANCES)), dtype=np.float64)
    for batch in _batches(len(texts), batch_size):
        logits = model(**_encode(tokenizer, [texts[i] for i in batch], device)).logits
        out[batch] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    return out


def _evaluate(
    model: PreTrainedModel,
    tokenizer: Any,
    sentences: Sequence[tdw.LabeledSentence],
    device: torch.device,
) -> tuple[float, float]:
    """(macro-F1, mean cross-entropy) on labelled sentences."""
    probs = _probabilities(model, tokenizer, [s.sentence for s in sentences], device)
    gold = [LABEL2ID[s.stance] for s in sentences]
    loss = -float(np.mean(np.log(np.clip(probs[np.arange(len(gold)), gold], 1e-12, None))))
    pred = [STANCES[i] for i in probs.argmax(axis=1)]
    return classification_metrics([s.stance for s in sentences], pred)["macro_f1"], loss


def _load_base(base: str | Path, revision: str | None) -> tuple[PreTrainedModel, Any]:
    tokenizer = AutoTokenizer.from_pretrained(base, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        base,
        revision=revision,
        num_labels=len(STANCES),
        id2label=dict(enumerate(STANCES)),
        label2id=LABEL2ID,
        use_safetensors=True,
    )
    head: torch.nn.Linear = model.classifier
    torch.nn.init.normal_(head.weight, std=model.config.initializer_range)
    torch.nn.init.zeros_(head.bias)
    return model, tokenizer


def _optimizer(model: PreTrainedModel, lr: float) -> torch.optim.AdamW:
    """AdamW with weight decay on every weight except biases and LayerNorm parameters."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        (no_decay if name.endswith("bias") or "LayerNorm" in name else decay).append(param)
    groups = [
        {"params": decay, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr)


def fine_tune(
    train: Sequence[tdw.LabeledSentence],
    seed: int,
    out_dir: Path,
    *,
    base: str | Path = BASE_MODEL,
    revision: str | None = BASE_REVISION,
    device: torch.device | None = None,
    max_epochs: int = MAX_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    log: Log = print,
) -> TrainingReport:
    """Fine-tune `base` on a train split and save the best epoch's model to out_dir (replacing
    it only once training has finished)."""
    device = device or pick_device()
    seed_everything(seed)
    fit, validation = validation_split(train, VALIDATION_FRACTION, seed)
    model, tokenizer = _load_base(base, revision)
    model.to(device)
    optimizer = _optimizer(model, lr)
    steps = max_epochs * math.ceil(len(fit) / batch_size)
    warmup = max(1, round(WARMUP_FRACTION * steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min((step + 1) / warmup, max(0.0, (steps - step) / max(1, steps - warmup))),
    )
    generator = torch.Generator().manual_seed(seed)
    labels = torch.tensor([LABEL2ID[s.stance] for s in fit])
    partial = out_dir.with_name(out_dir.name + ".part")
    shutil.rmtree(partial, ignore_errors=True)
    history: dict[str, list[float]] = {"val_macro_f1": [], "val_loss": [], "train_loss": []}
    best_epoch, started = 0, time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for batch in _batches(len(fit), batch_size, generator):
            inputs = _encode(tokenizer, [fit[i].sentence for i in batch], device)
            loss = model(**inputs, labels=labels[batch].to(device)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item())
        f1, val_loss = _evaluate(model, tokenizer, validation, device)
        history["train_loss"].append(float(np.mean(losses)))
        history["val_macro_f1"].append(f1)
        history["val_loss"].append(val_loss)
        log(
            f"seed {seed} epoch {epoch}: train loss {history['train_loss'][-1]:.4f}, "
            f"val loss {val_loss:.4f}, val macro-F1 {f1:.4f} "
            f"({time.perf_counter() - started:.0f} s)"
        )
        if best_epoch and f1 <= history["val_macro_f1"][best_epoch - 1]:
            break  # early stopping: this epoch did not beat the best one
        best_epoch = epoch
        model.save_pretrained(partial)
        tokenizer.save_pretrained(partial)
    report = TrainingReport(
        seed=seed,
        base_model=str(base),
        base_revision=revision or "",
        device=device.type,
        n_train=len(fit),
        n_validation=len(validation),
        epochs_run=len(history["train_loss"]),
        best_epoch=best_epoch,
        train_seconds=round(time.perf_counter() - started, 1),
        hyperparameters={
            "learning_rate": lr,
            "weight_decay": WEIGHT_DECAY,
            "warmup_fraction": WARMUP_FRACTION,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "max_length": MAX_LENGTH,
            "validation_fraction": VALIDATION_FRACTION,
        },
        **{name: [round(v, 6) for v in values] for name, values in history.items()},
    )
    (partial / TRAINING_FILE).write_text(json.dumps(asdict(report), indent=2) + "\n")
    shutil.rmtree(out_dir, ignore_errors=True)
    partial.replace(out_dir)
    return report


def read_report(path: Path) -> TrainingReport | None:
    """The training.json of a saved model directory; None when the model is missing."""
    file = path / TRAINING_FILE
    return TrainingReport(**json.loads(file.read_text())) if file.is_file() else None


def train_seeds(
    seeds: Sequence[int] = tdw.SEEDS,
    settings: Settings | None = None,
    *,
    force: bool = False,
    log: Log = print,
) -> list[TrainingReport]:
    """Fine-tune one model per seed on its TDW train split, skipping seeds already trained."""
    reports = []
    for seed in seeds:
        target = model_dir(seed, settings)
        existing = None if force else read_report(target)
        if existing is None:
            existing = fine_tune(tdw.load("train", seed, settings), seed, target, log=log)
        else:
            log(f"seed {seed}: using the model in {target}")
        reports.append(existing)
    return reports


class FinbertClassifier:
    """Fine-tuned FinBERT models; with several (one per seed) their probabilities are averaged."""

    name: StanceClassifier = "finbert"

    def __init__(self, model_dirs: Path | Sequence[Path], device: torch.device | None = None):
        dirs = [model_dirs] if isinstance(model_dirs, Path) else list(model_dirs)
        missing = [str(d) for d in dirs if read_report(d) is None]
        if not dirs or missing:
            raise FileNotFoundError(f"no fine-tuned FinBERT in {missing or dirs}")
        self.device = device or pick_device()
        self.model_dirs = dirs
        self.reports = [read_report(d) for d in dirs]
        self._models = []
        for d in dirs:
            model = AutoModelForSequenceClassification.from_pretrained(d, use_safetensors=True)
            if [model.config.id2label[i] for i in range(len(STANCES))] != list(STANCES):
                raise ValueError(f"{d}: labels {model.config.id2label} are not {STANCES}")
            self._models.append((model.to(self.device), AutoTokenizer.from_pretrained(d)))

    @property
    def version(self) -> str:
        seeds = "/".join(str(report.seed) for report in self.reports if report)
        return f"{BASE_MODEL}@{BASE_REVISION[:7]} + TDW fine-tune (seeds {seeds}; mean probs)"

    def probabilities(self, sentences: Sequence[str]) -> np.ndarray:
        """Mean class probabilities of the models, columns in STANCES order."""
        if not sentences:
            return np.zeros((0, len(STANCES)))
        runs = [_probabilities(m, t, sentences, self.device) for m, t in self._models]
        return np.mean(runs, axis=0)

    def predict_sentences(self, sentences: Sequence[str]) -> list[SentencePrediction]:
        return [
            SentencePrediction(
                STANCES[int(row.argmax())], {s: float(p) for s, p in zip(STANCES, row, strict=True)}
            )
            for row in self.probabilities(sentences)
        ]


def finbert_factory(settings: Settings | None = None) -> Callable[[int], FinbertClassifier]:
    """seed -> the FinBERT model fine-tuned on that seed's train split (train_seeds first)."""
    return lambda seed: FinbertClassifier(model_dir(seed, settings))


def ensemble(
    seeds: Sequence[int] = tdw.SEEDS, settings: Settings | None = None
) -> FinbertClassifier:
    """The seed models together, for statements."""
    return FinbertClassifier([model_dir(seed, settings) for seed in seeds])
