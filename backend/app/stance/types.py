"""Types shared by the stance classifiers, their evaluation and the stance API."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Protocol, get_args

from app.schemas.common import Stance, StanceClassifier

STANCES: tuple[Stance, ...] = ("dovish", "neutral", "hawkish")
# A stance as a number: the sign of a statement score, and the scale rank correlation uses.
ORDINAL: dict[Stance, int] = {"dovish": -1, "neutral": 0, "hawkish": 1}
CLASSIFIERS: tuple[StanceClassifier, ...] = get_args(StanceClassifier)
# Under Settings.data_dir: the reference labels and the results file, read by the CLI and the API.
GOLD_FILE = Path("eval") / "stance_gold.jsonl"
BENCHMARK_FILE = Path("eval") / "stance_benchmark.json"


@dataclass(frozen=True)
class SentencePrediction:
    """One sentence's stance. probs maps each stance to a score summing to 1 when the classifier
    produces a distribution; it is empty for classifiers that only emit a label."""

    stance: Stance
    probs: dict[Stance, float] = field(default_factory=dict)


@dataclass(frozen=True)
class StatementPrediction:
    """One FOMC statement's stance. score is in [-1, 1] (+1 most hawkish, -1 most dovish);
    confidence is model-reported or derived, never a calibrated probability."""

    doc_id: str
    meeting_date: date
    classifier: StanceClassifier
    stance: Stance
    score: float
    confidence: float | None
    key_factors: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()  # sentences quoted verbatim from the statement
    model_version: str = ""


class SentenceClassifier(Protocol):
    """Labels sentences; used for the sentence-level benchmark and, aggregated, for statements."""

    name: StanceClassifier

    def predict_sentences(self, sentences: Sequence[str]) -> list[SentencePrediction]: ...
