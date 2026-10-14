"""Stance files for routes: the reference labels (data/eval/stance_gold.jsonl) and the benchmark
(data/eval/stance_benchmark.json), both read-only and re-read whenever they change on disk."""

import json
import logging
import statistics
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeVar

from fastapi import status
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from app.core.config import Settings
from app.schemas.common import ErrorResponse, Stance
from app.schemas.stance import ReferenceLabel, ReferenceLabelSummary, StanceBenchmark
from app.stance.types import BENCHMARK_FILE, GOLD_FILE, STANCES

logger = logging.getLogger(__name__)

STANCE_NOT_LOADED = "Stance results not loaded: run make stance."
BENCHMARK_MISSING = "Stance benchmark not computed yet: run make stance."
BENCHMARK_INVALID = "Stance benchmark file data/eval/stance_benchmark.json is invalid: {problems}"

STANCE_RESPONSES: dict[int | str, dict[str, Any]] = {
    status.HTTP_503_SERVICE_UNAVAILABLE: {
        "model": ErrorResponse,
        "description": (
            "Database not configured (DATABASE_URL is not set) or not reachable, or no stance "
            "results stored for the classifier yet (run make stance)."
        ),
    }
}
BENCHMARK_RESPONSES: dict[int | str, dict[str, Any]] = {
    status.HTTP_503_SERVICE_UNAVAILABLE: {
        "model": ErrorResponse,
        "description": "The benchmark file has not been computed yet (run make stance).",
    }
}

T = TypeVar("T")


class BenchmarkUnavailableError(RuntimeError):
    """The benchmark file is missing or does not match the StanceBenchmark contract."""


class _GoldRow(BaseModel):
    """The fields of a gold-file line the API uses; the rest (annotators, notes) are ignored."""

    model_config = ConfigDict(extra="ignore")

    doc_id: str
    label: Stance
    confidence: float | str | None = None
    human_label: Stance | None = None

    @field_validator("confidence", "human_label", mode="before")
    @classmethod
    def _blank_is_none(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value


class _FileCache:
    """Parsed file contents keyed on the file's mtime and size, so an edit is picked up at once."""

    def __init__(self) -> None:
        self._entries: dict[tuple[Path, str], tuple[tuple[int, int], Any]] = {}
        self._lock = threading.Lock()

    def get(self, path: Path, kind: str, parse: Callable[[Path], T]) -> T | None:
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None
        version = (stat.st_mtime_ns, stat.st_size)
        with self._lock:
            entry = self._entries.get((path, kind))
        if entry and entry[0] == version:
            return entry[1]
        value = parse(path)
        with self._lock:
            self._entries[(path, kind)] = (version, value)
        return value


_cache = _FileCache()


def _parse_gold(path: Path) -> dict[str, ReferenceLabel]:
    labels: dict[str, ReferenceLabel] = {}
    skipped = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = _GoldRow.model_validate_json(line)
        except ValidationError:
            skipped += 1
            continue
        human = row.human_label is not None
        labels[row.doc_id] = ReferenceLabel(
            label=row.human_label or row.label,
            source="human" if human else "llm_adjudicated",
            confidence=None if human else row.confidence,
        )
    if skipped:
        logger.warning("%s: skipped %d line(s) that are not valid reference labels", path, skipped)
    return labels


def reference_labels(settings: Settings) -> dict[str, ReferenceLabel]:
    """doc_id -> the statement's reference label: the human label when a human reviewed it,
    otherwise the LLM-adjudicated one. Empty while the gold file does not exist."""
    return _cache.get(settings.data_dir / GOLD_FILE, "gold", _parse_gold) or {}


def reference_summary(labels: Mapping[str, ReferenceLabel]) -> ReferenceLabelSummary | None:
    if not labels:
        return None
    human = sum(label.source == "human" for label in labels.values())
    return ReferenceLabelSummary(n_labels=len(labels), n_human_reviewed=human)


def load_benchmark(settings: Settings) -> StanceBenchmark:
    """The benchmark file with the reference labels as they stand now; raises
    BenchmarkUnavailableError when it is missing or invalid."""
    benchmark = _cache.get(settings.data_dir / BENCHMARK_FILE, "benchmark", _parse_benchmark)
    if benchmark is None:
        raise BenchmarkUnavailableError(BENCHMARK_MISSING)
    if isinstance(benchmark, str):
        raise BenchmarkUnavailableError(benchmark)
    summary = reference_summary(reference_labels(settings))
    return benchmark.model_copy(update={"reference_labels": summary})


def benchmark_from_json(raw: Any) -> StanceBenchmark:
    """The API view of the benchmark file's contents; raises ValidationError, or ValueError and
    friends, when they do not fit the contract (the stance CLI checks a file before writing it)."""
    if not isinstance(raw, dict):
        raise ValueError("the top level is not a JSON object")
    raw = raw | {
        "sentence_level": [
            row for entry in raw.get("sentence_level", []) for row in _sentence_rows(entry)
        ],
        "statement_level": [_statement_row(entry) for entry in raw.get("statement_level", [])],
        "sentence_baselines": [
            row
            for entry in raw.get("sentence_baselines", [])
            for row in _sentence_rows(entry, ("name", "label"))
        ],
        "statement_baselines": [
            _baseline_row(entry) for entry in raw.get("statement_baselines", [])
        ],
    }
    return StanceBenchmark.model_validate(raw)


def _parse_benchmark(path: Path) -> StanceBenchmark | str:
    """The validated benchmark, or the error message to report (cached too, so a bad file is
    parsed once per version)."""
    try:
        return benchmark_from_json(json.loads(path.read_text(encoding="utf-8")))
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(map(str, err['loc'])) or 'file'}: {err['msg']}"
            for err in exc.errors(include_url=False)[:5]
        )
        return BENCHMARK_INVALID.format(problems=problems)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        return BENCHMARK_INVALID.format(problems=exc)


# The file holds API rows ({classifier, split, macro_f1_mean, ...}). Entries in the shape of the
# app.stance.metrics reports (evaluate_sentences: {classifier, seeds, official: {...},
# deduplicated: {...}}; evaluate_statements: {classifier, variant, all: {...}, human_subset}) are
# accepted too and flattened into those rows; an entry without a variant is pre-registered.
# Baselines are flattened the same way, named by {name, label} instead of a classifier.

_SCORE_KEYS = ("n", "accuracy", "macro_f1", "kappa", "accuracy_ci", "macro_f1_ci")


def _sentence_rows(entry: Any, identity: tuple[str, ...] = ("classifier",)) -> list[Any]:
    if not isinstance(entry, dict) or "split" in entry:
        return [entry]
    rows = []
    for split in ("official", "deduplicated"):
        report = entry.get(split)
        if not isinstance(report, dict):
            continue
        n = report.get("n") or []
        rows.append(
            {
                **{key: entry.get(key) for key in identity},
                "split": split,
                "macro_f1_mean": report["macro_f1"]["mean"],
                "macro_f1_sd": report["macro_f1"]["sd"],
                "accuracy_mean": report["accuracy"]["mean"],
                "accuracy_sd": report["accuracy"]["sd"],
                "per_class_f1_mean": (  # baselines have no per-class F1 in the API
                    {stance: per_class[stance]["f1"]["mean"] for stance in STANCES}
                    if isinstance(per_class := report.get("per_class"), dict)
                    else None
                ),
                "n_seeds": len(entry.get("seeds") or n),
                "n_test_mean": statistics.fmean(n) if n else 0,
            }
        )
    return rows


def _statement_row(entry: Any) -> Any:
    if not isinstance(entry, dict) or not isinstance(entry.get("all"), dict):
        return entry
    scores, human = entry["all"], entry.get("human_subset")
    if "reference" in entry:
        reference = entry["reference"]
    elif not human:
        reference = "llm_adjudicated"
    else:
        reference = "human" if human.get("n") == scores.get("n") else "mixed"
    return {
        "classifier": entry.get("classifier"),
        "variant": entry.get("variant", "pre_registered"),
        **{key: scores.get(key) for key in _SCORE_KEYS},
        "reference": reference,
        "spearman": scores.get("spearman"),
        "human_subset": {key: human.get(key) for key in _SCORE_KEYS} if human else None,
    }


def _baseline_row(entry: Any) -> Any:
    if not isinstance(entry, dict) or not isinstance(entry.get("all"), dict):
        return entry
    scores, human = entry["all"], entry.get("human_subset")
    return {
        "name": entry.get("name"),
        "label": entry.get("label"),
        "post_hoc": entry.get("post_hoc", False),
        **{key: scores.get(key) for key in _SCORE_KEYS},
        "spearman": scores.get("spearman"),
        "human_subset": {key: human.get(key) for key in _SCORE_KEYS} if human else None,
    }
