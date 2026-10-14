"""Tests for the stance API with the database faked out (Postgres tests live in tests/db)."""

import json
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any

import psycopg
import pytest
from fastapi.testclient import TestClient

from app.api.database import NOT_LOADED, get_db
from app.api.stance import BENCHMARK_MISSING, STANCE_NOT_LOADED
from app.db import stance as stance_db
from app.db.connection import NOT_CONFIGURED
from app.schemas.stance import FedFundsTarget, StanceResult

MakeClient = Callable[..., TestClient]

CONNECTION = object()
URL = "https://www.federalreserve.gov/newsevents/pressreleases/monetary{}a.htm"


def _result(meeting_date: str, classifier: str = "llm_few_shot", **fields: Any) -> StanceResult:
    return StanceResult.model_validate(
        {
            "meeting_date": meeting_date,
            "doc_id": f"statement-{meeting_date}",
            "classifier": classifier,
            "stance": "dovish",
            "score": -0.55,
            "confidence": 0.7,
            "key_factors": ["rate cut"],
            "evidence": ["The Committee decided to lower the target range."],
            "model_version": "gpt-5.4-mini-2026-03-17/stance-v1",
            "url": URL.format(meeting_date.replace("-", "")),
        }
        | fields
    )


POINTS = [_result("2024-09-18"), _result("2024-11-07", stance="neutral", score=-0.1)]
FED_FUNDS = [
    FedFundsTarget(date=date(2023, 7, 27), lower=5.25, upper=5.5),
    FedFundsTarget(date=date(2024, 9, 19), lower=4.75, upper=5.0),
]


class FakeDatabase:
    """Stands in for app.db.connection.Database: lends a sentinel connection or raises `error`."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    @contextmanager
    def connection(self) -> Iterator[Any]:
        if self.error:
            raise self.error
        yield CONNECTION


class FakeQueries:
    """Records the stance queries and answers them, or raises `error` from each."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.error: Exception | None = None
        self.meeting: StanceResult | None = POINTS[0]

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in ("stance_timeline", "fed_funds_window", "stance_for_meeting"):
            monkeypatch.setattr(stance_db, name, self._recorder(name))

    def _recorder(self, name: str) -> Callable[..., Any]:
        answers = {
            "stance_timeline": POINTS,
            "fed_funds_window": FED_FUNDS,
            "stance_for_meeting": None,
        }

        def fake(conn: Any, *args: Any) -> Any:
            assert conn is CONNECTION
            self.calls.append((name, args))
            if self.error:
                raise self.error
            return self.meeting if name == "stance_for_meeting" else answers[name]

        return fake


@pytest.fixture
def queries(monkeypatch: pytest.MonkeyPatch) -> FakeQueries:
    fake = FakeQueries()
    fake.install(monkeypatch)
    return fake


@pytest.fixture
def api(client: TestClient) -> TestClient:
    client.app.dependency_overrides[get_db] = lambda: FakeDatabase()
    return client


def _write_gold(data_dir: Path, rows: list[dict[str, Any]], extra: str = "") -> None:
    (data_dir / "eval").mkdir(exist_ok=True)
    lines = "".join(json.dumps(row) + "\n" for row in rows) + extra
    (data_dir / "eval" / "stance_gold.jsonl").write_text(lines, encoding="utf-8")


GOLD = [
    {
        "doc_id": "statement-2024-09-18",
        "meeting_date": "2024-09-18",
        "label": "dovish",
        "confidence": 0.92,
        "label_source": "agreement",
        "annotator_a": "dovish",
        "annotator_b": "dovish",
        "evidence": ["The Committee decided to lower the target range."],
        "note": "",
        "human_label": None,
        "human_note": None,
    },
    {
        "doc_id": "statement-2024-11-07",
        "meeting_date": "2024-11-07",
        "label": "dovish",
        "confidence": "medium",
        "label_source": "adjudicated",
        "human_label": "neutral",
        "human_note": "A smaller cut, framed as recalibration.",
    },
]


@pytest.mark.parametrize(
    "url", ["/api/stance", "/api/stance?classifier=finbert", "/api/stance/2024-09-18"]
)
def test_503_without_database(client: TestClient, url: str) -> None:
    response = client.get(url)

    assert response.status_code == 503
    assert response.json() == {"detail": NOT_CONFIGURED}


def test_timeline_without_reference_labels(api: TestClient, queries: FakeQueries) -> None:
    response = api.get("/api/stance")

    assert response.status_code == 200
    body = response.json()
    assert body["classifier"] == "llm_few_shot"
    assert [p["meeting_date"] for p in body["points"]] == ["2024-09-18", "2024-11-07"]
    assert body["points"][0] == {
        "meeting_date": "2024-09-18",
        "doc_id": "statement-2024-09-18",
        "classifier": "llm_few_shot",
        "stance": "dovish",
        "score": -0.55,
        "confidence": 0.7,
        "key_factors": ["rate cut"],
        "evidence": ["The Committee decided to lower the target range."],
        "model_version": "gpt-5.4-mini-2026-03-17/stance-v1",
        "url": URL.format("20240918"),
        "reference": None,
    }
    assert body["fed_funds"] == [
        {"date": "2023-07-27", "lower": 5.25, "upper": 5.5},
        {"date": "2024-09-19", "lower": 4.75, "upper": 5.0},
    ]
    assert queries.calls == [
        ("stance_timeline", ("llm_few_shot", None, None)),
        ("fed_funds_window", (None, None)),
    ]


def test_timeline_params_reach_the_queries(api: TestClient, queries: FakeQueries) -> None:
    response = api.get("/api/stance?classifier=keyword&start=2024-01-01&end=2024-12-31")

    assert response.status_code == 200
    window = (date(2024, 1, 1), date(2024, 12, 31))
    assert queries.calls == [
        ("stance_timeline", ("keyword", *window)),
        ("fed_funds_window", window),
    ]


def test_reference_labels_come_from_the_gold_file(
    api: TestClient, queries: FakeQueries, tmp_path: Path
) -> None:
    _write_gold(tmp_path, GOLD)

    points = api.get("/api/stance").json()["points"]

    assert points[0]["reference"] == {
        "label": "dovish",
        "source": "llm_adjudicated",
        "confidence": 0.92,
    }
    # A human label overrides the LLM-adjudicated one, and carries no model confidence.
    assert points[1]["reference"] == {"label": "neutral", "source": "human", "confidence": None}


def test_gold_file_edits_are_picked_up(
    api: TestClient, queries: FakeQueries, tmp_path: Path
) -> None:
    _write_gold(tmp_path, GOLD[:1])
    assert api.get("/api/stance").json()["points"][1]["reference"] is None

    _write_gold(tmp_path, GOLD)
    gold = tmp_path / "eval" / "stance_gold.jsonl"
    os.utime(gold, ns=(gold.stat().st_atime_ns, gold.stat().st_mtime_ns + 10**9))

    assert api.get("/api/stance").json()["points"][1]["reference"]["source"] == "human"


def test_bad_gold_lines_are_skipped(
    api: TestClient, queries: FakeQueries, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    blank_human = GOLD[1] | {"human_label": "  "}
    _write_gold(
        tmp_path,
        [{"doc_id": "statement-2024-09-18", "label": "very hawkish"}, blank_human],
        extra="not json\n\n",
    )

    with caplog.at_level("WARNING"):
        points = api.get("/api/stance").json()["points"]

    assert points[0]["reference"] is None
    assert points[1]["reference"] == {
        "label": "dovish",
        "source": "llm_adjudicated",
        "confidence": "medium",
    }
    assert "skipped 2 line(s)" in caplog.text


@pytest.mark.parametrize(
    "url", ["/api/stance?classifier=finbert", "/api/stance/2024-09-18?classifier=finbert"]
)
def test_not_loaded_is_503(api: TestClient, queries: FakeQueries, url: str) -> None:
    queries.error = stance_db.StanceNotLoadedError("finbert")

    response = api.get(url)

    assert response.status_code == 503
    assert response.json() == {"detail": STANCE_NOT_LOADED}
    assert STANCE_NOT_LOADED == "Stance results not loaded: run make stance."


def test_unloaded_database_is_503_db_load(api: TestClient, queries: FakeQueries) -> None:
    queries.error = psycopg.errors.UndefinedTable('relation "fedrag.documents" does not exist')

    response = api.get("/api/stance")

    assert response.status_code == 503
    assert response.json() == {"detail": NOT_LOADED}


def test_meeting(api: TestClient, queries: FakeQueries, tmp_path: Path) -> None:
    _write_gold(tmp_path, GOLD)

    response = api.get("/api/stance/2024-09-18")

    assert response.status_code == 200
    assert response.json()["doc_id"] == "statement-2024-09-18"
    assert response.json()["reference"]["label"] == "dovish"
    assert queries.calls == [("stance_for_meeting", (date(2024, 9, 18), "llm_few_shot"))]


def test_unknown_meeting_is_404(api: TestClient, queries: FakeQueries) -> None:
    queries.meeting = None

    response = api.get("/api/stance/2024-09-19?classifier=keyword")

    assert response.status_code == 404
    assert "2024-09-19" in response.json()["detail"]
    assert queries.calls == [("stance_for_meeting", (date(2024, 9, 19), "keyword"))]


# --- GET /api/stance/benchmark ----------------------------------------------------------------

DATASET = {
    "name": "Trillion Dollar Words",
    "citation": "Shah, Paturi and Chava (2023), ACL.",
    "license": "CC BY-NC 4.0",
    "url": "https://github.com/gtfintechlab/fomc-hawkish-dovish",
    "commit": "98646987452b326507479cf641571b33814bb73f",
}
SENTENCE_ROW = {
    "classifier": "finbert",
    "split": "official",
    "macro_f1_mean": 0.69,
    "macro_f1_sd": 0.012,
    "accuracy_mean": 0.71,
    "accuracy_sd": 0.01,
    "per_class_f1_mean": {"dovish": 0.66, "neutral": 0.74, "hawkish": 0.67},
    "n_seeds": 3,
    "n_test_mean": 496.0,
}
STATEMENT_ROW = {
    "classifier": "llm_few_shot",
    "variant": "pre_registered",
    "n": 97,
    "accuracy": 0.81,
    "macro_f1": 0.78,
    "kappa": 0.69,
    "accuracy_ci": {"low": 0.73, "high": 0.88},
    "macro_f1_ci": {"low": 0.7, "high": 0.86},
    "reference": "llm_adjudicated",
    "spearman": 0.88,
    "human_subset": None,
}
BENCHMARK = {
    "generated_at": "2026-10-13T08:00:00Z",
    "dataset": DATASET,
    "sentence_level": [SENTENCE_ROW, SENTENCE_ROW | {"split": "deduplicated"}],
    "statement_level": [STATEMENT_ROW],
    "models": {"llm": "gpt-5.4-mini-2026-03-17"},
    "cost": {"usd": 1.23},
}


def _write_benchmark(data_dir: Path, benchmark: Any) -> None:
    (data_dir / "eval").mkdir(exist_ok=True)
    text = benchmark if isinstance(benchmark, str) else json.dumps(benchmark)
    (data_dir / "eval" / "stance_benchmark.json").write_text(text, encoding="utf-8")


def test_benchmark_missing_is_503(client: TestClient) -> None:
    response = client.get("/api/stance/benchmark")

    assert response.status_code == 503
    assert response.json() == {"detail": BENCHMARK_MISSING}


def test_benchmark(client: TestClient, tmp_path: Path) -> None:
    _write_benchmark(tmp_path, BENCHMARK)

    response = client.get("/api/stance/benchmark")  # no database needed

    assert response.status_code == 200
    body = response.json()
    assert body["dataset"] == DATASET
    assert body["sentence_level"] == [SENTENCE_ROW, SENTENCE_ROW | {"split": "deduplicated"}]
    assert body["statement_level"] == [STATEMENT_ROW]
    assert body["generated_at"] == "2026-10-13T08:00:00Z"
    assert body["reference_labels"] is None
    assert "models" not in body and "cost" not in body


def test_benchmark_counts_reference_labels_as_they_stand(
    client: TestClient, tmp_path: Path
) -> None:
    _write_benchmark(tmp_path, BENCHMARK)
    _write_gold(tmp_path, GOLD)

    body = client.get("/api/stance/benchmark").json()

    assert body["reference_labels"] == {"n_labels": 2, "n_human_reviewed": 1}


def test_benchmark_accepts_metrics_reports(client: TestClient, tmp_path: Path) -> None:
    def mean_sd(mean: float, sd: float | None) -> dict[str, float | None]:
        return {"mean": mean, "sd": sd}

    def split_report(f1: float) -> dict[str, Any]:
        return {
            "n": [476, 476, 476],
            "macro_f1": mean_sd(f1, 0.02),
            "accuracy": mean_sd(f1 + 0.01, None),
            "kappa": mean_sd(0.5, 0.01),
            "per_class": {
                stance: {"f1": mean_sd(f1, 0.03), "support": [100, 100, 100]}
                for stance in ("dovish", "neutral", "hawkish")
            },
        }

    sentence_report = {
        "classifier": "keyword",
        "seeds": [5768, 78516, 944601],
        "official": split_report(0.45),
        "deduplicated": split_report(0.44),
    }
    ci = {"low": 0.5, "high": 0.7}
    statement_report = {
        "classifier": "keyword",
        "all": {"n": 97, "accuracy": 0.6, "macro_f1": 0.55, "kappa": None, "spearman": 0.4},
        "human_subset": {"n": 25, "accuracy": 0.6, "macro_f1": 0.5, "kappa": 0.3, "spearman": 0.2},
    }
    post_hoc_report = {
        "classifier": "keyword",
        "variant": "post_hoc_rubric_ordered",
        "reference": "mixed",
        "all": {"n": 97, "accuracy": 0.6, "macro_f1": 0.55, "kappa": 0.4, "accuracy_ci": ci},
        "human_subset": None,
    }
    _write_benchmark(
        tmp_path,
        BENCHMARK
        | {
            "sentence_level": [sentence_report],
            "statement_level": [statement_report, post_hoc_report],
        },
    )

    body = client.get("/api/stance/benchmark").json()

    assert [(r["split"], r["macro_f1_mean"]) for r in body["sentence_level"]] == [
        ("official", 0.45),
        ("deduplicated", 0.44),
    ]
    assert body["sentence_level"][0]["accuracy_sd"] is None
    assert body["sentence_level"][0]["n_seeds"] == 3
    assert body["sentence_level"][0]["n_test_mean"] == 476
    no_ci = {"accuracy_ci": None, "macro_f1_ci": None}
    assert body["statement_level"] == [
        {
            "classifier": "keyword",
            "variant": "pre_registered",  # entries without a variant are pre-registered
            "n": 97,
            "accuracy": 0.6,
            "macro_f1": 0.55,
            "kappa": None,
            **no_ci,
            "reference": "mixed",
            "spearman": 0.4,
            "human_subset": {"n": 25, "accuracy": 0.6, "macro_f1": 0.5, "kappa": 0.3, **no_ci},
        },
        {
            "classifier": "keyword",
            "variant": "post_hoc_rubric_ordered",
            "n": 97,
            "accuracy": 0.6,
            "macro_f1": 0.55,
            "kappa": 0.4,
            "accuracy_ci": ci,
            "macro_f1_ci": None,
            "reference": "mixed",
            "spearman": None,
            "human_subset": None,
        },
    ]
    assert body["statement_bootstrap"] is None


def test_benchmark_serves_the_bootstrap(client: TestClient, tmp_path: Path) -> None:
    def difference(value: float) -> dict[str, Any]:
        return {"difference": value, "ci": {"low": value - 0.05, "high": value + 0.05}}

    bootstrap = {
        "method": "Percentile bootstrap",
        "resamples": 10_000,
        "seed": 1,
        "level": 0.95,
        "paired_differences": [
            {
                "minuend": {"classifier": "finbert", "variant": "post_hoc_rubric_ordered"},
                "subtrahend": {"classifier": "finbert", "variant": "pre_registered"},
                "subset": "all",
                "n": 97,
                "accuracy": difference(0.2),
                "macro_f1": difference(0.25),
            }
        ],
        "baseline_differences": [
            {
                "row": {"classifier": "finbert", "variant": "post_hoc_rubric_ordered"},
                "baseline": "rate_move_rule",
                "subset": "all",
                "n": 97,
                "accuracy": difference(0.0),
                "macro_f1": difference(-0.002),
            }
        ],
    }
    year_blocks = {"method": "whole years", "years": 12, "paired_differences": []}
    _write_benchmark(
        tmp_path,
        BENCHMARK | {"statement_bootstrap": bootstrap | {"year_blocks": year_blocks}},
    )

    body = client.get("/api/stance/benchmark").json()

    assert body["statement_bootstrap"] == bootstrap  # year_blocks stays in the file


def test_benchmark_serves_the_baselines(client: TestClient, tmp_path: Path) -> None:
    def split_report(accuracy: float) -> dict[str, Any]:
        stat = {"mean": accuracy, "sd": 0.012}
        return {"n": [476, 476, 476], "macro_f1": {"mean": 0.217, "sd": 0.004}, "accuracy": stat}

    sentence_baseline = {
        "name": "majority_class",
        "label": "neutral",
        "labels": {"5768": "neutral"},
        "seeds": [5768, 78516, 944601],
        "official": split_report(0.482),
        "deduplicated": split_report(0.48),
    }
    ci = {"low": 0.29, "high": 0.49}
    statement_baselines = [
        {
            "name": "majority_class",
            "label": "hawkish",
            "post_hoc": False,
            "description": "Always the most frequent reference label.",
            "predicted": {"dovish": 0, "neutral": 0, "hawkish": 97},
            "all": {
                "n": 97,
                "accuracy": 0.381,
                "macro_f1": 0.184,
                "kappa": 0.0,
                "spearman": None,
                "accuracy_ci": ci,
                "accuracy_ci_year_blocks": {"low": 0.17, "high": 0.62},
            },
            "human_subset": {"n": 25, "accuracy": 0.36, "macro_f1": 0.176, "kappa": 0.0},
        },
        {
            "name": "rate_move_rule",
            "label": None,
            "post_hoc": True,
            "all": {"n": 97, "accuracy": 0.691, "macro_f1": 0.678, "kappa": 0.522},
            "human_subset": None,
        },
    ]
    _write_benchmark(
        tmp_path,
        BENCHMARK
        | {"sentence_baselines": [sentence_baseline], "statement_baselines": statement_baselines},
    )

    body = client.get("/api/stance/benchmark").json()

    assert body["sentence_baselines"] == [
        {
            "name": "majority_class",
            "label": "neutral",
            "split": split,
            "macro_f1_mean": 0.217,
            "macro_f1_sd": 0.004,
            "accuracy_mean": accuracy,
            "accuracy_sd": 0.012,
            "n_seeds": 3,
            "n_test_mean": 476,
        }
        for split, accuracy in (("official", 0.482), ("deduplicated", 0.48))
    ]
    majority, rule = body["statement_baselines"]
    assert majority == {
        "name": "majority_class",
        "label": "hawkish",
        "post_hoc": False,
        "n": 97,
        "accuracy": 0.381,
        "macro_f1": 0.184,
        "kappa": 0.0,
        "spearman": None,
        "accuracy_ci": ci,
        "macro_f1_ci": None,
        "human_subset": {
            "n": 25,
            "accuracy": 0.36,
            "macro_f1": 0.176,
            "kappa": 0.0,
            "accuracy_ci": None,
            "macro_f1_ci": None,
        },
    }
    assert (rule["name"], rule["label"], rule["post_hoc"], rule["human_subset"]) == (
        "rate_move_rule",
        None,
        True,
        None,
    )
    assert client.get("/api/stance/benchmark").json() == body  # served from the cache


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        "[]",
        json.dumps(BENCHMARK | {"sentence_level": [SENTENCE_ROW | {"split": "train"}]}),
        json.dumps({k: v for k, v in BENCHMARK.items() if k != "dataset"}),
        json.dumps(BENCHMARK | {"statement_level": [STATEMENT_ROW | {"variant": "tuned"}]}),
        json.dumps(
            BENCHMARK
            | {"statement_level": [STATEMENT_ROW | {"accuracy_ci": {"low": 0.9, "high": 0.8}}]}
        ),
        json.dumps(
            BENCHMARK
            | {
                "statement_baselines": [
                    {"name": "coin_flip", "label": None, "post_hoc": False, "all": {"n": 1}}
                ]
            }
        ),
    ],
    ids=[
        "not-json",
        "not-an-object",
        "unknown-split",
        "no-dataset",
        "unknown-variant",
        "bad-ci",
        "unknown-baseline",
    ],
)
def test_invalid_benchmark_is_503(client: TestClient, tmp_path: Path, content: str) -> None:
    _write_benchmark(tmp_path, content)

    response = client.get("/api/stance/benchmark")

    assert response.status_code == 503
    assert response.json()["detail"].startswith("Stance benchmark file")
