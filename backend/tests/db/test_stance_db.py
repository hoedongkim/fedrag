"""Tests for load_stance, the stance queries and the stance API against a real Postgres."""

import json
from collections.abc import Callable
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import psycopg
import pytest
from fastapi.testclient import TestClient

from app.db.load import LoadError, load_processed
from app.db.stance import (
    StanceNotLoadedError,
    fed_funds_window,
    load_stance,
    stance_for_meeting,
    stance_timeline,
)
from app.schemas.stance import FedFundsTarget
from app.stance.types import StatementPrediction

pytestmark = pytest.mark.db

Conn = psycopg.Connection[Any]
Row = dict[str, Any]
WriteProcessed = Callable[[list[Row], list[Row]], Path]
MakeClient = Callable[..., TestClient]

FED = "https://www.federalreserve.gov/newsevents/pressreleases"


def _prediction(meeting_date: str, classifier: str = "llm_few_shot", **fields: Any) -> Any:
    return StatementPrediction(
        doc_id=f"statement-{meeting_date}",
        meeting_date=date.fromisoformat(meeting_date),
        classifier=classifier,  # type: ignore[arg-type]
        stance=fields.pop("stance", "dovish"),
        score=fields.pop("score", -0.6),
        confidence=fields.pop("confidence", 0.8),
        key_factors=fields.pop("key_factors", ("rate cut",)),
        evidence=fields.pop("evidence", ("Rates were cut sharply.",)),
        model_version=fields.pop("model_version", "gpt-5.4-mini/stance-v1"),
        **fields,
    )


PREDICTIONS = [
    _prediction("2024-11-07", score=-0.4),
    _prediction("2020-03-15", score=-0.9),
    _prediction("2024-09-18", score=-0.7, evidence=("The Committee lowered the target range.",)),
    _prediction("2024-09-18", "keyword", stance="neutral", score=0.1, confidence=None),
]

# Daily values, as FRED publishes them: five change points.
DAILY = [
    FedFundsTarget(date=date(2020, 3, 1), lower=1.5, upper=1.75),
    FedFundsTarget(date=date(2020, 3, 2), lower=1.5, upper=1.75),
    FedFundsTarget(date=date(2020, 3, 4), lower=1.0, upper=1.25),
    FedFundsTarget(date=date(2020, 3, 16), lower=0.0, upper=0.25),
    FedFundsTarget(date=date(2020, 3, 17), lower=0.0, upper=0.25),
    FedFundsTarget(date=date(2024, 9, 19), lower=4.75, upper=5.0),
    FedFundsTarget(date=date(2024, 11, 8), lower=4.5, upper=4.75),
    FedFundsTarget(date=date(2024, 11, 9), lower=4.5, upper=4.75),
]


@pytest.fixture
def documents_loaded(
    db_conn: Conn,
    write_processed: WriteProcessed,
    sample_meetings: list[Row],
    sample_documents: list[Row],
) -> Conn:
    load_processed(db_conn, write_processed(sample_meetings, sample_documents))
    return db_conn


@pytest.fixture
def loaded(documents_loaded: Conn) -> Conn:
    load_stance(documents_loaded, PREDICTIONS, DAILY)
    return documents_loaded


def _count(conn: Conn, table: str) -> int:
    return conn.execute(f"SELECT count(*) AS n FROM fedrag.{table}").fetchone()["n"]


def _fed_funds(conn: Conn) -> list[tuple[str, float, float]]:
    rows = conn.execute("SELECT date, lower, upper FROM fedrag.fed_funds_target ORDER BY date")
    return [(r["date"].isoformat(), float(r["lower"]), float(r["upper"])) for r in rows]


def test_load_returns_counts_and_is_idempotent(documents_loaded: Conn) -> None:
    first = load_stance(documents_loaded, PREDICTIONS, DAILY)
    second = load_stance(documents_loaded, PREDICTIONS, DAILY)

    assert first == second == {"stance_results": 4, "fed_funds_target": 5}
    assert _count(documents_loaded, "stance_results") == 4
    assert _fed_funds(documents_loaded) == [
        ("2020-03-01", 1.5, 1.75),
        ("2020-03-04", 1.0, 1.25),
        ("2020-03-16", 0.0, 0.25),
        ("2024-09-19", 4.75, 5.0),
        ("2024-11-08", 4.5, 4.75),
    ]


def test_loaded_row_matches_prediction(loaded: Conn) -> None:
    row = loaded.execute(
        "SELECT doc_id, meeting_date, classifier, stance, score, confidence, key_factors, "
        "evidence, model_version, computed_at FROM fedrag.stance_results "
        "WHERE doc_id = 'statement-2024-09-18' AND classifier = 'llm_few_shot'"
    ).fetchone()

    assert row.pop("computed_at") is not None
    assert row == {
        "doc_id": "statement-2024-09-18",
        "meeting_date": date(2024, 9, 18),
        "classifier": "llm_few_shot",
        "stance": "dovish",
        "score": -0.7,
        "confidence": 0.8,
        "key_factors": ["rate cut"],
        "evidence": ["The Committee lowered the target range."],
        "model_version": "gpt-5.4-mini/stance-v1",
    }


def test_reload_replaces_one_prediction_and_keeps_the_rest(loaded: Conn) -> None:
    revised = replace(PREDICTIONS[2], stance="neutral", score=0.05, key_factors=())

    load_stance(loaded, [revised])

    row = loaded.execute(
        "SELECT stance, score, key_factors FROM fedrag.stance_results "
        "WHERE doc_id = 'statement-2024-09-18' AND classifier = 'llm_few_shot'"
    ).fetchone()
    assert row == {"stance": "neutral", "score": 0.05, "key_factors": []}
    assert _count(loaded, "stance_results") == 4
    assert len(_fed_funds(loaded)) == 5  # an empty series leaves the stored one alone


def test_new_fed_funds_series_replaces_the_old_one(loaded: Conn) -> None:
    load_stance(loaded, [], [FedFundsTarget(date=date(2020, 3, 16), lower=0.0, upper=0.25)])

    assert _fed_funds(loaded) == [("2020-03-16", 0.0, 0.25)]


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        (_prediction("2024-11-07", score=1.5), "score"),
        (_prediction("2024-11-07", confidence=-0.1), "confidence"),
        (_prediction("2024-11-07", stance="very hawkish"), "stance"),
        (_prediction("2024-11-07", classifier="vibes"), "classifier"),
        (replace(_prediction("2024-11-07"), doc_id="statement-2024-09-18"), "not a document of"),
        (PREDICTIONS[0], "appears 2 times"),
    ],
    ids=["score", "confidence", "stance", "classifier", "doc-for-other-meeting", "duplicate"],
)
def test_invalid_prediction_loads_nothing(
    documents_loaded: Conn, bad: StatementPrediction, message: str
) -> None:
    with pytest.raises(LoadError, match=message):
        load_stance(documents_loaded, [*PREDICTIONS, bad], DAILY)

    assert _count(documents_loaded, "stance_results") == 0
    assert _count(documents_loaded, "fed_funds_target") == 0


def test_unknown_document_rolls_back(documents_loaded: Conn) -> None:
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        load_stance(documents_loaded, [*PREDICTIONS, _prediction("2024-07-31")], DAILY)

    assert _count(documents_loaded, "stance_results") == 0
    assert _count(documents_loaded, "fed_funds_target") == 0


def test_conflicting_fed_funds_dates_are_rejected(documents_loaded: Conn) -> None:
    clash = [
        FedFundsTarget(date=date(2020, 3, 16), lower=0.0, upper=0.25),
        FedFundsTarget(date=date(2020, 3, 16), lower=0.25, upper=0.5),
    ]

    with pytest.raises(LoadError, match="appears more than once"):
        load_stance(documents_loaded, PREDICTIONS, clash)


def test_timeline_in_meeting_order_with_document_url(loaded: Conn) -> None:
    points = stance_timeline(loaded, "llm_few_shot")

    assert [p.meeting_date.isoformat() for p in points] == [
        "2020-03-15",
        "2024-09-18",
        "2024-11-07",
    ]
    assert points[1].url == f"{FED}/monetary20240918a.htm"
    assert points[1].evidence == ["The Committee lowered the target range."]
    assert points[1].reference is None  # attached by the route, not the query


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        ("2024-01-01", None, ["2024-09-18", "2024-11-07"]),
        (None, "2024-09-18", ["2020-03-15", "2024-09-18"]),
        ("2024-09-18", "2024-09-18", ["2024-09-18"]),
        ("2021-01-01", "2021-12-31", []),
    ],
    ids=["start", "end", "one-day", "empty-window"],
)
def test_timeline_window(
    loaded: Conn, start: str | None, end: str | None, expected: list[str]
) -> None:
    points = stance_timeline(
        loaded,
        "llm_few_shot",
        date.fromisoformat(start) if start else None,
        date.fromisoformat(end) if end else None,
    )

    assert [p.meeting_date.isoformat() for p in points] == expected


@pytest.mark.parametrize("missing", ["tables", "rows", "classifier"])
def test_not_loaded_raises(documents_loaded: Conn, missing: str) -> None:
    if missing == "tables":  # a database loaded before the stance tables were added
        documents_loaded.execute("DROP TABLE fedrag.stance_results, fedrag.fed_funds_target")
    if missing == "classifier":
        load_stance(documents_loaded, PREDICTIONS)

    with pytest.raises(StanceNotLoadedError):
        stance_timeline(documents_loaded, "finbert")
    with pytest.raises(StanceNotLoadedError):
        stance_for_meeting(documents_loaded, date(2024, 9, 18), "finbert")


def test_missing_documents_table_is_not_a_stance_problem(db_conn: Conn) -> None:
    # Nothing loaded at all: the API's answer is "run make db-load", not "run make stance".
    with pytest.raises(psycopg.errors.UndefinedTable):
        stance_timeline(db_conn, "finbert")
    with pytest.raises(psycopg.errors.UndefinedTable):
        fed_funds_window(db_conn)


def test_stance_for_meeting(loaded: Conn) -> None:
    result = stance_for_meeting(loaded, date(2024, 9, 18), "keyword")

    assert result is not None
    assert (result.doc_id, result.stance, result.score, result.confidence) == (
        "statement-2024-09-18",
        "neutral",
        0.1,
        None,
    )
    assert stance_for_meeting(loaded, date(2024, 9, 19), "keyword") is None
    assert stance_for_meeting(loaded, date(2024, 11, 7), "keyword") is None


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        (None, None, ["2020-03-01", "2020-03-04", "2020-03-16", "2024-09-19", "2024-11-08"]),
        ("2024-01-01", None, ["2020-03-16", "2024-09-19", "2024-11-08"]),
        ("2020-03-04", "2024-10-01", ["2020-03-04", "2020-03-16", "2024-09-19"]),
        ("2019-01-01", "2020-03-10", ["2020-03-01", "2020-03-04"]),
        (None, "2019-12-31", []),
    ],
    ids=["all", "in-force-at-start", "start-on-change", "start-before-first", "before-first"],
)
def test_fed_funds_window(
    loaded: Conn, start: str | None, end: str | None, expected: list[str]
) -> None:
    points = fed_funds_window(
        loaded,
        date.fromisoformat(start) if start else None,
        date.fromisoformat(end) if end else None,
    )

    assert [p.date.isoformat() for p in points] == expected


def test_api_reads_postgres_and_reference_labels(
    loaded: Conn, make_client: MakeClient, test_database_url: str, tmp_path: Path
) -> None:
    (tmp_path / "eval").mkdir()
    gold = [
        {"doc_id": "statement-2024-09-18", "label": "dovish", "confidence": 0.9},
        {"doc_id": "statement-2020-03-15", "label": "neutral", "human_label": "dovish"},
    ]
    (tmp_path / "eval" / "stance_gold.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in gold), encoding="utf-8"
    )

    with make_client(database_url=test_database_url) as client:
        timeline = client.get("/api/stance", params={"start": "2020-03-10", "end": "2024-10-01"})
        meeting = client.get("/api/stance/2024-09-18", params={"classifier": "keyword"})
        missing = client.get("/api/stance/2024-09-19")
        unloaded = client.get("/api/stance", params={"classifier": "finbert"})

    assert timeline.status_code == 200
    body = timeline.json()
    assert [p["meeting_date"] for p in body["points"]] == ["2020-03-15", "2024-09-18"]
    assert body["points"][0]["reference"] == {
        "label": "dovish",
        "source": "human",
        "confidence": None,
    }
    assert body["points"][1]["reference"] == {
        "label": "dovish",
        "source": "llm_adjudicated",
        "confidence": 0.9,
    }
    assert body["fed_funds"] == [
        {"date": "2020-03-04", "lower": 1.0, "upper": 1.25},
        {"date": "2020-03-16", "lower": 0.0, "upper": 0.25},
        {"date": "2024-09-19", "lower": 4.75, "upper": 5.0},
    ]
    assert meeting.status_code == 200
    assert meeting.json()["stance"] == "neutral"
    assert meeting.json()["reference"]["label"] == "dovish"
    assert missing.status_code == 404
    assert unloaded.status_code == 503
    assert unloaded.json() == {"detail": "Stance results not loaded: run make stance."}
