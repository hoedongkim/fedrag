"""Tests for list_meetings and GET /api/meetings against a real Postgres."""

from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import psycopg
import pytest
from fastapi.testclient import TestClient

from app.db.connection import apply_schema
from app.db.load import load_processed
from app.db.queries import list_meetings

pytestmark = pytest.mark.db

Conn = psycopg.Connection[Any]
Row = dict[str, Any]
MakeDocument = Callable[..., Row]
WriteProcessed = Callable[[list[Row], list[Row]], Path]
MakeClient = Callable[..., TestClient]

BOTH = ["statement", "minutes"]

# (doc_type, as_of) -> {meeting_date: doc_types}. Minutes for 2024-09-18 came out on 2024-10-09,
# for 2024-11-07 on 2024-11-26; 2026-10-28 is a future meeting with no documents yet.
CASES: dict[str, tuple[str | None, str | None, dict[str, list[str]]]] = {
    "no-filter": (
        None,
        None,
        {"2020-03-15": ["statement"], "2024-09-18": BOTH, "2024-11-07": BOTH, "2026-10-28": []},
    ),
    "day-before-minutes": (
        None,
        "2024-10-08",
        {"2020-03-15": ["statement"], "2024-09-18": ["statement"]},
    ),
    "minutes-release-day": (None, "2024-10-09", {"2020-03-15": ["statement"], "2024-09-18": BOTH}),
    "first-day-of-meeting": (None, "2024-09-17", {"2020-03-15": ["statement"]}),
    "statement-day": (
        None,
        "2024-11-07",
        {"2020-03-15": ["statement"], "2024-09-18": BOTH, "2024-11-07": ["statement"]},
    ),
    "before-everything": (None, "2019-12-31", {}),
    "minutes": ("minutes", None, {"2024-09-18": BOTH, "2024-11-07": BOTH}),
    "minutes-not-yet-released": ("minutes", "2024-11-25", {"2024-09-18": BOTH}),
    "statement": (
        "statement",
        None,
        {"2020-03-15": ["statement"], "2024-09-18": BOTH, "2024-11-07": BOTH},
    ),
    "no-sep-documents": ("sep", None, {}),
}


@pytest.fixture
def loaded(
    db_conn: Conn,
    write_processed: WriteProcessed,
    sample_meetings: list[Row],
    sample_documents: list[Row],
) -> Conn:
    load_processed(db_conn, write_processed(sample_meetings, sample_documents))
    return db_conn


@pytest.mark.parametrize(("doc_type", "as_of", "expected"), CASES.values(), ids=CASES.keys())
def test_list_meetings_filters(
    loaded: Conn, doc_type: str | None, as_of: str | None, expected: dict[str, list[str]]
) -> None:
    meetings = list_meetings(
        loaded, doc_type=doc_type, as_of=date.fromisoformat(as_of) if as_of else None
    )

    assert {m.meeting_date.isoformat(): m.doc_types for m in meetings} == expected
    assert [m.meeting_date for m in meetings] == sorted(m.meeting_date for m in meetings)


def test_unscheduled_meeting_is_hidden_until_its_first_document(
    db_conn: Conn,
    write_processed: WriteProcessed,
    sample_meetings: list[Row],
    sample_documents: list[Row],
    make_document: MakeDocument,
) -> None:
    # Like October 4, 2019: held on day D, made public by a statement released on D+7.
    unscheduled = sample_meetings[1] | {"meeting_date": "2019-10-04", "start_date": "2019-10-04"}
    statement = make_document(
        "statement", "2019-10-04", "2019-10-11", [(None, "Reserve management purchases.")]
    )
    load_processed(
        db_conn,
        write_processed([*sample_meetings, unscheduled], [*sample_documents, statement]),
    )

    def listed(as_of: str) -> dict[str, list[str]]:
        meetings = list_meetings(db_conn, as_of=date.fromisoformat(as_of))
        return {m.meeting_date.isoformat(): m.doc_types for m in meetings}

    assert "2019-10-04" not in listed("2019-10-10")
    assert listed("2019-10-11") == {"2019-10-04": ["statement"]}


def test_list_meetings_carries_meeting_facts(loaded: Conn) -> None:
    by_date = {m.meeting_date.isoformat(): m for m in list_meetings(loaded)}

    assert by_date["2020-03-15"].unscheduled is True
    assert by_date["2020-03-15"].start_date == date(2020, 3, 15)
    assert by_date["2024-09-18"].start_date == date(2024, 9, 17)
    assert by_date["2024-09-18"].has_sep is True
    assert by_date["2024-11-07"].has_sep is False
    assert by_date["2024-11-07"].has_press_conference is True


def test_list_meetings_empty_schema(db_conn: Conn) -> None:
    apply_schema(db_conn)

    assert list_meetings(db_conn) == []


def test_api_reads_postgres(loaded: Conn, make_client: MakeClient, test_database_url: str) -> None:
    with make_client(database_url=test_database_url) as client:  # runs the lifespan: pool closes
        response = client.get("/api/meetings", params={"as_of": "2024-10-08"})
        health = client.get("/api/health").json()

    assert response.status_code == 200
    assert response.json() == [
        {
            "meeting_date": "2020-03-15",
            "start_date": "2020-03-15",
            "unscheduled": True,
            "has_sep": False,
            "has_press_conference": True,
            "doc_types": ["statement"],
        },
        {
            "meeting_date": "2024-09-18",
            "start_date": "2024-09-17",
            "unscheduled": False,
            "has_sep": True,
            "has_press_conference": True,
            "doc_types": ["statement"],
        },
    ]
    assert health["database_configured"] is True
