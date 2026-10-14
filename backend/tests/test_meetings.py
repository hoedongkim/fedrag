"""Tests for GET /api/meetings with the database faked out (Postgres tests live in tests/db)."""

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import date
from typing import Any

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg import OperationalError
from psycopg_pool import PoolTimeout
from pydantic import SecretStr

from app.api.database import NOT_LOADED, UNAVAILABLE, get_db
from app.db import queries
from app.db.connection import MALFORMED_URL, NOT_CONFIGURED, Database, DatabaseUrlError
from app.schemas.meetings import MeetingSummary

MakeClient = Callable[..., TestClient]

CONNECTION = object()
MEETINGS = [
    MeetingSummary(
        meeting_date=date(2024, 9, 18),
        start_date=date(2024, 9, 17),
        unscheduled=False,
        has_sep=True,
        has_press_conference=True,
        doc_types=["statement", "minutes"],
    ),
    MeetingSummary(
        meeting_date=date(2024, 11, 7),
        start_date=date(2024, 11, 6),
        unscheduled=False,
        has_sep=False,
        has_press_conference=True,
        doc_types=["statement"],
    ),
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


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace the query layer with a recorder that returns MEETINGS."""
    recorded: list[dict[str, Any]] = []

    def fake_list_meetings(conn: Any, doc_type: Any = None, as_of: Any = None) -> list:
        recorded.append({"conn": conn, "doc_type": doc_type, "as_of": as_of})
        return MEETINGS

    monkeypatch.setattr(queries, "list_meetings", fake_list_meetings)
    return recorded


def _with_db(client: TestClient, database: Any) -> TestClient:
    client.app.dependency_overrides[get_db] = lambda: database
    return client


@pytest.mark.parametrize(
    "url", ["/api/meetings", "/api/meetings?doc_type=minutes&as_of=2024-10-01"]
)
def test_503_without_database(client: TestClient, url: str) -> None:
    response = client.get(url)

    assert response.status_code == 503
    assert response.json() == {"detail": NOT_CONFIGURED}
    assert NOT_CONFIGURED == "Database not configured: set DATABASE_URL in backend/.env."


def test_response_shape(client: TestClient, calls: list[dict[str, Any]]) -> None:
    response = _with_db(client, FakeDatabase()).get("/api/meetings")

    assert response.status_code == 200
    assert response.json() == [
        {
            "meeting_date": "2024-09-18",
            "start_date": "2024-09-17",
            "unscheduled": False,
            "has_sep": True,
            "has_press_conference": True,
            "doc_types": ["statement", "minutes"],
        },
        {
            "meeting_date": "2024-11-07",
            "start_date": "2024-11-06",
            "unscheduled": False,
            "has_sep": False,
            "has_press_conference": True,
            "doc_types": ["statement"],
        },
    ]
    assert calls == [{"conn": CONNECTION, "doc_type": None, "as_of": None}]


@pytest.mark.parametrize(
    ("query", "doc_type", "as_of"),
    [
        ("doc_type=minutes", "minutes", None),
        ("as_of=2024-10-09", None, date(2024, 10, 9)),
        ("doc_type=statement&as_of=2020-03-15", "statement", date(2020, 3, 15)),
    ],
)
def test_params_reach_the_query(
    client: TestClient,
    calls: list[dict[str, Any]],
    query: str,
    doc_type: str | None,
    as_of: date | None,
) -> None:
    response = _with_db(client, FakeDatabase()).get(f"/api/meetings?{query}")

    assert response.status_code == 200
    assert calls == [{"conn": CONNECTION, "doc_type": doc_type, "as_of": as_of}]


@pytest.mark.parametrize(
    "query",
    ["doc_type=memo", "as_of=2024-13-01", "as_of=yesterday", "asof=2024-10-01", "start=2024-01-01"],
    ids=["unknown-doc-type", "bad-date", "not-a-date", "misspelled-as-of", "unknown-param"],
)
@pytest.mark.parametrize("database", [None, FakeDatabase()], ids=["no-database", "database"])
def test_bad_params_are_422(
    client: TestClient,
    calls: list[dict[str, Any]],
    query: str,
    database: FakeDatabase | None,
) -> None:
    if database:
        _with_db(client, database)

    assert client.get(f"/api/meetings?{query}").status_code == 422
    assert calls == []


@pytest.mark.parametrize(
    "error",
    [PoolTimeout("couldn't get a connection after 15.00 sec"), OperationalError("server closed")],
    ids=["pool-timeout", "connection-lost"],
)
def test_unreachable_database_is_503(client: TestClient, error: Exception) -> None:
    response = _with_db(client, FakeDatabase(error)).get("/api/meetings")

    assert response.status_code == 503
    assert response.json() == {"detail": UNAVAILABLE}


def test_unloaded_database_is_503(client: TestClient) -> None:
    error = psycopg.errors.UndefinedTable('relation "fedrag.meetings" does not exist')

    response = _with_db(client, FakeDatabase(error)).get("/api/meetings")

    assert response.status_code == 503
    assert response.json() == {"detail": NOT_LOADED}


def test_malformed_database_url_never_reaches_the_pool_or_the_logs(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    database = Database(SecretStr("postgresql://u:Sentinel%ZZpw@127.0.0.1:1/db"))

    with caplog.at_level("DEBUG"):
        with pytest.raises(DatabaseUrlError), database.connection():
            pass
        response = _with_db(client, database).get("/api/meetings")
    database.close()

    assert response.status_code == 503
    assert response.json() == {"detail": MALFORMED_URL}
    assert "Senti" not in caplog.text


def test_real_pool_gives_up_with_503(client: TestClient) -> None:
    # Nothing listens on port 1, so the pool cannot connect and gives up after its timeout.
    database = Database(SecretStr("postgresql://fedrag@127.0.0.1:1/fedrag"), timeout=0.5)
    try:
        started = time.monotonic()
        response = _with_db(client, database).get("/api/meetings")
        elapsed = time.monotonic() - started
    finally:
        database.close()

    assert response.status_code == 503
    assert response.json() == {"detail": UNAVAILABLE}
    assert elapsed < 5


def test_lifespan_closes_database(make_client: MakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    closed: list[Database] = []
    monkeypatch.setattr(Database, "close", lambda self: closed.append(self))
    client = make_client()

    with client:
        assert closed == []

    assert closed == [client.app.state.database]
