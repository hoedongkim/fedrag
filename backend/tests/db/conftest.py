"""Fixtures for the Postgres layer: synthetic processed files and a throwaway test database.

The `db`-marked tests run only when TEST_DATABASE_URL points at a throwaway database whose name
contains "test", because they DROP SCHEMA fedrag CASCADE before and after each test. Never point it
at the database in DATABASE_URL. To run them against a local Postgres in Docker, from the repo root:

    make db-up
    TEST_DATABASE_URL=postgresql://fedrag:fedrag@127.0.0.1:55432/fedrag_test make test-db

In CI (the CI environment variable is set) a missing TEST_DATABASE_URL fails them instead.
"""

import hashlib
import json
import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict
from psycopg.rows import DictRow, dict_row

Row = dict[str, Any]
MakeDocument = Callable[..., Row]
WriteProcessed = Callable[[list[Row], list[Row]], Path]

FED = "https://www.federalreserve.gov"


def _meeting(meeting_date: str, start_date: str, **fields: Any) -> Row:
    compact = meeting_date.replace("-", "")
    return {
        "meeting_date": meeting_date,
        "start_date": start_date,
        "unscheduled": False,
        "has_sep": False,
        "has_press_conference": True,
        "statement_url": f"{FED}/newsevents/pressreleases/monetary{compact}a.htm",
        "minutes_url": None,
        "minutes_release_date": None,
        "source_url": f"{FED}/monetarypolicy/fomccalendars.htm",
    } | fields


def _document(
    doc_type: str, meeting_date: str, release_date: str, paragraphs: list[tuple[str | None, str]]
) -> Row:
    """A document row that satisfies the contract: text, word_count and sha256 match paragraphs."""
    text = "\n\n".join(body for _, body in paragraphs)
    compact = meeting_date.replace("-", "")
    url = (
        f"{FED}/monetarypolicy/fomcminutes{compact}.htm"
        if doc_type == "minutes"
        else f"{FED}/newsevents/pressreleases/monetary{compact}a.htm"
    )
    return {
        "doc_id": f"{doc_type}-{meeting_date}",
        "doc_type": doc_type,
        "meeting_date": meeting_date,
        "release_date": release_date,
        "release_date_source": (
            "calendar:released-text" if doc_type == "minutes" else "press-release-url"
        ),
        "page_updated": None,
        "title": f"FOMC {doc_type} {meeting_date}",
        "url": url,
        "paragraphs": [{"section": section, "text": body} for section, body in paragraphs],
        "text": text,
        "word_count": len(text.split()),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "parser_version": 1,
    }


@pytest.fixture
def make_document() -> MakeDocument:
    return _document


@pytest.fixture
def sample_meetings() -> list[Row]:
    """Four meetings, deliberately out of date order: unscheduled, SEP, ordinary, future."""
    return [
        _meeting(
            "2024-09-18",
            "2024-09-17",
            has_sep=True,
            minutes_url=f"{FED}/monetarypolicy/fomcminutes20240918.htm",
            minutes_release_date="2024-10-09",
        ),
        _meeting(
            "2020-03-15",
            "2020-03-15",
            unscheduled=True,
            source_url=f"{FED}/monetarypolicy/fomchistorical2020.htm",
        ),
        _meeting("2026-10-28", "2026-10-27", statement_url=None),
        _meeting(
            "2024-11-07",
            "2024-11-06",
            minutes_url=f"{FED}/monetarypolicy/fomcminutes20241107.htm",
            minutes_release_date="2024-11-26",
        ),
    ]


@pytest.fixture
def sample_documents() -> list[Row]:
    """Statements for the three past meetings; minutes for the two 2024 ones."""
    return [
        _document("statement", "2020-03-15", "2020-03-15", [(None, "Rates were cut sharply.")]),
        _document(
            "statement",
            "2024-09-18",
            "2024-09-18",
            [(None, "The Committee lowered the target range."), (None, "Voting for the action.")],
        ),
        _document(
            "minutes",
            "2024-09-18",
            "2024-10-09",
            [
                ("Developments in Financial Markets", "Market participants expected a cut."),
                ("Committee Policy Actions", "Members agreed to lower the range."),
            ],
        ),
        _document("statement", "2024-11-07", "2024-11-07", [(None, "A further cut was made.")]),
        _document(
            "minutes", "2024-11-07", "2024-11-26", [("Staff Review", "Inflation eased further.")]
        ),
    ]


@pytest.fixture
def write_processed(tmp_path: Path) -> WriteProcessed:
    """Return a function that writes meetings.jsonl and documents.jsonl and returns their dir."""

    def _write(meetings: list[Row], documents: list[Row]) -> Path:
        processed = tmp_path / "processed"
        processed.mkdir(exist_ok=True)
        for name, rows in (("meetings.jsonl", meetings), ("documents.jsonl", documents)):
            lines = "".join(json.dumps(row) + "\n" for row in rows)
            (processed / name).write_text(lines, encoding="utf-8")
        return processed

    return _write


@pytest.fixture(scope="session")
def test_database_url() -> str:
    url = os.environ.get("TEST_DATABASE_URL", "").strip()
    if not url:
        if os.environ.get("CI"):
            pytest.fail("TEST_DATABASE_URL is not set in CI, so the Postgres tests cannot run")
        pytest.skip("TEST_DATABASE_URL is not set, so the Postgres tests are skipped")
    dbname = str(conninfo_to_dict(url).get("dbname", ""))
    if "test" not in dbname:
        pytest.fail(
            f"TEST_DATABASE_URL names database {dbname!r}; the db tests drop schema fedrag, so "
            "they only run against a throwaway database whose name contains 'test'."
        )
    return url


@pytest.fixture
def db_conn(test_database_url: str) -> Iterator[psycopg.Connection[DictRow]]:
    """An autocommit connection to the test database; schema fedrag is dropped before and after."""
    with psycopg.connect(
        test_database_url, autocommit=True, row_factory=dict_row, connect_timeout=5
    ) as conn:
        conn.execute("DROP SCHEMA IF EXISTS fedrag CASCADE")
        yield conn
        conn.execute("DROP SCHEMA IF EXISTS fedrag CASCADE")
