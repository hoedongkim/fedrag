"""Tests for load_processed against a real Postgres."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import psycopg
import pytest

from app.core.config import Settings
from app.db.connection import apply_schema, connect
from app.db.load import load_processed

pytestmark = pytest.mark.db

Conn = psycopg.Connection[Any]
Row = dict[str, Any]
MakeDocument = Callable[..., Row]
WriteProcessed = Callable[[list[Row], list[Row]], Path]


def _table_counts(conn: Conn) -> dict[str, int]:
    return {
        table: conn.execute(f"SELECT count(*) AS n FROM fedrag.{table}").fetchone()["n"]
        for table in ("meetings", "documents")
    }


@pytest.fixture
def processed(
    write_processed: WriteProcessed, sample_meetings: list[Row], sample_documents: list[Row]
) -> Path:
    return write_processed(sample_meetings, sample_documents)


def test_load_commits_and_returns_counts(
    db_conn: Conn, test_database_url: str, processed: Path
) -> None:
    settings = Settings(_env_file=None, database_url=test_database_url)
    with connect(settings) as conn:  # an ordinary (non-autocommit) connection, as the CLI uses
        counts = load_processed(conn, processed)
        # visible to others before the caller's connection closes
        assert _table_counts(db_conn) == {"meetings": 4, "documents": 5}

    assert counts == {"meetings": 4, "documents": 5}


def test_loaded_document_matches_file(
    db_conn: Conn, processed: Path, sample_documents: list[Row]
) -> None:
    load_processed(db_conn, processed)

    stored = db_conn.execute(
        "SELECT doc_id, doc_type, meeting_date::text, release_date::text, release_date_source, "
        "page_updated::text, "
        "title, url, paragraphs, text, word_count, sha256, parser_version "
        "FROM fedrag.documents WHERE doc_id = 'minutes-2024-09-18'"
    ).fetchone()

    assert stored == next(d for d in sample_documents if d["doc_id"] == "minutes-2024-09-18")


def test_load_is_idempotent(db_conn: Conn, processed: Path) -> None:
    first = load_processed(db_conn, processed)
    second = load_processed(db_conn, processed)

    assert first == second == {"meetings": 4, "documents": 5}
    assert _table_counts(db_conn) == {"meetings": 4, "documents": 5}


def test_reload_replaces_changed_rows(
    db_conn: Conn,
    write_processed: WriteProcessed,
    sample_meetings: list[Row],
    sample_documents: list[Row],
    make_document: MakeDocument,
) -> None:
    load_processed(db_conn, write_processed(sample_meetings, sample_documents))
    revised = make_document("statement", "2024-09-18", "2024-09-18", [(None, "Revised text.")])
    sample_documents[1] = revised
    sample_meetings[0]["has_press_conference"] = False

    load_processed(db_conn, write_processed(sample_meetings, sample_documents))

    document = db_conn.execute(
        "SELECT text, sha256, word_count, paragraphs FROM fedrag.documents "
        "WHERE doc_id = 'statement-2024-09-18'"
    ).fetchone()
    meeting = db_conn.execute(
        "SELECT has_press_conference FROM fedrag.meetings WHERE meeting_date = '2024-09-18'"
    ).fetchone()
    assert document == {k: revised[k] for k in ("text", "sha256", "word_count", "paragraphs")}
    assert meeting == {"has_press_conference": False}
    assert _table_counts(db_conn) == {"meetings": 4, "documents": 5}


def test_failed_load_writes_nothing(db_conn: Conn, processed: Path) -> None:
    # A statement for 2024-09-18 stored under another doc_id makes the documents upsert fail
    # (one statement per meeting) after the meetings upsert has already run.
    apply_schema(db_conn)
    db_conn.execute(
        "INSERT INTO fedrag.meetings VALUES ('2024-09-18', '2024-09-17', false, true, true)"
    )
    db_conn.execute(
        "INSERT INTO fedrag.documents (doc_id, doc_type, meeting_date, release_date, "
        "release_date_source, title, url, text, paragraphs, word_count, sha256, parser_version) "
        "VALUES ('legacy-statement', 'statement', '2024-09-18', '2024-09-18', 'manual', 't', "
        "'https://example.org', 'x', '[]', 1, '', 0)"
    )

    with pytest.raises(psycopg.errors.UniqueViolation):
        load_processed(db_conn, processed)

    assert _table_counts(db_conn) == {"meetings": 1, "documents": 1}
