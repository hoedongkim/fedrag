"""Tests for schema.sql and its constraints against a real Postgres."""

from datetime import date
from typing import Any

import psycopg
import pytest
from psycopg import errors, sql
from psycopg.types.json import Jsonb

from app.db.connection import apply_schema

pytestmark = pytest.mark.db

Conn = psycopg.Connection[Any]

MEETING = {
    "meeting_date": date(2024, 9, 18),
    "start_date": date(2024, 9, 17),
    "unscheduled": False,
    "has_sep": True,
    "has_press_conference": True,
}
DOCUMENT = {
    "doc_id": "statement-2024-09-18",
    "doc_type": "statement",
    "meeting_date": date(2024, 9, 18),
    "release_date": date(2024, 9, 18),
    "release_date_source": "press-release-url",
    "title": "FOMC statement",
    "url": "https://www.federalreserve.gov/newsevents/pressreleases/monetary20240918a.htm",
    "text": "The Committee lowered the target range.",
    "paragraphs": Jsonb([{"section": None, "text": "The Committee lowered the target range."}]),
    "word_count": 6,
    "sha256": "0" * 64,
    "parser_version": 1,
}


def _insert(conn: Conn, table: str, row: dict[str, Any]) -> None:
    query = sql.SQL("INSERT INTO fedrag.{} ({}) VALUES ({})").format(
        sql.Identifier(table),
        sql.SQL(", ").join(map(sql.Identifier, row)),
        sql.SQL(", ").join([sql.Placeholder()] * len(row)),
    )
    conn.execute(query, list(row.values()))


@pytest.fixture
def schema(db_conn: Conn) -> Conn:
    apply_schema(db_conn)
    return db_conn


def test_schema_applies_twice(db_conn: Conn) -> None:
    apply_schema(db_conn)
    apply_schema(db_conn)

    tables = db_conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'fedrag'"
    ).fetchall()
    indexes = db_conn.execute("SELECT indexname FROM pg_indexes WHERE schemaname = 'fedrag'")
    assert {row["table_name"] for row in tables} == {
        "meetings",
        "documents",
        "stance_results",
        "fed_funds_target",
        "statement_comparisons",
    }
    names = {row["indexname"] for row in indexes}
    assert {
        "documents_release_date_idx",
        "stance_results_classifier_date_idx",
        "statement_comparisons_pair_idx",
    } <= names


def test_valid_rows_insert(schema: Conn) -> None:
    _insert(schema, "meetings", MEETING)
    _insert(schema, "documents", DOCUMENT)

    row = schema.execute("SELECT paragraphs, loaded_at FROM fedrag.documents").fetchone()
    assert row is not None
    assert row["paragraphs"] == [
        {"section": None, "text": "The Committee lowered the target range."}
    ]
    assert row["loaded_at"] is not None


@pytest.mark.parametrize(
    ("meeting", "error", "constraint"),
    [
        ({"start_date": date(2024, 9, 19)}, errors.CheckViolation, "meetings_start_not_after_end"),
        ({"has_sep": None}, errors.NotNullViolation, None),
    ],
    ids=["start-after-meeting", "null-flag"],
)
def test_bad_meeting_is_rejected(
    schema: Conn, meeting: dict[str, Any], error: type[psycopg.Error], constraint: str | None
) -> None:
    with pytest.raises(error) as raised:
        _insert(schema, "meetings", MEETING | meeting)

    assert raised.value.diag.constraint_name == constraint


@pytest.mark.parametrize(
    ("document", "error", "constraint"),
    [
        (
            {"release_date": date(2024, 9, 17)},
            errors.CheckViolation,
            "documents_released_after_meeting",
        ),
        ({"doc_type": "memo"}, errors.CheckViolation, "documents_doc_type_known"),
        (
            {"meeting_date": date(2024, 7, 31), "release_date": date(2024, 7, 31)},
            errors.ForeignKeyViolation,
            "documents_meeting_date_fkey",
        ),
        ({}, errors.UniqueViolation, "documents_one_per_meeting"),
        ({"doc_id": "statement-2024-09-18"}, errors.UniqueViolation, "documents_pkey"),
        ({"text": None}, errors.NotNullViolation, None),
    ],
    ids=[
        "released-before-meeting",
        "unknown-doc-type",
        "unknown-meeting",
        "second-statement-for-meeting",
        "duplicate-doc-id",
        "null-text",
    ],
)
def test_bad_document_is_rejected(
    schema: Conn, document: dict[str, Any], error: type[psycopg.Error], constraint: str | None
) -> None:
    _insert(schema, "meetings", MEETING)
    _insert(schema, "documents", DOCUMENT)

    with pytest.raises(error) as raised:
        _insert(schema, "documents", DOCUMENT | {"doc_id": "another-doc"} | document)

    assert raised.value.diag.constraint_name == constraint


COMPARISON = {
    "cache_key": "0" * 64,
    "base_doc_id": "statement-2024-09-18",
    "target_doc_id": "minutes-2024-09-18",
    "model": "gpt-5.4-mini-2026-03-17",
    "prompt_version": "compare-v1.00000000",
    "analysis": Jsonb({"summary": "...", "key_changes": []}),
    "usage": Jsonb({"calls": 1}),
}


@pytest.mark.parametrize(
    ("comparison", "error", "constraint"),
    [
        (
            {"target_doc_id": "statement-2024-09-18"},
            errors.CheckViolation,
            "statement_comparisons_two_documents",
        ),
        (
            {"target_doc_id": "statement-2024-11-07"},
            errors.ForeignKeyViolation,
            "statement_comparisons_target_doc_id_fkey",
        ),
        (
            {"analysis": Jsonb(["not", "an", "object"])},
            errors.CheckViolation,
            "statement_comparisons_analysis_object",
        ),
        ({"usage": Jsonb(None)}, errors.CheckViolation, "statement_comparisons_usage_object"),
        ({"cache_key": "0" * 64}, errors.UniqueViolation, "statement_comparisons_pkey"),
    ],
    ids=["same-document", "unknown-document", "analysis-array", "usage-null", "duplicate-key"],
)
def test_bad_comparison_is_rejected(
    schema: Conn, comparison: dict[str, Any], error: type[psycopg.Error], constraint: str
) -> None:
    _insert(schema, "meetings", MEETING)
    _insert(schema, "documents", DOCUMENT)
    minutes = DOCUMENT | {"doc_id": "minutes-2024-09-18", "doc_type": "minutes"}
    _insert(schema, "documents", minutes | {"release_date": date(2024, 10, 9)})
    _insert(schema, "statement_comparisons", COMPARISON)

    with pytest.raises(error) as raised:
        _insert(schema, "statement_comparisons", COMPARISON | {"cache_key": "1" * 64} | comparison)

    assert raised.value.diag.constraint_name == constraint
