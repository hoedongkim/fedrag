"""Statement Comparison in Postgres: statements by meeting date, their stored stance, and the
cached analyses in fedrag.statement_comparisons."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.compare.types import StatementText
from app.db.connection import apply_schema
from app.schemas.common import SourceDocument, StanceClassifier
from app.schemas.compare import ComparisonAnalysis, StanceSnapshot

STATEMENT_COLUMNS = """
SELECT doc_id, doc_type, meeting_date, release_date, page_updated, title, url, paragraphs
FROM fedrag.documents
WHERE doc_type = 'statement'
"""

STATEMENTS_ON = STATEMENT_COLUMNS + "AND meeting_date = ANY(%(dates)s::date[])"
ALL_STATEMENTS = STATEMENT_COLUMNS + "ORDER BY meeting_date"

NEAREST_STATEMENTS = """
SELECT
    (SELECT max(meeting_date) FROM fedrag.documents
     WHERE doc_type = 'statement' AND meeting_date < %(day)s::date) AS before,
    (SELECT min(meeting_date) FROM fedrag.documents
     WHERE doc_type = 'statement' AND meeting_date > %(day)s::date) AS after
"""

TABLE_EXISTS = "SELECT to_regclass(%(name)s) IS NOT NULL AS present"
# Held until the transaction ends: one writer at a time creates the missing table.
SCHEMA_LOCK = "SELECT pg_advisory_xact_lock(hashtext('fedrag.schema'))"

STANCES = """
SELECT doc_id, classifier, stance, score
FROM fedrag.stance_results
WHERE classifier = %(classifier)s AND doc_id = ANY(%(doc_ids)s::text[])
"""

READ_COMPARISON = """
SELECT cache_key, base_doc_id, target_doc_id, model, prompt_version, analysis, usage, created_at
FROM fedrag.statement_comparisons
WHERE cache_key = %(cache_key)s
"""

# A key fixes the doc ids, texts, diff, prompt and model, so only the analysis can differ: a new
# one (say, after a stored one stopped validating) replaces it.
UPSERT_COMPARISON = """
INSERT INTO fedrag.statement_comparisons (
    cache_key, base_doc_id, target_doc_id, model, prompt_version, analysis, usage
)
VALUES (
    %(cache_key)s, %(base_doc_id)s, %(target_doc_id)s, %(model)s, %(prompt_version)s,
    %(analysis)s, %(usage)s
)
ON CONFLICT (cache_key) DO UPDATE SET
    analysis = EXCLUDED.analysis,
    usage = EXCLUDED.usage,
    created_at = now()
"""


@dataclass(frozen=True)
class StatementRecord:
    """A stored FOMC statement: its API description and its text."""

    source: SourceDocument
    statement: StatementText


@dataclass(frozen=True)
class StoredComparison:
    """A row of fedrag.statement_comparisons; analysis is the stored JSON, not yet validated."""

    cache_key: str
    base_doc_id: str
    target_doc_id: str
    model: str
    prompt_version: str
    analysis: dict[str, Any]
    usage: dict[str, Any]
    created_at: datetime


def _record(row: dict[str, Any]) -> StatementRecord:
    paragraphs = tuple(paragraph["text"] for paragraph in row["paragraphs"])
    statement = StatementText(row["doc_id"], row["meeting_date"], row["release_date"], paragraphs)
    return StatementRecord(SourceDocument.model_validate(row), statement)


def statements_on(
    conn: psycopg.Connection[Any], dates: Sequence[date]
) -> dict[date, StatementRecord]:
    """The statements of the meetings on `dates`, by meeting date; a date without one is left
    out."""
    with conn.cursor(row_factory=dict_row) as cur:
        rows = cur.execute(STATEMENTS_ON, {"dates": list(dates)}).fetchall()
    return {row["meeting_date"]: _record(row) for row in rows}


def all_statements(conn: psycopg.Connection[Any]) -> list[StatementRecord]:
    """Every stored statement, oldest meeting first."""
    with conn.cursor(row_factory=dict_row) as cur:
        return [_record(row) for row in cur.execute(ALL_STATEMENTS).fetchall()]


def nearest_statement_dates(
    conn: psycopg.Connection[Any], day: date
) -> tuple[date | None, date | None]:
    """Meeting dates of the closest statements before and after `day` (None where there is
    none)."""
    with conn.cursor(row_factory=dict_row) as cur:
        row = cur.execute(NEAREST_STATEMENTS, {"day": day}).fetchone()
    return (row["before"], row["after"]) if row else (None, None)


def stances_of(
    conn: psycopg.Connection[Any], doc_ids: Sequence[str], classifier: StanceClassifier
) -> dict[str, StanceSnapshot]:
    """The classifier's stored stance of each document, by doc_id; empty while the stance table
    does not exist."""
    if not _table_exists(conn, "fedrag.stance_results"):
        return {}
    params = {"classifier": classifier, "doc_ids": list(doc_ids)}
    with conn.cursor(row_factory=dict_row) as cur:
        rows = cur.execute(STANCES, params).fetchall()
    return {row["doc_id"]: StanceSnapshot.model_validate(row) for row in rows}


def read_comparison(conn: psycopg.Connection[Any], cache_key: str) -> StoredComparison | None:
    """The stored analysis under `cache_key`; None when there is none (or no table yet)."""
    if not _table_exists(conn, "fedrag.statement_comparisons"):
        return None
    with conn.cursor(row_factory=dict_row) as cur:
        row = cur.execute(READ_COMPARISON, {"cache_key": cache_key}).fetchone()
    return None if row is None else StoredComparison(**row)


def save_comparison(
    conn: psycopg.Connection[Any],
    *,
    cache_key: str,
    base_doc_id: str,
    target_doc_id: str,
    model: str,
    prompt_version: str,
    analysis: ComparisonAnalysis,
    usage: dict[str, Any],
) -> None:
    """Store an analysis under `cache_key` (replacing one stored under it), in one transaction
    that first applies schema.sql while the table is missing, so the first write creates it.

    Only while it is missing: schema.sql's CREATE INDEX IF NOT EXISTS takes a SHARE lock on the
    table even when the index exists, and two writers that each hold one before inserting
    deadlock (seen with 4 concurrent writers on Neon). Concurrent first saves on a database
    without the table would deadlock the same way, so creating it is serialized by a
    transaction-level advisory lock, after which the table is checked again.
    """
    params = {
        "cache_key": cache_key,
        "base_doc_id": base_doc_id,
        "target_doc_id": target_doc_id,
        "model": model,
        "prompt_version": prompt_version,
        "analysis": Jsonb(analysis.model_dump(mode="json")),
        "usage": Jsonb(usage),
    }
    with conn.transaction():
        if not _table_exists(conn, "fedrag.statement_comparisons"):
            conn.execute(SCHEMA_LOCK)
            if not _table_exists(conn, "fedrag.statement_comparisons"):
                apply_schema(conn)
        conn.execute(UPSERT_COMPARISON, params)


def _table_exists(conn: psycopg.Connection[Any], name: str) -> bool:
    with conn.cursor(row_factory=dict_row) as cur:
        row = cur.execute(TABLE_EXISTS, {"name": name}).fetchone()
    return bool(row and row["present"])
