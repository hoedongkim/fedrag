"""Stance results and the fed funds target range in Postgres: the loader and the read queries."""

from collections import Counter
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import date
from typing import Any, Self

import psycopg
from psycopg.rows import dict_row, tuple_row
from psycopg.types.json import Jsonb
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.db.connection import apply_schema
from app.db.load import _raise_if
from app.schemas.common import Stance, StanceClassifier
from app.schemas.stance import FedFundsTarget, StanceResult
from app.stance.types import StatementPrediction

UPSERT_STANCE = """
INSERT INTO fedrag.stance_results (
    doc_id, meeting_date, classifier, stance, score, confidence, key_factors, evidence,
    model_version
)
VALUES (
    %(doc_id)s, %(meeting_date)s, %(classifier)s, %(stance)s, %(score)s, %(confidence)s,
    %(key_factors)s, %(evidence)s, %(model_version)s
)
ON CONFLICT (doc_id, classifier) DO UPDATE SET
    meeting_date = EXCLUDED.meeting_date,
    stance = EXCLUDED.stance,
    score = EXCLUDED.score,
    confidence = EXCLUDED.confidence,
    key_factors = EXCLUDED.key_factors,
    evidence = EXCLUDED.evidence,
    model_version = EXCLUDED.model_version,
    computed_at = now()
"""

UPSERT_FED_FUNDS = """
INSERT INTO fedrag.fed_funds_target (date, lower, upper)
VALUES (%(date)s, %(lower)s, %(upper)s)
ON CONFLICT (date) DO UPDATE SET lower = EXCLUDED.lower, upper = EXCLUDED.upper
"""

# The new series replaces the stored one: a change point that is no longer one would draw a
# step that never happened.
DELETE_STALE_FED_FUNDS = "DELETE FROM fedrag.fed_funds_target WHERE NOT (date = ANY(%(dates)s))"

STANCE_COLUMNS = """
SELECT s.meeting_date, s.doc_id, s.classifier, s.stance, s.score, s.confidence, s.key_factors,
       s.evidence, s.model_version, d.url
FROM fedrag.stance_results AS s
JOIN fedrag.documents AS d ON d.doc_id = s.doc_id
"""

STANCE_TIMELINE = (
    STANCE_COLUMNS
    + """
WHERE s.classifier = %(classifier)s
  AND (%(start)s::date IS NULL OR s.meeting_date >= %(start)s::date)
  AND (%(end)s::date IS NULL OR s.meeting_date <= %(end)s::date)
ORDER BY s.meeting_date
"""
)

STANCE_FOR_MEETING = (
    STANCE_COLUMNS
    + """
WHERE s.classifier = %(classifier)s AND s.meeting_date = %(meeting_date)s
ORDER BY s.doc_id
LIMIT 1
"""
)

CLASSIFIER_LOADED = """
SELECT EXISTS (SELECT 1 FROM fedrag.stance_results WHERE classifier = %(classifier)s) AS loaded
"""

# Change points up to `end`, starting from the one in force on `start` (the latest on or before
# it), so a chart of the window starts with the right level.
FED_FUNDS = """
SELECT date, lower, upper
FROM fedrag.fed_funds_target
WHERE (%(end)s::date IS NULL OR date <= %(end)s::date)
  AND (
      %(start)s::date IS NULL
      OR date >= coalesce(
          (SELECT max(date) FROM fedrag.fed_funds_target WHERE date <= %(start)s::date),
          %(start)s::date
      )
  )
ORDER BY date
"""

STANCE_TABLES = ("stance_results", "fed_funds_target")


class StanceNotLoadedError(LookupError):
    """No stance results are stored for the requested classifier (or the tables do not exist)."""


class StanceRow(BaseModel):
    """A StatementPrediction checked against the stance_results contract before it is written."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    doc_id: str = Field(min_length=1)
    meeting_date: date
    classifier: StanceClassifier
    stance: Stance
    score: float = Field(ge=-1, le=1, allow_inf_nan=False)
    confidence: float | None = Field(ge=0, le=1, allow_inf_nan=False)
    key_factors: list[str]
    evidence: list[str]
    model_version: str

    @model_validator(mode="after")
    def _doc_matches_meeting(self) -> Self:
        if self.doc_id.split("-", 1)[-1] != self.meeting_date.isoformat():
            raise ValueError(f"doc_id {self.doc_id!r} is not a document of {self.meeting_date}")
        return self


def change_points(series: Iterable[FedFundsTarget]) -> list[FedFundsTarget]:
    """The series in date order, keeping a day only when the range differs from the day before."""
    points: list[FedFundsTarget] = []
    for point in sorted(series, key=lambda p: p.date):
        if not points or (point.lower, point.upper) != (points[-1].lower, points[-1].upper):
            points.append(point)
    return points


def load_stance(
    conn: psycopg.Connection[Any],
    predictions: Iterable[StatementPrediction],
    fed_funds: Iterable[FedFundsTarget] = (),
) -> dict[str, int]:
    """Validate statement predictions and the fed funds series, then upsert both in one
    transaction (after applying schema.sql).

    Results are keyed on (doc_id, classifier), so re-running is idempotent and a new prediction
    replaces the stored one; results of other documents or classifiers are kept. A non-empty
    fed_funds series (daily values or change points) is reduced to its change points and replaces
    the stored series; an empty one leaves it as is. Every row is checked before anything is
    written, so a bad row (or a doc_id missing from fedrag.documents) loads nothing. Returns the
    number of results and change points written.
    """
    rows = _stance_rows(list(predictions))
    points = change_points(fed_funds)
    duplicates = [d for d, n in Counter(p.date for p in points).items() if n > 1]
    _raise_if([f"fed_funds: date {d} appears more than once" for d in duplicates], "bad fed funds")

    with conn.transaction(), conn.cursor() as cur:
        apply_schema(conn)
        if rows:
            cur.executemany(UPSERT_STANCE, [_stance_params(row) for row in rows])
        if points:
            cur.executemany(UPSERT_FED_FUNDS, [p.model_dump() for p in points])
            cur.execute(DELETE_STALE_FED_FUNDS, {"dates": [p.date for p in points]})
    return {"stance_results": len(rows), "fed_funds_target": len(points)}


def _stance_rows(predictions: list[StatementPrediction]) -> list[StanceRow]:
    rows: list[StanceRow] = []
    errors: list[str] = []
    for i, prediction in enumerate(predictions):
        try:
            rows.append(StanceRow.model_validate(asdict(prediction)))
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(map(str, err['loc'])) or 'row'}: {err['msg']}"
                for err in exc.errors(include_url=False)
            )
            key = f"{prediction.doc_id}, {prediction.classifier}"
            errors.append(f"prediction {i} ({key}): {problems}")
    keys = Counter((row.doc_id, row.classifier) for row in rows)
    errors += [f"{doc} / {clf} appears {n} times" for (doc, clf), n in keys.items() if n > 1]
    _raise_if(errors, "stance predictions are invalid")
    return rows


def _stance_params(row: StanceRow) -> dict[str, Any]:
    return row.model_dump() | {
        "key_factors": Jsonb(row.key_factors),
        "evidence": Jsonb(row.evidence),
    }


def stance_timeline(
    conn: psycopg.Connection[Any],
    classifier: StanceClassifier,
    start: date | None = None,
    end: date | None = None,
) -> list[StanceResult]:
    """The classifier's results for meetings between start and end (inclusive), in date order.

    Raises StanceNotLoadedError when nothing is stored for the classifier, so an empty list means
    only that no meeting falls in the window.
    """
    params = {"classifier": classifier, "start": start, "end": end}
    with _stance_tables(conn), conn.cursor(row_factory=dict_row) as cur:
        rows = cur.execute(STANCE_TIMELINE, params).fetchall()
        if not rows:
            _require_loaded(cur, classifier)
    return [StanceResult.model_validate(row) for row in rows]


def stance_for_meeting(
    conn: psycopg.Connection[Any], meeting_date: date, classifier: StanceClassifier
) -> StanceResult | None:
    """The classifier's result for one meeting's statement; None when there is none.

    Raises StanceNotLoadedError when nothing is stored for the classifier at all.
    """
    params = {"classifier": classifier, "meeting_date": meeting_date}
    with _stance_tables(conn), conn.cursor(row_factory=dict_row) as cur:
        row = cur.execute(STANCE_FOR_MEETING, params).fetchone()
        if row is None:
            _require_loaded(cur, classifier)
    return None if row is None else StanceResult.model_validate(row)


def fed_funds_window(
    conn: psycopg.Connection[Any], start: date | None = None, end: date | None = None
) -> list[FedFundsTarget]:
    """Change points of the target range up to `end`, from the one in force on `start`."""
    with _stance_tables(conn), conn.cursor(row_factory=dict_row) as cur:
        rows = cur.execute(FED_FUNDS, {"start": start, "end": end}).fetchall()
    return [FedFundsTarget.model_validate(row) for row in rows]


def _require_loaded(cur: psycopg.Cursor[Any], classifier: StanceClassifier) -> None:
    if not cur.execute(CLASSIFIER_LOADED, {"classifier": classifier}).fetchone()["loaded"]:
        raise StanceNotLoadedError(classifier)


@contextmanager
def _stance_tables(conn: psycopg.Connection[Any]) -> Iterator[None]:
    """Turn "relation fedrag.stance_results does not exist" into StanceNotLoadedError, unless the
    documents table is missing too: then nothing is loaded, and the UndefinedTable stands (the
    API's "run make db-load")."""
    try:
        yield
    except psycopg.errors.UndefinedTable as exc:
        if any(f"fedrag.{table}" in str(exc) for table in STANCE_TABLES) and _has_documents(conn):
            raise StanceNotLoadedError("the stance tables do not exist") from None
        raise


def _has_documents(conn: psycopg.Connection[Any]) -> bool:
    if conn.info.transaction_status == psycopg.pq.TransactionStatus.INERROR:
        return True  # cannot ask inside a failed transaction; assume only the stance tables lack
    with conn.cursor(row_factory=tuple_row) as cur:
        row = cur.execute("SELECT to_regclass('fedrag.documents') IS NOT NULL").fetchone()
    return bool(row and row[0])
