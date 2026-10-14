"""Load data/processed/*.jsonl into Postgres: validate every row, then upsert in one transaction."""

import hashlib
from collections import Counter
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import Any, Self, TypeVar

import psycopg
from psycopg.types.json import Jsonb
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.db.connection import apply_schema
from app.schemas.common import DocType

MEETINGS_FILE = "meetings.jsonl"
DOCUMENTS_FILE = "documents.jsonl"
MAX_REPORTED_ERRORS = 20

UPSERT_MEETING = """
INSERT INTO fedrag.meetings (meeting_date, start_date, unscheduled, has_sep, has_press_conference)
VALUES (%(meeting_date)s, %(start_date)s, %(unscheduled)s, %(has_sep)s, %(has_press_conference)s)
ON CONFLICT (meeting_date) DO UPDATE SET
    start_date = EXCLUDED.start_date,
    unscheduled = EXCLUDED.unscheduled,
    has_sep = EXCLUDED.has_sep,
    has_press_conference = EXCLUDED.has_press_conference
"""

UPSERT_DOCUMENT = """
INSERT INTO fedrag.documents (
    doc_id, doc_type, meeting_date, release_date, release_date_source, page_updated, title, url,
    text, paragraphs, word_count, sha256, parser_version
)
VALUES (
    %(doc_id)s, %(doc_type)s, %(meeting_date)s, %(release_date)s, %(release_date_source)s,
    %(page_updated)s, %(title)s, %(url)s, %(text)s, %(paragraphs)s, %(word_count)s, %(sha256)s,
    %(parser_version)s
)
ON CONFLICT (doc_id) DO UPDATE SET
    release_date = EXCLUDED.release_date,
    release_date_source = EXCLUDED.release_date_source,
    page_updated = EXCLUDED.page_updated,
    title = EXCLUDED.title,
    url = EXCLUDED.url,
    text = EXCLUDED.text,
    paragraphs = EXCLUDED.paragraphs,
    word_count = EXCLUDED.word_count,
    sha256 = EXCLUDED.sha256,
    parser_version = EXCLUDED.parser_version,
    loaded_at = now()
"""


class LoadError(ValueError):
    """A processed file is missing, or holds rows that break the data contract."""


class _Row(BaseModel):
    """A JSONL row: exact types (no coercion), no unknown keys, every contract key present."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class MeetingRow(_Row):
    """One line of meetings.jsonl. Only the meeting facts are stored; the URLs are provenance."""

    meeting_date: date
    start_date: date
    unscheduled: bool
    has_sep: bool
    has_press_conference: bool
    statement_url: str | None
    minutes_url: str | None
    minutes_release_date: date | None
    source_url: str

    @model_validator(mode="after")
    def _dates_in_order(self) -> Self:
        if self.start_date > self.meeting_date:
            raise ValueError("start_date is after meeting_date")
        if self.minutes_release_date and self.minutes_release_date < self.meeting_date:
            raise ValueError("minutes_release_date is before meeting_date")
        return self


class ParagraphRow(_Row):
    """One paragraph of a document, in document order."""

    section: str | None
    text: str = Field(min_length=1)


class DocumentRow(_Row):
    """One line of documents.jsonl."""

    doc_id: str
    doc_type: DocType
    meeting_date: date
    release_date: date
    release_date_source: str = Field(min_length=1)
    page_updated: date | None
    title: str = Field(min_length=1)
    url: str = Field(pattern=r"^https://")
    paragraphs: list[ParagraphRow] = Field(min_length=1)
    text: str
    word_count: int = Field(ge=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parser_version: int = Field(ge=0)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        expected_id = f"{self.doc_type}-{self.meeting_date.isoformat()}"
        if self.doc_id != expected_id:
            raise ValueError(f"doc_id should be {expected_id!r}")
        if self.release_date < self.meeting_date:
            raise ValueError("release_date is before meeting_date")
        if self.text != "\n\n".join(p.text for p in self.paragraphs):
            raise ValueError("text is not the paragraph texts joined with blank lines")
        if self.sha256 != hashlib.sha256(self.text.encode("utf-8")).hexdigest():
            raise ValueError("sha256 does not match text")
        if self.word_count != len(self.text.split()):
            raise ValueError("word_count does not match text")
        return self


RowT = TypeVar("RowT", bound=_Row)


def read_jsonl(path: Path, model: type[RowT]) -> list[RowT]:
    """Parse and validate every line of a JSONL file; raise LoadError listing all bad lines."""
    if not path.is_file():
        raise LoadError(
            f"{path} not found: run `make ingest` (python -m app.ingest all --no-load) first."
        )
    rows: list[RowT] = []
    errors: list[str] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(model.model_validate_json(line))
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(map(str, err['loc'])) or 'row'}: {err['msg']}"
                for err in exc.errors(include_url=False)
            )
            errors.append(f"{path.name} line {line_no}: {problems}")
    _raise_if(errors, f"{path.name} has invalid rows")
    return rows


def load_processed(conn: psycopg.Connection[Any], processed_dir: Path) -> dict[str, int]:
    """Validate meetings.jsonl and documents.jsonl, then apply schema.sql and upsert both.

    Every row is checked before anything is written, and the writes happen in one transaction, so
    a load lands completely or not at all (it commits on exit unless the caller already has a
    transaction open). Rows are keyed on meeting_date and doc_id: re-running is idempotent, and a
    changed document replaces the stored one. Returns the number of meetings and documents written.
    """
    meetings = read_jsonl(processed_dir / MEETINGS_FILE, MeetingRow)
    documents = read_jsonl(processed_dir / DOCUMENTS_FILE, DocumentRow)
    _check_together(meetings, documents)

    with conn.transaction(), conn.cursor() as cur:
        apply_schema(conn)
        cur.executemany(UPSERT_MEETING, [_meeting_params(m) for m in meetings])
        cur.executemany(UPSERT_DOCUMENT, [_document_params(d) for d in documents])
    return {"meetings": len(meetings), "documents": len(documents)}


def _check_together(meetings: list[MeetingRow], documents: list[DocumentRow]) -> None:
    """Checks that span rows: no duplicate keys, and every document belongs to a known meeting."""
    if not meetings:
        raise LoadError(f"{MEETINGS_FILE} has no rows: nothing to load.")
    known = {m.meeting_date for m in meetings}
    errors = [
        *_duplicates(MEETINGS_FILE, "meeting_date", (m.meeting_date for m in meetings)),
        *_duplicates(DOCUMENTS_FILE, "doc_id", (d.doc_id for d in documents)),
        *(
            f"{DOCUMENTS_FILE}: {d.doc_id} has no meeting {d.meeting_date} in {MEETINGS_FILE}"
            for d in documents
            if d.meeting_date not in known
        ),
    ]
    _raise_if(errors, "processed files are inconsistent")


def _duplicates(file: str, key: str, values: Iterable[object]) -> list[str]:
    return [f"{file}: {key} {v} appears {n} times" for v, n in Counter(values).items() if n > 1]


def _raise_if(errors: list[str], what: str) -> None:
    if errors:
        shown = errors[:MAX_REPORTED_ERRORS]
        more = len(errors) - len(shown)
        lines = shown + ([f"... and {more} more"] if more else [])
        raise LoadError(f"{what} ({len(errors)} problems):\n  " + "\n  ".join(lines))


def _meeting_params(row: MeetingRow) -> dict[str, Any]:
    return row.model_dump(
        include={"meeting_date", "start_date", "unscheduled", "has_sep", "has_press_conference"}
    )


def _document_params(row: DocumentRow) -> dict[str, Any]:
    return row.model_dump(exclude={"paragraphs"}) | {
        "paragraphs": Jsonb([p.model_dump() for p in row.paragraphs])
    }
