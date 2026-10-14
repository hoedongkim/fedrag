"""Tests for the processed-data records: derived fields, JSON shape and JSONL files."""

import hashlib
import json
from datetime import date
from pathlib import Path

from app.ingest.records import (
    PARSER_VERSION,
    Document,
    Meeting,
    Paragraph,
    write_jsonl,
)

DOCUMENT_KEYS = [
    "doc_id",
    "doc_type",
    "meeting_date",
    "release_date",
    "release_date_source",
    "page_updated",
    "title",
    "url",
    "paragraphs",
    "text",
    "word_count",
    "sha256",
    "parser_version",
]
MEETING_KEYS = [
    "meeting_date",
    "start_date",
    "unscheduled",
    "has_sep",
    "has_press_conference",
    "statement_url",
    "minutes_url",
    "minutes_release_date",
    "source_url",
]


def make_document() -> Document:
    return Document(
        doc_type="minutes",
        meeting_date=date(2025, 9, 17),
        release_date=date(2025, 10, 8),
        release_date_source="calendar:released-text",
        title="Minutes of the FOMC Meeting, September 16-17, 2025",
        url="https://www.federalreserve.gov/monetarypolicy/fomcminutes20250917.htm",
        paragraphs=(
            Paragraph(section="Preamble", text="A joint meeting was held."),
            Paragraph(section="Staff Economic Outlook", text="The staff forecast was revised."),
        ),
    )


def test_document_derived_fields() -> None:
    doc = make_document()
    text = "A joint meeting was held.\n\nThe staff forecast was revised."

    assert doc.doc_id == "minutes-2025-09-17"
    assert doc.text == text
    assert doc.word_count == 10
    assert doc.sha256 == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert doc.parser_version == PARSER_VERSION


def test_document_json_matches_the_contract() -> None:
    row = make_document().to_json()

    assert list(row) == DOCUMENT_KEYS
    assert row["meeting_date"] == "2025-09-17" and row["release_date"] == "2025-10-08"
    assert row["paragraphs"][0] == {"section": "Preamble", "text": "A joint meeting was held."}
    json.dumps(row)  # serializable as is


def test_meeting_json_matches_the_contract() -> None:
    meeting = Meeting(
        meeting_date=date(2026, 10, 28),
        start_date=date(2026, 10, 27),
        unscheduled=False,
        has_sep=False,
        has_press_conference=True,
        statement_url="https://www.federalreserve.gov/newsevents/pressreleases/monetary20261028a.htm",
        minutes_url=None,
        minutes_release_date=None,
        source_url="https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
    )

    row = meeting.to_json()

    assert list(row) == MEETING_KEYS
    assert row["start_date"] == "2026-10-27"
    assert row["minutes_url"] is None and row["minutes_release_date"] is None


def test_jsonl_round_trip_is_byte_stable(tmp_path: Path) -> None:
    path = tmp_path / "processed" / "documents.jsonl"
    rows = [make_document().to_json(), make_document().to_json()]

    assert write_jsonl(path, rows) == 2
    first = path.read_bytes()
    write_jsonl(path, [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()])

    assert path.read_bytes() == first
    assert first.count(b"\n") == 2
    assert not list(path.parent.glob("*.part"))
