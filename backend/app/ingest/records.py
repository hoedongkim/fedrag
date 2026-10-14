"""Meeting and document records written to data/processed, and their JSONL files."""

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

PARSER_VERSION = 1

IngestDocType = Literal["statement", "minutes"]


@dataclass(frozen=True)
class Meeting:
    """One FOMC meeting: a line of meetings.jsonl and a row of fedrag.meetings."""

    meeting_date: date
    start_date: date
    unscheduled: bool
    has_sep: bool
    has_press_conference: bool
    statement_url: str | None
    minutes_url: str | None
    minutes_release_date: date | None
    source_url: str

    def to_json(self) -> dict[str, Any]:
        return {
            "meeting_date": self.meeting_date.isoformat(),
            "start_date": self.start_date.isoformat(),
            "unscheduled": self.unscheduled,
            "has_sep": self.has_sep,
            "has_press_conference": self.has_press_conference,
            "statement_url": self.statement_url,
            "minutes_url": self.minutes_url,
            "minutes_release_date": iso_or_none(self.minutes_release_date),
            "source_url": self.source_url,
        }


@dataclass(frozen=True)
class Paragraph:
    """A paragraph of document text; `section` is the nearest heading (None for statements)."""

    section: str | None
    text: str


@dataclass(frozen=True)
class Document:
    """One parsed statement or minutes page: a line of documents.jsonl."""

    doc_type: IngestDocType
    meeting_date: date
    release_date: date
    release_date_source: str
    title: str
    url: str
    paragraphs: tuple[Paragraph, ...]
    page_updated: date | None = None  # the page's "Last Update" stamp
    parser_version: int = PARSER_VERSION

    @property
    def doc_id(self) -> str:
        return f"{self.doc_type}-{self.meeting_date.isoformat()}"

    @property
    def text(self) -> str:
        return "\n\n".join(paragraph.text for paragraph in self.paragraphs)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    def to_json(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "doc_type": self.doc_type,
            "meeting_date": self.meeting_date.isoformat(),
            "release_date": self.release_date.isoformat(),
            "release_date_source": self.release_date_source,
            "page_updated": iso_or_none(self.page_updated),
            "title": self.title,
            "url": self.url,
            "paragraphs": [{"section": p.section, "text": p.text} for p in self.paragraphs],
            "text": self.text,
            "word_count": self.word_count,
            "sha256": self.sha256,
            "parser_version": self.parser_version,
        }


def iso_or_none(value: date | None) -> str | None:
    return value.isoformat() if value else None


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    """Write one JSON object per line, atomically; returns the number of lines."""
    lines = [json.dumps(row, ensure_ascii=False) + "\n" for row in rows]
    write_text_atomic(path, "".join(lines))
    return len(lines)


def write_text_atomic(path: Path, text: str) -> None:
    """Replace `path` in one step so readers never see a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    partial.write_text(text, encoding="utf-8")
    partial.replace(path)
