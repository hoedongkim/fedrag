"""Ingest steps (calendar, fetch, parse) and the data-quality report written next to the data."""

import json
import re
import statistics
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from app.ingest.calendar import (
    CALENDAR_URL,
    HISTORICAL_URL,
    CalendarEntry,
    parse_calendar_page,
    parse_historical_page,
)
from app.ingest.http import CacheMissError, FedClient
from app.ingest.parse import (
    ATTENDANCE,
    SIGNATURE_NOTE,
    ParsedPage,
    parse_minutes,
    parse_statement,
)
from app.ingest.records import (
    Document,
    IngestDocType,
    Meeting,
    iso_or_none,
    write_jsonl,
    write_text_atomic,
)

GENERIC_STATEMENT_HEADLINE = "Federal Reserve issues FOMC statement"
STATEMENT_URL_DATE_RE = re.compile(r"monetary(\d{4})(\d{2})(\d{2})[a-z]")
MINUTES_NOTE_DATE_RE = re.compile(r"minutes of (?:the )?([A-Z][a-z]+) (\d{1,2})\b")
SCHEDULED_MEETINGS_PER_YEAR = 8
MINUTES_LAG_BAND = (14, 35)
WORD_FLOOR = {"statement": 75, "minutes": 3000}
# Page furniture that must never survive parsing, as it appears on the page: whole paragraphs,
# link labels at the start of a paragraph, or "PDF | HTML" link pairs. Matched case-sensitively
# per paragraph, so ordinary prose ("Share prices rose", "the Implementation Note issued in
# December") does not trip them.
CHROME_PATTERNS = {
    "release-time line": r"^For (immediate )?release\b",
    "media contact": r"For media inquiries",
    "implementation note link": r"^Implementation Note issued",
    "last update stamp": r"Last Update:",
    "footnote back-link": r"Return to text",
    "back-to-top link": r"Back to Top",
    "obfuscated email": r"email protected|\[email",
    "share widget": r"^Share$",
    "javascript notice": r"enable JavaScript",
    "pdf/html link pair": r"\bPDF \| HTML\b|\bHTML \| PDF\b|^(PDF|HTML)$",
    "related content box": r"^Related Content",
    "site banner": r"An official website of the United States",
    "social links": r"twitter\.com|facebook\.com|linkedin\.com|Stay Connected",
    "signature rule": r"_{4,}",
}


@dataclass(frozen=True)
class IngestPaths:
    """Where the pipeline reads and writes: data/raw for HTML, data/processed for outputs."""

    raw_dir: Path
    processed_dir: Path

    @classmethod
    def under(cls, data_dir: Path) -> "IngestPaths":
        return cls(raw_dir=data_dir / "raw", processed_dir=data_dir / "processed")

    @property
    def meetings_file(self) -> Path:
        return self.processed_dir / "meetings.jsonl"

    @property
    def documents_file(self) -> Path:
        return self.processed_dir / "documents.jsonl"

    @property
    def report_file(self) -> Path:
        return self.processed_dir / "report.json"


@dataclass
class Selection:
    """Calendar entries split into meetings to ingest and entries skipped, with reasons."""

    meetings: list[Meeting] = field(default_factory=list)
    entries: dict[date, CalendarEntry] = field(default_factory=dict)
    skipped: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# --- calendar --------------------------------------------------------------------------------


def collect_calendar(client: FedClient, start_year: int, end_year: int) -> list[CalendarEntry]:
    """Entries for the years: fomccalendars.htm, plus fomchistoricalYYYY.htm for older years.

    fomccalendars.htm is re-downloaded once per run (once per client), so statements and minutes
    linked since the last run are seen; an offline client reads the copy just refreshed. The
    fomchistoricalYYYY.htm pages are archives: they are downloaded only when missing (or with
    --refresh).
    """
    calendar_page = client.get(CALENDAR_URL, "calendar", refresh=True)
    entries = parse_calendar_page(calendar_page, CALENDAR_URL)
    covered = {entry.end_date.year for entry in entries}
    for year in range(start_year, end_year + 1):
        if year not in covered:
            url = HISTORICAL_URL.format(year=year)
            entries += parse_historical_page(client.get(url, "calendar"), url)
    selected = [e for e in entries if start_year <= e.end_date.year <= end_year]
    return sorted(selected, key=lambda e: (e.end_date, e.start_date))


def select_meetings(entries: Iterable[CalendarEntry], today: date) -> Selection:
    """Meetings held by `today` that issued a statement or were scheduled; the rest are skipped."""
    selection = Selection()
    for entry in entries:
        reason = _skip_reason(entry, today)
        if reason:
            selection.skipped.append(
                {"date": entry.end_date.isoformat(), "label": entry.label, "reason": reason}
            )
            continue
        if entry.end_date in selection.entries:
            selection.warnings.append(
                f"two calendar entries end on {entry.end_date}; kept the first"
            )
            continue
        if entry.minutes_url and entry.minutes_released is None:
            selection.warnings.append(
                f"{entry.end_date}: minutes link without a '(Released ...)' date; not ingested"
            )
        if entry.kind == "meeting" and not entry.statement_url:
            selection.warnings.append(
                f"{entry.end_date}: scheduled meeting without a statement link"
            )
        selection.entries[entry.end_date] = entry
        selection.meetings.append(
            Meeting(
                meeting_date=entry.end_date,
                start_date=entry.start_date,
                unscheduled=entry.unscheduled,
                has_sep=entry.has_sep,
                has_press_conference=entry.has_press_conference,
                statement_url=entry.statement_url,
                minutes_url=entry.minutes_url,
                minutes_release_date=entry.minutes_released if entry.minutes_url else None,
                source_url=entry.source_url,
            )
        )
    return selection


def _skip_reason(entry: CalendarEntry, today: date) -> str | None:
    if entry.end_date > today:
        return "upcoming meeting"
    if entry.kind == "cancelled":
        return "cancelled meeting"
    if entry.statement_url is None and entry.kind == "notation_vote":
        return "notation vote without a policy statement"
    if entry.statement_url is None and entry.kind == "unscheduled":
        return "unscheduled meeting without a statement"
    return None


# --- fetch -----------------------------------------------------------------------------------


def fetch_documents(
    client: FedClient, meetings: Iterable[Meeting], log: Callable[[str], None] = print
) -> dict[str, int]:
    """Download (or confirm cached) every statement and minutes page; returns counts."""
    counts = Counter[str]()
    for meeting in meetings:
        for url, category in (
            (meeting.statement_url, "statements"),
            (meeting.minutes_url, "minutes"),
        ):
            if url is None:
                continue
            cached = client.is_cached(url, category) and not client.refresh
            client.get(url, category)
            counts["cached" if cached else "downloaded"] += 1
            if not cached:
                log(f"  fetched {url}")
    return dict(counts)


# --- parse -----------------------------------------------------------------------------------


@dataclass
class ParseResult:
    """Documents built by the parse step, with notes on deliberate drops and any warnings.

    `revised` lists documents whose page carries a 'Last Update' stamp later than their
    release_date: the stored text is the Fed's current version, which may differ from the text
    first released.
    """

    documents: list[Document] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    signatures_dropped: int = 0
    revised: list[dict[str, str]] = field(default_factory=list)


def build_documents(client: FedClient, selection: Selection) -> ParseResult:
    """Parse every cached statement and minutes page of the selected meetings."""
    result = ParseResult()
    for meeting in selection.meetings:
        entry = selection.entries[meeting.meeting_date]
        if meeting.statement_url:
            _add_document(result, client, meeting, entry, "statement")
        if meeting.minutes_url and meeting.minutes_release_date:
            _add_document(result, client, meeting, entry, "minutes")
    result.documents.sort(key=lambda doc: (doc.meeting_date, doc.doc_type))
    return result


def _add_document(
    result: ParseResult,
    client: FedClient,
    meeting: Meeting,
    entry: CalendarEntry,
    doc_type: IngestDocType,
) -> None:
    is_statement = doc_type == "statement"
    url = meeting.statement_url if is_statement else meeting.minutes_url
    assert url is not None
    label = f"{doc_type}-{meeting.meeting_date}"
    try:
        html = client.get(url, "statements" if is_statement else "minutes")
        page = parse_statement(html) if is_statement else parse_minutes(html)
    except CacheMissError:
        result.warnings.append(f"{label}: page not downloaded yet (run the fetch step)")
        return
    except ValueError as exc:
        result.warnings.append(f"{label}: could not parse {url}: {exc}")
        return
    doc = _statement_doc(meeting, page, entry) if is_statement else _minutes_doc(meeting, page)
    for note in page.notes:
        if note.startswith(SIGNATURE_NOTE):
            result.signatures_dropped += 1
        else:
            result.notes.append(f"{label}: {note}")
    result.warnings.extend(f"{label}: {warning}" for warning in page.warnings)
    if not page.paragraphs:
        result.warnings.append(f"{label}: no text extracted from {url}")
    elif doc is None:
        result.warnings.append(f"{label}: no authoritative release date; not ingested")
    elif doc.release_date < doc.meeting_date:
        result.warnings.append(f"{label}: release date {doc.release_date} precedes the meeting")
    else:
        result.documents.append(doc)
        if page.page_updated and page.page_updated > doc.release_date:
            result.revised.append(
                {
                    "doc_id": doc.doc_id,
                    "release_date": doc.release_date.isoformat(),
                    "page_updated": page.page_updated.isoformat(),
                }
            )


def statement_release(
    page_date: date | None, calendar_date: date | None, url: str
) -> tuple[date, str, list[str]] | None:
    """A statement's release date, where it came from, and any source that disagrees with it.

    Sources in order of preference: the date printed on the press release, the "(Released ...)"
    note next to the statement link on a historical calendar page, and the date in the press
    release URL (monetaryYYYYMMDDa.htm). None when no source has a date: never guess one.
    """
    found = [
        (value, source)
        for value, source in (
            (page_date, "press-release:date"),
            (calendar_date, "historical:released-text"),
            (_statement_url_date(url), "press-release-url"),
        )
        if value is not None
    ]
    if not found:
        return None
    release, source = found[0]
    conflicts = [
        f"release date {release} ({source}) disagrees with {other_source} {other}"
        for other, other_source in found[1:]
        if other != release
    ]
    return release, source, conflicts


def _statement_doc(meeting: Meeting, page: ParsedPage, entry: CalendarEntry) -> Document | None:
    url = meeting.statement_url
    assert url is not None
    release = statement_release(page.page_date, entry.statement_released, url)
    if release is None:
        return None
    release_date, source, conflicts = release
    page.warnings.extend(conflicts)
    headline = page.headline or GENERIC_STATEMENT_HEADLINE
    name = "FOMC Statement" if headline == GENERIC_STATEMENT_HEADLINE else headline
    return Document(
        doc_type="statement",
        meeting_date=meeting.meeting_date,
        release_date=release_date,
        release_date_source=source,
        title=f"{name}, {long_date(release_date)}",
        url=url,
        paragraphs=tuple(page.paragraphs),
        page_updated=page.page_updated,
    )


def _minutes_doc(meeting: Meeting, page: ParsedPage) -> Document:
    assert meeting.minutes_url is not None and meeting.minutes_release_date is not None
    on_calendar = meeting.source_url == CALENDAR_URL
    span = format_meeting_span(meeting.start_date, meeting.meeting_date)
    return Document(
        doc_type="minutes",
        meeting_date=meeting.meeting_date,
        release_date=meeting.minutes_release_date,
        release_date_source="calendar:released-text" if on_calendar else "historical:released-text",
        title=f"Minutes of the FOMC Meeting, {span}",
        url=meeting.minutes_url,
        paragraphs=tuple(page.paragraphs),
        page_updated=page.page_updated,
    )


def _statement_url_date(url: str) -> date | None:
    match = STATEMENT_URL_DATE_RE.search(url)
    return date(int(match[1]), int(match[2]), int(match[3])) if match else None


def long_date(value: date) -> str:
    return f"{value:%B} {value.day}, {value.year}"


def format_meeting_span(start: date, end: date) -> str:
    """'September 16-17, 2025', 'April 30-May 1, 2019' or 'March 15, 2020'."""
    if start == end:
        return long_date(end)
    if start.month == end.month:
        return f"{start:%B} {start.day}-{end.day}, {end.year}"
    return f"{start:%B} {start.day}-{end:%B} {end.day}, {end.year}"


# --- report ----------------------------------------------------------------------------------


def build_report(
    entries: list[CalendarEntry],
    selection: Selection,
    result: ParseResult,
    *,
    start_year: int,
    end_year: int,
    today: date,
) -> dict[str, Any]:
    """Counts, coverage, release-date sanity, text sanity and a list of pass/fail checks."""
    meetings, documents = selection.meetings, result.documents
    by_key = {(doc.doc_type, doc.meeting_date): doc for doc in documents}
    statements = [doc for doc in documents if doc.doc_type == "statement"]
    minutes = [doc for doc in documents if doc.doc_type == "minutes"]

    by_year: dict[str, dict[str, int]] = {}
    for year in range(start_year, end_year + 1):
        held = [m for m in meetings if m.meeting_date.year == year]
        by_year[str(year)] = {
            "meetings": len(held),
            "unscheduled": sum(m.unscheduled for m in held),
            "statements": sum(d.meeting_date.year == year for d in statements),
            "minutes": sum(d.meeting_date.year == year for d in minutes),
        }
    # Cancelled meetings (March 17-18, 2020) were scheduled meetings too.
    scheduled = Counter(e.end_date.year for e in entries if e.kind in ("meeting", "cancelled"))
    scheduled_off = {
        str(year): scheduled[year]
        for year in range(start_year, end_year + 1)
        if scheduled[year] != SCHEDULED_MEETINGS_PER_YEAR
    }

    missing_minutes = [
        {
            "meeting_date": m.meeting_date.isoformat(),
            "reason": _no_minutes_reason(m, selection, today),
        }
        for m in meetings
        if ("minutes", m.meeting_date) not in by_key
    ]
    release_exceptions = [
        {
            "meeting_date": d.meeting_date.isoformat(),
            "release_date": d.release_date.isoformat(),
            "days_after_meeting": (d.release_date - d.meeting_date).days,
            "release_date_source": d.release_date_source,
            "calendar_label": selection.entries[d.meeting_date].label,
        }
        for d in statements
        if d.release_date != d.meeting_date
    ]
    lags = {d.meeting_date.isoformat(): (d.release_date - d.meeting_date).days for d in minutes}
    low, high = MINUTES_LAG_BAND
    lag_outliers = {key: lag for key, lag in lags.items() if not low <= lag <= high}

    checks = _checks(
        selection, result, by_key, release_exceptions, lag_outliers, scheduled_off, today
    )
    # Documents per section name, most common first; ties by name so the report is reproducible.
    section_docs = Counter(
        section for d in minutes for section in dict.fromkeys(p.section for p in d.paragraphs)
    )
    sections = sorted(section_docs.items(), key=lambda item: (-item[1], str(item[0])))
    return {
        "as_of": today.isoformat(),
        "years": [start_year, end_year],
        "counts": {
            "meetings": len(meetings),
            "unscheduled_meetings": sum(m.unscheduled for m in meetings),
            "documents": len(documents),
            "statements": len(statements),
            "minutes": len(minutes),
        },
        "by_year": by_year,
        "date_range": {
            "first_meeting": iso_or_none(min((m.meeting_date for m in meetings), default=None)),
            "last_meeting": iso_or_none(max((m.meeting_date for m in meetings), default=None)),
            "last_release": iso_or_none(max((d.release_date for d in documents), default=None)),
        },
        "scheduled_meetings_per_year_not_8": scheduled_off,
        "skipped_calendar_entries": selection.skipped,
        "meetings_without_minutes": missing_minutes,
        "statement_release_date_exceptions": release_exceptions,
        "pages_updated_after_release": result.revised,
        "minutes_release_lag_days": {
            **_stats(lags.values()),
            "band": [low, high],
            "outliers": lag_outliers,
        },
        "word_count": {
            doc_type: _stats_with_docs([d for d in documents if d.doc_type == doc_type])
            for doc_type in ("statement", "minutes")
        },
        "paragraphs_per_doc": {
            doc_type: _stats(len(d.paragraphs) for d in documents if d.doc_type == doc_type)
            for doc_type in ("statement", "minutes")
        },
        "minutes_sections": dict(sections),
        "minutes_signature_blocks_dropped": result.signatures_dropped,
        "parse_notes": result.notes,
        "warnings": selection.warnings + result.warnings,
        "checks": checks,
    }


def _no_minutes_reason(meeting: Meeting, selection: Selection, today: date) -> str:
    entry = selection.entries[meeting.meeting_date]
    if entry.minutes_note:
        host = _host_meeting(entry, selection)
        where = f"the minutes of the {host} meeting" if host else "another meeting's minutes"
        return f"recorded in {where} (Fed note: '{entry.minutes_note}')"
    if entry.kind == "notation_vote":
        return "notation vote: no minutes of its own"
    if meeting.minutes_url and meeting.minutes_release_date is None:
        return "minutes link has no release date on the calendar; not ingested"
    if meeting.minutes_url:
        return "minutes page not ingested (see warnings)"
    days = (today - meeting.meeting_date).days
    if days <= 2 * MINUTES_LAG_BAND[1]:
        return f"not yet released ({days} days after the meeting; minutes follow in about 3 weeks)"
    return "no minutes link on the Fed calendar page"


def _host_meeting(entry: CalendarEntry, selection: Selection) -> date | None:
    """The meeting whose minutes cover this one, from 'See end of minutes of March 15 meeting'."""
    match = MINUTES_NOTE_DATE_RE.search(entry.minutes_note or "")
    if not match:
        return None
    month, day = match[1], int(match[2])
    for candidate in selection.entries.values():
        start = candidate.start_date
        if start > entry.end_date and f"{start:%B}" == month and start.day == day:
            return candidate.end_date
    return None


def _checks(
    selection: Selection,
    result: ParseResult,
    by_key: dict[tuple[str, date], Document],
    release_exceptions: list[dict[str, Any]],
    lag_outliers: dict[str, int],
    scheduled_off: dict[str, int],
    today: date,
) -> list[dict[str, Any]]:
    documents = result.documents
    ids = Counter(doc.doc_id for doc in documents)
    no_statement = [
        m.meeting_date.isoformat()
        for m in selection.meetings
        if m.statement_url and ("statement", m.meeting_date) not in by_key
    ]
    no_minutes = [
        m.meeting_date.isoformat()
        for m in selection.meetings
        if m.minutes_release_date
        and m.minutes_release_date <= today
        and ("minutes", m.meeting_date) not in by_key
    ]
    unexplained = [
        f"{e['meeting_date']} released {e['release_date']}"
        for e in release_exceptions
        if selection.entries[date.fromisoformat(e["meeting_date"])].kind == "meeting"
    ]
    empty = [doc.doc_id for doc in documents if any(not p.text.strip() for p in doc.paragraphs)]
    thin = [
        f"{doc.doc_id} ({doc.word_count} words)"
        for doc in documents
        if doc.word_count < WORD_FLOOR[doc.doc_type]
    ]
    leaks = [
        f"{doc.doc_id}: {name}"
        for doc in documents
        for name, pattern in CHROME_PATTERNS.items()
        if any(re.search(pattern, p.text) for p in doc.paragraphs)
    ]
    no_roster = [
        doc.doc_id
        for doc in documents
        if doc.doc_type == "minutes" and not any(p.section == ATTENDANCE for p in doc.paragraphs)
    ]
    unlabelled = [
        doc.doc_id
        for doc in documents
        if doc.doc_type == "minutes" and any(p.section is None for p in doc.paragraphs)
    ]
    return [
        _check("unique doc_ids", [i for i, n in ids.items() if n > 1]),
        _check("one statement per meeting with a statement link", no_statement),
        _check("minutes for every meeting whose minutes are released", no_minutes),
        _check("statements released on the meeting date, except unscheduled ones", unexplained),
        _check(
            "minutes release lag within band", [f"{k}: {v} days" for k, v in lag_outliers.items()]
        ),
        _check(
            "8 scheduled meetings per calendar year",
            [f"{year}: {n}" for year, n in scheduled_off.items()],
        ),
        _check("no empty paragraphs", empty),
        _check("word count above floor", thin, detail=f"floors: {WORD_FLOOR}"),
        _check("no page chrome in text", leaks, detail=f"patterns: {sorted(CHROME_PATTERNS)}"),
        _check("every minutes doc has an Attendance section", no_roster),
        _check("every minutes paragraph has a section", unlabelled),
        _check("no parse warnings", selection.warnings + result.warnings),
    ]


def _check(name: str, failures: list[str], detail: str | None = None) -> dict[str, Any]:
    check: dict[str, Any] = {"name": name, "passed": not failures, "failures": failures}
    if detail:
        check["detail"] = detail
    return check


def _stats(values: Iterable[int]) -> dict[str, float | int | None]:
    data = sorted(values)
    if not data:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {"count": len(data), "min": data[0], "median": statistics.median(data), "max": data[-1]}


def _stats_with_docs(documents: list[Document]) -> dict[str, Any]:
    stats: dict[str, Any] = _stats(doc.word_count for doc in documents)
    if documents:
        stats["min_doc"] = min(documents, key=lambda d: d.word_count).doc_id
        stats["max_doc"] = max(documents, key=lambda d: d.word_count).doc_id
    return stats


# --- outputs ---------------------------------------------------------------------------------


def write_meetings(paths: IngestPaths, meetings: Iterable[Meeting]) -> int:
    ordered = sorted(meetings, key=lambda m: m.meeting_date)
    return write_jsonl(paths.meetings_file, (m.to_json() for m in ordered))


def write_documents(paths: IngestPaths, documents: Iterable[Document]) -> int:
    ordered = sorted(documents, key=lambda d: (d.meeting_date, d.doc_type))
    return write_jsonl(paths.documents_file, (d.to_json() for d in ordered))


def write_report(paths: IngestPaths, report: dict[str, Any]) -> None:
    write_text_atomic(paths.report_file, json.dumps(report, indent=2, ensure_ascii=False) + "\n")


def failed_checks(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [check for check in report["checks"] if not check["passed"]]
