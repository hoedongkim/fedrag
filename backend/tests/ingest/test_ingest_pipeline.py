"""Tests for the pipeline steps on the fixture cache: selection, release dates, report, outputs."""

import json
from collections.abc import Callable
from datetime import date
from pathlib import Path

import httpx2
import pytest

from app.ingest.calendar import CALENDAR_URL
from app.ingest.http import CacheMissError, FedClient
from app.ingest.pipeline import (
    IngestPaths,
    build_documents,
    build_report,
    collect_calendar,
    fetch_documents,
    format_meeting_span,
    select_meetings,
    statement_release,
)

TODAY = date(2026, 10, 12)
STATEMENT_URL = "https://www.federalreserve.gov/newsevents/pressreleases/monetary20200303a.htm"
CALENDAR_PATH = "/monetarypolicy/fomccalendars.htm"


def offline_client(data_dir: Path) -> FedClient:
    return FedClient(data_dir / "raw", offline=True)


def checks_by_name(report: dict) -> dict[str, dict]:
    return {check["name"]: check for check in report["checks"]}


def test_collect_calendar_uses_the_calendar_page_for_recent_years(data_dir: Path) -> None:
    entries = collect_calendar(offline_client(data_dir), 2025, 2026)

    assert [e.end_date.isoformat() for e in entries] == [
        "2025-08-22",
        "2025-09-17",
        "2026-09-16",
        "2026-10-28",
        "2026-12-09",
    ]
    assert {e.source_url for e in entries} == {CALENDAR_URL}


def test_collect_calendar_adds_historical_pages_for_older_years(data_dir: Path) -> None:
    entries = collect_calendar(offline_client(data_dir), 2015, 2015)

    assert [e.end_date for e in entries] == [date(2015, 1, 28), date(2015, 3, 18)]
    with pytest.raises(CacheMissError):  # 2016 is neither on the calendar nor cached
        collect_calendar(offline_client(data_dir), 2015, 2016)


def test_select_meetings_skips_upcoming_cancelled_and_statementless_votes(data_dir: Path) -> None:
    client = offline_client(data_dir)
    entries = collect_calendar(client, 2020, 2020) + collect_calendar(client, 2026, 2026)

    selection = select_meetings(entries, TODAY)

    assert [m.meeting_date.isoformat() for m in selection.meetings] == [
        "2020-03-02",
        "2020-03-15",
        "2020-03-23",
        "2020-06-10",
        "2026-09-16",
    ]
    assert [m.unscheduled for m in selection.meetings] == [True, True, True, False, False]
    assert {(s["date"], s["reason"]) for s in selection.skipped} == {
        ("2020-03-18", "cancelled meeting"),
        ("2020-03-19", "notation vote without a policy statement"),
        ("2026-10-28", "upcoming meeting"),
        ("2026-12-09", "upcoming meeting"),
    }
    assert selection.warnings == []


def test_held_meeting_awaiting_minutes_is_kept_without_minutes(data_dir: Path) -> None:
    entries = collect_calendar(offline_client(data_dir), 2026, 2026)

    selection = select_meetings(entries, date(2026, 10, 29))
    october = selection.meetings[-1]

    assert october.meeting_date == date(2026, 10, 28)
    assert october.minutes_url is None and october.minutes_release_date is None


@pytest.mark.parametrize(
    ("page_date", "calendar_date", "expected"),
    [
        (date(2020, 3, 3), date(2020, 3, 3), (date(2020, 3, 3), "press-release:date", [])),
        (None, date(2020, 3, 3), (date(2020, 3, 3), "historical:released-text", [])),
        (None, None, (date(2020, 3, 3), "press-release-url", [])),
        (
            date(2020, 3, 4),
            None,
            (
                date(2020, 3, 4),
                "press-release:date",
                [
                    "release date 2020-03-04 (press-release:date) disagrees with "
                    "press-release-url 2020-03-03"
                ],
            ),
        ),
    ],
)
def test_statement_release_prefers_the_press_release(
    page_date: date | None, calendar_date: date | None, expected: tuple
) -> None:
    assert statement_release(page_date, calendar_date, STATEMENT_URL) == expected


def test_statement_release_never_guesses() -> None:
    assert statement_release(None, None, "https://www.federalreserve.gov/statement.htm") is None


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        (date(2025, 9, 16), date(2025, 9, 17), "September 16-17, 2025"),
        (date(2019, 4, 30), date(2019, 5, 1), "April 30-May 1, 2019"),
        (date(2020, 3, 15), date(2020, 3, 15), "March 15, 2020"),
    ],
)
def test_format_meeting_span(start: date, end: date, expected: str) -> None:
    assert format_meeting_span(start, end) == expected


def test_build_documents_and_report_for_2025(data_dir: Path) -> None:
    client = offline_client(data_dir)
    entries = collect_calendar(client, 2025, 2025)
    selection = select_meetings(entries, TODAY)

    result = build_documents(client, selection)
    report = build_report(entries, selection, result, start_year=2025, end_year=2025, today=TODAY)

    statement, minutes = sorted(result.documents, key=lambda d: d.doc_type, reverse=True)
    assert statement.doc_id == "statement-2025-09-17"
    assert (statement.release_date, statement.release_date_source) == (
        date(2025, 9, 17),
        "press-release:date",
    )
    assert statement.title == "FOMC Statement, September 17, 2025"
    assert minutes.doc_id == "minutes-2025-09-17"
    assert (minutes.release_date, minutes.release_date_source) == (
        date(2025, 10, 8),
        "calendar:released-text",
    )
    assert minutes.title == "Minutes of the FOMC Meeting, September 16-17, 2025"

    assert report["counts"] == {
        "meetings": 1,
        "unscheduled_meetings": 0,
        "documents": 2,
        "statements": 1,
        "minutes": 1,
    }
    assert report["minutes_release_lag_days"]["median"] == 21
    assert report["statement_release_date_exceptions"] == []
    assert report["minutes_signature_blocks_dropped"] == 1
    assert report["parse_notes"] == []
    checks = checks_by_name(report)
    # the trimmed calendar fixture lists one scheduled 2025 meeting, so the cross-check fails
    assert checks["8 scheduled meetings per calendar year"]["failures"] == ["2025: 1"]
    assert all(c["passed"] for name, c in checks.items() if not name.startswith("8 scheduled"))


def test_report_explains_missing_minutes_and_documents(data_dir: Path) -> None:
    client = offline_client(data_dir)
    entries = collect_calendar(client, 2020, 2020)
    selection = select_meetings(entries, TODAY)

    result = build_documents(client, selection)
    report = build_report(entries, selection, result, start_year=2020, end_year=2020, today=TODAY)

    reasons = {row["meeting_date"]: row["reason"] for row in report["meetings_without_minutes"]}
    assert reasons["2020-03-02"].startswith("recorded in the minutes of the 2020-03-15 meeting")
    assert reasons["2020-03-23"] == "notation vote: no minutes of its own"
    # the fixture cache holds no 2020 statements or minutes: reported, never invented
    assert any("statement-2020-03-02: page not downloaded" in w for w in report["warnings"])
    assert result.documents[0].doc_id == "statement-2020-03-23"
    assert not checks_by_name(report)["no parse warnings"]["passed"]


def test_fetch_documents_downloads_only_missing_pages(
    tmp_path: Path, fed_page: Callable[[str], bytes]
) -> None:
    # A stale cached calendar (no meetings at all) is replaced by the one the Fed serves now.
    calendar = tmp_path / "raw" / "calendar" / "fomccalendars.htm"
    calendar.parent.mkdir(parents=True)
    calendar.write_bytes(b"<html></html>")
    requested: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requested.append(request.url.path)
        if request.url.path == CALENDAR_PATH:
            return httpx2.Response(200, content=fed_page("fomccalendars.htm"))
        return httpx2.Response(200, content=b"<html></html>")

    def run() -> dict[str, int]:
        with FedClient(
            tmp_path / "raw", transport=httpx2.MockTransport(handler), sleep=lambda _: None
        ) as client:
            selection = select_meetings(collect_calendar(client, 2025, 2025), TODAY)
            return fetch_documents(client, selection.meetings, log=lambda _: None)

    assert run() == {"downloaded": 2}
    assert sorted(requested) == [
        CALENDAR_PATH,
        "/monetarypolicy/fomcminutes20250917.htm",
        "/newsevents/pressreleases/monetary20250917a.htm",
    ]
    assert calendar.read_bytes() == fed_page("fomccalendars.htm")
    requested.clear()

    assert run() == {"cached": 2}
    assert requested == [CALENDAR_PATH]  # only the calendar page is downloaded again


def test_ingest_paths() -> None:
    paths = IngestPaths.under(Path("/data"))

    assert paths.raw_dir == Path("/data/raw")
    assert paths.meetings_file == Path("/data/processed/meetings.jsonl")
    assert paths.documents_file == Path("/data/processed/documents.jsonl")
    assert paths.report_file == Path("/data/processed/report.json")


def test_report_is_json_serializable(data_dir: Path) -> None:
    client = offline_client(data_dir)
    entries = collect_calendar(client, 2025, 2025)
    selection = select_meetings(entries, TODAY)
    result = build_documents(client, selection)

    report = build_report(entries, selection, result, start_year=2025, end_year=2025, today=TODAY)

    assert json.loads(json.dumps(report)) == report
