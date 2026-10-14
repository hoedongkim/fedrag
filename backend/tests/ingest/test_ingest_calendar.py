"""Tests for reading meetings off the Fed calendar (2021+) and historical (through 2020) pages."""

from collections.abc import Callable
from datetime import date

import pytest

from app.ingest.calendar import (
    CALENDAR_URL,
    HISTORICAL_URL,
    CalendarEntry,
    meeting_span,
    parse_calendar_page,
    parse_historical_page,
    parse_release_date,
)

BASE = "https://www.federalreserve.gov"
FedPage = Callable[[str], bytes]


def by_end_date(entries: list[CalendarEntry]) -> dict[date, CalendarEntry]:
    return {entry.end_date: entry for entry in entries}


def test_calendar_page_meeting_with_sep_and_released_minutes(fed_page: FedPage) -> None:
    entries = by_end_date(parse_calendar_page(fed_page("fomccalendars.htm")))
    meeting = entries[date(2025, 9, 17)]

    assert meeting.kind == "meeting"
    assert (meeting.start_date, meeting.end_date) == (date(2025, 9, 16), date(2025, 9, 17))
    assert meeting.has_sep and meeting.has_press_conference
    assert meeting.statement_url == f"{BASE}/newsevents/pressreleases/monetary20250917a.htm"
    assert meeting.minutes_url == f"{BASE}/monetarypolicy/fomcminutes20250917.htm"
    assert meeting.minutes_released == date(2025, 10, 8)
    assert meeting.statement_released is None
    assert meeting.source_url == CALENDAR_URL


def test_calendar_page_future_meetings_have_no_documents(fed_page: FedPage) -> None:
    entries = by_end_date(parse_calendar_page(fed_page("fomccalendars.htm")))
    october, december = entries[date(2026, 10, 28)], entries[date(2026, 12, 9)]

    for meeting in (october, december):
        assert meeting.kind == "meeting"
        assert meeting.statement_url is meeting.minutes_url is meeting.minutes_released is None
        assert not meeting.has_press_conference
    assert not october.has_sep
    assert december.has_sep  # the asterisk marks an SEP meeting before its materials exist


def test_calendar_page_cross_month_meeting_and_notation_vote(fed_page: FedPage) -> None:
    entries = by_end_date(parse_calendar_page(fed_page("fomccalendars.htm")))

    apr_may = entries[date(2024, 5, 1)]
    assert apr_may.start_date == date(2024, 4, 30)
    assert apr_may.minutes_released == date(2024, 5, 22)
    assert not apr_may.has_sep

    vote = entries[date(2025, 8, 22)]
    assert vote.kind == "notation_vote" and vote.unscheduled
    assert vote.statement_url is None  # it only re-issued the longer-run goals statement


def test_calendar_page_entries_cover_listed_rows_only(fed_page: FedPage) -> None:
    entries = parse_calendar_page(fed_page("fomccalendars.htm"))

    assert sorted(e.end_date.isoformat() for e in entries) == [
        "2024-05-01",
        "2025-08-22",
        "2025-09-17",
        "2026-09-16",
        "2026-10-28",
        "2026-12-09",
    ]


def test_historical_2015_page(fed_page: FedPage) -> None:
    url = HISTORICAL_URL.format(year=2015)
    entries = by_end_date(parse_historical_page(fed_page("fomchistorical2015.htm"), url))

    january, march = entries[date(2015, 1, 28)], entries[date(2015, 3, 18)]
    assert january.start_date == date(2015, 1, 27)
    assert not january.has_sep and not january.has_press_conference
    assert january.statement_url == f"{BASE}/newsevents/pressreleases/monetary20150128a.htm"
    assert january.minutes_url == f"{BASE}/monetarypolicy/fomcminutes20150128.htm"
    assert january.minutes_released == date(2015, 2, 18)
    assert march.has_sep and march.has_press_conference
    assert march.minutes_released == date(2015, 4, 8)
    assert {e.source_url for e in entries.values()} == {url}


def test_historical_2020_unscheduled_cancelled_and_notation_votes(fed_page: FedPage) -> None:
    url = HISTORICAL_URL.format(year=2020)
    entries = by_end_date(parse_historical_page(fed_page("fomchistorical2020.htm"), url))

    march_2 = entries[date(2020, 3, 2)]
    assert march_2.kind == "unscheduled" and march_2.unscheduled
    assert march_2.statement_url == f"{BASE}/newsevents/pressreleases/monetary20200303a.htm"
    assert march_2.statement_released == date(2020, 3, 3)
    assert march_2.minutes_url is None
    assert march_2.minutes_note == "See end of minutes of March 15 meeting"

    march_15 = entries[date(2020, 3, 15)]
    assert march_15.kind == "unscheduled"
    assert march_15.minutes_released == date(2020, 4, 8)

    assert entries[date(2020, 3, 18)].kind == "cancelled"
    assert entries[date(2020, 3, 18)].statement_url is None
    assert entries[date(2020, 3, 19)].kind == "notation_vote"
    assert entries[date(2020, 3, 19)].statement_url is None  # a press release, not a statement
    march_23 = entries[date(2020, 3, 23)]
    assert march_23.kind == "notation_vote"
    assert march_23.statement_url == f"{BASE}/newsevents/pressreleases/monetary20200323a.htm"
    assert march_23.statement_released == date(2020, 3, 23)

    june = entries[date(2020, 6, 10)]
    assert june.has_sep and june.has_press_conference


@pytest.mark.parametrize(
    ("months", "days", "expected"),
    [
        ("September", "16-17", (date(2025, 9, 16), date(2025, 9, 17))),
        ("Apr/May", "30-1", (date(2025, 4, 30), date(2025, 5, 1))),
        ("Jan/Feb", "31-1", (date(2025, 1, 31), date(2025, 2, 1))),
        ("October", "31-1", (date(2025, 10, 31), date(2025, 11, 1))),
        ("March", "15", (date(2025, 3, 15), date(2025, 3, 15))),
    ],
)
def test_meeting_span(months: str, days: str, expected: tuple[date, date]) -> None:
    assert meeting_span(2025, months, days) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("October 09, 2024", date(2024, 10, 9)),
        ("March 3, 2020", date(2020, 3, 3)),
        ("Sept. 3, 2020", date(2020, 9, 3)),
    ],
)
def test_parse_release_date(text: str, expected: date) -> None:
    assert parse_release_date(text) == expected


def test_parse_release_date_rejects_other_text() -> None:
    with pytest.raises(ValueError, match="Unrecognised date"):
        parse_release_date("sometime in 2020")


def test_unknown_calendar_note_fails_loudly() -> None:
    html = """<div class="panel"><div class="panel-heading"><h4>2030 FOMC Meetings</h4></div>
    <div class="row fomc-meeting"><div class="fomc-meeting__month">May</div>
    <div class="fomc-meeting__date">5 (postponed)</div></div></div>"""

    with pytest.raises(ValueError, match="Unknown calendar note"):
        parse_calendar_page(html)
