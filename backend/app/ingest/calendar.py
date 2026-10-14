"""Parse the FOMC calendar (2021+) and historical (through 2020) pages into calendar entries."""

import re
from dataclasses import dataclass
from datetime import date
from typing import Literal

from bs4 import BeautifulSoup, Tag

from app.ingest.http import absolute_url

CALENDAR_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
HISTORICAL_URL = "https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"

EntryKind = Literal["meeting", "unscheduled", "notation_vote", "cancelled"]

MONTHS = {
    name: number
    for number, name in enumerate(
        ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"),
        start=1,
    )
}
KINDS: dict[str | None, EntryKind] = {
    None: "meeting",
    "unscheduled": "unscheduled",
    "notation vote": "notation_vote",
    "cancelled": "cancelled",
    "canceled": "cancelled",
}
DASHES = str.maketrans({"‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-"})
DAYS_RE = re.compile(
    r"(?P<d1>\d{1,2})(?:\s*-\s*(?P<d2>\d{1,2}))?\s*(?P<sep>\*)?\s*(?:\((?P<note>[^)]*)\))?"
)
HISTORICAL_HEADING_RE = re.compile(
    r"(?P<months>[A-Za-z]+(?:/[A-Za-z]+)?)\s+(?P<days>\d{1,2}(?:\s*-\s*\d{1,2})?)\s*"
    r"(?:\((?P<note>[^)]*)\))?\s*(?:Meeting)?\s*-\s*(?P<year>\d{4})"
)
RELEASED_RE = re.compile(r"\(\s*Released\s+([A-Za-z]+\.?\s+\d{1,2},\s*\d{4})\s*\)")
STATEMENT_HREF_RE = re.compile(r"/newsevents/pressreleases/monetary\d{8}[a-z]\d*\.htm$")
MINUTES_HREF_RE = re.compile(r"/monetarypolicy/fomcminutes\d{8}\.htm$")


@dataclass(frozen=True)
class CalendarEntry:
    """One meeting row of a Fed calendar page, including notation votes and cancelled meetings."""

    kind: EntryKind
    label: str
    start_date: date
    end_date: date
    has_sep: bool
    has_press_conference: bool
    statement_url: str | None
    statement_released: date | None
    minutes_url: str | None
    minutes_released: date | None
    minutes_note: str | None
    source_url: str

    @property
    def unscheduled(self) -> bool:
        # A notation vote is an unscheduled action too (2020-03-23 issued a statement).
        return self.kind in ("unscheduled", "notation_vote")


def parse_release_date(text: str) -> date:
    """Parse 'October 09, 2024' / 'March 3, 2020' / 'Sept. 3, 2020' style dates."""
    match = re.fullmatch(r"([A-Za-z]+)\.?\s+(\d{1,2}),\s*(\d{4})", text.strip())
    if not match:
        raise ValueError(f"Unrecognised date {text!r}")
    month, day, year = match.groups()
    return date(int(year), MONTHS[month[:3].lower()], int(day))


def find_released(text: str) -> date | None:
    """The date in a '(Released Month DD, YYYY)' note, if the text has one."""
    match = RELEASED_RE.search(text)
    return parse_release_date(match[1]) if match else None


def meeting_span(year: int, months: str, days: str) -> tuple[date, date]:
    """Start and end dates from 'Apr/May' + '30-1', 'September' + '16-17' or 'March' + '15'."""
    month_numbers = [MONTHS[m.strip()[:3].lower()] for m in months.split("/")]
    first, _, last = days.partition("-")
    start = date(year, month_numbers[0], int(first))
    if not last:
        return start, start
    end_month = month_numbers[-1]
    if len(month_numbers) == 1 and int(last) < int(first):
        end_month += 1
    return start, date(year, end_month, int(last))


def _kind(note: str | None, label: str) -> EntryKind:
    key = note.strip().lower() if note else None
    if key not in KINDS:
        raise ValueError(f"Unknown calendar note {note!r} in {label!r}")
    return KINDS[key]


def _text(node: Tag) -> str:
    return " ".join(node.get_text(" ", strip=True).split()).translate(DASHES)


def _link(node: Tag, text: str, href_re: re.Pattern[str]) -> str | None:
    """Absolute URL of the first link in `node` with this exact text and a matching href."""
    for anchor in node.find_all("a", href=True):
        if _text(anchor) == text and href_re.search(anchor["href"]):
            return absolute_url(anchor["href"])
    return None


def _has_link(node: Tag, pattern: str) -> bool:
    return any(re.search(pattern, _text(a), re.IGNORECASE) for a in node.find_all("a"))


def parse_calendar_page(html: bytes | str, source_url: str = CALENDAR_URL) -> list[CalendarEntry]:
    """Entries from fomccalendars.htm: one panel per year, one div.fomc-meeting per row."""
    soup = BeautifulSoup(html, "lxml")
    entries: list[CalendarEntry] = []
    for panel in soup.select("div.panel"):
        heading = panel.select_one(".panel-heading")
        match = heading and re.fullmatch(r"(\d{4}) FOMC Meetings", _text(heading))
        if not match:
            continue
        year = int(match[1])
        for row in panel.select("div.fomc-meeting"):
            entries.append(_calendar_row(row, year, source_url))
    return entries


def _calendar_row(row: Tag, year: int, source_url: str) -> CalendarEntry:
    months = _text(row.select_one(".fomc-meeting__month"))
    days_text = _text(row.select_one(".fomc-meeting__date"))
    label = f"{months} {days_text} {year}"
    days = DAYS_RE.fullmatch(days_text)
    if not days:
        raise ValueError(f"Unrecognised meeting dates {label!r} on {source_url}")
    day_range = days["d1"] + (f"-{days['d2']}" if days["d2"] else "")
    start, end = meeting_span(year, months, day_range)

    statement_url = None
    for column in row.find_all("div", recursive=False):
        if column.find("strong", string=re.compile(r"^\s*Statement:?\s*$")):
            statement_url = _link(column, "HTML", STATEMENT_HREF_RE)
    minutes_url = minutes_released = None
    minutes_column = row.select_one(".fomc-meeting__minutes")
    if minutes_column is not None:
        minutes_url = _link(minutes_column, "HTML", MINUTES_HREF_RE)
        minutes_released = find_released(_text(minutes_column))
    return CalendarEntry(
        kind=_kind(days["note"], label),
        label=label,
        start_date=start,
        end_date=end,
        has_sep=bool(days["sep"]) or _has_link(row, r"^Projection Materials"),
        has_press_conference=_has_link(row, r"^Press Conference"),
        statement_url=statement_url,
        statement_released=None,
        minutes_url=minutes_url,
        minutes_released=minutes_released,
        minutes_note=None,
        source_url=source_url,
    )


def parse_historical_page(html: bytes | str, source_url: str) -> list[CalendarEntry]:
    """Entries from fomchistoricalYYYY.htm: one div.panel per meeting, titled by an h5."""
    soup = BeautifulSoup(html, "lxml")
    entries: list[CalendarEntry] = []
    for heading in soup.select("div.panel h5.panel-heading"):
        label = _text(heading)
        match = HISTORICAL_HEADING_RE.fullmatch(label)
        if not match:
            raise ValueError(f"Unrecognised meeting heading {label!r} on {source_url}")
        start, end = meeting_span(int(match["year"]), match["months"], match["days"])
        entries.append(
            _historical_panel(heading.parent, match["note"], label, start, end, source_url)
        )
    return entries


def _historical_panel(
    panel: Tag, note: str | None, label: str, start: date, end: date, source_url: str
) -> CalendarEntry:
    statement_url = statement_released = None
    for anchor in panel.find_all("a", href=True):
        if _text(anchor) == "Statement" and STATEMENT_HREF_RE.search(anchor["href"]):
            statement_url = absolute_url(anchor["href"])
            statement_released = find_released(_text(anchor.parent))
            break

    minutes_url = minutes_released = minutes_note = None
    for paragraph in panel.find_all("p"):
        text = _text(paragraph)
        if not text.startswith("Minutes"):
            continue
        if link := _link(paragraph, "HTML", MINUTES_HREF_RE):
            minutes_url, minutes_released = link, find_released(text)
        elif emphasis := paragraph.find("em"):
            minutes_note = _text(emphasis)
    return CalendarEntry(
        kind=_kind(note, label),
        label=label,
        start_date=start,
        end_date=end,
        has_sep=_has_link(panel, r"^SEP\b|Projection"),
        has_press_conference=_has_link(panel, r"^Press Conference"),
        statement_url=statement_url,
        statement_released=statement_released,
        minutes_url=minutes_url,
        minutes_released=minutes_released,
        minutes_note=minutes_note,
        source_url=source_url,
    )
