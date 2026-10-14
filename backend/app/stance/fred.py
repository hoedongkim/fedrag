"""The federal funds target range from FRED (DFEDTARL / DFEDTARU), reduced to its change points.

FRED's public CSV export needs no API key. It has one row per day, with an empty value where a
series has no observation; only the days on which the range changed are kept, plus the first day
of the window so the range in force at its start is known.
"""

import csv
import io
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

import httpx2

from app.ingest.records import write_text_atomic

SERIES_UPPER = "DFEDTARU"
SERIES_LOWER = "DFEDTARL"
START = date(2015, 1, 1)
CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
USER_AGENT = "FedRAG/0.1 (research project; +https://github.com/hoedongkim/fedrag)"
SOURCE = (
    f"Board of Governors of the Federal Reserve System, {SERIES_LOWER}/{SERIES_UPPER}, via FRED "
    "(Federal Reserve Bank of St. Louis)"
)
CHANGE_POINTS_HEADER = ("date", "lower", "upper")


@dataclass(frozen=True)
class TargetRange:
    """The target range in force from `date` until the next change point (percent)."""

    date: date
    lower: Decimal
    upper: Decimal


def csv_url(start: date = START) -> str:
    return f"{CSV_URL}?id={SERIES_UPPER},{SERIES_LOWER}&cosd={start.isoformat()}"


def download_csv(start: date = START, *, transport: httpx2.BaseTransport | None = None) -> str:
    """The raw CSV export of both series (FRED may return earlier rows too)."""
    with httpx2.Client(
        timeout=60,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
        transport=transport,
    ) as client:
        response = client.get(csv_url(start))
        response.raise_for_status()
        return response.text


def _value(raw: str | None) -> Decimal | None:
    """A FRED value; None for a missing one ("", "." or "#N/A")."""
    try:
        return Decimal(raw.strip()) if raw and raw.strip() not in {".", "#N/A"} else None
    except InvalidOperation as exc:
        raise ValueError(f"not a number: {raw!r}") from exc


def parse_target_range(text: str, start: date = START) -> list[TargetRange]:
    """Change points of the target range on or after `start`, oldest first."""
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    fields = reader.fieldnames or []
    date_field = next((f for f in fields if f.lower() in {"observation_date", "date"}), None)
    if date_field is None or SERIES_UPPER not in fields or SERIES_LOWER not in fields:
        raise ValueError(f"unexpected FRED header: {fields}")
    rows = []
    for row in reader:
        day = date.fromisoformat(row[date_field].strip())
        lower, upper = _value(row[SERIES_LOWER]), _value(row[SERIES_UPPER])
        if day >= start and lower is not None and upper is not None:
            if lower > upper:
                raise ValueError(f"{day}: lower bound {lower} above upper bound {upper}")
            rows.append(TargetRange(day, lower, upper))
    rows.sort(key=lambda r: r.date)
    points: list[TargetRange] = []
    for row in rows:
        if not points or (row.lower, row.upper) != (points[-1].lower, points[-1].upper):
            points.append(row)
    return points


def fetch_target_range(start: date = START) -> list[TargetRange]:
    """Download and parse the change points from `start` on."""
    return parse_target_range(download_csv(start), start)


def in_force(points: Sequence[TargetRange], day: date) -> TargetRange | None:
    """The change point in force on `day` (the latest one on or before it); None before the
    first."""
    current = None
    for point in points:
        if point.date > day:
            break
        current = point
    return current


def write_change_points(path: Path, points: Sequence[TargetRange]) -> None:
    """Save change points as CSV (date, lower, upper), atomically."""
    rows = [",".join(CHANGE_POINTS_HEADER)]
    rows += [f"{p.date.isoformat()},{p.lower},{p.upper}" for p in points]
    write_text_atomic(path, "\n".join(rows) + "\n")


def read_change_points(path: Path) -> list[TargetRange]:
    """Change points saved by write_change_points."""
    reader = csv.DictReader(io.StringIO(path.read_text(encoding="utf-8")))
    if tuple(reader.fieldnames or ()) != CHANGE_POINTS_HEADER:
        raise ValueError(f"{path.name}: unexpected header {reader.fieldnames}")
    return [
        TargetRange(date.fromisoformat(row["date"]), Decimal(row["lower"]), Decimal(row["upper"]))
        for row in reader
    ]
