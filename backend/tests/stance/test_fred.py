"""Tests for parsing FRED's federal funds target range export (no network)."""

from datetime import date
from decimal import Decimal

import httpx2
import pytest

from app.stance.fred import (
    TargetRange,
    csv_url,
    download_csv,
    in_force,
    parse_target_range,
)

# The shape of fredgraph.csv?id=DFEDTARU,DFEDTARL: daily rows, empty where a series is missing.
FIXTURE = """observation_date,DFEDTARU,DFEDTARL
2014-12-30,,0.00
2014-12-31,0.25,0.00
2015-01-01,0.25,0.00
2015-01-02,0.25,0.00
2015-12-15,0.25,0.00
2015-12-16,0.50,0.25
2015-12-17,0.50,0.25
2015-12-18,.,.
2015-12-19,0.50,0.25
2020-03-04,1.25,1.00
2020-03-05,1.25,1.00
2020-03-16,0.25,0.00
2020-03-17,0.25,0.00
"""


def r(day: str, lower: str, upper: str) -> TargetRange:
    return TargetRange(date.fromisoformat(day), Decimal(lower), Decimal(upper))


def test_keeps_only_change_points_from_the_start_date() -> None:
    assert parse_target_range(FIXTURE) == [
        r("2015-01-01", "0.00", "0.25"),
        r("2015-12-16", "0.25", "0.50"),
        r("2020-03-04", "1.00", "1.25"),
        r("2020-03-16", "0.00", "0.25"),
    ]


def test_start_date_is_configurable_and_skips_missing_values() -> None:
    points = parse_target_range(FIXTURE, start=date(2015, 12, 18))
    assert points == [
        r("2015-12-19", "0.25", "0.50"),
        r("2020-03-04", "1.00", "1.25"),
        r("2020-03-16", "0.00", "0.25"),
    ]


def test_accepts_the_old_date_header_and_a_byte_order_mark() -> None:
    text = "\ufeffDATE,DFEDTARL,DFEDTARU\n2015-01-02,0.00,0.25\n"
    assert parse_target_range(text) == [r("2015-01-02", "0.00", "0.25")]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("observation_date,DFEDTARU\n2015-01-02,0.25\n", "unexpected FRED header"),
        ("observation_date,DFEDTARU,DFEDTARL\n2015-01-02,0.25,0.50\n", "above upper bound"),
        ("observation_date,DFEDTARU,DFEDTARL\n2015-01-02,0.25,abc\n", "not a number"),
    ],
)
def test_rejects_malformed_exports(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_target_range(text)


def test_in_force_returns_the_latest_change_point_on_or_before_a_day() -> None:
    points = parse_target_range(FIXTURE)
    assert in_force(points, date(2014, 6, 1)) is None
    assert in_force(points, date(2015, 12, 16)) == points[1]
    assert in_force(points, date(2019, 1, 1)) == points[1]
    assert in_force(points, date(2026, 1, 1)) == points[-1]


def test_download_requests_both_series_without_a_key() -> None:
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, text=FIXTURE)

    assert download_csv(transport=httpx2.MockTransport(handler)) == FIXTURE
    (request,) = seen
    assert str(request.url) == csv_url()
    assert "id=DFEDTARU,DFEDTARL" in str(request.url) and "api_key" not in str(request.url)
    assert request.headers["User-Agent"].startswith("FedRAG/")
