"""Tests for `python -m app.ingest`: argument handling, command dispatch and the load step."""

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any

import httpx2
import pytest

import app.ingest.__main__ as cli
from app.core.config import Settings
from app.db.connection import MALFORMED_URL
from app.ingest.http import FedClient

TODAY = date(2026, 10, 12)
CALENDAR_PATH = "/monetarypolicy/fomccalendars.htm"
Factory = Callable[..., FedClient]


def test_defaults() -> None:
    args = cli.build_parser(TODAY).parse_args(["parse"])

    assert (args.command, args.start_year, args.end_year, args.refresh) == (
        "parse",
        2015,
        2026,
        False,
    )


def test_options() -> None:
    args = cli.build_parser(TODAY).parse_args(
        ["fetch", "--start-year", "2020", "--end-year", "2021", "--refresh"]
    )

    assert (args.command, args.start_year, args.end_year, args.refresh) == (
        "fetch",
        2020,
        2021,
        True,
    )


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["download"],
        ["parse", "--start-year", "twenty"],
        ["parse", "--start-year", "2021", "--end-year", "2020"],
        ["parse", "--start-year", "1900"],
        ["parse", "--end-year", "2030"],
    ],
)
def test_invalid_arguments_exit_2(argv: list[str], settings: Settings) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(argv, settings=settings, today=TODAY)
    assert exc.value.code == 2


def test_parse_writes_processed_files_offline_and_idempotently(
    settings: Settings, offline_factory: Factory, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["parse", "--start-year", "2025", "--end-year", "2025"]
    processed = settings.data_dir / "processed"

    status = cli.main(argv, settings=settings, today=TODAY, client_factory=offline_factory)
    first = {
        name: (processed / name).read_bytes() for name in ("meetings.jsonl", "documents.jsonl")
    }
    cli.main(argv, settings=settings, today=TODAY, client_factory=offline_factory)

    # only the trimmed fixture calendar's "8 meetings a year" cross-check fails
    assert status == 1
    output = capsys.readouterr().out
    assert "parse: 1 meetings, 1 statements, 1 minutes" in output
    assert "FAILED 8 scheduled meetings per calendar year: 2025: 1" in output
    for name, content in first.items():
        assert (processed / name).read_bytes() == content
    documents = [json.loads(line) for line in first["documents.jsonl"].splitlines()]
    assert [d["doc_id"] for d in documents] == ["minutes-2025-09-17", "statement-2025-09-17"]
    report = json.loads((processed / "report.json").read_text())
    assert report["counts"]["documents"] == 2


def test_calendar_and_fetch_use_the_cache(
    settings: Settings,
    fed_page: Callable[[str], bytes],
    capsys: pytest.CaptureFixture[str],
) -> None:
    requested: list[str] = []

    def calendar_only(request: httpx2.Request) -> httpx2.Response:
        requested.append(request.url.path)
        assert request.url.path == CALENDAR_PATH, f"unexpected network request: {request.url}"
        return httpx2.Response(200, content=fed_page("fomccalendars.htm"))

    def factory(raw_dir: Path, **kwargs: Any) -> FedClient:
        transport = httpx2.MockTransport(calendar_only)
        return FedClient(raw_dir, transport=transport, sleep=lambda _: None, **kwargs)

    years = ["--start-year", "2025", "--end-year", "2025"]
    kwargs = {"settings": settings, "today": TODAY, "client_factory": factory}

    assert cli.main(["calendar", *years], **kwargs) == 0
    assert requested == [CALENDAR_PATH]
    assert cli.main(["fetch", *years], **kwargs) == 0
    assert requested == [CALENDAR_PATH, CALENDAR_PATH]  # the calendar page, once per command

    output = capsys.readouterr().out
    assert "calendar: 2 calendar entries for 2025-2025, 1 meetings held by 2026-10-12" in output
    assert "skipped August 22 (notation vote) 2025: notation vote without a policy statement" in (
        output
    )
    assert "fetch: 0 pages downloaded, 2 already cached" in output
    assert not (settings.data_dir / "processed" / "meetings.jsonl").exists()


def test_load_without_database_url_exits_2(
    settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["load"], settings=settings, today=TODAY) == 2
    assert "DATABASE_URL is not set" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("parse_status", "database_url", "expected_status", "expected_output", "loads"),
    [
        (0, None, 0, "all: note: DATABASE_URL is not set", 0),
        (1, "postgresql://u:p@localhost/db", 1, "skipped the database load", 0),
        (0, "postgresql://u:p@localhost/db", 0, "", 1),
    ],
)
def test_all_loads_only_when_configured_and_checks_pass(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    data_dir: Path,
    offline_factory: Factory,
    parse_status: int,
    database_url: str | None,
    expected_status: int,
    expected_output: str,
    loads: int,
) -> None:
    settings = Settings(_env_file=None, data_dir=data_dir, database_url=database_url)
    calls: list[str] = []
    monkeypatch.setattr(cli, "run_calendar", lambda *a: calls.append("calendar"))
    monkeypatch.setattr(cli, "run_fetch", lambda *a: calls.append("fetch"))
    monkeypatch.setattr(cli, "run_parse", lambda *a: calls.append("parse") or parse_status)
    monkeypatch.setattr(cli, "run_load", lambda *a: calls.append("load") or 0)

    status = cli.main(["all"], settings=settings, today=TODAY, client_factory=offline_factory)

    assert status == expected_status
    assert calls == ["calendar", "fetch", "parse"] + ["load"] * loads
    assert expected_output in capsys.readouterr().out


def test_all_no_load_never_loads(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    data_dir: Path,
    offline_factory: Factory,
) -> None:
    settings = Settings(_env_file=None, data_dir=data_dir, database_url="postgresql://x@h/db")
    calls: list[str] = []
    monkeypatch.setattr(cli, "run_calendar", lambda *a: calls.append("calendar"))
    monkeypatch.setattr(cli, "run_fetch", lambda *a: calls.append("fetch"))
    monkeypatch.setattr(cli, "run_parse", lambda *a: calls.append("parse") or 0)
    monkeypatch.setattr(cli, "run_load", lambda *a: calls.append("load") or 0)

    argv = ["all", "--no-load"]
    status = cli.main(argv, settings=settings, today=TODAY, client_factory=offline_factory)

    assert status == 0
    assert calls == ["calendar", "fetch", "parse"]
    assert capsys.readouterr().out == ""


class FakeConnection:
    pass


def write_report(data_dir: Path, *checks: tuple[str, bool]) -> None:
    """A report.json holding only these (name, passed) checks, as the parse step writes it."""
    report = {
        "checks": [
            {"name": name, "passed": passed, "failures": [] if passed else ["bad"]}
            for name, passed in checks
        ]
    }
    processed = data_dir / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    (processed / "report.json").write_text(json.dumps(report), encoding="utf-8")


def forbid_connect(monkeypatch: pytest.MonkeyPatch) -> list[Settings]:
    """Record any call to app.db.connection.connect, which these tests must never reach."""
    import app.db.connection

    calls: list[Settings] = []
    monkeypatch.setattr(app.db.connection, "connect", calls.append)
    return calls


def test_load_calls_the_db_loader(
    monkeypatch: pytest.MonkeyPatch, data_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import app.db.connection
    import app.db.load

    settings = Settings(_env_file=None, data_dir=data_dir, database_url="postgresql://x@h/db")
    write_report(data_dir, ("x", True))
    seen: dict[str, Any] = {}

    @contextmanager
    def fake_connect(received: Settings) -> Iterator[FakeConnection]:
        seen["settings"] = received
        yield FakeConnection()

    def fake_load(conn: FakeConnection, processed_dir: Path) -> dict[str, int]:
        seen["processed_dir"] = processed_dir
        return {"meetings": 97, "documents": 191}

    monkeypatch.setattr(app.db.connection, "connect", fake_connect)
    monkeypatch.setattr(app.db.load, "load_processed", fake_load)

    assert cli.main(["load"], settings=settings, today=TODAY) == 0
    assert seen == {"settings": settings, "processed_dir": data_dir / "processed"}
    assert "upserted 97 meetings and 191 documents" in capsys.readouterr().out


def test_load_reports_loader_errors(
    monkeypatch: pytest.MonkeyPatch, data_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import app.db.connection
    import app.db.load

    settings = Settings(_env_file=None, data_dir=data_dir, database_url="postgresql://x@h/db")
    write_report(data_dir, ("x", True))

    @contextmanager
    def fake_connect(received: Settings) -> Iterator[FakeConnection]:
        yield FakeConnection()

    def failing_load(conn: FakeConnection, processed_dir: Path) -> dict[str, int]:
        raise app.db.load.LoadError("meetings.jsonl has no rows: nothing to load.")

    monkeypatch.setattr(app.db.connection, "connect", fake_connect)
    monkeypatch.setattr(app.db.load, "load_processed", failing_load)

    assert cli.main(["load"], settings=settings, today=TODAY) == 1
    assert "load: failed: meetings.jsonl has no rows" in capsys.readouterr().err


def test_load_refuses_when_the_last_parse_failed_a_check(
    monkeypatch: pytest.MonkeyPatch, data_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = Settings(_env_file=None, data_dir=data_dir, database_url="postgresql://x@h/db")
    write_report(data_dir, ("unique doc_ids", True), ("no page chrome in text", False))
    connects = forbid_connect(monkeypatch)

    assert cli.main(["load"], settings=settings, today=TODAY) == 1
    err = capsys.readouterr().err
    assert "load: refused: the last parse failed 1 check(s): no page chrome in text" in err
    assert connects == []


def test_load_refuses_without_a_report(
    monkeypatch: pytest.MonkeyPatch, data_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = Settings(_env_file=None, data_dir=data_dir, database_url="postgresql://x@h/db")
    connects = forbid_connect(monkeypatch)

    assert cli.main(["load"], settings=settings, today=TODAY) == 1
    err = capsys.readouterr().err
    assert "load: refused:" in err and "report.json not found; run `make ingest` first" in err
    assert connects == []


@pytest.mark.parametrize(
    "database_url",
    [
        "postgresql://u:Sentinel%ZZpw@127.0.0.1:1/db",
        "postgresql://u:Senti nelpw@127.0.0.1:1/db",
        "postgresql://u:Senti@nelpw@127.0.0.1:1/db",
        "u:Sentinelpw@127.0.0.1:1/db",
    ],
    ids=["bad-percent-escape", "space", "unencoded-at", "no-scheme"],
)
def test_load_never_prints_a_malformed_database_url(
    data_dir: Path, capsys: pytest.CaptureFixture[str], database_url: str
) -> None:
    # Every case fails while parsing the URL, before any network connection is attempted.
    settings = Settings(_env_file=None, data_dir=data_dir, database_url=database_url)
    write_report(data_dir, ("x", True))

    assert cli.main(["load"], settings=settings, today=TODAY) == 1
    captured = capsys.readouterr()
    assert captured.err == f"load: failed: {MALFORMED_URL}\n"
    assert "malformed" in captured.err
    assert "Senti" not in captured.out + captured.err
