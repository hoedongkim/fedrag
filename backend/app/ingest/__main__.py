"""Command line for the ingest pipeline: python -m app.ingest {calendar,fetch,parse,load,all}."""

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from datetime import date

from app.core.config import Settings, get_settings
from app.ingest.calendar import CalendarEntry
from app.ingest.http import FedClient
from app.ingest.pipeline import (
    IngestPaths,
    Selection,
    build_documents,
    build_report,
    collect_calendar,
    failed_checks,
    fetch_documents,
    select_meetings,
    write_documents,
    write_meetings,
    write_report,
)

COMMANDS = {
    "calendar": (
        "download the FOMC calendar pages and list the meetings held "
        "(writes nothing to data/processed)"
    ),
    "fetch": (
        "re-download the calendar page and download any statement or minutes page "
        "not yet in data/raw"
    ),
    "parse": "parse cached pages into documents.jsonl, meetings.jsonl and report.json (offline)",
    "load": "upsert data/processed into Postgres schema fedrag (needs DATABASE_URL)",
    "all": "calendar + fetch + parse, then load when DATABASE_URL is set (unless --no-load)",
}
FIRST_YEAR = 1936  # the first year on the Fed's historical FOMC pages
NO_DATABASE = "DATABASE_URL is not set: add it to backend/.env (see .env.example) to load Postgres."

ClientFactory = Callable[..., FedClient]


def build_parser(today: date) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.ingest",
        description="Collect FOMC statements and minutes from federalreserve.gov into data/.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="commands:\n" + "\n".join(f"  {name:<9} {text}" for name, text in COMMANDS.items()),
    )
    parser.add_argument("command", choices=COMMANDS, help="pipeline step to run (see below)")
    parser.add_argument(
        "--start-year", type=int, default=2015, help="first meeting year (default: 2015)"
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=today.year,
        help=f"last meeting year (default: the current year, {today.year})",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "re-download every page, even cached ones (otherwise only fomccalendars.htm and "
            "pages missing from data/raw are downloaded)"
        ),
    )
    parser.add_argument(
        "--no-load",
        action="store_true",
        help="with all: stop after parse, even when DATABASE_URL is set",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    settings: Settings | None = None,
    today: date | None = None,
    client_factory: ClientFactory = FedClient,
) -> int:
    """Run one command; returns the process exit code."""
    today = today or date.today()
    parser = build_parser(today)
    args = parser.parse_args(argv)
    if not FIRST_YEAR <= args.start_year <= args.end_year <= today.year + 1:
        parser.error(
            f"need {FIRST_YEAR} <= --start-year <= --end-year <= {today.year + 1}, "
            f"got {args.start_year} and {args.end_year}"
        )
    settings = settings or get_settings()
    paths = IngestPaths.under(settings.data_dir)
    years = (args.start_year, args.end_year)

    if args.command == "load":
        return run_load(settings, paths)
    if args.command in ("calendar", "fetch", "all"):
        with client_factory(paths.raw_dir, refresh=args.refresh) as client:
            if args.command in ("calendar", "all"):
                run_calendar(client, years, today)
            if args.command in ("fetch", "all"):
                run_fetch(client, years, today)
    if args.command in ("calendar", "fetch"):
        return 0

    with client_factory(paths.raw_dir, offline=True) as client:
        status = run_parse(client, paths, years, today)
    if args.command == "all" and not args.no_load:
        if status:
            print("all: skipped the database load because some checks failed.")
        elif settings.database_url is None:
            print(f"all: note: {NO_DATABASE}")
        else:
            status = run_load(settings, paths)
    return status


def _selection(
    client: FedClient, years: tuple[int, int], today: date
) -> tuple[list[CalendarEntry], Selection]:
    entries = collect_calendar(client, *years)
    return entries, select_meetings(entries, today)


def run_calendar(client: FedClient, years: tuple[int, int], today: date) -> None:
    entries, selection = _selection(client, years, today)
    print(
        f"calendar: {len(entries)} calendar entries for {years[0]}-{years[1]}, "
        f"{len(selection.meetings)} meetings held by {today}"
    )
    for skipped in selection.skipped:
        print(f"  skipped {skipped['label']}: {skipped['reason']}")
    for warning in selection.warnings:
        print(f"  warning: {warning}")


def run_fetch(client: FedClient, years: tuple[int, int], today: date) -> None:
    _, selection = _selection(client, years, today)
    counts = fetch_documents(client, selection.meetings)
    print(
        f"fetch: {counts.get('downloaded', 0)} pages downloaded, {counts.get('cached', 0)} already "
        f"cached under {client.raw_dir}"
    )


def run_parse(client: FedClient, paths: IngestPaths, years: tuple[int, int], today: date) -> int:
    entries, selection = _selection(client, years, today)
    result = build_documents(client, selection)
    report = build_report(
        entries, selection, result, start_year=years[0], end_year=years[1], today=today
    )
    write_meetings(paths, selection.meetings)
    write_documents(paths, result.documents)
    write_report(paths, report)
    counts = report["counts"]
    print(
        f"parse: {counts['meetings']} meetings, {counts['statements']} statements, "
        f"{counts['minutes']} minutes -> {paths.processed_dir}"
    )
    failed = failed_checks(report)
    for check in failed:
        print(f"  FAILED {check['name']}: {'; '.join(check['failures'][:5])}")
    print(f"  {len(report['checks']) - len(failed)}/{len(report['checks'])} checks passed")
    return 1 if failed else 0


def run_load(settings: Settings, paths: IngestPaths) -> int:
    if settings.database_url is None:
        print(f"load: {NO_DATABASE}", file=sys.stderr)
        return 2
    if not paths.report_file.is_file():
        print(
            f"load: refused: {paths.report_file} not found; run `make ingest` first.",
            file=sys.stderr,
        )
        return 1
    failed = failed_checks(json.loads(paths.report_file.read_text(encoding="utf-8")))
    if failed:
        names = "; ".join(check["name"] for check in failed)
        print(
            f"load: refused: the last parse failed {len(failed)} check(s): {names} "
            f"(see {paths.report_file})",
            file=sys.stderr,
        )
        return 1
    # Imported here: only this command needs the database layer.
    import psycopg

    from app.db.connection import connect
    from app.db.load import load_processed

    try:
        with connect(settings) as conn:
            counts = load_processed(conn, paths.processed_dir)
    except (ValueError, psycopg.Error) as exc:
        print(f"load: failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"load: upserted {counts['meetings']} meetings and {counts['documents']} documents "
        "into schema fedrag"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
