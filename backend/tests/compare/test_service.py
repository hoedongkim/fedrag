"""Tests for the comparison service and its cache, with the database and the LLM faked out."""

import logging
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, date, datetime
from typing import Any

import httpx
import openai
import psycopg
import pytest

from app.compare import service
from app.compare.types import (
    DIFF_VERSION,
    AnalysisRun,
    ComparisonInput,
    StatementText,
    cache_key,
    restrict_to_changes,
)
from app.db import compare as compare_db
from app.db.compare import StatementRecord, StoredComparison
from app.rag.answer import LLMNotConfiguredError
from app.schemas.common import SourceDocument
from app.schemas.compare import (
    CompareRequest,
    ComparisonAnalysis,
    KeyChange,
    StanceSnapshot,
)

BASE_DAY, TARGET_DAY = date(2024, 7, 31), date(2024, 9, 18)
URL = "https://www.federalreserve.gov/newsevents/pressreleases/monetary{}a.htm"
BASE_PARAGRAPHS = (
    "Job gains have moderated, and the unemployment rate has moved up but remains low. "
    "Inflation remains somewhat elevated.",
    "The Committee decided to maintain the target range for the federal funds rate at 5-1/4 to "
    "5-1/2 percent.",
)
TARGET_PARAGRAPHS = (
    "Job gains have slowed, and the unemployment rate has moved up but remains low. "
    "Inflation remains somewhat elevated.",
    "The Committee decided to lower the target range for the federal funds rate by 1/2 "
    "percentage point to 4-3/4 to 5 percent.",
    "Voting against this action was Michelle W. Bowman.",
)
# The diff of the two: 0 modified, 1 unchanged, 2 modified, 3 added (procedural).
ANALYSIS = ComparisonAnalysis(
    summary="The Committee cut the target range and described job gains as slower.",
    key_changes=[
        KeyChange(
            change_ids=[2],
            what_changed='"maintain" became "lower ... by 1/2 percentage point".',
            interpretation="A half-point cut: easier policy.",
            direction="dovish",
        ),
        KeyChange(
            change_ids=[0],
            what_changed='"moderated" became "slowed".',
            interpretation="A softer labor market.",
            direction="dovish",
        ),
    ],
    stance_shift="more_dovish",
    confidence=0.8,
)
USAGE = {"calls": 1, "input_tokens": 900, "output_tokens": 300, "usd": 0.002}


def _record(day: date, paragraphs: tuple[str, ...]) -> StatementRecord:
    doc_id = f"statement-{day}"
    source = SourceDocument(
        doc_id=doc_id,
        doc_type="statement",
        meeting_date=day,
        release_date=day,
        page_updated=None,
        title=f"FOMC Statement, {day}",
        url=URL.format(day.strftime("%Y%m%d")),
    )
    return StatementRecord(source, StatementText(doc_id, day, day, paragraphs))


BASE, TARGET = _record(BASE_DAY, BASE_PARAGRAPHS), _record(TARGET_DAY, TARGET_PARAGRAPHS)
CONNECTION = object()


class FakeDatabase:
    """Lends a sentinel connection, counting those the current thread holds, or raises `error`."""

    def __init__(self) -> None:
        self.error: Exception | None = None
        self._held = threading.local()

    @property
    def open(self) -> int:
        return getattr(self._held, "count", 0)

    @contextmanager
    def connection(self) -> Iterator[Any]:
        if self.error:
            raise self.error
        self._held.count = self.open + 1
        try:
            yield CONNECTION
        finally:
            self._held.count -= 1


class FakeStore:
    """Stands in for app.db.compare: statements, stances and stored analyses in memory."""

    def __init__(self) -> None:
        self.statements = {BASE_DAY: BASE, TARGET_DAY: TARGET}
        self.stances = {
            BASE.source.doc_id: StanceSnapshot(
                classifier="llm_zero_shot", stance="neutral", score=0
            ),
            TARGET.source.doc_id: StanceSnapshot(
                classifier="llm_zero_shot", stance="dovish", score=-0.6
            ),
        }
        self.nearest: tuple[date | None, date | None] = (date(2024, 6, 12), date(2024, 7, 31))
        self.rows: dict[str, StoredComparison] = {}
        self.save_error: Exception | None = None
        self.stance_classifiers: list[str] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in (
            "statements_on",
            "nearest_statement_dates",
            "stances_of",
            "read_comparison",
            "save_comparison",
        ):
            monkeypatch.setattr(compare_db, name, getattr(self, name))

    def statements_on(self, conn: Any, dates: list[date]) -> dict[date, StatementRecord]:
        assert conn is CONNECTION
        return {day: self.statements[day] for day in dates if day in self.statements}

    def nearest_statement_dates(self, conn: Any, day: date) -> tuple[date | None, date | None]:
        assert conn is CONNECTION
        return self.nearest

    def stances_of(self, conn: Any, doc_ids: list[str], classifier: str) -> dict:
        self.stance_classifiers.append(classifier)
        return {doc_id: self.stances[doc_id] for doc_id in doc_ids if doc_id in self.stances}

    def read_comparison(self, conn: Any, key: str) -> StoredComparison | None:
        assert conn is CONNECTION
        return self.rows.get(key)

    def save_comparison(self, conn: Any, **row: Any) -> None:
        assert conn is CONNECTION
        if self.save_error:
            raise self.save_error
        analysis = row.pop("analysis").model_dump(mode="json")
        self.rows[row["cache_key"]] = StoredComparison(
            **row, analysis=analysis, created_at=datetime.now(UTC)
        )


class FakeAnalyzer:
    """Returns `analysis` (or raises `error`), checking that no connection is held meanwhile."""

    model = "gpt-test-2026-01-01"
    prompt_version = "compare-test.0000"

    def __init__(self, db: FakeDatabase, analysis: ComparisonAnalysis = ANALYSIS) -> None:
        self.db = db
        self.analysis = analysis
        self.error: Exception | None = None
        self.calls: list[ComparisonInput] = []
        self.before: Callable[[], None] = lambda: None

    def __call__(self, comparison: ComparisonInput) -> AnalysisRun:
        assert self.db.open == 0, "a connection is held during the LLM call"
        self.calls.append(comparison)
        self.before()
        if self.error:
            raise self.error
        return AnalysisRun(self.analysis, dict(USAGE))


@pytest.fixture
def db() -> FakeDatabase:
    return FakeDatabase()


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> FakeStore:
    fake = FakeStore()
    fake.install(monkeypatch)
    return fake


@pytest.fixture
def analyzer(db: FakeDatabase) -> FakeAnalyzer:
    return FakeAnalyzer(db)


@pytest.fixture(autouse=True)
def _fresh_locks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service, "_locks", {})


def _request(mode: str = "cached_only", **fields: Any) -> CompareRequest:
    return CompareRequest.model_validate(
        {"base_date": BASE_DAY, "target_date": TARGET_DAY, "analysis": mode} | fields
    )


def _compare(db: FakeDatabase, analyzer: FakeAnalyzer, mode: str = "cached_only", llm: bool = True):
    return service.compare_statements(db, _request(mode), analyzer, llm_configured=llm)


def _key() -> str:
    comparison = service.comparison_input(BASE.statement, TARGET.statement)
    return cache_key(
        comparison, model=FakeAnalyzer.model, prompt_version=FakeAnalyzer.prompt_version
    )


# --- the diff, the documents and the stance ----------------------------------------------------


def test_response_holds_diff_documents_and_stance(
    db: FakeDatabase, store: FakeStore, analyzer: FakeAnalyzer
) -> None:
    response = _compare(db, analyzer)

    assert (response.base, response.target) == (BASE.source, TARGET.source)
    assert [(c.id, c.kind, c.procedural) for c in response.changes] == [
        (0, "modified", False),
        (1, "unchanged", False),
        (2, "modified", False),
        (3, "added", True),
    ]
    assert response.changes[0].similarity == 0.9286
    assert [s.op for s in response.changes[0].segments or []] == [
        "equal",
        "delete",
        "insert",
        "equal",
    ]
    assert response.changes[1].segments is None and response.changes[3].similarity is None
    assert response.stats.model_dump(exclude={"similarity"}) == {
        "unchanged": 1,
        "modified": 2,
        "added": 1,
        "removed": 0,
    }
    assert response.stance.base == store.stances[BASE.source.doc_id]
    assert response.stance.target is not None and response.stance.target.score == -0.6
    assert store.stance_classifiers == ["llm_zero_shot"]
    assert db.open == 0


def test_missing_stance_is_null(db: FakeDatabase, store: FakeStore, analyzer: FakeAnalyzer) -> None:
    store.stances.pop(TARGET.source.doc_id)

    stance = _compare(db, analyzer).stance

    assert stance.base is not None and stance.target is None


@pytest.mark.parametrize(
    ("missing", "nearest", "message"),
    [
        (
            [TARGET_DAY],
            (date(2024, 7, 31), date(2024, 11, 7)),
            "No FOMC statement for a meeting on 2024-09-18; nearest statement dates: "
            "2024-07-31, 2024-11-07.",
        ),
        (
            [TARGET_DAY],
            (date(2024, 7, 31), None),
            "No FOMC statement for a meeting on 2024-09-18; nearest statement date: 2024-07-31.",
        ),
        (
            [BASE_DAY, TARGET_DAY],
            (None, None),
            "No FOMC statement for a meeting on 2024-07-31: no statements are loaded. "
            "No FOMC statement for a meeting on 2024-09-18: no statements are loaded.",
        ),
    ],
    ids=["both-sides", "only-before", "none-loaded"],
)
def test_missing_statement_names_the_nearest_dates(
    db: FakeDatabase,
    store: FakeStore,
    analyzer: FakeAnalyzer,
    missing: list[date],
    nearest: tuple[date | None, date | None],
    message: str,
) -> None:
    for day in missing:
        store.statements.pop(day)
    store.nearest = nearest

    with pytest.raises(service.StatementNotFoundError) as caught:
        _compare(db, analyzer, "generate")

    assert str(caught.value) == message
    assert analyzer.calls == []


# --- analysis status ---------------------------------------------------------------------------


def test_cached_only_without_a_stored_analysis_is_not_cached(
    db: FakeDatabase, store: FakeStore, analyzer: FakeAnalyzer
) -> None:
    response = _compare(db, analyzer)

    assert (response.analysis, response.analysis_status) == (None, "not_cached")
    assert response.analysis_model_version is None
    assert analyzer.calls == [] and store.rows == {}


@pytest.mark.parametrize("mode", ["cached_only", "generate"])
def test_without_llm_and_nothing_stored_is_llm_not_configured(
    db: FakeDatabase, store: FakeStore, analyzer: FakeAnalyzer, mode: str
) -> None:
    response = _compare(db, analyzer, mode, llm=False)

    assert (response.analysis, response.analysis_status) == (None, "llm_not_configured")
    assert analyzer.calls == []
    assert len(response.changes) == 4  # the diff is still there


@pytest.mark.parametrize("mode", ["cached_only", "generate"])
@pytest.mark.parametrize("llm", [True, False], ids=["with-key", "without-key"])
def test_the_demo_never_writes_an_analysis(
    db: FakeDatabase, store: FakeStore, analyzer: FakeAnalyzer, mode: str, llm: bool
) -> None:
    response = service.compare_statements(
        db, _request(mode), analyzer, llm_configured=llm, demo=True
    )

    assert (response.analysis, response.analysis_status) == (None, "disabled_on_demo")
    assert analyzer.calls == [] and store.rows == {}
    assert len(response.changes) == 4  # the diff is still there


def test_the_demo_serves_a_stored_analysis(
    db: FakeDatabase, store: FakeStore, analyzer: FakeAnalyzer
) -> None:
    _compare(db, analyzer, "generate")  # stored by a local run, e.g. make compare-precompute

    response = service.compare_statements(
        db, _request("generate"), analyzer, llm_configured=False, demo=True
    )

    assert (response.analysis_status, response.analysis) == ("ready", ANALYSIS)
    assert len(analyzer.calls) == 1


def test_generate_calls_once_stores_and_then_serves_the_cache(
    db: FakeDatabase, store: FakeStore, analyzer: FakeAnalyzer
) -> None:
    first = _compare(db, analyzer, "generate")
    second = _compare(db, analyzer, "generate")
    cached = _compare(db, analyzer, "cached_only")

    assert len(analyzer.calls) == 1
    comparison = analyzer.calls[0]
    assert (comparison.base, comparison.target) == (BASE.statement, TARGET.statement)
    assert [c.id for c in comparison.changed] == [0, 2, 3]
    assert comparison.as_of == TARGET_DAY
    for response in (first, second, cached):
        assert response.analysis_status == "ready"
        assert response.analysis == ANALYSIS
        assert response.analysis_model_version == "gpt-test-2026-01-01/compare-test.0000"
    [(key, row)] = store.rows.items()
    assert key == _key()
    assert (row.base_doc_id, row.target_doc_id) == (BASE.source.doc_id, TARGET.source.doc_id)
    assert (row.model, row.prompt_version) == (analyzer.model, analyzer.prompt_version)
    assert row.usage == USAGE
    assert ComparisonAnalysis.model_validate(row.analysis) == ANALYSIS


def test_stored_analysis_is_served_without_an_llm(
    db: FakeDatabase, store: FakeStore, analyzer: FakeAnalyzer
) -> None:
    _compare(db, analyzer, "generate")

    response = _compare(db, analyzer, "cached_only", llm=False)

    assert (response.analysis_status, response.analysis) == ("ready", ANALYSIS)
    assert len(analyzer.calls) == 1


def test_analysis_of_another_model_or_prompt_is_not_reused(
    db: FakeDatabase, store: FakeStore, analyzer: FakeAnalyzer
) -> None:
    _compare(db, analyzer, "generate")
    other = FakeAnalyzer(db)
    other.prompt_version = "compare-test.1111"

    assert _compare(db, other).analysis_status == "not_cached"


@pytest.mark.parametrize(
    "error",
    [
        openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.com")),
        openai.AuthenticationError(
            "Incorrect API key provided: sk-proj-SECRET",
            response=httpx.Response(401, request=httpx.Request("POST", "https://x")),
            body=None,
        ),
    ],
    ids=["connection", "auth"],
)
def test_openai_failure_is_failed_and_never_logs_the_message(
    db: FakeDatabase,
    store: FakeStore,
    analyzer: FakeAnalyzer,
    error: Exception,
    caplog: pytest.LogCaptureFixture,
) -> None:
    analyzer.error = error

    with caplog.at_level(logging.WARNING):
        response = _compare(db, analyzer, "generate")

    assert (response.analysis, response.analysis_status) == (None, "failed")
    assert len(response.changes) == 4
    assert store.rows == {}
    assert type(error).__name__ in caplog.text
    assert "SECRET" not in caplog.text


def test_other_analyzer_errors_are_failed(
    db: FakeDatabase, store: FakeStore, analyzer: FakeAnalyzer, caplog: pytest.LogCaptureFixture
) -> None:
    analyzer.error = RuntimeError("3 replies in a row could not be parsed")

    with caplog.at_level(logging.ERROR):
        response = _compare(db, analyzer, "generate")

    assert response.analysis_status == "failed"
    assert "Comparison analysis failed" in caplog.text
    assert store.rows == {}


def test_analyzer_without_key_is_llm_not_configured(
    db: FakeDatabase, store: FakeStore, analyzer: FakeAnalyzer
) -> None:
    analyzer.error = LLMNotConfiguredError()

    assert _compare(db, analyzer, "generate").analysis_status == "llm_not_configured"


def test_failed_store_still_returns_the_analysis(
    db: FakeDatabase, store: FakeStore, analyzer: FakeAnalyzer, caplog: pytest.LogCaptureFixture
) -> None:
    store.save_error = psycopg.OperationalError("server closed the connection")

    with caplog.at_level(logging.WARNING):
        response = _compare(db, analyzer, "generate")

    assert (response.analysis_status, response.analysis) == ("ready", ANALYSIS)
    assert "not stored (OperationalError)" in caplog.text


def test_invalid_stored_analysis_is_regenerated(
    db: FakeDatabase, store: FakeStore, analyzer: FakeAnalyzer, caplog: pytest.LogCaptureFixture
) -> None:
    _compare(db, analyzer, "generate")
    key = _key()
    store.rows[key] = replace(store.rows[key], analysis={"summary": "no other fields"})

    with caplog.at_level(logging.WARNING):
        cached = _compare(db, analyzer)
        regenerated = _compare(db, analyzer, "generate")

    assert cached.analysis_status == "not_cached"
    assert "is invalid" in caplog.text
    assert regenerated.analysis == ANALYSIS
    assert len(analyzer.calls) == 2
    assert ComparisonAnalysis.model_validate(store.rows[key].analysis) == ANALYSIS


def test_key_changes_cite_only_changed_items(
    db: FakeDatabase, store: FakeStore, analyzer: FakeAnalyzer
) -> None:
    sloppy = ANALYSIS.model_copy(
        update={
            "key_changes": [
                ANALYSIS.key_changes[0].model_copy(update={"change_ids": [2, 99, 2, 3]}),
                ANALYSIS.key_changes[1].model_copy(update={"change_ids": [1]}),  # unchanged
            ]
        }
    )
    analyzer.analysis = sloppy

    generated = _compare(db, analyzer, "generate").analysis
    store.rows[_key()] = replace(store.rows[_key()], analysis=sloppy.model_dump(mode="json"))
    cached = _compare(db, analyzer).analysis

    for analysis in (generated, cached):
        assert analysis is not None
        assert [k.change_ids for k in analysis.key_changes] == [[2, 3]]


def test_concurrent_requests_for_a_pair_pay_once(
    db: FakeDatabase, store: FakeStore, analyzer: FakeAnalyzer, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The first request's call is held until the other two have also found nothing stored, so
    # without the per-key lock each of them would call the LLM too.
    started, release, others_checked = threading.Event(), threading.Event(), threading.Event()
    reads: list[str] = []
    read = store.read_comparison

    def counting_read(conn: Any, key: str) -> StoredComparison | None:
        reads.append(key)
        if len(reads) == 4:  # the first request's two reads, then one by each of the others
            others_checked.set()
        return read(conn, key)

    def slow() -> None:
        started.set()
        assert release.wait(5)

    monkeypatch.setattr(compare_db, "read_comparison", counting_read)
    analyzer.before = slow
    results: list[str] = []

    def request() -> None:
        results.append(_compare(db, analyzer, "generate").analysis_status)

    threads = [threading.Thread(target=request) for _ in range(3)]
    threads[0].start()
    assert started.wait(5)
    for thread in threads[1:]:
        thread.start()
    assert others_checked.wait(5)
    release.set()
    for thread in threads:
        thread.join(5)

    assert results == ["ready"] * 3
    assert len(analyzer.calls) == 1


def test_database_errors_propagate(
    db: FakeDatabase, store: FakeStore, analyzer: FakeAnalyzer
) -> None:
    db.error = psycopg.OperationalError("unreachable")

    with pytest.raises(psycopg.OperationalError):
        _compare(db, analyzer, "generate")


# --- cache key and id checks -------------------------------------------------------------------


def test_cache_key_covers_ids_texts_diff_prompt_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    comparison = service.comparison_input(BASE.statement, TARGET.statement)
    key = cache_key(comparison, model="m", prompt_version="p")
    again = service.comparison_input(BASE.statement, TARGET.statement)
    assert len(key) == 64 and key == cache_key(again, model="m", prompt_version="p")
    revised = replace(TARGET.statement, paragraphs=(*TARGET_PARAGRAPHS[:2], "Revised."))
    variants = [
        cache_key(comparison, model="m2", prompt_version="p"),
        cache_key(comparison, model="m", prompt_version="p2"),
        cache_key(replace(comparison, target=revised), model="m", prompt_version="p"),
        cache_key(
            replace(comparison, base=replace(BASE.statement, doc_id="statement-x")),
            model="m",
            prompt_version="p",
        ),
    ]
    monkeypatch.setattr("app.compare.types.DIFF_VERSION", DIFF_VERSION + "-next")
    variants.append(cache_key(comparison, model="m", prompt_version="p"))

    assert len({key, *variants}) == len(variants) + 1


def test_restrict_to_changes_drops_key_changes_without_valid_ids() -> None:
    comparison = service.comparison_input(BASE.statement, TARGET.statement)
    analysis = ANALYSIS.model_copy(
        update={
            "key_changes": [
                ANALYSIS.key_changes[0].model_copy(update={"change_ids": [7, 1]}),
                ANALYSIS.key_changes[1].model_copy(update={"change_ids": [3, 0, 3]}),
            ]
        }
    )

    restricted = restrict_to_changes(analysis, comparison.changes)

    assert [k.change_ids for k in restricted.key_changes] == [[3, 0]]
    assert restricted.summary == ANALYSIS.summary
