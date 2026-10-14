"""Tests for the Statement Comparison queries, the analysis cache and POST /api/compare against a
real Postgres."""

import threading
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import psycopg
import pytest
from fastapi.testclient import TestClient

from app.api.compare import get_analyzer
from app.compare.service import comparison_input
from app.compare.types import AnalysisRun, ComparisonInput, cache_key
from app.db.compare import (
    all_statements,
    nearest_statement_dates,
    read_comparison,
    save_comparison,
    stances_of,
    statements_on,
)
from app.db.connection import apply_schema
from app.db.load import load_processed
from app.db.stance import load_stance
from app.schemas.compare import ComparisonAnalysis, KeyChange
from app.stance.types import StatementPrediction

pytestmark = pytest.mark.db

Conn = psycopg.Connection[Any]
Row = dict[str, Any]
WriteProcessed = Callable[[list[Row], list[Row]], Path]
MakeClient = Callable[..., TestClient]

SEPTEMBER, NOVEMBER = date(2024, 9, 18), date(2024, 11, 7)
ANALYSIS = ComparisonAnalysis(
    summary="Another cut.",
    key_changes=[
        KeyChange(
            change_ids=[0],
            what_changed='"lowered" became "A further cut".',
            interpretation="Easing continues.",
            direction="dovish",
        )
    ],
    stance_shift="unchanged",
    confidence=0.6,
)


def _prediction(day: str, classifier: str, stance: str, score: float) -> StatementPrediction:
    return StatementPrediction(
        doc_id=f"statement-{day}",
        meeting_date=date.fromisoformat(day),
        classifier=classifier,  # type: ignore[arg-type]
        stance=stance,  # type: ignore[arg-type]
        score=score,
        confidence=0.7,
        model_version="test",
    )


@pytest.fixture
def loaded(
    db_conn: Conn,
    write_processed: WriteProcessed,
    sample_meetings: list[Row],
    sample_documents: list[Row],
) -> Conn:
    load_processed(db_conn, write_processed(sample_meetings, sample_documents))
    return db_conn


def _save(conn: Conn, key: str, analysis: ComparisonAnalysis = ANALYSIS, **fields: Any) -> None:
    row = {
        "cache_key": key,
        "base_doc_id": "statement-2024-09-18",
        "target_doc_id": "statement-2024-11-07",
        "model": "gpt-test",
        "prompt_version": "compare-test",
        "analysis": analysis,
        "usage": {"calls": 1, "usd": 0.002},
    }
    save_comparison(conn, **(row | fields))


def test_statements_on_reads_the_statement_not_the_minutes(loaded: Conn) -> None:
    found = statements_on(loaded, [SEPTEMBER, NOVEMBER, date(2024, 10, 1)])

    assert set(found) == {SEPTEMBER, NOVEMBER}
    september = found[SEPTEMBER]
    assert september.source.model_dump() == {
        "doc_id": "statement-2024-09-18",
        "doc_type": "statement",
        "meeting_date": SEPTEMBER,
        "release_date": SEPTEMBER,
        "page_updated": None,
        "title": "FOMC statement 2024-09-18",
        "url": "https://www.federalreserve.gov/newsevents/pressreleases/monetary20240918a.htm",
    }
    assert september.statement.paragraphs == (
        "The Committee lowered the target range.",
        "Voting for the action.",
    )
    stored = loaded.execute(
        "SELECT text, sha256 FROM fedrag.documents WHERE doc_id = 'statement-2024-09-18'"
    ).fetchone()
    assert (september.statement.text, september.statement.sha256) == (
        stored["text"],
        stored["sha256"],
    )
    assert statements_on(loaded, []) == {}


def test_all_statements_oldest_first(loaded: Conn) -> None:
    records = all_statements(loaded)

    assert [r.source.doc_id for r in records] == [
        "statement-2020-03-15",
        "statement-2024-09-18",
        "statement-2024-11-07",
    ]


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        ("2024-10-01", ("2024-09-18", "2024-11-07")),
        ("2024-09-18", ("2020-03-15", "2024-11-07")),
        ("2019-01-01", (None, "2020-03-15")),
        ("2025-01-01", ("2024-11-07", None)),
    ],
    ids=["between", "on-a-statement", "before-all", "after-all"],
)
def test_nearest_statement_dates(
    loaded: Conn, day: str, expected: tuple[str | None, str | None]
) -> None:
    nearest = nearest_statement_dates(loaded, date.fromisoformat(day))

    assert tuple(d.isoformat() if d else None for d in nearest) == expected


def test_nearest_without_statements(db_conn: Conn) -> None:
    apply_schema(db_conn)

    assert nearest_statement_dates(db_conn, SEPTEMBER) == (None, None)


def test_stances_of_one_classifier(loaded: Conn) -> None:
    load_stance(
        loaded,
        [
            _prediction("2024-09-18", "llm_zero_shot", "dovish", -0.6),
            _prediction("2024-09-18", "keyword", "neutral", 0.0),
            _prediction("2024-11-07", "keyword", "dovish", -0.3),
        ],
    )

    stances = stances_of(loaded, ["statement-2024-09-18", "statement-2024-11-07"], "llm_zero_shot")

    assert {doc: s.model_dump() for doc, s in stances.items()} == {
        "statement-2024-09-18": {"classifier": "llm_zero_shot", "stance": "dovish", "score": -0.6}
    }


def test_stances_of_without_the_stance_table(loaded: Conn) -> None:
    loaded.execute("DROP TABLE fedrag.stance_results")

    assert stances_of(loaded, ["statement-2024-09-18"], "llm_zero_shot") == {}


def test_save_then_read_and_replace(loaded: Conn) -> None:
    _save(loaded, "k1")
    first = read_comparison(loaded, "k1")
    revised = ANALYSIS.model_copy(update={"summary": "Revised.", "key_changes": []})
    _save(loaded, "k1", revised, usage={"calls": 2})
    second = read_comparison(loaded, "k1")

    assert first is not None and second is not None
    assert (first.base_doc_id, first.target_doc_id, first.model, first.prompt_version) == (
        "statement-2024-09-18",
        "statement-2024-11-07",
        "gpt-test",
        "compare-test",
    )
    assert ComparisonAnalysis.model_validate(first.analysis) == ANALYSIS
    assert first.usage == {"calls": 1, "usd": 0.002}
    assert ComparisonAnalysis.model_validate(second.analysis) == revised
    assert second.usage == {"calls": 2}
    assert second.created_at >= first.created_at
    count = loaded.execute("SELECT count(*) AS n FROM fedrag.statement_comparisons").fetchone()
    assert count["n"] == 1
    assert read_comparison(loaded, "missing") is None


def test_first_save_creates_the_table(loaded: Conn) -> None:
    # A database loaded before Phase 6 has no comparisons table: reads see nothing, the first
    # write creates it.
    loaded.execute("DROP TABLE fedrag.statement_comparisons")
    assert read_comparison(loaded, "k1") is None

    _save(loaded, "k1")

    assert read_comparison(loaded, "k1") is not None


def test_concurrent_saves_do_not_deadlock(loaded: Conn, test_database_url: str) -> None:
    # Writers in parallel, as make compare-precompute runs them: re-applying schema.sql in every
    # write (CREATE INDEX IF NOT EXISTS takes a SHARE lock) made them deadlock on Neon.
    _save(loaded, "warm-up")  # the table exists, as it does after the first write
    writers, rounds = 4, 5
    barrier = threading.Barrier(writers)
    errors: list[BaseException] = []

    def write(n: int) -> None:
        with psycopg.connect(test_database_url, autocommit=True, connect_timeout=5) as conn:
            for r in range(rounds):
                barrier.wait()
                try:
                    _save(conn, f"k{n}-{r}")
                except psycopg.Error as exc:
                    errors.append(exc)

    threads = [threading.Thread(target=write, args=(n,)) for n in range(writers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    count = loaded.execute("SELECT count(*) AS n FROM fedrag.statement_comparisons").fetchone()
    assert count["n"] == 1 + writers * rounds


def test_concurrent_first_saves_create_the_table_once(loaded: Conn, test_database_url: str) -> None:
    # Writers in parallel on a database without the table: each would apply schema.sql (the
    # CREATE TABLE ... REFERENCES fedrag.documents and CREATE INDEX IF NOT EXISTS locks), which
    # deadlocked until creating the table was serialized.
    writers = 4
    errors: list[BaseException] = []

    def write(key: str, barrier: threading.Barrier) -> None:
        with psycopg.connect(test_database_url, autocommit=True, connect_timeout=5) as conn:
            barrier.wait()
            try:
                _save(conn, key)
            except psycopg.Error as exc:
                errors.append(exc)

    for r in range(3):
        loaded.execute("DROP TABLE fedrag.statement_comparisons")
        barrier = threading.Barrier(writers)
        threads = [
            threading.Thread(target=write, args=(f"k{n}-{r}", barrier)) for n in range(writers)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert errors == []
        count = loaded.execute("SELECT count(*) AS n FROM fedrag.statement_comparisons").fetchone()
        assert count["n"] == writers


def test_save_rejects_unknown_documents(loaded: Conn) -> None:
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        _save(loaded, "k1", target_doc_id="statement-2030-01-01")

    assert read_comparison(loaded, "k1") is None


class FakeAnalyzer:
    model = "gpt-test"
    prompt_version = "compare-test"

    def __init__(self) -> None:
        self.calls: list[ComparisonInput] = []

    def __call__(self, comparison: ComparisonInput) -> AnalysisRun:
        self.calls.append(comparison)
        return AnalysisRun(ANALYSIS, {"calls": 1, "usd": 0.001})


def test_api_against_postgres(
    loaded: Conn, make_client: MakeClient, test_database_url: str
) -> None:
    load_stance(loaded, [_prediction("2024-11-07", "llm_zero_shot", "dovish", -0.4)])
    analyzer = FakeAnalyzer()
    body = {"base_date": "2024-09-18", "target_date": "2024-11-07"}

    with make_client(database_url=test_database_url, openai_api_key="sk-test") as client:
        client.app.dependency_overrides[get_analyzer] = lambda: analyzer
        before = client.post("/api/compare", json=body)
        generated = client.post("/api/compare", json=body | {"analysis": "generate"})
        cached = client.post("/api/compare", json=body)
        missing = client.post("/api/compare", json=body | {"target_date": "2024-10-01"})
    with make_client(database_url=test_database_url) as keyless:
        without_key = keyless.post("/api/compare", json=body | {"analysis": "generate"})

    assert before.status_code == 200
    assert before.json()["analysis_status"] == "not_cached"
    assert [c["kind"] for c in before.json()["changes"]] == ["removed", "removed", "added"]
    assert before.json()["stance"] == {
        "base": None,
        "target": {"classifier": "llm_zero_shot", "stance": "dovish", "score": -0.4},
    }
    assert generated.json()["analysis_status"] == cached.json()["analysis_status"] == "ready"
    assert cached.json()["analysis"] == ANALYSIS.model_dump(mode="json")
    assert len(analyzer.calls) == 1
    base, target = (
        statements_on(loaded, [SEPTEMBER, NOVEMBER])[d].statement for d in (SEPTEMBER, NOVEMBER)
    )
    key = cache_key(comparison_input(base, target), model="gpt-test", prompt_version="compare-test")
    stored = read_comparison(loaded, key)
    assert stored is not None and stored.usage == {"calls": 1, "usd": 0.001}
    assert missing.status_code == 404
    assert missing.json()["detail"] == (
        "No FOMC statement for a meeting on 2024-10-01; nearest statement dates: 2024-09-18, "
        "2024-11-07."
    )
    # The real analyzer without a key: the diff, and llm_not_configured (its model and prompt
    # version differ from the fake's, so the stored analysis is not this one's).
    assert without_key.status_code == 200
    assert without_key.json()["analysis_status"] == "llm_not_configured"
