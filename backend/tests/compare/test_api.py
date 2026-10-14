"""Tests for POST /api/compare with the database and the LLM faked out (Postgres in tests/db)."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime
from typing import Any

import psycopg
import pytest
from fastapi.testclient import TestClient

from app.api.compare import get_analyzer
from app.api.database import NOT_LOADED, get_db
from app.compare.types import AnalysisRun, ComparisonInput, StatementText
from app.db import compare as compare_db
from app.db.compare import StatementRecord, StoredComparison
from app.db.connection import NOT_CONFIGURED
from app.schemas.common import SourceDocument
from app.schemas.compare import ComparisonAnalysis, KeyChange, StanceSnapshot

MakeClient = Callable[..., TestClient]

BASE_DAY, TARGET_DAY = date(2024, 7, 31), date(2024, 9, 18)
BODY = {"base_date": "2024-07-31", "target_date": "2024-09-18"}
PARAGRAPHS = {
    BASE_DAY: (
        "The Committee decided to maintain the target range for the federal funds rate at 5-1/4 "
        "to 5-1/2 percent.",
    ),
    TARGET_DAY: (
        "The Committee decided to lower the target range for the federal funds rate by 1/2 "
        "percentage point to 4-3/4 to 5 percent.",
        "Voting against this action was Michelle W. Bowman.",
    ),
}
ANALYSIS = ComparisonAnalysis(
    summary="A half-point cut.",
    key_changes=[
        KeyChange(
            change_ids=[0],
            what_changed='"maintain" became "lower".',
            interpretation="Easier policy.",
            direction="dovish",
        )
    ],
    stance_shift="more_dovish",
    confidence=0.9,
)
CONNECTION = object()


def _record(day: date) -> StatementRecord:
    doc_id = f"statement-{day}"
    source = SourceDocument(
        doc_id=doc_id,
        doc_type="statement",
        meeting_date=day,
        release_date=day,
        page_updated=None,
        title=f"FOMC Statement, {day}",
        url=f"https://www.federalreserve.gov/newsevents/pressreleases/monetary{day:%Y%m%d}a.htm",
    )
    return StatementRecord(source, StatementText(doc_id, day, day, PARAGRAPHS[day]))


class FakeDatabase:
    @contextmanager
    def connection(self) -> Iterator[Any]:
        yield CONNECTION


class FakeStore:
    """Stands in for app.db.compare; `error` is raised by the statement lookup."""

    def __init__(self) -> None:
        self.statements = {day: _record(day) for day in PARAGRAPHS}
        self.rows: dict[str, StoredComparison] = {}
        self.error: Exception | None = None

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(compare_db, "statements_on", self.statements_on)
        monkeypatch.setattr(
            compare_db, "nearest_statement_dates", lambda conn, day: (BASE_DAY, TARGET_DAY)
        )
        monkeypatch.setattr(compare_db, "stances_of", self.stances_of)
        monkeypatch.setattr(compare_db, "read_comparison", lambda conn, key: self.rows.get(key))
        monkeypatch.setattr(compare_db, "save_comparison", self.save_comparison)

    def statements_on(self, conn: Any, dates: list[date]) -> dict[date, StatementRecord]:
        if self.error:
            raise self.error
        return {day: self.statements[day] for day in dates if day in self.statements}

    def stances_of(self, conn: Any, doc_ids: list[str], classifier: str) -> dict:
        snapshot = StanceSnapshot(classifier=classifier, stance="neutral", score=0.05)  # type: ignore[arg-type]
        return {doc_ids[0]: snapshot}

    def save_comparison(self, conn: Any, **row: Any) -> None:
        analysis = row.pop("analysis").model_dump(mode="json")
        self.rows[row["cache_key"]] = StoredComparison(
            **row, analysis=analysis, created_at=datetime.now(UTC)
        )


class FakeAnalyzer:
    model = "gpt-test"
    prompt_version = "compare-test"

    def __init__(self) -> None:
        self.calls: list[ComparisonInput] = []

    def __call__(self, comparison: ComparisonInput) -> AnalysisRun:
        self.calls.append(comparison)
        return AnalysisRun(ANALYSIS, {"calls": 1})


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> FakeStore:
    fake = FakeStore()
    fake.install(monkeypatch)
    return fake


@pytest.fixture
def analyzer() -> FakeAnalyzer:
    return FakeAnalyzer()


@pytest.fixture
def api(make_client: MakeClient, analyzer: FakeAnalyzer) -> TestClient:
    """A client with a database and an LLM key, the analyzer faked."""
    client = make_client(openai_api_key="sk-test")
    client.app.dependency_overrides[get_db] = lambda: FakeDatabase()
    client.app.dependency_overrides[get_analyzer] = lambda: analyzer
    return client


def test_503_without_database(client: TestClient) -> None:
    response = client.post("/api/compare", json=BODY | {"analysis": "generate"})

    assert response.status_code == 503
    assert response.json() == {"detail": NOT_CONFIGURED}


def test_response_shape(api: TestClient, store: FakeStore) -> None:
    response = api.post("/api/compare", json=BODY)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "base",
        "target",
        "changes",
        "stats",
        "analysis",
        "analysis_status",
        "analysis_model_version",
        "stance",
    }
    assert body["base"]["doc_id"] == "statement-2024-07-31"
    assert body["target"]["meeting_date"] == "2024-09-18"
    modified, added = body["changes"]
    assert set(modified) == {
        "id",
        "kind",
        "base",
        "target",
        "similarity",
        "procedural",
        "segments",
    }
    assert (modified["id"], modified["kind"], modified["procedural"]) == (0, "modified", False)
    assert modified["segments"][1] == {"op": "delete", "text": "maintain"}
    assert (
        "".join(s["text"] for s in modified["segments"] if s["op"] != "delete")
        == (PARAGRAPHS[TARGET_DAY][0])
    )
    assert added == {
        "id": 1,
        "kind": "added",
        "base": None,
        "target": "Voting against this action was Michelle W. Bowman.",
        "similarity": None,
        "procedural": True,
        "segments": None,
    }
    assert body["stats"] == {
        "unchanged": 0,
        "modified": 1,
        "added": 1,
        "removed": 0,
        "similarity": body["stats"]["similarity"],
    }
    assert body["stance"] == {
        "base": {"classifier": "llm_zero_shot", "stance": "neutral", "score": 0.05},
        "target": None,
    }
    assert (body["analysis"], body["analysis_status"]) == (None, "not_cached")
    assert body["analysis_model_version"] is None


def test_generate_writes_once_then_serves_the_cache(
    api: TestClient, store: FakeStore, analyzer: FakeAnalyzer
) -> None:
    generated = api.post("/api/compare", json=BODY | {"analysis": "generate"}).json()
    cached = api.post("/api/compare", json=BODY).json()

    assert len(analyzer.calls) == 1 and len(store.rows) == 1
    for body in (generated, cached):
        assert body["analysis_status"] == "ready"
        assert body["analysis"] == ANALYSIS.model_dump(mode="json")
        assert body["analysis_model_version"] == "gpt-test/compare-test"


@pytest.mark.parametrize("mode", ["cached_only", "generate"])
def test_without_key_the_diff_is_still_returned(
    make_client: MakeClient, store: FakeStore, mode: str
) -> None:
    client = make_client()  # no OPENAI_API_KEY; the real analyzer dependency
    client.app.dependency_overrides[get_db] = lambda: FakeDatabase()

    response = client.post("/api/compare", json=BODY | {"analysis": mode})

    assert response.status_code == 200
    assert response.json()["analysis_status"] == "llm_not_configured"
    assert len(response.json()["changes"]) == 2


@pytest.mark.parametrize("mode", ["cached_only", "generate"])
def test_the_demo_returns_the_diff_and_writes_no_analysis(
    make_client: MakeClient, store: FakeStore, analyzer: FakeAnalyzer, mode: str
) -> None:
    client = make_client(openai_api_key="sk-test", public_demo=True)
    client.app.dependency_overrides[get_db] = lambda: FakeDatabase()
    client.app.dependency_overrides[get_analyzer] = lambda: analyzer

    response = client.post("/api/compare", json=BODY | {"analysis": mode})

    assert response.status_code == 200
    assert (response.json()["analysis"], response.json()["analysis_status"]) == (
        None,
        "disabled_on_demo",
    )
    assert len(response.json()["changes"]) == 2
    assert analyzer.calls == [] and store.rows == {}


def test_the_demo_serves_stored_analyses(
    api: TestClient, make_client: MakeClient, store: FakeStore, analyzer: FakeAnalyzer
) -> None:
    api.post("/api/compare", json=BODY | {"analysis": "generate"})  # stored by a local run
    demo = make_client(public_demo=True)
    demo.app.dependency_overrides[get_db] = lambda: FakeDatabase()
    demo.app.dependency_overrides[get_analyzer] = lambda: analyzer

    body = demo.post("/api/compare", json=BODY | {"analysis": "generate"}).json()

    assert (body["analysis_status"], body["analysis"]) == (
        "ready",
        ANALYSIS.model_dump(mode="json"),
    )
    assert len(analyzer.calls) == 1  # the local run's


def test_unknown_meeting_is_404_with_nearest_dates(api: TestClient, store: FakeStore) -> None:
    response = api.post("/api/compare", json=BODY | {"target_date": "2024-09-19"})

    assert response.status_code == 404
    assert response.json() == {
        "detail": "No FOMC statement for a meeting on 2024-09-19; nearest statement dates: "
        "2024-07-31, 2024-09-18."
    }


def test_unloaded_database_is_503_db_load(api: TestClient, store: FakeStore) -> None:
    store.error = psycopg.errors.UndefinedTable('relation "fedrag.documents" does not exist')

    response = api.post("/api/compare", json=BODY)

    assert response.status_code == 503
    assert response.json() == {"detail": NOT_LOADED}


@pytest.mark.parametrize(
    "body",
    [
        BODY | {"doc_type": "statement"},
        BODY | {"analysis": "always"},
        BODY | {"target_date": "2024-07-31"},
        {"base_date": "2024-09-18", "target_date": "2024-07-31"},
        {"base_date": "2024-07-31"},
        BODY | {"base_date": "2024-02-30"},
    ],
    ids=["doc-type-dropped", "unknown-mode", "same-date", "reversed", "missing-target", "bad-date"],
)
def test_invalid_request_is_422_before_the_database(
    api: TestClient, store: FakeStore, body: dict[str, Any]
) -> None:
    store.error = AssertionError("the database was queried")

    assert api.post("/api/compare", json=body).status_code == 422


def test_openapi_documents_the_contract(client: TestClient) -> None:
    schema = client.app.openapi()
    operation = schema["paths"]["/api/compare"]["post"]
    components = schema["components"]["schemas"]
    error = {"$ref": "#/components/schemas/ErrorResponse"}

    for code in ("404", "503"):
        assert operation["responses"][code]["content"]["application/json"]["schema"] == error
    assert "501" not in operation["responses"]
    request = components["CompareRequest"]
    assert set(request["properties"]) == {"base_date", "target_date", "analysis"}
    assert request["properties"]["analysis"]["default"] == "cached_only"
    assert set(components["CompareResponse"]["required"]) == {
        "base",
        "target",
        "changes",
        "stats",
        "analysis",
        "analysis_status",
        "analysis_model_version",
        "stance",
    }
    assert set(components["SentenceChange"]["required"]) == {
        "id",
        "kind",
        "base",
        "target",
        "similarity",
        "procedural",
        "segments",
    }
