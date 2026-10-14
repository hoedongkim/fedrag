"""Tests for python -m app.compare with the database and the LLM faked out (no network)."""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import psycopg
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import Field

import app.compare.analysis as analysis_module
import app.db.connection as connection_module
from app.compare.__main__ import main
from app.compare.analysis import LLMAnalyzer
from app.compare.prompts import ComparisonJudgment, KeyChangeJudgment
from app.compare.service import comparison_input
from app.compare.types import StatementText, cache_key
from app.core.config import Settings
from app.db import compare as compare_db
from app.db.compare import StatementRecord, StoredComparison
from app.schemas.common import SourceDocument
from app.schemas.compare import ComparisonAnalysis, StanceSnapshot

TEXTS = {
    date(2024, 6, 12): (
        "Job gains have remained strong. The Committee decided to maintain the target range for "
        "the federal funds rate at 5-1/4 to 5-1/2 percent."
    ),
    date(2024, 7, 31): (
        "Job gains have moderated. The Committee decided to maintain the target range for the "
        "federal funds rate at 5-1/4 to 5-1/2 percent."
    ),
    date(2024, 9, 18): (
        "Job gains have slowed. The Committee decided to lower the target range for the federal "
        "funds rate by 1/2 percentage point. Voting against this action was Michelle W. Bowman."
    ),
}
USAGE = {
    "input_tokens": 2_000,
    "output_tokens": 500,
    "total_tokens": 2_500,
    "output_token_details": {"reasoning": 50},
}


def _record(day: date, text: str) -> StatementRecord:
    doc_id = f"statement-{day}"
    source = SourceDocument(
        doc_id=doc_id,
        doc_type="statement",
        meeting_date=day,
        release_date=day,
        page_updated=None,
        title=f"FOMC Statement, {day}",
        url=f"https://www.federalreserve.gov/{doc_id}",
    )
    return StatementRecord(source, StatementText(doc_id, day, day, (text,)))


RECORDS = [_record(day, text) for day, text in sorted(TEXTS.items())]


class FakeLLM(BaseChatModel):
    """Answers every structured call with a cut citing the target's rate sentence, counting
    calls."""

    prompts: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-structured"

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        raise NotImplementedError

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable[Any, Any]:
        def run(messages: list[BaseMessage]) -> dict[str, Any]:
            self.prompts.append(messages)
            parsed = ComparisonJudgment(
                key_changes=[
                    KeyChangeJudgment(
                        change_ids=[0, 1],
                        what_changed='"maintain" became "lower".',
                        interpretation="Easier policy.",
                        direction="dovish",
                    )
                ],
                summary="A cut.",
                stance_shift="more_dovish",
                confidence=0.9,
            )
            raw = AIMessage(content="{}", usage_metadata=USAGE)
            return {"raw": raw, "parsed": parsed, "parsing_error": None}

        return RunnableLambda(run)


class FakeConnection:
    @contextmanager
    def transaction(self) -> Iterator[None]:
        yield

    def __enter__(self) -> "FakeConnection":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class FakeDatabase:
    def __init__(self, url: Any = None) -> None:
        self.closed = False

    @contextmanager
    def connection(self) -> Iterator[FakeConnection]:
        yield FakeConnection()

    def close(self) -> None:
        self.closed = True


class Store:
    """Stored analyses in memory, the schema applications and the fake LLM's calls."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
        self.rows: dict[str, StoredComparison] = {}
        self.schema_applied = 0
        self.llm = FakeLLM()
        snapshot = StanceSnapshot(classifier="llm_zero_shot", stance="neutral", score=0.1)
        monkeypatch.setattr(compare_db, "all_statements", lambda conn: list(RECORDS))
        by_day = {record.source.meeting_date: record for record in RECORDS}
        monkeypatch.setattr(
            compare_db,
            "statements_on",
            lambda conn, days: {day: by_day[day] for day in days if day in by_day},
        )
        monkeypatch.setattr(compare_db, "read_comparison", lambda conn, key: self.rows.get(key))
        monkeypatch.setattr(compare_db, "save_comparison", self.save)
        monkeypatch.setattr(
            compare_db, "stances_of", lambda conn, ids, name: {doc: snapshot for doc in ids}
        )
        monkeypatch.setattr(connection_module, "connect", lambda settings: FakeConnection())
        monkeypatch.setattr(connection_module, "apply_schema", self.apply_schema)
        monkeypatch.setattr(connection_module, "Database", FakeDatabase)
        monkeypatch.setattr(
            analysis_module,
            "get_analyzer",
            lambda settings, **kwargs: LLMAnalyzer(settings, llm=self.llm, **kwargs),
        )
        self.settings = settings

    def apply_schema(self, conn: Any) -> None:
        self.schema_applied += 1

    def save(self, conn: Any, *, cache_key: str, analysis: ComparisonAnalysis, **row: Any) -> None:
        self.rows[cache_key] = StoredComparison(
            cache_key=cache_key,
            analysis=analysis.model_dump(mode="json"),
            created_at=datetime.now(UTC),
            **row,
        )

    def store(self, base: StatementRecord, target: StatementRecord) -> str:
        """Store an analysis for the pair under its current cache key, as an earlier run would."""
        analyzer = LLMAnalyzer(self.settings)
        key = cache_key(
            comparison_input(base.statement, target.statement),
            model=analyzer.model,
            prompt_version=analyzer.prompt_version,
        )
        analysis = ComparisonAnalysis(
            summary="Stored earlier.", key_changes=[], stance_shift="unchanged", confidence=0.7
        )
        self.save(
            None,
            cache_key=key,
            analysis=analysis,
            base_doc_id=base.source.doc_id,
            target_doc_id=target.source.doc_id,
            model=analyzer.model,
            prompt_version=analyzer.prompt_version,
            usage={"calls": 1},
        )
        return key


def _settings(tmp_path: Path, key: str | None = "sk-test") -> Settings:
    return Settings(
        _env_file=None, openai_api_key=key, database_url="postgresql://x@h/db", data_dir=tmp_path
    )


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Store:
    return Store(monkeypatch, _settings(tmp_path))


def test_precompute_estimates_first_and_analyses_only_the_pairs_not_stored(
    store: Store, capsys: pytest.CaptureFixture[str]
) -> None:
    store.store(RECORDS[0], RECORDS[1])

    assert main(["precompute"], settings=store.settings) == 0

    out = capsys.readouterr().out
    assert "3 statements, 2 consecutive pairs; 1 already stored" in out
    assert "1 calls for 1 uncached of 1 items" in out and "limit --max-usd 1.00" in out
    written = "Wrote 1 analyses (more_dovish 1); 1 of 1 pairs now stored"
    assert out.index("OpenAI estimate") < out.index(written)
    assert "OpenAI: 1 calls, 2,000 input" in out
    assert len(store.llm.prompts) == 1 and len(store.rows) == 2 and store.schema_applied == 1
    stored = next(row for row in store.rows.values() if row.base_doc_id == "statement-2024-07-31")
    assert stored.usage["calls"] == 1 and stored.analysis["stance_shift"] == "more_dovish"

    assert main(["precompute"], settings=store.settings) == 0  # everything stored: no calls
    assert "2 already stored" in capsys.readouterr().out and len(store.llm.prompts) == 1


def test_precompute_counts_what_was_stored_not_what_was_written(
    store: Store, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def deadlock(conn: Any, **row: Any) -> None:
        raise psycopg.errors.DeadlockDetected("deadlock detected")

    monkeypatch.setattr(compare_db, "save_comparison", deadlock)

    assert main(["precompute"], settings=store.settings) == 1

    captured = capsys.readouterr()
    assert "Wrote 2 analyses (more_dovish 2); 0 of 2 pairs now stored" in captured.out
    assert "2 not stored ({'ready': 2}); run again to retry them" in captured.err


def test_precompute_refuses_an_estimate_over_max_usd(
    store: Store, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["precompute", "--max-usd", "0.0001"], settings=store.settings) == 1

    assert "refused" in capsys.readouterr().err
    assert store.llm.prompts == [] and store.rows == {}


def test_precompute_dry_run_and_missing_key_call_nothing(
    store: Store, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["precompute", "--dry-run"], settings=store.settings) == 0
    assert "2 calls for 2 uncached" in capsys.readouterr().out
    assert store.schema_applied == 0 and store.rows == {}

    assert main(["precompute"], settings=_settings(tmp_path, key=None)) == 1
    assert "OPENAI_API_KEY" in capsys.readouterr().err
    assert store.llm.prompts == [] and store.rows == {}


def test_show_prints_the_diff_and_the_stored_analysis(
    store: Store, capsys: pytest.CaptureFixture[str]
) -> None:
    store.store(RECORDS[1], RECORDS[2])

    assert main(["show", "2024-07-31", "2024-09-18"], settings=store.settings) == 0

    out = capsys.readouterr().out
    assert "Base:   FOMC Statement, 2024-07-31" in out
    assert "Job gains have [-moderated.-]{+slowed.+}" in out
    assert "added, procedural\n  + Voting against this action was Michelle W. Bowman." in out
    assert "unchanged, confidence 0.70 (model-reported, not calibrated)" in out
    assert "Stance (llm_zero_shot, each statement scored alone): neutral +0.10" in out
    assert store.llm.prompts == []


def test_show_without_a_stored_analysis_calls_only_with_generate(
    store: Store, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["show", "2024-06-12", "2024-07-31"], settings=store.settings) == 0
    assert "Analysis: not_cached" in capsys.readouterr().out and store.llm.prompts == []

    assert main(["show", "2024-06-12", "2024-07-31", "--generate"], settings=store.settings) == 0
    assert "Analysis (gpt-" in capsys.readouterr().out and len(store.llm.prompts) == 1


def test_show_reports_bad_dates(store: Store, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["show", "2024-09-18", "2024-07-31"], settings=store.settings) == 1
    assert "earlier than the target" in capsys.readouterr().err


def test_evaluate_scores_the_stored_analyses_and_writes_the_results_file(
    store: Store, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    from app.compare import evaluate
    from app.compare.evaluate import PairRecord

    gold = store.settings.data_dir / "eval" / "stance_gold.jsonl"
    gold.parent.mkdir(parents=True)
    labels = {"statement-2024-06-12": "neutral", "statement-2024-07-31": "neutral"}
    labels["statement-2024-09-18"] = "dovish"
    gold.write_text(
        "".join(json.dumps({"doc_id": d, "label": x}) + "\n" for d, x in labels.items())
    )
    stored = {
        "summary": "A cut.",
        "key_changes": [
            {
                "change_ids": [0],
                "what_changed": "“lower” replaced “maintain”.",
                "interpretation": "Easier.",
                "direction": "dovish",
            }
        ],
        "stance_shift": "more_dovish",
        "confidence": 0.9,
    }

    def load_pairs(conn: Any, settings: Settings, prompt_version: str | None = None) -> Any:
        days = [r.source for r in RECORDS]
        records = [
            PairRecord(a.doc_id, b.doc_id, b.meeting_date, frozenset({0}), stored, {"calls": 1})
            for a, b in zip(days, days[1:], strict=False)
        ]
        return records, {"statement-2024-07-31": 0.0}, f"m/{prompt_version or 'compare-v3.x'}"

    monkeypatch.setattr(evaluate, "load_pairs", load_pairs)

    assert main(["evaluate"], settings=store.settings) == 0

    out = capsys.readouterr().out
    assert "2 of 2 consecutive pairs analysed (2 with gold labels on both statements" in out
    assert "LLM (compare-v3) vs gold-implied shift: 0.500" in out
    assert "LLM (compare-v2) vs gold-implied shift: 0.500" in out
    assert "m/compare-v2.3b53da07: pre-registered" in out
    assert "key changes without a quote 0/2" in out
    written = json.loads((store.settings.data_dir / "eval" / "compare_benchmark.json").read_text())
    assert written["analyses"] == "m/compare-v3.x" and written["prompt"].startswith("post hoc")
    assert [run["analyses"] for run in written["earlier_prompts"]] == ["m/compare-v2.3b53da07"]
