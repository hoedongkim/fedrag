"""Tests for the stance CLI on fake TDW splits, statements and chat model (offline: no OpenAI,
no model download, no database)."""

import json
import re
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, Field

from app.api.stance import benchmark_from_json
from app.core.config import Settings
from app.ingest.records import Document, Paragraph, write_jsonl
from app.schemas.common import Stance
from app.stance import fred, llm, tdw
from app.stance.__main__ import check_budget, describe, main
from app.stance.keyword import Lexicon
from app.stance.llm import CostEstimate
from app.stance.prompts import SentenceLabel, SentenceLabels, StatementJudgment
from app.stance.report import confusion_table, rounded
from app.stance.statements import read_predictions
from app.stance.tdw import LabeledSentence

USAGE = {"input_tokens": 1_000, "output_tokens": 200, "total_tokens": 1_200}
HAWKISH = "The Committee decided to raise the target range because inflation pressures rose"
DOVISH = "The Committee decided to cut the target range because downside risks to growth rose"
NEUTRAL = "The Committee decided to maintain the target range and will monitor incoming data"
STATEMENTS = {
    date(2022, 6, 15): "The Committee decided to raise the target range. Inflation is elevated.",
    date(2024, 9, 18): "The Committee decided to cut the target range. Risks have shifted.",
    date(2025, 3, 19): "The Committee decided to maintain the target range. Data will guide it.",
}


def rule(text: str) -> Stance:
    return "hawkish" if "raise" in text else "dovish" if "cut" in text else "neutral"


class FakeChat(BaseChatModel):
    """Structured replies by `rule`; counts its calls."""

    calls: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-stance-cli"

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        raise NotImplementedError

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable[Any, Any]:
        def run(messages: list[BaseMessage]) -> dict[str, Any]:
            human = str(messages[-1].content)
            self.calls.append(schema.__name__)
            parsed: BaseModel
            if schema is SentenceLabels:
                found = re.findall(r"^\[(\d+)\] (.*)$", human, re.MULTILINE)
                parsed = SentenceLabels(
                    labels=[SentenceLabel(id=int(n), stance=rule(s)) for n, s in found]
                )
            else:
                text = human.split("<statement>")[1].split("</statement>")[0].strip()
                stance = rule(text)
                parsed = StatementJudgment(
                    key_factors=["rate decision"],
                    evidence=[text.split(". ")[0] + ".", "Not in the statement at all."],
                    stance=stance,
                    score={"hawkish": 0.8, "dovish": -0.8, "neutral": 0.0}[stance],
                    confidence=0.9,
                )
            raw = AIMessage(content="{}", usage_metadata=USAGE)
            return {"raw": raw, "parsed": parsed, "parsing_error": None}

        return RunnableLambda(run)


def split(seed: int, part: str) -> list[LabeledSentence]:
    """Fake TDW split: 6 sentences per class in train, 3 per class (one shared) in test."""
    count = 6 if part == "train" else 3
    rows = [
        LabeledSentence(i, f"{text} in round {seed}-{part}-{i}.", 2020, stance)
        for i in range(count)
        for text, stance in ((HAWKISH, "hawkish"), (DOVISH, "dovish"), (NEUTRAL, "neutral"))
    ]
    if part == "test":  # a test sentence whose text is also in train: dropped when deduplicated
        rows[0] = LabeledSentence(0, f"{HAWKISH} in round {seed}-train-0.", 2020, "hawkish")
    return rows


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """A key is set (the CLI checks for one) but never used: the chat model is a fake."""
    return Settings(
        _env_file=None, environment="test", openai_api_key="unused-fake-key", data_dir=tmp_path
    )


@pytest.fixture
def chat(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> FakeChat:
    fake = FakeChat()
    monkeypatch.setattr(tdw, "load", lambda part, seed, _settings=None: split(seed, part))
    monkeypatch.setattr(llm, "get_llm", lambda _settings: fake)
    monkeypatch.setattr(llm, "count_tokens", lambda text: len(text.split()))
    documents = [
        Document(
            doc_type="statement",
            meeting_date=day,
            release_date=day,
            release_date_source="press_release",
            title="FOMC Statement",
            url="https://www.federalreserve.gov/x.htm",
            paragraphs=(
                Paragraph(None, text),
                Paragraph(None, "Voting for the monetary policy action were all members."),
            ),
        ).to_json()
        for day, text in STATEMENTS.items()
    ]
    write_jsonl(settings.data_dir / "processed" / "documents.jsonl", documents)
    return fake


def results(settings: Settings) -> dict[str, Any]:
    return json.loads((settings.data_dir / "eval" / "stance_benchmark.json").read_text())


def write_gold(settings: Settings, human: dict[str, Stance] | None = None) -> None:
    rows = [
        {
            "doc_id": f"statement-{day.isoformat()}",
            "label": "hawkish",  # wrong for two of three: the scores must not be perfect
            "confidence": "high",
            "human_label": (human or {}).get(f"statement-{day.isoformat()}"),
        }
        for day in STATEMENTS
    ]
    write_jsonl(settings.data_dir / "eval" / "stance_gold.jsonl", rows)


def test_benchmark_writes_sentence_level_cost_and_models(
    settings: Settings, chat: FakeChat, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["benchmark", "--classifiers", "llm_few_shot", "keyword", "llm_zero_shot"]
    assert main(argv, settings=settings) == 0
    out = capsys.readouterr().out
    assert "OpenAI estimate" in out and "llm_few_shot, seed 5768" in out
    report = results(settings)
    assert [e["classifier"] for e in report["sentence_level"]] == [
        "keyword",
        "llm_zero_shot",
        "llm_few_shot",
    ]
    api = benchmark_from_json(report)  # the file is what GET /api/stance/benchmark serves
    official = {(r.classifier, r.split): r for r in api.sentence_level}
    assert official[("llm_zero_shot", "official")].macro_f1_mean == 1.0
    assert official[("llm_zero_shot", "deduplicated")].n_test_mean == 8
    assert report["cost"]["steps"]["sentence_benchmark"]["calls"] == len(chat.calls)
    assert report["cost"]["total"]["usd"] > 0
    few_shot = report["models"]["llm_few_shot"]["sentence"]
    assert sorted(few_shot) == sorted(str(seed) for seed in tdw.SEEDS)
    assert all(len(info["examples_sha256"]) == 64 for info in few_shot.values())
    (majority,) = report["sentence_baselines"]  # 6 train sentences per class: a tie, dovish
    assert (majority["name"], majority["label"]) == ("majority_class", "dovish")
    baselines = {(r.name, r.split): r for r in api.sentence_baselines}
    assert baselines[("majority_class", "official")].accuracy_mean == pytest.approx(1 / 3, abs=1e-4)
    assert baselines[("majority_class", "deduplicated")].n_test_mean == 8
    text = json.dumps(report)
    assert "round" not in text and "Committee decided" not in text  # no TDW sentence

    calls = len(chat.calls)
    assert main(argv, settings=settings) == 0  # served from the cache
    assert len(chat.calls) == calls
    assert results(settings)["cost"] == report["cost"]
    assert "no calls (everything was cached)" in capsys.readouterr().out


def test_benchmark_refuses_a_run_over_budget(
    settings: Settings, chat: FakeChat, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["benchmark", "--classifiers", "llm_zero_shot", "--max-usd", "0"]
    assert main(argv, settings=settings) == 1
    assert chat.calls == []
    assert "nothing was called" in capsys.readouterr().err
    assert not (settings.data_dir / "eval" / "stance_benchmark.json").exists()


def test_predict_merges_classifiers_and_evaluate_scores_them(
    settings: Settings, chat: FakeChat, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(["predict", "--classifiers", "llm_zero_shot", "llm_few_shot"], settings=settings) == 0
    )
    assert main(["predict", "--classifiers", "keyword"], settings=settings) == 0
    path = settings.data_dir / "stance" / "predictions.jsonl"
    predictions = read_predictions(path)
    assert len(predictions) == 3 * len(STATEMENTS)
    assert [p.classifier for p in predictions[:3]] == ["keyword", "llm_zero_shot", "llm_few_shot"]
    zero_shot = {p.doc_id: p for p in predictions if p.classifier == "llm_zero_shot"}
    cut = zero_shot["statement-2024-09-18"]
    assert (cut.stance, cut.score, cut.confidence) == ("dovish", -0.8, 0.9)
    assert cut.evidence == ("The Committee decided to cut the target range.",)  # verbatim only
    keyword = [p for p in predictions if p.classifier == "keyword"]
    assert all(-1 <= p.score <= 1 and p.confidence == abs(p.score) for p in keyword)
    statement_calls = chat.calls.count("StatementJudgment")
    assert statement_calls == 2 * len(STATEMENTS)
    post_hoc = read_predictions(settings.data_dir / "stance" / "predictions_post_hoc.jsonl")
    assert [p.classifier for p in post_hoc] == ["keyword"] * len(STATEMENTS)  # no LLM rows
    assert all("rubric-ordered (post hoc)" in p.model_version for p in post_hoc)

    write_gold(settings, human={"statement-2024-09-18": "dovish"})
    capsys.readouterr()
    assert main(["evaluate"], settings=settings) == 0
    out = capsys.readouterr().out
    assert "3 reference labels (1 human, 2 LLM-adjudicated)" in out
    assert "reference / predicted" in out
    assert "post hoc" in out and "Paired bootstrap differences" in out
    assert "keyword (post hoc) - rate_move_rule, all (n=3)" in out
    assert "whole calendar years resampled" in out
    report = results(settings)
    rows = [(e["classifier"], e["variant"]) for e in report["statement_level"]]
    assert rows == [
        ("keyword", "pre_registered"),
        ("llm_zero_shot", "pre_registered"),
        ("llm_few_shot", "pre_registered"),
        ("keyword", "post_hoc_rubric_ordered"),
    ]
    entries = {e["classifier"]: e for e in report["statement_level"][:3]}
    assert set(report["statement_variants"]) == {"pre_registered", "post_hoc_rubric_ordered"}
    decided_by = report["models"]["keyword"]["statement_post_hoc"]["decided_by"]
    assert sum(decided_by.values()) == len(STATEMENTS)
    zero = entries["llm_zero_shot"]
    assert zero["reference"] == "mixed"
    assert zero["all"]["n"] == 3 and zero["all"]["accuracy"] == pytest.approx(2 / 3, abs=1e-4)
    assert zero["human_subset"]["n"] == 1 and zero["human_subset"]["accuracy"] == 1.0
    assert zero["all"]["accuracy_ci"]["low"] <= zero["all"]["accuracy"]
    assert zero["human_subset"]["accuracy_ci"] == {"low": 1.0, "high": 1.0}
    bootstrap = report["statement_bootstrap"]
    assert bootstrap["resamples"] == 10_000
    paired = [(d["minuend"]["classifier"], d["subset"]) for d in bootstrap["paired_differences"]]
    assert paired == [("llm_zero_shot", "all"), ("llm_zero_shot", "human_subset")]  # no finbert
    assert report["references"]["n_human"] == 1
    assert report["references"]["llm_vs_human"]["accuracy"] == 0.0
    assert report["models"]["llm_few_shot"]["statement"]["model_version"].startswith(
        settings.llm_model
    )
    assert report["cost"]["steps"]["statement_predictions"]["calls"] == statement_calls
    api = benchmark_from_json(report)
    assert {r.classifier: r.reference for r in api.statement_level}["keyword"] == "mixed"
    assert [r.variant for r in api.statement_level].count("post_hoc_rubric_ordered") == 1
    assert api.statement_bootstrap and len(api.statement_bootstrap.paired_differences) == 2
    # Baselines with no classifier: the effective references are hawkish, dovish, hawkish.
    baselines = {b.name: b for b in api.statement_baselines}
    assert (baselines["majority_class"].label, baselines["majority_class"].post_hoc) == (
        "hawkish",
        False,
    )
    assert baselines["majority_class"].accuracy == pytest.approx(2 / 3, abs=1e-4)
    rule = baselines["rate_move_rule"]  # raise, cut, maintain: hawkish, dovish, neutral
    assert (rule.label, rule.post_hoc, rule.n) == (None, True, 3)
    assert rule.accuracy == pytest.approx(2 / 3, abs=1e-4)
    assert [
        (d.row.classifier, d.row.variant, d.baseline, d.subset)
        for d in api.statement_bootstrap.baseline_differences
    ] == [
        ("keyword", "post_hoc_rubric_ordered", "rate_move_rule", "all"),
        ("keyword", "post_hoc_rubric_ordered", "rate_move_rule", "human_subset"),
    ]  # no finbert rows, so no majority_class comparison
    years = report["statement_bootstrap"]["year_blocks"]
    assert years["years"] == 3 and len(years["paired_differences"]) == 2
    assert "accuracy_ci_year_blocks" in zero["all"]


def test_uncached_llm_predictions_without_a_key_fail_before_any_work(
    settings: Settings, chat: FakeChat, capsys: pytest.CaptureFixture[str]
) -> None:
    keyless = settings.model_copy(update={"openai_api_key": None})
    assert main(["predict", "--classifiers", "keyword", "llm_zero_shot"], settings=keyless) == 1
    assert "LLM not configured" in capsys.readouterr().err
    assert chat.calls == []
    assert not (settings.data_dir / "stance" / "predictions.jsonl").exists()


def test_evaluate_without_reference_labels_writes_an_empty_section(
    settings: Settings, chat: FakeChat, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["predict", "--classifiers", "keyword"], settings=settings) == 0
    assert main(["evaluate"], settings=settings) == 0
    assert "statement-level evaluation skipped" in capsys.readouterr().out
    report = results(settings)
    assert report["statement_level"] == [] and report["references"] is None
    assert "statement_bootstrap" not in report and "statement_baselines" not in report


def test_evaluate_and_load_need_predictions(
    settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["evaluate"], settings=settings) == 1
    assert "predict` first" in capsys.readouterr().err


def test_load_without_a_database_fails_cleanly(
    settings: Settings, chat: FakeChat, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["predict", "--classifiers", "keyword"], settings=settings) == 0
    points = [fred.TargetRange(date(2024, 9, 19), Decimal("4.75"), Decimal("5.00"))]
    fred.write_change_points(settings.data_dir / "stance" / "fed_funds_target.csv", points)
    capsys.readouterr()
    assert main(["load"], settings=settings) == 1
    assert "Database not configured" in capsys.readouterr().err


def test_a_malformed_input_file_fails_cleanly(
    settings: Settings, chat: FakeChat, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["predict", "--classifiers", "keyword"], settings=settings) == 0
    path = settings.data_dir / "stance" / "fed_funds_target.csv"
    path.write_text("day,lo,hi\n2024-09-19,4.75,5.00\n", encoding="utf-8")
    capsys.readouterr()
    assert main(["load"], settings=settings) == 1
    err = capsys.readouterr().err
    assert err.startswith("load: ") and "unexpected header" in err
    assert "Traceback" not in err


def test_check_budget_allows_free_runs_and_refuses_unknown_prices() -> None:
    free = CostEstimate(items=5, cached=5, calls=0, input_tokens=0, output_tokens=0, usd=0.0)
    assert check_budget([("cached", free)], 0.0) == 0
    unknown = CostEstimate(
        items=5, cached=0, calls=1, input_tokens=9, output_tokens=9, usd=float("nan")
    )
    with pytest.raises(RuntimeError, match="nothing was called"):
        check_budget([("unknown", unknown)], 5.0)


def test_describe_never_includes_sentences() -> None:
    class Keyword:
        lexicons = (Lexicon(("raise",), ("cut",), 1),)

    assert describe(Keyword())["hawkish"] == ["raise"]

    class FewShot:
        model_version = "m+p"
        examples = (LabeledSentence(0, "A labelled training sentence.", 2020, "hawkish"),)
        examples_digest = "d" * 64

    assert describe(FewShot()) == {"model_version": "m+p", "examples_sha256": "d" * 64}


def test_rounded_keeps_significant_digits_and_drops_nan() -> None:
    assert rounded({"lr": [2e-5, 0.6102345, float("nan")], "n": 3}) == {
        "lr": [2e-5, 0.61023, None],
        "n": 3,
    }
    table = confusion_table([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    assert table.splitlines()[0].split()[-3:] == ["dovish", "neutral", "hawkish"]
