"""python -m app.evaluation end to end with the retriever, the index and both LLMs faked out (no
network): estimates and refusals, caching, the report and the annotation round trip."""

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from eval_fakes import CHUNKS, FakeAnswerLLM, FakeJudgeLLM, FakeRetriever

import app.evaluation.__main__ as cli
from app.api.evaluation import evaluation_file_from_json
from app.core.config import Settings


@pytest.fixture
def keyed(settings: Settings) -> Settings:
    return settings.model_copy(update={"openai_api_key": "sk-test-not-real"})


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    retriever, answer_llm, judge_llm = FakeRetriever(), FakeAnswerLLM(), FakeJudgeLLM()
    monkeypatch.setattr(
        cli, "retriever_factory", lambda settings, qs, methods: lambda s, m: retriever
    )
    monkeypatch.setattr(cli, "unembedded", lambda settings, qs, methods: [])
    monkeypatch.setattr(cli, "index_chunks", lambda settings: dict(CHUNKS))
    monkeypatch.setattr("app.rag.answer.get_llm", lambda settings: answer_llm)
    monkeypatch.setattr(
        "app.evaluation.judge.get_judge_llm", lambda settings, model, effort: judge_llm
    )
    return {"retriever": retriever, "answer": answer_llm, "judge": judge_llm}


UNEMBEDDED = cli.unembedded  # the real check, before the fakes replace it


def run(settings: Settings, *argv: str) -> int:
    return cli.main(list(argv), settings=settings)


def test_validate(keyed: Settings, questions_file: Path, fakes: dict, capsys) -> None:
    assert run(keyed, "validate") == 0
    out = capsys.readouterr().out
    assert "rag_questions.jsonl (rag_questions): 4 questions, 3 answerable" in out
    assert "0 errors, 0 warnings" in out

    bad = questions_file.read_text(encoding="utf-8").replace(
        '"statement-2015-12-16#0000"', '"statement-2015-12-16#0099"'
    )
    questions_file.write_text(bad, encoding="utf-8")
    assert run(keyed, "validate") == 1
    assert (
        "error: t1: chunk statement-2015-12-16#0099 is not in the index" in capsys.readouterr().out
    )


def test_retrieval_ranks_every_question_and_scores_the_answerable(
    keyed: Settings, questions_file: Path, fakes: dict, capsys
) -> None:
    assert run(keyed, "retrieval", "--split", "all", "--methods", "bm25", "hybrid") == 0

    out = capsys.readouterr().out
    assert "4 all questions ranked, the 3 answerable ones scored" in out
    assert "| bm25 | 3 |" in out and "| hybrid | 3 |" in out
    # 2 methods x (1 warm-up + 4 questions), each with depth 10.
    assert len(fakes["retriever"].calls) == 10
    assert {c["top_k"] for c in fakes["retriever"].calls} == {10}


def test_the_cost_guard_refuses_before_any_call(
    keyed: Settings, questions_file: Path, fakes: dict, capsys
) -> None:
    assert run(keyed, "answers", "--split", "all", "--max-usd", "0.00001") == 1

    err = capsys.readouterr()
    assert "16 calls" in err.out
    assert "refused: the estimate" in err.err
    assert fakes["answer"].calls == []
    assert not (keyed.data_dir / "eval_cache" / "answers.jsonl").exists()


def test_max_calls_refuses_before_any_call(
    keyed: Settings, questions_file: Path, fakes: dict, capsys
) -> None:
    assert run(keyed, "answers", "--split", "dev", "--max-calls", "2") == 1

    assert "12 calls needed but only 2" in capsys.readouterr().err
    assert fakes["answer"].calls == []


def test_without_a_key_nothing_is_called(
    settings: Settings, questions_file: Path, fakes: dict, capsys
) -> None:
    assert run(settings, "answers", "--methods", "bm25") == 1

    assert "LLM not configured" in capsys.readouterr().err
    assert fakes["answer"].calls == []


def test_a_dry_run_embeds_nothing(
    keyed: Settings,
    questions_file: Path,
    fakes: dict,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    def no_embeddings(settings: Settings) -> Any:
        raise AssertionError("a dry run must not create an embedding client")

    monkeypatch.setattr(cli, "unembedded", UNEMBEDDED)
    monkeypatch.setattr("app.rag.index.make_embeddings", no_embeddings)

    assert run(keyed, "answers", "--split", "test", "--methods", "dense", "--dry-run") == 0

    assert "Dry run: 1 questions would be embedded first" in capsys.readouterr().out
    assert fakes["retriever"].calls == [] and fakes["answer"].calls == []


def test_dry_run_estimates_only(keyed: Settings, questions_file: Path, fakes: dict, capsys) -> None:
    assert run(keyed, "answers", "--split", "test", "--dry-run") == 0

    out = capsys.readouterr().out
    assert "4 calls for 4 uncached of 4 items" in out and "Dry run" in out
    assert fakes["answer"].calls == []


def test_answers_judge_report_and_annotation_round_trip(
    keyed: Settings, questions_file: Path, fakes: dict, capsys
) -> None:
    assert run(keyed, "retrieval", "--split", "all") == 0
    assert run(keyed, "answers", "--split", "all", "--concurrency", "2") == 0
    assert len(fakes["answer"].calls) == 16
    out = capsys.readouterr().out
    assert "Generated 16 answers now" in out and "16 calls" in out

    # Everything cached: no retrieval for answers, no LLM call.
    retrieved = len(fakes["retriever"].calls)
    assert run(keyed, "answers", "--split", "all") == 0
    assert len(fakes["answer"].calls) == 16 and len(fakes["retriever"].calls) == retrieved
    assert "0 calls for 0 uncached of 16 items" in capsys.readouterr().out

    assert run(keyed, "judge", "--split", "all", "--max-usd", "5") == 0
    # The fake retriever ranks alike for every method, so the 4 methods' answers to a question
    # are identical: one judgment (one call) serves all 4.
    assert len(fakes["judge"].calls) == 4
    # --config default: the production preset (RETRIEVER_PRESET, tuned unless set).
    assert "tuned/hybrid_rerank: 4 answers; faithfulness 1.000" in capsys.readouterr().out
    assert run(keyed, "judge", "--split", "all") == 0
    assert len(fakes["judge"].calls) == 4
    assert "no OpenAI calls" in capsys.readouterr().out

    assert run(keyed, "report", "--select", "ndcg_at_10") == 0
    out = capsys.readouterr().out
    assert "Wrote " in out and "Judge validation: not_started" in out
    # Not the full rebuild (one config, no --switch-rule): a scratch file, never the tracked one.
    assert not (keyed.data_dir / "eval" / "rag_benchmark.json").exists()
    path = keyed.data_dir / "eval_cache" / "rag_benchmark.all-tuned.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    file = evaluation_file_from_json(report)
    assert {(r.split, r.method) for r in file.answers} == {
        (split, method)
        for split in ("dev", "test", "all")
        for method in ("bm25", "dense", "hybrid", "hybrid_rerank")
    }
    assert file.selection is not None and file.selection.split == "dev"
    d1 = next(q for q in file.questions if q.id == "d1")
    assert all(r.retrieved[0].chunk_id != "minutes-2024-09-18#0001" for r in d1.runs)  # as_of
    lines = {line.stage: line for line in file.cost.lines}
    assert set(lines) == {"generation", "judge", "embedding"}
    assert lines["generation"].calls == 16 and lines["judge"].calls == 4

    # The annotator's file closes the loop.
    assert run(keyed, "export-judge-sample", "--split", "all", "--n", "4") == 0
    sample_path = keyed.data_dir / "eval" / "rag_judge_sample.json"
    key_path = keyed.data_dir / "eval" / "rag_judge_sample_key.json"
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    key = json.loads(key_path.read_text(encoding="utf-8"))
    assert len(sample["items"]) == 4 and key["judge"]["model"] == "gpt-5.5"
    # The tracked sample is not overwritten: annotations label it.
    assert run(keyed, "export-judge-sample", "--split", "all", "--n", "4") == 1
    assert "pass --out" in capsys.readouterr().err
    assert json.loads(sample_path.read_text(encoding="utf-8")) == sample
    assert run(keyed, "report") == 0
    assert "Judge validation: pending (0 annotated)" in capsys.readouterr().out
    verdicts = {k["item_id"]: k for k in key["items"]}
    for item in sample["items"]:
        for kind in ("claims", "citations", "facts"):
            labels = {v["id"]: v["verdict"] for v in verdicts[item["item_id"]][kind]}
            for unit in item[kind]:
                unit["human"] = labels[unit["id"]]
        item["response_type"]["human"] = verdicts[item["item_id"]]["response_type"]
    sample["annotator"] = "one annotator, blind to the verdicts"
    (keyed.data_dir / "eval" / "rag_judge_annotations.json").write_text(
        json.dumps(sample), encoding="utf-8"
    )
    assert run(keyed, "report") == 0
    assert "Judge validation: complete (4 annotated)" in capsys.readouterr().out
    file = evaluation_file_from_json(json.loads(path.read_text(encoding="utf-8")))
    assert file.judge is not None and file.judge.validation is not None
    assert file.judge.validation.agreement[0].agreement == 1.0
    assert file.judge.validation.inter_annotator == []

    # Two annotators: judge vs each, and A vs B. B flips one claim label.
    items = [
        {k: item[k] for k in ("item_id", "claims", "citations", "facts", "response_type")}
        for item in sample["items"]
    ]
    flipped = json.loads(json.dumps(items))
    first_claims = next(item["claims"] for item in flipped if item["claims"])
    first_claims[0]["human"] = (
        "unsupported" if first_claims[0]["human"] == "supported" else "supported"
    )
    (keyed.data_dir / "eval" / "rag_judge_annotations.json").write_text(
        json.dumps(
            {
                "annotators": [
                    {"id": "A", "annotator": "first", "isolated": True, "items": items},
                    {"id": "B", "annotator": "second", "items": flipped},
                ],
                "note": "Both are test fixtures.",
            }
        ),
        encoding="utf-8",
    )
    assert run(keyed, "report") == 0
    out = capsys.readouterr().out
    assert "| judge vs A | claim_support |" in out and "| A vs B | claim_support |" in out
    validation = evaluation_file_from_json(json.loads(path.read_text(encoding="utf-8"))).judge
    assert validation is not None and validation.validation is not None
    checks = {(a.annotator, a.check): a for a in validation.validation.agreement}
    assert checks[("A", "claim_support")].agreement == 1.0
    assert checks[("B", "claim_support")].agreement < 1.0
    assert [(a.id, a.isolated) for a in validation.validation.annotators] == [
        ("A", True),
        ("B", False),
    ]
    assert sum(r.n for r in validation.validation.sampled_runs) == 4
    pair = next(a for a in validation.validation.inter_annotator if a.check == "claim_support")
    assert pair.annotators == ["A", "B"] and pair.agreement < 1.0
    assert validation.validation.note and "Both are test fixtures." in validation.validation.note


def test_switch_rule_keeps_the_incumbent_without_a_clear_gain(
    keyed: Settings, questions_file: Path, fakes: dict, capsys
) -> None:
    configs = ("--config", "phase3", "tuned", "--methods", "hybrid_rerank")
    assert run(keyed, "retrieval", "--split", "all", *configs) == 0
    assert run(keyed, "answers", "--split", "all", *configs) == 0
    assert run(keyed, "judge", "--split", "all", *configs, "--max-usd", "5") == 0
    capsys.readouterr()

    assert run(keyed, "report", *configs, "--switch-rule") == 0

    out = capsys.readouterr().out
    assert "Default-switch rule: phase3/hybrid_rerank" in out
    path = keyed.data_dir / "eval" / "rag_benchmark.json"
    file = evaluation_file_from_json(json.loads(path.read_text(encoding="utf-8")))
    # The fake retriever ranks alike under both configs: a zero gain, so (a) fails.
    assert file.selection is not None
    assert (file.selection.config, file.selection.split) == ("phase3", "test")
    assert "(a) +0.000 [+0.000, +0.000], n = 1: fails" in file.selection.rule
    assert "(b) +0.000 [+0.000, +0.000], n = 1: holds" in file.selection.rule
    assert "phase3 stays the default" in file.selection.rule
    assert any(note.startswith("Selection: the default-switch rule") for note in file.notes)

    assert run(keyed, "report", "--methods", "hybrid_rerank", "--switch-rule") == 1
    assert "--switch-rule needs two configs" in capsys.readouterr().err
    assert run(keyed, "report", *configs, "--switch-rule", "--ids", "t1") == 1
    assert "--switch-rule needs the full TEST split: drop --ids" in capsys.readouterr().err

    # Any other selection writes a scratch file, not the tracked benchmark.
    before = path.read_text(encoding="utf-8")
    assert run(keyed, "report", *configs, "--split", "dev") == 0
    assert path.read_text(encoding="utf-8") == before
    scratch = keyed.data_dir / "eval_cache" / "rag_benchmark.dev-phase3-tuned.json"
    assert evaluation_file_from_json(json.loads(scratch.read_text(encoding="utf-8")))


def test_another_question_file_never_overwrites_the_tracked_benchmark(
    keyed: Settings, questions_file: Path, fakes: dict, capsys
) -> None:
    scratch = questions_file.with_name("scratch.jsonl")
    shutil.copy(questions_file, scratch)

    assert run(keyed, "retrieval", "--questions", str(scratch), "--methods", "bm25") == 0
    assert run(keyed, "report", "--questions", str(scratch), "--methods", "bm25") == 0

    assert not (keyed.data_dir / "eval" / "rag_benchmark.json").exists()
    written = keyed.data_dir / "eval_cache" / "rag_benchmark.scratch.json"
    assert evaluation_file_from_json(json.loads(written.read_text(encoding="utf-8")))
    assert "incomplete: tuned/bm25" in capsys.readouterr().out


def test_unknown_ids_and_presets_are_reported(
    keyed: Settings, questions_file: Path, fakes: dict, capsys
) -> None:
    assert run(keyed, "retrieval", "--ids", "nope") == 1
    assert "unknown question ids: nope" in capsys.readouterr().err
    assert run(keyed, "retrieval", "--config", "no-such-preset") == 1
    assert "unknown retriever preset" in capsys.readouterr().err
