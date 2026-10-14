"""The benchmark file built from synthetic cached results: rows, intervals, differences, question
records and the API's EvaluationFile contract (no OpenAI, no index)."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from eval_fakes import CHUNKS

from app.api.evaluation import build_report as api_report
from app.api.evaluation import evaluation_file_from_json
from app.core.config import Settings
from app.evaluation.judge import judge_input, score_judgment, verdicts
from app.evaluation.judge_prompts import AnswerJudgment, ClaimVerdict, FactVerdict
from app.evaluation.questions import EvalQuestion, load_questions
from app.evaluation.report import (
    KNOWN_ERRORS_IDS,
    KNOWN_ERRORS_NOTE,
    KNOWN_ERRORS_PROMPT,
    RunData,
    answers_table,
    build_report,
    comparison_table,
    coverage_lines,
    judge_validation,
    newest_result,
    retrieval_table,
    stale_answers,
    switch_selection,
    validation_table,
    write_report,
)
from app.evaluation.runner import resolve_config
from app.stance.metrics import Bootstrap

FAST = Bootstrap(resamples=500, seed=1)
NOW = datetime(2026, 10, 13, 12, 0, tzinfo=UTC)
# Hits per method and question: bm25 finds d1's statement 2nd, hybrid 1st; both miss d2's.
RANKS = {
    "bm25": {
        "d1": ["minutes-2024-07-31#0003", "statement-2024-09-18#0000"],
        "d2": ["statement-2024-07-31#0000"],
        "d3": ["statement-2015-12-16#0000"],
        "t1": ["statement-2015-12-16#0000"],
    },
    "hybrid": {
        "d1": ["statement-2024-09-18#0000", "minutes-2024-07-31#0003"],
        "d2": ["statement-2024-07-31#0000"],
        "d3": ["statement-2015-12-16#0000"],
        "t1": ["statement-2015-12-16#0000"],
    },
}
# Whether each answer's one claim is supported.
SUPPORTED = {
    "bm25": {"d1": True, "d2": False, "d3": False, "t1": True},
    "hybrid": {"d1": True, "d2": True, "d3": False, "t1": True},
}
USAGE = {"input_tokens": 2000, "cached_input_tokens": 0, "output_tokens": 300}
GENERATOR = {"model": "gpt-5.4-mini-2026-03-17", "prompt_version": "answer.x", "top_k": 5}
JUDGE = {"model": "gpt-5.5", "reasoning_effort": "low", "prompt_version": "j", "validation": None}


def hit(chunk_id: str, rank: int) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "doc_id": chunk_id.split("#")[0],
        "score": 1.0 / rank,
        "release_date": "2015-12-16",
    }


def make_runs(
    settings: Settings, questions: Any, *, drop_answer: str | None = None
) -> list[RunData]:
    setup = resolve_config("phase3", settings)
    runs = []
    for method, ranks in RANKS.items():
        run = RunData(setup, method)
        for q in questions:
            hits = [hit(c, r) for r, c in enumerate(ranks[q.id], start=1)]
            run.retrieval[q.id] = {
                "hits": hits,
                "latency_ms": 10.0 if method == "bm25" else 30.0,
                "leaks": 0,
            }
            if method == "bm25" and q.id == drop_answer:
                continue
            record = {
                "question_id": q.id,
                "config": setup.name,
                "method": method,
                "answer": f"Answer to {q.id} [1].",
                "citations": [ranks[q.id][0]],
                "passages": hits,
                "notes": [],
                "latency_ms": 900.0,
                "usage": USAGE,
                "model": GENERATOR["model"],
                "llm_called": True,
            }
            run.answers[q.id] = record
            judgment = AnswerJudgment(
                claims=[
                    ClaimVerdict(
                        sentence=1,
                        claim="c",
                        reason="r",
                        verdict="supported" if SUPPORTED[method][q.id] else "unsupported",
                    )
                ],
                citations=[],
                facts=[
                    FactVerdict(id=i, reason="r", verdict="covered")
                    for i in range(1, len(q.required_facts) + 1)
                ],
                response_type="answers" if q.answerable else "declines",
            )
            inp = judge_input(q, record, CHUNKS)
            run.judged[q.id] = {
                "scores": score_judgment(judgment, q, citation_checks=0),
                "verdicts": verdicts(judgment, inp),
                "usage": {**USAGE, "calls": 1},
                "model": "gpt-5.5",
            }
        runs.append(run)
    return runs


def report_for(settings: Settings, questions_file: Path, **kwargs: Any) -> dict[str, Any]:
    question_set = load_questions(questions_file)
    runs = make_runs(settings, question_set.questions, **kwargs)
    return build_report(
        question_set,
        runs,
        file_label="data/eval/rag_questions.jsonl",
        generator=GENERATOR,
        judge=JUDGE,
        select="ndcg_at_10",
        embedding={"model": "text-embedding-3-small", "texts": 4, "tokens": 50},
        bootstrap=FAST,
        now=NOW,
    )


def by_key(rows: list[dict[str, Any]], split: str) -> dict[str, dict[str, Any]]:
    return {r["method"]: r for r in rows if r["split"] == split}


def test_the_file_fits_the_api_contract(settings: Settings, questions_file: Path) -> None:
    report = report_for(settings, questions_file)

    file = evaluation_file_from_json(json.loads(json.dumps(report)))
    served = api_report(file)
    assert [q.id for q in served.questions] == ["d1", "d2", "d3", "t1"]
    assert report["generated_at"] == "2026-10-13T12:00:00+00:00"
    assert report["dataset"]["n_questions"] == 4 and report["dataset"]["n_with_as_of"] == 1
    assert report["dataset"]["splits"] == [
        {"split": "dev", "n": 3, "n_answerable": 2},
        {"split": "test", "n": 1, "n_answerable": 1},
    ]
    assert report["configs"][0]["id"] == "phase3"
    assert report["configs"][0]["params"]["rrf_weights"] == "1.0, 1.0"
    assert report["judge"]["n_answers"] == 8
    assert {(r["split"], r["method"]) for r in report["retrieval"]} == {
        (split, method) for split in ("dev", "test", "all") for method in ("bm25", "hybrid")
    }


def test_retrieval_rows_score_the_answerable_questions(
    settings: Settings, questions_file: Path
) -> None:
    report = report_for(settings, questions_file)
    dev = by_key(report["retrieval"], "dev")

    bm25 = dev["bm25"]
    assert bm25["n_questions"] == 2  # d3 is unanswerable
    # d1: its statement at rank 2; d2: nothing relevant retrieved.
    assert bm25["mrr_at_10"]["mean"] == pytest.approx((1 / 2 + 0) / 2)
    assert bm25["mrr_at_10"]["n"] == 2
    assert dev["hybrid"]["mrr_at_10"]["mean"] == pytest.approx(1 / 2)
    assert bm25["recall_at_5"]["mean"] == pytest.approx((1 + 0) / 2)
    assert bm25["latency_ms_p50"] == 10.0 and bm25["as_of_leaks"] == 0
    date_specific = next(
        r
        for r in report["per_type"]
        if (r["split"], r["method"], r["type"]) == ("dev", "bm25", "date_specific")
    )
    assert date_specific["n_questions"] == 1
    assert date_specific["ndcg_at_10"]["mean"] == pytest.approx(1 / 1.584962500721156)
    assert date_specific["faithfulness"]["mean"] == 1.0


def test_answer_rows_and_costs(settings: Settings, questions_file: Path) -> None:
    report = report_for(settings, questions_file)
    hybrid = by_key(report["answers"], "dev")["hybrid"]

    assert hybrid["n_questions"] == 3
    assert hybrid["faithfulness"]["mean"] == pytest.approx(2 / 3)
    assert hybrid["answer_correctness"]["mean"] == 1.0
    assert hybrid["citation_accuracy"] is None  # no citation checks at all
    assert hybrid["false_refusal_rate"]["mean"] == 0.0
    assert hybrid["refusal_accuracy"] == {
        "mean": 0.0,  # d3 declines but its one claim is unsupported
        "ci": {"low": 0.0, "high": 0.0},
        "n": 1,
    }
    assert hybrid["response_types"] == {"answers": 2, "declines": 1}
    assert hybrid["passages_differ_from_retrieval"] == 0
    answer_usd = (2000 * 0.75 + 300 * 4.5) / 1e6
    judge_usd = (2000 * 5 + 300 * 30) / 1e6
    assert hybrid["cost_per_query_usd"] == pytest.approx(answer_usd)
    assert hybrid["judge_cost_per_query_usd"] == pytest.approx(judge_usd)
    lines = {line["stage"]: line for line in report["cost"]["lines"]}
    assert lines["generation"]["calls"] == 8 and lines["judge"]["calls"] == 8
    assert lines["generation"]["usd"] == pytest.approx(8 * answer_usd)
    assert lines["embedding"]["input_tokens"] == 50
    assert report["cost"]["total_usd"] == pytest.approx(
        8 * answer_usd + 8 * judge_usd + 50 * 0.02 / 1e6
    )


def test_differences_put_the_later_method_first(settings: Settings, questions_file: Path) -> None:
    report = report_for(settings, questions_file)
    rows = {
        (r["split"], r["metric"]): r
        for r in report["differences"]
        if r["minuend"]["method"] == "hybrid" and r["subtrahend"]["method"] == "bm25"
    }

    mrr = rows[("dev", "mrr_at_10")]
    assert mrr["difference"] == pytest.approx(1 / 4) and mrr["n"] == 2
    faithfulness = rows[("all", "faithfulness")]
    assert faithfulness["difference"] == pytest.approx(1 / 4) and faithfulness["n"] == 4
    assert rows[("test", "recall_at_5")]["difference"] == 0.0
    assert ("dev", "refusal_accuracy") in rows
    assert not any(r["metric"] == "citation_accuracy_strict" for r in report["differences"])
    table = comparison_table(report["differences"], ["mrr_at_10"])
    assert "| hybrid - bm25 | mrr_at_10 | +0.250" in table


def test_selection_takes_the_best_dev_run(settings: Settings, questions_file: Path) -> None:
    report = report_for(settings, questions_file)

    assert report["selection"]["method"] == "hybrid"
    assert report["selection"]["metric"] == "ndcg_at_10"
    assert report["selection"]["split"] == "dev"


def test_rows_wait_until_a_split_is_complete(settings: Settings, questions_file: Path) -> None:
    question_set = load_questions(questions_file)
    report = report_for(settings, questions_file, drop_answer="d2")

    assert "bm25" not in by_key(report["answers"], "dev")
    assert "bm25" in by_key(report["answers"], "test")
    assert "bm25" in by_key(report["retrieval"], "dev")
    assert not any(
        r["split"] == "dev" and r["metric"] == "faithfulness" for r in report["differences"]
    )
    runs = make_runs(settings, question_set.questions, drop_answer="d2")
    assert coverage_lines(runs, question_set.questions) == [
        "phase3/bm25: retrieval 3/3 answerable, answers 3/4, judged 3/4"
    ]


def test_question_records(settings: Settings, questions_file: Path) -> None:
    report = report_for(settings, questions_file)

    d1 = next(q for q in report["questions"] if q["id"] == "d1")
    assert d1["as_of"] == "2024-10-01" and d1["type"] == "date_specific"
    assert d1["relevant_chunks"] == [{"chunk_id": "statement-2024-09-18#0000", "role": "primary"}]
    bm25 = d1["runs"][0]
    assert (bm25["method"], bm25["config"]) == ("bm25", "phase3")
    assert bm25["scores"]["first_relevant_rank"] == 2
    assert bm25["scores"]["mrr_at_10"] == 0.5
    assert bm25["retrieved"][0] == {"chunk_id": "minutes-2024-07-31#0003", "score": 1.0}
    assert bm25["answer"]["text"] == "Answer to d1 [1]."
    assert bm25["answer"]["judge"]["faithfulness"] == 1.0
    assert bm25["answer"]["judge"]["claims"][0]["supported"] is True
    d3 = next(q for q in report["questions"] if q["id"] == "d3")
    assert d3["runs"][0]["scores"]["recall_at_5"] is None  # unanswerable: not scored
    assert d3["runs"][0]["answer"]["judge"]["refusal"]["declined"] is True


def test_write_report_rounds_and_checks_the_contract(
    settings: Settings, questions_file: Path, tmp_path: Path
) -> None:
    report = report_for(settings, questions_file)
    path = tmp_path / "eval" / "rag_benchmark.json"

    write_report(path, report)

    written = json.loads(path.read_text(encoding="utf-8"))
    evaluation_file_from_json(written)
    ndcg = next(r for r in written["per_type"] if r["type"] == "date_specific")["ndcg_at_10"]
    assert ndcg["mean"] == 0.63093  # 5 significant digits
    broken = {**report, "configs": [{**report["configs"][0], "id": "Not An Id"}]}
    with pytest.raises(ValueError):
        write_report(path, broken)


VALIDATION_KEY = {
    "items": [
        {
            "item_id": "phase3/hybrid_rerank/a",
            "claims": [
                {"id": 1, "verdict": "supported"},
                {"id": 2, "verdict": "supported"},
                {"id": 3, "verdict": "unsupported"},
            ],
            "citations": [{"id": 1, "verdict": "partially_supports"}],
            "facts": [],
            "response_type": "answers",
        },
        {
            "item_id": "phase3/hybrid_rerank/b",
            "claims": [],
            "citations": [],
            "facts": [{"id": 1, "verdict": "missing"}],
            "response_type": "declines",
        },
    ]
}


def annotated(
    claims: list[str], citation: str, fact: str, types: tuple[str, str]
) -> list[dict[str, Any]]:
    return [
        {
            "item_id": "phase3/hybrid_rerank/a",
            "claims": [{"id": i, "human": label} for i, label in enumerate(claims, start=1)],
            "citations": [{"id": 1, "human": citation}],
            "facts": [],
            "response_type": {"human": types[0]},
        },
        {
            "item_id": "phase3/hybrid_rerank/b",
            "claims": [],
            "citations": [],
            "facts": [{"id": 1, "human": fact}],
            "response_type": {"human": types[1]},
        },
    ]


def test_judge_validation_status() -> None:
    assert judge_validation(None, None, None, None)["status"] == "not_started"
    sample = {
        "items": [{"item_id": "phase3/hybrid_rerank/a"}, {"item_id": "phase3/hybrid_rerank/b"}]
    }
    pending = judge_validation(sample, None, VALIDATION_KEY, "data/eval/rag_judge_sample.json")
    assert (pending["status"], pending["n_sampled"], pending["n_annotated"]) == ("pending", 2, 0)
    assert pending["agreement"] == [] and pending["inter_annotator"] == []
    assert pending["note"] == "Sampled answers by run: phase3/hybrid_rerank 2."

    one = {
        "annotator": "HK",
        "items": annotated(
            ["supported", "supported", "supported"], "supports", "missing", ("answers", "declines")
        ),
    }
    complete = judge_validation(sample, one, VALIDATION_KEY, "f")
    assert complete["status"] == "complete" and complete["n_annotated"] == 2
    assert complete["annotator"] == "HK" and complete["annotators"] == []
    claims = complete["agreement"][0]
    assert claims["check"] == "claim_support" and claims["annotator"] is None
    assert (claims["n"], claims["agreement"], claims["exact_agreement"]) == (3, 2 / 3, 2 / 3)
    citations = complete["agreement"][1]
    # partially_supports vs supports: the same pass/fail verdict, a different label.
    assert (citations["agreement"], citations["exact_agreement"]) == (1.0, 0.0)
    assert [r["check"] for r in complete["agreement"]] == [
        "claim_support",
        "citation_support",
        "fact_coverage",
        "refusal",
    ]


def test_judge_validation_with_two_annotators() -> None:
    sample = {
        "items": [{"item_id": "phase3/hybrid_rerank/a"}, {"item_id": "phase3/hybrid_rerank/b"}]
    }
    both = {
        "annotator": "two test annotators",
        "note": "B is a copy of A but for one claim.",
        "annotators": [
            {
                "id": "A",
                "annotator": "first",
                "items": annotated(
                    ["supported", "supported", "unsupported"],
                    "partially_supports",
                    "missing",
                    ("answers", "declines"),
                ),
            },
            {
                "id": "B",
                "annotator": "second",
                "items": annotated(
                    ["supported", "unsupported", "unsupported"],
                    "partially_supports",
                    "missing",
                    ("answers", "declines"),
                ),
            },
        ],
    }

    result = judge_validation(sample, both, VALIDATION_KEY, "f")

    assert result["annotator"] == "two test annotators"
    assert result["annotators"] == [
        {"id": "A", "description": "first", "isolated": False},
        {"id": "B", "description": "second", "isolated": False},
    ]
    assert result["sampled_runs"] == [{"method": "hybrid_rerank", "config": "phase3", "n": 2}]
    rows = {(r["annotator"], r["check"]): r for r in result["agreement"]}
    assert rows[("A", "claim_support")]["agreement"] == 1.0
    assert rows[("B", "claim_support")]["agreement"] == pytest.approx(2 / 3)
    pairs = {r["check"]: r for r in result["inter_annotator"]}
    assert pairs["claim_support"]["annotators"] == ["A", "B"]
    assert pairs["claim_support"]["agreement"] == pytest.approx(2 / 3)
    assert pairs["refusal"]["agreement"] == 1.0 and pairs["refusal"]["kappa"] == 1.0
    assert result["note"].endswith("B is a copy of A but for one claim.")
    assert "Sampled answers by run: phase3/hybrid_rerank 2." in result["note"]
    table = validation_table(result)
    assert "| judge vs B | claim_support | 3 | 0.667 |" in table
    assert "| A vs B | claim_support | 3 | 0.667 |" in table

    # A third, isolated annotator: the pair of the two others is marked as not independent.
    isolated = {
        "id": "C",
        "annotator": "third",
        "isolated": True,
        "items": annotated(
            ["supported", "supported", "unsupported"],
            "partially_supports",
            "missing",
            ("answers", "declines"),
        ),
    }
    three = {**both, "annotators": [isolated, *both["annotators"]]}
    result = judge_validation(sample, three, VALIDATION_KEY, "f")
    assert [(a["id"], a["isolated"]) for a in result["annotators"]] == [
        ("C", True),
        ("A", False),
        ("B", False),
    ]
    assert [r["annotator"] for r in result["agreement"]][:1] == ["C"]
    table = validation_table(result)
    assert "| C vs A | claim_support | 3 | 1.000 |" in table
    assert "| A vs B (not independent) | claim_support | 3 | 0.667 |" in table


def rule_rows(ndcg: tuple[float, float, float], correct: tuple[float, float, float]) -> list:
    def row(metric: str, values: tuple[float, float, float]) -> dict[str, Any]:
        difference, low, high = values
        return {
            "metric": metric,
            "split": "test",
            "minuend": {"method": "hybrid_rerank", "config": "tuned"},
            "subtrahend": {"method": "hybrid_rerank", "config": "phase3"},
            "difference": difference,
            "ci": {"low": low, "high": high},
            "n": 50,
        }

    return [row("ndcg_at_10", ndcg), row("answer_correctness", correct)]


@pytest.mark.parametrize(
    ("ndcg", "correct", "chosen", "verdicts"),
    [
        ((0.222, 0.158, 0.293), (0.14, 0.068, 0.218), "tuned", ("holds", "holds")),
        ((0.05, -0.01, 0.11), (0.14, 0.068, 0.218), "phase3", ("fails", "holds")),
        ((0.222, 0.158, 0.293), (-0.02, -0.06, 0.03), "phase3", ("holds", "fails")),
        ((0.222, 0.158, 0.293), (-0.02, -0.05, 0.03), "phase3", ("holds", "fails")),
    ],
)
def test_switch_rule(
    ndcg: tuple[float, float, float],
    correct: tuple[float, float, float],
    chosen: str,
    verdicts: tuple[str, str],
) -> None:
    selection = switch_selection(rule_rows(ndcg, correct), "phase3", "tuned")

    assert selection is not None
    assert (selection["method"], selection["config"], selection["split"]) == (
        "hybrid_rerank",
        chosen,
        "test",
    )
    assert f"n = 50: {verdicts[0]}; (b)" in selection["rule"]
    assert selection["rule"].count(f": {verdicts[1]}. ") == 1
    assert switch_selection(rule_rows(ndcg, correct)[:1], "phase3", "tuned") is None


def test_tables_render(settings: Settings, questions_file: Path) -> None:
    report = report_for(settings, questions_file)

    retrieval = retrieval_table([r for r in report["retrieval"] if r["split"] == "dev"])
    assert retrieval.splitlines()[2].startswith("| bm25 | 2 | 0.500 [")
    assert "| hybrid | 3 |" in answers_table([r for r in report["answers"] if r["split"] == "dev"])


def test_only_the_selected_questions_are_covered(settings: Settings, questions_file: Path) -> None:
    question_set = load_questions(questions_file)
    dev = [q for q in question_set.questions if q.split == "dev"]
    runs = make_runs(settings, question_set.questions)

    report = build_report(
        question_set,
        runs,
        file_label="x",
        generator=GENERATOR,
        judge=JUDGE,
        splits=("dev",),
        questions=dev,
        bootstrap=FAST,
    )

    assert [q["id"] for q in report["questions"]] == ["d1", "d2", "d3"]
    assert {r["split"] for r in report["retrieval"]} == {"dev"}
    assert report["selection"] is None
    evaluation_file_from_json(json.loads(json.dumps(report)))
    assert isinstance(dev[0], EvalQuestion)


def test_generated_at_is_the_newest_cached_result(settings: Settings, questions_file: Path) -> None:
    question_set = load_questions(questions_file)
    runs = make_runs(settings, question_set.questions)
    assert newest_result(runs) is None  # the fixtures carry no timestamps
    runs[0].retrieval["d1"]["run_at"] = "2026-10-13T18:00:18+00:00"
    runs[1].answers["d2"]["created_at"] = "2026-10-13T18:05:03+00:00"
    runs[1].judged["t1"]["created_at"] = "2026-10-13T18:10:05+00:00"
    assert newest_result(runs) == datetime(2026, 10, 13, 18, 10, 5, tzinfo=UTC)

    def build() -> dict[str, Any]:
        return build_report(
            question_set,
            runs,
            file_label="data/eval/rag_questions.jsonl",
            generator=GENERATOR,
            judge=JUDGE,
            bootstrap=FAST,
        )

    first, second = build(), build()
    assert first["generated_at"] == "2026-10-13T18:10:05+00:00"
    assert first == second  # nothing new: the same file


def test_known_judge_errors_note_needs_their_judgments(
    settings: Settings, questions_file: Path
) -> None:
    question_set = load_questions(questions_file)
    runs = make_runs(settings, question_set.questions)

    def notes(judge: dict[str, Any]) -> list[str]:
        return build_report(
            question_set, runs, file_label="f", generator=GENERATOR, judge=judge, bootstrap=FAST
        )["notes"]

    assert KNOWN_ERRORS_NOTE not in notes({**JUDGE, "prompt_version": KNOWN_ERRORS_PROMPT})
    tuned = RunData(resolve_config("tuned", settings), "hybrid_rerank")
    tuned.judged = {qid: runs[0].judged["d1"] for qid in KNOWN_ERRORS_IDS}
    runs.append(tuned)
    assert KNOWN_ERRORS_NOTE in notes({**JUDGE, "prompt_version": KNOWN_ERRORS_PROMPT})
    assert KNOWN_ERRORS_NOTE not in notes(JUDGE)  # another judge prompt: other verdicts


def test_stale_answers_counts_passages_that_differ_from_the_ranking(
    settings: Settings, questions_file: Path
) -> None:
    question_set = load_questions(questions_file)
    run = make_runs(settings, question_set.questions)[0]
    assert stale_answers(run) == 0
    run.retrieval["d1"]["hits"] = list(reversed(run.retrieval["d1"]["hits"]))
    assert stale_answers(run) == 1


def test_within_config_differences_name_their_config_when_several() -> None:
    def row(a: dict[str, str], b: dict[str, str]) -> dict[str, Any]:
        return {
            "metric": "ndcg_at_10",
            "split": "test",
            "minuend": a,
            "subtrahend": b,
            "difference": 0.1,
            "ci": {"low": 0.0, "high": 0.2},
            "n": 50,
        }

    dense, bm25 = {"method": "dense", "config": "phase3"}, {"method": "bm25", "config": "phase3"}
    tuned = {"method": "dense", "config": "tuned"}
    alone = comparison_table([row(dense, bm25)], ["ndcg_at_10"])
    assert "| dense - bm25 | ndcg_at_10 |" in alone
    both = comparison_table([row(dense, bm25), row(tuned, dense)], ["ndcg_at_10"])
    assert "| phase3: dense - bm25 | ndcg_at_10 |" in both
    assert "| tuned - phase3 (dense) | ndcg_at_10 |" in both
