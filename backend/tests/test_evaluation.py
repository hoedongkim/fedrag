"""Tests for the RAG evaluation API: the report and per-question drilldowns served from
data/eval/rag_benchmark.json, with passage text from the search index's chunks.jsonl."""

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.evaluation import (
    EVALUATION_MISSING,
    build_report,
    doc_id_of,
    evaluation_file_from_json,
)
from app.rag.index import CHUNKS_FILE, MANIFEST_FILE
from app.schemas.evaluation import Estimate, EvaluationFile, MetricInterval

MakeClient = Callable[..., TestClient]

STATEMENT = "statement-2024-09-18"
MINUTES = "minutes-2024-09-18"
OTHER = "minutes-2024-07-31"


def _estimate(value: float | None, n: int = 2) -> dict[str, Any]:
    ci = None if value is None else {"low": max(0.0, value - 0.2), "high": min(1.0, value + 0.1)}
    return {"value": value, "ci": ci, "n": n}


RETRIEVAL_METRICS = (
    "recall_at_5",
    "recall_at_10",
    "mrr_at_10",
    "ndcg_at_10",
    "doc_recall_at_5",
    "doc_recall_at_10",
)
ANSWER_METRICS = (
    "faithfulness",
    "citation_accuracy",
    "answer_correctness",
    "refusal_accuracy",
    "false_refusal_rate",
)


def _retrieval_row(method: str, config: str, split: str = "all", value: float = 0.5) -> dict:
    return {
        "method": method,
        "config": config,
        "split": split,
        "n_questions": 2,
        **{metric: _estimate(value) for metric in RETRIEVAL_METRICS},
        "latency_ms_p50": 12.0,
        "latency_ms_p95": 30.5,
        "as_of_leaks": 0,
    }


def _answer_row(method: str, config: str, split: str = "all") -> dict:
    return {
        "method": method,
        "config": config,
        "split": split,
        "n_questions": 2,
        **{metric: _estimate(0.75) for metric in ANSWER_METRICS},
        "latency_ms_p50": 2100.0,
        "latency_ms_p95": 3900.0,
        "cost_per_query_usd": 0.0011,
        "judge_cost_per_query_usd": 0.004,
    }


def _run(method: str, config: str, *, answered: bool = True) -> dict:
    run: dict[str, Any] = {
        "method": method,
        "config": config,
        "latency_ms": 8.5,
        "scores": {
            "first_relevant_rank": 2,
            "recall_at_5": 0.5,
            "recall_at_10": 1.0,
            "mrr_at_10": 0.5,
            "ndcg_at_10": 0.61,
            "doc_recall_at_5": 1.0,
            "doc_recall_at_10": 1.0,
        },
        "retrieved": [
            {"chunk_id": f"{OTHER}#0003", "score": 9.1},
            {"chunk_id": f"{STATEMENT}#0000", "score": 8.7},
            {"chunk_id": f"{MINUTES}#0012", "score": 7.0},
            {"chunk_id": f"{MINUTES}#0013", "score": 6.2},
        ],
        "answer": None,
    }
    if answered:
        run["answer"] = {
            "text": "The Committee lowered the target range by 1/2 percentage point [1]. "
            "Governor Bowman dissented [2].",
            "citations": [f"{STATEMENT}#0000", f"{MINUTES}#0099"],
            "notes": [],
            "latency_ms": 2300.0,
            "cost_usd": 0.0012,
            "judge": {
                "faithfulness": 0.5,
                "citation_accuracy": 0.5,
                "answer_correctness": 1.0,
                "refusal_correct": None,
                "false_refusal": False,
                "claims": [
                    {
                        "claim": "The Committee lowered the range by 1/2 point.",
                        "citations": [1],
                        "supported": True,
                        "reason": "Stated in the decision paragraph.",
                    },
                    {
                        "claim": "Governor Bowman dissented.",
                        "citations": [2],
                        "supported": False,
                        "reason": "The cited passage does not mention the vote.",
                    },
                ],
                "citations": [
                    {"n": 1, "sentence": "The Committee lowered ...", "supported": True},
                    {"n": 2, "sentence": "Governor Bowman dissented.", "supported": False},
                ],
                "facts": [{"fact": "Cut by 50 basis points", "covered": True, "reason": None}],
                "refusal": {"declined": False, "fabricated": False, "reason": None},
            },
        }
    return run


def _question(qid: str = "q001", **fields: Any) -> dict:
    return {
        "id": qid,
        "question": "How much did the FOMC cut rates in September 2024, and who dissented?",
        "as_of": "2024-10-01",
        "type": "date_specific",
        "split": "test",
        "answerable": True,
        "human_reviewed": True,
        "reference_answer": "By 1/2 point to 4-3/4 to 5 percent; Governor Bowman dissented.",
        "required_facts": ["Cut by 50 basis points", "Bowman dissented"],
        "relevant_chunks": [
            {"chunk_id": f"{MINUTES}#0013", "role": "supporting"},
            {"chunk_id": f"{STATEMENT}#0000", "role": "primary"},
        ],
        "relevant_doc_ids": [STATEMENT, MINUTES],
        "notes": None,
        "runs": [
            _run("bm25", "phase3", answered=False),
            _run("hybrid_rerank", "phase3"),
            _run("hybrid_rerank", "tuned"),
        ],
    } | fields


UNANSWERABLE = _question(
    "q002",
    question="Why did Paul Volcker raise rates in 1979?",
    as_of=None,
    type="unanswerable",
    split="dev",
    answerable=False,
    human_reviewed=False,
    reference_answer=None,
    required_facts=[],
    relevant_chunks=[],
    relevant_doc_ids=[],
    runs=[
        {
            "method": "hybrid_rerank",
            "config": "tuned",
            "latency_ms": 250.0,
            "scores": {},
            "retrieved": [{"chunk_id": f"{OTHER}#0001", "score": 0.1}],
            "answer": {
                "text": "The provided passages do not say.",
                "citations": [],
                "judge": {
                    "faithfulness": None,
                    "citation_accuracy": None,
                    "answer_correctness": None,
                    "refusal_correct": True,
                    "false_refusal": None,
                    "refusal": {"declined": True, "fabricated": False, "reason": "Declines."},
                },
            },
        }
    ],
)


def _file(**fields: Any) -> dict:
    methods = ("bm25", "hybrid_rerank")
    configs = ("phase3", "tuned")
    return {
        "generated_at": "2026-10-13T12:00:00Z",
        "dataset": {
            "file": "data/eval/rag_questions.jsonl",
            "sha256": "ab" * 32,
            "n_questions": 2,
            "n_answerable": 1,
            "n_with_as_of": 1,
            "n_human_reviewed": 1,
            "splits": [
                {"split": "dev", "n": 1, "n_answerable": 0},
                {"split": "test", "n": 1, "n_answerable": 1},
            ],
            "types": [
                {"type": "date_specific", "n": 1, "n_answerable": 1},
                {"type": "unanswerable", "n": 1, "n_answerable": 0},
            ],
            "construction": "Drafted by LLMs from the indexed chunks; one reviewed by a human.",
        },
        "configs": [
            {
                "id": "phase3",
                "label": "Phase 3 settings",
                "description": "As shipped.",
                "hash": "1a2b3c",
                "params": {"rrf_k": 60, "stemming": False},
            },
            {
                "id": "tuned",
                "label": "Tuned",
                "description": "Stemming and statement boost.",
                "params": {"rrf_k": 60, "stemming": True},
            },
        ],
        "generator": {
            "model": "gpt-5.4-mini-2026-03-17",
            "reasoning_effort": "low",
            "prompt_version": "answer-v4",
            "top_k": 5,
        },
        "judge": {
            "model": "gpt-5.5",
            "reasoning_effort": "medium",
            "prompt_version": "judge-v1",
            "n_answers": 4,
            "validation": {
                "status": "complete",
                "n_sampled": 20,
                "n_annotated": 20,
                "annotator": "one human annotator, blind to the verdicts",
                "sample_file": "data/eval/rag_judge_sample.jsonl",
                "agreement": [{"check": "claim_support", "n": 61, "agreement": 0.9, "kappa": 0.71}],
            },
        },
        "bootstrap": {
            "method": "Percentile bootstrap",
            "resamples": 10000,
            "seed": 7,
            "level": 0.95,
        },
        "selection": {
            "method": "hybrid_rerank",
            "config": "tuned",
            "metric": "ndcg_at_10",
            "split": "dev",
            "rule": "Highest dev nDCG@10; ties go to the faster method.",
        },
        "retrieval": [_retrieval_row(m, c) for m in methods for c in configs],
        "answers": [_answer_row("hybrid_rerank", c) for c in configs],
        "per_type": [
            {
                "method": "bm25",
                "config": "phase3",
                "split": "all",
                "type": "date_specific",
                "n_questions": 1,
                "recall_at_5": _estimate(0.5, 1),
            }
        ],
        "differences": [
            {
                "metric": "recall_at_5",
                "split": "all",
                "minuend": {"method": "hybrid_rerank", "config": "tuned"},
                "subtrahend": {"method": "hybrid_rerank", "config": "phase3"},
                "n": 2,
                "difference": 0.25,
                "ci": {"low": -0.1, "high": 0.5},
            }
        ],
        "cost": {
            "total_usd": 0.42,
            "max_usd": 1.0,
            "lines": [
                {
                    "stage": "judge",
                    "model": "gpt-5.5",
                    "calls": 8,
                    "input_tokens": 12000,
                    "cached_input_tokens": 0,
                    "output_tokens": 3000,
                    "usd": 0.4,
                }
            ],
        },
        "notes": ["Small n: intervals are wide."],
        "questions": [_question(), copy.deepcopy(UNANSWERABLE)],
    } | fields


def _chunk(chunk_id: str, text: str) -> dict[str, Any]:
    doc_id = doc_id_of(chunk_id)
    statement = doc_id.startswith("statement")
    day = doc_id.split("-", 1)[1]
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "doc_type": "statement" if statement else "minutes",
        "meeting_date": day,
        "release_date": day if statement else "2024-10-09",
        "page_updated": day,
        "title": f"{'FOMC Statement' if statement else 'Minutes'}, {day}",
        "url": f"https://www.federalreserve.gov/{doc_id}.htm",
        "section": None if statement else "Committee Policy Actions",
        "index": int(chunk_id.rsplit("#", 1)[1]),
        "text": text,
        "context": doc_id,
    }


INDEX_CHUNKS = [
    _chunk(f"{STATEMENT}#0000", "The Committee decided to lower the target range by 1/2 point."),
    _chunk(f"{MINUTES}#0012", "Participants discussed the labor market."),
    _chunk(f"{MINUTES}#0013", "Governor Bowman voted against the action."),
    _chunk(f"{MINUTES}#0099", "A footnote."),
    _chunk(f"{OTHER}#0003", "Inflation had eased."),
]


def _write_file(data_dir: Path, content: dict | str) -> Path:
    path = data_dir / "eval" / "rag_benchmark.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
    return path


def _write_index(index_dir: Path, chunks: list[dict[str, Any]]) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    lines = "".join(json.dumps(c) + "\n" for c in chunks)
    (index_dir / CHUNKS_FILE).write_text(lines, encoding="utf-8")
    manifest = {
        "embedding_model": "text-embedding-3-small",
        "collection": "fedrag_chunks",
        "chunker_version": 2,
        "chunk_count": len(chunks),
        "document_count": 3,
        "corpus_sha256": "0" * 64,
        "chunks_sha256": "1" * 64,
        "source": "processed",
        "built_at": "2026-10-13T00:00:00Z",
    }
    (index_dir / MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return tmp_path


# --- GET /api/evaluation ------------------------------------------------------------------------


@pytest.mark.parametrize("url", ["/api/evaluation", "/api/evaluation/questions/q001"])
def test_not_run_yet_is_503(client: TestClient, url: str) -> None:
    response = client.get(url)

    assert response.status_code == 503
    assert (
        response.json()["detail"]
        == EVALUATION_MISSING
        == ("Evaluation not run yet: run make rag-eval.")
    )


def test_report_serves_the_file_with_questions_summarised(
    client: TestClient, data_dir: Path
) -> None:
    _write_file(data_dir, _file())

    response = client.get("/api/evaluation")

    assert response.status_code == 200
    body = response.json()
    assert [c["id"] for c in body["configs"]] == ["phase3", "tuned"]
    assert len(body["retrieval"]) == 4 and len(body["answers"]) == 2
    row = body["retrieval"][0]
    assert row["recall_at_5"] == {"value": 0.5, "ci": {"low": 0.3, "high": 0.6}, "n": 2}
    assert body["selection"]["config"] == "tuned"
    assert body["judge"]["validation"]["agreement"][0]["kappa"] == 0.71
    assert body["differences"][0]["minuend"] == {"method": "hybrid_rerank", "config": "tuned"}
    # Every row field is serialized, defaults included.
    assert body["per_type"][0]["faithfulness"] is None
    assert body["questions"][0] == {
        "id": "q001",
        "question": _question()["question"],
        "as_of": "2024-10-01",
        "type": "date_specific",
        "split": "test",
        "answerable": True,
        "human_reviewed": True,
        "runs": [
            {
                "method": "bm25",
                "config": "phase3",
                "first_relevant_rank": 2,
                "recall_at_5": 0.5,
                "ndcg_at_10": 0.61,
                "answered": False,
                "faithfulness": None,
                "citation_accuracy": None,
                "answer_correctness": None,
                "refusal_correct": None,
                "false_refusal": None,
                "declined": None,
            },
            {
                "method": "hybrid_rerank",
                "config": "phase3",
                "first_relevant_rank": 2,
                "recall_at_5": 0.5,
                "ndcg_at_10": 0.61,
                "answered": True,
                "faithfulness": 0.5,
                "citation_accuracy": 0.5,
                "answer_correctness": 1.0,
                "refusal_correct": None,
                "false_refusal": False,
                "declined": False,
            },
            body["questions"][0]["runs"][2],
        ],
    }
    unanswerable = body["questions"][1]["runs"][0]
    assert unanswerable["refusal_correct"] is True and unanswerable["recall_at_5"] is None


def test_report_has_no_answers_or_passages(client: TestClient, data_dir: Path) -> None:
    _write_file(data_dir, _file())

    text = client.get("/api/evaluation").text

    assert "retrieved" not in text and "Governor Bowman dissented [2]" not in text


def test_report_is_reread_when_the_file_changes(client: TestClient, data_dir: Path) -> None:
    _write_file(data_dir, _file())
    assert client.get("/api/evaluation").json()["generated_at"] == "2026-10-13T12:00:00Z"

    _write_file(data_dir, _file(generated_at="2026-10-14T08:30:00Z", notes=[]))

    assert client.get("/api/evaluation").json()["generated_at"] == "2026-10-14T08:30:00Z"


INVALID_FILES = {
    "not-json": ("{", "Evaluation file data/eval/rag_benchmark.json is invalid:"),
    "top-level-list": ("[]", "the top level is not a JSON object"),
    "metric-above-one": (
        _file(retrieval=[_retrieval_row("bm25", "phase3", value=1.5)]),
        "retrieval.0.recall_at_5.value",
    ),
    "unknown-config": (
        _file(retrieval=[_retrieval_row("bm25", "phase4")]),
        "unknown config(s) phase4",
    ),
    "duplicate-row": (
        _file(retrieval=[_retrieval_row("bm25", "phase3")] * 2),
        "duplicate retrieval row: ('bm25', 'phase3', 'all')",
    ),
    "duplicate-question": (
        _file(questions=[_question(), _question()]),
        "duplicate question id: q001",
    ),
    "duplicate-run": (
        _file(questions=[_question(runs=[_run("bm25", "phase3")] * 2)]),
        "duplicate run of q001: ('bm25', 'phase3')",
    ),
    "run-config-unknown": (
        _file(questions=[_question(runs=[_run("bm25", "phase9")])]),
        "unknown config(s) phase9",
    ),
    "bad-question-type": (
        _file(questions=[_question(type="trivia")]),
        "questions.0.type",
    ),
    "no-configs": (_file(configs=[]), "configs"),
}


@pytest.mark.parametrize(("content", "problem"), INVALID_FILES.values(), ids=INVALID_FILES.keys())
def test_invalid_file_is_503_naming_the_problem(
    client: TestClient, data_dir: Path, content: dict | str, problem: str
) -> None:
    _write_file(data_dir, content)

    for url in ("/api/evaluation", "/api/evaluation/questions/q001"):
        response = client.get(url)
        assert response.status_code == 503
        detail = response.json()["detail"]
        assert detail.startswith("Evaluation file data/eval/rag_benchmark.json is invalid: ")
        assert problem in detail


def test_file_check_for_the_cli() -> None:
    file = evaluation_file_from_json(_file())

    assert isinstance(file, EvaluationFile)
    assert build_report(file).questions[0].runs[0].answered is False
    with pytest.raises(ValidationError):
        evaluation_file_from_json(_file(selection={**_file()["selection"], "config": "nope"}))
    with pytest.raises(ValueError, match="top level"):
        evaluation_file_from_json([])


def test_lenient_shapes() -> None:
    """The file accepts what app.evaluation's helpers write: an estimate as {"mean", "ci", "n"}
    (or a bare number), an interval as [low, high], a latency summary, and hits for retrieved."""
    assert Estimate.model_validate(0.8) == Estimate(value=0.8, ci=None, n=None)
    assert Estimate.model_validate({"mean": 0.4, "ci": [0.3, 0.5], "n": 9}) == Estimate(
        value=0.4, ci=MetricInterval(low=0.3, high=0.5), n=9
    )
    row = _retrieval_row("bm25", "phase3")
    del row["latency_ms_p50"], row["latency_ms_p95"]
    row["latency_ms"] = {"p50": 4.0, "p95": 9.5, "mean": 5.0, "n": 2}
    run = _run("bm25", "phase3", answered=False)
    run["hits"] = [{"chunk_id": "a#0001", "doc_id": "a", "score": 1.0, "release_date": "2024-01"}]
    del run["retrieved"]
    file = evaluation_file_from_json(_file(retrieval=[row], questions=[_question(runs=[run])]))
    assert (file.retrieval[0].latency_ms_p50, file.retrieval[0].latency_ms_p95) == (4.0, 9.5)
    assert file.questions[0].runs[0].retrieved[0].chunk_id == "a#0001"
    assert Estimate.model_validate(None).value is None
    assert Estimate.model_validate({"value": 0.8, "ci": [0.6, 0.9], "n": 12}).ci == (
        MetricInterval(low=0.6, high=0.9)
    )
    with pytest.raises(ValidationError, match="low must not be above high"):
        MetricInterval.model_validate([0.9, 0.1])
    with pytest.raises(ValidationError):
        Estimate.model_validate(True)


# --- GET /api/evaluation/questions/{question_id} ----------------------------------------------


def test_question_detail_with_passages(make_client: MakeClient, tmp_path: Path) -> None:
    client = make_client()
    _write_file(tmp_path, _file())
    _write_index(tmp_path / "index", INDEX_CHUNKS)

    response = client.get("/api/evaluation/questions/q001")

    assert response.status_code == 200
    body = response.json()
    assert body["passages_available"] is True
    assert body["required_facts"] == ["Cut by 50 basis points", "Bowman dissented"]
    assert "relevant_chunks" not in body
    # Gold: primary first.
    assert body["gold"] == [
        {"chunk_id": f"{STATEMENT}#0000", "role": "primary"},
        {"chunk_id": f"{MINUTES}#0013", "role": "supporting"},
    ]
    # Every chunk mentioned, once, as the reader sees it: gold first, then retrieved, then cited.
    passages = body["passages"]
    assert list(passages) == [
        f"{MINUTES}#0013",
        f"{STATEMENT}#0000",
        f"{OTHER}#0003",
        f"{MINUTES}#0012",
        f"{MINUTES}#0099",
    ]
    primary = passages[f"{STATEMENT}#0000"]
    assert primary["chunk_id"] == f"{STATEMENT}#0000"
    assert primary["text"].startswith("The Committee decided")
    assert primary["section"] is None
    assert primary["document"]["doc_id"] == STATEMENT
    assert primary["document"]["url"].endswith(f"{STATEMENT}.htm")
    assert passages[f"{MINUTES}#0013"]["section"] == "Committee Policy Actions"

    runs = {(r["method"], r["config"]): r for r in body["runs"]}
    assert list(runs) == [
        ("bm25", "phase3"),
        ("hybrid_rerank", "phase3"),
        ("hybrid_rerank", "tuned"),
    ]
    hits = runs["bm25", "phase3"]["retrieved"]
    assert [(h["rank"], h["role"], h["doc_match"]) for h in hits] == [
        (1, None, False),
        (2, "primary", True),
        (3, None, True),
        (4, "supporting", True),
    ]
    assert hits[0]["doc_id"] == OTHER and hits[0]["score"] == 9.1
    assert "passage" not in hits[0]
    assert runs["bm25", "phase3"]["answer"] is None

    answer = runs["hybrid_rerank", "tuned"]["answer"]
    assert [(c["n"], c["rank"], c["role"], c["doc_match"]) for c in answer["citations"]] == [
        (1, 2, "primary", True),
        (2, None, None, True),  # cited but not among the stored top 10
    ]
    assert passages[answer["citations"][1]["chunk_id"]]["text"] == "A footnote."
    assert answer["judge"]["claims"][1]["supported"] is False
    assert answer["judge"]["citations"][0]["reason"] is None


def test_question_detail_without_index_keeps_markers(client: TestClient, data_dir: Path) -> None:
    _write_file(data_dir, _file())

    body = client.get("/api/evaluation/questions/q001").json()

    assert body["passages_available"] is False and body["passages"] == {}
    assert [g["role"] for g in body["gold"]] == ["primary", "supporting"]
    hits = body["runs"][0]["retrieved"]
    assert [h["role"] for h in hits] == [None, "primary", None, "supporting"]


def test_question_detail_skips_chunks_the_index_lacks(
    make_client: MakeClient, tmp_path: Path
) -> None:
    client = make_client()
    _write_file(tmp_path, _file())
    _write_index(tmp_path / "index", INDEX_CHUNKS[:1])

    body = client.get("/api/evaluation/questions/q001").json()

    assert body["passages_available"] is True
    assert list(body["passages"]) == [f"{STATEMENT}#0000"]


def test_question_detail_picks_up_an_index_built_later(
    make_client: MakeClient, tmp_path: Path
) -> None:
    client = make_client()
    _write_file(tmp_path, _file())
    assert client.get("/api/evaluation/questions/q001").json()["passages_available"] is False

    _write_index(tmp_path / "index", INDEX_CHUNKS)

    assert client.get("/api/evaluation/questions/q001").json()["passages_available"] is True


def test_unanswerable_question_detail(client: TestClient, data_dir: Path) -> None:
    _write_file(data_dir, _file())

    body = client.get("/api/evaluation/questions/q002").json()

    assert body["answerable"] is False and body["gold"] == []
    run = body["runs"][0]
    assert run["scores"]["recall_at_5"] is None
    assert run["retrieved"][0]["role"] is None and run["retrieved"][0]["doc_match"] is False
    assert run["answer"]["judge"]["refusal"]["declined"] is True


def test_unknown_question_is_404(client: TestClient, data_dir: Path) -> None:
    _write_file(data_dir, _file())

    response = client.get("/api/evaluation/questions/q999")

    assert response.status_code == 404
    assert response.json()["detail"] == "No evaluation question with id q999."


@pytest.mark.parametrize("question_id", ["bad%20id", "-q1", "q1%23x", "x" * 65])
def test_malformed_question_id_is_422(client: TestClient, question_id: str) -> None:
    assert client.get(f"/api/evaluation/questions/{question_id}").status_code == 422


def test_first_relevant_rank_is_derived_when_not_recorded(
    client: TestClient, data_dir: Path
) -> None:
    run = _run("bm25", "phase3", answered=False)
    run["scores"] = {"recall_at_5": 1.0}  # as app.evaluation's question_metrics: no rank
    _write_file(data_dir, _file(questions=[_question(runs=[run])]))

    summary = client.get("/api/evaluation").json()["questions"][0]["runs"][0]
    detail = client.get("/api/evaluation/questions/q001").json()["runs"][0]

    assert summary["first_relevant_rank"] == 2 == detail["scores"]["first_relevant_rank"]


def test_doc_id_of() -> None:
    assert doc_id_of("minutes-2024-09-18#0013") == "minutes-2024-09-18"
    assert doc_id_of("no-hash") == "no-hash"


def test_openapi_documents_the_evaluation_routes(client: TestClient) -> None:
    schema = client.app.openapi()
    report = schema["paths"]["/api/evaluation"]["get"]
    question = schema["paths"]["/api/evaluation/questions/{question_id}"]["get"]
    error = {"$ref": "#/components/schemas/ErrorResponse"}

    assert report["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/EvaluationReport"
    }
    assert report["responses"]["503"]["content"]["application/json"]["schema"] == error
    for code in ("404", "503"):
        assert question["responses"][code]["content"]["application/json"]["schema"] == error
    assert question["parameters"][0]["schema"]["pattern"]
    # The file-only models are not part of the published contract.
    components = schema["components"]["schemas"]
    assert {"QuestionDetail", "QuestionSummary", "RetrievalRow", "EvalPassage"} <= set(components)
    assert not {"EvaluationFile", "QuestionRecord", "RunRecord"} & set(components)
