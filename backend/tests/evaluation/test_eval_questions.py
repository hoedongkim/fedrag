"""Loading and validating the evaluation set, and converting the smoke questions."""

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from eval_fakes import CHUNKS, QUESTIONS

from app.core.config import Settings
from app.evaluation.questions import (
    EvalQuestion,
    QuestionSetError,
    load_questions,
    validate,
)


def question(**fields: Any) -> EvalQuestion:
    return EvalQuestion.model_validate({**QUESTIONS[0], **fields})


def test_the_evaluation_set_loads_with_counts(questions_file: Path) -> None:
    question_set = load_questions(questions_file, CHUNKS)

    assert question_set.format == "rag_questions"
    assert [q.id for q in question_set.questions] == ["d1", "d2", "d3", "t1"]
    assert question_set.summary() == {
        "n_questions": 4,
        "n_answerable": 3,
        "splits": {"dev": 3, "test": 1},
        "types": {
            "statement_fact": 1,
            "minutes_fact": 1,
            "point_in_time": 0,
            "date_specific": 1,
            "multi_doc": 0,
            "paraphrase": 0,
            "unanswerable": 1,
        },
    }
    assert question_set.warnings == ()
    assert len(question_set.sha256) == 64


def test_select_filters_by_split_and_ids(questions_file: Path) -> None:
    question_set = load_questions(questions_file, CHUNKS)

    assert [q.id for q in question_set.select("dev")] == ["d1", "d2", "d3"]
    assert [q.id for q in question_set.select("test")] == ["t1"]
    assert [q.id for q in question_set.select("all", ["t1", "d2"])] == ["d2", "t1"]
    with pytest.raises(QuestionSetError, match="unknown question ids: zz"):
        question_set.select("all", ["zz"])


def test_grades_and_fingerprint() -> None:
    q = EvalQuestion.model_validate(QUESTIONS[1])

    assert q.grades == {"minutes-2024-09-18#0001": 2, "minutes-2024-09-18#0002": 1}
    assert q.fingerprint != question(as_of="2024-10-02").fingerprint
    assert q.fingerprint == EvalQuestion.model_validate(QUESTIONS[1]).fingerprint


@pytest.mark.parametrize(
    ("fields", "problem"),
    [
        ({"type": "unanswerable"}, "type unanswerable but answerable is true"),
        ({"relevant_chunks": []}, "answerable but no relevant chunk"),
        (
            {"relevant_chunks": [{"chunk_id": "statement-2024-09-18#0000", "role": "primary"}] * 2},
            "listed more than once",
        ),
        (
            {"relevant_chunks": [{"chunk_id": "statement-2024-09-18#0009", "role": "primary"}]},
            "is not in the index",
        ),
        (
            {
                "relevant_chunks": [{"chunk_id": "minutes-2024-09-18#0001", "role": "primary"}],
                "relevant_doc_ids": ["minutes-2024-09-18"],
            },
            "released 2024-10-09, after as_of 2024-10-01",
        ),
        ({"relevant_doc_ids": ["statement-2024-07-31"]}, "not in relevant_doc_ids"),
        ({"relevant_doc_ids": ["statement-2024-09-18", "statement-2030-01-01"]}, "document "),
    ],
    ids=[
        "unanswerable-type",
        "no-chunks",
        "repeated-chunk",
        "unknown-chunk",
        "after-as-of",
        "doc-mismatch",
        "unknown-doc",
    ],
)
def test_validation_errors(fields: dict[str, Any], problem: str) -> None:
    result = validate([question(**fields)], CHUNKS)

    assert any(problem in error for error in result.errors), result.errors


def test_repeated_ids_are_an_error_and_warnings_do_not_block(tmp_path: Path) -> None:
    rows = [
        QUESTIONS[0],
        {**QUESTIONS[0], "required_facts": [], "reviewer": "hk"},
    ]
    path = tmp_path / "q.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    with pytest.raises(QuestionSetError, match="question id 'd1' appears 2 times"):
        load_questions(path, CHUNKS)
    unchecked = load_questions(path, CHUNKS, check=False)
    assert unchecked.errors == ("question id 'd1' appears 2 times",)
    assert "d1: unknown fields ['reviewer']" in unchecked.warnings
    assert any("no required facts" in w for w in unchecked.warnings)


def test_optional_facts_are_a_known_field() -> None:
    q = question(optional_facts=["the vote was 11-1"])
    result = validate([q], CHUNKS)

    assert q.optional_facts == ("the vote was 11-1",)
    assert question().optional_facts == ()
    assert result.errors == [] and result.warnings == []


def test_without_the_index_only_the_file_itself_is_checked() -> None:
    result = validate([question(relevant_chunks=[{"chunk_id": "x#0001", "role": "primary"}])])

    assert result.errors == ["d1: chunk x#0001's document x not in relevant_doc_ids"]


def test_a_malformed_row_names_the_field(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps({**QUESTIONS[0], "split": "train"}) + "\n", encoding="utf-8")

    with pytest.raises(QuestionSetError, match="split"):
        load_questions(path)
    with pytest.raises(QuestionSetError, match="not found"):
        load_questions(tmp_path / "missing.jsonl")


def test_the_smoke_file_converts_on_the_fly() -> None:
    smoke = Settings(_env_file=None).data_dir / "eval" / "smoke.jsonl"
    question_set = load_questions(smoke)

    assert question_set.format == "smoke"
    by_id = {q.id: q for q in question_set.questions}
    assert len(by_id) == 12
    assert all(q.split == "dev" for q in by_id.values())
    assert not by_id["q12"].answerable and by_id["q12"].type == "unanswerable"
    q01 = by_id["q01"]
    assert q01.grades == {"statement-2015-12-16#0000": 2}
    assert q01.relevant_doc_ids == ("statement-2015-12-16", "minutes-2015-12-16")
    assert q01.type == "multi_doc" and q01.required_facts == ()
    assert by_id["q02"].relevant_chunks == () and by_id["q02"].as_of == date(2016, 7, 10)
    assert by_id["q02"].type == "point_in_time"
    assert question_set.errors == ()  # answerable with document labels only is fine for smoke
