"""The tracked evaluation files themselves (no index, no API): the question set loads, the
benchmark fits the API contract, and the benchmark was built from the question file as it is now,
so GET /api/evaluation never serves numbers for an older question set without a failing test; the
recorded example answers are the benchmark's own."""

import hashlib
import json
import re

from app.api.ask_examples import ASK_EXAMPLES_FILE
from app.api.evaluation import evaluation_file_from_json
from app.core.config import REPO_ROOT, Settings
from app.evaluation.ask_examples import METHOD, PICKS, judgement
from app.evaluation.questions import QUESTIONS_FILE, load_questions
from app.evaluation.report import BENCHMARK_FILE
from app.schemas.ask import AskExamplesFile

DATA = REPO_ROOT / "data"


def test_the_question_set_loads_without_errors() -> None:
    question_set = load_questions(DATA / QUESTIONS_FILE)  # raises on any validation error

    assert question_set.format == "rag_questions" and question_set.questions


def test_the_benchmark_fits_the_contract_and_matches_the_question_file() -> None:
    questions = DATA / QUESTIONS_FILE
    report = evaluation_file_from_json(
        json.loads((DATA / BENCHMARK_FILE).read_text(encoding="utf-8"))
    )

    assert report.dataset.sha256 == hashlib.sha256(questions.read_bytes()).hexdigest(), (
        "rag_questions.jsonl changed since rag_benchmark.json was built: run make rag-eval-report"
    )
    assert {q.id for q in report.questions} == {q.id for q in load_questions(questions).questions}


def test_the_ask_examples_are_the_benchmarks_production_answers() -> None:
    """data/eval/ask_examples.json (GET /api/ask/examples) fits the contract, lists PICKS, and
    each example is the benchmark's answer of the production configuration to that question,
    with the benchmark's verdicts: the demo never shows an answer the RAG Performance page
    scores differently."""
    examples = AskExamplesFile.model_validate_json((DATA / ASK_EXAMPLES_FILE).read_bytes())
    report = evaluation_file_from_json(
        json.loads((DATA / BENCHMARK_FILE).read_text(encoding="utf-8"))
    )
    questions = {q.id: q for q in report.questions}
    production = Settings(_env_file=None).retriever_preset

    assert [(e.id, e.label, e.description) for e in examples.examples] == [
        (p.question_id, p.label, p.description) for p in PICKS
    ]
    assert (examples.retriever_preset, examples.retrieval) == (production, METHOD)
    assert (examples.model, examples.prompt_version, examples.top_k) == (
        report.generator.model,
        report.generator.prompt_version,
        report.generator.top_k,
    )
    for example in examples.examples:
        question = questions[example.id]
        run = next(r for r in question.runs if (r.config, r.method) == (production, METHOD))
        assert run.answer is not None and run.answer.judge is not None
        assert question.split == "test"
        assert (example.question, example.as_of, example.question_type) == (
            question.question,
            question.as_of,
            question.type,
        )
        assert example.answer == run.answer.text
        assert [c.chunk_id for c in example.citations] == run.answer.citations
        assert example.notes == run.answer.notes
        assert example.judge == judgement(run.answer.judge, report.judge.model)
        markers = {int(n) for n in re.findall(r"\[(\d+)\]", example.answer)}
        assert markers == set(range(1, len(example.citations) + 1))
