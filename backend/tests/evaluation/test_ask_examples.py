"""The recorded example answers: cached answers resolved like an AskResponse and checked against
the cache and the benchmark (fake retriever and chat model; no OpenAI)."""

from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

import pytest
from eval_fakes import CHUNKS, QUESTIONS, FakeAnswerLLM, FakeRetriever

from app.core.config import Settings
from app.evaluation import ask_examples
from app.evaluation.ask_examples import METHOD, ExampleExportError, Pick, build_examples
from app.evaluation.questions import EvalQuestion
from app.evaluation.runner import AnswerItem, AnswerRunner, RetrieverSetup, resolve_config
from app.rag.answer import trim_snippet
from app.schemas.ask import AskExamplesFile
from app.schemas.evaluation import (
    AnswerRecord,
    ClaimVerdict,
    EvaluationFile,
    FactVerdict,
    JudgeInfo,
    JudgeVerdicts,
    QuestionRecord,
    RunRecord,
    RunScores,
)

QUESTION_BY_ID = {q["id"]: EvalQuestion.model_validate(q) for q in QUESTIONS}
D1, D2 = QUESTION_BY_ID["d1"], QUESTION_BY_ID["d2"]
PICKS = (Pick("d2", "Minutes quantifier", "Some participants."), Pick("d1", "Dissent", "A vote."))
# Cites the second passage first: the export must renumber it as POST /api/ask does.
REPLY = "Some participants preferred a smaller cut [2], and the range was lowered [1][2]."


@pytest.fixture
def setup(settings: Settings) -> RetrieverSetup:
    return resolve_config("phase3", settings)


@pytest.fixture
def answers(settings: Settings, setup: RetrieverSetup) -> AnswerRunner:
    """A runner whose cache holds a hybrid_rerank answer to d1 and d2."""
    runner = AnswerRunner(settings, lambda s, m: FakeRetriever(), llm=FakeAnswerLLM(reply=REPLY))
    runner.generate(runner.prepare([AnswerItem(q, METHOD, setup) for q in (D1, D2)]))
    return runner


def record(answers: AnswerRunner, setup: RetrieverSetup, question: EvalQuestion) -> dict:
    cached = answers.cached(AnswerItem(question, METHOD, setup))
    assert cached is not None
    return cached


def benchmark(answers: AnswerRunner, setup: RetrieverSetup, **text: str) -> EvaluationFile:
    """An evaluation file with only what the export reads: the run of each question, judged."""
    verdicts = JudgeVerdicts(
        faithfulness=1.0,
        citation_accuracy=0.5,
        answer_correctness=0.5,
        refusal_correct=None,
        false_refusal=False,
        claims=[
            ClaimVerdict(claim="a", supported=True),
            ClaimVerdict(claim="b", supported=False),
        ],
        facts=[FactVerdict(fact="x", covered=True), FactVerdict(fact="y", covered=False)],
    )
    questions = []
    for question in (D1, D2):
        cached = record(answers, setup, question)
        answer = AnswerRecord(
            text=text.get(question.id, cached["answer"]),
            citations=cached["citations"],
            notes=cached["notes"],
            judge=verdicts,
        )
        run = RunRecord(
            method=METHOD, config=setup.name, scores=RunScores(), retrieved=[], answer=answer
        )
        questions.append(
            QuestionRecord(
                id=question.id,
                question=question.question,
                as_of=question.as_of,
                type=question.type,
                split=question.split,
                answerable=question.answerable,
                runs=[run],
            )
        )
    judge = JudgeInfo(model="judge-model", prompt_version="judge-v1", n_answers=2)
    fields: dict[str, Any] = {
        "generated_at": datetime(2026, 10, 13, tzinfo=UTC),
        "judge": judge,
        "questions": questions,
    }
    return EvaluationFile.model_construct(**fields)  # the report sections are not read


def test_examples_resolve_like_ask_responses(answers: AnswerRunner, setup: RetrieverSetup) -> None:
    exported = build_examples(PICKS, QUESTION_BY_ID, answers, setup, CHUNKS, None)

    assert [e.id for e in exported.examples] == ["d2", "d1"]  # the picks' order
    d2 = exported.examples[0]
    cached = record(answers, setup, D2)
    assert d2.answer == cached["answer"]
    assert d2.answer.startswith("Some participants preferred a smaller cut [1]")
    cited = CHUNKS[cached["citations"][0]]
    first = d2.citations[0]
    assert (first.chunk_id, first.section, first.snippet) == (
        cited.chunk_id,
        cited.section,
        trim_snippet(cited.text),
    )
    assert first.document.release_date == cited.release_date
    assert first.passage is None  # the snippet is the whole (short) passage
    scores = {p["chunk_id"]: p["score"] for p in cached["passages"]}
    assert [c.score for c in d2.citations] == [scores[c.chunk_id] for c in d2.citations]
    assert (d2.label, d2.question, d2.question_type, d2.as_of) == (
        "Minutes quantifier",
        D2.question,
        "minutes_fact",
        None,
    )
    assert (d2.retrieval, d2.retriever_preset, d2.model) == (METHOD, "phase3", answers.model)
    assert d2.recorded_at == datetime.fromisoformat(cached["created_at"])
    assert d2.judge is None  # no benchmark
    assert exported.examples[1].as_of == D1.as_of
    assert (exported.retriever_config_hash, exported.top_k) == (setup.hash, 5)
    # What render writes is what GET /api/ask/examples validates.
    assert AskExamplesFile.model_validate_json(ask_examples.render(exported)) == exported


def test_the_judge_comes_from_the_benchmark(answers: AnswerRunner, setup: RetrieverSetup) -> None:
    exported = build_examples(
        PICKS, QUESTION_BY_ID, answers, setup, CHUNKS, benchmark(answers, setup)
    )

    judge = exported.examples[0].judge
    assert judge is not None
    assert (judge.model, judge.faithfulness, judge.citation_accuracy) == ("judge-model", 1.0, 0.5)
    assert (judge.n_claims, judge.n_claims_supported, judge.n_facts, judge.n_facts_covered) == (
        2,
        1,
        2,
        1,
    )
    assert exported.benchmark_generated_at == datetime(2026, 10, 13, tzinfo=UTC)


def test_a_stale_benchmark_is_refused(answers: AnswerRunner, setup: RetrieverSetup) -> None:
    stale = benchmark(answers, setup, d1="An older answer [1].")

    with pytest.raises(ExampleExportError, match="d1: .*another phase3/hybrid_rerank answer"):
        build_examples(PICKS, QUESTION_BY_ID, answers, setup, CHUNKS, stale)


def test_an_edited_cache_record_is_refused(answers: AnswerRunner, setup: RetrieverSetup) -> None:
    key = answers.key(AnswerItem(D1, METHOD, setup))
    answers.cache.put(key, {**record(answers, setup, D1), "answer": "Edited by hand [1]."})

    with pytest.raises(ExampleExportError, match="d1: re-resolving the cached generation"):
        build_examples(PICKS, QUESTION_BY_ID, answers, setup, CHUNKS, None)


@pytest.mark.parametrize(
    ("picks", "message"),
    [
        ((Pick("t1", "Statement fact", "Not answered."),), "t1: no cached phase3/hybrid_rerank"),
        ((Pick("zz-01", "Unknown", "Not a question."),), "zz-01: not in the question set"),
    ],
    ids=["not-answered", "unknown-id"],
)
def test_a_missing_answer_or_question_is_refused(
    answers: AnswerRunner, setup: RetrieverSetup, picks: tuple[Pick, ...], message: str
) -> None:
    with pytest.raises(ExampleExportError, match=message):
        build_examples(picks, QUESTION_BY_ID, answers, setup, CHUNKS, None)


def test_a_passage_missing_from_the_index_is_refused(
    answers: AnswerRunner, setup: RetrieverSetup
) -> None:
    cited = record(answers, setup, D2)["citations"][0]
    chunks = {cid: chunk for cid, chunk in CHUNKS.items() if cid != cited}

    with pytest.raises(ExampleExportError, match=f"not in the search index: .*{cited}"):
        build_examples(PICKS, QUESTION_BY_ID, answers, setup, chunks, None)


def test_a_trimmed_snippet_comes_with_the_whole_passage(
    answers: AnswerRunner, setup: RetrieverSetup
) -> None:
    """A snippet keeps only the first ~600 characters, so the sentence an answer relies on may be
    cut off: a recorded answer's citation then carries the whole passage too."""
    long_id = record(answers, setup, D2)["citations"][0]
    tail = "Runoff of Treasury securities was capped at $25 billion per month."
    text = " ".join(["Participants discussed the economic outlook at some length."] * 15 + [tail])
    chunks = {**CHUNKS, long_id: replace(CHUNKS[long_id], text=text)}

    exported = build_examples(PICKS, QUESTION_BY_ID, answers, setup, chunks, None)

    citations = {c.chunk_id: c for c in exported.examples[0].citations}
    trimmed = citations.pop(long_id)
    assert tail not in trimmed.snippet and text.startswith(trimmed.snippet)
    assert trimmed.passage == text
    assert citations and all(c.passage is None for c in citations.values())
    assert AskExamplesFile.model_validate_json(ask_examples.render(exported)) == exported
