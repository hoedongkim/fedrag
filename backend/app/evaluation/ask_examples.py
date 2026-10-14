"""The recorded answers GET /api/ask/examples serves: hand-picked answers of the production
configuration from the evaluation's cache, exported to data/eval/ask_examples.json (tracked).

The public demo makes no OpenAI calls, so Ask the Fed shows these instead of answering new
questions. Each one is the answer the production chain gave to a TEST question of the evaluation
set (hybrid + rerank under the production retriever preset, top 5, the benchmark's answer prompt
and model), read from data/eval_cache/answers.jsonl under the cache key the report uses. Its
citations are resolved from the search index the way POST /api/ask resolves them
(app.rag.answer.resolve_citations and answer_notes on the recorded generation), and the result
must match the cached record: answer text, cited chunk ids and notes. Unlike a live answer, each
citation also carries its whole passage when the snippet is only its beginning (passage). The
judge's verdicts are those of data/eval/rag_benchmark.json, whose answer text must match too, so
an example never disagrees with what the RAG Performance page shows for its question. No API
calls.

PICKS was chosen by hand for range, not at random: a statement fact, a minutes quantifier, a
point-in-time question, an answer drawn from two meetings, a paraphrase, a correct decline, and
two more minutes facts. They are among the better answers; the judge's verdicts ship with each,
and the RAG Performance page scores all 55 TEST answers.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.evaluation.questions import EvalQuestion
from app.evaluation.runner import AnswerItem, AnswerRunner, RetrieverSetup
from app.rag.answer import answer_notes, resolve_citations
from app.rag.types import Chunk, RetrievedChunk
from app.schemas.ask import AskExample, AskExamplesFile, AskRequest, Citation, ExampleJudgement
from app.schemas.common import RetrievalMethod
from app.schemas.evaluation import EvaluationFile, JudgeVerdicts, QuestionRecord

# The retrieval method of Ask the Fed's form, which the examples were answered with.
METHOD: RetrievalMethod = AskRequest.model_fields["retrieval"].default


@dataclass(frozen=True)
class Pick:
    """One example: the evaluation question, a short label and why it was picked."""

    question_id: str
    label: str
    description: str


PICKS: tuple[Pick, ...] = (
    Pick(
        "sf-14",
        "Statement fact",
        "The May 2024 decision to slow balance sheet runoff, from the statement and the policy "
        "directive in that meeting's minutes.",
    ),
    Pick(
        "ds-10",
        "Minutes quantifier",
        'The minutes say "some participants": the answer keeps that wording and says the '
        "passages give no count, instead of inventing one.",
    ),
    Pick(
        "pit-04",
        "Point in time",
        'Asked as of July 5, 2017, so "so far" means what was public that day: the June 2017 '
        "minutes, released on that date.",
    ),
    Pick(
        "pp-04",
        "Two meetings",
        "Combines the March 23, 2020 statement with the minutes of the April 2020 meeting.",
    ),
    Pick(
        "pp-05",
        "Paraphrase",
        'Everyday wording ("kicked off QT", "shed") matched to the monthly caps on balance '
        "sheet runoff in the May 2022 minutes.",
    ),
    Pick(
        "ua-06",
        "Declines",
        "Asks about the October 2026 meeting, which no indexed document covers (the newest are "
        "from September 2026): the answer says so rather than guessing.",
    ),
    Pick(
        "mf-13",
        "Vote and terms",
        "How the Committee voted on the July 2021 standing repo facilities, and the domestic "
        "facility's terms, from the minutes.",
    ),
    Pick(
        "ds-05",
        "Staff outlook",
        "The staff's recession risk and inflation projections from the November 2022 minutes, "
        "with their numbers.",
    ),
)

FILE_DESCRIPTION = (
    "Recorded answers of FedRAG's production configuration, served by GET /api/ask/examples and "
    "shown by Ask the Fed on the public demo, which makes no OpenAI calls. Each is the cached "
    "answer from the RAG evaluation (make rag-eval) to one of its TEST questions, exported by "
    "python -m app.evaluation export-ask-examples with its citations resolved from the search "
    "index and the evaluation judge's verdicts from data/eval/rag_benchmark.json."
)


class ExampleExportError(ValueError):
    """An example cannot be exported: its answer is not cached, or the caches disagree."""


def build_examples(
    picks: Sequence[Pick],
    questions: Mapping[str, EvalQuestion],
    runner: AnswerRunner,
    setup: RetrieverSetup,
    chunks: Mapping[str, Chunk],
    evaluation: EvaluationFile | None,
) -> AskExamplesFile:
    """The examples file for `picks`: each pick's cached answer under `setup` and METHOD (from
    `runner`'s cache), resolved like an AskResponse and checked against the cache, with the
    judge's verdicts from `evaluation` (the tracked benchmark; None leaves them out). Raises
    ExampleExportError when a question, answer or passage is missing or the records disagree."""
    records = {q.id: q for q in evaluation.questions} if evaluation else {}
    examples = []
    for pick in picks:
        question = questions.get(pick.question_id)
        if question is None:
            raise ExampleExportError(f"{pick.question_id}: not in the question set")
        record = runner.cached(AnswerItem(question, METHOD, setup))
        if record is None:
            raise ExampleExportError(
                f"{pick.question_id}: no cached {setup.name}/{METHOD} answer: run make rag-eval"
            )
        judge = _benchmark_judgement(
            pick.question_id, record, records.get(pick.question_id), setup, evaluation
        )
        examples.append(_example(pick, question, record, chunks, setup, judge))
    return AskExamplesFile(
        description=FILE_DESCRIPTION,
        retriever_preset=setup.name,
        retriever_config_hash=setup.hash,
        retrieval=METHOD,
        top_k=runner.top_k,
        model=runner.model,
        prompt_version=runner.prompt_version,
        benchmark_generated_at=evaluation.generated_at if evaluation else None,
        examples=examples,
    )


def _example(
    pick: Pick,
    question: EvalQuestion,
    record: Mapping[str, Any],
    chunks: Mapping[str, Chunk],
    setup: RetrieverSetup,
    judge: ExampleJudgement | None,
) -> AskExample:
    """The AskExample of one cached answer: its generation re-resolved over its passages exactly
    as build_response does, which must reproduce the cached answer, citations and notes."""
    missing = [p["chunk_id"] for p in record["passages"] if p["chunk_id"] not in chunks]
    if missing:
        raise ExampleExportError(
            f"{pick.question_id}: passages not in the search index: {', '.join(missing)}"
        )
    passages = [RetrievedChunk(chunks[p["chunk_id"]], p["score"]) for p in record["passages"]]
    request = AskRequest(
        question=question.question,
        as_of=question.as_of,
        retrieval=METHOD,
        top_k=record["top_k"],
    )
    answer, citations = resolve_citations(record["generated"], passages)
    notes = answer_notes(request, citations)
    resolved = (answer, [c.chunk_id for c in citations], notes)
    if resolved != (record["answer"], record["citations"], record["notes"]):
        raise ExampleExportError(
            f"{pick.question_id}: re-resolving the cached generation does not reproduce the "
            "cached answer, citations and notes"
        )
    citations = [_with_passage(c, chunks[c.chunk_id]) for c in citations]
    return AskExample(
        answer=answer,
        citations=citations,
        notes=notes,
        retrieval=METHOD,
        as_of=question.as_of,
        model=record["model"],
        latency_ms=record["latency_ms"],
        id=question.id,
        label=pick.label,
        description=pick.description,
        question=question.question,
        question_type=question.type,
        retriever_preset=setup.name,
        recorded_at=record["created_at"],
        judge=judge,
    )


def _with_passage(citation: Citation, chunk: Chunk) -> Citation:
    """`citation` with the whole passage when its snippet is only the beginning: a snippet ends
    after about 600 characters, so the sentence an answer relies on is often not in it."""
    text = chunk.text.strip()
    return citation.model_copy(update={"passage": text if text != citation.snippet else None})


def _benchmark_judgement(
    question_id: str,
    record: Mapping[str, Any],
    question: QuestionRecord | None,
    setup: RetrieverSetup,
    evaluation: EvaluationFile | None,
) -> ExampleJudgement | None:
    """The benchmark's verdicts on this answer; None without a benchmark or a judged run. The
    benchmark's answer must be this one: a stale benchmark is refused, not quietly mixed in."""
    if evaluation is None or question is None:
        return None
    run = next(
        (r for r in question.runs if r.config == setup.name and r.method == METHOD and r.answer),
        None,
    )
    if run is None or run.answer is None:
        return None
    if run.answer.text != record["answer"] or run.answer.citations != record["citations"]:
        raise ExampleExportError(
            f"{question_id}: data/eval/rag_benchmark.json has another {setup.name}/{METHOD} "
            "answer than the cache: run make rag-eval-report"
        )
    verdicts = run.answer.judge
    return None if verdicts is None else judgement(verdicts, evaluation.judge.model)


def judgement(verdicts: JudgeVerdicts, model: str) -> ExampleJudgement:
    """The summary of one answer's verdicts that an example carries."""
    return ExampleJudgement(
        model=model,
        faithfulness=verdicts.faithfulness,
        citation_accuracy=verdicts.citation_accuracy,
        answer_correctness=verdicts.answer_correctness,
        refusal_correct=verdicts.refusal_correct,
        false_refusal=verdicts.false_refusal,
        n_claims=len(verdicts.claims),
        n_claims_supported=sum(c.supported for c in verdicts.claims),
        n_facts=len(verdicts.facts),
        n_facts_covered=sum(f.covered for f in verdicts.facts),
    )


def render(file: AskExamplesFile) -> str:
    """The file's JSON as written: indented, keys in schema order, non-ASCII kept."""
    return file.model_dump_json(indent=2) + "\n"


def summary_lines(file: AskExamplesFile) -> list[str]:
    """One line per example, for the command's output."""
    lines = []
    for example in file.examples:
        judge = example.judge
        scores = (
            "not judged"
            if judge is None
            else ", ".join(
                f"{name} {value:.2f}" if isinstance(value, float) else f"{name} {value}"
                for name, value in (
                    ("faithfulness", judge.faithfulness),
                    ("citations", judge.citation_accuracy),
                    ("correctness", judge.answer_correctness),
                    ("refusal_correct", judge.refusal_correct),
                )
                if value is not None
            )
        )
        lines.append(
            f"  {example.id:<7} {example.label:<19} {len(example.citations)} citation(s); {scores}"
        )
    return lines
