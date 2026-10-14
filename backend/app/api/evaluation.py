"""The RAG evaluation for routes: data/eval/rag_benchmark.json (read-only, re-read whenever it
changes on disk), summarised for the report and expanded per question with the passages' text
from the search index (data/index/chunks.jsonl)."""

import json
import logging
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import status
from pydantic import ValidationError

from app.api.file_cache import VersionedFileCache
from app.core.config import Settings
from app.rag.index import CHUNKS_FILE, IndexNotBuiltError, load_chunks
from app.rag.types import Chunk
from app.schemas.common import ErrorResponse, SourceDocument
from app.schemas.evaluation import (
    AnswerCitation,
    AnswerDetail,
    AnswerRecord,
    ChunkRole,
    EvalPassage,
    EvaluationFile,
    EvaluationReport,
    QuestionDetail,
    QuestionRecord,
    QuestionSummary,
    RetrievedHit,
    RunDetail,
    RunRecord,
    RunSummary,
)

logger = logging.getLogger(__name__)

RAG_BENCHMARK_FILE = Path("eval") / "rag_benchmark.json"  # under settings.data_dir
EVALUATION_MISSING = "Evaluation not run yet: run make rag-eval."
EVALUATION_INVALID = "Evaluation file data/eval/rag_benchmark.json is invalid: {problems}"
QUESTION_NOT_FOUND = "No evaluation question with id {question_id}."

EVALUATION_RESPONSES: dict[int | str, dict[str, Any]] = {
    status.HTTP_503_SERVICE_UNAVAILABLE: {
        "model": ErrorResponse,
        "description": (
            "The evaluation has not been run yet (run make rag-eval), or its file does not fit "
            "the contract."
        ),
    }
}
QUESTION_RESPONSES: dict[int | str, dict[str, Any]] = {
    status.HTTP_404_NOT_FOUND: {
        "model": ErrorResponse,
        "description": "No question with that id in the evaluation.",
    },
    **EVALUATION_RESPONSES,
}

# The report fields an EvaluationFile and an EvaluationReport share (all but `questions`).
_REPORT_FIELDS = tuple(name for name in EvaluationReport.model_fields if name != "questions")
_PRIMARY_FIRST: dict[ChunkRole, int] = {"primary": 0, "supporting": 1}
FIRST_RANK_DEPTH = 10  # first_relevant_rank looks this deep, like MRR@10


class EvaluationUnavailableError(RuntimeError):
    """The evaluation file is missing or does not match the EvaluationFile contract."""


@dataclass(frozen=True)
class Evaluation:
    """One version of the evaluation file: as stored, and as GET /api/evaluation serves it."""

    file: EvaluationFile
    report: EvaluationReport
    questions: Mapping[str, QuestionRecord]


_evaluations = VersionedFileCache()
_chunk_tables = VersionedFileCache()


def evaluation_file_from_json(raw: Any) -> EvaluationFile:
    """The evaluation file's contents as an EvaluationFile; raises ValidationError, or ValueError,
    when they do not fit the contract (the evaluation CLI can check a file before writing it)."""
    if not isinstance(raw, dict):
        raise ValueError("the top level is not a JSON object")
    return EvaluationFile.model_validate(raw)


def build_report(file: EvaluationFile) -> EvaluationReport:
    """The report GET /api/evaluation serves: the file's report fields, questions summarised."""
    return EvaluationReport(
        **{name: getattr(file, name) for name in _REPORT_FIELDS},
        questions=[summarize_question(q) for q in file.questions],
    )


def summarize_question(question: QuestionRecord) -> QuestionSummary:
    gold = {c.chunk_id for c in question.relevant_chunks}
    return QuestionSummary(
        id=question.id,
        question=question.question,
        as_of=question.as_of,
        type=question.type,
        split=question.split,
        answerable=question.answerable,
        human_reviewed=question.human_reviewed,
        runs=[_run_summary(run, gold) for run in question.runs],
    )


def first_relevant_rank(run: RunRecord, gold: Collection[str]) -> int | None:
    """The run's recorded rank of the first gold chunk, else its position in the top 10."""
    if run.scores.first_relevant_rank is not None:
        return run.scores.first_relevant_rank
    top = run.retrieved[:FIRST_RANK_DEPTH]
    return next((rank for rank, hit in enumerate(top, start=1) if hit.chunk_id in gold), None)


def _run_summary(run: RunRecord, gold: Collection[str]) -> RunSummary:
    judge = run.answer.judge if run.answer else None
    return RunSummary(
        method=run.method,
        config=run.config,
        first_relevant_rank=first_relevant_rank(run, gold),
        recall_at_5=run.scores.recall_at_5,
        ndcg_at_10=run.scores.ndcg_at_10,
        answered=run.answer is not None,
        faithfulness=judge.faithfulness if judge else None,
        citation_accuracy=judge.citation_accuracy if judge else None,
        answer_correctness=judge.answer_correctness if judge else None,
        refusal_correct=judge.refusal_correct if judge else None,
        false_refusal=judge.false_refusal if judge else None,
        declined=judge.refusal.declined if judge and judge.refusal else None,
    )


def _parse_evaluation(path: Path) -> Evaluation | str:
    """The validated evaluation, or the error message to report (cached too, so a bad file is
    parsed once per version)."""
    try:
        file = evaluation_file_from_json(json.loads(path.read_text(encoding="utf-8")))
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(map(str, err['loc'])) or 'file'}: {err['msg']}"
            for err in exc.errors(include_url=False)[:5]
        )
        return EVALUATION_INVALID.format(problems=problems)
    except (ValueError, TypeError, OSError) as exc:
        return EVALUATION_INVALID.format(problems=exc)
    return Evaluation(
        file=file, report=build_report(file), questions={q.id: q for q in file.questions}
    )


def load_evaluation(settings: Settings) -> Evaluation:
    """The evaluation as it is on disk now; raises EvaluationUnavailableError when the file is
    missing or invalid."""
    evaluation = _evaluations.get(settings.data_dir / RAG_BENCHMARK_FILE, _parse_evaluation)
    if evaluation is None:
        raise EvaluationUnavailableError(EVALUATION_MISSING)
    if isinstance(evaluation, str):
        raise EvaluationUnavailableError(evaluation)
    return evaluation


# --- Per-question drilldown --------------------------------------------------------------------


def _parse_chunks(settings: Settings) -> dict[str, Chunk] | None:
    try:
        return {chunk.chunk_id: chunk for chunk in load_chunks(settings)}
    except IndexNotBuiltError as exc:
        logger.info("Evaluation passages unavailable: %s", exc)
        return None


def index_chunks(settings: Settings) -> Mapping[str, Chunk] | None:
    """chunk_id -> chunk of the search index, or None when there is no index to read (not kept,
    so an index built later is picked up)."""
    return _chunk_tables.get(settings.index_dir / CHUNKS_FILE, lambda _: _parse_chunks(settings))


def passage(chunk: Chunk) -> EvalPassage:
    return EvalPassage(
        chunk_id=chunk.chunk_id,
        document=SourceDocument(
            doc_id=chunk.doc_id,
            doc_type=chunk.doc_type,
            meeting_date=chunk.meeting_date,
            release_date=chunk.release_date,
            page_updated=chunk.page_updated,
            title=chunk.title,
            url=chunk.url,
        ),
        section=chunk.section,
        text=chunk.text,
    )


def doc_id_of(chunk_id: str) -> str:
    """The document of a chunk id ({doc_id}#{index:04d})."""
    head, sep, _ = chunk_id.rpartition("#")
    return head if sep else chunk_id


def question_detail(question: QuestionRecord, chunks: Mapping[str, Chunk] | None) -> QuestionDetail:
    """One question with its gold chunks (primary first), hit markers on every retrieved chunk,
    resolved answer citations, and the text of every chunk they mention that `chunks` (the
    search index) has."""
    roles: dict[str, ChunkRole] = {c.chunk_id: c.role for c in question.relevant_chunks}
    relevant_docs = set(question.relevant_doc_ids)
    mentioned = [
        *roles,
        *(hit.chunk_id for run in question.runs for hit in run.retrieved),
        *(cid for run in question.runs if run.answer for cid in run.answer.citations),
    ]
    passages = {
        cid: passage(chunks[cid])
        for cid in dict.fromkeys(mentioned)
        if chunks is not None and cid in chunks
    }
    return QuestionDetail(
        **question.model_dump(exclude={"relevant_chunks", "runs"}),
        gold=sorted(question.relevant_chunks, key=lambda c: _PRIMARY_FIRST[c.role]),
        runs=[_run_detail(run, roles, relevant_docs) for run in question.runs],
        passages=passages,
        passages_available=chunks is not None,
    )


def _run_detail(
    run: RunRecord, roles: Mapping[str, ChunkRole], relevant_docs: set[str]
) -> RunDetail:
    retrieved = [
        RetrievedHit(
            rank=rank,
            chunk_id=hit.chunk_id,
            doc_id=doc_id_of(hit.chunk_id),
            score=hit.score,
            role=roles.get(hit.chunk_id),
            doc_match=doc_id_of(hit.chunk_id) in relevant_docs,
        )
        for rank, hit in enumerate(run.retrieved, start=1)
    ]
    ranks = {hit.chunk_id: hit.rank for hit in reversed(retrieved)}  # first rank of a repeated id
    scores = run.scores.model_copy(
        update={"first_relevant_rank": first_relevant_rank(run, roles.keys())}
    )
    return RunDetail(
        method=run.method,
        config=run.config,
        latency_ms=run.latency_ms,
        scores=scores,
        retrieved=retrieved,
        answer=_answer_detail(run.answer, ranks, roles, relevant_docs) if run.answer else None,
    )


def _answer_detail(
    answer: AnswerRecord,
    ranks: Mapping[str, int],
    roles: Mapping[str, ChunkRole],
    relevant_docs: set[str],
) -> AnswerDetail:
    return AnswerDetail(
        text=answer.text,
        citations=[
            AnswerCitation(
                n=n,
                chunk_id=chunk_id,
                rank=ranks.get(chunk_id),
                role=roles.get(chunk_id),
                doc_match=doc_id_of(chunk_id) in relevant_docs,
            )
            for n, chunk_id in enumerate(answer.citations, start=1)
        ],
        notes=answer.notes,
        latency_ms=answer.latency_ms,
        cost_usd=answer.cost_usd,
        judge=answer.judge,
    )
