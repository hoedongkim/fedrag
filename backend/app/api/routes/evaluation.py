"""GET /api/evaluation and GET /api/evaluation/questions/{question_id}."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, status

from app.api.evaluation import (
    EVALUATION_RESPONSES,
    QUESTION_NOT_FOUND,
    QUESTION_RESPONSES,
    EvaluationUnavailableError,
    index_chunks,
    load_evaluation,
    question_detail,
)
from app.core.config import Settings, get_settings
from app.schemas.evaluation import QUESTION_ID_PATTERN, EvaluationReport, QuestionDetail

router = APIRouter(tags=["evaluation"])

SettingsDep = Annotated[Settings, Depends(get_settings)]


@router.get(
    "/evaluation",
    operation_id="getEvaluationReport",
    summary="RAG evaluation report",
    responses=EVALUATION_RESPONSES,
)
def get_evaluation_report(settings: SettingsDep) -> EvaluationReport:
    """Retrieval and answer quality of BM25, dense, hybrid and hybrid + rerank retrieval under
    each retriever config, on the evaluation question set.

    Retrieval: chunk-level recall@5/@10, MRR@10 and graded nDCG@10, document-level recall, per
    question type and split, and latency. Answers (the production chain of POST /api/ask), scored
    by an LLM judge: faithfulness, citation accuracy, answer correctness, refusals. Means come
    with percentile bootstrap intervals over questions, and paired intervals for differences
    between runs. Every question is listed with its outcome per run; GET
    /api/evaluation/questions/{question_id} has the passages and answers. Read from
    data/eval/rag_benchmark.json (make rag-eval); serving it calls no LLM.
    """
    try:
        return load_evaluation(settings).report
    except EvaluationUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from None


@router.get(
    "/evaluation/questions/{question_id}",
    operation_id="getEvaluationQuestion",
    summary="One evaluation question in detail",
    responses=QUESTION_RESPONSES,
)
def get_evaluation_question(
    question_id: Annotated[
        str, Path(pattern=QUESTION_ID_PATTERN, description="Question id, e.g. sf-05.")
    ],
    settings: SettingsDep,
) -> QuestionDetail:
    """One question of the evaluation: its labels, the gold passages, and under each run the top
    retrieved passages with hit markers, the answer, and the judge's verdicts on its claims,
    citations, required facts and refusal.

    Passage text comes from the local search index; without one (make index) the ids and
    markers are still returned, with `passages_available` false.
    """
    try:
        evaluation = load_evaluation(settings)
    except EvaluationUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from None
    question = evaluation.questions.get(question_id)
    if question is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, QUESTION_NOT_FOUND.format(question_id=question_id)
        )
    return question_detail(question, index_chunks(settings))
