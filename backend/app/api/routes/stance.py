"""GET /api/stance, GET /api/stance/benchmark and GET /api/stance/{meeting_date}."""

from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status

from app.api.database import Db
from app.api.stance import (
    BENCHMARK_RESPONSES,
    STANCE_NOT_LOADED,
    STANCE_RESPONSES,
    BenchmarkUnavailableError,
    load_benchmark,
    reference_labels,
)
from app.core.config import Settings, get_settings
from app.db import stance as stance_db
from app.schemas.common import ErrorResponse, StanceClassifier
from app.schemas.stance import (
    CLASSIFIER_QUERY,
    StanceBenchmark,
    StanceResult,
    StanceTimeline,
    StanceTimelineQuery,
)

router = APIRouter(tags=["stance"])

SettingsDep = Annotated[Settings, Depends(get_settings)]


@router.get(
    "/stance",
    operation_id="getStanceTimeline",
    summary="Policy stance timeline",
    responses=STANCE_RESPONSES,
)
def get_stance_timeline(
    query: Annotated[StanceTimelineQuery, Query()], db: Db, settings: SettingsDep
) -> StanceTimeline:
    """Hawkish/neutral/dovish stance of every statement in a date range according to one
    classifier, with each statement's reference label and the federal funds target range.

    Scores and confidences are model outputs, not calibrated probabilities.
    """
    try:
        with db.connection() as conn:
            points = stance_db.stance_timeline(conn, query.classifier, query.start, query.end)
            fed_funds = stance_db.fed_funds_window(conn, query.start, query.end)
    except stance_db.StanceNotLoadedError:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, STANCE_NOT_LOADED) from None
    references = reference_labels(settings)
    return StanceTimeline(
        classifier=query.classifier,
        points=[p.model_copy(update={"reference": references.get(p.doc_id)}) for p in points],
        fed_funds=fed_funds,
    )


# Declared before /stance/{meeting_date}, which would otherwise match "benchmark" (as a bad date).
@router.get(
    "/stance/benchmark",
    operation_id="getStanceBenchmark",
    summary="Stance classifier benchmark",
    responses=BENCHMARK_RESPONSES,
)
def get_stance_benchmark(settings: SettingsDep) -> StanceBenchmark:
    """Scores of the four stance classifiers on the Trillion Dollar Words sentence test splits
    (official and deduplicated, mean and sd over three seeds) and on whole FOMC statements
    against the reference labels."""
    try:
        return load_benchmark(settings)
    except BenchmarkUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from None


@router.get(
    "/stance/{meeting_date}",
    operation_id="getStanceForMeeting",
    summary="Policy stance for one meeting",
    responses={
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "No FOMC statement for a meeting on that date.",
        },
        **STANCE_RESPONSES,
    },
)
def get_stance_for_meeting(
    meeting_date: Annotated[date, Path(description="FOMC meeting date (YYYY-MM-DD).")],
    db: Db,
    settings: SettingsDep,
    classifier: Annotated[
        StanceClassifier,
        Query(description=CLASSIFIER_QUERY),
    ] = "llm_few_shot",
) -> StanceResult:
    """Stance of a single meeting's statement, with the evidence behind it and its reference
    label."""
    try:
        with db.connection() as conn:
            result = stance_db.stance_for_meeting(conn, meeting_date, classifier)
    except stance_db.StanceNotLoadedError:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, STANCE_NOT_LOADED) from None
    if result is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"No stance result for {meeting_date}: no FOMC statement for a meeting on that date.",
        )
    return result.model_copy(update={"reference": reference_labels(settings).get(result.doc_id)})
