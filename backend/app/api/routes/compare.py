"""POST /api/compare."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.compare import COMPARE_RESPONSES, ComparisonAnalyzer
from app.api.database import Db
from app.compare import service
from app.core.config import Settings, get_settings
from app.schemas.compare import CompareRequest, CompareResponse

router = APIRouter(tags=["compare"])


@router.post(
    "/compare",
    operation_id="compareStatements",
    summary="Compare two FOMC statements",
    responses=COMPARE_RESPONSES,
)
def compare_statements(
    request: CompareRequest,
    db: Db,
    analyzer: ComparisonAnalyzer,
    settings: Annotated[Settings, Depends(get_settings)],
) -> CompareResponse:
    """Sentence-level diff of the FOMC statements of two meetings, looked up by meeting date,
    with an LLM reading of what the changed wording signals and each statement's stored stance.

    The diff is deterministic and needs no LLM: sentences are aligned in order, each item is
    unchanged, modified (with a word-level diff), added or removed, and ids index `changes`.
    The analysis reads only the changed items and the target statement, as of the target's
    release date; it is stored per pair, so `analysis="generate"` normally pays for one LLM call
    per pair (a second when the first reply cannot be parsed); a failed generation is not
    stored, so asking again pays again. Without OPENAI_API_KEY the diff is still returned
    (analysis_status llm_not_configured, unless an analysis is already stored). The public demo
    never writes one: there the status of a pair without a stored analysis is disabled_on_demo.
    """
    # A plain def: FastAPI runs it in its threadpool, so the diff and the LLM call never hold up
    # the event loop.
    try:
        return service.compare_statements(
            db,
            request,
            analyzer,
            llm_configured=settings.llm_configured,
            demo=settings.public_demo,
        )
    except service.StatementNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from None
