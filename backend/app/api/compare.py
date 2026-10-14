"""Statement Comparison for routes: the analyzer dependency and the documented errors."""

from typing import Annotated, Any

from fastapi import Depends, status

from app.api.database import DATABASE_RESPONSES
from app.compare.analysis import get_analyzer as build_analyzer
from app.compare.types import Analyzer
from app.core.config import Settings, get_settings
from app.schemas.common import ErrorResponse

COMPARE_RESPONSES: dict[int | str, dict[str, Any]] = {
    status.HTTP_404_NOT_FOUND: {
        "model": ErrorResponse,
        "description": (
            "No FOMC statement for a meeting on base_date or target_date; the message names the "
            "nearest statement dates."
        ),
    },
    **DATABASE_RESPONSES,
}


def get_analyzer(settings: Annotated[Settings, Depends(get_settings)]) -> Analyzer:
    """The comparison analyzer; override this dependency in tests.

    Building it needs no API key (its model and prompt version find stored analyses) and never
    raises, so request validation comes first; only calling it needs OPENAI_API_KEY.
    """
    return build_analyzer(settings)


ComparisonAnalyzer = Annotated[Analyzer, Depends(get_analyzer)]
