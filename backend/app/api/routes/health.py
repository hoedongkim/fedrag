"""GET /api/health."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app import __version__
from app.core.config import Settings, get_settings
from app.rag.index import index_ready
from app.schemas.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", operation_id="getHealth", summary="Service health")
def get_health(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    """Report the backend version, whether the LLM, database and vector store are set up, and
    whether this is the public demo, where every feature that would call OpenAI is off."""
    return HealthResponse(
        status="ok",
        version=__version__,
        environment=settings.environment,
        llm_configured=settings.llm_configured,
        database_configured=settings.database_configured,
        vector_store_ready=index_ready(settings),
        demo_mode=settings.public_demo,
    )
