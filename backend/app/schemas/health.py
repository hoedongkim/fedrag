"""Health check response."""

from typing import Literal

from pydantic import Field

from app.schemas.common import ApiModel


class HealthResponse(ApiModel):
    """Service status and readiness of the pieces later phases depend on."""

    status: Literal["ok"] = Field(description='Always "ok" when the API is serving requests.')
    version: str = Field(description="Backend version.")
    environment: str = Field(description="Runtime environment: development, test or production.")
    llm_configured: bool = Field(description="True when an OpenAI API key is configured.")
    database_configured: bool = Field(
        description="True when DATABASE_URL is set. Reachability is not checked."
    )
    vector_store_ready: bool = Field(
        description=(
            "True when the search index is built: its manifest exists and Chroma holds every "
            "indexed chunk (run make index)."
        )
    )
    demo_mode: bool = Field(
        description=(
            "True on the public demo (PUBLIC_DEMO=true), where every feature that would call "
            "OpenAI is off: POST /api/ask and /api/ask/stream answer 403 (GET /api/ask/examples "
            'serves recorded answers instead), and POST /api/compare reports "disabled_on_demo" '
            "for a pair without a stored analysis. Stored results are served as usual."
        )
    )
