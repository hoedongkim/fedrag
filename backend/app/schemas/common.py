"""Types shared across the API contract."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DocType = Literal["statement", "minutes", "sep", "press_conference"]
RetrievalMethod = Literal["bm25", "dense", "hybrid", "hybrid_rerank"]
Stance = Literal["hawkish", "neutral", "dovish"]
StanceClassifier = Literal["keyword", "finbert", "llm_zero_shot", "llm_few_shot"]


class ApiModel(BaseModel):
    """Base model: response fields with defaults are always serialized, so mark them required."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)


class RequestModel(ApiModel):
    """Base for request bodies and query models: unknown fields are a 422, not silently dropped."""

    model_config = ConfigDict(extra="forbid")


class ErrorResponse(ApiModel):
    """Error body returned by HTTPException."""

    detail: str = Field(description="Human-readable error message.")


class SourceDocument(ApiModel):
    """An FOMC document that an answer or comparison is grounded in."""

    doc_id: str = Field(description="Stable identifier of the document.")
    doc_type: DocType = Field(description="Kind of FOMC document.")
    meeting_date: date = Field(description="Date of the FOMC meeting the document belongs to.")
    release_date: date = Field(
        description=(
            "Date the document became public. Point-in-time retrieval filters on this, not "
            "meeting_date, because minutes are released about three weeks after the meeting."
        )
    )
    page_updated: date | None = Field(
        description=(
            'The Fed\'s "Last Update" stamp on the page. Later than release_date means the stored '
            "text was revised after release; null when the page has no stamp."
        )
    )
    title: str = Field(description="Human-readable document title.")
    url: str = Field(description="Canonical source URL of the document.")
