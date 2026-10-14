"""FOMC meetings and the documents collected for each."""

from datetime import date

from pydantic import Field

from app.schemas.common import ApiModel, DocType, RequestModel


class MeetingsQuery(RequestModel):
    """Query parameters for the meeting list."""

    doc_type: DocType | None = Field(
        default=None,
        description="Only meetings that have this document type (released by as_of, if given).",
    )
    as_of: date | None = Field(
        default=None,
        description=(
            "Point-in-time view: only meetings whose meeting_date is on or before this date (an "
            "unscheduled meeting only once one of its documents is released), listing only "
            "documents released on or before it. Null means no cutoff."
        ),
    )


class MeetingSummary(ApiModel):
    """An FOMC meeting and the document types collected for it."""

    meeting_date: date = Field(
        description=(
            "Last day of the FOMC meeting, normally the day its statement was released (two "
            "unscheduled meetings, 2019-10-04 and 2020-03-02, released theirs later)."
        )
    )
    start_date: date = Field(
        description="First day of the meeting; equal to meeting_date for a one-day meeting."
    )
    unscheduled: bool = Field(
        description=(
            "True for an unscheduled meeting, such as the emergency meetings of March 2020, or a "
            "notation vote that issued a statement (2020-03-23)."
        )
    )
    has_sep: bool = Field(
        description="True when a Summary of Economic Projections was released with the meeting."
    )
    has_press_conference: bool = Field(
        description="True when the Chair held a press conference after the meeting."
    )
    doc_types: list[DocType] = Field(
        description=(
            "Document types collected for this meeting; with as_of, only those released by then."
        )
    )
