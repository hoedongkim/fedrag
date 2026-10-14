"""GET /api/meetings."""

from typing import Annotated

from fastapi import APIRouter, Query

from app.api.database import DATABASE_RESPONSES, Db
from app.db import queries
from app.schemas.meetings import MeetingsQuery, MeetingSummary

router = APIRouter(tags=["meetings"])


@router.get(
    "/meetings",
    operation_id="listMeetings",
    summary="List FOMC meetings",
    responses=DATABASE_RESPONSES,
)
def list_meetings(query: Annotated[MeetingsQuery, Query()], db: Db) -> list[MeetingSummary]:
    """List FOMC meetings in date order and the document types collected for each.

    With `as_of` the list is the point-in-time view on that date: later meetings are left out and
    so are documents released after it, such as minutes that came out weeks after the meeting.
    """
    with db.connection() as conn:
        return queries.list_meetings(conn, doc_type=query.doc_type, as_of=query.as_of)
