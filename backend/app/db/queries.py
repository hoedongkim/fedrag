"""Read queries over the fedrag schema that back the API."""

from datetime import date
from typing import Any, get_args

import psycopg
from psycopg.rows import dict_row

from app.schemas.common import DocType
from app.schemas.meetings import MeetingSummary

DOC_TYPE_ORDER: tuple[str, ...] = get_args(DocType)

# The as_of cutoff sits in the JOIN, not the WHERE, so a scheduled meeting whose documents were all
# released after as_of still appears (with no doc types): it was on the public calendar. An
# unscheduled meeting was not public until its first document, so with as_of it appears only once
# one is released (m.unscheduled can be used in HAVING because the query groups by the key).
LIST_MEETINGS = """
SELECT m.meeting_date, m.start_date, m.unscheduled, m.has_sep, m.has_press_conference,
       array_remove(array_agg(d.doc_type), NULL) AS doc_types
FROM fedrag.meetings AS m
LEFT JOIN fedrag.documents AS d
       ON d.meeting_date = m.meeting_date
      AND (%(as_of)s::date IS NULL OR d.release_date <= %(as_of)s::date)
WHERE %(as_of)s::date IS NULL OR m.meeting_date <= %(as_of)s::date
GROUP BY m.meeting_date
HAVING (%(as_of)s::date IS NULL OR NOT m.unscheduled OR count(d.doc_type) > 0)
   AND (%(doc_type)s::text IS NULL OR coalesce(bool_or(d.doc_type = %(doc_type)s::text), false))
ORDER BY m.meeting_date
"""


def list_meetings(
    conn: psycopg.Connection[Any], doc_type: DocType | None = None, as_of: date | None = None
) -> list[MeetingSummary]:
    """Meetings in date order with their document types, as they stood on `as_of` if given.

    as_of hides meetings after it and documents released after it, so minutes published weeks
    after a meeting do not leak into a point-in-time view. It also hides an unscheduled meeting
    until one of its documents is released. doc_type keeps only the meetings that have that
    type, under the same cutoff.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        rows = cur.execute(LIST_MEETINGS, {"doc_type": doc_type, "as_of": as_of}).fetchall()
    return [
        MeetingSummary.model_validate(
            row | {"doc_types": sorted(row["doc_types"], key=DOC_TYPE_ORDER.index)}
        )
        for row in rows
    ]
