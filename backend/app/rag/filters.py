"""Point-in-time and document-type filters that every retriever applies before ranking, and the
metadata priors some retriever configurations apply after it.

A prior multiplies the scores of chunks that are already eligible: it reorders them and never
adds or removes one, so it cannot weaken the point-in-time guarantee.
"""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from app.rag.types import Chunk
from app.schemas.common import DocType

ChunkFactor = Callable[[Chunk], float]  # one question's prior: a score factor for each chunk


def release_ord(day: date) -> int:
    """A date as the integer YYYYMMDD, so Chroma's numeric $lte can compare release dates."""
    return day.year * 10_000 + day.month * 100 + day.day


def is_eligible(chunk: Chunk, as_of: date | None, doc_types: Sequence[DocType] | None) -> bool:
    """True if the chunk was public on as_of and has one of doc_types (None means no limit)."""
    return (as_of is None or chunk.release_date <= as_of) and (
        doc_types is None or chunk.doc_type in doc_types
    )


def chroma_where(as_of: date | None, doc_types: Sequence[DocType] | None) -> dict[str, Any] | None:
    """The same filter as a Chroma `where` clause on the release_ord and doc_type metadata."""
    clauses: list[dict[str, Any]] = []
    if as_of is not None:
        clauses.append({"release_ord": {"$lte": release_ord(as_of)}})
    if doc_types is not None:
        clauses.append({"doc_type": {"$in": list(doc_types)}})
    if len(clauses) > 1:
        return {"$and": clauses}
    return clauses[0] if clauses else None


_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4,
    "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9,
    "sept": 9, "sep": 9, "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12,
    "dec": 12,
}  # fmt: skip
_YEAR = r"(?:19|20)\d{2}"
_MONTH_NAME = "|".join(sorted(_MONTHS, key=len, reverse=True))
_ORDINAL = r"(?:st|nd|rd|th)?"
# "September 2024", "Sept. 2024", "September of 2024", "September 18, 2024", "Sept. 17-18, 2024".
_MONTH_DATE = re.compile(
    rf"\b(?P<month>{_MONTH_NAME})\.?"
    rf"(?:\s+(?P<first>\d{{1,2}}){_ORDINAL}(?:\s*[-–]\s*(?P<last>\d{{1,2}}){_ORDINAL})?)?"
    rf",?\s+(?:of\s+)?(?P<year>{_YEAR})\b",
    re.IGNORECASE,
)
_ISO_DATE = re.compile(rf"\b(?P<year>{_YEAR})-(?P<month>\d{{1,2}})-(?P<first>\d{{1,2}})\b")
_US_DATE = re.compile(rf"\b(?P<month>\d{{1,2}})/(?P<first>\d{{1,2}})/(?P<year>{_YEAR})\b")
# A bare year, but not one inside a number ("1,2015" or "2015.5").
_BARE_YEAR = re.compile(rf"(?<![\d.,])\b(?P<year>{_YEAR})\b(?![.,]\d)")
# Words that ask about the state of things on the as_of date rather than at a named time.
_CURRENT_STATE = re.compile(
    r"\b(?:now|nowadays|currently|current(?!\s+account)|still|as of|latest|most recent|"
    r"recently|today|so far|yet|at present|presently|anymore|any longer|to date)\b"
    r"|\bha(?:s|ve)\s+(?:the\s+)?(?:fed|federal reserve|fomc|committee)\s+(?:already\s+)?"
    r"(?:started|begun|began|stopped|ended|finished|resumed|paused|cut|raised|lowered)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DateMention:
    """A time a question names: a year, a month of a year, or a day (or day range) in a month."""

    year: int
    month: int | None = None
    first: date | None = None  # the days named, for a day mention
    last: date | None = None

    def matches(self, meeting: date) -> bool:
        """True if a meeting held (ending) on `meeting` is the one named.

        Two-day meetings are dated by their second day, so a day mention also matches the day
        after, and a month mention also matches a meeting that ended on the 1st of the next month
        ("the April 2024 meeting" was April 30-May 1).
        """
        if self.first is not None and self.last is not None:
            return self.first - timedelta(days=1) <= meeting <= self.last + timedelta(days=1)
        if self.month is not None:
            return any(
                (day.year, day.month) == (self.year, self.month)
                for day in (meeting, meeting - timedelta(days=1))
            )
        return meeting.year == self.year


def parse_date_mentions(text: str) -> tuple[DateMention, ...]:
    """The years, months and days a question names, in order of appearance, without repeats.

    A month needs a year ("the September meeting" names none); a year inside a month or day
    mention is not counted again on its own. "SEP" is the projections, not a month.
    """
    found: list[tuple[int, DateMention]] = []
    masked = text
    for pattern in (_MONTH_DATE, _ISO_DATE, _US_DATE):
        for match in pattern.finditer(masked):
            if (mention := _mention(match)) is not None:
                found.append((match.start(), mention))
                masked = masked[: match.start()] + " " * len(match[0]) + masked[match.end() :]
    for match in _BARE_YEAR.finditer(masked):
        found.append((match.start(), DateMention(int(match["year"]))))
    mentions: list[DateMention] = []
    for _, mention in sorted(found, key=lambda item: item[0]):
        if mention not in mentions:
            mentions.append(mention)
    return tuple(mentions)


def _mention(match: re.Match[str]) -> DateMention | None:
    if match["month"] == "SEP":  # the Summary of Economic Projections, not September
        return None
    year = int(match["year"])
    month_text = match["month"].lower()
    month = _MONTHS[month_text] if month_text in _MONTHS else int(month_text)
    if not 1 <= month <= 12:
        return None
    if match["first"] is None:
        return DateMention(year, month)
    last = match.groupdict().get("last") or match["first"]
    try:
        first_day, last_day = date(year, month, int(match["first"])), date(year, month, int(last))
    except ValueError:  # no such day ("February 30, 2024"): keep the month
        return DateMention(year, month)
    return DateMention(year, month, min(first_day, last_day), max(first_day, last_day))


def asks_current_state(question: str) -> bool:
    """True if the question asks how things stand ("still", "currently", "has the Fed started")."""
    return _CURRENT_STATE.search(question) is not None


@dataclass(frozen=True)
class MetadataPrior:
    """Score factors from chunk metadata, multiplied together; a neutral value turns one off.

    date_boost: chunks of a meeting in a month or on a day the question names ("September 2024").
    year_boost: chunks of a meeting in a year the question names without a month ("in 2022"),
        unless date_boost already applied.
    recency_boost: when as_of is set and the question asks about the current state, chunks
        released age days before as_of get 1 + recency_boost * 0.5 ** (age / half-life).
    statement_boost: statement chunks.
    """

    date_boost: float = 1.0
    year_boost: float = 1.0
    recency_boost: float = 0.0
    recency_half_life_days: float = 180.0
    statement_boost: float = 1.0

    @property
    def neutral(self) -> bool:
        """True if no question can get a factor other than 1."""
        return (
            self.date_boost == 1.0
            and self.year_boost == 1.0
            and self.recency_boost == 0.0
            and self.statement_boost == 1.0
        )

    def for_query(self, question: str, as_of: date | None) -> ChunkFactor | None:
        """The factor for each chunk under this question and as_of, or None if every factor is 1
        (so a retriever can skip re-scoring)."""
        mentions = (
            parse_date_mentions(question) if self.date_boost != 1 or self.year_boost != 1 else ()
        )
        months = [m for m in mentions if m.month is not None] if self.date_boost != 1 else []
        years = [m for m in mentions if m.month is None] if self.year_boost != 1 else []
        recent = (
            as_of if self.recency_boost > 0 and as_of and asks_current_state(question) else None
        )
        statements = self.statement_boost != 1
        if not (months or years or recent or statements):
            return None

        def factor(chunk: Chunk) -> float:
            value = 1.0
            if any(m.matches(chunk.meeting_date) for m in months):
                value *= self.date_boost
            elif any(m.matches(chunk.meeting_date) for m in years):
                value *= self.year_boost
            if recent is not None:
                age = max((recent - chunk.release_date).days, 0)
                value *= 1.0 + self.recency_boost * 0.5 ** (age / self.recency_half_life_days)
            if statements and chunk.doc_type == "statement":
                value *= self.statement_boost
            return value

        return factor
