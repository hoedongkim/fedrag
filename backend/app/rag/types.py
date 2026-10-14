"""Types shared by indexing, retrieval and answer generation."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Protocol

from app.schemas.common import DocType


@dataclass(frozen=True)
class Chunk:
    """A retrievable passage of one FOMC document, carrying what point-in-time filters need."""

    chunk_id: str  # f"{doc_id}#{index:04d}", stable across rebuilds of the same text
    doc_id: str
    doc_type: DocType
    meeting_date: date
    # Point-in-time filters compare as_of against this: the document's release date, or for text
    # the Fed added later under a dated revision note, the note's posting date.
    release_date: date
    page_updated: date | None  # later than release_date when the Fed revised the page
    title: str
    url: str
    section: str | None  # minutes section heading; None for statements
    index: int  # position of the chunk within its document
    text: str  # the passage itself: what the reader sees and what a citation quotes
    context: str  # short header (document title, section) prepended when embedding and searching

    @property
    def search_text(self) -> str:
        """Text that embeddings and BM25 see: the context header, then the passage."""
        return f"{self.context}\n\n{self.text}"


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by a retriever; score is higher-is-better and only comparable within one
    retrieval method."""

    chunk: Chunk
    score: float


class Retriever(Protocol):
    """Returns the top_k chunks for a query. Filters apply BEFORE ranking, so as_of can never
    let a chunk released after it in, and never shrinks the result below top_k when enough
    eligible chunks exist."""

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        as_of: date | None = None,
        doc_types: Sequence[DocType] | None = None,
    ) -> list[RetrievedChunk]: ...
