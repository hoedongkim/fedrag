"""Split FOMC documents into chunks: whole statements, section-aware passages of the minutes."""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from itertools import groupby, pairwise

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.db.load import DocumentRow
from app.rag.types import Chunk

# Bump when chunk boundaries change; recorded in the index manifest. 2: a dated revision note and
# the text after it get a chunk of their own, released on the note's posting date.
CHUNKER_VERSION = 2
ENCODING = "cl100k_base"  # the text-embedding-3 tokenizer, so sizes are what the embedder sees
TARGET_TOKENS = 350  # a minutes chunk closes once its text reaches this many tokens
MAX_TOKENS = 450  # hard cap on a minutes chunk's text; longer paragraphs are split
MIN_TOKENS = 250  # a section's last chunk below this joins the previous one when it fits
OVERLAP_TOKENS = 60  # trailing sentences of a chunk that open the next one in the same section
WHOLE_DOCUMENT_TOKENS = 1000  # statements (about 375 words, at most ~730 tokens) stay whole
SKIPPED_SECTIONS = frozenset({"Attendance"})  # the roster of attendees: names, not policy

# A sentence ends at . ! or ? (and any closing quote or bracket) followed by whitespace and a
# capital, digit or opening quote. _NO_SENTENCE_END vetoes breaks after initials ("Jerome H."),
# dotted abbreviations ("U.S.", "p.m.") and titles ("Mr.").
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])[\"')\]\u2019\u201d]*\s+(?=[\"'(\[\u2018\u201c]?[A-Z0-9])")
_NO_SENTENCE_END = re.compile(
    r"(?:\b[A-Z]|\b(?:[A-Za-z]\.)+[A-Za-z]|\b(?:Mr|Ms|Mrs|Dr|St|No|Jr))\.$"
)
# Revisions the Fed dates on the page: the text from an update note on was added after release,
# and a parenthetical correction note was added with the correction.
_UPDATE_POSTED = re.compile(r"This update was posted on ([A-Z][a-z]+ \d{1,2}, \d{4})")
_CORRECTION_NOTE = re.compile(r"\s*\([^()]*\bwas corrected on [A-Z][a-z]+ \d{1,2}, \d{4}[^()]*\)")


@dataclass(frozen=True)
class _Piece:
    """A paragraph, or part of one too long for a chunk; `overlap` marks repeated text."""

    paragraph: int
    text: str
    overlap: bool = False


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(ENCODING)


def count_tokens(text: str) -> int:
    """Tokens in `text` under the embedding model's encoding (special-token text is plain text)."""
    return len(_encoding().encode(text, disallowed_special=()))


@lru_cache(maxsize=1)
def _paragraph_splitter() -> RecursiveCharacterTextSplitter:
    """Splits a paragraph over MAX_TOKENS at sentence breaks, else clauses, else words."""
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=ENCODING,
        chunk_size=TARGET_TOKENS,
        chunk_overlap=0,  # overlap is added once, between chunks, by _pack
        separators=[". ", "; ", ", ", " ", ""],
        keep_separator="end",
        disallowed_special=(),
    )


def chunk_documents(documents: Iterable[DocumentRow]) -> list[Chunk]:
    """Chunk every document, in the order given."""
    return [chunk for document in documents for chunk in chunk_document(document)]


def chunk_document(document: DocumentRow) -> list[Chunk]:
    """Split one document into chunks with deterministic ids f"{doc_id}#{index:04d}".

    A statement is one chunk unless it exceeds WHOLE_DOCUMENT_TOKENS. Minutes are chunked section
    by section: chunks never span two sections, keep whole paragraphs where they fit and overlap
    only within a section. The Attendance roster is left out; Footnotes get chunks of their own.
    A chunk's release_date is the document's, except for text the Fed added later (see _dated).
    """
    if document.doc_type != "minutes" and count_tokens(document.text) <= WHOLE_DOCUMENT_TOKENS:
        passages: list[tuple[str | None, str, date]] = [
            (None, document.text, document.release_date)
        ]
    else:
        passages = [
            (section, text, released)
            for section, paragraphs in groupby(document.paragraphs, key=lambda p: p.section)
            if section not in SKIPPED_SECTIONS
            for released, texts in _dated([p.text for p in paragraphs], document.release_date)
            for text in _pack(texts)
        ]
    return [
        Chunk(
            chunk_id=f"{document.doc_id}#{index:04d}",
            doc_id=document.doc_id,
            doc_type=document.doc_type,
            meeting_date=document.meeting_date,
            release_date=released,
            page_updated=document.page_updated,
            title=document.title,
            url=document.url,
            section=section,
            index=index,
            text=text,
            context=document.title if section is None else f"{document.title} | {section}",
        )
        for index, (section, text, released) in enumerate(passages)
    ]


def _dated(paragraphs: Sequence[str], released: date) -> list[tuple[date, list[str]]]:
    """A section's paragraphs grouped by the day their text became public.

    The stored text is the Fed's current page. A note "This update was posted on <date>" opens
    text added after release: it and the rest of the section are dated by the note, so as_of
    filters hold them back until then. Parenthetical correction notes are stripped.
    """
    texts = [_CORRECTION_NOTE.sub("", text) for text in paragraphs]
    for i, text in enumerate(texts):
        if posted := _UPDATE_POSTED.search(text):
            day = datetime.strptime(posted[1], "%B %d, %Y").date()
            return [(released, texts[:i]), (max(released, day), texts[i:])]
    return [(released, texts)]


def _pack(paragraphs: Sequence[str]) -> list[str]:
    """Pack consecutive paragraphs of one section into chunks of about TARGET..MAX tokens.

    Paragraphs stay whole unless longer than MAX_TOKENS. Each chunk after the first opens with the
    previous chunk's trailing sentences (at most OVERLAP_TOKENS), and a short final chunk joins
    the one before it when the two fit in MAX_TOKENS.
    """
    pieces = [_Piece(i, part) for i, text in enumerate(paragraphs) for part in _split(text)]
    chunks: list[list[_Piece]] = []
    current: list[_Piece] = []
    for piece in pieces:
        if _has_new(current) and _size([*current, piece]) > MAX_TOKENS:
            chunks.append(current)
            current = _overlap(current)
        if not _has_new(current) and _size([*current, piece]) > MAX_TOKENS:
            current = []  # the overlap does not fit next to this piece
        current.append(piece)
        if _size(current) >= TARGET_TOKENS:
            chunks.append(current)
            current = _overlap(current)
    if _has_new(current):
        new = [piece for piece in current if not piece.overlap]
        if chunks and _size(current) < MIN_TOKENS and _size(chunks[-1] + new) <= MAX_TOKENS:
            chunks[-1] = chunks[-1] + new
        else:
            chunks.append(current)
    return [_join(chunk) for chunk in chunks]


def _split(paragraph: str) -> list[str]:
    if count_tokens(paragraph) <= MAX_TOKENS:
        return [paragraph]
    return _paragraph_splitter().split_text(paragraph)


def _has_new(pieces: list[_Piece]) -> bool:
    return any(not piece.overlap for piece in pieces)


def _join(pieces: Sequence[_Piece]) -> str:
    """Parts of one paragraph join with a space, separate paragraphs with a blank line."""
    text = pieces[0].text
    for previous, piece in pairwise(pieces):
        text += (" " if piece.paragraph == previous.paragraph else "\n\n") + piece.text
    return text


def _size(pieces: Sequence[_Piece]) -> int:
    return count_tokens(_join(pieces)) if pieces else 0


def _overlap(chunk: Sequence[_Piece]) -> list[_Piece]:
    """The longest run of trailing sentences of `chunk` that fits in OVERLAP_TOKENS."""
    last = chunk[-1]
    for start in sentence_starts(last.text):
        if count_tokens(tail := last.text[start:]) <= OVERLAP_TOKENS:
            return [_Piece(last.paragraph, tail, overlap=True)]
    return []


def sentence_starts(text: str) -> list[int]:
    """Offsets where the sentences of `text` begin, in order; the first is always 0."""
    starts = [0]
    for match in _SENTENCE_BREAK.finditer(text):
        if not _NO_SENTENCE_END.search(text, max(0, match.start() - 16), match.start()):
            starts.append(match.end())
    return starts
