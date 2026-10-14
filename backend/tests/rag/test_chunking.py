"""Tests for chunking: whole statements, section-aware minutes chunks, overlap, revision notes and
determinism."""

import hashlib
from datetime import date
from itertools import groupby

import pytest

from app.db.load import DocumentRow, ParagraphRow
from app.rag.bm25 import BM25Retriever
from app.rag.chunking import (
    MAX_TOKENS,
    MIN_TOKENS,
    OVERLAP_TOKENS,
    TARGET_TOKENS,
    WHOLE_DOCUMENT_TOKENS,
    chunk_document,
    chunk_documents,
    count_tokens,
    sentence_starts,
)

MINUTES_TITLE = "Minutes of the FOMC Meeting, September 17-18, 2024"
STATEMENT_TITLE = "FOMC Statement, September 18, 2024"


def make_document(
    doc_type: str, paragraphs: list[tuple[str | None, str]], *, title: str | None = None
) -> DocumentRow:
    text = "\n\n".join(t for _, t in paragraphs)
    return DocumentRow(
        doc_id=f"{doc_type}-2024-09-18",
        doc_type=doc_type,  # type: ignore[arg-type]
        meeting_date=date(2024, 9, 18),
        release_date=date(2024, 10, 9) if doc_type == "minutes" else date(2024, 9, 18),
        release_date_source="test",
        page_updated=None,
        title=title or (MINUTES_TITLE if doc_type == "minutes" else STATEMENT_TITLE),
        url="https://www.federalreserve.gov/monetarypolicy/test.htm",
        paragraphs=[ParagraphRow(section=s, text=t) for s, t in paragraphs],
        text=text,
        word_count=len(text.split()),
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        parser_version=1,
    )


def paragraph(topic: str, n: int, sentences: int = 6) -> str:
    """A paragraph of `sentences` distinct sentences (about 17 tokens each) about `topic`."""
    return " ".join(
        f"Participants noted that {topic} conditions in district {n} item {i} remained "
        f"broadly stable over the intermeeting period."
        for i in range(sentences)
    )


def minutes(sections: dict[str, int], sentences: int = 6) -> DocumentRow:
    rows: list[tuple[str | None, str]] = [
        (name, paragraph(name.split()[0].lower(), n, sentences))
        for name, count in sections.items()
        for n in range(count)
    ]
    return make_document("minutes", rows)


STATEMENT = make_document(
    "statement",
    [
        (None, "Recent indicators suggest that economic activity has continued to expand."),
        (None, "The Committee decided to lower the target range to 4-3/4 to 5 percent."),
        (None, "Voting for the monetary policy action were Jerome H. Powell, Chair; and others."),
    ],
)


def test_statement_is_one_chunk() -> None:
    [chunk] = chunk_document(STATEMENT)

    assert chunk.chunk_id == "statement-2024-09-18#0000"
    assert (chunk.text, chunk.section, chunk.index) == (STATEMENT.text, None, 0)
    assert chunk.context == STATEMENT_TITLE
    assert chunk.search_text == f"{STATEMENT_TITLE}\n\n{STATEMENT.text}"
    assert (chunk.release_date, chunk.doc_type, chunk.url) == (
        STATEMENT.release_date,
        "statement",
        STATEMENT.url,
    )


def test_a_very_long_statement_is_split_by_paragraph() -> None:
    long_statement = make_document(
        "statement", [(None, paragraph(f"topic{n}", n, 8)) for n in range(12)]
    )
    assert count_tokens(long_statement.text) > WHOLE_DOCUMENT_TOKENS

    chunks = chunk_document(long_statement)

    assert len(chunks) > 1
    assert all(count_tokens(c.text) <= MAX_TOKENS and c.section is None for c in chunks)
    assert all(c.context == STATEMENT_TITLE for c in chunks)


def test_attendance_is_skipped_and_footnotes_kept() -> None:
    document = make_document(
        "minutes",
        [
            ("Preamble", "A joint meeting of the Committee was held on Tuesday."),
            ("Staff Economic Outlook", paragraph("outlook", 1)),
            ("Attendance", "Jerome H. Powell, Chair; John C. Williams, Vice Chair"),
            ("Attendance", "Thomas Laubach, Secretary"),
            ("Footnotes", "Attended Tuesday's session only."),
        ],
    )

    chunks = chunk_document(document)

    assert [c.section for c in chunks] == ["Preamble", "Staff Economic Outlook", "Footnotes"]
    assert not any("Laubach" in c.text or "Williams" in c.text for c in chunks)
    assert chunks[-1].context == f"{MINUTES_TITLE} | Footnotes"
    assert chunks[1].context == f"{MINUTES_TITLE} | Staff Economic Outlook"


def test_a_dated_update_note_holds_its_text_back_until_posted() -> None:
    document = make_document(
        "minutes",
        [
            ("Committee Policy Actions", "The Committee decided to lower the target range."),
            ("Footnotes", "Attended Wednesday session only."),
            (
                "Footnotes",
                "Note: The following information was inadvertently omitted in the original "
                "publication of these minutes. This update was posted on October 9, 2026.",
            ),
            (
                "Footnotes",
                "It was agreed that the next meeting of the Committee would be held on "
                "Tuesday-Wednesday, November 6-7, 2024.",
            ),
        ],
    )

    chunks = chunk_document(document)

    [note] = [c for c in chunks if "This update was posted" in c.text]
    assert note.release_date == date(2026, 10, 9)
    assert note.section == "Footnotes" and "next meeting of the Committee" in note.text
    assert all(c.release_date == document.release_date for c in chunks if c != note)
    assert not any("2026" in c.text for c in chunks if c != note)
    # The as_of filter now keeps the added text out until its posting date.
    hits = BM25Retriever(chunks).retrieve("next meeting", top_k=10, as_of=date(2024, 10, 15))
    assert hits and not any("next meeting" in h.chunk.text for h in hits)


def test_a_correction_note_is_stripped() -> None:
    corrected = (
        "Attended through the discussion of developments in financial markets and open market "
        "operations."
    )
    document = make_document(
        "minutes",
        [
            (
                "Footnotes",
                f"{corrected} (This footnote was corrected on November 26, 2021, to replace an "
                'erroneous reference to a "discussion of asset purchases.")',
            ),
            ("Footnotes", "Attended Tuesday's session only."),
        ],
    )

    [chunk] = chunk_document(document)

    assert chunk.text == f"{corrected}\n\nAttended Tuesday's session only."
    assert chunk.release_date == document.release_date


def test_minutes_chunks_never_cross_sections_and_cover_every_paragraph() -> None:
    document = minutes({"Developments": 5, "Staff": 9, "Participants": 14, "Committee": 3})
    chunks = chunk_document(document)
    by_section = {
        section: [p.text for p in paragraphs]
        for section, paragraphs in groupby(document.paragraphs, key=lambda p: p.section)
    }

    for chunk in chunks:
        own = by_section[chunk.section]
        # Every paragraph-sized part of a chunk is a passage of a paragraph of its own section.
        assert all(any(part in p for p in own) for part in chunk.text.split("\n\n")), chunk.chunk_id
    for section, texts in by_section.items():
        joined = "\n\n".join(c.text for c in chunks if c.section == section)
        for text in texts:
            assert all(text[s:].split(". ")[0] in joined for s in sentence_starts(text))


def test_minutes_chunk_sizes() -> None:
    document = minutes({"Participants": 20, "Committee": 1})
    chunks = chunk_document(document)
    sizes = {
        section: [count_tokens(c.text) for c in group]
        for section, group in groupby(chunks, key=lambda c: c.section)
    }

    assert all(size <= MAX_TOKENS for group in sizes.values() for size in group)
    # Inside a long section every chunk but the last reaches the target size.
    assert all(size >= TARGET_TOKENS for size in sizes["Participants"][:-1])
    # A short section is one small chunk: chunks never borrow text from the next section.
    assert len(sizes["Committee"]) == 1 and sizes["Committee"][0] < MIN_TOKENS


def test_an_oversized_paragraph_is_split_within_the_cap() -> None:
    giant = paragraph("liquidity", 1, sentences=60)
    assert count_tokens(giant) > 2 * MAX_TOKENS
    document = make_document("minutes", [("Developments", giant), ("Developments", "Short.")])

    chunks = chunk_document(document)

    assert len(chunks) >= 3
    assert all(count_tokens(c.text) <= MAX_TOKENS for c in chunks)
    assert all(c.text.replace("\n\n", " ") in giant + " Short." for c in chunks)


def test_overlap_repeats_trailing_sentences_only_within_a_section() -> None:
    document = minutes({"Staff": 8, "Participants": 8})
    chunks = chunk_document(document)

    for previous, chunk in zip(chunks, chunks[1:], strict=False):
        shared = next(
            (
                previous.text[s:]
                for s in sentence_starts(previous.text)
                if chunk.text.startswith(previous.text[s:])
            ),
            "",
        )
        if chunk.section == previous.section:
            assert shared and count_tokens(shared) <= OVERLAP_TOKENS
        else:
            assert not shared


def test_chunking_is_deterministic() -> None:
    document = minutes({"Staff": 6, "Participants": 7})
    copy = DocumentRow.model_validate(document.model_dump())

    first = chunk_documents([STATEMENT, document])
    second = chunk_documents([copy, STATEMENT])

    assert sorted(first, key=lambda c: c.chunk_id) == sorted(second, key=lambda c: c.chunk_id)
    ids = [c.chunk_id for c in first if c.doc_id == document.doc_id]
    assert ids == [f"{document.doc_id}#{i:04d}" for i in range(len(ids))]


@pytest.mark.parametrize(
    ("text", "starts"),
    [
        ("The U.S. economy grew. Inflation eased.", ["The U.S.", "Inflation"]),
        ("Jerome H. Powell spoke. Others agreed.", ["Jerome H.", "Others"]),
        ('He said "rates rose." The Committee agreed.', ["He said", "The Committee"]),
        ("At 9:00 a.m. Tuesday. Rates were 5-1/4 percent. 2024 ended.", ["At 9", "Rates", "2024"]),
        ("Mr. Smith left. Growth was 2.5 percent.", ["Mr. Smith", "Growth"]),
    ],
)
def test_sentence_starts(text: str, starts: list[str]) -> None:
    found = [text[s:] for s in sentence_starts(text)]

    assert [f[: len(s)] for f, s in zip(found, starts, strict=True)] == starts
