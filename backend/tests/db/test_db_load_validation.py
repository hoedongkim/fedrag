"""Tests that load_processed rejects bad processed files before touching the database."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from app.db.load import DocumentRow, LoadError, MeetingRow, load_processed, read_jsonl

Row = dict[str, Any]
MakeDocument = Callable[..., Row]
WriteProcessed = Callable[[list[Row], list[Row]], Path]

# A connection that fails the test if load_processed ever uses it.
NO_CONNECTION: Any = object()

Mutation = Callable[[Row], Any]

BAD_MEETINGS: dict[str, tuple[Mutation, str]] = {
    "missing-key": (lambda r: r.pop("has_sep"), "has_sep: Field required"),
    "unknown-key": (lambda r: r.update(chair="Powell"), "chair: Extra inputs are not permitted"),
    "string-bool": (lambda r: r.update(unscheduled="false"), "unscheduled: Input should be"),
    "bad-date": (lambda r: r.update(meeting_date="2024-9-18"), "meeting_date: Input should be"),
    "start-after-end": (lambda r: r.update(start_date="2024-09-19"), "start_date is after"),
    "minutes-before-meeting": (
        lambda r: r.update(minutes_release_date="2024-09-17"),
        "minutes_release_date is before meeting_date",
    ),
}

BAD_DOCUMENTS: dict[str, tuple[Mutation, str]] = {
    "doc-id-mismatch": (lambda r: r.update(doc_id="statement-2024-9-18"), "doc_id should be"),
    "released-before-meeting": (
        lambda r: r.update(release_date="2024-09-17"),
        "release_date is before meeting_date",
    ),
    "unknown-doc-type": (lambda r: r.update(doc_type="memo"), "doc_type: Input should be"),
    "sha-mismatch": (lambda r: r.update(sha256="0" * 64), "sha256 does not match text"),
    "word-count-mismatch": (
        lambda r: r.update(word_count=r["word_count"] + 1),
        "word_count does not match text",
    ),
    "text-mismatch": (lambda r: r.update(text=r["text"] + " extra"), "text is not the paragraph"),
    "no-paragraphs": (lambda r: r.update(paragraphs=[]), "paragraphs: List should have at least"),
    "blank-paragraph": (lambda r: r["paragraphs"][0].update(text=""), "paragraphs.0.text"),
    "http-url": (lambda r: r.update(url="http://example.com"), "url: String should match"),
    "zero-words": (lambda r: r.update(word_count=0), "word_count: Input should be greater"),
    "missing-release-source": (lambda r: r.pop("release_date_source"), "Field required"),
    "null-release-date": (lambda r: r.update(release_date=None), "release_date: Input should be"),
}


def test_sample_rows_are_valid(
    write_processed: WriteProcessed, sample_meetings: list[Row], sample_documents: list[Row]
) -> None:
    processed = write_processed(sample_meetings, sample_documents)

    assert len(read_jsonl(processed / "meetings.jsonl", MeetingRow)) == 4
    assert len(read_jsonl(processed / "documents.jsonl", DocumentRow)) == 5


@pytest.mark.parametrize(("mutate", "message"), BAD_MEETINGS.values(), ids=BAD_MEETINGS.keys())
def test_bad_meeting_row_is_rejected(
    write_processed: WriteProcessed,
    sample_meetings: list[Row],
    sample_documents: list[Row],
    mutate: Mutation,
    message: str,
) -> None:
    mutate(sample_meetings[0])
    processed = write_processed(sample_meetings, sample_documents)

    with pytest.raises(LoadError, match="meetings.jsonl line 1: ") as error:
        load_processed(NO_CONNECTION, processed)

    assert message in str(error.value)


@pytest.mark.parametrize(("mutate", "message"), BAD_DOCUMENTS.values(), ids=BAD_DOCUMENTS.keys())
def test_bad_document_row_is_rejected(
    write_processed: WriteProcessed,
    sample_meetings: list[Row],
    sample_documents: list[Row],
    mutate: Mutation,
    message: str,
) -> None:
    mutate(sample_documents[1])  # statement-2024-09-18
    processed = write_processed(sample_meetings, sample_documents)

    with pytest.raises(LoadError, match="documents.jsonl line 2: ") as error:
        load_processed(NO_CONNECTION, processed)

    assert message in str(error.value)


def test_every_bad_line_is_reported(
    write_processed: WriteProcessed, sample_meetings: list[Row], sample_documents: list[Row]
) -> None:
    for row in sample_documents:
        row["word_count"] = 0
    processed = write_processed(sample_meetings, sample_documents)

    with pytest.raises(LoadError, match=r"\(5 problems\)") as error:
        load_processed(NO_CONNECTION, processed)

    assert all(f"line {n}:" in str(error.value) for n in range(1, 6))


def test_duplicate_keys_are_rejected(
    write_processed: WriteProcessed, sample_meetings: list[Row], sample_documents: list[Row]
) -> None:
    processed = write_processed(
        sample_meetings + sample_meetings[:1], sample_documents + sample_documents[:1]
    )

    with pytest.raises(LoadError) as error:
        load_processed(NO_CONNECTION, processed)

    assert "meeting_date 2024-09-18 appears 2 times" in str(error.value)
    assert "doc_id statement-2020-03-15 appears 2 times" in str(error.value)


def test_document_without_meeting_is_rejected(
    write_processed: WriteProcessed,
    sample_meetings: list[Row],
    sample_documents: list[Row],
    make_document: MakeDocument,
) -> None:
    orphan = make_document("statement", "2019-07-31", "2019-07-31", [(None, "Rates were cut.")])
    processed = write_processed(sample_meetings, [*sample_documents, orphan])

    with pytest.raises(LoadError, match="statement-2019-07-31 has no meeting 2019-07-31"):
        load_processed(NO_CONNECTION, processed)


def test_empty_meetings_file_is_rejected(write_processed: WriteProcessed) -> None:
    with pytest.raises(LoadError, match="meetings.jsonl has no rows"):
        load_processed(NO_CONNECTION, write_processed([], []))


@pytest.mark.parametrize("missing", ["meetings.jsonl", "documents.jsonl"])
def test_missing_file_is_rejected(
    write_processed: WriteProcessed,
    sample_meetings: list[Row],
    sample_documents: list[Row],
    missing: str,
) -> None:
    processed = write_processed(sample_meetings, sample_documents)
    (processed / missing).unlink()

    with pytest.raises(LoadError, match=f"{missing} not found"):
        load_processed(NO_CONNECTION, processed)


def test_blank_lines_are_ignored(
    write_processed: WriteProcessed, sample_meetings: list[Row], sample_documents: list[Row]
) -> None:
    processed = write_processed(sample_meetings, sample_documents)
    path: Path = processed / "meetings.jsonl"
    path.write_text("\n" + path.read_text() + "\n\n")

    assert len(read_jsonl(path, MeetingRow)) == 4
