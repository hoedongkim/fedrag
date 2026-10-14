"""Tests for citation-marker renumbering, whole and streamed chunk by chunk."""

import random
import re
from collections.abc import Sequence
from itertools import pairwise

import pytest

from app.rag.markers import Renumbered, StreamRenumberer, renumber_markers


def feed_all(chunks: Sequence[str], passage_count: int) -> list[Renumbered]:
    """The pieces a StreamRenumberer returns for `chunks`, the flush last."""
    renumberer = StreamRenumberer(passage_count)
    return [renumberer.feed(chunk) for chunk in chunks] + [renumberer.flush()]


def streamed(chunks: Sequence[str], passage_count: int = 5) -> str:
    return "".join(piece.text for piece in feed_all(chunks, passage_count))


@pytest.mark.parametrize(
    ("answer", "expected", "cited"),
    [
        ("Cut [3][1]. Risks [2]. Jobs [3].", "Cut [1][2]. Risks [3]. Jobs [1].", [3, 1, 2]),
        ("Rates fell [1, 3].", "Rates fell [1][2].", [1, 3]),
        ("Rates fell [2-4].", "Rates fell [1][2][3].", [2, 3, 4]),
        ("Rates fell [1] [3]; risks [3].", "Rates fell [1][2]; risks [2].", [1, 3]),
        ("Rates fell [9]. Risks [2][7]. Jobs [0].", "Rates fell. Risks [1]. Jobs.", [2]),
        ("In [2024] the range [1] held.", "In [2024] the range [1] held.", [1]),
        ("  The passages do not say.\n", "The passages do not say.", []),
    ],
    ids=["first-appearance", "comma-list", "range", "adjacent", "invalid", "year", "trimmed"],
)
def test_renumber_markers(answer: str, expected: str, cited: list[int]) -> None:
    assert renumber_markers(answer, 5) == (expected, cited)


@pytest.mark.parametrize(
    ("chunks", "expected"),
    [
        (["Rates fell [", "1", "]."], "Rates fell [1]."),
        (["Rates fell [1", ", ", "3]."], "Rates fell [1][2]."),
        (["Rates fell [2", "-", "4]."], "Rates fell [1][2][3]."),
        (["Rates fell [1]", " [3]", "."], "Rates fell [1][2]."),
        (["Rates fell", " [", "9", "]", "."], "Rates fell."),
        (["Rates fell [1", "2]."], "Rates fell."),  # 12 names no passage of 5
        (["In [20", "24] rates [", "1] held."], "In [2024] rates [1] held."),
        (["\n\n", "Rates", " fell. ", "\n"], "Rates fell."),
        (["Rates fell.\n\n  [", "9]\n"], "Rates fell."),
        (["Rates fell [1"], "Rates fell [1"),  # the answer ended before the marker did
    ],
    ids=[
        "split-marker",
        "split-comma-list",
        "split-range",
        "run-across-chunks",
        "invalid-dropped-with-its-space",
        "two-digit-number",
        "year-untouched",
        "trimmed",
        "dropped-marker-leaves-no-trailing-space",
        "unfinished-marker-is-text",
    ],
)
def test_stream_renumbers_markers_split_across_chunks(chunks: list[str], expected: str) -> None:
    assert streamed(chunks) == expected
    assert renumber_markers("".join(chunks), 5)[0] == expected


def test_stream_holds_back_only_what_a_later_chunk_can_change() -> None:
    renumberer = StreamRenumberer(5)

    assert renumberer.feed("Rates fell [2").text == "Rates fell"  # " [2" may become " [21]"
    assert renumberer.feed("]").text == ""  # "[2]" may still join "[2][3]"
    assert renumberer.feed(" and").text == " [1] and"
    assert renumberer.feed(" [20").text == ""  # " [20]" would be a marker
    assert renumberer.feed("24] rates. ").text == " [2024] rates."  # a year; trailing space held
    assert renumberer.feed("Done").text == " Done"
    assert renumberer.flush().text == ""


def test_stream_reports_each_citation_with_the_text_that_first_cites_it() -> None:
    pieces = feed_all(["Cut [3", "]. Risks [1]", " [3]. Jobs", " [2]."], 5)

    assert [(piece.text, piece.cited) for piece in pieces] == [
        ("Cut", {}),
        (" [1]. Risks", {1: 3}),
        (" [1][2]. Jobs", {2: 1}),
        (" [3].", {3: 2}),
        ("", {}),
    ]


MARKER = re.compile(r"\[\d{1,2}\]")  # a renumbered marker in the output
INPUT_MARKER = re.compile(r"\[\s*\d{1,2}(?:\s*[-–,;]\s*\d{1,2})*\s*\]")

# Fragments that combine into hard cases: markers of every shape, numbers that are not markers,
# brackets, and whitespace of every kind next to them.
FRAGMENTS = [
    *["Rates", "fell", ".", ",", "(", "x", "-", "–", ";", "0", "1", "2", "3", "9", "12"],
    *[" ", " ", "  ", "\t", "\n", "\n\n", "[", "]", "[[", "]]"],
    *["[1]", "[2]", "[3]", "[9]", "[0]", "[12]", "[1, 3]", "[1,3]", "[2-4]", "[3–1]", "[1;2]"],
    *["[ 1 ]", "[1\n]", "[1][3]", "[1] [2]", "[2024]", "[20", "24]"],
]


@pytest.mark.parametrize("seed", range(4))
def test_stream_equals_the_whole_answer_for_any_chunking(seed: int) -> None:
    rng = random.Random(seed)
    for _ in range(600):
        answer = "".join(rng.choice(FRAGMENTS) for _ in range(rng.randint(0, 24)))
        passage_count = rng.randint(0, 5)
        expected, cited = renumber_markers(answer, passage_count)
        cuts = rng.sample(range(1, len(answer)), rng.randint(0, max(len(answer) - 1, 0)))
        bounds = [0, *sorted(cuts), len(answer)]
        chunkings = [
            [answer[start:end] for start, end in pairwise(bounds)],
            list(answer),  # one character at a time
            [answer],
        ]
        for chunks in chunkings:
            pieces = feed_all(chunks, passage_count)
            numbers = {n: ref for piece in pieces for n, ref in piece.cited.items()}
            assert "".join(piece.text for piece in pieces) == expected, (answer, chunks)
            # Every output marker is in one piece: none is split across deltas. (Only when the
            # answer has no other brackets: "[20" + dropped "[9]" + "]" also reads "[20]".)
            if not re.search(r"[\[\]]", INPUT_MARKER.sub("", answer)):
                whole = sum(len(MARKER.findall(piece.text)) for piece in pieces)
                assert whole == len(MARKER.findall(expected)), (answer, chunks)
            assert list(numbers) == list(range(1, len(cited) + 1)), (answer, chunks)
            assert list(numbers.values()) == cited, (answer, chunks)
