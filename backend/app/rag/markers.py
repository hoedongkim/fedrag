"""Inline citation markers: the LLM cites passage n of its prompt as [n], and these renumber the
markers to the order in which the answer first cites each passage, for a whole answer or for one
that arrives in chunks."""

import re
from typing import NamedTuple

# One marker: "[2]", "[1, 3]" or "[2-4]". Numbers have one or two digits, so "[2024]" is not one.
_REFS = r"\[\s*\d{1,2}(?:\s*[-–,;]\s*\d{1,2})*\s*\]"
# A run of adjacent markers such as "[1][3]" or "[1] [3]".
_RUN = rf"{_REFS}(?:[ \t]?{_REFS})*"
# A run with the spaces before it, which go with it when none of its markers names a passage.
_MARKERS = re.compile(rf"(?P<lead>[ \t]*)(?P<run>{_RUN})")
_REF = re.compile(r"(\d{1,2})(?:\s*[-–]\s*(\d{1,2}))?")
# A marker before it is complete: "[", "[1", "[1,", "[ 2 -".
_PARTIAL = r"\[\s*(?:\d{1,2}(?:\s*[-–,;]\s*\d{1,2})*\s*(?:[-–,;]\s*)?)?"
# The end of streamed text that later text can still turn into a different match: the lead of a
# run, a run the next marker would join, and the start of a marker. Every part is optional, so
# it always matches, at the latest at the very end.
_OPEN_END = re.compile(rf"[ \t]*(?:{_RUN}[ \t]?)?(?:{_PARTIAL})?\Z")


class CitationOrder:
    """Numbers the passages an answer cites 1, 2, ... in the order it first cites them."""

    def __init__(self, passage_count: int) -> None:
        self.passage_count = passage_count
        self.numbers: dict[int, int] = {}  # passage number -> citation number, in citation order

    def renumber(self, text: str) -> str:
        """`text` with every marker run renumbered: markers naming no passage dropped, the rest
        deduplicated and sorted, and a run left empty dropped with the spaces before it."""
        return _MARKERS.sub(self._renumber, text)

    def _renumber(self, match: re.Match[str]) -> str:
        numbers = set()
        for ref in _expand(match["run"]):
            if 1 <= ref <= self.passage_count:
                numbers.add(self.numbers.setdefault(ref, len(self.numbers) + 1))
        return match["lead"] + "".join(f"[{n}]" for n in sorted(numbers)) if numbers else ""


def renumber_markers(answer: str, passage_count: int) -> tuple[str, list[int]]:
    """Renumber the markers of a whole answer by first citation.

    Returns the answer, trimmed, in which [k] stands for the k-th passage it cites, and the
    numbers of the cited passages in that order (the first is the passage [1] now refers to).
    """
    order = CitationOrder(passage_count)
    return order.renumber(answer).strip(), list(order.numbers)


class Renumbered(NamedTuple):
    """A piece of a streamed answer that later chunks can no longer change."""

    text: str  # answer text, its markers renumbered
    cited: dict[int, int]  # citation number -> passage number, for passages first cited in text


class StreamRenumberer:
    """renumber_markers for an answer that arrives in chunks.

    feed() returns at once the text that no later chunk can change. It holds back the open end of
    the input (the spaces before a marker, a marker run that the next marker would join, and an
    incomplete marker such as "[", "[1" or "[1,") and trailing whitespace of the output, which is
    trimmed if the answer ends there. flush() returns the rest once the answer is complete.
    Joined, the pieces are exactly renumber_markers(whole answer), with the same numbering.
    """

    def __init__(self, passage_count: int) -> None:
        self._order = CitationOrder(passage_count)
        self._pending = ""  # input not renumbered yet
        self._space = ""  # renumbered trailing whitespace not returned yet
        self._started = False  # whether any text was returned: until then leading space is cut

    def feed(self, chunk: str) -> Renumbered:
        self._pending += chunk
        open_end = _OPEN_END.search(self._pending)
        cut = open_end.start() if open_end else len(self._pending)
        ready, self._pending = self._pending[:cut], self._pending[cut:]
        return self._renumber(ready)

    def flush(self) -> Renumbered:
        ready, self._pending = self._pending, ""
        piece = self._renumber(ready)
        self._space = ""  # the end of the answer: trimmed
        return piece

    def _renumber(self, text: str) -> Renumbered:
        known = len(self._order.numbers)
        text = self._space + self._order.renumber(text)
        if not self._started:
            text = text.lstrip()
        body = text.rstrip()
        self._space = text[len(body) :]
        self._started = self._started or bool(body)
        cited = {n: ref for ref, n in self._order.numbers.items() if n > known}
        return Renumbered(body, cited)


def _expand(run: str) -> list[int]:
    """Passage numbers in a marker run, ranges such as 2-4 expanded."""
    refs: list[int] = []
    for start, end in _REF.findall(run):
        first, last = int(start), int(end or start)
        refs.extend(range(first, last + 1) if first <= last else (first, last))
    return refs
