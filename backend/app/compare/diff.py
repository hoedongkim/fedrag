"""Deterministic sentence alignment of two FOMC statements, with word diffs of reworded pairs.

Each statement is split into sentences paragraph by paragraph (app.stance.statements
.split_sentences); procedural sentences such as roll calls are kept and flagged, not dropped. The
two sentence sequences are aligned in order (Needleman-Wunsch): pairing base sentence i with
target sentence j scores their similarity s, the difflib ratio of their lower-cased word tokens,
and leaving a sentence unpaired (added or removed) costs THRESHOLD / 2. The best alignment thus
maximises the sum of (s - THRESHOLD) over its pairs, and no pair below THRESHOLD is ever aligned.
Ties go to pairing, then to listing removed sentences before added ones, so the result depends on
the texts alone. A pair identical up to whitespace is "unchanged", any other pair "modified".

THRESHOLD = 0.5 was fixed before running the diff on the real statements and is never tuned on
labels: at 0.5 at least half of the two sentences' words line up in order. A moved sentence shows
up as removed plus added (the alignment keeps document order), and a sentence split in two as one
modified pair plus one added sentence.
"""

import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher

from app.compare.types import Change, Segment, StatementText
from app.schemas.compare import ComparisonStats
from app.stance.statements import is_procedural, split_sentences

THRESHOLD = 0.5
SIMILARITY_DIGITS = 4  # reported similarities are rounded; alignment uses the exact ratios
_SCALE = 10**9  # pair scores become integers, so equal-scoring alignments tie exactly
_WORD = re.compile(r"\w+")
_PIECE = re.compile(r"\s+|\S+")  # words (with their punctuation) and the whitespace between them


@dataclass(frozen=True)
class Diff:
    """The alignment of two statements, in document order, and its summary."""

    changes: tuple[Change, ...]
    stats: ComparisonStats


def split_statement(paragraphs: Sequence[str]) -> list[str]:
    """Every sentence of the statement in order, procedural ones included, each verbatim."""
    return [sentence for paragraph in paragraphs for sentence in split_sentences(paragraph)]


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def word_tokens(text: str) -> list[str]:
    """Lower-cased word tokens (NFKC, punctuation dropped): what similarity compares."""
    return _WORD.findall(unicodedata.normalize("NFKC", text).casefold())


def similarity(base: Sequence[str], target: Sequence[str]) -> float:
    """difflib ratio of two token sequences: 2 * matched tokens / total tokens, in [0, 1]."""
    return SequenceMatcher(None, base, target, autojunk=False).ratio()


def _similarities(base: list[list[str]], target: list[list[str]]) -> list[list[float | None]]:
    """s[i][j] for every pair at or above THRESHOLD, else None (cheap upper bounds first)."""
    scores: list[list[float | None]] = [[None] * len(target) for _ in base]
    matcher = SequenceMatcher(None, autojunk=False)
    for j, b in enumerate(target):
        matcher.set_seq2(b)
        for i, a in enumerate(base):
            matcher.set_seq1(a)
            if (
                matcher.real_quick_ratio() >= THRESHOLD
                and matcher.quick_ratio() >= THRESHOLD
                and (ratio := matcher.ratio()) >= THRESHOLD
            ):
                scores[i][j] = ratio
    return scores


def align(base: Sequence[str], target: Sequence[str]) -> list[Change]:
    """The order-preserving alignment of two sentence sequences (see the module docstring)."""
    n, m = len(base), len(target)
    scores = _similarities([word_tokens(s) for s in base], [word_tokens(s) for s in target])
    # best[i][j]: the best total over base[:i] and target[:j]; move[i][j]: the step that gets it
    # ("d" pair, "a" target j-1 added, "r" base i-1 removed). Walking back from (n, m), "a" is
    # preferred to "r" on a tie, so once reversed the removed sentences come first.
    best = [[0] * (m + 1) for _ in range(n + 1)]
    move = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            options: list[tuple[int, str]] = []
            if i and j and (s := scores[i - 1][j - 1]) is not None:
                options.append((best[i - 1][j - 1] + round((s - THRESHOLD) * _SCALE), "d"))
            if j:
                options.append((best[i][j - 1], "a"))
            if i:
                options.append((best[i - 1][j], "r"))
            best[i][j], move[i][j] = max(options, key=lambda option: option[0])  # first wins ties

    items: list[tuple[str | None, str | None, float | None]] = []
    i, j = n, m
    while i or j:
        step = move[i][j]
        if step == "d":
            i, j = i - 1, j - 1
            items.append((base[i], target[j], scores[i][j]))
        elif step == "a":
            j -= 1
            items.append((None, target[j], None))
        else:
            i -= 1
            items.append((base[i], None, None))
    return [_change(k, *item) for k, item in enumerate(reversed(items))]


def _change(id_: int, base: str | None, target: str | None, score: float | None) -> Change:
    procedural = any(is_procedural(s) for s in (base, target) if s is not None)
    if base is None or target is None:
        kind = "added" if base is None else "removed"
        return Change(id_, kind, base, target, None, procedural)
    rounded = round(score or 0.0, SIMILARITY_DIGITS)
    if normalize_whitespace(base) == normalize_whitespace(target):
        return Change(id_, "unchanged", base, target, rounded, procedural)
    segments = word_segments(base, target)
    return Change(id_, "modified", base, target, rounded, procedural, segments)


def word_segments(base: str, target: str) -> tuple[Segment, ...]:
    """Word-level diff of two sentences: the equal and delete segments rebuild `base` exactly,
    the equal and insert segments `target`.

    Words and whitespace runs are matched as they are (case and punctuation count). Whitespace
    alone is never shown as common ground between two edits, and within each stretch of edits
    the deleted words come first, as one segment, then the inserted ones.
    """
    a, b = _PIECE.findall(base), _PIECE.findall(target)
    pieces: list[tuple[str, str]] = []  # (op, text); op "both" is an edit on both sides
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            pieces.append(("equal", "".join(a[i1:i2])))
            continue
        if i2 > i1:
            pieces.append(("delete", "".join(a[i1:i2])))
        if j2 > j1:
            pieces.append(("insert", "".join(b[j1:j2])))
    for k in range(1, len(pieces) - 1):  # whitespace alone between two edits joins them
        op, text = pieces[k]
        if (
            op == "equal"
            and not text.strip()
            and "equal" not in (pieces[k - 1][0], pieces[k + 1][0])
        ):
            pieces[k] = ("both", text)

    segments: list[Segment] = []
    deleted: list[str] = []
    inserted: list[str] = []
    for op, text in [*pieces, ("equal", "")]:
        if op != "equal":
            if op != "insert":
                deleted.append(text)
            if op != "delete":
                inserted.append(text)
            continue
        if deleted:
            segments.append(Segment("delete", "".join(deleted)))
        if inserted:
            segments.append(Segment("insert", "".join(inserted)))
        deleted, inserted = [], []
        if text:
            segments.append(Segment("equal", text))
    return tuple(segments)


def diff_statements(base: StatementText, target: StatementText) -> Diff:
    """Align the sentences of two statements and summarise the alignment."""
    changes = align(split_statement(base.paragraphs), split_statement(target.paragraphs))
    counts = Counter(change.kind for change in changes)
    overall = similarity(word_tokens(base.text), word_tokens(target.text))
    stats = ComparisonStats(
        unchanged=counts["unchanged"],
        modified=counts["modified"],
        added=counts["added"],
        removed=counts["removed"],
        similarity=round(overall, SIMILARITY_DIGITS),
    )
    return Diff(tuple(changes), stats)
