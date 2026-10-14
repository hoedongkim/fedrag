"""Tests for the sentence alignment and word diff of two statements (pure, no database)."""

import json
import os
import random
import subprocess
import sys
from dataclasses import asdict
from datetime import date

import pytest

from app.compare.diff import (
    THRESHOLD,
    align,
    diff_statements,
    similarity,
    split_statement,
    word_segments,
    word_tokens,
)
from app.compare.types import Change, Segment, StatementText

DAY = date(2024, 9, 18)

# Abridged from the FOMC statements of 2024-07-31 and 2024-09-18 (public domain).
JULY = (
    "Recent indicators suggest that economic activity has continued to expand at a solid pace. "
    "Job gains have moderated, and the unemployment rate has moved up but remains low. "
    "Inflation has eased over the past year but remains somewhat elevated.",
    "The Committee seeks to achieve maximum employment and inflation at the rate of 2 percent "
    "over the longer run. The Committee judges that the risks to achieving its employment and "
    "inflation goals continue to move into better balance.",
    "In support of its goals, the Committee decided to maintain the target range for the federal "
    "funds rate at 5-1/4 to 5-1/2 percent.",
    "Voting for the monetary policy action were Jerome H. Powell, Chair; John C. Williams, Vice "
    "Chair; and Michelle W. Bowman.",
)
SEPTEMBER = (
    "Recent indicators suggest that economic activity has continued to expand at a solid pace. "
    "Job gains have slowed, and the unemployment rate has moved up but remains low. "
    "Inflation has made further progress toward the Committee's 2 percent objective but remains "
    "somewhat elevated.",
    "The Committee seeks to achieve maximum employment and inflation at the rate of 2 percent "
    "over the longer run. The Committee has gained greater confidence that inflation is moving "
    "sustainably toward 2 percent, and judges that the risks to achieving its employment and "
    "inflation goals are roughly in balance.",
    "In light of the progress on inflation and the balance of risks, the Committee decided to "
    "lower the target range for the federal funds rate by 1/2 percentage point to 4-3/4 to 5 "
    "percent.",
    "Voting for the monetary policy action were Jerome H. Powell, Chair; John C. Williams, Vice "
    "Chair; and Michael S. Barr.",
    "Voting against this action was Michelle W. Bowman, who preferred to lower the target range "
    "by 1/4 percentage point at this meeting.",
)


def _statement(*paragraphs: str, doc_id: str = "statement-2024-09-18") -> StatementText:
    return StatementText(doc_id, DAY, DAY, tuple(paragraphs))


def _kinds(changes: list[Change] | tuple[Change, ...]) -> list[str]:
    return [change.kind for change in changes]


def _rebuild(segments: tuple[Segment, ...]) -> tuple[str, str]:
    base = "".join(s.text for s in segments if s.op != "insert")
    target = "".join(s.text for s in segments if s.op != "delete")
    return base, target


def _sides(changes: list[Change]) -> tuple[list[str], list[str]]:
    base = [c.base for c in changes if c.base is not None]
    target = [c.target for c in changes if c.target is not None]
    return base, target


# --- alignment -------------------------------------------------------------------------------


def test_identical_statements_are_all_unchanged() -> None:
    diff = diff_statements(_statement(*JULY), _statement(*JULY))

    assert set(_kinds(diff.changes)) == {"unchanged"}
    assert [c.id for c in diff.changes] == list(range(len(split_statement(JULY))))
    assert all(c.similarity == 1.0 and c.segments is None for c in diff.changes)
    assert diff.stats.model_dump() == {
        "unchanged": len(diff.changes),
        "modified": 0,
        "added": 0,
        "removed": 0,
        "similarity": 1.0,
    }


def test_whitespace_only_differences_are_unchanged() -> None:
    changes = align(["Rates  were\tcut. "], ["Rates were cut."])

    assert _kinds(changes) == ["unchanged"]
    assert changes[0].base == "Rates  were\tcut. "  # verbatim, not normalised


def test_real_pair() -> None:
    diff = diff_statements(_statement(*JULY), _statement(*SEPTEMBER))
    by_kind = {kind: [c for c in diff.changes if c.kind == kind] for kind in _kinds(diff.changes)}

    assert _kinds(diff.changes) == [
        "unchanged",  # economic activity
        "modified",  # job gains moderated -> slowed
        "modified",  # inflation eased -> made further progress (0.52, just over the threshold)
        "unchanged",  # mandate
        "modified",  # risks into better balance -> greater confidence, roughly in balance
        "modified",  # maintain -> lower by 1/2 percentage point
        "modified",  # roll call
        "added",  # dissent
    ], "the change ids moved: bump app.compare.types.DIFF_VERSION (stored analyses cite them)"
    assert [c.id for c in diff.changes] == list(range(8))
    assert [c.procedural for c in diff.changes] == [False] * 6 + [True, True]
    job = by_kind["modified"][0]
    assert job.similarity == 0.9286
    assert job.segments == (
        Segment("equal", "Job gains have "),
        Segment("delete", "moderated,"),
        Segment("insert", "slowed,"),
        Segment("equal", " and the unemployment rate has moved up but remains low."),
    )
    assert all(THRESHOLD <= c.similarity < 1 for c in by_kind["modified"])  # type: ignore[operator]
    assert diff.stats.model_dump(exclude={"similarity"}) == {
        "unchanged": 2,
        "modified": 5,
        "added": 1,
        "removed": 0,
    }
    assert 0.5 < diff.stats.similarity < 1


def test_fully_rewritten_lists_removed_before_added() -> None:
    base = ["Inflation is low.", "Rates are unchanged.", "The outlook is uncertain."]
    target = ["Payrolls surged in May.", "Purchases will taper next year."]

    changes = align(base, target)

    assert _kinds(changes) == ["removed"] * 3 + ["added"] * 2
    assert _sides(changes) == (base, target)
    assert all(c.similarity is None and c.segments is None for c in changes)
    stats = diff_statements(_statement(*base), _statement(*target)).stats
    assert (stats.removed, stats.added, stats.similarity) == (3, 2, 0.0)


@pytest.mark.parametrize(
    ("base", "target", "kinds"),
    [
        (["A b c d.", "E f g h."], ["New first one.", "A b c d.", "E f g h."], ["added", "u", "u"]),
        (["A b c d.", "E f g h."], ["A b c d.", "E f g h.", "New last."], ["u", "u", "added"]),
        (["Old first.", "A b c d.", "E f g h."], ["A b c d.", "E f g h."], ["removed", "u", "u"]),
        (["A b c d.", "E f g h.", "Old last."], ["A b c d.", "E f g h."], ["u", "u", "removed"]),
        ([], ["Only one."], ["added"]),
        (["Only one."], [], ["removed"]),
        ([], [], []),
    ],
    ids=["add-start", "add-end", "remove-start", "remove-end", "from-empty", "to-empty", "empty"],
)
def test_added_and_removed_at_the_edges(
    base: list[str], target: list[str], kinds: list[str]
) -> None:
    changes = align(base, target)

    assert [k.replace("unchanged", "u") for k in _kinds(changes)] == kinds
    assert _sides(changes) == (base, target)


def test_reordered_paragraphs_show_as_removed_and_added() -> None:
    first = "Economic activity expanded at a solid pace."
    second = "Job gains have been strong and unemployment is low."
    third = "Inflation remains somewhat elevated."
    changes = align([first, second, third], [first, third, second])

    assert sorted(_kinds(changes)) == ["added", "removed", "unchanged", "unchanged"]
    moved = [c.base for c in changes if c.kind == "removed"]
    assert moved == [c.target for c in changes if c.kind == "added"]
    assert changes[0].kind == "unchanged" and changes[0].base == first
    assert _sides(changes) == ([first, second, third], [first, third, second])


def test_sentence_split_in_two_is_modified_plus_added() -> None:
    whole = (
        "Job gains have slowed and the unemployment rate has moved up, but it remains low by "
        "historical standards."
    )
    head = "Job gains have slowed and the unemployment rate has moved up."
    tail = "It remains low by historical standards."

    changes = align([whole], [head, tail])

    assert _kinds(changes) == ["modified", "added"]
    assert (changes[0].base, changes[0].target, changes[1].target) == (whole, head, tail)


def test_threshold_is_inclusive() -> None:
    base = "alpha beta gamma delta."
    at = align([base], ["alpha beta zeta eta."])  # 2 of 4 + 4 tokens match: 0.5
    below = align([base], ["alpha beta zeta eta theta."])  # 2 * 2 / 9 = 0.444

    assert _kinds(at) == ["modified"] and at[0].similarity == THRESHOLD == 0.5
    assert _kinds(below) == ["removed", "added"]


def test_strong_pair_beats_two_weak_ones() -> None:
    # p-q2 (0.59) and q-p2 (0.64) keep document order too and sum to more than p-p2 (0.95), so
    # maximising the summed similarity would pick them; Needleman-Wunsch with a gap cost of
    # THRESHOLD / 2 weighs each pair by s - THRESHOLD and keeps the near-identical pair.
    p = "The Committee will closely monitor incoming information for the outlook."
    q = "The Committee will watch incoming data and the economic outlook carefully."
    q2 = "The Committee will closely monitor inflation expectations."
    p2 = "The Committee will closely monitor incoming information for the economic outlook."
    weak = [similarity(word_tokens(a), word_tokens(b)) for a, b in ((p, q2), (q, p2))]
    assert min(weak) >= THRESHOLD and sum(weak) > similarity(word_tokens(p), word_tokens(p2))

    changes = align([p, q], [q2, p2])

    assert [(c.kind, c.base, c.target) for c in changes] == [
        ("added", None, q2),
        ("modified", p, p2),
        ("removed", q, None),
    ]


def test_procedural_sentences_are_kept_and_flagged() -> None:
    statement = (
        "The Committee decided to maintain the target range. Voting for the monetary policy "
        "action were Jerome H. Powell, Chair; and John C. Williams, Vice Chair."
    )
    sentences = split_statement([statement])

    changes = align(sentences, sentences[:1])

    assert len(sentences) == 2
    assert [(c.kind, c.procedural) for c in changes] == [("unchanged", False), ("removed", True)]


def test_case_and_punctuation_changes_are_modified() -> None:
    changes = align(["Inflation remains elevated."], ["inflation remains elevated!"])

    assert _kinds(changes) == ["modified"]
    assert changes[0].similarity == 1.0  # same words; the segments show what differs
    assert changes[0].segments == (
        Segment("delete", "Inflation"),
        Segment("insert", "inflation"),
        Segment("equal", " remains "),
        Segment("delete", "elevated."),
        Segment("insert", "elevated!"),
    )


def test_unicode_and_punctuation() -> None:
    base = "The Committee\u2019s 2\u00a0percent goal \u2014 \u201cpersistent\u201d \ufb01rms."
    target = 'The Committee\'s 2 percent goal - "persistent" firms.'

    assert word_tokens(base) == word_tokens(target)  # NFKC, case and punctuation aside
    changes = align([base], [target])
    assert _kinds(changes) == ["modified"]
    assert changes[0].similarity == 1.0
    assert _rebuild(changes[0].segments or ()) == (base, target)
    assert word_tokens("Rate: 4-3/4 to 5 percent.") == ["rate", "4", "3", "4", "to", "5", "percent"]


# --- word segments ---------------------------------------------------------------------------


def test_segments_group_adjacent_edits() -> None:
    segments = word_segments(
        "Information received since June indicates that activity has expanded.",
        "Information received since July suggests that activity is expanding.",
    )

    assert segments == (
        Segment("equal", "Information received since "),
        Segment("delete", "June indicates"),
        Segment("insert", "July suggests"),
        Segment("equal", " that activity "),
        Segment("delete", "has expanded."),
        Segment("insert", "is expanding."),
    )


WORDS = str.split(
    "the Committee inflation labor market rate target range federal funds percent 2 1/4 "
    "elevated moderate solid pace economic activity, further; risks balance. "
    "\u201cquoted\u201d na\u00efve \u00e9lan \u2014 3-3/4"
)


def _random_sentence(rng: random.Random) -> list[str]:
    return [rng.choice(WORDS) for _ in range(rng.randint(0, 25))]


def _join(rng: random.Random, words: list[str]) -> str:
    gaps = [rng.choice([" ", " ", " ", "  ", "\t", "\u00a0", "\n"]) for _ in words]
    edges = [rng.choice(["", " "]) for _ in range(2)]
    return edges[0] + "".join(w + g for w, g in zip(words, gaps, strict=True))[:-1] + edges[1]


def _edit(rng: random.Random, words: list[str]) -> list[str]:
    edited = list(words)
    for _ in range(rng.randint(0, 6)):
        op = rng.choice(["insert", "delete", "replace", "swap"])
        k = rng.randint(0, max(len(edited) - 1, 0))
        if op == "insert" or not edited:
            edited.insert(k, rng.choice(WORDS))
        elif op == "delete":
            del edited[k]
        elif op == "replace":
            edited[k] = rng.choice(WORDS)
        else:
            j = rng.randint(0, len(edited) - 1)
            edited[k], edited[j] = edited[j], edited[k]
    return edited


@pytest.mark.parametrize("seed", range(20))
def test_segments_rebuild_both_sentences_exactly(seed: int) -> None:
    rng = random.Random(seed)
    for _ in range(50):
        words = _random_sentence(rng)
        base, target = _join(rng, words), _join(rng, _edit(rng, words))

        segments = word_segments(base, target)

        assert _rebuild(segments) == (base, target)
        assert all(s.text for s in segments)
        assert all(a.op != b.op for a, b in zip(segments, segments[1:], strict=False))
        for a, b in zip(segments, segments[1:], strict=False):
            assert (a.op, b.op) != ("insert", "delete")  # deletions come first in an edit
        for prev, seg, nxt in zip(segments, segments[1:], segments[2:], strict=False):
            if seg.op == "equal" and not seg.text.strip():
                assert "equal" in (prev.op, nxt.op)  # whitespace alone never splits an edit


@pytest.mark.parametrize("seed", range(10))
def test_alignment_keeps_both_statements_in_order(seed: int) -> None:
    rng = random.Random(seed)
    base = [_join(rng, _random_sentence(rng)).strip() or "x." for _ in range(rng.randint(0, 12))]
    target = [s if rng.random() < 0.4 else _join(rng, _edit(rng, s.split())) for s in base]
    head = target[: len(target) // 3]  # move some sentences: shuffle the first third
    rng.shuffle(head)
    target[: len(head)] = head
    target += [_join(rng, _random_sentence(rng)) for _ in range(rng.randint(0, 3))]

    changes = align(base, target)

    assert _sides(changes) == (base, target)
    assert [c.id for c in changes] == list(range(len(changes)))
    for change in changes:
        if change.kind == "modified":
            assert change.similarity is not None and change.similarity >= THRESHOLD
            assert _rebuild(change.segments or ()) == (change.base, change.target)
        else:
            assert change.segments is None


# --- determinism -----------------------------------------------------------------------------

DETERMINISM_SCRIPT = """
import json, sys
from dataclasses import asdict
from datetime import date
from app.compare.diff import diff_statements
from app.compare.types import StatementText
base, target = json.loads(sys.argv[1])
day = date(2024, 9, 18)
base, target = (StatementText(i, day, day, tuple(t)) for i, t in (("a", base), ("b", target)))
diff = diff_statements(base, target)
print(json.dumps([[asdict(c) for c in diff.changes], diff.stats.model_dump()]))
"""


def test_diff_is_deterministic_across_runs_and_hash_seeds() -> None:
    first = diff_statements(_statement(*JULY), _statement(*SEPTEMBER))
    again = diff_statements(_statement(*JULY), _statement(*SEPTEMBER))
    assert first == again

    outputs = set()
    for seed in ("0", "1", "2"):
        result = subprocess.run(
            [sys.executable, "-c", DETERMINISM_SCRIPT, json.dumps([JULY, SEPTEMBER])],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        outputs.add(result.stdout)
    assert len(outputs) == 1
    changes, stats = json.loads(outputs.pop())
    assert changes == json.loads(json.dumps([asdict(c) for c in first.changes]))
    assert stats == first.stats.model_dump()
