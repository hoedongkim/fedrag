"""Tests for statement sentence splitting, the procedural filter and score aggregation."""

from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pytest

from app.core.config import Settings
from app.ingest.records import Document, IngestDocType, Paragraph, write_jsonl
from app.schemas.common import Stance
from app.stance.keyword import KeywordClassifier, Lexicon
from app.stance.statements import (
    AGGREGATION,
    RUBRIC_ORDERED,
    Statement,
    aggregate_statements,
    is_procedural,
    is_rate_guidance,
    label_statements,
    load_statements,
    predict_statements,
    prediction_from_json,
    prediction_to_json,
    predictions_path,
    rate_action,
    rate_move_rule,
    read_predictions,
    rubric_decision,
    split_sentences,
    stance_from_score,
    write_predictions,
)
from app.stance.types import SentencePrediction


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "The fundamentals of the U.S. economy remain strong. Rates were held.",
            ["The fundamentals of the U.S. economy remain strong.", "Rates were held."],
        ),
        (
            "The statement was released at 2:00 p.m. EST today. Markets rallied.",
            ["The statement was released at 2:00 p.m. EST today.", "Markets rallied."],
        ),
        (
            "Voting were Jerome H. Powell, Chair; and John C. Williams. Ms. Daly voted.",
            ["Voting were Jerome H. Powell, Chair; and John C. Williams.", "Ms. Daly voted."],
        ),
        (
            "The range is 5-1/4 to 5-1/2 percent. A 0.5 percentage point cut was weighed.",
            ["The range is 5-1/4 to 5-1/2 percent.", "A 0.5 percentage point cut was weighed."],
        ),
        (
            'The Committee said it can be "patient." Rates were held.',
            ['The Committee said it can be "patient."', "Rates were held."],
        ),
        (
            'It wrote "Growth slowed. Prices rose" in its notes. Next came the vote.',
            ['It wrote "Growth slowed. Prices rose" in its notes.', "Next came the vote."],
        ),
        (
            "It said \u201cpolicy is well positioned.\u201d The outlook is uncertain!  Is it?",
            [
                "It said \u201cpolicy is well positioned.\u201d",
                "The outlook is uncertain!",
                "Is it?",
            ],
        ),
        (
            "approved by a 12 - 0 vote: The Committee decided to act.",
            ["approved by a 12 - 0 vote: The Committee decided to act."],
        ),
        ("  Inflation remains elevated.  ", ["Inflation remains elevated."]),
        ("", []),
    ],
    ids=[
        "us",
        "pm",
        "initials",
        "fractions",
        "quote-end",
        "quote-inside",
        "curly",
        "colon",
        "spaces",
        "empty",
    ],
)
def test_split_sentences(text: str, expected: list[str]) -> None:
    sentences = split_sentences(text)
    assert sentences == expected
    assert all(sentence in text for sentence in sentences)


@pytest.mark.parametrize(
    "sentence",
    [
        "Voting for the monetary policy action were Jerome H. Powell, Chair; and Lisa D. Cook.",
        "Voting for the FOMC monetary policy action were: Janet L. Yellen, Chair.",
        "Voting (by notation) for the monetary policy action were Jerome H. Powell, Chair.",
        "Voting against the action was Esther L. George, who preferred to raise the target range.",
        "Patrick Harker voted as an alternate member at this meeting.",
        "Absent and not voting was Adriana D. Kugler.",
        "For media inquiries, please email media@frb.gov or call 202-452-2955.",
        "The Federal Open Market Committee approved the following statement for release by a "
        "12 - 0 vote:",
    ],
)
def test_procedural_sentences(sentence: str) -> None:
    assert is_procedural(sentence)


@pytest.mark.parametrize(
    "sentence",
    [
        "The Committee decided to raise the target range for the federal funds rate.",
        "In connection with these plans, the Federal Open Market Committee voted unanimously to "
        "authorize the purchases.",
        "Inflation remains elevated.",
    ],
)
def test_substantive_sentences(sentence: str) -> None:
    assert not is_procedural(sentence)


def statement(*paragraphs: str, doc_id: str = "statement-2024-09-18") -> Statement:
    return Statement(doc_id, date.fromisoformat(doc_id.removeprefix("statement-")), paragraphs)


def test_statement_sentences_drop_procedural_ones() -> None:
    st = statement(
        "Inflation has eased. Job gains have slowed.",
        "Voting for the monetary policy action were Jerome H. Powell, Chair; and Lisa D. Cook. "
        "Voting against this action was Michelle W. Bowman. Neel Kashkari voted as an alternate "
        "member at this meeting.",
    )
    assert st.sentences() == ["Inflation has eased.", "Job gains have slowed."]


@pytest.mark.parametrize(
    ("score", "stance"),
    [
        (0.15, "neutral"),
        (0.1501, "hawkish"),
        (-0.15, "neutral"),
        (-0.1501, "dovish"),
        (0, "neutral"),
    ],
)
def test_fixed_thresholds(score: float, stance: Stance) -> None:
    assert stance_from_score(score) == stance


class Fixed:
    """A fake sentence classifier: each sentence's p(hawkish) and p(dovish) from a table."""

    name = "finbert"

    def __init__(self, table: dict[str, tuple[float, float]]) -> None:
        self.table = table
        self.calls = 0

    def predict_sentences(self, sentences: Sequence[str]) -> list[SentencePrediction]:
        self.calls += 1
        out = []
        for s in sentences:
            hawk, dove = self.table[s]
            probs: dict[Stance, float] = {
                "hawkish": hawk,
                "dovish": dove,
                "neutral": 1 - hawk - dove,
            }
            out.append(SentencePrediction(max(probs, key=probs.__getitem__), probs))
        return out


def test_score_is_the_mean_of_p_hawkish_minus_p_dovish() -> None:
    table = {"A rise.": (0.9, 0.0), "A hold.": (0.1, 0.1), "A cut.": (0.0, 0.5), "B.": (0.2, 0.0)}
    classifier = Fixed(table)
    first, second = predict_statements(
        classifier,
        [statement("A rise. A hold. A cut."), statement("B.", doc_id="statement-2024-11-07")],
        model_version="v1",
    )
    assert classifier.calls == 1  # every statement's sentences in one call
    assert first.score == pytest.approx((0.9 + 0.0 - 0.5) / 3)
    assert (first.stance, first.confidence) == ("neutral", pytest.approx(0.4 / 3))
    assert first.evidence == ("A rise.", "A cut.")  # neutral: the strongest pull each way
    assert first.key_factors == ()  # FinBERT names no terms
    assert (first.classifier, first.model_version) == ("finbert", "v1")
    assert (second.stance, second.score, second.evidence) == (
        "hawkish",
        pytest.approx(0.2),
        ("B.",),
    )
    assert second.meeting_date == date(2024, 11, 7)


def test_evidence_follows_the_stance_and_stays_verbatim() -> None:
    text = "Prices rose sharply. Growth held up. Wages rose. Rents fell."
    table = {
        "Prices rose sharply.": (0.9, 0.0),
        "Growth held up.": (0.5, 0.1),
        "Wages rose.": (0.6, 0.0),
        "Rents fell.": (0.0, 0.8),
    }
    (result,) = predict_statements(Fixed(table), [statement(text)])
    assert result.stance == "hawkish"
    assert result.evidence == ("Prices rose sharply.", "Wages rose.")
    assert all(sentence in text for sentence in result.evidence)


def test_label_only_classifiers_count_plus_minus_one() -> None:
    lexicon = Lexicon(("rose", "tight"), ("fell",), 2)
    st = statement("Prices rose. Wages rose. Rents fell. Policy is tight. Nothing happened.")
    (result,) = predict_statements(KeywordClassifier(lexicon), [st])
    assert result.score == pytest.approx((1 + 1 - 1 + 1 + 0) / 5)
    assert (result.stance, result.classifier) == ("hawkish", "keyword")
    assert result.key_factors == ("rose", "tight")  # hawkish terms only, most frequent first
    assert result.confidence == pytest.approx(0.4)
    assert result.model_version.startswith("keyword-lexicon-v1 (TDW train; top_k 2; vote);")
    assert result.model_version.endswith(AGGREGATION)


def test_neutral_evidence_takes_one_sentence_from_each_side() -> None:
    table = {"Up a lot.": (0.6, 0.0), "Up.": (0.3, 0.0), "Down.": (0.0, 0.5), "Flat.": (0.0, 0.0)}
    (result,) = predict_statements(Fixed(table), [statement("Down. Up a lot. Up. Flat.")])
    assert result.stance == "neutral"
    assert result.evidence == ("Down.", "Up a lot.")
    (one_sided,) = predict_statements(Fixed(table), [statement("Up. Flat. Flat.")])
    assert (one_sided.stance, one_sided.evidence) == ("neutral", ("Up.",))


def test_scores_and_confidence_are_clipped() -> None:
    (result,) = predict_statements(Fixed({"Up.": (1.0, 0.0)}), [statement("Up.")])
    assert (result.score, result.confidence) == (1.0, 1.0)


def test_predictions_round_trip_through_jsonl(tmp_path: Path) -> None:
    table = {"Prices rose.": (0.7, 0.1)}
    predictions = predict_statements(Fixed(table), [statement("Prices rose.")], model_version="v")
    path = tmp_path / "stance" / "predictions.jsonl"
    assert write_predictions(path, predictions) == 1
    assert read_predictions(path) == [
        prediction_from_json(prediction_to_json(p)) for p in predictions
    ]
    row = prediction_to_json(predictions[0])
    assert row["meeting_date"] == "2024-09-18"
    assert row["evidence"] == ["Prices rose."]


def test_load_statements_reads_processed_documents(tmp_path: Path) -> None:
    def document(doc_type: IngestDocType, day: date) -> dict[str, object]:
        return Document(
            doc_type=doc_type,
            meeting_date=day,
            release_date=day,
            release_date_source="calendar",
            title="FOMC",
            url="https://www.federalreserve.gov/x.htm",
            paragraphs=(Paragraph(None, "Rates were held."), Paragraph(None, "Voting for ...")),
        ).to_json()

    rows = [
        document("statement", date(2024, 9, 18)),
        document("minutes", date(2024, 9, 18)),
        document("statement", date(2015, 1, 28)),
    ]
    write_jsonl(tmp_path / "processed" / "documents.jsonl", rows)
    statements = load_statements(Settings(_env_file=None, data_dir=tmp_path))
    assert [s.doc_id for s in statements] == ["statement-2015-01-28", "statement-2024-09-18"]
    assert statements[0].paragraphs == ("Rates were held.", "Voting for ...")


# --- Post-hoc variant: rubric-ordered aggregation ---------------------------------------------

RAISE = (
    "In support of these goals, the Committee decided to raise the target range for the federal "
    "funds rate to 5 to 5-1/4 percent."
)
HOLD = (
    "In support of its goals, the Committee decided to maintain the target range for the federal "
    "funds rate at 4-1/4 to 4-1/2 percent."
)
GUIDE = (
    "In considering the extent and timing of additional adjustments to the target range for the "
    "federal funds rate, the Committee will carefully assess incoming data."
)
FIRMING = "The Committee anticipates that some additional policy firming may be appropriate."
# No rate-path term, so never guidance: a filler sentence, not a test that outlook sentences are
# excluded (the corpus's own outlook sentences do count; see test_rate_guidance).
NO_RATE_PATH = "Inflation is expected to rise to 2 percent over the medium term."
RUNOFF = "The Committee will continue reducing its holdings of Treasury securities."


@pytest.mark.parametrize(
    ("sentence", "kind"),
    [
        (RAISE, "move"),
        (HOLD, "hold"),
        (
            "In light of these risks, the Federal Open Market Committee decided today to lower the "
            "target range for the federal funds rate by 1/2 percentage point, to 1 to 1-1/4 "
            "percent.",
            "move",
        ),
        (
            "The Committee decided to raise the target range for the federal funds rate by 1/4 "
            "percentage point to 3-3/4 to 4 percent, in support of the Federal Reserve's dual "
            "mandate.",
            "move",
        ),
        (
            "The Committee decided to keep the target range for the federal funds rate at 0 to "
            "1/4 percent and expects it will be appropriate to maintain this target range until "
            "labor market conditions have reached maximum employment.",
            "hold",
        ),
        (
            "To support continued progress, the Committee today reaffirmed its view that the "
            "current 0 to 1/4 percent target range for the federal funds rate remains appropriate.",
            "hold",
        ),
        (
            "The Committee decided to begin reducing the monthly pace of its net asset purchases.",
            None,
        ),
        ("The Committee reaffirmed its policy of maintaining ample reserves.", None),
        (GUIDE, None),
    ],
    ids=[
        "raise",
        "maintain",
        "decided-today",
        "2026-format",
        "keep-and-guidance",
        "reaffirmed",
        "taper",
        "reserves",
        "guidance",
    ],
)
def test_rate_action(sentence: str, kind: str | None) -> None:
    found = rate_action(sentence)
    assert (found[0] if found else None) == kind


@pytest.mark.parametrize(
    ("sentence", "guidance"),
    [
        (GUIDE, True),
        (FIRMING, True),
        (
            "The Committee judges that the case for an increase in the federal funds rate has "
            "strengthened but decided, for the time being, to wait for further evidence.",
            True,
        ),
        ("The stance of monetary policy remains accommodative.", True),
        (
            "The Committee decided to keep the target range for the federal funds rate at 0 to "
            "1/4 percent and expects it will be appropriate to maintain this target range until "
            "inflation has risen to 2 percent.",
            True,
        ),
        (
            "In view of realized and expected labor market conditions and inflation, the "
            "Committee decided to raise the target range for the federal funds rate to 1 percent.",
            False,
        ),
        (HOLD, False),
        (NO_RATE_PATH, False),
        # Corpus wording that the post-hoc variant counts as guidance (documented, not fixed:
        # leaving it out would be one more post-hoc fork).
        (
            "The Committee expects that, with gradual adjustments in the stance of monetary "
            "policy, economic activity will expand at a moderate pace and labor market conditions "
            "will strengthen somewhat further.",
            True,
        ),
        (
            "The Committee would be prepared to adjust the stance of monetary policy as "
            "appropriate if risks emerge that could impede the attainment of the Committee's "
            "goals.",
            True,
        ),
        (RUNOFF, False),
        ("The Committee will deliver price stability.", False),
        ("The Committee is attentive to the risks to both sides of its dual mandate.", False),
    ],
    ids=[
        "adjustments",
        "firming",
        "case-for-increase",
        "accommodative",
        "hold-with-guidance",
        "expected-before-decision",
        "plain-hold",
        "no-rate-path-term",
        "outlook-with-adjustments",
        "prepared-if-risks-emerge",
        "balance-sheet",
        "2026-commitment",
        "risk-balance",
    ],
)
def test_rate_guidance(sentence: str, guidance: bool) -> None:
    assert is_rate_guidance(sentence) is guidance


@pytest.mark.parametrize(
    ("sentences", "stance"),
    [
        ([NO_RATE_PATH, RAISE, FIRMING], "hawkish"),
        (
            [
                "In light of these risks, the Federal Open Market Committee decided today to "
                "lower the target range for the federal funds rate by 1/2 percentage point."
            ],
            "dovish",
        ),
        (["The Committee decided to cut the target range by 1/4 percentage point."], "dovish"),
        ([HOLD, GUIDE, FIRMING], "neutral"),  # guidance never counts: no classifier
        ([HOLD, RAISE], "neutral"),  # the first rate-action sentence decides, as in the rubric
        ([NO_RATE_PATH, RUNOFF], "neutral"),
        ([], "neutral"),
    ],
    ids=["raise", "decided-today-to-lower", "cut", "hold", "first-action", "no-action", "empty"],
)
def test_rate_move_rule_reads_only_the_decision_verb(sentences: list[str], stance: Stance) -> None:
    assert rate_move_rule(sentences) == stance


def probs(hawk: float, dove: float) -> SentencePrediction:
    table: dict[Stance, float] = {"hawkish": hawk, "dovish": dove, "neutral": 1 - hawk - dove}
    return SentencePrediction(max(table, key=table.__getitem__), table)


def test_rate_move_label_decides_with_its_own_score() -> None:
    decision = rubric_decision(
        [NO_RATE_PATH, RAISE, FIRMING], [probs(0.0, 0.9), probs(0.7, 0.1), probs(0.0, 0.8)]
    )
    assert (decision.step, decision.stance, decision.sentences) == ("rate_action", "hawkish", (1,))
    assert decision.score == pytest.approx(0.6)
    # The classifier decides the direction: a raise it calls dovish is dovish.
    dovish = rubric_decision([RAISE], [probs(0.1, 0.6)])
    assert (dovish.step, dovish.stance) == ("rate_action", "dovish")


def test_neutral_move_and_hold_fall_through_to_guidance() -> None:
    sentences = [RAISE, NO_RATE_PATH, GUIDE, FIRMING, RUNOFF]
    labels = [probs(0.3, 0.2), probs(0.0, 0.9), probs(0.2, 0.0), probs(0.5, 0.0), probs(0, 0.9)]
    decision = rubric_decision(sentences, labels)  # the raise sentence is labelled neutral
    assert (decision.step, decision.stance, decision.sentences) == ("guidance", "hawkish", (2, 3))
    assert decision.score == pytest.approx(0.35)
    held = rubric_decision([HOLD, GUIDE], [probs(0.9, 0.0), probs(0.0, 0.1)])
    assert (held.step, held.stance, held.score) == ("none", "neutral", pytest.approx(-0.1))


def test_no_action_and_no_guidance_is_neutral_zero() -> None:
    decision = rubric_decision([NO_RATE_PATH, RUNOFF], [probs(0.9, 0.0), probs(0.0, 0.9)])
    assert (decision.step, decision.stance, decision.score, decision.sentences) == (
        "none",
        "neutral",
        0.0,
        (),
    )
    with pytest.raises(ValueError):
        rubric_decision([NO_RATE_PATH], [])


def test_post_hoc_variant_from_the_same_sentence_labels() -> None:
    table = {RAISE: (0.8, 0.0), NO_RATE_PATH: (0.0, 0.6), GUIDE: (0.0, 0.0), FIRMING: (0.1, 0.0)}
    classifier = Fixed(table)
    st = statement(" ".join([NO_RATE_PATH, RAISE, GUIDE, FIRMING]))
    labelled = label_statements(classifier, [st])
    (pre,) = aggregate_statements(classifier, labelled)
    (post,) = aggregate_statements(classifier, labelled, variant="post_hoc_rubric_ordered")
    assert classifier.calls == 1
    assert (pre.stance, pre.score) == ("neutral", pytest.approx(0.3 / 4))
    assert (post.stance, post.score, post.confidence) == ("hawkish", 0.8, 0.8)
    assert post.evidence == (RAISE,)
    assert post.model_version.endswith(RUBRIC_ORDERED)
    assert pre.model_version.endswith(AGGREGATION)
    (same,) = predict_statements(Fixed(table), [st], variant="post_hoc_rubric_ordered")
    assert same == post


def test_post_hoc_key_factors_come_from_the_deciding_sentences() -> None:
    lexicon = Lexicon(("raise", "firming"), ("expected",), 2)
    st = statement(" ".join([HOLD, NO_RATE_PATH, FIRMING]))
    (result,) = predict_statements(
        KeywordClassifier(lexicon), [st], variant="post_hoc_rubric_ordered"
    )
    # The hold goes to guidance: only FIRMING is a guidance sentence (NO_RATE_PATH is not).
    assert (result.stance, result.score) == ("hawkish", 1.0)
    assert result.key_factors == ("firming",)
    assert result.evidence == (FIRMING,)


def test_post_hoc_predictions_have_their_own_file(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path)
    assert predictions_path(settings).name == "predictions.jsonl"
    assert predictions_path(settings, "post_hoc_rubric_ordered").name == (
        "predictions_post_hoc.jsonl"
    )
