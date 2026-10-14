"""Statement-level stance from a sentence classifier: split, drop procedural sentences, aggregate.

A statement's score is the mean over its non-procedural sentences of p(hawkish) - p(dovish), or
of +1 / 0 / -1 for a classifier that only emits labels (keyword). The stance comes from fixed
thresholds set before any gold label existed and never tuned on them: score > 0.15 is hawkish,
score < -0.15 dovish, anything else neutral. confidence is |score| clipped to [0, 1], derived from
the score and not a probability. Evidence is up to two verbatim sentences that pull hardest in the
statement's direction (for neutral, the strongest pull each way); key factors are the most frequent
matched lexicon terms (keyword only).

That averaging is the pre-registered method, and it stays the default everywhere (predictions.jsonl,
the database, the API). aggregate_rubric_ordered is a post-hoc variant, designed after the
pre-registered results were seen, that follows the labelling rubric's decision order instead of
averaging; its predictions go to a separate file and are reported as post hoc.
"""

import json
import re
import statistics
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from app.core.config import Settings, get_settings
from app.db.load import DOCUMENTS_FILE, DocumentRow, read_jsonl
from app.ingest.records import write_jsonl
from app.schemas.common import Stance
from app.schemas.stance import StatementVariant
from app.stance.types import (
    ORDINAL,
    SentenceClassifier,
    SentencePrediction,
    StatementPrediction,
)

THRESHOLD = 0.15
EVIDENCE_COUNT = 2
KEY_FACTOR_COUNT = 3
PREDICTIONS_FILES: dict[StatementVariant, str] = {  # under data/stance
    "pre_registered": "predictions.jsonl",  # every classifier's statements
    "post_hoc_rubric_ordered": "predictions_post_hoc.jsonl",  # sentence classifiers only
}
AGGREGATION = f"mean sentence score, thresholds +-{THRESHOLD}"  # appended to model_version
RUBRIC_ORDERED = (
    "rubric-ordered (post hoc): rate-action sentence's label, else mean score of the "
    f"rate-guidance sentences with thresholds +-{THRESHOLD}, else neutral"
)
AGGREGATIONS: dict[StatementVariant, str] = {
    "pre_registered": AGGREGATION,
    "post_hoc_rubric_ordered": RUBRIC_ORDERED,
}

# Statement Comparison's change ids and procedural flags come from split_sentences, is_procedural
# and the ABBREVIATIONS / PROCEDURAL tables below (app.compare.diff): any change in their output
# requires bumping app.compare.types.DIFF_VERSION, or stored analyses cite the wrong rows.

# Words that end with a period without ending the sentence (compared lower-case, final period
# removed). Dotted abbreviations (U.S., a.m., e.g.) and single initials (Jerome H. Powell) are
# recognized by shape, so "the U.S. The ..." stays one sentence: rare in FOMC statements.
ABBREVIATIONS = frozenset(
    str.split(
        "mr ms mrs dr jr sr st no nos vs etc inc co corp ltd dept gov "
        "approx jan feb mar apr jun jul aug sep sept oct nov dec"
    )
)
_BOUNDARY = re.compile(r"[.!?]+[\"'\u201d\u2019)\]]*(?=\s+[\"'\u201c\u2018(\[]*[A-Z0-9])")
_LAST_WORD = re.compile(r"(\S+)$")
_DOTTED = re.compile(r"^[A-Za-z](?:\.[A-Za-z])+$")
PROCEDURAL = (
    re.compile(r"^Voting\b"),  # "Voting for the monetary policy action were ...", "Voting against"
    re.compile(r"\bvoted as an? alternate member\b"),
    re.compile(r"^Absent and not voting\b"),
    re.compile(r"^For media inquiries\b", re.IGNORECASE),
    re.compile(r"\bapproved the following statement for release\b"),
)


def _inside_quote(text: str) -> bool:
    """Whether `text` leaves a quotation open (curly quotes, or an odd number of straight ones)."""
    return text.count("\u201c") > text.count("\u201d") or text.count('"') % 2 == 1


def split_sentences(text: str) -> list[str]:
    """Sentences of a paragraph, each a verbatim (stripped) substring of it."""
    sentences: list[str] = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        before = _LAST_WORD.search(text, start, match.start())
        word = (before.group(1) if before else "").lstrip("\"'(\u201c\u2018[")
        if (
            (match.group(0) == "." and word.lower() in ABBREVIATIONS)
            or (match.group(0) == "." and (_DOTTED.match(word) or re.fullmatch(r"[A-Z]", word)))
            or _inside_quote(text[start : match.end()])
        ):
            continue
        sentences.append(text[start : match.end()].strip())
        start = match.end()
    rest = text[start:].strip()
    return [s for s in [*sentences, rest] if s]


def is_procedural(sentence: str) -> bool:
    """Roll calls, dissent lists, alternate-member notes, vote tallies and media contacts."""
    return any(pattern.search(sentence) for pattern in PROCEDURAL)


@dataclass(frozen=True)
class Statement:
    """An FOMC statement's text as paragraphs, from data/processed/documents.jsonl."""

    doc_id: str
    meeting_date: date
    paragraphs: tuple[str, ...]

    @property
    def text(self) -> str:
        """The whole statement as stored (paragraphs joined by blank lines), for the LLM."""
        return "\n\n".join(self.paragraphs)

    def sentences(self) -> list[str]:
        """Its substantive sentences, in order: procedural ones are dropped."""
        return [
            sentence
            for paragraph in self.paragraphs
            for sentence in split_sentences(paragraph)
            if not is_procedural(sentence)
        ]


def load_statements(settings: Settings | None = None) -> list[Statement]:
    """Every statement in data/processed/documents.jsonl, oldest first."""
    rows = read_jsonl(
        (settings or get_settings()).data_dir / "processed" / DOCUMENTS_FILE, DocumentRow
    )
    return sorted(
        (
            Statement(row.doc_id, row.meeting_date, tuple(p.text for p in row.paragraphs))
            for row in rows
            if row.doc_type == "statement"
        ),
        key=lambda s: s.meeting_date,
    )


def sentence_score(prediction: SentencePrediction) -> float:
    """p(hawkish) - p(dovish), or +1 / 0 / -1 from the label when there are no probabilities."""
    if prediction.probs:
        return prediction.probs.get("hawkish", 0.0) - prediction.probs.get("dovish", 0.0)
    return float(ORDINAL[prediction.stance])


def stance_from_score(score: float) -> Stance:
    return "hawkish" if score > THRESHOLD else "dovish" if score < -THRESHOLD else "neutral"


@runtime_checkable
class ExplainsTerms(Protocol):
    """A sentence classifier that can name the lexicon terms behind its labels (keyword)."""

    def matches(self, sentence: str) -> list[tuple[str, int]]: ...


def _evidence(sentences: Sequence[str], scores: Sequence[float], stance: Stance) -> tuple[str, ...]:
    """Up to EVIDENCE_COUNT sentences scoring furthest in the stance's direction, or for neutral
    the strongest pull each way; in statement order. A zero score never qualifies."""

    def strongest(sign: int, count: int) -> list[int]:
        pulls = [(-score * sign, i) for i, score in enumerate(scores) if score * sign > 0]
        return [i for _, i in sorted(pulls)[:count]]

    sign = ORDINAL[stance]
    chosen = strongest(sign, EVIDENCE_COUNT) if sign else strongest(1, 1) + strongest(-1, 1)
    return tuple(sentences[i] for i in sorted(chosen))


def _key_factors(classifier: object, sentences: Sequence[str], stance: Stance) -> tuple[str, ...]:
    """The most frequent matched terms that point the stance's way (any way for neutral)."""
    if not isinstance(classifier, ExplainsTerms):
        return ()
    sign = ORDINAL[stance]
    counts = Counter(
        term
        for sentence in sentences
        for term, polarity in classifier.matches(sentence)
        if not sign or polarity == sign
    )
    ranked = sorted(counts, key=lambda term: (-counts[term], term))
    return tuple(ranked[:KEY_FACTOR_COUNT])


def aggregate(
    statement: Statement,
    sentences: Sequence[str],
    predictions: Sequence[SentencePrediction],
    classifier: SentenceClassifier,
    model_version: str,
) -> StatementPrediction:
    """One statement's prediction from its sentences' predictions."""
    scores = [sentence_score(p) for p in predictions]
    score = sum(scores) / len(scores) if scores else 0.0
    stance = stance_from_score(score)
    return StatementPrediction(
        doc_id=statement.doc_id,
        meeting_date=statement.meeting_date,
        classifier=classifier.name,
        stance=stance,
        score=max(-1.0, min(1.0, score)),
        confidence=min(1.0, abs(score)),
        key_factors=_key_factors(classifier, sentences, stance),
        evidence=_evidence(sentences, scores, stance),
        model_version=model_version,
    )


# --- Post-hoc variant: the rubric's decision order -------------------------------------------
#
# Written from docs/stance-labeling.md (decision order, compact rubric) and from reading the
# statements' text, never from reference labels (the step-3/4 exclusion followed FinBERT's labels
# on these statements' balance-sheet sentences). On the 97 statements of 2015-2026, RATE_MOVE
# finds 32 statements and RATE_HOLD 63; the two without a rate-action sentence are
# statement-2019-10-04 (reserve-management bill purchases) and statement-2020-03-23 (asset
# purchases only), and none has more than one.

# This meeting's decision on the target range. "decided today to lower" is the March 2020 wording;
# the 2026 format ("The Committee decided to raise the target range ... by 1/4 percentage point
# ..., in support of the Federal Reserve's dual mandate.") matches the plain form.
RATE_MOVE = re.compile(
    r"\bdecided(?:\s+today)?\s+to\s+(?P<verb>raise|increase|lower|reduce|cut)\s+the\s+target\s+"
    r"range\b",
    re.IGNORECASE,
)
MOVE_DIRECTIONS: dict[str, Stance] = {
    "raise": "hawkish",
    "increase": "hawkish",
    "lower": "dovish",
    "reduce": "dovish",
    "cut": "dovish",
}
RATE_HOLD = re.compile(
    r"\bdecided(?:\s+today)?\s+to\s+(?:maintain|keep|hold|leave)\s+the\s+target\s+range\b"
    # 2015: "reaffirmed its view that the current 0 to 1/4 percent target range for the federal
    # funds rate remains appropriate"
    r"|\breaffirm(?:s|ed)?\s+its\s+view\s+that\s+the\s+current\b[^.;]*?\btarget\s+range\b"
    r"[^.;]*?\bremains\s+appropriate\b",
    re.IGNORECASE,
)
# Rate guidance (step 2) = a rate-path topic plus a cue from the rubric's step-2 wording, in a
# sentence that is not about the balance sheet. In a rate-action sentence only the text after the
# decision counts ("decided to keep the target range ... and expects it will be appropriate to
# maintain this target range until ..."), so "In view of realized and expected ..." before a
# decision is not guidance.
GUIDANCE_CUE = re.compile(
    r"\b(?:expect(?:s|ed)?|anticipat(?:es|ed)|will|appropriate|additional|adjustments?|firming"
    r"|patient|extent\s+and\s+timing|increases?|accommodative|until|soon|next\s+meeting)\b",
    re.IGNORECASE,
)
RATE_PATH = re.compile(
    r"\b(?:target\s+range|federal\s+funds\s+rate|monetary\s+policy"
    r"|policy\s+(?:accommodation|firming)|adjustments?|act\s+as\s+appropriate)\b",
    re.IGNORECASE,
)
BALANCE_SHEET = re.compile(
    r"\b(?:securities|purchas\w*|holdings|balance\s+sheet|reinvest\w*|runoff|redemptions?"
    r"|reserves|repurchase)\b",
    re.IGNORECASE,
)

RateAction = Literal["move", "hold"]
RubricStep = Literal["rate_action", "guidance", "none"]


def rate_action(sentence: str) -> tuple[RateAction, int] | None:
    """("move" or "hold", where the decision phrase ends) when the sentence states this
    meeting's target-range decision; None otherwise. A move wins over a hold."""
    if match := RATE_MOVE.search(sentence):
        return "move", match.end()
    if match := RATE_HOLD.search(sentence):
        return "hold", match.end()
    return None


def is_rate_guidance(sentence: str) -> bool:
    """Whether the sentence is forward guidance on the rate path (rubric step 2)."""
    if BALANCE_SHEET.search(sentence):
        return False
    action = rate_action(sentence)
    rest = sentence[action[1] :] if action else sentence
    return bool(GUIDANCE_CUE.search(rest) and RATE_PATH.search(rest))


@dataclass(frozen=True)
class RubricDecision:
    """Which rubric step set a statement's stance, with the score and the sentences it used."""

    step: RubricStep  # "none": no step leaned, the stance is neutral
    stance: Stance
    score: float
    sentences: tuple[int, ...]  # indices of the action sentence, or of the guidance sentences


def rubric_decision(
    sentences: Sequence[str], predictions: Sequence[SentencePrediction]
) -> RubricDecision:
    """The rubric's order applied to a sentence classifier's labels.

    1. Rate action: the statement's first rate-action sentence. If it is a move (raise / lower)
       and the classifier labels that sentence hawkish or dovish, that label is the stance and
       that sentence's score the score. A hold, a move labelled neutral, or no action sentence
       goes on.
    2. Rate guidance: the mean score of the guidance sentences, cut at the fixed +-THRESHOLD.
    3. Otherwise neutral, with that mean (or 0 without guidance sentences) as the score.

    The rubric's balance-sheet and risk-language steps (3 and 4) are left out on purpose: the
    classifiers learned sentence tone from Trillion Dollar Words, and FinBERT in particular is
    unreliable on balance-sheet sentences (it reads "reducing ... holdings" as dovish and
    "increase its holdings" as hawkish, the reverse of the rubric), so those steps would add
    noise rather than the rubric's judgment. A purchases-only statement (March 2020) therefore
    comes out neutral here.

    is_rate_guidance also counts outlook sentences that mention policy adjustments ("expects
    that, with gradual adjustments in the stance of monetary policy, economic activity will
    expand ...") and the "prepared to adjust the stance of monetary policy ... if risks emerge"
    sentence. Leaving the outlook sentences out would move the post-hoc accuracy by 0.07 to 0.09
    (keyword 0.825 to 0.732, FinBERT 0.691 to 0.619): one more choice in a post-hoc variant.
    """
    if len(sentences) != len(predictions):
        raise ValueError(f"{len(predictions)} predictions for {len(sentences)} sentences")
    scores = [sentence_score(p) for p in predictions]
    actions = [(i, found[0]) for i, s in enumerate(sentences) if (found := rate_action(s))]
    if actions and actions[0][1] == "move":
        i = actions[0][0]
        if predictions[i].stance != "neutral":
            return RubricDecision("rate_action", predictions[i].stance, scores[i], (i,))
    guidance = tuple(i for i, sentence in enumerate(sentences) if is_rate_guidance(sentence))
    score = statistics.fmean(scores[i] for i in guidance) if guidance else 0.0
    stance = stance_from_score(score)
    return RubricDecision("guidance" if stance != "neutral" else "none", stance, score, guidance)


def rate_move_rule(sentences: Sequence[str]) -> Stance:
    """A post-hoc control that uses no classifier: the statement's first rate-action sentence
    decides by its verb when it is a move (raise or increase hawkish, lower, reduce or cut
    dovish); a hold or no rate-action sentence is neutral. It scores what the rubric-ordered
    variant gets from detecting the move alone."""
    for sentence in sentences:
        if rate_action(sentence):
            match = RATE_MOVE.search(sentence)
            return MOVE_DIRECTIONS[match.group("verb").lower()] if match else "neutral"
    return "neutral"


def aggregate_rubric_ordered(
    statement: Statement,
    sentences: Sequence[str],
    predictions: Sequence[SentencePrediction],
    classifier: SentenceClassifier,
    model_version: str,
) -> StatementPrediction:
    """One statement's prediction by rubric_decision (post hoc). confidence is |score| clipped
    to [0, 1], as in aggregate; evidence and key factors come from the sentences the deciding
    step used."""
    decision = rubric_decision(sentences, predictions)
    used = [sentences[i] for i in decision.sentences]
    scores = [sentence_score(predictions[i]) for i in decision.sentences]
    return StatementPrediction(
        doc_id=statement.doc_id,
        meeting_date=statement.meeting_date,
        classifier=classifier.name,
        stance=decision.stance,
        score=max(-1.0, min(1.0, decision.score)),
        confidence=min(1.0, abs(decision.score)),
        key_factors=_key_factors(classifier, used, decision.stance),
        evidence=_evidence(used, scores, decision.stance),
        model_version=model_version,
    )


Aggregator = Callable[
    [Statement, Sequence[str], Sequence[SentencePrediction], SentenceClassifier, str],
    StatementPrediction,
]
AGGREGATORS: dict[StatementVariant, Aggregator] = {
    "pre_registered": aggregate,
    "post_hoc_rubric_ordered": aggregate_rubric_ordered,
}


@dataclass(frozen=True)
class LabelledStatement:
    """A statement's substantive sentences with a sentence classifier's predictions."""

    statement: Statement
    sentences: tuple[str, ...]
    predictions: tuple[SentencePrediction, ...]


def label_statements(
    classifier: SentenceClassifier, statements: Sequence[Statement]
) -> list[LabelledStatement]:
    """Every statement's sentences labelled in one classifier call."""
    per_statement = [statement.sentences() for statement in statements]
    flat = [sentence for sentences in per_statement for sentence in sentences]
    predictions = classifier.predict_sentences(flat)
    if len(predictions) != len(flat):
        raise ValueError(f"{len(predictions)} predictions for {len(flat)} sentences")
    results, offset = [], 0
    for statement, sentences in zip(statements, per_statement, strict=True):
        chunk = predictions[offset : offset + len(sentences)]
        offset += len(sentences)
        results.append(LabelledStatement(statement, tuple(sentences), tuple(chunk)))
    return results


def aggregate_statements(
    classifier: SentenceClassifier,
    labelled: Sequence[LabelledStatement],
    *,
    variant: StatementVariant = "pre_registered",
    model_version: str = "",
) -> list[StatementPrediction]:
    """Statement predictions from labelled sentences with the variant's aggregation.
    model_version defaults to the classifier's version plus the variant's AGGREGATIONS entry."""
    version = model_version or (
        f"{getattr(classifier, 'version', classifier.name)}; {AGGREGATIONS[variant]}"
    )
    aggregator = AGGREGATORS[variant]
    return [
        aggregator(item.statement, item.sentences, item.predictions, classifier, version)
        for item in labelled
    ]


def predict_statements(
    classifier: SentenceClassifier,
    statements: Sequence[Statement],
    *,
    model_version: str = "",
    variant: StatementVariant = "pre_registered",
) -> list[StatementPrediction]:
    """Label every statement's sentences in one classifier call, then aggregate per statement
    (pre-registered averaging unless `variant` says otherwise)."""
    labelled = label_statements(classifier, statements)
    return aggregate_statements(classifier, labelled, variant=variant, model_version=model_version)


def prediction_to_json(prediction: StatementPrediction) -> dict[str, Any]:
    return {
        **asdict(prediction),
        "meeting_date": prediction.meeting_date.isoformat(),
        "score": round(prediction.score, 6),
        "confidence": None if prediction.confidence is None else round(prediction.confidence, 6),
        "key_factors": list(prediction.key_factors),
        "evidence": list(prediction.evidence),
    }


def prediction_from_json(row: dict[str, Any]) -> StatementPrediction:
    return StatementPrediction(
        **{
            **row,
            "meeting_date": date.fromisoformat(row["meeting_date"]),
            "key_factors": tuple(row["key_factors"]),
            "evidence": tuple(row["evidence"]),
        }
    )


def predictions_path(
    settings: Settings | None = None, variant: StatementVariant = "pre_registered"
) -> Path:
    return (settings or get_settings()).data_dir / "stance" / PREDICTIONS_FILES[variant]


def write_predictions(path: Path, predictions: Sequence[StatementPrediction]) -> int:
    return write_jsonl(path, (prediction_to_json(p) for p in predictions))


def read_predictions(path: Path) -> list[StatementPrediction]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [prediction_from_json(json.loads(line)) for line in lines if line.strip()]
