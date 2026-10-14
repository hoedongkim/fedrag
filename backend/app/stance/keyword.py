"""Keyword baseline: hawkish and dovish term lists learned from a TDW train split, with negation.

The lexicon is chosen from the train split only; nothing here reads a test split.

1. Tokens: a sentence is lower-cased and cut into words; punctuation ends a clause. A word within
   NEGATION_WINDOW words after a negator (not, no, never, without, nor, cannot, ...n't) in the
   same clause is negated.
2. Candidates: unigrams and bigrams (adjacent words of one clause) without stopwords or negated
   words that occur in at least MIN_SUPPORT train sentences.
3. Ranking: for hawkish and for dovish, each candidate's one-vs-rest log odds ratio (its share of
   the class's term occurrences against its share of the other sentences', add-0.5 smoothed)
   divided by its standard error: a Wald z-score. Terms with z > Z_MIN are ranked by z; a term
   that qualifies for both sides keeps the side with the larger z.
4. Size: top_k terms per side, with top_k picked from TOP_K_GRID by 5-fold stratified
   cross-validation on the train split, folds grouped by sentence text (best mean macro-F1; ties go
   to the smaller lexicon). The final lists are then ranked on the whole train split.

A sentence's score is (hawkish matches) - (dovish matches); a negated match counts for the other
side, and bigrams are matched before unigrams so their words do not count twice. Positive is
hawkish, negative dovish, zero neutral. Several lexicons (one per seed) vote: the majority label
wins and a three-way split is neutral.
"""

import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from sklearn.model_selection import StratifiedGroupKFold

from app.core.config import Settings
from app.schemas.common import Stance, StanceClassifier
from app.stance import tdw
from app.stance.metrics import classification_metrics
from app.stance.types import SentencePrediction

LEXICON_VERSION = 1
NEGATION_WINDOW = 3
MIN_SUPPORT = 5
Z_MIN = 1.96
TOP_K_GRID = (5, 10, 15, 20, 30, 40, 80, 160)
CV_FOLDS = 5
CV_SEED = 0

NEGATORS = frozenset({"not", "no", "never", "without", "nor", "neither", "cannot", "none"})
STOPWORDS = frozenset(
    str.split(
        """
        a about above after again against all also am an and any are as at be because been
        before being below between both but by can could did do does doing down during each few
        for from further had has have having he her here hers herself him himself his how i if
        in into is it its itself just me more most my myself of off on once only or other our
        ours ourselves out over own same she should so some such than that the their theirs them
        themselves then there these they this those through to too under until up very was we
        were what when where which while who whom why will with would you your yours yourself
        yourselves s t us one two well may might must shall whether yet upon per via within
        among
        """
    )
)

_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?|[.,;:!?()\[\]\"]")

Clause = list[tuple[str, bool]]  # (word, negated)


def clauses(sentence: str) -> list[Clause]:
    """The sentence's clauses as (word, negated) pairs; negators themselves are left out."""
    text = sentence.lower().replace("\u2019", "'").replace("\u2018", "'").replace("u.s.", "us")
    result: list[Clause] = []
    current: Clause = []
    since_negator: int | None = None
    for token in _TOKEN.findall(text):
        if not token[0].isalpha():
            if current:
                result.append(current)
            current, since_negator = [], None
            continue
        word = token.removesuffix("'s")
        if word in NEGATORS or word.endswith("n't"):
            since_negator = 0
            continue
        negated = since_negator is not None and since_negator < NEGATION_WINDOW
        if since_negator is not None:
            since_negator += 1
        current.append((word, negated))
    if current:
        result.append(current)
    return result


def candidate_terms(sentence: str) -> list[str]:
    """Unigrams and bigrams that may enter a lexicon: no stopword and no negated word."""
    terms: list[str] = []
    for clause in clauses(sentence):
        usable = [not negated and word not in STOPWORDS for word, negated in clause]
        terms += [word for (word, _), ok in zip(clause, usable, strict=True) if ok]
        terms += [
            f"{clause[i][0]} {clause[i + 1][0]}"
            for i in range(len(clause) - 1)
            if usable[i] and usable[i + 1]
        ]
    return terms


def _z_scores(in_class: Counter[str], rest: Counter[str], terms: Iterable[str]) -> dict[str, float]:
    """Wald z of each term's log odds ratio, class against rest, add-0.5 smoothed."""
    n_class, n_rest = sum(in_class.values()), sum(rest.values())
    scores: dict[str, float] = {}
    for term in terms:
        a, b = in_class[term] + 0.5, n_class - in_class[term] + 0.5
        c, d = rest[term] + 0.5, n_rest - rest[term] + 0.5
        scores[term] = (math.log(a / b) - math.log(c / d)) / math.sqrt(
            1 / a + 1 / b + 1 / c + 1 / d
        )
    return scores


def rank_terms(
    sentences: Sequence[str], labels: Sequence[Stance]
) -> tuple[list[str], list[str], dict[str, float]]:
    """Hawkish and dovish terms with z > Z_MIN, most discriminative first, and each selected
    term's z signed by side (positive hawkish, negative dovish)."""
    counts: dict[Stance, Counter[str]] = {s: Counter() for s in ("hawkish", "dovish", "neutral")}
    support: Counter[str] = Counter()
    for sentence, label in zip(sentences, labels, strict=True):
        terms = candidate_terms(sentence)
        counts[label].update(terms)
        support.update(set(terms))
    eligible = [term for term, n in support.items() if n >= MIN_SUPPORT]
    z: dict[Stance, dict[str, float]] = {}
    for side in ("hawkish", "dovish"):
        rest = sum((c for s, c in counts.items() if s != side), Counter())
        z[side] = _z_scores(counts[side], rest, eligible)
    hawkish = [t for t in eligible if z["hawkish"][t] > Z_MIN and z["hawkish"][t] >= z["dovish"][t]]
    dovish = [t for t in eligible if z["dovish"][t] > Z_MIN and z["dovish"][t] > z["hawkish"][t]]
    hawkish.sort(key=lambda t: (-z["hawkish"][t], t))
    dovish.sort(key=lambda t: (-z["dovish"][t], t))
    signed = {t: z["hawkish"][t] for t in hawkish} | {t: -z["dovish"][t] for t in dovish}
    return hawkish, dovish, signed


@dataclass(frozen=True)
class Lexicon:
    """Term lists (most discriminative first; a bigram is two words joined by a space) and how
    their size was chosen."""

    hawkish: tuple[str, ...]
    dovish: tuple[str, ...]
    top_k: int
    cv_macro_f1: dict[int, float] = field(default_factory=dict)  # mean CV macro-F1 per top_k
    train_size: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "version": LEXICON_VERSION,
            "top_k": self.top_k,
            "cv_macro_f1": {str(k): round(v, 4) for k, v in self.cv_macro_f1.items()},
            "train_size": self.train_size,
            "hawkish": list(self.hawkish),
            "dovish": list(self.dovish),
        }


class KeywordClassifier:
    """Labels sentences by counting lexicon terms; with several lexicons, they vote."""

    name: StanceClassifier = "keyword"

    def __init__(self, lexicons: Lexicon | Sequence[Lexicon]) -> None:
        self.lexicons = (lexicons,) if isinstance(lexicons, Lexicon) else tuple(lexicons)
        if not self.lexicons:
            raise ValueError("at least one lexicon is needed")
        self._polarity = [
            {t: 1 for t in lex.hawkish} | {t: -1 for t in lex.dovish} for lex in self.lexicons
        ]

    @property
    def version(self) -> str:
        sizes = "/".join(str(lexicon.top_k) for lexicon in self.lexicons)
        return f"keyword-lexicon-v{LEXICON_VERSION} (TDW train; top_k {sizes}; vote)"

    def matches(self, sentence: str) -> list[tuple[str, int]]:
        """(term, +1 hawkish / -1 dovish after negation) for each match of every lexicon."""
        return [m for polarity in self._polarity for m in _match(clauses(sentence), polarity)]

    def label(self, sentence: str) -> Stance:
        parsed = clauses(sentence)
        votes = Counter(
            _stance(sum(sign for _, sign in _match(parsed, polarity)))
            for polarity in self._polarity
        )
        (top, count), *others = votes.most_common()
        return top if not others or count > others[0][1] else "neutral"

    def predict_sentences(self, sentences: Sequence[str]) -> list[SentencePrediction]:
        return [SentencePrediction(self.label(sentence)) for sentence in sentences]


def _match(parsed: Sequence[Clause], polarity: dict[str, int]) -> list[tuple[str, int]]:
    """Greedy longest-first matching of one lexicon in each clause."""
    found: list[tuple[str, int]] = []
    for clause in parsed:
        i = 0
        while i < len(clause):
            word, negated = clause[i]
            bigram = f"{word} {clause[i + 1][0]}" if i + 1 < len(clause) else ""
            term = bigram if bigram in polarity else word if word in polarity else None
            if term is None:
                i += 1
                continue
            found.append((term, -polarity[term] if negated else polarity[term]))
            i += 2 if term == bigram else 1
    return found


def _stance(score: float) -> Stance:
    return "hawkish" if score > 0 else "dovish" if score < 0 else "neutral"


def build_lexicon(
    sentences: Sequence[str], labels: Sequence[Stance], top_k: int, **info: Any
) -> Lexicon:
    hawkish, dovish, _ = rank_terms(sentences, labels)
    return Lexicon(tuple(hawkish[:top_k]), tuple(dovish[:top_k]), top_k, **info)


def fit_keyword(
    train: Sequence[tdw.LabeledSentence], *, grid: Sequence[int] = TOP_K_GRID
) -> KeywordClassifier:
    """Learn a lexicon from train sentences alone, picking top_k by grouped cross-validation."""
    sentences = [s.sentence for s in train]
    labels: list[Stance] = [s.stance for s in train]
    folds = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_SEED)
    scores: dict[int, list[float]] = {k: [] for k in grid}
    for fit_idx, held_idx in folds.split(sentences, labels, groups=sentences):
        hawkish, dovish, _ = rank_terms(
            [sentences[i] for i in fit_idx], [labels[i] for i in fit_idx]
        )
        gold = [labels[i] for i in held_idx]
        for k in grid:
            classifier = KeywordClassifier(Lexicon(tuple(hawkish[:k]), tuple(dovish[:k]), k))
            pred = [classifier.label(sentences[i]) for i in held_idx]
            scores[k].append(classification_metrics(gold, pred)["macro_f1"])
    cv = {k: sum(v) / len(v) for k, v in scores.items()}
    best = max(grid, key=lambda k: (round(cv[k], 6), -k))
    lexicon = build_lexicon(sentences, labels, best, cv_macro_f1=cv, train_size=len(train))
    return KeywordClassifier(lexicon)


def keyword_factory(settings: Settings | None = None) -> Callable[[int], KeywordClassifier]:
    """seed -> a keyword classifier fitted on that seed's TDW train split."""
    return lambda seed: fit_keyword(tdw.load("train", seed, settings))


def ensemble(
    seeds: Sequence[int] = tdw.SEEDS, settings: Settings | None = None
) -> KeywordClassifier:
    """The seed lexicons together (they vote), for statements."""
    return KeywordClassifier(
        [fit_keyword(tdw.load("train", s, settings)).lexicons[0] for s in seeds]
    )
