"""BM25 keyword retrieval over chunks, with a tokenizer that keeps policy rates and years whole.

Two options for the retriever configurations (app/rag/config.py), both off in the Phase 3
default: Porter stemming of corpus and query words, and a small FOMC synonym map that adds
weighted terms to a query ("cut rates" also searches "lower", "reduce" and "decrease").
"""

import copy
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi

from app.rag.filters import is_eligible
from app.rag.types import Chunk, RetrievedChunk
from app.schemas.common import DocType

# Numbers first, so "5-1/4", "1/2", "2.5", "1,000" and "2024" each stay one token; then words.
_TOKEN = re.compile(r"\d+(?:[.,]\d+)*(?:-\d+)?(?:/\d+)?|[^\W\d_]+")
_FRACTION = re.compile(r"(?:(\d+)-)?([1-7])/([248])")
# Hyphen, non-breaking hyphen, figure dash, en dash and minus read as "-"; the em dash does not.
_DASHES = str.maketrans(dict.fromkeys("‐‑‒–−", "-"))
_STOPWORD_LIST = """
a about above after again against all also am an and any are as at be been before being below
between both but by can could did do does doing down during each few for from further had has
have having he her here hers him his how i if in into is it its itself just me more most my no
nor not of off on once only or other our ours out over own s same she should so some such t than
that the their theirs them then there these they this those through to too under until up very
was we were what when where which while who whom why will with would you your
"""
STOPWORDS = frozenset(_STOPWORD_LIST.split())
DEFAULT_EXPANSION_WEIGHT = 0.5


def tokenize(text: str, *, stem: bool = False) -> list[str]:
    """Lower-cased word and number tokens without stopwords; with stem, words are Porter stems.

    Fed rates are written as fractions ("5-1/4 to 5-1/2 percent"), so a fraction also yields its
    decimal ("5.25"), which lets a query for "5.25 percent" match the statement that set it.
    Stopwords are dropped before stemming, and numbers are never stemmed.
    """
    tokens: list[str] = []
    for token in _TOKEN.findall(text.translate(_DASHES).lower()):
        if token in STOPWORDS:
            continue
        tokens.append(porter_stem(token) if stem and not token[0].isdigit() else token)
        if fraction := _FRACTION.fullmatch(token):
            whole, numerator, denominator = fraction.groups()
            tokens.append(f"{int(whole or 0) + int(numerator) / int(denominator):g}")
    return tokens


# Porter stemmer (Porter, 1980, "An algorithm for suffix stripping"), as in his reference
# implementation (which also maps -bli to -ble and -logi to -log). Written out here because no
# stemming package is installed; tests/rag/test_config_features.py checks it on the paper's
# examples.
_VOWELS = frozenset("aeiou")
_STEP2 = (
    ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"), ("izer", "ize"),
    ("bli", "ble"), ("alli", "al"), ("entli", "ent"), ("eli", "e"), ("ousli", "ous"),
    ("ization", "ize"), ("ation", "ate"), ("ator", "ate"), ("alism", "al"), ("iveness", "ive"),
    ("fulness", "ful"), ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"),
    ("logi", "log"),
)  # fmt: skip
_STEP3 = (
    ("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"), ("ical", "ic"), ("ful", ""),
    ("ness", ""),
)  # fmt: skip
# Longest first: a word takes the longest suffix it ends with, and keeps it if the stem is short.
_STEP4 = (
    "ement", "ance", "ence", "able", "ible", "ment", "ant", "ent", "ion", "ism", "ate", "iti",
    "ous", "ive", "ize", "al", "er", "ic", "ou",
)  # fmt: skip


@lru_cache(maxsize=1 << 16)
def porter_stem(word: str) -> str:
    """The Porter stem of a lower-case word: raising, raised and raises all become "rais".

    Words of one or two letters are returned unchanged. Stems need not be words ("inflat").
    """
    if len(word) <= 2:
        return word
    word = _step1(word)
    word = _replace_suffix(word, _STEP2)
    word = _replace_suffix(word, _STEP3)
    word = _step4(word)
    return _step5(word)


def _consonants(word: str) -> list[bool]:
    """Per letter, whether it is a consonant: not a, e, i, o or u, and not a y after one."""
    mask: list[bool] = []
    for i, char in enumerate(word):
        mask.append(char not in _VOWELS and (char != "y" or i == 0 or not mask[i - 1]))
    return mask


def _measure(stem: str) -> int:
    """Porter's m: the number of vowel-consonant sequences in the form [C](VC)^m[V]."""
    mask = _consonants(stem)
    return sum(mask[i] and not mask[i - 1] for i in range(1, len(mask)))


def _has_vowel(stem: str) -> bool:
    return not all(_consonants(stem))


def _double_consonant(stem: str) -> bool:
    return len(stem) >= 2 and stem[-1] == stem[-2] and _consonants(stem)[-1]


def _cvc(stem: str) -> bool:
    """Ends consonant-vowel-consonant, the last not w, x or y (hop, but not snow, box or tray)."""
    if len(stem) < 3 or stem[-1] in "wxy":
        return False
    mask = _consonants(stem)
    return mask[-3] and not mask[-2] and mask[-1]


def _step1(word: str) -> str:
    """Plurals, -ed and -ing, and a final y after a vowel-bearing stem."""
    if word.endswith(("sses", "ies")):
        word = word[:-2]
    elif word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    if word.endswith("eed"):
        if _measure(word[:-3]) > 0:
            word = word[:-1]
    else:
        for suffix in ("ed", "ing"):
            if word.endswith(suffix) and _has_vowel(word[: -len(suffix)]):
                word = word[: -len(suffix)]
                if word.endswith(("at", "bl", "iz")):
                    word += "e"
                elif _double_consonant(word) and word[-1] not in "lsz":
                    word = word[:-1]
                elif _measure(word) == 1 and _cvc(word):
                    word += "e"
                break
    if word.endswith("y") and _has_vowel(word[:-1]):
        word = word[:-1] + "i"
    return word


def _replace_suffix(word: str, rules: Sequence[tuple[str, str]]) -> str:
    """Steps 2 and 3: the first rule whose suffix the word ends with applies if m(stem) > 0."""
    for suffix, replacement in rules:
        if word.endswith(suffix):
            stem = word[: -len(suffix)]
            return stem + replacement if _measure(stem) > 0 else word
    return word


def _step4(word: str) -> str:
    for suffix in _STEP4:
        if word.endswith(suffix):
            stem = word[: -len(suffix)]
            if _measure(stem) > 1 and (suffix != "ion" or stem.endswith(("s", "t"))):
                return stem
            return word
    return word


def _step5(word: str) -> str:
    """A final e after a long enough stem, and the second l of a final ll."""
    if word.endswith("e"):
        measure = _measure(word[:-1])
        if measure > 1 or (measure == 1 and not _cvc(word[:-1])):
            word = word[:-1]
    if word.endswith("ll") and _measure(word) > 1:
        word = word[:-1]
    return word


@dataclass(frozen=True)
class SynonymGroup:
    """Words and phrases a question may use for the same thing; when a question contains one
    (compared as Porter stems without stopwords, so "cutting" matches "cut"), the others and
    `also` are added to it. With context, only if the question also contains a context word.
    Terms in `also` are added but never trigger the group ("voting against" is just "vote")."""

    terms: tuple[str, ...]
    context: tuple[str, ...] = ()
    also: tuple[str, ...] = ()


# Rate-action words are ambiguous on their own ("prices increased"), so their groups expand only
# in a question about the policy rate.
_RATE_CONTEXT = ("rate", "rates", "range", "funds", "policy", "basis points")
SYNONYM_GROUPS: tuple[SynonymGroup, ...] = (
    # The statements "lower" and "raise" the target range; questions "cut" and "hike".
    SynonymGroup(("cut", "lower", "reduce", "decrease"), _RATE_CONTEXT),
    SynonymGroup(("hike", "raise", "increase", "tighten"), _RATE_CONTEXT),
    SynonymGroup(("pause", "maintain", "unchanged"), _RATE_CONTEXT),
    SynonymGroup(("fed funds", "federal funds")),
    SynonymGroup(("policy rate", "federal funds rate", "target range")),
    # The balance sheet.
    SynonymGroup(("qt", "quantitative tightening", "runoff", "balance sheet reduction")),
    SynonymGroup(("qe", "quantitative easing", "asset purchases", "large-scale asset purchases")),
    SynonymGroup(("taper", "reduce the pace of net asset purchases")),
    # The economy.
    SynonymGroup(("jobs", "employment", "payrolls", "job gains", "labor market")),
    SynonymGroup(("jobless", "unemployment")),
    SynonymGroup(("wages", "wage growth", "compensation", "labor costs")),
    SynonymGroup(("prices", "inflation")),
    SynonymGroup(("gdp", "economic activity", "economic growth")),
    SynonymGroup(("oil", "energy prices")),
    SynonymGroup(("housing", "residential investment")),
    SynonymGroup(("tariffs", "trade policy")),
    SynonymGroup(("covid", "pandemic", "coronavirus")),
    SynonymGroup(("banking stress", "bank failures", "banking sector", "banking system")),
    # Votes and meetings.
    SynonymGroup(("dissent",), also=("voting against",)),
    SynonymGroup(("emergency", "unscheduled")),
)


@lru_cache(maxsize=1024)
def _stems(phrase: str) -> tuple[str, ...]:
    return tuple(porter_stem(token) for token in tokenize(phrase))


def _contains(stems: Sequence[str], phrase: str) -> bool:
    """True if the phrase's stems occur in `stems` as a contiguous run."""
    target = _stems(phrase)
    n = len(target)
    return n > 0 and any(tuple(stems[i : i + n]) == target for i in range(len(stems) - n + 1))


def expansion_terms(query: str, groups: Sequence[SynonymGroup] = SYNONYM_GROUPS) -> list[str]:
    """The synonym-map terms to add to `query`: for each group it uses a term of, the group's
    other terms, in map order."""
    stems = _stems(query)
    terms: list[str] = []
    for group in groups:
        if group.context and not any(_contains(stems, word) for word in group.context):
            continue
        present = [term for term in group.terms if _contains(stems, term)]
        if present:
            added = [term for term in group.terms if term not in present] + list(group.also)
            terms += [term for term in added if term not in terms]
    return terms


class BM25Retriever:
    """Okapi BM25 over Chunk.search_text, scored only over the chunks that pass the filters.

    Custom rather than LangChain's BM25Retriever (langchain-community), which ranks the whole
    corpus and cannot restrict scoring to chunks released by as_of. Scores are raw BM25 (higher is
    better, 0 when no query term matches); IDF comes from the whole corpus, a corpus statistic that
    never lets an ineligible chunk through. With stem, corpus and query words are Porter stems.
    """

    def __init__(
        self, chunks: Sequence[Chunk], *, k1: float = 1.5, b: float = 0.75, stem: bool = False
    ) -> None:
        self._chunks = list(chunks)
        self._stem = stem
        self._expansion_weight: float | None = None
        self._bm25 = (
            BM25Okapi([tokenize(c.search_text, stem=stem) for c in self._chunks], k1=k1, b=b)
            if self._chunks
            else None
        )

    @property
    def stemmed(self) -> bool:
        return self._stem

    @property
    def expansion_weight(self) -> float | None:
        """Weight of synonym-map terms relative to the query's own (None: no expansion)."""
        return self._expansion_weight

    def with_query_expansion(self, weight: float = DEFAULT_EXPANSION_WEIGHT) -> "BM25Retriever":
        """A retriever over this same index that adds synonym-map terms to every query, each
        scored at `weight` times a query term's BM25 contribution (0 < weight <= 1)."""
        if not 0 < weight <= 1:
            raise ValueError(f"expansion weight must be in (0, 1], not {weight}")
        view = copy.copy(self)  # shares the chunks and the BM25 statistics
        view._expansion_weight = weight
        return view

    def query_terms(self, query: str) -> tuple[list[str], list[str]]:
        """The query's own tokens, and the expansion tokens it does not already contain."""
        tokens = tokenize(query, stem=self._stem)
        if self._expansion_weight is None:
            return tokens, []
        extra: list[str] = []
        for term in expansion_terms(query):
            extra += [t for t in tokenize(term, stem=self._stem) if t not in tokens + extra]
        return tokens, extra

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        as_of: date | None = None,
        doc_types: Sequence[DocType] | None = None,
    ) -> list[RetrievedChunk]:
        eligible = [
            i for i, chunk in enumerate(self._chunks) if is_eligible(chunk, as_of, doc_types)
        ]
        if self._bm25 is None or not eligible or top_k <= 0:
            return []
        tokens, extra = self.query_terms(query)
        scores = np.asarray(self._bm25.get_batch_scores(tokens, eligible))
        if extra and self._expansion_weight is not None:
            scores = scores + self._expansion_weight * np.asarray(
                self._bm25.get_batch_scores(extra, eligible)
            )
        best = np.argsort(-scores, kind="stable")[:top_k]  # ties keep corpus (release) order
        return [RetrievedChunk(self._chunks[eligible[i]], float(scores[i])) for i in best]
