"""Tests for the keyword baseline: tokens and negation, lexicon learning, matching and voting."""

import pytest

from app.schemas.common import Stance
from app.stance import tdw
from app.stance.keyword import (
    KeywordClassifier,
    Lexicon,
    candidate_terms,
    clauses,
    fit_keyword,
    keyword_factory,
    rank_terms,
)
from app.stance.metrics import evaluate_sentences

HAWKISH = ["tightening", "upward pressure", "rising"]
DOVISH = ["easing", "accommodative", "weak"]
FILLER = ["labor market", "committee", "outlook", "spending", "survey", "business", "housing"]


def toy_train(n: int = 20, offset: int = 0) -> list[tdw.LabeledSentence]:
    """Distinct synthetic sentences: hawkish ones use HAWKISH words, dovish ones DOVISH words."""
    rows: list[tdw.LabeledSentence] = []
    for i in range(n):
        filler = FILLER[i % len(FILLER)]
        k = i + offset
        items: list[tuple[str, Stance]] = [
            (f"The {filler} saw {HAWKISH[i % 3]} prices in district {k}.", "hawkish"),
            (f"The {filler} called for {DOVISH[i % 3]} policy in district {k}.", "dovish"),
            (f"The {filler} discussed the report in district {k}.", "neutral"),
        ]
        rows += [tdw.LabeledSentence(len(rows), text, 2020, stance) for text, stance in items]
    return rows


def test_clauses_mark_words_after_a_negator() -> None:
    assert clauses("Inflation is not expected to rise much further, but wages rose.") == [
        [
            ("inflation", False),
            ("is", False),
            ("expected", True),
            ("to", True),
            ("rise", True),
            ("much", False),
            ("further", False),
        ],
        [("but", False), ("wages", False), ("rose", False)],
    ]


def test_clauses_handle_contractions_and_curly_apostrophes() -> None:
    assert clauses("Prices don\u2019t rise.") == [[("prices", False), ("rise", True)]]
    assert clauses("The Committee's view") == [
        [("the", False), ("committee", False), ("view", False)]
    ]


def test_candidates_skip_stopwords_and_negated_words() -> None:
    terms = candidate_terms("Policy is not tightening; upward pressure on prices remains.")
    assert "tightening" not in terms
    assert {"upward", "pressure", "upward pressure", "prices", "remains"} <= set(terms)
    assert not any(word in terms for word in ("is", "on", "on prices"))


def test_rank_terms_finds_each_sides_words() -> None:
    train = toy_train()
    hawkish, dovish, z = rank_terms([s.sentence for s in train], [s.stance for s in train])
    assert {"tightening", "upward pressure", "rising"} <= set(hawkish)
    assert {"easing", "accommodative", "weak"} <= set(dovish)
    assert not set(hawkish) & set(dovish)
    assert all(z[t] > 0 for t in hawkish) and all(z[t] < 0 for t in dovish)
    assert "labor market" not in hawkish + dovish  # equally common in every class


def test_matching_prefers_bigrams_and_flips_negated_terms() -> None:
    lexicon = Lexicon(("upward pressure", "upward", "tightening"), ("easing", "pressure"), 3)
    classifier = KeywordClassifier(lexicon)
    assert classifier.matches("Upward pressure persists.") == [("upward pressure", 1)]
    assert classifier.matches("No further easing is needed.") == [("easing", 1)]
    assert classifier.matches("Tightening, not easing.") == [("tightening", 1), ("easing", 1)]
    assert classifier.label("Tightening and easing.") == "neutral"
    assert classifier.label("Easing helps, but not tightening.") == "dovish"


def test_lexicons_vote_and_a_three_way_split_is_neutral() -> None:
    hawk, dove, empty = Lexicon(("rise",), (), 1), Lexicon((), ("rise",), 1), Lexicon((), (), 1)
    sentence = "Rates rise."
    assert KeywordClassifier([hawk, hawk, dove]).label(sentence) == "hawkish"
    assert KeywordClassifier([hawk, dove, empty]).label(sentence) == "neutral"
    assert (
        KeywordClassifier([dove, dove, empty]).predict_sentences([sentence])[0].stance == "dovish"
    )
    assert KeywordClassifier(hawk).predict_sentences([sentence])[0].probs == {}


def test_fit_keyword_is_deterministic_and_records_its_choice() -> None:
    train = toy_train()
    first, second = fit_keyword(train, grid=(1, 3)), fit_keyword(train, grid=(1, 3))
    assert first.lexicons == second.lexicons
    lexicon = first.lexicons[0]
    assert lexicon.top_k in (1, 3)
    assert set(lexicon.cv_macro_f1) == {1, 3}
    assert lexicon.train_size == len(train)
    assert lexicon.to_json()["hawkish"] == list(lexicon.hawkish)


def test_lexicon_depends_on_the_train_split_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """The factory reads only train splits, and changing the test split changes nothing."""
    requested: list[tuple[str, int]] = []
    test_variants = {1: toy_train(5, offset=100), 2: toy_train(5, offset=200)}

    def fake_load(
        split: tdw.Split, seed: int, settings: object = None
    ) -> list[tdw.LabeledSentence]:
        requested.append((split, seed))
        return toy_train() if split == "train" else test_variants[seed]

    monkeypatch.setattr(tdw, "load", fake_load)
    factory = keyword_factory()
    lexicons = [factory(seed).lexicons for seed in (1, 2)]
    assert requested == [("train", 1), ("train", 2)]
    assert lexicons[0] == lexicons[1]  # same train split, different test splits

    requested.clear()
    read_while_fitting: list[tuple[str, int]] = []

    def watched(seed: int) -> KeywordClassifier:
        classifier = factory(seed)
        read_while_fitting.extend(requested)
        return classifier

    evaluate_sentences(watched, (1,))
    assert read_while_fitting == [("train", 1)]
    assert ("test", 1) in requested  # read by the evaluator once the lexicon is fixed
