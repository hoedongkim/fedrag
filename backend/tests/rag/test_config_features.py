"""Tests for the retrieval improvements behind RetrieverConfig: the Porter stemmer, the FOMC
synonym map, date mentions, the current-state trigger and the metadata prior."""

import hashlib
from datetime import date, timedelta

import pytest

from app.rag.bm25 import (
    SYNONYM_GROUPS,
    BM25Retriever,
    expansion_terms,
    porter_stem,
    tokenize,
)
from app.rag.filters import DateMention, MetadataPrior, asks_current_state, parse_date_mentions
from app.rag.types import Chunk
from app.schemas.common import DocType

# Final Porter stems of the examples in Porter (1980), "An algorithm for suffix stripping".
PORTER_EXAMPLES = {
    "caresses": "caress", "ponies": "poni", "ties": "ti", "caress": "caress", "cats": "cat",
    "feed": "feed", "agreed": "agre", "plastered": "plaster", "bled": "bled", "motoring": "motor",
    "sing": "sing", "conflated": "conflat", "troubled": "troubl", "sized": "size",
    "hopping": "hop", "tanned": "tan", "falling": "fall", "hissing": "hiss", "fizzed": "fizz",
    "failing": "fail", "filing": "file", "happy": "happi", "sky": "sky", "relational": "relat",
    "conditional": "condit", "rational": "ration", "valenci": "valenc", "hesitanci": "hesit",
    "digitizer": "digit", "conformabli": "conform", "radicalli": "radic",
    "differentli": "differ", "vileli": "vile", "analogousli": "analog",
    "vietnamization": "vietnam", "predication": "predic", "operator": "oper",
    "feudalism": "feudal", "decisiveness": "decis", "hopefulness": "hope",
    "callousness": "callous", "formaliti": "formal", "sensitiviti": "sensit",
    "sensibiliti": "sensibl", "triplicate": "triplic", "formative": "form", "formalize": "formal",
    "electriciti": "electr", "electrical": "electr", "hopeful": "hope", "goodness": "good",
    "revival": "reviv", "allowance": "allow", "inference": "infer", "airliner": "airlin",
    "gyroscopic": "gyroscop", "adjustable": "adjust", "defensible": "defens",
    "irritant": "irrit", "replacement": "replac", "adjustment": "adjust", "dependent": "depend",
    "adoption": "adopt", "homologou": "homolog", "communism": "commun", "activate": "activ",
    "angulariti": "angular", "homologous": "homolog", "effective": "effect",
    "bowdlerize": "bowdler", "probate": "probat", "rate": "rate", "cease": "ceas",
    "controll": "control", "roll": "roll", "generalization": "gener", "oscillators": "oscil",
}  # fmt: skip


@pytest.mark.parametrize(("word", "stem"), PORTER_EXAMPLES.items())
def test_porter_stem_matches_the_paper(word: str, stem: str) -> None:
    assert porter_stem(word) == stem


def test_porter_stem_conflates_fomc_inflections() -> None:
    groups = [
        ("raise", "raises", "raised", "raising"),
        ("cut", "cuts", "cutting"),
        ("lower", "lowered", "lowering"),
        ("increase", "increased", "increases", "increasing"),
        ("hike", "hikes", "hiking"),
        ("dissent", "dissents", "dissented", "dissenting"),
        ("policy", "policies"),
        ("rate", "rates"),
    ]
    for words in groups:
        assert len({porter_stem(w) for w in words}) == 1, words
    assert porter_stem("is") == "is" and porter_stem("qt") == "qt"  # 2 letters: unchanged
    assert porter_stem("yyyyyyyyyy" * 50)  # long nonsense neither recurses nor fails


def test_tokenize_with_stemming_keeps_numbers_and_drops_stopwords_first() -> None:
    text = "Raising the target range to 5-1/4 percent in 2023; rates were cut."

    assert tokenize(text, stem=True) == [
        "rais", "target", "rang", "5-1/4", "5.25", "percent", "2023", "rate", "cut",
    ]  # fmt: skip
    assert tokenize(text) == [
        "raising", "target", "range", "5-1/4", "5.25", "percent", "2023", "rates", "cut",
    ]  # fmt: skip


def chunk(
    n: int,
    text: str,
    *,
    meeting: date = date(2024, 9, 18),
    released: date | None = None,
    doc_type: DocType = "statement",
) -> Chunk:
    doc_id = f"{doc_type}-{meeting.isoformat()}"
    return Chunk(
        chunk_id=f"{doc_id}#{n:04d}",
        doc_id=doc_id,
        doc_type=doc_type,
        meeting_date=meeting,
        release_date=released or meeting,
        page_updated=None,
        title=doc_id,
        url=f"https://www.federalreserve.gov/test/{hashlib.sha256(text.encode()).hexdigest()}.htm",
        section=None,
        index=n,
        text=text,
        context="FOMC",
    )


def test_stemmed_bm25_matches_inflections() -> None:
    chunks = [
        chunk(0, "The Committee decided to raise the target range."),
        chunk(1, "Participants discussed raising rates."),
        chunk(2, "Housing activity weakened."),
    ]

    plain = BM25Retriever(chunks).retrieve("raises", top_k=3)
    stemmed = BM25Retriever(chunks, stem=True)
    hits = stemmed.retrieve("raises", top_k=3)

    assert all(hit.score == 0 for hit in plain)
    assert stemmed.stemmed and not BM25Retriever(chunks).stemmed
    assert {h.chunk.index for h in hits if h.score > 0} == {0, 1}


@pytest.mark.parametrize(
    ("question", "added"),
    [
        ("Why did the Fed cut rates?", ["lower", "reduce", "decrease"]),
        ("When did the Committee start hiking rates?", ["raise", "increase", "tighten"]),
        ("What is the policy rate?", ["federal funds rate", "target range"]),
        ("Did QT start in 2022?", ["quantitative tightening", "runoff", "balance sheet reduction"]),
        ("What about jobs?", ["employment", "payrolls", "job gains", "labor market"]),
        ("Who dissented in July 2019?", ["voting against"]),
        ("How did the covid pandemic hit activity?", ["coronavirus"]),
        ("Prices increased sharply.", ["inflation"]),  # no rate word: "increased" is not a hike
        ("How did the vote go?", []),  # "voting against" never triggers the dissent group
        ("What did the minutes say about Brexit?", []),
    ],
)
def test_expansion_terms(question: str, added: list[str]) -> None:
    assert expansion_terms(question) == added


def test_synonym_groups_are_well_formed() -> None:
    for group in SYNONYM_GROUPS:
        stems = [tuple(porter_stem(t) for t in tokenize(term)) for term in group.terms]
        assert all(stems) and len(set(stems)) == len(stems), group  # one form per stem
        assert all(tokenize(term) for term in group.also + group.context), group


def test_query_expansion_adds_weighted_terms_bm25_only() -> None:
    chunks = [
        chunk(0, "The Committee decided to lower the target range for the federal funds rate."),
        chunk(1, "The Committee decided to maintain the target range."),
        chunk(2, "Participants expected a further decrease in inflation."),
        chunk(3, "Housing activity weakened."),
    ]
    plain = BM25Retriever(chunks)
    expanded = plain.with_query_expansion(0.25)

    tokens, extra = expanded.query_terms("rate cut")
    by_query = {h.chunk.chunk_id: h.score for h in plain.retrieve("rate cut", top_k=4)}
    by_extra = {h.chunk.chunk_id: h.score for h in plain.retrieve(" ".join(extra), top_k=4)}
    hits = expanded.retrieve("rate cut", top_k=4)

    assert (tokens, extra) == (["rate", "cut"], ["lower", "reduce", "decrease"])
    assert plain.query_terms("rate cut") == (["rate", "cut"], [])  # the original is unchanged
    assert plain.expansion_weight is None and expanded.expansion_weight == 0.25
    for hit in hits:
        expected = by_query[hit.chunk.chunk_id] + 0.25 * by_extra[hit.chunk.chunk_id]
        assert hit.score == pytest.approx(expected)
    assert hits[0].chunk.index == 0  # "lower the target range" now answers "rate cut"
    with pytest.raises(ValueError, match="expansion weight"):
        plain.with_query_expansion(0)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("cut rates in September 2024", [DateMention(2024, 9)]),
        ("Sept. 2024", [DateMention(2024, 9)]),
        ("the Sep 2024 meeting", [DateMention(2024, 9)]),
        ("September of 2024", [DateMention(2024, 9)]),
        ("the March 2020 meeting", [DateMention(2020, 3)]),
        ("in 2022", [DateMention(2022)]),
        ("from 2022-2023 to mid-2024", [DateMention(2022), DateMention(2023), DateMention(2024)]),
        (
            "Sept. 17–18, 2024",
            [DateMention(2024, 9, date(2024, 9, 17), date(2024, 9, 18))],
        ),
        ("December 16th, 2015", [DateMention(2015, 12, date(2015, 12, 16), date(2015, 12, 16))]),
        ("on 2024-09-18", [DateMention(2024, 9, date(2024, 9, 18), date(2024, 9, 18))]),
        ("on 9/18/2024", [DateMention(2024, 9, date(2024, 9, 18), date(2024, 9, 18))]),
        ("February 30, 2024", [DateMention(2024, 2)]),  # no such day: the month
        ("in March 2022 and later in 2022", [DateMention(2022, 3), DateMention(2022)]),
        ("2024 and 2024", [DateMention(2024)]),
        ("the September meeting", []),  # a month needs a year
        ("the SEP 2024 median", [DateMention(2024)]),  # the projections, not September
        ("SEP Sep 2024", [DateMention(2024, 9)]),
        ("5-1/4 percent, 2.5%, 1,000 jobs, 2015.5 and the 2010s", []),
    ],
)
def test_parse_date_mentions(text: str, expected: list[DateMention]) -> None:
    assert list(parse_date_mentions(text)) == expected


def test_date_mention_matches_meetings() -> None:
    september = DateMention(2024, 9)
    april = DateMention(2024, 4)
    day = DateMention(2024, 9, date(2024, 9, 17), date(2024, 9, 17))

    assert september.matches(date(2024, 9, 18)) and not september.matches(date(2024, 11, 7))
    assert april.matches(date(2024, 5, 1))  # April 30-May 1 began in April
    assert not april.matches(date(2024, 5, 2))
    assert day.matches(date(2024, 9, 18)) and not day.matches(date(2024, 9, 19))
    assert DateMention(2024).matches(date(2024, 12, 18))
    assert not DateMention(2024).matches(date(2025, 1, 29))


@pytest.mark.parametrize(
    "question",
    [
        "Is the Committee still raising rates?",
        "What is the current target range?",
        "Where is the federal funds rate now?",
        "What are the latest projections?",
        "Has the Fed started cutting rates in 2024?",
        "Has the FOMC already begun reducing its holdings?",
        "Have rates been cut yet?",
    ],
)
def test_asks_current_state(question: str) -> None:
    assert asks_current_state(question)


@pytest.mark.parametrize(
    "question",
    [
        "Why did the FOMC raise rates in December 2015?",
        "How did the staff see the current account deficit?",
        "What did the Committee say about known risks?",
    ],
)
def test_does_not_ask_current_state(question: str) -> None:
    assert not asks_current_state(question)


def test_metadata_prior_factors() -> None:
    sept = chunk(0, "a", meeting=date(2024, 9, 18))
    minutes = chunk(1, "b", meeting=date(2024, 9, 18), doc_type="minutes")
    july = chunk(2, "c", meeting=date(2024, 7, 31))
    old = chunk(3, "d", meeting=date(2019, 7, 31))
    prior = MetadataPrior(date_boost=3.0, year_boost=1.5, statement_boost=2.0)

    factor = prior.for_query("Did they cut in September 2024 or later in 2019?", None)

    assert factor is not None
    assert factor(sept) == 6.0  # the month named, and a statement
    assert factor(minutes) == 3.0
    assert factor(july) == 2.0  # neither the month nor a year named alone
    assert factor(old) == 3.0  # 2019, named without a month, and a statement
    assert MetadataPrior().neutral and MetadataPrior().for_query("in 2024", None) is None
    assert MetadataPrior(date_boost=2).for_query("what about rates?", None) is None
    assert MetadataPrior(year_boost=2).for_query("in September 2024", None) is None


def test_recency_prior_needs_as_of_and_a_current_state_question() -> None:
    as_of = date(2024, 12, 31)
    prior = MetadataPrior(recency_boost=1.0, recency_half_life_days=100)
    fresh = chunk(0, "a", meeting=as_of, released=as_of)
    half = chunk(1, "b", meeting=as_of - timedelta(days=100))
    stale = chunk(2, "c", meeting=as_of - timedelta(days=2000))

    factor = prior.for_query("Is the Committee still lowering rates?", as_of)

    assert factor is not None
    assert factor(fresh) == 2.0 and factor(half) == 1.5
    assert 1.0 < factor(stale) < 1.0001
    assert prior.for_query("Is the Committee still lowering rates?", None) is None
    assert prior.for_query("Why did the Committee lower rates?", as_of) is None
