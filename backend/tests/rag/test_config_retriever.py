"""Tests for RetrieverConfig, its presets, and get_retriever under a config: the default is the
Phase 3 retriever, and each improvement changes what it should and nothing else."""

import hashlib
import json
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from pydantic import ValidationError

from app.core.config import Settings
from app.db.load import DocumentRow, ParagraphRow
from app.rag.bm25 import BM25Retriever, tokenize
from app.rag.config import (
    PHASE3_CONFIG,
    PHASE3_PRESET,
    PRESETS_FILE,
    RetrieverConfig,
    get_preset,
    load_presets,
    production_config,
)
from app.rag.filters import MetadataPrior
from app.rag.index import build_index, load_chunks
from app.rag.retrievers import (
    MIN_CANDIDATES,
    RRF_K,
    CrossEncoderScorer,
    DenseRetriever,
    HybridRetriever,
    PriorRetriever,
    RerankRetriever,
    boosted,
    candidate_count,
    clear_retriever_cache,
    get_retriever,
    preload,
    reciprocal_rank_fusion,
)
from app.rag.types import Chunk, RetrievedChunk
from app.schemas.common import RetrievalMethod

METHODS: list[RetrievalMethod] = ["bm25", "dense", "hybrid", "hybrid_rerank"]
TEXTS = {
    date(2015, 12, 16): "The Committee decided to raise the target range to 1/4 to 1/2 percent.",
    date(2019, 7, 31): "The Committee decided to lower the target range to 2 to 2-1/4 percent.",
    date(2020, 3, 15): "The coronavirus outbreak has harmed communities and disrupted activity.",
    date(2023, 7, 26): "The Committee decided to raise the target range to 5-1/4 to 5-1/2 percent.",
    date(2024, 9, 18): "The Committee decided to lower the target range to 4-3/4 to 5 percent.",
}
QUERIES = [
    "Did the Committee lower the target range in September 2024?",
    "Is the Committee still raising rates?",
    "coronavirus outbreak in 2020",
    "5-1/4 percent",
    "cutting rates",
]


def document(doc_type: str, meeting: date, text: str) -> DocumentRow:
    released = meeting if doc_type == "statement" else meeting + timedelta(days=21)
    return DocumentRow(
        doc_id=f"{doc_type}-{meeting.isoformat()}",
        doc_type=doc_type,  # type: ignore[arg-type]
        meeting_date=meeting,
        release_date=released,
        release_date_source="test",
        page_updated=None,
        title=f"{doc_type.title()}, {meeting:%B %d, %Y}",
        url=f"https://www.federalreserve.gov/test/{doc_type}{meeting:%Y%m%d}.htm",
        paragraphs=[ParagraphRow(section=None if doc_type == "statement" else "Policy", text=text)],
        text=text,
        word_count=len(text.split()),
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        parser_version=1,
    )


DOCUMENTS = [document("statement", day, text) for day, text in TEXTS.items()] + [
    document("minutes", day, f"Participants discussed the decision. {text}")
    for day, text in TEXTS.items()
]


def fake_scorer(query: str, passages: Sequence[str]) -> list[float]:
    """Counts query words in each passage: a stand-in for the cross-encoder."""
    words = set(tokenize(query))
    return [float(len(words & set(tokenize(p)))) for p in passages]


def embeddings() -> DeterministicFakeEmbedding:
    return DeterministicFakeEmbedding(size=16)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    # The Phase 3 retriever as get_retriever's default: these tests check its parts and caching;
    # the production preset (tuned) has tests of its own in test_config_retriever.py.
    built = Settings(
        _env_file=None,
        environment="test",
        retriever_preset="phase3",
        openai_api_key=None,
        database_url=None,
        data_dir=tmp_path / "data",
        chroma_dir=tmp_path / "chroma",
        index_dir=tmp_path / "index",
    )
    build_index(built, documents=DOCUMENTS, embeddings=embeddings(), log=lambda _: None)
    clear_retriever_cache()
    return built


@pytest.fixture
def chunks(settings: Settings) -> list[Chunk]:
    return load_chunks(settings)


def ranked(retriever: Any, query: str, **kwargs: Any) -> list[tuple[str, float]]:
    return [(h.chunk.chunk_id, h.score) for h in retriever.retrieve(query, **kwargs)]


# The config itself


def test_default_config_is_phase3_with_every_improvement_off() -> None:
    config = RetrieverConfig(name="anything")

    assert PHASE3_CONFIG.name == PHASE3_PRESET == "phase3"
    assert config.same_behaviour(PHASE3_CONFIG) and config.fingerprint == PHASE3_CONFIG.fingerprint
    assert not PHASE3_CONFIG.bm25_stemming and not PHASE3_CONFIG.query_expansion
    assert PHASE3_CONFIG.prior() is None
    assert PHASE3_CONFIG.rrf_k == RRF_K == 60 and PHASE3_CONFIG.rrf_weights == (1.0, 1.0)
    assert PHASE3_CONFIG.min_candidates == MIN_CANDIDATES == 50
    assert PHASE3_CONFIG.reranker_model is None and PHASE3_CONFIG.reranker_revision is None
    assert "name" not in PHASE3_CONFIG.behaviour and "description" not in PHASE3_CONFIG.behaviour


def test_fingerprint_changes_with_behaviour_only() -> None:
    renamed = PHASE3_CONFIG.model_copy(update={"name": "x", "description": "y"})
    fingerprints = {
        RetrieverConfig(name="a", **change).fingerprint
        for change in (
            {},
            {"bm25_stemming": True},
            {"query_expansion": True},
            {"date_prior": 2.0},
            {"year_prior": 2.0},
            {"recency_prior": 1.0},
            {"statement_boost": 1.2},
            {"rrf_k": 20},
            {"rrf_weights": (1.0, 0.5)},
            {"min_candidates": 100},
            {"reranker_model": "cross-encoder/ms-marco-MiniLM-L-12-v2"},
        )
    }

    assert renamed.fingerprint == PHASE3_CONFIG.fingerprint
    assert len(fingerprints) == 11
    assert len(PHASE3_CONFIG.fingerprint) == 16


@pytest.mark.parametrize(
    "fields",
    [
        {"unknown": 1},
        {"rrf_weights": (-1.0, 1.0)},
        {"rrf_weights": (0.0, 0.0)},
        {"rrf_k": 0},
        {"date_prior": 0},
        {"recency_prior": -1},
        {"expansion_weight": 1.5},
        {"min_candidates": 0},
        {"reranker_revision": "abc123"},  # a revision without a model
    ],
)
def test_invalid_configs_are_refused(fields: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        RetrieverConfig(name="bad", **fields)


def test_config_is_frozen_and_hashable() -> None:
    with pytest.raises(ValidationError):
        PHASE3_CONFIG.bm25_stemming = True  # type: ignore[misc]
    assert {PHASE3_CONFIG: 1}[RetrieverConfig.model_validate(PHASE3_CONFIG.model_dump())] == 1


def test_reranker_follows_settings_unless_set() -> None:
    settings = Settings(_env_file=None)
    other = RetrieverConfig(name="l12", reranker_model="cross-encoder/ms-marco-MiniLM-L-12-v2")

    assert PHASE3_CONFIG.reranker(settings) == (
        settings.reranker_model,
        settings.reranker_revision,
    )
    assert other.reranker(settings) == ("cross-encoder/ms-marco-MiniLM-L-12-v2", None)
    resolved = PHASE3_CONFIG.resolved(settings)
    assert resolved.reranker_model == settings.reranker_model
    assert resolved.fingerprint != PHASE3_CONFIG.fingerprint
    assert other.resolved(settings) is other


# Presets


def test_presets_file_loads_phase3_and_tuned() -> None:
    presets = load_presets()

    assert list(presets)[:2] == ["phase3", "tuned"]
    assert presets["phase3"] is PHASE3_CONFIG and get_preset("phase3") is PHASE3_CONFIG
    assert all(name == config.name for name, config in presets.items())
    assert get_preset("tuned").name == "tuned"
    assert "phase3" not in json.loads(PRESETS_FILE.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="one of phase3, tuned"):
        get_preset("best")


def test_presets_file_format(tmp_path: Path) -> None:
    path = tmp_path / "presets.json"
    path.write_text(
        json.dumps({"stemmed": {"bm25_stemming": True, "rrf_weights": [1.0, 0.5]}}),
        encoding="utf-8",
    )
    stemmed = get_preset("stemmed", path)
    assert stemmed.bm25_stemming and stemmed.rrf_weights == (1.0, 0.5)
    assert stemmed.date_prior == 1.0  # unset fields keep the phase3 values

    for bad, message in [
        ([], "JSON object"),
        ({"phase3": {}}, "fixed in code"),
        ({"x": {"name": "y"}}, "without a name"),
    ]:
        path.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_presets(path)
    path.write_text(json.dumps({"x": {"bm25_stemming": "very"}}), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_presets(path)


# get_retriever under the default


def test_default_config_builds_the_phase3_retriever(settings: Settings) -> None:
    for method in METHODS:
        plain = get_retriever(method, settings, embeddings=embeddings(), scorer=fake_scorer)
        for config in (PHASE3_CONFIG, get_preset("phase3"), RetrieverConfig(name="same")):
            configured = get_retriever(
                method, settings, config, embeddings=embeddings(), scorer=fake_scorer
            )
            assert type(configured) is type(plain)
            for query in QUERIES:
                for as_of in (None, date(2023, 12, 31)):
                    kwargs = {"top_k": 6, "as_of": as_of}
                    assert ranked(configured, query, **kwargs) == ranked(plain, query, **kwargs)
    assert get_retriever("bm25", settings, PHASE3_CONFIG) is get_retriever("bm25", settings)


# The production default: settings.retriever_preset (RETRIEVER_PRESET)


def test_the_production_default_is_the_tuned_preset() -> None:
    default = Settings(_env_file=None)

    assert default.retriever_preset == "tuned"
    assert production_config(default) == get_preset("tuned")
    pinned = default.model_copy(update={"retriever_preset": PHASE3_PRESET})
    assert production_config(pinned) is PHASE3_CONFIG
    with pytest.raises(ValueError, match="RETRIEVER_PRESET: unknown retriever preset 'best'"):
        production_config(default.model_copy(update={"retriever_preset": "best"}))
    with pytest.raises(ValidationError):
        Settings(_env_file=None, retriever_preset=" ")


def test_get_retriever_defaults_to_the_production_preset(settings: Settings) -> None:
    production = settings.model_copy(update={"retriever_preset": "tuned"})
    kwargs: dict[str, Any] = {"embeddings": embeddings(), "scorer": fake_scorer}
    for method in METHODS:
        by_default = get_retriever(method, production, **kwargs)
        tuned = get_retriever(method, production, get_preset("tuned"), **kwargs)
        phase3 = get_retriever(method, production, PHASE3_CONFIG, **kwargs)
        assert type(by_default) is type(tuned)
        for query in QUERIES:
            assert ranked(by_default, query, top_k=6) == ranked(tuned, query, top_k=6)
        # The tuned priors reorder at least one dated question, else this proves nothing.
        assert any(
            ranked(phase3, query, top_k=6) != ranked(tuned, query, top_k=6) for query in QUERIES
        ), method


def test_the_api_builds_retrievers_under_the_production_preset(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.api import rag as api_rag

    seen: list[RetrieverConfig | None] = []
    monkeypatch.setattr(
        api_rag, "get_retriever", lambda method, s, config=None: seen.append(config)
    )
    for preset in ("tuned", PHASE3_PRESET):
        api_rag.get_retriever_factory(settings.model_copy(update={"retriever_preset": preset}))(
            "hybrid_rerank"
        )

    assert seen == [get_preset("tuned"), PHASE3_CONFIG]


# get_retriever under other configs


def test_stemming_uses_a_separate_cached_index(settings: Settings) -> None:
    stemmed = RetrieverConfig(name="s", bm25_stemming=True)

    first = get_retriever("bm25", settings, stemmed)

    assert isinstance(first, BM25Retriever) and first.stemmed
    assert get_retriever("bm25", settings, stemmed) is first
    assert not get_retriever("bm25", settings).stemmed  # the Phase 3 index is untouched
    assert all(
        score == 0 for _, score in ranked(get_retriever("bm25", settings), "lowering", top_k=4)
    )
    lowering = ranked(first, "lowering", top_k=4)  # "lower" in the 2019 and 2024 documents
    assert all(score > 0 for _, score in lowering)
    assert {cid.split("#")[0][-10:] for cid, _ in lowering} == {"2019-07-31", "2024-09-18"}


def test_query_expansion_changes_bm25_but_not_dense(settings: Settings) -> None:
    config = RetrieverConfig(name="e", query_expansion=True, bm25_stemming=True)

    bm25 = get_retriever("bm25", settings, config)
    dense = get_retriever("dense", settings, config, embeddings=embeddings())
    plain_dense = get_retriever("dense", settings, embeddings=embeddings())

    assert isinstance(bm25, BM25Retriever) and bm25.expansion_weight == 0.5
    hits = ranked(bm25, "cutting rates", top_k=4)  # cut -> lower, via the synonym map
    assert all(score > 0 for _, score in hits)
    assert {cid.split("#")[0][-10:] for cid, _ in hits} == {"2019-07-31", "2024-09-18"}
    assert all(
        score == 0 for _, score in ranked(get_retriever("bm25", settings), "cutting rates", top_k=4)
    )
    assert isinstance(dense, DenseRetriever)
    assert ranked(dense, "cutting rates", top_k=5) == ranked(plain_dense, "cutting rates", top_k=5)


def test_rrf_weights_and_k_reach_the_fusion(chunks: list[Chunk]) -> None:
    a, b, c = chunks[:3]
    sparse = [RetrievedChunk(a, 3.0), RetrievedChunk(b, 2.0)]
    dense = [RetrievedChunk(c, 0.9), RetrievedChunk(b, 0.8)]

    fused = reciprocal_rank_fusion([sparse, dense], k=10, weights=[2.0, 0.5])

    assert [h.chunk for h in fused] == [b, a, c]  # b's second places beat a's first
    assert fused[0].score == pytest.approx(2.0 / 12 + 0.5 / 12)
    assert fused[1].score == pytest.approx(2.0 / 11)
    assert fused[2].score == pytest.approx(0.5 / 11)
    with pytest.raises(ValueError, match="weights"):
        reciprocal_rank_fusion([sparse, dense], weights=[1.0])


def test_get_retriever_passes_the_fusion_settings(settings: Settings) -> None:
    config = RetrieverConfig(name="f", rrf_k=5, rrf_weights=(1.0, 0.0), min_candidates=7)

    hybrid = get_retriever("hybrid", settings, config, embeddings=embeddings())
    hits = hybrid.retrieve("coronavirus", top_k=3)

    assert isinstance(hybrid, HybridRetriever)
    # Dense weight 0: the two BM25 matches lead in BM25 order, dense only fills up.
    bm25 = get_retriever("bm25", settings).retrieve("coronavirus", top_k=2)
    assert [h.chunk for h in hits[:2]] == [h.chunk for h in bm25]
    assert [h.score for h in hits] == [pytest.approx(1 / 6), pytest.approx(1 / 7), 0.0]


class Recorder:
    """A retriever stub that returns fixed hits and records the top_k it was asked for."""

    def __init__(self, hits: list[RetrievedChunk]) -> None:
        self.hits = hits
        self.asked: list[int] = []

    def retrieve(self, query: str, *, top_k: int, **_: Any) -> list[RetrievedChunk]:
        self.asked.append(top_k)
        return self.hits[:top_k]


def test_prior_retriever_reorders_a_deeper_pool_only_when_the_prior_applies(
    chunks: list[Chunk],
) -> None:
    inner = Recorder([RetrievedChunk(c, 1.0) for c in chunks])  # all tied, corpus order
    prior = MetadataPrior(date_boost=2.0)
    retriever = PriorRetriever(inner, prior, minimum=12)

    plain = retriever.retrieve("lower the target range", top_k=3)
    dated = retriever.retrieve("lower the target range in September 2024", top_k=3)

    assert inner.asked == [3, 12]  # passed through, then candidate_count(3, 12)
    assert [h.chunk for h in plain] == chunks[:3]
    assert {h.chunk.meeting_date for h in dated[:2]} == {date(2024, 9, 18)}
    assert dated[0].score == 2.0 and dated[2].score == 1.0
    assert retriever.retrieve("q", top_k=0) == []


def test_priors_rerank_but_never_filter(settings: Settings, chunks: list[Chunk]) -> None:
    strong = RetrieverConfig(
        name="strong", date_prior=50.0, year_prior=5.0, recency_prior=20.0, statement_boost=10.0
    )
    query = "Is the Committee still discussing the decision in July 2019?"
    as_of = date(2023, 12, 31)
    eligible = {c.chunk_id for c in chunks if c.release_date <= as_of}

    for method in METHODS:
        plain = get_retriever(method, settings, embeddings=embeddings(), scorer=fake_scorer)
        boosted_ = get_retriever(
            method, settings, strong, embeddings=embeddings(), scorer=fake_scorer
        )
        everything = boosted_.retrieve(query, top_k=len(chunks), as_of=as_of)
        top = boosted_.retrieve(query, top_k=2, as_of=as_of)

        assert {h.chunk.chunk_id for h in everything} == eligible, method
        assert {
            h.chunk.chunk_id for h in plain.retrieve(query, top_k=len(chunks), as_of=as_of)
        } == (eligible)
        # The July 2019 statement gets 50 x 10: first wherever scores are not negative (fake
        # embeddings give negative cosines, which a factor brings toward 0 instead).
        if method != "dense":
            assert top[0].chunk.chunk_id == "statement-2019-07-31#0000", method


def test_recency_prior_prefers_newer_chunks_for_current_state_questions(
    settings: Settings,
) -> None:
    config = RetrieverConfig(name="r", recency_prior=5.0, recency_half_life_days=90)
    query = "Is the Committee still raising the target range?"
    as_of = date(2024, 1, 31)

    bm25 = get_retriever("bm25", settings, config)
    hits = bm25.retrieve(query, top_k=2, as_of=as_of)
    plain = get_retriever("bm25", settings).retrieve(query, top_k=2, as_of=as_of)

    assert all(h.chunk.meeting_date == date(2023, 7, 26) for h in hits)
    assert {h.chunk.meeting_date for h in plain} != {date(2023, 7, 26)}
    # Not a current-state question: no change.
    asked = "Why did the Committee raise the target range?"
    assert ranked(bm25, asked, top_k=4, as_of=as_of) == ranked(
        get_retriever("bm25", settings), asked, top_k=4, as_of=as_of
    )


def test_statement_boost_applies_to_reranked_scores(chunks: list[Chunk]) -> None:
    pool = Recorder([RetrievedChunk(c, 1.0) for c in chunks if c.doc_type == "minutes"])
    pool.hits += [RetrievedChunk(c, 1.0) for c in chunks if c.doc_type == "statement"]
    reranked = RerankRetriever(
        pool, lambda q, ps: [0.5] * len(ps), prior=MetadataPrior(statement_boost=1.5)
    )

    hits = reranked.retrieve("anything", top_k=3)

    assert all(h.chunk.doc_type == "statement" and h.score == 0.75 for h in hits)


def test_boosted_raises_negative_scores_too() -> None:
    assert boosted(0.4, 2.0) == 0.8 and boosted(-0.4, 2.0) == -0.2 and boosted(0.0, 2.0) == 0.0
    assert boosted(-0.4, 0.5) == -0.8  # a factor below 1 always lowers


def test_hybrid_rerank_prior_also_ranks_the_candidates(settings: Settings) -> None:
    config = RetrieverConfig(name="d", date_prior=100.0, min_candidates=1)
    seen: list[list[str]] = []

    def scorer(query: str, passages: Sequence[str]) -> list[float]:
        seen.append(list(passages))
        return [1.0] * len(passages)

    reranked = get_retriever(
        "hybrid_rerank", settings, config, embeddings=embeddings(), scorer=scorer
    )
    hits = reranked.retrieve("What happened at the March 2020 meeting?", top_k=1)

    assert isinstance(reranked, RerankRetriever)
    assert len(seen[0]) == candidate_count(1, 1) == 4  # a pool of 4 from about 10 chunks
    assert sum("March 15, 2020" in p for p in seen[0]) == 2  # both March 2020 chunks got in
    assert hits[0].chunk.meeting_date == date(2020, 3, 15)


def test_reranker_model_alternative_is_cached_beside_the_default(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    loads: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        CrossEncoderScorer, "load", lambda self: loads.append((self.model_name, self.revision))
    )
    l12 = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    config = RetrieverConfig(name="l12", reranker_model=l12)
    clear_retriever_cache()

    built = [
        get_retriever("hybrid_rerank", settings, chosen, embeddings=embeddings())
        for chosen in (config, PHASE3_CONFIG, config, PHASE3_CONFIG)
    ]
    for retriever in built:
        preload(retriever)

    scorers = [r.scorer for r in built if isinstance(r, RerankRetriever)]
    assert len(scorers) == 4 and scorers[0] is scorers[2] and scorers[1] is scorers[3]
    assert isinstance(scorers[0], CrossEncoderScorer) and isinstance(scorers[1], CrossEncoderScorer)
    assert (scorers[0].model_name, scorers[0].revision) == (l12, None)
    assert scorers[1].model_name == settings.reranker_model
    assert loads == [(l12, None), (settings.reranker_model, settings.reranker_revision)] * 2
    clear_retriever_cache()
