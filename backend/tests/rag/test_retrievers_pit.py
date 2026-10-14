"""Point-in-time property test: every method, under every retriever configuration, filters by
as_of and doc_types before ranking."""

import hashlib
import random
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from app.core.config import Settings
from app.db.load import DocumentRow, ParagraphRow
from app.rag.config import PHASE3_CONFIG, RetrieverConfig, load_presets
from app.rag.index import build_index, load_chunks
from app.rag.retrievers import clear_retriever_cache, get_retriever
from app.rag.types import Chunk

TOPICS = [
    "inflation", "employment", "balance sheet", "liquidity", "housing", "energy prices",
    "tariffs", "banking stress", "productivity", "wage growth", "5-1/4 percent", "2-1/2 percent",
]  # fmt: skip
SECTIONS = ["Staff Economic Outlook", "Participants' Views", "Committee Policy Action"]
COMBOS_PER_METHOD = 150
# phase3, every preset in retriever_presets.json, and two configurations with every improvement
# on at strong settings (one of them a penalty), so that the priors decide most rankings.
CONFIGS = {
    **load_presets(),
    "all_on": RetrieverConfig(
        name="all_on",
        bm25_stemming=True,
        query_expansion=True,
        date_prior=20.0,
        year_prior=4.0,
        recency_prior=10.0,
        recency_half_life_days=60,
        statement_boost=5.0,
        rrf_k=10,
        rrf_weights=(1.0, 0.3),
        min_candidates=20,
    ),
    "dense_heavy": RetrieverConfig(
        name="dense_heavy",
        date_prior=3.0,
        statement_boost=0.5,
        rrf_weights=(0.2, 1.0),
        min_candidates=5,
    ),
}
MONTHS = ["March", "Sept.", "September", "Mar"]


def document(doc_type: str, meeting: date, release: date, rng: random.Random) -> DocumentRow:
    paragraphs = [
        ParagraphRow(
            section=None if doc_type == "statement" else rng.choice(SECTIONS),
            text=" ".join(
                f"The {doc_type} discussed {rng.choice(TOPICS)} and {rng.choice(TOPICS)} in "
                f"{meeting.year}, noting item {n}-{i}."
                for i in range(rng.randint(2, 12))
            ),
        )
        for n in range(1 if doc_type == "statement" else rng.randint(3, 9))
    ]
    paragraphs.sort(key=lambda p: SECTIONS.index(p.section) if p.section else 0)
    text = "\n\n".join(p.text for p in paragraphs)
    return DocumentRow(
        doc_id=f"{doc_type}-{meeting.isoformat()}",
        doc_type=doc_type,  # type: ignore[arg-type]
        meeting_date=meeting,
        release_date=release,
        release_date_source="test",
        page_updated=release + timedelta(days=40) if rng.random() < 0.2 else None,
        title=f"{doc_type.title()} {meeting.isoformat()}",
        url=f"https://www.federalreserve.gov/test/{doc_type}{meeting:%Y%m%d}.htm",
        paragraphs=paragraphs,
        text=text,
        word_count=len(text.split()),
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        parser_version=1,
    )


def corpus(seed: int = 7) -> list[DocumentRow]:
    """Two meetings a year 2015-2026; statements out the same day (sometimes a few days late),
    minutes three weeks later."""
    rng = random.Random(seed)
    documents = []
    for year in range(2015, 2027):
        for month in (3, 9):
            meeting = date(year, month, rng.randint(10, 25))
            late = timedelta(days=rng.choice([0, 0, 0, 3]))
            documents.append(document("statement", meeting, meeting + late, rng))
            documents.append(document("minutes", meeting, meeting + timedelta(days=21), rng))
    return documents


def fake_scorer(query: str, passages: Sequence[str]) -> list[float]:
    """Deterministic pseudo-relevance, unrelated to any other method's ranking."""
    return [
        int(hashlib.sha256(f"{query}|{p}".encode()).hexdigest()[:8], 16) / 2**32 for p in passages
    ]


@pytest.fixture(scope="module")
def settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    root: Path = tmp_path_factory.mktemp("pit")
    built = Settings(
        _env_file=None,
        environment="test",
        openai_api_key=None,
        database_url=None,
        embedding_model="text-embedding-3-small",
        data_dir=root / "data",
        chroma_dir=root / "chroma",
        index_dir=root / "index",
    )
    build_index(
        built,
        documents=corpus(),
        embeddings=DeterministicFakeEmbedding(size=24),
        log=lambda _: None,
    )
    clear_retriever_cache()
    return built


@pytest.mark.parametrize("config", CONFIGS.values(), ids=list(CONFIGS))
@pytest.mark.parametrize("method", ["bm25", "dense", "hybrid", "hybrid_rerank"])
def test_no_method_returns_a_chunk_from_after_as_of(
    settings: Settings, method: str, config: RetrieverConfig
) -> None:
    chunks = load_chunks(settings)
    release_dates = sorted({c.release_date for c in chunks})
    retriever = get_retriever(
        method,  # type: ignore[arg-type]
        settings,
        config,
        embeddings=DeterministicFakeEmbedding(size=24),
        scorer=fake_scorer,
    )
    seed = f"pit-{method}" if config is PHASE3_CONFIG else f"pit-{method}-{config.name}"
    rng = random.Random(seed)

    for _ in range(COMBOS_PER_METHOD):
        as_of = random_as_of(rng, release_dates)
        doc_types = rng.choice(
            [None, ["statement"], ["minutes"], ["statement", "minutes"], ["sep"]]
        )
        top_k = rng.randint(1, 25)
        query = " ".join(rng.sample(TOPICS, rng.randint(1, 3)))
        if config is not PHASE3_CONFIG:  # the Phase 3 draws stay as they were
            query = random_prior_triggers(rng, query)

        hits = retriever.retrieve(query, top_k=top_k, as_of=as_of, doc_types=doc_types)

        eligible = [c for c in chunks if allowed(c, as_of, doc_types)]
        context = (
            f"{config.name} {method} {query!r} as_of={as_of} doc_types={doc_types} top_k={top_k}"
        )
        assert all(allowed(h.chunk, as_of, doc_types) for h in hits), context
        assert len(hits) == min(top_k, len(eligible)), context
        assert len({h.chunk.chunk_id for h in hits}) == len(hits), context
        assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True), context


def random_prior_triggers(rng: random.Random, query: str) -> str:
    """The query, often naming a month or a year of the corpus (2015-2026, meetings in March and
    September) or asking about the current state, so that the priors apply."""
    roll = rng.random()
    if roll < 0.35:
        query += f" in {rng.choice(MONTHS)} {rng.randint(2014, 2027)}"
    elif roll < 0.6:
        query += f" in {rng.randint(2014, 2027)}"
    if rng.random() < 0.4:
        query = f"Is the Committee still discussing {query}?"
    return query


def random_as_of(rng: random.Random, release_dates: list[date]) -> date | None:
    """None, a release date or a day either side of one (the boundaries), or any day."""
    roll = rng.random()
    if roll < 0.1:
        return None
    if roll < 0.5:
        return rng.choice(release_dates) + timedelta(days=rng.choice([-1, 0, 1]))
    return date(2014, 6, 1) + timedelta(days=rng.randint(0, 13 * 365))


def allowed(chunk: Chunk, as_of: date | None, doc_types: list[str] | None) -> bool:
    return (as_of is None or chunk.release_date <= as_of) and (
        doc_types is None or chunk.doc_type in doc_types
    )
