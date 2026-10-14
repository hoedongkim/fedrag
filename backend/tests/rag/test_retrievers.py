"""Tests for the BM25 tokenizer and the four retrievers, the RRF fusion, the factory and the CLI."""

import hashlib
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import httpx
import openai
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

import app.rag.__main__ as cli
from app.core.config import Settings
from app.db.load import DocumentRow, ParagraphRow
from app.rag.bm25 import BM25Retriever, tokenize
from app.rag.index import IndexNotBuiltError, build_index, load_chunks
from app.rag.retrievers import (
    CrossEncoderScorer,
    DenseRetriever,
    HybridRetriever,
    RerankRetriever,
    candidate_count,
    clear_retriever_cache,
    get_retriever,
    preload,
    reciprocal_rank_fusion,
)
from app.rag.smoke import SmokeQuestion, first_relevant_rank, run_retrieval
from app.rag.types import Chunk, RetrievedChunk

TEXTS = {
    date(2015, 12, 16): "The Committee decided to raise the target range to 1/4 to 1/2 percent.",
    date(2019, 7, 31): "The Committee decided to lower the target range to 2 to 2-1/4 percent.",
    date(2020, 3, 15): "The coronavirus outbreak has harmed communities and disrupted activity.",
    date(2023, 7, 26): "The Committee decided to raise the target range to 5-1/4 to 5-1/2 percent.",
    date(2024, 9, 18): "The Committee decided to lower the target range to 4-3/4 to 5 percent.",
}


def make_statement(meeting: date, text: str) -> DocumentRow:
    return DocumentRow(
        doc_id=f"statement-{meeting.isoformat()}",
        doc_type="statement",
        meeting_date=meeting,
        release_date=meeting,
        release_date_source="test",
        page_updated=None,
        title=f"FOMC Statement, {meeting:%B %d, %Y}",
        url=f"https://www.federalreserve.gov/test/{meeting:%Y%m%d}.htm",
        paragraphs=[ParagraphRow(section=None, text=text)],
        text=text,
        word_count=len(text.split()),
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        parser_version=1,
    )


def make_minutes(meeting: date, text: str) -> DocumentRow:
    statement = make_statement(meeting, text)
    return DocumentRow.model_validate(
        statement.model_dump()
        | {
            "doc_id": f"minutes-{meeting.isoformat()}",
            "doc_type": "minutes",
            "release_date": meeting + timedelta(days=21),
            "title": f"Minutes of the FOMC Meeting, {meeting:%B %d, %Y}",
            "paragraphs": [{"section": "Committee Policy Action", "text": text}],
        }
    )


DOCUMENTS = [make_statement(day, text) for day, text in TEXTS.items()] + [
    make_minutes(day, f"Participants discussed the decision. {text}") for day, text in TEXTS.items()
]


def fake_scorer(query: str, passages: Sequence[str]) -> list[float]:
    """Counts query words in each passage: a stand-in for the cross-encoder."""
    words = set(tokenize(query))
    return [float(len(words & set(tokenize(p)))) for p in passages]


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
    build_index(
        built,
        documents=DOCUMENTS,
        embeddings=DeterministicFakeEmbedding(size=16),
        log=lambda _: None,
    )
    clear_retriever_cache()
    return built


@pytest.fixture
def chunks(settings: Settings) -> list[Chunk]:
    return load_chunks(settings)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("raise the range to 5-1/4 to 5-1/2 percent", ["raise", "range", "5-1/4", "5.25"]),
        ("in 2024 and 2015", ["2024", "2015"]),
        ("1/4 percentage point", ["1/4", "0.25", "percentage", "point"]),
        ("5‑1/4 percent", ["5-1/4", "5.25", "percent"]),  # a non-breaking hyphen
        ("2.5% and 1,000 jobs", ["2.5", "1,000", "jobs"]),
        ("The Committee's views", ["committee", "views"]),
        ("rates—which rose", ["rates", "rose"]),  # the em dash separates words
    ],
)
def test_tokenize(text: str, expected: list[str]) -> None:
    assert [t for t in tokenize(text) if t in expected] == expected
    assert "the" not in tokenize(text) and "to" not in tokenize(text)


def test_bm25_matches_rates_and_their_decimals(chunks: list[Chunk]) -> None:
    retriever = BM25Retriever(chunks)

    for query in ("5-1/4 percent", "5.25 percent"):
        top = retriever.retrieve(query, top_k=2)
        assert {hit.chunk.meeting_date for hit in top} == {date(2023, 7, 26)}
        assert top[0].score > 0


def test_bm25_filters_before_ranking(chunks: list[Chunk]) -> None:
    retriever = BM25Retriever(chunks)

    hits = retriever.retrieve(
        "5-1/4 percent", top_k=3, as_of=date(2020, 12, 31), doc_types=["statement"]
    )

    # The best match was released in 2023, so the filler comes from what was public then.
    assert len(hits) == 3
    assert all(h.chunk.release_date <= date(2020, 12, 31) for h in hits)
    assert all(h.chunk.doc_type == "statement" for h in hits)
    assert BM25Retriever([]).retrieve("rates", top_k=3) == []
    assert retriever.retrieve("rates", top_k=3, as_of=date(2010, 1, 1)) == []


def test_dense_scores_are_cosine_similarity(settings: Settings, chunks: list[Chunk]) -> None:
    dense = get_retriever("dense", settings, embeddings=DeterministicFakeEmbedding(size=16))
    target = chunks[3]

    hits = dense.retrieve(target.search_text, top_k=4)

    assert isinstance(dense, DenseRetriever)
    assert hits[0].chunk == target and hits[0].score == pytest.approx(1.0, abs=1e-5)
    assert all(-1.0 <= h.score <= 1.0 + 1e-6 for h in hits)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_reciprocal_rank_fusion(chunks: list[Chunk]) -> None:
    a, b, c, d = chunks[:4]
    sparse = [RetrievedChunk(a, 9.0), RetrievedChunk(b, 5.0), RetrievedChunk(c, 1.0)]
    dense = [RetrievedChunk(c, 0.9), RetrievedChunk(a, 0.8), RetrievedChunk(d, 0.1)]

    fused = reciprocal_rank_fusion([sparse, dense], k=60)

    assert [h.chunk for h in fused] == [a, c, b, d]
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert fused[1].score == pytest.approx(1 / 63 + 1 / 61)
    assert (fused[2].score, fused[3].score) == (pytest.approx(1 / 62), pytest.approx(1 / 63))
    # A tie goes to the chunk reached first: same rank, earlier list.
    tied = reciprocal_rank_fusion([[RetrievedChunk(d, 1.0)], [RetrievedChunk(a, 1.0)]])
    assert [h.chunk for h in tied] == [d, a]
    assert reciprocal_rank_fusion([[], []]) == []


class Recorder:
    """A retriever stub that returns fixed hits and records the top_k it was asked for."""

    def __init__(self, hits: list[RetrievedChunk]) -> None:
        self.hits = hits
        self.asked: list[int] = []

    def retrieve(self, query: str, *, top_k: int, **_: Any) -> list[RetrievedChunk]:
        self.asked.append(top_k)
        return self.hits[:top_k]


def test_hybrid_fetches_candidates_from_both_sides(chunks: list[Chunk]) -> None:
    sparse = Recorder([RetrievedChunk(c, 1.0) for c in chunks])
    dense = Recorder([RetrievedChunk(c, 1.0) for c in reversed(chunks)])

    hits = HybridRetriever(sparse, dense).retrieve("q", top_k=20)

    assert sparse.asked == dense.asked == [candidate_count(20)] == [80]
    assert len(hits) == min(20, len(chunks))
    assert candidate_count(5) == 50


def test_hybrid_ignores_sparse_hits_that_match_no_query_term(chunks: list[Chunk]) -> None:
    sparse = Recorder([RetrievedChunk(c, 0.0) for c in chunks])  # BM25's score-0 padding
    dense = Recorder([RetrievedChunk(c, 0.5) for c in reversed(chunks)])

    hits = HybridRetriever(sparse, dense).retrieve("Volcker", top_k=3)

    assert [h.chunk for h in hits] == list(reversed(chunks))[:3]


def test_rerank_reorders_the_hybrid_pool_by_the_scorer(chunks: list[Chunk]) -> None:
    pool = Recorder([RetrievedChunk(c, 1.0) for c in chunks])

    hits = RerankRetriever(pool, fake_scorer).retrieve("coronavirus outbreak", top_k=2)

    assert pool.asked == [50]
    assert {h.chunk.meeting_date for h in hits} == {date(2020, 3, 15)}
    assert hits[0].score == 2.0
    with pytest.raises(ValueError, match="scores"):
        RerankRetriever(pool, lambda q, p: [1.0]).retrieve("q", top_k=2)


def test_get_retriever_caches_and_builds_every_method(settings: Settings) -> None:
    embeddings = DeterministicFakeEmbedding(size=16)

    assert get_retriever("bm25", settings) is get_retriever("bm25", settings)
    for method in ("bm25", "dense", "hybrid", "hybrid_rerank"):
        retriever = get_retriever(method, settings, embeddings=embeddings, scorer=fake_scorer)
        hits = retriever.retrieve("raise the target range", top_k=3, doc_types=["minutes"])
        assert len(hits) == 3 and all(h.chunk.doc_type == "minutes" for h in hits)


def test_get_retriever_picks_up_a_rebuild(settings: Settings) -> None:
    before = get_retriever("bm25", settings)
    extra = make_statement(date(2024, 11, 7), "The Committee decided to lower the target range.")
    build_index(
        settings,
        documents=[*DOCUMENTS, extra],
        embeddings=DeterministicFakeEmbedding(size=16),
        log=lambda _: None,
    )

    after = get_retriever("bm25", settings)

    assert after is not before
    assert any(h.chunk.doc_id == extra.doc_id for h in after.retrieve("lower", top_k=20))


def test_get_retriever_caches_dense_and_the_scorer(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The production path: make_embeddings and the CrossEncoder replaced, nothing injected."""
    scorers: list[tuple[str, str]] = []

    def make_scorer(model_name: str, *, revision: str) -> Any:
        scorers.append((model_name, revision))
        return fake_scorer

    monkeypatch.setattr(
        "app.rag.retrievers.make_embeddings", lambda _: DeterministicFakeEmbedding(size=16)
    )
    monkeypatch.setattr("app.rag.retrievers.CrossEncoderScorer", make_scorer)
    clear_retriever_cache()

    dense = get_retriever("dense", settings)
    assert get_retriever("dense", settings) is dense
    assert get_retriever("hybrid_rerank", settings).retrieve("raise the target range", top_k=3)
    assert get_retriever("hybrid_rerank", settings).retrieve("lower the target range", top_k=3)
    assert scorers == [(settings.reranker_model, settings.reranker_revision)]  # loaded once

    extra = make_statement(date(2024, 11, 7), "The Committee decided to lower the target range.")
    build_index(
        settings,
        documents=[*DOCUMENTS, extra],
        embeddings=DeterministicFakeEmbedding(size=16),
        log=lambda _: None,
    )
    rebuilt = get_retriever("dense", settings)
    assert rebuilt is not dense
    assert any(h.chunk.doc_id == extra.doc_id for h in rebuilt.retrieve("lower", top_k=20))
    clear_retriever_cache()


def test_preload_loads_only_a_cross_encoder_and_only_once(
    chunks: list[Chunk], monkeypatch: pytest.MonkeyPatch
) -> None:
    loads: list[str] = []
    monkeypatch.setattr(
        CrossEncoderScorer, "_load", lambda self, torch: loads.append(self.model_name) or object()
    )
    bm25 = BM25Retriever(chunks)
    reranked = RerankRetriever(bm25, CrossEncoderScorer("some/cross-encoder"))

    for retriever in (bm25, RerankRetriever(bm25, fake_scorer), reranked, reranked):
        preload(retriever)

    assert loads == ["some/cross-encoder"]


def test_get_retriever_without_an_index(tmp_path: Path) -> None:
    empty = Settings(_env_file=None, index_dir=tmp_path / "index", chroma_dir=tmp_path / "c")

    with pytest.raises(IndexNotBuiltError, match="make index"):
        get_retriever("bm25", empty)


def test_get_retriever_refuses_a_dense_index_from_another_model(settings: Settings) -> None:
    other = settings.model_copy(update={"embedding_model": "text-embedding-3-large"})
    embeddings = DeterministicFakeEmbedding(size=16)

    with pytest.raises(IndexNotBuiltError, match="--rebuild"):
        get_retriever("dense", other, embeddings=embeddings)
    assert get_retriever("bm25", other).retrieve("rates", top_k=1)  # BM25 needs no vectors


def test_cli_search(settings: Settings, capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["search", "5-1/4 percent", "--method", "bm25", "--top-k", "2"]
    argv += ["--as-of", "2023-07-26", "--doc-type", "statement", "minutes"]

    assert cli.main(argv, settings=settings) == 0

    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith(" 1") and "statement-2023-07-26" in out[0]
    assert "released 2023-07-26" in out[0] and len(out) == 4


def test_cli_search_without_an_index(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    empty = Settings(_env_file=None, index_dir=tmp_path / "index", chroma_dir=tmp_path / "c")

    assert cli.main(["search", "rates", "--method", "bm25"], settings=empty) == 1
    assert "make index" in capsys.readouterr().err


def test_cli_index_dry_run(settings: Settings, capsys: pytest.CaptureFixture[str]) -> None:
    processed = settings.data_dir / "processed"
    processed.mkdir(parents=True)
    (processed / "documents.jsonl").write_text(
        "".join(d.model_dump_json() + "\n" for d in DOCUMENTS), encoding="utf-8"
    )

    assert cli.main(["index", "--source", "processed", "--dry-run"], settings=settings) == 0

    out = capsys.readouterr().out
    assert "0 chunks to embed" in out and "Dry run" in out


def scripted_llm(*answers: str) -> GenericFakeChatModel:
    """A chat model that gives these answers in turn (no OpenAI)."""
    return GenericFakeChatModel(messages=iter(answers))


def test_cli_ask_prints_the_answer_and_numbered_citations(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    llm = scripted_llm("It raised the range [1].")
    monkeypatch.setattr("app.rag.answer.get_llm", lambda _: llm)
    argv = ["ask", "raise the target range", "--method", "bm25", "--as-of", "2016-01-01"]

    assert cli.main(argv, settings=settings) == 0

    out = capsys.readouterr().out
    assert out.startswith("It raised the range [1].\nCitations:\n  [1] statement-2015-12-16 ")
    assert "chunk statement-2015-12-16#0000" in out and "2023" not in out


def test_cli_ask_prints_notes_apart_from_the_answer(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("app.rag.answer.get_llm", lambda _: scripted_llm("It rose [1]."))
    argv = ["ask", "raise the target range", "--method", "bm25", "--as-of", "2016-06-01"]

    assert cli.main([*argv, "--doc-type", "statement"], settings=settings) == 0

    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "It rose [1]."
    assert lines[1] == (
        "Note: The most recent passage cited was released Dec 16, 2015, 168 days before "
        "Jun 1, 2016; later decisions may not be reflected."
    )
    assert lines[2] == "Citations:"


def test_cli_ask_without_a_key(settings: Settings, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["ask", "Did rates rise?", "--method", "bm25"], settings=settings) == 1
    assert "OPENAI_API_KEY" in capsys.readouterr().err


SMOKE = [
    SmokeQuestion(
        id="q1",
        question="raise the target range 5-1/4 percent",
        as_of=date(2023, 12, 31),
        expected_doc_ids=["statement-2023-07-26"],
        expect="5-1/4 to 5-1/2 percent",
    ),
    SmokeQuestion(
        id="q2",
        question="coronavirus outbreak",
        expected_doc_ids=["minutes-2020-03-15"],
        expect="minutes",
    ),
    SmokeQuestion(id="q3", question="Volcker", expected_doc_ids=[], expect="nothing"),
]


def test_smoke_metrics(settings: Settings, chunks: list[Chunk]) -> None:
    result = run_retrieval("bm25", get_retriever("bm25", settings), SMOKE)

    assert set(result.ranks) == {"q1", "q2"}  # the unanswerable question is not scored
    assert result.ranks["q1"] == 1 and result.ranks["q2"] == 2  # the statement outranks it
    assert result.hit_at_k == 1.0 and result.mrr == pytest.approx((1 + 1 / 2) / 2)
    assert result.leaks == 0

    class Leaky:
        def retrieve(self, query: str, **filters: Any) -> list[RetrievedChunk]:
            return [RetrievedChunk(chunk, 1.0) for chunk in chunks]  # ignores as_of

    leaky = run_retrieval("dense", Leaky(), SMOKE)
    assert leaky.leaks == sum(c.release_date > date(2023, 12, 31) for c in chunks) > 0
    assert first_relevant_rank(["a", "b", "c"], ["c", "b"]) == 2
    assert first_relevant_rank(["a"], ["z"]) is None


def test_cli_smoke(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "smoke.jsonl"
    path.write_text("".join(q.model_dump_json() + "\n" for q in SMOKE), encoding="utf-8")
    llm = scripted_llm(
        "Raised to 5-1/4 to 5-1/2 percent [1].",
        "The coronavirus disrupted activity [2].",
        "The passages do not say.",
    )
    monkeypatch.setattr("app.rag.answer.get_llm", lambda _: llm)
    argv = ["smoke", "--file", str(path), "--methods", "bm25"]

    assert cli.main([*argv, "--answers", "--answer-method", "bm25"], settings=settings) == 0

    out = capsys.readouterr().out
    assert out.startswith("SMOKE CHECK - 3 hand-labelled questions (2 answerable)")
    assert "| `bm25` | 1.00 | 0.75 | 0 |" in out
    assert out.count("check: cites an expected document; citations after as_of: 0") == 2
    assert "check: no citations" in out


def test_cli_smoke_with_cached_embeddings_and_a_preset(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from app.evaluation import runner
    from app.stance.llm import PredictionCache

    path = tmp_path / "smoke.jsonl"
    path.write_text("".join(q.model_dump_json() + "\n" for q in SMOKE), encoding="utf-8")
    cache = PredictionCache(tmp_path / "query_embeddings.jsonl")
    monkeypatch.setattr(
        runner,
        "query_embeddings",
        lambda s: runner.CachedQueryEmbeddings(
            s.embedding_model, cache, lambda: DeterministicFakeEmbedding(size=16)
        ),
    )
    argv = ["smoke", "--file", str(path), "--methods", "dense", "--cached-embeddings"]

    assert cli.main([*argv, "--preset", "tuned"], settings=settings) == 0
    out = capsys.readouterr().out
    assert "Retriever preset: tuned" in out and "(3 embedded now, 0 cached)" in out
    assert cli.main(argv, settings=settings) == 0  # the fixture's preset, phase3
    out = capsys.readouterr().out
    assert "Retriever preset: phase3" in out and "(0 embedded now, 3 cached)" in out
    assert "no OpenAI round trip" in out

    assert cli.main([*argv, "--preset", "best"], settings=settings) == 1
    assert "unknown retriever preset 'best'" in capsys.readouterr().err


def test_cli_reports_openai_errors_by_class_only(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    error = openai.AuthenticationError(
        "Incorrect API key provided: sk-abc...wxyz",
        response=httpx.Response(401, request=request),
        body=None,
    )

    def refuse(_: Settings) -> Any:
        raise error

    monkeypatch.setattr("app.rag.answer.get_llm", refuse)

    assert cli.main(["ask", "Did rates rise?", "--method", "bm25"], settings=settings) == 1

    err = capsys.readouterr().err
    assert err == "ask: OpenAI request failed (AuthenticationError, HTTP 401)\n"
