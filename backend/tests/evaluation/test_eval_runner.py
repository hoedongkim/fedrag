"""Retrieval runs and cached answers with a fake retriever and a fake chat model (no OpenAI)."""

import itertools
from datetime import date
from typing import Any

import pytest
from eval_fakes import QUESTIONS, FakeAnswerLLM, FakeRetriever
from langchain_core.embeddings import Embeddings

from app.core.config import Settings
from app.evaluation import runner
from app.evaluation.questions import EvalQuestion
from app.evaluation.runner import (
    ANSWER_OUTPUT_TOKENS_PER_CALL,
    AnswerItem,
    AnswerRunner,
    CachedQueryEmbeddings,
    resolve_config,
    retrieval_key,
    run_retrieval,
)
from app.rag.config import PHASE3_CONFIG, RetrieverConfig, get_preset
from app.stance.llm import CallLimitError, PredictionCache, Usage, cost_usd

ALL = [EvalQuestion.model_validate(q) for q in QUESTIONS]
D1, D2, D3, T1 = ALL


@pytest.fixture
def setup(settings: Settings) -> runner.RetrieverSetup:
    return resolve_config("phase3", settings)


def ticking() -> Any:
    counter = itertools.count()
    return lambda: next(counter) * 0.010  # every call 10 ms after the previous one


def test_retrieval_passes_as_of_and_depth_and_stores_each_row(
    settings: Settings, setup: runner.RetrieverSetup
) -> None:
    fake = FakeRetriever()
    cache = PredictionCache(settings.data_dir / "eval_cache" / "retrieval.jsonl")

    rows = run_retrieval(
        [D1, D2], ["bm25", "dense"], setup, lambda s, m: fake, cache, clock=ticking()
    )

    # Per method: an untimed warm-up with the first question, then every question.
    assert [(c["query"], c["top_k"], c["as_of"]) for c in fake.calls[:3]] == [
        (D1.question, 10, date(2024, 10, 1)),
        (D1.question, 10, date(2024, 10, 1)),
        (D2.question, 10, None),
    ]
    assert len(fake.calls) == 6
    d1 = rows["bm25"][0]
    assert [h["chunk_id"] for h in d1["hits"]] == [
        "minutes-2024-07-31#0003",
        "statement-2024-09-18#0000",
        "statement-2024-07-31#0000",
    ]
    assert d1["leaks"] == 0 and d1["latency_ms"] == pytest.approx(10.0)
    assert d1["config"] == "phase3" and d1["depth"] == 10
    assert cache.get(retrieval_key(D1, "bm25", setup)) == d1
    assert PredictionCache(cache.path).get(retrieval_key(D2, "dense", setup)) == rows["dense"][1]


def test_retrieval_counts_leaks(settings: Settings, setup: runner.RetrieverSetup) -> None:
    class Leaky(FakeRetriever):
        def retrieve(self, query: str, *, top_k: int, as_of: date | None = None, **kw: Any):
            return super().retrieve(query, top_k=top_k, as_of=None)

    cache = PredictionCache(settings.data_dir / "r.jsonl")
    rows = run_retrieval([D1], ["bm25"], setup, lambda s, m: Leaky(), cache)

    assert rows["bm25"][0]["leaks"] == 1  # minutes-2024-09-18#0001, released 2024-10-09


def make_runner(
    settings: Settings, fake: FakeRetriever, llm: FakeAnswerLLM, **kwargs: Any
) -> AnswerRunner:
    return AnswerRunner(
        settings,
        lambda s, m: fake,
        llm=llm,
        cache=PredictionCache(settings.data_dir / "eval_cache" / "answers.jsonl"),
        clock=ticking(),
        **kwargs,
    )


def test_answers_use_the_production_chain_and_are_cached(
    settings: Settings, setup: runner.RetrieverSetup
) -> None:
    fake, llm = FakeRetriever(), FakeAnswerLLM()
    answers = make_runner(settings, fake, llm)
    items = [AnswerItem(q, "hybrid", setup) for q in (D1, D2)]

    pending = answers.prepare(items)
    records = answers.generate(pending)

    # An untimed warm-up query first, then each question, as POST /api/ask passes it.
    assert [(c["query"], c["top_k"], c["as_of"]) for c in fake.calls] == [
        (D1.question, 5, date(2024, 10, 1)),
        (D1.question, 5, date(2024, 10, 1)),
        (D2.question, 5, None),
    ]
    assert len(llm.calls) == 2
    assert "Answer as of 2024-10-01" in str(llm.calls[0][0].content)  # the production prompt
    d1 = records[0]
    assert d1["answer"] == "The Committee lowered the target range by 1/2 percentage point [1]."
    assert d1["citations"] == ["minutes-2024-07-31#0003"]  # passage [1] of d1's retrieval
    assert [p["chunk_id"] for p in d1["passages"]][:2] == [
        "minutes-2024-07-31#0003",
        "statement-2024-09-18#0000",
    ]
    assert all(p["release_date"] <= "2024-10-01" for p in d1["passages"])
    assert d1["usage"] == {
        "input_tokens": 1000,
        "cached_input_tokens": 100,
        "output_tokens": 200,
        "reasoning_tokens": 50,
    }
    assert d1["llm_called"] and d1["model"] == settings.llm_model
    assert d1["prompt_version"].startswith("answer.")
    assert d1["latency_ms"] == pytest.approx(d1["retrieval_ms"] + d1["generation_ms"])
    assert answers.usage.calls == 2 and answers.usage.output_tokens == 400

    # A new runner over the same cache: no retrieval, no LLM call.
    again_retriever, again_llm = FakeRetriever(), FakeAnswerLLM()
    again = make_runner(settings, again_retriever, again_llm)
    assert again.prepare(items) == []
    assert again.generate([]) == []
    assert again_retriever.calls == [] and again_llm.calls == []
    assert again.cached(items[0]) == d1


def test_the_cache_key_covers_question_method_config_top_k_prompt_and_model(
    settings: Settings, setup: runner.RetrieverSetup
) -> None:
    answers = make_runner(settings, FakeRetriever(), FakeAnswerLLM())
    base = answers.key(AnswerItem(D1, "hybrid", setup))
    other_setup = resolve_config('{"bm25_stemming": true}', settings)
    later = EvalQuestion.model_validate({**QUESTIONS[0], "as_of": "2024-10-02"})
    variants = {
        answers.key(AnswerItem(D1, "dense", setup)),
        answers.key(AnswerItem(D1, "hybrid", other_setup)),
        answers.key(AnswerItem(later, "hybrid", setup)),
        make_runner(settings, FakeRetriever(), FakeAnswerLLM(), top_k=3).key(
            AnswerItem(D1, "hybrid", setup)
        ),
        make_runner(
            settings.model_copy(update={"llm_model": "gpt-other"}),
            FakeRetriever(),
            FakeAnswerLLM(),
        ).key(AnswerItem(D1, "hybrid", setup)),
    }
    assert base not in variants and len(variants) == 5


def test_no_passages_means_no_call_and_the_chains_own_answer(
    settings: Settings, setup: runner.RetrieverSetup
) -> None:
    fake, llm = FakeRetriever(rankings={}), FakeAnswerLLM()
    answers = make_runner(settings, fake, llm)

    pending = answers.prepare([AnswerItem(D1, "bm25", setup)])
    estimate = answers.estimate(pending, 1)
    [record] = answers.generate(pending)

    assert estimate.calls == 0 and estimate.usd == 0
    assert llm.calls == [] and not record["llm_called"]
    assert record["answer"].startswith("No indexed FOMC passages match the filters")
    assert record["citations"] == [] and record["usage"]["input_tokens"] == 0


def test_estimate_prices_the_exact_prompts_before_any_call(
    settings: Settings, setup: runner.RetrieverSetup
) -> None:
    llm = FakeAnswerLLM()
    answers = make_runner(settings, FakeRetriever(), llm)
    pending = answers.prepare([AnswerItem(q, "bm25", setup) for q in (D1, D2, D3)])

    estimate = answers.estimate(pending, 5)

    assert llm.calls == []
    assert estimate.items == 5 and estimate.cached == 2 and estimate.calls == 3
    assert all(p.input_tokens > 300 for p in pending)  # the system prompt alone is longer
    assert estimate.input_tokens == sum(p.input_tokens for p in pending)
    assert estimate.output_tokens == 3 * ANSWER_OUTPUT_TOKENS_PER_CALL
    assert estimate.usd == pytest.approx(
        cost_usd(settings.llm_model, estimate.input_tokens, 0, estimate.output_tokens)
    )


def test_max_calls_refuses_before_any_call(
    settings: Settings, setup: runner.RetrieverSetup
) -> None:
    llm = FakeAnswerLLM()
    answers = make_runner(settings, FakeRetriever(), llm, usage=Usage(max_calls=1))
    pending = answers.prepare([AnswerItem(q, "bm25", setup) for q in (D1, D2)])

    with pytest.raises(CallLimitError, match="2 calls needed but only 1"):
        answers.generate(pending)
    assert llm.calls == []


def test_without_a_key_the_runner_fails_before_calling(
    settings: Settings, setup: runner.RetrieverSetup
) -> None:
    from app.rag.answer import LLMNotConfiguredError

    answers = AnswerRunner(settings, lambda s, m: FakeRetriever())
    pending = answers.prepare([AnswerItem(D1, "bm25", setup)])

    with pytest.raises(LLMNotConfiguredError):
        answers.generate(pending)
    assert answers.usage.calls == 0


def test_configs_resolve_to_rag_presets(settings: Settings) -> None:
    # "default" is the production retriever: the preset RETRIEVER_PRESET names (tuned unless set).
    default = resolve_config(None, settings)
    assert default.value == get_preset("tuned") and default.name == "tuned"
    assert resolve_config("default", settings) == default
    pinned = resolve_config(None, settings.model_copy(update={"retriever_preset": "phase3"}))
    assert pinned.value == PHASE3_CONFIG and pinned.name == "phase3"
    phase3 = resolve_config("phase3", settings)
    assert phase3.value == PHASE3_CONFIG and phase3.hash != default.hash
    # Same behaviour as phase3 (name and description aside): same hash, shared cache entries.
    assert resolve_config('{"name": "same"}', settings).hash == phase3.hash
    custom = resolve_config('{"bm25_stemming": true, "statement_boost": 1.5}', settings)
    assert isinstance(custom.value, RetrieverConfig) and custom.value.bm25_stemming
    assert custom.name.startswith("custom-") and custom.hash != default.hash
    other = resolve_config('{"statement_boost": 2.0}', settings)
    assert other.name.startswith("custom-") and other.name != custom.name
    named = resolve_config('{"name": "stemmed", "bm25_stemming": true}', settings)
    assert (
        named.name == "stemmed"
        and named.hash == resolve_config('{"bm25_stemming": true}', settings).hash
    )
    with pytest.raises(ValueError, match="unknown retriever preset"):
        resolve_config("no-such-preset", settings)
    with pytest.raises(ValueError, match=r"--config missing\.json: cannot read the file"):
        resolve_config("missing.json", settings)


def test_the_factory_passes_config_and_embeddings_to_get_retriever(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[str, dict[str, Any]]] = []

    def fake_get_retriever(
        method: str, settings: Settings, config: Any = None, **kwargs: Any
    ) -> FakeRetriever:
        seen.append((method, {**kwargs, "config": config}))
        return FakeRetriever()

    monkeypatch.setattr("app.rag.retrievers.get_retriever", fake_get_retriever)
    embeddings = object()
    factory = runner.make_retriever_factory(settings, embeddings)  # type: ignore[arg-type]
    custom = resolve_config('{"bm25_stemming": true}', settings)

    first = factory(custom, "hybrid")
    assert factory(custom, "hybrid") is first  # built once
    factory(custom, "bm25")

    assert seen == [
        ("hybrid", {"embeddings": embeddings, "config": custom.value}),
        ("bm25", {"config": custom.value}),
    ]


class CountingEmbeddings(Embeddings):
    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        return [[float(len(t)), 0.5] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        raise AssertionError("queries are embedded in batches")


def test_query_embeddings_are_batched_and_cached_on_disk(settings: Settings) -> None:
    base = CountingEmbeddings()
    path = settings.data_dir / "eval_cache" / "query_embeddings.jsonl"
    embeddings = CachedQueryEmbeddings("model-a", PredictionCache(path), lambda: base)

    assert embeddings.prefetch(["one", "three", "one"]) == 2
    assert embeddings.prefetch(["one", "three"]) == 0
    assert base.batches == [["one", "three"]]
    assert embeddings.embed_query("three") == [5.0, 0.5]
    assert embeddings.requests == 1 and embeddings.tokens > 0

    reopened = CachedQueryEmbeddings("model-a", PredictionCache(path), lambda: base)
    assert reopened.embed_query("one") == [3.0, 0.5]
    assert reopened.embed_query("seven") == [5.0, 0.5]  # a miss is embedded on its own
    assert base.batches == [["one", "three"], ["seven"]]
    other_model = CachedQueryEmbeddings("model-b", PredictionCache(path), lambda: base)
    assert other_model.missing(["one"]) == ["one"]
