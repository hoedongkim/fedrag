"""The answer judge with a fake structured-output chat model: prompts, parsing, caching, scores
and the annotation hooks (no OpenAI)."""

from typing import Any

import pytest
from eval_fakes import CHUNKS, QUESTIONS, USAGE, FakeJudgeLLM, auto_judgment
from langchain_core.messages import BaseMessage

from app.core.config import Settings
from app.evaluation import judge as judge_module
from app.evaluation.judge import (
    JUDGE_OUTPUT_TOKENS_PER_CALL,
    MAX_ATTEMPTS,
    AnswerJudge,
    CitationUnit,
    agreement,
    export_sample,
    judge_input,
    model_usd,
    no_call_judgment,
    prompt_tokens,
    score_judgment,
    segment_answer,
    verdicts,
)
from app.evaluation.judge_prompts import (
    UNANSWERABLE_FACTS,
    AnswerJudgment,
    CitationVerdict,
    ClaimVerdict,
    FactVerdict,
    prompt_label,
)
from app.evaluation.questions import EvalQuestion
from app.stance.llm import CallLimitError, LLMReplyError, PredictionCache, Usage

D1, D2, D3, T1 = (EvalQuestion.model_validate(q) for q in QUESTIONS)
ANSWER = (
    "The Committee lowered the target range by 1/2 percentage point to 4-3/4 to 5 percent [1]. "
    "Michelle W. Bowman dissented [1], and some participants preferred a smaller cut [2][1]."
)


def record(question: EvalQuestion, answer: str = ANSWER, **fields: Any) -> dict[str, Any]:
    return {
        "question_id": question.id,
        "config": "phase3",
        "method": "hybrid_rerank",
        "answer": answer,
        "citations": ["statement-2024-09-18#0000", "minutes-2024-09-18#0001"],
        "passages": [],
        "llm_called": True,
        **fields,
    }


@pytest.mark.parametrize(
    ("answer", "sentences", "units"),
    [
        (
            ANSWER,
            [
                "The Committee lowered the target range by 1/2 percentage point to 4-3/4 to 5 "
                "percent [1].",
                "Michelle W. Bowman dissented [1], and some participants preferred a smaller cut "
                "[2][1].",
            ],
            [
                (
                    1,
                    1,
                    "The Committee lowered the target range by 1/2 percentage point to 4-3/4 "
                    "to 5 percent",
                ),
                (1, 2, "Michelle W. Bowman dissented"),
                (2, 2, "and some participants preferred a smaller cut"),
                (1, 2, "and some participants preferred a smaller cut"),
            ],
        ),
        (  # a marker after the period belongs to the sentence it follows
            "Rates rose. [1] The staff saw risks.\n\n- Item one [2]\n- Item two",
            ["Rates rose. [1]", "The staff saw risks.", "- Item one [2]", "- Item two"],
            [(1, 1, "Rates rose."), (2, 3, "- Item one")],
        ),
        ("The provided FOMC passages do not say.", ["The provided FOMC passages do not say."], []),
        ("[1] Rates rose.", ["[1] Rates rose."], [(1, 1, "Rates rose.")]),
        ("Rates rose [3].", ["Rates rose [3]."], []),  # [3] names no passage of two
    ],
    ids=["clauses", "after-period-and-lines", "no-markers", "leading-marker", "invalid"],
)
def test_segment_answer(answer: str, sentences: list[str], units: list[tuple]) -> None:
    got_sentences, got_units = segment_answer(answer, passages=2)

    assert got_sentences == sentences
    assert got_units == [CitationUnit(*u) for u in units]


def test_the_judge_sees_cited_passages_numbered_as_cited_and_the_checks() -> None:
    inp = judge_input(D1, record(D1), CHUNKS)
    system, human = (str(m.content) for m in inp.messages)

    assert "Judge only against the cited passages" in system
    assert human.startswith(f"Question: {D1.question}\nAs of: 2024-10-01\n")
    assert "[1] FOMC Statement, September 18, 2024 | released 2024-09-18\nThe Committee" in human
    assert (
        "[2] Minutes of the FOMC Meeting, September 18, 2024 | Committee Policy Actions | "
        "released 2024-10-09" in human
    )
    assert "minutes-2024-09-18#0002" not in human and "Text of" not in human  # never cited
    assert "S2: Michelle W. Bowman dissented [1], and some" in human
    assert "C3: passage [2] for S2: “and some participants preferred a smaller cut”" in human
    assert "F1: cut by 1/2 percentage point\nF2: Michelle W. Bowman dissented" in human
    assert len(inp.units) == 4 and inp.item_id == "phase3/hybrid_rerank/d1"


def test_optional_facts_are_not_sent_to_the_judge() -> None:
    with_optional = EvalQuestion.model_validate(
        {**D1.model_dump(mode="json"), "optional_facts": ["the vote was 11-1"]}
    )
    inp = judge_input(with_optional, record(with_optional), CHUNKS)
    human = str(inp.messages[-1].content)

    assert "F1: cut by 1/2 percentage point\nF2: Michelle W. Bowman dissented\n" in human
    assert "11-1" not in human and "F3:" not in human
    assert inp.messages == judge_input(D1, record(D1), CHUNKS).messages


def test_a_cited_chunk_missing_from_the_index_is_an_error() -> None:
    with pytest.raises(ValueError, match="not in the search index"):
        judge_input(D1, record(D1, citations=["minutes-2099-01-01#0001"]), CHUNKS)


def test_an_unanswerable_question_has_no_facts_and_a_reference_saying_so() -> None:
    answer = "The provided FOMC passages do not cover 1979."
    inp = judge_input(D3, record(D3, answer, citations=[]), CHUNKS)
    human = str(inp.messages[-1].content)

    assert UNANSWERABLE_FACTS in human
    assert "The corpus starts in 2015." in human
    assert "(none: the answer cites no passage)" in human
    assert "(none: the answer has no citation markers)" in human


def judge_with(settings: Settings, llm: FakeJudgeLLM, **kwargs: Any) -> AnswerJudge:
    cache = PredictionCache(settings.data_dir / "eval_cache" / "judgments.jsonl")
    return AnswerJudge(settings, llm=llm, cache=cache, **kwargs)


def test_judgments_are_parsed_cached_and_not_asked_again(settings: Settings) -> None:
    llm = FakeJudgeLLM()
    judge = judge_with(settings, llm)
    inputs = [judge_input(D1, record(D1), CHUNKS), judge_input(D2, record(D2), CHUNKS)]

    judgments = judge.judge(inputs)

    assert len(llm.calls) == 2
    assert [len(j.citations) for j in judgments] == [4, 4]
    assert judge.usage.calls == 2 and judge.usage.input_tokens == 2 * USAGE["input_tokens"]
    stored = judge.cached_record(inputs[0])
    assert stored is not None
    assert stored["model"] == "gpt-5.5" and stored["prompt_version"] == prompt_label()
    assert stored["usage"]["calls"] == 1 and stored["usage"]["output_tokens"] == 200

    again_llm = FakeJudgeLLM()
    again = judge_with(settings, again_llm)
    assert again.pending(inputs) == []
    assert again.judge(inputs) == judgments
    assert again_llm.calls == []
    # Another judge model is another cache entry.
    other = judge_with(settings, FakeJudgeLLM(), model="gpt-other")
    assert other.pending(inputs) == inputs


def test_an_edited_reference_is_judged_again(settings: Settings) -> None:
    judge = judge_with(settings, FakeJudgeLLM())
    judge.judge([judge_input(D1, record(D1), CHUNKS)])
    edited = EvalQuestion.model_validate({**QUESTIONS[0], "required_facts": ["a new fact"]})

    assert judge.pending([judge_input(edited, record(edited), CHUNKS)]) != []


def test_a_misaligned_reply_is_asked_again(settings: Settings) -> None:
    replies = iter(["skip", "bad-sentence", "good"])

    def reply(messages: list[BaseMessage]) -> AnswerJudgment | None:
        good = auto_judgment(messages)
        kind = next(replies)
        if kind == "skip":
            return good.model_copy(update={"citations": good.citations[1:]})
        if kind == "bad-sentence":
            claim = ClaimVerdict(sentence=9, claim="x", reason="x", verdict="supported")
            return good.model_copy(update={"claims": [claim]})
        return good

    llm = FakeJudgeLLM(reply=reply)
    judge = judge_with(settings, llm)
    inp = judge_input(D1, record(D1), CHUNKS)

    [judgment] = judge.judge([inp])

    assert len(llm.calls) == 3 and len(judgment.citations) == 4
    stored = judge.cached_record(inp)
    assert stored is not None and stored["attempts"] == 3
    assert stored["usage"]["calls"] == 3 and stored["usage"]["input_tokens"] == 3000


def test_unusable_replies_give_up_after_max_attempts(settings: Settings) -> None:
    llm = FakeJudgeLLM(reply=lambda messages: None)
    judge = judge_with(settings, llm)

    with pytest.raises(LLMReplyError, match="unusable"):
        judge.judge([judge_input(D1, record(D1), CHUNKS)])
    assert len(llm.calls) == MAX_ATTEMPTS


def test_the_call_limit_is_checked_before_any_call(settings: Settings) -> None:
    llm = FakeJudgeLLM()
    judge = judge_with(settings, llm, usage=Usage(max_calls=1))
    inputs = [judge_input(q, record(q), CHUNKS) for q in (D1, D2)]

    with pytest.raises(CallLimitError, match="2 calls needed but only 1"):
        judge.judge(inputs)
    assert llm.calls == []


def test_an_answer_written_without_a_call_is_graded_in_code(settings: Settings) -> None:
    llm = FakeJudgeLLM()
    judge = judge_with(settings, llm)
    inp = judge_input(
        D1, record(D1, "No indexed FOMC passages match.", citations=[], llm_called=False), CHUNKS
    )

    [judgment] = judge.judge([inp])

    assert llm.calls == [] and judge.estimate([inp]).calls == 0
    assert judgment == no_call_judgment(inp)
    assert judgment.response_type == "declines"
    assert [f.verdict for f in judgment.facts] == ["missing", "missing"]


def test_estimate_prices_the_judge_prompts(settings: Settings) -> None:
    judge = judge_with(settings, FakeJudgeLLM())
    inputs = [judge_input(q, record(q), CHUNKS) for q in (D1, D2)]

    estimate = judge.estimate(inputs)

    assert estimate.calls == 2 and estimate.cached == 0
    assert estimate.input_tokens == sum(prompt_tokens(i.messages) for i in inputs)
    assert estimate.output_tokens == 2 * JUDGE_OUTPUT_TOKENS_PER_CALL
    assert estimate.usd == pytest.approx(
        (estimate.input_tokens * 5.0 + estimate.output_tokens * 30.0) / 1e6
    )


def test_model_prices() -> None:
    assert model_usd("gpt-5.5", 1_000_000, 0, 0) == pytest.approx(5.0)
    assert model_usd("gpt-5.5", 1_000_000, 1_000_000, 1_000_000) == pytest.approx(30.5)
    assert model_usd("gpt-5.4-mini-2026-03-17", 0, 0, 1_000_000) == pytest.approx(4.5)
    assert model_usd("unknown-model", 1, 0, 1) != model_usd("unknown-model", 1, 0, 1)  # NaN


def judgment(
    claims: list[str], citations: list[str], facts: list[str], response: str = "answers"
) -> AnswerJudgment:
    return AnswerJudgment(
        claims=[
            ClaimVerdict(sentence=1, claim=f"c{i}", reason="r", verdict=v)  # type: ignore[arg-type]
            for i, v in enumerate(claims)
        ],
        citations=[
            CitationVerdict(id=i, reason="r", verdict=v)  # type: ignore[arg-type]
            for i, v in enumerate(citations, start=1)
        ],
        facts=[
            FactVerdict(id=i, reason="r", verdict=v)  # type: ignore[arg-type]
            for i, v in enumerate(facts, start=1)
        ],
        response_type=response,  # type: ignore[arg-type]
    )


def test_scores_of_an_answerable_question() -> None:
    graded = judgment(
        ["supported", "unsupported", "supported", "contradicted"],
        ["supports", "partially_supports", "does_not_support", "supports"],
        ["covered", "contradicted"],
    )

    scores = score_judgment(graded, D1, citation_checks=4)

    assert scores["faithfulness"] == 0.5
    assert scores["citation_accuracy"] == 0.75
    assert scores["citation_accuracy_strict"] == 0.5
    assert scores["answer_correctness"] == 0.5
    assert scores["false_refusal"] == 0.0 and scores["refusal_correct"] is None
    assert scores["claims_contradicted"] == 1 and scores["facts_contradicted"] == 1


def test_scores_of_refusals() -> None:
    clean = score_judgment(judgment([], [], [], "declines"), D3, citation_checks=0)
    assert clean["refusal_correct"] == 1.0 and clean["false_refusal"] is None
    assert clean["faithfulness"] is None and clean["citation_accuracy"] is None
    assert clean["answer_correctness"] is None
    fabricated = score_judgment(judgment(["unsupported"], [], [], "declines"), D3, 0)
    assert fabricated["refusal_correct"] == 0.0
    answered = score_judgment(judgment(["supported"], [], [], "answers"), D3, 0)
    assert answered["refusal_correct"] == 0.0
    declined = score_judgment(judgment([], [], ["missing", "missing"], "declines"), D1, 0)
    assert declined["false_refusal"] == 1.0 and declined["answer_correctness"] == 0.0


def test_verdicts_take_the_api_shape() -> None:
    from app.schemas.evaluation import JudgeVerdicts

    inp = judge_input(D1, record(D1), CHUNKS)
    graded = AnswerJudgment(
        claims=[
            ClaimVerdict(sentence=1, claim="a cut", reason="r", verdict="supported"),
            ClaimVerdict(sentence=2, claim="smaller cut", reason="r", verdict="contradicted"),
        ],
        citations=[
            CitationVerdict(id=i, reason="r", verdict=v)
            for i, v in enumerate(
                ["supports", "does_not_support", "supports", "partially_supports"], start=1
            )
        ],
        facts=[
            FactVerdict(id=2, reason="r", verdict="missing"),
            FactVerdict(id=1, reason="r", verdict="covered"),
        ],
        response_type="answers",
    )

    found = verdicts(graded, inp)

    JudgeVerdicts.model_validate(found)
    assert found["faithfulness"] == 0.5 and found["citation_accuracy"] == 0.75
    assert found["refusal_correct"] is None and found["false_refusal"] is False
    assert [(c["citations"], c["supported"]) for c in found["claims"]] == [
        ([1], True),
        ([1, 2], False),
    ]
    assert [(c["n"], c["sentence"], c["supported"]) for c in found["citations"]] == [
        (1, inp.units[0].text, True),
        (1, "Michelle W. Bowman dissented", False),
        (2, "and some participants preferred a smaller cut", True),
        (1, "and some participants preferred a smaller cut", True),
    ]
    assert [(f["fact"], f["covered"]) for f in found["facts"]] == [
        ("cut by 1/2 percentage point", True),
        ("Michelle W. Bowman dissented", False),
    ]
    assert found["refusal"] == {
        "declined": False,
        "fabricated": True,
        "reason": "response type: answers",
    }


def test_the_annotation_sample_is_blind_stratified_and_scored(settings: Settings) -> None:
    inputs = [
        judge_input(q, record(q, method=method), CHUNKS)
        for method in ("bm25", "hybrid")
        for q in (D1, D2, D3)
    ]
    judgments = [auto_judgment(i.messages) for i in inputs]

    blind, key = export_sample(inputs, judgments, n=4, seed=3)

    assert len(blind["items"]) == 4
    assert sorted(i["item_id"].split("/")[1] for i in blind["items"]) == [
        "bm25",
        "bm25",
        "hybrid",
        "hybrid",
    ]
    assert blind == export_sample(inputs, judgments, n=4, seed=3)[0]  # reproducible
    item = blind["items"][0]
    assert all(unit["human"] is None for unit in item["claims"] + item["citations"])
    assert "verdict" not in str(blind["items"])  # the verdicts are only in the key
    assert [k["item_id"] for k in key["items"]] == [i["item_id"] for i in blind["items"]]

    # The annotator agrees on everything but one citation.
    for blind_item, key_item in zip(blind["items"], key["items"], strict=True):
        verdicts = {
            kind: {v["id"]: v["verdict"] for v in key_item[kind]}
            for kind in ("claims", "citations", "facts")
        }
        for kind in ("claims", "citations", "facts"):
            for unit in blind_item[kind]:
                unit["human"] = verdicts[kind][unit["id"]]
        blind_item["response_type"]["human"] = key_item["response_type"]
    first_with_citations = next(i for i in blind["items"] if i["citations"])
    first_with_citations["citations"][0]["human"] = "does_not_support"

    scores = agreement(key, blind)

    assert scores["claims"]["exact_agreement"] == 1.0
    assert scores["claims"]["kappa"] is None  # one class only
    citations = scores["citations"]
    assert citations["exact_agreement"] == pytest.approx(1 - 1 / citations["n"])
    assert scores["response_type"]["n"] == 4
    assert agreement(key, {"items": []})["claims"] is None


def test_labels_offered_to_the_annotator_match_the_schema() -> None:
    assert judge_module.LABELS["claims"] == ["supported", "unsupported", "contradicted"]
    assert judge_module.LABELS["citations"] == [
        "supports",
        "partially_supports",
        "does_not_support",
    ]
    assert judge_module.LABELS["facts"] == ["covered", "missing", "contradicted"]
