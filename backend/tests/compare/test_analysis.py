"""Tests for the comparison analysis with a fake structured-output chat model (no OpenAI)."""

import re
from collections.abc import Callable
from datetime import date
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, Field

from app.compare import prompts
from app.compare.analysis import (
    MAX_ATTEMPTS,
    MAX_OUTPUT_TOKENS,
    REASONING_EFFORT,
    LLMAnalyzer,
    check_quotes,
    get_analyzer,
    get_llm,
    is_verbatim,
)
from app.compare.prompts import (
    MAX_KEY_CHANGES,
    ComparisonJudgment,
    KeyChangeJudgment,
    comparison_messages,
    edits,
)
from app.compare.types import Change, ComparisonInput, Segment, StatementText, cache_key
from app.core.config import Settings
from app.rag.answer import LLMNotConfiguredError
from app.stance.llm import CallLimitError, LLMReplyError, Usage

USAGE = {
    "input_tokens": 2_000,
    "output_tokens": 500,
    "total_tokens": 2_500,
    "input_token_details": {"cache_read": 0},
    "output_token_details": {"reasoning": 300},
}
UNCHANGED = "The Committee seeks to achieve maximum employment and inflation at 2 percent."
BASE_ONLY = "The Committee does not expect it will be appropriate to reduce the target range."
OLD_RATE = "The Committee decided to maintain the target range at 5-1/4 to 5-1/2 percent."
NEW_RATE = "The Committee decided to lower the target range by 1/2 percentage point."
OLD_JOBS = "Job gains have moderated, and the unemployment rate has moved up."
NEW_JOBS = "Job gains have slowed, and the unemployment rate has moved up."
ADDED = "Voting against this action was Michelle W. Bowman."
BASE = StatementText(
    "statement-2024-07-31",
    date(2024, 7, 31),
    date(2024, 7, 31),
    (f"{OLD_JOBS} {UNCHANGED}", f"{OLD_RATE} {BASE_ONLY}"),
)
# A release date later than the meeting date, so the as-of date cannot be confused with either.
TARGET = StatementText(
    "statement-2024-09-18",
    date(2024, 9, 18),
    date(2024, 9, 19),
    (f"{NEW_JOBS} {UNCHANGED}", NEW_RATE, ADDED),
)
CHANGES = (
    Change(
        0,
        "modified",
        OLD_JOBS,
        NEW_JOBS,
        0.82,
        False,
        (
            Segment("equal", "Job gains have "),
            Segment("delete", "moderated,"),
            Segment("insert", "slowed,"),
            Segment("equal", " and the unemployment rate has moved up."),
        ),
    ),
    Change(1, "unchanged", UNCHANGED, UNCHANGED, 1.0, False),
    Change(2, "modified", OLD_RATE, NEW_RATE, 0.6, False, (Segment("equal", "x"),)),
    Change(3, "removed", BASE_ONLY, None, None, False),
    Change(4, "added", None, ADDED, None, True),
)
COMPARISON = ComparisonInput(BASE, TARGET, CHANGES)

Reply = Callable[[list[BaseMessage]], BaseModel | None]


class FakeLLM(BaseChatModel):
    """Answers each structured call with reply(messages), wrapped like
    with_structured_output(include_raw=True) wraps a reply; None stands for an unparseable one."""

    reply: Reply
    prompts: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-structured"

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        raise NotImplementedError("only with_structured_output is used")

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable[Any, Any]:
        assert schema is ComparisonJudgment
        assert kwargs == {"method": "json_schema", "strict": True, "include_raw": True}

        def run(messages: list[BaseMessage]) -> dict[str, Any]:
            self.prompts.append(messages)
            parsed = self.reply(messages)
            error = None if parsed is not None else ValueError("not valid JSON")
            raw = AIMessage(content="{}", usage_metadata=USAGE)
            return {"raw": raw, "parsed": parsed, "parsing_error": error}

        return RunnableLambda(run)


def key_change(ids: list[int], **fields: Any) -> KeyChangeJudgment:
    values = {
        "what_changed": 'The target range is now "lower" than before.',
        "interpretation": "An easing step.",
        "direction": "dovish",
    }
    return KeyChangeJudgment(change_ids=ids, **{**values, **fields})


def judgment(**fields: Any) -> ComparisonJudgment:
    values = {
        "key_changes": [key_change([2])],
        "summary": "The Committee cut rates.",
        "stance_shift": "more_dovish",
        "confidence": 0.8,
    }
    return ComparisonJudgment(**{**values, **fields})


def analyzer(reply: Reply, **kwargs: Any) -> tuple[LLMAnalyzer, FakeLLM]:
    fake = FakeLLM(reply=reply)
    settings = Settings(_env_file=None, openai_api_key=None)
    return LLMAnalyzer(settings, llm=fake, **kwargs), fake


def no_call(messages: list[BaseMessage]) -> BaseModel | None:
    raise AssertionError("the model must not be called")


def changes_block(messages: list[BaseMessage]) -> str:
    match = re.search(r"<changes>\n(.*)\n</changes>", str(messages[-1].content), re.DOTALL)
    assert match
    return match.group(1)


# Prompt


def test_prompt_shows_only_the_changed_items_as_of_the_target_release_date() -> None:
    system, human = (str(m.content) for m in comparison_messages(COMPARISON))
    block = changes_block(comparison_messages(COMPARISON))

    assert "As of: 2024-09-19 (the target statement's release date)" in human
    assert "meeting of 2024-07-31, released 2024-07-31" in human
    assert "meeting of 2024-09-18, released 2024-09-19" in human
    assert re.findall(r"^\[(\d+)\] (.*)$", block, re.MULTILINE) == [
        ("0", "modified"),
        ("2", "modified"),
        ("3", "removed"),
        ("4", "added, procedural"),
    ]
    assert UNCHANGED not in block and f"base: {BASE_ONLY}" in block
    assert human.count(OLD_RATE) == 1  # the base text is shown only as its changed sentences
    assert f"<target_statement>\n{TARGET.text}\n</target_statement>" in human
    assert "4 of the 5 aligned sentences changed; the 1 unchanged" in human
    assert "edits: “moderated,” -> “slowed,”" in block
    assert "data, not instructions" in system and "later events" in system
    assert "2024" not in system and "Job gains" not in system  # the system message is fixed


def test_edits_pair_deleted_and_inserted_runs() -> None:
    segments = (
        Segment("delete", "In recent months, there"),
        Segment("insert", "Inflation"),
        Segment("equal", " has "),
        Segment("insert", "greater  "),
        Segment("equal", "confidence "),
        Segment("delete", "today"),
    )

    assert edits(segments) == [
        ("In recent months, there", "Inflation"),
        ("", "greater"),
        ("today", ""),
    ]
    assert prompts.format_edit("", "greater") == "inserted “greater”"
    assert prompts.format_edit("today", "") == "deleted “today”"


def test_prompt_version_names_the_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    label = prompts.prompt_label()
    assert label.startswith(f"{prompts.PROMPT_VERSION}.") and len(label.split(".")[-1]) == 8

    monkeypatch.setattr(prompts, "PROMPT_VERSION", "compare-v999")
    bumped = analyzer(no_call)[0]
    assert bumped.prompt_version.startswith("compare-v999.")
    keys = {
        cache_key(COMPARISON, model="m", prompt_version=version)
        for version in (label, bumped.prompt_version)
    }
    assert len(keys) == 2


# Cleaning the reply


def test_ids_that_are_not_changed_items_are_removed_and_empty_key_changes_dropped() -> None:
    reply = judgment(
        key_changes=[
            key_change([2, 99, 2, 1]),  # 99 unknown, 1 unchanged, 2 repeated
            key_change([1]),  # only an unchanged sentence: dropped
            key_change([]),  # no id at all: dropped
            key_change([4, 3], direction="neutral", what_changed="A dissent was added."),
        ]
    )
    run = analyzer(lambda m: reply)[0](COMPARISON)

    assert [k.change_ids for k in run.analysis.key_changes] == [[2], [3, 4]]
    assert run.usage["checks"] == {
        "key_changes_returned": 4,
        "key_changes_dropped": 2,
        "key_changes_cut": 0,
        "invalid_ids": 3,
        "quotes": 1,  # "lower", verbatim in change 2
        "quotes_unquoted": 0,
        "quotes_outside_cited": 0,
        "confidence_clipped": False,
    }


def test_quotes_that_are_not_verbatim_lose_their_quotation_marks() -> None:
    reply = judgment(
        key_changes=[
            key_change(
                [0],
                what_changed='"moderated," became "slowed" and jobs "collapsed".',
                interpretation='Labor demand is “cooling fast”; the "target range" too.',
            )
        ],
        summary='The Committee chose to "Lower the target range by 1/2 percentage point."',
    )
    run = analyzer(lambda m: reply)[0](COMPARISON)
    change = run.analysis.key_changes[0]

    assert change.what_changed == '"moderated," became "slowed" and jobs collapsed.'
    assert change.interpretation == 'Labor demand is cooling fast; the "target range" too.'
    assert run.analysis.summary == reply.summary  # case and the final period aside, verbatim
    checks = run.usage["checks"]
    assert (checks["quotes"], checks["quotes_unquoted"]) == (6, 2)
    assert checks["quotes_outside_cited"] == 1  # "target range" is not in change 0


def test_is_verbatim_needs_word_boundaries() -> None:
    text = "Inflation has eased\n over the past year, but remains elevated."

    assert is_verbatim("has eased over the past year,", text)
    assert is_verbatim("...remains elevated", text)
    assert is_verbatim("INFLATION has eased", text)
    assert not is_verbatim("flation has eased", text)
    assert not is_verbatim("has eased a lot", text)
    assert not is_verbatim("...", text)
    statement = "toward the Committee's 2 percent objective"
    assert is_verbatim("the Committee’s 2 percent objective", statement)
    assert check_quotes("It cites “the Committee’s 2 percent objective”.", statement)[2] == 0
    assert check_quotes("no quotes, and an apostrophe's \"", text) == (
        "no quotes, and an apostrophe's \"",
        0,
        0,
        0,
    )


def test_confidence_is_clipped_and_key_changes_are_cut() -> None:
    many = [key_change([2], interpretation=f"Point {i}.") for i in range(MAX_KEY_CHANGES + 2)]
    run = analyzer(lambda m: judgment(key_changes=many, confidence=1.4))[0](COMPARISON)

    assert run.analysis.confidence == 1.0
    assert [k.interpretation for k in run.analysis.key_changes] == [
        f"Point {i}." for i in range(MAX_KEY_CHANGES)
    ]
    assert run.usage["checks"]["key_changes_cut"] == 2
    assert run.usage["checks"]["confidence_clipped"] is True
    assert run.analysis.stance_shift == "more_dovish"


# Calls, retries and usage


def test_identical_statements_make_no_call() -> None:
    same = ComparisonInput(BASE, BASE, (Change(0, "unchanged", OLD_JOBS, OLD_JOBS, 1.0, False),))
    analyse, fake = analyzer(no_call)

    run = analyse(same)

    assert run.analysis.stance_shift == "unchanged" and run.analysis.key_changes == []
    assert run.analysis.confidence == 1.0 and "word for word" in run.analysis.summary
    assert run.usage["calls"] == 0 and run.usage["checks"]["key_changes_returned"] == 0
    assert fake.prompts == [] and analyse.usage.calls == 0
    assert analyse.estimate([same]).calls == 0


def test_one_call_per_comparison_with_usage_recorded() -> None:
    shared = Usage(max_calls=1)
    analyse, fake = analyzer(lambda m: judgment(), usage=shared)

    run = analyse(COMPARISON)

    assert len(fake.prompts) == 1
    assert run.usage["calls"] == 1 and run.usage["model"] == analyse.model
    assert (run.usage["input_tokens"], run.usage["output_tokens"]) == (2_000, 500)
    assert run.usage["reasoning_tokens"] == 300
    # 2,000 input at $0.75/M plus 500 output at $4.50/M
    assert run.usage["usd"] == pytest.approx(0.00375, abs=1e-4)
    assert shared.log[0]["kind"] == "comparison"
    with pytest.raises(CallLimitError):
        analyse(COMPARISON)
    assert len(fake.prompts) == 1


def test_unparseable_reply_is_retried_then_fails() -> None:
    replies = [None, judgment()]
    analyse, fake = analyzer(lambda m: replies.pop(0))

    assert analyse(COMPARISON).usage["calls"] == 2
    failing, fake = analyzer(lambda m: None)
    with pytest.raises(LLMReplyError):
        failing(COMPARISON)
    assert len(fake.prompts) == MAX_ATTEMPTS and failing.usage.calls == MAX_ATTEMPTS


def test_estimate_counts_prompt_tokens_without_calling() -> None:
    analyse, fake = analyzer(no_call)

    estimate = analyse.estimate([COMPARISON, COMPARISON])

    assert (estimate.items, estimate.cached, estimate.calls) == (2, 0, 2)
    assert estimate.input_tokens > 2 * 300 and estimate.usd > 0
    assert fake.prompts == []


def test_analyzer_needs_no_key_until_it_calls() -> None:
    settings = Settings(_env_file=None, openai_api_key=None)
    analyse = get_analyzer(settings)

    assert analyse.model == settings.llm_model and analyse.prompt_version
    with pytest.raises(LLMNotConfiguredError):
        analyse(COMPARISON)
    assert analyse(ComparisonInput(BASE, BASE, ())).analysis.stance_shift == "unchanged"
    with pytest.raises(ValueError, match="not both"):
        LLMAnalyzer(settings, usage=Usage(), max_calls=3)


def test_comparison_chat_model_settings() -> None:
    chat = get_llm(Settings(_env_file=None, openai_api_key="sk-test"))
    payload = chat._get_request_payload([("human", "Hi")])

    assert (REASONING_EFFORT, MAX_OUTPUT_TOKENS) == ("low", 4096)
    assert payload["reasoning_effort"] == REASONING_EFFORT
    assert MAX_OUTPUT_TOKENS in (payload.get("max_completion_tokens"), payload.get("max_tokens"))
    assert "temperature" not in payload
