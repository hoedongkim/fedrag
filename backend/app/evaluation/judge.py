"""LLM-as-judge for the cached answers: claims, citations, required facts and refusals.

Our own implementation (Ragas does not install next to the pinned openai / langchain-openai
stack). The judge model is configurable and defaults to DEFAULT_JUDGE_MODEL, a larger model than
the generator, to limit self-preference. One strict structured-output call per answer
(judge_prompts) returns claims with verdicts, one verdict per citation check and per required
fact, and the response type. segment_answer() finds the citation checks in code: one per marker,
pairing the cited passage with the text the marker is attached to (from the previous marker in
the sentence, or the sentence start, up to the marker; a marker after the sentence's period
belongs to that sentence). A reply that skips, repeats or invents an id is asked again, up to
MAX_ATTEMPTS calls. Judgments are cached in data/eval_cache/judgments.jsonl, keyed by the judge
model, reasoning effort, prompt label and the rendered prompt (so an edited answer, reference or
fact is judged again); an answer written without an LLM call (no passage passed the filters) is
graded in code as a decline with no claims.

Metrics per answer (score_judgment):

- faithfulness = supported claims / claims, every claim checked against all the passages the
  answer cited (None without claims);
- citation_accuracy = citation checks judged supports or partially_supports / checks (None
  without citations); citation_accuracy_strict counts supports only. Partial support is expected
  when several markers share one clause, each passage covering its part;
- answer_correctness = required facts covered / required facts (answerable questions with facts);
- refusal_correct (unanswerable questions): the answer declines and makes no claim the cited
  passages do not support (no fabrication);
- false_refusal (answerable questions): the answer declines.

Validation hooks: export_sample() writes a stratified sample of judged answers for independent
annotators, blind (judge verdicts in a separate key file); agreement() scores one annotator's
labels against the judge's, and annotator_agreement() two annotators' labels against each other.
An annotations file is either the filled sample (one annotator, its labels in the items' "human"
fields) or {"annotators": [{"id", "annotator", "items"}, ...]} for several (annotator_sets()).
"""

import json
import logging
import math
import random
import re
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import SecretStr, ValidationError

from app.core.config import Settings
from app.evaluation import judge_prompts
from app.evaluation.judge_prompts import AnswerJudgment, judge_messages
from app.evaluation.questions import EvalQuestion
from app.evaluation.runner import _map, cache_dir, now_iso, usage_row
from app.rag.answer import LLMNotConfiguredError
from app.rag.chunking import sentence_starts
from app.rag.prompts import format_passages
from app.rag.types import Chunk, RetrievedChunk
from app.stance.llm import (
    MESSAGE_OVERHEAD_TOKENS,
    CostEstimate,
    LLMReplyError,
    PredictionCache,
    Price,
    Usage,
    cache_key,
    count_tokens,
    price_for,
)

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_MODEL = "gpt-5.5"
DEFAULT_REASONING_EFFORT = "low"
MAX_OUTPUT_TOKENS = 8192  # reasoning plus the JSON reply: a cost ceiling per call
TIMEOUT_S = 120.0
MAX_ATTEMPTS = 3  # calls per answer: the first, plus two retries on an unusable reply
JUDGE_CACHE = "judgments.jsonl"
USAGE_KIND = "judge"
# Output tokens per call that estimate() assumes (reasoning, billed as output, plus the JSON).
# The validation call (gpt-5.5, reasoning low, a two-sentence answer with 2 claims, 1 citation
# and 3 facts) used 427, 179 of them reasoning; longer answers list more claims, so 1500 leaves
# room for answers several times that size.
JUDGE_OUTPUT_TOKENS_PER_CALL = 1500
# USD per million tokens, standard tier, from https://developers.openai.com/api/docs/pricing
# (checked October 13, 2026; gpt-5.5 for inputs under 272K tokens). app.stance.llm.PRICES has the
# generator's price.
JUDGE_PRICES: dict[str, Price] = {"gpt-5.5": Price(input=5.00, cached_input=0.50, output=30.00)}
ANSWER_METRICS = (
    "faithfulness",
    "citation_accuracy",
    "citation_accuracy_strict",
    "answer_correctness",
    "refusal_correct",
    "false_refusal",
)
# A final answer's markers are renumbered and contiguous ("[1][3]"); leading spaces go with them.
_MARKER_RUN = re.compile(r"[ \t]*((?:\[\d{1,2}\])+)")
_NUMBER = re.compile(r"\d+")
_LINE_BREAK = re.compile(r"\n\s*")


def model_usd(model: str, input_tokens: int, cached_input_tokens: int, output_tokens: int) -> float:
    """Dollar cost of the tokens for a judge or generator model; NaN for an unknown price."""
    price = JUDGE_PRICES.get(model) or JUDGE_PRICES.get(re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model))
    price = price or price_for(model)
    if price is None:
        return math.nan
    uncached = input_tokens - cached_input_tokens
    total = uncached * price.input + cached_input_tokens * price.cached_input
    return (total + output_tokens * price.output) / 1_000_000


# --- Citation checks ------------------------------------------------------------------------------


@dataclass(frozen=True)
class CitationUnit:
    """One marker of an answer: passage [passage] cited for `text` in sentence `sentence`."""

    passage: int  # citation number: [k] is the answer's k-th cited passage
    sentence: int  # 1-based
    text: str  # what the marker is attached to


def segment_answer(
    answer: str, passages: int | None = None
) -> tuple[list[str], list[CitationUnit]]:
    """The answer's sentences (numbered S1.. in order, markers kept) and one CitationUnit per
    marker, in order. Sentences end at sentence punctuation (app.rag.chunking.sentence_starts on
    the marker-free text) and at line breaks. Markers naming no passage (above `passages`) are
    skipped."""
    pieces: list[str] = []
    runs: list[tuple[int, str, list[int]]] = []  # (offset in the marker-free text, the run as
    # written with its leading spaces, the passage numbers it names)
    position = length = 0
    for match in _MARKER_RUN.finditer(answer):
        pieces.append(answer[position : match.start()])
        length += match.start() - position
        numbers = [int(n) for n in _NUMBER.findall(match.group(1))]
        valid = [n for n in numbers if passages is None or 1 <= n <= passages]
        runs.append((length, match.group(0), valid))
        position = match.end()
    pieces.append(answer[position:])
    text = "".join(pieces)
    breaks = {*sentence_starts(text), *(m.end() for m in _LINE_BREAK.finditer(text))}
    starts = [0, *sorted(b for b in breaks if 0 < b < len(text))]
    ends = [*starts[1:], len(text)]
    by_sentence: dict[int, list[tuple[int, str, list[int]]]] = defaultdict(list)
    for run in runs:  # a marker after a sentence's period belongs to that sentence
        by_sentence[bisect_right(starts, run[0]) - 1].append(run)
    sentences: list[str] = []
    units: list[CitationUnit] = []
    for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
        mine = by_sentence.get(index, [])
        shown = text[start:end]
        for offset, written, _ in reversed(mine):  # put the markers back, last first
            shown = shown[: offset - start] + written + shown[offset - start :]
        if not shown.strip():
            continue
        sentences.append(shown.strip())
        previous = start
        for offset, _, numbers in mine:
            attached = text[previous:offset].strip().lstrip(",;: ")
            if not re.search(r"\w", attached):  # a marker right after another, or at the start
                attached = text[start:end].strip()
            units.extend(CitationUnit(n, len(sentences), attached) for n in numbers)
            previous = offset
    return sentences, units


# --- Judge inputs ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class JudgeInput:
    """Everything one judgment needs, and the prompt built from it."""

    item_id: str  # "config/method/question_id"
    question: EvalQuestion
    record: Mapping[str, Any]  # the cached answer
    sentences: list[str]
    units: list[CitationUnit]
    passages: str  # the cited passages, numbered as the answer cites them
    messages: list[BaseMessage]

    @property
    def llm_called(self) -> bool:
        return bool(self.record.get("llm_called", True))


def item_id(record: Mapping[str, Any]) -> str:
    return f"{record['config']}/{record['method']}/{record['question_id']}"


def judge_input(
    question: EvalQuestion, record: Mapping[str, Any], chunks: Mapping[str, Chunk]
) -> JudgeInput:
    """The JudgeInput of one cached answer; ValueError when a cited chunk is not in `chunks` (the
    index changed since the answer was written)."""
    missing = [chunk_id for chunk_id in record["citations"] if chunk_id not in chunks]
    if missing:
        raise ValueError(
            f"{item_id(record)}: cited chunks {missing} are not in the search index; was it "
            "rebuilt after the answer was written?"
        )
    cited = [RetrievedChunk(chunks[chunk_id], 0.0) for chunk_id in record["citations"]]
    sentences, units = segment_answer(record["answer"], len(cited))
    passages = format_passages(cited)
    messages = judge_messages(
        question=question.question,
        as_of=question.as_of.isoformat() if question.as_of else "no cutoff",
        passages=passages,
        sentences=sentences,
        citations=[(u.passage, u.sentence, u.text) for u in units],
        reference=question.reference_answer,
        facts=list(question.required_facts) if question.answerable else [],
        answerable=question.answerable,
    )
    return JudgeInput(item_id(record), question, record, sentences, units, passages, messages)


def alignment_problems(judgment: AnswerJudgment, inp: JudgeInput) -> list[str]:
    """Why a judgment cannot be used: verdict ids that are not exactly 1..n for the citation
    checks and the required facts, or a claim citing a sentence that does not exist."""
    problems = []
    facts = len(inp.question.required_facts) if inp.question.answerable else 0
    for name, items, expected in (
        ("citations", judgment.citations, len(inp.units)),
        ("facts", judgment.facts, facts),
    ):
        ids = sorted(item.id for item in items)
        if ids != list(range(1, expected + 1)):
            problems.append(f"{name} ids {ids} are not 1..{expected}")
    bad = [c.sentence for c in judgment.claims if not 1 <= c.sentence <= len(inp.sentences)]
    if bad:
        problems.append(f"claims cite sentences {bad} of {len(inp.sentences)}")
    return problems


def no_call_judgment(inp: JudgeInput) -> AnswerJudgment:
    """The grading of an answer written without an LLM call (no passage passed the filters): a
    decline, no claims or citations, every required fact missing."""
    facts = inp.question.required_facts if inp.question.answerable else ()
    return AnswerJudgment(
        claims=[],
        citations=[],
        facts=[
            judge_prompts.FactVerdict(id=i, reason="No passage was retrieved.", verdict="missing")
            for i in range(1, len(facts) + 1)
        ],
        response_type="declines",
    )


# --- The judge ------------------------------------------------------------------------------------


def get_judge_llm(settings: Settings, model: str, reasoning_effort: str) -> ChatOpenAI:
    """The judge chat model, cached per model, effort and key; LLMNotConfiguredError without a
    key."""
    if settings.openai_api_key is None:
        raise LLMNotConfiguredError
    return _chat_model(model, reasoning_effort, settings.openai_api_key)


@lru_cache(maxsize=4)
def _chat_model(model: str, reasoning_effort: str, api_key: SecretStr) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        reasoning_effort=reasoning_effort,
        max_tokens=MAX_OUTPUT_TOKENS,
        timeout=TIMEOUT_S,
        max_retries=2,
    )


def prompt_tokens(messages: Sequence[BaseMessage]) -> int:
    """Input tokens of one judge request: the messages plus the output schema it sends."""
    schema = json.dumps(AnswerJudgment.model_json_schema())
    content = sum(count_tokens(str(m.content)) + MESSAGE_OVERHEAD_TOKENS for m in messages)
    return content + 3 + count_tokens(schema)


class AnswerJudge:
    """Grades answers with one structured-output call each (see the module docstring).

    `usage` (or a new one with `max_calls`) counts calls and tokens and refuses calls past its
    max_calls. The chat model is created on the first uncached judgment, so a fully cached run
    needs no API key.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        model: str = DEFAULT_JUDGE_MODEL,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        llm: BaseChatModel | None = None,
        cache: PredictionCache | None = None,
        usage: Usage | None = None,
        max_concurrency: int = 1,
    ) -> None:
        self.settings = settings
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.prompt_version = judge_prompts.prompt_label()
        self.cache = cache or PredictionCache(cache_dir(settings) / JUDGE_CACHE)
        self.usage = usage if usage is not None else Usage()
        self.max_concurrency = max_concurrency
        self._llm = llm
        self._structured: Runnable[Any, dict[str, Any]] | None = None

    def info(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "prompt_version": self.prompt_version,
        }

    def key(self, inp: JudgeInput) -> str:
        contents = [str(m.content) for m in inp.messages]
        return cache_key("judge", self.model, self.reasoning_effort, self.prompt_version, *contents)

    def judgment(self, inp: JudgeInput) -> AnswerJudgment | None:
        """The cached (or code-made) judgment of `inp`, or None when it has not been judged."""
        if not inp.llm_called:
            return no_call_judgment(inp)
        hit = self.cache.get(self.key(inp))
        if hit is None:
            return None
        try:
            return AnswerJudgment.model_validate(hit["judgment"])
        except (ValidationError, KeyError):
            logger.warning("%s: ignoring an invalid cached judgment", inp.item_id)
            return None

    def cached_record(self, inp: JudgeInput) -> dict[str, Any] | None:
        return self.cache.get(self.key(inp)) if inp.llm_called else None

    def pending(self, inputs: Sequence[JudgeInput]) -> list[JudgeInput]:
        """The inputs that need a call: answered by the LLM and not judged yet (deduplicated)."""
        seen: set[str] = set()
        todo = []
        for inp in inputs:
            key = self.key(inp)
            if inp.llm_called and key not in seen and self.judgment(inp) is None:
                seen.add(key)
                todo.append(inp)
        return todo

    def estimate(self, inputs: Sequence[JudgeInput]) -> CostEstimate:
        """What judging `inputs` would cost, from the prompts it would send; no calls."""
        todo = self.pending(inputs)
        input_tokens = sum(prompt_tokens(inp.messages) for inp in todo)
        output_tokens = len(todo) * JUDGE_OUTPUT_TOKENS_PER_CALL
        return CostEstimate(
            items=len(inputs),
            cached=len(inputs) - len(todo),
            calls=len(todo),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            usd=model_usd(self.model, input_tokens, 0, output_tokens),
        )

    def judge(self, inputs: Sequence[JudgeInput]) -> list[AnswerJudgment]:
        """The judgment of every input, calling the judge for the uncached ones."""
        todo = self.pending(inputs)
        if todo:
            self.usage.require(len(todo))
            self._structured_llm()  # LLMNotConfiguredError before the first call
        _map(self._judge, todo, self.max_concurrency)
        judgments = [self.judgment(inp) for inp in inputs]
        if any(j is None for j in judgments):  # unreachable unless the cache changes meanwhile
            raise LLMReplyError("a judgment is missing from the cache")
        return [j for j in judgments if j is not None]

    def _judge(self, inp: JudgeInput) -> None:
        spent: dict[str, int] = {}  # tokens of every attempt, stored with the judgment
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self.usage.reserve()
            reply = self._structured_llm().invoke(inp.messages)
            metadata = getattr(reply.get("raw"), "usage_metadata", None)
            self.usage.record(USAGE_KIND, 1, metadata)  # type: ignore[arg-type]
            row = usage_row(metadata)
            spent = {name: spent.get(name, 0) + value for name, value in row.items()}
            parsed = reply.get("parsed")
            if reply.get("parsing_error") is None and isinstance(parsed, AnswerJudgment):
                problems = alignment_problems(parsed, inp)
                if not problems:
                    self.cache.put(
                        self.key(inp),
                        {
                            "item_id": inp.item_id,
                            "judgment": parsed.model_dump(),
                            **self.info(),
                            "attempts": attempt,
                            "usage": {**spent, "calls": attempt},
                            "created_at": now_iso(),
                        },
                    )
                    return
                logger.warning(
                    "%s: judgment %d/%d unusable: %s",
                    inp.item_id,
                    attempt,
                    MAX_ATTEMPTS,
                    "; ".join(problems),
                )
            else:
                logger.warning(
                    "%s: reply %d/%d unparseable: %s",
                    inp.item_id,
                    attempt,
                    MAX_ATTEMPTS,
                    reply.get("parsing_error"),
                )
        raise LLMReplyError(f"{inp.item_id}: {MAX_ATTEMPTS} judge replies in a row were unusable")

    def _structured_llm(self) -> Runnable[Any, dict[str, Any]]:
        if self._structured is None:
            llm = self._llm
            if llm is None:
                llm = get_judge_llm(self.settings, self.model, self.reasoning_effort)
            self._structured = llm.with_structured_output(
                AnswerJudgment, method="json_schema", strict=True, include_raw=True
            )
        return self._structured


# Labels that count as a pass: in citation_accuracy and the API's booleans, and when agreement
# with an annotator is collapsed to two classes.
PASSING = {
    "claims": {"supported"},
    "citations": {"supports", "partially_supports"},
    "facts": {"covered"},
    "response_type": {"answers", "partial"},
}


# --- Scores ---------------------------------------------------------------------------------------


def score_judgment(
    judgment: AnswerJudgment, question: EvalQuestion, citation_checks: int
) -> dict[str, Any]:
    """The answer metrics of one judgment (see the module docstring) and the counts behind them."""
    claims = judgment.claims
    supported = sum(c.verdict == "supported" for c in claims)
    citations = [c.verdict for c in judgment.citations]
    full = citations.count("supports")
    partial = citations.count("partially_supports")
    facts = [f.verdict for f in judgment.facts]
    n_facts = len(question.required_facts) if question.answerable else 0
    declines = judgment.response_type == "declines"
    return {
        "faithfulness": supported / len(claims) if claims else None,
        "citation_accuracy": (full + partial) / citation_checks if citation_checks else None,
        "citation_accuracy_strict": full / citation_checks if citation_checks else None,
        "answer_correctness": facts.count("covered") / n_facts if n_facts else None,
        "refusal_correct": (
            float(declines and supported == len(claims)) if not question.answerable else None
        ),
        "false_refusal": float(declines) if question.answerable else None,
        "response_type": judgment.response_type,
        "claims": len(claims),
        "claims_supported": supported,
        "claims_contradicted": sum(c.verdict == "contradicted" for c in claims),
        "citation_checks": citation_checks,
        "citations_supporting": full,
        "citations_partial": partial,
        "facts": n_facts,
        "facts_covered": facts.count("covered"),
        "facts_contradicted": facts.count("contradicted"),
    }


def verdicts(judgment: AnswerJudgment, inp: JudgeInput) -> dict[str, Any]:
    """The judge's verdicts on one answer and its scores, as the API's JudgeVerdicts (pass/fail
    booleans), each verdict also carrying the judge's own label, reason and sentence."""
    scores = score_judgment(judgment, inp.question, len(inp.units))
    cited_in: dict[int, list[int]] = defaultdict(list)  # sentence -> citation numbers
    for unit in inp.units:
        if unit.passage not in cited_in[unit.sentence]:
            cited_in[unit.sentence].append(unit.passage)
    facts = inp.question.required_facts if inp.question.answerable else ()
    checks = sorted(judgment.citations, key=lambda c: c.id)
    declined = judgment.response_type == "declines"
    return {
        "faithfulness": scores["faithfulness"],
        "citation_accuracy": scores["citation_accuracy"],
        "answer_correctness": scores["answer_correctness"],
        "refusal_correct": _flag(scores["refusal_correct"]),
        "false_refusal": _flag(scores["false_refusal"]),
        "claims": [
            {
                "claim": c.claim,
                "citations": sorted(cited_in.get(c.sentence, [])),
                "supported": c.verdict == "supported",
                "reason": c.reason,
                "verdict": c.verdict,
                "sentence": c.sentence,
            }
            for c in judgment.claims
        ],
        "citations": [
            {
                "n": unit.passage,
                "sentence": unit.text,
                "supported": check.verdict in PASSING["citations"],
                "reason": check.reason,
                "verdict": check.verdict,
            }
            for check, unit in zip(checks, inp.units, strict=False)
        ],
        "facts": [
            {
                "fact": facts[f.id - 1],
                "covered": f.verdict == "covered",
                "reason": f.reason,
                "verdict": f.verdict,
            }
            for f in sorted(judgment.facts, key=lambda f: f.id)
            if 1 <= f.id <= len(facts)
        ],
        "refusal": {
            "declined": declined,
            "fabricated": scores["claims_supported"] < scores["claims"],
            "reason": f"response type: {judgment.response_type}",
        },
        "response_type": judgment.response_type,
    }


def _flag(value: float | None) -> bool | None:
    return None if value is None else bool(value)


# --- Validation hooks -----------------------------------------------------------------------------

LABELS = {
    "claims": list(judge_prompts.ClaimVerdict.model_fields["verdict"].annotation.__args__),
    "citations": list(judge_prompts.CitationVerdict.model_fields["verdict"].annotation.__args__),
    "facts": list(judge_prompts.FactVerdict.model_fields["verdict"].annotation.__args__),
    "response_type": ["answers", "partial", "declines"],
}
ANNOTATION_INSTRUCTIONS = (
    "Fill in every null 'human' field with one of the labels listed under 'labels', judging each "
    "item from the question, the cited passages and the answer only (not from your own knowledge "
    "of the FOMC). claims: is the claim supported by the cited passages taken together "
    "(supported), contradicted by one of them, or neither (unsupported)? A different holder "
    "(Committee, members, participants, staff), a stronger or weaker quantifier, or a different "
    "number or date is not supported. citations: does passage [n] support the quoted text its "
    "marker is attached to (supports), some of its claims (partially_supports), or none "
    "(does_not_support)? facts: does the answer state the required fact (covered), state "
    "something incompatible (contradicted), or neither (missing)? response_type: does the answer "
    "answer the question (answers), answer part of it and say what is missing (partial), or say "
    "the passages do not answer it (declines)? The judge's verdicts are in a separate key file: "
    "do not open it before you finish. Save the filled file as rag_judge_annotations.json next "
    "to this one."
)


def sample(inputs: Sequence[JudgeInput], n: int, seed: int) -> list[JudgeInput]:
    """A stratified sample of n judged answers: inputs grouped by retrieval method, each group
    shuffled with random.Random(seed), then drawn round-robin, so every method is represented."""
    groups: dict[str, list[JudgeInput]] = defaultdict(list)
    for inp in sorted(inputs, key=lambda i: i.item_id):
        groups[str(inp.record["method"])].append(inp)
    rng = random.Random(seed)
    for name in sorted(groups):
        rng.shuffle(groups[name])
    chosen: list[JudgeInput] = []
    while len(chosen) < n and any(groups.values()):
        for name in sorted(groups):
            if groups[name] and len(chosen) < n:
                chosen.append(groups[name].pop(0))
    return chosen


def annotation_item(inp: JudgeInput, judgment: AnswerJudgment) -> tuple[dict, dict]:
    """The blind item for the annotator (labels null) and the judge's verdicts for the key."""
    q = inp.question
    blind = {
        "item_id": inp.item_id,
        "question": q.question,
        "as_of": q.as_of.isoformat() if q.as_of else None,
        "cited_passages": inp.passages,
        "answer_sentences": [f"S{n}: {s}" for n, s in enumerate(inp.sentences, start=1)],
        "reference_answer": q.reference_answer,
        "claims": [
            {"id": i, "sentence": c.sentence, "claim": c.claim, "human": None}
            for i, c in enumerate(judgment.claims, start=1)
        ],
        "citations": [
            {"id": i, "passage": u.passage, "sentence": u.sentence, "text": u.text, "human": None}
            for i, u in enumerate(inp.units, start=1)
        ],
        "facts": [
            {"id": i, "fact": fact, "human": None}
            for i, fact in enumerate(q.required_facts if q.answerable else (), start=1)
        ],
        "response_type": {"human": None},
    }
    key = {
        "item_id": inp.item_id,
        "claims": [{"id": i, "verdict": c.verdict} for i, c in enumerate(judgment.claims, 1)],
        "citations": [{"id": c.id, "verdict": c.verdict} for c in judgment.citations],
        "facts": [{"id": f.id, "verdict": f.verdict} for f in judgment.facts],
        "response_type": judgment.response_type,
    }
    return blind, key


def export_sample(
    inputs: Sequence[JudgeInput], judgments: Sequence[AnswerJudgment], *, n: int, seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The annotation file (instructions, labels, blind items) and its key (judge verdicts and
    the judge's identity are added by the caller)."""
    by_id = {inp.item_id: (inp, j) for inp, j in zip(inputs, judgments, strict=True)}
    chosen = sample([inp for inp, _ in by_id.values()], n, seed)
    pairs = [annotation_item(inp, by_id[inp.item_id][1]) for inp in chosen]
    blind = {
        "instructions": ANNOTATION_INSTRUCTIONS,
        "labels": LABELS,
        "seed": seed,
        "items": [b for b, _ in pairs],
    }
    return blind, {"seed": seed, "items": [k for _, k in pairs]}


UNIT_KINDS = ("claims", "citations", "facts")
Labels = dict[str, dict[tuple[str, int], str]]  # kind -> (item id, unit id) -> label


def judge_labels(key: Mapping[str, Any]) -> Labels:
    """The judge's labels in the key file, per kind of verdict."""
    labels: Labels = {kind: {} for kind in PASSING}
    for item in key.get("items", []):
        item_id = str(item.get("item_id"))
        for kind in UNIT_KINDS:
            for unit in item.get(kind, []):
                labels[kind][(item_id, int(unit["id"]))] = unit["verdict"]
        if item.get("response_type") is not None:
            labels["response_type"][(item_id, 0)] = item["response_type"]
    return labels


def human_labels(items: Sequence[Mapping[str, Any]]) -> Labels:
    """An annotator's labels (the filled "human" fields of annotation items), per kind."""
    labels: Labels = {kind: {} for kind in PASSING}
    for item in items:
        item_id = str(item.get("item_id"))
        for kind in UNIT_KINDS:
            for unit in item.get(kind, []):
                if unit.get("human") is not None:
                    labels[kind][(item_id, int(unit["id"]))] = unit["human"]
        human = (item.get("response_type") or {}).get("human")
        if human is not None:
            labels["response_type"][(item_id, 0)] = human
    return labels


def annotator_sets(annotations: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """[{"id", "annotator", "isolated", "items"}] of an annotations file: its "annotators" list,
    or the file itself (the filled sample of one annotator) as one set with id "annotator".
    "isolated" is true for an annotator that worked from a copy of the sample alone."""
    if not annotations:
        return []
    if "annotators" in annotations:
        return [
            {
                "id": str(entry["id"]),
                "annotator": entry.get("annotator"),
                "isolated": bool(entry.get("isolated", False)),
                "items": list(entry.get("items", [])),
            }
            for entry in annotations["annotators"]
        ]
    return [
        {
            "id": "annotator",
            "annotator": annotations.get("annotator"),
            "isolated": bool(annotations.get("isolated", False)),
            "items": list(annotations.get("items", [])),
        }
    ]


def compare_labels(first: Labels, second: Labels) -> dict[str, Any]:
    """Per kind of verdict, on the units both labelled: n, exact agreement (the three-way labels),
    and agreement and Cohen's kappa with the labels collapsed to pass / fail (PASSING); kappa is
    None when it is undefined (one class only). A kind neither labelled is None."""
    from sklearn.metrics import cohen_kappa_score

    result: dict[str, Any] = {}
    for kind in PASSING:
        shared = sorted(first[kind].keys() & second[kind].keys())
        if not shared:
            result[kind] = None
            continue
        pairs = [(first[kind][unit], second[kind][unit]) for unit in shared]
        first_pass = [a in PASSING[kind] for a, _ in pairs]
        second_pass = [b in PASSING[kind] for _, b in pairs]
        kappa = None
        if len(set(first_pass) | set(second_pass)) > 1:
            value = float(cohen_kappa_score(first_pass, second_pass))
            kappa = None if math.isnan(value) else value
        result[kind] = {
            "n": len(pairs),
            "exact_agreement": sum(a == b for a, b in pairs) / len(pairs),
            "pass_fail_agreement": sum(a == b for a, b in zip(first_pass, second_pass, strict=True))
            / len(pairs),
            "kappa": kappa,
            "disagreements": [
                {"item_id": item_id, "unit": unit, "labels": [a, b]}
                for (item_id, unit), (a, b) in zip(shared, pairs, strict=True)
                if a != b
            ],
        }
    return result


def agreement(key: Mapping[str, Any], annotations: Mapping[str, Any]) -> dict[str, Any]:
    """Judge vs one annotator (the filled sample: items with "human" labels), per kind of
    verdict, on the units both labelled: see compare_labels (labels: [judge, annotator])."""
    return compare_labels(judge_labels(key), human_labels(annotations.get("items", [])))


def annotator_agreement(
    first: Sequence[Mapping[str, Any]], second: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Two annotators' items against each other, per kind of verdict: see compare_labels."""
    return compare_labels(human_labels(first), human_labels(second))
