"""Ask the Fed: grounded question answering over FOMC documents, whole or streamed, and the
recorded example answers that GET /api/ask/examples serves (data/eval/ask_examples.json)."""

from datetime import date, datetime
from typing import Literal, Self, get_args

from pydantic import ConfigDict, Field, field_validator, model_validator

from app.schemas.common import ApiModel, DocType, RequestModel, RetrievalMethod, SourceDocument
from app.schemas.evaluation import QuestionId, QuestionType


class AskRequest(RequestModel):
    """A question about Fed policy, optionally limited to what was public on a given date."""

    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(
        min_length=3,
        max_length=2000,
        description="Natural-language question about FOMC policy.",
        examples=["How did the Committee describe inflation risks?"],
    )
    as_of: date | None = Field(
        default=None,
        description=(
            "Point-in-time cutoff: only documents with release_date <= as_of are retrievable, and "
            "the answer is written as of that date. Null means no cutoff."
        ),
        examples=["2024-10-01"],
    )
    retrieval: RetrievalMethod = Field(
        default="hybrid_rerank", description="Retrieval strategy used to find supporting passages."
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description=(
            "Number of passages retrieved and given to the LLM; the citations are the ones the "
            "answer actually cites."
        ),
    )
    doc_types: list[DocType] | None = Field(
        default=None,
        min_length=1,
        max_length=len(get_args(DocType)),
        description="Restrict retrieval to these document types. Null means all.",
    )

    @field_validator("doc_types")
    @classmethod
    def _dedupe_doc_types(cls, value: list[DocType] | None) -> list[DocType] | None:
        return list(dict.fromkeys(value)) if value else value


class Citation(ApiModel):
    """A retrieved passage that the answer cites."""

    document: SourceDocument = Field(description="Document the passage comes from.")
    chunk_id: str = Field(
        description="Stable identifier of the passage: the doc_id, '#', and its position."
    )
    section: str | None = Field(
        description="Minutes section the passage comes from; null for statements."
    )
    snippet: str = Field(
        description=(
            "The cited passage text, trimmed at a sentence boundary to about 600 characters."
        )
    )
    passage: str | None = Field(
        default=None,
        description=(
            "The whole cited passage when snippet is only its beginning. Set on the recorded "
            "answers (GET /api/ask/examples), so the text an answer relies on can be shown in "
            "full; null when snippet is the whole passage, and in live answers."
        ),
    )
    score: float | None = Field(
        default=None,
        description=(
            "Retrieval score of the passage, higher is more relevant. Its scale depends on the "
            "retrieval method (BM25, vector similarity, fused rank or cross-encoder), so compare "
            "scores only within one response."
        ),
    )


class AskResponse(ApiModel):
    """An answer grounded in cited FOMC passages."""

    answer: str = Field(
        description=(
            "Generated answer with inline citation markers: [n] refers to citations[n - 1]. Only "
            "the model's text; notes about it are in notes. Without supporting passages it says "
            "so and has no markers."
        )
    )
    citations: list[Citation] = Field(
        description=(
            "Passages the answer cites, in order of first citation; empty when the retrieved "
            "passages do not support an answer."
        )
    )
    notes: list[str] = Field(
        description=(
            "System notes about the answer, e.g. that the newest cited passage is old relative "
            "to as_of, so later decisions may be missing; empty when there is nothing to note."
        )
    )
    retrieval: RetrievalMethod = Field(description="Retrieval strategy that was used.")
    as_of: date | None = Field(description="Point-in-time cutoff that was applied, if any.")
    model: str = Field(description="LLM that generated the answer.")
    latency_ms: float = Field(
        description="End-to-end latency of retrieval and generation in milliseconds."
    )


StreamStage = Literal["retrieving", "generating"]


class StreamStatus(ApiModel):
    """Data of a `status` event: the stage a streamed answer has reached."""

    stage: StreamStage = Field(
        description=(
            "retrieving: searching for passages; generating: the LLM is writing the answer."
        )
    )


class StreamDelta(ApiModel):
    """Data of a `delta` event: the next piece of a streamed answer."""

    text: str = Field(
        description=(
            "Answer text to append. Its [n] markers already carry their final citation numbers, "
            "and a marker is never split across deltas."
        )
    )


class StreamCitation(ApiModel):
    """Data of a `citation` event: a passage the answer cites for the first time, sent just
    before the delta that cites it."""

    n: int = Field(
        ge=1,
        description=(
            "Citation number: [n] markers refer to this passage, and it is citations[n - 1] of "
            "the final answer."
        ),
    )
    citation: Citation = Field(description="The cited passage.")


class StreamError(ApiModel):
    """Data of an `error` event: the answer failed after the stream started, which ends it."""

    status: int = Field(
        description=(
            "HTTP status POST /api/ask returns for the same failure: 502 when the OpenAI request "
            "failed, 503 when the index or the reranker became unavailable, 500 otherwise."
        )
    )
    detail: str = Field(description="Human-readable error message.")


# The data of every event POST /api/ask/stream sends; `done` carries the final AskResponse.
AskStreamEvent = StreamStatus | StreamDelta | StreamCitation | AskResponse | StreamError


class ExampleJudgement(ApiModel):
    """The RAG evaluation's LLM judge on a recorded answer: the same verdicts the RAG Performance
    page shows for it (GET /api/evaluation/questions/{id})."""

    model: str = Field(description="Judge model.")
    faithfulness: float | None = Field(
        ge=0, le=1, description="Supported claims / claims; null for an answer without claims."
    )
    citation_accuracy: float | None = Field(
        ge=0, le=1, description="Supporting citations / citations; null without citations."
    )
    answer_correctness: float | None = Field(
        ge=0,
        le=1,
        description=(
            "Required facts the answer states / required facts; null for an unanswerable question."
        ),
    )
    refusal_correct: bool | None = Field(
        description="Unanswerable question: declined without fabricating. Null if answerable."
    )
    false_refusal: bool | None = Field(
        description="Answerable question: declined anyway. Null if unanswerable."
    )
    n_claims: int = Field(ge=0, description="Claims the judge extracted from the answer.")
    n_claims_supported: int = Field(ge=0, description="Of those, supported by their passages.")
    n_facts: int = Field(ge=0, description="Required facts of the question.")
    n_facts_covered: int = Field(ge=0, description="Of those, stated by the answer.")


class AskExample(AskResponse):
    """A real answer of the production configuration, recorded during the RAG evaluation and
    served as it was given, with no LLM call: what Ask the Fed shows on the public demo, where it
    does not answer new questions.

    The AskResponse fields are the answer exactly as POST /api/ask returned it (citations
    resolved from the search index the same way); the rest says where it comes from and how the
    evaluation's judge scored it.
    """

    id: QuestionId = Field(
        description=(
            "Evaluation question id, e.g. sf-14: GET /api/evaluation/questions/{id} has its gold "
            "passages, every retrieval run and the judge's verdicts."
        )
    )
    label: str = Field(
        min_length=1, description="What the example shows, in a few words, e.g. Point in time."
    )
    description: str = Field(min_length=1, description="One sentence on why it was picked.")
    question: str = Field(min_length=3, description="The question as it was asked.")
    question_type: QuestionType = Field(description="Its type in the evaluation question set.")
    retriever_preset: str = Field(
        description="Retriever preset that found the passages (RETRIEVER_PRESET), e.g. tuned."
    )
    recorded_at: datetime = Field(description="When the answer was generated (UTC).")
    judge: ExampleJudgement | None = Field(
        description="The evaluation judge's verdicts on the answer; null if it was not judged."
    )


class AskExamplesFile(ApiModel):
    """data/eval/ask_examples.json, written by python -m app.evaluation export-ask-examples from
    the evaluation's cached answers (no API calls)."""

    description: str = Field(description="What the file holds and how it was made.")
    retriever_preset: str = Field(description="Retriever preset of every example.")
    retriever_config_hash: str = Field(
        description="Hash of that preset's behaviour, the index and the models (the cache key)."
    )
    retrieval: RetrievalMethod = Field(description="Retrieval method of every example.")
    top_k: int = Field(ge=1, description="Passages given to the LLM.")
    model: str = Field(description="LLM that wrote the answers.")
    prompt_version: str = Field(description="Version of the answer prompt.")
    benchmark_generated_at: datetime | None = Field(
        description="generated_at of the rag_benchmark.json the verdicts were read from."
    )
    examples: list[AskExample] = Field(min_length=1, description="The examples, in display order.")

    @model_validator(mode="after")
    def _ids_unique(self) -> Self:
        ids = [example.id for example in self.examples]
        if len(set(ids)) != len(ids):
            raise ValueError("example ids must be unique")
        return self
