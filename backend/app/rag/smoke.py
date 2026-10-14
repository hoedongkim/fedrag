"""Smoke check: 12 hand-labelled questions through every retriever, optionally answered too.

A sanity check that the pipeline works end to end, not the Phase 7 benchmark: n is tiny, the
labels are document ids (a hit is any passage of an expected document), and nothing is tuned on it.
"""

import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from app.db.load import read_jsonl
from app.rag.types import Retriever
from app.schemas.common import DocType, RetrievalMethod

SMOKE_FILE = Path("eval") / "smoke.jsonl"  # under settings.data_dir
HIT_K = 5
MRR_K = 10


class SmokeQuestion(BaseModel):
    """One line of data/eval/smoke.jsonl."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(description="Short identifier, e.g. q01.")
    question: str = Field(description="The question, as a user would ask it.")
    as_of: date | None = Field(default=None, description="Point-in-time cutoff, if any.")
    doc_types: list[DocType] | None = Field(default=None, description="Document-type filter.")
    expected_doc_ids: list[str] = Field(
        description="Documents that answer it (verified by reading them); empty if none does."
    )
    expect: str = Field(description="What a correct answer says, for reading generated answers.")


@dataclass(frozen=True)
class MethodResult:
    """One method's retrieval over the answerable questions."""

    method: RetrievalMethod
    ranks: dict[str, int | None]  # question id -> rank of the first expected document, or None
    leaks: int  # retrieved chunks released after the question's as_of: must be 0
    median_ms: float

    @property
    def hit_at_k(self) -> float:
        return _mean([rank is not None and rank <= HIT_K for rank in self.ranks.values()])

    @property
    def mrr(self) -> float:
        return _mean([1 / rank if rank else 0.0 for rank in self.ranks.values()])


def load_questions(path: Path) -> list[SmokeQuestion]:
    return read_jsonl(path, SmokeQuestion)


def first_relevant_rank(doc_ids: Sequence[str], expected: Sequence[str]) -> int | None:
    """1-based rank of the first doc id in `expected`, or None if there is none."""
    return next((rank for rank, d in enumerate(doc_ids, start=1) if d in expected), None)


def run_retrieval(
    method: RetrievalMethod,
    retriever: Retriever,
    questions: Sequence[SmokeQuestion],
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> MethodResult:
    """Retrieve the top MRR_K passages for every answerable question and score them."""
    ranks: dict[str, int | None] = {}
    leaks = 0
    timings = []
    for q in questions:
        if not q.expected_doc_ids:
            continue
        started = clock()
        hits = retriever.retrieve(q.question, top_k=MRR_K, as_of=q.as_of, doc_types=q.doc_types)
        timings.append((clock() - started) * 1000)
        leaks += sum(q.as_of is not None and h.chunk.release_date > q.as_of for h in hits)
        ranks[q.id] = first_relevant_rank([h.chunk.doc_id for h in hits], q.expected_doc_ids)
    return MethodResult(method, ranks, leaks, statistics.median(timings) if timings else 0.0)


def format_table(results: Sequence[MethodResult]) -> str:
    """The summary table, as Markdown so it pastes into the technical report unchanged."""
    lines = [
        f"| Method | hit@{HIT_K} | MRR@{MRR_K} | as_of leaks | median latency |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        lines.append(
            f"| `{r.method}` | {r.hit_at_k:.2f} | {r.mrr:.2f} | {r.leaks} | {r.median_ms:.0f} ms |"
        )
    return "\n".join(lines)


def format_ranks(results: Sequence[MethodResult], questions: Sequence[SmokeQuestion]) -> str:
    """Per question: the rank of the first expected document under each method (- if not found)."""
    header = "id   " + "".join(f"{r.method:>15}" for r in results) + "  question"
    lines = [header]
    for q in questions:
        if not q.expected_doc_ids:
            continue
        cells = "".join(f"{_rank(r.ranks.get(q.id)):>15}" for r in results)
        lines.append(f"{q.id:<5}{cells}  {q.question[:70]}")
    return "\n".join(lines)


def _rank(rank: int | None) -> str:
    return "-" if rank is None else str(rank)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
