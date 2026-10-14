"""The RAG evaluation set: loading, validation, splits, and the smoke file converted on the fly.

data/eval/rag_questions.jsonl holds one question per line:

    {"id": "...", "question": "...", "as_of": "YYYY-MM-DD" | null,
     "type": "statement_fact" | "minutes_fact" | "point_in_time" | "date_specific" | "multi_doc"
             | "paraphrase" | "unanswerable",
     "answerable": true,
     "relevant_chunks": [{"chunk_id": "minutes-2024-09-18#0012", "role": "primary" | "supporting"}],
     "relevant_doc_ids": ["minutes-2024-09-18"],
     "reference_answer": "...", "required_facts": ["...", ...], "optional_facts": ["...", ...],
     "notes": "...",
     "split": "dev" | "test"}

Chunk ids are the index's stable ids (data/index/chunks.jsonl). Retrieval metrics use the
answerable questions only; a primary chunk has relevance grade 2 and a supporting one grade 1.

validate() finds what would make a metric wrong (errors) or undefined (warnings). Errors: repeated
question ids, an unanswerable-typed question marked answerable, an answerable question without a
relevant chunk, a chunk listed twice, a chunk or document id the index does not have, a relevant
chunk released after the question's as_of (no retriever may return it: an impossible target),
and a relevant chunk whose document is not in relevant_doc_ids. Warnings: an answerable question
without required facts or a reference answer, an unanswerable one with relevant chunks, a
relevant document with no relevant chunk, and fields the schema does not know.

The 12-question smoke file (data/eval/smoke.jsonl, document labels only) converts with
from_smoke(): an expected statement becomes its single chunk, labelled primary; expected minutes
stay document labels only, so chunk-level metrics skip the questions that have none. Converted
questions are all in the dev split and have no required facts (answer correctness undefined).
"""

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal, Protocol, get_args

from pydantic import BaseModel, ConfigDict, Field, ValidationError

QUESTIONS_FILE = Path("eval") / "rag_questions.jsonl"  # under settings.data_dir
SMOKE_FILE = Path("eval") / "smoke.jsonl"

QuestionType = Literal[
    "statement_fact",
    "minutes_fact",
    "point_in_time",
    "date_specific",
    "multi_doc",
    "paraphrase",
    "unanswerable",
]
QUESTION_TYPES: tuple[QuestionType, ...] = get_args(QuestionType)
Split = Literal["dev", "test"]
SPLITS: tuple[Split, ...] = get_args(Split)
SplitChoice = Literal["dev", "test", "all"]
Role = Literal["primary", "supporting"]
GRADES: dict[Role, int] = {"primary": 2, "supporting": 1}
Format = Literal["rag_questions", "smoke"]


class QuestionSetError(ValueError):
    """The question file is missing, unreadable or fails validation."""


class RelevantChunk(BaseModel):
    """A passage that answers the question, or part of it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str = Field(min_length=1)
    role: Role
    # Indexes into required_facts that this chunk states (the set builder's verbatim check).
    facts: tuple[int, ...] = ()


class EvalQuestion(BaseModel):
    """One line of data/eval/rag_questions.jsonl. Unknown fields are kept (validate() warns)."""

    model_config = ConfigDict(extra="allow", frozen=True)

    id: str = Field(min_length=1)
    question: str = Field(min_length=3, max_length=2000)  # what AskRequest accepts
    as_of: date | None = None
    type: QuestionType
    answerable: bool
    relevant_chunks: tuple[RelevantChunk, ...] = ()
    relevant_doc_ids: tuple[str, ...] = ()
    reference_answer: str = ""
    required_facts: tuple[str, ...] = ()
    # Facts the reference answer may mention but the question does not ask for. Answer
    # correctness is scored on required_facts only; these are never sent to the judge.
    optional_facts: tuple[str, ...] = ()
    notes: str | None = None
    split: Split
    human_reviewed: bool = False  # set once the owner's review of this item is applied

    @property
    def grades(self) -> dict[str, int]:
        """chunk_id -> relevance grade (primary 2, supporting 1); the first role listed wins."""
        grades: dict[str, int] = {}
        for chunk in self.relevant_chunks:
            grades.setdefault(chunk.chunk_id, GRADES[chunk.role])
        return grades

    @property
    def fingerprint(self) -> str:
        """Short hash of what retrieval and answering see (question and as_of): cached results of
        an edited question are not reused."""
        payload = json.dumps([self.question, self.as_of.isoformat() if self.as_of else None])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


class ChunkRef(Protocol):
    """What validation needs from an indexed chunk (app.rag.types.Chunk has these)."""

    @property
    def chunk_id(self) -> str: ...
    @property
    def doc_id(self) -> str: ...
    @property
    def release_date(self) -> date: ...


@dataclass(frozen=True)
class QuestionSet:
    """The loaded questions and where they came from."""

    questions: tuple[EvalQuestion, ...]
    path: Path
    format: Format
    sha256: str  # of the file's bytes
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()  # only when loaded with check=False

    def select(
        self, split: SplitChoice = "all", ids: Sequence[str] | None = None
    ) -> list[EvalQuestion]:
        """The questions of `split` ("all" for both), limited to `ids` when given (in file
        order); QuestionSetError for an id the file does not have."""
        if ids:
            unknown = sorted(set(ids) - {q.id for q in self.questions})
            if unknown:
                raise QuestionSetError(f"unknown question ids: {', '.join(unknown)}")
        return [
            q
            for q in self.questions
            if (split == "all" or q.split == split) and (not ids or q.id in ids)
        ]

    def summary(self) -> dict[str, object]:
        """Counts for the results file: questions, answerable, per split and per type."""
        return {
            "n_questions": len(self.questions),
            "n_answerable": sum(q.answerable for q in self.questions),
            "splits": {s: sum(q.split == s for q in self.questions) for s in SPLITS},
            "types": {t: sum(q.type == t for q in self.questions) for t in QUESTION_TYPES},
        }


@dataclass
class Validation:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate(
    questions: Sequence[EvalQuestion],
    chunks: Mapping[str, ChunkRef] | None = None,
    *,
    strict: bool = True,
) -> Validation:
    """Check the set (see the module docstring); `chunks` (chunk_id -> chunk) enables the checks
    against the index. strict=False (converted smoke questions) accepts an answerable question
    labelled with documents only."""
    report = Validation()
    counts = Counter(q.id for q in questions)
    report.errors += [f"question id {i!r} appears {n} times" for i, n in counts.items() if n > 1]
    doc_chunks: dict[str, list[ChunkRef]] = {}
    for chunk in (chunks or {}).values():
        doc_chunks.setdefault(chunk.doc_id, []).append(chunk)
    for q in questions:
        _check(q, chunks, doc_chunks, strict, report)
    return report


def _check(
    q: EvalQuestion,
    chunks: Mapping[str, ChunkRef] | None,
    doc_chunks: Mapping[str, list[ChunkRef]],
    strict: bool,
    report: Validation,
) -> None:
    error, warn = report.errors.append, report.warnings.append
    if q.model_extra:
        warn(f"{q.id}: unknown fields {sorted(q.model_extra)}")
    if q.type == "unanswerable" and q.answerable:
        error(f"{q.id}: type unanswerable but answerable is true")
    if q.answerable and not q.relevant_chunks and (strict or not q.relevant_doc_ids):
        error(f"{q.id}: answerable but no relevant chunk is listed")
    if q.answerable and not q.required_facts and strict:
        warn(f"{q.id}: answerable but no required facts (answer correctness undefined)")
    if q.answerable and not q.reference_answer.strip() and strict:
        warn(f"{q.id}: answerable but no reference answer")
    if not q.answerable and q.relevant_chunks:
        warn(f"{q.id}: unanswerable but lists relevant chunks (not scored)")
    repeated = [c for c, n in Counter(c.chunk_id for c in q.relevant_chunks).items() if n > 1]
    if repeated:
        error(f"{q.id}: chunks listed more than once: {repeated}")
    chunk_docs = {c.chunk_id.split("#", 1)[0] for c in q.relevant_chunks}
    for chunk_ref in q.relevant_chunks:
        chunk = chunks.get(chunk_ref.chunk_id) if chunks is not None else None
        if chunks is not None and chunk is None:
            error(f"{q.id}: chunk {chunk_ref.chunk_id} is not in the index")
            continue
        doc_id = chunk.doc_id if chunk is not None else chunk_ref.chunk_id.split("#", 1)[0]
        if doc_id not in q.relevant_doc_ids:
            error(f"{q.id}: chunk {chunk_ref.chunk_id}'s document {doc_id} not in relevant_doc_ids")
        if chunk is not None and q.as_of is not None and chunk.release_date > q.as_of:
            error(
                f"{q.id}: chunk {chunk_ref.chunk_id} was released {chunk.release_date}, after "
                f"as_of {q.as_of}"
            )
    for doc_id in q.relevant_doc_ids:
        if chunks is not None and doc_id not in doc_chunks:
            error(f"{q.id}: document {doc_id} is not in the index")
        elif (
            chunks is not None
            and q.as_of is not None
            and min(c.release_date for c in doc_chunks[doc_id]) > q.as_of
        ):
            error(f"{q.id}: document {doc_id} was released after as_of {q.as_of}")
        if q.relevant_chunks and doc_id not in chunk_docs:
            warn(f"{q.id}: document {doc_id} has no relevant chunk listed")


def from_smoke(rows: Iterable[Mapping[str, object]]) -> list[EvalQuestion]:
    """Questions in the evaluation schema from smoke.jsonl rows (see the module docstring)."""
    questions = []
    for row in rows:
        doc_ids = [str(d) for d in row.get("expected_doc_ids") or []]  # type: ignore[attr-defined]
        statements = [d for d in doc_ids if d.startswith("statement-")]
        questions.append(
            EvalQuestion(
                id=str(row["id"]),
                question=str(row["question"]),
                as_of=row.get("as_of"),  # type: ignore[arg-type]
                type=_smoke_type(doc_ids, row.get("as_of") is not None),
                answerable=bool(doc_ids),
                # A statement is indexed whole, as chunk #0000.
                relevant_chunks=tuple(
                    RelevantChunk(chunk_id=f"{d}#0000", role="primary") for d in statements
                ),
                relevant_doc_ids=tuple(doc_ids),
                reference_answer=str(row.get("expect") or ""),
                notes="converted from smoke.jsonl: document labels only",
                split="dev",
            )
        )
    return questions


def _smoke_type(doc_ids: Sequence[str], has_as_of: bool) -> QuestionType:
    if not doc_ids:
        return "unanswerable"
    if has_as_of:
        return "point_in_time"
    kinds = {d.split("-", 1)[0] for d in doc_ids}
    if kinds == {"statement"}:
        return "statement_fact"
    if kinds == {"minutes"}:
        return "minutes_fact"
    return "multi_doc"


def read_rows(path: Path) -> list[dict[str, object]]:
    """The JSON objects of a JSONL file (blank lines skipped); QuestionSetError naming the line
    of the first unreadable one."""
    if not path.is_file():
        raise QuestionSetError(
            f"{path} not found. The evaluation set is data/eval/rag_questions.jsonl; to try the "
            "harness on the smoke questions, pass --questions data/eval/smoke.jsonl."
        )
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise QuestionSetError(f"{path.name} line {number}: not JSON ({exc})") from None
        if not isinstance(row, dict):
            raise QuestionSetError(f"{path.name} line {number}: not a JSON object")
        rows.append(row)
    return rows


def load_questions(
    path: Path, chunks: Mapping[str, ChunkRef] | None = None, *, check: bool = True
) -> QuestionSet:
    """Load the evaluation set at `path` (or a smoke file, recognized by its expected_doc_ids
    field, converted). With `check`, QuestionSetError lists every validation error; warnings are
    returned on the set."""
    rows = read_rows(path)
    smoke = bool(rows) and "expected_doc_ids" in rows[0]
    try:
        questions = from_smoke(rows) if smoke else [EvalQuestion.model_validate(r) for r in rows]
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(map(str, e['loc'])) or 'row'}: {e['msg']}"
            for e in exc.errors(include_url=False)
        )
        raise QuestionSetError(f"{path.name}: invalid question ({problems})") from None
    result = validate(questions, chunks, strict=not smoke)
    if check and result.errors:
        listed = "\n  ".join(result.errors)
        raise QuestionSetError(f"{path.name} has {len(result.errors)} problems:\n  {listed}")
    return QuestionSet(
        questions=tuple(questions),
        path=path,
        format="smoke" if smoke else "rag_questions",
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        warnings=tuple(result.warnings),
        errors=tuple(result.errors),
    )
