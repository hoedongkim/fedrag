"""Command line for the search index: python -m app.rag {index,search,ask,smoke}."""

import argparse
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, get_args

from app.core.config import Settings, get_settings
from app.schemas.common import DocType, RetrievalMethod

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

    from app.rag.smoke import SmokeQuestion
    from app.rag.types import Retriever
    from app.schemas.ask import AskRequest, AskResponse

PREVIEW_CHARS = 120


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.rag", description="Build and query the FOMC search index."
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")
    index = commands.add_parser(
        "index",
        help="chunk the documents and embed new or changed chunks into Chroma",
        description=(
            "Chunk every document, write data/index/chunks.jsonl and manifest.json, and bring the "
            "Chroma collection up to date: only new or changed chunks are embedded."
        ),
    )
    index.add_argument(
        "--source",
        choices=["db", "processed"],
        default="db",
        help="read documents from Postgres (default) or data/processed/documents.jsonl",
    )
    index.add_argument(
        "--rebuild",
        action="store_true",
        help="drop the collection and re-embed every chunk (needed to change EMBEDDING_MODEL)",
    )
    index.add_argument(
        "--dry-run",
        action="store_true",
        help="chunk and estimate the embedding cost only: no API calls, nothing written",
    )
    for name, text in (
        ("search", "print the passages a retrieval method finds"),
        ("ask", "answer a question from retrieved passages, with citations (calls the LLM)"),
    ):
        sub = commands.add_parser(name, help=text, description=text)
        sub.add_argument("query", help="the question or search text")
        sub.add_argument(
            "--method",
            choices=get_args(RetrievalMethod),
            default="hybrid_rerank",
            help="retrieval method (default: hybrid_rerank)",
        )
        sub.add_argument("--top-k", type=int, default=5, help="passages to return (default: 5)")
        sub.add_argument(
            "--preset",
            help="retriever preset, e.g. phase3 or tuned (default: RETRIEVER_PRESET, tuned)",
        )
        sub.add_argument(
            "--as-of",
            type=date.fromisoformat,
            help="only documents released on or before this date (YYYY-MM-DD)",
        )
        sub.add_argument(
            "--doc-type",
            dest="doc_types",
            nargs="+",
            action="extend",
            choices=get_args(DocType),
            help="only these document types (default: all)",
        )
    smoke = commands.add_parser(
        "smoke",
        help="SMOKE CHECK: hit@5 and MRR@10 of every method on data/eval/smoke.jsonl",
        description=(
            "Smoke check (n=12, not the Phase 7 benchmark): run each retrieval method on the "
            "hand-labelled questions and print hit@5 and MRR@10 at the document level. Retrieval "
            "embeds each question once (dense methods); --answers also calls the LLM once per "
            "question."
        ),
    )
    smoke.add_argument("--file", type=Path, help="questions file (default: data/eval/smoke.jsonl)")
    smoke.add_argument(
        "--methods",
        nargs="+",
        choices=get_args(RetrievalMethod),
        default=list(get_args(RetrievalMethod)),
        help="retrieval methods to score (default: all four)",
    )
    smoke.add_argument(
        "--answers",
        action="store_true",
        help="also generate an answer to every question (one LLM call each)",
    )
    smoke.add_argument(
        "--answer-method",
        choices=get_args(RetrievalMethod),
        default="hybrid_rerank",
        help="retrieval method for --answers (default: hybrid_rerank)",
    )
    smoke.add_argument(
        "--top-k", type=int, default=5, help="passages given to the LLM with --answers (default: 5)"
    )
    smoke.add_argument(
        "--preset",
        help="retriever preset, e.g. phase3 or tuned (default: RETRIEVER_PRESET, tuned)",
    )
    smoke.add_argument(
        "--cached-embeddings",
        action="store_true",
        help="serve query vectors from data/eval_cache/query_embeddings.jsonl (the evaluation's "
        "cache), embedding only the questions missing from it; latency then excludes the "
        "OpenAI round trip",
    )
    return parser


def main(argv: Sequence[str] | None = None, *, settings: Settings | None = None) -> int:
    """Run one command; returns the process exit code."""
    args = build_parser().parse_args(argv)
    settings = settings or get_settings()
    if getattr(args, "preset", None):
        settings = settings.model_copy(update={"retriever_preset": args.preset})
    # Imported here so --help stays fast; these pull in Chroma, LangChain and the DB driver.
    import openai
    import psycopg

    from app.db.connection import DatabaseNotConfiguredError, DatabaseUrlError
    from app.db.load import LoadError
    from app.rag.answer import LLMNotConfiguredError
    from app.rag.index import EmbeddingsNotConfiguredError, IndexBuildError, IndexNotBuiltError
    from app.rag.retrievers import RerankerUnavailableError

    commands = {"index": run_index, "search": run_search, "ask": run_ask, "smoke": run_smoke}
    try:
        return commands[args.command](settings, args)
    except (
        IndexNotBuiltError,
        IndexBuildError,
        EmbeddingsNotConfiguredError,
        RerankerUnavailableError,
        LLMNotConfiguredError,
        LoadError,
        DatabaseNotConfiguredError,
        DatabaseUrlError,
        psycopg.Error,
        openai.OpenAIError,
        ValueError,  # an unknown --preset or RETRIEVER_PRESET
    ) as exc:
        message = str(exc)
        if isinstance(exc, openai.OpenAIError):  # its message can quote part of the API key
            detail = type(exc).__name__
            if (status := getattr(exc, "status_code", None)) is not None:
                detail += f", HTTP {status}"
            message = f"OpenAI request failed ({detail})"
        print(f"{args.command}: {message}", file=sys.stderr)
        return 1


def run_index(settings: Settings, args: argparse.Namespace) -> int:
    from app.rag.index import build_index

    report = build_index(settings, args.source, rebuild=args.rebuild, dry_run=args.dry_run)
    if report.dry_run:
        print("Dry run: nothing was embedded or written.")
    return 0


def run_search(settings: Settings, args: argparse.Namespace) -> int:
    hits = _retriever(settings, args.method).retrieve(
        args.query, top_k=args.top_k, as_of=args.as_of, doc_types=args.doc_types
    )
    if not hits:
        print("No passages match these filters.")
    for rank, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        preview = " ".join(chunk.text.split())[:PREVIEW_CHARS]
        print(
            f"{rank:>2}  {hit.score:9.4f}  {chunk.doc_id:<22} released {chunk.release_date}  "
            f"{chunk.section or '-'}\n    {preview}"
        )
    return 0


def run_ask(settings: Settings, args: argparse.Namespace) -> int:
    """Answer one question with citations, as POST /api/ask would."""
    from pydantic import ValidationError

    from app.schemas.ask import AskRequest

    try:
        request = AskRequest(
            question=args.query,
            as_of=args.as_of,
            retrieval=args.method,
            top_k=args.top_k,
            doc_types=args.doc_types,
        )
    except ValidationError as exc:
        print(f"ask: {exc}", file=sys.stderr)
        return 2
    print_answer(_answer(settings, request))
    return 0


def run_smoke(settings: Settings, args: argparse.Namespace) -> int:
    """SMOKE CHECK: retrieval metrics for each method and, with --answers, generated answers."""
    from app.db.load import LoadError
    from app.rag.smoke import SMOKE_FILE, format_ranks, format_table, load_questions, run_retrieval

    path = args.file or settings.data_dir / SMOKE_FILE
    if not path.is_file():
        raise LoadError(f"{path} not found.")
    questions = load_questions(path)
    answerable = sum(bool(q.expected_doc_ids) for q in questions)
    print(
        f"SMOKE CHECK - {len(questions)} hand-labelled questions ({answerable} answerable); a "
        "sanity check, not the Phase 7 benchmark.\nDocument-level: a hit is any passage of an "
        "expected document. Unanswerable questions are left out of the metrics.\nRetriever "
        f"preset: {settings.retriever_preset} (--preset or RETRIEVER_PRESET; phase3 is the "
        "Phase 3 retriever).\n"
    )
    embeddings = None
    if args.cached_embeddings and any(method != "bm25" for method in args.methods):
        from app.evaluation.runner import query_embeddings

        embeddings = query_embeddings(settings)
        embedded = embeddings.prefetch([q.question for q in questions])
        print(
            f"Query vectors from data/eval_cache/query_embeddings.jsonl ({embedded} embedded now, "
            f"{len(questions) - embedded} cached).\n"
        )
    results = [
        run_retrieval(method, _retriever(settings, method, embeddings), questions)
        for method in args.methods
    ]
    print(format_table(results))
    print(
        "Latency is the median per question, with every query vector served from the cache: no "
        "OpenAI round trip."
        if embeddings is not None
        else "Latency is the median per question. The methods share one cached query embedding, "
        "so only the first dense-based method run pays the OpenAI round trip."
    )
    print("\nRank of the first expected document in the top 10 (- = not found):")
    print(format_ranks(results, questions))
    if args.answers:
        print_smoke_answers(settings, questions, method=args.answer_method, top_k=args.top_k)
    return 1 if any(r.leaks for r in results) else 0


def print_smoke_answers(
    settings: Settings, questions: Sequence["SmokeQuestion"], *, method: RetrievalMethod, top_k: int
) -> None:
    """Answer every question with one method and print the answers next to what to expect."""
    from langchain_core.callbacks import get_usage_metadata_callback

    from app.schemas.ask import AskRequest

    print(f"\nAnswers ({method}, top_k={top_k}): one LLM call per question.")
    with get_usage_metadata_callback() as usage:
        for q in questions:
            request = AskRequest(
                question=q.question,
                as_of=q.as_of,
                retrieval=method,
                top_k=top_k,
                doc_types=q.doc_types,
            )
            response = _answer(settings, request)
            cited = [c.document.doc_id for c in response.citations]
            late = [
                c.chunk_id
                for c in response.citations
                if q.as_of and c.document.release_date > q.as_of
            ]
            if q.expected_doc_ids:
                verdict = (
                    "cites an expected document"
                    if set(cited) & set(q.expected_doc_ids)
                    else "cites NO expected document"
                )
            else:
                verdict = "no citations" if not cited else "CITES passages (expected none)"
            print(f"\n=== {q.id}  as_of={q.as_of or '-'}  {q.question}")
            print(f"expect: {q.expect}")
            print_answer(response)
            print(f"check: {verdict}; citations after as_of: {len(late)}")
    for model, tokens in usage.usage_metadata.items():
        print(
            f"\nLLM usage ({model}): {tokens['input_tokens']:,} input + "
            f"{tokens['output_tokens']:,} output tokens for {len(questions)} questions."
        )


def print_answer(response: "AskResponse") -> None:
    print(response.answer)
    for note in response.notes:
        print(f"Note: {note}")
    if response.citations:
        print("Citations:")
    for number, citation in enumerate(response.citations, start=1):
        document = citation.document
        section = f" | {citation.section}" if citation.section else ""
        score = "-" if citation.score is None else f"{citation.score:.4f}"
        print(
            f"  [{number}] {document.doc_id} (released {document.release_date}){section}  "
            f"chunk {citation.chunk_id}, score {score}\n      {document.url}"
        )


def _retriever(
    settings: Settings, method: RetrievalMethod, embeddings: "Embeddings | None" = None
) -> "Retriever":
    """The retriever for `method` under settings.retriever_preset, after the same readiness
    check as the API (`embeddings`: query vectors from elsewhere than OpenAI)."""
    from app.rag.config import production_config
    from app.rag.index import require_index
    from app.rag.retrievers import get_retriever

    config = production_config(settings)
    require_index(settings)
    return get_retriever(method, settings, config, embeddings=embeddings)


def _answer(settings: Settings, request: "AskRequest") -> "AskResponse":
    from app.rag.answer import answer_question, get_llm

    return answer_question(
        request,
        retriever=_retriever(settings, request.retrieval),
        llm=get_llm(settings),
        model_name=settings.llm_model,
    )


if __name__ == "__main__":
    sys.exit(main())
