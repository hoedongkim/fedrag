"""Command line for the RAG evaluation: python -m app.evaluation {validate,retrieval,answers,...}.

    validate             check the question file against the schema and the index (no calls)
    retrieval            rank the top 10 chunks per question with each method; print the metrics
    answers              answer each question with each method (OpenAI; cached; cost-guarded)
    judge                grade the cached answers with the LLM judge (OpenAI; cached; guarded)
    report               rebuild data/eval/rag_benchmark.json from the caches (no calls)
    export-judge-sample  write a blind sample of judged answers for an independent annotator
    export-ask-examples  write data/eval/ask_examples.json, the recorded answers that
                         GET /api/ask/examples serves (no calls)

Every command but validate and export-ask-examples takes --questions, --split dev|test|all
(default dev; report: all), --ids, --methods and --config (one or more retriever configurations;
"default" is the production retriever).
answers and judge print a tiktoken estimate first and refuse to call when it is over --max-usd
(default 1.0) or needs more than --max-calls; --dry-run stops after the estimate.
"""

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args

from app.core.config import REPO_ROOT, Settings, get_settings
from app.schemas.common import RetrievalMethod

if TYPE_CHECKING:
    from app.evaluation.judge import AnswerJudge, JudgeInput
    from app.evaluation.judge_prompts import AnswerJudgment
    from app.evaluation.questions import EvalQuestion, QuestionSet
    from app.evaluation.report import RunData
    from app.evaluation.runner import AnswerRunner, RetrieverFactory, RetrieverSetup
    from app.rag.types import Chunk
    from app.stance.llm import Usage

DEFAULT_MAX_USD = 1.0
DEFAULT_CONCURRENCY = 4
DEFAULT_SAMPLE_SIZE = 40
DEFAULT_SAMPLE_SEED = 20261013
SAMPLE_FILE = Path("eval") / "rag_judge_sample.json"  # under settings.data_dir
SAMPLE_KEY_FILE = Path("eval") / "rag_judge_sample_key.json"
ANNOTATIONS_FILE = Path("eval") / "rag_judge_annotations.json"
ASK_EXAMPLES_FILE = Path("eval") / "ask_examples.json"  # app.api.ask_examples serves it
# Metrics a run can be selected by (higher is better).
SELECTABLE = (
    "recall_at_5",
    "recall_at_10",
    "mrr_at_10",
    "ndcg_at_10",
    "doc_recall_at_5",
    "doc_recall_at_10",
    "faithfulness",
    "citation_accuracy",
    "answer_correctness",
    "refusal_accuracy",
)
COMMANDS = {
    "validate": "check the question file against the schema and the search index (no API calls)",
    "retrieval": "rank the top 10 chunks per question with each method and print the metrics",
    "answers": "answer each question with each method through the production chain (OpenAI)",
    "judge": "grade the cached answers with the LLM judge (OpenAI)",
    "report": "rebuild the benchmark file from the caches (no API calls)",
    "export-judge-sample": "write a blind sample of judged answers for an independent annotator",
    "export-ask-examples": "write the recorded example answers of the production configuration "
    "that GET /api/ask/examples serves (no API calls)",
}


def build_parser() -> argparse.ArgumentParser:
    from app.evaluation.judge import DEFAULT_JUDGE_MODEL, DEFAULT_REASONING_EFFORT
    from app.evaluation.runner import ANSWER_TOP_K

    parser = argparse.ArgumentParser(
        prog="python -m app.evaluation",
        description="Evaluate retrieval and grounded answers on the RAG question set.",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")
    subs = {
        name: commands.add_parser(name, help=text, description=text)
        for name, text in COMMANDS.items()
    }
    for name, sub in subs.items():
        sub.add_argument(
            "--questions",
            type=Path,
            help="question file (default: data/eval/rag_questions.jsonl; a smoke.jsonl file is "
            "converted on the fly)",
        )
        if name in ("validate", "export-ask-examples"):
            continue
        sub.add_argument(
            "--split",
            choices=["dev", "test", "all"],
            default="all" if name == "report" else "dev",
            help="questions to use (default: dev; report: all, with dev, test and all sections)",
        )
        sub.add_argument("--ids", nargs="+", help="only these question ids")
        sub.add_argument(
            "--methods",
            nargs="+",
            choices=get_args(RetrievalMethod),
            default=list(get_args(RetrievalMethod)),
            help="retrieval methods (default: all four)",
        )
        sub.add_argument(
            "--config",
            dest="configs",
            nargs="+",
            default=["default"],
            help="retriever configurations: default (the production retriever, the preset "
            "RETRIEVER_PRESET names: tuned unless set), phase3, a preset of "
            "app/rag/retriever_presets.json, or a JSON object of RetrieverConfig fields; the "
            "first is the baseline of config comparisons",
        )
        if name != "retrieval":
            sub.add_argument(
                "--top-k",
                type=int,
                default=ANSWER_TOP_K,
                help=f"passages given to the LLM (default: {ANSWER_TOP_K}, as Ask the Fed)",
            )
        if name in ("judge", "report", "export-judge-sample"):
            sub.add_argument(
                "--judge-model",
                default=DEFAULT_JUDGE_MODEL,
                help=f"judge model (default: {DEFAULT_JUDGE_MODEL})",
            )
            sub.add_argument(
                "--judge-reasoning",
                choices=["minimal", "low", "medium", "high"],
                default=DEFAULT_REASONING_EFFORT,
                help=f"judge reasoning effort (default: {DEFAULT_REASONING_EFFORT})",
            )
        if name in ("answers", "judge"):
            sub.add_argument(
                "--max-usd",
                type=float,
                default=DEFAULT_MAX_USD,
                help="refuse to start when the estimated cost of the uncached calls is above this "
                f"(default: {DEFAULT_MAX_USD})",
            )
            sub.add_argument(
                "--max-calls",
                type=int,
                help="refuse to start when more calls are needed (default: the estimate plus a "
                "retry margin)",
            )
            sub.add_argument(
                "--concurrency",
                type=int,
                default=DEFAULT_CONCURRENCY,
                help=f"OpenAI calls in flight at once (default: {DEFAULT_CONCURRENCY})",
            )
            sub.add_argument(
                "--dry-run", action="store_true", help="print the estimate only: no LLM calls"
            )
    subs["report"].add_argument(
        "--out",
        type=Path,
        help="output file (default: data/eval/rag_benchmark.json for the full rebuild, --split "
        "all with two configs and --switch-rule as make rag-eval-report runs it; any other "
        "selection goes to data/eval_cache/rag_benchmark.<split>-<configs>.json)",
    )
    choice = subs["report"].add_mutually_exclusive_group()
    choice.add_argument(
        "--select",
        choices=SELECTABLE,
        metavar="METRIC",
        help="record the run with the highest dev mean of METRIC (e.g. ndcg_at_10) as selection",
    )
    choice.add_argument(
        "--switch-rule",
        action="store_true",
        help="record the default-switch rule (decided before the TEST run) as selection: the "
        "second --config replaces the first only if, on test, its hybrid_rerank nDCG@10 gain has "
        "a paired 95%% CI above 0 and its answer_correctness difference a CI lower bound above "
        "-0.05; needs the full test split (no --ids)",
    )
    sample = subs["export-judge-sample"]
    sample.add_argument(
        "--n", type=int, default=DEFAULT_SAMPLE_SIZE, help="answers to sample (default: 40)"
    )
    sample.add_argument("--seed", type=int, default=DEFAULT_SAMPLE_SEED, help="sampling seed")
    sample.add_argument(
        "--out",
        type=Path,
        help="annotation file (default: data/eval/rag_judge_sample.json, which is refused once it "
        "exists: the annotations label it)",
    )
    subs["export-ask-examples"].add_argument(
        "--out", type=Path, help="output file (default: data/eval/ask_examples.json)"
    )
    return parser


def main(argv: Sequence[str] | None = None, *, settings: Settings | None = None) -> int:
    """Run one command; returns the process exit code."""
    args = build_parser().parse_args(argv)
    settings = settings or get_settings()
    # Imported here so --help stays fast; these pull in Chroma, LangChain and torch.
    import openai

    from app.evaluation.questions import QuestionSetError
    from app.rag.answer import LLMNotConfiguredError
    from app.rag.index import EmbeddingsNotConfiguredError, IndexNotBuiltError
    from app.rag.retrievers import RerankerUnavailableError
    from app.stance.__main__ import BudgetError
    from app.stance.llm import CallLimitError, LLMReplyError

    commands: dict[str, Callable[[Settings, argparse.Namespace], int]] = {
        "validate": run_validate,
        "retrieval": run_retrieval,
        "answers": run_answers,
        "judge": run_judge,
        "report": run_report,
        "export-judge-sample": run_export,
        "export-ask-examples": run_export_ask_examples,
    }
    try:
        return commands[args.command](settings, args)
    except (
        BudgetError,
        CallLimitError,
        LLMReplyError,
        LLMNotConfiguredError,
        EmbeddingsNotConfiguredError,
        IndexNotBuiltError,
        RerankerUnavailableError,
        QuestionSetError,
        openai.OpenAIError,
        ValueError,  # a malformed file or option; also pydantic's ValidationError
    ) as exc:
        message = str(exc)
        if isinstance(exc, openai.OpenAIError):  # its message can quote part of the API key
            detail = type(exc).__name__
            if (status := getattr(exc, "status_code", None)) is not None:
                detail += f", HTTP {status}"
            message = f"OpenAI request failed ({detail})"
        print(f"{args.command}: {message}", file=sys.stderr)
        return 1


# --- Shared steps ---------------------------------------------------------------------------------


def questions_path(settings: Settings, args: argparse.Namespace) -> Path:
    from app.evaluation.questions import QUESTIONS_FILE

    return args.questions or settings.data_dir / QUESTIONS_FILE


def index_chunks(settings: Settings) -> "dict[str, Chunk] | None":
    """chunk_id -> chunk of the built index, or None when there is no index."""
    from app.rag.index import IndexNotBuiltError, load_chunks

    try:
        return {chunk.chunk_id: chunk for chunk in load_chunks(settings)}
    except IndexNotBuiltError:
        return None


def load_set(settings: Settings, args: argparse.Namespace) -> "QuestionSet":
    from app.evaluation.questions import load_questions

    chunks = index_chunks(settings)
    question_set = load_questions(questions_path(settings, args), chunks)
    if chunks is None:
        print("No search index: chunk ids were not checked against it.", file=sys.stderr)
    if question_set.format == "smoke":
        print(
            f"{question_set.path.name}: smoke questions converted on the fly (document labels "
            "only, all in dev): a harness check, not the benchmark."
        )
    return question_set


def selected(question_set: "QuestionSet", args: argparse.Namespace) -> "list[EvalQuestion]":
    questions = question_set.select(args.split, args.ids)
    if not questions:
        raise ValueError(
            f"no questions in split {args.split}" + (" with those ids" * bool(args.ids))
        )
    return questions


def setups(settings: Settings, args: argparse.Namespace) -> "list[RetrieverSetup]":
    from app.evaluation.runner import resolve_config

    resolved = [resolve_config(name, settings) for name in dict.fromkeys(args.configs)]
    return list({s.hash: s for s in resolved}.values())


def retriever_factory(
    settings: Settings, questions: "Sequence[EvalQuestion]", methods: Sequence[str]
) -> "RetrieverFactory":
    """The production retrievers, with every question's query vector embedded up front (one
    batched request for the ones not cached) when a dense-based method is used."""
    from app.evaluation.runner import make_retriever_factory, query_embeddings
    from app.rag.index import require_index

    require_index(settings)
    if all(method == "bm25" for method in methods):
        return make_retriever_factory(settings)
    embeddings = query_embeddings(settings)
    embedded = embeddings.prefetch([q.question for q in questions])
    if embedded:
        print(
            f"Embedded {embedded} questions in {embeddings.requests} request(s) "
            f"({embeddings.tokens:,} tokens, {settings.embedding_model}); cached for reruns."
        )
    return make_retriever_factory(settings, embeddings)


def unembedded(
    settings: Settings, questions: "Sequence[EvalQuestion]", methods: Sequence[str]
) -> list[str]:
    """The questions whose query vector a dense-based method would have to embed first."""
    from app.evaluation.runner import query_embeddings

    if all(method == "bm25" for method in methods):
        return []
    return query_embeddings(settings).missing([q.question for q in questions])


def budget(estimates: Sequence[tuple[str, Any]], args: argparse.Namespace) -> int:
    """Print the estimates and refuse (BudgetError) over --max-usd; returns the calls needed."""
    from app.stance.__main__ import check_budget

    return check_budget(estimates, args.max_usd)


def call_limit(calls: int, args: argparse.Namespace) -> int:
    from app.stance.__main__ import call_ceiling

    return args.max_calls if args.max_calls is not None else call_ceiling(calls)


def require_llm(settings: Settings, calls: int) -> None:
    from app.rag.answer import LLMNotConfiguredError

    if calls and not settings.llm_configured:
        raise LLMNotConfiguredError


def print_usage(label: str, usage: "Usage", model: str) -> None:
    from app.evaluation.judge import model_usd

    if not usage.calls:
        print(f"\n{label}: no OpenAI calls (everything was cached).")
        return
    usd = model_usd(model, usage.input_tokens, usage.cached_input_tokens, usage.output_tokens)
    print(
        f"\n{label} ({model}): {usage.calls} calls, {usage.input_tokens:,} input "
        f"({usage.cached_input_tokens:,} cached) + {usage.output_tokens:,} output tokens "
        f"({usage.reasoning_tokens:,} reasoning), about ${usd:.4f}."
    )


def answer_runner(
    settings: Settings,
    args: argparse.Namespace,
    retrievers: "RetrieverFactory | None" = None,
    usage: "Usage | None" = None,
) -> "AnswerRunner":
    from app.evaluation.runner import AnswerRunner

    def no_retrieval(setup: "RetrieverSetup", method: RetrievalMethod) -> Any:
        raise RuntimeError("this command reads cached answers only")

    return AnswerRunner(
        settings,
        retrievers or no_retrieval,
        usage=usage,
        top_k=args.top_k,
        max_concurrency=getattr(args, "concurrency", 1),
    )


def judge_for(
    settings: Settings, args: argparse.Namespace, usage: "Usage | None" = None
) -> "AnswerJudge":
    from app.evaluation.judge import AnswerJudge

    return AnswerJudge(
        settings,
        model=args.judge_model,
        reasoning_effort=args.judge_reasoning,
        usage=usage,
        max_concurrency=getattr(args, "concurrency", 1),
    )


def judge_inputs(
    settings: Settings, args: argparse.Namespace, questions: "Sequence[EvalQuestion]"
) -> "tuple[list[JudgeInput], int]":
    """A JudgeInput for every cached answer of the selection, and how many answers are missing."""
    from app.evaluation.judge import judge_input
    from app.evaluation.runner import AnswerItem

    chunks = index_chunks(settings)
    if chunks is None:
        from app.rag.index import IndexNotBuiltError

        raise IndexNotBuiltError
    runner = answer_runner(settings, args)
    inputs, missing = [], 0
    for setup in setups(settings, args):
        for method in args.methods:
            for q in questions:
                record = runner.cached(AnswerItem(q, method, setup))
                if record is None:
                    missing += 1
                else:
                    inputs.append(judge_input(q, record, chunks))
    return inputs, missing


def default_out(settings: Settings, question_set: "QuestionSet", tracked: Path) -> Path:
    """`tracked` (under data/) for the evaluation set itself; for any other question file (the
    smoke questions, a scratch file), a file in data/eval_cache named after it, so a harness
    check never overwrites the tracked benchmark."""
    from app.evaluation.questions import QUESTIONS_FILE
    from app.evaluation.runner import cache_dir

    if (
        question_set.format == "rag_questions"
        and question_set.path.resolve() == (settings.data_dir / QUESTIONS_FILE).resolve()
    ):
        return settings.data_dir / tracked
    return cache_dir(settings) / f"{tracked.stem}.{question_set.path.stem}{tracked.suffix}"


def shown(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# --- Commands -------------------------------------------------------------------------------------


def run_validate(settings: Settings, args: argparse.Namespace) -> int:
    from app.evaluation.questions import load_questions

    chunks = index_chunks(settings)
    question_set = load_questions(questions_path(settings, args), chunks, check=False)
    summary = question_set.summary()
    print(
        f"{shown(question_set.path)} ({question_set.format}): {summary['n_questions']} questions, "
        f"{summary['n_answerable']} answerable; splits {summary['splits']}"
    )
    print("types: " + ", ".join(f"{k} {v}" for k, v in summary["types"].items()))  # type: ignore[union-attr]
    if chunks is None:
        print("No search index: chunk and document ids were not checked.")
    for warning in question_set.warnings:
        print(f"warning: {warning}")
    for error in question_set.errors:
        print(f"error: {error}")
    print(f"{len(question_set.errors)} errors, {len(question_set.warnings)} warnings")
    return 1 if question_set.errors else 0


def run_retrieval(settings: Settings, args: argparse.Namespace) -> int:
    from app.evaluation import report as rep
    from app.evaluation import runner
    from app.stance.llm import PredictionCache

    question_set = load_set(settings, args)
    questions = selected(question_set, args)
    answerable = [q for q in questions if q.answerable]
    factory = retriever_factory(settings, questions, args.methods)
    cache = PredictionCache(runner.cache_dir(settings) / runner.RETRIEVAL_CACHE)
    rows = []
    leaks = 0
    for setup in setups(settings, args):
        ranked = runner.run_retrieval(questions, args.methods, setup, factory, cache)
        for method, method_rows in ranked.items():
            leaks += sum(r["leaks"] for r in method_rows)
            run = rep.RunData(setup, method, retrieval={r["question_id"]: r for r in method_rows})
            values = {q.id: rep.question_values(run, q) for q in questions}
            row = rep.retrieval_row(run, questions, args.split, values)
            if row is not None:
                rows.append(row)
    print(
        f"\nRetrieval: {len(questions)} {args.split} questions ranked, the {len(answerable)} "
        "answerable ones scored (top 10; mean [95% bootstrap CI]; tie-aware):"
    )
    print(rep.retrieval_table(rows) if rows else "(no answerable question to score)")
    print(
        "Latency per question after a warm-up query, query embedding excluded (served from "
        "data/eval_cache). Rankings saved to data/eval_cache/retrieval.jsonl."
    )
    if leaks:
        print(f"ERROR: {leaks} retrieved chunks were released after their question's as_of.")
    return 1 if leaks else 0


def run_answers(settings: Settings, args: argparse.Namespace) -> int:
    from app.evaluation.runner import AnswerItem
    from app.stance.llm import Usage

    question_set = load_set(settings, args)
    questions = selected(question_set, args)
    configs = setups(settings, args)
    usage = Usage()
    probe = answer_runner(settings, args)
    items = [AnswerItem(q, m, s) for s in configs for m in args.methods for q in questions]
    uncached = [i for i in items if probe.cached(i) is None]
    print(f"{len(items)} answers asked for, {len(items) - len(uncached)} cached.")
    if args.dry_run and (
        todo := unembedded(settings, [i.question for i in uncached], [i.method for i in uncached])
    ):
        from app.evaluation.runner import _embedding_tokens

        tokens = sum(_embedding_tokens(text) for text in todo)
        print(
            f"Dry run: {len(todo)} questions would be embedded first (~{tokens:,} tokens, "
            f"{settings.embedding_model}) before their passages can be retrieved and the answers "
            "estimated; no request made."
        )
        return 0
    factory = (
        retriever_factory(settings, [i.question for i in uncached], [i.method for i in uncached])
        if uncached
        else None
    )
    runner = answer_runner(settings, args, factory, usage)
    pending = runner.prepare(uncached)
    estimate = runner.estimate(pending, len(items))
    print(f"OpenAI estimate ({runner.model}, prompt {runner.prompt_version}):")
    calls = budget([("answers", estimate)], args)
    if args.dry_run:
        print("Dry run: no LLM calls.")
        return 0
    require_llm(settings, calls)
    usage.max_calls = call_limit(calls, args)
    records = runner.generate(pending)
    leaked = 0
    for setup in configs:
        for method in args.methods:
            mine = [runner.cached(AnswerItem(q, method, setup)) for q in questions]
            got = [r for r in mine if r is not None]
            cited = sum(bool(r["citations"]) for r in got)
            leaked += sum(
                q.as_of is not None
                and any(p["release_date"] > q.as_of.isoformat() for p in r["passages"])
                for q, r in zip(questions, mine, strict=True)
                if r is not None
            )
            print(f"  {setup.name}/{method}: {len(got)} answers, {cited} with citations")
    print(f"Generated {len(records)} answers now; cached in data/eval_cache/answers.jsonl.")
    print_usage("Answer generation", usage, runner.model)
    if leaked:
        print(f"ERROR: {leaked} answers were given a passage released after their as_of.")
    return 1 if leaked else 0


def run_judge(settings: Settings, args: argparse.Namespace) -> int:
    from app.evaluation import report as rep
    from app.evaluation.judge import score_judgment
    from app.stance.llm import Usage

    question_set = load_set(settings, args)
    questions = selected(question_set, args)
    inputs, missing = judge_inputs(settings, args, questions)
    if missing:
        print(f"{missing} answers are not cached yet and are skipped: run answers first.")
    if not inputs:
        raise ValueError("no cached answers to judge")
    usage = Usage()
    judge = judge_for(settings, args, usage)
    estimate = judge.estimate(inputs)
    print(
        f"OpenAI estimate ({judge.model}, reasoning {judge.reasoning_effort}, prompt "
        f"{judge.prompt_version}):"
    )
    calls = budget([("judgments", estimate)], args)
    if args.dry_run:
        print("Dry run: no LLM calls.")
        return 0
    require_llm(settings, calls)
    usage.max_calls = call_limit(calls, args)
    judgments = judge.judge(inputs)
    by_run: dict[str, Any] = {}
    for inp, judgment in zip(inputs, judgments, strict=True):
        record = inp.record
        key = f"{record['config']}/{record['method']}"
        by_run.setdefault(key, []).append(score_judgment(judgment, inp.question, len(inp.units)))
    print(f"\nJudged answers ({args.split}; means over the answers where a metric is defined):")
    for key, scores in by_run.items():
        means = {
            name: _mean([s[name] for s in scores if s[name] is not None])
            for name in ("faithfulness", "citation_accuracy", "answer_correctness")
        }
        cells = ", ".join(
            f"{name} {'-' if value is None else f'{value:.3f}'}" for name, value in means.items()
        )
        print(f"  {key}: {len(scores)} answers; {cells}; {rep.response_counts(scores)}")
    print_usage("Judge", usage, judge.model)
    return 0


def collect_runs(
    settings: Settings, args: argparse.Namespace, questions: "Sequence[EvalQuestion]"
) -> "list[RunData]":
    """Everything cached for the selection, as report.RunData (no API calls)."""
    from app.evaluation import report as rep
    from app.evaluation import runner
    from app.evaluation.judge import judge_input, score_judgment, verdicts
    from app.stance.llm import PredictionCache

    chunks = index_chunks(settings) or {}
    retrieval_cache = PredictionCache(runner.cache_dir(settings) / runner.RETRIEVAL_CACHE)
    answers = answer_runner(settings, args)
    judge = judge_for(settings, args)
    runs = []
    for setup in setups(settings, args):
        for method in args.methods:
            run = rep.RunData(setup, method)
            for q in questions:
                row = retrieval_cache.get(runner.retrieval_key(q, method, setup))
                if row is not None:
                    run.retrieval[q.id] = row
                record = answers.cached(runner.AnswerItem(q, method, setup))
                if record is None:
                    continue
                run.answers[q.id] = record
                if any(chunk_id not in chunks for chunk_id in record["citations"]):
                    continue
                inp = judge_input(q, record, chunks)
                judgment = judge.judgment(inp)
                if judgment is not None:
                    cached = judge.cached_record(inp)
                    run.judged[q.id] = {
                        "scores": score_judgment(judgment, q, len(inp.units)),
                        "verdicts": verdicts(judgment, inp),
                        "usage": cached.get("usage") if cached else None,
                        "model": judge.model,
                        "key": judge.key(inp),  # identical prompts share one judgment
                        "created_at": cached.get("created_at") if cached else None,
                    }
            runs.append(run)
    return runs


def report_out(
    settings: Settings, args: argparse.Namespace, question_set: "QuestionSet", names: Sequence[str]
) -> Path:
    """Where `report` writes without --out. The tracked data/eval/rag_benchmark.json only for the
    full rebuild of the evaluation set (--split all, no --ids, two or more configs and
    --switch-rule, as make rag-eval-report runs it); any other selection goes to data/eval_cache,
    named after it, and another question file to its scratch path (default_out)."""
    from app.evaluation import report as rep
    from app.evaluation.runner import cache_dir

    out = default_out(settings, question_set, rep.BENCHMARK_FILE)
    canonical = args.split == "all" and not args.ids and len(names) >= 2 and args.switch_rule
    if out != settings.data_dir / rep.BENCHMARK_FILE or canonical:
        return out
    label = "-".join([args.split, *names]) + ("-ids" if args.ids else "")
    return cache_dir(settings) / f"{rep.BENCHMARK_FILE.stem}.{label}{rep.BENCHMARK_FILE.suffix}"


def run_report(settings: Settings, args: argparse.Namespace) -> int:
    from app.evaluation import report as rep
    from app.evaluation.runner import _embedding_tokens, answer_prompt_label
    from app.rag.answer import REASONING_EFFORT

    if args.switch_rule and args.ids:
        raise ValueError("--switch-rule needs the full TEST split: drop --ids")
    question_set = load_set(settings, args)
    splits = rep.SPLIT_ORDER if args.split == "all" else (args.split,)
    questions = question_set.select(args.split, args.ids)
    runs = collect_runs(settings, args, questions)
    sample_path = default_out(settings, question_set, SAMPLE_FILE)
    key_path = sample_path.with_name(sample_path.stem + "_key" + sample_path.suffix)
    annotations_path = sample_path.with_name(ANNOTATIONS_FILE.name)
    sample, key, annotations = (
        json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
        for path in (sample_path, key_path, annotations_path)
    )
    judge_info: dict[str, Any] = {
        "model": args.judge_model,
        "reasoning_effort": args.judge_reasoning,
        "prompt_version": judge_for(settings, args).prompt_version,
        "validation": rep.judge_validation(
            sample, annotations, key, shown(sample_path) if sample else None
        ),
    }
    dense = [m for m in args.methods if m != "bm25"]
    texts = list(dict.fromkeys(q.question for q in questions)) if dense else []
    report = rep.build_report(
        question_set,
        runs,
        file_label=shown(question_set.path),
        generator={
            "model": settings.llm_model,
            "prompt_version": answer_prompt_label(),
            "reasoning_effort": REASONING_EFFORT,
            "top_k": args.top_k,
        },
        judge=judge_info,
        splits=splits,
        questions=questions,
        select=args.select,
        switch_rule=args.switch_rule,
        embedding={
            "model": settings.embedding_model,
            "texts": len(texts),
            "tokens": sum(_embedding_tokens(t) for t in texts),
        },
    )
    names = list(dict.fromkeys(run.setup.name for run in runs))
    out = args.out or report_out(settings, args, question_set, names)
    rep.write_report(out, report)
    for line in rep.coverage_lines(runs, questions):
        print(f"incomplete: {line}")
    for run in runs:
        if stale := rep.stale_answers(run):
            print(
                f"WARNING: {run.label}: {stale} answers were generated from passages that differ "
                "from the retrieval run"
            )
    for split in splits:
        retrieval = [r for r in report["retrieval"] if r["split"] == split]
        answers = [r for r in report["answers"] if r["split"] == split]
        if not retrieval and not answers:
            continue
        print(f"\n{split} split, retrieval (mean [95% CI]):")
        print(rep.retrieval_table(retrieval) if retrieval else "(incomplete)")
        print(f"\n{split} split, answers (LLM judge):")
        print(rep.answers_table(answers) if answers else "(incomplete)")
    shown_split = "test" if any(r["split"] == "test" for r in report["retrieval"]) else splits[0]
    rows = [d for d in report["differences"] if d["split"] == shown_split]
    if rows:
        print(f"\nPaired differences ({shown_split} split):")
        print(
            rep.comparison_table(
                rows, ("recall_at_5", "ndcg_at_10", "faithfulness", "answer_correctness")
            )
        )
    if report["selection"]:
        chosen = report["selection"]
        if args.switch_rule:
            print(f"\nDefault-switch rule: {chosen['config']}/{chosen['method']}. {chosen['rule']}")
        else:
            print(f"\nSelected on dev by {chosen['metric']}: {chosen['config']}/{chosen['method']}")
    validation = judge_info["validation"]
    print(f"\nJudge validation: {validation['status']} ({validation['n_annotated']} annotated)")
    if validation["agreement"] or validation.get("inter_annotator"):
        print(rep.validation_table(validation))
    print(f"\nWrote {shown(out)}")
    return 0


def run_export(settings: Settings, args: argparse.Namespace) -> int:
    from app.evaluation.judge import export_sample

    question_set = load_set(settings, args)
    questions = selected(question_set, args)
    inputs, _ = judge_inputs(settings, args, questions)
    judge = judge_for(settings, args)
    judged = [(inp, judge.judgment(inp)) for inp in inputs if inp.llm_called]
    pairs: list[tuple[JudgeInput, AnswerJudgment]] = [(i, j) for i, j in judged if j is not None]
    if not pairs:
        raise ValueError("no judged answers in the selection: run judge first")
    blind, key = export_sample(
        [i for i, _ in pairs], [j for _, j in pairs], n=args.n, seed=args.seed
    )
    key["judge"] = judge.info()
    out = args.out or default_out(settings, question_set, SAMPLE_FILE)
    if args.out is None and out == settings.data_dir / SAMPLE_FILE and out.exists():
        raise ValueError(
            f"{shown(out)} exists and {ANNOTATIONS_FILE.name} labels it: pass --out to write a "
            "new sample elsewhere"
        )
    key_out = out.with_name(out.stem + "_key" + out.suffix)
    from app.ingest.records import write_text_atomic

    write_text_atomic(out, json.dumps(blind, indent=2, ensure_ascii=False) + "\n")
    write_text_atomic(key_out, json.dumps(key, indent=2, ensure_ascii=False) + "\n")
    print(
        f"Wrote {len(blind['items'])} blind items to {shown(out)} and the judge's verdicts to "
        f"{shown(key_out)}. The annotator fills the null 'human' fields and saves "
        f"{ANNOTATIONS_FILE.name} next to them; report then adds the agreement."
    )
    return 0


def run_export_ask_examples(settings: Settings, args: argparse.Namespace) -> int:
    """data/eval/ask_examples.json from the cached answers of the production configuration
    (app.evaluation.ask_examples.PICKS), with the verdicts of data/eval/rag_benchmark.json."""
    from app.api.evaluation import EvaluationUnavailableError, load_evaluation
    from app.evaluation import ask_examples
    from app.evaluation.runner import ANSWER_TOP_K, DEFAULT_CONFIG, resolve_config
    from app.ingest.records import write_text_atomic

    question_set = load_set(settings, args)
    chunks = index_chunks(settings)
    if chunks is None:
        from app.rag.index import IndexNotBuiltError

        raise IndexNotBuiltError
    try:
        evaluation = load_evaluation(settings).file
    except EvaluationUnavailableError as exc:
        print(f"No judge verdicts: {exc}", file=sys.stderr)
        evaluation = None
    exported = ask_examples.build_examples(
        ask_examples.PICKS,
        {q.id: q for q in question_set.questions},
        answer_runner(settings, argparse.Namespace(top_k=ANSWER_TOP_K)),  # Ask the Fed's
        resolve_config(DEFAULT_CONFIG, settings),
        chunks,
        evaluation,
    )
    out = args.out or settings.data_dir / ASK_EXAMPLES_FILE
    write_text_atomic(out, ask_examples.render(exported))
    print(
        f"Wrote {len(exported.examples)} examples ({exported.retriever_preset}/"
        f"{exported.retrieval}, {exported.model}) to {shown(out)}:"
    )
    print("\n".join(ask_examples.summary_lines(exported)))
    return 0


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


if __name__ == "__main__":
    sys.exit(main())
