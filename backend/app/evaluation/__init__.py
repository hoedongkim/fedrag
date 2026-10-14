"""RAG evaluation (Phase 7): retrieval metrics, cached answers, an LLM judge and the benchmark file.

questions loads and validates the evaluation set, retrieval_metrics scores ranked lists,
runner retrieves and answers with the production chain (cached on disk), judge and
judge_prompts grade the answers, and report writes data/eval/rag_benchmark.json. The command line
is python -m app.evaluation {validate,retrieval,answers,judge,report,export-judge-sample}.
"""
