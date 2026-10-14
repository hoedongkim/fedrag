"""The DEV-only search that froze the `tuned` retriever preset (docs/rag-tuning.md).

    cd backend && .venv/bin/python scripts/tune_retriever.py [--questions FILE] [--out FILE]

Only questions of the DEV split are ranked or scored; the script stops if any other gets in. It
runs the production retrievers (app.rag.retrievers.get_retriever under an app.rag.config
RetrieverConfig) and the production metric functions (app.evaluation.retrieval_metrics), with:

- query vectors from data/eval_cache/query_embeddings.jsonl only: a question without a cached
  vector is an error, and the embedding client refuses any request;
- cross-encoder scores memoized per (query, passage), for speed only (the evaluation CLI, which
  has no memo, matched it to 3 decimals);
- no download: Hugging Face runs offline, so both cross-encoders must be cached locally.

The search, as run on 2026-10-13:

1. ablations: each option alone on top of phase3, scored with all four methods;
2. a staged grid scored with hybrid_rerank:
   - stage 1: date_prior x statement_boost x reranker (24 configs);
   - stage 2: year_prior x recency (21 combinations) on each stage-1 config within BAND nDCG@10
     of the stage-1 best, then stage 3 (the options without a one-factor gain, one at a time)
     on the pick so far, and a leave-one-out from it;
   - the disclosed revision: stage 2 had run on MiniLM-L-12 bases only, so stages 2 and 3 are
     repeated on the best stage-1 config with the settings' reranker (2b, 3b), with a
     leave-one-out from the new pick;
3. the rule, over every config evaluated: the fewest non-default options among the configs
   within BAND of the best hybrid_rerank nDCG@10 (coupled fields count once: prior + half-life,
   expansion + weight, model + revision), then the higher nDCG@10, then the higher recall@10.

Every config's DEV metrics go to data/eval/rag_tuning_dev.json (--out): the means of each method
it was scored with, hybrid_rerank's per-question nDCG@10, and the pick. The documented search
ran on the gold before the owner's fix (docs/rag-eval.md, "Owner review"); --questions takes that
file to reproduce it exactly.
"""

import argparse
import hashlib
import itertools
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # never download a model from here

from app.core.config import REPO_ROOT, get_settings  # noqa: E402
from app.evaluation.questions import QUESTIONS_FILE, EvalQuestion, load_questions  # noqa: E402
from app.evaluation.retrieval_metrics import (  # noqa: E402
    DEPTH,
    RETRIEVAL_METRICS,
    RankedHit,
    paired_difference,
    question_metrics,
)
from app.evaluation.runner import query_embeddings  # noqa: E402
from app.rag.config import RetrieverConfig  # noqa: E402
from app.rag.index import load_chunks  # noqa: E402
from app.rag.retrievers import CrossEncoderScorer, get_retriever  # noqa: E402
from app.stance.metrics import DEFAULT_BOOTSTRAP  # noqa: E402

METHODS = ("bm25", "dense", "hybrid", "hybrid_rerank")
HR = "hybrid_rerank"
BAND = 0.01
L12 = {
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "reranker_revision": "7b0235231ca2674cb8ca8f022859a6eba2b1c968",
}
COUPLED = {"reranker_revision", "recency_half_life_days", "expansion_weight"}
RECENCY = [None] + [(p, h) for p in (0.5, 1.0) for h in (90.0, 180.0, 365.0)]
ABLATIONS: dict[str, dict[str, Any]] = {
    "bm25_stemming": {"bm25_stemming": True},
    **{
        f"query_expansion w={w}": {"query_expansion": True, "expansion_weight": w}
        for w in (0.25, 0.5, 1.0)
    },
    **{f"date_prior {v}": {"date_prior": v} for v in (1.5, 2.0, 3.0)},
    **{f"year_prior {v}": {"year_prior": v} for v in (1.25, 1.5)},
    **{
        f"recency_prior {p}, {h:g} d": {"recency_prior": p, "recency_half_life_days": h}
        for p, h in RECENCY[1:]
    },
    **{f"statement_boost {v}": {"statement_boost": v} for v in (1.2, 1.5)},
    **{f"rrf_weights {w}": {"rrf_weights": w} for w in ((1.0, 0.5), (0.5, 1.0), (0.25, 1.0))},
    "rrf_k 30": {"rrf_k": 30},
    "min_candidates 100": {"min_candidates": 100},
    "reranker MiniLM-L-12": L12,
}
EXTRAS: list[dict[str, Any]] = [  # stage 3: no clear one-factor gain for hybrid_rerank
    {"bm25_stemming": True},
    *({"query_expansion": True, "expansion_weight": w} for w in (0.25, 0.5, 1.0)),
    {"rrf_weights": (0.5, 1.0)},
    {"rrf_weights": (0.25, 1.0)},
    {"rrf_weights": (1.0, 0.5)},
    {"rrf_k": 30},
    {"min_candidates": 100},
]
LEAVE_OUT = (
    ["date_prior"],
    ["statement_boost"],
    ["year_prior"],
    ["recency_prior", "recency_half_life_days"],
    ["reranker_model", "reranker_revision"],
)


class MemoScorer:
    """A CrossEncoderScorer whose (query, passage) scores are memoized (speed only)."""

    def __init__(self, model: str, revision: str | None) -> None:
        self.inner = CrossEncoderScorer(model, revision=revision)
        self.memo: dict[str, float] = {}

    def __call__(self, query: str, passages: Sequence[str]) -> list[float]:
        keys = [hashlib.sha1(f"{query}\x00{p}".encode()).hexdigest() for p in passages]
        todo = {k: p for k, p in zip(keys, passages, strict=True) if k not in self.memo}
        if todo:
            scores = self.inner(query, list(todo.values()))
            self.memo.update(zip(todo, scores, strict=True))
        return [self.memo[k] for k in keys]


def label(fields: dict[str, Any]) -> str:
    parts = [
        "reranker=L-12" if key == "reranker_model" else f"{key}={value}"
        for key, value in fields.items()
        if key != "reranker_revision"
    ]
    return ", ".join(parts) or "phase3"


def options(fields: dict[str, Any]) -> int:
    """Non-default options, coupled fields counted once."""
    return len(set(fields) - COUPLED)


class Search:
    """Scores configs on DEV and keeps every result (see the module docstring)."""

    def __init__(self, questions: Sequence[EvalQuestion]) -> None:
        self.settings = get_settings()
        self.dev = [q for q in questions if q.split == "dev"]  # the only questions ever ranked
        if not self.dev:
            raise SystemExit("no DEV questions")
        self.answerable = [q for q in self.dev if q.answerable]
        self.embeddings = query_embeddings(self.settings)
        if missing := self.embeddings.missing([q.question for q in self.dev]):
            raise SystemExit(f"{len(missing)} DEV query vectors are not cached; none is embedded")
        self.embeddings._embedder = _no_embedding_requests  # belt and braces
        self.scorers: dict[tuple[str, str | None], MemoScorer] = {}
        self.results: dict[tuple[str, str], dict[str, Any]] = {}
        self.evaluated: dict[str, dict[str, Any]] = {}

    def config(self, fields: dict[str, Any]) -> RetrieverConfig:
        """phase3 (RetrieverConfig's defaults) with `fields` set."""
        return RetrieverConfig(name="search", **fields)

    def method(self, fields: dict[str, Any], method: str) -> dict[str, Any]:
        """{"mean": {metric: mean}, "ndcg": {question id: nDCG@10}} of one method on DEV."""
        config = self.config(fields)
        key = (config.resolved(self.settings).fingerprint, method)
        if key not in self.results:
            extra: dict[str, Any] = {"embeddings": self.embeddings} if method != "bm25" else {}
            if method == HR:
                reranker = config.reranker(self.settings)
                extra["scorer"] = self.scorers.setdefault(reranker, MemoScorer(*reranker))
            retriever = get_retriever(method, self.settings, config, **extra)  # type: ignore[arg-type]
            values: dict[str, dict[str, Any]] = {}
            for q in self.dev:
                hits = retriever.retrieve(q.question, top_k=DEPTH, as_of=q.as_of)
                if any(q.as_of and h.chunk.release_date > q.as_of for h in hits):
                    raise SystemExit(f"{q.id}: a hit released after as_of")
                if q.answerable:
                    ranked = [RankedHit(h.chunk.chunk_id, h.chunk.doc_id, h.score) for h in hits]
                    values[q.id] = question_metrics(ranked, q)
            self.results[key] = {
                "mean": {m: _mean([v[m] for v in values.values()]) for m in RETRIEVAL_METRICS},
                "ndcg": {qid: v["ndcg_at_10"] for qid, v in values.items()},
            }
        return self.results[key]

    def evaluate(self, fields: dict[str, Any], stage: str, methods: Sequence[str] = (HR,)) -> dict:
        name = label(fields)
        row = self.evaluated.setdefault(
            name, {"stage": stage, "fields": fields, "options": options(fields), "methods": {}}
        )
        for method in methods:
            if method not in row["methods"]:
                row["methods"][method] = self.method(fields, method)["mean"]
        return row

    def score(self, name: str) -> tuple[float, float]:
        mean = self.evaluated[name]["methods"][HR]
        return mean["ndcg_at_10"], mean["recall_at_10"]

    def pick(self, names: Sequence[str] | None = None) -> tuple[str, str]:
        """(the best config, the rule's pick) among `names` (default: every config evaluated)."""
        names = list(names if names is not None else self.evaluated)
        best = max(names, key=self.score)
        near = [n for n in names if self.score(n)[0] >= self.score(best)[0] - BAND]
        chosen = min(
            near, key=lambda n: (self.evaluated[n]["options"], *(-v for v in self.score(n)))
        )
        return best, chosen

    def stage2(self, base: dict[str, Any], stage: str) -> None:
        for year, recency in itertools.product((1.0, 1.25, 1.5), RECENCY):
            fields = dict(base)
            if year != 1.0:
                fields["year_prior"] = year
            if recency:
                fields |= {"recency_prior": recency[0], "recency_half_life_days": recency[1]}
            self.evaluate(fields, stage)

    def stage3(self, base: dict[str, Any], stage: str) -> None:
        for extra in [*EXTRAS, *([] if "reranker_model" in base else [L12])]:
            self.evaluate(dict(base) | extra, stage)

    def leave_one_out(self, fields: dict[str, Any]) -> None:
        for group in LEAVE_OUT:
            if group[0] in fields:
                self.evaluate({k: v for k, v in fields.items() if k not in group}, "loo")


def _no_embedding_requests() -> Any:
    raise RuntimeError("the tuning search must not call the embeddings API")


def _mean(values: Sequence[float | None]) -> float:
    present = [v for v in values if v is not None]
    return sum(present) / len(present)


def run(search: Search) -> dict[str, Any]:
    search.evaluate({}, "0", METHODS)
    for fields in ABLATIONS.values():
        search.evaluate(fields, "ablation", METHODS)

    stage1 = []
    for date, boost, l12 in itertools.product((1.0, 1.5, 2.0, 3.0), (1.0, 1.2, 1.5), (0, 1)):
        fields: dict[str, Any] = {}
        if date != 1.0:
            fields["date_prior"] = date
        if boost != 1.0:
            fields["statement_boost"] = boost
        if l12:
            fields |= L12
        stage1.append(label(search.evaluate(fields, "1")["fields"]))
    best1, _ = search.pick(stage1)
    band = [n for n in stage1 if search.score(n)[0] >= search.score(best1)[0] - BAND]
    for name in band:
        search.stage2(search.evaluated[name]["fields"], "2")
    _, chosen = search.pick()
    search.stage3(search.evaluated[chosen]["fields"], "3")
    _, chosen = search.pick()
    search.leave_one_out(search.evaluated[chosen]["fields"])

    # The disclosed revision: stages 2 and 3 again on the best stage-1 config with the settings'
    # reranker, which stage 2 had not covered.
    l6 = max((n for n in stage1 if "reranker_model" not in search.evaluated[n]["fields"]),
             key=search.score)  # fmt: skip
    search.stage2(search.evaluated[l6]["fields"], "2b")
    _, chosen = search.pick()
    search.stage3(search.evaluated[chosen]["fields"], "3b")
    _, chosen = search.pick()
    search.leave_one_out(search.evaluated[chosen]["fields"])
    best, chosen = search.pick()
    return {"best": best, "chosen": chosen}


def contributions(search: Search, chosen: str) -> list[dict[str, Any]]:
    """The pick minus the pick without each option, on hybrid_rerank nDCG@10 (paired, 95% CI)."""
    fields = search.evaluated[chosen]["fields"]
    picked = search.method(fields, HR)["ndcg"]
    rows = []
    for group in LEAVE_OUT:
        if group[0] not in fields:
            continue
        without = search.method({k: v for k, v in fields.items() if k not in group}, HR)["ndcg"]
        ids = list(picked)
        diff = paired_difference(
            [picked[i] for i in ids], [without[i] for i in ids], DEFAULT_BOOTSTRAP
        )
        changed = sum(abs(picked[i] - without[i]) > 1e-9 for i in ids)
        rows.append({"without": group[0], **(diff or {}), "questions_changed": changed})
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    data = get_settings().data_dir
    parser.add_argument("--questions", type=Path, default=data / QUESTIONS_FILE)
    parser.add_argument("--out", type=Path, default=data / "eval" / "rag_tuning_dev.json")
    args = parser.parse_args(argv)

    chunks = {chunk.chunk_id: chunk for chunk in load_chunks(get_settings())}
    question_set = load_questions(args.questions, chunks)
    search = Search(question_set.questions)
    picks = run(search)
    best, chosen = picks["best"], picks["chosen"]
    hr = {n: search.score(n) for n in search.evaluated}
    print(f"{len(search.evaluated)} configs on {len(search.answerable)} answerable DEV questions")
    print(f"best: {hr[best][0]:.4f} nDCG@10  {best}")
    print(f"rule (fewest options within {BAND}): {hr[chosen][0]:.4f}  {chosen}")
    for row in contributions(search, chosen):
        print(
            f"  without {row['without']}: {row['difference']:+.3f} "
            f"[{row['ci']['low']:+.3f}, {row['ci']['high']:+.3f}], "
            f"{row['questions_changed']} questions changed"
        )
    out = {
        "questions": _shown(args.questions),
        "questions_sha256": question_set.sha256,
        "split": "dev",
        "n_answerable": len(search.answerable),
        "band": BAND,
        "best": best,
        "chosen": chosen,
        "chosen_fields": search.evaluated[chosen]["fields"],
        "contributions": contributions(search, chosen),
        "configs": [
            {
                "label": name,
                **row,
                "ndcg_at_10_by_question": search.method(row["fields"], HR)["ndcg"],
            }
            for name, row in search.evaluated.items()
        ],
    }
    args.out.write_text(json.dumps(out, indent=1, default=list) + "\n", encoding="utf-8")
    print(f"Wrote {_shown(args.out)}")
    return 0


def _shown(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    sys.exit(main())
