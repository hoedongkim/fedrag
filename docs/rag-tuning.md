# Retrieval Tuning (DEV only)

This file records how the `tuned` retriever preset (`backend/app/rag/retriever_presets.json`) was chosen. It was
frozen on 2026-10-13 using the DEV split only. Whether Ask the Fed should switch to it was left to a rule decided
before the TEST run, applied once to the TEST split. The rule held, so `tuned` is now the
production default (`RETRIEVER_PRESET`, set in `backend/app/core/config.py`), and `phase3` stays selectable. See
"The switch rule and the TEST result" at the end.

## Protocol

- **DEV only.** All 30 DEV questions were ranked, and the 26 answerable ones were scored (top 10, tie-aware
  metrics from `app/evaluation/retrieval_metrics.py`). No TEST question was ranked or scored, and TEST gold was
  not used. TEST is scored once, after the owner's review of the TEST items, with the frozen preset.
- **Fixed inputs.** Same index (chunker v2, 3,077 chunks, `chunks_sha256` `0f81bdce...c305ca`), with no rebuild
  and no chunk changes. No chat-model calls. Query vectors come from `data/eval_cache/query_embeddings.jsonl`.
  The 18 DEV questions not cached yet were embedded in one request (465 tokens, `text-embedding-3-small`, about
  $0.00001). The tuning harness could not make any other embedding request.
- **Objective.** The highest DEV nDCG@10 of `hybrid_rerank` (the production method), with recall@10 as the
  tie-break. To keep the config simple, the rule picks the config with the fewest non-default options among all
  configs within 0.01 nDCG@10 of the best one evaluated. Remaining ties go to the higher nDCG@10, then the higher
  recall@10. Coupled fields count as one option: prior + half-life, expansion + weight, model + revision.
- **Harness.** `backend/scripts/tune_retriever.py` runs the whole search below:
  `cd backend && .venv/bin/python scripts/tune_retriever.py`. It calls the production
  `get_retriever(method, settings, config)` and the production metric functions, with cross-encoder scores
  memoized per (query, passage) for speed. It ranks DEV questions only, reads query vectors only from the
  cache (an uncached one is an error, and the embedding client refuses any request) and runs Hugging Face
  offline. It writes every configuration's DEV metrics, and the rule's pick, to `data/eval/rag_tuning_dev.json`;
  a rerun should match that file. With `--questions` pointing at the question file from before the owner's
  gold fix, it reproduces the search exactly: the ablation and grid values, the pick and the leave-one-out
  table below (all 144 configurations, `phase3` included). `phase3` and
  the frozen `tuned` preset were then re-run through
  `python -m app.evaluation retrieval --split dev --config phase3 tuned`, which uses no memo. The CLI
  matched the harness to 3 decimals on every metric and method. The intervals below
  come from `python -m app.evaluation report --split dev --config phase3 tuned` (written to
  `data/eval_cache/rag_benchmark.dev-phase3-tuned.json`, never to the tracked benchmark): paired percentile
  bootstrap over questions, 10,000 resamples, seed 20261013.
- **Download.** One extra model was downloaded from Hugging Face:
  `cross-encoder/ms-marco-MiniLM-L-12-v2` at commit `7b0235231ca2674cb8ca8f022859a6eba2b1c968`
  (safetensors). The chosen preset does not use it.

## Search space

| Option | Values tried (phase3 value first) |
|---|---|
| `bm25_stemming` | false, true |
| `query_expansion` + `expansion_weight` | off; on with 0.25, 0.5, 1.0 |
| `date_prior` | 1, 1.5, 2, 3 |
| `year_prior` | 1, 1.25, 1.5 |
| `recency_prior` + `recency_half_life_days` | 0; 0.5 or 1 with 90, 180 or 365 days |
| `statement_boost` | 1, 1.2, 1.5 |
| `rrf_weights` (BM25, dense) | (1, 1), (1, 0.5), (0.5, 1), (0.25, 1) |
| `rrf_k` | 60, 30 |
| `min_candidates` | 50, 100 |
| reranker | MiniLM-L-6 (settings), MiniLM-L-12 |

In all, 143 distinct configurations besides `phase3` were scored on DEV:

1. 23 one-factor ablations, scored with all four methods.
2. A staged grid of 126 configurations for `hybrid_rerank`, besides `phase3`. It includes 6 of the ablations and
   the leave-one-out checks below.
   - **Stage 1.** `date_prior` × `statement_boost` × reranker (24 configs).
   - **Stage 2.** `year_prior` × recency (21 combinations), added on top of each stage-1 config within 0.01 of
     the stage-1 best.
   - **Stage 3.** The options with no clear one-factor gain for `hybrid_rerank`, added one at a time on top of the
     pick.

**Disclosed revision.** The first pass picked a 5-option config that included MiniLM-L-12. A leave-one-out check
on that pick showed L-12 added only +0.004 nDCG@10 there. Stage 2 had run only on L-12 bases: the L-6 config
`date_prior` 3 + `statement_boost` 1.5 scored 0.5877 and missed the stage-1 band by 0.0005. So stages 2 and 3 were
repeated on that L-6 base, and the rule was re-applied over every evaluated config.

The DEV figures below were computed before the owner's gold fix (`docs/rag-eval.md`, "Owner review"), which
turned ten DEV primary chunks into supporting ones. Re-scored on the fixed gold, `hybrid_rerank` nDCG@10 is
0.322 for `phase3` and 0.638 for `tuned`, against 0.319 and 0.637 below. Re-run on the fixed gold after TEST was
scored (`data/eval/rag_tuning_dev.json`), the search takes a slightly different path through the stages (153
configurations, `phase3` included), its best scores 0.642 (`tuned` plus query expansion at weight 0.25), and the rule still picks
`tuned` (0.638). The frozen preset was not changed.

## Baseline (phase3, DEV, 26 answerable)

| Method | recall@5 | recall@10 | MRR@10 | nDCG@10 | doc recall@5 |
|---|---:|---:|---:|---:|---:|
| bm25 | 0.145 | 0.252 | 0.321 | 0.212 | 0.440 |
| dense | 0.256 | 0.389 | 0.406 | 0.314 | 0.514 |
| hybrid | 0.157 | 0.335 | 0.364 | 0.271 | 0.392 |
| hybrid_rerank | 0.277 | 0.368 | 0.417 | 0.319 | 0.560 |

Per-type values are in the table under "DEV gains" (phase3 → tuned).

## One-factor ablations (DEV)

Each cell is the change from phase3 when only this option is set: nDCG@10, with recall@10 in parentheses.

| Option | bm25 | dense | hybrid | hybrid_rerank |
|---|---:|---:|---:|---:|
| bm25_stemming | +0.008 (+0.029) | n/a | +0.044 (−0.013) | −0.011 (−0.008) |
| query_expansion w=0.25 | +0.010 (−0.003) | n/a | +0.012 (+0.025) | +0.002 (−0.003) |
| query_expansion w=0.5 | +0.004 (−0.013) | n/a | +0.011 (−0.016) | −0.007 (−0.016) |
| query_expansion w=1.0 | +0.006 (−0.019) | n/a | +0.020 (−0.007) | −0.010 (−0.016) |
| date_prior 1.5 | +0.108 (+0.088) | +0.224 (+0.117) | +0.118 (+0.112) | +0.140 (+0.136) |
| date_prior 2 | +0.117 (+0.096) | +0.224 (+0.117) | +0.185 (+0.155) | +0.174 (+0.157) |
| date_prior 3 | +0.117 (+0.096) | +0.224 (+0.117) | +0.222 (+0.189) | +0.189 (+0.182) |
| year_prior 1.25 | −0.003 (−0.005) | +0.022 (+0.032) | +0.001 (−0.026) | +0.012 (+0.022) |
| year_prior 1.5 | +0.002 (+0.001) | +0.022 (+0.032) | +0.010 (−0.019) | +0.010 (+0.019) |
| recency 0.5 or 1, 90 d | +0.004 (+0.006) | 0 | 0 | +0.020 (+0.032) |
| recency 0.5 or 1, 180 d | +0.004 (+0.006) | 0 | 0 | +0.015 (+0.022) |
| recency 0.5 or 1, 365 d | +0.004 (+0.006) | 0 | 0 | +0.006 (+0.006) |
| statement_boost 1.2 | 0.000 (−0.026) | +0.047 (−0.044) | +0.040 (+0.043) | +0.087 (+0.028) |
| statement_boost 1.5 | +0.010 (−0.026) | +0.047 (−0.044) | +0.057 (+0.055) | +0.075 (+0.030) |
| rrf_weights (1, 0.5) | n/a | n/a | −0.018 (−0.057) | −0.039 (−0.034) |
| rrf_weights (0.5, 1) | n/a | n/a | +0.024 (+0.039) | −0.000 (+0.012) |
| rrf_weights (0.25, 1) | n/a | n/a | +0.039 (+0.083) | −0.006 (−0.005) |
| rrf_k 30 | n/a | n/a | +0.001 (+0.000) | −0.001 (+0.009) |
| min_candidates 100 | 0 | 0 | −0.011 (−0.032) | −0.012 (−0.012) |
| reranker MiniLM-L-12 | n/a | n/a | n/a | +0.064 (+0.109) |

How to read the table:

- **"0" and "n/a".** "0" means no change. "n/a" marks an option the method does not use.
- **Recency rows.** Prior 0.5 and prior 1 gave the same values to 3 decimals, except bm25 at 90 days (+0.005 with
  1).
- **Gains for `hybrid_rerank`.** `date_prior`, `statement_boost`, the L-12 reranker and, more weakly, `recency_prior`
  and `year_prior` helped. Stemming, query expansion, RRF settings and more candidates did not.

## Grid results (hybrid_rerank, DEV nDCG@10)

| Non-default options | Best config with that many | nDCG@10 | recall@10 |
|---:|---|---:|---:|
| 0 | phase3 | 0.319 | 0.368 |
| 1 | date_prior 3 | 0.508 | 0.551 |
| 2 | + statement_boost 1.5 | 0.588 | 0.574 |
| 3 | + year_prior 1.25 | 0.610 | 0.589 |
| 4 | + recency_prior 1, 90 d (**tuned**) | 0.637 | 0.621 |
| 5 | tuned + MiniLM-L-12 | 0.641 | 0.639 |
| 6 | tuned + MiniLM-L-12 + min_candidates 100 | 0.645 | 0.642 |

The best config evaluated scored 0.645, which puts the band at ≥ 0.635. The fewest-options config in the band is
`tuned`, at 0.637. It sits 0.002 above the band's edge.

## Chosen config (`tuned`)

- **Options.** `date_prior` 3, `statement_boost` 1.5, `year_prior` 1.25, and `recency_prior` 1 with
  `recency_half_life_days` 90.
- **Unchanged from phase3.** BM25 without stemming or expansion, RRF k=60 with equal weights, and
  `min_candidates` 50.
- **Reranker.** MiniLM-L-6 at commit `233902d25c440f23af6f7d6e94d2946bac0bee0a`. This is the reranker the settings
  already use. It is pinned in the preset so that `tuned` does not change if `RERANKER_MODEL` does. Because it is
  the same model, this is not an option.

The preset applies to all four methods. Latency on the DEV CLI run, p50/p95 in ms:

| Method | phase3 | tuned |
|---|---:|---:|
| hybrid_rerank | 276/294 | 275/293 |
| dense | 2/10 | 7/23 |

Dense is slower under `tuned` because with a prior active it fetches 50 candidates instead of 10.

**Each option's contribution.** This is a leave-one-out from `tuned` on `hybrid_rerank`. Each row shows nDCG@10
without the option, then `tuned` minus that, with a 95% CI and the number of the 26 questions whose nDCG@10
changed.

| Without | nDCG@10 | tuned − this [95% CI] | Questions changed |
|---|---:|---:|---:|
| date_prior | 0.443 | +0.193 [+0.116, +0.274] | 16 |
| statement_boost | 0.536 | +0.100 [+0.054, +0.151] | 12 |
| year_prior | 0.602 | +0.034 [+0.003, +0.072] | 6 |
| recency_prior | 0.610 | +0.027 [+0.000, +0.069] | 2 |

Adding MiniLM-L-12 on top of `tuned` gives:

- nDCG@10: +0.004 [−0.051, +0.061].
- recall@10: +0.018 [−0.009, +0.049].

## DEV gains: tuned − phase3, paired, 95% CI (n = 26)

| Method | Metric | phase3 | tuned | Difference [95% CI] |
|---|---|---:|---:|---:|
| hybrid_rerank | recall@5 | 0.277 | 0.469 | +0.192 [+0.108, +0.288] |
| hybrid_rerank | recall@10 | 0.368 | 0.621 | +0.253 [+0.156, +0.357] |
| hybrid_rerank | MRR@10 | 0.417 | 0.881 | +0.465 [+0.315, +0.615] |
| hybrid_rerank | nDCG@10 | 0.319 | 0.637 | +0.318 [+0.231, +0.408] |
| hybrid_rerank | doc recall@5 | 0.560 | 0.679 | +0.119 [−0.003, +0.250] |
| hybrid | recall@5 | 0.157 | 0.448 | +0.291 [+0.177, +0.418] |
| hybrid | recall@10 | 0.335 | 0.551 | +0.216 [+0.129, +0.317] |
| hybrid | MRR@10 | 0.364 | 0.679 | +0.314 [+0.185, +0.454] |
| hybrid | nDCG@10 | 0.271 | 0.546 | +0.275 [+0.183, +0.378] |
| hybrid | doc recall@5 | 0.392 | 0.620 | +0.228 [+0.119, +0.349] |
| dense | recall@5 | 0.256 | 0.432 | +0.176 [+0.104, +0.251] |
| dense | recall@10 | 0.389 | 0.508 | +0.119 [+0.024, +0.206] |
| dense | MRR@10 | 0.406 | 0.768 | +0.362 [+0.221, +0.506] |
| dense | nDCG@10 | 0.314 | 0.570 | +0.256 [+0.176, +0.335] |
| dense | doc recall@5 | 0.514 | 0.638 | +0.124 [+0.026, +0.223] |
| bm25 | recall@5 | 0.145 | 0.275 | +0.129 [+0.040, +0.238] |
| bm25 | recall@10 | 0.252 | 0.349 | +0.097 [+0.038, +0.162] |
| bm25 | MRR@10 | 0.321 | 0.472 | +0.151 [+0.061, +0.252] |
| bm25 | nDCG@10 | 0.212 | 0.340 | +0.128 [+0.062, +0.203] |
| bm25 | doc recall@5 | 0.440 | 0.476 | +0.035 [−0.061, +0.151] |

**Per type, nDCG@10, phase3 → tuned** (n is the number of answerable DEV questions of that type):

| Method | date_specific (4) | multi_doc (4) | minutes_fact (6) | point_in_time (4) | paraphrase (3) | statement_fact (5) |
|---|---:|---:|---:|---:|---:|---:|
| bm25 | 0.378 → 0.665 | 0.032 → 0.032 | 0.390 → 0.663 | 0.220 → 0.245 | 0.045 → 0.044 | 0.105 → 0.193 |
| dense | 0.321 → 0.712 | 0.355 → 0.606 | 0.488 → 0.858 | 0.298 → 0.358 | 0.047 → 0.251 | 0.240 → 0.446 |
| hybrid | 0.434 → 0.775 | 0.085 → 0.564 | 0.527 → 0.802 | 0.239 → 0.397 | 0.000 → 0.143 | 0.172 → 0.403 |
| hybrid_rerank | 0.662 → 0.879 | 0.059 → 0.562 | 0.601 → 0.827 | 0.164 → 0.486 | 0.086 → 0.245 | 0.177 → 0.629 |

**Per question, nDCG@10, tuned vs phase3:**

| Method | Better | Worse | Same |
|---|---:|---:|---:|
| hybrid_rerank | 22 | 2 | 2 |
| hybrid | 20 | 0 | 6 |
| dense | 20 | 0 | 6 |
| bm25 | 13 | 3 | 10 |

**Not every slice improved:**

- Dense recall@10 on `point_in_time` fell from 0.438 to 0.292.
- bm25 recall@10 on `paraphrase` fell from 0.081 to 0.033.

**Other methods.** These options were left out because they do not help `hybrid_rerank`, but they would help other
methods on top of `tuned` (DEV nDCG@10 changes):

| Option on top of `tuned` | bm25 | hybrid | hybrid_rerank |
|---|---:|---:|---:|
| query_expansion w=1.0 | +0.115 | +0.058 | −0.005 |
| bm25_stemming | +0.069 | +0.023 | −0.005 |
| rrf_weights (0.5, 1) | n/a | +0.037 | −0.031 |

## Overfitting caveat

- **Selection bias.** The DEV numbers for `tuned` are optimistic. The preset was chosen as the best of 126 grid
  configs on the same 26 questions it is scored on. The intervals above treat the config as fixed, so they do not
  account for that selection. Only the TEST score measures the frozen preset without this bias.
- **Dates drive most of the gain.** 20 of the 26 answerable DEV questions name a month and year, or a day, and `date_prior`
  alone adds +0.19 nDCG@10. On traffic that names meeting dates less often, the gain will be smaller.
  `date_prior` 3 is also the largest value tried.
- **Thin evidence for two options.** Only 2 DEV questions trigger `recency_prior` (an `as_of` date plus wording
  like "still" or "currently"). `year_prior` changed 6 questions. Both leave-one-out intervals reach down to about
  0.
- **A close call on the reranker.** The pick is 0.0017 above the band's edge. Had the best config scored 0.002
  higher, `tuned` would have fallen out of the band, and the rule would have picked `tuned` + MiniLM-L-12 (0.641). The selection procedure was also revised once during the search
  (see "Disclosed revision" above).
- **Small groups.** Per-type means rest on 3–6 questions each.

## The switch rule and the TEST result

**The rule (decided before the TEST run, unchanged).** It was recorded by the project itself, and no
timestamped copy predates the run: the repository had no commits yet, and the earliest file found holding it was
written after TEST was ranked, answered and judged. Later phases should commit such a rule, or a sha256 of it,
before scoring TEST. Switch the production default from `phase3` to `tuned` only if, on TEST:

- (a) the paired-bootstrap 95% CI of `hybrid_rerank` nDCG@10, `tuned` minus `phase3`, lies entirely above 0;
- (b) `hybrid_rerank` answer_correctness is not worse: the paired difference, `tuned` minus `phase3`, has a CI
  lower bound above −0.05.

Otherwise `phase3` stays the default. The rule is coded as `switch_selection` in
`backend/app/evaluation/report.py`. `make rag-eval-report` applies it (`report --switch-rule`) and writes the
outcome to `selection` in `data/eval/rag_benchmark.json`.

**TEST result** (55 questions, 50 of them answerable; one run; 10,000 paired resamples, seed 20261013):

| Condition | tuned − phase3 [95% CI] | n | Holds? |
|---|---:|---:|---|
| (a) `hybrid_rerank` nDCG@10 | +0.222 [+0.158, +0.293] | 50 | yes: the CI is above 0 |
| (b) `hybrid_rerank` answer_correctness (LLM judge) | +0.140 [+0.068, +0.218] | 50 | yes: the lower bound is above −0.05 |

Both hold, so **`tuned` became the default** for `POST /api/ask`, `/api/ask/stream`, `make ask` and
`make smoke`. The default is set in one place, `Settings.retriever_preset`, through the `RETRIEVER_PRESET`
environment variable. `RETRIEVER_PRESET=phase3`, `make ask PRESET=phase3` or `smoke --preset phase3` bring
back the Phase 3 retriever, and an unknown name stops the API at startup.

**DEV to TEST shrinkage.** The `hybrid_rerank` nDCG@10 gain is smaller on TEST (+0.222) than on DEV (+0.318
before the gold fix, +0.317 after). That is consistent with the selection bias described above, but the gap
itself is not resolved (unpaired bootstrap CI about [−0.02, +0.21]), and TEST has 5 questions that name no
time, which DEV lacks. `phase3`'s own MRR@10 also differs between the splits (0.417 DEV, 0.502 TEST), so the
drop in `tuned` MRR@10 (0.881 to 0.674) is not by itself evidence of selection bias. The gain is smaller, but
its interval is still well clear of 0.

**Where the gain comes from (post hoc, not part of the rule).** 40 of the 50 answerable TEST questions name a
month or a day. On those 40, the nDCG@10 gain is +0.255 [+0.178, +0.337]. Of the 10 that name no month or
day, 5 name a year (`md-03`, `md-06`, `pp-01`, `pp-05`, `pp-09`): +0.188 [+0.121, +0.243], where `year_prior`
applies. The 5 that name no meeting, month or year (`md-05`, `pit-04`, `pit-05`, `pit-07`, `pp-03`) gain
−0.006 [−0.018, 0.000], although `recency_prior` fired on `pit-04`, `pit-05` and `pit-07`. DEV had no such
question. On questions that name no time at all, TEST shows no measurable gain (n = 5).

