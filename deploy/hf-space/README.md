---
title: FedRAG API
emoji: 🏦
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
short_description: FedRAG demo API: FOMC Q&A, stance and statement diffs
license: mit
pinned: false
---

# FedRAG API (public demo)

The FastAPI backend of [FedRAG](https://github.com/hoedongkim/fedrag), which analyses Federal
Reserve (FOMC) statements and minutes: grounded answers with citations, the policy stance of every
statement, sentence-level comparison of two statements, and a benchmark of the retrieval behind
the answers. The web app (Next.js) proxies its `/api` calls to this Space.

This Space runs with `PUBLIC_DEMO=true`, so it makes **no OpenAI calls** and costs nothing to run:

| Endpoint | On this demo |
|---|---|
| `GET /api/health` | Live; reports `demo_mode: true` |
| `GET /api/meetings`, `GET /api/stance*` | Live, from Postgres (a read-only role) |
| `POST /api/compare` | The sentence diff and the stored LLM analysis of a pair (stored ahead of time for every pair of consecutive statements); a pair without one reports `analysis_status: "disabled_on_demo"` |
| `GET /api/evaluation`, `GET /api/evaluation/questions/{id}` | Live, from the tracked benchmark and the search index's passages |
| `GET /api/ask/examples` | Eight real answers recorded during the evaluation, with citations and the judge's verdicts |
| `POST /api/ask`, `POST /api/ask/stream` | `403`: run FedRAG locally to ask your own questions |

Interactive API docs: `/docs`.

## Configuration

Set in the Space's settings, never in this folder:

- **Secret `DATABASE_URL`**: a Postgres connection string for a role that can only read schema
  `fedrag` (the repository's `deploy/neon_readonly.sql` creates one on Neon).
- **Variable `CORS_ORIGINS`** (optional): a JSON list of browser origins allowed to call this API
  directly, e.g. `["https://your-app.vercel.app"]` (one origin, or several separated by commas,
  works too). The web app proxies `/api` on its server, so it does not need this.
- `PUBLIC_DEMO=true` is the image's default. Do not turn it off here: the Space has no OpenAI key
  and no vector store.

## Contents

Built by `scripts/build_hf_space.sh` in the FedRAG repository: the backend's code with its
pinned dependencies (CPU-only PyTorch), the evaluation and stance results (`data/eval`), and the
search index's passages (`data/index/chunks.jsonl`, for the evaluation's passage drilldowns). No
vector store, model weights, caches or secrets.

FOMC statements and minutes are works of the US government. The code is MIT-licensed (`LICENSE`).
