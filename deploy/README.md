# Deploying the FedRAG public demo

This folder holds the deployment packaging. The full step-by-step guide, with the checks to run
after each step, is [`docs/deploy.md`](../docs/deploy.md); the steps below are its summary.

The public demo costs nothing to run and makes no OpenAI calls:

| Piece | Where | Plan |
|---|---|---|
| Frontend (Next.js) | Vercel, root directory `frontend` | Hobby (free) |
| Backend (FastAPI) | Hugging Face Space, Docker SDK, `PUBLIC_DEMO=true` | CPU basic (free: 2 vCPU, 16 GB RAM; sleeps after 48 h without traffic) |
| Database | Neon, the existing project, through a read-only role | Free |

With `PUBLIC_DEMO=true` the backend serves everything that reads stored results (meetings, stance,
stored comparison analyses, the RAG evaluation, eight recorded example answers) and nothing that
would call OpenAI: `POST /api/ask` and `/api/ask/stream` answer `403`, and a comparison without a
stored analysis reports `analysis_status: "disabled_on_demo"`. It needs no OpenAI key, loads no
vector store or reranker, and never imports torch or the openai package
(`backend/app/api/demo.py`; `backend/tests/test_demo.py` checks it in a fresh interpreter).
`GET /api/health` reports `demo_mode`, which the frontend adapts to.

## What is in this folder

| File | What it is |
|---|---|
| `hf-space/Dockerfile` | The Space's image: Python 3.13 slim, CPU-only PyTorch from PyTorch's index, every other package at the version `backend/constraints.txt` pins, a non-root user (uid 1000, as Spaces require), uvicorn on port 7860, `PUBLIC_DEMO=true` |
| `hf-space/README.md` | The Space's README, with the YAML header Hugging Face reads (`sdk: docker`, `app_port: 7860`, title, emoji, colors, short description) |
| `hf-space/.dockerignore` | Keeps caches and any `.env` out of the build context |
| `neon_readonly.sql` | Creates `fedrag_demo_reader`, a role that can only `SELECT` from schema `fedrag`, with read-only sessions, and checks it |
| `build/` | Output of `make space-bundle` (git-ignored) |

`make space-bundle` (`scripts/build_hf_space.sh`) assembles the Space's whole repository in
`deploy/build/hf-space`: the files above, `LICENSE`, the backend's code, `constraints.txt`, a
`requirements.txt` generated from `backend/pyproject.toml`, the tracked results in `data/eval`
(`*.json` and the stance reference labels) and the search index's passages
(`data/index/chunks.jsonl` and `manifest.json`, which the evaluation's passage drilldown reads).
Never the vector store (`data/chroma`), models, caches (`data/eval_cache`), raw or processed
documents, or `.env`; the script fails if any file looks like an API key, a Neon password, a
Postgres URL with a password, or holds the `OPENAI_API_KEY` or `DATABASE_URL` value in
`backend/.env`. The folder is about 12 MB; its largest file,
`chunks.jsonl`, is 6.9 MB, under the 10 MB a plain git push to Hugging Face accepts without LFS.

## Steps

These are the owner's steps: each one needs an account or a secret.

### 1. A read-only database role (Neon)

Open `deploy/neon_readonly.sql`, replace `<choose-a-strong-password>` (for example with the output
of `openssl rand -hex 24`), and run it in the Neon SQL editor on the database that holds schema
`fedrag`, as the role that loads FedRAG (usually `neondb_owner`). The checks at the end should
show `can_select` true and `can_write` false for every table, no `CREATE` anywhere, and an empty
`member_of`. Then take Neon's **pooled** connection string and put the new role and password in
it:

```
postgresql://fedrag_demo_reader:<password>@<endpoint>-pooler.<region>.aws.neon.tech/neondb?sslmode=require
```

This is the Space's `DATABASE_URL`. Do not commit it or paste it into any file in the repository.

### 2. Build the Space folder

```bash
make ask-examples   # only after re-running the evaluation: refreshes data/eval/ask_examples.json
make space-bundle   # -> deploy/build/hf-space
```

The bundle needs `data/index/chunks.jsonl` and `manifest.json` from `make index` (git-ignored, so a
fresh clone needs one `make index`), and the tracked files in `data/eval`.

Optionally, check the image locally first (Docker Desktop running). The URL is read without
echoing it and kept in a file outside the repository that only you can read, so it never appears
on the command line or in your shell history (as in docs/deploy.md, 2.3):

```bash
docker build -t fedrag-space deploy/build/hf-space
read -rs RO_URL   # paste the read-only URL from step 1
(umask 077; printf 'DATABASE_URL=%s\n' "$RO_URL" > ~/.fedrag-space.env); unset RO_URL
docker run --rm -p 127.0.0.1:7860:7860 --env-file ~/.fedrag-space.env fedrag-space
curl -s http://127.0.0.1:7860/api/health    # "demo_mode": true (in another terminal)
```

Delete `~/.fedrag-space.env` when you are done with it.

### 3. Create the Space and set its secret

1. On huggingface.co, **New Space**: a name (e.g. `fedrag`), SDK **Docker** (blank
   template), hardware **CPU basic** (free), public.
2. Space **Settings > Variables and secrets**:
   - **Secret** `DATABASE_URL`: the URL from step 1.
   - **Variable** `CORS_ORIGINS` (optional): a JSON list of browser origins allowed to call the
     Space directly, e.g. `["https://fedrag.vercel.app"]` (one origin, or several separated by
     commas, works too). The frontend proxies `/api` through its own server, so it works without
     this.
   - Do not add `OPENAI_API_KEY`, and leave `PUBLIC_DEMO` at the image's `true`.

The Space's URL is `https://<user>-<space>.hf.space` (e.g. `https://hoedongkim-fedrag.hf.space`).

### 4. Upload the folder

With the Hugging Face CLI that `make setup` installed (`backend/.venv/bin/hf`; do not upgrade it
there, since `backend/constraints.txt` pins it), after `backend/.venv/bin/hf auth login` with a
write token:

```bash
backend/.venv/bin/hf upload <user>/<space> deploy/build/hf-space . --repo-type space
```

Or with git (no LFS needed: every file is under 10 MB). Clone outside this repository, so the
Space's repository is never committed into it:

```bash
git clone https://huggingface.co/spaces/<user>/<space> ../<space>-space
rsync -a --delete --exclude .git deploy/build/hf-space/ ../<space>-space/
(cd ../<space>-space && git add -A && git commit -m "FedRAG API" && git push)
```

`hf upload` adds and replaces files but does not delete ones you removed locally (add
`--delete '*'` for that, as docs/deploy.md does for a rollback); the git route mirrors the folder
exactly. Each push rebuilds the image (a few minutes: the dependencies are
most of it). When the Space shows **Running**:

```bash
curl -s https://<user>-<space>.hf.space/api/health          # demo_mode: true, database_configured: true
curl -s https://<user>-<space>.hf.space/api/meetings | head -c 300
curl -s -X POST https://<user>-<space>.hf.space/api/ask \
     -H 'Content-Type: application/json' -d '{"question": "What did the FOMC decide?"}'   # 403
```

### 5. The frontend on Vercel

Import the GitHub repository in Vercel with **Root Directory** `frontend` (Next.js is detected; no
`vercel.json` is needed) and set the environment variable

```
BACKEND_URL=https://<user>-<space>.hf.space
```

for Production (and Preview, if you use preview deployments). The `/api` rewrite reads it when the
app is built, so redeploy after changing it.

## Updating

- Code or results changed: `make space-bundle`, then upload again (step 4).
- New FOMC meeting: locally `make ingest`, `make db-load`, `make stance`,
  `make compare-precompute` (they write to Neon with your own role), then `make index` and
  `make space-bundle` and upload, so the drilldown passages match. The Space reads the database
  live, so meetings, stance and comparisons update without a redeploy.
- Rotate the database password: `ALTER ROLE fedrag_demo_reader WITH PASSWORD '...'` in Neon, then
  update the Space secret (the Space restarts).

## Measured locally

The image built from this folder and run with Docker Desktop on an Apple Silicon Mac (linux/arm64;
the Space builds linux/amd64), with `PUBLIC_DEMO=true` and a `DATABASE_URL` pointing at the Neon
database (the owner's own role, passed at run time: the read-only role of step 1 did not exist yet,
and the demo only runs `SELECT`s):

| | |
|---|---|
| Image | 2.42 GB unpacked (site-packages 1.7 GB, of which torch 652 MB), about 495 MB compressed |
| Cold start | 1.3-1.9 s from `docker run` to the first `200` from `/api/health` (image already local); 2.8 s for the first run right after Docker Desktop starts |
| Memory | 120 MB RSS idle after startup; 158 MB after serving every endpoint once (the evaluation file and the 3,077 passages stay cached) |
| Loaded | no libtorch, Chroma or onnxruntime library in the server process |

The linux/amd64 image, built here under emulation as the Space will build it, is about 535 MB
compressed (site-packages 1.8 GB, torch 769 MB), has the same packages at the pinned versions
(`torch 2.14.1+cpu`, no CUDA packages; 130 in `pip list` on arm64), and served every endpoint the
same way. On the Space, a
request to a sleeping Space also waits for Hugging Face to start the container.
