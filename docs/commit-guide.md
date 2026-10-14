# First commits: a suggested plan

The repository has no commits yet: branch `main`, no remote, every file untracked. This guide
turns the finished project into eight commits, one per phase, runs the checks that should pass
before anything leaves your machine, and then pushes to GitHub. Every command here is yours to run.

**What the history will and will not show.** Each file is committed once, in its final form, in
the phase that introduced it. Files that every phase touched (the Makefile, `app/main.py`, the API
schemas, the generated API types, the root layout and home page, the shared design-system
components in `frontend/src/components/ui/`) go into the first commit as they are today, and a
later file inside a folder that a phase adds whole (such as the recorded-answers view in
`frontend/src/components/ask/`) goes in with that folder. The commits therefore show how the
project is organised, phase by phase, not how each file evolved, and an intermediate commit may
not build on its own: CI runs on the commit you push last. If you prefer one honest commit, skip
to [the alternative](#alternative-one-commit).

The eight commits are on GitHub now. The packaging of the API as a Vercel function came after
them: commit and push it as [section 4](#4-after-the-first-push-the-vercel-packaging) shows.

## 0. Before the first commit

Run every command from the repository root. The blocks below carry `#` comments, which zsh (the
default shell on macOS) treats as commands unless you run this once in the terminal first:

```bash
setopt interactivecomments
```

Commits use your global git name and email. To keep your email private, set the GitHub "noreply"
address for this repository instead (GitHub → Settings → Emails → "Keep my email addresses
private" shows it); copy the real address, not a placeholder:

```bash
git config user.email "1234567+your-user@users.noreply.github.com"   # optional; your own address
```

Then:

```bash
make check                       # must pass (contract-check is skipped until the first commit)
git status --short --ignored     # see "What must stay ignored" below
```

### What must stay ignored

In `git status --short --ignored`, ignored paths are marked `!!`. Check that these are among them,
and that none of them shows up as untracked (`??`):

- `backend/.env` (your OpenAI key and database URL) and any other `.env*` except the two
  `.env.example` files;
- `backend/.venv/`, `frontend/node_modules/`, `frontend/.next/`, and caches and generated files
  (`__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `backend/fedrag_backend.egg-info/`,
  `frontend/next-env.d.ts`, `frontend/tsconfig.tsbuildinfo`);
- `data/raw/*`, `data/processed/*` (only their `.gitkeep` files are tracked), `data/chroma/`,
  `data/index/*` (except `chunks.jsonl` and `manifest.json`, tracked since the Vercel packaging for
  the public demo), `data/external/`, `data/models/`, `data/stance/`, `data/eval_cache/`;
- `deploy/build/` (the Hugging Face Space folder that `make space-bundle` assembles, and the
  virtualenv and file copy of `make vercel-check`);
- `.vercel/`, `pyproject.toml` and `uv.lock` in the repository root, if you ever ran `vercel build`
  there (Vercel would install that `pyproject.toml` instead of `requirements.txt`);
- `.claude/launch.json`, `.claude/settings.local.json`, `.DS_Store`.

`data/eval/` is tracked on purpose: the questions and results are part of the project's claims.

## 1. The eight commits

Run each block in order. `git add` with a directory adds everything under it that is not ignored.
After each `git add`, `git status --short` shows what is staged (`A`) and what is still untracked
(`??`), so a missing or unexpected path is easy to spot before you commit.

### 1. Scaffold (Phase 1)

```bash
git add .gitignore LICENSE Makefile .github \
  backend/pyproject.toml backend/constraints.txt backend/.env.example \
  backend/app/__init__.py backend/app/main.py backend/app/core backend/app/schemas \
  backend/app/api/__init__.py backend/app/api/router.py backend/app/api/stubs.py \
  backend/app/api/routes/__init__.py backend/app/api/routes/health.py \
  backend/scripts/export_openapi.py \
  backend/tests/conftest.py backend/tests/test_config.py backend/tests/test_contract.py \
  backend/tests/test_health.py backend/tests/test_openapi.py \
  frontend/.env.example frontend/.gitignore frontend/README.md frontend/eslint.config.mjs \
  frontend/next.config.ts frontend/package.json frontend/package-lock.json frontend/tsconfig.json \
  frontend/vitest.config.mts frontend/openapi.json \
  frontend/src/app/globals.css frontend/src/app/icon.svg frontend/src/app/layout.tsx \
  frontend/src/app/not-found.tsx frontend/src/app/page.tsx \
  frontend/src/components/BackendStatus.tsx frontend/src/components/DataTable.tsx \
  frontend/src/components/Footer.tsx frontend/src/components/Header.tsx \
  frontend/src/components/PageHeader.tsx frontend/src/components/TabNav.tsx \
  frontend/src/components/ui \
  frontend/src/lib/api/client.ts frontend/src/lib/api/types.ts frontend/src/lib/api/schema.d.ts \
  data/raw/.gitkeep data/processed/.gitkeep data/eval/.gitkeep
git commit -m "chore: scaffold FastAPI backend, Next.js frontend and typed API contract" \
  -m "Monorepo with a FastAPI app, Pydantic schemas exported as OpenAPI and generated TypeScript types, a Next.js app with four tabs, a Makefile and a GitHub Actions workflow."
```

### 2. Data pipeline (Phase 2)

```bash
git add backend/app/ingest backend/app/db/__init__.py backend/app/db/connection.py \
  backend/app/db/load.py backend/app/db/queries.py backend/app/db/schema.sql \
  backend/app/api/database.py backend/app/api/routes/meetings.py \
  backend/tests/ingest backend/tests/fixtures backend/tests/db backend/tests/test_meetings.py \
  data/README.md
git commit -m "feat(data): scrape FOMC statements and minutes into Postgres with release dates" \
  -m "Calendar, statement and minutes parsers for 2015 on; Fed-sourced release_date for every document; a quality report; an idempotent loader into schema fedrag; GET /api/meetings with the as_of filter."
```

### 3. RAG pipeline (Phase 3)

```bash
git add backend/app/rag backend/app/api/rag.py backend/app/api/routes/ask.py \
  backend/tests/rag backend/tests/test_ask.py backend/tests/test_warmup.py \
  data/eval/smoke.jsonl
git commit -m "feat(rag): section-aware chunking, hybrid retrieval and grounded answers" \
  -m "Incremental embedding index in Chroma; BM25, dense, RRF hybrid and cross-encoder rerank, all filtered by as_of before ranking; answers that cite every claim; POST /api/ask; a 12-question smoke check."
```

### 4. Ask the Fed (Phase 4)

```bash
git add backend/app/api/event_stream.py backend/tests/test_ask_stream.py \
  frontend/src/app/ask frontend/src/components/ask frontend/src/lib/ask \
  frontend/src/lib/sse.ts frontend/src/lib/sse.test.ts \
  frontend/src/lib/api/stream.ts frontend/src/lib/api/stream.test.ts \
  docs/images/ask-desktop-light.png docs/images/ask-mobile-dark.jpg
git commit -m "feat(ask): stream answers with live citations on the Ask the Fed page" \
  -m "POST /api/ask/stream (server-sent events, markers renumbered as they stream, cancelled on disconnect); the /ask page with citation cards, as-of date, method controls and shareable URLs."
```

### 5. Policy Stance (Phase 5)

```bash
git add backend/app/stance backend/app/db/stance.py backend/app/api/stance.py \
  backend/app/api/routes/stance.py backend/tests/stance backend/tests/test_stance.py \
  data/eval/stance_benchmark.json data/eval/stance_gold.jsonl data/eval/stance_review_items.json \
  docs/stance-labeling.md \
  frontend/src/app/stance frontend/src/components/stance frontend/src/lib/stance \
  docs/images/stance-desktop-light.png
git commit -m "feat(stance): classify statement stance with four models and benchmark them" \
  -m "Keyword lexicon, FinBERT fine-tuned on Trillion Dollar Words, gpt-5.4-mini zero- and few-shot; sentence and statement benchmarks with bootstrap intervals; the /stance timeline against the fed funds target range."
```

### 6. Statement Comparison (Phase 6)

```bash
git add backend/app/compare backend/app/db/compare.py backend/app/api/compare.py \
  backend/app/api/routes/compare.py backend/tests/compare \
  data/eval/compare_benchmark.json \
  frontend/src/app/compare frontend/src/components/compare frontend/src/lib/compare \
  docs/images/compare-desktop-light.png docs/images/compare-analysis-desktop-light.png
git commit -m "feat(compare): sentence diff of two statements with a cached LLM reading" \
  -m "Needleman-Wunsch alignment over word-token similarity with word-level diffs; a structured LLM analysis checked against the diff and cached per pair in Postgres; POST /api/compare and the /compare page."
```

### 7. RAG evaluation (Phase 7)

```bash
git add backend/app/evaluation backend/app/api/evaluation.py backend/app/api/routes/evaluation.py \
  backend/scripts/tune_retriever.py backend/tests/evaluation backend/tests/test_evaluation.py \
  data/eval/rag_questions.jsonl data/eval/rag_review_items.json data/eval/rag_benchmark.json \
  data/eval/rag_tuning_dev.json data/eval/rag_judge_sample.json \
  data/eval/rag_judge_sample_key.json data/eval/rag_judge_annotations.json \
  docs/rag-eval.md docs/rag-tuning.md \
  frontend/src/app/performance frontend/src/components/performance frontend/src/lib/performance \
  docs/images/performance-desktop-light.png
git commit -m "feat(eval): benchmark retrieval and answers on a held-out question set" \
  -m "85 questions (30 DEV / 55 TEST) with chunk-level gold; tie-aware recall, MRR and nDCG; a gpt-5.5 judge checked against annotators; DEV-only tuning and a pre-registered switch rule that made the tuned retriever the default; the /performance page."
```

### 8. Deployment (Phase 8)

```bash
git add deploy scripts/build_hf_space.sh data/eval/ask_examples.json \
  backend/app/api/demo.py backend/app/api/ask_examples.py backend/app/api/file_cache.py \
  backend/tests/test_demo.py backend/tests/demo_import_probe.py \
  frontend/src/app/opengraph-image.tsx frontend/src/app/twitter-image.tsx \
  frontend/src/components/ShareImage.tsx frontend/src/components/WakeNotice.tsx \
  frontend/src/components/useBackend.ts frontend/src/components/landing \
  frontend/src/lib/api/backend.ts frontend/src/lib/api/backend.test.ts \
  frontend/src/lib/api/hosting.ts frontend/src/lib/api/hosting.test.ts frontend/src/lib/api/wake.ts \
  frontend/src/lib/landing frontend/src/lib/site.ts frontend/src/lib/site.test.ts \
  docs/deploy.md docs/commit-guide.md docs/technical-report.md README.md \
  docs/images/home-desktop-light.png docs/images/ask-demo-desktop-light.png \
  docs/images/ask-tile-light.png docs/images/stance-tile-light.png \
  docs/images/compare-tile-light.png docs/images/performance-tile-light.png
git status --short               # anything still marked ?? belongs in this commit: add it, or ask why
git commit -m "feat(deploy): public demo mode, landing page and Hugging Face Spaces + Vercel packaging" \
  -m "PUBLIC_DEMO turns off every OpenAI path (Ask returns 403, recorded answers at GET /api/ask/examples, no new comparison analyses); the frontend adapts to it from /api/health and waits for a sleeping Space; a landing page with link-preview images; a Docker Space bundle, a read-only Neon role, the deploy guide, a short README and the full technical report."
```

### Nothing left behind

```bash
git status --short                               # prints nothing
git ls-files --others --exclude-standard         # prints nothing
git log --oneline                                # eight commits
make check                                       # now with the real API contract drift check
```

## 2. Pre-push checklist

Run these after the last commit and before the first push. Each says what you should see.

```bash
# 1. Secrets: no .env file is tracked, only the two templates.
git ls-files | grep -E '(^|/)\.env' 
#    expected: backend/.env.example and frontend/.env.example, nothing else

# 2. Ignored data is not tracked: only the .gitkeep placeholders, and the search index's passages
#    and manifest, which the Vercel demo serves (tracked since the Vercel deployment).
git ls-files data/raw data/processed data/chroma data/index data/external data/models \
  data/stance data/eval_cache
#    expected: data/index/chunks.jsonl, data/index/manifest.json, data/processed/.gitkeep and
#    data/raw/.gitkeep (before the Vercel deployment: only the two .gitkeep files)

# 3. No build output or dependencies.
git ls-files deploy/build frontend/.next frontend/node_modules backend/.venv \
  .vercel pyproject.toml uv.lock | head
#    expected: nothing

# 4. No key or password in any tracked file (every match is shown cut short, never in full).
git grep -hoIE 'sk-(proj-)?[A-Za-z0-9_-]{20,}|npg_[A-Za-z0-9]{8,}|postgres(ql)?://[^:/@ "]+:[^@/ "]+@' \
  | sed -E 's#(://[^:]+:)(...)[^@]*@#\1\2...@#; s#^(sk-|npg_)(...).*#\1\2...#' | sort | uniq -c
#    expected: only placeholders and test values, i.e.
#      USER:PAS...@                (backend/.env.example, backend/app/db/connection.py)
#      fedrag:fed...@              (Makefile, CI, backend/tests/db, this guide)
#      fedrag_demo_reader:<NE...@  (docs/deploy.md) and fedrag_demo_reader:<pa...@
#                                  (deploy/README.md, deploy/neon_readonly.sql): placeholders
#      u:p@, u:Sen...@, fedrag:hun...@, fedrag:sec...@   (fake values in backend/tests)
#    No "sk-" or "npg_" line at all.

# 5. Size: the biggest tracked files and the total.
git ls-files -z | xargs -0 du -k | sort -n | tail -8
git ls-files -z | xargs -0 du -ck | tail -1
#    expected: data/eval/rag_benchmark.json (about 2.5 MB) and the screenshots are the largest;
#    the total is about 11 MB. Since the Vercel deployment, data/index/chunks.jsonl (about 7 MB)
#    leads and the total is about 19 MB. GitHub warns at 50 MB per file and refuses 100 MB.

# 6. Everything CI runs, except the Postgres tests.
make check
#    optional, with Docker running: make db-up, then
#    TEST_DATABASE_URL=postgresql://fedrag:fedrag@127.0.0.1:55432/fedrag_test make test-db
```

If a check fails after committing, fix it with a new commit, or, since nothing has been pushed
yet, amend the commit concerned. If a secret was ever committed, do not push: remove it from
history first (the simplest way before a first push is to delete `.git` and start this guide
again), and rotate the secret anyway.

## 3. Create the GitHub repository and push

1. On [github.com/new](https://github.com/new): owner `<GITHUB_USER>`, name `fedrag`, **Public**,
   and **no** README, `.gitignore` or licence (the repository has its own).
2. Push:

   ```bash
   git remote add origin https://github.com/<GITHUB_USER>/fedrag.git
   git push -u origin main
   ```

3. Open the repository's **Actions** tab and check that CI passes (backend with Postgres,
   frontend, API contract).
4. Then deploy: [`deploy.md`](deploy.md).

## 4. After the first push: the Vercel packaging

Hugging Face now charges for Docker Spaces, so the public demo's API moved to a Vercel function
after the eight commits above were pushed. That work is one more commit on `main`: the entrypoint,
its pinned packages and settings, the search index's passages (6.9 MB, which the function serves),
the local check, and the updated guides and wording. Run from the repository root:

```bash
make check                       # must pass, now with the API contract drift check
git status --short --ignored     # "What must stay ignored" above still holds
git add api requirements.txt .python-version vercel.json .vercelignore \
  data/index/chunks.jsonl data/index/manifest.json \
  scripts/check_vercel.sh scripts/build_hf_space.sh \
  backend/tests/test_vercel.py backend/tests/vercel_entry_probe.py \
  backend/tests/test_config.py backend/tests/test_demo.py backend/.env.example \
  .gitignore .github/workflows/ci.yml Makefile README.md data/README.md deploy docs \
  frontend/.env.example frontend/README.md frontend/next.config.ts frontend/src
git status --short               # every line staged (A or M in the first column), nothing left ??
git commit -m "feat(deploy): host the demo API as a Vercel Python function" \
  -m "Hugging Face Docker Spaces now need PRO, so the API runs as a second Vercel project from this repository: api/index.py (always PUBLIC_DEMO), requirements.txt with the 49 packages the demo imports, Python 3.13, vercel.json (FastAPI preset, region cle1, excludeFiles) and the tracked search-index passages. make vercel-check and backend/tests/test_vercel.py check the install, the entrypoint and every demo call; the Space stays as the documented alternative."
```

Then run the [pre-push checklist](#2-pre-push-checklist) again (checks 2 and 5 now expect the two
`data/index` files), push with `git push`, check CI in the **Actions** tab, and deploy:
[`deploy.md`](deploy.md).

## Alternative: one commit

```bash
git add -A
git status --short               # review every path; nothing from "What must stay ignored"
git commit -m "feat: FedRAG, grounded Q&A and analysis of FOMC communications" \
  -m "Phases 1-8: data pipeline, hybrid RAG with citations, Ask the Fed, policy stance, statement comparison, RAG evaluation, public demo packaging."
```

Then run the pre-push checklist above.
