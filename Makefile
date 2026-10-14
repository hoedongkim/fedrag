# FedRAG developer commands. Run `make` (or `make help`) to list them.
# No virtualenv activation needed: backend tools run straight from backend/.venv/bin.

SHELL := /bin/bash
.DEFAULT_GOAL := help

PYTHON ?= python3
BACKEND := backend
FRONTEND := frontend
VENV_BIN := $(BACKEND)/.venv/bin
API_FILES := $(FRONTEND)/openapi.json $(FRONTEND)/src/lib/api/schema.d.ts

# Local Postgres for db-up/db-down: a throwaway Docker container reachable only from this machine.
DB_CONTAINER := fedrag-postgres
DB_IMAGE := postgres:17
DB_PORT := 55432
LOCAL_DB_URL := postgresql://fedrag:fedrag@127.0.0.1:$(DB_PORT)
INGEST_ARGS ?=
INDEX_ARGS ?=
SMOKE_ARGS ?=
STANCE_ARGS ?=
TRAIN_ARGS ?=
COMPARE_ARGS ?=
EVAL_ARGS ?=
MAX_USD ?=
ENV_FILE ?=
# The benchmark's two retriever configs: the Phase 3 baseline first, then the frozen candidate.
EVAL_CONFIGS := phase3 tuned
EVAL := cd $(BACKEND) && .venv/bin/python -m app.evaluation

.PHONY: help setup dev dev-backend dev-frontend test test-frontend test-db lint typecheck build gen-api \
	contract-check check clean ingest db-load db-up db-down index ask smoke \
	stance-fetch stance-train stance-benchmark stance compare-precompute compare-evaluate \
	rag-eval rag-eval-report ask-examples space-bundle vercel-check

help: ## List the available targets
	@echo "FedRAG - make targets:"
	@awk 'BEGIN { FS = ":.*## " } /^[a-z-]+:.*## / { printf "  %-19s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

setup: ## Create backend/.venv, install the backend with dev tools, install the frontend
	@$(PYTHON) -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null || { echo "make setup: Python 3.11+ is required, but $(PYTHON) is $$($(PYTHON) --version 2>&1). Try: make setup PYTHON=python3.12"; exit 1; }
	@if [ -x $(VENV_BIN)/python ] && ! { $(VENV_BIN)/python -c 'import sys; sys.exit(sys.version_info < (3, 11))' && $(VENV_BIN)/python -m pip --version; } >/dev/null 2>&1; then echo "Removing stale $(BACKEND)/.venv"; rm -rf $(BACKEND)/.venv; fi
	test -x $(VENV_BIN)/python || $(PYTHON) -m venv $(BACKEND)/.venv
	cd $(BACKEND) && .venv/bin/python -m pip install -e ".[dev]" -c constraints.txt
	cd $(FRONTEND) && npm install

dev-backend: ## Run the FastAPI backend with auto-reload on http://127.0.0.1:8000
	cd $(BACKEND) && .venv/bin/uvicorn --factory app.main:create_app --reload --reload-dir app --host 127.0.0.1 --port 8000

dev-frontend: ## Run the Next.js dev server on http://127.0.0.1:3000 (proxies /api to the backend)
	cd $(FRONTEND) && npm run dev

dev: ## Run backend and frontend together (Ctrl-C stops both)
	@for port in 8000 3000; do \
		if (exec 3<>/dev/tcp/127.0.0.1/$$port) 2>/dev/null; then \
			echo "make dev: port $$port is already in use; stop that process first (lsof -nP -iTCP:$$port -sTCP:LISTEN)."; exit 1; \
		fi; \
	done
	@$(MAKE) --no-print-directory -j2 dev-backend dev-frontend

test: ## Run the backend tests (pytest; Postgres tests only if TEST_DATABASE_URL is set) and frontend tests (vitest)
	cd $(BACKEND) && .venv/bin/pytest
	cd $(FRONTEND) && npm test

test-frontend: ## Run only the frontend unit tests (vitest: SSE and stream client, answers and citations, recorded answers, stance, comparison, evaluation report, backend health and wake-up, overview, URL state)
	cd $(FRONTEND) && npm test

test-db: ## Run only the Postgres tests (pytest -m db) against TEST_DATABASE_URL
	@if [ -z "$${TEST_DATABASE_URL// /}" ]; then \
		echo "make test-db: TEST_DATABASE_URL is not set. It must name a throwaway database whose name"; \
		echo "contains 'test' (the tests drop schema fedrag). For a local one: make db-up, then"; \
		echo "  TEST_DATABASE_URL=$(LOCAL_DB_URL)/fedrag_test make test-db"; \
		exit 1; \
	fi
	cd $(BACKEND) && .venv/bin/pytest -m db

lint: ## Ruff lint + format check on the backend and api/ (the Vercel entrypoint), ESLint on the frontend
	cd $(BACKEND) && .venv/bin/ruff check . ../api && .venv/bin/ruff format --check . ../api
	cd $(FRONTEND) && npm run lint

typecheck: ## Type-check the frontend (tsc --noEmit)
	cd $(FRONTEND) && npm run typecheck

build: ## Production build of the frontend (next build, as CI runs it)
	cd $(FRONTEND) && npm run build

gen-api: ## Regenerate frontend/openapi.json and src/lib/api/schema.d.ts from the backend
	$(VENV_BIN)/python $(BACKEND)/scripts/export_openapi.py $(FRONTEND)/openapi.json
	cd $(FRONTEND) && npm run gen:api

contract-check: gen-api ## Fail if the regenerated API files differ from git (skipped before the first commit)
	@if ! git rev-parse --verify --quiet HEAD >/dev/null 2>&1; then \
		echo "contract-check: skipped, the repo has no commits to compare against yet."; \
		echo "                The API files were regenerated; commit them with the backend schemas."; \
	elif ! git diff --quiet -- $(API_FILES) || [ -n "$$(git ls-files --others --exclude-standard -- $(API_FILES))" ]; then \
		echo "contract-check: the generated API files changed:"; \
		git status --short -- $(API_FILES); \
		echo "Review them (git diff), then stage or commit them together with the backend schema change."; \
		exit 1; \
	else \
		echo "contract-check: generated API files match the backend contract."; \
	fi

ingest: ## Download new FOMC pages and parse them into data/ (no DB load; INGEST_ARGS=--refresh re-downloads every page)
	cd $(BACKEND) && .venv/bin/python -m app.ingest all --no-load $(INGEST_ARGS)

db-load: ## Upsert data/processed into Postgres schema fedrag (DATABASE_URL from env or backend/.env)
	cd $(BACKEND) && .venv/bin/python -m app.ingest load

index: ## Chunk the documents and embed new/changed chunks into data/chroma (INDEX_ARGS=--dry-run|--rebuild|--source processed)
	cd $(BACKEND) && .venv/bin/python -m app.rag index $(INDEX_ARGS)

# Q goes to the CLI literally: make would expand "$60" in a command-line Q to "0".
ifdef Q
override Q := $(value Q)
export Q
endif
ask: ## Answer a question with citations (calls OpenAI): make ask Q='...' [AS_OF=YYYY-MM-DD] [METHOD=hybrid_rerank] [PRESET=phase3]
	@if [ -z "$${Q// /}" ]; then \
		echo "make ask: set Q, e.g. make ask Q='Why did the FOMC cut rates in September 2024?' AS_OF=2024-10-31"; exit 1; \
	fi
	cd $(BACKEND) && .venv/bin/python -m app.rag ask "$$Q" $${AS_OF:+--as-of "$$AS_OF"} $${METHOD:+--method "$$METHOD"} $${PRESET:+--preset "$$PRESET"}

smoke: ## Smoke check (n=12): hit@5/MRR@10 of the 4 retrieval methods under RETRIEVER_PRESET (SMOKE_ARGS='--preset phase3', --answers)
	cd $(BACKEND) && .venv/bin/python -m app.rag smoke $(SMOKE_ARGS)

stance-fetch: ## Download the TDW sentence splits (CC BY-NC 4.0, git-ignored) and the FRED fed funds range
	cd $(BACKEND) && .venv/bin/python -m app.stance fetch

stance-train: ## Fine-tune FinBERT on each TDW seed, about 3 min per seed on MPS (TRAIN_ARGS=--retrain redoes saved ones)
	cd $(BACKEND) && .venv/bin/python -m app.stance train $(TRAIN_ARGS)

stance-benchmark: ## TDW sentence benchmark of the 4 classifiers (LLM calls cached; refuses an estimate over 1 USD; STANCE_ARGS='--max-usd 2' raises it)
	cd $(BACKEND) && .venv/bin/python -m app.stance benchmark $(STANCE_ARGS)

stance: ## Classify the statements with the 4 classifiers (LLM calls cached), score them vs data/eval/stance_gold.jsonl, load Postgres
	cd $(BACKEND) && .venv/bin/python -m app.stance predict $(STANCE_ARGS)
	cd $(BACKEND) && .venv/bin/python -m app.stance evaluate
	cd $(BACKEND) && .venv/bin/python -m app.stance load

compare-precompute: ## Analyse every consecutive statement pair (one OpenAI call per pair not stored yet) into Postgres; refuses an estimate over 1 USD (COMPARE_ARGS='--dry-run' estimates only)
	cd $(BACKEND) && .venv/bin/python -m app.compare precompute $(COMPARE_ARGS)

compare-evaluate: ## Score the stored analyses against the gold-implied stance shifts (no LLM calls) -> data/eval/compare_benchmark.json
	cd $(BACKEND) && .venv/bin/python -m app.compare evaluate

rag-eval: ## RAG benchmark (OpenAI, cached): validate, retrieval, TEST answers + judge, report; needs MAX_USD (each OpenAI step refuses to start above it), EVAL_ARGS for retrieval, answers and judge
	@if [ -z "$(strip $(MAX_USD))" ]; then \
		echo "make rag-eval: set MAX_USD: each OpenAI step (answers, then judge) refuses to start when its estimate exceeds it, e.g."; \
		echo "  make rag-eval MAX_USD=0.5 EVAL_ARGS='--ids sf-05 pp-01'   (cached results cost nothing)"; \
		echo "  make rag-eval MAX_USD=6.5   (a fresh run: the judge step is estimated at about \$$6.40, as the"; \
		echo "                               estimate assumes 1,500 output tokens a call; the Phase 7 run cost about \$$3.50)"; \
		exit 1; \
	fi
	$(EVAL) validate
	$(EVAL) retrieval --split all --config $(EVAL_CONFIGS) $(EVAL_ARGS)
	$(EVAL) answers --split test --config $(EVAL_CONFIGS) --methods hybrid_rerank --max-usd $(MAX_USD) $(EVAL_ARGS)
	$(EVAL) judge --split test --config $(EVAL_CONFIGS) --methods hybrid_rerank --max-usd $(MAX_USD) $(EVAL_ARGS)
	@$(MAKE) --no-print-directory rag-eval-report

rag-eval-report: ## Rebuild data/eval/rag_benchmark.json from data/eval_cache (offline, no API calls): all splits, phase3 vs tuned, the switch rule, judge validation
	$(EVAL) report --split all --config $(EVAL_CONFIGS) --switch-rule

ask-examples: ## Export the recorded example answers GET /api/ask/examples serves to data/eval/ask_examples.json (from data/eval_cache; no API calls)
	$(EVAL) export-ask-examples

vercel-check: ## Run the demo API's Vercel function locally: a clean venv from requirements.txt, the files git would push, every demo call (ENV_FILE=~/.fedrag-space.env adds the database)
	scripts/check_vercel.sh $(ENV_FILE)

space-bundle: ## Assemble the Hugging Face Space folder of the API (public demo, the alternative to Vercel) in deploy/build/hf-space and check it for secrets (deploy/README.md)
	scripts/build_hf_space.sh

db-up: ## Start a local Postgres 17 in Docker on 127.0.0.1:55432 (databases fedrag, fedrag_test)
	@docker info >/dev/null 2>&1 || { echo "make db-up: Docker is not running. Start Docker Desktop (open -a Docker) and retry."; exit 1; }
	@if docker container inspect $(DB_CONTAINER) >/dev/null 2>&1; then \
		docker start $(DB_CONTAINER) >/dev/null; \
	else \
		docker run -d --name $(DB_CONTAINER) -e POSTGRES_USER=fedrag -e POSTGRES_PASSWORD=fedrag \
			-e POSTGRES_DB=fedrag -p 127.0.0.1:$(DB_PORT):5432 $(DB_IMAGE) >/dev/null; \
	fi
	@for i in $$(seq 60); do \
		docker exec $(DB_CONTAINER) pg_isready -q -h 127.0.0.1 -U fedrag -d fedrag && break; sleep 1; \
	done; \
	docker exec $(DB_CONTAINER) pg_isready -q -h 127.0.0.1 -U fedrag -d fedrag || \
		{ echo "make db-up: Postgres did not become ready; see: docker logs $(DB_CONTAINER)"; exit 1; }
	@docker exec $(DB_CONTAINER) psql -U fedrag -d fedrag -tAc \
		"SELECT 1 FROM pg_database WHERE datname = 'fedrag_test'" | grep -q 1 || \
		docker exec $(DB_CONTAINER) createdb -U fedrag fedrag_test
	@echo "Postgres is up in container $(DB_CONTAINER) on 127.0.0.1:$(DB_PORT):"
	@echo "  DATABASE_URL=$(LOCAL_DB_URL)/fedrag             for make db-load and the API"
	@echo "  TEST_DATABASE_URL=$(LOCAL_DB_URL)/fedrag_test   for make test-db"

db-down: ## Remove the local Postgres container and its data
	@if ! docker info >/dev/null 2>&1; then \
		echo "make db-down: Docker is not running, so $(DB_CONTAINER) could not be checked or removed. Start Docker Desktop (open -a Docker) and retry."; exit 1; \
	elif docker container inspect $(DB_CONTAINER) >/dev/null 2>&1; then \
		docker rm -f -v $(DB_CONTAINER) >/dev/null && echo "Removed container $(DB_CONTAINER)."; \
	else \
		echo "make db-down: no $(DB_CONTAINER) container to remove."; \
	fi

check: gen-api lint typecheck test build contract-check ## What CI checks except the Postgres tests: lint, typecheck, backend and frontend tests, build, API drift

clean: ## Remove caches and build output (keeps backend/.venv and node_modules)
	find $(BACKEND) -path $(BACKEND)/.venv -prune -o -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf $(BACKEND)/.pytest_cache $(BACKEND)/.ruff_cache .pytest_cache .ruff_cache
	rm -rf $(FRONTEND)/.next $(FRONTEND)/tsconfig.tsbuildinfo
