# data/

Shared data for the backend. `make ingest` (`python -m app.ingest all --no-load`, run in backend/)
fills `raw/` and `processed/` from federalreserve.gov. Both are git-ignored and can be rebuilt any
time.

**`raw/`** holds the Federal Reserve's HTML pages exactly as downloaded, so every processed document
can be traced back to its source. Files are never edited after download.

- `calendar/`: `fomccalendars.htm` (2021 on) and `fomchistoricalYYYY.htm` (2015-2020)
- `statements/`: the FOMC statement press releases, `monetaryYYYYMMDDa.htm`
- `minutes/`: the minutes, `fomcminutesYYYYMMDD.htm`

Pages already in `raw/` are reused. Each run re-downloads the calendar page, so documents the Fed
has released since the last run are picked up. `--refresh` (or `make ingest INGEST_ARGS=--refresh`)
downloads every page again (about 8 minutes), e.g. to pick up a correction to a page already
cached. For 2015 to October 2026 this is 198 pages, about 22 MB.

**`processed/`** holds what the parse step builds from `raw/`. Rerunning it on the same pages writes
the same meetings and documents files, byte for byte (`report.json` also records the run date).

- `meetings.jsonl`: one line per meeting, sorted by date. `meeting_date` (the last day of the
  meeting), `start_date`, `unscheduled`, `has_sep`, `has_press_conference`, `statement_url`,
  `minutes_url`, `minutes_release_date` and `source_url` (the calendar page it came from).
- `documents.jsonl`: one line per statement or set of minutes. `doc_id` (e.g.
  `statement-2024-09-18`), `doc_type`, `meeting_date`, `release_date`, `release_date_source`,
  `title`, `url`, `paragraphs` (`section` and `text` for each, in reading order; `section` is the
  minutes heading, null for statements), `text` (the paragraphs joined by blank lines),
  `word_count`, `sha256` (of `text`) and `parser_version`.
- `report.json`: the quality report. It holds counts per year, meetings without minutes and why,
  statements not released on the meeting day, minutes release lag, word counts, skipped calendar
  entries, and 12 pass/fail checks.

`release_date` is the date a document became public, read from the Fed (the press release date for
statements, the "(Released ...)" note on the calendar for minutes) and never estimated. Minutes come
out about three weeks after the meeting, so point-in-time retrieval filters on `release_date`, not
`meeting_date`.

`make db-load` loads `meetings.jsonl` and `documents.jsonl` into the Postgres schema `fedrag` (see
`backend/app/db/schema.sql`). It validates every row first and upserts, so a load can be repeated.

**`chroma/`** holds the local Chroma vector store (collection `fedrag_chunks`, cosine distance) that
`make index` (`python -m app.rag index`) fills with OpenAI embeddings of the chunks, about 78 MB for
3,077 chunks. The backend reads its location from `CHROMA_DIR`. Each vector's metadata carries the
chunk's `doc_id`, `doc_type`, dates (`release_ord` is the release date as YYYYMMDD, which point-in-time
filters compare against), section and a content hash, so a rebuild embeds only new or changed
chunks. Git-ignored, and not created until the first index build.

**`index/`** holds the rest of the search index, written by the same build (`INDEX_DIR`):

- `chunks.jsonl`: one line per chunk, in release order: `chunk_id` (`{doc_id}#{index:04d}`),
  `doc_id`, `doc_type`, `meeting_date`, `release_date`, `page_updated`, `title`, `url`, `section`,
  `index`, `text` (the passage) and `context` (the header embedded and searched with it). BM25 is
  built in memory from this file.
- `manifest.json`: embedding model, chunker version, chunk and document counts, hashes of the corpus
  and of `chunks.jsonl`, the source (`db` or `processed`) and `built_at`. It is written last, so its
  presence means the build completed, and `/api/health` reports `vector_store_ready: true` only when
  it matches the Chroma collection's count.

Both files are tracked in git (everything else in `index/` is ignored): the public demo's API, a
Vercel function built from the repository (see `docs/deploy.md`), serves the RAG Performance
drilldown's passage text from `chunks.jsonl` without a vector store, and the Hugging Face Space
bundle (`make space-bundle`) copies both. `make index` rewrites them; `make index
INDEX_ARGS=--rebuild` re-embeds everything (needed after changing `EMBEDDING_MODEL`); a build that
changes nothing leaves both files as they are. Commit them when the chunks change. `chroma/` is
never tracked or deployed.

**`eval/`** is tracked in git, because the questions and results are part of the project's claims.
`smoke.jsonl` holds the 12 smoke-check questions (`id`, `question`, `as_of`, `expected_doc_ids`
verified by reading the documents, and `expect`, what a correct answer says) that `make smoke` runs.
It is a sanity check, not a benchmark. For the RAG evaluation (Phase 7; guide in `docs/rag-eval.md`):

- `rag_questions.jsonl`: the 85 evaluation questions, one per line, sorted by `id` (30 `dev`, 55
  `test`). Each has `question`, `as_of`, `type`, `split`, `answerable`, `human_reviewed`,
  `relevant_chunks` (chunk ids of the search index, each `primary` or `supporting`, with the indices
  of the required facts it states), `relevant_doc_ids`, `reference_answer`, `required_facts` (the
  only facts scored), `optional_facts` and `notes` (provenance, derivations, `MUST NOT:` facts,
  dated fixes). `python -m app.evaluation validate` checks it against the index.
- `rag_review_items.json`: the 20 TEST items the project owner reviewed, with the reviewer guide and
  each item's passages.
- `rag_benchmark.json`: written by `make rag-eval-report` (`python -m app.evaluation report`) from
  the caches below, and served by `GET /api/evaluation`. It holds the retrieval and answer metrics
  per split, retriever config and method, with bootstrap intervals, per-type rows and paired
  differences. It also has the configs, the generator and judge, the judge validation, the
  selection made by the pre-registered switch rule, the OpenAI cost and notes. Each question is
  recorded with its labels and, per run, its top 10 chunk ids, scores, answer text and the judge's
  verdicts. It holds no secret and no passage text (the API reads passages from the index).
- `rag_judge_sample.json` and `rag_judge_sample_key.json`: the blind sample of 40 judged answers for
  annotators (`python -m app.evaluation export-judge-sample`), and the judge's verdicts on them, kept
  in a separate file so annotators do not see them.
- `ask_examples.json`: the eight recorded answers that Ask the Fed shows on the public demo, served
  by `GET /api/ask/examples` in every mode. They are real answers of the production setup (`tuned`
  hybrid + rerank, top 5, the production answer prompt and model) to TEST questions, picked (not
  at random) to cover different kinds of question: a statement fact, a minutes quantifier, a
  point-in-time question, a question spanning two meetings, a paraphrase, a vote and its terms, a
  staff outlook and a correct decline. `make ask-examples`
  (`python -m app.evaluation export-ask-examples`) copies them from `eval_cache/answers.jsonl` with
  no API call. Each example is an `AskResponse` (answer, citations resolved from the search index,
  notes, model, retrieval) plus its question id, type, `as_of`, a label and a one-line reason for
  the pick, `recorded_at`, and the judge's verdicts from `rag_benchmark.json`.
- `rag_judge_annotations.json`: the annotators' labels for that sample. It has one `annotators`
  entry per annotator (`id`, `annotator`, `notes`, and per item the `human` label of each claim,
  citation, fact and response type), plus a note on how independent they are. The report scores
  the judge against each annotator and the annotators against each other.

For Policy Stance (Phase 5):

- `stance_gold.jsonl`: the reference label of each of the 97 statements (`label`, `confidence`,
  both annotators' labels, `evidence`, and `human_label` / `human_note` for the 25 re-labelled
  blind by a human), made under `docs/stance-labeling.md`. The effective label is `human_label`
  where set, else `label`. `stance_review_items.json` holds the items of that blind review.
- `stance_benchmark.json`: written by `python -m app.stance benchmark` (sentence level) and
  `evaluate` (statement level, with bootstrap intervals, paired differences and the post-hoc
  rows). Metrics per seed and per class, confusion matrices, each classifier's configuration and
  the OpenAI cost; never a Trillion Dollar Words sentence.

For Statement Comparison (Phase 6):

- `compare_benchmark.json`: written by `python -m app.compare evaluate` (`make compare-evaluate`,
  no LLM calls) from the analyses stored in Postgres (`fedrag.statement_comparisons`) for the 96
  pairs of consecutive statements. For the current prompt (`analyses`, with `prompt` saying whether
  it is pre-registered or post hoc) and, under `earlier_prompts`, for each earlier prompt still
  stored: agreement of `stance_shift` with the shift implied by the effective labels in
  `stance_gold.jsonl` and with the `llm_zero_shot` score delta (accuracy, macro-F1 and kappa with
  bootstrap and year-block intervals, on all pairs and on the pairs whose reference changed or not),
  the always-unchanged and score-delta baselines, structural checks (cited ids, key changes,
  verbatim quotes), the OpenAI cost and one row per pair. Labels, doc ids and numbers only, never
  statement or analysis text.

**`eval_cache/`** (git-ignored) holds the RAG evaluation's caches, so that a rerun, or `make
rag-eval-report`, makes no API call for anything already done:

- `retrieval.jsonl`: each question's top 10 chunks and latency per method and retriever config, keyed
  by question, `as_of`, method and config hash;
- `query_embeddings.jsonl`: the question vectors (`make smoke SMOKE_ARGS=--cached-embeddings` reuses
  them too);
- `answers.jsonl`: the generated answers, with passages, citations, usage and latency, keyed by
  question, method, config hash, top_k, prompt version and model;
- `judgments.jsonl`: the judge's verdicts, keyed by judge model, effort, prompt and rendered input.

A report built from another question file (the smoke set, a scratch file) is written here too,
never over the tracked benchmark. None of it is deployed: the public demo needs only the tracked
`eval/ask_examples.json` extracted from `answers.jsonl`.

**`external/`** holds downloaded third-party data: `tdw/`, the Trillion Dollar Words train/test
splits (CC BY-NC 4.0, pinned to one commit and checked by SHA-256), fetched by `make stance-fetch`.
**`models/`** holds the FinBERT models fine-tuned on each TDW seed (`finbert-tdw-<seed>/`, about
420 MB each, with `training.json`), written by `make stance-train`; they derive from FinBERT (CC
BY-NC-SA 4.0) and are not redistributed. **`stance/`** holds `fed_funds_target.csv` (the FRED
target-range change points), `predictions.jsonl` (each classifier's call on each statement, loaded
into Postgres by `make stance`), `predictions_post_hoc.jsonl` (the post-hoc rubric-ordered calls of
keyword and FinBERT, scored but never loaded) and `cache/`, every LLM prediction keyed by method,
model, prompt fingerprint, example set and text, so a rerun makes no OpenAI calls. All are
git-ignored.
