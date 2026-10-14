# FedRAG frontend

Next.js 16 (App Router, TypeScript, Tailwind CSS v4) web app for FedRAG. It has an overview page
and four tabs:

| Route          | Page                 | Status                |
| -------------- | -------------------- | --------------------- |
| `/`            | Overview             | Live (Phase 8)        |
| `/ask`         | Ask the Fed          | Live (Phase 4)        |
| `/stance`      | Policy Stance        | Live (Phase 5)        |
| `/compare`     | Statement Comparison | Live (Phase 6)        |
| `/performance` | RAG Performance      | Live (Phase 7)        |

The overview page introduces the project and gives each tab a plain row with one headline number,
each read from the API when the page loads (`GET /api/evaluation`, `GET /api/stance/benchmark`,
and a stored analysis from `POST /api/compare` with `analysis: "cached_only"`), never typed into
the page. The header links to every page: Overview, Ask the Fed, Policy Stance, Compare and
Performance (on a phone the short labels Ask, Stance, Compare and Performance, without Overview:
the FedRAG wordmark leads home). Its status (`BackendStatus`) reads `GET /api/health` and shows a
dot and a word: Connecting, Online, Setup needed (the OpenAI key is not set or the search index is
not built; its tooltip and screen-reader text say which), Demo on the public demo (below), Waking
or Offline.

Every page keeps its methodology and caveats in one closed "About" block at its end
(`components/ui/About.tsx`), and what the public demo switches off is said in a quiet line of
text, never in a banner or a box.

## Public demo mode

`GET /api/health` reports `demo_mode`, true when the backend runs with `PUBLIC_DEMO=true` (the
hosted demo, which makes no OpenAI calls). One store fetches health once per page load
(`src/lib/api/backend.ts`, read through `useBackend()` and `useDemoMode()` in
`src/components/useBackend.ts`), and the pages adapt to it:

| Page | On the demo |
| ---- | ----------- |
| `/` | One line under the pitch: no paid API calls, so Ask the Fed shows recorded answers (`DemoNote`) |
| `/ask` | No question form: one line saying live questions are off, and the eight recorded answers from `GET /api/ask/examples`, listed beside the selected one (a dropdown below 1024 px) and shown with the same answer and sources as a live answer, plus their recorded date, model, search settings and judge verdict (`RecordedAnswers`). `?example=<id>` opens one; a shared `?q=` link gets a note that only a local copy can answer it. Until health answers, `/ask` shows a skeleton, so the live form never flashes (`AskGate`) |
| `/compare` | No **Generate analysis** button and no "Ask about this change" link. A pair without a stored analysis shows the diff and offers the consecutive pair that ends at the same statement |
| `/stance`, `/performance` | Complete; the "Ask about this meeting" and "Ask it in Ask the Fed" links are hidden, since the demo takes no new questions |

**A sleeping backend.** The demo's API runs on free hosting (a Hugging Face Space) that sleeps
after two days without visitors. When `BACKEND_MAY_SLEEP` is true (the default for any
`BACKEND_URL` that is not on this machine), a backend that does not answer within 2.5 s is shown as
waking: the header status reads "Waking", a line above the page says so (`WakeNotice`), health
is polled with backoff (2, 4, 8, then every 10 s), and requests that failed meanwhile are sent
again once it answers (`apiFetchAwake`). After 150 s the pages report the demo server as not
responding. A local backend that does not answer is reported offline at once, with how to start
it. `src/lib/api/wake.ts` decides which failures mean "not up yet": no response, a timeout, or a
5xx without FastAPI's JSON `detail`.

## Ask the Fed

`/ask` streams answers from `POST /api/ask/stream` (server-sent events):

- **Form**: question (up to 2,000 characters; Cmd/Ctrl+Enter submits), optional as-of date
  (2015-01-28 to today), and a closed **Search settings** section (its summary shows the current
  choice) with the retrieval method and the number of passages (3-10). Six example questions,
  plain links under the form, only fill it.
- **Shareable URL**: the form is mirrored in the query string (`q`, `as_of`, `method`, `k`;
  defaults left out). Opening such a link fills the form but never submits it, since every answer
  spends OpenAI credits.
- **Streaming**: a stage indicator ("Searching documents…", "Writing answer…"), the answer text as
  it arrives, a Stop button (aborting the request also stops generation on the backend), and the
  numbered **Sources** list under the answer, which grows as each passage is first cited
  (`CitationCard`, one unboxed entry per source). The final `done` event replaces the streamed
  text and adds the metadata line and any notes.
- **Citations**: `[n]` markers are superscript buttons; hovering or focusing one previews the
  passage, activating it scrolls to and highlights source n. Each source links to
  federalreserve.gov ("Open on federalreserve.gov") and flags pages revised after release.
- **Errors**: validation errors are shown inline; 503s explain the fix (API key, `make index`,
  reranker with a "Retry with Hybrid" button); 502s, cut-off streams and network failures get a
  notice (a colored rule at the left, no box) with Retry.
- **No supported answer**: when the answer cites nothing, a note says so and suggests what to
  change (wording, the as-of date, more passages, the 2015 start of the documents).
- **About these answers**: the method and caveats, closed at the end of the page (`AskAbout`).

The SSE parser is hand-written (`src/lib/sse.ts`) because `EventSource` cannot send a POST body.
`src/lib/api/stream.ts` posts the request with `fetch`, reports errors decided before the stream
starts (422, 503) as `ApiError` like `apiFetch`, and passes each typed event to the
`useAskStream` hook, whose reducer (`src/lib/ask/answerState.ts`) holds the answer state.

### Streaming through the `/api` proxy

The stream goes through the same `/api` rewrite as every other call. Next.js gzips compressible
responses, `text/event-stream` included, and a gzip stream reaches the browser only when it ends,
so an answer would appear all at once. The backend sends event streams with
`Cache-Control: no-cache, no-transform`, which Next.js honours by not compressing them. This was
verified with a browser-like `Accept-Encoding` against the real backend under both `next dev` and
`next build` + `next start`: events arrive one by one (first answer text about 2 s after the
request, with `status` events before it). Keep `no-transform` on the backend if the stream ever
moves, and check with
`curl -N --compressed -X POST http://127.0.0.1:3000/api/ask/stream -H 'Content-Type: application/json' -d '{"question": "..."}'`.

## Policy Stance

`/stance` shows how four classifiers (LLM few-shot, LLM zero-shot, FinBERT, keyword) label every
FOMC statement since 2015, from `GET /api/stance?classifier=` (one request per classifier, full
range, fetched once in parallel) and `GET /api/stance/benchmark`:

- **Timeline**: three panels on one time axis, drawn with Recharts: each chosen classifier's score
  (-1 most dovish to +1 most hawkish), the reference label of each statement (filled dot
  human-reviewed, ring LLM-adjudicated) and the federal funds target range as a step line. They
  are separate panels rather than a second y axis, since the score and the rate have unrelated
  scales. One crosshair spans the panels; the tooltip lists every plotted score, the reference
  label and the range the meeting set. The **Show methods** buttons above the chart, each with its
  line key (`SeriesKey`), are the legend and toggles (LLM few-shot by default); a From/To year
  filter scopes the chart, table and meeting list.
- **Colors**: one fixed categorical slot per classifier (`--series-1..4` in `globals.css`, with
  dark-mode steps), checked for color-vision-deficiency separation against the chart surface (the
  page background) in both modes. Lines carry direct end labels when they do not collide. Chart chrome is slate.
- **Accessibility**: the charts are hidden from screen readers; a data table with the same values
  is always in the accessibility tree and can be shown on screen ("Show data table"). Keyboard
  users pick meetings with the meeting select and Previous/Next buttons.
- **Selected statement** (`MeetingPanel`): clicking the chart (or the date select) shows every
  classifier's stance and score, the chosen classifier's confidence, key factors and verbatim
  evidence sentences, the reference label with its source (LLM-adjudicated or human-reviewed), the
  target range before and after, a link to the statement on federalreserve.gov, a link to
  `/compare` against the previous statement and a link to `/ask` with a question about the meeting
  filled in (never submitted). A stance is a colored word with a direction mark (`StanceLabel`:
  ▲ hawkish, – neutral, ▼ dovish), not a chip.
- **Benchmark** (**How the methods compare**): sentence-level macro-F1 on Trillion Dollar Words
  (official and deduplicated test splits, mean ± sd over three seeds) and statement-level
  accuracy, macro-F1 and Cohen's kappa against the reference labels. The notes that scores and
  confidences are not calibrated, on the reference labels and the human review, and the paired
  differences are in the closed **About these numbers** block (`StanceAbout`).
- **URL state**: `c` (classifiers), `m` (meeting), `from` and `to` (years); defaults left out.
- **Errors**: a classifier without stored results shows as "not loaded" with the `make stance`
  hint while the others still plot; 503s and an unreachable backend explain the fix with Retry.

## Statement Comparison

`/compare` compares two FOMC statements from `POST /api/compare`, with the statement list from
`GET /api/meetings?doc_type=statement`:

- **Pickers**: base (earlier) and target (later) statement, newest first and grouped by year. Dates
  that would put the base on or after the target are disabled in each list, with a screen-reader
  note saying why, so a choice never inverts the pair. Shortcuts: "Use the meeting just before"
  and Earlier pair / Later pair (both sides move one statement, keeping the gap). The default is the latest
  statement against the one before it.
- **URL state**: `base` and `target` (YYYY-MM-DD). Opening a link loads the diff and only an
  analysis that is already stored (`analysis: "cached_only"`), so loading the page never spends
  OpenAI credits. A link with only `target` (or only `base`) fills the other side with the
  neighbouring statement; an inverted pair is explained, with a button to compare it the right
  way round, rather than swapped silently. `#change-N` opens the page at that change.
- **Diff**: split view from 768px (base left, target right, column headings that stay on screen),
  a unified view below it. Each change row has an anchor (`#change-N`, N = the change id), a kind
  label, and for modified pairs the similarity and word-level marks in the stance review page's
  redline colors (deleted words red and struck through, inserted words on a green wash; tokens
  `--ins-*` / `--del-*` in `globals.css`). Added sentences appear only on the right, removed only
  on the left; procedural sentences (the vote) are smaller and grey. Runs of two or more unchanged
  sentences fold into "N unchanged sentences" expanders, with a checkbox to show them all. Screen
  readers hear "deleted:" and "inserted:" before marked words. If a modified pair's segments do
  not rebuild both sentences exactly, the two sentences are shown without word marks.
- **Analysis** (**What the change signals**, above the diff): the stance shift as a colored word
  with its direction mark (More hawkish / Unchanged / More dovish; `ShiftLabel`), the summary and
  the key changes, each with a direction label and links that scroll to and highlight its change
  rows (opening a folded run if needed). The confidence is labelled model-reported, not
  calibrated, and the LLM zero-shot classifier's scores of the two statements are shown as a
  same-model-family cross-check ("+0.10 → −0.90"). The closed **About this comparison** block
  (`CompareAbout`) holds the caveats and names the model and prompt version that wrote the
  analysis (`analysis_model_version`).
- **Analysis states**: not stored yet → "Generate analysis" (explains it makes one OpenAI call;
  the result is stored), a spinner while it runs, the API-key setup hint when the backend has no
  key, and an error with Retry when generation fails. The diff is shown in every state.
- **Errors**: a 404 (no statement for a date) offers the nearest statements as quick picks; 503s
  and an unreachable backend explain the fix with Retry.
- **Links**: each statement to federalreserve.gov, "Ask about this change" to `/ask` with a
  question about the pair and `as_of` set to the target's release date (filled in, never
  submitted), and "Compare with the previous statement" in Policy Stance's selected statement.

## RAG Performance

`/performance` shows the RAG evaluation from `GET /api/evaluation` (one request: every number on
the page, and every question summarised) and, for an opened question,
`GET /api/evaluation/questions/{id}`:

- **Split**: Test (default), Dev or All questions, in one control above everything it scopes. The
  configuration was chosen on dev, so the page says test is the number to quote.
- **Chosen configuration**: the report's `selection` (made on dev) as eight headline numbers on
  the current split, under "Chosen setup" (or "Ask the Fed's default" when a test-split rule made
  it the default): Recall@5, nDCG@10, faithfulness, citation accuracy, correctness, refusal
  accuracy, median latency and cost per question, each with its interval and n and, against Ask
  the Fed's shipped setup (hybrid + rerank under the first config) or the setup the rule replaced,
  the paired difference, colored and called a "clear gain" or "clear drop" only when its interval
  excludes 0 ("within noise" otherwise). Without a selection the page picks the best dev nDCG@10
  and says so.
- **Search** and **Answers**: a dot-and-whisker chart of one metric for every method and config
  (picked with the **Chart** select), and the full table with 95% intervals, grouped by config,
  best per column in bold (lowest for false refusals). Under each table, the report's paired
  differences in the charted metric (between methods within a config, and each config against the
  first), folded when there are more than six.
- **By question type**: a heatmap of one metric, a row per run and a column per question type, on
  one blue ramp in five equal classes (`--heat-*` in `globals.css`; dark mode flips the ramp so
  low values sit near the surface), with the value printed in each cell. Selecting a cell filters
  **Every question** to that type and run.
- **Speed and cost**: retrieval and end-to-end p50/p95, answer and judge cost per question, and
  the evaluation's OpenAI usage per stage.
- **Every question**: the question browser, with filters by type, run and outcome (missed in the
  top 10, unsupported claim, wrong refusal, ...) and text. A question opens a drilldown: its
  reference answer and required facts, the gold passages, every run's top 10 as a strip of rank
  cells (primary gold, supporting gold, right document, other), and one run's retrieved passages
  next to its answer, whose `[n]` markers (Ask the Fed's `AnswerText`) jump to the cited passage,
  with the judge's verdict on every claim, citation and required fact. "Ask it in Ask the Fed"
  fills that form, never sends it (hidden on the demo).
- **About these numbers** (`PerformanceAbout`), closed at the end of the page: how the headline
  setup was chosen, the configs compared (description and settings), how to read each section
  (the LLM judge, the LLM-built question set and its human review, dev/test discipline, small n),
  the judge check (`JudgeCheck`: the judge model and prompt version, and its agreement with
  independent annotators, or that the check is pending) and the report's notes.
- **URL state**: `split`, `type`, `outcome`, `run` (`method:config`) and `q` (the open question).
- **Colors**: each config owns a categorical slot and a mark shape (circle, diamond, square), so
  identity never rests on color alone; the three slots pass the palette validator all-pairs in
  both modes, and every chart has a table with the same values.
- **Errors**: before `make rag-eval` has run, a setup notice names the command; an invalid file
  shows the backend's list of problems.

## Run

```bash
npm install
cp .env.example .env.local   # optional: only needed to change BACKEND_URL
npm run dev                  # http://127.0.0.1:3000
```

Start the FastAPI backend separately (`make dev-backend` from the repo root). The frontend runs
without it, and the header status then reads "Offline". `PUBLIC_DEMO=true make dev` from the repo
root runs both in demo mode.

Environment variables (`.env.example`), all read when `next dev` starts or `next build` runs:

| Variable | Default | What it does |
| -------- | ------- | ------------ |
| `BACKEND_URL` | `http://127.0.0.1:8000` | The FastAPI backend the `/api` rewrite and server code call. On Vercel, the Space's URL |
| `BACKEND_MAY_SLEEP` | `true` unless `BACKEND_URL` is local | Wait for a sleeping backend (above) instead of reporting it offline |
| `SITE_URL` | Vercel's production domain, else `http://localhost:3000` | The origin of absolute link-preview (Open Graph) URLs |

Deployment (Vercel, root directory `frontend`) is in [`docs/deploy.md`](../docs/deploy.md).

## Scripts

| Script              | What it does                                                          |
| ------------------- | --------------------------------------------------------------------- |
| `npm run dev`       | Dev server on port 3000                                               |
| `npm run build`     | Production build (does not need the backend)                          |
| `npm run lint`      | ESLint (`eslint-config-next`)                                         |
| `npm run typecheck` | `tsc --noEmit`                                                        |
| `npm test`          | Vitest unit tests of the pure logic (`src/**/*.test.ts`, Node, no DOM); also run by `make test`, `make check` and CI |
| `npm run gen:api`   | Generate `src/lib/api/schema.d.ts` from `./openapi.json` (backend export) |

## Talking to the backend

- `next.config.ts` rewrites `/api/:path*` to `${BACKEND_URL}/api/:path*`. `BACKEND_URL` defaults to
  `http://127.0.0.1:8000`. Use 127.0.0.1 here, not `localhost`: Node may resolve `localhost` to `::1`
  while uvicorn listens on IPv4 only.
- Browser code calls relative `/api/...` URLs through that proxy, so CORS is not involved.
  Server components call `BACKEND_URL` directly.
- `src/lib/api/client.ts` provides `apiFetch<T>()`, which picks the base URL, sends and expects JSON,
  and throws `ApiError(status, detail)` with FastAPI's `detail` message on any non-2xx response.
- Do not add route handlers under `src/app/api/`, because they would shadow the rewrite.

## API types

The backend owns the API contract. `openapi.json` is exported from the FastAPI app, and
`npm run gen:api` turns it into `src/lib/api/schema.d.ts`. Both files are checked in and generated,
so never edit them by hand. Components import types only from `src/lib/api/types.ts`, which
re-exports named aliases over the generated schema (`AskRequest`, `StanceResult`, `DocType`, ...),
including the stream event payloads (`StreamStatus`, `StreamDelta`, `StreamCitation`,
`StreamError`; `done` carries an `AskResponse`). Only the `AskStreamEvent` union, which pairs each
SSE event name with its payload, is written by hand, because the generated types describe the
stream's data only as a union of the payloads.

To change the contract, edit the Pydantic schemas in `backend/app/schemas/`, then run
`make gen-api` from the repo root. CI regenerates both files and fails if they differ from what is
committed.

`gen:api` passes `--default-non-nullable false`, so request fields that have a server-side default
(such as `top_k` on `AskRequest`) stay optional. Response fields are still required, because the
backend marks every serialized field as required in its schema.

## Layout

```
src/
  app/              layout, overview page (/), one folder per tab, 404, icon, link-preview
                    images (opengraph-image, twitter-image)
  components/       Header, TabNav, BackendStatus, WakeNotice, useBackend, Footer, PageHeader,
                    ShareImage, DataTable (table pieces shared by the benchmark tables)
  components/ui/    the design system's shared pieces: About (the closed methodology block),
                    Section, ExternalLink, styles (class strings for controls and text roles)
  components/landing/ LandingData (the overview's API reads and headline numbers), DemoNote
  components/ask/   AskGate (live form or recorded answers), AskExperience (client root of the
                    live /ask), AskForm, RecordedAnswers, AnswerPanel, AnswerText, CitationCard,
                    AskAbout, AskSkeleton, useAskStream, icons
  components/stance/ StanceExplorer (client root of /stance), StanceChart, StanceFilters,
                    SeriesKey, StanceTable, MeetingPanel, BenchmarkSection, StanceLabel,
                    StanceAbout, FailureNotice, StanceSkeleton, useStanceData
  components/compare/ CompareExperience (client root of /compare), CompareControls,
                    AnalysisPanel, DiffView, SegmentText, ShiftLabel, PairNotices, CompareAbout,
                    CompareSkeleton, useCompareData
  components/performance/ PerformanceExplorer (client root of /performance), Headline,
                    RetrievalSection, AnswerSection, TypeHeatmap, CostSection, MetricTable,
                    IntervalChart, QuestionBrowser, QuestionDrilldown, JudgeCheck,
                    PerformanceAbout, parts, PerformanceSkeleton, usePerformanceData
  lib/api/          client.ts (fetch helper), backend.ts (health store, demo mode, wake-aware
                    fetch), wake.ts and hosting.ts (sleeping-backend rules), stream.ts (POST
                    /api/ask/stream), types.ts (API type aliases), schema.d.ts (generated)
  lib/ask/          pure logic: URL/form state, answer reducer, [n] marker splitting, failures,
                    formatting, recorded answers (each with a *.test.ts), example questions
  lib/landing/      the overview's headline results from the API responses (with a *.test.ts)
  lib/stance/       pure logic: chart series and fed funds steps, URL state, formatting,
                    benchmark tables, failures, classifier metadata (with *.test.ts)
  lib/compare/      pure logic: URL state and pair resolution, diff rows and folding, word
                    segments, failures, formatting (with *.test.ts); fixtures.ts is a contract
                    fixture for the tests only (ESLint blocks importing it from app code)
  lib/performance/  pure logic: report rows per split, best run, headline and baseline runs,
                    paired differences, heatmap grid and bins, question filters, rank strips,
                    URL state, failures, formatting (with *.test.ts); fixtures.ts as above
  lib/sse.ts        server-sent events parser
  lib/site.ts       site name, repository and technical-report links, page metadata
openapi.json        backend's OpenAPI schema (generated by make gen-api)
```
