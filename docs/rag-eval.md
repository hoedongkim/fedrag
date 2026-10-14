# RAG Evaluation Set

## Purpose

This set is the hand-built question set for Phase 7. It scores FedRAG's retrieval (`bm25`, `dense`, `hybrid`,
`hybrid_rerank`, and the fixes Phase 7 tunes) and the grounded answers built on it. It replaces the
12-question smoke check as the basis for any quality claim in the README and the technical report.

- **What it measures.** For each method: whether the passages that answer a question come back, and how high
  they rank; whether anything released after `as_of` comes back; and whether answers state the right facts,
  cite passages that support them, and decline when the corpus cannot answer.
- **What it targets.** The weaknesses the smoke check found ([technical report](technical-report.md), "Smoke check"): wording mismatch (no
  stemming, so "cut" does not match "lower"); no sense of dates or recency; statements being only 97 of 3,077
  chunks; vote paragraphs cut off by the reranker's 512-wordpiece limit; stale verdicts under `as_of`; and
  attribution slips (dropped quantifiers, mixing up staff, participants and the Committee).
- **Honest tuning.** Phase 7 chooses improvements on DEV only. TEST is scored once for each final configuration,
  and that score is reported as it comes out.

## Files and corpus snapshot

| File | Content |
|---|---|
| `data/eval/rag_questions.jsonl` | The question set, one record per line. Each record's `split` field puts it in DEV (~35%, including all 12 smoke questions with new gold) or TEST (~65%) |
| `data/eval/smoke.jsonl` | Unchanged; `make smoke` still runs it |
| `backend/app/schemas/evaluation.py` | The contract the scores are reported in (`QuestionRecord`, `RetrievalRow`, `AnswerRow`) |

The gold is tied to `data/index/chunks.jsonl` at chunker version 2: 3,077 chunks, with `chunks_sha256`
`0f81bdce...c305ca` in `data/index/manifest.json`. Chunk ids are stable. A rebuild can still move text between
chunks, so if the hash changes, the validator below must pass again before any score is reported.

## Independence from the system under test

- The set is built only from `data/index/chunks.jsonl` and `data/processed/documents.jsonl`, using plain
  `json`, `re` and `grep` and careful reading. Nothing in `backend/app/rag` is imported or run.
  `/api/ask`, `make ask` and `make smoke` are not called, and no retrieval ranking is looked at while writing
  questions or gold.
- Building the set makes 0 OpenAI (or other LLM) calls.
- The smoke questions were used to adjust the Phase 3 prompts, so they go into DEV only.
- The gold is frozen before the first scored run. After that, a gold record may change only to fix an error
  that breaks the rules below: a wrong number, a chunk that does not state its fact, or a missed chunk that
  does. Disagreeing with a method's output is not a reason to change it. Each fix is logged with its date in
  `notes` and reported with the results.

## Question types and targets

| Type | What it tests | Target | DEV | TEST | Smoke items (DEV) |
|---|---|---:|---:|---:|---|
| `statement_fact` | Decision facts in one statement: the rate action and target range, balance-sheet actions, vote tallies, dissents by name and what the dissenters preferred | 14 | 5 | 9 | q01, q03, q04, q08 |
| `minutes_fact` | One set of minutes: the staff outlook, participants' views with their quantifier, financial developments, special topics | 18 | 6 | 12 | q07, q10 |
| `point_in_time` | `as_of` is set, and the correct answer depends on what was public on that date | 12 | 4 | 8 | q02, q09 |
| `date_specific` | A named meeting or month on a topic that recurs at many meetings, so only the date identifies the gold | 10 | 4 | 6 | q11 |
| `multi_doc` | Needs 2 or more meetings: comparisons, sequences, first or last, counts | 10 | 4 | 6 | q05, q06 |
| `paraphrase` | The user's wording for the key term differs from the Fed's | 8 | 3 | 5 | none |
| `unanswerable` | The corpus does not contain the answer | 8 | 3 | 5 | q12 |
| extra | Up to 5 more, of any type; `paraphrase`, `date_specific` and `point_in_time` come first | up to 5 | | | |
| **Total** | | **80 + up to 5** | **29** | **51** | **12** |

This is the plan before the extras; the final counts (30 DEV, 55 TEST) are under "Evaluation set". The 12
smoke items keep their question and `as_of` word for word and get new chunk-level gold, so 68 new items
plus up to 5 extras are written.

**`statement_fact`.** The evidence is the statement, which is a single chunk.
- At least 5 items are on votes or dissents, including some whose vote paragraph comes at the end of a long
  2015-2017 or 2021 statement.
- At least 2 items are on balance-sheet decisions.
- A vote tally is a count derived from the text, and `notes` list the names counted.

**`minutes_fact`.** Items are spread across sections: about 5 on the staff economic situation or outlook, 7 on
participants' views, 4 on the financial situation or the Desk's market review, and 2 on special topics such as
framework reviews, balance-sheet planning or organizational matters.
- Quantifiers stay verbatim in `required_facts` ("several participants", "a few", "most").
- If the question does not say whose view it wants, the reference answer still names the holder of each claim.

**`point_in_time`.** Four patterns:
- (a) `as_of` falls between a statement and its minutes, so the answer rests on the statement and on earlier
  minutes.
- (b) A later decision changed the answer, and `as_of` comes before that change. This is the only type allowed
  to use wording like "current", "latest", "yet" or "so far".
- (c) At least 2 boundary items have `as_of` equal to a release date. The cutoff is inclusive, so that document
  counts.
- (d) At most 2 items where nothing released by `as_of` answers the question. These have `answerable: false`
  and still count as `point_in_time`.

`notes` begin with `MUST NOT:` and list the later facts that an answer must not state, as a check on leakage
from the model's own knowledge.

**`date_specific`.** The question names the meeting by month and year ("the June 2022 statement", "the minutes
of the July 2019 meeting"). The topic recurs in nearly the same words at neighboring meetings: the labor-market
paragraph, the wording on inflation, an unchanged range, the runoff pace, or dissents in a period with many
of them.
- At least 3 items target minutes.
- `notes` name the nearest hard negative, a chunk from a neighboring meeting with almost the same text.

**`multi_doc`.** The question compares two meetings, lists a sequence (every cut in 2019), asks for the first
or last move in a cycle, or asks how a phrase changed.
- An item has at most 6 primary chunks.
- A first, last or count claim includes the boundary meetings that prove it (for example, the hold before a
  first increase), and `notes` give the derivation.
- The question may give a period instead of one meeting ("during 2022").

**`paraphrase`.** The key term is in everyday or market wording, and the Fed's own term for it does not appear
in the question:

| User wording | Fed wording |
|---|---|
| cut, hike, pause | lower, raise, maintain the target range |
| QT, taper | reduce holdings (runoff), slow the pace of net asset purchases |
| jobs, hiring | labor market, job gains, payroll employment |
| prices, cost of living | inflation |
| bps, 5.25% | 1/4 percentage point, 5-1/4 percent |
| the Fed | the Committee |
| interest on reserves | IORB, IOER |

The period may be loose if only one answer fits (for example, "the Fed's first cut after the post-pandemic
hikes").

**`unanswerable`.** Eight items:
- 2 from before 2015
- 2 asking for SEP numbers (a median, the dot plot, one participant's projection)
- 1 asking for a press-conference remark
- 1 about another central bank's decision that the minutes do not mention
- 1 about the future: a meeting after September 15-16, 2026, or a forecast of a decision
- 1 free choice

Each item must survive a near-miss check. The minutes do mention SEP submissions in general terms, market
reactions to press conferences, ECB and BOJ actions, medians from the dealer surveys, and pre-2015 history in
passing. The question must ask for something that none of those passages contains, and `notes` list the
near-miss chunk ids that were found.

**Coverage across the answerable items:**
- Every year from 2015 to 2026 is the target period of at least 4 items (at least 3 for 2026).
- At least 40% of items have a minutes chunk as primary, and at least 30% have a statement chunk as primary.
- At least 3 items cover the unscheduled actions (2019-10-04, 2020-03-02, 2020-03-15, 2020-03-23). Their
  record is in the next set of minutes: the "Videoconference meeting of ..." sections of `minutes-2019-10-30`
  and `minutes-2020-03-15`, and the "Notation Votes" section of `minutes-2020-04-29` for March 23.
- No meeting is the target of more than 3 items, and no two items share a reference answer.
- Question forms are mixed (what, how much, who, did, why, compare, list), lengths run from about 8 to 30
  words, and some questions are casual.

## Record format

The files are JSON Lines, one record per line, with these fields:

| Field | Rule |
|---|---|
| `id` | `<prefix>-<NN>`, with prefix `sf`, `mf`, `pit`, `ds`, `md`, `pp` or `ua` (e.g. `mf-07`); never reused |
| `question` | Worded the way a user would ask it |
| `as_of` | `YYYY-MM-DD`, or null for no cutoff (the whole corpus, latest release 2026-10-09) |
| `type` | One of the seven types |
| `split` | `dev` or `test`, assigned as described under Split |
| `answerable` | true or false |
| `human_reviewed` | false until the project owner has reviewed and, where needed, corrected the labels |
| `relevant_chunks` | `[{chunk_id, role, facts}]`, with `role` `primary` or `supporting`; primary chunks first |
| `relevant_doc_ids` | The distinct `doc_id`s of `relevant_chunks`, in the order they first appear |
| `reference_answer` | 1-3 sentences, with numbers, ranges, names and quantifiers verbatim |
| `required_facts` | 1-4 atomic facts, only those the question asks for; `[]` when `answerable` is false |
| `optional_facts` | Facts the question does not ask for (the reference answer may still state them); not scored; `[]` if none |
| `notes` | Where the item came from (e.g. "smoke q08"), derivations, hard negatives, `MUST NOT:` facts, logged fixes |

`facts` adds to the base field list. It holds the 0-based indices of the `required_facts` that a chunk states,
or `[]` for a supporting chunk that only gives context or states only optional facts. The validator uses it to
check numbers and names. A per-fact retrieval diagnostic built on it, which would still work when a fact is
repeated in several chunks, is planned, not implemented.
`RelevantChunk` in the API contract has only `chunk_id` and `role`, and the loader ignores `facts`.

Here is an abridged example (smoke q08). A real record lists every relevant chunk; this one also has
`minutes-2024-09-18#0016`, `#0019` and `#0020`.

```json
{"id": "sf-01", "question": "How much did the FOMC cut rates in September 2024, and did anyone dissent?",
 "as_of": null, "type": "statement_fact", "split": "dev", "answerable": true, "human_reviewed": false,
 "relevant_chunks": [
   {"chunk_id": "statement-2024-09-18#0000", "role": "primary", "facts": [0, 1, 2, 3]},
   {"chunk_id": "minutes-2024-09-18#0021", "role": "primary", "facts": [0, 1]},
   {"chunk_id": "minutes-2024-09-18#0022", "role": "primary", "facts": [2, 3]}],
 "relevant_doc_ids": ["statement-2024-09-18", "minutes-2024-09-18"],
 "reference_answer": "At its September 17-18, 2024 meeting the FOMC lowered the target range for the federal funds rate by 1/2 percentage point, to 4-3/4 to 5 percent. Michelle W. Bowman dissented; she preferred a 1/4 percentage point cut.",
 "required_facts": ["lowered the target range by 1/2 percentage point",
                    "new target range of 4-3/4 to 5 percent",
                    "Michelle W. Bowman dissented",
                    "Bowman preferred a 1/4 percentage point cut"],
 "notes": "smoke q08."}
```

## Gold rules

**Relevance**

1. A chunk is relevant if it contains information the reference answer needs.
   - **Primary:** the chunk states at least one required fact (`facts` is not empty) and belongs to a target
     meeting's own documents. These are its statement and its minutes, including the statement text and
     directive that the minutes repeat under "Committee Policy Action(s)". If the question names a document
     type ("the statement", "the minutes"), only chunks of that type can be primary.
   - **Supporting:** every other relevant chunk. This includes restatements in another meeting's documents
     (for example, the next minutes' market review recapping the decision), chunks of the other document type
     when the question names one, and chunks that give needed context but state no required fact: rationale,
     a definition, or the baseline of a comparison.
2. "Every" relevant chunk means:
   - Read each target document in full in `documents.jsonl`.
   - Search all chunks with regexes for each fact's numbers, names, key terms and their synonyms. The search
     covers the target meetings' documents and the minutes of the next scheduled meeting. Read every hit.
   - A chunk counts if its leading overlap (up to 60 tokens repeated from the previous chunk) contains a whole
     sentence stating a fact. It does not count if the overlap has only part of that sentence.
   - Footnote chunks count when they carry the fact.
3. Order: primary chunks first, starting with the main source (the statement for decision facts, the minutes
   for minutes facts), then supporting chunks; within each group, by document, then by chunk index.
4. With `as_of`, only chunks whose own `release_date` is on or before `as_of` count. Use the chunk's
   `release_date`, not the document's: `minutes-2018-11-08#0031`, a paragraph left out of the original minutes,
   is dated 2026-10-09.
   - Statements are released on `meeting_date`.
   - Minutes come out 19-24 days after the meeting (median 21).
   - The unscheduled actions of 2019-10-04, 2020-03-02 and 2020-03-23 have no minutes of their own.

**Answers**

5. `reference_answer` follows these rules:
   - Numbers, ranges, names and quantifiers are copied exactly ("4-3/4 to 5 percent", "several participants").
   - Every claim names who holds it: the Committee, staff, participants, members or the manager.
   - Dates refer to the meeting, not the month the minutes came out.
   - An `as_of` answer is written as of that date.
   - For `answerable: false`, the answer says what the corpus lacks and why (the SEP and press conferences
     were not collected; the corpus starts in 2015; the meeting has not happened yet), with no facts from
     memory.
6. `required_facts` are atomic, one claim each, with the number or name verbatim. A fact derived from several
   chunks (a vote count, a cumulative change) is allowed only if `notes` show how it was derived. At least one
   primary chunk states each fact. A required fact must answer something the question asks; any other fact
   goes in `optional_facts` (see "Owner review").

**Questions**

7. Write the question the way a user would, without copying the passage. No word 4-gram of the question
   (lowercased, punctuation removed) may appear in a primary chunk unless it also appears in at least 30 chunks
   across the corpus. Common phrases such as "the federal funds rate" pass; the check is scripted. The smoke
   questions are exempt.
8. The period must be unambiguous, except in `multi_doc` and `paraphrase`.
   - Month and year identify a scheduled meeting. In months with more than one action (October 2019, March
     2020), give the full date.
   - Name minutes by their meeting ("the minutes of the June 2022 meeting"), never by the month they were
     released.
   - A question with a null `as_of` must not depend on when it is asked: no "current", "latest" or "yet".
9. An `answerable: false` record has `relevant_chunks`, `relevant_doc_ids` and `required_facts` all `[]`.
   Before marking it, run at least 3 regex sweeps (Fed wording, everyday wording, key numbers or names) and read
   every hit.

## Split

1. Every item that came from the smoke set goes to DEV.
2. For each type, DEV gets 0.35 × the type's final count, rounded half up. This is recomputed after extras are
   added (85 items give 30 DEV and 55 TEST). The open DEV slots are filled by sorting the type's other
   items by `id`, shuffling them with `random.Random(7)` and taking the first ones; the rest go to TEST. The
   result is written to each record's `split` field. The split is made once, after validation, and is not
   changed later to balance difficulty.
3. TEST items are not looked at while tuning, and no failure analysis is done on them until the final run is
   reported.

## Validation

A standalone script that uses no `backend/app/rag` code checks the set before it is frozen:

- Schema and enums are valid. Ids and questions are unique. Per-type and per-split counts match the table.
  Every smoke item is in DEV.
- Every `chunk_id` exists, and `relevant_doc_ids` matches the chunks. An answerable item has at least one
  primary chunk, and primary chunks come before supporting ones.
- Every relevant chunk has `release_date` on or before `as_of`. Every `point_in_time` item has an `as_of`.
  Items with a null `as_of` do not contain "current", "latest", "yet", "so far" or "now".
- Each required fact is in the `facts` of at least one primary chunk. The fact's fractions, percentages,
  decimals, dollar amounts and capitalized names appear verbatim in at least one chunk that states it. Derived
  counts are exempt if `notes` explain them.
- The 4-gram rule (rule 7) holds. Unanswerable records are empty as described in rule 9.

## How Phase 7 scores the set

The metric names below are the fields of `RetrievalRow` and `AnswerRow` in `backend/app/schemas/evaluation.py`.

**Retrieval** uses answerable items only, with `as_of` applied as in `/api/ask`. k = 5 is what the LLM
receives.
- `recall_at_5` and `recall_at_10`: the share of a question's gold chunks (primary and supporting) in the top k.
- `mrr_at_10`: the reciprocal rank of the first gold chunk.
- `ndcg_at_10`: gain 2 for a primary chunk and 1 for a supporting one.
- `doc_recall_at_5` and `doc_recall_at_10`: the share of `relevant_doc_ids` in the top k.
- `as_of_leaks`: must be 0.

Restatements make full chunk recall hard to reach, so doc recall shows whether the top k is enough to answer
(a per-fact diagnostic via `facts` is planned, not implemented).

**Answers** are scored by an LLM judge. The judge is checked against annotators on a blind sample (in Phase 7,
three Claude sessions rather than a person, one of them isolated; see "Judge validation" below):
- `answer_correctness`: the share of required facts the answer states, with exact numbers and names (5.25%
  counts as equal to 5-1/4 percent), and no contradiction of the reference. `optional_facts` are not sent to
  the judge and do not count.
- `faithfulness`: the share of claims supported by the passages given.
- `citation_accuracy`: the share of citations that support their sentence.
- `refusal_accuracy`: unanswerable items, and items with nothing released by `as_of`, are declined without
  invented facts. `false_refusal_rate` counts answerable items that are declined.
- Point-in-time: `MUST NOT` facts are recorded in `notes`, but no metric counts them yet (planned); look-ahead
  is measured only on retrieval (`as_of_leaks`).

Results are reported by type and by split, with bootstrap 95% intervals. With 55 TEST items (50 answerable), one
item moves a rate by about 0.02, and differences inside the intervals are not used to rank methods.

## Evaluation set

The set is in `data/eval/rag_questions.jsonl`: 85 questions,
sorted by `id`, with gold tied to `chunks_sha256` `0f81bdce...c305ca`. 76 are answerable; 9 are not (the 8
`unanswerable` items and one pattern-(d) `point_in_time` item).

**How it was built.** Three writers drafted 87 candidate items independently (33, 27 and 27), and a fourth
agent verified them. All four were Claude Opus 5.5 subagents without access to the system under test. Their
scratch scripts import only the Python standard library and read `data/index/chunks.jsonl`; none imports
`backend/app/rag` or calls `/api/ask`, and no OpenAI calls were made. The verifier checked every item against
its chunks, searched `chunks.jsonl` for missed chunks and checked `as_of`. 14 kept items carry a dated
"Verifier fix" in `notes` (role or fact changes, added chunks, reworded questions or facts, one rebuilt `as_of`
item). 14 candidates were dropped: one duplicate question and 13 items over the per-type targets. The 12 smoke questions were added with their question and `as_of`
unchanged and new chunk-level gold, all in DEV. The verifier's checks (fields, chunk ids, `as_of`, numbers and
names verbatim, the 4-gram rule, empty unanswerable records) pass on the written file. They ran from a scratch
script; the standalone validator described above is not written yet.

| Type | Total | DEV | TEST | Relevant chunks (min / median / max) | Primary (min / median / max) | Lexical overlap |
|---|---:|---:|---:|---|---|---:|
| `statement_fact` | 14 | 5 | 9 | 3 / 6 / 10 | 1 / 4.5 / 6 | 0.49 |
| `minutes_fact` | 18 | 6 | 12 | 1 / 3 / 8 | 1 / 2 / 5 | 0.40 |
| `point_in_time` | 15 | 5 | 10 | 1 / 4 / 12 | 1 / 2.5 / 9 | 0.26 |
| `date_specific` | 10 | 4 | 6 | 1 / 3.5 / 5 | 1 / 1.5 / 4 | 0.53 |
| `multi_doc` | 11 | 4 | 7 | 2 / 9 / 26 | 2 / 5 / 6 | 0.31 |
| `paraphrase` | 9 | 3 | 6 | 4 / 8 / 10 | 2 / 3 / 5 | 0.29 |
| `unanswerable` | 8 | 3 | 5 | none | none | none |
| **Total** | **85** | **30** | **55** | 1 / 5 / 26 (mean 5.7) | 1 / 3 / 9 (mean 3.1) | 0.38 |

The 5 extras are 3 `point_in_time`, 1 `multi_doc` and 1 `paraphrase`. Chunk counts cover answerable items
only. Lexical overlap is the mean share of a question's content words (lowercased; stopwords and
Fed/FOMC/Federal/Reserve/Committee removed) found in its best-matching primary chunk. The low values for
`point_in_time`, `paraphrase` and `multi_doc` are where a BM25 bias against the gold should show.

**Coverage.** Answerable items per target year: 2015 5, 2016 7, 2017 6, 2018 8, 2019 10, 2020 5, 2021 7,
2022 6, 2023 5, 2024 7, 2025 6, 2026 7. 67 of 76 answerable items (88%) have a minutes chunk as primary and 45
(59%) a statement chunk. Four items cover unscheduled actions (`md-01`, `md-11`, `pp-04`, `sf-12`), and no
meeting is the target of more than 3 items.

**Split.** Per type, DEV gets 0.35 × the count rounded half up. Smoke items fill DEV first. The other items are
sorted by `id`, shuffled with a new `random.Random(7)` per type, and the open DEV slots are filled in that order.

**Human review.** Before the review, every record had `human_reviewed: false`. 20 TEST items, stratified by
type (3 `statement_fact`, 4 `minutes_fact`, 3 `point_in_time`, 2 `date_specific`, 3 `multi_doc`, 3 `paraphrase`,
2 `unanswerable`) and chosen for the most judgment calls, went to the project owner for review in
`data/eval/rag_review_items.json`, which holds the reviewer guide, each item's passages and, for unanswerable
items, the near-miss passages. The result is under "Owner review" below.

## Owner review

On 2026-10-13 the project owner reviewed the 20 TEST items in `data/eval/rag_review_items.json`: 14 were correct
as written and 6 needed a fix. None was dropped and no passage was missing. All 20 now have
`human_reviewed: true`. The six fixes (`sf-05`, `mf-11`, `mf-13`, `pit-03`, `pp-03`, `pp-04`) all found the same
flaw: `required_facts` asked for facts the question does not ask for. These were a dissenter's rationale, the
progress assessments behind a timing view, an abstention reason, the reasons for holding and the outlook, the
absence of dissent, and earlier purchase minimums. In each case the owner chose to make those facts optional and
keep the question as it was.

**The rule, applied to all 85 items.** `required_facts` may contain only facts the question asks for. Every
other fact moves, word for word, to the new `optional_facts` list. Questions, `as_of`, reference answers (which
may still mention optional facts) and the chunk lists stay the same. The `facts` indices are renumbered, and a
primary chunk that states only optional facts becomes supporting.

The owner's 14 "correct" verdicts settle how "asks for" is read. These facts stay required:
- the outcome of the decision a question is about, such as the range set at a meeting whose vote or dissent is
  asked about;
- a dissenter's preferred alternative when the question asks who dissented;
- details of the asked quantity or timing, such as the level after a change or a phase-in schedule;
- boundary facts that prove a first, last, count or complete-list claim.

These become optional unless the question asks for them: reasons, assessments, outlook and guidance, vote
facts, earlier history, and facts about a different meeting, period, measure or instrument.

**What changed.** The rule was applied the same way to DEV and TEST. Unanswerable items needed no change.

| Type | DEV items changed | TEST items changed |
|---|---:|---:|
| `statement_fact` | 3 | 4 |
| `minutes_fact` | 4 | 6 |
| `point_in_time` | 1 | 3 |
| `date_specific` | 1 | 3 |
| `multi_doc` | 2 | 0 |
| `paraphrase` | 2 | 4 |
| **Total** | **13** | **20** |

- 33 of the 85 items changed, and 41 facts moved (18 DEV, 23 TEST).
- 15 chunks became supporting: 10 chunks in 7 DEV items and 5 chunks in 5 TEST items.
- Answerable items now have 1 to 4 required facts, with a mean of 3.0 (3.6 before).
- Each changed record's `notes` has a dated line. It reads "Owner review (2026-10-13)" for the six reviewed fixes
  and "Owner's required-facts rule applied (2026-10-13)" for the rest.
- The judge sees `required_facts` only, so `answer_correctness` ignores `optional_facts`.
- `python -m app.evaluation validate` passes with 0 errors and 0 warnings.
- No TEST item was ranked, answered or judged for this fix, and it made no OpenAI calls.

The DEV retrieval figures in `docs/rag-tuning.md` were computed before this fix. Ten DEV chunks are now
supporting, so they get a grade of 1 instead of 2 in nDCG@10. Re-scored on the fixed gold, DEV `hybrid_rerank`
nDCG@10 is 0.322 for `phase3` and 0.638 for `tuned` (0.319 and 0.637 before). This does not change the frozen
`tuned` preset or the pre-registered TEST rule.

## Judge validation

The answer metrics come from an LLM judge (`gpt-5.5`, reasoning effort low, prompt `rag-judge-v1.f9315670`).
To check it, `python -m app.evaluation export-judge-sample --split test` drew a blind sample of 40 judged TEST
answers (`data/eval/rag_judge_sample.json`, seed 20261013). The judge's verdicts went to a separate key file
(`rag_judge_sample_key.json`). Each answer comes with its question, cited passages, sentences and reference
answer, and 383 slots to label: 155 claims, 73 citation checks, 115 required facts and 40 response types.
The export ran with the default `--config`, which was then `phase3`, and only `hybrid_rerank` answers had
been judged at the time. All 40 items are therefore `phase3/hybrid_rerank`, and the sample says nothing
direct about the judge on `tuned` or BM25 answers. A later export should pass `--config phase3 tuned` and
`--out` (for example `--out data/eval/rag_judge_sample_v2.json`): without `--out` it refuses to overwrite the
annotated sample.

**Who annotated.** Three annotators labelled every slot. Their labels are in
`data/eval/rag_judge_annotations.json`, one list per annotator. **All three are Claude Opus 5.5 sessions, not
people.** None saw the judge's verdicts while labelling, and none made OpenAI calls.

- **C, isolated (copied sample only): the main check.** C worked in its own directory, which held only a copy
  of `rag_judge_sample.json`. It read that copy and the judge prompt's definitions
  (`backend/app/evaluation/judge_prompts.py`), labelled all 383 slots from the passages, and saved and
  write-protected its file (sha256 `1e48968a...ed08`) before it opened the key, A's and B's labels, the
  benchmark file or the caches. Two limits on its independence: it is the same model as A and B, and its task
  brief named the judge errors already found on `tuned` answers (`ua-03`, `pp-03`, `sf-05`) and `pp-01` as a
  question-design issue, though no `phase3` verdict. Its checks afterwards: the key matches the cached
  judgments on all 40 items, and the sample's facts equal the current `required_facts`, so the sample is not
  stale.
- **A and B.** Two earlier sessions worked from the sample and the labelling instructions only, without the
  key, the caches or the benchmark file. They gave the same label on all 383 slots. B finished about a minute
  after A, and A's working files were in the same scratch directory. B reports not opening A's file, but
  their independence cannot be verified, so A vs B says nothing about reproducibility.

**Reproducibility.** C matched A on 376 of 383 labels (0.982; pass/fail κ 0.89 claims, 0.93 citations, 0.98
facts, 0.95 response type). It differed on `mf-03` claims 8-9 and citation 4, `mf-14` claim 3 and citation 3,
`sf-08` fact 2, and the `pp-01` response type. On `mf-03`, C read "it was noted", inside a list of
policymakers' observations, as support for the claims; A did not. Judge vs annotator compares `gpt-5.5` with
Claude, not with a person.

**Agreement** (`make rag-eval-report` writes it to `judge.validation` in `rag_benchmark.json`). Pass/fail
collapses the labels: a claim is supported; a citation supports its text fully or in part; a fact is
covered; the response answers or partly answers, or else declines. Kappa is Cohen's, on pass/fail.

| Check | n | Judge vs C: pass/fail | exact label | kappa | Judge vs A (= B): pass/fail, kappa | C vs A: pass/fail, kappa | A vs B: pass/fail (not independent) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Claim supported | 155 | 0.935 | 0.929 | 0.631 | 0.942, 0.677 | 0.981, 0.886 | 1.000 |
| Citation supports its text | 73 | 0.945 | 0.836 | 0.638 | 0.932, 0.580 | 0.986, 0.926 | 1.000 |
| Required fact stated | 115 | 0.983 | 0.983 | 0.962 | 0.974, 0.943 | 0.991, 0.981 | 1.000 |
| Response declines | 40 | 0.950 | 0.875 | 0.900 | 0.925, 0.850 | 0.975, 0.950 | 1.000 |

On 30 of the 383 slots the judge's label differs from C's. On 18 of them the pass/fail verdict differs:

- **Claims:** the judge passed 4 that C failed (`mf-07` claims 7 and 8, `sf-14` claim 6, `ua-04` claim 1) and
  failed 6 that C passed (`mf-03` claims 4, 8 and 9, `mf-08` claim 1, `pit-05` claim 3, `sf-05` claim 1).
- **Citations:** it passed 3 that C failed (`mf-07`, `sf-14`, `ua-04`) and failed 1 (`sf-07`).
- **Facts:** it failed 2 that C passed (`mf-09` and `sf-10`, fact 2 of each).
- **Response type:** it counted 1 decline as a partial answer (`mf-08`) and 1 partial answer as a decline
  (`md-07`).

Against A, the judge's label also differs on 30 slots, and the pass/fail verdict on 20; 16 of those 20 are
among C's 18. On this sample the judge is not biased in one direction on faithfulness. It looks slightly
strict on required facts: 2 of 115 against C, 3 against A.

**Disagreements read.** A third reader (Claude, in the Phase 7 integration step, before C existed) read these
against the passages, with the same rubric. C's labels are noted with each case:

1. **`mf-07`, claims 7 and 8: the annotators are right.** The answer says "Some participants noted that trade
   tensions were already affecting investment decisions and the agricultural sector". The passage credits
   the investment cuts to business contacts in a few Districts, and says that a number of participants
   reported most businesses had *not yet* cut back. Only "a couple of participants" tied the farm-price
   declines "partly" to trade tensions. The judge's own reasons name these mismatches but still say supported.
   This breaks its rubric: a different holder or a weaker quantifier is not supported. C agrees with A.
2. **`ua-04`, claim 1: the annotators are right on the rubric.** "Some participants worried..." where the
   passage says "Several participants". It is a minor slip, but the rubric counts it. It matters here because
   the question is unanswerable: with the claim unsupported, this phase3 decline would fail the refusal check
   (fabrication), so the strict reading would lower phase3's refusal accuracy. C agrees with A.
3. **`sf-05` and `mf-08`, claim 1: the annotators are right.** Both answers decline, then point to the
   nearest passage: "The closest related passage is the October 27-28, 2015 minutes". The judge extracted
   that statement about the passages as a claim and marked it unsupported. Its prompt says to leave out
   statements about the passages. This is the same error as the `ua-03` judgment of `tuned`, where the judge
   gave as its reason that the statement "is excluded from claim checking", yet marked it unsupported, and so
   failed a correct decline. C agrees with A.
4. **`mf-08`, response type: the annotators are right.** The answer says the passages lack the April 2020
   minutes, then summarizes the July 2020 outlook. That is a decline with related context, not a partial
   answer (the judge's label), because nothing the question asks is answered. The judge's label hides a false
   refusal. C agrees with A.
5. **`sf-10`, fact 2: the judge is right on the letter.** The answer gives the 50 basis point preference to
   "one member", as the passage does, and names George as the dissenter in the sentence before. The
   annotators (C too) counted the fact as covered by inference. The judge required the answer to say it was
   George. Both readings are defensible; the judge's matches "states the fact".
6. **`pp-01`, response type: a close call, leaning toward the judge.** The answer gives the shape of the 2017
   caps (low at first, raised every three months, then held) and says the passages lack the dollar limits,
   which are all the question asks for. A called it partial; the judge and C called it a decline. Since none
   of the asked numbers is given, "declines" fits the definition slightly better.
7. **`sf-07`, citation 1: neither label is right.** The answer's "Passage [1] says only that..." refers to
   a passage by number; it is not a citation. The citation segmenter still made it a check of the text
   "Passage". This is a harness limit, not a judge or annotator error. A and C both labelled it supports.

**Verdict.** In the cases above where one side is clearly wrong, the annotators are right more often than the
judge. The judge has two repeatable failure modes:

- It is too lenient on holder and quantifier slips (`mf-07`, `ua-04`, `sf-14`), so faithfulness is slightly
  overstated. `mf-03`'s citation 4 is not counted here: A failed it, but C reads the passage as supporting
  it, as the judge does.
- It is too strict on an answer's statements about its own passages (`sf-05`, `mf-08`, and `ua-03` for
  `tuned`), so faithfulness and refusal accuracy are understated for declines that explain themselves.

Across this sample the two roughly cancel for faithfulness. Agreement on facts is high (kappa 0.96 against C,
0.94 against A). Citation support has the lowest kappa (0.64 against C, 0.58 against A). Its pass/fail
disagreements with C are 3 holder or quantifier slips the judge passed (`mf-07`, `sf-14`, `ua-04`) and the
"Passage" segmenter artifact (`sf-07`); against A there is a fourth slip, `mf-03`'s citation 4, which C passes.
"Supports" vs "partially supports" (8 labels against C, 6 against A, mostly the judge marking an answer's
framing of its own passages as partial) lowers only exact agreement (0.836 against C, 0.849 against A), since
both count as a pass. Kappa is also low because nearly every citation passes (68 of 73 for the judge, 66 for
C). None of this affects the switch rule: its retrieval half uses no judge, and its answer half (correctness)
is the check the judge and the annotators agree on most.

## Known judge errors

Verdicts found wrong by reading judged TEST answers against their passages, outside the blind sample above.
The metrics in `rag_benchmark.json` keep the judge's verdicts. The `tuned` entries are also in the report's
notes and on `/performance`.

**`tuned` hybrid + rerank.**

- **`ua-03`.** The judge failed a correct decline. The answer added the uncited remark "They mention the June
  SEP median projections", which is a statement about the passages. The judge marked it unsupported while
  giving as its reason that such statements are excluded from checking. With that verdict fixed, `tuned`
  refusal accuracy is 5/5 and its faithfulness 0.956 (n = 51) instead of 0.937.
- **`pp-03`.** The answer is about the wrong meeting (September 2018, not Powell's first), yet the judge
  counted "25 basis points" as covered. Correctness should be 0, not 0.5.
- **`sf-05`.** The judge marked both citations of a sentence as not supporting it, although together they do.
- **`pp-01`** (a question-design issue, under review). The answer gives exactly the starting and maximum caps
  the question asks for, but scores 0/3, because the required facts bundle in the step schedule.

**`phase3` BM25**, judged after the sample was drawn. Read against their passages: 12 answers in full
(`ua-03`, `pp-01`, `sf-11`, `pit-05`, `sf-09`, `sf-08`, `ds-07`, `mf-13`, `pit-03`, `pit-07`, `pit-09`,
`pit-10`) and all 18 claims the judge failed, in 12 answers. The response types of the 38 answerable
questions BM25 declined (28) or partly answered (10) were checked against the answer text: every decline is
one.

- **`pit-07` and `pit-10`: statements about the passages failed as claims.** Both answers decline, then say
  what the passages do cover: "They discuss policy normalization, tool settings, and the pace of raising the
  federal funds rate" (`pit-07`) and "They mention market expectations for rate increases in earlier years"
  (`pit-10`). The prompt says to leave such statements out. The judge listed them as claims (3 and 2) and
  failed them as uncited; for one of the `pit-10` claims its own reason says the statement is "omitted from
  factual claim checking". Each answer scores faithfulness 0 where it should have no claims.
- **`pit-09` and `pit-03`: the same error, one claim each.** "The provided passages do not show the Fed having
  started tapering by late October 2021" (`pit-09`; the judge's reason calls it "a statement about passage
  coverage") and "That is an expectation, not a reported policy action", said of the passage the answer cites
  (`pit-03`). Faithfulness should be 1.0, not 0.8 and 0.67.
- **Effect.** With those 7 claims left out, BM25 faithfulness is 0.942 (n = 43) instead of 0.889 (n = 45). All
  four questions are answerable, so refusal accuracy does not change. This is the failure mode under "Verdict"
  above (`sf-05`, `mf-08`, and `ua-03` for `tuned`). It lowers `phase3` hybrid + rerank faithfulness too, so
  the paired difference is left as judged.
- **`ds-07`: one uncited sentence judged two ways.** The answer's first sentence carries no marker (its only
  [1] follows the second sentence). Of the three claims from that sentence, the judge failed two as uncited
  and passed the third ("Eric Rosengren dissented at the October 29-30, 2019 meeting"), also uncited. By the
  rubric all three fail (faithfulness 0.4, not 0.6); all three are in the passage nearly word for word.
- **`ua-03`: the "Passage" artifact again, not a judge error.** The decline says "Passage [1] refers only to the
  medians of the projections". As with `sf-07` above, the segmenter made that "[1]" a citation of the text
  "Passage", and the judge failed it. It is the only BM25 citation that failed: without it, BM25 citation
  accuracy is 1.000 (n = 44), not 0.978.

Checked and left as judged: `sf-11` (survey expectations credited to the Desk's schedule: unsupported,
rightly); `sf-08` (the market's reaction instead of the statement's wording: 0 of 2 facts); `sf-09` (both
dissenters' preferences without their names: all 4 facts missing, by the letter, as with `sf-10` above);
`pp-01` (a decline that says the caps were then held: 1 of 3 facts, as the rubric says); `pit-05` (an uncited
"Yes." fails, as the rubric says); and `mf-13`, partial rather than a decline, a close call: it gives the
staff's proposed 25 basis point minimum bid rate, which was adopted, but says the final terms are not in the
passages.
