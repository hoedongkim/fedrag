[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Next.js](https://img.shields.io/badge/Next.js-16-black.svg)](https://nextjs.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

# FedRAG

**Ask what the Federal Reserve actually said, and get an answer built only from its own documents,
with a source for every claim.**

*Solo project. Data: every FOMC statement and set of meeting minutes from 2015 to 2026.*

[![Ask the Fed: an answer with numbered citations and the passages behind them](docs/images/ask-demo-desktop-light.png)](https://fedrag-seven.vercel.app/ask)

*[Open the live demo →](https://fedrag-seven.vercel.app) · [Read the full technical report →](docs/technical-report.md)*

---

## 🎯 Problem

Eight times a year, the Federal Reserve's rate-setting committee (the **FOMC**) meets and publishes
a short **statement**. About three weeks later it publishes the **minutes**, a longer record of the
discussion. Markets read both word by word, because one changed phrase can shift expectations about
interest rates within minutes.

General chatbots handle this badly. They answer from memory, so they mix up meetings, paraphrase
where the exact wording matters, and sometimes invent a quote that sounds right. For a question like
*"What did the Fed say about inflation in March 2022?"*, a confident wrong answer is worse than no
answer.

FedRAG answers **only from the Fed's documents**. It cites the passage behind every claim and says so
when the documents do not cover a question. It can also answer **as of a past date**, using only what
had been published by then. That is what you need to study how markets could have read the Fed at
the time.

## 📊 Data

Every FOMC statement and set of minutes from January 2015 to September 2026, collected from
[federalreserve.gov](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm).

|  | Count | Note |
|---|---|---|
| Meetings | 97 | Includes 4 unscheduled meetings or votes in 2019–2020 |
| Statements | 97 | 124 to 614 words each |
| Minutes | 94 | Median of about 8,200 words |
| Searchable passages | 3,077 | Statements are kept whole; minutes are split into passages |

**Release dates are read from the Fed, never estimated.** That is what keeps "as of" answers honest.
Minutes come out about three weeks after a meeting, so the day after a meeting only the statement
can be used.

Two more sources:

- The **federal funds target range** from [FRED](https://fred.stlouisfed.org/series/DFEDTARU).
  Statement tone is plotted against it, so tone can be compared with actual rate decisions.
- **[Trillion Dollar Words](https://aclanthology.org/2023.acl-long.368/)**, FOMC sentences that
  researchers labelled by hand. It is used to train and test the tone models.

## 🔍 Approach

**Two terms first.**
- *Hawkish* means leaning toward higher interest rates to cool inflation.
- *Dovish* means leaning toward lower rates to support jobs and growth.

**1 — Find the right passages.**
- Two searches run side by side: one matches exact words (BM25) and one matches meaning (embeddings).
- Their results are merged, then a second model re-reads the top candidates and reorders them.
- Before any search, every document released after the "as of" date is removed, so later
  information cannot leak into the answer.

**2 — Answer from those passages only.**
- An LLM (`gpt-5.4-mini`) writes the answer from the top 5 passages and marks each claim with a
  numbered citation such as `[1]`.
- It declines when the passages do not contain the answer.
- The answer appears on screen as it is written.

**3 — Read the tone of each statement.** Four methods label every statement hawkish, neutral or
dovish:
- a keyword list;
- FinBERT, a finance language model fine-tuned on the labelled sentences;
- an LLM given worked examples;
- the same LLM given none.

**4 — Show what changed between two statements.**
- The two texts are lined up sentence by sentence, with word edits highlighted. No LLM is involved
  in this step.
- An LLM then explains what the new wording signals. Each point has to cite the changed sentences
  it rests on, and the code checks those citations.

**5 — Grade the system on questions it was not tuned on.**
- There are 85 questions. 76 come with the passages that answer them. The other 9 cannot be
  answered from the documents, which tests whether the system declines.
- The questions were drafted with an LLM; 20 were reviewed by hand.
- Search settings were tuned on 30 questions only, then scored once on the other 55.
- A stronger LLM (`gpt-5.5`) graded each answer claim by claim.

## 📈 Results

**Ask the Fed.** Scores on the 55 held-out test questions, before and after tuning the search:

| What was measured | Before | After |
|---|---:|---:|
| Right passages ranked near the top (nDCG@10, 0 to 1) | 0.357 | **0.579** |
| Required facts the answer states (correctness, 0 to 1) | 0.413 | **0.553** |
| Claims supported by the passage they cite (faithfulness, 0 to 1) | 0.883 | **0.937** |
| Answerable questions wrongly declined | 36% | **14%** |

*Search, correctness and wrong refusals are measured on the 50 answerable questions; faithfulness*
*on the answers that make claims (47 before, 52 after). Both main gains hold at 95% confidence:*
*+0.222 [+0.158, +0.293] for search and +0.140 [+0.068, +0.218] for correctness, from a paired*
*bootstrap over questions. The switch to the tuned search followed a rule written down before the*
*single test run. An answer takes about 2 seconds and costs about $0.003.*

**Policy Stance.** Share of statements labelled correctly, out of 25 that a person labelled without
seeing any model's answer:

| Method | Accuracy |
|---|---:|
| Always answer "hawkish" (no model) | 0.36 |
| FinBERT, averaging its sentence scores | 0.36 |
| Keyword list | 0.56 |
| LLM with worked examples | 0.76 |
| **LLM without examples** | **0.84** |

*25 statements is a small sample. The best method's 95% interval is 0.68 to 0.96, and the gap*
*between the two LLM versions is not statistically clear.*

**Statement Comparison.** There are 22 consecutive pairs where the stance really moved. In 20 of
them, the LLM's reading named the right direction. But it also reads a shift into 49 of the 74 pairs
where nothing moved, so it over-calls change.

## 💡 Key Insights

**1 — Respecting dates did most of the work.** Most test questions name a month or a day, such as
"in March 2022". Teaching the search to favour documents from that date produced most of the
improvement: +0.255 on the 40 dated questions, and about zero on the 5 that name no date at all.

**2 — A model that reads sentences well can still misread a whole statement.** On single sentences
FinBERT holds up reasonably (0.60 macro-F1, against 0.67 for the LLM). But most of a statement is
standard wording repeated every meeting, and the signal sits in one or two sentences about rates.
Averaging every sentence washes that signal out. FinBERT called 85 of 97 statements neutral and did
no better than always guessing "hawkish".

**3 — Saying "I don't know" is safe, but only up to a point.** Keyword-only search found the right
passages so rarely that its answers declined 56% of answerable questions. The tuned search cut wrong
refusals to 14%, and its answers became more faithful, not less.

**4 — Check the grader too.** Every answer score comes from an LLM grader. On 40 answers it agreed
with a separate LLM reviewer on 94–98% of its pass/fail calls. No person has checked it yet. Reading
the answers still turned up four wrong verdicts. They are listed in the report, not quietly fixed.

## 🧭 What This Would Mean in Practice

- **For an analyst:** a question about Fed wording gets an answer in seconds, for well under a cent.
  Every claim is one click from its source.
- **For historical studies:** an "as of" answer uses only documents that were public on that date.
  No test run ever returned a later passage. The model's own background knowledge was not measured
  separately.
- **Where not to trust it yet:** two kinds of question still find the right passages less often.
  These are questions that span several meetings (nDCG@10 0.33) and questions about what was known
  on a past date (0.29). The comparison tab also over-calls shifts.

## 📌 Takeaways

**Measure on questions you did not tune on.** The tuned search gained +0.317 on the tuning questions
but +0.222 on the held-out test. Only the second number is a fair estimate.

**Decide the rule before you see the result.** Whether to switch to the tuned search was settled by
a rule written before the single test run, so the result could not bend the decision.

**Simple baselines keep you honest.** Always guessing "hawkish" matched FinBERT on stance. Always
saying "unchanged" agrees with the reference more often than the LLM's comparison does (0.77 against
0.47). Without these baselines, both weaknesses would have gone unnoticed.

## 🖥 The Web App

<table>
  <tr>
    <td width="50%" valign="top"><img src="docs/images/ask-tile-light.png" alt="Ask the Fed: an answer about the U.S. banking system as of April 30, 2023, with numbered citation markers, its metadata line and the first source"><br><b>Ask the Fed</b>: ask a question, optionally as of a date, and get an answer with a source for every claim</td>
    <td width="50%" valign="top"><img src="docs/images/stance-tile-light.png" alt="Policy Stance: four methods' tone scores for every statement since 2015, each statement's reference label and the federal funds target range"><br><b>Policy Stance</b>: every statement's tone since 2015 by four methods, against the actual interest rate</td>
  </tr>
  <tr>
    <td width="50%" valign="top"><img src="docs/images/compare-tile-light.png" alt="Statement Comparison: the July and September 2024 statements lined up sentence by sentence, with deleted words struck through and inserted words highlighted"><br><b>Statement Comparison</b>: pick two statements and see which sentences changed, word by word</td>
    <td width="50%" valign="top"><img src="docs/images/performance-tile-light.png" alt="RAG Performance: the tuned search's scores on the held-out test questions, each with its interval and its change from the old setup"><br><b>RAG Performance</b>: search and answer scores on held-out test questions, with their intervals</td>
  </tr>
</table>

**On the public demo, OpenAI features are switched off**, so it costs nothing to host:

- Ask the Fed shows eight real answers recorded during the evaluation.
- Statement Comparison shows the stored readings of the 96 consecutive pairs.
- The API runs as a Vercel function, which does not sleep: after a quiet spell, the first request
  waits a few seconds for a cold start.

**Run it yourself:**

```bash
make setup                             # Python and Node dependencies
cp backend/.env.example backend/.env   # settings, all optional
make dev                               # open http://127.0.0.1:3000
```

With nothing else added, RAG Performance works from the saved results, and `PUBLIC_DEMO=true make
dev` shows Ask the Fed's eight recorded answers, as the public demo does. Statement Comparison needs
the collected documents in a database, and Policy Stance also needs its classifiers run. Asking new
questions needs an OpenAI key and the search index (about $0.02 to build). The steps are in the
[technical report](docs/technical-report.md#-run-it-yourself).

## 📁 Repository Structure

```
├── backend/             FastAPI app: data collection, search, answers, stance, comparison, evaluation
├── frontend/            Next.js web app (the overview page and four tabs)
├── api/                 The public demo's API as a Vercel function (with the root vercel.json)
├── data/eval/           Test questions, recorded answers and every benchmark result
├── docs/                Technical report, evaluation and labelling notes, deploy guide
├── deploy/              A read-only database role, and a Hugging Face Space image (alternative host)
├── .github/workflows/   CI: backend tests with Postgres, frontend checks, API contract
└── Makefile             make setup / dev / check, plus every pipeline step
```

## 🛠 Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?logo=langchain&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?logo=openai&logoColor=white)
![Chroma](https://img.shields.io/badge/Chroma-FF6446)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?logo=postgresql&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-000000?logo=nextdotjs&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?logo=typescript&logoColor=white)
![Tailwind CSS](https://img.shields.io/badge/Tailwind%20CSS-06B6D4?logo=tailwindcss&logoColor=white)
![Vercel](https://img.shields.io/badge/Vercel-000000?logo=vercel&logoColor=white)

## License

[MIT](LICENSE). Third-party data and models keep their own licenses ([details](docs/technical-report.md#license)).
