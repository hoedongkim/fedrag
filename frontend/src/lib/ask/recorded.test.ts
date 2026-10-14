import { describe, expect, it } from "vitest";

import { ApiError } from "../api/client";
import type { AskExample, Citation } from "../api/types";
import {
  describeRecordedError,
  parseJudgeSummary,
  parseRecordedAnswers,
  pickRecorded,
  recordedAskState,
} from "./recorded";

const citation = (n: number): Citation => ({
  document: {
    doc_id: "minutes-2023-03-22",
    doc_type: "minutes",
    meeting_date: "2023-03-22",
    release_date: "2023-04-12",
    page_updated: "2023-04-12",
    title: "Minutes of the Federal Open Market Committee, March 21-22, 2023",
    url: "https://www.federalreserve.gov/monetarypolicy/fomcminutes20230322.htm",
  },
  chunk_id: `minutes-2023-03-22#00${20 + n}`,
  section: "Committee Policy Actions",
  snippet: `Passage ${n}.`,
  passage: null,
  score: 0.9,
});

const example = {
  id: "pt-03",
  question: "How did the FOMC assess the banking system after the March 2023 bank failures?",
  as_of: "2023-04-30",
  answer: "The Committee said the banking system was sound and resilient [1][2].",
  citations: [citation(1), citation(2)],
  notes: [],
  model: "gpt-5.4-mini-2026-03-17",
  retrieval: "hybrid_rerank",
  config: "tuned",
  top_k: 5,
  latency_ms: 2371.1,
  recorded_at: "2026-10-13T14:05:00+00:00",
  type: "point_in_time",
  judge: {
    model: "gpt-5.5",
    faithfulness: 1,
    citation_accuracy: 1,
    answer_correctness: 0.5,
    refusal_correct: null,
    false_refusal: false,
    claims: [{ claim: "a", supported: true }, { claim: "b", supported: true }],
    citations: [{ n: 1, supported: true }, { n: 2, supported: false }],
    facts: [{ fact: "x", covered: true }, { fact: "y", covered: false }],
  },
};

describe("parseRecordedAnswers", () => {
  it("reads a list or an {examples} object", () => {
    const [a] = parseRecordedAnswers([example]);
    expect(a).toMatchObject({
      id: "pt-03",
      asOf: "2023-04-30",
      retrieval: "hybrid_rerank",
      config: "tuned",
      topK: 5,
      type: "point_in_time",
      recordedAt: "2026-10-13T14:05:00+00:00",
    });
    expect(a.citations).toHaveLength(2);
    expect(parseRecordedAnswers({ examples: [example], generated_at: "x" })).toHaveLength(1);
    expect(parseRecordedAnswers({ nope: true })).toEqual([]);
    expect(parseRecordedAnswers(null)).toEqual([]);
  });

  it("accepts an AskResponse under `response` and the question under `request`", () => {
    const { question, as_of, ...rest } = example;
    const nested = {
      id: "x",
      request: { question, as_of },
      response: { answer: rest.answer, citations: rest.citations, notes: [], model: rest.model, retrieval: "bm25" },
    };
    const [a] = parseRecordedAnswers([nested]);
    expect(a).toMatchObject({ question, asOf: "2023-04-30", retrieval: "bm25" });
  });

  it("leaves out examples whose answer or citations do not hold up", () => {
    const broken = { ...example, id: "b", citations: [citation(1), { chunk_id: "x" }] };
    const noAnswer = { ...example, id: "c", answer: undefined };
    const noQuestion = { ...example, id: "d", question: " " };
    expect(parseRecordedAnswers([broken, noAnswer, noQuestion, example]).map((e) => e.id)).toEqual(["pt-03"]);
  });

  it("keeps a citation's whole passage when its snippet is only the beginning", () => {
    const long = { ...citation(1), snippet: "Passage 1.", passage: "Passage 1. The supporting figure." };
    const [a] = parseRecordedAnswers([{ ...example, citations: [long, citation(2)] }]);
    expect(a.citations.map((c) => c.passage)).toEqual(["Passage 1. The supporting figure.", null]);
    const older: Record<string, unknown> = { ...citation(1) }; // a file written before `passage`
    delete older.passage;
    expect(parseRecordedAnswers([{ ...example, citations: [older] }])[0].citations[0].passage).toBeNull();
  });

  it("keeps a declined answer without citations", () => {
    const declined = { ...example, id: "ua-01", answer: "The documents do not say.", citations: [] };
    expect(parseRecordedAnswers([declined])[0].citations).toEqual([]);
  });

  it("fills defaults and makes ids unique", () => {
    const bare = { question: "Q?", answer: "A", citations: [], retrieval: "hybrid_rerank:tuned" };
    const [a, b] = parseRecordedAnswers([bare, bare]);
    expect(a).toMatchObject({
      id: "example-1",
      asOf: null,
      model: null,
      retrieval: "hybrid_rerank",
      topK: null,
      latencyMs: null,
      judge: null,
    });
    expect(b.id).toBe("example-2");
    const [c, d] = parseRecordedAnswers([example, example]);
    expect([c.id, d.id]).toEqual(["pt-03", "pt-03-2"]);
  });
});

/** One example exactly as GET /api/ask/examples serves it (data/eval/ask_examples.json). */
const SERVED = {
  answer: "In May 2024, the FOMC decided to slow the pace of decline of its securities holdings [1][2].",
  citations: [citation(1), citation(2)],
  notes: [],
  retrieval: "hybrid_rerank",
  as_of: null,
  model: "gpt-5.4-mini-2026-03-17",
  latency_ms: 2111.71,
  id: "sf-14",
  label: "Statement fact",
  description: "The May 2024 decision to slow balance sheet runoff.",
  question: "What did the FOMC decide in May 2024 about how fast to keep shrinking its bond holdings?",
  question_type: "statement_fact",
  retriever_preset: "tuned",
  recorded_at: "2026-10-13T18:05:01Z",
  judge: {
    model: "gpt-5.5",
    faithfulness: 1.0,
    citation_accuracy: 1.0,
    answer_correctness: 0.66667,
    refusal_correct: null,
    false_refusal: false,
    n_claims: 6,
    n_claims_supported: 6,
    n_facts: 3,
    n_facts_covered: 2,
  },
} satisfies AskExample;

describe("the served contract", () => {
  it("reads every field of an AskExample", () => {
    const [a] = parseRecordedAnswers([SERVED]);
    expect(a).toEqual({
      id: "sf-14",
      question: SERVED.question,
      asOf: null,
      answer: SERVED.answer,
      citations: SERVED.citations,
      notes: [],
      model: "gpt-5.4-mini-2026-03-17",
      retrieval: "hybrid_rerank",
      config: "tuned",
      topK: null,
      latencyMs: 2111.71,
      recordedAt: "2026-10-13T18:05:01Z",
      type: "statement_fact",
      label: "Statement fact",
      description: "The May 2024 decision to slow balance sheet runoff.",
      judge: {
        model: "gpt-5.5",
        text: null,
        items: [
          { label: "Claims supported", value: "6 of 6" },
          { label: "Citation accuracy", value: "1.00" },
          { label: "Required facts stated", value: "2 of 3" },
        ],
      },
    });
  });

  it("reads a correctly declined unanswerable question", () => {
    const judge = { ...SERVED.judge, answer_correctness: null, refusal_correct: true, false_refusal: null, n_facts: 0, n_facts_covered: 0 };
    expect(parseJudgeSummary(judge)?.items).toEqual([
      { label: "Claims supported", value: "6 of 6" },
      { label: "Citation accuracy", value: "1.00" },
      { label: "Unanswerable question", value: "declined, nothing invented" },
    ]);
  });
});

describe("parseJudgeSummary", () => {
  it("counts per-claim, per-citation and per-fact verdicts", () => {
    expect(parseJudgeSummary(example.judge)).toEqual({
      model: "gpt-5.5",
      text: null,
      items: [
        { label: "Claims supported", value: "2 of 2" },
        { label: "Citations that support their sentence", value: "1 of 2" },
        { label: "Required facts stated", value: "1 of 2" },
      ],
    });
  });

  it("falls back to the scores, and reports refusals", () => {
    expect(parseJudgeSummary({ faithfulness: 0.8, citation_accuracy: 1, answer_correctness: 0.667 })?.items).toEqual([
      { label: "Faithfulness", value: "0.80" },
      { label: "Citation accuracy", value: "1.00" },
      { label: "Correctness", value: "0.67" },
    ]);
    expect(parseJudgeSummary({ refusal_correct: true })?.items).toEqual([
      { label: "Unanswerable question", value: "declined, nothing invented" },
    ]);
    expect(parseJudgeSummary({ false_refusal: true })?.items).toEqual([
      { label: "Answerable question", value: "declined anyway" },
    ]);
  });

  it("takes a summary string, and is null when there is nothing to show", () => {
    expect(parseJudgeSummary("All claims supported.")).toEqual({ model: null, items: [], text: "All claims supported." });
    expect(parseJudgeSummary({ summary: "Fine", model: "gpt-5.5" })).toMatchObject({ text: "Fine", items: [] });
    expect(parseJudgeSummary({})).toBeNull();
    expect(parseJudgeSummary("  ")).toBeNull();
    expect(parseJudgeSummary(undefined)).toBeNull();
  });
});

describe("recordedAskState", () => {
  it("is a finished answer with numbered citations", () => {
    const [a] = parseRecordedAnswers([example]);
    const state = recordedAskState(a, 3);
    expect(state).toMatchObject({ status: "done", text: example.answer, failure: null, run: 3 });
    expect(state.request).toEqual({
      question: example.question,
      as_of: "2023-04-30",
      retrieval: "hybrid_rerank",
      top_k: 5,
    });
    expect(state.citations.map((c) => c.n)).toEqual([1, 2]);
    expect(state.result).toMatchObject({ model: example.model, latency_ms: 2371.1, as_of: "2023-04-30" });
  });

  it("leaves top_k out when it was not recorded, and marks a missing latency as NaN", () => {
    const [a] = parseRecordedAnswers([{ ...example, top_k: undefined, latency_ms: undefined }]);
    const state = recordedAskState(a);
    expect(state.request).not.toHaveProperty("top_k");
    expect(Number.isNaN(state.result?.latency_ms)).toBe(true);
  });
});

describe("pickRecorded", () => {
  it("finds the example named in the URL, else the first", () => {
    const list = parseRecordedAnswers([example, { ...example, id: "sf-01" }]);
    expect(pickRecorded(list, "sf-01")?.id).toBe("sf-01");
    expect(pickRecorded(list, "nope")?.id).toBe("pt-03");
    expect(pickRecorded(list, null)?.id).toBe("pt-03");
    expect(pickRecorded([], null)).toBeNull();
  });
});

describe("describeRecordedError", () => {
  it("says when the backend has no recorded answers, and reads other errors as elsewhere", () => {
    expect(describeRecordedError(new ApiError(404, "Not Found", { detail: "Not Found" }))).toMatchObject({
      tone: "setup",
      title: "Recorded answers are not available",
    });
    const detail = "data/eval/ask_examples.json is missing";
    expect(describeRecordedError(new ApiError(503, detail, { detail }))).toMatchObject({ message: detail });
    expect(describeRecordedError(new TypeError("fetch failed"))).toMatchObject({ title: "Backend unreachable" });
  });
});
