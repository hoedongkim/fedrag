import { describe, expect, it } from "vitest";

import { DEFAULT_VIEW, isQuestionId, parsePerformanceParams, serializePerformanceParams } from "./params";

const parse = (query: string) => parsePerformanceParams(new URLSearchParams(query));

describe("parsePerformanceParams", () => {
  it("defaults to the test split, every question and the headline run", () => {
    expect(parse("")).toEqual(DEFAULT_VIEW);
  });

  it("reads valid values and drops invalid ones", () => {
    expect(parse("split=dev&type=paraphrase&outcome=missed&run=bm25:phase3&q=q012")).toEqual({
      split: "dev",
      type: "paraphrase",
      outcome: "missed",
      run: { method: "bm25", config: "phase3" },
      question: "q012",
    });
    expect(parse("split=train&type=trivia&outcome=bad&run=tfidf:x&q=../etc")).toEqual(DEFAULT_VIEW);
  });

  it("checks question ids like the backend", () => {
    expect(isQuestionId("s01")).toBe(true);
    expect(isQuestionId("q-12.b_3")).toBe(true);
    expect(isQuestionId("-q1")).toBe(false);
    expect(isQuestionId("a b")).toBe(false);
    expect(isQuestionId("x".repeat(65))).toBe(false);
    expect(isQuestionId(null)).toBe(false);
  });
});

describe("serializePerformanceParams", () => {
  it("leaves defaults out", () => {
    expect(serializePerformanceParams(DEFAULT_VIEW)).toBe("");
  });

  it("round-trips a full view, keeping the run readable", () => {
    const view = {
      split: "all",
      type: "multi_doc",
      outcome: "unfaithful",
      run: { method: "hybrid_rerank", config: "tuned" },
      question: "q007",
    } as const;
    const query = serializePerformanceParams(view);
    expect(query).toBe("split=all&type=multi_doc&outcome=unfaithful&run=hybrid_rerank:tuned&q=q007");
    expect(parse(query)).toEqual(view);
  });
});
