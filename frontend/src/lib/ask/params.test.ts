import { describe, expect, it } from "vitest";

import {
  EMPTY_FORM,
  fieldErrorsFromValidation,
  isIsoDate,
  parseAskParams,
  serializeAskParams,
  validateAskForm,
} from "./params";

const parse = (query: string) => parseAskParams(new URLSearchParams(query));

describe("parseAskParams", () => {
  it("defaults an empty query", () => {
    expect(parse("")).toEqual(EMPTY_FORM);
  });

  it("reads every field", () => {
    expect(parse("q=Why+hike%3F&as_of=2023-04-30&method=bm25&k=8")).toEqual({
      question: "Why hike?",
      asOf: "2023-04-30",
      method: "bm25",
      topK: 8,
    });
  });

  it("falls back to defaults for invalid values", () => {
    expect(parse("as_of=2023-02-30&method=magic&k=50")).toEqual(EMPTY_FORM);
    expect(parse("as_of=yesterday&k=2")).toEqual(EMPTY_FORM);
    expect(parse("k=5.5")).toEqual(EMPTY_FORM);
    expect(parse("k=-3")).toEqual(EMPTY_FORM);
  });

  it("accepts the edges of the passage range", () => {
    expect(parse("k=3").topK).toBe(3);
    expect(parse("k=10").topK).toBe(10);
  });
});

describe("serializeAskParams", () => {
  it("leaves out defaults and blanks", () => {
    expect(serializeAskParams(EMPTY_FORM)).toBe("");
    expect(serializeAskParams({ ...EMPTY_FORM, question: "   " })).toBe("");
  });

  it("writes non-default values", () => {
    expect(
      serializeAskParams({ question: " Why? ", asOf: "2020-03-15", method: "hybrid", topK: 7 }),
    ).toBe("q=Why%3F&as_of=2020-03-15&method=hybrid&k=7");
  });

  it("round-trips through parseAskParams", () => {
    const values = {
      question: "What did participants say about tariffs & inflation?",
      asOf: "2025-07-30",
      method: "dense" as const,
      topK: 3,
    };
    expect(parse(serializeAskParams(values))).toEqual(values);
  });
});

describe("isIsoDate", () => {
  it("accepts real dates only", () => {
    expect(isIsoDate("2024-02-29")).toBe(true);
    expect(isIsoDate("2023-02-29")).toBe(false);
    expect(isIsoDate("2023-13-01")).toBe(false);
    expect(isIsoDate("2023-4-1")).toBe(false);
    expect(isIsoDate("")).toBe(false);
  });
});

describe("validateAskForm", () => {
  const today = "2026-10-12";

  it("builds the request from valid values", () => {
    expect(
      validateAskForm({ question: "  Why cut?  ", asOf: "", method: "hybrid_rerank", topK: 5 }, today),
    ).toEqual({
      ok: true,
      request: { question: "Why cut?", as_of: null, retrieval: "hybrid_rerank", top_k: 5 },
    });
  });

  it("rejects short, empty and long questions", () => {
    const errors = (question: string) => {
      const result = validateAskForm({ ...EMPTY_FORM, question }, today);
      return result.ok ? {} : result.errors;
    };
    expect(errors("").question).toBe("Enter a question.");
    expect(errors(" ab ").question).toBe("Enter at least 3 characters.");
    expect(errors("x".repeat(2001)).question).toMatch(/2,000 characters or fewer/);
    expect(errors("x".repeat(2000)).question).toBeUndefined();
  });

  it("bounds the as-of date", () => {
    const errors = (asOf: string, day: string | null = today) => {
      const result = validateAskForm({ ...EMPTY_FORM, question: "Why?", asOf }, day);
      return result.ok ? {} : result.errors;
    };
    expect(errors("2015-01-27").asOf).toMatch(/January 28, 2015/);
    expect(errors("2015-01-28").asOf).toBeUndefined();
    expect(errors(today).asOf).toBeUndefined();
    expect(errors("2026-10-13").asOf).toBe("Choose today or an earlier date.");
    expect(errors("2026-10-13", null).asOf).toBeUndefined();
    expect(errors("2023-02-30").asOf).toMatch(/valid date/);
  });

  it("rejects an out-of-range passage count", () => {
    const result = validateAskForm({ ...EMPTY_FORM, question: "Why?", topK: 11 }, today);
    expect(result.ok ? {} : result.errors).toEqual({ topK: "Choose between 3 and 10 passages." });
  });
});

describe("fieldErrorsFromValidation", () => {
  it("maps API field names onto form fields", () => {
    expect(
      fieldErrorsFromValidation({
        detail: [
          { loc: ["body", "question"], msg: "String should have at least 3 characters", type: "x" },
          { loc: ["body", "as_of"], msg: "Input should be a valid date", type: "x" },
          { loc: ["body", "top_k"], msg: "Input should be less than or equal to 20", type: "x" },
          { loc: ["body", "doc_types", 0], msg: "Input should be 'statement'", type: "x" },
        ],
      }),
    ).toEqual({
      question: "String should have at least 3 characters",
      asOf: "Input should be a valid date",
      topK: "Input should be less than or equal to 20",
      form: "Input should be 'statement'",
    });
  });

  it("falls back to a form error for other bodies", () => {
    expect(fieldErrorsFromValidation({ detail: "Bad request" })).toEqual({ form: "Bad request" });
    expect(fieldErrorsFromValidation("oops")).toEqual({ form: "The request was not valid." });
  });
});
