import { describe, expect, it } from "vitest";

import { ApiError } from "../api/client";
import { COMPARE_FIXTURE } from "./fixtures";
import { analysisStatusNotice, describeCompareError, describeGenerateError } from "./failures";
import { describeStats, formatChangeIds, formatSimilarity, sentences } from "./format";

describe("describeCompareError", () => {
  it("turns a 404 into quick picks of the nearest statement dates", () => {
    const failure = describeCompareError(
      new ApiError(404, "No FOMC statement for meeting date 2024-08-15. Nearest statements: 2024-07-31 and 2024-09-18."),
    );
    expect(failure).toMatchObject({ kind: "not_found", dates: ["2024-08-15", "2024-07-31", "2024-09-18"] });
  });

  it("explains a missing or empty database", () => {
    expect(describeCompareError(new ApiError(503, "Database not configured: set DATABASE_URL."))).toMatchObject({
      kind: "notice",
      notice: { tone: "setup", title: "Database not configured" },
    });
    expect(describeCompareError(new ApiError(503, "No documents: run make db-load."))).toMatchObject({
      notice: { tone: "setup", command: "make db-load" },
    });
  });

  it("points to the backend when it is down or unreachable", () => {
    expect(describeCompareError(new ApiError(500, "Internal Server Error"))).toMatchObject({
      notice: { tone: "error", command: "make dev-backend" },
    });
    expect(describeCompareError(new TypeError("fetch failed"))).toMatchObject({
      notice: { title: "Backend unreachable" },
    });
  });
});

describe("analysis notices", () => {
  it("has a notice only for the statuses without an analysis to show", () => {
    expect(analysisStatusNotice("ready")).toBeNull();
    expect(analysisStatusNotice("not_cached")).toBeNull();
    expect(analysisStatusNotice("llm_not_configured")).toMatchObject({ tone: "setup" });
    expect(analysisStatusNotice("failed")).toMatchObject({ tone: "error" });
  });

  it("describes a failed generate request", () => {
    expect(describeGenerateError(new ApiError(502, "OpenAI timed out"))).toMatchObject({
      tone: "error",
      title: "The analysis could not be generated",
    });
    expect(describeGenerateError(new ApiError(503, "LLM not configured: set OPENAI_API_KEY"))).toMatchObject({
      tone: "setup",
    });
    expect(describeGenerateError(new TypeError("fetch failed"))).toMatchObject({ title: "Backend unreachable" });
  });
});

describe("format", () => {
  it("formats similarity as a whole percent", () => {
    expect(formatSimilarity(0.7962)).toBe("80%");
    expect(formatSimilarity(1)).toBe("100%");
    expect(formatSimilarity(null)).toBe("n/a");
    expect(formatSimilarity(1.5)).toBe("n/a");
  });

  it("describes the stats and change ids", () => {
    expect(describeStats(COMPARE_FIXTURE.stats)).toBe(
      "8 modified, 1 added, 3 removed, 6 unchanged; 80% similar overall.",
    );
    expect(formatChangeIds([7, 9])).toBe("#7, #9");
    expect(sentences(1)).toBe("1 sentence");
    expect(sentences(3)).toBe("3 sentences");
  });
});
