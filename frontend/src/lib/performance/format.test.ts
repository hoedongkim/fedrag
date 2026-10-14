import { describe, expect, it } from "vitest";

import { ApiError } from "../api/client";
import { describeEvaluationError } from "./failures";
import {
  configLabel,
  configStyle,
  formatDelta,
  formatMs,
  formatSpend,
  formatTokens,
  formatUsd,
  runLabel,
} from "./format";

const configs = [{ id: "tuned", label: "Tuned on dev" }];

describe("labels", () => {
  it("names configs and runs, falling back to the id", () => {
    expect(configLabel(configs, "tuned")).toBe("Tuned on dev");
    expect(configLabel(configs, "phase9")).toBe("phase9");
    expect(runLabel(configs, { method: "hybrid_rerank", config: "tuned" })).toBe(
      "Hybrid + rerank · Tuned on dev",
    );
  });
});

describe("numbers", () => {
  it("formats signed differences with a true minus", () => {
    expect(formatDelta(0.05, { low: -0.02, high: 0.12 })).toBe("+0.05 (−0.02 to +0.12)");
    expect(formatDelta(-0.004)).toBe("0.00");
  });

  it("gives a resolved difference a third decimal rather than a bound of 0.00", () => {
    expect(formatDelta(0.074, { low: 0.0046, high: 0.154 })).toBe("+0.074 (+0.005 to +0.154)");
    expect(formatDelta(-0.074, { low: -0.154, high: -0.0046 })).toBe("−0.074 (−0.154 to −0.005)");
    expect(formatDelta(0.1, { low: 0, high: 0.2 })).toBe("+0.10 (0.00 to +0.20)"); // not resolved
  });

  it("formats costs down to fractions of a cent", () => {
    expect(formatUsd(0.91)).toBe("$0.91");
    expect(formatUsd(1.234)).toBe("$1.23");
    expect(formatUsd(0.0012345)).toBe("$0.0012");
    expect(formatUsd(0.04)).toBe("$0.04");
    expect(formatUsd(0)).toBe("$0");
    expect(formatUsd(null)).toBe("n/a");
    expect(formatSpend({ total_usd: 0.91, max_usd: 1 })).toBe("$0.91 (cap $1.00 per command)");
    expect(formatSpend({ total_usd: 0.91, max_usd: null })).toBe("$0.91");
  });

  it("formats latencies and token counts", () => {
    expect(formatMs(5.2)).toBe("5 ms");
    expect(formatMs(2400)).toBe("2.4 s");
    expect(formatMs(null)).toBe("n/a");
    expect(formatTokens(1632000)).toBe("1.63M");
    expect(formatTokens(176800)).toBe("177k");
    expect(formatTokens(900)).toBe("900");
  });
});

describe("describeEvaluationError", () => {
  it("explains an evaluation that has not run as setup, with the command", () => {
    const failure = describeEvaluationError(
      new ApiError(503, "Evaluation not run yet: run make rag-eval."),
    );
    expect(failure).toMatchObject({ tone: "setup", command: "make rag-eval" });
  });

  it("passes an invalid file's problems through", () => {
    const detail = "Evaluation file data/eval/rag_benchmark.json is invalid: configs: too short";
    expect(describeEvaluationError(new ApiError(503, detail))).toEqual({
      tone: "error",
      title: "Evaluation file is invalid",
      message: detail,
    });
  });

  it("reports a missing question and an unreachable backend", () => {
    expect(describeEvaluationError(new ApiError(404, "No evaluation question with id q9.")).title).toBe(
      "Question not found",
    );
    expect(describeEvaluationError(new TypeError("fetch failed"))).toMatchObject({
      title: "Backend unreachable",
      command: "make dev-backend",
    });
  });
});

describe("configStyle", () => {
  it("gives each config a fixed slot and shape, and later ones a neutral ink", () => {
    const list = [{ id: "phase3" }, { id: "tuned" }, { id: "x" }, { id: "y" }];
    expect(configStyle(list, "phase3")).toEqual({ color: "var(--series-1)", shape: "circle" });
    expect(configStyle(list, "tuned")).toEqual({ color: "var(--series-2)", shape: "diamond" });
    expect(configStyle(list, "y")).toEqual({ color: "var(--chart-ink)", shape: "square" });
    expect(configStyle(list, "missing").color).toBe("var(--chart-ink)");
  });
});
