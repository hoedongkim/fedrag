import { describe, expect, it } from "vitest";

import type { EvaluationReport, RetrievalRow } from "../api/types";
import { REPORT_FIXTURE } from "./fixtures";
import {
  agreementGrid,
  baselineRun,
  bestRun,
  defaultSplit,
  differencesFor,
  estimateFor,
  groupByConfig,
  headlineRun,
  heatBin,
  heatmap,
  isResolved,
  leadAnnotators,
  pairedDifference,
  parseRunKey,
  reportSplits,
  rowFor,
  rowsForSplit,
  runKey,
  uncheckedRuns,
  valueOf,
  valueRange,
} from "./report";

const report = REPORT_FIXTURE;

describe("runs", () => {
  it("round-trips a run key and rejects malformed ones", () => {
    expect(runKey({ method: "hybrid_rerank", config: "tuned" })).toBe("hybrid_rerank:tuned");
    expect(parseRunKey("hybrid_rerank:tuned")).toEqual({ method: "hybrid_rerank", config: "tuned" });
    expect(parseRunKey("tfidf:tuned")).toBeNull();
    expect(parseRunKey("bm25")).toBeNull();
    expect(parseRunKey("bm25:a:b")).toBeNull();
    expect(parseRunKey(null)).toBeNull();
  });
});

describe("splits and rows", () => {
  it("offers test first and defaults to it", () => {
    expect(reportSplits(report)).toEqual(["test", "dev", "all"]);
    expect(defaultSplit(["dev", "all"])).toBe("all");
    expect(defaultSplit(["dev"])).toBe("dev");
    expect(defaultSplit([])).toBe("test");
  });

  it("orders a split's rows by config, then from BM25 to the full pipeline", () => {
    const shuffled = [...report.retrieval].reverse();
    const rows = rowsForSplit(shuffled, "test", report.configs);
    expect(rows.map(runKey)).toEqual([
      "bm25:phase3",
      "dense:phase3",
      "hybrid:phase3",
      "hybrid_rerank:phase3",
      "bm25:tuned",
      "dense:tuned",
      "hybrid:tuned",
      "hybrid_rerank:tuned",
    ]);
    expect(rows.every((r) => r.split === "test")).toBe(true);
  });

  it("groups rows by config in report order", () => {
    const groups = groupByConfig(rowsForSplit(report.retrieval, "dev", report.configs), report.configs);
    expect(groups.map((g) => [g.config.id, g.rows.length])).toEqual([
      ["phase3", 4],
      ["tuned", 4],
    ]);
  });
});

describe("bestRun", () => {
  const rows = rowsForSplit(report.retrieval, "test", report.configs);

  it("picks the highest value, first in display order on a tie", () => {
    expect(bestRun(rows, (r) => valueOf(r.recall_at_5))).toBe("hybrid_rerank:tuned");
    expect(bestRun(rows, () => 0.5)).toBe("bm25:phase3");
  });

  it("picks the lowest value when lower is better", () => {
    expect(bestRun(rows, (r) => r.latency_ms_p50, true)).toBe("bm25:phase3");
  });

  it("names no best when fewer than two rows have a value", () => {
    expect(bestRun(rows, (r) => (r.method === "bm25" && r.config === "phase3" ? 1 : null))).toBeNull();
    expect(bestRun([], () => 1)).toBeNull();
  });
});

describe("headlineRun", () => {
  it("uses the report's selection", () => {
    expect(headlineRun(report)).toEqual({
      run: { method: "hybrid_rerank", config: "tuned" },
      metric: "ndcg_at_10",
      chosenOn: "dev",
      rule: "Highest dev nDCG@10.",
      fromReport: true,
    });
  });

  it("falls back to the best dev nDCG@10, saying so", () => {
    const dev = report.retrieval.map((r): RetrievalRow =>
      r.split === "dev" && r.method === "dense" && r.config === "phase3"
        ? { ...r, ndcg_at_10: { value: 0.99, ci: null, n: 20 } }
        : r,
    );
    const headline = headlineRun({ ...report, selection: null, retrieval: dev });
    expect(headline).toMatchObject({
      run: { method: "dense", config: "phase3" },
      chosenOn: "dev",
      fromReport: false,
    });
  });

  it("uses all questions when there is no dev split, and null without rows", () => {
    const all = report.retrieval.filter((r) => r.split === "all");
    expect(headlineRun({ ...report, selection: null, retrieval: all })?.chosenOn).toBe("all");
    expect(headlineRun({ ...report, selection: null, retrieval: [] })).toBeNull();
  });
});

describe("baselineRun", () => {
  it("is the shipped method under the first config, unless that is the headline", () => {
    expect(baselineRun(report, { method: "hybrid_rerank", config: "tuned" })).toEqual({
      method: "hybrid_rerank",
      config: "phase3",
    });
    expect(baselineRun(report, { method: "hybrid_rerank", config: "phase3" })).toBeNull();
    expect(baselineRun({ configs: [] } as Pick<EvaluationReport, "configs">, null)).toBeNull();
  });
});

describe("pairedDifference", () => {
  const tuned = { method: "hybrid_rerank", config: "tuned" } as const;
  const phase3 = { method: "hybrid_rerank", config: "phase3" } as const;

  it("finds a stored difference", () => {
    expect(pairedDifference(report.differences, "recall_at_5", "test", tuned, phase3)).toMatchObject({
      difference: 0.05,
      ci: { low: -0.02, high: 0.12 },
    });
  });

  it("negates a difference stored the other way round, interval included", () => {
    expect(pairedDifference(report.differences, "faithfulness", "test", tuned, phase3)).toEqual({
      difference: 0.04,
      ci: { low: -0.01, high: 0.09 },
      n: 60,
    });
  });

  it("is null for another split or metric", () => {
    expect(pairedDifference(report.differences, "recall_at_5", "dev", tuned, phase3)).toBeNull();
    expect(pairedDifference(report.differences, "mrr_at_10", "test", tuned, phase3)).toBeNull();
  });

  it("filters by split and metric, and tells a resolved interval", () => {
    expect(differencesFor(report.differences, "test", ["recall_at_5"])).toHaveLength(2);
    expect(isResolved({ low: 0.07, high: 0.24 })).toBe(true);
    expect(isResolved({ low: -0.02, high: 0.12 })).toBe(false);
    expect(isResolved({ low: -0.3, high: -0.1 })).toBe(true);
  });
});

describe("heatmap", () => {
  it("keeps the types with a value and a row per run in display order", () => {
    const grid = heatmap(report.per_type, "test", "recall_at_5", report.configs);
    expect(grid.types).toEqual(["statement_fact", "paraphrase"]);
    expect(grid.rows.map(runKey)).toEqual(["bm25:tuned", "hybrid_rerank:tuned"]);
    const bm25 = grid.rows[0].cells;
    expect(bm25[0]).toEqual({ type: "statement_fact", estimate: null, nQuestions: 0 });
    expect(bm25[1].estimate?.value).toBe(0.15);
    expect(bm25[1].nQuestions).toBe(8);
  });

  it("shows unanswerable questions for a refusal metric only", () => {
    expect(heatmap(report.per_type, "test", "refusal_accuracy", report.configs).types).toEqual([
      "unanswerable",
    ]);
    expect(heatmap(report.per_type, "dev", "recall_at_5", report.configs)).toEqual({ types: [], rows: [] });
  });

  it("bins shares into five equal classes", () => {
    expect([0, 0.19, 0.2, 0.45, 0.6, 0.79, 0.8, 1].map(heatBin)).toEqual([0, 0, 1, 2, 3, 3, 4, 4]);
    expect(heatBin(null)).toBeNull();
    expect(heatBin(Number.NaN)).toBeNull();
  });
});

describe("rowFor and estimateFor", () => {
  const run = { method: "hybrid_rerank", config: "tuned" } as const;

  it("finds a run's rows on a split and reads a metric from the row that has it", () => {
    const retrieval = rowFor(report.retrieval, "test", run);
    const answers = rowFor(report.answers, "test", run);
    expect(retrieval?.split).toBe("test");
    expect(estimateFor("recall_at_5", retrieval, answers)?.value).toBe(0.75);
    expect(estimateFor("faithfulness", retrieval, answers)?.value).toBe(0.95);
    expect(estimateFor("faithfulness", retrieval, null)).toBeNull();
    expect(rowFor(report.answers, "test", { method: "bm25", config: "nope" })).toBeNull();
  });
});

describe("agreementGrid", () => {
  it("puts the judge vs each annotator, then annotator pairs, in columns", () => {
    const validation = report.judge!.validation!;
    const grid = agreementGrid(validation);
    expect(grid.columns.map((c) => c.label)).toEqual(["Judge vs A", "Judge vs B", "A vs B"]);
    expect(grid.rows.map((r) => r.check)).toEqual(["claim_support", "refusal"]);
    expect(grid.rows[0].cells["judge:B"]).toEqual({ agreement: 0.875, exact: 0.86, kappa: 0.61, n: 112 });
    expect(grid.rows[0].cells["pair:A:B"]?.agreement).toBe(0.95);
    expect(grid.rows[1].cells["judge:B"]).toBeUndefined();
  });

  it("names a single unnamed annotator", () => {
    const grid = agreementGrid({
      agreement: [{ check: "fact_coverage", annotator: null, n: 3, agreement: 1, exact_agreement: 1, kappa: null }],
    });
    expect(grid.columns).toEqual([{ key: "judge:", label: "Judge vs annotator" }]);
    expect(grid.rows[0].cells["judge:"]?.n).toBe(3);
  });

  it("leads with an isolated annotator, merges identical columns and flags pairs that may not be independent", () => {
    const cell = (annotator: string, agreement: number) => ({
      check: "claim_support" as const,
      annotator,
      n: 155,
      agreement,
      exact_agreement: agreement,
      kappa: 0.6,
    });
    const pair = (annotators: string[], agreement: number) => ({
      check: "claim_support" as const,
      annotators,
      n: 155,
      agreement,
      exact_agreement: agreement,
      kappa: 0.9,
    });
    const validation = {
      annotators: [
        { id: "C", description: "isolated", isolated: true },
        { id: "A", description: "a", isolated: false },
        { id: "B", description: "b", isolated: false },
      ],
      agreement: [cell("C", 0.935), cell("A", 0.942), cell("B", 0.942)],
      inter_annotator: [pair(["C", "A"], 0.981), pair(["C", "B"], 0.981), pair(["A", "B"], 1)],
    };
    expect(agreementGrid(validation).columns.map((c) => c.label)).toEqual([
      "Judge vs C",
      "Judge vs A (= Judge vs B)",
      "C vs A (= C vs B)",
      "A vs B (not independent)",
    ]);
    expect(leadAnnotators(validation)).toEqual(["C"]);
    expect(leadAnnotators({ annotators: validation.annotators.map((a) => ({ ...a, isolated: false })) })).toEqual([
      "C",
      "A",
      "B",
    ]);
  });

  it("lists the judged runs the sample does not cover", () => {
    expect(uncheckedRuns(report, report.judge!.validation!)).toEqual(
      [...new Set(report.answers.map((r) => runKey(r)))]
        .filter((key) => key !== "hybrid_rerank:phase3")
        .map((key) => parseRunKey(key)),
    );
  });

  it("finds the range of present values", () => {
    expect(valueRange([0.5, null, 0.9, 0.7])).toEqual([0.5, 0.9]);
    expect(valueRange([null])).toBeNull();
  });
});
