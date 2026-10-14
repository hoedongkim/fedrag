import { describe, expect, it } from "vitest";

import type {
  BaselineDifference,
  PairedDifference,
  SentenceBaseline,
  SentenceBenchmark,
  StatementBenchmark,
  StatementVariant,
} from "../api/types";
import { ApiError } from "../api/client";
import { markWakeExhausted } from "../api/wake";
import {
  baselineComparisons,
  baselineDetail,
  best,
  isPostHoc,
  pairedComparisons,
  reviewStatus,
  sentenceBaselineTable,
  sentenceTable,
  statementRowKey,
  statementTable,
} from "./benchmark";
import { describeStanceError, isNotLoaded, suggestsBackendDown } from "./failures";

function sentence(
  classifier: SentenceBenchmark["classifier"],
  split: SentenceBenchmark["split"],
  macro: number,
): SentenceBenchmark {
  return {
    classifier,
    split,
    macro_f1_mean: macro,
    macro_f1_sd: 0.01,
    accuracy_mean: macro,
    accuracy_sd: null,
    per_class_f1_mean: { dovish: macro, neutral: macro, hawkish: macro },
    n_seeds: 3,
    n_test_mean: 496,
  };
}

function statement(
  classifier: StatementBenchmark["classifier"],
  kappa: number | null,
  variant: StatementVariant = "pre_registered",
) {
  return {
    classifier,
    variant,
    n: 97,
    accuracy: 0.5,
    macro_f1: 0.5,
    kappa,
    accuracy_ci: { low: 0.4, high: 0.6 },
    macro_f1_ci: null,
    reference: "llm_adjudicated",
    spearman: null,
    human_subset: null,
  } satisfies StatementBenchmark;
}

function difference(
  minuend: StatementBenchmark["classifier"],
  variant: StatementVariant,
  subset: PairedDifference["subset"],
): PairedDifference {
  const metric = { difference: 0.05, ci: { low: -0.03, high: 0.13 } };
  return {
    minuend: { classifier: minuend, variant },
    subtrahend: {
      classifier: minuend === "finbert" ? "finbert" : "llm_few_shot",
      variant: "pre_registered",
    },
    subset,
    n: subset === "all" ? 97 : 25,
    accuracy: metric,
    macro_f1: metric,
  };
}

describe("sentenceTable", () => {
  it("pairs the two splits per classifier, in display order", () => {
    const table = sentenceTable([
      sentence("keyword", "deduplicated", 0.4),
      sentence("llm_few_shot", "official", 0.66),
      sentence("keyword", "official", 0.41),
    ]);

    expect(table.map((r) => r.classifier)).toEqual(["llm_few_shot", "keyword"]);
    expect(table[0].deduplicated).toBeNull();
    expect(table[1].official?.macro_f1_mean).toBe(0.41);
    expect(table[1].deduplicated?.macro_f1_mean).toBe(0.4);
  });
});

describe("statementTable / best", () => {
  it("orders statement rows and finds the best, skipping missing values", () => {
    const rows = statementTable([statement("keyword", 0.3), statement("finbert", null)]);

    expect(rows.map((r) => r.classifier)).toEqual(["finbert", "keyword"]);
    expect(best(rows, (r) => r.kappa)).toBe("keyword");
    expect(best([], (r: StatementBenchmark) => r.kappa)).toBeNull();
  });

  it("keeps the first classifier on a tie", () => {
    const rows = [statement("llm_few_shot", 0.7), statement("llm_zero_shot", 0.7)];
    expect(best(rows, (r) => r.kappa)).toBe("llm_few_shot");
  });

  it("puts post-hoc rows after the pre-registered ones, with their own keys", () => {
    const rows = statementTable([
      statement("finbert", 0.5, "post_hoc_rubric_ordered"),
      statement("keyword", 0.3),
      statement("finbert", 0.1),
      statement("keyword", 0.6, "post_hoc_rubric_ordered"),
    ]);

    expect(rows.map(statementRowKey)).toEqual([
      "finbert:pre_registered",
      "keyword:pre_registered",
      "finbert:post_hoc_rubric_ordered",
      "keyword:post_hoc_rubric_ordered",
    ]);
    expect(rows.map(isPostHoc)).toEqual([false, false, true, true]);
    expect(best(rows.filter((r) => !isPostHoc(r)), (r) => r.kappa)).toBe("keyword");
  });
});

describe("pairedComparisons", () => {
  it("groups the two subsets of each compared pair, in file order", () => {
    const pairs = pairedComparisons({
      method: "Percentile bootstrap",
      resamples: 10_000,
      seed: 1,
      level: 0.95,
      paired_differences: [
        difference("llm_zero_shot", "pre_registered", "all"),
        difference("llm_zero_shot", "pre_registered", "human_subset"),
        difference("finbert", "post_hoc_rubric_ordered", "all"),
      ],
      baseline_differences: [],
    });

    expect(pairs).toHaveLength(2);
    expect(pairs[0].minuend.classifier).toBe("llm_zero_shot");
    expect(pairs[0].all?.n).toBe(97);
    expect(pairs[0].humanSubset?.n).toBe(25);
    expect(pairs[1].minuend.variant).toBe("post_hoc_rubric_ordered");
    expect(pairs[1].humanSubset).toBeNull();
    expect(pairedComparisons(null)).toEqual([]);
  });
});

describe("baselines", () => {
  const metric = { difference: 0, ci: { low: -0.03, high: 0.03 } };
  const against = (
    classifier: StatementBenchmark["classifier"],
    subset: BaselineDifference["subset"],
  ): BaselineDifference => ({
    row: { classifier, variant: "post_hoc_rubric_ordered" },
    baseline: "rate_move_rule",
    subset,
    n: subset === "all" ? 97 : 25,
    accuracy: metric,
    macro_f1: metric,
  });

  it("groups each row's differences from a baseline, in file order", () => {
    const pairs = baselineComparisons({
      method: "Percentile bootstrap",
      resamples: 10_000,
      seed: 1,
      level: 0.95,
      paired_differences: [],
      baseline_differences: [
        against("finbert", "all"),
        against("finbert", "human_subset"),
        against("keyword", "all"),
      ],
    });

    expect(pairs.map((p) => [p.row.classifier, p.baseline])).toEqual([
      ["finbert", "rate_move_rule"],
      ["keyword", "rate_move_rule"],
    ]);
    expect(pairs[0].humanSubset?.n).toBe(25);
    expect(pairs[1].humanSubset).toBeNull();
    expect(baselineComparisons(null)).toEqual([]);
  });

  it("pairs the sentence baseline's splits and says what each baseline predicts", () => {
    const split = (name: SentenceBaseline["split"], accuracy: number): SentenceBaseline => ({
      name: "majority_class",
      label: "neutral",
      split: name,
      macro_f1_mean: 0.217,
      macro_f1_sd: 0.004,
      accuracy_mean: accuracy,
      accuracy_sd: 0.012,
      n_seeds: 3,
      n_test_mean: 476,
    });
    const [row] = sentenceBaselineTable([split("deduplicated", 0.48), split("official", 0.482)]);

    expect(row.official?.accuracy_mean).toBe(0.482);
    expect(row.deduplicated?.accuracy_mean).toBe(0.48);
    expect(baselineDetail(row)).toBe("always neutral");
    expect(baselineDetail({ name: "rate_move_rule", label: null })).toBe("decision verb only");
    expect(sentenceBaselineTable([])).toEqual([]);
  });
});

describe("reviewStatus", () => {
  it("tells pending from reviewed reference labels", () => {
    expect(reviewStatus({ reference_labels: null })).toEqual({ kind: "none" });
    expect(reviewStatus({ reference_labels: { n_labels: 97, n_human_reviewed: 0 } })).toEqual({
      kind: "pending",
      nLabels: 97,
    });
    expect(reviewStatus({ reference_labels: { n_labels: 97, n_human_reviewed: 25 } })).toEqual({
      kind: "reviewed",
      nLabels: 97,
      nHuman: 25,
    });
  });
});

describe("describeStanceError", () => {
  it("explains each 503 with its fix", () => {
    const notLoaded = new ApiError(503, "Stance results not loaded: run make stance.");
    expect(describeStanceError(notLoaded)).toMatchObject({
      tone: "setup",
      command: "make stance",
    });
    expect(isNotLoaded(notLoaded)).toBe(true);

    const noDb = new ApiError(503, "Database not configured: set DATABASE_URL in backend/.env.");
    expect(describeStanceError(noDb).title).toBe("Database not configured");
    expect(isNotLoaded(noDb)).toBe(false);

    const unloaded = new ApiError(503, "Database has no fedrag tables yet: run make db-load.");
    expect(describeStanceError(unloaded).command).toBe("make db-load && make stance");

    const noBenchmark = new ApiError(503, "Stance benchmark not computed yet: run make stance.");
    expect(describeStanceError(noBenchmark).title).toBe("Benchmark not computed yet");
  });

  it("treats a 5xx or a network error as the backend failing or being down", () => {
    expect(describeStanceError(new ApiError(500, "Internal Server Error"))).toMatchObject({
      tone: "error",
      command: "make dev-backend",
    });
    expect(describeStanceError(new TypeError("fetch failed")).title).toBe("Backend unreachable");
    expect(describeStanceError(new ApiError(404, "nope")).title).toBe("Request failed (HTTP 404)");
    expect(suggestsBackendDown(new ApiError(500, "x"))).toBe(true);
    expect(suggestsBackendDown(new TypeError("fetch failed"))).toBe(true);
    expect(suggestsBackendDown(new ApiError(503, "Stance results not loaded"))).toBe(false);
    // Already waited out a whole wake-up: re-checking would only start another one.
    const exhausted = new TypeError("fetch failed");
    markWakeExhausted(exhausted);
    expect(suggestsBackendDown(exhausted)).toBe(false);
  });
});
