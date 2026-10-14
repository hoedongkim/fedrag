import { describe, expect, it } from "vitest";

import type { CompareResponse, EvaluationReport, StanceBenchmark } from "../api/types";
import { COMPARE_FIXTURE } from "../compare/fixtures";
import { REPORT_FIXTURE } from "../performance/fixtures";
import {
  askHeadline,
  compareHeadline,
  evaluationItems,
  evaluationFacts,
  formatPercent,
  ragHeadline,
  stanceHeadline,
} from "./results";

/** The fixture, plus the paired nDCG@10 difference the real report stores (tuned minus phase3). */
const REPORT: EvaluationReport = {
  ...REPORT_FIXTURE,
  differences: [
    ...REPORT_FIXTURE.differences,
    {
      metric: "ndcg_at_10",
      split: "test",
      minuend: { method: "hybrid_rerank", config: "phase3" },
      subtrahend: { method: "hybrid_rerank", config: "tuned" },
      n: 60,
      difference: -0.05,
      ci: { low: -0.09, high: -0.01 },
    },
  ],
};

const STANCE = {
  reference_labels: { n_labels: 97, n_human_reviewed: 25 },
  statement_level: [
    {
      classifier: "llm_few_shot",
      variant: "pre_registered",
      human_subset: { n: 25, accuracy: 0.76, accuracy_ci: { low: 0.6, high: 0.92 } },
    },
    {
      classifier: "llm_zero_shot",
      variant: "post_hoc_rubric_ordered",
      human_subset: { n: 25, accuracy: 0.5, accuracy_ci: null },
    },
    {
      classifier: "llm_zero_shot",
      variant: "pre_registered",
      human_subset: { n: 25, accuracy: 0.84, accuracy_ci: { low: 0.68, high: 0.96 } },
    },
  ],
} as unknown as StanceBenchmark;

describe("ragHeadline", () => {
  it("quotes the chosen run's test nDCG@10 against the setup it replaced, with the paired difference", () => {
    // Fixture: hybrid_rerank tuned test = 0.7 + 0.05 - 0.05 = 0.70; phase3 = 0.65.
    expect(ragHeadline(REPORT)).toEqual({
      label: "nDCG@10, held-out test questions",
      value: "0.70",
      versus: "vs 0.65",
      caption:
        "Ranking quality of the passages hybrid + rerank retrieval finds, on 60 answerable held-out test questions: Tuned on dev settings against Phase 3 settings, a paired difference of +0.05 (+0.01 to +0.09).",
    });
  });

  it("leaves the difference out when the report has none, and is null without a value", () => {
    expect(ragHeadline(REPORT_FIXTURE)?.caption).toMatch(/against Phase 3 settings\.$/);
    expect(ragHeadline({ ...REPORT_FIXTURE, retrieval: [] })).toBeNull();
  });
});

describe("askHeadline", () => {
  it("quotes the chosen run's test faithfulness with its interval and n", () => {
    expect(askHeadline(REPORT)).toMatchObject({
      label: "Faithfulness, held-out test answers",
      value: "0.95",
      versus: null,
    });
    expect(askHeadline(REPORT)?.caption).toContain("averaged over 60 answers (95% CI 0.85 to 1.00)");
    expect(askHeadline({ ...REPORT, answers: [] })).toBeNull();
  });
});

describe("stanceHeadline", () => {
  it("quotes the pre-registered zero-shot classifier on the human-labelled statements", () => {
    expect(stanceHeadline(STANCE)).toEqual({
      label: "Accuracy on human labels",
      value: "0.84",
      versus: null,
      caption: "The zero-shot LLM classifier on the 25 statements a person labelled blind (95% CI 0.68 to 0.96).",
    });
  });

  it("is null without a human subset", () => {
    expect(stanceHeadline({ ...STANCE, statement_level: [] })).toBeNull();
  });
});

describe("compareHeadline", () => {
  const data = COMPARE_FIXTURE as unknown as CompareResponse;

  it("reads the stored shift and the size of the diff", () => {
    expect(compareHeadline(data)).toMatchObject({ value: "More dovish", label: "Stance shift, stored LLM reading" });
    expect(compareHeadline(data).caption).toContain("Jul 31, 2024 → Sep 18, 2024");
    expect(compareHeadline(data).caption).toContain("12 of 18 aligned sentences changed");
  });

  it("falls back to the diff when no analysis is stored", () => {
    const bare = { ...data, analysis: null, analysis_status: "not_cached" } as CompareResponse;
    expect(compareHeadline(bare)).toMatchObject({ label: "Sentence diff", value: "12 of 18" });
  });
});

describe("evaluationFacts", () => {
  it("collects the split sizes, human review, judge check and leaks", () => {
    expect(evaluationFacts(REPORT, STANCE)).toEqual({
      devQuestions: 25,
      testQuestions: 60,
      humanReviewed: 20,
      humanStatements: 25,
      judgeModel: "gpt-5.5",
      judgeSample: 40,
      judgeAgreement: { low: 0.875, high: 0.975 },
      judgeIsolated: false,
      leaks: 0,
      leakRuns: 8,
    });
  });

  it("leads with the isolated annotator when there is one", () => {
    const validation = REPORT.judge!.validation!;
    const report: EvaluationReport = {
      ...REPORT,
      judge: {
        ...REPORT.judge!,
        validation: {
          ...validation,
          annotators: [{ id: "C", description: "Isolated", isolated: true }, ...validation.annotators],
          agreement: [
            ...validation.agreement,
            { check: "claim_support", annotator: "C", n: 155, agreement: 0.94, exact_agreement: 0.93, kappa: 0.63 },
            { check: "fact_coverage", annotator: "C", n: 115, agreement: 0.98, exact_agreement: 0.98, kappa: 0.96 },
          ],
        },
      },
    };
    expect(evaluationFacts(report, null)).toMatchObject({
      judgeAgreement: { low: 0.94, high: 0.98 },
      judgeIsolated: true,
      humanStatements: null,
    });
  });

  it("is all unknown without the files", () => {
    expect(evaluationFacts(null, null)).toMatchObject({ devQuestions: null, leaks: null, judgeAgreement: null });
  });
});

describe("formatPercent", () => {
  it("rounds a share to a whole percent", () => {
    expect([0.93548, 0.98261, 1].map(formatPercent)).toEqual(["94%", "98%", "100%"]);
  });
});

describe("evaluationItems", () => {
  it("quotes the live numbers", () => {
    const items = evaluationItems(evaluationFacts(REPORT, STANCE));
    expect(items.map((i) => i.title)).toEqual([
      "Tuned on DEV, scored once on TEST",
      "Checked by a person",
      "An LLM judge, itself checked",
      "No look-ahead",
    ]);
    expect(items[0].body).toMatch(/^25 DEV and 60 TEST questions/);
    expect(items[1].body).toContain("reviewed 20 TEST questions");
    expect(items[1].body).toContain("re-labelled 25 statements");
    expect(items[2].body).toContain("(gpt-5.5)");
    expect(items[2].body).toContain("40 answers, Claude annotators agreed with 88% to 98%");
    expect(items[3].body).toContain("Across 8 evaluation runs, no retrieved passage was released after");
  });

  it("falls back to plain words, never a stale number", () => {
    const items = evaluationItems(null);
    expect(items).toHaveLength(4);
    expect(items.map((i) => i.body).join(" ")).not.toMatch(/\d/);
  });
});
