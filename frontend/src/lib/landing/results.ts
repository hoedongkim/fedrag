/**
 * The overview page's headline results, computed from the live API responses (never typed into
 * the page, so they cannot drift from the tracked evaluation files): one per tab, plus the facts
 * behind "How it is evaluated".
 */
import type { CompareResponse, EvaluationReport, ReportSplit, StanceBenchmark } from "../api/types";
import { formatDate } from "../ask/format";
import { SHIFT_LABELS } from "../compare/format";
import { configLabel, formatDelta, formatMetric, methodLabel } from "../performance/format";
import { baselineRun, headlineRun, pairedDifference, reportSplits, rowFor, valueOf } from "../performance/report";

/** The statement pair the Statement Comparison card reads: the first cut after the 2022-23 hikes. */
export const SHOWCASE_PAIR = { base: "2024-07-31", target: "2024-09-18" } as const;

export interface HeadlineResult {
  /** What the number is, in a few words (sentence case). */
  label: string;
  /** The number (or word) itself. */
  value: string;
  /** The baseline next to it, e.g. "vs 0.36 (phase3)"; absent when there is none. */
  versus: string | null;
  /** One or two sentences: what it measures, on what, with its uncertainty. */
  caption: string;
}

const DIGITS = 2;

/** The split to quote: test when the report has it (settings were chosen on dev), else its default. */
function quoteSplit(report: Pick<EvaluationReport, "retrieval">): ReportSplit | null {
  const splits = reportSplits(report);
  if (splits.includes("test")) return "test";
  return splits[0] ?? null;
}

function splitWords(split: ReportSplit): string {
  return split === "test" ? "held-out test" : split === "dev" ? "dev" : "evaluation";
}

/** RAG Performance: the chosen run's nDCG@10 against the setup it replaced, with the paired difference. */
export function ragHeadline(report: EvaluationReport): HeadlineResult | null {
  const headline = headlineRun(report);
  const split = quoteSplit(report);
  if (!headline || !split) return null;
  const { run } = headline;
  const row = rowFor(report.retrieval, split, run);
  const value = valueOf(row?.ndcg_at_10);
  if (!row || value === null) return null;

  const baseline = baselineRun(report, run);
  const baseValue = baseline ? valueOf(rowFor(report.retrieval, split, baseline)?.ndcg_at_10) : null;
  const diff = baseline ? pairedDifference(report.differences, "ndcg_at_10", split, run, baseline) : null;
  const n = row.ndcg_at_10.n ?? row.n_questions;
  const settings = configLabel(report.configs, run.config);

  let caption = `Ranking quality of the passages ${methodLabel(run.method).toLowerCase()} retrieval finds, on ${n} answerable ${splitWords(split)} questions`;
  if (baseline && baseValue !== null) {
    caption += `: ${settings} settings against ${configLabel(report.configs, baseline.config)}`;
    caption += diff ? `, a paired difference of ${formatDelta(diff.difference, diff.ci)}.` : ".";
  } else {
    caption += ".";
  }
  return {
    label: `nDCG@10, ${splitWords(split)} questions`,
    value: formatMetric(value, DIGITS),
    versus: baseline && baseValue !== null ? `vs ${formatMetric(baseValue, DIGITS)}` : null,
    caption,
  };
}

/** Ask the Fed: how much of what the production setup's answers claim their citations support. */
export function askHeadline(report: EvaluationReport): HeadlineResult | null {
  const headline = headlineRun(report);
  const split = quoteSplit(report);
  if (!headline || !split) return null;
  const estimate = rowFor(report.answers, split, headline.run)?.faithfulness;
  const value = valueOf(estimate);
  if (!estimate || value === null) return null;
  const n = estimate.n;
  return {
    label: `Faithfulness, ${splitWords(split)} answers`,
    value: formatMetric(value, DIGITS),
    versus: null,
    caption: `The share of each answer’s claims that an LLM judge found supported by the passages it cites${
      n ? `, averaged over ${n} answers` : ""
    }${estimate.ci ? ` (95% CI ${formatMetric(estimate.ci.low, DIGITS)} to ${formatMetric(estimate.ci.high, DIGITS)})` : ""}.`,
  };
}

/** Policy Stance: the zero-shot LLM's accuracy on the statements a person labelled blind. */
export function stanceHeadline(benchmark: StanceBenchmark): HeadlineResult | null {
  const row =
    benchmark.statement_level.find((r) => r.classifier === "llm_zero_shot" && r.variant === "pre_registered") ??
    benchmark.statement_level.find((r) => r.classifier === "llm_zero_shot");
  const human = row?.human_subset;
  if (!human || !Number.isFinite(human.accuracy) || human.n === 0) return null;
  const ci = human.accuracy_ci;
  return {
    label: "Accuracy on human labels",
    value: formatMetric(human.accuracy, DIGITS),
    versus: null,
    caption: `The zero-shot LLM classifier on the ${human.n} statements a person labelled blind${
      ci ? ` (95% CI ${formatMetric(ci.low, DIGITS)} to ${formatMetric(ci.high, DIGITS)})` : ""
    }.`,
  };
}

/** Statement Comparison: the showcase pair's diff, and its stored LLM reading when there is one. */
export function compareHeadline(data: CompareResponse): HeadlineResult {
  const { unchanged, modified, added, removed } = data.stats;
  const changed = modified + added + removed;
  const total = changed + unchanged;
  const pair = `${formatDate(data.base.meeting_date)} → ${formatDate(data.target.meeting_date)}`;
  const diff = `${changed} of ${total} aligned sentences changed`;
  if (data.analysis_status === "ready" && data.analysis) {
    return {
      label: "Stance shift, stored LLM reading",
      value: SHIFT_LABELS[data.analysis.stance_shift],
      versus: null,
      caption: `${pair}, the first rate cut after the 2022-23 hikes: ${diff}, each key change linked to its sentences.`,
    };
  }
  return {
    label: "Sentence diff",
    value: `${changed} of ${total}`,
    versus: null,
    caption: `Aligned sentences that changed from ${pair}, the first rate cut after the 2022-23 hikes.`,
  };
}

export interface EvaluationFacts {
  devQuestions: number | null;
  testQuestions: number | null;
  /** Test questions the project owner reviewed. */
  humanReviewed: number | null;
  /** Statements a person re-labelled blind for the stance benchmark. */
  humanStatements: number | null;
  judgeModel: string | null;
  /** Answers in the blind sample the judge was checked on. */
  judgeSample: number | null;
  /** Range of the lead annotators' agreement with the judge, by kind of check. */
  judgeAgreement: { low: number; high: number } | null;
  /** True when the lead annotator worked in isolation (saw only a copy of the sample). */
  judgeIsolated: boolean;
  /** Retrieved passages released after their question's as-of date, over every run. */
  leaks: number | null;
  /** Retrieval runs those leaks were counted over. */
  leakRuns: number;
}

/** The numbers "How it is evaluated" quotes; any part may be null when a file lacks it. */
export function evaluationFacts(
  report: EvaluationReport | null,
  benchmark: StanceBenchmark | null,
): EvaluationFacts {
  const split = (name: string) => report?.dataset.splits.find((s) => s.split === name)?.n ?? null;
  const validation = report?.judge?.validation ?? null;
  const isolated = new Set((validation?.annotators ?? []).filter((a) => a.isolated).map((a) => a.id));
  const leadRows = (validation?.agreement ?? []).filter(
    (row) => isolated.size === 0 || (row.annotator !== null && isolated.has(row.annotator)),
  );
  const agreements = leadRows.map((row) => row.agreement).filter(Number.isFinite);

  // Leaks over every run once: the "all" rows when there are any, else each split's rows.
  const rows = report ? report.retrieval.filter((r) => r.split === "all") : [];
  const leakRows = rows.length > 0 ? rows : (report?.retrieval ?? []);

  return {
    devQuestions: split("dev"),
    testQuestions: split("test"),
    humanReviewed: report ? report.dataset.n_human_reviewed : null,
    humanStatements: benchmark?.reference_labels?.n_human_reviewed ?? null,
    judgeModel: report?.judge?.model ?? null,
    judgeSample: validation && validation.status === "complete" ? validation.n_annotated : null,
    judgeAgreement:
      validation?.status === "complete" && agreements.length > 0
        ? { low: Math.min(...agreements), high: Math.max(...agreements) }
        : null,
    judgeIsolated: isolated.size > 0,
    leaks: report ? leakRows.reduce((sum, r) => sum + (r.as_of_leaks ?? 0), 0) : null,
    leakRuns: leakRows.length,
  };
}

/** 0.93548 -> "94%". */
export function formatPercent(share: number): string {
  return `${Math.round(share * 100)}%`;
}

export interface EvaluationItem {
  title: string;
  body: string;
}

/**
 * "How it is evaluated", with the live numbers where the files have them and plain words where
 * they do not (still loading, or unavailable), so the section never shows a stale number.
 */
export function evaluationItems(facts: EvaluationFacts | null): EvaluationItem[] {
  const f = facts;
  const split =
    f?.devQuestions && f.testQuestions
      ? `${f.devQuestions} DEV and ${f.testQuestions} TEST questions`
      : "A DEV and a held-out TEST split of questions";
  const review = [
    f?.humanReviewed ? `The owner reviewed ${f.humanReviewed} TEST questions and their gold passages` : "The owner reviewed a sample of the TEST questions",
    f?.humanStatements
      ? `a person re-labelled ${f.humanStatements} statements blind for the stance benchmark`
      : "a person re-labelled statements blind for the stance benchmark",
  ];
  const judge = f?.judgeModel ? `an LLM judge (${f.judgeModel})` : "an LLM judge";
  const agreement =
    f?.judgeAgreement && f.judgeSample
      ? ` On a blind sample of ${f.judgeSample} answers, ${f.judgeIsolated ? "an isolated Claude annotator" : "Claude annotators"} agreed with ${formatPercent(f.judgeAgreement.low)} to ${formatPercent(f.judgeAgreement.high)} of its verdicts, by kind of check: Claude, not people.`
      : " It was checked against Claude annotators on a blind sample: Claude, not people.";
  const leaks =
    f?.leaks !== null && f?.leaks !== undefined && f.leakRuns > 0
      ? `Across ${f.leakRuns} evaluation runs, ${
          f.leaks === 0
            ? "no retrieved passage was"
            : `${f.leaks} retrieved ${f.leaks === 1 ? "passage was" : "passages were"}`
        } released after its question’s as-of date.`
      : "The evaluation counts retrieved passages released after the question’s as-of date; there must be none.";
  return [
    {
      title: "Tuned on DEV, scored once on TEST",
      body: `${split}, drafted by an LLM from the documents, each with the passages that answer it. Settings were tuned on DEV only; TEST was scored once, and Ask the Fed’s default changed only under a rule fixed before that run. Every number has a bootstrap interval.`,
    },
    {
      title: "Checked by a person",
      body: `${review[0]}, and its fixes were applied before TEST was scored. Separately, ${review[1]}.`,
    },
    {
      title: "An LLM judge, itself checked",
      body: `Answers are graded for faithfulness, citation accuracy and correctness by ${judge}.${agreement}`,
    },
    {
      title: "No look-ahead",
      body: `With an as-of date, retrieval drops every document released after it before ranking. ${leaks}`,
    },
  ];
}
