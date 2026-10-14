/** Benchmark tables: rows per classifier in display order, and the best score per column. */
import type {
  BaselineDifference,
  BenchmarkSplit,
  PairedDifference,
  SentenceBaseline,
  SentenceBenchmark,
  StanceBenchmark,
  StanceClassifier,
  StatementBaseline,
  StatementBaselineName,
  StatementBenchmark,
  StatementBootstrap,
  StatementVariant,
} from "../api/types";
import { CLASSIFIER_IDS } from "./classifiers";
import { STANCE_LABELS } from "./format";

/** Pre-registered rows first; post-hoc rows are shown apart and never compete for "best". */
export const STATEMENT_VARIANTS: readonly StatementVariant[] = [
  "pre_registered",
  "post_hoc_rubric_ordered",
];

export interface SentenceTableRow {
  classifier: StanceClassifier;
  official: SentenceBenchmark | null;
  deduplicated: SentenceBenchmark | null;
}

/** One row per benchmarked classifier, in display order, with both test splits. */
export function sentenceTable(rows: SentenceBenchmark[]): SentenceTableRow[] {
  const pick = (classifier: StanceClassifier, split: BenchmarkSplit) =>
    rows.find((r) => r.classifier === classifier && r.split === split) ?? null;
  return CLASSIFIER_IDS.filter((id) => rows.some((r) => r.classifier === id)).map((classifier) => ({
    classifier,
    official: pick(classifier, "official"),
    deduplicated: pick(classifier, "deduplicated"),
  }));
}

/** Statement rows in display order: pre-registered rows, then post-hoc ones, each by classifier. */
export function statementTable(rows: StatementBenchmark[]): StatementBenchmark[] {
  return STATEMENT_VARIANTS.flatMap((variant) =>
    CLASSIFIER_IDS.flatMap((id) => rows.filter((r) => r.classifier === id && r.variant === variant)),
  );
}

/** True for a row whose aggregation was designed after the pre-registered results were seen. */
export function isPostHoc(row: Pick<StatementBenchmark, "variant">): boolean {
  return row.variant !== "pre_registered";
}

/** A unique key per statement row (a classifier can have one row per variant). */
export function statementRowKey(row: Pick<StatementBenchmark, "classifier" | "variant">): string {
  return `${row.classifier}:${row.variant}`;
}

export interface PairedComparison {
  minuend: PairedDifference["minuend"];
  subtrahend: PairedDifference["subtrahend"];
  all: PairedDifference | null;
  humanSubset: PairedDifference | null;
}

/** The paired bootstrap differences grouped by compared pair, in file order. */
export function pairedComparisons(bootstrap: StatementBootstrap | null): PairedComparison[] {
  const pairs = new Map<string, PairedComparison>();
  for (const diff of bootstrap?.paired_differences ?? []) {
    const key = `${statementRowKey(diff.minuend)}-${statementRowKey(diff.subtrahend)}`;
    const pair = pairs.get(key) ?? {
      minuend: diff.minuend,
      subtrahend: diff.subtrahend,
      all: null,
      humanSubset: null,
    };
    if (diff.subset === "all") pair.all = diff;
    else pair.humanSubset = diff;
    pairs.set(key, pair);
  }
  return [...pairs.values()];
}

/** Baselines that use no classifier: the floor (majority class) and the post-hoc control. */
export const BASELINE_LABELS: Record<StatementBaselineName, string> = {
  majority_class: "Majority class",
  rate_move_rule: "Rate-move rule",
};

/** What a baseline does, in a few words: "always hawkish", "decision verb only". */
export function baselineDetail(baseline: Pick<StatementBaseline, "name" | "label">): string {
  if (baseline.label) return `always ${STANCE_LABELS[baseline.label].toLowerCase()}`;
  return baseline.name === "rate_move_rule" ? "decision verb only" : "no classifier";
}

export interface SentenceBaselineRow {
  name: SentenceBaseline["name"];
  label: SentenceBaseline["label"];
  official: SentenceBaseline | null;
  deduplicated: SentenceBaseline | null;
}

/** One row per sentence baseline, in file order, with both test splits. */
export function sentenceBaselineTable(rows: SentenceBaseline[]): SentenceBaselineRow[] {
  const names = [...new Set(rows.map((r) => r.name))];
  return names.map((name) => {
    const pick = (split: BenchmarkSplit) =>
      rows.find((r) => r.name === name && r.split === split) ?? null;
    const official = pick("official");
    const deduplicated = pick("deduplicated");
    return { name, label: (official ?? deduplicated)?.label ?? null, official, deduplicated };
  });
}

export interface BaselineComparison {
  row: BaselineDifference["row"];
  baseline: StatementBaselineName;
  all: BaselineDifference | null;
  humanSubset: BaselineDifference | null;
}

/** The paired differences of a row minus a baseline, grouped by pair, in file order. */
export function baselineComparisons(bootstrap: StatementBootstrap | null): BaselineComparison[] {
  const pairs = new Map<string, BaselineComparison>();
  for (const diff of bootstrap?.baseline_differences ?? []) {
    const key = `${statementRowKey(diff.row)}-${diff.baseline}`;
    const pair = pairs.get(key) ?? { row: diff.row, baseline: diff.baseline, all: null, humanSubset: null };
    if (diff.subset === "all") pair.all = diff;
    else pair.humanSubset = diff;
    pairs.set(key, pair);
  }
  return [...pairs.values()];
}

/** The classifier with the highest value (ties: the first in display order); null if none. */
export function best<T extends { classifier: StanceClassifier }>(
  rows: T[],
  value: (row: T) => number | null | undefined,
): StanceClassifier | null {
  let top: { classifier: StanceClassifier; v: number } | null = null;
  for (const row of rows) {
    const v = value(row);
    if (v === null || v === undefined || !Number.isFinite(v)) continue;
    if (!top || v > top.v) top = { classifier: row.classifier, v };
  }
  return top?.classifier ?? null;
}

export type ReviewStatus =
  | { kind: "none" }
  | { kind: "pending"; nLabels: number }
  | { kind: "reviewed"; nLabels: number; nHuman: number };

/** Whether the statement reference labels have been human-reviewed yet (as they stand now). */
export function reviewStatus(benchmark: Pick<StanceBenchmark, "reference_labels">): ReviewStatus {
  const summary = benchmark.reference_labels;
  if (!summary || summary.n_labels === 0) return { kind: "none" };
  if (summary.n_human_reviewed === 0) return { kind: "pending", nLabels: summary.n_labels };
  return { kind: "reviewed", nLabels: summary.n_labels, nHuman: summary.n_human_reviewed };
}
