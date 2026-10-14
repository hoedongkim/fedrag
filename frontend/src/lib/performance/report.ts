/**
 * The evaluation report as the page shows it: rows in display order per split, the best run
 * per column, the headline run, paired differences and the per-type heatmap.
 */
import type {
  AnswerRow,
  EvalDifference,
  JudgeCheck,
  JudgeValidation,
  EvalMetric,
  Estimate,
  EvaluationReport,
  QuestionType,
  ReportSplit,
  RetrievalMethod,
  RetrievalRow,
  RunKey,
  TypeRow,
} from "../api/types";

/** Left to right: from keywords alone to the full pipeline. */
export const METHOD_ORDER: readonly RetrievalMethod[] = ["bm25", "dense", "hybrid", "hybrid_rerank"];
/** Test first: settings were chosen on dev, so test is the number to quote. */
export const SPLIT_ORDER: readonly ReportSplit[] = ["test", "dev", "all"];
export const TYPE_ORDER: readonly QuestionType[] = [
  "statement_fact",
  "minutes_fact",
  "date_specific",
  "point_in_time",
  "multi_doc",
  "paraphrase",
  "unanswerable",
];

export const RETRIEVAL_COLUMNS = [
  "recall_at_5",
  "recall_at_10",
  "mrr_at_10",
  "ndcg_at_10",
  "doc_recall_at_5",
  "doc_recall_at_10",
] as const satisfies readonly EvalMetric[];
export type RetrievalMetric = (typeof RETRIEVAL_COLUMNS)[number];

export const ANSWER_COLUMNS = [
  "faithfulness",
  "citation_accuracy",
  "answer_correctness",
  "refusal_accuracy",
  "false_refusal_rate",
] as const satisfies readonly EvalMetric[];
export type AnswerMetric = (typeof ANSWER_COLUMNS)[number];

/** The method and config Ask the Fed shipped with: the reference every change is measured from. */
export const SHIPPED_METHOD: RetrievalMethod = "hybrid_rerank";

/** "hybrid_rerank:tuned": a run's key in lookups and React lists. */
export function runKey(run: Pick<RunKey, "method" | "config">): string {
  return `${run.method}:${run.config}`;
}

export function sameRun(a: Pick<RunKey, "method" | "config">, b: Pick<RunKey, "method" | "config">) {
  return a.method === b.method && a.config === b.config;
}

/** Parses "method:config" back into a run; null if the method is unknown or the config empty. */
export function parseRunKey(value: string | null | undefined): RunKey | null {
  if (!value) return null;
  const [method, config, ...rest] = value.split(":");
  if (rest.length || !config || !METHOD_ORDER.includes(method as RetrievalMethod)) return null;
  return { method: method as RetrievalMethod, config };
}

/** Splits the report has retrieval rows for, test first. */
export function reportSplits(report: Pick<EvaluationReport, "retrieval">): ReportSplit[] {
  return SPLIT_ORDER.filter((split) => report.retrieval.some((r) => r.split === split));
}

/** The split to show first: test when there is one, else all questions, else whatever exists. */
export function defaultSplit(splits: readonly ReportSplit[]): ReportSplit {
  if (splits.includes("test")) return "test";
  if (splits.includes("all")) return "all";
  return splits[0] ?? "test";
}

/** Rows of one split in display order: by config (report order), then by method. */
export function rowsForSplit<T extends { method: RetrievalMethod; config: string; split: ReportSplit }>(
  rows: readonly T[],
  split: ReportSplit,
  configs: readonly { id: string }[],
): T[] {
  const configIndex = (id: string) => {
    const i = configs.findIndex((c) => c.id === id);
    return i < 0 ? configs.length : i;
  };
  const methodIndex = (m: RetrievalMethod) => {
    const i = METHOD_ORDER.indexOf(m);
    return i < 0 ? METHOD_ORDER.length : i;
  };
  return rows
    .filter((r) => r.split === split)
    .sort((a, b) => configIndex(a.config) - configIndex(b.config) || methodIndex(a.method) - methodIndex(b.method));
}

/** Rows grouped by config, in report order; configs without rows are left out. */
export function groupByConfig<T extends { config: string }, C extends { id: string }>(
  rows: readonly T[],
  configs: readonly C[],
): { config: C; rows: T[] }[] {
  return configs
    .map((config) => ({ config, rows: rows.filter((r) => r.config === config.id) }))
    .filter((group) => group.rows.length > 0);
}

/** The value of an estimate, or null. */
export function valueOf(estimate: Estimate | null | undefined): number | null {
  const v = estimate?.value;
  return v === null || v === undefined || !Number.isFinite(v) ? null : v;
}

/**
 * The run with the best value (highest, or lowest with `lowerIsBetter`); ties go to the first
 * row in display order. Null when no row has a value, or when only one does (nothing to beat).
 */
export function bestRun<T extends { method: RetrievalMethod; config: string }>(
  rows: readonly T[],
  value: (row: T) => number | null,
  lowerIsBetter = false,
): string | null {
  let top: { key: string; v: number } | null = null;
  let counted = 0;
  for (const row of rows) {
    const v = value(row);
    if (v === null) continue;
    counted += 1;
    if (!top || (lowerIsBetter ? v < top.v : v > top.v)) top = { key: runKey(row), v };
  }
  return counted > 1 ? (top?.key ?? null) : null;
}

export interface Headline {
  run: RunKey;
  metric: EvalMetric;
  /** Split the run was chosen on. */
  chosenOn: ReportSplit;
  rule: string;
  /** False when the report named no selection and the page picked the best dev nDCG@10 itself. */
  fromReport: boolean;
}

/**
 * The run to lead with: the report's selection (made on dev before test was scored), else the
 * run with the best nDCG@10 on dev (or on all questions without a dev split).
 */
export function headlineRun(report: Pick<EvaluationReport, "selection" | "retrieval" | "configs">): Headline | null {
  const { selection } = report;
  if (selection) {
    return {
      run: { method: selection.method, config: selection.config },
      metric: selection.metric,
      chosenOn: selection.split,
      rule: selection.rule,
      fromReport: true,
    };
  }
  const splits = reportSplits(report);
  const split: ReportSplit | undefined = splits.includes("dev") ? "dev" : splits.includes("all") ? "all" : splits[0];
  if (!split) return null;
  const rows = rowsForSplit(report.retrieval, split, report.configs);
  const key = bestRun(rows, (r) => valueOf(r.ndcg_at_10)) ?? (rows[0] ? runKey(rows[0]) : null);
  const run = parseRunKey(key);
  if (!run) return null;
  return {
    run,
    metric: "ndcg_at_10",
    chosenOn: split,
    rule: `Highest nDCG@10 on ${split === "dev" ? "the dev split" : "all questions"} (chosen on this page: the report names no selection).`,
    fromReport: false,
  };
}

/**
 * The reference the headline is compared with: the method Ask the Fed shipped with, under the
 * first config (the settings as shipped). Null when it is the headline run itself.
 */
export function baselineRun(report: Pick<EvaluationReport, "configs">, headline: RunKey | null): RunKey | null {
  const config = report.configs[0]?.id;
  if (!config) return null;
  const baseline = { method: SHIPPED_METHOD, config };
  return headline && sameRun(headline, baseline) ? null : baseline;
}

/**
 * The paired difference `a − b` of a metric on a split, from the report (a stored `b − a` is
 * negated, interval included); null when the report does not have it.
 */
export function pairedDifference(
  differences: readonly EvalDifference[],
  metric: EvalMetric,
  split: ReportSplit,
  a: RunKey,
  b: RunKey,
): Pick<EvalDifference, "difference" | "ci" | "n"> | null {
  for (const d of differences) {
    if (d.metric !== metric || d.split !== split) continue;
    if (sameRun(d.minuend, a) && sameRun(d.subtrahend, b)) return d;
    if (sameRun(d.minuend, b) && sameRun(d.subtrahend, a)) {
      return { difference: -d.difference, ci: { low: -d.ci.high, high: -d.ci.low }, n: d.n };
    }
  }
  return null;
}

/** True when a paired interval excludes zero: a difference the bootstrap can tell apart. */
export function isResolved(ci: { low: number; high: number }): boolean {
  return ci.low > 0 || ci.high < 0;
}

/** The report's paired differences on one split and set of metrics, in report order. */
export function differencesFor(
  differences: readonly EvalDifference[],
  split: ReportSplit,
  metrics: readonly EvalMetric[],
): EvalDifference[] {
  return differences.filter((d) => d.split === split && metrics.includes(d.metric));
}

export interface HeatCell {
  type: QuestionType;
  estimate: Estimate | null;
  /** Questions of this type in the split (answerable or not). */
  nQuestions: number;
}

export interface HeatRow {
  method: RetrievalMethod;
  config: string;
  cells: HeatCell[];
}

/**
 * The per-type grid of one metric on one split: a row per method and config (display order) and
 * a column per question type that has a value for the metric in some row (an unanswerable type
 * has no recall, for one).
 */
export function heatmap(
  perType: readonly TypeRow[],
  split: ReportSplit,
  metric: EvalMetric,
  configs: readonly { id: string }[],
): { types: QuestionType[]; rows: HeatRow[] } {
  const rows = perType.filter((r) => r.split === split);
  const types = TYPE_ORDER.filter((type) =>
    rows.some((r) => r.type === type && valueOf(r[metric]) !== null),
  );
  const runs = rowsForSplit(
    uniqueRuns(rows).map((run) => ({ ...run, split })),
    split,
    configs,
  );
  return {
    types,
    rows: runs.map(({ method, config }) => ({
      method,
      config,
      cells: types.map((type) => {
        const row = rows.find((r) => r.method === method && r.config === config && r.type === type);
        return { type, estimate: row?.[metric] ?? null, nQuestions: row?.n_questions ?? 0 };
      }),
    })),
  };
}

function uniqueRuns(rows: readonly { method: RetrievalMethod; config: string }[]): RunKey[] {
  const seen = new Map<string, RunKey>();
  for (const { method, config } of rows) seen.set(runKey({ method, config }), { method, config });
  return [...seen.values()];
}

/** Upper bounds of the heatmap's color classes: five equal bins of [0, 1]. */
export const HEAT_BOUNDS = [0.2, 0.4, 0.6, 0.8] as const;

/** The color class (0-4) of a share in [0, 1]; null for a missing value. */
export function heatBin(value: number | null): number | null {
  if (value === null || !Number.isFinite(value)) return null;
  const bin = HEAT_BOUNDS.findIndex((bound) => value < bound);
  return bin < 0 ? HEAT_BOUNDS.length : bin;
}

/** The interval chart's x domain: [0, 1], since every metric is a share. */
export function toPercent(value: number): number {
  return Math.min(100, Math.max(0, value * 100));
}

/** A run's row on a split, or null. */
export function rowFor<T extends { method: RetrievalMethod; config: string; split: ReportSplit }>(
  rows: readonly T[],
  split: ReportSplit,
  run: RunKey,
): T | null {
  return rows.find((r) => r.split === split && sameRun(r, run)) ?? null;
}

export function isRetrievalMetric(metric: EvalMetric): metric is RetrievalMetric {
  return (RETRIEVAL_COLUMNS as readonly EvalMetric[]).includes(metric);
}

/** A metric's estimate from a run's retrieval or answer row, whichever holds it. */
export function estimateFor(
  metric: EvalMetric,
  retrieval: RetrievalRow | null,
  answers: AnswerRow | null,
): Estimate | null {
  return isRetrievalMetric(metric) ? (retrieval?.[metric] ?? null) : (answers?.[metric] ?? null);
}

export interface AgreementCell {
  agreement: number;
  /** Share with the same three-way label, before collapsing to pass/fail. */
  exact: number | null;
  kappa: number | null;
  n: number;
}

export interface AgreementGrid {
  /** One column per comparison: the judge vs each annotator, then annotator pairs. */
  columns: { key: string; label: string }[];
  rows: { check: JudgeCheck; cells: Record<string, AgreementCell | undefined> }[];
}

const CHECK_ORDER: readonly JudgeCheck[] = ["claim_support", "citation_support", "fact_coverage", "refusal"];

/**
 * The judge-validation agreement as a grid: a row per kind of check, a column for the judge
 * against each annotator (in report order, the main annotator first) and one per pair of
 * annotators. A pair of annotators neither of which worked in isolation is labelled "not
 * independent" when the report marks some annotator isolated, and columns of the same kind with
 * identical cells are shown once: "Judge vs A (= Judge vs B)".
 */
export function agreementGrid(
  validation: Pick<JudgeValidation, "agreement"> &
    Partial<Pick<JudgeValidation, "inter_annotator" | "annotators">>,
): AgreementGrid {
  const isolated = new Set((validation.annotators ?? []).filter((a) => a.isolated).map((a) => a.id));
  const all: AgreementGrid["columns"] = [];
  const cells = new Map<string, AgreementCell>();
  const addColumn = (key: string, label: string) => {
    if (!all.some((c) => c.key === key)) all.push({ key, label });
  };
  for (const row of validation.agreement) {
    const key = `judge:${row.annotator ?? ""}`;
    addColumn(key, row.annotator ? `Judge vs ${row.annotator}` : "Judge vs annotator");
    cells.set(`${row.check}|${key}`, {
      agreement: row.agreement,
      exact: row.exact_agreement,
      kappa: row.kappa,
      n: row.n,
    });
  }
  for (const row of validation.inter_annotator ?? []) {
    const key = `pair:${row.annotators.join(":")}`;
    const independent = isolated.size === 0 || row.annotators.some((id) => isolated.has(id));
    addColumn(key, `${row.annotators.join(" vs ")}${independent ? "" : " (not independent)"}`);
    cells.set(`${row.check}|${key}`, {
      agreement: row.agreement,
      exact: row.exact_agreement,
      kappa: row.kappa,
      n: row.n,
    });
  }
  // Merge a column into an earlier one of the same kind with the same cells.
  const signature = (key: string) =>
    `${key.split(":")[0]}|${JSON.stringify(CHECK_ORDER.map((check) => cells.get(`${check}|${key}`) ?? null))}`;
  const kept: { key: string; label: string; same: string[] }[] = [];
  for (const column of all) {
    const twin = kept.find((k) => signature(k.key) === signature(column.key));
    if (twin) twin.same.push(column.label);
    else kept.push({ ...column, same: [] });
  }
  const columns = kept.map(({ key, label, same }) => ({
    key,
    label: same.length ? `${label} (= ${same.join(", ")})` : label,
  }));
  const rows = CHECK_ORDER.filter((check) => columns.some((c) => cells.has(`${check}|${c.key}`))).map(
    (check) => ({
      check,
      cells: Object.fromEntries(columns.map((c) => [c.key, cells.get(`${check}|${c.key}`)])),
    }),
  );
  return { columns, rows };
}

/**
 * The annotators whose agreement with the judge to lead with: the isolated ones when the report
 * marks any, else every annotator (an empty list for a file that names none).
 */
export function leadAnnotators(validation: Pick<JudgeValidation, "annotators">): string[] {
  const isolated = validation.annotators.filter((a) => a.isolated).map((a) => a.id);
  return isolated.length ? isolated : validation.annotators.map((a) => a.id);
}

/**
 * Runs with judged answers (an answer row on some split) that the judge-validation sample does
 * not cover: the judge was never checked on their answers.
 */
export function uncheckedRuns(
  report: Pick<EvaluationReport, "answers">,
  validation: Pick<JudgeValidation, "sampled_runs">,
): RunKey[] {
  return uniqueRuns(report.answers).filter((run) => !validation.sampled_runs.some((s) => sameRun(s, run)));
}

/** The lowest and highest of some values, or null without any. */
export function valueRange(values: readonly (number | null)[]): [number, number] | null {
  const present = values.filter((v): v is number => v !== null && Number.isFinite(v));
  return present.length ? [Math.min(...present), Math.max(...present)] : null;
}
