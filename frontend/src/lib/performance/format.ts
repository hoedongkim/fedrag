/** Display names and formatting for RAG Performance. Fixed locale, so server and client agree. */
import type {
  ChunkRole,
  CostSummary,
  EvalMetric,
  JudgeCheck,
  QuestionType,
  ReportSplit,
  RetrievalMethod,
} from "../api/types";
import { formatLatency } from "../ask/format";
import { methodLabel } from "../ask/params";
import { formatInterval, formatLevel, formatMetric, formatScore, formatTimestamp } from "../stance/format";

export { formatInterval, formatLatency, formatLevel, formatMetric, formatTimestamp, methodLabel };

/** Metrics are shares in [0, 1]: two decimals, since one question moves them by more than 0.01. */
export const METRIC_DIGITS = 2;

export const SPLIT_LABELS: Record<ReportSplit, string> = {
  test: "Test",
  dev: "Dev",
  all: "All questions",
};

export const TYPE_LABELS: Record<QuestionType, string> = {
  statement_fact: "Statement fact",
  minutes_fact: "Minutes fact",
  point_in_time: "Point in time",
  date_specific: "Date-specific",
  multi_doc: "Multi-document",
  paraphrase: "Paraphrase",
  unanswerable: "Unanswerable",
};

export const TYPE_DESCRIPTIONS: Record<QuestionType, string> = {
  statement_fact: "A fact stated in one FOMC statement.",
  minutes_fact: "A fact from one set of minutes.",
  point_in_time: "Asked with an as-of date: only what was public then counts.",
  date_specific: "About one meeting, named by its date.",
  multi_doc: "Needs passages from more than one document.",
  paraphrase: "Worded differently from the documents, so keywords do not match.",
  unanswerable: "The corpus cannot answer it: the right reply declines.",
};

export interface MetricInfo {
  key: EvalMetric;
  label: string;
  /** What one point of it means, in a sentence. */
  description: string;
  /** False refusals: a lower share is better. */
  lowerIsBetter?: boolean;
}

export const METRICS: Record<EvalMetric, MetricInfo> = {
  recall_at_5: {
    key: "recall_at_5",
    label: "Recall@5",
    description: "Share of a question’s gold passages in the top 5 (what the answer sees).",
  },
  recall_at_10: {
    key: "recall_at_10",
    label: "Recall@10",
    description: "Share of a question’s gold passages in the top 10.",
  },
  mrr_at_10: {
    key: "mrr_at_10",
    label: "MRR@10",
    description: "1 / rank of the first gold passage, 0 if none is in the top 10.",
  },
  ndcg_at_10: {
    key: "ndcg_at_10",
    label: "nDCG@10",
    description: "Ranking quality with graded gold: primary passages count 2, supporting 1.",
  },
  doc_recall_at_5: {
    key: "doc_recall_at_5",
    label: "Doc recall@5",
    description: "Share of the relevant documents with a passage in the top 5.",
  },
  doc_recall_at_10: {
    key: "doc_recall_at_10",
    label: "Doc recall@10",
    description: "Share of the relevant documents with a passage in the top 10.",
  },
  faithfulness: {
    key: "faithfulness",
    label: "Faithfulness",
    description: "Share of an answer’s claims that the passages it cites support.",
  },
  citation_accuracy: {
    key: "citation_accuracy",
    label: "Citation accuracy",
    description: "Share of citations whose passage supports the sentence they are attached to.",
  },
  answer_correctness: {
    key: "answer_correctness",
    label: "Correctness",
    description: "Share of a question’s required facts the answer states.",
  },
  refusal_accuracy: {
    key: "refusal_accuracy",
    label: "Refusal accuracy",
    description: "Share of unanswerable questions declined without making something up.",
  },
  false_refusal_rate: {
    key: "false_refusal_rate",
    label: "False refusals",
    description: "Share of answerable questions declined anyway (lower is better).",
    lowerIsBetter: true,
  },
};

export const ROLE_LABELS: Record<ChunkRole, string> = {
  primary: "Primary gold",
  supporting: "Supporting gold",
};

export const JUDGE_CHECK_LABELS: Record<JudgeCheck, string> = {
  claim_support: "Claim supported",
  citation_support: "Citation supports its sentence",
  fact_coverage: "Required fact stated",
  refusal: "Declines",
};

/** A config's display name from the report, falling back to its id. */
export function configLabel(configs: readonly { id: string; label: string }[], id: string): string {
  return configs.find((c) => c.id === id)?.label ?? id;
}

/** "Hybrid + rerank · Tuned on dev". */
export function runLabel(
  configs: readonly { id: string; label: string }[],
  run: { method: RetrievalMethod; config: string },
): string {
  return `${methodLabel(run.method)} · ${configLabel(configs, run.config)}`;
}

/**
 * A difference of two shares with its interval: "+0.05 (−0.02 to +0.12)". An interval that
 * excludes 0 with a bound that would round to 0.00 gets three decimals, so a resolved difference
 * never shows a bound of 0: "+0.074 (+0.005 to +0.154)".
 */
export function formatDelta(difference: number, ci?: { low: number; high: number } | null): string {
  const resolved = ci && (ci.low > 0 || ci.high < 0);
  const nearZero = (v: number) => Math.abs(v) < 0.005;
  const digits = resolved && (nearZero(ci.low) || nearZero(ci.high)) ? METRIC_DIGITS + 1 : METRIC_DIGITS;
  const signed = (v: number) => formatScore(v, digits);
  return ci ? `${signed(difference)} (${signed(ci.low)} to ${signed(ci.high)})` : signed(difference);
}

/** USD with enough digits to show sub-cent costs: 0.00123 -> "$0.0012", 0.91 -> "$0.91". */
export function formatUsd(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "n/a";
  if (value === 0) return "$0";
  if (value >= 0.1) return `$${value.toFixed(2)}`;
  return `$${Number(value.toPrecision(2))}`;
}

/** A latency that may be missing. */
export function formatMs(value: number | null | undefined): string {
  return value === null || value === undefined ? "n/a" : formatLatency(value);
}

/** 1632000 -> "1.63M", 176800 -> "177k", 900 -> "900". */
export function formatTokens(value: number): string {
  if (value >= 1e6) return `${Number((value / 1e6).toFixed(2))}M`;
  if (value >= 1e3) return `${Math.round(value / 1e3)}k`;
  return String(value);
}

/** "$0.91 of a $1.00 cap" or "$0.91". */
export function formatSpend(cost: Pick<CostSummary, "total_usd" | "max_usd">): string {
  const total = formatUsd(cost.total_usd);
  return cost.max_usd ? `${total} (cap ${formatUsd(cost.max_usd)} per command)` : total;
}

/** "n = 60" for a count that may be missing. */
export function formatN(n: number | null | undefined): string {
  return n === null || n === undefined ? "" : `n = ${n}`;
}

/**
 * Each retriever config owns a fixed categorical slot (--series-1..3 in globals.css, the
 * validated palette's order) and a mark shape, so identity never rests on color alone. Configs
 * past the third share a neutral ink rather than a generated hue.
 */
const CONFIG_COLORS = ["var(--series-1)", "var(--series-2)", "var(--series-3)"] as const;
export type ConfigShape = "circle" | "diamond" | "square";
const CONFIG_SHAPES: readonly ConfigShape[] = ["circle", "diamond", "square"];

export interface ConfigStyle {
  color: string;
  shape: ConfigShape;
}

export function configStyle(configs: readonly { id: string }[], id: string): ConfigStyle {
  const index = configs.findIndex((c) => c.id === id);
  if (index < 0 || index >= CONFIG_COLORS.length) return { color: "var(--chart-ink)", shape: "square" };
  return { color: CONFIG_COLORS[index], shape: CONFIG_SHAPES[index] };
}
