/** Display formatting for Policy Stance. Fixed locale and UTC, so server and client render alike. */
import type { BenchmarkReference, ReferenceSource, Stance } from "../api/types";

const MINUS = "−";

export const STANCE_LABELS: Record<Stance, string> = {
  hawkish: "Hawkish",
  neutral: "Neutral",
  dovish: "Dovish",
};

export const REFERENCE_SOURCE_LABELS: Record<ReferenceSource, string> = {
  llm_adjudicated: "LLM-adjudicated",
  human: "Human-reviewed",
};

export const BENCHMARK_REFERENCE_LABELS: Record<BenchmarkReference, string> = {
  llm_adjudicated: "LLM-adjudicated",
  mixed: "LLM-adjudicated + human-reviewed",
  human: "Human-reviewed",
};

/** A signed stance score: 0.42 -> "+0.42", -0.4 -> "−0.40" (a true minus sign), 0 -> "0.00". */
export function formatScore(score: number, digits = 2): string {
  if (!Number.isFinite(score)) return "-";
  const fixed = Math.abs(score).toFixed(digits);
  if (Number(fixed) === 0) return fixed;
  return `${score > 0 ? "+" : MINUS}${fixed}`;
}

/** A metric in [0, 1] (F1, accuracy, kappa) to `digits` decimals; null -> "n/a". */
export function formatMetric(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "n/a";
  return value < 0 ? `${MINUS}${Math.abs(value).toFixed(digits)}` : value.toFixed(digits);
}

/** A confidence interval of a metric: "0.835–0.948"; null -> null. */
export function formatInterval(
  ci: { low: number; high: number } | null | undefined,
  digits = 3,
): string | null {
  if (!ci || !Number.isFinite(ci.low) || !Number.isFinite(ci.high)) return null;
  return `${formatMetric(ci.low, digits)}–${formatMetric(ci.high, digits)}`;
}

/** A signed difference with its interval: "+0.052 (−0.031 to +0.134)". */
export function formatDifference(
  difference: number,
  ci: { low: number; high: number },
  digits = 3,
): string {
  const signed = (v: number) => formatScore(v, digits);
  return `${signed(difference)} (${signed(ci.low)} to ${signed(ci.high)})`;
}

/** A confidence level as a percent: 0.95 -> "95%". */
export function formatLevel(level: number): string {
  return `${Number((level * 100).toFixed(1))}%`;
}

/** Mean ± sample sd over seeds: "0.684 ± 0.011"; without an sd just the mean. */
export function formatMeanSd(mean: number, sd: number | null, digits = 3): string {
  const m = formatMetric(mean, digits);
  return sd === null || !Number.isFinite(sd) ? m : `${m} ± ${sd.toFixed(digits)}`;
}

/** A model-reported confidence: a number as 0.82, a level such as "high" as is. */
export function formatConfidence(confidence: number | string | null): string | null {
  if (confidence === null || confidence === "") return null;
  return typeof confidence === "number" ? confidence.toFixed(2) : confidence;
}

/** A percent with up to two decimals and no trailing zeros: 4.25 -> "4.25", 4.5 -> "4.5", 0 -> "0". */
export function formatRate(value: number): string {
  return String(Number(value.toFixed(2)));
}

/** A target range: (4.25, 4.5) -> "4.25–4.5%". */
export function formatRange(lower: number, upper: number): string {
  return `${formatRate(lower)}–${formatRate(upper)}%`;
}

/** Average test sentences per seed, at most one decimal: 496 -> "496", 482.67 -> "482.7". */
export function formatCount(value: number): string {
  return Number.isInteger(value) ? value.toLocaleString("en-US") : value.toFixed(1);
}

const DATE_TIME = new Intl.DateTimeFormat("en-US", {
  year: "numeric",
  month: "short",
  day: "numeric",
  timeZone: "UTC",
});

/** An ISO timestamp as a UTC calendar date: "Oct 13, 2026"; unparseable input is returned as is. */
export function formatTimestamp(iso: string): string {
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? iso : DATE_TIME.format(date);
}
