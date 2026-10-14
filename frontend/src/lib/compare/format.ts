/** Display text for Statement Comparison. */
import type { ChangeKind, CompareStats, StanceShift } from "../api/types";

export const SHIFT_LABELS: Record<StanceShift, string> = {
  more_hawkish: "More hawkish",
  unchanged: "Unchanged",
  more_dovish: "More dovish",
};

export const KIND_LABELS: Record<ChangeKind, string> = {
  modified: "Modified",
  added: "Added",
  removed: "Removed",
  unchanged: "Unchanged",
};

/** A similarity in [0, 1] as a whole percent: 0.873 -> "87%"; anything else -> "n/a". */
export function formatSimilarity(value: number | null): string {
  if (value === null || !Number.isFinite(value) || value < 0 || value > 1) return "n/a";
  return `${Math.round(value * 100)}%`;
}

/** "1 sentence", "3 sentences". */
export function sentences(count: number): string {
  return `${count} ${count === 1 ? "sentence" : "sentences"}`;
}

/** One line for screen readers and the live region: "2 modified, 1 added, 0 removed; 87% similar." */
export function describeStats(stats: CompareStats): string {
  return `${stats.modified} modified, ${stats.added} added, ${stats.removed} removed, ${stats.unchanged} unchanged; ${formatSimilarity(stats.similarity)} similar overall.`;
}

/** Change ids as written next to rows and in key-change links: [3, 4] -> "#3, #4". */
export function formatChangeIds(ids: readonly number[]): string {
  return ids.map((id) => `#${id}`).join(", ");
}
