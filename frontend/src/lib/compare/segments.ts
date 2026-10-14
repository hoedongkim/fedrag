/**
 * Word-level diff segments of a modified sentence pair. Equal and delete segments rebuild the base
 * sentence, equal and insert segments the target; the unified view shows all of them in order.
 */
import type { DiffSegment, SentenceChange } from "../api/types";

export type Side = "base" | "target";

/** The segments that make up one side of the pair, in order. */
export function sideSegments(segments: readonly DiffSegment[], side: Side): DiffSegment[] {
  const hidden = side === "base" ? "insert" : "delete";
  return segments.filter((segment) => segment.op !== hidden && segment.text !== "");
}

/** One side's sentence rebuilt from the segments. */
export function sideText(segments: readonly DiffSegment[], side: Side): string {
  return sideSegments(segments, side)
    .map((segment) => segment.text)
    .join("");
}

/** Segments with empty runs dropped and neighbours of the same kind merged. */
export function normalizeSegments(segments: readonly DiffSegment[]): DiffSegment[] {
  const merged: DiffSegment[] = [];
  for (const segment of segments) {
    if (!segment.text) continue;
    const last = merged[merged.length - 1];
    if (last && last.op === segment.op) last.text += segment.text;
    else merged.push({ ...segment });
  }
  return merged;
}

/**
 * The segments to highlight for a modified change, or null when there are none or they do not
 * rebuild both sentences exactly (the view then shows the two sentences without word marks
 * rather than a misleading redline).
 */
export function usableSegments(change: SentenceChange): DiffSegment[] | null {
  if (change.kind !== "modified" || !change.segments || change.segments.length === 0) return null;
  const segments = normalizeSegments(change.segments);
  return sideText(segments, "base") === (change.base ?? "") &&
    sideText(segments, "target") === (change.target ?? "")
    ? segments
    : null;
}

/** One run to render: `text` goes inside the mark (for insert/delete), `after` outside it. */
export interface RenderRun {
  op: DiffSegment["op"];
  text: string;
  after: string;
}

/**
 * The runs to render for one side, or for `both` (the unified redline). A marked run's trailing
 * whitespace is moved outside its mark, so the wash and the strike end at the word; and where a
 * deletion runs straight into the insertion that replaces it ("objective." then "objective but"),
 * a space is added between the two marks so they do not read as one word.
 */
export function renderRuns(segments: readonly DiffSegment[], side: Side | "both"): RenderRun[] {
  const shown = side === "both" ? segments.filter((s) => s.text !== "") : sideSegments(segments, side);
  return shown.map((segment, i) => {
    if (segment.op === "equal") return { op: "equal", text: segment.text, after: "" };
    const text = segment.text.trimEnd();
    const trailing = segment.text.slice(text.length);
    const next = shown[i + 1];
    const touching =
      !trailing && next !== undefined && next.op !== "equal" && next.op !== segment.op && !/^\s/.test(next.text);
    return { op: segment.op, text, after: touching ? " " : trailing };
  });
}
