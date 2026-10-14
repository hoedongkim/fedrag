/**
 * Turning answer text into renderable pieces: paragraphs, and within them plain text and [n]
 * citation markers. The server sends markers already renumbered to final citation numbers and
 * never splits one across deltas; the trailing-fragment guard below is only a safety net.
 */

export type AnswerSegment = { kind: "text"; text: string } | { kind: "cite"; n: number };

/** Plain text, or a run of adjacent markers with the punctuation right after it ("[1][2]."). */
export type AnswerPiece = { kind: "text"; text: string } | { kind: "cites"; ns: number[]; after: string };

const MARKER = /\[(\d{1,2})\]/g;
/**
 * An unfinished marker at the very end of streamed text: "[", "[1", "[12" (one or two digits, as
 * the backend's markers).
 */
const PARTIAL_MARKER_AT_END = /\[\d{0,2}$/;

/** Paragraphs separated by blank lines; single newlines stay inside their paragraph. */
export function splitParagraphs(text: string): string[] {
  return text
    .split(/\n[ \t]*\n/)
    .map((paragraph) => paragraph.trim())
    .filter(Boolean);
}

/**
 * Plain-text and marker segments of one paragraph. Markers with n = 0 stay text. With
 * `streaming`, an unfinished marker at the end is held back until the next delta completes it.
 */
export function splitCitationMarkers(text: string, { streaming = false } = {}): AnswerSegment[] {
  const source = streaming ? text.replace(PARTIAL_MARKER_AT_END, "") : text;
  const segments: AnswerSegment[] = [];
  let last = 0;
  for (const match of source.matchAll(MARKER)) {
    const n = Number(match[1]);
    if (n < 1) continue;
    const start = match.index;
    if (start > last) segments.push({ kind: "text", text: source.slice(last, start) });
    segments.push({ kind: "cite", n });
    last = start + match[0].length;
  }
  if (last < source.length) segments.push({ kind: "text", text: source.slice(last) });
  return segments;
}

/** Closing punctuation that belongs on the line of the markers before it. */
const TRAILING_PUNCTUATION = /^[.,;:!?)\]"'\u2019\u201d]+/;

/**
 * Segments with each run of adjacent markers gathered, together with the punctuation right after
 * it, so that an answer is never wrapped between two markers or between a marker and its period.
 */
export function groupCitationRuns(segments: readonly AnswerSegment[]): AnswerPiece[] {
  const pieces: AnswerPiece[] = [];
  for (const segment of segments) {
    const previous = pieces.at(-1);
    if (segment.kind === "cite") {
      if (previous?.kind === "cites" && previous.after === "") previous.ns.push(segment.n);
      else pieces.push({ kind: "cites", ns: [segment.n], after: "" });
      continue;
    }
    let text = segment.text;
    if (previous?.kind === "cites") {
      previous.after = TRAILING_PUNCTUATION.exec(text)?.[0] ?? "";
      text = text.slice(previous.after.length);
    }
    if (text) pieces.push({ kind: "text", text });
  }
  return pieces;
}

/** At most `max` characters of `text` on one line, cut at a word boundary with an ellipsis. */
export function excerpt(text: string, max = 200): string {
  const flat = text.replace(/\s+/g, " ").trim();
  if (flat.length <= max) return flat;
  const cut = flat.slice(0, max - 1);
  const space = cut.lastIndexOf(" ");
  const head = space > max * 0.6 ? cut.slice(0, space) : cut;
  return `${head.replace(/[\s,;:.]+$/, "")}…`;
}
