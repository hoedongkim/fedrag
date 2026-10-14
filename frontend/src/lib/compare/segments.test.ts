import { describe, expect, it } from "vitest";

import type { DiffSegment, SentenceChange } from "../api/types";
import { COMPARE_FIXTURE } from "./fixtures";
import { normalizeSegments, renderRuns, sideSegments, sideText, usableSegments } from "./segments";

const SEGMENTS: DiffSegment[] = [
  { op: "equal", text: "Job gains have " },
  { op: "delete", text: "moderated, " },
  { op: "insert", text: "slowed, " },
  { op: "equal", text: "and the unemployment rate has moved up." },
];

const modified = (segments: DiffSegment[] | null): SentenceChange => ({
  id: 1,
  kind: "modified",
  base: "Job gains have moderated, and the unemployment rate has moved up.",
  target: "Job gains have slowed, and the unemployment rate has moved up.",
  similarity: 0.9,
  procedural: false,
  segments,
});

describe("sides of a word diff", () => {
  it("shows equal + deleted words on the base side and equal + inserted on the target", () => {
    expect(sideSegments(SEGMENTS, "base").map((s) => s.op)).toEqual(["equal", "delete", "equal"]);
    expect(sideSegments(SEGMENTS, "target").map((s) => s.op)).toEqual(["equal", "insert", "equal"]);
  });

  it("rebuilds both sentences exactly", () => {
    expect(sideText(SEGMENTS, "base")).toBe(modified(null).base);
    expect(sideText(SEGMENTS, "target")).toBe(modified(null).target);
  });

  it("rebuilds every modified sentence of the fixture", () => {
    const pairs = COMPARE_FIXTURE.changes.filter((c) => c.kind === "modified");
    expect(pairs.length).toBeGreaterThan(0);
    for (const change of pairs) {
      expect(sideText(change.segments ?? [], "base")).toBe(change.base);
      expect(sideText(change.segments ?? [], "target")).toBe(change.target);
    }
  });
});

describe("normalizeSegments", () => {
  it("drops empty runs and merges neighbours of the same kind", () => {
    expect(
      normalizeSegments([
        { op: "equal", text: "A " },
        { op: "equal", text: "B " },
        { op: "insert", text: "" },
        { op: "delete", text: "C" },
      ]),
    ).toEqual([
      { op: "equal", text: "A B " },
      { op: "delete", text: "C" },
    ]);
  });

  it("does not modify its input", () => {
    const input: DiffSegment[] = [
      { op: "equal", text: "A " },
      { op: "equal", text: "B" },
    ];
    normalizeSegments(input);
    expect(input[0].text).toBe("A ");
  });
});

describe("usableSegments", () => {
  it("returns the segments of a modified pair that rebuild both sentences", () => {
    expect(usableSegments(modified(SEGMENTS))).toEqual(SEGMENTS);
  });

  it("refuses segments that do not match the sentences, or a change that is not modified", () => {
    expect(usableSegments(modified([{ op: "equal", text: "Something else." }]))).toBeNull();
    expect(usableSegments(modified(null))).toBeNull();
    expect(usableSegments(modified([]))).toBeNull();
    expect(usableSegments({ ...modified(SEGMENTS), kind: "unchanged" })).toBeNull();
  });
});

describe("renderRuns", () => {
  const joined = (runs: ReturnType<typeof renderRuns>) => runs.map((r) => r.text + r.after).join("");

  it("moves a marked run's trailing space outside the mark", () => {
    expect(renderRuns(SEGMENTS, "base")).toEqual([
      { op: "equal", text: "Job gains have ", after: "" },
      { op: "delete", text: "moderated,", after: " " },
      { op: "equal", text: "and the unemployment rate has moved up.", after: "" },
    ]);
  });

  it("still rebuilds each side's sentence exactly", () => {
    for (const change of COMPARE_FIXTURE.changes.filter((c) => c.kind === "modified")) {
      expect(joined(renderRuns(change.segments ?? [], "base"))).toBe(change.base);
      expect(joined(renderRuns(change.segments ?? [], "target"))).toBe(change.target);
    }
  });

  it("keeps a space between a deletion and the insertion it runs into in the unified view", () => {
    const runs = renderRuns(
      [
        { op: "equal", text: "toward the 2 percent " },
        { op: "delete", text: "inflation objective." },
        { op: "insert", text: "objective but remains elevated." },
      ],
      "both",
    );
    expect(runs.map((r) => [r.op, r.text, r.after])).toEqual([
      ["equal", "toward the 2 percent ", ""],
      ["delete", "inflation objective.", " "],
      ["insert", "objective but remains elevated.", ""],
    ]);
  });

  it("adds nothing where the text already has a space, and shows every segment in both", () => {
    const runs = renderRuns(SEGMENTS, "both");
    expect(runs.map((r) => r.op)).toEqual(["equal", "delete", "insert", "equal"]);
    expect(joined(runs)).toBe(SEGMENTS.map((s) => s.text).join(""));
  });
});
