import { describe, expect, it } from "vitest";

import type { ChangeKind, SentenceChange } from "../api/types";
import { COMPARE_FIXTURE } from "./fixtures";
import {
  buildDiffBlocks,
  changeAnchor,
  changedOnly,
  changeIdFromHash,
  foldedRunOf,
  MIN_FOLDED_RUN,
  runIds,
} from "./rows";

function changes(kinds: ChangeKind[]): SentenceChange[] {
  return kinds.map((kind, id) => ({
    id,
    kind,
    base: kind === "added" ? null : `Base ${id}.`,
    target: kind === "removed" ? null : `Target ${id}.`,
    similarity: kind === "unchanged" ? 1 : kind === "modified" ? 0.8 : null,
    procedural: false,
    segments: null,
  }));
}

const summary = (blocks: ReturnType<typeof buildDiffBlocks>) =>
  blocks.map((b) => (b.type === "row" ? `${b.change.kind[0]}${b.change.id}` : `run${b.runId}x${b.changes.length}${b.expanded ? "+" : ""}`));

describe("buildDiffBlocks", () => {
  it("folds runs of unchanged sentences and keeps every change in order", () => {
    const blocks = buildDiffBlocks(changes(["unchanged", "unchanged", "unchanged", "modified", "added", "unchanged", "removed", "unchanged", "unchanged"]), new Set());
    expect(summary(blocks)).toEqual(["run0x3", "m3", "a4", "u5", "r6", "run7x2"]);
  });

  it(`shows a lone unchanged sentence (runs shorter than ${MIN_FOLDED_RUN} are not folded)`, () => {
    expect(summary(buildDiffBlocks(changes(["modified", "unchanged", "modified"]), new Set()))).toEqual(["m0", "u1", "m2"]);
  });

  it("opens the runs whose id is expanded", () => {
    const blocks = buildDiffBlocks(changes(["unchanged", "unchanged", "modified", "unchanged", "unchanged"]), new Set([3]));
    expect(summary(blocks)).toEqual(["run0x2", "m2", "run3x2+"]);
  });

  it("handles an empty diff and a diff with no changes", () => {
    expect(buildDiffBlocks([], new Set())).toEqual([]);
    expect(summary(buildDiffBlocks(changes(["unchanged", "unchanged"]), new Set()))).toEqual(["run0x2"]);
  });

  it("accounts for every change of the fixture exactly once", () => {
    const blocks = buildDiffBlocks(COMPARE_FIXTURE.changes, new Set());
    const ids = blocks.flatMap((b) => (b.type === "row" ? [b.change.id] : b.changes.map((c) => c.id)));
    expect(ids).toEqual(COMPARE_FIXTURE.changes.map((c) => c.id));
    expect(summary(blocks)).toEqual([
      "u0", "m1", "r2", "m3", "u4", "m5", "u6", "m7", "m8", "r9", "m10", "m11", "run12x3", "m15", "r16", "a17",
    ]);
  });
});

describe("run helpers", () => {
  const list = changes(["unchanged", "unchanged", "modified", "unchanged", "unchanged", "unchanged"]);

  it("lists the foldable runs", () => {
    expect(runIds(buildDiffBlocks(list, new Set()))).toEqual([0, 3]);
  });

  it("finds the folded run that hides a change, and nothing once it is open", () => {
    expect(foldedRunOf(buildDiffBlocks(list, new Set()), 4)).toBe(3);
    expect(foldedRunOf(buildDiffBlocks(list, new Set([3])), 4)).toBeNull();
    expect(foldedRunOf(buildDiffBlocks(list, new Set()), 2)).toBeNull();
  });

  it("keeps only the changes that differ", () => {
    expect(changedOnly(list).map((c) => c.id)).toEqual([2]);
  });
});

describe("anchors", () => {
  it("round-trips a change anchor through the URL hash", () => {
    expect(changeAnchor(7)).toBe("change-7");
    expect(changeIdFromHash(`#${changeAnchor(7)}`)).toBe(7);
    expect(changeIdFromHash("#change-")).toBeNull();
    expect(changeIdFromHash("#main")).toBeNull();
  });
});
