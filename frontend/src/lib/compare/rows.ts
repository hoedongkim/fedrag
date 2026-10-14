/**
 * Rows of the sentence diff: every change in document order, with runs of unchanged sentences
 * folded into "N unchanged sentences" expanders. A lone unchanged sentence stays visible, since
 * an expander would hide it without saving any room.
 */
import type { SentenceChange } from "../api/types";

/** Shortest run of unchanged sentences that is folded. */
export const MIN_FOLDED_RUN = 2;

export type DiffBlock =
  /** A change shown as a row (modified, added, removed, or a lone unchanged sentence). */
  | { type: "row"; change: SentenceChange }
  /** A run of unchanged sentences; `runId` is the id of its first sentence. */
  | { type: "run"; runId: number; changes: SentenceChange[]; expanded: boolean };

/** Blocks for the diff view; runs whose id is in `expanded` are open. */
export function buildDiffBlocks(
  changes: readonly SentenceChange[],
  expanded: ReadonlySet<number>,
): DiffBlock[] {
  const blocks: DiffBlock[] = [];
  let run: SentenceChange[] = [];

  const flush = () => {
    if (run.length >= MIN_FOLDED_RUN) {
      blocks.push({ type: "run", runId: run[0].id, changes: run, expanded: expanded.has(run[0].id) });
    } else {
      for (const change of run) blocks.push({ type: "row", change });
    }
    run = [];
  };

  for (const change of changes) {
    if (change.kind === "unchanged") {
      run.push(change);
    } else {
      flush();
      blocks.push({ type: "row", change });
    }
  }
  flush();
  return blocks;
}

/** Ids of every foldable run, for "show all unchanged sentences". */
export function runIds(blocks: readonly DiffBlock[]): number[] {
  return blocks.flatMap((block) => (block.type === "run" ? [block.runId] : []));
}

/** The run that holds change `id` while it is folded, or null when the change is on screen. */
export function foldedRunOf(blocks: readonly DiffBlock[], id: number): number | null {
  for (const block of blocks) {
    if (block.type === "run" && !block.expanded && block.changes.some((c) => c.id === id)) {
      return block.runId;
    }
  }
  return null;
}

/** The changes that differ between the statements (not unchanged), in order. */
export function changedOnly(changes: readonly SentenceChange[]): SentenceChange[] {
  return changes.filter((change) => change.kind !== "unchanged");
}

/** Element id of a change's row: the target of key-change links. */
export function changeAnchor(id: number): string {
  return `change-${id}`;
}

/** The change id in a "#change-N" hash, or null. */
export function changeIdFromHash(hash: string): number | null {
  const match = /^#change-(\d{1,4})$/.exec(hash);
  return match ? Number(match[1]) : null;
}
