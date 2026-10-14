import type { DiffSegment } from "@/lib/api/types";
import { renderRuns, type Side } from "@/lib/compare/segments";

/** Deleted words: red and struck through (the stance review page's redline). */
export const delClass = "text-del-ink line-through decoration-1";
/** Inserted words: green ink on a green wash, underlined. */
export const insClass =
  "rounded-[3px] bg-ins text-ins-ink underline decoration-1 underline-offset-3 box-decoration-clone";

/**
 * A modified sentence as a word-level redline: one side (`base` shows deletions, `target`
 * insertions) or `both` for the unified view. Screen readers hear "deleted:" and "inserted:"
 * before each marked run, since most do not announce <del> and <ins>.
 */
export function SegmentText({ segments, side }: { segments: readonly DiffSegment[]; side: Side | "both" }) {
  return (
    <>
      {renderRuns(segments, side).map((run, i) => {
        if (run.op === "equal") return <span key={i}>{run.text}</span>;
        const Mark = run.op === "delete" ? "del" : "ins";
        return (
          <span key={i}>
            <Mark className={run.op === "delete" ? delClass : insClass}>
              <span className="sr-only">{run.op === "delete" ? "deleted: " : "inserted: "}</span>
              {run.text}
            </Mark>
            {run.after}
          </span>
        );
      })}
    </>
  );
}
