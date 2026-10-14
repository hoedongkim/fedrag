/** Table pieces shared by the benchmark tables of Policy Stance and RAG Performance. */
import type { ReactNode } from "react";

import { formatInterval } from "@/lib/stance/format";

export const th = "px-3 py-2 text-left text-label font-medium text-faint";
export const td = "px-3 py-2 align-top tabular-nums whitespace-nowrap";
// The first column stays put while a phone scrolls the table sideways, so every row keeps its
// name; its background is the page's (panels share it), with a hairline to mark the edge. -left-1: sticky offsets
// count from inside TableScroller's 4px padding, so this puts the cell flush with its edge and
// no scrolled cell shows beside it.
export const stickyCell = "sticky -left-1 z-10 bg-canvas bg-clip-padding shadow-[1px_0_0_var(--line)]";
export const rowCell = `${stickyCell} px-3 py-2 text-left align-top font-medium whitespace-nowrap text-ink`;
export const groupCell = "px-3 pt-4 pb-2 text-left text-label font-medium text-faint";

export function TableScroller({ label, children }: { label: string; children: ReactNode }) {
  return (
    // Scrolls sideways on its own on a phone, never the page; focusable so keys can scroll it.
    // The padding leaves room for the focus ring of the cells at its edges (a scroller clips it).
    // relative: the sr-only text in its cells is absolutely positioned and must be clipped here.
    <div role="region" aria-label={label} tabIndex={0} className="relative mt-4 -mx-1 overflow-x-auto px-1 pb-1">
      {children}
    </div>
  );
}

/** A row-group label; the text sticks to the left edge while the table scrolls sideways. */
export function GroupRow({ colSpan, children }: { colSpan: number; children: ReactNode }) {
  return (
    <tr>
      <th scope="rowgroup" colSpan={colSpan} className={groupCell}>
        <span className="sticky left-3 inline-block">{children}</span>
      </th>
    </tr>
  );
}

/** A metric, bold when it is the column's best, with its interval in small type below. */
export function MetricCell({
  best,
  ci,
  digits = 3,
  note,
  children,
}: {
  best: boolean;
  ci?: { low: number; high: number } | null;
  digits?: number;
  /** A short line under the value and interval, e.g. "n = 10". */
  note?: string;
  children: ReactNode;
}) {
  const interval = formatInterval(ci, digits);
  return (
    <td className={`${td} ${best ? "font-semibold text-ink" : ""}`}>
      {children}
      {best && <span className="sr-only"> (best)</span>}
      {interval && (
        <span className="block text-label font-normal text-faint">
          <span className="sr-only">interval </span>
          {interval}
        </span>
      )}
      {note && (
        <span className="block text-label font-normal text-faint">{note}</span>
      )}
    </td>
  );
}
