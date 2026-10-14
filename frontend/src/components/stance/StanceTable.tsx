"use client";

import { useId, useState } from "react";

import type { FedFundsTarget, StanceClassifier } from "@/lib/api/types";
import { formatDate } from "@/lib/ask/format";
import { classifierInfo } from "@/lib/stance/classifiers";
import {
  formatRange,
  formatScore,
  REFERENCE_SOURCE_LABELS,
  STANCE_LABELS,
} from "@/lib/stance/format";
import { type MeetingRow, rangeAfterMeeting } from "@/lib/stance/series";

import { textButtonClass } from "../ui/styles";

interface StanceTableProps {
  rows: MeetingRow[];
  classifiers: StanceClassifier[];
  fedFunds: FedFundsTarget[];
}

/**
 * The chart's data as a table: always in the accessibility tree (the chart itself is hidden from
 * screen readers), and shown on screen with the toggle.
 */
export function StanceTable({ rows, classifiers, fedFunds }: StanceTableProps) {
  const [open, setOpen] = useState(false);
  const id = useId();
  const infos = classifiers.map(classifierInfo);

  return (
    <div className="mt-5">
      <button
        type="button"
        aria-expanded={open}
        aria-controls={id}
        onClick={() => setOpen((v) => !v)}
        className={textButtonClass}
      >
        {open ? "Hide data table" : "Show data table"}
      </button>
      <div
        id={id}
        role={open ? "region" : undefined}
        aria-label={open ? "Stance by meeting, as a table" : undefined}
        tabIndex={open ? 0 : undefined}
        className={
          open
            ? "relative mt-3 max-h-96 overflow-auto border-y border-line"
            : "sr-only"
        }
      >
        <table className="w-full min-w-[36rem] border-collapse text-left text-small">
          <caption className="sr-only">
            Stance score of each FOMC statement (+1 most hawkish, −1 most dovish) by classifier,
            with the reference label and the federal funds target range the meeting set.{" "}
            {rows.length} meetings.
          </caption>
          {/* z-20: above the sticky row headers while the table scrolls down. */}
          <thead className="sticky top-0 z-20 bg-canvas text-label text-faint shadow-[0_1px_0_var(--line)]">
            <tr>
              <th scope="col" className="sticky left-0 z-10 bg-canvas px-3 py-2 font-medium">
                Meeting
              </th>
              {infos.map((info) => (
                <th key={info.id} scope="col" className="px-3 py-2 font-medium">
                  {info.label}
                </th>
              ))}
              <th scope="col" className="px-3 py-2 font-medium">
                Reference
              </th>
              <th scope="col" className="px-3 py-2 font-medium">
                Target range
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-line text-muted">
            {[...rows].reverse().map((row) => {
              const range = rangeAfterMeeting(fedFunds, row.date);
              return (
                <tr key={row.date}>
                  {/* Sticky, so a phone scrolling sideways keeps each row's meeting in view. */}
                  <th
                    scope="row"
                    className="sticky left-0 z-10 bg-canvas bg-clip-padding px-3 py-1.5 font-medium whitespace-nowrap text-ink shadow-[1px_0_0_var(--line)]"
                  >
                    {formatDate(row.date)}
                  </th>
                  {infos.map((info) => {
                    const result = row.results[info.id];
                    return (
                      <td key={info.id} className="px-3 py-1.5 whitespace-nowrap tabular-nums">
                        {result ? `${formatScore(result.score)} ${STANCE_LABELS[result.stance]}` : "-"}
                      </td>
                    );
                  })}
                  <td className="px-3 py-1.5 whitespace-nowrap">
                    {row.reference
                      ? `${STANCE_LABELS[row.reference.label]} (${REFERENCE_SOURCE_LABELS[row.reference.source]})`
                      : "-"}
                  </td>
                  <td className="px-3 py-1.5 whitespace-nowrap tabular-nums">
                    {range ? formatRange(range.lower, range.upper) : "-"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
