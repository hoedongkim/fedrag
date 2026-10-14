"use client";

import Link from "next/link";
import { type Ref, useId } from "react";

import type { FedFundsTarget, StanceClassifier } from "@/lib/api/types";
import { formatDate } from "@/lib/ask/format";
import { compareHref } from "@/lib/compare/params";
import { CLASSIFIERS } from "@/lib/stance/classifiers";
import {
  formatConfidence,
  formatRange,
  formatScore,
  REFERENCE_SOURCE_LABELS,
} from "@/lib/stance/format";
import { askHref } from "@/lib/stance/params";
import {
  adjacentMeetings,
  type MeetingRow,
  rangeAfterMeeting,
  rangeBeforeMeeting,
} from "@/lib/stance/series";

import { ArrowRightIcon, ChevronRightIcon } from "../ask/icons";
import { ExternalLink } from "../ui/ExternalLink";
import { labelTextClass, smallFieldClass, textButtonClass } from "../ui/styles";
import { useDemoMode } from "../useBackend";
import { SeriesKey } from "./SeriesKey";
import { StanceLabel } from "./StanceLabel";
import type { TimelineLoads } from "./useStanceData";

interface MeetingPanelProps {
  row: MeetingRow;
  rows: MeetingRow[];
  loads: TimelineLoads;
  fedFunds: FedFundsTarget[];
  detail: StanceClassifier;
  onDetail: (id: StanceClassifier) => void;
  onSelect: (meetingDate: string) => void;
  /** The statement before this one in the whole collection (not just the shown years), if any. */
  previousStatement: string | null;
  /** The meeting heading, focused when the panel should take focus. */
  headingRef?: Ref<HTMLHeadingElement>;
}

// Square: built from the outline button's look without its horizontal padding.
const stepButton =
  "inline-flex size-9 shrink-0 items-center justify-center rounded-md border border-line-strong bg-canvas text-ink transition-colors hover:bg-subtle aria-disabled:cursor-not-allowed aria-disabled:opacity-40 aria-disabled:hover:bg-canvas";

/**
 * One meeting: every method's call and score, the evidence behind the chosen method's call, the
 * reference label, the rate decision, and links to the statement, to Statement Comparison
 * (against the previous statement) and to Ask the Fed.
 */
export function MeetingPanel({
  row,
  rows,
  loads,
  fedFunds,
  detail,
  onDetail,
  onSelect,
  previousStatement,
  headingRef,
}: MeetingPanelProps) {
  const baseId = useId();
  const headingId = `${baseId}-heading`;
  const demo = useDemoMode();
  const { previous, next } = adjacentMeetings(rows, row.date);
  const before = rangeBeforeMeeting(fedFunds, row.date);
  const after = rangeAfterMeeting(fedFunds, row.date);
  const changed = before && after && (before.lower !== after.lower || before.upper !== after.upper);
  const chosen = row.results[detail] ?? null;
  const chosenInfo = CLASSIFIERS.find((c) => c.id === detail) ?? CLASSIFIERS[0];
  const confidence = chosen?.confidence ?? null;

  return (
    <section aria-labelledby={headingId}>
      <div className="flex flex-wrap items-end justify-between gap-x-6 gap-y-4">
        <div className="min-w-0">
          <p className={labelTextClass}>Selected statement</p>
          <h2
            id={headingId}
            ref={headingRef}
            tabIndex={-1}
            className="mt-0.5 rounded-sm text-heading font-semibold text-ink"
          >
            {formatDate(row.date)}
          </h2>
        </div>
        <div className="flex items-center gap-2">
          <label htmlFor={`${baseId}-meeting`} className="sr-only">
            Meeting
          </label>
          <select
            id={`${baseId}-meeting`}
            value={row.date}
            onChange={(event) => onSelect(event.target.value)}
            className={`h-9 w-40 ${smallFieldClass}`}
          >
            {[...rows].reverse().map((r) => (
              <option key={r.date} value={r.date}>
                {formatDate(r.date)}
              </option>
            ))}
          </select>
          <button
            type="button"
            onClick={() => previous && onSelect(previous)}
            aria-disabled={!previous || undefined}
            aria-label="Previous meeting"
            className={stepButton}
          >
            <ChevronRightIcon className="size-4 rotate-180" />
          </button>
          <button
            type="button"
            onClick={() => next && onSelect(next)}
            aria-disabled={!next || undefined}
            aria-label="Next meeting"
            className={stepButton}
          >
            <ChevronRightIcon className="size-4" />
          </button>
        </div>
      </div>

      <div className="mt-5 grid gap-x-14 gap-y-10 border-t border-line pt-6 lg:grid-cols-[minmax(0,21rem)_minmax(0,1fr)]">
        <div className="min-w-0">
          <dl className="grid grid-cols-2 gap-x-6 gap-y-4 text-small">
            <div>
              <dt className={labelTextClass}>Reference label</dt>
              <dd className="mt-1">
                {row.reference ? (
                  <>
                    <StanceLabel stance={row.reference.label} />
                    <span className="block text-label text-faint">
                      {REFERENCE_SOURCE_LABELS[row.reference.source]}
                      {formatConfidence(row.reference.confidence) &&
                        ` · confidence ${formatConfidence(row.reference.confidence)}`}
                    </span>
                  </>
                ) : (
                  <span className="text-faint">None yet</span>
                )}
              </dd>
            </div>
            <div>
              <dt className={labelTextClass}>Policy rate</dt>
              <dd className="mt-1 text-ink tabular-nums">
                {after ? (
                  changed && before ? (
                    <>
                      <span className="font-medium">{formatRange(after.lower, after.upper)}</span>
                      <span className="block text-label text-faint">from {formatRange(before.lower, before.upper)}</span>
                    </>
                  ) : (
                    <>
                      {formatRange(after.lower, after.upper)}
                      <span className="block text-label text-faint">unchanged</span>
                    </>
                  )
                ) : (
                  <span className="text-faint">Not loaded</span>
                )}
              </dd>
            </div>
          </dl>

          <fieldset className="mt-8">
            <legend className={labelTextClass}>Each method&rsquo;s call (choose one for its evidence)</legend>
            <div className="mt-2 border-t border-line">
              {CLASSIFIERS.map((info) => {
                const result = row.results[info.id];
                const load = loads[info.id];
                const inputId = `${baseId}-${info.id}`;
                return (
                  <div
                    key={info.id}
                    className="relative flex items-center gap-3 border-b border-line px-2 py-2.5 transition-colors has-checked:bg-accent-soft has-checked:shadow-[inset_2px_0_0_var(--accent)] [&:has(:enabled):not(:has(:checked)):hover]:bg-subtle"
                  >
                    <input
                      id={inputId}
                      type="radio"
                      name={`${baseId}-classifier`}
                      value={info.id}
                      checked={detail === info.id}
                      disabled={!result}
                      onChange={() => onDetail(info.id)}
                      className="size-3.5 shrink-0 accent-accent"
                    />
                    {/* The label covers the row, so the whole line is the target. */}
                    <label
                      htmlFor={inputId}
                      className="flex min-w-0 flex-1 items-center gap-3 text-small after:absolute after:inset-0 after:content-['']"
                    >
                      <span className="flex min-w-0 flex-1 items-center gap-2 font-medium text-ink">
                        <SeriesKey info={info} />
                        <span className="truncate">{info.label}</span>
                      </span>
                      {result ? (
                        <>
                          <StanceLabel stance={result.stance} className="w-[5.5rem]" />
                          <span className="w-11 text-right font-medium text-ink tabular-nums">
                            <span className="sr-only">score </span>
                            {formatScore(result.score)}
                          </span>
                        </>
                      ) : (
                        <span className="text-faint">
                          {load.status === "loading"
                            ? "Loading…"
                            : load.status === "error"
                              ? "Not loaded"
                              : "No result"}
                        </span>
                      )}
                    </label>
                  </div>
                );
              })}
            </div>
          </fieldset>
        </div>

        {chosen ? (
          <div className="min-w-0">
            <h3 className="text-body font-semibold text-ink">
              Why {chosenInfo.label} called it {chosen.stance}
            </h3>
            <p className="mt-1 text-small text-faint">
              {chosenInfo.description}{" "}
              <span className="tabular-nums">
                Score {formatScore(chosen.score)}
                {confidence !== null &&
                  ` · confidence ${confidence.toFixed(2)}${chosenInfo.sentenceBased ? " (the score’s size)" : ""}`}
              </span>
            </p>
            {chosen.key_factors.length > 0 && (
              <div className="mt-4 text-small text-muted">
                <span className="text-faint">Key factors: </span>
                <ul aria-label="Key factors" className="inline">
                  {chosen.key_factors.map((factor, i) => (
                    <li key={factor} className="inline">
                      {i > 0 && <span aria-hidden="true"> · </span>}
                      {factor}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {chosen.evidence.length > 0 ? (
              <div className="mt-5 space-y-3">
                <p className="sr-only">Evidence, quoted from the statement:</p>
                {chosen.evidence.map((sentence) => (
                  <blockquote
                    key={sentence}
                    className="max-w-[70ch] border-l-2 border-line-strong pl-4 text-body text-ink"
                  >
                    {sentence}
                  </blockquote>
                ))}
              </div>
            ) : (
              <p className="mt-4 text-small text-faint">This method quotes no evidence sentences.</p>
            )}
            <p className="mt-5 font-mono text-label break-all text-faint">{chosen.model_version}</p>
          </div>
        ) : (
          <p className="text-small text-faint">No method has a result for this statement.</p>
        )}
      </div>

      <div className="mt-8 flex flex-wrap gap-x-6 gap-y-2 border-t border-line pt-5 text-small">
        <ExternalLink href={row.url} className={textButtonClass}>
          Read the statement
        </ExternalLink>
        {previousStatement && (
          <Link href={compareHref(previousStatement, row.date)} className={`${textButtonClass} inline-flex items-center gap-1`}>
            Compare with the previous statement
            <ArrowRightIcon className="size-3.5" />
          </Link>
        )}
        {/* The public demo cannot answer new questions, so it does not offer to ask one. */}
        {demo === false && (
          <Link href={askHref(row.date)} className={`${textButtonClass} inline-flex items-center gap-1`}>
            Ask about this meeting
            <ArrowRightIcon className="size-3.5" />
          </Link>
        )}
      </div>
    </section>
  );
}
