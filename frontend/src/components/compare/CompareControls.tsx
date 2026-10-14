"use client";

import { useId } from "react";

import type { MeetingSummary } from "@/lib/api/types";
import { formatDate } from "@/lib/ask/format";
import { describeLoadError } from "@/lib/compare/failures";
import { previousStatement, shiftPair } from "@/lib/compare/params";

import { ChevronRightIcon } from "../ask/icons";
import { FailureNotice } from "../stance/FailureNotice";
import { fieldClass, labelClass, secondaryButtonClass } from "../ui/styles";
import type { Load } from "../stance/useStanceData";

interface CompareControlsProps {
  statements: Load<MeetingSummary[]>;
  onRetryStatements: () => void;
  /** False when the same failure is already shown for the comparison below. */
  showStatementsError: boolean;
  /** The pair on screen; a side is null when it could not be resolved. */
  base: string | null;
  target: string | null;
  onChange: (base: string | null, target: string | null) => void;
}

const DISABLED = "aria-disabled:cursor-not-allowed aria-disabled:opacity-40";

/**
 * The two statement pickers (newest first, grouped by year) and shortcuts. Dates that would put
 * the base on or after the target are disabled in each list, so a choice can never invert the
 * pair (a screen-reader note says why).
 */
export function CompareControls({
  statements,
  onRetryStatements,
  showStatementsError,
  base,
  target,
  onChange,
}: CompareControlsProps) {
  const baseId = useId();
  const helpId = `${baseId}-help`;
  const meetings = statements.status === "ready" ? statements.data : null;
  const dates = meetings?.map((m) => m.meeting_date) ?? null;

  const previous = dates && target ? previousStatement(dates, target) : null;
  // Stepping only from a valid pair: from an inverted or same-date one it lands on another.
  const earlier = dates && base && target && base < target ? shiftPair(dates, base, target, -1) : null;
  const later = dates && base && target && base < target ? shiftPair(dates, base, target, 1) : null;
  const previousIsBase = previous !== null && previous === base;

  return (
    <section aria-labelledby={`${baseId}-heading`}>
      <h2 id={`${baseId}-heading`} className="sr-only">
        Statements to compare
      </h2>

      <div className="grid max-w-[46rem] gap-4 sm:grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)] sm:items-end">
        <StatementSelect
          id={`${baseId}-base`}
          label="Earlier statement"
          value={base}
          meetings={meetings}
          loading={statements.status === "loading"}
          isDisabled={(date) => target !== null && date >= target}
          describedBy={helpId}
          onChange={(date) => onChange(date, target)}
        />
        <span aria-hidden="true" className="hidden pb-2 text-faint sm:block">
          →
        </span>
        <StatementSelect
          id={`${baseId}-target`}
          label="Later statement"
          value={target}
          meetings={meetings}
          loading={statements.status === "loading"}
          isDisabled={(date) => base !== null && date <= base}
          describedBy={helpId}
          onChange={(date) => onChange(base, date)}
        />
      </div>
      <p id={helpId} className="sr-only">
        Dates that would put the earlier statement on or after the later one are unavailable.
      </p>

      <div className="mt-4 flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={() => earlier && onChange(earlier.base, earlier.target)}
          aria-disabled={!earlier || undefined}
          className={`${secondaryButtonClass} ${DISABLED}`}
        >
          <ChevronRightIcon className="size-3.5 rotate-180" />
          Earlier pair
        </button>
        <button
          type="button"
          onClick={() => later && onChange(later.base, later.target)}
          aria-disabled={!later || undefined}
          className={`${secondaryButtonClass} ${DISABLED}`}
        >
          Later pair
          <ChevronRightIcon className="size-3.5" />
        </button>
        <button
          type="button"
          onClick={() => previous && !previousIsBase && onChange(previous, target)}
          aria-disabled={!previous || previousIsBase || undefined}
          className={`${secondaryButtonClass} ${DISABLED}`}
        >
          Use the meeting just before
          {previousIsBase && <span className="sr-only"> (already the earlier statement)</span>}
        </button>
      </div>

      {statements.status === "error" && showStatementsError && (
        <div className="mt-6">
          <FailureNotice failure={describeLoadError(statements.error)} onRetry={onRetryStatements} />
        </div>
      )}
    </section>
  );
}

interface StatementSelectProps {
  id: string;
  label: string;
  value: string | null;
  meetings: MeetingSummary[] | null;
  loading: boolean;
  isDisabled: (date: string) => boolean;
  describedBy: string;
  onChange: (date: string) => void;
}

function StatementSelect({ id, label, value, meetings, loading, isDisabled, describedBy, onChange }: StatementSelectProps) {
  const byYear = new Map<string, MeetingSummary[]>();
  for (const meeting of [...(meetings ?? [])].reverse()) {
    const year = meeting.meeting_date.slice(0, 4);
    byYear.set(year, [...(byYear.get(year) ?? []), meeting]);
  }
  const listed = meetings?.some((m) => m.meeting_date === value) ?? false;

  return (
    <div className="min-w-0">
      <label htmlFor={id} className={labelClass}>
        {label}
      </label>
      <select
        id={id}
        value={value ?? ""}
        disabled={meetings === null && value === null}
        aria-describedby={describedBy}
        onChange={(event) => event.target.value && onChange(event.target.value)}
        className={`mt-1.5 ${fieldClass} disabled:cursor-wait disabled:opacity-60`}
      >
        {value === null && <option value="">{loading ? "Loading statements…" : "Choose a statement"}</option>}
        {value !== null && !listed && (
          // A date from the URL that is not in the list (yet, or at all) still shows as chosen.
          <option value={value}>
            {formatDate(value)}
            {meetings ? " (no statement)" : ""}
          </option>
        )}
        {[...byYear].map(([year, group]) => (
          <optgroup key={year} label={year}>
            {group.map((meeting) => (
              <option key={meeting.meeting_date} value={meeting.meeting_date} disabled={isDisabled(meeting.meeting_date)}>
                {formatDate(meeting.meeting_date)}
                {meeting.unscheduled ? " (unscheduled)" : ""}
              </option>
            ))}
          </optgroup>
        ))}
      </select>
    </div>
  );
}
