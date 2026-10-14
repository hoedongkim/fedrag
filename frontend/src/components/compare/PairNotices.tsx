import type { ReactNode } from "react";

import { formatDate } from "@/lib/ask/format";
import { datesInMessage, nearestStatements, nextStatement, previousStatement, type Resolution } from "@/lib/compare/params";

import { secondaryButtonClass } from "../ui/styles";

type Choose = (base: string, target: string) => void;
interface Pick {
  label: string;
  base: string;
  target: string;
}

/** A problem with the pair: an amber rule at the left (as FailureNotice), no box, and quick picks. */
function NoticeBox({ title, children, picks, onChoose }: { title: string; children: ReactNode; picks: Pick[]; onChoose: Choose }) {
  return (
    <div className="max-w-[70ch] border-l-2 border-warn py-0.5 pl-4">
      <h2 className="text-body font-semibold text-ink">{title}</h2>
      <div className="mt-1 text-body text-muted">{children}</div>
      {picks.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-2">
          {picks.map((pick) => (
            <button
              key={`${pick.base}|${pick.target}`}
              type="button"
              onClick={() => onChoose(pick.base, pick.target)}
              className={secondaryButtonClass}
            >
              {pick.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

const pairLabel = (base: string, target: string) => `Compare ${formatDate(base)} → ${formatDate(target)}`;

/** Why the pair in the URL is not compared, with the pair the user probably meant. */
export function PairProblemNotice({
  resolution,
  dates,
  onChoose,
}: {
  resolution: Extract<Resolution, { ok: false }>;
  dates: readonly string[] | null;
  onChoose: Choose;
}) {
  const { problem, base, target } = resolution;
  if (problem === "order" && base && target) {
    const same = base === target;
    const previous = same && dates ? previousStatement(dates, target) : null;
    const picks = same
      ? previous
        ? [{ label: pairLabel(previous, target), base: previous, target }]
        : []
      : [{ label: pairLabel(target, base), base: target, target: base }];
    return (
      <NoticeBox title="The earlier statement must come first" picks={picks} onChoose={onChoose}>
        {same
          ? `Both sides are ${formatDate(base)}: choose two different statements.`
          : `This link lists ${formatDate(base)} first, but it is later than ${formatDate(target)}, so nothing was compared.`}
      </NoticeBox>
    );
  }
  if (problem === "no-base" && target) {
    const next = dates ? nextStatement(dates, target) : null;
    return (
      <NoticeBox
        title="No earlier statement"
        picks={next ? [{ label: pairLabel(target, next), base: target, target: next }] : []}
        onChoose={onChoose}
      >
        {`${formatDate(target)} is the first statement, so there is nothing earlier to compare it with.`}
      </NoticeBox>
    );
  }
  if (problem === "no-target" && base) {
    const previous = dates ? previousStatement(dates, base) : null;
    return (
      <NoticeBox
        title="No later statement"
        picks={previous ? [{ label: pairLabel(previous, base), base: previous, target: base }] : []}
        onChoose={onChoose}
      >
        {`${formatDate(base)} is the latest statement, so there is nothing later to compare it with.`}
      </NoticeBox>
    );
  }
  return (
    <NoticeBox title="Nothing to compare" picks={[]} onChoose={onChoose}>
      Fewer than two FOMC statements are loaded.
    </NoticeBox>
  );
}

/**
 * HTTP 404: no statement for one of the dates. Offers the nearest statements on either side of
 * each missing date as quick picks (from the statement list, or the dates the message names).
 */
export function NotFoundNotice({
  message,
  base,
  target,
  dates,
  onChoose,
}: {
  message: string;
  base: string;
  target: string;
  dates: readonly string[] | null;
  onChoose: Choose;
}) {
  const named = datesInMessage(message).filter((d) => d !== base && d !== target);
  const near = (date: string) => {
    if (dates) {
      const { before, after } = nearestStatements(dates, date);
      return [before, after].filter((d): d is string => d !== null);
    }
    return named;
  };
  const missingBase = dates ? !dates.includes(base) : message.includes(base);
  const missingTarget = dates ? !dates.includes(target) : message.includes(target);

  const picks: Pick[] = [];
  if (missingBase) {
    for (const d of near(base)) {
      if (d < target) picks.push({ label: `Earlier: ${formatDate(d)}`, base: d, target });
    }
  }
  if (missingTarget) {
    for (const d of near(target)) {
      if (d > base) picks.push({ label: `Later: ${formatDate(d)}`, base, target: d });
    }
  }

  return (
    <NoticeBox title="No statement for that date" picks={picks} onChoose={onChoose}>
      <p>{message}</p>
      {picks.length > 0 && <p className="mt-1">Pick a nearby statement instead:</p>}
    </NoticeBox>
  );
}
