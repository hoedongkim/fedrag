/**
 * Statement Comparison state in the URL (base and target meeting dates), resolving it against the
 * statement dates, moving between pairs, and the links in and out of the tab.
 */
import { formatDate } from "../ask/format";
import { isIsoDate, METHOD_DEFAULT, serializeAskParams, TOP_K_DEFAULT } from "../ask/params";

/** The two meeting dates in the URL; null means "not given" (resolved from the other one). */
export interface CompareParams {
  base: string | null;
  target: string | null;
}

export const EMPTY_PARAMS: CompareParams = { base: null, target: null };

/** Why a pair cannot be compared as given. */
export type PairProblem =
  /** The base is on or after the target: the base must be the earlier statement. */
  | "order"
  /** Only a target was given and no statement precedes it. */
  | "no-base"
  /** Only a base was given and no statement follows it. */
  | "no-target"
  /** Fewer than two statements to compare. */
  | "empty";

export type Resolution =
  | { ok: true; base: string; target: string }
  | { ok: false; problem: PairProblem; base: string | null; target: string | null };

interface ParamSource {
  get(name: string): string | null;
}

/** Dates from a query string; anything that is not a real YYYY-MM-DD date is ignored. */
export function parseCompareParams(params: ParamSource): CompareParams {
  const base = params.get("base");
  const target = params.get("target");
  return {
    base: base && isIsoDate(base) ? base : null,
    target: target && isIsoDate(target) ? target : null,
  };
}

/** The query string (without "?"): the dates that are set, base first. */
export function serializeCompareParams(params: CompareParams): string {
  const query = new URLSearchParams();
  if (params.base) query.set("base", params.base);
  if (params.target) query.set("target", params.target);
  return query.toString();
}

/** The latest statement date before `date` (dates sorted ascending), or null. */
export function previousStatement(dates: readonly string[], date: string): string | null {
  let found: string | null = null;
  for (const d of dates) {
    if (d >= date) break;
    found = d;
  }
  return found;
}

/** The earliest statement date after `date` (dates sorted ascending), or null. */
export function nextStatement(dates: readonly string[], date: string): string | null {
  return dates.find((d) => d > date) ?? null;
}

/** The statement dates on either side of a date that has no statement, for quick picks. */
export function nearestStatements(
  dates: readonly string[],
  date: string,
): { before: string | null; after: string | null } {
  return { before: previousStatement(dates, date), after: nextStatement(dates, date) };
}

/**
 * The pair to compare. With no dates, the latest statement against the one before it; with one
 * date, its neighbour fills the other side. The pair is never reordered: a base on or after the
 * target is reported, so the page can say why instead of silently swapping it. `dates` is only
 * needed to fill a missing side (sorted ascending; null while still loading).
 */
export function resolvePair(params: CompareParams, dates: readonly string[] | null): Resolution | null {
  let { base, target } = params;
  if (base && target) {
    return base < target ? { ok: true, base, target } : { ok: false, problem: "order", base, target };
  }
  if (dates === null) return null;
  if (!base && !target) {
    if (dates.length < 2) return { ok: false, problem: "empty", base: null, target: null };
    return { ok: true, base: dates[dates.length - 2], target: dates[dates.length - 1] };
  }
  if (target) {
    base = previousStatement(dates, target);
    return base ? { ok: true, base, target } : { ok: false, problem: "no-base", base, target };
  }
  target = nextStatement(dates, base as string);
  return target ? { ok: true, base: base as string, target } : { ok: false, problem: "no-target", base, target };
}

/**
 * The pair `step` statements earlier (-1) or later (+1), keeping the gap between the two; null
 * when either date is not a statement date or the move would leave the list.
 */
export function shiftPair(
  dates: readonly string[],
  base: string,
  target: string,
  step: -1 | 1,
): { base: string; target: string } | null {
  const i = dates.indexOf(base);
  const j = dates.indexOf(target);
  if (i < 0 || j < 0) return null;
  const nextBase = dates[i + step];
  const nextTarget = dates[j + step];
  return nextBase && nextTarget ? { base: nextBase, target: nextTarget } : null;
}

/** ISO dates named in an error message (the 404 names the nearest statement dates), in order. */
export function datesInMessage(message: string): string[] {
  const found = message.match(/\b\d{4}-\d{2}-\d{2}\b/g) ?? [];
  return [...new Set(found.filter(isIsoDate))];
}

/** Link to this tab for a pair, e.g. from the Policy Stance meeting panel. */
export function compareHref(base: string, target: string): string {
  return `/compare?${serializeCompareParams({ base, target })}`;
}

/** The question /ask is prefilled with for a pair; the user still has to submit it. */
export function changeQuestion(base: string, target: string): string {
  return `What changed between the FOMC statements of ${formatDate(base)} and ${formatDate(target)}, and what does the change signal about monetary policy?`;
}

/**
 * Link to Ask the Fed about the change, as of the target statement's release, so the answer
 * cannot draw on anything published later. Opening it never submits.
 */
export function compareAskHref(base: string, target: string, targetRelease: string): string {
  const query = serializeAskParams({
    question: changeQuestion(base, target),
    asOf: targetRelease,
    method: METHOD_DEFAULT,
    topK: TOP_K_DEFAULT,
  });
  return `/ask?${query}`;
}
