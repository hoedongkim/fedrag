/**
 * Policy Stance view state in the URL: the plotted classifiers (c), the selected meeting (m) and
 * the year range (from, to). Defaults are left out, so a plain /stance link stays clean.
 */
import type { StanceClassifier } from "../api/types";
import { formatDate } from "../ask/format";
import { isIsoDate, METHOD_DEFAULT, serializeAskParams, TOP_K_DEFAULT } from "../ask/params";
import { CLASSIFIER_IDS, DEFAULT_CLASSIFIER, isClassifier } from "./classifiers";

/** The first year of the corpus; the last comes from the data. */
export const FIRST_YEAR = 2015;
const MAX_YEAR = 2100;

export interface StanceViewState {
  /** Plotted classifiers, in display order; never empty. */
  classifiers: StanceClassifier[];
  /** Selected meeting (YYYY-MM-DD); null for the latest. */
  meeting: string | null;
  /** First and last year shown; null for no bound. */
  fromYear: number | null;
  toYear: number | null;
}

export const DEFAULT_VIEW: StanceViewState = {
  classifiers: [DEFAULT_CLASSIFIER],
  meeting: null,
  fromYear: null,
  toYear: null,
};

interface ParamSource {
  get(name: string): string | null;
}

function parseYear(value: string | null): number | null {
  if (value === null || !/^\d{4}$/.test(value)) return null;
  const year = Number(value);
  return year >= FIRST_YEAR && year <= MAX_YEAR ? year : null;
}

/** Classifiers in display order, deduplicated; the default when none is valid. */
export function orderClassifiers(ids: Iterable<string>): StanceClassifier[] {
  const wanted = new Set(ids);
  const ordered = CLASSIFIER_IDS.filter((id) => wanted.has(id));
  return ordered.length > 0 ? ordered : [DEFAULT_CLASSIFIER];
}

/** View state from a query string; anything invalid falls back to its default. */
export function parseStanceParams(params: ParamSource): StanceViewState {
  const c = params.get("c");
  const meeting = params.get("m");
  let fromYear = parseYear(params.get("from"));
  let toYear = parseYear(params.get("to"));
  if (fromYear !== null && toYear !== null && fromYear > toYear) {
    fromYear = null;
    toYear = null;
  }
  return {
    classifiers: orderClassifiers((c ?? "").split(",").filter(isClassifier)),
    meeting: meeting && isIsoDate(meeting) ? meeting : null,
    fromYear,
    toYear,
  };
}

/** The query string (without "?") for a view state; defaults are left out. */
export function serializeStanceParams(state: StanceViewState): string {
  const params = new URLSearchParams();
  const classifiers = orderClassifiers(state.classifiers);
  if (classifiers.length !== 1 || classifiers[0] !== DEFAULT_CLASSIFIER) {
    params.set("c", classifiers.join(","));
  }
  if (state.meeting) params.set("m", state.meeting);
  if (state.fromYear !== null) params.set("from", String(state.fromYear));
  if (state.toYear !== null) params.set("to", String(state.toYear));
  // Commas are safe in a query value; keep them readable.
  return params.toString().replaceAll("%2C", ",");
}

/** Toggle one classifier, keeping at least one plotted. */
export function toggleClassifier(
  current: StanceClassifier[],
  id: StanceClassifier,
): StanceClassifier[] {
  if (!current.includes(id)) return orderClassifiers([...current, id]);
  return current.length === 1 ? current : current.filter((c) => c !== id);
}

/** The question /ask is prefilled with for a meeting; the user still has to submit it. */
export function meetingQuestion(meetingDate: string): string {
  return `What did the FOMC decide at its ${formatDate(meetingDate)} meeting, and how did it explain its policy stance?`;
}

/** Link to Ask the Fed with the meeting question filled in (opening it never submits). */
export function askHref(meetingDate: string): string {
  const query = serializeAskParams({
    question: meetingQuestion(meetingDate),
    asOf: "",
    method: METHOD_DEFAULT,
    topK: TOP_K_DEFAULT,
  });
  return `/ask?${query}`;
}
