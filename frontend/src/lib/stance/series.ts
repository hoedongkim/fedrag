/**
 * Chart data for the stance timeline: one row per meeting with every loaded classifier's score,
 * the reference-label lane, and the federal funds target range as a step series over the same
 * time domain. Pure functions over the API types; times are UTC milliseconds.
 */
import type {
  FedFundsTarget,
  ReferenceLabel,
  Stance,
  StanceClassifier,
  StanceResult,
  StanceTimeline,
} from "../api/types";
import { CLASSIFIER_IDS } from "./classifiers";

const DAY_MS = 86_400_000;
/** Space before the first and after the last meeting, so their points are not on the frame. */
export const DOMAIN_PAD_MS = 20 * DAY_MS;
/** Days after a meeting by which FRED shows its decision (see rangeAfterMeeting). */
const SETTLE_DAYS = 3;
/** A time axis shorter than this gets month ticks instead of year ticks. */
const SHORT_DOMAIN_MS = 3 * 365 * DAY_MS;
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

export type Timelines = Partial<Record<StanceClassifier, StanceTimeline>>;

/** One meeting: a score per classifier that has a result (null otherwise), for Recharts. */
export type MeetingRow = {
  /** Meeting date, YYYY-MM-DD. */
  date: string;
  /** Meeting date as UTC milliseconds (the x value). */
  t: number;
  docId: string;
  url: string;
  reference: ReferenceLabel | null;
  /** The reference label's lane in the reference strip: dovish -1, neutral 0, hawkish 1. */
  referenceLane: number | null;
  results: Partial<Record<StanceClassifier, StanceResult>>;
} & Record<StanceClassifier, number | null>;

/** A point of the fed funds step series: the range in force from t until the next point. */
export interface FedFundsPoint {
  t: number;
  date: string;
  lower: number;
  upper: number;
  /** [lower, upper] for the shaded range band. */
  range: [number, number];
}

export const LANES: Record<Stance, number> = { dovish: -1, neutral: 0, hawkish: 1 };

/** "2024-09-18" -> UTC milliseconds of that day. */
export function toTime(isoDate: string): number {
  const [y, m, d] = isoDate.split("-").map(Number);
  return Date.UTC(y, m - 1, d);
}

/** UTC milliseconds -> "YYYY-MM-DD". */
export function toIsoDate(t: number): string {
  return new Date(t).toISOString().slice(0, 10);
}

/** The meetings of every loaded timeline, in date order, each with all classifiers' results. */
export function buildMeetingRows(timelines: Timelines): MeetingRow[] {
  const byDate = new Map<string, MeetingRow>();
  for (const classifier of CLASSIFIER_IDS) {
    for (const point of timelines[classifier]?.points ?? []) {
      let row = byDate.get(point.meeting_date);
      if (!row) {
        row = {
          date: point.meeting_date,
          t: toTime(point.meeting_date),
          docId: point.doc_id,
          url: point.url,
          reference: null,
          referenceLane: null,
          results: {},
          keyword: null,
          finbert: null,
          llm_zero_shot: null,
          llm_few_shot: null,
        };
        byDate.set(point.meeting_date, row);
      }
      row.results[classifier] = point;
      row[classifier] = point.score;
      if (!row.reference && point.reference) {
        row.reference = point.reference;
        row.referenceLane = LANES[point.reference.label];
      }
    }
  }
  return [...byDate.values()].sort((a, b) => a.t - b.t);
}

/** Rows whose meeting year lies in [fromYear, toYear]. */
export function rowsInYears(rows: MeetingRow[], fromYear: number, toYear: number): MeetingRow[] {
  return rows.filter((row) => {
    const year = Number(row.date.slice(0, 4));
    return year >= fromYear && year <= toYear;
  });
}

/** The x domain for rows: their first and last meeting, padded; null for no rows. */
export function timeDomain(rows: MeetingRow[]): [number, number] | null {
  if (rows.length === 0) return null;
  return [rows[0].t - DOMAIN_PAD_MS, rows[rows.length - 1].t + DOMAIN_PAD_MS];
}

/** The fed funds change points merged from every loaded timeline (they agree), in date order. */
export function mergeFedFunds(timelines: Timelines): FedFundsTarget[] {
  const byDate = new Map<string, FedFundsTarget>();
  for (const classifier of CLASSIFIER_IDS) {
    for (const point of timelines[classifier]?.fed_funds ?? []) byDate.set(point.date, point);
  }
  return [...byDate.values()].sort((a, b) => a.date.localeCompare(b.date));
}

/** The range in force on a date: the latest change point on or before it; null before the first. */
export function rangeInForce(changes: FedFundsTarget[], isoDate: string): FedFundsTarget | null {
  let found: FedFundsTarget | null = null;
  for (const change of changes) {
    if (change.date > isoDate) break;
    found = change;
  }
  return found;
}

/** The range in force going into the meeting: the one on the day before it. */
export function rangeBeforeMeeting(
  changes: FedFundsTarget[],
  meetingDate: string,
): FedFundsTarget | null {
  return rangeInForce(changes, toIsoDate(toTime(meetingDate) - DAY_MS));
}

/**
 * The range the meeting set: the one in force SETTLE_DAYS after it. FRED dates a change on the
 * meeting day, the next day, or (March 2020) two days later; meetings are at least 8 days apart,
 * so the window never reaches the next decision.
 */
export function rangeAfterMeeting(
  changes: FedFundsTarget[],
  meetingDate: string,
): FedFundsTarget | null {
  return rangeInForce(changes, toIsoDate(toTime(meetingDate) + SETTLE_DAYS * DAY_MS));
}

/**
 * The step series over a domain: the range in force at its start, every change inside it, a
 * point at each meeting (the range it set, so the step lines up with the decision and a hover
 * there reads it), and a point at the end so the last step reaches the frame.
 */
export function fedFundsSeries(
  changes: FedFundsTarget[],
  domain: [number, number],
  meetingDates: string[] = [],
): FedFundsPoint[] {
  const [start, end] = domain;
  const values = new Map<number, FedFundsTarget>();
  const startRange = rangeInForce(changes, toIsoDate(start));
  if (startRange) values.set(start, startRange);
  for (const change of changes) {
    const t = toTime(change.date);
    if (t > start && t <= end) values.set(t, change);
  }
  for (const date of meetingDates) {
    const t = toTime(date);
    const range = rangeAfterMeeting(changes, date);
    if (t >= start && t <= end && range) values.set(t, range);
  }
  if (values.size > 0) {
    const last = rangeInForce(changes, toIsoDate(end));
    if (last) values.set(end, last);
  }
  return [...values.entries()]
    .sort(([a], [b]) => a - b)
    .map(([t, { lower, upper }]) => ({ t, date: toIsoDate(t), lower, upper, range: [lower, upper] }));
}

/** January 1 of each year inside the domain; with `step`, only years divisible by it. */
export function yearTicks(domain: [number, number], step = 1): number[] {
  const first = new Date(domain[0]).getUTCFullYear();
  const last = new Date(domain[1]).getUTCFullYear();
  const ticks: number[] = [];
  for (let year = first; year <= last; year += 1) {
    const t = Date.UTC(year, 0, 1);
    if (t >= domain[0] && t <= domain[1] && year % step === 0) ticks.push(t);
  }
  return ticks;
}

/** The first day of every `stepMonths`-th month (from January) inside the domain. */
export function monthTicks(domain: [number, number], stepMonths = 3): number[] {
  const first = new Date(domain[0]).getUTCFullYear();
  const last = new Date(domain[1]).getUTCFullYear();
  const ticks: number[] = [];
  for (let year = first; year <= last; year += 1) {
    for (let month = 0; month < 12; month += stepMonths) {
      const t = Date.UTC(year, month, 1);
      if (t >= domain[0] && t <= domain[1]) ticks.push(t);
    }
  }
  return ticks;
}

/**
 * The x ticks and their labels: January 1 of each year ("2021"), or on an axis shorter than
 * three years the start of each quarter ("Apr", the year on January). The first tick always
 * names its year ("Apr 2020"), so a one-year window says which year it shows.
 */
export function timeTicks(
  domain: [number, number],
  narrow = false,
): { ticks: number[]; label: (t: number) => string } {
  if (domain[1] - domain[0] >= SHORT_DOMAIN_MS) {
    return { ticks: yearTicks(domain, narrow ? 2 : 1), label: (t) => String(new Date(t).getUTCFullYear()) };
  }
  const ticks = monthTicks(domain, narrow ? 6 : 3);
  const label = (t: number) => {
    const date = new Date(t);
    const year = String(date.getUTCFullYear());
    if (date.getUTCMonth() === 0) return year;
    const month = MONTHS[date.getUTCMonth()];
    return t === ticks[0] ? `${month} ${year}` : month;
  };
  return { ticks, label };
}

/** The ceiling of the y axis for the fed funds panel: the highest upper bound, rounded up to 1%. */
export function fedFundsCeiling(points: FedFundsPoint[]): number {
  const max = Math.max(0, ...points.map((p) => p.upper));
  return Math.max(1, Math.ceil(max));
}

/** The date of the meeting closest to t; null for no rows. */
export function nearestMeeting(rows: MeetingRow[], t: number): string | null {
  let best: MeetingRow | null = null;
  for (const row of rows) {
    if (!best || Math.abs(row.t - t) < Math.abs(best.t - t)) best = row;
  }
  return best?.date ?? null;
}

/** The meetings before and after `date` among rows (null at either end or if absent). */
export function adjacentMeetings(
  rows: MeetingRow[],
  date: string,
): { previous: string | null; next: string | null } {
  const i = rows.findIndex((row) => row.date === date);
  if (i < 0) return { previous: null, next: null };
  return { previous: rows[i - 1]?.date ?? null, next: rows[i + 1]?.date ?? null };
}

/**
 * Whether direct labels at the series' last points fit without overlapping: their pixel
 * positions on a [-1, 1] axis of `plotHeight` pixels must be at least `minGap` apart. When they
 * do not, the legend and tooltip carry identity instead (labels are never nudged off their lines).
 */
export function endLabelsFit(values: number[], plotHeight: number, minGap = 14): boolean {
  const ys = values.map((v) => ((1 - v) / 2) * plotHeight).sort((a, b) => a - b);
  return ys.every((y, i) => i === 0 || y - ys[i - 1] >= minGap);
}
