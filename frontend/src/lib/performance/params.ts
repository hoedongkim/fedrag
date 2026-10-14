/**
 * RAG Performance view state in the URL: the split every section shows (split), the question
 * browser's filters (type, outcome, run) and the open question (q). Defaults are left out, so a
 * plain /performance link stays clean.
 */
import type { QuestionType, ReportSplit, RunKey } from "../api/types";
import { isOutcomeFilter, type OutcomeFilter } from "./questions";
import { parseRunKey, runKey, SPLIT_ORDER, TYPE_ORDER } from "./report";

/** The backend's question id pattern (QUESTION_ID_PATTERN). */
const QUESTION_ID = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$/;

export interface PerformanceViewState {
  /** Split for every section; null for the default (test when there is one). */
  split: ReportSplit | null;
  /** Question browser: one question type, or null for all. */
  type: QuestionType | null;
  outcome: OutcomeFilter;
  /** Run the browser judges outcomes on; null for the headline run. */
  run: RunKey | null;
  /** Question open in the drilldown. */
  question: string | null;
}

export const DEFAULT_VIEW: PerformanceViewState = {
  split: null,
  type: null,
  outcome: "all",
  run: null,
  question: null,
};

interface ParamSource {
  get(name: string): string | null;
}

export function isQuestionId(value: string | null | undefined): value is string {
  return !!value && QUESTION_ID.test(value);
}

/** View state from a query string; anything invalid falls back to its default. */
export function parsePerformanceParams(params: ParamSource): PerformanceViewState {
  const split = params.get("split");
  const type = params.get("type");
  const outcome = params.get("outcome");
  const question = params.get("q");
  return {
    split: SPLIT_ORDER.find((s) => s === split) ?? null,
    type: TYPE_ORDER.find((t) => t === type) ?? null,
    outcome: isOutcomeFilter(outcome) ? outcome : "all",
    run: parseRunKey(params.get("run")),
    question: isQuestionId(question) ? question : null,
  };
}

/** The query string (without "?") for a view state; defaults are left out. */
export function serializePerformanceParams(state: PerformanceViewState): string {
  const params = new URLSearchParams();
  if (state.split) params.set("split", state.split);
  if (state.type) params.set("type", state.type);
  if (state.outcome !== "all") params.set("outcome", state.outcome);
  if (state.run) params.set("run", runKey(state.run));
  if (state.question) params.set("q", state.question);
  // Colons are safe in a query value; keep the run readable.
  return params.toString().replaceAll("%3A", ":");
}
