/** Why a comparison or its analysis could not be shown, and how to fix it. */
import { ApiError } from "../api/client";
import { backendMaySleep, DEMO_UNREACHABLE } from "../api/hosting";
import type { AnalysisStatus } from "../api/types";
import { isWakeError } from "../api/wake";
import type { StanceFailure } from "../stance/failures";
import { datesInMessage } from "./params";

/** A setup hint or an error, with the command that fixes it (rendered by FailureNotice). */
export type Notice = StanceFailure;

export type CompareFailure =
  /** No statement for one of the dates (HTTP 404); the message names the nearest ones. */
  | { kind: "not_found"; message: string; dates: string[] }
  | { kind: "notice"; notice: Notice };

const UNREACHABLE: Notice = {
  tone: "error",
  title: "Backend unreachable",
  message: "The API did not respond. Check that the backend is running, then try again.",
  command: "make dev-backend",
};

function serverError(status: number): Notice {
  // Also what the Next.js /api proxy answers (a plain 500) when the backend is down.
  return {
    tone: "error",
    title: `The API returned an error (HTTP ${status})`,
    message:
      "The backend failed, or it is not running and the proxy could not reach it. Check its log, or start it, then try again:",
    command: "make dev-backend",
  };
}

/** The failure for an error thrown while loading a comparison. */
export function describeCompareError(error: unknown): CompareFailure {
  // The hosted demo: no answer from the API even after waiting for it to wake.
  if (backendMaySleep() && isWakeError(error)) return { kind: "notice", notice: { ...DEMO_UNREACHABLE } };
  if (!(error instanceof ApiError)) return { kind: "notice", notice: UNREACHABLE };
  const detail = error.detail;
  if (error.status === 404) {
    return { kind: "not_found", message: detail, dates: datesInMessage(detail) };
  }
  if (error.status === 503) {
    if (/DATABASE_URL/i.test(detail)) {
      return {
        kind: "notice",
        notice: {
          tone: "setup",
          title: "Database not configured",
          message:
            "Statements are read from Postgres. Set DATABASE_URL in backend/.env and restart the backend.",
        },
      };
    }
    if (/make db-load/i.test(detail)) {
      return {
        kind: "notice",
        notice: {
          tone: "setup",
          title: "Database not loaded",
          message: "The database has no FOMC documents yet. Load them from the repo root:",
          command: "make db-load",
        },
      };
    }
    return { kind: "notice", notice: { tone: "error", title: "Service unavailable", message: detail } };
  }
  if (error.status >= 500) return { kind: "notice", notice: serverError(error.status) };
  return {
    kind: "notice",
    notice: { tone: "error", title: `The request was rejected (HTTP ${error.status})`, message: detail },
  };
}

/** The notice for any failed request of this tab (a 404 included), e.g. the statement list. */
export function describeLoadError(error: unknown): Notice {
  const failure = describeCompareError(error);
  return failure.kind === "notice"
    ? failure.notice
    : { tone: "error", title: "Statement not found", message: failure.message };
}

const LLM_NOT_CONFIGURED: Notice = {
  tone: "setup",
  title: "OpenAI API key not set",
  message:
    "The analysis is written by an LLM, so it needs OPENAI_API_KEY in backend/.env (then restart the backend). The sentence diff below works without it.",
};

const ANALYSIS_FAILED: Notice = {
  tone: "error",
  title: "The analysis could not be generated",
  message:
    "The language model call failed or returned an analysis that did not pass validation. The diff below is unaffected. Rate limits and network errors are usually temporary; Retry calls OpenAI again (nothing was stored).",
};

/** The public demo never calls the LLM, so a pair without a stored analysis stays without one. */
const DISABLED_ON_DEMO: Notice = {
  tone: "setup",
  title: "Analyses are not written on this demo",
  message:
    "The public demo does not call the LLM, so it shows only the analyses stored for consecutive statements. The sentence diff below is complete without one.",
};

/** The notice for an analysis status without an analysis to show, or null for the others. */
export function analysisStatusNotice(status: AnalysisStatus): Notice | null {
  if (status === "disabled_on_demo") return DISABLED_ON_DEMO;
  if (status === "llm_not_configured") return LLM_NOT_CONFIGURED;
  if (status === "failed") return ANALYSIS_FAILED;
  return null;
}

/** The notice for an error thrown by the "generate" request itself (the diff stays on screen). */
export function describeGenerateError(error: unknown): Notice {
  if (backendMaySleep() && isWakeError(error)) return { ...DEMO_UNREACHABLE };
  if (!(error instanceof ApiError)) return UNREACHABLE;
  if (error.status === 403) return { ...DISABLED_ON_DEMO, message: error.detail || DISABLED_ON_DEMO.message };
  if (error.status === 503 && /OPENAI_API_KEY|LLM not configured/i.test(error.detail)) {
    return LLM_NOT_CONFIGURED;
  }
  if (error.status === 502) return { ...ANALYSIS_FAILED, message: `${ANALYSIS_FAILED.message} (${error.detail})` };
  return describeLoadError(error);
}
