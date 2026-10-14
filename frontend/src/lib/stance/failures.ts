/** Why stance data could not be shown, and how to fix it. */
import { ApiError } from "../api/client";
import { backendMaySleep, DEMO_UNREACHABLE } from "../api/hosting";
import { isWakeError, isWakeExhausted } from "../api/wake";

export interface StanceFailure {
  /** "setup" for something the operator has to run or configure; "error" otherwise. */
  tone: "setup" | "error";
  title: string;
  message: string;
  /** A command that fixes it, shown as code. */
  command?: string;
}

/** The failure for an error thrown while loading a stance timeline or the benchmark. */
export function describeStanceError(error: unknown): StanceFailure {
  // The hosted demo: no answer from the API even after waiting for it to wake.
  if (backendMaySleep() && isWakeError(error)) return { ...DEMO_UNREACHABLE };
  if (!(error instanceof ApiError)) {
    return {
      tone: "error",
      title: "Backend unreachable",
      message: "The API did not respond. Check that the backend is running, then try again.",
      command: "make dev-backend",
    };
  }
  const detail = error.detail;
  if (error.status === 503) {
    if (/DATABASE_URL/i.test(detail)) {
      return {
        tone: "setup",
        title: "Database not configured",
        message:
          "Stance results are stored in Postgres. Set DATABASE_URL in backend/.env and restart the backend.",
      };
    }
    if (/make db-load/i.test(detail)) {
      return {
        tone: "setup",
        title: "Database not loaded",
        message: "The database has no FOMC documents yet. Load them, then compute the stances:",
        command: "make db-load && make stance",
      };
    }
    if (/benchmark/i.test(detail)) {
      return {
        tone: "setup",
        title: "Benchmark not computed yet",
        message: "The classifier benchmark is written by the stance pipeline:",
        command: "make stance",
      };
    }
    if (/make stance|not loaded/i.test(detail)) {
      return {
        tone: "setup",
        title: "Stance results not loaded",
        message: "No classifier results are stored yet. Compute and load them from the repo root:",
        command: "make stance",
      };
    }
    return { tone: "error", title: "Service unavailable", message: detail };
  }
  if (error.status >= 500) {
    // Also what the Next.js /api proxy answers (a plain 500) when the backend is down.
    return {
      tone: "error",
      title: `The API returned an error (HTTP ${error.status})`,
      message:
        "The backend failed, or it is not running and the proxy could not reach it. Check its log, or start it, then try again:",
      command: "make dev-backend",
    };
  }
  return { tone: "error", title: `Request failed (HTTP ${error.status})`, message: detail };
}

/**
 * True when the failure may mean the backend is down, so the header's status should re-check.
 * Not when the request already waited out a whole wake-up: that check has just failed, and a new
 * one would show "Waking" next to the page's error.
 */
export function suggestsBackendDown(error: unknown): boolean {
  if (isWakeExhausted(error)) return false;
  return !(error instanceof ApiError) || (error.status >= 500 && error.status !== 503);
}

/** True when the error only means this classifier has no stored results yet. */
export function isNotLoaded(error: unknown): boolean {
  return error instanceof ApiError && error.status === 503 && /make stance/i.test(error.detail);
}
