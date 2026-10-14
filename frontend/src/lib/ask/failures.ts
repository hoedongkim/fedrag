/** What went wrong with an answer, and how to present it with a way forward. */
import { ApiError } from "../api/client";
import { backendMaySleep, DEMO_UNREACHABLE } from "../api/hosting";
import { StreamInterruptedError } from "../api/stream";
import { isWakeError } from "../api/wake";

export type AskFailure =
  /** An HTTP error decided before the stream started (422, 503, ...). */
  | { kind: "http"; status: number; detail: string; body?: unknown }
  /** An `error` event after the stream started. */
  | { kind: "stream"; status: number; detail: string }
  /** The request never got a response (backend or proxy unreachable, offline). */
  | { kind: "network" }
  /** The stream ended without a `done` or `error` event. */
  | { kind: "interrupted" };

export interface FailureDescription {
  /** "setup" for configuration the operator has to fix; "error" for failed requests. */
  tone: "setup" | "error";
  title: string;
  message: string;
  /** A command that fixes it, shown as code. */
  command?: string;
  /** The server's own message, shown subdued, when it adds information. */
  detail?: string;
  /** "retry-hybrid" offers to retry without the reranker; "none" offers nothing (retrying cannot help). */
  action: "retry" | "retry-hybrid" | "none";
}

/** The failure for an error thrown by streamAsk (never called for an abort). */
export function failureFromError(error: unknown): AskFailure {
  if (error instanceof ApiError) {
    return { kind: "http", status: error.status, detail: error.detail, body: error.body };
  }
  if (error instanceof StreamInterruptedError) return { kind: "interrupted" };
  return { kind: "network" };
}

function describeUnavailable(detail: string): FailureDescription {
  if (/OPENAI_API_KEY|LLM not configured/i.test(detail)) {
    return {
      tone: "setup",
      title: "OpenAI API key not set",
      message:
        "The backend needs an OpenAI API key to write answers. Add OPENAI_API_KEY to backend/.env, restart the backend, then try again.",
      action: "retry",
    };
  }
  if (/reranker/i.test(detail)) {
    return {
      tone: "setup",
      title: "Reranker unavailable",
      message:
        "The cross-encoder reranker could not be loaded: it is downloaded on first use. Retry with Hybrid retrieval, which needs no extra model, or run the backend once with network access.",
      detail,
      action: "retry-hybrid",
    };
  }
  if (/index/i.test(detail)) {
    return {
      tone: "setup",
      title: "Search index not built",
      message:
        "The backend has no usable search index to retrieve passages from. Build it from the repo root, then try again:",
      command: "make index",
      // The default message says no more than the title; others name the cause (e.g. a model mismatch).
      detail: /^Search index not built/i.test(detail) ? undefined : detail,
      action: "retry",
    };
  }
  return { tone: "setup", title: "Service unavailable", message: detail, action: "retry" };
}

// The backend's 502 detail when OpenAI's 429 says the account has no credit left
// (insufficient_quota or credit_balance_exhausted; app/api/rag.py). Retrying cannot help.
const NO_CREDIT = /no credit left/i;

function describeModelFailure(detail: string): FailureDescription {
  if (NO_CREDIT.test(detail)) {
    return {
      tone: "setup",
      title: "OpenAI account out of credit",
      message:
        "The OpenAI account behind this server has no credit left; add credit at platform.openai.com/billing.",
      action: "retry",
    };
  }
  return {
    tone: "error",
    title: "The language model request failed",
    message:
      "OpenAI did not return an answer. Rate limits and network errors are usually temporary, so try again in a moment.",
    detail,
    action: "retry",
  };
}

/** The hosted demo's backend did not answer, even after waiting for it to wake. */
const DEMO_DOWN: FailureDescription = { ...DEMO_UNREACHABLE, action: "retry" };

/** True when a failure means no answer came from the API at all (see isWakeError). */
function noApiAnswer(failure: AskFailure): boolean {
  if (failure.kind === "network") return true;
  if (failure.kind !== "http") return false;
  return isWakeError(new ApiError(failure.status, failure.detail, failure.body));
}

export function describeFailure(failure: AskFailure): FailureDescription {
  if (backendMaySleep() && noApiAnswer(failure)) return DEMO_DOWN;
  switch (failure.kind) {
    case "network":
      return {
        tone: "error",
        title: "Couldn’t reach the FedRAG API",
        message:
          "Check your connection and that the backend is running, then try again. To start it locally:",
        command: "make dev-backend",
        action: "retry",
      };
    case "interrupted":
      return {
        tone: "error",
        title: "The answer was cut off",
        message:
          "The connection closed before the answer finished. Try again. If it keeps happening, check that the backend is still running.",
        action: "retry",
      };
    case "stream":
      if (failure.status === 502) return describeModelFailure(failure.detail);
      if (failure.status === 503) return describeUnavailable(failure.detail);
      return {
        tone: "error",
        title: "The answer failed",
        message: "Something went wrong while writing the answer. Try again.",
        detail: failure.detail,
        action: "retry",
      };
    case "http":
      if (failure.status === 403) {
        // The public demo turns live questions off (no OpenAI calls); the detail says how to ask.
        return {
          tone: "setup",
          title: "Live questions are off on this demo",
          message: failure.detail,
          action: "none",
        };
      }
      if (failure.status === 503) return describeUnavailable(failure.detail);
      if (failure.status === 502) return describeModelFailure(failure.detail);
      if (failure.status >= 500) {
        // Also what the Next.js /api proxy answers (a plain 500) when the backend is down.
        return {
          tone: "error",
          title: `The API returned an error (HTTP ${failure.status})`,
          message:
            "The backend failed, or it is not running and the proxy could not reach it. Check its log, or start it, then try again:",
          command: "make dev-backend",
          detail: failure.detail,
          action: "retry",
        };
      }
      return {
        tone: "error",
        title: `The request was rejected (HTTP ${failure.status})`,
        message: failure.detail,
        action: "retry",
      };
  }
}
