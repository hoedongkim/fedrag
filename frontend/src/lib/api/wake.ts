/** Telling a backend that is still waking up from one that answered with an error of its own. */
import { ApiError } from "./client";

/** The first health check: a hosted backend that has not answered by then is shown as waking. */
export const SLOW_AFTER_MS = 2500;
/** Timeout of the first health check, and of each check while the backend wakes. */
export const FIRST_CHECK_TIMEOUT_MS = 6000;
export const WAKE_CHECK_TIMEOUT_MS = 12000;
/** How long to keep waiting for a sleeping backend before reporting it unreachable. */
export const WAKE_LIMIT_MS = 150_000;

const BASE_DELAY_MS = 2000;
const MAX_DELAY_MS = 10_000;

/** Pause before health check number `attempt` + 1 of a wake-up (attempt >= 1): 2, 4, 8, 10, 10 s... */
export function wakeDelay(attempt: number): number {
  const n = Math.max(1, Math.floor(attempt));
  return Math.min(MAX_DELAY_MS, BASE_DELAY_MS * 2 ** (n - 1));
}

/**
 * Errors of requests that already waited out a whole wake-up (the health check then failed for
 * good). Re-checking health for them would only start another wake-up the page is not waiting for.
 */
const wakeExhaustedErrors = new WeakSet<object>();

/** Mark `error` as thrown after a full wait for the backend to wake; see isWakeExhausted. */
export function markWakeExhausted(error: unknown): void {
  if (typeof error === "object" && error !== null) wakeExhaustedErrors.add(error);
}

/** True when the request failed after waiting for the backend to wake, and it never did. */
export function isWakeExhausted(error: unknown): boolean {
  return typeof error === "object" && error !== null && wakeExhaustedErrors.has(error);
}

/** A FastAPI error body: `{"detail": ...}`. Its presence means the API itself answered. */
export function isApiErrorBody(body: unknown): boolean {
  return typeof body === "object" && body !== null && !Array.isArray(body) && "detail" in body;
}

/**
 * True when a failed request looks like a backend that is not up yet (asleep, starting, or not
 * reachable) rather than one that answered with an error of its own: no response at all, a
 * timeout, a 5xx without FastAPI's JSON `detail` (a hosting or proxy page, e.g. a sleeping Space's
 * 503 or the proxy's 500/504), or a non-JSON page where JSON was expected. A 503 with a `detail`,
 * such as "Database not configured", is the API speaking, so it is not.
 *
 * An abort the caller asked for also looks like this: check the caller's signal first.
 */
export function isWakeError(error: unknown): boolean {
  if (error instanceof ApiError) {
    if (error.status < 400) return true;
    if (error.status >= 500) return !isApiErrorBody(error.body);
    return false;
  }
  if (typeof DOMException !== "undefined" && error instanceof DOMException) {
    return error.name === "AbortError" || error.name === "TimeoutError";
  }
  // fetch rejects with a TypeError when there is no response (offline, refused, DNS, CORS).
  return error instanceof TypeError;
}
