/**
 * Is the backend up, and in which mode? One store for the whole app, fed by GET /api/health:
 * the header's status pill, the "waking" notice, the pages' demo-mode switches and the
 * wake-aware fetch all read it, so health is fetched once per page load (and re-checked when a
 * request fails or the window regains focus), never once per component.
 *
 * A hosted backend that sleeps (see hosting.ts) is polled with backoff while it wakes, for up to
 * WAKE_LIMIT_MS; requests that fail meanwhile wait for it and are sent again. A local backend that
 * does not answer is reported offline at once.
 */
import { apiFetch, type ApiFetchInit, getHealth, HEALTH_RECHECK_EVENT } from "./client";
import { backendMaySleep } from "./hosting";
import type { HealthResponse } from "./types";
import {
  FIRST_CHECK_TIMEOUT_MS,
  isWakeError,
  markWakeExhausted,
  SLOW_AFTER_MS,
  WAKE_CHECK_TIMEOUT_MS,
  WAKE_LIMIT_MS,
  wakeDelay,
} from "./wake";

/**
 * checking: the first health check is under way. waking: a hosted backend has not answered yet and
 * is being polled. online: the last check succeeded. offline: it failed (a local backend at once,
 * a hosted one after the wake limit).
 */
export type BackendPhase = "checking" | "waking" | "online" | "offline";

export interface BackendState {
  phase: BackendPhase;
  /** The last successful health response; kept through later failed checks, so a page keeps its mode. */
  health: HealthResponse | null;
  /** When the current wake-up started (ms since the epoch); null unless waking. */
  wakingSince: number | null;
}

export const INITIAL_BACKEND_STATE: BackendState = { phase: "checking", health: null, wakingSince: null };

/** Re-check when the window regains focus, at most this often. */
const RECHECK_AFTER_MS = 5000;
/** A wake-aware request is sent at most this many times. */
const MAX_REQUEST_ATTEMPTS = 4;

export interface BackendStoreOptions {
  fetchHealth: (signal: AbortSignal) => Promise<HealthResponse>;
  /** Whether a backend that does not answer may be asleep rather than down. */
  maySleep: () => boolean;
  now?: () => number;
}

export interface BackendStore {
  getState: () => BackendState;
  subscribe: (listener: () => void) => () => void;
  /** Start checking (once); listeners and waiters call it themselves. */
  start: () => void;
  /** Check now, starting a new wake-up if needed; a check already under way is restarted. */
  recheck: () => void;
  /** Check now unless a check is already under way. */
  requestCheck: () => void;
  /**
   * Resolves true once a check that started after the call succeeds, false when it fails for good
   * or `signal` aborts. Starts a check unless one is under way.
   */
  waitUntilOnline: (signal?: AbortSignal | null) => Promise<boolean>;
}

export function createBackendStore({ fetchHealth, maySleep, now = Date.now }: BackendStoreOptions): BackendStore {
  let state = INITIAL_BACKEND_STATE;
  const listeners = new Set<() => void>();
  const waiters = new Set<(online: boolean) => void>();
  let started = false;
  let cycle = 0;
  let checking = false;
  let lastCheckAt = 0;
  let retryTimer: ReturnType<typeof setTimeout> | undefined;
  let slowTimer: ReturnType<typeof setTimeout> | undefined;

  const emit = (patch: Partial<BackendState>) => {
    state = { ...state, ...patch };
    for (const listener of [...listeners]) listener();
  };

  const settle = (online: boolean) => {
    checking = false;
    clearTimeout(slowTimer);
    const pending = [...waiters];
    waiters.clear();
    for (const waiter of pending) waiter(online);
  };

  const runCycle = () => {
    const id = ++cycle;
    checking = true;
    clearTimeout(retryTimer);
    clearTimeout(slowTimer);
    const startedAt = now();
    const hosted = maySleep();
    let attempt = 0;

    // Slow to answer at all: on free hosting that is a backend waking up, so say so early.
    if (hosted && state.phase !== "waking") {
      slowTimer = setTimeout(() => {
        if (id === cycle && checking && state.phase !== "waking") emit({ phase: "waking", wakingSince: startedAt });
      }, SLOW_AFTER_MS);
    }

    const attemptOnce = () => {
      lastCheckAt = now();
      const controller = new AbortController();
      const timeout = setTimeout(
        () => controller.abort(),
        attempt === 0 ? FIRST_CHECK_TIMEOUT_MS : WAKE_CHECK_TIMEOUT_MS,
      );
      fetchHealth(controller.signal)
        .then((health) => {
          if (id !== cycle) return;
          emit({ phase: "online", health, wakingSince: null });
          settle(true);
        })
        .catch((error: unknown) => {
          if (id !== cycle) return;
          if (hosted && isWakeError(error) && now() - startedAt < WAKE_LIMIT_MS) {
            if (state.phase !== "waking") emit({ phase: "waking", wakingSince: startedAt });
            attempt += 1;
            retryTimer = setTimeout(attemptOnce, wakeDelay(attempt));
            return;
          }
          emit({ phase: "offline", wakingSince: null });
          settle(false);
        })
        .finally(() => clearTimeout(timeout));
    };
    attemptOnce();
  };

  const onFocus = () => {
    if (!checking && now() - lastCheckAt > RECHECK_AFTER_MS) runCycle();
  };
  const requestCheck = () => {
    if (!checking) runCycle();
  };

  const start = () => {
    if (started) return;
    started = true;
    if (typeof window !== "undefined") {
      window.addEventListener("focus", onFocus);
      // A request just failed in a way that suggests the backend is down.
      window.addEventListener(HEALTH_RECHECK_EVENT, requestCheck);
    }
    runCycle();
  };

  return {
    getState: () => state,
    subscribe(listener) {
      listeners.add(listener);
      start();
      return () => {
        listeners.delete(listener);
      };
    },
    start,
    recheck() {
      if (!started) start();
      else runCycle();
    },
    requestCheck() {
      if (!started) start();
      else requestCheck();
    },
    waitUntilOnline(signal) {
      return new Promise<boolean>((resolve) => {
        if (signal?.aborted) {
          resolve(false);
          return;
        }
        const onAbort = () => {
          waiters.delete(waiter);
          resolve(false);
        };
        const waiter = (online: boolean) => {
          signal?.removeEventListener("abort", onAbort);
          resolve(online);
        };
        signal?.addEventListener("abort", onAbort, { once: true });
        waiters.add(waiter);
        if (!started) start();
        else requestCheck();
      });
    },
  };
}

/**
 * apiFetch that rides out a sleeping backend: when the request fails the way a backend that is
 * still waking up fails (see isWakeError), it waits for the health check to succeed and sends the
 * request again, up to MAX_REQUEST_ATTEMPTS times. Only on hosting that sleeps, and only for
 * requests that are safe to repeat (reads; never one that spends OpenAI credits). An error thrown
 * because the backend never woke is marked (isWakeExhausted).
 */
export async function fetchWhenAwake<T>(
  store: Pick<BackendStore, "waitUntilOnline">,
  maySleep: () => boolean,
  request: () => Promise<T>,
  signal?: AbortSignal | null,
): Promise<T> {
  for (let attempt = 1; ; attempt += 1) {
    try {
      return await request();
    } catch (error) {
      if (signal?.aborted || !maySleep() || !isWakeError(error) || attempt >= MAX_REQUEST_ATTEMPTS) throw error;
      const online = await store.waitUntilOnline(signal);
      if (signal?.aborted) throw error;
      if (!online) {
        markWakeExhausted(error); // the health check gave up: callers do not start another one
        throw error;
      }
    }
  }
}

// --- The app's store ---------------------------------------------------------------------------

export const backendStore = createBackendStore({
  fetchHealth: (signal) => getHealth({ signal, cache: "no-store" }),
  maySleep: backendMaySleep,
});

/** GET (or idempotent POST) through the /api proxy, waiting out a sleeping demo backend. */
export function apiFetchAwake<T>(path: string, init: ApiFetchInit = {}): Promise<T> {
  return fetchWhenAwake(backendStore, backendMaySleep, () => apiFetch<T>(path, init), init.signal);
}

/** True on the public demo (no request calls OpenAI); null while the backend's mode is unknown. */
export function demoModeOf(state: Pick<BackendState, "health" | "phase">): boolean | null {
  if (state.health) return state.health.demo_mode === true;
  // A local backend that is down is not the demo; a hosted one that never answered is unknown.
  if (state.phase === "offline" && !backendMaySleep()) return false;
  return null;
}
