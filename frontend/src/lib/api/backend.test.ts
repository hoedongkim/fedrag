import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { createBackendStore, demoModeOf, fetchWhenAwake } from "./backend";
import { ApiError } from "./client";
import type { HealthResponse } from "./types";
import { isWakeError, isWakeExhausted, SLOW_AFTER_MS, WAKE_LIMIT_MS, wakeDelay } from "./wake";

const HEALTH: HealthResponse = {
  status: "ok",
  version: "0.1.0",
  environment: "production",
  llm_configured: false,
  database_configured: true,
  vector_store_ready: false,
  demo_mode: true,
};

/** What a sleeping Space (or the proxy in front of it) answers: no FastAPI `detail`. */
const asleep = () => new ApiError(503, "<html>Starting</html>", "<html>Starting</html>");
const refused = () => new TypeError("fetch failed");

describe("isWakeError", () => {
  it("counts no response, timeouts and hosting pages as a backend not up yet", () => {
    expect(isWakeError(refused())).toBe(true);
    expect(isWakeError(new DOMException("aborted", "AbortError"))).toBe(true);
    expect(isWakeError(new DOMException("timed out", "TimeoutError"))).toBe(true);
    expect(isWakeError(asleep())).toBe(true);
    expect(isWakeError(new ApiError(500, "Internal Server Error", "Internal Server Error"))).toBe(true);
    expect(isWakeError(new ApiError(504, "Gateway Timeout"))).toBe(true);
    // A 200 HTML page where JSON was expected: a hosting page, not the API.
    expect(isWakeError(new ApiError(200, "Expected a JSON response from the API.", "<html>"))).toBe(true);
  });

  it("does not count errors the API answered itself", () => {
    const detail = "Database not configured: set DATABASE_URL in backend/.env.";
    expect(isWakeError(new ApiError(503, detail, { detail }))).toBe(false);
    expect(isWakeError(new ApiError(502, "OpenAI failed", { detail: "OpenAI failed" }))).toBe(false);
    expect(isWakeError(new ApiError(403, "Disabled on the demo", { detail: "x" }))).toBe(false);
    expect(isWakeError(new ApiError(404, "Not Found", "Not Found"))).toBe(false);
    expect(isWakeError(new ApiError(422, "bad", { detail: [] }))).toBe(false);
    expect(isWakeError(new Error("other"))).toBe(false);
  });
});

describe("wakeDelay", () => {
  it("backs off from 2 s and caps at 10 s", () => {
    expect([1, 2, 3, 4, 5, 9].map(wakeDelay)).toEqual([2000, 4000, 8000, 10000, 10000, 10000]);
    expect(wakeDelay(0)).toBe(2000);
  });
});

describe("demoModeOf", () => {
  it("reads demo_mode, and is unknown until the backend answers", () => {
    expect(demoModeOf({ phase: "online", health: HEALTH })).toBe(true);
    expect(demoModeOf({ phase: "online", health: { ...HEALTH, demo_mode: false } })).toBe(false);
    // An older backend without the field (not in today's contract) is not the demo.
    const { demo_mode: _omit, ...old } = HEALTH;
    void _omit;
    expect(demoModeOf({ phase: "online", health: old as HealthResponse })).toBe(false);
    expect(demoModeOf({ phase: "waking", health: null })).toBeNull();
    // A later failed check keeps the mode the page already shows.
    expect(demoModeOf({ phase: "offline", health: HEALTH })).toBe(true);
  });
});

describe("createBackendStore", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  /** A health endpoint that fails with `failures` errors in turn, then answers. */
  function scripted(failures: (() => unknown)[]) {
    const calls = { count: 0 };
    const fetchHealth = vi.fn(async () => {
      const failure = failures[calls.count++];
      if (failure) throw failure();
      return HEALTH;
    });
    return { fetchHealth, calls };
  }

  it("reports a local backend that does not answer as offline at once", async () => {
    const { fetchHealth } = scripted([refused]);
    const store = createBackendStore({ fetchHealth, maySleep: () => false });
    store.start();
    await vi.advanceTimersByTimeAsync(0);
    expect(store.getState().phase).toBe("offline");
    expect(fetchHealth).toHaveBeenCalledTimes(1);
  });

  it("polls a sleeping hosted backend with backoff until it answers", async () => {
    const { fetchHealth } = scripted([asleep, asleep, refused]);
    const store = createBackendStore({ fetchHealth, maySleep: () => true });
    const seen: string[] = [];
    store.subscribe(() => seen.push(store.getState().phase));
    await vi.advanceTimersByTimeAsync(0);
    expect(store.getState().phase).toBe("waking");
    expect(store.getState().wakingSince).not.toBeNull();
    await vi.advanceTimersByTimeAsync(wakeDelay(1)); // second check: still asleep
    expect(fetchHealth).toHaveBeenCalledTimes(2);
    await vi.advanceTimersByTimeAsync(wakeDelay(2)); // third: refused
    await vi.advanceTimersByTimeAsync(wakeDelay(3)); // fourth: up
    expect(fetchHealth).toHaveBeenCalledTimes(4);
    expect(store.getState()).toMatchObject({ phase: "online", health: HEALTH, wakingSince: null });
    expect(seen).toEqual(["waking", "online"]);
  });

  it("shows a slow first check as waking before it fails", async () => {
    const fetchHealth = vi.fn(
      (signal: AbortSignal) =>
        new Promise<HealthResponse>((_, reject) => {
          signal.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
        }),
    );
    const store = createBackendStore({ fetchHealth, maySleep: () => true });
    store.start();
    await vi.advanceTimersByTimeAsync(SLOW_AFTER_MS - 1);
    expect(store.getState().phase).toBe("checking");
    await vi.advanceTimersByTimeAsync(1);
    expect(store.getState().phase).toBe("waking");
  });

  it("gives up after the wake limit and reports offline", async () => {
    const fetchHealth = vi.fn(async () => {
      throw asleep();
    });
    const store = createBackendStore({ fetchHealth, maySleep: () => true });
    store.start();
    await vi.advanceTimersByTimeAsync(WAKE_LIMIT_MS + 20_000);
    expect(store.getState().phase).toBe("offline");
    const calls = fetchHealth.mock.calls.length;
    await vi.advanceTimersByTimeAsync(60_000);
    expect(fetchHealth.mock.calls.length).toBe(calls); // stopped polling
  });

  it("does not wait on an error the API answered itself", async () => {
    const detail = "Something the API said";
    const { fetchHealth } = scripted([() => new ApiError(500, detail, { detail })]);
    const store = createBackendStore({ fetchHealth, maySleep: () => true });
    store.start();
    await vi.advanceTimersByTimeAsync(0);
    expect(store.getState().phase).toBe("offline");
  });

  it("resolves waiters when the backend comes up, and false when it never does", async () => {
    const { fetchHealth } = scripted([asleep]);
    const store = createBackendStore({ fetchHealth, maySleep: () => true });
    const online = store.waitUntilOnline();
    await vi.advanceTimersByTimeAsync(wakeDelay(1));
    await expect(online).resolves.toBe(true);

    const down = createBackendStore({ fetchHealth: async () => Promise.reject(refused()), maySleep: () => false });
    await expect(down.waitUntilOnline()).resolves.toBe(false);
  });

  it("lets an aborted waiter go without waiting", async () => {
    const store = createBackendStore({ fetchHealth: async () => Promise.reject(asleep()), maySleep: () => true });
    const controller = new AbortController();
    const waiting = store.waitUntilOnline(controller.signal);
    controller.abort();
    await expect(waiting).resolves.toBe(false);
  });

  it("keeps the last health response through a failed re-check", async () => {
    const { fetchHealth } = scripted([undefined as never, refused]);
    const store = createBackendStore({ fetchHealth, maySleep: () => false });
    store.start();
    await vi.advanceTimersByTimeAsync(0);
    expect(store.getState().phase).toBe("online");
    store.recheck();
    await vi.advanceTimersByTimeAsync(0);
    expect(store.getState()).toMatchObject({ phase: "offline", health: HEALTH });
  });
});

describe("fetchWhenAwake", () => {
  const store = (online: boolean) => ({ waitUntilOnline: vi.fn(async () => online) });

  it("waits for a sleeping backend and sends the request again", async () => {
    let calls = 0;
    const request = async () => {
      calls += 1;
      if (calls < 3) throw asleep();
      return "data";
    };
    const s = store(true);
    await expect(fetchWhenAwake(s, () => true, request)).resolves.toBe("data");
    expect(s.waitUntilOnline).toHaveBeenCalledTimes(2);
  });

  it("gives up after a few attempts, when the backend stays down, or for real API errors", async () => {
    const always = async () => {
      throw asleep();
    };
    const s = store(true);
    await expect(fetchWhenAwake(s, () => true, always)).rejects.toBeInstanceOf(ApiError);
    expect(s.waitUntilOnline).toHaveBeenCalledTimes(3);

    await expect(fetchWhenAwake(store(false), () => true, always)).rejects.toBeInstanceOf(ApiError);

    const detail = "Database not configured";
    const apiError = async () => {
      throw new ApiError(503, detail, { detail });
    };
    const s3 = store(true);
    await expect(fetchWhenAwake(s3, () => true, apiError)).rejects.toMatchObject({ status: 503 });
    expect(s3.waitUntilOnline).not.toHaveBeenCalled();
  });

  it("marks an error thrown because the backend never woke, and only that one", async () => {
    const neverWoke = await fetchWhenAwake(store(false), () => true, async () => Promise.reject(asleep())).catch(
      (error: unknown) => error,
    );
    expect(isWakeExhausted(neverWoke)).toBe(true);

    const tooManyTries = await fetchWhenAwake(store(true), () => true, async () => Promise.reject(asleep())).catch(
      (error: unknown) => error,
    );
    expect(isWakeExhausted(tooManyTries)).toBe(false);

    const local = await fetchWhenAwake(store(false), () => false, async () => Promise.reject(refused())).catch(
      (error: unknown) => error,
    );
    expect(isWakeExhausted(local)).toBe(false);
    expect(isWakeExhausted(asleep())).toBe(false);
    expect(isWakeExhausted("not an object")).toBe(false);
  });

  it("never retries on a local backend or after the caller aborted", async () => {
    const s = store(true);
    await expect(fetchWhenAwake(s, () => false, async () => Promise.reject(refused()))).rejects.toBeInstanceOf(TypeError);
    const controller = new AbortController();
    controller.abort();
    await expect(
      fetchWhenAwake(s, () => true, async () => Promise.reject(refused()), controller.signal),
    ).rejects.toBeInstanceOf(TypeError);
    expect(s.waitUntilOnline).not.toHaveBeenCalled();
  });
});
