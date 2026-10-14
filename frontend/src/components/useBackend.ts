"use client";

import { useEffect, useSyncExternalStore } from "react";

import { type BackendState, backendStore, demoModeOf, INITIAL_BACKEND_STATE } from "@/lib/api/backend";

const getServerState = () => INITIAL_BACKEND_STATE;

/** The backend's phase and last health response, shared by every component (one health check). */
export function useBackend(): BackendState {
  return useSyncExternalStore(backendStore.subscribe, backendStore.getState, getServerState);
}

/**
 * Calls `retry` when the backend comes back (offline or waking, then online) while `failed` is
 * true, so a page whose data failed to load while the backend was down fills in by itself, as the
 * waking notice says. Not after the first check ("checking"): a page that failed for a reason of
 * its own is not refetched on load. `retry` should be stable (useCallback).
 */
export function useRetryWhenBackOnline(failed: boolean, retry: () => void): void {
  useEffect(() => {
    if (!failed) return;
    let previous = backendStore.getState().phase;
    return backendStore.subscribe(() => {
      const { phase } = backendStore.getState();
      if (phase === "online" && (previous === "offline" || previous === "waking")) retry();
      previous = phase;
    });
  }, [failed, retry]);
}

/** True on the public demo, false otherwise, null until the backend has answered. */
export function useDemoMode(): boolean | null {
  return demoModeOf(useBackend());
}
