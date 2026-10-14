"use client";

import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from "react";

import { apiFetchAwake } from "@/lib/api/backend";
import { ApiError, apiFetch, HEALTH_RECHECK_EVENT } from "@/lib/api/client";
import type { CompareRequest, CompareResponse, MeetingSummary } from "@/lib/api/types";
import { suggestsBackendDown } from "@/lib/stance/failures";

import type { Load } from "../stance/useStanceData";
import { useRetryWhenBackOnline } from "../useBackend";

const LOADING = { status: "loading" } as const;

/** Re-check health when the backend may be down, or (403) turns out to be the public demo. */
function reportFailure(error: unknown) {
  const demo = error instanceof ApiError && error.status === 403;
  if (demo || suggestsBackendDown(error)) window.dispatchEvent(new Event(HEALTH_RECHECK_EVENT));
}

/**
 * POST /api/compare. A "cached_only" request never calls the LLM, so it may be repeated while a
 * sleeping demo backend wakes; "generate" is sent once.
 */
export function postCompare(request: CompareRequest, signal?: AbortSignal): Promise<CompareResponse> {
  const init = { method: "POST", json: request, signal, cache: "no-store" } as const;
  return request.analysis === "generate"
    ? apiFetch<CompareResponse>("/api/compare", init)
    : apiFetchAwake<CompareResponse>("/api/compare", init);
}

/** Every meeting with a statement, oldest first, fetched once (about 100 rows). */
export function useStatements() {
  const [load, setLoad] = useState<Load<MeetingSummary[]>>(LOADING);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    apiFetchAwake<MeetingSummary[]>("/api/meetings?doc_type=statement", { signal: controller.signal })
      .then((data) => {
        const sorted = [...data].sort((a, b) => a.meeting_date.localeCompare(b.meeting_date));
        setLoad({ status: "ready", data: sorted });
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        reportFailure(error);
        setLoad({ status: "error", error });
      });
    return () => controller.abort();
  }, [attempt]);

  const retry = useCallback(() => {
    setLoad(LOADING);
    setAttempt((n) => n + 1);
  }, []);
  useRetryWhenBackOnline(load.status === "error", retry);

  return { load, retry };
}

export type Generation =
  | { status: "idle" }
  | { status: "generating" }
  | { status: "error"; error: unknown };

const IDLE: Generation = { status: "idle" };

/**
 * The comparison of one pair. Loading a pair only ever asks for a cached analysis
 * ("cached_only"), so opening or sharing a link never spends OpenAI credits; `generate` is the
 * one place that asks the backend to write an analysis (one LLM call, then stored). State is
 * keyed by the pair, so a response for an earlier pair never shows under a later one.
 */
export function useComparison(base: string | null, target: string | null) {
  const key = base && target ? `${base}|${target}` : null;
  const [current, setCurrent] = useState<{ key: string; load: Load<CompareResponse> } | null>(null);
  const [generation, setGeneration] = useState<{ key: string; state: Generation } | null>(null);
  const [attempt, setAttempt] = useState(0);
  const generateRef = useRef<AbortController | null>(null);

  useEffect(() => {
    if (!base || !target) return;
    const pairKey = `${base}|${target}`;
    const controller = new AbortController();
    postCompare({ base_date: base, target_date: target, analysis: "cached_only" }, controller.signal)
      .then((data) => setCurrent({ key: pairKey, load: { status: "ready", data } }))
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        reportFailure(error);
        setCurrent({ key: pairKey, load: { status: "error", error } });
      });
    return () => {
      controller.abort();
      // Stop waiting for an analysis of the pair being left. The backend may still finish and
      // store it, so it is there next time.
      generateRef.current?.abort();
    };
  }, [base, target, attempt]);

  const retry = useCallback(() => {
    setCurrent(null);
    setAttempt((n) => n + 1);
  }, []);
  // Only the cached_only load: a failed generation is never sent again by itself.
  useRetryWhenBackOnline(current !== null && current.key === key && current.load.status === "error", retry);

  const generate = useCallback(() => {
    if (!base || !target) return;
    const pairKey = `${base}|${target}`;
    generateRef.current?.abort();
    const controller = new AbortController();
    generateRef.current = controller;
    setGeneration({ key: pairKey, state: { status: "generating" } });
    postCompare({ base_date: base, target_date: target, analysis: "generate" }, controller.signal)
      .then((data) => {
        setCurrent({ key: pairKey, load: { status: "ready", data } });
        setGeneration(null);
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        reportFailure(error);
        setGeneration({ key: pairKey, state: { status: "error", error } });
      });
  }, [base, target]);

  return {
    load: current && current.key === key ? current.load : LOADING,
    generation: generation && generation.key === key ? generation.state : IDLE,
    generate,
    retry,
  };
}

/** Whether a media query matches, kept in sync; `serverValue` while server rendering. */
export function useMediaQuery(query: string, serverValue = true): boolean {
  const subscribe = useCallback(
    (onChange: () => void) => {
      const list = window.matchMedia(query);
      list.addEventListener("change", onChange);
      return () => list.removeEventListener("change", onChange);
    },
    [query],
  );
  return useSyncExternalStore(
    subscribe,
    () => window.matchMedia(query).matches,
    () => serverValue,
  );
}
