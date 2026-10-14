"use client";

import { useCallback, useEffect, useState } from "react";

import { apiFetchAwake } from "@/lib/api/backend";
import { HEALTH_RECHECK_EVENT } from "@/lib/api/client";
import type { StanceBenchmark, StanceClassifier, StanceTimeline } from "@/lib/api/types";
import { CLASSIFIER_IDS } from "@/lib/stance/classifiers";
import { suggestsBackendDown } from "@/lib/stance/failures";

import { useRetryWhenBackOnline } from "../useBackend";

export type Load<T> =
  | { status: "loading" }
  | { status: "ready"; data: T }
  | { status: "error"; error: unknown };

export type TimelineLoads = Record<StanceClassifier, Load<StanceTimeline>>;

const LOADING = { status: "loading" } as const;

function allLoading(): TimelineLoads {
  return Object.fromEntries(CLASSIFIER_IDS.map((id) => [id, LOADING])) as TimelineLoads;
}

/**
 * Every classifier's full timeline (2015 onwards, about 100 meetings each) and the benchmark,
 * fetched once in parallel: the chart's year filter and the meeting panel then work without
 * further requests. Each request succeeds or fails on its own, so one classifier that is not
 * loaded yet does not hide the others.
 */
export function useStanceData() {
  const [timelines, setTimelines] = useState<TimelineLoads>(allLoading);
  const [benchmark, setBenchmark] = useState<Load<StanceBenchmark>>(LOADING);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    const { signal } = controller;
    const failed = (error: unknown) => {
      if (suggestsBackendDown(error)) window.dispatchEvent(new Event(HEALTH_RECHECK_EVENT));
    };

    for (const id of CLASSIFIER_IDS) {
      apiFetchAwake<StanceTimeline>(`/api/stance?classifier=${id}`, { signal })
        .then((data) => setTimelines((current) => ({ ...current, [id]: { status: "ready", data } })))
        .catch((error: unknown) => {
          if (signal.aborted) return;
          failed(error);
          setTimelines((current) => ({ ...current, [id]: { status: "error", error } }));
        });
    }
    apiFetchAwake<StanceBenchmark>("/api/stance/benchmark", { signal })
      .then((data) => setBenchmark({ status: "ready", data }))
      .catch((error: unknown) => {
        if (!signal.aborted) setBenchmark({ status: "error", error });
      });

    return () => controller.abort();
  }, [attempt]);

  const retry = useCallback(() => {
    setTimelines(allLoading());
    setBenchmark(LOADING);
    setAttempt((n) => n + 1);
  }, []);
  const failed =
    benchmark.status === "error" || Object.values(timelines).some((load) => load.status === "error");
  useRetryWhenBackOnline(failed, retry);

  return { timelines, benchmark, retry };
}
