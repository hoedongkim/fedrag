"use client";

import { useCallback, useEffect, useState } from "react";

import { apiFetchAwake } from "@/lib/api/backend";
import { HEALTH_RECHECK_EVENT } from "@/lib/api/client";
import type { EvaluationReport, QuestionDetail } from "@/lib/api/types";
import { suggestsBackendDown } from "@/lib/stance/failures";

import type { Load } from "../stance/useStanceData";
import { useRetryWhenBackOnline } from "../useBackend";

const LOADING = { status: "loading" } as const;

/** The evaluation report, fetched once (it holds every number the page shows but passages). */
export function useEvaluationReport() {
  const [report, setReport] = useState<Load<EvaluationReport>>(LOADING);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    apiFetchAwake<EvaluationReport>("/api/evaluation", { signal: controller.signal })
      .then((data) => setReport({ status: "ready", data }))
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        if (suggestsBackendDown(error)) window.dispatchEvent(new Event(HEALTH_RECHECK_EVENT));
        setReport({ status: "error", error });
      });
    return () => controller.abort();
  }, [attempt]);

  const retry = useCallback(() => {
    setReport(LOADING);
    setAttempt((n) => n + 1);
  }, []);
  useRetryWhenBackOnline(report.status === "error", retry);

  return { report, retry };
}

/**
 * One question's drilldown, fetched when `id` changes; questions already seen come from memory,
 * so stepping back and forth through the list does not refetch.
 */
export function useQuestionDetail(id: string | null) {
  const [cache, setCache] = useState<ReadonlyMap<string, QuestionDetail>>(() => new Map());
  const [failure, setFailure] = useState<{ id: string; error: unknown } | null>(null);
  const [attempt, setAttempt] = useState(0);
  const cached = id ? cache.get(id) : undefined;

  useEffect(() => {
    if (!id || cached) return;
    const controller = new AbortController();
    apiFetchAwake<QuestionDetail>(`/api/evaluation/questions/${encodeURIComponent(id)}`, {
      signal: controller.signal,
    })
      .then((data) => setCache((current) => new Map(current).set(id, data)))
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        if (suggestsBackendDown(error)) window.dispatchEvent(new Event(HEALTH_RECHECK_EVENT));
        setFailure({ id, error });
      });
    return () => controller.abort();
  }, [id, cached, attempt]);

  const retry = useCallback(() => {
    setFailure(null);
    setAttempt((n) => n + 1);
  }, []);

  let detail: Load<QuestionDetail> | null = null;
  if (id) {
    if (cached) detail = { status: "ready", data: cached };
    else if (failure?.id === id) detail = { status: "error", error: failure.error };
    else detail = LOADING;
  }
  useRetryWhenBackOnline(detail?.status === "error", retry);
  return { detail, retry };
}
