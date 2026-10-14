"use client";

import { useCallback, useEffect, useReducer, useRef } from "react";

import { HEALTH_RECHECK_EVENT } from "@/lib/api/client";
import { streamAsk } from "@/lib/api/stream";
import type { AskRequest } from "@/lib/api/types";
import { INITIAL_ASK_STATE, askReducer } from "@/lib/ask/answerState";
import { type AskFailure, failureFromError } from "@/lib/ask/failures";

/**
 * Failures after which the backend's health should be re-checked: it may be down, or (403) it is
 * the public demo, which the re-check reports so the page switches to recorded answers.
 */
function shouldRecheckHealth(failure: AskFailure): boolean {
  return (
    failure.kind === "network" ||
    failure.kind === "interrupted" ||
    (failure.kind === "http" && (failure.status >= 500 || failure.status === 403))
  );
}

/**
 * Streams answers from POST /api/ask/stream. One answer at a time: `ask` returns false while one
 * is in progress. `stop` aborts the request, which also makes the backend stop generating; the
 * text so far stays. Leaving the page (unmounting, or hiding the route) stops it too.
 */
export function useAskStream() {
  const [state, dispatch] = useReducer(askReducer, INITIAL_ASK_STATE);
  const controllerRef = useRef<AbortController | null>(null);

  const ask = useCallback((request: AskRequest): boolean => {
    if (controllerRef.current) return false;
    const controller = new AbortController();
    controllerRef.current = controller;
    const release = () => {
      if (controllerRef.current === controller) controllerRef.current = null;
    };

    dispatch({ type: "start", request });
    streamAsk(request, {
      signal: controller.signal,
      onEvent: (event) => {
        if (controller.signal.aborted) return;
        if (event.event === "done" || event.event === "error") release();
        dispatch({ type: "event", event });
      },
    })
      .catch((error: unknown) => {
        // A stop, or leaving the page, aborted it: the state already says so.
        if (controller.signal.aborted) return;
        const failure = failureFromError(error);
        dispatch({ type: "fail", failure });
        if (shouldRecheckHealth(failure)) window.dispatchEvent(new Event(HEALTH_RECHECK_EVENT));
      })
      .finally(release);
    return true;
  }, []);

  // Always dispatches: the reducer ignores a stop unless an answer is streaming.
  const stop = useCallback(() => {
    const controller = controllerRef.current;
    controllerRef.current = null;
    controller?.abort();
    dispatch({ type: "stop" });
  }, []);

  // Leaving the route hides it (cacheComponents keeps it in an <Activity>) or unmounts it: either
  // way the request is aborted and the answer ends as stopped.
  useEffect(() => stop, [stop]);

  return { state, ask, stop };
}
