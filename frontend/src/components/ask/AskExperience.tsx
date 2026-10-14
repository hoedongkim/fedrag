"use client";

import { useSearchParams } from "next/navigation";
import { useEffect, useRef, useState, useSyncExternalStore } from "react";

import type { StreamStage } from "@/lib/api/types";
import type { AskStatus } from "@/lib/ask/answerState";
import type { ExampleQuestion } from "@/lib/ask/examples";
import { localToday } from "@/lib/ask/format";
import {
  type AskField,
  type AskFormValues,
  type FieldErrors,
  fieldErrorsFromValidation,
  parseAskParams,
  serializeAskParams,
  validateAskForm,
} from "@/lib/ask/params";

import { AnswerPanel } from "./AnswerPanel";
import { ASK_FIELD_IDS, AskForm } from "./AskForm";
import { AskAbout } from "./AskAbout";
import { useAskStream } from "./useAskStream";

const URL_SYNC_DELAY_MS = 300;
const FIELD_ORDER: AskField[] = ["question", "asOf", "method", "topK"];

const subscribeNever = () => () => {};

/** Today's local date (YYYY-MM-DD) on the client; null while server rendering and hydrating. */
function useToday(): string | null {
  return useSyncExternalStore(subscribeNever, () => localToday(), () => null);
}

function prefersReducedMotion(): boolean {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/**
 * Ask the Fed: the question form, the streamed answer and its sources. The form state lives in
 * the URL (q, as_of, method, k), so a link reproduces the question; opening one only fills the
 * form, since asking spends OpenAI credits.
 */
export function AskExperience() {
  const searchParams = useSearchParams();
  const [values, setValues] = useState<AskFormValues>(() => parseAskParams(searchParams));
  const [clientErrors, setClientErrors] = useState<FieldErrors>({});
  const { state, ask, stop } = useAskStream();
  const today = useToday();
  const resultsRef = useRef<HTMLDivElement>(null);

  const streaming = state.status === "streaming";
  const failure = state.failure;
  const rejected = state.status === "failed" && failure?.kind === "http" && failure.status === 422;
  const errors: FieldErrors = rejected
    ? { ...fieldErrorsFromValidation(failure.body), ...clientErrors }
    : clientErrors;

  // Mirror the form in the query string, without a navigation or a history entry per keystroke.
  useEffect(() => {
    const timer = window.setTimeout(() => {
      const query = serializeAskParams(values);
      const { pathname, search, hash } = window.location;
      const next = query ? `?${query}` : "";
      if (next !== search) window.history.replaceState(null, "", `${pathname}${next}${hash}`);
    }, URL_SYNC_DELAY_MS);
    return () => window.clearTimeout(timer);
  }, [values]);

  const update = (patch: Partial<AskFormValues>) => {
    setValues((current) => ({ ...current, ...patch }));
    // Editing a field clears its message; the rest wait for the next submit.
    const edited = Object.keys(patch) as AskField[];
    if (edited.some((field) => clientErrors[field])) {
      setClientErrors((current) => {
        const next = { ...current };
        for (const field of edited) delete next[field];
        return next;
      });
    }
  };

  const revealResults = () => {
    window.requestAnimationFrame(() => {
      const results = resultsRef.current;
      if (results && results.getBoundingClientRect().top > window.innerHeight * 0.7) {
        results.scrollIntoView({ behavior: prefersReducedMotion() ? "auto" : "smooth", block: "start" });
      }
    });
  };

  const submit = () => {
    if (streaming) return;
    const result = validateAskForm(values, today);
    if (!result.ok) {
      setClientErrors(result.errors);
      const first = FIELD_ORDER.find((field) => result.errors[field]);
      if (first) document.getElementById(ASK_FIELD_IDS[first])?.focus();
      return;
    }
    setClientErrors({});
    if (ask(result.request)) revealResults();
  };

  const retry = () => {
    if (state.request) ask(state.request);
  };

  const retryHybrid = () => {
    if (!state.request) return;
    setValues((current) => ({ ...current, method: "hybrid" }));
    ask({ ...state.request, retrieval: "hybrid" });
  };

  const applyExample = (example: ExampleQuestion) => {
    update({ question: example.question, asOf: example.asOf ?? "" });
    const textarea = document.getElementById(ASK_FIELD_IDS.question);
    if (textarea instanceof HTMLTextAreaElement) {
      textarea.focus();
      window.requestAnimationFrame(() => {
        const end = textarea.value.length;
        textarea.setSelectionRange(end, end);
      });
    }
  };

  const showPanel = state.status !== "idle" && !rejected;

  return (
    <div className="mt-10 max-w-3xl">
      <AskForm
        values={values}
        errors={errors}
        today={today}
        busy={streaming}
        onChange={update}
        onSubmit={submit}
        onExample={applyExample}
      />

      <div ref={resultsRef} className={`scroll-mt-6 ${showPanel ? "mt-14" : ""}`}>
        {showPanel && (
          <AnswerPanel
            key={state.run}
            state={state}
            onStop={stop}
            onRetry={retry}
            onRetryHybrid={retryHybrid}
          />
        )}
      </div>

      <AskAbout recorded={false} />

      <p role="status" aria-live="polite" className="sr-only">
        {announcement(state.status, state.stage, state.result?.citations.length ?? 0, rejected)}
      </p>
    </div>
  );
}

/** What the live region says: stage changes and the outcome, never individual tokens. */
function announcement(status: AskStatus, stage: StreamStage, cited: number, rejected: boolean): string {
  switch (status) {
    case "streaming":
      return stage === "retrieving" ? "Searching documents…" : "Writing answer…";
    case "done":
      return cited === 0
        ? "Done. The documents do not support an answer to this question."
        : `Answer ready, with ${cited} ${cited === 1 ? "citation" : "citations"}.`;
    case "stopped":
      return "Stopped. The answer is incomplete.";
    case "failed":
      return rejected ? "The question was not accepted. Check the form." : "";
    default:
      return "";
  }
}
