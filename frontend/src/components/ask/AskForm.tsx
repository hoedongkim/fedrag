"use client";

import { type KeyboardEvent, useState, useSyncExternalStore } from "react";

import type { RetrievalMethod } from "@/lib/api/types";
import { EXAMPLE_QUESTIONS, type ExampleQuestion } from "@/lib/ask/examples";
import { formatDate } from "@/lib/ask/format";
import {
  AS_OF_MIN,
  type AskField,
  type AskFormValues,
  type FieldErrors,
  QUESTION_MAX,
  RETRIEVAL_METHODS,
  TOP_K_MAX,
  TOP_K_MIN,
  hasAdvancedOverrides,
  methodLabel,
} from "@/lib/ask/params";

import {
  errorTextClass,
  fieldBaseClass,
  fieldClass,
  helpClass,
  labelClass,
  primaryButtonClass,
  textButtonClass,
} from "../ui/styles";
import { ChevronRightIcon, Spinner } from "./icons";

/** Element ids of the form fields, for focusing the first invalid one. */
export const ASK_FIELD_IDS: Record<AskField, string> = {
  question: "ask-question",
  asOf: "ask-as-of",
  method: "ask-method",
  topK: "ask-top-k",
};

const TOP_K_OPTIONS = Array.from({ length: TOP_K_MAX - TOP_K_MIN + 1 }, (_, i) => TOP_K_MIN + i);
const QUESTION_MAX_LABEL = QUESTION_MAX.toLocaleString("en-US");

const subscribeNever = () => () => {};

/** "mac", "other", or null during server rendering and hydration. */
function usePlatform(): "mac" | "other" | null {
  return useSyncExternalStore(
    subscribeNever,
    () => (/Mac|iPhone|iPad/.test(navigator.userAgent) ? "mac" : "other"),
    () => null,
  );
}

interface AskFormProps {
  values: AskFormValues;
  errors: FieldErrors;
  /** Latest allowed as-of date (local YYYY-MM-DD); null until hydrated. */
  today: string | null;
  /** An answer is streaming: submitting is blocked. */
  busy: boolean;
  onChange: (patch: Partial<AskFormValues>) => void;
  onSubmit: () => void;
  onExample: (example: ExampleQuestion) => void;
}

function describedBy(...ids: (string | false | undefined)[]): string | undefined {
  const value = ids.filter(Boolean).join(" ");
  return value || undefined;
}

export function AskForm({ values, errors, today, busy, onChange, onSubmit, onExample }: AskFormProps) {
  const platform = usePlatform();
  const [advancedOpen, setAdvancedOpen] = useState(
    () => hasAdvancedOverrides(values) || Boolean(errors.method || errors.topK),
  );
  const showAdvanced = advancedOpen || Boolean(errors.method || errors.topK);

  const length = values.question.trim().length;
  const counterTone =
    length > QUESTION_MAX ? "font-medium text-bad" : length > QUESTION_MAX * 0.9 ? "text-warn" : "text-faint";

  const onQuestionKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey) && !event.nativeEvent.isComposing) {
      event.preventDefault();
      onSubmit();
    }
  };

  const ids = ASK_FIELD_IDS;

  return (
    <form
      noValidate
      aria-label="Ask a question"
      onSubmit={(event) => {
        event.preventDefault();
        onSubmit();
      }}
    >
      {errors.form && (
        <p role="alert" className={`mb-4 ${errorTextClass}`}>
          {errors.form}
        </p>
      )}

      <div className="flex items-baseline justify-between gap-3">
        <label htmlFor={ids.question} className={labelClass}>
          Your question
        </label>
        <span id="ask-question-count" className={`text-label tabular-nums ${counterTone}`}>
          {length.toLocaleString("en-US")} / {QUESTION_MAX_LABEL}
          <span className="sr-only"> characters</span>
        </span>
      </div>
      <textarea
        id={ids.question}
        name="q"
        rows={3}
        value={values.question}
        onChange={(event) => onChange({ question: event.target.value })}
        onKeyDown={onQuestionKeyDown}
        placeholder="e.g. How did the Committee describe inflation risks in 2022?"
        aria-invalid={errors.question ? true : undefined}
        aria-describedby={describedBy(errors.question && "ask-question-error", "ask-question-count")}
        className={`${fieldBaseClass} mt-2 max-h-80 min-h-28 resize-y px-3.5 py-3 text-prose field-sizing-content`}
      />
      {errors.question && (
        <p id="ask-question-error" role="alert" className={`mt-1.5 ${errorTextClass}`}>
          {errors.question}
        </p>
      )}

      <div className="mt-5 grid gap-x-10 gap-y-5 md:grid-cols-2">
        <div>
          <label htmlFor={ids.asOf} className={labelClass}>
            As of <span className="font-normal text-faint">(optional)</span>
          </label>
          <div className="mt-2 flex items-center gap-3">
            <input
              id={ids.asOf}
              name="as_of"
              type="date"
              min={AS_OF_MIN}
              max={today ?? undefined}
              value={values.asOf}
              onChange={(event) => onChange({ asOf: event.target.value })}
              aria-invalid={errors.asOf ? true : undefined}
              aria-describedby={describedBy(errors.asOf && "ask-as-of-error", "ask-as-of-help")}
              className={`${fieldClass} max-w-48`}
            />
            {values.asOf && (
              <button
                type="button"
                onClick={() => onChange({ asOf: "" })}
                className={textButtonClass}
                aria-label="Clear the as-of date"
              >
                Clear
              </button>
            )}
          </div>
          {errors.asOf && (
            <p id="ask-as-of-error" role="alert" className={`mt-1.5 ${errorTextClass}`}>
              {errors.asOf}
            </p>
          )}
          <p id="ask-as-of-help" className={`mt-1.5 ${helpClass}`}>
            Search only documents that were public on this date. Leave empty to search everything.
          </p>
        </div>

        <details
          open={showAdvanced}
          onToggle={(event) => setAdvancedOpen(event.currentTarget.open)}
          className="group min-w-0"
        >
          <summary className="flex w-fit list-none flex-wrap items-center gap-x-1.5 rounded-sm text-small font-medium text-ink select-none">
            <ChevronRightIcon className="size-3.5 text-faint transition-transform group-open:rotate-90 motion-reduce:transition-none" />
            Search settings
            <span className="font-normal text-faint">
              {methodLabel(values.method)} · {values.topK} passages
            </span>
          </summary>
          <div className="mt-3 grid gap-4 sm:grid-cols-2">
            <div>
              <label htmlFor={ids.method} className={labelClass}>
                Retrieval method
              </label>
              <select
                id={ids.method}
                name="method"
                value={values.method}
                onChange={(event) => onChange({ method: event.target.value as RetrievalMethod })}
                aria-invalid={errors.method ? true : undefined}
                aria-describedby={describedBy(errors.method && "ask-method-error", "ask-method-help")}
                className={`${fieldClass} mt-2`}
              >
                {RETRIEVAL_METHODS.map((method) => (
                  <option key={method.value} value={method.value}>
                    {method.label}
                  </option>
                ))}
              </select>
              {errors.method && (
                <p id="ask-method-error" role="alert" className={`mt-1.5 ${errorTextClass}`}>
                  {errors.method}
                </p>
              )}
              <p id="ask-method-help" className={`mt-1.5 ${helpClass}`}>
                Hybrid mixes keyword and meaning search; rerank re-scores the results.
              </p>
            </div>
            <div>
              <label htmlFor={ids.topK} className={labelClass}>
                Passages
              </label>
              <select
                id={ids.topK}
                name="k"
                value={values.topK}
                onChange={(event) => onChange({ topK: Number(event.target.value) })}
                aria-invalid={errors.topK ? true : undefined}
                aria-describedby={describedBy(errors.topK && "ask-top-k-error", "ask-top-k-help")}
                className={`${fieldClass} mt-2`}
              >
                {TOP_K_OPTIONS.map((k) => (
                  <option key={k} value={k}>
                    {k}
                  </option>
                ))}
              </select>
              {errors.topK && (
                <p id="ask-top-k-error" role="alert" className={`mt-1.5 ${errorTextClass}`}>
                  {errors.topK}
                </p>
              )}
              <p id="ask-top-k-help" className={`mt-1.5 ${helpClass}`}>
                How many passages the model reads.
              </p>
            </div>
          </div>
        </details>
      </div>

      <div className="mt-6 flex items-center justify-end gap-4">
        {platform && (
          <span className={`hidden pointer-fine:inline ${helpClass}`}>
            <kbd className="font-sans text-[1em]">{platform === "mac" ? "⌘" : "Ctrl"}</kbd>
            {" + "}
            <kbd className="font-sans text-[1em]">Enter</kbd> to ask
          </span>
        )}
        <button type="submit" aria-disabled={busy || undefined} className={`${primaryButtonClass} min-w-28`}>
          {busy ? (
            <>
              <Spinner className="size-4" />
              Answering…
            </>
          ) : (
            "Ask"
          )}
        </button>
      </div>

      <div className="mt-8 border-t border-line pt-5">
        <h2 id="ask-examples-heading" className="text-small font-medium text-ink">
          Examples <span className="font-normal text-faint">(fills the form only)</span>
        </h2>
        <ul aria-labelledby="ask-examples-heading" className="mt-3 flex flex-wrap gap-x-5 gap-y-2">
          {EXAMPLE_QUESTIONS.map((example) => (
            <li key={example.label} className="max-w-full">
              <button
                type="button"
                onClick={() => onExample(example)}
                title={example.question}
                className="max-w-full rounded-sm text-left text-small text-muted underline decoration-line-strong underline-offset-4 transition-colors hover:text-ink hover:decoration-current"
              >
                {example.label}
                {example.asOf && <span className="text-faint"> · as of {formatDate(example.asOf)}</span>}
              </button>
            </li>
          ))}
        </ul>
      </div>
    </form>
  );
}
