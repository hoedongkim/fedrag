"use client";

import { useSearchParams } from "next/navigation";
import { useCallback, useEffect, useId, useMemo, useRef, useState } from "react";

import { apiFetchAwake } from "@/lib/api/backend";
import { HEALTH_RECHECK_EVENT } from "@/lib/api/client";
import { formatDate } from "@/lib/ask/format";
import {
  describeRecordedError,
  parseRecordedAnswers,
  pickRecorded,
  type RecordedAnswer,
  recordedAskState,
} from "@/lib/ask/recorded";
import { TYPE_LABELS } from "@/lib/performance/format";
import { DOCS } from "@/lib/site";
import { suggestsBackendDown } from "@/lib/stance/failures";

import { FailureNotice } from "../stance/FailureNotice";
import type { Load } from "../stance/useStanceData";
import { useRetryWhenBackOnline } from "../useBackend";
import { ExternalLink } from "../ui/ExternalLink";
import { fieldClass, skeletonClass } from "../ui/styles";
import { AnswerPanel } from "./AnswerPanel";
import { AskAbout } from "./AskAbout";

const LOADING = { status: "loading" } as const;

/** GET /api/ask/examples, fetched once (waiting out a sleeping demo backend). */
function useRecordedAnswers() {
  const [load, setLoad] = useState<Load<RecordedAnswer[]>>(LOADING);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    apiFetchAwake<unknown>("/api/ask/examples", { signal: controller.signal })
      .then((payload) => setLoad({ status: "ready", data: parseRecordedAnswers(payload) }))
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        if (suggestsBackendDown(error)) window.dispatchEvent(new Event(HEALTH_RECHECK_EVENT));
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

function typeLabel(type: string | null): string | null {
  if (!type) return null;
  return (TYPE_LABELS as Record<string, string>)[type] ?? type.replaceAll("_", " ");
}

/**
 * Ask the Fed on the public demo: live questions are off (they would spend OpenAI credits), so
 * the page shows answers the system recorded during the evaluation, in the same answer panel and
 * citation cards as a live answer. The chosen one is mirrored in the URL (`example`).
 */
export function RecordedAnswers() {
  const searchParams = useSearchParams();
  // A link to a live question (e.g. from Policy Stance) cannot be answered here: say so once.
  const [linkedQuestion] = useState(() => searchParams.get("q")?.trim() || null);
  const [selectedId, setSelectedId] = useState<string | null>(() => searchParams.get("example"));
  const { load, retry } = useRecordedAnswers();
  const answerRef = useRef<HTMLDivElement>(null);
  const pickerId = useId();

  const examples = useMemo(() => (load.status === "ready" ? load.data : []), [load]);
  const selected = pickRecorded(examples, selectedId);

  const choose = (id: string) => {
    setSelectedId(id);
    const { pathname, hash } = window.location;
    window.history.replaceState(null, "", `${pathname}?${new URLSearchParams({ example: id })}${hash}`);
    // On a phone the answer sits below the list: bring it into view.
    window.requestAnimationFrame(() => {
      const answer = answerRef.current;
      if (answer && answer.getBoundingClientRect().top > window.innerHeight * 0.7) {
        const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
        answer.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
      }
    });
  };

  return (
    <div className="mt-6">
      <DemoLine linkedQuestion={linkedQuestion} />

      <div className="mt-10">
        {load.status === "loading" && <PickerSkeleton />}
        {load.status === "error" && <FailureNotice failure={describeRecordedError(load.error)} onRetry={retry} />}
        {load.status === "ready" && examples.length === 0 && (
          <p className="text-body text-muted">No recorded answers are available.</p>
        )}

        {selected && (
          <div className="grid gap-x-14 gap-y-8 lg:grid-cols-[15rem_minmax(0,1fr)]">
            <section aria-labelledby="recorded-heading" className="min-w-0 lg:sticky lg:top-8 lg:self-start">
              <h2 id="recorded-heading" className="text-small font-semibold text-ink">
                Recorded questions
              </h2>
              {/* Below the wide layout the list would push the answer a screen down: a menu instead. */}
              <div className="mt-2 lg:hidden">
                <label htmlFor={pickerId} className="sr-only">
                  Recorded question
                </label>
                <select
                  id={pickerId}
                  value={selected.id}
                  onChange={(event) => choose(event.target.value)}
                  className={`${fieldClass} truncate`}
                >
                  {examples.map((example) => (
                    <option key={example.id} value={example.id}>
                      {example.question}
                    </option>
                  ))}
                </select>
              </div>
              <ul className="mt-3 hidden border-l border-line lg:block">
                {examples.map((example) => {
                  const active = example.id === selected.id;
                  const tag = example.label ?? typeLabel(example.type);
                  return (
                    <li key={example.id} className="min-w-0">
                      <button
                        type="button"
                        aria-pressed={active}
                        onClick={() => choose(example.id)}
                        className={[
                          "group -ml-px block w-full rounded-r-sm border-l-2 py-2 pr-1 pl-4 text-left transition-colors focus-visible:-outline-offset-2",
                          active ? "border-accent" : "border-transparent hover:border-line-strong",
                        ].join(" ")}
                      >
                        {(tag || example.asOf) && (
                          <span className={`block text-label ${active ? "text-accent" : "text-faint"}`}>
                            {[tag, example.asOf ? `as of ${formatDate(example.asOf)}` : null].filter(Boolean).join(" · ")}
                          </span>
                        )}
                        <span
                          className={`mt-0.5 line-clamp-2 text-small break-words lg:line-clamp-3 ${
                            active ? "text-ink" : "text-muted group-hover:text-ink"
                          }`}
                        >
                          {example.question}
                        </span>
                      </button>
                    </li>
                  );
                })}
              </ul>
            </section>

            <div ref={answerRef} className="min-w-0 scroll-mt-6">
              {selected.description && (
                <p className="mb-8 max-w-[70ch] text-small text-faint">
                  <span className="font-medium text-muted">Why this one: </span>
                  {selected.description}
                </p>
              )}
              <AnswerPanel
                key={selected.id}
                state={recordedAskState(selected)}
                recorded={{ recordedAt: selected.recordedAt, config: selected.config, judge: selected.judge }}
              />
            </div>
          </div>
        )}
      </div>

      <AskAbout recorded />
    </div>
  );
}

/** One quiet line: live questions are off on the public demo, and why. */
function DemoLine({ linkedQuestion }: { linkedQuestion: string | null }) {
  return (
    <div className="max-w-[46rem] space-y-2 text-small text-faint">
      <p>
        <span className="font-medium text-muted">Public demo:</span> live questions are off, so these are answers
        recorded in testing. <ExternalLink href={DOCS.quickstart}>Run it locally</ExternalLink> to ask your own.
      </p>
      {linkedQuestion && (
        <p className="break-words">
          The link you followed asks <q className="text-muted">{linkedQuestion}</q>; only a local copy of FedRAG can
          answer new questions.
        </p>
      )}
    </div>
  );
}

function PickerSkeleton() {
  return (
    <div className="grid gap-x-14 gap-y-12 lg:grid-cols-[15rem_minmax(0,1fr)]">
      <p className="sr-only" role="status">
        Loading the recorded answers…
      </p>
      <div aria-hidden="true" className="space-y-4 border-l border-line pl-4">
        {[0, 1, 2, 3, 4].map((i) => (
          <div key={i}>
            <div className={`h-2.5 w-20 ${skeletonClass}`} />
            <div className={`mt-2 h-3 w-full ${skeletonClass}`} />
            <div className={`mt-1.5 h-3 w-2/3 ${skeletonClass}`} />
          </div>
        ))}
      </div>
      <div aria-hidden="true" className="space-y-3">
        <div className={`h-3 w-16 ${skeletonClass}`} />
        <div className={`h-5 w-3/4 ${skeletonClass}`} />
        <div className="h-px bg-line" />
        {["w-11/12", "w-full", "w-4/5"].map((width) => (
          <div key={width} className={`h-3.5 ${width} ${skeletonClass}`} />
        ))}
      </div>
    </div>
  );
}
