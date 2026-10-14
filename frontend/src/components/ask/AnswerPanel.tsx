"use client";

import { useEffect, useId, useMemo, useRef, useState } from "react";

import type { AskRequest, AskResponse, Citation, StreamStage } from "@/lib/api/types";
import type { AskState, AskStatus, CitationEntry } from "@/lib/ask/answerState";
import { describeFailure } from "@/lib/ask/failures";
import { formatDate, formatLatency } from "@/lib/ask/format";
import { AS_OF_MIN, TOP_K_DEFAULT, TOP_K_MAX, methodLabel } from "@/lib/ask/params";
import type { JudgeSummary } from "@/lib/ask/recorded";
import { formatTimestamp } from "@/lib/stance/format";

import { AnswerText } from "./AnswerText";
import { CitationCard } from "./CitationCard";
import { secondaryButtonClass } from "../ui/styles";
import { RetryIcon, Spinner, StopIcon } from "./icons";

const HIGHLIGHT_MS = 2000;

const STAGE_LABELS: Record<StreamStage, string> = {
  retrieving: "Searching documents…",
  generating: "Writing answer…",
};

const END_LABELS: Record<Exclude<AskStatus, "idle" | "streaming">, string> = {
  done: "Done",
  stopped: "Stopped",
  failed: "Failed",
};

function prefersReducedMotion(): boolean {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/** What is known about a recorded answer (the public demo shows these instead of live ones). */
export interface RecordedInfo {
  /** When it was generated (ISO date or date-time). */
  recordedAt: string | null;
  /** Retriever settings it was retrieved with, e.g. "tuned". */
  config: string | null;
  judge: JudgeSummary | null;
}

interface AnswerPanelProps {
  state: AskState;
  onStop?: () => void;
  onRetry?: () => void;
  /** Retry the same request with Hybrid retrieval (offered when the reranker is unavailable). */
  onRetryHybrid?: () => void;
  /** A recorded answer, shown as it was written: no streaming, no retry, and its judge verdict. */
  recorded?: RecordedInfo;
}

const noop = () => {};

/** The streamed answer with its stage, notes, metadata and errors, then its source passages. */
export function AnswerPanel({ state, onStop = noop, onRetry = noop, onRetryHybrid = noop, recorded }: AnswerPanelProps) {
  const { status, request, stage, text, citations, result, failure } = state;
  const streaming = status === "streaming";

  const baseId = useId();
  const headingId = `${baseId}-answer`;
  const sourcesId = `${baseId}-sources`;
  const cardId = (n: number) => `${baseId}-citation-${n}`;

  const headingRef = useRef<HTMLHeadingElement>(null);
  const stopRef = useRef<HTMLButtonElement>(null);
  const askAgainRef = useRef<HTMLButtonElement>(null);
  const stopFocused = useRef(false);
  const highlightTimer = useRef<number | undefined>(undefined);
  const [highlighted, setHighlighted] = useState<number | null>(null);

  const citationMap = useMemo(
    () => new Map<number, Citation>(citations.map(({ n, citation }) => [n, citation])),
    [citations],
  );

  // A retry from the previous panel removed the focused button: keep focus in the answer. (A
  // recorded answer is on screen from the start, so it leaves focus alone.)
  const isRecorded = recorded !== undefined;
  useEffect(() => {
    if (isRecorded) return;
    if (document.activeElement && document.activeElement !== document.body) return;
    (stopRef.current ?? headingRef.current)?.focus({ preventScroll: true });
  }, [isRecorded]);

  // The Stop button disappears when the answer ends: if it had focus, keep focus in the panel
  // instead of letting it fall back to <body>.
  useEffect(() => {
    if (streaming || !stopFocused.current) return;
    stopFocused.current = false;
    if (document.activeElement && document.activeElement !== document.body) return;
    if (status === "stopped") askAgainRef.current?.focus();
    else headingRef.current?.focus();
  }, [streaming, status]);

  useEffect(() => () => window.clearTimeout(highlightTimer.current), []);

  const showCitation = (n: number) => {
    const card = document.getElementById(cardId(n));
    if (!card) return;
    const rect = card.getBoundingClientRect();
    if (rect.top < 0 || rect.bottom > window.innerHeight) {
      card.scrollIntoView({
        behavior: prefersReducedMotion() ? "auto" : "smooth",
        block: rect.height > window.innerHeight * 0.8 ? "start" : "center",
      });
    }
    card.focus({ preventScroll: true });
    setHighlighted(n);
    window.clearTimeout(highlightTimer.current);
    highlightTimer.current = window.setTimeout(() => setHighlighted(null), HIGHLIGHT_MS);
  };

  const finished = status === "done" && result !== null;
  const unsupported = finished && result.citations.length === 0;
  const failed = status === "failed" && failure !== null;

  return (
    <div className="space-y-12">
      <section aria-labelledby={headingId} aria-busy={streaming} className="motion-safe:animate-fade-in">
        <div className="flex min-h-8 flex-wrap items-center justify-between gap-x-4 gap-y-2">
          <h2 id={headingId} ref={headingRef} tabIndex={-1} className="rounded-sm text-small font-semibold text-ink">
            Answer
          </h2>
          {/* Mounted for the whole answer, so the line keeps its height when Stop goes away. */}
          {status !== "idle" && (
            <div className="flex shrink-0 items-center gap-3">
              {status === "streaming" ? (
                <>
                  <StageIndicator stage={stage} />
                  <button
                    type="button"
                    ref={stopRef}
                    onClick={onStop}
                    onFocus={() => {
                      stopFocused.current = true;
                    }}
                    onBlur={(event) => {
                      // No relatedTarget: the button is being removed, or focus left the page.
                      if (event.relatedTarget) stopFocused.current = false;
                    }}
                    className={secondaryButtonClass}
                  >
                    <StopIcon className="size-3.5" />
                    Stop
                  </button>
                </>
              ) : (
                <EndIndicator
                  status={status}
                  label={recorded && status === "done" ? "Recorded" : END_LABELS[status]}
                />
              )}
            </div>
          )}
        </div>
        {request && <QuestionEcho request={request} />}

        <div className="mt-5 border-t border-line pt-5">
          {failed && (
            <FailureBanner
              failure={describeFailure(failure)}
              onRetry={onRetry}
              onRetryHybrid={onRetryHybrid}
            />
          )}

          {streaming && !text && <AnswerSkeleton />}

          {unsupported ? (
            <NoSupportedAnswer
              answer={text}
              asOf={result.as_of}
              topK={request?.top_k ?? TOP_K_DEFAULT}
              recorded={isRecorded}
            />
          ) : (
            text && (
              <div className={failed || status === "stopped" ? "mt-4" : undefined}>
                {(failed || status === "stopped") && (
                  <p className="mb-2 text-label font-medium text-faint">Partial answer</p>
                )}
                <AnswerText
                  text={text}
                  citations={citationMap}
                  streaming={streaming}
                  onCite={showCitation}
                  className={`max-w-[70ch] text-prose ${failed || status === "stopped" ? "text-muted" : "text-ink"}`}
                />
              </div>
            )
          )}

          {status === "stopped" && (
            <div className="mt-5 flex flex-wrap items-center gap-x-4 gap-y-2">
              <p className="text-small text-muted">
                {text
                  ? "Stopped. The answer above is incomplete."
                  : "Stopped before the answer started."}
              </p>
              <button type="button" ref={askAgainRef} onClick={onRetry} className={secondaryButtonClass}>
                <RetryIcon className="size-3.5" />
                Ask again
              </button>
            </div>
          )}

          {finished && result.notes.length > 0 && <Notes notes={result.notes} />}
          {finished && request && (
            <AnswerMeta result={result} request={request} cited={citations.length} recorded={recorded} />
          )}
          {finished && recorded?.judge && <JudgeVerdict judge={recorded.judge} />}
        </div>
      </section>

      {citations.length > 0 && (
        <section aria-labelledby={sourcesId}>
          <h2 id={sourcesId} className="text-small font-semibold text-ink">
            Sources <span className="font-normal text-faint tabular-nums">({citations.length})</span>
          </h2>
          <ol className="mt-3 divide-y divide-line border-y border-line">
            {citations.map(({ n, citation }: CitationEntry) => (
              <li key={n}>
                <CitationCard
                  n={n}
                  citation={citation}
                  id={cardId(n)}
                  method={request?.retrieval}
                  highlighted={highlighted === n}
                />
              </li>
            ))}
          </ol>
        </section>
      )}
    </div>
  );
}

function QuestionEcho({ request }: { request: AskRequest }) {
  return (
    <p className="mt-1 max-w-[70ch] text-[1.125rem] leading-snug font-medium tracking-[-0.01em] break-words text-ink">
      <span className="sr-only">Question: </span>
      {request.question}
      {request.as_of && (
        <span className="mt-1 block text-small font-normal tracking-normal text-faint">
          As of {formatDate(request.as_of)}
        </span>
      )}
    </p>
  );
}

function StageIndicator({ stage }: { stage: StreamStage }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-small text-muted">
      <Spinner className="size-3.5 text-accent" />
      {STAGE_LABELS[stage]}
    </span>
  );
}

const END_TONES: Record<Exclude<AskStatus, "idle" | "streaming">, string> = {
  done: "bg-good",
  stopped: "bg-warn",
  failed: "bg-bad",
};

function EndIndicator({ status, label }: { status: Exclude<AskStatus, "idle" | "streaming">; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-small text-muted">
      <span aria-hidden="true" className={`size-1.5 rounded-full ${END_TONES[status]}`} />
      {label}
    </span>
  );
}

function AnswerSkeleton() {
  return (
    <div aria-hidden="true" className="max-w-[70ch] space-y-3 py-1">
      {["w-11/12", "w-full", "w-4/5", "w-2/3"].map((width) => (
        <div key={width} className={`h-3.5 rounded-sm bg-line/70 motion-safe:animate-pulse ${width}`} />
      ))}
    </div>
  );
}

interface NoSupportedAnswerProps {
  answer: string;
  asOf: string | null;
  /** Passages requested (top_k): asking for more is suggested only below the maximum. */
  topK: number;
  /** A recorded answer: there is no form to change, so no suggestions. */
  recorded: boolean;
}

function NoSupportedAnswer({ answer, asOf, topK, recorded }: NoSupportedAnswerProps) {
  return (
    <div className="max-w-[70ch] border-l-2 border-line-strong pl-4">
      <h3 className="text-body font-semibold text-ink">No supported answer</h3>
      <p className="mt-1.5 text-prose text-ink">{answer}</p>
      {recorded ? (
        <p className="mt-3 text-small text-muted">
          The passages found did not support an answer, so FedRAG declined instead of guessing.
        </p>
      ) : (
        <SupportSuggestions asOf={asOf} topK={topK} />
      )}
    </div>
  );
}

function SupportSuggestions({ asOf, topK }: { asOf: string | null; topK: number }) {
  return (
    <>
      <p className="mt-3 text-small text-muted">
        The passages found do not support an answer, so FedRAG does not guess. You could:
      </p>
      <ul className="mt-1.5 list-disc space-y-1 pl-5 text-small text-muted marker:text-faint">
        <li>Rephrase with the Committee&rsquo;s own terms, such as &ldquo;federal funds rate&rdquo; or &ldquo;balance sheet&rdquo;.</li>
        {asOf && (
          <li>
            Move or clear the as-of date ({formatDate(asOf)}): documents released after it are not
            searched.
          </li>
        )}
        {topK < TOP_K_MAX && <li>Read more passages under Search settings.</li>}
        <li>
          Ask about {formatDate(AS_OF_MIN)} or later: the documents FedRAG searches start then.
        </li>
      </ul>
    </>
  );
}

function Notes({ notes }: { notes: string[] }) {
  return (
    <aside
      aria-label="Notes about this answer"
      className="mt-5 max-w-[70ch] border-l-2 border-line-strong pl-3 text-small text-muted"
    >
      {notes.length === 1 ? (
        <p>{notes[0]}</p>
      ) : (
        <ul className="space-y-1">
          {notes.map((note) => (
            <li key={note}>{note}</li>
          ))}
        </ul>
      )}
    </aside>
  );
}

interface AnswerMetaProps {
  result: AskResponse;
  request: AskRequest;
  cited: number;
  recorded?: RecordedInfo;
}

function AnswerMeta({ result, request, cited, recorded }: AnswerMetaProps) {
  const retrieval = methodLabel(result.retrieval);
  const items = [
    recorded?.recordedAt ? { label: "Recorded", value: formatTimestamp(recorded.recordedAt) } : null,
    { label: "Model", value: result.model || "Not recorded" },
    { label: "Search", value: recorded?.config ? `${retrieval} (${recorded.config})` : retrieval },
    { label: "Cutoff", value: result.as_of ? `As of ${formatDate(result.as_of)}` : "None" },
    Number.isFinite(result.latency_ms) ? { label: "Time", value: formatLatency(result.latency_ms) } : null,
    { label: "Passages", value: `${cited} cited of up to ${request.top_k ?? TOP_K_DEFAULT}` },
  ].filter((item): item is { label: string; value: string } => item !== null);
  return (
    <dl className="mt-6 flex flex-wrap gap-x-6 gap-y-1.5 border-t border-line pt-4 text-small">
      {items.map(({ label, value }) => (
        <div key={label} className="flex min-w-0 items-baseline gap-1.5">
          <dt className="text-faint">{label}</dt>
          <dd className={`break-all text-muted tabular-nums ${label === "Model" ? "font-mono text-label" : ""}`}>{value}</dd>
        </div>
      ))}
    </dl>
  );
}

/** The LLM judge's verdict on a recorded answer, from the evaluation (its caveat is in AskAbout). */
function JudgeVerdict({ judge }: { judge: JudgeSummary }) {
  return (
    <div className="mt-3 text-small">
      <div className="flex flex-wrap gap-x-6 gap-y-1.5">
        <h3 className="text-faint">Judge{judge.model ? ` (${judge.model})` : ""}</h3>
        {judge.items.length > 0 && (
          <dl className="flex flex-wrap gap-x-6 gap-y-1.5">
            {judge.items.map(({ label, value }) => (
              <div key={label} className="flex min-w-0 flex-wrap gap-x-1.5">
                <dt className="text-faint">{label}</dt>
                <dd className="font-medium text-ink tabular-nums">{value}</dd>
              </div>
            ))}
          </dl>
        )}
      </div>
      {judge.text && <p className="mt-2 max-w-[70ch] text-muted">{judge.text}</p>}
    </div>
  );
}

interface FailureBannerProps {
  failure: ReturnType<typeof describeFailure>;
  onRetry: () => void;
  onRetryHybrid: () => void;
}

function FailureBanner({ failure, onRetry, onRetryHybrid }: FailureBannerProps) {
  const setup = failure.tone === "setup";
  return (
    <div role="alert" className={`max-w-[70ch] border-l-2 py-0.5 pl-4 ${setup ? "border-warn" : "border-bad"}`}>
      <h3 className="text-body font-semibold text-ink">{failure.title}</h3>
      <p className="mt-1 text-body text-muted">
        {failure.message}
        {failure.command && (
          <>
            {" "}
            <code className="rounded-sm bg-subtle px-1.5 py-0.5 text-ink ring-1 ring-line">{failure.command}</code>
          </>
        )}
      </p>
      {failure.detail && <p className="mt-2 font-mono text-label break-words text-faint">{failure.detail}</p>}
      {failure.action !== "none" && (
        <div className="mt-3 flex flex-wrap gap-2">
          {failure.action === "retry-hybrid" && (
            <button type="button" onClick={onRetryHybrid} className={secondaryButtonClass}>
              <RetryIcon className="size-3.5" />
              Retry with Hybrid
            </button>
          )}
          <button type="button" onClick={onRetry} className={secondaryButtonClass}>
            <RetryIcon className="size-3.5" />
            {failure.action === "retry-hybrid" ? "Retry with rerank" : "Retry"}
          </button>
        </div>
      )}
    </div>
  );
}
