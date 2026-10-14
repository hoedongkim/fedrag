"use client";

import Link from "next/link";
import { useEffect, useId, useMemo, useRef } from "react";

import type { AnalysisStatus, CompareResponse, CompareStance, ComparisonAnalysis } from "@/lib/api/types";
import { formatDate } from "@/lib/ask/format";
import { analysisStatusNotice, describeGenerateError } from "@/lib/compare/failures";
import { compareAskHref, previousStatement } from "@/lib/compare/params";
import { changeAnchor } from "@/lib/compare/rows";
import { classifierInfo } from "@/lib/stance/classifiers";
import { formatConfidence, formatScore } from "@/lib/stance/format";

import { ArrowRightIcon, Spinner } from "../ask/icons";
import { FailureNotice } from "../stance/FailureNotice";
import { StanceLabel } from "../stance/StanceLabel";
import {
  helpClass,
  labelTextClass,
  linkClass,
  primaryButtonClass,
  secondaryButtonClass,
  sectionHeadingClass,
  skeletonClass,
  textButtonClass,
} from "../ui/styles";
import { ShiftLabel } from "./ShiftLabel";
import type { Generation } from "./useCompareData";

interface AnalysisPanelProps {
  data: CompareResponse;
  generation: Generation;
  /** The public demo, which never writes analyses; null while the backend's mode is unknown. */
  demo: boolean | null;
  /** Every statement date, oldest first (null while loading): for the previous-meeting fix. */
  dates: readonly string[] | null;
  /** Ask the backend to write the analysis: one OpenAI call (two if the reply cannot be parsed), then stored. */
  onGenerate: () => void;
  /** Scroll to change `id` in the diff and highlight `group`. */
  onJump: (id: number, group: number[]) => void;
  onChoose: (base: string, target: string) => void;
}

/** Statuses for which the demo has nothing to show but the diff (no analysis is stored). */
const NOT_STORED: readonly AnalysisStatus[] = ["not_cached", "llm_not_configured", "disabled_on_demo"];

/**
 * The LLM's reading of the change (stance shift, summary, key changes linked to their rows), or
 * the way to get one, plus the zero-shot classifier's scores of the two statements as a
 * cross-check and a link to Ask the Fed.
 */
export function AnalysisPanel({ data, generation, demo, dates, onGenerate, onJump, onChoose }: AnalysisPanelProps) {
  const headingId = useId();
  const headingRef = useRef<HTMLHeadingElement>(null);
  const requested = useRef(false);
  const { analysis, base, target, stance } = data;
  const status: AnalysisStatus = data.analysis_status;
  const generating = generation.status === "generating";
  const ready = status === "ready" && analysis !== null;
  // A backend that says so is the demo, whatever the (possibly stale) health check said.
  const onDemo = demo === true || status === "disabled_on_demo";
  const notice =
    generation.status === "error" ? describeGenerateError(generation.error) : analysisStatusNotice(status);

  // The button or Retry that asked for the analysis goes away when it arrives or fails: keep
  // focus in the panel instead of letting it fall back to <body>.
  useEffect(() => {
    if (!requested.current) return;
    if (!document.activeElement || document.activeElement === document.body) headingRef.current?.focus();
    if (!generating) requested.current = false;
  }, [ready, generating, generation.status]);

  const request = () => {
    if (generating) return;
    requested.current = true;
    onGenerate();
  };

  return (
    <section aria-labelledby={headingId} aria-busy={generating}>
      <h2 id={headingId} ref={headingRef} tabIndex={-1} className={`rounded-sm ${sectionHeadingClass}`}>
        What the change signals
      </h2>
      <p className="mt-1 text-body text-muted">
        An LLM&rsquo;s reading of the sentences that changed, as of {formatDate(target.release_date)}.
      </p>

      <div className="mt-6">
        {ready ? (
          <AnalysisBody
            analysis={analysis}
            validIds={data.changes.map((c) => c.id)}
            onJump={onJump}
            stance={stance}
          />
        ) : (
          <>
            {onDemo && NOT_STORED.includes(status) ? (
              <DemoNotStored base={base.meeting_date} target={target.meeting_date} dates={dates} onChoose={onChoose} />
            ) : demo === null && status === "not_cached" && generation.status === "idle" ? (
              // Until health says whether this is the demo, offer neither Generate nor the demo's note.
              <div aria-hidden="true" className={`h-16 max-w-[70ch] ${skeletonClass}`} />
            ) : generating || (!notice && status === "not_cached") ? (
              <GeneratePrompt generating={generating} onGenerate={request} />
            ) : notice ? (
              <FailureNotice failure={notice} onRetry={onDemo ? undefined : request} />
            ) : null}
            <dl className="mt-6 text-small">
              <StanceCrossCheck stance={stance} />
            </dl>
          </>
        )}
      </div>

      {/* The demo cannot answer new questions, so it does not offer to ask one. */}
      {!onDemo && (
        <p className="mt-8 text-small">
          <Link
            href={compareAskHref(base.meeting_date, target.meeting_date, target.release_date)}
            className={`${textButtonClass} inline-flex items-center gap-1`}
          >
            Ask about this change
            <ArrowRightIcon className="size-3.5" />
          </Link>
        </p>
      )}
    </section>
  );
}

function AnalysisBody({
  analysis,
  validIds,
  onJump,
  stance,
}: {
  analysis: ComparisonAnalysis;
  validIds: number[];
  onJump: (id: number, group: number[]) => void;
  stance: CompareStance;
}) {
  const valid = useMemo(() => new Set(validIds), [validIds]);
  return (
    <>
      <ShiftLabel shift={analysis.stance_shift} className="text-heading" />
      <p className="mt-3 max-w-[70ch] text-prose text-ink">{analysis.summary}</p>
      <dl className="mt-5 flex flex-wrap gap-x-10 gap-y-3 text-small">
        <div>
          <dt className={labelTextClass}>Confidence</dt>
          <dd className="mt-0.5 font-medium text-ink tabular-nums" title="Reported by the model, not calibrated">
            {formatConfidence(analysis.confidence)}
          </dd>
        </div>
        <StanceCrossCheck stance={stance} />
      </dl>

      <h3 className="mt-12 text-body font-semibold text-ink">Key changes</h3>
      {analysis.key_changes.length > 0 ? (
        <ol className="mt-3 divide-y divide-line border-y border-line">
          {analysis.key_changes.map((item, i) => {
            const ids = item.change_ids.filter((id) => valid.has(id));
            return (
              <li key={i} className="grid gap-x-8 gap-y-2 py-5 sm:grid-cols-[7.5rem_minmax(0,1fr)]">
                <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1 text-small sm:block">
                  <span className="sr-only">Direction:</span>
                  <StanceLabel stance={item.direction} />
                  {ids.length > 0 && (
                    <span className="block text-label text-faint sm:mt-1">
                      <span aria-hidden="true">See </span>
                      {ids.map((id, j) => (
                        <span key={id}>
                          {j > 0 && ", "}
                          <a
                            href={`#${changeAnchor(id)}`}
                            onClick={(event) => {
                              event.preventDefault();
                              onJump(id, ids);
                            }}
                            aria-label={`Show change #${id} in the comparison`}
                            className={`font-mono ${linkClass}`}
                          >
                            #{id}
                          </a>
                        </span>
                      ))}
                    </span>
                  )}
                </div>
                <div className="min-w-0 max-w-[70ch]">
                  <p className="text-body font-medium break-words text-ink">{item.what_changed}</p>
                  <p className="mt-1 text-body text-muted">{item.interpretation}</p>
                </div>
              </li>
            );
          })}
        </ol>
      ) : (
        <p className="mt-2 text-small text-faint">The analysis lists no key changes.</p>
      )}
    </>
  );
}

function GeneratePrompt({ generating, onGenerate }: { generating: boolean; onGenerate: () => void }) {
  return (
    <div className="max-w-[70ch]">
      <p className="text-body text-muted">
        No analysis is stored for this pair yet. The sentence comparison below is complete without one.
      </p>
      <button
        type="button"
        onClick={onGenerate}
        aria-disabled={generating || undefined}
        className={`mt-4 ${primaryButtonClass}`}
      >
        {generating ? (
          <>
            <Spinner className="size-4" />
            Generating analysis…
          </>
        ) : (
          "Generate analysis"
        )}
      </button>
      <p className={`mt-2 ${helpClass}`}>One OpenAI call, a few seconds. The result is saved for next time.</p>
    </div>
  );
}

/**
 * The public demo, for a pair without a stored analysis: analyses exist for every pair of
 * consecutive statements, so offer the pair that ends at the same target.
 */
function DemoNotStored({
  base,
  target,
  dates,
  onChoose,
}: {
  base: string;
  target: string;
  dates: readonly string[] | null;
  onChoose: (base: string, target: string) => void;
}) {
  const previous = dates ? previousStatement(dates, target) : null;
  const consecutive = previous === base;
  return (
    <div className="max-w-[70ch]">
      <p className="text-body text-muted">
        {consecutive
          ? "No analysis is stored for this pair, and the public demo does not write new ones."
          : "On the public demo, analyses exist only for back-to-back meetings."}{" "}
        The sentence comparison below is complete without one.
      </p>
      {previous && !consecutive && (
        <button type="button" onClick={() => onChoose(previous, target)} className={`mt-4 ${secondaryButtonClass}`}>
          Compare with the previous meeting ({formatDate(previous)})
        </button>
      )}
    </div>
  );
}

/** The two statements' stored stance scores from one method, as shown on Policy Stance. */
function StanceCrossCheck({ stance }: { stance: CompareStance }) {
  const classifier = stance.base?.classifier ?? stance.target?.classifier ?? "llm_zero_shot";
  const label = classifierInfo(classifier).label;
  return (
    <div>
      <dt className={labelTextClass}>Stance score ({label})</dt>
      <dd className="mt-0.5">
        {stance.base && stance.target ? (
          <span className="flex flex-wrap items-center gap-x-2 gap-y-1 tabular-nums">
            <StanceLabel stance={stance.base.stance} />
            <span className="font-medium text-ink">{formatScore(stance.base.score)}</span>
            <span aria-hidden="true" className="text-faint">
              →
            </span>
            <span className="sr-only">to</span>
            <StanceLabel stance={stance.target.stance} />
            <span className="font-medium text-ink">{formatScore(stance.target.score)}</span>
          </span>
        ) : (
          <span className="text-faint">No stored score for one of the statements.</span>
        )}
      </dd>
    </div>
  );
}
