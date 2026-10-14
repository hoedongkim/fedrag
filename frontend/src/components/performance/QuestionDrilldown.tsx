"use client";

import Link from "next/link";
import { type ReactNode, useEffect, useId, useRef, useState } from "react";

import type { ChunkRole, EvalPassage, JudgeVerdicts, QuestionDetail, RunDetail, RunKey } from "@/lib/api/types";
import { formatDate, formatScore } from "@/lib/ask/format";
import {
  formatMetric,
  formatMs,
  formatUsd,
  METRIC_DIGITS,
  ROLE_LABELS,
  runLabel,
  SPLIT_LABELS,
  TYPE_LABELS,
} from "@/lib/performance/format";
import {
  answerCitations,
  askHref,
  type HitMarker,
  hitMarker,
  missedGold,
  orderedRunDetails,
  rankLabel,
  rankStrip,
} from "@/lib/performance/questions";
import { describeEvaluationError } from "@/lib/performance/failures";
import { runKey, sameRun } from "@/lib/performance/report";

import { AnswerText } from "../ask/AnswerText";
import { ExternalLinkIcon } from "../ask/icons";
import { FailureNotice } from "../stance/FailureNotice";
import type { Load } from "../stance/useStanceData";
import { labelTextClass, secondaryButtonClass, skeletonClass, textButtonClass } from "../ui/styles";
import { useDemoMode } from "../useBackend";
import { MetaLine } from "./parts";

const HIGHLIGHT_MS = 1600;

interface QuestionDrilldownProps {
  load: Load<QuestionDetail>;
  configs: { id: string; label: string }[];
  /** The run to open first (the browser's run). */
  preferredRun: RunKey;
  /** Changes each time the user opens a question: the heading then takes focus once loaded. */
  focusKey?: number;
  onRetry: () => void;
  onClose: () => void;
}

/** One question: its labels, every run's ranking at a glance, and one run's passages and answer. */
export function QuestionDrilldown({
  load,
  configs,
  preferredRun,
  focusKey = 0,
  onRetry,
  onClose,
}: QuestionDrilldownProps) {
  if (load.status === "loading") {
    return (
      <div aria-busy="true" className="space-y-3">
        <span className="sr-only">Loading the question…</span>
        <div className={`h-5 w-2/3 ${skeletonClass}`} />
        <div className={`h-40 ${skeletonClass}`} />
      </div>
    );
  }
  if (load.status === "error") {
    return <FailureNotice failure={describeEvaluationError(load.error)} onRetry={onRetry} />;
  }
  return (
    <Detail
      key={load.data.id}
      detail={load.data}
      configs={configs}
      preferredRun={preferredRun}
      focusKey={focusKey}
      onClose={onClose}
    />
  );
}

function Detail({
  detail,
  configs,
  preferredRun,
  focusKey,
  onClose,
}: {
  detail: QuestionDetail;
  configs: { id: string; label: string }[];
  preferredRun: RunKey;
  focusKey: number;
  onClose: () => void;
}) {
  const headingId = useId();
  const headingRef = useRef<HTMLHeadingElement>(null);
  const demo = useDemoMode();
  // Opened from the list: move focus here, so keyboard and screen-reader users land on it.
  useEffect(() => {
    if (focusKey) headingRef.current?.focus({ preventScroll: true });
  }, [focusKey]);
  const runs = orderedRunDetails(detail.runs, configs);
  const [chosen, setChosen] = useState<string | null>(null);
  const active =
    runs.find((r) => runKey(r) === chosen) ?? runs.find((r) => sameRun(r, preferredRun)) ?? runs[0] ?? null;
  const passages = detail.passages;

  return (
    <article aria-labelledby={headingId} className="space-y-10">
      <header>
        <div className="flex flex-wrap items-start justify-between gap-3">
          <MetaLine
            className="pt-1"
            items={[
              <span key="id" className="font-mono">
                {detail.id}
              </span>,
              TYPE_LABELS[detail.type],
              SPLIT_LABELS[detail.split],
              detail.as_of && `as of ${formatDate(detail.as_of)}`,
              !detail.answerable && <span className="font-medium text-warn">unanswerable</span>,
              detail.human_reviewed && "checked by a person",
            ]}
          />
          <button type="button" onClick={onClose} className={secondaryButtonClass}>
            Close
          </button>
        </div>
        <h3
          id={headingId}
          ref={headingRef}
          tabIndex={-1}
          className="mt-2 max-w-[70ch] rounded-sm text-[1.125rem] leading-snug font-semibold tracking-[-0.01em] text-ink"
        >
          {detail.question}
        </h3>
        {/* The public demo takes no new questions, and the runs below already show its answers. */}
        {demo === false && (
          <p className="mt-2 text-small">
            <Link href={askHref(detail, active?.method ?? "hybrid_rerank")} className={textButtonClass}>
              Ask it in Ask the Fed
            </Link>
            <span className="ml-2 text-label text-faint">(fills the form, does not send)</span>
          </p>
        )}
      </header>

      <div className="grid gap-x-10 gap-y-8 md:grid-cols-2">
        <div>
          <SubHeading>Reference answer</SubHeading>
          <p className="mt-1.5 text-small text-muted">
            {detail.reference_answer ?? (detail.answerable ? "None recorded." : "None: the documents cannot answer it, so the right reply declines.")}
          </p>
          {detail.required_facts.length > 0 && (
            <>
              <SubHeading className="mt-5">Required facts</SubHeading>
              <ul className="mt-1.5 list-disc space-y-1 pl-5 text-small text-muted marker:text-faint">
                {detail.required_facts.map((fact) => (
                  <li key={fact}>{fact}</li>
                ))}
              </ul>
            </>
          )}
          {detail.notes && <p className="mt-3 text-label text-faint">Note: {detail.notes}</p>}
        </div>
        <div>
          <SubHeading>Gold passages</SubHeading>
          {detail.gold.length === 0 ? (
            <p className="mt-1.5 text-small text-muted">None: no passage answers it.</p>
          ) : (
            <ul className="mt-1.5 divide-y divide-line border-y border-line">
              {detail.gold.map((gold) => (
                <li key={gold.chunk_id}>
                  <PassageBlock chunkId={gold.chunk_id} passage={passages[gold.chunk_id]} marker={gold.role} />
                </li>
              ))}
            </ul>
          )}
          {!detail.passages_available && (
            <p className="mt-2 text-label text-faint">
              Passage text comes from the local search index, which is not built here (make index), so ids are
              shown instead.
            </p>
          )}
        </div>
      </div>

      {runs.length > 0 && (
        <RunOverview runs={runs} configs={configs} active={active} answerable={detail.answerable} onSelect={(run) => setChosen(runKey(run))} />
      )}

      {active && <RunPanel key={runKey(active)} run={active} detail={detail} configs={configs} />}
    </article>
  );
}

function SubHeading({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <h4 className={`${labelTextClass} ${className}`}>{children}</h4>;
}

const MARKER_LABELS: Record<HitMarker, string> = {
  primary: "Primary gold",
  supporting: "Supporting gold",
  same_doc: "Right document",
  none: "Other",
};

const MARKER_SWATCH: Record<HitMarker, string> = {
  primary: "bg-ink",
  supporting: "bg-faint",
  same_doc: "bg-transparent ring-[1.5px] ring-faint ring-inset",
  none: "bg-line",
};

/** Every run's top 10 as a strip of rank cells; selecting a row opens that run below. */
function RunOverview({
  runs,
  configs,
  active,
  answerable,
  onSelect,
}: {
  runs: RunDetail[];
  configs: { id: string; label: string }[];
  active: RunDetail | null;
  answerable: boolean;
  onSelect: (run: RunDetail) => void;
}) {
  return (
    <section aria-label="Every run’s top 10">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <SubHeading>Top 10 of every run</SubHeading>
        <MarkerLegend />
      </div>
      <ul className="mt-2 border-y border-line">
        {runs.map((run) => {
          const current = !!active && sameRun(active, run);
          const strip = rankStrip(run);
          return (
            <li key={runKey(run)}>
              <button
                type="button"
                aria-pressed={current}
                onClick={() => onSelect(run)}
                className={[
                  "grid w-full grid-cols-[minmax(0,1fr)_auto] items-center gap-x-4 gap-y-1 px-2 py-2 text-left transition-colors sm:grid-cols-[14rem_auto_minmax(0,1fr)] focus-visible:-outline-offset-2",
                  current ? "bg-accent-soft shadow-[inset_2px_0_0_var(--accent)]" : "hover:bg-subtle",
                ].join(" ")}
              >
                <span className={`truncate text-small ${current ? "font-medium text-ink" : "text-muted"}`}>
                  {runLabel(configs, run)}
                </span>
                <span className="order-3 col-span-2 flex gap-0.5 sm:order-none sm:col-span-1" aria-hidden="true">
                  {strip.map((marker, i) => (
                    <span key={i} className={`size-3 rounded-[2px] ${MARKER_SWATCH[marker]}`} />
                  ))}
                </span>
                <span className="justify-self-end text-label text-faint tabular-nums sm:justify-self-start">
                  <span className="sr-only">
                    {strip.filter((m) => m === "primary" || m === "supporting").length} gold passages in the top 10;
                  </span>
                  first gold {rankLabel(answerable, run.scores.first_relevant_rank ?? null)}
                </span>
              </button>
            </li>
          );
        })}
      </ul>
    </section>
  );
}

function MarkerLegend() {
  return (
    <span className="flex flex-wrap gap-x-3 gap-y-1 text-label text-faint" aria-hidden="true">
      {(Object.keys(MARKER_LABELS) as HitMarker[]).map((marker) => (
        <span key={marker} className="inline-flex items-center gap-1">
          <span className={`size-2.5 rounded-[2px] ${MARKER_SWATCH[marker]}`} />
          {MARKER_LABELS[marker]}
        </span>
      ))}
    </span>
  );
}

/** One run on the question: what it retrieved, and its answer with the judge's verdicts. */
function RunPanel({ run, detail, configs }: { run: RunDetail; detail: QuestionDetail; configs: { id: string; label: string }[] }) {
  const baseId = useId();
  const [flash, setFlash] = useState<string | null>(null);
  const timer = useRef<number | undefined>(undefined);
  useEffect(() => () => window.clearTimeout(timer.current), []);
  const missed = missedGold(detail.gold, run);
  const roles = new Map(detail.gold.map((g) => [g.chunk_id, g.role]));
  const answer = run.answer;
  const citations = answer ? answerCitations(answer, detail.passages, run) : new Map();
  const itemId = (chunkId: string) => `${baseId}-${chunkId.replace(/[^A-Za-z0-9_-]/g, "_")}`;

  const cite = (n: number) => {
    const cited = answer?.citations.find((c) => c.n === n);
    if (!cited) return;
    const element = document.getElementById(itemId(cited.chunk_id));
    if (!element) return;
    element.scrollIntoView({ block: "nearest", behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
    element.focus({ preventScroll: true });
    setFlash(cited.chunk_id);
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setFlash(null), HIGHLIGHT_MS);
  };

  return (
    <section aria-label={runLabel(configs, run)}>
      <h4 className="text-body font-semibold text-ink">{runLabel(configs, run)}</h4>
      <p className="mt-0.5 text-label text-faint tabular-nums">
        {detail.answerable
          ? `First gold ${rankLabel(true, run.scores.first_relevant_rank ?? null)} · Recall@5 ${formatMetric(run.scores.recall_at_5, METRIC_DIGITS)} · nDCG@10 ${formatMetric(run.scores.ndcg_at_10, METRIC_DIGITS)} · `
          : "No gold passages (unanswerable) · "}
        retrieval {formatMs(run.latency_ms)}
      </p>

      <div className="mt-5 grid gap-x-10 gap-y-8 lg:grid-cols-2">
        <div className="min-w-0">
          <SubHeading>Retrieved, in rank order</SubHeading>
          {run.retrieved.length === 0 ? (
            <p className="mt-1.5 text-small text-muted">Nothing was retrieved.</p>
          ) : (
            <ol className="mt-1.5 divide-y divide-line border-y border-line">
              {run.retrieved.map((hit) => (
                <li
                  key={`${hit.rank}-${hit.chunk_id}`}
                  id={itemId(hit.chunk_id)}
                  tabIndex={-1}
                  className={[
                    "scroll-mt-6 transition-[box-shadow,background-color] duration-300 motion-reduce:transition-none focus-visible:-outline-offset-2",
                    flash === hit.chunk_id ? "bg-accent-soft shadow-[inset_2px_0_0_var(--accent)]" : "",
                  ].join(" ")}
                >
                  <PassageBlock
                    chunkId={hit.chunk_id}
                    passage={detail.passages[hit.chunk_id]}
                    marker={hitMarker(hit)}
                    rank={hit.rank}
                    score={hit.score}
                    citedAs={answer?.citations.filter((c) => c.chunk_id === hit.chunk_id).map((c) => c.n)}
                  />
                </li>
              ))}
            </ol>
          )}
          {missed.length > 0 && (
            <div className="mt-3 border-l-2 border-line-strong pl-3 text-label text-faint">
              <p className="font-medium text-muted">Gold passages it did not retrieve:</p>
              <ul className="mt-1 space-y-0.5">
                {missed.map((chunkId) => {
                  const passage = detail.passages[chunkId];
                  const role = roles.get(chunkId);
                  return (
                    <li key={chunkId}>
                      {role ? `${ROLE_LABELS[role]}: ` : ""}
                      {passage ? `${passage.document.title}${passage.section ? ` · ${passage.section}` : ""}` : <code className="font-mono">{chunkId}</code>}
                    </li>
                  );
                })}
              </ul>
            </div>
          )}
        </div>

        <div className="min-w-0">
          <SubHeading>Answer from its top 5</SubHeading>
          {!answer ? (
            <p className="mt-1.5 text-small text-muted">This run was not answered.</p>
          ) : (
            <>
              <div className="mt-1.5 border-l-2 border-line-strong pl-4 text-body text-ink">
                <AnswerText text={answer.text} citations={citations} onCite={cite} />
              </div>
              {answer.notes.map((note) => (
                <p key={note} className="mt-2 text-label text-faint">
                  Note: {note}
                </p>
              ))}
              <p className="mt-2 text-label text-faint tabular-nums">
                {answer.citations.length} citation{answer.citations.length === 1 ? "" : "s"}
                {answer.latency_ms !== null && ` · ${formatMs(answer.latency_ms)} end to end`}
                {answer.cost_usd !== null && ` · ${formatUsd(answer.cost_usd)}`}
              </p>
              {answer.judge ? (
                <Verdicts judge={answer.judge} answerable={detail.answerable} />
              ) : (
                <p className="mt-4 text-small text-muted">Not judged.</p>
              )}
            </>
          )}
        </div>
      </div>
    </section>
  );
}

/** A passage: its marker and rank, document, section and date, and its text (folded if long). */
function PassageBlock({
  chunkId,
  passage,
  marker,
  rank,
  score,
  citedAs = [],
}: {
  chunkId: string;
  passage: EvalPassage | undefined;
  marker: HitMarker | ChunkRole;
  rank?: number;
  score?: number | null;
  citedAs?: number[];
}) {
  const [open, setOpen] = useState(false);
  const long = (passage?.text.length ?? 0) > 280;
  return (
    <div className="py-3">
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-label">
        {rank !== undefined && <span className="font-semibold text-ink tabular-nums">#{rank}</span>}
        <span className="inline-flex items-center gap-1 text-muted">
          <span aria-hidden="true" className={`size-2.5 rounded-[2px] ${MARKER_SWATCH[marker]}`} />
          {MARKER_LABELS[marker]}
        </span>
        {citedAs.length > 0 && <span className="text-accent">cited as {citedAs.map((n) => `[${n}]`).join(" ")}</span>}
        {score !== undefined && score !== null && (
          <span className="ml-auto text-faint tabular-nums" title="Search score, comparable only within one method">
            score {formatScore(score)}
          </span>
        )}
      </div>
      {passage ? (
        <>
          <p className="mt-1 text-label break-words text-faint">
            <a
              href={passage.document.url}
              target="_blank"
              rel="noopener noreferrer"
              className="rounded-sm font-medium text-muted underline-offset-2 hover:text-ink hover:underline"
            >
              {passage.document.title}
              <ExternalLinkIcon className="ml-0.5 inline size-3 align-[-1px]" />
              <span className="sr-only"> (opens in a new tab)</span>
            </a>
            {passage.section ? ` · ${passage.section}` : ""} · released {formatDate(passage.document.release_date)}
          </p>
          <p className={`mt-1 text-small break-words text-muted ${long && !open ? "line-clamp-3" : ""}`}>
            {passage.text}
          </p>
          {long && (
            <button type="button" aria-expanded={open} onClick={() => setOpen((v) => !v)} className={`${textButtonClass} mt-1 text-label`}>
              {open ? "Show less" : "Show all"}
            </button>
          )}
        </>
      ) : (
        <p className="mt-1 font-mono text-label break-all text-faint">{chunkId}</p>
      )}
    </div>
  );
}

/** The judge's verdicts on one answer: scores, then each claim, citation, fact and refusal. */
function Verdicts({ judge, answerable }: { judge: JudgeVerdicts; answerable: boolean }) {
  const supportedClaims = judge.claims.filter((c) => c.supported).length;
  const supportedCitations = judge.citations.filter((c) => c.supported).length;
  const covered = judge.facts.filter((f) => f.covered).length;
  return (
    <div className="mt-5 space-y-5">
      <dl className="flex flex-wrap gap-x-8 gap-y-3 text-small">
        <Score label="Faithfulness" value={judge.faithfulness} detail={`${supportedClaims} of ${judge.claims.length} claims`} />
        <Score label="Citation accuracy" value={judge.citation_accuracy} detail={`${supportedCitations} of ${judge.citations.length} citations`} />
        {answerable ? (
          <Score label="Correctness" value={judge.answer_correctness} detail={`${covered} of ${judge.facts.length} facts`} />
        ) : (
          <div>
            <dt className={labelTextClass}>Refusal</dt>
            <dd className="mt-0.5">
              <Verdict ok={judge.refusal_correct === true} okLabel="Correct" badLabel={judge.refusal_correct === null ? "Unjudged" : "Wrong"} />
            </dd>
          </div>
        )}
      </dl>
      {answerable && judge.false_refusal && (
        <p className="text-small text-muted">
          <Verdict ok={false} okLabel="" badLabel="False refusal" />: the answer declines although the
          documents answer the question.
        </p>
      )}

      {judge.claims.length > 0 && (
        <VerdictList title="Claims">
          {judge.claims.map((claim, i) => (
            <VerdictItem key={i} ok={claim.supported} okLabel="Supported" badLabel="Not supported" reason={claim.reason}>
              {claim.claim}
              {claim.citations.length > 0 && (
                <span className="ml-1 text-label text-accent">{claim.citations.map((n) => `[${n}]`).join("")}</span>
              )}
              {claim.citations.length === 0 && <span className="ml-1 text-label text-faint">(uncited)</span>}
            </VerdictItem>
          ))}
        </VerdictList>
      )}
      {judge.citations.length > 0 && (
        <VerdictList title="Citations">
          {/* One check per marker: a passage cited in two sentences has two checks with the same n. */}
          {judge.citations.map((c, i) => (
            <VerdictItem key={`${i}-${c.n}`} ok={c.supported} okLabel="Supports" badLabel="Does not support" reason={c.reason}>
              <span className="text-label font-semibold text-accent">[{c.n}]</span> {c.sentence}
            </VerdictItem>
          ))}
        </VerdictList>
      )}
      {judge.facts.length > 0 && (
        <VerdictList title="Required facts">
          {judge.facts.map((f) => (
            <VerdictItem key={f.fact} ok={f.covered} okLabel="Stated" badLabel="Missing" reason={f.reason}>
              {f.fact}
            </VerdictItem>
          ))}
        </VerdictList>
      )}
      {judge.refusal && (judge.refusal.declined || judge.refusal.fabricated || !answerable) && (
        <p className="text-label text-faint">
          Refusal verdict: {judge.refusal.declined ? "declines" : "does not decline"}
          {judge.refusal.fabricated ? ", and asserts facts the passages do not contain" : ""}.
          {judge.refusal.reason ? ` ${judge.refusal.reason}` : ""}
        </p>
      )}
    </div>
  );
}

function Score({ label, value, detail }: { label: string; value: number | null; detail: string }) {
  return (
    <div>
      <dt className={labelTextClass}>{label}</dt>
      <dd className="mt-0.5 text-heading font-semibold text-ink tabular-nums">{formatMetric(value, METRIC_DIGITS)}</dd>
      <dd className="text-label text-faint">{detail}</dd>
    </div>
  );
}

function VerdictList({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div>
      <SubHeading>{title}</SubHeading>
      <ul className="mt-1.5 space-y-2.5">{children}</ul>
    </div>
  );
}

function VerdictItem({
  ok,
  okLabel,
  badLabel,
  reason,
  children,
}: {
  ok: boolean;
  okLabel: string;
  badLabel: string;
  reason: string | null;
  children: ReactNode;
}) {
  return (
    <li className="text-small text-muted">
      <Verdict ok={ok} okLabel={okLabel} badLabel={badLabel} /> {children}
      {reason && <span className="mt-0.5 block text-label text-faint">{reason}</span>}
    </li>
  );
}

/** A verdict as icon plus word, never color alone. */
function Verdict({ ok, okLabel, badLabel }: { ok: boolean; okLabel: string; badLabel: string }) {
  return (
    <span
      className={`inline-flex items-center gap-1 align-[1px] text-label font-semibold ${ok ? "text-good" : "text-bad"}`}
    >
      <svg viewBox="0 0 16 16" aria-hidden="true" className="size-3.5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        {ok ? <path d="M3 8.5 6.5 12 13 4.5" /> : <path d="M4 4l8 8M12 4l-8 8" />}
      </svg>
      {ok ? okLabel : badLabel}
    </span>
  );
}
