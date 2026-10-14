"use client";

import { useSearchParams } from "next/navigation";
import { useEffect, useId, useRef, useState } from "react";

import type { EvaluationReport, QuestionType, ReportSplit, RunKey } from "@/lib/api/types";
import { describeEvaluationError } from "@/lib/performance/failures";
import { SPLIT_LABELS } from "@/lib/performance/format";
import {
  DEFAULT_VIEW,
  parsePerformanceParams,
  type PerformanceViewState,
  serializePerformanceParams,
} from "@/lib/performance/params";
import { runChoices } from "@/lib/performance/questions";
import { baselineRun, defaultSplit, headlineRun, reportSplits, sameRun } from "@/lib/performance/report";

import { FailureNotice } from "../stance/FailureNotice";
import { Section } from "../ui/Section";
import { labelTextClass } from "../ui/styles";
import { AnswerSection } from "./AnswerSection";
import { CostSection } from "./CostSection";
import { Headline } from "./Headline";
import { PerformanceAbout } from "./PerformanceAbout";
import { PerformanceSkeleton } from "./PerformanceSkeleton";
import { QuestionBrowser } from "./QuestionBrowser";
import { QuestionDrilldown } from "./QuestionDrilldown";
import { RetrievalSection } from "./RetrievalSection";
import { TypeHeatmap } from "./TypeHeatmap";
import { useEvaluationReport, useQuestionDetail } from "./usePerformanceData";

const URL_SYNC_DELAY_MS = 300;

function prefersReducedMotion(): boolean {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/** Scroll an element into view when its top is not already in the upper part of the screen. */
function reveal(element: HTMLElement | null) {
  if (!element) return;
  const top = element.getBoundingClientRect().top;
  if (top < 0 || top > window.innerHeight * 0.6) {
    element.scrollIntoView({ behavior: prefersReducedMotion() ? "auto" : "smooth", block: "start" });
  }
}

/**
 * RAG Performance: the evaluation of the four retrieval methods under each retriever config, on
 * one split at a time, with a browser into every question. The view lives in the URL.
 */
export function PerformanceExplorer() {
  const searchParams = useSearchParams();
  const [view, setView] = useState<PerformanceViewState>(() => parsePerformanceParams(searchParams));
  const { report, retry } = useEvaluationReport();

  // Mirror the view in the query string without a history entry per change.
  useEffect(() => {
    const timer = window.setTimeout(() => {
      const query = serializePerformanceParams(view);
      const { pathname, search, hash } = window.location;
      const next = query ? `?${query}` : "";
      if (next !== search) window.history.replaceState(null, "", `${pathname}${next}${hash}`);
    }, URL_SYNC_DELAY_MS);
    return () => window.clearTimeout(timer);
  }, [view]);

  if (report.status === "loading") return <PerformanceSkeleton />;
  if (report.status === "error") {
    return (
      <div className="mt-12">
        <FailureNotice failure={describeEvaluationError(report.error)} onRetry={retry} />
        <WhatItShows />
      </div>
    );
  }
  return <Results report={report.data} view={view} setView={setView} />;
}

function Results({
  report,
  view,
  setView,
}: {
  report: EvaluationReport;
  view: PerformanceViewState;
  setView: (update: (current: PerformanceViewState) => PerformanceViewState) => void;
}) {
  const splitGroup = useId();
  const browserRef = useRef<HTMLElement>(null);
  const drilldownRef = useRef<HTMLDivElement>(null);
  const [search, setSearch] = useState("");
  const [focusKey, setFocusKey] = useState(0);
  const { detail, retry } = useQuestionDetail(view.question);
  const revealedOnLoad = useRef(false);

  // A link with ?q= opens a question far down the page: bring it into view once.
  useEffect(() => {
    if (revealedOnLoad.current) return;
    revealedOnLoad.current = true;
    if (view.question) window.requestAnimationFrame(() => reveal(drilldownRef.current));
  }, [view.question]);

  const splits = reportSplits(report);
  const split: ReportSplit = view.split && splits.includes(view.split) ? view.split : defaultSplit(splits);
  const headline = headlineRun(report);
  const baseline = baselineRun(report, headline?.run ?? null);
  const runs = runChoices(report.questions, report.configs);
  const browserRun: RunKey | undefined =
    (view.run && runs.find((r) => sameRun(r, view.run!))) || headline?.run || runs[0];
  const update = (patch: Partial<PerformanceViewState>) => setView((current) => ({ ...current, ...patch }));
  const { dataset } = report;

  const showType = (type: QuestionType, run: RunKey) => {
    update({ type, run, outcome: "all" });
    window.requestAnimationFrame(() => reveal(browserRef.current));
  };
  const openQuestion = (id: string) => {
    update({ question: id });
    setFocusKey((key) => key + 1);
    window.requestAnimationFrame(() => reveal(drilldownRef.current));
  };
  const closeQuestion = () => {
    const id = view.question;
    update({ question: DEFAULT_VIEW.question });
    // Back to the question's button in the list, rather than to the top of the page.
    window.requestAnimationFrame(() => {
      browserRef.current?.querySelector<HTMLElement>(`[data-question-id="${id}"]`)?.focus();
    });
  };

  return (
    <div className="mt-12">
      <div>
        <fieldset>
          <legend className={labelTextClass}>Questions</legend>
          <div className="mt-1 flex gap-6 border-b border-line">
            {splits.map((s) => {
              const count = s === "all" ? dataset.n_questions : dataset.splits.find((x) => x.split === s)?.n;
              const active = s === split;
              return (
                <label
                  key={s}
                  className={[
                    "-mb-px cursor-pointer rounded-t-sm border-b-[1.5px] pt-1 pb-2 text-small font-medium transition-colors has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-[var(--focus-ring)]",
                    active ? "border-ink text-ink" : "border-transparent text-muted hover:text-ink",
                  ].join(" ")}
                >
                  <input
                    type="radio"
                    name={splitGroup}
                    value={s}
                    checked={active}
                    onChange={() => update({ split: s === defaultSplit(splits) ? null : s })}
                    className="sr-only"
                  />
                  {SPLIT_LABELS[s]}
                  {count !== undefined && <span className="ml-1.5 font-normal text-faint tabular-nums">{count}</span>}
                </label>
              );
            })}
          </div>
        </fieldset>
        <p className="mt-3 max-w-[70ch] text-small text-faint">
          {dataset.n_questions} questions, {dataset.n_human_reviewed} checked by a person. Settings were tuned on Dev,
          so Test is the fair score.
        </p>
      </div>

      {headline && <Headline report={report} split={split} headline={headline} baseline={baseline} />}
      <RetrievalSection report={report} split={split} highlight={headline?.run ?? null} />
      <TypeHeatmap report={report} split={split} highlight={headline?.run ?? null} onSelect={showType} />
      <AnswerSection report={report} split={split} highlight={headline?.run ?? null} />
      <CostSection report={report} split={split} highlight={headline?.run ?? null} />

      <Section
        id="perf-questions"
        sectionRef={browserRef}
        title="Every question"
        lead="Open one to see what each method found and how its answer was judged."
      >
        <div className="mt-6">
          {browserRun ? (
            <QuestionBrowser
              questions={report.questions}
              configs={report.configs}
              split={split}
              type={view.type}
              outcome={view.outcome}
              run={browserRun}
              runs={runs}
              headline={headline?.run ?? null}
              search={search}
              selected={view.question}
              onChange={({ search: text, ...patch }) => {
                if (text !== undefined) setSearch(text);
                if (Object.keys(patch).length) update(patch);
              }}
              onSelect={openQuestion}
            />
          ) : (
            <p className="text-small text-faint">No questions in the report.</p>
          )}
        </div>
        {view.question && detail && browserRun && (
          <div ref={drilldownRef} className="mt-10 scroll-mt-6">
            <QuestionDrilldown
              load={detail}
              configs={report.configs}
              preferredRun={browserRun}
              focusKey={focusKey}
              onRetry={retry}
              onClose={closeQuestion}
            />
          </div>
        )}
      </Section>

      <PerformanceAbout report={report} split={split} headline={headline} />
    </div>
  );
}

/** What the page shows once the evaluation has run: shown with the setup hint. */
function WhatItShows() {
  return (
    <p className="mt-8 max-w-[70ch] text-body text-muted">
      Once the evaluation has run, this page compares four search methods (BM25, dense, hybrid, and hybrid with
      reranking) on the same questions, grades the answers built on them, and opens every question&rsquo;s passages
      and answer.
    </p>
  );
}
