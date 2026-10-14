"use client";

import { useSearchParams } from "next/navigation";
import { useCallback, useEffect, useEffectEvent, useMemo, useRef, useState } from "react";
import { flushSync } from "react-dom";

import { demoModeOf } from "@/lib/api/backend";
import type { CompareResponse } from "@/lib/api/types";
import { formatDate } from "@/lib/ask/format";
import { describeCompareError } from "@/lib/compare/failures";
import { describeStats, SHIFT_LABELS } from "@/lib/compare/format";
import {
  type CompareParams,
  parseCompareParams,
  resolvePair,
  serializeCompareParams,
} from "@/lib/compare/params";
import { buildDiffBlocks, changeAnchor, changeIdFromHash, foldedRunOf, runIds } from "@/lib/compare/rows";

import { FailureNotice } from "../stance/FailureNotice";
import type { Load } from "../stance/useStanceData";
import { useBackend } from "../useBackend";
import { AnalysisPanel } from "./AnalysisPanel";
import { CompareAbout } from "./CompareAbout";
import { CompareControls } from "./CompareControls";
import { ComparisonSkeleton } from "./CompareSkeleton";
import { DiffView } from "./DiffView";
import { NotFoundNotice, PairProblemNotice } from "./PairNotices";
import { type Generation, useComparison, useMediaQuery, useStatements } from "./useCompareData";

const URL_SYNC_DELAY_MS = 300;
const HIGHLIGHT_MS = 2000;
/** Split view from Tailwind's md breakpoint up; unified below it. */
const SPLIT_QUERY = "(min-width: 768px)";

function prefersReducedMotion(): boolean {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/**
 * Statement Comparison: two statement pickers, the LLM analysis of the change and the sentence
 * diff. The pair lives in the URL (base, target). Opening a link loads the diff and only an
 * analysis that is already stored; writing a new one takes an explicit click, since it spends
 * OpenAI credits.
 */
export function CompareExperience() {
  const searchParams = useSearchParams();
  const [params, setParams] = useState<CompareParams>(() => parseCompareParams(searchParams));
  const statements = useStatements();
  const dates = useMemo(
    () => (statements.load.status === "ready" ? statements.load.data.map((m) => m.meeting_date) : null),
    [statements.load],
  );
  const resolution = resolvePair(params, dates);
  const base = resolution?.ok ? resolution.base : null;
  const target = resolution?.ok ? resolution.target : null;
  const { load, generation, generate, retry } = useComparison(base, target);
  const split = useMediaQuery(SPLIT_QUERY);
  // The public demo never writes analyses: no Generate button there (null until health answers).
  const demo = demoModeOf(useBackend());

  // Mirror the pair in the query string without a history entry per change.
  const written = useRef<string | null>(null);
  useEffect(() => {
    const timer = window.setTimeout(() => {
      const query = serializeCompareParams(params);
      const { pathname, search } = window.location;
      const next = query ? `?${query}` : "";
      if (next === search) return;
      written.current = query;
      window.history.replaceState(null, "", `${pathname}${next}`);
    }, URL_SYNC_DELAY_MS);
    return () => window.clearTimeout(timer);
  }, [params]);

  // A navigation to this route with other params (e.g. the tab link, or Back) replaces the pair.
  // Keyed on the query string, so a hash-only change is no navigation; the query this page wrote
  // itself is skipped, as it may arrive after a newer pick.
  const query = searchParams.toString();
  useEffect(() => {
    if (query === written.current) {
      written.current = null;
      return;
    }
    const next = parseCompareParams(new URLSearchParams(query));
    setParams((p) => (p.base === next.base && p.target === next.target ? p : next));
  }, [query]);

  const choose = useCallback((b: string | null, t: string | null) => {
    // A #change-N hash belongs to the pair it was made for: drop it before the next pair renders.
    const { hash, pathname, search } = window.location;
    if (hash) window.history.replaceState(null, "", `${pathname}${search}`);
    setParams({ base: b, target: t });
  }, []);
  // With the database down both requests fail alike: one notice (the comparison's) is enough,
  // and its Retry reloads the statement list too.
  const comparisonFailed = base !== null && load.status === "error";
  const retryAll = () => {
    retry();
    if (statements.load.status === "error") statements.retry();
  };

  return (
    <div className="mt-12">
      <CompareControls
        statements={statements.load}
        onRetryStatements={statements.retry}
        showStatementsError={!comparisonFailed}
        base={resolution ? resolution.base : params.base}
        target={resolution ? resolution.target : params.target}
        onChange={choose}
      />

      {resolution && !resolution.ok && (
        <div className="mt-10">
          <PairProblemNotice resolution={resolution} dates={dates} onChoose={choose} />
        </div>
      )}

      {base && target ? (
        <div className="mt-16">
          <ComparisonView
            key={`${base}|${target}`}
            base={base}
            target={target}
            load={load}
            generation={generation}
            dates={dates}
            split={split}
            demo={demo}
            onGenerate={generate}
            onRetry={retryAll}
            onChoose={choose}
          />
        </div>
      ) : (
        resolution === null &&
        statements.load.status === "loading" && (
          <div className="mt-16">
            <ComparisonSkeleton />
          </div>
        )
      )}

      <CompareAbout
        modelVersion={load.status === "ready" ? load.data.analysis_model_version : null}
        demo={demo === true}
      />

      <p role="status" aria-live="polite" className="sr-only">
        {base && target ? announcement(base, target, load, generation) : ""}
      </p>
    </div>
  );
}

/** What the live region says: the outcome of each load and of generating, never row details. */
function announcement(base: string, target: string, load: Load<CompareResponse>, generation: Generation): string {
  if (generation.status === "generating") return "Generating the analysis…";
  if (generation.status === "error") return "The analysis could not be generated.";
  if (load.status === "loading") return "";
  if (load.status === "error") return "The comparison could not be loaded.";
  const { stats, analysis, analysis_status: status } = load.data;
  const lead = `Compared the ${formatDate(base)} and ${formatDate(target)} statements: ${describeStats(stats)}`;
  if (status === "ready" && analysis) return `${lead} Analysis: ${SHIFT_LABELS[analysis.stance_shift].toLowerCase()}.`;
  if (status !== "ready") return `${lead} No stored analysis.`;
  return lead;
}

interface ComparisonViewProps {
  base: string;
  target: string;
  load: Load<CompareResponse>;
  generation: Generation;
  dates: readonly string[] | null;
  split: boolean;
  demo: boolean | null;
  onGenerate: () => void;
  onRetry: () => void;
  onChoose: (base: string, target: string) => void;
}

function ComparisonView({
  base,
  target,
  load,
  generation,
  dates,
  split,
  demo,
  onGenerate,
  onRetry,
  onChoose,
}: ComparisonViewProps) {
  if (load.status === "loading") return <ComparisonSkeleton />;
  if (load.status === "error") {
    const failure = describeCompareError(load.error);
    return failure.kind === "not_found" ? (
      <NotFoundNotice message={failure.message} base={base} target={target} dates={dates} onChoose={onChoose} />
    ) : (
      <FailureNotice failure={failure.notice} onRetry={onRetry} />
    );
  }
  return (
    <ComparisonResult
      data={load.data}
      generation={generation}
      split={split}
      demo={demo}
      dates={dates}
      onGenerate={onGenerate}
      onChoose={onChoose}
    />
  );
}

function ComparisonResult({
  data,
  generation,
  split,
  demo,
  dates,
  onGenerate,
  onChoose,
}: {
  data: CompareResponse;
  generation: Generation;
  split: boolean;
  demo: boolean | null;
  dates: readonly string[] | null;
  onGenerate: () => void;
  onChoose: (base: string, target: string) => void;
}) {
  const [expanded, setExpanded] = useState<ReadonlySet<number>>(() => new Set());
  const [highlighted, setHighlighted] = useState<ReadonlySet<number>>(() => new Set());
  const timer = useRef<number | undefined>(undefined);
  const blocks = useMemo(() => buildDiffBlocks(data.changes, expanded), [data.changes, expanded]);

  useEffect(() => () => window.clearTimeout(timer.current), []);

  /**
   * Open the run that hides change `id` if needed, scroll its row into view (smoothly unless
   * `instant` or reduced motion), focus it and highlight `group`.
   */
  const jump = (id: number, group: number[], instant = false) => {
    const run = foldedRunOf(blocks, id);
    if (run !== null) flushSync(() => setExpanded((current) => new Set(current).add(run)));
    const row = document.getElementById(changeAnchor(id));
    if (!row) return;
    const rect = row.getBoundingClientRect();
    if (rect.top < 96 || rect.bottom > window.innerHeight) {
      const smooth = !instant && !prefersReducedMotion() && !document.hidden;
      row.scrollIntoView({ behavior: smooth ? "smooth" : "auto", block: "center" });
    }
    row.focus({ preventScroll: true });
    setHighlighted(new Set(group));
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setHighlighted(new Set()), HIGHLIGHT_MS);
  };

  // A link with #change-N (e.g. a key change opened in a new tab) lands on that row, at once
  // like any anchor.
  const jumpToHash = useEffectEvent(() => {
    const id = changeIdFromHash(window.location.hash);
    if (id !== null && data.changes.some((c) => c.id === id)) jump(id, [id], true);
  });
  useEffect(() => {
    // After the first paint; a timeout rather than requestAnimationFrame, which waits for a tab
    // opened in the background to be shown.
    const timeout = window.setTimeout(() => jumpToHash(), 0);
    return () => window.clearTimeout(timeout);
  }, []);

  const toggleRun = (runId: number) =>
    setExpanded((current) => {
      const next = new Set(current);
      if (next.has(runId)) next.delete(runId);
      else next.add(runId);
      return next;
    });

  return (
    <div className="motion-safe:animate-fade-in">
      <AnalysisPanel
        data={data}
        generation={generation}
        demo={demo}
        dates={dates}
        onGenerate={onGenerate}
        onJump={jump}
        onChoose={onChoose}
      />
      <DiffView
        base={data.base}
        target={data.target}
        stats={data.stats}
        blocks={blocks}
        split={split}
        highlighted={highlighted}
        onToggleRun={toggleRun}
        onToggleAll={(open) => setExpanded(open ? new Set(runIds(blocks)) : new Set())}
      />
    </div>
  );
}
