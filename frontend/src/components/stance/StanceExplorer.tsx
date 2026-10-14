"use client";

import { useSearchParams } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";

import type { StanceClassifier, StanceTimeline } from "@/lib/api/types";
import { formatDate } from "@/lib/ask/format";
import { CLASSIFIER_IDS, classifierInfo } from "@/lib/stance/classifiers";
import { describeStanceError } from "@/lib/stance/failures";
import { STANCE_LABELS } from "@/lib/stance/format";
import {
  FIRST_YEAR,
  parseStanceParams,
  serializeStanceParams,
  type StanceViewState,
  toggleClassifier,
} from "@/lib/stance/params";
import {
  adjacentMeetings,
  buildMeetingRows,
  mergeFedFunds,
  rowsInYears,
  type Timelines,
} from "@/lib/stance/series";

import { skeletonClass } from "../ui/styles";
import { BenchmarkSection } from "./BenchmarkSection";
import { FailureNotice } from "./FailureNotice";
import { MeetingPanel } from "./MeetingPanel";
import { StanceAbout } from "./StanceAbout";
import { StanceChart } from "./StanceChart";
import { StanceFilters } from "./StanceFilters";
import { StanceTable } from "./StanceTable";
import { useStanceData } from "./useStanceData";

const URL_SYNC_DELAY_MS = 300;

function prefersReducedMotion(): boolean {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/**
 * Policy Stance: the stance timeline of every FOMC statement for the chosen classifiers, a panel
 * for one meeting, and the benchmark. The view (classifiers, meeting, years) lives in the URL.
 */
export function StanceExplorer() {
  const searchParams = useSearchParams();
  const [view, setView] = useState<StanceViewState>(() => parseStanceParams(searchParams));
  const [detailChoice, setDetailChoice] = useState<StanceClassifier | null>(null);
  const { timelines: loads, benchmark, retry } = useStanceData();
  const panelRef = useRef<HTMLDivElement>(null);

  const loaded: Timelines = useMemo(() => {
    const ready: Timelines = {};
    for (const id of CLASSIFIER_IDS) {
      const load = loads[id];
      if (load.status === "ready") ready[id] = load.data as StanceTimeline;
    }
    return ready;
  }, [loads]);
  const allRows = useMemo(() => buildMeetingRows(loaded), [loaded]);
  const fedFunds = useMemo(() => mergeFedFunds(loaded), [loaded]);

  const lastYear = allRows.length ? Number(allRows[allRows.length - 1].date.slice(0, 4)) : FIRST_YEAR;
  const firstYear = allRows.length ? Number(allRows[0].date.slice(0, 4)) : FIRST_YEAR;
  const years = Array.from({ length: lastYear - firstYear + 1 }, (_, i) => firstYear + i);
  const fromYear = Math.min(Math.max(view.fromYear ?? firstYear, firstYear), lastYear);
  const toYear = Math.max(Math.min(view.toYear ?? lastYear, lastYear), fromYear);
  const rows = useMemo(() => rowsInYears(allRows, fromYear, toYear), [allRows, fromYear, toYear]);

  const plotted = view.classifiers.filter((id) => loads[id].status === "ready");
  const selectedRow = rows.find((r) => r.date === view.meeting) ?? rows[rows.length - 1] ?? null;
  const detail =
    [detailChoice, ...plotted, ...CLASSIFIER_IDS].find(
      (id): id is StanceClassifier => id !== null && !!selectedRow?.results[id],
    ) ?? plotted[0] ?? "llm_few_shot";

  // Mirror the view in the query string without a history entry per change.
  useEffect(() => {
    const timer = window.setTimeout(() => {
      const query = serializeStanceParams(view);
      const { pathname, search, hash } = window.location;
      const next = query ? `?${query}` : "";
      if (next !== search) window.history.replaceState(null, "", `${pathname}${next}${hash}`);
    }, URL_SYNC_DELAY_MS);
    return () => window.clearTimeout(timer);
  }, [view]);

  const update = (patch: Partial<StanceViewState>) => setView((current) => ({ ...current, ...patch }));

  const selectFromChart = (date: string) => {
    update({ meeting: date });
    window.requestAnimationFrame(() => {
      const panel = panelRef.current;
      if (panel && panel.getBoundingClientRect().top > window.innerHeight * 0.75) {
        panel.scrollIntoView({ behavior: prefersReducedMotion() ? "auto" : "smooth", block: "start" });
      }
    });
  };

  const statuses = CLASSIFIER_IDS.map((id) => loads[id].status);
  const loading = statuses.every((s) => s === "loading");
  const failedAll = statuses.every((s) => s === "error");
  const firstError = CLASSIFIER_IDS.map((id) => loads[id]).find((l) => l.status === "error");

  return (
    <div className="mt-12">
      <section aria-labelledby="stance-timeline">
        <h2 id="stance-timeline" className="sr-only">
          Stance of every FOMC statement since 2015
        </h2>
        <StanceFilters
          loads={loads}
          plotted={plotted}
          onToggle={(id) => update({ classifiers: toggleClassifier(view.classifiers, id) })}
          years={years}
          fromYear={fromYear}
          toYear={toYear}
          onYears={(from, to) =>
            update({
              fromYear: from === firstYear ? null : from,
              toYear: to === lastYear ? null : to,
            })
          }
        />

        <div className="mt-8">
          {loading ? (
            <div aria-hidden="true" className={`h-[30rem] ${skeletonClass}`} />
          ) : failedAll && firstError?.status === "error" ? (
            <FailureNotice failure={describeStanceError(firstError.error)} onRetry={retry} />
          ) : plotted.length === 0 ? (
            <p className="text-body text-muted">No method is shown. Pick one above.</p>
          ) : rows.length === 0 ? (
            <p className="text-body text-muted">
              No meetings in {fromYear}–{toYear}.
            </p>
          ) : (
            <>
              <StanceChart
                rows={rows}
                fedFunds={fedFunds}
                plotted={plotted}
                selected={selectedRow?.date ?? null}
                onSelect={selectFromChart}
              />
              <StanceTable
                rows={rows}
                classifiers={CLASSIFIER_IDS.filter((id) => loads[id].status === "ready")}
                fedFunds={fedFunds}
              />
            </>
          )}
        </div>
      </section>

      {selectedRow && (
        <div ref={panelRef} className="mt-20 scroll-mt-6">
          <MeetingPanel
            row={selectedRow}
            rows={rows}
            loads={loads}
            fedFunds={fedFunds}
            detail={detail}
            onDetail={setDetailChoice}
            onSelect={(date) => update({ meeting: date })}
            previousStatement={adjacentMeetings(allRows, selectedRow.date).previous}
          />
        </div>
      )}

      <BenchmarkSection load={benchmark} onRetry={retry} />

      <StanceAbout benchmark={benchmark.status === "ready" ? benchmark.data : null} />

      <p role="status" aria-live="polite" className="sr-only">
        {selectedRow
          ? `Showing the ${formatDate(selectedRow.date)} statement: ${plotted
              .map((id) => {
                const result = selectedRow.results[id];
                return result ? `${classifierInfo(id).label} ${STANCE_LABELS[result.stance]}` : null;
              })
              .filter(Boolean)
              .join(", ")}.`
          : ""}
      </p>
    </div>
  );
}
