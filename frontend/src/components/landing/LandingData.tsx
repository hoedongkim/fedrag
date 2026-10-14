"use client";

import { createContext, type ReactNode, useCallback, useContext, useEffect, useState } from "react";

import { apiFetchAwake } from "@/lib/api/backend";
import type { CompareResponse, EvaluationReport, StanceBenchmark } from "@/lib/api/types";
import {
  askHeadline,
  compareHeadline,
  evaluationFacts,
  evaluationItems,
  type HeadlineResult,
  ragHeadline,
  SHOWCASE_PAIR,
  stanceHeadline,
} from "@/lib/landing/results";

import type { Load } from "../stance/useStanceData";
import { About, AboutList } from "../ui/About";
import { ExternalLink } from "../ui/ExternalLink";
import { skeletonClass, textButtonClass } from "../ui/styles";
import { useRetryWhenBackOnline } from "../useBackend";

interface LandingLoads {
  report: Load<EvaluationReport>;
  stance: Load<StanceBenchmark>;
  compare: Load<CompareResponse>;
  retry: () => void;
}

const LOADING = { status: "loading" } as const;
const LandingContext = createContext<LandingLoads | null>(null);

/**
 * Fetches what the overview quotes, once, from the API (which serves the tracked evaluation
 * files and the stored comparison), so the numbers can never drift from the results: the RAG
 * evaluation report, the stance benchmark and the showcase comparison (stored analysis only).
 */
export function LandingDataProvider({ children }: { children: ReactNode }) {
  const [report, setReport] = useState<Load<EvaluationReport>>(LOADING);
  const [stance, setStance] = useState<Load<StanceBenchmark>>(LOADING);
  const [compare, setCompare] = useState<Load<CompareResponse>>(LOADING);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    const { signal } = controller;
    const settle =
      <T,>(set: (load: Load<T>) => void) =>
      (promise: Promise<T>) =>
        promise
          .then((data) => set({ status: "ready", data }))
          .catch((error: unknown) => {
            if (!signal.aborted) set({ status: "error", error });
          });
    settle(setReport)(apiFetchAwake<EvaluationReport>("/api/evaluation", { signal }));
    settle(setStance)(apiFetchAwake<StanceBenchmark>("/api/stance/benchmark", { signal }));
    settle(setCompare)(
      apiFetchAwake<CompareResponse>("/api/compare", {
        method: "POST",
        json: { base_date: SHOWCASE_PAIR.base, target_date: SHOWCASE_PAIR.target, analysis: "cached_only" },
        signal,
      }),
    );
    return () => controller.abort();
  }, [attempt]);

  const retry = useCallback(() => {
    setReport(LOADING);
    setStance(LOADING);
    setCompare(LOADING);
    setAttempt((n) => n + 1);
  }, []);
  useRetryWhenBackOnline([report, stance, compare].some((load) => load.status === "error"), retry);

  return <LandingContext.Provider value={{ report, stance, compare, retry }}>{children}</LandingContext.Provider>;
}

function useLanding(): LandingLoads {
  const value = useContext(LandingContext);
  if (!value) throw new Error("Landing results must be inside LandingDataProvider.");
  return value;
}

export type HeadlineKind = "ask" | "stance" | "compare" | "rag";

function headlineFor(kind: HeadlineKind, loads: LandingLoads): Load<HeadlineResult | null> {
  const from = <T,>(load: Load<T>, pick: (data: T) => HeadlineResult | null): Load<HeadlineResult | null> =>
    load.status === "ready" ? { status: "ready", data: pick(load.data) } : load;
  switch (kind) {
    case "ask":
      return from(loads.report, askHeadline);
    case "rag":
      return from(loads.report, ragHeadline);
    case "stance":
      return from(loads.stance, stanceHeadline);
    case "compare":
      return from(loads.compare, compareHeadline);
  }
}

/** A row's headline number: the value, large, over its metric name (whose title says what it means). */
export function HeadlineSlot({ kind, hint }: { kind: HeadlineKind; hint: string }) {
  const loads = useLanding();
  const load = headlineFor(kind, loads);

  if (load.status === "loading") {
    return (
      <div className="min-h-[4.5rem]">
        <p className="sr-only">Loading the result…</p>
        <div aria-hidden="true">
          <div className={`h-9 w-24 ${skeletonClass}`} />
          <div className={`mt-2.5 h-3 w-44 max-w-full ${skeletonClass}`} />
        </div>
      </div>
    );
  }
  if (load.status === "error" || load.data === null) {
    return (
      <div className="min-h-[4.5rem] text-small text-faint">
        <p aria-hidden="true" className="text-[2.25rem] leading-none font-semibold text-line-strong">
          –
        </p>
        <p className="mt-2.5">
          Could not load this number.
          {load.status === "error" && (
            <>
              {" "}
              <button type="button" onClick={loads.retry} className={textButtonClass}>
                Try again
              </button>
            </>
          )}
        </p>
      </div>
    );
  }
  const { label, value, versus } = load.data;
  return (
    <div className="min-h-[4.5rem]">
      <p className="flex flex-wrap items-baseline gap-x-2">
        <span className="text-[2.25rem] leading-none font-semibold tracking-[-0.03em] text-ink tabular-nums">{value}</span>
        {versus && <span className="text-small text-faint tabular-nums">{versus}</span>}
      </p>
      <p className="mt-2.5 text-small text-faint">
        <span title={hint} className="cursor-help underline decoration-line-strong decoration-dotted underline-offset-[3px]">
          {label}
        </span>
      </p>
    </div>
  );
}

/**
 * The overview's "About these numbers": what each headline measures (its caption, live) and how
 * the evaluation was run, in plain words until the numbers load (or without them).
 */
export function LandingAbout({ limitationsHref }: { limitationsHref: string }) {
  const loads = useLanding();
  const { report, stance } = loads;
  const facts =
    report.status === "ready"
      ? evaluationFacts(report.data, stance.status === "ready" ? stance.data : null)
      : null;
  const headlines = (["ask", "stance", "compare", "rag"] as const)
    .map((kind) => headlineFor(kind, loads))
    .flatMap((load) => (load.status === "ready" && load.data ? [load.data] : []));

  return (
    <About>
      {headlines.length > 0 && <h3 className="pt-1 text-label font-medium text-faint">The numbers above</h3>}
      {headlines.length > 0 && (
        <AboutList
          items={headlines.map((h) => ({
            term: `${h.label}: ${h.value}${h.versus ? ` (${h.versus})` : ""}`,
            body: h.caption,
          }))}
        />
      )}
      <h3 className="pt-3 text-label font-medium text-faint">How it was tested</h3>
      <AboutList items={evaluationItems(facts).map((item) => ({ term: item.title, body: item.body }))} />
      <p>
        What it does not show: the test questions were written by an LLM and only some were checked by a
        person, the samples are small, the judge comes from the same vendor as the model it grades, and most of the
        retrieval gain is on questions that name a date.{" "}
        <ExternalLink href={limitationsHref}>Limitations</ExternalLink>
      </p>
    </About>
  );
}
