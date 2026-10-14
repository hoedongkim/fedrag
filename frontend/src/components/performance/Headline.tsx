import type { ReactNode } from "react";

import type { EvalMetric, EvaluationReport, ReportSplit, RunKey } from "@/lib/api/types";
import {
  configLabel,
  formatDelta,
  formatInterval,
  formatMetric,
  formatMs,
  formatUsd,
  METRIC_DIGITS,
  METRICS,
  methodLabel,
  runLabel,
  SPLIT_LABELS,
} from "@/lib/performance/format";
import {
  estimateFor,
  type Headline as HeadlineRun,
  isResolved,
  pairedDifference,
  rowFor,
  valueOf,
} from "@/lib/performance/report";

import { labelTextClass } from "../ui/styles";

const KNOWN_ERRORS_PREFIX = "Known judge errors";

const TILE_METRICS: EvalMetric[] = [
  "recall_at_5",
  "ndcg_at_10",
  "faithfulness",
  "citation_accuracy",
  "answer_correctness",
  "refusal_accuracy",
];

interface HeadlineProps {
  report: EvaluationReport;
  split: ReportSplit;
  headline: HeadlineRun;
  baseline: RunKey | null;
}

/**
 * The chosen run in a few numbers on the current split, each against the baseline: the setup
 * Ask the Fed shipped with, or the one a test-split rule replaced.
 */
export function Headline({ report, split, headline, baseline }: HeadlineProps) {
  const { run } = headline;
  const retrieval = rowFor(report.retrieval, split, run);
  const answers = rowFor(report.answers, split, run);
  const baseRetrieval = baseline ? rowFor(report.retrieval, split, baseline) : null;
  const baseAnswers = baseline ? rowFor(report.answers, split, baseline) : null;
  const splitInfo = report.dataset.splits.find((s) => s.split === split);
  // A rule applied once to test (the default-switch rule) made this run the default.
  const byRule = headline.fromReport && headline.chosenOn === "test";
  const versus = baseline ? (byRule ? `vs ${configLabel(report.configs, baseline.config)}` : "vs shipped") : "";
  const n = split === "all" ? report.dataset.n_questions : splitInfo?.n;
  // The report notes judge verdicts of this run found wrong on reading (technical report, "TEST results").
  const knownErrors = report.notes.some(
    (note) => note.startsWith(KNOWN_ERRORS_PREFIX) && note.includes(` ${run.config} `),
  );

  return (
    <section aria-labelledby="perf-headline" className="mt-14">
      <p className={labelTextClass}>{byRule ? "Ask the Fed’s default" : "Chosen setup"}</p>
      <h2 id="perf-headline" className="mt-0.5 text-heading font-semibold text-ink">
        {methodLabel(run.method)} <span className="font-normal text-faint">·</span>{" "}
        {configLabel(report.configs, run.config)}
      </h2>
      <p className="mt-1 max-w-[70ch] text-body text-muted">
        {split === "all" ? `All ${n ?? ""} questions` : `${n ?? ""} ${SPLIT_LABELS[split].toLowerCase()} questions`}
        {split === "test"
          ? headline.chosenOn === "dev"
            ? ", none used in the choice"
            : ", none used in tuning"
          : " (optimistic: the settings were chosen on dev)"}
        {baseline
          ? `. Compared with ${runLabel(report.configs, baseline)}, the setup ${byRule ? "it replaced" : "Ask the Fed shipped with"}.`
          : "."}
      </p>

      <dl className="mt-8 grid grid-cols-2 gap-x-8 gap-y-8 sm:grid-cols-4">
        {TILE_METRICS.map((metric) => {
          const estimate = estimateFor(metric, retrieval, answers);
          const base = estimateFor(metric, baseRetrieval, baseAnswers);
          const diff = baseline ? pairedDifference(report.differences, metric, split, run, baseline) : null;
          const value = valueOf(estimate);
          const baseValue = valueOf(base);
          return (
            <Stat key={metric} label={METRICS[metric].label} title={METRICS[metric].description}>
              <dd className="mt-2 flex flex-wrap items-baseline gap-x-2">
                <Value>
                  {formatMetric(value, METRIC_DIGITS)}
                  {knownErrors && metric === "refusal_accuracy" && answers && (
                    <span className="text-small font-normal text-faint" title="Includes a judge error: see About these numbers">
                      *
                    </span>
                  )}
                </Value>
                {diff && (
                  <Delta
                    difference={diff.difference}
                    detail={`${versus}: ${formatDelta(diff.difference, diff.ci)} (95% interval)`}
                    resolved={isResolved(diff.ci)}
                    lowerIsBetter={METRICS[metric].lowerIsBetter}
                  />
                )}
              </dd>
              {diff ? (
                <dd className="mt-1.5 text-label text-muted">
                  {versus} · <Verdict difference={diff.difference} resolved={isResolved(diff.ci)} lowerIsBetter={METRICS[metric].lowerIsBetter} />
                </dd>
              ) : baseline && baseValue !== null ? (
                <dd className="mt-1.5 text-label text-muted tabular-nums">
                  {byRule ? configLabel(report.configs, baseline.config) : "shipped"} {formatMetric(baseValue, METRIC_DIGITS)}
                </dd>
              ) : null}
              <dd className="mt-0.5 text-label text-faint tabular-nums" title="95% interval and number of questions">
                {formatInterval(estimate?.ci, METRIC_DIGITS) ?? "no interval"}
                {estimate?.n !== null && estimate?.n !== undefined && ` · n = ${estimate.n}`}
              </dd>
            </Stat>
          );
        })}
        <Stat
          label="Time, median"
          title={`End to end per question: ${answers ? "the search and the answer" : "the search only"}. p95: the slowest 5% take longer.`}
        >
          <dd className="mt-2">
            <Value>{formatMs(answers?.latency_ms_p50 ?? retrieval?.latency_ms_p50)}</Value>
          </dd>
          <dd className="mt-1.5 text-label text-muted tabular-nums">
            {answers ? "" : "search only · "}p95 {formatMs(answers?.latency_ms_p95 ?? retrieval?.latency_ms_p95)}
          </dd>
          {answers && (
            <dd className="mt-0.5 text-label text-faint tabular-nums">search alone {formatMs(retrieval?.latency_ms_p50)}</dd>
          )}
        </Stat>
        <Stat
          label="Cost per question"
          title="OpenAI cost of the answer for one question, estimated from token counts (the query embedding, about $0.000001, is left out)."
        >
          <dd className="mt-2">
            <Value>{formatUsd(answers?.cost_per_query_usd)}</Value>
          </dd>
          <dd className="mt-1.5 text-label text-muted">the answer only</dd>
          {answers?.judge_cost_per_query_usd !== null && answers?.judge_cost_per_query_usd !== undefined && (
            <dd className="mt-0.5 text-label text-faint tabular-nums">
              judging it {formatUsd(answers.judge_cost_per_query_usd)}
            </dd>
          )}
        </Stat>
      </dl>
      {!answers && (
        <p className="mt-4 text-small text-faint">No answers were judged for this run on this split, so its answer numbers are n/a.</p>
      )}
    </section>
  );
}

function Stat({ label, title, children }: { label: string; title: string; children: ReactNode }) {
  return (
    <div className="min-w-0 border-t border-line pt-4">
      <dt className="text-small text-muted">
        <span title={title} className="cursor-help underline decoration-line-strong decoration-dotted underline-offset-4">
          {label}
        </span>
      </dt>
      {children}
    </div>
  );
}

/** A headline value: large, in proportional figures (tabular ones look loose at this size). */
function Value({ children }: { children: ReactNode }) {
  return <span className="text-[1.75rem] leading-none font-semibold tracking-[-0.02em] text-ink">{children}</span>;
}

/** The change against the baseline beside a value: colored only when it is clear of noise. */
function Delta({
  difference,
  detail,
  resolved,
  lowerIsBetter,
}: {
  difference: number;
  /** The difference with its interval, for the tooltip and screen readers. */
  detail: string;
  resolved: boolean;
  lowerIsBetter?: boolean;
}) {
  const better = lowerIsBetter ? difference < 0 : difference > 0;
  const tone = !resolved ? "text-faint" : better ? "text-good" : "text-bad";
  return (
    <span title={detail} className={`text-small font-medium tabular-nums ${tone}`}>
      <span aria-hidden="true">{formatDelta(difference)}</span>
      <span className="sr-only">{detail}</span>
    </span>
  );
}

/** Whether a paired difference is clear of noise, and which way (the delta beside the value carries the color). */
function Verdict({ difference, resolved, lowerIsBetter }: { difference: number; resolved: boolean; lowerIsBetter?: boolean }) {
  if (!resolved) return <>within noise</>;
  const better = lowerIsBetter ? difference < 0 : difference > 0;
  return <>{better ? "clear gain" : "clear drop"}</>;
}
