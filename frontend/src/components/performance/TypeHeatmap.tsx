"use client";

import { useLayoutEffect, useRef, useState } from "react";

import type { EvalMetric, EvaluationReport, QuestionType, ReportSplit, RunKey } from "@/lib/api/types";
import {
  configStyle,
  formatInterval,
  formatMetric,
  METRIC_DIGITS,
  methodLabel,
  METRICS,
  runLabel,
  TYPE_DESCRIPTIONS,
  TYPE_LABELS,
} from "@/lib/performance/format";
import {
  groupByConfig,
  HEAT_BOUNDS,
  type HeatCell,
  heatBin,
  heatmap,
  runKey,
  sameRun,
  valueOf,
} from "@/lib/performance/report";

import { GroupRow, rowCell, stickyCell, TableScroller, th } from "../DataTable";
import { Section } from "../ui/Section";
import { noteClass, overlayClass } from "../ui/styles";
import { Mark } from "./IntervalChart";
import { ChosenTag, MetricSelect } from "./parts";

const HEAT_METRICS = [
  "recall_at_5",
  "recall_at_10",
  "ndcg_at_10",
  "mrr_at_10",
  "faithfulness",
  "citation_accuracy",
  "answer_correctness",
  "refusal_accuracy",
] as const satisfies readonly EvalMetric[];
type HeatMetric = (typeof HEAT_METRICS)[number];

const BIN_LABELS = ["0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1"];

interface Tip {
  anchor: HTMLElement;
  run: RunKey;
  cell: HeatCell;
}

/**
 * One metric per question type and run: a grid colored on one blue ramp (darker is higher in
 * light mode, lighter is higher in dark mode), the value printed in each cell. Clicking a cell
 * opens those questions in the browser below.
 */
export function TypeHeatmap({
  report,
  split,
  highlight,
  onSelect,
}: {
  report: EvaluationReport;
  split: ReportSplit;
  highlight: RunKey | null;
  onSelect: (type: QuestionType, run: RunKey) => void;
}) {
  const [metric, setMetric] = useState<HeatMetric>("recall_at_5");
  const [tip, setTip] = useState<Tip | null>(null);
  const full = heatmap(report.per_type, split, metric, report.configs);
  // A run with no value in any cell (answers not generated or not judged) gets no row.
  const grid = { ...full, rows: full.rows.filter((r) => r.cells.some((c) => valueOf(c.estimate) !== null)) };
  const dropped = full.rows.length - grid.rows.length;
  const counts = new Map(
    grid.types.map((type) => [type, Math.max(0, ...grid.rows.map((r) => r.cells.find((c) => c.type === type)?.nQuestions ?? 0))]),
  );

  return (
    <Section
      id="perf-types"
      title="By question type"
      lead="Where each method struggles. Select a cell to list its questions."
      action={<MetricSelect value={metric} options={HEAT_METRICS} onChange={setMetric} />}
    >
      {grid.types.length === 0 ? (
        <p className="mt-6 text-small text-faint">No per-type results for {METRICS[metric].label} on this split.</p>
      ) : (
        <>
          <HeatLegend metricLabel={METRICS[metric].label} />
          <TableScroller label={`${METRICS[metric].label} by question type`}>
            <table className="w-full min-w-[40rem] border-separate border-spacing-0.5 text-small" onMouseLeave={() => setTip(null)}>
              <caption className="sr-only">
                {METRICS[metric].label} of each method under each config, by question type; select a cell to list
                those questions.
              </caption>
              <thead>
                <tr>
                  <th scope="col" className={`${th} ${stickyCell}`}>
                    Method
                  </th>
                  {grid.types.map((type) => (
                    <th key={type} scope="col" className={`${th} align-bottom`} title={TYPE_DESCRIPTIONS[type]}>
                      {TYPE_LABELS[type]}
                      <span className="block font-normal tabular-nums">n = {counts.get(type)}</span>
                    </th>
                  ))}
                </tr>
              </thead>
              {groupByConfig(grid.rows, report.configs).map(({ config, rows }) => (
                <tbody key={config.id}>
                  <GroupRow colSpan={grid.types.length + 1}>
                    <span className="inline-flex items-center gap-1.5">
                      <Mark style={configStyle(report.configs, config.id)} size={9} />
                      <span className="text-muted">{config.label}</span>
                    </span>
                  </GroupRow>
                  {rows.map((row) => (
                    <tr key={runKey(row)}>
                      <th scope="row" className={rowCell}>
                        <span className="flex items-center gap-2">
                          {methodLabel(row.method)}
                          {highlight && sameRun(highlight, row) && <ChosenTag />}
                        </span>
                      </th>
                      {row.cells.map((cell) => {
                        const value = valueOf(cell.estimate);
                        const bin = heatBin(value);
                        const label = `${METRICS[metric].label} ${formatMetric(value, METRIC_DIGITS)} for ${runLabel(report.configs, row)}, ${TYPE_LABELS[cell.type]}`;
                        return (
                          <td key={cell.type} className="p-0">
                            <button
                              type="button"
                              aria-label={`${label}. Show these questions.`}
                              onClick={() => onSelect(cell.type, row)}
                              onMouseEnter={(event) => setTip({ anchor: event.currentTarget, run: row, cell })}
                              onFocus={(event) => setTip({ anchor: event.currentTarget, run: row, cell })}
                              onBlur={() => setTip(null)}
                              className={[
                                "flex h-9 w-full min-w-16 items-center justify-center rounded-[3px] text-small tabular-nums transition-shadow hover:ring-2 hover:ring-ink/50 focus-visible:outline-offset-1",
                                bin === null ? "bg-subtle text-faint" : "",
                              ].join(" ")}
                              style={
                                bin === null ? undefined : { background: `var(--heat-${bin})`, color: `var(--heat-ink-${bin})` }
                              }
                            >
                              {value === null ? "–" : formatMetric(value, METRIC_DIGITS)}
                            </button>
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              ))}
            </table>
          </TableScroller>
        </>
      )}
      {dropped > 0 && (
        <p className={noteClass}>
          {`${dropped} run${dropped === 1 ? " has" : "s have"} no ${METRICS[metric].label.toLowerCase()} on this split (answers not generated or not judged) and ${dropped === 1 ? "is" : "are"} left out.`}
        </p>
      )}
      {tip && <HeatTooltip tip={tip} metric={metric} report={report} />}
    </Section>
  );
}

function HeatLegend({ metricLabel }: { metricLabel: string }) {
  return (
    <div aria-hidden="true" className="mt-6 flex flex-wrap items-center gap-x-3 gap-y-1 text-label text-faint">
      <span className="mr-1 font-medium text-muted">{metricLabel}</span>
      {[0, 1, 2, 3, HEAT_BOUNDS.length].map((bin) => (
        <span key={bin} className="inline-flex items-center gap-1">
          <span className="size-2.5 rounded-[2px]" style={{ background: `var(--heat-${bin})` }} />
          <span className="tabular-nums">{BIN_LABELS[bin]}</span>
        </span>
      ))}
      <span className="inline-flex items-center gap-1">
        <span className="size-2.5 rounded-[2px] bg-subtle ring-1 ring-line ring-inset" />
        none
      </span>
    </div>
  );
}

function HeatTooltip({ tip, metric, report }: { tip: Tip; metric: EvalMetric; report: EvaluationReport }) {
  const ref = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const box = ref.current;
    if (!box) return;
    const place = () => {
      const anchor = tip.anchor.getBoundingClientRect();
      const width = box.offsetWidth;
      const viewport = document.documentElement.clientWidth;
      const left = Math.max(8, Math.min(anchor.left + anchor.width / 2 - width / 2, viewport - width - 8));
      const above = anchor.top - box.offsetHeight - 6;
      box.style.left = `${left}px`;
      box.style.top = `${above >= 8 ? above : anchor.bottom + 6}px`;
      box.style.visibility = "visible";
    };
    place();
    window.addEventListener("scroll", place, { capture: true, passive: true });
    window.addEventListener("resize", place);
    return () => {
      window.removeEventListener("scroll", place, { capture: true });
      window.removeEventListener("resize", place);
    };
  }, [tip]);

  const estimate = tip.cell.estimate;
  const value = valueOf(estimate);
  const interval = formatInterval(estimate?.ci, METRIC_DIGITS);
  return (
    <div
      ref={ref}
      aria-hidden="true"
      style={{ visibility: "hidden" }}
      className={`pointer-events-none fixed z-50 w-max max-w-64 ${overlayClass}`}
    >
      <p>
        <span className="font-semibold text-ink tabular-nums">{formatMetric(value, METRIC_DIGITS)}</span>
        {interval && <span className="ml-1 text-faint tabular-nums">({interval})</span>}
        <span className="ml-1">{METRICS[metric].label}</span>
      </p>
      <p className="mt-0.5">
        {runLabel(report.configs, tip.run)} · {TYPE_LABELS[tip.cell.type]}
      </p>
      <p className="text-faint">
        {estimate?.n !== null && estimate?.n !== undefined
          ? `${estimate.n} of ${tip.cell.nQuestions} questions scored`
          : `${tip.cell.nQuestions} questions`}
        {" · click to list them"}
      </p>
    </div>
  );
}
