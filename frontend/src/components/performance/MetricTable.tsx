import type { EvalDifference, EvalMetric, Estimate, RetrievalMethod, RunKey } from "@/lib/api/types";
import {
  configStyle,
  formatDelta,
  formatLevel,
  formatMetric,
  METRIC_DIGITS,
  METRICS,
  runLabel,
} from "@/lib/performance/format";
import { bestRun, groupByConfig, isResolved, runKey, sameRun, valueOf } from "@/lib/performance/report";

import { ChevronRightIcon } from "../ask/icons";
import { GroupRow, MetricCell, stickyCell, TableScroller, th } from "../DataTable";
import { Mark } from "./IntervalChart";
import { RunNameCell } from "./parts";

type MetricRow = { method: RetrievalMethod; config: string } & Partial<Record<EvalMetric, Estimate | null>>;

interface MetricTableProps<T extends MetricRow> {
  label: string;
  caption: string;
  rows: T[];
  metrics: readonly EvalMetric[];
  configs: { id: string; label: string }[];
  highlight: RunKey | null;
  /** Shows n under each value (when n varies between columns, e.g. refusals). */
  showN?: boolean;
}

/**
 * Runs as rows grouped by config, metrics as columns with their intervals; the best value per
 * column in bold (lowest for a lower-is-better metric).
 */
export function MetricTable<T extends MetricRow>({
  label,
  caption,
  rows,
  metrics,
  configs,
  highlight,
  showN = false,
}: MetricTableProps<T>) {
  const best = Object.fromEntries(
    metrics.map((metric) => [metric, bestRun(rows, (r) => valueOf(r[metric]), METRICS[metric].lowerIsBetter)]),
  ) as Record<EvalMetric, string | null>;
  const columns = metrics.length + 1;

  return (
    <TableScroller label={label}>
      <table className="w-full min-w-[44rem] border-collapse text-small">
        <caption className="sr-only">{caption}</caption>
        <thead className="border-b border-line">
          <tr>
            <th scope="col" className={`${th} ${stickyCell}`}>
              Method
            </th>
            {metrics.map((metric) => (
              <th key={metric} scope="col" className={th} title={METRICS[metric].description}>
                {METRICS[metric].label}
                {METRICS[metric].lowerIsBetter && <span className="font-normal"> (lower is better)</span>}
              </th>
            ))}
          </tr>
        </thead>
        {groupByConfig(rows, configs).map(({ config, rows: group }) => (
          <tbody
            key={config.id}
            className="divide-y divide-line text-muted [&:not(:first-of-type)]:border-t [&:not(:first-of-type)]:border-line-strong"
          >
            <GroupRow colSpan={columns}>
              <span className="inline-flex items-center gap-1.5">
                <Mark style={configStyle(configs, config.id)} size={9} />
                <span className="text-muted">{config.label}</span>
              </span>
            </GroupRow>
            {group.map((row) => (
              <tr key={runKey(row)}>
                <RunNameCell
                  method={row.method}
                  config={row.config}
                  configs={configs}
                  highlight={!!highlight && sameRun(highlight, row)}
                />
                {metrics.map((metric) => {
                  const estimate = row[metric] ?? null;
                  return (
                    <MetricCell
                      key={metric}
                      best={best[metric] === runKey(row)}
                      ci={estimate?.ci}
                      digits={METRIC_DIGITS}
                      note={showN && estimate?.n !== null && estimate?.n !== undefined ? `n = ${estimate.n}` : undefined}
                    >
                      {formatMetric(valueOf(estimate), METRIC_DIGITS)}
                    </MetricCell>
                  );
                })}
              </tr>
            ))}
          </tbody>
        ))}
      </table>
    </TableScroller>
  );
}

/** Lists longer than this start folded. */
const OPEN_DIFFERENCES = 6;

/**
 * The report's paired differences of one metric (the one the chart shows), one line each,
 * marked when the interval excludes 0.
 */
export function DifferenceList({
  differences,
  configs,
  level,
  metricLabel,
}: {
  differences: EvalDifference[];
  configs: { id: string; label: string }[];
  level: number | null;
  metricLabel: string;
}) {
  if (differences.length === 0) return null;
  const resolved = differences.filter((d) => isResolved(d.ci)).length;
  return (
    <details open={differences.length <= OPEN_DIFFERENCES} className="group mt-4 text-label text-faint">
      <summary className="inline-flex list-none items-center gap-1.5 rounded-sm font-medium text-muted transition-colors select-none hover:text-ink">
        <ChevronRightIcon className="size-3 transition-transform group-open:rotate-90 motion-reduce:transition-none" />
        Paired differences in {metricLabel}: {resolved} of {differences.length} clear of zero
      </summary>
      <p className="mt-2 max-w-[70ch]">
        Both runs scored on the same resampled questions{level ? ` (${formatLevel(level)} intervals)` : ""}. An
        interval that includes 0 means the evaluation cannot tell the two apart.
      </p>
      <ul className="mt-1 space-y-0.5">
        {differences.map((d) => (
          <li key={`${d.metric}-${runKey(d.minuend)}-${runKey(d.subtrahend)}`}>
            {runLabel(configs, d.minuend)} − {runLabel(configs, d.subtrahend)}:{" "}
            <span className="text-muted tabular-nums">{formatDelta(d.difference, d.ci)}</span>
            {isResolved(d.ci) && <span className="ml-1 font-medium text-ink">clear of 0</span>}
            <span> · n = {d.n}</span>
          </li>
        ))}
      </ul>
    </details>
  );
}
