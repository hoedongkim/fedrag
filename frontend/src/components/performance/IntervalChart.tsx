"use client";

import { useState } from "react";

import type { Estimate, RetrievalMethod, RunKey } from "@/lib/api/types";
import {
  type ConfigStyle,
  configLabel,
  configStyle,
  formatInterval,
  formatMetric,
  formatN,
  METRIC_DIGITS,
  methodLabel,
} from "@/lib/performance/format";
import { METHOD_ORDER, runKey, sameRun, toPercent, valueOf } from "@/lib/performance/report";

import { overlayClass } from "../ui/styles";

export interface IntervalPoint {
  method: RetrievalMethod;
  config: string;
  estimate: Estimate;
}

interface IntervalChartProps {
  points: IntervalPoint[];
  configs: { id: string; label: string }[];
  metricLabel: string;
  /** What the metric means, in a few words (shown after its name). */
  metricDescription?: string;
  /** The run to ring (the headline run). */
  highlight?: RunKey | null;
}

const TICKS = [0, 0.25, 0.5, 0.75, 1];
const ROW_GAP = 16; // px between the configs of one method
const ROW_PAD = 10; // px above the first and below the last config of a method

/**
 * One metric for every run: a dot at the mean and a whisker over its bootstrap interval, a row
 * per method with one line per config (color and shape). Hidden from screen readers: the table
 * below it has the same values. Hovering a mark shows its value, interval and n.
 */
export function IntervalChart({ points, configs, metricLabel, metricDescription, highlight }: IntervalChartProps) {
  const [hover, setHover] = useState<string | null>(null);
  const methods = METHOD_ORDER.filter((m) => points.some((p) => p.method === m));
  const shown = configs.filter((c) => points.some((p) => p.config === c.id));
  const rowHeight = ROW_PAD * 2 + ROW_GAP * Math.max(0, shown.length - 1);
  const hovered = points.find((p) => runKey(p) === hover) ?? null;

  if (methods.length === 0) {
    return <p className="mt-6 text-small text-faint">No values for {metricLabel}.</p>;
  }

  return (
    <figure className="mt-6">
      <figcaption className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1.5 text-label text-faint">
        <span>
          <span className="font-medium text-muted">{metricLabel}</span>
          {metricDescription ? `: ${metricDescription.replace(/\.$/, "")}` : ""}. Dot: mean; line: 95% interval.
        </span>
        <span className="inline-flex flex-wrap items-center gap-x-4 gap-y-1">
          {shown.map((config) => (
            <span key={config.id} className="inline-flex items-center gap-1.5">
              <Mark style={configStyle(configs, config.id)} size={10} />
              {config.label}
            </span>
          ))}
        </span>
      </figcaption>

      <div aria-hidden="true" className="mt-4 grid grid-cols-[6.5rem_minmax(0,1fr)] gap-x-3 select-none sm:grid-cols-[8.5rem_minmax(0,1fr)]">
        <div>
          {methods.map((method) => (
            <div
              key={method}
              style={{ height: rowHeight }}
              className="flex items-center text-small text-muted"
            >
              {methodLabel(method)}
            </div>
          ))}
        </div>

        <div className="relative mr-3" onMouseLeave={() => setHover(null)}>
          {TICKS.map((tick) => (
            <div
              key={tick}
              className="pointer-events-none absolute top-0 bottom-0 w-px"
              style={{
                left: `${toPercent(tick)}%`,
                background: tick === 0 ? "var(--chart-axis)" : "var(--chart-grid)",
              }}
            />
          ))}
          {methods.map((method) => (
            <div key={method} className="relative" style={{ height: rowHeight }}>
              {shown.map((config, index) => {
                const point = points.find((p) => p.method === method && p.config === config.id);
                const value = valueOf(point?.estimate);
                if (!point || value === null) return null;
                const key = runKey(point);
                const style = configStyle(configs, config.id);
                const ci = point.estimate.ci;
                const top = ROW_PAD + index * ROW_GAP;
                const active = hover === key;
                const ringed = !!highlight && sameRun(highlight, point);
                return (
                  <div key={key}>
                    {ci && (
                      <div
                        className="pointer-events-none absolute h-[1.5px] -translate-y-1/2 rounded-full"
                        style={{
                          top,
                          left: `${toPercent(ci.low)}%`,
                          width: `${Math.max(0.5, toPercent(ci.high) - toPercent(ci.low))}%`,
                          background: style.color,
                          opacity: hover && !active ? 0.35 : 0.9,
                        }}
                      />
                    )}
                    <div
                      className="absolute -translate-x-1/2 -translate-y-1/2"
                      style={{ top, left: `${toPercent(value)}%`, opacity: hover && !active ? 0.45 : 1 }}
                    >
                      <Mark style={style} size={active ? 12 : 10} ring={ringed} />
                    </div>
                    {/* Hit target: the whole whisker plus a margin, at least 24px tall. */}
                    <div
                      className="absolute h-6 -translate-y-1/2 cursor-default"
                      style={{
                        top,
                        left: `calc(${toPercent(Math.min(ci?.low ?? value, value))}% - 10px)`,
                        width: `calc(${toPercent(Math.max(ci?.high ?? value, value)) - toPercent(Math.min(ci?.low ?? value, value))}% + 20px)`,
                      }}
                      onMouseEnter={() => setHover(key)}
                    />
                  </div>
                );
              })}
            </div>
          ))}

          {hovered && valueOf(hovered.estimate) !== null && (
            <ChartTooltip
              point={hovered}
              label={`${methodLabel(hovered.method)} · ${configLabel(configs, hovered.config)}`}
              style={configStyle(configs, hovered.config)}
              row={methods.indexOf(hovered.method)}
              rowHeight={rowHeight}
            />
          )}
        </div>

        <div />
        <div className="relative mt-1.5 mr-3 h-4 text-label text-faint tabular-nums">
          {TICKS.map((tick) => (
            <span
              key={tick}
              className="absolute -translate-x-1/2"
              style={{ left: `${toPercent(tick)}%` }}
            >
              {tick === 0 || tick === 1 ? tick : tick.toFixed(2)}
            </span>
          ))}
        </div>
      </div>
    </figure>
  );
}

function ChartTooltip({
  point,
  label,
  style,
  row,
  rowHeight,
}: {
  point: IntervalPoint;
  label: string;
  style: ConfigStyle;
  row: number;
  rowHeight: number;
}) {
  const value = valueOf(point.estimate) ?? 0;
  const interval = formatInterval(point.estimate.ci, METRIC_DIGITS);
  // Above the row, flipped toward the middle so it never leaves the plot sideways.
  const left = toPercent(value);
  const align = left > 60 ? "-translate-x-full" : left < 40 ? "" : "-translate-x-1/2";
  return (
    <div
      className={`pointer-events-none absolute z-20 w-max max-w-56 -translate-y-full ${overlayClass} ${align}`}
      style={{ top: row * rowHeight + 4, left: `${left}%` }}
    >
      <p className="flex items-center gap-2">
        <span className="h-0.5 w-3 shrink-0 rounded-full" style={{ background: style.color }} />
        <span className="font-semibold text-ink tabular-nums">{formatMetric(value, METRIC_DIGITS)}</span>
        {interval && <span className="text-faint tabular-nums">({interval})</span>}
      </p>
      <p className="mt-0.5">{label}</p>
      {point.estimate.n !== null && <p className="text-faint">{formatN(point.estimate.n)} questions</p>}
    </div>
  );
}

/** A config's mark: its color in its shape, with a 2px surface ring (and an outer ring if set). */
export function Mark({ style, size, ring = false }: { style: ConfigStyle; size: number; ring?: boolean }) {
  const inner =
    style.shape === "circle" ? (
      <circle cx="6" cy="6" r="4.5" />
    ) : style.shape === "diamond" ? (
      <path d="M6 0.8 L11.2 6 L6 11.2 L0.8 6 Z" />
    ) : (
      <rect x="1.5" y="1.5" width="9" height="9" rx="1" />
    );
  return (
    <svg
      viewBox="0 0 12 12"
      width={size}
      height={size}
      aria-hidden="true"
      className="block shrink-0 overflow-visible"
    >
      {ring && <circle cx="6" cy="6" r="8.5" fill="none" stroke="var(--chart-ink)" strokeWidth="1.25" />}
      <g fill={style.color} stroke="var(--chart-surface)" strokeWidth="1.5" paintOrder="stroke">
        {inner}
      </g>
    </svg>
  );
}
