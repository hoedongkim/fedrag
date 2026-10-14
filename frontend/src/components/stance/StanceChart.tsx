"use client";

import { type ReactNode, type RefObject, useEffect, useMemo, useRef, useState } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  LineChart,
  ReferenceArea,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  type TooltipContentProps,
  XAxis,
  YAxis,
} from "recharts";
import type { MouseHandlerDataParam, SyncMethod } from "recharts/types/synchronisation/types";

import type { FedFundsTarget, StanceClassifier } from "@/lib/api/types";
import { formatDate } from "@/lib/ask/format";
import {
  classifierInfo,
  type ClassifierInfo,
  SENTENCE_THRESHOLD,
  seriesDash,
} from "@/lib/stance/classifiers";
import {
  formatRange,
  formatRate,
  formatScore,
  REFERENCE_SOURCE_LABELS,
  STANCE_LABELS,
} from "@/lib/stance/format";
import {
  endLabelsFit,
  fedFundsCeiling,
  fedFundsSeries,
  type MeetingRow,
  nearestMeeting,
  rangeAfterMeeting,
  timeDomain,
  timeTicks,
} from "@/lib/stance/series";

import { overlayClass } from "../ui/styles";
import { SeriesKey } from "./SeriesKey";

// Every panel uses the same margins and y-axis width, so their plot areas line up on one x scale.
const Y_AXIS_WIDTH = 52;
const MARGIN_TOP = 8;
const SCORE_HEIGHT = 240;
const STRIP_HEIGHT = 76;
const RATE_HEIGHT = 132;
const END_LABEL_SPACE = 96;
const SYNC_ID = "stance-timeline";
const TICK = { fill: "var(--chart-tick)", fontSize: 12 };
const SCORE_TICKS = [-1, -0.5, 0, 0.5, 1];
const LANE_LABELS: Record<number, string> = { 1: "Hawkish", 0: "Neutral", [-1]: "Dovish" };

/** Synchronise the panels by time, not by index: their rows sit at different dates. */
const syncByTime: SyncMethod = (ticks, data: MouseHandlerDataParam) => {
  const t = Number(data.activeLabel);
  if (!Number.isFinite(t) || ticks.length === 0) return -1;
  let best = 0;
  ticks.forEach((tick, i) => {
    if (Math.abs(Number(tick.value) - t) < Math.abs(Number(ticks[best].value) - t)) best = i;
  });
  return ticks[best].index ?? best;
};

/** The element's content width; 0 until it is first measured. */
function useWidth(ref: RefObject<HTMLElement | null>): number {
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => setWidth(Math.round(entry.contentRect.width)));
    observer.observe(element);
    return () => observer.disconnect();
  }, [ref]);
  return width;
}

const scoreTick = (v: number) => (v === 0 ? "0" : `${v > 0 ? "+" : "−"}${Math.abs(v)}`);
const rateTick = (v: number) => `${formatRate(v)}%`;

interface StanceChartProps {
  rows: MeetingRow[];
  /** Change points of the target range (all loaded, any window). */
  fedFunds: FedFundsTarget[];
  /** Classifiers to draw, in display order. */
  plotted: StanceClassifier[];
  selected: string | null;
  onSelect: (meetingDate: string) => void;
}

/**
 * The stance timeline as three panels on one time axis: each plotted classifier's score
 * (-1 dovish to +1 hawkish), the statements' reference labels, and the federal funds target
 * range. Separate panels, not a second y axis: the score and the rate have unrelated scales.
 */
export function StanceChart({ rows, fedFunds, plotted, selected, onSelect }: StanceChartProps) {
  const wrapperRef = useRef<HTMLDivElement>(null);
  const width = useWidth(wrapperRef);
  const narrow = width > 0 && width < 560;

  const domain = useMemo(() => timeDomain(rows), [rows]);
  const series = useMemo(
    () => (domain ? fedFundsSeries(fedFunds, domain, rows.map((r) => r.date)) : []),
    [fedFunds, rows, domain],
  );
  const infos = plotted.map(classifierInfo);
  const sentenceBased = infos.some((info) => info.sentenceBased);
  const selectedRow = rows.find((r) => r.date === selected) ?? null;

  // Direct labels at the line ends, when there is room for them and they do not collide.
  const lastIndex = Object.fromEntries(
    plotted.map((id) => [id, rows.findLastIndex((r) => r[id] !== null)]),
  ) as Record<StanceClassifier, number>;
  const lastValues = plotted
    .map((id) => (lastIndex[id] >= 0 ? rows[lastIndex[id]][id] : null))
    .filter((v): v is number => v !== null);
  const showEndLabels =
    !narrow && width > 0 && endLabelsFit(lastValues, SCORE_HEIGHT - MARGIN_TOP - 8);
  // Panels without a visible x axis keep 8px below the plot, so their lowest tick label fits.
  const margin = { top: MARGIN_TOP, right: showEndLabels ? END_LABEL_SPACE : 12, bottom: 8, left: 0 };
  const axisMargin = { ...margin, bottom: 0 };

  if (!domain) return <div ref={wrapperRef} />;
  const { ticks, label: tickLabel } = timeTicks(domain, narrow);
  const xAxis = (visible: boolean) => (
    <XAxis
      dataKey="t"
      type="number"
      scale="time"
      domain={domain}
      ticks={ticks}
      tickFormatter={tickLabel}
      hide={!visible}
      tick={TICK}
      tickLine={false}
      axisLine={{ stroke: "var(--chart-axis)" }}
      height={24}
      allowDataOverflow
    />
  );
  const select = (state: MouseHandlerDataParam) => {
    const t = Number(state.activeLabel);
    const date = Number.isFinite(t) ? nearestMeeting(rows, t) : null;
    if (date) onSelect(date);
  };
  const selectedLine = selectedRow && (
    <ReferenceLine x={selectedRow.t} stroke="var(--chart-ink)" strokeOpacity={0.45} />
  );
  const cursor = { stroke: "var(--chart-tick)", strokeWidth: 1 };

  return (
    <div ref={wrapperRef} className="select-none [&_.recharts-surface]:outline-none">
      <Panel
        title="Stance score"
        note={
          <>
            +1 hawkish, −1 dovish
            {sentenceBased && <> · shaded band: neutral for the dashed methods (±{SENTENCE_THRESHOLD})</>}
          </>
        }
      >
        <ResponsiveContainer width="100%" height={SCORE_HEIGHT}>
          <LineChart
            data={rows}
            margin={margin}
            syncId={SYNC_ID}
            syncMethod={syncByTime}
            onClick={select}
            accessibilityLayer={false}
            className="cursor-pointer"
          >
            <CartesianGrid vertical={false} stroke="var(--chart-grid)" />
            {xAxis(false)}
            <YAxis
              domain={[-1, 1]}
              ticks={SCORE_TICKS}
              tickFormatter={scoreTick}
              width={Y_AXIS_WIDTH}
              interval={0}
              tick={TICK}
              tickLine={false}
              axisLine={false}
            />
            {sentenceBased && (
              <ReferenceArea
                y1={-SENTENCE_THRESHOLD}
                y2={SENTENCE_THRESHOLD}
                fill="var(--chart-grid)"
                fillOpacity={0.7}
                stroke="none"
              />
            )}
            <ReferenceLine y={0} stroke="var(--chart-axis)" />
            {selectedLine}
            <Tooltip
              content={({ active, payload }) => (
                <ScoreTooltip active={active} payload={payload} infos={infos} fedFunds={fedFunds} />
              )}
              cursor={cursor}
              isAnimationActive={false}
            />
            {infos.map((info) => (
              <Line
                key={info.id}
                dataKey={info.id}
                name={info.label}
                type="linear"
                stroke={info.color}
                strokeWidth={2}
                strokeDasharray={seriesDash(info)}
                strokeLinejoin="round"
                strokeLinecap={seriesDash(info) ? "butt" : "round"}
                dot={false}
                activeDot={{ r: 4, fill: info.color, stroke: "var(--chart-surface)", strokeWidth: 2 }}
                connectNulls
                isAnimationActive={false}
                label={
                  showEndLabels
                    ? (props: { x?: number | string; y?: number | string; index?: number }) =>
                        props.index === lastIndex[info.id] ? (
                          <text
                            key={`end-${info.id}`}
                            x={Number(props.x) + 8}
                            y={Number(props.y)}
                            dy={4}
                            fontSize={12}
                            fill="var(--chart-ink)"
                          >
                            {info.label}
                          </text>
                        ) : (
                          <g key={`end-${info.id}-${props.index}`} />
                        )
                    : undefined
                }
              />
            ))}
            {selectedRow &&
              infos.map((info) =>
                selectedRow[info.id] === null ? null : (
                  <ReferenceDot
                    key={info.id}
                    x={selectedRow.t}
                    y={selectedRow[info.id] as number}
                    r={4}
                    fill={info.color}
                    stroke="var(--chart-surface)"
                    strokeWidth={2}
                  />
                ),
              )}
          </LineChart>
        </ResponsiveContainer>
      </Panel>

      <Panel
        title="Reference label"
        note={
          <>
            <span className="inline-flex items-center gap-1">
              <MarkerKey filled narrow={narrow} /> checked by a person
            </span>
            <span className="inline-flex items-center gap-1">
              <MarkerKey narrow={narrow} /> labelled by LLMs
            </span>
          </>
        }
      >
        <ResponsiveContainer width="100%" height={STRIP_HEIGHT}>
          <LineChart
            data={rows}
            margin={margin}
            syncId={SYNC_ID}
            syncMethod={syncByTime}
            onClick={select}
            accessibilityLayer={false}
            className="cursor-pointer"
          >
            <CartesianGrid vertical={false} stroke="var(--chart-grid)" />
            {xAxis(false)}
            <YAxis
              domain={[-1.5, 1.5]}
              ticks={[-1, 0, 1]}
              tickFormatter={(v: number) => LANE_LABELS[v] ?? ""}
              width={Y_AXIS_WIDTH}
              interval={0}
              tick={{ ...TICK, fontSize: 11 }}
              tickLine={false}
              axisLine={false}
            />
            {selectedLine}
            <Tooltip content={() => null} cursor={cursor} isAnimationActive={false} />
            <Line
              dataKey="referenceLane"
              stroke="none"
              dot={(props: { cx?: number; cy?: number; index?: number }) => (
                <ReferenceMarker key={props.index} {...props} rows={rows} narrow={narrow} />
              )}
              activeDot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </Panel>

      <Panel
        title="Policy rate"
        note="federal funds target range, %"
        source="Source: Federal Reserve Board (DFEDTARL, DFEDTARU) via FRED, Federal Reserve Bank of St. Louis."
      >
        <ResponsiveContainer width="100%" height={RATE_HEIGHT}>
          <ComposedChart
            data={series}
            margin={axisMargin}
            syncId={SYNC_ID}
            syncMethod={syncByTime}
            onClick={select}
            accessibilityLayer={false}
            className="cursor-pointer"
          >
            <CartesianGrid vertical={false} stroke="var(--chart-grid)" />
            {xAxis(true)}
            <YAxis
              domain={[0, fedFundsCeiling(series)]}
              tickFormatter={rateTick}
              width={Y_AXIS_WIDTH}
              interval={0}
              tick={TICK}
              tickLine={false}
              axisLine={false}
              allowDecimals={false}
              tickCount={4}
            />
            {selectedLine}
            <Tooltip content={() => null} cursor={cursor} isAnimationActive={false} />
            <Area
              dataKey="range"
              type="stepAfter"
              stroke="none"
              fill="var(--chart-ink)"
              fillOpacity={0.1}
              activeDot={false}
              isAnimationActive={false}
            />
            <Line
              dataKey="upper"
              type="stepAfter"
              stroke="var(--chart-ink)"
              strokeWidth={1.5}
              dot={false}
              activeDot={false}
              isAnimationActive={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </Panel>
    </div>
  );
}

function Panel({
  title,
  note,
  source,
  children,
}: {
  title: string;
  note: ReactNode;
  source?: string;
  children: ReactNode;
}) {
  return (
    <div className="mt-6 first:mt-0">
      <p className="flex flex-wrap items-baseline gap-x-3 gap-y-0.5 text-label text-faint">
        <span className="font-medium text-muted">{title}</span>
        <span className="inline-flex flex-wrap items-center gap-x-3">{note}</span>
      </p>
      {/* The chart repeats the data table below for sighted users; screen readers use the table.
          Recharts' layers are tabIndex=-1, so a click could move focus into this aria-hidden
          subtree: keep click focus out of it (the chart's click handlers still fire). Not
          `inert`, which would block the clicks too. */}
      <div aria-hidden="true" className="mt-1" onMouseDown={(event) => event.preventDefault()}>
        {children}
      </div>
      {source && <p className="mt-2 text-label text-faint">{source}</p>}
    </div>
  );
}

function MarkerKey({ filled = false, narrow = false }: { filled?: boolean; narrow?: boolean }) {
  if (narrow) {
    return (
      <svg viewBox="0 0 4 12" className="h-3 w-1" aria-hidden="true">
        <rect
          x={1.25}
          y={filled ? 0 : 3}
          width={1.5}
          height={filled ? 12 : 6}
          rx={0.75}
          fill="var(--chart-ink)"
        />
      </svg>
    );
  }
  return (
    <svg viewBox="0 0 10 10" className="size-2.5" aria-hidden="true">
      <circle
        cx="5"
        cy="5"
        r={filled ? 4 : 3.4}
        fill={filled ? "var(--chart-ink)" : "var(--chart-surface)"}
        stroke="var(--chart-ink)"
        strokeWidth={filled ? 0 : 1.4}
      />
    </svg>
  );
}

function ReferenceMarker({
  cx,
  cy,
  index,
  rows,
  narrow,
}: {
  cx?: number;
  cy?: number;
  index?: number;
  rows: MeetingRow[];
  narrow: boolean;
}) {
  const row = index === undefined ? undefined : rows[index];
  if (!row?.reference || cx === undefined || cy === undefined || !Number.isFinite(cy)) return <g />;
  const human = row.reference.source === "human";
  if (narrow) {
    // About 2 px per meeting on a phone: circles would merge, so a thin tick whose height tells
    // the source (tall: human-reviewed, short: LLM-adjudicated).
    const height = human ? 12 : 6;
    return (
      <rect
        x={cx - 0.75}
        y={cy - height / 2}
        width={1.5}
        height={height}
        rx={0.75}
        fill="var(--chart-ink)"
      />
    );
  }
  return (
    <circle
      cx={cx}
      cy={cy}
      r={human ? 4 : 3.4}
      fill={human ? "var(--chart-ink)" : "var(--chart-surface)"}
      stroke={human ? "var(--chart-surface)" : "var(--chart-ink)"}
      strokeWidth={human ? 1 : 1.4}
    />
  );
}

type ScoreTooltipProps = Pick<TooltipContentProps, "active" | "payload"> & {
  infos: ClassifierInfo[];
  fedFunds: FedFundsTarget[];
};

/** One readout per meeting: every plotted score, the reference label and the range it set. */
function ScoreTooltip({ active, payload, infos, fedFunds }: ScoreTooltipProps) {
  const row = payload?.[0]?.payload as MeetingRow | undefined;
  if (!active || !row) return null;
  const range = rangeAfterMeeting(fedFunds, row.date);
  return (
    <div className={`max-w-64 ${overlayClass}`}>
      <p className="font-semibold text-ink">{formatDate(row.date)}</p>
      <ul className="mt-1.5 space-y-1">
        {infos.map((info) => {
          const result = row.results[info.id];
          return (
            <li key={info.id} className="flex items-center gap-2">
              <SeriesKey info={info} className="w-3" />
              <span className="w-10 font-semibold text-ink tabular-nums">
                {result ? formatScore(result.score) : "-"}
              </span>
              <span className="truncate">
                {info.label}
                {result ? ` · ${STANCE_LABELS[result.stance]}` : ""}
              </span>
            </li>
          );
        })}
      </ul>
      <dl className="mt-2 space-y-0.5 border-t border-line pt-1.5">
        <div className="flex gap-1">
          <dt>Reference:</dt>
          <dd className="text-ink">
            {row.reference
              ? `${STANCE_LABELS[row.reference.label]} (${REFERENCE_SOURCE_LABELS[row.reference.source]})`
              : "none"}
          </dd>
        </div>
        {range && (
          <div className="flex gap-1">
            <dt>Target range:</dt>
            <dd className="text-ink tabular-nums">{formatRange(range.lower, range.upper)}</dd>
          </div>
        )}
      </dl>
      <p className="mt-1.5 text-faint">Click for details</p>
    </div>
  );
}
