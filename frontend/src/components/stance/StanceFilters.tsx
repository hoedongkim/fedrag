"use client";

import { useId } from "react";

import type { StanceClassifier } from "@/lib/api/types";
import { CLASSIFIERS } from "@/lib/stance/classifiers";
import { isNotLoaded } from "@/lib/stance/failures";

import { Spinner } from "../ask/icons";
import { labelTextClass, smallFieldClass } from "../ui/styles";
import { SeriesKey } from "./SeriesKey";
import type { TimelineLoads } from "./useStanceData";

interface StanceFiltersProps {
  loads: TimelineLoads;
  plotted: StanceClassifier[];
  onToggle: (id: StanceClassifier) => void;
  years: number[];
  fromYear: number;
  toYear: number;
  onYears: (fromYear: number, toYear: number) => void;
}

/**
 * The row above the chart. The method toggles are also the chart's legend: each carries its line
 * key next to its name, faded when the line is hidden. The years scope the chart, the table and
 * the meeting list alike.
 */
export function StanceFilters({
  loads,
  plotted,
  onToggle,
  years,
  fromYear,
  toYear,
  onYears,
}: StanceFiltersProps) {
  const baseId = useId();
  const unloaded = CLASSIFIERS.filter((c) => {
    const load = loads[c.id];
    return load.status === "error" && isNotLoaded(load.error);
  }).map((c) => c.label);
  return (
    <div className="flex flex-wrap items-end justify-between gap-x-8 gap-y-5">
      <fieldset className="min-w-0">
        <legend className={labelTextClass}>Show methods</legend>
        <ul className="mt-1.5 -ml-2 flex flex-wrap gap-x-1 gap-y-1">
          {CLASSIFIERS.map((info) => {
            const load = loads[info.id];
            const on = plotted.includes(info.id);
            const unavailable = load.status === "error";
            return (
              <li key={info.id}>
                <button
                  type="button"
                  aria-pressed={on && !unavailable}
                  aria-disabled={unavailable || undefined}
                  title={info.description}
                  onClick={() => {
                    if (!unavailable) onToggle(info.id);
                  }}
                  className={[
                    "inline-flex h-8 items-center gap-2 rounded-md px-2 text-small font-medium transition-colors",
                    unavailable
                      ? "cursor-not-allowed text-faint"
                      : on
                        ? "text-ink hover:bg-subtle"
                        : "text-faint hover:bg-subtle hover:text-muted",
                  ].join(" ")}
                >
                  <SeriesKey info={info} className={on && !unavailable ? "" : "opacity-25 grayscale"} />
                  <span>{info.label}</span>
                  {load.status === "loading" && <Spinner className="size-3.5 text-faint" />}
                  {unavailable && (
                    <span className="text-label font-normal">
                      {isNotLoaded(load.error) ? "not loaded" : "unavailable"}
                    </span>
                  )}
                </button>
              </li>
            );
          })}
        </ul>
        {unloaded.length > 0 && unloaded.length < CLASSIFIERS.length && (
          <p className="mt-1 text-label text-faint">
            No results yet for {unloaded.join(" and ")}: run <code className="text-muted">make stance</code>.
          </p>
        )}
      </fieldset>

      {years.length > 1 && (
        <fieldset className="flex items-center gap-2 text-small text-faint">
          <legend className="sr-only">Years</legend>
          <label htmlFor={`${baseId}-from`}>From</label>
          <YearSelect
            id={`${baseId}-from`}
            value={fromYear}
            years={years.filter((y) => y <= toYear)}
            onChange={(year) => onYears(year, toYear)}
          />
          <label htmlFor={`${baseId}-to`}>to</label>
          <YearSelect
            id={`${baseId}-to`}
            value={toYear}
            years={years.filter((y) => y >= fromYear)}
            onChange={(year) => onYears(fromYear, year)}
          />
        </fieldset>
      )}
    </div>
  );
}

function YearSelect(props: { id: string; value: number; years: number[]; onChange: (year: number) => void }) {
  return (
    <select
      id={props.id}
      value={props.value}
      onChange={(event) => props.onChange(Number(event.target.value))}
      className={`w-[5.5rem] tabular-nums ${smallFieldClass}`}
    >
      {props.years.map((year) => (
        <option key={year} value={year}>
          {year}
        </option>
      ))}
    </select>
  );
}
