/** Small pieces shared by the RAG Performance sections. */
import { type ReactNode, useId } from "react";

import type { EvalMetric, RetrievalMethod, RetrieverConfig } from "@/lib/api/types";
import { configStyle, METRICS, methodLabel } from "@/lib/performance/format";

import { rowCell } from "../DataTable";
import { labelTextClass, smallFieldClass } from "../ui/styles";
import { Mark } from "./IntervalChart";

/** A table's row header: the method's name, with the config's mark as in the chart. */
export function RunNameCell({
  method,
  config,
  configs,
  highlight = false,
}: {
  method: RetrievalMethod;
  config: string;
  configs: { id: string }[];
  highlight?: boolean;
}) {
  return (
    <th scope="row" className={rowCell}>
      <span className="flex items-center gap-2">
        <Mark style={configStyle(configs, config)} size={9} />
        {methodLabel(method)}
        {highlight && <ChosenTag />}
      </span>
    </th>
  );
}

/** Marks the headline run (the one Ask the Fed is set up with) in a table row. */
export function ChosenTag() {
  return <span className="text-label font-normal text-accent">chosen</span>;
}

/** A labelled select of metrics, for what a chart plots. */
export function MetricSelect<M extends EvalMetric>({
  value,
  options,
  onChange,
  label = "Metric",
}: {
  value: M;
  options: readonly M[];
  onChange: (metric: M) => void;
  label?: string;
}) {
  const id = useId();
  return (
    <div className="flex items-center gap-2">
      <label htmlFor={id} className="text-small text-faint">
        {label}
      </label>
      <select
        id={id}
        value={value}
        onChange={(event) => onChange(event.target.value as M)}
        className={`${smallFieldClass} w-auto pr-8`}
      >
        {options.map((metric) => (
          <option key={metric} value={metric}>
            {METRICS[metric].label}
          </option>
        ))}
      </select>
    </div>
  );
}

/** Short facts about a question as one quiet line: "Date-specific · as of Jul 5, 2017". */
export function MetaLine({ items, className = "" }: { items: (ReactNode | false | null | undefined)[]; className?: string }) {
  const shown = items.filter((item): item is ReactNode => item !== false && item !== null && item !== undefined);
  return (
    <span className={`text-label text-faint ${className}`}>
      {shown.map((item, i) => (
        <span key={i}>
          {i > 0 && <span aria-hidden="true"> · </span>}
          {item}
        </span>
      ))}
    </span>
  );
}

/** A config setting as shown: an unset reranker is the one the settings name, not none. */
function paramValue(name: string, value: RetrieverConfig["params"][string]): string {
  return value === null && name.startsWith("reranker_") ? "settings default" : String(value);
}

/** The retriever configs compared: mark, name, what each changes and its settings. */
export function ConfigList({ configs }: { configs: RetrieverConfig[] }) {
  return (
    <dl className="space-y-3">
      {configs.map((config) => {
        const params = Object.entries(config.params);
        return (
          <div key={config.id}>
            <dt className="flex items-center gap-2 font-medium text-ink">
              <Mark style={configStyle(configs, config.id)} size={10} />
              {config.label}
              {config.id !== config.label && <code className="font-normal text-faint">{config.id}</code>}
            </dt>
            <dd className="mt-0.5">{config.description}</dd>
            {params.length > 0 && (
              <dd className="mt-1 font-mono text-label break-words text-faint">
                {params.map(([name, value]) => `${name}: ${paramValue(name, value)}`).join(" · ")}
              </dd>
            )}
          </div>
        );
      })}
    </dl>
  );
}

/** A heading inside the About disclosure. */
export function AboutHeading({ children }: { children: ReactNode }) {
  return <h3 className={`pt-3 first:pt-1 ${labelTextClass}`}>{children}</h3>;
}
