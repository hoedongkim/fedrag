"use client";

import { useState } from "react";

import type { EvaluationReport, ReportSplit, RunKey } from "@/lib/api/types";
import { METRICS } from "@/lib/performance/format";
import {
  differencesFor,
  RETRIEVAL_COLUMNS,
  type RetrievalMetric,
  rowsForSplit,
} from "@/lib/performance/report";

import { Section } from "../ui/Section";
import { noteClass } from "../ui/styles";
import { IntervalChart } from "./IntervalChart";
import { DifferenceList, MetricTable } from "./MetricTable";
import { MetricSelect } from "./parts";

/** Chunk- and document-level retrieval metrics of every run, as a chart and a table. */
export function RetrievalSection({
  report,
  split,
  highlight,
}: {
  report: EvaluationReport;
  split: ReportSplit;
  highlight: RunKey | null;
}) {
  const [metric, setMetric] = useState<RetrievalMetric>("recall_at_5");
  const rows = rowsForSplit(report.retrieval, split, report.configs);

  return (
    <Section
      id="perf-retrieval"
      title="Search"
      lead="Does the search return the passages that answer the question?"
      action={rows.length > 0 && <MetricSelect value={metric} options={RETRIEVAL_COLUMNS} onChange={setMetric} label="Chart" />}
    >
      {rows.length === 0 ? (
        <p className="mt-6 text-small text-faint">No search results on this split.</p>
      ) : (
        <>
          <IntervalChart
            points={rows.map((r) => ({ method: r.method, config: r.config, estimate: r[metric] }))}
            configs={report.configs}
            metricLabel={METRICS[metric].label}
            metricDescription={METRICS[metric].description}
            highlight={highlight}
          />
          <MetricTable
            label="Search metrics"
            caption="Search metrics of each method under each config: mean with its 95% bootstrap interval over questions."
            rows={rows}
            metrics={RETRIEVAL_COLUMNS}
            configs={report.configs}
            highlight={highlight}
          />
          <p className={noteClass}>Bold: best per column. Small type: 95% interval.</p>
          <DifferenceList
            differences={differencesFor(report.differences, split, [metric])}
            metricLabel={METRICS[metric].label}
            configs={report.configs}
            level={report.bootstrap?.level ?? null}
          />
        </>
      )}
    </Section>
  );
}
