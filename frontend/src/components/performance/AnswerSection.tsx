"use client";

import { useState } from "react";

import type { EvaluationReport, ReportSplit, RunKey } from "@/lib/api/types";
import { METRICS, SPLIT_LABELS } from "@/lib/performance/format";
import {
  ANSWER_COLUMNS,
  type AnswerMetric,
  differencesFor,
  rowsForSplit,
  SPLIT_ORDER,
} from "@/lib/performance/report";

import { Section } from "../ui/Section";
import { noteClass } from "../ui/styles";
import { IntervalChart } from "./IntervalChart";
import { DifferenceList, MetricTable } from "./MetricTable";
import { MetricSelect } from "./parts";

/** LLM-judged answer quality of every run, as a chart and a table. The judge's check: PerformanceAbout. */
export function AnswerSection({
  report,
  split,
  highlight,
}: {
  report: EvaluationReport;
  split: ReportSplit;
  highlight: RunKey | null;
}) {
  const [metric, setMetric] = useState<AnswerMetric>("faithfulness");
  const rows = rowsForSplit(report.answers, split, report.configs);
  const { judge } = report;
  const judgedSplits = SPLIT_ORDER.filter((s) => report.answers.some((r) => r.split === s));

  return (
    <Section
      id="perf-answers"
      title="Answers"
      lead={
        <>
          Are the answers backed by their sources, correctly cited and complete?{" "}
          {judge ? (
            <>
              A larger model, <span className="whitespace-nowrap">{judge.model}</span>, grades each one.
            </>
          ) : (
            "An LLM judge grades each one."
          )}
        </>
      }
      action={rows.length > 0 && <MetricSelect value={metric} options={ANSWER_COLUMNS} onChange={setMetric} label="Chart" />}
    >
      {rows.length === 0 ? (
        <p className="mt-6 text-small text-faint">
          {judgedSplits.length
            ? `Answers were judged on the ${judgedSplits.map((s) => SPLIT_LABELS[s].toLowerCase()).join(" and ")} split only: switch to it above.`
            : "No judged answers on this split yet."}
        </p>
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
            label="Answer quality"
            caption="LLM-judged answer quality of each method under each config: mean with its 95% bootstrap interval over questions, and the number of questions behind it."
            rows={rows}
            metrics={ANSWER_COLUMNS}
            configs={report.configs}
            highlight={highlight}
            showN
          />
          <p className={noteClass}>
            Bold: best per column (lowest for false refusals). Small type: 95% interval and number of questions.
          </p>
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
