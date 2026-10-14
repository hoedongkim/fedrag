import type { EvaluationReport, ReportSplit, RunKey } from "@/lib/api/types";
import { formatMs, formatSpend, formatTokens, formatUsd } from "@/lib/performance/format";
import { bestRun, groupByConfig, rowFor, rowsForSplit, runKey, sameRun } from "@/lib/performance/report";

import { GroupRow, stickyCell, TableScroller, td, th } from "../DataTable";
import { Section, SubHeading } from "../ui/Section";
import { noteClass } from "../ui/styles";
import { RunNameCell } from "./parts";

const STAGE_LABELS = { generation: "Answers", judge: "Judge", embedding: "Query embeddings" } as const;

/** Latency per question and OpenAI cost per question for every run, and what the run cost. */
export function CostSection({
  report,
  split,
  highlight,
}: {
  report: EvaluationReport;
  split: ReportSplit;
  highlight: RunKey | null;
}) {
  const rows = rowsForSplit(report.retrieval, split, report.configs).map((retrieval) => ({
    method: retrieval.method,
    config: retrieval.config,
    retrieval,
    answers: rowFor(report.answers, split, retrieval),
  }));
  const fastest = bestRun(rows, (r) => r.retrieval.latency_ms_p50, true);
  const cheapest = bestRun(rows, (r) => r.answers?.cost_per_query_usd ?? null, true);
  const { cost } = report;
  const bold = "font-semibold text-ink";

  return (
    <Section id="perf-cost" title="Speed and cost" lead="How long each question takes, and what its OpenAI calls cost.">

      <TableScroller label="Latency and cost per question">
        <table className="w-full min-w-[44rem] border-collapse text-small">
          <caption className="sr-only">
            Median and 95th-percentile latency of retrieval and of the whole answer, and OpenAI cost
            per question, for each method under each config.
          </caption>
          <thead className="border-b border-line">
            <tr>
              <th scope="col" className={`${th} ${stickyCell}`}>
                Method
              </th>
              <th scope="col" className={th} title="The search alone, median">Search, median</th>
              <th scope="col" className={th} title="The search alone, slowest 5% start here">Search, p95</th>
              <th scope="col" className={th} title="Search plus writing the answer, median">With answer, median</th>
              <th scope="col" className={th} title="Search plus writing the answer, slowest 5% start here">
                With answer, p95
              </th>
              <th scope="col" className={th} title="OpenAI cost of writing one answer">Answer cost</th>
              <th scope="col" className={th} title="OpenAI cost of judging one answer (evaluation only)">Judge cost</th>
            </tr>
          </thead>
          {groupByConfig(rows, report.configs).map(({ config, rows: group }) => (
            <tbody
              key={config.id}
              className="divide-y divide-line text-muted [&:not(:first-of-type)]:border-t [&:not(:first-of-type)]:border-line-strong"
            >
              <GroupRow colSpan={7}>{config.label}</GroupRow>
              {group.map(({ method, config: id, retrieval, answers }) => {
                const key = runKey({ method, config: id });
                return (
                  <tr key={key}>
                    <RunNameCell
                      method={method}
                      config={id}
                      configs={report.configs}
                      highlight={!!highlight && sameRun(highlight, { method, config: id })}
                    />
                    <td className={`${td} ${fastest === key ? bold : ""}`}>{formatMs(retrieval.latency_ms_p50)}</td>
                    <td className={td}>{formatMs(retrieval.latency_ms_p95)}</td>
                    <td className={td}>{formatMs(answers?.latency_ms_p50)}</td>
                    <td className={td}>{formatMs(answers?.latency_ms_p95)}</td>
                    <td className={`${td} ${cheapest === key ? bold : ""}`}>{formatUsd(answers?.cost_per_query_usd)}</td>
                    <td className={td}>{formatUsd(answers?.judge_cost_per_query_usd)}</td>
                  </tr>
                );
              })}
            </tbody>
          ))}
        </table>
      </TableScroller>
      <p className={noteClass}>Bold: fastest search and cheapest answer. p95: the slowest 5% of questions take longer.</p>

      {cost && (
        <div className="mt-12">
          <SubHeading>
            What this evaluation cost: <span className="tabular-nums">{formatSpend(cost)}</span>
          </SubHeading>
          {cost.lines.length > 0 && (
            <TableScroller label="OpenAI usage of the evaluation">
              <table className="w-full min-w-[36rem] border-collapse text-small">
                <caption className="sr-only">OpenAI calls, tokens and cost of each stage of the evaluation.</caption>
                <thead className="border-b border-line">
                  <tr>
                    <th scope="col" className={`${th} ${stickyCell}`}>Stage</th>
                    <th scope="col" className={th}>Model</th>
                    <th scope="col" className={th}>Calls</th>
                    <th scope="col" className={th}>Input tokens</th>
                    <th scope="col" className={th}>Output tokens</th>
                    <th scope="col" className={th}>Cost</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-line text-muted">
                  {cost.lines.map((line) => (
                    <tr key={`${line.stage}-${line.model}`}>
                      <th scope="row" className={`${stickyCell} px-3 py-2 text-left align-top font-medium whitespace-nowrap text-ink`}>
                        {STAGE_LABELS[line.stage]}
                      </th>
                      <td className={`${td} font-mono text-label`}>{line.model}</td>
                      <td className={td}>
                        {line.calls.toLocaleString("en-US")}
                        {line.stage === "embedding" && " texts"}
                      </td>
                      <td className={td}>
                        {formatTokens(line.input_tokens)}
                        {line.cached_input_tokens > 0 && (
                          <span className="block text-label text-faint">
                            {formatTokens(line.cached_input_tokens)} cached
                          </span>
                        )}
                      </td>
                      <td className={td}>{formatTokens(line.output_tokens)}</td>
                      <td className={td}>{formatUsd(line.usd)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </TableScroller>
          )}
        </div>
      )}
    </Section>
  );
}
