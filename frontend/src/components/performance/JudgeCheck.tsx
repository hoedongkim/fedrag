import type { EvaluationReport, JudgeInfo } from "@/lib/api/types";
import { formatMetric, JUDGE_CHECK_LABELS, runLabel } from "@/lib/performance/format";
import { agreementGrid, uncheckedRuns } from "@/lib/performance/report";

/**
 * Who judged the answers, with which prompt, and how often annotators agreed with it on a blind
 * sample, per kind of check. Shown inside the page's About disclosure.
 */
export function JudgeCheck({ judge, report }: { judge: JudgeInfo; report: EvaluationReport }) {
  const validation = judge.validation;
  const complete = validation?.status === "complete";
  const grid = complete && validation ? agreementGrid(validation) : null;
  const sampled = (validation?.sampled_runs ?? []).map((run) => runLabel(report.configs, run));
  const unchecked = validation ? uncheckedRuns(report, validation).map((run) => runLabel(report.configs, run)) : [];
  return (
    <div className="space-y-3">
      <p>
        Judge <code>{judge.model}</code>
        {judge.reasoning_effort ? ` (reasoning effort ${judge.reasoning_effort})` : ""}, prompt{" "}
        <code>{judge.prompt_version}</code>, {judge.n_answers.toLocaleString("en-US")} answers judged.{" "}
        {!validation || validation.status === "not_started"
          ? "Its verdicts have not been checked against an annotator yet, so the answer numbers are its opinion."
          : validation.status === "pending"
            ? `A sample of ${validation.n_sampled} answers is out for an independent annotator${validation.n_annotated ? ` (${validation.n_annotated} done)` : ""}; until then the answer numbers are the judge’s opinion.`
            : `Checked on a blind sample: ${validation.n_annotated} of ${validation.n_sampled} answers labelled${validation.annotator ? ` by ${validation.annotator.replace(/[.\s]+$/, "")}` : ""}.`}
        {complete &&
          sampled.length > 0 &&
          ` The sample is all ${sampled.join(", ")}${unchecked.length ? `; no ${unchecked.join(" or ")} answer was checked.` : "."}`}
      </p>
      {grid && grid.rows.length > 0 && (
        <div
          role="region"
          aria-label="Judge agreement with annotators, per check (scrolls sideways)"
          tabIndex={0}
          className="overflow-x-auto"
        >
          <table className="w-full min-w-[30rem] text-small">
            <caption className="sr-only">
              Share of pass/fail verdicts two raters agree on, with Cohen&rsquo;s kappa, per kind of check
            </caption>
            <thead>
              <tr className="border-b border-line text-label text-faint">
                <th scope="col" className="py-1.5 pr-3 text-left font-medium">
                  Check
                </th>
                {grid.columns.map((column) => (
                  <th key={column.key} scope="col" className="py-1.5 pl-3 text-right align-bottom font-medium">
                    {column.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-line">
              {grid.rows.map((row) => {
                const n = Object.values(row.cells).find(Boolean)?.n;
                return (
                  <tr key={row.check}>
                    <th scope="row" className="py-1.5 pr-3 text-left font-normal">
                      {JUDGE_CHECK_LABELS[row.check]}
                      {n !== undefined && <span className="ml-1 text-label whitespace-nowrap text-faint">n = {n}</span>}
                    </th>
                    {grid.columns.map((column) => {
                      const cell = row.cells[column.key];
                      return (
                        <td
                          key={column.key}
                          className="py-1.5 pl-3 text-right whitespace-nowrap"
                          title={
                            cell?.exact !== null && cell?.exact !== undefined
                              ? `Same three-way label: ${formatMetric(cell.exact, 2)}`
                              : undefined
                          }
                        >
                          {cell ? (
                            <>
                              <span className="text-ink">{formatMetric(cell.agreement, 2)}</span>
                              {cell.kappa !== null && (
                                <span className="ml-1 text-label text-faint">κ {formatMetric(cell.kappa, 2)}</span>
                              )}
                            </>
                          ) : (
                            "–"
                          )}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      {complete && validation && validation.annotators.length > 0 && (
        <ul className="space-y-0.5 text-label text-faint">
          {validation.annotators.map((annotator) => (
            <li key={annotator.id}>
              <span className="font-medium text-muted">{annotator.id}:</span> {annotator.description ?? "no description"}
            </li>
          ))}
        </ul>
      )}
      {validation?.note && <p className="text-label text-faint">{validation.note}</p>}
    </div>
  );
}
