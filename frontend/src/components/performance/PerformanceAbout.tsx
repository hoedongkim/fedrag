import type { EvaluationReport, JudgeValidation, ReportSplit, RunKey } from "@/lib/api/types";
import { formatMetric, formatTimestamp, runLabel } from "@/lib/performance/format";
import { runChoices } from "@/lib/performance/questions";
import {
  type Headline,
  leadAnnotators,
  rowsForSplit,
  sameRun,
  uncheckedRuns,
  valueRange,
} from "@/lib/performance/report";

import { About, AboutList } from "../ui/About";
import { JudgeCheck } from "./JudgeCheck";
import { AboutHeading, ConfigList } from "./parts";

/**
 * The judge check in a sentence: who labelled what, the range of agreement with the annotators it
 * leads with (the isolated ones), and which runs the sample does not cover.
 */
function judgeSummary(validation: JudgeValidation): string {
  const lead = leadAnnotators(validation);
  const rows = validation.agreement.filter((a) => a.annotator === null || lead.includes(a.annotator));
  const agreement = valueRange(rows.map((a) => a.agreement));
  const kappa = valueRange(rows.map((a) => a.kappa));
  const range = ([low, high]: [number, number], format: (v: number) => string) =>
    low === high ? format(low) : `${format(low)}–${format(high)}`;
  const percent = (v: number) => `${Math.round(v * 100)}%`;
  const decimal = (v: number) => formatMetric(v, 2);
  const whose = lead.length === 1 && validation.annotators.length > 1 ? `${lead[0]}’s` : "the annotators’";
  return (
    `On a blind sample of ${validation.n_annotated} answers, its pass/fail verdicts matched ${whose} ` +
    `${agreement ? range(agreement, percent) : "n/a"} of the time${kappa ? ` (Cohen’s κ ${range(kappa, decimal)})` : ""}, ` +
    "depending on the kind of check."
  );
}

/** RAG Performance's method and caveats: the one place for them, closed at the end of the page. */
export function PerformanceAbout({
  report,
  split,
  headline,
}: {
  report: EvaluationReport;
  split: ReportSplit;
  headline: Headline | null;
}) {
  const { dataset, judge, generator, cost } = report;
  const test = dataset.splits.find((s) => s.split === "test");
  const dev = dataset.splits.find((s) => s.split === "dev");
  const validation = judge?.validation;
  const leaks = rowsForSplit(report.retrieval, split, report.configs).reduce((sum, r) => sum + r.as_of_leaks, 0);
  // Runs with an answer row on some split were judged; runs with answers but no row were not.
  const runs = runChoices(report.questions, report.configs);
  const judgedRuns = runs.filter((run) => report.answers.some((r) => sameRun(r, run)));
  const unjudgedRuns = runs.filter(
    (run) =>
      !judgedRuns.some((r) => sameRun(r, run)) &&
      report.questions.some((q) => q.runs.some((r) => sameRun(r, run) && r.answered)),
  );
  const names = (list: RunKey[], separator = ", ") => list.map((run) => runLabel(report.configs, run)).join(separator);
  const unchecked = validation?.status === "complete" ? uncheckedRuns(report, validation) : [];

  return (
    <About>
      {headline && (
        <>
          <AboutHeading>How the headline setup was chosen</AboutHeading>
          <p>{headline.rule}</p>
        </>
      )}

      <AboutHeading>Settings compared</AboutHeading>
      <ConfigList configs={report.configs} />

      <AboutHeading>Reading the numbers</AboutHeading>
      <AboutList
        items={[
          {
            term: "The questions",
            body: `${dataset.n_questions} questions: ${dataset.n_answerable} answerable, ${dataset.n_questions - dataset.n_answerable} unanswerable, ${dataset.n_with_as_of} with an as-of date, ${dataset.n_human_reviewed} checked by a person. ${dataset.construction ?? "They and their gold passages were drafted with LLMs."} Questions drafted from a passage tend to reuse its words, which favours keyword search; the paraphrase questions counter that. Gold passages can be incomplete: finding another passage that says the same thing counts as a miss, which document recall partly corrects.`,
          },
          {
            term: "Search",
            body: `Each question has gold passages: primary ones answer it, supporting ones help. Recall counts how many a method returns in its top 5 or 10 (the answer sees the top 5); MRR and nDCG reward ranking them first; document recall asks only for a passage from the right document. Each question’s as-of date is applied before ranking, as in Ask the Fed: ${leaks === 0 ? "no method returned a passage published after it." : `${leaks} returned passages were published after it (a point-in-time leak).`}`,
          },
          {
            term: "By question type",
            body: "Each cell is the mean over one type’s questions. Groups are small (a cell over 8 questions moves by 0.12 per question), so read patterns, not single cells. A dash means the type has no such score: an unanswerable question has no gold passage, an answerable one no refusal score.",
          },
          {
            term: "Answers",
            body: `The top ${generator?.top_k ?? 5} passages of ${judgedRuns.length ? names(judgedRuns, " and ") : "each run"} go through the same answer chain as Ask the Fed${generator ? ` (${generator.model}, reasoning effort ${generator.reasoning_effort ?? "default"})` : ""}.${unjudgedRuns.length ? ` ${names(unjudgedRuns)} answers were generated but not judged.` : ""} ${judge ? `${judge.model} then` : "An LLM judge"} splits each answer into claims and checks each against the passages it cites, checks every citation, and checks the reference answer’s required facts. Faithfulness and citation accuracy cover answers that make claims or cite; correctness, the answerable questions; refusal accuracy, only the unanswerable ones, so its n is small. A false refusal is an answerable question declined.`,
          },
          {
            term: "An LLM grades the answers",
            body: `The answer scores are ${judge ? `${judge.model}’s` : "an LLM judge’s"} verdicts, not a person’s. A larger model than the one that wrote the answers limits self-preference but does not remove it, and both come from one vendor, so they can share blind spots. ${validation?.status === "complete" ? judgeSummary(validation) : "Its agreement with an annotator has not been measured yet."}${unchecked.length ? ` No ${names(unchecked, " or ")} answer was in the sample.` : ""}`,
          },
          {
            term: "Dev and test",
            body:
              report.selection?.split === "test"
                ? `The tuned settings were picked on the dev split${dev ? ` (${dev.n} questions)` : ""} and frozen. A rule set before the test run (recorded by the project itself, with no timestamped copy from before it) then decided on the test split${test ? ` (${test.n} questions)` : ""} whether Ask the Fed switches to them. Quote test; dev numbers are optimistic by construction.`
                : `The settings and the headline setup were picked on the dev split${dev ? ` (${dev.n} questions)` : ""}; the test split${test ? ` (${test.n} questions)` : ""} was scored once the choice was fixed. Quote test; dev numbers are optimistic by construction.`,
          },
          {
            term: "Small samples",
            body: `One question moves a test mean by about ${test?.n_answerable ? (1 / test.n_answerable).toFixed(2) : "0.02"}. The 95% intervals resample questions as if they were independent; questions about the same meeting are not, so the real uncertainty is larger. A gap between two rows means little unless its paired difference is clear of 0.`,
          },
          {
            term: "One run of the answers",
            body: "GPT-5.x models at low reasoning effort take no temperature, so a regenerated answer can differ. Answers and verdicts are cached, which makes these numbers reproducible but not stable across regeneration.",
          },
          {
            term: "Speed and cost",
            body: `Timed one question at a time on the machine that ran the evaluation: indicative, not a load test. Cost is estimated from the API’s token counts at the prices in the code, not from an invoice.${cost ? " Each command estimates its cost first and refuses to start above its cap; cached answers and verdicts are counted at what they cost when first made, and query embeddings are counted in texts (they go out in batches)." : ""}`,
          },
        ]}
      />

      {judge && (
        <>
          <AboutHeading>Checking the judge</AboutHeading>
          <JudgeCheck judge={judge} report={report} />
        </>
      )}

      {report.notes.length > 0 && (
        <>
          <AboutHeading>Notes from the report</AboutHeading>
          <ul className="list-disc space-y-1 pl-5 text-label marker:text-faint">
            {report.notes.map((note) => (
              <li key={note}>{note}</li>
            ))}
          </ul>
        </>
      )}

      <p>Newest result behind these numbers: {formatTimestamp(report.generated_at)}.</p>
    </About>
  );
}
