import type { PairedDifference, StanceBenchmark, StanceClassifier } from "@/lib/api/types";
import {
  BASELINE_LABELS,
  baselineComparisons,
  pairedComparisons,
  reviewStatus,
  statementRowKey,
} from "@/lib/stance/benchmark";
import { classifierInfo, SENTENCE_THRESHOLD } from "@/lib/stance/classifiers";
import { formatDifference, formatLevel, formatTimestamp } from "@/lib/stance/format";

import { About, AboutList } from "../ui/About";

function rowName(key: { classifier: StanceClassifier; variant: string }): string {
  const label = classifierInfo(key.classifier).label;
  return key.variant === "pre_registered" ? label : `${label} (post hoc)`;
}

type Measured = Pick<PairedDifference, "n" | "accuracy">;

/** "+0.052 (−0.031 to +0.134) on all 97; +0.080 (...) on the 25 checked by a person". */
function accuracyDifferences({ all, humanSubset }: { all: Measured | null; humanSubset: Measured | null }): string {
  return [
    all && `${formatDifference(all.accuracy.difference, all.accuracy.ci)} on all ${all.n}`,
    humanSubset &&
      `${formatDifference(humanSubset.accuracy.difference, humanSubset.accuracy.ci)} on the ${humanSubset.n} checked by a person`,
  ]
    .filter(Boolean)
    .join("; ");
}

/** Policy Stance's methodology and caveats, closed at the end of the page. */
export function StanceAbout({ benchmark }: { benchmark: StanceBenchmark | null }) {
  const review = benchmark ? reviewStatus(benchmark) : null;
  const n = benchmark?.statement_level[0]?.n;
  const bootstrap = benchmark?.statement_bootstrap ?? null;
  const comparisons = pairedComparisons(bootstrap);
  const againstBaselines = baselineComparisons(bootstrap);

  return (
    <About>
      <AboutList
        items={[
          {
            term: "How each method scores",
            body: `The two LLM methods read the whole statement and give a score from −1 (dovish) to +1 (hawkish). FinBERT and Keyword score each policy sentence and average them (chance of hawkish minus chance of dovish): above +${SENTENCE_THRESHOLD} is hawkish, below −${SENTENCE_THRESHOLD} dovish, a cut-off fixed before evaluation and never tuned.`,
          },
          {
            term: "Scores are not probabilities",
            body: "Scores and confidences are what the models report, not calibrated odds: a confidence of 0.9 does not mean the call is right 9 times in 10.",
          },
          {
            term: "Reference labels",
            body: `Each statement was labelled by two LLMs following a written rubric (they agreed on every one). ${
              review?.kind === "reviewed"
                ? `A person re-labelled ${review.nHuman} of ${review.nLabels} blind, and their label is used for those.`
                : review?.kind === "pending"
                  ? "A blind check by a person is pending."
                  : ""
            } The statement scores therefore measure agreement with these labels, not ground truth. The LLM methods use the same rubric, so part of their agreement is built in; the person-checked columns, on a small set chosen to over-represent hard cases, are the stricter test.`,
          },
          {
            term: "Post hoc rows and baselines",
            body: "Post-hoc rows let the rate-decision sentence decide instead of averaging every sentence. They were designed after the results were seen, so they explain why averaging fails rather than compete. The rate-move rule reads only the decision verb (raise: hawkish, lower: dovish); the majority class always predicts the most common label.",
          },
          ...(benchmark
            ? [
                {
                  term: "Labelled sentences",
                  body: `${benchmark.dataset.citation} License: ${benchmark.dataset.license}, at commit ${benchmark.dataset.commit.slice(0, 7)}. Each method learns only from each split’s training part; "no duplicates" drops test sentences whose text also appears in training. A lead smaller than the spread over splits is not a real difference.`,
                },
              ]
            : []),
          {
            term: "Wide intervals",
            body: `With ${n ?? "about 100"} statements, the 95% intervals span several to tens of points. They treat statements as independent, but consecutive statements are near-duplicates, so the real uncertainty is larger. A gap between two rows means little unless its paired difference below excludes zero.`,
          },
        ]}
      />
      {bootstrap && comparisons.length + againstBaselines.length > 0 && (
        <div>
          <p className="font-medium text-ink">Paired differences in accuracy ({formatLevel(bootstrap.level)} intervals)</p>
          <ul className="mt-1 list-disc space-y-0.5 pl-5 tabular-nums marker:text-faint">
            {comparisons.map((pair) => (
              <li key={`${statementRowKey(pair.minuend)}-${statementRowKey(pair.subtrahend)}`}>
                {rowName(pair.minuend)} − {rowName(pair.subtrahend)}: {accuracyDifferences(pair)}
              </li>
            ))}
            {againstBaselines.map((pair) => (
              <li key={`${statementRowKey(pair.row)}-${pair.baseline}`}>
                {rowName(pair.row)} − {BASELINE_LABELS[pair.baseline].toLowerCase()}: {accuracyDifferences(pair)}
              </li>
            ))}
          </ul>
        </div>
      )}
      {benchmark && <p>Computed {formatTimestamp(benchmark.generated_at)}.</p>}
    </About>
  );
}
