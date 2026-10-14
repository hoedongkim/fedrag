import type {
  StanceBenchmark,
  StanceClassifier,
  StatementBaseline,
  StatementBaselineName,
  StatementBenchmark,
} from "@/lib/api/types";
import {
  BASELINE_LABELS,
  baselineDetail,
  best,
  isPostHoc,
  sentenceBaselineTable,
  sentenceTable,
  statementRowKey,
  statementTable,
} from "@/lib/stance/benchmark";
import { classifierInfo } from "@/lib/stance/classifiers";
import { describeStanceError } from "@/lib/stance/failures";
import { formatCount, formatMeanSd, formatMetric } from "@/lib/stance/format";

import { GroupRow, MetricCell, rowCell, stickyCell, TableScroller, td, th } from "../DataTable";
import { ExternalLink } from "../ui/ExternalLink";
import { Section, SubHeading } from "../ui/Section";
import { noteClass, skeletonClass } from "../ui/styles";
import { FailureNotice } from "./FailureNotice";
import { SeriesKey } from "./SeriesKey";
import type { Load } from "./useStanceData";

/** How the methods score on labelled sentences and on whole statements. Caveats: StanceAbout. */
export function BenchmarkSection({
  load,
  onRetry,
}: {
  load: Load<StanceBenchmark>;
  onRetry: () => void;
}) {
  return (
    <Section
      id="stance-benchmark"
      title="How the methods compare"
      lead="How often each method agrees with labels: on sentences from a public dataset, and on the statements above."
    >
      <div className="mt-8">
        {load.status === "loading" && <div className={`h-48 ${skeletonClass}`} aria-hidden="true" />}
        {load.status === "error" && <FailureNotice failure={describeStanceError(load.error)} onRetry={onRetry} />}
        {load.status === "ready" && <BenchmarkTables benchmark={load.data} />}
      </div>
    </Section>
  );
}

const TITLES = {
  macroF1: "F1 score averaged over hawkish, neutral and dovish, from 0 to 1",
  dedup: "The same, without test sentences whose text also appears in training",
  accuracy: "Share of calls that match the label",
  kappa: "Cohen’s kappa: agreement corrected for chance (0 = chance, 1 = perfect)",
  spearman: "Spearman’s rho: does the score rise from dovish to neutral to hawkish labels? No threshold involved",
  human: "Only the statements a person re-labelled blind",
};

function BenchmarkTables({ benchmark }: { benchmark: StanceBenchmark }) {
  const sentences = sentenceTable(benchmark.sentence_level);
  const statements = statementTable(benchmark.statement_level);
  const sentenceBaselines = sentenceBaselineTable(benchmark.sentence_baselines);
  const statementBaselines = benchmark.statement_baselines;
  const bestOfficial = best(sentences, (r) => r.official?.macro_f1_mean);
  const bestDeduplicated = best(sentences, (r) => r.deduplicated?.macro_f1_mean);
  const bestAccuracy = best(sentences, (r) => r.official?.accuracy_mean);
  const { dataset } = benchmark;
  const preRegistered = statements.filter((r) => !isPostHoc(r));
  const postHoc = statements.filter(isPostHoc);
  const humanN = statements.find((r) => r.human_subset)?.human_subset?.n;
  // Bold marks the best pre-registered value; post-hoc rows never compete.
  const isBest = (
    row: StatementBenchmark,
    value: (r: StatementBenchmark) => number | null | undefined,
  ) => !isPostHoc(row) && best(preRegistered, value) === row.classifier;
  // The metric cells of a statement row or a baseline (which never competes for bold).
  const statementCells = (
    row: StatementBenchmark | StatementBaseline,
    isTop: (value: (r: StatementBenchmark) => number | null | undefined) => boolean,
  ) => (
    <>
      <td className={td}>{row.n}</td>
      <MetricCell best={isTop((r) => r.accuracy)} ci={row.accuracy_ci}>
        {formatMetric(row.accuracy)}
      </MetricCell>
      <MetricCell best={isTop((r) => r.macro_f1)} ci={row.macro_f1_ci}>
        {formatMetric(row.macro_f1)}
      </MetricCell>
      <MetricCell best={isTop((r) => r.kappa)}>{formatMetric(row.kappa)}</MetricCell>
      <MetricCell best={isTop((r) => r.spearman)}>{formatMetric(row.spearman)}</MetricCell>
      <MetricCell best={isTop((r) => r.human_subset?.accuracy)} ci={row.human_subset?.accuracy_ci}>
        {formatMetric(row.human_subset?.accuracy)}
      </MetricCell>
      <MetricCell best={isTop((r) => r.human_subset?.kappa)}>
        {formatMetric(row.human_subset?.kappa)}
      </MetricCell>
    </>
  );
  const statementRow = (row: StatementBenchmark) => (
    <tr key={statementRowKey(row)}>
      <ClassifierCell id={row.classifier} postHoc={isPostHoc(row)} />
      {statementCells(row, (value) => isBest(row, value))}
    </tr>
  );

  return (
    <div className="space-y-14">
      <div>
        <SubHeading>Whole statements</SubHeading>
        <p className="mt-1 max-w-[70ch] text-small text-muted">
          Each method&rsquo;s call on the {statements[0]?.n ?? "FOMC"} statements above, against their reference labels
          {humanN ? `; the last two columns use only the ${humanN} a person re-labelled` : ""}.
        </p>
        {statements.length === 0 ? (
          <p className="mt-4 text-small text-faint">Not evaluated yet: the reference labels are still being made.</p>
        ) : (
          <>
            <TableScroller label="Statement-level benchmark">
              <table className="w-full min-w-[44rem] border-collapse text-small">
                <caption className="sr-only">
                  Statement-level agreement of each method with the reference labels, on all statements and on the
                  human-reviewed ones
                </caption>
                <thead className="border-b border-line">
                  <tr>
                    <th scope="col" className={`${th} ${stickyCell}`}>Method</th>
                    <th scope="col" className={th}>Statements</th>
                    <th scope="col" className={th} title={TITLES.accuracy}>Accuracy</th>
                    <th scope="col" className={th} title={TITLES.macroF1}>Macro-F1</th>
                    <th scope="col" className={th} title={TITLES.kappa}>Cohen&rsquo;s&nbsp;κ</th>
                    <th scope="col" className={th} title={TITLES.spearman}>Spearman&nbsp;ρ</th>
                    <th scope="col" className={th} title={TITLES.human}>Person-checked accuracy</th>
                    <th scope="col" className={th} title={TITLES.human}>Person-checked κ</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-line text-muted">{preRegistered.map(statementRow)}</tbody>
                {postHoc.length > 0 && (
                  <tbody className="divide-y divide-line border-t border-line-strong text-muted">
                    <GroupRow colSpan={8}>Post hoc: designed after seeing the rows above</GroupRow>
                    {postHoc.map(statementRow)}
                  </tbody>
                )}
                {statementBaselines.length > 0 && (
                  <tbody className="divide-y divide-line border-t border-line-strong text-muted">
                    <GroupRow colSpan={8}>Baselines (no classifier)</GroupRow>
                    {statementBaselines.map((baseline) => (
                      <tr key={baseline.name}>
                        <BaselineCell name={baseline.name} detail={baselineDetail(baseline)} postHoc={baseline.post_hoc} />
                        {statementCells(baseline, () => false)}
                      </tr>
                    ))}
                  </tbody>
                )}
              </table>
            </TableScroller>
            <p className={noteClass}>
              Bold: best pre-registered value per column. Small type: 95% interval.
            </p>
          </>
        )}
      </div>

      <div>
        <SubHeading>Labelled sentences</SubHeading>
        <p className="mt-1 max-w-[70ch] text-small text-muted">
          The {dataset.name} test sets, averaged over {benchmark.sentence_level[0]?.n_seeds ?? "three"} train/test splits.
        </p>
        {sentences.length === 0 ? (
          <p className="mt-4 text-small text-faint">No sentence results yet.</p>
        ) : (
          <TableScroller label="Sentence-level benchmark">
            <table className="w-full min-w-[40rem] border-collapse text-small">
              <caption className="sr-only">
                Sentence-level macro-F1 and accuracy of each method, mean ± standard deviation over seeds
              </caption>
              <thead className="border-b border-line">
                <tr>
                  <th scope="col" className={`${th} ${stickyCell}`}>Method</th>
                  <th scope="col" className={th} title={TITLES.macroF1}>Macro-F1</th>
                  <th scope="col" className={th} title={TITLES.dedup}>Macro-F1, no duplicates</th>
                  <th scope="col" className={th} title={TITLES.accuracy}>Accuracy</th>
                  <th scope="col" className={th} title="Per split: all / without duplicates">
                    Test sentences
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-line text-muted">
                {sentences.map(({ classifier, official, deduplicated }) => (
                  <tr key={classifier}>
                    <ClassifierCell id={classifier} />
                    <MetricCell best={bestOfficial === classifier}>
                      {official ? formatMeanSd(official.macro_f1_mean, official.macro_f1_sd) : "n/a"}
                    </MetricCell>
                    <MetricCell best={bestDeduplicated === classifier}>
                      {deduplicated ? formatMeanSd(deduplicated.macro_f1_mean, deduplicated.macro_f1_sd) : "n/a"}
                    </MetricCell>
                    <MetricCell best={bestAccuracy === classifier}>
                      {official ? formatMeanSd(official.accuracy_mean, official.accuracy_sd) : "n/a"}
                    </MetricCell>
                    <td className={td}>
                      {official ? formatCount(official.n_test_mean) : "n/a"} /{" "}
                      {deduplicated ? formatCount(deduplicated.n_test_mean) : "n/a"}
                    </td>
                  </tr>
                ))}
              </tbody>
              {sentenceBaselines.length > 0 && (
                <tbody className="divide-y divide-line border-t border-line-strong text-muted">
                  <GroupRow colSpan={5}>Baseline (no classifier)</GroupRow>
                  {sentenceBaselines.map(({ name, label, official, deduplicated }) => (
                    <tr key={name}>
                      <BaselineCell name={name} detail={baselineDetail({ name, label })} />
                      <td className={td}>
                        {official ? formatMeanSd(official.macro_f1_mean, official.macro_f1_sd) : "n/a"}
                      </td>
                      <td className={td}>
                        {deduplicated ? formatMeanSd(deduplicated.macro_f1_mean, deduplicated.macro_f1_sd) : "n/a"}
                      </td>
                      <td className={td}>
                        {official ? formatMeanSd(official.accuracy_mean, official.accuracy_sd) : "n/a"}
                      </td>
                      <td className={td}>
                        {official ? formatCount(official.n_test_mean) : "n/a"} /{" "}
                        {deduplicated ? formatCount(deduplicated.n_test_mean) : "n/a"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              )}
            </table>
          </TableScroller>
        )}
        <p className={noteClass}>
          Mean ± standard deviation over splits; bold: best mean. Data: {dataset.name}, {dataset.license}.{" "}
          <ExternalLink href={dataset.url}>Dataset</ExternalLink>
        </p>
      </div>
    </div>
  );
}

function ClassifierCell({ id, postHoc = false }: { id: StanceClassifier; postHoc?: boolean }) {
  const info = classifierInfo(id);
  return (
    <th scope="row" className={rowCell}>
      {/* flex, not inline-flex: an inline box would sit on the empty key line's baseline and
          push the name below the first line of the metric cells. */}
      <span className="flex items-center gap-2">
        <SeriesKey info={info} />
        {info.label}
        {postHoc && <PostHocTag />}
      </span>
    </th>
  );
}

/** The name cell of a baseline: no line key (it is not plotted), and what it predicts. */
function BaselineCell({
  name,
  detail,
  postHoc = false,
}: {
  name: StatementBaselineName;
  detail: string;
  postHoc?: boolean;
}) {
  return (
    <th scope="row" className={rowCell}>
      <span className="flex items-center gap-2">
        <span aria-hidden="true" className="w-4 shrink-0" />
        {BASELINE_LABELS[name]}
        {postHoc && <PostHocTag />}
      </span>
      <span className="block pl-6 text-label font-normal text-faint">{detail}</span>
    </th>
  );
}

function PostHocTag() {
  return <span className="text-label font-normal text-faint">post hoc</span>;
}
