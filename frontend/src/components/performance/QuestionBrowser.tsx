"use client";

import { useId } from "react";

import type { QuestionSummary, QuestionType, ReportSplit, RunKey } from "@/lib/api/types";
import { formatDate } from "@/lib/ask/format";
import { runLabel, SPLIT_LABELS, TYPE_LABELS } from "@/lib/performance/format";
import {
  answerOutcome,
  filterQuestions,
  OUTCOME_FILTERS,
  type OutcomeFilter,
  questionTypes,
  rankLabel,
  rankTier,
  runOf,
} from "@/lib/performance/questions";
import { parseRunKey, runKey, sameRun } from "@/lib/performance/report";

import { labelTextClass, smallFieldClass } from "../ui/styles";
import { MetaLine } from "./parts";

/** Outcome filters that read the judge's verdicts. */
const ANSWER_OUTCOMES: readonly OutcomeFilter[] = ["incorrect", "unfaithful", "refusal_error"];

interface QuestionBrowserProps {
  questions: QuestionSummary[];
  configs: { id: string; label: string }[];
  split: ReportSplit;
  type: QuestionType | null;
  outcome: OutcomeFilter;
  run: RunKey;
  runs: RunKey[];
  headline: RunKey | null;
  search: string;
  selected: string | null;
  onChange: (patch: { type?: QuestionType | null; outcome?: OutcomeFilter; run?: RunKey; search?: string }) => void;
  onSelect: (id: string) => void;
}

/** The questions of the split with the chosen run's outcome; selecting one opens its drilldown. */
export function QuestionBrowser({
  questions,
  configs,
  split,
  type,
  outcome,
  run,
  runs,
  headline,
  search,
  selected,
  onChange,
  onSelect,
}: QuestionBrowserProps) {
  const baseId = useId();
  const inSplit = questions.filter((q) => split === "all" || q.split === split);
  const shown = filterQuestions(questions, { split, type, outcome, run, search });
  // An answer filter on a run whose answers here were never judged matches nothing by design.
  const unjudged =
    ANSWER_OUTCOMES.includes(outcome) &&
    !inSplit.some((q) => {
      const r = runOf(q, run);
      return r !== null && (r.answer_correctness !== null || r.refusal_correct !== null || r.faithfulness !== null);
    });

  return (
    <div>
      <div className="flex flex-wrap items-end gap-x-4 gap-y-3">
        <div className="min-w-0">
          <label htmlFor={`${baseId}-type`} className={labelTextClass}>
            Type
          </label>
          <select
            id={`${baseId}-type`}
            value={type ?? ""}
            onChange={(event) => onChange({ type: (event.target.value || null) as QuestionType | null })}
            className={`${smallFieldClass} mt-1 w-auto pr-8`}
          >
            <option value="">All types</option>
            {questionTypes(inSplit).map((t) => (
              <option key={t} value={t}>
                {TYPE_LABELS[t]}
              </option>
            ))}
          </select>
        </div>
        <div className="min-w-0">
          <label htmlFor={`${baseId}-run`} className={labelTextClass}>
            Outcomes of
          </label>
          <select
            id={`${baseId}-run`}
            value={runKey(run)}
            onChange={(event) => {
              const next = parseRunKey(event.target.value);
              if (next) onChange({ run: next });
            }}
            className={`${smallFieldClass} mt-1 w-auto max-w-full pr-8`}
          >
            {runs.map((r) => (
              <option key={runKey(r)} value={runKey(r)}>
                {runLabel(configs, r)}
                {headline && sameRun(headline, r) ? " (chosen)" : ""}
              </option>
            ))}
          </select>
        </div>
        <div className="min-w-0">
          <label htmlFor={`${baseId}-outcome`} className={labelTextClass}>
            Show
          </label>
          <select
            id={`${baseId}-outcome`}
            value={outcome}
            onChange={(event) => onChange({ outcome: event.target.value as OutcomeFilter })}
            className={`${smallFieldClass} mt-1 w-auto max-w-full pr-8`}
          >
            {OUTCOME_FILTERS.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        </div>
        <div className="min-w-0 flex-1 basis-48">
          <label htmlFor={`${baseId}-search`} className={labelTextClass}>
            Search
          </label>
          <input
            id={`${baseId}-search`}
            type="search"
            value={search}
            placeholder="Words in the question, or its id"
            onChange={(event) => onChange({ search: event.target.value })}
            className={`${smallFieldClass} mt-1`}
          />
        </div>
      </div>

      <div className="mt-4 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-0.5 text-label text-faint">
        <p role="status">
          {shown.length} of {inSplit.length} questions
          {split === "all" ? "" : ` in the ${SPLIT_LABELS[split].toLowerCase()} split`}
        </p>
        {/* A key for the right-hand column (each row also says it to screen readers). */}
        <p aria-hidden="true" className="ml-auto pr-2 text-right whitespace-nowrap">
          First right passage · answer
        </p>
      </div>

      {shown.length === 0 ? (
        <p className="mt-3 border-y border-line py-6 text-small text-muted">
          {unjudged
            ? `The answers of ${runLabel(configs, run)} were not judged${split === "all" ? "" : ` on the ${SPLIT_LABELS[split].toLowerCase()} split`}, so answer filters do not apply.`
            : "No question matches these filters."}
        </p>
      ) : (
        <ul
          aria-label="Questions"
          // relative: the sr-only text in each row is absolutely positioned and must be clipped here,
          // or the rows hidden below the fold would stretch the page.
          className="relative mt-2 max-h-[28rem] divide-y divide-line overflow-y-auto border-y border-line"
        >
          {shown.map((q) => {
            const outcomeRun = runOf(q, run);
            const current = q.id === selected;
            return (
              <li key={q.id}>
                <button
                  type="button"
                  data-question-id={q.id}
                  aria-current={current ? "true" : undefined}
                  onClick={() => onSelect(q.id)}
                  className={[
                    "flex w-full items-start gap-3 px-2 py-3 text-left transition-colors focus-visible:-outline-offset-2",
                    current ? "bg-accent-soft shadow-[inset_2px_0_0_var(--accent)]" : "hover:bg-subtle",
                  ].join(" ")}
                >
                  <span className="mt-px w-11 shrink-0 font-mono text-label text-faint">{q.id}</span>
                  <span className="min-w-0 flex-1">
                    <span className="line-clamp-2 text-small text-ink">{q.question}</span>
                    <MetaLine
                      className="mt-0.5 block"
                      items={[
                        TYPE_LABELS[q.type],
                        split === "all" && SPLIT_LABELS[q.split],
                        q.as_of && `as of ${formatDate(q.as_of)}`,
                        q.human_reviewed && "checked by a person",
                      ]}
                    />
                  </span>
                  <Outcome answerable={q.answerable} run={outcomeRun} />
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}

const TIER_CLASSES = {
  top5: "font-semibold text-ink",
  top10: "font-medium text-muted",
  miss: "font-medium text-bad",
  na: "text-faint",
} as const;

/** The chosen run on one question: rank of the first gold passage, and the answer's scores. */
function Outcome({
  answerable,
  run,
}: {
  answerable: boolean;
  run: ReturnType<typeof runOf>;
}) {
  if (!run) return <span className="shrink-0 text-label text-faint">not run</span>;
  const tier = rankTier(answerable, run.first_relevant_rank);
  return (
    <span className="flex w-24 shrink-0 flex-col items-end text-right">
      <span
        title={answerable ? "Rank of the first gold passage" : "Unanswerable: no gold passage"}
        className={`text-small tabular-nums ${TIER_CLASSES[tier]}`}
      >
        <span className="sr-only">first gold passage: </span>
        {rankLabel(answerable, run.first_relevant_rank)}
      </span>
      <span className="text-label text-faint tabular-nums">{answerOutcome(answerable, run)}</span>
    </span>
  );
}
