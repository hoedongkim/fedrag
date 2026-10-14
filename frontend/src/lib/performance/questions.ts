/** The question browser: filtering the evaluation questions and reading one run's outcome. */
import type {
  AnswerDetail,
  Citation,
  EvalPassage,
  QuestionDetail,
  QuestionSummary,
  QuestionType,
  ReportSplit,
  RetrievedHit,
  RunDetail,
  RunKey,
  RunSummary,
} from "../api/types";
import { serializeAskParams, TOP_K_DEFAULT } from "../ask/params";
import { METHOD_ORDER, sameRun, TYPE_ORDER } from "./report";

/** What to narrow the list to, judged on the chosen run. */
export type OutcomeFilter = "all" | "missed" | "found" | "incorrect" | "unfaithful" | "refusal_error";

export const OUTCOME_FILTERS: readonly { value: OutcomeFilter; label: string }[] = [
  { value: "all", label: "Any outcome" },
  { value: "missed", label: "No gold passage in the top 10" },
  { value: "found", label: "Gold passage in the top 5" },
  { value: "incorrect", label: "Answer misses a required fact" },
  { value: "unfaithful", label: "Answer has an unsupported claim" },
  { value: "refusal_error", label: "Wrong refusal (or none)" },
];

export function isOutcomeFilter(value: string | null | undefined): value is OutcomeFilter {
  return OUTCOME_FILTERS.some((o) => o.value === value);
}

export interface QuestionFilters {
  /** Questions of this split; "all" for every question. */
  split: ReportSplit;
  type: QuestionType | null;
  outcome: OutcomeFilter;
  /** Run the outcome is judged on. */
  run: RunKey;
  /** Case-insensitive text in the question or its id. */
  search: string;
}

/** A question's summary under one run, or null if that run did not cover it. */
export function runOf(question: Pick<QuestionSummary, "runs">, run: RunKey): RunSummary | null {
  return question.runs.find((r) => sameRun(r, run)) ?? null;
}

/** True when the run's outcome on the question matches the filter. */
export function matchesOutcome(
  question: Pick<QuestionSummary, "answerable">,
  run: RunSummary | null,
  outcome: OutcomeFilter,
): boolean {
  if (outcome === "all") return true;
  if (!run) return false;
  switch (outcome) {
    case "missed":
      return question.answerable && run.first_relevant_rank === null;
    case "found":
      return question.answerable && run.first_relevant_rank !== null && run.first_relevant_rank <= 5;
    case "incorrect":
      return run.answer_correctness !== null && run.answer_correctness < 1;
    case "unfaithful":
      return run.faithfulness !== null && run.faithfulness < 1;
    case "refusal_error":
      return run.refusal_correct === false || run.false_refusal === true;
  }
}

/** The questions that pass every filter, in id order. */
export function filterQuestions(
  questions: readonly QuestionSummary[],
  filters: QuestionFilters,
): QuestionSummary[] {
  const needle = filters.search.trim().toLowerCase();
  return questions
    .filter(
      (q) =>
        (filters.split === "all" || q.split === filters.split) &&
        (filters.type === null || q.type === filters.type) &&
        (!needle || q.question.toLowerCase().includes(needle) || q.id.toLowerCase().includes(needle)) &&
        matchesOutcome(q, runOf(q, filters.run), filters.outcome),
    )
    .sort((a, b) => a.id.localeCompare(b.id, "en", { numeric: true }));
}

/** Question types present in the list, in display order. */
export function questionTypes(questions: readonly Pick<QuestionSummary, "type">[]): QuestionType[] {
  return TYPE_ORDER.filter((type) => questions.some((q) => q.type === type));
}

/** The runs of the report, by config (report order) then method: what the run picker offers. */
export function runChoices(
  questions: readonly Pick<QuestionSummary, "runs">[],
  configs: readonly { id: string }[],
): RunKey[] {
  const runs: RunKey[] = [];
  for (const config of configs) {
    for (const method of METHOD_ORDER) {
      if (questions.some((q) => q.runs.some((r) => r.method === method && r.config === config.id))) {
        runs.push({ method, config: config.id });
      }
    }
  }
  return runs;
}

export type RankTier = "top5" | "top10" | "miss" | "na";

/** How far down the first gold passage came: in the top 5, the top 10, not at all, or n/a. */
export function rankTier(answerable: boolean, rank: number | null): RankTier {
  if (!answerable) return "na";
  if (rank === null) return "miss";
  return rank <= 5 ? "top5" : "top10";
}

/** "#3", "miss" or "n/a": the first gold rank as the question list shows it. */
export function rankLabel(answerable: boolean, rank: number | null): string {
  const tier = rankTier(answerable, rank);
  if (tier === "na") return "n/a";
  return tier === "miss" ? "miss" : `#${rank}`;
}

/**
 * What happened to the answer, in a few words: "correct 0.50", "declined", "answered anyway", or
 * "declined, claim faulted" when an unanswerable question was declined but the judge found an
 * unsupported claim beside the decline (which fails the refusal check).
 */
export function answerOutcome(
  answerable: boolean,
  run: Pick<RunSummary, "answered" | "answer_correctness" | "refusal_correct" | "false_refusal"> &
    Partial<Pick<RunSummary, "declined">>,
): string {
  if (!run.answered) return "no answer";
  if (!answerable) {
    if (run.refusal_correct === null) return "unjudged";
    if (run.refusal_correct) return "declined";
    return run.declined ? "declined, claim faulted" : "answered anyway";
  }
  if (run.false_refusal) return "declined";
  if (run.answer_correctness === null) return "unjudged";
  return `correct ${run.answer_correctness.toFixed(2)}`;
}

/** One hit's marker: a gold passage (by role), another passage of a relevant document, or none. */
export type HitMarker = "primary" | "supporting" | "same_doc" | "none";

export function hitMarker(hit: Pick<RetrievedHit, "role" | "doc_match">): HitMarker {
  if (hit.role) return hit.role;
  return hit.doc_match ? "same_doc" : "none";
}

/**
 * The ranks 1..depth of a run as markers, for the rank strip: a slot past the end of the list
 * (fewer hits than depth) is "none".
 */
export function rankStrip(run: Pick<RunDetail, "retrieved">, depth = 10): HitMarker[] {
  const byRank = new Map(run.retrieved.map((hit) => [hit.rank, hit]));
  return Array.from({ length: depth }, (_, i) => {
    const hit = byRank.get(i + 1);
    return hit ? hitMarker(hit) : "none";
  });
}

/** Gold chunks a run did not retrieve at all (in its stored top list). */
export function missedGold(
  gold: readonly { chunk_id: string }[],
  run: Pick<RunDetail, "retrieved">,
): string[] {
  const found = new Set(run.retrieved.map((hit) => hit.chunk_id));
  return gold.map((g) => g.chunk_id).filter((id) => !found.has(id));
}

/** The drilldown's run in display order: by config (report order), then method. */
export function orderedRunDetails<T extends { method: RunKey["method"]; config: string }>(
  runs: readonly T[],
  configs: readonly { id: string }[],
): T[] {
  const ci = (id: string) => {
    const i = configs.findIndex((c) => c.id === id);
    return i < 0 ? configs.length : i;
  };
  return [...runs].sort(
    (a, b) => ci(a.config) - ci(b.config) || METHOD_ORDER.indexOf(a.method) - METHOD_ORDER.indexOf(b.method),
  );
}

/**
 * An answer's citations by [n], as Ask the Fed's answer text takes them: the cited passage with
 * its retrieval score. A citation whose passage is not available here is left out (its marker
 * then shows as a plain number).
 */
export function answerCitations(
  answer: Pick<AnswerDetail, "citations">,
  passages: Readonly<Record<string, EvalPassage>>,
  run: Pick<RunDetail, "retrieved">,
): Map<number, Citation> {
  const scores = new Map(run.retrieved.map((hit) => [hit.chunk_id, hit.score]));
  const citations = new Map<number, Citation>();
  for (const cited of answer.citations) {
    const passage = passages[cited.chunk_id];
    if (!passage) continue;
    citations.set(cited.n, {
      document: passage.document,
      chunk_id: cited.chunk_id,
      section: passage.section,
      snippet: passage.text, // the whole passage already
      passage: null,
      score: scores.get(cited.chunk_id) ?? null,
    });
  }
  return citations;
}

/** Link to Ask the Fed with the question, its as-of date and the method filled in (not sent). */
export function askHref(
  question: Pick<QuestionDetail, "question" | "as_of">,
  method: RunKey["method"],
): string {
  const query = serializeAskParams({
    question: question.question,
    asOf: question.as_of ?? "",
    method,
    topK: TOP_K_DEFAULT,
  });
  return `/ask?${query}`;
}
