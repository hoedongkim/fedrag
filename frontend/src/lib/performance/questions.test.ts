import { describe, expect, it } from "vitest";

import { parseAskParams } from "../ask/params";
import { QUESTION_DETAIL_FIXTURE, QUESTIONS_FIXTURE, REPORT_FIXTURE } from "./fixtures";
import {
  answerCitations,
  answerOutcome,
  askHref,
  filterQuestions,
  hitMarker,
  matchesOutcome,
  missedGold,
  orderedRunDetails,
  type QuestionFilters,
  questionTypes,
  rankLabel,
  rankStrip,
  rankTier,
  runChoices,
  runOf,
} from "./questions";

const best = { method: "hybrid_rerank", config: "tuned" } as const;
const all: QuestionFilters = { split: "all", type: null, outcome: "all", run: best, search: "" };
const ids = (filters: Partial<QuestionFilters>) =>
  filterQuestions(QUESTIONS_FIXTURE, { ...all, ...filters }).map((q) => q.id);

describe("filterQuestions", () => {
  it("keeps every question by default, in id order", () => {
    expect(ids({})).toEqual(["q001", "q002", "q003", "q004"]);
    expect(filterQuestions([...QUESTIONS_FIXTURE].reverse(), all).map((q) => q.id)).toEqual(
      ids({}),
    );
  });

  it("narrows by split, type and text", () => {
    expect(ids({ split: "dev" })).toEqual(["q002"]);
    expect(ids({ split: "test", type: "unanswerable" })).toEqual(["q003"]);
    expect(ids({ search: "  VOLCKER " })).toEqual(["q003"]);
    expect(ids({ search: "q00" })).toHaveLength(4);
  });

  it("narrows by the chosen run's outcome", () => {
    expect(ids({ outcome: "found" })).toEqual(["q001", "q002"]);
    expect(ids({ outcome: "missed" })).toEqual(["q004"]);
    expect(ids({ outcome: "missed", run: { method: "bm25", config: "phase3" } })).toEqual([
      "q001",
      "q002",
    ]);
    expect(ids({ outcome: "incorrect" })).toEqual(["q002", "q004"]);
    expect(ids({ outcome: "unfaithful" })).toEqual(["q002"]);
    expect(ids({ outcome: "refusal_error" })).toEqual(["q004"]);
    expect(ids({ outcome: "refusal_error", run: { method: "bm25", config: "tuned" } })).toEqual([
      "q003",
      "q004",
    ]);
  });

  it("drops a question the chosen run did not cover unless any outcome goes", () => {
    const run = { method: "bm25", config: "nope" } as const;
    expect(ids({ run, outcome: "found" })).toEqual([]);
    expect(ids({ run })).toHaveLength(4);
    expect(runOf(QUESTIONS_FIXTURE[0], run)).toBeNull();
    expect(matchesOutcome(QUESTIONS_FIXTURE[0], null, "missed")).toBe(false);
  });
});

describe("runs and types", () => {
  it("lists types in display order and runs by config, then method", () => {
    expect(questionTypes(QUESTIONS_FIXTURE)).toEqual([
      "minutes_fact",
      "date_specific",
      "paraphrase",
      "unanswerable",
    ]);
    const runs = runChoices(QUESTIONS_FIXTURE, REPORT_FIXTURE.configs);
    expect(runs).toHaveLength(8);
    expect(runs[0]).toEqual({ method: "bm25", config: "phase3" });
    expect(runs[7]).toEqual({ method: "hybrid_rerank", config: "tuned" });
  });
});

describe("ranks and hits", () => {
  it("tiers the first gold rank", () => {
    expect(rankTier(true, 1)).toBe("top5");
    expect(rankTier(true, 5)).toBe("top5");
    expect(rankTier(true, 6)).toBe("top10");
    expect(rankTier(true, null)).toBe("miss");
    expect(rankTier(false, null)).toBe("na");
  });

  it("labels the rank and the answer's outcome", () => {
    expect(rankLabel(true, 3)).toBe("#3");
    expect(rankLabel(true, null)).toBe("miss");
    expect(rankLabel(false, null)).toBe("n/a");
    const run = { answered: true, answer_correctness: 0.5, refusal_correct: null, false_refusal: false };
    expect(answerOutcome(true, run)).toBe("correct 0.50");
    expect(answerOutcome(true, { ...run, false_refusal: true })).toBe("declined");
    expect(answerOutcome(true, { ...run, answer_correctness: null })).toBe("unjudged");
    expect(answerOutcome(true, { ...run, answered: false })).toBe("no answer");
    expect(answerOutcome(false, { ...run, refusal_correct: true })).toBe("declined");
    expect(answerOutcome(false, { ...run, refusal_correct: false })).toBe("answered anyway");
    expect(answerOutcome(false, { ...run, refusal_correct: false, declined: false })).toBe("answered anyway");
    expect(answerOutcome(false, { ...run, refusal_correct: false, declined: true })).toBe(
      "declined, claim faulted",
    );
    expect(answerOutcome(false, run)).toBe("unjudged");
  });

  it("marks hits by gold role, then by document", () => {
    expect(hitMarker({ role: "primary", doc_match: true })).toBe("primary");
    expect(hitMarker({ role: "supporting", doc_match: true })).toBe("supporting");
    expect(hitMarker({ role: null, doc_match: true })).toBe("same_doc");
    expect(hitMarker({ role: null, doc_match: false })).toBe("none");
  });

  it("lays out ranks 1-10, with empty slots past the list", () => {
    const strip = rankStrip(QUESTION_DETAIL_FIXTURE.runs[0]);
    expect(strip).toEqual([
      "primary",
      "same_doc",
      "none",
      "none",
      "none",
      "none",
      "supporting",
      "none",
      "none",
      "none",
    ]);
    expect(rankStrip(QUESTION_DETAIL_FIXTURE.runs[1], 3)).toEqual(["none", "none", "none"]);
  });

  it("lists the gold chunks a run never retrieved", () => {
    const [found, empty] = QUESTION_DETAIL_FIXTURE.runs;
    expect(missedGold(QUESTION_DETAIL_FIXTURE.gold, found)).toEqual([]);
    expect(missedGold(QUESTION_DETAIL_FIXTURE.gold, empty)).toEqual([
      "statement-2024-09-18#0000",
      "minutes-2024-09-18#0013",
    ]);
  });

  it("orders drilldown runs by config, then method", () => {
    expect(
      orderedRunDetails(QUESTION_DETAIL_FIXTURE.runs, REPORT_FIXTURE.configs).map((r) => r.method),
    ).toEqual(["bm25", "hybrid_rerank"]);
  });
});

describe("answer citations and links", () => {
  it("maps [n] to the cited passage with its score, skipping passages not available", () => {
    const run = QUESTION_DETAIL_FIXTURE.runs[0];
    const citations = answerCitations(run.answer!, QUESTION_DETAIL_FIXTURE.passages, run);
    expect([...citations.keys()]).toEqual([1]);
    expect(citations.get(1)).toMatchObject({
      chunk_id: "statement-2024-09-18#0000",
      section: null,
      score: 0.98,
      document: { title: "FOMC Statement, September 18, 2024" },
    });
  });

  it("links to Ask the Fed with the question, as-of date and method filled in", () => {
    const href = askHref(QUESTION_DETAIL_FIXTURE, "bm25");
    expect(href.startsWith("/ask?")).toBe(true);
    expect(parseAskParams(new URLSearchParams(href.slice(5)))).toMatchObject({
      question: QUESTION_DETAIL_FIXTURE.question,
      asOf: "2024-10-01",
      method: "bm25",
    });
  });
});
