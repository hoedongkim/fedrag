/**
 * FIXTURE for unit tests only: never import it from app code, so it stays out of the bundle
 * (ESLint enforces this for src/app and src/components).
 *
 * A GET /api/evaluation report and one GET /api/evaluation/questions/{id} drilldown, shaped
 * exactly like the Phase 7 contract. Every number is invented for the tests.
 */
import type {
  AnswerRow,
  Estimate,
  EvaluationReport,
  QuestionDetail,
  QuestionSummary,
  ReportSplit,
  RetrievalMethod,
  RetrievalRow,
  RunSummary,
} from "../api/types";

const METHODS: RetrievalMethod[] = ["bm25", "dense", "hybrid", "hybrid_rerank"];
const CONFIGS = ["phase3", "tuned"];
const SPLITS: ReportSplit[] = ["dev", "test", "all"];
const BASE: Record<RetrievalMethod, number> = { bm25: 0.5, dense: 0.55, hybrid: 0.6, hybrid_rerank: 0.7 };

function estimate(value: number | null, n = 60): Estimate {
  if (value === null) return { value: null, ci: null, n: 0 };
  const v = Number(value.toFixed(3));
  return { value: v, ci: { low: Math.max(0, v - 0.1), high: Math.min(1, v + 0.08) }, n };
}

function score(method: RetrievalMethod, config: string, split: ReportSplit): number {
  return BASE[method] + (config === "tuned" ? 0.05 : 0) + (split === "dev" ? 0.02 : 0);
}

function retrievalRow(method: RetrievalMethod, config: string, split: ReportSplit): RetrievalRow {
  const v = score(method, config, split);
  return {
    method,
    config,
    split,
    n_questions: 60,
    recall_at_5: estimate(v),
    recall_at_10: estimate(v + 0.1),
    mrr_at_10: estimate(v - 0.1),
    ndcg_at_10: estimate(v - 0.05),
    doc_recall_at_5: estimate(v + 0.05),
    doc_recall_at_10: estimate(v + 0.15),
    latency_ms_p50: method === "bm25" ? 5 : method === "hybrid_rerank" ? 300 : 180,
    latency_ms_p95: method === "bm25" ? 9 : method === "hybrid_rerank" ? 520 : 260,
    as_of_leaks: 0,
  };
}

function answerRow(method: RetrievalMethod, config: string, split: ReportSplit): AnswerRow {
  const v = score(method, config, split);
  return {
    method,
    config,
    split,
    n_questions: 70,
    faithfulness: estimate(Math.min(0.95, v + 0.2)),
    citation_accuracy: estimate(Math.min(0.95, v + 0.15)),
    answer_correctness: estimate(v),
    refusal_accuracy: estimate(0.9, 10),
    false_refusal_rate: estimate(method === "bm25" ? 0.15 : 0.08),
    latency_ms_p50: 2400,
    latency_ms_p95: 4100,
    cost_per_query_usd: 0.0012,
    judge_cost_per_query_usd: 0.004,
  };
}

function run(
  method: RetrievalMethod,
  config: string,
  rank: number | null,
  fields: Partial<RunSummary> = {},
): RunSummary {
  return {
    method,
    config,
    first_relevant_rank: rank,
    recall_at_5: rank !== null && rank <= 5 ? 1 : 0,
    ndcg_at_10: rank ? Number((1 / Math.log2(rank + 1)).toFixed(3)) : 0,
    answered: true,
    faithfulness: 1,
    citation_accuracy: 1,
    answer_correctness: 1,
    refusal_correct: null,
    false_refusal: false,
    declined: false,
    ...fields,
  };
}

function runs(ranks: Record<string, number | null>, fields: Partial<RunSummary> = {}): RunSummary[] {
  return CONFIGS.flatMap((config) =>
    METHODS.map((method) => run(method, config, ranks[`${method}:${config}`] ?? null, fields)),
  );
}

export const QUESTIONS_FIXTURE: QuestionSummary[] = [
  {
    id: "q001",
    question: "How much did the FOMC cut rates in September 2024, and who dissented?",
    as_of: "2024-10-01",
    type: "date_specific",
    split: "test",
    answerable: true,
    human_reviewed: true,
    runs: runs({
      "bm25:phase3": null,
      "dense:phase3": 4,
      "hybrid:phase3": 12,
      "hybrid_rerank:phase3": 7,
      "bm25:tuned": 3,
      "dense:tuned": 4,
      "hybrid:tuned": 2,
      "hybrid_rerank:tuned": 1,
    }),
  },
  {
    id: "q002",
    question: "Were officials worried that prices were rising too fast in early 2022?",
    as_of: null,
    type: "paraphrase",
    split: "dev",
    answerable: true,
    human_reviewed: false,
    runs: runs(
      { "dense:phase3": 1, "hybrid_rerank:phase3": 2, "dense:tuned": 1, "hybrid_rerank:tuned": 1 },
      { answer_correctness: 0.5, faithfulness: 0.75 },
    ),
  },
  {
    id: "q003",
    question: "Why did Paul Volcker raise rates in 1979?",
    as_of: null,
    type: "unanswerable",
    split: "test",
    answerable: false,
    human_reviewed: false,
    runs: runs(
      {},
      {
        recall_at_5: null,
        ndcg_at_10: null,
        faithfulness: null,
        citation_accuracy: null,
        answer_correctness: null,
        refusal_correct: true,
        false_refusal: null,
      },
    ).map((r) => (r.method === "bm25" ? { ...r, refusal_correct: false } : r)),
  },
  {
    id: "q004",
    question: "How did participants see tariff effects on inflation in May 2025?",
    as_of: null,
    type: "minutes_fact",
    split: "test",
    answerable: true,
    human_reviewed: true,
    runs: runs(
      { "bm25:phase3": 2, "bm25:tuned": 1, "hybrid:tuned": 9 },
      { answer_correctness: 0, false_refusal: true, faithfulness: null, citation_accuracy: null },
    ),
  },
];

export const REPORT_FIXTURE: EvaluationReport = {
  generated_at: "2026-10-13T18:42:07Z",
  dataset: {
    file: "data/eval/rag_questions.jsonl",
    sha256: null,
    n_questions: 85,
    n_answerable: 75,
    n_with_as_of: 21,
    n_human_reviewed: 20,
    splits: [
      { split: "dev", n: 25, n_answerable: 22 },
      { split: "test", n: 60, n_answerable: 53 },
    ],
    types: [
      { type: "statement_fact", n: 14, n_answerable: 14 },
      { type: "minutes_fact", n: 14, n_answerable: 14 },
      { type: "paraphrase", n: 11, n_answerable: 11 },
      { type: "unanswerable", n: 10, n_answerable: 0 },
    ],
    construction: "Drafted by LLMs from the indexed chunks; 20 reviewed by a human.",
  },
  configs: [
    {
      id: "phase3",
      label: "Phase 3 settings",
      description: "As shipped in Phase 3.",
      hash: "c0ffee12",
      params: { stemming: false },
    },
    {
      id: "tuned",
      label: "Tuned on dev",
      description: "Stemming and a statement boost.",
      hash: "beef4567",
      params: { stemming: true },
    },
  ],
  generator: {
    model: "gpt-5.4-mini-2026-03-17",
    reasoning_effort: "low",
    prompt_version: "answer.3fa2c1d0",
    top_k: 5,
  },
  judge: {
    model: "gpt-5.5",
    reasoning_effort: "medium",
    prompt_version: "judge-v1",
    n_answers: 680,
    validation: {
      status: "complete",
      n_sampled: 40,
      n_annotated: 40,
      annotator: "Two annotators, blind to the verdicts.",
      annotators: [
        { id: "A", description: "First annotator.", isolated: false },
        { id: "B", description: "Second annotator.", isolated: false },
      ],
      sample_file: "data/eval/rag_judge_sample.json",
      sampled_runs: [{ method: "hybrid_rerank", config: "phase3", n: 40 }],
      agreement: [
        { check: "claim_support", annotator: "A", n: 112, agreement: 0.893, exact_agreement: 0.88, kappa: 0.68 },
        { check: "refusal", annotator: "A", n: 40, agreement: 0.975, exact_agreement: 0.9, kappa: null },
        { check: "claim_support", annotator: "B", n: 112, agreement: 0.875, exact_agreement: 0.86, kappa: 0.61 },
      ],
      inter_annotator: [
        { check: "claim_support", annotators: ["A", "B"], n: 112, agreement: 0.95, exact_agreement: 0.94, kappa: 0.8 },
      ],
      note: null,
    },
  },
  bootstrap: { method: "Percentile bootstrap over questions.", resamples: 10000, seed: 20261013, level: 0.95 },
  selection: {
    method: "hybrid_rerank",
    config: "tuned",
    metric: "ndcg_at_10",
    split: "dev",
    rule: "Highest dev nDCG@10.",
  },
  retrieval: SPLITS.flatMap((split) =>
    CONFIGS.flatMap((config) => METHODS.map((method) => retrievalRow(method, config, split))),
  ),
  answers: SPLITS.flatMap((split) =>
    CONFIGS.flatMap((config) => METHODS.map((method) => answerRow(method, config, split))),
  ),
  per_type: [
    {
      method: "hybrid_rerank",
      config: "tuned",
      split: "test",
      type: "statement_fact",
      n_questions: 10,
      recall_at_5: estimate(0.9, 10),
      recall_at_10: null,
      mrr_at_10: null,
      ndcg_at_10: null,
      doc_recall_at_5: null,
      doc_recall_at_10: null,
      faithfulness: null,
      citation_accuracy: null,
      answer_correctness: null,
      refusal_accuracy: null,
      false_refusal_rate: null,
    },
    {
      method: "bm25",
      config: "tuned",
      split: "test",
      type: "paraphrase",
      n_questions: 8,
      recall_at_5: estimate(0.15, 8),
      recall_at_10: null,
      mrr_at_10: null,
      ndcg_at_10: null,
      doc_recall_at_5: null,
      doc_recall_at_10: null,
      faithfulness: null,
      citation_accuracy: null,
      answer_correctness: null,
      refusal_accuracy: null,
      false_refusal_rate: null,
    },
    {
      method: "bm25",
      config: "tuned",
      split: "test",
      type: "unanswerable",
      n_questions: 7,
      recall_at_5: estimate(null),
      recall_at_10: null,
      mrr_at_10: null,
      ndcg_at_10: null,
      doc_recall_at_5: null,
      doc_recall_at_10: null,
      faithfulness: null,
      citation_accuracy: null,
      answer_correctness: null,
      refusal_accuracy: estimate(0.86, 7),
      false_refusal_rate: null,
    },
  ],
  differences: [
    {
      metric: "recall_at_5",
      split: "test",
      minuend: { method: "hybrid_rerank", config: "tuned" },
      subtrahend: { method: "hybrid_rerank", config: "phase3" },
      n: 53,
      difference: 0.05,
      ci: { low: -0.02, high: 0.12 },
    },
    {
      metric: "faithfulness",
      split: "test",
      minuend: { method: "hybrid_rerank", config: "phase3" },
      subtrahend: { method: "hybrid_rerank", config: "tuned" },
      n: 60,
      difference: -0.04,
      ci: { low: -0.09, high: 0.01 },
    },
    {
      metric: "recall_at_5",
      split: "test",
      minuend: { method: "hybrid_rerank", config: "tuned" },
      subtrahend: { method: "bm25", config: "tuned" },
      n: 53,
      difference: 0.15,
      ci: { low: 0.07, high: 0.24 },
    },
  ],
  cost: {
    total_usd: 0.91,
    max_usd: 1,
    lines: [
      {
        stage: "generation",
        model: "gpt-5.4-mini-2026-03-17",
        calls: 680,
        input_tokens: 1632000,
        cached_input_tokens: 204000,
        output_tokens: 176800,
        usd: 0.82,
      },
      {
        stage: "embedding",
        model: "text-embedding-3-small",
        calls: 85,
        input_tokens: 1900,
        cached_input_tokens: 0,
        output_tokens: 0,
        usd: 0.0001,
      },
    ],
  },
  notes: [],
  questions: QUESTIONS_FIXTURE,
};

const STATEMENT = "statement-2024-09-18";
const MINUTES = "minutes-2024-09-18";

export const QUESTION_DETAIL_FIXTURE: QuestionDetail = {
  id: "q001",
  question: QUESTIONS_FIXTURE[0].question,
  as_of: "2024-10-01",
  type: "date_specific",
  split: "test",
  answerable: true,
  human_reviewed: true,
  reference_answer: "By 1/2 point, to 4-3/4 to 5 percent; Governor Bowman dissented.",
  required_facts: ["Cut by 50 basis points", "Bowman dissented"],
  relevant_doc_ids: [STATEMENT, MINUTES],
  notes: null,
  gold: [
    { chunk_id: `${STATEMENT}#0000`, role: "primary" },
    { chunk_id: `${MINUTES}#0013`, role: "supporting" },
  ],
  runs: [
    {
      method: "hybrid_rerank",
      config: "tuned",
      latency_ms: 288,
      scores: {
        first_relevant_rank: 1,
        recall_at_5: 0.5,
        recall_at_10: 1,
        mrr_at_10: 1,
        ndcg_at_10: 0.84,
        doc_recall_at_5: 1,
        doc_recall_at_10: 1,
      },
      retrieved: [
        { rank: 1, chunk_id: `${STATEMENT}#0000`, doc_id: STATEMENT, score: 0.98, role: "primary", doc_match: true },
        { rank: 2, chunk_id: `${MINUTES}#0012`, doc_id: MINUTES, score: 0.71, role: null, doc_match: true },
        { rank: 3, chunk_id: "minutes-2024-07-31#0003", doc_id: "minutes-2024-07-31", score: 0.4, role: null, doc_match: false },
        { rank: 7, chunk_id: `${MINUTES}#0013`, doc_id: MINUTES, score: 0.2, role: "supporting", doc_match: true },
      ],
      answer: {
        text: "The Committee lowered the target range by 1/2 percentage point [1]. Governor Bowman dissented [2].",
        citations: [
          { n: 1, chunk_id: `${STATEMENT}#0000`, rank: 1, role: "primary", doc_match: true },
          { n: 2, chunk_id: `${MINUTES}#0012`, rank: 2, role: null, doc_match: true },
        ],
        notes: [],
        latency_ms: 2300,
        cost_usd: 0.0012,
        judge: {
          faithfulness: 0.5,
          citation_accuracy: 0.5,
          answer_correctness: 1,
          refusal_correct: null,
          false_refusal: false,
          claims: [
            { claim: "The Committee lowered the range by 1/2 point.", citations: [1], supported: true, reason: null },
            { claim: "Governor Bowman dissented.", citations: [2], supported: false, reason: "Not in the passage." },
          ],
          citations: [
            { n: 1, sentence: "The Committee lowered the target range by 1/2 percentage point [1].", supported: true, reason: null },
            { n: 2, sentence: "Governor Bowman dissented [2].", supported: false, reason: "Wrong passage." },
          ],
          facts: [
            { fact: "Cut by 50 basis points", covered: true, reason: null },
            { fact: "Bowman dissented", covered: true, reason: null },
          ],
          refusal: { declined: false, fabricated: false, reason: null },
        },
      },
    },
    {
      method: "bm25",
      config: "phase3",
      latency_ms: 5,
      scores: {
        first_relevant_rank: null,
        recall_at_5: 0,
        recall_at_10: 0,
        mrr_at_10: 0,
        ndcg_at_10: 0,
        doc_recall_at_5: 0,
        doc_recall_at_10: 0.5,
      },
      retrieved: [],
      answer: null,
    },
  ],
  passages: {
    [`${STATEMENT}#0000`]: {
      chunk_id: `${STATEMENT}#0000`,
      document: {
        doc_id: STATEMENT,
        doc_type: "statement",
        meeting_date: "2024-09-18",
        release_date: "2024-09-18",
        page_updated: "2024-09-18",
        title: "FOMC Statement, September 18, 2024",
        url: "https://www.federalreserve.gov/newsevents/pressreleases/monetary20240918a.htm",
      },
      section: null,
      text: "The Committee decided to lower the target range for the federal funds rate by 1/2 percentage point to 4-3/4 to 5 percent.",
    },
  },
  passages_available: true,
};
