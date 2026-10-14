/**
 * Recorded answers (GET /api/ask/examples): real answers the system wrote during the evaluation,
 * shown on the public demo in place of live questions. The payload (AskExample[] in the API
 * contract) is validated here rather than trusted: the file behind it is exported by a script and
 * re-read on change, so an example whose answer or citations do not hold up is left out instead
 * of breaking the page.
 */
import { ApiError } from "../api/client";
import type { AskRequest, AskResponse, Citation, RetrievalMethod } from "../api/types";
import { isApiErrorBody } from "../api/wake";
import { describeStanceError, type StanceFailure } from "../stance/failures";
import type { AskState } from "./answerState";
import { RETRIEVAL_METHODS } from "./params";

/** One item of the judge's verdict on a recorded answer, ready to print ("Claims supported", "3 of 3"). */
export interface JudgeItem {
  label: string;
  value: string;
}

export interface JudgeSummary {
  /** The judge model, e.g. "gpt-5.5"; null when not recorded. */
  model: string | null;
  items: JudgeItem[];
  /** A one-line summary written by the backend, if any. */
  text: string | null;
}

export interface RecordedAnswer {
  /** Unique within the list: the evaluation question id when there is one. */
  id: string;
  question: string;
  asOf: string | null;
  answer: string;
  citations: Citation[];
  notes: string[];
  model: string | null;
  retrieval: RetrievalMethod;
  /** Retriever settings, e.g. "tuned"; null when not recorded. */
  config: string | null;
  topK: number | null;
  latencyMs: number | null;
  /** When the answer was generated (ISO date or date-time); null when not recorded. */
  recordedAt: string | null;
  /** The evaluation question type, e.g. "point_in_time"; null when not recorded. */
  type: string | null;
  /** What the example shows, in a few words, e.g. "Point in time". */
  label: string | null;
  /** One sentence on why it was picked. */
  description: string | null;
  judge: JudgeSummary | null;
}

type Rec = Record<string, unknown>;

function isRecord(value: unknown): value is Rec {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function text(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value : null;
}

function finite(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

const ISO_DATE = /^\d{4}-\d{2}-\d{2}/;

function isoDate(value: unknown): string | null {
  const s = text(value);
  return s && ISO_DATE.test(s) ? s.slice(0, 10) : null;
}

const METHODS: readonly string[] = RETRIEVAL_METHODS.map((m) => m.value);

/** A citation shaped like AskResponse's, or null. */
function citation(value: unknown): Citation | null {
  if (!isRecord(value) || !isRecord(value.document)) return null;
  const doc = value.document;
  const snippet = text(value.snippet);
  const chunkId = text(value.chunk_id);
  if (!snippet || !chunkId || !text(doc.title) || !text(doc.url) || !isoDate(doc.release_date)) return null;
  return {
    ...(value as unknown as Citation),
    section: text(value.section),
    // The whole passage when the snippet is only its beginning (recorded answers only).
    passage: text(value.passage),
    score: finite(value.score),
    document: {
      ...(doc as unknown as Citation["document"]),
      page_updated: isoDate(doc.page_updated),
    },
  };
}

/**
 * How many of a kind of verdict passed: from the per-item list when there is one, else from the
 * counts (`n_claims` and `n_claims_supported`, ...); null when there is nothing to count.
 */
function count(
  items: unknown,
  passes: (item: Rec) => boolean,
  total: unknown,
  passed: unknown,
): { passed: number; total: number } | null {
  if (Array.isArray(items) && items.length > 0) {
    const records = items.filter(isRecord);
    return { passed: records.filter(passes).length, total: records.length };
  }
  const n = finite(total);
  const k = finite(passed);
  return n !== null && n > 0 && k !== null ? { passed: k, total: n } : null;
}

const share = (value: number) => value.toFixed(2);

/**
 * The judge's verdict as a few printable items. Takes the backend's ExampleJudgement (scores,
 * claim and fact counts, refusal checks), the evaluation's JudgeVerdicts (per-claim, per-citation
 * and per-fact verdicts, counted here), or a summary string.
 */
export function parseJudgeSummary(value: unknown): JudgeSummary | null {
  if (typeof value === "string") return value.trim() ? { model: null, items: [], text: value.trim() } : null;
  if (!isRecord(value)) return null;
  const items: JudgeItem[] = [];

  const claims = count(
    value.claims,
    (c) => c.supported === true || c.verdict === "supported",
    value.n_claims,
    value.n_claims_supported,
  );
  const faithfulness = finite(value.faithfulness);
  if (claims) items.push({ label: "Claims supported", value: `${claims.passed} of ${claims.total}` });
  else if (faithfulness !== null) items.push({ label: "Faithfulness", value: share(faithfulness) });

  const cites = count(
    value.citations,
    (c) => c.supported === true || c.verdict === "supports",
    value.n_citations,
    value.n_citations_supported,
  );
  const citationAccuracy = finite(value.citation_accuracy);
  if (cites) items.push({ label: "Citations that support their sentence", value: `${cites.passed} of ${cites.total}` });
  else if (citationAccuracy !== null) items.push({ label: "Citation accuracy", value: share(citationAccuracy) });

  const facts = count(
    value.facts,
    (f) => f.covered === true || f.verdict === "covered",
    value.n_facts,
    value.n_facts_covered,
  );
  const correctness = finite(value.answer_correctness ?? value.correctness);
  if (facts) items.push({ label: "Required facts stated", value: `${facts.passed} of ${facts.total}` });
  else if (correctness !== null) items.push({ label: "Correctness", value: share(correctness) });

  if (typeof value.refusal_correct === "boolean") {
    items.push({
      label: "Unanswerable question",
      value: value.refusal_correct ? "declined, nothing invented" : "not declined correctly",
    });
  }
  if (value.false_refusal === true) items.push({ label: "Answerable question", value: "declined anyway" });

  const summary = text(value.summary) ?? text(value.text);
  if (items.length === 0 && !summary) return null;
  return { model: text(value.model) ?? text(value.judge_model), items, text: summary };
}

function recordedAnswer(item: unknown, index: number): RecordedAnswer | null {
  if (!isRecord(item)) return null;
  // Accept the fields at the top level, or an AskResponse under `response` and a request under `request`.
  const src: Rec = {
    ...(isRecord(item.request) ? item.request : {}),
    ...(isRecord(item.response) ? item.response : {}),
    ...item,
  };
  const question = text(src.question);
  const answer = typeof src.answer === "string" ? src.answer : null;
  if (!question || answer === null || !Array.isArray(src.citations)) return null;
  const citations = src.citations.map(citation);
  if (citations.some((c) => c === null)) return null;

  const method = text(src.retrieval) ?? text(src.method);
  return {
    id: text(src.id) ?? text(src.question_id) ?? `example-${index + 1}`,
    question,
    asOf: isoDate(src.as_of),
    answer,
    citations: citations as Citation[],
    notes: Array.isArray(src.notes) ? src.notes.filter((n): n is string => typeof n === "string") : [],
    model: text(src.model),
    retrieval: method && METHODS.includes(method) ? (method as RetrievalMethod) : "hybrid_rerank",
    config: text(src.retriever_preset) ?? text(src.config) ?? text(src.preset),
    topK: finite(src.top_k),
    latencyMs: finite(src.latency_ms),
    recordedAt: text(src.recorded_at) ?? text(src.created_at),
    type: text(src.type) ?? text(src.question_type),
    label: text(src.label) ?? text(src.title),
    description: text(src.description),
    judge: parseJudgeSummary(src.judge ?? src.judge_summary ?? src.verdict),
  };
}

/** The recorded answers in a GET /api/ask/examples payload (a list, or `{examples: [...]}`). */
export function parseRecordedAnswers(payload: unknown): RecordedAnswer[] {
  const list = Array.isArray(payload) ? payload : isRecord(payload) && Array.isArray(payload.examples) ? payload.examples : [];
  const seen = new Set<string>();
  const out: RecordedAnswer[] = [];
  list.forEach((item, i) => {
    const example = recordedAnswer(item, i);
    if (!example) return;
    let id = example.id;
    for (let k = 2; seen.has(id); k += 1) id = `${example.id}-${k}`;
    seen.add(id);
    out.push({ ...example, id });
  });
  return out;
}

/** The answer panel's state for a recorded answer: a finished answer, as if just streamed. */
export function recordedAskState(example: RecordedAnswer, run = 0): AskState {
  const request: AskRequest = {
    question: example.question,
    as_of: example.asOf,
    retrieval: example.retrieval,
    ...(example.topK !== null ? { top_k: example.topK } : {}),
  };
  const result: AskResponse = {
    answer: example.answer,
    citations: example.citations,
    notes: example.notes,
    retrieval: example.retrieval,
    as_of: example.asOf,
    model: example.model ?? "",
    latency_ms: example.latencyMs ?? Number.NaN,
  };
  return {
    status: "done",
    request,
    stage: "generating",
    text: example.answer,
    citations: example.citations.map((c, i) => ({ n: i + 1, citation: c })),
    result,
    failure: null,
    run,
  };
}

/** The example to show for an `example` URL parameter: that one if it exists, else the first. */
export function pickRecorded(examples: readonly RecordedAnswer[], id: string | null): RecordedAnswer | null {
  return (id ? examples.find((e) => e.id === id) : undefined) ?? examples[0] ?? null;
}

/** Why the recorded answers could not be shown. */
export function describeRecordedError(error: unknown): StanceFailure {
  if (error instanceof ApiError && (error.status === 404 || (error.status === 503 && isApiErrorBody(error.body)))) {
    return {
      tone: "setup",
      title: "Recorded answers are not available",
      message:
        error.status === 404
          ? "This backend does not serve recorded answers (GET /api/ask/examples); update it to the current version."
          : error.detail,
    };
  }
  return describeStanceError(error);
}
