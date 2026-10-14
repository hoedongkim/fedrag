/**
 * Every API type the frontend uses lives in this one file.
 *
 * The backend owns the contract. These are aliases over `schema.d.ts`, which
 * `npm run gen:api` (or `make gen-api` from the repo root) generates from the
 * backend's exported `openapi.json`. Never edit `schema.d.ts` by hand: change
 * the Pydantic schemas in backend/app/schemas, then regenerate. CI fails if
 * the checked-in files drift from the backend.
 */
import type { components, operations } from "./schema";

type Schemas = components["schemas"];

// --- Shared ---------------------------------------------------------------

export type SourceDocument = Schemas["SourceDocument"];
export type MeetingSummary = Schemas["MeetingSummary"];

export type DocType = SourceDocument["doc_type"];
export type RetrievalMethod = Schemas["AskResponse"]["retrieval"];
export type Stance = Schemas["StanceResult"]["stance"];
export type StanceClassifier = Schemas["StanceResult"]["classifier"];

// --- GET /api/health ------------------------------------------------------

/**
 * `demo_mode` is true on the public demo, where Ask the Fed shows recorded answers and nothing
 * calls OpenAI.
 */
export type HealthResponse = Schemas["HealthResponse"];

// --- GET /api/meetings ----------------------------------------------------

export type ListMeetingsQuery = NonNullable<operations["listMeetings"]["parameters"]["query"]>;

// --- POST /api/ask, POST /api/ask/stream ------------------------------------

export type AskRequest = Schemas["AskRequest"];
export type AskResponse = Schemas["AskResponse"];
export type Citation = Schemas["Citation"];

// The data payloads of the POST /api/ask/stream events (see AskStreamEvent).
/** `event: status`: the pipeline stage that has started. */
export type StreamStatus = Schemas["StreamStatus"];
/** `event: delta`: the next piece of answer text, its [n] markers already final citation numbers. */
export type StreamDelta = Schemas["StreamDelta"];
/** `event: citation`: a passage cited for the first time; [n] in the answer refers to it. */
export type StreamCitation = Schemas["StreamCitation"];
/** `event: error`: a failure after the stream started, with an HTTP-style status. */
export type StreamError = Schemas["StreamError"];

export type StreamStage = StreamStatus["stage"];

// --- GET /api/ask/examples ------------------------------------------------

/**
 * A real answer of the production configuration, recorded during the evaluation: what Ask the
 * Fed shows on the public demo. lib/ask/recorded.ts validates the payload before using it.
 */
export type AskExample = Schemas["AskExample"];
/** The evaluation judge's verdicts on a recorded answer. */
export type ExampleJudgement = Schemas["ExampleJudgement"];

/**
 * One event of POST /api/ask/stream, by SSE event name. `done` carries the authoritative final
 * AskResponse; `error` ends the stream without one. Hand-written over the generated payload
 * types above: the generated schema types the stream's data only as the union of the payloads,
 * not which event name carries which.
 */
export type AskStreamEvent =
  | { event: "status"; data: StreamStatus }
  | { event: "delta"; data: StreamDelta }
  | { event: "citation"; data: StreamCitation }
  | { event: "done"; data: AskResponse }
  | { event: "error"; data: StreamError };

// --- GET /api/stance, GET /api/stance/{meeting_date}, GET /api/stance/benchmark

export type StanceResult = Schemas["StanceResult"];
export type StanceTimeline = Schemas["StanceTimeline"];
export type FedFundsTarget = Schemas["FedFundsTarget"];
export type ReferenceLabel = Schemas["ReferenceLabel"];
export type ReferenceSource = ReferenceLabel["source"];
export type StanceBenchmark = Schemas["StanceBenchmark"];
export type SentenceBenchmark = Schemas["SentenceBenchmark"];
export type StatementBenchmark = Schemas["StatementBenchmark"];
export type BenchmarkSplit = SentenceBenchmark["split"];
export type BenchmarkReference = StatementBenchmark["reference"];
export type StatementVariant = StatementBenchmark["variant"];
export type ConfidenceInterval = Schemas["ConfidenceInterval"];
export type StatementBootstrap = Schemas["StatementBootstrap"];
export type PairedDifference = Schemas["PairedDifference"];
export type SentenceBaseline = Schemas["SentenceBaseline"];
export type StatementBaseline = Schemas["StatementBaseline"];
export type StatementBaselineName = StatementBaseline["name"];
export type BaselineDifference = Schemas["BaselineDifference"];
export type ReferenceLabelSummary = Schemas["ReferenceLabelSummary"];
export type StanceTimelineQuery = NonNullable<
  operations["getStanceTimeline"]["parameters"]["query"]
>;
export type StanceForMeetingQuery = NonNullable<
  operations["getStanceForMeeting"]["parameters"]["query"]
>;

// --- POST /api/compare ----------------------------------------------------

export type CompareRequest = Schemas["CompareRequest"];
/** "cached_only" never calls the LLM (what opening a link sends); "generate" may call it once. */
export type AnalysisMode = NonNullable<CompareRequest["analysis"]>;
export type CompareResponse = Schemas["CompareResponse"];
/** One aligned item of the sentence diff, in document order; `id` is its index in `changes`. */
export type SentenceChange = Schemas["SentenceChange"];
export type ChangeKind = SentenceChange["kind"];
/** One run of a word-level diff: equal + delete rebuild the base sentence, equal + insert the target. */
export type DiffSegment = Schemas["DiffSegment"];
export type SegmentOp = DiffSegment["op"];
export type CompareStats = Schemas["ComparisonStats"];
/** The LLM reading of the changes, as of the target statement's release. */
export type ComparisonAnalysis = Schemas["ComparisonAnalysis"];
export type KeyChange = Schemas["KeyChange"];
/** Policy direction of one change; the same three values as a stance. */
export type ChangeDirection = KeyChange["direction"];
export type StanceShift = ComparisonAnalysis["stance_shift"];
/** "disabled_on_demo": none is stored, and the public demo does not write analyses. */
export type AnalysisStatus = CompareResponse["analysis_status"];
/** The stored stance of both statements from one classifier (llm_zero_shot), as a cross-check. */
export type CompareStance = Schemas["ComparisonStance"];
export type CompareStanceResult = Schemas["StanceSnapshot"];

// --- GET /api/evaluation, GET /api/evaluation/questions/{question_id} ---------

export type EvaluationReport = Schemas["EvaluationReport"];
/** A mean over questions with its bootstrap interval and n; value null when nothing qualifies. */
export type Estimate = Schemas["Estimate"];
export type MetricInterval = Schemas["MetricInterval"];
export type RetrieverConfig = Schemas["RetrieverConfig"];
export type RetrievalRow = Schemas["RetrievalRow"];
export type AnswerRow = Schemas["AnswerRow"];
export type TypeRow = Schemas["TypeRow"];
export type EvalDifference = Schemas["EvalDifference"];
export type EvalMetric = EvalDifference["metric"];
export type RunKey = Schemas["RunKey"];
export type ConfigSelection = Schemas["ConfigSelection"];
export type EvalDataset = Schemas["EvalDataset"];
export type JudgeInfo = Schemas["JudgeInfo"];
export type JudgeValidation = Schemas["JudgeValidation"];
export type JudgeCheck = Schemas["JudgeAgreement"]["check"];
export type GeneratorInfo = Schemas["GeneratorInfo"];
export type BootstrapInfo = Schemas["BootstrapInfo"];
export type CostSummary = Schemas["CostSummary"];
export type QuestionType = Schemas["QuestionSummary"]["type"];
export type QuestionSplit = Schemas["QuestionSummary"]["split"];
/** The split a report row is computed on: a question split or "all". */
export type ReportSplit = RetrievalRow["split"];
export type QuestionSummary = Schemas["QuestionSummary"];
export type RunSummary = Schemas["RunSummary"];
export type QuestionDetail = Schemas["QuestionDetail"];
export type RunDetail = Schemas["RunDetail"];
export type RunScores = Schemas["RunScores"];
export type RetrievedHit = Schemas["RetrievedHit"];
/** A gold chunk of a question: "primary" answers it (gain 2), "supporting" helps (gain 1). */
export type RelevantChunk = Schemas["RelevantChunk"];
export type ChunkRole = RelevantChunk["role"];
export type EvalPassage = Schemas["EvalPassage"];
export type AnswerDetail = Schemas["AnswerDetail"];
export type AnswerCitation = Schemas["AnswerCitation"];
export type JudgeVerdicts = Schemas["JudgeVerdicts"];

// --- Errors ---------------------------------------------------------------

/** Body of the 501 stubs and other HTTPException responses. */
export type ErrorResponse = Schemas["ErrorResponse"];
/** Body of a FastAPI request-validation error (HTTP 422). */
export type HTTPValidationError = Schemas["HTTPValidationError"];
/** One entry of a 422 `detail` list. */
export type ValidationErrorItem = Schemas["ValidationError"];

/**
 * Any FastAPI error body. `detail` is a message for HTTPException responses
 * (e.g. the 501 stubs) and a list of field errors for validation failures.
 */
export interface ApiErrorBody {
  detail: ErrorResponse["detail"] | ValidationErrorItem[];
}
