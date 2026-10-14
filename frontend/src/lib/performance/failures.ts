/** Why the evaluation could not be shown, and how to fix it. */
import { ApiError } from "../api/client";
import { describeStanceError, type StanceFailure } from "../stance/failures";

export type EvaluationFailure = StanceFailure;

/** The failure for an error thrown while loading the report or one question. */
export function describeEvaluationError(error: unknown): EvaluationFailure {
  if (error instanceof ApiError) {
    if (error.status === 503 && /make rag-eval|not run/i.test(error.detail)) {
      return {
        tone: "setup",
        title: "Evaluation not run yet",
        message:
          "This tab shows the results of the RAG evaluation, which are written to data/eval/rag_benchmark.json by:",
        command: "make rag-eval",
      };
    }
    if (error.status === 503 && /invalid/i.test(error.detail)) {
      return { tone: "error", title: "Evaluation file is invalid", message: error.detail };
    }
    if (error.status === 404) {
      return { tone: "error", title: "Question not found", message: error.detail };
    }
  }
  // Unreachable backend, proxy 500s and other errors read as on the other tabs.
  return describeStanceError(error);
}
