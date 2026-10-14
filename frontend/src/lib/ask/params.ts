/**
 * Ask the Fed form state: defaults and limits, the shareable URL query string (q, as_of, method,
 * k), client-side validation, and the mapping of a 422 response onto form fields.
 */
import type { AskRequest, RetrievalMethod, ValidationErrorItem } from "../api/types";

export const QUESTION_MIN = 3;
export const QUESTION_MAX = 2000;
/** Date of the first FOMC document in the corpus: no earlier cutoff can find anything. */
export const AS_OF_MIN = "2015-01-28";
export const TOP_K_MIN = 3;
export const TOP_K_MAX = 10;
export const TOP_K_DEFAULT = 5;
export const METHOD_DEFAULT: RetrievalMethod = "hybrid_rerank";

export const RETRIEVAL_METHODS = [
  { value: "hybrid_rerank", label: "Hybrid + rerank (default)", short: "Hybrid + rerank" },
  { value: "hybrid", label: "Hybrid", short: "Hybrid" },
  { value: "dense", label: "Dense", short: "Dense" },
  { value: "bm25", label: "BM25", short: "BM25" },
] as const satisfies readonly { value: RetrievalMethod; label: string; short: string }[];

export function methodLabel(method: RetrievalMethod): string {
  return RETRIEVAL_METHODS.find((m) => m.value === method)?.short ?? method;
}

export interface AskFormValues {
  question: string;
  /** "YYYY-MM-DD", or "" for no cutoff. */
  asOf: string;
  method: RetrievalMethod;
  topK: number;
}

export const EMPTY_FORM: AskFormValues = {
  question: "",
  asOf: "",
  method: METHOD_DEFAULT,
  topK: TOP_K_DEFAULT,
};

export type AskField = "question" | "asOf" | "method" | "topK";
/** Messages per field; `form` holds errors that belong to no single field. */
export type FieldErrors = Partial<Record<AskField | "form", string>>;

/** Anything with URLSearchParams' `get`, such as Next's ReadonlyURLSearchParams. */
interface ParamSource {
  get(name: string): string | null;
}

const ISO_DATE = /^(\d{4})-(\d{2})-(\d{2})$/;

/** True for a real calendar date written as YYYY-MM-DD. */
export function isIsoDate(value: string): boolean {
  const match = ISO_DATE.exec(value);
  if (!match) return false;
  const [year, month, day] = match.slice(1).map(Number);
  const date = new Date(Date.UTC(year, month - 1, day));
  return (
    date.getUTCFullYear() === year && date.getUTCMonth() === month - 1 && date.getUTCDate() === day
  );
}

function isMethod(value: string | null): value is RetrievalMethod {
  return RETRIEVAL_METHODS.some((m) => m.value === value);
}

/**
 * Form values from a query string. Unknown or out-of-range values fall back to the defaults, so
 * a hand-edited URL never breaks the form; the question is kept as written.
 */
export function parseAskParams(params: ParamSource): AskFormValues {
  const question = (params.get("q") ?? "").slice(0, QUESTION_MAX * 2);
  const asOf = params.get("as_of") ?? "";
  const method = params.get("method");
  const k = params.get("k");
  const topK = k !== null && /^\d{1,2}$/.test(k) ? Number(k) : NaN;
  return {
    question,
    asOf: isIsoDate(asOf) ? asOf : "",
    method: isMethod(method) ? method : METHOD_DEFAULT,
    topK: topK >= TOP_K_MIN && topK <= TOP_K_MAX ? topK : TOP_K_DEFAULT,
  };
}

/** The query string (without "?") for form values; defaults and blanks are left out. */
export function serializeAskParams(values: AskFormValues): string {
  const params = new URLSearchParams();
  const question = values.question.trim();
  if (question) params.set("q", question);
  if (values.asOf) params.set("as_of", values.asOf);
  if (values.method !== METHOD_DEFAULT) params.set("method", values.method);
  if (values.topK !== TOP_K_DEFAULT) params.set("k", String(values.topK));
  return params.toString();
}

/** True when the advanced options differ from their defaults. */
export function hasAdvancedOverrides(values: AskFormValues): boolean {
  return values.method !== METHOD_DEFAULT || values.topK !== TOP_K_DEFAULT;
}

export type ValidationResult =
  | { ok: true; request: AskRequest }
  | { ok: false; errors: FieldErrors };

/**
 * Check the form against the API contract before sending it. `today` (YYYY-MM-DD, local time) is
 * the latest allowed cutoff; null skips that bound (not known yet during hydration).
 */
export function validateAskForm(values: AskFormValues, today: string | null): ValidationResult {
  const errors: FieldErrors = {};
  const question = values.question.trim();
  if (question.length < QUESTION_MIN) {
    errors.question =
      question.length === 0
        ? "Enter a question."
        : `Enter at least ${QUESTION_MIN} characters.`;
  } else if (question.length > QUESTION_MAX) {
    errors.question = `Shorten the question to ${QUESTION_MAX.toLocaleString("en-US")} characters or fewer (it has ${question.length.toLocaleString("en-US")}).`;
  }

  if (values.asOf) {
    if (!isIsoDate(values.asOf)) {
      errors.asOf = "Enter a valid date, or clear it for no cutoff.";
    } else if (values.asOf < AS_OF_MIN) {
      errors.asOf = "Choose January 28, 2015 or later: the documents start then.";
    } else if (today && values.asOf > today) {
      errors.asOf = "Choose today or an earlier date.";
    }
  }

  if (!isMethod(values.method)) errors.method = "Choose a retrieval method.";
  if (!Number.isInteger(values.topK) || values.topK < TOP_K_MIN || values.topK > TOP_K_MAX) {
    errors.topK = `Choose between ${TOP_K_MIN} and ${TOP_K_MAX} passages.`;
  }

  if (Object.keys(errors).length > 0) return { ok: false, errors };
  return {
    ok: true,
    request: {
      question,
      as_of: values.asOf || null,
      retrieval: values.method,
      top_k: values.topK,
    },
  };
}

const FIELD_BY_API_NAME: Record<string, AskField> = {
  question: "question",
  as_of: "asOf",
  retrieval: "method",
  top_k: "topK",
};

/** Form errors from a 422 body ({"detail": [{loc, msg}, ...]}); anything unmapped goes to `form`. */
export function fieldErrorsFromValidation(body: unknown): FieldErrors {
  const errors: FieldErrors = {};
  const detail =
    typeof body === "object" && body !== null && "detail" in body
      ? (body as { detail: unknown }).detail
      : undefined;
  if (!Array.isArray(detail)) {
    errors.form = typeof detail === "string" && detail ? detail : "The request was not valid.";
    return errors;
  }
  for (const item of detail as Partial<ValidationErrorItem>[]) {
    const loc = Array.isArray(item.loc) ? item.loc : [];
    const name = loc.find((part) => part !== "body");
    const field = typeof name === "string" ? FIELD_BY_API_NAME[name] : undefined;
    const key = field ?? "form";
    const message = typeof item.msg === "string" && item.msg ? item.msg : "Invalid value.";
    if (!errors[key]) errors[key] = message;
  }
  return errors;
}
