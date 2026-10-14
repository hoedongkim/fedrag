import type { ApiErrorBody, HealthResponse, ValidationErrorItem } from "./types";

const DEFAULT_BACKEND_URL = "http://127.0.0.1:8000";

/** Window event that asks BackendStatus to re-check /api/health now, e.g. after a failed request. */
export const HEALTH_RECHECK_EVENT = "fedrag:health-recheck";

/**
 * Browser: "" so requests stay relative ("/api/...") and go through the
 * Next.js rewrite to FastAPI. Server: the absolute backend URL, since a
 * server component has no origin to resolve a relative URL against.
 */
export function baseUrl(): string {
  if (typeof window !== "undefined") return "";
  return (process.env.BACKEND_URL || DEFAULT_BACKEND_URL).replace(/\/+$/, "");
}

/** A non-2xx response from the API, carrying FastAPI's `detail` message. */
export class ApiError extends Error {
  readonly status: number;
  readonly detail: string;
  /** Parsed response body, when there was one (e.g. 422 field errors). */
  readonly body: unknown;

  constructor(status: number, detail: string, body?: unknown) {
    super(`API ${status}: ${detail}`);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
    this.body = body;
  }
}

function formatValidationErrors(items: ValidationErrorItem[]): string {
  return items
    .map((item) => {
      const field = item.loc.filter((part) => part !== "body").join(".");
      return field ? `${field}: ${item.msg}` : item.msg;
    })
    .join("; ");
}

function detailFrom(body: unknown, fallback: string): string {
  if (body && typeof body === "object" && "detail" in body) {
    const detail = (body as ApiErrorBody).detail;
    if (typeof detail === "string" && detail) return detail;
    if (Array.isArray(detail) && detail.length > 0) {
      return formatValidationErrors(detail);
    }
  }
  if (typeof body === "string" && body.trim()) return body.trim();
  return fallback;
}

async function readBody(response: Response): Promise<unknown> {
  const text = await response.text();
  if (!text) return undefined;
  const contentType = response.headers.get("content-type") ?? "";
  if (!contentType.includes("json")) return text;
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

/** The ApiError for a non-2xx response, with FastAPI's `detail` message when it sent one. */
export async function errorFromResponse(response: Response): Promise<ApiError> {
  const payload = await readBody(response).catch(() => undefined);
  const fallback = response.statusText || `Request failed with status ${response.status}`;
  return new ApiError(response.status, detailFrom(payload, fallback), payload);
}

export interface ApiFetchInit extends RequestInit {
  /** Serialized with JSON.stringify and sent as the request body. */
  json?: unknown;
}

/**
 * Typed fetch for the FedRAG API. `path` starts with "/api/". Sends and
 * expects JSON; throws ApiError on any non-2xx response.
 */
export async function apiFetch<T>(path: string, init: ApiFetchInit = {}): Promise<T> {
  const { json, headers: initHeaders, ...rest } = init;
  const headers = new Headers(initHeaders);
  if (!headers.has("Accept")) headers.set("Accept", "application/json");

  let body = rest.body;
  if (json !== undefined) {
    body = JSON.stringify(json);
    if (!headers.has("Content-Type")) headers.set("Content-Type", "application/json");
  }

  const response = await fetch(`${baseUrl()}${path}`, { ...rest, headers, body });
  if (!response.ok) throw await errorFromResponse(response);

  const payload = await readBody(response);
  if (typeof payload === "string") {
    throw new ApiError(response.status, "Expected a JSON response from the API.", payload);
  }
  return payload as T;
}

/** GET /api/health */
export function getHealth(init?: ApiFetchInit): Promise<HealthResponse> {
  return apiFetch<HealthResponse>("/api/health", { ...init, method: "GET" });
}
