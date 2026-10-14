import { type SseMessage, readSseStream } from "../sse";
import { ApiError, baseUrl, errorFromResponse } from "./client";
import type { AskRequest, AskResponse, AskStreamEvent, Citation } from "./types";

/** The stream ended without a `done` or `error` event, e.g. the connection dropped mid-answer. */
export class StreamInterruptedError extends Error {
  constructor() {
    super("The connection closed before the answer was complete.");
    this.name = "StreamInterruptedError";
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * The typed event for one SSE message of POST /api/ask/stream, or null for an unknown event name
 * or a payload that is not what the contract says. An `error` event is always kept, so a
 * malformed one still ends the answer as a failure.
 */
export function parseAskStreamEvent(message: SseMessage): AskStreamEvent | null {
  let data: unknown;
  try {
    data = JSON.parse(message.data);
  } catch {
    data = undefined;
  }
  const payload = isRecord(data) ? data : {};

  switch (message.event) {
    case "status": {
      const { stage } = payload;
      return stage === "retrieving" || stage === "generating"
        ? { event: "status", data: { stage } }
        : null;
    }
    case "delta":
      return typeof payload.text === "string" ? { event: "delta", data: { text: payload.text } } : null;
    case "citation": {
      const { n, citation } = payload;
      return typeof n === "number" && Number.isInteger(n) && n >= 1 && isRecord(citation)
        ? { event: "citation", data: { n, citation: citation as unknown as Citation } }
        : null;
    }
    case "done": {
      if (typeof payload.answer !== "string" || !Array.isArray(payload.citations)) return null;
      const notes = Array.isArray(payload.notes)
        ? payload.notes.filter((note): note is string => typeof note === "string")
        : [];
      return { event: "done", data: { ...(payload as unknown as AskResponse), notes } };
    }
    case "error": {
      const status = typeof payload.status === "number" ? payload.status : 500;
      const detail =
        typeof payload.detail === "string" && payload.detail.trim()
          ? payload.detail
          : "The answer could not be completed.";
      return { event: "error", data: { status, detail } };
    }
    default:
      return null;
  }
}

export interface StreamAskOptions {
  /** Aborting it cancels the request; the backend then stops generating. */
  signal?: AbortSignal;
  /** Called for each event, in order. */
  onEvent: (event: AskStreamEvent) => void;
}

/**
 * POST /api/ask/stream. Errors decided before the stream starts (422, 503) reject with ApiError,
 * like apiFetch. Once streaming, every event goes to `onEvent`, and the promise resolves after a
 * `done` or `error` event. It rejects with StreamInterruptedError if the stream ends, or reading it
 * fails (the connection dropped), before either, and with the abort error if the request is
 * aborted. The fetch error (TypeError) is thrown as is only when there is no response at all.
 */
export async function streamAsk(request: AskRequest, { signal, onEvent }: StreamAskOptions) {
  const response = await fetch(`${baseUrl()}/api/ask/stream`, {
    method: "POST",
    headers: { Accept: "text/event-stream", "Content-Type": "application/json" },
    body: JSON.stringify(request),
    cache: "no-store",
    signal,
  });
  if (!response.ok) throw await errorFromResponse(response);

  const contentType = response.headers.get("content-type") ?? "";
  if (!response.body || !contentType.includes("text/event-stream")) {
    throw new ApiError(response.status, "Expected an event stream from the API.");
  }

  let finished = false;
  try {
    await readSseStream(response.body, (message) => {
      if (finished) return;
      const event = parseAskStreamEvent(message);
      if (!event) return;
      if (event.event === "done" || event.event === "error") finished = true;
      onEvent(event);
    });
  } catch (error) {
    if (signal?.aborted) throw error;
    // After `done` or `error` the answer is complete: a failed read changes nothing.
    if (!finished) throw new StreamInterruptedError();
  }
  if (!finished) throw new StreamInterruptedError();
}
