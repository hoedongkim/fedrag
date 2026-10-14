/** The state of one streamed answer, as a reducer over POST /api/ask/stream events. */
import type { AskRequest, AskResponse, AskStreamEvent, Citation, StreamStage } from "../api/types";
import type { AskFailure } from "./failures";

export interface CitationEntry {
  /** The 1-based number the answer's [n] markers use. */
  n: number;
  citation: Citation;
}

export type AskStatus = "idle" | "streaming" | "done" | "stopped" | "failed";

export interface AskState {
  status: AskStatus;
  /** The request being (or last) answered. */
  request: AskRequest | null;
  stage: StreamStage;
  /** Answer text so far; on `done`, the authoritative final answer. */
  text: string;
  /** Citations received so far, sorted by n. */
  citations: CitationEntry[];
  result: AskResponse | null;
  failure: AskFailure | null;
  /** Counts asks, so views can reset their own state for each answer. */
  run: number;
}

export type AskAction =
  | { type: "start"; request: AskRequest }
  | { type: "event"; event: AskStreamEvent }
  | { type: "fail"; failure: AskFailure }
  | { type: "stop" };

export const INITIAL_ASK_STATE: AskState = {
  status: "idle",
  request: null,
  stage: "retrieving",
  text: "",
  citations: [],
  result: null,
  failure: null,
  run: 0,
};

function withCitation(citations: CitationEntry[], entry: CitationEntry): CitationEntry[] {
  return [...citations.filter((c) => c.n !== entry.n), entry].sort((a, b) => a.n - b.n);
}

export function askReducer(state: AskState, action: AskAction): AskState {
  if (action.type === "start") {
    return {
      ...INITIAL_ASK_STATE,
      status: "streaming",
      request: action.request,
      run: state.run + 1,
    };
  }
  // Anything after the answer ended (a late event, a stop after done) changes nothing.
  if (state.status !== "streaming") return state;

  switch (action.type) {
    case "stop":
      return { ...state, status: "stopped" };
    case "fail":
      return { ...state, status: "failed", failure: action.failure };
    case "event": {
      const { event } = action;
      switch (event.event) {
        case "status":
          return { ...state, stage: event.data.stage };
        case "delta":
          return { ...state, stage: "generating", text: state.text + event.data.text };
        case "citation":
          return { ...state, citations: withCitation(state.citations, event.data) };
        case "done":
          return {
            ...state,
            status: "done",
            text: event.data.answer,
            citations: event.data.citations.map((citation, i) => ({ n: i + 1, citation })),
            result: event.data,
          };
        case "error":
          return {
            ...state,
            status: "failed",
            failure: { kind: "stream", status: event.data.status, detail: event.data.detail },
          };
      }
    }
  }
}
