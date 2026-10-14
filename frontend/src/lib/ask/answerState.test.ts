import { describe, expect, it } from "vitest";

import type { AskResponse, AskStreamEvent, Citation } from "../api/types";
import { type AskAction, type AskState, INITIAL_ASK_STATE, askReducer } from "./answerState";

const citation = (id: string): Citation => ({
  document: {
    doc_id: id,
    doc_type: "statement",
    meeting_date: "2024-09-18",
    release_date: "2024-09-18",
    page_updated: "2024-09-18",
    title: `FOMC Statement ${id}`,
    url: `https://www.federalreserve.gov/${id}.htm`,
  },
  chunk_id: `${id}#0000`,
  section: null,
  snippet: "The Committee decided to lower the target range.",
  passage: null,
  score: 0.9,
});

const REQUEST = { question: "Why cut in September 2024?", top_k: 5 };

function run(...events: (AskStreamEvent | AskAction)[]): AskState {
  return events.reduce<AskState>(
    (state, item) => askReducer(state, "type" in item ? item : { type: "event", event: item }),
    askReducer(INITIAL_ASK_STATE, { type: "start", request: REQUEST }),
  );
}

describe("askReducer", () => {
  it("starts streaming at the retrieval stage", () => {
    const state = run();
    expect(state).toMatchObject({ status: "streaming", stage: "retrieving", text: "", run: 1 });
    expect(state.request).toEqual(REQUEST);
  });

  it("accumulates deltas and citations", () => {
    const state = run(
      { event: "status", data: { stage: "generating" } },
      { event: "citation", data: { n: 2, citation: citation("b") } },
      { event: "citation", data: { n: 1, citation: citation("a") } },
      { event: "delta", data: { text: "Inflation eased [1]" } },
      { event: "delta", data: { text: " and jobs slowed [2]." } },
    );
    expect(state.stage).toBe("generating");
    expect(state.text).toBe("Inflation eased [1] and jobs slowed [2].");
    expect(state.citations.map((c) => c.n)).toEqual([1, 2]);
  });

  it("moves to generating on the first delta even without a status event", () => {
    expect(run({ event: "delta", data: { text: "x" } }).stage).toBe("generating");
  });

  it("replaces streamed text and citations with the final answer", () => {
    const done: AskResponse = {
      answer: "Final [1].",
      citations: [citation("final")],
      notes: ["The newest passage is old."],
      retrieval: "hybrid_rerank",
      as_of: null,
      model: "gpt-5.4-mini-2026-03-17",
      latency_ms: 3200,
    };
    const state = run(
      { event: "citation", data: { n: 1, citation: citation("streamed") } },
      { event: "delta", data: { text: "Streamed [1]" } },
      { event: "done", data: done },
    );
    expect(state.status).toBe("done");
    expect(state.text).toBe("Final [1].");
    expect(state.citations).toEqual([{ n: 1, citation: citation("final") }]);
    expect(state.result).toBe(done);
  });

  it("fails on an error event and keeps the partial text", () => {
    const state = run(
      { event: "delta", data: { text: "Partial" } },
      { event: "error", data: { status: 502, detail: "OpenAI request failed." } },
    );
    expect(state.status).toBe("failed");
    expect(state.text).toBe("Partial");
    expect(state.failure).toEqual({ kind: "stream", status: 502, detail: "OpenAI request failed." });
  });

  it("ignores events after the answer ended", () => {
    const stopped = run({ event: "delta", data: { text: "Part" } }, { type: "stop" });
    expect(stopped.status).toBe("stopped");
    expect(askReducer(stopped, { type: "event", event: { event: "delta", data: { text: "more" } } })).toBe(
      stopped,
    );
    expect(askReducer(stopped, { type: "fail", failure: { kind: "network" } })).toBe(stopped);
  });

  it("resets everything but the run counter on a new ask", () => {
    const failed = run({ type: "fail", failure: { kind: "network" } });
    const again = askReducer(failed, { type: "start", request: REQUEST });
    expect(again).toMatchObject({ status: "streaming", failure: null, text: "", run: 2 });
  });
});
