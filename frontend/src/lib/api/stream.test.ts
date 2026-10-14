import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "./client";
import { StreamInterruptedError, parseAskStreamEvent, streamAsk } from "./stream";
import type { AskStreamEvent } from "./types";

const CITATION = {
  document: {
    doc_id: "minutes-2023-03-22",
    doc_type: "minutes",
    meeting_date: "2023-03-22",
    release_date: "2023-04-12",
    page_updated: "2023-04-12",
    title: "Minutes of the FOMC Meeting, March 21-22, 2023",
    url: "https://www.federalreserve.gov/monetarypolicy/fomcminutes20230322.htm",
  },
  chunk_id: "minutes-2023-03-22#0014",
  section: "Participants' Views on Current Conditions and the Economic Outlook",
  snippet: "They commented that recent developments in the banking sector were likely to result in tighter credit conditions.",
  score: 0.91,
};

const DONE = {
  answer: "Credit conditions were expected to tighten [1].",
  citations: [CITATION],
  notes: [],
  retrieval: "hybrid_rerank",
  as_of: "2023-04-30",
  model: "gpt-5.4-mini-2026-03-17",
  latency_ms: 4210.5,
};

describe("parseAskStreamEvent", () => {
  const parse = (event: string, data: unknown) =>
    parseAskStreamEvent({ event, data: typeof data === "string" ? data : JSON.stringify(data), id: "" });

  it("types each contract event", () => {
    expect(parse("status", { stage: "generating" })).toEqual({
      event: "status",
      data: { stage: "generating" },
    });
    expect(parse("delta", { text: "Rates [1]" })).toEqual({ event: "delta", data: { text: "Rates [1]" } });
    expect(parse("citation", { n: 1, citation: CITATION })).toEqual({
      event: "citation",
      data: { n: 1, citation: CITATION },
    });
    expect(parse("done", DONE)).toEqual({ event: "done", data: DONE });
    expect(parse("error", { status: 502, detail: "OpenAI request failed (RateLimitError)." })).toEqual({
      event: "error",
      data: { status: 502, detail: "OpenAI request failed (RateLimitError)." },
    });
  });

  it("drops unknown events and payloads that break the contract", () => {
    expect(parse("message", { text: "x" })).toBeNull();
    expect(parse("status", { stage: "thinking" })).toBeNull();
    expect(parse("delta", { text: 3 })).toBeNull();
    expect(parse("delta", "not json")).toBeNull();
    expect(parse("citation", { n: 0, citation: CITATION })).toBeNull();
    expect(parse("citation", { n: 1.5, citation: CITATION })).toBeNull();
    expect(parse("done", { answer: "x" })).toBeNull();
  });

  it("defaults notes on done and keeps a malformed error as a failure", () => {
    const withoutNotes: Partial<typeof DONE> = { ...DONE };
    delete withoutNotes.notes;
    expect(parse("done", withoutNotes)).toEqual({ event: "done", data: { ...withoutNotes, notes: [] } });
    expect(parse("error", "{}")).toEqual({
      event: "error",
      data: { status: 500, detail: "The answer could not be completed." },
    });
  });
});

function sseResponse(body: string, chunkSize = 7): Response {
  const bytes = new TextEncoder().encode(body);
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (let i = 0; i < bytes.length; i += chunkSize) controller.enqueue(bytes.slice(i, i + chunkSize));
      controller.close();
    },
  });
  return new Response(stream, { status: 200, headers: { "Content-Type": "text/event-stream" } });
}

function sse(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

const REQUEST = { question: "How did the FOMC view banking stress?", top_k: 5 };

describe("streamAsk", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.unstubAllEnvs();
  });

  it("posts the request and delivers events in order", async () => {
    vi.stubEnv("BACKEND_URL", ""); // baseUrl() then falls back to its default
    const fetchMock = vi.fn(async () =>
      sseResponse(
        ": open\n\n" +
          sse("status", { stage: "retrieving" }) +
          sse("status", { stage: "generating" }) +
          sse("citation", { n: 1, citation: CITATION }) +
          sse("delta", { text: "Credit conditions were expected to tighten [1]." }) +
          sse("done", DONE),
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    const events: AskStreamEvent[] = [];
    await streamAsk(REQUEST, { onEvent: (event) => events.push(event) });

    expect(events.map((e) => e.event)).toEqual(["status", "status", "citation", "delta", "done"]);
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("http://127.0.0.1:8000/api/ask/stream"); // server-side base URL under Node
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body as string)).toEqual(REQUEST);
  });

  it("resolves after an error event", async () => {
    vi.stubGlobal("fetch", async () =>
      sseResponse(sse("delta", { text: "Partial" }) + sse("error", { status: 502, detail: "OpenAI request failed." })),
    );
    const events: AskStreamEvent[] = [];
    await streamAsk(REQUEST, { onEvent: (event) => events.push(event) });
    expect(events.at(-1)).toEqual({ event: "error", data: { status: 502, detail: "OpenAI request failed." } });
  });

  it("rejects with StreamInterruptedError when the stream ends without done", async () => {
    vi.stubGlobal("fetch", async () => sseResponse(sse("delta", { text: "Partial" })));
    await expect(streamAsk(REQUEST, { onEvent: () => {} })).rejects.toBeInstanceOf(StreamInterruptedError);
  });

  it("rejects with StreamInterruptedError when the connection drops mid-answer", async () => {
    let pulls = 0;
    const body = new ReadableStream<Uint8Array>({
      // One delta, then the read fails as when undici loses the connection.
      pull(controller) {
        if (pulls++ === 0) controller.enqueue(new TextEncoder().encode(sse("delta", { text: "Partial" })));
        else controller.error(new TypeError("terminated"));
      },
    });
    vi.stubGlobal(
      "fetch",
      async () => new Response(body, { status: 200, headers: { "Content-Type": "text/event-stream" } }),
    );
    const events: AskStreamEvent[] = [];
    const error = await streamAsk(REQUEST, { onEvent: (event) => events.push(event) }).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(StreamInterruptedError);
    expect(events).toEqual([{ event: "delta", data: { text: "Partial" } }]);
  });

  it("rethrows the abort error when the request is aborted mid-answer", async () => {
    const controller = new AbortController();
    const body = new ReadableStream<Uint8Array>({
      start(stream) {
        stream.enqueue(new TextEncoder().encode(sse("delta", { text: "Partial" })));
        controller.signal.addEventListener("abort", () => stream.error(controller.signal.reason));
      },
    });
    vi.stubGlobal(
      "fetch",
      async () => new Response(body, { status: 200, headers: { "Content-Type": "text/event-stream" } }),
    );
    const error = await streamAsk(REQUEST, { signal: controller.signal, onEvent: () => controller.abort() }).catch(
      (e: unknown) => e,
    );
    expect(error).not.toBeInstanceOf(StreamInterruptedError);
    expect(error).toMatchObject({ name: "AbortError" });
  });

  it("rejects with ApiError carrying the detail for an error before the stream", async () => {
    const detail = "Search index not built: run make index.";
    vi.stubGlobal(
      "fetch",
      async () =>
        new Response(JSON.stringify({ detail }), {
          status: 503,
          headers: { "Content-Type": "application/json" },
        }),
    );
    const error = await streamAsk(REQUEST, { onEvent: () => {} }).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({ status: 503, detail });
  });

  it("rejects a 200 response that is not an event stream", async () => {
    vi.stubGlobal(
      "fetch",
      async () => new Response("{}", { status: 200, headers: { "Content-Type": "application/json" } }),
    );
    await expect(streamAsk(REQUEST, { onEvent: () => {} })).rejects.toBeInstanceOf(ApiError);
  });
});
