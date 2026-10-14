import { describe, expect, it } from "vitest";

import { type SseMessage, createSseParser, readSseStream } from "./sse";

function parseChunks(chunks: string[]): SseMessage[] {
  const messages: SseMessage[] = [];
  const parser = createSseParser((message) => messages.push(message));
  for (const chunk of chunks) parser.push(chunk);
  parser.end();
  return messages;
}

const STREAM =
  ": stream open\n\n" +
  'event: status\ndata: {"stage": "retrieving"}\n\n' +
  'event: delta\ndata: {"text": "Rates rose [1]."}\n\n' +
  'event: done\ndata: {"answer": "Rates rose [1]."}\n\n';

describe("createSseParser", () => {
  it("parses events with names and data", () => {
    expect(parseChunks([STREAM])).toEqual([
      { event: "status", data: '{"stage": "retrieving"}', id: "" },
      { event: "delta", data: '{"text": "Rates rose [1]."}', id: "" },
      { event: "done", data: '{"answer": "Rates rose [1]."}', id: "" },
    ]);
  });

  it("gives the same events however the stream is split into chunks", () => {
    const whole = parseChunks([STREAM]);
    for (let cut = 1; cut < STREAM.length; cut++) {
      expect(parseChunks([STREAM.slice(0, cut), STREAM.slice(cut)])).toEqual(whole);
    }
    expect(parseChunks([...STREAM])).toEqual(whole); // one character at a time
  });

  it("accepts CRLF and bare CR line endings", () => {
    const crlf = STREAM.replaceAll("\n", "\r\n");
    const cr = STREAM.replaceAll("\n", "\r");
    const whole = parseChunks([STREAM]);
    expect(parseChunks([crlf])).toEqual(whole);
    expect(parseChunks([cr])).toEqual(whole);
    for (let cut = 1; cut < crlf.length; cut++) {
      expect(parseChunks([crlf.slice(0, cut), crlf.slice(cut)])).toEqual(whole);
    }
  });

  it("treats a CRLF split between chunks as one line ending", () => {
    // A LF opening the next chunk must not count as a blank line that dispatches early.
    expect(parseChunks(["data: a\r", "\ndata: b\r", "\n\r", "\n"])).toEqual([
      { event: "message", data: "a\nb", id: "" },
    ]);
  });

  it("joins multi-line data with newlines and keeps empty data lines", () => {
    expect(parseChunks(["data: first\ndata:\ndata: third\n\n"])).toEqual([
      { event: "message", data: "first\n\nthird", id: "" },
    ]);
  });

  it("strips exactly one space after the colon", () => {
    expect(parseChunks(["data:x\n\n", "data:  two spaces\n\n"]).map((m) => m.data)).toEqual([
      "x",
      " two spaces",
    ]);
  });

  it("ignores comments, unknown fields and retry, and resets the event name", () => {
    const messages = parseChunks([
      ": ping\nretry: 1000\nfoo: bar\nevent: delta\ndata: one\n\n",
      "data: two\n\n",
    ]);
    expect(messages).toEqual([
      { event: "delta", data: "one", id: "" },
      { event: "message", data: "two", id: "" },
    ]);
  });

  it("does not dispatch an event without data", () => {
    expect(parseChunks(["event: status\n\n", "\n\n"])).toEqual([]);
  });

  it("treats a field name without a colon as an empty value", () => {
    expect(parseChunks(["data\n\n"])).toEqual([{ event: "message", data: "", id: "" }]);
  });

  it("tracks the last event id", () => {
    expect(parseChunks(["id: 7\ndata: a\n\ndata: b\n\n"]).map((m) => m.id)).toEqual(["7", "7"]);
  });

  it("discards an unterminated event at the end of the stream", () => {
    expect(parseChunks(["data: complete\n\n", "event: done\ndata: cut off"])).toEqual([
      { event: "message", data: "complete", id: "" },
    ]);
  });

  it("strips a leading byte order mark", () => {
    expect(parseChunks(["﻿data: x\n\n"])).toEqual([{ event: "message", data: "x", id: "" }]);
  });
});

function streamOf(chunks: Uint8Array[]): ReadableStream<Uint8Array> {
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(chunk);
      controller.close();
    },
  });
}

describe("readSseStream", () => {
  it("decodes UTF-8 characters split across chunks", async () => {
    const bytes = new TextEncoder().encode('data: {"text": "rates — 5-1/4 %"}\n\n');
    const dash = bytes.indexOf(0xe2); // first byte of the em dash
    const messages: SseMessage[] = [];
    await readSseStream(
      streamOf([bytes.slice(0, dash + 1), bytes.slice(dash + 1, dash + 2), bytes.slice(dash + 2)]),
      (message) => messages.push(message),
    );
    expect(messages).toEqual([{ event: "message", data: '{"text": "rates — 5-1/4 %"}', id: "" }]);
  });

  it("rejects when the stream errors", async () => {
    const failing = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode("data: a\n\n"));
        controller.error(new DOMException("aborted", "AbortError"));
      },
    });
    const messages: SseMessage[] = [];
    await expect(readSseStream(failing, (m) => messages.push(m))).rejects.toThrow("aborted");
  });
});
