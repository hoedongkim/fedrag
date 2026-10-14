/**
 * A small parser for server-sent events (the text/event-stream format), for streams read with
 * fetch: EventSource cannot send a POST body. It follows the WHATWG parsing rules: lines end in
 * LF, CRLF or CR (also when a CRLF is split across chunks), consecutive `data:` lines join with
 * "\n", lines starting with ":" are comments, a blank line dispatches the event, and an
 * unterminated event at the end of the stream is discarded.
 */

export interface SseMessage {
  /** The `event:` field, or "message" when the event had none. */
  event: string;
  /** The `data:` lines joined with "\n". */
  data: string;
  /** The last `id:` seen on the stream, or "". */
  id: string;
}

export interface SseParser {
  /** Feed the next decoded piece of the stream; may dispatch any number of messages. */
  push(chunk: string): void;
  /** The stream ended: drop any unterminated event. */
  end(): void;
}

export function createSseParser(onMessage: (message: SseMessage) => void): SseParser {
  let buffer = "";
  let started = false;
  // The previous chunk ended in CR: a LF opening the next chunk belongs to the same CRLF.
  let skipLeadingLf = false;
  let eventType = "";
  let dataLines: string[] = [];
  let lastEventId = "";

  function dispatch() {
    if (dataLines.length === 0) {
      eventType = "";
      return;
    }
    const message = { event: eventType || "message", data: dataLines.join("\n"), id: lastEventId };
    eventType = "";
    dataLines = [];
    onMessage(message);
  }

  function processLine(line: string) {
    if (line === "") {
      dispatch();
      return;
    }
    if (line.startsWith(":")) return; // comment, e.g. a keep-alive ping
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") eventType = value;
    else if (field === "data") dataLines.push(value);
    else if (field === "id" && !value.includes("\0")) lastEventId = value;
    // "retry" and unknown fields are ignored.
  }

  return {
    push(chunk) {
      if (chunk === "") return;
      if (!started) {
        started = true;
        if (chunk.charCodeAt(0) === 0xfeff) chunk = chunk.slice(1); // byte order mark
      }
      if (skipLeadingLf && chunk.startsWith("\n")) chunk = chunk.slice(1);
      skipLeadingLf = false;
      buffer += chunk;

      let lineStart = 0;
      for (let i = 0; i < buffer.length; i++) {
        const char = buffer[i];
        if (char !== "\n" && char !== "\r") continue;
        processLine(buffer.slice(lineStart, i));
        if (char === "\r") {
          if (i + 1 < buffer.length) {
            if (buffer[i + 1] === "\n") i++;
          } else {
            skipLeadingLf = true;
          }
        }
        lineStart = i + 1;
      }
      buffer = buffer.slice(lineStart);
    },
    end() {
      buffer = "";
      eventType = "";
      dataLines = [];
    },
  };
}

/**
 * Read a fetch response body as server-sent events until it ends. Rejects when the stream errors,
 * e.g. with an AbortError when the request's signal aborts.
 */
export async function readSseStream(
  body: ReadableStream<Uint8Array>,
  onMessage: (message: SseMessage) => void,
): Promise<void> {
  const reader = body.getReader();
  const decoder = new TextDecoder(); // UTF-8; `stream: true` keeps split multi-byte characters
  const parser = createSseParser(onMessage);
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      parser.push(decoder.decode(value, { stream: true }));
    }
    parser.push(decoder.decode());
    parser.end();
  } finally {
    reader.releaseLock();
  }
}
