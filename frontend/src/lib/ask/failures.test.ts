import { describe, expect, it } from "vitest";

import { ApiError } from "../api/client";
import { StreamInterruptedError } from "../api/stream";
import { describeFailure, failureFromError } from "./failures";

// The backend's actual 503 details (app/rag/answer.py, app/rag/index.py, app/rag/retrievers.py).
const NO_KEY = "LLM not configured: set OPENAI_API_KEY in backend/.env.";
const NO_INDEX = "Search index not built: run make index.";
const STALE_INDEX =
  "The index was embedded with text-embedding-3-large, but EMBEDDING_MODEL is text-embedding-3-small: run make index with --rebuild.";
const NO_RERANKER =
  "Reranker BAAI/bge-reranker-base is not downloaded and could not be fetched from Hugging Face: run once with network access, or use retrieval bm25, dense or hybrid.";

describe("describeFailure", () => {
  it("explains each 503 cause with what to do", () => {
    expect(describeFailure({ kind: "http", status: 503, detail: NO_KEY })).toMatchObject({
      tone: "setup",
      title: "OpenAI API key not set",
      action: "retry",
    });
    const noIndex = describeFailure({ kind: "http", status: 503, detail: NO_INDEX });
    expect(noIndex).toMatchObject({ title: "Search index not built", command: "make index" });
    expect(noIndex.detail).toBeUndefined(); // the default message adds nothing to the title
    expect(describeFailure({ kind: "http", status: 503, detail: STALE_INDEX })).toMatchObject({
      title: "Search index not built",
      detail: STALE_INDEX,
    });
    expect(describeFailure({ kind: "http", status: 503, detail: NO_RERANKER })).toMatchObject({
      title: "Reranker unavailable",
      action: "retry-hybrid",
    });
  });

  it("treats 502 the same before and during the stream", () => {
    const detail = "OpenAI request failed (RateLimitError). Try again shortly.";
    const before = describeFailure({ kind: "http", status: 502, detail });
    const during = describeFailure({ kind: "stream", status: 502, detail });
    expect(before).toEqual(during);
    expect(before).toMatchObject({ tone: "error", detail, action: "retry" });
    expect(before.message).toMatch(/try again in a moment/);
  });

  it("says when the OpenAI account has no credit left, instead of suggesting a retry later", () => {
    // The backend's 502 detail for a 429 with code insufficient_quota or credit_balance_exhausted.
    const detail =
      "OpenAI request failed (RateLimitError). The OpenAI account behind this server has no credit left; add credit at platform.openai.com/billing.";
    for (const kind of ["http", "stream"] as const) {
      const described = describeFailure({ kind, status: 502, detail });
      expect(described).toEqual({
        tone: "setup",
        title: "OpenAI account out of credit",
        message:
          "The OpenAI account behind this server has no credit left; add credit at platform.openai.com/billing.",
        action: "retry",
      });
      expect(described.message).not.toMatch(/try again in a moment/);
    }
    // Any other 429 is still a transient rate limit.
    const rateLimited = describeFailure({
      kind: "stream",
      status: 502,
      detail: "OpenAI request failed (RateLimitError). Try again shortly.",
    });
    expect(rateLimited).toMatchObject({ tone: "error", title: "The language model request failed" });
  });

  it("covers network failures, cut-off streams and other statuses", () => {
    expect(describeFailure({ kind: "network" })).toMatchObject({ command: "make dev-backend" });
    expect(describeFailure({ kind: "interrupted" }).title).toBe("The answer was cut off");
    expect(describeFailure({ kind: "http", status: 500, detail: "Internal Server Error" })).toMatchObject({
      title: "The API returned an error (HTTP 500)",
      command: "make dev-backend",
    });
    expect(describeFailure({ kind: "stream", status: 500, detail: "boom" })).toMatchObject({
      title: "The answer failed",
      detail: "boom",
    });
  });
});

describe("failureFromError", () => {
  it("classifies what streamAsk throws", () => {
    expect(failureFromError(new ApiError(503, NO_INDEX, { detail: NO_INDEX }))).toEqual({
      kind: "http",
      status: 503,
      detail: NO_INDEX,
      body: { detail: NO_INDEX },
    });
    expect(failureFromError(new StreamInterruptedError())).toEqual({ kind: "interrupted" });
    expect(failureFromError(new TypeError("Failed to fetch"))).toEqual({ kind: "network" });
  });
});
