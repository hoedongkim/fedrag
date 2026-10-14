import { afterEach, describe, expect, it } from "vitest";

import { describeFailure } from "../ask/failures";
import { analysisStatusNotice, describeCompareError, describeGenerateError } from "../compare/failures";
import { describeEvaluationError } from "../performance/failures";
import { describeStanceError } from "../stance/failures";
import { ApiError } from "./client";
import { backendMaySleep, DEMO_UNREACHABLE } from "./hosting";

const asleep = new ApiError(503, "<html>Space is starting</html>", "<html>Space is starting</html>");
const proxy500 = new ApiError(500, "Internal Server Error", "Internal Server Error");
const refused = new TypeError("fetch failed");
const dbDetail = "Database not configured: set DATABASE_URL in backend/.env.";
const noDb = new ApiError(503, dbDetail, { detail: dbDetail });

function hosted(on: boolean) {
  if (on) process.env.FEDRAG_BACKEND_MAY_SLEEP = "true";
  else delete process.env.FEDRAG_BACKEND_MAY_SLEEP;
}

afterEach(() => hosted(false));

describe("backendMaySleep", () => {
  it("follows the build-time flag", () => {
    expect(backendMaySleep()).toBe(false);
    hosted(true);
    expect(backendMaySleep()).toBe(true);
  });
});

describe("failures on the hosted demo", () => {
  it("say the demo server is not responding instead of how to start a local backend", () => {
    hosted(true);
    for (const error of [asleep, proxy500, refused]) {
      expect(describeStanceError(error)).toEqual(DEMO_UNREACHABLE);
      expect(describeEvaluationError(error)).toEqual(DEMO_UNREACHABLE);
      expect(describeCompareError(error)).toEqual({ kind: "notice", notice: DEMO_UNREACHABLE });
      expect(describeGenerateError(error)).toEqual(DEMO_UNREACHABLE);
    }
    expect(describeFailure({ kind: "network" })).toMatchObject({ title: DEMO_UNREACHABLE.title, action: "retry" });
    expect(describeFailure({ kind: "http", status: 503, detail: asleep.detail, body: asleep.body })).toMatchObject({
      title: DEMO_UNREACHABLE.title,
    });
    expect(JSON.stringify(describeStanceError(refused))).not.toContain("make dev-backend");
  });

  it("still explain errors the API answered itself", () => {
    hosted(true);
    expect(describeStanceError(noDb)).toMatchObject({ title: "Database not configured" });
    expect(describeCompareError(noDb)).toMatchObject({ notice: { title: "Database not configured" } });
  });

  it("are unchanged for a local backend", () => {
    expect(describeStanceError(refused)).toMatchObject({ title: "Backend unreachable", command: "make dev-backend" });
    expect(describeFailure({ kind: "network" })).toMatchObject({ command: "make dev-backend" });
  });
});

describe("the public demo's 403s", () => {
  const detail =
    "Ask the Fed is disabled on the public demo to avoid API costs. Run FedRAG locally to ask your own questions.";

  it("explain that live questions are off, without a pointless Retry", () => {
    expect(describeFailure({ kind: "http", status: 403, detail, body: { detail } })).toEqual({
      tone: "setup",
      title: "Live questions are off on this demo",
      message: detail,
      action: "none",
    });
  });

  it("explain that analyses are not written on the demo", () => {
    expect(describeGenerateError(new ApiError(403, "Disabled.", { detail: "Disabled." }))).toMatchObject({
      tone: "setup",
      message: "Disabled.",
    });
    expect(analysisStatusNotice("disabled_on_demo")).toMatchObject({ title: "Analyses are not written on this demo" });
    expect(analysisStatusNotice("not_cached")).toBeNull();
  });
});
