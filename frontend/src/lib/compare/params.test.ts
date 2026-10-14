import { describe, expect, it } from "vitest";

import { parseAskParams } from "../ask/params";
import {
  changeQuestion,
  compareAskHref,
  compareHref,
  datesInMessage,
  EMPTY_PARAMS,
  nearestStatements,
  nextStatement,
  parseCompareParams,
  previousStatement,
  resolvePair,
  serializeCompareParams,
  shiftPair,
} from "./params";

const parse = (query: string) => parseCompareParams(new URLSearchParams(query));
const DATES = ["2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07"];

describe("parseCompareParams", () => {
  it("reads both dates", () => {
    expect(parse("base=2024-07-31&target=2024-09-18")).toEqual({
      base: "2024-07-31",
      target: "2024-09-18",
    });
  });

  it("ignores missing and invalid dates", () => {
    expect(parse("")).toEqual(EMPTY_PARAMS);
    expect(parse("base=2024-02-30&target=last")).toEqual(EMPTY_PARAMS);
    expect(parse("target=2024-09-18")).toEqual({ base: null, target: "2024-09-18" });
  });

  it("keeps an inverted pair as written, so the page can say what is wrong", () => {
    expect(parse("base=2024-09-18&target=2024-07-31")).toEqual({
      base: "2024-09-18",
      target: "2024-07-31",
    });
  });
});

describe("serializeCompareParams", () => {
  it("writes the dates that are set, base first, and round-trips", () => {
    const params = { base: "2024-07-31", target: "2024-09-18" };
    expect(serializeCompareParams(params)).toBe("base=2024-07-31&target=2024-09-18");
    expect(parse(serializeCompareParams(params))).toEqual(params);
    expect(serializeCompareParams(EMPTY_PARAMS)).toBe("");
    expect(serializeCompareParams({ base: null, target: "2024-09-18" })).toBe("target=2024-09-18");
  });
});

describe("neighbouring statements", () => {
  it("finds the statement before and after a date", () => {
    expect(previousStatement(DATES, "2024-09-18")).toBe("2024-07-31");
    expect(previousStatement(DATES, "2024-06-12")).toBeNull();
    expect(nextStatement(DATES, "2024-09-18")).toBe("2024-11-07");
    expect(nextStatement(DATES, "2024-11-07")).toBeNull();
  });

  it("names the nearest statements around a date without one", () => {
    expect(nearestStatements(DATES, "2024-08-15")).toEqual({ before: "2024-07-31", after: "2024-09-18" });
    expect(nearestStatements(DATES, "2030-01-01")).toEqual({ before: "2024-11-07", after: null });
  });
});

describe("resolvePair", () => {
  it("defaults to the latest statement against the one before it", () => {
    expect(resolvePair(EMPTY_PARAMS, DATES)).toEqual({ ok: true, base: "2024-09-18", target: "2024-11-07" });
  });

  it("waits for the statement list only when a side has to be filled in", () => {
    expect(resolvePair(EMPTY_PARAMS, null)).toBeNull();
    expect(resolvePair({ base: "2024-07-31", target: "2024-09-18" }, null)).toEqual({
      ok: true,
      base: "2024-07-31",
      target: "2024-09-18",
    });
  });

  it("fills a missing side with the neighbouring statement", () => {
    expect(resolvePair({ base: null, target: "2024-09-18" }, DATES)).toMatchObject({ ok: true, base: "2024-07-31" });
    expect(resolvePair({ base: "2024-07-31", target: null }, DATES)).toMatchObject({ ok: true, target: "2024-09-18" });
  });

  it("reports a pair it cannot compare instead of swapping or guessing", () => {
    expect(resolvePair({ base: "2024-09-18", target: "2024-07-31" }, DATES)).toEqual({
      ok: false,
      problem: "order",
      base: "2024-09-18",
      target: "2024-07-31",
    });
    expect(resolvePair({ base: "2024-09-18", target: "2024-09-18" }, DATES)).toMatchObject({ problem: "order" });
    expect(resolvePair({ base: null, target: "2024-06-12" }, DATES)).toMatchObject({ problem: "no-base" });
    expect(resolvePair({ base: "2024-11-07", target: null }, DATES)).toMatchObject({ problem: "no-target" });
    expect(resolvePair(EMPTY_PARAMS, ["2024-06-12"])).toMatchObject({ problem: "empty" });
  });
});

describe("shiftPair", () => {
  it("moves both sides one statement, keeping the gap", () => {
    expect(shiftPair(DATES, "2024-07-31", "2024-09-18", 1)).toEqual({ base: "2024-09-18", target: "2024-11-07" });
    expect(shiftPair(DATES, "2024-06-12", "2024-09-18", 1)).toEqual({ base: "2024-07-31", target: "2024-11-07" });
    expect(shiftPair(DATES, "2024-07-31", "2024-09-18", -1)).toEqual({ base: "2024-06-12", target: "2024-07-31" });
  });

  it("stops at either end and for dates that are not statements", () => {
    expect(shiftPair(DATES, "2024-06-12", "2024-07-31", -1)).toBeNull();
    expect(shiftPair(DATES, "2024-09-18", "2024-11-07", 1)).toBeNull();
    expect(shiftPair(DATES, "2024-08-01", "2024-09-18", 1)).toBeNull();
  });
});

describe("datesInMessage", () => {
  it("pulls the nearest statement dates out of a 404 message", () => {
    expect(
      datesInMessage("No FOMC statement for meeting date 2024-08-15. Nearest statements: 2024-07-31 and 2024-09-18."),
    ).toEqual(["2024-08-15", "2024-07-31", "2024-09-18"]);
    expect(datesInMessage("Not found (2024-02-30)")).toEqual([]);
  });
});

describe("links", () => {
  it("links to a pair", () => {
    expect(compareHref("2024-07-31", "2024-09-18")).toBe("/compare?base=2024-07-31&target=2024-09-18");
  });

  it("prefills Ask the Fed with the change, as of the target's release", () => {
    const href = compareAskHref("2024-07-31", "2024-09-18", "2024-09-18");
    expect(href.startsWith("/ask?")).toBe(true);
    const values = parseAskParams(new URLSearchParams(href.slice("/ask?".length)));
    expect(values.question).toBe(changeQuestion("2024-07-31", "2024-09-18"));
    expect(values.question).toContain("Jul 31, 2024");
    expect(values.question).toContain("Sep 18, 2024");
    expect(values.asOf).toBe("2024-09-18");
  });
});
