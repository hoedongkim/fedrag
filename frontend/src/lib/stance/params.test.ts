import { describe, expect, it } from "vitest";

import { parseAskParams } from "../ask/params";
import {
  askHref,
  DEFAULT_VIEW,
  orderClassifiers,
  parseStanceParams,
  serializeStanceParams,
  toggleClassifier,
} from "./params";

const parse = (query: string) => parseStanceParams(new URLSearchParams(query));

describe("parseStanceParams", () => {
  it("defaults to LLM few-shot, the latest meeting and every year", () => {
    expect(parse("")).toEqual(DEFAULT_VIEW);
  });

  it("reads classifiers in display order, dropping unknown and duplicate ones", () => {
    expect(parse("c=keyword,vibes,llm_few_shot,keyword").classifiers).toEqual([
      "llm_few_shot",
      "keyword",
    ]);
    expect(parse("c=vibes").classifiers).toEqual(["llm_few_shot"]);
  });

  it("reads a valid meeting date only", () => {
    expect(parse("m=2024-09-18").meeting).toBe("2024-09-18");
    expect(parse("m=2024-02-30").meeting).toBeNull();
    expect(parse("m=yesterday").meeting).toBeNull();
  });

  it("reads the year range and drops an inverted or out-of-range one", () => {
    expect(parse("from=2019&to=2022")).toMatchObject({ fromYear: 2019, toYear: 2022 });
    expect(parse("from=2022&to=2019")).toMatchObject({ fromYear: null, toYear: null });
    expect(parse("from=1999&to=20x5")).toMatchObject({ fromYear: null, toYear: null });
  });
});

describe("serializeStanceParams", () => {
  it("leaves defaults out", () => {
    expect(serializeStanceParams(DEFAULT_VIEW)).toBe("");
  });

  it("round-trips a full view with readable commas", () => {
    const view = {
      classifiers: ["llm_few_shot", "finbert"] as const,
      meeting: "2020-03-15",
      fromYear: 2019,
      toYear: 2021,
    };
    const query = serializeStanceParams({ ...view, classifiers: [...view.classifiers] });

    expect(query).toBe("c=llm_few_shot,finbert&m=2020-03-15&from=2019&to=2021");
    expect(parse(query)).toEqual({ ...view, classifiers: [...view.classifiers] });
  });
});

describe("classifier selection", () => {
  it("orders and deduplicates", () => {
    expect(orderClassifiers(["keyword", "finbert", "keyword"])).toEqual(["finbert", "keyword"]);
    expect(orderClassifiers([])).toEqual(["llm_few_shot"]);
  });

  it("toggles but never empties the selection", () => {
    expect(toggleClassifier(["llm_few_shot"], "keyword")).toEqual(["llm_few_shot", "keyword"]);
    expect(toggleClassifier(["llm_few_shot", "keyword"], "llm_few_shot")).toEqual(["keyword"]);
    expect(toggleClassifier(["keyword"], "keyword")).toEqual(["keyword"]);
  });
});

describe("askHref", () => {
  it("prefills Ask the Fed with a question about the meeting", () => {
    const href = askHref("2024-09-18");
    const [path, query] = href.split("?");
    const values = parseAskParams(new URLSearchParams(query));

    expect(path).toBe("/ask");
    expect(values.question).toContain("Sep 18, 2024 meeting");
    expect(values.asOf).toBe("");
    expect(values.method).toBe("hybrid_rerank");
  });
});
