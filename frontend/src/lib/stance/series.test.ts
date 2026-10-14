import { describe, expect, it } from "vitest";

import type { FedFundsTarget, StanceResult, StanceTimeline } from "../api/types";
import {
  adjacentMeetings,
  buildMeetingRows,
  DOMAIN_PAD_MS,
  endLabelsFit,
  fedFundsCeiling,
  fedFundsSeries,
  mergeFedFunds,
  nearestMeeting,
  monthTicks,
  rangeAfterMeeting,
  rangeBeforeMeeting,
  rangeInForce,
  rowsInYears,
  timeDomain,
  timeTicks,
  toIsoDate,
  toTime,
  yearTicks,
} from "./series";

function result(date: string, classifier: StanceResult["classifier"], score: number): StanceResult {
  return {
    meeting_date: date,
    doc_id: `statement-${date}`,
    classifier,
    stance: score > 0.15 ? "hawkish" : score < -0.15 ? "dovish" : "neutral",
    score,
    confidence: null,
    key_factors: [],
    evidence: [],
    model_version: "test",
    url: `https://www.federalreserve.gov/${date}`,
    reference: null,
  };
}

const CHANGES: FedFundsTarget[] = [
  { date: "2015-01-01", lower: 0, upper: 0.25 },
  { date: "2015-12-16", lower: 0.25, upper: 0.5 },
  { date: "2016-12-15", lower: 0.5, upper: 0.75 },
];

function timeline(
  classifier: StanceResult["classifier"],
  points: StanceResult[],
  fed_funds: FedFundsTarget[] = CHANGES,
): StanceTimeline {
  return { classifier, points, fed_funds };
}

describe("toTime / toIsoDate", () => {
  it("round-trips a UTC day", () => {
    expect(toTime("2024-09-18")).toBe(Date.UTC(2024, 8, 18));
    expect(toIsoDate(toTime("2016-02-29"))).toBe("2016-02-29");
  });
});

describe("buildMeetingRows", () => {
  it("merges classifiers per meeting, in date order, with null where a result is missing", () => {
    const reference = { label: "dovish", source: "human", confidence: null } as const;
    const rows = buildMeetingRows({
      finbert: timeline("finbert", [result("2016-12-14", "finbert", 0.2)]),
      llm_few_shot: timeline("llm_few_shot", [
        { ...result("2016-12-14", "llm_few_shot", 0.6), reference },
        result("2015-12-16", "llm_few_shot", 0.7),
      ]),
    });

    expect(rows.map((r) => r.date)).toEqual(["2015-12-16", "2016-12-14"]);
    expect(rows[0]).toMatchObject({ llm_few_shot: 0.7, finbert: null, keyword: null });
    expect(rows[1]).toMatchObject({ llm_few_shot: 0.6, finbert: 0.2, referenceLane: -1 });
    expect(rows[1].reference).toEqual(reference);
    expect(rows[1].results.finbert?.score).toBe(0.2);
    expect(rows[0].reference).toBeNull();
    expect(rows[0].referenceLane).toBeNull();
  });

  it("is empty without timelines", () => {
    expect(buildMeetingRows({})).toEqual([]);
  });
});

describe("rowsInYears / timeDomain", () => {
  const rows = buildMeetingRows({
    keyword: timeline("keyword", [
      result("2015-01-28", "keyword", 0),
      result("2019-07-31", "keyword", -0.3),
      result("2026-09-16", "keyword", 0.2),
    ]),
  });

  it("keeps the meetings of the chosen years", () => {
    expect(rowsInYears(rows, 2019, 2026).map((r) => r.date)).toEqual(["2019-07-31", "2026-09-16"]);
    expect(rowsInYears(rows, 2020, 2025)).toEqual([]);
  });

  it("pads the domain around the first and last meeting", () => {
    expect(timeDomain(rows)).toEqual([
      toTime("2015-01-28") - DOMAIN_PAD_MS,
      toTime("2026-09-16") + DOMAIN_PAD_MS,
    ]);
    expect(timeDomain([])).toBeNull();
  });
});

describe("fed funds", () => {
  it("merges the change points of every timeline", () => {
    const later = { date: "2017-03-16", lower: 0.75, upper: 1 };
    const merged = mergeFedFunds({
      keyword: timeline("keyword", [], CHANGES.slice(0, 2)),
      finbert: timeline("finbert", [], [...CHANGES.slice(1), later]),
    });

    expect(merged.map((c) => c.date)).toEqual([
      "2015-01-01",
      "2015-12-16",
      "2016-12-15",
      "2017-03-16",
    ]);
  });

  it("finds the range in force on a date", () => {
    expect(rangeInForce(CHANGES, "2014-12-31")).toBeNull();
    expect(rangeInForce(CHANGES, "2015-12-15")?.upper).toBe(0.25);
    expect(rangeInForce(CHANGES, "2015-12-16")?.upper).toBe(0.5);
    expect(rangeInForce(CHANGES, "2030-01-01")?.upper).toBe(0.75);
  });

  it("reads the range a meeting set, whether FRED dates it on the day or the day after", () => {
    expect(rangeAfterMeeting(CHANGES, "2015-12-16")?.upper).toBe(0.5); // same day
    expect(rangeAfterMeeting(CHANGES, "2016-12-14")?.upper).toBe(0.75); // next day
    expect(rangeAfterMeeting(CHANGES, "2016-11-02")?.upper).toBe(0.5);
  });

  it("reads the range before and after the meetings FRED dates on the day or two days later", () => {
    // FRED change points as stored: December 2015 and 2016 on the meeting day, and the cut of
    // the March 2, 2020 meeting (announced March 3) on March 4.
    const fred: FedFundsTarget[] = [
      { date: "2015-01-01", lower: 0, upper: 0.25 },
      { date: "2015-12-16", lower: 0.25, upper: 0.5 },
      { date: "2016-12-14", lower: 0.5, upper: 0.75 },
      { date: "2019-10-31", lower: 1.5, upper: 1.75 },
      { date: "2020-03-04", lower: 1, upper: 1.25 },
      { date: "2020-03-16", lower: 0, upper: 0.25 },
    ];
    const both = (date: string) => [
      rangeBeforeMeeting(fred, date)?.upper,
      rangeAfterMeeting(fred, date)?.upper,
    ];

    expect(both("2015-12-16")).toEqual([0.25, 0.5]);
    expect(both("2016-12-14")).toEqual([0.5, 0.75]);
    expect(both("2020-03-02")).toEqual([1.75, 1.25]);
    expect(both("2020-03-15")).toEqual([1.25, 0.25]);
    expect(both("2020-01-29")).toEqual([1.75, 1.75]); // a hold
  });

  it("builds a step series that starts with the range in force and reaches the frame", () => {
    const domain: [number, number] = [toTime("2015-06-01"), toTime("2017-01-31")];
    const points = fedFundsSeries(CHANGES, domain, ["2016-12-14"]);

    expect(points.map((p) => [p.date, p.upper])).toEqual([
      ["2015-06-01", 0.25], // in force at the start of the window
      ["2015-12-16", 0.5],
      ["2016-12-14", 0.75], // the meeting reads the range it set
      ["2016-12-15", 0.75],
      ["2017-01-31", 0.75], // the last step reaches the end of the frame
    ]);
    expect(points[1].range).toEqual([0.25, 0.5]);
  });

  it("is empty when the window ends before the first change point", () => {
    expect(fedFundsSeries(CHANGES, [toTime("2014-01-01"), toTime("2014-06-01")])).toEqual([]);
    expect(fedFundsSeries([], [toTime("2015-01-01"), toTime("2016-01-01")])).toEqual([]);
  });

  it("rounds the axis ceiling up to a whole percent", () => {
    const points = fedFundsSeries(CHANGES, [toTime("2015-01-01"), toTime("2017-01-01")]);
    expect(fedFundsCeiling(points)).toBe(1);
    expect(fedFundsCeiling([{ ...points[0], upper: 5.5 }])).toBe(6);
    expect(fedFundsCeiling([])).toBe(1);
  });
});

describe("yearTicks", () => {
  const domain: [number, number] = [toTime("2015-01-08"), toTime("2026-10-06")];

  it("puts a tick on every January 1 inside the domain", () => {
    const ticks = yearTicks(domain);
    expect(ticks.map((t) => new Date(t).getUTCFullYear())).toEqual([
      2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026,
    ]);
  });

  it("thins ticks to every other year", () => {
    expect(yearTicks(domain, 2).map((t) => new Date(t).getUTCFullYear())).toEqual([
      2016, 2018, 2020, 2022, 2024, 2026,
    ]);
  });
});

describe("timeTicks", () => {
  it("uses year ticks on a long axis", () => {
    const { ticks, label } = timeTicks([toTime("2015-01-08"), toTime("2026-10-06")]);
    expect(ticks).toEqual(yearTicks([toTime("2015-01-08"), toTime("2026-10-06")]));
    expect(label(ticks[0])).toBe("2016");
  });

  it("names the year of a one-year window with quarter ticks", () => {
    // The 2020 view: its first and last meeting, padded by 20 days.
    const domain: [number, number] = [toTime("2020-01-09"), toTime("2021-01-05")];
    const { ticks, label } = timeTicks(domain);

    expect(ticks.map(toIsoDate)).toEqual(["2020-04-01", "2020-07-01", "2020-10-01", "2021-01-01"]);
    expect(ticks.map(label)).toEqual(["Apr 2020", "Jul", "Oct", "2021"]);
    expect(timeTicks(domain, true).ticks.map(toIsoDate)).toEqual(["2020-07-01", "2021-01-01"]);
  });

  it("puts month ticks every stepMonths months from January", () => {
    const ticks = monthTicks([toTime("2023-12-01"), toTime("2024-12-31")], 6);
    expect(ticks.map(toIsoDate)).toEqual(["2024-01-01", "2024-07-01"]);
  });
});

describe("meeting navigation", () => {
  const rows = buildMeetingRows({
    keyword: timeline("keyword", [
      result("2024-07-31", "keyword", 0),
      result("2024-09-18", "keyword", -0.4),
      result("2024-11-07", "keyword", -0.2),
    ]),
  });

  it("snaps a time to the nearest meeting", () => {
    expect(nearestMeeting(rows, toTime("2024-09-30"))).toBe("2024-09-18");
    expect(nearestMeeting(rows, toTime("2030-01-01"))).toBe("2024-11-07");
    expect(nearestMeeting([], 0)).toBeNull();
  });

  it("finds the previous and next meeting", () => {
    expect(adjacentMeetings(rows, "2024-09-18")).toEqual({
      previous: "2024-07-31",
      next: "2024-11-07",
    });
    expect(adjacentMeetings(rows, "2024-07-31").previous).toBeNull();
    expect(adjacentMeetings(rows, "2023-01-01")).toEqual({ previous: null, next: null });
  });
});

describe("endLabelsFit", () => {
  it("accepts labels far enough apart and rejects colliding ones", () => {
    expect(endLabelsFit([0.6, -0.2], 200)).toBe(true);
    expect(endLabelsFit([0.6, 0.55], 200)).toBe(false); // 5 px apart
    expect(endLabelsFit([0.5], 200)).toBe(true);
  });
});
