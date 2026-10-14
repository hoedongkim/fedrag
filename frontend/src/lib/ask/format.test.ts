import { describe, expect, it } from "vitest";

import { displayHost, formatDate, formatLatency, formatScore, isRevised, localToday } from "./format";

describe("formatDate", () => {
  it("formats ISO dates without a time-zone shift", () => {
    expect(formatDate("2023-03-22")).toBe("Mar 22, 2023");
    expect(formatDate("2015-01-01")).toBe("Jan 1, 2015");
  });

  it("returns unparseable input unchanged", () => {
    expect(formatDate("soon")).toBe("soon");
  });
});

describe("formatLatency", () => {
  it("uses ms under a second and seconds above", () => {
    expect(formatLatency(850.4)).toBe("850 ms");
    expect(formatLatency(4213.7)).toBe("4.2 s");
    expect(formatLatency(-1)).toBe("-");
  });
});

describe("formatScore", () => {
  it("fits the precision to the scale", () => {
    expect(formatScore(23.456)).toBe("23.5");
    expect(formatScore(3.14159)).toBe("3.14");
    expect(formatScore(0.94812)).toBe("0.948");
    expect(formatScore(0.032787)).toBe("0.0328");
    expect(formatScore(0)).toBe("0");
  });
});

describe("isRevised", () => {
  it("is true only when the page changed after release", () => {
    expect(isRevised("2018-11-29", "2026-10-09")).toBe(true);
    expect(isRevised("2018-11-29", "2018-11-29")).toBe(false);
    expect(isRevised("2018-11-29", null)).toBe(false);
  });
});

describe("localToday", () => {
  it("formats the local calendar date", () => {
    expect(localToday(new Date(2026, 0, 5, 23, 59))).toBe("2026-01-05");
  });
});

describe("displayHost", () => {
  it("drops www and tolerates bad URLs", () => {
    expect(displayHost("https://www.federalreserve.gov/monetarypolicy/x.htm")).toBe("federalreserve.gov");
    expect(displayHost("not a url")).toBe("");
  });
});
