import { describe, expect, it } from "vitest";

import {
  formatConfidence,
  formatCount,
  formatDifference,
  formatInterval,
  formatLevel,
  formatMeanSd,
  formatMetric,
  formatRange,
  formatRate,
  formatScore,
  formatTimestamp,
} from "./format";

describe("formatScore", () => {
  it("signs scores with a plus or a true minus", () => {
    expect(formatScore(0.42)).toBe("+0.42");
    expect(formatScore(-0.4)).toBe("−0.40");
    expect(formatScore(1)).toBe("+1.00");
  });

  it("leaves zero (and what rounds to it) unsigned", () => {
    expect(formatScore(0)).toBe("0.00");
    expect(formatScore(-0.001)).toBe("0.00");
    expect(formatScore(Number.NaN)).toBe("-");
  });
});

describe("metrics", () => {
  it("formats metrics and missing values", () => {
    expect(formatMetric(0.6846)).toBe("0.685");
    expect(formatMetric(-0.05, 2)).toBe("−0.05");
    expect(formatMetric(null)).toBe("n/a");
  });

  it("formats intervals, signed differences and confidence levels", () => {
    expect(formatInterval({ low: 0.83505, high: 0.94845 })).toBe("0.835–0.948");
    expect(formatInterval(null)).toBeNull();
    expect(formatInterval(undefined)).toBeNull();
    expect(formatDifference(0.051546, { low: -0.030928, high: 0.13402 })).toBe(
      "+0.052 (−0.031 to +0.134)",
    );
    expect(formatLevel(0.95)).toBe("95%");
    expect(formatLevel(0.9)).toBe("90%");
  });

  it("formats mean ± sd, or just the mean", () => {
    expect(formatMeanSd(0.684, 0.0113)).toBe("0.684 ± 0.011");
    expect(formatMeanSd(0.5, null)).toBe("0.500");
  });

  it("formats test-set sizes", () => {
    expect(formatCount(496)).toBe("496");
    expect(formatCount(482.666)).toBe("482.7");
    expect(formatCount(1903)).toBe("1,903");
  });
});

describe("formatConfidence", () => {
  it("shows numbers to two decimals and levels as written", () => {
    expect(formatConfidence(0.8)).toBe("0.80");
    expect(formatConfidence("high")).toBe("high");
    expect(formatConfidence(null)).toBeNull();
  });
});

describe("rates", () => {
  it("drops trailing zeros", () => {
    expect(formatRate(4.25)).toBe("4.25");
    expect(formatRate(4.5)).toBe("4.5");
    expect(formatRate(0)).toBe("0");
  });

  it("formats a target range with an en dash", () => {
    expect(formatRange(4.25, 4.5)).toBe("4.25–4.5%");
  });
});

describe("formatTimestamp", () => {
  it("shows the UTC date of a timestamp", () => {
    expect(formatTimestamp("2026-10-13T23:30:00Z")).toBe("Oct 13, 2026");
    expect(formatTimestamp("not a date")).toBe("not a date");
  });
});
