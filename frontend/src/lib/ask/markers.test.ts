import { describe, expect, it } from "vitest";

import { excerpt, groupCitationRuns, splitCitationMarkers, splitParagraphs } from "./markers";

describe("splitCitationMarkers", () => {
  it("splits text and markers", () => {
    expect(splitCitationMarkers("Rates rose [1]. Inflation eased [2].")).toEqual([
      { kind: "text", text: "Rates rose " },
      { kind: "cite", n: 1 },
      { kind: "text", text: ". Inflation eased " },
      { kind: "cite", n: 2 },
      { kind: "text", text: "." },
    ]);
  });

  it("keeps adjacent markers as separate citations", () => {
    expect(splitCitationMarkers("Both [1][3]")).toEqual([
      { kind: "text", text: "Both " },
      { kind: "cite", n: 1 },
      { kind: "cite", n: 3 },
    ]);
  });

  it("handles markers at the start and multi-digit numbers", () => {
    expect(splitCitationMarkers("[12] leads")).toEqual([
      { kind: "cite", n: 12 },
      { kind: "text", text: " leads" },
    ]);
  });

  it("leaves non-markers as text", () => {
    const text = "Ranges [1-2], lists [1, 2], [0], [a] and [] stay text.";
    expect(splitCitationMarkers(text)).toEqual([{ kind: "text", text }]);
  });

  it("leaves three-digit numbers as text, as the backend does", () => {
    expect(splitCitationMarkers("see table [100]")).toEqual([{ kind: "text", text: "see table [100]" }]);
    expect(splitCitationMarkers("see table [100", { streaming: true })).toEqual([
      { kind: "text", text: "see table [100" },
    ]);
  });

  it("holds back an unfinished marker at the end while streaming", () => {
    for (const tail of ["[", "[1", "[12"]) {
      expect(splitCitationMarkers(`Rates rose ${tail}`, { streaming: true })).toEqual([
        { kind: "text", text: "Rates rose " },
      ]);
    }
    expect(splitCitationMarkers("Rates rose [1", { streaming: false })).toEqual([
      { kind: "text", text: "Rates rose [1" },
    ]);
    expect(splitCitationMarkers("Rates rose [1]", { streaming: true })).toEqual([
      { kind: "text", text: "Rates rose " },
      { kind: "cite", n: 1 },
    ]);
  });

  it("returns nothing for empty text", () => {
    expect(splitCitationMarkers("")).toEqual([]);
  });
});

describe("groupCitationRuns", () => {
  const pieces = (text: string) => groupCitationRuns(splitCitationMarkers(text));

  it("keeps adjacent markers and the punctuation after them together", () => {
    expect(pieces('It was "sound and resilient" [1][2]. Participants agreed [3], then [1]')).toEqual([
      { kind: "text", text: 'It was "sound and resilient" ' },
      { kind: "cites", ns: [1, 2], after: "." },
      { kind: "text", text: " Participants agreed " },
      { kind: "cites", ns: [3], after: "," },
      { kind: "text", text: " then " },
      { kind: "cites", ns: [1], after: "" },
    ]);
  });

  it("starts a new run after text or punctuation between markers", () => {
    expect(pieces("[1] and [2]")).toEqual([
      { kind: "cites", ns: [1], after: "" },
      { kind: "text", text: " and " },
      { kind: "cites", ns: [2], after: "" },
    ]);
    expect(pieces("[1].[2]")).toEqual([
      { kind: "cites", ns: [1], after: "." },
      { kind: "cites", ns: [2], after: "" },
    ]);
  });

  it("leaves text without markers alone", () => {
    expect(pieces("No markers here.")).toEqual([{ kind: "text", text: "No markers here." }]);
    expect(pieces("")).toEqual([]);
  });
});

describe("splitParagraphs", () => {
  it("splits on blank lines and keeps single newlines", () => {
    expect(splitParagraphs("First [1].\n\nSecond\nline.\n \nThird.\n")).toEqual([
      "First [1].",
      "Second\nline.",
      "Third.",
    ]);
  });

  it("drops empty paragraphs", () => {
    expect(splitParagraphs("\n\n\nOnly.\n\n\n")).toEqual(["Only."]);
    expect(splitParagraphs("")).toEqual([]);
  });
});

describe("excerpt", () => {
  it("returns short text unchanged, on one line", () => {
    expect(excerpt("Rates\n\nrose.", 200)).toBe("Rates rose.");
  });

  it("cuts long text at a word boundary with an ellipsis", () => {
    const text = "The Committee decided to raise the target range for the federal funds rate.";
    const result = excerpt(text, 40);
    expect(result.length).toBeLessThanOrEqual(40);
    expect(result).toBe("The Committee decided to raise the…");
  });

  it("cuts inside a very long word when there is no good boundary", () => {
    expect(excerpt("x".repeat(50), 10)).toBe(`${"x".repeat(9)}…`);
  });
});
