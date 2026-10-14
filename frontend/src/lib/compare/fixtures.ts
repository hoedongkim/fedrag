/**
 * FIXTURE for unit tests only: never import it from app code, so it stays out of the bundle
 * (ESLint enforces this for src/app and src/components).
 *
 * A POST /api/compare response for July 31 -> September 18, 2024, shaped exactly like the Phase 6
 * contract. The diff is the real statements' text aligned the way the contract describes; the
 * analysis is hand-written for the fixture, not model output.
 */
import type { CompareResponse, MeetingSummary } from "../api/types";

export const COMPARE_FIXTURE = {
  base: {
    doc_id: "statement-2024-07-31",
    doc_type: "statement",
    meeting_date: "2024-07-31",
    release_date: "2024-07-31",
    page_updated: "2024-07-31",
    title: "FOMC Statement, July 31, 2024",
    url: "https://www.federalreserve.gov/newsevents/pressreleases/monetary20240731a.htm"
  },
  target: {
    doc_id: "statement-2024-09-18",
    doc_type: "statement",
    meeting_date: "2024-09-18",
    release_date: "2024-09-18",
    page_updated: "2024-09-18",
    title: "FOMC Statement, September 18, 2024",
    url: "https://www.federalreserve.gov/newsevents/pressreleases/monetary20240918a.htm"
  },
  changes: [
    {
      id: 0,
      kind: "unchanged",
      base: "Recent indicators suggest that economic activity has continued to expand at a solid pace.",
      target: "Recent indicators suggest that economic activity has continued to expand at a solid pace.",
      similarity: 1.0,
      procedural: false,
      segments: null
    },
    {
      id: 1,
      kind: "modified",
      base: "Job gains have moderated, and the unemployment rate has moved up but remains low.",
      target: "Job gains have slowed, and the unemployment rate has moved up but remains low.",
      similarity: 0.9286,
      procedural: false,
      segments: [
        { op: "equal", text: "Job gains have " },
        { op: "delete", text: "moderated, " },
        { op: "insert", text: "slowed, " },
        { op: "equal", text: "and the unemployment rate has moved up but remains low." }
      ]
    },
    {
      id: 2,
      kind: "removed",
      base: "Inflation has eased over the past year but remains somewhat elevated.",
      target: null,
      similarity: null,
      procedural: false,
      segments: null
    },
    {
      id: 3,
      kind: "modified",
      base: "In recent months, there has been some further progress toward the Committee's 2 percent inflation objective.",
      target: "Inflation has made further progress toward the Committee's 2 percent objective but remains somewhat elevated.",
      similarity: 0.6061,
      procedural: false,
      segments: [
        { op: "delete", text: "In recent months, there " },
        { op: "insert", text: "Inflation " },
        { op: "equal", text: "has " },
        { op: "delete", text: "been some " },
        { op: "insert", text: "made " },
        { op: "equal", text: "further progress toward the Committee's 2 percent " },
        { op: "delete", text: "inflation objective." },
        { op: "insert", text: "objective but remains somewhat elevated." }
      ]
    },
    {
      id: 4,
      kind: "unchanged",
      base: "The Committee seeks to achieve maximum employment and inflation at the rate of 2 percent over the longer run.",
      target: "The Committee seeks to achieve maximum employment and inflation at the rate of 2 percent over the longer run.",
      similarity: 1.0,
      procedural: false,
      segments: null
    },
    {
      id: 5,
      kind: "modified",
      base: "The Committee judges that the risks to achieving its employment and inflation goals continue to move into better balance.",
      target: "The Committee has gained greater confidence that inflation is moving sustainably toward 2 percent, and judges that the risks to achieving its employment and inflation goals are roughly in balance.",
      similarity: 0.5714,
      procedural: false,
      segments: [
        { op: "equal", text: "The Committee " },
        { op: "insert", text: "has gained greater confidence that inflation is moving sustainably toward 2 percent, and " },
        { op: "equal", text: "judges that the risks to achieving its employment and inflation goals " },
        { op: "delete", text: "continue to move into better " },
        { op: "insert", text: "are roughly in " },
        { op: "equal", text: "balance." }
      ]
    },
    {
      id: 6,
      kind: "unchanged",
      base: "The economic outlook is uncertain, and the Committee is attentive to the risks to both sides of its dual mandate.",
      target: "The economic outlook is uncertain, and the Committee is attentive to the risks to both sides of its dual mandate.",
      similarity: 1.0,
      procedural: false,
      segments: null
    },
    {
      id: 7,
      kind: "modified",
      base: "In support of its goals, the Committee decided to maintain the target range for the federal funds rate at 5-1/4 to 5-1/2 percent.",
      target: "In light of the progress on inflation and the balance of risks, the Committee decided to lower the target range for the federal funds rate by 1/2 percentage point to 4-3/4 to 5 percent.",
      similarity: 0.5938,
      procedural: false,
      segments: [
        { op: "equal", text: "In " },
        { op: "delete", text: "support " },
        { op: "insert", text: "light " },
        { op: "equal", text: "of " },
        { op: "delete", text: "its goals, " },
        { op: "insert", text: "the progress on inflation and the balance of risks, " },
        { op: "equal", text: "the Committee decided to " },
        { op: "delete", text: "maintain " },
        { op: "insert", text: "lower " },
        { op: "equal", text: "the target range for the federal funds rate " },
        { op: "delete", text: "at 5-1/4 " },
        { op: "insert", text: "by 1/2 percentage point " },
        { op: "equal", text: "to " },
        { op: "delete", text: "5-1/2 " },
        { op: "insert", text: "4-3/4 to 5 " },
        { op: "equal", text: "percent." }
      ]
    },
    {
      id: 8,
      kind: "modified",
      base: "In considering any adjustments to the target range for the federal funds rate, the Committee will carefully assess incoming data, the evolving outlook, and the balance of risks.",
      target: "In considering additional adjustments to the target range for the federal funds rate, the Committee will carefully assess incoming data, the evolving outlook, and the balance of risks.",
      similarity: 0.9643,
      procedural: false,
      segments: [
        { op: "equal", text: "In considering " },
        { op: "delete", text: "any " },
        { op: "insert", text: "additional " },
        { op: "equal", text: "adjustments to the target range for the federal funds rate, the Committee will carefully assess incoming data, the evolving outlook, and the balance of risks." }
      ]
    },
    {
      id: 9,
      kind: "removed",
      base: "The Committee does not expect it will be appropriate to reduce the target range until it has gained greater confidence that inflation is moving sustainably toward 2 percent.",
      target: null,
      similarity: null,
      procedural: false,
      segments: null
    },
    {
      id: 10,
      kind: "modified",
      base: "In addition, the Committee will continue reducing its holdings of Treasury securities and agency debt and agency mortgage-backed securities.",
      target: "The Committee will continue reducing its holdings of Treasury securities and agency debt and agency mortgage-backed securities.",
      similarity: 0.9474,
      procedural: false,
      segments: [
        { op: "delete", text: "In addition, the " },
        { op: "insert", text: "The " },
        { op: "equal", text: "Committee will continue reducing its holdings of Treasury securities and agency debt and agency mortgage-backed securities." }
      ]
    },
    {
      id: 11,
      kind: "modified",
      base: "The Committee is strongly committed to returning inflation to its 2 percent objective.",
      target: "The Committee is strongly committed to supporting maximum employment and returning inflation to its 2 percent objective.",
      similarity: 0.8667,
      procedural: false,
      segments: [
        { op: "equal", text: "The Committee is strongly committed to " },
        { op: "insert", text: "supporting maximum employment and " },
        { op: "equal", text: "returning inflation to its 2 percent objective." }
      ]
    },
    {
      id: 12,
      kind: "unchanged",
      base: "In assessing the appropriate stance of monetary policy, the Committee will continue to monitor the implications of incoming information for the economic outlook.",
      target: "In assessing the appropriate stance of monetary policy, the Committee will continue to monitor the implications of incoming information for the economic outlook.",
      similarity: 1.0,
      procedural: false,
      segments: null
    },
    {
      id: 13,
      kind: "unchanged",
      base: "The Committee would be prepared to adjust the stance of monetary policy as appropriate if risks emerge that could impede the attainment of the Committee's goals.",
      target: "The Committee would be prepared to adjust the stance of monetary policy as appropriate if risks emerge that could impede the attainment of the Committee's goals.",
      similarity: 1.0,
      procedural: false,
      segments: null
    },
    {
      id: 14,
      kind: "unchanged",
      base: "The Committee's assessments will take into account a wide range of information, including readings on labor market conditions, inflation pressures and inflation expectations, and financial and international developments.",
      target: "The Committee's assessments will take into account a wide range of information, including readings on labor market conditions, inflation pressures and inflation expectations, and financial and international developments.",
      similarity: 1.0,
      procedural: false,
      segments: null
    },
    {
      id: 15,
      kind: "modified",
      base: "Voting for the monetary policy action were Jerome H. Powell, Chair; John C. Williams, Vice Chair; Thomas I. Barkin; Michael S. Barr; Raphael W. Bostic; Michelle W. Bowman; Lisa D. Cook; Mary C. Daly; Austan D. Goolsbee; Philip N. Jefferson; Adriana D. Kugler; and Christopher J. Waller.",
      target: "Voting for the monetary policy action were Jerome H. Powell, Chair; John C. Williams, Vice Chair; Thomas I. Barkin; Michael S. Barr; Raphael W. Bostic; Lisa D. Cook; Mary C. Daly; Beth M. Hammack; Philip N. Jefferson; Adriana D. Kugler; and Christopher J. Waller.",
      similarity: 0.9011,
      procedural: true,
      segments: [
        { op: "equal", text: "Voting for the monetary policy action were Jerome H. Powell, Chair; John C. Williams, Vice Chair; Thomas I. Barkin; Michael S. Barr; Raphael W. Bostic; " },
        { op: "delete", text: "Michelle W. Bowman; " },
        { op: "equal", text: "Lisa D. Cook; Mary C. Daly; " },
        { op: "delete", text: "Austan D. Goolsbee; " },
        { op: "insert", text: "Beth M. Hammack; " },
        { op: "equal", text: "Philip N. Jefferson; Adriana D. Kugler; and Christopher J. Waller." }
      ]
    },
    {
      id: 16,
      kind: "removed",
      base: "Austan D. Goolsbee voted as an alternate member at this meeting.",
      target: null,
      similarity: null,
      procedural: true,
      segments: null
    },
    {
      id: 17,
      kind: "added",
      base: null,
      target: "Voting against this action was Michelle W. Bowman, who preferred to lower the target range for the federal funds rate by 1/4 percentage point at this meeting.",
      similarity: null,
      procedural: true,
      segments: null
    }
  ],
  stats: {
    unchanged: 6,
    modified: 8,
    added: 1,
    removed: 3,
    similarity: 0.7962
  },
  analysis: {
    summary: "The Committee moves from holding the target range at 5-1/4 to 5-1/2 percent to cutting it by 1/2 percentage point, citing progress on inflation and the balance of risks. The July condition that it needed greater confidence on inflation before reducing rates is dropped: the statement now says it has gained that confidence and calls the risks roughly in balance. It also adds supporting maximum employment to its stated commitment.",
    key_changes: [
      {
        change_ids: [7, 9],
        what_changed: "\"decided to maintain the target range\" became \"decided to lower the target range for the federal funds rate by 1/2 percentage point\"",
        interpretation: "A rate cut, and a larger first step than the usual quarter point.",
        direction: "dovish"
      },
      {
        change_ids: [5],
        what_changed: "\"continue to move into better balance\" became \"has gained greater confidence that inflation is moving sustainably toward 2 percent\"",
        interpretation: "The precondition for easing stated in July is declared met.",
        direction: "dovish"
      },
      {
        change_ids: [1],
        what_changed: "\"Job gains have moderated\" became \"Job gains have slowed\"",
        interpretation: "A softer description of the labor market.",
        direction: "dovish"
      },
      {
        change_ids: [11],
        what_changed: "Added \"supporting maximum employment\" to the commitment sentence",
        interpretation: "Puts the employment side of the mandate next to inflation.",
        direction: "neutral"
      }
    ],
    stance_shift: "more_dovish",
    confidence: 0.86
  },
  analysis_status: "ready",
  analysis_model_version: "gpt-5.4-mini-2026-03-17/compare-v3.edfeb8da",
  stance: {
    base: { classifier: "llm_zero_shot", stance: "neutral", score: 0.1 },
    target: { classifier: "llm_zero_shot", stance: "dovish", score: -0.9 }
  }
} satisfies CompareResponse;

/** A few statement meetings from GET /api/meetings?doc_type=statement, oldest first. */
export const STATEMENT_FIXTURE: MeetingSummary[] = ["2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07"].map(
  (meeting_date) => ({
    meeting_date,
    start_date: meeting_date,
    unscheduled: false,
    has_sep: false,
    has_press_conference: true,
    doc_types: ["statement"],
  }),
);
