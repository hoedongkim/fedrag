# Policy Stance Labeling Guideline

## Purpose

These labels are the reference labels for the Policy Stance tab (Phase 5): 25 human-reviewed, 72 LLM-adjudicated
only. Each of the 97 FOMC statements in
`data/processed/documents.jsonl` (`doc_type == "statement"`, 2015-01-28 to 2026-09-16) gets one label:
**hawkish**, **neutral** or **dovish**. The labels are used to evaluate the four stance classifiers. Two annotators
label every statement independently, agreement between them is measured, and the project owner re-labels
a subset blind. So any label must be reproducible from the statement text and this rubric alone, and the
confidence you record must be honest.

## Unit and label

- **Unit:** one statement, meaning one `doc_id`, read in full (`text`) including the vote paragraph.
  Unscheduled statements (`statement-2019-10-04`, `statement-2020-03-02`, `statement-2020-03-15`,
  `statement-2020-03-23`) are labeled the same way as scheduled ones.
- **Label:** the stance the statement itself communicates about the policy path *at release time*. The label
  is absolute: do not judge it against the previous statement, and never use hindsight.
  - **hawkish** (tightening): raises the target range, signals further firming, makes upside inflation risk
    the dominant concern, or starts or accelerates balance-sheet runoff.
  - **dovish** (easing): cuts the target range, signals cuts or continued accommodation, makes downside
    risk to employment or activity dominant, expands asset purchases, or slows or ends runoff for easing reasons.
  - **neutral**: holds the rate with balanced or data-dependent guidance and no clear lean.

## Decision order

Work through the steps in order. The first step with a clear lean sets the label. Later steps only change
confidence, except where a tie-break below says otherwise. If no step leans either way, the label is **neutral**.

1. **Rate action at this meeting.** A hike is hawkish, a cut is dovish, and a hold sends you to step 2.
2. **Forward guidance on the rate path (holds only).**
   - *Tightening bias:* further or ongoing increases are expected; "additional policy firming may be
     appropriate"; an increase "soon" or "at its next meeting"; "the case for an increase has strengthened".
   - *Easing bias:* the statement signals cuts or says it "will act as appropriate to sustain the expansion";
     it commits to keep the rate at the lower bound, or policy accommodative, *until* stated goals are met.
   - *No lean:* "patient"; "any adjustments" or "timing and size of future adjustments"; "additional
     adjustments" with no stated direction; guidance about when the next move may start, tied to conditions not
     yet met ("does not expect it will be appropriate to reduce ... until ..."; "appropriate to raise ... when
     it has seen further improvement").
     Normalization boilerplate that only limits the pace also has no lean: "(only) gradual increases",
     "remain, for some time, below levels ... expected to prevail in the longer run", "remains accommodative".
3. **Balance sheet.**
   - *Tightening:* the statement starts or accelerates runoff, ends net purchases, or begins or speeds up a
     taper (this counts even though holdings are still growing). Signaling any of these also counts.
   - *Easing:* the statement starts or expands net asset purchases, or keeps their pace; or it slows or ends
     runoff with a stated easing rationale or alongside a cut.
   - *No lean:* operations described as technical or for reserve management; reinvestment that only keeps
     holdings unchanged; runoff slowed with no easing rationale; runoff continuing at its existing pace.
4. **Risk-balance language.** This step leans only when one side clearly dominates. Hawkish examples: "highly
   attentive to inflation risks" with no balancing judgment, or elevated inflation as the only stated concern.
   Dovish examples: "downside risks to employment have risen", or rising uncertainty paired with muted
   inflation. An explicit balance judgment ("roughly balanced", "moving into better balance", "attentive to
   the risks to both sides") counts as balanced and outranks attentiveness phrases. The paragraph describing
   the economy is not a signal on its own.

## Tie-breaks for mixed signals

| # | Situation | Label | Confidence |
|---|---|---|---|
| T1 | Hike plus a hint that hikes are ending or pausing | hawkish | medium |
| T2 | Cut plus guidance pointing to a pause or a slower pace (a "hawkish cut") | dovish | medium |
| T3 | Hold with a tightening or easing bias. An explicit "further", "ongoing" or "additional" outweighs pace-limiting boilerplate in the same sentence | follows the bias | medium |
| T4 | Emergency actions in 2020: unscheduled cuts and open-ended or expanded purchases that the statement ties to supporting the economy or the flow of credit | dovish | high |
| T5 | QE or QT against rate guidance: a balance-sheet **decision** taken at this meeting outranks rate guidance that points the other way. A balance-sheet **signal** only lowers confidence | follows the decision | low |
| T6 | Technical or implementation statement that says it does not change the stance | neutral | high |

A rate move is never flipped by guidance, balance-sheet signals or risk language. Those only lower confidence.

## What not to use

- **Hindsight:** later meetings, minutes, projections and dot plots, press conferences, speeches, revised data,
  or knowing that "this was the last hike".
- **Market reaction**, media framing or analyst framing (for example a "hawkish cut" headline).
- **The previous statement.** Do not compare wording across statements. A hold is not hawkish just because a
  dovish phrase was dropped.
- **Dissents alone.** Dissents are context. They can point you to wording, but the label has to rest on the
  Committee's own text. `statement-2026-06-17` and `statement-2026-07-29` have the same policy text, and the
  three July dissents in favor of a hike do not change the July label.
- **The level of the rate** ("restrictive", "near zero"). Label the path the statement communicates, not where
  the rate sits.

## Confidence

- **high:** the label rests on an action taken or continued in this statement (a rate move, asset purchases,
  the start of runoff) or on an explicit statement about the stance, and nothing in the statement points the
  other way. A hold where no step leans and the guidance is plainly two-sided is also high.
- **medium:** explicit hold guidance (step 2) decides the label, since a bias is not yet an action; or a rate
  move comes with guidance or risk language that leans the other way (T1, T2).
- **low:** only a balance-sheet signal (not a decision) or risk language (step 4) decides the label, or you
  applied T5, or a careful reader could reasonably pick the adjacent label. Low still means your best label,
  not a guess.

For each statement, record the label, the confidence, the deciding step (1 to 4, or a T number) and the phrase
you relied on (25 words or fewer).

## Examples

| doc_id | Key text (quoted) | Label, confidence | Why |
|---|---|---|---|
| statement-2022-05-04 | "...decided to raise the target range ... and anticipates that ongoing increases in the target range will be appropriate." | hawkish, high | Step 1. Guidance and the start of runoff agree |
| statement-2020-03-15 | "...the Committee decided to lower the target range for the federal funds rate to 0 to 1/4 percent." | dovish, high | Unscheduled emergency cut plus purchases of at least $500 billion (T4) |
| statement-2019-12-11 | "...the current stance of monetary policy is appropriate..." | neutral, high | Hold. Guidance has no lean and no risk side dominates |
| statement-2019-10-04 | "These actions are purely technical measures ... and do not represent a change in the stance of monetary policy." | neutral, high | T6. Bill purchases for reserve management |
| statement-2023-05-03 | "In determining the extent to which additional policy firming may be appropriate..." | hawkish, medium | T1. The hike decides, and the hedged firming language hints at a pause |
| statement-2023-06-14 | "Holding the target range steady at this meeting allows the Committee to assess additional information..." | hawkish, medium | T3. Hold that keeps "additional policy firming that may be appropriate" |
| statement-2019-06-19 | "...will act as appropriate to sustain the expansion..." | dovish, medium | T3. Easing bias, with uncertainties described as having "increased" |
| statement-2024-12-18 | "In considering the extent and timing of additional adjustments..." | dovish, medium | T2. Cut with a slower-pace hint. The dissent for a hold is context only |
| statement-2021-11-03 | "...the Committee decided to begin reducing the monthly pace of its net asset purchases..." | hawkish, low | T5. Taper decision against guidance to hold at the lower bound "until" goals are met |

## Compact rubric

Label one FOMC statement **hawkish** (tightening), **neutral** (hold, balanced or data-dependent, no clear lean)
or **dovish** (easing) by the stance it communicates about the policy path at release. Use only this
statement's text: no comparison with earlier statements, hindsight, market reaction, press conference or
projections. Dissents are context, never the label.

Decide in order; the first step with a clear lean sets the label (none → neutral):

1. Rate action: hike → hawkish; cut → dovish. Opposing guidance lowers confidence, never flips.
2. Hold guidance. Hawkish: further/ongoing increases, "additional policy firming may be appropriate", a hike
   soon. Dovish: signals cuts, "act as appropriate to sustain the expansion", lower bound or accommodation
   until goals are met. No lean: "patient", "any"/"additional" adjustments without direction, next move
   tied to unmet conditions, "(only) gradual increases" without "further".
3. Balance sheet. Hawkish: starts/accelerates runoff, ends purchases, begins/speeds a taper. Dovish: starts,
   expands or maintains net purchases; slows/ends runoff for stated easing reasons. Reinvestment and
   technical/reserve operations: no lean. A decision this meeting outranks opposing rate guidance (low
   confidence).
4. Risk language: dominant upside inflation risk → hawkish; dominant downside employment/activity risk → dovish.

Confidence: high = an action (rate move, purchases, runoff start) or explicit stance statement decides and
nothing opposes, or a plainly two-sided hold; medium = hold guidance decides, or a move with opposing guidance;
low = a balance-sheet signal or risk language alone decides, or the adjacent label is defensible.

## Gold labels

The reference labels are in `data/eval/stance_gold.jsonl` ("gold" in the file name is a name, not a claim): 97 rows,
one per statement, sorted by `meeting_date`.

- **How the labels were made.** Two LLM annotators (A and B) each labeled all 97 statements independently,
  working only from the statement text and this guideline. Each recorded a label, a confidence, evidence and a
  rationale. An LLM adjudicator then compared the two sets. A, B and the adjudicator were Anthropic Claude Opus 5.5
  (`claude-opus-5-5`) subagents run on 2026-10-13, each in a fresh context; the stance classifiers evaluated against
  these labels use OpenAI `gpt-5.4-mini`, a different model family, but the same rubric. Every row has `label_source: "llm_adjudicated"`,
  keeps both annotators' labels (`annotator_a`, `annotator_b`) and quotes one or two verbatim evidence sentences.
- **Agreement.** A and B gave the same label on all 97 statements: raw agreement 1.00, Cohen's kappa 1.00
  (scikit-learn). They gave the same confidence on 89 of the 97. Both annotators are LLMs that followed the same
  rubric, so this shows the rubric is applied consistently. It is not a measure of agreement between human readers.
- **Adjudication.** There were no disagreements to resolve. The adjudicator checked the agreed labels against
  this guideline and overrode none, so every `note` is empty.
- **Final confidence.** high when A and B agreed and both said high or medium; medium when they agreed and at
  least one said low; low when they disagreed. Result: 88 high, 9 medium, 0 low. The 9 medium statements are
  2016-03-16, 2017-07-26, 2021-09-22, 2021-11-03, 2021-12-15, 2023-12-13, 2026-04-29, 2026-06-17 and 2026-07-29.
- **Label counts.** hawkish 39, neutral 33, dovish 25.
- **Human blind review.** The project owner re-labeled 25 statements blind with the compact rubric above.
  The items are in `data/eval/stance_review_items.json`, with the full text, a word-level diff against the
  previous statement and the adjudicated draft label. With no disagreements, the 9 medium-confidence statements
  were selected first. The other 16 were chosen so that every year from 2015 to 2026 appears and each label has
  at least 5 items (11 hawkish, 7 neutral, 7 dovish). They favor policy turning points (the first hikes of 2015,
  2022 and 2026, the first cuts of 2019, 2024 and 2025, March 2020, and the January 2022 "soon" guidance) and add
  borderline rubric cases, such as a "patient" hold, a cut with only data-dependent guidance and a runoff slowdown
  with no stated easing reason. The review ran on a web page that showed each statement's redline (a departure from
  "What not to use"; hide the diff in future reviews), took a label
  before revealing any draft, and locked the blind labels before the drafts appeared.
- **Human review result (2026-10-13).** The owner's blind labels matched the adjudicated draft on 23 of 25
  statements (raw agreement 0.92, Cohen's kappa 0.88, no "not sure" answers). Both disagreements are the short
  2026 statements, 2026-06-17 and 2026-07-29: the owner labeled them neutral, the draft hawkish. Neither
  statement gives rate guidance; the draft rested on step 4 (risk language) alone, and both annotators had
  marked them low confidence with neutral as the adjacent reading. The 25 were chosen to over-represent hard
  cases, so this agreement is not an estimate for the other 72 statements.
- **Effective reference label.** `human_label` where set (all 25 reviewed rows, with `human_note` recording the
  review and any draft it overrides), else the adjudicated `label`. Effective counts: hawkish 37, neutral 35,
  dovish 25. The other 72 labels are LLM-adjudicated only.
