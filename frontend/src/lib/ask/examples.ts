/** Example questions offered as chips under the Ask form. Choosing one only fills the form. */
export interface ExampleQuestion {
  /** Short chip label. */
  label: string;
  question: string;
  /** Point-in-time cutoff the example is meant to be asked with. */
  asOf?: string;
}

export const EXAMPLE_QUESTIONS: readonly ExampleQuestion[] = [
  {
    label: "March 2020 emergency cuts",
    question: "Why did the FOMC cut rates at its emergency meetings in March 2020?",
  },
  {
    label: "2022 rate hikes",
    question: "How did the Committee justify the 75 basis point rate increases in 2022?",
  },
  {
    label: "Banking stress after SVB",
    question:
      "How did the FOMC assess the banking-sector stress after Silicon Valley Bank failed, and what did it expect the effect on the economy to be?",
    asOf: "2023-04-30",
  },
  {
    label: "2024 rate cuts",
    question: "Why did the FOMC begin lowering the federal funds rate in September 2024?",
  },
  {
    label: "Balance-sheet runoff",
    question: "How and why did the FOMC slow the pace of its balance sheet runoff?",
  },
  {
    label: "Tariffs in 2025 minutes",
    question: "What did FOMC participants say about the effects of tariffs on inflation in 2025?",
  },
];
