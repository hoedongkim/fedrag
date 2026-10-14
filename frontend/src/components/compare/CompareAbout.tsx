import { About, AboutList } from "../ui/About";

/** Statement Comparison's method and caveats, closed at the end of the page. */
export function CompareAbout({ modelVersion, demo }: { modelVersion: string | null; demo: boolean }) {
  return (
    <About title="About this comparison">
      <AboutList
        items={[
          {
            term: "How sentences are matched",
            body: "Both statements are split into sentences and lined up in order. A sentence close enough to one in the other statement counts as modified, with the changed words marked; the rest count as added or removed. Similarity measures how alike the wording is: 100% means identical.",
          },
          {
            term: "The analysis",
            body: (
              <>
                Written by GPT-5.4 mini from the wording shown here, told to read it as of the later statement&rsquo;s
                release and to ignore later events and market reactions (it cannot fully forget what it learned in
                training). It is an interpretation and can be wrong, so check it against the marked sentences. Not
                investment advice.
                {modelVersion && (
                  <>
                    {" "}
                    Model and prompt: <code className="break-all">{modelVersion}</code>.
                  </>
                )}
              </>
            ),
          },
          {
            term: "Confidence and stance scores",
            body: "The confidence is the model’s own, not calibrated. The stance scores come from the LLM zero-shot method on Policy Stance, each statement scored on its own: a cross-check from the same model family, not independent evidence.",
          },
          {
            term: "Stored analyses",
            body: demo
              ? "The public demo calls no LLM, so it shows only the analyses stored for back-to-back meetings."
              : "An analysis is written once (one OpenAI call, two if the reply cannot be read) and then stored, so a pair opens with it from then on.",
          },
        ]}
      />
    </About>
  );
}
