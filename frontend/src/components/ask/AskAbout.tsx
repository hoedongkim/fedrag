import { About } from "../ui/About";

/** Ask the Fed's methodology and caveats, closed at the end of the page. */
export function AskAbout({ recorded }: { recorded: boolean }) {
  return (
    <About title="About these answers">
      <p>
        Answers come only from FOMC statements and minutes, 2015 to 2026. A search picks the most
        relevant passages, and a language model writes the answer from those passages alone.
      </p>
      <p>
        Each numbered marker links a claim to its passage. A passage&rsquo;s score is how relevant the
        search found it; compare scores only within one answer.
      </p>
      <p>
        With an as-of date, only documents that were public that day are searched. Minutes come out
        about three weeks after each meeting.
      </p>
      {recorded && (
        <p>
          Recorded answers were graded during testing by an LLM judge, which was itself checked
          against Claude annotators on a sample, not against people.
        </p>
      )}
      <p>
        The model can still misread a passage or miss context, so check the sources before relying
        on an answer. Educational project; not investment advice.
      </p>
    </About>
  );
}
