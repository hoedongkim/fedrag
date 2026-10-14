import type { Metadata } from "next";
import { Suspense } from "react";

import { PageHeader } from "@/components/PageHeader";
import { CompareExperience } from "@/components/compare/CompareExperience";
import { CompareSkeleton } from "@/components/compare/CompareSkeleton";
import { pageMetadata } from "@/lib/site";

export const metadata: Metadata = pageMetadata({
  title: "Statement Comparison",
  description:
    "Compare two FOMC statements sentence by sentence, with word-level changes and an LLM reading of what the change in wording signals about policy.",
  path: "/compare",
});

export default function ComparePage() {
  return (
    <>
      <PageHeader title="Statement Comparison">
        Pick two FOMC statements to see which sentences were added, removed or reworded, and what
        the change may signal.
      </PageHeader>

      {/* CompareExperience reads the query string (useSearchParams), so it renders on the client
          only: the prerendered page shows CompareSkeleton until hydration. */}
      <Suspense fallback={<CompareSkeleton />}>
        <CompareExperience />
      </Suspense>
    </>
  );
}
