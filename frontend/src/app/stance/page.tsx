import type { Metadata } from "next";
import { Suspense } from "react";

import { PageHeader } from "@/components/PageHeader";
import { StanceExplorer } from "@/components/stance/StanceExplorer";
import { StanceSkeleton } from "@/components/stance/StanceSkeleton";
import { pageMetadata } from "@/lib/site";

export const metadata: Metadata = pageMetadata({
  title: "Policy Stance",
  description:
    "Hawkish, neutral or dovish: every FOMC statement since 2015 classified by four methods, plotted against the federal funds target range and benchmarked.",
  path: "/stance",
});

export default function StancePage() {
  return (
    <>
      <PageHeader title="Policy Stance">
        Every FOMC statement since 2015, rated from dovish (leaning toward lower rates) to hawkish
        (toward higher rates) by four methods, next to the policy rate.
      </PageHeader>

      {/* StanceExplorer reads the query string (useSearchParams), so it renders on the client
          only: the prerendered page shows StanceSkeleton until hydration. */}
      <Suspense fallback={<StanceSkeleton />}>
        <StanceExplorer />
      </Suspense>
    </>
  );
}
