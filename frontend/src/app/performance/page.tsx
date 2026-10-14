import type { Metadata } from "next";
import { Suspense } from "react";

import { PageHeader } from "@/components/PageHeader";
import { PerformanceExplorer } from "@/components/performance/PerformanceExplorer";
import { PerformanceSkeleton } from "@/components/performance/PerformanceSkeleton";
import { pageMetadata } from "@/lib/site";

export const metadata: Metadata = pageMetadata({
  title: "RAG Performance",
  description:
    "How well BM25, dense, hybrid and reranked retrieval find FOMC passages, and how grounded, cited and correct the answers built on them are, with bootstrap intervals and a browser into every question.",
  path: "/performance",
});

export default function PerformancePage() {
  return (
    <>
      <PageHeader title="RAG Performance">
        How often Ask the Fed&rsquo;s search finds the right passages, how accurate its answers are,
        and what each costs, on questions set aside for testing.
      </PageHeader>

      {/* PerformanceExplorer reads the query string (useSearchParams), so it renders on the
          client only: the prerendered page shows PerformanceSkeleton until hydration. */}
      <Suspense fallback={<PerformanceSkeleton />}>
        <PerformanceExplorer />
      </Suspense>
    </>
  );
}
