import type { Metadata } from "next";
import { Suspense } from "react";

import { PageHeader } from "@/components/PageHeader";
import { AskGate } from "@/components/ask/AskGate";
import { AskSkeleton } from "@/components/ask/AskSkeleton";
import { pageMetadata } from "@/lib/site";

export const metadata: Metadata = pageMetadata({
  title: "Ask the Fed",
  description:
    "Ask questions about Federal Reserve monetary policy and get streamed answers grounded in FOMC statements and minutes, with a citation for every claim.",
  path: "/ask",
});

export default function AskPage() {
  return (
    <>
      <PageHeader title="Ask the Fed">
        Ask about Fed policy and get an answer drawn only from FOMC statements and minutes, with a
        source for every claim.
      </PageHeader>

      {/* AskGate waits for GET /api/health, then shows the live form (AskExperience) or, on the
          public demo, recorded answers. Both read the query string (useSearchParams), so they
          render on the client only: the prerendered page shows AskSkeleton until then. */}
      <Suspense fallback={<AskSkeleton />}>
        <AskGate />
      </Suspense>
    </>
  );
}
