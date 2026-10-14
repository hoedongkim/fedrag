"use client";

import { backendStore, demoModeOf } from "@/lib/api/backend";
import { DEMO_UNREACHABLE } from "@/lib/api/hosting";

import { FailureNotice } from "../stance/FailureNotice";
import { useBackend } from "../useBackend";
import { AskExperience } from "./AskExperience";
import { AskSkeleton } from "./AskSkeleton";
import { RecordedAnswers } from "./RecordedAnswers";

/**
 * The live form, or the public demo's recorded answers, depending on GET /api/health: neither is
 * shown until the backend has said which mode it runs in, so the live form never flashes on the
 * demo. A local backend that is down gets the live form (its errors say how to start it).
 */
export function AskGate() {
  const backend = useBackend();
  const demo = demoModeOf(backend);

  if (demo === true) return <RecordedAnswers />;
  if (demo === false) return <AskExperience />;
  if (backend.phase === "offline") {
    return (
      <div className="mt-10">
        <FailureNotice failure={DEMO_UNREACHABLE} onRetry={backendStore.recheck} />
      </div>
    );
  }
  return <AskSkeleton />;
}
