"use client";

import { useDemoMode } from "../useBackend";

/** On the public demo only: one quiet line on what is off. Nothing until the backend has answered. */
export function DemoNote() {
  const demo = useDemoMode();
  if (demo !== true) return null;
  return (
    <p className="mt-1.5 max-w-[46rem] text-small text-faint motion-safe:animate-fade-in">
      <span className="font-medium text-muted">Public demo:</span> no paid API calls, so Ask the Fed shows recorded
      answers. Everything else is the real data.
    </p>
  );
}
