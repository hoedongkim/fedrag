"use client";

import { useEffect, useState } from "react";

import { Spinner } from "./ask/icons";
import { useBackend } from "./useBackend";

/**
 * Shown above every page while the hosted demo's backend starts: after a quiet spell the first
 * request waits for it (a cold start). The pages keep their loading state and fill in on their own
 * when it answers, so this replaces an error, not the page.
 */
export function WakeNotice() {
  const { phase, wakingSince } = useBackend();
  const waking = phase === "waking" && wakingSince !== null;
  const seconds = useElapsedSeconds(waking ? wakingSince : null);

  if (!waking) return null;
  return (
    <p role="status" className="mb-8 flex items-start gap-2 text-small text-muted motion-safe:animate-fade-in">
      <Spinner className="mt-[0.2rem] size-3.5 shrink-0 text-warn" />
      <span>
        Waking the demo server after a quiet spell. This usually takes a few seconds; the page fills
        in by itself
        {/* Hidden from screen readers: a live region would read every tick. */}
        {seconds >= 5 && (
          <span aria-hidden="true" className="text-faint tabular-nums">
            {" "}
            · {seconds} s
          </span>
        )}
        .
      </span>
    </p>
  );
}

/** Whole seconds since `since`, ticking once a second; 0 when null or before the first tick. */
function useElapsedSeconds(since: number | null): number {
  // Only read the clock in the interval: never during render (or prerender).
  const [now, setNow] = useState<number | null>(null);
  useEffect(() => {
    if (since === null) return;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [since]);
  return since === null || now === null ? 0 : Math.max(0, Math.floor((now - since) / 1000));
}
