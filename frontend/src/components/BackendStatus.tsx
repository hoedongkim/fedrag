"use client";

import type { BackendPhase } from "@/lib/api/backend";
import { backendMaySleep } from "@/lib/api/hosting";

import { useBackend } from "./useBackend";

const DOT_TONES: Record<BackendPhase | "setup", string> = {
  checking: "bg-faint motion-safe:animate-pulse",
  waking: "bg-warn motion-safe:animate-pulse",
  online: "bg-good",
  setup: "bg-warn",
  offline: "bg-bad",
};

/**
 * The header's API status, from the shared health check (see lib/api/backend.ts): a dot and a
 * word. The details (version, demo mode, what a local setup is missing) are in its title and,
 * for screen readers, in hidden text.
 */
export function BackendStatus() {
  const { phase, health } = useBackend();
  const hosted = backendMaySleep();
  const demo = health?.demo_mode === true;

  // The demo needs neither an LLM key nor (for its recorded answers) a fresh index check.
  const missing =
    phase === "online" && health && !demo
      ? [
          !health.llm_configured && "LLM key not set (OPENAI_API_KEY in backend/.env)",
          !health.vector_store_ready && "search index not built (run make index)",
        ].filter((item): item is string => Boolean(item))
      : [];

  const label =
    phase === "online"
      ? demo
        ? "Demo"
        : missing.length > 0
          ? "Setup needed"
          : "Online"
      : { checking: "Connecting", waking: "Waking", offline: "Offline" }[phase];

  const title =
    phase === "online" && health
      ? [
          `FedRAG API ${health.version} (${health.environment})`,
          demo ? "Public demo: live questions and new analyses are off, so it costs nothing to run" : null,
          ...missing,
        ]
          .filter(Boolean)
          .join(". ")
      : phase === "waking"
        ? "Free hosting sleeps when idle; waking it can take a minute or two"
        : phase === "offline"
          ? hosted
            ? "The demo server did not answer; try again in a moment"
            : "Start the backend: make dev-backend"
          : "Contacting the FedRAG API";

  return (
    <div role="status" aria-live="polite" title={title} className="inline-flex items-center gap-2 text-small text-muted">
      <span aria-hidden="true" className={`size-1.5 rounded-full ${DOT_TONES[missing.length > 0 ? "setup" : phase]}`} />
      <span>
        <span className="sr-only">API status: </span>
        {label}
        {phase === "online" && demo && <span className="sr-only"> (public demo: live questions are off)</span>}
        {missing.length > 0 && <span className="sr-only">: {missing.join("; ")}</span>}
        {phase === "offline" && !hosted && <span className="sr-only"> (start the backend: make dev-backend)</span>}
      </span>
    </div>
  );
}
