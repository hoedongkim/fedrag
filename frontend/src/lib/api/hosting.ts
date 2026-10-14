/**
 * Where the backend runs. The public demo's API sits on free hosting (a Hugging Face Space) that
 * sleeps when idle and takes a minute or two to wake, so the pages wait for it instead of showing
 * an error. A local backend that does not answer is simply not running, and the pages say how to
 * start it.
 */

/**
 * True when the backend may be asleep. next.config.ts sets FEDRAG_BACKEND_MAY_SLEEP at build time:
 * true for any BACKEND_URL that is not on this machine, unless BACKEND_MAY_SLEEP overrides it.
 * Read on each call, so tests can set it.
 */
export function backendMaySleep(): boolean {
  return process.env.FEDRAG_BACKEND_MAY_SLEEP === "true";
}

/** A request failed after the wait for a sleeping backend ran out (the hosted demo only). */
export const DEMO_UNREACHABLE = {
  tone: "error",
  title: "The demo server is not responding",
  message:
    "Its free hosting sleeps when idle and usually wakes within a minute or two, but it has not answered yet. Try again in a moment.",
} as const;
