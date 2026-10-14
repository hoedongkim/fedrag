import type { NextConfig } from "next";

// Base URL of the FastAPI backend (no trailing slash, no /api suffix).
// Prefer 127.0.0.1 over "localhost": Node may resolve localhost to ::1 while
// uvicorn binds IPv4 only. Read when this config loads (`next dev` start or
// `next build`), so restart or rebuild after changing it.
const BACKEND_URL = (process.env.BACKEND_URL || "http://127.0.0.1:8000").replace(
  /\/+$/,
  "",
);

/** True when the backend runs on this machine (or this machine's Docker), not on a host. */
function isLocalUrl(url: string): boolean {
  try {
    const { hostname } = new URL(url);
    return ["127.0.0.1", "localhost", "[::1]", "::1", "0.0.0.0"].includes(hostname) || hostname.endsWith(".localhost");
  } catch {
    return true;
  }
}

// A hosted backend may need a moment before it answers: the public demo's Vercel function after a
// quiet spell (a cold start of a few seconds), or a Hugging Face Space, the alternative host, that
// sleeps after two idle days (a minute or two). The pages then wait for it ("Waking the demo
// server") instead of showing an error. On by default for any non-local BACKEND_URL;
// BACKEND_MAY_SLEEP=true or false overrides that. Inlined into the browser bundle at build time.
const BACKEND_MAY_SLEEP = process.env.BACKEND_MAY_SLEEP
  ? process.env.BACKEND_MAY_SLEEP === "true"
  : !isLocalUrl(BACKEND_URL);

const nextConfig: NextConfig = {
  // Don't let `next dev` write AGENTS.md into the project.
  agentRules: false,
  // `next dev` only trusts localhost by default; also allow http://127.0.0.1:3000
  // so HMR and hydration work there too.
  allowedDevOrigins: ["127.0.0.1"],
  env: {
    FEDRAG_BACKEND_MAY_SLEEP: String(BACKEND_MAY_SLEEP),
  },
  cacheComponents: true,
  partialPrefetching: true,
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
  // Browser code calls relative /api/... URLs; Next proxies them to FastAPI,
  // so no CORS is involved. Do not add route handlers under src/app/api:
  // they would shadow this rewrite. The proxy also carries the answer stream
  // (POST /api/ask/stream): Next gzips compressible responses, which would
  // hold the events back until the answer is complete, unless they are sent
  // with Cache-Control: no-transform, as the backend does for event streams.
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${BACKEND_URL}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
