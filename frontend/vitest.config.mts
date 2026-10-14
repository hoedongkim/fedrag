import { fileURLToPath } from "node:url";

import { defineConfig } from "vitest/config";

// Unit tests for pure logic only (SSE parsing, answer markers, URL state): Node, no DOM.
export default defineConfig({
  resolve: {
    alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
  },
  test: {
    include: ["src/**/*.test.ts"],
    environment: "node",
  },
});
