import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import { DOCS, pageMetadata, siteUrl, TECHNICAL_REPORT_URL } from "./site";

/**
 * The anchors GitHub gives a Markdown file: its headings' slugs (lowercased, punctuation and emoji
 * dropped, spaces to "-"; ASCII headings only) and explicit `<a id>`s, outside code fences.
 */
function githubAnchors(markdown: string): Set<string> {
  const anchors = new Set<string>();
  let inFence = false;
  for (const line of markdown.split("\n")) {
    if (line.startsWith("```")) inFence = !inFence;
    if (inFence) continue;
    const heading = /^#{1,6} (.+)$/.exec(line);
    if (heading) anchors.add(heading[1].trim().toLowerCase().replace(/[^a-z0-9 _-]/g, "").replace(/ /g, "-"));
    for (const [, id] of line.matchAll(/<a id="([^"]+)">/g)) anchors.add(id);
  }
  return anchors;
}

describe("siteUrl", () => {
  it("prefers SITE_URL, then Vercel's production domain, then the dev server", () => {
    expect(siteUrl({ SITE_URL: "https://fedrag.example", VERCEL_PROJECT_PRODUCTION_URL: "x.vercel.app" }).href).toBe(
      "https://fedrag.example/",
    );
    expect(siteUrl({ VERCEL_PROJECT_PRODUCTION_URL: "fedrag.vercel.app" }).href).toBe("https://fedrag.vercel.app/");
    expect(siteUrl({ SITE_URL: "not a url" }).href).toBe("http://localhost:3000/");
    expect(siteUrl({}).href).toBe("http://localhost:3000/");
  });
});

describe("pageMetadata", () => {
  it("keeps the share image on pages that set their own Open Graph tags", () => {
    const meta = pageMetadata({ title: "Ask the Fed", description: "d", path: "/ask" });
    expect(meta.title).toBe("Ask the Fed");
    expect(meta.openGraph).toMatchObject({ title: "Ask the Fed | FedRAG", url: "/ask" });
    expect(JSON.stringify(meta.openGraph)).toContain("/opengraph-image");
    expect(JSON.stringify(meta.twitter)).toContain("/twitter-image");
  });
});

describe("docs links", () => {
  it("point at the technical report's sections", () => {
    expect(DOCS.quickstart).toBe("https://github.com/hoedongkim/fedrag/blob/main/docs/technical-report.md#-run-it-yourself");
    expect(DOCS.limitations).toBe("https://github.com/hoedongkim/fedrag/blob/main/docs/technical-report.md#limitations");
  });

  it("name anchors that exist in docs/technical-report.md", () => {
    const report = readFileSync(new URL("../../../docs/technical-report.md", import.meta.url), "utf8");
    const anchors = githubAnchors(report);
    for (const href of Object.values(DOCS)) {
      const [page, anchor] = href.split("#");
      expect(page).toBe(TECHNICAL_REPORT_URL);
      expect(anchors, href).toContain(anchor);
    }
  });
});
