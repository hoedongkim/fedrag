/** Site-wide names, links and page metadata. */
import type { Metadata } from "next";

export const SITE_NAME = "FedRAG";
export const SITE_TITLE = "FedRAG - Cited answers from FOMC communications";
export const SITE_DESCRIPTION =
  "Ask what the Federal Reserve actually said and get answers built only from FOMC statements and minutes, with a citation for every claim; the policy stance of every statement since 2015; sentence-level comparisons of any two statements; and an evaluation of it all on a held-out test split.";

export const REPO_URL = "https://github.com/hoedongkim/fedrag";

/** The link-preview image (app/opengraph-image.tsx and app/twitter-image.tsx). */
export const SHARE_IMAGE_SIZE = { width: 1200, height: 630 };
export const SHARE_IMAGE_ALT =
  "FedRAG: ask what the Federal Reserve actually said, with answers built only from FOMC statements and minutes and a citation for every claim.";

/** The technical report on GitHub (docs/technical-report.md), which holds the long write-up. */
export const TECHNICAL_REPORT_URL = `${REPO_URL}/blob/main/docs/technical-report.md`;

/**
 * Sections of the technical report on GitHub that the pages link to (anchors of the report's
 * emoji headings start with "-").
 */
export const DOCS = {
  quickstart: `${TECHNICAL_REPORT_URL}#-run-it-yourself`,
  architecture: `${TECHNICAL_REPORT_URL}#-architecture`,
  evaluation: `${TECHNICAL_REPORT_URL}#-rag-evaluation`,
  limitations: `${TECHNICAL_REPORT_URL}#limitations`,
} as const;

/**
 * The deployed site's origin, for absolute Open Graph URLs: SITE_URL if set, else Vercel's
 * production domain, else the local dev server. Read at build time.
 */
export function siteUrl(env: Record<string, string | undefined> = process.env): URL {
  const vercel = env.VERCEL_PROJECT_PRODUCTION_URL;
  const candidates = [env.SITE_URL, vercel ? `https://${vercel}` : undefined, "http://localhost:3000"];
  for (const candidate of candidates) {
    if (!candidate) continue;
    try {
      return new URL(candidate);
    } catch {
      // not a URL: try the next one
    }
  }
  return new URL("http://localhost:3000");
}

/**
 * A page's title, description and the matching Open Graph and Twitter tags. A page that sets its
 * own `openGraph` replaces the root layout's, image included, so the share image (from
 * app/opengraph-image.tsx and app/twitter-image.tsx) is named here again.
 */
export function pageMetadata({ title, description, path }: { title: string; description: string; path: string }): Metadata {
  const fullTitle = `${title} | ${SITE_NAME}`;
  return {
    title,
    description,
    alternates: { canonical: path },
    openGraph: {
      type: "website",
      siteName: SITE_NAME,
      locale: "en_US",
      url: path,
      title: fullTitle,
      description,
      images: [{ url: "/opengraph-image", ...SHARE_IMAGE_SIZE, alt: SHARE_IMAGE_ALT, type: "image/png" }],
    },
    twitter: {
      card: "summary_large_image",
      title: fullTitle,
      description,
      images: [{ url: "/twitter-image", alt: SHARE_IMAGE_ALT }],
    },
  };
}
