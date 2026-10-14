/** Display formatting for Ask the Fed. Fixed locale and UTC, so server and client render alike. */

const DATE_FORMAT = new Intl.DateTimeFormat("en-US", {
  year: "numeric",
  month: "short",
  day: "numeric",
  timeZone: "UTC",
});

/** "2023-03-22" -> "Mar 22, 2023"; anything unparseable is returned as is. */
export function formatDate(isoDate: string): string {
  const date = new Date(`${isoDate}T00:00:00Z`);
  return Number.isNaN(date.getTime()) ? isoDate : DATE_FORMAT.format(date);
}

/** 850 -> "850 ms", 4213.7 -> "4.2 s". */
export function formatLatency(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return "-";
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`;
}

/**
 * A retrieval score at a precision that fits its scale: BM25 scores run to tens, similarities
 * and cross-encoder scores lie in 0-1, and fused (RRF) scores are a few hundredths.
 */
export function formatScore(score: number): string {
  const abs = Math.abs(score);
  if (abs >= 10) return score.toFixed(1);
  if (abs >= 1) return score.toFixed(2);
  if (abs === 0) return "0";
  return String(Number(score.toPrecision(3)));
}

/** True when the page was updated after the document's release (both YYYY-MM-DD). */
export function isRevised(releaseDate: string, pageUpdated: string | null): boolean {
  return pageUpdated !== null && pageUpdated > releaseDate;
}

/** Today's date in the viewer's time zone, as YYYY-MM-DD. */
export function localToday(now: Date = new Date()): string {
  const month = String(now.getMonth() + 1).padStart(2, "0");
  const day = String(now.getDate()).padStart(2, "0");
  return `${now.getFullYear()}-${month}-${day}`;
}

/** The host of a URL without "www.", e.g. "federalreserve.gov"; "" if it does not parse. */
export function displayHost(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return "";
  }
}
