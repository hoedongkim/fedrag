import { useId, useState } from "react";

import type { Citation, RetrievalMethod } from "@/lib/api/types";
import { displayHost, formatDate, formatScore, isRevised } from "@/lib/ask/format";
import { methodLabel } from "@/lib/ask/params";

import { ExternalLink } from "../ui/ExternalLink";
import { textButtonClass } from "../ui/styles";

interface CitationCardProps {
  n: number;
  citation: Citation;
  /** Element id, the target of the answer's [n] marker. */
  id: string;
  /** Retrieval method that produced the score, for its tooltip. */
  method: RetrievalMethod | undefined;
  /** Briefly emphasised after its marker was activated. */
  highlighted?: boolean;
}

export function CitationCard({ n, citation, id, method, highlighted = false }: CitationCardProps) {
  const titleId = useId();
  const passageId = useId();
  const { document: doc, section, snippet, passage, score } = citation;
  // A snippet is the passage's first ~600 characters; a recorded answer also carries the whole
  // passage, whose later sentences often hold what the answer cites.
  const fuller = passage && passage.length > snippet.length ? passage : null;
  const [expanded, setExpanded] = useState(false);
  // Following the answer's [n] marker opens the whole passage: that is what the reader came for.
  const [wasHighlighted, setWasHighlighted] = useState(highlighted);
  if (highlighted !== wasHighlighted) {
    setWasHighlighted(highlighted);
    if (highlighted && fuller) setExpanded(true);
  }
  const revised = isRevised(doc.release_date, doc.page_updated);
  const host = displayHost(doc.url) || "the source site";

  return (
    <article
      id={id}
      tabIndex={-1}
      aria-labelledby={titleId}
      className={[
        "-mx-3 scroll-mt-6 rounded-md px-3 py-5 transition-colors duration-300 motion-safe:animate-fade-in motion-reduce:transition-none",
        highlighted ? "bg-accent-soft" : "",
      ].join(" ")}
    >
      <div className="flex items-baseline gap-3">
        <span aria-hidden="true" className="w-5 shrink-0 text-small font-semibold text-accent tabular-nums">
          {n}
        </span>
        <div className="min-w-0 flex-1">
          <h3 id={titleId} className="text-body font-semibold break-words text-ink">
            <span className="sr-only">Citation {n}: </span>
            {doc.title}
          </h3>
          {/* Inline flow, not flex: the section wraps word by word, and a no-break space keeps the
              separator on the line of the section's first word. */}
          <p className="mt-0.5 text-small break-words text-faint">
            Released <time dateTime={doc.release_date}>{formatDate(doc.release_date)}</time>
            {section && (
              <>
                {" "}
                <span aria-hidden="true">·</span>
                {"\u00a0"}
                {section}
              </>
            )}
            {revised && doc.page_updated && (
              <span
                title={`The Fed updated this page on ${formatDate(doc.page_updated)}, after its release on ${formatDate(doc.release_date)}. The passage is from the updated page, which may differ from what was public at release.`}
                className="cursor-help text-warn"
              >
                {" "}
                <span aria-hidden="true">·</span>
                {"\u00a0"}Revised after release
                <span className="sr-only">
                  {`: the page was updated on ${formatDate(doc.page_updated)}, so the passage may differ from the text released on ${formatDate(doc.release_date)}.`}
                </span>
              </span>
            )}
          </p>
        </div>
        {score !== null && (
          <span
            title={`Search relevance${method ? ` (${methodLabel(method)})` : ""}: higher is more relevant. Its scale depends on the method, so compare scores only within one answer.`}
            className="shrink-0 cursor-help text-label text-faint tabular-nums"
          >
            <span className="sr-only">Retrieval score </span>
            <span aria-hidden="true">score </span>
            {formatScore(score)}
          </span>
        )}
      </div>

      <blockquote
        id={passageId}
        className="mt-3 max-w-[70ch] text-body break-words whitespace-pre-line text-muted sm:ml-8"
      >
        {fuller && expanded ? fuller : snippet}
        {fuller && !expanded && <span aria-hidden="true"> …</span>}
      </blockquote>

      <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-2 sm:ml-8">
        {fuller && (
          <button
            type="button"
            aria-expanded={expanded}
            aria-controls={passageId}
            onClick={() => setExpanded((open) => !open)}
            className={textButtonClass}
          >
            {expanded ? "Show less" : "Show full passage"}
          </button>
        )}
        <ExternalLink href={doc.url} className={textButtonClass}>
          Open on {host}
        </ExternalLink>
      </div>
    </article>
  );
}
