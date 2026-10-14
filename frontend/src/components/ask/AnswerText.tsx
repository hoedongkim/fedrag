"use client";

import { useCallback, useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

import type { Citation } from "@/lib/api/types";
import { formatDate } from "@/lib/ask/format";
import { excerpt, groupCitationRuns, splitCitationMarkers, splitParagraphs } from "@/lib/ask/markers";

interface AnswerTextProps {
  text: string;
  /** Citations by their [n] number; markers without one yet render as plain numbers. */
  citations: ReadonlyMap<number, Citation>;
  /** Still streaming: show a caret and hold back an unfinished trailing marker. */
  streaming?: boolean;
  /** A marker was activated: show citation n. */
  onCite: (n: number) => void;
  className?: string;
}

const markerClass =
  "mx-px inline-flex h-[1.15rem] min-w-[1.15rem] items-center justify-center rounded-sm px-1 align-baseline font-sans text-[0.6875rem] leading-none font-semibold tabular-nums";

const HIDE_DELAY_MS = 120;
const GAP_PX = 6;
const EDGE_PX = 8;

interface Tip {
  n: number;
  anchor: HTMLElement;
}

/**
 * Put the tooltip above its marker (below when there is no room), clamped to the viewport. It is
 * position: fixed, so it never widens the page.
 */
function placeTooltip(tooltip: HTMLElement, anchorElement: HTMLElement) {
  const anchor = anchorElement.getBoundingClientRect();
  const box = tooltip.getBoundingClientRect();
  const viewportWidth = document.documentElement.clientWidth;
  const centered = anchor.left + anchor.width / 2 - box.width / 2;
  const left = Math.max(EDGE_PX, Math.min(centered, viewportWidth - box.width - EDGE_PX));
  const above = anchor.top - box.height - GAP_PX;
  const top = above >= EDGE_PX ? above : anchor.bottom + GAP_PX;
  tooltip.style.left = `${left}px`;
  tooltip.style.top = `${top}px`;
  tooltip.style.visibility = "visible";
}

/**
 * Answer paragraphs with [n] markers as superscript buttons. Hovering or focusing one shows a
 * preview of the passage (section, release date, first ~200 characters) in a tooltip that stays
 * inside the viewport (portaled to <body>, so no ancestor clips or offsets it); activating a
 * marker calls `onCite`.
 */
export function AnswerText({ text, citations, streaming = false, onCite, className = "" }: AnswerTextProps) {
  const tooltipId = useId();
  const tooltipRef = useRef<HTMLDivElement>(null);
  const hideTimer = useRef<number | undefined>(undefined);
  const [tip, setTip] = useState<Tip | null>(null);

  const cancelHide = useCallback(() => window.clearTimeout(hideTimer.current), []);
  const show = useCallback(
    (n: number, anchor: HTMLElement) => {
      cancelHide();
      setTip({ n, anchor });
    },
    [cancelHide],
  );
  const hide = useCallback(() => {
    cancelHide();
    setTip(null);
  }, [cancelHide]);
  const hideSoon = useCallback(() => {
    cancelHide();
    hideTimer.current = window.setTimeout(() => setTip(null), HIDE_DELAY_MS);
  }, [cancelHide]);

  // Place it when it opens, and keep it on its marker while the page scrolls (focusing a marker
  // can itself scroll it into view) or resizes. Escape dismisses it.
  useLayoutEffect(() => {
    const tooltip = tooltipRef.current;
    if (!tip || !tooltip) return;
    placeTooltip(tooltip, tip.anchor);

    let frame = 0;
    const follow = () => {
      window.cancelAnimationFrame(frame);
      frame = window.requestAnimationFrame(() => placeTooltip(tooltip, tip.anchor));
    };
    const onKeyDown = (event: globalThis.KeyboardEvent) => {
      if (event.key === "Escape") hide();
    };
    window.addEventListener("scroll", follow, { capture: true, passive: true });
    window.addEventListener("resize", follow);
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.cancelAnimationFrame(frame);
      window.removeEventListener("scroll", follow, { capture: true });
      window.removeEventListener("resize", follow);
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [tip, hide]);

  useEffect(() => cancelHide, [cancelHide]);

  const paragraphs = splitParagraphs(text);
  const tipCitation = tip ? citations.get(tip.n) : undefined;

  const marker = (n: number, key: number) => {
    const citation = citations.get(n);
    if (!citation) {
      return (
        <sup key={key}>
          <span className={`${markerClass} bg-subtle text-faint`}>
            <span className="sr-only">citation </span>
            {n}
          </span>
        </sup>
      );
    }
    return (
      <sup key={key}>
        <button
          type="button"
          aria-label={`Citation ${n}: ${citation.document.title}`}
          aria-describedby={tip?.n === n ? tooltipId : undefined}
          onClick={() => {
            hide();
            onCite(n);
          }}
          onPointerEnter={(event) => {
            if (event.pointerType === "mouse") show(n, event.currentTarget);
          }}
          onPointerLeave={(event) => {
            if (event.pointerType === "mouse") hideSoon();
          }}
          onFocus={(event) => show(n, event.currentTarget)}
          onBlur={hide}
          className={`${markerClass} bg-accent-soft text-accent transition-colors hover:bg-accent hover:text-canvas`}
        >
          {n}
        </button>
      </sup>
    );
  };

  return (
    <>
      <div className={`space-y-4 [overflow-wrap:anywhere] ${className}`}>
        {paragraphs.map((paragraph, index) => {
          const last = index === paragraphs.length - 1;
          return (
            <p key={index} className="whitespace-pre-line">
              {groupCitationRuns(splitCitationMarkers(paragraph, { streaming: streaming && last })).map(
                (piece, i) =>
                  piece.kind === "text" ? (
                    piece.text
                  ) : (
                    // A run of markers and its closing punctuation never wrap apart.
                    <span key={i} className="whitespace-nowrap">
                      {piece.ns.map(marker)}
                      {piece.after}
                    </span>
                  ),
              )}
              {streaming && last && (
                <span
                  aria-hidden="true"
                  className="ml-0.5 inline-block h-[1.05em] w-0.5 translate-y-[0.15em] rounded-full bg-accent motion-safe:animate-pulse"
                />
              )}
            </p>
          );
        })}
      </div>

      {tip &&
        tipCitation &&
        createPortal(
          <div
            ref={tooltipRef}
            id={tooltipId}
            role="tooltip"
            style={{ position: "fixed", top: 0, left: 0, visibility: "hidden" }}
            onPointerEnter={cancelHide}
            onPointerLeave={hideSoon}
            className="z-50 w-80 max-w-[calc(100vw-1rem)] rounded-md border border-line-strong bg-canvas p-3 text-left font-sans shadow-lg"
          >
            <p className="text-label font-semibold text-ink">
              <span className="text-accent tabular-nums">{tip.n}</span> {tipCitation.document.title}
            </p>
            <p className="mt-0.5 text-label text-faint">
              {tipCitation.section ? `${tipCitation.section} · ` : ""}Released{" "}
              {formatDate(tipCitation.document.release_date)}
            </p>
            <p className="mt-2 text-label leading-5 text-muted">
              {excerpt(tipCitation.snippet, 200)}
            </p>
          </div>,
          document.body,
        )}
    </>
  );
}
