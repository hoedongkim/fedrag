import type { ReactNode } from "react";

import { ExternalLinkIcon } from "../ask/icons";
import { linkClass } from "./styles";

/**
 * A link that opens in a new tab, with a small arrow and the screen-reader note. Inline flow (no
 * flex), so the arrow stays with the last word when the text wraps. Pass `className` to replace
 * the default inline-link look (e.g. a button class).
 */
export function ExternalLink({
  href,
  className = linkClass,
  children,
}: {
  href: string;
  className?: string;
  children: ReactNode;
}) {
  return (
    <a href={href} target="_blank" rel="noopener noreferrer" className={className}>
      {children}
      <ExternalLinkIcon className="ml-0.5 inline size-[0.85em] shrink-0 align-[-0.05em]" />
      <span className="sr-only"> (opens in a new tab)</span>
    </a>
  );
}
