import type { ReactNode } from "react";

import { ChevronRightIcon } from "../ask/icons";

/**
 * The one place a page keeps its methodology and caveats: a closed disclosure at the end of the
 * page ("About these numbers"), so the numbers above it can stay short. Children are short
 * paragraphs, or an <AboutList>.
 */
export function About({
  title = "About these numbers",
  className = "",
  children,
}: {
  title?: string;
  className?: string;
  children: ReactNode;
}) {
  return (
    <details className={`group mt-16 border-t border-line pt-5 ${className}`}>
      <summary className="inline-flex list-none items-center gap-1.5 rounded-sm text-small font-medium text-muted transition-colors select-none hover:text-ink">
        <ChevronRightIcon className="size-3.5 transition-transform group-open:rotate-90 motion-reduce:transition-none" />
        {title}
      </summary>
      <div className="mt-4 max-w-[70ch] space-y-3 text-small text-muted">{children}</div>
    </details>
  );
}

/** Term-and-explanation pairs inside <About>. */
export function AboutList({ items }: { items: { term: ReactNode; body: ReactNode }[] }) {
  return (
    <dl className="space-y-3">
      {items.map((item, i) => (
        <div key={i}>
          <dt className="font-medium text-ink">{item.term}</dt>
          <dd className="mt-0.5">{item.body}</dd>
        </div>
      ))}
    </dl>
  );
}
