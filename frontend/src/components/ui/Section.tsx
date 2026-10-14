import type { ReactNode, Ref } from "react";

import { sectionHeadingClass } from "./styles";

/**
 * A titled block of a page: a section heading, at most one plain sentence under it, then the
 * content. Sections are grouped by whitespace, never by a box.
 */
export function Section({
  id,
  title,
  lead,
  action,
  className = "mt-20",
  sectionRef,
  children,
}: {
  id: string;
  title: ReactNode;
  /** One short sentence on what the section shows. */
  lead?: ReactNode;
  /** A control at the right of the heading (e.g. a metric picker). */
  action?: ReactNode;
  className?: string;
  sectionRef?: Ref<HTMLElement>;
  children: ReactNode;
}) {
  return (
    <section ref={sectionRef} aria-labelledby={id} className={`scroll-mt-6 ${className}`}>
      <div className="flex flex-wrap items-end justify-between gap-x-6 gap-y-3">
        <div className="min-w-0 max-w-[70ch]">
          <h2 id={id} className={sectionHeadingClass}>
            {title}
          </h2>
          {lead && <p className="mt-1 text-body text-muted">{lead}</p>}
        </div>
        {action}
      </div>
      {children}
    </section>
  );
}

/** A sub-heading inside a section (h3). */
export function SubHeading({ id, children, className = "" }: { id?: string; children: ReactNode; className?: string }) {
  return (
    <h3 id={id} className={`text-body font-semibold text-ink ${className}`}>
      {children}
    </h3>
  );
}
