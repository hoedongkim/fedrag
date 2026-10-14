import type { ReactNode } from "react";

/** A page's title and one plain sentence on what it does; the content follows. */
export function PageHeader({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="max-w-[44rem]">
      <h1 className="text-title font-semibold text-ink">{title}</h1>
      <p className="mt-3 text-lede text-muted">{children}</p>
    </div>
  );
}
