"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useRef } from "react";

// `short` is the label on a phone, where the five full labels would not fit in one row; there the
// wordmark is the way home, so Overview is left out.
const TABS = [
  { href: "/", label: "Overview", short: null },
  { href: "/ask", label: "Ask the Fed", short: "Ask" },
  { href: "/stance", label: "Policy Stance", short: "Stance" },
  { href: "/compare", label: "Compare", short: "Compare" },
  { href: "/performance", label: "Performance", short: "Performance" },
] as const;

function isActive(pathname: string, href: string): boolean {
  if (href === "/") return pathname === "/";
  return pathname === href || pathname.startsWith(`${href}/`);
}

export function TabNav() {
  const pathname = usePathname();
  const activeRef = useRef<HTMLAnchorElement>(null);

  // On narrow screens the tab row scrolls; keep the current tab in view without
  // scrolling the page (scrollIntoView would also scroll the document).
  useEffect(() => {
    const link = activeRef.current;
    const list = link?.closest("ul");
    if (!link || !list) return;
    const l = link.getBoundingClientRect();
    const c = list.getBoundingClientRect();
    if (l.left < c.left || l.right > c.right) {
      list.scrollLeft += l.left - c.left - (c.width - l.width) / 2;
    }
  }, [pathname]);

  return (
    <nav aria-label="Primary" className="order-last -mx-4 w-[calc(100%+2rem)] min-w-0 md:order-none md:mx-0 md:w-auto">
      {/* The padding keeps the focus ring inside the scrolling list, which would clip it. */}
      <ul className="flex gap-6 overflow-x-auto px-4 pt-1 pb-3 [scrollbar-width:none] md:-mx-1.5 md:gap-7 md:px-1.5 md:py-1.5 [&::-webkit-scrollbar]:hidden">
        {TABS.map((tab) => {
          const active = isActive(pathname, tab.href);
          return (
            <li key={tab.href} className={tab.short === null ? "hidden shrink-0 sm:block" : "shrink-0"}>
              <Link
                href={tab.href}
                ref={active ? activeRef : undefined}
                aria-current={active ? "page" : undefined}
                className={[
                  "inline-block rounded-sm py-1 text-small font-medium whitespace-nowrap underline-offset-[7px] transition-colors",
                  active
                    ? "text-ink underline decoration-ink decoration-[1.5px]"
                    : "text-muted hover:text-ink",
                ].join(" ")}
              >
                {tab.short === null || tab.short === tab.label ? (
                  tab.label
                ) : (
                  <>
                    <span className="sm:hidden">{tab.short}</span>
                    <span className="hidden sm:inline">{tab.label}</span>
                  </>
                )}
              </Link>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
