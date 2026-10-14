import Link from "next/link";

import { BackendStatus } from "./BackendStatus";
import { TabNav } from "./TabNav";

/**
 * One row on a wide screen (wordmark, tabs, API status); on a phone the tabs take a second row
 * of their own and scroll sideways.
 */
export function Header() {
  return (
    <header className="border-b border-line">
      <div className="mx-auto flex w-full max-w-5xl flex-wrap items-center gap-x-8 px-4 sm:px-6 md:h-14 md:flex-nowrap">
        <Link
          href="/"
          aria-label="FedRAG, overview"
          className="flex h-12 items-center rounded-sm text-[1.0625rem] font-semibold tracking-[-0.015em] text-ink md:h-auto"
        >
          FedRAG
        </Link>
        <div className="ml-auto md:order-last">
          <BackendStatus />
        </div>
        <TabNav />
      </div>
    </header>
  );
}
