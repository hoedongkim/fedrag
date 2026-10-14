import type { Metadata } from "next";
import Link from "next/link";

import { textButtonClass } from "@/components/ui/styles";

export const metadata: Metadata = {
  title: "Page not found",
};

export default function NotFound() {
  return (
    <div className="max-w-[44rem]">
      <p className="text-small font-medium text-faint tabular-nums">404</p>
      <h1 className="mt-2 text-title font-semibold text-ink">Page not found</h1>
      <p className="mt-3 text-lede text-muted">There is no page at this address.</p>
      <Link href="/" className={`mt-6 inline-block ${textButtonClass}`}>
        Go to the overview
      </Link>
    </div>
  );
}
