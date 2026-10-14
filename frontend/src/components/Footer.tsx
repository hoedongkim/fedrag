import type { ReactNode } from "react";

import { REPO_URL } from "@/lib/site";

function FooterLink({ href, children }: { href: string; children: ReactNode }) {
  return (
    <a href={href} className="rounded-sm whitespace-nowrap text-muted underline decoration-line-strong underline-offset-2 hover:text-ink hover:decoration-current">
      {children}
    </a>
  );
}

/** Two quiet lines: what this is, and where the data, models and code come from. */
export function Footer() {
  return (
    <footer className="border-t border-line">
      <div className="mx-auto w-full max-w-5xl space-y-1.5 px-4 py-8 text-small text-faint sm:px-6">
        <p>
          <span className="font-medium text-muted">FedRAG</span> is an educational project, not investment advice.
          Answers can be wrong, so check the cited sources. Code: <FooterLink href={REPO_URL}>MIT, on GitHub</FooterLink>.
        </p>
        <p>
          Data: <FooterLink href="https://www.federalreserve.gov/monetarypolicy/fomc.htm">Federal Reserve Board</FooterLink> ·{" "}
          <FooterLink href="https://fred.stlouisfed.org/series/DFEDTARU">FRED</FooterLink> ·{" "}
          <FooterLink href="https://aclanthology.org/2023.acl-long.368/">Trillion Dollar Words</FooterLink> (Shah et al., ACL
          2023, CC BY-NC 4.0) · <FooterLink href="https://huggingface.co/ProsusAI/finbert">FinBERT</FooterLink> (ProsusAI, CC
          BY-NC-SA 4.0).
        </p>
      </div>
    </footer>
  );
}
