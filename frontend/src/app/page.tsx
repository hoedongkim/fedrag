import type { Metadata } from "next";
import Link from "next/link";

import { ArrowRightIcon } from "@/components/ask/icons";
import { DemoNote } from "@/components/landing/DemoNote";
import { type HeadlineKind, HeadlineSlot, LandingAbout, LandingDataProvider } from "@/components/landing/LandingData";
import { ExternalLink } from "@/components/ui/ExternalLink";
import { linkClass, outlineButtonClass, primaryButtonClass, sectionHeadingClass } from "@/components/ui/styles";
import { DOCS, REPO_URL, SITE_DESCRIPTION, SITE_NAME, SITE_TITLE } from "@/lib/site";

export const metadata: Metadata = {
  title: { absolute: SITE_TITLE },
  description: SITE_DESCRIPTION,
  alternates: { canonical: "/" },
  openGraph: { type: "website", siteName: SITE_NAME, locale: "en_US", url: "/", title: SITE_TITLE, description: SITE_DESCRIPTION },
  twitter: { card: "summary_large_image", title: SITE_TITLE, description: SITE_DESCRIPTION },
};

interface Feature {
  kind: HeadlineKind;
  href: string;
  title: string;
  body: string;
  /** What the headline number means, in a few plain words (its title tooltip). */
  hint: string;
}

const FEATURES: Feature[] = [
  {
    kind: "ask",
    href: "/ask",
    title: "Ask the Fed",
    body: "Ask a question in plain English. The answer uses only FOMC statements and minutes, and every claim links to its source.",
    hint: "The share of an answer’s claims that its cited passages support, from 0 to 1",
  },
  {
    kind: "stance",
    href: "/stance",
    title: "Policy Stance",
    body: "Every statement since 2015 rated hawkish (leaning toward higher rates), neutral or dovish by four methods, next to the actual policy rate.",
    hint: "How often the classifier agreed with a person’s label, from 0 to 1",
  },
  {
    kind: "compare",
    href: "/compare",
    title: "Statement Comparison",
    body: "Pick any two statements to see which sentences were added, removed or reworded, and what the change may signal.",
    hint: "Which way the wording moved between the two statements, as read by an LLM",
  },
  {
    kind: "rag",
    href: "/performance",
    title: "RAG Performance",
    body: "How often the search finds the right passages and how accurate the answers are, measured on questions kept aside for testing.",
    hint: "How high the right passages rank in the top 10 results, from 0 to 1",
  },
];

const STEPS = [
  { title: "Collect", body: "Every FOMC statement and set of minutes since 2015, from federalreserve.gov." },
  { title: "Search", body: "Find passages by keyword and by meaning, skipping anything published after your chosen date." },
  { title: "Answer", body: "A language model writes only from those passages and cites each one." },
  { title: "Measure", body: "Every step is scored on test questions it never saw while it was being tuned." },
];

const STACK = [
  "Python",
  "FastAPI",
  "Pydantic",
  "LangChain",
  "OpenAI",
  "Chroma",
  "rank-bm25",
  "sentence-transformers",
  "PyTorch",
  "Postgres",
  "Next.js",
  "TypeScript",
  "Tailwind CSS",
  "Recharts",
];

const READ_MORE = [
  { href: DOCS.quickstart, label: "Run it locally" },
  { href: DOCS.architecture, label: "Architecture" },
  { href: DOCS.evaluation, label: "Evaluation" },
  { href: DOCS.limitations, label: "Limitations" },
];

/** The overview: what FedRAG is, one live number per tool, and how it works. */
export default function Home() {
  return (
    <LandingDataProvider>
      <section aria-labelledby="home-title" className="pt-2 sm:pt-8">
        <h1 id="home-title" className="max-w-[16ch] text-display font-semibold text-ink">
          Ask what the Fed actually said.
        </h1>
        <p className="mt-5 max-w-[40rem] text-lede text-muted sm:text-[1.25rem] sm:leading-[1.55]">
          {SITE_NAME} answers questions about U.S. monetary policy using only the Federal Reserve&rsquo;s own
          statements and minutes, and shows the source for every claim.
        </p>
        <div className="mt-8 flex flex-wrap items-center gap-3">
          <Link href="/ask" className={primaryButtonClass}>
            Try it
            <ArrowRightIcon className="size-4" />
          </Link>
          <ExternalLink href={REPO_URL} className={outlineButtonClass}>
            GitHub
          </ExternalLink>
        </div>
        <p className="mt-6 text-small text-faint">
          A solo project by Hoedong Kim, MS in Data Science, Columbia University.
        </p>
        <DemoNote />
      </section>

      <section aria-labelledby="home-features" className="mt-20 sm:mt-24">
        <h2 id="home-features" className={sectionHeadingClass}>
          What it does
        </h2>
        <ul className="mt-6 border-b border-line">
          {FEATURES.map(({ kind, href, title, body, hint }) => (
            <li
              key={kind}
              className="grid gap-x-12 gap-y-5 border-t border-line py-7 md:grid-cols-[minmax(0,1fr)_15rem] md:py-8"
            >
              <div className="min-w-0">
                <h3 className="text-[1.0625rem] leading-snug font-semibold">
                  <Link
                    href={href}
                    className="group inline-flex items-center gap-1.5 rounded-sm text-ink transition-colors hover:text-accent"
                  >
                    {title}
                    <ArrowRightIcon className="size-4 text-faint transition-transform group-hover:translate-x-0.5 group-hover:text-accent motion-reduce:transition-none" />
                  </Link>
                </h3>
                <p className="mt-2 max-w-[54ch] text-body text-muted">{body}</p>
              </div>
              <HeadlineSlot kind={kind} hint={hint} />
            </li>
          ))}
        </ul>
      </section>

      <section aria-labelledby="home-how" className="mt-20 sm:mt-24">
        <h2 id="home-how" className={sectionHeadingClass}>
          How it works
        </h2>
        <ol className="mt-6 grid gap-x-8 gap-y-6 sm:grid-cols-2 lg:grid-cols-4">
          {STEPS.map((step, i) => (
            <li key={step.title} className="border-t border-line pt-4">
              <p className="text-label font-medium text-faint tabular-nums">
                <span className="sr-only">Step </span>
                {String(i + 1).padStart(2, "0")}
              </p>
              <h3 className="mt-2 text-body font-semibold text-ink">{step.title}</h3>
              <p className="mt-1 text-small text-muted">{step.body}</p>
            </li>
          ))}
        </ol>
        <p className="mt-8 max-w-[70ch] text-small text-faint">
          Built with{" "}
          {STACK.map((name, i) => (
            <span key={name}>
              {i > 0 && (i === STACK.length - 1 ? " and " : ", ")}
              <span className="whitespace-nowrap">{name}</span>
            </span>
          ))}
          .
        </p>
        <p className="mt-2 text-small text-faint">
          Read more on GitHub:{" "}
          {READ_MORE.map((link, i) => (
            <span key={link.href}>
              {i > 0 && <span aria-hidden="true"> · </span>}
              <ExternalLink href={link.href} className={linkClass}>
                {link.label}
              </ExternalLink>
            </span>
          ))}
        </p>
      </section>

      <LandingAbout limitationsHref={DOCS.limitations} />
    </LandingDataProvider>
  );
}
