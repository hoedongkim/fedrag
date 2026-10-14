"use client";

import { type ReactNode, useId } from "react";

import type { ChangeKind, CompareStats, SentenceChange, SourceDocument } from "@/lib/api/types";
import { formatDate, isRevised } from "@/lib/ask/format";
import { formatSimilarity, KIND_LABELS } from "@/lib/compare/format";
import { changeAnchor, type DiffBlock, runIds } from "@/lib/compare/rows";
import { type Side, usableSegments } from "@/lib/compare/segments";

import { ChevronRightIcon, ExternalLinkIcon } from "../ask/icons";
import { Section } from "../ui/Section";
import { delClass, insClass, SegmentText } from "./SegmentText";

interface DiffViewProps {
  base: SourceDocument;
  target: SourceDocument;
  stats: CompareStats;
  blocks: DiffBlock[];
  /** Split view (base left, target right) on wide screens; unified below 768px. */
  split: boolean;
  highlighted: ReadonlySet<number>;
  onToggleRun: (runId: number) => void;
  onToggleAll: (open: boolean) => void;
}

/**
 * The sentence diff: each change in document order with an anchor (#change-N), word-level marks
 * on modified pairs, and runs of unchanged sentences folded into expanders.
 */
export function DiffView({ base, target, stats, blocks, split, highlighted, onToggleRun, onToggleAll }: DiffViewProps) {
  const toggleId = useId();
  const runs = runIds(blocks);
  const allOpen = runs.length > 0 && blocks.every((b) => b.type !== "run" || b.expanded);
  const Row = split ? SplitRow : UnifiedRow;

  return (
    <Section id="compare-diff" title="Sentence by sentence" lead="Sentences are lined up in order, with the changed words marked.">
      <div className="mt-6 flex flex-wrap items-center justify-between gap-x-8 gap-y-3">
        <StatsLine stats={stats} />
        {runs.length > 0 && (
          <div className="flex items-center gap-2">
            <input
              id={toggleId}
              type="checkbox"
              checked={allOpen}
              onChange={(event) => onToggleAll(event.target.checked)}
              className="size-3.5 accent-accent"
            />
            <label htmlFor={toggleId} className="text-small text-muted">
              Show unchanged sentences
            </label>
          </div>
        )}
      </div>
      <Legend />

      {split ? <SplitHeader base={base} target={target} /> : <UnifiedHeader base={base} target={target} />}

      <ol aria-label="Aligned sentences" className={`divide-y divide-line border-b border-line ${split ? "" : "mt-4 border-t"}`}>
        {blocks.map((block) =>
          block.type === "row" ? (
            <Row key={block.change.id} change={block.change} highlighted={highlighted.has(block.change.id)} />
          ) : (
            <RunBlock
              key={`run-${block.runId}`}
              block={block}
              onToggle={() => onToggleRun(block.runId)}
              renderRow={(change) => <Row key={change.id} change={change} highlighted={highlighted.has(change.id)} />}
            />
          ),
        )}
      </ol>
      {blocks.length === 0 && <p className="mt-4 text-small text-faint">Neither statement has any sentences.</p>}
    </Section>
  );
}

function StatsLine({ stats }: { stats: CompareStats }) {
  const count = (n: number) => <span className="font-semibold text-ink tabular-nums">{n}</span>;
  return (
    <ul aria-label="Summary of changes" className="flex flex-wrap items-center gap-x-5 gap-y-1 text-small text-muted">
      <li>{count(stats.modified)} modified</li>
      <li className="inline-flex items-center gap-1.5">
        <span aria-hidden="true" className="size-2.5 rounded-[2px] bg-ins ring-1 ring-ins-ink/25 ring-inset" />
        {count(stats.added)} added
      </li>
      <li className="inline-flex items-center gap-1.5">
        <span aria-hidden="true" className="size-2.5 rounded-[2px] bg-del ring-1 ring-del-ink/25 ring-inset" />
        {count(stats.removed)} removed
      </li>
      <li>{count(stats.unchanged)} unchanged</li>
      <li title="How alike the two statements’ wording is overall: 100% means identical.">
        <span className="font-semibold text-ink tabular-nums">{formatSimilarity(stats.similarity)}</span> similar
        overall
      </li>
    </ul>
  );
}

function Legend() {
  // Spans styled like the marks rather than <del>/<ins>, so screen readers hear plain words.
  return (
    <p className="mt-2 text-label text-faint">
      <span className={delClass}>deleted words</span> <span aria-hidden="true">·</span>{" "}
      <span className={insClass}>inserted words</span> <span aria-hidden="true">·</span> procedural sentences (the
      vote) in small type
    </p>
  );
}

function DocLink({ doc }: { doc: SourceDocument }) {
  return (
    <a
      href={doc.url}
      target="_blank"
      rel="noopener noreferrer"
      className="inline-flex items-center gap-1 rounded-sm font-medium text-ink underline decoration-line-strong underline-offset-[3px] hover:decoration-ink"
    >
      {formatDate(doc.meeting_date)}
      <ExternalLinkIcon className="size-3.5 text-faint" />
      <span className="sr-only"> statement on federalreserve.gov (opens in a new tab)</span>
    </a>
  );
}

function DocNotes({ doc }: { doc: SourceDocument }) {
  const revised = isRevised(doc.release_date, doc.page_updated);
  return (
    <>
      {doc.release_date !== doc.meeting_date && (
        <span className="text-label text-faint">released {formatDate(doc.release_date)}</span>
      )}
      {revised && doc.page_updated && (
        <span
          title={`The Fed updated this page on ${formatDate(doc.page_updated)}, after its release on ${formatDate(doc.release_date)}; the text compared is the updated page.`}
          className="text-label font-medium text-warn"
        >
          revised after release
        </span>
      )}
    </>
  );
}

/** Column headings that stay on screen while the rows scroll under them. */
function SplitHeader({ base, target }: { base: SourceDocument; target: SourceDocument }) {
  const heading = (label: string, doc: SourceDocument) => (
    <div className="min-w-0 px-2.5">
      <p className="text-label text-faint">{label}</p>
      <p className="mt-0.5 flex flex-wrap items-baseline gap-x-2 gap-y-1 text-body">
        <DocLink doc={doc} />
        <DocNotes doc={doc} />
      </p>
    </div>
  );
  return (
    <div className="sticky top-0 z-10 mt-6 grid grid-cols-[2.5rem_minmax(0,1fr)_minmax(0,1fr)] gap-x-6 border-b border-line bg-canvas/95 py-3 backdrop-blur-sm">
      <span aria-hidden="true" />
      {heading("Earlier", base)}
      {heading("Later", target)}
    </div>
  );
}

function UnifiedHeader({ base, target }: { base: SourceDocument; target: SourceDocument }) {
  return (
    <dl className="mt-6 grid gap-1.5 text-body">
      <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
        <dt className="w-14 text-label text-faint">Earlier</dt>
        <dd className="contents">
          <DocLink doc={base} />
          <DocNotes doc={base} />
        </dd>
      </div>
      <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
        <dt className="w-14 text-label text-faint">Later</dt>
        <dd className="contents">
          <DocLink doc={target} />
          <DocNotes doc={target} />
        </dd>
      </div>
    </dl>
  );
}

const KIND_TONES: Record<ChangeKind, string> = {
  modified: "font-medium text-muted",
  added: "font-medium text-ins-ink",
  removed: "font-medium text-del-ink",
  unchanged: "text-faint",
};

function MetaLine({ change }: { change: SentenceChange }) {
  return (
    <p className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5 text-label text-faint">
      <span className={KIND_TONES[change.kind]}>{KIND_LABELS[change.kind]}</span>
      {change.kind === "modified" && change.similarity !== null && (
        <span className="tabular-nums">{formatSimilarity(change.similarity)} similar</span>
      )}
      {change.procedural && <span>· procedural</span>}
    </p>
  );
}

/** The change number in the left gutter: what the analysis's "See #n" links point at. */
function ChangeNumber({ change }: { change: SentenceChange }) {
  return <span className="pt-px font-mono text-label text-faint tabular-nums">#{change.id}</span>;
}

function rowClass(highlighted: boolean): string {
  return [
    "scroll-mt-28 py-4 transition-[background-color,box-shadow] duration-300 motion-reduce:transition-none focus-visible:-outline-offset-2",
    highlighted ? "bg-accent-soft shadow-[inset_2px_0_0_var(--accent)]" : "",
  ].join(" ");
}

function textClass(change: SentenceChange): string {
  // muted, not faint: small type on the added/removed washes still needs 4.5:1 (WCAG AA).
  if (change.procedural) return "text-small text-muted";
  if (change.kind === "unchanged") return "text-body text-muted";
  return "text-body text-ink";
}

const TINTS: Partial<Record<ChangeKind, string>> = {
  added: "bg-ins/45",
  removed: "bg-del/80",
};

interface RowProps {
  change: SentenceChange;
  highlighted: boolean;
}

function SideCell({ change, side }: { change: SentenceChange; side: Side }) {
  const text = side === "base" ? change.base : change.target;
  const label = side === "base" ? "Earlier" : "Later";
  if (text === null) {
    return (
      <div>
        <span className="sr-only">{`${label}: no sentence.`}</span>
      </div>
    );
  }
  const segments = usableSegments(change);
  return (
    <p className={`min-w-0 rounded-sm px-2.5 py-1 break-words ${TINTS[change.kind] ?? ""} ${textClass(change)}`}>
      <span className="sr-only">{`${label}: `}</span>
      {segments ? <SegmentText segments={segments} side={side} /> : text}
    </p>
  );
}

function SplitRow({ change, highlighted }: RowProps) {
  return (
    <li
      id={changeAnchor(change.id)}
      tabIndex={-1}
      className={`grid grid-cols-[2.5rem_minmax(0,1fr)_minmax(0,1fr)] gap-x-6 ${rowClass(highlighted)}`}
    >
      <ChangeNumber change={change} />
      <div className="col-span-2 px-2.5">
        <MetaLine change={change} />
      </div>
      <div className="col-start-2 mt-1.5">
        <SideCell change={change} side="base" />
      </div>
      <div className="mt-1.5">
        <SideCell change={change} side="target" />
      </div>
    </li>
  );
}

function UnifiedRow({ change, highlighted }: RowProps) {
  const segments = usableSegments(change);
  let body;
  if (change.kind === "modified") {
    body = segments ? (
      <SegmentText segments={segments} side="both" />
    ) : (
      // Segments missing or inconsistent: show both sentences whole rather than a wrong redline.
      <>
        <del className={delClass}>
          <span className="sr-only">deleted: </span>
          {change.base}
        </del>{" "}
        <ins className={insClass}>
          <span className="sr-only">inserted: </span>
          {change.target}
        </ins>
      </>
    );
  } else {
    body = change.kind === "removed" ? change.base : change.target;
  }
  return (
    <li id={changeAnchor(change.id)} tabIndex={-1} className={rowClass(highlighted)}>
      <div className="flex items-baseline gap-2 px-1">
        <ChangeNumber change={change} />
        <MetaLine change={change} />
      </div>
      <p className={`mt-1.5 rounded-sm px-2.5 py-1 break-words ${TINTS[change.kind] ?? ""} ${textClass(change)}`}>
        {body}
      </p>
    </li>
  );
}

function RunBlock({
  block,
  onToggle,
  renderRow,
}: {
  block: Extract<DiffBlock, { type: "run" }>;
  onToggle: () => void;
  renderRow: (change: SentenceChange) => ReactNode;
}) {
  const first = block.changes[0].id;
  const last = block.changes[block.changes.length - 1].id;
  return (
    <>
      <li>
        <button
          type="button"
          aria-expanded={block.expanded}
          onClick={onToggle}
          className="flex w-full items-center gap-2 rounded-sm py-3 text-left text-small text-muted transition-colors hover:bg-subtle hover:text-ink focus-visible:-outline-offset-2"
        >
          <span className="flex w-10 shrink-0 justify-center">
            <ChevronRightIcon
              className={`size-3.5 transition-transform motion-reduce:transition-none ${block.expanded ? "rotate-90" : ""}`}
            />
          </span>
          <span>
            {block.changes.length} unchanged sentences{" "}
            <span className="font-mono text-label text-faint tabular-nums">
              #{first}–#{last}
            </span>
          </span>
        </button>
      </li>
      {block.expanded && block.changes.map(renderRow)}
    </>
  );
}
