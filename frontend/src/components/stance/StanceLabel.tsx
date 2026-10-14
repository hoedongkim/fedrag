import type { Stance } from "@/lib/api/types";
import { STANCE_LABELS } from "@/lib/stance/format";

// Meaning colors: hawkish (toward higher rates) and dovish (toward lower rates); neutral stays ink.
const TONES: Record<Stance, string> = {
  hawkish: "text-hawk",
  neutral: "text-muted",
  dovish: "text-dove",
};

// The shape says the direction, so the stance never rests on color alone.
const DIRECTION_PATHS: Record<Stance, string> = {
  hawkish: "M5 1.5 9 8.5H1z",
  neutral: "M1.5 5h7",
  dovish: "M5 8.5 9 1.5H1z",
};

/** A direction mark: ▲ hawkish, – neutral, ▼ dovish, in the current text color. */
export function DirectionMark({ stance, className = "size-2.5" }: { stance: Stance; className?: string }) {
  return (
    <svg viewBox="0 0 10 10" className={`shrink-0 ${className}`} aria-hidden="true" focusable="false">
      <path
        d={DIRECTION_PATHS[stance]}
        fill={stance === "neutral" ? "none" : "currentColor"}
        stroke="currentColor"
        strokeWidth={stance === "neutral" ? 2 : 1}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}

/** A stance as a colored word with its direction mark: no fill, no box. */
export function StanceLabel({ stance, className = "" }: { stance: Stance; className?: string }) {
  return (
    <span className={`inline-flex shrink-0 items-center gap-1 font-medium ${TONES[stance]} ${className}`}>
      <DirectionMark stance={stance} />
      {STANCE_LABELS[stance]}
    </span>
  );
}
