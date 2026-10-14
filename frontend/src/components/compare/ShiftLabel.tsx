import type { Stance, StanceShift } from "@/lib/api/types";
import { SHIFT_LABELS } from "@/lib/compare/format";

import { DirectionMark } from "../stance/StanceLabel";

// Same marks and meaning colors as a stance: up/hawk is more hawkish, down/dove more dovish.
const DIRECTION: Record<StanceShift, Stance> = {
  more_hawkish: "hawkish",
  unchanged: "neutral",
  more_dovish: "dovish",
};

const TONES: Record<StanceShift, string> = {
  more_hawkish: "text-hawk",
  unchanged: "text-muted",
  more_dovish: "text-dove",
};

/** The direction of the stance shift between two statements, as a colored word with its mark. */
export function ShiftLabel({ shift, className = "" }: { shift: StanceShift; className?: string }) {
  return (
    <span className={`inline-flex shrink-0 items-center gap-1.5 font-semibold ${TONES[shift]} ${className}`}>
      <DirectionMark stance={DIRECTION[shift]} className="size-3" />
      <span className="sr-only">Stance shift: </span>
      {SHIFT_LABELS[shift]}
    </span>
  );
}
