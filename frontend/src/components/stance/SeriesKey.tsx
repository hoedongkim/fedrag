import { type ClassifierInfo, seriesDash } from "@/lib/stance/classifiers";

/** A classifier's line key: its color, dashed for the classifiers that average sentences. */
export function SeriesKey({ info, className = "" }: { info: ClassifierInfo; className?: string }) {
  return (
    <svg
      viewBox="0 0 16 2"
      preserveAspectRatio="none"
      aria-hidden="true"
      className={`h-0.5 w-4 shrink-0 overflow-visible ${className}`}
    >
      <line
        x1="0"
        y1="1"
        x2="16"
        y2="1"
        stroke={info.color}
        strokeWidth="2"
        strokeDasharray={seriesDash(info)}
      />
    </svg>
  );
}
