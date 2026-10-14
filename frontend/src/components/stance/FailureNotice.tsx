import type { StanceFailure } from "@/lib/stance/failures";

import { RetryIcon } from "../ask/icons";
import { secondaryButtonClass } from "../ui/styles";

/**
 * A setup hint (amber rule) or an error (red rule), with the command that fixes it and a Retry
 * button. Shared by every page; no box, just a colored rule at the left.
 */
export function FailureNotice({ failure, onRetry }: { failure: StanceFailure; onRetry?: () => void }) {
  const setup = failure.tone === "setup";
  return (
    <div className={`max-w-[70ch] border-l-2 py-0.5 pl-4 ${setup ? "border-warn" : "border-bad"}`}>
      <h3 className="text-body font-semibold text-ink">
        <span className="sr-only">{setup ? "Setup needed: " : "Error: "}</span>
        {failure.title}
      </h3>
      <p className="mt-1 text-body text-muted">
        {failure.message}
        {failure.command && (
          <>
            {" "}
            <code className="rounded-sm bg-subtle px-1.5 py-0.5 break-words text-ink ring-1 ring-line">{failure.command}</code>
          </>
        )}
      </p>
      {onRetry && (
        <button type="button" onClick={onRetry} className={`mt-3 ${secondaryButtonClass}`}>
          <RetryIcon className="size-3.5" />
          Retry
        </button>
      )}
    </div>
  );
}
