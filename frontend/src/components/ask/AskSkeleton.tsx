import { skeletonClass } from "../ui/styles";

/** Placeholder with the form's footprint while the page reads its URL parameters. */
export function AskSkeleton() {
  return (
    <div className="mt-10 max-w-3xl">
      <p className="sr-only" role="status">
        Loading the question form…
      </p>
      <div aria-hidden="true">
        <div className={`h-3.5 w-28 ${skeletonClass}`} />
        <div className="mt-2 h-28 rounded-md border border-line" />
        <div className="mt-5 grid gap-5 md:grid-cols-2">
          <div className={`h-14 ${skeletonClass}`} />
          <div className={`h-4 w-56 ${skeletonClass}`} />
        </div>
        <div className={`mt-6 ml-auto h-9 w-28 ${skeletonClass}`} />
        <div className="mt-8 h-px bg-line" />
        <div className="mt-5 flex flex-wrap gap-x-5 gap-y-2">
          {[40, 32, 48, 28, 36, 44].map((width) => (
            <div key={width} style={{ width: `${width * 0.25}rem` }} className={`h-4 ${skeletonClass}`} />
          ))}
        </div>
      </div>
    </div>
  );
}
