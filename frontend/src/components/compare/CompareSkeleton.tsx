import { skeletonClass } from "../ui/styles";

/** Placeholder for the analysis and the diff while a comparison loads. */
export function ComparisonSkeleton() {
  return (
    <div aria-hidden="true">
      <div className={`h-5 w-56 max-w-full ${skeletonClass}`} />
      <div className="mt-6 max-w-[70ch] space-y-2.5">
        {["w-full", "w-11/12", "w-2/3"].map((width) => (
          <div key={width} className={`h-3.5 ${width} ${skeletonClass}`} />
        ))}
      </div>
      <div className={`mt-20 h-5 w-48 max-w-full ${skeletonClass}`} />
      <div className="mt-6 space-y-3">
        {[0, 1, 2, 3].map((i) => (
          <div key={i} className={`h-12 ${skeletonClass}`} />
        ))}
      </div>
    </div>
  );
}

/** Server-rendered placeholder until the client reads the URL and loads the statements. */
export function CompareSkeleton() {
  return (
    <div className="mt-12">
      <p className="sr-only" role="status">
        Loading the statement comparison…
      </p>
      <div aria-hidden="true">
        <div className="grid max-w-[46rem] gap-4 sm:grid-cols-2">
          {[0, 1].map((i) => (
            <div key={i}>
              <div className={`h-3 w-28 ${skeletonClass}`} />
              <div className={`mt-2 h-9 ${skeletonClass}`} />
            </div>
          ))}
        </div>
        <div className="mt-4 flex flex-wrap gap-2">
          {[104, 92, 176].map((w) => (
            <div key={w} className={`h-8 ${skeletonClass}`} style={{ width: w }} />
          ))}
        </div>
      </div>
      <div className="mt-16">
        <ComparisonSkeleton />
      </div>
    </div>
  );
}
