import { skeletonClass } from "../ui/styles";

/** Server-rendered placeholder until the client reads the URL and loads the report. */
export function PerformanceSkeleton() {
  return (
    <div className="mt-12" aria-hidden="true">
      <div className="flex gap-6 border-b border-line pb-2.5">
        {[48, 44, 104].map((w) => (
          <div key={w} className={`h-4 ${skeletonClass}`} style={{ width: w }} />
        ))}
      </div>
      <div className={`mt-14 h-3 w-32 ${skeletonClass}`} />
      <div className={`mt-2 h-6 w-72 max-w-full ${skeletonClass}`} />
      <div className="mt-8 grid grid-cols-2 gap-x-8 gap-y-8 sm:grid-cols-4">
        {Array.from({ length: 8 }, (_, i) => (
          <div key={i} className="border-t border-line pt-4">
            <div className={`h-3 w-20 ${skeletonClass}`} />
            <div className={`mt-3 h-7 w-16 ${skeletonClass}`} />
          </div>
        ))}
      </div>
      <div className={`mt-20 h-80 ${skeletonClass}`} />
    </div>
  );
}
