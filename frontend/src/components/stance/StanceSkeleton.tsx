import { skeletonClass } from "../ui/styles";

/** Server-rendered placeholder until the client reads the URL and loads the timelines. */
export function StanceSkeleton() {
  return (
    <div className="mt-12" aria-hidden="true">
      <div className={`h-2.5 w-16 ${skeletonClass}`} />
      <div className="mt-3 flex flex-wrap gap-5">
        {[104, 112, 76, 80].map((w) => (
          <div key={w} className={`h-4 ${skeletonClass}`} style={{ width: w }} />
        ))}
      </div>
      <div className={`mt-8 h-[30rem] ${skeletonClass}`} />
    </div>
  );
}
