import Link from "next/link";
import Image from "next/image";
import type { SeriesContext } from "@/lib/getRelatedPosts";

const FALLBACK_IMAGE = "/images/blog/default-post.jpg";

function SiblingCard({
  label,
  sib,
  side,
}: {
  label: string;
  sib: NonNullable<SeriesContext["prev"] | SeriesContext["next"]>;
  side: "prev" | "next";
}) {
  return (
    <Link href={`/blog/${sib.slug}`} className={`series-card series-card-${side}`}>
      <div className="series-card-image">
        <Image
          src={sib.image || FALLBACK_IMAGE}
          alt=""
          fill
          sizes="(min-width: 720px) 240px, 100vw"
          className="series-card-img"
        />
      </div>
      <div className="series-card-body">
        <span className="series-card-direction">
          {side === "prev" ? "← Previous" : "Next →"}
          <span className="series-card-pos">
            Part {sib.position} of {sib.total}
          </span>
        </span>
        <span className="series-card-label">{label}</span>
        <span className="series-card-title">{sib.title}</span>
      </div>
    </Link>
  );
}

export default function SeriesModule({ ctx }: { ctx: SeriesContext | null }) {
  if (!ctx || (!ctx.prev && !ctx.next)) return null;
  return (
    <section className="series-module" aria-label="Series navigation">
      <header className="series-module-header">
        <span className="series-module-icon" aria-hidden="true">
          📚
        </span>
        <div>
          <p className="series-module-eyebrow">Series</p>
          <h2 className="series-module-title">{ctx.collection}</h2>
        </div>
        <span className="series-module-progress">
          You’re on part {ctx.current.position} of {ctx.current.total}
        </span>
      </header>
      <div className="series-module-grid">
        {ctx.prev ? (
          <SiblingCard label="Previous in series" sib={ctx.prev} side="prev" />
        ) : (
          <div className="series-card series-card-empty">Series start</div>
        )}
        {ctx.next ? (
          <SiblingCard label="Up next" sib={ctx.next} side="next" />
        ) : (
          <div className="series-card series-card-empty">Series end</div>
        )}
      </div>
    </section>
  );
}
