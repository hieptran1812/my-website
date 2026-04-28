"use client";

import Link from "next/link";
import Image from "next/image";
import { useState } from "react";
import type { SeriesContext } from "@/lib/getRelatedPosts";
import { getPostCoverUrl } from "@/lib/getPostCover";

function SiblingImage({
  slug,
  image,
}: {
  slug: string;
  image?: string;
}) {
  const [errored, setErrored] = useState(false);
  if (errored) {
    return (
      <div
        aria-hidden="true"
        className="related-img-fallback"
        style={{
          background: `linear-gradient(135deg, hsl(${(slug.length * 31) % 360} 60% 35%), hsl(${(slug.length * 31 + 60) % 360} 60% 25%))`,
        }}
      />
    );
  }
  return (
    <Image
      src={getPostCoverUrl(slug, image)}
      alt=""
      fill
      sizes="160px"
      className="series-card-img"
      unoptimized
      onError={() => setErrored(true)}
    />
  );
}

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
        <SiblingImage slug={sib.slug} image={sib.image} />
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
    <section className="series-module section-flat" aria-label="Series navigation">
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
