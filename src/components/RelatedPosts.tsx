"use client";

import Link from "next/link";
import Image from "next/image";
import { useState } from "react";
import type { RelatedPost, RelatedReason } from "@/lib/getRelatedPosts";
import { getPostCoverUrl } from "@/lib/getPostCover";

const REASON_ICON: Record<RelatedReason, string> = {
  tags: "🏷️",
  similar: "🔍",
  series: "📚",
  subcategory: "📂",
  category: "📁",
};

function formatDate(iso: string): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleDateString("en-US", { month: "short", year: "numeric" });
}

/**
 * Cover image with graceful fallback. Uses unoptimized so the request bypasses
 * /_next/image (the cover route already returns the exact size we need; the
 * proxy step adds latency and a failure mode where 502s render as broken icons).
 */
function CardImage({
  post,
  variant,
}: {
  post: RelatedPost;
  variant: "hero" | "secondary";
}) {
  const [errored, setErrored] = useState(false);
  const sizes =
    variant === "hero"
      ? "(min-width: 1024px) 1024px, 100vw"
      : "(min-width: 1024px) 360px, (min-width: 720px) 50vw, 100vw";
  const src = getPostCoverUrl(post.slug, post.image);

  if (errored) {
    return (
      <div
        aria-hidden="true"
        className="related-img-fallback"
        style={{
          background: `linear-gradient(135deg, hsl(${(post.slug.length * 31) % 360} 60% 35%), hsl(${(post.slug.length * 31 + 60) % 360} 60% 25%))`,
        }}
      />
    );
  }

  return (
    <Image
      src={src}
      alt=""
      fill
      sizes={sizes}
      className="related-img"
      unoptimized
      onError={() => setErrored(true)}
    />
  );
}

function MetricStrip({ post }: { post: RelatedPost }) {
  const pct = Math.max(8, Math.min(100, post.relevancePercent));
  const cosinePct = Math.round(post.similarity * 100);
  return (
    <div className="related-metric" aria-label={`${pct}% match`}>
      <div className="related-metric-bar" role="presentation">
        <span
          className="related-metric-fill"
          style={{ width: `${pct}%` }}
        />
      </div>
      <div className="related-metric-stats">
        <span className="related-metric-pct">{pct}%</span>
        <span className="related-metric-sep" aria-hidden="true">
          ·
        </span>
        {post.sharedTags.length > 0 ? (
          <span title={`Shared tags: ${post.sharedTags.join(", ")}`}>
            🏷️ {post.sharedTags.length}
          </span>
        ) : null}
        {cosinePct >= 5 ? (
          <span title="Title + excerpt cosine similarity">
            🔍 {cosinePct}%
          </span>
        ) : null}
      </div>
    </div>
  );
}

function ReasonBadge({ post }: { post: RelatedPost }) {
  const icon = REASON_ICON[post.reason];
  const text = post.reasonDetail || post.reason;
  return (
    <span className={`related-reason related-reason-${post.reason}`}>
      <span aria-hidden="true">{icon}</span>
      <span>{text}</span>
    </span>
  );
}

function HeroCard({ post }: { post: RelatedPost }) {
  const date = formatDate(post.publishDate);
  return (
    <Link
      href={`/blog/${post.slug}`}
      className="related-card related-card-hero"
    >
      <div className="related-card-image related-card-image-hero">
        <CardImage post={post} variant="hero" />
        <span className="related-card-best-badge">Best match</span>
      </div>
      <div className="related-card-body related-card-body-hero">
        <div className="related-card-meta">
          <span className="related-card-kicker">
            {post.subcategory || post.category || "Article"}
          </span>
          {date ? <span className="related-card-date">{date}</span> : null}
        </div>
        <h3 className="related-card-title related-card-title-hero">
          {post.title}
        </h3>
        {post.excerpt ? (
          <p className="related-card-excerpt related-card-excerpt-hero">
            {post.excerpt}
          </p>
        ) : null}
        <MetricStrip post={post} />
        <div className="related-card-footer">
          <ReasonBadge post={post} />
          {post.sharedTags.length > 0 ? (
            <ul className="related-card-tags" aria-label="Shared tags">
              {post.sharedTags.slice(0, 4).map((t) => (
                <li key={t} className="related-card-tag">
                  {t}
                </li>
              ))}
              {post.sharedTags.length > 4 ? (
                <li className="related-card-tag related-card-tag-more">
                  +{post.sharedTags.length - 4}
                </li>
              ) : null}
            </ul>
          ) : null}
        </div>
      </div>
    </Link>
  );
}

function SecondaryCard({ post }: { post: RelatedPost }) {
  const date = formatDate(post.publishDate);
  return (
    <Link
      href={`/blog/${post.slug}`}
      className="related-card related-card-secondary"
    >
      <div className="related-card-image related-card-image-secondary">
        <CardImage post={post} variant="secondary" />
      </div>
      <div className="related-card-body related-card-body-secondary">
        <div className="related-card-meta">
          <span className="related-card-kicker">
            {post.subcategory || post.category || "Article"}
          </span>
          {date ? <span className="related-card-date">{date}</span> : null}
        </div>
        <h3 className="related-card-title related-card-title-secondary">
          {post.title}
        </h3>
        <MetricStrip post={post} />
        <div className="related-card-footer related-card-footer-secondary">
          <ReasonBadge post={post} />
        </div>
      </div>
    </Link>
  );
}

const METHOD_DESCRIPTION =
  "Selected from the full corpus by IDF-weighted shared tags + TF-IDF cosine on title and excerpt, blended with subcategory/category overlap and a 2-year recency decay, then diversified with MMR (λ=0.7).";

export default function RelatedPosts({
  posts,
  heading = "Related reading",
}: {
  posts: RelatedPost[];
  heading?: string;
}) {
  if (!posts || posts.length === 0) return null;
  const [hero, ...rest] = posts;
  const corpusNote = `${posts.length} ${posts.length === 1 ? "post" : "posts"} ranked by tag IDF, content cosine, recency, and MMR diversity.`;

  return (
    <section className="related-posts section-flat" aria-label={heading}>
      <header className="related-posts-header">
        <div className="related-posts-heading-row">
          <h2>{heading}</h2>
          <span
            className="related-posts-info"
            tabIndex={0}
            aria-label="How posts are selected"
            data-tooltip={METHOD_DESCRIPTION}
          >
            ⓘ
          </span>
        </div>
        <p className="related-posts-sub">{corpusNote}</p>
      </header>

      {posts.length >= 3 ? (
        <>
          <HeroCard post={hero} />
          {rest.length > 0 ? (
            <ul className="related-secondary-grid" role="list">
              {rest.map((p) => (
                <li key={p.slug}>
                  <SecondaryCard post={p} />
                </li>
              ))}
            </ul>
          ) : null}
        </>
      ) : (
        <ul
          className="related-secondary-grid related-secondary-grid-fallback"
          role="list"
        >
          {posts.map((p) => (
            <li key={p.slug}>
              <SecondaryCard post={p} />
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
