import Link from "next/link";
import Image from "next/image";
import type { RelatedPost, RelatedReason } from "@/lib/getRelatedPosts";

const FALLBACK_IMAGE = "/images/blog/default-post.jpg";

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
    <Link href={`/blog/${post.slug}`} className="related-card related-card-hero">
      <div className="related-card-image related-card-image-hero">
        <Image
          src={post.image || FALLBACK_IMAGE}
          alt=""
          fill
          sizes="(min-width: 1024px) 60vw, 100vw"
          className="related-img"
          priority={false}
        />
        <div className="related-card-image-overlay" />
      </div>
      <div className="related-card-body related-card-body-hero">
        <div className="related-card-meta">
          <span className="related-card-kicker">
            {post.subcategory || post.category || "Article"}
          </span>
          {date ? <span className="related-card-date">{date}</span> : null}
        </div>
        <h3 className="related-card-title related-card-title-hero">{post.title}</h3>
        {post.excerpt ? (
          <p className="related-card-excerpt related-card-excerpt-hero">
            {post.excerpt}
          </p>
        ) : null}
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
    <Link href={`/blog/${post.slug}`} className="related-card related-card-secondary">
      <div className="related-card-image related-card-image-secondary">
        <Image
          src={post.image || FALLBACK_IMAGE}
          alt=""
          fill
          sizes="(min-width: 1024px) 240px, (min-width: 720px) 50vw, 100vw"
          className="related-img"
        />
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
        <div className="related-card-footer related-card-footer-secondary">
          <ReasonBadge post={post} />
        </div>
      </div>
    </Link>
  );
}

export default function RelatedPosts({
  posts,
  heading = "Related reading",
  subheading = "Selected by shared tags, content similarity, and category overlap, then diversified to avoid repetition.",
}: {
  posts: RelatedPost[];
  heading?: string;
  subheading?: string;
}) {
  if (!posts || posts.length === 0) return null;

  // Single post → centered single card. 2 posts → grid 2-up. 3+ → hero + rest.
  const [hero, ...rest] = posts;

  return (
    <section className="related-posts" aria-label={heading}>
      <header className="related-posts-header">
        <h2>{heading}</h2>
        <p className="related-posts-sub">{subheading}</p>
      </header>

      {posts.length >= 3 ? (
        <div className="related-layout">
          <HeroCard post={hero} />
          <ul className="related-secondary-list" role="list">
            {rest.map((p) => (
              <li key={p.slug}>
                <SecondaryCard post={p} />
              </li>
            ))}
          </ul>
        </div>
      ) : (
        <ul className="related-secondary-list related-secondary-list-fallback" role="list">
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
