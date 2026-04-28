import Link from "next/link";
import type { RelatedPost } from "@/lib/getRelatedPosts";

const REASON_LABEL: Record<RelatedPost["reason"], string> = {
  tags: "Shared topics",
  subcategory: "Same subcategory",
  category: "Same category",
};

function formatDate(iso: string): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleDateString("en-US", { month: "short", year: "numeric" });
}

export default function RelatedPosts({
  posts,
  heading = "Related reading",
  subheading = "Posts ranked by shared tags, then by subcategory and category overlap.",
}: {
  posts: RelatedPost[];
  heading?: string;
  subheading?: string;
}) {
  if (!posts || posts.length === 0) return null;
  return (
    <section className="related-posts" aria-label={heading}>
      <header className="related-posts-header">
        <h2>{heading}</h2>
        <p className="related-posts-sub">{subheading}</p>
      </header>
      <ul className="related-posts-grid" role="list">
        {posts.map((p) => {
          const date = formatDate(p.publishDate);
          const showTags = p.sharedTags.slice(0, 3);
          return (
            <li key={p.slug} className="related-card-wrap">
              <Link
                href={`/blog/${p.slug}`}
                className="related-card"
                aria-label={p.title}
              >
                <div className="related-card-meta">
                  <span className="related-card-kicker">
                    {p.subcategory || p.category || "Article"}
                  </span>
                  {date ? (
                    <span className="related-card-date">{date}</span>
                  ) : null}
                </div>

                <h3 className="related-card-title">{p.title}</h3>

                {p.excerpt ? (
                  <p className="related-card-excerpt">{p.excerpt}</p>
                ) : null}

                <div className="related-card-footer">
                  {showTags.length > 0 ? (
                    <ul className="related-card-tags" aria-label="Shared tags">
                      {showTags.map((t) => (
                        <li key={t} className="related-card-tag">
                          {t}
                        </li>
                      ))}
                      {p.sharedTags.length > showTags.length ? (
                        <li className="related-card-tag related-card-tag-more">
                          +{p.sharedTags.length - showTags.length}
                        </li>
                      ) : null}
                    </ul>
                  ) : (
                    <span className="related-card-reason">
                      {REASON_LABEL[p.reason]}
                    </span>
                  )}
                </div>
              </Link>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
