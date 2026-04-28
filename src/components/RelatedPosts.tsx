import Link from "next/link";
import type { RelatedPost } from "@/lib/getRelatedPosts";

export default function RelatedPosts({
  posts,
  heading = "Related posts",
}: {
  posts: RelatedPost[];
  heading?: string;
}) {
  if (!posts || posts.length === 0) return null;
  return (
    <section className="related-posts" aria-label={heading}>
      <h2>{heading}</h2>
      <div className="related-posts-grid">
        {posts.map((p) => (
          <Link key={p.slug} href={`/blog/${p.slug}`} className="related-card">
            <span className="related-card-cat">
              {[p.category, p.subcategory].filter(Boolean).join(" · ")}
            </span>
            <span className="related-card-title">{p.title}</span>
            {p.excerpt ? (
              <span className="related-card-excerpt">{p.excerpt}</span>
            ) : null}
          </Link>
        ))}
      </div>
    </section>
  );
}
