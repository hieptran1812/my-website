// Utility functions for processing markdown blog posts
import { calculateReadTimeWithTags, getReadTime } from "./readTimeCalculator";

// Article interface matching the one used in blog pages
export interface Article {
  id: string;
  title: string;
  excerpt: string;
  content: string;
  category: string;
  subcategory?: string;
  tags: string[];
  date: string;
  readTime: string;
  difficulty: string;
  slug: string;
  featured: boolean;
  author?: string;
  image?: string;
  collection?: string;
}

export interface BlogPost {
  slug: string;
  title: string;
  publishDate: string;
  readTime: string;
  category: string;
  author: string;
  tags: string[];
  image: string;
  excerpt: string;
  content: string;
  collection?: string;
}

export interface BlogPostMetadata {
  slug: string;
  title: string;
  publishDate: string;
  readTime: string;
  category: string;
  author: string;
  tags: string[];
  image: string;
  excerpt: string;
  collection?: string;
}

// Lightweight metadata shape returned by /api/blog/posts (no markdown body).
interface LitePost {
  slug: string;
  title: string;
  excerpt: string;
  publishDate: string;
  readTime: string;
  category: string;
  subcategory?: string;
  author?: string;
  tags?: string[];
  image?: string;
  collection?: string;
  featured?: boolean;
}

/** Map a lite metadata post to the Article shape the listing pages render.
 *  `content` is intentionally empty — listing cards never use the body. */
function litePostToArticle(p: LitePost): Article {
  const readTime = p.readTime || "5 min read";
  const readTimeNum = parseInt(readTime.replace(/[^\d]/g, ""), 10) || 0;
  const difficulty =
    readTimeNum >= 10
      ? "Advanced"
      : readTimeNum >= 5
        ? "Intermediate"
        : "Beginner";
  return {
    id: p.slug.replace(/\//g, "-"),
    title: p.title,
    excerpt: p.excerpt || "",
    content: "",
    category: p.category,
    subcategory: p.subcategory || "",
    tags: p.tags || [],
    date: p.publishDate || "",
    readTime,
    difficulty,
    slug: p.slug,
    featured: p.featured ?? false,
    author: p.author,
    image: p.image,
    collection: p.collection,
  };
}

// Get markdown articles by category using API route (client-side function).
// Uses the lightweight metadata index (/api/blog/posts) — prebuilt at build
// time, no markdown bodies — instead of the full-corpus /api/blog/articles
// endpoint. Listing pages only render preview cards + subcategory facets, so
// shipping the post bodies was pure waste: this cuts the per-category payload
// from ~10 MB to a few hundred KB and skips the server-side corpus disk walk.
export async function getMarkdownArticlesByCategory(
  targetCategory: string,
  page: number = 1,
  limit: number = 50,
): Promise<{ articles: Article[]; total: number; hasMore: boolean }> {
  try {
    const baseUrl =
      typeof window !== "undefined"
        ? window.location.origin
        : process.env.NEXT_PUBLIC_BASE_URL || "http://localhost:3000";

    // No page/limit → legacy mode returns the full (presorted) category list,
    // which the pages need for client-side search + subcategory facet counts.
    const response = await fetch(
      `${baseUrl}/api/blog/posts?category=${encodeURIComponent(targetCategory)}`,
    );

    if (!response.ok) {
      throw new Error(`Failed to fetch articles: ${response.status}`);
    }

    const data = await response.json();
    const posts: LitePost[] = Array.isArray(data)
      ? data
      : Array.isArray(data?.posts)
        ? data.posts
        : [];
    const articles = posts.map(litePostToArticle);

    // Honour page/limit for callers that want a slice (the category pages pass
    // a high limit to pull the whole category in one shot).
    const start = (page - 1) * limit;
    const sliced = articles.slice(start, start + limit);
    return {
      articles: sliced,
      total: articles.length,
      hasMore: start + limit < articles.length,
    };
  } catch (error) {
    console.error("Error fetching articles:", error);
    return { articles: [], total: 0, hasMore: false };
  }
}

// Utility function to process blog post content and calculate readTime
export function processBlogPostWithReadTime(
  content: string,
  frontmatter: Record<string, unknown> = {},
): { processedContent: string; readTime: string } {
  const readTimeResult = calculateReadTimeWithTags(
    content,
    (frontmatter.tags as string[]) || [],
    (frontmatter.category as string) || "general",
  );

  return {
    processedContent: content,
    readTime: (frontmatter.readTime as string) || readTimeResult.readTime,
  };
}

// Quick utility to get just the readTime for any content
export function calculateContentReadTime(content: string): string {
  return getReadTime(content, { wordsPerMinute: 200 });
}
