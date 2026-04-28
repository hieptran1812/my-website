import { NextResponse } from "next/server";
import { calculateReadTimeWithTags } from "@/lib/readTimeCalculator";
import { loadAllPosts } from "@/lib/blogIndex";

interface BlogPostMetadata {
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
  aiGenerated?: boolean;
  seo?: {
    type: string;
    datePublished: string;
    dateModified: string;
    author: string;
  };
}

function deriveExcerpt(content: string): string {
  if (!content) return "No excerpt available";
  const lines = content.split("\n").filter((line) => {
    const trimmed = line.trim();
    return (
      trimmed.length > 0 &&
      !trimmed.startsWith("#") &&
      !trimmed.startsWith("!") &&
      !trimmed.startsWith("---") &&
      !trimmed.startsWith("```") &&
      !trimmed.startsWith("|") &&
      !trimmed.match(/^\[.*\]\(.*\)$/)
    );
  });
  const first = lines.slice(0, 3).join(" ");
  if (!first) return "No excerpt available";
  let out = first
    .replace(/\*\*(.*?)\*\*/g, "$1")
    .replace(/\*(.*?)\*/g, "$1")
    .replace(/`(.*?)`/g, "$1")
    .replace(/\[(.*?)\]\(.*?\)/g, "$1")
    .replace(/\s+/g, " ")
    .trim();
  if (out.length > 200) out = out.substring(0, 197) + "...";
  return out;
}

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const category = searchParams.get("category");
    const tag = searchParams.get("tag");
    const search = searchParams.get("search");

    const corpus = await loadAllPosts();
    let posts: BlogPostMetadata[] = corpus.map((entry) => {
      const readTime = calculateReadTimeWithTags(
        entry.content,
        entry.tags,
        entry.category,
      ).readTime;
      const fmReadTime =
        typeof entry.frontmatter.readTime === "string"
          ? entry.frontmatter.readTime
          : null;
      return {
        slug: entry.slug,
        title: entry.title,
        publishDate: entry.publishDate || "2024-01-01",
        readTime: fmReadTime || readTime,
        category: entry.category,
        author: entry.author || "Anonymous",
        tags: entry.tags,
        image: entry.image || "/images/default-blog.jpg",
        excerpt: entry.excerpt || deriveExcerpt(entry.content),
        collection:
          entry.collection ||
          (typeof entry.frontmatter.subcategory === "string"
            ? entry.frontmatter.subcategory
            : undefined),
        aiGenerated: entry.aiGenerated,
        seo: {
          type: "BlogPosting",
          datePublished: entry.publishDate || "2024-01-01",
          dateModified:
            (typeof entry.frontmatter.modifiedDate === "string"
              ? entry.frontmatter.modifiedDate
              : null) ||
            entry.publishDate ||
            "2024-01-01",
          author: entry.author || "Anonymous",
        },
      };
    });

    posts.sort((a, b) => {
      const dateA = new Date(a.publishDate);
      const dateB = new Date(b.publishDate);
      if (isNaN(dateA.getTime()) && isNaN(dateB.getTime())) return 0;
      if (isNaN(dateA.getTime())) return 1;
      if (isNaN(dateB.getTime())) return -1;
      return dateB.getTime() - dateA.getTime();
    });

    if (search) {
      const q = search.toLowerCase();
      posts = posts.filter(
        (p) =>
          p.title.toLowerCase().includes(q) ||
          p.excerpt.toLowerCase().includes(q),
      );
    } else if (category) {
      posts = posts.filter(
        (p) => p.category.toLowerCase() === category.toLowerCase(),
      );
    } else if (tag) {
      posts = posts.filter((p) =>
        p.tags.some((t) => t.toLowerCase() === tag.toLowerCase()),
      );
    }

    return NextResponse.json(posts, {
      headers: {
        "Cache-Control":
          "public, s-maxage=3600, stale-while-revalidate=86400",
      },
    });
  } catch (error) {
    console.error("Error fetching blog posts:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 },
    );
  }
}
