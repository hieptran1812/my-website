import { NextRequest, NextResponse } from "next/server";
import { calculateReadTimeWithTags } from "../../../../lib/readTimeCalculator";
import { loadAllPosts } from "../../../../lib/blogIndex";

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
  collection?: string;
}

// Building each BlogPost runs calculateReadTimeWithTags() — costly across the
// whole corpus. loadAllPosts() returns a stable array reference while its cache
// is warm, so memoize the fully built + sorted list against that reference and
// only rebuild when the corpus is actually refreshed.
type Corpus = Awaited<ReturnType<typeof loadAllPosts>>;
let builtCorpusRef: Corpus | null = null;
let builtPosts: BlogPost[] = [];

function buildAllPosts(corpus: Corpus): BlogPost[] {
  if (corpus === builtCorpusRef) return builtPosts;

  const posts: BlogPost[] = corpus.map((entry) => {
    const readTimeResult = calculateReadTimeWithTags(
      entry.content,
      entry.tags,
      entry.category || "General",
    );

    let excerpt = entry.excerpt;
    if (!excerpt && entry.content) {
      excerpt = entry.content.split("\n\n")[0].substring(0, 160).trim();
      if (entry.content.length > 160) excerpt += "...";
    }

    return {
      slug: entry.slug,
      title: entry.title,
      publishDate: entry.publishDate || new Date().toISOString().split("T")[0],
      readTime: readTimeResult.readTime,
      category: entry.category || "General",
      author: entry.author || "Hiep Tran",
      tags: entry.tags,
      image: entry.image || "/images/blog/default-post.jpg",
      excerpt,
      collection: entry.collection,
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

  builtCorpusRef = corpus;
  builtPosts = posts;
  return posts;
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const category = searchParams.get("category");

    const corpus = await loadAllPosts();
    const allPosts = buildAllPosts(corpus);

    // Filtering preserves the already-sorted order.
    const posts = category
      ? allPosts.filter((post) => {
          const q = category.toLowerCase();
          return (
            post.category.toLowerCase().includes(q) ||
            post.tags.some((t) => t.toLowerCase().includes(q)) ||
            post.slug.toLowerCase().includes(q)
          );
        })
      : allPosts;

    return NextResponse.json(posts, {
      headers: {
        "Cache-Control":
          "public, s-maxage=3600, stale-while-revalidate=86400",
      },
    });
  } catch (error) {
    console.error("Error reading blog posts:", error);
    return NextResponse.json([]);
  }
}
