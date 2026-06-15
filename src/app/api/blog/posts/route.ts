import { NextRequest, NextResponse } from "next/server";
import {
  getAllPostsLite,
  filterPosts,
  getCategoryCounts,
  paginate,
  type BlogPostLite,
} from "../../../../lib/blogPostsIndex";

// Kept for the legacy (un-paginated) response shape consumed by older callers.
export interface BlogPost {
  slug: string;
  title: string;
  publishDate: string;
  readTime: string;
  category: string;
  subcategory: string;
  author: string;
  tags: string[];
  image: string;
  excerpt: string;
  collection?: string;
  featured: boolean;
}

const CACHE_HEADERS = {
  "Cache-Control": "public, s-maxage=3600, stale-while-revalidate=86400",
};

// Listing cards expect a string `image`; the lite index leaves it undefined
// when a post has no resolvable cover, so callers fall back to the generated
// OG thumbnail. Preserve that contract.
function toBlogPost(p: BlogPostLite): BlogPost {
  return {
    slug: p.slug,
    title: p.title,
    publishDate: p.publishDate || "2024-01-01",
    readTime: p.readTime,
    category: p.category,
    subcategory: p.subcategory,
    author: p.author,
    tags: p.tags,
    image: p.image || "",
    excerpt: p.excerpt,
    collection: p.collection,
    featured: p.featured,
  };
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const category = searchParams.get("category");
    const tag = searchParams.get("tag");
    const collection = searchParams.get("collection");
    const search = searchParams.get("search");
    const pageParam = searchParams.get("page");
    const limitParam = searchParams.get("limit");

    const all = await getAllPostsLite();
    const filtered = filterPosts(all, { category, tag, collection, search });

    // Paginated mode: any of page/limit present → wrapped response.
    if (pageParam !== null || limitParam !== null) {
      const page = pageParam ? parseInt(pageParam, 10) : 1;
      const limit = limitParam ? parseInt(limitParam, 10) : 9;
      const { items, pagination } = paginate(filtered, page, limit);
      return NextResponse.json(
        {
          posts: items.map(toBlogPost),
          pagination,
          // Counts are over the unfiltered corpus (used by the category pills).
          categoryCounts: getCategoryCounts(all),
        },
        { headers: CACHE_HEADERS },
      );
    }

    // Legacy mode: full filtered array (backward compatible).
    return NextResponse.json(filtered.map(toBlogPost), {
      headers: CACHE_HEADERS,
    });
  } catch (error) {
    console.error("Error reading blog posts:", error);
    return NextResponse.json([]);
  }
}
