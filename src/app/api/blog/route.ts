import { NextResponse } from "next/server";
import {
  getAllBlogPosts,
  getBlogPostsByCategory,
  getBlogPostsByTag,
  searchBlogPosts,
} from "../../../lib/blog";

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const category = searchParams.get("category");
    const tag = searchParams.get("tag");
    const search = searchParams.get("search");

    let posts;

    if (search) {
      posts = searchBlogPosts(search);
    } else if (category) {
      posts = getBlogPostsByCategory(category);
    } else if (tag) {
      posts = getBlogPostsByTag(tag);
    } else {
      posts = getAllBlogPosts();
    }

    return NextResponse.json(posts);
  } catch (error) {
    console.error("Error fetching blog posts:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
