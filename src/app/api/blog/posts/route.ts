import { getAllBlogPosts } from "../../../../lib/blog";
import { NextResponse } from "next/server";

export async function GET() {
  try {
    const posts = getAllBlogPosts();
    return NextResponse.json(posts);
  } catch (error) {
    console.error("Error fetching blog posts:", error);
    return NextResponse.json([], { status: 500 });
  }
}
