import { NextRequest, NextResponse } from "next/server";
import fs from "fs";
import path from "path";
import matter from "gray-matter";

export async function GET(
  request: NextRequest,
  { params }: { params: { slug: string } }
) {
  try {
    const { slug } = params;

    if (!slug) {
      return NextResponse.json({ error: "Slug is required" }, { status: 400 });
    }

    const contentDirectory = path.join(process.cwd(), "content", "blog");
    const filePath = path.join(contentDirectory, `${slug}.md`);

    if (!fs.existsSync(filePath)) {
      return NextResponse.json({ error: "Post not found" }, { status: 404 });
    }

    const fileContents = fs.readFileSync(filePath, "utf8");
    const { data, content } = matter(fileContents);

    const post = {
      slug,
      title: data.title || "Untitled",
      publishDate: data.date || data.publishDate || "2024-01-01",
      readTime: data.readTime || "5 min read",
      category: data.category || "general",
      author: data.author || "Anonymous",
      tags: data.tags || [],
      image: data.image || "/images/default-blog.jpg",
      excerpt: data.excerpt || data.description || "No excerpt available",
      content: content,
    };

    return NextResponse.json(post);
  } catch (error) {
    console.error("Error fetching blog post:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
