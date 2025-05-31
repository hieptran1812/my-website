import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";
import matter from "gray-matter";
import { calculateReadTimeWithTags } from "@/lib/readTimeCalculator";

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
}

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const category = searchParams.get("category");
    const tag = searchParams.get("tag");
    const search = searchParams.get("search");

    const contentDirectory = path.join(process.cwd(), "content", "blog");

    if (!fs.existsSync(contentDirectory)) {
      return NextResponse.json([]);
    }

    const files = fs
      .readdirSync(contentDirectory)
      .filter((file) => file.endsWith(".md"));

    let posts: BlogPostMetadata[] = files.map((filename) => {
      const filePath = path.join(contentDirectory, filename);
      const fileContents = fs.readFileSync(filePath, "utf8");
      const { data, content } = matter(fileContents);

      // Calculate automatic read time based on content
      const automaticReadTime = calculateReadTimeWithTags(
        content,
        data.tags || [],
        data.category || "general"
      );

      return {
        slug: filename.replace(".md", ""),
        title: data.title || "Untitled",
        publishDate: data.date || data.publishDate || "2024-01-01",
        readTime: data.readTime || automaticReadTime.readTime, // Use manual if provided, otherwise auto-calculate
        category: data.category || "general",
        author: data.author || "Anonymous",
        tags: data.tags || [],
        image: data.image || "/images/default-blog.jpg",
        excerpt: data.excerpt || data.description || "No excerpt available",
      };
    });

    // Apply filters
    if (search) {
      posts = posts.filter(
        (post) =>
          post.title.toLowerCase().includes(search.toLowerCase()) ||
          post.excerpt.toLowerCase().includes(search.toLowerCase())
      );
    } else if (category) {
      posts = posts.filter(
        (post) => post.category.toLowerCase() === category.toLowerCase()
      );
    } else if (tag) {
      posts = posts.filter((post) =>
        post.tags.some((t) => t.toLowerCase() === tag.toLowerCase())
      );
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
