import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";
import matter from "gray-matter";

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

export async function GET() {
  try {
    const contentDirectory = path.join(process.cwd(), "content", "blog");

    if (!fs.existsSync(contentDirectory)) {
      return NextResponse.json([]);
    }

    const files = fs
      .readdirSync(contentDirectory)
      .filter((file) => file.endsWith(".md"));

    const posts: BlogPostMetadata[] = files.map((filename) => {
      const filePath = path.join(contentDirectory, filename);
      const fileContents = fs.readFileSync(filePath, "utf8");
      const { data } = matter(fileContents);

      return {
        slug: filename.replace(".md", ""),
        title: data.title || "Untitled",
        publishDate: data.date || data.publishDate || "2024-01-01",
        readTime: data.readTime || "5 min read",
        category: data.category || "general",
        author: data.author || "Anonymous",
        tags: data.tags || [],
        image: data.image || "/images/default-blog.jpg",
        excerpt: data.excerpt || data.description || "No excerpt available",
      };
    });

    return NextResponse.json(posts);
  } catch (error) {
    console.error("Error fetching blog posts:", error);
    return NextResponse.json([], { status: 500 });
  }
}
