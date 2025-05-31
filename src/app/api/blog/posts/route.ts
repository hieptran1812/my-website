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

    const posts: BlogPostMetadata[] = [];

    // Helper function to recursively read markdown files from a directory
    const readMarkdownFiles = (dirPath: string, categoryOverride?: string) => {
      const entries = fs.readdirSync(dirPath, { withFileTypes: true });

      for (const entry of entries) {
        const entryPath = path.join(dirPath, entry.name);

        if (entry.isDirectory()) {
          // If it's a directory, use its name as the category and read its files
          readMarkdownFiles(entryPath, entry.name);
        } else if (entry.name.endsWith(".md")) {
          // If it's a markdown file, parse it
          const fileContents = fs.readFileSync(entryPath, "utf8");
          const { data } = matter(fileContents);

          // Determine the category - use the directory name if it's in a category folder
          const effectiveCategory =
            categoryOverride || data.category || "general";

          posts.push({
            slug: categoryOverride
              ? `${categoryOverride}/${entry.name.replace(".md", "")}`
              : entry.name.replace(".md", ""),
            title: data.title || "Untitled",
            publishDate: data.date || data.publishDate || "2024-01-01",
            readTime: data.readTime || "5 min read",
            category: effectiveCategory,
            author: data.author || "Anonymous",
            tags: data.tags || [],
            image: data.image || "/images/default-blog.jpg",
            excerpt: data.excerpt || data.description || "No excerpt available",
          });
        }
      }
    };

    // Start reading from the root blog directory
    readMarkdownFiles(contentDirectory);

    // Sort by publishDate (newest first)
    posts.sort((a, b) => {
      return (
        new Date(b.publishDate).getTime() - new Date(a.publishDate).getTime()
      );
    });

    return NextResponse.json(posts);
  } catch (error) {
    console.error("Error fetching blog posts:", error);
    return NextResponse.json([], { status: 500 });
  }
}
