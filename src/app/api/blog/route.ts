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

    // Get all blog posts from all category folders
    let posts: BlogPostMetadata[] = [];

    // Read directories inside blog folder (categories)
    const categoryFolders = fs
      .readdirSync(contentDirectory, { withFileTypes: true })
      .filter((dirent) => dirent.isDirectory())
      .map((dirent) => dirent.name);

    // Process files from each category folder
    for (const categoryFolder of categoryFolders) {
      const categoryPath = path.join(contentDirectory, categoryFolder);
      const files = fs
        .readdirSync(categoryPath)
        .filter((file) => file.endsWith(".md"));

      for (const filename of files) {
        const filePath = path.join(categoryPath, filename);
        const fileContents = fs.readFileSync(filePath, "utf8");
        const { data, content } = matter(fileContents);

        // Calculate automatic read time based on content
        const automaticReadTime = calculateReadTimeWithTags(
          content,
          data.tags || [],
          data.category || categoryFolder
        );

        posts.push({
          slug: `${categoryFolder}/${filename.replace(".md", "")}`,
          title: data.title || "Untitled",
          publishDate: data.date || data.publishDate || "2024-01-01",
          readTime: data.readTime || automaticReadTime.readTime,
          category: data.category || categoryFolder,
          author: data.author || "Anonymous",
          tags: data.tags || [],
          image: data.image || "/images/default-blog.jpg",
          excerpt: data.excerpt || data.description || "No excerpt available",
        });
      }
    }

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
