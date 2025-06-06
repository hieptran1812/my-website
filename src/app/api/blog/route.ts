import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";
import matter from "gray-matter";
import { calculateReadTimeWithTags } from "@/lib/readTimeCalculator";

// Implement caching for blog posts
const CACHE_DURATION = 60 * 60 * 1000; // 1 hour in milliseconds
let cachedPosts = null;
let lastCacheTime = 0;

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

    // Use cached posts if available and not expired
    const now = Date.now();
    if (
      cachedPosts &&
      now - lastCacheTime < CACHE_DURATION &&
      !process.env.NODE_ENV?.includes("development")
    ) {
      // Filter cached posts based on query parameters
      let filteredPosts = [...cachedPosts];

      if (search) {
        const searchLower = search.toLowerCase();
        filteredPosts = filteredPosts.filter(
          (post) =>
            post.title.toLowerCase().includes(searchLower) ||
            post.excerpt.toLowerCase().includes(searchLower)
        );
      } else if (category) {
        filteredPosts = filteredPosts.filter(
          (post) => post.category.toLowerCase() === category.toLowerCase()
        );
      } else if (tag) {
        filteredPosts = filteredPosts.filter((post) =>
          post.tags.some((t) => t.toLowerCase() === tag.toLowerCase())
        );
      }

      return NextResponse.json(filteredPosts);
    }

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
          // Add SEO-friendly JSON-LD data
          seo: {
            type: "BlogPosting",
            datePublished: data.date || data.publishDate || "2024-01-01",
            dateModified: data.modifiedDate || data.date || "2024-01-01",
            author: data.author || "Anonymous",
          },
        });
      }
    }

    // Sort posts by date (newest first)
    posts.sort(
      (a, b) =>
        new Date(b.publishDate).getTime() - new Date(a.publishDate).getTime()
    );

    // Update cache
    cachedPosts = posts;
    lastCacheTime = now;

    // Apply filters
    if (search) {
      const searchLower = search.toLowerCase();
      posts = posts.filter(
        (post) =>
          post.title.toLowerCase().includes(searchLower) ||
          post.excerpt.toLowerCase().includes(searchLower)
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
