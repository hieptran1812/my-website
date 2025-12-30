import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";
import matter from "gray-matter";
import { calculateReadTimeWithTags } from "@/lib/readTimeCalculator";

// Implement caching for blog posts
const CACHE_DURATION = 60 * 60 * 1000; // 1 hour in milliseconds
let cachedPosts: BlogPostMetadata[] | null = null;
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
  collection?: string;
  seo?: {
    type: string;
    datePublished: string;
    dateModified: string;
    author: string;
  };
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

    // Helper function to recursively get all .md files
    const getAllMarkdownFiles = (
      dir: string,
      baseCategory: string
    ): { filePath: string; relativePath: string; category: string }[] => {
      const results: {
        filePath: string;
        relativePath: string;
        category: string;
      }[] = [];

      const items = fs.readdirSync(dir, { withFileTypes: true });

      for (const item of items) {
        const fullPath = path.join(dir, item.name);

        if (item.isDirectory()) {
          // Recursively search subdirectories
          results.push(...getAllMarkdownFiles(fullPath, baseCategory));
        } else if (item.isFile() && item.name.endsWith(".md")) {
          // Calculate relative path from category folder
          const categoryPath = path.join(contentDirectory, baseCategory);
          const relativePath = path.relative(categoryPath, fullPath);
          // Remove .md extension and use as slug
          const slug = relativePath.replace(/\.md$/, "");

          results.push({
            filePath: fullPath,
            relativePath: slug,
            category: baseCategory,
          });
        }
      }

      return results;
    };

    // Read directories inside blog folder (categories)
    const categoryFolders = fs
      .readdirSync(contentDirectory, { withFileTypes: true })
      .filter((dirent) => dirent.isDirectory())
      .map((dirent) => dirent.name);

    // Process files from each category folder (including nested subfolders)
    for (const categoryFolder of categoryFolders) {
      const categoryPath = path.join(contentDirectory, categoryFolder);
      const markdownFiles = getAllMarkdownFiles(categoryPath, categoryFolder);

      for (const { filePath, relativePath, category } of markdownFiles) {
        const fileContents = fs.readFileSync(filePath, "utf8");
        const { data, content } = matter(fileContents);

        // Calculate automatic read time based on content
        const automaticReadTime = calculateReadTimeWithTags(
          content,
          data.tags || [],
          data.category || category
        );

        // Generate auto-excerpt from content if not provided
        let excerpt = data.excerpt || data.description || "";
        if (!excerpt || excerpt.trim() === "") {
          // Extract first meaningful paragraph from content (skip headers, images, etc.)
          const lines = content.split("\n").filter((line) => {
            const trimmed = line.trim();
            return (
              trimmed.length > 0 &&
              !trimmed.startsWith("#") &&
              !trimmed.startsWith("!") &&
              !trimmed.startsWith("---") &&
              !trimmed.startsWith("```") &&
              !trimmed.startsWith("|") &&
              !trimmed.match(/^\[.*\]\(.*\)$/) // Skip standalone links
            );
          });

          // Get first paragraph text
          const firstParagraph = lines.slice(0, 3).join(" ");
          if (firstParagraph.length > 0) {
            // Clean up markdown syntax and limit to ~200 chars
            excerpt = firstParagraph
              .replace(/\*\*(.*?)\*\*/g, "$1") // Remove bold
              .replace(/\*(.*?)\*/g, "$1") // Remove italic
              .replace(/`(.*?)`/g, "$1") // Remove inline code
              .replace(/\[(.*?)\]\(.*?\)/g, "$1") // Replace links with text
              .replace(/\s+/g, " ") // Normalize whitespace
              .trim();

            if (excerpt.length > 200) {
              excerpt = excerpt.substring(0, 197) + "...";
            }
          } else {
            excerpt = "No excerpt available";
          }
        }

        posts.push({
          slug: `${category}/${relativePath}`,
          title: data.title || "Untitled",
          publishDate: data.publishDate || data.date || "2024-01-01",
          readTime: data.readTime || automaticReadTime.readTime,
          category: data.category || category,
          author: data.author || "Anonymous",
          tags: data.tags || [],
          image: data.image || "/images/default-blog.jpg",
          excerpt,
          collection: data.collection || data.subcategory,
          // Add SEO-friendly JSON-LD data
          seo: {
            type: "BlogPosting",
            datePublished: data.publishDate || data.date || "2024-01-01",
            dateModified:
              data.modifiedDate ||
              data.publishDate ||
              data.date ||
              "2024-01-01",
            author: data.author || "Anonymous",
          },
        });
      }
    }

    // Sort posts by date (newest first) - more robust date parsing
    posts.sort((a, b) => {
      const dateA = new Date(a.publishDate);
      const dateB = new Date(b.publishDate);

      // Handle invalid dates by putting them at the end
      if (isNaN(dateA.getTime()) && isNaN(dateB.getTime())) return 0;
      if (isNaN(dateA.getTime())) return 1;
      if (isNaN(dateB.getTime())) return -1;

      return dateB.getTime() - dateA.getTime();
    });

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
