import { NextRequest, NextResponse } from "next/server";
import fs from "fs";
import path from "path";
import matter from "gray-matter";
import { calculateReadTimeWithTags } from "../../../../lib/readTimeCalculator";

export interface BlogPost {
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
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const category = searchParams.get("category");

    const blogDir = path.join(process.cwd(), "content", "blog");

    if (!fs.existsSync(blogDir)) {
      return NextResponse.json([]);
    }

    const posts: BlogPost[] = [];

    // Helper function to read markdown files from a directory
    const readMarkdownFiles = (dirPath: string, categoryPrefix?: string) => {
      if (!fs.existsSync(dirPath)) return;

      const files = fs.readdirSync(dirPath, { withFileTypes: true });

      for (const file of files) {
        // Process directories recursively
        if (file.isDirectory()) {
          readMarkdownFiles(
            path.join(dirPath, file.name),
            file.name // Use directory name as category
          );
        }
        // Process markdown files
        else if (file.name.endsWith(".md")) {
          const filePath = path.join(dirPath, file.name);
          const fileContent = fs.readFileSync(filePath, "utf8");
          const { data: metadata, content } = matter(fileContent);

          // Calculate read time from actual content
          const readTimeResult = calculateReadTimeWithTags(
            content,
            metadata.tags || [],
            categoryPrefix || metadata.category || "General"
          );

          // Generate slug in format: category/post-name
          const slugBase = file.name.replace(/\.md$/, "");
          const slug = categoryPrefix
            ? `${categoryPrefix}/${slugBase}`
            : slugBase;

          // Extract excerpt from content if not specified in frontmatter
          let excerpt = metadata.excerpt || "";
          if (!excerpt && content) {
            // Take first paragraph, limited to ~160 chars
            excerpt = content.split("\n\n")[0].substring(0, 160).trim();
            if (content.length > 160) excerpt += "...";
          }

          // Default image if not provided
          const defaultImage = "/images/blog/default-post.jpg";

          const post: BlogPost = {
            slug,
            title: metadata.title || "Untitled",
            publishDate:
              metadata.publishDate ||
              metadata.date ||
              new Date().toISOString().split("T")[0],
            readTime: readTimeResult.readTime,
            category: categoryPrefix || metadata.category || "General",
            author: metadata.author || "Hiep Tran",
            tags: metadata.tags || [],
            image: metadata.image || defaultImage,
            excerpt,
            collection: metadata.collection,
          };

          // Apply category filter if requested
          if (category) {
            const normalizedCategory = category.toLowerCase();
            const postCategory = post.category.toLowerCase();

            if (
              postCategory.includes(normalizedCategory) ||
              post.tags.some((tag) =>
                tag.toLowerCase().includes(normalizedCategory)
              ) ||
              slug.toLowerCase().includes(normalizedCategory)
            ) {
              posts.push(post);
            }
          } else {
            posts.push(post);
          }
        }
      }
    };

    // Start reading from the root blog directory
    readMarkdownFiles(blogDir);

    // Sort by date (newest first) - more robust date parsing
    posts.sort((a, b) => {
      const dateA = new Date(a.publishDate);
      const dateB = new Date(b.publishDate);

      // Handle invalid dates by putting them at the end
      if (isNaN(dateA.getTime()) && isNaN(dateB.getTime())) return 0;
      if (isNaN(dateA.getTime())) return 1;
      if (isNaN(dateB.getTime())) return -1;

      return dateB.getTime() - dateA.getTime();
    });

    return NextResponse.json(posts);
  } catch (error) {
    console.error("Error reading blog posts:", error);
    return NextResponse.json([]);
  }
}
