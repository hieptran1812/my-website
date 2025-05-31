import { NextRequest, NextResponse } from "next/server";
import fs from "fs";
import path from "path";
import matter from "gray-matter";
import { remark } from "remark";
import remarkGfm from "remark-gfm";
import remarkHtml from "remark-html";
import { Article } from "../articles/route";

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const slug = searchParams.get("slug");

    if (!slug) {
      return NextResponse.json(
        { error: "Slug parameter is required" },
        { status: 400 }
      );
    }

    const blogDir = path.join(process.cwd(), "content", "blog");

    // Function to search for the file based on slug parts
    const findArticleFile = (
      baseDir: string,
      slugParts: string[]
    ): string | null => {
      if (slugParts.length === 0) return null;

      // If we're at the last part of the slug, look for a matching markdown file
      if (slugParts.length === 1) {
        const fileName = `${slugParts[0]}.md`;
        const filePath = path.join(baseDir, fileName);

        if (fs.existsSync(filePath)) {
          return filePath;
        }
        return null;
      }

      // Otherwise, check if the first part is a directory and recurse
      const dirPath = path.join(baseDir, slugParts[0]);
      if (fs.existsSync(dirPath) && fs.statSync(dirPath).isDirectory()) {
        return findArticleFile(dirPath, slugParts.slice(1));
      }

      return null;
    };

    // Split the slug into parts and search for the file
    const slugParts = slug.split("/");
    const articlePath = findArticleFile(blogDir, slugParts);

    if (!articlePath) {
      return NextResponse.json({ error: "Article not found" }, { status: 404 });
    }

    // Read and parse the article
    const fileContent = fs.readFileSync(articlePath, "utf8");
    const { data: metadata, content: markdownContent } = matter(fileContent);

    // Process markdown to HTML
    const processedContent = await remark()
      .use(remarkGfm)
      .use(remarkHtml, { sanitize: false })
      .process(markdownContent);

    const htmlContent = processedContent.toString();

    // Determine category from directory structure if not specified
    let category = metadata.category;
    if (!category && slugParts.length > 1) {
      category = slugParts[0];
    }

    const article: Article = {
      id: slug,
      title: metadata.title || "Untitled",
      excerpt: metadata.excerpt || metadata.description || "",
      content: htmlContent,
      author: metadata.author || "Hiep Tran",
      date: metadata.date || new Date().toISOString().split("T")[0],
      readTime: metadata.readTime || "5 min read",
      category: category || "General",
      subcategory: metadata.subcategory || "General",
      tags: metadata.tags || [],
      difficulty: metadata.difficulty || "Beginner",
      featured: metadata.featured || false,
      slug,
    };

    return NextResponse.json({ article });
  } catch (error) {
    console.error("Error fetching article:", error);
    return NextResponse.json(
      { error: "Failed to fetch article" },
      { status: 500 }
    );
  }
}
