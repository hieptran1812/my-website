import { NextRequest, NextResponse } from "next/server";
import fs from "fs";
import path from "path";
import matter from "gray-matter";

export interface Article {
  id: string;
  title: string;
  excerpt: string;
  content: string;
  author: string;
  date: string;
  readTime: string;
  category: string;
  subcategory: string;
  tags: string[];
  difficulty: "Beginner" | "Intermediate" | "Advanced";
  featured: boolean;
  slug: string;
}

function convertToArticle(
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  metadata: any,
  slug: string,
  category?: string
): Article {
  // Calculate difficulty based on read time
  let difficulty: "Beginner" | "Intermediate" | "Advanced" = "Beginner";
  if (metadata.readTime) {
    const readTimeNum = parseInt(metadata.readTime.replace(/[^\\d]/g, ""));
    if (readTimeNum >= 10) difficulty = "Advanced";
    else if (readTimeNum >= 5) difficulty = "Intermediate";
  }

  // Use explicit subcategory from metadata if available, otherwise extract from tags
  let subcategory = metadata.subcategory || category || "General";

  // Only auto-generate subcategory from tags if no explicit subcategory is provided
  if (!metadata.subcategory && metadata.tags && metadata.tags.length > 0) {
    // Use the first tag as subcategory, or map common patterns
    const firstTag = metadata.tags[0];
    if (
      firstTag.includes("Deep Learning") ||
      firstTag.includes("Neural Networks")
    ) {
      subcategory = "Deep Learning";
    } else if (firstTag.includes("NLP") || firstTag.includes("Transformer")) {
      subcategory = "NLP";
    } else if (
      firstTag.includes("Computer Vision") ||
      firstTag.includes("CNN")
    ) {
      subcategory = "Computer Vision";
    } else if (firstTag.includes("React") || firstTag.includes("Frontend")) {
      subcategory = "Frontend";
    } else if (firstTag.includes("Backend") || firstTag.includes("API")) {
      subcategory = "Backend";
    } else if (firstTag.includes("DevOps") || firstTag.includes("Docker")) {
      subcategory = "DevOps";
    } else if (
      firstTag.includes("DeFi") ||
      firstTag.includes("Decentralized")
    ) {
      subcategory = "DeFi";
    } else if (
      firstTag.includes("Blockchain") ||
      firstTag.includes("Technology")
    ) {
      subcategory = "Technology";
    } else if (firstTag.includes("NFT")) {
      subcategory = "NFTs";
    } else if (
      firstTag.includes("Fundamentals") ||
      firstTag.includes("Basics")
    ) {
      subcategory = "Fundamentals";
    } else {
      subcategory = firstTag;
    }
  }

  // Determine if article is featured (recent articles)
  const articleDate = new Date(metadata.date || Date.now());
  const monthsAgo = new Date();
  monthsAgo.setMonth(monthsAgo.getMonth() - 3);
  const featured = articleDate > monthsAgo;

  // Ensure unique ID by including timestamp and path for better uniqueness
  const timestamp = metadata.date
    ? new Date(metadata.date).getTime()
    : Date.now();
  const uniqueId = category
    ? `${category}/${slug}-${timestamp}`
    : `${slug}-${timestamp}`;

  return {
    id: uniqueId, // Use the enhanced uniqueId
    title: metadata.title || "Untitled",
    excerpt: metadata.excerpt || metadata.description || "",
    content: metadata.content || "",
    author: metadata.author || "Hiep Tran",
    date: metadata.date || new Date().toISOString().split("T")[0],
    readTime: metadata.readTime || "5 min read",
    category: metadata.category || category || "General",
    subcategory,
    tags: metadata.tags || [],
    difficulty,
    featured,
    slug, // Original slug remains for URL purposes
  };
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const categoryFilter = searchParams.get("category");
    const subcategoryFilter = searchParams.get("subcategory");
    const page = parseInt(searchParams.get("page") || "1");
    const limit = parseInt(searchParams.get("limit") || "50");

    const contentDir = path.join(process.cwd(), "content", "blog");
    const articles: Article[] = [];
    const processedFiles = new Set<string>();

    const readArticlesFromDir = (
      dir: string,
      currentCategory?: string,
      basePath = ""
    ) => {
      const entries = fs.readdirSync(dir, { withFileTypes: true });
      for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);
        if (entry.isDirectory()) {
          const newBasePath = basePath
            ? `${basePath}/${entry.name}`
            : entry.name;
          readArticlesFromDir(fullPath, entry.name, newBasePath);
        } else if (entry.name.endsWith(".md")) {
          const fileKey = basePath ? `${basePath}/${entry.name}` : entry.name;

          if (processedFiles.has(fileKey)) {
            continue;
          }
          processedFiles.add(fileKey);

          const fileContent = fs.readFileSync(fullPath, "utf8");
          const { data: metadata, content: fileMatterContent } =
            matter(fileContent);
          const fileName = entry.name.replace(/\.md$/, "");
          const slug = basePath ? `${basePath}/${fileName}` : fileName;

          // Create unique ID using full path and file stats
          const stats = fs.statSync(fullPath);
          const uniqueId = `${slug.replace(/\//g, "-")}-${stats.mtimeMs}`;

          const article = convertToArticle(metadata, slug, currentCategory);
          article.id = uniqueId; // Override with truly unique ID
          article.content = fileMatterContent;

          if (categoryFilter && article.category !== categoryFilter) {
            continue;
          }
          if (subcategoryFilter && article.subcategory !== subcategoryFilter) {
            continue;
          }
          articles.push(article);
        }
      }
    };

    readArticlesFromDir(contentDir);

    // Sort articles by date (newest first)
    articles.sort(
      (a, b) => new Date(b.date).getTime() - new Date(a.date).getTime()
    );

    // Apply pagination
    const startIndex = (page - 1) * limit;
    const endIndex = startIndex + limit;
    const paginatedArticles = articles.slice(startIndex, endIndex);

    return NextResponse.json({
      articles: paginatedArticles,
      total: articles.length,
      page,
      limit,
      hasMore: endIndex < articles.length,
    });
  } catch (error) {
    console.error("Error fetching articles:", error);
    return NextResponse.json(
      { articles: [], total: 0, page: 1, limit: 50, hasMore: false },
      { status: 500 }
    );
  }
}
