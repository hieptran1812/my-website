import { NextRequest, NextResponse } from "next/server";
import { calculateReadTimeWithTags } from "../../../../lib/readTimeCalculator";
import { loadAllPosts } from "../../../../lib/blogIndex";

export interface Article {
  id: string;
  title: string;
  excerpt: string;
  content: string;
  author: string;
  date: string;
  publishDate: string;
  readTime: string;
  category: string;
  subcategory: string;
  tags: string[];
  difficulty: string;
  featured: boolean;
  slug: string;
  image?: string;
  collection?: string;
  aiGenerated?: boolean;
}

function convertToArticle(
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  metadata: any,
  slug: string,
  resolvedCategory: string,
  resolvedSubcategory: string,
  content?: string,
): Article {
  // Calculate read time from content if available
  let readTime = "5 min read";
  if (content) {
    const readTimeResult = calculateReadTimeWithTags(
      content,
      metadata.tags || [],
      resolvedCategory || "General",
    );
    readTime = readTimeResult.readTime;
  } else if (metadata.readTime) {
    readTime = metadata.readTime;
  }

  // Calculate difficulty based on read time
  let difficulty: "Beginner" | "Intermediate" | "Advanced" = "Beginner";
  const readTimeNum = parseInt(readTime.replace(/[^\\d]/g, ""));
  if (readTimeNum >= 10) difficulty = "Advanced";
  else if (readTimeNum >= 5) difficulty = "Intermediate";

  // Folder-derived subcategory wins; fall back to tag-based inference only if both
  // folder layout and frontmatter omit a subcategory.
  let subcategory = resolvedSubcategory || "General";

  if (!resolvedSubcategory && metadata.tags && metadata.tags.length > 0) {
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
  const uniqueId = resolvedCategory
    ? `${resolvedCategory}/${slug}-${timestamp}`
    : `${slug}-${timestamp}`;

  return {
    id: uniqueId, // Use the enhanced uniqueId
    title: metadata.title || "Untitled",
    excerpt: metadata.excerpt || metadata.description || "",
    content: metadata.content || "",
    author: metadata.author || "",
    date:
      metadata.publishDate ||
      metadata.date ||
      "",
    publishDate:
      metadata.publishDate ||
      metadata.date ||
      "",
    readTime: readTime,
    category: resolvedCategory,
    subcategory,
    tags: metadata.tags || [],
    difficulty,
    featured,
    slug, // Original slug remains for URL purposes
    image: metadata.image || "/blog-placeholder.jpg", // Add image handling
    collection: metadata.collection,
    aiGenerated: metadata.aiGenerated === true,
  };
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const categoryFilter = searchParams.get("category");
    const subcategoryFilter = searchParams.get("subcategory");
    const page = parseInt(searchParams.get("page") || "1");
    const limit = parseInt(searchParams.get("limit") || "500");
    const excludeContent = searchParams.get("excludeContent") === "true";

    const corpus = await loadAllPosts();
    const articles: Article[] = [];

    for (const entry of corpus) {
      const article = convertToArticle(
        entry.frontmatter,
        entry.slug,
        entry.category,
        entry.subcategory,
        entry.content,
      );
      article.id = entry.slug.replace(/\//g, "-");
      article.content = entry.content;

      if (categoryFilter && article.category !== categoryFilter) continue;
      if (subcategoryFilter && article.subcategory !== subcategoryFilter)
        continue;
      articles.push(article);
    }

    // Sort articles by date (newest first) - more robust date parsing
    articles.sort((a, b) => {
      const dateA = new Date(a.date);
      const dateB = new Date(b.date);

      // Handle invalid dates by putting them at the end
      if (isNaN(dateA.getTime()) && isNaN(dateB.getTime())) return 0;
      if (isNaN(dateA.getTime())) return 1;
      if (isNaN(dateB.getTime())) return -1;

      return dateB.getTime() - dateA.getTime();
    });

    // Apply pagination
    const startIndex = (page - 1) * limit;
    const endIndex = startIndex + limit;
    const paginatedArticles = articles.slice(startIndex, endIndex);

    // Strip content field when not needed (saves significant bandwidth for search)
    const responseArticles = excludeContent
      ? // eslint-disable-next-line @typescript-eslint/no-unused-vars
        paginatedArticles.map(({ content, ...rest }) => rest)
      : paginatedArticles;

    return NextResponse.json(
      {
        articles: responseArticles,
        total: articles.length,
        page,
        limit,
        hasMore: endIndex < articles.length,
      },
      {
        headers: {
          "Cache-Control":
            "public, s-maxage=3600, stale-while-revalidate=86400",
        },
      },
    );
  } catch (error) {
    console.error("Error fetching articles:", error);
    return NextResponse.json(
      { articles: [], total: 0, page: 1, limit: 50, hasMore: false },
      { status: 500 },
    );
  }
}
