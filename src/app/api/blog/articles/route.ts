import { NextRequest, NextResponse } from "next/server";
import fs from "fs";
import path from "path";
import matter from "gray-matter";
import { remark } from "remark";
import remarkGfm from "remark-gfm";
import remarkHtml from "remark-html";

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

function convertToArticle(metadata: any, slug: string): Article {
  // Calculate difficulty based on read time
  let difficulty: "Beginner" | "Intermediate" | "Advanced" = "Beginner";
  if (metadata.readTime) {
    const readTimeNum = parseInt(metadata.readTime.replace(/[^\d]/g, ""));
    if (readTimeNum >= 10) difficulty = "Advanced";
    else if (readTimeNum >= 5) difficulty = "Intermediate";
  }

  // Extract subcategory from tags or use a default based on category
  let subcategory = "General";
  if (metadata.tags && metadata.tags.length > 0) {
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

  return {
    id: slug,
    title: metadata.title || "Untitled",
    excerpt: metadata.excerpt || metadata.description || "",
    content: metadata.content || "",
    author: metadata.author || "Hiep Tran",
    date: metadata.date || new Date().toISOString().split("T")[0],
    readTime: metadata.readTime || "5 min read",
    category: metadata.category || "General",
    subcategory,
    tags: metadata.tags || [],
    difficulty,
    featured,
    slug,
  };
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const category = searchParams.get("category");

    const blogDir = path.join(process.cwd(), "content", "blog");

    if (!fs.existsSync(blogDir)) {
      return NextResponse.json({ articles: [] });
    }

    const files = fs
      .readdirSync(blogDir)
      .filter((file) => file.endsWith(".md"));
    const articles: Article[] = [];

    for (const file of files) {
      const filePath = path.join(blogDir, file);
      const fileContent = fs.readFileSync(filePath, "utf8");
      const { data: metadata, content } = matter(fileContent);

      const slug = file.replace(".md", "");
      const article = convertToArticle({ ...metadata, content }, slug);

      // Category mapping to handle different naming conventions
      const categoryMap: { [key: string]: string[] } = {
        "machine-learning": [
          "Machine Learning",
          "ML",
          "machine-learning",
          "machine learning",
        ],
        "software-development": [
          "Software Development",
          "Development",
          "software-development",
          "software development",
        ],
        "paper-reading": [
          "Paper Reading",
          "Research",
          "paper-reading",
          "paper reading",
        ],
        crypto: ["Crypto", "Cryptocurrency", "Blockchain", "crypto"],
        notes: ["Notes", "Development Notes", "notes"],
      };

      // If category filter is provided, filter articles
      if (category) {
        const validCategories = categoryMap[category] || [category];
        if (
          validCategories.some(
            (cat) =>
              article.category.toLowerCase().includes(cat.toLowerCase()) ||
              article.tags.some((tag) =>
                tag.toLowerCase().includes(cat.toLowerCase())
              )
          )
        ) {
          articles.push(article);
        }
      } else {
        articles.push(article);
      }
    }

    // Sort by date (newest first)
    articles.sort(
      (a, b) => new Date(b.date).getTime() - new Date(a.date).getTime()
    );

    return NextResponse.json({ articles });
  } catch (error) {
    console.error("Error reading blog articles:", error);
    return NextResponse.json({ articles: [] });
  }
}
