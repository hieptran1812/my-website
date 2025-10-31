// Utility functions for processing markdown blog posts
import { remark } from "remark";
import html from "remark-html";
import remarkGfm from "remark-gfm";
import { calculateReadTimeWithTags, getReadTime } from "./readTimeCalculator";

// Article interface matching the one used in blog pages
export interface Article {
  id: string;
  title: string;
  excerpt: string;
  content: string;
  category: string;
  subcategory?: string;
  tags: string[];
  date: string;
  readTime: string;
  difficulty: "Beginner" | "Intermediate" | "Advanced";
  slug: string;
  featured: boolean;
  author?: string;
  image?: string;
  collection?: string;
}

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
  content: string;
  collection?: string;
}

export interface BlogPostMetadata {
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

// Get markdown articles by category using API route (client-side function)
export async function getMarkdownArticlesByCategory(
  targetCategory: string,
  page: number = 1,
  limit: number = 50
): Promise<{ articles: Article[]; total: number; hasMore: boolean }> {
  try {
    const baseUrl =
      typeof window !== "undefined"
        ? window.location.origin
        : process.env.NEXT_PUBLIC_BASE_URL || "http://localhost:3000";

    const params = new URLSearchParams({
      category: targetCategory,
      page: page.toString(),
      limit: limit.toString(),
    });

    const response = await fetch(`${baseUrl}/api/blog/articles?${params}`);

    if (!response.ok) {
      throw new Error(`Failed to fetch articles: ${response.status}`);
    }

    const data = await response.json();

    // Ensure we always return a proper structure with arrays
    return {
      articles: Array.isArray(data.articles) ? data.articles : [],
      total: data.total || 0,
      hasMore: data.hasMore || false,
    };
  } catch (error) {
    console.error("Error fetching articles:", error);
    return { articles: [], total: 0, hasMore: false };
  }
}

// Process markdown content to HTML
export async function processMarkdown(content: string): Promise<string> {
  const processedContent = await remark()
    .use(remarkGfm)
    .use(html, { sanitize: false })
    .process(content);

  return processedContent.toString();
}

// Utility function to process blog post content and calculate readTime
export function processBlogPostWithReadTime(
  content: string,
  frontmatter: Record<string, unknown> = {}
): { processedContent: string; readTime: string } {
  const readTimeResult = calculateReadTimeWithTags(
    content,
    (frontmatter.tags as string[]) || [],
    (frontmatter.category as string) || "general"
  );

  return {
    processedContent: content,
    readTime: (frontmatter.readTime as string) || readTimeResult.readTime,
  };
}

// Quick utility to get just the readTime for any content
export function calculateContentReadTime(content: string): string {
  return getReadTime(content, { wordsPerMinute: 200 });
}
