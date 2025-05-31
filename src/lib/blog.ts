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
}

// Get markdown articles by category using API route (client-side function)
export async function getMarkdownArticlesByCategory(
  targetCategory: string
): Promise<Article[]> {
  try {
    // Use window.location.origin to get the full URL in browser environment
    const baseUrl =
      typeof window !== "undefined"
        ? window.location.origin
        : process.env.NEXT_PUBLIC_BASE_URL || "http://localhost:3000";
    const response = await fetch(
      `${baseUrl}/api/blog/articles?category=${targetCategory}`
    );

    if (!response.ok) {
      throw new Error(`Failed to fetch articles: ${response.status}`);
    }

    const data = await response.json();
    return data.articles || [];
  } catch (error) {
    console.error("Error fetching articles:", error);
    return [];
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
  frontmatter: any = {}
): { processedContent: string; readTime: string } {
  const readTimeResult = calculateReadTimeWithTags(
    content,
    frontmatter.tags || [],
    frontmatter.category || "general"
  );

  return {
    processedContent: content,
    readTime: frontmatter.readTime || readTimeResult.readTime,
  };
}

// Quick utility to get just the readTime for any content
export function calculateContentReadTime(
  content: string,
  tags: string[] = [],
  category: string = "general"
): string {
  return getReadTime(content, { wordsPerMinute: 200 });
}
