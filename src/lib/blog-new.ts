// Utility functions for processing markdown blog posts
import { remark } from "remark";
import html from "remark-html";
import remarkGfm from "remark-gfm";

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
    const response = await fetch(
      `/api/blog/articles?category=${encodeURIComponent(targetCategory)}`
    );
    if (!response.ok) {
      throw new Error(`Failed to fetch articles: ${response.statusText}`);
    }
    const data = await response.json();
    return data.articles || [];
  } catch (error) {
    console.error("Error fetching markdown articles:", error);
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
