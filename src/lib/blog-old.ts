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
export async function getMarkdownArticlesByCategory(targetCategory: string): Promise<Article[]> {
  try {
    const response = await fetch(`/api/blog/articles?category=${encodeURIComponent(targetCategory)}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch articles: ${response.statusText}`);
    }
    const data = await response.json();
    return data.articles || [];
  } catch (error) {
    console.error('Error fetching markdown articles:', error);
    return [];
  }
}

const postsDirectory = path.join(process.cwd(), "content/blog");

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

// Get all blog post metadata (for listing pages)
export function getAllBlogPosts(): BlogPostMetadata[] {
  try {
    const fileNames = fs.readdirSync(postsDirectory);
    const allPostsData = fileNames
      .filter((fileName) => fileName.endsWith(".md"))
      .map((fileName) => {
        const slug = fileName.replace(/\.md$/, "");
        const fullPath = path.join(postsDirectory, fileName);
        const fileContents = fs.readFileSync(fullPath, "utf8");
        const matterResult = matter(fileContents);

        return {
          slug,
          ...matterResult.data,
        } as BlogPostMetadata;
      });

    // Sort posts by date (newest first)
    return allPostsData.sort((a, b) => {
      return (
        new Date(b.publishDate).getTime() - new Date(a.publishDate).getTime()
      );
    });
  } catch (error) {
    console.error("Error reading blog posts:", error);
    return [];
  }
}

// Get a single blog post by slug
export async function getBlogPost(slug: string): Promise<BlogPost | null> {
  try {
    const fullPath = path.join(postsDirectory, `${slug}.md`);

    if (!fs.existsSync(fullPath)) {
      return null;
    }

    const fileContents = fs.readFileSync(fullPath, "utf8");
    const matterResult = matter(fileContents);

    // Process markdown content to HTML
    const processedContent = await remark()
      .use(remarkGfm)
      .use(html, { sanitize: false })
      .process(matterResult.content);

    const contentHtml = processedContent.toString();

    return {
      slug,
      content: contentHtml,
      ...matterResult.data,
    } as BlogPost;
  } catch (error) {
    console.error(`Error reading blog post ${slug}:`, error);
    return null;
  }
}

// Get blog posts by category
export function getBlogPostsByCategory(category: string): BlogPostMetadata[] {
  const allPosts = getAllBlogPosts();
  return allPosts.filter(
    (post) => post.category.toLowerCase() === category.toLowerCase()
  );
}

// Get blog posts by tag
export function getBlogPostsByTag(tag: string): BlogPostMetadata[] {
  const allPosts = getAllBlogPosts();
  return allPosts.filter((post) =>
    post.tags.some((postTag) => postTag.toLowerCase() === tag.toLowerCase())
  );
}

// Get all unique categories
export function getAllCategories(): string[] {
  const allPosts = getAllBlogPosts();
  const categories = new Set<string>();

  allPosts.forEach((post) => {
    categories.add(post.category);
  });

  return Array.from(categories).sort();
}

// Get all unique tags
export function getAllTags(): string[] {
  const allPosts = getAllBlogPosts();
  const tags = new Set<string>();

  allPosts.forEach((post) => {
    post.tags.forEach((tag) => {
      tags.add(tag);
    });
  });

  return Array.from(tags).sort();
}

// Get featured posts (latest 4 posts)
export function getFeaturedBlogPosts(limit: number = 4): BlogPostMetadata[] {
  const allPosts = getAllBlogPosts();
  return allPosts.slice(0, limit);
}

// Search blog posts
export function searchBlogPosts(query: string): BlogPostMetadata[] {
  const allPosts = getAllBlogPosts();
  const searchTerm = query.toLowerCase();

  return allPosts.filter(
    (post) =>
      post.title.toLowerCase().includes(searchTerm) ||
      post.excerpt.toLowerCase().includes(searchTerm) ||
      post.tags.some((tag) => tag.toLowerCase().includes(searchTerm)) ||
      post.category.toLowerCase().includes(searchTerm)
  );
}

// Check if a blog post exists
export function blogPostExists(slug: string): boolean {
  const fullPath = path.join(postsDirectory, `${slug}.md`);
  return fs.existsSync(fullPath);
}

// Get related posts based on tags and category
export function getRelatedPosts(
  currentSlug: string,
  limit: number = 3
): BlogPostMetadata[] {
  const allPosts = getAllBlogPosts();
  const currentPost = allPosts.find((post) => post.slug === currentSlug);

  if (!currentPost) {
    return [];
  }

  // Score posts based on similarity
  const scoredPosts = allPosts
    .filter((post) => post.slug !== currentSlug)
    .map((post) => {
      let score = 0;

      // Same category gets higher score
      if (post.category === currentPost.category) {
        score += 10;
      }

      // Shared tags get points
      const sharedTags = post.tags.filter((tag) =>
        currentPost.tags.includes(tag)
      );
      score += sharedTags.length * 5;

      return { post, score };
    })
    .filter((item) => item.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
    .map((item) => item.post);

  return scoredPosts;
}

// Convert BlogPostMetadata to Article format
function convertToArticle(post: BlogPostMetadata): Article {
  // Determine difficulty based on readTime or tags
  let difficulty: "Beginner" | "Intermediate" | "Advanced" = "Intermediate";
  const readTimeNumber = parseInt(post.readTime);
  if (readTimeNumber < 10) {
    difficulty = "Beginner";
  } else if (readTimeNumber > 20) {
    difficulty = "Advanced";
  }

  // Determine if featured (can be based on recency or specific tags)
  const featured =
    post.tags.includes("Featured") ||
    new Date(post.publishDate) > new Date("2024-03-01");

  // Extract subcategory from tags or use a default mapping
  const subcategoryMappings: { [key: string]: string } = {
    "Machine Learning":
      post.tags.find((tag) =>
        ["Deep Learning", "NLP", "Computer Vision", "LLM"].includes(tag)
      ) || "Deep Learning",
    "Software Development":
      post.tags.find((tag) =>
        ["Best Practices", "Architecture", "System Design", "DevOps", "Frontend", "Backend"].includes(tag)
      ) || "Best Practices",
    "Paper Reading":
      post.tags.find((tag) =>
        ["Computer Vision", "NLP", "Machine Learning", "Deep Learning"].includes(tag)
      ) || "Machine Learning",
    "Crypto":
      post.tags.find((tag) =>
        ["DeFi", "Fundamentals", "Technology", "NFTs"].includes(tag)
      ) || "Fundamentals",
    "Notes":
      post.tags.find((tag) =>
        ["Frontend", "Development", "DevOps", "Backend", "Interview Prep"].includes(tag)
      ) || "Development",
  };

  const subcategory = subcategoryMappings[post.category];

  return {
    id: post.slug,
    title: post.title,
    excerpt: post.excerpt,
    content: "",
    category: post.category.toLowerCase().replace(/\s+/g, "-"),
    subcategory,
    tags: post.tags,
    date: post.publishDate,
    readTime: post.readTime,
    difficulty,
    slug: post.slug,
    featured,
    author: post.author,
    image: post.image,
  };
}

// Get markdown articles by category and convert to Article format
export function getMarkdownArticlesByCategory(targetCategory: string): Article[] {
  const allPosts = getAllBlogPosts();

  // Category mapping to handle different naming conventions
  const categoryMappings: { [key: string]: string[] } = {
    "machine-learning": ["Machine Learning", "machine-learning"],
    "software-development": [
      "Software Development",
      "software-development",
      "Engineering",
    ],
    "paper-reading": ["Paper Reading"],
    "crypto": ["Crypto"],
    "notes": ["Notes", "notes"],
  };

  const validCategories = categoryMappings[targetCategory] || [targetCategory];

  return allPosts
    .filter((post) => validCategories.includes(post.category))
    .map(convertToArticle)
    .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
}
