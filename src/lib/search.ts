// Search utility module for integrating real blog and project data
import { getAllArticles } from "../data/articles";
import type { Article } from "../data/articles";

// Define the project interface based on the projects data structure
export interface Project {
  id: number;
  title: string;
  description: string;
  category: string;
  technologies: string[];
  githubUrl?: string;
  liveUrl?: string | null;
  featured: boolean;
  status: string;
  highlights: string[];
  image?: string;
}

// Define unified search result interface
export interface SearchResult {
  title: string;
  description: string;
  url: string;
  type: "blog" | "page" | "project";
  category?: string;
  tags?: string[];
  relevanceScore?: number;
  date?: string;
  difficulty?: string;
  readTime?: string;
  technologies?: string[];
  status?: string;
  highlights?: string[];
  featured?: boolean;
}

// Static page data
const staticPages: SearchResult[] = [
  {
    title: "About",
    description:
      "Learn more about Hiep Tran - AI Research Engineer and Full-Stack Developer specializing in machine learning and web development.",
    url: "/about",
    type: "page",
    tags: ["About", "Profile", "Background"],
  },
  {
    title: "Projects",
    description:
      "Explore my portfolio of AI research projects, web applications, and open source contributions.",
    url: "/projects",
    type: "page",
    tags: ["Portfolio", "Projects", "AI", "Web Development"],
  },
  {
    title: "Contact",
    description:
      "Get in touch for collaboration opportunities, research discussions, or project inquiries.",
    url: "/contact",
    type: "page",
    tags: ["Contact", "Collaboration", "Hire"],
  },
];

// Transform articles to search results
export const getArticleSearchResults = (): SearchResult[] => {
  const articles = getAllArticles();
  return articles.map((article: Article) => ({
    title: article.title,
    description: article.excerpt,
    url: `/blog/${article.slug}`, // slug already includes category path
    type: "blog" as const,
    category: article.category,
    tags: article.tags,
    date: article.date,
    difficulty: article.difficulty,
    readTime: article.readTime,
    featured: article.featured,
  }));
};

// Transform projects to search results
export const getProjectSearchResults = (
  projects: Project[]
): SearchResult[] => {
  return projects.map((project: Project) => ({
    title: project.title,
    description: project.description,
    url: `/projects#project-${project.id}`,
    type: "project" as const,
    category: project.category,
    tags: project.technologies,
    technologies: project.technologies,
    status: project.status,
    highlights: project.highlights,
    featured: project.featured,
  }));
};

// Get all searchable content
export const getAllSearchableContent = (
  projects: Project[] = []
): SearchResult[] => {
  const articleResults = getArticleSearchResults();
  const projectResults = getProjectSearchResults(projects);

  return [...articleResults, ...projectResults, ...staticPages];
};

// Enhanced relevance scoring algorithm
export const calculateRelevance = (
  item: SearchResult,
  query: string
): number => {
  // Normalize input
  const normalizedQuery = query.toLowerCase().trim();
  const queryWords = normalizedQuery
    .split(/\s+/)
    .filter((word) => word.length > 0);

  // If query is empty, return 0
  if (queryWords.length === 0) return 0;

  let score = 0;

  // Normalize content fields
  const normalizedTitle = item.title.toLowerCase();
  const normalizedDescription = item.description.toLowerCase();
  const normalizedCategory = item.category?.toLowerCase() || "";
  const normalizedTags = item.tags?.map((tag) => tag.toLowerCase()) || [];
  const normalizedTechnologies =
    item.technologies?.map((tech) => tech.toLowerCase()) || [];
  const normalizedHighlights = item.highlights?.join(" ").toLowerCase() || "";

  // Exact phrase matching (highest weight)
  if (normalizedTitle.includes(normalizedQuery)) {
    score += 100;
  }
  if (normalizedDescription.includes(normalizedQuery)) {
    score += 80;
  }
  if (normalizedCategory.includes(normalizedQuery)) {
    score += 60;
  }

  // Individual word matching with position weighting
  queryWords.forEach((word) => {
    // Title matches (high weight)
    if (normalizedTitle.includes(word)) {
      const titleIndex = normalizedTitle.indexOf(word);
      // Higher score for words appearing earlier in title
      score += Math.max(50 - titleIndex * 2, 20);
    }

    // Description matches (medium weight)
    if (normalizedDescription.includes(word)) {
      const descIndex = normalizedDescription.indexOf(word);
      score += Math.max(30 - descIndex * 0.1, 10);
    }

    // Category matches (medium-high weight)
    if (normalizedCategory.includes(word)) {
      score += 40;
    }

    // Tag matches (medium weight)
    normalizedTags.forEach((tag) => {
      if (tag.includes(word)) {
        score += 25;
      }
    });

    // Technology matches (medium weight for projects)
    normalizedTechnologies.forEach((tech) => {
      if (tech.includes(word)) {
        score += 25;
      }
    });

    // Highlights matches (low-medium weight)
    if (normalizedHighlights.includes(word)) {
      score += 15;
    }
  });

  // Fuzzy matching for single words (lower weight)
  if (queryWords.length === 1) {
    const queryWord = queryWords[0];

    // Check for partial matches in title
    if (
      normalizedTitle
        .split(/\s+/)
        .some(
          (word) => word.startsWith(queryWord) || queryWord.startsWith(word)
        )
    ) {
      score += 15;
    }

    // Check for partial matches in tags/technologies
    [...normalizedTags, ...normalizedTechnologies].forEach((item) => {
      if (item.startsWith(queryWord) || queryWord.startsWith(item)) {
        score += 10;
      }
    });
  }

  // Multi-word query bonus - reward items that match multiple query words
  const matchedWords = queryWords.filter(
    (word) =>
      normalizedTitle.includes(word) ||
      normalizedDescription.includes(word) ||
      normalizedCategory.includes(word) ||
      normalizedTags.some((tag) => tag.includes(word)) ||
      normalizedTechnologies.some((tech) => tech.includes(word))
  );

  if (matchedWords.length > 1) {
    score += matchedWords.length * 20;
  }

  // Content type bonuses
  switch (item.type) {
    case "blog":
      score += 10; // Slight preference for blog content
      break;
    case "project":
      score += 15; // Higher preference for projects
      break;
    case "page":
      score += 5; // Lower preference for static pages
      break;
  }

  // Featured content bonus
  if (item.featured) {
    score += 20;
  }

  // Recency bonus for blog posts
  if (item.type === "blog" && item.date) {
    const articleDate = new Date(item.date);
    const now = new Date();
    const daysDiff =
      (now.getTime() - articleDate.getTime()) / (1000 * 60 * 60 * 24);

    // Bonus for articles published within the last year
    if (daysDiff < 365) {
      score += Math.max(10 - daysDiff / 36.5, 0);
    }
  }

  // Active project bonus
  if (item.type === "project" && item.status === "Production") {
    score += 15;
  } else if (item.type === "project" && item.status === "Active Development") {
    score += 10;
  }

  return Math.round(score * 100) / 100; // Round to 2 decimal places
};

// Search function with filtering and sorting
export const searchContent = (
  query: string,
  content: SearchResult[],
  typeFilter?: "blog" | "project" | "page" | null,
  minRelevance: number = 1
): SearchResult[] => {
  if (!query.trim()) {
    // Return all content if no query, filtered by type
    const filtered = typeFilter
      ? content.filter((item) => item.type === typeFilter)
      : content;

    // Sort by featured status and recency for blog posts
    return filtered.sort((a, b) => {
      if (a.featured && !b.featured) return -1;
      if (!a.featured && b.featured) return 1;

      if (a.type === "blog" && b.type === "blog" && a.date && b.date) {
        return new Date(b.date).getTime() - new Date(a.date).getTime();
      }

      return a.title.localeCompare(b.title);
    });
  }

  // Calculate relevance for all items
  const scoredResults = content
    .map((item) => ({
      ...item,
      relevanceScore: calculateRelevance(item, query),
    }))
    .filter((item) => item.relevanceScore >= minRelevance);

  // Apply type filter if specified
  const filteredResults = typeFilter
    ? scoredResults.filter((item) => item.type === typeFilter)
    : scoredResults;

  // Sort by relevance score (descending)
  return filteredResults.sort(
    (a, b) => (b.relevanceScore || 0) - (a.relevanceScore || 0)
  );
};

// Utility to highlight matching text
export const highlightMatch = (text: string, query: string): string => {
  if (!query.trim()) return text;

  const normalizedQuery = query.toLowerCase();
  const queryWords = normalizedQuery
    .split(/\s+/)
    .filter((word) => word.length > 0);

  let highlightedText = text;

  // Highlight exact phrase first
  const exactIndex = text.toLowerCase().indexOf(normalizedQuery);
  if (exactIndex !== -1) {
    const before = text.substring(0, exactIndex);
    const match = text.substring(
      exactIndex,
      exactIndex + normalizedQuery.length
    );
    const after = text.substring(exactIndex + normalizedQuery.length);

    return `${before}<mark class="bg-blue-200 dark:bg-blue-800 px-1 rounded">${match}</mark>${after}`;
  }

  // Highlight individual words
  queryWords.forEach((word) => {
    const regex = new RegExp(`(${word})`, "gi");
    highlightedText = highlightedText.replace(
      regex,
      '<mark class="bg-blue-200 dark:bg-blue-800 px-1 rounded">$1</mark>'
    );
  });

  return highlightedText;
};
