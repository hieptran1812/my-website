// Article data and utilities for the blog
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

// Centralized article data
const articles: Article[] = [];

// Utility functions
export function getAllArticles(): Article[] {
  return articles.sort(
    (a, b) => new Date(b.date).getTime() - new Date(a.date).getTime()
  );
}

export function getArticlesByCategory(category: string): Article[] {
  return articles
    .filter((article) => article.category === category)
    .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
}

export function getArticlesBySubcategory(subcategory: string): Article[] {
  return articles
    .filter((article) => article.subcategory === subcategory)
    .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
}

export function getFeaturedArticles(): Article[] {
  return articles
    .filter((article) => article.featured)
    .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
}

export function getArticleBySlug(slug: string): Article | undefined {
  return articles.find((article) => article.slug === slug);
}

export function getArticleById(id: string): Article | undefined {
  return articles.find((article) => article.id === id);
}

export function getRelatedArticles(
  currentArticle: Article,
  limit: number = 3
): Article[] {
  return articles
    .filter(
      (article) =>
        article.id !== currentArticle.id &&
        (article.category === currentArticle.category ||
          article.tags.some((tag) => currentArticle.tags.includes(tag)))
    )
    .sort((a, b) => {
      // Score based on matching tags and category
      const aScore =
        (a.category === currentArticle.category ? 2 : 0) +
        a.tags.filter((tag) => currentArticle.tags.includes(tag)).length;
      const bScore =
        (b.category === currentArticle.category ? 2 : 0) +
        b.tags.filter((tag) => currentArticle.tags.includes(tag)).length;
      return bScore - aScore;
    })
    .slice(0, limit);
}

export function getCategories(): {
  name: string;
  slug: string;
  count: number;
}[] {
  const categoryMap = new Map<string, number>();

  articles.forEach((article) => {
    categoryMap.set(
      article.category,
      (categoryMap.get(article.category) || 0) + 1
    );
  });

  return Array.from(categoryMap.entries()).map(([category, count]) => ({
    name: category
      .split("-")
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" "),
    slug: category,
    count,
  }));
}

export function getSubcategories(
  category?: string
): { name: string; slug: string; count: number }[] {
  const subcategoryMap = new Map<string, number>();

  articles
    .filter((article) => !category || article.category === category)
    .forEach((article) => {
      if (article.subcategory) {
        subcategoryMap.set(
          article.subcategory,
          (subcategoryMap.get(article.subcategory) || 0) + 1
        );
      }
    });

  return Array.from(subcategoryMap.entries()).map(([subcategory, count]) => ({
    name: subcategory,
    slug: subcategory,
    count,
  }));
}

export function getTags(): { name: string; count: number }[] {
  const tagMap = new Map<string, number>();

  articles.forEach((article) => {
    article.tags.forEach((tag) => {
      tagMap.set(tag, (tagMap.get(tag) || 0) + 1);
    });
  });

  return Array.from(tagMap.entries())
    .map(([tag, count]) => ({ name: tag, count }))
    .sort((a, b) => b.count - a.count);
}

export function searchArticles(query: string): Article[] {
  const searchTerm = query.toLowerCase();

  return articles
    .filter(
      (article) =>
        article.title.toLowerCase().includes(searchTerm) ||
        article.excerpt.toLowerCase().includes(searchTerm) ||
        article.tags.some((tag) => tag.toLowerCase().includes(searchTerm)) ||
        article.category.toLowerCase().includes(searchTerm) ||
        (article.subcategory &&
          article.subcategory.toLowerCase().includes(searchTerm))
    )
    .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
}
