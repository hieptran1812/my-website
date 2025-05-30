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
const articles: Article[] = [
  // Crypto articles
  {
    id: "defi-fundamentals",
    title: "DeFi Fundamentals: Understanding Decentralized Finance",
    excerpt:
      "A comprehensive guide to DeFi protocols, yield farming, and the future of decentralized financial services.",
    content: "Detailed content about DeFi fundamentals...",
    category: "crypto",
    subcategory: "DeFi",
    tags: ["DeFi", "Yield Farming", "Smart Contracts"],
    date: "2024-03-15",
    readTime: "12 min read",
    difficulty: "Intermediate",
    slug: "defi-fundamentals",
    featured: true,
    author: "Hiep Tran",
    image: "/blog-placeholder.jpg",
  },
  {
    id: "blockchain-consensus",
    title: "Blockchain Consensus Mechanisms Explained",
    excerpt:
      "Deep dive into Proof of Work, Proof of Stake, and other consensus algorithms that secure blockchain networks.",
    content: "Detailed content about consensus mechanisms...",
    category: "crypto",
    subcategory: "Fundamentals",
    tags: ["Blockchain", "Consensus", "PoW", "PoS"],
    date: "2024-03-10",
    readTime: "18 min read",
    difficulty: "Advanced",
    slug: "blockchain-consensus",
    featured: false,
    author: "Hiep Tran",
    image: "/blog-placeholder.jpg",
  },
  {
    id: "smart-contracts-101",
    title: "Smart Contracts 101: Building on Ethereum",
    excerpt:
      "Learn the basics of smart contract development, from Solidity syntax to deployment on Ethereum.",
    content: "Detailed content about smart contracts...",
    category: "crypto",
    subcategory: "Technology",
    tags: ["Smart Contracts", "Solidity", "Ethereum", "Web3"],
    date: "2024-02-20",
    readTime: "20 min read",
    difficulty: "Intermediate",
    slug: "smart-contracts-101",
    featured: true,
    author: "Hiep Tran",
    image: "/blog-placeholder.jpg",
  },

  // Software Development articles
  {
    id: "typescript-best-practices",
    title: "TypeScript Best Practices for Large Applications",
    excerpt:
      "Essential patterns and practices for building scalable TypeScript applications with proper type safety.",
    content: "Detailed content about TypeScript best practices...",
    category: "software-development",
    subcategory: "Frontend",
    tags: ["TypeScript", "Best Practices", "Scalability"],
    date: "2024-03-12",
    readTime: "15 min read",
    difficulty: "Intermediate",
    slug: "typescript-best-practices",
    featured: true,
    author: "Hiep Tran",
    image: "/blog-placeholder.jpg",
  },
  {
    id: "microservices-architecture",
    title: "Microservices Architecture: Design Patterns and Trade-offs",
    excerpt:
      "Exploring microservices patterns, when to use them, and how to handle distributed system challenges.",
    content: "Detailed content about microservices...",
    category: "software-development",
    subcategory: "Backend",
    tags: ["Microservices", "Architecture", "Distributed Systems"],
    date: "2024-02-25",
    readTime: "22 min read",
    difficulty: "Advanced",
    slug: "microservices-architecture",
    featured: false,
    author: "Hiep Tran",
    image: "/blog-placeholder.jpg",
  },

  // Machine Learning articles
  {
    id: "transformer-architecture",
    title: "Understanding Transformer Architecture in Deep Learning",
    excerpt:
      "A detailed breakdown of the Transformer model that revolutionized NLP and beyond.",
    content: "Detailed content about transformer architecture...",
    category: "machine-learning",
    subcategory: "Deep Learning",
    tags: ["Transformers", "NLP", "Attention", "BERT", "GPT"],
    date: "2024-03-08",
    readTime: "25 min read",
    difficulty: "Advanced",
    slug: "transformer-architecture",
    featured: true,
    author: "Hiep Tran",
    image: "/blog-placeholder.jpg",
  },
  {
    id: "mlops-pipeline",
    title: "Building Production-Ready ML Pipelines",
    excerpt:
      "Best practices for MLOps, from model training to deployment and monitoring in production.",
    content: "Detailed content about MLOps pipelines...",
    category: "machine-learning",
    subcategory: "MLOps",
    tags: ["MLOps", "Pipelines", "Production", "Monitoring"],
    date: "2024-02-15",
    readTime: "18 min read",
    difficulty: "Intermediate",
    slug: "mlops-pipeline",
    featured: false,
    author: "Hiep Tran",
    image: "/blog-placeholder.jpg",
  },

  // Math & Paper Reading articles
  {
    id: "attention-is-all-you-need",
    title: "Paper Review: Attention Is All You Need",
    excerpt:
      "In-depth analysis of the groundbreaking paper that introduced the Transformer architecture.",
    content: "Detailed paper review...",
    category: "paper-reading",
    subcategory: "NLP",
    tags: ["Paper Review", "Transformers", "Attention", "Research"],
    date: "2024-03-05",
    readTime: "30 min read",
    difficulty: "Advanced",
    slug: "attention-is-all-you-need",
    featured: true,
    author: "Hiep Tran",
    image: "/blog-placeholder.jpg",
  },
  {
    id: "linear-algebra-ml",
    title: "Linear Algebra for Machine Learning",
    excerpt:
      "Essential linear algebra concepts every ML practitioner should understand.",
    content: "Detailed content about linear algebra...",
    category: "math-equations",
    subcategory: "Linear Algebra",
    tags: ["Linear Algebra", "Mathematics", "Machine Learning"],
    date: "2024-02-10",
    readTime: "16 min read",
    difficulty: "Intermediate",
    slug: "linear-algebra-ml",
    featured: false,
    author: "Hiep Tran",
    image: "/blog-placeholder.jpg",
  },

  // Notes articles
  {
    id: "system-design-notes",
    title: "System Design Interview Notes",
    excerpt:
      "Comprehensive notes on system design patterns, scalability, and architecture principles.",
    content: "Detailed system design notes...",
    category: "notes",
    subcategory: "Interview Prep",
    tags: ["System Design", "Interview", "Scalability", "Architecture"],
    date: "2024-02-28",
    readTime: "14 min read",
    difficulty: "Intermediate",
    slug: "system-design-notes",
    featured: false,
    author: "Hiep Tran",
    image: "/blog-placeholder.jpg",
  },
];

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
