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
  // Machine Learning articles
  {
    id: "deep-reinforcement-learning",
    title: "Deep Reinforcement Learning: From Q-Learning to Policy Gradients",
    excerpt:
      "Comprehensive guide to deep reinforcement learning algorithms and their applications.",
    content: "Detailed content about deep reinforcement learning...",
    category: "machine-learning",
    subcategory: "Reinforcement Learning",
    tags: [
      "Deep Learning",
      "Reinforcement Learning",
      "Q-Learning",
      "Policy Gradients",
    ],
    date: "2024-03-15",
    readTime: "25 min read",
    difficulty: "Advanced",
    slug: "deep-reinforcement-learning",
    featured: true,
    author: "Hiep Tran",
    image: "/blog-placeholder.jpg",
  },
  {
    id: "ensemble-methods-traditional-ml",
    title:
      "Ensemble Methods in Traditional Machine Learning: Bagging, Boosting, and Stacking",
    excerpt:
      "Explore ensemble methods that combine multiple models for better performance.",
    content: "Detailed content about ensemble methods...",
    category: "machine-learning",
    subcategory: "Traditional ML",
    tags: ["Ensemble Learning", "Bagging", "Boosting", "Random Forest"],
    date: "2024-03-12",
    readTime: "18 min read",
    difficulty: "Intermediate",
    slug: "ensemble-methods-traditional-ml",
    featured: false,
    author: "Hiep Tran",
    image: "/blog-placeholder.jpg",
  },
  {
    id: "lora-fine-tuning",
    title: "Fine-tuning Large Language Models with LoRA",
    excerpt:
      "Learn about Low-Rank Adaptation (LoRA) for efficient fine-tuning of large language models.",
    content: "Detailed content about LoRA fine-tuning...",
    category: "machine-learning",
    subcategory: "LLM",
    tags: ["LLM", "Fine-tuning", "LoRA", "Transformers"],
    date: "2024-03-10",
    readTime: "20 min read",
    difficulty: "Advanced",
    slug: "lora-fine-tuning",
    featured: true,
    author: "Hiep Tran",
    image: "/blog-placeholder.jpg",
  },
  {
    id: "mlops-best-practices",
    title: "MLOps Best Practices: From Model Development to Production",
    excerpt:
      "Best practices for MLOps, covering the entire ML lifecycle from development to production.",
    content: "Detailed content about MLOps best practices...",
    category: "machine-learning",
    subcategory: "MLOps",
    tags: ["MLOps", "Production", "Model Deployment", "Monitoring"],
    date: "2024-03-08",
    readTime: "22 min read",
    difficulty: "Intermediate",
    slug: "mlops-best-practices",
    featured: false,
    author: "Hiep Tran",
    image: "/blog-placeholder.jpg",
  },
  {
    id: "neural-architecture-search",
    title: "Neural Architecture Search: Automated Design of Deep Networks",
    excerpt:
      "Explore automated methods for designing neural network architectures.",
    content: "Detailed content about neural architecture search...",
    category: "machine-learning",
    subcategory: "Neural Architecture",
    tags: ["NAS", "AutoML", "Neural Networks", "Architecture"],
    date: "2024-03-05",
    readTime: "24 min read",
    difficulty: "Advanced",
    slug: "neural-architecture-search",
    featured: false,
    author: "Hiep Tran",
    image: "/blog-placeholder.jpg",
  },
  {
    id: "neural-networks-fundamentals",
    title: "Neural Networks Fundamentals: From Perceptrons to Deep Learning",
    excerpt:
      "Understanding the fundamentals of neural networks and deep learning.",
    content: "Detailed content about neural networks fundamentals...",
    category: "machine-learning",
    subcategory: "Deep Learning",
    tags: [
      "Neural Networks",
      "Deep Learning",
      "Perceptrons",
      "Backpropagation",
    ],
    date: "2024-03-02",
    readTime: "16 min read",
    difficulty: "Beginner",
    slug: "neural-networks-fundamentals",
    featured: true,
    author: "Hiep Tran",
    image: "/blog-placeholder.jpg",
  },
  {
    id: "optimization-algorithms-ml",
    title:
      "Optimization Algorithms for Machine Learning: From SGD to Adam and Beyond",
    excerpt:
      "Comprehensive guide to optimization algorithms used in machine learning.",
    content: "Detailed content about optimization algorithms...",
    category: "machine-learning",
    subcategory: "Optimization",
    tags: ["Optimization", "SGD", "Adam", "Gradient Descent"],
    date: "2024-02-28",
    readTime: "19 min read",
    difficulty: "Intermediate",
    slug: "optimization-algorithms-ml",
    featured: false,
    author: "Hiep Tran",
    image: "/blog-placeholder.jpg",
  },
  {
    id: "support-vector-machines",
    title: "Support Vector Machines: Theory, Implementation, and Applications",
    excerpt:
      "Deep dive into Support Vector Machines, their theory and practical applications.",
    content: "Detailed content about support vector machines...",
    category: "machine-learning",
    subcategory: "Traditional ML",
    tags: ["SVM", "Machine Learning", "Classification", "Kernel Methods"],
    date: "2024-02-25",
    readTime: "21 min read",
    difficulty: "Intermediate",
    slug: "support-vector-machines",
    featured: false,
    author: "Hiep Tran",
    image: "/blog-placeholder.jpg",
  },

  // Notes articles
  {
    id: "atomic-habits-book-summary",
    title: "Atomic Habits by James Clear: Building Better Systems",
    excerpt:
      "Key insights and takeaways from James Clear's book on building good habits.",
    content: "Detailed book summary of Atomic Habits...",
    category: "notes",
    subcategory: "Book Summaries",
    tags: ["Book Summary", "Habits", "Personal Development", "Productivity"],
    date: "2024-02-20",
    readTime: "12 min read",
    difficulty: "Beginner",
    slug: "atomic-habits-book-summary",
    featured: false,
    author: "Hiep Tran",
    image: "/blog-placeholder.jpg",
  },
  {
    id: "random-ideas-collection",
    title: "Random Ideas & Quick Thoughts Collection",
    excerpt:
      "A collection of random ideas, thoughts, and observations worth sharing.",
    content: "Collection of random ideas and thoughts...",
    category: "notes",
    subcategory: "Idea Dump",
    tags: ["Ideas", "Thoughts", "Brainstorming", "Collection"],
    date: "2024-02-15",
    readTime: "8 min read",
    difficulty: "Beginner",
    slug: "random-ideas-collection",
    featured: false,
    author: "Hiep Tran",
    image: "/blog-placeholder.jpg",
  },
  {
    id: "senior-developer-reflection",
    title: "Reflecting on My First Year as a Senior Developer",
    excerpt:
      "Personal reflections on growth, challenges, and lessons learned as a senior developer.",
    content: "Personal reflection on senior developer experience...",
    category: "notes",
    subcategory: "Self-reflection Entries",
    tags: ["Career", "Reflection", "Senior Developer", "Growth"],
    date: "2024-02-10",
    readTime: "14 min read",
    difficulty: "Beginner",
    slug: "senior-developer-reflection",
    featured: true,
    author: "Hiep Tran",
    image: "/blog-placeholder.jpg",
  },

  // Paper Reading articles
  {
    id: "autogpt-autonomous-agents",
    title: "AutoGPT and Autonomous AI Agents: Architecture and Implications",
    excerpt:
      "Analysis of AutoGPT and the emerging field of autonomous AI agents.",
    content: "Detailed analysis of AutoGPT and autonomous agents...",
    category: "paper-reading",
    subcategory: "AI Agent",
    tags: ["AutoGPT", "AI Agents", "Autonomy", "LLM"],
    date: "2024-03-18",
    readTime: "26 min read",
    difficulty: "Advanced",
    slug: "autogpt-autonomous-agents",
    featured: true,
    author: "Hiep Tran",
    image: "/blog-placeholder.jpg",
  },
  {
    id: "blip-vision-language",
    title:
      "BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding",
    excerpt:
      "Analysis of BLIP model for unified vision-language understanding.",
    content: "Detailed analysis of BLIP paper...",
    category: "paper-reading",
    subcategory: "Multimodal",
    tags: ["BLIP", "Vision-Language", "Multimodal", "Pre-training"],
    date: "2024-03-16",
    readTime: "28 min read",
    difficulty: "Advanced",
    slug: "blip-vision-language",
    featured: false,
    author: "Hiep Tran",
    image: "/blog-placeholder.jpg",
  },
  {
    id: "chain-of-thought-prompting",
    title: "Chain of Thought Prompting in Large Language Models",
    excerpt:
      "Analysis of chain-of-thought prompting techniques for improving LLM reasoning.",
    content: "Detailed analysis of chain-of-thought prompting...",
    category: "paper-reading",
    subcategory: "LLM",
    tags: ["Chain of Thought", "LLM", "Prompting", "Reasoning"],
    date: "2024-03-14",
    readTime: "22 min read",
    difficulty: "Intermediate",
    slug: "chain-of-thought-prompting",
    featured: false,
    author: "Hiep Tran",
    image: "/blog-placeholder.jpg",
  },
  {
    id: "clip-multimodal-learning",
    title: "CLIP: Connecting Text and Images with Contrastive Learning",
    excerpt:
      "Analysis of CLIP's approach to learning visual concepts from natural language supervision.",
    content: "Detailed analysis of CLIP paper...",
    category: "paper-reading",
    subcategory: "Multimodal",
    tags: ["CLIP", "Contrastive Learning", "Vision-Language", "Multimodal"],
    date: "2024-03-12",
    readTime: "24 min read",
    difficulty: "Advanced",
    slug: "clip-multimodal-learning",
    featured: true,
    author: "Hiep Tran",
    image: "/blog-placeholder.jpg",
  },
  {
    id: "efficientnet-model-scaling",
    title:
      "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks",
    excerpt:
      "Analysis of EfficientNet's compound scaling method for CNN architectures.",
    content: "Detailed analysis of EfficientNet paper...",
    category: "paper-reading",
    subcategory: "Computer Vision",
    tags: ["EfficientNet", "Model Scaling", "CNN", "Computer Vision"],
    date: "2024-03-09",
    readTime: "20 min read",
    difficulty: "Intermediate",
    slug: "efficientnet-model-scaling",
    featured: false,
    author: "Hiep Tran",
    image: "/blog-placeholder.jpg",
  },
  {
    id: "lime-interpretability",
    title: "LIME: Local Interpretable Model-Agnostic Explanations",
    excerpt:
      "Analysis of LIME for explaining individual predictions of machine learning models.",
    content: "Detailed analysis of LIME paper...",
    category: "paper-reading",
    subcategory: "AI Interpretability",
    tags: ["LIME", "Interpretability", "Explainable AI", "Model Explanations"],
    date: "2024-03-06",
    readTime: "18 min read",
    difficulty: "Intermediate",
    slug: "lime-interpretability",
    featured: false,
    author: "Hiep Tran",
    image: "/blog-placeholder.jpg",
  },
  {
    id: "llm-scaling-laws",
    title:
      "Scaling Laws for Language Models: Understanding the Power-Law Relationship",
    excerpt:
      "Analysis of scaling laws that govern the performance of large language models.",
    content: "Detailed analysis of LLM scaling laws...",
    category: "paper-reading",
    subcategory: "LLM",
    tags: ["Scaling Laws", "LLM", "Model Performance", "Training"],
    date: "2024-03-03",
    readTime: "25 min read",
    difficulty: "Advanced",
    slug: "llm-scaling-laws",
    featured: true,
    author: "Hiep Tran",
    image: "/blog-placeholder.jpg",
  },
  {
    id: "random-forest-ensemble-learning",
    title: "Random Forest: From Decision Trees to Ensemble Excellence",
    excerpt:
      "Analysis of Random Forest algorithm and ensemble learning principles.",
    content: "Detailed analysis of Random Forest...",
    category: "paper-reading",
    subcategory: "Machine Learning",
    tags: [
      "Random Forest",
      "Ensemble Learning",
      "Decision Trees",
      "Machine Learning",
    ],
    date: "2024-02-28",
    readTime: "17 min read",
    difficulty: "Intermediate",
    slug: "random-forest-ensemble-learning",
    featured: false,
    author: "Hiep Tran",
    image: "/blog-placeholder.jpg",
  },
  {
    id: "react-reasoning-acting",
    title: "ReAct: Synergizing Reasoning and Acting in Language Models",
    excerpt:
      "Analysis of ReAct paradigm for combining reasoning and acting in language models.",
    content: "Detailed analysis of ReAct paper...",
    category: "paper-reading",
    subcategory: "AI Agent",
    tags: ["ReAct", "Reasoning", "Acting", "Language Models"],
    date: "2024-02-24",
    readTime: "23 min read",
    difficulty: "Advanced",
    slug: "react-reasoning-acting",
    featured: false,
    author: "Hiep Tran",
    image: "/blog-placeholder.jpg",
  },
  {
    id: "vision-transformer-paper-analysis",
    title: "An Image is Worth 16x16 Words: Vision Transformer Analysis",
    excerpt:
      "Analysis of Vision Transformer (ViT) and its impact on computer vision.",
    content: "Detailed analysis of Vision Transformer paper...",
    category: "paper-reading",
    subcategory: "Computer Vision",
    tags: ["Vision Transformer", "ViT", "Computer Vision", "Transformers"],
    date: "2024-02-20",
    readTime: "27 min read",
    difficulty: "Advanced",
    slug: "vision-transformer-paper-analysis",
    featured: true,
    author: "Hiep Tran",
    image: "/blog-placeholder.jpg",
  },
  {
    id: "wav2vec2-speech-representation",
    title:
      "Wav2Vec 2.0: A Framework for Self-Supervised Learning of Speech Representations",
    excerpt:
      "Analysis of Wav2Vec 2.0 for self-supervised speech representation learning.",
    content: "Detailed analysis of Wav2Vec 2.0 paper...",
    category: "paper-reading",
    subcategory: "Speech Processing",
    tags: ["Wav2Vec", "Speech Processing", "Self-Supervised Learning", "Audio"],
    date: "2024-02-16",
    readTime: "21 min read",
    difficulty: "Advanced",
    slug: "wav2vec2-speech-representation",
    featured: false,
    author: "Hiep Tran",
    image: "/blog-placeholder.jpg",
  },
  {
    id: "whisper-robust-speech-recognition",
    title:
      "Whisper: Robust Speech Recognition via Large-Scale Weak Supervision",
    excerpt:
      "Analysis of OpenAI's Whisper model for robust speech recognition.",
    content: "Detailed analysis of Whisper paper...",
    category: "paper-reading",
    subcategory: "Speech Processing",
    tags: ["Whisper", "Speech Recognition", "OpenAI", "Weak Supervision"],
    date: "2024-02-12",
    readTime: "19 min read",
    difficulty: "Intermediate",
    slug: "whisper-robust-speech-recognition",
    featured: false,
    author: "Hiep Tran",
    image: "/blog-placeholder.jpg",
  },

  // Software Development articles
  {
    id: "algorithms-data-structures",
    title: "Algorithms and Data Structures: Foundations for Efficient Software",
    excerpt:
      "Comprehensive guide to essential algorithms and data structures for software development.",
    content: "Detailed content about algorithms and data structures...",
    category: "software-development",
    subcategory: "Algorithms",
    tags: ["Algorithms", "Data Structures", "Programming", "Computer Science"],
    date: "2024-03-20",
    readTime: "30 min read",
    difficulty: "Intermediate",
    slug: "algorithms-data-structures",
    featured: true,
    author: "Hiep Tran",
    image: "/blog-placeholder.jpg",
  },
  {
    id: "data-engineering-fundamentals",
    title: "Data Engineering Fundamentals: Building Robust Data Pipelines",
    excerpt:
      "Essential concepts and practices for building scalable data engineering pipelines.",
    content: "Detailed content about data engineering fundamentals...",
    category: "software-development",
    subcategory: "Data Engineering",
    tags: ["Data Engineering", "Pipelines", "ETL", "Big Data"],
    date: "2024-03-17",
    readTime: "26 min read",
    difficulty: "Intermediate",
    slug: "data-engineering-fundamentals",
    featured: false,
    author: "Hiep Tran",
    image: "/blog-placeholder.jpg",
  },
  {
    id: "database-design-optimization",
    title: "Database Design and Optimization: From Schema to Scale",
    excerpt:
      "Best practices for database design, optimization, and scaling strategies.",
    content: "Detailed content about database design and optimization...",
    category: "software-development",
    subcategory: "Database",
    tags: ["Database", "SQL", "Optimization", "Schema Design"],
    date: "2024-03-13",
    readTime: "24 min read",
    difficulty: "Intermediate",
    slug: "database-design-optimization",
    featured: false,
    author: "Hiep Tran",
    image: "/blog-placeholder.jpg",
  },
  {
    id: "microservices-distributed-systems",
    title:
      "Microservices and Distributed Systems: Building Scalable Architecture",
    excerpt:
      "Guide to designing and implementing microservices and distributed system architectures.",
    content: "Detailed content about microservices and distributed systems...",
    category: "software-development",
    subcategory: "Distributed Systems",
    tags: [
      "Microservices",
      "Distributed Systems",
      "Architecture",
      "Scalability",
    ],
    date: "2024-03-11",
    readTime: "28 min read",
    difficulty: "Advanced",
    slug: "microservices-distributed-systems",
    featured: true,
    author: "Hiep Tran",
    image: "/blog-placeholder.jpg",
  },
  {
    id: "site-reliability-engineering",
    title: "Site Reliability Engineering: Building Resilient Systems",
    excerpt:
      "Principles and practices of SRE for building and maintaining reliable systems.",
    content: "Detailed content about site reliability engineering...",
    category: "software-development",
    subcategory: "Site Reliability Engineering",
    tags: ["SRE", "Reliability", "DevOps", "Monitoring"],
    date: "2024-03-07",
    readTime: "22 min read",
    difficulty: "Intermediate",
    slug: "site-reliability-engineering",
    featured: false,
    author: "Hiep Tran",
    image: "/blog-placeholder.jpg",
  },
  {
    id: "software-development-best-practices",
    title: "Understanding Software Development Best Practices",
    excerpt:
      "Essential best practices for writing clean, maintainable, and scalable code.",
    content: "Detailed content about software development best practices...",
    category: "software-development",
    subcategory: "Coding Practices",
    tags: ["Best Practices", "Clean Code", "Code Quality", "Development"],
    date: "2024-03-04",
    readTime: "20 min read",
    difficulty: "Beginner",
    slug: "software-development-best-practices",
    featured: false,
    author: "Hiep Tran",
    image: "/blog-placeholder.jpg",
  },
  {
    id: "system-design-fundamentals",
    title: "System Design Fundamentals: Building Scalable Applications",
    excerpt:
      "Core principles and patterns for designing scalable and maintainable systems.",
    content: "Detailed content about system design fundamentals...",
    category: "software-development",
    subcategory: "System Design",
    tags: ["System Design", "Architecture", "Scalability", "Design Patterns"],
    date: "2024-03-01",
    readTime: "25 min read",
    difficulty: "Intermediate",
    slug: "system-design-fundamentals",
    featured: true,
    author: "Hiep Tran",
    image: "/blog-placeholder.jpg",
  },
  {
    id: "typescript-best-practices",
    title: "TypeScript Best Practices for Large Applications",
    excerpt:
      "Essential patterns and practices for building scalable TypeScript applications with proper type safety.",
    content: "Detailed content about TypeScript best practices...",
    category: "software-development",
    subcategory: "Coding Practices",
    tags: ["TypeScript", "Best Practices", "Type Safety", "Scalability"],
    date: "2024-02-26",
    readTime: "18 min read",
    difficulty: "Intermediate",
    slug: "typescript-best-practices",
    featured: false,
    author: "Hiep Tran",
    image: "/blog-placeholder.jpg",
  },

  // Crypto articles (keeping a few as placeholders)
  {
    id: "defi-fundamentals",
    title: "DeFi Fundamentals: Understanding Decentralized Finance",
    excerpt:
      "A comprehensive guide to DeFi protocols, yield farming, and the future of decentralized financial services.",
    content: "Detailed content about DeFi fundamentals...",
    category: "crypto",
    subcategory: "DeFi",
    tags: ["DeFi", "Yield Farming", "Smart Contracts"],
    date: "2024-02-22",
    readTime: "12 min read",
    difficulty: "Intermediate",
    slug: "defi-fundamentals",
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
