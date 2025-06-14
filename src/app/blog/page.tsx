"use client";

import React, { useState, useEffect } from "react";
import FadeInWrapper from "@/components/FadeInWrapper";
import ArticleCard from "@/components/ArticleCard";
import ArticleGrid from "@/components/ArticleGrid";

interface BlogPostMetadata {
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

interface Article {
  title: string;
  summary: string;
  image: string;
  date: string;
  link: string;
  category: string;
  readTime: string;
}

// Convert BlogPostMetadata to Article format
function convertToArticle(post: BlogPostMetadata): Article {
  return {
    title: post.title,
    summary: post.excerpt,
    image: post.image,
    date: post.publishDate,
    link: `/blog/${post.slug}`, // This now includes the full category/post-name format
    category: post.category,
    readTime: post.readTime,
  };
}

// Convert local Article to lib Article format
function convertToLibArticle(
  article: Article,
  originalSlug?: string
): import("@/lib/blog").Article {
  return {
    id: originalSlug || article.link.split("/").pop() || "unknown",
    title: article.title,
    excerpt: article.summary,
    content: "",
    image: article.image || "/placeholder-image.jpg",
    category: article.category,
    subcategory: article.category,
    date: article.date,
    readTime: article.readTime,
    tags: [],
    difficulty: "Intermediate" as const,
    slug: originalSlug || article.link.split("/").pop() || "unknown",
    featured: false,
  };
}

export default function BlogPage() {
  const [articles, setArticles] = useState<Article[]>([]);
  const [articleSlugs, setArticleSlugs] = useState<Map<string, string>>(
    new Map()
  );
  const [isLoading, setIsLoading] = useState(true);
  const [displayedCount, setDisplayedCount] = useState(9);
  const [loadingMore, setLoadingMore] = useState(false);

  // Items per page for button loading
  const ITEMS_PER_PAGE = 9;

  useEffect(() => {
    async function fetchPosts() {
      try {
        const response = await fetch("/api/blog/posts");
        const blogPosts: BlogPostMetadata[] = await response.json();
        const convertedArticles = blogPosts.map(convertToArticle);

        // Create a map of article links to original slugs
        const slugMap = new Map<string, string>();
        blogPosts.forEach((post) => {
          const articleLink = `/blog/${post.slug}`;
          slugMap.set(articleLink, post.slug);
        });

        setArticles(convertedArticles);
        setArticleSlugs(slugMap);
      } catch (error) {
        console.error("Error loading blog posts:", error);
        // Fallback to empty array
        setArticles([]);
        setArticleSlugs(new Map());
      } finally {
        setIsLoading(false);
      }
    }

    fetchPosts();
  }, []);

  // Get currently displayed articles
  const displayedArticles = articles.slice(0, displayedCount);

  // Check if there are more articles to load
  const hasMoreArticles = displayedCount < articles.length;

  // Function to load more articles
  const loadMoreArticles = async () => {
    if (loadingMore || !hasMoreArticles) return;

    setLoadingMore(true);
    // Simulate network delay for better UX
    await new Promise((resolve) => setTimeout(resolve, 500));
    setDisplayedCount((prev) => prev + ITEMS_PER_PAGE);
    setLoadingMore(false);
  };

  // Improved category count calculation using exact category matching
  const getCategoryCount = (categoryName: string) => {
    const normalizedName = categoryName.toLowerCase().replace(/\s+/g, "-");
    return articles.filter((article) => {
      const articleCategory = article.category
        .toLowerCase()
        .replace(/\s+/g, "-");
      return (
        articleCategory === normalizedName ||
        articleCategory.includes(normalizedName) ||
        article.link.includes(`/blog/${normalizedName}/`)
      );
    }).length;
  };

  // Update categories to reflect actual blog posts with dynamic counts
  const categories = [
    {
      name: "Software Development",
      description: "Programming best practices and tutorials",
      link: "/blog/software-development",
      icon: "💻",
      count: getCategoryCount("software-development"),
      color: "from-green-500 to-teal-600",
    },
    {
      name: "Machine Learning",
      description: "AI, ML algorithms, and deep learning insights",
      link: "/blog/machine-learning",
      icon: "🤖",
      count: getCategoryCount("machine-learning"),
      color: "from-purple-500 to-pink-600",
    },
    {
      name: "Cryptocurrency",
      description: "Blockchain, DeFi, and crypto technology insights",
      link: "/blog/crypto",
      icon: "₿",
      count: getCategoryCount("crypto"),
      color: "from-orange-500 to-yellow-500",
    },
    {
      name: "Paper Reading",
      description: "Research paper reviews and analysis",
      link: "/blog/paper-reading",
      icon: "📚",
      count: getCategoryCount("paper-reading"),
      color: "from-indigo-500 to-blue-600",
    },
    {
      name: "Notes",
      description: "Quick thoughts and learning notes",
      link: "/blog/notes",
      icon: "📝",
      count: getCategoryCount("notes"),
      color: "from-cyan-500 to-blue-600",
    },
  ];

  if (isLoading) {
    return (
      <FadeInWrapper duration={800}>
        <div className="flex flex-col min-h-screen items-center justify-center">
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-500"></div>
          <p className="mt-4 text-lg">Loading blog posts...</p>
        </div>
      </FadeInWrapper>
    );
  }

  if (articles.length === 0) {
    return (
      <FadeInWrapper duration={800}>
        <div className="flex flex-col min-h-screen items-center justify-center">
          <h1 className="text-4xl font-bold mb-4">No blog posts found</h1>
          <p className="text-lg text-gray-600">
            Check back later for new content!
          </p>
        </div>
      </FadeInWrapper>
    );
  }
  return (
    <FadeInWrapper duration={800}>
      <div
        className="flex flex-col min-h-screen transition-colors duration-300"
        style={{
          backgroundColor: "var(--background)",
          color: "var(--text-primary)",
        }}
      >
        <main className="flex-1">
          <div className="max-w-6xl mx-auto px-6 py-16">
            {/* Header with Enhanced Animations */}
            <FadeInWrapper duration={600} delay={100}>
              <div className="mb-16 text-center">
                <h1 className="text-4xl md:text-6xl font-bold mb-6 blog-title-animated">
                  Blog
                </h1>
                <p
                  className="text-xl max-w-3xl mx-auto blog-subtitle-animated"
                  style={{
                    color: "var(--text-secondary)",
                  }}
                >
                  Thoughts on technology, software development, AI, and
                  everything in between. Here I share insights, tutorials, and
                  lessons learned from my journey in tech.
                </p>
              </div>
            </FadeInWrapper>

            {/* Categories Section */}
            <FadeInWrapper duration={600} delay={200}>
              <div className="mb-16">
                <h2 className="text-3xl font-bold mb-8 gradient-text-green blog-content-stagger">
                  Browse by Category
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {categories.map((category, index) => (
                    <FadeInWrapper
                      key={category.name}
                      duration={400}
                      delay={300 + Math.random() * 200}
                    >
                      <a
                        href={category.link}
                        className="group relative overflow-hidden rounded-xl p-6 border transition-all duration-300 hover:scale-105 hover:shadow-lg blog-content-stagger flex flex-col justify-between h-full"
                        style={{
                          backgroundColor: "var(--card-bg)",
                          borderColor: "var(--card-border)",
                          animationDelay: `${0.3 + index * 0.1}s`,
                        }}
                      >
                        <div
                          className={`absolute inset-0 bg-gradient-to-br ${category.color} opacity-0 group-hover:opacity-10 transition-opacity duration-300`}
                        ></div>
                        <div className="relative z-10">
                          <div className="flex items-center justify-between mb-3">
                            <span className="text-2xl transform group-hover:scale-110 transition-transform duration-300">
                              {category.icon}
                            </span>
                            <span
                              className="text-sm font-medium px-3 py-1 rounded-full"
                              style={{
                                backgroundColor: "var(--accent-subtle)",
                                color: "var(--accent)",
                              }}
                            >
                              {category.count} posts
                            </span>
                          </div>
                          <h3
                            className="text-lg font-semibold mb-2 group-hover:text-sky-600 dark:group-hover:text-sky-400 transition-colors duration-300"
                            style={{ color: "var(--text-primary)" }}
                          >
                            {category.name}
                          </h3>
                          <p
                            className="text-sm leading-relaxed"
                            style={{ color: "var(--text-secondary)" }}
                          >
                            {category.description}
                          </p>
                        </div>
                      </a>
                    </FadeInWrapper>
                  ))}
                </div>
              </div>
            </FadeInWrapper>

            {/* All Articles with Lazy Loading */}
            <FadeInWrapper duration={600} delay={400}>
              <div>
                <h2 className="text-3xl font-bold mb-8 gradient-text-orange blog-content-stagger">
                  All Articles
                </h2>

                {displayedArticles.length > 0 ? (
                  <>
                    <ArticleGrid>
                      {displayedArticles.map((article, index) => (
                        <FadeInWrapper
                          key={`${article.link}-${index}`}
                          duration={400}
                          delay={500 + index * 50}
                        >
                          <ArticleCard
                            article={convertToLibArticle(
                              article,
                              articleSlugs.get(article.link)
                            )}
                            index={index}
                            variant="default"
                          />
                        </FadeInWrapper>
                      ))}
                    </ArticleGrid>

                    {hasMoreArticles && (
                      <div className="flex justify-center mt-12">
                        <button
                          onClick={loadMoreArticles}
                          disabled={loadingMore}
                          className="px-8 py-4 rounded-lg font-medium transition-all duration-300 hover:scale-105 hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 border"
                          style={{
                            backgroundColor: loadingMore
                              ? "var(--surface)"
                              : "var(--accent)",
                            color: loadingMore
                              ? "var(--text-secondary)"
                              : "white",
                            borderColor: loadingMore
                              ? "var(--border)"
                              : "var(--accent)",
                          }}
                        >
                          {loadingMore ? (
                            <div className="flex items-center gap-3">
                              <div
                                className="inline-block animate-spin rounded-full h-5 w-5 border-b-2"
                                style={{ borderColor: "var(--accent)" }}
                              ></div>
                              Loading more articles...
                            </div>
                          ) : (
                            `Load More Articles (${Math.min(
                              ITEMS_PER_PAGE,
                              articles.length - displayedCount
                            )} more)`
                          )}
                        </button>
                      </div>
                    )}
                  </>
                ) : (
                  <div className="text-center py-12">
                    <p
                      className="text-lg"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      No articles found.
                    </p>
                  </div>
                )}
              </div>
            </FadeInWrapper>

            {/* Newsletter CTA */}
            <FadeInWrapper duration={600} delay={500}>
              <div
                className="mt-16 p-8 rounded-xl border blog-content-stagger relative overflow-hidden"
                style={{
                  backgroundColor: "var(--card-bg)",
                  borderColor: "var(--card-border)",
                  animationDelay: "0.6s",
                }}
              >
                <div className="absolute inset-0 bg-gradient-to-r from-blue-500/5 via-purple-500/5 to-cyan-500/5"></div>
                <div className="relative z-10 text-center">
                  <h3 className="text-2xl font-bold gradient-text-cyan mb-4">
                    Stay Updated
                  </h3>
                  <p
                    className="mb-6 max-w-2xl mx-auto"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    Get notified when I publish new articles about technology,
                    development, and AI. No spam, just quality content.
                  </p>
                  <div className="flex flex-col sm:flex-row gap-4 max-w-md mx-auto">
                    <input
                      type="email"
                      placeholder="Enter your email"
                      className="flex-1 px-4 py-3 rounded-lg border focus:outline-none focus:ring-2 focus:ring-sky-500 focus:border-transparent transition-all duration-300"
                      style={{
                        backgroundColor: "var(--surface)",
                        borderColor: "var(--border)",
                        color: "var(--text-primary)",
                      }}
                    />
                    <button
                      className="px-6 py-3 font-medium rounded-lg transition-all duration-300 hover:scale-105 hover:shadow-lg relative overflow-hidden group"
                      style={{
                        backgroundColor: "var(--accent)",
                        color: "white",
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.backgroundColor =
                          "var(--accent-hover)";
                        e.currentTarget.style.transform = "scale(1.05)";
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.backgroundColor = "var(--accent)";
                        e.currentTarget.style.transform = "scale(1)";
                      }}
                    >
                      <span className="relative z-10">Subscribe</span>
                      <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-500"></div>
                    </button>
                  </div>
                </div>
              </div>
            </FadeInWrapper>
          </div>
        </main>
      </div>
    </FadeInWrapper>
  );
}
