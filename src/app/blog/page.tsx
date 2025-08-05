"use client";

import React, { useState, useEffect } from "react";
import Image from "next/image";
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
  // Get featured article (first one) and other recent articles for the latest section
  const featuredArticle = displayedArticles[0];
  const recentArticles = displayedArticles.slice(1, 5); // Next 4 articles

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
            {/* Header */}
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

            {/* Latest Articles Section */}
            {featuredArticle && (
              <FadeInWrapper duration={600} delay={200}>
                <div className="mb-16">
                  <h2 className="text-3xl font-bold mb-8 gradient-text-green">
                    Latest Articles
                  </h2>

                  {/* Featured Article - Two Column Layout */}
                  <div
                    className="mb-12 rounded-2xl border overflow-hidden"
                    style={{
                      backgroundColor: "var(--card-bg)",
                      borderColor: "var(--card-border)",
                    }}
                  >
                    <div className="grid md:grid-cols-5 gap-0">
                      {/* Left: Featured Image (60%) */}
                      <div className="md:col-span-3 relative h-80 md:h-96">
                        <Image
                          src={featuredArticle.image || "/blog-placeholder.jpg"}
                          alt={featuredArticle.title}
                          fill
                          className="object-cover"
                          sizes="(max-width: 768px) 100vw, 60vw"
                        />
                        <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent"></div>
                      </div>

                      {/* Right: Article Info (40%) */}
                      <div className="md:col-span-2 p-8 flex flex-col justify-center">
                        <div className="mb-4">
                          <span
                            className="text-sm font-bold uppercase tracking-wider"
                            style={{ color: "var(--accent)" }}
                          >
                            {featuredArticle.category}
                          </span>
                        </div>
                        <h3
                          className="text-2xl md:text-3xl font-bold mb-4 leading-tight"
                          style={{ color: "var(--text-primary)" }}
                        >
                          <a
                            href={featuredArticle.link}
                            className="hover:text-[var(--accent)] transition-colors duration-300"
                          >
                            {featuredArticle.title}
                          </a>
                        </h3>
                        <div
                          className="text-sm flex items-center gap-4 mb-4"
                          style={{ color: "var(--text-secondary)" }}
                        >
                          <span>
                            {new Date(
                              featuredArticle.date
                            ).toLocaleDateString()}
                          </span>
                          <span>•</span>
                          <span>{featuredArticle.readTime}</span>
                        </div>
                        <p
                          className="text-base leading-relaxed mb-6"
                          style={{ color: "var(--text-secondary)" }}
                        >
                          {featuredArticle.summary}
                        </p>
                        <a
                          href={featuredArticle.link}
                          className="inline-flex items-center gap-2 text-sm font-medium hover:gap-3 transition-all duration-300"
                          style={{ color: "var(--accent)" }}
                        >
                          Read Article
                          <svg
                            className="w-4 h-4"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M9 5l7 7-7 7"
                            />
                          </svg>
                        </a>
                      </div>
                    </div>
                  </div>

                  {/* Four Column Grid of Recent Articles */}
                  {recentArticles.length > 0 && (
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                      {recentArticles.map((article) => (
                        <div
                          key={article.link}
                          className="group rounded-lg border overflow-hidden transition-all duration-300 hover:shadow-lg hover:scale-[1.02]"
                          style={{
                            backgroundColor: "var(--card-bg)",
                            borderColor: "var(--card-border)",
                          }}
                        >
                          <div className="relative h-32 overflow-hidden">
                            <Image
                              src={article.image || "/blog-placeholder.jpg"}
                              alt={article.title}
                              fill
                              className="object-cover transition-transform duration-300 group-hover:scale-105"
                              sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 25vw"
                            />
                          </div>
                          <div className="p-4">
                            <div className="mb-2">
                              <span
                                className="text-xs font-medium uppercase tracking-wider"
                                style={{ color: "var(--accent)" }}
                              >
                                {article.category}
                              </span>
                            </div>
                            <h4
                              className="text-sm font-semibold mb-2 leading-tight line-clamp-2 group-hover:text-[var(--accent)] transition-colors duration-300"
                              style={{ color: "var(--text-primary)" }}
                            >
                              <a href={article.link}>{article.title}</a>
                            </h4>
                            <div
                              className="text-xs flex items-center gap-2"
                              style={{ color: "var(--text-secondary)" }}
                            >
                              <span>
                                {new Date(article.date).toLocaleDateString(
                                  "en-US",
                                  { month: "short", day: "numeric" }
                                )}
                              </span>
                              <span>•</span>
                              <span>{article.readTime}</span>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </FadeInWrapper>
            )}

            {/* All Articles Section */}
            <FadeInWrapper duration={600} delay={400}>
              <div>
                {/* Section Title */}
                <div className="text-center mb-12">
                  <h2 className="text-4xl font-bold gradient-text-orange mb-4">
                    All Articles
                  </h2>
                  <p
                    className="text-lg max-w-2xl mx-auto"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    Explore all our articles across different categories and
                    topics
                  </p>
                </div>

                {/* Category Filter */}
                <div className="flex flex-wrap justify-center gap-3 mb-12">
                  {categories.map((category) => (
                    <a
                      key={category.name}
                      href={category.link}
                      className="group px-6 py-3 rounded-full border transition-all duration-300 hover:scale-105 hover:shadow-md"
                      style={{
                        backgroundColor: "var(--surface)",
                        borderColor: "var(--border)",
                        color: "var(--text-primary)",
                      }}
                    >
                      <div className="flex items-center gap-2">
                        <span className="text-lg group-hover:scale-110 transition-transform duration-300">
                          {category.icon}
                        </span>
                        <span className="font-medium">{category.name}</span>
                        <span
                          className="text-sm px-2 py-1 rounded-full"
                          style={{
                            backgroundColor: "var(--accent-subtle)",
                            color: "var(--accent)",
                          }}
                        >
                          {category.count}
                        </span>
                      </div>
                    </a>
                  ))}
                </div>

                {/* Three Column Article Grid */}
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
          </div>
        </main>
      </div>
    </FadeInWrapper>
  );
}
