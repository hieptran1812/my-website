"use client";

import React, { useState, useEffect } from "react";
import Image from "next/image";
import FadeInWrapper from "@/components/FadeInWrapper";
import ArticleCard from "@/components/ArticleCard";
import ArticleGrid from "@/components/ArticleGrid";
import LoadMoreTrigger from "@/components/LoadMoreTrigger";
import { useLazyLoading } from "@/components/hooks/useLazyLoading";

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
function convertToLibArticle(article: Article): import("@/lib/blog").Article {
  return {
    id: article.link.split("/").pop() || "unknown",
    title: article.title,
    excerpt: article.summary,
    content: "",
    image: article.image || "/placeholder-image.jpg",
    category: article.category,
    subcategory: "",
    date: article.date,
    readTime: article.readTime,
    tags: [],
    difficulty: "Intermediate" as const,
    slug: article.link.split("/").pop() || "unknown",
    featured: false,
  };
}

export default function BlogPage() {
  const [articles, setArticles] = useState<Article[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    async function fetchPosts() {
      try {
        const response = await fetch("/api/blog/posts");
        const blogPosts: BlogPostMetadata[] = await response.json();
        const convertedArticles = blogPosts.map(convertToArticle);
        setArticles(convertedArticles);
      } catch (error) {
        console.error("Error loading blog posts:", error);
        // Fallback to empty array
        setArticles([]);
      } finally {
        setIsLoading(false);
      }
    }

    fetchPosts();
  }, []);

  // Items per page for lazy loading
  const ITEMS_PER_PAGE = 9;

  // Lazy loading for articles (excluding the featured article)
  const articlesForLazyLoading = articles.slice(1); // Remove first article (featured)

  const {
    data: displayedArticles,
    loading: loadingMore,
    hasMoreData,
    loadMore,
  } = useLazyLoading({
    initialData: articlesForLazyLoading.slice(0, ITEMS_PER_PAGE),
    loadMoreData: async (page: number, limit: number) => {
      // Simulate network delay for smooth loading experience
      await new Promise((resolve) => setTimeout(resolve, 500));
      const startIndex = (page - 1) * limit;
      const endIndex = startIndex + limit;
      return articlesForLazyLoading.slice(startIndex, endIndex);
    },
    itemsPerPage: ITEMS_PER_PAGE,
    hasMore: articlesForLazyLoading.length > ITEMS_PER_PAGE,
  });

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
                        className="group relative overflow-hidden rounded-xl p-6 border transition-all duration-300 hover:scale-105 hover:shadow-lg blog-content-stagger"
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
                              className="text-sm font-medium px-2 py-1 rounded-full"
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

            {/* Featured Article */}
            {articles.length > 0 && (
              <FadeInWrapper duration={600} delay={400}>
                <div className="mb-16">
                  <h2 className="text-3xl font-bold mb-8 gradient-text-purple blog-content-stagger">
                    Featured Article
                  </h2>
                  <a
                    href={articles[0].link}
                    className="group block rounded-xl overflow-hidden border transition-all duration-300 card-enhanced blog-content-stagger"
                    style={{
                      backgroundColor: "var(--card-bg)",
                      borderColor: "var(--card-border)",
                      animationDelay: "0.4s",
                    }}
                  >
                    <div className="md:flex">
                      <div className="md:w-1/2 relative overflow-hidden">
                        <Image
                          src={articles[0].image}
                          alt={articles[0].title}
                          width={600}
                          height={400}
                          className="w-full h-64 md:h-full object-cover group-hover:scale-105 transition-transform duration-500"
                        />
                        <div className="absolute inset-0 bg-gradient-to-r from-transparent to-black/10 group-hover:to-black/20 transition-all duration-300"></div>
                      </div>
                      <div className="md:w-1/2 p-8">
                        <div className="flex items-center gap-4 mb-4">
                          <span
                            className="px-3 py-1 text-sm font-medium rounded-full transform group-hover:scale-105 transition-transform duration-300"
                            style={{
                              backgroundColor: "var(--accent-subtle)",
                              color: "var(--accent)",
                            }}
                          >
                            {articles[0].category}
                          </span>
                          <span
                            className="text-sm"
                            style={{ color: "var(--text-muted)" }}
                          >
                            {articles[0].readTime}
                          </span>
                        </div>
                        <h3
                          className="text-2xl font-bold mb-4 transition-colors duration-300 group-hover:bg-gradient-to-r group-hover:from-blue-600 group-hover:to-purple-600 group-hover:bg-clip-text group-hover:text-transparent"
                          style={{ color: "var(--text-primary)" }}
                        >
                          {articles[0].title}
                        </h3>
                        <p
                          className="mb-6 leading-relaxed"
                          style={{ color: "var(--text-secondary)" }}
                        >
                          {articles[0].summary}
                        </p>
                        <div className="flex items-center justify-between">
                          <span
                            className="text-sm"
                            style={{ color: "var(--text-muted)" }}
                          >
                            {new Date(articles[0].date).toLocaleDateString(
                              "en-US",
                              {
                                year: "numeric",
                                month: "long",
                                day: "numeric",
                              }
                            )}
                          </span>
                          <span
                            className="font-medium transition-all duration-300 group-hover:text-blue-500 group-hover:translate-x-1"
                            style={{ color: "var(--accent)" }}
                          >
                            Read More →
                          </span>
                        </div>
                      </div>
                    </div>
                  </a>
                </div>
              </FadeInWrapper>
            )}

            {/* All Articles with Lazy Loading */}
            <FadeInWrapper duration={600} delay={500}>
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
                          delay={600 + index * 50}
                        >
                          <ArticleCard
                            article={convertToLibArticle(article)}
                            index={index}
                            variant="default"
                          />
                        </FadeInWrapper>
                      ))}
                    </ArticleGrid>

                    {hasMoreData && (
                      <LoadMoreTrigger
                        onLoadMore={loadMore}
                        loading={loadingMore}
                        hasMore={hasMoreData}
                      />
                    )}
                  </>
                ) : (
                  <div className="text-center py-12">
                    <p className="text-lg text-gray-600">No articles found.</p>
                  </div>
                )}
              </div>
            </FadeInWrapper>

            {/* Newsletter CTA */}
            <FadeInWrapper duration={600} delay={700}>
              <div
                className="mt-16 p-8 rounded-xl border blog-content-stagger relative overflow-hidden"
                style={{
                  backgroundColor: "var(--card-bg)",
                  borderColor: "var(--card-border)",
                  animationDelay: "0.8s",
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
