"use client";

import React, { useState, useEffect, useMemo, useCallback } from "react";
import Link from "next/link";
import Image from "next/image";
import { getMarkdownArticlesByCategory, Article } from "@/lib/blog";
import FadeInWrapper from "@/components/FadeInWrapper";
import { useLazyLoading } from "@/components/hooks/useLazyLoading";
import { formatDateShort, formatDateMedium } from "@/lib/dateUtils";

// Define the list of trading subtopics
const tradingSubtopics = [
  { name: "Trading", slug: "trading" },
  { name: "Crypto", slug: "crypto" },
  { name: "Economics", slug: "economics" },
  { name: "Investing", slug: "investing" },
  { name: "Quantitative Analysis", slug: "quantitative analysis" },
  { name: "Finance", slug: "finance" },
];

export default function TradingBlogPage() {
  const [selectedCategory, setSelectedCategory] = useState("all");
  const [allArticles, setAllArticles] = useState<Article[]>([]);
  const [loading, setLoading] = useState(true);

  // Fetch initial articles from markdown files via API
  useEffect(() => {
    const fetchArticles = async () => {
      try {
        const { articles } = await getMarkdownArticlesByCategory("trading");
        // Ensure articles is always an array
        setAllArticles(Array.isArray(articles) ? articles : []);
      } catch (error) {
        console.error("Error fetching articles:", error);
        setAllArticles([]); // Set to empty array on error
      } finally {
        setLoading(false);
      }
    };

    fetchArticles();
  }, []);

  // Lazy loading configuration
  const ITEMS_PER_PAGE = 9;

  // Categories based on subtopics
  const categories = useMemo(() => {
    // Add safety check for allArticles
    if (!Array.isArray(allArticles) || allArticles.length === 0) {
      return [{ name: "All", slug: "all", count: 0 }];
    }

    // Count articles for each subtopic
    const subtopicCounts: Record<string, number> = {};
    tradingSubtopics.forEach((subtopic) => {
      const count = allArticles.filter(
        (article) =>
          article.subcategory &&
          article.subcategory.toLowerCase() === subtopic.slug.toLowerCase()
      ).length;
      if (count > 0) {
        subtopicCounts[subtopic.slug.toLowerCase()] = count;
      }
    });

    // Create categories from subtopics that have articles
    const subtopicCategories = tradingSubtopics
      .filter((subtopic) => subtopicCounts[subtopic.slug.toLowerCase()] > 0)
      .map((subtopic) => ({
        name: subtopic.name,
        slug: subtopic.slug.toLowerCase(),
        count: subtopicCounts[subtopic.slug.toLowerCase()],
      }));

    return [
      { name: "All", slug: "all", count: allArticles.length },
      ...subtopicCategories,
    ];
  }, [allArticles]);

  // Filter based on subcategory
  const filteredArticles = useMemo(() => {
    let articlesToFilter = allArticles;

    if (selectedCategory !== "all") {
      // Filter by subcategory only
      articlesToFilter = articlesToFilter.filter(
        (article) =>
          article.subcategory &&
          article.subcategory.toLowerCase() === selectedCategory.toLowerCase()
      );
    }

    return articlesToFilter;
  }, [allArticles, selectedCategory]);

  // Initialize lazy loading with filtered articles
  const {
    data: displayedArticles,
    loading: loadingMore,
    hasMoreData,
    reset,
  } = useLazyLoading({
    initialData: filteredArticles.slice(0, ITEMS_PER_PAGE),
    loadMoreData: async (page: number, limit: number) => {
      // Simulate API delay
      await new Promise((resolve) => setTimeout(resolve, 300));

      const startIndex = (page - 1) * limit;
      const endIndex = startIndex + limit;

      return filteredArticles.slice(startIndex, endIndex);
    },
    itemsPerPage: ITEMS_PER_PAGE,
    hasMore: filteredArticles.length > ITEMS_PER_PAGE,
    getItemId: (article: Article) => article.id,
  });

  // Reset lazy loading when filters change
  const resetLazyLoading = useCallback(() => {
    const initialItems = filteredArticles.slice(0, ITEMS_PER_PAGE);
    const hasMore = filteredArticles.length > ITEMS_PER_PAGE;
    reset(initialItems, hasMore);
  }, [filteredArticles, reset, ITEMS_PER_PAGE]);

  useEffect(() => {
    resetLazyLoading();
  }, [resetLazyLoading]);

  if (loading) {
    return (
      <FadeInWrapper duration={800}>
        <div className="flex flex-col min-h-screen items-center justify-center">
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-orange-500"></div>
          <p className="mt-4 text-lg">Loading articles...</p>
        </div>
      </FadeInWrapper>
    );
  }

  if (allArticles.length === 0) {
    return (
      <FadeInWrapper duration={800}>
        <div className="flex flex-col min-h-screen items-center justify-center">
          <h1 className="text-4xl font-bold mb-4">No articles found</h1>
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
            {/* Header with Category Name */}
            <div className="text-center mb-16">
              <div className="inline-flex items-center gap-3 mb-6">
                <div
                  className="w-12 h-12 rounded-full flex items-center justify-center text-white text-xl font-bold"
                  style={{
                    background: "linear-gradient(135deg, #f97316, #fb923c)",
                  }}
                >
                  ₿
                </div>
                <h1
                  className="text-4xl md:text-5xl font-bold"
                  style={{
                    background:
                      "linear-gradient(135deg, #f97316, #fb923c, #fdba74)",
                    WebkitBackgroundClip: "text",
                    WebkitTextFillColor: "transparent",
                    backgroundClip: "text",
                  }}
                >
                  Trading
                </h1>
              </div>
              <p
                className="text-xl max-w-3xl mx-auto leading-relaxed mb-8"
                style={{ color: "var(--text-secondary)" }}
              >
                A collection of my learnings and reflections on economics,
                finance, and trading, written to document my growth and
                understanding of the financial markets.
              </p>
            </div>

            {/* Latest Articles Section */}
            {featuredArticle && (
              <FadeInWrapper duration={600} delay={200}>
                <div className="mb-16">
                  <h2
                    className="text-3xl font-bold mb-8"
                    style={{ color: "var(--text-primary)" }}
                  >
                    Latest Articles
                  </h2>

                  {/* Featured Article - Two Column Layout (60/40) */}
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
                          src={
                            featuredArticle.image &&
                            featuredArticle.image.trim() !== "" &&
                            featuredArticle.image !== "/blog-placeholder.jpg" &&
                            featuredArticle.image !== "/images/default-blog.jpg"
                              ? featuredArticle.image
                              : "/blog-placeholder.jpg"
                          }
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
                          <Link
                            href={`/blog/${featuredArticle.slug}`}
                            className="hover:text-[var(--accent)] transition-colors duration-300"
                          >
                            {featuredArticle.title}
                          </Link>
                        </h3>
                        <div
                          className="text-sm flex items-center gap-4 mb-4"
                          style={{ color: "var(--text-secondary)" }}
                        >
                          <span>{formatDateMedium(featuredArticle.date)}</span>
                          <span>•</span>
                          <span>{featuredArticle.readTime}</span>
                        </div>
                        <p
                          className="text-base leading-relaxed mb-6"
                          style={{ color: "var(--text-secondary)" }}
                        >
                          {featuredArticle.excerpt}
                        </p>
                        <Link
                          href={`/blog/${featuredArticle.slug}`}
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
                        </Link>
                      </div>
                    </div>
                  </div>

                  {/* Four Column Grid of Recent Articles */}
                  {recentArticles.length > 0 && (
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                      {recentArticles.map((article) => (
                        <div
                          key={article.id}
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
                              <Link href={`/blog/${article.slug}`}>
                                {article.title}
                              </Link>
                            </h4>
                            <div
                              className="text-xs flex items-center gap-2"
                              style={{ color: "var(--text-secondary)" }}
                            >
                              <span>{formatDateShort(article.date)}</span>
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
              <div className="mb-16">
                {/* Section Title */}
                <div className="text-center mb-12">
                  <h2
                    className="text-3xl font-bold mb-4"
                    style={{ color: "var(--text-primary)" }}
                  >
                    All Articles
                  </h2>
                  <p style={{ color: "var(--text-secondary)" }}>
                    {selectedCategory !== "all"
                      ? `Showing ${filteredArticles.length} articles in ${
                          categories.find((c) => c.slug === selectedCategory)
                            ?.name || selectedCategory
                        }`
                      : `${allArticles.length} total trading articles`}
                  </p>
                </div>

                {/* Predefined Subtopic Filter Pills */}
                <div className="flex flex-wrap justify-center gap-3 mb-12">
                  <button
                    onClick={() => setSelectedCategory("all")}
                    className={`px-5 py-2.5 rounded-full text-sm font-semibold transition-all duration-300 border hover:scale-105 active:scale-95 ${
                      selectedCategory === "all"
                        ? "text-white shadow-lg"
                        : "hover:bg-[var(--surface)] hover:shadow-md"
                    }`}
                    style={
                      selectedCategory === "all"
                        ? {
                            backgroundColor: "var(--accent)",
                            borderColor: "var(--accent)",
                            boxShadow: "0 4px 12px rgba(255, 165, 0, 0.3)",
                          }
                        : {
                            borderColor: "var(--border)",
                            backgroundColor: "var(--surface)",
                            color: "var(--text-secondary)",
                          }
                    }
                  >
                    <span className="flex items-center gap-2">
                      All
                      <span
                        className="px-2 py-0.5 rounded-full text-xs font-bold"
                        style={{
                          backgroundColor:
                            selectedCategory === "all"
                              ? "rgba(255, 255, 255, 0.2)"
                              : "var(--accent)",
                          color: selectedCategory === "all" ? "white" : "white",
                        }}
                      >
                        {allArticles.length}
                      </span>
                    </span>
                  </button>
                  {tradingSubtopics.map((subtopic) => {
                    const articleCount = allArticles.filter(
                      (article) =>
                        article.subcategory &&
                        article.subcategory.toLowerCase() ===
                          subtopic.slug.toLowerCase()
                    ).length;

                    if (articleCount === 0) return null;

                    return (
                      <button
                        key={subtopic.slug}
                        onClick={() =>
                          setSelectedCategory(subtopic.slug.toLowerCase())
                        }
                        className={`px-5 py-2.5 rounded-full text-sm font-semibold transition-all duration-300 border hover:scale-105 active:scale-95 ${
                          selectedCategory === subtopic.slug.toLowerCase()
                            ? "text-white shadow-lg"
                            : "hover:bg-[var(--surface)] hover:shadow-md"
                        }`}
                        style={
                          selectedCategory === subtopic.slug.toLowerCase()
                            ? {
                                backgroundColor: "var(--accent)",
                                borderColor: "var(--accent)",
                                boxShadow: "0 4px 12px rgba(255, 165, 0, 0.3)",
                              }
                            : {
                                borderColor: "var(--border)",
                                backgroundColor: "var(--surface)",
                                color: "var(--text-secondary)",
                              }
                        }
                      >
                        <span className="flex items-center gap-2">
                          {subtopic.name}
                          <span
                            className="px-2 py-0.5 rounded-full text-xs font-bold"
                            style={{
                              backgroundColor:
                                selectedCategory === subtopic.slug.toLowerCase()
                                  ? "rgba(255, 255, 255, 0.2)"
                                  : "var(--accent)",
                              color:
                                selectedCategory === subtopic.slug.toLowerCase()
                                  ? "white"
                                  : "white",
                            }}
                          >
                            {articleCount}
                          </span>
                        </span>
                      </button>
                    );
                  })}
                </div>

                {/* Articles Grid */}
                {filteredArticles.length > 0 ? (
                  <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
                    {displayedArticles.map((article) => (
                      <div
                        key={article.id}
                        className="group rounded-lg border overflow-hidden transition-all duration-300 hover:shadow-lg hover:scale-[1.02]"
                        style={{
                          backgroundColor: "var(--card-bg)",
                          borderColor: "var(--card-border)",
                        }}
                      >
                        <div className="relative h-48 overflow-hidden">
                          <Image
                            src={article.image || "/blog-placeholder.jpg"}
                            alt={article.title}
                            fill
                            className="object-cover transition-transform duration-300 group-hover:scale-105"
                            sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
                          />
                        </div>
                        <div className="p-6">
                          <div className="mb-3">
                            <span
                              className="text-xs font-medium uppercase tracking-wider"
                              style={{ color: "var(--accent)" }}
                            >
                              {article.category}
                            </span>
                          </div>
                          <h3
                            className="text-lg font-semibold mb-3 leading-tight group-hover:text-[var(--accent)] transition-colors duration-300"
                            style={{ color: "var(--text-primary)" }}
                          >
                            <Link href={`/blog/${article.slug}`}>
                              {article.title}
                            </Link>
                          </h3>
                          <div
                            className="text-sm flex items-center gap-3 mb-4"
                            style={{ color: "var(--text-secondary)" }}
                          >
                            <span>{formatDateMedium(article.date)}</span>
                            <span>•</span>
                            <span>{article.readTime}</span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-12">
                    <div className="text-6xl mb-4">🔍</div>
                    <h3
                      className="text-xl font-semibold mb-2"
                      style={{ color: "var(--text-primary)" }}
                    >
                      No articles found
                    </h3>
                    <p
                      className="mb-4"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      {`No articles in "${
                        categories.find((c) => c.slug === selectedCategory)
                          ?.name
                      }" category`}
                    </p>
                    <button
                      onClick={() => {
                        setSelectedCategory("all");
                      }}
                      className="px-4 py-2 rounded-lg transition-colors text-white"
                      style={{ backgroundColor: "var(--accent)" }}
                    >
                      Clear filters
                    </button>
                  </div>
                )}
              </div>
            </FadeInWrapper>

            {/* Load More Indicator */}
            {hasMoreData && loadingMore && (
              <div className="flex justify-center my-8">
                <div
                  className="animate-spin rounded-full h-8 w-8 border-b-2"
                  style={{ borderColor: "var(--accent)" }}
                ></div>
              </div>
            )}

            {/* Navigation */}
            <nav
              className="mt-12 pt-8 border-t"
              style={{ borderColor: "var(--border)" }}
            >
              <div className="flex justify-center">
                <Link
                  href="/blog"
                  className="inline-flex items-center px-6 py-3 rounded-lg transition-colors border"
                  style={{
                    backgroundColor: "var(--surface)",
                    color: "var(--text-primary)",
                    borderColor: "var(--border)",
                  }}
                >
                  ← Back to All Blogs
                </Link>
              </div>
            </nav>
          </div>
        </main>
      </div>
    </FadeInWrapper>
  );
}
