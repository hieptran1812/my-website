"use client";

import React, { useState, useEffect, useMemo, useCallback } from "react";
import Link from "next/link";
import { getMarkdownArticlesByCategory, Article } from "@/lib/blog";
import FadeInWrapper from "@/components/FadeInWrapper";
import { useLazyLoading } from "@/components/hooks/useLazyLoading";

export default function PaperReadingBlogPage() {
  const [selectedCategory, setSelectedCategory] = useState("all");
  const [searchTerm, setSearchTerm] = useState("");
  const [allArticles, setAllArticles] = useState<Article[]>([]);
  const [loading, setLoading] = useState(true);

  // Fetch articles from markdown files via API
  useEffect(() => {
    const fetchArticles = async () => {
      try {
        const { articles } = await getMarkdownArticlesByCategory(
          "paper-reading"
        );
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

  // Categories for filtering - based on subcategories present in the paper-reading articles
  const categories = useMemo(() => {
    // Add safety check for allArticles
    if (!Array.isArray(allArticles) || allArticles.length === 0) {
      return [{ name: "All", slug: "all", count: 0 }];
    }

    // Count subcategories
    const subcategoryCounts: Record<string, number> = {};
    allArticles.forEach((article) => {
      if (article.subcategory) {
        const normalizedSubcategory = article.subcategory.toLowerCase();
        subcategoryCounts[normalizedSubcategory] =
          (subcategoryCounts[normalizedSubcategory] || 0) + 1;
      }
    });

    const sortedSubcategories = Object.entries(subcategoryCounts)
      .sort(([, countA], [, countB]) => countB - countA)
      .map(([subcategory]) => subcategory);

    const subcategoryCategories = sortedSubcategories.map((subcategory) => ({
      name: subcategory.charAt(0).toUpperCase() + subcategory.slice(1),
      slug: subcategory,
      count: subcategoryCounts[subcategory],
    }));

    return [
      { name: "All", slug: "all", count: allArticles.length },
      ...subcategoryCategories,
    ];
  }, [allArticles]);

  // Filter articles based on subcategory and search
  const filteredArticles = useMemo(() => {
    let articlesToFilter = allArticles;

    if (selectedCategory !== "all") {
      articlesToFilter = articlesToFilter.filter(
        (article) =>
          article.subcategory &&
          article.subcategory.toLowerCase() === selectedCategory.toLowerCase()
      );
    }

    if (searchTerm) {
      articlesToFilter = articlesToFilter.filter(
        (article) =>
          article.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
          article.excerpt?.toLowerCase().includes(searchTerm.toLowerCase()) ||
          (article.subcategory &&
            article.subcategory
              .toLowerCase()
              .includes(searchTerm.toLowerCase())) ||
          (Array.isArray(article.tags) &&
            article.tags.some((tag) =>
              tag.toLowerCase().includes(searchTerm.toLowerCase())
            ))
      );
    }
    return articlesToFilter;
  }, [allArticles, selectedCategory, searchTerm]);

  // Initialize lazy loading with filtered articles
  const {
    data: displayedArticles,
    loading: loadingMore,
    hasMoreData,
    reset,
  } = useLazyLoading({
    initialData: filteredArticles.slice(0, ITEMS_PER_PAGE),
    loadMoreData: async (page: number, limit: number) => {
      // Simulate network delay for better UX
      await new Promise((resolve) => setTimeout(resolve, 500));

      const startIndex = (page - 1) * limit;
      const endIndex = startIndex + limit;
      return filteredArticles.slice(startIndex, endIndex);
    },
    itemsPerPage: ITEMS_PER_PAGE,
    hasMore: filteredArticles.length > ITEMS_PER_PAGE,
  });

  // Memoized reset function to prevent infinite loops
  const resetLazyLoading = useCallback(() => {
    reset(filteredArticles.slice(0, ITEMS_PER_PAGE));
  }, [filteredArticles, reset, ITEMS_PER_PAGE]);

  // Reset lazy loading when filters change
  useEffect(() => {
    resetLazyLoading();
  }, [resetLazyLoading]);

  // Featured articles (used if you want to display featured section)
  // const featuredArticles = allArticles.filter((article) => article.featured);

  if (loading) {
    return (
      <FadeInWrapper duration={600}>
        <div
          className="flex flex-col min-h-screen transition-colors duration-300"
          style={{
            backgroundColor: "var(--background)",
            color: "var(--text-primary)",
          }}
        >
          <main className="flex-1">
            <div className="max-w-6xl mx-auto px-6 py-16">
              <div className="text-center py-20">
                <div
                  className="inline-block animate-spin rounded-full h-8 w-8 border-b-2"
                  style={{ borderColor: "var(--accent)" }}
                ></div>
                <p
                  className="mt-4 text-lg"
                  style={{ color: "var(--text-secondary)" }}
                >
                  Loading articles...
                </p>
              </div>
            </div>
          </main>
        </div>
      </FadeInWrapper>
    );
  }

  return (
    <FadeInWrapper duration={800}>
      <div
        className="flex flex-col min-h-screen transition-colors duration-300 relative overflow-hidden"
        style={{
          backgroundColor: "var(--background)",
          color: "var(--text-primary)",
        }}
      >
        {/* Academic-themed Background Decorations */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute top-24 left-24 w-32 h-32 bg-gradient-to-br from-blue-500/10 to-indigo-500/10 rounded-full animate-pulse"></div>
          <div
            className="absolute top-36 right-24 w-28 h-28 bg-gradient-to-br from-indigo-500/10 to-blue-600/10 rounded-full animate-bounce"
            style={{ animationDelay: "2s" }}
          ></div>
          <div
            className="absolute bottom-40 left-20 w-24 h-24 bg-gradient-to-br from-blue-600/10 to-indigo-600/10 rounded-full animate-ping"
            style={{ animationDelay: "4s" }}
          ></div>
          <div
            className="absolute bottom-24 right-20 w-36 h-36 bg-gradient-to-br from-indigo-600/10 to-blue-500/10 rounded-full animate-pulse"
            style={{ animationDelay: "6s" }}
          ></div>

          {/* Academic symbols */}
          <div className="absolute top-40 right-40 text-blue-500/20 text-7xl animate-float">
            📊
          </div>
          <div
            className="absolute bottom-40 left-40 text-indigo-500/20 text-6xl animate-float"
            style={{ animationDelay: "3s" }}
          >
            📈
          </div>
          <div
            className="absolute top-1/2 left-1/3 text-blue-600/20 text-5xl animate-float"
            style={{ animationDelay: "1.5s" }}
          >
            🔬
          </div>
          <div
            className="absolute bottom-1/3 right-1/3 text-indigo-600/20 text-4xl animate-float"
            style={{ animationDelay: "4.5s" }}
          >
            📝
          </div>

          {/* Research dots */}
          <div className="absolute top-1/4 left-1/2 w-2 h-2 bg-blue-500/30 rounded-full animate-ping"></div>
          <div
            className="absolute bottom-1/4 right-1/2 w-2 h-2 bg-indigo-500/30 rounded-full animate-ping"
            style={{ animationDelay: "2s" }}
          ></div>
        </div>

        <main className="flex-1 relative z-10">
          <div className="max-w-6xl mx-auto px-6 py-16">
            {/* Header */}
            <div className="text-center mb-16">
              <div className="inline-flex items-center gap-3 mb-6">
                <div
                  className="w-12 h-12 rounded-full flex items-center justify-center text-white text-xl font-bold animate-pulse"
                  style={{
                    background: "linear-gradient(135deg, #3b82f6, #6366f1)",
                  }}
                >
                  📄
                </div>
                <h1
                  className="text-4xl md:text-5xl font-bold"
                  style={{
                    background:
                      "linear-gradient(135deg, #3b82f6, #6366f1, #8b5cf6)",
                    WebkitBackgroundClip: "text",
                    WebkitTextFillColor: "transparent",
                    backgroundClip: "text",
                  }}
                >
                  Paper Reading
                </h1>
              </div>
              <p
                className="text-xl max-w-3xl mx-auto leading-relaxed mb-8"
                style={{ color: "var(--text-secondary)" }}
              >
                In-depth reviews and analysis of cutting-edge research papers in
                AI, machine learning, and computer science.
              </p>
              <div className="flex flex-wrap justify-center gap-2">
                {[
                  "Transformers",
                  "Computer Vision",
                  "LLM",
                  "AI Research",
                  "Deep Learning",
                  "NLP",
                ].map((tag) => (
                  <span
                    key={tag}
                    className="px-3 py-1 text-sm rounded-full border transition-colors duration-200 hover:bg-[var(--surface)] cursor-pointer"
                    style={{
                      borderColor: "var(--border)",
                      backgroundColor: "var(--surface)",
                      color: "var(--text-secondary)",
                    }}
                  >
                    {tag}
                  </span>
                ))}
              </div>
            </div>

            {/* Paper Topics */}
            <div className="mb-12">
              <h2
                className="text-2xl font-bold mb-6"
                style={{ color: "var(--text-primary)" }}
              >
                Paper Reading Topics
              </h2>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3 mb-8">
                {[
                  { name: "Multimodal", slug: "Multimodal" },
                  { name: "Computer Vision", slug: "Computer Vision" },
                  { name: "LLM", slug: "LLM" },
                  { name: "AI Agent", slug: "AI Agent" },
                  {
                    name: "AI Interpretability",
                    slug: "AI Interpretability",
                  },
                  { name: "Machine Learning", slug: "Machine Learning" },
                  { name: "Speech Processing", slug: "Speech Processing" },
                ].map((topic, index) => {
                  const topicCount = // Renamed variable to avoid conflict
                    topic.slug === "all"
                      ? allArticles.length
                      : allArticles.filter((a) => a.subcategory === topic.slug)
                          .length;
                  return (
                    <FadeInWrapper
                      key={topic.slug}
                      delay={300 + index * 100}
                      duration={400}
                      direction="up"
                    >
                      <button
                        onClick={() => setSelectedCategory(topic.slug)}
                        className="group relative p-4 rounded-xl border transition-all duration-200 hover:shadow-lg hover:scale-105 text-center w-full h-[100px] flex flex-col justify-center items-center"
                        style={{
                          backgroundColor:
                            selectedCategory === topic.slug
                              ? "var(--accent-subtle)"
                              : "var(--surface)",
                          borderColor:
                            selectedCategory === topic.slug
                              ? "var(--accent)"
                              : "var(--border)",
                          color:
                            selectedCategory === topic.slug
                              ? "var(--accent)"
                              : "var(--text-primary)",
                        }}
                      >
                        <div className="font-semibold text-sm mb-2 leading-tight text-center px-1">
                          {topic.name}
                        </div>
                        {topicCount > 0 && ( // Use renamed variable
                          <div
                            className="text-xs opacity-75 font-medium px-2 py-1 rounded-full"
                            style={{
                              backgroundColor:
                                selectedCategory === topic.slug
                                  ? "var(--accent)"
                                  : "var(--accent-subtle)",
                              color:
                                selectedCategory === topic.slug
                                  ? "white"
                                  : "var(--accent)",
                            }}
                          >
                            {topicCount} article{topicCount !== 1 ? "s" : ""}
                          </div>
                        )}
                        {selectedCategory === topic.slug && (
                          <div className="absolute top-2 right-2 w-2 h-2 rounded-full bg-current opacity-75"></div>
                        )}
                      </button>
                    </FadeInWrapper>
                  );
                })}
              </div>

              {/* Results Summary */}
              <div className="text-center mb-4">
                <p style={{ color: "var(--text-secondary)" }}>
                  {selectedCategory !== "all"
                    ? `Showing ${filteredArticles.length} papers in ${
                        categories.find((c) => c.slug === selectedCategory)
                          ?.name || selectedCategory
                      }`
                    : `${allArticles.length} total research papers`}
                </p>
              </div>
            </div>

            {/* Articles Grid */}
            <div className="mb-16">
              <div className="flex items-center justify-between mb-8">
                <h2
                  className="text-2xl font-bold"
                  style={{ color: "var(--text-primary)" }}
                >
                  All Paper Reading Articles
                </h2>

                {/* Category Filter Pills */}
                <div className="flex flex-wrap gap-2">
                  <button
                    onClick={() => setSelectedCategory("all")}
                    className={`px-3 py-1 rounded-full text-xs font-medium transition-all duration-200 border ${
                      selectedCategory === "all"
                        ? "text-white"
                        : "hover:bg-[var(--surface)]"
                    }`}
                    style={
                      selectedCategory === "all"
                        ? {
                            backgroundColor: "var(--accent)",
                            borderColor: "var(--accent)",
                          }
                        : {
                            borderColor: "var(--border)",
                            backgroundColor: "var(--surface)",
                            color: "var(--text-secondary)",
                          }
                    }
                  >
                    All
                  </button>
                  {categories.slice(1, 6).map(
                    (
                      category // Display up to 5 tag-based categories
                    ) => (
                      <button
                        key={category.slug}
                        onClick={() => setSelectedCategory(category.slug)}
                        className={`px-3 py-1 rounded-full text-xs font-medium transition-all duration-200 border ${
                          selectedCategory === category.slug
                            ? "text-white"
                            : "hover:bg-[var(--surface)]"
                        }`}
                        style={
                          selectedCategory === category.slug
                            ? {
                                backgroundColor: "var(--accent)",
                                borderColor: "var(--accent)",
                              }
                            : {
                                borderColor: "var(--border)",
                                backgroundColor: "var(--surface)",
                                color: "var(--text-secondary)",
                              }
                        }
                      >
                        {category.name}
                      </button>
                    )
                  )}
                </div>
              </div>

              {/* Article List */}
              {filteredArticles.length > 0 ? (
                <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
                  {displayedArticles.map((article) => (
                    <article
                      key={article.id}
                      className="group rounded-xl border transition-all duration-300 hover:shadow-xl hover:scale-105 overflow-hidden"
                      style={{
                        backgroundColor: "var(--surface)",
                        borderColor: "var(--border)",
                      }}
                    >
                      <div className="p-6">
                        <div className="flex items-center justify-between mb-4">
                          <span
                            className="px-2 py-1 text-xs font-medium rounded-full"
                            style={{
                              backgroundColor: "var(--surface)",
                              color: "var(--text-secondary)",
                              border: "1px solid var(--border)",
                            }}
                          >
                            {article.subcategory}
                          </span>
                          <span
                            className="text-xs"
                            style={{ color: "var(--text-muted)" }}
                          >
                            {article.readTime}
                          </span>
                        </div>

                        <h3
                          className="text-lg font-semibold mb-3 group-hover:text-[var(--accent)] transition-colors line-clamp-2"
                          style={{ color: "var(--text-primary)" }}
                        >
                          {article.title}
                        </h3>

                        <p
                          className="text-sm mb-4 leading-relaxed line-clamp-3"
                          style={{ color: "var(--text-secondary)" }}
                        >
                          {article.excerpt}
                        </p>

                        <div className="flex flex-wrap gap-1 mb-4">
                          {article.tags.slice(0, 3).map((tag) => (
                            <span
                              key={tag}
                              className="px-2 py-1 text-xs rounded-full"
                              style={{
                                backgroundColor: "var(--accent-subtle)",
                                color: "var(--accent)",
                              }}
                            >
                              {tag}
                            </span>
                          ))}
                        </div>

                        <div
                          className="flex items-center justify-between text-xs mt-4"
                          style={{ color: "var(--text-muted)" }}
                        >
                          <span>
                            📅{" "}
                            {new Date(article.date).toLocaleDateString(
                              "en-US",
                              {
                                year: "numeric",
                                month: "short",
                                day: "numeric",
                              }
                            )}
                          </span>
                          <Link
                            href={`/blog/${article.slug}`}
                            className="inline-flex items-center text-[var(--accent)]"
                          >
                            Read more
                            <svg
                              className="w-3 h-3 ml-1"
                              fill="none"
                              stroke="currentColor"
                              viewBox="0 0 24 24"
                            >
                              <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth="2"
                                d="M9 5l7 7-7 7"
                              />
                            </svg>
                          </Link>
                        </div>
                      </div>
                    </article>
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
                    {searchTerm
                      ? `No articles match "${searchTerm}"`
                      : `No articles in "${
                          categories.find((c) => c.slug === selectedCategory)
                            ?.name
                        }" category`}
                  </p>
                  <button
                    onClick={() => {
                      setSearchTerm("");
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

            {/* Replace Load More Button with a loading indicator */}
            {hasMoreData && loadingMore && (
              <div className="flex justify-center my-8">
                <div
                  className="w-8 h-8 rounded-full border-2 border-b-transparent animate-spin"
                  style={{
                    borderColor:
                      "var(--accent) transparent var(--accent) var(--accent)",
                  }}
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
