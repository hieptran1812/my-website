"use client";

import React, { useState, useEffect, useMemo, useCallback } from "react";
import Link from "next/link";
import { getMarkdownArticlesByCategory, Article } from "@/lib/blog";
import FadeInWrapper from "@/components/FadeInWrapper";
import { useLazyLoading } from "@/components/hooks/useLazyLoading";
import LoadMoreTrigger from "@/components/LoadMoreTrigger";

export default function NotesBlogPage() {
  const [selectedCategory, setSelectedCategory] = useState("all");
  const [searchTerm, setSearchTerm] = useState("");
  const [allArticles, setAllArticles] = useState<Article[]>([]);
  const [loading, setLoading] = useState(true);

  // Fetch articles from markdown files via API
  useEffect(() => {
    const fetchArticles = async () => {
      try {
        const articles = await getMarkdownArticlesByCategory("notes");
        setAllArticles(articles);
      } catch (error) {
        console.error("Error fetching articles:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchArticles();
  }, []);

  // Lazy loading configuration
  const ITEMS_PER_PAGE = 9;

  // Categories for filtering - based on subcategories present in the notes articles
  const categories = useMemo(() => {
    if (!allArticles || allArticles.length === 0) {
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
    loadMore,
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
        className="flex flex-col min-h-screen transition-colors duration-300"
        style={{
          backgroundColor: "var(--background)",
          color: "var(--text-primary)",
        }}
      >
        <main className="flex-1">
          <div className="max-w-6xl mx-auto px-6 py-16">
            {/* Header */}
            <div className="text-center mb-16">
              <div className="inline-flex items-center gap-3 mb-6">
                <div
                  className="w-12 h-12 rounded-full flex items-center justify-center text-white text-xl font-bold"
                  style={{
                    background:
                      "linear-gradient(135deg, var(--accent), var(--accent-hover))",
                  }}
                >
                  📝
                </div>
                <h1
                  className="text-4xl md:text-5xl font-bold"
                  style={{
                    background:
                      "linear-gradient(135deg, var(--accent), var(--accent-hover))",
                    WebkitBackgroundClip: "text",
                    WebkitTextFillColor: "transparent",
                    backgroundClip: "text",
                  }}
                >
                  Notes
                </h1>
              </div>
              <p
                className="text-xl max-w-3xl mx-auto leading-relaxed mb-8"
                style={{ color: "var(--text-secondary)" }}
              >
                A collection of random thoughts, book summaries, and personal
                reflections on various topics including technology,
                productivity, and career development.
              </p>
              <div className="flex flex-wrap justify-center gap-2">
                {[
                  "Book Summaries",
                  "Random Ideas",
                  "Self-reflection",
                  "Learning",
                  "Productivity",
                  "Career",
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

            {/* Notes Topics - show only categories with content */}
            <div className="mb-12">
              <h2
                className="text-2xl font-bold mb-6"
                style={{ color: "var(--text-primary)" }}
              >
                Notes Topics
              </h2>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3 mb-8">
                {[
                  { name: "Book Summaries", slug: "Book Summaries" },
                  { name: "Idea Dump", slug: "Idea Dump" },
                  {
                    name: "Self-reflection Entries",
                    slug: "Self-reflection Entries",
                  },
                ].map((topic, index) => {
                  const count =
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
                          {count} article{count !== 1 ? "s" : ""}
                        </div>
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
                    ? `Showing ${filteredArticles.length} articles in ${
                        categories.find((c) => c.slug === selectedCategory)
                          ?.name || selectedCategory
                      }`
                    : `${allArticles.length} total notes articles`}
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
                  All Notes
                </h2>

                {/* Category Filter Pills */}
                <div className="flex flex-wrap gap-2">
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

            {/* Load More Trigger */}
            <LoadMoreTrigger
              onLoadMore={loadMore}
              loading={loadingMore}
              hasMore={hasMoreData}
            />

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
