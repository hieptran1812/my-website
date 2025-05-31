"use client";

import React, { useState, useEffect, useMemo, useCallback } from "react";
import Link from "next/link";
import { getMarkdownArticlesByCategory, Article } from "@/lib/blog";
import FadeInWrapper from "@/components/FadeInWrapper";
import { useLazyLoading } from "@/components/hooks/useLazyLoading";

export default function CryptoBlogPage() {
  const [selectedCategory, setSelectedCategory] = useState("all");
  const [searchTerm, setSearchTerm] = useState("");
  const [allArticles, setAllArticles] = useState<Article[]>([]);
  const [loading, setLoading] = useState(true);

  // Fetch initial articles from markdown files via API
  useEffect(() => {
    const fetchArticles = async () => {
      try {
        const articles = await getMarkdownArticlesByCategory("crypto");
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

  // Categories updated to include "All" and up to 5 most relevant tags
  const categories = useMemo(() => {
    if (!allArticles || allArticles.length === 0) {
      return [{ name: "All", slug: "all", count: 0 }];
    }

    const tagCounts: Record<string, number> = {};
    allArticles.forEach((article) => {
      // Ensure article.tags exists and is an array before iterating
      if (Array.isArray(article.tags)) {
        article.tags.forEach((tag) => {
          const normalizedTag = tag.toLowerCase();
          tagCounts[normalizedTag] = (tagCounts[normalizedTag] || 0) + 1;
        });
      }
    });

    const sortedTags = Object.entries(tagCounts)
      .sort(([, countA], [, countB]) => countB - countA)
      .map(([tag]) => tag);

    const topTags = sortedTags.slice(0, 5);

    const tagCategories = topTags.map((tag) => ({
      name: tag.charAt(0).toUpperCase() + tag.slice(1), // Capitalize first letter
      slug: tag,
      count: allArticles.filter(
        (a) =>
          Array.isArray(a.tags) && a.tags.some((t) => t.toLowerCase() === tag)
      ).length,
    }));

    return [
      { name: "All", slug: "all", count: allArticles.length },
      ...tagCategories,
    ];
  }, [allArticles]);

  // Update filtering to handle "All" and tag-based categories
  const filteredArticles = useMemo(() => {
    let articlesToFilter = allArticles;

    if (selectedCategory !== "all") {
      articlesToFilter = articlesToFilter.filter(
        (article) =>
          Array.isArray(article.tags) &&
          article.tags.some((tag) => tag.toLowerCase() === selectedCategory)
      );
    }

    if (searchTerm) {
      articlesToFilter = articlesToFilter.filter(
        (article) =>
          article.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
          article.excerpt?.toLowerCase().includes(searchTerm.toLowerCase()) || // Use excerpt
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

  // Reset lazy loading when filters change - use callback to prevent infinite loop
  const resetLazyLoading = useCallback(() => {
    const initialData = filteredArticles.slice(0, ITEMS_PER_PAGE);
    reset(initialData);
  }, [filteredArticles, reset, ITEMS_PER_PAGE]);

  useEffect(() => {
    resetLazyLoading();
  }, [resetLazyLoading]);

  const featuredArticles = allArticles.filter((article) => article.featured);

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
                  ₿
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
                  Cryptocurrency
                </h1>
              </div>
              <p
                className="text-xl max-w-3xl mx-auto leading-relaxed mb-8"
                style={{ color: "var(--text-secondary)" }}
              >
                Exploring blockchain technology, DeFi protocols, and the future
                of decentralized finance. From fundamentals to advanced trading
                strategies.
              </p>
              <div className="flex flex-wrap justify-center gap-2">
                {[
                  "Blockchain",
                  "DeFi",
                  "Smart Contracts",
                  "Trading",
                  "NFTs",
                  "Web3",
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

            {/* Crypto Stats */}
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-4 mb-16">
              <div
                className="p-4 rounded-xl border text-center"
                style={{
                  backgroundColor: "var(--surface)",
                  borderColor: "var(--border)",
                }}
              >
                <div
                  className="text-2xl font-bold mb-1"
                  style={{ color: "var(--accent)" }}
                >
                  {allArticles.length}
                </div>
                <div
                  className="text-sm"
                  style={{ color: "var(--text-secondary)" }}
                >
                  Articles Published
                </div>
              </div>
              <div
                className="p-4 rounded-xl border text-center"
                style={{
                  backgroundColor: "var(--surface)",
                  borderColor: "var(--border)",
                }}
              >
                <div
                  className="text-2xl font-bold mb-1"
                  style={{ color: "var(--accent)" }}
                >
                  {categories.length - 1}
                </div>
                <div
                  className="text-sm"
                  style={{ color: "var(--text-secondary)" }}
                >
                  Categories
                </div>
              </div>
              <div
                className="p-4 rounded-xl border text-center"
                style={{
                  backgroundColor: "var(--surface)",
                  borderColor: "var(--border)",
                }}
              >
                <div
                  className="text-2xl font-bold mb-1"
                  style={{ color: "var(--accent)" }}
                >
                  2024-2025
                </div>
                <div
                  className="text-sm"
                  style={{ color: "var(--text-secondary)" }}
                >
                  Publication Range
                </div>
              </div>
              <div
                className="p-4 rounded-xl border text-center"
                style={{
                  backgroundColor: "var(--surface)",
                  borderColor: "var(--border)",
                }}
              >
                <div
                  className="text-2xl font-bold mb-1"
                  style={{ color: "var(--accent)" }}
                >
                  {Math.round(
                    allArticles.reduce(
                      (acc, article) => acc + parseInt(article.readTime),
                      0
                    ) / allArticles.length
                  )}
                </div>
                <div
                  className="text-sm"
                  style={{ color: "var(--text-secondary)" }}
                >
                  Avg. Read Time
                </div>
              </div>
            </div>

            {/* Featured Article */}
            {featuredArticles.length > 0 && (
              <div className="mb-16">
                <div className="flex items-center gap-3 mb-8">
                  <svg
                    className="w-6 h-6"
                    style={{ color: "var(--accent)" }}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                  </svg>
                  <h2
                    className="text-2xl font-bold"
                    style={{ color: "var(--text-primary)" }}
                  >
                    Featured Article
                  </h2>
                </div>

                <div
                  className="rounded-2xl p-8 border transition-all duration-300 hover:shadow-xl"
                  style={{
                    backgroundColor: "var(--surface)",
                    borderColor: "var(--border)",
                    background:
                      "linear-gradient(145deg, var(--surface), var(--surface-hover))",
                  }}
                >
                  {featuredArticles.slice(0, 1).map((article) => (
                    <div key={article.id}>
                      <div className="grid md:grid-cols-3 gap-6 mb-6">
                        <div>
                          <div
                            className="text-sm font-medium mb-1"
                            style={{ color: "var(--text-secondary)" }}
                          >
                            Category
                          </div>
                          <div
                            className="font-semibold"
                            style={{ color: "var(--text-primary)" }}
                          >
                            {article.subcategory}
                          </div>
                        </div>
                        <div>
                          <div
                            className="text-sm font-medium mb-1"
                            style={{ color: "var(--text-secondary)" }}
                          >
                            Category
                          </div>
                          <div
                            className="font-semibold"
                            style={{ color: "var(--text-primary)" }}
                          >
                            {article.subcategory}
                          </div>
                        </div>
                        <div>
                          <div
                            className="text-sm font-medium mb-1"
                            style={{ color: "var(--text-secondary)" }}
                          >
                            Reading Time
                          </div>
                          <div
                            className="font-semibold"
                            style={{ color: "var(--text-primary)" }}
                          >
                            {article.readTime}
                          </div>
                        </div>
                      </div>

                      <div className="flex flex-wrap gap-2 mb-4">
                        {article.tags.map((tag) => (
                          <span
                            key={tag}
                            className="px-3 py-1 text-xs font-medium rounded-full"
                            style={{
                              backgroundColor: "var(--accent-subtle)",
                              color: "var(--accent)",
                            }}
                          >
                            {tag}
                          </span>
                        ))}
                      </div>

                      <h3
                        className="text-2xl md:text-3xl font-bold mb-4"
                        style={{ color: "var(--text-primary)" }}
                      >
                        {article.title}
                      </h3>

                      <p
                        className="text-lg mb-6 leading-relaxed"
                        style={{ color: "var(--text-secondary)" }}
                      >
                        {article.excerpt}
                      </p>

                      <div className="flex items-center justify-between">
                        <div
                          className="flex items-center gap-4 text-sm"
                          style={{ color: "var(--text-muted)" }}
                        >
                          <span>📝 {article.date}</span>
                          <span>•</span>
                          <span>₿ Crypto Analysis</span>
                        </div>
                        <Link
                          href={`/blog/${article.slug}`}
                          className="inline-flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-all duration-200 hover:scale-105"
                          style={{
                            backgroundColor: "var(--accent)",
                            color: "white",
                          }}
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
                              strokeWidth="2"
                              d="M9 5l7 7-7 7"
                            />
                          </svg>
                        </Link>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Crypto Topics - updated to match available content */}
            <div className="mb-12">
              <h2
                className="text-2xl font-bold mb-6"
                style={{ color: "var(--text-primary)" }}
              >
                Crypto Topics
              </h2>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3 mb-8">
                {[
                  { name: "DeFi", slug: "defi" },
                  { name: "NFTs", slug: "nfts" },
                  { name: "Trading", slug: "trading" },
                  { name: "Blockchain", slug: "blockchain" },
                  { name: "Technology", slug: "technology" },
                  { name: "Fundamentals", slug: "fundamentals" },
                ].map((topic, index) => {
                  const count =
                    topic.slug === "all"
                      ? allArticles.length
                      : allArticles.filter(
                          (a) =>
                            a.subcategory === topic.slug ||
                            a.tags?.some((tag) => {
                              const tagLower = tag.toLowerCase();
                              const topicLower = topic.slug.toLowerCase();
                              return (
                                tagLower.includes(topicLower) ||
                                (topic.slug === "defi" &&
                                  tagLower.includes("defi")) ||
                                (topic.slug === "blockchain" &&
                                  tagLower.includes("blockchain")) ||
                                (topic.slug === "nfts" &&
                                  tagLower.includes("nft"))
                              );
                            })
                        ).length;
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
                    : `${allArticles.length} total crypto articles`}
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
                  All Crypto Articles
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

            {/* Loading indicator for lazy loading */}
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
