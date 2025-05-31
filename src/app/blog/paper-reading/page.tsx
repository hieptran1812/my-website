"use client";

import React, { useState, useEffect, useMemo, useCallback } from "react";
import Link from "next/link";
import { getMarkdownArticlesByCategory, Article } from "@/lib/blog";
import FadeInWrapper from "@/components/FadeInWrapper";
import ArticleCard from "@/components/ArticleCard";
import ArticleGrid from "@/components/ArticleGrid";
import LoadMoreTrigger from "@/components/LoadMoreTrigger";
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
        const articles = await getMarkdownArticlesByCategory("paper-reading");
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

  // Filter articles based on category and search - memoized to prevent infinite re-renders
  const filteredArticles = useMemo(() => {
    return allArticles.filter((article) => {
      const matchesCategory =
        selectedCategory === "all" || article.subcategory === selectedCategory;
      const matchesSearch =
        article.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
        article.excerpt.toLowerCase().includes(searchTerm.toLowerCase()) ||
        article.tags.some((tag) =>
          tag.toLowerCase().includes(searchTerm.toLowerCase())
        );
      return matchesCategory && matchesSearch;
    });
  }, [allArticles, selectedCategory, searchTerm]);

  // Categories for filtering - memoized to prevent unnecessary recalculation
  const categories = useMemo(
    () => [
      { name: "All", slug: "all", count: allArticles.length },
      {
        name: "Computer Vision",
        slug: "Computer Vision",
        count: allArticles.filter((a) => a.subcategory === "Computer Vision")
          .length,
      },
      {
        name: "LLM",
        slug: "LLM",
        count: allArticles.filter((a) => a.subcategory === "LLM").length,
      },
      {
        name: "AI Interpretability",
        slug: "AI Interpretability",
        count: allArticles.filter(
          (a) => a.subcategory === "AI Interpretability"
        ).length,
      },
      {
        name: "Multimodal",
        slug: "Multimodal",
        count: allArticles.filter((a) => a.subcategory === "Multimodal").length,
      },
      {
        name: "Speech & Audio",
        slug: "Speech & Audio",
        count: allArticles.filter((a) => a.subcategory === "Speech & Audio")
          .length,
      },
      {
        name: "Autonomous Agents",
        slug: "Autonomous Agents",
        count: allArticles.filter((a) => a.subcategory === "Autonomous Agents")
          .length,
      },
      {
        name: "Ensemble Learning",
        slug: "Ensemble Learning",
        count: allArticles.filter((a) => a.subcategory === "Ensemble Learning")
          .length,
      },
    ],
    [allArticles]
  );

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
                  Loading papers...
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
            <FadeInWrapper direction="up" delay={100} duration={600}>
              <div className="text-center mb-16">
                <div className="inline-flex items-center gap-3 mb-6">
                  <div
                    className="w-12 h-12 rounded-full flex items-center justify-center text-white text-xl font-bold"
                    style={{
                      background:
                        "linear-gradient(135deg, var(--accent), var(--accent-hover))",
                    }}
                  >
                    📄
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
                    Paper Reading
                  </h1>
                </div>
                <p
                  className="text-xl max-w-3xl mx-auto leading-relaxed mb-8"
                  style={{ color: "var(--text-secondary)" }}
                >
                  Deep dives into influential research papers across AI, machine
                  learning, and computer science. Exploring cutting-edge ideas
                  and their practical implications.
                </p>
              </div>
            </FadeInWrapper>

            {/* Research Topics */}
            <FadeInWrapper direction="up" delay={200} duration={600}>
              <div className="mb-12">
                <h2
                  className="text-2xl font-bold mb-6"
                  style={{ color: "var(--text-primary)" }}
                >
                  Research Areas
                </h2>
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-4 mb-8">
                  {[
                    "Computer Vision",
                    "LLM",
                    "AI Interpretability",
                    "Multimodal",
                    "Speech & Audio",
                    "Autonomous Agents",
                    "Ensemble Learning",
                  ].map((topic) => (
                    <FadeInWrapper
                      key={topic}
                      delay={300 + Math.random() * 200}
                      duration={400}
                      direction="up"
                    >
                      <button
                        onClick={() => setSelectedCategory(topic)}
                        className="p-4 rounded-lg border transition-all duration-200 hover:shadow-md hover:scale-105 text-center min-h-[80px] flex flex-col justify-center"
                        style={{
                          backgroundColor: "var(--surface)",
                          borderColor: "var(--border)",
                          color: "var(--text-primary)",
                        }}
                      >
                        <div className="font-medium text-sm leading-tight">
                          {topic}
                        </div>
                      </button>
                    </FadeInWrapper>
                  ))}
                </div>

                {/* Results Summary */}
                <div className="text-center mb-4">
                  <p style={{ color: "var(--text-secondary)" }}>
                    {selectedCategory !== "all"
                      ? `Showing ${filteredArticles.length} papers in ${
                          categories.find((c) => c.slug === selectedCategory)
                            ?.name || selectedCategory
                        }`
                      : `${allArticles.length} total research papers analyzed`}
                  </p>
                </div>
              </div>
            </FadeInWrapper>

            {/* Articles Grid */}
            <FadeInWrapper direction="up" delay={300} duration={600}>
              <div className="mb-16">
                <div className="flex items-center justify-between mb-8">
                  <h2
                    className="text-2xl font-bold"
                    style={{ color: "var(--text-primary)" }}
                  >
                    Paper Reviews & Analysis ({filteredArticles.length})
                  </h2>

                  {/* Category Filter Pills */}
                  <div className="flex flex-wrap gap-2">
                    {categories.slice(0, 3).map((category) => (
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
                        {category.name} ({category.count})
                      </button>
                    ))}
                  </div>
                </div>

                {displayedArticles.length > 0 ? (
                  <>
                    <ArticleGrid variant="default">
                      {displayedArticles.map((article, index) => (
                        <ArticleCard
                          key={article.id}
                          article={article}
                          index={index}
                          variant="default"
                        />
                      ))}
                    </ArticleGrid>

                    {/* Load More Trigger */}
                    <LoadMoreTrigger
                      onLoadMore={loadMore}
                      loading={loadingMore}
                      hasMore={hasMoreData}
                    />
                  </>
                ) : (
                  <div className="text-center py-12">
                    <div className="text-6xl mb-4">🔍</div>
                    <h3
                      className="text-xl font-semibold mb-2"
                      style={{ color: "var(--text-primary)" }}
                    >
                      No papers found
                    </h3>
                    <p
                      className="mb-4"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      {searchTerm
                        ? `No papers match "${searchTerm}"`
                        : `No papers in "${
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
            </FadeInWrapper>

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
