"use client";

import React, { useState, useEffect, useMemo, useCallback } from "react";
import Link from "next/link";
import Image from "next/image";
import { getMarkdownArticlesByCategory, Article } from "@/lib/blog";
import FadeInWrapper from "@/components/FadeInWrapper";
import { useLazyLoading } from "@/components/hooks/useLazyLoading";
import { formatDateShort, formatDateMedium } from "@/lib/dateUtils";
import CollectionTag from "@/components/CollectionTag";
import { getArticleImageUrl } from "@/lib/articleImage";
import SubcategoryFilter from "@/components/SubcategoryFilter";
import BlogSearchBar from "@/components/BlogSearchBar";
import { useArticleSearch } from "@/components/hooks/useArticleSearch";


export default function PaperReadingBlogPage() {
  const [selectedCategories, setSelectedCategories] = useState<string[]>([]);
  const [allArticles, setAllArticles] = useState<Article[]>([]);
  const [loading, setLoading] = useState(true);

  // Fetch articles from markdown files via API
  useEffect(() => {
    const fetchArticles = async () => {
      try {
        // Use a high limit to fetch all articles (default is 50)
        const { articles } = await getMarkdownArticlesByCategory(
          "paper-reading",
          1,
          2000, // Fetch the full category — needed so the subcategory filter shows all values
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

  const { searchTerm, setSearchTerm, filteredArticles } = useArticleSearch(
    allArticles,
    selectedCategories,
  );

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
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-500"></div>
          <p className="mt-4 text-lg">Loading papers...</p>
        </div>
      </FadeInWrapper>
    );
  }

  if (allArticles.length === 0) {
    return (
      <FadeInWrapper duration={800}>
        <div className="flex flex-col min-h-screen items-center justify-center">
          <h1 className="text-4xl font-bold mb-4">
            No paper reading articles found
          </h1>
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
                    background: "linear-gradient(135deg, #3b82f6, #6366f1)",
                  }}
                >
                  📚
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
                Some summaries, explanations, and thoughts of mine while reading
                research papers on Computer Science, AI, and Machine Learning
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
                      <div className="md:col-span-3 relative w-full aspect-[672/366]">
                        <Image
                          src={getArticleImageUrl(featuredArticle)}
                          alt={featuredArticle.title}
                          fill
                          className="object-cover"
                          sizes="(max-width: 768px) 100vw, 60vw"
                        />
                        <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent"></div>
                      </div>

                      {/* Right: Article Info (40%) */}
                      <div className="md:col-span-2 p-8 flex flex-col justify-center md:overflow-hidden">
                        <div className="mb-4">
                          <span
                            className="text-sm font-bold uppercase tracking-wider"
                            style={{ color: "var(--accent)" }}
                          >
                            {featuredArticle.category}
                          </span>
                        </div>

                        {/* Collection tag */}
                        {featuredArticle.collection && (
                          <div className="mb-4">
                            <CollectionTag
                              collection={featuredArticle.collection}
                              variant="default"
                            />
                          </div>
                        )}

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
                          className="text-base leading-relaxed mb-6 line-clamp-4"
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
                              src={getArticleImageUrl(article)}
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

                            {/* Collection tag */}
                            {article.collection && (
                              <div className="mb-2">
                                <CollectionTag
                                  collection={article.collection}
                                  variant="compact"
                                />
                              </div>
                            )}

                            <h4
                              className="text-sm font-semibold mb-2 leading-tight line-clamp-2 group-hover:text-[var(--accent)] transition-colors duration-300"
                              style={{ color: "var(--text-primary)" }}
                            >
                              <Link href={`/blog/${article.slug}`}>
                                {article.title}
                              </Link>
                            </h4>

                            {/* Tags under title */}
                            {article.tags && article.tags.length > 0 && (
                              <div className="flex flex-wrap gap-1 mb-2">
                                {article.tags.slice(0, 1).map((tag) => (
                                  <span
                                    key={tag}
                                    className="px-1.5 py-0.5 text-xs rounded-full"
                                    style={{
                                      backgroundColor: "var(--accent-subtle)",
                                      color: "var(--accent)",
                                    }}
                                  >
                                    {tag}
                                  </span>
                                ))}
                              </div>
                            )}

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
                    {selectedCategories.length > 0 || searchTerm.trim()
                      ? `Showing ${filteredArticles.length} of ${allArticles.length} papers`
                      : `${allArticles.length} total research papers`}
                  </p>
                </div>

                {/* Search & Filter Bar */}
                <div className="flex flex-col sm:flex-row items-center justify-center gap-3 mb-12">
                  <BlogSearchBar
                    value={searchTerm}
                    onChange={setSearchTerm}
                    placeholder="Search papers..."
                  />
                  <SubcategoryFilter
                    categories={categories.slice(1)}
                    selectedSlugs={selectedCategories}
                    onSelectionChange={setSelectedCategories}
                    totalCount={allArticles.length}
                  />
                </div>

                {/* Articles Grid - 3 columns */}
                {filteredArticles.length > 0 ? (
                  <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
                    {displayedArticles.map((article) => (
                      <Link
                        key={article.id}
                        href={`/blog/${article.slug}`}
                        className="block"
                      >
                        <div
                          className="group rounded-lg border overflow-hidden transition-all duration-300 hover:shadow-lg hover:scale-[1.02] cursor-pointer"
                          style={{
                            backgroundColor: "var(--card-bg)",
                            borderColor: "var(--card-border)",
                          }}
                        >
                          <div className="relative h-48 overflow-hidden">
                            <Image
                              src={getArticleImageUrl(article)}
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

                            {/* Collection tag */}
                            {article.collection && (
                              <div className="mb-3">
                                <CollectionTag
                                  collection={article.collection}
                                  variant="default"
                                />
                              </div>
                            )}

                            <h3
                              className="text-lg font-semibold mb-3 leading-tight group-hover:text-[var(--accent)] transition-colors duration-300"
                              style={{ color: "var(--text-primary)" }}
                            >
                              {article.title}
                            </h3>

                            {/* Tags under title */}
                            {article.tags && article.tags.length > 0 && (
                              <div className="flex flex-wrap gap-1 mb-3">
                                {article.tags.slice(0, 2).map((tag) => (
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
                            )}

                            <div
                              className="text-sm flex items-center gap-3 mb-4"
                              style={{ color: "var(--text-secondary)" }}
                            >
                              <span>
                                {new Date(article.date).toLocaleDateString()}
                              </span>
                              <span>•</span>
                              <span>{article.readTime}</span>
                            </div>
                          </div>
                        </div>
                      </Link>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-12">
                    <div className="text-6xl mb-4">�</div>
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
                      No articles found for the selected filters.
                    </p>
                    <button
                      onClick={() => {
                        setSelectedCategories([]);
                        setSearchTerm("");
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
