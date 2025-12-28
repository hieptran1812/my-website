"use client";

import React, { useState, useEffect, useMemo, useCallback, useRef } from "react";
import Link from "next/link";
import Image from "next/image";
import { useParams } from "next/navigation";
import FadeInWrapper from "@/components/FadeInWrapper";
import TagBadge, { TagList } from "@/components/TagBadge";
import { formatDateMedium } from "@/lib/dateUtils";

interface BlogPost {
  slug: string;
  title: string;
  publishDate: string;
  readTime: string;
  category: string;
  subcategory?: string;
  author: string;
  tags: string[];
  image: string;
  excerpt: string;
}

export default function TagPage() {
  const params = useParams();
  const tagSlug = params.tag as string;

  // Decode and format the tag name for display
  const tagName = decodeURIComponent(tagSlug).replace(/-/g, " ");
  const displayTagName =
    tagName.charAt(0).toUpperCase() + tagName.slice(1).toLowerCase();

  const [articles, setArticles] = useState<BlogPost[]>([]);
  const [loading, setLoading] = useState(true);
  const [displayedCount, setDisplayedCount] = useState(12);
  const [loadingMore, setLoadingMore] = useState(false);

  const ITEMS_PER_PAGE = 12;
  const SCROLL_THRESHOLD = 85; // Percentage of page scrolled to trigger loading

  // Ref for the loading sentinel element
  const loadingRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    async function fetchArticles() {
      try {
        const response = await fetch("/api/blog/posts");
        const allPosts: BlogPost[] = await response.json();

        // Filter articles that have the selected tag (case-insensitive)
        const filteredArticles = allPosts.filter((post) =>
          post.tags?.some(
            (t) => t.toLowerCase().replace(/\s+/g, "-") === tagSlug.toLowerCase()
          )
        );

        // Sort by date (newest first)
        filteredArticles.sort((a, b) => {
          const dateA = new Date(a.publishDate);
          const dateB = new Date(b.publishDate);
          return dateB.getTime() - dateA.getTime();
        });

        setArticles(filteredArticles);
      } catch (error) {
        console.error("Error fetching articles:", error);
        setArticles([]);
      } finally {
        setLoading(false);
      }
    }

    fetchArticles();
  }, [tagSlug]);

  // Get displayed articles
  const displayedArticles = articles.slice(0, displayedCount);
  const hasMoreArticles = displayedCount < articles.length;

  // Load more function
  const loadMoreArticles = useCallback(async () => {
    if (loadingMore || !hasMoreArticles) return;
    setLoadingMore(true);
    // Small delay for smooth UX
    await new Promise((resolve) => setTimeout(resolve, 200));
    setDisplayedCount((prev) => prev + ITEMS_PER_PAGE);
    setLoadingMore(false);
  }, [loadingMore, hasMoreArticles]);

  // Lazy loading on scroll
  useEffect(() => {
    const handleScroll = () => {
      if (loadingMore || !hasMoreArticles) return;

      const windowHeight = window.innerHeight;
      const documentHeight = document.documentElement.scrollHeight;
      const scrollTop = window.scrollY || document.documentElement.scrollTop;

      // Calculate how far down the user has scrolled (as a percentage)
      const scrolledPercentage =
        ((scrollTop + windowHeight) / documentHeight) * 100;

      // If the user has scrolled beyond the threshold, load more data
      if (scrolledPercentage > SCROLL_THRESHOLD) {
        loadMoreArticles();
      }
    };

    window.addEventListener("scroll", handleScroll, { passive: true });

    return () => {
      window.removeEventListener("scroll", handleScroll);
    };
  }, [loadMoreArticles, loadingMore, hasMoreArticles]);

  // Get related tags (tags that often appear together with the current tag)
  const relatedTags = useMemo(() => {
    const relatedTagCounts: Record<string, number> = {};

    articles.forEach((article) => {
      article.tags?.forEach((tag) => {
        if (tag.toLowerCase().replace(/\s+/g, "-") !== tagSlug.toLowerCase()) {
          relatedTagCounts[tag] = (relatedTagCounts[tag] || 0) + 1;
        }
      });
    });

    return Object.entries(relatedTagCounts)
      .map(([tag, count]) => ({ tag, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 10);
  }, [articles, tagSlug]);

  if (loading) {
    return (
      <FadeInWrapper duration={800}>
        <div className="flex flex-col min-h-screen items-center justify-center">
          <div className="relative">
            <div
              className="animate-spin rounded-full h-16 w-16 border-4 border-t-transparent"
              style={{ borderColor: "var(--accent)", borderTopColor: "transparent" }}
            />
            <div
              className="absolute inset-0 animate-ping rounded-full h-16 w-16 border-4 opacity-20"
              style={{ borderColor: "var(--accent)" }}
            />
          </div>
          <p className="mt-6 text-lg" style={{ color: "var(--text-secondary)" }}>
            Loading articles...
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
            {/* Header */}
            <FadeInWrapper duration={600} delay={100}>
              <div className="text-center mb-12">
                {/* Tag Icon */}
                <div className="inline-flex items-center justify-center mb-6">
                  <div
                    className="w-16 h-16 rounded-2xl flex items-center justify-center shadow-lg"
                    style={{
                      backgroundColor: "var(--accent)",
                      boxShadow: "0 8px 32px rgba(var(--accent-rgb, 130, 170, 255), 0.3)",
                    }}
                  >
                    <svg
                      className="w-8 h-8 text-white"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z"
                      />
                    </svg>
                  </div>
                </div>

                {/* Tag Name */}
                <h1
                  className="text-4xl md:text-5xl font-bold mb-4"
                  style={{ color: "var(--text-primary)" }}
                >
                  #{displayTagName}
                </h1>

                {/* Article Count */}
                <p
                  className="text-lg mb-6"
                  style={{ color: "var(--text-secondary)" }}
                >
                  {articles.length === 0
                    ? "No articles found with this tag"
                    : articles.length === 1
                    ? "1 article"
                    : `${articles.length} articles`}
                </p>

                {/* Breadcrumb */}
                <div
                  className="flex items-center justify-center gap-2 text-sm"
                  style={{ color: "var(--text-muted)" }}
                >
                  <Link
                    href="/blog"
                    className="hover:text-[var(--accent)] transition-colors"
                  >
                    Blog
                  </Link>
                  <span>/</span>
                  <Link
                    href="/blog/tags"
                    className="hover:text-[var(--accent)] transition-colors"
                  >
                    Tags
                  </Link>
                  <span>/</span>
                  <span style={{ color: "var(--accent)" }}>{displayTagName}</span>
                </div>
              </div>
            </FadeInWrapper>

            {/* Related Tags Section */}
            {relatedTags.length > 0 && (
              <FadeInWrapper duration={600} delay={200}>
                <div className="mb-12">
                  <h2
                    className="text-sm font-semibold uppercase tracking-wider mb-4 text-center"
                    style={{ color: "var(--text-muted)" }}
                  >
                    Related Tags
                  </h2>
                  <div className="flex flex-wrap justify-center gap-2">
                    {relatedTags.map(({ tag, count }) => (
                      <TagBadge
                        key={tag}
                        tag={tag}
                        count={count}
                        variant="default"
                        clickable={true}
                      />
                    ))}
                  </div>
                </div>
              </FadeInWrapper>
            )}

            {/* Articles Grid */}
            {articles.length > 0 ? (
              <FadeInWrapper duration={600} delay={300}>
                <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
                  {displayedArticles.map((article, index) => (
                    <FadeInWrapper
                      key={article.slug}
                      duration={400}
                      delay={Math.min(350 + index * 50, 600)}
                    >
                      <Link href={`/blog/${article.slug}`} className="block group">
                        <article
                          className="rounded-xl border overflow-hidden transition-all duration-300 hover:shadow-xl hover:scale-[1.02]"
                          style={{
                            backgroundColor: "var(--surface)",
                            borderColor: "var(--border)",
                          }}
                        >
                          {/* Image */}
                          <div className="relative h-48 overflow-hidden">
                            <Image
                              src={article.image || "/blog-placeholder.jpg"}
                              alt={article.title}
                              fill
                              className="object-cover transition-transform duration-500 group-hover:scale-110"
                              sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
                            />
                            <div className="absolute inset-0 bg-gradient-to-t from-black/40 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                          </div>

                          {/* Content */}
                          <div className="p-6">
                            {/* Category Badge */}
                            <div className="flex items-center justify-between mb-3">
                              <span
                                className="text-xs font-semibold uppercase tracking-wider"
                                style={{ color: "var(--accent)" }}
                              >
                                {article.category}
                              </span>
                              <span
                                className="text-xs"
                                style={{ color: "var(--text-muted)" }}
                              >
                                {article.readTime}
                              </span>
                            </div>

                            {/* Title */}
                            <h3
                              className="text-lg font-semibold mb-3 line-clamp-2 group-hover:text-[var(--accent)] transition-colors duration-300"
                              style={{ color: "var(--text-primary)" }}
                            >
                              {article.title}
                            </h3>

                            {/* Tags */}
                            {article.tags && article.tags.length > 0 && (
                              <div
                                className="mb-4"
                                onClick={(e) => e.stopPropagation()}
                              >
                                <TagList
                                  tags={article.tags}
                                  maxTags={3}
                                  variant="compact"
                                  clickable={true}
                                />
                              </div>
                            )}

                            {/* Excerpt */}
                            <p
                              className="text-sm mb-4 line-clamp-2"
                              style={{ color: "var(--text-secondary)" }}
                            >
                              {article.excerpt}
                            </p>

                            {/* Footer */}
                            <div
                              className="flex items-center justify-between text-xs"
                              style={{ color: "var(--text-muted)" }}
                            >
                              <span>{formatDateMedium(article.publishDate)}</span>
                              <span className="inline-flex items-center gap-1 text-[var(--accent)] group-hover:gap-2 transition-all duration-300">
                                Read more
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
                              </span>
                            </div>
                          </div>
                        </article>
                      </Link>
                    </FadeInWrapper>
                  ))}
                </div>

                {/* Loading indicator for infinite scroll */}
                {hasMoreArticles && (
                  <div
                    ref={loadingRef}
                    className="flex justify-center items-center py-12 mt-8"
                  >
                    {loadingMore ? (
                      <div className="flex items-center gap-3">
                        <div
                          className="animate-spin rounded-full h-8 w-8 border-3 border-t-transparent"
                          style={{
                            borderColor: "var(--accent)",
                            borderTopColor: "transparent",
                          }}
                        />
                        <span
                          className="text-sm font-medium"
                          style={{ color: "var(--text-secondary)" }}
                        >
                          Loading more articles...
                        </span>
                      </div>
                    ) : (
                      <div className="flex flex-col items-center gap-2">
                        <div className="flex gap-1">
                          <div
                            className="w-2 h-2 rounded-full animate-bounce"
                            style={{
                              backgroundColor: "var(--accent)",
                              animationDelay: "0ms",
                            }}
                          />
                          <div
                            className="w-2 h-2 rounded-full animate-bounce"
                            style={{
                              backgroundColor: "var(--accent)",
                              animationDelay: "150ms",
                            }}
                          />
                          <div
                            className="w-2 h-2 rounded-full animate-bounce"
                            style={{
                              backgroundColor: "var(--accent)",
                              animationDelay: "300ms",
                            }}
                          />
                        </div>
                        <span
                          className="text-xs"
                          style={{ color: "var(--text-muted)" }}
                        >
                          Scroll for more
                        </span>
                      </div>
                    )}
                  </div>
                )}

                {/* End of list indicator */}
                {!hasMoreArticles && displayedArticles.length > ITEMS_PER_PAGE && (
                  <div className="flex justify-center items-center py-8 mt-4">
                    <div
                      className="flex items-center gap-2 px-4 py-2 rounded-full"
                      style={{
                        backgroundColor: "var(--surface)",
                        border: "1px solid var(--border)",
                      }}
                    >
                      <svg
                        className="w-4 h-4"
                        style={{ color: "var(--accent)" }}
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M5 13l4 4L19 7"
                        />
                      </svg>
                      <span
                        className="text-sm"
                        style={{ color: "var(--text-secondary)" }}
                      >
                        You&apos;ve seen all {articles.length} articles
                      </span>
                    </div>
                  </div>
                )}
              </FadeInWrapper>
            ) : (
              <FadeInWrapper duration={600} delay={300}>
                <div className="text-center py-16">
                  <div
                    className="w-24 h-24 rounded-full flex items-center justify-center mx-auto mb-6"
                    style={{ backgroundColor: "var(--accent-subtle)" }}
                  >
                    <svg
                      className="w-12 h-12"
                      style={{ color: "var(--accent)" }}
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={1.5}
                        d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                      />
                    </svg>
                  </div>
                  <h2
                    className="text-2xl font-bold mb-4"
                    style={{ color: "var(--text-primary)" }}
                  >
                    No articles found
                  </h2>
                  <p
                    className="text-lg mb-8"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    There are no articles with the tag &quot;{displayTagName}&quot; yet.
                  </p>
                  <Link
                    href="/blog/tags"
                    className="inline-flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-all duration-300 hover:scale-105"
                    style={{
                      backgroundColor: "var(--accent)",
                      color: "white",
                    }}
                  >
                    <svg
                      className="w-5 h-5"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z"
                      />
                    </svg>
                    Browse All Tags
                  </Link>
                </div>
              </FadeInWrapper>
            )}

            {/* Navigation */}
            <nav
              className="mt-16 pt-8 border-t"
              style={{ borderColor: "var(--border)" }}
            >
              <div className="flex flex-wrap justify-center gap-4">
                <Link
                  href="/blog/tags"
                  className="inline-flex items-center gap-2 px-6 py-3 rounded-lg transition-all duration-300 hover:scale-105 border"
                  style={{
                    backgroundColor: "var(--surface)",
                    color: "var(--text-primary)",
                    borderColor: "var(--border)",
                  }}
                >
                  <svg
                    className="w-5 h-5"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z"
                    />
                  </svg>
                  All Tags
                </Link>
                <Link
                  href="/blog"
                  className="inline-flex items-center gap-2 px-6 py-3 rounded-lg transition-all duration-300 hover:scale-105 border"
                  style={{
                    backgroundColor: "var(--surface)",
                    color: "var(--text-primary)",
                    borderColor: "var(--border)",
                  }}
                >
                  <svg
                    className="w-5 h-5"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M15 19l-7-7 7-7"
                    />
                  </svg>
                  Back to Blog
                </Link>
              </div>
            </nav>
          </div>
        </main>
      </div>
    </FadeInWrapper>
  );
}
