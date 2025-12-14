"use client";

import React, { useEffect, useState, useCallback, useRef } from "react";
import Image from "next/image";
import Link from "next/link";
import CollectionTag from "@/components/CollectionTag";
import { BlogPostMetadata, calculateContentReadTime } from "../lib/blog";
import { formatDateMedium } from "../lib/dateUtils";

// Constants
const ARTICLES_TO_SHOW = 4;
const FETCH_TIMEOUT = 10000; // 10 seconds

interface Article {
  title: string;
  summary: string;
  link: string;
  image: string;
  date: string;
  tags: string[];
  readTime: string;
  collection?: string;
}

// Utility function to format read time consistently
const formatReadTime = (readTime: string): string => {
  if (!readTime || readTime.trim() === "") {
    return "2 min read";
  }

  // Ensure it always ends with "read" if not already
  const formatted = readTime.trim();
  if (!formatted.toLowerCase().includes("read")) {
    return `${formatted} read`;
  }

  return formatted;
};

export default function BlogSection() {
  const [articles, setArticles] = useState<Article[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isVisible, setIsVisible] = useState(false);
  const sectionRef = useRef<HTMLElement>(null);

  // Intersection Observer for scroll animation
  useEffect(() => {
    // Only setup observer after loading is complete
    if (loading || error || articles.length === 0) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          observer.disconnect();
        }
      },
      {
        threshold: 0.15,
        rootMargin: "0px 0px -40% 0px",
      }
    );

    if (sectionRef.current) {
      observer.observe(sectionRef.current);
    }

    return () => observer.disconnect();
  }, [loading, error, articles.length]);

  const fetchArticles = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      // Create abort controller for timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), FETCH_TIMEOUT);

      // Fetch latest blog posts from API
      const response = await fetch("/api/blog", {
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`Failed to fetch articles: ${response.status}`);
      }

      const blogPosts: BlogPostMetadata[] = await response.json();

      // Validate response
      if (!Array.isArray(blogPosts)) {
        throw new Error("Invalid response format from API");
      }

      // Convert blog posts to Article format, take latest articles
      const latestArticles: Article[] = blogPosts
        .slice(0, ARTICLES_TO_SHOW)
        .map((post) => {
          // Calculate read time as fallback if not provided
          const fallbackReadTime = post.excerpt
            ? calculateContentReadTime(post.excerpt)
            : "2 min read";

          return {
            title: post.title || "Untitled",
            summary: post.excerpt || "No summary available",
            link: `/blog/${post.slug}`,
            image: post.image || "/blog-placeholder.jpg",
            date: formatDateMedium(post.publishDate || ""),
            tags: Array.isArray(post.tags) ? post.tags : [],
            readTime: formatReadTime(post.readTime || fallbackReadTime), // Use formatted read time
            collection: post.collection,
          };
        })
        .filter((article) => article.title !== "Untitled"); // Filter out invalid articles

      setArticles(latestArticles);
    } catch (err) {
      console.error("Failed to fetch blog posts:", err);

      // Handle different error types appropriately
      if (err instanceof Error) {
        if (err.name === "AbortError") {
          setError("Request timed out. Please check your connection.");
        } else {
          setError(err.message);
        }
      } else {
        setError("An unexpected error occurred");
      }

      // Set empty array as fallback
      setArticles([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchArticles();
  }, [fetchArticles]);

  if (loading) {
    return (
      <section
        className="py-16 md:py-24 transition-colors duration-300"
        style={{ backgroundColor: "var(--surface)" }}
      >
        <div className="container mx-auto px-6 max-w-6xl">
          <div className="text-center mb-12">
            <h3 className="section-heading text-3xl md:text-4xl lg:text-5xl font-bold mb-4 transition-colors duration-300 relative">
              <span className="bg-gradient-to-r from-emerald-600 via-teal-600 to-cyan-600 bg-clip-text text-transparent">
                Latest Articles
              </span>
            </h3>
            <div className="flex justify-center items-center mt-6">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-emerald-600"></div>
            </div>
          </div>
        </div>
      </section>
    );
  }

  if (error) {
    return (
      <section className="py-16 md:py-24 section-primary">
        <div className="container mx-auto px-6 max-w-6xl">
          <div className="text-center mb-12">
            <h3 className="section-heading text-3xl md:text-4xl lg:text-5xl font-bold mb-4 transition-colors duration-300 relative">
              <span className="bg-gradient-to-r from-emerald-600 via-teal-600 to-cyan-600 bg-clip-text text-transparent">
                Latest Articles
              </span>
            </h3>
            <p
              className="text-center mt-6"
              style={{ color: "var(--text-secondary)" }}
            >
              Unable to load articles at the moment. Please try again later.
            </p>
            <Link
              href="/blog"
              className="inline-flex items-center mt-4 font-medium transition-colors"
              style={{ color: "var(--accent)" }}
            >
              Browse All Articles
              <svg
                className="ml-2 w-4 h-4"
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
      </section>
    );
  }

  if (articles.length === 0) {
    return (
      <section className="py-16 md:py-24 section-primary">
        <div className="container mx-auto px-6 max-w-6xl">
          <div className="text-center mb-12">
            <h3 className="section-heading text-3xl md:text-4xl lg:text-5xl font-bold mb-4 transition-colors duration-300 relative">
              <span className="bg-gradient-to-r from-emerald-600 via-teal-600 to-cyan-600 bg-clip-text text-transparent">
                Latest Articles
              </span>
            </h3>
            <p
              className="text-center mt-6"
              style={{ color: "var(--text-secondary)" }}
            >
              No articles available yet. Check back soon for new content!
            </p>
          </div>
        </div>
      </section>
    );
  }

  // Featured article (first one) and remaining articles
  const featuredArticle = articles[0];
  const remainingArticles = articles.slice(1, 4); // Show 3 more articles

  return (
    <section
      ref={sectionRef}
      className="py-20 md:py-28 section-primary relative overflow-hidden"
    >
      {/* Background decorative elements */}
      <div className="absolute inset-0 -z-10">
        <div className="absolute top-20 left-10 w-32 h-32 bg-gradient-to-br from-emerald-500/10 to-teal-500/10 rounded-full blur-xl"></div>
        <div className="absolute bottom-20 right-10 w-40 h-40 bg-gradient-to-br from-cyan-500/10 to-emerald-500/10 rounded-full blur-xl"></div>
      </div>

      <div className="container mx-auto px-6 max-w-7xl">
        {/* Enhanced Header Section */}
        <div
          className="text-center mb-16"
          style={{
            opacity: isVisible ? 1 : 0,
            transform: isVisible ? "translateY(0)" : "translateY(32px)",
            filter: isVisible ? "blur(0)" : "blur(8px)",
            transition:
              "opacity 1000ms cubic-bezier(0.25, 0.46, 0.45, 0.94), transform 1000ms cubic-bezier(0.25, 0.46, 0.45, 0.94), filter 1000ms cubic-bezier(0.25, 0.46, 0.45, 0.94)",
          }}
        >
          <div
            className="inline-flex items-center gap-2 px-4 py-2 mb-6 rounded-full border"
            style={{
              backgroundColor: "var(--surface)",
              borderColor: "var(--card-border)",
              color: "var(--text-muted)",
            }}
          >
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
                d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.746 0 3.332.477 4.5 1.253v13C19.832 18.477 18.246 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"
              />
            </svg>
            <span className="text-sm font-medium">Blog</span>
          </div>

          <h3 className="section-heading text-4xl md:text-5xl lg:text-6xl font-bold mb-6 transition-colors duration-300 relative">
            <span className="bg-gradient-to-r from-emerald-600 via-teal-600 to-cyan-600 bg-clip-text text-transparent">
              Latest Articles
            </span>
            <div className="absolute -bottom-3 left-1/2 transform -translate-x-1/2 w-32 h-1.5 bg-gradient-to-r from-emerald-600 via-teal-600 to-cyan-600 rounded-full"></div>
          </h3>

          <p
            className="text-lg md:text-xl max-w-3xl mx-auto transition-colors duration-300 leading-relaxed"
            style={{ color: "var(--text-secondary)" }}
          >
            A collection of articles where I document and summarize what I have
            learned about{" "}
            <span className="font-semibold" style={{ color: "var(--accent)" }}>
              AI and Machine Learning
            </span>
            ,{" "}
            <span className="font-semibold" style={{ color: "var(--accent)" }}>
              Software Engineering
            </span>
            , and{" "}
            <span className="font-semibold" style={{ color: "var(--accent)" }}>
              Product Development
            </span>
            . These notes focus on capturing key concepts, practical
            experiences, and lessons that connect theory with real world
            practice.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 items-center justify-center mt-8">
            <Link
              href="/blog"
              className="inline-flex items-center px-6 py-3 font-semibold rounded-lg transition-all duration-300 hover:shadow-lg hover:scale-105"
              style={{
                backgroundColor: "var(--accent)",
                color: "white",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = "var(--accent-hover)";
                e.currentTarget.style.transform =
                  "translateY(-2px) scale(1.05)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = "var(--accent)";
                e.currentTarget.style.transform = "translateY(0) scale(1)";
              }}
            >
              <svg
                className="w-5 h-5 mr-2"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                />
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
                />
              </svg>
              View All Articles
            </Link>

            <div
              className="flex items-center gap-4 text-sm"
              style={{ color: "var(--text-muted)" }}
            >
              <div className="flex items-center gap-1">
                <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse"></div>
                <span>50+ Articles</span>
              </div>
              <div className="w-1 h-1 bg-current rounded-full opacity-50"></div>
              <div className="flex items-center gap-1">
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
                    d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
                <span>Weekly Updates</span>
              </div>
            </div>
          </div>
        </div>

        {/* Featured Article Section */}
        {featuredArticle && (
          <div
            className="mb-16"
            style={{
              opacity: isVisible ? 1 : 0,
              transform: isVisible
                ? "translateY(0) scale(1)"
                : "translateY(64px) scale(0.97)",
              filter: isVisible ? "blur(0)" : "blur(8px)",
              transition:
                "opacity 1000ms cubic-bezier(0.25, 0.46, 0.45, 0.94), transform 1000ms cubic-bezier(0.25, 0.46, 0.45, 0.94), filter 1000ms cubic-bezier(0.25, 0.46, 0.45, 0.94)",
              transitionDelay: isVisible ? "150ms" : "0ms",
            }}
          >
            <div
              className="mb-8"
              style={{
                opacity: isVisible ? 1 : 0,
                transform: isVisible ? "translateX(0)" : "translateX(-16px)",
                transition: "opacity 700ms ease-out, transform 700ms ease-out",
                transitionDelay: isVisible ? "300ms" : "0ms",
              }}
            >
              <h4
                className="text-2xl font-bold mb-2"
                style={{ color: "var(--text-primary)" }}
              >
                Latest Article
              </h4>
              <div
                className="h-1 bg-gradient-to-r from-emerald-600 to-teal-600 rounded-full"
                style={{
                  width: isVisible ? "64px" : "0px",
                  transition: "width 700ms ease-out",
                  transitionDelay: isVisible ? "450ms" : "0ms",
                }}
              ></div>
            </div>

            <Link
              href={featuredArticle.link}
              className="group block"
              aria-label={`Read latest article: ${featuredArticle.title}`}
            >
              <div
                className="rounded-2xl border overflow-hidden transition-all duration-500 hover:scale-[1.01] hover:shadow-2xl hover:shadow-emerald-500/10"
                style={{
                  backgroundColor: "var(--card-bg)",
                  borderColor: "var(--card-border)",
                }}
              >
                <div className="grid lg:grid-cols-5 gap-0">
                  {/* Featured Image (60%) */}
                  <div className="lg:col-span-3 relative h-80 lg:h-96 overflow-hidden">
                    <Image
                      src={
                        featuredArticle.image &&
                        featuredArticle.image.trim() !== "" &&
                        featuredArticle.image !== "/blog-placeholder.jpg"
                          ? featuredArticle.image
                          : "/blog-placeholder.jpg"
                      }
                      alt={`Cover image for ${featuredArticle.title}`}
                      fill
                      sizes="(max-width: 1024px) 100vw, 60vw"
                      style={{ objectFit: "cover" }}
                      className="transition-transform duration-700 group-hover:scale-110"
                      priority
                    />
                    <div className="absolute top-6 left-6">
                      <span className="px-3 py-1 bg-white/90 backdrop-blur-sm text-emerald-600 text-sm font-medium rounded-full">
                        Featured
                      </span>
                    </div>
                  </div>

                  {/* Article Info (40%) */}
                  <div className="lg:col-span-2 p-8 lg:p-10 flex flex-col justify-center">
                    <div className="flex items-center gap-3 mb-4">
                      <time
                        className="text-sm font-medium px-3 py-1 rounded-full"
                        style={{
                          backgroundColor: "var(--surface-accent)",
                          color: "var(--accent)",
                        }}
                        dateTime={featuredArticle.date}
                      >
                        {featuredArticle.date}
                      </time>
                      <div
                        className="flex items-center gap-1 text-sm"
                        style={{ color: "var(--text-muted)" }}
                      >
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
                            d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
                          />
                        </svg>
                        <span>{featuredArticle.readTime}</span>
                      </div>
                    </div>

                    {/* Collection tag */}
                    {featuredArticle.collection && (
                      <div className="mb-4">
                        <CollectionTag
                          collection={featuredArticle.collection}
                          variant="detailed"
                          clickable={false}
                        />
                      </div>
                    )}

                    <h5
                      className="text-2xl lg:text-3xl font-bold mb-4 leading-tight transition-colors group-hover:text-[var(--accent)]"
                      style={{ color: "var(--text-primary)" }}
                    >
                      {featuredArticle.title}
                    </h5>

                    <p
                      className="text-base lg:text-lg mb-6 leading-relaxed transition-colors duration-300"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      {featuredArticle.summary}
                    </p>

                    {featuredArticle.tags &&
                      featuredArticle.tags.length > 0 && (
                        <div className="flex flex-wrap gap-2 mb-6">
                          {featuredArticle.tags
                            .slice(0, 3)
                            .map((tag: string) => (
                              <span
                                key={tag}
                                className="text-xs px-3 py-1 rounded-full transition-colors duration-300"
                                style={{
                                  backgroundColor: "var(--surface)",
                                  color: "var(--text-muted)",
                                  border: "1px solid var(--border)",
                                }}
                              >
                                {tag}
                              </span>
                            ))}
                        </div>
                      )}

                    <div
                      className="inline-flex items-center text-sm font-semibold transition-all duration-300 group-hover:gap-3"
                      style={{ color: "var(--accent)" }}
                    >
                      Read Full Article
                      <svg
                        className="ml-2 w-4 h-4 transition-transform duration-300 group-hover:translate-x-1"
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
                    </div>
                  </div>
                </div>
              </div>
            </Link>
          </div>
        )}

        {/* Recent Articles Grid */}
        {remainingArticles.length > 0 && (
          <div
            style={{
              opacity: isVisible ? 1 : 0,
              transform: isVisible ? "translateY(0)" : "translateY(48px)",
              filter: isVisible ? "blur(0)" : "blur(8px)",
              transition:
                "opacity 1000ms cubic-bezier(0.25, 0.46, 0.45, 0.94), transform 1000ms cubic-bezier(0.25, 0.46, 0.45, 0.94), filter 1000ms cubic-bezier(0.25, 0.46, 0.45, 0.94)",
              transitionDelay: isVisible ? "400ms" : "0ms",
            }}
          >
            <div
              className="mb-8"
              style={{
                opacity: isVisible ? 1 : 0,
                transform: isVisible ? "translateX(0)" : "translateX(-16px)",
                transition: "opacity 700ms ease-out, transform 700ms ease-out",
                transitionDelay: isVisible ? "500ms" : "0ms",
              }}
            >
              <h4
                className="text-2xl font-bold mb-2"
                style={{ color: "var(--text-primary)" }}
              >
                Recent Articles
              </h4>
              <div
                className="h-1 bg-gradient-to-r from-teal-600 to-cyan-600 rounded-full"
                style={{
                  width: isVisible ? "64px" : "0px",
                  transition: "width 700ms ease-out",
                  transitionDelay: isVisible ? "650ms" : "0ms",
                }}
              ></div>
            </div>

            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
              {remainingArticles.map((article: Article, idx: number) => (
                <Link
                  href={article.link}
                  key={`${article.link}-${idx}`}
                  className="group block"
                  style={{
                    opacity: isVisible ? 1 : 0,
                    transform: isVisible
                      ? "translateY(0) scale(1)"
                      : "translateY(40px) scale(0.95)",
                    filter: isVisible ? "blur(0)" : "blur(8px)",
                    transition:
                      "opacity 700ms cubic-bezier(0.25, 0.46, 0.45, 0.94), transform 700ms cubic-bezier(0.25, 0.46, 0.45, 0.94), filter 700ms cubic-bezier(0.25, 0.46, 0.45, 0.94)",
                    transitionDelay: isVisible ? `${700 + idx * 120}ms` : "0ms",
                  }}
                  aria-label={`Read article: ${article.title}`}
                >
                  <article
                    className="h-full rounded-xl border overflow-hidden transition-all duration-300 hover:scale-[1.03] hover:border-[var(--accent)]"
                    style={{
                      backgroundColor: "var(--card-bg)",
                      borderColor: "var(--card-border)",
                    }}
                  >
                    <div className="relative h-48 overflow-hidden">
                      <Image
                        src={
                          article.image &&
                          article.image.trim() !== "" &&
                          article.image !== "/blog-placeholder.jpg"
                            ? article.image
                            : "/blog-placeholder.jpg"
                        }
                        alt={`Cover image for ${article.title}`}
                        fill
                        sizes="(max-width: 768px) 100vw, (max-width: 1024px) 50vw, 33vw"
                        style={{ objectFit: "cover" }}
                        className="transition-transform duration-500 group-hover:scale-110"
                      />
                    </div>

                    <div className="p-6">
                      <div className="flex items-center justify-between mb-3">
                        <time
                          className="text-xs font-medium px-2 py-1 rounded-full"
                          style={{
                            backgroundColor: "var(--surface)",
                            color: "var(--text-muted)",
                          }}
                          dateTime={article.date}
                        >
                          {article.date}
                        </time>
                        <div
                          className="flex items-center gap-1 text-xs"
                          style={{ color: "var(--text-muted)" }}
                        >
                          <svg
                            className="w-3 h-3"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth="2"
                              d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
                            />
                          </svg>
                          <span>{article.readTime}</span>
                        </div>
                      </div>

                      {/* Collection tag for other articles */}
                      {article.collection && (
                        <div className="mb-3">
                          <CollectionTag
                            collection={article.collection}
                            variant="compact"
                            clickable={false}
                          />
                        </div>
                      )}

                      <h5
                        className="text-lg font-bold mb-3 leading-tight transition-colors group-hover:text-[var(--accent)] line-clamp-2"
                        style={{ color: "var(--text-primary)" }}
                      >
                        {article.title}
                      </h5>

                      <p
                        className="text-sm mb-4 leading-relaxed transition-colors duration-300 line-clamp-3"
                        style={{ color: "var(--text-secondary)" }}
                      >
                        {article.summary}
                      </p>

                      {article.tags && article.tags.length > 0 && (
                        <div className="flex flex-wrap gap-1.5 mb-4">
                          {article.tags.slice(0, 2).map((tag: string) => (
                            <span
                              key={tag}
                              className="text-xs px-2 py-1 rounded-full transition-colors duration-300"
                              style={{
                                backgroundColor: "var(--surface-accent)",
                                color: "var(--accent)",
                              }}
                            >
                              {tag}
                            </span>
                          ))}
                        </div>
                      )}

                      <div
                        className="flex items-center text-sm font-medium transition-all duration-300 group-hover:gap-2"
                        style={{ color: "var(--accent)" }}
                      >
                        Read More
                        <svg
                          className="ml-1 w-4 h-4 transition-transform duration-300 group-hover:translate-x-1"
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
                      </div>
                    </div>
                  </article>
                </Link>
              ))}
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
