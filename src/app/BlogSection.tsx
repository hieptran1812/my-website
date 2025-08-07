"use client";

import React, { useEffect, useState, useCallback } from "react";
import Image from "next/image";
import Link from "next/link";
import { BlogPostMetadata } from "../lib/blog";
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
}

export default function BlogSection() {
  const [articles, setArticles] = useState<Article[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

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
        .map((post) => ({
          title: post.title || "Untitled",
          summary: post.excerpt || "No summary available",
          link: `/blog/${post.slug}`,
          image: post.image || "/blog-placeholder.jpg",
          date: formatDateMedium(post.publishDate || ""),
          tags: Array.isArray(post.tags) ? post.tags : [],
        }))
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
            <div className="absolute -bottom-2 left-1/2 transform -translate-x-1/2 w-24 h-1 bg-gradient-to-r from-emerald-600 via-teal-600 to-cyan-600 rounded-full"></div>
          </h3>
          <p
            className="text-lg max-w-2xl mx-auto transition-colors duration-300 mt-6"
            style={{ color: "var(--text-secondary)" }}
          >
            Insights, tutorials, and thoughts on AI, software engineering, and
            technology trends.
          </p>
          <Link
            href="/blog"
            className="inline-flex items-center mt-4 font-medium transition-colors"
            style={{ color: "var(--accent)" }}
            onMouseEnter={(e) => {
              e.currentTarget.style.color = "var(--accent-hover)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.color = "var(--accent)";
            }}
          >
            View All Posts
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
        <div className="grid md:grid-cols-2 gap-6">
          {articles.map((article: Article, idx: number) => (
            <Link
              href={article.link}
              key={`${article.link}-${idx}`}
              className="group block card-enhanced rounded-xl overflow-hidden hover:border-[var(--accent)] transition-all duration-300"
              style={{
                backgroundColor: "var(--card-bg)",
                borderColor: "var(--card-border)",
              }}
              aria-label={`Read article: ${article.title}`}
            >
              <div className="relative w-full h-44 overflow-hidden">
                <Image
                  src={
                    article.image &&
                    article.image.trim() !== "" &&
                    article.image !== "/blog-placeholder.jpg" &&
                    article.image !== "/images/default-blog.jpg"
                      ? article.image
                      : "/blog-placeholder.jpg"
                  }
                  alt={`Cover image for ${article.title}`}
                  fill
                  sizes="(max-width: 768px) 100vw, 50vw"
                  style={{ objectFit: "cover" }}
                  className="transition-transform duration-300 group-hover:scale-105"
                  priority={idx < 2} // Prioritize loading first 2 images
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent"></div>
              </div>
              <div className="p-5">
                <div className="flex items-center gap-2 mb-2">
                  <time
                    className="text-xs px-2 py-1 rounded-full transition-colors duration-300"
                    style={{ color: "var(--text-muted)" }}
                    dateTime={article.date}
                  >
                    {article.date}
                  </time>
                </div>
                <h4
                  className="text-lg font-semibold mb-2 transition-colors leading-tight line-clamp-2 group-hover:text-[var(--accent)]"
                  style={{ color: "var(--text-primary)" }}
                >
                  {article.title}
                </h4>
                <p
                  className="text-sm mb-3 leading-relaxed transition-colors duration-300 line-clamp-2"
                  style={{ color: "var(--text-secondary)" }}
                >
                  {article.summary}
                </p>
                <div className="flex flex-wrap gap-1.5 mt-auto" role="list">
                  {article.tags?.map((tag: string) => (
                    <span
                      key={tag}
                      className="text-xs px-2 py-1 rounded-full transition-colors duration-300"
                      style={{
                        backgroundColor: "var(--surface-accent)",
                        color: "var(--accent)",
                      }}
                      role="listitem"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
            </Link>
          ))}
        </div>
      </div>
    </section>
  );
}
