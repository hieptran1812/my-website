"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import FadeInWrapper from "@/components/FadeInWrapper";
import ArticleCard from "@/components/ArticleCard";

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
  collection?: string;
}

interface Article {
  id: string;
  title: string;
  excerpt: string;
  content: string;
  category: string;
  subcategory?: string;
  tags: string[];
  date: string;
  readTime: string;
  difficulty: "Beginner" | "Intermediate" | "Advanced";
  slug: string;
  featured: boolean;
  author?: string;
  image?: string;
  collection?: string;
}

// Convert BlogPostMetadata to Article format
function convertToArticle(post: BlogPostMetadata): Article {
  return {
    id: post.slug,
    title: post.title,
    excerpt: post.excerpt,
    content: "",
    category: post.category,
    subcategory: post.category,
    tags: post.tags || [],
    date: post.publishDate,
    readTime: post.readTime,
    difficulty: "Intermediate" as const,
    slug: post.slug,
    featured: false,
    author: post.author,
    image: post.image,
    collection: post.collection,
  };
}

// Convert slug back to collection name for display
function slugToCollectionName(slug: string): string {
  return slug
    .split("-")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

// Get collection icon
function getCollectionIcon(collection: string): string {
  if (
    collection.toLowerCase().includes("finance") ||
    collection.toLowerCase().includes("trading")
  ) {
    return "💰";
  }
  if (
    collection.toLowerCase().includes("machine learning") ||
    collection.toLowerCase().includes("ai")
  ) {
    return "🤖";
  }
  if (
    collection.toLowerCase().includes("software") ||
    collection.toLowerCase().includes("development")
  ) {
    return "💻";
  }
  if (
    collection.toLowerCase().includes("paper") ||
    collection.toLowerCase().includes("research")
  ) {
    return "📚";
  }
  if (collection.toLowerCase().includes("notes")) {
    return "📝";
  }
  return "📖";
}

export default function CollectionPage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const [articles, setArticles] = useState<Article[]>([]);
  const [loading, setLoading] = useState(true);
  const [sortOrder, setSortOrder] = useState<"newest" | "oldest">("newest");
  const [collectionSlug, setCollectionSlug] = useState<string>("");

  // Resolve params
  useEffect(() => {
    async function resolveParams() {
      const resolvedParams = await params;
      setCollectionSlug(resolvedParams.slug);
    }
    resolveParams();
  }, [params]);

  const collectionName = slugToCollectionName(collectionSlug);

  useEffect(() => {
    if (!collectionSlug) return; // Wait for slug to be resolved

    const fetchCollectionArticles = async () => {
      try {
        // Fetch all blog posts
        const response = await fetch("/api/blog/posts");
        const posts: BlogPostMetadata[] = await response.json();

        // Filter posts by collection (convert collection to slug for comparison)
        const filteredPosts = posts.filter((post) => {
          if (!post.collection) return false;
          const postCollectionSlug = post.collection
            .toLowerCase()
            .replace(/[^a-z0-9\s-]/g, "")
            .replace(/\s+/g, "-")
            .trim();

          return postCollectionSlug === collectionSlug;
        });

        // Convert to Article format
        const articlesData = filteredPosts.map(convertToArticle);

        // Sort articles
        const sortedArticles = articlesData.sort((a, b) => {
          const dateA = new Date(a.date);
          const dateB = new Date(b.date);
          return sortOrder === "newest"
            ? dateB.getTime() - dateA.getTime()
            : dateA.getTime() - dateB.getTime();
        });

        setArticles(sortedArticles);
      } catch (error) {
        console.error("Error fetching collection articles:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchCollectionArticles();
  }, [collectionSlug, sortOrder]);

  const handleSortChange = (newSortOrder: "newest" | "oldest") => {
    setSortOrder(newSortOrder);
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="w-8 h-8 border-4 border-[var(--accent)] border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p style={{ color: "var(--text-secondary)" }}>
            Loading collection...
          </p>
        </div>
      </div>
    );
  }

  return (
    <div
      className="min-h-screen"
      style={{ backgroundColor: "var(--background)" }}
    >
      {/* Hero Section */}
      <section className="py-16 px-6">
        <div className="max-w-6xl mx-auto">
          <FadeInWrapper delay={0} duration={800} direction="up">
            {/* Breadcrumb */}
            <div className="mb-8">
              <nav
                className="flex items-center gap-2 text-sm"
                style={{ color: "var(--text-secondary)" }}
              >
                <Link
                  href="/"
                  className="hover:text-[var(--accent)] transition-colors"
                >
                  Home
                </Link>
                <span>•</span>
                <Link
                  href="/blog"
                  className="hover:text-[var(--accent)] transition-colors"
                >
                  Blog
                </Link>
                <span>•</span>
                <span style={{ color: "var(--text-primary)" }}>
                  Collections
                </span>
                <span>•</span>
                <span style={{ color: "var(--accent)" }}>{collectionName}</span>
              </nav>
            </div>

            {/* Collection Header */}
            <div className="text-center mb-12">
              <div className="inline-flex items-center gap-3 mb-4">
                <span
                  className="text-4xl"
                  role="img"
                  aria-label="Collection icon"
                >
                  {getCollectionIcon(collectionName)}
                </span>
                <h1
                  className="text-4xl md:text-5xl font-bold"
                  style={{ color: "var(--text-primary)" }}
                >
                  {collectionName}
                </h1>
              </div>

              <p
                className="text-lg md:text-xl mb-6 max-w-3xl mx-auto leading-relaxed"
                style={{ color: "var(--text-secondary)" }}
              >
                A curated collection of articles covering various topics and
                insights. Explore {articles.length} article
                {articles.length !== 1 ? "s" : ""} in this collection.
              </p>

              {/* Collection Stats */}
              <div className="flex flex-wrap justify-center gap-8 mb-8">
                <div className="text-center">
                  <div
                    className="text-2xl font-bold"
                    style={{ color: "var(--accent)" }}
                  >
                    {articles.length}
                  </div>
                  <div
                    className="text-sm"
                    style={{ color: "var(--text-muted)" }}
                  >
                    Articles
                  </div>
                </div>
                <div className="text-center">
                  <div
                    className="text-2xl font-bold"
                    style={{ color: "var(--accent)" }}
                  >
                    {articles.reduce((total, article) => {
                      const readTimeNum =
                        parseInt(article.readTime.split(" ")[0]) || 5;
                      return total + readTimeNum;
                    }, 0)}
                  </div>
                  <div
                    className="text-sm"
                    style={{ color: "var(--text-muted)" }}
                  >
                    Total Read Time (min)
                  </div>
                </div>
                <div className="text-center">
                  <div
                    className="text-2xl font-bold"
                    style={{ color: "var(--accent)" }}
                  >
                    {new Set(articles.flatMap((article) => article.tags)).size}
                  </div>
                  <div
                    className="text-sm"
                    style={{ color: "var(--text-muted)" }}
                  >
                    Unique Tags
                  </div>
                </div>
              </div>
            </div>
          </FadeInWrapper>
        </div>
      </section>

      {/* Articles Section */}
      <section className="pb-16 px-6">
        <div className="max-w-6xl mx-auto">
          {/* Sort Controls */}
          {articles.length > 0 && (
            <FadeInWrapper delay={200} duration={600} direction="up">
              <div className="flex items-center justify-between mb-8">
                <h2
                  className="text-2xl font-bold"
                  style={{ color: "var(--text-primary)" }}
                >
                  All Articles ({articles.length})
                </h2>

                <div className="flex items-center gap-4">
                  <span
                    className="text-sm font-medium"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    Sort by:
                  </span>
                  <div
                    className="flex rounded-lg overflow-hidden border"
                    style={{ borderColor: "var(--border)" }}
                  >
                    <button
                      onClick={() => handleSortChange("newest")}
                      className={`px-4 py-2 text-sm font-medium transition-all ${
                        sortOrder === "newest"
                          ? "text-white"
                          : "hover:bg-[var(--surface-hover)]"
                      }`}
                      style={{
                        backgroundColor:
                          sortOrder === "newest"
                            ? "var(--accent)"
                            : "var(--surface)",
                        color:
                          sortOrder === "newest"
                            ? "white"
                            : "var(--text-primary)",
                      }}
                    >
                      Newest First
                    </button>
                    <button
                      onClick={() => handleSortChange("oldest")}
                      className={`px-4 py-2 text-sm font-medium transition-all border-l ${
                        sortOrder === "oldest"
                          ? "text-white"
                          : "hover:bg-[var(--surface-hover)]"
                      }`}
                      style={{
                        backgroundColor:
                          sortOrder === "oldest"
                            ? "var(--accent)"
                            : "var(--surface)",
                        color:
                          sortOrder === "oldest"
                            ? "white"
                            : "var(--text-primary)",
                        borderLeftColor: "var(--border)",
                      }}
                    >
                      Oldest First
                    </button>
                  </div>
                </div>
              </div>
            </FadeInWrapper>
          )}

          {/* Articles Grid */}
          {articles.length > 0 ? (
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
              {articles.map((article, index) => (
                <ArticleCard
                  key={article.id}
                  article={article}
                  index={index}
                  variant="default"
                />
              ))}
            </div>
          ) : (
            <FadeInWrapper delay={200} duration={600} direction="up">
              <div className="text-center py-16">
                <div className="text-6xl mb-4">📚</div>
                <h3
                  className="text-xl font-semibold mb-2"
                  style={{ color: "var(--text-primary)" }}
                >
                  No Articles Found
                </h3>
                <p
                  className="text-base mb-6"
                  style={{ color: "var(--text-secondary)" }}
                >
                  This collection doesn&apos;t have any articles yet, or the
                  collection name might have changed.
                </p>
                <Link
                  href="/blog"
                  className="inline-flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-all duration-200 hover:scale-105"
                  style={{
                    backgroundColor: "var(--accent)",
                    color: "white",
                  }}
                >
                  Browse All Articles
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
            </FadeInWrapper>
          )}
        </div>
      </section>
    </div>
  );
}
