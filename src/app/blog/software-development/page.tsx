"use client";

import React, { useState } from "react";
import Link from "next/link";
import { getArticlesByCategory, Article } from "@/data/articles";

export default function SoftwareDevelopmentBlogPage() {
  const [selectedCategory, setSelectedCategory] = useState("all");
  const [searchTerm, setSearchTerm] = useState("");

  // Get articles from centralized data system
  const devArticles = getArticlesByCategory("software-development");

  // Additional placeholder articles for display
  const additionalArticles: Article[] = [
    {
      id: "software-development-best-practices",
      title: "Software Development Best Practices: A Comprehensive Guide",
      excerpt:
        "Essential best practices for modern software development, covering code quality, testing, and maintainability.",
      content: "",
      category: "software-development",
      tags: ["Best Practices", "Code Quality", "Testing"],
      date: "2024-03-25",
      readTime: "20 min read",
      difficulty: "Intermediate",
      slug: "software-development-best-practices",
      subcategory: "Best Practices",
      featured: true,
    },
    {
      id: "microservices-nodejs-docker",
      title: "Building Microservices with Node.js and Docker",
      excerpt:
        "Learn how to design, develop, and deploy scalable microservices using Node.js and Docker containers.",
      content: "",
      category: "software-development",
      tags: ["Microservices", "Node.js", "Docker"],
      date: "2024-03-22",
      readTime: "25 min read",
      difficulty: "Advanced",
      slug: "microservices-nodejs-docker",
      subcategory: "Architecture",
      featured: false,
    },
    {
      id: "system-design-scalability-patterns",
      title: "System Design: Scalability Patterns and Best Practices",
      excerpt:
        "Comprehensive guide to designing scalable systems, covering load balancing, caching, and distributed architectures.",
      content: "",
      category: "software-development",
      tags: ["System Design", "Scalability", "Architecture"],
      date: "2024-03-20",
      readTime: "30 min read",
      difficulty: "Advanced",
      slug: "system-design-scalability-patterns",
      subcategory: "System Design",
      featured: false,
    },
  ];

  // Combine articles
  const allArticles = [...devArticles, ...additionalArticles];

  const categories = [
    { name: "All", slug: "all", count: allArticles.length },
    {
      name: "Best Practices",
      slug: "Best Practices",
      count: allArticles.filter((a) => a.subcategory === "Best Practices")
        .length,
    },
    {
      name: "Architecture",
      slug: "Architecture",
      count: allArticles.filter((a) => a.subcategory === "Architecture").length,
    },
    {
      name: "System Design",
      slug: "System Design",
      count: allArticles.filter((a) => a.subcategory === "System Design")
        .length,
    },
    {
      name: "DevOps",
      slug: "DevOps",
      count: allArticles.filter((a) => a.subcategory === "DevOps").length,
    },
  ];

  const filteredArticles = allArticles.filter((article) => {
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

  const featuredArticles = allArticles.filter((article) => article.featured);

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case "Beginner":
        return "#22c55e";
      case "Intermediate":
        return "#f59e0b";
      case "Advanced":
        return "#ef4444";
      default:
        return "var(--text-secondary)";
    }
  };

  return (
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
                💻
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
                Software Development
              </h1>
            </div>
            <p
              className="text-xl max-w-3xl mx-auto leading-relaxed mb-8"
              style={{ color: "var(--text-secondary)" }}
            >
              Comprehensive guides on software engineering practices, system
              design, architecture patterns, and development methodologies for
              building robust applications.
            </p>
            <div className="flex flex-wrap justify-center gap-2">
              {[
                "System Design",
                "Architecture",
                "Best Practices",
                "DevOps",
                "Microservices",
                "Scalability",
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

          {/* Development Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-16">
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
                Topics Covered
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
                {allArticles.length > 0
                  ? Math.round(
                      allArticles.reduce(
                        (acc, article) => acc + parseInt(article.readTime),
                        0
                      ) / allArticles.length
                    )
                  : 0}
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
                          Difficulty
                        </div>
                        <span
                          className="px-3 py-1 text-sm font-medium rounded-full text-white"
                          style={{
                            backgroundColor: getDifficultyColor(
                              article.difficulty
                            ),
                          }}
                        >
                          {article.difficulty}
                        </span>
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
                        <span>💻 Development</span>
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

          {/* Development Topics */}
          <div className="mb-12">
            <h2
              className="text-2xl font-bold mb-6"
              style={{ color: "var(--text-primary)" }}
            >
              Development Topics
            </h2>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-8">
              {[
                "Best Practices",
                "Architecture",
                "System Design",
                "DevOps",
                "Scalability",
              ].map((topic) => (
                <button
                  key={topic}
                  onClick={() => setSelectedCategory(topic)}
                  className="p-4 rounded-lg border transition-all duration-200 hover:shadow-md hover:scale-105 text-center"
                  style={{
                    backgroundColor: "var(--surface)",
                    borderColor: "var(--border)",
                    color: "var(--text-primary)",
                  }}
                >
                  <div className="font-medium text-sm">{topic}</div>
                </button>
              ))}
            </div>

            {/* Results Summary */}
            <div className="text-center mb-4">
              <p style={{ color: "var(--text-secondary)" }}>
                {selectedCategory !== "all"
                  ? `Showing ${filteredArticles.length} articles in ${
                      categories.find((c) => c.slug === selectedCategory)
                        ?.name || selectedCategory
                    }`
                  : `${allArticles.length} total development articles`}
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
                All Development Articles
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
                    {category.name}
                  </button>
                ))}
              </div>
            </div>
            {filteredArticles.length > 0 ? (
              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
                {filteredArticles.map((article) => (
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

                      <div
                        className="text-sm mb-3"
                        style={{ color: "var(--text-secondary)" }}
                      >
                        <span
                          className="px-2 py-1 rounded text-xs"
                          style={{
                            color: getDifficultyColor(article.difficulty),
                            backgroundColor: `${getDifficultyColor(
                              article.difficulty
                            )}20`,
                          }}
                        >
                          {article.difficulty}
                        </span>
                      </div>

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
                          {new Date(article.date).toLocaleDateString("en-US", {
                            year: "numeric",
                            month: "short",
                            day: "numeric",
                          })}
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
                <p className="mb-4" style={{ color: "var(--text-secondary)" }}>
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

          {/* Development Resources */}
          <div
            className="p-8 rounded-2xl border mb-12"
            style={{
              backgroundColor: "var(--surface)",
              borderColor: "var(--border)",
            }}
          >
            <h3
              className="text-xl font-bold mb-4"
              style={{ color: "var(--text-primary)" }}
            >
              🛠️ Development Resources
            </h3>
            <p className="mb-6" style={{ color: "var(--text-secondary)" }}>
              Essential tools and resources for modern software development:
            </p>
            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
              {[
                {
                  name: "System Design Primer",
                  desc: "Learn how to design large-scale systems",
                  icon: "🏗️",
                },
                {
                  name: "Clean Code",
                  desc: "Best practices for writing maintainable code",
                  icon: "✨",
                },
                {
                  name: "Design Patterns",
                  desc: "Common software design patterns",
                  icon: "🎨",
                },
                {
                  name: "DevOps Practices",
                  desc: "CI/CD and deployment strategies",
                  icon: "🚀",
                },
              ].map((resource) => (
                <a
                  key={resource.name}
                  href="#"
                  className="flex items-center gap-3 p-4 rounded-lg border transition-colors hover:shadow-md"
                  style={{
                    backgroundColor: "var(--card-bg)",
                    borderColor: "var(--card-border)",
                  }}
                >
                  <div className="text-2xl">{resource.icon}</div>
                  <div>
                    <div
                      className="font-medium"
                      style={{ color: "var(--text-primary)" }}
                    >
                      {resource.name}
                    </div>
                    <div
                      className="text-xs"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      {resource.desc}
                    </div>
                  </div>
                </a>
              ))}
            </div>
          </div>

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
  );
}
