"use client";

import React, { useState } from "react";
import Link from "next/link";
import { getArticlesByCategory, Article } from "@/data/articles";

export default function CryptoBlogPage() {
  const [selectedCategory, setSelectedCategory] = useState("all");
  const [searchTerm, setSearchTerm] = useState("");

  // Get articles from centralized data system
  const cryptoArticles = getArticlesByCategory("crypto");

  // Additional placeholder articles for display
  const additionalArticles: Article[] = [
    {
      id: "bitcoin-ethereum-comparison",
      title: "Bitcoin vs Ethereum: Technical Analysis",
      excerpt:
        "Comprehensive comparison of Bitcoin and Ethereum architectures, consensus mechanisms, and ecosystem differences.",
      content: "",
      category: "crypto",
      tags: ["Bitcoin", "Ethereum", "Analysis"],
      date: "2024-03-05",
      readTime: "15 min read",
      difficulty: "Advanced",
      slug: "bitcoin-ethereum-comparison",
      subcategory: "Technology",
      featured: false,
    },
    {
      id: "nft-revolution",
      title: "NFTs and Digital Ownership",
      excerpt:
        "Understanding Non-Fungible Tokens and their impact on digital ownership and creative economies.",
      content: "",
      category: "crypto",
      tags: ["NFT", "Digital Rights", "Blockchain"],
      date: "2024-02-28",
      readTime: "10 min read",
      difficulty: "Intermediate",
      slug: "nft-revolution",
      subcategory: "NFTs",
      featured: false,
    },
  ];

  // Combine articles
  const allArticles = [...cryptoArticles, ...additionalArticles];

  const categories = [
    { name: "All", slug: "all", count: allArticles.length },
    {
      name: "DeFi",
      slug: "DeFi",
      count: allArticles.filter((a) => a.subcategory === "DeFi").length,
    },
    {
      name: "Fundamentals",
      slug: "Fundamentals",
      count: allArticles.filter((a) => a.subcategory === "Fundamentals").length,
    },
    {
      name: "Technology",
      slug: "Technology",
      count: allArticles.filter((a) => a.subcategory === "Technology").length,
    },
    {
      name: "NFTs",
      slug: "NFTs",
      count: allArticles.filter((a) => a.subcategory === "NFTs").length,
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

  const featuredArticles = allArticles.filter((article) =>
    cryptoArticles.some((ca) => ca.id === article.id && ca.featured)
  );

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
              Exploring blockchain technology, DeFi protocols, and the future of
              decentralized finance. From fundamentals to advanced trading
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

          {/* Stats Overview */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-16">
            <div
              className="rounded-lg p-6 text-center border"
              style={{
                backgroundColor: "var(--surface)",
                borderColor: "var(--border)",
              }}
            >
              <div
                className="w-12 h-12 rounded-full mx-auto mb-4 flex items-center justify-center"
                style={{ backgroundColor: "var(--accent)" }}
              >
                <svg
                  className="w-6 h-6 text-white"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                  />
                </svg>
              </div>
              <div
                className="text-2xl font-bold mb-2"
                style={{ color: "var(--text-primary)" }}
              >
                {allArticles.length}
              </div>
              <div
                className="text-sm"
                style={{ color: "var(--text-secondary)" }}
              >
                Crypto Articles
              </div>
            </div>
            <div
              className="rounded-lg p-6 text-center border"
              style={{
                backgroundColor: "var(--surface)",
                borderColor: "var(--border)",
              }}
            >
              <div
                className="w-12 h-12 rounded-full mx-auto mb-4 flex items-center justify-center"
                style={{ backgroundColor: "var(--accent)" }}
              >
                <svg
                  className="w-6 h-6 text-white"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.99 1.99 0 013 12V7a4 4 0 014-4z"
                  />
                </svg>
              </div>
              <div
                className="text-2xl font-bold mb-2"
                style={{ color: "var(--text-primary)" }}
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
              className="rounded-lg p-6 text-center border"
              style={{
                backgroundColor: "var(--surface)",
                borderColor: "var(--border)",
              }}
            >
              <div
                className="w-12 h-12 rounded-full mx-auto mb-4 flex items-center justify-center"
                style={{ backgroundColor: "var(--accent)" }}
              >
                <svg
                  className="w-6 h-6 text-white"
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
              </div>
              <div
                className="text-2xl font-bold mb-2"
                style={{ color: "var(--text-primary)" }}
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
                Avg Read Time (min)
              </div>
            </div>
          </div>

          {/* Search and Filter Section */}
          <div className="mb-12">
            <div className="max-w-2xl mx-auto">
              <div className="relative mb-8">
                <input
                  type="text"
                  placeholder="Search articles..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-full px-4 py-3 pl-12 rounded-lg border transition-colors duration-200 focus:ring-2 focus:ring-[var(--accent)] focus:border-transparent"
                  style={{
                    borderColor: "var(--border)",
                    backgroundColor: "var(--surface)",
                    color: "var(--text-primary)",
                  }}
                />
                <div className="absolute left-4 top-1/2 transform -translate-y-1/2">
                  <svg
                    className="w-5 h-5"
                    style={{ color: "var(--text-secondary)" }}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                    />
                  </svg>
                </div>
              </div>

              {/* Category Filter */}
              <div className="flex flex-wrap justify-center gap-2 mb-8">
                {categories.map((category) => (
                  <button
                    key={category.slug}
                    onClick={() => setSelectedCategory(category.slug)}
                    className={`px-4 py-2 rounded-full text-sm font-medium transition-all duration-200 border ${
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
          </div>

          {/* Featured Articles */}
          {featuredArticles.length > 0 && (
            <div className="mb-12">
              <h2
                className="text-3xl font-bold text-center mb-8"
                style={{ color: "var(--text-primary)" }}
              >
                Featured Articles
              </h2>
              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
                {featuredArticles.map((article) => (
                  <div
                    key={article.id}
                    className="rounded-lg p-6 shadow-lg hover:shadow-xl transition-shadow border-l-4"
                    style={{
                      backgroundColor: "var(--surface)",
                      borderLeftColor: "var(--accent)",
                    }}
                  >
                    <div className="flex items-center justify-between mb-3">
                      <span
                        className="text-xs font-medium px-2 py-1 rounded"
                        style={{
                          backgroundColor: "var(--accent)",
                          color: "white",
                        }}
                      >
                        Featured
                      </span>
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
                    <h3
                      className="text-xl font-semibold mb-2"
                      style={{ color: "var(--text-primary)" }}
                    >
                      {article.title}
                    </h3>
                    <p
                      className="mb-4"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      {article.excerpt}
                    </p>
                    <div
                      className="flex items-center justify-between text-sm"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      <span>{article.date}</span>
                      <span>{article.readTime}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Results Summary */}
          <div className="text-center mb-8">
            <p style={{ color: "var(--text-secondary)" }}>
              {searchTerm || selectedCategory !== "all"
                ? `Found ${filteredArticles.length} articles`
                : `${allArticles.length} total crypto articles`}
            </p>
          </div>

          {/* Article List */}
          <div className="space-y-6">
            {filteredArticles.length > 0 ? (
              filteredArticles.map((article) => (
                <div
                  key={article.id}
                  className="rounded-lg p-6 shadow-md hover:shadow-lg transition-shadow border"
                  style={{
                    backgroundColor: "var(--surface)",
                    borderColor: "var(--border)",
                  }}
                >
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <span
                        className="text-xs font-medium px-2 py-1 rounded"
                        style={{
                          backgroundColor: "var(--surface)",
                          color: "var(--text-secondary)",
                          border: "1px solid var(--border)",
                        }}
                      >
                        {article.subcategory}
                      </span>
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
                    <span
                      className="text-sm"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      {article.readTime}
                    </span>
                  </div>
                  <h3
                    className="text-xl font-semibold mb-2 transition-colors cursor-pointer hover:opacity-80"
                    style={{ color: "var(--text-primary)" }}
                  >
                    {article.title}
                  </h3>
                  <p
                    className="mb-4 leading-relaxed"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    {article.excerpt}
                  </p>
                  <div className="flex items-center justify-between">
                    <div className="flex flex-wrap gap-2">
                      {article.tags.slice(0, 3).map((tag) => (
                        <span
                          key={tag}
                          className="text-xs px-2 py-1 rounded"
                          style={{
                            backgroundColor: `var(--accent)20`,
                            color: "var(--accent)",
                          }}
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                    <span
                      className="text-sm"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      {new Date(article.date).toLocaleDateString("en-US", {
                        year: "numeric",
                        month: "short",
                        day: "numeric",
                      })}
                    </span>
                  </div>
                </div>
              ))
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
