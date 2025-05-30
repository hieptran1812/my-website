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
        return "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300";
      case "Intermediate":
        return "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-300";
      case "Advanced":
        return "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300";
      default:
        return "bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-300";
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
                  background: "linear-gradient(135deg, var(--accent), var(--accent-hover))",
                }}
              >
                ₿
              </div>
              <h1
                className="text-4xl md:text-5xl font-bold"
                style={{
                  background: "linear-gradient(135deg, var(--accent), var(--accent-hover))",
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
              decentralized finance. From fundamentals to advanced trading strategies.
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

      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Search and Filter Section */}
        <div className="mb-12">
          <div className="max-w-2xl mx-auto">
            <div className="relative mb-8">
              <input
                type="text"
                placeholder="Search articles..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full px-4 py-3 pl-12 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-2 focus:ring-orange-500 focus:border-transparent"
              />
              <div className="absolute left-4 top-1/2 transform -translate-y-1/2">
                <svg
                  className="w-5 h-5 text-gray-400"
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
                  className={`px-4 py-2 rounded-full text-sm font-medium transition-all duration-200 ${
                    selectedCategory === category.slug
                      ? "bg-orange-500 text-white"
                      : "bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-orange-100 dark:hover:bg-gray-600"
                  }`}
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
            <h2 className="text-3xl font-bold text-center text-gray-900 dark:text-white mb-8">
              Featured Articles
            </h2>
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
              {featuredArticles.map((article) => (
                <div
                  key={article.id}
                  className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg hover:shadow-xl transition-shadow border-l-4 border-orange-500"
                >
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-xs font-medium px-2 py-1 bg-orange-100 dark:bg-orange-900 text-orange-800 dark:text-orange-300 rounded">
                      Featured
                    </span>
                    <span
                      className={`px-2 py-1 rounded text-xs ${getDifficultyColor(
                        article.difficulty
                      )}`}
                    >
                      {article.difficulty}
                    </span>
                  </div>
                  <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
                    {article.title}
                  </h3>
                  <p className="text-gray-600 dark:text-gray-300 mb-4">
                    {article.excerpt}
                  </p>
                  <div className="flex items-center justify-between text-sm text-gray-500 dark:text-gray-400">
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
          <p className="text-gray-600 dark:text-gray-400">
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
                className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-md hover:shadow-lg transition-shadow"
              >
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center gap-3">
                    <span className="text-xs font-medium px-2 py-1 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded">
                      {article.subcategory}
                    </span>
                    <span
                      className={`px-2 py-1 rounded text-xs ${getDifficultyColor(
                        article.difficulty
                      )}`}
                    >
                      {article.difficulty}
                    </span>
                  </div>
                  <span className="text-sm text-gray-500 dark:text-gray-400">
                    {article.readTime}
                  </span>
                </div>
                <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2 hover:text-orange-600 dark:hover:text-orange-400 transition-colors">
                  {article.title}
                </h3>
                <p className="text-gray-600 dark:text-gray-300 mb-4 leading-relaxed">
                  {article.excerpt}
                </p>
                <div className="flex items-center justify-between">
                  <div className="flex flex-wrap gap-2">
                    {article.tags.slice(0, 3).map((tag) => (
                      <span
                        key={tag}
                        className="text-xs px-2 py-1 bg-orange-50 dark:bg-orange-900/20 text-orange-600 dark:text-orange-400 rounded"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                  <span className="text-sm text-gray-500 dark:text-gray-400">
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
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
                No articles found
              </h3>
              <p className="text-gray-600 dark:text-gray-400 mb-4">
                {searchTerm
                  ? `No articles match "${searchTerm}"`
                  : `No articles in "${
                      categories.find((c) => c.slug === selectedCategory)?.name
                    }" category`}
              </p>
              <button
                onClick={() => {
                  setSearchTerm("");
                  setSelectedCategory("all");
                }}
                className="px-4 py-2 bg-orange-500 text-white rounded-lg hover:bg-orange-600 transition-colors"
              >
                Clear filters
              </button>
            </div>
          )}
        </div>

        {/* Navigation */}
        <nav className="mt-12 pt-8 border-t border-gray-200 dark:border-gray-700">
          <div className="flex justify-center">
            <Link
              href="/blog"
              className="inline-flex items-center px-6 py-3 bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-white rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
            >
              ← Back to All Blogs
            </Link>
          </div>
        </nav>
      </div>
    </div>
  );
}
