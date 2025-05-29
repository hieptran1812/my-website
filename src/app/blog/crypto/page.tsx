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
    <div className="min-h-screen bg-gradient-to-br from-orange-50 via-yellow-50 to-amber-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900">
      {/* Header */}
      <div className="bg-gradient-to-r from-orange-500 to-yellow-500 text-white">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
          <div className="text-center">
            <div className="flex justify-center mb-6">
              <div className="bg-white/20 backdrop-blur-sm rounded-full p-4">
                <span className="text-6xl">₿</span>
              </div>
            </div>
            <h1 className="text-4xl md:text-6xl font-bold mb-6">
              Cryptocurrency Blog
            </h1>
            <p className="text-xl md:text-2xl text-orange-100 max-w-3xl mx-auto">
              Exploring blockchain technology, DeFi protocols, and the future of
              decentralized finance
            </p>
            <div className="mt-8 flex flex-wrap justify-center gap-4 text-sm">
              <div className="bg-white/20 backdrop-blur-sm rounded-full px-4 py-2">
                <span className="font-semibold">{allArticles.length}</span>{" "}
                Articles
              </div>
              <div className="bg-white/20 backdrop-blur-sm rounded-full px-4 py-2">
                <span className="font-semibold">{categories.length - 1}</span>{" "}
                Categories
              </div>
              <div className="bg-white/20 backdrop-blur-sm rounded-full px-4 py-2">
                Weekly Updates
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <h1 className="text-4xl font-bold text-center text-gray-900 dark:text-white mb-8">
          Cryptocurrency Blog
        </h1>
        <p className="text-lg text-center text-gray-600 dark:text-gray-300 mb-8">
          Exploring blockchain technology, DeFi protocols, and the future of
          decentralized finance
        </p>

        {/* Display article count */}
        <div className="text-center mb-8">
          <p className="text-gray-600 dark:text-gray-400">
            Found {allArticles.length} crypto articles
          </p>
        </div>

        {/* Basic article list */}
        <div className="space-y-4">
          {filteredArticles.map((article) => (
            <div
              key={article.id}
              className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-md"
            >
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
                {article.title}
              </h3>
              <p className="text-gray-600 dark:text-gray-300 mb-4">
                {article.excerpt}
              </p>
              <div className="flex items-center justify-between text-sm text-gray-500 dark:text-gray-400">
                <span>{article.date}</span>
                <span>{article.readTime}</span>
                <span
                  className={`px-2 py-1 rounded text-xs ${getDifficultyColor(
                    article.difficulty
                  )}`}
                >
                  {article.difficulty}
                </span>
              </div>
            </div>
          ))}
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
