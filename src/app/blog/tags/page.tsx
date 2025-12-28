"use client";

import React, { useState, useEffect, useMemo } from "react";
import Link from "next/link";
import FadeInWrapper from "@/components/FadeInWrapper";
import TagBadge from "@/components/TagBadge";

interface TagInfo {
  tag: string;
  slug: string;
  count: number;
  categories: string[];
}

type SortOption = "count" | "alphabetical" | "recent";

export default function TagsPage() {
  const [tags, setTags] = useState<TagInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState("");
  const [sortBy, setSortBy] = useState<SortOption>("count");
  const [selectedCategory, setSelectedCategory] = useState<string>("all");

  useEffect(() => {
    async function fetchTags() {
      try {
        const response = await fetch("/api/blog/tags");
        const data: TagInfo[] = await response.json();
        setTags(data);
      } catch (error) {
        console.error("Error fetching tags:", error);
        setTags([]);
      } finally {
        setLoading(false);
      }
    }

    fetchTags();
  }, []);

  // Get unique categories from all tags
  const categories = useMemo(() => {
    const categorySet = new Set<string>();
    tags.forEach((tag) => {
      tag.categories.forEach((cat) => categorySet.add(cat));
    });
    return Array.from(categorySet).sort();
  }, [tags]);

  // Filter and sort tags
  const filteredTags = useMemo(() => {
    let result = [...tags];

    // Filter by search term
    if (searchTerm) {
      result = result.filter((tag) =>
        tag.tag.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    // Filter by category
    if (selectedCategory !== "all") {
      result = result.filter((tag) =>
        tag.categories.includes(selectedCategory)
      );
    }

    // Sort
    switch (sortBy) {
      case "alphabetical":
        result.sort((a, b) => a.tag.localeCompare(b.tag));
        break;
      case "count":
      default:
        result.sort((a, b) => b.count - a.count);
        break;
    }

    return result;
  }, [tags, searchTerm, sortBy, selectedCategory]);

  // Group tags by first letter for alphabetical view
  const groupedTags = useMemo(() => {
    if (sortBy !== "alphabetical") return null;

    const groups: Record<string, TagInfo[]> = {};
    filteredTags.forEach((tag) => {
      const firstLetter = tag.tag[0].toUpperCase();
      if (!groups[firstLetter]) {
        groups[firstLetter] = [];
      }
      groups[firstLetter].push(tag);
    });

    return groups;
  }, [filteredTags, sortBy]);

  // Calculate statistics
  const stats = useMemo(() => {
    const totalTags = tags.length;
    const totalArticles = tags.reduce((sum, tag) => sum + tag.count, 0);
    const avgArticlesPerTag = totalTags > 0 ? (totalArticles / totalTags).toFixed(1) : "0";
    const topTag = tags[0];

    return { totalTags, totalArticles, avgArticlesPerTag, topTag };
  }, [tags]);

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
            Loading tags...
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
                {/* Icon */}
                <div className="inline-flex items-center justify-center mb-6">
                  <div
                    className="w-20 h-20 rounded-2xl flex items-center justify-center shadow-xl"
                    style={{
                      backgroundColor: "var(--accent)",
                      boxShadow: "0 12px 40px rgba(var(--accent-rgb, 130, 170, 255), 0.35)",
                    }}
                  >
                    <svg
                      className="w-10 h-10 text-white"
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

                <h1
                  className="text-4xl md:text-5xl font-bold mb-4"
                  style={{ color: "var(--text-primary)" }}
                >
                  All Tags
                </h1>
                <p
                  className="text-lg max-w-2xl mx-auto"
                  style={{ color: "var(--text-secondary)" }}
                >
                  Explore articles by topics. Click on any tag to see related articles.
                </p>
              </div>
            </FadeInWrapper>

            {/* Statistics Cards */}
            <FadeInWrapper duration={600} delay={200}>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-12">
                <div
                  className="p-6 rounded-xl border text-center transition-all duration-300 hover:scale-105 hover:shadow-lg"
                  style={{
                    backgroundColor: "var(--surface)",
                    borderColor: "var(--border)",
                  }}
                >
                  <div
                    className="text-3xl font-bold mb-1"
                    style={{ color: "var(--accent)" }}
                  >
                    {stats.totalTags}
                  </div>
                  <div
                    className="text-sm"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    Total Tags
                  </div>
                </div>
                <div
                  className="p-6 rounded-xl border text-center transition-all duration-300 hover:scale-105 hover:shadow-lg"
                  style={{
                    backgroundColor: "var(--surface)",
                    borderColor: "var(--border)",
                  }}
                >
                  <div
                    className="text-3xl font-bold mb-1"
                    style={{ color: "var(--accent)" }}
                  >
                    {stats.totalArticles}
                  </div>
                  <div
                    className="text-sm"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    Tag Uses
                  </div>
                </div>
                <div
                  className="p-6 rounded-xl border text-center transition-all duration-300 hover:scale-105 hover:shadow-lg"
                  style={{
                    backgroundColor: "var(--surface)",
                    borderColor: "var(--border)",
                  }}
                >
                  <div
                    className="text-3xl font-bold mb-1"
                    style={{ color: "var(--accent)" }}
                  >
                    {stats.avgArticlesPerTag}
                  </div>
                  <div
                    className="text-sm"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    Avg. per Tag
                  </div>
                </div>
                <div
                  className="p-6 rounded-xl border text-center transition-all duration-300 hover:scale-105 hover:shadow-lg"
                  style={{
                    backgroundColor: "var(--surface)",
                    borderColor: "var(--border)",
                  }}
                >
                  <div
                    className="text-lg font-bold mb-1 truncate"
                    style={{ color: "var(--accent)" }}
                    title={stats.topTag?.tag}
                  >
                    #{stats.topTag?.tag || "N/A"}
                  </div>
                  <div
                    className="text-sm"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    Most Popular
                  </div>
                </div>
              </div>
            </FadeInWrapper>

            {/* Search and Filter Controls */}
            <FadeInWrapper duration={600} delay={300}>
              <div className="mb-8">
                {/* Search Input */}
                <div className="relative max-w-md mx-auto mb-6">
                  <input
                    type="text"
                    placeholder="Search tags..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="w-full px-4 py-3 pl-12 rounded-xl border transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-[var(--accent)] focus:border-transparent"
                    style={{
                      backgroundColor: "var(--surface)",
                      borderColor: "var(--border)",
                      color: "var(--text-primary)",
                    }}
                  />
                  <svg
                    className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5"
                    style={{ color: "var(--text-muted)" }}
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
                  {searchTerm && (
                    <button
                      onClick={() => setSearchTerm("")}
                      className="absolute right-4 top-1/2 -translate-y-1/2 p-1 rounded-full hover:bg-[var(--accent-subtle)] transition-colors"
                      style={{ color: "var(--text-muted)" }}
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  )}
                </div>

                {/* Filter Controls */}
                <div className="flex flex-wrap justify-center items-center gap-4">
                  {/* Sort Options */}
                  <div className="flex items-center gap-2">
                    <span
                      className="text-sm font-medium"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      Sort by:
                    </span>
                    <div className="flex rounded-lg border overflow-hidden" style={{ borderColor: "var(--border)" }}>
                      {[
                        { value: "count", label: "Popular" },
                        { value: "alphabetical", label: "A-Z" },
                      ].map((option) => (
                        <button
                          key={option.value}
                          onClick={() => setSortBy(option.value as SortOption)}
                          className="px-4 py-2 text-sm font-medium transition-all duration-200"
                          style={{
                            backgroundColor:
                              sortBy === option.value
                                ? "var(--accent)"
                                : "var(--surface)",
                            color:
                              sortBy === option.value
                                ? "white"
                                : "var(--text-secondary)",
                          }}
                        >
                          {option.label}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Category Filter */}
                  <div className="flex items-center gap-2">
                    <span
                      className="text-sm font-medium"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      Category:
                    </span>
                    <select
                      value={selectedCategory}
                      onChange={(e) => setSelectedCategory(e.target.value)}
                      className="px-4 py-2 rounded-lg border text-sm font-medium transition-all duration-200 cursor-pointer"
                      style={{
                        backgroundColor: "var(--surface)",
                        borderColor: "var(--border)",
                        color: "var(--text-primary)",
                      }}
                    >
                      <option value="all">All Categories</option>
                      {categories.map((cat) => (
                        <option key={cat} value={cat}>
                          {cat.charAt(0).toUpperCase() + cat.slice(1).replace(/-/g, " ")}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>

                {/* Results Count */}
                <div className="text-center mt-4">
                  <span
                    className="text-sm"
                    style={{ color: "var(--text-muted)" }}
                  >
                    Showing {filteredTags.length} of {tags.length} tags
                  </span>
                </div>
              </div>
            </FadeInWrapper>

            {/* Tags Display */}
            <FadeInWrapper duration={600} delay={400}>
              {filteredTags.length > 0 ? (
                sortBy === "alphabetical" && groupedTags ? (
                  // Alphabetical grouped view
                  <div className="space-y-8">
                    {Object.entries(groupedTags).map(([letter, letterTags]) => (
                      <div key={letter}>
                        <div className="flex items-center gap-4 mb-4">
                          <div
                            className="w-10 h-10 rounded-lg flex items-center justify-center font-bold text-xl"
                            style={{
                              backgroundColor: "var(--accent-subtle)",
                              color: "var(--accent)",
                            }}
                          >
                            {letter}
                          </div>
                          <div
                            className="flex-1 h-px"
                            style={{ backgroundColor: "var(--border)" }}
                          />
                        </div>
                        <div className="flex flex-wrap gap-3 pl-14">
                          {letterTags.map((tag, index) => (
                            <FadeInWrapper
                              key={tag.slug}
                              duration={300}
                              delay={index * 30}
                            >
                              <TagBadge
                                tag={tag.tag}
                                count={tag.count}
                                variant="large"
                                clickable={true}
                              />
                            </FadeInWrapper>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  // Cloud view (by popularity)
                  <div
                    className="p-8 rounded-2xl border"
                    style={{
                      backgroundColor: "var(--surface)",
                      borderColor: "var(--border)",
                    }}
                  >
                    <div className="flex flex-wrap justify-center gap-3">
                      {filteredTags.map((tag, index) => (
                        <FadeInWrapper
                          key={tag.slug}
                          duration={300}
                          delay={index * 20}
                        >
                          <TagBadge
                            tag={tag.tag}
                            count={tag.count}
                            variant={
                              tag.count > 10
                                ? "large"
                                : tag.count > 5
                                ? "default"
                                : "compact"
                            }
                            clickable={true}
                            showIcon={tag.count > 15}
                          />
                        </FadeInWrapper>
                      ))}
                    </div>
                  </div>
                )
              ) : (
                // No results
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
                        d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                      />
                    </svg>
                  </div>
                  <h2
                    className="text-2xl font-bold mb-4"
                    style={{ color: "var(--text-primary)" }}
                  >
                    No tags found
                  </h2>
                  <p
                    className="text-lg mb-8"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    No tags match your search criteria.
                  </p>
                  <button
                    onClick={() => {
                      setSearchTerm("");
                      setSelectedCategory("all");
                    }}
                    className="px-6 py-3 rounded-lg font-medium transition-all duration-300 hover:scale-105"
                    style={{
                      backgroundColor: "var(--accent)",
                      color: "white",
                    }}
                  >
                    Clear Filters
                  </button>
                </div>
              )}
            </FadeInWrapper>

            {/* Popular Tags Section (Quick Access) */}
            {tags.length > 0 && (
              <FadeInWrapper duration={600} delay={500}>
                <div className="mt-16">
                  <h2
                    className="text-2xl font-bold mb-6 text-center"
                    style={{ color: "var(--text-primary)" }}
                  >
                    Top 10 Popular Tags
                  </h2>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
                    {tags.slice(0, 10).map((tag, index) => (
                      <Link
                        key={tag.slug}
                        href={`/blog/tags/${tag.slug}`}
                        className="group p-4 rounded-xl border transition-all duration-300 hover:scale-105 hover:shadow-lg"
                        style={{
                          backgroundColor: "var(--surface)",
                          borderColor: "var(--border)",
                        }}
                      >
                        <div className="flex items-center gap-3">
                          <div
                            className="w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold"
                            style={{
                              backgroundColor: "var(--accent)",
                              color: "white",
                            }}
                          >
                            {index + 1}
                          </div>
                          <div className="flex-1 min-w-0">
                            <div
                              className="font-medium truncate group-hover:text-[var(--accent)] transition-colors"
                              style={{ color: "var(--text-primary)" }}
                            >
                              #{tag.tag}
                            </div>
                            <div
                              className="text-xs"
                              style={{ color: "var(--text-muted)" }}
                            >
                              {tag.count} {tag.count === 1 ? "article" : "articles"}
                            </div>
                          </div>
                        </div>
                      </Link>
                    ))}
                  </div>
                </div>
              </FadeInWrapper>
            )}

            {/* Navigation */}
            <nav
              className="mt-16 pt-8 border-t"
              style={{ borderColor: "var(--border)" }}
            >
              <div className="flex justify-center">
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
