"use client";

import React, { useState, useEffect, useMemo } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";

interface SearchResult {
  title: string;
  description: string;
  url: string;
  type: "blog" | "page" | "project";
  category?: string;
  tags?: string[];
}

const searchableContent: SearchResult[] = [
  {
    title: "Modern Portfolio with Next.js",
    description:
      "Building a modern, responsive portfolio website using Next.js 14, TypeScript, and Tailwind CSS with dark mode support.",
    url: "/blog/modern-portfolio-nextjs",
    type: "blog",
    category: "Software Development",
    tags: ["Next.js", "TypeScript", "Tailwind CSS", "React"],
  },
  {
    title: "Open Source Guide",
    description:
      "A comprehensive guide to contributing to open source projects, from finding projects to making your first contribution.",
    url: "/blog/open-source-guide",
    type: "blog",
    category: "Software Development",
    tags: ["Open Source", "Git", "GitHub", "Contributing"],
  },
  {
    title: "About",
    description:
      "Learn more about Hiep Tran - AI Research Engineer and Full-Stack Developer specializing in machine learning and web development.",
    url: "/about",
    type: "page",
    tags: ["About", "Profile", "Background"],
  },
  {
    title: "Projects",
    description:
      "Explore my portfolio of AI research projects, web applications, and open source contributions.",
    url: "/projects",
    type: "page",
    tags: ["Portfolio", "Projects", "AI", "Web Development"],
  },
  {
    title: "Contact",
    description:
      "Get in touch for collaboration opportunities, research discussions, or project inquiries.",
    url: "/contact",
    type: "page",
    tags: ["Contact", "Collaboration", "Hire"],
  },
  {
    title: "Machine Learning Blog",
    description:
      "Articles and insights about machine learning, AI research, and data science.",
    url: "/blog/machine-learning",
    type: "blog",
    category: "Machine Learning",
    tags: ["Machine Learning", "AI", "Data Science", "Research"],
  },
  {
    title: "Paper Reading Notes",
    description:
      "Summaries and insights from reading research papers in AI and machine learning.",
    url: "/blog/paper-reading",
    type: "blog",
    category: "Research",
    tags: ["Research Papers", "AI", "Academic", "Notes"],
  },
  {
    title: "Cryptocurrency & Blockchain",
    description:
      "Exploring blockchain technology, DeFi, and cryptocurrency development.",
    url: "/blog/crypto",
    type: "blog",
    category: "Blockchain",
    tags: ["Cryptocurrency", "Blockchain", "DeFi", "Web3"],
  },
  {
    title: "Development Notes",
    description:
      "Quick notes, tips, and tricks for software development and programming.",
    url: "/blog/notes",
    type: "blog",
    category: "Development",
    tags: ["Programming", "Tips", "Development", "Best Practices"],
  },
];

const typeConfig = {
  blog: {
    label: "Blog Post",
    color: "text-blue-600 dark:text-blue-400",
    bgColor: "bg-blue-50 dark:bg-blue-950/30",
    icon: (
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
    ),
  },
  project: {
    label: "Project",
    color: "text-emerald-600 dark:text-emerald-400",
    bgColor: "bg-emerald-50 dark:bg-emerald-950/30",
    icon: (
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
          d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"
        />
      </svg>
    ),
  },
  page: {
    label: "Page",
    color: "text-purple-600 dark:text-purple-400",
    bgColor: "bg-purple-50 dark:bg-purple-950/30",
    icon: (
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
          d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
        />
      </svg>
    ),
  },
};

export default function SearchComponent() {
  const searchParams = useSearchParams();
  const [query, setQuery] = useState(searchParams.get("q") || "");
  const [selectedFilter, setSelectedFilter] = useState<string>("all");
  const [isSearching, setIsSearching] = useState(false);

  // Perform search
  const searchResults = useMemo(() => {
    if (!query.trim()) return [];

    const lowercaseQuery = query.toLowerCase();
    return searchableContent.filter(
      (item) =>
        item.title.toLowerCase().includes(lowercaseQuery) ||
        item.description.toLowerCase().includes(lowercaseQuery) ||
        item.category?.toLowerCase().includes(lowercaseQuery) ||
        item.tags?.some((tag) => tag.toLowerCase().includes(lowercaseQuery))
    );
  }, [query]);

  // Filter results by type
  const filteredResults = useMemo(() => {
    if (selectedFilter === "all") return searchResults;
    return searchResults.filter((result) => result.type === selectedFilter);
  }, [searchResults, selectedFilter]);

  // Calculate filter counts
  const filterCounts = useMemo(() => {
    const counts = { all: searchResults.length, blog: 0, project: 0, page: 0 };
    searchResults.forEach((result) => {
      counts[result.type]++;
    });
    return counts;
  }, [searchResults]);

  useEffect(() => {
    if (query.trim()) {
      setIsSearching(true);
      const timer = setTimeout(() => setIsSearching(false), 300);
      return () => clearTimeout(timer);
    }
  }, [query]);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      const url = new URL(window.location.href);
      url.searchParams.set("q", query);
      window.history.replaceState({}, "", url);
    }
  };

  const popularSuggestions = [
    "Next.js",
    "Machine Learning",
    "TypeScript",
    "AI Research",
    "Open Source",
    "Web Development",
    "React",
    "Python",
  ];

  return (
    <div className="space-y-8">
      {/* Search Input */}
      <div className="max-w-3xl mx-auto">
        <form onSubmit={handleSearch} className="relative group">
          <div className="relative">
            <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
              <svg
                className="w-5 h-5 text-gray-400"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                />
              </svg>
            </div>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search for articles, projects, or topics..."
              className="block w-full pl-12 pr-16 py-4 text-lg border border-gray-200 dark:border-gray-700 rounded-2xl bg-white dark:bg-gray-800 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200 shadow-sm hover:shadow-md focus:shadow-lg"
            />
            <div className="absolute inset-y-0 right-0 pr-3 flex items-center">
              {isSearching ? (
                <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
              ) : (
                <button
                  type="submit"
                  className="p-2 text-white bg-blue-600 hover:bg-blue-700 rounded-xl transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
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
                      d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                    />
                  </svg>
                </button>
              )}
            </div>
          </div>
        </form>

        {/* Search suggestions when empty */}
        {!query && (
          <div className="mt-6 text-center">
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
              Popular searches:
            </p>
            <div className="flex flex-wrap justify-center gap-2">
              {popularSuggestions.map((suggestion) => (
                <button
                  key={suggestion}
                  onClick={() => setQuery(suggestion)}
                  className="px-3 py-1.5 text-sm font-medium text-gray-600 dark:text-gray-300 bg-gray-100 dark:bg-gray-800 hover:bg-blue-100 dark:hover:bg-blue-900 hover:text-blue-700 dark:hover:text-blue-300 rounded-full transition-all duration-200 border border-gray-200 dark:border-gray-700 hover:border-blue-300 dark:hover:border-blue-600"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Results */}
      {query && (
        <div className="space-y-6">
          {/* Results header and filters */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div>
              {!isSearching && (
                <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                  {filteredResults.length === 0
                    ? `No results found for "${query}"`
                    : `${filteredResults.length} result${
                        filteredResults.length === 1 ? "" : "s"
                      } for "${query}"`}
                </h2>
              )}
              {isSearching && (
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
                  <span className="text-gray-600 dark:text-gray-400">
                    Searching...
                  </span>
                </div>
              )}
            </div>

            {/* Filter buttons */}
            {searchResults.length > 0 && (
              <div className="flex gap-2">
                {Object.entries(filterCounts).map(([filter, count]) => (
                  <button
                    key={filter}
                    onClick={() => setSelectedFilter(filter)}
                    className={`px-4 py-2 text-sm font-medium rounded-lg transition-all duration-200 border ${
                      selectedFilter === filter
                        ? "bg-blue-600 text-white border-blue-600"
                        : "bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700"
                    }`}
                  >
                    {filter === "all"
                      ? "All"
                      : typeConfig[filter as keyof typeof typeConfig]?.label}
                    {count > 0 && (
                      <span
                        className={`ml-1.5 px-1.5 py-0.5 text-xs rounded-full ${
                          selectedFilter === filter
                            ? "bg-blue-500 text-white"
                            : "bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-400"
                        }`}
                      >
                        {count}
                      </span>
                    )}
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Results grid */}
          {filteredResults.length > 0 && (
            <div className="grid gap-4">
              {filteredResults.map((result, index) => {
                const config = typeConfig[result.type];
                return (
                  <Link
                    key={index}
                    href={result.url}
                    className="group block p-6 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 hover:border-blue-300 dark:hover:border-blue-600 transition-all duration-200 hover:shadow-lg hover:-translate-y-0.5"
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex items-start gap-4 flex-1">
                        <div
                          className={`p-2.5 rounded-lg ${config.bgColor} ${config.color} flex-shrink-0`}
                        >
                          {config.icon}
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-2">
                            <span
                              className={`text-xs font-medium px-2 py-1 rounded-full ${config.bgColor} ${config.color}`}
                            >
                              {config.label}
                            </span>
                            {result.category && (
                              <span className="text-xs text-gray-500 dark:text-gray-400">
                                • {result.category}
                              </span>
                            )}
                          </div>
                          <h3 className="text-lg font-semibold text-gray-900 dark:text-white group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors duration-200 mb-2">
                            {result.title}
                          </h3>
                          <p className="text-gray-600 dark:text-gray-300 line-clamp-2 leading-relaxed">
                            {result.description}
                          </p>
                          {result.tags && (
                            <div className="flex flex-wrap gap-1 mt-3">
                              {result.tags.slice(0, 3).map((tag) => (
                                <span
                                  key={tag}
                                  className="text-xs px-2 py-0.5 bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400 rounded-md"
                                >
                                  {tag}
                                </span>
                              ))}
                              {result.tags.length > 3 && (
                                <span className="text-xs px-2 py-0.5 bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400 rounded-md">
                                  +{result.tags.length - 3}
                                </span>
                              )}
                            </div>
                          )}
                        </div>
                      </div>
                      <div className="group-hover:translate-x-1 transition-transform duration-200 flex-shrink-0 ml-4">
                        <svg
                          className="w-5 h-5 text-gray-400 group-hover:text-blue-500"
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
                  </Link>
                );
              })}
            </div>
          )}

          {/* No Results State */}
          {query && !isSearching && filteredResults.length === 0 && (
            <div className="text-center py-12">
              <div className="max-w-md mx-auto">
                <div className="w-16 h-16 mx-auto mb-4 bg-gray-100 dark:bg-gray-800 rounded-full flex items-center justify-center">
                  <svg
                    className="w-8 h-8 text-gray-400"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                    />
                  </svg>
                </div>
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                  No results found
                </h3>
                <p className="text-gray-600 dark:text-gray-400 mb-6">
                  Try different keywords or browse our content directly
                </p>
                <div className="flex justify-center gap-3">
                  <Link
                    href="/blog"
                    className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors duration-200"
                  >
                    Browse Blog
                  </Link>
                  <Link
                    href="/projects"
                    className="px-4 py-2 bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300 font-medium rounded-lg transition-colors duration-200"
                  >
                    View Projects
                  </Link>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Popular Content (when no search) */}
      {!query && (
        <div className="space-y-6">
          <div className="text-center">
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
              Popular Content
            </h2>
            <p className="text-gray-600 dark:text-gray-400">
              Discover the most popular articles and projects
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {searchableContent.slice(0, 6).map((item, index) => {
              const config = typeConfig[item.type];
              return (
                <Link
                  key={index}
                  href={item.url}
                  className="group block p-5 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 hover:border-blue-300 dark:hover:border-blue-600 transition-all duration-200 hover:shadow-lg hover:-translate-y-1"
                >
                  <div className="flex items-start gap-3 mb-3">
                    <div
                      className={`p-2 rounded-lg ${config.bgColor} ${config.color} flex-shrink-0`}
                    >
                      {config.icon}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div
                        className={`text-xs font-medium px-2 py-1 rounded-full ${config.bgColor} ${config.color} inline-block mb-2`}
                      >
                        {config.label}
                      </div>
                      <h4 className="font-semibold text-gray-900 dark:text-white group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors duration-200 line-clamp-2 leading-snug">
                        {item.title}
                      </h4>
                    </div>
                  </div>
                  <p className="text-sm text-gray-600 dark:text-gray-300 line-clamp-3 leading-relaxed mb-3">
                    {item.description}
                  </p>
                  {item.category && (
                    <span className="text-xs text-gray-500 dark:text-gray-400">
                      {item.category}
                    </span>
                  )}
                </Link>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
