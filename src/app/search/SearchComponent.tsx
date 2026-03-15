"use client";

import React, { useState, useEffect, useMemo, useCallback, useRef } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { searchContent, type SearchResult } from "@/lib/search";

// Hook to fetch all searchable content dynamically
function useSearchableContent() {
  const [articles, setArticles] = useState<SearchResult[]>([]);
  const [projects, setProjects] = useState<SearchResult[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      try {
        setIsLoading(true);
        setError(null);

        const [articlesResponse, projectsResponse] = await Promise.all([
          fetch("/api/blog/articles?excludeContent=true"),
          fetch("/api/projects"),
        ]);

        if (!articlesResponse.ok) throw new Error("Failed to fetch articles");
        if (!projectsResponse.ok) throw new Error("Failed to fetch projects");

        const [articlesData, projectsData] = await Promise.all([
          articlesResponse.json(),
          projectsResponse.json(),
        ]);

        // Convert articles to SearchResult format
        const articleResults: SearchResult[] = (
          articlesData.articles || []
        ).map(
          (article: {
            title: string;
            excerpt: string;
            slug: string;
            category: string;
            tags: string[];
            featured: boolean;
            date: string;
            difficulty: string;
            readTime: string;
          }) => ({
            title: article.title,
            description: article.excerpt,
            url: `/blog/${article.slug}`,
            type: "blog" as const,
            category: article.category,
            tags: article.tags,
            featured: article.featured,
            date: article.date,
            difficulty: article.difficulty,
            readTime: article.readTime,
          })
        );

        // Convert projects to SearchResult format
        const projectResults: SearchResult[] = (
          projectsData.projects || []
        ).map(
          (project: {
            title: string;
            description: string;
            id: string;
            category: string;
            technologies: string[];
            featured: boolean;
            status: string;
            highlights: string[];
          }) => ({
            title: project.title,
            description: project.description,
            url: `/projects/${project.id}`,
            type: "project" as const,
            category: project.category,
            technologies: project.technologies,
            featured: project.featured,
            status: project.status,
            highlights: project.highlights,
          })
        );

        setArticles(articleResults);
        setProjects(projectResults);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to fetch data");
        console.error("Error fetching searchable content:", err);
      } finally {
        setIsLoading(false);
      }
    }

    fetchData();
  }, []);

  const searchableContent = useMemo(() => {
    const staticPages: SearchResult[] = [
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
    ];

    return [...articles, ...projects, ...staticPages];
  }, [articles, projects]);

  return { searchableContent, isLoading, error };
}

// --- Shared constants (module-level, stable references) ---

const popularSuggestions = [
  "Machine Learning",
  "Neural Networks",
  "Computer Vision",
  "NLP",
  "Deep Learning",
  "AI Research",
  "Next.js",
  "TypeScript",
  "Distributed Systems",
  "Blockchain",
  "Research Papers",
  "Open Source",
];

const typeIcons = {
  blog: (
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
  project: (
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
  page: (
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
};

const typeLabels: Record<string, string> = {
  blog: "Blog Post",
  project: "Project",
  page: "Page",
};

const filterOptions = ["all", "blog", "project", "page"] as const;

// --- Relevance badge helper ---

function RelevanceBadge({ score }: { score?: number }) {
  if (!score || score <= 100) return null;

  let label: string;
  let accent: boolean;

  if (score > 200) {
    label = "Perfect Match";
    accent = true;
  } else if (score > 150) {
    label = "High Match";
    accent = true;
  } else {
    label = "Good Match";
    accent = false;
  }

  return (
    <>
      <span style={{ color: "var(--text-secondary)" }}>•</span>
      <span
        className="text-xs font-bold px-2 py-1 rounded-full"
        style={{
          backgroundColor: accent ? "var(--accent)" : "var(--surface-accent)",
          color: accent ? "white" : "var(--accent)",
        }}
      >
        {label}
      </span>
    </>
  );
}

// --- Highlight helper ---

function HighlightedText({
  text,
  query,
}: {
  text: string;
  query: string;
}) {
  if (!query.trim()) return <>{text}</>;

  const queryWords = query
    .toLowerCase()
    .trim()
    .split(/\s+/)
    .filter((w) => w.length > 0);

  const escaped = queryWords
    .map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
    .join("|");
  const fullEscaped = query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

  const pattern =
    queryWords.length > 1 ? `${fullEscaped}|${escaped}` : escaped;

  let regex: RegExp;
  try {
    regex = new RegExp(`(${pattern})`, "gi");
  } catch {
    return <>{text}</>;
  }

  const parts = text.split(regex);

  return (
    <>
      {parts.map((part, i) =>
        i % 2 === 1 ? (
          <mark
            key={i}
            className="font-semibold px-0.5 rounded bg-[var(--accent)]/15 text-[var(--accent)]"
          >
            {part}
          </mark>
        ) : (
          <React.Fragment key={i}>{part}</React.Fragment>
        )
      )}
    </>
  );
}

// --- Main component ---

export default function SearchComponent() {
  const searchParams = useSearchParams();
  const [query, setQuery] = useState(searchParams.get("q") || "");
  const [debouncedQuery, setDebouncedQuery] = useState(query);
  const [selectedFilter, setSelectedFilter] = useState<string>("all");
  const inputRef = useRef<HTMLInputElement>(null);

  const {
    searchableContent,
    isLoading: dataLoading,
    error: dataError,
  } = useSearchableContent();

  // Keyboard shortcut: focus search with /
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (
        e.key === "/" &&
        !["INPUT", "TEXTAREA", "SELECT"].includes(
          (e.target as HTMLElement).tagName
        )
      ) {
        e.preventDefault();
        inputRef.current?.focus();
      }
      if (e.key === "Escape" && document.activeElement === inputRef.current) {
        inputRef.current?.blur();
      }
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, []);

  // Single debounce effect
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedQuery(query);
      // Update URL
      if (query.trim()) {
        const url = new URL(window.location.href);
        url.searchParams.set("q", query);
        window.history.replaceState({}, "", url);
      }
    }, 300);
    return () => clearTimeout(timer);
  }, [query]);

  // Search results
  const searchResults = useMemo(() => {
    if (!debouncedQuery.trim() || dataLoading) return [];
    return searchContent(debouncedQuery, searchableContent, undefined, 0.1);
  }, [debouncedQuery, searchableContent, dataLoading]);

  // Filtered results + counts
  const { filteredResults, filterCounts } = useMemo(() => {
    const counts = { all: searchResults.length, blog: 0, project: 0, page: 0 };
    searchResults.forEach((r) => {
      if (r.type in counts) counts[r.type as keyof typeof counts]++;
    });
    const filtered =
      selectedFilter === "all"
        ? searchResults
        : searchResults.filter((r) => r.type === selectedFilter);
    return { filteredResults: filtered, filterCounts: counts };
  }, [searchResults, selectedFilter]);

  const handleSearch = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      setDebouncedQuery(query);
    },
    [query]
  );

  const handleSuggestionClick = useCallback((suggestion: string) => {
    setQuery(suggestion);
    setDebouncedQuery(suggestion);
  }, []);

  if (dataError) {
    return (
      <div className="text-center py-16">
        <div className="max-w-md mx-auto">
          <div
            className="w-24 h-24 mx-auto mb-6 rounded-full flex items-center justify-center"
            style={{ backgroundColor: "var(--surface)" }}
          >
            <svg
              className="w-12 h-12"
              style={{ color: "var(--text-secondary)" }}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
          </div>
          <h3
            className="text-2xl font-bold mb-3"
            style={{ color: "var(--text-primary)" }}
          >
            Error Loading Content
          </h3>
          <p
            className="text-lg mb-8 leading-relaxed"
            style={{ color: "var(--text-secondary)" }}
          >
            {dataError}
          </p>
          <button
            onClick={() => window.location.reload()}
            className="px-6 py-3 font-semibold rounded-xl text-white transition-all duration-300 hover:scale-105"
            style={{ backgroundColor: "var(--accent)" }}
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-12">
      {/* Search Input */}
      <div className="max-w-4xl mx-auto">
        <form onSubmit={handleSearch} className="relative group">
          <div className="relative">
            <div className="absolute inset-y-0 left-0 pl-6 flex items-center pointer-events-none">
              <svg
                className="w-6 h-6 text-[var(--text-secondary)]"
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
              ref={inputRef}
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search articles, projects, and more..."
              className="block w-full pl-16 pr-20 py-5 text-lg border-2 rounded-2xl transition-all duration-300 focus:outline-none focus:ring-4 focus:ring-[var(--accent)]/20 focus:border-[var(--accent)] shadow-lg hover:shadow-xl focus:shadow-2xl"
              style={{
                backgroundColor: "var(--card-bg)",
                borderColor: "var(--border)",
                color: "var(--text-primary)",
              }}
            />
            <div className="absolute inset-y-0 right-0 pr-4 flex items-center gap-2">
              {query && (
                <button
                  type="button"
                  onClick={() => {
                    setQuery("");
                    setDebouncedQuery("");
                    inputRef.current?.focus();
                  }}
                  className="p-2 rounded-lg text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--surface)] transition-colors"
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
                      strokeWidth="2"
                      d="M6 18L18 6M6 6l12 12"
                    />
                  </svg>
                </button>
              )}
              <button
                type="submit"
                className="p-3 text-white rounded-xl transition-all duration-300 focus:outline-none focus:ring-4 shadow-lg hover:shadow-xl hover:scale-105 hover:brightness-110"
                style={{ backgroundColor: "var(--accent)" }}
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
                    strokeWidth="2"
                    d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                  />
                </svg>
              </button>
            </div>
          </div>
        </form>

        {/* Keyboard shortcut hint */}
        {!query && (
          <div className="mt-3 text-center">
            <span
              className="text-sm"
              style={{ color: "var(--text-secondary)" }}
            >
              Press{" "}
              <kbd className="px-2 py-1 text-xs rounded border font-mono bg-[var(--surface)] border-[var(--border)]">
                /
              </kbd>{" "}
              to focus search
            </span>
          </div>
        )}

        {/* Popular suggestions */}
        {!debouncedQuery && (
          <div className="mt-8 text-center">
            <p
              className="text-lg mb-6 font-medium"
              style={{ color: "var(--text-secondary)" }}
            >
              Popular Topics
            </p>
            <div className="flex flex-wrap justify-center gap-3">
              {popularSuggestions.map((suggestion) => (
                <button
                  key={suggestion}
                  onClick={() => handleSuggestionClick(suggestion)}
                  className="px-5 py-2.5 text-sm font-medium rounded-full transition-all duration-200 border-2 hover:scale-105 hover:shadow-md hover:border-[var(--accent)] hover:text-[var(--accent)] hover:bg-[var(--surface-accent)]"
                  style={{
                    backgroundColor: "var(--surface)",
                    color: "var(--text-secondary)",
                    borderColor: "var(--border)",
                  }}
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Results Section */}
      {debouncedQuery && (
        <div className="space-y-8">
          {/* Results header + filters */}
          <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6">
            <div className="space-y-2">
              <h2
                className="text-2xl lg:text-3xl font-bold"
                style={{ color: "var(--text-primary)" }}
              >
                {dataLoading
                  ? "Searching..."
                  : filteredResults.length === 0
                    ? "No results found"
                    : `${filteredResults.length} result${filteredResults.length === 1 ? "" : "s"}`}
              </h2>
              {!dataLoading && (
                <p
                  className="text-lg"
                  style={{ color: "var(--text-secondary)" }}
                >
                  {filteredResults.length === 0
                    ? "Try different keywords or browse suggested content below"
                    : `for "${debouncedQuery}"`}
                </p>
              )}
            </div>

            {/* Filter buttons */}
            {searchResults.length > 0 && (
              <div className="flex flex-wrap gap-2">
                {filterOptions.map((filter) => {
                  const count =
                    filterCounts[filter as keyof typeof filterCounts] || 0;
                  const isActive = selectedFilter === filter;
                  return (
                    <button
                      key={filter}
                      onClick={() => setSelectedFilter(filter)}
                      className={`px-5 py-3 text-sm font-semibold rounded-xl transition-all duration-200 border-2 hover:scale-105 ${
                        isActive
                          ? "text-white border-[var(--accent)] shadow-lg"
                          : "border-[var(--border)] hover:border-[var(--accent)] hover:text-[var(--accent)]"
                      }`}
                      style={{
                        backgroundColor: isActive
                          ? "var(--accent)"
                          : "var(--surface)",
                        color: isActive ? "white" : "var(--text-secondary)",
                      }}
                    >
                      <div className="flex items-center gap-2">
                        <span>
                          {filter === "all" ? "All" : typeLabels[filter]}
                        </span>
                        {count > 0 && (
                          <span
                            className="px-2 py-1 text-xs rounded-full font-bold"
                            style={{
                              backgroundColor: isActive
                                ? "rgba(255,255,255,0.2)"
                                : "var(--accent)",
                              color: "white",
                            }}
                          >
                            {count}
                          </span>
                        )}
                      </div>
                    </button>
                  );
                })}
              </div>
            )}
          </div>

          {/* Results grid */}
          {filteredResults.length > 0 && (
            <div className="grid gap-6">
              {filteredResults.map((result, index) => (
                <Link
                  key={`${result.url}-${index}`}
                  href={result.url}
                  className="group block p-8 rounded-2xl border-2 transition-all duration-300 hover:scale-[1.01] hover:shadow-2xl hover:border-[var(--accent)]"
                  style={{
                    backgroundColor: "var(--card-bg)",
                    borderColor: "var(--card-border)",
                  }}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex items-start gap-5 flex-1">
                      <div
                        className="p-4 rounded-xl flex-shrink-0"
                        style={{
                          backgroundColor: "var(--surface-accent)",
                          color: "var(--accent)",
                        }}
                      >
                        {typeIcons[result.type]}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-3 mb-3 flex-wrap">
                          <span
                            className="text-xs font-bold px-3 py-1.5 rounded-full uppercase tracking-wide"
                            style={{
                              backgroundColor: "var(--surface-accent)",
                              color: "var(--accent)",
                            }}
                          >
                            {typeLabels[result.type]}
                          </span>
                          {result.category && (
                            <>
                              <span
                                style={{ color: "var(--text-secondary)" }}
                              >
                                •
                              </span>
                              <span
                                className="text-sm font-medium"
                                style={{ color: "var(--text-secondary)" }}
                              >
                                {result.category}
                              </span>
                            </>
                          )}
                          <RelevanceBadge score={result.relevanceScore} />
                        </div>
                        <h3 className="text-xl lg:text-2xl font-bold mb-3 leading-tight group-hover:text-[var(--accent)] transition-colors duration-200">
                          <HighlightedText
                            text={result.title}
                            query={debouncedQuery}
                          />
                        </h3>
                        <p
                          className="text-base leading-relaxed mb-4"
                          style={{ color: "var(--text-secondary)" }}
                        >
                          <HighlightedText
                            text={result.description}
                            query={debouncedQuery}
                          />
                        </p>
                        {/* Meta info row */}
                        <div className="flex flex-wrap items-center gap-3">
                          {result.date && (
                            <span
                              className="text-xs"
                              style={{ color: "var(--text-secondary)" }}
                            >
                              {new Date(result.date).toLocaleDateString(
                                "en-US",
                                {
                                  year: "numeric",
                                  month: "short",
                                  day: "numeric",
                                }
                              )}
                            </span>
                          )}
                          {result.readTime && (
                            <span
                              className="text-xs"
                              style={{ color: "var(--text-secondary)" }}
                            >
                              {result.readTime}
                            </span>
                          )}
                          {result.tags &&
                            result.tags.slice(0, 4).map((tag) => (
                              <span
                                key={tag}
                                className="text-xs px-3 py-1 rounded-lg font-medium"
                                style={{
                                  backgroundColor: "var(--surface)",
                                  color: "var(--text-secondary)",
                                }}
                              >
                                {tag}
                              </span>
                            ))}
                          {result.tags && result.tags.length > 4 && (
                            <span
                              className="text-xs px-3 py-1 rounded-lg font-medium"
                              style={{
                                backgroundColor: "var(--surface)",
                                color: "var(--text-secondary)",
                              }}
                            >
                              +{result.tags.length - 4} more
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                    <div className="group-hover:translate-x-2 transition-transform duration-300 flex-shrink-0 ml-6">
                      <svg
                        className="w-6 h-6 text-[var(--text-secondary)] group-hover:text-[var(--accent)] transition-colors"
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
              ))}
            </div>
          )}

          {/* No Results State */}
          {!dataLoading && debouncedQuery && filteredResults.length === 0 && (
            <div className="text-center py-16">
              <div className="max-w-md mx-auto">
                <div
                  className="w-24 h-24 mx-auto mb-6 rounded-full flex items-center justify-center"
                  style={{ backgroundColor: "var(--surface)" }}
                >
                  <svg
                    className="w-12 h-12"
                    style={{ color: "var(--text-secondary)" }}
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
                <h3
                  className="text-2xl font-bold mb-3"
                  style={{ color: "var(--text-primary)" }}
                >
                  No Results Found
                </h3>
                <p
                  className="text-lg mb-8 leading-relaxed"
                  style={{ color: "var(--text-secondary)" }}
                >
                  Try different keywords or explore content below
                </p>
                <div className="flex flex-col sm:flex-row justify-center gap-4">
                  <Link
                    href="/blog"
                    className="px-6 py-3 font-semibold rounded-xl transition-all duration-300 text-white hover:scale-105 hover:brightness-110"
                    style={{ backgroundColor: "var(--accent)" }}
                  >
                    Browse Articles
                  </Link>
                  <Link
                    href="/projects"
                    className="px-6 py-3 font-semibold rounded-xl transition-all duration-300 border-2 hover:scale-105 hover:border-[var(--accent)] hover:text-[var(--accent)]"
                    style={{
                      backgroundColor: "var(--surface)",
                      color: "var(--text-primary)",
                      borderColor: "var(--border)",
                    }}
                  >
                    View Projects
                  </Link>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Discovery Section - when no query */}
      {!query && (
        <div className="space-y-8">
          <div className="text-center">
            <h2
              className="text-2xl lg:text-3xl font-bold mb-4"
              style={{ color: "var(--text-primary)" }}
            >
              Recent Content
            </h2>
            <p
              className="text-xl max-w-3xl mx-auto leading-relaxed"
              style={{ color: "var(--text-secondary)" }}
            >
              Explore research, technical insights, and projects in AI, machine
              learning, and software development
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {searchableContent
              .filter((item) => item.type !== "page")
              .slice(0, 6)
              .map((item, index) => (
                <Link
                  key={`${item.url}-${index}`}
                  href={item.url}
                  className="group block p-6 rounded-2xl border-2 transition-all duration-300 hover:scale-[1.03] hover:shadow-2xl hover:border-[var(--accent)]"
                  style={{
                    backgroundColor: "var(--card-bg)",
                    borderColor: "var(--card-border)",
                  }}
                >
                  <div className="flex items-start gap-4 mb-4">
                    <div
                      className="p-3 rounded-xl flex-shrink-0"
                      style={{
                        backgroundColor: "var(--surface-accent)",
                        color: "var(--accent)",
                      }}
                    >
                      {typeIcons[item.type]}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div
                        className="text-xs font-bold px-3 py-1.5 rounded-full inline-block mb-3 uppercase tracking-wide"
                        style={{
                          backgroundColor: "var(--surface-accent)",
                          color: "var(--accent)",
                        }}
                      >
                        {typeLabels[item.type]}
                      </div>
                      <h4 className="font-bold text-lg leading-snug group-hover:text-[var(--accent)] transition-colors duration-200">
                        {item.title}
                      </h4>
                    </div>
                  </div>
                  <p
                    className="text-sm leading-relaxed mb-4 line-clamp-3"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    {item.description}
                  </p>
                  {item.category && (
                    <div className="flex items-center justify-between">
                      <span
                        className="text-sm font-medium"
                        style={{ color: "var(--accent)" }}
                      >
                        {item.category}
                      </span>
                      <svg
                        className="w-5 h-5 transition-transform duration-300 group-hover:translate-x-1 text-[var(--text-secondary)]"
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
                  )}
                </Link>
              ))}
          </div>
        </div>
      )}
    </div>
  );
}
