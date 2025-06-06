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
  relevanceScore?: number;
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
    title: "AI-Powered Natural Language Processing",
    description:
      "Deep dive into NLP techniques using transformer models, including BERT, GPT, and custom implementations for text analysis.",
    url: "/blog/nlp-transformers",
    type: "blog",
    category: "Machine Learning",
    tags: ["NLP", "Transformers", "BERT", "GPT", "Deep Learning"],
  },
  {
    title: "Computer Vision Research",
    description:
      "Advanced computer vision techniques for object detection, image segmentation, and real-time video analysis.",
    url: "/blog/computer-vision",
    type: "blog",
    category: "Machine Learning",
    tags: ["Computer Vision", "CNN", "YOLO", "OpenCV", "PyTorch"],
  },
  {
    title: "Distributed Systems Architecture",
    description:
      "Designing scalable microservices architecture with Kubernetes, Docker, and cloud-native technologies.",
    url: "/blog/distributed-systems",
    type: "blog",
    category: "Software Engineering",
    tags: ["Microservices", "Kubernetes", "Docker", "Cloud", "Scalability"],
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
      "Articles and insights about machine learning, AI research, and data science methodologies.",
    url: "/blog/machine-learning",
    type: "blog",
    category: "Machine Learning",
    tags: ["Machine Learning", "AI", "Data Science", "Research"],
  },
  {
    title: "Paper Reading Notes",
    description:
      "Summaries and insights from reading research papers in AI and machine learning with critical analysis.",
    url: "/blog/paper-reading",
    type: "blog",
    category: "Research",
    tags: ["Research Papers", "AI", "Academic", "Notes"],
  },
  {
    title: "Cryptocurrency & Blockchain",
    description:
      "Exploring blockchain technology, DeFi protocols, and cryptocurrency development with smart contracts.",
    url: "/blog/crypto",
    type: "blog",
    category: "Blockchain",
    tags: ["Cryptocurrency", "Blockchain", "DeFi", "Web3"],
  },
  {
    title: "Development Notes",
    description:
      "Quick notes, tips, and tricks for software development and programming best practices.",
    url: "/blog/notes",
    type: "blog",
    category: "Development",
    tags: ["Programming", "Tips", "Development", "Best Practices"],
  },
];

const typeConfig = {
  blog: {
    label: "Blog Post",
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

// Enhanced search algorithm with relevance scoring
const calculateRelevance = (item: SearchResult, query: string): number => {
  // Normalize input
  const normalizedQuery = query.toLowerCase().trim();
  const queryWords = normalizedQuery
    .split(/\s+/)
    .filter((word) => word.length > 0);

  // If query is empty, return 0
  if (queryWords.length === 0) return 0;

  let score = 0;

  // Normalize content fields
  const normalizedTitle = item.title.toLowerCase();
  const normalizedDescription = item.description.toLowerCase();
  const normalizedCategory = item.category?.toLowerCase() || "";
  const normalizedTags = item.tags?.map((tag) => tag.toLowerCase()) || [];

  // Exact phrase matching (highest weight)
  if (normalizedTitle.includes(normalizedQuery)) {
    score += 150;
    // Exact title match (perfect match)
    if (normalizedTitle === normalizedQuery) {
      score += 100;
    }
    // Title starts with query (very high relevance)
    if (normalizedTitle.startsWith(normalizedQuery)) {
      score += 75;
    }
  }

  // Category exact phrase matching (high weight)
  if (normalizedCategory.includes(normalizedQuery)) {
    score += 100;
    // Exact category match
    if (normalizedCategory === normalizedQuery) {
      score += 50;
    }
  }

  // Tag exact phrase matching (high weight)
  normalizedTags.some((tag) => {
    if (tag.includes(normalizedQuery)) {
      score += 75;
      // Exact tag match
      if (tag === normalizedQuery) {
        score += 50;
      }
      return true;
    }
    return false;
  });

  // Description exact phrase matching (medium weight)
  if (normalizedDescription.includes(normalizedQuery)) {
    score += 50;
  }

  // Multi-word query word-by-word matching with position bonus
  if (queryWords.length > 1) {
    let titleWordMatches = 0;
    let descriptionWordMatches = 0;
    let categoryWordMatches = 0;
    let tagWordMatches = 0;

    // Check how many query words match in each field
    queryWords.forEach((word, index) => {
      // Title word matches
      if (normalizedTitle.includes(word)) {
        titleWordMatches++;
        // Position bonus - earlier words in query matching has higher relevance
        score += Math.max(15, 25 - index * 3);

        // Word boundary matching bonus (matches whole words, not parts)
        const titleWords = normalizedTitle.split(/\s+/);
        if (titleWords.includes(word)) {
          score += 15;
        }
      }

      // Description word matches
      if (normalizedDescription.includes(word)) {
        descriptionWordMatches++;
        score += Math.max(5, 10 - index * 2);

        // Word boundary matching in description
        const descWords = normalizedDescription.split(/\s+/);
        if (descWords.includes(word)) {
          score += 5;
        }
      }

      // Category word matches
      if (normalizedCategory.includes(word)) {
        categoryWordMatches++;
        score += Math.max(10, 20 - index * 2);
      }

      // Tag word matches
      normalizedTags.forEach((tag) => {
        if (tag.includes(word)) {
          tagWordMatches++;
          score += Math.max(10, 15 - index * 2);
        }
      });
    });

    // Proximity bonus - if we matched all words in the query in a field
    if (titleWordMatches === queryWords.length) score += 50;
    if (descriptionWordMatches === queryWords.length) score += 20;
    if (categoryWordMatches === queryWords.length) score += 30;
    if (tagWordMatches >= queryWords.length) score += 25;

    // Percentage match bonus
    const titleMatchPercentage = titleWordMatches / queryWords.length;
    const descMatchPercentage = descriptionWordMatches / queryWords.length;
    const categoryMatchPercentage = categoryWordMatches / queryWords.length;
    const tagMatchPercentage = tagWordMatches / queryWords.length;

    score += titleMatchPercentage * 30;
    score += descMatchPercentage * 15;
    score += categoryMatchPercentage * 20;
    score += tagMatchPercentage * 20;
  }

  // Fuzzy matching for typo tolerance (only for single-word queries or each word in multi-word queries)
  queryWords.forEach((word) => {
    if (word.length > 3) {
      // Only do fuzzy matching for words longer than 3 chars
      // Simple edit distance simulation - check if title contains a substring with 1 character different
      for (let i = 0; i < word.length - 2; i++) {
        const fuzzyPattern = word.substring(0, i) + word.substring(i + 1);
        if (normalizedTitle.includes(fuzzyPattern)) {
          score += 15;
          break;
        }
      }

      // Check tags for fuzzy matches
      normalizedTags.forEach((tag) => {
        for (let i = 0; i < word.length - 2; i++) {
          const fuzzyPattern = word.substring(0, i) + word.substring(i + 1);
          if (tag.includes(fuzzyPattern)) {
            score += 10;
            break;
          }
        }
      });
    }
  });

  // Content type bonuses
  if (item.type === "blog") score += 5; // Slight preference for blog content
  if (item.type === "project") score += 3; // Some preference for projects

  // Normalization - cap the score to avoid extreme values
  score = Math.min(500, score);

  return score;
};

export default function SearchComponent() {
  const searchParams = useSearchParams();
  const [query, setQuery] = useState(searchParams.get("q") || "");
  const [debouncedQuery, setDebouncedQuery] = useState(query);
  const [selectedFilter, setSelectedFilter] = useState<string>("all");
  const [isSearching, setIsSearching] = useState(false);

  // Debounce the query to prevent excessive searching
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedQuery(query);
      if (query.trim()) {
        setIsSearching(true);
      }
    }, 300);

    return () => clearTimeout(timer);
  }, [query]);

  // Set isSearching to false after search is complete
  useEffect(() => {
    if (debouncedQuery.trim()) {
      const timer = setTimeout(() => setIsSearching(false), 300);
      return () => clearTimeout(timer);
    }
  }, [debouncedQuery]);

  // Update URL when query changes
  useEffect(() => {
    if (debouncedQuery.trim()) {
      const url = new URL(window.location.href);
      url.searchParams.set("q", debouncedQuery);
      window.history.replaceState({}, "", url);
    }
  }, [debouncedQuery]);

  // Enhanced search with relevance scoring
  const searchResults = useMemo(() => {
    if (!debouncedQuery.trim()) return [];

    const results = searchableContent
      .map((item) => ({
        ...item,
        relevanceScore: calculateRelevance(item, debouncedQuery),
      }))
      .filter((item) => item.relevanceScore > 0)
      .sort((a, b) => (b.relevanceScore || 0) - (a.relevanceScore || 0));

    return results;
  }, [debouncedQuery]);

  // Function to highlight the matched query in text
  const highlightMatches = (text: string, query: string) => {
    if (!query.trim()) return text;

    // Escape regex special characters in the query
    const escapedQuery = query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

    // For multi-word queries, create a regex that matches any of the words
    const queryWords = query
      .toLowerCase()
      .trim()
      .split(/\s+/)
      .filter((word) => word.length > 0);
    let regex: RegExp;

    if (queryWords.length > 1) {
      // Match both the full phrase and individual words
      const fullPhrase = escapedQuery;
      const wordsPattern = queryWords
        .map((word) => `\\b${word.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b`)
        .join("|");
      regex = new RegExp(`(${fullPhrase}|${wordsPattern})`, "gi");
    } else {
      // For single word queries, just match that word
      regex = new RegExp(`(${escapedQuery})`, "gi");
    }

    // Split by the regex
    const parts = text.split(regex);

    // Return the highlighted text
    return parts.map((part, i) => {
      // Check if this part matches the query (every other part will match)
      if (i % 2 === 1) {
        return (
          <span
            key={i}
            className="font-semibold px-0.5 rounded"
            style={{
              backgroundColor: "var(--accent)/15",
              color: "var(--accent)",
            }}
          >
            {part}
          </span>
        );
      }
      return part;
    });
  };

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

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    // Update the debounced query immediately when form is submitted
    setDebouncedQuery(query);
  };

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

  return (
    <div className="space-y-12">
      {/* Advanced Search Input */}
      <div className="max-w-4xl mx-auto">
        <form onSubmit={handleSearch} className="relative group">
          <div className="relative">
            <div className="absolute inset-y-0 left-0 pl-6 flex items-center pointer-events-none">
              <svg
                className="w-6 h-6"
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
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search for research articles, technical insights, or projects..."
              className="block w-full pl-16 pr-20 py-5 text-lg border-2 rounded-2xl transition-all duration-300 focus:outline-none focus:ring-4 shadow-lg hover:shadow-xl focus:shadow-2xl"
              style={{
                backgroundColor: "var(--card-bg)",
                borderColor: "var(--border)",
                color: "var(--text-primary)",
              }}
              onFocus={(e) => {
                e.currentTarget.style.borderColor = "var(--accent)";
                e.currentTarget.style.boxShadow = `0 0 0 4px var(--accent)20, 0 20px 25px -5px var(--shadow)`;
              }}
              onBlur={(e) => {
                e.currentTarget.style.borderColor = "var(--border)";
                e.currentTarget.style.boxShadow =
                  "0 10px 15px -3px var(--shadow)";
              }}
            />
            <div className="absolute inset-y-0 right-0 pr-4 flex items-center">
              {isSearching ? (
                <div
                  className="w-8 h-8 border-3 border-t-transparent rounded-full animate-spin"
                  style={{ borderColor: "var(--accent)" }}
                />
              ) : (
                <button
                  type="submit"
                  className="p-3 text-white rounded-xl transition-all duration-300 focus:outline-none focus:ring-4 shadow-lg hover:shadow-xl transform hover:scale-105"
                  style={{
                    backgroundColor: "var(--accent)",
                    boxShadow: "0 4px 12px var(--accent)/30",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor =
                      "var(--accent-hover)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = "var(--accent)";
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
                      strokeWidth="2"
                      d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                    />
                  </svg>
                </button>
              )}
            </div>
          </div>
        </form>{" "}
        {/* Enhanced search suggestions */}
        {!debouncedQuery && (
          <div className="mt-8 text-center">
            <p
              className="text-lg mb-6 font-medium"
              style={{ color: "var(--text-secondary)" }}
            >
              🔬 Popular Research Topics:
            </p>
            <div className="flex flex-wrap justify-center gap-3">
              {popularSuggestions.map((suggestion) => (
                <button
                  key={suggestion}
                  onClick={() => setQuery(suggestion)}
                  className="px-5 py-2.5 text-sm font-medium rounded-full transition-all duration-300 border-2 transform hover:scale-105 hover:shadow-lg"
                  style={{
                    backgroundColor: "var(--surface)",
                    color: "var(--text-secondary)",
                    borderColor: "var(--border)",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor =
                      "var(--surface-accent)";
                    e.currentTarget.style.color = "var(--accent)";
                    e.currentTarget.style.borderColor = "var(--accent)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = "var(--surface)";
                    e.currentTarget.style.color = "var(--text-secondary)";
                    e.currentTarget.style.borderColor = "var(--border)";
                  }}
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Enhanced Results Section */}
      {debouncedQuery && (
        <div className="space-y-8">
          {/* Results header and advanced filters */}
          <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6">
            <div>
              {!isSearching && (
                <div className="space-y-2">
                  <h2
                    className="text-2xl lg:text-3xl font-bold"
                    style={{ color: "var(--text-primary)" }}
                  >
                    {filteredResults.length === 0
                      ? `No results found`
                      : `${filteredResults.length} result${
                          filteredResults.length === 1 ? "" : "s"
                        }`}
                  </h2>
                  <p
                    className="text-lg"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    {filteredResults.length === 0
                      ? `Try different keywords or browse suggested content below`
                      : `for "${debouncedQuery}" • Sorted by relevance`}
                  </p>
                </div>
              )}
              {isSearching && (
                <div className="flex items-center gap-3">
                  <div
                    className="w-6 h-6 border-3 border-t-transparent rounded-full animate-spin"
                    style={{ borderColor: "var(--accent)" }}
                  />
                  <span
                    className="text-lg font-medium"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    Analyzing content...
                  </span>
                </div>
              )}
            </div>

            {/* Enhanced filter buttons */}
            {searchResults.length > 0 && (
              <div className="flex flex-wrap gap-2">
                {Object.entries(filterCounts).map(([filter, count]) => (
                  <button
                    key={filter}
                    onClick={() => setSelectedFilter(filter)}
                    className="px-5 py-3 text-sm font-semibold rounded-xl transition-all duration-300 border-2 transform hover:scale-105"
                    style={{
                      backgroundColor:
                        selectedFilter === filter
                          ? "var(--accent)"
                          : "var(--surface)",
                      color:
                        selectedFilter === filter
                          ? "white"
                          : "var(--text-secondary)",
                      borderColor:
                        selectedFilter === filter
                          ? "var(--accent)"
                          : "var(--border)",
                      boxShadow:
                        selectedFilter === filter
                          ? "0 4px 12px var(--accent)/30"
                          : "none",
                    }}
                    onMouseEnter={(e) => {
                      if (selectedFilter !== filter) {
                        e.currentTarget.style.backgroundColor =
                          "var(--surface-accent)";
                        e.currentTarget.style.color = "var(--accent)";
                        e.currentTarget.style.borderColor = "var(--accent)";
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (selectedFilter !== filter) {
                        e.currentTarget.style.backgroundColor =
                          "var(--surface)";
                        e.currentTarget.style.color = "var(--text-secondary)";
                        e.currentTarget.style.borderColor = "var(--border)";
                      }
                    }}
                  >
                    <div className="flex items-center gap-2">
                      <span>
                        {filter === "all"
                          ? "All"
                          : typeConfig[filter as keyof typeof typeConfig]
                              ?.label}
                      </span>
                      {count > 0 && (
                        <span
                          className="px-2 py-1 text-xs rounded-full font-bold"
                          style={{
                            backgroundColor:
                              selectedFilter === filter
                                ? "rgba(255,255,255,0.2)"
                                : "var(--accent)",
                            color:
                              selectedFilter === filter ? "white" : "white",
                          }}
                        >
                          {count}
                        </span>
                      )}
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Enhanced results grid */}
          {filteredResults.length > 0 && (
            <div className="grid gap-6">
              {filteredResults.map((result, index) => {
                const config = typeConfig[result.type];
                return (
                  <Link
                    key={index}
                    href={result.url}
                    className="group block p-8 rounded-2xl border-2 transition-all duration-300 hover:scale-[1.02] transform hover:shadow-2xl"
                    style={{
                      backgroundColor: "var(--card-bg)",
                      borderColor: "var(--card-border)",
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.borderColor = "var(--accent)";
                      e.currentTarget.style.boxShadow =
                        "0 20px 40px var(--accent)/20";
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.borderColor = "var(--card-border)";
                      e.currentTarget.style.boxShadow = "none";
                    }}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex items-start gap-5 flex-1">
                        <div
                          className="p-4 rounded-xl flex-shrink-0 transition-colors duration-300"
                          style={{
                            backgroundColor: "var(--surface-accent)",
                            color: "var(--accent)",
                          }}
                        >
                          {config.icon}
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-3 mb-3">
                            <span
                              className="text-xs font-bold px-3 py-1.5 rounded-full uppercase tracking-wide"
                              style={{
                                backgroundColor: "var(--surface-accent)",
                                color: "var(--accent)",
                              }}
                            >
                              {config.label}
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
                            {result.relevanceScore &&
                              result.relevanceScore > 200 && (
                                <>
                                  <span
                                    style={{ color: "var(--text-secondary)" }}
                                  >
                                    •
                                  </span>
                                  <span
                                    className="text-xs font-bold px-2 py-1 rounded-full"
                                    style={{
                                      backgroundColor: "var(--accent)",
                                      color: "white",
                                    }}
                                  >
                                    ⭐ Perfect Match
                                  </span>
                                </>
                              )}
                            {result.relevanceScore &&
                              result.relevanceScore > 150 &&
                              result.relevanceScore <= 200 && (
                                <>
                                  <span
                                    style={{ color: "var(--text-secondary)" }}
                                  >
                                    •
                                  </span>
                                  <span
                                    className="text-xs font-bold px-2 py-1 rounded-full"
                                    style={{
                                      backgroundColor: "var(--accent)",
                                      color: "white",
                                    }}
                                  >
                                    ⭐ High Match
                                  </span>
                                </>
                              )}
                            {result.relevanceScore &&
                              result.relevanceScore > 100 &&
                              result.relevanceScore <= 150 && (
                                <>
                                  <span
                                    style={{ color: "var(--text-secondary)" }}
                                  >
                                    •
                                  </span>
                                  <span
                                    className="text-xs font-bold px-2 py-1 rounded-full"
                                    style={{
                                      backgroundColor: "var(--surface-accent)",
                                      color: "var(--accent)",
                                    }}
                                  >
                                    Good Match
                                  </span>
                                </>
                              )}
                          </div>
                          <h3
                            className="text-xl lg:text-2xl font-bold mb-3 transition-colors duration-300 leading-tight"
                            style={{ color: "var(--text-primary)" }}
                            onMouseEnter={(e) => {
                              e.currentTarget.style.color = "var(--accent)";
                            }}
                            onMouseLeave={(e) => {
                              e.currentTarget.style.color =
                                "var(--text-primary)";
                            }}
                          >
                            {highlightMatches(result.title, debouncedQuery)}
                          </h3>
                          <p
                            className="text-base leading-relaxed mb-4"
                            style={{ color: "var(--text-secondary)" }}
                          >
                            {highlightMatches(
                              result.description,
                              debouncedQuery
                            )}
                          </p>
                          {result.tags && (
                            <div className="flex flex-wrap gap-2 mt-4">
                              {result.tags.slice(0, 4).map((tag) => (
                                <span
                                  key={tag}
                                  className="text-xs px-3 py-1.5 rounded-lg font-medium"
                                  style={{
                                    backgroundColor: "var(--surface)",
                                    color: "var(--text-secondary)",
                                  }}
                                >
                                  {tag}
                                </span>
                              ))}
                              {result.tags.length > 4 && (
                                <span
                                  className="text-xs px-3 py-1.5 rounded-lg font-medium"
                                  style={{
                                    backgroundColor: "var(--surface)",
                                    color: "var(--text-secondary)",
                                  }}
                                >
                                  +{result.tags.length - 4} more
                                </span>
                              )}
                            </div>
                          )}
                        </div>
                      </div>
                      <div className="group-hover:translate-x-2 transition-transform duration-300 flex-shrink-0 ml-6">
                        <svg
                          className="w-6 h-6 transition-colors duration-300"
                          style={{ color: "var(--text-secondary)" }}
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                          onMouseEnter={(e) => {
                            e.currentTarget.style.color = "var(--accent)";
                          }}
                          onMouseLeave={(e) => {
                            e.currentTarget.style.color =
                              "var(--text-secondary)";
                          }}
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

          {/* Enhanced No Results State */}
          {query && !isSearching && filteredResults.length === 0 && (
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
                      d="M9.172 16.172a4 4 0 015.656 0M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                    />
                  </svg>
                </div>
                <h3
                  className="text-2xl font-bold mb-3"
                  style={{ color: "var(--text-primary)" }}
                >
                  No Research Found
                </h3>
                <p
                  className="text-lg mb-8 leading-relaxed"
                  style={{ color: "var(--text-secondary)" }}
                >
                  Try different keywords or explore our curated content
                  collections below
                </p>
                <div className="flex flex-col sm:flex-row justify-center gap-4">
                  <Link
                    href="/blog"
                    className="px-6 py-3 font-semibold rounded-xl transition-all duration-300 text-white transform hover:scale-105"
                    style={{
                      backgroundColor: "var(--accent)",
                      boxShadow: "0 4px 12px var(--accent)/30",
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.backgroundColor =
                        "var(--accent-hover)";
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.backgroundColor = "var(--accent)";
                    }}
                  >
                    📚 Browse Research Articles
                  </Link>
                  <Link
                    href="/projects"
                    className="px-6 py-3 font-semibold rounded-xl transition-all duration-300 border-2 transform hover:scale-105"
                    style={{
                      backgroundColor: "var(--surface)",
                      color: "var(--text-primary)",
                      borderColor: "var(--border)",
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.backgroundColor =
                        "var(--surface-accent)";
                      e.currentTarget.style.color = "var(--accent)";
                      e.currentTarget.style.borderColor = "var(--accent)";
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.backgroundColor = "var(--surface)";
                      e.currentTarget.style.color = "var(--text-primary)";
                      e.currentTarget.style.borderColor = "var(--border)";
                    }}
                  >
                    🚀 View Projects
                  </Link>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Enhanced Popular Content Section */}
      {!query && (
        <div className="space-y-8">
          <div className="text-center">
            <h2
              className="text-3xl lg:text-4xl font-bold mb-4"
              style={{ color: "var(--text-primary)" }}
            >
              🔬 Research Highlights
            </h2>
            <p
              className="text-lg lg:text-xl max-w-3xl mx-auto leading-relaxed"
              style={{ color: "var(--text-secondary)" }}
            >
              Discover cutting-edge research, technical insights, and innovative
              projects in AI, machine learning, and software development
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {searchableContent.slice(0, 6).map((item, index) => {
              const config = typeConfig[item.type];
              return (
                <Link
                  key={index}
                  href={item.url}
                  className="group block p-6 rounded-2xl border-2 transition-all duration-300 hover:scale-105 transform hover:shadow-2xl"
                  style={{
                    backgroundColor: "var(--card-bg)",
                    borderColor: "var(--card-border)",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = "var(--accent)";
                    e.currentTarget.style.boxShadow =
                      "0 20px 40px var(--accent)/20";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = "var(--card-border)";
                    e.currentTarget.style.boxShadow = "none";
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
                      {config.icon}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div
                        className="text-xs font-bold px-3 py-1.5 rounded-full inline-block mb-3 uppercase tracking-wide"
                        style={{
                          backgroundColor: "var(--surface-accent)",
                          color: "var(--accent)",
                        }}
                      >
                        {config.label}
                      </div>
                      <h4
                        className="font-bold text-lg leading-snug transition-colors duration-300"
                        style={{ color: "var(--text-primary)" }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.color = "var(--accent)";
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.color = "var(--text-primary)";
                        }}
                      >
                        {item.title}
                      </h4>
                    </div>
                  </div>
                  <p
                    className="text-sm leading-relaxed mb-4"
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
                        className="w-5 h-5 transition-transform duration-300 group-hover:translate-x-1"
                        style={{ color: "var(--text-secondary)" }}
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
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
