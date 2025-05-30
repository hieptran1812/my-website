"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import { useTheme } from "../ThemeProvider";
import "katex/dist/katex.min.css";
import MathJax from "./MathJax";
import "./BlogContent.css";

interface TocItem {
  id: string;
  title: string;
  level: number;
  element: HTMLElement;
}

interface BlogReaderProps {
  children?: React.ReactNode;
  title: string;
  publishDate: string;
  readTime?: string;
  tags?: string[];
  category?: string;
  author?: string;
  dangerouslySetInnerHTML?: { __html: string };
}

export default function BlogReader({
  children,
  title,
  publishDate,
  readTime = "5 min read",
  tags = [],
  category = "Article",
  author = "Hiep Tran",
  dangerouslySetInnerHTML,
}: BlogReaderProps) {
  const [isReadingMode, setIsReadingMode] = useState(false);
  const [fontSize, setFontSize] = useState(16);
  const [lineHeight, setLineHeight] = useState(1.6);
  const [tocItems, setTocItems] = useState<TocItem[]>([]);
  const [activeSection, setActiveSection] = useState<string>("");
  const [showToc, setShowToc] = useState(true);
  const [tocPosition, setTocPosition] = useState<"center" | "top">("center");
  const { theme } = useTheme();
  const contentRef = useRef<HTMLDivElement>(null);
  const observerRef = useRef<IntersectionObserver | null>(null);

  // Save reading preferences to localStorage
  useEffect(() => {
    const savedPreferences = localStorage.getItem("blog-reading-preferences");
    if (savedPreferences) {
      const preferences = JSON.parse(savedPreferences);
      setIsReadingMode(preferences.isReadingMode || false);
      setFontSize(preferences.fontSize || 16);
      setLineHeight(preferences.lineHeight || 1.6);
    }
  }, []);

  useEffect(() => {
    localStorage.setItem(
      "blog-reading-preferences",
      JSON.stringify({
        isReadingMode,
        fontSize,
        lineHeight,
      })
    );
  }, [isReadingMode, fontSize, lineHeight]);

  const handleFontSizeChange = (newSize: number) => {
    setFontSize(Math.max(12, Math.min(24, newSize)));
  };

  const handleLineHeightChange = (newHeight: number) => {
    setLineHeight(Math.max(1.2, Math.min(2.0, newHeight)));
  };

  // Extract headings and create TOC
  const generateToc = useCallback(() => {
    if (!contentRef.current) return;

    const headings = contentRef.current.querySelectorAll(
      "h1, h2, h3, h4, h5, h6"
    );
    const items: TocItem[] = [];

    headings.forEach((heading, index) => {
      const level = parseInt(heading.tagName.charAt(1));
      let id = heading.id;

      // Generate ID if not present
      if (!id) {
        id = `heading-${index}-${
          heading.textContent
            ?.toLowerCase()
            .replace(/[^a-z0-9]+/g, "-")
            .replace(/(^-|-$)/g, "") || ""
        }`;
        heading.id = id;
      }

      items.push({
        id,
        title: heading.textContent || "",
        level,
        element: heading as HTMLElement,
      });
    });

    setTocItems(items);

    // Set up intersection observer for active section highlighting
    if (observerRef.current) {
      observerRef.current.disconnect();
    }

    observerRef.current = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setActiveSection(entry.target.id);
          }
        });
      },
      {
        rootMargin: "-100px 0px -66%",
        threshold: 0,
      }
    );

    headings.forEach((heading) => {
      observerRef.current?.observe(heading);
    });
  }, []);

  // Smooth scroll to section
  const scrollToSection = (id: string) => {
    const element = document.getElementById(id);
    if (element) {
      const yOffset = -100; // Account for fixed header
      const y =
        element.getBoundingClientRect().top + window.pageYOffset + yOffset;

      window.scrollTo({ top: y, behavior: "smooth" });
    }
  };

  // Generate TOC after content is rendered
  useEffect(() => {
    const timer = setTimeout(() => {
      generateToc();
    }, 100);

    return () => {
      clearTimeout(timer);
      if (observerRef.current) {
        observerRef.current.disconnect();
      }
    };
  }, [generateToc]);

  // Handle scroll to prevent TOC from overlapping footer with smooth animations
  useEffect(() => {
    let animationFrame: number;

    const handleScroll = () => {
      const footer = document.querySelector("footer");
      if (!footer) return;

      const footerRect = footer.getBoundingClientRect();
      const viewportHeight = window.innerHeight;
      const currentScrollY = window.scrollY;

      // TOC dimensions and positioning calculations
      const tocHeight = 500; // Approximate TOC height in pixels
      const safetyMargin = 100; // Safety margin to prevent overlap
      const tocCenterPosition = viewportHeight / 2;
      const tocBottomWhenCentered = tocCenterPosition + tocHeight / 2;

      // Calculate footer visibility and potential overlap
      const footerTopFromViewport = footerRect.top;
      const footerIsVisible = footerTopFromViewport < viewportHeight;
      const footerWouldOverlap =
        footerIsVisible &&
        footerTopFromViewport < tocBottomWhenCentered + safetyMargin;

      // Calculate document proximity to bottom
      const documentHeight = document.documentElement.scrollHeight;
      const distanceFromBottom =
        documentHeight - (currentScrollY + viewportHeight);
      const nearBottomThreshold = 300; // Switch when within 300px of bottom

      // Determine TOC position with hysteresis to prevent flickering
      const shouldSwitchToTop =
        footerWouldOverlap || distanceFromBottom <= nearBottomThreshold;

      // Only update if position actually needs to change
      setTocPosition((prev) => {
        if (prev === "center" && shouldSwitchToTop) {
          return "top";
        } else if (
          prev === "top" &&
          !shouldSwitchToTop &&
          distanceFromBottom > nearBottomThreshold + 50
        ) {
          // Add small buffer to prevent flickering when scrolling back up
          return "center";
        }
        return prev;
      });
    };

    // Use requestAnimationFrame for smoother scroll handling
    const throttledHandleScroll = () => {
      if (animationFrame) {
        cancelAnimationFrame(animationFrame);
      }
      animationFrame = requestAnimationFrame(handleScroll);
    };

    // Attach event listeners
    window.addEventListener("scroll", throttledHandleScroll, { passive: true });
    window.addEventListener("resize", handleScroll, { passive: true });

    // Initial check
    handleScroll();

    return () => {
      window.removeEventListener("scroll", throttledHandleScroll);
      window.removeEventListener("resize", handleScroll);
      if (animationFrame) {
        cancelAnimationFrame(animationFrame);
      }
    };
  }, []);

  return (
    <div className="min-h-screen transition-all duration-300">
      {/* Reading Mode Overlay */}
      <div
        className={`fixed inset-0 z-0 transition-all duration-500 ${
          isReadingMode ? "bg-amber-50 dark:bg-amber-900/20" : "bg-transparent"
        }`}
        style={{
          backgroundColor: isReadingMode
            ? theme === "dark"
              ? "#451a03"
              : "#fffbeb"
            : "transparent",
        }}
      />

      {/* Table of Contents - Fixed Left Sidebar */}
      {tocItems.length > 0 && (
        <div
          className={`fixed left-4 z-40 hidden xl:block max-w-xs transition-all duration-700 ease-out ${
            tocPosition === "center"
              ? "top-1/2 -translate-y-1/2"
              : "top-24 translate-y-0"
          }`}
        >
          <div
            className={`p-4 rounded-xl shadow-lg border backdrop-blur-md transition-all duration-300 max-h-[70vh] overflow-y-auto toc-scrollbar ${
              showToc
                ? "opacity-100 translate-x-0"
                : "opacity-0 -translate-x-full"
            }`}
            style={{
              backgroundColor: "var(--background)/95",
              borderColor: "var(--border)",
            }}
          >
            <div className="flex items-center justify-between mb-4">
              <h3
                className="text-sm font-semibold"
                style={{ color: "var(--text-primary)" }}
              >
                Table of Contents
              </h3>
              <button
                onClick={() => setShowToc(!showToc)}
                className="p-1 rounded transition-colors duration-200"
                style={{
                  color: "var(--text-secondary)",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor =
                    "var(--surface-accent)";
                  e.currentTarget.style.color = "var(--accent)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = "transparent";
                  e.currentTarget.style.color = "var(--text-secondary)";
                }}
                aria-label="Toggle table of contents"
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
                    d="M4 6h16M4 12h16M4 18h16"
                  />
                </svg>
              </button>
            </div>

            <nav className="space-y-1">
              {tocItems.map((item) => (
                <button
                  key={item.id}
                  onClick={() => scrollToSection(item.id)}
                  className={`block w-full text-left py-2 px-3 rounded text-sm transition-all duration-200 hover:scale-105 ${
                    activeSection === item.id ? "font-medium" : "font-normal"
                  }`}
                  style={{
                    paddingLeft: `${12 + (item.level - 1) * 12}px`,
                    backgroundColor:
                      activeSection === item.id
                        ? "var(--surface-accent)"
                        : "transparent",
                    color:
                      activeSection === item.id
                        ? "var(--accent)"
                        : "var(--text-secondary)",
                    borderLeft:
                      activeSection === item.id
                        ? `2px solid var(--accent)`
                        : `2px solid transparent`,
                  }}
                  onMouseEnter={(e) => {
                    if (activeSection !== item.id) {
                      e.currentTarget.style.backgroundColor = "var(--surface)";
                      e.currentTarget.style.color = "var(--text-primary)";
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (activeSection !== item.id) {
                      e.currentTarget.style.backgroundColor = "transparent";
                      e.currentTarget.style.color = "var(--text-secondary)";
                    }
                  }}
                  title={item.title}
                >
                  <span className="line-clamp-2 leading-tight">
                    {item.title}
                  </span>
                </button>
              ))}
            </nav>
          </div>
        </div>
      )}

      {/* TOC Toggle Button - Mobile & Tablet */}
      {tocItems.length > 0 && (
        <div className="fixed left-4 top-1/2 -translate-y-1/2 z-40 xl:hidden">
          <button
            onClick={() => setShowToc(!showToc)}
            className="w-10 h-10 rounded-full shadow-lg border transition-all duration-300"
            style={{
              backgroundColor: "var(--background)",
              borderColor: "var(--border)",
              color: "var(--text-secondary)",
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = "var(--surface-accent)";
              e.currentTarget.style.color = "var(--accent)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = "var(--background)";
              e.currentTarget.style.color = "var(--text-secondary)";
            }}
            aria-label="Toggle table of contents"
          >
            <svg
              className="w-5 h-5 mx-auto"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M4 6h16M4 12h16M4 18h16"
              />
            </svg>
          </button>
        </div>
      )}

      {/* Mobile TOC Overlay */}
      {tocItems.length > 0 && showToc && (
        <div className="fixed inset-0 z-50 xl:hidden">
          <div
            className="absolute inset-0 bg-black/50"
            onClick={() => setShowToc(false)}
          />
          <div
            className="absolute left-4 top-1/2 -translate-y-1/2 w-80 max-w-[calc(100vw-2rem)] p-4 rounded-xl shadow-xl border backdrop-blur-md max-h-[70vh] overflow-y-auto toc-scrollbar"
            style={{
              backgroundColor: "var(--background)",
              borderColor: "var(--border)",
            }}
          >
            <div className="flex items-center justify-between mb-4">
              <h3
                className="text-sm font-semibold"
                style={{ color: "var(--text-primary)" }}
              >
                Table of Contents
              </h3>
              <button
                onClick={() => setShowToc(false)}
                className="p-1 rounded transition-colors duration-200"
                style={{
                  color: "var(--text-secondary)",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor =
                    "var(--surface-accent)";
                  e.currentTarget.style.color = "var(--accent)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = "transparent";
                  e.currentTarget.style.color = "var(--text-secondary)";
                }}
                aria-label="Close table of contents"
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
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              </button>
            </div>

            <nav className="space-y-1">
              {tocItems.map((item) => (
                <button
                  key={item.id}
                  onClick={() => {
                    scrollToSection(item.id);
                    setShowToc(false);
                  }}
                  className={`block w-full text-left py-2 px-3 rounded text-sm transition-all duration-200 ${
                    activeSection === item.id ? "font-medium" : "font-normal"
                  }`}
                  style={{
                    paddingLeft: `${12 + (item.level - 1) * 12}px`,
                    backgroundColor:
                      activeSection === item.id
                        ? "var(--surface-accent)"
                        : "transparent",
                    color:
                      activeSection === item.id
                        ? "var(--accent)"
                        : "var(--text-secondary)",
                    borderLeft:
                      activeSection === item.id
                        ? `2px solid var(--accent)`
                        : `2px solid transparent`,
                  }}
                  onMouseEnter={(e) => {
                    if (activeSection !== item.id) {
                      e.currentTarget.style.backgroundColor = "var(--surface)";
                      e.currentTarget.style.color = "var(--text-primary)";
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (activeSection !== item.id) {
                      e.currentTarget.style.backgroundColor = "transparent";
                      e.currentTarget.style.color = "var(--text-secondary)";
                    }
                  }}
                  title={item.title}
                >
                  <span className="leading-tight">{item.title}</span>
                </button>
              ))}
            </nav>
          </div>
        </div>
      )}

      {/* Reading Controls - Fixed Position */}
      <div className="fixed top-20 right-4 z-50 hidden lg:block">
        <div
          className="p-4 rounded-xl shadow-lg border backdrop-blur-md"
          style={{
            backgroundColor: "var(--background)/95",
            borderColor: "var(--border)",
          }}
        >
          <h3
            className="text-sm font-semibold mb-3"
            style={{ color: "var(--text-primary)" }}
          >
            Reading Options
          </h3>

          {/* Reading Mode Toggle */}
          <div className="flex items-center justify-between mb-3">
            <span
              className="text-xs"
              style={{ color: "var(--text-secondary)" }}
            >
              Eye Comfort
            </span>
            <button
              onClick={() => setIsReadingMode(!isReadingMode)}
              className={`relative w-10 h-5 rounded-full transition-colors duration-200 ${
                isReadingMode ? "bg-amber-500" : "bg-gray-300 dark:bg-gray-600"
              }`}
              aria-label="Toggle reading mode"
            >
              <div
                className={`absolute top-0.5 w-4 h-4 bg-white rounded-full transition-transform duration-200 ${
                  isReadingMode ? "translate-x-5" : "translate-x-0.5"
                }`}
              />
            </button>
          </div>

          {/* Font Size Control */}
          <div className="mb-3">
            <div className="flex items-center justify-between mb-1">
              <span
                className="text-xs"
                style={{ color: "var(--text-secondary)" }}
              >
                Font Size
              </span>
              <span
                className="text-xs font-mono"
                style={{ color: "var(--text-primary)" }}
              >
                {fontSize}px
              </span>
            </div>
            <div className="flex items-center gap-1">
              <button
                onClick={() => handleFontSizeChange(fontSize - 2)}
                className="w-6 h-6 rounded text-xs font-bold transition-colors duration-200"
                style={{
                  backgroundColor: "var(--surface)",
                  color: "var(--text-secondary)",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor =
                    "var(--surface-accent)";
                  e.currentTarget.style.color = "var(--accent)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = "var(--surface)";
                  e.currentTarget.style.color = "var(--text-secondary)";
                }}
                aria-label="Decrease font size"
              >
                A-
              </button>
              <div className="flex-1 mx-2">
                <input
                  type="range"
                  min="12"
                  max="24"
                  value={fontSize}
                  onChange={(e) => handleFontSizeChange(Number(e.target.value))}
                  className="w-full h-1 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
                  style={{
                    background: `linear-gradient(to right, var(--accent) 0%, var(--accent) ${
                      ((fontSize - 12) / (24 - 12)) * 100
                    }%, var(--border) ${
                      ((fontSize - 12) / (24 - 12)) * 100
                    }%, var(--border) 100%)`,
                  }}
                />
              </div>
              <button
                onClick={() => handleFontSizeChange(fontSize + 2)}
                className="w-6 h-6 rounded text-xs font-bold transition-colors duration-200"
                style={{
                  backgroundColor: "var(--surface)",
                  color: "var(--text-secondary)",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor =
                    "var(--surface-accent)";
                  e.currentTarget.style.color = "var(--accent)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = "var(--surface)";
                  e.currentTarget.style.color = "var(--text-secondary)";
                }}
                aria-label="Increase font size"
              >
                A+
              </button>
            </div>
          </div>

          {/* Line Height Control */}
          <div>
            <div className="flex items-center justify-between mb-1">
              <span
                className="text-xs"
                style={{ color: "var(--text-secondary)" }}
              >
                Line Spacing
              </span>
              <span
                className="text-xs font-mono"
                style={{ color: "var(--text-primary)" }}
              >
                {lineHeight.toFixed(1)}
              </span>
            </div>
            <div className="flex items-center gap-1">
              <button
                onClick={() => handleLineHeightChange(lineHeight - 0.1)}
                className="w-6 h-6 rounded text-xs font-bold transition-colors duration-200"
                style={{
                  backgroundColor: "var(--surface)",
                  color: "var(--text-secondary)",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor =
                    "var(--surface-accent)";
                  e.currentTarget.style.color = "var(--accent)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = "var(--surface)";
                  e.currentTarget.style.color = "var(--text-secondary)";
                }}
                aria-label="Decrease line height"
              >
                ≡
              </button>
              <div className="flex-1 mx-2">
                <input
                  type="range"
                  min="1.2"
                  max="2.0"
                  step="0.1"
                  value={lineHeight}
                  onChange={(e) =>
                    handleLineHeightChange(Number(e.target.value))
                  }
                  className="w-full h-1 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
                  style={{
                    background: `linear-gradient(to right, var(--accent) 0%, var(--accent) ${
                      ((lineHeight - 1.2) / (2.0 - 1.2)) * 100
                    }%, var(--border) ${
                      ((lineHeight - 1.2) / (2.0 - 1.2)) * 100
                    }%, var(--border) 100%)`,
                  }}
                />
              </div>
              <button
                onClick={() => handleLineHeightChange(lineHeight + 0.1)}
                className="w-6 h-6 rounded text-xs font-bold transition-colors duration-200"
                style={{
                  backgroundColor: "var(--surface)",
                  color: "var(--text-secondary)",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor =
                    "var(--surface-accent)";
                  e.currentTarget.style.color = "var(--accent)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = "var(--surface)";
                  e.currentTarget.style.color = "var(--text-secondary)";
                }}
                aria-label="Increase line height"
              >
                ☰
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Table of Contents - Mobile */}
      <div className="lg:hidden fixed top-20 right-4 z-50">
        <button
          onClick={() => setShowToc(!showToc)}
          className="p-2 rounded-full shadow-md transition-all duration-300"
          style={{
            backgroundColor: "var(--surface)",
            color: "var(--text-secondary)",
            border: "1px solid var(--border)",
          }}
          aria-label="Toggle table of contents"
        >
          <svg
            className="w-6 h-6"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              d="M4 6h16M4 12h16m-7 6h7"
            />
          </svg>
        </button>

        {showToc && tocItems.length > 0 && (
          <div
            className="mt-2 p-4 rounded-xl shadow-lg border backdrop-blur-md"
            style={{
              backgroundColor: "var(--background)/95",
              borderColor: "var(--border)",
            }}
          >
            <h3
              className="text-sm font-semibold mb-3"
              style={{ color: "var(--text-primary)" }}
            >
              Table of Contents
            </h3>
            <div className="flex flex-col gap-2">
              {tocItems.map((item) => (
                <button
                  key={item.id}
                  onClick={() => scrollToSection(item.id)}
                  className={`text-left w-full rounded-lg px-3 py-2 transition-all duration-200 flex items-center gap-2 ${
                    activeSection === item.id
                      ? "bg-amber-100 dark:bg-amber-800"
                      : "hover:bg-gray-100 dark:hover:bg-gray-700"
                  }`}
                  style={{
                    color:
                      activeSection === item.id
                        ? "var(--accent)"
                        : "var(--text-primary)",
                    fontWeight: activeSection === item.id ? "500" : "400",
                  }}
                  aria-label={`Go to section ${item.title}`}
                >
                  <span
                    className="block w-1.5 h-1.5 rounded-full"
                    style={{
                      backgroundColor:
                        activeSection === item.id
                          ? "var(--accent)"
                          : "transparent",
                    }}
                  />
                  <span className="text-sm" style={{}}>
                    {item.title}
                  </span>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Main Content */}
      <div className="relative z-10">
        <div className="max-w-4xl mx-auto px-6 py-8 xl:px-12" ref={contentRef}>
          {/* Article Header */}
          <header className="mb-8">
            <h1
              className="text-4xl lg:text-5xl font-bold mb-6 leading-tight"
              style={{
                color: isReadingMode
                  ? theme === "dark"
                    ? "#fbbf24"
                    : "#92400e"
                  : "var(--text-primary)",
                fontSize: `${fontSize * 1.5}px`,
                lineHeight: lineHeight,
              }}
            >
              {title}
            </h1>

            {/* Article Meta */}
            <div className="flex flex-wrap items-center gap-4 mb-6">
              <div className="flex items-center gap-2">
                <div
                  className="w-8 h-8 rounded-full flex items-center justify-center text-white text-sm font-bold"
                  style={{ backgroundColor: "var(--accent)" }}
                >
                  {author.charAt(0)}
                </div>
                <span
                  className="text-sm font-medium"
                  style={{
                    color: isReadingMode
                      ? theme === "dark"
                        ? "#fbbf24"
                        : "#92400e"
                      : "var(--text-primary)",
                  }}
                >
                  {author}
                </span>
              </div>

              <div className="flex items-center gap-1">
                <svg
                  className="w-4 h-4"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                  style={{
                    color: isReadingMode
                      ? theme === "dark"
                        ? "#d97706"
                        : "#78350f"
                      : "var(--text-secondary)",
                  }}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"
                  />
                </svg>
                <time
                  className="text-sm"
                  style={{
                    color: isReadingMode
                      ? theme === "dark"
                        ? "#d97706"
                        : "#78350f"
                      : "var(--text-secondary)",
                  }}
                >
                  {new Date(publishDate).toLocaleDateString("en-US", {
                    year: "numeric",
                    month: "long",
                    day: "numeric",
                  })}
                </time>
              </div>

              <div className="flex items-center gap-1">
                <svg
                  className="w-4 h-4"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                  style={{
                    color: isReadingMode
                      ? theme === "dark"
                        ? "#d97706"
                        : "#78350f"
                      : "var(--text-secondary)",
                  }}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
                <span
                  className="text-sm"
                  style={{
                    color: isReadingMode
                      ? theme === "dark"
                        ? "#d97706"
                        : "#78350f"
                      : "var(--text-secondary)",
                  }}
                >
                  {readTime}
                </span>
              </div>

              <span
                className="px-3 py-1 rounded-full text-xs font-medium"
                style={{
                  backgroundColor: isReadingMode
                    ? theme === "dark"
                      ? "#78350f"
                      : "#fef3c7"
                    : "var(--surface-accent)",
                  color: isReadingMode
                    ? theme === "dark"
                      ? "#fbbf24"
                      : "#92400e"
                    : "var(--accent)",
                }}
              >
                {category}
              </span>
            </div>

            {/* Tags */}
            {tags.length > 0 && (
              <div className="flex flex-wrap gap-2 mb-6">
                {tags.map((tag, index) => (
                  <span
                    key={index}
                    className="px-2 py-1 rounded text-xs font-medium border transition-colors duration-200"
                    style={{
                      backgroundColor: isReadingMode
                        ? theme === "dark"
                          ? "rgba(120, 53, 15, 0.3)"
                          : "rgba(254, 243, 199, 0.5)"
                        : "var(--surface)",
                      borderColor: isReadingMode
                        ? theme === "dark"
                          ? "#78350f"
                          : "#f3e8ff"
                        : "var(--border)",
                      color: isReadingMode
                        ? theme === "dark"
                          ? "#fbbf24"
                          : "#92400e"
                        : "var(--text-secondary)",
                    }}
                  >
                    #{tag}
                  </span>
                ))}
              </div>
            )}
          </header>

          {/* Article Content */}
          <article
            className="prose prose-lg max-w-none"
            style={{
              fontSize: `${fontSize}px`,
              lineHeight: lineHeight,
              color: isReadingMode
                ? theme === "dark"
                  ? "#fbbf24"
                  : "#92400e"
                : "var(--text-primary)",
            }}
          >
            <MathJax
              className="blog-content"
              style={
                {
                  "--reading-text-color": isReadingMode
                    ? theme === "dark"
                      ? "#fbbf24"
                      : "#92400e"
                    : "var(--text-primary)",
                  "--reading-text-secondary": isReadingMode
                    ? theme === "dark"
                      ? "#d97706"
                      : "#78350f"
                    : "var(--text-secondary)",
                  "--reading-accent": isReadingMode
                    ? theme === "dark"
                      ? "#f59e0b"
                      : "#b45309"
                    : "var(--accent)",
                  "--reading-border": isReadingMode
                    ? theme === "dark"
                      ? "#78350f"
                      : "#fef3c7"
                    : "var(--border)",
                  "--reading-surface": isReadingMode
                    ? theme === "dark"
                      ? "rgba(120, 53, 15, 0.2)"
                      : "rgba(254, 243, 199, 0.3)"
                    : "var(--surface)",
                } as React.CSSProperties
              }
            >
              {dangerouslySetInnerHTML ? (
                <div dangerouslySetInnerHTML={dangerouslySetInnerHTML} />
              ) : (
                children
              )}
            </MathJax>
          </article>
        </div>
      </div>

      {/* Mobile Reading Controls */}
      <div className="lg:hidden fixed bottom-4 right-4 z-50">
        <button
          onClick={() => setIsReadingMode(!isReadingMode)}
          className={`w-12 h-12 rounded-full shadow-lg border-2 transition-all duration-300 ${
            isReadingMode
              ? "bg-amber-500 border-amber-600 text-white"
              : "bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300"
          }`}
          aria-label="Toggle reading mode"
        >
          <svg
            className="w-6 h-6 mx-auto"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z"
            />
          </svg>
        </button>
      </div>
    </div>
  );
}
