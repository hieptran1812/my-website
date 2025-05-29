"use client";

import React, { useState, useEffect } from "react";
import { useTheme } from "../ThemeProvider";

interface BlogReaderProps {
  children: React.ReactNode;
  title: string;
  publishDate: string;
  readTime?: string;
  tags?: string[];
  category?: string;
  author?: string;
}

export default function BlogReader({
  children,
  title,
  publishDate,
  readTime = "5 min read",
  tags = [],
  category = "Article",
  author = "Hiep Tran",
}: BlogReaderProps) {
  const [isReadingMode, setIsReadingMode] = useState(false);
  const [fontSize, setFontSize] = useState(16);
  const [lineHeight, setLineHeight] = useState(1.6);
  const { theme } = useTheme();

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

      {/* Main Content */}
      <div className="relative z-10">
        <div className="max-w-4xl mx-auto px-6 py-8">
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
            <div
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
              {children}
            </div>
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
