import React from "react";
import Link from "next/link";
import { Article } from "@/lib/blog";
import FadeInWrapper from "./FadeInWrapper";

interface ArticleCardProps {
  article: Article;
  index: number;
  variant?: "default" | "featured" | "compact";
}

export default function ArticleCard({
  article,
  index,
  variant = "default",
}: ArticleCardProps) {
  const delay = index * 100; // Stagger animation by 100ms per card

  if (variant === "featured") {
    return (
      <FadeInWrapper delay={delay} duration={800} direction="up">
        <article
          className="group rounded-2xl p-8 border transition-all duration-300 hover:shadow-xl hover:scale-[1.02]"
          style={{
            backgroundColor: "var(--surface)",
            borderColor: "var(--border)",
            background:
              "linear-gradient(145deg, var(--surface), var(--surface-hover))",
          }}
        >
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
                Reading Time
              </div>
              <div
                className="font-semibold"
                style={{ color: "var(--text-primary)" }}
              >
                {article.readTime}
              </div>
            </div>
            <div>
              <div
                className="text-sm font-medium mb-1"
                style={{ color: "var(--text-secondary)" }}
              >
                Difficulty
              </div>
              <div
                className="font-semibold"
                style={{ color: "var(--text-primary)" }}
              >
                {article.difficulty || "Intermediate"}
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
            className="text-2xl md:text-3xl font-bold mb-4 group-hover:text-[var(--accent)] transition-colors"
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
              <span>✨ Featured</span>
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
        </article>
      </FadeInWrapper>
    );
  }

  if (variant === "compact") {
    return (
      <FadeInWrapper delay={delay} duration={600} direction="up">
        <article
          className="group rounded-lg border transition-all duration-300 hover:shadow-lg hover:scale-[1.02] overflow-hidden"
          style={{
            backgroundColor: "var(--surface)",
            borderColor: "var(--border)",
          }}
        >
          <div className="p-4">
            <div className="flex items-center justify-between mb-3">
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
              <span className="text-xs" style={{ color: "var(--text-muted)" }}>
                {article.readTime}
              </span>
            </div>

            <h3
              className="text-base font-semibold mb-2 group-hover:text-[var(--accent)] transition-colors line-clamp-2"
              style={{ color: "var(--text-primary)" }}
            >
              {article.title}
            </h3>

            <p
              className="text-sm mb-3 leading-relaxed line-clamp-2"
              style={{ color: "var(--text-secondary)" }}
            >
              {article.excerpt}
            </p>

            <div className="flex flex-wrap gap-1 mb-3">
              {article.tags.slice(0, 2).map((tag) => (
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
              className="flex items-center justify-between text-xs"
              style={{ color: "var(--text-muted)" }}
            >
              <span>
                📅{" "}
                {new Date(article.date).toLocaleDateString("en-US", {
                  month: "short",
                  day: "numeric",
                })}
              </span>
              <Link
                href={`/blog/${article.slug}`}
                className="inline-flex items-center text-[var(--accent)] hover:scale-105 transition-transform"
              >
                Read
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
      </FadeInWrapper>
    );
  }

  // Default variant
  return (
    <FadeInWrapper delay={delay} duration={600} direction="up">
      <article
        className="group rounded-xl border transition-all duration-300 hover:shadow-xl hover:scale-[1.02] overflow-hidden"
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
            <span className="text-xs" style={{ color: "var(--text-muted)" }}>
              {article.readTime}
            </span>
          </div>

          <h3
            className="text-lg font-semibold mb-3 group-hover:text-[var(--accent)] transition-colors line-clamp-2"
            style={{ color: "var(--text-primary)" }}
          >
            {article.title}
          </h3>

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
              className="inline-flex items-center text-[var(--accent)] hover:scale-105 transition-transform"
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
    </FadeInWrapper>
  );
}
