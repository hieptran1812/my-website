import React from "react";
import Link from "next/link";
import Image from "next/image";
import { Article } from "@/lib/blog";
import FadeInWrapper from "./FadeInWrapper";
import CollectionTag from "./CollectionTag";
import { TagList } from "./TagBadge";
import { formatDateShort, formatDateMedium } from "@/lib/dateUtils";

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
        <Link href={`/blog/${article.slug}`} className="block">
          <article
            className="group rounded-2xl overflow-hidden border transition-all duration-300 hover:shadow-xl hover:scale-[1.02] cursor-pointer"
            style={{
              backgroundColor: "var(--surface)",
              borderColor: "var(--border)",
              background:
                "linear-gradient(145deg, var(--surface), var(--surface-hover))",
            }}
          >
            {/* Image Section */}
            <div className="relative w-full h-48 overflow-hidden">
              <Image
                src={
                  article.image &&
                  article.image.trim() !== "" &&
                  article.image !== "/blog-placeholder.jpg" &&
                  article.image !== "/images/default-blog.jpg"
                    ? article.image
                    : "/blog-placeholder.jpg"
                }
                alt={article.title}
                fill
                className="object-cover transition-transform duration-300 group-hover:scale-105"
                sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent"></div>
              <div className="absolute top-4 right-4">
                <span className="px-3 py-1 text-xs font-bold rounded-full bg-[var(--accent)] text-white shadow-lg">
                  ✨ Featured
                </span>
              </div>
            </div>

            <div className="p-8">
              {/* Collection tag at the top */}
              {article.collection && (
                <div className="mb-4">
                  <CollectionTag
                    collection={article.collection}
                    variant="detailed"
                    clickable={false}
                  />
                </div>
              )}

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

              <h3
                className="text-2xl md:text-3xl font-bold mb-4 group-hover:text-[var(--accent)] transition-colors"
                style={{ color: "var(--text-primary)" }}
              >
                {article.title}
              </h3>

              {/* Tags under title */}
              {article.tags && article.tags.length > 0 && (
                <div className="mb-4" onClick={(e) => e.stopPropagation()}>
                  <TagList
                    tags={article.tags}
                    variant="default"
                    clickable={true}
                  />
                </div>
              )}

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
                <button
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
                </button>
              </div>
            </div>
          </article>
        </Link>
      </FadeInWrapper>
    );
  }

  if (variant === "compact") {
    return (
      <FadeInWrapper delay={delay} duration={600} direction="up">
        <Link href={`/blog/${article.slug}`} className="block">
          <article
            className="group rounded-lg border transition-all duration-300 hover:shadow-lg hover:scale-[1.02] overflow-hidden cursor-pointer"
            style={{
              backgroundColor: "var(--surface)",
              borderColor: "var(--border)",
            }}
          >
            {/* Image Section */}
            <div className="relative w-full h-32 overflow-hidden">
              <Image
                src={
                  article.image &&
                  article.image.trim() !== "" &&
                  article.image !== "/blog-placeholder.jpg" &&
                  article.image !== "/images/default-blog.jpg"
                    ? article.image
                    : "/blog-placeholder.jpg"
                }
                alt={article.title}
                fill
                className="object-cover transition-transform duration-300 group-hover:scale-105"
                sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 25vw"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent"></div>
            </div>

            <div className="p-4">
              {/* Collection tag at the top */}
              {article.collection && (
                <div className="mb-3">
                  <CollectionTag
                    collection={article.collection}
                    variant="compact"
                    clickable={false}
                  />
                </div>
              )}

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
                <span
                  className="text-xs"
                  style={{ color: "var(--text-muted)" }}
                >
                  {article.readTime}
                </span>
              </div>

              <h3
                className="text-base font-semibold mb-2 group-hover:text-[var(--accent)] transition-colors line-clamp-2"
                style={{ color: "var(--text-primary)" }}
              >
                {article.title}
              </h3>

              {/* Tags under title */}
              {article.tags && article.tags.length > 0 && (
                <div className="mb-3" onClick={(e) => e.stopPropagation()}>
                  <TagList
                    tags={article.tags}
                    maxTags={2}
                    variant="compact"
                    clickable={true}
                    showMoreCount={false}
                  />
                </div>
              )}

              <p
                className="text-sm mb-3 leading-relaxed line-clamp-2"
                style={{ color: "var(--text-secondary)" }}
              >
                {article.excerpt}
              </p>

              <div
                className="flex items-center justify-between text-xs"
                style={{ color: "var(--text-muted)" }}
              >
                <span>📅 {formatDateShort(article.date)}</span>
                <span className="inline-flex items-center text-[var(--accent)] hover:scale-105 transition-transform">
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
                </span>
              </div>
            </div>
          </article>
        </Link>
      </FadeInWrapper>
    );
  }

  // Default variant
  return (
    <FadeInWrapper delay={delay} duration={600} direction="up">
      <Link href={`/blog/${article.slug}`} className="block">
        <article
          className="group rounded-xl border transition-all duration-300 hover:shadow-xl hover:scale-[1.02] overflow-hidden cursor-pointer"
          style={{
            backgroundColor: "var(--surface)",
            borderColor: "var(--border)",
          }}
        >
          {/* Image Section */}
          <div className="relative w-full h-40 overflow-hidden">
            <Image
              src={
                article.image &&
                article.image.trim() !== "" &&
                article.image !== "/blog-placeholder.jpg" &&
                article.image !== "/images/default-blog.jpg"
                  ? article.image
                  : "/blog-placeholder.jpg"
              }
              alt={article.title}
              fill
              className="object-cover transition-transform duration-300 group-hover:scale-105"
              sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
            />
            <div className="absolute inset-0 bg-gradient-to-t from-black/30 to-transparent"></div>
          </div>

          <div className="p-6">
            {/* Collection tag at the top */}
            {article.collection && (
              <div className="mb-4">
                <CollectionTag
                  collection={article.collection}
                  variant="default"
                  clickable={false}
                />
              </div>
            )}

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

            {/* Tags under title */}
            {article.tags && article.tags.length > 0 && (
              <div className="mb-3" onClick={(e) => e.stopPropagation()}>
                <TagList
                  tags={article.tags}
                  maxTags={3}
                  variant="compact"
                  clickable={true}
                />
              </div>
            )}

            <p
              className="text-sm mb-4 leading-relaxed line-clamp-3"
              style={{ color: "var(--text-secondary)" }}
            >
              {article.excerpt}
            </p>

            <div
              className="flex items-center justify-between text-xs mt-4"
              style={{ color: "var(--text-muted)" }}
            >
              <span>📅 {formatDateMedium(article.date)}</span>
              <span className="inline-flex items-center text-[var(--accent)] hover:scale-105 transition-transform">
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
              </span>
            </div>
          </div>
        </article>
      </Link>
    </FadeInWrapper>
  );
}
