"use client";

import React from "react";
import Image from "next/image";
import Link from "next/link";

const articles = [
  {
    title:
      "Fine-tuning Llama 3.2 (1B, 3B) and Using It Locally with Llama Assistant 🌟",
    summary:
      "Hey there, AI enthusiasts! Ready to dive into the exciting world of Llama 3.2? This guide is your ticket to turning this powerful but pint-sized AI model into your very own customized assistant.",
    link: "/blog/llama-3-2-fine-tuning",
    image: "/blog-placeholder.jpg",
    date: "October 6, 2024",
    tags: ["LLM", "Llama Assistant", "Llama"],
    external: true,
  },
  {
    title: "Privacy in AI: Why I Created Llama Assistant",
    summary:
      "Privacy in AI is a hot topic. I created Llama Assistant to provide a privacy-focused alternative to popular AI assistants like ChatGPT or Claude AI.",
    link: "/blog/privacy-in-ai-llama-assistant",
    image: "/blog-placeholder.jpg",
    date: "September 29, 2024",
    tags: ["LLM", "Llama Assistant", "Llama"],
  },
  {
    title: "Performant Django - How to optimize your Django application?",
    summary:
      "Django is a powerful and popular Python web framework known for its ease of use and flexibility. However, as your web application grows in complexity and traffic, it's crucial to optimize it for high performance.",
    link: "/blog/performant-django",
    image: "/blog-placeholder.jpg",
    date: "September 19, 2023",
    tags: ["Backend", "Django"],
  },
  {
    title: "Review YOLO-NAS - Search for a better YOLO",
    summary:
      "A short review of advancements in YOLO-NAS - a new YOLO architecture born from Neural Architecture Search.",
    link: "/blog/yolo-nas-review",
    image: "/blog-placeholder.jpg",
    date: "May 8, 2023",
    tags: ["Object Detection", "YOLO", "Computer Vision"],
  },
];
export default function BlogSection() {
  return (
    <section
      className="py-16 md:py-24 transition-colors duration-300"
      style={{ backgroundColor: "var(--surface)" }}
    >
      <div className="container mx-auto px-6 max-w-6xl">
        <div className="text-center mb-12">
          <h3 className="section-heading text-3xl md:text-4xl lg:text-5xl font-bold mb-4 transition-colors duration-300 relative">
            <span className="bg-gradient-to-r from-emerald-600 via-teal-600 to-cyan-600 bg-clip-text text-transparent">
              Latest Articles
            </span>
            <div className="absolute -bottom-2 left-1/2 transform -translate-x-1/2 w-24 h-1 bg-gradient-to-r from-emerald-600 via-teal-600 to-cyan-600 rounded-full"></div>
          </h3>
          <p
            className="text-lg max-w-2xl mx-auto transition-colors duration-300 mt-6"
            style={{ color: "var(--text-secondary)" }}
          >
            Insights, tutorials, and thoughts on AI, software engineering, and
            technology trends.
          </p>
          <Link
            href="/blog"
            className="inline-flex items-center mt-4 font-medium transition-colors"
            style={{ color: "var(--accent)" }}
            onMouseEnter={(e) => {
              e.currentTarget.style.color = "var(--accent-hover)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.color = "var(--accent)";
            }}
          >
            View All Posts
            <svg
              className="ml-2 w-4 h-4"
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
        <div className="grid md:grid-cols-2 gap-6">
          {articles.map((article, idx) => (
            <Link
              href={article.link}
              key={idx}
              className="group block card-enhanced rounded-xl overflow-hidden hover:border-[var(--accent)] transition-all duration-300"
              style={{
                backgroundColor: "var(--card-bg)",
                borderColor: "var(--card-border)",
              }}
            >
              <div className="relative w-full h-44 overflow-hidden">
                <Image
                  src={article.image}
                  alt={article.title}
                  fill
                  style={{ objectFit: "cover" }}
                  className="transition-transform duration-300 group-hover:scale-105"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent"></div>
                {article.external && (
                  <div className="absolute top-3 right-3">
                    <span className="text-xs bg-black/50 text-white px-2 py-1 rounded-full">
                      External
                    </span>
                  </div>
                )}
              </div>
              <div className="p-5">
                <div className="flex items-center gap-2 mb-2">
                  <span
                    className="text-xs px-2 py-1 rounded-full transition-colors duration-300"
                    style={{ color: "var(--text-muted)" }}
                  >
                    {article.date}
                  </span>
                </div>
                <h4
                  className="text-lg font-semibold mb-2 transition-colors leading-tight line-clamp-2 group-hover:text-[var(--accent)]"
                  style={{ color: "var(--text-primary)" }}
                >
                  {article.title}
                </h4>
                <p
                  className="text-sm mb-3 leading-relaxed transition-colors duration-300 line-clamp-2"
                  style={{ color: "var(--text-secondary)" }}
                >
                  {article.summary}
                </p>
                <div className="flex flex-wrap gap-1.5 mt-auto">
                  {article.tags?.map((tag) => (
                    <span
                      key={tag}
                      className="text-xs px-2 py-1 rounded-full transition-colors duration-300"
                      style={{
                        backgroundColor: "var(--surface-accent)",
                        color: "var(--accent)",
                      }}
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
            </Link>
          ))}
        </div>
      </div>
    </section>
  );
}
