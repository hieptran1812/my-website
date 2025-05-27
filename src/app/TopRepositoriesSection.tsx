"use client";

import React from "react";
import Link from "next/link";

const repos = [
  {
    name: "anylabeling",
    description:
      "Effortless AI-assisted data labeling with SAM(Segment Anything Model)",
    stars: 2682,
    forks: 318,
    language: "Python",
    url: "https://github.com/hieptran1812/anylabeling",
  },
  {
    name: "portfolio-nextjs",
    description: "Modern personal portfolio built with Next.js and TypeScript",
    stars: 156,
    forks: 42,
    language: "TypeScript",
    url: "https://github.com/hieptran1812/portfolio-nextjs",
  },
  {
    name: "react-components",
    description: "Collection of reusable React components with Tailwind CSS",
    stars: 89,
    forks: 23,
    language: "JavaScript",
    url: "https://github.com/hieptran1812/react-components",
  },
  {
    name: "data-structures",
    description: "Implementation of common data structures and algorithms",
    stars: 67,
    forks: 18,
    language: "Python",
    url: "https://github.com/hieptran1812/data-structures",
  },
];

export default function TopRepositoriesSection() {
  return (
    <section
      className="py-16 md:py-24 transition-colors duration-300 section-pattern"
      style={{ backgroundColor: "var(--surface-tertiary)" }}
    >
      <div className="container mx-auto px-6 max-w-6xl">
        <div className="flex items-center justify-between mb-10">
          <h3
            className="text-3xl md:text-4xl font-bold transition-colors duration-300"
            style={{ color: "var(--text-primary)" }}
          >
            Top Repos
          </h3>
          <Link
            href="https://github.com/hieptran1812"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center text-sm font-medium transition-colors duration-200"
            style={{ color: "var(--text-secondary)" }}
            onMouseEnter={(e) => {
              e.currentTarget.style.color = "var(--accent)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.color = "var(--text-secondary)";
            }}
          >
            <span>View all →</span>
          </Link>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          {repos.map((repo) => (
            <Link
              key={repo.name}
              href={repo.url}
              target="_blank"
              rel="noopener noreferrer"
              className="group block p-5 bg-white dark:bg-[#011627] rounded-xl border border-gray-100 dark:border-[#1e3a8a]/20 hover:border-sky-200 dark:hover:border-[#82aaff]/30 transition-all duration-200 hover:shadow-md dark:hover:shadow-[#0e293f]/50"
            >
              <div className="flex items-start justify-between mb-3">
                <h4 className="text-lg font-semibold text-gray-800 dark:text-[#d6deeb] group-hover:text-sky-600 dark:group-hover:text-[#82aaff] transition-colors duration-200">
                  {repo.name}
                </h4>
                <svg
                  className="w-5 h-5 text-gray-400 dark:text-[#5f7e97] group-hover:text-sky-600 dark:group-hover:text-[#82aaff] transition-colors duration-200"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="1.5"
                    d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"
                  />
                </svg>
              </div>

              <p className="text-gray-600 dark:text-[#637777] mb-4 text-sm line-clamp-2 transition-colors duration-200">
                {repo.description}
              </p>

              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  <div className="flex items-center space-x-1.5">
                    <svg
                      className="w-4 h-4 text-yellow-500 dark:text-[#ffcb6b]"
                      fill="currentColor"
                      viewBox="0 0 20 20"
                    >
                      <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                    </svg>
                    <span className="text-gray-600 dark:text-[#637777] text-sm transition-colors duration-200">
                      {repo.stars}
                    </span>
                  </div>

                  <div className="flex items-center space-x-1.5">
                    <svg
                      className="w-4 h-4 text-gray-500 dark:text-[#5f7e97]"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth="1.5"
                        d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.367 2.684 3 3 0 00-5.367-2.684z"
                      />
                    </svg>
                    <span className="text-gray-600 dark:text-[#637777] text-sm transition-colors duration-200">
                      {repo.forks}
                    </span>
                  </div>
                </div>

                <span className="px-2 py-1 bg-sky-50 dark:bg-[#82aaff]/10 text-sky-600 dark:text-[#82aaff] rounded-full text-xs font-medium transition-colors duration-200">
                  {repo.language}
                </span>
              </div>
            </Link>
          ))}
        </div>
      </div>
    </section>
  );
}
