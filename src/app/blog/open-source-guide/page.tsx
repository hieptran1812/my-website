import React from "react";
import Image from "next/image";

export default function OpenSourceGuideArticle() {
  const structuredData = {
    "@context": "https://schema.org",
    "@type": "Article",
    headline: "Open Source: Why and How to Start",
    description:
      "A comprehensive guide to getting started with open source development, from finding projects to making your first contribution.",
    author: {
      "@type": "Person",
      name: "Hiep Tran",
      url: "https://hieptran.dev",
      sameAs: [
        "https://github.com/hieptran1812",
        "https://linkedin.com/in/hieptran1812",
      ],
    },
    publisher: {
      "@type": "Person",
      name: "Hiep Tran",
      url: "https://hieptran.dev",
    },
    datePublished: "2025-04-15",
    dateModified: "2025-04-15",
    mainEntityOfPage: {
      "@type": "WebPage",
      "@id": "https://hieptran.dev/blog/open-source-guide",
    },
    image: {
      "@type": "ImageObject",
      url: "https://hieptran.dev/blog-placeholder.jpg",
      width: 640,
      height: 320,
    },
    articleSection: "Guide",
    keywords: ["Open Source", "Git", "GitHub", "Programming", "Collaboration"],
    wordCount: 250,
    timeRequired: "PT6M",
  };

  return (
    <div
      className="flex flex-col min-h-screen transition-colors duration-300"
      style={{
        backgroundColor: "var(--background)",
        color: "var(--text-primary)",
      }}
    >
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(structuredData) }}
      />
      <main className="flex-1 max-w-4xl mx-auto py-12 px-4">
        <article itemScope itemType="https://schema.org/Article">
          <header className="mb-8">
            <h1 className="text-4xl font-bold mb-4" itemProp="headline">
              Open Source: Why and How to Start
            </h1>
            <div
              className="flex items-center gap-4 text-sm"
              style={{ color: "var(--text-secondary)" }}
            >
              <time
                dateTime="2025-04-15"
                itemProp="datePublished"
                className="flex items-center gap-1"
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
                    d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"
                  />
                </svg>
                April 15, 2025
              </time>
              <span className="flex items-center gap-1">
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
                    d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
                <span itemProp="timeRequired" content="PT6M">
                  6 min read
                </span>
              </span>
              <span className="flex items-center gap-1">
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
                    d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z"
                  />
                </svg>
                <span itemProp="articleSection">Guide</span>
              </span>
            </div>
          </header>

          <div
            className="hidden"
            itemProp="author"
            itemScope
            itemType="https://schema.org/Person"
          >
            <span itemProp="name">Hiep Tran</span>
            <link itemProp="url" href="https://hieptran.dev" />
          </div>

          <Image
            src="/blog-placeholder.jpg"
            alt="Open Source Development Guide - Getting Started with Contributing"
            width={640}
            height={320}
            className="rounded-lg mb-8 object-cover w-full"
            itemProp="image"
          />

          <div
            className="prose dark:prose-invert max-w-none"
            itemProp="articleBody"
          >
            <p>
              Open source is a great way to learn, collaborate, and give back to
              the community. In this article, I share my journey into open
              source and tips for beginners looking to contribute.
            </p>
            <h2>Why Open Source?</h2>
            <ul>
              <li>Learn from real-world codebases</li>
              <li>Collaborate with developers worldwide</li>
              <li>Build your portfolio and reputation</li>
            </ul>
            <h2>How to Start</h2>
            <ol>
              <li>Find a project that interests you</li>
              <li>Read the documentation and contribution guidelines</li>
              <li>Start with small issues or documentation improvements</li>
            </ol>
            <p>Remember, every contribution counts. Happy coding!</p>
          </div>
        </article>
      </main>
    </div>
  );
}
