import React from "react";
import Image from "next/image";
import Navigation from "../../components/Navigation";
import Footer from "../../Footer";

export default function ModernPortfolioNextjsArticle() {
  const structuredData = {
    "@context": "https://schema.org",
    "@type": "Article",
    headline: "How to Build a Modern Portfolio with Next.js",
    description:
      "A comprehensive guide to creating a personal portfolio website using Next.js, TypeScript, and Tailwind CSS with modern design patterns.",
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
    datePublished: "2025-05-01",
    dateModified: "2025-05-01",
    mainEntityOfPage: {
      "@type": "WebPage",
      "@id": "https://hieptran.dev/blog/modern-portfolio-nextjs",
    },
    image: {
      "@type": "ImageObject",
      url: "https://hieptran.dev/blog-placeholder.jpg",
      width: 640,
      height: 320,
    },
    articleSection: "Tutorial",
    keywords: [
      "Next.js",
      "TypeScript",
      "Tailwind CSS",
      "Portfolio",
      "Web Development",
    ],
    wordCount: 300,
    timeRequired: "PT8M",
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
      <Navigation />
      <main className="flex-1 max-w-4xl mx-auto py-12 px-4">
        <article itemScope itemType="https://schema.org/Article">
          <header className="mb-8">
            <h1 className="text-4xl font-bold mb-4" itemProp="headline">
              How to Build a Modern Portfolio with Next.js
            </h1>
            <div
              className="flex items-center gap-4 text-sm"
              style={{ color: "var(--text-secondary)" }}
            >
              <time
                dateTime="2025-05-01"
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
                May 1, 2025
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
                <span itemProp="timeRequired" content="PT8M">
                  8 min read
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
                <span itemProp="articleSection">Tutorial</span>
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
            alt="Modern Portfolio with Next.js - Tutorial Preview"
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
              This article guides you through building a personal portfolio
              website using Next.js, TypeScript, and Tailwind CSS. You&apos;ll
              learn about project structure, modular React components, and
              deploying your site.
            </p>
            <h2>Getting Started</h2>
            <p>
              Start by creating a new Next.js project and installing Tailwind
              CSS. Organize your code into reusable components for each section
              of your portfolio.
            </p>
            <h2>Key Features</h2>
            <ul>
              <li>Hero section with intro and profile image</li>
              <li>Latest projects and open source highlights</li>
              <li>Blog/articles with previews</li>
              <li>Contact and collaboration form</li>
            </ul>
            <h2>Deployment</h2>
            <p>
              Deploy your site easily with Vercel or your preferred hosting
              provider. Enjoy your new modern portfolio!
            </p>
          </div>
        </article>
      </main>
      <Footer />
    </div>
  );
}
