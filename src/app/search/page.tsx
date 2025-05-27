import { Metadata } from "next";
import SearchComponent from "./SearchComponent";

export const metadata: Metadata = {
  title: "Search - Hiep Tran",
  description:
    "Search through blog posts, projects, and content on hieptran.dev",
  keywords: [
    "Search",
    "Hiep Tran",
    "Blog Search",
    "Project Search",
    "Content Discovery",
    "AI Articles",
    "Machine Learning",
    "Software Development",
  ],
  openGraph: {
    title: "Search - Hiep Tran",
    description:
      "Search through blog posts, projects, and content on hieptran.dev",
    type: "website",
    url: "https://hieptran.dev/search",
    images: [
      {
        url: "/api/og?title=Search&description=Find content on hieptran.dev",
        width: 1200,
        height: 630,
        alt: "Search page for hieptran.dev",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "Search - Hiep Tran",
    description:
      "Search through blog posts, projects, and content on hieptran.dev",
    images: ["/api/og?title=Search&description=Find content on hieptran.dev"],
  },
  alternates: {
    canonical: "https://hieptran.dev/search",
  },
};

export default function SearchPage() {
  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "SearchAction",
    target: {
      "@type": "EntryPoint",
      urlTemplate: "https://hieptran.dev/search?q={search_term_string}",
    },
    "query-input": "required name=search_term_string",
  };

  return (
    <>
      {/* JSON-LD structured data for search */}
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />

      <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-blue-50/30 dark:from-gray-900 dark:via-gray-900 dark:to-blue-950/20">
        {/* Hero Section */}
        <section className="relative py-12 sm:py-16 lg:py-20">
          {/* Background decorations */}
          <div className="absolute inset-0 overflow-hidden">
            <div className="absolute -top-40 -right-32 w-80 h-80 rounded-full bg-gradient-to-br from-blue-400/10 to-purple-400/10 blur-3xl" />
            <div className="absolute -bottom-40 -left-32 w-80 h-80 rounded-full bg-gradient-to-br from-emerald-400/10 to-blue-400/10 blur-3xl" />
          </div>

          <div className="container mx-auto px-6 relative z-10">
            <div className="max-w-4xl mx-auto text-center">
              {/* Badge */}
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-blue-50 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-800 mb-6">
                <svg
                  className="w-4 h-4 text-blue-600 dark:text-blue-400"
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
                <span className="text-sm font-medium text-blue-700 dark:text-blue-300">
                  Search & Discovery
                </span>
              </div>

              {/* Title */}
              <h1 className="text-3xl sm:text-4xl lg:text-5xl xl:text-6xl font-bold mb-6 bg-gradient-to-r from-gray-900 via-blue-700 to-gray-900 dark:from-white dark:via-blue-300 dark:to-white bg-clip-text text-transparent leading-tight">
                Find What You Need
              </h1>

              {/* Description */}
              <p className="text-lg sm:text-xl text-gray-600 dark:text-gray-300 leading-relaxed max-w-3xl mx-auto">
                Search through my collection of articles, projects, and insights
                on AI, machine learning, and software development.
              </p>
            </div>
          </div>
        </section>

        {/* Main Search Content */}
        <section className="pb-16 lg:pb-24">
          <div className="container mx-auto px-6">
            <div className="max-w-6xl mx-auto">
              <SearchComponent />
            </div>
          </div>
        </section>
      </div>
    </>
  );
}
