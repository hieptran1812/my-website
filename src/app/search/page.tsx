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

      <div
        className="flex flex-col min-h-screen transition-colors duration-300"
        style={{
          backgroundColor: "var(--background)",
          color: "var(--text-primary)",
        }}
      >
        <main className="flex-1">
          <div className="max-w-7xl mx-auto px-6 py-16">
            {/* Hero Section */}
            <div className="text-center mb-20 relative">
              {/* Background decoration */}
              <div className="absolute inset-0 -z-10">
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-gradient-to-r from-blue-500/10 via-purple-500/10 to-indigo-500/10 rounded-full blur-3xl animate-pulse"></div>
                <div className="absolute top-1/3 left-1/3 w-64 h-64 bg-gradient-to-br from-cyan-400/5 to-blue-600/5 rounded-full blur-2xl animate-pulse delay-700"></div>
                <div className="absolute bottom-1/3 right-1/3 w-48 h-48 bg-gradient-to-tl from-purple-400/5 to-pink-600/5 rounded-full blur-2xl animate-pulse delay-1000"></div>
              </div>

              {/* Animated title with enhanced effects */}
              <div className="relative inline-block group">
                <h1 className="text-3xl md:text-5xl lg:text-6xl font-black mb-6 relative overflow-hidden">
                  {/* Main gradient text */}
                  <span className="relative z-10 bg-gradient-to-r from-blue-600 via-purple-600 to-blue-800 bg-clip-text text-transparent animate-gradient bg-300% font-extrabold tracking-tight">
                    Search & Discovery
                  </span>

                  {/* Glowing shadow effect */}
                  <span className="absolute inset-0 bg-gradient-to-r from-blue-600 via-purple-600 to-blue-800 bg-clip-text text-transparent blur-sm opacity-30 scale-105">
                    Search & Discovery
                  </span>

                  {/* Sparkle decorations */}
                  <div className="absolute -top-4 left-8 w-2 h-2 bg-blue-400 rounded-full animate-pulse opacity-60"></div>
                  <div className="absolute -top-2 right-12 w-1.5 h-1.5 bg-purple-400 rounded-full animate-pulse delay-300 opacity-80"></div>
                  <div className="absolute top-4 left-4 w-1 h-1 bg-indigo-400 rounded-full animate-pulse delay-500 opacity-70"></div>
                  <div className="absolute -bottom-2 right-8 w-2 h-2 bg-cyan-400 rounded-full animate-pulse delay-700 opacity-60"></div>

                  {/* Animated underline */}
                  <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-0 h-1 bg-gradient-to-r from-blue-500 via-purple-500 to-indigo-500 rounded-full group-hover:w-full transition-all duration-1000 ease-out"></div>
                </h1>

                {/* Floating particles */}
                <div className="absolute inset-0 pointer-events-none">
                  <div className="absolute top-1/4 left-1/4 w-1 h-1 bg-blue-300 rounded-full animate-bounce opacity-40"></div>
                  <div className="absolute top-3/4 right-1/4 w-1 h-1 bg-purple-300 rounded-full animate-bounce delay-300 opacity-50"></div>
                  <div className="absolute bottom-1/4 left-3/4 w-1 h-1 bg-indigo-300 rounded-full animate-bounce delay-500 opacity-30"></div>
                </div>
              </div>

              {/* Enhanced subtitle */}
              <div className="relative">
                <p
                  className="text-xl md:text-2xl mb-8 max-w-4xl mx-auto leading-relaxed opacity-0 animate-fade-in-up"
                  style={{
                    color: "var(--text-secondary)",
                    animationDelay: "0.5s",
                    animationFillMode: "forwards",
                  }}
                >
                  Discover insights in{" "}
                  <span className="font-semibold bg-gradient-to-r from-blue-500 to-purple-600 bg-clip-text text-transparent">
                    artificial intelligence
                  </span>
                  ,{" "}
                  <span className="font-semibold bg-gradient-to-r from-purple-500 to-indigo-600 bg-clip-text text-transparent">
                    machine learning
                  </span>
                  , and{" "}
                  <span className="font-semibold bg-gradient-to-r from-indigo-500 to-blue-600 bg-clip-text text-transparent">
                    software development
                  </span>
                  . Search through my comprehensive collection of research
                  articles, project documentation, and technical insights.
                </p>
              </div>

              {/* Enhanced search stats badges */}
              <div
                className="flex flex-wrap justify-center gap-4 opacity-0 animate-fade-in-up"
                style={{ animationDelay: "1s", animationFillMode: "forwards" }}
              >
                <span
                  className="group px-6 py-3 rounded-full text-sm font-semibold transition-all duration-300 hover:scale-110 hover:shadow-lg cursor-default border"
                  style={{
                    backgroundColor: "var(--surface-accent)",
                    color: "var(--accent)",
                    borderColor: "var(--accent)/20",
                  }}
                >
                  🔍 Advanced Search
                </span>
                <span
                  className="group px-6 py-3 rounded-full text-sm font-semibold transition-all duration-300 hover:scale-110 hover:shadow-lg cursor-default border"
                  style={{
                    backgroundColor: "var(--surface-accent)",
                    color: "var(--accent)",
                    borderColor: "var(--accent)/20",
                  }}
                >
                  📚 Research Content
                </span>
                <span
                  className="group px-6 py-3 rounded-full text-sm font-semibold transition-all duration-300 hover:scale-110 hover:shadow-lg cursor-default border"
                  style={{
                    backgroundColor: "var(--surface-accent)",
                    color: "var(--accent)",
                    borderColor: "var(--accent)/20",
                  }}
                >
                  🎯 Precise Results
                </span>
              </div>
            </div>

            {/* Main Search Content */}
            <div className="mb-20">
              <SearchComponent />
            </div>
          </div>
        </main>
      </div>
    </>
  );
}
