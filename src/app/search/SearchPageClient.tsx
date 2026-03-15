"use client";

import SearchComponent from "./SearchComponent";

export default function SearchPageClient() {
  return (
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
          <div className="text-center mb-16 relative">
            {/* Background decoration */}
            <div className="absolute inset-0 -z-10">
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-gradient-to-r from-blue-500/10 via-purple-500/10 to-indigo-500/10 rounded-full blur-3xl"></div>
            </div>

            <h1 className="text-3xl md:text-5xl lg:text-6xl font-black mb-6">
              <span className="bg-gradient-to-r from-blue-600 via-purple-600 to-blue-800 bg-clip-text text-transparent animate-gradient bg-300% font-extrabold tracking-tight">
                Search & Discovery
              </span>
            </h1>

            <p
              className="text-xl md:text-2xl mb-8 max-w-4xl mx-auto leading-relaxed opacity-0 animate-fade-in-up"
              style={{
                color: "var(--text-secondary)",
                animationDelay: "0.3s",
                animationFillMode: "forwards",
              }}
            >
              Find insights across{" "}
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
              .
            </p>
          </div>

          {/* Main Search Content */}
          <div className="mb-20">
            <SearchComponent />
          </div>
        </div>
      </main>
    </div>
  );
}
