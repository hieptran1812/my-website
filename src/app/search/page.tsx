import { Metadata } from "next";
import { Suspense } from "react";
import SearchPageClient from "./SearchPageClient";

export const metadata: Metadata = {
  title: "Search - Hiep Tran",
  description:
    "Search through blog posts, projects, and content on halleyverse.dev",
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
      "Search through blog posts, projects, and content on halleyverse.dev",
    type: "website",
    url: "https://halleyverse.dev/search",
    images: [
      {
        url: "/api/og?title=Search&description=Find content on halleyverse.dev",
        width: 1200,
        height: 630,
        alt: "Search page for halleyverse.dev",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "Search - Hiep Tran",
    description:
      "Search through blog posts, projects, and content on halleyverse.dev",
    images: [
      "/api/og?title=Search&description=Find content on halleyverse.dev",
    ],
  },
  alternates: {
    canonical: "https://halleyverse.dev/search",
  },
};

export default function SearchPage() {
  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "SearchAction",
    target: {
      "@type": "EntryPoint",
      urlTemplate: "https://halleyverse.dev/search?q={search_term_string}",
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
      <Suspense
        fallback={
          <div className="flex flex-col min-h-screen justify-center items-center">
            <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-500"></div>
            <p className="mt-4 text-lg">Loading search...</p>
          </div>
        }
      >
        <SearchPageClient />
      </Suspense>
    </>
  );
}
