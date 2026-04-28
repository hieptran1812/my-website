import Link from "next/link";
import { getPopularPosts } from "@/lib/getRelatedPosts";
import RelatedPosts from "@/components/RelatedPosts";

export default function NotFound() {
  const popular = getPopularPosts(6);
  return (
    <main
      className="min-h-screen flex flex-col items-center px-4 sm:px-6 lg:px-8 pt-24 pb-16"
      style={{ backgroundColor: "var(--background)", color: "var(--text-primary)" }}
    >
      <div className="max-w-4xl w-full">
        <div className="text-center mb-12">
          <p
            className="text-sm font-mono mb-3 uppercase tracking-widest"
            style={{ color: "var(--text-secondary)" }}
          >
            404 — Page not found
          </p>
          <h1 className="text-4xl sm:text-5xl font-bold mb-4">
            This page is somewhere else.
          </h1>
          <p
            className="text-lg max-w-xl mx-auto"
            style={{ color: "var(--text-secondary)" }}
          >
            The link may have moved when posts were reorganized into category
            folders, or it never existed. Try one of the recent posts below, or
            search the blog.
          </p>
          <div className="mt-6 flex flex-wrap gap-3 justify-center">
            <Link
              href="/blog"
              className="px-5 py-2.5 rounded-lg font-medium"
              style={{ backgroundColor: "var(--accent)", color: "white" }}
            >
              Browse the blog
            </Link>
            <Link
              href="/"
              className="px-5 py-2.5 rounded-lg font-medium border"
              style={{
                borderColor: "var(--border)",
                color: "var(--text-primary)",
              }}
            >
              Go home
            </Link>
          </div>
          <p
            className="mt-6 text-sm"
            style={{ color: "var(--text-secondary)" }}
          >
            Tip: press{" "}
            <kbd className="cmdk-kbd">⌘</kbd>
            <kbd className="cmdk-kbd">K</kbd> to search anywhere on the site.
          </p>
        </div>

        <RelatedPosts posts={popular} heading="Recent posts" />
      </div>
    </main>
  );
}
