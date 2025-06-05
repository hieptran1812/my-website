"use client";

import { useState, useEffect } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import BlogReader from "../../components/BlogReader";
import FadeInWrapper from "@/components/FadeInWrapper";

interface BlogPost {
  title: string;
  content: string;
  publishDate: string;
  readTime: string;
  tags: string[];
  category: string;
  author: string;
}

export default function BlogPostPage() {
  const params = useParams();
  const [post, setPost] = useState<BlogPost | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchPost() {
      try {
        // Construct slug from URL params - handle both single and nested slugs
        let slug: string;
        if (Array.isArray(params.slug)) {
          slug = params.slug.join("/");
        } else {
          slug = params.slug as string;
        }

        const response = await fetch(
          `/api/blog/article?slug=${encodeURIComponent(slug)}`
        );

        if (!response.ok) {
          throw new Error(`Failed to fetch post: ${response.statusText}`);
        }

        const data = await response.json();

        if (!data || !data.article) {
          throw new Error("Article not found");
        }

        setPost({
          title: data.article.title,
          content: data.article.content,
          publishDate: data.article.date,
          readTime: data.article.readTime,
          tags: data.article.tags,
          category: data.article.category,
          author: data.article.author,
        });
      } catch (err) {
        console.error("Error fetching blog post:", err);
        setError(
          err instanceof Error ? err.message : "Failed to load blog post"
        );
      } finally {
        setIsLoading(false);
      }
    }

    fetchPost();
  }, [params.slug]);

  if (isLoading) {
    return (
      <FadeInWrapper duration={600}>
        <div
          className="flex flex-col min-h-screen items-center justify-center transition-colors duration-300"
          style={{
            backgroundColor: "var(--background)",
            color: "var(--text-primary)",
          }}
        >
          <div
            className="animate-spin rounded-full h-12 w-12 border-b-2 mb-4"
            style={{ borderColor: "var(--accent)" }}
          ></div>
          <p className="text-lg" style={{ color: "var(--text-secondary)" }}>
            Loading blog post...
          </p>
        </div>
      </FadeInWrapper>
    );
  }

  if (error || !post) {
    return (
      <FadeInWrapper duration={600}>
        <div
          className="flex flex-col min-h-screen items-center justify-center transition-colors duration-300"
          style={{
            backgroundColor: "var(--background)",
            color: "var(--text-primary)",
          }}
        >
          <h1
            className="text-4xl font-bold mb-4"
            style={{ color: "var(--text-primary)" }}
          >
            Blog Post Not Found
          </h1>
          <p
            className="text-lg mb-6"
            style={{ color: "var(--text-secondary)" }}
          >
            {error || "The requested blog post could not be found."}
          </p>
          <Link
            href="/blog"
            className="px-6 py-3 rounded-lg transition-colors border"
            style={{
              backgroundColor: "var(--accent)",
              color: "white",
              borderColor: "var(--accent)",
            }}
          >
            Back to Blog
          </Link>
        </div>
      </FadeInWrapper>
    );
  }

  return (
    <BlogReader
      title={post.title}
      publishDate={post.publishDate}
      readTime={post.readTime}
      tags={post.tags}
      category={post.category}
      author={post.author}
      dangerouslySetInnerHTML={{ __html: post.content }}
    />
  );
}
