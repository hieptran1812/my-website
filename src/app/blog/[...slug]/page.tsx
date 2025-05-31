"use client";

import { useState, useEffect } from "react";
import { useParams } from "next/navigation";
import BlogReader from "../../components/BlogReader";

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
      <div className="flex flex-col min-h-screen items-center justify-center">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-500"></div>
        <p className="mt-4 text-lg">Loading blog post...</p>
      </div>
    );
  }

  if (error || !post) {
    return (
      <div className="flex flex-col min-h-screen items-center justify-center">
        <h1 className="text-4xl font-bold mb-4">Blog Post Not Found</h1>
        <p className="text-lg text-gray-600">
          {error || "The requested blog post could not be found."}
        </p>
        <a
          href="/blog"
          className="mt-6 px-6 py-2 rounded-lg bg-blue-500 text-white hover:bg-blue-600 transition-colors"
        >
          Back to Blog
        </a>
      </div>
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
