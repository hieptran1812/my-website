import Link from "next/link";
import { Metadata } from "next";
import BlogReader from "../../components/BlogReader";
import FadeInWrapper from "@/components/FadeInWrapper";
import { getArticle, getAllBlogSlugs } from "@/lib/getArticle";

export const revalidate = 3600;

export async function generateStaticParams() {
  const slugs = getAllBlogSlugs();
  return slugs.map((slug) => ({ slug }));
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ slug: string[] }>;
}): Promise<Metadata> {
  const { slug } = await params;
  const slugStr = slug.join("/");
  const article = await getArticle(slugStr);

  if (!article) {
    return { title: "Blog Post Not Found" };
  }

  return {
    title: `${article.title} | Hiep Tran`,
    description: article.excerpt || `Read ${article.title} by ${article.author}`,
    keywords: article.tags,
    openGraph: {
      title: article.title,
      description: article.excerpt || "",
      type: "article",
      publishedTime: article.publishDate,
      authors: article.author ? [article.author] : undefined,
      tags: article.tags,
    },
  };
}

export default async function BlogPostPage({
  params,
}: {
  params: Promise<{ slug: string[] }>;
}) {
  const { slug } = await params;
  const slugStr = slug.join("/");
  const article = await getArticle(slugStr);

  if (!article) {
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
            The requested blog post could not be found.
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
      title={article.title}
      publishDate={article.publishDate}
      readTime={article.readTime}
      tags={article.tags}
      category={article.category}
      author={article.author}
      postSlug={article.slug}
      collection={article.collection}
      aiGenerated={article.aiGenerated}
      dangerouslySetInnerHTML={{ __html: article.content }}
    />
  );
}
