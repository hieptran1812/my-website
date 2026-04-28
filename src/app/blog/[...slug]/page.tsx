import Link from "next/link";
import { Metadata } from "next";
import BlogReader from "../../components/BlogReader";
import FadeInWrapper from "@/components/FadeInWrapper";
import { getArticle, getAllBlogSlugs } from "@/lib/getArticle";
import { getRelatedPosts, getSeriesContext } from "@/lib/getRelatedPosts";
import RelatedPosts from "@/components/RelatedPosts";
import SeriesModule from "@/components/SeriesModule";
import CodeBlockEnhancer from "@/components/CodeBlockEnhancer";
import { getPostCoverUrl } from "@/lib/getPostCover";

const SITE_URL = "https://halleyverse.dev";

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

  const coverPath = getPostCoverUrl(article.slug, article.image);
  const coverAbsolute = coverPath.startsWith("http")
    ? coverPath
    : `${SITE_URL}${coverPath}`;
  const ogImage = {
    url: coverAbsolute,
    width: 1200,
    height: 675,
    alt: article.title,
  };

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
      images: [ogImage],
    },
    twitter: {
      card: "summary_large_image",
      title: article.title,
      description: article.excerpt || "",
      images: [coverAbsolute],
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

  // The corpus index inside getRelatedPosts/getSeriesContext is the source of
  // truth for tags / category / subcategory — no need to recompute here.
  const [series, related] = await Promise.all([
    getSeriesContext(article.slug),
    getRelatedPosts(article.slug, [], "", "", 6),
  ]);

  return (
    <>
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
        footer={
          <>
            <SeriesModule ctx={series} />
            <RelatedPosts posts={related} />
          </>
        }
      />
      <CodeBlockEnhancer />
    </>
  );
}
