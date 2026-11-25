import fs from "fs";
import path from "path";
import { cache } from "react";
import matter from "gray-matter";
import { remark } from "remark";
import remarkGfm from "remark-gfm";
import remarkRehype from "remark-rehype";
import rehypeHighlight from "rehype-highlight";
import rehypeStringify from "rehype-stringify";
import { calculateReadTimeWithTags } from "./readTimeCalculator";
import { protectMathBlocks, restoreMathBlocks } from "./markdown";
import remarkCallouts from "./remarkCallouts";
import rehypeImageOptimize from "./rehypeImageOptimize";
import { resolvePostCover } from "./getRelatedPosts";
import { applyPaywallToFile } from "./paywall";
import { readPostRaw, slugToRelPath } from "./contentSource";

export interface ArticleData {
  title: string;
  content: string;
  publishDate: string;
  readTime: string;
  tags: string[];
  category: string;
  author: string;
  slug: string;
  collection?: string;
  substackUrl?: string;
  excerpt?: string;
  image?: string;
  /** True when `content` is only the teaser — see src/lib/paywall.ts. */
  paywalled: boolean;
}

const blogDir = path.join(process.cwd(), "content", "blog");

/**
 * Reads and renders a single article. Wrapped in React.cache() below so that
 * the detail page's generateMetadata() and the page component — which both
 * request the same slug within one render pass — share a single file read +
 * markdown parse instead of doing the full remark/rehype pipeline twice.
 */
async function getArticleImpl(slug: string): Promise<ArticleData | null> {
  // Disk in dev/build, R2 in serverless functions (see contentSource.ts).
  const rel = slugToRelPath(slug);
  if (!rel) return null;
  const fileContent = await readPostRaw(rel);
  if (fileContent === null) return null;
  const articlePath = path.join(blogDir, ...rel.split("/"));
  const slugParts = rel.replace(/\.md$/, "").split("/");

  const { data: metadata, content: markdownContent } = matter(fileContent);

  const { protectedContent, mathBlocks } = protectMathBlocks(markdownContent);

  const processedContent = await remark()
    .use(remarkGfm)
    .use(remarkCallouts)
    .use(remarkRehype, { allowDangerousHtml: true })
    .use(rehypeHighlight)
    .use(rehypeImageOptimize)
    .use(rehypeStringify, { allowDangerousHtml: true })
    .process(protectedContent);

  const renderedHtml = restoreMathBlocks(
    processedContent.toString(),
    mathBlocks,
  );

  // Gate on the resolved path, not the requested slug: `articlePath` is where
  // the post actually lives, so `..` segments in the slug cannot open it.
  const { html: htmlContent, paywalled } = applyPaywallToFile(
    articlePath,
    renderedHtml,
  );

  let category = metadata.category;
  if (!category && slugParts.length > 1) {
    category = slugParts[0];
  }

  const readTimeResult = calculateReadTimeWithTags(
    markdownContent,
    metadata.tags || [],
    category || "General",
  );

  return {
    title: metadata.title || "Untitled",
    content: htmlContent,
    publishDate: metadata.publishDate || metadata.date || "",
    readTime: readTimeResult.readTime,
    tags: metadata.tags || [],
    category: category || "",
    author: metadata.author || "",
    slug,
    collection: metadata.collection,
    substackUrl:
      typeof metadata.substackUrl === "string"
        ? metadata.substackUrl.trim() || undefined
        : typeof metadata.substack === "string"
          ? metadata.substack.trim() || undefined
          : undefined,
    excerpt: metadata.excerpt || metadata.description || "",
    image: resolvePostCover(metadata, markdownContent),
    paywalled,
  };
}

/** Per-request memoized article reader — see getArticleImpl. */
export const getArticle = cache(getArticleImpl);

/** Enumerate all blog slugs for static generation */
export function getAllBlogSlugs(): string[][] {
  const slugs: string[][] = [];

  function walk(dir: string, prefix: string[]) {
    if (!fs.existsSync(dir)) return;
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    for (const entry of entries) {
      if (entry.isDirectory()) {
        walk(path.join(dir, entry.name), [...prefix, entry.name]);
      } else if (entry.name.endsWith(".md")) {
        slugs.push([...prefix, entry.name.replace(/\.md$/, "")]);
      }
    }
  }

  walk(blogDir, []);
  return slugs;
}
