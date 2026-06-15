/**
 * Lightweight blog metadata index — the single source the listing endpoints
 * read from. One presorted array of card-sized metadata (no markdown body).
 *
 * Resolution order:
 *   1. Production + a non-empty generated index → use it (parse one JSON,
 *      no corpus walk). `scripts/buildBlogIndex.ts` produces it per deploy.
 *   2. Otherwise (dev, or missing/empty file) → build at runtime from
 *      loadAllPosts() and memoize against the corpus reference, so freshly
 *      added posts appear without a rebuild.
 *
 * Either way the result is the same shape, sorted newest-first, and cached in
 * module scope for the life of the process.
 */

import fs from "fs";
import path from "path";
import { loadAllPosts } from "./blogIndex";
import { calculateReadTimeWithTags } from "./readTimeCalculator";

export interface BlogPostLite {
  slug: string;
  title: string;
  excerpt: string;
  publishDate: string;
  readTime: string;
  category: string;
  subcategory: string;
  author: string;
  tags: string[];
  image?: string;
  collection?: string;
  featured: boolean;
}

export interface Pagination {
  page: number;
  limit: number;
  total: number;
  totalPages: number;
  hasMore: boolean;
}

const GENERATED_PATH = path.join(
  process.cwd(),
  "src",
  "lib",
  "generated",
  "blogPostsIndex.json",
);

// Generated index (read once). `undefined` = not yet attempted.
let generatedCache: BlogPostLite[] | null | undefined;
// Runtime-built index, memoized against the corpus array reference.
type Corpus = Awaited<ReturnType<typeof loadAllPosts>>;
let builtCorpusRef: Corpus | null = null;
let builtPosts: BlogPostLite[] = [];

function loadGenerated(): BlogPostLite[] | null {
  if (generatedCache !== undefined) return generatedCache;
  // In dev we always prefer the runtime walk so new/edited posts show instantly.
  if (process.env.NODE_ENV !== "production") {
    generatedCache = null;
    return null;
  }
  try {
    const raw = fs.readFileSync(GENERATED_PATH, "utf8");
    const parsed = JSON.parse(raw) as BlogPostLite[];
    generatedCache = Array.isArray(parsed) && parsed.length > 0 ? parsed : null;
  } catch {
    generatedCache = null;
  }
  return generatedCache;
}

function deriveExcerpt(content: string): string {
  if (!content) return "";
  let out = content.split("\n\n")[0].substring(0, 200).trim();
  if (content.length > 200) out += "...";
  return out;
}

function buildFromCorpus(corpus: Corpus): BlogPostLite[] {
  if (corpus === builtCorpusRef) return builtPosts;

  const posts: BlogPostLite[] = corpus.map((entry) => {
    const readTime =
      (typeof entry.frontmatter.readTime === "string" &&
        entry.frontmatter.readTime) ||
      calculateReadTimeWithTags(entry.content, entry.tags, entry.category)
        .readTime;
    return {
      slug: entry.slug,
      title: entry.title,
      excerpt: entry.excerpt || deriveExcerpt(entry.content),
      publishDate: entry.publishDate || "",
      readTime,
      category: entry.category || "General",
      subcategory: entry.subcategory || entry.category || "",
      author: entry.author || "Hiep Tran",
      tags: entry.tags,
      image: entry.image || undefined,
      collection: entry.collection,
      featured: entry.featured,
    };
  });

  posts.sort((a, b) => {
    const da = new Date(a.publishDate).getTime();
    const db = new Date(b.publishDate).getTime();
    if (isNaN(da) && isNaN(db)) return 0;
    if (isNaN(da)) return 1;
    if (isNaN(db)) return -1;
    return db - da;
  });

  builtCorpusRef = corpus;
  builtPosts = posts;
  return posts;
}

/** The full corpus as presorted (newest-first) lightweight metadata. */
export async function getAllPostsLite(): Promise<BlogPostLite[]> {
  const generated = loadGenerated();
  if (generated) return generated;
  const corpus = await loadAllPosts();
  return buildFromCorpus(corpus);
}

const slugify = (s: string): string =>
  s
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, "")
    .replace(/\s+/g, "-")
    .trim();

export interface PostFilter {
  category?: string | null;
  tag?: string | null;
  collection?: string | null;
  search?: string | null;
}

/** Filter the presorted list, preserving order. */
export function filterPosts(
  posts: BlogPostLite[],
  { category, tag, collection, search }: PostFilter,
): BlogPostLite[] {
  let out = posts;
  if (search) {
    const q = search.toLowerCase();
    out = out.filter(
      (p) =>
        p.title.toLowerCase().includes(q) ||
        p.excerpt.toLowerCase().includes(q),
    );
  }
  if (category) {
    const q = category.toLowerCase();
    out = out.filter(
      (p) =>
        p.category.toLowerCase() === q ||
        p.category.toLowerCase().includes(q) ||
        p.slug.toLowerCase().startsWith(`${q}/`),
    );
  }
  if (tag) {
    // Match the exact slug TagBadge generates for links: lowercase, spaces→dashes.
    const q = tag.toLowerCase();
    const tagKey = (t: string) => t.toLowerCase().replace(/\s+/g, "-");
    out = out.filter((p) => p.tags.some((t) => tagKey(t) === q));
  }
  if (collection) {
    const q = collection.toLowerCase();
    out = out.filter((p) => !!p.collection && slugify(p.collection) === q);
  }
  return out;
}

/** Top-level category → post count, over the whole corpus. */
export function getCategoryCounts(posts: BlogPostLite[]): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const p of posts) {
    // The leading path segment of the slug is the canonical category folder.
    const top = p.slug.includes("/") ? p.slug.split("/")[0] : slugify(p.category);
    counts[top] = (counts[top] || 0) + 1;
  }
  return counts;
}

/** Slice a presorted/filtered list into a page + pagination metadata. */
export function paginate<T>(
  items: T[],
  page: number,
  limit: number,
): { items: T[]; pagination: Pagination } {
  const safeLimit = Math.max(1, Math.min(100, limit || 12));
  const total = items.length;
  const totalPages = Math.max(1, Math.ceil(total / safeLimit));
  const safePage = Math.max(1, Math.min(page || 1, totalPages));
  const start = (safePage - 1) * safeLimit;
  return {
    items: items.slice(start, start + safeLimit),
    pagination: {
      page: safePage,
      limit: safeLimit,
      total,
      totalPages,
      hasMore: start + safeLimit < total,
    },
  };
}
