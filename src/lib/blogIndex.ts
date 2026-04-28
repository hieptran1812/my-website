/**
 * Shared in-memory corpus index for the blog.
 *
 * Replaces the per-route filesystem walks in /api/blog/*. One walk, cached for
 * `BLOG_INDEX_TTL_MS`, reused by every route that needs metadata or full
 * content. Saves ~150-300ms of redundant fs work on cold requests and keeps
 * memory bounded (one copy in process).
 */

import fs from "fs";
import path from "path";
import matter from "gray-matter";
import { derivePostLocation } from "./postPath";

const FIRST_BODY_IMAGE_RE = /!\[[^\]]*\]\(([^)\s]+)(?:\s+["'][^"']*["'])?\)/;

export function extractFirstBodyImage(markdown: string): string | undefined {
  if (!markdown) return undefined;
  const m = FIRST_BODY_IMAGE_RE.exec(markdown);
  if (!m) return undefined;
  const url = m[1].trim();
  return url || undefined;
}

export function resolvePostCover(
  frontmatter: Record<string, unknown>,
  body: string,
): string | undefined {
  const candidates: (string | undefined)[] = [];
  const fm = frontmatter.image;
  if (typeof fm === "string" && fm.trim().length > 0)
    candidates.push(fm.trim());
  candidates.push(extractFirstBodyImage(body));
  for (const c of candidates) {
    if (!c) continue;
    if (/^https?:\/\//i.test(c)) continue; // remote — needs remotePatterns
    return c;
  }
  return undefined;
}

export interface BlogIndexEntry {
  /** "category/subcategory/file-name" — matches the public URL slug. */
  slug: string;
  filePath: string;
  title: string;
  excerpt: string;
  category: string;
  subcategory: string;
  author: string;
  publishDate: string;
  tags: string[];
  collection?: string;
  image?: string;
  featured: boolean;
  aiGenerated: boolean;
  /** Raw markdown body (post-frontmatter). Reused for read-time, search, etc. */
  content: string;
  /** Frontmatter as parsed — keep around so per-route quirks can read extras. */
  frontmatter: Record<string, unknown>;
}

const blogDir = path.join(process.cwd(), "content", "blog");
// 5-minute TTL is plenty for a static-content blog and matches the existing
// caches scattered across routes.
const BLOG_INDEX_TTL_MS = 5 * 60 * 1000;

let cached: BlogIndexEntry[] | null = null;
let cachedAt = 0;
let inFlight: Promise<BlogIndexEntry[]> | null = null;

function isString(value: unknown): value is string {
  return typeof value === "string";
}

function readEntry(absPath: string): BlogIndexEntry | null {
  try {
    const raw = fs.readFileSync(absPath, "utf8");
    const parsed = matter(raw);
    const data = parsed.data as Record<string, unknown>;
    const { category, subcategory } = derivePostLocation(absPath, data, blogDir);

    const slug = path
      .relative(blogDir, absPath)
      .replace(/\.md$/, "")
      .split(path.sep)
      .join("/");

    const tags = Array.isArray(data.tags)
      ? data.tags.map((t) => String(t))
      : [];

    return {
      slug,
      filePath: absPath,
      title: isString(data.title) ? data.title : slug,
      excerpt: isString(data.excerpt)
        ? data.excerpt
        : isString(data.description)
          ? data.description
          : "",
      category,
      subcategory,
      author: isString(data.author) ? data.author : "",
      publishDate: isString(data.publishDate)
        ? data.publishDate
        : isString(data.date)
          ? data.date
          : "",
      tags,
      collection: isString(data.collection) ? data.collection : undefined,
      image: resolvePostCover(data, parsed.content),
      featured: data.featured === true,
      aiGenerated: data.aiGenerated === true,
      content: parsed.content,
      frontmatter: data,
    };
  } catch {
    return null;
  }
}

function walkBlog(): BlogIndexEntry[] {
  const out: BlogIndexEntry[] = [];
  const seen = new Set<string>();
  if (!fs.existsSync(blogDir)) return out;

  const stack: string[] = [blogDir];
  while (stack.length > 0) {
    const dir = stack.pop()!;
    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      continue;
    }
    for (const e of entries) {
      const full = path.join(dir, e.name);
      if (e.isDirectory()) {
        stack.push(full);
      } else if (e.isFile() && e.name.endsWith(".md")) {
        if (seen.has(full)) continue;
        seen.add(full);
        const entry = readEntry(full);
        if (entry) out.push(entry);
      }
    }
  }
  return out;
}

/** Load (or return cached) the full blog corpus. Awaitable so callers can
 *  refactor to async without breaking shape later. */
export async function loadAllPosts(): Promise<BlogIndexEntry[]> {
  const now = Date.now();
  if (cached && now - cachedAt < BLOG_INDEX_TTL_MS) return cached;
  if (inFlight) return inFlight;

  inFlight = (async () => {
    try {
      const out = walkBlog();
      cached = out;
      cachedAt = Date.now();
      return out;
    } finally {
      inFlight = null;
    }
  })();
  return inFlight;
}

export async function findPostBySlug(
  slug: string,
): Promise<BlogIndexEntry | null> {
  const all = await loadAllPosts();
  return all.find((p) => p.slug === slug) ?? null;
}

/** Force-rebuild the cache. Useful for tests or post-deploy hot reload. */
export function invalidateBlogIndex(): void {
  cached = null;
  cachedAt = 0;
}
