import fs from "fs";
import path from "path";
import matter from "gray-matter";
import { derivePostLocation } from "./postPath";

const blogDir = path.join(process.cwd(), "content", "blog");

export interface RelatedPost {
  slug: string;
  title: string;
  excerpt: string;
  category: string;
  subcategory: string;
  publishDate: string;
  score: number;
}

interface IndexEntry {
  slug: string;
  title: string;
  excerpt: string;
  category: string;
  subcategory: string;
  tags: string[];
  publishDate: string;
}

let cachedIndex: IndexEntry[] | null = null;
let cachedAt = 0;
const CACHE_TTL_MS = 5 * 60 * 1000;

function buildIndex(): IndexEntry[] {
  const out: IndexEntry[] = [];
  const walk = (dir: string) => {
    if (!fs.existsSync(dir)) return;
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        walk(full);
      } else if (entry.isFile() && entry.name.endsWith(".md")) {
        const raw = fs.readFileSync(full, "utf8");
        const { data } = matter(raw);
        const { category, subcategory } = derivePostLocation(full, data, blogDir);
        const rel = path.relative(blogDir, full).replace(/\.md$/, "");
        const slug = rel.split(path.sep).join("/");
        out.push({
          slug,
          title: data.title || rel,
          excerpt: data.excerpt || data.description || "",
          category,
          subcategory,
          tags: Array.isArray(data.tags) ? data.tags.map(String) : [],
          publishDate: data.publishDate || data.date || "",
        });
      }
    }
  };
  walk(blogDir);
  return out;
}

function getIndex(): IndexEntry[] {
  const now = Date.now();
  if (cachedIndex && now - cachedAt < CACHE_TTL_MS) return cachedIndex;
  cachedIndex = buildIndex();
  cachedAt = now;
  return cachedIndex;
}

export function getRelatedPosts(
  currentSlug: string,
  currentTags: string[],
  currentCategory: string,
  currentSubcategory: string,
  limit = 4,
): RelatedPost[] {
  const idx = getIndex();
  const tagSet = new Set(currentTags.map((t) => t.toLowerCase()));

  const scored: RelatedPost[] = [];
  for (const entry of idx) {
    if (entry.slug === currentSlug) continue;

    let score = 0;
    const sharedTags = entry.tags.filter((t) => tagSet.has(t.toLowerCase()));
    score += sharedTags.length * 3;
    if (currentSubcategory && entry.subcategory === currentSubcategory) score += 2;
    if (currentCategory && entry.category === currentCategory) score += 1;
    if (score === 0) continue;

    scored.push({
      slug: entry.slug,
      title: entry.title,
      excerpt: entry.excerpt,
      category: entry.category,
      subcategory: entry.subcategory,
      publishDate: entry.publishDate,
      score,
    });
  }

  scored.sort((a, b) => {
    if (b.score !== a.score) return b.score - a.score;
    const da = Date.parse(a.publishDate) || 0;
    const db = Date.parse(b.publishDate) || 0;
    return db - da;
  });

  return scored.slice(0, limit);
}

export function getPopularPosts(limit = 6): RelatedPost[] {
  const idx = getIndex();
  return [...idx]
    .sort((a, b) => {
      const da = Date.parse(a.publishDate) || 0;
      const db = Date.parse(b.publishDate) || 0;
      return db - da;
    })
    .slice(0, limit)
    .map((e) => ({
      slug: e.slug,
      title: e.title,
      excerpt: e.excerpt,
      category: e.category,
      subcategory: e.subcategory,
      publishDate: e.publishDate,
      score: 0,
    }));
}
