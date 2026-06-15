/**
 * Precompute a lightweight, presorted blog metadata index.
 *
 * Walks the whole corpus ONCE at build time, computes read-time + excerpt per
 * post, drops the full markdown body, sorts newest-first, and writes
 * `src/lib/generated/blogPostsIndex.json`.
 *
 * Why: the listing endpoints (/api/blog, /api/blog/posts and the pages they
 * feed) only need card metadata, never the body. Reading + gray-matter parsing
 * 1,200 files (~29 MB) and running the read-time regex over all of it on every
 * cold serverless start cost ~1-3s. Doing it here, once per deploy, turns the
 * runtime cost into "parse one ~1 MB JSON". The loader (src/lib/blogPostsIndex.ts)
 * falls back to a runtime walk when this file is empty/missing (dev), so new
 * posts still appear without a rebuild.
 *
 * Resilient by design: any failure writes an empty index and exits 0 so a build
 * never breaks — the runtime fallback covers the gap.
 *
 * Usage: tsx scripts/buildBlogIndex.ts
 */

import fs from "fs";
import path from "path";
import { loadAllPosts } from "../src/lib/blogIndex";
import { calculateReadTimeWithTags } from "../src/lib/readTimeCalculator";

const OUT_PATH = path.join(
  process.cwd(),
  "src",
  "lib",
  "generated",
  "blogPostsIndex.json",
);

interface BlogPostLite {
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

function deriveExcerpt(content: string): string {
  if (!content) return "";
  const lines = content.split("\n").filter((line) => {
    const t = line.trim();
    return (
      t.length > 0 &&
      !t.startsWith("#") &&
      !t.startsWith("!") &&
      !t.startsWith("---") &&
      !t.startsWith("```") &&
      !t.startsWith("|") &&
      !t.match(/^\[.*\]\(.*\)$/)
    );
  });
  let out = lines
    .slice(0, 3)
    .join(" ")
    .replace(/\*\*(.*?)\*\*/g, "$1")
    .replace(/\*(.*?)\*/g, "$1")
    .replace(/`(.*?)`/g, "$1")
    .replace(/\[(.*?)\]\(.*?\)/g, "$1")
    .replace(/\s+/g, " ")
    .trim();
  if (out.length > 200) out = out.substring(0, 197) + "...";
  return out;
}

function writeIndex(posts: BlogPostLite[]): void {
  fs.mkdirSync(path.dirname(OUT_PATH), { recursive: true });
  fs.writeFileSync(OUT_PATH, JSON.stringify(posts) + "\n");
}

async function main() {
  console.log("🗂️  Building lightweight blog index…");
  const corpus = await loadAllPosts();

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

  writeIndex(posts);
  console.log(
    `✅ Wrote ${posts.length} posts → ${path.relative(process.cwd(), OUT_PATH)}`,
  );
}

main().catch((err) => {
  console.error("⚠️  buildBlogIndex failed; writing empty index (runtime fallback will cover):", err);
  try {
    writeIndex([]);
  } catch {
    /* ignore */
  }
  process.exit(0);
});
