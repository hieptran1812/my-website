import fs from "fs";
import path from "path";
import matter from "gray-matter";
import { derivePostLocation } from "./postPath";

export interface PostMeta {
  slug: string;
  title: string;
  category: string;
  subcategory: string;
  author: string;
  publishDate: string;
  tags: string[];
  collection?: string;
}

const blogDir = path.join(process.cwd(), "content", "blog");

/** Resolve a slug like "paper-reading/foo/bar" to its on-disk markdown file. */
function findFile(slug: string): string | null {
  const parts = slug.split("/").filter(Boolean);
  if (parts.length === 0) return null;
  const candidate = path.join(blogDir, ...parts) + ".md";
  if (fs.existsSync(candidate) && fs.statSync(candidate).isFile()) return candidate;
  return null;
}

export function getPostMeta(slug: string): PostMeta | null {
  const file = findFile(slug);
  if (!file) return null;
  const raw = fs.readFileSync(file, "utf8");
  const { data } = matter(raw);
  const { category, subcategory } = derivePostLocation(file, data, blogDir);
  return {
    slug,
    title: typeof data.title === "string" ? data.title : slug,
    category,
    subcategory,
    author: typeof data.author === "string" ? data.author : "Hiep Tran",
    publishDate:
      (typeof data.publishDate === "string" && data.publishDate) ||
      (typeof data.date === "string" && data.date) ||
      "",
    tags: Array.isArray(data.tags) ? data.tags.map(String) : [],
    collection: typeof data.collection === "string" ? data.collection : undefined,
  };
}
