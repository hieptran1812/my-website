import path from "path";
import matter from "gray-matter";
import { derivePostLocation } from "./postPath";
import { readPostRaw, slugToRelPath } from "./contentSource";

export interface PostMeta {
  slug: string;
  title: string;
  category: string;
  subcategory: string;
  author: string;
  publishDate: string;
  tags: string[];
  collection?: string;
  substackUrl?: string;
}

const blogDir = path.join(process.cwd(), "content", "blog");

export async function getPostMeta(slug: string): Promise<PostMeta | null> {
  // Disk in dev/build, R2 in serverless functions (see contentSource.ts).
  const rel = slugToRelPath(slug);
  if (!rel) return null;
  const raw = await readPostRaw(rel);
  if (raw === null) return null;
  const file = path.join(blogDir, ...rel.split("/"));
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
    substackUrl:
      typeof data.substackUrl === "string"
        ? data.substackUrl.trim() || undefined
        : typeof data.substack === "string"
          ? data.substack.trim() || undefined
          : undefined,
  };
}
