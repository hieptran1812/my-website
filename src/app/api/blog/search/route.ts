import { NextRequest, NextResponse } from "next/server";
import fs from "fs";
import path from "path";
import matter from "gray-matter";
import { derivePostLocation } from "@/lib/postPath";

interface SearchDoc {
  slug: string;
  title: string;
  excerpt: string;
  category: string;
  subcategory: string;
  tags: string[];
  publishDate: string;
  haystack: string;
  body: string;
}

interface SearchHit {
  slug: string;
  title: string;
  category: string;
  subcategory: string;
  publishDate: string;
  snippet: string;
  score: number;
}

const blogDir = path.join(process.cwd(), "content", "blog");

let docs: SearchDoc[] | null = null;
let docsBuiltAt = 0;
const TTL_MS = 5 * 60 * 1000;
const SNIPPET_RADIUS = 90; // chars on each side of the first match

function buildDocs(): SearchDoc[] {
  const out: SearchDoc[] = [];
  const walk = (dir: string) => {
    if (!fs.existsSync(dir)) return;
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) walk(full);
      else if (entry.isFile() && entry.name.endsWith(".md")) {
        const raw = fs.readFileSync(full, "utf8");
        const { data, content } = matter(raw);
        const { category, subcategory } = derivePostLocation(full, data, blogDir);
        const slug = path
          .relative(blogDir, full)
          .replace(/\.md$/, "")
          .split(path.sep)
          .join("/");
        // Strip code fences, images, links, headings markers from body for cleaner snippets.
        const body = content
          .replace(/```[\s\S]*?```/g, " ")
          .replace(/!\[[^\]]*\]\([^)]*\)/g, " ")
          .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
          .replace(/^#{1,6}\s+/gm, "")
          .replace(/[*_`>~]/g, " ")
          .replace(/\s+/g, " ")
          .trim();
        const tags = Array.isArray(data.tags) ? data.tags.map(String) : [];
        const haystack = [
          data.title || "",
          data.excerpt || data.description || "",
          tags.join(" "),
          body,
        ]
          .join("  ")
          .toLowerCase();
        out.push({
          slug,
          title: data.title || slug,
          excerpt: data.excerpt || data.description || "",
          category,
          subcategory,
          tags,
          publishDate: data.publishDate || data.date || "",
          haystack,
          body,
        });
      }
    }
  };
  walk(blogDir);
  return out;
}

function getDocs(): SearchDoc[] {
  const now = Date.now();
  if (docs && now - docsBuiltAt < TTL_MS) return docs;
  docs = buildDocs();
  docsBuiltAt = now;
  return docs;
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function buildSnippet(body: string, query: string): string {
  const lowerBody = body.toLowerCase();
  const tokens = query
    .toLowerCase()
    .split(/\s+/)
    .filter((t) => t.length >= 2);
  if (tokens.length === 0) {
    return escapeHtml(body.slice(0, 180)) + (body.length > 180 ? "…" : "");
  }
  // Find earliest match for any token.
  let bestIdx = -1;
  for (const t of tokens) {
    const idx = lowerBody.indexOf(t);
    if (idx >= 0 && (bestIdx === -1 || idx < bestIdx)) bestIdx = idx;
  }
  if (bestIdx === -1) {
    return escapeHtml(body.slice(0, 180)) + (body.length > 180 ? "…" : "");
  }
  const start = Math.max(0, bestIdx - SNIPPET_RADIUS);
  const end = Math.min(body.length, bestIdx + SNIPPET_RADIUS);
  let slice = body.slice(start, end);
  if (start > 0) slice = "…" + slice;
  if (end < body.length) slice = slice + "…";

  let html = escapeHtml(slice);
  for (const t of tokens) {
    const re = new RegExp(`(${t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")})`, "gi");
    html = html.replace(re, "<mark>$1</mark>");
  }
  return html;
}

function scoreDoc(doc: SearchDoc, tokens: string[]): number {
  let score = 0;
  const titleLower = doc.title.toLowerCase();
  for (const t of tokens) {
    if (!t) continue;
    if (titleLower.includes(t)) score += 6;
    if (doc.tags.some((x) => x.toLowerCase().includes(t))) score += 3;
    const matches = doc.haystack.split(t).length - 1;
    score += Math.min(matches, 8);
  }
  return score;
}

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const q = (searchParams.get("q") || "").trim();
  const limit = Math.min(20, parseInt(searchParams.get("limit") || "8", 10));
  if (q.length < 2) {
    return NextResponse.json({ hits: [] });
  }

  const tokens = q
    .toLowerCase()
    .split(/\s+/)
    .filter((t) => t.length >= 2);

  const all = getDocs();
  const hits: SearchHit[] = [];
  for (const doc of all) {
    const score = scoreDoc(doc, tokens);
    if (score <= 0) continue;
    hits.push({
      slug: doc.slug,
      title: doc.title,
      category: doc.category,
      subcategory: doc.subcategory,
      publishDate: doc.publishDate,
      snippet: buildSnippet(doc.body, q),
      score,
    });
  }
  hits.sort((a, b) => b.score - a.score);
  return NextResponse.json({ hits: hits.slice(0, limit) });
}
