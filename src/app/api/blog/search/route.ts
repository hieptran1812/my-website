import { NextRequest, NextResponse } from "next/server";
import { loadAllPosts, type BlogIndexEntry } from "@/lib/blogIndex";
import {
  parseQuery,
  scoreNormalized,
  normalizeFields,
  type NormalizedFields,
} from "@/lib/searchScore";

interface SearchHit {
  slug: string;
  title: string;
  category: string;
  subcategory: string;
  publishDate: string;
  snippet: string;
  score: number;
}

const SNIPPET_RADIUS = 90;

interface PreparedDoc extends BlogIndexEntry {
  /** Normalized title/tags/content, scored with strict title ≫ tags ≫ body tiers. */
  fields: NormalizedFields;
  /** Cleaned (markdown-stripped) body, used only to render the result snippet. */
  body: string;
}

let preparedCache: PreparedDoc[] | null = null;
let preparedFor: BlogIndexEntry[] | null = null;

function prepare(corpus: BlogIndexEntry[]): PreparedDoc[] {
  if (preparedCache && preparedFor === corpus) return preparedCache;
  const out: PreparedDoc[] = corpus.map((entry) => {
    const body = entry.content
      .replace(/```[\s\S]*?```/g, " ")
      .replace(/!\[[^\]]*\]\([^)]*\)/g, " ")
      .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
      .replace(/^#{1,6}\s+/gm, "")
      .replace(/[*_`>~]/g, " ")
      .replace(/\s+/g, " ")
      .trim();
    const fields = normalizeFields({
      title: entry.title,
      tags: entry.tags,
      excerpt: entry.excerpt,
      body,
    });
    return { ...entry, fields, body };
  });
  preparedCache = out;
  preparedFor = corpus;
  return out;
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

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const q = (searchParams.get("q") || "").trim();
  const limit = Math.min(20, parseInt(searchParams.get("limit") || "8", 10));
  if (q.length < 2) {
    return NextResponse.json({ hits: [] });
  }

  const query = parseQuery(q);
  if (query.tokens.length === 0) {
    return NextResponse.json({ hits: [] });
  }

  const corpus = await loadAllPosts();
  const docs = prepare(corpus);
  const hits: SearchHit[] = [];
  for (const doc of docs) {
    const score = scoreNormalized(doc.fields, query);
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
  // Strict tiers already encode title ≫ tags ≫ content; on a tie, newer first.
  hits.sort((a, b) =>
    b.score !== a.score
      ? b.score - a.score
      : (b.publishDate || "").localeCompare(a.publishDate || ""),
  );
  return NextResponse.json(
    { hits: hits.slice(0, limit) },
    {
      headers: {
        "Cache-Control":
          "public, s-maxage=300, stale-while-revalidate=3600",
      },
    },
  );
}
