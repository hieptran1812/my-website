import fs from "fs";
import path from "path";
import matter from "gray-matter";
import { NextRequest, NextResponse } from "next/server";
import { buildSnippet, cleanMarkdown, search } from "@/lib/searchEngine";
import { getSearchIndex } from "@/lib/searchIndexLoader";

interface SearchHit {
  slug: string;
  title: string;
  category: string;
  subcategory: string;
  publishDate: string;
  snippet: string;
  score: number;
}

const BLOG_DIR = path.join(process.cwd(), "content", "blog");

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/** Query-independent snippet: the same text no matter what was searched. */
function fixedSnippet(text: string): string {
  const clean = text.replace(/\s+/g, " ").trim();
  return escapeHtml(clean.slice(0, 180)) + (clean.length > 180 ? "…" : "");
}

function readBody(file: string): string {
  try {
    return cleanMarkdown(matter(fs.readFileSync(path.join(BLOG_DIR, file), "utf8")).content);
  } catch {
    return "";
  }
}

/**
 * GET /api/blog/search?q=…[&limit=10][&offset=0][&category=machine-learning][&format=slugs]
 *
 * Full-text BM25F search over every post (see src/lib/searchEngine.ts).
 * `format=slugs` returns only the ranked slugs (up to 1000), for listing pages
 * that already hold the card metadata and just need the order.
 */
export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const q = (searchParams.get("q") || "").slice(0, 200);
  const slugsOnly = searchParams.get("format") === "slugs";
  const maxLimit = slugsOnly ? 1000 : 100;
  const limit = Math.max(
    1,
    Math.min(maxLimit, parseInt(searchParams.get("limit") || "8", 10) || 8),
  );
  const offset = Math.max(0, parseInt(searchParams.get("offset") || "0", 10) || 0);
  const category = searchParams.get("category") || undefined;

  if (q.trim().length < 2) {
    return NextResponse.json({ hits: [], total: 0 });
  }

  const index = await getSearchIndex();
  const result = search(index, q, { limit, offset, category });
  const headers = {
    "Cache-Control": "public, s-maxage=300, stale-while-revalidate=3600",
  };

  if (slugsOnly) {
    return NextResponse.json(
      {
        slugs: result.hits.map((h) => h.doc.slug),
        total: result.total,
        correctedQuery: result.correctedQuery,
      },
      { headers },
    );
  }

  const hits: SearchHit[] = result.hits.map(({ doc, score }) => ({
    slug: doc.slug,
    title: doc.title,
    category: doc.category,
    subcategory: doc.subcategory,
    publishDate: doc.publishDate,
    // A match-anchored snippet is a movable window into the body: query the
    // tail of one window to reveal the next, and a determined caller walks a
    // gated post out ~200 chars at a time. For paywalled posts the snippet is
    // the fixed excerpt instead, so the query cannot steer it. Ranking still
    // uses the full body — only what is echoed back is pinned.
    snippet: doc.paywalled
      ? fixedSnippet(doc.excerpt || doc.title)
      : buildSnippet(readBody(doc.file) || doc.excerpt, result.matchedWords),
    score,
  }));

  return NextResponse.json(
    { hits, total: result.total, correctedQuery: result.correctedQuery },
    { headers },
  );
}
