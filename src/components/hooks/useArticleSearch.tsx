"use client";

import { useEffect, useMemo, useState } from "react";
import type { Article } from "@/lib/blog";
import {
  hasQuery,
  normalizeFields,
  parseQuery,
  scoreNormalized,
  type NormalizedFields,
} from "@/lib/searchScore";

interface ServerRanking {
  query: string;
  slugs: string[];
  correctedQuery?: string;
}

export function useArticleSearch(
  allArticles: Article[],
  selectedCategories: string[],
) {
  const [searchTerm, setSearchTerm] = useState("");
  const [server, setServer] = useState<ServerRanking | null>(null);

  // Normalize each article's searchable fields once per corpus, not per keystroke.
  // Listing cards ship without the body (`content` is empty by design), so the
  // "content" tier here falls back to the excerpt — the closest available proxy.
  // This local ranking is shown instantly and is the fallback if the server
  // search fails; the full-text server ranking replaces it once it arrives.
  const normalized = useMemo<NormalizedFields[]>(
    () =>
      allArticles.map((a) =>
        normalizeFields({
          title: a.title,
          tags: a.tags,
          excerpt: a.excerpt,
          body: a.content,
        }),
      ),
    [allArticles],
  );

  // Listing pages hold one category's cards; scope the server search to it.
  const category = useMemo(() => {
    const cats = new Set(allArticles.map((a) => a.category));
    return cats.size === 1 ? [...cats][0] : undefined;
  }, [allArticles]);

  // Full-text search (post bodies, typo tolerance, synonyms) on the server.
  useEffect(() => {
    const q = searchTerm.trim();
    if (q.length < 2) return;
    const ctrl = new AbortController();
    const handle = setTimeout(async () => {
      try {
        const params = new URLSearchParams({
          q,
          format: "slugs",
          limit: "1000",
        });
        if (category) params.set("category", category);
        const res = await fetch(`/api/blog/search?${params}`, {
          signal: ctrl.signal,
        });
        if (!res.ok) return;
        const data = await res.json();
        setServer({
          query: q,
          slugs: Array.isArray(data.slugs) ? data.slugs : [],
          correctedQuery: data.correctedQuery,
        });
      } catch {
        // Aborted or offline: keep the local ranking.
      }
    }, 200);
    return () => {
      clearTimeout(handle);
      ctrl.abort();
    };
  }, [searchTerm, category]);

  const filteredArticles = useMemo(() => {
    const inSelected = (a: Article) =>
      selectedCategories.length === 0 ||
      (!!a.subcategory &&
        selectedCategories.includes(a.subcategory.toLowerCase()));

    const query = parseQuery(searchTerm);
    if (!hasQuery(query)) {
      return allArticles.filter(inSelected);
    }

    if (server && server.query === searchTerm.trim()) {
      const bySlug = new Map(allArticles.map((a) => [a.slug, a]));
      return server.slugs
        .map((s) => bySlug.get(s))
        .filter((a): a is Article => !!a && inSelected(a));
    }

    // Score, drop non-matches, then rank by strict tiers (title ≫ tags ≫
    // content). Ties keep the incoming order (presorted newest-first).
    return allArticles
      .map((a, i) => ({ a, i, score: scoreNormalized(normalized[i], query) }))
      .filter((r) => r.score > 0 && inSelected(r.a))
      .sort((x, y) => (y.score !== x.score ? y.score - x.score : x.i - y.i))
      .map((r) => r.a);
  }, [allArticles, normalized, selectedCategories, searchTerm, server]);

  const correctedQuery =
    server && server.query === searchTerm.trim()
      ? server.correctedQuery
      : undefined;

  return { searchTerm, setSearchTerm, filteredArticles, correctedQuery };
}
