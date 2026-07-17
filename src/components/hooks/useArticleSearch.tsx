"use client";

import { useMemo, useState } from "react";
import type { Article } from "@/lib/blog";
import {
  hasQuery,
  normalizeFields,
  parseQuery,
  scoreNormalized,
  type NormalizedFields,
} from "@/lib/searchScore";

export function useArticleSearch(
  allArticles: Article[],
  selectedCategories: string[],
) {
  const [searchTerm, setSearchTerm] = useState("");

  // Normalize each article's searchable fields once per corpus, not per keystroke.
  // Listing cards ship without the body (`content` is empty by design), so the
  // "content" tier here falls back to the excerpt — the closest available proxy.
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

  const filteredArticles = useMemo(() => {
    let indices = allArticles.map((_, i) => i);

    if (selectedCategories.length > 0) {
      indices = indices.filter((i) => {
        const sub = allArticles[i].subcategory;
        return sub && selectedCategories.includes(sub.toLowerCase());
      });
    }

    const query = parseQuery(searchTerm);
    if (!hasQuery(query)) {
      return indices.map((i) => allArticles[i]);
    }

    // Score, drop non-matches, then rank by strict tiers (title ≫ tags ≫
    // content). Ties keep the incoming order (presorted newest-first).
    return indices
      .map((i) => ({ i, score: scoreNormalized(normalized[i], query) }))
      .filter((r) => r.score > 0)
      .sort((a, b) => (b.score !== a.score ? b.score - a.score : a.i - b.i))
      .map((r) => allArticles[r.i]);
  }, [allArticles, normalized, selectedCategories, searchTerm]);

  return { searchTerm, setSearchTerm, filteredArticles };
}
