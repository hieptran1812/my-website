"use client";

import { useState, useMemo } from "react";
import type { Article } from "@/lib/blog";

export function useArticleSearch(
  allArticles: Article[],
  selectedCategories: string[]
) {
  const [searchTerm, setSearchTerm] = useState("");

  const filteredArticles = useMemo(() => {
    let result = allArticles;

    if (selectedCategories.length > 0) {
      result = result.filter(
        (article) =>
          article.subcategory &&
          selectedCategories.includes(article.subcategory.toLowerCase()),
      );
    }

    if (searchTerm.trim()) {
      const q = searchTerm.toLowerCase();
      result = result.filter(
        (article) =>
          article.title.toLowerCase().includes(q) ||
          article.excerpt?.toLowerCase().includes(q) ||
          article.tags?.some((tag) => tag.toLowerCase().includes(q)),
      );
    }

    return result;
  }, [allArticles, selectedCategories, searchTerm]);

  return { searchTerm, setSearchTerm, filteredArticles };
}
