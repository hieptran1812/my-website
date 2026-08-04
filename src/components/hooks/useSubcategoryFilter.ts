"use client";

import { useCallback, useEffect, useState } from "react";

export interface SubcategoryOption {
  name: string;
  slug: string;
}

/** Canonical query key written back to the URL. */
const PARAM = "subcategory";
/** Keys accepted when reading, so short hand-written links keep working. */
const READ_KEYS = [PARAM, "sub"];

/**
 * Normalize anything a URL might carry ("Computer Vision", "computer_vision",
 * "Computer-Vision") into the folder-style slug the articles are tagged with.
 */
function slugify(raw: string): string {
  return raw
    .trim()
    .toLowerCase()
    .replace(/[\s_]+/g, "-")
    .replace(/-{2,}/g, "-")
    .replace(/^-|-$/g, "");
}

function dedupe(slugs: string[]): string[] {
  const out: string[] = [];
  for (const slug of slugs) {
    if (slug && slug !== "all" && !out.includes(slug)) out.push(slug);
  }
  return out;
}

function readSelection(): string[] {
  if (typeof window === "undefined") return [];
  const params = new URLSearchParams(window.location.search);
  const values: string[] = [];
  for (const key of READ_KEYS) {
    for (const value of params.getAll(key)) {
      values.push(...value.split(","));
    }
  }
  return dedupe(values.map(slugify));
}

function currentUrl(): string {
  return `${window.location.pathname}${window.location.search}${window.location.hash}`;
}

function buildUrl(slugs: string[]): string {
  const params = new URLSearchParams(window.location.search);
  READ_KEYS.forEach((key) => params.delete(key));
  const others = params.toString();
  // Joined with a literal comma: URLSearchParams would escape it to %2C and
  // this URL is meant to be read and shared by humans.
  const mine = slugs.length
    ? `${PARAM}=${slugs.map(encodeURIComponent).join(",")}`
    : "";
  const query = [others, mine].filter(Boolean).join("&");
  return `${window.location.pathname}${query ? `?${query}` : ""}${window.location.hash}`;
}

/**
 * Push or replace the query string without a navigation. Next 15 supports the
 * native History API for same-page URL updates, which keeps scroll position and
 * skips a router round-trip.
 */
function syncUrl(slugs: string[], mode: "push" | "replace") {
  const url = buildUrl(slugs);
  if (url === currentUrl()) return;
  if (mode === "push") window.history.pushState(null, "", url);
  else window.history.replaceState(null, "", url);
}

function isSame(a: string[], b: string[]): boolean {
  return a.length === b.length && a.every((slug, i) => slug === b[i]);
}

/**
 * Subcategory filter state mirrored in the URL (`?subcategory=nlp,rag`), so a
 * filtered listing is refreshable, shareable and reachable by Back/Forward.
 *
 * `available` is the page's derived subcategory list; once it arrives, values
 * from the URL are canonicalized against it and unknown ones are dropped.
 */
export function useSubcategoryFilter(available: SubcategoryOption[] = []) {
  const [selectedSlugs, setSelected] = useState<string[]>(readSelection);

  const setSelectedSlugs = useCallback(
    (next: string[]) => {
      const cleaned = dedupe(next.map(slugify));
      if (isSame(cleaned, selectedSlugs)) return;
      // Turning filtering on (or clearing it) is a state worth stepping back
      // to; toggling one more box inside an active filter is not, so that only
      // rewrites the current history entry.
      const crossesBoundary =
        (selectedSlugs.length === 0) !== (cleaned.length === 0);
      syncUrl(cleaned, crossesBoundary ? "push" : "replace");
      setSelected(cleaned);
    },
    [selectedSlugs],
  );

  // Back/Forward: adopt whatever the restored URL says.
  useEffect(() => {
    const onPopState = () => setSelected(readSelection());
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  // Once the article list has loaded, reconcile the URL with reality: keep the
  // slugs that exist, resolve display names, drop the rest, tidy the URL.
  useEffect(() => {
    const canonicalBy = new Map<string, string>();
    for (const option of available) {
      const slug = slugify(option?.slug ?? "");
      if (!slug || slug === "all") continue;
      canonicalBy.set(slug, slug);
      if (option.name) canonicalBy.set(slugify(option.name), slug);
    }
    // Still loading (pages seed the list with a lone "All" entry) — pruning now
    // would throw away a perfectly good deep link.
    if (canonicalBy.size === 0) return;

    const canonical = dedupe(
      selectedSlugs.map((slug) => canonicalBy.get(slug) ?? ""),
    );

    if (!isSame(canonical, selectedSlugs)) setSelected(canonical);
    // Also normalizes casing, `sub=` aliases and duplicates already in the URL.
    syncUrl(canonical, "replace");
  }, [available, selectedSlugs]);

  return { selectedSlugs, setSelectedSlugs };
}
