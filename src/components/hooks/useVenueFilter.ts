"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import type { Article } from "@/lib/blog";
import { getVenue, TIER_ORDER, type Venue, type VenueTier } from "@/lib/venue";

/**
 * Conference / year / tier filter for the paper-reading listing, mirrored in
 * the URL (`?conf=cvpr,neurips&year=2025&tier=oral,award`) so a filtered view
 * is shareable and survives Back/Forward, like the subcategory filter.
 *
 * Each dimension is an OR within itself and an AND across dimensions. Facet
 * counts for a dimension are computed with the other two applied, so a count
 * always says how many papers you'd get by adding that value.
 */

export interface VenueSelection {
  confs: string[];
  years: number[];
  tiers: VenueTier[];
}

export interface FacetOption<T> {
  value: T;
  count: number;
}

const EMPTY: VenueSelection = { confs: [], years: [], tiers: [] };
const KEYS = ["conf", "year", "tier"] as const;

function readList(params: URLSearchParams, key: string): string[] {
  return params
    .getAll(key)
    .flatMap((v) => v.split(","))
    .map((v) => v.trim().toLowerCase())
    .filter(Boolean);
}

function readSelection(): VenueSelection {
  if (typeof window === "undefined") return EMPTY;
  const p = new URLSearchParams(window.location.search);
  return {
    confs: [...new Set(readList(p, "conf"))],
    years: [
      ...new Set(
        readList(p, "year")
          .map((y) => parseInt(y, 10))
          .filter((y) => !Number.isNaN(y)),
      ),
    ],
    tiers: [
      ...new Set(
        readList(p, "tier").filter((t): t is VenueTier =>
          (TIER_ORDER as string[]).includes(t),
        ),
      ),
    ],
  };
}

function buildUrl(sel: VenueSelection): string {
  const params = new URLSearchParams(window.location.search);
  KEYS.forEach((k) => params.delete(k));
  const others = params.toString();
  // Literal commas, as in useSubcategoryFilter: these URLs are read by humans.
  const mine = [
    sel.confs.length ? `conf=${sel.confs.join(",")}` : "",
    sel.years.length ? `year=${sel.years.join(",")}` : "",
    sel.tiers.length ? `tier=${sel.tiers.join(",")}` : "",
  ].filter(Boolean);
  const query = [others, ...mine].filter(Boolean).join("&");
  return `${window.location.pathname}${query ? `?${query}` : ""}${window.location.hash}`;
}

export function isEmptySelection(sel: VenueSelection): boolean {
  return !sel.confs.length && !sel.years.length && !sel.tiers.length;
}

function matches(
  v: Venue | null,
  sel: VenueSelection,
  skip?: keyof VenueSelection,
): boolean {
  if (isEmptySelection(sel)) return true;
  if (!v) {
    // Only the dimension being counted is active: an un-venued post can't
    // match any other dimension either, so it never counts.
    return false;
  }
  if (skip !== "confs" && sel.confs.length && !sel.confs.includes(v.conf))
    return false;
  if (
    skip !== "years" &&
    sel.years.length &&
    !(v.year && sel.years.includes(v.year))
  )
    return false;
  if (
    skip !== "tiers" &&
    sel.tiers.length &&
    !(v.tier && sel.tiers.includes(v.tier))
  )
    return false;
  return true;
}

/**
 * `allArticles` is the whole listing (venues are parsed once per corpus);
 * `articles` is what the other filters (search, subcategory) left, which this
 * narrows further and computes facet counts over.
 */
export function useVenueFilter(allArticles: Article[], articles: Article[]) {
  const [selection, setSelection] = useState<VenueSelection>(readSelection);

  const venues = useMemo(() => {
    const map = new Map<string, Venue | null>();
    for (const a of allArticles) map.set(a.id, getVenue(a));
    return map;
  }, [allArticles]);

  const setVenueSelection = useCallback(
    (next: VenueSelection) => {
      const url = buildUrl(next);
      const current = `${window.location.pathname}${window.location.search}${window.location.hash}`;
      if (url !== current) {
        // Entering or leaving a filtered view is worth a history step; refining
        // an active filter only rewrites the current entry.
        const crosses = isEmptySelection(selection) !== isEmptySelection(next);
        if (crosses) window.history.pushState(null, "", url);
        else window.history.replaceState(null, "", url);
      }
      setSelection(next);
    },
    [selection],
  );

  useEffect(() => {
    const onPop = () => setSelection(readSelection());
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  const filtered = useMemo(
    () =>
      isEmptySelection(selection)
        ? articles
        : articles.filter((a) => matches(venues.get(a.id) ?? null, selection)),
    [articles, venues, selection],
  );

  const facets = useMemo(() => {
    const confs = new Map<string, number>();
    const years = new Map<number, number>();
    const tiers = new Map<VenueTier, number>();
    let venued = 0;
    for (const a of articles) {
      const v = venues.get(a.id);
      if (!v) continue;
      venued++;
      if (matches(v, selection, "confs"))
        confs.set(v.conf, (confs.get(v.conf) ?? 0) + 1);
      if (v.year && matches(v, selection, "years"))
        years.set(v.year, (years.get(v.year) ?? 0) + 1);
      if (v.tier && matches(v, selection, "tiers"))
        tiers.set(v.tier, (tiers.get(v.tier) ?? 0) + 1);
    }
    // Keep selected values visible even when their count drops to zero, so
    // they can still be unticked.
    selection.confs.forEach((c) => confs.set(c, confs.get(c) ?? 0));
    selection.years.forEach((y) => years.set(y, years.get(y) ?? 0));
    selection.tiers.forEach((t) => tiers.set(t, tiers.get(t) ?? 0));
    return {
      venued,
      confs: [...confs]
        .map(([value, count]) => ({ value, count }))
        .sort((a, b) => b.count - a.count || a.value.localeCompare(b.value)),
      years: [...years]
        .map(([value, count]) => ({ value, count }))
        .sort((a, b) => b.value - a.value),
      tiers: [...tiers]
        .map(([value, count]) => ({ value, count }))
        .sort(
          (a, b) => TIER_ORDER.indexOf(a.value) - TIER_ORDER.indexOf(b.value),
        ),
    };
  }, [articles, venues, selection]);

  return {
    selection,
    setVenueSelection,
    filtered,
    facets,
    venueOf: (a: Article) => venues.get(a.id) ?? null,
  };
}
