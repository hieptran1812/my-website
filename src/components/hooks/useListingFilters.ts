"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Article } from "@/lib/blog";

/**
 * Tag / reading-length / sort state for a blog listing, plus the search query,
 * all mirrored in the URL (`?q=kv+cache&tag=attention,cuda&len=deep&sort=oldest`)
 * so any filtered view can be bookmarked and shared.
 *
 * Topic (subcategory) selection stays in useSubcategoryFilter; this hook takes
 * it as input so it can compute faceted counts for every dimension: each count
 * says how many results you'd have if you toggled that value, given all the
 * other active filters.
 */

export type LengthBucket = "quick" | "medium" | "deep";
export type SortKey =
  "relevance" | "newest" | "oldest" | "longest" | "shortest" | "az";

export const LENGTH_BUCKETS: { key: LengthBucket; label: string }[] = [
  { key: "quick", label: "Under 15 min" },
  { key: "medium", label: "15–40 min" },
  { key: "deep", label: "40+ min" },
];

export const SORT_OPTIONS: { key: SortKey; label: string }[] = [
  { key: "relevance", label: "Best match" },
  { key: "newest", label: "Newest" },
  { key: "oldest", label: "Oldest" },
  { key: "longest", label: "Longest read" },
  { key: "shortest", label: "Shortest read" },
  { key: "az", label: "Title A–Z" },
];

export function readMinutes(a: Article): number {
  return parseInt((a.readTime || "").replace(/[^\d]/g, ""), 10) || 0;
}

export function lengthBucket(a: Article): LengthBucket {
  const m = readMinutes(a);
  return m < 15 ? "quick" : m <= 40 ? "medium" : "deep";
}

/**
 * Time window, single-select: a relative window ("7d", "30d", "90d"), a year
 * ("2025") or a month ("2025-11"), matched against the article's date.
 */
export const RECENT_WINDOWS: { key: string; label: string; days: number }[] = [
  { key: "7d", label: "Past week", days: 7 },
  { key: "30d", label: "Past month", days: 30 },
  { key: "90d", label: "Past 3 months", days: 90 },
];

const MONTHS = [
  "Jan", "Feb", "Mar", "Apr", "May", "Jun",
  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

function isTimeKey(k: string | null): k is string {
  return (
    !!k &&
    (RECENT_WINDOWS.some((w) => w.key === k) || /^\d{4}(-(0[1-9]|1[0-2]))?$/.test(k))
  );
}

/** "YYYY-MM-DD" of the article, or "" when it has no usable date. */
function dayOf(a: Article): string {
  const d = (a.date || "").slice(0, 10);
  return /^\d{4}-\d{2}-\d{2}$/.test(d) ? d : "";
}

export function matchesTime(a: Article, key: string, now = Date.now()): boolean {
  const day = dayOf(a);
  if (!day) return false;
  const w = RECENT_WINDOWS.find((x) => x.key === key);
  if (w) return now - new Date(day).getTime() <= w.days * 86_400_000;
  return day.startsWith(key);
}

export function timeLabel(key: string): string {
  const w = RECENT_WINDOWS.find((x) => x.key === key);
  if (w) return w.label;
  const [y, m] = key.split("-");
  return m ? `${MONTHS[parseInt(m, 10) - 1]} ${y}` : y;
}

export function monthName(key: string): string {
  return MONTHS[parseInt(key.slice(5, 7), 10) - 1] ?? key;
}

function normTag(t: string): string {
  return t.trim().toLowerCase();
}

function readUrl() {
  if (typeof window === "undefined") {
    return {
      q: "",
      tags: [],
      lengths: [],
      sort: null as SortKey | null,
      time: null as string | null,
    };
  }
  const p = new URLSearchParams(window.location.search);
  const list = (k: string) =>
    p
      .getAll(k)
      .flatMap((v) => v.split(","))
      .map(normTag)
      .filter(Boolean);
  const sort = p.get("sort") as SortKey | null;
  return {
    q: p.get("q") ?? "",
    tags: [...new Set(list("tag"))],
    lengths: [
      ...new Set(
        list("len").filter((l): l is LengthBucket =>
          LENGTH_BUCKETS.some((b) => b.key === l),
        ),
      ),
    ],
    sort: SORT_OPTIONS.some((o) => o.key === sort) ? sort : null,
    time: isTimeKey(p.get("time")) ? p.get("time") : null,
  };
}

function writeUrl(state: {
  q: string;
  tags: string[];
  lengths: LengthBucket[];
  sort: SortKey | null;
  time: string | null;
}) {
  const p = new URLSearchParams(window.location.search);
  ["q", "tag", "len", "sort", "time"].forEach((k) => p.delete(k));
  const others = p.toString();
  const mine = [
    state.q.trim() ? `q=${encodeURIComponent(state.q.trim())}` : "",
    state.tags.length
      ? `tag=${state.tags.map(encodeURIComponent).join(",")}`
      : "",
    state.lengths.length ? `len=${state.lengths.join(",")}` : "",
    state.sort ? `sort=${state.sort}` : "",
    state.time ? `time=${state.time}` : "",
  ].filter(Boolean);
  const query = [others, ...mine].filter(Boolean).join("&");
  const url = `${window.location.pathname}${query ? `?${query}` : ""}${window.location.hash}`;
  const current = `${window.location.pathname}${window.location.search}${window.location.hash}`;
  // Filters refine the current view; they never deserve their own history
  // entry per keystroke or toggle.
  if (url !== current) window.history.replaceState(null, "", url);
}

interface Options {
  /** Every article on the page (tag universe for the type-ahead). */
  allArticles: Article[];
  /** Search results, in ranked order, before topic/tag/length filters. */
  searched: Article[];
  searchTerm: string;
  setSearchTerm: (q: string) => void;
  selectedTopics: string[];
}

export function useListingFilters({
  allArticles,
  searched,
  searchTerm,
  setSearchTerm,
  selectedTopics,
}: Options) {
  const initial = useRef(readUrl());
  const [tags, setTags] = useState<string[]>(initial.current.tags);
  const [lengths, setLengths] = useState<LengthBucket[]>(
    initial.current.lengths,
  );
  const [sortChoice, setSortChoice] = useState<SortKey | null>(
    initial.current.sort,
  );
  const [time, setTime] = useState<string | null>(initial.current.time);

  // Seed the search box from ?q= once.
  useEffect(() => {
    if (initial.current.q) setSearchTerm(initial.current.q);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    writeUrl({ q: searchTerm, tags, lengths, sort: sortChoice, time });
  }, [searchTerm, tags, lengths, sortChoice, time]);

  const hasQuery = searchTerm.trim().length >= 2;
  // "Best match" only means something while searching; otherwise the
  // server's newest-first order is the natural default.
  const sort: SortKey = sortChoice ?? (hasQuery ? "relevance" : "newest");

  const tagSets = useMemo(() => {
    const m = new Map<string, Set<string>>();
    for (const a of allArticles) {
      m.set(a.id, new Set((a.tags ?? []).map(normTag)));
    }
    return m;
  }, [allArticles]);

  const inTopics = useCallback(
    (a: Article) =>
      selectedTopics.length === 0 ||
      (!!a.subcategory && selectedTopics.includes(a.subcategory.toLowerCase())),
    [selectedTopics],
  );
  const inTags = useCallback(
    (a: Article) => {
      if (tags.length === 0) return true;
      const set = tagSets.get(a.id);
      return !!set && tags.every((t) => set.has(t));
    },
    [tags, tagSets],
  );
  const inLengths = useCallback(
    (a: Article) => lengths.length === 0 || lengths.includes(lengthBucket(a)),
    [lengths],
  );

  const inTime = useCallback(
    (a: Article) => !time || matchesTime(a, time),
    [time],
  );

  const filtered = useMemo(() => {
    const out = searched.filter(
      (a) => inTopics(a) && inTags(a) && inLengths(a) && inTime(a),
    );
    if (sort === "relevance" || (sort === "newest" && !hasQuery)) return out;
    const time = (a: Article) => new Date(a.date).getTime() || 0;
    const sorted = [...out];
    switch (sort) {
      case "newest":
        sorted.sort((a, b) => time(b) - time(a));
        break;
      case "oldest":
        sorted.sort((a, b) => time(a) - time(b));
        break;
      case "longest":
        sorted.sort((a, b) => readMinutes(b) - readMinutes(a));
        break;
      case "shortest":
        sorted.sort((a, b) => readMinutes(a) - readMinutes(b));
        break;
      case "az":
        sorted.sort((a, b) => a.title.localeCompare(b.title));
        break;
    }
    return sorted;
  }, [searched, inTopics, inTags, inLengths, inTime, sort, hasQuery]);

  const facets = useMemo(() => {
    const topics = new Map<string, number>();
    const lens = new Map<LengthBucket, number>();
    const tagCounts = new Map<string, number>();
    // Time counts: keys are windows, years and months ("2025", "2025-11").
    const times = new Map<string, number>();
    const bump = (m: Map<string, number>, k: string) =>
      m.set(k, (m.get(k) ?? 0) + 1);
    const now = Date.now();
    for (const a of searched) {
      const t = inTopics(a);
      const g = inTags(a);
      const l = inLengths(a);
      const d = inTime(a);
      if (g && l && d && a.subcategory) {
        const k = a.subcategory.toLowerCase();
        topics.set(k, (topics.get(k) ?? 0) + 1);
      }
      if (t && g && d) {
        const b = lengthBucket(a);
        lens.set(b, (lens.get(b) ?? 0) + 1);
      }
      if (t && g && l && d) {
        for (const tag of tagSets.get(a.id) ?? []) {
          tagCounts.set(tag, (tagCounts.get(tag) ?? 0) + 1);
        }
      }
      if (t && g && l) {
        const day = dayOf(a);
        if (day) {
          bump(times, day.slice(0, 4));
          bump(times, day.slice(0, 7));
          for (const w of RECENT_WINDOWS) {
            if (matchesTime(a, w.key, now)) bump(times, w.key);
          }
        }
      }
    }
    return { topics, lengths: lens, tags: tagCounts, times };
  }, [searched, inTopics, inTags, inLengths, inTime, tagSets]);

  // Tags worth suggesting as a next refinement: frequent in the current
  // results but not in all of them (a tag every result has narrows nothing).
  const suggestedTags = useMemo(() => {
    const total = filtered.length;
    return [...facets.tags]
      .filter(([tag, n]) => !tags.includes(tag) && n >= 2 && n < total)
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
      .slice(0, 10)
      .map(([tag, count]) => ({ tag, count }));
  }, [facets.tags, tags, filtered.length]);

  const toggleTag = useCallback((tag: string) => {
    const t = normTag(tag);
    setTags((cur) =>
      cur.includes(t) ? cur.filter((x) => x !== t) : [...cur, t],
    );
  }, []);
  const toggleLength = useCallback((b: LengthBucket) => {
    setLengths((cur) =>
      cur.includes(b) ? cur.filter((x) => x !== b) : [...cur, b],
    );
  }, []);
  const clear = useCallback(() => {
    setTags([]);
    setLengths([]);
    setSortChoice(null);
    setTime(null);
  }, []);

  return {
    filtered,
    facets,
    suggestedTags,
    tags,
    toggleTag,
    lengths,
    toggleLength,
    sort,
    hasQuery,
    setSort: (s: SortKey) => setSortChoice(s),
    time,
    setTime,
    clear,
    active: tags.length > 0 || lengths.length > 0 || !!time,
  };
}

export type ListingFilters = ReturnType<typeof useListingFilters>;
