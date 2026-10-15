"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import BlogSearchBar from "@/components/BlogSearchBar";
import {
  LENGTH_BUCKETS,
  SORT_OPTIONS,
  type ListingFilters,
} from "@/components/hooks/useListingFilters";

export interface TopicOption {
  slug: string;
  name: string;
}

interface BlogFilterBarProps {
  searchTerm: string;
  setSearchTerm: (q: string) => void;
  placeholder?: string;
  /** Server-side typo correction, shown as "Showing results for …". */
  correctedQuery?: string;
  topics: TopicOption[];
  selectedTopics: string[];
  setSelectedTopics: (slugs: string[]) => void;
  filters: ListingFilters;
  resultCount: number;
  totalCount: number;
  noun?: string;
  /** Page-specific controls (e.g. the conference filter), shown in row 2. */
  extra?: React.ReactNode;
  /** Pills for page-specific filters, shown with the other active filters. */
  extraPills?: { label: string; onRemove: () => void }[];
  /** Clears page-specific filters as part of "Clear all". */
  onClearExtra?: () => void;
  /** Rendered under the search row (e.g. a one-click venue suggestion). */
  hint?: React.ReactNode;
}

/** How many topic chips show before "+N more". */
const TOPIC_PREVIEW = 8;

function Chip({
  active,
  disabled,
  onClick,
  children,
  title,
}: {
  active?: boolean;
  disabled?: boolean;
  onClick: () => void;
  children: React.ReactNode;
  title?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-pressed={active}
      title={title}
      className="shrink-0 inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm border transition-colors duration-150 disabled:opacity-40 disabled:cursor-not-allowed"
      style={{
        backgroundColor: active ? "var(--accent)" : "var(--surface)",
        borderColor: active ? "var(--accent)" : "var(--border)",
        color: active ? "white" : "var(--text-primary)",
      }}
    >
      {children}
    </button>
  );
}

function Count({ n, active }: { n: number; active?: boolean }) {
  return (
    <span
      className="text-xs tabular-nums"
      style={{ opacity: active ? 0.85 : 0.55 }}
    >
      {n.toLocaleString("en-US")}
    </span>
  );
}

function Label({ children }: { children: React.ReactNode }) {
  return (
    <span
      className="shrink-0 text-[11px] font-semibold uppercase tracking-wider"
      style={{ color: "var(--text-secondary)" }}
    >
      {children}
    </span>
  );
}

/** Type-ahead over the tags present in the current results. */
function TagFinder({
  counts,
  selected,
  onPick,
}: {
  counts: Map<string, number>;
  selected: string[];
  onPick: (tag: string) => void;
}) {
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const [cursor, setCursor] = useState(0);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node))
        setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [open]);

  const matches = useMemo(() => {
    const needle = q
      .trim()
      .toLowerCase()
      .replace(/^#/, "")
      .replace(/\s+/g, "-");
    const all = [...counts].filter(([t]) => !selected.includes(t));
    const scored = needle
      ? all
          .map(([t, n]) => {
            const i = t.indexOf(needle);
            // Prefix beats word-start beats anywhere; then popularity.
            const rank =
              i === 0
                ? 0
                : i > 0 && /[-\s]/.test(t[i - 1])
                  ? 1
                  : i > 0
                    ? 2
                    : -1;
            return { t, n, rank };
          })
          .filter((m) => m.rank >= 0)
      : all.map(([t, n]) => ({ t, n, rank: 0 }));
    return scored
      .sort((a, b) => a.rank - b.rank || b.n - a.n || a.t.localeCompare(b.t))
      .slice(0, 8);
  }, [q, counts, selected]);

  const pick = (tag: string) => {
    onPick(tag);
    setQ("");
    setCursor(0);
  };

  return (
    <div className="relative w-full sm:w-64" ref={ref}>
      <div
        className="flex items-center gap-2 px-3 py-2 rounded-xl border"
        style={{
          backgroundColor: "var(--surface)",
          borderColor: open ? "var(--accent)" : "var(--border)",
        }}
      >
        <span aria-hidden style={{ color: "var(--accent)" }}>
          #
        </span>
        <input
          type="text"
          value={q}
          onChange={(e) => {
            setQ(e.target.value);
            setOpen(true);
            setCursor(0);
          }}
          onFocus={() => setOpen(true)}
          onKeyDown={(e) => {
            if (e.key === "ArrowDown") {
              e.preventDefault();
              setCursor((c) => Math.min(c + 1, matches.length - 1));
            } else if (e.key === "ArrowUp") {
              e.preventDefault();
              setCursor((c) => Math.max(c - 1, 0));
            } else if (e.key === "Enter" && matches[cursor]) {
              e.preventDefault();
              pick(matches[cursor].t);
            } else if (e.key === "Escape") {
              setOpen(false);
            }
          }}
          placeholder="Find a tag…"
          aria-label="Filter by tag"
          className="w-full bg-transparent text-sm outline-none"
          style={{ color: "var(--text-primary)" }}
        />
      </div>
      {open && (
        <div
          className="absolute z-50 mt-2 w-full rounded-xl border shadow-xl overflow-hidden"
          style={{
            backgroundColor: "var(--surface)",
            borderColor: "var(--border)",
          }}
        >
          {matches.length === 0 ? (
            <div
              className="px-3 py-3 text-sm"
              style={{ color: "var(--text-secondary)" }}
            >
              No tag matches in these results
            </div>
          ) : (
            <ul className="max-h-72 overflow-y-auto py-1" role="listbox">
              {matches.map((m, i) => (
                <li key={m.t}>
                  <button
                    type="button"
                    role="option"
                    aria-selected={i === cursor}
                    onMouseEnter={() => setCursor(i)}
                    onClick={() => pick(m.t)}
                    className="w-full flex items-center justify-between gap-3 px-3 py-2 text-sm text-left"
                    style={{
                      backgroundColor:
                        i === cursor ? "var(--accent-subtle)" : "transparent",
                      color: "var(--text-primary)",
                    }}
                  >
                    <span className="truncate">#{m.t}</span>
                    <Count n={m.n} />
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}

/**
 * Search + filters for a blog listing. Three rows, each with one job:
 *   1. find: search box and sort
 *   2. browse: topic chips (busiest first, live counts)
 *   3. narrow: tag finder, reading length, page-specific extras
 * then the active filters as removable pills, and "refine by" tag suggestions
 * drawn from the current results.
 */
export default function BlogFilterBar({
  searchTerm,
  setSearchTerm,
  placeholder,
  correctedQuery,
  topics,
  selectedTopics,
  setSelectedTopics,
  filters,
  resultCount,
  totalCount,
  noun = "articles",
  extra,
  extraPills = [],
  onClearExtra,
  hint,
}: BlogFilterBarProps) {
  const [showAllTopics, setShowAllTopics] = useState(false);

  const topicChips = useMemo(() => {
    const withCounts = topics.map((t) => ({
      ...t,
      count: filters.facets.topics.get(t.slug) ?? 0,
    }));
    // Selected topics lead (they must stay visible in the mobile scroll row),
    // then the busiest.
    const picked = (slug: string) => (selectedTopics.includes(slug) ? 0 : 1);
    withCounts.sort(
      (a, b) =>
        picked(a.slug) - picked(b.slug) ||
        b.count - a.count ||
        a.name.localeCompare(b.name),
    );
    return withCounts;
  }, [topics, filters.facets.topics, selectedTopics]);

  const visibleTopics = showAllTopics
    ? topicChips
    : [
        ...topicChips.slice(0, TOPIC_PREVIEW),
        // A selected topic never hides behind "+N more".
        ...topicChips
          .slice(TOPIC_PREVIEW)
          .filter((t) => selectedTopics.includes(t.slug)),
      ];
  const hiddenCount = topicChips.length - visibleTopics.length;

  const toggleTopic = (slug: string) =>
    setSelectedTopics(
      selectedTopics.includes(slug)
        ? selectedTopics.filter((s) => s !== slug)
        : [...selectedTopics, slug],
    );

  const topicName = (slug: string) =>
    topics.find((t) => t.slug === slug)?.name ?? slug;

  const pills = [
    ...(searchTerm.trim()
      ? [{ label: `“${searchTerm.trim()}”`, onRemove: () => setSearchTerm("") }]
      : []),
    ...selectedTopics.map((s) => ({
      label: topicName(s),
      onRemove: () => toggleTopic(s),
    })),
    ...filters.tags.map((t) => ({
      label: `#${t}`,
      onRemove: () => filters.toggleTag(t),
    })),
    ...filters.lengths.map((l) => ({
      label: LENGTH_BUCKETS.find((b) => b.key === l)?.label ?? l,
      onRemove: () => filters.toggleLength(l),
    })),
    ...extraPills,
  ];

  const clearAll = () => {
    setSearchTerm("");
    setSelectedTopics([]);
    filters.clear();
    onClearExtra?.();
  };

  return (
    <div className="mb-12 space-y-4">
      {/* 1. Find */}
      <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-center gap-3">
        <BlogSearchBar
          value={searchTerm}
          onChange={setSearchTerm}
          placeholder={placeholder}
        />
        <label
          className="flex items-center gap-2 px-3 py-2.5 rounded-xl border text-sm"
          style={{
            backgroundColor: "var(--surface)",
            borderColor: "var(--border)",
          }}
        >
          <span style={{ color: "var(--text-secondary)" }}>Sort</span>
          <select
            value={filters.sort}
            onChange={(e) =>
              filters.setSort(e.target.value as typeof filters.sort)
            }
            className="bg-transparent outline-none font-medium cursor-pointer"
            style={{ color: "var(--text-primary)" }}
          >
            {SORT_OPTIONS.filter(
              (o) => o.key !== "relevance" || filters.hasQuery,
            ).map((o) => (
              <option key={o.key} value={o.key}>
                {o.label}
              </option>
            ))}
          </select>
        </label>
      </div>

      {correctedQuery && (
        <p
          className="text-center text-sm"
          style={{ color: "var(--text-secondary)" }}
        >
          Showing results for{" "}
          <button
            type="button"
            className="font-medium underline underline-offset-2"
            style={{ color: "var(--accent)" }}
            onClick={() => setSearchTerm(correctedQuery)}
          >
            {correctedQuery}
          </button>
        </p>
      )}
      {hint}

      {/* 2. Browse by topic */}
      {topicChips.length > 1 && (
        <div
          className="flex items-center gap-2 overflow-x-auto -mx-6 px-6 pb-1 sm:mx-0 sm:px-0 sm:pb-0 sm:flex-wrap sm:justify-center sm:overflow-visible"
          style={{ scrollbarWidth: "none" }}
        >
          <Label>Topics</Label>
          <Chip
            active={selectedTopics.length === 0}
            onClick={() => setSelectedTopics([])}
          >
            All
          </Chip>
          {visibleTopics.map((t) => {
            const active = selectedTopics.includes(t.slug);
            return (
              <Chip
                key={t.slug}
                active={active}
                disabled={t.count === 0 && !active}
                onClick={() => toggleTopic(t.slug)}
              >
                {t.name}
                <Count n={t.count} active={active} />
              </Chip>
            );
          })}
          {(hiddenCount > 0 || showAllTopics) && (
            <button
              type="button"
              onClick={() => setShowAllTopics(!showAllTopics)}
              className="shrink-0 px-3 py-1.5 text-sm font-medium"
              style={{ color: "var(--accent)" }}
            >
              {showAllTopics ? "Show fewer" : `+${hiddenCount} more`}
            </button>
          )}
        </div>
      )}

      {/* 3. Narrow */}
      <div className="flex flex-col sm:flex-row flex-wrap items-stretch sm:items-center justify-center gap-3">
        <TagFinder
          counts={filters.facets.tags}
          selected={filters.tags}
          onPick={filters.toggleTag}
        />
        <div className="flex flex-wrap items-center justify-center gap-2">
          <Label>Length</Label>
          {LENGTH_BUCKETS.map((b) => {
            const active = filters.lengths.includes(b.key);
            const n = filters.facets.lengths.get(b.key) ?? 0;
            return (
              <Chip
                key={b.key}
                active={active}
                disabled={n === 0 && !active}
                onClick={() => filters.toggleLength(b.key)}
              >
                {b.label}
                <Count n={n} active={active} />
              </Chip>
            );
          })}
        </div>
        {extra}
      </div>

      {/* Refine: tags common in the current results */}
      {(pills.length > 0 || searchTerm.trim()) &&
        filters.suggestedTags.length > 0 && (
          <div
            className="flex items-center gap-2 overflow-x-auto -mx-6 px-6 pb-1 sm:mx-0 sm:px-0 sm:pb-0 sm:flex-wrap sm:justify-center sm:overflow-visible"
            style={{ scrollbarWidth: "none" }}
          >
            <Label>Refine</Label>
            {filters.suggestedTags.map(({ tag, count }) => (
              <button
                key={tag}
                type="button"
                onClick={() => filters.toggleTag(tag)}
                className="shrink-0 inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs transition-colors"
                style={{
                  backgroundColor: "var(--accent-subtle)",
                  color: "var(--accent)",
                }}
              >
                #{tag}
                <Count n={count} />
              </button>
            ))}
          </div>
        )}

      {/* Active filters */}
      {pills.length > 0 && (
        <div
          className="flex flex-wrap items-center justify-center gap-2 pt-4 border-t"
          style={{ borderColor: "var(--border)" }}
        >
          <span
            className="text-sm mr-1"
            style={{ color: "var(--text-secondary)" }}
          >
            <strong style={{ color: "var(--text-primary)" }}>
              {resultCount.toLocaleString("en-US")}
            </strong>{" "}
            of {totalCount.toLocaleString("en-US")} {noun}
          </span>
          {pills.map((p) => (
            <button
              key={p.label}
              type="button"
              onClick={p.onRemove}
              aria-label={`Remove filter ${p.label}`}
              className="inline-flex items-center gap-1.5 pl-3 pr-2 py-1 rounded-full text-sm border"
              style={{
                borderColor: "var(--accent)",
                color: "var(--accent)",
                backgroundColor: "var(--accent-subtle)",
              }}
            >
              {p.label}
              <span aria-hidden className="text-base leading-none">
                ×
              </span>
            </button>
          ))}
          <button
            type="button"
            onClick={clearAll}
            className="px-2 py-1 text-sm font-medium underline underline-offset-2"
            style={{ color: "var(--text-secondary)" }}
          >
            Clear all
          </button>
        </div>
      )}
    </div>
  );
}
