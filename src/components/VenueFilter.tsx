"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  conferenceName,
  conferencesByField,
  FIELD_LABELS,
  TIER_LABELS,
  type VenueTier,
} from "@/lib/venue";
import {
  isEmptySelection,
  type FacetOption,
  type VenueSelection,
} from "@/components/hooks/useVenueFilter";

interface VenueFilterProps {
  selection: VenueSelection;
  onChange: (next: VenueSelection) => void;
  facets: {
    venued: number;
    confs: FacetOption<string>[];
    years: FacetOption<number>[];
    tiers: FacetOption<VenueTier>[];
  };
}

function toggle<T>(list: T[], value: T): T[] {
  return list.includes(value)
    ? list.filter((v) => v !== value)
    : [...list, value];
}

function Chip({
  label,
  count,
  active,
  onClick,
}: {
  label: string;
  count: number;
  active: boolean;
  onClick: () => void;
}) {
  const disabled = count === 0 && !active;
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-pressed={active}
      className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium border transition-colors duration-150 disabled:opacity-40 disabled:cursor-not-allowed"
      style={{
        backgroundColor: active ? "var(--accent)" : "var(--card-bg)",
        borderColor: active ? "var(--accent)" : "var(--border)",
        color: active ? "white" : "var(--text-primary)",
      }}
    >
      <span>{label}</span>
      <span className="tabular-nums" style={{ opacity: active ? 0.85 : 0.6 }}>
        {count}
      </span>
    </button>
  );
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="px-3 py-2.5">
      <div
        className="text-[11px] font-semibold uppercase tracking-wider mb-2"
        style={{ color: "var(--text-secondary)" }}
      >
        {title}
      </div>
      <div className="flex flex-wrap gap-1.5">{children}</div>
    </div>
  );
}

export function selectionLabel(sel: VenueSelection): string {
  const parts = [
    sel.confs.map(conferenceName).join(" / "),
    sel.years
      .slice()
      .sort((a, b) => b - a)
      .join(" / "),
    sel.tiers.map((t) => TIER_LABELS[t]).join(" / "),
  ].filter(Boolean);
  return parts.join(" · ");
}

export default function VenueFilter({
  selection,
  onChange,
  facets,
}: VenueFilterProps) {
  const [isOpen, setIsOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const close = useCallback(() => setIsOpen(false), []);

  useEffect(() => {
    if (!isOpen) return;
    const onClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) close();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
    };
    document.addEventListener("mousedown", onClick);
    window.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onClick);
      window.removeEventListener("keydown", onKey);
    };
  }, [isOpen, close]);

  if (facets.venued === 0) return null;

  // Every top venue is listed (dimmed at 0) so the reader sees the coverage;
  // other venues appear once a post has them. Busiest first within a field.
  const counts = new Map(facets.confs.map((c) => [c.value, c.count]));
  const conferenceGroups = conferencesByField()
    .map(({ field, confs }) => ({
      field,
      confs: confs
        .filter(
          (c) =>
            c.top || counts.has(c.slug) || selection.confs.includes(c.slug),
        )
        .map((c, order) => ({
          slug: c.slug,
          count: counts.get(c.slug) ?? 0,
          order,
        }))
        .sort((a, b) => b.count - a.count || a.order - b.order),
    }))
    .filter((g) => g.confs.length > 0)
    .sort(
      (a, b) =>
        b.confs.reduce((s, c) => s + c.count, 0) -
        a.confs.reduce((s, c) => s + c.count, 0),
    );

  const empty = isEmptySelection(selection);
  const activeCount =
    selection.confs.length + selection.years.length + selection.tiers.length;
  const topTier: VenueTier[] = ["award", "oral", "spotlight"];
  const topTierActive =
    selection.tiers.length > 0 &&
    selection.tiers.every((t) => topTier.includes(t));

  return (
    <div className="relative inline-block" ref={ref}>
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        aria-expanded={isOpen}
        className="flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium border transition-all duration-200 hover:shadow-md"
        style={{
          backgroundColor: "var(--surface)",
          borderColor: isOpen ? "var(--accent)" : "var(--border)",
          color: "var(--text-primary)",
          minWidth: "180px",
          maxWidth: "min(320px, calc(100vw - 32px))",
        }}
      >
        <svg
          className="w-4 h-4 shrink-0"
          style={{ color: "var(--accent)" }}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M12 14l9-5-9-5-9 5 9 5zm0 0v7m-6-10.5V16c0 1.5 2.7 3 6 3s6-1.5 6-3v-5.5"
          />
        </svg>
        <span className="truncate flex-1 text-left">
          {empty ? "All conferences" : selectionLabel(selection)}
        </span>
        {!empty && (
          <span
            className="px-1.5 py-0.5 rounded-md text-xs font-bold"
            style={{ backgroundColor: "var(--accent)", color: "white" }}
          >
            {activeCount}
          </span>
        )}
        <svg
          className={`w-4 h-4 shrink-0 transition-transform duration-200 ${isOpen ? "rotate-180" : ""}`}
          style={{ color: "var(--text-secondary)" }}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M19 9l-7 7-7-7"
          />
        </svg>
      </button>

      {isOpen && (
        <div
          className="absolute z-50 mt-2 right-0 sm:right-auto sm:left-0 rounded-xl border shadow-xl overflow-hidden"
          style={{
            backgroundColor: "var(--surface)",
            borderColor: "var(--border)",
            width: "min(340px, calc(100vw - 32px))",
          }}
        >
          <div className="max-h-[70vh] overflow-y-auto py-1">
            <Section title="Quick picks">
              <Chip
                label="Orals, spotlights & awards"
                count={facets.tiers
                  .filter((t) => topTier.includes(t.value))
                  .reduce((s, t) => s + t.count, 0)}
                active={topTierActive}
                onClick={() =>
                  onChange({
                    ...selection,
                    tiers: topTierActive ? [] : topTier,
                  })
                }
              />
            </Section>
            <Section title="Conference">
              <div className="w-full space-y-2.5">
                {conferenceGroups.map(({ field, confs }) => (
                  <div key={field}>
                    <div
                      className="text-[11px] mb-1.5"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      {FIELD_LABELS[field]}
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {confs.map((c) => (
                        <Chip
                          key={c.slug}
                          label={conferenceName(c.slug)}
                          count={c.count}
                          active={selection.confs.includes(c.slug)}
                          onClick={() =>
                            onChange({
                              ...selection,
                              confs: toggle(selection.confs, c.slug),
                            })
                          }
                        />
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </Section>
            {facets.years.length > 0 && (
              <Section title="Year">
                {facets.years.map((y) => (
                  <Chip
                    key={y.value}
                    label={String(y.value)}
                    count={y.count}
                    active={selection.years.includes(y.value)}
                    onClick={() =>
                      onChange({
                        ...selection,
                        years: toggle(selection.years, y.value),
                      })
                    }
                  />
                ))}
              </Section>
            )}
            {facets.tiers.length > 0 && (
              <Section title="Acceptance">
                {facets.tiers.map((t) => (
                  <Chip
                    key={t.value}
                    label={TIER_LABELS[t.value]}
                    count={t.count}
                    active={selection.tiers.includes(t.value)}
                    onClick={() =>
                      onChange({
                        ...selection,
                        tiers: toggle(selection.tiers, t.value),
                      })
                    }
                  />
                ))}
              </Section>
            )}
          </div>
          {!empty && (
            <div
              className="p-2 border-t"
              style={{ borderColor: "var(--border)" }}
            >
              <button
                type="button"
                onClick={() => onChange({ confs: [], years: [], tiers: [] })}
                className="w-full px-3 py-2 rounded-lg text-sm font-medium"
                style={{ color: "var(--accent)" }}
              >
                Clear conference filter
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
