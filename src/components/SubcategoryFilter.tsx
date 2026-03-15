"use client";

import React, { useState, useRef, useEffect, useMemo, useCallback } from "react";

export interface CategoryOption {
  name: string;
  slug: string;
  count: number;
}

interface SubcategoryFilterProps {
  categories: CategoryOption[];
  selectedSlugs: string[];
  onSelectionChange: (slugs: string[]) => void;
  totalCount: number;
}

export default function SubcategoryFilter({
  categories,
  selectedSlugs,
  onSelectionChange,
  totalCount,
}: SubcategoryFilterProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [search, setSearch] = useState("");
  const dropdownRef = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);

  const closeDropdown = useCallback(() => {
    setIsOpen(false);
    setSearch("");
  }, []);

  // Manage dropdown: outside click, escape key, focus search
  useEffect(() => {
    if (!isOpen) return;

    searchRef.current?.focus();

    const handleClick = (e: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(e.target as Node)
      ) {
        closeDropdown();
      }
    };
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") closeDropdown();
    };

    document.addEventListener("mousedown", handleClick);
    window.addEventListener("keydown", handleKey);
    return () => {
      document.removeEventListener("mousedown", handleClick);
      window.removeEventListener("keydown", handleKey);
    };
  }, [isOpen, closeDropdown]);

  const filtered = useMemo(() => {
    if (!search.trim()) return categories;
    const q = search.toLowerCase();
    return categories.filter(
      (c) =>
        c.name.toLowerCase().includes(q) ||
        c.slug.toLowerCase().includes(q),
    );
  }, [categories, search]);

  const isAll = selectedSlugs.length === 0;

  const toggleCategory = (slug: string) => {
    if (selectedSlugs.includes(slug)) {
      onSelectionChange(selectedSlugs.filter((s) => s !== slug));
    } else {
      onSelectionChange([...selectedSlugs, slug]);
    }
  };

  const selectAll = () => {
    onSelectionChange([]);
  };

  const displayLabel = isAll
    ? `All (${totalCount})`
    : selectedSlugs.length === 1
      ? `${categories.find((c) => c.slug === selectedSlugs[0])?.name || selectedSlugs[0]}`
      : `${selectedSlugs.length} selected`;

  return (
    <div className="relative inline-block" ref={dropdownRef}>
      {/* Trigger Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium border transition-all duration-200 hover:shadow-md"
        style={{
          backgroundColor: "var(--surface)",
          borderColor: isOpen ? "var(--accent)" : "var(--border)",
          color: "var(--text-primary)",
          minWidth: "180px",
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
            d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z"
          />
        </svg>
        <span className="truncate flex-1 text-left">{displayLabel}</span>
        {!isAll && (
          <span
            className="px-1.5 py-0.5 rounded-md text-xs font-bold"
            style={{
              backgroundColor: "var(--accent)",
              color: "white",
            }}
          >
            {selectedSlugs.length}
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

      {/* Dropdown */}
      {isOpen && (
        <div
          className="absolute z-50 mt-2 w-72 rounded-xl border shadow-xl overflow-hidden"
          style={{
            backgroundColor: "var(--surface)",
            borderColor: "var(--border)",
          }}
        >
          {/* Search */}
          <div
            className="p-3 border-b"
            style={{ borderColor: "var(--border)" }}
          >
            <div
              className="flex items-center gap-2 px-3 py-2 rounded-lg"
              style={{
                backgroundColor: "var(--background)",
                border: "1px solid var(--border)",
              }}
            >
              <svg
                className="w-4 h-4 shrink-0"
                style={{ color: "var(--text-secondary)" }}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                />
              </svg>
              <input
                ref={searchRef}
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search subcategories..."
                className="w-full bg-transparent text-sm outline-none"
                style={{ color: "var(--text-primary)" }}
              />
              {search && (
                <button
                  onClick={() => setSearch("")}
                  className="shrink-0"
                  style={{ color: "var(--text-secondary)" }}
                >
                  <svg
                    className="w-3.5 h-3.5"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M6 18L18 6M6 6l12 12"
                    />
                  </svg>
                </button>
              )}
            </div>
          </div>

          {/* Options List */}
          <div className="max-h-64 overflow-y-auto p-1.5">
            {/* All option */}
            <button
              onClick={selectAll}
              className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors duration-150 text-left"
              style={{
                backgroundColor: isAll
                  ? "var(--surface-accent, var(--surface))"
                  : "transparent",
                color: "var(--text-primary)",
              }}
            >
              <div
                className="w-5 h-5 rounded-md border-2 flex items-center justify-center shrink-0 transition-colors duration-150"
                style={{
                  borderColor: isAll ? "var(--accent)" : "var(--border)",
                  backgroundColor: isAll ? "var(--accent)" : "transparent",
                }}
              >
                {isAll && (
                  <svg
                    className="w-3 h-3 text-white"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={3}
                      d="M5 13l4 4L19 7"
                    />
                  </svg>
                )}
              </div>
              <span className="flex-1 font-medium">All</span>
              <span
                className="text-xs font-semibold px-2 py-0.5 rounded-md"
                style={{
                  backgroundColor: "var(--accent)",
                  color: "white",
                }}
              >
                {totalCount}
              </span>
            </button>

            {/* Divider */}
            <div
              className="mx-3 my-1 border-b"
              style={{ borderColor: "var(--border)" }}
            />

            {/* Category options */}
            {filtered.length === 0 ? (
              <div
                className="px-3 py-4 text-sm text-center"
                style={{ color: "var(--text-secondary)" }}
              >
                No subcategories found
              </div>
            ) : (
              filtered.map((category) => {
                const isSelected = selectedSlugs.includes(category.slug);
                return (
                  <button
                    key={category.slug}
                    onClick={() => toggleCategory(category.slug)}
                    className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors duration-150 text-left"
                    style={{
                      backgroundColor: isSelected
                        ? "var(--surface-accent, var(--surface))"
                        : "transparent",
                      color: "var(--text-primary)",
                    }}
                  >
                    <div
                      className="w-5 h-5 rounded-md border-2 flex items-center justify-center shrink-0 transition-colors duration-150"
                      style={{
                        borderColor: isSelected
                          ? "var(--accent)"
                          : "var(--border)",
                        backgroundColor: isSelected
                          ? "var(--accent)"
                          : "transparent",
                      }}
                    >
                      {isSelected && (
                        <svg
                          className="w-3 h-3 text-white"
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={3}
                            d="M5 13l4 4L19 7"
                          />
                        </svg>
                      )}
                    </div>
                    <span className="flex-1">{category.name}</span>
                    <span
                      className="text-xs font-semibold px-2 py-0.5 rounded-md"
                      style={{
                        backgroundColor: isSelected
                          ? "var(--accent)"
                          : "var(--border)",
                        color: isSelected ? "white" : "var(--text-secondary)",
                      }}
                    >
                      {category.count}
                    </span>
                  </button>
                );
              })
            )}
          </div>

          {/* Footer: Clear */}
          {!isAll && (
            <div
              className="p-2 border-t"
              style={{ borderColor: "var(--border)" }}
            >
              <button
                onClick={selectAll}
                className="w-full px-3 py-2 rounded-lg text-sm font-medium transition-colors duration-150"
                style={{
                  color: "var(--accent)",
                }}
              >
                Clear all filters
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
