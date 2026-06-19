"use client";

// A compact, searchable language picker with a flag in front of each option.
// Replaces the plain <select> in the Reading Options "Translate Article" panel.
// It expands inline (rather than as an absolutely-positioned popover) so it is
// never clipped by the sidebar's / mobile sheet's overflow containers, and it
// is fully theme-aware via the same CSS variables the rest of the panel uses.

import React, { useEffect, useMemo, useRef, useState } from "react";
import { LANGUAGES } from "./translate";

interface LanguageSelectProps {
  value: string;
  onChange: (code: string) => void;
  disabled?: boolean;
  /** Tighter sizing for the narrow desktop sidebar. */
  compact?: boolean;
}

export default function LanguageSelect({
  value,
  onChange,
  disabled = false,
  compact = false,
}: LanguageSelectProps) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);
  const rootRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  const selected = useMemo(
    () => LANGUAGES.find((l) => l.code === value) ?? LANGUAGES[0],
    [value],
  );

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return LANGUAGES;
    return LANGUAGES.filter(
      (l) =>
        l.label.toLowerCase().includes(q) ||
        l.native.toLowerCase().includes(q) ||
        l.code.toLowerCase().includes(q),
    );
  }, [query]);

  // Close when clicking outside the picker.
  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [open]);

  // Reset + focus search each time it opens.
  useEffect(() => {
    if (!open) return;
    setQuery("");
    const idx = LANGUAGES.findIndex((l) => l.code === value);
    setActiveIndex(idx >= 0 ? idx : 0);
    const t = setTimeout(() => inputRef.current?.focus(), 20);
    return () => clearTimeout(t);
  }, [open, value]);

  // Keep the highlighted row within the filtered range.
  useEffect(() => {
    setActiveIndex((i) => Math.min(i, Math.max(0, filtered.length - 1)));
  }, [filtered.length]);

  // Scroll the highlighted row into view.
  useEffect(() => {
    if (!open || !listRef.current) return;
    const el = listRef.current.querySelector(
      `[data-idx="${activeIndex}"]`,
    ) as HTMLElement | null;
    el?.scrollIntoView({ block: "nearest" });
  }, [activeIndex, open]);

  const choose = (code: string) => {
    onChange(code);
    setOpen(false);
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Escape") {
      setOpen(false);
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      setActiveIndex((i) => Math.min(i + 1, filtered.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActiveIndex((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter") {
      e.preventDefault();
      const item = filtered[activeIndex];
      if (item) choose(item.code);
    }
  };

  const textSize = compact ? "text-xs" : "text-sm";

  return (
    <div ref={rootRef} className="relative">
      {/* Trigger */}
      <button
        type="button"
        disabled={disabled}
        onClick={() => setOpen((o) => !o)}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-label="Target language for translation"
        className={`w-full rounded-lg flex items-center gap-2 transition-colors ${
          compact ? "px-2.5 py-2" : "px-3 py-2.5"
        } ${textSize}`}
        style={{
          backgroundColor: "var(--surface)",
          color: "var(--text-primary)",
          border: `1px solid ${open ? "var(--accent)" : "var(--border)"}`,
          cursor: disabled ? "default" : "pointer",
          opacity: disabled ? 0.6 : 1,
        }}
      >
        <span className="text-base leading-none shrink-0">{selected.flag}</span>
        <span className="font-medium truncate">{selected.label}</span>
        <span className="truncate" style={{ color: "var(--text-secondary)" }}>
          · {selected.native}
        </span>
        <svg
          className={`ml-auto w-4 h-4 shrink-0 transition-transform duration-200 ${
            open ? "rotate-180" : ""
          }`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
          style={{ color: "var(--text-secondary)" }}
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth="2"
            d="M19 9l-7 7-7-7"
          />
        </svg>
      </button>

      {/* Inline panel: search + scrollable list */}
      {open && (
        <div
          className="mt-2 rounded-lg shadow-lg overflow-hidden"
          style={{
            border: "1px solid var(--border)",
            backgroundColor: "var(--background)",
          }}
        >
          <div
            className="p-2 border-b"
            style={{ borderColor: "var(--border)" }}
          >
            <div
              className="flex items-center gap-2 rounded-md px-2"
              style={{ backgroundColor: "var(--surface)" }}
            >
              <svg
                className="w-4 h-4 shrink-0"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                style={{ color: "var(--text-secondary)" }}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M21 21l-4.35-4.35M17 11a6 6 0 11-12 0 6 6 0 0112 0z"
                />
              </svg>
              <input
                ref={inputRef}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={onKeyDown}
                placeholder="Search language…"
                className={`w-full bg-transparent outline-none py-1.5 ${textSize}`}
                style={{ color: "var(--text-primary)" }}
                aria-label="Search language"
              />
            </div>
          </div>

          <div
            ref={listRef}
            role="listbox"
            className="max-h-56 overflow-y-auto py-1"
          >
            {filtered.length === 0 ? (
              <div
                className={`px-3 py-3 ${textSize}`}
                style={{ color: "var(--text-secondary)" }}
              >
                No matching language
              </div>
            ) : (
              filtered.map((l, idx) => {
                const isSel = l.code === value;
                const isActive = idx === activeIndex;
                return (
                  <button
                    key={l.code}
                    type="button"
                    data-idx={idx}
                    role="option"
                    aria-selected={isSel}
                    onMouseEnter={() => setActiveIndex(idx)}
                    onClick={() => choose(l.code)}
                    className={`w-full flex items-center gap-2.5 px-3 py-2 text-left ${textSize}`}
                    style={{
                      backgroundColor: isActive
                        ? "var(--surface)"
                        : "transparent",
                      color: "var(--text-primary)",
                    }}
                  >
                    <span className="text-base leading-none shrink-0">
                      {l.flag}
                    </span>
                    <span className="font-medium">{l.label}</span>
                    <span
                      className="truncate"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      {l.native}
                    </span>
                    {isSel && (
                      <svg
                        className="ml-auto w-4 h-4 shrink-0"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                        style={{ color: "var(--accent)" }}
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth="2"
                          d="M5 13l4 4L19 7"
                        />
                      </svg>
                    )}
                  </button>
                );
              })
            )}
          </div>
        </div>
      )}
    </div>
  );
}
