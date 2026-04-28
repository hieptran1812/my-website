"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { Command } from "cmdk";

interface SearchHit {
  slug: string;
  title: string;
  category: string;
  subcategory: string;
  snippet: string;
}

const NAV_LINKS = [
  { label: "Home", href: "/" },
  { label: "Blog", href: "/blog" },
  { label: "Projects", href: "/projects" },
  { label: "About", href: "/about" },
  { label: "Contact", href: "/contact" },
  { label: "Tags", href: "/blog/tags" },
  { label: "Paper Reading", href: "/blog/paper-reading" },
  { label: "Machine Learning", href: "/blog/machine-learning" },
  { label: "Software Development", href: "/blog/software-development" },
  { label: "Notes", href: "/blog/notes" },
];

export default function CommandPalette() {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<SearchHit[]>([]);
  const [loading, setLoading] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const isModK = (e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k";
      if (isModK) {
        e.preventDefault();
        setOpen((v) => !v);
      } else if (e.key === "Escape" && open) {
        setOpen(false);
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open]);

  // Debounced search.
  useEffect(() => {
    if (!open) return;
    const trimmed = query.trim();
    if (trimmed.length < 2) {
      setHits([]);
      setLoading(false);
      return;
    }
    setLoading(true);
    const handle = setTimeout(async () => {
      abortRef.current?.abort();
      const ctrl = new AbortController();
      abortRef.current = ctrl;
      try {
        const res = await fetch(
          `/api/blog/search?q=${encodeURIComponent(trimmed)}&limit=8`,
          { signal: ctrl.signal },
        );
        if (!res.ok) throw new Error("search failed");
        const data = await res.json();
        setHits(data.hits || []);
      } catch (err) {
        if ((err as Error).name !== "AbortError") setHits([]);
      } finally {
        setLoading(false);
      }
    }, 180);
    return () => clearTimeout(handle);
  }, [query, open]);

  // Reset state when closed.
  useEffect(() => {
    if (!open) {
      setQuery("");
      setHits([]);
    }
  }, [open]);

  const go = (href: string) => {
    setOpen(false);
    router.push(href);
  };

  if (!open) return null;

  return (
    <div
      className="cmdk-overlay"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) setOpen(false);
      }}
    >
      <Command
        className="cmdk-dialog"
        label="Site command palette"
        shouldFilter={false}
      >
        <Command.Input
          autoFocus
          placeholder="Search posts, jump to a page…"
          value={query}
          onValueChange={setQuery}
          className="cmdk-input"
        />
        <Command.List className="cmdk-list">
          {loading && (
            <Command.Loading>
              <div className="cmdk-empty">Searching…</div>
            </Command.Loading>
          )}
          {!loading && query.trim().length >= 2 && hits.length === 0 && (
            <Command.Empty className="cmdk-empty">
              No posts match “{query.trim()}”.
            </Command.Empty>
          )}

          {hits.length > 0 && (
            <Command.Group heading="Posts" className="cmdk-group">
              <div className="cmdk-group-heading">Posts</div>
              {hits.map((h) => (
                <Command.Item
                  key={h.slug}
                  value={`post:${h.slug}`}
                  onSelect={() => go(`/blog/${h.slug}`)}
                  className="cmdk-item"
                >
                  <span style={{ fontWeight: 600 }}>{h.title}</span>
                  <span
                    className="cmdk-item-snippet"
                    dangerouslySetInnerHTML={{ __html: h.snippet }}
                  />
                  <span style={{ fontSize: "0.7rem", opacity: 0.6 }}>
                    {[h.category, h.subcategory].filter(Boolean).join(" · ")}
                  </span>
                </Command.Item>
              ))}
            </Command.Group>
          )}

          {(query.trim().length < 2 || hits.length === 0) && (
            <Command.Group heading="Navigate">
              <div className="cmdk-group-heading">Navigate</div>
              {NAV_LINKS.filter((l) =>
                query.trim().length === 0
                  ? true
                  : l.label.toLowerCase().includes(query.trim().toLowerCase()),
              ).map((l) => (
                <Command.Item
                  key={l.href}
                  value={`nav:${l.href}`}
                  onSelect={() => go(l.href)}
                  className="cmdk-item"
                >
                  <span style={{ fontWeight: 500 }}>{l.label}</span>
                  <span style={{ fontSize: "0.72rem", opacity: 0.6 }}>
                    {l.href}
                  </span>
                </Command.Item>
              ))}
            </Command.Group>
          )}
        </Command.List>
        <div className="cmdk-footer">
          <span>
            <kbd className="cmdk-kbd">↵</kbd> open
            <kbd className="cmdk-kbd">↑↓</kbd> nav
            <kbd className="cmdk-kbd">esc</kbd> close
          </span>
          <span>
            <kbd className="cmdk-kbd">⌘</kbd>
            <kbd className="cmdk-kbd">K</kbd>
          </span>
        </div>
      </Command>
    </div>
  );
}
