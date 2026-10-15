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

const PAGE = 10;
const MAX_RESULTS = 100;

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
  { label: "Finance & Markets", href: "/blog/trading" },
  { label: "Notes", href: "/blog/notes" },
];

export default function CommandPaletteDialog({
  onClose,
}: {
  onClose: () => void;
}) {
  const router = useRouter();
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<SearchHit[]>([]);
  const [total, setTotal] = useState(0);
  const [limit, setLimit] = useState(PAGE);
  const [corrected, setCorrected] = useState<string | undefined>();
  const [loading, setLoading] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    const trimmed = query.trim();
    if (trimmed.length < 2) {
      setHits([]);
      setTotal(0);
      setCorrected(undefined);
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
          `/api/blog/search?q=${encodeURIComponent(query)}&limit=${limit}`,
          { signal: ctrl.signal },
        );
        if (!res.ok) throw new Error("search failed");
        const data = await res.json();
        setHits(data.hits || []);
        setTotal(data.total || 0);
        setCorrected(data.correctedQuery);
      } catch (err) {
        if ((err as Error).name !== "AbortError") {
          setHits([]);
          setTotal(0);
          setCorrected(undefined);
        }
      } finally {
        setLoading(false);
      }
    }, 150);
    return () => clearTimeout(handle);
  }, [query, limit]);

  // A new query starts again from the first page.
  useEffect(() => setLimit(PAGE), [query]);

  const go = (href: string) => {
    onClose();
    router.push(href);
  };

  return (
    <div
      className="cmdk-overlay"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
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

          {corrected && hits.length > 0 && (
            <div className="cmdk-empty" style={{ padding: "6px 12px" }}>
              Showing results for{" "}
              <button
                type="button"
                onClick={() => setQuery(corrected)}
                style={{ fontWeight: 600, textDecoration: "underline" }}
              >
                {corrected}
              </button>
            </div>
          )}

          {hits.length > 0 && (
            <Command.Group heading="Posts" className="cmdk-group">
              <div className="cmdk-group-heading">
                Posts
                {total > hits.length && (
                  <span style={{ opacity: 0.6, fontWeight: 400 }}>
                    {" "}
                    · top {hits.length} of {total}
                  </span>
                )}
              </div>
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
              {total > hits.length && hits.length < MAX_RESULTS && (
                <Command.Item
                  value="more-results"
                  onSelect={() =>
                    setLimit((l) => Math.min(MAX_RESULTS, l + 20))
                  }
                  className="cmdk-item"
                >
                  <span style={{ fontWeight: 500, color: "var(--accent)" }}>
                    Show more results ({total - hits.length} more)
                  </span>
                </Command.Item>
              )}
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
            <kbd className="cmdk-kbd">/</kbd> search
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
