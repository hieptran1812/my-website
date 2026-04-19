import type { Highlight } from "./types";

const KEY_PREFIX = "blog-highlights:";

export function loadHighlights(slug: string): Highlight[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(KEY_PREFIX + slug);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed as Highlight[];
  } catch {
    return [];
  }
}

export function saveHighlights(slug: string, highlights: Highlight[]): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(
      KEY_PREFIX + slug,
      JSON.stringify(highlights),
    );
  } catch {
    // quota exceeded or disabled
  }
}

export function createId(): string {
  return (
    Date.now().toString(36) + Math.random().toString(36).slice(2, 8)
  );
}
