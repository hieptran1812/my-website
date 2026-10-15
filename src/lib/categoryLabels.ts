/**
 * Display names for the top-level blog categories. The slug is the URL (the
 * content folder), so a category is renamed here, never by moving its folder.
 */
export const CATEGORY_LABELS: Record<string, string> = {
  "machine-learning": "Machine Learning",
  "software-development": "Software Development",
  "paper-reading": "Paper Reading",
  trading: "Finance & Markets",
  notes: "Notes",
};

export function categoryLabel(slug: string | undefined | null): string {
  if (!slug) return "";
  const key = slug.toLowerCase();
  return (
    CATEGORY_LABELS[key] ??
    key
      .split(/[-_\s]+/)
      .filter(Boolean)
      .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
      .join(" ")
  );
}
