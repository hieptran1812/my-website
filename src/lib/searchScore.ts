/**
 * Shared blog-search relevance scoring.
 *
 * Ranking is strictly tiered so a match's *location* decides its rank before
 * anything else:
 *
 *     title match  ≫  tag match  ≫  content (excerpt + body) match
 *
 * Any post that matches in the title outranks every post that only matches in
 * tags, which in turn outranks every post that only matches in the body. This
 * is enforced with order-of-magnitude tier weights: the maximum a lower tier
 * can contribute is always smaller than the minimum a single higher-tier match
 * contributes, so the tiers never cross. Within a tier, finer signals (whole-
 * query phrase match, word-boundary match, token coverage, body density) break
 * ties — and a post matching in several fields still edges out one matching in
 * only the top field, because the tiers are additive.
 *
 * Matching is accent-insensitive (Vietnamese diacritics + đ are folded) so a
 * query typed without tone marks still finds accented text.
 */

// Tier weights. Each is far larger than the highest possible score the tier
// below it can reach (a single field's relevance is capped at FIELD_CAP), so a
// higher-tier match can never be overtaken by lower-tier signal.
const TITLE_WEIGHT = 1_000_000;
const TAG_WEIGHT = 1_000;
const CONTENT_WEIGHT = 1;

// Per-field relevance ceiling. Kept well under TAG_WEIGHT (1_000) so that even
// a maxed-out content field (CONTENT_WEIGHT * FIELD_CAP) stays below a single
// tag match, and a maxed-out tag field (TAG_WEIGHT * FIELD_CAP) stays below a
// single title match.
const FIELD_CAP = 500;

/** A query parsed once and reused across every candidate. */
export interface QuerySpec {
  /** Normalized full query string — used for whole-phrase matching. */
  raw: string;
  /** Normalized tokens (each length >= 2). */
  tokens: string[];
}

/** Raw searchable fields for one post. */
export interface ScoreInput {
  title: string;
  tags: string[];
  excerpt?: string;
  /** Full markdown body. Omit where it isn't available (e.g. listing pages). */
  body?: string;
}

/** Pre-normalized fields, so normalization happens once per post, not per key. */
export interface NormalizedFields {
  title: string;
  tags: string;
  content: string;
}

/** Lowercase + strip diacritics (incl. Vietnamese đ) for accent-insensitive matching. */
export function normalize(s: string): string {
  return s
    .toLowerCase()
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "") // strip combining diacritical marks
    .replace(/đ/g, "d"); // đ (not a combining mark, so map explicitly)
}

/** Parse a raw user query into a reusable spec. Returns empty tokens if too short. */
export function parseQuery(q: string): QuerySpec {
  const raw = normalize(q.trim());
  const tokens = raw.split(/\s+/).filter((t) => t.length >= 2);
  return { raw, tokens };
}

/** True when the query is substantive enough to score against. */
export function hasQuery(query: QuerySpec): boolean {
  return query.raw.length >= 2 && query.tokens.length > 0;
}

export function normalizeFields(input: ScoreInput): NormalizedFields {
  return {
    title: normalize(input.title || ""),
    tags: normalize((input.tags || []).join(" ")),
    content: normalize(`${input.excerpt || ""} ${input.body || ""}`),
  };
}

/** Whole-word (boundary-delimited) presence of `needle` in already-normalized `text`. */
function boundaryMatch(text: string, needle: string): boolean {
  const escaped = needle.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return new RegExp(`(?:^|[^a-z0-9])${escaped}(?:[^a-z0-9]|$)`).test(text);
}

function countOccurrences(text: string, needle: string): number {
  return text.split(needle).length - 1;
}

/**
 * Relevance of a single normalized field to the query, in [0, FIELD_CAP].
 * `countCap` > 0 rewards repeated occurrences (use it for the body only).
 */
function fieldRelevance(
  text: string,
  query: QuerySpec,
  countCap: number,
): number {
  if (!text) return 0;
  const { raw, tokens } = query;
  let score = 0;

  // Whole-phrase match (only meaningful for multi-word queries, but harmless
  // for single-word ones where raw === the sole token).
  if (raw.length >= 2 && text.includes(raw)) {
    score += 30;
    if (boundaryMatch(text, raw)) score += 10;
  }

  let matched = 0;
  for (const t of tokens) {
    if (!text.includes(t)) continue;
    matched++;
    score += 4;
    if (boundaryMatch(text, t)) score += 2;
    if (countCap > 0) {
      score += Math.min(countOccurrences(text, t) - 1, countCap);
    }
  }
  if (matched === 0) return 0;

  // Reward covering the whole query in one field.
  if (matched === tokens.length) score += 10;

  return Math.min(score, FIELD_CAP);
}

/** Score pre-normalized fields. Returns 0 when nothing matches. */
export function scoreNormalized(
  fields: NormalizedFields,
  query: QuerySpec,
): number {
  const title = fieldRelevance(fields.title, query, 0);
  const tags = fieldRelevance(fields.tags, query, 0);
  const content = fieldRelevance(fields.content, query, 6);
  if (title === 0 && tags === 0 && content === 0) return 0;
  return title * TITLE_WEIGHT + tags * TAG_WEIGHT + content * CONTENT_WEIGHT;
}

/** Convenience: normalize then score. Prefer the split form in hot loops. */
export function scoreEntry(input: ScoreInput, query: QuerySpec): number {
  return scoreNormalized(normalizeFields(input), query);
}
