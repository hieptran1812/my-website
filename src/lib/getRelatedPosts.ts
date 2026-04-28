import fs from "fs";
import path from "path";
import matter from "gray-matter";
import { removeStopwords, eng, vie } from "stopword";
import { derivePostLocation } from "./postPath";

const blogDir = path.join(process.cwd(), "content", "blog");

// ─────────────── Public types ───────────────

export type RelatedReason =
  | "tags"
  | "similar"
  | "subcategory"
  | "category"
  | "series";

export interface RelatedPost {
  slug: string;
  title: string;
  excerpt: string;
  category: string;
  subcategory: string;
  publishDate: string;
  image?: string;
  score: number;
  /** 0-100, normalised against the top pick in the same batch. */
  relevancePercent: number;
  sharedTags: string[];
  reason: RelatedReason;
  /** Human-readable detail for the dominant signal, e.g. "Shares rare tag: casteer". */
  reasonDetail?: string;
  similarity: number;
}

export interface SeriesSibling {
  slug: string;
  title: string;
  publishDate: string;
  image?: string;
  position: number;
  total: number;
}

export interface SeriesContext {
  collection: string;
  prev?: SeriesSibling;
  next?: SeriesSibling;
  current: { position: number; total: number };
}

// ─────────────── Tunables ───────────────

const WEIGHTS = {
  tag: 1.0,
  cosine: 0.7,
  subcategory: 0.4,
  category: 0.15,
  collection: 2.0,
};
const RECENCY_HALF_LIFE_YEARS = 2;
const RECENCY_FLOOR = 0.7;
const MMR_LAMBDA = 0.7;
const MIN_RELEVANCE = 0.05;

// ─────────────── Corpus index ───────────────

interface IndexEntry {
  slug: string;
  title: string;
  excerpt: string;
  category: string;
  subcategory: string;
  tags: string[];
  publishDate: string;
  image?: string;
  collection?: string;
  /** Token frequency map for title+excerpt. */
  tokenTf: Map<string, number>;
  /** Norm of the TF-IDF vector (cached). */
  tfidfNorm: number;
}

interface CorpusIndex {
  entries: IndexEntry[];
  bySlug: Map<string, IndexEntry>;
  /** IDF for tags (Map<lowercased tag, idf weight>). */
  tagIdf: Map<string, number>;
  /** IDF for tokens (Map<token, idf weight>). */
  tokenIdf: Map<string, number>;
  /** Total document count. */
  N: number;
  /** Posts grouped by collection name (lowercased). */
  byCollection: Map<string, IndexEntry[]>;
}

let cachedIndex: CorpusIndex | null = null;
let cachedAt = 0;
const CACHE_TTL_MS = 5 * 60 * 1000;

const STOP_WORDS = new Set([...eng, ...vie]);
const TOKEN_RE = /[\p{L}\p{N}]+/gu;
// First markdown image reference: ![alt](url) — captures the URL.
const FIRST_BODY_IMAGE_RE = /!\[[^\]]*\]\(([^)\s]+)(?:\s+["'][^"']*["'])?\)/;

/** Pull the first inline markdown image URL out of the body, if any. */
export function extractFirstBodyImage(markdown: string): string | undefined {
  if (!markdown) return undefined;
  const m = FIRST_BODY_IMAGE_RE.exec(markdown);
  if (!m) return undefined;
  const url = m[1].trim();
  if (!url) return undefined;
  return url;
}

/**
 * Resolve a post's cover: frontmatter image wins, then first body image.
 * External URLs are skipped (next/image would need remotePatterns config).
 */
export function resolvePostCover(
  frontmatter: Record<string, unknown>,
  body: string,
): string | undefined {
  const candidates: (string | undefined)[] = [];
  const fm = frontmatter.image;
  if (typeof fm === "string" && fm.trim().length > 0) candidates.push(fm.trim());
  candidates.push(extractFirstBodyImage(body));
  for (const c of candidates) {
    if (!c) continue;
    if (/^https?:\/\//i.test(c)) continue; // remote — would need next.config remotePatterns
    return c;
  }
  return undefined;
}

function tokenize(text: string): string[] {
  if (!text) return [];
  const raw = (text.toLowerCase().match(TOKEN_RE) || []).filter(
    (t) => t.length >= 2 && t.length <= 30,
  );
  return removeStopwords(raw, [...STOP_WORDS]);
}

function termFreq(tokens: string[]): Map<string, number> {
  const tf = new Map<string, number>();
  for (const tok of tokens) tf.set(tok, (tf.get(tok) ?? 0) + 1);
  return tf;
}

function buildIndex(): CorpusIndex {
  const entries: IndexEntry[] = [];
  const tagDf = new Map<string, number>();
  const tokenDf = new Map<string, number>();

  const walk = (dir: string) => {
    if (!fs.existsSync(dir)) return;
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) walk(full);
      else if (entry.isFile() && entry.name.endsWith(".md")) {
        const raw = fs.readFileSync(full, "utf8");
        const { data, content: body } = matter(raw);
        const { category, subcategory } = derivePostLocation(full, data, blogDir);
        const slug = path
          .relative(blogDir, full)
          .replace(/\.md$/, "")
          .split(path.sep)
          .join("/");

        const title = data.title || slug;
        const excerpt = data.excerpt || data.description || "";
        const tags: string[] = Array.isArray(data.tags)
          ? data.tags.map((t) => String(t).toLowerCase())
          : [];
        const tokens = tokenize(`${title} ${excerpt}`);
        const tokenTf = termFreq(tokens);

        entries.push({
          slug,
          title,
          excerpt,
          category,
          subcategory,
          tags,
          publishDate: data.publishDate || data.date || "",
          image: resolvePostCover(data, body),
          collection: typeof data.collection === "string" ? data.collection : undefined,
          tokenTf,
          tfidfNorm: 0, // filled in pass 2
        });

        for (const t of new Set(tags)) tagDf.set(t, (tagDf.get(t) ?? 0) + 1);
        for (const tok of new Set(tokenTf.keys()))
          tokenDf.set(tok, (tokenDf.get(tok) ?? 0) + 1);
      }
    }
  };
  walk(blogDir);

  const N = entries.length;
  const tagIdf = new Map<string, number>();
  for (const [tag, df] of tagDf) tagIdf.set(tag, Math.log((N + 1) / (df + 1)) + 1);
  const tokenIdf = new Map<string, number>();
  for (const [tok, df] of tokenDf) tokenIdf.set(tok, Math.log((N + 1) / (df + 1)) + 1);

  // Pass 2: TF-IDF norms.
  for (const e of entries) {
    let sumSq = 0;
    for (const [tok, tf] of e.tokenTf) {
      const w = tf * (tokenIdf.get(tok) ?? 0);
      sumSq += w * w;
    }
    e.tfidfNorm = Math.sqrt(sumSq);
  }

  const bySlug = new Map<string, IndexEntry>();
  const byCollection = new Map<string, IndexEntry[]>();
  for (const e of entries) {
    bySlug.set(e.slug, e);
    if (e.collection) {
      const key = e.collection.toLowerCase();
      const arr = byCollection.get(key) ?? [];
      arr.push(e);
      byCollection.set(key, arr);
    }
  }
  // Sort each collection by publishDate ascending for stable position numbers.
  for (const arr of byCollection.values()) {
    arr.sort((a, b) => {
      const da = Date.parse(a.publishDate) || 0;
      const db = Date.parse(b.publishDate) || 0;
      return da - db;
    });
  }

  return { entries, bySlug, tagIdf, tokenIdf, N, byCollection };
}

function getIndex(): CorpusIndex {
  const now = Date.now();
  if (cachedIndex && now - cachedAt < CACHE_TTL_MS) return cachedIndex;
  cachedIndex = buildIndex();
  cachedAt = now;
  return cachedIndex;
}

// ─────────────── Similarity primitives ───────────────

function cosineSimilarity(
  a: IndexEntry,
  b: IndexEntry,
  tokenIdf: Map<string, number>,
): number {
  if (a.tfidfNorm === 0 || b.tfidfNorm === 0) return 0;
  // Iterate over the smaller map.
  const [s, l] =
    a.tokenTf.size < b.tokenTf.size ? [a.tokenTf, b.tokenTf] : [b.tokenTf, a.tokenTf];
  let dot = 0;
  for (const [tok, tfA] of s) {
    const tfB = l.get(tok);
    if (!tfB) continue;
    const idf = tokenIdf.get(tok) ?? 0;
    dot += tfA * tfB * idf * idf;
  }
  return dot / (a.tfidfNorm * b.tfidfNorm);
}

function recencyFactor(publishDate: string): number {
  const t = Date.parse(publishDate);
  if (Number.isNaN(t)) return RECENCY_FLOOR;
  const ageYears = (Date.now() - t) / (365.25 * 24 * 3600 * 1000);
  if (ageYears < 0) return 1;
  return RECENCY_FLOOR + (1 - RECENCY_FLOOR) * Math.exp(-ageYears / RECENCY_HALF_LIFE_YEARS);
}

interface CandidateScore {
  entry: IndexEntry;
  relevance: number;
  similarity: number;
  sharedTags: string[];
  rareSharedTag?: string;
  rareSharedTagWeight: number;
  reason: RelatedReason;
  reasonDetail?: string;
}

function scoreCandidate(
  current: IndexEntry,
  candidate: IndexEntry,
  idx: CorpusIndex,
): CandidateScore | null {
  // IDF-weighted shared tags.
  const currentTagSet = new Set(current.tags);
  const sharedTags = candidate.tags.filter((t) => currentTagSet.has(t));
  let tagScore = 0;
  let rareSharedTag: string | undefined;
  let rareSharedTagWeight = 0;
  for (const t of sharedTags) {
    const w = idx.tagIdf.get(t) ?? 0;
    tagScore += w;
    if (w > rareSharedTagWeight) {
      rareSharedTagWeight = w;
      rareSharedTag = t;
    }
  }
  // Normalize tag score to roughly [0,1] using max possible (all current tags rare).
  const maxTagScore =
    current.tags.reduce((s, t) => s + (idx.tagIdf.get(t) ?? 0), 0) || 1;
  const tagScoreNorm = Math.min(1, tagScore / maxTagScore);

  const cosine = cosineSimilarity(current, candidate, idx.tokenIdf);

  const sameSub =
    !!current.subcategory && current.subcategory === candidate.subcategory;
  const sameCat = !!current.category && current.category === candidate.category;
  const sameCollection =
    !!current.collection &&
    current.collection.toLowerCase() === (candidate.collection || "").toLowerCase();

  const raw =
    WEIGHTS.tag * tagScoreNorm +
    WEIGHTS.cosine * cosine +
    (sameSub ? WEIGHTS.subcategory : 0) +
    (sameCat ? WEIGHTS.category : 0) +
    (sameCollection ? WEIGHTS.collection : 0);

  const relevance = raw * recencyFactor(candidate.publishDate);
  if (relevance < MIN_RELEVANCE && !sameCollection) return null;

  // Determine dominant reason — series first, then strongest signal.
  let reason: RelatedReason;
  let reasonDetail: string | undefined;
  if (sameCollection) {
    reason = "series";
    reasonDetail = current.collection;
  } else if (
    WEIGHTS.tag * tagScoreNorm >= WEIGHTS.cosine * cosine &&
    sharedTags.length > 0
  ) {
    reason = "tags";
    if (rareSharedTag && rareSharedTagWeight > 1.5) {
      reasonDetail = `Shares rare tag: ${rareSharedTag}`;
    } else if (sharedTags.length === 1) {
      reasonDetail = `Shares tag: ${sharedTags[0]}`;
    } else {
      reasonDetail = `Shares ${sharedTags.length} tags`;
    }
  } else if (cosine > 0.05) {
    reason = "similar";
    reasonDetail = "Similar topic";
  } else if (sameSub) {
    reason = "subcategory";
    reasonDetail = `Same subcategory: ${current.subcategory}`;
  } else {
    reason = "category";
    reasonDetail = `Same category: ${current.category}`;
  }

  return {
    entry: candidate,
    relevance,
    similarity: cosine,
    sharedTags,
    rareSharedTag,
    rareSharedTagWeight,
    reason,
    reasonDetail,
  };
}

function pairwiseSimilarity(
  a: CandidateScore,
  b: CandidateScore,
  tokenIdf: Map<string, number>,
): number {
  // Combine tag overlap (Jaccard) with title cosine for diversity penalty.
  const tagsA = new Set(a.entry.tags);
  const tagsB = new Set(b.entry.tags);
  let intersection = 0;
  for (const t of tagsA) if (tagsB.has(t)) intersection += 1;
  const union = tagsA.size + tagsB.size - intersection;
  const tagJaccard = union === 0 ? 0 : intersection / union;
  const cosine = cosineSimilarity(a.entry, b.entry, tokenIdf);
  return Math.max(tagJaccard, cosine);
}

function mmrSelect(
  candidates: CandidateScore[],
  k: number,
  tokenIdf: Map<string, number>,
): CandidateScore[] {
  const picked: CandidateScore[] = [];
  const remaining = [...candidates];
  while (picked.length < k && remaining.length > 0) {
    let bestIdx = 0;
    let bestScore = -Infinity;
    for (let i = 0; i < remaining.length; i++) {
      const c = remaining[i];
      let maxSim = 0;
      for (const p of picked) {
        const sim = pairwiseSimilarity(c, p, tokenIdf);
        if (sim > maxSim) maxSim = sim;
      }
      const mmr = MMR_LAMBDA * c.relevance - (1 - MMR_LAMBDA) * maxSim;
      if (mmr > bestScore) {
        bestScore = mmr;
        bestIdx = i;
      }
    }
    picked.push(remaining.splice(bestIdx, 1)[0]);
  }
  return picked;
}

// ─────────────── Public API ───────────────

export function getRelatedPosts(
  currentSlug: string,
  // currentTags/currentCategory/currentSubcategory kept for back-compat callers
  // but values are sourced authoritatively from the corpus index.
  _currentTags: string[],
  _currentCategory: string,
  _currentSubcategory: string,
  limit = 6,
): RelatedPost[] {
  const idx = getIndex();
  const current = idx.bySlug.get(currentSlug);
  if (!current) return [];

  const seriesSlugs = new Set<string>();
  if (current.collection) {
    const key = current.collection.toLowerCase();
    for (const sib of idx.byCollection.get(key) ?? []) seriesSlugs.add(sib.slug);
  }

  const scored: CandidateScore[] = [];
  for (const candidate of idx.entries) {
    if (candidate.slug === currentSlug) continue;
    if (seriesSlugs.has(candidate.slug)) continue; // series sits in its own module
    const s = scoreCandidate(current, candidate, idx);
    if (s) scored.push(s);
  }

  scored.sort((a, b) => b.relevance - a.relevance);
  // Pre-truncate to a reasonable pool before MMR for cheaper diversity selection.
  const pool = scored.slice(0, Math.max(limit * 4, 24));
  const selected = mmrSelect(pool, limit, idx.tokenIdf);

  // Normalise relevance into a 0-100 percentage relative to the top pick in
  // this batch, so the UI can render a comparable progress bar / "% match".
  const topScore = selected[0]?.relevance ?? 0;
  return selected.map((s) => ({
    slug: s.entry.slug,
    title: s.entry.title,
    excerpt: s.entry.excerpt,
    category: s.entry.category,
    subcategory: s.entry.subcategory,
    publishDate: s.entry.publishDate,
    image: s.entry.image,
    score: s.relevance,
    relevancePercent:
      topScore > 0 ? Math.round((s.relevance / topScore) * 100) : 0,
    sharedTags: s.sharedTags,
    reason: s.reason,
    reasonDetail: s.reasonDetail,
    similarity: s.similarity,
  }));
}

export function getSeriesContext(
  currentSlug: string,
): SeriesContext | null {
  const idx = getIndex();
  const current = idx.bySlug.get(currentSlug);
  if (!current?.collection) return null;
  const sibs = idx.byCollection.get(current.collection.toLowerCase()) ?? [];
  if (sibs.length < 2) return null;
  const i = sibs.findIndex((s) => s.slug === currentSlug);
  if (i < 0) return null;
  const total = sibs.length;
  const make = (e: IndexEntry, position: number): SeriesSibling => ({
    slug: e.slug,
    title: e.title,
    publishDate: e.publishDate,
    image: e.image,
    position,
    total,
  });
  return {
    collection: current.collection,
    prev: i > 0 ? make(sibs[i - 1], i) : undefined,
    next: i < sibs.length - 1 ? make(sibs[i + 1], i + 2) : undefined,
    current: { position: i + 1, total },
  };
}

export function getPopularPosts(limit = 6): RelatedPost[] {
  const idx = getIndex();
  return [...idx.entries]
    .sort((a, b) => {
      const da = Date.parse(a.publishDate) || 0;
      const db = Date.parse(b.publishDate) || 0;
      return db - da;
    })
    .slice(0, limit)
    .map((e) => ({
      slug: e.slug,
      title: e.title,
      excerpt: e.excerpt,
      category: e.category,
      subcategory: e.subcategory,
      publishDate: e.publishDate,
      image: e.image,
      score: 0,
      relevancePercent: 0,
      sharedTags: [],
      reason: "category" as const,
      reasonDetail: undefined,
      similarity: 0,
    }));
}
