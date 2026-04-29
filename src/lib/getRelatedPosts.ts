import { removeStopwords, eng, vie } from "stopword";
import {
  loadAllPosts,
  extractFirstBodyImage as _extractFirstBodyImage,
  resolvePostCover as _resolvePostCover,
} from "./blogIndex";

// Re-export so existing import paths (e.g. getArticle.ts) keep working.
export const extractFirstBodyImage = _extractFirstBodyImage;
export const resolvePostCover = _resolvePostCover;

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

export interface IndexEntry {
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

export type DominantSignal =
  | "series"
  | "reference"
  | "tags"
  | "similar"
  | "structural";

export interface AdjacencyEdge {
  /** Composite weight, clamped to [0, 1]. */
  weight: number;
  /** Which signal contributed the most to this weight (for hover tooltips). */
  dominant: DominantSignal;
  /** Human-readable evidence shown on hover. */
  evidence: string;
  /** True only when source links to target via markdown body. */
  reference?: boolean;
}

export interface CorpusIndex {
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
  /** Forward markdown reference graph: slug → set of slugs it links to. */
  outgoingRefs: Map<string, Set<string>>;
  /** Reverse: slug → set of slugs that link to it. */
  incomingRefs: Map<string, Set<string>>;
  /** Composite-weighted adjacency: slug → (slug → edge). Dense enough that
   *  every relevant pair appears once; weights are symmetric. Used as the
   *  transition matrix for Personalized PageRank. */
  adjacency: Map<string, Map<string, AdjacencyEdge>>;
}

let cachedIndex: CorpusIndex | null = null;
let cachedAt = 0;
const CACHE_TTL_MS = 5 * 60 * 1000;

const STOP_WORDS = new Set([...eng, ...vie]);
const TOKEN_RE = /[\p{L}\p{N}]+/gu;

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

async function buildIndex(): Promise<CorpusIndex> {
  const corpus = await loadAllPosts();
  const entries: IndexEntry[] = [];
  const tagDf = new Map<string, number>();
  const tokenDf = new Map<string, number>();

  for (const e of corpus) {
    const tags = e.tags.map((t) => t.toLowerCase());
    const tokens = tokenize(`${e.title} ${e.excerpt}`);
    const tokenTf = termFreq(tokens);

    entries.push({
      slug: e.slug,
      title: e.title,
      excerpt: e.excerpt,
      category: e.category,
      subcategory: e.subcategory,
      tags,
      publishDate: e.publishDate,
      image: e.image,
      collection: e.collection,
      tokenTf,
      tfidfNorm: 0, // filled below
    });

    for (const t of new Set(tags)) tagDf.set(t, (tagDf.get(t) ?? 0) + 1);
    for (const tok of new Set(tokenTf.keys()))
      tokenDf.set(tok, (tokenDf.get(tok) ?? 0) + 1);
  }

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

  // Build the markdown-link reference graph. We parse each post's body for
  // [text](slug) — slugs may include /blog/ prefix, trailing slashes, or be
  // relative — and keep only links whose target resolves to another post.
  const outgoingRefs = new Map<string, Set<string>>();
  const incomingRefs = new Map<string, Set<string>>();
  const linkRe = /\[([^\]]*)\]\(([^)\s]+)\)/g;
  for (const post of corpus) {
    const out = new Set<string>();
    let m: RegExpExecArray | null;
    while ((m = linkRe.exec(post.content)) !== null) {
      const target = m[2]
        .replace(/^\/+/, "")
        .replace(/^blog\//, "")
        .replace(/\/+$/, "")
        .replace(/^#.+$/, "");
      if (!target || target === post.slug) continue;
      if (bySlug.has(target)) out.add(target);
    }
    outgoingRefs.set(post.slug, out);
    for (const t of out) {
      const arr = incomingRefs.get(t) ?? new Set<string>();
      arr.add(post.slug);
      incomingRefs.set(t, arr);
    }
  }

  // ─── Composite-weighted adjacency (one pass) ───
  const adjacency = new Map<string, Map<string, AdjacencyEdge>>();
  for (const e of entries) adjacency.set(e.slug, new Map());

  const W_TAG = 1.0;
  const W_COS = 0.7;
  const W_REF = 1.5;
  const W_COL = 2.0;
  const W_SUB = 0.4;
  const W_CAT = 0.15;
  const W_TIME = 0.3;
  const TIME_HALF_LIFE_MS = 60 * 24 * 3600 * 1000; // 60 days
  const tmpIndex: CorpusIndex = {
    entries,
    bySlug,
    tagIdf,
    tokenIdf,
    N,
    byCollection,
    outgoingRefs,
    incomingRefs,
    adjacency,
  };

  for (let i = 0; i < entries.length; i++) {
    const a = entries[i];
    for (let j = i + 1; j < entries.length; j++) {
      const b = entries[j];

      // IDF tag overlap
      const aTagSet = new Set(a.tags);
      const sharedTags = b.tags.filter((t) => aTagSet.has(t));
      let tagWeight = 0;
      let rareTag: string | undefined;
      let rareIdf = 0;
      for (const t of sharedTags) {
        const idf = tagIdf.get(t) ?? 0;
        tagWeight += idf;
        if (idf > rareIdf) {
          rareIdf = idf;
          rareTag = t;
        }
      }
      const aMaxTagScore =
        a.tags.reduce((s, t) => s + (tagIdf.get(t) ?? 0), 0) || 1;
      const tagNorm = Math.min(1, tagWeight / aMaxTagScore);

      const cos = cosineSimilarity(a, b, tokenIdf);
      const sameSub = !!a.subcategory && a.subcategory === b.subcategory;
      const sameCat = !!a.category && a.category === b.category;
      const sameCollection =
        !!a.collection &&
        a.collection.toLowerCase() === (b.collection || "").toLowerCase();
      const aLinksB = outgoingRefs.get(a.slug)?.has(b.slug) ?? false;
      const bLinksA = outgoingRefs.get(b.slug)?.has(a.slug) ?? false;
      const hasRef = aLinksB || bLinksA;

      // Time proximity (Gaussian-ish, half-life 60 days)
      let timeBonus = 0;
      const ta = Date.parse(a.publishDate);
      const tb = Date.parse(b.publishDate);
      if (!Number.isNaN(ta) && !Number.isNaN(tb)) {
        const diffDays = Math.abs(ta - tb) / TIME_HALF_LIFE_MS;
        timeBonus = Math.exp(-diffDays); // 1 when same day, ~0.37 at 60d
      }

      const raw =
        W_TAG * tagNorm +
        W_COS * cos +
        W_REF * (hasRef ? 1 : 0) +
        W_COL * (sameCollection ? 1 : 0) +
        W_SUB * (sameSub ? 1 : 0) +
        W_CAT * (sameCat ? 1 : 0) +
        W_TIME * timeBonus;

      // No edge if too weak — reduces matrix density.
      if (raw < 0.12) continue;

      // Normalise: divide by sum of upper-bound weights (with all signals on,
      // tagNorm/cos/timeBonus capped at 1) — gives weight ≈ how saturated this
      // pair is across signals. Cap at 1 so we render nicely.
      const cap =
        W_TAG + W_COS + W_REF + W_COL + W_SUB + W_CAT + W_TIME;
      const weight = Math.min(1, raw / cap);

      // Pick dominant signal (largest contribution).
      const contribs: Array<[number, DominantSignal, string]> = [
        [W_COL * (sameCollection ? 1 : 0), "series", `Series: ${a.collection ?? ""}`],
        [W_REF * (hasRef ? 1 : 0), "reference", aLinksB ? "References this article" : "Cited by this article"],
        [W_TAG * tagNorm, "tags", rareTag ? `Shares '${rareTag}'` : `${sharedTags.length} shared tags`],
        [W_COS * cos, "similar", `${Math.round(cos * 100)}% similar`],
      ];
      contribs.sort((x, y) => y[0] - x[0]);
      const [, dominant, evidence] = contribs[0][0] > 0
        ? contribs[0]
        : [0, "structural" as DominantSignal, sameSub ? "Same subcategory" : "Same category"];

      const edgeAB: AdjacencyEdge = {
        weight,
        dominant,
        evidence,
        reference: aLinksB,
      };
      const edgeBA: AdjacencyEdge = {
        weight,
        dominant,
        evidence,
        reference: bLinksA,
      };
      adjacency.get(a.slug)!.set(b.slug, edgeAB);
      adjacency.get(b.slug)!.set(a.slug, edgeBA);
    }
  }
  void tmpIndex; // keeps W_* constants referenced for clarity

  return {
    entries,
    bySlug,
    tagIdf,
    tokenIdf,
    N,
    byCollection,
    outgoingRefs,
    incomingRefs,
    adjacency,
  };
}

export async function getIndex(): Promise<CorpusIndex> {
  const now = Date.now();
  if (cachedIndex && now - cachedAt < CACHE_TTL_MS) return cachedIndex;
  cachedIndex = await buildIndex();
  cachedAt = now;
  return cachedIndex;
}

// ─────────────── Similarity primitives ───────────────

export function cosineSimilarity(
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

export interface CandidateScore {
  entry: IndexEntry;
  relevance: number;
  similarity: number;
  sharedTags: string[];
  rareSharedTag?: string;
  rareSharedTagWeight: number;
  reason: RelatedReason;
  reasonDetail?: string;
}

export function scoreCandidate(
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

export async function getRelatedPosts(
  currentSlug: string,
  // currentTags/currentCategory/currentSubcategory kept for back-compat callers
  // but values are sourced authoritatively from the corpus index.
  _currentTags: string[],
  _currentCategory: string,
  _currentSubcategory: string,
  limit = 6,
): Promise<RelatedPost[]> {
  const idx = await getIndex();
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

export async function getSeriesContext(
  currentSlug: string,
): Promise<SeriesContext | null> {
  const idx = await getIndex();
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

export async function getPopularPosts(limit = 6): Promise<RelatedPost[]> {
  const idx = await getIndex();
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

// ─────────────── Personalized PageRank ───────────────

/**
 * Power-iteration PPR over the composite-weighted adjacency. Starts random
 * walk at `seedSlug`; each step has probability α of restarting at the seed
 * and probability (1-α) of following an outgoing edge proportional to its
 * weight. Returns the stationary probability map.
 */
export function personalizedPageRank(
  seedSlug: string,
  idx: CorpusIndex,
  alpha = 0.85,
  iterations = 30,
): Map<string, number> {
  const slugs = Array.from(idx.bySlug.keys());
  const n = slugs.length;
  if (n === 0 || !idx.bySlug.has(seedSlug)) return new Map();

  const slugToIdx = new Map<string, number>();
  slugs.forEach((s, i) => slugToIdx.set(s, i));
  const seedIdx = slugToIdx.get(seedSlug)!;

  // Pre-compute row-normalised outgoing weights for each node.
  const rows: Array<Array<[number, number]>> = new Array(n);
  for (let i = 0; i < n; i++) {
    const slug = slugs[i];
    const adj = idx.adjacency.get(slug);
    if (!adj || adj.size === 0) {
      rows[i] = [];
      continue;
    }
    let sum = 0;
    for (const e of adj.values()) sum += e.weight;
    const row: Array<[number, number]> = [];
    if (sum > 0) {
      for (const [t, e] of adj) {
        row.push([slugToIdx.get(t)!, e.weight / sum]);
      }
    }
    rows[i] = row;
  }

  // Initial: all probability on seed.
  let p = new Float64Array(n);
  p[seedIdx] = 1;

  for (let iter = 0; iter < iterations; iter++) {
    const next = new Float64Array(n);
    next[seedIdx] += alpha; // restart contribution
    for (let i = 0; i < n; i++) {
      const pi = p[i];
      if (pi === 0) continue;
      const row = rows[i];
      const flow = (1 - alpha) * pi;
      if (row.length === 0) {
        // Dangling node: send mass back to seed.
        next[seedIdx] += flow;
        continue;
      }
      for (const [j, w] of row) next[j] += flow * w;
    }
    p = next;
  }

  const out = new Map<string, number>();
  for (let i = 0; i < n; i++) if (p[i] > 0) out.set(slugs[i], p[i]);
  return out;
}

/**
 * Greedy MMR pick from a ranked list. Penalises candidates that are too
 * similar to already-picked nodes (uses cosine on TF-IDF as similarity proxy).
 */
export function selectWithMmr(
  ranked: Array<{ slug: string; score: number }>,
  k: number,
  idx: CorpusIndex,
  lambda = 0.75,
): string[] {
  const picked: string[] = [];
  const remaining = [...ranked];
  while (picked.length < k && remaining.length > 0) {
    let bestIdx = 0;
    let bestVal = -Infinity;
    for (let i = 0; i < remaining.length; i++) {
      const c = remaining[i];
      const cEntry = idx.bySlug.get(c.slug);
      if (!cEntry) continue;
      let maxSim = 0;
      for (const ps of picked) {
        const pEntry = idx.bySlug.get(ps);
        if (!pEntry) continue;
        const sim = cosineSimilarity(cEntry, pEntry, idx.tokenIdf);
        if (sim > maxSim) maxSim = sim;
      }
      const mmr = lambda * c.score - (1 - lambda) * maxSim;
      if (mmr > bestVal) {
        bestVal = mmr;
        bestIdx = i;
      }
    }
    picked.push(remaining.splice(bestIdx, 1)[0].slug);
  }
  return picked;
}
