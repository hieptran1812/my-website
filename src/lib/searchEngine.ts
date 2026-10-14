/**
 * Full-text blog search engine (BM25F over the whole corpus).
 *
 * Why not the substring scorer in `searchScore.ts`: that one is fine for
 * filtering a few hundred cards client-side, but over ~3.5k full posts it
 * matches "rag" inside "storage", treats "the" like "kubernetes", can't forgive
 * a typo and has to re-scan ~170 MB of markdown on every query.
 *
 * This engine tokenizes every post once (title, tags/subcategory, excerpt,
 * headings, body) into an inverted index, then answers a query with:
 *
 *   - BM25F ranking: rare words count more than common ones, a word in the
 *     title or a heading counts more than one buried in the body, and a long
 *     post can't win just by being long.
 *   - Prefix matching, so results appear while the last word is half-typed
 *     ("transf" → transformer) and "train" also finds "training".
 *   - Typo tolerance: an unknown word is corrected to the closest common word
 *     in the corpus ("kubernets" → kubernetes), and the correction is reported
 *     back so the UI can say what it searched for.
 *   - A small synonym/abbreviation table (k8s ↔ kubernetes, llm, rag, moe…).
 *   - Coverage + title boosts: posts matching every query word, and posts whose
 *     title contains the query, rise to the top.
 *   - Accent folding, so Vietnamese queries typed without tone marks still match.
 *
 * The index is built at deploy time by `scripts/buildSearchIndex.ts` into
 * `src/lib/generated/searchIndex.{json,bin}` (~3 bytes per term/post pair) and
 * loaded once per server instance. In dev, or if the files are missing, it is
 * built in-process from the live corpus instead.
 */

import fs from "fs";
import path from "path";
import type { BlogIndexEntry } from "./blogIndex";

// ---------------------------------------------------------------------------
// Text processing
// ---------------------------------------------------------------------------

/** Lowercase + strip diacritics (incl. Vietnamese đ). */
export function fold(s: string): string {
  return s
    .toLowerCase()
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .replace(/đ/g, "d");
}

const STOPWORDS = new Set(
  (
    "a an the and or but of to in on at by for with from as is are was were be been being " +
    "it its this that these those into over under about than then so if not no do does did " +
    "how what why when where which who whom your you we our i my me can could should would will " +
    "vs via using use used per all any some more most such each other also just only very " +
    // Vietnamese function words, accent-folded.
    "va cua la cac nhung cho voi trong mot nay khi thi duoc khong nhu tu den ve ra vao " +
    "nao gi sao bang theo tai hay cung da se dang"
  ).split(" "),
);

/** Very light English stemmer: plural folding only, applied to docs and queries alike. */
export function stem(w: string): string {
  if (w.length <= 3 || /\d/.test(w)) return w;
  if (w.endsWith("ies") && w.length > 4) return w.slice(0, -3) + "y";
  if (w.endsWith("sses")) return w.slice(0, -2);
  if (/(?:ss|us|is)$/.test(w)) return w;
  if (/(?:ch|sh|x|z)es$/.test(w)) return w.slice(0, -2);
  if (w.endsWith("s")) return w.slice(0, -1);
  return w;
}

const WORD_RE = /[a-z0-9]+/g;

/** Folded raw words (no stemming / stopword removal). */
function words(text: string): string[] {
  return fold(text).match(WORD_RE) ?? [];
}

/** Index terms for a document field. */
export function tokenize(text: string): string[] {
  const out: string[] = [];
  for (const w of words(text)) {
    if (w.length < 2 && !/\d/.test(w)) continue;
    if (STOPWORDS.has(w)) continue;
    out.push(stem(w));
  }
  return out;
}

/** Strip markdown noise: code blocks, math, images, html, link targets. */
export function cleanMarkdown(md: string): string {
  return md
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/\$\$[\s\S]*?\$\$/g, " ")
    .replace(/<svg[\s\S]*?<\/svg>/gi, " ")
    .replace(/!\[[^\]]*\]\([^)]*\)/g, " ")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/<[^>]+>/g, " ")
    .replace(/^#{1,6}\s+/gm, "")
    .replace(/[*_`>~|]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function extractHeadings(md: string): string {
  const out: string[] = [];
  for (const m of md.replace(/```[\s\S]*?```/g, " ").matchAll(/^#{1,6}\s+(.+)$/gm)) {
    out.push(m[1]);
  }
  return out.join(" ");
}

// ---------------------------------------------------------------------------
// Index structure
// ---------------------------------------------------------------------------

export interface SearchDoc {
  slug: string;
  title: string;
  category: string;
  subcategory: string;
  publishDate: string;
  excerpt: string;
  /** Path relative to content/blog, used to read the body for snippets. */
  file: string;
  paywalled: boolean;
}

export interface SearchIndex {
  docs: SearchDoc[];
  /** Sorted term list; term id = position. */
  vocab: string[];
  /** offsets[t]..offsets[t+1] index into docIds/weights for term t. */
  offsets: Uint32Array;
  docIds: Uint16Array;
  /** Low 7 bits: quantized BM25F tf weight. High bit: term occurs in the title. */
  weights: Uint8Array;
  /** Folded titles, for phrase matching. */
  foldedTitles: string[];
}

const FIELDS = [
  { name: "title", weight: 6, b: 0.3 },
  { name: "meta", weight: 4, b: 0.3 }, // tags + subcategory + category
  { name: "headings", weight: 2.5, b: 0.5 },
  { name: "excerpt", weight: 2, b: 0.5 },
  { name: "body", weight: 1, b: 0.75 },
] as const;
const K1 = 1.2;
const W_MAX = K1 + 1;
const TITLE_BIT = 128;

function humanize(s: string): string {
  return s.replace(/[-_/]+/g, " ");
}

/** Build the inverted index from corpus entries. */
export function buildIndex(
  corpus: BlogIndexEntry[],
  isPaywalled: (filePath: string) => boolean,
  blogDir: string,
): SearchIndex {
  const docs: SearchDoc[] = [];
  const foldedTitles: string[] = [];
  // Per doc: per field term-frequency maps.
  const fieldTfs: Map<string, number>[][] = [];
  const fieldLens: number[][] = [];

  for (const entry of corpus) {
    const fieldTexts = [
      entry.title,
      `${entry.tags.join(" ")} ${humanize(entry.subcategory)} ${humanize(entry.category)}`,
      extractHeadings(entry.content),
      entry.excerpt,
      cleanMarkdown(entry.content),
    ];
    const tfs: Map<string, number>[] = [];
    const lens: number[] = [];
    for (const text of fieldTexts) {
      const m = new Map<string, number>();
      const toks = tokenize(text || "");
      for (const t of toks) m.set(t, (m.get(t) ?? 0) + 1);
      tfs.push(m);
      lens.push(toks.length);
    }
    fieldTfs.push(tfs);
    fieldLens.push(lens);
    docs.push({
      slug: entry.slug,
      title: entry.title,
      category: entry.category,
      subcategory: entry.subcategory,
      publishDate: entry.publishDate,
      excerpt: entry.excerpt,
      file: path.relative(blogDir, entry.filePath).split(path.sep).join("/"),
      paywalled: isPaywalled(entry.filePath),
    });
    foldedTitles.push(words(entry.title).join(" "));
  }

  const avgLen = FIELDS.map(
    (_, f) =>
      fieldLens.reduce((s, l) => s + l[f], 0) / Math.max(1, fieldLens.length) || 1,
  );

  // term -> list of [docId, weightByte]
  const postings = new Map<string, number[]>();
  for (let d = 0; d < docs.length; d++) {
    const terms = new Set<string>();
    for (const m of fieldTfs[d]) for (const t of m.keys()) terms.add(t);
    for (const t of terms) {
      let tfd = 0;
      for (let f = 0; f < FIELDS.length; f++) {
        const tf = fieldTfs[d][f].get(t);
        if (!tf) continue;
        const { weight, b } = FIELDS[f];
        tfd += (weight * tf) / (1 - b + (b * fieldLens[d][f]) / avgLen[f]);
      }
      const w = (tfd / (K1 + tfd)) * W_MAX;
      let q = Math.max(1, Math.min(127, Math.round((w / W_MAX) * 127)));
      if (fieldTfs[d][0].has(t)) q |= TITLE_BIT;
      let list = postings.get(t);
      if (!list) postings.set(t, (list = []));
      list.push(d, q);
    }
  }

  const vocab = [...postings.keys()].sort();
  const offsets = new Uint32Array(vocab.length + 1);
  let total = 0;
  for (let i = 0; i < vocab.length; i++) {
    offsets[i] = total;
    total += postings.get(vocab[i])!.length / 2;
  }
  offsets[vocab.length] = total;
  const docIds = new Uint16Array(total);
  const weights = new Uint8Array(total);
  for (let i = 0; i < vocab.length; i++) {
    const list = postings.get(vocab[i])!;
    let o = offsets[i];
    for (let j = 0; j < list.length; j += 2, o++) {
      docIds[o] = list[j];
      weights[o] = list[j + 1];
    }
  }
  return { docs, vocab, offsets, docIds, weights, foldedTitles };
}

// ---------------------------------------------------------------------------
// Serialization: JSON (docs + vocab) + binary (offsets, docIds, weights)
// ---------------------------------------------------------------------------

export const GENERATED_DIR = path.join(process.cwd(), "src", "lib", "generated");
export const INDEX_JSON = path.join(GENERATED_DIR, "searchIndex.json");
export const INDEX_BIN = path.join(GENERATED_DIR, "searchIndex.bin");

export function writeIndex(index: SearchIndex): { jsonBytes: number; binBytes: number } {
  const json = JSON.stringify({
    docs: index.docs,
    vocab: index.vocab,
    foldedTitles: index.foldedTitles,
  });
  const n = index.vocab.length;
  const total = index.docIds.length;
  // Layout: offsets (u32 × n+1) | docIds (u16 × total) | weights (u8 × total)
  const buf = Buffer.alloc(4 * (n + 1) + 2 * total + total);
  Buffer.from(index.offsets.buffer, index.offsets.byteOffset, index.offsets.byteLength).copy(buf, 0);
  Buffer.from(index.docIds.buffer, index.docIds.byteOffset, index.docIds.byteLength).copy(buf, 4 * (n + 1));
  Buffer.from(index.weights.buffer, index.weights.byteOffset, index.weights.byteLength).copy(
    buf,
    4 * (n + 1) + 2 * total,
  );
  fs.mkdirSync(GENERATED_DIR, { recursive: true });
  fs.writeFileSync(INDEX_JSON, json);
  fs.writeFileSync(INDEX_BIN, buf);
  return { jsonBytes: json.length, binBytes: buf.length };
}

export function readIndex(): SearchIndex | null {
  try {
    const meta = JSON.parse(fs.readFileSync(INDEX_JSON, "utf8")) as {
      docs: SearchDoc[];
      vocab: string[];
      foldedTitles: string[];
    };
    if (!meta.docs?.length) return null;
    const raw = fs.readFileSync(INDEX_BIN);
    // Copy into a fresh, aligned ArrayBuffer for the typed-array views.
    const ab = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
    const n = meta.vocab.length;
    const offsets = new Uint32Array(ab, 0, n + 1);
    const total = offsets[n];
    const docIds = new Uint16Array(ab, 4 * (n + 1), total);
    const weights = new Uint8Array(ab, 4 * (n + 1) + 2 * total, total);
    return { ...meta, offsets, docIds, weights };
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Query
// ---------------------------------------------------------------------------

/** Abbreviations / synonyms, both directions. Folded + stemmed at lookup time. */
const SYNONYM_GROUPS: string[][] = [
  ["k8s", "kubernetes"],
  ["llm", "llms"],
  ["rag", "retrieval"],
  ["moe", "mixture"],
  ["rl", "reinforcement"],
  ["rlhf", "reinforcement"],
  ["kv", "cache"],
  ["db", "database"],
  ["postgres", "postgresql"],
  ["js", "javascript"],
  ["ts", "typescript"],
  ["py", "python"],
  ["ml", "machine"],
  ["nn", "neural"],
  ["gpu", "cuda"],
  ["btc", "bitcoin"],
  ["eth", "ethereum"],
  ["fx", "forex"],
  ["vn", "vietnam"],
  ["ai", "artificial"],
  ["api", "apis"],
  ["sre", "reliability"],
  ["ci", "cicd"],
  ["cd", "cicd"],
  ["tts", "speech"],
  ["asr", "speech"],
  ["vlm", "vision"],
  ["mla", "latent"],
  ["dpo", "preference"],
  ["sft", "finetuning"],
  ["finetune", "finetuning"],
  // Common Vietnamese query words -> the English the posts are written in.
  ["vang", "gold"],
  ["gia", "price"],
];
const SYNONYMS = new Map<string, string[]>();
for (const group of SYNONYM_GROUPS) {
  const g = group.map((w) => stem(fold(w)));
  for (const w of g) {
    const others = g.filter((o) => o !== w);
    SYNONYMS.set(w, [...(SYNONYMS.get(w) ?? []), ...others]);
  }
}

function findTerm(vocab: string[], term: string): number {
  let lo = 0;
  let hi = vocab.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const v = vocab[mid];
    if (v === term) return mid;
    if (v < term) lo = mid + 1;
    else hi = mid - 1;
  }
  return -lo - 1;
}

function df(index: SearchIndex, t: number): number {
  return index.offsets[t + 1] - index.offsets[t];
}

/** Optimal-string-alignment distance (transposition = 1 edit), capped at max+1. */
function editDistance(a: string, b: string, max: number): number {
  if (Math.abs(a.length - b.length) > max) return max + 1;
  let prev2: number[] = [];
  let prev = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const cur = [i];
    let rowMin = i;
    for (let j = 1; j <= b.length; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      let v = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost);
      if (i > 1 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1]) {
        v = Math.min(v, prev2[j - 2] + 1);
      }
      cur.push(v);
      if (v < rowMin) rowMin = v;
    }
    if (rowMin > max) return max + 1;
    prev2 = prev;
    prev = cur;
  }
  return prev[b.length];
}

interface Expansion {
  term: number;
  weight: number;
}

interface ParsedToken {
  /** The word as typed (folded). */
  word: string;
  expansions: Expansion[];
  /** Set when the word was unknown and replaced by a close corpus word. */
  correction?: string;
}

const MAX_PREFIX = 40;
const SYNONYM_WEIGHT = 0.8;

function expandToken(
  index: SearchIndex,
  word: string,
  isLast: boolean,
): ParsedToken {
  const { vocab } = index;
  const term = stem(word);
  const exp = new Map<number, number>();
  const add = (t: number, w: number) => {
    if (t >= 0 && (exp.get(t) ?? 0) < w) exp.set(t, w);
  };

  const exact = findTerm(vocab, term);
  add(exact, 1);
  if (term !== word) add(findTerm(vocab, word), 1);

  const synonyms = SYNONYMS.get(term) ?? [];
  for (const syn of synonyms) add(findTerm(vocab, syn), SYNONYM_WEIGHT);

  // Prefix completions: strong for the word being typed, weaker for earlier
  // words (they let "train" find "training" without drowning exact hits).
  if (word.length >= (isLast ? 2 : 4)) {
    const start = exact >= 0 ? exact : -exact - 1;
    const cands: number[] = [];
    for (let i = start; i < vocab.length && vocab[i].startsWith(word); i++) {
      if (i !== exact) cands.push(i);
    }
    cands.sort((a, b) => df(index, b) - df(index, a));
    // A word with a known synonym was most likely typed in full ("vang" =
    // gold, not the start of "vanguard"), so its completions count for less.
    const w = synonyms.length > 0 ? 0.5 : isLast ? 0.9 : 0.7;
    for (const c of cands.slice(0, MAX_PREFIX)) add(c, w);
  }

  let correction: string | undefined;
  const exactDf = exact >= 0 ? df(index, exact) : 0;
  if (exactDf <= 1 && exp.size <= (exact >= 0 ? 1 : 0) && word.length >= 4 && !/\d/.test(word)) {
    // Unknown (or near-unknown) word: look for close, more common words.
    const maxEd = word.length >= 8 ? 2 : 1;
    let best: { t: number; d: number; df: number }[] = [];
    for (let i = 0; i < vocab.length; i++) {
      const v = vocab[i];
      if (Math.abs(v.length - term.length) > maxEd) continue;
      if (v[0] !== term[0] && maxEd === 1) continue;
      const vdf = df(index, i);
      if (vdf < 3 || vdf <= exactDf * 3) continue;
      const d = editDistance(term, v, maxEd);
      if (d <= maxEd) best.push({ t: i, d, df: vdf });
    }
    best = best.sort((a, b) => a.d - b.d || b.df - a.df).slice(0, 3);
    for (const b of best) add(b.t, b.d === 1 ? 0.75 : 0.6);
    if (best.length > 0 && exactDf === 0) {
      // Vocab terms are stemmed; re-attach the suffix the stemmer took off the
      // typed word so "kubernets" reads back as "kubernetes", not "kubernete".
      const suffix = word.startsWith(term) ? word.slice(term.length) : "";
      correction = vocab[best[0].t] + suffix;
    }
  }

  return {
    word,
    expansions: [...exp].map(([t, weight]) => ({ term: t, weight })),
    correction,
  };
}

export interface SearchOptions {
  limit?: number;
  offset?: number;
  /** Restrict to one top-level category (e.g. "machine-learning"). */
  category?: string;
}

export interface SearchResult {
  hits: { doc: SearchDoc; score: number }[];
  total: number;
  /** Folded query terms that matched, for snippet highlighting. */
  matchedWords: string[];
  /** Query rewritten with typo corrections, when any were made. */
  correctedQuery?: string;
}

export function search(
  index: SearchIndex,
  rawQuery: string,
  opts: SearchOptions = {},
): SearchResult {
  const limit = opts.limit ?? 10;
  const offset = opts.offset ?? 0;
  const empty: SearchResult = { hits: [], total: 0, matchedWords: [] };

  const allWords = words(rawQuery);
  const endsWithSpace = /\s$/.test(rawQuery);
  let qWords = allWords.filter((w) => (w.length >= 2 || /\d/.test(w)) && !STOPWORDS.has(w));
  if (qWords.length === 0) qWords = allWords.filter((w) => w.length >= 2);
  if (qWords.length === 0) return empty;
  qWords = [...new Set(qWords)].slice(0, 8);

  const tokens = qWords.map((w, i) =>
    expandToken(index, w, i === qWords.length - 1 && !endsWithSpace),
  );
  const live = tokens.filter((t) => t.expansions.length > 0);
  if (live.length === 0) return empty;

  const N = index.docs.length;
  const nTok = tokens.length;
  // Per doc: best contribution per query token, and title-hit bitmask.
  const tokScore = new Float32Array(N * nTok);
  const titleMask = new Uint16Array(N);
  const touched: number[] = [];
  const seen = new Uint8Array(N);

  const idfOf = (n: number) => Math.log(1 + (N - n + 0.5) / (n + 0.5));
  tokens.forEach((tok, ti) => {
    // Completions and corrections may not out-score the word as typed (or its
    // synonyms): a rare
    // completion ("qwen3guard") would otherwise beat the exact word ("qwen")
    // just for being rare. Cap their idf at the typed word's idf.
    const exactDf = Math.max(
      0,
      ...tok.expansions
        .filter((e) => e.weight >= SYNONYM_WEIGHT)
        .map((e) => df(index, e.term)),
    );
    const baseDf =
      exactDf || Math.max(...tok.expansions.map((e) => df(index, e.term)));
    const idfCap = idfOf(baseDf);
    for (const { term, weight } of tok.expansions) {
      const lo = index.offsets[term];
      const hi = index.offsets[term + 1];
      const idf = Math.min(idfOf(hi - lo), idfCap);
      for (let o = lo; o < hi; o++) {
        const d = index.docIds[o];
        const wb = index.weights[o];
        // A word in the title is worth double: BM25F saturates per term, so
        // without this a long body that repeats the word catches the title up.
        const s =
          idf * ((wb & 127) / 127) * W_MAX * weight * (wb & TITLE_BIT ? 2 : 1);
        const k = d * nTok + ti;
        if (s > tokScore[k]) tokScore[k] = s;
        if (wb & TITLE_BIT && weight >= 0.7) titleMask[d] |= 1 << ti;
        if (!seen[d]) {
          seen[d] = 1;
          touched.push(d);
        }
      }
    }
  });

  const phrase = qWords.join(" ");
  const category = opts.category?.toLowerCase();
  const scored: { d: number; score: number }[] = [];
  for (const d of touched) {
    const doc = index.docs[d];
    if (category && doc.category.toLowerCase() !== category) continue;
    let sum = 0;
    let matched = 0;
    for (let ti = 0; ti < nTok; ti++) {
      const s = tokScore[d * nTok + ti];
      if (s > 0) {
        sum += s;
        matched++;
      }
    }
    const coverage = matched / nTok;
    // Multi-word queries: strongly prefer posts that match every word, and
    // drop posts that match under half of them.
    if (nTok >= 3 && coverage < 0.5) continue;
    let score = sum * Math.pow(coverage, 2);
    if (nTok >= 2 && bitCount(titleMask[d]) === nTok) score *= 1.3;
    const title = index.foldedTitles[d];
    if (nTok >= 2 && title.includes(phrase)) score *= 1.5;
    if (title === phrase) score *= 2;
    scored.push({ d, score });
  }

  scored.sort(
    (a, b) =>
      b.score - a.score ||
      (index.docs[b.d].publishDate || "").localeCompare(index.docs[a.d].publishDate || ""),
  );

  const matchedWords = new Set<string>();
  for (const tok of tokens) {
    for (const { term, weight } of tok.expansions) {
      if (weight >= 0.75) matchedWords.add(index.vocab[term]);
    }
    matchedWords.add(tok.word);
  }
  const corrected = tokens.some((t) => t.correction)
    ? tokens.map((t) => t.correction ?? t.word).join(" ")
    : undefined;

  return {
    hits: scored.slice(offset, offset + limit).map(({ d, score }) => ({
      doc: index.docs[d],
      score: Math.round(score * 1000) / 1000,
    })),
    total: scored.length,
    matchedWords: [...matchedWords],
    correctedQuery: corrected,
  };
}

function bitCount(x: number): number {
  let c = 0;
  while (x) {
    c += x & 1;
    x >>= 1;
  }
  return c;
}

// ---------------------------------------------------------------------------
// Snippets
// ---------------------------------------------------------------------------

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/** Per-UTF-16-unit accent fold that keeps string length, so indices line up. */
function foldSameLength(s: string): string {
  let out = "";
  for (let i = 0; i < s.length; i++) {
    const f = fold(s[i]);
    out += f.length === 1 ? f : s[i].toLowerCase().slice(0, 1) || " ";
  }
  return out;
}

const SNIPPET_LEN = 200;

/**
 * Pick the ~200-char window of `text` containing the most distinct query
 * words, and highlight every word that starts with a matched term.
 */
export function buildSnippet(text: string, matchedWords: string[]): string {
  const folded = foldSameLength(text);
  const terms = [...new Set(matchedWords)].filter((t) => t.length >= 2);
  const hits: { pos: number; len: number; term: number }[] = [];
  if (terms.length > 0) {
    const re = new RegExp(
      `(?<![a-z0-9])(?:${terms
        .map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
        .sort((a, b) => b.length - a.length)
        .join("|")})[a-z0-9]*`,
      "g",
    );
    for (const m of folded.matchAll(re)) {
      const term = terms.findIndex((t) => m[0].startsWith(t));
      hits.push({ pos: m.index!, len: m[0].length, term });
      if (hits.length > 2000) break;
    }
  }

  let start = 0;
  if (hits.length > 0) {
    let bestCount = -1;
    let j = 0;
    const counts = new Map<number, number>();
    for (let i = 0; i < hits.length; i++) {
      counts.set(hits[i].term, (counts.get(hits[i].term) ?? 0) + 1);
      while (hits[i].pos - hits[j].pos > SNIPPET_LEN - 40) {
        const c = counts.get(hits[j].term)! - 1;
        if (c === 0) counts.delete(hits[j].term);
        else counts.set(hits[j].term, c);
        j++;
      }
      const distinct = counts.size * 100 + (i - j);
      if (distinct > bestCount) {
        bestCount = distinct;
        start = Math.max(0, hits[j].pos - 40);
      }
    }
    // Snap to a word boundary.
    if (start > 0) {
      const sp = text.indexOf(" ", start);
      if (sp > 0 && sp - start < 20) start = sp + 1;
    }
  }
  const end = Math.min(text.length, start + SNIPPET_LEN);

  let html = start > 0 ? "…" : "";
  let cursor = start;
  for (const h of hits) {
    if (h.pos < start || h.pos + h.len > end) continue;
    html += escapeHtml(text.slice(cursor, h.pos));
    html += `<mark>${escapeHtml(text.slice(h.pos, h.pos + h.len))}</mark>`;
    cursor = h.pos + h.len;
  }
  html += escapeHtml(text.slice(cursor, end));
  if (end < text.length) html += "…";
  return html;
}
