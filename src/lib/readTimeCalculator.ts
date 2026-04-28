/**
 * Read-time calculator.
 *
 * Cộng dồn chi phí theo từng phần tử (prose / code / math / image / table /
 * callout). WPM gốc thay đổi theo category. Hệ số khó là một continuous
 * multiplier ∈ [0.85, 1.20] dựa trên mật độ citation, độ dài câu, vocab
 * uniqueness, và mật độ symbol LaTeX nâng cao.
 */

import { createHash } from "crypto";

// ─────────────────────────── Public types ───────────────────────────

export interface ContentAnalysis {
  wordCount: number;
  codeBlocks: number;
  codeLines: number;
  mathEquations: number;
  mathBlockLines: number;
  images: number;
  imageStudyCount: number;
  tables: number;
  tableRows: number;
  callouts: number;
  headings: number;
  lists: number;
  links: number;
  citations: number;
  avgSentenceLength: number;
  typeTokenRatio: number;
  advancedMathSymbols: number;
  difficultyScore: number;
  difficultyMultiplier: number;
  complexity: "beginner" | "intermediate" | "advanced";
}

export interface ReadTimeBreakdown {
  baseReadingTime: number;
  codeBlockTime: number;
  mathTime: number;
  imageTime: number;
  tableTime: number;
  calloutTime: number;
  complexityAdjustment: number;
  totalTime: number;
}

export interface ReadTimeResult {
  readTime: string;
  readTimeMinutes: number;
  analysis: ContentAnalysis;
  breakdown: ReadTimeBreakdown;
}

export interface ReadTimeConfig {
  /** Override base words-per-minute (otherwise derived from category) */
  wordsPerMinute?: number;
  /** Category slug for WPM calibration (e.g. "paper-reading") */
  category?: string;
  /** Seconds per non-blank LOC (default 5) */
  codeSecPerLine?: number;
  /** Seconds per inline math symbol (default 3 with cap) */
  inlineMathSec?: number;
  /** Seconds per block math equation header (default 10) */
  blockMathBaseSec?: number;
  /** Seconds per line inside block math (default 5) */
  blockMathPerLineSec?: number;
  /** Seconds per regular image (default 12) */
  imageSec?: number;
  /** Seconds per study-image (figure/diagram/architecture) (default 25) */
  studyImageSec?: number;
  /** Seconds per table base (default 5) */
  tableBaseSec?: number;
  /** Seconds per data row (default 2) */
  tableRowSec?: number;
  /** Seconds per callout/blockquote (default 2) */
  calloutSec?: number;
  /** Minimum read time in minutes (default 1) */
  minReadTime?: number;
  /** Maximum read time in minutes (default 90) */
  maxReadTime?: number;
}

// ─────────────────────────── Tunables ───────────────────────────

const DEFAULTS: Required<Omit<ReadTimeConfig, "wordsPerMinute" | "category">> = {
  codeSecPerLine: 5,
  inlineMathSec: 3,
  blockMathBaseSec: 10,
  blockMathPerLineSec: 5,
  imageSec: 12,
  studyImageSec: 25,
  tableBaseSec: 5,
  tableRowSec: 2,
  calloutSec: 2,
  minReadTime: 1,
  maxReadTime: 90,
};

const WPM_BY_CATEGORY: Record<string, number> = {
  "paper-reading": 180,
  "machine-learning": 200,
  "software-development": 200,
  notes: 230,
  trading: 230,
  crypto: 220,
  demo: 220,
};
const DEFAULT_WPM = 210;

// Code language → cognitive load multiplier (cost = codeSecPerLine × multiplier)
const CODE_LANG_MULTIPLIER: Record<string, number> = {
  python: 1.0,
  py: 1.0,
  javascript: 1.0,
  js: 1.0,
  typescript: 1.05,
  ts: 1.05,
  tsx: 1.05,
  jsx: 1.0,
  bash: 0.8,
  sh: 0.8,
  shell: 0.8,
  json: 0.6,
  yaml: 0.6,
  yml: 0.6,
  toml: 0.6,
  markdown: 0.5,
  md: 0.5,
  text: 0.5,
  html: 0.7,
  css: 0.7,
  sql: 1.2,
  regex: 1.3,
  rust: 1.4,
  rs: 1.4,
  cpp: 1.4,
  "c++": 1.4,
  c: 1.3,
  haskell: 1.5,
  hs: 1.5,
  scala: 1.4,
  go: 1.0,
  java: 1.1,
  kotlin: 1.1,
};
const DEFAULT_CODE_MULTIPLIER = 1.0;

const ADVANCED_MATH_SYMBOLS = [
  "\\sum",
  "\\int",
  "\\oint",
  "\\prod",
  "\\nabla",
  "\\partial",
  "\\infty",
  "\\lim",
  "\\sup",
  "\\inf",
  "\\forall",
  "\\exists",
  "\\mathbb",
  "\\mathcal",
  "\\mathrm",
  "\\operatorname",
  "\\frac",
  "\\sqrt",
  "\\binom",
  "\\det",
  "\\arg",
  "\\nabla",
  "\\otimes",
  "\\oplus",
  "\\cdot",
];

const STUDY_IMAGE_HINTS = [
  "figure",
  "fig.",
  "diagram",
  "architecture",
  "schematic",
  "flowchart",
  "graph",
  "plot",
  "chart",
];

// ─────────────────────────── Caches ───────────────────────────

const analysisCache = new Map<string, ContentAnalysis>();
const resultCache = new Map<string, ReadTimeResult>();
const MAX_CACHE_ENTRIES = 1024;

function hashContent(content: string): string {
  return createHash("sha1").update(content).digest("hex").slice(0, 16);
}

function cacheSet<K, V>(map: Map<K, V>, key: K, value: V) {
  if (map.size >= MAX_CACHE_ENTRIES) {
    // Drop oldest entry (Map preserves insertion order).
    const first = map.keys().next().value;
    if (first !== undefined) map.delete(first);
  }
  map.set(key, value);
}

// ─────────────────────────── Tokenizers ───────────────────────────

const FRONT_MATTER_RE = /^---[\s\S]*?---\n/;
const FENCED_CODE_RE = /^([ \t]*)```([^\n`]*)\n([\s\S]*?)\n\1```/gm;
const INLINE_CODE_RE = /`[^`\n]+`/g;
const IMAGE_RE = /!\[([^\]]*)\]\(([^)]*)\)/g;
const LINK_RE = /\[([^\]]+)\]\(([^)]+)\)/g;
const HEADING_RE = /^#{1,6}\s+/gm;
const LIST_RE = /^\s*([-*+]|\d+\.)\s+/gm;
const BLOCKQUOTE_RE = /^>\s?/gm;
const BLOCK_MATH_RE = /\$\$([\s\S]+?)\$\$/g;
const INLINE_MATH_RE = /(?<!\$)\$(?!\$)([^\n$]+?)\$(?!\$)/g;
// Pandoc-style citations: [@key], [@key1; @key2]
const CITATION_RE = /\[(?:@[\w:.-]+(?:\s*;\s*@[\w:.-]+)*)\]/g;
// Numeric/footnote refs: [1], [12], [^1]
const NUMERIC_REF_RE = /\[\^?\d{1,3}\]/g;
// Pipe-table separator row, e.g. |---|---|
const TABLE_SEP_RE = /^\s*\|?(?:\s*:?-{2,}:?\s*\|)+\s*:?-{2,}:?\s*\|?\s*$/gm;

interface CodeChunk {
  lang: string;
  lines: number;
}

function extractCodeBlocks(content: string): {
  cleaned: string;
  blocks: CodeChunk[];
} {
  const blocks: CodeChunk[] = [];
  const cleaned = content.replace(
    FENCED_CODE_RE,
    (_match, _indent, langRaw: string, body: string) => {
      const lang = (langRaw || "").trim().toLowerCase();
      const lines = body
        .split("\n")
        .filter((l) => l.trim().length > 0).length;
      blocks.push({ lang, lines });
      return "\n";
    },
  );
  return { cleaned, blocks };
}

interface MathStats {
  inline: number;
  inlineSymbols: number;
  block: number;
  blockLines: number;
  advanced: number;
}

function countAdvancedMathSymbols(text: string): number {
  let n = 0;
  for (const sym of ADVANCED_MATH_SYMBOLS) {
    const re = new RegExp(sym.replace(/\\/g, "\\\\"), "g");
    const m = text.match(re);
    if (m) n += m.length;
  }
  return n;
}

function extractMath(content: string): {
  cleaned: string;
  stats: MathStats;
} {
  const stats: MathStats = {
    inline: 0,
    inlineSymbols: 0,
    block: 0,
    blockLines: 0,
    advanced: 0,
  };

  let cleaned = content.replace(BLOCK_MATH_RE, (_m, body: string) => {
    stats.block += 1;
    stats.blockLines += Math.max(1, body.trim().split("\n").length);
    stats.advanced += countAdvancedMathSymbols(body);
    return " ";
  });

  cleaned = cleaned.replace(INLINE_MATH_RE, (_m, body: string) => {
    stats.inline += 1;
    // Each backslash command or single letter symbol counts as one "symbol".
    const symbols = body.match(/\\[a-zA-Z]+|[a-zA-Z]/g);
    stats.inlineSymbols += symbols ? Math.min(8, symbols.length) : 1;
    stats.advanced += countAdvancedMathSymbols(body);
    return " ";
  });

  return { cleaned, stats };
}

function countImages(content: string): {
  cleaned: string;
  total: number;
  study: number;
} {
  let total = 0;
  let study = 0;
  const cleaned = content.replace(IMAGE_RE, (_m, alt: string) => {
    total += 1;
    const a = (alt || "").toLowerCase();
    if (STUDY_IMAGE_HINTS.some((h) => a.includes(h))) study += 1;
    return " ";
  });
  return { cleaned, total, study };
}

function countTables(content: string): { tables: number; rows: number } {
  const lines = content.split("\n");
  let tables = 0;
  let rows = 0;
  let inTable = false;
  let currentRows = 0;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const isPipeRow = /^\s*\|.*\|\s*$/.test(line);
    const isSep = TABLE_SEP_RE.test(line);
    TABLE_SEP_RE.lastIndex = 0; // reset global regex state

    if (isPipeRow) {
      if (!inTable) {
        inTable = true;
        currentRows = 0;
        tables += 1;
      }
      if (!isSep) currentRows += 1;
    } else if (inTable) {
      // Header row was counted but isn't a "data row"; subtract one.
      rows += Math.max(0, currentRows - 1);
      inTable = false;
      currentRows = 0;
    }
  }
  if (inTable) rows += Math.max(0, currentRows - 1);
  return { tables, rows };
}

function countCallouts(content: string): number {
  // Treat each contiguous blockquote run as one callout.
  const lines = content.split("\n");
  let n = 0;
  let inQuote = false;
  for (const line of lines) {
    if (/^>\s?/.test(line)) {
      if (!inQuote) {
        n += 1;
        inQuote = true;
      }
    } else {
      inQuote = false;
    }
  }
  return n;
}

function countCitations(content: string): number {
  const pandoc = (content.match(CITATION_RE) || []).length;
  const numeric = (content.match(NUMERIC_REF_RE) || []).length;
  return pandoc + numeric;
}

// ─────────────────────────── Difficulty signals ───────────────────────────

function computeDifficulty(args: {
  wordCount: number;
  proseText: string;
  citations: number;
  advancedMathSymbols: number;
}): { score: number; multiplier: number; avgSentenceLength: number; ttr: number } {
  const { wordCount, proseText, citations, advancedMathSymbols } = args;

  // Average sentence length
  const sentences = proseText
    .split(/[.!?]+\s+/)
    .filter((s) => s.trim().length > 0);
  const avgSentenceLength =
    sentences.length > 0 ? wordCount / sentences.length : 0;

  // Type-token ratio (vocab uniqueness)
  let ttr = 0;
  if (wordCount > 0) {
    const tokens = proseText
      .toLowerCase()
      .match(/[\p{L}\p{N}']+/gu) || [];
    const types = new Set(tokens);
    ttr = tokens.length > 0 ? types.size / tokens.length : 0;
  }

  let score = 0;
  // Citation density per 1000 words.
  const citationDensity = wordCount > 0 ? (citations / wordCount) * 1000 : 0;
  if (citationDensity >= 5) score += 0.10;
  else if (citationDensity >= 2) score += 0.05;

  if (avgSentenceLength > 25) score += 0.05;
  else if (avgSentenceLength > 35) score += 0.10;

  if (ttr > 0.5 && wordCount > 200) score += 0.05;

  const advDensity = wordCount > 0 ? (advancedMathSymbols / wordCount) * 1000 : 0;
  if (advDensity >= 10) score += 0.10;
  else if (advDensity >= 3) score += 0.05;

  const multiplier = Math.max(0.85, Math.min(1.20, 1 + score));
  return { score, multiplier, avgSentenceLength, ttr };
}

// ─────────────────────────── Public API ───────────────────────────

export function analyzeContent(content: string): ContentAnalysis {
  const cacheKey = hashContent(content);
  const cached = analysisCache.get(cacheKey);
  if (cached) return cached;

  const noFront = content.replace(FRONT_MATTER_RE, "");
  const { cleaned: noCode, blocks } = extractCodeBlocks(noFront);
  const { cleaned: noMath, stats: mathStats } = extractMath(noCode);
  const { cleaned: noImg, total: images, study: imageStudyCount } =
    countImages(noMath);
  const { tables, rows: tableRows } = countTables(noImg);
  const callouts = countCallouts(noImg);
  const citations = countCitations(noFront);

  // Final prose: strip lists, headings, blockquote markers, links → text.
  const prose = noImg
    .replace(LINK_RE, "$1")
    .replace(HEADING_RE, "")
    .replace(LIST_RE, "")
    .replace(BLOCKQUOTE_RE, "")
    .replace(INLINE_CODE_RE, " ")
    .replace(/\|[^\n]*\|/g, " ") // strip remaining table rows
    .replace(/\s+/g, " ")
    .trim();

  const wordCount = prose.length === 0 ? 0 : prose.split(/\s+/).length;

  const codeLines = blocks.reduce((s, b) => s + b.lines, 0);
  const codeBlocks = blocks.length;
  const headings = (noFront.match(HEADING_RE) || []).length;
  const lists = (noFront.match(LIST_RE) || []).length;
  const links = (noFront.match(LINK_RE) || []).length;

  const diff = computeDifficulty({
    wordCount,
    proseText: prose,
    citations,
    advancedMathSymbols: mathStats.advanced,
  });

  // Map continuous multiplier to legacy 3-bucket label for backward compat.
  let complexity: "beginner" | "intermediate" | "advanced" = "intermediate";
  if (diff.multiplier <= 0.95) complexity = "beginner";
  else if (diff.multiplier >= 1.10) complexity = "advanced";

  const analysis: ContentAnalysis = {
    wordCount,
    codeBlocks,
    codeLines,
    mathEquations: mathStats.inline + mathStats.block,
    mathBlockLines: mathStats.blockLines,
    images,
    imageStudyCount,
    tables,
    tableRows,
    callouts,
    headings,
    lists,
    links,
    citations,
    avgSentenceLength: diff.avgSentenceLength,
    typeTokenRatio: diff.ttr,
    advancedMathSymbols: mathStats.advanced,
    difficultyScore: diff.score,
    difficultyMultiplier: diff.multiplier,
    complexity,
  };

  cacheSet(analysisCache, cacheKey, analysis);
  cacheSet(codeBlockSidecar, cacheKey, blocks);
  cacheSet(mathStatsSidecar, cacheKey, mathStats);

  return analysis;
}

const codeBlockSidecar = new Map<string, CodeChunk[]>();
const mathStatsSidecar = new Map<string, MathStats>();

function resolveWpm(config: ReadTimeConfig): number {
  if (typeof config.wordsPerMinute === "number" && config.wordsPerMinute > 0) {
    return config.wordsPerMinute;
  }
  if (config.category) {
    const slug = config.category.toLowerCase().trim();
    if (WPM_BY_CATEGORY[slug] !== undefined) return WPM_BY_CATEGORY[slug];
    // Try the first segment of "machine-learning/foo".
    const first = slug.split("/")[0];
    if (WPM_BY_CATEGORY[first] !== undefined) return WPM_BY_CATEGORY[first];
  }
  return DEFAULT_WPM;
}

export function calculateReadTime(
  content: string,
  config: ReadTimeConfig = {},
): ReadTimeResult {
  const cfg = { ...DEFAULTS, ...config };
  const wpm = resolveWpm(config);
  const cacheKey = `${hashContent(content)}|${wpm}|${JSON.stringify(cfg)}`;
  const cached = resultCache.get(cacheKey);
  if (cached) return cached;

  const analysis = analyzeContent(content);
  const contentHash = hashContent(content);
  const blocks = codeBlockSidecar.get(contentHash) || [];
  const mathStats = mathStatsSidecar.get(contentHash) || {
    inline: 0,
    inlineSymbols: 0,
    block: 0,
    blockLines: 0,
    advanced: 0,
  };

  // Prose
  const baseReadingTime = analysis.wordCount / wpm;

  // Code: per-line × language multiplier
  const codeSeconds = blocks.reduce((sum, b) => {
    const mult = CODE_LANG_MULTIPLIER[b.lang] ?? DEFAULT_CODE_MULTIPLIER;
    return sum + b.lines * cfg.codeSecPerLine * mult;
  }, 0);
  const codeBlockTime = codeSeconds / 60;

  // Math: inline scaled by symbol count (capped); block = base + per-line.
  const inlineMathSeconds =
    Math.min(mathStats.inlineSymbols, mathStats.inline * 8) *
    (cfg.inlineMathSec / 3); // inlineMathSec calibrated against ~3 symbols
  const blockMathSeconds =
    mathStats.block * cfg.blockMathBaseSec +
    mathStats.blockLines * cfg.blockMathPerLineSec;
  const mathTime = (inlineMathSeconds + blockMathSeconds) / 60;

  // Images
  const regularImages = analysis.images - analysis.imageStudyCount;
  const imageSeconds =
    regularImages * cfg.imageSec + analysis.imageStudyCount * cfg.studyImageSec;
  const imageTime = imageSeconds / 60;

  // Tables
  const tableSeconds =
    analysis.tables * cfg.tableBaseSec + analysis.tableRows * cfg.tableRowSec;
  const tableTime = tableSeconds / 60;

  // Callouts
  const calloutTime = (analysis.callouts * cfg.calloutSec) / 60;

  // Difficulty multiplier applied to prose time only (other costs are already
  // calibrated to their per-element effort).
  const adjustedProse = baseReadingTime * analysis.difficultyMultiplier;
  const complexityAdjustment = adjustedProse - baseReadingTime;

  let totalTime =
    adjustedProse +
    codeBlockTime +
    mathTime +
    imageTime +
    tableTime +
    calloutTime;

  totalTime = Math.max(cfg.minReadTime, totalTime);
  totalTime = Math.min(cfg.maxReadTime, totalTime);

  const readTimeMinutes = Math.max(1, Math.ceil(totalTime));
  const result: ReadTimeResult = {
    readTime: `${readTimeMinutes} min read`,
    readTimeMinutes,
    analysis,
    breakdown: {
      baseReadingTime,
      codeBlockTime,
      mathTime,
      imageTime,
      tableTime,
      calloutTime,
      complexityAdjustment,
      totalTime,
    },
  };

  cacheSet(resultCache, cacheKey, result);
  return result;
}

export function getReadTime(content: string, config?: ReadTimeConfig): string {
  return calculateReadTime(content, config).readTime;
}

/**
 * Compatibility wrapper — kept so existing call sites continue to work.
 * `tags` is no longer used (the previous "machine learning" tag bump was
 * effectively dead code). `category` drives the WPM baseline instead.
 */
export function calculateReadTimeWithTags(
  content: string,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  tags: string[] = [],
  category: string = "",
  config: ReadTimeConfig = {},
): ReadTimeResult {
  return calculateReadTime(content, { ...config, category });
}
