// Full-article translation by in-place text-node replacement.
//
// The goal: translate the WHOLE article into a target language while keeping the
// exact same HTML structure, classes, fonts, images, math and code untouched —
// only the human-readable prose changes. We do that by walking the rendered DOM,
// collecting the translatable text nodes, translating them in batches through the
// same-origin /api/translate proxy, and writing the result straight back into
// each node's `nodeValue`. Because we never touch elements (only the text inside
// them), every bit of formatting/styling is preserved automatically.
//
// Originals are captured up front so "Show original" restores the article byte
// for byte, and re-translating to another language always starts from the source
// text (no compounding drift).

import { translateText } from "./translate";

/** Tags whose text must never be translated (code, math source, raw markup). */
const SKIP_TAGS = new Set([
  "SCRIPT",
  "STYLE",
  "NOSCRIPT",
  "PRE",
  "CODE",
  "KBD",
  "SAMP",
  "VAR",
  "TEXTAREA",
  "MATH",
]);

/** Class names / attributes that mark a subtree as do-not-translate. KaTeX is
 *  rendered client-side into `.math-expression` / `.katex`; those (and animated
 *  SVG figures) must stay untouched. */
const SKIP_CLASSES = [
  "katex",
  "katex-display",
  "katex-mathml",
  "math-expression",
  "math-error",
  "blog-anim",
];
const SVG_NS = "http://www.w3.org/2000/svg";

// For the server API: 3000 chars fits under the 4000-char proxy limit.
// For the on-device model: 1200 chars ≈ 343 output tokens, which avoids
// the truncation that occurs when MAX_NEW_TOKENS is too small relative to
// the number of segments packed into one batch.
const MAX_BATCH = 1200;
// Blank line between segments: survives translation as a clean boundary and is
// extremely unlikely to appear *inside* a single (whitespace-collapsed) node.
const SEP = "\n\n";

export interface TextNodeEntry {
  node: Text;
  /** Original nodeValue, verbatim — used for exact restore. */
  original: string;
  /** Leading whitespace of the original (preserves inter-element spacing). */
  prefix: string;
  /** Trailing whitespace of the original. */
  suffix: string;
  /** Trimmed, whitespace-collapsed core — the actual text we translate. */
  core: string;
}

export interface TranslateSummary {
  total: number;
  failed: number;
}

function isSkipped(el: Element | null, root: HTMLElement): boolean {
  while (el && el !== root.parentElement) {
    // Anything inside an SVG (e.g. animated diagrams) is layout-sensitive.
    if ((el as Element).namespaceURI === SVG_NS) return true;
    if (SKIP_TAGS.has(el.tagName)) return true;
    if (el.classList) {
      for (const c of SKIP_CLASSES) if (el.classList.contains(c)) return true;
    }
    if (el.getAttribute && el.getAttribute("data-no-translate") !== null) {
      return true;
    }
    if (el.getAttribute && el.getAttribute("aria-hidden") === "true") {
      // KaTeX duplicates content in an aria-hidden span; skip such mirrors.
      return true;
    }
    if (el === root) break;
    el = el.parentElement;
  }
  return false;
}

/** Collect every translatable text node in document order. Called ONCE while the
 *  article is still in its original language so `original`/`core` are the source. */
export function collectTranslatable(root: HTMLElement): TextNodeEntry[] {
  const entries: TextNodeEntry[] = [];
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      const text = node.nodeValue ?? "";
      if (!text.trim()) return NodeFilter.FILTER_REJECT;
      if (isSkipped((node as Text).parentElement, root)) {
        return NodeFilter.FILTER_REJECT;
      }
      return NodeFilter.FILTER_ACCEPT;
    },
  });

  let n = walker.nextNode();
  while (n) {
    const t = n as Text;
    const raw = t.nodeValue ?? "";
    const prefix = raw.match(/^\s*/)?.[0] ?? "";
    const suffix = raw.match(/\s*$/)?.[0] ?? "";
    const coreRaw = raw.slice(prefix.length, raw.length - suffix.length);
    // Collapse internal whitespace so no stray "\n\n" can collide with our
    // batch separator. Display whitespace is collapsed by the browser anyway.
    const core = coreRaw.replace(/\s+/g, " ");
    // Only bother with nodes that actually contain a letter — pure numbers,
    // punctuation or symbols ("→", "42", "—") need no translation.
    if (/\p{L}/u.test(core)) {
      entries.push({ node: t, original: raw, prefix, suffix, core });
    }
    n = walker.nextNode();
  }
  return entries;
}

/** Greedily pack entries into batches that stay under MAX_BATCH characters. */
function buildBatches(entries: TextNodeEntry[]): TextNodeEntry[][] {
  const batches: TextNodeEntry[][] = [];
  let cur: TextNodeEntry[] = [];
  let len = 0;
  for (const e of entries) {
    const add = e.core.length + SEP.length;
    if (cur.length && len + add > MAX_BATCH) {
      batches.push(cur);
      cur = [];
      len = 0;
    }
    cur.push(e);
    len += add;
  }
  if (cur.length) batches.push(cur);
  return batches;
}

function applyTranslation(e: TextNodeEntry, translated: string): void {
  const core = translated.trim();
  // Never blank out a node if the service returned nothing useful.
  e.node.nodeValue = e.prefix + (core || e.core) + e.suffix;
}

async function translateBatch(
  batch: TextNodeEntry[],
  target: string,
  signal?: AbortSignal,
): Promise<void> {
  if (batch.length === 1) {
    const { translation } = await translateText(batch[0].core, target, signal);
    applyTranslation(batch[0], translation);
    return;
  }

  const joined = batch.map((e) => e.core).join(SEP);
  const { translation } = await translateText(joined, target, signal);
  const parts = translation.split(SEP);

  if (parts.length === batch.length) {
    batch.forEach((e, i) => applyTranslation(e, parts[i]));
    return;
  }

  // Boundary count drifted (rare): fall back to per-node translation so the
  // mapping is guaranteed correct, trading a few extra requests for accuracy.
  for (const e of batch) {
    if (signal?.aborted) throw new DOMException("Aborted", "AbortError");
    const { translation: t } = await translateText(e.core, target, signal);
    applyTranslation(e, t);
  }
}

export interface TranslateOptions {
  signal?: AbortSignal;
  /** Max concurrent upstream requests (kept low to stay friendly to the proxy). */
  concurrency?: number;
  onProgress?: (done: number, total: number) => void;
}

/** Translate all collected nodes in place. Resolves with a summary; throws only
 *  on abort. Per-batch failures are counted, not fatal, so a single hiccup
 *  doesn't discard the rest of the translated article. */
export async function translateArticleDom(
  entries: TextNodeEntry[],
  target: string,
  opts: TranslateOptions = {},
): Promise<TranslateSummary> {
  const batches = buildBatches(entries);
  const total = batches.length;
  const concurrency = Math.max(1, Math.min(opts.concurrency ?? 3, total || 1));
  let idx = 0;
  let done = 0;
  let failed = 0;

  opts.onProgress?.(0, total);

  async function worker() {
    while (idx < batches.length) {
      if (opts.signal?.aborted) throw new DOMException("Aborted", "AbortError");
      const my = batches[idx++];
      try {
        await translateBatch(my, target, opts.signal);
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") throw err;
        if (err instanceof Error && err.name === "AbortError") throw err;
        failed++;
      }
      done++;
      opts.onProgress?.(done, total);
    }
  }

  const workers = Array.from({ length: concurrency }, () => worker());
  await Promise.all(workers);
  return { total, failed };
}

/** Restore every node to its captured original text. */
export function restoreOriginals(entries: TextNodeEntry[]): void {
  for (const e of entries) {
    if (e.node.nodeValue !== e.original) e.node.nodeValue = e.original;
  }
}
