/**
 * Substack-style paywall for the `trading/` section.
 *
 * The gate is enforced on the SERVER: `buildPreviewHtml()` cuts the rendered
 * article HTML down to its opening blocks and only that prefix is ever sent to
 * the browser. There is no hidden-then-CSS-blurred body to un-hide from
 * devtools, and no API route that will hand back the rest (see the guards in
 * `/api/blog/[slug]`, `/api/blog/article` and `/api/blog/articles`).
 *
 * **Gate on the resolved file path, not the requested slug.** A slug arrives
 * from the URL and can be spelled more than one way for the same file:
 * `findArticleFile()` walks `..` segments, so
 * `?slug=paper-reading/../trading/x` resolves to a trading post while the
 * string does not start with `trading/`. Page routes happen to be safe because
 * HTTP path normalisation collapses `../` before Next sees it, but API query
 * params are never normalised. `isPaywalledFile()` asks the filesystem where
 * the post actually lives, which no amount of slug spelling can change.
 */

import path from "path";

export { PAYWALL_SUBSCRIBE_URL } from "./paywallConfig";

/** Corpus-relative directories whose posts are gated. */
const PAYWALLED_PREFIXES = ["trading/"];

/** Frontmatter categories whose posts are gated, for posts filed elsewhere. */
const PAYWALLED_CATEGORIES = new Set(["trading"]);

const BLOG_ROOT = path.join(process.cwd(), "content", "blog");

/** Top-level blocks kept before the gate closes. */
const PREVIEW_MAX_PARAGRAPHS = 3;

/**
 * The heading cut only fires once this many paragraphs are visible, so a post
 * whose first section header lands after one sentence still gets a real teaser.
 */
const PREVIEW_MIN_PARAGRAPHS = 2;

/**
 * Soft cap on preview size, applied only once at least this many paragraphs
 * are visible — so a long TL;DR callout can never eat the whole allowance.
 */
const PREVIEW_MAX_CHARS = 6000;
const PREVIEW_CHAR_CAP_AFTER_PARAGRAPHS = 2;

/**
 * Is the gate switched off for this process?
 *
 * `next dev` runs ungated by default: the gate exists to send readers to
 * Substack, and locally it only hides the post being written from its author.
 * Production is never opened implicitly — only an explicit `DISABLE_PAYWALL=1`
 * does that — and `next build` / `next start` set `NODE_ENV=production`, so a
 * local production run behaves exactly like the deployed site.
 *
 * Set `DISABLE_PAYWALL=0` to exercise the real teaser in dev.
 */
function paywallDisabled(): boolean {
  const flag = process.env.DISABLE_PAYWALL?.trim().toLowerCase();
  if (flag === "1" || flag === "true") return true;
  if (flag === "0" || flag === "false") return false;
  return process.env.NODE_ENV === "development";
}

/**
 * Ground truth: does this file live under a gated directory of the corpus?
 * `absPath` is the path `findArticleFile()` actually resolved, so `..` games in
 * the requested slug cannot change the answer.
 *
 * Fails closed — a path that resolves outside the blog root is not something
 * this site should be rendering, so it is treated as gated rather than open.
 */
export function isPaywalledFile(absPath: string, blogRoot = BLOG_ROOT): boolean {
  if (paywallDisabled()) return false;

  const rel = path
    .relative(path.resolve(blogRoot), path.resolve(absPath))
    .split(path.sep)
    .join("/")
    .toLowerCase();

  if (rel.startsWith("../") || rel === ".." || path.isAbsolute(rel)) return true;

  return PAYWALLED_PREFIXES.some((prefix) => rel.startsWith(prefix));
}

/**
 * Slug/category fallback, for callers that never resolved a path. Weaker than
 * `isPaywalledFile()` — prefer that whenever the file path is in hand.
 */
export function isPaywalledPost(slug: string, category?: string): boolean {
  if (paywallDisabled()) return false;

  const normalized = (slug || "").replace(/^\/+/, "").toLowerCase();
  if (PAYWALLED_PREFIXES.some((prefix) => normalized.startsWith(prefix))) {
    return true;
  }

  return PAYWALLED_CATEGORIES.has((category || "").trim().toLowerCase());
}

const VOID_TAGS = new Set([
  "area",
  "base",
  "br",
  "col",
  "embed",
  "hr",
  "img",
  "input",
  "link",
  "meta",
  "param",
  "source",
  "track",
  "wbr",
]);

const HEADING_TAG = /^h[1-6]$/;

interface HtmlBlock {
  /** Lowercased tag name, or null for text / comments. */
  tag: string | null;
  html: string;
}

/**
 * Index of the `>` closing the tag that starts at `start`, skipping any `>`
 * that sits inside a quoted attribute value.
 */
function findTagEnd(html: string, start: number): number {
  let quote: string | null = null;
  for (let i = start + 1; i < html.length; i++) {
    const ch = html[i];
    if (quote) {
      if (ch === quote) quote = null;
    } else if (ch === '"' || ch === "'") {
      quote = ch;
    } else if (ch === ">") {
      return i;
    }
  }
  return -1;
}

/**
 * Index just past the `</tag>` that closes the element opened before `from`,
 * counting nested same-name tags. -1 when the document never closes it.
 */
function findMatchingClose(html: string, tag: string, from: number): number {
  let depth = 1;
  let i = from;

  while (i < html.length) {
    const lt = html.indexOf("<", i);
    if (lt === -1) return -1;

    if (html.startsWith("<!--", lt)) {
      const end = html.indexOf("-->", lt);
      i = end === -1 ? html.length : end + 3;
      continue;
    }

    const nameMatch = /^<\/?([a-zA-Z][a-zA-Z0-9:-]*)/.exec(
      html.slice(lt, lt + 64),
    );
    if (!nameMatch) {
      i = lt + 1;
      continue;
    }

    const tagEnd = findTagEnd(html, lt);
    if (tagEnd === -1) return -1;

    const name = nameMatch[1].toLowerCase();
    if (name === tag) {
      const isClose = html[lt + 1] === "/";
      if (isClose) {
        depth--;
        if (depth === 0) return tagEnd + 1;
      } else if (html[tagEnd - 1] !== "/" && !VOID_TAGS.has(name)) {
        depth++;
      }
    }

    i = tagEnd + 1;
  }

  return -1;
}

/**
 * Split rendered article HTML into its top-level blocks so a cut always lands
 * on an element boundary and never mid-tag.
 *
 * Callouts matter here: `remarkCallouts` emits the wrapper as raw
 * `<div class="callout">…` / `</div></div>` html nodes with the body's `<p>`
 * elements as siblings between them. Depth counting re-joins those into one
 * block, so a TL;DR box is kept or dropped whole.
 */
function splitTopLevelBlocks(html: string): HtmlBlock[] {
  const blocks: HtmlBlock[] = [];
  let i = 0;

  while (i < html.length) {
    const lt = html.indexOf("<", i);

    if (lt === -1) {
      const text = html.slice(i);
      if (text.trim()) blocks.push({ tag: null, html: text });
      break;
    }

    if (lt > i) {
      const text = html.slice(i, lt);
      if (text.trim()) blocks.push({ tag: null, html: text });
    }

    if (html.startsWith("<!--", lt)) {
      const end = html.indexOf("-->", lt);
      const stop = end === -1 ? html.length : end + 3;
      blocks.push({ tag: null, html: html.slice(lt, stop) });
      i = stop;
      continue;
    }

    const nameMatch = /^<([a-zA-Z][a-zA-Z0-9:-]*)/.exec(html.slice(lt, lt + 64));
    if (!nameMatch) {
      blocks.push({ tag: null, html: html.slice(lt, lt + 1) });
      i = lt + 1;
      continue;
    }

    const tag = nameMatch[1].toLowerCase();
    const tagEnd = findTagEnd(html, lt);
    if (tagEnd === -1) {
      blocks.push({ tag, html: html.slice(lt) });
      break;
    }

    if (html[tagEnd - 1] === "/" || VOID_TAGS.has(tag)) {
      blocks.push({ tag, html: html.slice(lt, tagEnd + 1) });
      i = tagEnd + 1;
      continue;
    }

    const close = findMatchingClose(html, tag, tagEnd + 1);
    if (close === -1) {
      blocks.push({ tag, html: html.slice(lt) });
      break;
    }

    blocks.push({ tag, html: html.slice(lt, close) });
    i = close;
  }

  return blocks;
}

export interface PaywallSplit {
  /** The HTML actually shipped to the client. */
  preview: string;
  /** False when the post was short enough that nothing was withheld. */
  truncated: boolean;
  /** Top-level blocks withheld — used for the "N sections left" line. */
  hiddenBlocks: number;
}

/**
 * Cut rendered HTML to its teaser: everything up to the first heading that
 * follows a paragraph, capped at `PREVIEW_MAX_PARAGRAPHS` top-level paragraphs.
 * A leading TL;DR callout is kept — it is the hook, not the article.
 */
export function buildPreviewHtml(html: string): PaywallSplit {
  const blocks = splitTopLevelBlocks(html);

  const kept: string[] = [];
  let paragraphs = 0;
  let chars = 0;
  let index = 0;

  for (; index < blocks.length; index++) {
    const block = blocks[index];
    const tag = block.tag ?? "";

    if (paragraphs >= PREVIEW_MAX_PARAGRAPHS) break;
    if (paragraphs >= PREVIEW_MIN_PARAGRAPHS && HEADING_TAG.test(tag)) break;
    if (
      paragraphs >= PREVIEW_CHAR_CAP_AFTER_PARAGRAPHS &&
      chars >= PREVIEW_MAX_CHARS
    ) {
      break;
    }

    kept.push(block.html);
    chars += block.html.length;
    if (tag === "p") paragraphs++;
  }

  return {
    preview: kept.join("\n"),
    truncated: index < blocks.length,
    hiddenBlocks: blocks.length - index,
  };
}

/** Cut `html` down to its preview. Shared tail of the two helpers below. */
function gate(html: string): { html: string; paywalled: boolean } {
  const { preview, truncated } = buildPreviewHtml(html);
  if (!truncated) return { html, paywalled: false };
  return { html: preview, paywalled: true };
}

/**
 * Preferred entry point: gate a rendered article by where its source file
 * lives. Immune to `..` segments and frontmatter that omits `category`.
 */
export function applyPaywallToFile(
  absPath: string,
  html: string,
): { html: string; paywalled: boolean } {
  if (!isPaywalledFile(absPath)) return { html, paywalled: false };
  return gate(html);
}

/**
 * Slug/category variant, for callers with no file path. Prefer
 * `applyPaywallToFile()`.
 */
export function applyPaywall(
  slug: string,
  category: string | undefined,
  html: string,
): { html: string; paywalled: boolean } {
  if (!isPaywalledPost(slug, category)) return { html, paywalled: false };
  return gate(html);
}
