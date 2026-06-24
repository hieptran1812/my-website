#!/usr/bin/env node
// Validate an animated-figure block authored for the blog (Phase C-anim).
//
// Usage: node check-anim.mjs <slug>-anim-<i>.fig.html
//
// The file must contain exactly one `<figure class="blog-anim"> … </figure>`
// raw-HTML block holding an inline <svg> whose motion is driven by CSS
// @keyframes (or SMIL <animate>). These figures are pasted verbatim into the
// post markdown, so the rules here mirror what the markdown pipeline + the
// browser + the Phase E gate require. See references/animated-figures.md.
//
// Prints `PASS  <rule>` / `FAIL  <rule> — <detail>` per check and exits 1 on any
// FAIL so it composes with the rest of the diagram pipeline. WARN never fails.

import fs from "node:fs";

const file = process.argv[2];
if (!file) {
  console.error("usage: node check-anim.mjs <fig.html>");
  process.exit(2);
}
if (!fs.existsSync(file)) {
  console.error(`FAIL  exists — no such file: ${file}`);
  process.exit(2);
}

const raw = fs.readFileSync(file, "utf8");
const lines = raw.split("\n");

let failed = 0;
const pass = (r) => console.log(`PASS  ${r}`);
const warn = (r, d) => console.log(`WARN  ${r}${d ? ` — ${d}` : ""}`);
const fail = (r, d) => {
  console.log(`FAIL  ${r}${d ? ` — ${d}` : ""}`);
  failed++;
};

// ── locate the single <figure class="blog-anim"> … </figure> block ──────────
// Match anywhere on the line so an *indented* figure still anchors the block and
// gets the precise col-0 message (a common authoring slip) instead of "not found".
const openIdx = lines.findIndex((l) => /<figure class="blog-anim"/.test(l));
const closeIdx = lines.findIndex((l, i) => i >= openIdx && openIdx !== -1 && /<\/figure>\s*$/.test(l));

if (openIdx === -1) {
  fail("figure-wrapper", 'no line contains `<figure class="blog-anim"`');
} else if (!/^<figure class="blog-anim"/.test(lines[openIdx])) {
  fail("col-0", `\`<figure>\` must start at column 0 (line ${openIdx + 1} is indented — CommonMark won't treat it as a raw-HTML block)`);
} else {
  pass("figure-wrapper");
}
if (openIdx !== -1 && closeIdx === -1) fail("figure-wrapper", "missing a `</figure>` close line");

const blk = openIdx !== -1 && closeIdx !== -1
  ? lines.slice(openIdx, closeIdx + 1)
  : lines;
const block = blk.join("\n");

// 1. No blank lines inside the block — CommonMark ends a raw-HTML block at the
//    first blank line, which would shatter the SVG into escaped text.
{
  const blanks = [];
  // interior only (between the open and close tags)
  for (let i = 1; i < blk.length - 1; i++) {
    if (blk[i].trim() === "") blanks.push(openIdx + i + 1);
  }
  if (blanks.length) fail("no-blank-lines", `blank line(s) inside block at line ${blanks.join(", ")}`);
  else pass("no-blank-lines");
}

// 2. Inline <svg> with a viewBox (responsive intrinsic ratio).
if (!/<svg[\s>]/.test(block)) fail("inline-svg", "no `<svg>` element in the block");
else if (!/<svg[^>]*\bviewBox=/i.test(block)) fail("inline-svg", "`<svg>` is missing a `viewBox`");
else pass("inline-svg");

// 3. No script / event handlers / remote refs — declarative only.
{
  const bad = [];
  if (/<script[\s>]/i.test(block)) bad.push("<script>");
  if (/\son[a-z]+\s*=/i.test(block)) bad.push("on*= handler");
  if (/javascript:/i.test(block)) bad.push("javascript: URL");
  if (/(?:href|xlink:href|src)\s*=\s*["']\s*https?:/i.test(block)) bad.push("remote href/src");
  if (bad.length) fail("declarative-only", bad.join(", "));
  else pass("declarative-only");
}

// 4. It actually animates: a @keyframes referenced by an animation, or SMIL.
{
  const keyframes = [...block.matchAll(/@keyframes\s+([\w-]+)/g)].map((m) => m[1]);
  const hasCssAnim = /\banimation(?:-name)?\s*:/.test(block) && keyframes.length > 0;
  const hasSmil = /<animate[\s/>]|<animateTransform|<animateMotion/.test(block);
  if (!hasCssAnim && !hasSmil) {
    fail("has-animation", "no `@keyframes` + `animation:` pair and no SMIL `<animate>`");
  } else {
    pass("has-animation");
    // soft: keyframes defined but never referenced by an animation property
    const used = keyframes.filter((k) => new RegExp(`animation(?:-name)?\\s*:[^;}]*\\b${k}\\b`).test(block));
    const orphan = keyframes.filter((k) => !used.includes(k));
    if (hasCssAnim && orphan.length) warn("has-animation", `@keyframes never referenced: ${orphan.join(", ")}`);
  }
}

// 5. Honors prefers-reduced-motion (freezes/pauses motion).
{
  const rm = block.match(/@media[^{]*prefers-reduced-motion\s*:\s*reduce[^{]*\{([\s\S]*?)\}\s*\}?/i);
  if (!/prefers-reduced-motion/i.test(block)) {
    fail("reduced-motion", "no `@media (prefers-reduced-motion: reduce)` guard");
  } else if (rm && !/animation\s*:\s*none|animation-play-state\s*:\s*paused/i.test(rm[1])) {
    warn("reduced-motion", "guard present but doesn't set `animation:none` / `animation-play-state:paused`");
    pass("reduced-motion");
  } else {
    pass("reduced-motion");
  }
}

// 6. Accessible label: role=img + aria-label, or a <title>.
{
  const hasAria = /<svg[^>]*\brole=["']img["']/i.test(block) && /<svg[^>]*\baria-label=/i.test(block);
  const hasTitle = /<title>[^<]+<\/title>/i.test(block);
  if (!hasAria && !hasTitle) fail("accessible-label", "`<svg>` needs role=\"img\" + aria-label, or a <title>");
  else pass("accessible-label");
}

// 7. Caption present.
if (!/<figcaption>[\s\S]*?<\/figcaption>/.test(block)) fail("figcaption", "no `<figcaption>` describing the motion");
else pass("figcaption");

// 8. Responsive sizing — viewBox + a width/max-width in the svg style; no big
//    fixed pixel width/height attributes that overflow the ~896px column.
{
  const svgTag = (block.match(/<svg[^>]*>/i) || [""])[0];
  const styleHasWidth = /style=["'][^"']*\b(?:max-)?width\s*:/i.test(svgTag);
  const fixedW = svgTag.match(/\bwidth=["'](\d+)(?:px)?["']/i);
  const fixedH = svgTag.match(/\bheight=["'](\d+)(?:px)?["']/i);
  if (!styleHasWidth) {
    warn("responsive", "add `style=\"width:100%;height:auto;max-width:NNNpx\"` to the <svg> for fluid scaling");
  } else {
    pass("responsive");
  }
  if (fixedW && Number(fixedW[1]) > 896) fail("responsive", `fixed width=${fixedW[1]} overflows the 896px column — use viewBox + style width`);
  if (fixedH && Number(fixedH[1]) > 900) warn("responsive", `large fixed height=${fixedH[1]} — prefer viewBox aspect`);
}

// 9. Accent discipline + size sanity (soft, matches the squint rule).
{
  const hexes = new Set([...block.matchAll(/#[0-9a-fA-F]{3,8}\b/g)].map((m) => m[0].toLowerCase()));
  if (hexes.size > 6) warn("squint", `${hexes.size} distinct hex colors — keep to ~1 accent + neutrals (prefer CSS vars)`);
  const kb = Buffer.byteLength(block, "utf8") / 1024;
  if (kb > 200) fail("size", `block is ${kb.toFixed(0)} KB — far too heavy for inline SVG`);
  else if (kb > 60) warn("size", `block is ${kb.toFixed(0)} KB — lean inline SVG is usually < 20 KB`);
}

console.log("");
if (failed) {
  console.log(`RESULT: ${failed} FAIL — fix the .fig.html and re-run check-anim.mjs`);
  process.exit(1);
}
console.log("RESULT: all animated-figure rules passed");
process.exit(0);
