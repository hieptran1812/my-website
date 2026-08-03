#!/usr/bin/env node
/**
 * Render a GFM table to a WebP image.
 *
 * Substack has no table node — not in its editor, not in its document format —
 * so a pasted or pushed markdown table collapses into one paragraph of literal
 * pipes. An image is the only way to keep the grid, and it uploads like any
 * other figure.
 *
 * Widths are measured with a per-character estimate rather than real font
 * metrics: SVG has no text-measuring API here, and the estimate only has to be
 * generous enough that nothing clips.
 *
 *   import { renderTable } from "./table-to-image.mjs";
 *   await renderTable(markdownTable, "/path/out.webp");
 */
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const sharp = require("sharp");

const FONT = "Georgia, 'Times New Roman', serif";
const FONT_SIZE = 15;
const HEAD_SIZE = 15;
const PAD_X = 14;
const ROW_H = 34;
const HEAD_H = 38;
const SCALE = 2; // retina
// Substack lays the body out at ~700 CSS px. Stay near that and wrap long cells
// instead of growing sideways — a 1200px table shrinks to 9px type on the page.
const MAX_WIDTH = 720;
const MAX_COL = 260;
const LINE_H = 21;

const COLOURS = {
  text: "#1a1a1a",
  head: "#111111",
  headBg: "#f2f1ee",
  zebra: "#fafaf8",
  rule: "#dcdcd6",
  frame: "#c9c9c2",
};

const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

/** Markdown inside a cell: keep the words, remember whether the cell is bold. */
function cellText(raw) {
  let s = raw.trim();
  const bold = /^\*\*[\s\S]*\*\*$/.test(s);
  s = s.replace(/\*\*(.*?)\*\*/g, "$1").replace(/\*(.*?)\*/g, "$1")
    .replace(/`(.*?)`/g, "$1").replace(/\[(.*?)\]\([^)]*\)/g, "$1")
    .replace(/\\([$%&#_{}])/g, "$1")
    .replace(/\$\{?([^$]*?)\}?\$/g, "$1") // math spans read as their content
    .replace(/(^|\s)--(\s|$)/g, "$1—$2")
    .replace(/^--$/, "—");
  return { text: s, bold };
}

/** Parse a GFM table into a header row, body rows and per-column alignment. */
export function parseTable(md) {
  const lines = md.trim().split("\n").map((l) => l.trim()).filter(Boolean);
  const split = (l) => l.replace(/^\||\|$/g, "").split("|").map(cellText);
  const header = split(lines[0]);
  const align = lines[1].replace(/^\||\|$/g, "").split("|").map((s) => {
    const t = s.trim();
    if (t.startsWith(":") && t.endsWith(":")) return "middle";
    if (t.endsWith(":")) return "end";
    return "start";
  });
  const body = lines.slice(2).map(split);
  return { header, body, align };
}

const textWidth = (s, size) => {
  // Digits and capitals run wider than lowercase; one factor, chosen generous.
  let w = 0;
  for (const ch of s) w += /[MWmw@]/.test(ch) ? 0.95 : /[il.,:;'|]/.test(ch) ? 0.32 : /[A-Z0-9]/.test(ch) ? 0.6 : 0.52;
  return w * size;
};

export function tableToSvg(md) {
  const { header, body, align } = parseTable(md);
  const cols = header.length;
  const rows = body.map((r) => {
    while (r.length < cols) r.push({ text: "", bold: false });
    return r.slice(0, cols);
  });

  // Natural width per column, capped so one verbose column can't stretch the
  // table past the page; the surplus is taken from the widest columns first.
  let widths = header.map((h, i) => {
    const cells = [h, ...rows.map((r) => r[i])];
    const natural = Math.max(...cells.map((c) => textWidth(c.text, c.bold ? HEAD_SIZE : FONT_SIZE))) + PAD_X * 2;
    return Math.min(natural, MAX_COL);
  });
  let total = widths.reduce((a, b) => a + b, 0);
  while (total > MAX_WIDTH) {
    const widest = widths.indexOf(Math.max(...widths));
    if (widths[widest] <= 90) break;
    widths[widest] -= 10;
    total -= 10;
  }

  /** Greedy word wrap to the cell's usable width. */
  const wrap = (text, colWidth, size) => {
    const usable = colWidth - PAD_X * 2;
    const out = [];
    let line = "";
    for (const word of String(text).split(/\s+/).filter(Boolean)) {
      const candidate = line ? `${line} ${word}` : word;
      if (textWidth(candidate, size) <= usable || !line) line = candidate;
      else { out.push(line); line = word; }
    }
    if (line) out.push(line);
    return out.length ? out : [""];
  };

  const headLines = header.map((h, i) => wrap(h.text, widths[i], HEAD_SIZE));
  const bodyLines = rows.map((row) => row.map((c, i) => wrap(c.text, widths[i], FONT_SIZE)));

  const headH = Math.max(HEAD_H, Math.max(...headLines.map((l) => l.length)) * LINE_H + 16);
  const rowH = bodyLines.map((lines) => Math.max(ROW_H, Math.max(...lines.map((l) => l.length)) * LINE_H + 13));

  const W = Math.ceil(widths.reduce((a, b) => a + b, 0));
  const H = headH + rowH.reduce((a, b) => a + b, 0);
  const xs = widths.reduce((acc, w) => [...acc, acc[acc.length - 1] + w], [0]);
  const ys = rowH.reduce((acc, h) => [...acc, acc[acc.length - 1] + h], [headH]);

  const cellX = (i, a) => (a === "start" ? xs[i] + PAD_X : a === "end" ? xs[i + 1] - PAD_X : xs[i] + widths[i] / 2);
  const anchor = (a) => (a === "start" ? "start" : a === "end" ? "end" : "middle");

  const parts = [];
  parts.push(`<rect x="0" y="0" width="${W}" height="${H}" fill="#ffffff"/>`);
  parts.push(`<rect x="0" y="0" width="${W}" height="${headH}" fill="${COLOURS.headBg}"/>`);
  rows.forEach((_, r) => {
    if (r % 2 === 1) parts.push(`<rect x="0" y="${ys[r]}" width="${W}" height="${rowH[r]}" fill="${COLOURS.zebra}"/>`);
  });

  /** Lay a wrapped cell out vertically centred in its row. */
  const drawCell = (lines, i, top, height, size, bold) => {
    const first = top + (height - lines.length * LINE_H) / 2 + size - 1;
    lines.forEach((line, n) => {
      parts.push(`<text x="${cellX(i, align[i]).toFixed(1)}" y="${(first + n * LINE_H).toFixed(1)}" text-anchor="${anchor(align[i])}" font-family="${FONT}" font-size="${size}"${bold ? ' font-weight="700"' : ""} fill="${bold ? COLOURS.head : COLOURS.text}">${esc(line)}</text>`);
    });
  };

  headLines.forEach((lines, i) => drawCell(lines, i, 0, headH, HEAD_SIZE, true));
  bodyLines.forEach((row, r) => row.forEach((lines, i) => drawCell(lines, i, ys[r], rowH[r], FONT_SIZE, rows[r][i].bold)));

  // Rules: under the header, between rows, and one frame around the whole grid.
  parts.push(`<line x1="0" y1="${headH}" x2="${W}" y2="${headH}" stroke="${COLOURS.frame}" stroke-width="1.2"/>`);
  ys.slice(1, -1).forEach((y) => parts.push(`<line x1="0" y1="${y}" x2="${W}" y2="${y}" stroke="${COLOURS.rule}" stroke-width="0.8"/>`));
  xs.slice(1, -1).forEach((x) => parts.push(`<line x1="${x.toFixed(1)}" y1="0" x2="${x.toFixed(1)}" y2="${H}" stroke="${COLOURS.rule}" stroke-width="0.8"/>`));
  parts.push(`<rect x="0.5" y="0.5" width="${W - 1}" height="${H - 1}" fill="none" stroke="${COLOURS.frame}" stroke-width="1"/>`);

  return { svg: `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">${parts.join("")}</svg>`, W, H };
}

export async function renderTable(md, outPath) {
  const { svg, W, H } = tableToSvg(md);
  await sharp(Buffer.from(svg), { density: 72 * SCALE })
    .resize(W * SCALE, H * SCALE)
    .webp({ quality: 92 })
    .toFile(outPath);
  return { width: W * SCALE, height: H * SCALE };
}

// CLI: node table-to-image.mjs <table.md> <out.webp>
if (process.argv[1] && process.argv[1].endsWith("table-to-image.mjs")) {
  const [src, out] = process.argv.slice(2);
  if (src && out) {
    const fs = await import("node:fs");
    const info = await renderTable(fs.readFileSync(src, "utf8"), out);
    console.log(`${out} ${info.width}x${info.height}`);
  }
}
