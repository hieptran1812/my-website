/**
 * dedash.mjs: remove em dashes (and spaced en dashes / ` -- `) from Markdown
 * prose before it goes to Substack, per the no-em-dash house rule.
 *
 * Fenced code blocks, inline code and link/image URLs are left untouched.
 * Unspaced en dashes in ranges (2007–2021) are fine and kept.
 *
 * Rules, per dash, from the context around it:
 *   paired dashes in one sentence  -> commas, or parentheses if the aside has a comma
 *   right side opens with a conjunction / relative word -> comma
 *   a colon already elsewhere in the sentence -> full stop + capital
 *   otherwise -> colon
 *   dash opening a line (attribution) -> dropped; dash ending a line -> colon
 */

const DASH = String.raw`(?:\s*—\s*|\s+–\s+|\s+--\s+)`;
const CONJ = /^(and|but|or|so|yet|nor|which|who|whom|whose|where|when|while|because|though|although|not|as|if|unless|until|plus|then|even|especially|including|usually|often|typically)\b/i;

function protect(text) {
  const saved = [];
  const put = (s) => `\u0000${saved.push(s) - 1}\u0000`;
  // inline code and link/image targets
  let t = text.replace(/`[^`\n]*`/g, put).replace(/\]\([^)\s]*(?:\s+"[^"]*")?\)/g, put);
  return { t, restore: (s) => s.replace(/\u0000(\d+)\u0000/g, (_, i) => saved[+i]) };
}

function fixSentence(s) {
  const re = new RegExp(DASH, "g");
  const hits = [...s.matchAll(re)];
  if (!hits.length) return s;
  // paired aside: exactly two dashes, short middle
  if (hits.length === 2) {
    const [a, b] = hits;
    const mid = s.slice(a.index + a[0].length, b.index);
    if (mid.length > 0 && mid.length < 160) {
      const after = s.slice(b.index + b[0].length);
      const before = s.slice(0, a.index);
      if (/[,;]/.test(mid)) return `${before} (${mid})${/^[.,;:!?)]/.test(after) ? "" : " "}${after}`;
      return `${before}, ${mid}${/^[.,;:!?)]/.test(after) ? "" : ", "}${after}`;
    }
  }
  let out = "";
  let last = 0;
  for (const h of hits) {
    const before = s.slice(last, h.index);
    const rest = s.slice(h.index + h[0].length);
    out += before;
    if (!before.trim() && !out.trim()) { last = h.index + h[0].length; continue; } // leading dash
    if (!rest.trim()) { out += ":"; last = h.index + h[0].length; continue; }
    const hasColon = /:\s/.test(s.slice(0, h.index)) || /:\s/.test(rest);
    if (CONJ.test(rest.replace(/^[*_]+/, ""))) out += ", ";
    else if (hasColon && /^[*_]*[a-z]/.test(rest)) {
      out += ". ";
      last = h.index + h[0].length;
      const m = rest.match(/^([*_]*)([a-z])/);
      out += m[1] + m[2].toUpperCase();
      last += m[0].length;
      continue;
    } else out += ": ";
    last = h.index + h[0].length;
  }
  return out + s.slice(last);
}

export function dedashLine(line) {
  if (!/—|\s–\s|\s--\s/.test(line)) return line;
  if (/^\s*\|/.test(line)) return line.replace(/\s+—\s+/g, ", "); // table rows: keep cell structure
  const { t, restore } = protect(line);
  // split into sentences so a colon in one sentence does not steer another
  const parts = t.split(/(?<=[.!?])\s+(?=[*_"(]*[A-Z0-9\\])/);
  let fixed = parts.map(fixSentence).join(" ");
  fixed = fixed.replace(/,\s*,/g, ",").replace(/:\s*:/g, ":").replace(/ ,/g, ",");
  return restore(fixed);
}

export function dedashMarkdown(md) {
  let inFence = false;
  let fm = 0;
  return md.split("\n").map((line, i) => {
    if (i === 0 && line === "---") { fm = 1; return line; }
    if (fm === 1) { if (line === "---") fm = 2; return line; }
    if (/^\s*(```|~~~)/.test(line)) { inFence = !inFence; return line; }
    return inFence ? line : dedashLine(line);
  }).join("\n");
}

export function dedashText(s) {
  return s ? dedashLine(s) : s;
}

export function countDashes(md) {
  let inFence = false, n = 0;
  for (const line of md.split("\n")) {
    if (/^\s*(```|~~~)/.test(line)) { inFence = !inFence; continue; }
    if (!inFence) n += (line.replace(/`[^`]*`/g, "").match(/—|\s–\s|\s--\s/g) || []).length;
  }
  return n;
}
