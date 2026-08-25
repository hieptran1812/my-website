#!/usr/bin/env node
/**
 * to-substack.mjs — turn a repo blog post into paste-ready Substack content.
 *
 *   node .claude/scripts/to-substack.mjs <slug|path> [more…] [flags]
 *   npm run substack -- <slug|path>
 *
 * Writes `.substack/<slug>.html`: a self-contained preview page with a
 * "Copy post body" button. Click it, then paste into the Substack editor —
 * headings, bold, links, lists, quotes, code and images all survive, and
 * Substack re-hosts every image by fetching the absolute halleywiki.com URL.
 *
 * What the conversion does, and why:
 *   images     `/imgs/blogs/x.png` → absolute https URL, resolved through the
 *              WebP manifest (the repo rewrites png→webp at serve time).
 *   links      cross-references to other posts are re-pointed at the matching
 *              post on the Substack publication — matched against its live
 *              archive by title, then by slug. A reference whose post isn't
 *              published there yet becomes plain text and is listed as
 *              "publish these first". Links to other domains are untouched, and
 *              nothing ever points back at halleywiki.com.
 *   callouts   `> [!important]` → plain blockquote with a bold label.
 *   headings   h1→h2, h4+→bold paragraph (Substack's editor is h1–h3).
 *   math       inline `$…$` → Unicode, but only when KaTeX confirms the Unicode
 *              states exactly what the LaTeX states; anything else, and every
 *              display `$$…$$`, is rendered to a PNG from the exact LaTeX.
 *              A formula KaTeX cannot parse is broken in the source too — it is
 *              left verbatim and reported rather than silently mangled.
 *   anim figs  inline animated <svg> can't survive the paste — dropped, saved
 *              next to the output as .svg so you can screenshot them.
 *
 * Math accuracy is the one thing this script never guesses at. Check a whole
 * series before you publish any of it:
 *
 *   node .claude/scripts/to-substack.mjs all --audit      # every post, math only
 *
 * Flags:
 *   --out <dir>          output dir (default .substack)
 *   --site <url>         absolute base (default https://halleywiki.com)
 *   --open / --no-open   open the preview in the browser (default: open)
 *   --clipboard          also put the HTML on the macOS clipboard directly
 *   --inline-math auto|unicode|image|raw   (default auto)
 *   --display-math image|raw               (default image)
 *   --math-dpi <n>       display-math render dpi (default 200)
 *   --math-bg white|transparent            (default white)
 *   --tables image|keep|list  Substack has no table node, so the default renders
 *                        each table to a WebP; `keep` and `list` leave the
 *                        markdown table or flatten it to bullets instead
 *   --keep-deep-headings leave h4–h6 as headings
 *   --canonical          append an "Originally published at …" footer
 *   --no-figure-alt      drop long alt text (Substack shows it on hover)
 *   --no-smart-dashes    keep the source's literal `--` instead of — / –
 *   --substack <url>     the publication (default https://halleytech.substack.com)
 *   --internal-links substack|strip|absolute   cross-post links: point at the
 *                        matching Substack post (default), drop them, or
 *                        rewrite to halleywiki.com
 *   --refresh-archive    re-fetch the publication archive (cached one hour)
 *   --no-check-images    skip the HEAD check that every image URL is live
 *   --no-verify-math     skip the KaTeX cross-check (not recommended)
 *   --audit              report math only, write nothing; `all` audits every post
 *   --plan               take a folder and print what order to publish it in,
 *                        so cross-links resolve; writes nothing
 *   --markdown           also write `.substack/<slug>.md` (real LaTeX, local
 *                        image paths) — the input for the draft API
 *   --draft              create an **unpublished** Substack draft from that
 *                        Markdown via the `substack` CLI (python-substack).
 *                        Never publishes, schedules or sends. Implies
 *                        --markdown and raw LaTeX. Auth is the CLI's own .env
 *                        or --cookies <file>; this script reads no credential.
 *   --cookies <file>     cookie file handed straight to the `substack` CLI
 *   --link-batch         cross-links to other posts in this same run resolve to
 *                        their future Substack URL, so a whole series links up
 *                        on the first pass instead of needing a second export
 *   --delay <sec>        pause between drafts (default 30) — Substack rate-limits
 *   --cooldown <sec>     wait after a 429 before the next post (default 900)
 *   --max-rate-limits <n>  give up the run after this many 429s (default 8)
 *   --no-tags            create the drafts without tags. Each tag is its own
 *                        API call after the draft exists — on a long run they
 *                        are what trips the rate limit, and `--covers` fills
 *                        them in afterwards at a gentler pace.
 *   --no-slug            let Substack derive the slug from the title
 *   --covers             second pass over the pushed drafts to attach cover
 *                        images (the CLI has no cover support); takes no post
 *                        arguments, works off .substack-created.json
 */

import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";

import matter from "gray-matter";
import katex from "katex";
import { unified } from "unified";
import remarkParse from "remark-parse";
import remarkGfm from "remark-gfm";
import remarkRehype from "remark-rehype";
import rehypeStringify from "rehype-stringify";

import { renderTable } from "./table-to-image.mjs";

const ROOT = path.resolve(fileURLToPath(new URL("../..", import.meta.url)));
const CONTENT = path.join(ROOT, "content/blog");
const PUBLIC = path.join(ROOT, "public");

/* ─────────────────────────────── args ─────────────────────────────── */

function parseArgs(argv) {
  const opts = {
    out: path.join(ROOT, ".substack"),
    site: "https://halleywiki.com",
    open: null, // decided later: open only for a single post
    clipboard: false,
    inlineMath: "auto",
    displayMath: "image",
    mathDpi: 200,
    mathBg: "white",
    tables: "image",
    deepHeadings: false,
    canonical: false,
    figureAlt: true,
    smartDashes: true,
    substack: "https://halleytech.substack.com",
    internalLinks: "substack",
    refreshArchive: false,
    checkImages: true,
    verifyMath: true,
    delay: 30,        // seconds between drafts
    tags: true,       // one API call per tag; --no-tags defers them to --covers
    cooldown: 900,    // wait after a 429 before trying the next post
    maxRateLimits: 8, // consecutive-ish 429s before giving up on the run
    audit: false,
    targets: [],
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = () => argv[++i];
    switch (a) {
      case "--out": opts.out = path.resolve(next()); break;
      case "--site": opts.site = next().replace(/\/+$/, ""); break;
      case "--open": opts.open = true; break;
      case "--no-open": opts.open = false; break;
      case "--clipboard": case "-c": opts.clipboard = true; break;
      case "--inline-math": opts.inlineMath = next(); break;
      case "--display-math": opts.displayMath = next(); break;
      case "--math-dpi": opts.mathDpi = Number(next()); break;
      case "--math-bg": opts.mathBg = next(); break;
      case "--tables": opts.tables = next(); break;
      case "--keep-deep-headings": opts.deepHeadings = true; break;
      case "--canonical": opts.canonical = true; break;
      case "--no-figure-alt": opts.figureAlt = false; break;
      case "--no-smart-dashes": opts.smartDashes = false; break;
      case "--internal-links": opts.internalLinks = next(); break;
      case "--substack": opts.substack = next().replace(/\/+$/, ""); break;
      case "--refresh-archive": opts.refreshArchive = true; break;
      case "--no-check-images": opts.checkImages = false; break;
      case "--no-verify-math": opts.verifyMath = false; break;
      case "--audit": opts.audit = true; opts.checkImages = false; break;
      case "--plan": opts.plan = true; opts.checkImages = false; break;
      case "--covers": opts.covers = true; break;
      case "--markdown": opts.markdown = true; break;
      case "--cookies": opts.cookies = next(); break;
      case "--no-slug": opts.slug = false; break;
      case "--link-batch": opts.linkBatch = true; break;
      case "--force-push": opts.forcePush = true; break;
      case "--delay": opts.delay = Number(next()); break;
      case "--no-tags": opts.tags = false; break;
      case "--cooldown": opts.cooldown = Number(next()); break;
      case "--max-rate-limits": opts.maxRateLimits = Number(next()); break;
      case "--draft":
        opts.draft = true;
        opts.markdown = true;
        opts.open = opts.open ?? false;
        // Split the work by who does it better. python-substack renders `$$…$$`
        // into a real equation image — faithful, and no external renderer. But
        // it degrades inline `$…$` to an unverified, internally inconsistent
        // approximation (measured: "R_win" and "Rₗₒₛₛ" in the same sentence),
        // so inline math stays with our KaTeX-verified converter.
        if (!argv.includes("--inline-math")) opts.inlineMath = "unicode";
        if (!argv.includes("--display-math")) opts.displayMath = "raw";
        break;
      case "--help": case "-h": opts.help = true; break;
      default:
        if (a.startsWith("-")) die(`unknown flag ${a}`);
        opts.targets.push(a);
    }
  }
  return opts;
}

function die(msg) {
  console.error(`to-substack: ${msg}`);
  process.exit(1);
}

/** Accept a path, a slug, or a partial slug; resolve to one markdown file. */
function resolvePost(target) {
  const direct = path.resolve(target);
  if (fs.existsSync(direct) && direct.endsWith(".md")) return direct;

  const slug = path.basename(target, ".md");
  const hits = [];
  (function walk(dir) {
    for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
      const p = path.join(dir, e.name);
      if (e.isDirectory()) walk(p);
      else if (e.name === `${slug}.md`) hits.unshift(p);
      else if (e.name.endsWith(".md") && e.name.includes(slug)) hits.push(p);
    }
  })(CONTENT);

  if (!hits.length) die(`no post matching "${target}" under content/blog`);
  if (hits.length > 1 && path.basename(hits[0], ".md") !== slug) {
    console.error(`to-substack: "${target}" is ambiguous:`);
    for (const h of hits.slice(0, 8)) console.error(`  ${path.relative(ROOT, h)}`);
    process.exit(1);
  }
  return hits[0];
}

/* ─────────────────────── masking helpers ─────────────────────── */

/**
 * Placeholders must be inert to both markdown and HTML: letters and digits
 * only, so nothing italicises, escapes or line-breaks them.
 */
const tok = (kind, i) => `ZZ${kind}${i}ZZ`;
const tokRe = (kind) => new RegExp(`ZZ${kind}(\\d+)ZZ`, "g");

/** Pull fenced/inline code out of the markdown so later regexes can't see it. */
function maskCode(md) {
  const store = [];
  const stash = (m) => { store.push(m); return tok("CODE", store.length - 1); };
  let out = md.replace(
    /(^|\n)([ \t]*)(`{3,}|~{3,})[^\n]*\n[\s\S]*?\n[ \t]*\3[ \t]*(?=\n|$)/g,
    (m, pre) => pre + stash(m.slice(pre.length)),
  );
  out = out.replace(/(`+)(?:(?!\1)[\s\S])+?\1/g, stash);
  return { md: out, store };
}

const unmask = (s, kind, store) => s.replace(tokRe(kind), (_, i) => store[Number(i)]);

/**
 * Extract math before markdown parsing — `_`, `^`, `*` and `\` inside formulas
 * would otherwise be eaten as emphasis. Regex lifted from src/lib/markdown.ts
 * so this script splits math exactly the way the site does.
 */
function maskMath(md) {
  const store = [];
  const stash = (tex, display) => {
    store.push({ tex, display });
    return tok("MATH", store.length - 1);
  };
  let out = md.replace(/\$\$([\s\S]*?)\$\$/g, (_, tex) => stash(tex.trim(), true));
  out = out.replace(
    /(?<![\\$])\$(?!\s)(?:(?=\d)(?=[^$\n]*[\\{}^_])|(?=\D))((?:[^$\n\\]|\\.)+?)(?<!\s)\$(?![A-Za-z0-9])/g,
    (_, tex) => stash(tex, false),
  );
  // That rule deliberately skips digit-only spans so it can't eat currency,
  // which leaves `$0$` rendering as literal dollars on the site. Here the
  // dollars would read as a typo, so unwrap the unambiguous bare numbers —
  // still `\$`-escaped currency is excluded by the lookbehind.
  // Letter-free, digit-led, and the closing `$` may not be followed by a word
  // character — the same guard the site's rule uses to stay off "$5 and $10".
  out = out.replace(
    /(?<![\\$\w])\$(\d|\d[\d\s.,()+\-*/=<>%]{0,40}[\d)%])\$(?![\w$])/g,
    (_, tex) => stash(tex, false),
  );
  return { md: out, store };
}

/* ─────────────────────── LaTeX → Unicode ─────────────────────── */

const SYMBOLS = {
  alpha: "α", beta: "β", gamma: "γ", delta: "δ", epsilon: "ε", varepsilon: "ε",
  zeta: "ζ", eta: "η", theta: "θ", vartheta: "ϑ", iota: "ι", kappa: "κ",
  lambda: "λ", mu: "μ", nu: "ν", xi: "ξ", pi: "π", rho: "ρ", sigma: "σ",
  tau: "τ", upsilon: "υ", phi: "φ", varphi: "φ", chi: "χ", psi: "ψ", omega: "ω",
  Gamma: "Γ", Delta: "Δ", Theta: "Θ", Lambda: "Λ", Xi: "Ξ", Pi: "Π",
  Sigma: "Σ", Upsilon: "Υ", Phi: "Φ", Psi: "Ψ", Omega: "Ω",
  times: "×", cdot: "·", div: "÷", pm: "±", mp: "∓", ast: "∗", star: "⋆",
  approx: "≈", sim: "∼", simeq: "≃", cong: "≅", equiv: "≡", propto: "∝",
  neq: "≠", ne: "≠", leq: "≤", le: "≤", geq: "≥", ge: "≥", ll: "≪", gg: "≫",
  infty: "∞", partial: "∂", nabla: "∇", sum: "∑", prod: "∏", int: "∫",
  in: "∈", notin: "∉", subset: "⊂", subseteq: "⊆", supset: "⊃",
  cup: "∪", cap: "∩", emptyset: "∅", varnothing: "∅",
  forall: "∀", exists: "∃", neg: "¬", land: "∧", lor: "∨",
  to: "→", rightarrow: "→", longrightarrow: "⟶", Rightarrow: "⇒",
  leftarrow: "←", Leftarrow: "⇐", leftrightarrow: "↔", Leftrightarrow: "⇔",
  mapsto: "↦", uparrow: "↑", downarrow: "↓",
  perp: "⊥", parallel: "∥", angle: "∠", degree: "°", circ: "∘", bullet: "•",
  ldots: "…", dots: "…", cdots: "⋯", vdots: "⋮", prime: "′",
  langle: "⟨", rangle: "⟩", lvert: "|", rvert: "|", lVert: "‖", rVert: "‖",
  mid: "∣", nmid: "∤", vert: "|", Vert: "‖",
  lceil: "⌈", rceil: "⌉", lfloor: "⌊", rfloor: "⌋",
  quad: " ", qquad: "  ", oplus: "⊕", otimes: "⊗", odot: "⊙",
  hbar: "ℏ", ell: "ℓ", Re: "ℜ", Im: "ℑ", aleph: "ℵ", surd: "√",
  dagger: "†", ddagger: "‡", top: "⊤", bot: "⊥", setminus: "∖",
  leftrightarrows: "⇄", rightleftarrows: "⇄", implies: "⇒", iff: "⇔",
};
const BLACKBOARD = { E: "𝔼", R: "ℝ", N: "ℕ", Z: "ℤ", Q: "ℚ", C: "ℂ", P: "ℙ", 1: "𝟙" };
const FUNCS = ["log", "ln", "exp", "min", "max", "sin", "cos", "tan", "det",
  "dim", "gcd", "lim", "sup", "inf", "arg", "argmax", "argmin", "mod", "bmod"];
const SUP = { ...chars("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"), ...chars("+-=()", "⁺⁻⁼⁽⁾"),
  ...chars("abcdefghijklmnoprstuvwxyz", "ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻ"),
  ...chars("ABDEGHIJKLMNOPRTUVW", "ᴬᴮᴰᴱᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾᴿᵀᵁⱽᵂ") };
const SUB = { ...chars("0123456789", "₀₁₂₃₄₅₆₇₈₉"), ...chars("+-=()", "₊₋₌₍₎"),
  ...chars("aehijklmnoprstuvx", "ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ") };

function chars(from, to) {
  const m = {};
  [...from].forEach((c, i) => { m[c] = [...to][i]; });
  return m;
}

/** Read the `{…}` starting at `i`; returns the inner text and the index after `}`. */
function readBrace(s, i) {
  let depth = 0;
  for (let j = i; j < s.length; j++) {
    if (s[j] === "\\") { j++; continue; }
    if (s[j] === "{") depth++;
    else if (s[j] === "}" && --depth === 0) return { inner: s.slice(i + 1, j), end: j + 1 };
  }
  return null;
}

/**
 * Rewrite every `\cmd{…}` (n args) via `fn`, leaving the rest untouched.
 * An unbraced argument counts too — `\bar S` and `\sqrt 2` are legal TeX and
 * the posts use both.
 */
function mapCommand(s, names, arity, fn) {
  let out = "", i = 0;
  while (i < s.length) {
    const m = s[i] === "\\" && /^\\([A-Za-z]+)\s*/.exec(s.slice(i));
    if (m && names.includes(m[1])) {
      const args = [];
      let j = i + m[0].length;
      while (args.length < arity) {
        while (s[j] === " ") j++;
        if (s[j] === "{") {
          const b = readBrace(s, j);
          if (!b) break;
          args.push(b.inner);
          j = b.end;
        } else if (s[j] === "\\") {
          const c = /^\\[A-Za-z]+/.exec(s.slice(j));
          if (!c) break;
          args.push(c[0]);
          j += c[0].length;
        } else if (s[j] && !/[\s^_}]/.test(s[j])) {
          args.push(s[j++]);
        } else break;
      }
      if (args.length === arity) { out += fn(m[1], ...args); i = j; continue; }
    }
    out += s[i++];
  }
  return out;
}

/** Superscript/subscript a token, or null when a character has no Unicode form. */
function script(text, table) {
  let out = "";
  for (const c of text) {
    if (!(c in table)) return null;
    out += table[c];
  }
  return out;
}

/**
 * Best-effort LaTeX → Unicode. Returns null when the formula needs real
 * typesetting (fractions of fractions, matrices, big operators with limits),
 * which is the caller's signal to fall back to an image or raw TeX.
 *
 * `lenient` keeps a formula alive when only its sub/superscripts are
 * unmappable — `R_{\text{win}}` becomes `R_win` instead of failing. That beats
 * an image for inline math, which Substack would break out into its own block.
 */
function texToUnicode(tex, depth = 0, lenient = false) {
  if (depth > 6) return null;
  let s = tex.trim();
  if (!s) return "";

  s = s.replace(/\\(?:left|right|displaystyle|limits|nonumber|,|;|:|!|\s)/g, (m) =>
    m === "\\," || m === "\\;" || m === "\\:" || m === "\\ " ? " " : "");

  // Structural commands, innermost first.
  s = mapCommand(s, ["text", "textrm", "textbf", "textit", "mathrm", "mathbf",
    "mathit", "mathsf", "mathtt", "mathcal", "mathscr", "boldsymbol",
    "operatorname", "bm"], 1, (_, a) => a);
  const sub = (a) => texToUnicode(a, depth + 1, lenient);
  s = mapCommand(s, ["mathbb"], 1, (_, a) => BLACKBOARD[a] ?? a);
  s = mapCommand(s, ["frac", "dfrac", "tfrac"], 2, (_, a, b) => {
    const [n, d] = [sub(a), sub(b)];
    if (n === null || d === null) return "\\FAIL";
    const wrap = (x) => (/^[\w.]+$/.test(x) ? x : `(${x})`);
    return `${wrap(n)}/${wrap(d)}`;
  });
  s = mapCommand(s, ["sqrt"], 1, (_, a) => {
    const inner = sub(a);
    if (inner === null) return "\\FAIL";
    return /^[\w.]+$/.test(inner) ? `√${inner}` : `√(${inner})`;
  });
  s = mapCommand(s, ["hat"], 1, (_, a) => `${sub(a) ?? "\\FAIL"}\u0302`);
  s = mapCommand(s, ["bar", "overline"], 1, (_, a) => `${sub(a) ?? "\\FAIL"}\u0304`);
  s = mapCommand(s, ["vec"], 1, (_, a) => `${sub(a) ?? "\\FAIL"}\u20D7`);
  s = mapCommand(s, ["tilde"], 1, (_, a) => `${sub(a) ?? "\\FAIL"}\u0303`);

  // Escaped literals and named symbols.
  s = s.replace(/\\([%$&#_{}])/g, "$1");
  s = s.replace(/\\([A-Za-z]+)/g, (m, name) => {
    if (name in SYMBOLS) return SYMBOLS[name];
    if (FUNCS.includes(name)) return name + " ";
    return m;
  });

  // Scripts, after the commands they might attach to are resolved.
  const scripts = (input, marker, table) => {
    let out = "", i = 0;
    while (i < input.length) {
      if (input[i] === marker) {
        let raw = null, next = i;
        if (input[i + 1] === "{") {
          const b = readBrace(input, i + 1);
          if (b) { raw = b.inner; next = b.end; }
        } else if (input[i + 1]) {
          raw = input[i + 1];
          next = i + 2;
        }
        if (raw !== null) {
          const inner = texToUnicode(raw, depth + 1, lenient);
          // A word-length subscript reads better spelled out: `R_loss` beside
          // `R_win` beats `Rₗₒₛₛ` beside `R_win`, which is what you get when
          // one word happens to have Unicode subscripts and its neighbour
          // doesn't. Single characters and digits still use the real glyphs.
          const spellOut = inner !== null && inner.length > 1 && /[A-Za-z]/.test(inner);
          const mapped = inner === null || spellOut ? null : script(inner, table);
          if (mapped === null) {
            if (inner === null) return null;
            if (!lenient && !spellOut) return null;
            // `S_max` reads fine bare; anything with an operator in it needs
            // the parens to stay unambiguous.
            const bare = /^[^\s()[\]+\-*/=,^_]{1,8}$/.test(inner);
            out += bare ? `${marker}${inner}` : `${marker}(${inner})`;
          } else {
            out += mapped;
          }
          i = next;
          continue;
        }
      }
      out += input[i++];
    }
    return out;
  };
  const sup = scripts(s, "^", SUP);
  if (sup === null) return null;
  s = scripts(sup, "_", SUB);
  if (s === null) return null;

  s = s.replace(/[{}]/g, "").replace(/ {2,}/g, " ").trim();
  return /\\/.test(s) ? null : s;
}

/* ─────────────────── math accuracy verification ─────────────────── */

/**
 * A Unicode rendering is only allowed to ship if it provably says the same
 * thing as the LaTeX. KaTeX — the same engine the site renders with — parses
 * the formula and emits MathML; the symbols in that MathML are the ground
 * truth. If our string doesn't carry exactly those symbols in exactly that
 * order, we don't guess: the formula goes out as a rendered image instead.
 */
const OP_CANON = {
  "−": "-", "‐": "-", "‑": "-", "–": "-", "—": "-",
  "⋅": "·", "∗": "*", "∼": "~", "′": "'", "⁄": "/",
  "≦": "≤", "≧": "≥", "ϵ": "ε", "ϑ": "θ", "ϕ": "φ", "ϖ": "π", "ϱ": "ρ", "ς": "σ",
  "𝑥": "x", "ℓ": "l",
};
/**
 * Structural glyphs either side may add for grouping, plus the spacing accents
 * KaTeX emits for `\bar`/`\hat` (we emit the combining form instead).
 */
const STRUCTURAL = /[\s(){}[\]|/\\^_,.\u00b7\u2032\u2016!\u00af\u02b0-\u02ff\u2223\u2225\u221a]/g;
const INVISIBLE = /[\u200b-\u200f\u2061-\u2064\u00a0\u2000-\u200a\u202f\u205f\u3000\ufeff]/g;

const SUP_BACK = invert(SUP);
const SUB_BACK = invert(SUB);
function invert(table) {
  const out = {};
  for (const [k, v] of Object.entries(table)) if (!(v in out)) out[v] = k;
  return out;
}

/**
 * Reduce a rendering to the bare sequence of content symbols it states.
 * NFKD folds blackboard letters to plain ones and sub/superscripts to their
 * base, so a Unicode rendering and KaTeX's MathML meet on the same ground.
 * Accents are dropped (notation, and the two sides spell them differently) but
 * the combining solidus is kept: without it a negated relation would canonicalise
 * to its positive form and could slip through unnoticed.
 */
function canonicalSymbols(s) {
  let out = "";
  for (const ch of String(s).normalize("NFKD")) {
    if (/[\u0300-\u0336\u0339-\u036f\u20d0-\u20ff]/.test(ch)) continue;
    const base = SUP_BACK[ch] ?? SUB_BACK[ch] ?? OP_CANON[ch] ?? ch;
    out += base;
  }
  return out.replace(INVISIBLE, "").replace(STRUCTURAL, "");
}

const decodeEntities = (s) =>
  s.replace(/&#x([0-9a-f]+);/gi, (_, h) => String.fromCodePoint(parseInt(h, 16)))
    .replace(/&#(\d+);/g, (_, d) => String.fromCodePoint(Number(d)))
    .replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"')
    .replace(/&amp;/g, "&").replace(/&nbsp;/g, " ");

/** Ground-truth symbols for a formula, or null when the LaTeX itself is invalid. */
function texSymbols(tex) {
  let mathml;
  try {
    mathml = katex.renderToString(tex, { output: "mathml", throwOnError: true, strict: false });
  } catch {
    return null;
  }
  const body = mathml.replace(/<annotation[\s\S]*?<\/annotation>/g, "");
  return canonicalSymbols(decodeEntities(body.replace(/<[^>]+>/g, "")));
}

/* ─────────────────────── math → image ─────────────────────── */

function mathImage(tex, { mathDpi, mathBg }, display, sink) {
  const prefix = `\\dpi{${mathDpi}}` + (mathBg === "white" ? "\\bg{white}" : "");
  const src = `https://latex.codecogs.com/png.image?${encodeURIComponent(prefix + tex)}`;
  sink?.push(src); // checked for a 200 later — the renderer 400s on bad LaTeX
  // Height is expressed in em so the image tracks the reader's font size, and
  // dpi 200 keeps it sharp on retina at that size.
  return display
    ? `<img src="${src}" alt="${escapeAttr(tex)}" style="max-width:100%;height:auto" />`
    : `<img src="${src}" alt="${escapeAttr(tex)}" style="height:1.15em;vertical-align:-0.2em" />`;
}

/* ─────────────────────── markdown transforms ─────────────────────── */

const CALLOUT_LABEL = {
  note: "Note", tip: "Tip", important: "Important", warning: "Warning",
  caution: "Caution", info: "Info", success: "Success", error: "Error",
  tldr: "TL;DR",
};

/** `> [!important]` → a plain blockquote led by a bold label. */
function convertCallouts(md) {
  return md.replace(
    /^([ \t]*)>\s*\[!(\w+)\]([^\n]*)\n((?:[ \t]*>[^\n]*\n?)*)/gm,
    (_, indent, type, rest, body) => {
      const label = CALLOUT_LABEL[type.toLowerCase()] ?? type;
      const title = rest.trim() || label;
      // Posts often open the body with **TL;DR** already — don't say it twice.
      const dupe = /^[ \t]*>\s*\*\*(TL;DR|TLDR)/i.test(body) && /^(important|tldr|note)$/i.test(type);
      const head = dupe ? "" : `${indent}> **${title}**\n${indent}>\n`;
      return head + body;
    },
  );
}

/** Resolve a repo image ref to a public URL the Substack fetcher can reach. */
function imageUrl(src, site, manifest) {
  if (/^https?:\/\//.test(src)) return src;
  let rel = src.startsWith("/") ? src : `/${src}`;
  if (manifest[rel]) rel = rel.replace(/\.(png|jpe?g)$/i, ".webp");
  else if (!fs.existsSync(path.join(PUBLIC, rel))) {
    const webp = rel.replace(/\.(png|jpe?g)$/i, ".webp");
    if (fs.existsSync(path.join(PUBLIC, webp))) rel = webp;
  }
  return site + rel;
}

/**
 * The posts write `--`; a newsletter should show real dashes. Both rules need a
 * word character on the tight side, so `<!--` and `-->` come through untouched.
 */
const smartDashes = (s) =>
  String(s).replace(/(\s)--(\s)/g, "$1—$2").replace(/(?<=[A-Za-z0-9%)])--(?=[A-Za-z0-9$(\\])/g, "–");

const escapeAttr = (s) =>
  String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

/* ─────────────────────── the conversion ─────────────────────── */

function convert(file, opts, manifest) {
  const slug = path.basename(file, ".md");
  const raw = fs.readFileSync(file, "utf8");
  const { data: fm, content } = matter(raw);
  if (opts.smartDashes) {
    // Title and subtitle go into Substack's own fields — same typography.
    if (fm.title) fm.title = smartDashes(fm.title);
    if (fm.description) fm.description = smartDashes(fm.description);
  }
  const stats = {
    images: [], animations: [], tables: 0, mathImageUrls: [],
    internalLinks: 0, linkedToSubstack: 0, needsPublishing: new Map(), deadLinks: new Set(),
    tableSources: [], tableImages: [],
    mathUnicode: 0, mathImages: 0, mathDisplayRaw: 0,
    mathRaw: [], mathInvalid: [], mathUnfaithful: [],
  };

  let md = content;

  // 1. Animated figures: inline SVG + CSS keyframes never survives a paste.
  md = md.replace(/<figure[^>]*class="[^"]*blog-anim[^"]*"[\s\S]*?<\/figure>/g, (block) => {
    const caption =
      /<figcaption[^>]*>([\s\S]*?)<\/figcaption>/.exec(block)?.[1] ??
      /aria-label="([^"]*)"/.exec(block)?.[1] ??
      /<title>([\s\S]*?)<\/title>/.exec(block)?.[1] ??
      "animated figure";
    stats.animations.push({ caption: caption.replace(/<[^>]+>/g, "").trim(), svg: block });
    return "";
  });

  // 2. Mask code, then math — order matters, `$` inside code is not math.
  const code = maskCode(md);
  md = code.md;
  const math = maskMath(md);
  md = math.md;
  // A formula can contain a `code span` (`$\lceil`d_model`/16\rceil$`); that
  // span was masked first, so put it back before the TeX is interpreted.
  for (const entry of math.store) {
    entry.tex = unmask(entry.tex, "CODE", code.store).replace(/`/g, "");
  }

  // 3. Text-level rewrites (safe now that code and math are out of the way).
  md = convertCallouts(md);
  const images = [];
  md = md.replace(/!\[([^\]]*)\]\(([^)\s]+)(?:\s+"[^"]*")?\)/g, (_, alt, src) => {
    const url = imageUrl(src, opts.site, manifest);
    stats.images.push(url);
    // The draft path uploads the file itself, so keep the local path too — and
    // keep it **repo-relative**. python-substack's markdown reader strips the
    // leading `/` from an absolute path, so `os.path.exists()` then fails, no
    // upload happens, and the mangled path is written into the draft as the
    // image src (which renders as nothing). The child process runs with cwd set
    // to the repo root so a relative path resolves.
    const local = url.startsWith(opts.site)
      ? path.relative(ROOT, path.join(PUBLIC, url.slice(opts.site.length)))
      : null;
    const text = opts.figureAlt ? alt : "";
    images.push({ alt: opts.smartDashes ? smartDashes(text) : text, url, local });
    return tok("IMG", images.length - 1);
  });
  // Site-relative links are cross-references to other posts. The newsletter
  // shouldn't point readers back at the site, so the anchor text stays and the
  // link goes. Links to other domains are left alone.
  md = md.replace(/\[([^\]\n]+)\]\((\/(?!\/)[^)\s]*)\)/g, (_, text, href) => {
    stats.internalLinks++;
    if (opts.internalLinks === "absolute") return `[${text}](${opts.site}${href})`;
    if (opts.internalLinks === "substack" && opts.archive) {
      const slug = href.replace(/[#?].*$/, "").replace(/\/$/, "").split("/").pop();
      const target = substackTarget(slug, opts.archive);
      if (target.state === "published") {
        stats.linkedToSubstack++;
        return `[${text}](${target.url})`;
      }
      // Everything in this batch gets `--slug <repo slug>`, so its Substack URL
      // is known before the draft exists. That's what makes a whole series
      // cross-link on the first pass instead of needing a second export.
      if (opts.batch?.has(slug)) {
        stats.linkedToSubstack++;
        return `[${text}](${opts.substack}/p/${slug})`;
      }
      // Not on Substack yet — keep the sentence readable and report the gap.
      if (target.state === "unpublished") stats.needsPublishing.set(slug, target.title);
      else stats.deadLinks.add(href);
    }
    return text;
  });
  if (opts.smartDashes) md = smartDashes(md);

  // 4. Code goes back in as markdown so remark renders real <pre><code>.
  //    Currency inside a code span is often written `\$96` out of habit, and a
  //    code span has no math parser to escape from — but leave `\$VAR` alone,
  //    that backslash is load-bearing in a shell snippet.
  // 2b. Tables. Substack has no table node at all — its editor has none and its
  //     document format has none — so a markdown table arrives as one paragraph
  //     of literal pipes. Pull each one out and ship it as an image instead.
  if (opts.tables === "image") {
    md = md.replace(
      /^\|[^\n]*\|[ \t]*\n\|[ :|\-]+\|[ \t]*\n(?:\|[^\n]*\|[ \t]*\n?)+/gm,
      (table) => {
        stats.tableSources.push(table);
        return `${tok("TBL", stats.tableSources.length - 1)}\n`;
      },
    );
  }

  const codeRestored = code.store.map((c) =>
    c.startsWith("`") && !c.startsWith("```") ? c.replace(/\\\$(?=\d)/g, "$") : c);
  const mdMasked = md; // code still masked — the Markdown path needs this
  md = unmask(md, "CODE", codeRestored);

  // 5. Markdown → HTML. `\$` was escaped only to keep the site's math parser off
  //    prose currency; in HTML it should read as a plain dollar sign.
  let html = String(
    unified()
      .use(remarkParse)
      .use(remarkGfm)
      .use(remarkRehype, { allowDangerousHtml: true })
      .use(rehypeStringify, { allowDangerousHtml: true })
      .processSync(md.replace(/\\\$/g, "$")),
  );

  // 6. Math back in, now as Unicode text or a rendered image.
  html = html.replace(
    new RegExp(`(<p>\\s*)?ZZMATH(\\d+)ZZ(\\s*</p>)?`, "g"),
    (whole, open, i, close) => {
      const { tex, display } = math.store[Number(i)];
      const block = Boolean(open && close);
      const mode = display ? opts.displayMath : opts.inlineMath;
      const raw = display ? `$$${tex}$$` : `$${tex}$`;
      const asText = (t) => (block ? `<p>${escapeAttr(t)}</p>` : escapeAttr(t));

      // Ground truth first: if KaTeX can't parse it, the formula is broken in
      // the source too — flag it rather than hand a bad string to a renderer.
      // Whatever is decided here is recorded on the entry so the Markdown
      // emitter below reaches the same verdict instead of re-deciding.
      const entry = math.store[Number(i)];
      const truth = opts.verifyMath ? texSymbols(tex) : undefined;
      if (truth === null) {
        stats.mathInvalid.push(raw);
        entry.resolved = { kind: "raw" };
        return asText(raw);
      }

      if (mode === "raw") {
        stats.mathRaw.push(raw);
        if (display) stats.mathDisplayRaw++;
        entry.resolved = { kind: "raw" };
        return asText(raw);
      }
      if (mode !== "image") {
        // Inline math is text-first: an <img> mid-sentence gets promoted to its
        // own block by Substack's editor and shatters the paragraph.
        const uni = texToUnicode(tex) ?? (display ? null : texToUnicode(tex, 0, true));
        // Ship the Unicode only when it states exactly what the LaTeX states.
        const faithful = uni !== null && (truth === undefined || canonicalSymbols(uni) === truth);
        if (faithful) {
          stats.mathUnicode++;
          entry.resolved = { kind: "text", value: uni };
          return asText(uni);
        }
        if (uni !== null) stats.mathUnfaithful.push({ tex, uni });
        if (mode === "unicode") {
          stats.mathRaw.push(raw);
          if (display) stats.mathDisplayRaw++;
          entry.resolved = { kind: "raw" };
          return asText(raw);
        }
      }
      stats.mathImages++;
      entry.resolved = { kind: "image" };
      const img = mathImage(tex, opts, display || block, stats.mathImageUrls);
      return block ? `<p style="text-align:center">${img}</p>` : img;
    },
  );

  // 7. Images.
  html = html.replace(tokRe("IMG"), (_, i) => {
    const img = images[Number(i)];
    const alt = img.alt ? ` alt="${escapeAttr(img.alt)}"` : ' alt=""';
    return `<img src="${img.url}"${alt} style="max-width:100%;height:auto" />`;
  });

  // 8. Heading levels Substack's editor actually has.
  if (!opts.deepHeadings) {
    html = html.replace(/<h1>([\s\S]*?)<\/h1>/g, "<h2>$1</h2>");
    html = html.replace(/<h([456])>([\s\S]*?)<\/h\1>/g, "<p><strong>$2</strong></p>");
  }

  // 9. Tables.
  html = html.replace(new RegExp(`(<p>\\s*)?ZZTBL(\\d+)ZZ(\\s*</p>)?`, "g"), (_, __, i) =>
    String(unified().use(remarkParse).use(remarkGfm).use(remarkRehype).use(rehypeStringify)
      .processSync(stats.tableSources[Number(i)])));

  stats.tables = (html.match(/<table>/g) || []).length; // the HTML path re-renders extracted tables, so this already counts them
  if (opts.tables === "list") html = tablesToLists(html);

  // 10. The same content as Markdown, for the draft API. Nothing is degraded
  //     here: python-substack turns `$…$` into Substack's own math nodes and
  //     uploads the image files, so the LaTeX stays LaTeX and the images don't
  //     depend on the site being deployed. `\$` stays escaped — Pandoc rules
  //     would otherwise read prose currency as math.
  // Escape every literal dollar that survived, *before* the real math goes in.
  // python-substack applies plain Pandoc rules, which are looser than the site's
  // — the site skips digit-led spans like `$1/(1+R)$` so they reach here as
  // prose, and python-substack would turn them into an inline `latex` node.
  // Substack's editor has no such node and refuses to open the post at all
  // ("Something has gone wrong"), with nothing in the API to show why.
  //     Escape on the *masked* text so code spans keep their bare `$`: a code
  //     span is never parsed as math, and markdown-it prints the backslash
  //     verbatim inside one — which is how `\$300` ended up visible in a draft.
  let markdown = mdMasked.replace(/(?<!\\)\$/g, "\\$");
  markdown = unmask(markdown, "CODE", codeRestored);
  markdown = markdown.replace(tokRe("MATH"), (_, i) => {
    const { tex, display, resolved } = math.store[Number(i)];
    // A converted formula can itself contain currency (`$100` out of `\$100`),
    // and it lands after the escaping pass — so escape it here too.
    if (resolved?.kind === "text") return resolved.value.replace(/\$/g, "\\$");
    if (resolved?.kind === "image") return `![](${mathImage(tex, opts, true).match(/src="([^"]+)"/)[1]})`;
    // Only display math is handed over as LaTeX — a block becomes `latex_block`,
    // which the editor does understand. Inline stays literal for the same reason.
    return display ? `$$\n${tex}\n$$` : `\\$${tex}\\$`;
  });
  markdown = markdown.replace(tokRe("IMG"), (_, i) => {
    const img = images[Number(i)];
    const alt = img.alt.replace(/[[\]]/g, "").replace(/\$/g, "\\$");
    return `![${alt}](${img.local ?? img.url})`;
  });
  markdown = markdown.replace(tokRe("TBL"), (_, i) => {
    const rel = path.join(path.relative(ROOT, opts.out), "tables", `${slug}-table-${Number(i) + 1}.webp`);
    stats.tableImages.push({ index: Number(i), file: path.join(ROOT, rel) });
    return `![](${rel})`;
  });
  if (!opts.deepHeadings) markdown = markdown.replace(/^#{4,6} +(.*)$/gm, "**$1**");

  // Guard: any bare `$` left outside a `$$` block is a formula waiting to be
  // mis-parsed into a node the editor can't open. Count it and say so.
  // Code spans are exempt — a `$` in one is inert, and escaping it would show
  // the backslash. Everything else must be escaped or inside a `$$` block.
  stats.strayDollars = markdown
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/(`+)(?:(?!\1)[\s\S])+?\1/g, " ")
    .split("\n")
    .filter((line) => line !== "$$")
    .join("\n")
    .match(/(?<![\\$])\$(?!\$)/g)?.length ?? 0;

  // A surviving placeholder means a mask/restore pair went wrong — never ship it.
  stats.leaks = (html.match(/ZZ(?:CODE|MATH)\d+ZZ/g) || []).length;

  // Unrendered LaTeX still wearing its dollars. Everything we deliberately left
  // verbatim is accounted for, so whatever is left over is an unbalanced `$` in
  // the source — the site mis-renders it the same way.
  let scan = html.replace(/<(pre|code)\b[\s\S]*?<\/\1>/g, " ");
  for (const t of new Set([...stats.mathInvalid, ...stats.mathRaw])) {
    scan = scan.split(escapeAttr(t)).join(" ");
  }
  stats.strayTex = scan.match(/\$[^$<]{0,90}\\[^$<]{0,90}\$/g) || [];

  if (opts.canonical) {
    const url = `${opts.site}/blog/${fm.category ?? ""}/${slugifySub(fm.subcategory)}/${slug}`
      .replace(/\/+/g, "/").replace(":/", "://");
    html += `\n<hr />\n<p><em>Originally published at <a href="${url}">${url}</a>.</em></p>\n`;
  }

  return { slug, fm, html: html.trim(), markdown: markdown.trim(), stats };
}

const slugifySub = (s) => String(s ?? "").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");

/** Substack has no table node; a bullet per row survives the paste intact. */
function tablesToLists(html) {
  return html.replace(/<table>[\s\S]*?<\/table>/g, (table) => {
    const rows = [...table.matchAll(/<tr>([\s\S]*?)<\/tr>/g)].map((m) =>
      [...m[1].matchAll(/<t[hd][^>]*>([\s\S]*?)<\/t[hd]>/g)].map((c) => c[1].trim()),
    );
    if (!rows.length) return table;
    const [head, ...body] = rows;
    const items = body.map((cells) => {
      const first = cells[0] ?? "";
      const rest = cells.slice(1)
        .map((c, i) => (head[i + 1] ? `${head[i + 1]}: ${c}` : c))
        .filter((c) => c && !/^:\s*$/.test(c))
        .join(" · ");
      return `<li><strong>${first}</strong>${rest ? ` — ${rest}` : ""}</li>`;
    });
    return `<ul>\n${items.join("\n")}\n</ul>`;
  });
}

/* ─────────────────────── preview page ─────────────────────── */

function previewPage({ slug, fm, html, stats }, opts, coverUrl) {
  const notes = [];
  if (stats.mathInvalid.length)
    notes.push(`<b>${stats.mathInvalid.length} formula(s) are invalid LaTeX</b> and render as an error on the site too — they are left verbatim here: ${stats.mathInvalid.slice(0, 3).map((t) => `<code>${escapeAttr(clip(t, 40))}</code>`).join(", ")}. Fix the source post.`);
  if (stats.animations.length)
    notes.push(`${stats.animations.length} animated figure(s) dropped — screenshot the saved .svg files if you want them.`);
  if (stats.mathRaw.length)
    notes.push(`${stats.mathRaw.length} formula(s) left as raw LaTeX — search the pasted draft for "$".`);
  if (stats.strayTex.length)
    notes.push(`${stats.strayTex.length} LaTeX span(s) never closed their <code>$</code> in the source, so they stay literal here — same as on the site.`);
  if (stats.mathUnicode || stats.mathImages)
    notes.push(`Math: ${stats.mathUnicode} formula(s) as text, each cross-checked against KaTeX; ${stats.mathImages} rendered from the exact LaTeX.`);
  if (stats.needsPublishing.size)
    notes.push(`<b>${stats.needsPublishing.size} referenced post(s) are not on Substack yet</b> — their links are plain text here. Publish them first and re-run to turn them into links: ${[...stats.needsPublishing.keys()].slice(0, 4).map((s) => `<code>${s}</code>`).join(", ")}${stats.needsPublishing.size > 4 ? " …" : ""}`);
  if (stats.linkedToSubstack)
    notes.push(`${stats.linkedToSubstack} cross-post link(s) point at the matching Substack post.`);
  if (stats.tables && opts.tables === "keep")
    notes.push(`${stats.tables} table(s) pasted as HTML — if Substack flattens them, re-run with --tables list.`);
  notes.push("Images upload themselves when you paste — wait for every spinner before publishing.");

  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Substack · ${escapeAttr(fm.title ?? slug)}</title>
<style>
  :root { color-scheme: light }
  body { margin:0; background:#f4f4f2; color:#1a1a1a;
         font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif }
  header { position:sticky; top:0; z-index:5; background:#fff; border-bottom:1px solid #e2e2df;
           padding:14px 20px; box-shadow:0 1px 6px rgba(0,0,0,.05) }
  .bar { max-width:800px; margin:0 auto; display:flex; gap:10px; align-items:flex-start; flex-wrap:wrap }
  .meta { flex:1 1 340px; min-width:260px; font-size:13px; line-height:1.5 }
  .meta b { display:inline-block; width:64px; color:#888; font-weight:500 }
  .meta div { margin-bottom:2px }
  button { font:inherit; font-size:13px; padding:8px 14px; border-radius:6px; border:1px solid #d0d0cc;
           background:#fff; cursor:pointer }
  button:hover { background:#f6f6f4 }
  button.primary { background:#ff6719; border-color:#ff6719; color:#fff; font-weight:600 }
  button.primary:hover { filter:brightness(1.06) }
  button.done { background:#127a3d; border-color:#127a3d; color:#fff }
  .notes { max-width:800px; margin:16px auto 0; padding:0 20px; font-size:13px; color:#6a6a68 }
  .notes li { margin:3px 0 }
  main { max-width:800px; margin:0 auto; padding:28px 20px 80px }
  article { background:#fff; border:1px solid #e6e6e3; border-radius:8px; padding:44px 52px;
            font-family:Georgia,"Times New Roman",serif; font-size:20px; line-height:1.62; color:#1a1a1a }
  article h2 { font-size:27px; line-height:1.25; margin:2em 0 .5em; font-family:-apple-system,sans-serif }
  article h3 { font-size:22px; margin:1.6em 0 .4em; font-family:-apple-system,sans-serif }
  article p { margin:0 0 1.1em }
  article img { max-width:100%; height:auto; display:inline-block }
  article blockquote { margin:1.4em 0; padding:2px 0 2px 20px; border-left:3px solid #ddd; color:#333 }
  article pre { background:#f7f7f5; border:1px solid #e6e6e3; border-radius:6px; padding:14px 16px;
                overflow-x:auto; font-size:14px; line-height:1.5 }
  article code { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.85em }
  article table { border-collapse:collapse; width:100%; font-size:15px; margin:1.4em 0; display:block; overflow-x:auto }
  article th, article td { border:1px solid #e0e0dd; padding:7px 10px; text-align:left }
  article hr { border:0; border-top:1px solid #e0e0dd; margin:2.2em 0 }
</style></head>
<body>
<header><div class="bar">
  <div class="meta">
    <div><b>Title</b><span id="t">${escapeAttr(fm.title ?? slug)}</span></div>
    <div><b>Subtitle</b><span id="s">${escapeAttr(fm.description ?? "")}</span></div>
    ${coverUrl ? `<div><b>Cover</b><a href="${coverUrl}">${coverUrl.split("/").pop()}</a></div>` : ""}
    ${fm.tags?.length ? `<div><b>Tags</b>${escapeAttr(fm.tags.join(", "))}</div>` : ""}
  </div>
  <button onclick="copyText(document.getElementById('t').textContent,this)">Copy title</button>
  <button onclick="copyText(document.getElementById('s').textContent,this)">Copy subtitle</button>
  <button class="primary" onclick="copyBody(this)">Copy post body</button>
</div>
<ul class="notes">${notes.map((n) => `<li>${n}</li>`).join("")}</ul>
</header>
<main><article id="post">
${html}
</article></main>
<script>
function flash(btn, msg) {
  const old = btn.textContent;
  btn.textContent = msg; btn.classList.add('done');
  setTimeout(() => { btn.textContent = old; btn.classList.remove('done'); }, 1600);
}
async function copyText(text, btn) {
  try { await navigator.clipboard.writeText(text); flash(btn, 'Copied'); }
  catch { flash(btn, 'Press ⌘C'); }
}
async function copyBody(btn) {
  const el = document.getElementById('post');
  // Rich HTML on the clipboard is what makes Substack keep the formatting.
  try {
    await navigator.clipboard.write([new ClipboardItem({
      'text/html': new Blob([el.innerHTML], { type: 'text/html' }),
      'text/plain': new Blob([el.innerText], { type: 'text/plain' }),
    })]);
    flash(btn, 'Copied — paste into Substack');
    return;
  } catch (e) {}
  // Fallback for browsers that block the async clipboard on file:// URLs.
  const range = document.createRange();
  range.selectNodeContents(el);
  const sel = getSelection(); sel.removeAllRanges(); sel.addRange(range);
  flash(btn, document.execCommand('copy') ? 'Copied — paste into Substack' : 'Selected — press ⌘C');
}
</script>
</body></html>
`;
}

/* ─────────────────────── the Substack archive ─────────────────────── */

/** Slug → post file, for every post in the repo. Built once, no file reads. */
let SLUG_INDEX;
function slugIndex() {
  if (!SLUG_INDEX) {
    SLUG_INDEX = new Map();
    for (const file of allPosts()) SLUG_INDEX.set(path.basename(file, ".md"), file);
  }
  return SLUG_INDEX;
}

/** Titles are matched loosely — Substack keeps the text, not the punctuation. */
const normTitle = (s) =>
  String(s).toLowerCase().normalize("NFKD").replace(/[̀-ͯ]/g, "")
    .replace(/[^a-z0-9]+/g, " ").trim();

/**
 * Everything the publication has posted, keyed by both normalised title and
 * slug. Cached for an hour — the archive changes only when you publish.
 */
async function loadArchive(opts) {
  const cache = path.join(opts.out, "archive.json");
  if (!opts.refreshArchive && fs.existsSync(cache)) {
    const age = Date.now() - fs.statSync(cache).mtimeMs;
    if (age < 60 * 60 * 1000) return indexArchive(JSON.parse(fs.readFileSync(cache, "utf8")));
  }

  const posts = [];
  const seen = new Set();
  try {
    // The archive endpoint returns a short first page — fewer rows than `limit`
    // even though more posts follow — so a page shorter than the limit is not
    // the end of the archive. Walk it in overlapping windows and stop only on a
    // genuinely empty page, deduping by slug.
    for (let offset = 0; offset < 4000; offset += 20) {
      const url = `${opts.substack}/api/v1/archive?sort=new&limit=50&offset=${offset}`;
      const res = await fetch(url, { signal: AbortSignal.timeout(20_000) });
      if (!res.ok) throw new Error(`${res.status} from ${url}`);
      const page = await res.json();
      if (page.length === 0) break;
      for (const p of page) {
        if (!p.slug || seen.has(p.slug)) continue;
        seen.add(p.slug);
        posts.push({ slug: p.slug, title: p.title, url: p.canonical_url });
      }
    }
  } catch (e) {
    return { error: String(e.message ?? e), byTitle: new Map(), bySlug: new Map(), size: 0 };
  }
  fs.mkdirSync(opts.out, { recursive: true });
  fs.writeFileSync(cache, JSON.stringify(posts, null, 2));
  return indexArchive(posts);
}

function indexArchive(posts) {
  const byTitle = new Map(), bySlug = new Map();
  for (const p of posts) {
    if (p.title) byTitle.set(normTitle(p.title), p);
    if (p.slug) bySlug.set(p.slug, p);
  }
  return { byTitle, bySlug, size: posts.length, error: null };
}

/**
 * Where a cross-referenced post lives on Substack, or why it can't be linked.
 * Matching is by title first (Substack derives its slug from the title, and the
 * author may edit it), then by slug as a fallback.
 */
function substackTarget(slug, archive) {
  const file = slugIndex().get(slug);
  if (!file) return { state: "missing" };
  const title = matter(fs.readFileSync(file, "utf8")).data.title ?? slug;
  const hit = archive.byTitle.get(normTitle(title)) ?? archive.bySlug.get(slug);
  return hit ? { state: "published", url: hit.url, title } : { state: "unpublished", title };
}

/* ─────────────────────── image reachability ─────────────────────── */

/**
 * Substack pulls each image from its URL at paste time, so an image that isn't
 * live yet silently never arrives. A freshly-pushed post is the usual cause —
 * the site is ISR + a 15–30 min build lag behind the commit.
 */
async function checkImages(urls) {
  const unique = [...new Set(urls)];
  const bad = [];
  let offline = false;
  const queue = unique.slice();
  const worker = async () => {
    for (let url = queue.pop(); url; url = queue.pop()) {
      try {
        const res = await fetch(url, { method: "HEAD", signal: AbortSignal.timeout(10_000) });
        if (!res.ok) bad.push(`${res.status} ${url}`);
      } catch (e) {
        if (/fetch failed|ENOTFOUND|ETIMEDOUT/i.test(String(e))) offline = true;
        else bad.push(`error ${url}`);
      }
    }
  };
  await Promise.all(Array.from({ length: Math.min(8, unique.length) }, worker));
  return { bad, offline, total: unique.length };
}

/* ─────────────────────── draft creation ─────────────────────── */

/**
 * Hand the Markdown to python-substack, which creates an **unpublished draft**
 * and nothing else — `drafts create` never sends, schedules or publishes. That
 * boundary is the whole point: a wrong run costs you a draft to delete, not an
 * email to your list. Authentication is the CLI's business (its own .env or
 * `--cookies`); this script never reads or passes a credential.
 */
// Substack rejects longer subtitles server-side; 272 was refused, this passes.
const SUBTITLE_MAX = 200;

/** Cut at a sentence if one lands near the limit, otherwise at a word. */
function trimSubtitle(text) {
  const s = String(text).trim();
  if (s.length <= SUBTITLE_MAX) return s;
  const head = s.slice(0, SUBTITLE_MAX);
  const sentence = Math.max(head.lastIndexOf(". "), head.lastIndexOf("? "), head.lastIndexOf("! "));
  if (sentence > SUBTITLE_MAX * 0.5) return head.slice(0, sentence + 1);
  return `${head.slice(0, head.lastIndexOf(" "))}…`;
}

function substackEnv() {
  // The CLI reads its cookie from the environment, not from ./.env, so forward
  // it. Only the cookie keys — EMAIL/PASSWORD are deliberately not passed on,
  // and nothing here is ever printed or logged.
  const envFile = path.join(ROOT, ".env");
  const extra = {};
  if (fs.existsSync(envFile)) {
    for (const line of fs.readFileSync(envFile, "utf8").split("\n")) {
      const m = /^(COOKIES_STRING|COOKIES_PATH|PUBLICATION_URL)=(.*)$/.exec(line.trim());
      if (m && !process.env[m[1]]) extra[m[1]] = m[2].replace(/^(["'])(.*)\1$/, "$2");
    }
  }
  return { ...process.env, ...extra };
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const isRateLimit = (msg) => /too many requests|rate.?limit|\b429\b/i.test(String(msg));

/**
 * A local record of what has been pushed, because the remote list can't be
 * trusted for this: `drafts list` caps at 25 and ignores `--offset`, so past
 * the 25th draft there is no way to ask Substack what already exists.
 */
const LEDGER = path.join(ROOT, ".substack-created.json");
const loadLedger = () => {
  try { return JSON.parse(fs.readFileSync(LEDGER, "utf8")); } catch { return {}; }
};
const saveLedger = (l) => fs.writeFileSync(LEDGER, `${JSON.stringify(l, null, 2)}\n`);

/** The newest drafts Substack will admit to (its list tops out at 25). */
function recentDrafts(opts) {
  try {
    const out = execFileSync("substack",
      ["--json", "--publication-url", opts.substack, "drafts", "list", "--limit", "25"],
      { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"], env: substackEnv(), timeout: 120_000 });
    return JSON.parse(out).drafts?.posts ?? [];
  } catch {
    return [];
  }
}

/**
 * Why there is no retry here: a rate-limited create is **not** atomic. The
 * images upload, the draft is created, and only then does the one-call-per-tag
 * loop trip the limit — so the CLI reports failure for a draft that exists.
 * Retrying produced duplicates (measured: 3 of them). Instead a 429 stops the
 * batch, and whatever did land is written to the ledger so the resumed run
 * skips it.
 */
/**
 * Read a draft back and check the two things that are invisible from this side:
 * an inline `latex` node (Substack's editor refuses to open the post, with only
 * a generic modal to show for it) and an image src that never got uploaded.
 * Run on the first draft of a batch — one call to catch a systemic break before
 * it repeats thirty times.
 */
function verifyDraft(id, opts) {
  try {
    const out = execFileSync("substack",
      ["--json", "--publication-url", opts.substack, "drafts", "get", String(id)],
      { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"], env: substackEnv(),
        timeout: 120_000, maxBuffer: 32 * 1024 * 1024 });
    const body = JSON.parse(JSON.parse(out).draft.draft_body);
    let inlineLatex = 0, images = 0, unuploaded = 0;
    (function walk(n) {
      if (Array.isArray(n)) return n.forEach(walk);
      if (!n || typeof n !== "object") return;
      if (n.type === "latex") inlineLatex++;
      if (n.type === "image2") {
        images++;
        if (!String(n.attrs?.src).startsWith("http")) unuploaded++;
      }
      if (n.content) walk(n.content);
    })(body.content);
    return { inlineLatex, images, unuploaded, nodes: body.content.length };
  } catch {
    return null;
  }
}

function claimAfterRateLimit(slug, fm, opts) {
  const match = recentDrafts(opts).find((p) => p.slug === slug || p.draft_title === fm.title);
  return match ? { id: match.id, tagsIncomplete: true } : null;
}

function createDraft(mdFile, slug, fm, opts) {
  const args = ["--json"];
  if (opts.cookies) args.push("--cookies", opts.cookies);
  args.push("--publication-url", opts.substack, "drafts", "create", mdFile);
  // Pin the Substack slug to the repo slug so cross-links resolve by slug, not
  // just by title — Substack would otherwise derive its own from the title.
  if (opts.slug !== false) args.push("--slug", slug);
  if (fm.title) args.push("--title", fm.title);
  // Substack rejects a long subtitle outright ("Subtitle is too long"); these
  // descriptions are written for the site's meta tag and run past 400 chars.
  if (fm.description) args.push("--subtitle", trimSubtitle(fm.description));
  // Every frontmatter tag, not a sample — the library adds them one API call
  // each after the draft exists, so the create response reports what stuck.
  const tags = opts.tags === false
    ? []
    : (fm.tags ?? []).map((t) => String(t).trim()).filter(Boolean);
  for (const tag of tags) args.push("--tag", tag);

  try {
    const out = execFileSync("substack", args, {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"],
      env: substackEnv(),
      cwd: ROOT, // image paths in the Markdown are relative to the repo root
      timeout: 600_000,
    });
    const parsed = JSON.parse(out);
    const added = parsed.tags?.tags_added?.length ?? (Array.isArray(parsed.tags) ? parsed.tags.length : 0);
    return { ok: true, id: parsed.draft_id ?? parsed.draft?.id, tagsAsked: tags.length, tagsAdded: added };
  } catch (e) {
    if (e.code === "ENOENT") return { ok: false, missing: true };
    // The CLI reports failures as a JSON envelope on stdout; pull the message
    // out of it so the one useful line isn't buried in 100KB of payload.
    const raw = (e.stderr || e.stdout || e.message).toString().trim();
    let msg = raw;
    try {
      const j = JSON.parse(raw);
      msg = j.error?.message ?? j.error ?? j.message ?? j.detail ?? raw;
    } catch { /* not JSON */ }
    return { ok: false, error: String(msg).replace(/\s+/g, " ").trim(), status: e.status };
  }
}

/* ─────────────────────── clipboard (macOS) ─────────────────────── */

function copyHtmlToClipboard(html) {
  if (process.platform !== "darwin") return false;
  const hex = Buffer.from(html, "utf8").toString("hex");
  const script = path.join(os.tmpdir(), `substack-clip-${process.pid}.applescript`);
  fs.writeFileSync(script, `set the clipboard to «data HTML${hex}»\n`);
  try {
    execFileSync("osascript", [script], { stdio: "ignore" });
    return true;
  } catch {
    return false;
  } finally {
    fs.rmSync(script, { force: true });
  }
}

/* ─────────────────────── cover images ─────────────────────── */

/**
 * Second pass over the pushed drafts to attach cover images. It has to be
 * separate: `substack drafts create` has no cover option — the package has no
 * notion of covers at all — so this drives the two low-level API calls through
 * `substack-cover.py`. Covers go over as public URLs, which means the site must
 * be deployed for the run to find them.
 */
async function setCovers(opts) {
  const ledger = loadLedger();
  const items = [];
  const missing = [];
  for (const [slug, rec] of Object.entries(ledger)) {
    const item = { slug, id: rec.id };
    if (!rec.cover) {
      const rel = [`/imgs/blogs/${slug}-1.cover.webp`, `/imgs/blogs/${slug}-1.webp`]
        .find((p) => fs.existsSync(path.join(PUBLIC, p)));
      if (rel) item.cover = opts.site + rel;
      else missing.push(slug);
    }
    // A draft whose tag loop was cut short by a rate limit gets the whole set
    // re-applied; Substack ignores the ones it already has. Tags cost one call
    // each and are what exhausts the quota on a long run — `--no-tags` here
    // limits the pass to covers, which are the part a reader actually sees.
    if (rec.tagsIncomplete && opts.tags !== false) {
      const file = slugIndex().get(slug);
      if (file) item.tags = (matter(fs.readFileSync(file, "utf8")).data.tags ?? []).map(String);
    }
    if (item.cover || item.tags?.length) items.push(item);
  }

  const withCover = items.filter((i) => i.cover).length;
  const withTags = items.filter((i) => i.tags?.length).length;
  console.log(`\x1b[1mrepair\x1b[0m — ${withCover} cover(s) to set, ${withTags} draft(s) needing tags, ${missing.length} with no cover asset`);
  for (const s of missing) console.log(`  \x1b[33mno cover file\x1b[0m ${s}`);
  if (!items.length) return;

  const payload = JSON.stringify({ publication_url: opts.substack, delay: opts.delay, items });
  const helper = path.join(ROOT, ".claude/scripts/substack-cover.py");
  let out = "";
  try {
    out = execFileSync("uv",
      ["run", "--quiet", "--python", "3.12", "--with", "python-substack", "python", helper],
      { input: payload, encoding: "utf8", env: substackEnv(), timeout: 3_600_000, maxBuffer: 32 * 1024 * 1024 });
  } catch (e) {
    out = (e.stdout || "").toString();
    console.log(`  \x1b[31mcover pass ended early\x1b[0m — ${clip((e.stderr || e.message || "").toString().replace(/\s+/g, " "), 200)}`);
  }

  for (const line of out.split("\n").filter(Boolean)) {
    let r;
    try { r = JSON.parse(line); } catch { continue; }
    if (r.ok) {
      const rec = { ...ledger[r.slug] };
      if (r.cover) rec.cover = r.cover;
      if (r.tags) delete rec.tagsIncomplete;
      ledger[r.slug] = rec;
      const what = [r.cover && "cover", r.tags && `${r.tags} tags`].filter(Boolean).join(" + ");
      console.log(`  \x1b[32mrepaired\x1b[0m ${r.slug} — ${what}`);
    } else {
      console.log(`  \x1b[31mrepair failed\x1b[0m ${r.slug} — ${clip(r.error, 160)}`);
    }
  }
  saveLedger(ledger);
}

/* ─────────────────────── publish plan ─────────────────────── */

/** Slugs this post links to, ignoring images, code and external URLs. */
function internalRefs(file) {
  const body = maskCode(matter(fs.readFileSync(file, "utf8")).content).md
    .replace(/!\[[^\]]*\]\([^)]*\)/g, "");
  const refs = new Set();
  for (const m of body.matchAll(/\[[^\]\n]+\]\((\/(?!\/)[^)\s]*)\)/g)) {
    const slug = m[1].replace(/[#?].*$/, "").replace(/\/$/, "").split("/").pop();
    if (slug && slug !== path.basename(file, ".md")) refs.add(slug);
  }
  return refs;
}

/**
 * Which posts can go out first, and in what order. A post is ready when every
 * post it links to is already on Substack — otherwise those links degrade to
 * plain text. Kahn's algorithm over the set gives the tiers; a cycle means the
 * posts reference each other and one of them has to ship with plain-text links.
 */
async function plan(files, opts) {
  const archive = opts.internalLinks === "substack" ? await loadArchive(opts) : { byTitle: new Map(), bySlug: new Map(), size: 0 };
  if (archive.error) console.log(`\x1b[33mcould not read ${opts.substack} (${archive.error}) — treating nothing as published\x1b[0m`);
  else console.log(`${archive.size} post(s) already on ${opts.substack}\n`);

  const set = new Map(); // slug -> {file, title, deps(in set), outside(unpublished, elsewhere)}
  for (const f of files) set.set(path.basename(f, ".md"), { file: f });
  for (const [slug, node] of set) {
    node.title = matter(fs.readFileSync(node.file, "utf8")).data.title ?? slug;
    node.deps = new Set();
    node.outside = new Set();
    for (const ref of internalRefs(node.file)) {
      if (substackTarget(ref, archive).state === "published") continue; // already satisfied
      if (set.has(ref)) node.deps.add(ref);
      else if (slugIndex().has(ref)) node.outside.add(ref);
    }
  }

  // How many posts in the set link *to* each one — the most-referenced post is
  // the most valuable to get out early, it unblocks the most links.
  for (const node of set.values()) node.inDegree = 0;
  for (const node of set.values()) for (const d of node.deps) set.get(d).inDegree++;

  // These series are densely cross-linked, so a clean topological order usually
  // doesn't exist. Publish greedily instead: always take the post with the
  // fewest still-unpublished references, breaking ties toward the one the most
  // other posts point at. Whatever it still can't link to ships as plain text.
  const done = new Set(), order = [];
  while (done.size < set.size) {
    const next = [...set]
      .filter(([s]) => !done.has(s))
      .map(([s, n]) => ({ slug: s, node: n, blocked: [...n.deps].filter((d) => !done.has(d)) }))
      // Only in-set links are order-dependent; refs to other folders are a
      // fixed cost. Weighting them here defers the hubs and triples the damage.
      .sort((a, b) => a.blocked.length - b.blocked.length || b.node.inDegree - a.node.inDegree)[0];
    done.add(next.slug);
    order.push(next);
  }

  const clean = order.filter((o) => !o.blocked.length && !o.node.outside.size);
  const broken = order.reduce((n, o) => n + o.blocked.length, 0);
  const outside = order.reduce((n, o) => n + o.node.outside.size, 0);
  console.log(`\x1b[1mpublish plan\x1b[0m — ${set.size} posts; in this order ${clean.length} come out with every cross-link resolved`);
  console.log(`${broken} link(s) inside the set stay plain text, plus ${outside} to posts in other folders\n`);

  order.forEach((o, i) => {
    const total = o.blocked.length + o.node.outside.size;
    const cost = total ? `\x1b[33m${String(total).padStart(2)} plain\x1b[0m` : `\x1b[32m  linked\x1b[0m`;
    const out = o.node.outside.size ? ` \x1b[90m(${o.node.outside.size} to other folders)\x1b[0m` : "";
    console.log(`${String(i + 1).padStart(3)}. ${cost}  ←${String(o.node.inDegree).padStart(3)} refs   ${o.slug}${out}`);
    if (o.blocked.length) console.log(`              waits on: ${o.blocked.join(", ")}`);
  });

  // Only posts that reference nothing unpublished *right now* can go out
  // together; the rest are clean only because of what precedes them here.
  const ready = order.filter((o) => o.node.deps.size === 0);
  const first = (ready.length ? ready : order.slice(0, 1)).map((o) => o.slug);
  console.log(ready.length
    ? `\n\x1b[1mStart here\x1b[0m — these reference nothing unpublished, they can go out in any order`
    : `\n\x1b[1mStart here\x1b[0m — every post references another, so publish the most-referenced one first and accept ${order[0].blocked.length + order[0].node.outside.size} plain-text link(s); re-run the export after each publish`);
  console.log(`  npm run substack -- \\\n    ${first.join(" \\\n    ")}\n`);
}

/* ─────────────────────── audit ─────────────────────── */

const clip = (s, n = 84) => (s.length > n ? `${s.slice(0, n)}…` : s);

function allPosts() {
  const out = [];
  (function walk(dir) {
    for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
      const p = path.join(dir, e.name);
      if (e.isDirectory()) walk(p);
      else if (e.name.endsWith(".md")) out.push(p);
    }
  })(CONTENT);
  return out.sort();
}

/**
 * Dry run over any number of posts that reports only the math: how much
 * converted to verified Unicode, how much fell back to an image, and every
 * formula KaTeX rejects. Nothing is written.
 */
function audit(files, opts, manifest) {
  const total = { unicode: 0, image: 0, raw: 0, invalid: 0, unfaithful: 0, posts: 0 };
  const invalid = [], unfaithful = [];

  for (const file of files) {
    let stats;
    try { ({ stats } = convert(file, opts, manifest)); }
    catch (e) { console.log(`\x1b[31mFAIL\x1b[0m ${path.relative(ROOT, file)}: ${e.message}`); continue; }
    total.posts++;
    total.unicode += stats.mathUnicode;
    total.image += stats.mathImages;
    total.raw += stats.mathRaw.length;
    total.invalid += stats.mathInvalid.length;
    total.unfaithful += stats.mathUnfaithful.length;
    const slug = path.basename(file, ".md");
    for (const t of stats.mathInvalid) invalid.push(`${slug}: ${clip(t)}`);
    for (const u of stats.mathUnfaithful) unfaithful.push(`${slug}: ${clip(u.tex)}  →  ${clip(u.uni, 40)}`);
  }

  const formulas = total.unicode + total.image + total.raw + total.invalid;
  console.log(`\n\x1b[1mmath audit\x1b[0m — ${total.posts} posts, ${formulas.toLocaleString()} formulas`);
  console.log(`  ${total.unicode.toLocaleString()} rendered as Unicode and verified against KaTeX`);
  console.log(`  ${total.image.toLocaleString()} rendered as an image from the exact LaTeX`);
  if (total.raw) console.log(`  ${total.raw.toLocaleString()} left as raw LaTeX (by flag)`);
  console.log(`  ${total.unfaithful.toLocaleString()} Unicode attempts rejected as unfaithful → sent to an image instead`);
  console.log(`  \x1b[${total.invalid ? "31" : "32"}m${total.invalid.toLocaleString()} formulas KaTeX cannot parse\x1b[0m`);
  const cap = process.env.SUBSTACK_FULL ? invalid.length : 30;
  for (const line of invalid.slice(0, cap)) console.log(`    ${line}`);
  if (invalid.length > cap) console.log(`    … ${invalid.length - cap} more (SUBSTACK_FULL=1 lists all)`);
  if (process.env.SUBSTACK_SHOW_REJECTS) for (const line of unfaithful.slice(0, 40)) console.log(`    reject ${line}`);
}

/* ─────────────────────── main ─────────────────────── */

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  // `--covers` works off the ledger, so it needs no post arguments.
  if (opts.covers && !opts.targets.length) return setCovers(opts);
  if (opts.help || !opts.targets.length) {
    console.log(fs.readFileSync(fileURLToPath(import.meta.url), "utf8").split("*/")[0].replace(/^\/\*\*|^ ?\* ?/gm, ""));
    process.exit(opts.help ? 0 : 1);
  }

  let manifest = {};
  const manifestPath = path.join(ROOT, "src/lib/generated/blogImageManifest.json");
  if (fs.existsSync(manifestPath)) manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));

  const files = opts.targets.flatMap((t) => {
    if (t === "all" && (opts.audit || opts.plan)) return allPosts();
    const dir = path.resolve(t);
    if (fs.existsSync(dir) && fs.statSync(dir).isDirectory()) {
      return fs.readdirSync(dir).filter((f) => f.endsWith(".md")).sort().map((f) => path.join(dir, f));
    }
    return [resolvePost(t)];
  });
  if (opts.linkBatch) opts.batch = new Set(files.map((f) => path.basename(f, ".md")));

  if (opts.plan) return plan(files, opts);
  if (opts.audit) return audit(files, opts, manifest);

  fs.mkdirSync(opts.out, { recursive: true });
  const openIt = opts.open ?? files.length === 1;

  if (opts.internalLinks === "substack") {
    opts.archive = await loadArchive(opts);
    if (opts.archive.error) {
      console.log(`\x1b[33mcould not read ${opts.substack} (${opts.archive.error}) — cross-post links will be plain text\x1b[0m`);
    } else {
      console.log(`${opts.archive.size} post(s) already on ${opts.substack}`);
    }
  }
  const pending = new Map();
  const ledger = opts.draft ? loadLedger() : {};
  const already = Object.keys(ledger).length;
  if (already) console.log(`${already} post(s) already pushed (per ${path.basename(LEDGER)}) — those are skipped`);
  let paced = false;
  let rateLimited = false;
  let rateLimits = 0;
  const skipped = [];
  let verified = false;
  let aborted = false;

  for (const file of files) {
    const result = convert(file, opts, manifest);
    const { slug, fm, html, stats } = result;

    const coverRel = `/imgs/blogs/${slug}-1.cover.webp`;
    const coverUrl = fs.existsSync(path.join(PUBLIC, coverRel)) ? opts.site + coverRel : null;

    const outFile = path.join(opts.out, `${slug}.html`);
    fs.writeFileSync(outFile, previewPage(result, opts, coverUrl));
    fs.writeFileSync(path.join(opts.out, `${slug}.body.html`), html);

    let mdFile = null;
    if (opts.markdown || opts.draft) {
      mdFile = path.join(opts.out, `${slug}.md`);
      fs.writeFileSync(mdFile, `${result.markdown}\n`);
      // The Markdown points at these; render them before anything reads it.
      if (stats.tableImages.length) {
        fs.mkdirSync(path.join(opts.out, "tables"), { recursive: true });
        for (const t of stats.tableImages) {
          await renderTable(stats.tableSources[t.index], t.file);
        }
        console.log(`  ${stats.tableImages.length} table(s) rendered as images (Substack has no table node)`);
      }
    }

    stats.animations.forEach((a, i) => {
      fs.writeFileSync(path.join(opts.out, `${slug}-anim-${i + 1}.svg`),
        a.svg.replace(/<\/?figure[^>]*>/g, "").replace(/<figcaption[\s\S]*?<\/figcaption>/g, "").trim());
    });

    const words = html.replace(/<[^>]+>/g, " ").split(/\s+/).filter(Boolean).length;
    console.log(`\n\x1b[1m${slug}\x1b[0m`);
    console.log(`  ${words.toLocaleString()} words · ${stats.images.length} images · ${stats.tables} tables`);
    if (stats.internalLinks) {
      if (opts.internalLinks === "absolute") {
        console.log(`  ${stats.internalLinks} cross-post link(s) → ${opts.site}`);
      } else if (opts.internalLinks === "substack") {
        console.log(`  ${stats.internalLinks} cross-post link(s): ${stats.linkedToSubstack} linked to Substack, ${stats.needsPublishing.size} not published yet, ${stats.deadLinks.size} dead`);
      } else {
        console.log(`  ${stats.internalLinks} cross-post link(s) unwrapped to plain text`);
      }
    }
    if (stats.needsPublishing.size) {
      console.log(`  \x1b[33mpublish these first\x1b[0m so this post can link to them:`);
      for (const [s, t] of [...stats.needsPublishing].slice(0, 25)) console.log(`    ${s}  —  ${clip(t, 60)}`);
      if (stats.needsPublishing.size > 25) console.log(`    … ${stats.needsPublishing.size - 25} more`);
      for (const [s, t] of stats.needsPublishing) pending.set(s, t);
    }
    for (const d of stats.deadLinks) console.log(`  \x1b[31mdead cross-link\x1b[0m ${d} — no such post in the repo`);
    // In draft mode display LaTeX is handed over on purpose, not as a fallback.
    const nativeMath = opts.displayMath === "raw" && opts.draft;
    console.log(nativeMath
      ? `  math: ${stats.mathUnicode} inline → unicode (verified), ${stats.mathDisplayRaw} display → LaTeX for Substack to render, ${stats.mathRaw.length - stats.mathDisplayRaw} left raw`
      : `  math: ${stats.mathUnicode} → unicode (verified), ${stats.mathImages} → image, ${stats.mathRaw.length} left raw`);
    if (stats.animations.length) console.log(`  \x1b[33m${stats.animations.length} animated figure(s) dropped\x1b[0m (saved as ${slug}-anim-*.svg)`);
    if (stats.leaks) console.log(`  \x1b[31m${stats.leaks} unresolved placeholder(s) — search the output for "ZZ"\x1b[0m`);
    if (stats.mathInvalid.length) {
      console.log(`  \x1b[31m${stats.mathInvalid.length} formula(s) KaTeX cannot parse\x1b[0m — broken in the source too, left verbatim:`);
      for (const t of stats.mathInvalid.slice(0, 5)) console.log(`    ${clip(t)}`);
    }
    if (stats.mathRaw.length && !nativeMath) {
      for (const t of stats.mathRaw.slice(0, 5)) console.log(`    raw: ${clip(t)}`);
      if (stats.mathRaw.length > 5) console.log(`    … ${stats.mathRaw.length - 5} more`);
    }
    if (stats.strayTex.length) {
      console.log(`  \x1b[33m${stats.strayTex.length} unrendered LaTeX span(s)\x1b[0m — unbalanced \`$\` in the source, same on the site:`);
      for (const t of stats.strayTex.slice(0, 3)) console.log(`    ${clip(t)}`);
    }
    console.log(`  Title:    ${fm.title ?? slug}`);
    if (fm.description) console.log(`  Subtitle: ${fm.description.slice(0, 120)}${fm.description.length > 120 ? "…" : ""}`);
    if (coverUrl) console.log(`  Cover:    ${coverUrl}`);
    console.log(`  → ${path.relative(process.cwd(), outFile)}`);

    if (opts.checkImages && stats.images.length) {
      const { bad, offline, total } = await checkImages(stats.images);
      if (offline) console.log(`  \x1b[33mimage check skipped — ${opts.site} unreachable\x1b[0m`);
      else if (bad.length) {
        console.log(`  \x1b[31m${bad.length}/${total} images not live yet\x1b[0m (deploy first — Substack fetches them on paste)`);
        for (const b of bad.slice(0, 5)) console.log(`    ${b}`);
      } else console.log(`  ${total} image URLs live ✓`);
    }
    if (opts.checkImages && stats.mathImageUrls.length) {
      const { bad, offline, total } = await checkImages(stats.mathImageUrls);
      if (!offline && bad.length) {
        console.log(`  \x1b[31m${bad.length}/${total} formula images failed to render\x1b[0m — the LaTeX is rejected by the renderer:`);
        for (const b of bad.slice(0, 5)) {
          const tex = decodeURIComponent(b.split("png.image?")[1] ?? "").replace(/^\\dpi\{\d+\}(\\bg\{\w+\})?/, "");
          console.log(`    ${clip(tex)}`);
        }
      } else if (!offline) console.log(`  ${total} formula images render ✓`);
    }

    if (mdFile) console.log(`  → ${path.relative(process.cwd(), mdFile)}`);
    if (mdFile && stats.strayDollars) {
      console.log(`  \x1b[31m${stats.strayDollars} bare \$ in the Markdown\x1b[0m — unbalanced \`$\` in the source (it mis-renders on the site too)`);
    }
    if (opts.draft && stats.strayDollars && !opts.forcePush) {
      // Pushing this would create a draft Substack refuses to open, and the only
      // symptom is a modal with no detail. Better to leave it out and say why.
      console.log(`  \x1b[31mnot pushed\x1b[0m — fix the \`$\` pairing in the source first, or pass --force-push`);
      skipped.push(slug);
    } else if (opts.draft && ledger[slug]) {
      const note = ledger[slug].tagsIncomplete ? " (tags may be incomplete)" : "";
      console.log(`  \x1b[90mskipped\x1b[0m — already pushed as draft ${ledger[slug].id}${note}`);
    } else if (opts.draft) {
      if (paced) await sleep(opts.delay * 1000);
      const r = createDraft(mdFile, slug, fm, opts);
      paced = true;
      if (r.ok) {
        // --no-tags leaves the draft complete but untagged, which is the same
        // state a cut-short tag loop leaves behind — mark it so `--covers`
        // applies the full set on the repair pass.
        ledger[slug] = {
          id: r.id, tags: r.tagsAdded, at: new Date().toISOString(),
          ...(opts.tags === false ? { tagsIncomplete: true } : {}),
        };
        saveLedger(ledger);
        if (!verified) {
          verified = true;
          const v = verifyDraft(r.id, opts);
          if (!v) console.log(`  \x1b[33mcould not read the draft back to verify it\x1b[0m`);
          else if (v.inlineLatex || v.unuploaded) {
            console.log(`  \x1b[31mthe draft is not usable\x1b[0m — ${v.inlineLatex} inline latex node(s), ${v.unuploaded}/${v.images} image(s) not uploaded`);
            console.log(`  stopping before this repeats across the batch; draft ${r.id} needs deleting`);
            aborted = true;
            break;
          } else {
            console.log(`  verified: ${v.nodes} nodes, ${v.images} images uploaded, no inline latex ✓`);
          }
        }
      } else if (isRateLimit(r.error)) {
        // The draft and its images are written before the tag loop, so a 429
        // here leaves a complete post with partial tags. Record it (that is what
        // keeps a resumed run from duplicating it), let the window recover, and
        // carry on — `--covers` finishes the tags afterwards.
        const claimed = claimAfterRateLimit(slug, fm, opts);
        if (claimed) {
          ledger[slug] = { ...claimed, at: new Date().toISOString() };
          saveLedger(ledger);
        }
        rateLimits++;
        console.log(`  \x1b[33mrate limited\x1b[0m${claimed ? ` — draft ${claimed.id} landed, tags incomplete` : " — nothing landed"}`);
        if (rateLimits > opts.maxRateLimits) {
          console.log(`  \x1b[31mgiving up\x1b[0m after ${rateLimits} rate limits — re-run later, the ledger resumes`);
          rateLimited = true;
          break;
        }
        console.log(`  cooling down ${opts.cooldown}s before the next post`);
        await sleep(opts.cooldown * 1000);
        paced = false; // the cooldown already paid for the inter-post delay
        continue;
      }
      if (r.missing) {
        console.log(`  \x1b[31mno \`substack\` command\x1b[0m — install it, then re-run:`);
        console.log(`    pipx install python-substack   # or: pip install python-substack`);
        console.log(`    substack status                # confirm it sees your account`);
      } else if (!r.ok) {
        console.log(`  \x1b[31mdraft failed\x1b[0m — ${clip(r.error, 300)}`);
        if (/slug/i.test(r.error)) {
          console.log(`    a draft or post already claims "${slug}" — delete it, or re-run with --no-slug`);
          console.log(`    substack drafts list --limit 20`);
        }
      } else {
        const tagNote = r.tagsAsked
          ? ` · ${r.tagsAdded}/${r.tagsAsked} tags${r.tagsAdded < r.tagsAsked ? " \x1b[33m(some rejected)\x1b[0m" : ""}`
          : "";
        console.log(`  \x1b[32mdraft created\x1b[0m${r.id ? ` (id ${r.id})` : ""}${tagNote} — unpublished`);
      }
    }
    if (opts.clipboard && copyHtmlToClipboard(html)) console.log("  → copied to clipboard (rich HTML)");
    if (openIt) {
      try { execFileSync("open", [outFile]); } catch { /* headless */ }
    }
  }

  const inLedger = files.filter((f) => ledger[path.basename(f, ".md")]).length;
  const partial = Object.values(ledger).filter((r) => r.tagsIncomplete).length;
  if (opts.draft) {
    console.log(`
\x1b[1m${inLedger} of ${files.length} pushed\x1b[0m${rateLimits ? ` · ${rateLimits} rate limit(s) hit` : ""}${partial ? ` · ${partial} with incomplete tags` : ""}`);
    if (rateLimited) console.log(`  Gave up this run. Re-run the same command — the ledger resumes it.`);
    if (aborted) console.log(`  \x1b[31mAborted after the first draft failed verification.\x1b[0m`);
    if (skipped.length) {
      console.log(`  \x1b[31m${skipped.length} post(s) held back for unbalanced \`$\`:\x1b[0m ${skipped.join(", ")}`);
    }
    if (partial) console.log(`  Finish the tags (and set covers) with:  npm run substack -- --covers`);
  }

  console.log(opts.draft ? `
\x1b[1mNext\x1b[0m
  The drafts are unpublished. Open them in Substack, check the figures and the
  math rendered, set the cover image, then publish from the Substack UI.` : `
\x1b[1mPublish flow\x1b[0m
  1. Click "Copy post body" in the page that just opened.
  2. In Substack: New post → paste into the body (⌘V).
  3. Paste the title and subtitle into their own fields (buttons at the top).
  4. Wait for every image to finish uploading, then set the cover image.`);

  if (pending.size) {
    // Cross-links only resolve once the target exists on Substack, so the
    // referenced posts have to go out first — or their links stay plain text.
    console.log(`
\x1b[1m${pending.size} referenced post(s) are not on Substack yet\x1b[0m
  Publish them before this one, then re-run the export so the links resolve:

  npm run substack -- \\
    ${[...pending.keys()].join(" \\\n    ")}

  Or accept plain-text references for now — the prose still reads fine.`);
  }
  console.log("");
}

main().catch((err) => die(err.stack ?? String(err)));
