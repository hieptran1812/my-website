/**
 * Publication venue (conference + year + acceptance tier) inferred from a
 * paper-reading post, so the listing can be filtered by conference without
 * every post carrying a dedicated frontmatter field.
 *
 * Sources, strongest first:
 *   1. tags like `cvpr-2025-oral`, `neurips2024`, `iclr-25-spotlight`
 *   2. the title or excerpt naming one, e.g. "(NeurIPS 2024 Oral)"
 */

export type VenueTier =
  | "award"
  | "oral"
  | "spotlight"
  | "highlight"
  | "poster"
  | "findings"
  | "workshop";

export interface Venue {
  /** Canonical conference slug: "cvpr", "neurips", ... */
  conf: string;
  year?: number;
  tier?: VenueTier;
}

export type VenueField =
  "ml" | "vision" | "nlp" | "speech" | "ir" | "robotics" | "graphics" | "ai";

export const FIELD_LABELS: Record<VenueField, string> = {
  ml: "Machine learning",
  vision: "Computer vision",
  nlp: "NLP",
  speech: "Speech & audio",
  ir: "Data mining & IR",
  robotics: "Robotics",
  graphics: "Graphics",
  ai: "General AI",
};

/**
 * Canonical slug → display name, field, and spellings seen in the wild.
 * `top` venues are always offered in the filter, even before a post has one,
 * so the reader can see what the listing covers (or will).
 */
const CONFERENCES: Record<
  string,
  { name: string; field: VenueField; top?: boolean; aliases: string[] }
> = {
  neurips: { name: "NeurIPS", field: "ml", top: true, aliases: ["nips"] },
  icml: { name: "ICML", field: "ml", top: true, aliases: [] },
  iclr: { name: "ICLR", field: "ml", top: true, aliases: [] },
  colm: { name: "COLM", field: "ml", top: true, aliases: [] },
  aistats: { name: "AISTATS", field: "ml", aliases: [] },
  uai: { name: "UAI", field: "ml", aliases: [] },
  colt: { name: "COLT", field: "ml", aliases: [] },
  tmlr: { name: "TMLR", field: "ml", aliases: [] },
  jmlr: { name: "JMLR", field: "ml", aliases: [] },
  mlsys: { name: "MLSys", field: "ml", aliases: [] },
  cvpr: { name: "CVPR", field: "vision", top: true, aliases: [] },
  iccv: { name: "ICCV", field: "vision", top: true, aliases: [] },
  eccv: { name: "ECCV", field: "vision", top: true, aliases: [] },
  wacv: { name: "WACV", field: "vision", aliases: [] },
  bmvc: { name: "BMVC", field: "vision", aliases: [] },
  "3dv": { name: "3DV", field: "vision", aliases: [] },
  miccai: { name: "MICCAI", field: "vision", aliases: [] },
  tpami: { name: "TPAMI", field: "vision", aliases: [] },
  acl: { name: "ACL", field: "nlp", top: true, aliases: [] },
  emnlp: { name: "EMNLP", field: "nlp", top: true, aliases: [] },
  naacl: { name: "NAACL", field: "nlp", top: true, aliases: [] },
  eacl: { name: "EACL", field: "nlp", aliases: [] },
  coling: { name: "COLING", field: "nlp", aliases: [] },
  tacl: { name: "TACL", field: "nlp", aliases: [] },
  interspeech: { name: "Interspeech", field: "speech", top: true, aliases: [] },
  icassp: { name: "ICASSP", field: "speech", top: true, aliases: [] },
  asru: { name: "ASRU", field: "speech", aliases: [] },
  slt: { name: "SLT", field: "speech", aliases: [] },
  ismir: { name: "ISMIR", field: "speech", aliases: [] },
  kdd: { name: "KDD", field: "ir", top: true, aliases: ["sigkdd"] },
  sigir: { name: "SIGIR", field: "ir", top: true, aliases: [] },
  www: { name: "WWW", field: "ir", aliases: ["thewebconf"] },
  wsdm: { name: "WSDM", field: "ir", aliases: [] },
  cikm: { name: "CIKM", field: "ir", aliases: [] },
  recsys: { name: "RecSys", field: "ir", aliases: [] },
  corl: { name: "CoRL", field: "robotics", top: true, aliases: [] },
  rss: { name: "RSS", field: "robotics", aliases: [] },
  icra: { name: "ICRA", field: "robotics", aliases: [] },
  iros: { name: "IROS", field: "robotics", aliases: [] },
  siggraph: { name: "SIGGRAPH", field: "graphics", top: true, aliases: [] },
  "siggraph-asia": { name: "SIGGRAPH Asia", field: "graphics", aliases: [] },
  aaai: { name: "AAAI", field: "ai", top: true, aliases: [] },
  ijcai: { name: "IJCAI", field: "ai", aliases: [] },
};

/** Field → its conferences, top venues first, in registry order. */
export function conferencesByField(): {
  field: VenueField;
  confs: { slug: string; top: boolean }[];
}[] {
  const groups = new Map<VenueField, { slug: string; top: boolean }[]>();
  for (const [slug, c] of Object.entries(CONFERENCES)) {
    const list = groups.get(c.field) ?? [];
    list.push({ slug, top: Boolean(c.top) });
    groups.set(c.field, list);
  }
  return (Object.keys(FIELD_LABELS) as VenueField[])
    .filter((f) => groups.has(f))
    .map((field) => ({ field, confs: groups.get(field)! }));
}

const ALIAS_TO_CONF = new Map<string, string>();
for (const [slug, { aliases }] of Object.entries(CONFERENCES)) {
  ALIAS_TO_CONF.set(slug, slug);
  for (const alias of aliases) ALIAS_TO_CONF.set(alias, slug);
}

/** Strongest first; when a post has several, the strongest wins. */
export const TIER_ORDER: VenueTier[] = [
  "award",
  "oral",
  "spotlight",
  "highlight",
  "poster",
  "findings",
  "workshop",
];

export const TIER_LABELS: Record<VenueTier, string> = {
  award: "Award candidate",
  oral: "Oral",
  spotlight: "Spotlight",
  highlight: "Highlight",
  poster: "Poster",
  findings: "Findings",
  workshop: "Workshop",
};

const TIER_WORDS: [RegExp, VenueTier][] = [
  [/^(award(-?candidate)?|best-?paper|outstanding(-?paper)?)$/, "award"],
  [/^oral$/, "oral"],
  [/^spotlight$/, "spotlight"],
  [/^highlights?$/, "highlight"],
  [/^poster$/, "poster"],
  [/^findings$/, "findings"],
  [/^workshops?$/, "workshop"],
];

/** Track words that say "main conference" and carry no tier of their own. */
const NEUTRAL_SUFFIX = /^(main|main-?track|conference|long|short)$/;

function tierFromWord(word: string): VenueTier | undefined {
  const w = word.toLowerCase().replace(/[\s_]+/g, "-");
  for (const [re, tier] of TIER_WORDS) if (re.test(w)) return tier;
  return undefined;
}

function normalizeYear(raw: string): number | undefined {
  const n = parseInt(raw, 10);
  if (Number.isNaN(n)) return undefined;
  const year = raw.length === 2 ? 2000 + n : n;
  return year >= 1980 && year <= 2100 ? year : undefined;
}

// Multi-word venues ("siggraph-asia") also match with a space in prose.
const ALIAS_PATTERN = [...ALIAS_TO_CONF.keys()]
  .sort((a, b) => b.length - a.length)
  .map((a) => a.replace(/-/g, "[-\\s]"))
  .join("|");

function lookupConf(raw: string): string | undefined {
  return ALIAS_TO_CONF.get(raw.toLowerCase().replace(/\s+/g, "-"));
}

/** `cvpr-2025-oral`, `neurips2024`, `iclr_25`, `icml-2024-award-candidate`. */
const TAG_RE = new RegExp(
  `^(${ALIAS_PATTERN})[-_ ]?('?\\d{2}|\\d{4})?(?:[-_ ](.+))?$`,
  "i",
);

/** "NeurIPS 2024", "CVPR'25 Oral", "ICLR 2025 (Spotlight)". */
const TEXT_RE = new RegExp(
  `\\b(${ALIAS_PATTERN})\\s*'?(\\d{4}|\\d{2})\\b(?:\\s*[(\\[]?\\s*(award candidate|best paper|outstanding paper|oral|spotlight|highlight|poster|findings|workshop))?`,
  "i",
);

export function parseVenueTag(tag: string): Venue | null {
  const m = TAG_RE.exec(tag.trim());
  if (!m) return null;
  const conf = lookupConf(m[1]);
  if (!conf) return null;
  const year = m[2] ? normalizeYear(m[2].replace("'", "")) : undefined;
  const suffix = m[3]?.toLowerCase();
  const tier = suffix ? tierFromWord(suffix) : undefined;
  const neutral = suffix ? NEUTRAL_SUFFIX.test(suffix) : false;
  // A bare conference word with an unknown suffix ("acl-anthology") is a topic
  // tag, not a venue.
  if (suffix && !tier && !neutral) return null;
  // "acl" or "kdd" alone is too ambiguous to trust as a venue.
  if (!year && !tier) return null;
  return { conf, year, tier };
}

export function parseVenueText(text: string): Venue | null {
  const m = TEXT_RE.exec(text);
  if (!m) return null;
  const conf = lookupConf(m[1]);
  if (!conf) return null;
  return {
    conf,
    year: normalizeYear(m[2]),
    tier: m[3] ? tierFromWord(m[3]) : undefined,
  };
}

/** Fold several partial readings of the same post into one venue. */
function merge(found: Venue[]): Venue | null {
  if (found.length === 0) return null;
  const conf = found[0].conf;
  const same = found.filter((v) => v.conf === conf);
  const year = same.find((v) => v.year)?.year;
  const tiers = same
    .map((v) => v.tier)
    .filter((t): t is VenueTier => Boolean(t))
    .sort((a, b) => TIER_ORDER.indexOf(a) - TIER_ORDER.indexOf(b));
  return { conf, year, tier: tiers[0] };
}

export function getVenue(article: {
  tags?: string[];
  title?: string;
  excerpt?: string;
}): Venue | null {
  const fromTags = (article.tags ?? [])
    .map(parseVenueTag)
    .filter((v): v is Venue => v !== null);
  if (fromTags.length) return merge(fromTags);
  // A title naming a venue is about the paper itself. An excerpt often cites
  // other work ("unlike DINO (ICCV 2021)"), so it only counts when it says the
  // paper was accepted there or names an acceptance tier.
  const fromTitle = article.title ? parseVenueText(article.title) : null;
  if (fromTitle) return fromTitle;
  if (!article.excerpt) return null;
  const v = parseVenueText(article.excerpt);
  if (!v) return null;
  const accepted = new RegExp(
    `\\b(accepted|published|presented|appear(?:s|ed|ing)?|award)\\b[^.]{0,30}\\b(${ALIAS_PATTERN})\\b`,
    "i",
  ).test(article.excerpt);
  return accepted || v.tier ? v : null;
}

export function conferenceName(conf: string): string {
  return CONFERENCES[conf]?.name ?? conf.toUpperCase();
}

export function venueLabel(v: Venue): string {
  return [
    conferenceName(v.conf),
    v.year,
    v.tier ? TIER_LABELS[v.tier] : undefined,
  ]
    .filter(Boolean)
    .join(" ");
}

/** True when a tag only restates the venue (hidden on cards next to the badge). */
export function isVenueTag(tag: string): boolean {
  return parseVenueTag(tag) !== null;
}

/**
 * Read a venue out of a free-text search query, e.g. "cvpr 2025 oral 3d" →
 * { conf: "cvpr", year: 2025, tier: "oral" } with "3d" left over. Lets the
 * page offer a one-click filter for what the reader is obviously asking for.
 */
export function extractVenueFromQuery(
  query: string,
): { venue: Venue; rest: string } | null {
  const tokens = query
    .trim()
    .replace(/\bsiggraph\s+asia\b/gi, "siggraph-asia")
    .split(/\s+/)
    .filter(Boolean);
  let conf: string | undefined;
  let year: number | undefined;
  let tier: VenueTier | undefined;
  const rest: string[] = [];
  for (const raw of tokens) {
    const tok = raw.toLowerCase();
    const glued = TAG_RE.exec(tok);
    if (!conf && glued && lookupConf(glued[1])) {
      const parsed = {
        conf: lookupConf(glued[1])!,
        year: glued[2] ? normalizeYear(glued[2].replace("'", "")) : undefined,
        tier: glued[3] ? tierFromWord(glued[3]) : undefined,
      };
      if (!glued[3] || parsed.tier) {
        conf = parsed.conf;
        year = year ?? parsed.year;
        tier = tier ?? parsed.tier;
        continue;
      }
    }
    // A 4-digit year counts anywhere ("2025 cvpr"); a 2-digit one only after
    // a conference, where it can't be a plain number in the query.
    if (!year && (/^\d{4}$/.test(tok) || (conf && /^'?\d{2}$/.test(tok)))) {
      year = normalizeYear(tok.replace("'", ""));
      if (year) continue;
    }
    const t = tierFromWord(tok);
    if (t && !tier) {
      tier = t;
      continue;
    }
    rest.push(raw);
  }
  if (!conf) return null;
  return { venue: { conf, year, tier }, rest: rest.join(" ") };
}
