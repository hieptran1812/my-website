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
  | "poster";

export interface Venue {
  /** Canonical conference slug: "cvpr", "neurips", ... */
  conf: string;
  year?: number;
  tier?: VenueTier;
}

/** Canonical slug → display name, plus spellings seen in the wild. */
const CONFERENCES: Record<string, { name: string; aliases: string[] }> = {
  cvpr: { name: "CVPR", aliases: [] },
  iccv: { name: "ICCV", aliases: [] },
  eccv: { name: "ECCV", aliases: [] },
  wacv: { name: "WACV", aliases: [] },
  neurips: { name: "NeurIPS", aliases: ["nips"] },
  icml: { name: "ICML", aliases: [] },
  iclr: { name: "ICLR", aliases: [] },
  aaai: { name: "AAAI", aliases: [] },
  ijcai: { name: "IJCAI", aliases: [] },
  aistats: { name: "AISTATS", aliases: [] },
  uai: { name: "UAI", aliases: [] },
  colt: { name: "COLT", aliases: [] },
  acl: { name: "ACL", aliases: [] },
  emnlp: { name: "EMNLP", aliases: [] },
  naacl: { name: "NAACL", aliases: [] },
  coling: { name: "COLING", aliases: [] },
  colm: { name: "COLM", aliases: [] },
  tmlr: { name: "TMLR", aliases: [] },
  kdd: { name: "KDD", aliases: ["sigkdd"] },
  sigir: { name: "SIGIR", aliases: [] },
  recsys: { name: "RecSys", aliases: [] },
  interspeech: { name: "Interspeech", aliases: [] },
  icassp: { name: "ICASSP", aliases: [] },
  corl: { name: "CoRL", aliases: [] },
  icra: { name: "ICRA", aliases: [] },
  siggraph: { name: "SIGGRAPH", aliases: [] },
};

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
];

export const TIER_LABELS: Record<VenueTier, string> = {
  award: "Award candidate",
  oral: "Oral",
  spotlight: "Spotlight",
  highlight: "Highlight",
  poster: "Poster",
};

const TIER_WORDS: [RegExp, VenueTier][] = [
  [/^(award(-?candidate)?|best-?paper|outstanding(-?paper)?)$/, "award"],
  [/^oral$/, "oral"],
  [/^spotlight$/, "spotlight"],
  [/^highlights?$/, "highlight"],
  [/^poster$/, "poster"],
];

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

const ALIAS_PATTERN = [...ALIAS_TO_CONF.keys()]
  .sort((a, b) => b.length - a.length)
  .join("|");

/** `cvpr-2025-oral`, `neurips2024`, `iclr_25`, `icml-2024-award-candidate`. */
const TAG_RE = new RegExp(
  `^(${ALIAS_PATTERN})[-_ ]?('?\\d{2}|\\d{4})?(?:[-_ ](.+))?$`,
  "i",
);

/** "NeurIPS 2024", "CVPR'25 Oral", "ICLR 2025 (Spotlight)". */
const TEXT_RE = new RegExp(
  `\\b(${ALIAS_PATTERN})\\s*'?(\\d{4}|\\d{2})\\b(?:\\s*[(\\[]?\\s*(award candidate|best paper|outstanding paper|oral|spotlight|highlight|poster))?`,
  "i",
);

export function parseVenueTag(tag: string): Venue | null {
  const m = TAG_RE.exec(tag.trim());
  if (!m) return null;
  const conf = ALIAS_TO_CONF.get(m[1].toLowerCase());
  if (!conf) return null;
  const year = m[2] ? normalizeYear(m[2].replace("'", "")) : undefined;
  const tier = m[3] ? tierFromWord(m[3]) : undefined;
  // A bare conference word with an unknown suffix ("acl-anthology") is a topic
  // tag, not a venue.
  if (m[3] && !tier) return null;
  // "acl" or "kdd" alone is too ambiguous to trust as a venue.
  if (!year && !tier) return null;
  return { conf, year, tier };
}

export function parseVenueText(text: string): Venue | null {
  const m = TEXT_RE.exec(text);
  if (!m) return null;
  const conf = ALIAS_TO_CONF.get(m[1].toLowerCase());
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
  const tokens = query.trim().split(/\s+/).filter(Boolean);
  let conf: string | undefined;
  let year: number | undefined;
  let tier: VenueTier | undefined;
  const rest: string[] = [];
  for (const raw of tokens) {
    const tok = raw.toLowerCase();
    const glued = TAG_RE.exec(tok);
    if (!conf && glued && ALIAS_TO_CONF.has(glued[1].toLowerCase())) {
      const parsed = {
        conf: ALIAS_TO_CONF.get(glued[1].toLowerCase())!,
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
