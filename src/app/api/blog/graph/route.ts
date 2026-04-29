import { NextRequest, NextResponse } from "next/server";
import {
  getIndex,
  cosineSimilarity,
  type IndexEntry,
  type CorpusIndex,
} from "@/lib/getRelatedPosts";

// ─────────────── Wire types ───────────────

export type GraphEdgeType = "series" | "reference" | "topic" | "similar";

export interface GraphNode {
  id: string;
  slug: string;
  title: string;
  category: string;
  subcategory: string;
  /** 0 = current article, 1 = first hop, 2 = second hop, ... */
  hop: number;
  /** Composite relevance against the centre node, 0..1 normalised. */
  relevance: number;
  image?: string;
  publishDate: string;
}

export interface GraphEdge {
  source: string;
  target: string;
  type: GraphEdgeType;
  /** 0..1 normalised within the returned subgraph. */
  weight: number;
  /** Human-readable evidence shown on hover (e.g. "Shares 'concept-erasure'"). */
  evidence?: string;
  /** For reference edges, true when source links to target (else target links source). */
  directed?: boolean;
}

export interface GraphResponse {
  mode: "ego" | "universe";
  currentNodeId?: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
  /** subcategory → hex colour, deterministic by hash. */
  palette: Record<string, string>;
}

// ─────────────── Tunables ───────────────

const HOP1_LIMIT = 8;
const HOP2_PER_HOP1 = 3;
const MIN_TOPIC_SHARED_TAGS = 1;
const MIN_SIMILARITY = 0.06;
const MAX_DEPTH = 3;

// Composite weighting for hop-1 ranking. Mirrors getRelatedPosts but adds the
// reference-graph signal as a strong direct evidence channel.
const W = {
  tag: 1.0,
  cosine: 0.7,
  subcategory: 0.4,
  category: 0.15,
  collection: 2.0,
  reference: 1.0,
};

// ─────────────── Helpers ───────────────

function hashHue(seed: string): number {
  let h = 5381;
  for (let i = 0; i < seed.length; i++) h = ((h << 5) + h + seed.charCodeAt(i)) | 0;
  return Math.abs(h) % 360;
}

function buildPalette(subcats: Iterable<string>): Record<string, string> {
  const out: Record<string, string> = {};
  for (const s of subcats) {
    const hue = hashHue(s || "default");
    out[s] = `hsl(${hue} 65% 55%)`;
  }
  return out;
}

interface Scored {
  entry: IndexEntry;
  composite: number;
  cosine: number;
  sharedTags: string[];
  rareSharedTag?: string;
  rareWeight: number;
  references: "out" | "in" | null;
  sameCollection: boolean;
}

function scoreOne(
  current: IndexEntry,
  candidate: IndexEntry,
  idx: CorpusIndex,
): Scored {
  const currentTagSet = new Set(current.tags);
  const sharedTags = candidate.tags.filter((t) => currentTagSet.has(t));
  let tagScore = 0;
  let rareSharedTag: string | undefined;
  let rareWeight = 0;
  for (const t of sharedTags) {
    const w = idx.tagIdf.get(t) ?? 0;
    tagScore += w;
    if (w > rareWeight) {
      rareWeight = w;
      rareSharedTag = t;
    }
  }
  const maxTagScore =
    current.tags.reduce((s, t) => s + (idx.tagIdf.get(t) ?? 0), 0) || 1;
  const tagNorm = Math.min(1, tagScore / maxTagScore);

  const cos = cosineSimilarity(current, candidate, idx.tokenIdf);

  const sameSub =
    !!current.subcategory && current.subcategory === candidate.subcategory;
  const sameCat =
    !!current.category && current.category === candidate.category;
  const sameCollection =
    !!current.collection &&
    current.collection.toLowerCase() ===
      (candidate.collection || "").toLowerCase();

  const outgoing = idx.outgoingRefs.get(current.slug);
  const incoming = idx.incomingRefs.get(current.slug);
  let references: "out" | "in" | null = null;
  if (outgoing?.has(candidate.slug)) references = "out";
  else if (incoming?.has(candidate.slug)) references = "in";

  const composite =
    W.tag * tagNorm +
    W.cosine * cos +
    (sameSub ? W.subcategory : 0) +
    (sameCat ? W.category : 0) +
    (sameCollection ? W.collection : 0) +
    (references ? W.reference : 0);

  return {
    entry: candidate,
    composite,
    cosine: cos,
    sharedTags,
    rareSharedTag,
    rareWeight,
    references,
    sameCollection,
  };
}

/**
 * Inspect a (current, candidate) pair and return one or more typed edges with
 * their weights. The same pair can produce multiple edges (e.g. same series +
 * reference) — the renderer can stack them or pick the strongest.
 */
function deriveEdges(
  current: IndexEntry,
  candidate: IndexEntry,
  scored: Scored,
): GraphEdge[] {
  const edges: GraphEdge[] = [];
  if (scored.sameCollection) {
    edges.push({
      source: current.slug,
      target: candidate.slug,
      type: "series",
      weight: 1,
      evidence: `Series: ${current.collection ?? ""}`,
      directed: true,
    });
  }
  if (scored.references === "out") {
    edges.push({
      source: current.slug,
      target: candidate.slug,
      type: "reference",
      weight: 0.85,
      evidence: "References this article",
      directed: true,
    });
  } else if (scored.references === "in") {
    edges.push({
      source: candidate.slug,
      target: current.slug,
      type: "reference",
      weight: 0.85,
      evidence: "Cited by this article",
      directed: true,
    });
  }
  if (scored.sharedTags.length >= MIN_TOPIC_SHARED_TAGS) {
    const detail = scored.rareSharedTag
      ? `Shares '${scored.rareSharedTag}'`
      : `Shares ${scored.sharedTags.length} tag${scored.sharedTags.length > 1 ? "s" : ""}`;
    edges.push({
      source: current.slug,
      target: candidate.slug,
      type: "topic",
      // tagNorm-style weight, capped
      weight: Math.min(1, 0.4 + scored.rareWeight * 0.25),
      evidence: detail,
    });
  }
  if (scored.cosine >= MIN_SIMILARITY) {
    edges.push({
      source: current.slug,
      target: candidate.slug,
      type: "similar",
      weight: Math.min(1, scored.cosine * 1.3),
      evidence: `${Math.round(scored.cosine * 100)}% similar`,
    });
  }
  return edges;
}

function entryToNode(
  entry: IndexEntry,
  hop: number,
  relevance: number,
): GraphNode {
  return {
    id: entry.slug,
    slug: entry.slug,
    title: entry.title,
    category: entry.category,
    subcategory: entry.subcategory,
    hop,
    relevance,
    image: entry.image,
    publishDate: entry.publishDate,
  };
}

// ─────────────── Ego mode ───────────────

async function buildEgoGraph(
  slug: string,
  depth: number,
  idx: CorpusIndex,
): Promise<GraphResponse | null> {
  const current = idx.bySlug.get(slug);
  if (!current) return null;

  const nodes = new Map<string, GraphNode>();
  const edges: GraphEdge[] = [];
  const seen = new Set<string>([slug]);

  nodes.set(slug, entryToNode(current, 0, 1));

  // Hop-1: rank the whole corpus, take HOP1_LIMIT.
  const hop1Scored: Scored[] = [];
  for (const cand of idx.entries) {
    if (cand.slug === slug) continue;
    const s = scoreOne(current, cand, idx);
    if (s.composite <= 0 && !s.references && !s.sameCollection) continue;
    hop1Scored.push(s);
  }
  hop1Scored.sort((a, b) => b.composite - a.composite);
  const topComposite = hop1Scored[0]?.composite ?? 1;
  const hop1 = hop1Scored.slice(0, HOP1_LIMIT);

  for (const s of hop1) {
    seen.add(s.entry.slug);
    nodes.set(
      s.entry.slug,
      entryToNode(s.entry, 1, topComposite > 0 ? s.composite / topComposite : 0),
    );
    for (const e of deriveEdges(current, s.entry, s)) edges.push(e);
  }

  // Hop-2: from each hop-1 node, take its top HOP2_PER_HOP1 neighbours that
  // aren't already in the graph. This widens the network without flooding it.
  if (depth >= 2) {
    for (const hop1Score of hop1) {
      const hop1Entry = hop1Score.entry;
      const hop2Scored: Scored[] = [];
      for (const cand of idx.entries) {
        if (cand.slug === hop1Entry.slug) continue;
        if (seen.has(cand.slug)) continue;
        const s = scoreOne(hop1Entry, cand, idx);
        if (s.composite <= 0 && !s.references && !s.sameCollection) continue;
        hop2Scored.push(s);
      }
      hop2Scored.sort((a, b) => b.composite - a.composite);
      const top2Composite = hop2Scored[0]?.composite ?? 1;
      for (const s of hop2Scored.slice(0, HOP2_PER_HOP1)) {
        seen.add(s.entry.slug);
        nodes.set(
          s.entry.slug,
          entryToNode(
            s.entry,
            2,
            top2Composite > 0 ? (s.composite / top2Composite) * 0.5 : 0,
          ),
        );
        for (const e of deriveEdges(hop1Entry, s.entry, s)) edges.push(e);
      }
    }
  }

  // Hop-3: only connect existing nodes to each other if they happen to share
  // an edge (no new nodes). Cheap; surfaces lateral structure.
  if (depth >= 3) {
    const slugs = Array.from(nodes.keys());
    for (let i = 0; i < slugs.length; i++) {
      for (let j = i + 1; j < slugs.length; j++) {
        const a = idx.bySlug.get(slugs[i]);
        const b = idx.bySlug.get(slugs[j]);
        if (!a || !b) continue;
        if (a.slug === slug || b.slug === slug) continue;
        const s = scoreOne(a, b, idx);
        for (const e of deriveEdges(a, b, s)) edges.push(e);
      }
    }
  }

  // Normalise edge weights inside this subgraph so the renderer's
  // stroke-width / opacity has a comparable scale across articles.
  const maxW = edges.reduce((m, e) => Math.max(m, e.weight), 0) || 1;
  for (const e of edges) e.weight = e.weight / maxW;

  // Deduplicate stacked edges (same pair, same type) keeping the heaviest.
  const dedup = new Map<string, GraphEdge>();
  for (const e of edges) {
    const key = `${e.source}->${e.target}|${e.type}`;
    const reverse = `${e.target}->${e.source}|${e.type}`;
    const existing = dedup.get(key) ?? dedup.get(reverse);
    if (!existing || e.weight > existing.weight) dedup.set(key, e);
  }

  const palette = buildPalette(
    new Set(Array.from(nodes.values()).map((n) => n.subcategory)),
  );

  return {
    mode: "ego",
    currentNodeId: slug,
    nodes: Array.from(nodes.values()),
    edges: Array.from(dedup.values()),
    palette,
  };
}

// ─────────────── Universe mode ───────────────

function buildUniverseGraph(idx: CorpusIndex): GraphResponse {
  // Show every post but only reference edges (skip tag pairs — they explode).
  const nodes: GraphNode[] = idx.entries.map((e) => entryToNode(e, 1, 0));
  const edges: GraphEdge[] = [];
  for (const [src, targets] of idx.outgoingRefs) {
    for (const tgt of targets) {
      edges.push({
        source: src,
        target: tgt,
        type: "reference",
        weight: 1,
        directed: true,
      });
    }
  }
  const palette = buildPalette(new Set(nodes.map((n) => n.subcategory)));
  return { mode: "universe", nodes, edges, palette };
}

// ─────────────── Route handler ───────────────

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const slug = searchParams.get("slug")?.trim();
    const depth = Math.min(
      MAX_DEPTH,
      Math.max(1, parseInt(searchParams.get("depth") || "2", 10)),
    );

    const idx = await getIndex();

    let payload: GraphResponse;
    if (!slug) {
      payload = buildUniverseGraph(idx);
    } else {
      const ego = await buildEgoGraph(slug, depth, idx);
      payload = ego ?? buildUniverseGraph(idx);
    }

    return NextResponse.json(payload, {
      headers: {
        "Cache-Control":
          "public, s-maxage=3600, stale-while-revalidate=86400",
      },
    });
  } catch (err) {
    console.error("Error building graph:", err);
    return NextResponse.json(
      { mode: "ego", nodes: [], edges: [], palette: {} },
      { status: 500 },
    );
  }
}
