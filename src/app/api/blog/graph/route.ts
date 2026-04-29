import { NextRequest, NextResponse } from "next/server";
import {
  getIndex,
  personalizedPageRank,
  selectWithMmr,
  type CorpusIndex,
  type DominantSignal,
} from "@/lib/getRelatedPosts";

export interface GraphNode {
  id: string;
  slug: string;
  title: string;
  category: string;
  subcategory: string;
  /** PPR-derived importance, 0..1 normalised within the subgraph. */
  relevance: number;
  image?: string;
  publishDate: string;
}

export interface GraphEdge {
  source: string;
  target: string;
  /** Composite weight, 0..1 normalised within the subgraph. */
  weight: number;
  /** Dominant signal that contributed most to the weight (for hover hints). */
  dominant: DominantSignal;
  /** Human-readable evidence shown only on hover — kept off the default render. */
  evidence: string;
  /** Reference edges are directional (source → target). */
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

const NODE_BUDGET = 24; // top-K from PPR after MMR diversification
const PPR_POOL = 100; // size of MMR candidate pool
const PPR_ALPHA = 0.85;
const PPR_ITERS = 30;
const MMR_LAMBDA = 0.78;
const EDGE_DENSITY = 1.8; // edges = nodes * EDGE_DENSITY (cap)

// ─────────────── Universe mode (whole-corpus reference scatter) ───────────────

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

function buildUniverseGraph(idx: CorpusIndex): GraphResponse {
  const nodes: GraphNode[] = idx.entries.map((e) => ({
    id: e.slug,
    slug: e.slug,
    title: e.title,
    category: e.category,
    subcategory: e.subcategory,
    relevance: 0,
    image: e.image,
    publishDate: e.publishDate,
  }));
  const edges: GraphEdge[] = [];
  for (const [src, targets] of idx.outgoingRefs) {
    for (const tgt of targets) {
      edges.push({
        source: src,
        target: tgt,
        weight: 1,
        dominant: "reference",
        evidence: "References",
        directed: true,
      });
    }
  }
  return {
    mode: "universe",
    nodes,
    edges,
    palette: buildPalette(new Set(nodes.map((n) => n.subcategory))),
  };
}

// ─────────────── Ego mode (PPR + MMR) ───────────────

function buildEgoGraph(slug: string, idx: CorpusIndex): GraphResponse | null {
  const current = idx.bySlug.get(slug);
  if (!current) return null;

  // 1. Run Personalized PageRank from the current article.
  const ppr = personalizedPageRank(slug, idx, PPR_ALPHA, PPR_ITERS);

  // 2. Take the top PPR_POOL candidates (excluding the seed itself).
  const ranked: Array<{ slug: string; score: number }> = [];
  for (const [s, p] of ppr) {
    if (s === slug) continue;
    ranked.push({ slug: s, score: p });
  }
  ranked.sort((a, b) => b.score - a.score);
  const pool = ranked.slice(0, PPR_POOL);

  // 3. MMR-diversify down to NODE_BUDGET.
  const selectedSlugs = selectWithMmr(pool, NODE_BUDGET, idx, MMR_LAMBDA);

  // 4. Build node set (centre + selection).
  const includedSlugs = new Set<string>([slug, ...selectedSlugs]);
  const topPpr = pool[0]?.score ?? 1;
  const nodes: GraphNode[] = [];
  for (const s of includedSlugs) {
    const e = idx.bySlug.get(s);
    if (!e) continue;
    nodes.push({
      id: s,
      slug: s,
      title: e.title,
      category: e.category,
      subcategory: e.subcategory,
      relevance:
        s === slug
          ? 1
          : topPpr > 0
            ? Math.min(1, (ppr.get(s) ?? 0) / topPpr)
            : 0,
      image: e.image,
      publishDate: e.publishDate,
    });
  }

  // 5. Build edges among included nodes from cached adjacency.
  //    - Always keep edges incident to the centre.
  //    - Then top up with the heaviest internal edges, capped at density.
  const candidateEdges: GraphEdge[] = [];
  const seenPairs = new Set<string>();
  const arr = Array.from(includedSlugs);
  for (let i = 0; i < arr.length; i++) {
    for (let j = i + 1; j < arr.length; j++) {
      const a = arr[i];
      const b = arr[j];
      const e = idx.adjacency.get(a)?.get(b);
      if (!e) continue;
      const key = `${a}|${b}`;
      if (seenPairs.has(key)) continue;
      seenPairs.add(key);
      // Reference is directional — pick the actual link direction.
      let src = a;
      let tgt = b;
      let directed = false;
      if (e.dominant === "reference") {
        directed = true;
        if (e.reference === false) {
          // a doesn't link b — flip.
          src = b;
          tgt = a;
        }
      }
      candidateEdges.push({
        source: src,
        target: tgt,
        weight: e.weight,
        dominant: e.dominant,
        evidence: e.evidence,
        directed: directed || undefined,
      });
    }
  }
  candidateEdges.sort((a, b) => b.weight - a.weight);

  const edgeBudget = Math.round(nodes.length * EDGE_DENSITY);
  const finalEdges: GraphEdge[] = [];
  // Pass 1: keep every centre-incident edge.
  const remaining: GraphEdge[] = [];
  for (const e of candidateEdges) {
    if (e.source === slug || e.target === slug) finalEdges.push(e);
    else remaining.push(e);
  }
  // Pass 2: fill remaining budget with the heaviest non-centre edges.
  for (const e of remaining) {
    if (finalEdges.length >= edgeBudget) break;
    finalEdges.push(e);
  }

  // 6. Normalise weights within the subgraph for clean visual scaling.
  const maxW = finalEdges.reduce((m, e) => Math.max(m, e.weight), 0) || 1;
  for (const e of finalEdges) e.weight = e.weight / maxW;

  return {
    mode: "ego",
    currentNodeId: slug,
    nodes,
    edges: finalEdges,
    palette: buildPalette(new Set(nodes.map((n) => n.subcategory))),
  };
}

// ─────────────── Route handler ───────────────

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const slug = searchParams.get("slug")?.trim();
    const idx = await getIndex();

    let payload: GraphResponse;
    if (!slug) {
      payload = buildUniverseGraph(idx);
    } else {
      const ego = buildEgoGraph(slug, idx);
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
