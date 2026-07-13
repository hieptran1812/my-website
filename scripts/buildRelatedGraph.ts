/**
 * Precompute the per-article "Related Articles" ego graph for every post.
 *
 * The graph the sidebar shows is derived from an O(N²) composite-weighted
 * adjacency over the WHOLE corpus (tag IDF overlap + TF-IDF cosine + markdown
 * reference graph + series/structure/time), then Personalized-PageRanked and
 * MMR-diversified per article. Doing that at request time meant: a client fetch
 * to /api/blog/graph → the server rebuilding the entire adjacency (multi-second
 * on a cold start with ~3k posts) → PPR + MMR. The graph only appeared after a
 * spinner and a network round-trip.
 *
 * This script does all of it ONCE per deploy: build the corpus index, compute
 * each slug's ego graph, run a headless d3-force pass so nodes ship already
 * laid-out, trim the payload to what the renderer actually uses, and write
 * `src/lib/generated/blogGraph.json` keyed by slug. The article page inlines the
 * entry for its slug (see src/lib/blogGraphIndex.ts), so the graph renders
 * instantly — no fetch, no server compute, no spinner.
 *
 * Resilient by design: any failure writes an empty index and exits 0 so a build
 * never breaks — the client falls back to the /api/blog/graph route.
 *
 * Usage: tsx scripts/buildRelatedGraph.ts
 */

import fs from "fs";
import path from "path";
import {
  forceSimulation,
  forceLink,
  forceManyBody,
  forceCollide,
  forceCenter,
  type SimulationNodeDatum,
  type SimulationLinkDatum,
} from "d3-force";
import { getIndex, buildPprTransition } from "../src/lib/getRelatedPosts";
import { buildEgoGraph, type GraphResponse } from "../src/lib/blogGraph";

const OUT_PATH = path.join(
  process.cwd(),
  "src",
  "lib",
  "generated",
  "blogGraph.json",
);

// Trimmed payload actually consumed by BlogGraphView. Every node's slug is
// otherwise repeated ~4× (as id, as slug, and once per edge endpoint), so we:
//   - drop `id` (== slug; reconstructed on the client),
//   - drop `image`/`publishDate` (never rendered in the graph),
//   - reference edge endpoints by node INDEX instead of by slug string.
// This roughly halves both the shipped file and the per-page inlined payload.
interface StoredNode {
  slug: string;
  title: string;
  category: string;
  subcategory: string;
  relevance: number;
  x: number;
  y: number;
}
interface StoredEdge {
  /** Index into `nodes` of the source / target. */
  source: number;
  target: number;
  weight: number;
  dominant: GraphResponse["edges"][number]["dominant"];
  evidence: string;
  directed?: boolean;
}
interface StoredGraph {
  mode: "ego";
  /** Index into `nodes` of the centre article. */
  centre: number;
  nodes: StoredNode[];
  edges: StoredEdge[];
  palette: Record<string, string>;
}

// Mirror the inline (non-expanded) client force config so the seeded layout
// matches what BlogGraphView would otherwise settle into. Positions are centred
// on the origin (centre node pinned at 0,0); the client offsets by the container
// centre at mount time.
const LAYOUT_TICKS = 320;

function sizeFor(relevance: number, isCentre: boolean): number {
  if (isCentre) return 14;
  return 4.5 + Math.log1p(relevance * 8) * 3.2;
}

type LNode = SimulationNodeDatum & {
  id: string;
  relevance: number;
  isCentre: boolean;
};
type LEdge = SimulationLinkDatum<LNode> & { weight: number };

/** Run a headless force simulation and return slug → {x, y}, rounded. */
function layout(graph: GraphResponse): Map<string, { x: number; y: number }> {
  const nodes: LNode[] = graph.nodes.map((n) => ({
    id: n.id,
    relevance: n.relevance,
    isCentre: n.id === graph.currentNodeId,
  }));
  const links: LEdge[] = graph.edges.map((e) => ({
    source: e.source,
    target: e.target,
    weight: e.weight,
  }));

  const centre = nodes.find((n) => n.isCentre);
  if (centre) {
    centre.fx = 0;
    centre.fy = 0;
  }

  const sim = forceSimulation<LNode>(nodes)
    .force(
      "link",
      forceLink<LNode, LEdge>(links)
        .id((d) => d.id)
        .distance((l) => 50 + (1 - l.weight) * 90)
        .strength((l) => 0.15 + l.weight * 0.7),
    )
    .force("charge", forceManyBody<LNode>().strength(-110))
    .force(
      "collision",
      forceCollide<LNode>().radius((d) => sizeFor(d.relevance, d.isCentre) + 4),
    )
    .force("center", forceCenter<LNode>(0, 0).strength(0.04))
    .stop();

  sim.tick(LAYOUT_TICKS);

  const pos = new Map<string, { x: number; y: number }>();
  for (const n of nodes) {
    pos.set(n.id, {
      x: Math.round((n.x ?? 0) * 10) / 10,
      y: Math.round((n.y ?? 0) * 10) / 10,
    });
  }
  return pos;
}

function toStored(graph: GraphResponse): StoredGraph {
  const pos = layout(graph);
  const idxOf = new Map<string, number>();
  graph.nodes.forEach((n, i) => idxOf.set(n.id, i));
  return {
    mode: "ego",
    centre: idxOf.get(graph.currentNodeId!) ?? 0,
    nodes: graph.nodes.map((n) => {
      const p = pos.get(n.id) ?? { x: 0, y: 0 };
      return {
        slug: n.slug,
        title: n.title,
        category: n.category,
        subcategory: n.subcategory,
        relevance: Math.round(n.relevance * 1000) / 1000,
        x: p.x,
        y: p.y,
      };
    }),
    edges: graph.edges.map((e) => ({
      source: idxOf.get(e.source) ?? 0,
      target: idxOf.get(e.target) ?? 0,
      weight: Math.round(e.weight * 1000) / 1000,
      dominant: e.dominant,
      evidence: e.evidence,
      ...(e.directed ? { directed: true } : {}),
    })),
    palette: graph.palette,
  };
}

function writeIndex(map: Record<string, StoredGraph>): void {
  fs.mkdirSync(path.dirname(OUT_PATH), { recursive: true });
  fs.writeFileSync(OUT_PATH, JSON.stringify(map));
}

async function main() {
  console.log("🕸️  Precomputing related-articles ego graphs…");
  const t0 = Date.now();
  const idx = await getIndex();
  const slugs = Array.from(idx.bySlug.keys());
  console.log(
    `   corpus index ready: ${slugs.length} posts, adjacency built in ${(
      (Date.now() - t0) /
      1000
    ).toFixed(1)}s`,
  );

  // Build the row-normalised transition ONCE and reuse it for every seed.
  const transition = buildPprTransition(idx);

  const map: Record<string, StoredGraph> = {};
  let done = 0;
  let skipped = 0;
  for (const slug of slugs) {
    const ego = buildEgoGraph(slug, idx, transition);
    if (!ego || ego.nodes.length <= 1) {
      skipped++;
      continue;
    }
    map[slug] = toStored(ego);
    done++;
    if (done % 500 === 0) console.log(`   …${done}/${slugs.length}`);
  }

  writeIndex(map);
  const bytes = fs.statSync(OUT_PATH).size;
  console.log(
    `✅ Wrote ${done} ego graphs (${skipped} skipped) → ${path.relative(
      process.cwd(),
      OUT_PATH,
    )} (${(bytes / 1024 / 1024).toFixed(1)} MB) in ${(
      (Date.now() - t0) /
      1000
    ).toFixed(1)}s`,
  );
}

main().catch((err) => {
  console.error(
    "⚠️  buildRelatedGraph failed; writing empty index (client falls back to /api/blog/graph):",
    err,
  );
  try {
    writeIndex({});
  } catch {
    /* ignore */
  }
  process.exit(0);
});
