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
 * The per-slug half (PPR + MMR + force layout) is embarrassingly parallel and
 * used to run on a single core while the other nine sat idle — it was ~74% of
 * total `npm run build` wall clock. It is now sharded across child processes:
 * each child rebuilds the corpus index (the O(N²) adjacency is only ~12s and
 * cannot be cheaply serialised — the adjacency is near-complete), handles every
 * n-th slug, and writes its own part file. The parent concatenates the parts as
 * raw strings, so the merge never re-parses the ~36 MB payload.
 *
 * Resilient by design: any failure writes an empty index and exits 0 so a build
 * never breaks — the client falls back to the /api/blog/graph route. A failed
 * shard also degrades to the single-process path rather than shipping a partial
 * graph.
 *
 * Usage: tsx scripts/buildRelatedGraph.ts
 *   GRAPH_WORKERS=1  force the single-process path (debugging)
 *   GRAPH_WORKERS=N  override the auto-detected shard count
 */

import fs from "fs";
import os from "os";
import path from "path";
import { fork } from "child_process";
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

// ─────────────── Sharding ───────────────

const partPath = (shard: number): string => `${OUT_PATH}.part${shard}`;

/**
 * How many shards to run.
 *
 * Speedup saturates fast and then REVERSES, because every child pays the same
 * fixed cost — rebuilding the corpus index, which reads ~150 MB of markdown and
 * materialises a near-complete adjacency (~0.6 GB resident). Past ~4 shards the
 * duplicated index work and the memory/IO contention it causes cost more than
 * the extra parallelism buys. Measured on a 10-core / 16 GB laptop, 3.3k posts:
 *
 *   shards:  1      2       3        4       6       8
 *   wall:    145s   103s    64-88s   75s     97s     92s
 *
 * Hence the hard cap of 4 rather than one-per-core. The ~2 GB-per-shard
 * allowance additionally keeps a smaller CI builder (4 cores / 8 GB) at 3.
 * Override with GRAPH_WORKERS if a particular machine profiles differently.
 */
const MAX_SHARDS = 4;

function shardCount(): number {
  const override = Number(process.env.GRAPH_WORKERS);
  if (Number.isFinite(override) && override >= 1) return Math.floor(override);
  const byCpu = Math.max(1, os.cpus().length - 1);
  const byMem = Math.max(1, Math.floor(os.totalmem() / (2 * 1024 ** 3)));
  return Math.min(byCpu, byMem, MAX_SHARDS);
}

interface ShardResult {
  map: Record<string, StoredGraph>;
  done: number;
  skipped: number;
}

/**
 * Compute every `count`-th ego graph starting at `shard`. Builds its own corpus
 * index — cheap relative to the per-slug work it unlocks (~12s of index for
 * ~180s/shards of PPR + MMR + layout).
 */
async function computeShard(shard: number, count: number): Promise<ShardResult> {
  const idx = await getIndex();
  const transition = buildPprTransition(idx);
  const slugs = Array.from(idx.bySlug.keys());

  const map: Record<string, StoredGraph> = {};
  let done = 0;
  let skipped = 0;
  for (let i = shard; i < slugs.length; i += count) {
    const slug = slugs[i];
    const ego = buildEgoGraph(slug, idx, transition);
    if (!ego || ego.nodes.length <= 1) {
      skipped++;
      continue;
    }
    map[slug] = toStored(ego);
    done++;
  }
  return { map, done, skipped };
}

/** Child-process entry: compute one shard and write it as its own JSON file. */
async function runAsShard(shard: number, count: number): Promise<void> {
  const { map, done, skipped } = await computeShard(shard, count);
  fs.mkdirSync(path.dirname(OUT_PATH), { recursive: true });
  fs.writeFileSync(partPath(shard), JSON.stringify(map));
  process.send?.({ shard, done, skipped });
}

/**
 * Concatenate the shard files into the final index without re-parsing them.
 * Each part is a complete JSON object, so stripping the outer braces and joining
 * with commas rebuilds the whole map as a string — the keys are slugs and the
 * shards are disjoint, so no dedup is needed.
 */
function mergeParts(count: number): number {
  const bodies: string[] = [];
  for (let i = 0; i < count; i++) {
    const raw = fs.readFileSync(partPath(i), "utf8").trim();
    const body = raw.slice(1, -1); // drop the enclosing { }
    if (body.length > 0) bodies.push(body);
  }
  fs.mkdirSync(path.dirname(OUT_PATH), { recursive: true });
  fs.writeFileSync(OUT_PATH, `{${bodies.join(",")}}`);
  for (let i = 0; i < count; i++) {
    try {
      fs.unlinkSync(partPath(i));
    } catch {
      /* ignore */
    }
  }
  return bodies.length;
}

/** Run `count` shards as child processes. `ok` is false if any shard failed. */
function runShardedChildren(
  count: number,
): Promise<{ ok: boolean; done: number; skipped: number }> {
  return new Promise((resolve) => {
    let finished = 0;
    let failed = false;
    let done = 0;
    let skipped = 0;

    for (let shard = 0; shard < count; shard++) {
      const child = fork(__filename, [], {
        env: { ...process.env, GRAPH_SHARD: `${shard}:${count}` },
        stdio: ["ignore", "inherit", "inherit", "ipc"],
      });
      child.on("message", (msg: { done?: number; skipped?: number }) => {
        done += msg.done ?? 0;
        skipped += msg.skipped ?? 0;
      });
      child.on("error", () => {
        failed = true;
      });
      child.on("exit", (code) => {
        if (code !== 0) {
          failed = true;
          console.error(`   ⚠️  shard ${shard} exited with code ${code}`);
        } else {
          console.log(`   …shard ${shard + 1}/${count} done`);
        }
        if (++finished === count) resolve({ ok: !failed, done, skipped });
      });
    }
  });
}

async function main() {
  // ── Child mode ──
  const shardSpec = process.env.GRAPH_SHARD;
  if (shardSpec) {
    const [shard, count] = shardSpec.split(":").map(Number);
    await runAsShard(shard, count);
    return;
  }

  // ── Parent mode ──
  const t0 = Date.now();
  const shards = shardCount();
  console.log(
    `🕸️  Precomputing related-articles ego graphs (${shards} shard${
      shards === 1 ? "" : "s"
    })…`,
  );

  let done = 0;
  let skipped = 0;
  let wroteViaShards = false;

  if (shards > 1) {
    const res = await runShardedChildren(shards);
    if (res.ok) {
      const written = mergeParts(shards);
      if (written === 0) throw new Error("all shards produced empty output");
      done = res.done;
      skipped = res.skipped;
      wroteViaShards = true;
    } else {
      console.error(
        "   ⚠️  a shard failed — falling back to the single-process path",
      );
      for (let i = 0; i < shards; i++) {
        try {
          fs.unlinkSync(partPath(i));
        } catch {
          /* ignore */
        }
      }
    }
  }

  // Single-process path: chosen explicitly (GRAPH_WORKERS=1) or as the fallback
  // after a shard failure.
  if (!wroteViaShards) {
    const res = await computeShard(0, 1);
    writeIndex(res.map);
    done = res.done;
    skipped = res.skipped;
  }

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
