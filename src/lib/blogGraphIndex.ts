/**
 * Runtime loader for the precomputed per-article ego graphs.
 *
 * `scripts/buildRelatedGraph.ts` writes `src/lib/generated/blogGraph.json`
 * (slug → settled ego graph) per deploy. The article server component reads its
 * slug's entry here and inlines it into the page, so the graph renders instantly
 * with no client fetch and no O(N²) server compute.
 *
 * Resolution:
 *   - Production + a non-empty generated file → return the precomputed graph.
 *   - Dev, or missing/empty file, or slug absent → return null. The client then
 *     falls back to the /api/blog/graph route (which rebuilds live).
 *
 * The file is parsed once and cached in module scope. It is only ever touched
 * during static generation / ISR revalidation — never on a user-facing served
 * request, which gets the already-inlined HTML.
 */

import fs from "fs";
import path from "path";

export interface PrecomputedGraphNode {
  slug: string;
  title: string;
  category: string;
  subcategory: string;
  relevance: number;
  /** Origin-centred layout position from the build-time force pass. */
  x: number;
  y: number;
}

export type DominantSignal =
  | "series"
  | "reference"
  | "tags"
  | "similar"
  | "structural";

export interface PrecomputedGraphEdge {
  /** Index into `nodes` of the source / target (deduped to keep the payload
   *  small — see scripts/buildRelatedGraph.ts). */
  source: number;
  target: number;
  weight: number;
  dominant: DominantSignal;
  evidence: string;
  directed?: boolean;
}

export interface PrecomputedGraph {
  mode: "ego";
  /** Index into `nodes` of the centre article. */
  centre: number;
  nodes: PrecomputedGraphNode[];
  edges: PrecomputedGraphEdge[];
  palette: Record<string, string>;
}

const GENERATED_PATH = path.join(
  process.cwd(),
  "src",
  "lib",
  "generated",
  "blogGraph.json",
);

// `undefined` = not yet attempted; `null` = attempted, unavailable.
let cache: Record<string, PrecomputedGraph> | null | undefined;

function loadAll(): Record<string, PrecomputedGraph> | null {
  if (cache !== undefined) return cache;
  // In dev we never precompute — always let the client fall back to the live
  // API so freshly added/edited posts get an up-to-date graph without a rebuild.
  // Set FORCE_PRECOMPUTED_GRAPH=1 to exercise the production (inlined) path
  // locally against the generated file — used for prod-parity testing.
  if (process.env.NODE_ENV !== "production" && !process.env.FORCE_PRECOMPUTED_GRAPH) {
    cache = null;
    return null;
  }
  try {
    const raw = fs.readFileSync(GENERATED_PATH, "utf8");
    const parsed = JSON.parse(raw) as Record<string, PrecomputedGraph>;
    cache = parsed && typeof parsed === "object" ? parsed : null;
  } catch {
    cache = null;
  }
  return cache;
}

/** Return the precomputed ego graph for `slug`, or null if unavailable. */
export function getPrecomputedGraph(slug: string): PrecomputedGraph | null {
  const all = loadAll();
  if (!all) return null;
  return all[slug] ?? null;
}
