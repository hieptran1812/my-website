import { NextRequest, NextResponse } from "next/server";
import { getIndex } from "@/lib/getRelatedPosts";
import {
  buildEgoGraph,
  buildUniverseGraph,
  type GraphResponse,
} from "@/lib/blogGraph";

// Re-export the graph payload types so existing importers keep a stable path.
export type {
  GraphNode,
  GraphEdge,
  GraphResponse,
} from "@/lib/blogGraph";

// ─────────────── Route handler ───────────────
//
// This route is the runtime FALLBACK for the graph. In production the article
// page inlines a precomputed ego graph (scripts/buildRelatedGraph.ts →
// getPrecomputedGraph), so the client never has to hit this endpoint on a normal
// article view. It still serves: (a) dev, where nothing is precomputed, and
// (b) universe mode, which the precompute intentionally skips.

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
