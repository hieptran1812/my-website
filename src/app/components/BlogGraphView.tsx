"use client";

import React, { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import { useRouter } from "next/navigation";
import type { PrecomputedGraph } from "../../lib/blogGraphIndex";

type DominantSignal = "series" | "reference" | "tags" | "similar" | "structural";

interface GraphNode extends d3.SimulationNodeDatum {
  id: string;
  slug: string;
  title: string;
  category: string;
  subcategory: string;
  relevance: number;
  image?: string;
  publishDate?: string;
  /** Precomputed layout position (origin-centred), present when the graph was
   *  inlined at build time. Lets the render seed a settled layout. */
  seedX?: number;
  seedY?: number;
}

interface GraphEdge extends d3.SimulationLinkDatum<GraphNode> {
  source: string | GraphNode;
  target: string | GraphNode;
  weight: number;
  dominant: DominantSignal;
  evidence?: string;
  directed?: boolean;
}

interface GraphPayload {
  mode: "ego" | "universe";
  currentNodeId?: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
  palette: Record<string, string>;
}

interface BlogGraphViewProps {
  currentSlug?: string;
  /** Ego graph precomputed at build time and inlined by the server. When present
   *  (and not universe mode), the view renders it immediately — no fetch, no
   *  spinner — and seeds the simulation with its settled positions. */
  initialGraph?: PrecomputedGraph | null;
  isExpanded?: boolean;
  onClose?: () => void;
  width?: number;
  height?: number;
  theme?: string;
  /** Force universe (whole-corpus) mode regardless of slug. */
  universe?: boolean;
}

/** Adapt an inlined PrecomputedGraph into the view's payload shape. The stored
 *  form is deduped for size: nodes carry no `id` (== slug) and edges reference
 *  endpoints by node index — we rehydrate both here. Positions cross over as
 *  seedX/seedY (kept separate from d3's live x/y so the simulation can offset
 *  them to the container centre). */
function adaptInitial(g: PrecomputedGraph): GraphPayload {
  const nodes: GraphNode[] = g.nodes.map((n) => ({
    id: n.slug,
    slug: n.slug,
    title: n.title,
    category: n.category,
    subcategory: n.subcategory,
    relevance: n.relevance,
    seedX: n.x,
    seedY: n.y,
  }));
  return {
    mode: g.mode,
    currentNodeId: g.nodes[g.centre]?.slug,
    palette: g.palette,
    nodes,
    edges: g.edges.map((e) => ({
      source: nodes[e.source]?.slug ?? "",
      target: nodes[e.target]?.slug ?? "",
      weight: e.weight,
      dominant: e.dominant,
      evidence: e.evidence,
      directed: e.directed,
    })),
  };
}

/**
 * Edge stroke colour blended subtly by dominant signal so the visual stays
 * unified (one rendering style) but gives a hint of why the edge exists.
 * Reference dominates when present; otherwise the colour follows the
 * underlying signal mass.
 */
function edgeColor(d: GraphEdge, isDark: boolean): string {
  const base = isDark ? "rgba(255,255,255," : "rgba(15,23,42,";
  // alpha scaled by weight, capped 0.85
  const alpha = (0.18 + d.weight * 0.6).toFixed(2);
  switch (d.dominant) {
    case "series":
      return `hsla(270, 70%, 65%, ${alpha})`;
    case "reference":
      return `hsla(220, 80%, 60%, ${alpha})`;
    case "tags":
      return `hsla(38, 85%, 60%, ${alpha})`;
    case "similar":
      return `hsla(160, 65%, 50%, ${alpha})`;
    default:
      return `${base}${alpha})`;
  }
}

export default function BlogGraphView({
  currentSlug,
  initialGraph = null,
  isExpanded = false,
  onClose,
  width: propWidth,
  height: propHeight,
  theme = "light",
  universe = false,
}: BlogGraphViewProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  // Seed synchronously from inlined data (ego mode only) so the very first paint
  // already has the graph — no fetch, no spinner.
  const seeded = !universe && initialGraph ? adaptInitial(initialGraph) : null;
  const [payload, setPayload] = useState<GraphPayload | null>(seeded);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [hoveredEdgeEvidence, setHoveredEdgeEvidence] = useState<string | null>(
    null,
  );
  const [dimensions, setDimensions] = useState({ width: 300, height: 400 });
  const [isLoading, setIsLoading] = useState(!seeded);
  const router = useRouter();

  // ─── Fetch payload (fallback) ───
  // When an ego graph was inlined, skip the network entirely. We still fetch for
  // universe mode and when nothing was precomputed (dev / missing index).
  useEffect(() => {
    if (!universe && initialGraph) {
      setPayload(adaptInitial(initialGraph));
      setIsLoading(false);
      return;
    }
    let abort = false;
    const ctrl = new AbortController();
    setIsLoading(true);
    const params = new URLSearchParams();
    if (!universe && currentSlug) params.set("slug", currentSlug);
    fetch(`/api/blog/graph${params.toString() ? `?${params}` : ""}`, {
      signal: ctrl.signal,
    })
      .then((r) => (r.ok ? r.json() : Promise.reject(r.statusText)))
      .then((data: GraphPayload) => {
        if (!abort) setPayload(data);
      })
      .catch((err) => {
        if ((err as Error).name !== "AbortError")
          console.error("graph fetch failed", err);
      })
      .finally(() => {
        if (!abort) setIsLoading(false);
      });
    return () => {
      abort = true;
      ctrl.abort();
    };
  }, [currentSlug, universe, initialGraph]);

  // ─── Resize observer ───
  useEffect(() => {
    const update = () => {
      if (isExpanded) {
        setDimensions({
          width: Math.min(window.innerWidth - 32, 1400),
          height: Math.min(window.innerHeight - 120, 900),
        });
      } else if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        setDimensions({
          width: Math.max(240, rect.width || propWidth || 280),
          height: propHeight ?? 280,
        });
      }
    };
    update();
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, [isExpanded, propWidth, propHeight]);

  // ─── Render simulation ───
  useEffect(() => {
    if (!payload || !svgRef.current || payload.nodes.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    const { width, height } = dimensions;

    const nodes: GraphNode[] = payload.nodes.map((n) => ({ ...n }));
    const edges: GraphEdge[] = payload.edges.map((e) => ({
      ...e,
      source: typeof e.source === "string" ? e.source : e.source.id,
      target: typeof e.target === "string" ? e.target : e.target.id,
    }));

    const isDark = theme === "dark";
    const centreId = payload.currentNodeId;

    // Seed from precomputed positions (inline view only). The build lays graphs
    // out for the compact inline box; for the much larger fullscreen we let the
    // sim re-expand from scratch. Positions are origin-centred → offset to the
    // container centre so they line up with the pinned centre + forceCenter.
    const hasSeeds =
      !isExpanded &&
      nodes.length > 0 &&
      nodes.every((n) => typeof n.seedX === "number" && typeof n.seedY === "number");
    if (hasSeeds) {
      for (const n of nodes) {
        n.x = (n.seedX as number) + width / 2;
        n.y = (n.seedY as number) + height / 2;
      }
    }

    // Pin centre
    const centre = nodes.find((n) => n.id === centreId);
    if (centre) {
      centre.fx = width / 2;
      centre.fy = height / 2;
    }

    const sizeFor = (n: GraphNode) => {
      if (n.id === centreId) return isExpanded ? 18 : 14;
      // size by PPR relevance (logarithmic-ish curve)
      const base = isExpanded ? 6 : 4.5;
      return base + Math.log1p(n.relevance * 8) * (isExpanded ? 4.5 : 3.2);
    };

    const root = svg.append("g");

    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.4, 4])
      .on("zoom", (e) => root.attr("transform", e.transform.toString()));
    svg.call(zoom);

    const linkSel = root
      .append("g")
      .attr("class", "links")
      .selectAll<SVGLineElement, GraphEdge>("line")
      .data(edges)
      .enter()
      .append("line")
      .attr("stroke", (d) => edgeColor(d, isDark))
      .attr("stroke-width", (d) => Math.max(0.6, Math.sqrt(d.weight) * 2.6))
      .on("mouseenter", (_, d) => setHoveredEdgeEvidence(d.evidence ?? null))
      .on("mouseleave", () => setHoveredEdgeEvidence(null));

    const nodeSel = root
      .append("g")
      .attr("class", "nodes")
      .selectAll<SVGGElement, GraphNode>("g")
      .data(nodes)
      .enter()
      .append("g")
      .attr("cursor", "pointer")
      .on("click", (_, d) => {
        if (d.slug) router.push(`/blog/${d.slug}`);
      })
      .on("mouseenter", (_, d) => setHoveredId(d.id))
      .on("mouseleave", () => setHoveredId(null));

    nodeSel
      .append("circle")
      .attr("r", sizeFor)
      .attr("fill", (d) => payload.palette[d.subcategory] || "#6b7280")
      .attr("stroke", (d) =>
        d.id === centreId
          ? isDark
            ? "#fff"
            : "#0f172a"
          : isDark
            ? "rgba(255,255,255,0.4)"
            : "rgba(15,23,42,0.25)",
      )
      .attr("stroke-width", (d) => (d.id === centreId ? 2.5 : 1));

    nodeSel
      .filter((d) => d.id === centreId)
      .append("text")
      .attr("y", (d) => sizeFor(d) + 14)
      .attr("text-anchor", "middle")
      .attr("font-size", isExpanded ? 12 : 10)
      .attr("font-weight", 600)
      .attr("fill", isDark ? "#f1f5f9" : "#0f172a")
      .text((d) =>
        d.title.length > 38 ? d.title.slice(0, 36) + "…" : d.title,
      );

    const drag = d3
      .drag<SVGGElement, GraphNode>()
      .on("start", (e, d) => {
        if (!e.active) sim.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      })
      .on("drag", (e, d) => {
        d.fx = e.x;
        d.fy = e.y;
      })
      .on("end", (e, d) => {
        if (!e.active) sim.alphaTarget(0);
        if (d.id !== centreId) {
          d.fx = null;
          d.fy = null;
        }
      });
    nodeSel.call(drag);

    // Pure force-directed layout — physics + edge weights determine structure.
    const sim = d3
      .forceSimulation<GraphNode>(nodes)
      .force(
        "link",
        d3
          .forceLink<GraphNode, GraphEdge>(edges)
          .id((d) => d.id)
          .distance((l) => 50 + (1 - l.weight) * 90)
          .strength((l) => 0.15 + l.weight * 0.7),
      )
      .force("charge", d3.forceManyBody().strength(isExpanded ? -200 : -110))
      .force(
        "collision",
        d3.forceCollide<GraphNode>().radius((d) => sizeFor(d) + 4),
      )
      .force("center", d3.forceCenter(width / 2, height / 2).strength(0.04))
      // Seeded near the settled layout → start cool so it appears already-formed
      // and only micro-settles instead of exploding out from the centre.
      .alpha(hasSeeds ? 0.3 : 1);

    sim.on("tick", () => {
      linkSel
        .attr("x1", (d) => (d.source as GraphNode).x ?? 0)
        .attr("y1", (d) => (d.source as GraphNode).y ?? 0)
        .attr("x2", (d) => (d.target as GraphNode).x ?? 0)
        .attr("y2", (d) => (d.target as GraphNode).y ?? 0);
      nodeSel.attr("transform", (d) => `translate(${d.x ?? 0},${d.y ?? 0})`);
    });

    return () => {
      sim.stop();
    };
  }, [payload, dimensions, theme, isExpanded, router]);

  // ─── Hover dim ───
  useEffect(() => {
    if (!svgRef.current || !payload) return;
    const svg = d3.select(svgRef.current);
    if (!hoveredId) {
      svg.selectAll<SVGElement, GraphNode>(".nodes g").attr("opacity", 1);
      svg.selectAll<SVGElement, GraphEdge>(".links line").attr("opacity", 1);
      return;
    }
    const neighbours = new Set<string>([hoveredId]);
    for (const e of payload.edges) {
      const s = typeof e.source === "string" ? e.source : (e.source as GraphNode).id;
      const t = typeof e.target === "string" ? e.target : (e.target as GraphNode).id;
      if (s === hoveredId) neighbours.add(t);
      if (t === hoveredId) neighbours.add(s);
    }
    svg
      .selectAll<SVGGElement, GraphNode>(".nodes g")
      .attr("opacity", (d) => (neighbours.has(d.id) ? 1 : 0.18));
    svg
      .selectAll<SVGLineElement, GraphEdge>(".links line")
      .attr("opacity", (d) => {
        const s = typeof d.source === "string" ? d.source : (d.source as GraphNode).id;
        const t = typeof d.target === "string" ? d.target : (d.target as GraphNode).id;
        return s === hoveredId || t === hoveredId ? 1 : 0.08;
      });
  }, [hoveredId, payload]);

  const hoveredNode = hoveredId
    ? payload?.nodes.find((n) => n.id === hoveredId)
    : null;

  return (
    <div
      ref={containerRef}
      className="relative w-full"
      style={{
        height: isExpanded ? "100%" : "280px",
        background: theme === "dark" ? "#011627" : "#ffffff",
      }}
    >
      {isLoading && (
        <div
          className="absolute inset-0 flex items-center justify-center"
          style={{ color: "var(--text-secondary)" }}
        >
          <div className="flex flex-col items-center gap-2">
            <div
              className="w-8 h-8 border-2 border-t-transparent rounded-full animate-spin"
              style={{ borderColor: "var(--accent)", borderTopColor: "transparent" }}
            />
            <span className="text-xs">Building graph…</span>
          </div>
        </div>
      )}
      <svg ref={svgRef} width={dimensions.width} height={dimensions.height} />
      {hoveredNode && (
        <div
          className="absolute pointer-events-none rounded-md border px-2.5 py-1.5 text-xs"
          style={{
            background: "var(--background)",
            borderColor: "var(--border)",
            color: "var(--text-primary)",
            top: 8,
            left: 8,
            maxWidth: dimensions.width - 16,
            boxShadow: "0 8px 24px -10px rgba(0,0,0,0.25)",
          }}
        >
          <div style={{ fontWeight: 600 }}>{hoveredNode.title}</div>
          <div style={{ fontSize: 10, color: "var(--text-secondary)" }}>
            {[hoveredNode.subcategory, hoveredNode.category]
              .filter(Boolean)
              .join(" · ")}
            {hoveredNode.id === payload?.currentNodeId
              ? " · current"
              : ` · ${Math.round(hoveredNode.relevance * 100)}% rank`}
          </div>
        </div>
      )}
      {hoveredEdgeEvidence && !hoveredNode && (
        <div
          className="absolute pointer-events-none rounded-md border px-2.5 py-1 text-[11px]"
          style={{
            background: "var(--background)",
            borderColor: "var(--border)",
            color: "var(--text-secondary)",
            bottom: 8,
            left: 8,
          }}
        >
          {hoveredEdgeEvidence}
        </div>
      )}
      {onClose && isExpanded && (
        <button
          onClick={onClose}
          className="absolute top-3 right-3 rounded-md px-2 py-1 text-xs"
          style={{
            background: "var(--surface)",
            border: "1px solid var(--border)",
            color: "var(--text-secondary)",
          }}
        >
          Close
        </button>
      )}
    </div>
  );
}
