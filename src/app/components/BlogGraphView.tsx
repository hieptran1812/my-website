"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import * as d3 from "d3";
import { useRouter } from "next/navigation";

interface GraphNode extends d3.SimulationNodeDatum {
  id: string;
  slug: string;
  title: string;
  category: string;
  subcategory: string;
  hop: number;
  relevance: number;
  image?: string;
  publishDate: string;
}

type GraphEdgeType = "series" | "reference" | "topic" | "similar";

interface GraphEdge extends d3.SimulationLinkDatum<GraphNode> {
  source: string | GraphNode;
  target: string | GraphNode;
  type: GraphEdgeType;
  weight: number;
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
  isExpanded?: boolean;
  onClose?: () => void;
  width?: number;
  height?: number;
  theme?: string;
  /** Depth of the ego BFS, 1-3. Mobile defaults to 1, desktop to 2. */
  depth?: number;
  /** Visible edge types. Defaults to all four. */
  edgeTypes?: GraphEdgeType[];
  /** Force universe (whole-corpus) mode regardless of slug. */
  universe?: boolean;
}

const EDGE_STYLE: Record<
  GraphEdgeType,
  { stroke: string; dash?: string; arrow?: boolean }
> = {
  series: { stroke: "#a855f7", arrow: true },
  reference: { stroke: "#3b82f6", arrow: true },
  topic: { stroke: "#f59e0b", dash: "4 3" },
  similar: { stroke: "#10b981" },
};

const ALL_EDGE_TYPES: GraphEdgeType[] = [
  "series",
  "reference",
  "topic",
  "similar",
];

export default function BlogGraphView({
  currentSlug,
  isExpanded = false,
  onClose,
  width: propWidth,
  height: propHeight,
  theme = "light",
  depth = 2,
  edgeTypes = ALL_EDGE_TYPES,
  universe = false,
}: BlogGraphViewProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [payload, setPayload] = useState<GraphPayload | null>(null);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [hoveredEdgeEvidence, setHoveredEdgeEvidence] =
    useState<string | null>(null);
  const [dimensions, setDimensions] = useState({ width: 300, height: 400 });
  const [isLoading, setIsLoading] = useState(true);
  const router = useRouter();

  // ─── Fetch payload ───
  useEffect(() => {
    let abort = false;
    const ctrl = new AbortController();
    setIsLoading(true);
    const params = new URLSearchParams();
    if (!universe && currentSlug) {
      params.set("slug", currentSlug);
      params.set("depth", String(Math.min(3, Math.max(1, depth))));
    }
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
  }, [currentSlug, depth, universe]);

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

  // Filter edges to active types so the simulation only sees what we draw.
  const filteredEdges = useMemo(() => {
    if (!payload) return [];
    const allow = new Set(edgeTypes);
    return payload.edges.filter((e) => allow.has(e.type));
  }, [payload, edgeTypes]);

  // ─── Render simulation ───
  useEffect(() => {
    if (!payload || !svgRef.current || payload.nodes.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    const { width, height } = dimensions;

    const nodes: GraphNode[] = payload.nodes.map((n) => ({ ...n }));
    const edges: GraphEdge[] = filteredEdges.map((e) => ({
      ...e,
      source: typeof e.source === "string" ? e.source : e.source.id,
      target: typeof e.target === "string" ? e.target : e.target.id,
    }));

    const isDark = theme === "dark";
    const centreId = payload.currentNodeId;

    // Pin centre
    const centre = nodes.find((n) => n.id === centreId);
    if (centre) {
      centre.fx = width / 2;
      centre.fy = height / 2;
    }

    // Cluster forces by subcategory: each subcategory gets a target ring point.
    const subcats = Array.from(new Set(nodes.map((n) => n.subcategory || "_")));
    const subcatTargets = new Map<string, { x: number; y: number }>();
    const radius = Math.min(width, height) * 0.32;
    subcats.forEach((s, i) => {
      const angle = (i / subcats.length) * Math.PI * 2;
      subcatTargets.set(s, {
        x: width / 2 + radius * Math.cos(angle),
        y: height / 2 + radius * Math.sin(angle),
      });
    });

    const sizeFor = (n: GraphNode) => {
      if (n.id === centreId) return isExpanded ? 18 : 13;
      if (n.hop === 1) return (isExpanded ? 8 : 6) + n.relevance * (isExpanded ? 8 : 5);
      if (n.hop === 2) return isExpanded ? 6 : 4;
      return isExpanded ? 5 : 3.5;
    };

    // Defs: arrow markers, one per edge type so the head colour matches.
    const defs = svg.append("defs");
    for (const t of ALL_EDGE_TYPES) {
      const style = EDGE_STYLE[t];
      defs
        .append("marker")
        .attr("id", `arrow-${t}`)
        .attr("viewBox", "0 -5 10 10")
        .attr("refX", 12)
        .attr("refY", 0)
        .attr("markerWidth", 6)
        .attr("markerHeight", 6)
        .attr("orient", "auto")
        .append("path")
        .attr("d", "M0,-4L8,0L0,4")
        .attr("fill", style.stroke)
        .attr("opacity", 0.85);
    }

    const root = svg.append("g");

    // Zoom & pan
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
      .attr("stroke", (d) => EDGE_STYLE[d.type].stroke)
      .attr("stroke-width", (d) => Math.max(0.6, Math.sqrt(d.weight) * 2.4))
      .attr("stroke-opacity", (d) => 0.25 + d.weight * 0.65)
      .attr("stroke-dasharray", (d) => EDGE_STYLE[d.type].dash || null)
      .attr("marker-end", (d) =>
        EDGE_STYLE[d.type].arrow ? `url(#arrow-${d.type})` : null,
      )
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
        if (!d.slug) return;
        router.push(`/blog/${d.slug}`);
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

    // Always-on label for the centre, hover-only for others.
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

    // Drag
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

    const sim = d3
      .forceSimulation<GraphNode>(nodes)
      .force(
        "link",
        d3
          .forceLink<GraphNode, GraphEdge>(edges)
          .id((d) => d.id)
          .distance((l) => 60 + (1 - l.weight) * 80)
          .strength((l) => 0.2 + l.weight * 0.6),
      )
      .force("charge", d3.forceManyBody().strength(isExpanded ? -180 : -100))
      .force("collision", d3.forceCollide<GraphNode>().radius((d) => sizeFor(d) + 4))
      .force(
        "x",
        d3
          .forceX<GraphNode>((d) => subcatTargets.get(d.subcategory || "_")?.x ?? width / 2)
          .strength((d) => (d.id === centreId ? 0 : 0.08)),
      )
      .force(
        "y",
        d3
          .forceY<GraphNode>((d) => subcatTargets.get(d.subcategory || "_")?.y ?? height / 2)
          .strength((d) => (d.id === centreId ? 0 : 0.08)),
      )
      .force("center", d3.forceCenter(width / 2, height / 2).strength(0.02));

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
  }, [payload, filteredEdges, dimensions, theme, isExpanded, router]);

  // ─── Hover dim effect (post-render DOM tweak) ───
  useEffect(() => {
    if (!svgRef.current || !payload) return;
    const svg = d3.select(svgRef.current);
    if (!hoveredId) {
      svg.selectAll<SVGElement, GraphNode>(".nodes g").attr("opacity", 1);
      svg.selectAll<SVGElement, GraphEdge>(".links line").attr("opacity", 1);
      return;
    }
    const neighbours = new Set<string>([hoveredId]);
    for (const e of filteredEdges) {
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
  }, [hoveredId, payload, filteredEdges]);

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
            {hoveredNode.hop === 0
              ? " · current"
              : ` · ${Math.round(hoveredNode.relevance * 100)}% match`}
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
