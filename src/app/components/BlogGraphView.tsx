"use client";

import React, {
  useEffect,
  useRef,
  useState,
  useCallback,
  useMemo,
} from "react";
import * as d3 from "d3";
import { useRouter } from "next/navigation";

interface GraphNode extends d3.SimulationNodeDatum {
  id: string;
  slug: string;
  title: string;
  tags: string[];
  category: string;
  group: string;
  connections: number;
}

interface GraphLink extends d3.SimulationLinkDatum<GraphNode> {
  source: string | GraphNode;
  target: string | GraphNode;
  type: "reference" | "tag";
}

interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
  tagGroups: { [tag: string]: string[] };
  tagColors: { [tag: string]: string };
}

interface BlogGraphViewProps {
  currentSlug?: string;
  isExpanded?: boolean;
  onClose?: () => void;
  width?: number;
  height?: number;
  theme?: string;
}

// Color palette for tag clusters - matches API
const TAG_COLORS = [
  "#f97316", // orange
  "#22c55e", // green
  "#3b82f6", // blue
  "#a855f7", // purple
  "#ef4444", // red
  "#eab308", // yellow
  "#14b8a6", // teal
  "#ec4899", // pink
  "#6366f1", // indigo
  "#84cc16", // lime
  "#f43f5e", // rose
  "#06b6d4", // cyan
  "#8b5cf6", // violet
  "#10b981", // emerald
  "#f59e0b", // amber
];

export default function BlogGraphView({
  currentSlug,
  isExpanded = false,
  onClose,
  width: propWidth,
  height: propHeight,
  theme = "light",
}: BlogGraphViewProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null);
  const [dimensions, setDimensions] = useState({ width: 300, height: 400 });
  const [isLoading, setIsLoading] = useState(true);
  const router = useRouter();

  // Store theme in ref to avoid re-running simulation on theme change
  const themeRef = useRef(theme);
  themeRef.current = theme;

  // Fetch graph data
  useEffect(() => {
    const fetchGraphData = async () => {
      try {
        setIsLoading(true);
        const response = await fetch("/api/blog/graph");
        if (response.ok) {
          const data = await response.json();
          setGraphData(data);
        }
      } catch (error) {
        console.error("Error fetching graph data:", error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchGraphData();
  }, []);

  // Update dimensions on resize
  useEffect(() => {
    const updateDimensions = () => {
      if (isExpanded) {
        // Fullscreen mode
        setDimensions({
          width: propWidth || window.innerWidth * 0.9,
          height: propHeight || window.innerHeight * 0.85,
        });
      } else if (containerRef.current) {
        // Sidebar mode
        const rect = containerRef.current.getBoundingClientRect();
        setDimensions({
          width: propWidth || rect.width || 280,
          height: propHeight || 350,
        });
      }
    };

    updateDimensions();
    window.addEventListener("resize", updateDimensions);
    return () => window.removeEventListener("resize", updateDimensions);
  }, [isExpanded, propWidth, propHeight]);

  // Get node color based on primary tag
  const getNodeColor = useCallback(
    (node: GraphNode): string => {
      if (!graphData?.tagColors) return TAG_COLORS[0];

      const primaryTag = node.group || node.tags[0];
      if (primaryTag && graphData.tagColors[primaryTag]) {
        return graphData.tagColors[primaryTag];
      }

      // Fallback: hash the category to get consistent color
      const hash = node.category
        .split("")
        .reduce((acc, char) => acc + char.charCodeAt(0), 0);
      return TAG_COLORS[hash % TAG_COLORS.length];
    },
    [graphData?.tagColors]
  );

  // Get node size based on connections
  const getNodeSize = useCallback(
    (node: GraphNode): number => {
      const baseSize = isExpanded ? 6 : 4;
      const maxConnections = Math.max(
        ...(graphData?.nodes.map((n) => n.connections) || [1])
      );
      const scale = Math.min(node.connections / Math.max(maxConnections, 1), 1);
      return baseSize + scale * (isExpanded ? 10 : 6);
    },
    [graphData?.nodes, isExpanded]
  );

  // D3 Force simulation
  useEffect(() => {
    if (!graphData || !svgRef.current || graphData.nodes.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const { width, height } = dimensions;

    // Create a deep copy of nodes and links for D3
    const nodes: GraphNode[] = graphData.nodes.map((n) => ({ ...n }));
    const links: GraphLink[] = graphData.links.map((l) => ({ ...l }));

    // Create zoom behavior
    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.3, 4])
      .on("zoom", (event) => {
        container.attr("transform", event.transform);
      });

    svg.call(zoom);

    // Create container for zoom/pan
    const container = svg.append("g");

    // Create force simulation
    const simulation = d3
      .forceSimulation<GraphNode>(nodes)
      .force(
        "link",
        d3
          .forceLink<GraphNode, GraphLink>(links)
          .id((d) => d.id)
          .distance((d) => (d.type === "reference" ? 80 : 120))
          .strength((d) => (d.type === "reference" ? 0.8 : 0.2))
      )
      .force("charge", d3.forceManyBody().strength(isExpanded ? -150 : -80))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius((d) => getNodeSize(d as GraphNode) + 3))
      .force("x", d3.forceX(width / 2).strength(0.05))
      .force("y", d3.forceY(height / 2).strength(0.05));

    // Theme-based colors (use ref to get current theme)
    const currentTheme = themeRef.current;
    const linkColorRef = currentTheme === "dark" ? "#6b7280" : "#9ca3af";
    const linkColorTag = currentTheme === "dark" ? "#374151" : "#d1d5db";

    // Create links
    const link = container
      .append("g")
      .attr("class", "links")
      .selectAll("line")
      .data(links)
      .join("line")
      .attr("stroke", (d) => (d.type === "reference" ? linkColorRef : linkColorTag))
      .attr("stroke-opacity", (d) => (d.type === "reference" ? 0.6 : 0.2))
      .attr("stroke-width", (d) => (d.type === "reference" ? 2 : 1));

    // Create drag behavior
    const dragBehavior = d3
      .drag<SVGGElement, GraphNode>()
      .on("start", (event, d) => {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      })
      .on("drag", (event, d) => {
        d.fx = event.x;
        d.fy = event.y;
      })
      .on("end", (event, d) => {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      });

    // Create node group
    const nodeGroup = container
      .append("g")
      .attr("class", "nodes")
      .selectAll<SVGGElement, GraphNode>("g")
      .data(nodes)
      .join("g")
      .attr("cursor", "pointer")
      .call(dragBehavior);

    // Add glow filter for current node
    const defs = svg.append("defs");
    const filter = defs.append("filter").attr("id", "glow");
    filter
      .append("feGaussianBlur")
      .attr("stdDeviation", "3")
      .attr("result", "coloredBlur");
    const feMerge = filter.append("feMerge");
    feMerge.append("feMergeNode").attr("in", "coloredBlur");
    feMerge.append("feMergeNode").attr("in", "SourceGraphic");

    // Add outer glow circle for nodes
    nodeGroup
      .append("circle")
      .attr("class", "node-glow")
      .attr("r", (d) => getNodeSize(d) + 4)
      .attr("fill", (d) => getNodeColor(d))
      .attr("opacity", 0.3);

    // Add main circle for nodes
    nodeGroup
      .append("circle")
      .attr("class", "node-main")
      .attr("r", (d) => getNodeSize(d))
      .attr("fill", (d) => getNodeColor(d))
      .attr("stroke", (d) => (d.slug === currentSlug ? "#fff" : "none"))
      .attr("stroke-width", (d) => (d.slug === currentSlug ? 3 : 0))
      .attr("filter", (d) => (d.slug === currentSlug ? "url(#glow)" : "none"));

    // Add inner highlight circle
    nodeGroup
      .append("circle")
      .attr("class", "node-highlight")
      .attr("r", (d) => getNodeSize(d) * 0.4)
      .attr("fill", "#fff")
      .attr("opacity", 0.3)
      .attr("transform", (d) => {
        const offset = getNodeSize(d) * 0.25;
        return `translate(${-offset}, ${-offset})`;
      });

    // Hover effects
    nodeGroup
      .on("mouseenter", function (event, d) {
        setHoveredNode(d);

        // Highlight connected nodes and links
        const connectedNodeIds = new Set<string>();
        connectedNodeIds.add(d.id);

        links.forEach((l) => {
          const sourceId =
            typeof l.source === "string" ? l.source : (l.source as GraphNode).id;
          const targetId =
            typeof l.target === "string" ? l.target : (l.target as GraphNode).id;

          if (sourceId === d.id) connectedNodeIds.add(targetId);
          if (targetId === d.id) connectedNodeIds.add(sourceId);
        });

        // Dim non-connected nodes
        nodeGroup
          .transition()
          .duration(200)
          .attr("opacity", (n) => (connectedNodeIds.has(n.id) ? 1 : 0.2));

        // Highlight connected links
        link
          .transition()
          .duration(200)
          .attr("stroke-opacity", (l) => {
            const sourceId =
              typeof l.source === "string" ? l.source : (l.source as GraphNode).id;
            const targetId =
              typeof l.target === "string" ? l.target : (l.target as GraphNode).id;
            return sourceId === d.id || targetId === d.id ? 1 : 0.05;
          })
          .attr("stroke-width", (l) => {
            const sourceId =
              typeof l.source === "string" ? l.source : (l.source as GraphNode).id;
            const targetId =
              typeof l.target === "string" ? l.target : (l.target as GraphNode).id;
            return sourceId === d.id || targetId === d.id ? 3 : 1;
          });

        // Scale up hovered node
        d3.select(this)
          .select(".node-main")
          .transition()
          .duration(200)
          .attr("r", getNodeSize(d) * 1.3);

        d3.select(this)
          .select(".node-glow")
          .transition()
          .duration(200)
          .attr("r", getNodeSize(d) * 1.3 + 6)
          .attr("opacity", 0.5);
      })
      .on("mouseleave", function () {
        setHoveredNode(null);

        // Reset all nodes
        nodeGroup.transition().duration(200).attr("opacity", 1);

        // Reset all links
        link
          .transition()
          .duration(200)
          .attr("stroke-opacity", (d) => (d.type === "reference" ? 0.6 : 0.2))
          .attr("stroke-width", (d) => (d.type === "reference" ? 2 : 1));

        // Reset node size
        d3.select(this)
          .select(".node-main")
          .transition()
          .duration(200)
          .attr("r", (d) => getNodeSize(d as GraphNode));

        d3.select(this)
          .select(".node-glow")
          .transition()
          .duration(200)
          .attr("r", (d) => getNodeSize(d as GraphNode) + 4)
          .attr("opacity", 0.3);
      })
      .on("click", (event, d) => {
        event.stopPropagation();
        router.push(`/blog/${d.slug}`);
        if (onClose) onClose();
      });

    // Update positions on tick
    simulation.on("tick", () => {
      link
        .attr("x1", (d) => (d.source as GraphNode).x!)
        .attr("y1", (d) => (d.source as GraphNode).y!)
        .attr("x2", (d) => (d.target as GraphNode).x!)
        .attr("y2", (d) => (d.target as GraphNode).y!);

      nodeGroup.attr("transform", (d) => `translate(${d.x},${d.y})`);
    });

    // Center on current node if exists
    if (currentSlug) {
      const currentNode = nodes.find((n) => n.slug === currentSlug);
      if (currentNode) {
        setTimeout(() => {
          const transform = d3.zoomIdentity
            .translate(width / 2, height / 2)
            .scale(isExpanded ? 1.2 : 1)
            .translate(-(currentNode.x || width / 2), -(currentNode.y || height / 2));

          svg.transition().duration(750).call(zoom.transform, transform);
        }, 1000);
      }
    }

    // Cleanup
    return () => {
      simulation.stop();
    };
  }, [
    graphData,
    dimensions,
    currentSlug,
    isExpanded,
    getNodeColor,
    getNodeSize,
    router,
    onClose,
    // Note: theme is not included here to prevent animation restart on theme change
    // Theme colors are applied via themeRef and updated in separate useEffect
  ]);

  // Update link colors when theme changes (without restarting simulation)
  useEffect(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    const linkColorRef = theme === "dark" ? "#6b7280" : "#9ca3af";
    const linkColorTag = theme === "dark" ? "#374151" : "#d1d5db";

    // Update link colors without animation
    svg.selectAll(".links line")
      .attr("stroke", function() {
        const lineType = d3.select(this).datum() as { type: string } | undefined;
        return lineType?.type === "reference" ? linkColorRef : linkColorTag;
      });
  }, [theme]);

  // Tag legend (for expanded view)
  const tagLegend = useMemo(() => {
    if (!graphData || !isExpanded) return null;

    const topTags = Object.entries(graphData.tagGroups)
      .sort((a, b) => b[1].length - a[1].length)
      .slice(0, 10);

    return topTags.map(([tag, posts]) => ({
      tag,
      count: posts.length,
      color: graphData.tagColors[tag] || TAG_COLORS[0],
    }));
  }, [graphData, isExpanded]);

  if (isLoading) {
    return (
      <div
        ref={containerRef}
        className="flex items-center justify-center"
        style={{
          width: isExpanded ? "100%" : "100%",
          height: isExpanded ? "100%" : "350px",
          backgroundColor: "var(--background)",
        }}
      >
        <div className="flex flex-col items-center gap-2">
          <div
            className="w-8 h-8 border-2 border-t-transparent rounded-full animate-spin"
            style={{ borderColor: "var(--accent)", borderTopColor: "transparent" }}
          />
          <span className="text-sm" style={{ color: "var(--text-secondary)" }}>
            Loading graph...
          </span>
        </div>
      </div>
    );
  }

  if (!graphData || graphData.nodes.length === 0) {
    return null;
  }

  // Theme-based background colors
  const bgColor = theme === "dark" ? "#1a1a2e" : "#f8fafc";
  const textColor = theme === "dark" ? "#fff" : "#1e293b";
  const textSecondary = theme === "dark" ? "rgba(255,255,255,0.8)" : "rgba(0,0,0,0.7)";
  const textMuted = theme === "dark" ? "rgba(255,255,255,0.6)" : "rgba(0,0,0,0.5)";
  const overlayBg = theme === "dark" ? "rgba(0,0,0,0.6)" : "rgba(255,255,255,0.9)";
  const tooltipBg = theme === "dark" ? "rgba(0,0,0,0.9)" : "rgba(255,255,255,0.95)";
  const tooltipBorder = theme === "dark" ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)";
  const buttonBg = theme === "dark" ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.05)";
  const buttonHoverBg = theme === "dark" ? "rgba(255,255,255,0.2)" : "rgba(0,0,0,0.1)";

  return (
    <div
      ref={containerRef}
      className={`relative ${isExpanded ? "w-full h-full" : "w-full"}`}
      style={{
        backgroundColor: isExpanded ? bgColor : "transparent",
        borderRadius: isExpanded ? "12px" : "0",
      }}
    >
      {/* Header for expanded view */}
      {isExpanded && (
        <div
          className="absolute top-4 left-4 z-10 flex items-center gap-4"
          style={{ color: textColor }}
        >
          <div className="flex items-center gap-2">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1"
              />
            </svg>
            <span className="font-semibold">Graph view</span>
          </div>
        </div>
      )}

      {/* Close button for expanded view */}
      {isExpanded && onClose && (
        <button
          onClick={onClose}
          className="absolute top-4 right-4 z-10 p-2 rounded-lg transition-colors duration-200"
          style={{
            backgroundColor: buttonBg,
            color: textColor,
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.backgroundColor = buttonHoverBg;
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.backgroundColor = buttonBg;
          }}
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M6 18L18 6M6 6l12 12"
            />
          </svg>
        </button>
      )}

      {/* Tag Legend (expanded view only) */}
      {isExpanded && tagLegend && (
        <div
          className="absolute top-4 left-4 z-10 p-4 rounded-xl"
          style={{
            backgroundColor: overlayBg,
            backdropFilter: "blur(10px)",
            marginTop: "40px",
          }}
        >
          <h4 className="text-sm font-semibold mb-3" style={{ color: textColor }}>
            Groups
          </h4>
          <div className="space-y-2 max-h-60 overflow-y-auto">
            {tagLegend.map(({ tag, count, color }) => (
              <div key={tag} className="flex items-center gap-2">
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: color }}
                />
                <span className="text-xs" style={{ color: textSecondary }}>
                  {tag} ({count})
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Tooltip for hovered node */}
      {hoveredNode && (
        <div
          className="absolute z-20 p-3 rounded-lg shadow-xl max-w-xs pointer-events-none"
          style={{
            backgroundColor: tooltipBg,
            backdropFilter: "blur(10px)",
            border: `1px solid ${tooltipBorder}`,
            left: isExpanded ? "50%" : "50%",
            top: isExpanded ? "auto" : "10px",
            bottom: isExpanded ? "80px" : "auto",
            transform: "translateX(-50%)",
          }}
        >
          <h4 className="font-semibold text-sm mb-1 line-clamp-2" style={{ color: textColor }}>
            {hoveredNode.title}
          </h4>
          <div className="flex flex-wrap gap-1 mt-2">
            {hoveredNode.tags.slice(0, 3).map((tag) => (
              <span
                key={tag}
                className="px-2 py-0.5 rounded text-xs"
                style={{
                  backgroundColor:
                    graphData?.tagColors[tag] || buttonBg,
                  color: "#fff",
                }}
              >
                {tag}
              </span>
            ))}
          </div>
          <p className="text-xs mt-2" style={{ color: textMuted }}>
            {hoveredNode.connections} connections
          </p>
        </div>
      )}

      {/* SVG Graph */}
      <svg
        ref={svgRef}
        width={dimensions.width}
        height={dimensions.height}
        style={{
          display: "block",
          backgroundColor: isExpanded ? bgColor : "transparent",
        }}
      />

      {/* Instructions for expanded view */}
      {isExpanded && (
        <div
          className="absolute bottom-4 right-4 text-xs"
          style={{ color: textMuted }}
        >
          Scroll to zoom | Drag to pan | Click node to navigate
        </div>
      )}
    </div>
  );
}
