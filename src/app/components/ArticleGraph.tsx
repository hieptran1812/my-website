"use client";

import React, { useRef, useState } from "react";
import Link from "next/link";

interface Node {
  id: string;
  title: string;
  category: string;
  x: number;
  y: number;
  radius: number;
  color: string;
}

interface Edge {
  source: string;
  target: string;
}

interface ArticleGraphProps {
  currentArticle: string;
  relatedArticles: string[];
  className?: string;
}

export default function ArticleGraph({
  currentArticle,
  relatedArticles,
  className = "",
}: ArticleGraphProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);

  // Mock data - in real implementation, this would come from your blog data
  const getArticleData = (id: string) => {
    const mockArticles: { [key: string]: { title: string; category: string } } =
      {
        [currentArticle]: {
          title: "Current Article",
          category: "software-engineering",
        },
        "article-1": {
          title: "Understanding TypeScript",
          category: "software-engineering",
        },
        "article-2": {
          title: "React Best Practices",
          category: "software-engineering",
        },
        "article-3": {
          title: "Machine Learning Basics",
          category: "paper-reading",
        },
        "article-4": { title: "Blockchain Fundamentals", category: "crypto" },
        "article-5": { title: "System Design Notes", category: "notes" },
      };
    return mockArticles[id] || { title: `Article ${id}`, category: "notes" };
  };

  const getCategoryColor = (category: string) => {
    const colors: { [key: string]: string } = {
      "software-engineering": "#3b82f6",
      "paper-reading": "#8b5cf6",
      crypto: "#f59e0b",
      notes: "#10b981",
    };
    return colors[category] || "#6b7280";
  };

  const createNodes = (): Node[] => {
    const width = 400;
    const height = 300;
    const centerX = width / 2;
    const centerY = height / 2;

    const nodes: Node[] = [];

    // Current article as center node
    const currentData = getArticleData(currentArticle);
    nodes.push({
      id: currentArticle,
      title: currentData.title,
      category: currentData.category,
      x: centerX,
      y: centerY,
      radius: 25,
      color: getCategoryColor(currentData.category),
    });

    // Related articles in a circle around the center
    const angleStep = (2 * Math.PI) / relatedArticles.length;
    const radius = 100;

    relatedArticles.forEach((articleId, index) => {
      const angle = index * angleStep;
      const x = centerX + radius * Math.cos(angle);
      const y = centerY + radius * Math.sin(angle);
      const data = getArticleData(articleId);

      nodes.push({
        id: articleId,
        title: data.title,
        category: data.category,
        x,
        y,
        radius: 20,
        color: getCategoryColor(data.category),
      });
    });

    return nodes;
  };

  const createEdges = (): Edge[] => {
    return relatedArticles.map((articleId) => ({
      source: currentArticle,
      target: articleId,
    }));
  };

  const nodes = createNodes();
  const edges = createEdges();

  const handleNodeClick = (nodeId: string) => {
    setSelectedNode(nodeId);
    if (nodeId !== currentArticle) {
      // In a real implementation, navigate to the article
      console.log(`Navigate to article: ${nodeId}`);
    }
  };

  const handleNodeHover = (nodeId: string | null) => {
    setHoveredNode(nodeId);
  };

  return (
    <div className={`w-full h-full flex flex-col ${className}`}>
      <div className="mb-4">
        <h3
          className="text-lg font-semibold mb-2"
          style={{ color: "var(--text-primary)" }}
        >
          Article Relationship Graph
        </h3>
        <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
          Explore how this article connects to related content. Click on nodes
          to navigate.
        </p>
      </div>

      <div className="flex-1 relative">
        <svg
          ref={svgRef}
          width="100%"
          height="100%"
          viewBox="0 0 400 300"
          className="border rounded-lg"
          style={{ borderColor: "var(--border)" }}
        >
          <defs>
            <filter id="glow">
              <feGaussianBlur stdDeviation="3" result="coloredBlur" />
              <feMerge>
                <feMergeNode in="coloredBlur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
            <marker
              id="arrowhead"
              markerWidth="10"
              markerHeight="7"
              refX="9"
              refY="3.5"
              orient="auto"
            >
              <polygon points="0 0, 10 3.5, 0 7" fill="var(--border)" />
            </marker>
          </defs>

          {/* Edges */}
          {edges.map((edge, index) => {
            const sourceNode = nodes.find((n) => n.id === edge.source);
            const targetNode = nodes.find((n) => n.id === edge.target);

            if (!sourceNode || !targetNode) return null;

            const isHighlighted =
              hoveredNode === edge.source || hoveredNode === edge.target;

            return (
              <line
                key={index}
                x1={sourceNode.x}
                y1={sourceNode.y}
                x2={targetNode.x}
                y2={targetNode.y}
                stroke={isHighlighted ? "var(--accent)" : "var(--border)"}
                strokeWidth={isHighlighted ? 2 : 1}
                strokeOpacity={isHighlighted ? 0.8 : 0.4}
                markerEnd="url(#arrowhead)"
                className="transition-all duration-200"
              />
            );
          })}

          {/* Nodes */}
          {nodes.map((node) => {
            const isHovered = hoveredNode === node.id;
            const isSelected = selectedNode === node.id;
            const isCurrent = node.id === currentArticle;

            return (
              <g key={node.id}>
                {/* Node circle */}
                <circle
                  cx={node.x}
                  cy={node.y}
                  r={node.radius}
                  fill={node.color}
                  stroke={
                    isCurrent
                      ? "#fff"
                      : isHovered || isSelected
                      ? "var(--accent)"
                      : "transparent"
                  }
                  strokeWidth={isCurrent ? 3 : 2}
                  className="cursor-pointer transition-all duration-200"
                  style={{
                    filter: isHovered ? "url(#glow)" : "none",
                    transform: isHovered ? "scale(1.1)" : "scale(1)",
                    transformOrigin: `${node.x}px ${node.y}px`,
                  }}
                  onMouseEnter={() => handleNodeHover(node.id)}
                  onMouseLeave={() => handleNodeHover(null)}
                  onClick={() => handleNodeClick(node.id)}
                />

                {/* Node label */}
                <text
                  x={node.x}
                  y={node.y - node.radius - 8}
                  textAnchor="middle"
                  fontSize="12"
                  fill="var(--text-primary)"
                  className="pointer-events-none"
                  style={{
                    fontWeight: isCurrent ? "bold" : "normal",
                    opacity: isHovered || isCurrent ? 1 : 0.8,
                  }}
                >
                  {node.title.length > 15
                    ? `${node.title.substring(0, 15)}...`
                    : node.title}
                </text>

                {/* Category badge */}
                <text
                  x={node.x}
                  y={node.y + node.radius + 16}
                  textAnchor="middle"
                  fontSize="10"
                  fill="var(--text-secondary)"
                  className="pointer-events-none"
                  style={{
                    opacity: isHovered || isCurrent ? 1 : 0.6,
                  }}
                >
                  {node.category}
                </text>
              </g>
            );
          })}
        </svg>

        {/* Node details panel */}
        {hoveredNode && (
          <div
            className="absolute bottom-4 left-4 right-4 p-3 rounded-lg border shadow-lg"
            style={{
              backgroundColor: "var(--surface)",
              borderColor: "var(--border)",
            }}
          >
            {(() => {
              const node = nodes.find((n) => n.id === hoveredNode);
              if (!node) return null;

              return (
                <div>
                  <h4
                    className="font-semibold text-sm mb-1"
                    style={{ color: "var(--text-primary)" }}
                  >
                    {node.title}
                  </h4>
                  <div className="flex items-center justify-between">
                    <span
                      className="text-xs px-2 py-1 rounded-full"
                      style={{
                        backgroundColor: `${node.color}20`,
                        color: node.color,
                      }}
                    >
                      {node.category}
                    </span>
                    {node.id !== currentArticle && (
                      <Link
                        href={`/blog/${node.category}/${node.id}`}
                        className="text-xs text-blue-500 hover:underline"
                      >
                        Read Article →
                      </Link>
                    )}
                  </div>
                </div>
              );
            })()}
          </div>
        )}
      </div>

      {/* Legend */}
      <div className="mt-4 flex flex-wrap gap-4 text-xs">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-blue-500"></div>
          <span style={{ color: "var(--text-secondary)" }}>
            Software Engineering
          </span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-purple-500"></div>
          <span style={{ color: "var(--text-secondary)" }}>Paper Reading</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
          <span style={{ color: "var(--text-secondary)" }}>Crypto</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-green-500"></div>
          <span style={{ color: "var(--text-secondary)" }}>Notes</span>
        </div>
      </div>
    </div>
  );
}
