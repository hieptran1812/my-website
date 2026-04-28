import { NextResponse } from "next/server";
import { loadAllPosts } from "@/lib/blogIndex";

export interface GraphNode {
  id: string;
  slug: string;
  title: string;
  tags: string[];
  category: string;
  group: string;
  connections: number;
}

export interface GraphLink {
  source: string;
  target: string;
  type: "reference" | "tag";
}

export interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
  tagGroups: { [tag: string]: string[] };
  tagColors: { [tag: string]: string };
}

const TAG_COLORS = [
  "#f97316",
  "#22c55e",
  "#3b82f6",
  "#a855f7",
  "#ef4444",
  "#eab308",
  "#14b8a6",
  "#ec4899",
  "#6366f1",
  "#84cc16",
  "#f43f5e",
  "#06b6d4",
  "#8b5cf6",
  "#10b981",
  "#f59e0b",
];

export async function GET() {
  try {
    const corpus = await loadAllPosts();
    const nodes: GraphNode[] = [];
    const links: GraphLink[] = [];
    const tagGroups: { [tag: string]: string[] } = {};
    const postContents: { [slug: string]: string } = {};

    for (const entry of corpus) {
      const tags = entry.tags;
      const category = entry.category || "General";
      const primaryTag = tags[0] || category;
      for (const tag of tags) {
        if (!tagGroups[tag]) tagGroups[tag] = [];
        tagGroups[tag].push(entry.slug);
      }
      nodes.push({
        id: entry.slug,
        slug: entry.slug,
        title: entry.title,
        tags,
        category,
        group: primaryTag,
        connections: 0,
      });
      postContents[entry.slug] = entry.content;
    }

    const slugSet = new Set(nodes.map((n) => n.slug));

    for (const node of nodes) {
      const content = postContents[node.slug];
      if (!content) continue;
      const linkRegex = /\[([^\]]*)\]\((?:\/blog\/)?([^)]+)\)/g;
      let match;
      while ((match = linkRegex.exec(content)) !== null) {
        const linkedSlug = match[2].replace(/^\/blog\//, "").replace(/\/$/, "");
        if (slugSet.has(linkedSlug) && linkedSlug !== node.slug) {
          const exists = links.find(
            (l) =>
              (l.source === node.slug && l.target === linkedSlug) ||
              (l.source === linkedSlug && l.target === node.slug),
          );
          if (!exists) {
            links.push({
              source: node.slug,
              target: linkedSlug,
              type: "reference",
            });
          }
        }
      }
    }

    for (const tag of Object.keys(tagGroups)) {
      const arr = tagGroups[tag];
      if (arr.length >= 2 && arr.length <= 10) {
        for (let i = 0; i < arr.length; i++) {
          for (let j = i + 1; j < arr.length; j++) {
            const exists = links.find(
              (l) =>
                (l.source === arr[i] && l.target === arr[j]) ||
                (l.source === arr[j] && l.target === arr[i]),
            );
            if (!exists) {
              links.push({ source: arr[i], target: arr[j], type: "tag" });
            }
          }
        }
      }
    }

    for (const link of links) {
      const s = nodes.find((n) => n.id === link.source);
      const t = nodes.find((n) => n.id === link.target);
      if (s) s.connections++;
      if (t) t.connections++;
    }

    const tagColors: { [tag: string]: string } = {};
    const sortedTags = Object.keys(tagGroups).sort(
      (a, b) => tagGroups[b].length - tagGroups[a].length,
    );
    sortedTags.forEach((tag, index) => {
      tagColors[tag] = TAG_COLORS[index % TAG_COLORS.length];
    });

    return NextResponse.json(
      { nodes, links, tagGroups, tagColors },
      {
        headers: {
          "Cache-Control":
            "public, s-maxage=3600, stale-while-revalidate=86400",
        },
      },
    );
  } catch (error) {
    console.error("Error building graph data:", error);
    return NextResponse.json({
      nodes: [],
      links: [],
      tagGroups: {},
      tagColors: {},
    });
  }
}
