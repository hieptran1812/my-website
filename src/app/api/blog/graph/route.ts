import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";
import matter from "gray-matter";

export interface GraphNode {
  id: string;
  slug: string;
  title: string;
  tags: string[];
  category: string;
  group: string; // Tag cluster
  connections: number;
}

export interface GraphLink {
  source: string;
  target: string;
  type: "reference" | "tag"; // reference = direct link, tag = same tag
}

export interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
  tagGroups: { [tag: string]: string[] }; // tag -> node ids
  tagColors: { [tag: string]: string };
}

// Color palette for tag clusters
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

export async function GET() {
  try {
    const blogDir = path.join(process.cwd(), "content", "blog");

    if (!fs.existsSync(blogDir)) {
      return NextResponse.json({ nodes: [], links: [], tagGroups: {}, tagColors: {} });
    }

    const nodes: GraphNode[] = [];
    const links: GraphLink[] = [];
    const tagGroups: { [tag: string]: string[] } = {};
    const postContents: { [slug: string]: string } = {};

    // Helper function to read markdown files from a directory
    const readMarkdownFiles = (
      dirPath: string,
      basePath: string = ""
    ) => {
      if (!fs.existsSync(dirPath)) return;

      const files = fs.readdirSync(dirPath, { withFileTypes: true });

      for (const file of files) {
        if (file.isDirectory()) {
          const newBasePath = basePath ? `${basePath}/${file.name}` : file.name;
          readMarkdownFiles(path.join(dirPath, file.name), newBasePath);
        } else if (file.name.endsWith(".md")) {
          const filePath = path.join(dirPath, file.name);
          const fileContent = fs.readFileSync(filePath, "utf8");
          const { data: metadata, content } = matter(fileContent);

          const slugBase = file.name.replace(/\.md$/, "");
          const slug = basePath ? `${basePath}/${slugBase}` : slugBase;

          const tags: string[] = metadata.tags || [];
          const category = metadata.category || "General";

          // Use first tag as primary group, or category if no tags
          const primaryTag = tags[0] || category;

          // Add to tag groups
          for (const tag of tags) {
            if (!tagGroups[tag]) {
              tagGroups[tag] = [];
            }
            tagGroups[tag].push(slug);
          }

          nodes.push({
            id: slug,
            slug,
            title: metadata.title || "Untitled",
            tags,
            category,
            group: primaryTag,
            connections: 0,
          });

          postContents[slug] = content;
        }
      }
    };

    // Start reading from the root blog directory
    readMarkdownFiles(blogDir);

    // Find references (links between posts)
    const slugSet = new Set(nodes.map((n) => n.slug));

    for (const node of nodes) {
      const content = postContents[node.slug];
      if (!content) continue;

      // Find markdown links: [text](/blog/slug) or [text](slug)
      const linkRegex = /\[([^\]]*)\]\((?:\/blog\/)?([^)]+)\)/g;
      let match;

      while ((match = linkRegex.exec(content)) !== null) {
        const linkedSlug = match[2].replace(/^\/blog\//, "").replace(/\/$/, "");

        // Check if this links to another blog post
        if (slugSet.has(linkedSlug) && linkedSlug !== node.slug) {
          // Check if link already exists
          const existingLink = links.find(
            (l) =>
              (l.source === node.slug && l.target === linkedSlug) ||
              (l.source === linkedSlug && l.target === node.slug)
          );

          if (!existingLink) {
            links.push({
              source: node.slug,
              target: linkedSlug,
              type: "reference",
            });
          }
        }
      }
    }

    // Add tag-based connections (lighter weight - only for posts with same tags)
    // Only add if there's no direct reference already
    for (const tag of Object.keys(tagGroups)) {
      const postsWithTag = tagGroups[tag];

      // Only create connections for tags with 2-5 posts (avoid too many connections)
      if (postsWithTag.length >= 2 && postsWithTag.length <= 10) {
        for (let i = 0; i < postsWithTag.length; i++) {
          for (let j = i + 1; j < postsWithTag.length; j++) {
            const existingLink = links.find(
              (l) =>
                (l.source === postsWithTag[i] && l.target === postsWithTag[j]) ||
                (l.source === postsWithTag[j] && l.target === postsWithTag[i])
            );

            if (!existingLink) {
              links.push({
                source: postsWithTag[i],
                target: postsWithTag[j],
                type: "tag",
              });
            }
          }
        }
      }
    }

    // Update connection counts
    for (const link of links) {
      const sourceNode = nodes.find((n) => n.id === link.source);
      const targetNode = nodes.find((n) => n.id === link.target);
      if (sourceNode) sourceNode.connections++;
      if (targetNode) targetNode.connections++;
    }

    // Assign colors to tag groups
    const tagColors: { [tag: string]: string } = {};
    const sortedTags = Object.keys(tagGroups).sort(
      (a, b) => tagGroups[b].length - tagGroups[a].length
    );

    sortedTags.forEach((tag, index) => {
      tagColors[tag] = TAG_COLORS[index % TAG_COLORS.length];
    });

    return NextResponse.json({
      nodes,
      links,
      tagGroups,
      tagColors,
    });
  } catch (error) {
    console.error("Error building graph data:", error);
    return NextResponse.json({ nodes: [], links: [], tagGroups: {}, tagColors: {} });
  }
}
