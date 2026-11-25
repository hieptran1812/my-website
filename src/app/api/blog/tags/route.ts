import { NextResponse } from "next/server";
import { readCorpus } from "@/lib/contentSource";
import matter from "gray-matter";

interface TagInfo {
  tag: string;
  slug: string;
  count: number;
  categories: string[];
}

/**
 * API Route: GET /api/blog/tags
 * Returns all unique tags with their counts and associated categories
 */
export async function GET() {
  try {
    // Disk in dev/build, the R2 corpus bundle in serverless functions.
    const corpus = await readCorpus();

    // Process all files and extract tags
    const tagData: Record<string, { count: number; categories: Set<string> }> = {};

    for (const [filePath, fileContent] of Object.entries(corpus)) {
      try {
        const { data: frontmatter } = matter(fileContent);

        const tags = frontmatter.tags || [];
        const category = frontmatter.category || "uncategorized";

        for (const tag of tags) {
          if (typeof tag === "string" && tag.trim()) {
            const normalizedTag = tag.trim();

            if (!tagData[normalizedTag]) {
              tagData[normalizedTag] = {
                count: 0,
                categories: new Set(),
              };
            }

            tagData[normalizedTag].count += 1;
            tagData[normalizedTag].categories.add(category);
          }
        }
      } catch (error) {
        console.error(`Error processing file ${filePath}:`, error);
      }
    }

    // Convert to array and sort by count (descending)
    const tags: TagInfo[] = Object.entries(tagData)
      .map(([tag, data]) => ({
        tag,
        slug: tag.toLowerCase().replace(/\s+/g, "-"),
        count: data.count,
        categories: Array.from(data.categories),
      }))
      .sort((a, b) => b.count - a.count);

    // Return response with cache headers
    return NextResponse.json(tags, {
      headers: {
        "Cache-Control": "public, s-maxage=300, stale-while-revalidate=600",
      },
    });
  } catch (error) {
    console.error("Error fetching tags:", error);
    return NextResponse.json(
      { error: "Failed to fetch tags" },
      { status: 500 }
    );
  }
}
