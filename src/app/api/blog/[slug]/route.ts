import { readPostRaw, slugToRelPath } from "@/lib/contentSource";
import { NextRequest, NextResponse } from "next/server";
import path from "path";
import matter from "gray-matter";
import { remark } from "remark";
import remarkGfm from "remark-gfm";
import remarkHtml from "remark-html";
import { calculateReadTimeWithTags } from "../../../../lib/readTimeCalculator";
import { protectMathBlocks, restoreMathBlocks } from "../../../../lib/markdown";
import { derivePostLocation } from "../../../../lib/postPath";
import remarkCallouts from "../../../../lib/remarkCallouts";
import {
  applyPaywallToFile,
  PAYWALL_SUBSCRIBE_URL,
} from "../../../../lib/paywall";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ slug: string }> }
) {
  try {
    const { slug } = await params;

    if (!slug) {
      return NextResponse.json({ error: "Slug is required" }, { status: 400 });
    }

    const contentDirectory = path.join(process.cwd(), "content", "blog");

    // Disk in dev/build, R2 in serverless functions (see contentSource.ts).
    const rel = slugToRelPath(slug);
    const fileContents = rel ? await readPostRaw(rel) : null;
    if (fileContents === null || !rel) {
      return NextResponse.json({ error: "Post not found" }, { status: 404 });
    }
    const filePath = path.join(contentDirectory, ...rel.split("/"));
    const { data, content } = matter(fileContents);

    const { category: resolvedCategory } = derivePostLocation(
      filePath,
      data,
      contentDirectory
    );

    // Calculate read time from actual content
    const readTimeResult = calculateReadTimeWithTags(
      content,
      data.tags || [],
      resolvedCategory || "general"
    );

    // Protect math blocks from being mangled by remark
    const { protectedContent, mathBlocks } = protectMathBlocks(content);

    // Process markdown to HTML
    const processedContent = await remark()
      .use(remarkGfm)
      .use(remarkCallouts)
      .use(remarkHtml, { sanitize: false })
      .process(protectedContent);

    const htmlContent = restoreMathBlocks(processedContent.toString(), mathBlocks);

    // Same gate as the rendered page — otherwise this route hands back the
    // full body of a paywalled trading post to anyone who calls it. Keyed on
    // the resolved path, since the slug param is not path-normalised and
    // `findArticleFile()` follows `..` segments.
    const { html: gatedContent, paywalled } = applyPaywallToFile(
      filePath,
      htmlContent,
    );

    const post = {
      slug,
      title: data.title || "Untitled",
      publishDate: data.date || data.publishDate || "2024-01-01",
      readTime: readTimeResult.readTime,
      category: resolvedCategory || "general",
      author: data.author || "Anonymous",
      tags: data.tags || [],
      image: data.image || "/images/default-blog.jpg",
      excerpt: data.excerpt || data.description || "No excerpt available",
      substackUrl:
        typeof data.substackUrl === "string"
          ? data.substackUrl.trim() || undefined
          : typeof data.substack === "string"
            ? data.substack.trim() || undefined
            : undefined,
      content: gatedContent,
      paywalled,
      paywallUrl: paywalled ? PAYWALL_SUBSCRIBE_URL : undefined,
    };

    return NextResponse.json(post);
  } catch (error) {
    console.error("Error fetching blog post:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
