import { NextRequest, NextResponse } from "next/server";
import fs from "fs";
import path from "path";
import matter from "gray-matter";
import { remark } from "remark";
import remarkGfm from "remark-gfm";
import remarkHtml from "remark-html";
import { calculateReadTimeWithTags } from "../../../../lib/readTimeCalculator";

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
    const filePath = path.join(contentDirectory, `${slug}.md`);

    if (!fs.existsSync(filePath)) {
      return NextResponse.json({ error: "Post not found" }, { status: 404 });
    }

    const fileContents = fs.readFileSync(filePath, "utf8");
    const { data, content } = matter(fileContents);

    // Calculate read time from actual content
    const readTimeResult = calculateReadTimeWithTags(
      content,
      data.tags || [],
      data.category || "general"
    );

    // Process markdown to HTML
    const processedContent = await remark()
      .use(remarkGfm)
      .use(remarkHtml)
      .process(content);

    const htmlContent = processedContent.toString();

    const post = {
      slug,
      title: data.title || "Untitled",
      publishDate: data.date || data.publishDate || "2024-01-01",
      readTime: readTimeResult.readTime,
      category: data.category || "general",
      author: data.author || "Anonymous",
      tags: data.tags || [],
      image: data.image || "/images/default-blog.jpg",
      excerpt: data.excerpt || data.description || "No excerpt available",
      content: htmlContent,
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
