import fs from "fs";
import path from "path";
import matter from "gray-matter";
import { remark } from "remark";
import remarkGfm from "remark-gfm";
import remarkRehype from "remark-rehype";
import rehypeHighlight from "rehype-highlight";
import rehypeStringify from "rehype-stringify";
import { calculateReadTimeWithTags } from "./readTimeCalculator";
import { protectMathBlocks, restoreMathBlocks } from "./markdown";

export interface ArticleData {
  title: string;
  content: string;
  publishDate: string;
  readTime: string;
  tags: string[];
  category: string;
  author: string;
  slug: string;
  collection?: string;
  aiGenerated?: boolean;
  excerpt?: string;
}

const blogDir = path.join(process.cwd(), "content", "blog");

function findArticleFile(
  baseDir: string,
  slugParts: string[],
): string | null {
  if (slugParts.length === 0) return null;

  if (slugParts.length === 1) {
    const filePath = path.join(baseDir, `${slugParts[0]}.md`);
    return fs.existsSync(filePath) ? filePath : null;
  }

  const dirPath = path.join(baseDir, slugParts[0]);
  if (fs.existsSync(dirPath) && fs.statSync(dirPath).isDirectory()) {
    return findArticleFile(dirPath, slugParts.slice(1));
  }

  return null;
}

export async function getArticle(slug: string): Promise<ArticleData | null> {
  const slugParts = slug.split("/");
  const articlePath = findArticleFile(blogDir, slugParts);

  if (!articlePath) return null;

  const fileContent = fs.readFileSync(articlePath, "utf8");
  const { data: metadata, content: markdownContent } = matter(fileContent);

  const { protectedContent, mathBlocks } = protectMathBlocks(markdownContent);

  const processedContent = await remark()
    .use(remarkGfm)
    .use(remarkRehype, { allowDangerousHtml: true })
    .use(rehypeHighlight)
    .use(rehypeStringify, { allowDangerousHtml: true })
    .process(protectedContent);

  const htmlContent = restoreMathBlocks(processedContent.toString(), mathBlocks);

  let category = metadata.category;
  if (!category && slugParts.length > 1) {
    category = slugParts[0];
  }

  const readTimeResult = calculateReadTimeWithTags(
    markdownContent,
    metadata.tags || [],
    category || "General",
  );

  return {
    title: metadata.title || "Untitled",
    content: htmlContent,
    publishDate: metadata.publishDate || metadata.date || "",
    readTime: readTimeResult.readTime,
    tags: metadata.tags || [],
    category: category || "",
    author: metadata.author || "",
    slug,
    collection: metadata.collection,
    aiGenerated: metadata.aiGenerated === true,
    excerpt: metadata.excerpt || metadata.description || "",
  };
}

/** Enumerate all blog slugs for static generation */
export function getAllBlogSlugs(): string[][] {
  const slugs: string[][] = [];

  function walk(dir: string, prefix: string[]) {
    if (!fs.existsSync(dir)) return;
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    for (const entry of entries) {
      if (entry.isDirectory()) {
        walk(path.join(dir, entry.name), [...prefix, entry.name]);
      } else if (entry.name.endsWith(".md")) {
        slugs.push([...prefix, entry.name.replace(/\.md$/, "")]);
      }
    }
  }

  walk(blogDir, []);
  return slugs;
}
