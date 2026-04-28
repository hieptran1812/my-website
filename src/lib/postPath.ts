import path from "path";

const BLOG_DIR_NAME = path.join("content", "blog");

export interface PostLocation {
  category: string;
  subcategory: string;
}

function pickString(value: unknown): string | undefined {
  if (typeof value !== "string") return undefined;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
}

/**
 * Resolve a post's category and subcategory.
 *
 * Folder structure wins: subcategory = nearest parent folder, category = its parent.
 * Frontmatter values are used only when the folder layout cannot supply them
 * (e.g. file sits directly under content/blog/<category>/ with no subcategory folder,
 * or the file is outside content/blog entirely).
 */
export function derivePostLocation(
  absFilePath: string,
  frontmatter: Record<string, unknown> = {},
  blogRoot: string = path.join(process.cwd(), "content", "blog"),
): PostLocation {
  const fmCategory = pickString(frontmatter.category);
  const fmSubcategory = pickString(frontmatter.subcategory);

  let folderCategory: string | undefined;
  let folderSubcategory: string | undefined;

  const rel = path.relative(blogRoot, absFilePath);
  if (rel && !rel.startsWith("..") && !path.isAbsolute(rel)) {
    const parts = rel.split(path.sep).filter(Boolean);
    // parts: [<category>, ..., <subcategory>, <file>.md]
    if (parts.length >= 3) {
      folderCategory = parts[parts.length - 3];
      folderSubcategory = parts[parts.length - 2];
    } else if (parts.length === 2) {
      folderCategory = parts[0];
    }
  } else {
    // Fallback: scan path for the blog dir marker.
    const norm = absFilePath.split(path.sep);
    const idx = norm.findIndex(
      (p, i) => p === "blog" && norm[i - 1] === "content",
    );
    if (idx >= 0) {
      const parts = norm.slice(idx + 1);
      if (parts.length >= 3) {
        folderCategory = parts[parts.length - 3];
        folderSubcategory = parts[parts.length - 2];
      } else if (parts.length === 2) {
        folderCategory = parts[0];
      }
    }
  }

  return {
    category: folderCategory ?? fmCategory ?? "",
    subcategory: folderSubcategory ?? fmSubcategory ?? "",
  };
}

export { BLOG_DIR_NAME };
