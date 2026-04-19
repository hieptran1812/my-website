const PLACEHOLDER_IMAGES = ["/blog-placeholder.jpg", "/images/default-blog.jpg"];

/**
 * Generates an OG-style thumbnail URL for articles without a custom image.
 * Uses the /api/og endpoint with article metadata to create a clean, techy thumbnail.
 */
export function getArticleImageUrl(article: {
  image?: string;
  title: string;
  category?: string;
  subcategory?: string;
}): string {
  // Return existing image if valid
  if (
    article.image &&
    article.image.trim() !== "" &&
    !PLACEHOLDER_IMAGES.includes(article.image)
  ) {
    return article.image;
  }

  // Generate OG thumbnail URL. `v` is a cache-busting version bumped when
  // the generator algorithm changes (palette pool / shape layout).
  const params = new URLSearchParams({
    type: "article",
    title: article.title,
    v: "2",
  });

  if (article.category) params.set("category", article.category);
  if (article.subcategory) params.set("subcategory", article.subcategory);

  return `/api/og/?${params.toString()}`;
}

/**
 * Returns true if the resolved image URL points to a generated OG image.
 * Used to set `unoptimized` on Next.js Image components for dynamic endpoints.
 */
export function isGeneratedImage(article: { image?: string }): boolean {
  return (
    !article.image ||
    article.image.trim() === "" ||
    PLACEHOLDER_IMAGES.includes(article.image)
  );
}
