/**
 * Single source of truth for a post's cover image URL.
 *
 * - User-provided frontmatter image wins (relative paths only — external URLs
 *   would need next.config remotePatterns).
 * - Otherwise, returns the route-handler URL for the auto-generated cover.
 *   That route renders deterministic 1200×675 (16:9) PNGs via next/og.
 */

export const COVER_WIDTH = 1200;
export const COVER_HEIGHT = 675; // 16:9, matches displayed aspect everywhere
export const COVER_ASPECT = COVER_WIDTH / COVER_HEIGHT;

export function getPostCoverUrl(slug: string, frontmatterImage?: string): string {
  if (frontmatterImage) {
    const trimmed = frontmatterImage.trim();
    if (trimmed.length > 0 && !/^https?:\/\//i.test(trimmed)) return trimmed;
  }
  // Strip leading/trailing slashes from slug; route is catch-all.
  const clean = slug.replace(/^\/+|\/+$/g, "");
  return `/blog-cover/${clean}`;
}

export function getAbsoluteCoverUrl(
  slug: string,
  frontmatterImage: string | undefined,
  origin: string,
): string {
  const url = getPostCoverUrl(slug, frontmatterImage);
  if (/^https?:\/\//i.test(url)) return url;
  return `${origin.replace(/\/$/, "")}${url.startsWith("/") ? url : `/${url}`}`;
}
