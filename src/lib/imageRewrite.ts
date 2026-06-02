/**
 * Maps blog image references to their optimized static variants.
 *
 * `scripts/optimizeBlogImages.ts` converts every PNG/JPEG under
 * public/imgs/blogs to WebP (replacing the original) and emits
 * `generated/blogImageManifest.json`, keyed by the *original* public path
 * (e.g. "/imgs/blogs/foo.png"). Because the source files are gone after
 * conversion, every consumer of those paths must resolve through here:
 *
 *  - content <img> + cover refs  → toWebp()      (full-size .webp)
 *  - card-grid covers            → see getCardImageProps in articleImage.ts
 *
 * The manifest is the single source of truth for which images were converted,
 * their served dimensions (for width/height → no CLS), and whether a 16:9 card
 * thumbnail (.cover.webp) exists. Lookups are O(1) on the original path, and an
 * unknown path falls through unchanged — so a freshly-added image that hasn't
 * been run through the script yet still resolves to its on-disk original.
 */

import manifest from "./generated/blogImageManifest.json";

export interface BlogImageEntry {
  /** Served width after downscale (intrinsic px). */
  width: number;
  /** Served height after downscale (intrinsic px). */
  height: number;
  /** True when a `<name>.cover.webp` 16:9 thumbnail was generated. */
  cover?: boolean;
}

const dims = manifest as Record<string, BlogImageEntry>;

const RASTER_EXT_RE = /\.(png|jpe?g)$/i;

/** Manifest entry for an original image path, or undefined if not converted. */
export function imageDims(src: string): BlogImageEntry | undefined {
  return dims[src];
}

/**
 * Rewrites a converted PNG/JPEG path to its `.webp` sibling. Leaves anything
 * not in the manifest (remote URLs, OG endpoints, un-converted files) untouched.
 */
export function toWebp(src: string): string {
  if (!src) return src;
  return dims[src] ? src.replace(RASTER_EXT_RE, ".webp") : src;
}

/** Rewrites a converted path to its 16:9 card thumbnail (`.cover.webp`). */
export function toCoverThumb(src: string): string {
  return src.replace(RASTER_EXT_RE, ".cover.webp");
}
