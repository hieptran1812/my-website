/**
 * Where blog images are served from.
 *
 * Images live in Cloudflare R2 (bucket halleywiki-media, uploaded by
 * scripts/syncR2.ts) and are served from its custom domain, not from public/:
 * 4+ GB of images in the repo made every clone and deploy slow. Content keeps
 * referencing them as "/imgs/…"; this maps those paths to the CDN.
 *
 * NEXT_PUBLIC_MEDIA_BASE is inlined at build time. When it is unset (a local
 * checkout that still has public/imgs) paths are left as they are.
 */

const BASE = (process.env.NEXT_PUBLIC_MEDIA_BASE || "").replace(/\/$/, "");

export function mediaUrl(src: string): string {
  if (!BASE || !src) return src;
  return src.startsWith("/imgs/") ? BASE + src : src;
}

/** True when `src` is served by the media CDN (already optimized WebP). */
export function isMediaCdn(src: string): boolean {
  return !!BASE && src.startsWith(BASE + "/");
}
