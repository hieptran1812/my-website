/**
 * Convert blog images to WebP and build the image manifest.
 *
 * Usage: npm run optimize-blog-images
 *
 * Pass 1 — every PNG/JPEG under public/imgs/blogs is downscaled to the content
 *   column width, re-encoded as WebP (replacing the original, which is deleted),
 *   and recorded in src/lib/generated/blogImageManifest.json with its served
 *   dimensions. Content <img> tags and cover refs resolve through this manifest
 *   (see src/lib/imageRewrite.ts), so the deleted PNG paths keep working.
 *
 * Pass 2 — for each post's resolved cover image, a small 16:9 thumbnail
 *   (<name>.cover.webp) is generated for card grids and the entry is flagged
 *   `cover: true`, so cards serve it statically (unoptimized) instead of routing
 *   a full-size image through next/image.
 *
 * Idempotent: converted sources are gone, so a re-run only touches images added
 * since the last run. The manifest is merged, never rebuilt from scratch.
 */

import fs from "fs";
import path from "path";
import matter from "gray-matter";
import sharp from "sharp";
import { resolvePostCover } from "../src/lib/blogIndex";

const PUBLIC_DIR = path.join(process.cwd(), "public");
const BLOG_IMG_DIR = path.join(PUBLIC_DIR, "imgs", "blogs");
const BLOG_CONTENT_DIR = path.join(process.cwd(), "content", "blog");
const MANIFEST_PATH = path.join(
  process.cwd(),
  "src",
  "lib",
  "generated",
  "blogImageManifest.json",
);

// Article body renders in a ~896px (max-w-4xl) column; 2× for retina displays.
const CONTENT_MAX_WIDTH = 1792;
const WEBP_QUALITY = 80;
// Card cover thumbnail: 16:9, sized for the largest grid card at ~2×.
const COVER_THUMB_W = 800;
const COVER_THUMB_H = 450;
const COVER_QUALITY = 70;

const RASTER_EXT_RE = /\.(png|jpe?g)$/i;

interface ManifestEntry {
  width: number;
  height: number;
  cover?: boolean;
}
type Manifest = Record<string, ManifestEntry>;

function loadManifest(): Manifest {
  if (!fs.existsSync(MANIFEST_PATH)) return {};
  try {
    return JSON.parse(fs.readFileSync(MANIFEST_PATH, "utf8")) as Manifest;
  } catch {
    return {};
  }
}

function saveManifest(manifest: Manifest): void {
  fs.mkdirSync(path.dirname(MANIFEST_PATH), { recursive: true });
  // Sort keys for stable, reviewable diffs.
  const sorted: Manifest = {};
  for (const key of Object.keys(manifest).sort()) sorted[key] = manifest[key];
  fs.writeFileSync(MANIFEST_PATH, JSON.stringify(sorted, null, 2) + "\n");
}

/** Public URL (e.g. "/imgs/blogs/foo.png") for an absolute file path. */
function publicKey(absPath: string): string {
  return "/" + path.relative(PUBLIC_DIR, absPath).split(path.sep).join("/");
}

/** Recursively collect files matching a predicate. */
function walk(dir: string, match: (name: string) => boolean): string[] {
  const out: string[] = [];
  if (!fs.existsSync(dir)) return out;
  const stack = [dir];
  while (stack.length) {
    const cur = stack.pop()!;
    for (const e of fs.readdirSync(cur, { withFileTypes: true })) {
      const full = path.join(cur, e.name);
      if (e.isDirectory()) stack.push(full);
      else if (e.isFile() && match(e.name)) out.push(full);
    }
  }
  return out;
}

async function convertSources(manifest: Manifest): Promise<number> {
  const sources = walk(BLOG_IMG_DIR, (n) => RASTER_EXT_RE.test(n));
  let converted = 0;

  for (const src of sources) {
    const key = publicKey(src); // original "/imgs/blogs/foo.png"
    const webpPath = src.replace(RASTER_EXT_RE, ".webp");
    try {
      const meta = await sharp(src).metadata();
      const srcW = meta.width ?? CONTENT_MAX_WIDTH;
      const targetW = Math.min(srcW, CONTENT_MAX_WIDTH);

      const info = await sharp(src)
        .resize({ width: targetW, withoutEnlargement: true })
        .webp({ quality: WEBP_QUALITY })
        .toFile(webpPath);

      manifest[key] = {
        width: info.width,
        height: info.height,
        ...(manifest[key]?.cover ? { cover: true } : {}),
      };
      fs.unlinkSync(src); // replace: drop the original raster
      converted++;
      console.log(`✅ ${key} → .webp (${info.width}×${info.height})`);
    } catch (err) {
      console.error(`❌ ${key}: ${(err as Error).message}`);
    }
  }
  return converted;
}

async function generateCoverThumbs(manifest: Manifest): Promise<number> {
  const posts = walk(BLOG_CONTENT_DIR, (n) => n.endsWith(".md"));
  let thumbs = 0;
  const seen = new Set<string>();

  for (const file of posts) {
    let cover: string | undefined;
    try {
      const parsed = matter(fs.readFileSync(file, "utf8"));
      cover = resolvePostCover(parsed.data as Record<string, unknown>, parsed.content);
    } catch {
      continue;
    }
    // Skip OG-generated covers and remote/non-raster refs.
    if (!cover || !cover.startsWith("/imgs/") || !RASTER_EXT_RE.test(cover)) continue;
    if (seen.has(cover)) continue;
    seen.add(cover);

    // After pass 1 the source on disk is the .webp sibling.
    const webpDisk = path.join(PUBLIC_DIR, cover.replace(RASTER_EXT_RE, ".webp"));
    if (!fs.existsSync(webpDisk)) {
      console.warn(`⚠️  cover source missing for ${cover} — skipping thumbnail`);
      continue;
    }

    const thumbDisk = webpDisk.replace(/\.webp$/, ".cover.webp");
    try {
      if (!fs.existsSync(thumbDisk)) {
        await sharp(webpDisk)
          .resize(COVER_THUMB_W, COVER_THUMB_H, { fit: "cover" })
          .webp({ quality: COVER_QUALITY })
          .toFile(thumbDisk);
        thumbs++;
      }
      // Ensure a manifest entry exists and is flagged, even if pass 1 didn't
      // run this time (e.g. the source was converted on a previous run).
      if (!manifest[cover]) {
        const m = await sharp(webpDisk).metadata();
        manifest[cover] = { width: m.width ?? 0, height: m.height ?? 0, cover: true };
      } else {
        manifest[cover].cover = true;
      }
    } catch (err) {
      console.error(`❌ thumbnail ${cover}: ${(err as Error).message}`);
    }
  }
  return thumbs;
}

async function main() {
  console.log("🖼️  Optimizing blog images...\n");
  if (!fs.existsSync(BLOG_IMG_DIR)) {
    console.error("❌ Image directory not found:", BLOG_IMG_DIR);
    process.exit(1);
  }

  const manifest = loadManifest();

  console.log("— Pass 1: PNG/JPEG → WebP —");
  const converted = await convertSources(manifest);

  console.log("\n— Pass 2: card cover thumbnails —");
  const thumbs = await generateCoverThumbs(manifest);

  saveManifest(manifest);

  console.log("\n📊 Summary:");
  console.log(`   ✅ Converted: ${converted} image(s)`);
  console.log(`   🖼️  Cover thumbnails: ${thumbs} new`);
  console.log(`   📄 Manifest entries: ${Object.keys(manifest).length}`);
  console.log(`   💾 ${path.relative(process.cwd(), MANIFEST_PATH)}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
