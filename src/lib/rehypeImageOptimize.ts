/**
 * Rehype plugin: optimize <img> tags in rendered article content.
 *
 * Content images come from markdown and are emitted as plain <img> tags
 * (they bypass next/image). Without help the browser fetches every image
 * eagerly, decodes on the main thread, and reflows when each one lands —
 * hurting LCP, scroll performance, and CLS on image-heavy posts.
 *
 * Using the build-time image manifest (see imageRewrite.ts) this adds:
 *  - src .png/.jpg → .webp — serve the converted static WebP (≈80-90% smaller)
 *  - width/height         — reserve layout space up-front, eliminating CLS
 *  - sizes                — match the ~896px content column
 *  - loading="lazy"       — defer offscreen images until they near the viewport
 *  - decoding="async"     — decode off the main thread
 *  - fetchpriority="high" — on the first image (LCP candidate)
 *
 * The first content image is left eager so it can serve as the LCP element
 * when a post opens with an image near the top. Images absent from the
 * manifest (not yet run through the conversion script) are left untouched.
 */

import { imageDims, toWebp } from "./imageRewrite";

// The article body renders in a ~896px (max-w-4xl) column; 100vw below that.
const CONTENT_SIZES = "(max-width: 896px) 100vw, 896px";

interface HastNode {
  type: string;
  tagName?: string;
  properties?: Record<string, unknown>;
  children?: HastNode[];
}

export default function rehypeImageOptimize() {
  return (tree: HastNode) => {
    let isFirstImage = true;

    const walk = (node: HastNode) => {
      if (node.tagName === "img") {
        node.properties = node.properties || {};
        const src = typeof node.properties.src === "string" ? node.properties.src : "";
        const entry = src ? imageDims(src) : undefined;

        // Intrinsic dimensions → no layout shift while the image loads.
        if (entry) {
          if (node.properties.width === undefined) node.properties.width = entry.width;
          if (node.properties.height === undefined) node.properties.height = entry.height;
          if (node.properties.sizes === undefined) node.properties.sizes = CONTENT_SIZES;
        }

        if (isFirstImage) {
          // Keep the first image eager — likely LCP candidate.
          isFirstImage = false;
          if (node.properties.fetchpriority === undefined)
            node.properties.fetchpriority = "high";
        } else if (node.properties.loading === undefined) {
          node.properties.loading = "lazy";
        }

        if (node.properties.decoding === undefined) {
          node.properties.decoding = "async";
        }

        // Swap to the converted WebP last (no-op for un-converted images).
        if (entry) node.properties.src = toWebp(src);
      }

      if (node.children) {
        for (const child of node.children) {
          walk(child);
        }
      }
    };

    walk(tree);
  };
}
