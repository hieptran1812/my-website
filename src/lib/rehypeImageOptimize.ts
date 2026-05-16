/**
 * Rehype plugin: optimize <img> tags in rendered article content.
 *
 * Content images come from markdown and are emitted as plain <img> tags
 * (they bypass next/image). Without these hints the browser fetches every
 * image eagerly and decodes them on the main thread, hurting LCP and
 * scroll performance on image-heavy posts.
 *
 * Adds:
 *  - loading="lazy"   — defer offscreen images until they near the viewport
 *  - decoding="async" — decode off the main thread
 *
 * The first content image is left eager so it can still serve as the LCP
 * element when a post opens with an image near the top.
 */

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

        if (isFirstImage) {
          // Keep the first image eager — likely LCP candidate.
          isFirstImage = false;
        } else if (node.properties.loading === undefined) {
          node.properties.loading = "lazy";
        }

        if (node.properties.decoding === undefined) {
          node.properties.decoding = "async";
        }
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
