/**
 * remark plugin: convert GitHub-style alerts inside blockquotes into
 * styled callout containers.
 *
 *   > [!NOTE]
 *   > Body text…
 *
 * becomes
 *
 *   <div class="callout callout-note">
 *     <div class="callout-title">Note</div>
 *     <div class="callout-body"><p>Body text…</p></div>
 *   </div>
 *
 * Supported types: note, tip, important, warning, caution, info, success, error.
 */

import type { Plugin } from "unified";
import type { Root, Blockquote, Paragraph, Text } from "mdast";

const KNOWN_TYPES = new Set([
  "note",
  "tip",
  "important",
  "warning",
  "caution",
  "info",
  "success",
  "error",
]);

const TITLE_LABEL: Record<string, string> = {
  note: "Note",
  tip: "Tip",
  important: "Important",
  warning: "Warning",
  caution: "Caution",
  info: "Info",
  success: "Success",
  error: "Error",
};

interface HtmlNode {
  type: "html";
  value: string;
}

const remarkCallouts: Plugin<[], Root> = () => {
  return (tree) => {
    if (!tree || !Array.isArray(tree.children)) return;

    for (let i = 0; i < tree.children.length; i++) {
      const node = tree.children[i];
      if (node.type !== "blockquote") continue;

      const bq = node as Blockquote;
      const first = bq.children[0];
      if (!first || first.type !== "paragraph") continue;

      const para = first as Paragraph;
      const firstChild = para.children[0];
      if (!firstChild || firstChild.type !== "text") continue;

      const textNode = firstChild as Text;
      const match = /^\[!([A-Za-z]+)\]\s*\n?/.exec(textNode.value);
      if (!match) continue;

      const kind = match[1].toLowerCase();
      if (!KNOWN_TYPES.has(kind)) continue;

      // Strip the marker from the first text node (keep any text on the same line as title-extension).
      const remainder = textNode.value.slice(match[0].length);
      if (remainder.length === 0) {
        para.children.shift();
      } else {
        textNode.value = remainder;
      }
      // If the paragraph is now empty, remove it.
      if (para.children.length === 0) {
        bq.children.shift();
      }

      const label = TITLE_LABEL[kind] ?? kind;
      const openHtml: HtmlNode = {
        type: "html",
        value: `<div class="callout callout-${kind}"><div class="callout-title"><span class="callout-icon" aria-hidden="true"></span>${label}</div><div class="callout-body">`,
      };
      const closeHtml: HtmlNode = {
        type: "html",
        value: `</div></div>`,
      };

      // Replace the blockquote with: open-html + its (now-stripped) children + close-html.
      // We keep them as siblings so remark-rehype renders the inner mdast normally.
      tree.children.splice(
        i,
        1,
        openHtml as unknown as Root["children"][number],
        ...bq.children,
        closeHtml as unknown as Root["children"][number],
      );
      i += 1 + bq.children.length; // skip past the inserted nodes
    }
  };
};

export default remarkCallouts;
