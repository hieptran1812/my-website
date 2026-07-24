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
 * Supported types: note, tip, important, warning, caution, info, success, error, tldr.
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
  "tldr",
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
  tldr: "TL;DR",
};

interface HtmlNode {
  type: "html";
  value: string;
}

/** Titles come straight from the markdown source and land in raw HTML. */
function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
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
      // Group 2 is any text left on the marker's own line — Obsidian treats it
      // as a custom title, so `> [!tldr] TL;DR` must not repeat "TL;DR" in the body.
      const match = /^\[!([A-Za-z]+)\][^\S\n]*([^\n]*)(?:\n|$)/.exec(
        textNode.value,
      );
      if (!match) continue;

      const kind = match[1].toLowerCase();
      if (!KNOWN_TYPES.has(kind)) continue;

      // Only lift the same-line text into the title when the whole marker line
      // lives inside this text node — otherwise inline markup after it (say
      // `> [!note] see **this**`) would be orphaned in the body.
      const lineEndedHere =
        match[0].endsWith("\n") || para.children.length === 1;
      const customTitle = lineEndedHere ? match[2].trim() : "";

      // Strip the marker — plus its same-line title, when we took it — from the
      // first text node.
      const consumed = customTitle ? match[0].length : match[0].indexOf("]") + 1;
      const remainder = textNode.value.slice(consumed).replace(/^[^\S\n]*\n?/, "");
      if (remainder.length === 0) {
        para.children.shift();
      } else {
        textNode.value = remainder;
      }
      // If the paragraph is now empty, remove it.
      if (para.children.length === 0) {
        bq.children.shift();
      }

      const label = escapeHtml(customTitle || TITLE_LABEL[kind] || kind);
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
