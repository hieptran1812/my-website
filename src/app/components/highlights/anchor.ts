/**
 * Character-offset anchoring over a root element.
 * Offsets are computed against the concatenated textContent of text nodes
 * that live inside the root, skipping nodes inside <pre>, <code>, and
 * elements marked [data-bh-skip].
 */

const SKIP_SELECTOR = "pre, code, [data-bh-skip]";

function shouldSkip(node: Node): boolean {
  let el: Node | null = node;
  while (el && el.nodeType !== 1) el = el.parentNode;
  if (!el) return false;
  return !!(el as Element).closest?.(SKIP_SELECTOR);
}

function walker(root: Node): TreeWalker {
  return document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      if (!node.nodeValue) return NodeFilter.FILTER_REJECT;
      if (shouldSkip(node)) return NodeFilter.FILTER_REJECT;
      return NodeFilter.FILTER_ACCEPT;
    },
  });
}

/** Absolute char offset of a (node, offset) pair within root. */
export function offsetFromPoint(
  root: Node,
  node: Node,
  offset: number,
): number | null {
  if (!root.contains(node)) return null;
  if (shouldSkip(node)) return null;
  const w = walker(root);
  let total = 0;
  let current = w.nextNode();
  while (current) {
    if (current === node) return total + offset;
    total += current.nodeValue!.length;
    current = w.nextNode();
  }
  return null;
}

export function rangeToOffsets(
  root: Node,
  range: Range,
): { start: number; end: number } | null {
  const start = offsetFromPoint(root, range.startContainer, range.startOffset);
  const end = offsetFromPoint(root, range.endContainer, range.endOffset);
  if (start == null || end == null) return null;
  if (end <= start) return null;
  return { start, end };
}

/**
 * Walk text nodes within [start,end) and invoke cb for each slice.
 * cb receives (textNode, localStart, localEnd).
 */
export function forEachSlice(
  root: Node,
  start: number,
  end: number,
  cb: (node: Text, from: number, to: number) => void,
): void {
  const w = walker(root);
  let pos = 0;
  const queue: Array<[Text, number, number]> = [];
  let n = w.nextNode() as Text | null;
  while (n) {
    const len = n.nodeValue!.length;
    const nodeStart = pos;
    const nodeEnd = pos + len;
    if (nodeEnd > start && nodeStart < end) {
      const from = Math.max(0, start - nodeStart);
      const to = Math.min(len, end - nodeStart);
      if (to > from) queue.push([n, from, to]);
    }
    pos = nodeEnd;
    if (pos >= end) break;
    n = w.nextNode() as Text | null;
  }
  // apply in reverse order within each node to avoid offset shifts
  // (we're wrapping; each node processed once)
  for (const [node, from, to] of queue) cb(node, from, to);
}

export function extractText(root: Node, start: number, end: number): string {
  let out = "";
  forEachSlice(root, start, end, (node, from, to) => {
    out += node.nodeValue!.slice(from, to);
  });
  return out;
}
