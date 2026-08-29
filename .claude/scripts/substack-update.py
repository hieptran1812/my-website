#!/usr/bin/env python3
"""Rewrite the body of existing Substack drafts from fresh Markdown.

RUN THIS UNDER THE uv TOOL'S PYTHON, NOT bare `python3`:

    /Users/hieptran1812/.local/share/uv/tools/python-substack/bin/python \
        .claude/scripts/substack-update.py < input.json

The default `python3` here is 3.9 (micromamba) and carries an OLD python-substack
with no `mdrender.py` and no dollarmath support. Its `from_markdown` turns a
`$$...$$` block into three literal paragraphs -- `$$`, the LaTeX source, `$$` --
so every display formula ships as visible source. The uv install (3.12) has the
dollarmath plugin and emits a real `latex_block`. The guard below refuses to run
under the wrong one rather than silently shipping broken math.

Used when a draft is already in place — with its id, tags and cover — but the
body needs regenerating. Deleting and recreating would lose all of that and burn
a slug, so this converts the Markdown the same way `drafts create` does and PUTs
just `draft_body` over the top.

Images are re-uploaded (the converter does that while rendering), so this costs
the same API calls as a fresh push.

Input on stdin:

    {"publication_url": "https://x.substack.com",
     "delay": 5,
     "items": [{"slug": "...", "id": 123, "markdown": "path/to.md",
                "title": "...", "subtitle": "..."}]}

Output: one JSON object per line.
"""

import json
import os
import re
import sys
import time
from pathlib import Path

from PIL import Image
from substack import Api
from substack.post import Post, parse_inline


def _require_math_capable_substack():
    """Refuse to run under an install whose from_markdown cannot do `$$`."""
    try:
        import substack.mdrender  # noqa: F401
    except ImportError:
        sys.exit(
            "substack-update: this python's python-substack has no mdrender "
            "(no dollarmath), so every $$...$$ would ship as literal source.\n"
            "Re-run with the uv tool python:\n"
            "  /Users/hieptran1812/.local/share/uv/tools/python-substack/bin/python "
            f"{Path(__file__).name} < input.json"
        )


_require_math_capable_substack()


IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
BULLET_RE = re.compile(r"^[-*]\s+")


def text_node(token):
    """Turn one python-substack inline token into a valid ProseMirror text node."""
    node = {"type": "text", "text": token["content"]}
    if token.get("marks"):
        node["marks"] = token["marks"]
    return node


def repair_inline_nodes(node):
    """Give every inline token the node shape ProseMirror requires.

    python-substack renders bullet-list text straight from parse_inline, so the
    items land as {"content": "…", "marks": […]} — no `type`, and the text under
    `content` rather than `text`. Substack's editor cannot read that, so the
    whole list shows up empty. Paragraphs go through Post.text() and are fine;
    this rewrites the ones that don't.
    """
    if isinstance(node, dict):
        if "type" not in node and isinstance(node.get("content"), str):
            text, marks = node["content"], node.get("marks")
            node.clear()
            node["type"] = "text"
            node["text"] = text
            if marks:
                node["marks"] = marks
        for value in node.values():
            repair_inline_nodes(value)
    elif isinstance(node, list):
        for value in node:
            repair_inline_nodes(value)
    return node


def blockquote_runs(markdown):
    """Every contiguous run of `>` lines in the source, quote markers stripped."""
    runs, current = [], None
    for line in markdown.split("\n"):
        if line.startswith(">"):
            if current is None:
                current = []
            current.append(line[1:].lstrip(" ") if line != ">" else "")
        elif current is not None:
            runs.append(current)
            current = None
    if current is not None:
        runs.append(current)
    return runs


def unescape_dollars_outside_math(markdown):
    """Strip the site's `\\$` escape, but never inside a math region.

    The exporter escapes currency signs for the site's Markdown renderer, and
    python-substack reads that escape as literal text, so it has to come off.
    But inside `$$...$$`, `\\$` is *correct LaTeX* for a literal dollar, and
    unescaping it leaves a bare `$` in the middle of the block. dollarmath then
    mis-terminates the block and the whole formula ships as visible source --
    which is exactly how the banking and MACD posts broke.

    So: unescape outside math, leave math regions byte-for-byte alone.
    """
    out, fenced = [], False
    for line in markdown.split("\n"):
        if line.strip() == "$$":
            fenced = not fenced
            out.append(line)
            continue
        if fenced:
            out.append(line)
            continue
        # Protect single-line `$$...$$` spans, then unescape what is left.
        parts = re.split(r"(\$\$.*?\$\$)", line)
        out.append("".join(
            part if part.startswith("$$") else re.sub(r"\\+\$", "$", part)
            for part in parts
        ))
    return "\n".join(out)


def build_blockquote(lines):
    """Rebuild one blockquote: paragraphs keep their marks, `-` lines become a list."""
    content, bullets = [], []

    def flush_bullets():
        if not bullets:
            return
        content.append({
            "type": "bullet_list",
            "content": [
                {"type": "list_item",
                 "content": [{"type": "paragraph", "content": nodes}]}
                for nodes in bullets
            ],
        })
        bullets.clear()

    for line in lines:
        if not line.strip():
            flush_bullets()
            continue
        if BULLET_RE.match(line):
            tokens = parse_inline(BULLET_RE.sub("", line))
            if tokens:
                bullets.append([text_node(t) for t in tokens])
            continue
        flush_bullets()
        tokens = parse_inline(line)
        if tokens:
            content.append({"type": "paragraph",
                            "content": [text_node(t) for t in tokens]})
    flush_bullets()

    node = {"type": "blockquote"}
    if content:
        node["content"] = content
    return node


def repair_blockquotes(draft_body, markdown):
    """Restore what python-substack drops inside a blockquote.

    Its quote branch keeps the text but throws away every mark, and leaves the
    `- ` of a quoted bullet sitting in the prose as a literal dash. The TL;DR
    box is a blockquote, so that is the most-read part of the post. Rebuilding
    each quote from the source Markdown puts the bold lead-ins and the bullet
    list back.
    """
    rebuilt = [build_blockquote(lines) for lines in blockquote_runs(markdown)]
    index = 0

    def visit(node):
        nonlocal index
        if isinstance(node, dict):
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for position, value in enumerate(node):
                if isinstance(value, dict) and value.get("type") == "blockquote":
                    if index < len(rebuilt):
                        node[position] = rebuilt[index]
                    index += 1
                    continue
                visit(value)

    visit(draft_body)
    return draft_body


def repair_image_attrs(draft_body, markdown):
    """Preserve each local image's aspect ratio and alt text.

    python-substack's captioned_image() defaults every image to 1456×819,
    which distorts non-16:9 figures in the Substack editor. Keep the editor's
    2x content width, but derive height from the actual local image dimensions.
    """
    images = IMAGE_RE.findall(markdown)
    image_index = 0

    def visit(node):
        nonlocal image_index
        if isinstance(node, dict):
            if node.get("type") == "image2":
                if image_index >= len(images):
                    return
                alt, source = images[image_index]
                attrs = node.setdefault("attrs", {})
                attrs["alt"] = alt or None
                source_path = Path(source)
                if source_path.exists():
                    with Image.open(source_path) as image:
                        source_width, source_height = image.size
                    display_width = 1456
                    display_height = max(
                        1, round(display_width * source_height / source_width)
                    )
                    attrs["width"] = display_width
                    attrs["height"] = display_height
                    attrs["resizeWidth"] = 728
                image_index += 1
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(draft_body)
    return draft_body


def main() -> int:
    payload = json.load(sys.stdin)
    items = payload.get("items", [])
    delay = payload.get("delay", 5)

    api = Api(
        cookies_string=os.getenv("COOKIES_STRING"),
        cookies_path=os.getenv("COOKIES_PATH"),
        publication_url=payload.get("publication_url") or os.getenv("PUBLICATION_URL"),
    )
    user_id = api.get_user_id()

    failures = 0
    for index, item in enumerate(items):
        if index:
            time.sleep(delay)
        slug, draft_id = item["slug"], item["id"]
        try:
            with open(item["markdown"], encoding="utf-8") as handle:
                markdown = handle.read()
            # The generated Substack Markdown escapes currency signs for the
            # site's Markdown renderer. python-substack treats that escape as
            # literal text, so remove it before parsing the API body. A source
            # that already wrote `\$` comes out of the exporter as `\\$`, so
            # strip the whole run rather than a single backslash.
            markdown = unescape_dollars_outside_math(markdown)
            post = Post(item.get("title") or slug, item.get("subtitle") or "", user_id)
            post.from_markdown(markdown, api=api)
            draft = post.get_draft()
            body = json.loads(draft["draft_body"])
            body = repair_inline_nodes(body)
            body = repair_blockquotes(body, markdown)
            body = repair_image_attrs(body, markdown)
            body = json.dumps(body)
            api.put_draft(draft_id, draft_body=body)
            print(json.dumps({"slug": slug, "id": draft_id, "ok": True,
                              "bytes": len(body)}), flush=True)
        except Exception as exc:  # noqa: BLE001 - report and keep going
            failures += 1
            print(json.dumps({"slug": slug, "id": draft_id, "ok": False,
                              "error": str(exc)[:300]}), flush=True)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
