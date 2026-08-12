#!/usr/bin/env python3
"""Rewrite the body of existing Substack drafts from fresh Markdown.

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
from substack.post import Post


IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


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
            # literal text, so remove it before parsing the API body.
            markdown = markdown.replace(r"\$", "$")
            post = Post(item.get("title") or slug, item.get("subtitle") or "", user_id)
            post.from_markdown(markdown, api=api)
            draft = post.get_draft()
            body = json.loads(draft["draft_body"])
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
