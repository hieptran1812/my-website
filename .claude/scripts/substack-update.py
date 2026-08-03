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
import sys
import time

from substack import Api
from substack.post import Post


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
            post = Post(item.get("title") or slug, item.get("subtitle") or "", user_id)
            post.from_markdown(markdown, api=api)
            body = post.get_draft()["draft_body"]
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
