#!/usr/bin/env python3
"""Repair existing Substack drafts: attach cover images, finish missing tags.

The `substack` CLI has no cover support at all — no `cover` anywhere in the
package — so this reaches for the two low-level calls that do the job:
`get_image()` uploads and returns a Substack-hosted URL, `put_draft()` writes
`cover_image` onto the draft.

Tags are here for a different reason: `drafts create` adds them one API call at
a time *after* the draft exists, so a rate limit mid-way leaves a real draft with
partial tags. `add_tags_to_post()` finishes the job without deleting anything.

Covers are passed as **public URLs**, not local paths: `get_image()` labels every
local file it reads as `data:image/jpeg;base64,` regardless of the real type,
which is wrong for the .webp files this repo ships. Handing Substack the URL
lets its own fetcher sort out the format.

Input on stdin:

    {"publication_url": "https://x.substack.com",
     "delay": 5,
     "items": [{"slug": "...", "id": 123, "cover": "https://..."}]}

Output: one JSON object per line, {"slug":..., "ok":true|false, ...}.
Authentication comes from COOKIES_STRING / COOKIES_PATH in the environment.
"""

import json
import os
import sys
import time

from substack import Api


def main() -> int:
    payload = json.load(sys.stdin)
    items = payload.get("items", [])
    delay = payload.get("delay", 5)

    api = Api(
        cookies_string=os.getenv("COOKIES_STRING"),
        cookies_path=os.getenv("COOKIES_PATH"),
        publication_url=payload.get("publication_url") or os.getenv("PUBLICATION_URL"),
    )

    failures = 0
    for index, item in enumerate(items):
        if index:
            time.sleep(delay)
        slug, draft_id = item["slug"], item["id"]
        cover, tags = item.get("cover"), item.get("tags") or []
        result = {"slug": slug, "id": draft_id, "ok": True}
        errors = []

        # Cover and tags are reported independently: one failing must not throw
        # away a success the caller needs to record, or it gets redone next run.
        if cover:
            try:
                uploaded = api.get_image(cover)
                url = uploaded.get("url") if isinstance(uploaded, dict) else uploaded
                if not url:
                    raise RuntimeError(f"no url in image response: {uploaded!r}")
                api.put_draft(draft_id, cover_image=url)
                result["cover"] = url
            except Exception as exc:  # noqa: BLE001
                errors.append(f"cover: {exc}")

        # One call per tag, because add_tags_to_post() aborts the whole list on
        # the first "Tag already set" 400 — which is exactly what a re-run hits.
        applied = 0
        for tag in tags:
            try:
                api.add_tag_to_post(draft_id, tag)
                applied += 1
            except Exception as exc:  # noqa: BLE001
                if "already set" not in str(exc).lower():
                    errors.append(f"tag {tag}: {exc}")
            time.sleep(0.5)
        if tags:
            result["tags"] = applied
            result["tagsTotal"] = len(tags)

        if errors:
            result["ok"] = False
            result["error"] = "; ".join(errors)[:300]
            failures += 1
        print(json.dumps(result), flush=True)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
