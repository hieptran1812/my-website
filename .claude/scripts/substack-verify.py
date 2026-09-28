#!/usr/bin/env python3
"""Verify Substack drafts and published posts. Reads COOKIES_STRING from the env.

Three checks, in the order they matter:

  drafts   <slug>...   the document Substack stores. An **inline `latex` node makes
                       the published body_html empty**, which once shipped 101 blank
                       posts here, so that count must be 0. Also catches image srcs
                       that were never uploaded and leftover ZZMATH placeholders.
  posts    <slug>...   the rendered page. `body_html` length is the only thing that
                       proves a post is not blank; a valid document is not a
                       rendered page. Also counts LaTeX commands leaking as source.
  latex    <slug>...   just the source-leak count, for a quick sweep after a fix.

Query the POSTS endpoint for body_html: /drafts/<id> does not carry that field and
returns 0, which looks exactly like a blank post and is not one.

Usage: substack-verify.py {drafts|posts|latex} <slug> [slug...]
"""
import json, os, re, sys, time, urllib.request

PUB = "https://halleytech.substack.com"
HDR = {"Cookie": os.environ["COOKIES_STRING"],
       "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
REC = json.load(open(".substack-created.json"))

def get(url):
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=HDR)))

def node_types(doc):
    kinds = {}
    def walk(n):
        if isinstance(n, dict):
            t = n.get("type")
            if t:
                kinds[t] = kinds.get(t, 0) + 1
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)
    walk(doc)
    return kinds

def leaked_latex(html):
    """LaTeX commands visible as prose, ignoring Substack's own maths payloads."""
    return len(re.findall(r'\\[a-zA-Z]{2,}', re.sub(r'data-attrs="[^"]*"', ' ', html)))

def check_draft(slug):
    body = get(f"{PUB}/api/v1/drafts/{REC[slug]['id']}").get("draft_body") or "{}"
    doc = json.loads(body) if isinstance(body, str) else body
    k = node_types(doc)
    txt = json.dumps(doc)
    inline = k.get("latex", 0)
    raw_src = len(re.findall(r'"src":\s*"/imgs/', txt))
    zz = txt.count("ZZMATH")
    ok = inline == 0 and raw_src == 0 and zz == 0 and len(body) > 10000
    print(f"  {'OK ' if ok else 'BAD'} {slug[:44]:46} {len(body):7,}B "
          f"latex_inline={inline} block={k.get('latex_block',0)} "
          f"img={k.get('image2',0)} raw_src={raw_src} zz={zz}")
    return ok

def check_post(slug):
    d = get(f"{PUB}/api/v1/posts/{slug}")
    h = d.get("body_html") or ""
    leak = leaked_latex(h)
    ok = (d.get("is_published") and len(h) > 30000
          and bool(d.get("cover_image")) and leak == 0)
    print(f"  {'OK ' if ok else 'BAD'} {slug[:44]:46} {len(h):7,}B "
          f"{d.get('audience'):9} sec={d.get('section_id')} "
          f"cover={'y' if d.get('cover_image') else 'N'} leak={leak}")
    return ok

def check_latex(slug):
    h = get(f"{PUB}/api/v1/posts/{slug}").get("body_html") or ""
    n = leaked_latex(h)
    if n:
        print(f"  {n:4} LaTeX visible as source  {slug[:48]}")
    return n == 0

MODES = {"drafts": check_draft, "posts": check_post, "latex": check_latex}
mode, slugs = sys.argv[1], sys.argv[2:]
fn = MODES[mode]
good = 0
for s in slugs:
    try:
        good += bool(fn(s))
    except Exception as e:
        print(f"  ERR {s[:44]:46} {type(e).__name__}: {str(e)[:80]}")
    time.sleep(1.1)
print(f"\n  {good}/{len(slugs)} passed")
