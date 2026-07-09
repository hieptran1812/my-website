#!/usr/bin/env python3
"""
Extract figures (architecture diagrams, plots, tables rendered as images) out of a
paper PDF and ship them as lossless WebP into public/imgs/blogs/.

Pipeline per figure:
  1. render the source page at high DPI with pdftoppm  (cached per page+dpi)
  2. crop a *fractional* bounding box  [x0,y0,x1,y1] in 0..1 page coords  (Pillow)
  3. auto-trim near-white margins so the crop is tight to the ink   (optional)
  4. pad back a few px of white, upscale to a min long-side if tiny
  5. cwebp -lossless  ->  public/imgs/blogs/<slug>-fig<n>.webp

Why fractional boxes: you eyeball the box on a LOW-dpi page render (via Read), but the
crop is applied to a HIGH-dpi render — fractions are resolution independent, so the same
box works at any dpi. Auto-trim then forgives a loose estimate: give a box that is a
little too generous (but excludes the caption text) and the trim tightens it to the art.

Usage (single figure):
  python3 extract-figures.py --pdf paper.pdf --slug my-slug --n 1 --page 3 \
      --box 0.12 0.08 0.88 0.52 --dpi 400 --label "Transformer architecture"

Usage (batch, preferred — renders each page once):
  python3 extract-figures.py --manifest figures.json

manifest.json:
  {
    "pdf": "/abs/path/paper.pdf",
    "slug": "my-slug",
    "dpi": 400,                       # default dpi for every figure
    "out_dir": "public/imgs/blogs",   # default
    "cache_dir": ".cache/paper-writer/my-slug",   # default derived from slug
    "figures": [
      { "n": 1, "page": 3, "box": [0.12,0.08,0.88,0.52], "label": "Transformer architecture" },
      { "n": 2, "page": 5, "box": [0.10,0.55,0.92,0.80], "dpi": 500, "trim": true, "pad": 16 }
    ]
  }

Output for figure n is public/imgs/blogs/<slug>-fig<n>.webp  (the -fig<n> infix marks
it as an EXTRACTED original, distinct from redrawn Excalidraw diagrams <slug>-<n>.webp).
"""
import argparse, json, os, subprocess, sys
from PIL import Image

DEFAULT_DPI = 400
DEFAULT_TRIM_THRESH = 250   # pixels >= this (near-pure-white) count as background margin
DEFAULT_PAD = 14            # px of white re-added around the trimmed ink
DEFAULT_MIN_LONG = 1400     # upscale (LANCZOS) so max(w,h) >= this, keeps figures crisp
MAX_UPSCALE = 2.5           # never upscale more than this (blur guard) — bump --dpi instead


def log(msg): print(msg, file=sys.stderr)


def render_page(pdf, page, dpi, cache_dir):
    """Render one PDF page to PNG at dpi, cached by (page,dpi). Returns the PNG path."""
    os.makedirs(cache_dir, exist_ok=True)
    stem = os.path.join(cache_dir, f"page-{page}-{dpi}dpi")
    out = stem + ".png"
    if os.path.exists(out):
        return out
    # -singlefile makes pdftoppm write exactly <stem>.png (no -NN suffix) for a 1-page range
    subprocess.run(
        ["pdftoppm", "-png", "-singlefile", "-r", str(dpi),
         "-f", str(page), "-l", str(page), pdf, stem],
        check=True,
    )
    if not os.path.exists(out):
        raise SystemExit(f"pdftoppm produced no output for page {page} at {dpi}dpi")
    return out


def autotrim(img, thresh, pad):
    """Trim near-white margins; re-pad `pad` px of white. Returns trimmed image."""
    gray = img.convert("L")
    # content mask: pixels darker than thresh become white(255), background -> 0
    mask = gray.point(lambda p: 255 if p < thresh else 0)
    bbox = mask.getbbox()
    if bbox is None:
        log("  ! auto-trim found no ink (blank crop?) — keeping untrimmed")
        return img
    x0, y0, x1, y1 = bbox
    # sanity: if trim would nuke >97% of area, the threshold is wrong for this figure
    if (x1 - x0) * (y1 - y0) < 0.03 * img.width * img.height:
        log("  ! auto-trim would remove almost everything — keeping untrimmed")
        return img
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(img.width, x1 + pad); y1 = min(img.height, y1 + pad)
    return img.crop((x0, y0, x1, y1))


def extract_one(pdf, cache_dir, out_dir, slug, fig):
    n = fig["n"]
    page = fig["page"]
    dpi = int(fig.get("dpi", fig.get("_default_dpi", DEFAULT_DPI)))
    box = fig["box"]
    trim = fig.get("trim", True)
    pad = int(fig.get("pad", DEFAULT_PAD))
    thresh = int(fig.get("trim_thresh", DEFAULT_TRIM_THRESH))
    min_long = int(fig.get("min_long_side", DEFAULT_MIN_LONG))
    label = fig.get("label", "")

    if not (len(box) == 4 and all(0.0 <= float(v) <= 1.0 for v in box)):
        raise SystemExit(f"fig {n}: box must be 4 fractions in 0..1, got {box}")
    fx0, fy0, fx1, fy1 = (float(v) for v in box)
    if fx1 <= fx0 or fy1 <= fy0:
        raise SystemExit(f"fig {n}: box must have x1>x0 and y1>y0, got {box}")

    page_png = render_page(pdf, page, dpi, cache_dir)
    im = Image.open(page_png).convert("RGB")
    W, H = im.size
    crop = im.crop((round(fx0 * W), round(fy0 * H), round(fx1 * W), round(fy1 * H)))

    if trim:
        crop = autotrim(crop, thresh, pad)

    w, h = crop.size
    long_side = max(w, h)
    if long_side < min_long:
        scale = min(min_long / long_side, MAX_UPSCALE)
        if scale > 1.01:
            crop = crop.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
            log(f"  upscaled x{scale:.2f} (was {w}x{h}, long side < {min_long})")

    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(cache_dir, f"{slug}-fig{n}.png")
    webp_path = os.path.join(out_dir, f"{slug}-fig{n}.webp")
    crop.save(png_path)
    subprocess.run(["cwebp", "-quiet", "-lossless", "-m", "6", png_path, "-o", webp_path],
                   check=True)
    kb = os.path.getsize(webp_path) / 1024
    fw, fh = crop.size
    tag = f'  "{label}"' if label else ""
    log(f"fig {n}: page {page} @ {dpi}dpi -> {webp_path}  {fw}x{fh}  {kb:.0f}KB{tag}")
    if kb < 20:
        log(f"  ! WARNING fig {n} is only {kb:.0f}KB — likely too small/sparse; "
            f"raise --dpi or widen the box")
    return {"n": n, "webp": webp_path, "w": fw, "h": fh, "kb": round(kb, 1)}


def main():
    ap = argparse.ArgumentParser(description="Extract paper figures to WebP.")
    ap.add_argument("--manifest", help="JSON manifest (batch mode)")
    ap.add_argument("--pdf"); ap.add_argument("--slug")
    ap.add_argument("--n", type=int); ap.add_argument("--page", type=int)
    ap.add_argument("--box", nargs=4, type=float, metavar=("X0", "Y0", "X1", "Y1"))
    ap.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    ap.add_argument("--label", default="")
    ap.add_argument("--no-trim", action="store_true")
    ap.add_argument("--pad", type=int, default=DEFAULT_PAD)
    ap.add_argument("--out-dir", default="public/imgs/blogs")
    ap.add_argument("--cache-dir")
    args = ap.parse_args()

    if args.manifest:
        with open(args.manifest) as f:
            m = json.load(f)
        pdf = m["pdf"]; slug = m["slug"]
        out_dir = m.get("out_dir", "public/imgs/blogs")
        cache_dir = m.get("cache_dir", f".cache/paper-writer/{slug}")
        default_dpi = int(m.get("dpi", DEFAULT_DPI))
        figs = m["figures"]
        for fig in figs:
            fig.setdefault("_default_dpi", default_dpi)
        results = [extract_one(pdf, cache_dir, out_dir, slug, fig) for fig in figs]
        log(f"\nextracted {len(results)} figure(s) -> {out_dir}/{slug}-fig*.webp")
    else:
        req = [args.pdf, args.slug, args.n, args.page, args.box]
        if any(v is None for v in req):
            ap.error("single mode needs --pdf --slug --n --page --box")
        cache_dir = args.cache_dir or f".cache/paper-writer/{args.slug}"
        fig = {"n": args.n, "page": args.page, "box": args.box, "dpi": args.dpi,
               "trim": not args.no_trim, "pad": args.pad, "label": args.label}
        extract_one(args.pdf, cache_dir, args.out_dir, args.slug, fig)


if __name__ == "__main__":
    main()
