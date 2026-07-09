# Figure extraction reference

Read at the start of Phase C1, before cutting the first figure out of the PDF. This is the pipeline that lets a paper analysis show the reader **the authors' actual Figure 1**, not a memory or a re-drawing of it.

## The idea in one paragraph

You locate a figure on a **low-DPI page render** (a thumbnail you `Read`), estimate its bounding box as **fractions of the page** (`[x0, y0, x1, y1]` in 0..1), then `extract-figures.py` re-renders that page at **high DPI**, crops the fractional box, **auto-trims** the surrounding white margin, and ships a **lossless WebP**. Fractions are resolution-independent, so the box you eyeball on a 120-DPI thumbnail applies unchanged to a 400-DPI render. Auto-trim forgives a loose estimate: give a box that is a little too generous (but excludes the caption) and it tightens to the ink.

Coordinates: `x0,y0` = top-left, `x1,y1` = bottom-right, origin at the page's top-left. `y` grows **downward**.

## The one asymmetry that matters: trim reclaims margin, it never recovers clipped ink

Auto-trim can only *remove* whitespace you included; it can never *add back* figure content your box cut off. So the two directions of error are not equal:

- **Too generous** (box extends into whitespace) → free. Trim tightens to the ink. Zero cost.
- **Too tight** (box crosses into the figure) → permanent clip. A box, label, axis tick, or arrowhead is sliced off and no re-render fixes it without changing the box.

Therefore the estimation strategy is **asymmetric by edge**, not "eyeball all four tightly":

- **An edge that borders only whitespace** (page margin, blank gutter between the figure and nothing) → be *generous*: push it well past the visible ink, even to the page margin. Trim reclaims it. On a page whose only content is the figure + caption, `x0`/`x1` can safely be `0.08`/`0.92` and `y0` the top margin — you are not "measuring" these edges, you are giving trim room to work.
- **An edge that borders the caption or body text** → be *tight and careful*. This is the only edge worth measuring to ~0.01. Set `y1` just **above** the "Figure N:" line; set the text-facing edge just **outside** the nearest prose line. Here, and only here, too-generous means foreign text bleeds into the crop (which trim will *keep*, since it is ink).

Rule of thumb: **spend your measurement effort on the ≤2 edges touching text; make the whitespace-facing edges loose on purpose.** The fig that clips is almost always one where a whitespace-facing edge was estimated as if it needed to be tight.

### Watch the extremities: connectors, loops, and arrowheads reach past the box cluster

A figure's visual "footprint" is usually read as the cluster of labelled boxes — but diagrams routinely have **feedback loops, side connectors, return arrows, colorbars, or legends that extend well past that cluster** (a loop arrow wrapping down the right side; a `rejects`/`no` edge routed outside the boxes; a shared axis label hanging below). These are part of the figure and get clipped by a box drawn to the box-cluster edge. When a diagram has any routing that leaves and re-enters the box grid, **extend the box past the outermost stroke, not the outermost box** — and since that edge faces whitespace, being generous is free.

## Pipeline

`get-paper.sh` (Phase A) already rendered `pages/page-N.png` at ~120 DPI. Per figure:

1. **`Read` the page thumbnail** and estimate the box. The image is `W×H` px; a feature at pixel `(px, py)` is fraction `(px/W, py/H)`. You don't need pixel-perfect — round to ~0.02. Apply the asymmetry above: **measure only the edges that touch caption or body text; make the whitespace-facing edges deliberately loose** (trim reclaims the margin, but nothing recovers a clipped box or a cut-off connector). Concretely: **set the bottom edge `y1` just above the "Figure N:" caption line** so the caption is excluded, keep any text-facing side edge just outside the nearest prose line, and push the remaining whitespace-facing edges out past the outermost stroke — including feedback loops and side arrows, not just the box cluster.
2. **Write a manifest** `.cache/paper-writer/<slug>/figures.json`:
   ```json
   {
     "pdf": ".cache/paper-writer/<slug>/paper.pdf",
     "slug": "<slug>",
     "dpi": 400,
     "figures": [
       { "n": 1, "page": 3, "box": [0.22, 0.075, 0.74, 0.50], "label": "Transformer architecture" },
       { "n": 2, "page": 5, "box": [0.10, 0.14, 0.92, 0.40], "label": "Scaled dot-product & multi-head attention" }
     ]
   }
   ```
3. **Run it** (renders each page once, crops all figures on that page):
   ```bash
   python3 .claude/skills/paper-writer/scripts/extract-figures.py --manifest .cache/paper-writer/<slug>/figures.json
   ```
   → `public/imgs/blogs/<slug>-fig<n>.webp` for each entry. The `-fig<n>` infix is load-bearing: it marks the image as an **extracted original**, which the verify gate holds to a relaxed sharpness floor (a paper figure is rarely 16:9) and counts toward the "≥ 2 extracted" requirement. Redrawn Excalidraw diagrams use the plain `<slug>-<n>.webp` naming instead.
4. **`Read` each output WebP** and check the crop (Phase C3): tight to the figure, no caption, no body text, no adjacent figure, nothing clipped.

Single-figure mode (quick iteration on one box):
```bash
python3 .claude/skills/paper-writer/scripts/extract-figures.py \
  --pdf .cache/paper-writer/<slug>/paper.pdf --slug <slug> --n 1 --page 3 \
  --box 0.22 0.075 0.74 0.50 --dpi 400 --label "Transformer architecture"
```

## Tuning knobs (fields on a manifest entry, or CLI flags)

| Field / flag        | Default | When to change |
| ------------------- | ------- | -------------- |
| `dpi`               | 400     | Raise to 500–600 if the crop is soft or below the size floor; a half-page figure is ~1600 px at 400 DPI, a full-width figure ~2600 px. |
| `box`               | —       | The fractional crop. Loosen it if content is clipped; tighten (or trim harder) if foreign text bleeds in. |
| `trim` / `--no-trim`| `true`  | Turn **off** when the figure has intentional white interior regions that touch its own edge (rare — e.g. a plot with a white legend box at the corner) and auto-trim eats them. |
| `pad`               | 14      | White border re-added after trim. Increase for a bit of breathing room; decrease for a very tight crop. |
| `trim_thresh`       | 250     | Pixels ≥ this (near-pure-white) are "margin". Lower to ~245 to trim slightly-off-white scan backgrounds; raise toward 253 if trim is eating light-colored fills. |
| `min_long_side`     | 1400    | Long side the script upscales (LANCZOS, ≤ 2.5×) to if the crop is smaller. Prefer raising `dpi` over relying on upscale. |

## Auto-trim: what it does and its guardrails

After the fractional crop, the script converts to grayscale, masks everything darker than `trim_thresh` as content, takes the bounding box of that content, expands by `pad`, and re-crops. Guardrails:

- If the crop is **blank** (no content found), trim is skipped and a warning prints — your box missed the figure; fix the box.
- If trim would remove **> 97%** of the area, it's skipped (the threshold is wrong for this figure) — usually means a very light figure; lower `trim_thresh` or set `--no-trim`.

Because trim tightens to the ink, **always exclude the caption in your box** — otherwise the caption text becomes part of the "ink" and trim keeps it.

## Sharpness floor (Phase E gate)

Extracted originals must be **≥ 900 px on the long side and ≥ 20 KB**. My reference extraction of the Transformer figure came out 1238×1807 px, 51 KB at 400 DPI — comfortably clear. If a figure lands under the floor, it's almost always DPI: bump to 500–600 and re-extract. A lossless WebP under 20 KB means the crop is nearly empty (bad box) or tiny (bump DPI).

## Fallback: `pdfimages` for embedded bitmaps

The render-and-crop path works for **any** figure — vector diagrams, plots, tables — and gives you consistent DPI and clean cropping. It is the default. Reach for `pdfimages` only when a figure is a **high-resolution embedded photo/screenshot** and the rendered crop looks softer than the source raster:

```bash
# list embedded images with page, native resolution, and object id
pdfimages -list .cache/paper-writer/<slug>/paper.pdf | head -40
# dump the images on page 7 as PNGs
pdfimages -png -f 7 -l 7 .cache/paper-writer/<slug>/paper.pdf .cache/paper-writer/<slug>/raw
```
Then pick the right PNG, crop if needed (Pillow, or feed it back through the script's box on a re-render), and `cwebp -lossless -m 6` it to `public/imgs/blogs/<slug>-fig<n>.webp`. Note: `pdfimages` returns the raw stored bitmap (often unclipped and without vector overlays/labels), so composite figures — a photo with vector arrows drawn on top — come out *incomplete*. For those, stay on the render-and-crop path.

## Common failures and fixes

| Symptom | Cause | Fix |
| ------- | ----- | --- |
| Caption line included at the bottom | `y1` too low | Raise `y1` to just above the "Figure N:" text |
| Adjacent figure/text bleeds in on a side | box too wide on that axis | Tighten the offending edge fraction |
| Figure clipped (a box/label cut off) | box too tight on a whitespace-facing edge | Loosen that edge generously; trim re-tightens to real ink |
| A feedback loop / return arrow / side connector sliced off | box drawn to the box *cluster*, missing routing that reaches past it | Extend the edge past the outermost *stroke*, not the outermost box |
| Two sub-figures (a) and (b) you want separately | one box covers both | Two manifest entries, two boxes, same page |
| Soft / pixelated text | DPI too low | Raise `dpi` to 500–600, re-extract |
| WebP < 20 KB | near-empty crop or tiny figure | Fix the box, or raise DPI |
| Trim ate part of a light figure | `trim_thresh` too high | Lower to ~245, or `--no-trim` |

## Crediting (Phase D)

Every extracted figure's markdown caption **credits the paper**, e.g.:

```markdown
![Figure 1 from Vaswani et al. (2017): the Transformer encoder–decoder stack](/imgs/blogs/attention-is-all-you-need-fig1.webp)
```

Use "Figure N from <Authors> (<year>)" so the reader knows it is the authors' own figure, not yours. If you had to redraw a figure the extractor couldn't cleanly cut, say "redrawn from Figure N of <paper>" instead and ship it as a `<slug>-<n>.webp` Excalidraw diagram.
