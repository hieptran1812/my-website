---
title: "OCR, Chart, and Document Data: Manufacturing Ground Truth for Document AI"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "A field guide to building training data for document AI: OCR pairs and the alignment problem, the render-your-own-ground-truth trick that dominates the field, chart-to-table synthesis, layout and reading order, and how to close the synthetic-to-real domain gap — with runnable generators, a worked corpus build, and a symptom-to-fix troubleshooting section."
tags:
  - training-data
  - ocr
  - document-ai
  - chart-understanding
  - synthetic-data
  - layout-analysis
  - reading-order
  - vision-language
  - domain-gap
  - nougat
category: "machine-learning"
subcategory: "Training Data"
author: "Hiep Tran"
featured: true
readTime: 32
---

Most vision-language training advice quietly assumes the label is a caption: a short, forgiving sentence about a photo. Document data breaks that assumption in the first five minutes. When the "image" is a scanned invoice, an arXiv page, a financial chart, or a form, the target is not a gist. It is *every glyph, in the right place, in the right order, with the right structure* — a transcript that a downstream system will parse, index, or compute on. Get one digit wrong in a table cell and the extracted number is silently corrupted; get the reading order wrong on a two-column page and the paragraphs interleave into nonsense.

That single shift — from "describe this" to "reproduce this exactly" — changes everything about how you manufacture the training data. You cannot crowdsource a million perfectly-transcribed pages the way you can crowdsource captions; the annotation cost is brutal and the error rate is high. So the field converged on a different move, one that feels almost like cheating: *generate the document from a source you already control, and the source is the label.* Render LaTeX to a page image and you know the text for free. Sample a data table and render a bar chart and you know every value. This is the dominant trick in document AI, and understanding it — plus its failure mode, the synthetic-to-real domain gap — is most of the job.

![Why document data is not image captioning: captioning wants a gist, document AI must reproduce every glyph, its 2D position, and reading order](/imgs/blogs/ocr-chart-and-document-data-1.webp)

The diagram above is the mental model: on the left, the captioning contract you are used to — one short output, no layout, no reading order, and many outputs count as "correct." On the right, the document contract — a long, exact target where 2D layout and reading order *are* the answer. This post is a tour of how to build data that satisfies the right column: OCR pairs and their alignment problem, synthetic document and chart generation, layout and reading-order supervision, and the domain-gap work that decides whether any of it survives contact with a real scanner. If you have read the companion post on [synthetic data generation](/blog/machine-learning/training-data/synthetic-data-generation), this is the same philosophy applied to pixels instead of tokens.

## 1. Why document, chart, and figure data is different

The senior rule of thumb: **document ground truth is a structured, exact, long target — treat it like a compiler's expected output, not like a caption.** Four properties make it hard, and each one dictates a different data-engineering decision.

| Property | Natural-image captioning | Document / chart / figure AI |
| --- | --- | --- |
| Output length | ~10–20 tokens | 500–4,000 tokens (a full page) |
| Correctness | Many paraphrases pass | One exact transcript; character-level scoring |
| Spatial structure | Ignored | Columns, tables, headers, bounding boxes are load-bearing |
| Reading order | Irrelevant | The order *is* the semantics |
| Dense small text | Rare | The norm: 6–10 pt fonts, subscripts, footnotes |
| Label source | Human describes | Human transcribes (slow) or source is rendered (free) |

**Dense text at small scale.** A single A4 page at 300 dpi holds 2,000–4,000 characters. The model must resolve 8-point body text, distinguish `l`/`1`/`I` and `0`/`O`, and keep going for thousands of tokens without drifting. This is why document models push input resolution hard and why vision-token budgets matter — a theme explored from the compression side in [DeepSeek-OCR's optical context compression](/blog/machine-learning/computer-vision/deepseek-ocr-optical-context-compression) and from the tiling side in [DeepSeek-VL2's dynamic tiling](/blog/machine-learning/computer-vision/deepseek-vl-vl2-dynamic-tiling-moe).

**2D layout is signal, not noise.** In a caption, where the dog sits in the frame is a detail. In a document, whether a number sits in the "Q3" column or the "Q4" column changes its meaning entirely. Tables, multi-column text, headers, footnotes, and figure captions are structural, and the training target has to encode that structure — either as literal markup (Markdown/HTML/LaTeX) or as explicit coordinates.

**Reading order is the answer.** A two-column paper is *not* read left-to-right across the raster. You read the whole left column top-to-bottom, then the whole right column. A naive OCR that emits text in raster order shreds the paragraphs. So reading-order supervision has to be baked into the data, and — as we will see — narrow templates that only ever show single-column pages teach the model the wrong prior.

**The label is expensive to make by hand, and cheap to render.** This is the pivotal economic fact. A human transcribes a complex page in one to several minutes with a non-trivial error rate. Rendering a page from source you already hold takes tens of milliseconds and the label is *exact by construction*. That asymmetry is the reason synthetic generation dominates this corner of the field.

## 2. OCR pairs: image to text, and the alignment problem

The atomic unit of OCR training is a pair: an image (a page, a line, a crop) and its transcribed text. The senior rule here: **every text span in the label must correspond to the exact pixels it was read from — and when it does not, the error is silent and poisonous.**

![OCR pairs and the alignment problem: each transcribed span must map to the exact pixels it came from, and mis-binding silently teaches the wrong text](/imgs/blogs/ocr-chart-and-document-data-3.webp)

The figure walks the happy path down three rows — a title region transcribes to its string, a body paragraph aligns to its source text, a table cell resolves to a (row, col, value) — and then the failure row in red: a skewed, blurred region whose label no longer matches the pixels. That last row is the whole problem with real-scan data. If you obtained the "ground truth" by running an existing OCR engine over a scan (a common bootstrapping shortcut), then wherever that engine erred, you are now training your new model to reproduce the error. And because the loss is computed against the (wrong) label, the model that best fits the data is the one that best reproduces the mistake. There is no signal telling you it happened.

### Measuring OCR quality: CER, WER, and their traps

You score transcription with edit-distance metrics. Character Error Rate is the workhorse:

$$\text{CER} = \frac{S + D + I}{N}$$

where S, D, I are substitutions, deletions, and insertions to turn the prediction into the reference, and N is the number of reference characters. Word Error Rate is the same at word granularity. In practice:

```python
# pip install jiwer
import jiwer

reference  = "Attention Is All You Need"
prediction = "Attention ls All You Need"  # OCR read capital-I as lowercase-l

cer = jiwer.cer(reference, prediction)
wer = jiwer.wer(reference, prediction)
print(f"CER = {cer:.4f}")   # 0.0400  -> 1 wrong char in 25
print(f"WER = {wer:.4f}")   # 0.2000  -> 1 wrong word in 5
```

Two traps worth internalizing. First, **CER hides catastrophic structural errors.** A model that drops an entire table but nails the prose can post a low CER while being useless for extraction. For tables you need a structure-aware metric — TEDS (Tree-Edit-Distance-based Similarity), which compares the predicted table's HTML tree to the reference tree, so a swapped cell or a missing row actually costs you. Second, **CER on your synthetic held-out set is a vanity metric.** It tells you the model learned your renderer, not that it can read the world. The number that matters is CER on *real* held-out scans, and the gap between the two is the subject of section 6.

### The alignment problem on real scans

When you must use real documents (say, historical archives where no digital source exists), you cannot avoid the alignment problem — you can only manage it. The standard toolkit:

```python
# pip install rapidfuzz
from rapidfuzz import fuzz

def accept_alignment(ocr_span: str, source_line: str, floor: float = 92.0) -> bool:
    """Keep a (region, text) pair only if a trusted source line matches it
    closely. token_sort_ratio tolerates minor word-order and spacing noise
    while still rejecting spans the OCR mangled."""
    score = fuzz.token_sort_ratio(ocr_span, source_line)
    return score >= floor

# Example: a clean line survives, a garbled one is dropped.
assert accept_alignment("We propose a new architecture", "We propose a new architecture")
assert not accept_alignment("VVe prop0se a nevv arcnitecture", "We propose a new architecture")
```

The discipline is: **align aggressively, then drop anything below a high confidence floor.** It is better to throw away 30% of your pages than to train on mislabeled ones — a smaller clean corpus beats a larger poisoned one every time, the same lesson that governs [decontamination and leakage](/blog/machine-learning/training-data/decontamination-and-benchmark-leakage) and [deduplication at scale](/blog/machine-learning/training-data/deduplication-at-scale). The single most reliable way to sidestep the alignment problem entirely is to never introduce it: generate the document yourself.

## 3. Synthetic document generation: the dominant trick

Here is the move that reframes the whole field. **If you render a document from source markup, the source text is the ground truth — perfectly aligned, at zero annotation cost, in unlimited quantity.** There is no OCR engine in the loop to make mistakes, no human to pay, no alignment to fuzzy-match. You already have the answer key because you wrote the question.

![Synthetic document generation: render markup you already hold, and the source text is the ground truth, so every pixel is labeled for free](/imgs/blogs/ocr-chart-and-document-data-2.webp)

The figure shows the two branches that make it work. From a single source (LaTeX, HTML, or Markdown), one branch renders to a page image with a headless engine, and the other extracts the reading-order text directly from the source. The two rejoin as an (image, text) pair with zero human labels. This is exactly what Nougat did with arXiv LaTeX, what Donut's SynthDoG did with Wikipedia text on textured backgrounds, and what PubLayNet did with PubMed Central XML — all covered in the case studies.

### A renderer that also emits perfect labels

The cleanest modern approach uses a headless browser, because a browser gives you both the rendered pixels and the DOM — and the DOM hands you text *and* layout boxes for free. Here is a compact generator:

```python
# pip install playwright && playwright install chromium
from playwright.sync_api import sync_playwright
from pathlib import Path
import json

PAGE_TMPL = """
<!doctype html><html><head><meta charset="utf-8"><style>
  body {{ font-family: '{font}', serif; font-size: {size}px; margin: 64px;
         column-count: {cols}; column-gap: 48px; line-height: 1.4; }}
  h1 {{ column-span: all; font-size: {size_h}px; }}
  table {{ border-collapse: collapse; }} td, th {{ border: 1px solid #333; padding: 6px; }}
</style></head><body>{content}</body></html>
"""

def render_document(content_html: str, out_png: Path, *,
                    font="Georgia", size=18, size_h=30, cols=2) -> dict:
    html = PAGE_TMPL.format(content=content_html, font=font,
                            size=size, size_h=size_h, cols=cols)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1024, "height": 1400},
                                device_scale_factor=2)  # 2x = crisp text
        page.set_content(html, wait_until="networkidle")

        # (1) Ground-truth TEXT in DOM/reading order — exact by construction.
        gt_text = page.evaluate("() => document.body.innerText")

        # (2) Ground-truth LAYOUT: per-block type + bounding box, for free.
        gt_blocks = page.evaluate("""() => {
          const sel = 'h1,h2,h3,p,li,td,th,figcaption';
          return [...document.querySelectorAll(sel)].map(el => {
            const r = el.getBoundingClientRect();
            return {tag: el.tagName.toLowerCase(), text: el.innerText,
                    bbox: [Math.round(r.x), Math.round(r.y),
                           Math.round(r.right), Math.round(r.bottom)]};
          });
        }""")

        page.screenshot(path=str(out_png), full_page=True)
        browser.close()
    return {"text": gt_text, "blocks": gt_blocks}
```

Read what you got for free: `gt_text` is the transcript in true reading order (the browser's `innerText` respects the CSS `column-count`, so it emits the left column then the right column — not raster order). `gt_blocks` is a list of `(type, text, bbox)` triples: title, paragraph, list item, table cell, figure caption, each with pixel coordinates. That is a fully-labeled document — OCR target, layout target, and reading order — from a few lines of markup. Vary the `font`, `size`, `cols`, and content, and you have an infinite, perfectly-labeled corpus.

### A dependency-free line synthesizer

Sometimes you want OCR *line* data — image crops of single text lines — without a browser. Pillow is enough, and it lets you sweep a font bank, which (as we will see) is the single most important axis of diversity for OCR:

```python
# pip install pillow
from PIL import Image, ImageDraw, ImageFont
import random

def render_line(text: str, font_path: str, size: int = 36,
                pad: int = 12) -> Image.Image:
    font = ImageFont.truetype(font_path, size)
    # Measure, then size the canvas to the content (no wasted margins).
    l, t, r, b = font.getbbox(text)
    w, h = (r - l) + 2 * pad, (b - t) + 2 * pad
    img = Image.new("L", (w, h), color=255)          # white background
    draw = ImageDraw.Draw(img)
    ink = random.randint(0, 60)                        # jitter the ink darkness
    draw.text((pad - l, pad - t), text, fill=ink, font=font)
    return img

# One (image, label) pair per line of a corpus, per font in the bank.
fonts = ["/fonts/DejaVuSerif.ttf", "/fonts/LiberationSans.ttf",
         "/fonts/Courier.ttf", "/fonts/Georgia.ttf"]  # extend to hundreds
for line in open("corpus.txt"):
    line = line.strip()
    if not line:
        continue
    img = render_line(line, random.choice(fonts), size=random.randint(28, 44))
    # img -> save; label = line
```

This is how TrOCR and countless production OCR systems bootstrap: take a large text corpus, render each line under many fonts and sizes, and you have millions of exactly-labeled line images before lunch. The catch — always the same catch — is that these lines are *too clean*. That is section 6.

## 4. Chart and plot data: synthesize from known tables

Charts are the purest case of "the source is the label," because a chart is a deterministic rendering of a table. **Sample the numbers, render the chart, and you simultaneously own the extraction target (the table) and the answer key for any question you can ask about it.**

![Chart-to-table: one synthetic chart supervises two tasks — table extraction and figure QA — because the underlying values are known](/imgs/blogs/ocr-chart-and-document-data-4.webp)

The figure makes the leverage explicit: a single sampled table, rendered once, forks into two supervisions. The chart-to-table branch (the DePlot/MatCha style) teaches the model to invert the rendering — to read bar heights and axis ticks back into numbers. The figure-QA branch (the ChartQA style) teaches reasoning over the chart — "what is the largest value," "how much did APAC grow" — with answers computed directly from the known table. Both labels are free.

```python
# pip install matplotlib numpy
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np, random, json

def synth_bar_example(out_png: str, rng: np.random.Generator) -> dict:
    pool = ["Q1", "Q2", "Q3", "Q4", "APAC", "EMEA", "LatAm", "North", "South"]
    cats = rng.choice(pool, size=rng.integers(3, 6), replace=False).tolist()
    vals = np.round(rng.uniform(5, 100, size=len(cats)), 1)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.bar(cats, vals)
    ax.set_ylabel("Revenue (M)")
    if rng.random() < 0.5:               # diversify style: gridlines sometimes
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

    table = {"category": cats, "value": vals.tolist()}
    hi = int(np.argmax(vals))
    qa = [
        {"q": "What is the largest value?",              "a": float(vals[hi])},
        {"q": f"What is the value of {cats[0]}?",        "a": float(vals[0])},
        {"q": "How many bars are shown?",                "a": len(cats)},
        {"q": f"Is {cats[0]} greater than {cats[-1]}?",  "a": bool(vals[0] > vals[-1])},
    ]
    return {"image": out_png, "table": table, "qa": qa}

rng = np.random.default_rng(0)
ex = synth_bar_example("chart_0.png", rng)
print(json.dumps(ex["qa"], indent=2))
```

Every question's answer is computed from `vals`, so there is no annotation and no ambiguity. Scale this across bar, line, pie, stacked, and grouped charts; across styles, color maps, gridline settings, and legends; and across question templates (extremum, lookup, comparison, aggregation, trend) and you have a chart-understanding corpus with millions of exactly-labeled QA pairs.

The right metric for the chart-to-table task is not CER — a swapped value is worse than a typo. The field uses a relative-number-set-similarity score (RNSS) or an RMS-F1 that matches predicted cells to reference cells and scores numeric closeness, so reading `91.2` as `912` is penalized heavily. Score the *table*, not the string.

The one warning, which the domain-gap section formalizes: synthetic charts are seductive because they are so easy, and it is tempting to ship a model trained only on matplotlib defaults. Real charts — from Statista, from annual reports, from Excel — have dual axes, log scales, data labels, rotated ticks, broken axes, and a thousand styles matplotlib never emits by default. Diversify hard, and always keep a slice of real charts in the eval.

## 5. Layout and structure: bounding boxes, reading order, tables

Text is only half of document ground truth. The other half is *where* the text is and *in what order* it should be read. The senior rule: **reading order is a learned prior, and your data teaches it — so if your data is monotonous, the prior is wrong.**

The single most common failure I have seen in document models is reading-order collapse on multi-column pages. The traversal below is the thing the model has to internalize:

<figure class="blog-anim">
<svg viewBox="0 0 680 470" role="img" aria-label="Reading-order traversal over a two-column page: a highlight sweeps title, then the full left column top to bottom, then the full right column, then the footer" style="width:100%;height:auto;max-width:820px">
<title>Reading order on a two-column page: title, left column, right column, footer</title>
<style>
.o5-blk{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.o5-lbl{font:600 15px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.o5-path{fill:none;stroke:var(--text-secondary,#9ca3af);stroke-width:1.4;stroke-dasharray:4 5;opacity:.55}
.o5-badge{fill:var(--accent,#6366f1)}
.o5-bnum{font:700 13px ui-sans-serif,system-ui;fill:#fff;text-anchor:middle}
.o5-hl{fill:var(--accent,#6366f1);opacity:.85}
.o5-h1{animation:o5s1 12s ease-in-out infinite}
.o5-h2{animation:o5s2 12s ease-in-out infinite}
.o5-h3{animation:o5s3 12s ease-in-out infinite}
.o5-h4{animation:o5s4 12s ease-in-out infinite}
.o5-h5{animation:o5s5 12s ease-in-out infinite}
.o5-h6{animation:o5s6 12s ease-in-out infinite}
@keyframes o5s1{0%,13%{opacity:.85}17%,100%{opacity:.05}}
@keyframes o5s2{0%,14%{opacity:.05}18%,30%{opacity:.85}34%,100%{opacity:.05}}
@keyframes o5s3{0%,31%{opacity:.05}35%,47%{opacity:.85}51%,100%{opacity:.05}}
@keyframes o5s4{0%,48%{opacity:.05}52%,63%{opacity:.85}67%,100%{opacity:.05}}
@keyframes o5s5{0%,64%{opacity:.05}68%,80%{opacity:.85}84%,100%{opacity:.05}}
@keyframes o5s6{0%,81%{opacity:.05}85%,97%{opacity:.85}100%{opacity:.05}}
@media (prefers-reduced-motion:reduce){.o5-h1,.o5-h2,.o5-h3,.o5-h4,.o5-h5,.o5-h6{animation:none;opacity:.22}}
</style>
<path class="o5-path" d="M340 67 L180 168"/>
<path class="o5-path" d="M180 168 L180 280"/>
<path class="o5-path" d="M180 280 L500 168"/>
<path class="o5-path" d="M500 168 L500 280"/>
<path class="o5-path" d="M500 280 L340 375"/>
<rect class="o5-blk" x="40" y="40" width="600" height="54" rx="7"/>
<rect class="o5-blk" x="40" y="120" width="280" height="96" rx="7"/>
<rect class="o5-blk" x="360" y="120" width="280" height="96" rx="7"/>
<rect class="o5-blk" x="40" y="232" width="280" height="96" rx="7"/>
<rect class="o5-blk" x="360" y="232" width="280" height="96" rx="7"/>
<rect class="o5-blk" x="40" y="348" width="600" height="54" rx="7"/>
<rect class="o5-hl o5-h1" x="40" y="40" width="600" height="54" rx="7"/>
<rect class="o5-hl o5-h2" x="40" y="120" width="280" height="96" rx="7"/>
<rect class="o5-hl o5-h3" x="40" y="232" width="280" height="96" rx="7"/>
<rect class="o5-hl o5-h4" x="360" y="120" width="280" height="96" rx="7"/>
<rect class="o5-hl o5-h5" x="360" y="232" width="280" height="96" rx="7"/>
<rect class="o5-hl o5-h6" x="40" y="348" width="600" height="54" rx="7"/>
<text class="o5-lbl" x="340" y="72">title / header (full width)</text>
<text class="o5-lbl" x="180" y="172">left column, para 1</text>
<text class="o5-lbl" x="500" y="172">right column, para 1</text>
<text class="o5-lbl" x="180" y="284">left column, para 2</text>
<text class="o5-lbl" x="500" y="284">right column, para 2</text>
<text class="o5-lbl" x="340" y="380">footer / page number</text>
<circle class="o5-badge" cx="60" cy="60" r="12"/>
<text class="o5-bnum" x="60" y="64">1</text>
<circle class="o5-badge" cx="60" cy="140" r="12"/>
<text class="o5-bnum" x="60" y="144">2</text>
<circle class="o5-badge" cx="60" cy="252" r="12"/>
<text class="o5-bnum" x="60" y="256">3</text>
<circle class="o5-badge" cx="380" cy="140" r="12"/>
<text class="o5-bnum" x="380" y="144">4</text>
<circle class="o5-badge" cx="380" cy="252" r="12"/>
<text class="o5-bnum" x="380" y="256">5</text>
<circle class="o5-badge" cx="60" cy="368" r="12"/>
<text class="o5-bnum" x="60" y="372">6</text>
</svg>
<figcaption>Correct reading order on a two-column page is not raster left-to-right: after the title, the full left column is read top to bottom, then the full right column, then the footer. The highlight walks that path; block 3 to 4 is the column jump models get wrong.</figcaption>
</figure>

Watch the jump from block 3 (bottom of the left column) to block 4 (top of the right column). A model trained only on single-column pages has never seen that jump; it will happily run block 1, then read across the top of both columns, and interleave the two stories. The fix lives entirely in the data: your synthetic generator must emit multi-column layouts, sidebars, tables, and footnotes in roughly the proportions the real corpus contains, with the reading-order label computed from the source (the browser's `innerText` already does this for CSS columns).

### Encoding layout in the target

Two schools of thought, and you will use both:

- **Markup as target.** Emit the whole page as Markdown/HTML/LaTeX. Structure is implicit in the tags: a table becomes an HTML `<table>`, a heading becomes `#`, math becomes LaTeX. This is Nougat's approach and it is elegant because one sequence carries text, order, and structure together. The catch is that markup is verbose and the decoder can drift or repeat over long pages.
- **Coordinates as tokens.** Emit `(text, x0, y0, x1, y1, type)` tuples. This is what layout datasets like DocLayNet and PubLayNet supervise. It is explicit and easy to evaluate with detection metrics (mAP over boxes), and it composes with a separate reading-order model.

For reading order specifically, the classic algorithm is the recursive **XY-cut**: repeatedly split the page along the widest whitespace gutter (vertical first for columns, then horizontal for rows), which recovers column order on clean layouts. It fails on complex magazine-style pages, which is why learned reading-order models (LayoutReader and its successors) exist — and why the reading-order labels in your synthetic data need to be correct, because a learned model is only as good as its order supervision.

## 6. The synthetic-to-real domain gap, and how to close it

Now the hard part, the one that separates a demo from a product. **A model trained on clean renders learns clean renders. Point it at a real scanner output and it falls apart — because the real distribution has artifacts your renderer never produced.** This is the single biggest reason document models fail in deployment, and the good news is that it is almost entirely fixable in the data.

![The synthetic-to-real domain gap: clean renders lack the artifacts of real scans, and each row is a degradation you must inject to match the target distribution](/imgs/blogs/ocr-chart-and-document-data-6.webp)

The matrix names the five axes that matter, and for each one the gap between what real scans contain, what your renderer emits by default, and the augmentation that closes it. Resolution: real scans are soft and low-DPI; renders are crisp; blur and downscale. Compression: real scans carry JPEG blocking; renders are lossless; recompress at low quality. Geometry: real scans are skewed and warped; renders are axis-aligned; apply rotation and perspective. Show-through: real scans have bleed-through from the reverse side and stains; renders are on pure white; blend a faint back-page and coffee-ring textures. Typography: real documents use hundreds of fonts and handwriting; a lazy renderer uses three; expand the font bank aggressively.

Here is a degradation pipeline that injects those artifacts:

```python
# pip install pillow numpy
import io, numpy as np
from PIL import Image, ImageFilter

def degrade(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size

    # 1. Simulate low scan DPI: downscale then upscale -> soft edges.
    s = rng.uniform(0.5, 0.85)
    img = img.resize((int(w * s), int(h * s)), Image.BILINEAR) \
             .resize((w, h), Image.BILINEAR)

    # 2. Slight skew: rotate a fraction of a degree, expand to avoid clipping.
    img = img.rotate(rng.uniform(-1.5, 1.5), resample=Image.BICUBIC,
                     expand=False, fillcolor=(255, 255, 255))

    # 3. Optical blur.
    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.1)))

    # 4. Bleed-through: blend a faint, mirrored, dimmed copy of the page.
    if rng.random() < 0.4:
        ghost = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = Image.blend(img, ghost, alpha=rng.uniform(0.04, 0.10))

    # 5. Sensor noise.
    arr = np.asarray(img).astype(np.int16)
    arr += rng.normal(0, rng.uniform(2, 8), arr.shape).astype(np.int16)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # 6. JPEG recompression artifacts (do this LAST, like a real pipeline).
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(rng.integers(40, 72)))
    buf.seek(0)
    return Image.open(buf).convert("RGB")
```

Order matters: JPEG last, because in a real capture pipeline compression is the final step. Apply this to your clean renders and the input distribution moves toward the scanner's. For production, `albumentations` gives you a maintained, GPU-friendly superset (`ImageCompression`, `GaussNoise`, `OpticalDistortion`, `Downscale`, `RandomShadow`), and there are document-specific libraries (Augraphy) that model printing, photocopying, and ink bleed explicitly. The principle does not change: **measure the real distribution, then reproduce its artifacts.**

The most reliable closer of all is a small amount of *real* labeled data. Synthetic gets you 90% of the way for free; a few thousand hand-corrected real pages, mixed in or used for a final fine-tune, closes most of the remaining gap. The mix ratio is a knob worth tuning, and the framework is the same one from [data mixing and domain weighting](/blog/machine-learning/training-data/data-mixing-domain-weighting-and-curriculum): upweight the scarce, high-value real data relative to its raw count.

## 7. A worked scenario: build a synthetic OCR and chart corpus end to end

Let me make this concrete with numbers. The goal: assemble a training corpus for a document-OCR model that must read academic-style pages and business charts, and must survive real scans. Target size around one million page-level pairs plus a chart set.

![The document-data pipeline end to end: blend real and synthetic sources, align and degrade them, filter by error rate, then always evaluate on real held-out scans](/imgs/blogs/ocr-chart-and-document-data-7.webp)

The pipeline in the figure is the plan: mix real and synthetic sources, align and normalize each into an (image, target) pair, degrade and augment to match the real distribution, filter by error rate and dedup, shard, and — the non-negotiable last box — evaluate on real held-out scans, never on synthetic.

**Source 1: rendered arXiv pages (the Nougat recipe).** Take 200,000 papers with LaTeX source. At an average of 12 pages per paper, that is 2.4M candidate pages. Not all align cleanly — figure-heavy pages, `\includegraphics` with no text, and macro-heavy math defeat the source-to-page matching. Assume a 70% alignment success rate: 1.68M aligned pages. Drop pages whose text is under a length floor or whose alignment confidence is low, and dedup near-identical boilerplate (author templates, license footers): keep about 1.0M pages. That is roughly 42% yield from candidates, which is healthy for this source.

**Source 2: HTML/Markdown renders.** Convert a cleaned web corpus of articles and documentation into the browser generator from section 3. Rendering is cheap and alignment is exact (the source *is* the label), so yield is near 100%. Generate 400,000 pages across a font bank of 300 fonts and a template bank of 12 layouts (one/two/three-column, with and without tables, sidebars, footnotes).

**Source 3: synthetic charts.** Sample 300,000 tables, render one chart each across a diversified style bank, and emit `(chart, table)` plus an average of 3 QA per chart: 300,000 chart-to-table pairs and 900,000 QA pairs. Render success is essentially 100%.

**Degradation and expansion.** For the OCR page sets (Sources 1 and 2, 1.4M clean pages), produce two degraded variants each so the model sees both crisp and scanner-like inputs. That expands the OCR training set to about 2.8M image-target pairs while keeping the labels identical (degradation changes pixels, not text).

Now the economics, which is the whole point:

| Quantity | Synthetic pipeline | Human transcription |
| --- | --- | --- |
| Pages to label | 1.4M | 1.4M |
| Time per page | ~0.15 s (HTML) to ~0.35 s (LaTeX) | 1–3 min |
| Total compute/labor | ~80 CPU-hours (≈ 1.3 h on 64 cores) | ~35,000–70,000 person-hours |
| Label error rate | ~0 (exact by construction) | 1–3% per page, variable |
| Marginal cost of 2x more | Rerun the renderer | Hire twice the annotators |

The synthetic route produces 1.4M perfectly-labeled pages for a few hours of CPU. Human transcription of the same volume is tens of thousands of person-hours with a non-trivial error rate — the kind of budget that simply does not get approved. That asymmetry, not any modeling insight, is why document AI is dominated by rendered ground truth.

**The validation that keeps you honest.** Set aside 2,000 *real* scanned pages, transcribed carefully by hand (this is where you spend your human budget — on eval, not training). A model trained on the clean synthetic set will typically post something like 3% CER on synthetic held-out and 18% on the real set — a 15-point gap that is invisible if you only look at synthetic numbers. Turn on the degradation pipeline from section 6 and retrain, and the real CER commonly drops into the 6–8% range while synthetic stays near 3%. Mix in a few thousand real labeled pages for a final fine-tune and you can often reach 4–5% real CER. The exact numbers vary by domain, but the *shape* is universal: synthetic gets you most of the way, degradation closes most of the gap, a little real data closes the rest.

## 8. Case studies

### 1. Nougat: arXiv LaTeX as a labeling machine

Nougat (Neural Optical Understanding for Academic Documents, Meta AI, 2023) is the canonical demonstration of rendered ground truth. The team took arXiv papers, rendered the LaTeX source to PDF pages, and paired each page image with a lightweight markup target (Markdown-flavored text with inline and display LaTeX for math). A Donut-style architecture — a Swin Transformer visual encoder feeding an autoregressive text decoder — learned to map a page image to its markup, math and tables included, with no OCR engine in the loop.

The interesting engineering was the alignment problem in disguise. LaTeX source is one continuous stream; the rendered PDF is paginated. Matching *which* source spans landed on *which* page required a fuzzy-matching heuristic, and pages where the match was low-confidence (heavy figures, unusual macros) were dropped rather than shipped mislabeled — exactly the "align aggressively, then drop" discipline from section 2. Nougat also hit the long-sequence decoder's favorite failure, repetition: on out-of-distribution pages the decoder would loop, emitting the same line forever. They mitigated it with an anti-repetition signal and length controls. The lesson that generalizes: rendered data is free and exact, but the two hard parts are aligning source to pixels and keeping a long decoder from hallucinating on the pages your renderer handled badly.

### 2. Donut and SynthDoG: bootstrapping OCR-free understanding from pure synthesis

Donut (OCR-free Document Understanding Transformer, 2021) argued you do not need a separate OCR stage at all — read the document end-to-end into structured output. To pretrain it without any real labels, the authors built SynthDoG (Synthetic Document Generator): it composites text sampled from Wikipedia (in multiple languages) onto photographed paper and background textures, with randomized fonts, layouts, and rendering effects. The model pretrains on this synthetic stream to "read" text in reading order, then fine-tunes on small task-specific sets (receipts, forms).

SynthDoG is worth studying because it front-loads the domain-gap fix into generation: the backgrounds and effects are deliberately *not* clean, so the synthetic distribution already overlaps the real one. It is the counterexample to the lazy renderer — diversity and degradation are designed in, not bolted on. The result was that Donut could reach strong document-understanding performance while sidestepping the error propagation of a fixed OCR front-end, because its "OCR" was trained jointly with the task on data that looked like the real thing.

### 3. DePlot and MatCha: chart derendering as a pretraining task

DePlot (2022–2023) formalized plot-to-table translation: convert a chart image into a linearized data table, then hand that table to a general LLM for reasoning. The training data is the synthetic chart-to-table recipe from section 4 — sample tables, render charts, keep the table as the target — augmented with existing chart datasets. MatCha (Math + Chart pretraining) extended the idea, mixing chart derendering (image to table and image to rendering code) with math reasoning so the visual encoder learns both to read plots and to compute over them.

The strategic insight in this line of work is decomposition: rather than teach one model to look at a chart and answer a numeric question end-to-end, teach it the invertible, perfectly-supervisable sub-task (read the chart back into its table) and let a separate reasoner handle the arithmetic. Because the sub-task has exact free labels, it trains cleanly and transfers. It is the same "make the supervisable part synthetic" move that recurs throughout document AI.

### 4. ChartQA: real charts with known tables as the answer key

ChartQA (2022) is the reality check for the chart line of work. It is a benchmark of question-answer pairs over *real* charts scraped from sources like Statista, each accompanied by its underlying data table. Questions come in two flavors: machine-generated (templated from the table, like our synthetic QA) and human-authored (which demand more compositional and visual reasoning). The presence of the ground-truth table is what makes automatic scoring possible — the answer key is computed, not guessed.

ChartQA matters as a case study because it exposes the synthetic-to-real gap for charts specifically. Models trained on tidy matplotlib charts do well on the templated questions and stumble on the human ones and on the messy real styles — dual axes, data labels, unusual color schemes. It is the concrete evidence for section 4's warning: synthesize aggressively for volume, but hold out real charts, and expect the human-authored questions to be the hard slice.

### 5. PubLayNet and PubTabNet: structured source as web-scale free labels

IBM's PubLayNet and PubTabNet took the "source is the label" idea to a different structured input: the XML of PubMed Central open-access articles. Because the XML marks up titles, abstracts, paragraphs, figures, tables, and lists, and the corresponding PDFs render those elements, the two can be aligned automatically to produce layout labels (PubLayNet: on the order of 360,000 pages with bounding boxes and element types) and table-structure labels (PubTabNet: on the order of 500,000 table images paired with their HTML structure) — with no human annotation.

This is the trick at its most powerful: you do not even need to *render* the document if a structured source and a rendered artifact already both exist and can be aligned. Any corpus that ships as (structured markup, rendered document) pairs — journal archives, government filings in XBRL, e-books in EPUB — is a latent, web-scale, free labeling machine for layout and table structure. The engineering is all in the alignment, and the payoff is datasets that would cost millions to annotate by hand.

### 6. DeepSeek-OCR: when the image is cheaper than its text

DeepSeek-OCR (2024–2025) is a fascinating inversion. Instead of treating OCR as a data-labeling problem, it treats a page of *rendered text as a compression medium*: a vision encoder can carry the information of a long text passage in roughly a tenth of the tokens the raw text would need, and a decoder reconstructs the text with high fidelity. Training this requires exactly the data this post is about — enormous volumes of (rendered page, exact text) pairs, which are available precisely because the render-your-own-ground-truth trick makes them cheap and exact. The full architecture and the "optical memory" thesis are covered in the dedicated post on [DeepSeek-OCR's optical context compression](/blog/machine-learning/computer-vision/deepseek-ocr-optical-context-compression).

The reason to include it here is what it says about the value of this data pipeline. If optical compression pans out, then the ability to render text to images with perfectly-aligned labels is not just an OCR convenience — it becomes infrastructure for long-context language models, which could store part of their context as compressed vision tokens. The humble render-and-pair pipeline turns out to be load-bearing well beyond document reading.

### 7. TrOCR: synthetic lines at the foundation

TrOCR (2021) is the quiet, ubiquitous case: a transformer encoder-decoder for text-line recognition, pretrained on hundreds of millions of *synthetically rendered* text lines (printed and, in the handwritten variant, styled to mimic handwriting) before fine-tuning on real line datasets. It is the line-level version of everything above — the `render_line` function from section 3 at industrial scale — and it anchors why the font bank and degradation matter: TrOCR's printed and handwritten variants differ almost entirely in the *rendering distribution* of their synthetic pretraining data, not the architecture. The data recipe is the product.

## Troubleshooting: symptom, cause, fix

Document-data pipelines fail in a small number of recognizable ways. Here is the field guide.

| Symptom | Likely cause | Detection | Fix |
| --- | --- | --- | --- |
| Great on synthetic eval, terrible on real scans | Domain gap: clean fonts, no artifacts | Hold out real scans; compare CER synthetic vs real | Degradation augmentation; mix in real labeled data |
| Multi-column text interleaves / scrambles | Reading-order prior wrong; templates too monotonous | Sequence-alignment on multi-column held-out | Synthesize multi-column, tables, sidebars with correct order labels |
| Loss plateaus; model learns odd tokens | Ground-truth alignment errors (label ≠ pixels) | Inspect high-loss pairs; overlay label on image | Fuzzy-match + drop low-confidence pairs; prefer synthetic |
| Model overfits a few fonts; fails on new ones | Narrow rendered font/template set | Per-font CER variance on held-out fonts | Expand font bank to hundreds; randomize layouts; add handwriting |
| Decoder repeats / hallucinates on figures | Source had non-text content; long-decoder looping | Repeated-n-gram check; output-vs-GT length ratio | Filter figure-only pages; anti-repetition penalty; train correct empty targets |
| Chart values read at wrong scale | Charts too uniform; no axis/style diversity | Evaluate on real charts (ChartQA); RNSS by style | Diversify axes, scales, gridlines, labels; add real charts |

### The domain gap (the big one)

**Symptom.** Your model posts 3% CER on the synthetic validation set and you ship it, and it returns garbage on the first real scanned invoice. **Cause.** The model learned your renderer's exact fonts, spacing, and pure-white backgrounds; a real scan is soft, skewed, compressed, and full of show-through. **Detection.** This is only visible if you evaluate on real, hand-labeled scans — the single most important test set you will build. A large synthetic-to-real CER gap (say, more than 5–8 points) is the signature. **Fix.** Turn on the degradation pipeline from section 6 so the training input distribution overlaps the real one, and, if you can afford it, fine-tune on a few thousand real labeled pages. Re-measure the gap; iterate on whichever artifact axis the failures cluster on (often resolution and compression first).

### Reading-order collapse

**Symptom.** On single-column pages the transcript is perfect; on two-column academic pages the paragraphs interleave and the output is unreadable even though every *word* is correct. **Cause.** Your synthetic generator only ever produced single-column layouts, so the model never learned the column jump the animated figure highlights. **Detection.** Build a held-out set stratified by layout (one/two/three column, with/without tables) and score reading order separately from character accuracy — a sequence-alignment score against the correct order exposes it. **Fix.** Diversify layouts in generation and make sure the reading-order label is correct (the browser's `innerText` handles CSS columns; for coordinate targets, order with XY-cut or a learned reader). Match the layout mix to the real corpus.

### Silent alignment errors

**Symptom.** Training loss stops improving early and never reaches the level a clean run should; spot-checking predictions shows the model confidently emitting text that is subtly wrong in a consistent way. **Cause.** Some fraction of your (image, text) pairs are mislabeled — usually because you bootstrapped labels from an existing OCR engine, or your source-to-page alignment was loose. The model is faithfully learning the mislabels. **Detection.** Sort pairs by training loss and inspect the top few hundred; overlay the label on the image and read them side by side. If the labels themselves are wrong, that is your answer. **Fix.** Raise the alignment confidence floor and drop everything below it; where possible, replace bootstrapped labels with rendered ones (exact by construction). A smaller clean set trains better than a larger dirty one.

### Font and template overfitting

**Symptom.** CER is excellent on the fonts in your training bank and collapses on any unseen font or a slightly unusual layout. **Cause.** The rendered corpus used a handful of fonts and one or two templates, so the model memorized glyph shapes rather than learning to read. **Detection.** Hold out entire fonts and layouts from training and measure CER on them specifically; high per-font variance is the tell. **Fix.** Expand the font bank to hundreds of faces (serif, sans, monospace, condensed, and handwriting), randomize sizes and weights, and vary layout templates and margins. Diversity in the *rendering distribution* is what forces the model to learn reading instead of recall.

## When to reach for rendered document data — and when not to

**Reach for synthetic rendering when:**

- You control or can obtain the source markup (LaTeX, HTML, XML, Markdown) — the label is then exact and free, and you should almost always prefer it to human transcription.
- You need scale (hundreds of thousands to millions of pages) and layout/structure/reading-order labels, which are prohibitively expensive to annotate by hand.
- The target task is a deterministic rendering you can invert: charts from tables, forms from schemas, code from ASTs.
- You need to control the difficulty curriculum — start clean, add artifacts progressively — which rendering lets you dial exactly.

**Be careful, or add real data, when:**

- The deployment distribution has artifacts you cannot fully model (exotic scanners, phone photos at angles, historical degradation). Synthesize what you can, but budget for real labeled data and, critically, a real eval set.
- The documents are genuinely un-sourceable (handwritten archives, damaged originals) — here you cannot avoid real-scan pairs and the alignment problem; manage it with high-confidence fuzzy matching and aggressive dropping.
- The task needs reasoning the templated labels cannot express (the human-authored slice of ChartQA) — synthetic QA covers lookups and extrema, not open-ended visual reasoning; supplement with human or model-generated questions, as in [recaptioning and VLM instruction data](/blog/machine-learning/training-data/recaptioning-and-vlm-instruction-data).
- You are tempted to evaluate on synthetic data because the numbers look good. Do not. The only honest score is on real held-out documents, measured with a structure-aware metric where structure matters.

The through-line of this entire subject is one idea worth stating plainly: **in document AI, the label is manufactured, not collected.** You render the answer, pair it with the pixels, degrade it until it looks real, filter out what does not align, and measure yourself against the real world. Do those four things well and the modeling is almost the easy part. For the token-space version of the same manufacturing mindset, see [synthetic data generation](/blog/machine-learning/training-data/synthetic-data-generation); for how you decide the number is trustworthy at all, [measuring data quality](/blog/machine-learning/training-data/measuring-data-quality).

## Further reading

- Nougat: Neural Optical Understanding for Academic Documents (Blecher et al., 2023) — rendered LaTeX as a labeling machine, and the alignment/repetition failure modes.
- Donut and SynthDoG (Kim et al., 2021) — OCR-free understanding bootstrapped from a synthetic document generator with designed-in domain diversity.
- DePlot and MatCha (Liu et al., 2022–2023) — chart derendering as a perfectly-supervisable pretraining task.
- ChartQA (Masry et al., 2022) — real charts with known tables; the machine-vs-human question gap.
- PubLayNet and PubTabNet (Zhong et al., IBM) — structured XML aligned to rendered PDFs for web-scale free layout and table labels.
- Sibling posts on this blog: [DeepSeek-OCR](/blog/machine-learning/computer-vision/deepseek-ocr-optical-context-compression), [DeepSeek-VL2 dynamic tiling](/blog/machine-learning/computer-vision/deepseek-vl-vl2-dynamic-tiling-moe), [synthetic data generation](/blog/machine-learning/training-data/synthetic-data-generation), and [recaptioning and VLM instruction data](/blog/machine-learning/training-data/recaptioning-and-vlm-instruction-data).
