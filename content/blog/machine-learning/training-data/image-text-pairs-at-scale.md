---
title: "Image-Text Pairs at Scale: Turning the Web's Alt-Text Into Training Data"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "How CLIP, Stable Diffusion, and every open vision-language model were fed: harvesting alt-text from Common Crawl, the LAION pipeline, CLIP-score filtering, the alt-text noise problem, and the dedup and safety work the cosine threshold does not do — with runnable open_clip and img2dataset code, a worked scoring scenario, and a troubleshooting guide."
tags:
  - training-data
  - image-text-pairs
  - clip
  - laion
  - vision-language
  - data-filtering
  - common-crawl
  - img2dataset
  - webdataset
  - multimodal
category: "machine-learning"
subcategory: "Training Data"
author: "Hiep Tran"
featured: true
readTime: 30
---

Every open vision-language model you have used was trained on captions nobody wrote for it. CLIP, OpenCLIP, Stable Diffusion, the SigLIP encoders, LLaVA's projector — all of them were fed hundreds of millions to billions of (image, text) pairs scraped from the public web, where the "text" is the `alt` attribute a web developer typed so a screen reader could describe a picture. Nobody sat down to annotate two billion images. The web already did it, badly and for free, over three decades. The entire discipline of large-scale multimodal training is, at bottom, the craft of turning that accidental, noisy, half-abandoned caption layer into something a model can learn a shared image-text space from.

That craft has one central problem and one central trick. The problem: most alt-text is garbage — filenames, the literal word "image", SEO keyword salad, empty strings, "click to enlarge". The trick: use a model that already understands image-text alignment (CLIP) to score how well each caption matches its image, and throw away everything below a cosine-similarity threshold. That one idea — self-filtering the training set with a smaller version of the thing you are trying to build — is what made web-scale image-text data usable, and it is what this post is about.

![How a web crawler splits an <img> tag into its src URL and alt attribute and emits one image-text training pair](/imgs/blogs/image-text-pairs-at-scale-1.webp)

The diagram above is the mental model for the whole pipeline: a crawler reads an `<img>` element, pulls out the `src` URL and the `alt` attribute, and emits exactly one (image, caption) pair. Do that across every page in a Common Crawl snapshot and you have billions of raw pairs — most of them junk. The rest of this post is the machinery that separates the aligned minority from the noise: how pairs are harvested, how LAION was built, how CLIP-score filtering works and where it is blind, the scale-versus-quality dial you are turning when you pick a threshold, and the deduplication and safety work that the cosine score does not do for you. This is Wave 4 of the training-data series — the first post on vision and image-generation data — and it leans on the text-side machinery we have already built: [deduplication at scale](/blog/machine-learning/training-data/deduplication-at-scale) and [PII, safety, and toxicity filtering](/blog/machine-learning/training-data/pii-safety-and-toxicity-filtering) both reappear here, pointed at pixels instead of tokens.

## Why alt-text is the web's free caption layer

Before the mechanics, get the framing right, because the naive intuition is wrong in a way that costs you weeks if you believe it.

| Assumption | Naive view | Reality |
| --- | --- | --- |
| "Alt-text describes the image" | Every `<img>` has a human-written caption | 40-70% of images have empty or missing `alt`; of those with `alt`, a large fraction is a filename, "image", or boilerplate |
| "More pairs is better" | Download all 5.8B tags, train on everything | Training on unfiltered pairs produces a *worse* model than training on a filtered 10% of them |
| "The caption is roughly the ground truth" | Trust the text | The text is a weak, noisy label; alignment must be *measured*, not assumed |
| "We can hand-clean it" | Write regex rules for junk | Regex catches filenames; it cannot catch a real English sentence that describes the *wrong* image |
| "One good crawl is enough" | Scrape once, filter once | The same images recur across snapshots and domains; you will dedup more images than you keep |

The load-bearing insight is the second row. In 2021 the original CLIP paper and every LAION replication after it found the same thing: a model trained on a cleanly filtered subset beats a model trained on the full unfiltered set at the *same compute*, and often at the same token count. Noise is not neutral. A pair whose caption does not match its image is an actively wrong gradient — it teaches the model that "sunset" and "cheap nike shoes" belong near each other in embedding space. The web gives you scale for free; it charges you in alignment, and the whole game is buying alignment back cheaply.

## 1. How image-text pairs are harvested

**Senior rule of thumb: you are not scraping images, you are scraping HTML and keeping the two attributes of every `<img>` tag that matter.** The image bytes come later, and separately.

The raw material is [Common Crawl](https://commoncrawl.org/) — a nonprofit that has been crawling the web monthly since 2011 and publishes each snapshot as WARC (Web ARChive) files: the raw HTTP responses, HTML and all, in the tens-of-terabytes-per-snapshot range. You do not download images from Common Crawl. You download HTML, parse every `<img>` element, and extract two strings: the `src` (which you resolve against the page's base URL to get an absolute image URL) and the `alt` (the caption candidate). That is the harvest. It is cheap, embarrassingly parallel, and produces a giant table of (image URL, caption) rows — no image bytes yet, just pointers and text.

Here is the core of it, using `warcio` to iterate WARC records and `selectolax` (a fast HTML parser) to pull tags:

```python
from warcio.archiveiterator import ArchiveIterator
from selectolax.parser import HTMLParser
from urllib.parse import urljoin

def harvest_pairs(warc_path):
    """Yield (absolute_image_url, alt_text) from one Common Crawl WARC file."""
    with open(warc_path, "rb") as f:
        for rec in ArchiveIterator(f):
            if rec.rec_type != "response":
                continue
            page_url = rec.rec_headers.get_header("WARC-Target-URI")
            html = rec.content_stream().read()
            tree = HTMLParser(html)
            for img in tree.css("img"):
                src = img.attributes.get("src")
                alt = (img.attributes.get("alt") or "").strip()
                if not src or not alt:
                    continue                      # no URL or no caption -> skip
                yield urljoin(page_url, src), alt  # resolve relative src
```

A few things are already happening that matter downstream. First, we skip any image with an empty or missing `alt` — that alone discards a huge fraction of the web's images, because most `<img>` tags carry no caption at all. Second, we resolve relative URLs against the page URL, because `src="dog.jpg"` is meaningless without the page it came from. Third, we have not touched a single image pixel; this stage runs on cheap CPU boxes and produces a metadata table of maybe 5-6 billion rows from one full crawl. LAION-5B started from roughly 5.85 billion parsed image tags before any filtering.

What you do *not* do at this stage is trust the text. The `alt` string might be `"A golden retriever catching a frisbee"` or it might be `"DSC_0192.jpg"` or `"image"` or `"buy cheap watches online free shipping"`. You keep all of it and let the CLIP filter downstream decide. The harvest is deliberately permissive; the intelligence lives in the filter.

## 2. The LAION construction pipeline, end to end

**Senior rule of thumb: the pipeline is extract, download, embed, threshold — and 90%+ of the pairs die at the threshold.** Everything before the CLIP filter is plumbing; the filter is the product.

![The LAION pipeline: Common Crawl to parsed tags to downloaded images to CLIP embedding to a cosine-threshold filter to webdataset shards](/imgs/blogs/image-text-pairs-at-scale-2.webp)

[LAION](https://laion.ai/) (Large-scale Artificial Intelligence Open Network) is the community project that made this pipeline public and reproducible. LAION-400M (2021) and then LAION-5B (2022) were the first openly available image-text datasets at a scale that could train a CLIP-quality model, and they are the reason Stable Diffusion and OpenCLIP exist outside of a few large labs. The pipeline has five stages, shown above:

1. **Common Crawl** — start from a snapshot's WARC files, tens of TB of raw HTML.
2. **Parse HTML** — extract (image URL, alt-text) pairs, as in the harvest code above. Roughly 5.8B raw pairs for LAION-5B.
3. **Download images** — actually fetch the image bytes for each URL, decode them, resize, and store them alongside their caption. This is the expensive, failure-prone stage (dead links, timeouts, rate limits, malformed files).
4. **CLIP embed + threshold** — run each downloaded image and its caption through a CLIP model, compute cosine similarity, and keep only pairs scoring above a threshold (0.28 for LAION's ViT-B/32 English subset).
5. **Ship as webdataset shards** — write survivors as sharded `.tar` files for streaming training.

Stage 3 is where teams underestimate the work. Downloading two billion images from two billion different servers is a distributed-systems problem, not an ML problem. The community tool for it is [`img2dataset`](https://github.com/rom1504/img2dataset), which handles concurrency, retries, DNS caching, resizing, and writing directly to the `webdataset` format. A realistic invocation:

```bash
img2dataset \
  --url_list        laion_metadata/          --input_format parquet \
  --url_col         URL   --caption_col      TEXT \
  --output_format   webdataset               --output_folder laion_shards \
  --processes_count 16    --thread_count     64 \
  --image_size      384   --resize_mode      keep_ratio \
  --encode_quality  90    --skip_reencode    True \
  --retries         2     --timeout          10 \
  --enable_wandb    False
```

The knobs that matter: `--thread_count` and `--processes_count` set your fan-out (you are I/O bound on remote servers, so threads > cores); `--image_size 384 --resize_mode keep_ratio` resizes on ingest so you are not storing 4K originals; `--timeout 10 --retries 2` bounds how long you wait on a slow server before giving up. On a well-provisioned box `img2dataset` sustains thousands of images per second, but expect a 20-40% download-failure rate from dead links and blocked crawlers — that attrition is normal and happens *before* the CLIP filter even runs.

The output is `webdataset` shards — sequential `.tar` files, each holding thousands of `(NNNN.jpg, NNNN.txt, NNNN.json)` triples. This format streams sequentially from disk or cloud storage without random seeks, which is exactly what you want for feeding a data loader at GPU speed. Reading it back for training:

```python
import webdataset as wds

def make_loader(shard_pattern, preprocess, batch_size=256):
    ds = (
        wds.WebDataset(shard_pattern, shardshuffle=True, nodesplitter=wds.split_by_node)
        .shuffle(2000)                       # shuffle within a buffer of samples
        .decode("pilrgb")                    # decode .jpg to a PIL RGB image
        .to_tuple("jpg", "txt")              # (image, caption) per sample
        .map_tuple(preprocess, lambda s: s)  # apply CLIP preprocessing to the image
    )
    return wds.WebLoader(ds, batch_size=batch_size, num_workers=8)

loader = make_loader("laion_shards/{00000..01242}.tar", preprocess)
```

### Second-order optimization: filter before you download when you can

The stage ordering above downloads *then* filters, which means you pay to download billions of images you will throw away. LAION mitigated this by releasing the *metadata with precomputed CLIP scores* — so a downstream user can filter the parquet table on the score column and only `img2dataset` the survivors. If you are building your own pipeline, compute a cheap text-only pre-filter first (drop empty, drop pure-filename, drop captions under three words) to kill the obvious garbage before you spend bandwidth on it. You cannot compute the real CLIP score without the image, but you can eliminate a third of the rows for free.

## 3. CLIP-score filtering: the one number that made web-scale usable

**Senior rule of thumb: the CLIP score is a single cosine similarity between two 512-dimensional vectors, and the entire quality of your dataset rides on where you put the threshold.**

![How a CLIP score is computed: image and text pass through separate encoders to L2-normalized embeddings whose dot product is the cosine similarity the filter thresholds](/imgs/blogs/image-text-pairs-at-scale-3.webp)

CLIP (Contrastive Language-Image Pre-training) is two encoders trained jointly so that matching image-text pairs land close together in a shared embedding space and mismatched pairs land far apart. The image side is a vision transformer (ViT-B/32 in the LAION filter); the text side is a Transformer. Each produces a vector — 512-dimensional for ViT-B/32 — which is L2-normalized to unit length. Because both vectors are unit length, their dot product *is* their cosine similarity:

$$\text{CLIP score} \;=\; \cos(u, v) \;=\; \frac{u \cdot v}{\lVert u \rVert \, \lVert v \rVert} \;=\; u \cdot v \quad (\text{both } \lVert u \rVert = \lVert v \rVert = 1)$$

where ${u}$ is the normalized image embedding and ${v}$ is the normalized text embedding. The score lives in the range from minus one to one, but in practice web pairs cluster between roughly zero and 0.45. A perfectly matched, descriptive caption on a clear image scores in the low-to-mid 0.30s; random noise scores near zero; a wrong-but-related caption scores somewhere in between. (If you want the deeper story of *why* these encoders learn an aligned space — contrastive loss, the temperature, and how SigLIP and DINO differ — see [ViT, SigLIP, and DINO explained](/blog/machine-learning/computer-vision/vit-siglip-dino-explained).)

Computing scores for a batch is a dozen lines with `open_clip`:

```python
import torch, open_clip
from PIL import Image

# ViT-B/32 trained on LAION-2B — the standard filter model.
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model.eval()

@torch.no_grad()
def clip_scores(images, captions, device="cuda"):
    """Row-wise cosine similarity of each (image, caption) pair."""
    model.to(device)
    imgs = torch.stack([preprocess(im) for im in images]).to(device)
    toks = tokenizer(captions).to(device)
    with torch.autocast(device_type=device.split(":")[0]):
        img_emb = model.encode_image(imgs)
        txt_emb = model.encode_text(toks)
    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
    txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
    # matched pairs -> element-wise product summed over the embedding dim
    return (img_emb * txt_emb).sum(dim=-1).float().cpu()
```

Note the subtlety in the last line. You want the cosine of *each image with its own caption*, not the full image-text similarity matrix. The full matrix (`img_emb @ txt_emb.T`) is what CLIP's contrastive training uses and what you would compute for zero-shot classification; for filtering you only need the diagonal, so a row-wise `(img_emb * txt_emb).sum(-1)` is both correct and far cheaper than materializing an N-by-N matrix.

Applying the threshold is trivial once you have scores:

```python
scores  = clip_scores(images, captions)     # tensor of cosine similarities
keep    = scores >= 0.28                     # LAION's ViT-B/32 English threshold
kept    = [(u, c) for u, c, k in zip(urls, captions, keep.tolist()) if k]
print(f"keep-rate at 0.28: {keep.float().mean():.1%}  ({keep.sum()}/{len(keep)})")
```

That is the whole filter. The intelligence is entirely in the threshold, and the threshold is a policy decision with real consequences — which is the next two sections.

## 4. The score distribution and where to cut

**Senior rule of thumb: plot the score histogram on a real sample before you pick a threshold; the right cut depends on the distribution's shape, not on a number you read in a paper.**

The animation below is the whole tension in one picture. The bars are the distribution of CLIP cosine scores across a sample of raw web pairs — a broad hump peaking somewhere in the high-0.20s to low-0.30s, with a long low tail of mismatched junk. The red line is your threshold. Sweep it right and you drop more pairs; the survivors on the right are more aligned, but there are far fewer of them.

<figure class="blog-anim">
<svg viewBox="0 0 720 320" role="img" aria-label="A histogram of CLIP cosine scores for web image-text pairs; a vertical threshold line sweeps from 0.20 to 0.35 and the shaded drop region to its left grows as the threshold rises" style="width:100%;height:auto;max-width:820px">
<style>
.itp-bar{fill:var(--accent,#6366f1);opacity:.85}
.itp-axis{stroke:var(--text-secondary,#6b7280);stroke-width:2}
.itp-tick{font:500 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.itp-lbl{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.itp-drop{fill:#ef4444;opacity:.16}
.itp-cut{fill:#ef4444;opacity:.9}
.itp-cutlbl{font:700 14px ui-sans-serif,system-ui;fill:#ef4444;text-anchor:middle}
@keyframes itp-grow{0%{width:224px}50%{width:392px}100%{width:224px}}
@keyframes itp-move{0%{x:304px}50%{x:472px}100%{x:304px}}
@keyframes itp-movelbl{0%{transform:translateX(0)}50%{transform:translateX(168px)}100%{transform:translateX(0)}}
.itp-dropA{animation:itp-grow 9s ease-in-out infinite}
.itp-cutA{animation:itp-move 9s ease-in-out infinite}
.itp-cutlblA{animation:itp-movelbl 9s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.itp-dropA{animation:none;width:314px}.itp-cutA{animation:none;x:394px}.itp-cutlblA{animation:none;transform:translateX(90px)}}
</style>
<text class="itp-lbl" x="360" y="26">CLIP cosine score distribution of web pairs</text>
<rect class="itp-bar" x="90"  y="245" width="40" height="15"/>
<rect class="itp-bar" x="140" y="230" width="40" height="30"/>
<rect class="itp-bar" x="190" y="205" width="40" height="55"/>
<rect class="itp-bar" x="240" y="175" width="40" height="85"/>
<rect class="itp-bar" x="290" y="140" width="40" height="120"/>
<rect class="itp-bar" x="340" y="110" width="40" height="150"/>
<rect class="itp-bar" x="390" y="95"  width="40" height="165"/>
<rect class="itp-bar" x="440" y="110" width="40" height="150"/>
<rect class="itp-bar" x="490" y="145" width="40" height="115"/>
<rect class="itp-bar" x="540" y="190" width="40" height="70"/>
<rect class="itp-bar" x="590" y="225" width="40" height="35"/>
<rect class="itp-drop itp-dropA" x="80" y="60" height="200"/>
<rect class="itp-cut itp-cutA" x="304" y="55" width="3" height="205"/>
<line class="itp-axis" x1="80" y1="260" x2="650" y2="260"/>
<text class="itp-cutlbl itp-cutlblA" x="304" y="48">threshold</text>
<text class="itp-tick" x="90"  y="282">0.0</text>
<text class="itp-tick" x="230" y="282">0.15</text>
<text class="itp-tick" x="370" y="282">0.28</text>
<text class="itp-tick" x="510" y="282">0.40</text>
<text class="itp-tick" x="630" y="282">0.5</text>
<text class="itp-lbl" x="200" y="306" style="fill:#ef4444">drop (misaligned)</text>
<text class="itp-lbl" x="540" y="306" style="fill:var(--accent,#6366f1)">keep (aligned)</text>
</svg>
<figcaption>Sweeping the CLIP threshold from 0.20 to 0.35: the red drop region eats further into the distribution, so keep-rate falls sharply while the surviving pairs get more aligned. LAION-400M cut near 0.28.</figcaption>
</figure>

The distribution's shape is what makes the threshold a hard call. It is not bimodal — there is no clean valley between "junk" and "good" where you can drop a threshold and cleanly separate the two. It is a single broad hump, and your threshold slices *through* the middle of it. That means every threshold choice trades false positives (junk you kept) against false negatives (good pairs you dropped), and there is no cut that gives you both. LAION-400M and the English subset of LAION-5B settled on 0.28 for ViT-B/32; the multilingual and aesthetic subsets used different cuts. The right number for *your* data depends on your CLIP model (a different backbone shifts the whole distribution), your language mix, and how much you value scale over purity.

One trap worth flagging early: the threshold is model-specific. A ViT-L/14's scores are not comparable to a ViT-B/32's — the distributions sit at different centers. If you switch filter models you must re-plot and re-pick the threshold; carrying over 0.28 blindly will silently keep or drop the wrong fraction.

## 5. The alt-text noise problem

**Senior rule of thumb: assume the caption is not a caption until proven otherwise.** The CLIP filter is not a nice-to-have polish step; it exists because the raw text is dominated by things that are not descriptions at all.

![A taxonomy of scraped alt-text: a small aligned minority worth keeping versus filenames, generic tokens, SEO stuffing, and empty strings to drop](/imgs/blogs/image-text-pairs-at-scale-4.webp)

When you actually look at a sample of harvested `alt` strings, they fall into a handful of classes, shown in the taxonomy above:

- **Aligned captions (keep).** Real descriptions: "A golden retriever catching a red frisbee on green grass." These are the minority — often 10-30% of non-empty alts — and they are the entire point of the exercise. Good ones name specific nouns and attributes.
- **Filenames.** `DSC_0192.jpg`, `IMG_20180714_153042.png`, `header-logo-final-v2.png`. The developer never set `alt`, so a CMS filled it with the filename. Pure noise; scores near zero.
- **Generic tokens.** `image`, `photo`, `picture`, `thumbnail`, `avatar`. Technically a word about images, contentless as a caption. These are the *dangerous* class because they can score deceptively high (more on that below).
- **SEO keyword stuffing.** `cheap nike shoes buy online free shipping discount sale` slapped onto an unrelated image to game search rankings. Real English, entirely mismatched.
- **Boilerplate.** `click to enlarge`, `read more`, `advertisement`, cookie-banner text that leaked into an `alt`. UI chrome, not description.
- **Empty / missing.** The largest class of all — most web images have no `alt` at all. Discarded at harvest time before scoring.

Regex handles the easy classes. A filename matches `\.(jpg|png|gif|webp)$`; a generic token is a one-word alt from a small stoplist; boilerplate matches a phrase list. But regex is helpless against the hard case: a grammatically perfect English sentence that describes the *wrong* image. `"A beautiful sunset over the ocean"` on a photo of a car engine is indistinguishable from a good caption by any surface rule. Only a model that looks at *both* the image and the text can catch it — which is exactly why CLIP-score filtering, not rule-based cleaning, is the backbone of the pipeline. The rules are a cheap pre-filter; CLIP is the real judge.

## Worked scenario: scoring a Common Crawl shard

Let me make this concrete with numbers. Suppose you harvest one WARC shard and, after dropping empty and pure-filename alts with the cheap pre-filter, you are left with 10,000 (image, caption) pairs. You download the images (some fail; assume these 10,000 all succeeded), run `clip_scores` over them in batches, and bucket the results:

| Cosine bucket | Pairs | Share of sample |
| --- | --- | --- |
| 0.00 – 0.10 | 1,900 | 19% |
| 0.10 – 0.15 | 1,500 | 15% |
| 0.15 – 0.20 | 1,700 | 17% |
| 0.20 – 0.25 | 1,600 | 16% |
| 0.25 – 0.28 | 900 | 9% |
| 0.28 – 0.30 | 700 | 7% |
| 0.30 – 0.35 | 900 | 9% |
| 0.35 – 0.40 | 500 | 5% |
| 0.40+ | 300 | 3% |

The mode sits around 0.25-0.30 and the left tail is fat — nearly 20% of pairs score below 0.10, which is essentially "the caption and image have nothing to do with each other." Now sweep the threshold and watch the keep-rate collapse:

| Threshold | Pairs kept | Keep-rate | What you are buying |
| --- | --- | --- | --- |
| 0.20 | 4,900 | 49% | Scale; still a lot of loosely-related junk |
| 0.25 | 3,300 | 33% | Balanced; typical for "keep it big" runs |
| **0.28** | **2,400** | **24%** | LAION's English cut; the usual default |
| 0.30 | 1,700 | 17% | Cleaner; noticeable alignment gain |
| 0.35 | 800 | 8% | High-precision; small, tightly matched |

At the 0.28 default you keep 24% of the pre-filtered pairs — and remember you already discarded most of the raw web at the harvest and pre-filter stages, so 2,400 survivors out of a shard that started with maybe 60,000 raw `<img>` tags is a ~4% end-to-end yield. That ratio is why LAION needed 5.8 billion raw tags to produce a few hundred million to a couple billion usable pairs.

Now look at *what* the 0.28 cut removes and keeps. Three garbage pairs it correctly kills:

1. **Nav logo.** Image: a site's header logo. Caption: `"Home"`. Score: **0.09**. The word has nothing to do with the pixels. Dropped.
2. **CMS filename leak.** Image: a product photo. Caption: `"IMG_20180714_153042.jpg"` (survived the filename regex because of the timestamp format). Score: **0.11**. Dropped.
3. **SEO spam.** Image: a stock sunset. Caption: `"buy cheap watches online free shipping discount rolex sale"`. Score: **0.14**. Real words, wrong image. Dropped — and note no regex would have caught this one.

Three good pairs it correctly keeps:

1. **Descriptive.** Image: a dog with a frisbee. Caption: `"A golden retriever catching a red frisbee on green grass"`. Score: **0.33**. Kept.
2. **Food photo.** Image: a plated fish. Caption: `"Grilled salmon fillet with lemon and asparagus on a white plate"`. Score: **0.31**. Kept.
3. **Specific object.** Image: a vintage car. Caption: `"1972 Volkswagen Beetle in orange parked on a cobblestone street"`. Score: **0.34**. Kept.

And one pair that survives but *should not* — the CLIP blind spot we return to in troubleshooting:

- **Generic caption.** Image: the same golden retriever. Caption: `"a photo"`. Score: **0.29**. Survives the 0.28 cut, contributes nothing, and quietly biases the model toward contentless captions. The threshold cannot tell "correct but useless" from "correct and rich."

That last row is the whole reason the next generation of filters (aesthetic scoring, caption-quality models, [data filtering networks](/blog/machine-learning/training-data/datacomp-and-data-filtering-networks)) exists: a single cosine threshold is necessary but not sufficient.

## 6. The scale-vs-quality dial

**Senior rule of thumb: the threshold is not a quality knob, it is a scale-quality *exchange rate*, and the right setting depends on what you are training and how much compute you have.**

![The scale-versus-quality dial: a low threshold keeps billions of noisy pairs while a high threshold keeps a smaller, tightly-matched set](/imgs/blogs/image-text-pairs-at-scale-5.webp)

The figure above shows the two ends of the dial. Turn the threshold down toward 0.10 and you keep almost everything — billions of pairs, cheap scale, but a large fraction misaligned, so a lot of your gradient signal is noise. Turn it up toward 0.35 and you keep a small, tightly-matched core — cleaner signal per pair, but far less data, and you may starve a large model. Neither end is "correct"; the optimum depends on three things:

- **What you are training.** A contrastive model like CLIP is relatively robust to noise (the contrastive loss averages over huge batches), so it tolerates a lower threshold and thrives on scale. A generative model like a diffusion image generator is more sensitive to caption *quality* because the caption directly conditions generation — garbage captions produce garbage prompt-following. Diffusion pipelines often filter harder, or re-caption entirely.
- **Your compute budget.** If you are compute-bound (you will only see each pair a fraction of a time), scale wins and you keep more. If you are data-bound (small target model, many epochs), quality wins and you cut harder to avoid memorizing noise. This is the multimodal echo of the text-side [data scaling-law budgeting](/blog/machine-learning/training-data/data-scaling-laws-and-budgets) argument.
- **Downstream metric.** Zero-shot ImageNet accuracy, retrieval, and generation quality do not peak at the same threshold. LAION's ablations found the sweet spot for CLIP-style zero-shot around 0.28-0.30; teams optimizing for aesthetics or captioning picked differently.

The industry's answer to "you cannot win both" has been to stop treating it as a single dial. DataComp reframed the question as "given a fixed compute budget, which subset of a fixed candidate pool trains the best model?" — turning threshold-picking into a *data-selection benchmark*. The winning entries were not the biggest or the highest-threshold subsets; they were the ones that combined CLIP score with additional signals. That is the direction the field moved, and it is the subject of the companion post on [data filtering networks](/blog/machine-learning/training-data/datacomp-and-data-filtering-networks).

## 7. Dedup and safety: the part CLIP does not do

**Senior rule of thumb: the CLIP filter measures alignment and nothing else — not duplication, not safety, not legality. Those are separate passes, and skipping them is how datasets get recalled.**

CLIP-score filtering answers exactly one question: does this caption match this image? It says nothing about whether you have the same image a thousand times, whether the image is a copyrighted work, or whether it is unsafe or illegal content. Two failure modes here are severe enough to have caused real-world dataset takedowns.

**Near-duplicate images inflate the set and skew the distribution.** The same stock photo, meme, or product image appears across thousands of domains, each with a slightly different caption, each scoring above threshold on its own. The CLIP filter keeps all of them because each pair is individually aligned. The result is a training set that looks like it has two billion images but has far fewer *distinct* ones, with popular images (celebrity photos, viral memes, common stock) massively over-represented. This is the same problem — and the same solution — as text deduplication: you dedup by image embedding (near-duplicate detection in CLIP space) or by perceptual hash, exactly as [deduplication at scale](/blog/machine-learning/training-data/deduplication-at-scale) describes for documents. LAION shipped with a "dedup" annotation for precisely this reason. Left undone, near-duplicates cause the model to memorize and regurgitate specific images — the mechanism behind the Stable Diffusion training-image-extraction results.

**Unsafe and illegal content passes the alignment filter untouched.** A cosine score does not know or care whether an image is pornographic, violent, or — the catastrophic case — child sexual abuse material (CSAM). In late 2023, a Stanford Internet Observatory study found that LAION-5B contained thousands of suspected CSAM URLs; the dataset was pulled from distribution and later re-released as "Re-LAION" only after the URLs were removed against known CSAM hash lists. This is the single most important lesson of web-scraped data: **you must run a dedicated safety pass, and for CSAM specifically you must match against authoritative hash databases (PhotoDNA, the hash lists maintained by NCMEC and IWF) — you cannot classify your way out of it, and you have legal obligations, not just quality ones.** NSFW content more broadly needs its own classifier (LAION shipped NSFW-probability annotations). The full treatment of safety filtering — for both text and images — is in [PII, safety, and toxicity filtering](/blog/machine-learning/training-data/pii-safety-and-toxicity-filtering); the multimodal-specific point is that none of it is done by the CLIP score, and all of it is mandatory.

![A matrix of what the CLIP filter misses: generic captions, hard negatives, unsafe content, and near-duplicates, each with why it slips through and the fix](/imgs/blogs/image-text-pairs-at-scale-6.webp)

The matrix above collects the four things the CLIP threshold does not catch, why each slips through, and the separate defense each needs. Read it as the checklist of passes you must run *in addition to* the cosine filter: a low-information-caption penalty, hard-negative mining, NSFW/CSAM classifiers and hash matching, and embedding-or-hash deduplication. Ship a dataset with only the CLIP filter and you have shipped an unsafe, duplicate-inflated, generic-caption-biased set that happens to have aligned pairs.

## Case study: how LAION-400M and LAION-5B were built

LAION deserves the full case-study treatment because it is the reference implementation of everything above, and because its history — including its recall — is the field's clearest lesson in what the CLIP filter does and does not buy you.

**The construction.** LAION-400M (2021) started from Common Crawl, parsed roughly the 2014-2021 snapshots for `<img>` tags with non-empty `alt`, and produced billions of raw candidate pairs. The team dropped pairs with captions under three characters or images under 5KB, then ran the decisive step: CLIP ViT-B/32 scoring with a **0.28 cosine threshold** for the English subset (0.26 for the multilingual set, which used a multilingual CLIP). What survived — 400 million pairs — was released as a table of URLs, captions, CLIP similarities, NSFW probabilities, and dedup flags, not as image bytes (for copyright and storage reasons). LAION-5B (2022) scaled the same recipe to 5.85 billion pairs across an English subset (2.3B), a multilingual subset (2.2B), and a "rest" bucket, plus curated slices like LAION-Aesthetics (filtered further by a small aesthetic-scoring model, which is what Stable Diffusion's later training runs drew from).

**What it enabled.** Before LAION, training a CLIP-quality model required OpenAI's private 400M WIT dataset — you could use the released model but not reproduce it. LAION-400M let the OpenCLIP project reproduce and then *exceed* the original CLIP's zero-shot accuracy, and LAION-5B + LAION-Aesthetics were the training data behind Stable Diffusion. The entire open text-to-image and open-CLIP ecosystem traces to this pipeline. The CLIP-filter mechanics — self-filtering the training set with a smaller aligned model — are what made a purely web-scraped set good enough to train on. That is the positive lesson: a noisy web corpus plus a cheap alignment filter beats a small clean corpus, and it beats the raw web corpus.

**What went wrong, and the fix.** LAION-5B was a URL list, which meant (a) link rot degraded it over time as images went offline, and (b) LAION could not directly moderate the *content* behind URLs it did not host. In December 2023 the Stanford Internet Observatory identified thousands of suspected CSAM entries. LAION took the datasets down, worked with child-safety organizations (IWF, Canadian Centre for Child Protection) to match against known-CSAM hash lists, and in 2024 released **Re-LAION-5B** with the flagged content removed. The takeaway is not "LAION was careless" — it was the most transparent large dataset of its era, which is *why* the problem was findable and fixable. The takeaway is structural: **a CLIP-alignment filter is orthogonal to a safety filter, and a web-scale image dataset is not shippable on alignment alone.** Every practitioner building on web pairs inherits this obligation.

## Troubleshooting

Each entry is a real failure you will hit, framed as symptom, root cause, and fix.

### Symptom: alt-text garbage survives the CLIP filter

**Symptom.** You set the threshold at 0.28, but spot-checking survivors you still find filename-like captions, SEO strings, and boilerplate that clearly does not describe the image.

**Root cause.** Two possibilities. First, your threshold is too low for your filter model — a ViT-L/14 or a different-pretraining ViT-B/32 has a shifted distribution, and 0.28 on that model keeps more junk than 0.28 on the LAION-2B ViT-B/32. Second, and more common: some "garbage" scores above threshold because the garbage text happens to be loosely image-related (an SEO string full of product nouns on a product image can legitimately score 0.30+). CLIP rewards topical overlap, not caption *quality*.

**Fix.** Re-plot the score distribution on *your* filter model and re-pick the threshold from the histogram, not from a paper. Then stack a cheap text-quality pre-filter *before* CLIP: drop captions under three words, drop captions that are >50% stoplist/SEO tokens, drop captions with no alphabetic content. CLIP is the alignment judge; it was never the grammar or usefulness judge, so give it clean text to score.

### Symptom: the model learns to produce generic, contentless captions

**Symptom.** Your captioning or generation model prefers bland outputs ("a photo of a person", "an image of a landscape"); retrieval is dominated by short generic queries.

**Root cause.** The CLIP-score blind spot. Short generic captions ("a photo", "a picture of a dog") score deceptively high because they are trivially *true* of the image — there is nothing specific to be wrong about. So they sail through the threshold in large numbers and bias the training distribution toward low-information text. You filtered for alignment and accidentally selected for genericness.

**Fix.** Add a caption-information filter alongside the alignment filter: penalize or drop captions below a token-count floor, below a unique-noun count, or above a corpus-frequency threshold (very common exact-duplicate captions like "a photo" are a signal). Better still, *re-caption*: run a strong captioning model over the images and replace the alt-text (the approach BLIP, and later synthetic-caption pipelines, took). This is the multimodal instance of the general lesson from [synthetic data generation](/blog/machine-learning/training-data/synthetic-data-generation) — sometimes the cheapest way to get good labels is to generate them. And it is precisely the gap [data filtering networks](/blog/machine-learning/training-data/datacomp-and-data-filtering-networks) were built to close: learn a filter that predicts *downstream usefulness*, not just cosine alignment.

### Symptom: CLIP is fooled by plausible-but-wrong captions (hard negatives)

**Symptom.** Fine-grained errors survive: "a black cat" on a photo of a black dog, "the Eiffel Tower" on a different tower, "chocolate cake" on a coffee cake. All score above threshold.

**Root cause.** CLIP's alignment is *coarse*. It reliably separates "dog" from "car engine" but is weak at fine-grained distinctions — attributes, counts, spatial relations, and near-category confusions. A caption that is topically right but factually wrong lands in the same score band as a correct one, so the threshold cannot separate them. This is a known limitation of contrastive image-text models, not a bug in your pipeline.

**Fix.** You cannot fix this with a single threshold. The mitigations are hard-negative mining (explicitly training or filtering against plausible-but-wrong pairs) and using a stronger, more fine-grained filter model — SigLIP-based filters and learned data-filtering networks outperform vanilla CLIP here. If fine-grained correctness matters for your task, budget for a second, sharper filtering pass and do not expect the 0.28 cut to deliver attribute-level accuracy.

### Symptom: near-duplicate images inflate the dataset and cause regurgitation

**Symptom.** Your dataset reports two billion pairs, but distinct-image counts are far lower; the trained model memorizes and reproduces specific popular images near-verbatim; certain images dominate retrieval.

**Root cause.** The CLIP filter keeps each aligned pair independently, so the same image scraped from a thousand domains — each with a valid caption — yields a thousand survivors. Duplication is invisible to a per-pair alignment score. Over-represented images then get memorized (the same over-duplication-drives-memorization mechanism documented for text).

**Fix.** Run a dedup pass keyed on the *image*, not the pair: perceptual hashing (pHash) for exact and near-exact duplicates, and near-duplicate clustering in CLIP embedding space for semantic duplicates (crop/resize/watermark variants). Deduplicate *before* you count and *before* you train. This is the image-side of the exact machinery in [deduplication at scale](/blog/machine-learning/training-data/deduplication-at-scale) — MinHash/LSH gives way to embedding-space nearest-neighbor search, but the reweighting logic is identical: duplicates silently upweight whatever they contain.

### Symptom: high download-failure rate and shard skew

**Symptom.** `img2dataset` reports 30-45% failures; some shards are half-empty; certain domains dominate the successes.

**Root cause.** URL-list datasets suffer link rot (images taken down since the crawl), rate limiting and crawler blocks (some hosts return 403 to `img2dataset`'s user agent), timeouts on slow servers, and CDN concentration (a few big hosts serve a disproportionate share). A 20-40% failure rate is *normal*; anything higher usually means you are being blocked or timing out too aggressively.

**Fix.** Expect and budget for attrition — over-provision your candidate list by ~1.5x the pairs you need. Set a realistic `--timeout` (10s) and `--retries` (2), rotate/identify your user agent politely, and monitor per-domain success rates to catch a single host silently failing. Because failures are non-random (they correlate with domain and image age), re-check your language and content distribution *after* download, not just after harvest — attrition can skew the surviving distribution in ways the pre-download metadata does not show.

## When to reach for web image-text pairs — and when not

Reach for CLIP-filtered web pairs when:

- You are **pre-training a foundation vision-language model** (CLIP-style contrastive, a diffusion image generator, a VLM's alignment stage) and need billions of pairs at a cost no annotation budget could match.
- You need **broad visual and conceptual coverage** — the long tail of objects, scenes, styles, and languages that a curated academic dataset (COCO, ImageNet) does not span.
- You can afford the **full downstream pipeline**: download at scale, CLIP filter, dedup, safety pass. If you cannot commit to the safety and dedup passes, you are not ready to use web pairs.
- You value **reproducibility and openness** — LAION and DataComp exist precisely so the community can build on a shared, documented corpus.

Skip web pairs, or supplement them heavily, when:

- Your task needs **fine-grained, attribute-level, or factually precise** captions (medical imaging, technical diagrams, counting, spatial reasoning). The coarse CLIP filter will not deliver the precision, and you are better off with expert annotation or synthetic re-captioning.
- You are training a **small model with a fixed compute budget** where every pair is seen many times — noise memorizes. Cut the threshold hard, or use a curated set; scale is not your bottleneck.
- You **cannot meet the safety and legal obligations** of web-scraped imagery. CSAM hash-matching is not optional, and copyright exposure is real. If you cannot run the safety pass, do not ship the dataset.
- You need **caption quality over caption existence** — for generation and captioning, re-captioning with a strong model often beats any threshold on the original alt-text.

The one-line version: the web's alt-text is the cheapest caption layer that has ever existed, and a CLIP-score threshold is the cheapest way to make it usable — but "usable" means aligned, not clean, not safe, and not deduplicated. The CLIP filter is the first of several passes, not the only one. Build the rest, or you have not built a dataset — you have built a liability that happens to contain some good pairs.

## Further reading

- [ViT, SigLIP, and DINO explained](/blog/machine-learning/computer-vision/vit-siglip-dino-explained) — how the encoders behind the CLIP score actually learn an aligned space, and why SigLIP's loss makes a sharper filter.
- [DataComp and data filtering networks](/blog/machine-learning/training-data/datacomp-and-data-filtering-networks) — the next step beyond a single cosine threshold: learning a filter that predicts downstream usefulness.
- [Deduplication at scale](/blog/machine-learning/training-data/deduplication-at-scale) — the reweighting logic and the machinery you point at image embeddings to kill near-duplicates.
- [PII, safety, and toxicity filtering](/blog/machine-learning/training-data/pii-safety-and-toxicity-filtering) — the mandatory safety pass, including the CSAM obligations that recalled LAION-5B.
- [Synthetic data generation](/blog/machine-learning/training-data/synthetic-data-generation) — re-captioning as the fix for the generic-caption blind spot.
- The [LAION](https://laion.ai/) project pages and the [`img2dataset`](https://github.com/rom1504/img2dataset) and [`open_clip`](https://github.com/mlfoundations/open_clip) repositories — the reference implementations of everything above.
