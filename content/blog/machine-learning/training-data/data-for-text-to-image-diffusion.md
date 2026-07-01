---
title: "Data for Text-to-Image Diffusion: Aesthetics, Captions, and the Memorization Trap"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "What a text-to-image diffusion model actually needs from its data — image beauty, captions that describe, native-resolution buckets, and dedup — and how aesthetic filtering, DALL-E 3-style recaptioning, and the memorization trap change how you build the pipeline."
tags:
  - training-data
  - text-to-image
  - diffusion-models
  - aesthetic-filtering
  - recaptioning
  - aspect-ratio-bucketing
  - memorization
  - deduplication
  - stable-diffusion
  - dalle-3
category: "machine-learning"
subcategory: "Training Data"
author: "Hiep Tran"
featured: true
readTime: 30
---

The first time you train a text-to-image diffusion model on a web-scraped image-text corpus that passed a CLIP filter, you get a rude surprise. The images are, technically, aligned with their captions. The [CLIP score](/blog/machine-learning/image-generation/text-encoders-and-prompt-conditioning) says so. And yet the model produces muddy, low-contrast, badly-composed pictures that ignore half of every prompt you type. You filtered for "the caption matches the image," which is exactly the wrong thing to optimize when your downstream task is *generation* rather than *retrieval*. A retrieval model needs to know that this photo is more dog-like than cat-like. A generator needs the photo to be *beautiful*, needs the caption to actually *describe* the photo in enough detail that the model can learn to render a prompt, needs the image kept at its native shape so you are not square-cropping the subject out of frame, and — the one everybody learns the hard way — needs its near-duplicate copies removed so the model does not memorize and regurgitate training images.

![What a T2I generator needs from data versus what CLIP filtering actually checks for](/imgs/blogs/data-for-text-to-image-diffusion-1.webp)

The matrix above is the mental model for this entire post. Down the left are the properties that decide whether a generator turns out good; across the top, the two lenses — what a CLIP contrastive filter rewards, and what the diffusion model actually needs. They barely overlap. CLIP filtering is a coarse gate that removes the truly mismatched pairs and nothing more. It is blind to beauty, satisfied with two-word alt-text, actively encourages a 224-pixel square crop, tolerates duplicates, and ignores watermarks. Every one of those is a first-class concern for a generator, and each gets its own stage in a real text-to-image data pipeline. This post is a tour of those stages: aesthetic filtering, resolution and aspect-ratio bucketing, caption engineering (the recaptioning thesis that made modern models follow prompts), and dedup as the primary defense against memorization — plus a worked scenario with survival rates and a troubleshooting section for when each stage backfires.

If you have not already, it helps to have the [latent diffusion and Stable Diffusion](/blog/machine-learning/image-generation/latent-diffusion-and-stable-diffusion) mechanics in your head, because the data decisions here are downstream of that architecture. But you can read this cold; I define what matters as we go.

## Why text-to-image data is a different problem

Start with the mismatch, because "just filter the LAION dump with CLIP and train" is the assumption that produces a mediocre model, and a staff engineer needs the mechanism, not the slogan.

| Assumption (from the retrieval world) | Naive view | Reality for a generator |
| --- | --- | --- |
| CLIP score ≥ 0.28 means "good pair" | High alignment is all you need | Alignment is necessary, not sufficient; the model also learns *style, composition, and quality* from every image |
| Alt-text is the caption | The web already labeled the images | Alt-text is sparse, wrong, or SEO spam; the model learns to ignore prompt words it never saw described |
| Resize everything to 512×512 | Square is simplest for batching | Center-cropping to square deletes subject content and teaches the model bad framing |
| Duplicates are harmless noise | The model averages them out | Duplicates are *upweighted*, memorized, and regurgitated near-verbatim — a legal and privacy liability |
| More data is always better | Scale fixes everything | A smaller, prettier, better-captioned, deduped set beats a larger raw one at the same compute |

The through-line: a discriminative model only has to *rank* images, so it is robust to ugly or badly-captioned examples as long as the *relative* alignment is right. A generative model has to *reproduce the distribution* it was trained on. If that distribution is 60 percent mediocre stock photos with watermarks and one-word captions, the model learns to generate mediocre stock photos with watermarks that ignore your prompt. Garbage in is not averaged out; it is *sampled from*. That single fact reorganizes the whole pipeline around quality and description rather than mere alignment.

There is a second, subtler difference. Because diffusion training is a denoising objective over the entire image, every pixel is a training signal. A watermark in the corner is not ignored — it is a stable, high-frequency pattern the model happily learns to reproduce, which is why early Stable Diffusion loved to paint a Getty Images watermark onto its outputs. The model is an extremely diligent student of *everything* in the frame, wanted or not. That is the anxiety underneath every stage below.

## Aesthetic filtering: teaching the pipeline what "pretty" means

**Senior rule of thumb: you cannot make a model produce beautiful images by prompting; you make it produce beautiful images by training it almost exclusively on beautiful images.** The single change that turned Stable Diffusion from "interesting research artifact" into "people make art with this" was aesthetic filtering — training the final checkpoint on the high-scoring tail of LAION rather than the whole thing.

The trick is that "beautiful" is learnable from a small amount of human judgment. You do not hand-label a billion images. You collect a few hundred thousand human aesthetic ratings — the AVA dataset (photos rated 1–10 by a photography community) and the Simulacra Aesthetic Captions (SAC) set of human ratings on generated images — and fit a tiny regression head on top of frozen CLIP image embeddings. CLIP already encodes "what is in the image and roughly how it looks" into a 768-dimensional vector; a two-or-three-layer MLP on top of that vector is enough to predict a human aesthetic score with useful accuracy. Then you run that predictor over your entire raw pool and keep the high-scoring tail.

![Aesthetic filtering: fit a small predictor on human ratings, score the pool, keep the high scores and drop the rest](/imgs/blogs/data-for-text-to-image-diffusion-2.webp)

The graph above is the whole mechanism. Two inputs feed the scorer: the human ratings train the MLP head, and the raw image pool is embedded with CLIP. The head scores every image on a 0–10 scale; a threshold splits the pool into a kept set and a dropped set. LAION-Aesthetics V2 is exactly this — a predictor released alongside the LAION-5B subsets, with published cuts at score ≥ 4.5 (about 1.2 billion pairs), ≥ 5.0, and ≥ 6.0 (a few million of the prettiest). Stable Diffusion 1.x was fine-tuned on the ≥ 5.0-ish tail after a broader pretraining, and that is the step people *see* when they say SD "looks good."

Here is the predictor and the filter, using `open_clip` and the actual LAION MLP shape:

```python
import torch
import torch.nn as nn
import open_clip
from PIL import Image

class AestheticPredictor(nn.Module):
    # The LAION-Aesthetics V2 head: a small MLP on CLIP ViT-L/14 image embeds.
    def __init__(self, in_dim: int = 768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.Dropout(0.2),
            nn.Linear(1024, 128),    nn.Dropout(0.2),
            nn.Linear(128, 64),      nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, clip_embed: torch.Tensor) -> torch.Tensor:
        return self.layers(clip_embed)  # a single score per image

device = "cuda"
clip, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14", pretrained="openai"
)
clip = clip.to(device).eval()

predictor = AestheticPredictor(768).to(device).eval()
predictor.load_state_dict(torch.load("sac+logos+ava1-l14-linearMSE.pth"))

@torch.no_grad()
def aesthetic_score(paths: list[str]) -> torch.Tensor:
    batch = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths])
    feats = clip.encode_image(batch.to(device))
    feats = feats / feats.norm(dim=-1, keepdim=True)   # L2-normalize — this matters
    return predictor(feats.float()).squeeze(-1)         # shape [B]

scores = aesthetic_score(["a.jpg", "b.jpg", "c.jpg"])
keep = scores >= 5.0
```

The one non-obvious detail that bites people: **L2-normalize the CLIP embedding before the MLP**, because the predictor was trained on normalized features. Skip it and every score collapses toward the mean and your threshold does nothing. I have personally spent an afternoon debugging a "the aesthetic filter isn't filtering" ticket that was exactly this missing `.norm()`.

### Second-order optimization: where you set the threshold is a diversity knob

The threshold is not a quality dial you turn up for free. Aesthetic predictors have a *taste*, and it is a specific one: they were trained largely on the kind of images a photography community up-votes, so they reward shallow depth of field, golden-hour lighting, high saturation, and centered subjects. Push the threshold to ≥ 6.0 and you do not just remove ugly images — you remove *entire categories*: diagrams, documents, screenshots, product shots on white, most line art, and anything deliberately flat or minimal. Your model gets prettier and simultaneously loses the ability to render a whiteboard, a UI, or a technical illustration.

The practical move is a two-population strategy: a high aesthetic bar for the "make it pretty" bulk of the data, plus a deliberately-preserved slice of functional imagery (documents, charts, product shots) that bypasses the aesthetic gate entirely, kept for coverage. SDXL and later models are explicit about *not* aesthetic-filtering as aggressively as SD 1.5, precisely to keep this coverage, and instead lean harder on conditioning the model on aesthetic and resolution scores at training time so quality becomes a *controllable input* rather than a filtered-away property.

## Resolution and aspect-ratio bucketing

**Senior rule of thumb: never square-crop your training data if you can avoid it; group images by shape and train each batch at that shape's native resolution.** The naive pipeline resizes everything to 512×512, which does two kinds of damage. It center-crops a 16:9 landscape down to a square, deleting the left and right thirds — so a model trained this way has literally never seen the edges of wide images and generates cramped, centered compositions. And it forces portrait phone photos into a square, squashing or cropping faces. NovelAI's team documented this precisely: square-crop training is why early models could not draw a person's full body or a wide landscape without chopping it.

The fix is aspect-ratio bucketing. You define a set of buckets — target resolutions that all have roughly the same pixel budget (so they cost the same to train) but different shapes — assign each image to the closest-matching bucket, and build batches from within a single bucket so every image in a batch has identical dimensions and the tensor packs cleanly.

<figure class="blog-anim">
<svg viewBox="0 0 760 420" role="img" aria-label="Images of varied aspect ratios drop from a raw pool into portrait, square, and landscape buckets so each batch trains at native resolution" style="width:100%;height:auto;max-width:820px">
<style>
.bk-pool{fill:none;stroke:var(--border,#d1d5db);stroke-width:1.5;stroke-dasharray:6 6}
.bk-bucket{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.bk-ghost{fill:none;stroke:var(--border,#d1d5db);stroke-width:1.5;stroke-dasharray:4 5}
.bk-tile{fill:var(--accent,#6366f1);opacity:.9}
.bk-t{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.bk-s{font:600 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
@keyframes bk-drop{0%,22%{transform:translateY(-228px)}62%,100%{transform:translateY(0)}}
.bk-mv{animation:bk-drop 9s ease-in-out infinite alternate}
.bk-d2{animation-delay:1.2s}
.bk-d3{animation-delay:2.4s}
@media (prefers-reduced-motion:reduce){.bk-mv{animation:none}}
</style>
<text class="bk-t" x="380" y="26">raw pool: mixed aspect ratios, kept at native size</text>
<rect class="bk-pool" x="30" y="44" width="700" height="128" rx="10"/>
<rect class="bk-bucket" x="30" y="250" width="212" height="152" rx="10"/>
<rect class="bk-bucket" x="274" y="250" width="212" height="152" rx="10"/>
<rect class="bk-bucket" x="518" y="250" width="212" height="152" rx="10"/>
<text class="bk-t" x="136" y="276">portrait 3:4</text>
<text class="bk-t" x="380" y="276">square 1:1</text>
<text class="bk-t" x="624" y="276">landscape 16:9</text>
<rect class="bk-ghost" x="112" y="288" width="48" height="104" rx="6"/>
<rect class="bk-ghost" x="340" y="300" width="80" height="80" rx="6"/>
<rect class="bk-ghost" x="564" y="312" width="120" height="68" rx="6"/>
<rect class="bk-tile bk-mv" x="112" y="288" width="48" height="104" rx="6"/>
<rect class="bk-tile bk-mv bk-d2" x="340" y="300" width="80" height="80" rx="6"/>
<rect class="bk-tile bk-mv bk-d3" x="564" y="312" width="120" height="68" rx="6"/>
<text class="bk-s" x="380" y="418">each bucket batches one shape; no square crop discards content</text>
</svg>
<figcaption>Aspect-ratio bucketing: images of different shapes fall into the bucket that matches their ratio, so every batch trains at native resolution instead of being square-cropped.</figcaption>
</figure>

The animation shows the mechanism: each image drops into the bucket whose shape matches it — a portrait phone photo into the 3:4 bucket, a wide landscape into 16:9 — and every image in a bucket is resized to that bucket's exact resolution so the batch tensor is uniform. No content is thrown away by cropping; you only ever downscale to the bucket's pixel budget. Here is the assignment logic, and because bucketing runs right after (or fused with) aesthetic filtering in a real pipeline, I show them together:

```python
from dataclasses import dataclass

# Buckets share ~a 1-megapixel budget but differ in shape. SDXL trains near
# 1024x1024-equivalent; scale these down for a 512-budget model.
BUCKETS = [
    (1024, 1024),  # 1:1
    (896, 1152),   # ~3:4 portrait
    (1152, 896),   # ~4:3 landscape
    (832, 1216),   # ~2:3 portrait
    (1216, 832),   # ~3:2 landscape
    (768, 1344),   # ~9:16 tall
    (1344, 768),   # ~16:9 wide
]

@dataclass
class Sample:
    path: str
    width: int
    height: int
    aesthetic: float

def assign_bucket(w: int, h: int) -> tuple[int, int]:
    ar = w / h
    # closest bucket by aspect ratio in log space (symmetric for w/h vs h/w)
    import math
    return min(BUCKETS, key=lambda b: abs(math.log(ar) - math.log(b[0] / b[1])))

def curate(samples: list[Sample], min_side: int = 256, aes_thresh: float = 5.0):
    buckets: dict[tuple[int, int], list[Sample]] = {}
    dropped = {"tiny": 0, "ugly": 0}
    for s in samples:
        if min(s.width, s.height) < min_side:      # resolution floor
            dropped["tiny"] += 1
            continue
        if s.aesthetic < aes_thresh:               # aesthetic gate
            dropped["ugly"] += 1
            continue
        buckets.setdefault(assign_bucket(s.width, s.height), []).append(s)
    return buckets, dropped
```

### Second-order optimization: bucket population, not just bucket definition

The gotcha nobody mentions in the tutorials: your buckets will be *wildly* unbalanced. The web is overwhelmingly 4:3, 3:2, and 1:1; you will have a hundred times more landscape-ish images than 9:16 verticals. If you sample batches uniformly from buckets, the rare-shape buckets get seen constantly relative to their size (over-fitting those few images) or the common buckets dominate every epoch. The standard fix is to sample buckets in proportion to their population, and to set a minimum bucket size (drop or merge buckets with too few images to form a stable batch). A related failure: if you add a 9:16 bucket but only 3,000 images land in it, the model will memorize those 3,000 — a bucketing decision quietly created a memorization hot-spot, which is the trap the next two sections are about.

## Caption engineering: the recaptioning thesis

**Senior rule of thumb: the model can only learn to follow the words that appear in its training captions. If your captions are two-word alt-text, your model follows two-word prompts and ignores everything else.** This is the single biggest lever on prompt-following, and it is the insight that separated DALL-E 3 from everything before it.

Web alt-text is terrible training signal for generation. It is written for SEO and accessibility, not description: "IMG_2043.jpg", "click here", "dog", "photo", or a keyword-stuffed spam string. When the model trains on "dog" attached to a picture of a golden retriever puppy sitting on a red rug in soft window light, it learns to associate the *entire* image — puppy, rug, lighting, composition — with the single token "dog". It never learns that "red rug" or "soft light" are things a prompt can ask for, because it never saw those words paired with those pixels. So at inference, when you prompt "a golden retriever puppy on a red rug in soft window light," the model shrugs and gives you a generic dog.

![DALL-E 3 recaptioning: replacing sparse alt-text with dense synthetic captions teaches the model to bind prompt words to pixels](/imgs/blogs/data-for-text-to-image-diffusion-4.webp)

The before/after above is the recaptioning thesis. On the left, raw alt-text: sparse, sometimes garbage, and only a handful of words bind to the image, so the model ignores prompt details. On the right, a vision-language model reads the image and writes a dense, descriptive caption — "a golden retriever puppy sitting on a red woven rug, soft window light from the left, shallow depth of field" — and now dozens of words bind to real pixels, so the model learns to follow detailed prompts. DALL-E 3's paper, *Improving Image Generation with Better Captions*, is the definitive result here: they trained an image captioner to produce long, highly-descriptive synthetic captions, recaptioned their corpus, and showed a dramatic jump in prompt-following. The training pipeline for a caption generator and the broader recipe are the subject of the sibling post on [recaptioning and VLM instruction data](/blog/machine-learning/training-data/recaptioning-and-vlm-instruction-data); here we care about the *data blend* decision.

### The blend ratio, and why you don't go 100% synthetic

The counter-intuitive finding from DALL-E 3: the best mix is *mostly* synthetic captions, not entirely. They landed on roughly **95 percent synthetic, 5 percent original** captions. Why keep any of the noisy originals? Because a model trained on 100 percent VLM captions overfits to the captioner's specific style and vocabulary — its favorite sentence structures, its habit of starting every caption with "a photo of," its blind spots. Real user prompts do not sound like a VLM. The 5 percent of original human captions, messy as they are, keep the model anchored to the actual distribution of human language and proper nouns the VLM might not know. It is a regularizer.

| Caption strategy | Prompt-following | Failure mode |
| --- | --- | --- |
| 100% original alt-text | Poor — ignores most prompt words | Generic outputs, no fine control |
| 100% synthetic (VLM) | Strong on VLM-style prompts | Overfits captioner style; brittle on real/short prompts |
| ~95% synthetic + ~5% original | Best overall | Requires running a VLM over the whole corpus (expensive) |
| Descriptive + short-tag mix | Robust across prompt lengths | More pipeline complexity (dropout, length mixing) |

The cost is real: recaptioning a corpus means running a vision-language model over hundreds of millions of images, which is a serious inference bill and a multi-day GPU job even with a small captioner. But it is, dollar for dollar, the highest-leverage thing you can do to prompt-following — more than architecture tweaks, more than longer training. The [MMDiT modern text-to-image recipe](/blog/machine-learning/image-generation/mmdit-and-the-modern-text-to-image-recipe) assumes recaptioned data as a baseline; the architecture improvements sit on top of that data foundation, not instead of it.

## Dedup and the memorization trap

**Senior rule of thumb: a diffusion model does not just learn the distribution of your data — for any image it saw many times, it can memorize and reproduce it near-verbatim. Deduplication is the primary mitigation, and it is not optional.** This is where text-to-image data curation stops being about quality and starts being about liability.

The mechanism is the same one that drives memorization in language models, and it hinges on duplication. An image that appears once in a billion-image corpus contributes a faint, well-generalized signal. The same image duplicated ten thousand times — because it is a stock photo syndicated across thousands of sites, or a famous painting, or a product shot reused everywhere — is seen so often that the model does not generalize it; it *stores* it. At generation time, a prompt close to that image's caption can pull the stored pixels back out almost exactly.

![How duplicated images become regurgitated generations, and how dedup breaks the chain](/imgs/blogs/data-for-text-to-image-diffusion-5.webp)

The graph above traces both branches. Copies A, B, and C of the same image collapse into "seen N times" in the corpus. Down the top path — the failure — the model memorizes the exact pixels, and at inference a matching prompt extracts a near-verbatim regurgitation. Down the bottom path — the fix — dedup collapses the copies back to a single instance, the model generalizes instead of memorizing, and it produces a genuinely novel image. Two papers made this concrete and unavoidable. Carlini et al., *Extracting Training Data from Diffusion Models*, showed you can prompt Stable Diffusion into emitting near-exact copies of specific training images, and that the extractable images were overwhelmingly the *duplicated* ones. Somepalli et al., *Diffusion Art or Digital Forgery?*, showed the same replication behavior and tied it directly to duplication in the training set — deduplicate, and replication drops sharply.

Detecting whether a generation is a memorized copy is the same nearest-neighbor problem as [deduplication at scale](/blog/machine-learning/training-data/deduplication-at-scale), pointed at a different target. You embed generated images and search them against an index of the training set; a suspiciously high similarity to a training image is a replication flag. SSCD (a self-supervised copy-detection embedding) is the standard tool, but CLIP embeddings plus a FAISS index get you most of the way:

```python
import faiss
import numpy as np
import torch

# Assume `embed(images) -> np.ndarray[N, D]` uses SSCD or L2-normalized CLIP.
# Build the index once over the training set.
train_emb = embed(training_images)            # [N_train, D], L2-normalized
index = faiss.IndexFlatIP(train_emb.shape[1])  # inner product == cosine on unit vecs
index.add(train_emb.astype(np.float32))

def replication_flags(generated_images, threshold: float = 0.95):
    gen_emb = embed(generated_images).astype(np.float32)
    sims, idxs = index.search(gen_emb, k=1)     # nearest training image per gen
    flags = []
    for gi, (sim, ti) in enumerate(zip(sims[:, 0], idxs[:, 0])):
        if sim >= threshold:                    # near-verbatim copy of training[ti]
            flags.append({"gen": gi, "train_idx": int(ti), "sim": float(sim)})
    return flags
```

You run this two ways. Offline, over a sample of your own generations, as a *release gate*: if more than a fraction of a percent of generations trip the threshold, you have a memorization problem and need more aggressive dedup before shipping. And you deduplicate the *training set* up front with the same embedding-plus-clustering machinery so the duplicates never drive memorization in the first place. The order matters: dedup is cheap insurance you pay once; a memorization scandal after launch is not something you can filter your way out of retroactively.

> Dedup is the only stage that changes *how many times* a survivor is counted rather than *whether* it survives. That is why it is simultaneously your quality lever and your legal shield.

## Watermark and NSFW filtering

**Senior rule of thumb: the model reproduces whatever is stably present in the frame — so anything you don't want it to draw, you remove from the training data, not from the prompt.** Two categories dominate: watermarks and unsafe content.

Watermarks are the cleanest example of "the model learns everything in the frame." Stock-photo watermarks (Getty, Shutterstock, Alamy overlays) appear on millions of images with near-identical placement and style. Train on them and the model learns that "a professional-looking photo" statistically co-occurs with a semi-transparent watermark, so it paints one on. The fix is a watermark classifier — LAION shipped `pwatermark`, a predictor score per image — used to drop or down-weight watermarked images. It is imperfect (it misses subtle marks and false-positives on legitimate text overlays), so the pragmatic setting is a conservative threshold that removes the obvious cases plus periodic manual spot-checks of what survives.

NSFW filtering is higher-stakes and has its own failure modes in both directions. Under-filter and the model learns to produce unsafe content, including the categories that are legally radioactive. Over-filter and you strip out legitimate content — art, medical imagery, anything the classifier flags on skin tone or composition — and you can degrade the model's competence on human anatomy in general. Stable Diffusion 2.0 is the canonical cautionary tale: a much more aggressive NSFW filter on the training data than 1.5 had, and the community's immediate complaint was that 2.0 got noticeably worse at generating people, because "remove unsafe images of humans" over-generalized into "remove a lot of images of humans." The safety, watermarking, and provenance concerns downstream of the model — output filtering, C2PA signing — are covered in [safety, watermarking, and provenance](/blog/machine-learning/image-generation/safety-watermarking-and-provenance); here the point is that *training-data* filtering is the first and bluntest of these instruments, and it trades off against capability.

The practical stance: use a multi-signal NSFW classifier (not a single score), set thresholds per category rather than one global cut, keep a documented audit of what you removed, and validate on a held-out human-competence benchmark so you catch an SD-2.0-style regression *before* release rather than from the launch-day thread.

## A worked scenario: building the pipeline over a raw pool

Let me make all of this concrete with numbers. Suppose you start with a raw pool of 2.0 billion image-text pairs — a plausible dedup-of-a-dedup slice of a LAION-scale crawl, the kind of starting point discussed in [image-text pairs at scale](/blog/machine-learning/training-data/image-text-pairs-at-scale). You run the five stages in order and track survival at each.

![The full text-to-image data pipeline end to end, with survival rate at each stage](/imgs/blogs/data-for-text-to-image-diffusion-6.webp)

The pipeline figure above is the funnel; here is the same thing as a table you can plug your own rates into:

| Stage | Operation | Retained | Survivors | Why the cut |
| --- | --- | --- | --- | --- |
| 0. Raw pool | — | 100% | 2,000M | Starting corpus |
| 1. Safety | Drop NSFW + watermark + broken/CSAM-scanned | 75% | 1,500M | Liability + watermark reproduction |
| 2. Resolution / aspect | Require min side ≥ 256px, valid aspect | 80% | 1,200M | Tiny thumbnails and degenerate shapes are useless |
| 3. Aesthetic | Keep predictor score ≥ 5.0 | 40% | 480M | The big cut — most of the web is not pretty |
| 4. Dedup | Collapse near-duplicate clusters | 80% | 384M | Kill memorization + upweighting of syndicated images |
| 5. Recaption | Run VLM captioner over survivors | 100% | 384M | No drop; every survivor gets a dense caption |

You end with **384 million training-ready pairs — about 19 percent of the raw pool** — each of which is reasonably beautiful, natively-shaped, deduplicated, and densely captioned. That 19 percent is not a failure of the pipeline; it is the pipeline working. The aesthetic stage alone throws away 60 percent of what reaches it, and that is the stage most responsible for the model looking good. A team that ships the 2.0B raw pool because "more data is better" trains longer, spends more compute, and gets a worse-looking model that ignores prompts and occasionally regurgitates a stock photo. A team that ships the 384M curated set trains faster and gets the model people actually want.

Two numbers worth internalizing from this funnel. First, **recaptioning is applied to survivors, not the raw pool** — you recaption 384M images, not 2.0B, because running a VLM over the images you are about to throw away is a waste of a very expensive inference budget. Order your pipeline so the cheap filters (resolution, safety heuristics) run first and the expensive operations (aesthetic scoring needs a CLIP forward pass; recaptioning needs a full VLM forward pass) run last, on the smallest surviving set. Second, **the survival rate is a design parameter, not a constant.** Want a prettier model with narrower coverage? Push the aesthetic threshold to 6.0 and watch survivors drop to ~150M. Want broader coverage at some cost to average beauty? Drop to 4.5 and keep ~700M. The threshold is where you spend your quality-versus-diversity budget.

## Troubleshooting

Every stage above has a characteristic way of going wrong in production. Here is the symptom-to-fix map for the four that will actually page you.

### Memorization and replication: the model emits a near-copy of a training image

**Symptom.** A user prompts something close to a famous image's description and the model returns a near-pixel-perfect copy — a specific painting, a stock photo, a recognizable photograph. Or your offline replication gate lights up: more than a fraction of a percent of sampled generations have cosine similarity ≥ 0.95 to a training image.

**Cause.** That image (or a tight near-duplicate cluster of it) appeared many times in the training set, so the model stored it instead of generalizing from it. Almost always this traces to insufficient dedup — the copies were byte-different (re-encoded, resized, lightly cropped, re-watermarked) so exact-hash dedup missed them, and near-duplicate dedup was never run or was run at too loose a threshold.

**Fix.** Run near-duplicate dedup with embedding-based clustering (SSCD or CLIP + FAISS) at a threshold tight enough to catch re-encodes, and collapse each cluster to one representative *before* training — see [deduplication at scale](/blog/machine-learning/training-data/deduplication-at-scale) for the banding math. If you cannot retrain immediately, an interim mitigation is the release gate above plus output-side filtering that rejects generations too close to the training index. But dedup at the source is the only real fix; everything else is a bandage.

### Caption train/inference mismatch: dense-trained model chokes on short prompts

**Symptom.** After recaptioning, your benchmark prompt-following scores go *up*, but real users complain the model got worse. Short prompts like "a cat" produce generic, low-quality results, while long paragraph-style prompts are excellent.

![Caption train/inference mismatch and the caption-dropout fix](/imgs/blogs/data-for-text-to-image-diffusion-7.webp)

**Cause.** The before/after figure shows it. You trained almost entirely on 150–250-token dense synthetic captions, so the model's competence lives in that regime. A two-word prompt is now *off-distribution* — the model rarely saw two-word captions during training — so it falls back to a weak prior and gives you generic output. You optimized for the benchmark (which uses detailed prompts) and pessimized the median user (who types three words).

**Fix.** Keep both prompt lengths in-distribution. Two levers, usually combined: (1) **caption-length mixing** — for each image, sometimes train on the full dense caption, sometimes on a short summary or the original tag, so the model sees both regimes; and (2) **caption dropout** — the classifier-free-guidance dropout you already run for conditioning also acts as regularization, and biasing some of that dropout toward *truncated* captions rather than empty ones teaches robustness to short prompts. Here is the dataset logic:

```python
import random

def make_caption(sample, synthetic: str, original: str,
                 p_synthetic: float = 0.90,
                 p_short: float = 0.20,
                 p_drop: float = 0.10) -> str:
    # 1. Classifier-free-guidance dropout: sometimes no caption at all.
    if random.random() < p_drop:
        return ""                                  # unconditional training signal

    # 2. Blend synthetic vs original (DALL-E 3 style ~90/10).
    caption = synthetic if random.random() < p_synthetic else original

    # 3. Length mixing: sometimes truncate the dense caption to a short prompt
    #    so short user prompts stay in-distribution.
    if caption is synthetic and random.random() < p_short:
        words = caption.split()
        caption = " ".join(words[: random.randint(2, 8)])  # a short "prompt-like" caption
    return caption
```

Tune `p_short` up if short-prompt quality is your problem; the DALL-E 3-style `p_synthetic` around 0.9 keeps you anchored to real language. The empty-caption `p_drop` (typically 0.1) is the standard CFG dropout you need anyway.

### Aesthetic-filter bias: the model collapses toward one "look"

**Symptom.** Everything the model generates looks the same — shallow depth of field, warm lighting, centered subject, high saturation. It cannot render a flat diagram, a document, a screenshot, or a deliberately minimal composition. Diversity metrics on generations are low even at high guidance-scale variety.

**Cause.** The aesthetic predictor has a taste, and an aggressive threshold amplified it. By keeping only score ≥ 6.0, you removed entire visual categories that the predictor scores low — not because they are bad, but because they are not the *kind* of image a photography community up-votes. Your training distribution collapsed onto one aesthetic mode, and the model faithfully learned that mode.

**Fix.** Lower the threshold, or better, split the population: a high aesthetic bar for the bulk plus a preserved, un-gated slice of functional and diverse imagery for coverage. The more modern approach (SDXL onward) is to *condition on* the aesthetic score at training time rather than filter by it — feed the score in as a control signal — so the model learns the full distribution but can be *steered* toward high aesthetics at inference. That gives you beauty on demand without paying for it in diversity. Measure this deliberately: track category coverage (can it still draw a chart, a UI, a full-body figure?) on a held-out prompt set every time you touch the threshold.

### NSFW leakage or over-filtering: unsafe outputs, or a model that can't draw people

**Symptom (leakage).** The model produces unsafe content it should not. **Symptom (over-filter).** The model became visibly worse at human anatomy, faces, or bodies after you tightened the safety filter — the Stable Diffusion 2.0 regression.

**Cause.** Leakage means the safety classifier's recall was too low — unsafe images slipped through, especially borderline or stylized ones a single-score classifier misses. Over-filtering means precision was too low in the wrong direction: "remove unsafe images of humans" over-generalized into "remove a lot of images of humans," and human competence went with it.

**Fix.** Use a *multi-signal* classifier with per-category thresholds rather than one global NSFW score, so you can be strict on the legally-radioactive categories and lenient on, say, artistic nudity if that is in scope. Keep a documented audit of what each threshold removes. And gate every change on two held-out benchmarks: a safety benchmark (did leakage go down?) and a human-competence benchmark (did anatomy quality hold?). Shipping a safety change without the second benchmark is how SD 2.0's regression made it to release. When in doubt, filter more narrowly and lean on *output-side* safety (covered in [safety, watermarking, and provenance](/blog/machine-learning/image-generation/safety-watermarking-and-provenance)) for the residual risk.

## Case studies from production

### 1. LAION-Aesthetics and the "SD looks good" moment

Stable Diffusion 1.x pretrained on a broad LAION slice, then fine-tuned on the LAION-Aesthetics high-scoring tail (roughly score ≥ 5, with later checkpoints pushing higher). The community-visible effect was immediate and out of proportion to the engineering: the same architecture, same text encoder, same training objective, but the outputs went from "technically an image of the prompt" to "something you would post." The lesson that propagated through every subsequent model: the aesthetic fine-tune is not a nice-to-have polish step, it *is* the perceived quality of the model, and it comes almost entirely from *which images you kept*, learned from a few hundred thousand human ratings rather than from any change to the model.

### 2. DALL-E 3 and the recaptioning thesis

OpenAI's *Improving Image Generation with Better Captions* is the paper that reframed prompt-following as a *data* problem, not a *model* problem. They trained a bespoke image captioner to emit long, descriptive captions, recaptioned their training corpus, and found that a model trained on ~95 percent synthetic captions followed complex prompts dramatically better than one trained on alt-text — placing objects correctly, honoring counts and spatial relations, and rendering text. The 95/5 blend, not 100/0, was the other durable finding: a splash of noisy original captions regularizes against overfitting the captioner's style. Every serious open and closed model since (SD3, the MMDiT family, the current commercial systems) treats recaptioned data as table stakes.

### 3. Carlini et al.: extracting training images from diffusion models

*Extracting Training Data from Diffusion Models* demonstrated that Stable Diffusion could be prompted into emitting near-exact copies of specific training images, and — the load-bearing detail — that the extractable images were overwhelmingly the *duplicated* ones. Memorization was not uniform across the training set; it concentrated on images the model saw many times. This turned "dedup improves quality" into "dedup is a privacy and copyright control," and it is why a modern pipeline treats near-duplicate removal as mandatory rather than an optimization. It also gave teams a concrete red-team procedure: try to extract, measure the rate, gate the release.

### 4. Somepalli et al.: replication tied to duplication

*Diffusion Art or Digital Forgery?* independently showed diffusion models replicating training content and traced the behavior directly to training-set duplication, then showed that deduplicating the training data measurably reduced replication. The pairing with Carlini is what made the field stop arguing: two groups, different methods, same conclusion — duplication drives memorization, dedup mitigates it. If you ever need to justify the dedup engineering budget to someone who thinks it is premature optimization, these two papers are the argument.

### 5. Stable Diffusion 2.0's NSFW over-filter regression

SD 2.0 applied a much more aggressive training-data NSFW filter than 1.5. The intent was safety; the side effect was a model noticeably worse at generating people, because the filter's over-broad removal of images containing humans degraded the model's general human-anatomy competence. The community reaction was swift and the episode became the standard reference for "safety filtering trades against capability, and you must measure the trade." The fix pattern that emerged — per-category thresholds, multi-signal classifiers, and a human-competence benchmark gating every filter change — is now standard practice precisely because 2.0 shipped without it.

### 6. Aspect-ratio bucketing at NovelAI and SDXL

NovelAI publicized aspect-ratio bucketing as the fix for square-crop damage — the reason earlier models could not draw full bodies or wide scenes without chopping subjects at the frame edge. Rather than center-crop everything to 512×512, they bucketed by shape and trained each batch at native aspect. SDXL adopted the same idea at scale and additionally *conditioned on* the original resolution and crop parameters, so the model learns from cropped data without inheriting the crop artifacts and can be told at inference to generate an uncropped, full-resolution composition. Bucketing is now standard in essentially every serious training stack; the residual craft is in bucket population balancing, not the idea itself.

## When to reach for heavy data curation, and when not to

**Reach for the full pipeline when:**

- You are training a general-purpose text-to-image *foundation* model from scratch or doing a large continued-pretraining run. Every stage pays for itself at this scale.
- Prompt-following is a product requirement. Recaptioning is the single biggest lever and there is no substitute for it.
- You will *ship* the model publicly. Dedup (memorization/copyright) and NSFW filtering (safety) move from optimizations to obligations.
- Your raw pool is web-scraped. Web data is watermarked, duplicated, badly captioned, and mostly ugly by default; the pipeline is how you undo that.

**Skip or trim it when:**

- You are LoRA-fine-tuning on a small, hand-curated set (a style, a character, a product). A few hundred deliberately-chosen, hand-captioned images beat any automated pipeline, and aesthetic filtering a set you already curated is pointless. Do still dedup — training a LoRA on 50 near-copies of one image is the fastest way to a memorized, inflexible adapter.
- Your images are already clean and captioned (a licensed, professionally-tagged catalog). You are paying pipeline cost to fix problems you do not have; validate that assumption with a small audit, then skip the stages that come back empty.
- You are prototyping and need a signal fast. Run the cheap stages (resolution floor, exact dedup, a coarse aesthetic cut) and defer recaptioning until you have decided the run is worth the VLM inference bill.
- You are compute-bound, not data-bound. If your bottleneck is GPU-hours, spend them on the aesthetic fine-tune and recaptioning (highest quality-per-token), not on squeezing the last percent out of NSFW recall.

The meta-lesson across all of it: for text-to-image, **data curation is not preprocessing you do before the interesting modeling work — it is the interesting modeling work.** The architecture is increasingly commoditized; the difference between a model that looks good and follows prompts and one that does neither is, overwhelmingly, what you kept, how you captioned it, and whether you deduplicated it.

## Further reading

- Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models* — the SD architecture these data decisions sit under. See also [latent diffusion and Stable Diffusion](/blog/machine-learning/image-generation/latent-diffusion-and-stable-diffusion).
- Betker et al., *Improving Image Generation with Better Captions* (DALL-E 3) — the recaptioning thesis and the 95/5 blend.
- Carlini et al., *Extracting Training Data from Diffusion Models*, and Somepalli et al., *Diffusion Art or Digital Forgery?* — the memorization/replication case for dedup.
- Schuhmann et al., *LAION-5B* and the LAION-Aesthetics predictor — the aesthetic-filtering machinery.
- Podell et al., *SDXL* — aspect-ratio bucketing and resolution/crop conditioning at scale.
- Sibling posts: [deduplication at scale](/blog/machine-learning/training-data/deduplication-at-scale), [image-text pairs at scale](/blog/machine-learning/training-data/image-text-pairs-at-scale), [recaptioning and VLM instruction data](/blog/machine-learning/training-data/recaptioning-and-vlm-instruction-data), [text encoders and prompt conditioning](/blog/machine-learning/image-generation/text-encoders-and-prompt-conditioning), and [the MMDiT modern text-to-image recipe](/blog/machine-learning/image-generation/mmdit-and-the-modern-text-to-image-recipe).
