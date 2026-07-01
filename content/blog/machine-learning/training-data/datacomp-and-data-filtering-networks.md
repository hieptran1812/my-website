---
title: "DataComp and Data Filtering Networks: When Filtering Is the Whole Game"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "Fix the CLIP training recipe and the dataset becomes the only variable — so you can benchmark filtering strategies head to head. This is the practitioner's guide to DataComp, its winning filters (CLIP-score, image clustering, dedup), Data Filtering Networks that beat raw CLIP-score, and the thesis that a smaller well-filtered pool trains a better model than a larger raw one."
tags:
  - training-data
  - datacomp
  - data-filtering-networks
  - dfn
  - clip
  - clip-score
  - image-text-pairs
  - data-curation
  - data-quality
  - multimodal-pretraining
category: "machine-learning"
subcategory: "Training Data"
author: "Hiep Tran"
featured: true
readTime: 31
---

For most of deep learning's history the dataset was the fixed background and the model was the thing you tuned. You downloaded ImageNet once, then spent years sweeping architectures, optimizers, augmentations, and learning-rate schedules against it. Data was a constant; cleverness lived in the model. DataComp flips that experiment on its head, and the flip is the single most useful idea in modern data curation: **freeze the model, the compute, and the number of training steps, and the training set becomes the only independent variable left.** Once data is the only thing that moves, you can finally do the thing curation always lacked — run a controlled experiment where the leaderboard delta is *attributable to the data and nothing else*.

![Classic ML fixes the dataset and sweeps the model; DataComp fixes the model and sweeps the data](/imgs/blogs/datacomp-and-data-filtering-networks-1.webp)

The diagram above is the mental model for this entire post. On the left is the world we grew up in: a fixed dataset, endless architecture and hyperparameter sweeps, data held constant. On the right is DataComp: one fixed architecture, one fixed compute budget, one fixed step count, and *candidate subsets* swapped in and out. The whole discipline of "data-centric AI" is easy to say and hard to practice because you cannot usually isolate the effect of a data change from everything else drifting around it. DataComp makes the isolation mechanical. This post is a tour of that harness, the filtering strategies that win it, the [Data Filtering Networks](https://arxiv.org/abs/2309.17425) that beat the obvious baseline, and the uncomfortable-but-load-bearing result that a smaller, well-filtered pool trains a *better* CLIP than a larger, unfiltered one. It builds directly on [scaling image-text pairs to the billions](/blog/machine-learning/training-data/image-text-pairs-at-scale); if you have not read how those pairs are collected, start there.

## 1. The key insight: fix the recipe, and data is the only variable

Start with the senior rule of thumb: **you cannot benchmark a filtering strategy unless everything downstream of the filter is frozen byte-for-byte.** This sounds obvious and is violated constantly. A team tries a new caption filter, retrains, sees a two-point bump, and ships it — never noticing they also bumped the batch size, or trained 8% longer because the filtered set was smaller and an epoch got cheaper, or changed the random seed. The bump was real; its *cause* was unknown. Data curation is drowning in this exact confound, and it is why so much curation folklore does not replicate.

DataComp's contribution is not a new filter. It is a *protocol*. Every participant is handed the identical candidate pool, the identical model architecture, the identical optimizer and schedule, and — critically — the identical number of samples seen during training. The only degree of freedom is which subset of the pool you train on. When two entries differ on the leaderboard, the difference is the data, full stop. There is no architecture to credit, no learning-rate trick to suspect, no "we trained a bit longer" asterisk.

The subtle part is *fixing the compute correctly*. Compute here is measured in **total samples seen**, not epochs and not unique examples. If you filter a 128-million-pair pool down to 14 million and hold samples-seen constant, your model simply sees those 14 million pairs more times each. This is deliberate: it means a filter is never rewarded merely for making the dataset smaller (which would let you train more epochs) nor punished for it. Two subsets of wildly different sizes are compared at exactly the same FLOP budget. That single design choice is what makes "smaller filtered beats larger unfiltered" a *fair* claim rather than an artifact of unequal training.

| Assumption (data-centric folklore) | Reality under a fixed harness |
| --- | --- |
| "Our filter improved the model by 2 points." | Only true if architecture, schedule, seed, and samples-seen were all held constant. |
| "More data is always safer." | At fixed compute, more low-quality data means fewer passes over the good data — often a regression. |
| "We can compare filters across papers." | Not unless they share a pool, a model, and a compute budget. DataComp exists to supply all three. |
| "Dedup made us better." | Dedup rarely moves the eval much; it mostly recovers compute by deleting redundant gradients. |

The consequence is philosophical and practical at once: **the filter is the model.** Given the DataComp harness, the entire research surface of "how do I build a better CLIP at this scale" collapses to "how do I choose a better subset." You are not doing machine learning anymore in the usual sense; you are doing *data selection*, and the model training is a fixed measuring instrument. That reframing is the whole point, and it connects directly to the general theory of [data selection and pruning](/blog/machine-learning/training-data/data-selection-and-pruning) — DataComp is that theory with the training recipe nailed down so the selection can be scored.

## 2. The DataComp harness: a common CLIP-training benchmark

DataComp ships as four scales, and the scale is just how big the candidate pool and the compute budget are: **small** (12.8M pairs), **medium** (128M), **large** (1.28B), and **xlarge** (12.8B). The pool at each scale is a fixed crawl of image-text pairs (CommonPool), released so every entrant filters the exact same bytes. You pick a scale that matches your compute, filter the pool, train the prescribed CLIP with the prescribed samples-seen, and evaluate zero-shot on a suite of 38 downstream datasets — ImageNet, a battery of ImageNet distribution shifts, VTAB, retrieval, and more. Your leaderboard number is the average zero-shot accuracy across those 38 tasks (with ImageNet zero-shot reported separately because everyone stares at it).

![The DataComp filtering track: pool to filter to a fixed CLIP train to a 38-task eval to a leaderboard rank](/imgs/blogs/datacomp-and-data-filtering-networks-2.webp)

The pipeline above is the loop you run. CommonPool flows into *your filter* — the one thing you control — which produces a training subset. That subset flows into the fixed CLIP training (a ViT-B/32 at medium scale, a ViT-L/14 at large/xlarge), which produces weights, which are evaluated zero-shot across the 38 tasks, which produces the single averaged score that ranks you. Everything except the filter box is frozen by the harness. Note the serpentine: the *only* place a human decision enters is the second box.

A few engineering realities make this harness usable rather than aspirational:

- **The pool is distributed as URLs plus metadata, not images.** At xlarge that is 12.8 billion `(url, caption)` rows. You download what you keep. This is why filters that can decide from *metadata alone* (caption language, caption length, URL patterns) are so valuable: they cut your download bill before you ever fetch a pixel. See [image-text pairs at scale](/blog/machine-learning/training-data/image-text-pairs-at-scale) for the failure modes of the download step itself.
- **Precomputed CLIP features ship with the pool.** DataComp provides OpenAI CLIP ViT-L/14 image and text embeddings for every pair. That is what makes CLIP-score filtering a one-line dot product instead of a multi-day inference job. If your filter needs a *different* model's features, you pay for that inference yourself.
- **The eval suite is sealed.** You do not get to tune your filter on the 38 tasks directly — or rather, you can, and if you do you will overfit the benchmark, which is one of the three failure modes in the troubleshooting section. The honest way to use DataComp is to treat most of the 38 tasks as a held-out set you look at rarely.

The reason this benchmark mattered so much when it landed is that it produced a *released artifact*, not just a paper: DataComp-1B, the winning filtered dataset at xlarge scale, trains a ViT-L/14 to roughly 79% ImageNet zero-shot — competitive with datasets an order of magnitude larger, from a public, reproducible filtering recipe. The benchmark and the dataset are the same object viewed two ways.

## 3. Two tracks: filtering versus BYOD

DataComp runs two tracks, and confusing them is a common mistake. In the **filtering track**, the pool is fixed (CommonPool at your scale) and you may only choose a *subset* of it. In the **BYOD (bring-your-own-data) track**, the pool is *not* fixed — you may assemble training data from any source you like — and the only thing held constant is the compute budget.

![The filtering track fixes the pool and varies the subset; BYOD fixes only the compute and lets any source compete](/imgs/blogs/datacomp-and-data-filtering-networks-3.webp)

The matrix above lays out the difference. The filtering track is the clean scientific instrument: because the pool is shared, a filtering-track result is a pure statement about *selection*. If your subset beats mine, your selection function is better than mine, and nothing else can explain it. The BYOD track is closer to the real industrial question — "assemble the best possible training set under this budget" — but it is a *dirtier* measurement, because a BYOD winner might win on sourcing (they had access to a better corpus) rather than on filtering. Both are useful; they answer different questions.

The senior guidance: **use the filtering track to develop and validate a filter, then move to BYOD to deploy it.** A filter that wins the filtering track has been proven to select well on a shared pool; that is exactly the property you want before you unleash it on your proprietary corpus. Going straight to BYOD tempts you to attribute a sourcing win to your filtering, and you will carry that misattribution into your next dataset. The rest of this post lives mostly in the filtering-track world, because that is where filtering strategies can actually be compared.

## 4. The winning filters

So what actually wins? The DataComp paper and the leaderboard that followed converged on a small set of filters that stack. None of them is exotic; the art is in *combining* them and setting their thresholds.

![The winning DataComp filters: a similarity threshold, an image-cluster match, dedup, text filters, and their intersection](/imgs/blogs/datacomp-and-data-filtering-networks-4.webp)

The taxonomy above is the menu. Let me take each in turn, with the mechanism, the number it moves, and the failure it introduces.

### 4.1 CLIP-score thresholding

**The single biggest lever.** For every pair, compute the cosine similarity between the CLIP image embedding and the CLIP text embedding, and keep the pairs whose similarity is highest. Concretely, the canonical DataComp recipe is "CLIP score (L/14, top 30%)": rank all pairs by OpenAI ViT-L/14 cosine similarity and keep the top 30%.

The intuition is that cosine similarity between image and caption embeddings is a cheap proxy for *alignment* — does the caption actually describe the image? A stock photo captioned "buy now cheap watches free shipping" scores low because the caption is boilerplate unrelated to the pixels; a photo captioned "a golden retriever puppy on a red couch" scores high. You are filtering out the enormous long tail of web pairs where the alt-text is SEO spam, a filename, a navigation label, or a caption for a *different* image on the page. (For why cosine similarity in a contrastively-trained embedding space means "alignment," see [ViT, SigLIP, and DINO explained](/blog/machine-learning/computer-vision/vit-siglip-dino-explained).)

On DataComp medium, CLIP-score filtering at top-30% takes ImageNet zero-shot from 0.176 (no filter) to 0.273 — the largest jump any single filter delivers. It is also nearly free: the L/14 embeddings ship with the pool, so it is one dot product per pair.

```python
import numpy as np

# img_emb, txt_emb: (N, 768) L2-normalized OpenAI ViT-L/14 features shipped
# with CommonPool. cosine similarity is then a single elementwise product-sum.
img_emb = np.load("pool_img_emb.npy")
txt_emb = np.load("pool_txt_emb.npy")

cos = (img_emb * txt_emb).sum(axis=1)          # (N,) cosine in [-1, 1]
tau = np.quantile(cos, 0.70)                    # threshold at the 70th percentile
keep_clip = cos >= tau                          # keep the top 30%
print(f"CLIP-score kept {keep_clip.mean():.1%} at cosine >= {tau:.3f}")
```

The catch, which we return to in troubleshooting: CLIP-score inherits the *biases of the CLIP model that computed it*. OpenAI's CLIP was trained mostly on natural photographs with fluent English captions, so it scores those highest and quietly penalizes charts, documents, diagrams, and non-English pairs. Push the threshold hard and you do not just remove junk — you remove entire *domains*.

### 4.2 Image-based clustering toward an ImageNet-like distribution

**The second lever, and the one that pairs with CLIP-score.** Instead of asking "is this pair aligned?", image-based filtering asks "does this *image* look like the kind of image my downstream evaluation cares about?" The DataComp recipe: embed the ImageNet-1k training images, cluster them (or just use them as reference points), and keep pool images whose embedding is close to that reference distribution.

The intuition is distribution matching, the same idea behind importance resampling in [data selection](/blog/machine-learning/training-data/data-selection-and-pruning): you have a target distribution (ImageNet-like natural images) and you want your training pool to lean toward it. Where CLIP-score is a per-pair *quality* signal, image clustering is a per-image *relevance-to-target* signal. They are complementary, which is why the winning recipe intersects them.

```python
import faiss

# reference = ImageNet-1k training image features (same CLIP encoder), (M, 768).
reference = np.load("imagenet_train_img_emb.npy").astype("float32")
index = faiss.IndexFlatIP(reference.shape[1])   # inner product = cosine (normed)
index.add(reference)

# for each pool image, similarity to its single nearest reference image
sim, _ = index.search(img_emb.astype("float32"), 1)
keep_img = sim[:, 0] >= 0.50                     # near the ImageNet manifold
print(f"image-cluster kept {keep_img.mean():.1%}")
```

On DataComp medium, image-based filtering alone reaches about 0.268 ImageNet zero-shot — comparable to CLIP-score alone. The magic is the intersection.

### 4.3 The combined filter: image-based AND CLIP-score

The DataComp winner at medium and large scale is **image-based ∩ CLIP-score (L/14, 30%)**: keep only pairs that pass *both* the alignment threshold and the ImageNet-relevance threshold. On medium, this reaches 0.297 ImageNet zero-shot — a full 12 points over the unfiltered baseline, and 2–3 points over either filter alone. The intersection works because the two filters remove *different* garbage: CLIP-score removes misaligned pairs regardless of domain; image clustering removes off-target domains regardless of alignment. A well-aligned photo of a circuit-board schematic passes CLIP-score but fails image clustering (it is not ImageNet-like); a natural photo with spam alt-text passes image clustering but fails CLIP-score. Only pairs that are both aligned *and* on-target survive.

> The lesson generalizes far past CLIP: **the best filter is usually an intersection of orthogonal cheap filters, not a single expensive one.** Each cheap filter removes a different failure mode; intersection compounds them.

### 4.4 Deduplication and text filters

Two more filters round out the stack, both cheaper and lower-impact than the two above.

**Deduplication.** Web crawls are riddled with near-duplicate images — the same stock photo across a thousand sites, the same meme with different captions. Semantic deduplication ([SemDeDup](/blog/machine-learning/training-data/deduplication-at-scale)) embeds images and removes pairs whose image is a near-duplicate of another survivor. Dedup rarely moves ImageNet zero-shot by more than a fraction of a point, but it is not about accuracy — it is about *compute efficiency*. At a fixed samples-seen budget, deleting 12% redundant pairs means those FLOPs go to *unique* data instead of re-showing the model the same stock photo. Dedup buys you effective data diversity per FLOP.

**Text and language filters.** The cheapest filters of all, because they read metadata only: keep pairs whose caption is English (via a fast language classifier like fastText or cld3), whose caption has at least a couple of real words, and whose image meets a minimum resolution. DataComp calls the bundle of these "basic filtering," and it alone takes ImageNet from 0.176 to 0.226 — a five-point gain for a filter that never fetches an image. English filtering also happens to be where you *narrow domain coverage the most*, because it deletes every non-English pair by construction. That is a feature for an English ImageNet eval and a bug for a multilingual model, which is the tension the troubleshooting section is built around.

| Filter | What it optimizes | Reads pixels? | Typical keep-rate | ImageNet Δ (medium) | Cost |
| --- | --- | --- | --- | --- | --- |
| Basic + English | valid, English, real caption | No | ~47% | +0.050 | trivial (metadata) |
| CLIP-score top 30% | image-text alignment | Yes (features shipped) | 30% of pool | +0.047 over basic | one dot product |
| Image clustering | relevance to ImageNet | Yes (features shipped) | ~65% of survivors | +0.024 | one ANN query |
| Semantic dedup | redundancy removal | Yes | ~88% of survivors | ~+0.00 (efficiency) | one ANN pass |
| **image ∩ CLIP-score** | **alignment AND relevance** | **Yes** | **~11% of pool** | **+0.121 total** | **stacked** |

## 5. A worked scenario: stacking the filter stack

Let me make this concrete with numbers you can trace end to end. Take DataComp medium: a 128-million-pair pool. We will apply the winning stack — basic/English, then CLIP-score, then image clustering, then dedup — and watch the surviving pool shrink at each stage, then ask *which single filter moved the eval most*.

<figure class="blog-anim">
<svg viewBox="0 0 720 390" role="img" aria-label="Cumulative keep-rate shrinking across a filter stack: the surviving pool drops from 100 percent to 11 percent as basic, CLIP-score, cluster, and dedup filters stack" style="width:100%;height:auto;max-width:820px">
<style>
.kr-back{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.kr-keep{fill:var(--accent,#6366f1)}
.kr-val{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.kr-stg{font:500 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.kr-sweep{fill:var(--accent,#6366f1);opacity:.16}
@keyframes kr-move{0%{transform:translateX(0)}100%{transform:translateX(520px)}}
.kr-anim{animation:kr-move 11s ease-in-out infinite alternate}
@media (prefers-reduced-motion:reduce){.kr-anim{animation:none}}
</style>
<line x1="46" y1="312" x2="674" y2="312" stroke="var(--border,#d1d5db)" stroke-width="1.5"/>
<rect class="kr-back" x="60"  y="72"  width="80" height="240" rx="4"/>
<rect class="kr-back" x="190" y="72"  width="80" height="240" rx="4"/>
<rect class="kr-back" x="320" y="72"  width="80" height="240" rx="4"/>
<rect class="kr-back" x="450" y="72"  width="80" height="240" rx="4"/>
<rect class="kr-back" x="580" y="72"  width="80" height="240" rx="4"/>
<rect class="kr-keep" x="60"  y="72"  width="80" height="240" rx="4"/>
<rect class="kr-keep" x="190" y="199" width="80" height="113" rx="4"/>
<rect class="kr-keep" x="320" y="264" width="80" height="48"  rx="4"/>
<rect class="kr-keep" x="450" y="281" width="80" height="31"  rx="4"/>
<rect class="kr-keep" x="580" y="285" width="80" height="27"  rx="4"/>
<rect class="kr-sweep kr-anim" x="54" y="66" width="92" height="252" rx="6"/>
<text class="kr-val" x="100" y="58">100%</text>
<text class="kr-val" x="230" y="185">47%</text>
<text class="kr-val" x="360" y="250">20%</text>
<text class="kr-val" x="490" y="267">13%</text>
<text class="kr-val" x="620" y="271">11%</text>
<text class="kr-stg" x="100" y="334">raw pool</text>
<text class="kr-stg" x="230" y="334">+ basic</text>
<text class="kr-stg" x="360" y="334">+ CLIP-score</text>
<text class="kr-stg" x="490" y="334">+ cluster</text>
<text class="kr-stg" x="620" y="334">+ dedup</text>
</svg>
<figcaption>The surviving pool as each filter stacks: basic filtering keeps 47 percent, CLIP-score cuts to 20 percent, image clustering to 13 percent, and dedup to a final 11 percent. The sweep marks the filter being applied; the accent bars are what survives.</figcaption>
</figure>

The animation traces the shrink: 128M pairs, then 47% survive basic filtering (60.2M), then CLIP-score cuts to a cumulative 20% (25.3M), then image clustering to 13% (16.4M), then dedup to a final 11.3% (14.5M). You throw away nearly nine of every ten pairs. Here is the stack as code that reports the keep-rate at each stage — the exact instrumentation you want when developing a filter, because a filter that keeps 0.1% or 90% is almost always a bug.

```python
import numpy as np
import faiss

def keep_rate(mask, name):
    r = mask.mean()
    print(f"{name:<20} keep {r:6.1%}   ({mask.sum():>11,} / {len(mask):,})")
    return r

# Precomputed, L2-normalized OpenAI ViT-L/14 features for the whole pool.
img_emb = np.load("pool_img_emb.npy").astype("float32")   # (128_000_000, 768)
txt_emb = np.load("pool_txt_emb.npy").astype("float32")
basic_ok = np.load("pool_basic_ok.npy")                    # bool: English + len + size
N = len(img_emb)
alive = np.ones(N, dtype=bool)

# Stage 1 — basic / language / size filter (metadata only, no pixels).
alive &= basic_ok
keep_rate(alive, "1. basic+English")            # ~47%

# Stage 2 — CLIP-score threshold: keep top 30% of the ORIGINAL pool by cosine.
cos = (img_emb * txt_emb).sum(axis=1)
alive &= cos >= np.quantile(cos, 0.70)
keep_rate(alive, "2. + CLIP-score")             # cumulative ~20%

# Stage 3 — image-based clustering: near the ImageNet reference manifold.
reference = np.load("imagenet_train_img_emb.npy").astype("float32")
ref_index = faiss.IndexFlatIP(reference.shape[1])
ref_index.add(reference)
sim, _ = ref_index.search(img_emb, 1)
alive &= sim[:, 0] >= 0.50
keep_rate(alive, "3. + image cluster")          # cumulative ~13%

# Stage 4 — semantic dedup: drop near-duplicate images among survivors.
idx = np.where(alive)[0]
dd = faiss.IndexFlatIP(img_emb.shape[1])
dd.add(img_emb[idx])
D, I = dd.search(img_emb[idx], 2)                # [self, nearest-neighbor]
drop = np.zeros(len(idx), dtype=bool)
seen = set()
for i, (nn, d) in enumerate(zip(I[:, 1], D[:, 1])):
    if d >= 0.50 and nn in seen:                # too close to a kept survivor
        drop[i] = True
    else:
        seen.add(i)
alive[idx[drop]] = False
keep_rate(alive, "4. + semantic dedup")         # cumulative ~11%
```

Now the payoff question: **which single filter earned its keep-rate?** Here is the eval attribution, using DataComp's reported medium-scale ImageNet zero-shot for each cumulative stage:

| Stage | Cumulative keep | ImageNet zero-shot | Δ from previous |
| --- | --- | --- | --- |
| No filter | 100% (128M) | 0.176 | — |
| + basic + English | 47% (60.2M) | 0.226 | +0.050 |
| + CLIP-score (top 30%) | 20% (25.3M) | 0.273 | +0.047 |
| + image cluster | 13% (16.4M) | 0.297 | +0.024 |
| + semantic dedup | 11% (14.5M) | ~0.298 | ~+0.001 |

Read this table like a principal engineer, not a leaderboard chaser. Two filters — basic and CLIP-score — buy you almost 0.10 of the 0.12 total gain. Image clustering adds a real but smaller 0.024. Dedup adds essentially nothing *to ImageNet accuracy* — but it deleted ~12% of the pool as redundant, so at fixed compute the model spends those FLOPs on unique pairs, which shows up as better *data efficiency* and modestly better retrieval, not as an ImageNet number. If a teammate proposes an expensive new dedup pipeline "to boost accuracy," this table is your answer: dedup is a compute-efficiency play, and CLIP-score is where the accuracy lives.

## 6. Data Filtering Networks: train a network to filter

Here is the tension that CLIP-score filtering never resolves. You are using a *frozen, general-purpose* CLIP model to score fitness for training. But that CLIP model was itself trained on some distribution, with its own blind spots, and you are now propagating those blind spots into every dataset you filter. Worse, the model you are trying to *build* may want data that the frozen scorer undervalues. The scorer is a fixed oracle whose taste you cannot change.

Data Filtering Networks (DFN), from Fang et al. at Apple, take the obvious-in-hindsight next step: **stop reusing a general model as your filter, and train a network whose entire job is to predict "is this pair good training data?"** A DFN is a small CLIP model, but it is trained specifically on high-quality curated pairs so that its notion of alignment is *sharper and better-calibrated for filtering* than an off-the-shelf CLIP's. You then run the DFN over the raw pool, keep the top-scoring pairs, and train your real (large) model on that subset.

![Both a frozen CLIP-score threshold and a trained DFN filter the same pool and feed the same recipe; the DFN subset wins](/imgs/blogs/datacomp-and-data-filtering-networks-5.webp)

The graph above is the head-to-head. Both filters see the same raw pool and feed the same fixed training recipe, so the comparison is clean in the DataComp sense. The top path is the baseline: a frozen CLIP produces a cosine score, you threshold it, you get subset A. The bottom path is the DFN: you fit a small filtering network on a curated seed of clean pairs, run it over the pool, keep its top-k, and get subset B. Train the identical target CLIP on each; DFN's subset B wins the eval.

The results are striking. At DataComp xlarge, a learned DFN beats the strongest hand-built filter, and the DFN-filtered datasets Apple released (DFN-2B, DFN-5B) train models to accuracies that were state-of-the-art at their compute: a ViT-L/14 on DFN-2B reaches about 81% ImageNet zero-shot, and a ViT-H/14 on DFN-5B reaches 84.4%. The most counterintuitive finding in the paper: **the DFN does not need to be a strong model itself.** A modestly-sized filtering network — one that is a mediocre CLIP by zero-shot standards — produces datasets that train models *far better than the DFN*. The filter's job is discrimination ("keep or drop"), which is easier than the downstream task, so a small network suffices. You are distilling *curation taste*, not *task capability*.

Here is the DFN idea in code. The real DFN fine-tunes a full CLIP on curated data; the version below trains a tiny fitness head on top of frozen CLIP features, which captures the same principle — a *learned* filter, trained on your definition of "good" — at roughly 1% of the cost. It is the right first thing to build before committing to full DFN training.

```python
import torch
import torch.nn as nn

class DFNHead(nn.Module):
    """A learned fitness scorer on frozen CLIP features.
    Real DFN fine-tunes a full CLIP; this head is the same idea, cheaper."""
    def __init__(self, d=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d, 512), nn.GELU(),
            nn.Linear(512, 1),
        )

    def forward(self, img_f, txt_f):
        return self.net(torch.cat([img_f, txt_f], dim=-1)).squeeze(-1)

dfn = DFNHead().cuda()
opt = torch.optim.AdamW(dfn.parameters(), lr=1e-3, weight_decay=1e-2)
bce = nn.BCEWithLogitsLoss()

# Positives: pairs from a trusted curated set (e.g. cleaned Conceptual Captions).
# Negatives: random pool pairs AND (image, shuffled-caption) mismatches — this
# teaches the head that alignment, not just "looks like a caption," is the signal.
for img_f, txt_f, y in curated_vs_pool_loader:          # y = 1 good, 0 bad
    logit = dfn(img_f.cuda(), txt_f.cuda())
    loss = bce(logit, y.cuda().float())
    opt.zero_grad(); loss.backward(); opt.step()

@torch.no_grad()
def dfn_keep(img_emb, txt_emb, frac=0.15):
    s = dfn(torch.from_numpy(img_emb).cuda(),
            torch.from_numpy(txt_emb).cuda())
    tau = torch.quantile(s, 1.0 - frac)                 # keep the top `frac`
    return (s >= tau).cpu().numpy()

keep_dfn = dfn_keep(img_emb, txt_emb, frac=0.15)
print(f"DFN kept {keep_dfn.mean():.1%} of the pool")
```

The design choice that matters most is the *negatives*. If your negatives are only random pool pairs, the head learns "does this look like a real image-text pair," which frozen CLIP-score already does. Add **mismatched negatives** — real images paired with a *different* pair's caption — and the head is forced to learn alignment specifically, which is where it can exceed a generic scorer. The second choice that matters is the *curated positive set*: the DFN's taste is exactly the distribution of your positives, so if your positives are all English natural photos, congratulations, you have rebuilt CLIP-score's bias with extra steps. Curate positives that span the domains you actually care about.

## 7. Quality beats scale: the smaller filtered pool wins

We can now state the thesis this whole post has been circling, and it is worth stating baldly because it contradicts a decade of "more data is better" instinct: **at a fixed compute budget, a smaller pool filtered for quality trains a better model than a larger unfiltered pool.**

![A filtered 11% subset nearly doubles ImageNet zero-shot versus the full unfiltered pool at the same compute](/imgs/blogs/datacomp-and-data-filtering-networks-6.webp)

The before-after above is the medium-scale result from our worked scenario, stated as an experiment. Train on everything — 128M unfiltered pairs — at the harness's fixed compute, and you get 0.176 ImageNet zero-shot. Filter to the best 11% (14.5M pairs) and train at the *same* compute, and you get 0.297. Nearly double the accuracy, from *less than one-eighth the data*. The unfiltered run wastes most of its FLOPs learning from misaligned, redundant, off-target junk; the filtered run spends every FLOP on aligned, on-target, unique pairs, seen more times each.

Why does this happen, mechanistically? Three reinforcing reasons:

1. **Noise is not neutral; it is actively harmful.** A misaligned pair does not just fail to teach — it teaches the *wrong* association, pulling the image and text encoders toward a spurious alignment. Contrastive training on a caption that does not describe its image is a gradient in the wrong direction. Removing it is worth more than one clean pair.
2. **Fixed compute means data quality trades against data quantity.** At a fixed samples-seen budget, every junk pair you keep is a clean pair you *do not get to show again*. The relevant quantity is not "how many pairs" but "how many high-quality gradient steps," and filtering raises the fraction of your budget spent on those.
3. **Repetition of good data beats single-exposure to bad data.** Seeing 14.5M excellent pairs four times teaches more than seeing 128M mostly-bad pairs once, up to the point where the good pool starts to over-memorize — which, at these scales and epoch counts, you are nowhere near.

This is the same phenomenon the [data selection scaling law](/blog/machine-learning/training-data/data-selection-and-pruning) formalizes for LLMs: with a good difficulty or quality metric, you can beat the naive power-law scaling by pruning. DataComp is the vision-language proof of it, made rigorous by the fixed harness. It is also why [measuring data quality](/blog/machine-learning/training-data/measuring-data-quality) honestly is the prerequisite skill — the entire "quality beats scale" claim is only as trustworthy as your ability to *measure* quality without fooling yourself, and the next section is a catalog of exactly those self-deceptions.

## 8. Troubleshooting: when a winning filter is secretly losing

A filter that climbs the leaderboard can still be a bad filter, because the leaderboard is a proxy and every proxy can be gamed. These are the three failure modes I have seen bite hardest, each as symptom, detection, and fix.

![An aggressive CLIP-score threshold keeps natural photos and quietly deletes charts, documents, and non-English pairs](/imgs/blogs/datacomp-and-data-filtering-networks-7.webp)

### 8.1 The filter wins the benchmark but narrows domain coverage

**Symptom.** Your aggressive CLIP-score threshold pushes ImageNet zero-shot up nicely, and you ship it. Weeks later, the model is deployed on document images, charts, screenshots, or a non-English market — and it is dramatically worse there than the ImageNet number promised. The grid above shows why: an aggressive similarity threshold keeps 41% of natural photos but only 2–5% of charts, documents, diagrams, and non-English pairs. The filter did not remove *junk*; it removed *domains*, because the frozen CLIP scoring it never learned to value those domains in the first place.

**Detection.** Never evaluate a filter on ImageNet alone. Break your eval into per-domain slices — a document-VQA set, a chart-QA set, a multilingual retrieval set — and watch the *worst* slice, not the average. Independently, instrument your filter with per-domain keep-rates (exactly the `keep_rate` calls from the worked scenario, but stratified by a cheap domain classifier). If any domain's keep-rate falls off a cliff while ImageNet rises, you are trading coverage for a single-benchmark win.

**Fix.** Two options. The gentle one: relax the threshold, or use a *union* of filters (keep pairs that pass CLIP-score OR image-clustering) so domains the CLIP scorer dislikes get a second path in. The rigorous one: apply **per-cluster keep quotas** — cluster the pool, then keep the top fraction *within each cluster* rather than globally, so every domain contributes its best pairs instead of being globally out-competed by photos. This is stratified selection, and it is the single most reliable cure for coverage collapse.

### 8.2 Overfitting the filter to the eval suite

**Symptom.** Your filter's leaderboard score keeps climbing across iterations, but when you finally test on tasks you *never looked at*, the gains evaporate. You have been tuning the filter's thresholds and reference sets against the eval suite, and the filter has quietly learned the eval rather than learned quality. This is [Goodhart's law](/blog/machine-learning/training-data/measuring-data-quality) wearing a filtering costume: the moment the eval score became your optimization target, it stopped being a faithful measure.

**Detection.** Seal a portion of the eval you *do not tune against* and check it rarely — treat the 38-task suite as train/held-out, not all-train. A widening gap between your tuned tasks and your sealed tasks is the tell. A second, sharper detection: audit whether your filter *uses eval-derived signal*. Image-clustering toward ImageNet is legitimate for an ImageNet-heavy eval, but if you find yourself adding "keep pairs whose caption contains one of the 38 tasks' class names," you have crossed from selection into leakage.

**Fix.** Freeze the filter's hyperparameters against a small development eval and validate on the sealed set exactly once before shipping. Prefer filters whose signal is *intrinsic* (alignment, redundancy) over filters whose signal is *eval-derived* (target-class matching), because intrinsic filters cannot overfit a benchmark they never see. If you must use a target distribution, make it broad (all of ImageNet's images, not the eval's exact classes) so the filter generalizes past the specific benchmark.

### 8.3 Dedup against the eval — contamination hiding as quality

**Symptom.** Your filtered dataset produces a suspiciously high eval score — a jump larger than any filter should plausibly deliver. When you look closely, some of the "training" images in your pool *are* the evaluation images (or near-duplicates of them), scraped from the same corners of the web. Your model is not generalizing; it is recognizing. This is the most dangerous failure because it inflates *exactly the number you trust*, and it silently invalidates every comparison built on that eval.

**Detection.** Run near-duplicate detection *between your pool and every eval set*, not just within the pool. Embed the eval images, embed the pool images, and flag any pool image whose nearest eval neighbor exceeds a duplicate threshold. A nonzero contamination rate against a benchmark means every result on that benchmark is suspect. This is the vision-language version of the text-side problem covered in depth in [decontamination and benchmark leakage](/blog/machine-learning/training-data/decontamination-and-benchmark-leakage) — same disease, image embeddings instead of n-gram overlap.

**Fix.** Decontaminate the *pool* against all eval image sets **before** filtering, not after — remove pool images that near-duplicate any eval image, then run your quality filters on what remains. Do this once, up front, as a mandatory pipeline stage; contamination found after training means retraining. And report the contamination rate you removed, so downstream users of your dataset know the eval is clean.

## Case studies from the benchmark

### 1. DataComp: the benchmark that made filtering measurable

Before DataComp (Gadre et al., 2023), the state of image-text curation was a pile of incomparable claims. LAION, Conceptual Captions, ALIGN's internal data, and a dozen proprietary corpora each reported numbers on different models at different scales, and no one could say whether a filter or a bigger backbone drove a given result. DataComp's fix was social as much as technical: fix the pool, fix the model, fix the compute, and turn curation into a *competition on the data*. The immediate payoff was that the winning recipe — image-based ∩ CLIP-score (L/14, 30%) — was published, reproducible, and *simple*, which killed a lot of folklore about needing secret sauce. The lasting payoff was DataComp-1B: a public, filtered 1.4B-pair dataset that trains a ViT-L/14 to ~79% ImageNet zero-shot, competitive with OpenAI's private WIT and with LAION-2B at a fraction of the size. The lesson for any team: if you cannot compare your filters under a fixed recipe, you are not measuring filtering — you are measuring luck.

### 2. Data Filtering Networks: the learned filter beats the frozen one

Fang et al. (2023) asked why the whole field was using a *frozen, general* CLIP as its quality oracle and answered with a dedicated one. The headline is that a trained DFN produces datasets that beat the best DataComp hand-built filters at xlarge scale — but the deeper lesson is the *decoupling* the paper demonstrated: filtering quality and model quality are different axes. A small, unremarkable CLIP, trained specifically on curated pairs (with mismatched negatives so it learns alignment, not just plausibility), makes a better *filter* than a much larger general CLIP, and the datasets it selects train ViT-L/14 and ViT-H/14 models to 81% and 84.4% ImageNet zero-shot respectively. The practical takeaway is liberating: you do not need a frontier model to curate frontier data. You need a cheap discriminator trained on the right positives, and the taste it distills is what ends up in the dataset. The trap the paper implicitly warns about — and the reason curated-positive selection is load-bearing — is that the DFN inherits *your* biases: its taste is exactly your positive set's distribution.

### 3. Small-filtered beats large-unfiltered: the result that reframed scale

The single most-cited number out of DataComp is not any absolute accuracy — it is the *ratio*. An 11%-of-pool filtered subset nearly doubling ImageNet zero-shot over the full unfiltered pool, at identical compute, is the cleanest refutation available of "just add more data." It landed at the same moment the LLM world was internalizing the same lesson from FineWeb and from pruning scaling laws: past a point, the marginal web document is *negative* value at fixed compute, because it displaces a better document. What made the vision-language version so persuasive was the harness — nobody could wave it away as an artifact of unequal training, because the compute was pinned. The result quietly rewired how serious teams budget: the question shifted from "how much data can we collect" to "how good a filter can we build," because at fixed compute the filter is the lever and the raw pool size is nearly irrelevant above a floor.

### 4. The English-only coverage cliff

A recurring production incident, not from one paper but from the collective bruises of teams shipping CLIP-derived models: a filter tuned to maximize ImageNet zero-shot ships, and the multilingual or document-heavy product built on top of it underperforms catastrophically. The root cause is always the same — "basic filtering" deletes every non-English pair by construction, and CLIP-score deprioritizes documents and charts, so the filter that topped an English natural-image benchmark stripped exactly the domains the product needed. The fix that works in practice is stratified keep quotas (Section 8.1): cluster first, keep the best *within* each domain, and accept a slightly lower ImageNet number in exchange for a model that does not fall off a cliff outside the benchmark's distribution. The teams that learned this the hard way now treat "what does this filter delete?" as a first-class review question, right next to "what does it keep?"

## When to make filtering the whole game — and when not to

**Reach for the DataComp-style, filter-is-everything approach when:**

- You are training a **contrastive vision-language model** (CLIP/SigLIP-family) from web-scraped pairs — this is exactly the regime DataComp and DFN were built for.
- Your **compute is fixed** and your raw pool is far larger than your budget can consume. The whole argument depends on compute being the binding constraint; that is when filtering is the lever.
- You can **hold the training recipe constant** long enough to attribute deltas to data. Without that, you are back to measuring luck.
- You have, or can build, a **cheap quality signal** (shipped CLIP features, or a small DFN you can train). The filter must be cheaper than the training it feeds, or you have not saved anything.

**Skip it, or apply it cautiously, when:**

- Your **pool is already scarce** — if you have 50k pairs total, filtering to 11% is throwing away signal you cannot spare. Filtering-as-the-game assumes abundance.
- You need **broad domain or multilingual coverage** and your only quality signal is an English-natural-image CLIP. The filter's biases become your model's blind spots; use stratified quotas or a DFN trained on positives that span your domains.
- The **downstream task is narrow and known** — if you are only ever doing document retrieval, filter *toward documents*, not toward a general ImageNet-like distribution. The winning DataComp recipe is tuned for a broad 38-task eval; your target may not be broad.
- You **cannot decontaminate against your eval.** If you cannot guarantee the pool is clean of eval images, your filter's "wins" may be contamination, and no amount of clever selection fixes a poisoned measurement.

The through-line of this entire post is a single reframing: once the training recipe is fixed, machine learning on web-scale pairs *is* data selection, and the leaderboard is a filter benchmark wearing a model's clothes. CLIP-score gets you most of the way; intersecting it with image-clustering gets you the DataComp win; training a dedicated Data Filtering Network gets you past that; and the reward for all of it is the counterintuitive prize that a smaller, sharper pool beats a larger, dirtier one at the same cost. Filter like the model depends on it — because, with the recipe frozen, it entirely does.

## Further reading

- [Scaling image-text pairs to the billions](/blog/machine-learning/training-data/image-text-pairs-at-scale) — how CommonPool-style corpora are collected before you filter them.
- [Data selection and pruning](/blog/machine-learning/training-data/data-selection-and-pruning) — the general theory (DSIR, influence, coreset, the pruning scaling law) that DataComp instantiates for vision-language.
- [Measuring data quality](/blog/machine-learning/training-data/measuring-data-quality) — the ablation loop and Goodhart traps that make any "quality beats scale" claim trustworthy or not.
- [Deduplication at scale](/blog/machine-learning/training-data/deduplication-at-scale) — SemDeDup and the near-duplicate machinery the dedup stage relies on.
- [Decontamination and benchmark leakage](/blog/machine-learning/training-data/decontamination-and-benchmark-leakage) — why you decontaminate the pool against the eval before, not after, filtering.
- [ViT, SigLIP, and DINO explained](/blog/machine-learning/computer-vision/vit-siglip-dino-explained) — what the CLIP embedding space is and why cosine similarity in it means alignment.
- DataComp (Gadre et al., 2023) and Data Filtering Networks (Fang et al., 2023) — the two primary sources this post is built on.
