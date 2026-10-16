---
title: "Finding Distributed Object-Centric Properties in Self-Supervised Transformers"
publishDate: "2026-10-16"
date: "2026-10-16"
description: "Why the [CLS] attention map is the wrong place to look for objects in DINO, and how Object-DINO finds them instead in patch-to-patch query, key and value similarity spread across many heads and layers, with no training."
category: "paper-reading"
subcategory: "Computer Vision"
tags: ["interpretability-analysis", "cvpr-2026", "cvpr-2026-highlight", "object-dino", "dino", "vision-transformer", "attention-heads", "unsupervised-object-discovery", "object-hallucination", "self-supervised-learning"]
author: "Hiep Tran"
featured: true
readTime: 20
image: "/imgs/blogs/finding-distributed-object-centric-properties-in-self-supervised-transformers-fig1.webp"
excerpt: "DINO's famous [CLS] attention maps are a lossy summary. The object lives in how patches resemble each other, through queries, keys and values, in about two dozen heads scattered over the last four layers. Object-DINO finds those heads by clustering, with zero training."
paper:
  title: "Finding Distributed Object-Centric Properties in Self-Supervised Transformers"
  authors: "Samyak Rawlekar, Amitabh Swain, Yujun Cai, Yiwei Wang, Ming-Hsuan Yang, Narendra Ahuja"
  venue: "CVPR 2026 (highlight)"
  url: "https://arxiv.org/abs/2603.26127"
---

> [!tldr]
> **Claim.** In self-supervised ViTs like DINO, object information is not best read from the final-layer [CLS] attention map. It sits in patch-to-patch similarity computed from all three attention components (query, key, value), and it is spread over many heads in several late layers, not just the last one.
> **Method.** Object-DINO builds a similarity map per head from $q$, $k$ and $v$, averages them, clusters all heads with k-means, and keeps the cluster that holds the most final-layer heads. No training, no labels.
> **Most surprising result.** Swapping TokenCut's "all final-layer keys" for the selected heads lifts CorLoc by up to 12.4 points (DINOv2, VOC12: 18.3 to 30.7), and most of the gain comes from heads in layers *before* the last one.
> **Where it fails.** It only separates foreground from background (no instances), its baselines on DINOv2/v3 are far below classic DINO v1 TokenCut numbers, and the hallucination-decoding formula is stated two incompatible ways in the paper.

DINO made a famous picture: feed an image to a self-supervised ViT, look at where the [CLS] token attends in the last layer, and the object lights up. Nobody labeled anything. That picture launched a small industry of unsupervised object discovery methods.

This paper asks an uncomfortable question: if [CLS] is trained to summarize the *whole image*, why would its attention be the cleanest view of an *object*? The authors (UIUC, Queensland, UC Merced) argue it is not, and show where the cleaner signal lives.

![Figure 1 from Rawlekar et al. (2026): for each input image, inter-patch similarity maps from queries, keys and values, and their ensemble, which separates foreground from background](/imgs/blogs/finding-distributed-object-centric-properties-in-self-supervised-transformers-fig1.webp)

The figure above is the whole thesis: each similarity map sees the object, each makes a different mistake, and their average is cleaner than any one. The rest of the paper turns that into a method that needs no human picking heads by eye.

## 1. The problem: the [CLS] map is a summary, not a segmenter

DINO is trained with a global objective: a student network must match a teacher's output for different crops of the same image. The [CLS] token carries that output. So [CLS] has to encode everything that identifies the image, including texture, edges and context, not just the object.

The visible symptom is a [CLS] map with spurious background hot spots and parts of the object missing.

![Schematic before/after: a 10 by 10 patch grid where the [CLS] attention map misses the dog's leg and lights up background cells, while the patch-affinity map keeps the object's full shape](/imgs/blogs/finding-distributed-object-centric-properties-in-self-supervised-transformers-1.webp)

Think of a headline writer versus a photo editor. The headline ([CLS]) must mention the object but also the weather and the crowd; the photo editor (patch-to-patch similarity) just groups pixels that look alike, which is what cutting out an object needs.

Skipping [CLS] is not new: TokenCut and LOST already treat patches as graph nodes. But TokenCut uses only the **keys** of the **final layer**, and this paper's two findings attack both choices.

| Prior choice | What it assumes | What this paper finds |
| --- | --- | --- |
| [CLS] attention (DINO-seg) | the image summary points at the object | the summary dilutes the object with context |
| Final-layer keys only (TokenCut) | $k$ carries the object; last layer is enough | $q$ and $v$ carry complementary signal; earlier layers hold object heads too |
| Average all final-layer heads | every head helps | 4 of 12 final-layer heads add noise |

## 2. Finding 1: queries, keys and values all encode the object

### The mechanism

Inside every attention head, each patch token is projected three ways. The paper's roles: the **query** $q$ is what a patch is looking for, the **key** $k$ is what a patch offers to others, and the **value** $v$ is the content that gets passed along when a patch is attended to.

Normal attention compares queries against keys. Object-DINO instead compares each component **with itself**: queries with queries, keys with keys, values with values. Two patches with similar queries are "asking similar questions", which tends to happen when they belong to the same object.

![Data-flow for one head with tensors drawn to shape: N by D tokens become N by 64 q, k and v, each turns into an N by N similarity heatmap, and the three average into a clean A_ens that is flattened into one fingerprint vector](/imgs/blogs/finding-distributed-object-centric-properties-in-self-supervised-transformers-2.webp)

### The math

For layer $\ell$, head $h$ and component $r \in \{q, k, v\}$, stack the patch vectors into a matrix $r^{\ell,h} \in \mathbb{R}^{N \times d}$, where $N$ is the number of patches and $d$ the head dimension. Normalize each row to unit length to get $\tilde r^{\ell,h}$. The paper's Eq. 1 is

$$
A_r^{\ell,h} = \operatorname{softmax}\left( \frac{\tilde r^{\ell,h} (\tilde r^{\ell,h})^\top}{\tau} \right), \qquad A_r^{\ell,h} \in \mathbb{R}^{N \times N}
$$

Entry $(i, j)$ of the inner product is the cosine similarity between patch $i$ and patch $j$, so it lies in $[-1, 1]$. The softmax runs over each row, and $\tau$ is a temperature; the paper uses $\tau = 60$. Then the three views are blended (Eq. 2):

$$
A_{ens}^{\ell,h} = w_q A_q^{\ell,h} + w_k A_k^{\ell,h} + w_v A_v^{\ell,h}, \qquad w_q = w_k = w_v = 0.33
$$

Row $i$ of $A_{ens}^{\ell,h}$, reshaped back onto the patch grid, is a map of "which patches resemble patch $i$" in that head. The maps in Figure 1 are of this kind, inverted so objects look bright.

### A worked micro-example

Take four patches in a 2D head space: two on a dog, $p_1 = (1, 0.1)$ and $p_2 = (0.9, 0.2)$, and two on grass, $p_3 = (0.1, 1)$ and $p_4 = (0.2, 0.9)$. After normalization, $\cos(p_1, p_2) \approx 0.99$ while $\cos(p_1, p_3) \approx 0.20$. The similarity matrix has a bright 2 by 2 block for the dog and another for the grass: that block structure *is* the segmentation.

Each component errs differently: in Figure 1, $A_q$ and $A_v$ sometimes merge background into the foreground and $A_k$ misses part of a person. Averaging differently noisy views cancels some of each error.

### Why the ensemble, measured

Using each component alone as the clustering feature gives the same CorLoc ranking on all three benchmarks: $A_q \lt A_v \lt A_k \lt A_{ens}$ (their Fig. 6). Keys are the best single view, so TokenCut's choice was sensible, just incomplete.

The supplementary weight ablation agrees: CorLoc rises as the weights move from key-heavy to uniform.

| $w_q$ | $w_k$ | $w_v$ | CorLoc |
| --- | --- | --- | --- |
| 0.1 | 0.7 | 0.2 | 17.9 |
| 0.2 | 0.5 | 0.3 | 19.1 |
| 0.3 | 0.4 | 0.3 | 19.4 |
| 0.33 | 0.33 | 0.33 | 19.8 |

One caveat on that table: the paper labels it VOC2007, but 19.8 is also the exact COCO20k TokenCut baseline in its Table 1, and the VOC07 numbers there are around 26 to 31. I suspect a dataset label slip, but I cannot tell which number is wrong from the paper alone.

### The temperature is doing something unexpected

The temperature deserves a closer look, because $\tau = 60$ is unusual. The inputs to the softmax are cosines, so after dividing by 60 every logit lies in $[-0.017, 0.017]$. A softmax over values that close together is almost flat.

![Plotted curves of softmax weight relative to uniform versus cosine similarity: tau 0.1 is a steep exponential, tau 1 a gentle one, and tau 60 is nearly a straight line hugging 1.0](/imgs/blogs/finding-distributed-object-centric-properties-in-self-supervised-transformers-3.webp)

With $e^{x} \approx 1 + x$ for small $x$, and $s_{ij}$ the cosine between patches $i$ and $j$:

$$
A_{ij} = \frac{e^{s_{ij}/\tau}}{\sum_{m=1}^{N} e^{s_{im}/\tau}} \approx \frac{1}{N}\left(1 + \frac{s_{ij} - \bar s_i}{\tau}\right), \qquad \bar s_i = \frac{1}{N}\sum_{m} s_{im}
$$

For a 14 by 14 patch grid ($N = 196$), the most similar patch gets a weight of about $e^{1/60} \approx 1.017$ times uniform and the most dissimilar about $e^{-1/60} \approx 0.983$ times. So at the paper's chosen temperature, each $A_r$ is essentially uniform plus a centered, scaled copy of the cosine-similarity matrix. That reading is mine, not the authors'.

So the method is really about cosine similarity of $q$, $k$, $v$. It also explains the supplementary ablation: $\tau \in (10, 100)$ works best because the map stays linear in similarity there, while a small $\tau$ like 0.03 turns each row into a near one-hot and the object block falls apart into isolated pairs.

## 3. Finding 2: object-centric heads are spread across layers

### The question

Prior work says the final layer has the strongest object behavior. The paper accepts that but asks two sharper questions. Is every final-layer head object-centric? And are there object-centric heads in earlier layers that everyone ignores?

To answer, they run the clustering procedure (Section 4 below) on 4,000 random COCO images and count, per head, how often it lands in the object cluster.

![Figure 2 from Rawlekar et al. (2026): (a) frequency with which each head across all layers joins the object-centric cluster, (b) final-layer head frequencies with low-frequency heads circled, (c) number of strongly active heads per layer](/imgs/blogs/finding-distributed-object-centric-properties-in-self-supervised-transformers-fig2.webp)

### What they find

Two things. The final layer (layer 11 of a 12-layer ViT-B) has the highest concentration of object heads, but 4 of its 12 heads rarely join the object cluster. And layers 8 to 10 contribute many heads that do. The paper's own phrasing is that object discovery is a distributed property of the network.

The supplement backs the "noisy final-layer heads" claim with pictures. For DINOv3's layer 11, heads 5, 6, 7 and 11 are not selected: heads 5 and 6 miss regions of the person, and heads 7 and 11 pull in a large part of the background.

![Supplementary Figure 12 from Rawlekar et al. (2026): ensemble similarity maps for all 12 heads in layer 11 of DINOv3, where heads 5, 6, 7 and 11 are visibly worse at isolating the person](/imgs/blogs/finding-distributed-object-centric-properties-in-self-supervised-transformers-fig4.webp)

The pattern holds on DINO ViT-L/14 (10 strongly active heads in layer 23, 5 in layer 22, 1 in layer 21) and on MAE ViT-B/16, though MAE's maps are much noisier.

| Backbone | Layers x heads | Where object heads concentrate | Final layer clean? |
| --- | --- | --- | --- |
| DINO ViT-B (main analysis) | 12 x 12 | layers 8 to 11 | no, 4 of 12 noisy |
| DINO ViT-L/14 | 24 x 16 | layers 20 to 23 | no, several noisy |
| MAE ViT-B/16 | 12 x 12 | distributed, noisier | no |

The paper's explanation: a rich [CLS] summary needs patch correspondences, so patches of one object learn to attend to each other well before the last layer, and heads specialize. Averaging all of them blurs the useful ones. A companion CVPR 2026 paper finds the same few-heads pattern on the language side, in [LVLM visual-grounding heads](/blog/paper-reading/computer-vision/your-large-vision-language-model-only-needs-a-few-attention-heads-for-visual-grounding); for the backbones themselves, see [ViT, SigLIP and DINO explained](/blog/machine-learning/computer-vision/vit-siglip-dino-explained).

## 4. Object-DINO: cluster the heads, keep the object cluster

### The algorithm

![Figure 3 from Rawlekar et al. (2026): the Object-DINO pipeline, from per-head q, k and v similarity maps, to ensembled and flattened head vectors, to k-means over all heads, to selection of the cluster with the most final-layer heads and aggregation of its maps](/imgs/blogs/finding-distributed-object-centric-properties-in-self-supervised-transformers-fig3.webp)

The method has two phases.

1. **Head fingerprints.** For an image $x_i$, compute $A_{ens}^{\ell,h}$ for every one of the $L \times H$ heads and flatten each to a vector $f_{x_i}^{\ell,h} \in \mathbb{R}^{N^2}$. Heads that do the same job produce correlated maps, so their fingerprints sit close together.
2. **Cluster and pick.** Run k-means with $K = 5$ over all $L \times H$ fingerprints. Then choose the cluster that contains the most final-layer heads.

The selection rule, Eq. 4, written out:

$$
c_{obj} = \arg\max_{k} \; n_k^{(L)}, \qquad n_k^{(L)} = \left|\{(\ell, h) \in \mathcal{C}_k \;:\; \ell = L\}\right|
$$

Here $\mathcal{C}_k$ is the set of heads in cluster $k$, $L$ is the index of the last layer, and $n_k^{(L)}$ counts how many last-layer heads cluster $k$ received. The object heads are $H_{obj} = \{(\ell, h) \in \mathcal{C}_{c_{obj}}\}$, wherever in the network they live.

### A worked micro-example

Suppose the 12 final-layer heads split across the 5 clusters as 8, 2, 1, 1, 0. Cluster 1 wins with $n_1^{(L)} = 8$. Now read off every head in cluster 1, including the ones from earlier layers: in the illustration below that adds 5 heads from layer 10, 4 from layer 9, 6 from layer 8 and 2 from layer 7, giving 8 + 5 + 4 + 6 + 2 = 25 object heads in total.

![Illustrative 12 by 12 head map colored by cluster, with the object cluster reaching into layers 8 to 10 and skipping four circled final-layer heads, beside a bar chart of final-layer heads per cluster where the tallest bar marks c_obj](/imgs/blogs/finding-distributed-object-centric-properties-in-self-supervised-transformers-4.webp)

The "most final-layer heads" rule is clever bootstrapping. It uses the one thing everybody agrees on (the last layer is most object-like) only to *name* the cluster, then lets clustering pull in look-alike heads from other layers. The final layer is the anchor, not the answer.

### Choosing K

K is picked with the Davies-Bouldin score on 500 COCO images with DINOv3 features. The score is

$$
\mathrm{DB}(K) = \frac{1}{K} \sum_{i=1}^{K} \max_{j \neq i} \frac{s_i + s_j}{d_{ij}}
$$

where $s_i$ is the mean distance from cluster $i$'s points to its centroid and $d_{ij}$ is the distance between centroids $i$ and $j$. Lower is better. Across $K = 3$ to 10 it barely moves (1.717 to 1.737), minimum at $K = 5$. That flatness says the clusters are soft.

### A runnable sketch

The paper does not link code, so here is my own minimal reconstruction for DINOv2 ViT-B/14 from `torch.hub`. It hooks every block's fused `qkv` projection, drops the [CLS] token, and builds $A_{ens}$ per head.

```python
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").eval()
L, H = len(model.blocks), model.blocks[0].attn.num_heads  # 12 x 12
qkv_out = {}

def make_hook(layer):
    def hook(module, inp, out):
        qkv_out[layer] = out.detach()  # (B, 1 + N, 3 * D)
    return hook

for l, blk in enumerate(model.blocks):
    blk.attn.qkv.register_forward_hook(make_hook(l))

@torch.no_grad()
def ensemble_maps(x, tau=60.0, w=(1 / 3, 1 / 3, 1 / 3)):
    model(x)  # x: (1, 3, 224, 224) gives a 16 x 16 grid, N = 256
    maps = []
    for l in range(L):
        t = qkv_out[l]
        B, T, three_d = t.shape
        d = three_d // 3 // H
        qkv = t.reshape(B, T, 3, H, d).permute(2, 0, 3, 1, 4)  # (3, B, H, T, d)
        qkv = F.normalize(qkv[..., 1:, :], dim=-1)            # drop [CLS], unit rows
        A = (qkv @ qkv.transpose(-1, -2) / tau).softmax(-1)   # (3, B, H, N, N)
        maps.append(w[0] * A[0] + w[1] * A[1] + w[2] * A[2])  # (B, H, N, N)
    return torch.stack(maps, 1)[0]  # (L, H, N, N)

def object_heads(x, K=5):
    A = ensemble_maps(x)
    feats = A.flatten(2).reshape(L * H, -1).numpy()  # one fingerprint per head
    labels = KMeans(K, n_init=10, random_state=0).fit_predict(feats)
    c_obj = np.bincount(labels[(L - 1) * H:], minlength=K).argmax()
    heads = [(j // H, j % H) for j in np.flatnonzero(labels == c_obj)]
    return heads, A
```

The sketch has to decide two things the paper leaves open: Algorithm 1 clusters per image while Figure 2 aggregates 4,000 images, so per-image versus fixed head sets is unclear; and DINOv3's register tokens are not discussed (I would drop them with [CLS]).

## 5. Application 1: unsupervised object discovery

### Setup

TokenCut builds a patch graph weighted by final-layer key similarity and splits it with a normalized cut (the second eigenvector of the graph Laplacian). Object-DINO keeps that machinery and swaps only the affinity for the ensemble similarity of $H_{obj}$.

The metric is CorLoc: the percentage of images where the predicted box has IoU above 0.5 with at least one ground-truth box. If 100 images yield 31 boxes that overlap a real object by more than half, CorLoc is 31.0.

### Results

| Backbone | Method | VOC07 | VOC12 | COCO20k |
| --- | --- | --- | --- | --- |
| DINOv3 | TokenCut | 26.0 | 30.3 | 19.8 |
| DINOv3 | + Object-DINO | 30.8 (+4.8) | 36.0 (+5.7) | 23.4 (+3.6) |
| DINOv2 | TokenCut | 16.2 | 18.3 | 11.9 |
| DINOv2 | + Object-DINO | 25.7 (+9.5) | 30.7 (+12.4) | 19.7 (+7.8) |

### Which part of the method earns the gain

The layer ablation (their Table 4, DINOv3) is the cleanest evidence in the paper, because it splits the two findings apart.

| Head set | VOC07 | VOC12 | COCO20k |
| --- | --- | --- | --- |
| TokenCut (all final-layer keys) | 26.0 | 30.3 | 19.8 |
| Object heads, final layer only | 27.5 (+1.5) | 31.4 (+1.1) | 20.5 (+0.7) |
| Object heads, all layers | 30.8 (+4.8) | 36.0 (+5.7) | 23.4 (+3.6) |

Dropping the noisy final-layer heads buys 0.7 to 1.5 points. Adding the earlier-layer object heads buys another 3.3, 4.6 and 2.9. So "distributed" is not a footnote: roughly three quarters of the improvement comes from heads that final-layer methods never look at.

### What is load-bearing and might not transfer

The absolute numbers are low. As I recall, the original TokenCut paper reports CorLoc near 69 on VOC07 with DINO v1 ViT-S/16, far above the 16 to 31 reported here for DINOv2 and DINOv3. That gap is my context, not something this paper discusses.

A likely cause is the high-norm artifact tokens DINOv2-class models put in background patches, which "Vision Transformers Need Registers" showed damage this family of methods. Part of the gain may then be "routing around artifact heads", and a DINO v1 or register-equipped baseline is the missing row.

## 6. Application 2: grounding MLLMs against hallucination

### The decoding rule

At each decoding step the MLLM runs twice: on the image $u$ with $T_u$ ("Describe this image."), and on the Object-DINO map $v$ with $T_v$ ("Describe the highlighted regions."). Both see the text so far, $R$. Eq. 5 mixes the logits:

$$
\mathcal{L} = \alpha \,\mathrm{Logits}(y \mid T_u, R, u) + (1 - \alpha)\,\mathrm{Logits}(y \mid T_v, R, v), \qquad \alpha = 0.4
$$

![Illustrative logit fusion for the next token: the standard branch prefers Two, the object-map branch prefers Three, and the 0.4 to 0.6 blend picks Three](/imgs/blogs/finding-distributed-object-centric-properties-in-self-supervised-transformers-5.webp)

Worked example with made-up logits for the count after "There are". Standard branch: Two 5.0, Three 4.2, Four 1.0. Guidance branch: Two 2.0, Three 5.5, Four 1.5. Blend: Two ${0.4} \cdot 5.0 + 0.6 \cdot 2.0 = 3.2$, Three ${0.4} \cdot 4.2 + 0.6 \cdot 5.5 = 4.98$, Four 1.3. The grounded branch flips "Two" dogs into "Three", which is the paper's own Figure 4 example.

There is an inconsistency to flag. The Figure 4 caption writes the rule as $\mathrm{Logits}_u + \alpha \cdot \mathrm{Logits}_v$, and the supplement says $\alpha = 0$ "is equivalent to the original MLLM output", which matches the caption and contradicts Eq. 5 (where $\alpha = 0$ means *only* the guidance branch). Under Eq. 5, the guidance branch gets 60% of the weight; under the caption form, it gets ${0.4} / 1.4 \approx 29\%$ after normalization. Anyone reproducing the results needs to try both.

### Results

POPE asks yes/no questions of the form "Is there a {object} in the image?" (accuracy, precision, F1). CHAIR counts hallucinated objects in free captions: $C_S$ is the share of captions with at least one, $C_I$ the share of mentioned objects that are hallucinated.

| POPE | LLaVA-1.5 Acc / P / F1 | InstructBLIP Acc / P / F1 | Qwen-VL Acc / P / F1 |
| --- | --- | --- | --- |
| Regular | 77.4 / 73.3 / 79.2 | 74.6 / 71.2 / 76.4 | 79.8 / 80.1 / 79.7 |
| VCD | 77.1 / 72.1 / 79.4 | 77.2 / 74.2 / 78.4 | 81.3 / 80.6 / 81.5 |
| MARINE | 83.9 / n/a / 82.7 | 79.13 / n/a / 77.1 | n/a |
| DeGF | 81.6 / 80.5 / 81.9 | 80.3 / 80.9 / 80.1 | 83.4 / 84.4 / 82.9 |
| Object-DINO | 83.6 / 87.4 / 82.7 | 82.7 / 87.7 / 81.6 | 86.6 / 89.2 / 86.1 |

| CHAIR (64 new tokens) | LLaVA-1.5 $C_S$ / $C_I$ | InstructBLIP $C_S$ / $C_I$ |
| --- | --- | --- |
| Regular | 26.2 / 9.4 | 31.2 / 11.1 |
| DeGF | 18.4 / 6.1 | 24.0 / 7.7 |
| MARINE | 20.8 / 6.0 | 27.5 / 8.8 |
| Object-DINO | 18.4 / 5.9 | 21.4 / 8.8 |

The precision gain is the striking part (87.4 versus 73.3 on LLaVA-1.5 POPE): the model becomes reluctant to say "yes" to absent objects. On accuracy it is second to MARINE on LLaVA-1.5 (83.6 versus 83.9), it ties DeGF on LLaVA-1.5 $C_S$, and it is not the best $C_I$ on InstructBLIP (8.8 versus DeGF's 7.7).

Cost is moderate. On LLaVA-1.5 with 128 new tokens, latency is 7.1 s per instance (2.06x the regular 3.44 s) and peak memory 16,966 MB (1.07x), against 13.89 s for DeGF and 24.70 s for OPERA, with a CHAIR $C_S$ of 41.6 versus 55 for regular decoding.

## 7. Critique

**What is strong.** The core observation is simple, checkable and useful, the layer ablation cleanly separates the two findings, and the method is genuinely training-free and backbone-agnostic.

**What is weak.**

- The discovery baselines are weak, and the DINO v1 setting where TokenCut is strong is not reported.
- Several details needed to reproduce are missing: backbone size and patch size for the main tables, input resolution, register handling, per-image versus fixed head selection, and how the affinity is thresholded for TokenCut given that $A_{ens}$ is nearly uniform at $\tau = 60$.
- The MLLM rule is stated two incompatible ways, and the weight-ablation table's numbers do not match its stated dataset.
- The hallucination study lacks the obvious control: the same second pass with the [CLS] map or a generic saliency map as $v$. Without it, we cannot separate "Object-DINO's map" from "a second, highlight-focused look".

**What would change my mind.** If Object-DINO's head selection gave little or no gain on DINO v1 or on DINOv2 with registers (where artifact tokens are gone), I would read the paper as "avoid artifact heads" rather than "objects are distributed". Conversely, if the selected heads were stable across images and the gain held on artifact-free backbones, the distributed-object story would be very convincing.

## 8. Where this goes next

The authors name instance segmentation and vision-language models as future work. Here are five concrete directions, with the extrapolation being mine.

**1. From figure-ground to instances.** *Question:* can $A_{ens}$ restricted to $H_{obj}$ be decomposed into an unknown number of objects? *Why now:* the paper hands you a cleaner affinity matrix, and spectral methods read instance count from eigengaps. *First experiment:* run recursive normalized cuts (MaskCut style) on the Object-DINO affinity versus final-layer keys on COCO20k, and report multi-object recall.

**2. Object heads inside MLLM vision towers.** *Question:* do CLIP or SigLIP towers in LLaVA-style models contain an object cluster too, and can it replace the external DINO pass? *Why now:* the probe needs only a forward pass and k-means, and the paper already shows the pattern survives a change of objective (MAE). *First experiment:* run head clustering on LLaVA-1.5's CLIP ViT-L/14-336 tower, feed its object map as $v$, and compare POPE against the DINO-based guidance at lower latency.

**3. A fixed head mask per backbone.** *Question:* is $H_{obj}$ stable enough across images to precompute once? *Why now:* Figure 2's frequency map suggests some heads are selected almost always. *First experiment:* compute selection frequency on 4,000 images, keep heads above a threshold as a fixed mask, and measure the CorLoc gap against per-image selection; a small gap makes the method nearly free at inference.

**4. A better selector than "most final-layer heads".** *Question:* when does the anchor rule pick the wrong cluster? *Why now:* the DB score is flat across K, so cluster boundaries are soft, and backbones like CLIP may have global, non-object final layers. *First experiment:* score each cluster by spatial compactness of its maps (or agreement with a small set of 50 labeled boxes) and compare its choice to the anchor rule across DINOv2, DINOv3, MAE and CLIP.

**5. Training objectives that strengthen distributed objectness.** *Question:* can a regularizer that rewards block structure in $A_{ens}$ make more heads object-centric? *Why now:* the paper defines a differentiable, per-head objectness map. *First experiment:* fine-tune DINOv2 ViT-S with an auxiliary loss on $A_{ens}$ of the late layers and check whether CorLoc rises without hurting linear-probe accuracy.

## References

- Rawlekar, Swain, Cai, Wang, Yang, Ahuja. "Finding Distributed Object-Centric Properties in Self-Supervised Transformers." CVPR 2026 (highlight). [arXiv:2603.26127](https://arxiv.org/abs/2603.26127). CVPR page: [cvpr.thecvf.com/virtual/2026/poster/38825](https://cvpr.thecvf.com/virtual/2026/poster/38825).
- Caron et al. "Emerging Properties in Self-Supervised Vision Transformers" (DINO). ICCV 2021. [arXiv:2104.14294](https://arxiv.org/abs/2104.14294).
- Wang et al. "Self-Supervised Transformers for Unsupervised Object Discovery using Normalized Cut" (TokenCut). CVPR 2022. [arXiv:2202.11539](https://arxiv.org/abs/2202.11539).
- Darcet et al. "Vision Transformers Need Registers." ICLR 2024. [arXiv:2309.16588](https://arxiv.org/abs/2309.16588).
- Sibling posts: [Your large vision-language model only needs a few attention heads for visual grounding](/blog/paper-reading/computer-vision/your-large-vision-language-model-only-needs-a-few-attention-heads-for-visual-grounding), [ViT, SigLIP and DINO explained](/blog/machine-learning/computer-vision/vit-siglip-dino-explained).
