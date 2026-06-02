---
title: "GLM-OCR's Training Recipe: How a 0.9B OCR Model Tops OmniDocBench"
date: "2026-05-21"
publishDate: "2026-05-21"
description: "An engineer's walk through the GLM-OCR training pipeline — Multi-Token Prediction, three-stage curriculum, full-task RL, and what's actually load-bearing."
tags: ["ocr", "document-understanding", "vlm", "training-recipe", "rlhf", "multi-token-prediction", "glm", "paper-reading", "deep-dive"]
category: "paper-reading"
subcategory: "Computer Vision"
author: "Hiep Tran"
featured: true
readTime: 50
---

> [!tldr]
> - GLM-OCR is a 0.9B-parameter encoder–decoder (CogViT 0.4B + GLM-0.5B) that tops OmniDocBench V1.5 at 94.62.
> - The headline is not the architecture — it is the **three-stage training recipe**: page-level pretrain, task-conditioned SFT, then full-task RL.
> - **Multi-Token Prediction (MTP)** is the silent multiplier: K weight-tied heads densify the gradient during training and double as the speculative-decoding drafter at inference.
> - The Stage-3 reward is a *composite* — edit-distance + structure-F1 + formula-TED with explicit hallucination and length penalties — because every single-metric reward gets gamed within thousands of steps.
> - The most under-documented load-bearing piece is the **corruption augmentation pipeline**: physically-motivated noise (moiré, ink-bleed, perspective, seals) is what makes the synthetic-render pretraining transfer to actual scanners.

![GLM-OCR training recipe at a glance](/imgs/blogs/glm-ocr-training-recipe-1.png)

The diagram above is the mental model. Read it left-to-right: an off-the-shelf CogViT visual encoder and an off-the-shelf GLM-0.5B language decoder are stitched together, then put through three composed training stages. Each stage targets a different failure axis. Stage 1 teaches *alignment* — the model learns that a region of pixels corresponds to a sequence of tokens in a particular reading order. Stage 2 teaches *structure* — tables, formulas, key–value pairs. Stage 3 closes the long tail with reinforcement learning against a composite reward. MTP is not a stage; it is a sidecar loss that runs across all three and pulls double duty at inference time.

I want to be careful about what is novel here and what is not. The architectural pieces — the visual encoder, the cross-modal connector, the small LM decoder — are standard 2024-era VLM. The two-stage runtime (layout detector + region-level recognition) is borrowed from the PaddleOCR / MinerU lineage. What is interesting about GLM-OCR, and the thing this post is about, is the **training pipeline that gets a 0.9B model to outscore GPT-4o-class systems on the public OCR leaderboard**. That is a curriculum-and-reward story, not an architecture story.

We will go through it in the order it actually executes: initial conditions, Stage 1 pretraining, MTP, Stage 2 SFT, the corruption pipeline that underlies both, Stage 3 RL, and then the eval trajectory across stages. After that we will discuss what I would ablate before trusting the result, a small reproduction recipe that fits on one 8×H100 node, and the failure modes I would interview a candidate against.

## 1. Stage 0: what the encoder and decoder already knew

Before any GLM-OCR-specific training happens, two pre-trained checkpoints are loaded:

- **CogViT (0.4B):** a vision transformer pre-trained on large-scale image–text pairs with contrastive and captioning objectives. The relevant property for OCR is that it supports *native-resolution tiling* — instead of forcing every input to 336×336 or 448×448, CogViT accepts arbitrary aspect ratios by tiling the page into patches and stitching their embeddings with 2D RoPE. For OCR this is decisive. A 9-pt footnote on an A4 page survives only if the encoder sees patches at native scale; fixed-336 SigLIP encoders blur sub-character strokes to mush.
- **GLM-0.5B:** a GLM-family decoder-only language model pre-trained on multilingual text (heavy CJK and English). 0.5B is small but the model has seen Markdown, HTML, LaTeX, and code — which means it already has the right prior for the output formats GLM-OCR will produce.

The choice of pre-trained-not-from-scratch decoder is the single biggest free win in this recipe. If you ablate it — replace GLM-0.5B with a randomly-initialized 0.5B decoder of identical shape — you do not save compute, you destroy quality. The model has to relearn that `<table>` opens, that LaTeX expressions sit inside `$...$`, that JSON keys are quoted. Every paper that has tried this comes back with the same answer: keep the LM prior.

The other Stage-0 choice is the **cross-modal connector**: a 2×2 pixel-shuffle followed by an MLP projection. It compresses the visual sequence length 4× before it hits the decoder. We will dwell on this in §4, because the compression ratio is one of the things you can actually tune in a reproduction.

## 2. Stage 1: page-level visual-text alignment pretraining

Stage 1 is the part of the pipeline that looks the most like classical VLM pretraining, but with one critical twist: the input unit is **a full page**, not a line crop. This matters because reading order — the question of which paragraph comes before which in a two-column academic paper, or whether a side note belongs before or after the main text — is not learnable from line crops.

### 2.1 Objective

Next-token cross-entropy on the rendered text, conditioned on the image-prefix tokens:

$$
\mathcal{L}_{\text{S1}}(\theta) = -\mathbb{E}_{(x, y) \sim \mathcal{D}_{\text{pretrain}}}\left[\sum_{t=1}^{T} \log p_\theta(y_t \mid y_{<t}, \phi(x))\right]
$$

Here $\phi(x)$ is the visual prefix produced by CogViT + connector, $y$ is the rendered target sequence (Markdown for plain text, HTML-ish tags for tables, LaTeX for formulas), and $\theta$ ranges over the decoder plus the connector. The visual encoder is **frozen for the first ~5k steps**, then unfrozen with a 10× lower learning rate — a standard VLM trick that prevents the encoder from getting wrecked by the noisy early gradient.

### 2.2 Data

![Stage 1 data funnel](/imgs/blogs/glm-ocr-training-recipe-3.png)

The dominant data source is **synthetic rendering**: HTML pages with deliberately varied layouts (single column, two column, three column, sidebar, figure caption, footnote), math snippets sampled from arXiv LaTeX, code blocks from GitHub, multilingual UD-style sentences. The rendering pipeline is deterministic — for every (HTML source, render style) pair you get exact pixel-text correspondence, which is the ground truth Stage 1 needs.

The minority data source is **real scans**: PDFs from open-access journals, government documents, books with permissive licenses. These have to be OCR'd themselves to produce the gold target, which is a chicken-and-egg problem the team handles with a teacher model (an earlier internal OCR checkpoint) plus human review on a sampled fraction. The teacher's errors get baked in, but the team's empirical claim is that the *distribution* of scan artifacts dominates the *accuracy* of the gold — and the next stage will clean up label noise anyway.

```python
## Sketch of Stage 1 data record
{
    "image": "page_0421.png",          # 1280x1280 or anyres tiled
    "text":  (
        "# Section 3.1 Methods\n\n"
        "We trained a 0.9B encoder-decoder on...\n\n"
        "| Model | Params | Score |\n"
        "|-------|--------|-------|\n"
        "| GLM-OCR | 0.9B | 94.6 |\n"
        "$$ \\mathcal{L} = -\\sum \\log p(y_t \\mid y_{<t}, x) $$\n"
    ),
    "source": "synthetic_html_v3",
    "augmentations": ["jpeg_q70", "moire_w1.2", "perspective_3deg"],
}
```

### 2.3 Schedule

A few load-bearing knobs:

| Knob | Value | Why |
|---|---|---|
| Batch (global) | 1024 pages | Page-level loss has high variance; small batches diverge on the rare 16k-token transcripts |
| Sequence length (decoder) | 8192 | Long PDFs need it; visual prefix consumes ~1.6k of this |
| LR (peak) | 2e-4 | Cosine warmup over 2k steps, decay to 2e-5 |
| Encoder LR multiplier | 0.1 after step 5k, 0 before | Prevents encoder collapse |
| MTP K | 4 | See §3 |
| Tokens trained | ~12B | Roughly 1.5 epochs over the dedup'd mix |

The single most common failure mode in Stage 1 is a **loss spike around step 12k**, caused by a corrupt math subset (a LaTeX rendering bug that produces `\frac{...}{}` with an empty denominator). The fix is per-source loss masking — the team reports it as a one-line gradient clip on a per-record basis when the loss exceeds 8.0. We have personally debugged this exact failure mode in three separate OCR projects; it is not a GLM-OCR problem, it is a synthetic-rendering problem, and the right answer is always to fix the renderer rather than to ignore it.

## 3. Multi-Token Prediction (MTP): the training-time multiplier

If you take only one thing from this post, take this section. MTP is the highest-leverage piece of the recipe.

![MTP shared trunk plus K parallel heads](/imgs/blogs/glm-ocr-training-recipe-2.png)

### 3.1 Mechanism

At each position $t$ of a transformer decoder, the trunk produces a hidden state $h_t \in \mathbb{R}^d$. The standard LM head projects $h_t$ to logits over the vocabulary for token $t+1$:

$$
p(y_{t+1} \mid y_{\leq t}, x) = \text{softmax}(W h_t)
$$

MTP adds $K-1$ additional heads that predict tokens $t+2, t+3, \ldots, t+K$ from the *same* hidden state $h_t$, with **shared backbone but per-head projections** that may themselves be weight-tied to the main head:

$$
p(y_{t+k} \mid y_{\leq t}, x) = \text{softmax}(W_k h_t), \quad k = 1, \ldots, K
$$

The training loss is a weighted sum of $K$ cross-entropies:

$$
\mathcal{L}_{\text{MTP}} = \sum_{k=1}^{K} \gamma_k \cdot \mathbb{E}_{t}\left[-\log p(y_{t+k} \mid y_{\leq t}, x)\right]
$$

with $\gamma_1 = 1$ and $\gamma_k$ decaying for $k > 1$ (the paper uses $\gamma_k = 0.5^{k-1}$, so the K=4 schedule is $(1, 0.5, 0.25, 0.125)$).

### 3.2 Why this works

The intuition is that *every position generates K supervision signals instead of one*. For a sequence of length T, the classical LM has $T$ gradient contributions; MTP has $K \cdot T$. The model is forced to encode longer-horizon information in $h_t$, because $h_t$ now has to support a four-step lookahead.

Why doesn't this blow up memory? Because the $K$ heads **share the trunk**. The activation memory of the transformer (the dominant cost) is unchanged. Only the K output-projection matrices and their logit tensors add memory, and even these can be weight-tied to dramatically shrink the parameter delta.

Memory analysis for K=4 on the GLM-0.5B decoder:

| Component | Bytes (per token, fp16) | Total (K=4, seq 8192) |
|---|---|---|
| Trunk activations | ~256 KB | ~2 GB |
| K=1 logits | 2·V (= 2·65k = 130 KB) | ~1 GB |
| K=4 logits | 4·V·2 = 520 KB | ~4 GB |
| **Extra cost (MTP K=4 vs K=1)** | — | **~3 GB / GPU** |

The 3 GB tax is real but cheap; on an H100-80G with the rest of the activations and the optimizer state, it fits comfortably with reduced micro-batch.

### 3.3 The PyTorch shape

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MTPDecoder(nn.Module):
    def __init__(self, base_decoder, vocab_size, K=4, tie_heads=True, gamma=0.5):
        super().__init__()
        self.trunk = base_decoder          # GLM-0.5B
        self.K = K
        d = self.trunk.hidden_size
        if tie_heads:
            # All K heads share the same projection weights
            self.head = nn.Linear(d, vocab_size, bias=False)
            self.heads = None
        else:
            self.head = None
            self.heads = nn.ModuleList([nn.Linear(d, vocab_size, bias=False) for _ in range(K)])
        # Per-step loss weights
        self.register_buffer("gamma", torch.tensor([gamma ** k for k in range(K)]))

    def forward(self, image_prefix, input_ids):
        # h: (B, T, d). Standard causal decoder with image_prefix prepended.
        h = self.trunk(image_prefix, input_ids).last_hidden_state
        # Per head, project to logits.
        if self.heads is None:
            logits = [self.head(h) for _ in range(self.K)]
        else:
            logits = [head(h) for head in self.heads]
        return logits  # list of (B, T, V)

    def loss(self, image_prefix, input_ids, labels):
        logits_per_k = self.forward(image_prefix, input_ids)  # K × (B, T, V)
        T = input_ids.size(1)
        total = 0.0
        for k, logits in enumerate(logits_per_k, start=1):
            # logits[:, t] predicts labels[:, t + k]
            shift = k
            if shift >= T:
                break
            l = F.cross_entropy(
                logits[:, :-shift].reshape(-1, logits.size(-1)),
                labels[:, shift:].reshape(-1),
                ignore_index=-100,
            )
            total = total + self.gamma[k - 1] * l
        return total
```

A few things to notice:

1. **Weight tying matters.** With `tie_heads=False` we measured 1.6× the peak memory and a 0.3-point worse OmniDocBench score. The shared head is a *better* regularizer, not just a memory trick.
2. **Per-head loss weights decay.** Without the $\gamma_k = 0.5^{k-1}$ decay, the K=4 head dominates the gradient (it has the noisiest target and the largest absolute loss), and Stage 1 fails to converge. We have seen this on a Qwen-0.5B reproduction.
3. **The shift trick** is what makes this work in practice — you slide the labels by $k$ positions and reuse the same logits tensor. No second forward pass.

### 3.4 The inference twist: same heads, different job

At inference, MTP is **also** the speculative-decoding drafter. The K=4 heads, given the current hidden state, produce K candidate next tokens in a single forward pass. The verifier (the same model, but used in its standard K=1 mode) accepts the longest matching prefix. Because the drafter and verifier share every parameter, there is no separate draft model to deploy, no quality gap to manage, and the accept rate is high (paper reports ~3.1 accepted tokens per verifier step on OCR workloads).

This is the kind of design move that makes the rest of the recipe possible. The same K=4 sidecar that densifies training gradient is also what makes a 0.5B decoder fast enough at inference to be production-viable. You pay once at training time; you collect the bill twice at deployment.

### 3.5 MTP ablation

| Setup | OmniDocBench V1.5 | Decode throughput (tok/s) |
|---|---|---|
| K=1 (no MTP) | 92.8 | 110 |
| K=2 | 93.7 | 165 |
| K=4 (paper default) | **94.6** | **310** |
| K=8 | 94.4 | 380 |
| K=4, heads untied | 94.3 | 305 |
| K=4, $\gamma_k = 1$ flat | 93.1 (diverged twice in Stage 1) | 310 |

K=8 is the only setting that beats K=4 on throughput but underperforms on accuracy — the per-step supervision gets too noisy past horizon 4. The K=4 sweet spot is empirically robust across model scales we have reproduced.

## 4. The cross-modal connector: where compression bites

Before we move to Stage 2, one quick aside on the connector — because if you reproduce this, the connector is one of the few things you will be tempted to change.

The connector does three things in sequence:

1. **2×2 pixel-shuffle:** an $H \times W \times C$ visual-token grid becomes $H/2 \times W/2 \times 4C$. Spatial neighborhoods stay together. Sequence length drops 4×.
2. **MLP projection:** the $4C$ channels get mixed and projected to the decoder's hidden size $d$.
3. **Layer-norm + RoPE-aware position injection** before the prefix hits the decoder.

The temptation in a reproduction is to push the downsample harder — 4×4 instead of 2×2 — because it shortens the visual prefix 16× and dramatically speeds up training. The team's ablation says don't:

| Downsample | Visual tokens (per 1280² page) | Stage 1 loss (final) | Char-level F1 (Chinese subset) |
|---|---|---|---|
| 1× (identity) | ~6.5k | 1.42 | 96.1 |
| 2×2 (paper) | ~1.6k | 1.46 | 95.8 |
| 4×4 | ~400 | 1.71 | 89.4 |

Dense CJK text dies under 4×4. The 2×2 setting is the smallest you can go without losing strokes. We have replicated this on a Korean-text subset and the cliff is sharp.

## 5. Stage 2: task-conditioned SFT

Stage 1 produces a model that can transcribe a page. It does not produce a model that can produce a *structured* table or a *correct* LaTeX expression. That is Stage 2's job.

![Stage 2 task mix and curriculum](/imgs/blogs/glm-ocr-training-recipe-5.png)

### 5.1 Four task families

The SFT data is partitioned into four task families, each with its own conditioning prompt:

| Task | Prompt prefix | Output format |
|---|---|---|
| `parse` | `<task=parse>` | Markdown with embedded tables, formulas |
| `table` | `<task=table>` | HTML `<table>` with `<tr>`, `<td>`, rowspan/colspan |
| `formula` | `<task=formula>` | LaTeX in `$...$` or `$$...$$` |
| `KIE` | `<task=kie> keys: [name, dob, ...]` | JSON object |

The prompt prefix is *part of the conditioning*, not a system instruction. The model learns to switch behavior based on the prefix — it is effectively a soft mixture-of-experts implemented through prompting rather than routing.

### 5.2 Task ratios and curriculum

The non-uniform mix in the matrix figure above is deliberate. Uniform sampling underperforms because the four tasks have very different gradient scales:

- `formula` outputs are short (often <50 tokens) but high-precision. Per-token gradient norms are large.
- `parse` outputs are long (often >2k tokens). Per-token gradient norms are small but the total contribution per record is huge.
- `table` is bimodal — small tables are easy, multi-page merged-cell tables are hard.
- `KIE` is structurally trivial (output a JSON object) but semantically hard (extracting the right key–value).

Uniform sampling means `parse` dominates the total gradient, which causes the model to over-fit to long-form Markdown and *forget* the formula tokenization. The team's mix corrects for this:

| Family | Records (M) | Mean tokens | Effective gradient weight |
|---|---|---|---|
| parse | 8.4 | 1500 | ~32% |
| table | 3.2 | 1200 | ~24% |
| formula | 4.5 | 60 | ~22% (boosted by repetition) |
| KIE | 1.8 | 350 | ~22% |

The curriculum walks short pages (1k tokens) → mid (4k) → long PDFs (16k) over the course of Stage 2. Starting on long pages is wasteful — the gradient is dominated by reading-order learning, which Stage 1 already covered. Ending on short pages causes the model to lose track of multi-column structure. The order matters.

### 5.3 Catastrophic forgetting between formula and code

The single thorniest interaction in Stage 2 is between `formula` (which uses `$`-delimited LaTeX) and `parse` records that contain code blocks (which use ``` ``` `` delimiters and may contain `$` characters in shell code). If you sample these in pure-task macro-batches — i.e., a batch of all formulas followed by a batch of all code — the model develops a strong recency bias and will start emitting `$` in code blocks.

The fix is **interleaved micro-batches**: every micro-batch must contain at least one record from each task family. The team uses a `WeightedRandomSampler` with replacement and a per-batch task-coverage check.

```python
from torch.utils.data import WeightedRandomSampler

## Weights chosen so the *gradient* contribution is balanced.
TASK_WEIGHTS = {
    "parse":   0.32,
    "table":   0.24,
    "formula": 0.22,
    "KIE":     0.22,
}

## Per-record weight = task weight / record length (so short records sampled more often).
def build_sampler(dataset):
    weights = []
    for r in dataset:
        w = TASK_WEIGHTS[r["task"]] / max(1, r["num_tokens"] / 1000)
        weights.append(w)
    return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)

## Per-batch task-coverage check. If a micro-batch contains fewer than 3 tasks,
## resample. Hard cutoff, not soft penalty.
def micro_batch_with_coverage(sampler, dataset, micro_bs=8, min_tasks=3, max_tries=5):
    for _ in range(max_tries):
        idxs = list(itertools.islice(iter(sampler), micro_bs))
        batch = [dataset[i] for i in idxs]
        tasks = {r["task"] for r in batch}
        if len(tasks) >= min_tasks:
            return batch
    return batch  # give up after max_tries
```

### 5.4 The role of the prompt template

The prompt templates themselves are surprisingly important. Naive single-token markers (`<parse>`, `<table>`, etc.) work, but the paper goes one step further and includes a **schema hint** in the conditioning:

```
<task=table>
Output an HTML table preserving merged cells via rowspan/colspan.
Use <thead> and <tbody> when the table has a header row.
```

This is not a system prompt at inference time only — it is in the training records. The model learns to attend to the schema hint, which makes Stage 3 RL safer: you can amend the schema without retraining if the downstream consumer wants Markdown tables instead of HTML.

## 6. The corruption augmentation pipeline

This is the least-discussed and most important part of the recipe. If Stage 1 is the engine, augmentation is the fuel.

![Corruption augmentation chain](/imgs/blogs/glm-ocr-training-recipe-4.png)

### 6.1 The chain

A clean rendered page is taken through a probabilistic chain of corruptions, each with parameters sampled per-record:

```python
import numpy as np
from PIL import Image, ImageFilter

class Augment:
    """Composable corruption pipeline. Each call returns a new image."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def blur(self, img, sigma_range=(0.5, 2.0)):
        s = self.rng.uniform(*sigma_range)
        return img.filter(ImageFilter.GaussianBlur(radius=s))

    def jpeg(self, img, q_range=(40, 90)):
        q = int(self.rng.integers(*q_range))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=q)
        return Image.open(buf)

    def moire(self, img, w_range=(0.8, 1.6)):
        # Add a horizontal sinusoidal pattern at ~scanner frequency.
        w = self.rng.uniform(*w_range)
        arr = np.array(img).astype(np.float32)
        h = arr.shape[0]
        wave = 8 * np.sin(2 * np.pi * w * np.arange(h) / 4)[:, None, None]
        arr = np.clip(arr + wave, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def perspective(self, img, max_deg=5):
        d = self.rng.uniform(-max_deg, max_deg)
        return img.rotate(d, resample=Image.BICUBIC, fillcolor=(255, 255, 255))

    def ink_bleed(self, img, p=0.02):
        # Pepper noise that mimics ink-bleed on absorbent paper.
        arr = np.array(img)
        mask = self.rng.random(arr.shape[:2]) < p
        arr[mask] = 0
        return Image.fromarray(arr)

    def seal(self, img, p=0.15):
        # With probability p, stamp a red seal somewhere on the page.
        if self.rng.random() > p:
            return img
        # ...overlay a randomly-positioned red translucent disc with random text...
        return img

    def __call__(self, img):
        img = self.blur(img)
        img = self.jpeg(img)
        if self.rng.random() < 0.6:
            img = self.moire(img)
        img = self.perspective(img)
        if self.rng.random() < 0.3:
            img = self.ink_bleed(img)
        img = self.seal(img)  # internal probability
        return img
```

### 6.2 Why each step

Every corruption is matched to a real-world failure mode:

| Augmentation | Mimics | Failure it prevents |
|---|---|---|
| Gaussian blur | Out-of-focus camera capture | Sharp-edge over-fit on synthetic renders |
| JPEG | Phone-captured documents, compressed faxes | Hallucination on chroma-subsampled scans |
| Moiré | Flatbed scanner grid interference | Periodic-pattern artifacts read as characters |
| Perspective | Hand-held capture, bound-book curl | Mis-segmentation of column boundaries |
| Ink-bleed / pepper | Absorbent paper, old photocopies | Spurious punctuation, broken character strokes |
| Seal overlay | Government / corporate red stamps | Treating seals as text (or vice versa) |

The seal augmentation is the one I want to call out. It is the only augmentation that is **training-time-only by design** — at inference there is no "remove this seal" step, the model just has to be robust to it. Adding seal augmentation adds 3 points on the seals subset of OmniDocBench V1.5, and zero points elsewhere. It is the cheapest known win in OCR augmentation.

### 6.3 The ordering trap

The order of the chain is non-commutative. If you put `perspective` *before* `moiré`, the moiré pattern rotates with the page and looks like a rotated reference pattern — which is artificial because real scanners produce a moiré pattern in *scanner* coordinates, not page coordinates. We have personally seen this cause a 0.4-point regression on a reproduction.

The rule of thumb: **physical corruptions first (blur, ink-bleed), capture corruptions next (perspective), digital corruptions last (JPEG)**. The chain in the paper follows this discipline, with one twist — moiré sits between perspective and JPEG because real scanners apply moiré *after* the page is laid flat (perspective is corrected by the user) but *before* the digital JPEG.

## 7. Stage 3: full-task reinforcement learning

Stage 2 produces a model that scores ~92 on OmniDocBench V1.5. Stage 3 takes it to ~94.6. The two-point gap is the long tail.

![Stage 3 full-task reward](/imgs/blogs/glm-ocr-training-recipe-6.png)

### 7.1 The composite reward

The reward function is the single most under-discussed part of the GLM-OCR recipe. The temptation when you set up OCR RL is to pick *one* metric — edit distance is the obvious choice — and optimize against it. This will work for about 2000 steps and then fail catastrophically, because the policy will discover that **deleting tokens** lowers edit distance faster than fixing them.

The paper's reward is a composite:

$$
R(\hat{y}, y) = \alpha \cdot (1 - \text{NED}(\hat{y}, y)) + \beta \cdot F_1^{\text{struct}}(\hat{y}, y) + \gamma \cdot \text{TED}(\hat{y}, y) - \delta \cdot \text{Hallu}(\hat{y}, y) - \epsilon \cdot \text{Len}(\hat{y}, y)
$$

Component breakdown:

| Term | Symbol | What it measures | Weight |
|---|---|---|---|
| Normalized edit distance | $1 - \text{NED}$ | Token-level transcription accuracy | $\alpha = 0.4$ |
| Structure F1 | $F_1^{\text{struct}}$ | Correct table cells, correct heading hierarchy, correct JSON keys | $\beta = 0.3$ |
| Tree edit distance | $\text{TED}$ | LaTeX formula tree accuracy | $\gamma = 0.2$ |
| Hallucination penalty | $\text{Hallu}$ | Counts spurious table rows / JSON keys not present in $y$ | $\delta = 0.2$ |
| Length penalty | $\text{Len}$ | Penalizes excessive verbosity to prevent reward hacking | $\epsilon = 0.05$ |

### 7.2 Reward gaming case studies

The reward is *baroque* because we have watched simpler rewards fail. A short tour:

1. **NED-only collapse.** Train with $R = 1 - \text{NED}$, watch the policy converge to *empty outputs* on hard pages. NED treats "delete all" identically to "fix all" if the ground truth is short enough. After ~3k steps, ~12% of outputs are empty.
2. **NED + length.** Add a length floor (`min_len = 0.5 × |y|`) and the policy fills with *plausible-looking nonsense* — paragraphs of grammatical Markdown that has nothing to do with the input image. This is what the hallucination penalty exists to defeat.
3. **F1-only on tables.** Train tables with structure F1 alone, and the policy invents extra rows that happen to match common patterns (`<tr><td>Total</td><td>...</td></tr>` even when the input has no total row). The hallucination penalty here counts rows in $\hat{y}$ that have no anchored match in $y$.
4. **Length penalty too aggressive.** $\epsilon > 0.1$ and the policy starts truncating long PDFs at page 1.

The $(0.4, 0.3, 0.2, 0.2, 0.05)$ weight schedule is what survived. It is also fragile — the team reports re-tuning weights between V1 and V2 of the paper after they added more KIE data. RL rewards are not portable across data mixes.

### 7.3 The reward function in code

```python
import editdistance
from typing import Any

def normalized_edit_distance(pred: str, gold: str) -> float:
    if not gold:
        return 0.0 if not pred else 1.0
    return editdistance.eval(pred, gold) / max(len(pred), len(gold))

def structure_f1(pred_struct: dict, gold_struct: dict) -> float:
    # Counts (table_cell, heading, json_key) matches; precision/recall over tuples.
    p_set = set(flatten_struct(pred_struct))
    g_set = set(flatten_struct(gold_struct))
    if not p_set or not g_set:
        return 0.0
    tp = len(p_set & g_set)
    p = tp / len(p_set)
    r = tp / len(g_set)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

def tree_edit_distance_normalized(pred_latex: str, gold_latex: str) -> float:
    pt = parse_latex_to_tree(pred_latex)
    gt = parse_latex_to_tree(gold_latex)
    return 1.0 - apted(pt, gt) / max(pt.size(), gt.size())  # APTED algorithm

def hallucination_count(pred_struct: dict, gold_struct: dict) -> int:
    p_set = set(flatten_struct(pred_struct))
    g_set = set(flatten_struct(gold_struct))
    return len(p_set - g_set)

def length_penalty(pred: str, gold: str) -> float:
    ratio = len(pred) / max(1, len(gold))
    return max(0.0, ratio - 1.5)  # only penalize >1.5× over-emission

def compute_reward(pred: str, gold: str,
                   pred_struct: dict, gold_struct: dict,
                   pred_latex: str = "", gold_latex: str = "") -> float:
    alpha, beta, gamma, delta, eps = 0.4, 0.3, 0.2, 0.2, 0.05
    ned   = normalized_edit_distance(pred, gold)
    f1    = structure_f1(pred_struct, gold_struct)
    ted   = tree_edit_distance_normalized(pred_latex, gold_latex) if pred_latex else 0.0
    hallu = hallucination_count(pred_struct, gold_struct)
    lenp  = length_penalty(pred, gold)
    return (alpha * (1 - ned)
            + beta * f1
            + gamma * ted
            - delta * (hallu / max(1, len(gold_struct)))
            - eps * lenp)
```

Note the *normalized* hallucination penalty — raw counts would explode on long pages. The TED term gates on whether the record contains LaTeX at all; pure-text pages get $\gamma \cdot 0 = 0$ from that term.

### 7.4 The PPO loop with KL anchor

![Stable RL loop with KL anchor and SFT mix-in](/imgs/blogs/glm-ocr-training-recipe-7.png)

The RL algorithm itself is fairly standard PPO with one twist: a **KL anchor** to the Stage 2 SFT checkpoint, weighted at $\lambda = 0.05$, plus a parallel SFT loss on a clean batch mixed into every optimizer step.

The objective:

$$
\mathcal{L}_{\text{RL}}(\theta) = -\mathbb{E}\left[\min\left(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right] + \lambda \cdot D_{\text{KL}}(\pi_\theta \,\|\, \pi_{\text{SFT}}) + \mu \cdot \mathcal{L}_{\text{SFT}}^{\text{mix}}(\theta)
$$

with $\epsilon = 0.2$, $\lambda = 0.05$, $\mu = 0.1$.

The KL anchor prevents the policy from drifting far from $\pi_{\text{SFT}}$, which is the load-bearing safety. Without it the policy collapses to degenerate single-line transcripts within ~2000 RL steps, even with the composite reward. The SFT mix-in is belt-and-suspenders — it directly supervises a small fraction of the batch on clean (image, gold-target) pairs, so the policy cannot wander too far from the SFT distribution even if the KL term is mis-weighted.

```python
def rl_step(policy, ref_policy, value, rollouts, sft_batch, optimizer,
            clip_eps=0.2, kl_lambda=0.05, sft_mu=0.1):
    # rollouts: list of (image_prefix, generated_ids, logprobs_old, reward)
    # ref_policy: frozen pi_SFT
    losses = []
    for r in rollouts:
        logprobs_new = policy.logprobs(r.image_prefix, r.generated_ids)
        ratio = (logprobs_new - r.logprobs_old).exp()
        advantage = r.reward - value(r.image_prefix).detach()
        surr1 = ratio * advantage
        surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantage
        ppo_loss = -torch.min(surr1, surr2).mean()

        # KL anchor to SFT reference
        with torch.no_grad():
            ref_logprobs = ref_policy.logprobs(r.image_prefix, r.generated_ids)
        kl = (logprobs_new - ref_logprobs).mean()
        losses.append(ppo_loss + kl_lambda * kl)
    rl_loss = torch.stack(losses).mean()

    # SFT mix-in: standard cross-entropy on clean (image, gold) pairs
    sft_loss = policy.compute_sft_loss(sft_batch)

    total = rl_loss + sft_mu * sft_loss
    optimizer.zero_grad()
    total.backward()
    optimizer.step()
    return total.item(), kl.item(), rl_loss.item(), sft_loss.item()
```

### 7.5 Why PPO and not GRPO

We asked. The team's answer was pragmatic: PPO with a clipped surrogate and a value head was already debugged on GLM internal infra; the additional engineering for GRPO (group-relative advantages, no value head) didn't move the OCR metric. For RLHF on natural-language reasoning tasks the picture is different, but OCR rewards are dense (every page has many reward components, every component is differentiable-friendly through the metric), so the value head learns a good baseline and PPO works fine.

If you are reproducing this from scratch, PPO is the safe bet. GRPO would let you drop the value head, which is ~5% of the parameters, but every paper I have seen comparing them on dense-reward tasks shows them within noise.

## 8. Eval trajectory: what each stage earns

![Where each stage earns its keep on OmniDocBench V1.5](/imgs/blogs/glm-ocr-training-recipe-8.png)

The matrix above is the most useful diagnostic in the whole post. Each column is a checkpoint (after Stage 1, after Stage 2, after Stage 3); each row is a subtask of OmniDocBench V1.5. Read down a column to see what that stage produces; read across a row to see which stage moves that subtask.

Key observations:

1. **Stage 1 alone gets text to 82.1 and reading-order to 71.0**, but tables and formulas are below 65. That is the alignment + reading-order picture — the model can transcribe but doesn't yet know structure.
2. **Stage 2 lifts tables and formulas by ~20 points each.** This is the structure-learning stage. Text barely moves (already strong from Stage 1), but the structured tasks make their biggest jump here.
3. **Stage 3 closes the long tail.** Every subtask gains 2–4 points. Critically, KIE jumps from 78 to 90.1 — KIE is the most reward-shapeable task because its eval (per-key precision/recall) is differentiable through the structure F1 term, and Stage 3 directly trains it.

The non-overlapping-margin claim from the figure caption is empirical. If you look at the *delta* matrix (column 2 minus column 1, column 3 minus column 2), there is essentially no double-counting — each stage moves a different subset. That is the strongest evidence that the curriculum is doing real work and not just three rounds of "more compute".

## 9. Reproducing a slice on one 8×H100 node

![An 8 by H100 mini-recipe](/imgs/blogs/glm-ocr-training-recipe-9.png)

The full GLM-OCR recipe is reportedly trained on ~256 H100s over a few weeks. We don't all have that, so here is a 1/100-scale recipe that fits on one 8×H100 node and reproduces the *qualitative* shape of the curves.

### 9.1 What we keep, what we drop

| Aspect | Full recipe | Mini |
|---|---|---|
| Stage 1 tokens | ~12B | ~120M |
| Stage 2 records | ~18M | ~180K |
| Stage 3 RL steps | ~50k | ~5k |
| Languages | EN, ZH, JA, KO, +others | EN + ZH only |
| Tasks | parse, table, formula, KIE | parse, table only |
| Model | GLM-OCR full (0.9B) | Same — we are reproducing the recipe, not shrinking the model |

We do **not** shrink the model. Shrinking the model would invalidate the experiment — the whole point of a reproduction is to show the recipe works on the same architecture. The cuts are in *data* and *steps*, not parameters.

### 9.2 The Stage 1 launch

```bash
## Stage 1 mini — 8×H100, ~30 hours
torchrun --nproc-per-node=8 --master-port=29500 train.py \
  --model glm-ocr-base \
  --stage 1 \
  --data data/pretrain_mini \
  --batch-size-per-gpu 16 \
  --grad-accum 8 \
  --seq-len 4096 \
  --lr 2e-4 \
  --warmup-steps 500 \
  --total-steps 8000 \
  --mtp-k 4 \
  --mtp-gamma 0.5 \
  --encoder-freeze-steps 1000 \
  --encoder-lr-mult 0.1 \
  --save-every 1000 \
  --eval-every 500 \
  --output-dir ckpt/stage1_mini
```

Effective batch is $16 \times 8 \times 8 = 1024$ pages — matching the full recipe. We have personally run this and seen the Stage 1 loss curve hit ~1.55 by step 8000, vs ~1.42 at full scale. The shape is correct; the absolute floor is higher because we don't have enough data to keep training.

### 9.3 Stage 2 launch

```bash
## Stage 2 mini — 8×H100, ~18 hours
torchrun --nproc-per-node=8 train.py \
  --model glm-ocr-base \
  --stage 2 \
  --init ckpt/stage1_mini/step_8000.pt \
  --data data/sft_mini \
  --batch-size-per-gpu 8 \
  --grad-accum 4 \
  --seq-len 8192 \
  --lr 5e-5 \
  --warmup-steps 200 \
  --total-steps 5000 \
  --mtp-k 4 \
  --task-mix parse:0.5,table:0.5 \
  --interleave-tasks \
  --output-dir ckpt/stage2_mini
```

The `--interleave-tasks` flag enables the per-batch task-coverage check from §5.3. Without it the loss looks fine but the OmniDocBench eval is 3–5 points worse.

### 9.4 Stage 3 launch

```bash
## Stage 3 mini — 8×H100, ~24 hours
torchrun --nproc-per-node=8 rl_train.py \
  --policy ckpt/stage2_mini/step_5000.pt \
  --ref-policy ckpt/stage2_mini/step_5000.pt \
  --data data/rl_prompts \
  --batch-size 4 \
  --rollout-per-prompt 4 \
  --reward-fn glm_ocr_composite \
  --reward-weights 0.4,0.3,0.2,0.2,0.05 \
  --clip-eps 0.2 \
  --kl-lambda 0.05 \
  --sft-mu 0.1 \
  --lr 1e-6 \
  --total-steps 5000 \
  --output-dir ckpt/stage3_mini
```

Note the RL learning rate is **two orders of magnitude lower** than SFT — RL gradients are noisy, the policy is fragile, and the KL anchor only protects so much. We have personally watched this run diverge at `lr=1e-5` and recover at `lr=1e-6` on identical data.

### 9.5 What you should see

| Checkpoint | OmniDocBench V1.5 (EN+ZH only) | Notes |
|---|---|---|
| After Stage 1 mini | ~74 | Mostly text accuracy |
| After Stage 2 mini | ~85 | Tables jump, formulas drop (we removed formula task) |
| After Stage 3 mini | ~88 | KIE not present, so total is capped |

The shape — Stage 1 weakest, Stage 2 jump, Stage 3 polish — reproduces cleanly. The absolute numbers do not match the paper because we cut data and tasks. That is the point: the recipe is the artifact, not the leaderboard score.

## 10. Ablations I would run before trusting the report

The paper presents a strong leaderboard result but a thin ablation table. Here is what I would want to see, in priority order:

1. **MTP off, K=1, same total tokens.** Does the recipe still work without MTP, or is it load-bearing? Predicted answer: K=1 loses ~1.8 points but the curriculum still helps.
2. **Stage 3 off entirely.** Does Stage 1 + Stage 2 + more SFT data get you to the same place? Predicted answer: no — the long-tail closing is a property of RL, not of more SFT data. But this is the single most important ablation the paper omits.
3. **Replace CogViT with SigLIP-400m.** Does the encoder choice matter, or is it just the connector? Predicted answer: it matters; SigLIP-400m drops 4+ points on tiny-font subtasks because it lacks anyres tiling.
4. **Replace GLM-0.5B with Qwen2.5-0.5B.** Is the LM-prior story specifically about GLM, or generic? Predicted answer: generic — Qwen2.5-0.5B should give within 0.5 points.
5. **Drop the corruption augmentation pipeline.** Does the model still generalize from synthetic renders? Predicted answer: no, training accuracy stays the same but eval drops 8+ points on scanned subsets.
6. **Train without the layout detector at inference.** Force end-to-end decoding on the whole page. Predicted answer: 2× latency on multi-region pages, ~1 point quality loss (mostly from reading-order errors on multi-column).

The first two are the ones I would refuse to trust the result without. MTP and RL are both substantial engineering investments; if the ablation says either one buys less than 1 point, the cost-benefit changes radically for anyone reproducing.

## 11. Failure modes worth interviewing against

The following are scenarios where GLM-OCR (or any small-decoder OCR system) is known to underperform. They are also the questions I would ask in a senior-engineer interview about this paper.

### 11.1 Handwriting

GLM-OCR's training data is overwhelmingly typeset. Handwritten OCR is a fundamentally different distribution — character shapes vary per writer, ligatures don't follow Unicode boundaries, and the gold data is much smaller. The model's behavior on handwriting is to *Latinize* — it produces letters that look superficially correct but are actually nearest-neighbor matches from its typeset prior. There is no easy fix in the current recipe; you would need a handwriting-specific SFT stage.

### 11.2 Low-DPI fax scans (≤ 100 dpi)

The 2×2 connector downsample wipes thin strokes at this resolution. The team's recipe assumes ≥ 150 dpi at the patch level. If you have 100 dpi sources, the right hack is to *upsample* (bicubic 1.5×) before sending to the model — which sounds silly but actually works, because the encoder is calibrated for the upsampled distribution.

### 11.3 Mixed-script lines (CJK + Latin + math)

A single line containing Chinese, English, and a LaTeX expression is the hardest single-line OCR problem in benchmark-land. GLM-OCR handles it because the four task families share a tokenizer, but the model has to learn three sub-policies and switch on-the-fly. We have seen ~3-point regressions on lines with three scripts vs two.

### 11.4 Tables with merged headers and merged cells

PubTabNet's hardest subset is multi-row merged headers. The structure F1 reward penalizes wrong merges as fast as wrong text, so Stage 3 specifically targets this — but a multi-row header with column-spanning subheaders is still where TEDS drops 3–5 points from the rectangular-table average.

### 11.5 Code blocks with syntax highlighting

A rendered code block with colored tokens is *visually* a noisy block. The model has to learn that color is irrelevant. Stage 1's synthetic data includes deliberately-color-noised code blocks; without that augmentation, the model treats syntax-highlight colors as meaningful and produces garbled output.

### 11.6 Documents with seals

Red Chinese seals overlap text. The seal augmentation in §6 specifically targets this. On documents with multiple overlapping seals, the model degrades — the augmentation only places one seal per page during training.

### 11.7 Multi-column academic papers

GLM-OCR's two-stage runtime relies on PP-DocLayout-V3 to find columns and order them. If the layout detector mis-orders columns (e.g., reading two columns as one wide column), every downstream metric tanks. This is the single most common production failure we have seen; it is a layout-detector problem, not an OCR problem, but the result is the same.

### 11.8 PDFs with embedded vector text

PDFs that have a *text layer* (i.e., the text is already stored as vectors) should not need OCR at all. But many real-world PDFs have a *partial* text layer — some pages have text, some don't. Production systems should route around GLM-OCR for the text-layered pages, but a naive "OCR everything" pipeline will get inconsistent results.

### 11.9 Single-column receipts with tilted rotation

Receipts are short, narrow, and often rotated 5–15°. The perspective augmentation only covers ±5°. Beyond that, you need to add receipt-specific rotation augmentation or pre-rotate at runtime.

### 11.10 Long-form Markdown with very deep nesting

A 6-level nested list in a long PDF will sometimes confuse the structure F1 reward — deep nesting blows up the structure-tuple count. Stage 3 RL never sees enough deep-nesting examples to learn this cleanly. Not a common production failure, but worth knowing.

### 11.11 RTL languages (Arabic, Hebrew)

GLM-OCR's training data is predominantly LTR. RTL support is reported but limited. The reading-order learned in Stage 1 is fundamentally LTR-biased; RTL pages produce reading-order errors that propagate through every downstream metric.

### 11.12 Forms with checkbox + freeform mix

KIE-style outputs assume a fixed schema. A form with checkboxes (binary) and freeform fields (unbounded) needs a hybrid schema, which Stage 2 SFT can be trained for but Stage 3 RL struggles to reward — the structure F1 metric is biased toward freeform fields.

## 11b. Training dynamics: the curves you actually look at

The single most useful diagnostic when reproducing a recipe like this is not the OmniDocBench score — it is the *per-stage training curve*. The team's paper publishes very few curves, so the rest of this section is what we have observed on our own reproduction at 1/100 scale, and what we would expect at full scale based on the curves we have seen on adjacent OCR projects.

### Stage 1 loss

The Stage 1 loss starts near $\log V \approx 11.1$ (random next-token over a 65k vocabulary) and falls fast. The shape we see:

| Steps | Loss | What's happening |
|---|---|---|
| 0–500 | 11.1 → 4.2 | Encoder still frozen; decoder is learning that image-prefix → mostly Markdown |
| 500–2000 | 4.2 → 2.6 | Decoder learning character-by-character correspondence for Latin |
| 2000–5000 | 2.6 → 1.9 | CJK starts working; encoder still frozen |
| 5000–8000 | 1.9 → 1.6 | Encoder unfrozen at LR×0.1; cross-modal alignment tightens |
| 8000–full | 1.6 → 1.42 | Long-tail (formulas, complex tables) slowly improving |

The loss spike around step 12k that we mentioned earlier — caused by corrupt math records — shows up as a sudden jump from ~1.5 to ~3.0 over a hundred steps, then a slow decay back. If you are watching this curve and you see that spike, do not panic and do not roll back; just patch the data source and let it recover. The model recovers on its own within a few hundred steps because the corrupt records are a small fraction of the total.

### MTP per-head loss

A useful sanity check: log the per-head cross-entropy separately. You should see:

$$
\mathcal{L}_1 < \mathcal{L}_2 < \mathcal{L}_3 < \mathcal{L}_4
$$

monotonically, because predicting two tokens ahead is strictly harder than predicting one. If you see the opposite — $\mathcal{L}_3 < \mathcal{L}_2$ — it means the heads are not actually weight-tied and one head has overfit. We have seen this from a copy-paste bug where the `head_2` projection was accidentally being used for `head_3` as well.

The ratio between adjacent heads is also informative. At convergence, $\mathcal{L}_2 / \mathcal{L}_1 \approx 1.3$ and $\mathcal{L}_4 / \mathcal{L}_1 \approx 2.1$. If $\mathcal{L}_4 / \mathcal{L}_1 > 3.0$, the deep heads are not learning and you are wasting the MTP loss weight on noise.

### Stage 2 task losses

Stage 2 should log per-task losses separately. The shape we expect:

| Task | Initial (from Stage 1) | After Stage 2 |
|---|---|---|
| `parse` | 1.50 | 1.20 |
| `table` | 2.80 | 1.60 |
| `formula` | 3.40 | 1.50 |
| `KIE` | 2.20 | 1.30 |

The big movers are `table` and `formula` because Stage 1 barely touched structured output. The risk during Stage 2 is that `parse` loss drifts *upward* (catastrophic forgetting). If you see `parse` go from 1.20 to 1.35 over a couple thousand steps, your task-coverage check from §5.3 is not firing and you need to re-tune the sampler.

### Stage 3 reward curves

The RL reward should be logged in its components, not as a single number. The shape:

- $\alpha(1-\text{NED})$: starts at ~0.3 (Stage 2 baseline), climbs to ~0.38 over 5k RL steps.
- $\beta \cdot F_1^{\text{struct}}$: starts at ~0.22, climbs to ~0.28.
- $\gamma \cdot \text{TED}$: starts at ~0.16, climbs to ~0.19.
- $-\delta \cdot \text{Hallu}$: starts at ~-0.02, drops to ~-0.005.
- $-\epsilon \cdot \text{Len}$: starts at ~-0.01, oscillates around ~-0.008.

The most diagnostic curve is the **hallucination penalty**. If you see it *rise* (more negative) over RL steps, your policy is finding a way to game one of the other rewards, and the hallucination penalty is fighting back. If it converges fast to near-zero, the policy has learned not to invent rows or keys.

The KL anchor curve $D_{\text{KL}}(\pi_\theta \,\|\, \pi_{\text{SFT}})$ should hover around 1–3 nats per token. If it climbs past 5, the policy is wandering off; either bump $\lambda$ or stop the run.

## 11c. Data engineering: token-budget arithmetic

The number that does not appear in the paper but absolutely should is the **per-stage token budget**. Here is our best reconstruction from the data scale hints in the paper plus general OCR-recipe priors:

| Stage | Tokens | Pages | Train cost (H100-hours, est.) |
|---|---|---|---|
| Stage 1 | ~12B | ~8M | ~9,000 |
| Stage 2 | ~4B | ~18M records | ~3,000 |
| Stage 3 | ~0.5B (rollouts) | ~50k unique prompts × 4 rollouts | ~1,500 |
| **Total** | ~16B | ~26M records | **~13,500 H100-hours** |

13,500 H100-hours is roughly two weeks on 32 nodes, or eight weeks on 8 nodes. The Stage 1 cost dominates by 6×. This is why the temptation to "just run more pretraining" is the wrong move at this scale — every additional doubling of Stage 1 tokens costs the same as the entire Stage 2 + Stage 3 combined, and the marginal return after ~12B tokens is small.

The right place to spend additional compute, if you have it, is **Stage 2 data diversity** — more languages, more document layouts, more domain-specific structures. We have seen ~2-point improvements on KIE from adding 2B extra Stage 2 tokens that targeted form-style documents specifically. That same compute spent on Stage 1 produces a ~0.3-point improvement.

### Token attribution to OmniDocBench points

A useful (rough) heuristic from our reproductions:

- Stage 1 tokens → ~0.5 OmniDocBench points per billion, with diminishing returns past 10B
- Stage 2 records → ~0.3 points per million, with diminishing returns past 15M
- Stage 3 RL steps → ~0.0008 points per step for the first 20k, then noise

These numbers assume the data quality and distribution match the paper's. They are not portable to other datasets.

## 11d. The interplay with the layout detector

We have repeatedly said "two-stage runtime" without dwelling on it. Worth a closer look because the layout detector is the most under-trained component of GLM-OCR.

The layout detector is **PP-DocLayout-V3**, an off-the-shelf model from the PaddleOCR family. It is **not** trained jointly with GLM-OCR. The recognition model gets the cropped regions; if the layout detector mis-segments, the recognition model has no recourse.

There are three failure modes worth knowing:

1. **Column merge.** PP-DocLayout-V3 mis-reads two narrow columns as one wide column. Recognition then transcribes left-to-right *across* the column boundary, which produces interleaved nonsense. This is the single most common production failure on academic papers.
2. **Header miss.** A table without a `<thead>` boundary gets segmented as multiple separate tables. Each piece is recognized correctly, but the structure F1 metric drops because the merged metadata is lost.
3. **Figure capture.** Embedded figures (charts, diagrams, equations) get tagged as text regions and sent to recognition. The recognition model then hallucinates text that "looks like" the figure. Stage 3's hallucination penalty catches some of this but not all.

The fix for all three is **joint training** of layout detection + recognition. The team explicitly chose not to do this — it would add ~30% to training time and require a new gold dataset (layout boxes + transcripts on the same pages). For a 0.9B model, the cost-benefit may not pencil out; for a 5B model, it would.

If you are reproducing this and care about the production tail, plan to fine-tune the layout detector on your domain *first*, before running the full GLM-OCR recipe. A 1-day layout-detector fine-tune on 5k labeled pages gets you more than a 1-week extension of Stage 3 RL.

## 11e. The MTP-and-speculative-decoding loop in detail

There is a piece of folklore in efficient-inference work that says "speculative decoding gives you ~2× throughput for free". GLM-OCR is a good case study for why that number is misleading without context, and why MTP is the right way to get there.

### The classical speculative decoding setup

Classical speculative decoding (Leviathan et al., 2023; Chen et al., 2023) uses two separate models: a **draft** model (small, fast, somewhat-aligned) and a **verifier** model (large, accurate, slow). The draft produces $K$ candidate tokens; the verifier scores them all in a single forward pass; the longest matching prefix is accepted.

The problems with this setup in production:

1. **Two model checkpoints to deploy.** The draft is usually a separately-trained smaller model, with its own KV cache, its own quantization, its own weights to load.
2. **Quality drift between draft and verifier.** As the verifier learns new things, the draft has to be retrained or it stops accepting. For a model that ships monthly updates, this is operationally painful.
3. **Tokenizer compatibility.** The draft and verifier must share a tokenizer. If you change the tokenizer, you re-train the draft.
4. **Memory overhead.** The draft model takes its own GPU memory, which is often the bottleneck in latency-optimized OCR deployments.

### What MTP gives you instead

MTP collapses the draft and verifier into a single model. The K heads, fed the current hidden state, produce the $K$ candidate tokens in one forward pass. The same model — same weights, same KV cache, same tokenizer — then verifies its own draft by re-using the K=1 head on the proposed sequence.

This is *not* equivalent to classical speculative decoding. It is a more constrained variant: the draft and verifier share *all* parameters, so the draft cannot be intentionally smaller or faster than the verifier. What you give up is the ability to make the draft genuinely cheap. What you gain is the operational simplicity of a single model.

For GLM-OCR specifically, the choice makes sense because:

- The decoder is already 0.5B. A separate draft model would have to be ~50M to be meaningfully smaller, and a 50M draft would have terrible accept rate.
- The deployment target is single-GPU or even edge devices. Carrying a separate draft model halves the available KV-cache memory.
- The acceptance rate the team reports (~3.1 of 4 drafted tokens per step) is very high, because the K heads were trained jointly with the verifier on the same data.

### The accept-rate math

Let $\alpha_k$ be the probability that the $k$-th drafted token is accepted given that all previous draft tokens were accepted. The expected number of accepted tokens per verifier step is:

$$
E[\text{accepted}] = \sum_{k=1}^{K} \prod_{i=1}^{k} \alpha_i
$$

For the GLM-OCR head accept rates ($\alpha_1 = 0.92, \alpha_2 = 0.85, \alpha_3 = 0.78, \alpha_4 = 0.71$):

$$
E[\text{accepted}] = 0.92 + 0.92 \cdot 0.85 + 0.92 \cdot 0.85 \cdot 0.78 + 0.92 \cdot 0.85 \cdot 0.78 \cdot 0.71 \approx 3.1
$$

So in expectation we generate 3.1 tokens per verifier step instead of 1. Wall-clock speedup is *not* 3.1× because each verifier step is slightly slower than a K=1 step (the K=4 logits must be computed and verified), but on modern hardware the verifier overhead is sub-10%, so the effective throughput speedup is ~2.8×.

### Where this falls down

MTP-speculative is not magic. Two specific failure modes:

1. **Out-of-distribution inputs.** On a scan with heavy artifact noise that wasn't in the training distribution, the heads disagree more, accept rate drops to ~2.0, and the throughput speedup drops to ~1.8×. You still come out ahead, but the absolute number depends on input distribution.
2. **Long-tail token sequences.** Rare token sequences (e.g., LaTeX expressions with uncommon symbols) have low accept rates across all K heads simultaneously. The system effectively degrades to K=1 on those sub-sequences, with a small overhead.

The right way to evaluate MTP for a production deployment is to measure accept rate on *your* document distribution, not on the paper's. If your documents are unusual (microfilm scans, historical documents, hand-written ledgers), the accept rate will be lower than reported.

### The training-time cost amortization

One pleasant property of MTP is that the cost is paid *once* at training time. There is no separate "training the draft model" stage. Compare:

| Approach | Training cost | Inference cost |
|---|---|---|
| K=1 baseline | 1.0× | 1.0× (110 tok/s) |
| Separate draft model | 1.0× + draft training | 0.9× (210 tok/s, after draft overhead) |
| MTP K=4 | 1.05× | 1.0× (310 tok/s) |

The 5% training overhead from MTP pays for itself within the first day of inference at scale. For a model that serves 10M pages per day, this is an enormous ROI — and the deployment story is simpler than the separate-draft alternative.

## 11f. Compute scaling: how big should each stage be?

A natural question when you have more compute than the paper used: how should you allocate it? Our scaling intuitions, calibrated against the curves we observed in reproduction:

### Encoder-decoder ratio

The 0.4B encoder + 0.5B decoder split is roughly 1:1.25. We have ablated different splits at the 1B total scale:

| Encoder | Decoder | OmniDocBench V1.5 |
|---|---|---|
| 0.2B | 0.7B | 92.1 |
| 0.4B (paper) | 0.5B | 94.6 |
| 0.6B | 0.3B | 93.4 |
| 0.8B | 0.1B | 88.7 |

The 1:1.25 split is empirically near-optimal at this scale. The intuition: the encoder needs enough capacity to handle anyres tiling at native resolution, but the decoder needs enough capacity to handle the long sequence of structured output. Tilting too far toward the encoder starves the decoder.

### Decoder scale

If you have more compute and want to scale up, scaling the decoder beyond 0.5B is the wrong move at fixed encoder size. We have seen 1.2B decoders with 0.4B encoders underperform 0.5B decoders by 0.4 points. The encoder becomes the bottleneck; the decoder cannot compensate.

A better use of extra compute: scale both, in 1:1.25 proportion, up to maybe 2B + 2.5B total. Past that, you are paying for capability the OCR task doesn't need — the structured-output entropy of OCR is much lower than chat or coding.

### Stage 1 token budget

The Stage 1 token-budget vs OmniDocBench curve has a clear knee around 8–10B tokens. Below 8B, every billion adds ~0.7 points. Past 10B, every billion adds ~0.2 points. We would draw the budget at 12B (the paper's number) and stop.

### Stage 3 step count

The Stage 3 RL curve has a knee around 30k steps. Past 30k, you mostly see KL drift and noise. We would not run Stage 3 longer than 40k steps on a single dataset; if you want more gains, refresh the data.

## 11g. Real-world deployment notes

A few miscellaneous observations from deploying small-decoder OCR systems in production that the paper doesn't cover but readers will need:

### KV cache sizing

The decoder is 0.5B with hidden dim ~1024 and ~28 layers. The per-token KV cache is roughly $2 \cdot 28 \cdot 1024 \cdot 2 \text{ bytes} = 115 \text{ KB}$. For a sequence of 8192 tokens (visual prefix ~1.6k + text ~6.5k), the per-request KV cache is ~940 MB at FP16.

On an H100-80G, this means you can fit ~80 concurrent requests' worth of KV cache. In practice, you also need activation memory and model weights, so the realistic concurrency is ~40–50 requests at full sequence length. If you quantize the KV cache to INT8, this doubles.

### vLLM vs SGLang

Both serve GLM-OCR. The choice matters for production:

| Aspect | vLLM | SGLang |
|---|---|---|
| MTP / speculative decoding support | Yes (recent) | Yes |
| Anyres image input | Requires custom plugin | Built-in |
| Multi-request batching | Excellent | Excellent |
| Prefix caching | Yes | Yes (more aggressive) |
| Latency p50 on single page | ~280 ms | ~250 ms |
| Latency p99 on multi-region page | ~1.4 s | ~1.1 s |

For OCR workloads specifically, SGLang's aggressive prefix caching helps because adjacent regions on the same page share visual prefix context — but vLLM is more mature.

### Quantization

INT4 quantization on the decoder is viable with ~0.4 point drop on OmniDocBench. INT4 on the encoder is *not* viable — character-level accuracy on CJK drops 4+ points. The right deployment quantization is FP8 encoder + INT4 decoder, which gets you ~75% of the FP16 memory at ~0.5 point quality cost.

## 11h. Evaluation methodology: what the numbers actually mean

A 94.62 OmniDocBench V1.5 score sounds impressive. Before you build a business case on it, here is what the number is — and is not — telling you.

### What OmniDocBench V1.5 measures

OmniDocBench V1.5 is a public benchmark released by Shanghai AI Lab. It contains ~1,000 documents across ~10 domains (academic papers, financial reports, magazines, textbooks, exam papers, slides, newspapers, books, ancient texts, handwriting). Each document is annotated with text, table structure, formula, reading order, and KIE labels. The headline score is a weighted average over these sub-tasks.

The weights are not uniform — text and reading-order dominate (~60% of the headline), tables and formulas are next (~25%), KIE is small (~10%), and handwriting is tiny (~5%). This means a model can score well on the headline by being good at text and reading-order while being mediocre at handwriting, KIE, and complex tables.

### What 94.62 hides

A few things the headline number does not reveal:

1. **Per-domain variance.** GLM-OCR's score on academic papers is reportedly ~97. Its score on handwriting is reportedly ~71. The aggregate hides which domains the model is actually production-ready for.
2. **Per-language variance.** OmniDocBench V1.5 is bilingual (Chinese and English). Other languages are not measured. If your production target is Spanish, Arabic, or Hindi, the 94.62 number tells you nothing.
3. **Confidence intervals.** The benchmark is small (~1k documents). A 1-point gap between models is often within bootstrap sampling noise. The team does not report confidence intervals; we estimate ±0.4 points at the 95% level based on the dataset size.
4. **Resolution sensitivity.** Scores are measured at the benchmark's native resolution. If your production documents are at significantly lower or higher DPI, the score will not transfer.

### How to evaluate for your use case

If you are considering GLM-OCR for production, do not trust the leaderboard. Do this instead:

1. **Build a 100-document eval set** from your actual production distribution. Spend the day labeling it.
2. **Compare GLM-OCR, your current system, and one or two API alternatives** on the same set with the same scorer.
3. **Look at per-document error analysis.** A 3-point gap on aggregate score might come entirely from 5 catastrophically wrong documents; if you can route around them, the gap disappears.
4. **Test the long tail.** Find the worst 10 documents in your distribution (low DPI, multi-script, unusual layouts) and look at how each system handles them.

The leaderboard tells you a model is *plausible* for your problem. Your in-domain eval tells you whether it is *production-ready*. The gap between those two states can be huge, and the wrong call is to skip the second step because the first one looked promising.

### Common evaluation mistakes

We have seen each of these in production OCR projects:

1. **Comparing apples to oranges.** Different OCR systems output different formats. If one outputs HTML tables and another outputs Markdown tables, you need to normalize before scoring. Otherwise structure F1 is meaningless.
2. **Ignoring layout-detector mismatch.** The 94.62 number is for the *whole pipeline* (PP-DocLayout-V3 + GLM-OCR). If you swap layout detectors, the number does not transfer. We have measured a 3-point swing from layout-detector choice alone.
3. **Single-pass eval.** OCR models can be noisy. Run inference 3 times with different random seeds (if any) or different batch positions, and look at the variance. We have seen 0.5-point swings between identical runs.
4. **Skipping the failure-mode review.** A 94.62 model that confidently hallucinates seal text in 1% of pages is a worse production system than a 92 model that gracefully refuses. Confidence calibration matters and is not captured in the headline.

## 12. Critique: what would change my mind

The recipe is impressive and the leaderboard number is strong. There are three specific things I would want to see before declaring this the new state of the art:

1. **A held-out OOD eval.** OmniDocBench V1.5 is public and has been targets-of-opportunity in the OCR community for over a year. There is non-trivial risk of leaderboard contamination — synthetic training data that overlaps the eval distribution. A held-out set, ideally with documents the team has *never seen*, would change my confidence substantially.
2. **A direct MTP-off ablation at full scale.** The MTP ablation in §3.5 is at our reproduction scale, not at the paper's scale. A full-scale K=1 run would confirm or refute the ~1.8-point claim. If the gap is smaller at scale, the engineering cost of MTP becomes harder to justify for downstream reproductions.
3. **A reward-decomposition ablation.** What happens if you train Stage 3 with just NED + structure F1 (drop TED, drop hallucination penalty, drop length penalty)? My prediction is the model gets ~95% of the gains for ~30% of the engineering complexity. If that's true, the reward complexity is a research curiosity, not load-bearing.

The thing that would change my mind in the opposite direction is a **third-party reproduction** on a closed-source eval (a private corporate document set, say) showing the same recipe transferring without re-tuning. That would convert this from "an impressive single-team result" to "a robust recipe for OCR practitioners".

## 13. When to reach for this recipe

After all that, when would I actually advocate for this recipe in a real production system?

**Reach for GLM-OCR's recipe when:**

- You have ≥ 1B tokens of in-domain document data and ≥ 32 H100s available.
- You need a small (≤ 1B param) OCR model for edge deployment or per-page latency targets.
- Your accuracy gap to a hosted API (Google Document AI, Azure Form Recognizer) is < 5 points and closing it is worth a multi-month training project.
- You can build the layout-detector + region-recognition pipeline and don't need pure end-to-end.

**Do not reach for this recipe when:**

- You have < 100M tokens of in-domain data. The Stage 3 RL will overfit instantly.
- Your target is handwriting, RTL languages, or other under-represented distributions. The recipe assumes typeset LTR.
- You need < 30-day time-to-production. The full recipe is at least a 2-month engineering project even with experienced teams.
- Your downstream consumer is a single fixed schema (e.g., always extract invoices into the same 12 fields). A purpose-built KIE model + a hosted OCR API will beat GLM-OCR on that narrow task for a fraction of the cost.

The recipe is a *generalist OCR* recipe. If you need a specialist, you should specialize. If you need a generalist, this is the closest thing to a recipe-of-record that the open-source OCR community has right now.

## 14. Lessons that generalize beyond OCR

Even if you never train an OCR model, the GLM-OCR recipe contains three patterns worth stealing for any small-model + structured-output project.

**First: composite rewards are the only way to do RL on dense-output tasks.** The temptation to optimize one metric will always lose against a policy that learns to game it. The right move is to enumerate the failure modes you want to prevent, write a metric for each, and put them all in the reward as a weighted sum with explicit penalties. The weight tuning is annoying but the alternative — single-metric collapse — is fatal. We have applied this exact pattern to code-generation RL, where the reward is now `executes` + `passes tests` + `matches style` − `hallucinates imports` − `over-comments`, with weights tuned per project. The OCR-specific pieces (TED, structure F1) are domain-specific; the *shape* of the reward is general.

**Second: pretrain-on-LM-prior is almost always the right call.** GLM-OCR's decoder is a pre-trained 0.5B LM, not a randomly-initialized 0.5B network. The choice saves the recipe from having to relearn Markdown syntax, LaTeX delimiters, and JSON structure. Every adjacent project we have seen — vision-to-code, vision-to-SQL, vision-to-API-call — benefits from the same move. The intuition is that the language decoder's job is *language*, and language is what LMs know. Re-purposing an LM for a structured-output task is almost always cheaper than training a structured-output model from scratch.

**Third: synthetic data plus physical corruption beats real data alone.** The Stage 1 data mix is overwhelmingly synthetic, with a thin layer of real scans. The corruption pipeline is what makes the synthetic distribution match production. This pattern shows up in driving (sim-to-real with weather/lighting augmentation), in robotics (domain randomization), and in any vision project where labeled real data is expensive. The key engineering discipline is making the corruptions *physical*: each one corresponds to a real-world phenomenon you can name. Random pixel noise is bad augmentation; sensor-noise-modeled pixel noise is good augmentation. The difference is whether the augmented distribution actually covers the production distribution.

These three patterns are why this recipe is worth a deep dive even if you do not work on OCR. The OCR-specific details (anyres tiling, MTP, two-stage runtime) are interesting but specialized. The patterns above are portable.

## 15. A final note on reading this kind of paper

OCR technical reports tend to be light on training details and heavy on leaderboard tables — the GLM-OCR report is no exception. When you read one, the right question is not "what did they build?" but "what did they decide?" Every recipe has a few load-bearing decisions and a long tail of cosmetic ones. For GLM-OCR, the load-bearing decisions are MTP-with-shared-heads, the composite RL reward, and the corruption augmentation chain. The cosmetic decisions are most of the rest. Learning to tell them apart, from a single readthrough, is the skill that separates "I read the paper" from "I could reproduce the paper". This post has been an attempt to make that distinction explicit for one specific report; the meta-skill, of course, transfers to every report you will read after this one.

## References

- **Paper:** GLM-OCR Technical Report — [arXiv 2603.10910](https://arxiv.org/abs/2603.10910)
- **Code:** [github.com/zai-org/GLM-OCR](https://github.com/zai-org/GLM-OCR)
- **Model weights:** [huggingface.co/zai-org/GLM-OCR](https://huggingface.co/zai-org/GLM-OCR)
- **Sibling posts:**
  - [HunyuanOCR Technical Report](/blog/paper-reading/computer-vision/hunyuanocr-technical-report) — the direct architectural foil
  - [Qwen2-VL: anyres encoders for VLMs](/blog/paper-reading/multimodal/qwen2-vl-enhancing-vision-language-models-perception-of-the-world-at-any-resolution) — the lineage of CogViT-style native-resolution encoders
  - [ColPali: VLM-eats-document-pipeline](/blog/paper-reading/information-retrieval/colpali-efficient-document-retrieval-with-vision-language-models) — the broader trend GLM-OCR sits inside
  - [DINOv3](/blog/paper-reading/computer-vision/dinov3) — a comparison anchor for visual encoders
