---
title: "Seed1.5-VL: native-resolution vision and a hybrid-RL recipe for a 20B-active multimodal model"
date: "2026-06-10"
publishDate: "2026-06-10"
description: "A deep read of Seed1.5-VL: a from-scratch native-resolution ViT, dynamic frame-resolution video tokenization, and a single PPO loop that mixes RLHF and RLVR to make a 20B-active MoE VLM act like a frontier agent."
tags: ["seed1.5-vl", "vision-language-model", "multimodal", "native-resolution", "vision-transformer", "moe", "reinforcement-learning", "gui-agent", "video-understanding", "bytedance-seed"]
category: "paper-reading"
subcategory: "Multimodal"
author: "Hiep Tran"
featured: true
readTime: 75
---

There is a quiet tax that almost every vision-language model pays, and almost no benchmark line item shows it. The tax is resolution. You take a model whose vision encoder was trained at a fixed 224×224 or 336×336 grid — a CLIP or a SigLIP checkpoint that the field standardized on years ago — and you ask it to read a dense invoice, follow a 4K desktop screenshot, or count the rivets on a bridge. The encoder cannot natively see that detail, so the surrounding system improvises: it tiles the image into a grid of crops, runs the fixed-res encoder on each crop, and stitches the embeddings back together with positional hacks. Every tile boundary is a place where a character can be cut in half, a small object can fall between two crops, and the model's sense of global layout frays. The tiling machinery is the single most load-bearing, least-loved piece of plumbing in modern VLMs, and it exists entirely because the vision encoder was never built to see at native resolution in the first place.

Seed1.5-VL, the vision-language model from ByteDance's Seed team (arXiv:2505.07062, shipped as `doubao-1-5-thinking-vision-pro`), starts from the opposite premise. Instead of retrofitting a fixed-resolution CLIP encoder and papering over its limits with tiling, the team trains a **532M-parameter vision transformer (Seed-ViT) from scratch** to ingest images at their native resolution, with 2D rotary position embeddings so that a 700×1300 receipt and a 4096×2160 game frame both flow through the same encoder without cropping. That encoder feeds a conventional adapter into a **Mixture-of-Experts language model with roughly 20B active parameters**, and the whole stack — despite being small enough to serve interactively — sets state of the art on **38 of 60 public benchmarks** (21 of 34 image VL tasks, 14 of 19 video tasks, 3 of 7 GUI-agent tasks), matches or beats Gemini 2.5 Pro, GPT-4o, Claude 3.7, OpenAI o1, and Qwen2.5-VL-72B across a wide swath, and acts as a genuinely competent GUI and gameplay agent — beating OpenAI's Computer-Using Agent and Claude 3.7 on screen grounding.

<!-- FIGSPEC 1
kind: pipeline
claim: Seed1.5-VL routes native-resolution pixels through a 532M ViT and a 4x-pooling adapter into a 20B-active MoE LLM in one forward path.
caption: A compact 532M encoder and a deliberately boring adapter front a 20B-active MoE LLM, so high-resolution images arrive pre-compressed.
nodes:
  - id: a | label: "Pixels (any res)" | color: gray
  - id: b | label: "Seed-ViT 532M\npatch 14, 2D RoPE" | color: blue
  - id: c | label: "2x2 avg-pool\n+ 2-layer MLP\n(4x fewer tokens)" | color: green
  - id: d | label: "MoE LLM\n~20B active" | color: blue
  - id: e | label: "Text / actions" | color: gray
edges:
  - a -> b | label: "resize to 28x"
  - b -> c | label: "patch embeds"
  - c -> d | label: "vision tokens"
  - d -> e
notes: left-to-right pipeline; emphasize 4x token cut at adapter before LLM
-->
![Seed1.5-VL end-to-end architecture: a 532M native-resolution Seed-ViT feeds a 2x2-pooled MLP adapter into a 20B-active MoE language model](/imgs/blogs/seed1-5-vl-native-resolution-vision-language-1.png)

The diagram above is the mental model, and the whole article is a tour of it. Pixels enter on the left at whatever resolution they arrive in. Seed-ViT (532M) turns them into patch embeddings using 2D RoPE so position is encoded continuously rather than via a fixed learned table. A deliberately boring adapter — a 2×2 average-pool followed by a two-layer MLP — compresses four patch tokens into one and projects them into the language model's embedding space, cutting the vision-token count by 4× before a single decoder layer runs. Then a decoder-only MoE LLM with ~20B active parameters, initialized from an internal text model trained on trillions of tokens, does all the reasoning. Three things make this model interesting and none of them is the adapter: the **from-scratch native-resolution encoder**, the **dynamic frame-resolution video tokenizer** that lives in front of that encoder for video, and the **hybrid reinforcement-learning recipe** that runs RLHF and RLVR inside a single PPO loop with a shared critic. We will spend most of our time on those three.

> [!tldr] TL;DR
> - **What it claims:** A general-purpose VLM with a compact 532M native-resolution ViT and a 20B-active MoE LLM sets SOTA on 38/60 public benchmarks and matches frontier closed models (Gemini 2.5 Pro, GPT-4o, Claude 3.7, o1) while staying small enough to serve interactively — and it is a strong GUI/gameplay agent on top.
> - **Why it matters:** It is an existence proof that training the vision encoder *from scratch* for native resolution (rather than retrofitting fixed-res CLIP/SigLIP) plus a hybrid RLHF+RLVR PPO loop beats the conventional "big encoder, big LLM, tiling" recipe at a fraction of the encoder size — Seed-ViT matches InternVL-C-6B zero-shot at 9% of its parameters.
> - **Most surprising finding:** A single PPO loop that mixes generative-reward-model RLHF with verifier-based RLVR, sharing one critic and scoring only the post-`</think>` solution, improves *non-thinking* responses even though RL trained only on LongCoT — and vision-centric chain-of-thought ("let me look at the image again") emerged in RL with no SFT labels teaching it.
> - **Where it fails:** Counting on occluded or similar-colored objects, subtle inter-image differences, spatial relations under changing perspective, higher-level combinatorial reasoning (Klotski, mazes), 3D projection, and hallucination when pixels conflict with the LLM's text priors. The authors attribute the broad-knowledge gaps to the 20B-active scale — the loss had not saturated at 3T tokens.

## 1. Why native resolution is the whole ballgame

Let us be precise about the failure mode, because it motivates every design decision downstream. A fixed-resolution vision encoder maps any input to a fixed grid — say 16×16 = 256 patches at 224px with patch size 14. If your image is a 2480×3508 A4 scan of a contract, the resize-to-224 step throws away roughly 99.6% of the pixels before the encoder ever runs. The 8-point footnote font becomes a smear. The standard mitigation, popularized by InternVL and LLaVA-style tiling and refined in models like [DeepSeek-VL2's dynamic tiling](/blog/machine-learning/computer-vision/deepseek-vl-vl2-dynamic-tiling-moe), is to chop the image into a grid of native-resolution tiles, encode each tile at 224 or 448, plus one downsized thumbnail for global context, and concatenate. It works, but it is a hack with three structural costs.

First, **tile boundaries are information-destroying seams**. A character, a chart gridline, or a small object that straddles two tiles is split across two independent encoder passes that never attend to each other inside the encoder. Second, **token budget explodes non-linearly**. A 12-tile decomposition of a 4K screenshot produces 12×256 = 3072 vision tokens before the thumbnail, and the LLM pays full attention cost on all of them. Third, **positional coherence is synthetic**. The model has to learn, from the tile-index embeddings you bolt on, how 12 independently-encoded crops reassemble into one scene — a task the encoder itself never had to solve.

[Qwen2-VL](/blog/paper-reading/multimodal/qwen2-vl-enhancing-vision-language-models-perception-of-the-world-at-any-resolution) was the model that made the field take native resolution seriously as a *first-class* property rather than a tiling trick, using 2D-aware position encoding and dynamic resolution. Seed1.5-VL takes the same conviction further down the stack: it does not adapt an existing fixed-res encoder at all. It builds Seed-ViT from a standard ViT, from scratch, so that native resolution is not a finetuning afterthought but the encoder's native language.

<!-- FIGSPEC 2
kind: before-after
claim: A retrofitted fixed-resolution CLIP encoder loses detail at tiling seams while Seed-ViT ingests whole native-resolution images with 2D RoPE.
caption: Tiling a fixed-res encoder destroys detail at every seam; a from-scratch native-res encoder with 2D RoPE has no seams to begin with.
nodes:
  - id: a | label: "BEFORE: fixed-res\nCLIP/SigLIP" | color: red
  - id: b | label: "resize 224 +\ntile into crops" | color: red
  - id: c | label: "seams split\nchars/objects" | color: red
  - id: d | label: "AFTER: Seed-ViT\nfrom scratch" | color: green
  - id: e | label: "resize to 28x,\none pass" | color: green
  - id: f | label: "2D RoPE,\nno seams" | color: green
edges:
  - a -> b
  - b -> c | label: "detail loss"
  - d -> e
  - e -> f | label: "82.5 zero-shot"
notes: two horizontal lanes, red top (before), green bottom (after); contrast seams vs single pass
-->
![Before vs after: a retrofitted fixed-resolution CLIP/SigLIP encoder loses detail through tiling and resize, while Seed-ViT ingests native resolution from scratch with 2D RoPE](/imgs/blogs/seed1-5-vl-native-resolution-vision-language-2.png)

The before/after above names the bet. On the left is the conventional path: a fixed-res CLIP or SigLIP backbone, a resize-and-tile front end, learned absolute or interpolated position embeddings that have to be stretched to fit, and detail loss at every seam. On the right is Seed1.5-VL: bilinear resize to the nearest multiple of 28 (so the image divides evenly into 14×14 patches), one encoder pass over the whole image, and 2D RoPE that encodes each patch's (row, column) position continuously, so a 40×60 patch grid and a 16×16 patch grid use the *same* position machinery with no interpolation. The claim that this matters in numbers: Seed-ViT scores **82.5 average zero-shot accuracy across 6 ImageNet variants** while being roughly **9% the parameter count of InternVL-C-6B** that it matches, and stays competitive with EVA-CLIP-18B — a model **30× larger**. That is the headline efficiency result, and it is what justifies building an encoder from scratch instead of downloading one.

There is one design choice worth defending explicitly, because a vocal subset of the field argues the opposite: Seed1.5-VL keeps a *vision encoder at all*. Encoder-free VLMs (which feed raw patches straight into the LLM) are fashionable, but the Seed team rejected that route, and the reasoning is purely about token economics. An encoder is an efficient compressor: it does the heavy spatial pooling once, cheaply, in a 532M model, so that high-resolution inputs arrive at the expensive 20B-active LLM already compressed. Without the encoder, every high-res image dumps tens of thousands of raw patches onto the LLM's context, and you pay the LLM's per-token cost on all of them. The encoder is not legacy baggage; it is the thing that makes native resolution affordable.

| Strategy | How high-res is handled | Tokens for a 4K screenshot | Seam/positional cost |
|---|---|---|---|
| Fixed-res CLIP + tiling (InternVL-style) | Grid of 12+ crops + thumbnail | ~3072+ | Hard seams, synthetic tile-index positions |
| Fixed-res + dynamic tiling (DeepSeek-VL2) | Adaptive crop count + MoE LLM | Variable, still tiled | Softer but still seams |
| Native-res encoder, retrofitted (Qwen2-VL) | One pass, dynamic resolution, 2D-aware | One sequence | No seams; encoder was adapted, not built for it |
| **Native-res from scratch (Seed-ViT)** | One pass, resize-to-28-multiple, 2D RoPE | One sequence, 2×2-pooled | No seams; native from pretraining |

## 2. The architecture: Seed-ViT, a boring adapter, and a 20B-active MoE

The three-part decomposition is conventional on purpose — the team spends its novelty budget on the encoder and the RL, not on the connective tissue. Let us walk each component with its real hyperparameters.

**Seed-ViT (532M).** This is a standard vision transformer, not a CLIP or SigLIP finetune. The architecture: patch size **14**, embedding dimension **1280**, depth **27** layers, **20** attention heads with head dimension **64**, MLP expansion ratio **4.0**, and crucially **2D RoPE** in place of learned absolute position embeddings. The 2D RoPE is what lets a single set of weights handle arbitrary aspect ratios and resolutions: instead of a position table sized for one grid, each patch's position is injected via rotary phases computed from its 2D coordinate, so there is nothing to interpolate when the grid changes shape.

The native-resolution preprocessing is mechanical but worth stating exactly. An input image is **bilinearly resized to the nearest multiple of 28×28** — 28 because that is 2 patches of 14 in each dimension, which keeps the later 2×2 pool clean — then split into **14×14 patches**. Multiple images in one request are packed **NaViT-style** into a single token sequence, with attention masks that isolate each image so that patches from image A never attend to patches from image B. This packing is what makes batched multi-image and video inference efficient: you fill the sequence to capacity with patches from many images and let the mask enforce boundaries, rather than padding each image to a fixed length.

```python
import torch
import torch.nn.functional as F

PATCH = 14
ALIGN = 28  # nearest-multiple alignment so a clean 2x2 pool is possible later

def resize_to_native(img, max_pixels=None):
    """Bilinear-resize to the nearest 28x28 multiple (Seed-ViT preprocessing)."""
    _, _, h, w = img.shape
    nh = max(ALIGN, round(h / ALIGN) * ALIGN)
    nw = max(ALIGN, round(w / ALIGN) * ALIGN)
    if max_pixels and nh * nw > max_pixels:
        s = (max_pixels / (nh * nw)) ** 0.5
        nh = max(ALIGN, round(nh * s / ALIGN) * ALIGN)
        nw = max(ALIGN, round(nw * s / ALIGN) * ALIGN)
    return F.interpolate(img, size=(nh, nw), mode="bilinear", align_corners=False)

def pack_navit(images):
    """NaViT-style packing: many native-res images -> one patch sequence + a
    block-diagonal attention mask so each image only attends to itself."""
    seqs, sizes = [], []
    for img in images:
        img = resize_to_native(img)
        _, c, h, w = img.shape
        gh, gw = h // PATCH, w // PATCH                      # patch grid
        patches = (img.unfold(2, PATCH, PATCH)               # -> (1,c,gh,gw,14,14)
                      .unfold(3, PATCH, PATCH)
                      .reshape(c, gh * gw, PATCH * PATCH)
                      .permute(1, 0, 2)
                      .reshape(gh * gw, c * PATCH * PATCH))
        seqs.append(patches)
        sizes.append(gh * gw)                                # token count for this image

    tokens = torch.cat(seqs, dim=0)                          # (sum_i n_i, c*14*14)
    n = tokens.shape[0]
    mask = torch.zeros(n, n, dtype=torch.bool)              # block-diagonal: True = attend
    off = 0
    for sz in sizes:
        mask[off:off + sz, off:off + sz] = True
        off += sz
    return tokens, mask, sizes                               # mask isolates each image
```

That snippet is the heart of native-resolution handling. There is no tiling loop, no thumbnail, no tile-index embedding — just resize, patchify, pack, and a block-diagonal mask. The encoder sees each image whole.

**The adapter (deliberately conventional).** After Seed-ViT, a **2×2 average-pool** merges each 2×2 block of patch tokens into one, cutting the vision-token count by 4×, and a **two-layer MLP** projects the pooled features into the LLM's embedding dimension. This is the same family as the pixel-shuffle / MLP connectors used across the field. The paper is candid that this is not where the novelty lives — the connector and the post-hoc adaptation strategy are conventional. What matters is that the 4× pool happens *before* the LLM, so a 40×60 patch image (2400 patches) arrives at the decoder as 600 tokens, not 2400.

**The MoE LLM (~20B active).** A decoder-only Mixture-of-Experts model, initialized from an internal Seed text MoE pretrained on trillions of tokens. Only the **~20B active** figure is disclosed; the total parameter count is not, and I will not invent one. For scale intuition the paper compares the active footprint to Llama 4 Maverick's 17B active. The point of MoE here is the same as in [Kimi-VL](/blog/paper-reading/multimodal/kimi-vl): decouple knowledge capacity (total params) from per-token compute (active params), so the model can hold broad world knowledge while staying cheap enough to serve interactively and to run RL rollouts on.

| Component | Key spec | Role |
|---|---|---|
| Seed-ViT | 532M; patch 14; dim 1280; depth 27; 20 heads × 64; 2D RoPE | Native-res perception, one pass per image |
| Adapter | 2×2 avg-pool + 2-layer MLP | 4× token compression + projection (conventional) |
| MoE LLM | ~20B active (total undisclosed); decoder-only; text-MoE init | Reasoning, generation, agent control |

## 3. Why train the ViT from scratch instead of retrofitting CLIP

It is worth slowing down on this decision, because retrofitting is the default and from-scratch is the road less traveled. Everyone has CLIP and SigLIP checkpoints lying around; they are excellent, well-studied, and free. The standard move — used by LLaVA, InternVL, Qwen-VL's early versions, and most of the field — is to take such a checkpoint and continually pretrain it to accept higher or variable resolution. If you want the conceptual background on why CLIP, SigLIP, and DINO encoders behave the way they do, the [ViT/SigLIP/DINO explainer](/blog/machine-learning/computer-vision/vit-siglip-dino-explained) is the prerequisite read.

The problem with retrofitting is that a fixed-res CLIP encoder has *baked in* assumptions that fight native resolution. Its position embeddings are a learned table sized for one grid; stretching them to a new grid via interpolation is lossy and the model never trained against the interpolated positions. Its features were optimized under a global image-text contrastive objective at low resolution, so they are tuned to summarize a whole scene into one vector, not to preserve the local detail a document or a UI needs. You can finetune around these, but you are always pushing against the encoder's pretraining prior.

Seed-ViT sidesteps all of it by making native resolution and 2D RoPE part of the *pretraining* objective, not a post-hoc adaptation. The 2D RoPE means there is no position table to interpolate — position is computed from coordinates at every resolution, so the encoder trains on the full distribution of aspect ratios and grid sizes it will see at inference. The result is the efficiency number we already quoted: 82.5 average zero-shot over 6 ImageNet variants at 9% of InternVL-C-6B's params. A retrofitted encoder of the same size would carry the scars of its fixed-res origin; a from-scratch one does not.

The cost, of course, is that you have to pretrain a vision encoder, which is expensive and risky. The team derisks it with a three-stage curriculum that we turn to next.

### 3.1 Why 2D RoPE makes arbitrary resolution free

This is the single most important mechanical idea in the encoder, so it deserves a slow, intuitive walkthrough rather than a derivation. A learned absolute position embedding is a lookup table: you allocate one vector per grid cell at training time, so a model trained on a 16×16 grid owns exactly 256 position vectors. At inference, if an image produces a 74×74 grid, the table has no entry for position (50, 60) — it was never allocated. The standard workaround is to *interpolate* the table: treat the 16×16 grid of vectors as a tiny image, bilinearly upsample it to 74×74, and hope the interpolated vectors mean something. They sort of do, but the model never saw those interpolated vectors during training, so it is operating off-distribution exactly when you most need precision — on the high-resolution inputs that motivated native resolution in the first place. That is the prior a retrofitted CLIP encoder is fighting.

Rotary position embedding (RoPE) removes the table entirely. Instead of *adding* a position vector, RoPE *rotates* the query and key vectors by an angle proportional to position before the attention dot product. The dot product of two rotated vectors then depends only on the *difference* of their positions, so attention becomes relative-position-aware by construction. The 2D variant splits the head dimension into two halves and applies a rotation driven by the patch's row coordinate to one half and a rotation driven by its column coordinate to the other half. The key property: the rotation is a *function* of the coordinate, not a *lookup* of it. There is no table to run out of. A patch at row 50, column 60 gets a perfectly well-defined rotation whether the grid is 16×16 or 74×74 — the function is the same, only the input coordinate differs.

Work a concrete example. Take a square **1036×1036** image. Bilinear-resize to the nearest 28-multiple — 1036 is already 37×28, so no resize is needed — and patchify at 14: you get a **74×74 patch grid**, 5,476 patches. The patch at the centre of a logo in the lower-right sits at, say, coordinate (62, 68). Its 2D RoPE rotation is computed from (62, 68) directly. Now take a wide banner image, **2044×560**. Resize to 28-multiples gives 2044 = 73×28 and 560 = 20×28, so the patch grid is **146 columns × 40 rows** — a wildly different aspect ratio, 5,840 patches. A patch at (12, 130) gets a rotation computed from *its* coordinate. The encoder never had to allocate a position vector for a 146-wide grid, never interpolated anything, and the relative geometry — "this patch is 18 columns left and 3 rows up from that one" — is encoded identically in both images because it falls out of the *difference* of the rotations. That invariance is why a single 532M weight set covers receipts, A4 scans, 4K screenshots, and wide game frames with no per-resolution machinery. A learned table would have needed a separate (and never-trained) entry for every one of those grids.

There is a second-order benefit that matters for documents and UI specifically. Because the position signal is relative, a column of text that appears at the top of a tall screenshot and the same column appearing in the middle of a different screenshot are encoded with the *same local structure*. The model learns layout patterns ("label sits to the left of value", "row beneath header") once and reuses them at any absolute offset, instead of re-learning them per grid position. This is part of why the OCR and grounding numbers hold up at resolutions the model never explicitly trained on.

### 3.2 Why post-hoc adaptation beat joint-from-scratch here

A purist would train the whole VLM jointly from scratch — vision and language tied together from the first gradient step — so that the two modalities co-adapt with no seam. Seed1.5-VL deliberately does not. The LLM is pretrained on text first, the vision stack is grafted on afterward, and the paper is explicit that this *post-hoc adaptation* is a pragmatic engineering choice, not a claim of optimality. The reasoning is worth unpacking because it is a recurring fork in VLM design.

The case for joint-from-scratch is that the language model can shape its representations around vision from the beginning, potentially yielding tighter multimodal fusion. The case against it — and the one the Seed team acted on — is iteration speed and risk isolation. The LLM pretraining run is the single most expensive, least reversible part of the whole pipeline; the vision recipe (data mix, adapter design, stage boundaries, long-tail rebalancing) is the part you most want to *ablate* dozens of times. If vision and language are entangled in one run, every vision ablation forces you to re-pay the language-pretraining bill, which is prohibitive. By pretraining the LLM once and grafting vision on top, the team can iterate the vision side cheaply, and they can reuse a battle-tested text MoE checkpoint trained on trillions of tokens rather than re-deriving language ability from multimodal data. The 5%-text-replay in pretrain Stage 1 and the text-only SFT exist precisely to protect that inherited language ability from erosion during multimodal training — the defensive cost of the post-hoc choice, paid deliberately.

The honest tradeoff is that post-hoc adaptation likely leaves some fusion quality on the table relative to a perfectly-executed joint run, and a few of the model's weaknesses (hallucination when pixels conflict with text priors, multi-image interdependent reasoning) are the kind of thing tighter early fusion might have softened. The team judged the iteration-speed win worth that cost, and the 38/60 SOTA result suggests the judgment was sound for a first-generation model on a fixed compute budget.

## 4. The three-stage Seed-ViT pretrain

Training a competitive vision encoder from scratch, at only 532M params, in a way that produces features both *aligned to language* and *detailed enough for documents and grounding*, is a curriculum problem. Seed-ViT solves it in three stages, each with a different objective and data mix.

<!-- FIGSPEC 3
kind: timeline
claim: Seed-ViT pretrains in three ordered stages, moving from masked feature reconstruction to native-res contrastive to omni-modal alignment.
caption: Encoder quality is a curriculum: MIM builds spatial structure, contrastive installs language alignment, a thin MiCo layer adds cross-modal richness.
nodes:
  - id: a | label: "Stage 1: MIM\nEVA02-CLIP-E teacher\n75% mask, 2D RoPE" | color: blue
  - id: b | label: "Stage 2: contrastive\nSigLIP + SuperClass\nattn-pool 1280-d" | color: blue
  - id: c | label: "Stage 3: MiCo omni\nframes+audio+caption\n4.8% of tokens" | color: green
edges:
  - a -> b | label: "+ language"
  - b -> c | label: "+ temporal"
notes: horizontal timeline left-to-right; 3 ordered milestones with objective per stage
-->
![Seed-ViT three-stage pretraining timeline: masked image modeling with a CLIP teacher, then native-resolution contrastive learning, then MiCo omni-modal alignment over video and audio](/imgs/blogs/seed1-5-vl-native-resolution-vision-language-3.png)

The timeline above is the recipe. **Stage 1 — Masked Image Modeling (MIM) with 2D RoPE.** The encoder is trained to reconstruct masked image content, but not in pixel space: it reconstructs the *features* of a frozen **EVA02-CLIP-E teacher** under a cosine-similarity loss, with a **75% masking ratio**. This is distillation-flavored MIM — the encoder learns to predict, from 25% of the visible patches, what a strong CLIP teacher would have produced for the masked ones. The high mask ratio forces the encoder to build a genuine spatial model rather than memorizing local texture, and the CLIP-feature target seeds language-alignment cheaply before any text loss runs. The 2D RoPE is present from the very first stage, so position is native-resolution-aware throughout.

**Stage 2 — native-resolution contrastive learning.** Now the encoder learns to produce a single 1280-dimensional image embedding via attention pooling, trained with a combined **SigLIP + SuperClass** loss against image-text pairs at native resolution. SigLIP's sigmoid pairwise loss gives the contrastive image-text alignment; SuperClass adds a classification-style supervisory signal. This is the stage that turns the MIM-pretrained spatial model into something that aligns with text and can be probed zero-shot — it is where the 82.5 ImageNet-variant number comes from.

**Stage 3 — MiCo omni-modal alignment.** The final stage uses **MiCo** to align video frames, audio, and captions in a shared space, training on **video-audio-text** data that is only **4.8% of the token budget** yet produces large gains on both image and video understanding. The intuition: forcing the encoder to relate a video frame to its audio track and caption teaches temporally and semantically richer features than static image-text alone, and those features transfer back to single images. The whole ViT-pretrain data mix is **2.2B unlabeled images (4%)**, **4.8B image-text pairs (91.2%)**, and **65M video-audio-text examples (4.8%)** — that last sliver doing outsized work in Stage 3.

| Stage | Objective | Key ingredients | Data weight |
|---|---|---|---|
| 1. MIM | Reconstruct CLIP teacher features (cosine sim) | EVA02-CLIP-E teacher, 75% mask, 2D RoPE | unlabeled images (4%) |
| 2. Contrastive | Native-res image-text alignment | SigLIP + SuperClass, attn-pooled 1280-d | image-text pairs (91.2%) |
| 3. MiCo omni | Align frames + audio + captions | MiCo, omni-modal | video-audio-text (4.8%) |

The lesson encoded in this curriculum is that you do not need a giant encoder to get giant-encoder features; you need the *right objectives in the right order*. MIM-from-a-teacher builds spatial structure, contrastive learning installs language alignment, and a thin omni-modal layer adds temporal and cross-modal richness. The 532M result rivaling 6B and 18B encoders is the payoff.

## 5. VLM pretraining: three stages and a post-hoc adaptation

With Seed-ViT trained, the team assembles the full VLM and pretrains it on a **3-trillion-token** corpus. The adaptation is *post-hoc*: the LLM was pretrained first on text, and the vision stack is grafted on afterward. The paper is explicit that this is a pragmatic choice — post-hoc adaptation makes ablation faster because you can iterate on the vision side without re-running language pretraining — not a claim that it is optimal. The pretraining proceeds in three carefully staged phases.

<!-- FIGSPEC 4
kind: pipeline
claim: VLM pretraining ramps from a tiny adapter-only warmup to full-parameter training to a long-context stage that loads prior optimizer states.
caption: Three escalating stages: warm the adapter on 16B tokens, train everything on 3T, then extend to 128K context on 240B with no warmup.
nodes:
  - id: a | label: "Stage 0: 16B tok\nseq 32k\nadapter only" | color: amber
  - id: b | label: "Stage 1: 3T tok\nseq 32k, all params\n+5% text replay" | color: blue
  - id: c | label: "Stage 2: 240B tok\nseq 131k, all params\n+video/3D, no warmup" | color: green
edges:
  - a -> b | label: "unfreeze all"
  - b -> c | label: "load optimizer"
notes: left-to-right pipeline; token count grows then shrinks, seq len jumps at stage 2
-->
![Seed1.5-VL pretraining stages: a 16B-token adapter-only warmup, a 3T-token all-parameter stage, and a 240B-token long-context stage adding video and 3D](/imgs/blogs/seed1-5-vl-native-resolution-vision-language-4.png)

The pipeline above shows the staging. **Stage 0 (16B tokens, adapter-only).** Sequence length 32,768. Only the MLP adapter trains; Seed-ViT and the LLM are both frozen. This is the alignment warmup — it teaches the randomly-initialized adapter to map vision features into the LLM's embedding space without disturbing either pretrained component. Sixteen billion tokens is tiny relative to what follows, which is the point: you spend just enough to get a usable bridge.

**Stage 1 (3T tokens, all parameters).** Sequence length still 32,768, but now everything is trainable. This is the bulk of pretraining: knowledge acquisition, grounding, and OCR, with roughly **5% text-only data mixed in** to keep the language model from forgetting how to write. This text-replay trick is the same defensive move [Kimi-VL](/blog/paper-reading/multimodal/kimi-vl) uses — without it, multimodal training quietly degrades pure-language ability.

**Stage 2 (240B tokens, long context).** Sequence length is extended to **131,072** (128K), and the data adds **video, coding, and 3D**. Crucially, this stage uses a **constant learning rate**, **loads the Stage-1 optimizer states**, and runs with **no warmup** — it is a continuation of Stage 1 at longer context, not a fresh phase. The 128K window is what makes long video (up to 640 frames) and long documents tractable downstream. The optimizer is AdamW throughout with **β1=0.9, β2=0.95, weight decay 0.1**. Total pretraining compute is **1.3M H800-GPU-hours**.

| Stage | Tokens | Seq len | Trainable | New data | Notes |
|---|---|---|---|---|---|
| 0 | 16B | 32,768 | Adapter only | image-text | ViT + LLM frozen; alignment warmup |
| 1 | 3T | 32,768 | All | knowledge, grounding, OCR | ~5% text-only replay |
| 2 | 240B | 131,072 | All | +video, coding, 3D | constant LR, loads Stage-1 optimizer, no warmup |

## 6. The synthetic data engine

A model is its data, and Seed1.5-VL's data recipe is unusually synthetic-heavy and unusually well-documented. The taxonomy is worth laying out because the *scale per domain* tells you where the model's capabilities come from.

<!-- FIGSPEC 8
kind: tree
claim: The training data taxonomy spans five branches whose per-branch scale explains where each capability of the model comes from.
caption: Capabilities trace to data scale: a billion-sample OCR branch, a 200M auto-grounded branch, plus 3D, STEM, and GUI lineages.
nodes:
  - id: a | label: "Seed1.5-VL data" | color: gray
  - id: b | label: "OCR >1B\n+100M chart, 50M tbl" | color: blue
  - id: c | label: "Grounding/count\n~200M Grounding DINO" | color: green
  - id: d | label: "3D\n3.2B rel-depth tok" | color: blue
  - id: e | label: "STEM\n>100M K12 + synth" | color: blue
  - id: f | label: "GUI\nUI-TARS lineage" | color: green
edges:
  - a -> b
  - a -> c
  - a -> d
  - a -> e
  - a -> f
notes: root fans to 5 branches; each leaf labeled with its scale; tree layout top-down
-->
![Seed1.5-VL data taxonomy tree: OCR over a billion samples, ~200M auto-grounding via Grounding DINO, 3D depth, STEM, and GUI data from the UI-TARS lineage, with scale per branch](/imgs/blogs/seed1-5-vl-native-resolution-vision-language-8.png)

The tree above shows the five major branches and their scales. **OCR** is the largest: an in-house corpus of **>1B** samples, plus **>200M synthetic text-images**, **>100M chart** examples, and **>50M tables**. This is why Seed1.5-VL's document and OCR numbers are SOTA — you read dense text well when you have trained on a billion text-images at native resolution.

**Grounding and counting** is built largely by automation. Coordinates are normalized to **[0, 999]** (a fixed integer grid so the LLM emits stable coordinate tokens). The grounding data: **48M** open-source samples (41B tokens), **~200M** samples auto-annotated by **Grounding DINO** (200B tokens), **170M** point-data samples (110B tokens), and **8M** counting samples (13B tokens). The Grounding-DINO auto-annotation pipeline is the scalability story — you cannot hand-label 200M bounding boxes, so you bootstrap them from a strong open detector and train on the result. This is the same lineage of thinking that produces the LVIS-MG counting result (73.8, vs Grounding DINO-L's own 54.4) — the student beats its labeling teacher because the VLM integrates language context the detector lacks.

**3D** uses depth as the supervision signal: **relative depth** from DepthAnything V2 (3.2B tokens), **absolute depth** (18M samples, 28B tokens), and **3D grounding** (770K samples, 1.3B tokens). **STEM** is heavily synthetic: **3.2M** educational grounding samples, **10M** synthetic tables, **4.5M** chemistry diagrams, **1.5M** coordinate diagrams, and **>100M** K-12 problems. **GUI** data is curated from the **UI-TARS** lineage — screenshots annotated with element type, bounding box, text, and depth, Set-of-Mark visual markers, and multi-step interaction trajectories of (observation, thought, action) tuples. That GUI corpus is the direct reason Seed1.5-VL is a competent screen agent.

One technique deserves a callout because it generalizes: **long-tail rebalancing by alt-text duplication**. Domains whose frequency falls below 50% of the average get their alt-text-derived captions duplicated to rebalance the distribution. The ablation is striking — on the Biotrove biology benchmark, a **Max1k-46M** capped sampling strategy scores **62.01** versus **44.69** for random sampling. The lesson: in a synthetic-heavy regime, *how you sample the long tail* can swing a domain by 17 points, independent of model architecture.

| Branch | Scale | Source/method |
|---|---|---|
| OCR | >1B + >200M synth + >100M chart + >50M tables | In-house + synthetic rendering |
| Grounding/counting | 48M + ~200M (Grounding DINO) + 170M points + 8M count | Auto-annotation, coords [0,999] |
| 3D | 3.2B rel-depth + 18M abs-depth + 770K 3D-grounding | DepthAnything V2 + annotation |
| STEM | 3.2M + 10M tables + 4.5M chem + 1.5M coord + >100M K12 | Synthetic generation |
| GUI | UI-TARS screenshots + Set-of-Mark + trajectories | Curated agent data |

## 7. Video tokenization: dynamic frame-resolution sampling

Video is where token budgets go to die. A 10-minute clip at 30 FPS is 18,000 frames; even at 256 tokens per frame that is 4.6M tokens, which no context window survives. Every video VLM therefore makes a frames-vs-resolution tradeoff: sample fewer frames at higher resolution, or more frames at lower resolution. Most models hardcode this tradeoff. Seed1.5-VL makes it *dynamic* and *budget-bounded*, which is the cleverest piece of its perception stack after the encoder itself.

<!-- FIGSPEC 5
kind: graph
claim: A fixed 81,920-token video budget is allocated across six resolution levels and task-dependent frame rates, with uniform downsampling as fallback.
caption: One token budget, many ways to spend it: pick FPS by task, then the highest resolution level that still fits the 81,920-token cap.
nodes:
  - id: a | label: "Video in\nduration x FPS" | color: gray
  - id: b | label: "FPS by task\n1 / 2 / 5" | color: blue
  - id: c | label: "Budget 81,920 tok" | color: amber
  - id: d | label: "6 levels\n640..128 px" | color: blue
  - id: e | label: "+ timestamp tok\n[1.5 second]" | color: green
  - id: f | label: "fallback: uniform\ndownsample" | color: red
edges:
  - a -> b
  - b -> c | label: "frames"
  - c -> d | label: "fit budget"
  - d -> e | label: "if fits"
  - c -> f | label: "if overflow"
notes: branching dataflow graph; budget node fans to level selection or downsample fallback
-->
![Dynamic Frame-Resolution Sampling: an 81,920-token budget per video is split across six resolution levels with timestamp tokens and task-dependent frame rates](/imgs/blogs/seed1-5-vl-native-resolution-vision-language-5.png)

The graph above shows the allocator. Every video gets a **fixed budget of 81,920 tokens**. The system chooses a frame rate by task — **1 FPS by default**, **2 FPS for fine temporal detail**, **5 FPS for counting and motion** — then picks a per-frame resolution from **six levels {640, 512, 384, 256, 160, 128}** so that (frames × tokens-per-frame) fits inside 81,920. If even the lowest resolution at the desired frame rate overflows the budget, it **falls back to uniform temporal downsampling** — dropping frames evenly so coverage stays uniform rather than truncating the tail. Each frame is also prefixed with a **timestamp token** like `[1.5 second]`, so the model knows *when* each frame occurred, not just its order — which is what makes temporal grounding ("what happens at 0:47?") possible.

The reason this beats a fixed policy is that information density varies wildly across videos. A static lecture slide deck needs 1 FPS at high resolution to read the slides; a fast sports clip needs 5 FPS at lower resolution to catch motion. A fixed policy is wrong for one of them. The budget-bounded allocator spends the same total compute either way but distributes it where the task needs it.

```python
LEVELS = [640, 512, 384, 256, 160, 128]   # candidate per-frame edge sizes (px)
BUDGET = 81_920                             # total vision tokens per video
PATCH, POOL = 14, 2                         # Seed-ViT patch + 2x2 adapter pool

def tokens_per_frame(edge):
    """Square-ish frame at `edge` px -> tokens after patchify + 2x2 pool."""
    g = edge // PATCH                        # patch grid per side
    return (g // POOL) * (g // POOL)         # pooled token count

def fps_for_task(task):
    return {"counting": 5, "motion": 5, "fine_temporal": 2}.get(task, 1)

def allocate(duration_s, task="default"):
    """Pick (fps, resolution) so frames*tokens_per_frame <= BUDGET."""
    fps = fps_for_task(task)
    n_frames = max(1, int(duration_s * fps))
    for edge in LEVELS:                       # try high-res first, step down
        if n_frames * tokens_per_frame(edge) <= BUDGET:
            return {"fps": fps, "resolution": edge, "frames": n_frames,
                    "tokens": n_frames * tokens_per_frame(edge), "mode": "native"}
    # even the smallest level overflows -> uniform temporal downsample
    edge = LEVELS[-1]
    keep = BUDGET // tokens_per_frame(edge)
    stride = max(1, n_frames // keep)         # drop frames evenly
    return {"fps": fps, "resolution": edge, "frames": keep, "stride": stride,
            "tokens": keep * tokens_per_frame(edge), "mode": "uniform_downsample"}
```

That allocator, plus the 128K context window from pretraining Stage 2, is what lets Seed1.5-VL handle up to **640 frames** and set SOTA on short, streaming, and grounded-video benchmarks. The timestamp tokens are the unsung hero: they convert a bag of frames into a timeline.

### 7.1 A worked allocation: a 10-minute clip at 1 FPS

Let us run the allocator by hand on a concrete case, because the budget arithmetic is where the design earns its keep. Take a **10-minute clip** — 600 seconds — sampled at the default **1 FPS**, giving **600 frames**. The budget is 81,920 tokens, so the per-frame budget is 81,920 / 600 ≈ **136.5 tokens per frame**. Now walk the six resolution levels and compute tokens per frame after patchify-at-14 and the 2×2 pool, treating frames as roughly square for the estimate:

- **640px:** patch grid 640/14 ≈ 45 per side, pooled 22×22 ≈ **484 tok/frame** → 600 × 484 = 290,400. Overflows 81,920.
- **512px:** 36 per side, pooled 18×18 = 324 → 600 × 324 = 194,400. Overflows.
- **384px:** 27 per side, pooled 13×13 = 169 → 600 × 169 = 101,400. Still overflows.
- **256px:** 18 per side, pooled 9×9 = 81 → 600 × 81 = **48,600. Fits** (under 81,920).

So the allocator picks **256px**, spending 48,600 of the 81,920-token budget on 600 frames at 1 FPS, with headroom to spare. Notice that the two highest-detail levels (640, 512) and even 384 all overflowed: a 10-minute clip is simply too long to read every frame at high resolution, and the allocator correctly trades resolution for temporal coverage. If the task were *counting* and demanded 5 FPS, the same clip becomes 3,000 frames and the per-frame budget collapses to ~27 tokens; even the smallest level (128px → 9 per side, pooled 4×4 = 16 tok/frame) gives 3,000 × 16 = 48,000, which still fits — but a 20-minute counting clip at 5 FPS (6,000 frames × 16 = 96,000) would *overflow even at 128px*, and the allocator falls back to **uniform downsampling**: keep 81,920 / 16 = 5,120 frames, dropping every other frame or so (stride ≈ 1) so coverage stays uniform rather than truncating the back half of the video. The fallback is what prevents a long clip from silently losing its ending.

The contrast with a fixed policy is stark. A model hardcoded to, say, 8 frames at 448px would read this 10-minute clip with 8 snapshots — fine for "what is this video about", useless for "when does the speaker switch slides". A model hardcoded to 1 FPS at 640px would demand 290,400 tokens and blow the budget by 3.5×, forcing a truncation that drops most of the clip. The budget-bounded allocator is the only policy that keeps both *full temporal coverage* and *a fixed compute cost* — it moves on the resolution axis so it never has to move on the coverage axis until the budget physically cannot hold the frames.

### 7.2 Why the 2×2 pool is the difference between affordable and not

The same arithmetic explains why the adapter's 2×2 pool is not a throwaway detail but a load-bearing piece of token economy, and it is clearest on a single high-resolution still. Take a **4K image**, 3840×2160. Resize to 28-multiples (3836 ≈ 137×28, 2156 = 77×28) and patchify at 14: the patch grid is roughly **274 × 154 = 42,196 patches**. Without the pool, that is 42,196 vision tokens for *one* image — more than half the entire 81,920-token video budget, spent on a single frame, and a crushing load on the 20B-active LLM's attention, which scales quadratically in sequence length. With the **2×2 pool**, those 42,196 patches collapse 4× to **≈ 10,549 tokens**, and the LLM's attention cost drops by 16× (quadratic in the 4× shorter sequence). The pool happens in the cheap 532M encoder, *before* the expensive decoder ever runs, so the saving is pure profit.

This is the encoder-keeps-its-job argument made numerical. An encoder-free VLM would dump all 42,196 raw patches onto the LLM; the encoder-plus-pool design hands the LLM a pre-compressed 10,549-token summary that has already done the spatial integration. For an interactive product where every token of latency is felt, that 4× cut at the adapter — invisible on any benchmark line — is the difference between a model you can serve and one you cannot.

## 8. Post-training: SFT, the generative reward model, RLVR, and hybrid RL

This is the section the paper is really about, and where its most transferable ideas live. The post-training is an **iterative SFT↔RL loop** with hard-prompt mining: train SFT, run RL, mine the prompts the model still fails, and feed RL outputs back into SFT via rejection sampling. The model exposes a **user-toggleable LongCoT "thinking" mode** versus a concise mode, switched by system prompts — the same dual-mode design idea as the text-side [Seed1.5-Thinking RL recipe](/blog/paper-reading/large-language-model/seed1-5-thinking-rl-reasoning-vapo-dapo), whose VAPO/DAPO lineage this work inherits.

### 8.1 Supervised finetuning

The SFT set is roughly **50k multimodal examples**: **13k crowdsourced** plus **~30k distilled** from a 1.5M-example pool, where the distillation pipeline clusters the pool, generates candidate responses, and screens them through an **LLM judge** and a **reward model** to keep only the best. There is also text-only SFT and dedicated **LongCoT SFT** for the thinking mode. Training runs **2 epochs** at sequence length **131,072** with **Seed-ViT frozen** and a cosine learning rate from **2e-5 down to 2e-6**. Freezing the ViT during SFT is the right call — the perception is already good, and you do not want a small SFT set to perturb a billion-token-trained encoder.

### 8.2 The generative reward model

The reward model is the most architecturally interesting piece of the RL stack. Instead of a Bradley-Terry scalar head — the standard approach, where you train a model to output a single preference score — Seed1.5-VL uses a **generative VLM-as-classifier**: the reward model is itself a VLM that *emits a preference token*, treating preference judgment as a generation task. The team argues this is more robust than Bradley-Terry, presumably because it lets the reward model reason in-context before committing to a judgment rather than collapsing everything to one scalar.

Preferences are collected list-wise on a **5-point scale** (human plus synthetic). The reward model is **order-debiased**: position bias is a known plague of pairwise/list-wise judges (models prefer whichever answer they saw first), so the system **scores both orderings of every pair and averages**, cancelling the bias. This is a cheap, robust fix that more RLHF pipelines should adopt.

### 8.3 RLVR: verifiable rewards

For tasks with a checkable answer, Seed1.5-VL uses **Reinforcement Learning with Verifiable Rewards (RLVR)** — no reward model, just a verifier that returns correct/incorrect. The verifiable task suite:

- **Visual STEM (>1M problems):** answers extracted from `\boxed{}` and matched with **sympy** for symbolic equivalence. Multiple-choice problems are converted to open-ended (so the model cannot guess), and only problems with **0% < accuracy ≤ 75%** over 16 rollouts are kept — too-easy and too-hard problems carry no gradient signal.
- **Grounding (IoU reward):** the model emits `<bbox>` or `<point>` and is rewarded by intersection-over-union with ground truth.
- **Instruction-following:** regex verifiers check format constraints.
- **Visual puzzles (>20k synthetic):** decontaminated against PuzzleVQA to avoid test leakage.
- **"Spot the Differences" game:** image pairs generated by diffusion inpainting and SVG edits, with bounding-box localization of the differences as the output — a self-supervised game with a free verifier (you know where you edited).

The 0%-to-75% accuracy filter is the quiet workhorse: it is automatic curriculum construction. Problems the model already solves give no learning signal; problems it never solves give no reward to climb toward. Keeping the band in between maximizes gradient per rollout.

### 8.3.1 A full verifier walkthrough, end to end

Abstractions hide where the real engineering lives, so let us push one visual-STEM problem all the way through the RLVR pipeline, then trace the grounding and instruction-following paths, and finally assemble the reward.

**Step 1 — curriculum filtering by 16 rollouts.** Start from the >1M visual-STEM pool. Suppose a problem shows a geometry figure and asks for the area of a shaded region; the ground-truth answer is `3*pi - 9`. Before this problem is ever used for RL, the policy generates **16 rollouts** on it. Count how many produce the correct answer. Three regimes:

- If **0 of 16** are correct (accuracy 0%), the problem is *dropped* — the policy cannot reach it, so every rollout returns reward 0 and the advantage estimate is uniformly flat. No gradient signal.
- If **all 16** are correct (accuracy 100%), the problem is *dropped* — the policy already owns it, every rollout returns reward 1, advantages are again flat. No gradient signal.
- If, say, **5 of 16** are correct (accuracy 31%, inside the 0%–75% band), the problem is *kept*. Here the rollouts disagree, so some have positive advantage and some negative, and PPO has a gradient to climb. This is automatic curriculum: the kept set is precisely the frontier of the policy's current ability.

**Step 2 — multiple-choice to open-ended.** If the source problem were multiple choice ("A, B, C, or D"), it is first *converted to open-ended* so the model must produce the actual value `3*pi - 9` rather than guessing a letter. This matters because a 4-way MC problem has a 25% floor accuracy from guessing alone, which would corrupt the 0%–75% filter (a problem the model cannot reason about could still land at 25% by luck and look "learnable"). Forcing open-ended answers makes the accuracy signal honest.

**Step 3 — `\boxed{}` extraction.** The policy is trained to emit its final answer inside a `\boxed{}` delimiter, e.g. `...therefore the area is \boxed{3\pi - 9}`. The verifier extracts the boxed substring with a parser. If there is no `\boxed{}`, extraction fails and the answer reward is 0 — this is part of why format discipline matters.

**Step 4 — sympy symbolic match.** The extracted string `3\pi - 9` and the ground truth `3*pi - 9` are *not* string-equal, and a naive string comparison would wrongly mark a correct answer wrong. Instead the verifier parses both into symbolic expressions and asks sympy whether their *difference simplifies to zero*. So `3\pi - 9`, `-9 + 3\pi`, and `3*(pi - 3)` all verify as correct because `sympy.simplify(a - b) == 0` for each. Symbolic equivalence is what makes the verifier robust to the dozen ways a model can spell the same number.

```python
import sympy as sp
from sympy.parsing.latex import parse_latex

def stem_reward(response: str, gold: str) -> float:
    """Visual-STEM verifier: extract \boxed{}, symbolic-match with sympy."""
    box = extract_boxed(response)              # returns LaTeX inside \boxed{...} or None
    if box is None:
        return 0.0                             # no boxed answer -> no credit
    try:
        pred = parse_latex(box)
        target = parse_latex(gold)
        return 1.0 if sp.simplify(pred - target) == 0 else 0.0
    except Exception:
        return 0.0                             # unparseable -> treat as wrong

def grounding_reward(response: str, gt_box) -> float:
    """Grounding verifier: parse <bbox>x1,y1,x2,y2</bbox>, reward = IoU."""
    pred = parse_bbox(response)                # coords on the [0,999] grid
    if pred is None:
        return 0.0
    return iou(pred, gt_box)                    # continuous reward in [0,1]

def ifeval_reward(response: str, constraints) -> float:
    """Instruction-following verifier: fraction of regex constraints satisfied."""
    hits = sum(bool(re.search(c, response)) for c in constraints)
    return hits / len(constraints)             # e.g. "answer in 3 bullet points"

def assemble_reward(prompt, response, ref_logp, policy_logp) -> tuple[float, float]:
    """Format gate -> task verifier -> per-type KL. Returns (reward, kl_coef)."""
    m = THINK_RE.match(response.strip())
    if not m:
        return 0.0, 1e-5                       # malformed <think> wrapper -> zero
    solution = m.group(2).strip()              # CoT (group 1) is never scored
    if prompt.task == "stem":
        return stem_reward(solution, prompt.gold), 0.0
    if prompt.task == "grounding":
        return grounding_reward(solution, prompt.gt_box), 0.0
    if prompt.task == "ifeval":
        return ifeval_reward(solution, prompt.constraints), 0.0
    # open-ended -> generative reward model, light KL leash
    return generative_rm.preference(prompt, solution), 1e-5
```

**The grounding path** is the most different because its reward is *continuous*, not binary. The model emits a box like `<bbox>120,340,260,520</bbox>` on the normalized [0, 999] grid, the verifier parses it, and the reward is the **intersection-over-union (IoU)** with the ground-truth box — a number in [0, 1] rather than a hard pass/fail. A box that is 80% overlapping earns 0.8, so the policy gets a smooth gradient *toward* the right region instead of a cliff. The `<point>` variant works the same way for point-localization tasks. This continuous reward is why grounding RL converges smoothly: every near-miss is informative.

**The instruction-following path** uses regex verifiers. If the prompt says "answer in exactly three bullet points starting with a verb", a set of regexes checks each constraint, and the reward is the fraction satisfied. This is the cheapest verifier of the three and it directly trains the format discipline the rest of the system depends on.

**Reward assembly.** Now the unifying step. Every path first passes through the **format gate**: the response must match `<think>{thought}</think>{solution}`, or the reward is 0 regardless of content. The CoT inside `<think>` is then *stripped*, and only the `{solution}` is handed to the verifier (sympy, IoU, or regex) or, for open-ended prompts, to the generative reward model. Each verifier returns a value in [0, 1], the same range the generative RM is normalized to, so all reward sources are commensurate by the time they reach the shared critic. The KL coefficient is attached per task type — 0 for the three verifiable paths, 1e-5 for the RM path. That single `assemble_reward` function is the seam where RLHF and RLVR become one batch.

### 8.4 Hybrid RL: one PPO loop for everything

Here is the genuinely novel design. Most systems run RLHF and RLVR as **separate** training phases or separate loops. Seed1.5-VL runs them in a **single PPO loop**, mixing RLHF prompts and RLVR prompts in the **same batch**, with one **shared critic**.

<!-- FIGSPEC 6
kind: pipeline
claim: A single PPO loop mixes generative-reward-model RLHF and verifier RLVR in one batch, sharing one critic and scoring only the post-think solution.
caption: RLHF and RLVR want the same critic: one PPO loop, two reward sources normalized to [0,1], per-prompt-type KL, reward on the solution only.
nodes:
  - id: a | label: "Policy rollout\n<think>..</think>sol" | color: blue
  - id: b | label: "Strip CoT\nscore solution only" | color: green
  - id: c | label: "RLHF: generative RM\npreference token\nKL=1e-5" | color: blue
  - id: d | label: "RLVR: verifier\nsympy/IoU/regex\nKL=0" | color: green
  - id: e | label: "Shared critic\ninit from RM, [0,1]" | color: amber
  - id: f | label: "PPO update\nclip 0.2" | color: blue
  - id: g | label: "Format violation\n-> zero reward" | color: red
edges:
  - a -> b
  - b -> c | label: "open-ended"
  - b -> d | label: "verifiable"
  - c -> e
  - d -> e
  - e -> f
  - a -> g | label: "if malformed"
notes: two reward branches merge into one shared critic then PPO; red format-violation sink
-->
![Hybrid RL: a single PPO loop mixes RLHF generative-reward-model prompts and RLVR verifier prompts, sharing one critic, with per-prompt-type KL and reward applied only to the post-think solution](/imgs/blogs/seed1-5-vl-native-resolution-vision-language-6.png)

The graph above shows how the two reward sources merge into one optimization. Several details make it work:

**Format reward and CoT freedom.** Outputs must follow `<think>{thought}</think>{solution}`; violating the format yields **zero reward**. Critically, the reward model **scores only the truncated solution** — the chain of thought is stripped before the RM sees it. This is the key trick: by judging only the final answer, the RM does not penalize exploratory or unconventional reasoning inside `<think>`, which *frees CoT exploration*. The model can think however it wants as long as the answer is good.

**Shared critic.** Both reward sources are normalized to **[0, 1]**, and a single critic — **initialized from the reward-model weights** with a **100-step warmup** — estimates value for both. Sharing the critic is what makes mixing RLHF and RLVR in one batch coherent: the value baseline is consistent across reward types, so advantages are comparable.

**Per-prompt-type KL.** The KL penalty against the reference policy is **1e-5 for general (RLHF) prompts** and **0 for verifiable (RLVR) prompts**. The reasoning: on verifiable tasks you *want* the policy to move freely toward correct answers, so you remove the leash; on open-ended tasks you keep a small leash to prevent reward hacking and preserve fluency.

**The PPO recipe (real numbers):** context length **8192**, max output **16384**, **4096 rollouts per episode**, minibatch **512**, **8 gradient steps per episode**, PPO clip **0.2**, actor LR **6e-7**, critic LR **7.5e-7**, **1 rollout per prompt for RM-scored prompts** and **4–8 rollouts per prompt for verifier prompts** (more rollouts where the signal is cheap and binary). The framework is **verl** with **vLLM** for rollouts and 3D parallelism.

**The cold start and the surprising transfer.** The LongCoT cold-start SFT is bootstrapped from the base model prompted *in-context* to reason; the team found that a **stronger cold start yields a stronger final model**, and they run **4 rounds of rejection-sampling finetuning**. The most surprising empirical finding: **RL trained only on LongCoT data, yet non-thinking responses also improved** — optimizing the thinking mode transferred to the concise mode for free. Post-training compute is modest: **RL 60k GPU-hours, RM training 24k GPU-hours**.

```python
# Hybrid RL: one PPO loop, mixed RLHF + RLVR batch, shared critic.
import re

THINK_RE = re.compile(r"^<think>(.*?)</think>(.*)$", re.S)

def reward_and_kl(prompt, response, ref_logp, policy_logp):
    """Return (reward, kl_coef) for a single rollout in the mixed batch."""
    m = THINK_RE.match(response.strip())
    if not m:
        return 0.0, 1e-5                      # format violation -> zero reward
    solution = m.group(2).strip()             # CoT in group(1) is NOT scored

    if prompt.kind == "verifiable":           # RLVR: verifier on the solution
        r = verifier_score(prompt, solution)  # e.g. sympy match, IoU, regex -> [0,1]
        kl = 0.0                              # no leash on verifiable tasks
    else:                                     # RLHF: generative reward model
        r = generative_rm.preference(prompt, solution)  # scores SOLUTION ONLY
        kl = 1e-5                             # small leash on open-ended tasks
    return r, kl                              # both normalized to [0,1] for shared critic

# shared critic init from RM weights, 100-step warmup; PPO clip 0.2;
# actor_lr=6e-7, critic_lr=7.5e-7; minibatch=512; 8 grad steps/episode;
# 4096 rollouts/episode; 1 rollout/prompt (RM) vs 4-8 (verifier).
```

That single function captures the three ideas at once: format-gated reward, solution-only scoring, and per-prompt-type KL — all flowing into one shared-critic PPO update.

## 9. Capabilities: agent, video, OCR, grounding, 3D

The architecture and RL recipe exist to produce capabilities, so let us look at where they land. The standout is the **agentic** behavior, which is unusual for a model this size.

**GUI and computer-use agent (UI-TARS lineage).** On screen grounding, Seed1.5-VL hits **ScreenSpot-V2 95.2**, beating OpenAI's Computer-Using Agent (CUA) at **87.9** and Claude 3.7 at **87.6**. On the harder **ScreenSpot-Pro it scores 60.9** (UI-TARS-1.5 edges it at 61.6). On full agentic tasks: **OSWorld 36.7**, **WindowsAgentArena 39.6**, **WebVoyager 87.2**, **Online-Mind2Web 76.4**, **AndroidWorld 62.1**. These are end-to-end "look at a screen, decide, click" benchmarks, and a 20B-active model competing with dedicated agent systems is the surprise.

**Gameplay.** The team built a benchmark of **14 Poki browser games** run up to **100 steps**. Seed1.5-VL scores **870.6 on 2048** (vs CUA's 611.2) and **1414 on Hex-Frvr** (vs 651.6) — more than doubling the baseline. Gameplay is a brutal test of closed-loop perception-action because every move changes the board and errors compound.

**Video.** With the 128K context and dynamic frame-resolution sampling, Seed1.5-VL sets SOTA across short, streaming, and grounded video, handling up to 640 frames. On VideoMME (w/o subtitles) it reaches **77.9** (trailing Gemini 2.5 Pro's 87.0), **MLVU 82.1**, **LongVideoBench 74.0**, **Video-MMMU 81.4**.

**OCR and documents.** SOTA on TextVQA, InfographicVQA, and DocVQA — the direct payoff of native resolution plus a billion-sample OCR corpus.

**Grounding and counting.** SOTA, with **LVIS-MG 73.8** versus Grounding DINO-L's 54.4 — the student beating its own labeling teacher.

**3D.** On SUN-RGBD it scores **33.5 AP@15**, ahead of Gemini 2.0 Pro Exp's 32.5 — modest, but evidence the depth-supervised 3D data works.

**Emergent vision-centric CoT.** A genuinely notable behavior emerged in RL with no SFT labels teaching it: the model learned to say things like *"let me look at the image again"* mid-reasoning, re-grounding its chain of thought in the pixels. This is the multimodal analogue of the "wait, let me reconsider" reflection that emerges in text RL, and it appeared without supervision.

## 10. Results: the benchmark matrix

Here is the head-to-head, comparing Seed1.5-VL in **think / non-think** modes against **Gemini 2.5 Pro**, **GPT-4o**, and **Qwen2.5-VL-72B**. (Baselines are April 2025 snapshots.)

<!-- FIGSPEC 7
kind: matrix
claim: Seed1.5-VL leads on perception-and-grounding benchmarks and trails only on broadest-knowledge tasks against three frontier baselines.
caption: A capability matrix: green where Seed1.5-VL leads (MathVista, V*, DocVQA, ScreenSpot-V2), amber where a frontier model wins (MMMU).
notes: rows = benchmarks (MMMU, MathVista, DocVQA, ChartQA, V*, ScreenSpot-V2); columns = Seed1.5-VL think/non-think, Gemini 2.5 Pro, GPT-4o, Qwen2.5-VL-72B; cells colored green=Seed leads, amber=baseline leads, gray=tie
nodes:
  - id: a | label: "MMMU 77.9 vs 81.7\nGemini leads" | color: amber
  - id: b | label: "MathVista 85.6\nSeed leads" | color: green
  - id: c | label: "DocVQA 96.9\nSeed leads" | color: green
  - id: d | label: "V* 89.0/89.5\nSeed leads" | color: green
  - id: e | label: "ChartQA 89.1\nQwen 89.5 edges" | color: amber
  - id: f | label: "ScreenSpot-V2 95.2\nbeats CUA 87.9" | color: green
edges:
  - a -> b | label: "(grid)"
notes: render as a colored grid/matrix of benchmark cells, not a flow; numbers per cell
-->
![Capability matrix: Seed1.5-VL think and non-think modes versus Gemini 2.5 Pro, GPT-4o, and Qwen2.5-VL-72B across MMMU, MathVista, DocVQA, ChartQA, V* and MMStar](/imgs/blogs/seed1-5-vl-native-resolution-vision-language-7.png)

The matrix above plots where Seed1.5-VL leads and trails. It is not uniformly ahead — Gemini 2.5 Pro wins on raw MMMU and MathVision (broad knowledge and hardest math), which the authors tie to scale. But Seed1.5-VL *leads* on a striking set: MathVista, V*, VLM-are-Blind, DocVQA, InfographicVQA, MMStar, RefCOCO, and the perception-heavy tasks.

| Benchmark | Seed1.5-VL (think / non-think) | Gemini 2.5 Pro | GPT-4o | Qwen2.5-VL-72B |
|---|---|---|---|---|
| MMMU | 77.9 / 73.6 | **81.7** | 70.7 | 70.2 |
| MMMU-Pro | 67.6 / 59.9 | **68.8** | 54.5 | 51.1 |
| MathVista | **85.6** / 83.0 | 82.7 | 63.8 | 74.8 |
| MathVision | 68.7 / 65.5 | **73.3** | 31.2 | 38.1 |
| V* | 89.0 / **89.5** | 79.1 | 73.9 | 86.4 |
| VLM-are-Blind | **92.1** / 90.8 | 84.3 | 50.4 | 69.0 |
| ChartQA | 89.1 / 87.4 | 83.3 | 86.7 | **89.5** |
| DocVQA | **96.9** / 96.7 | 94.0 | 66.2 | 96.4 |
| InfographicVQA | **91.2** / 89.3 | 84.3 | 79.2 | 87.3 |
| TextVQA | 81.8 / **84.2** | 76.8 | 81.4 | 83.5 |
| OCRBench | 861 / 881 | 866 | 806 | **885** |
| MMStar | **77.8** / 76.2 | 77.5 | 65.1 | 70.8 |
| HallusionBench | 60.3 / 60.0 | **63.7** | 56.2 | 55.2 |
| RefCOCO-avg | 91.3 / **91.6** | 74.6 | — | 90.3 |
| CountBench | **93.7** / 93.5 | 91.0 | 85.7 | 93.6 |

On video (Table 7 in the paper): VideoMME w/o sub **77.9** (Gemini 2.5 Pro leads at 87.0), MLVU **82.1**, LongVideoBench **74.0**, Video-MMMU **81.4**. On the team's internal benchmark, Seed1.5-VL scores **59.3 overall**, second to Gemini 2.5 Pro's 61.6, but ahead of o1 (54.0), o4-mini (55.4), and Claude 3.7 (48.6). The pattern is consistent: Seed1.5-VL wins the *perception-and-grounding* benchmarks decisively and trails only on the *broadest-knowledge* benchmarks, where parameter scale dominates.

### 10.1 Reading the table row by row

The aggregate "wins perception, trails knowledge" summary is true but coarse. The interesting story is *why each individual row lands where it does*, because every win or loss traces back to a specific architecture or data decision we have already covered. Walk the wins first.

**MathVista (85.6, leads Gemini's 82.7).** MathVista is dominated by *visual* math — reading a chart, a geometry figure, a plotted function — rather than abstract proof. This is exactly the intersection of native-resolution perception (you must read the figure precisely) and the >1M-problem visual-STEM RLVR set with sympy verification. The model was trained to *look carefully and then compute*, which is what MathVista rewards, so the lead is unsurprising once you know the recipe.

**V\* (89.0 / 89.5, leads Gemini's 79.1 by ten points).** V\* is a needle-in-a-haystack visual search benchmark: find a tiny target in a high-resolution scene. It is almost a pure test of native resolution. A tiling model loses the target at a seam or in the resize; Seed-ViT sees the whole high-res image in one pass, so the ten-point lead is the single clearest vindication of the from-scratch native-resolution encoder. Note the non-think mode (89.5) actually edges the think mode (89.0) — finding a target is perception, not reasoning, so chain-of-thought adds little.

**VLM-are-Blind (92.1, leads by eight points over Gemini's 84.3).** This benchmark probes whether a model actually *sees* low-level geometric facts (do two lines intersect, how many circles overlap) that humans find trivial but tiling/resize destroys. The 42-point gap over GPT-4o (50.4) is the starkest number in the table and is, again, native resolution plus grounding data doing exactly what they were built for.

**DocVQA (96.9) and InfographicVQA (91.2), both leading.** These are the direct payoff of the >1B-sample OCR corpus rendered and read at native resolution. Dense documents and infographics are where tiling seams cut text in half; with no seams and a billion text-images of training, the model reads them better than models thirty times the encoder size. The 30-point gap over GPT-4o on DocVQA (96.9 vs 66.2) is an OCR-corpus-and-resolution story end to end.

**MMStar (77.8, narrowly leads Gemini's 77.5).** MMStar is a curated, leakage-controlled multimodal benchmark that rewards genuine visual dependency. The narrow win signals that Seed1.5-VL's gains are real perception rather than text-shortcut exploitation — consistent with the rest of the perception cluster.

Now the losses, which are just as informative.

**MMMU (77.9, trails Gemini's 81.7).** MMMU is college-exam breadth across dozens of disciplines — the most knowledge-hungry benchmark in the set. This is precisely where the 20B-active scale bites: broad factual recall is a function of total capacity and pretraining tokens, and the authors say the loss had not saturated at 3T tokens. Gemini's larger scale buys the four-point lead. This is the loss the authors most directly attribute to scale.

**MathVision (68.7, trails Gemini's 73.3).** Unlike MathVista's *visual* math, MathVision leans toward harder, more abstract multi-step reasoning where the bottleneck is the reasoner's depth, not perception. The same capacity ceiling that caps MMMU caps this — though note Seed1.5-VL still crushes GPT-4o (31.2) and Qwen2.5-VL-72B (38.1) here, so the gap is specifically to the frontier reasoner, not to peers.

**OlympiadBench / hardest reasoning.** On olympiad-tier problems the multi-step combinatorial search and long proof chains exceed what a 20B-active model reliably sustains — the same root cause flagged in the limitations (Klotski, mazes, combinatorial search). Perception is not the bottleneck; reasoning depth is.

**HallusionBench (60.3, trails Gemini's 63.7).** HallusionBench specifically baits the model into trusting its language priors over the pixels. Seed1.5-VL's post-hoc adaptation — language pretrained first, vision grafted on — is exactly the design most exposed to this failure: when pixels conflict with a strong text prior, the inherited LLM sometimes wins. This row is the clearest benchmark cost of the post-hoc choice we discussed in §3.2.

**VideoMME without subtitles (77.9, trails Gemini's 87.0).** Video is the modality where Seed1.5-VL most consistently trails the frontier. The dynamic frame-resolution sampler gives excellent *coverage*, but VideoMME's hardest items demand sustained long-horizon temporal reasoning across many frames — temporal ordering and multi-frame interdependence are both in the acknowledged failure set. The model perceives the frames well; integrating them into a long causal chain is the gap. The nine-point deficit to Gemini is the largest single-row gap in the comparison, and it is a reasoning-over-time gap, not a perception gap.

The unifying read: every Seed1.5-VL *lead* is a perception or grounding row where native resolution and the data engine dominate, and every *loss* is a breadth-of-knowledge or depth-of-reasoning row where the 20B-active capacity ceiling dominates. The table is a clean projection of the architecture onto the benchmark axis.

## 11. Limitations and the 20B-active ceiling

The paper's honesty about failure is one of its better qualities, and the failures cluster meaningfully. Read them as a map of where a 20B-active VLM still breaks.

**Counting fails on irregular, occluded, or similar-colored objects.** When objects overlap or share a color, the model loses the per-instance boundaries it needs to count — the same boundary-detection weakness that grounding depends on. **Subtle inter-image differences** slip past it; the "Spot the Differences" RL game targets exactly this, and it remains hard. **Spatial relations under varying perspective** confuse it — "is A in front of B?" depends on viewpoint, and the model does not robustly reconstruct 3D from a single view. **Visual-prompt region mis-identification**: when you mark a region and ask about it, the model sometimes attends to the wrong region.

**Higher-level reasoning gaps** are the most fundamental: Klotski puzzles, mazes, and combinatorial search defeat it because they need multi-step search the model cannot reliably carry out. **3D reasoning and projection** are weak. **Visual-puzzle logic errors**, **planning that omits given conditions or invents unfounded assumptions**, **temporal ordering** in video, and **multi-image interdependent reasoning** all appear in the failure analysis. And the classic multimodal failure: **hallucination when visual input conflicts with the LLM's text priors** — when the pixels say one thing and the language model "knows" another, the model sometimes trusts its priors over its eyes.

The authors' own diagnosis of the broad-knowledge and reasoning gaps is the most important sentence in the limitations section: they attribute these to the **20B-active scale**, noting the **loss had not saturated at 3T tokens**. In other words, this is not an architectural dead end — it is a model that ran out of capacity and data before it ran out of headroom. More scale and more tokens should help. That is a clean, falsifiable claim, and it frames Seed1.5-VL as a strong point on a curve rather than its ceiling.

It is worth being precise about what "loss not saturated at 3T tokens" implies, because it is the load-bearing claim for the whole limitations section. A pretraining loss curve that is still descending when you stop means the model was *data-and-compute-limited*, not *architecture-limited*: had you fed it more tokens or more active parameters, the loss would have kept falling and the downstream gaps (MMMU breadth, MathVision depth, internal-bench knowledge/reasoning/code/captioning) would have narrowed. This is the opposite of a saturated curve, where adding scale yields diminishing returns and the weakness is structural. The team is making the falsifiable bet that their weaknesses live on the *climbing* part of the curve. The internal-bench breakdown supports this reading: the categories they call out as weak — knowledge, reasoning, code, captioning — are precisely the ones most sensitive to total capacity and token count, not to perception quality (which is already SOTA).

The tradeoff the team consciously made is the reason this ceiling exists at all. They could have built a 70B-active dense or larger-MoE model that would close the MMMU gap, but it would not serve interactively, could not afford 4096-rollout PPO episodes at reasonable cost, and would not ship as a latency-sensitive product like `doubao-1-5-thinking-vision-pro`. The 20B-active footprint is a product decision: keep the model cheap enough to *use* in a loop — for RL rollouts during training and for interactive agents at inference — and accept that the broadest-knowledge benchmarks will go to bigger models. Given that the perception and grounding rows are already SOTA, the bet is that scale is the *easy* axis to push later (you can always train a bigger sibling on the same recipe), whereas getting native-resolution perception and the hybrid-RL recipe right is the hard, recipe-level work that transfers up the scale ladder. The limitations, read this way, are not a verdict on the approach — they are an invoice for a deliberate, defensible scale choice, and the loss curve says the invoice is payable by simply scaling the same recipe.

> The encoder is small because it can be; the LLM is small because someone chose to ship it interactively. The limitations are mostly the bill for that second choice, and the loss curve says the bill is payable.

## 12. Case studies: what each design decision actually bought

To make the design concrete, here are the moments where a specific choice in the paper paid off (or revealed its edge), each tied to a number.

### 1. The 9%-param encoder that matched a 6B one

The symptom that drives most teams to oversize their encoder is a perception ceiling: small encoders read documents badly and miss small objects, so the reflex is to reach for a bigger backbone. Seed-ViT refuses the reflex. At **532M** parameters it scores **82.5 average zero-shot accuracy over 6 ImageNet variants**, matching InternVL-C-6B at roughly **9% of its parameter count** and staying competitive with EVA-CLIP-18B, a model **30× larger**. The wrong first hypothesis — the one almost everyone starts with — is "frontier perception requires a frontier-sized encoder." If that were true, a 532M encoder could not sit on the same line as a 6B one.

The actual root cause of the efficiency is the three-stage curriculum, and each stage contributes a different ingredient. Stage 1's masked-image-modeling-from-a-CLIP-teacher (EVA02-CLIP-E target, 75% mask, cosine-similarity reconstruction) builds genuine spatial structure cheaply: by forcing the encoder to predict a strong teacher's features for masked patches from only 25% of the image, it learns global layout rather than local texture memorization. Stage 2's native-resolution contrastive loss (SigLIP + SuperClass) installs the language alignment that makes the features zero-shot-probeable — this is where the 82.5 number is actually produced. Stage 3's MiCo omni-modal alignment, on only **4.8% of the token budget**, adds temporal and cross-modal richness that transfers back to single images. The second-order consequence is the one that matters for the rest of the system: because the encoder is small *and* good, it is cheap enough to run inside RL rollouts and interactive serving, which is what makes the 4096-rollout PPO episodes and the `doubao-1-5-thinking-vision-pro` product economics work at all. A 6B encoder would have made the whole downstream pipeline more expensive without improving the perception that is already SOTA. The lesson, stated plainly: encoder *quality* is a curriculum problem, not a parameter-count problem, and getting the curriculum right buys you a 10× parameter saving that compounds through every later stage.

### 2. Tiling seams that never existed

The classic tiling failure looks like this in production: you OCR a dense table, and a column header that happens to straddle the boundary between tile 3 and tile 4 comes back garbled — "Reve | nue" read as two unrelated tokens — because the two tiles were encoded in independent forward passes that never attended to each other. Engineers spend real time building seam-repair heuristics: overlapping tiles, post-hoc token stitching, thumbnail-guided reassembly. All of it is treatment for a self-inflicted wound.

Seed1.5-VL never has the wound, because Seed-ViT ingests whole images via NaViT-style packing with per-image attention masks. There are no tile boundaries, so there is nothing to split a character, a chart gridline, or a small object across. A 274×154-patch 4K screenshot is one contiguous patch sequence inside the encoder; every patch can attend to every other patch of the same image, so the "Revenue" header is encoded as one continuous span regardless of where it falls in the frame. The visible consequence is **DocVQA 96.9 (think mode)** and **InfographicVQA 91.2**, both SOTA — dense-document and infographic reading is *precisely* the task tiling seams ruin, and Seed1.5-VL leads GPT-4o on DocVQA by 30 points (96.9 vs 66.2). The second-order consequence is that the entire seam-repair toolchain — overlapping crops, stitching logic, the heuristics that consume engineering attention and still leak errors — simply does not exist in this codebase. The fix was *structural* (no tiling at all), not a patch on top of tiling, and that is the cleaner kind of fix: you do not debug a class of errors you cannot produce. The cost was paid up front, once, by training the encoder for native resolution from scratch; the benefit is amortized across every dense-document inference forever.

### 3. The 81,920-token budget that adapts to the clip

The failure a fixed policy produces is a benchmark split-personality: tune your frame-vs-resolution policy for one video regime and you regress on another. Set it to 8 high-res frames and you ace "summarize this lecture" but fail "when does the speaker change slides" because you only saw 8 of 600 frames. Set it to many low-res frames and you catch the slide transitions but can no longer read the slides themselves. Most video VLMs pick one corner and live with the regression on the other.

The dynamic frame-resolution allocator escapes the dilemma by holding the *compute* fixed (the **81,920-token** budget) and moving on the resolution axis instead. We worked the arithmetic earlier: a 600-frame, 10-minute clip at 1 FPS lands on the **256px** level (81 tok/frame → 48,600 tokens, comfortably under budget), reading every frame; a fast clip that needs 5 FPS drops to lower levels to fit; and only when even 128px overflows does it fall back to **uniform temporal downsampling**, dropping frames evenly so the video's ending is never silently truncated. The mechanism that makes this work is the six-level ladder {640, 512, 384, 256, 160, 128} plus the timestamp tokens that keep the frames ordered in time. The payoff is the thing a fixed policy structurally cannot deliver: **SOTA across short, streaming, *and* grounded video simultaneously**, handling up to 640 frames within the 128K context. The second-order consequence is operational predictability — because the token budget is fixed at 81,920 regardless of clip length or task, the inference cost and latency of a video request are bounded and known in advance, which is exactly the property an interactive product needs. A policy that scaled tokens with video length would have unbounded worst-case cost; the budget-bounded allocator trades a little per-frame resolution on long clips for a hard ceiling on compute, and that trade is what makes long-video serving viable.

### 4. Timestamp tokens turning frames into a timeline

Here is a subtle failure that bites video models the moment you sample frames non-uniformly: the model loses track of *time*. If you feed 600 frames at a steady 1 FPS, the model could in principle infer that frame index *i* occurred at second *i* — but the dynamic allocator does not always sample uniformly (it downsamples long clips, it varies FPS by task), so frame index is no longer a reliable proxy for wall-clock time. Ask "what happens at 0:47?" and a model that only knows frame *order* cannot answer, because it does not know which frame is 0:47.

Prefixing each frame with a **timestamp token** like `[1.5 second]` is a one-line change in the input pipeline with outsized downstream effect: it converts an unordered (or non-uniformly-ordered) bag of frames into an explicit timeline. The model reads, literally, "here is the frame at 1.5 seconds, here is the frame at 2.5 seconds," so temporal grounding becomes a lookup the model can actually perform. This is what makes streaming-video and grounded-video benchmarks tractable — questions like "at what timestamp does the red car enter?" require the model to map a visual event back to a clock value, which is impossible without the clock being in the input. The second-order consequence shows up in the interaction with the allocator: because timestamps are absolute, the allocator is *free* to drop frames or vary FPS without confusing the model about timing, since each surviving frame still carries its true timestamp. Without the timestamp tokens, the dynamic-sampling design and the temporal-grounding capability would be in tension — you could not have both non-uniform sampling and reliable time references. The timestamp token is the cheap glue that lets the two coexist, which is why it is the unsung hero of the whole video stack and a contributing factor to the SOTA streaming-video results.

### 5. Scoring only the solution to free the chain of thought

The failure this design avoids is reward-model-induced mode collapse of the chain of thought. If the reward model scores the *entire* response — CoT included — it inevitably develops preferences about *how* the model reasons: it rewards reasoning that looks like the reasoning in its training distribution, penalizes unconventional or exploratory steps, and over many PPO updates squeezes the policy toward a single house style of thinking. That is fatal for a reasoning model, because the whole value of long chain-of-thought is exploration — trying a wrong path and backtracking, re-examining the image, considering a second interpretation.

Seed1.5-VL sidesteps this by **stripping the `<think>{thought}</think>` block before the reward model ever sees the response** and scoring only the truncated `{solution}`. The format reward enforces *structure* (the response must be wrapped correctly, or it gets zero reward), but the *content* of the thought is completely unconstrained — the RM has no opinion about it because the RM never reads it. The policy is therefore free to reason however it wants as long as the final answer is good. The most striking evidence that this freedom is real is the **emergent vision-centric chain-of-thought**: in RL, with no SFT labels teaching the behavior, the model began saying things like *"let me look at the image again"* mid-reasoning, re-grounding its thought in the pixels before committing to an answer. That behavior could not have survived if the RM were scoring the CoT, because "let me look at the image again" is not the kind of polished reasoning step a CoT-scoring RM would reward — it would have been optimized away. The second-order consequence is that this design *combines* with the per-prompt-type KL and the verifiers: on verifiable tasks, solution-only scoring plus KL=0 means the policy can explore arbitrarily wild reasoning as long as the boxed answer is sympy-correct, which is the maximal-exploration regime. The lesson generalizes far beyond this paper: if you want a model to explore in its reasoning, do not let your reward model grade the reasoning — grade only the outcome.

### 6. The shared critic that made mixed-reward PPO coherent

The naive way to combine RLHF and RLVR is to run them as two separate training phases or two loops with two critics, and the failure mode is subtle: the two reward sources drift onto different scales, and when you eventually mix them, their advantages are no longer comparable. In PPO the critic estimates a value baseline and the advantage is reward-minus-baseline; if the RLVR rewards live on one scale and the RLHF rewards on another, a single shared policy update sees inconsistent advantages and the gradients from the two sources fight rather than reinforce.

Seed1.5-VL makes the mix coherent with two moves. First, **both reward sources are normalized to [0, 1]** — the verifiers (sympy match, IoU, regex) already return values in that range, and the generative reward model's scores are normalized to match — so a verifier-correct grounding answer and an RM-preferred caption are on the same numerical footing before they ever reach the critic. Second, there is **one shared critic for both**, initialized from the reward-model weights with a **100-step warmup**. Sharing the critic is what actually unifies the batch: because a single value function estimates the baseline for both reward types, the advantages it produces are commensurate, so an update that pushes the policy toward a sympy-correct STEM answer and an update that pushes it toward an RM-preferred response are measured against the same yardstick. The RM-weight initialization is a nice touch — the critic starts already understanding response quality rather than from random, which is why a mere 100-step warmup suffices. The second-order consequence is the whole point of hybrid RL: you can put RLHF prompts and RLVR prompts in the *same minibatch* (4096 rollouts per episode, minibatch 512) and run *one* PPO update over both, instead of alternating phases. That is simpler to operate, avoids the catastrophic-forgetting risk of phase-switching, and lets the two reward signals co-shape the policy continuously. Without the shared, normalized, RM-initialized critic, none of that is possible — the two reward scales would pull the value baseline in incompatible directions.

### 7. Per-prompt-type KL: leash off where you can verify

The KL penalty against the reference policy is the standard PPO leash that stops the model from drifting so far it forgets how to write — but it is a *blunt* instrument when applied uniformly. A single KL coefficient forces a compromise: large enough to prevent reward hacking on open-ended tasks, but that same large leash also *holds the model back* on verifiable tasks where you actually want it to move aggressively toward correct answers. The compromise leaves performance on the table at both ends.

Seed1.5-VL splits the coefficient by prompt type: **KL = 0 on verifiable (RLVR) prompts, KL = 1e-5 on general (RLHF) prompts**. The asymmetry encodes a genuine insight about *why* you need the leash at all. The leash exists to prevent reward hacking — the policy finding a degenerate way to score high reward that the reward model failed to anticipate. But on verifiable tasks there is *nothing to hack*: a sympy equality check or an IoU computation cannot be gamed by fluent nonsense; the answer is either symbolically correct or it is not. So you remove the leash entirely (KL=0) and let the policy sprint toward provably-correct answers, exploring however wildly it likes in the unscored CoT. On open-ended tasks, by contrast, the generative reward model *can* in principle be fooled, and a model with no leash could drift toward verbose, sycophantic, or off-distribution text that games the RM, so you keep a light leash (1e-5) to anchor it near the fluent reference policy. The result is the best of both: aggressive, unconstrained improvement on STEM and grounding (where you have a ground-truth verifier) and stable, hack-resistant improvement on open-ended generation (where you do not). The second-order consequence pairs with case study 5: KL=0 plus solution-only scoring on verifiable prompts is the *maximal exploration* configuration — no leash on the policy, no reward-model opinion about the reasoning — which is exactly the regime in which emergent behaviors like re-examining the image are most likely to appear and survive. The two design choices are not independent tricks; they compose into a single coherent stance: where you can verify, set the model free.

### 8. The 0%-to-75% accuracy filter as automatic curriculum

The failure this filter prevents is wasted compute, and at the scale of **>1M visual-STEM problems** it is an expensive failure. In PPO, a problem only produces a gradient if the rollouts *disagree* about it — some succeed, some fail — because the advantage is computed relative to the rollout-group baseline. A problem the model solves on all 16 rollouts has every advantage equal to zero (all rewards are 1, baseline is 1); a problem it fails on all 16 also has every advantage zero (all rewards 0, baseline 0). Both contribute exactly nothing to the policy update while still consuming a full 16-rollout generation budget. Train naively on the raw pool and a large fraction of your rollouts are dead weight.

The fix is to pre-screen every candidate problem with **16 rollouts** and keep only those with accuracy in the open interval **(0%, 75%]** — problems the model can sometimes but not reliably solve. This is curriculum construction with no human in the loop: the kept set is, by definition, the frontier of the policy's current ability, where rollouts disagree and gradients are richest. The choice of the upper bound at 75% rather than, say, 99% is deliberate — problems the model already solves three times out of four have little headroom and a weak, noisy gradient, so trimming them at 75% concentrates the budget on the genuinely contested problems. The 0% floor is hard: a problem the model never solves gives no positive example to imitate, so it is pure wasted budget. The second-order consequence is that the curriculum is *adaptive over training*: as the policy improves, problems that were once in the (0%, 75%] band climb out the top (the model now solves them reliably) and harder problems that were once at 0% climb in the bottom (the model can now sometimes solve them). Re-screening therefore shifts the frontier upward automatically, which is why the filter is paired with the iterative SFT↔RL loop and hard-prompt mining. The lesson: in verifiable RL, your effective training-set size is not the number of problems you *have*, it is the number on which your current policy's rollouts disagree — and a cheap 16-rollout filter is how you find them.

### 9. Order-debiasing the generative reward model

Position bias is a quietly catastrophic failure in preference-based reward models, and it is easy to miss because it does not announce itself. A list-wise or pairwise judge that reads two candidate responses and picks the better one tends to systematically prefer whichever it saw *first* (or, for some models, last) — an artifact of attention and decoding, not of quality. If that bias leaks into the reward model, RLHF optimizes the policy toward whatever the RM spuriously favors, and you get a model that drifts in a direction nobody intended, with the usual end state being reward hacking — the policy learns to exploit the RM's quirk rather than to actually improve.

Seed1.5-VL attacks this with two compounding choices. First, the reward model is **generative** rather than Bradley-Terry: instead of a scalar regression head that collapses everything to one number, the RM is itself a VLM that reads the prompt and the candidate and *emits a preference token*, letting it reason in-context before committing — a formulation the team found more robust than the scalar approach. Second, and this is the cheap, decisive fix, the RM is **order-debiased by scoring both orderings of every pair and averaging**. If the RM has a first-position bias of magnitude *b*, then scoring (A, B) inflates A by *b* and scoring (B, A) inflates B by *b*; averaging the two cancels *b* exactly, leaving the true quality difference. It costs one extra forward pass per pair — trivial relative to the cost of a corrupted reward signal — and it removes an entire class of systematic error. The second-order consequence is what makes the rest of the post-training viable: because the reward model is robust, the policy can be optimized against it for many PPO steps without the RM's biases compounding into a reward-hacking spiral, and the light KL=1e-5 leash on RLHF prompts (case study 7) is *enough* precisely because the RM it is leashing toward is trustworthy. A biased RM would have required a heavier leash, which would have throttled improvement. Order-debiasing is therefore not a standalone trick but a load-bearing piece that lets the whole RLHF half of the hybrid loop run aggressively. The lesson: before you tune anything against a preference model, measure its position bias and cancel it — it is the cheapest robustness win in the RLHF stack.

### 10. The non-thinking transfer

The expectation going into RL was straightforward: train on LongCoT data, improve the thinking mode, and leave the concise (non-thinking) mode roughly where SFT left it. The surprise was that **RL trained only on LongCoT data, yet the non-thinking responses also improved** — the gain transferred across the mode boundary for free. The clearest single data point is **V\***, where the non-think score (**89.5**) actually *exceeds* the think score (**89.0**): a perception-heavy benchmark where the model's concise-mode answers came out ahead of its deliberate-mode answers, despite RL never optimizing the concise mode directly.

The mechanism worth understanding is *why* this transfer happens, because it is not obvious that it should. The thinking and non-thinking modes are not two separate models — they are one set of weights conditioned on a system prompt. When RL on LongCoT improves the model's underlying perception and grounding (it learns to read the image more carefully, to localize more precisely, to extract the right boxed answer), those improvements live in the *shared weights*, not in the CoT tokens. The concise mode draws on exactly the same perception-and-grounding substrate; it just skips the explicit reasoning trace. So when RL sharpens the substrate, both modes inherit the sharpening. On a perception benchmark like V\*, where the bottleneck is *seeing* rather than *reasoning*, the concise mode can even come out ahead because it avoids the small risk that a long chain of thought introduces an error or overthinks a target that was visible at a glance. The second-order consequence is a genuine product win: the dual-mode design lets a user toggle LongCoT on for hard reasoning and off for fast interactive perception, and the off mode is not a degraded afterthought — it benefited from all the RL compute too. This is the free lunch the toggle architecture enables, and it is why the team could ship a single model that is both a careful reasoner and a snappy perceiver rather than maintaining two. The lesson: when capabilities live in shared weights, optimizing one usage mode can lift the others, so a unified model with a mode toggle can beat two specialized models on total capability per parameter.

### 11. Long-tail rebalancing swinging a domain 17 points

The failure here is invisible until you measure it: a domain that is *present* in your training corpus but *underrepresented* gets quietly starved, and the model underperforms on it for no reason an architecture diagram would reveal. In a web-scale, synthetic-heavy corpus the frequency distribution across domains is brutally long-tailed — common scenes appear millions of times, rare specialist domains (a particular kind of biology imagery, an unusual chart type) appear far less. Random sampling proportional to frequency means the model sees the rare domain so seldom that it never properly learns it, even though the data is right there in the pool.

The fix is **long-tail rebalancing by alt-text duplication**: domains whose frequency falls below 50% of the average have their alt-text-derived captions duplicated to lift their effective sampling weight, flattening the tail. The ablation isolates the effect cleanly because it holds the architecture fixed and varies only the sampling: on the **Biotrove** biology benchmark, a capped **Max1k-46M** sampling strategy scores **62.01** versus **44.69** for random sampling — a **17-point swing** driven entirely by how the long tail was sampled. Seventeen points is an enormous gap to leave on the table for a domain whose data you already possess; it is larger than the gap between many models on the leaderboard, purchased with a data-loader change rather than any new model or compute. The second-order consequence is a discipline point about how to spend a synthetic-data budget: it is tempting to chase coverage by *generating more* rare-domain data, but the Biotrove result shows that *resampling* the data you have can recover most of the gap far more cheaply. The deeper lesson for synthetic-heavy training is that the long tail is exactly where benchmark points hide — the head domains are saturated and contested, so marginal model quality on the *tail* is what separates models — and how you sample that tail can matter as much as the architecture you wrap around it. A team that obsesses over the encoder and ignores the sampler is optimizing the wrong variable.

### 12. Beating the labeling teacher

The intuition most people hold is that a student trained on a teacher's labels is upper-bounded by the teacher — you cannot learn to be better at a task than the annotations you learned from. Seed1.5-VL violates that intuition cleanly: on **LVIS-MG** it scores **73.8** versus **Grounding DINO-L's own 54.4**, and recall from the data engine that roughly **200M of the grounding samples (200B tokens) were auto-annotated by Grounding DINO**. The model trained on Grounding DINO's boxes beats Grounding DINO at the task by nearly 20 points. How does the student exceed the teacher?

The mechanism is that the student and teacher are not solving the same problem. Grounding DINO is an open-vocabulary detector: it proposes boxes for objects, largely class-agnostic and without rich language conditioning. The VLM, by contrast, grounds *conditioned on language and context* — "count the chairs that are occupied," "find the leftmost red sign," "which of these is a defect" — and it brings the full reasoning of a 20B-active LLM plus native-resolution perception to bear on that conditioned task. The teacher's auto-annotations are noisy and class-agnostic, but they are *plentiful* (200M samples), and at that scale the student learns the general skill of mapping a region to a box while *also* learning, from the rest of its training, the language and reasoning that the detector never had. The aggregate effect is that the student's task-conditioned, context-aware grounding exceeds the teacher's context-free boxes — the noise in any single label is averaged out across 200M samples, while the *capability* the student layers on top (language conditioning) is something the teacher structurally lacks. This is the same student-beats-teacher dynamic that makes weak supervision and distillation-from-noisy-labels work in general: scale plus a more capable student architecture can transcend the label source. The second-order consequence is a scalability unlock — you cannot hand-label 200M bounding boxes, but you do not have to; a good-enough detector bootstraps the labels and the VLM transcends them. The lesson: a noisy automatic labeler is not a quality ceiling if your student is more capable than the labeler and you give it enough data to average out the noise. That is the entire reason the grounding/counting branch could be built by automation at all, and it is why Seed1.5-VL leads on grounding and counting despite never seeing a single hand-drawn box at that scale.

## When to reach for this recipe / when not to

**Reach for the Seed1.5-VL recipe when:**

- You need a VLM that reads **dense documents, charts, and UI at native resolution** and you are tired of tiling seams — train (or use) a native-resolution-from-scratch encoder rather than retrofitting CLIP.
- You are building a **GUI or gameplay agent** and need closed-loop perception-action; the UI-TARS-lineage data plus screen-grounding RL is the proven path, and a 20B-active model is enough to be competitive.
- You want **interactive serving** — a 532M encoder and 20B-active LLM keep latency and rollout cost low enough for RL and for production, unlike a 72B dense VLM.
- You have **both verifiable and open-ended tasks** to optimize and want them in one training run — the hybrid RLHF+RLVR PPO loop with a shared critic and per-prompt-type KL is the design to copy.
- You need **long video** understanding; the 128K context plus dynamic frame-resolution sampling with a fixed token budget is the scalable tokenizer.

**Skip it when:**

- Your task is **broadest-possible-knowledge** reasoning (frontier MMMU, hardest math) — here parameter scale wins, and Gemini 2.5 Pro's lead on MMMU/MathVision is exactly the 20B-active ceiling the authors acknowledge.
- You **cannot afford to pretrain a vision encoder** — the from-scratch route costs 1.3M H800-hours of VLM pretraining alone; if you only need decent perception, retrofitting an existing native-res encoder ([Qwen2-VL](/blog/paper-reading/multimodal/qwen2-vl-enhancing-vision-language-models-perception-of-the-world-at-any-resolution)-style) is far cheaper.
- Your workload is **single low-resolution images** with no documents, UI, or fine detail — native resolution and a billion-sample OCR corpus buy you nothing, and a smaller fixed-res model suffices.
- You need **provable 3D reasoning, combinatorial search, or multi-image interdependent logic** — these are in the model's acknowledged failure set, and no amount of prompt engineering fixes a capacity-limited reasoner.

The enduring contribution of Seed1.5-VL is not any single benchmark line. It is the demonstration that the two least-loved decisions in modern VLMs — *retrofit the encoder* and *run RLHF and RLVR separately* — are both avoidable, and that avoiding them buys you a model small enough to serve interactively yet good enough to act as a frontier agent. The encoder is native-resolution from scratch because native resolution is too important to bolt on. The RL is one loop because RLHF and RLVR want the same critic. And the limitations are, by the authors' own loss curve, a budget line rather than a wall.
