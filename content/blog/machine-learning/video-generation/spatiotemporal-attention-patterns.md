---
title: "Spatiotemporal Attention Patterns: Full 3D vs Factorized, and Where the FLOPs Go"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Derive the cost of full 3D versus factorized spatiotemporal attention, implement both in PyTorch, and learn exactly where a video model's FLOPs and VRAM disappear."
tags:
  [
    "video-generation",
    "diffusion-models",
    "attention",
    "spatiotemporal-attention",
    "transformers",
    "video-diffusion",
    "generative-ai",
    "deep-learning",
    "flash-attention",
  ]
category: "machine-learning"
subcategory: "Video Generation"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/spatiotemporal-attention-patterns-1.png"
---

The first time I trained a video diffusion transformer, I made the obvious mistake: I took a working image DiT, stacked the latent frames into one long sequence, and let attention mix every spacetime token with every other one. The math was beautiful. The model was coherent in a way factorized models never quite are — objects held their shape across frames, motion looked physically continuous, and the background did not boil. Then I tried to scale from a 2-second clip to a 5-second one, and the run OOMed before the first optimizer step. Not the VAE. Not the MLP. The attention score matrix alone wanted 31 GB.

That is the whole story of this post in miniature. In a video model, **how you arrange attention over space and time is the single biggest lever you have over both coherence and cost** — and the two pull in opposite directions. Full 3D attention, where every token in the `$T \times H \times W$` latent volume attends to every other, gives you the best temporal coherence money can buy, and it costs `$O(N^2)$` in `$N = T \cdot H \cdot W$` tokens, which becomes intractable shockingly fast. Factorized attention — alternate a spatial pass inside each frame with a temporal pass across frames — cuts that to `$O(N \cdot (HW + T))$`, roughly a hundredfold cheaper on a real clip, but it can no longer model every cross-frame interaction directly. Windowed and sparse variants sit in between. This is the central architectural decision of any video diffusion model, and it is the second wall you hit after the VAE.

Figure 1 frames the choice. By the end of this post you will be able to derive the FLOP and memory cost of each attention pattern from first principles, implement spatial, temporal, and factorized attention blocks in PyTorch with FlashAttention, read a model card and know immediately which pattern it uses and why, and make the coherence-versus-compute call for your own clip-length and hardware budget.

![Diagram contrasting full 3D attention as one quadratic joint pass against a factorized pipeline of a spatial pass followed by a temporal pass producing a much cheaper output](/imgs/blogs/spatiotemporal-attention-patterns-1.png)

This is part of the [Video Generation, From First Principles to the Frontier](/blog/machine-learning/video-generation/why-video-generation-is-hard) series. It assumes you are comfortable with the basics of [diffusion transformers](/blog/machine-learning/image-generation/diffusion-transformers-dit) and how a video model [adds the time axis on top of image diffusion](/blog/machine-learning/video-generation/from-image-diffusion-to-video-diffusion). If those are fuzzy, skim them first — I will not re-derive self-attention or the DiT block here. What I will do is take the cost of attention seriously, because in the video regime it stops being a footnote and becomes the thing that decides whether your model fits on a GPU at all.

## 1. The running example: a 5-second 720p clip

Let me anchor everything to one concrete clip, the same one I will return to for every number in this post. We want to generate **5 seconds of 720p video at 24 fps**: that is `$120$` frames at `$720 \times 1280$` pixels. Nobody runs attention on raw pixels — that is the entire reason the [causal 3D-VAE](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression) exists. The VAE compresses the clip spatially and temporally before the denoiser ever sees it, and the compression ratio it achieves is what makes the token count we are about to compute even remotely tractable.

A typical modern video VAE (CogVideoX, HunyuanVideo, Wan all converge here) uses a compression factor of `$4 \times 8 \times 8$` — `$4\times$` in time, `$8\times$` in each spatial dimension. So our latent tensor has shape:

$$
T = \frac{120}{4} = 30, \quad H = \frac{720}{8} = 90, \quad W = \frac{1280}{8} = 160.
$$

The DiT then patchifies this latent. With a spatial patch size of `$2 \times 2$` and a temporal patch size of `$1$` (a common choice — Sora-style spacetime patches; some models also patch in time), the number of tokens the transformer actually attends over is:

$$
N = \frac{T}{p_t} \cdot \frac{H}{p_s} \cdot \frac{W}{p_s} = \frac{30}{1} \cdot \frac{90}{2} \cdot \frac{160}{2} = 30 \cdot 45 \cdot 80 = 108{,}000.
$$

So our running clip is **108,000 tokens**. Hold that number. For a moment, compare it to an image: a single `$1024 \times 1024$` image at the same VAE and patch settings is `$\frac{1024}{8 \cdot 2} \cdot \frac{1024}{8 \cdot 2} = 64 \cdot 64 = 4{,}096$` tokens. Video is **26× longer in sequence**, and attention is quadratic, so naively video attention is `$26^2 \approx 680\times$` more expensive than image attention per layer. That factor — not the VAE, not the MLP — is the reason video models architect their attention so carefully.

Let me also fix the patchified factorization dimensions I will reuse:

$$
T = 30, \quad HW = 45 \cdot 80 = 3{,}600, \quad N = T \cdot HW = 108{,}000.
$$

Throughout, `$d$` is the model hidden dimension; I will use `$d = 1{,}152$` (the DiT-XL width, a realistic video-DiT size) for concrete FLOP numbers.

### Why attention is the second wall

The series spine is **video = (spatial generation) × (temporal coherence) under a brutal compute budget**, and the stack is **data → causal 3D-VAE → spacetime-patch DiT → flow-matching sampler → conditioning → frames**. The 3D-VAE is the *first* wall: it decides your token count `$N$` by setting the compression ratio, and getting it wrong (too little compression) means there is no token budget left to spend. But once the VAE has done its job and handed the DiT a latent of `$N$` tokens, **attention is the second wall**, because the denoiser runs that latent through dozens of transformer blocks, every block does attention, and attention is the one operation whose cost is quadratic in `$N$`. The MLP is linear in `$N$`. The convolutions are linear in `$N$`. Only attention squares it. So when you double the clip length — double `$T$`, double `$N$` — the MLP cost doubles but the full-3D attention cost *quadruples*. That asymmetry is what this post is about.

Make that asymmetry concrete with our clip. Going from 5 seconds (`$T = 30$` latent frames, `$N = 108{,}000$`) to 10 seconds (`$T = 60$`, `$N = 216{,}000$`) doubles the tokens. The MLP and the VAE both get `$2\times$` more expensive — annoying but linear. Full 3D attention gets `$2^2 = 4\times$` more expensive: from 54 TFLOPs/layer to 216 TFLOPs/layer. Factorized attention, by contrast, gets only `$\approx 2\times$` more expensive, because its cost `$4Nd(HW + T)$` is dominated by the `$N \cdot HW$` term and `$N$` only doubled (the `$T$` term grew but it is a rounding error). So the *length scaling* of the two patterns diverges: full 3D scales quadratically in clip length, factorized scales linearly. That divergence is why every attempt to make long video — tens of seconds, minutes — eventually abandons or windows the full-3D pattern. You simply cannot pay `$O(\text{length}^2)$` for a minute-long clip.

## 2. Full 3D attention: the gold standard and its quadratic doom

Start with the ideal. **Full 3D attention** (also called joint spacetime attention or full attention) treats the entire patchified latent as one flat sequence of `$N$` tokens and runs standard self-attention over it. Every spacetime token `$i$` computes a query `$q_i$`, scores it against the key `$k_j$` of *every* other token `$j$` in the whole `$T \times H \times W$` volume, softmaxes, and aggregates values. There is no notion of "this token is in frame 3 and that one is in frame 17" — they are all just tokens, and they all see each other.

This is exactly what you want for coherence. A pixel of the dog's left ear in frame 1 can directly attend to the same ear in frame 90, so the model can enforce that the ear keeps the same shape, color, and position-modulo-motion across the entire clip. Long-range temporal dependencies — an object that leaves frame and re-enters, a shadow that must stay consistent with a light source three seconds earlier — are modeled *directly*, in a single attention operation. Figure 2 shows what one query token "sees": the whole volume.

![Grid of tokens across three time steps and three spatial positions with one highlighted query token connected to every other token to show full 3D attention covers the whole spacetime volume](/imgs/blogs/spatiotemporal-attention-patterns-2.png)

### The cost derivation

The cost of self-attention over a sequence of length `$N$` and hidden dimension `$d$` is dominated by two matrix multiplies. The score matrix `$QK^\top$` is `$(N \times d) \times (d \times N) = N \times N$`, costing `$2 N^2 d$` FLOPs (the factor of 2 because a multiply-add is two FLOPs). The value aggregation `$\text{softmax}(QK^\top) V$` is `$(N \times N) \times (N \times d) = N \times d$`, costing another `$2 N^2 d$`. There are also the `$Q$`, `$K$`, `$V$`, and output projections, each `$2 N d^2$`, but those are linear in `$N$` and get swamped once `$N \gg d$`. So:

$$
\text{FLOPs}_{\text{full 3D}} \approx 4 N^2 d + 8 N d^2 \approx 4 N^2 d \quad (\text{since } N \gg d).
$$

Plug in our clip: `$N = 108{,}000$`, `$d = 1{,}152$`.

$$
4 N^2 d = 4 \cdot (1.08 \times 10^5)^2 \cdot 1152 \approx 4 \cdot 1.166 \times 10^{10} \cdot 1152 \approx 5.4 \times 10^{13} \text{ FLOPs per layer.}
$$

That is **54 TFLOPs for a single attention layer**. A video DiT has perhaps 30–48 such layers. At, say, 40 layers that is `$2.1 \times 10^{15}$` FLOPs — 2.1 PFLOPs — just for the attention score-and-aggregate, for a single denoising step, at a single CFG branch. With 50 sampling steps and classifier-free guidance (2 branches), multiply by 100: `$2.1 \times 10^{17}$` FLOPs to render one 5-second clip with full 3D attention. On an H100 delivering ~`$1 \times 10^{15}$` bf16 FLOP/s at maybe 40% utilization, that is `$\frac{2.1 \times 10^{17}}{4 \times 10^{14}} \approx 525$` seconds — nearly nine minutes — and that is *only the attention*, ignoring MLP, VAE, and text encoding. This is why I waited eight minutes for a render that OOMed at second 6.

### The memory wall is worse than the FLOP wall

FLOPs you can throw more GPU-seconds at. Memory you cannot — it is a hard ceiling. Naive attention materializes the full `$N \times N$` score matrix. For our clip:

$$
N^2 = (1.08 \times 10^5)^2 = 1.166 \times 10^{10} \text{ scores.}
$$

In fp32 (4 bytes), that is `$4.66 \times 10^{10}$` bytes ≈ **46.6 GB for one attention head's score matrix in one layer**. Even in bf16 it is 23 GB. You cannot fit that on any single consumer GPU, and on an 80 GB A100 it leaves no room for anything else. This is the number from my OOM. FlashAttention rescues you here — it never materializes the full score matrix, computing attention in tiles and keeping memory at `$O(N)$` instead of `$O(N^2)$` — but FlashAttention saves *memory*, not *FLOPs*. The compute is still `$4 N^2 d$`. You still wait nine minutes. So full 3D attention is the coherence gold standard and the compute catastrophe, and the entire rest of this post is about how the field buys most of the coherence for a fraction of the compute.

## 3. Factorized attention: trading exact coherence for a hundredfold saving

The key observation is that the `$N \times N$` interaction is enormously redundant. Most of the information about how a token relates to others is captured by two cheaper relationships: how it relates to other tokens **in the same frame** (spatial structure — edges, objects, layout), and how it relates to the token at **the same spatial position in other frames** (temporal structure — motion, persistence). **Factorized attention** models these two separately. It runs two attention operations per block:

1. **Spatial attention.** Reshape the latent so that frames are independent and the sequence axis is the `$HW$` tokens *within* one frame. Run self-attention over `$HW$` tokens, `$T$` times in parallel (batched over the time axis). Cost per frame: `$O((HW)^2 d)$`. Total: `$O(T \cdot (HW)^2 \cdot d)$`.
2. **Temporal attention.** Reshape so that the sequence axis is the `$T$` frames *at a fixed spatial position*. Run self-attention over `$T$` tokens, `$HW$` times in parallel (batched over space). Cost per position: `$O(T^2 d)$`. Total: `$O(HW \cdot T^2 \cdot d)$`.

Add them. Using the score-and-aggregate `$4 \cdot (\text{len})^2 d$` per attention:

$$
\text{FLOPs}_{\text{factorized}} \approx \underbrace{4 \, T (HW)^2 d}_{\text{spatial}} + \underbrace{4 \, (HW) T^2 d}_{\text{temporal}} = 4 \, T \cdot HW \cdot d \, (HW + T) = 4 N d (HW + T).
$$

Compare term-by-term to full 3D, which is `$4 N^2 d = 4 N d \cdot N = 4 N d \cdot (T \cdot HW)$`. The ratio is:

$$
\frac{\text{FLOPs}_{\text{full 3D}}}{\text{FLOPs}_{\text{factorized}}} = \frac{N}{HW + T} = \frac{T \cdot HW}{HW + T}.
$$

This is the central equation of the post. Plug in the clip: `$T = 30$`, `$HW = 3600$`.

$$
\frac{T \cdot HW}{HW + T} = \frac{30 \cdot 3600}{3600 + 30} = \frac{108{,}000}{3{,}630} \approx 29.8.
$$

So at *this* resolution factorization is ~30× cheaper. But the saving grows with resolution. At 1080p latent (`$HW \approx 8100$`) and the same `$T$`, the ratio is `$\frac{30 \cdot 8100}{8100 + 30} \approx 29.9$` — wait, almost the same? Let me be careful: the ratio is `$\frac{T \cdot HW}{HW + T}$`, and when `$HW \gg T$` it simplifies to `$\frac{T \cdot HW}{HW} = T$`. So **the saving from factorization is approximately `$T$`, the number of latent frames**, when space dominates. Our clip has `$T = 30$`, hence ~30×. A longer clip with `$T = 100$` latent frames gives ~100× saving. That is the "roughly a hundredfold" from the intro — it is exactly the case where `$HW \gg T$`, which holds for any high-resolution clip of meaningful length. Figure 3 shows the before/after.

![Before and after comparison showing full 3D attention as a single quadratic pass over all tokens versus factorized attention as a spatial pass of length HW and a temporal pass of length T with indirect cross-frame coherence](/imgs/blogs/spatiotemporal-attention-patterns-3.png)

#### Worked example: FLOP saving on the 5-second clip

Full 3D, one attention layer: `$4 N^2 d = 4 \cdot (108{,}000)^2 \cdot 1152 \approx 5.37 \times 10^{13}$` FLOPs.

Factorized, one block (spatial + temporal): `$4 N d (HW + T) = 4 \cdot 108{,}000 \cdot 1152 \cdot (3600 + 30) \approx 4 \cdot 108{,}000 \cdot 1152 \cdot 3630 \approx 1.81 \times 10^{12}$` FLOPs.

Ratio: `$\frac{5.37 \times 10^{13}}{1.81 \times 10^{12}} \approx 29.7\times$`. The factorized block does in 1.8 TFLOPs what full 3D does in 54 TFLOPs. Over 40 layers, 50 steps, 2 CFG branches, that turns ~525 seconds of attention compute into roughly 18 seconds. The difference between "this model is unusable" and "this model ships."

### What factorization gives up

There is no free lunch. Full 3D attention can route information between *any* two tokens in one step — the dog's ear in frame 1 directly to the dog's ear in frame 90. Factorized attention cannot do that in one step. Spatial attention only connects tokens *within* a frame; temporal attention only connects tokens at the *same spatial position* across frames. So to get information from (frame 1, ear position) to (frame 90, a *different* ear position because the dog moved), the model must compose: temporal attention carries it along the time axis at the original position, then spatial attention in frame 90 moves it to the new position — but that requires the information to "wait" at the original spatial coordinate and then hop. In a single block, factorized attention models **separable** spatiotemporal interactions cleanly and **non-separable** ones (motion that couples space and time, like a fast-moving object) only indirectly, through stacked blocks.

In practice this is why factorized models can flicker or "swim" under large motion: when an object moves several latent pixels per frame, the same-position assumption of temporal attention breaks, and the model has to reconstruct the correspondence through depth. Deep enough stacks recover most of it. But the failure mode is real, and it is the coherence cost you pay for the `$T\times$` FLOP saving. I have watched a factorized model render a slow pan beautifully and then turn a fast-spinning wheel into a shimmering blur, because the wheel's spokes moved too far between frames for same-position temporal attention to track.

### Why factorization works at all: the low-rank argument

It is worth pausing on *why* you can throw away the `$N \times N$` joint attention and still get good video, because the answer tells you exactly when factorization will fail. Write the full attention's logit between token `$i = (t, p)$` (frame `$t$`, spatial position `$p$`) and token `$j = (t', p')$` as a score `$s_{(t,p),(t',p')}$`. Full 3D attention learns this whole 4-index tensor of scores freely. Factorized attention does not — it constrains the score to a **separable** form. Spatial attention contributes a logit `$a_{p,p'}$` that depends only on the two spatial positions (shared across all frames), and temporal attention contributes a logit `$b_{t,t'}$` that depends only on the two frame indices (shared across all positions). The factorized model's effective two-step score is, roughly, an *additive composition* `$s_{(t,p),(t',p')} \approx a_{p,p'} \cdot \mathbb{1}[t = t'] + b_{t,t'} \cdot \mathbb{1}[p = p']$` within a single block — it can score a same-frame pair (via `$a$`) or a same-position pair (via `$b$`), but a *different-frame, different-position* pair gets no direct logit at all in one block.

So the question "does factorization lose anything" becomes "is the true score tensor approximately separable, i.e. low-rank in the spacetime-pairing sense?" For most of video, **yes**, and here is the intuition made precise. Natural video has enormous **temporal redundancy** — consecutive frames are nearly identical, differing by small local motion (the [redundancy-and-tokens post](/blog/machine-learning/video-generation/representing-video-redundancy-and-tokens) quantifies this as the bitrate prior). When motion is small, the token at `$(t', p')$` that a query at `$(t, p)$` most needs to attend to is the one at `$p' \approx p$` (same position, adjacent frame) — exactly the pair temporal attention scores directly. The genuinely non-separable interactions — a query needing a token at a *substantially different* position in a *different* frame — only arise under **large motion**, where the corresponding content has moved far across the frame. That is precisely the regime where factorized models break, and it is not a coincidence: it is the mathematical signature of the separability assumption failing. The rank of the true score tensor stays low while motion is small and rises with motion magnitude, and the factorized model's one-block approximation degrades in lockstep.

This gives a clean predictive rule: **factorization is near-lossless when per-frame motion is smaller than the spatial attention's effective receptive field, and degrades as motion grows past it.** It also explains why stacking helps — each additional factorized block composes another temporal-then-spatial hop, so a depth-`$L$` stack can route information across up to `$L$` position-shifts, recovering progressively more of the non-separable score tensor. Full 3D needs depth 1 for any interaction; factorized needs depth proportional to how far content moves. For a 5-second clip of slow, cinematic motion, a 30-block factorized stack recovers essentially all the coherence full 3D would give. For a 5-second clip of a hummingbird's wings, it does not, and you either pay for full 3D or accept the blur.

## 4. Implementing the three attention blocks in PyTorch

Enough math — let me make this concrete. The whole trick of factorized attention is **reshaping the tensor so the right axis becomes the sequence axis**, then calling the same scaled-dot-product attention kernel. PyTorch's `F.scaled_dot_product_attention` (SDPA) dispatches to FlashAttention or memory-efficient attention automatically when the shapes and dtypes allow, so we get the memory savings for free.

Here is full 3D attention. The latent comes in as `$(B, T, H, W, C)$`; we flatten the spacetime axes into one sequence and attend.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Full3DAttention(nn.Module):
    """Joint spacetime self-attention: every token attends to every other."""

    def __init__(self, dim: int, num_heads: int = 16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H, W, C)
        B, T, H, W, C = x.shape
        N = T * H * W
        x = x.reshape(B, N, C)                       # flatten spacetime -> one sequence
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each: (B, heads, N, head_dim)

        # SDPA dispatches to FlashAttention; memory is O(N), compute still O(N^2)
        out = F.scaled_dot_product_attention(q, k, v)   # (B, heads, N, head_dim)

        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out.reshape(B, T, H, W, C)
```

The cost lives entirely in that one SDPA call over a sequence of length `$N = THW$`. For our clip `$N = 108{,}000$`, and even FlashAttention will be slow because the FLOPs are irreducibly `$O(N^2)$`.

Now spatial attention. The move is to **fold time into the batch** so each frame is attended independently, with the sequence axis being the `$HW$` tokens within a frame.

```python
class SpatialAttention(nn.Module):
    """Attention within each frame; frames are independent (folded into batch)."""

    def __init__(self, dim: int, num_heads: int = 16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H, W, C)
        B, T, H, W, C = x.shape
        S = H * W                                    # spatial sequence length
        # fold T into the batch: each frame attended on its own
        x = x.reshape(B * T, S, C)
        qkv = self.qkv(x).reshape(B * T, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)   # (B*T, heads, S, head_dim)

        out = F.scaled_dot_product_attention(q, k, v)     # over S = HW only
        out = out.transpose(1, 2).reshape(B * T, S, C)
        out = self.proj(out)
        return out.reshape(B, T, H, W, C)
```

The only structural difference from full 3D is the reshape: `(B*T, HW, C)` instead of `(B, THW, C)`. Time has become a batch dimension, so the `$T$` frames are attended in parallel but never see each other. Cost per call: `$O(T \cdot (HW)^2 d)$` — the `$T$` is parallel batch, the `$(HW)^2$` is the real attention work.

Temporal attention is the mirror image: **fold space into the batch**, sequence axis is the `$T$` frames at a fixed position.

```python
class TemporalAttention(nn.Module):
    """Attention across frames at each spatial position (space folded into batch)."""

    def __init__(self, dim: int, num_heads: int = 16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H, W, C)
        B, T, H, W, C = x.shape
        S = H * W
        # move time to the sequence axis, fold space into batch
        x = x.permute(0, 2, 3, 1, 4).reshape(B * S, T, C)   # (B*HW, T, C)
        qkv = self.qkv(x).reshape(B * S, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)       # (B*HW, heads, T, head_dim)

        out = F.scaled_dot_product_attention(q, k, v)         # over T only
        out = out.transpose(1, 2).reshape(B * S, T, C)
        out = self.proj(out)
        out = out.reshape(B, H, W, T, C).permute(0, 3, 1, 2, 4)  # back to (B, T, H, W, C)
        return out
```

The `permute` before the reshape is the whole game: it makes `$T$` contiguous as the sequence axis while `$HW$` rides along in the batch. After attention we permute back. The sequence length here is just `$T = 30$`, so this attention is *tiny* — `$T^2 = 900$` scores per position — which is exactly why temporal attention is cheap.

Finally, the factorized block stitches spatial and temporal together with norms, residuals, and an MLP. This is the block that real models (SVD's temporal layers, AnimateDiff's motion module, many video DiTs) actually use.

```python
class FactorizedBlock(nn.Module):
    """Spatial attention, then temporal attention, then MLP -- all with residuals."""

    def __init__(self, dim: int, num_heads: int = 16, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm_s = nn.LayerNorm(dim)
        self.spatial = SpatialAttention(dim, num_heads)
        self.norm_t = nn.LayerNorm(dim)
        self.temporal = TemporalAttention(dim, num_heads)
        self.norm_m = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H, W, C); LayerNorm acts on the last (channel) dim throughout
        x = x + self.spatial(self.norm_s(x))     # mix within frames
        x = x + self.temporal(self.norm_t(x))    # mix across frames
        x = x + self.mlp(self.norm_m(x))         # per-token feedforward
        return x
```

Notice the ordering: spatial first, then temporal. The order matters less than the fact that *both* run every block — spatial establishes per-frame structure, temporal propagates it through time, and stacking these recovers most of what full 3D would have modeled jointly. Figure 5 shows the block, including the AdaLN timestep conditioning a real DiT block would also carry.

![Stacked layers of one factorized transformer block routing the latent through spatial attention, temporal attention, an MLP with residuals, and AdaLN timestep conditioning](/imgs/blogs/spatiotemporal-attention-patterns-5.png)

### A note on SDPA and FlashAttention

In all four blocks I called `F.scaled_dot_product_attention` rather than writing the `$QK^\top$` softmax by hand. This is not a style choice — it is the difference between fitting and OOMing on full 3D. SDPA in recent PyTorch dispatches to one of three backends: the FlashAttention-2 kernel (fastest, fused, `$O(N)$` memory), a memory-efficient kernel, or a math fallback (which *does* materialize the `$N\times N$` matrix). You want Flash. You can pin it:

```python
from torch.nn.attention import sdpa_kernel, SDPBackend

with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
    out = F.scaled_dot_product_attention(q, k, v)
```

For full 3D over 108k tokens, Flash keeps the *memory* tractable (no 46 GB score matrix), but the *time* is still nine minutes because the FLOPs are quadratic. For the factorized blocks, the sequences are short (`$HW = 3600$` and `$T = 30$`), so even the math fallback would not OOM — but Flash still helps. The lesson: FlashAttention solves the memory wall for any pattern; it does not solve the FLOP wall for full 3D. Only changing the *pattern* does that.

### Telling the model where each token is: 3D positional encoding

Attention is permutation-invariant — it has no built-in notion that token `$i$` is in frame 3 at the top-left and token `$j$` is in frame 17 at the bottom-right. The model only knows token positions because we encode them. For images this is a 2D positional encoding; for video it is **3D** (time, height, width), and getting it right is essential because the spatial-versus-temporal factorization *also* shows up here. With full 3D attention you need a single positional encoding that spans all three axes coherently. With factorized attention, the spatial pass needs only a 2D `$(h, w)$` encoding (frame index is irrelevant — every frame is attended independently) and the temporal pass needs only a 1D `$t$` encoding. The factorization of attention induces a factorization of position.

Modern video DiTs (CogVideoX, HunyuanVideo, Wan) use **3D rotary position embedding (RoPE)** — the same rotary scheme as LLMs, but the rotation frequencies are split across the time, height, and width axes so each token's `$(t, h, w)$` coordinate rotates its query and key vectors by a position-dependent angle. Here is the shape of it for a full-3D model:

```python
def rope_3d(q, k, coords, dim, base=10000.0):
    """Apply 3D rotary embedding. coords: (N, 3) integer (t, h, w) per token."""
    # split the head dim into three equal bands: one for t, one for h, one for w
    band = dim // 3
    inv_freq = 1.0 / (base ** (torch.arange(0, band, 2).float() / band))  # (band/2,)

    def axis_rotation(pos):                       # pos: (N,)
        ang = pos[:, None] * inv_freq[None, :]    # (N, band/2)
        return torch.cat([ang, ang], dim=-1)      # (N, band) -- cos/sin paired

    angles = torch.cat([axis_rotation(coords[:, a]) for a in range(3)], dim=-1)  # (N, dim)
    cos, sin = angles.cos(), angles.sin()

    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot
```

The key detail is that the head dimension is **partitioned across the three spatiotemporal axes** — a third of the rotary bands encode time, a third height, a third width. This is why a model trained at one resolution or clip length can sometimes extrapolate to another: RoPE encodes *relative* position, so a query and key that are 5 frames apart rotate by the same relative angle whether they sit at frames 0 and 5 or frames 100 and 105. That relative-position property is one reason full-3D RoPE models generalize across durations better than learned absolute position embeddings — a property that matters enormously when you want a model trained on 5-second clips to roll out to 30 seconds, the subject of the [long-video post](/blog/machine-learning/video-generation/efficient-and-real-time-video-generation). For a factorized model you would instead apply 2D RoPE inside the spatial pass and 1D RoPE inside the temporal pass — cheaper, but the two passes never share a joint positional frame, which is a subtle further reason factorized models model coupled spatiotemporal position less directly.

## 5. Windowed and sparse temporal attention: the long-clip regime

Factorized attention gets you a `$T\times$` saving, but it does not change the *spatial* attention's `$O((HW)^2)$` cost, which at 1080p (`$HW \approx 8100$`) is itself large, nor does it help when `$T$` grows into the hundreds for long video. The next lever is **locality**: do not let every token attend to every other even within the spatial or temporal pass — restrict attention to a **window**.

This is the direct video analogue of the **Swin Transformer**'s shifted windows, lifted into 3D. A **3D local window** of size `$w_t \times w_s \times w_s$` lets each query attend only to the `$w_t w_s^2$` tokens in its local spacetime neighborhood. Cost drops from `$O(N^2)$` to `$O(N \cdot w_t w_s^2)$` — **linear in `$N$`** for a fixed window. The price is that a token cannot, in one layer, attend to anything outside its window. To restore long-range connectivity, alternate layers **shift** the window grid by half a window, so tokens that were on a window boundary in layer `$\ell$` are in the same window in layer `$\ell+1$`. Information propagates across the whole volume in `$O(\text{diameter} / w)$` layers rather than one. Figure 6 shows the window layout.

![Grid of tokens across three time steps with the left two spatial columns grouped into one local window and the right column into a second window to show 3D local attention with shifted-window border sharing](/imgs/blogs/spatiotemporal-attention-patterns-6.png)

### Strided and dilated temporal attention

There is a complementary trick for the *temporal* axis specifically. When `$T$` is large — a 30-second clip might have `$T = 180$` latent frames — even temporal attention's `$O(T^2)$` becomes nontrivial, and more importantly the model rarely needs *dense* connectivity across 180 frames. **Strided** (or **dilated**) temporal attention has each query attend to frames at fixed strides — every 2nd, 4th, 8th frame — so it captures both nearby motion (small stride) and long-range persistence (large stride) at `$O(T \log T)$` or `$O(T \sqrt{T})$` cost. This is the same family of sparse-attention ideas from long-context language models, applied to the time axis. **Block-sparse** temporal attention (attend densely within a chunk of frames, sparsely across chunks) is how several long-video models keep the temporal pass affordable past a few seconds.

Put numbers on it. At `$T = 180$` latent frames, dense temporal attention costs `$T^2 = 32{,}400$` score pairs per spatial position. A block-sparse scheme with a dense window of 16 frames plus strided global frames every 8th frame attends to roughly `$16 + 180/8 \approx 38$` keys per query — `$T \cdot 38 = 6{,}840$` pairs, an `$\approx 4.7\times$` saving on the temporal pass at `$T = 180$`, growing with `$T$`. For a 5-second clip (`$T = 30$`) this is pointless — dense temporal is already cheap — which is exactly why you only see strided/block-sparse temporal attention in the long-video and autoregressive-rollout models, never in short-clip ones. The cost of the cleverness is connectivity: a query can no longer see *every* past frame directly, so identity that must persist across the full clip (a character who leaves and re-enters after 100 frames) relies on the strided global frames being chosen well. Choose the stride pattern badly and you get the long-video failure mode of identity drift, which the [efficient and real-time generation post](/blog/machine-learning/video-generation/efficient-and-real-time-video-generation) treats as its own problem.

Here is a windowed spatial attention in PyTorch — partition `$HW$` into non-overlapping `$w_s \times w_s$` windows, attend within each, then (in alternating layers) shift:

```python
def window_partition(x, ws):
    # x: (B*T, H, W, C) -> (num_windows*B*T, ws*ws, C)
    BT, H, W, C = x.shape
    x = x.reshape(BT, H // ws, ws, W // ws, ws, C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, C)
    return x


class WindowedSpatialAttention(nn.Module):
    def __init__(self, dim, num_heads=16, window=8, shift=0):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.window, self.shift = window, shift
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):                      # x: (B, T, H, W, C)
        B, T, H, W, C = x.shape
        x = x.reshape(B * T, H, W, C)
        if self.shift:                         # shifted windows in alternating layers
            x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(1, 2))
        win = window_partition(x, self.window)        # (nW*B*T, ws*ws, C)
        M = win.shape[0]
        qkv = self.qkv(win).reshape(M, self.window ** 2, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        out = F.scaled_dot_product_attention(q, k, v)  # only ws*ws = 64 tokens per window
        out = out.transpose(1, 2).reshape(M, self.window ** 2, C)
        out = self.proj(out)
        # (window-reverse and unroll omitted for brevity)
        return out
```

With `$w_s = 8$`, each window has only `$64$` tokens, so the per-window attention is trivial and the total spatial cost is `$O(\frac{HW}{64} \cdot 64^2 \cdot d) = O(HW \cdot 64 \cdot d)$` — linear in `$HW$`, a `$\frac{HW}{64}$` saving over dense spatial. The `torch.roll` is the shifted-window trick in one line. The catch is the bookkeeping (reversing the partition, masking the rolled border tokens) which I have elided; production implementations like the ones in HunyuanVideo's and Wan's long-clip variants handle it carefully.

## 6. Where the FLOPs actually go: putting the patterns side by side

Now I can answer the title question. Figure 4 lays the three patterns against the four numbers that matter, and the rest of this section unpacks it.

![Matrix comparing full 3D, spatial plus temporal, and windowed 3D attention across attention FLOPs, peak memory, coherence, and which models use each pattern](/imgs/blogs/spatiotemporal-attention-patterns-4.png)

Here is the same comparison as a table you can act on, with the numbers worked for our 5-second 720p clip (`$N = 108{,}000$`, `$HW = 3600$`, `$T = 30$`, `$d = 1152$`, one attention layer/block, `$w_s = 8$`, `$w_t = 4$`):

| Pattern | Attn FLOPs / layer | Peak score memory (bf16, naive) | Coherence | Used by |
| --- | --- | --- | --- | --- |
| Full 3D (joint) | `$4N^2 d \approx 5.4\times10^{13}$` | `$N^2 \approx 23$` GB | Exact, all token pairs | Small/short-clip DiTs; Sora-class with Flash |
| Spatial + temporal | `$4Nd(HW{+}T) \approx 1.8\times10^{12}$` | `$\max(HW^2,T^2) \approx 0.05$` GB | Indirect cross-frame, separable | SVD, AnimateDiff, many video DiTs |
| Windowed 3D | `$4Nd\,w_tw_s^2 \approx 1.2\times10^{12}$` | window scores, `$\ll 0.05$` GB | Local + shifted, weak long-range | Swin-style, long-clip variants |

Two things jump out. First, **the FLOP gap between full 3D and the factorized/windowed patterns is roughly `$30\times$` here and grows to `$\approx T \times$` at high resolution** — exactly the `$\frac{T \cdot HW}{HW+T}$` ratio we derived. Second, **the memory gap is far more dramatic than the FLOP gap**: full 3D's naive score matrix is 23 GB while factorized is 50 MB, a `$\sim 460\times$` difference. That memory gap is *why* factorization is not optional for long or high-res clips even when you have FLOPs to burn — the score matrix simply will not fit. FlashAttention shrinks full 3D's *resident* memory back to `$O(N)$`, closing the memory gap, but it cannot touch the FLOP gap. So the honest summary is:

- If you can afford the FLOPs (short clip, big GPU, Flash for memory), **full 3D gives the best coherence**.
- The moment FLOPs or memory bind — long clip, high resolution, small GPU — **factorization buys you `$\sim T\times$` headroom at a real but usually acceptable coherence cost**.
- For *very* long clips, **windowed/sparse on top of factorization** pushes cost to linear in `$N$`, at the cost of long-range coherence that you then patch with shifted windows or strided attention.

### Where the FLOPs go *within* a factorized block

A subtlety worth internalizing: once you factorize, **the spatial attention dominates the FLOPs, not the temporal**. Spatial is `$4 T (HW)^2 d$`; temporal is `$4 (HW) T^2 d$`. Their ratio is `$\frac{(HW)^2 T}{HW \cdot T^2} = \frac{HW}{T} = \frac{3600}{30} = 120$`. So **spatial attention is 120× more expensive than temporal attention** in our factorized block. Temporal attention — the part that makes video *video*, the part everyone worries about for coherence — is almost free. The expensive part is the per-frame spatial attention, which is essentially the same cost as image generation. This is deeply reassuring for cost modeling: a factorized video model costs *about `$T$` images' worth of spatial attention plus a rounding-error of temporal attention*, which is exactly why the field could bolt temporal modules onto frozen image backbones (AnimateDiff) and get video almost for free. It is also why windowing the *spatial* pass (Section 5) matters more for cost than windowing the temporal pass.

#### Worked example: the VRAM budget for a 5-second clip on an RTX 4090

An RTX 4090 has 24 GB. Can it render our clip? With **full 3D and naive attention**, no — the score matrix alone is 23 GB (bf16) per head per layer, and you have 16 heads. Dead on arrival. With **full 3D and FlashAttention**, the score matrix never materializes; resident attention memory is `$O(N \cdot d) = 108{,}000 \cdot 1152 \cdot 2 \approx 0.25$` GB for activations, so it *fits memory-wise*, but each denoising step takes ~13 seconds of attention alone (54 TFLOPs / ~4 TFLOP/s effective on a 4090), so 50 steps × 2 CFG ≈ 22 minutes per clip. Painful but possible. With **factorization**, attention drops to ~1.8 TFLOPs/block, the whole denoiser runs in seconds per step, and a 5-second clip renders in 1–2 minutes on the same 4090. The pattern choice turned a 22-minute render into a 90-second one on identical hardware. This is not a micro-optimization; it is the difference between a usable local model and a cloud-only one.

### Measuring it yourself: a FLOP-and-memory micro-benchmark

The numbers above are derived; the honest move is to measure them on your own hardware. Here is a small harness that times a full-3D and a factorized block on the *same* latent and reports wall-clock and peak VRAM, with a warm-up pass excluded (the first call pays kernel-compilation and allocator-warmup costs you must not count). This is the cost half of the comparison protocol I will lay out in Section 10.

```python
import torch, time

def bench(block, x, iters=20, warmup=5):
    block, x = block.cuda().to(torch.bfloat16), x.cuda().to(torch.bfloat16)
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):                 # warm up: compile kernels, warm allocator
        block(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        block(x)
    torch.cuda.synchronize()               # CUDA is async -- sync before stopping clock
    dt = (time.perf_counter() - t0) / iters
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    return dt * 1e3, peak_gb               # ms per call, peak GB

B, T, H, W, C = 1, 30, 45, 80, 1152
x = torch.randn(B, T, H, W, C)
for name, blk in [("full3d", Full3DAttention(C)), ("factorized", FactorizedBlock(C))]:
    ms, gb = bench(blk, x)
    print(f"{name:12s}  {ms:8.1f} ms/call   {gb:6.2f} GB peak")
```

Three things make this honest. First, the **`torch.cuda.synchronize()`** before stopping the timer — CUDA kernels launch asynchronously, and without the sync you time the launch, not the execution, which makes everything look impossibly fast. Second, the **warm-up loop** — the first few calls include `torch.compile`/cuDNN autotuning and allocator growth that you should not attribute to steady-state cost. Third, **`reset_peak_memory_stats` + `max_memory_allocated`** — peak, not current, memory is what determines whether you OOM, and the peak happens transiently inside the attention. Run this and you will see the factorized block clock in at a fraction of the full-3D time with a fraction of the peak memory, exactly tracking the `$\frac{T \cdot HW}{HW + T}$` and `$N^2$`-versus-`$\max(HW^2, T^2)$` ratios — and you will see whether your particular GPU's SDPA picked the Flash backend (if full-3D peak memory is `$O(N^2)$`-sized, it fell back to the math kernel and you should pin Flash).

## 7. The memory wall in detail: KV-cache and activation pressure

I keep saying memory is the harder wall. Let me make that precise, because it is where most people's intuition from language models misleads them. Figure 7 contrasts the resident score memory of the two regimes.

![Before and after comparison of attention memory showing full 3D storing an N by N score block of tens of gigabytes versus factorized attention keeping spatial and temporal score blocks under a gigabyte](/imgs/blogs/spatiotemporal-attention-patterns-7.png)

In autoregressive LLM decoding, the memory concern is the **KV-cache**: you cache past keys and values so you do not recompute them each step, and the cache grows linearly with sequence length. Video *diffusion* is different — it is not autoregressive over tokens; it denoises the *whole* `$N$`-token latent in parallel at each step. So there is no growing KV-cache across positions. But there is something analogous and arguably worse: at each layer, attention's **intermediate activations** must be held for the backward pass during training, and the score matrix `$N \times N$` is the largest single tensor in the model. For full 3D over our clip that is the 23–46 GB monster. Even with FlashAttention eliminating the materialized score matrix, training a full-3D video DiT requires gradient checkpointing on the attention blocks to avoid storing activations, which trades ~30% more compute for the memory headroom.

There is a second memory consumer that bites video specifically: **the VAE decode**. The denoiser produces a latent of `$N$` tokens, but the VAE must decode that back to `$T \times H \times W$` pixels — 120 frames at `$720\times1280\times3$` — and the decoder's intermediate activations at full pixel resolution can exceed the denoiser's peak. This is why `diffusers` video pipelines expose `enable_vae_slicing()`, `enable_vae_tiling()`, and `decode_chunk_size`: they decode the clip in temporal chunks so the VAE never holds all 120 frames of activations at once. So the memory hierarchy in a video model is roughly: **attention score matrix (if full 3D and naive) > VAE decode activations > everything else**. Attention is the second wall after the VAE — and if you factorize attention away, the VAE decode often becomes the *new* peak. Here is the real-world incantation:

```python
import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    "THUDM/CogVideoX-5b-I2V", torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()   # stream weights layer-by-layer to keep VRAM low
pipe.vae.enable_slicing()         # decode latent frames one batch element at a time
pipe.vae.enable_tiling()          # decode each frame in spatial tiles -> caps VAE peak

image = load_image("first_frame.png")
video = pipe(
    image=image,
    prompt="a golden retriever running across a sunlit field, cinematic",
    num_frames=49,                # latent frames after temporal compression
    num_inference_steps=50,
    guidance_scale=6.0,
).frames[0]

export_to_video(video, "out.mp4", fps=8)
```

Every line after `from_pretrained` is a memory lever. `enable_model_cpu_offload` keeps only the active module on GPU. `vae.enable_slicing` and `vae.enable_tiling` chunk the VAE decode so the *second* memory wall does not topple you after you have carefully managed the first. On a 24 GB card, those three calls are the difference between CogVideoX-5B running and OOMing.

## 8. Where each pattern is actually used (the 2024–2026 frontier)

Theory is clean; shipped models are pragmatic. Here is what the real models do, which is rarely "pick one pattern" and usually "mix them." Figure 8 traces the evolution.

![Timeline of video model attention patterns from VDM's factorized 3D U-Net through SVD and AnimateDiff temporal modules to Sora, CogVideoX, and Wan full and block spacetime attention](/imgs/blogs/spatiotemporal-attention-patterns-8.png)

**Stable Video Diffusion (SVD)** and **AnimateDiff** are the canonical *factorized* models. Both take a frozen image backbone and add **temporal attention modules** between the existing spatial layers — AnimateDiff's "motion module" is literally a temporal attention block inserted into a frozen Stable Diffusion UNet, trained on video while the spatial weights stay fixed. This is factorization in its purest, most modular form: spatial attention is inherited from the image model, temporal attention is the only new thing, and the `$\sim T\times$` saving is what makes training a motion module on a modest GPU budget feasible. The coherence cost shows up exactly as predicted — both can wobble under large motion.

**CogVideoX** (THUDM, 2024) is one of the first widely-used open models to use **full 3D attention** in a DiT, made affordable by an aggressive `$4\times8\times8$` causal 3D-VAE that keeps `$N$` small and by FlashAttention for memory. Its design choice — pay for full attention, but compress hard first — is a direct illustration of "the VAE is the first wall, attention is the second": shrink `$N$` at the VAE so you can afford the `$O(N^2)$` at attention. **HunyuanVideo (1.5)** and **Wan 2.x** similarly use full or block-wise spacetime attention in their DiTs, again leaning on heavy 3D-VAE compression and, for long clips, block-sparse/windowed temporal attention to stay tractable past a few seconds.

**Sora** (OpenAI, 2024), per its technical report, operates on **spacetime patches** with a transformer that scales attention across the whole spatiotemporal volume — full spacetime attention at scale, which the report frames as central to the model's emergent 3D consistency and object permanence. The thesis there is the inverse of factorization: *do not* factorize, because the long-range cross-frame interactions you would give up are exactly what produce world-simulator coherence — and instead pay for full attention by spending enormous compute. That is the bet only a very large compute budget can make, and it is why open models, which cannot, lean factorized or block-wise.

The pattern across the frontier: **factorized when compute is scarce or the backbone is frozen (SVD, AnimateDiff); full 3D when you can afford it and coherence is paramount and you have compressed `$N$` hard at the VAE (CogVideoX, HunyuanVideo, Wan, Sora-class); windowed/sparse bolted on for the long-clip tail.** There is no universal winner — there is a compute budget and a coherence bar, and the pattern is whatever clears the bar within the budget.

Here is the landscape as a table you can read a model card against. The "self-attention pattern" column is the one this post is about; note how it correlates with whether the backbone was trained from scratch (affords full 3D) or inflated from a frozen image model (forced toward factorized):

| Model | Params | Self-attention pattern | Backbone origin | Typical clip | License |
| --- | --- | --- | --- | --- | --- |
| AnimateDiff | ~1.5B (frozen SD) + motion module | Factorized: frozen spatial + new temporal | Inflated from frozen SD 1.5 | 16–32 frames | Apache-2.0 |
| Stable Video Diffusion | ~1.5B | Factorized: spatial + temporal layers | Inflated latent backbone | 14–25 frames | Stability community |
| CogVideoX-5B | ~5B | Full 3D (joint spacetime) | Trained as video DiT | 49 frames · 6 s | Apache-2.0 |
| HunyuanVideo | ~13B | Full / block-wise spacetime | Trained as video DiT | up to ~5 s base | Tencent community |
| Wan 2.x | 1.3B / 14B | Block-wise spacetime | Trained as video DiT | ~5 s, longer variants | Apache-2.0 |
| Sora-class | not disclosed | Full spacetime patches at scale | Trained from scratch | up to ~60 s | Proprietary |

The correlation is not an accident: inflating a frozen image model *forces* factorization (you cannot retrain the spatial weights jointly with new temporal ones without unfreezing them), while training a video DiT from scratch *permits* full 3D if you have compressed `$N$` enough at the VAE to afford it. The attention pattern is downstream of the training strategy, which is downstream of the compute budget.

### How text conditioning rides on top of the pattern

One more interaction worth surfacing, because it trips people up: the attention pattern discussion so far is about **self-attention** among video tokens, but a text-to-video model also has **cross-attention** to the text-prompt tokens (or, in newer MM-DiT designs, joint attention over concatenated video and text tokens). Cross-attention from `$N$` video queries to `$L$` text keys costs `$O(N L d)$` — linear in `$N$`, because `$L$` (a few hundred text tokens) is small and fixed. So cross-attention to text is *cheap* relative to video self-attention, and it does not change the full-versus-factorized calculus much. But in a **factorized** model you have a design choice: do you cross-attend to text in the spatial pass, the temporal pass, or both? Most factorized models inject text in the spatial pass (text describes per-frame content) and let temporal attention propagate it, which keeps the temporal pass purely about motion. In an **MM-DiT** full-3D model (the Stable Diffusion 3 / Flux lineage, lifted to video), text and video tokens are concatenated into one sequence and attended jointly, so text conditioning shares the `$O(N^2)$` cost — another reason MM-DiT video models lean hard on VAE compression to keep `$N$` small. The [conditioning post](/blog/machine-learning/video-generation/from-image-diffusion-to-video-diffusion) goes deeper on control injection; the point here is that the self-attention pattern you pick also shapes *where* and *how cheaply* you can inject conditioning.

### Case study numbers

A few concrete, citable data points (orders of magnitude where I am not certain of the exact figure — I flag those):

- **CogVideoX-5B** generates 6-second 720p clips (49 latent frames, `$4\times8\times8$` VAE) with full 3D attention; peak inference VRAM is roughly 18–24 GB with model CPU offload + VAE tiling, and a clip takes on the order of a few minutes on an A100. The full-3D choice is feasible *because* the VAE drives `$N$` down to ~17–18k patchified tokens for that resolution — small enough that `$O(N^2)$` is payable.
- **Stable Video Diffusion** (Blattmann et al., 2023) is image-to-video with factorized temporal attention over a latent backbone; it generates 14–25 frames and uses `decode_chunk_size` precisely to manage the VAE-decode memory wall after the cheap factorized denoiser. The factorized design is what let it run on consumer-class GPUs at launch.
- **AnimateDiff** (Guo et al., 2023) adds a temporal attention motion module to a frozen T2I model; because only the temporal module trains and the spatial cost is inherited, the *added* compute is the cheap `$O(HW \cdot T^2 d)$` temporal term — empirically a small fraction of the spatial cost, consistent with our `$\frac{HW}{T}$` ≈ 120× spatial-dominance derivation.
- **Sora** (Brooks et al., technical report, 2024) reports that scaling spacetime-patch transformer compute improves sample quality and consistency — the scaling-thesis evidence that, for a sufficiently large budget, full spacetime attention's coherence is worth its quadratic cost. Exact FLOP/VRAM figures are not public; treat any specific number as an estimate.

### The older lineage: (2+1)D and the 3D U-Net

Before video DiTs, the factorization story played out in convolutional U-Nets, and it is worth knowing because the same separability logic applies and the vocabulary still shows up in model cards. A native **3D convolution** over a `$T \times H \times W$` clip with a `$k_t \times k_s \times k_s$` kernel costs `$O(k_t k_s^2)$` multiply-adds per output element — the convolutional analogue of full 3D attention's joint mixing. The **(2+1)D** factorization (Tran et al., 2018, "A Closer Look at Spatiotemporal Convolutions") replaces each 3D conv with a 2D spatial conv (`$1 \times k_s \times k_s$`) followed by a 1D temporal conv (`$k_t \times 1 \times 1$`), cutting cost from `$k_t k_s^2$` to `$k_s^2 + k_t$` per element — *exactly* the same `$\frac{k_t k_s^2}{k_s^2 + k_t}$` shape as the attention factorization ratio `$\frac{T \cdot HW}{HW + T}$`. It is the same trade, one level down: separate the spatial and temporal operators, pay less compute, lose the ability to model the genuinely coupled spatiotemporal kernel directly, recover it through depth and through the extra non-linearity that (2+1)D's intermediate activation provides.

Video Diffusion Models (Ho et al., 2022) and Make-A-Video used a **3D U-Net** that was really a (2+1)D U-Net: spatial conv-and-attention layers inherited or initialized from an image model, with factorized temporal attention and temporal convolutions interleaved. This is the architecture the [from-image-to-video post](/blog/machine-learning/video-generation/from-image-diffusion-to-video-diffusion) calls "inflation" — you take a pretrained 2D backbone and inflate it to 3D by inserting temporal layers, and because temporal attention is the cheap `$O(HW \cdot T^2 d)$` term, the inflation is affordable. SVD and AnimateDiff are the latent-space descendants of this U-Net line; the DiT-video models (CogVideoX, Sora, Wan) are the transformer descendants. The attention-pattern question — full versus factorized — is the same question the conv world answered as 3D-versus-(2+1)D, and the answer rhymes: factorize when compute binds, keep it joint when coherence under large motion is paramount and you can pay.

### Debugging a coherence failure: a war story

Here is how this plays out in practice, because the theory only becomes useful when you can diagnose a bad clip. I once shipped a factorized 1.5B video model that scored beautifully on VBench's subject-consistency and background-consistency dimensions — near the top of the open leaderboard — and yet users complained that anything with fast motion "smeared." The metrics were not lying; they were measuring the wrong thing. The model was excellent at the separable interactions (a static background, a slowly-translating subject) and the consistency metrics rewarded exactly that. But on a clip of a tennis serve — the racket sweeping across most of the frame between two adjacent latent frames — the same-position temporal attention had nothing useful to attend to, because the racket's content at position `$p$` in frame `$t$` was at a completely different position in frame `$t+1$`. The model fell back on blurring across the ambiguity.

The fix was instructive and confirmed the low-rank argument. Pure factorization could not be rescued by more depth alone within our budget. What worked was **adding a small windowed temporal-spatial coupling** — a 3D local window attention (Section 5) every fourth block, with a window large enough in space to span a few latent pixels of motion, so the racket's displaced content fell inside the same window across adjacent frames. That restored the non-separable correspondence locally, where the motion actually was, without paying full-3D cost globally. Dynamic-degree-weighted quality jumped; subject-consistency held; wall-clock rose by maybe 20%. The lesson I carry from it: **factorized-plus-local-windows is often the right answer, not pure factorized and not full 3D** — you spend the expensive coupled attention only in the local neighborhoods where motion actually couples space and time, and you keep the cheap factorized passes everywhere else. The frontier models that handle large motion well almost all do some version of this hybrid.

## 9. When to reach for each pattern (and when not to)

Let me be decisive, because this is the section you came for.

**Reach for full 3D attention when:** the clip is short (a few seconds), you have aggressively compressed `$N$` at the VAE so that `$N \lesssim 20{,}000$` tokens, you have FlashAttention to keep memory in check, coherence is the top priority (faces, complex object motion, anything where cross-frame drift is unacceptable), and you have the compute. CogVideoX-class quality on short clips is the sweet spot. **Do not** use full 3D when `$N$` is large — at `$N = 108{,}000$` you are paying `$30\times$` to `$120\times$` more FLOPs than factorization for a coherence improvement that, past a sufficiently deep factorized stack, is often marginal. I have wasted GPU-weeks learning that you rarely need full 3D's exact all-pairs mixing if your factorized model is deep enough.

**Reach for factorized (spatial + temporal) attention when:** the clip is longer or higher resolution, you are inflating a frozen image backbone (AnimateDiff-style), compute or memory binds, or you want the modularity of a swappable temporal module. This is the default for most open and self-hosted setups, and it is correct far more often than people assume. **Do not** factorize away temporal attention entirely — some models tried per-frame-only attention plus a light temporal conv, and they flicker; the temporal *attention* term is so cheap (`$120\times$` less than spatial) that there is almost never a good reason to drop it.

**Reach for windowed/sparse attention when:** the clip is long (tens of seconds, `$T$` in the hundreds) and even factorized temporal attention's `$O(T^2)$` or factorized spatial's `$O((HW)^2)$` is too much. Shifted 3D windows or strided temporal attention bring cost to linear in `$N$`. **Do not** window when the clip is short — the shifted-window machinery adds complexity and a long-range-coherence penalty you do not need to pay for a 5-second clip.

There is also a strong middle-ground default worth stating plainly: **factorized-plus-periodic-local-windows**. As the war story in Section 8 showed, pure factorization breaks under large motion and pure full-3D is unaffordable, but a factorized stack with a 3D local-window attention every few blocks recovers the coupled spatiotemporal interactions exactly where motion needs them — locally — at a fraction of full-3D cost. If I were starting a new video model today and did not have a Sora-scale compute budget, this is where I would start: factorized everywhere for cost, full-3D-style local windows sprinkled in for the motion-coupled coherence, and strided temporal attention only if I needed clips longer than the VAE's trained length. It clears the coherence bar for the overwhelming majority of prompts at a tiny fraction of full-3D's FLOPs.

And the meta-rule above all of these: **compress at the VAE first.** Every token you do not create at the VAE is a token you never pay `$O(N^2)$` (or even `$O(N)$`) for at attention. The cheapest attention is attention over fewer tokens, and the VAE is where token count is set. Attention pattern choice is the *second* lever; VAE compression is the *first*. If your full-3D model OOMs, the first question is not "should I factorize" but "can I compress `$N$` harder at the VAE without unacceptable reconstruction loss" — the [video-autoencoder post](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression) covers that lever in full. Only after you have squeezed the VAE do you reach for the attention-pattern lever, and only after that for sampler-step reduction and feature caching.

## 10. Honest measurement: how to compare patterns without fooling yourself

If you are going to choose a pattern based on coherence-versus-cost, you need to measure both honestly, and both are easy to game.

For **cost**, report **peak VRAM and seconds-per-clip on a named GPU**, at a fixed resolution, frame count, step count, and CFG scale, with a warm-up pass excluded (the first call includes kernel compilation and allocator warmup that you should not count). Report attention FLOPs separately from total FLOPs so the pattern's contribution is visible — the whole point is that attention is the part that changes with the pattern while MLP and VAE do not. And report the *naive* and *FlashAttention* memory separately for full 3D, because the gap between them is the entire reason full 3D is even on the table.

For **coherence**, use [FVD and VBench](/blog/machine-learning/video-generation/the-metrics-of-video-generation), but know their failure modes. FVD (Fréchet distance on I3D video features) is noisy and sensitive to the reference set, frame count, and preprocessing — never compare FVD numbers across papers with different protocols. VBench decomposes quality into subject consistency, background consistency, motion smoothness, dynamic degree, and aesthetic/imaging quality, which is exactly the decomposition you want for attention patterns: factorized models tend to score *well* on subject/background consistency (slow, separable motion is their strength) and *worse* on dynamic degree under large motion (their known weakness). But beware the **dynamic-degree-versus-stability gaming problem**: a model that barely moves trivially maxes consistency and motion-smoothness while failing dynamic degree, and a model that moves wildly does the opposite. A factorized model that "wins" on consistency may simply be moving less. Always look at consistency *and* dynamic degree together, on the *same* prompts with a *fixed seed*, and sanity-check with human eval on a few clips that exercise large motion — the regime where the full-3D-versus-factorized difference is real. The fair comparison is: same VAE, same `$N$`, same step count and sampler, swap only the attention pattern, evaluate on a motion-stratified prompt set.

#### Worked example: a fair full-3D-versus-factorized A/B

Suppose you train two otherwise-identical 2B video DiTs on the same `$4\times8\times8$` VAE latents — one full 3D, one factorized — and evaluate on a 200-clip VBench-style set stratified into low-motion and high-motion prompts, A100 80GB, 50 steps, CFG 6, fixed seeds. A plausible, honest outcome: full 3D costs ~`$5\times$` the attention FLOPs and ~`$2.5\times$` the wall-clock per clip, scores roughly on par on subject/background consistency for low-motion prompts (both near the ceiling), and pulls ahead by a few VBench points on dynamic-degree-weighted quality for high-motion prompts (the spinning wheel, the fast pan), where factorized's same-position temporal assumption frays. The decision then is explicit: are those few points on high-motion clips worth `$2.5\times$` the serving cost? For a cinematic product where motion is the point, maybe. For a 5-second talking-head product where motion is small, almost never — factorized wins on cost with no perceptible coherence loss. That is the trade-off this whole post equips you to make, with numbers instead of vibes.

## 11. Tying it back to the stack

Step back to the series frame: **video = (spatial generation) × (temporal coherence) under a brutal compute budget.** Attention patterns are precisely where the "× temporal coherence" and the "brutal compute budget" terms collide. Full 3D maximizes coherence and maximizes cost; factorization separates the spatial and temporal factors literally — spatial attention *is* the spatial-generation term, temporal attention *is* the temporal-coherence term — and pays for the separation with `$\sim T\times$` less compute and some loss of non-separable, motion-coupled interactions. Windowing trades long-range coherence for linear cost when the clip gets long.

In the stack — **data → causal 3D-VAE → spacetime-patch DiT → flow-matching sampler → conditioning → frames** — attention lives inside the spacetime-patch DiT, and its pattern is set by two upstream facts: how hard the **3D-VAE** compressed (which sets `$N$`, and thus whether full 3D is even affordable), and how long a clip the **conditioning and sampler** need to produce (which decides whether you must window for length). So the attention pattern is not chosen in isolation — it is downstream of the VAE's compression and upstream of the sampler's clip-length demands. Get the VAE compression right and full 3D may be affordable; need long clips and you are forced toward windowed factorization no matter how good the VAE is.

This is also why the [next post on video diffusion transformers](/blog/machine-learning/video-generation/video-diffusion-transformers) can take the attention pattern as a solved sub-problem and focus on the spacetime-patch recipe, variable resolution and duration, and the scaling thesis — the attention cost is the foundation that recipe is built on. And it is why the [efficient and real-time generation post](/blog/machine-learning/video-generation/efficient-and-real-time-video-generation) returns to attention as the first thing to optimize when you need latency: step distillation and feature caching help, but if your attention pattern is full 3D over 100k tokens, no amount of caching saves you — you change the pattern first. When you assemble a real pipeline in the [capstone playbook](/blog/machine-learning/video-generation/building-with-video-generation-the-playbook), the attention pattern of your chosen base model is one of the first things to read off its config, because it bounds everything downstream: clip length, resolution, VRAM, and latency.

## Key takeaways

- **Attention is the second wall after the VAE.** The VAE sets your token count `$N = (T/c_t)(H/c_s)(W/c_s)$`; attention is the only operation that scales `$O(N^2)$` in it, so it dominates cost and decides whether the model fits at all.
- **Full 3D attention costs `$4N^2 d$` and gives exact all-pairs coherence.** On a 5-second 720p clip (`$N \approx 108{,}000$`) that is ~54 TFLOPs/layer and a 23–46 GB naive score matrix — the coherence gold standard and the compute catastrophe.
- **Factorization costs `$4Nd(HW+T)$` — a saving of `$\frac{T \cdot HW}{HW+T} \approx T$` at high resolution.** Roughly `$30\times$` on our clip, growing to `$\sim 100\times$` for longer clips. It trades exact, motion-coupled cross-frame interactions for separable ones recovered through depth.
- **Within a factorized block, spatial attention dominates `$120\times$` over temporal.** Temporal attention — the part that makes video video — is nearly free (`$O(HW \cdot T^2 d)$`), which is why temporal modules bolt cheaply onto frozen image backbones (AnimateDiff). Never drop it.
- **FlashAttention fixes the memory wall, not the FLOP wall.** It makes full 3D's memory `$O(N)$`, but the compute is still `$O(N^2)$`. Only changing the *pattern* changes the FLOPs.
- **Windowed/strided attention buys linear cost for long clips**, at the price of long-range coherence patched by shifted windows or temporal strides. Use it only when the clip is genuinely long.
- **Shipped models mix patterns to fit budget and bar:** SVD/AnimateDiff factorize; CogVideoX/HunyuanVideo/Wan/Sora-class pay for full or block-wise spacetime attention behind heavy VAE compression. There is no universal winner — only a compute budget and a coherence bar.
- **Measure both sides honestly:** seconds-per-clip and peak VRAM on a named GPU with warm-up excluded; FVD/VBench on a motion-stratified, fixed-seed prompt set, reading consistency *and* dynamic degree together to avoid the move-less gaming trap.

## Further reading

- Ho et al., **Video Diffusion Models** (2022) — the 3D U-Net with factorized space-time attention that started the diffusion-video line.
- Blattmann et al., **Stable Video Diffusion: Scaling Latent Video Diffusion Models** (2023) — factorized temporal attention over a latent backbone, with the curation and conditioning recipe.
- Guo et al., **AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models** (2023) — the motion module as a plug-in temporal attention block over a frozen T2I backbone.
- Brooks et al., **Video Generation Models as World Simulators** (Sora technical report, OpenAI, 2024) — spacetime patches and full spacetime attention at scale; the scaling-coherence thesis.
- Peebles & Xie, **Scalable Diffusion Models with Transformers** (DiT, 2023) — the transformer-diffusion backbone the video DiTs inflate; see also the image-series [DiT deep-dive](/blog/machine-learning/image-generation/diffusion-transformers-dit).
- Liu et al., **Swin Transformer** (2021) and **Video Swin Transformer** (2022) — shifted 3D windows, the basis for windowed video attention.
- Dao et al., **FlashAttention** (2022) and **FlashAttention-2** (2023) — the `$O(N)$`-memory exact-attention kernel that keeps full 3D's memory tractable.
- 🤗 `diffusers` video pipeline docs (`CogVideoXPipeline`, `StableVideoDiffusionPipeline`, `AnimateDiffPipeline`) — the real APIs, `num_frames`, `decode_chunk_size`, VAE slicing/tiling.
- Within this series: [why video generation is hard](/blog/machine-learning/video-generation/why-video-generation-is-hard) (foundations), [from image diffusion to video diffusion](/blog/machine-learning/video-generation/from-image-diffusion-to-video-diffusion), [video autoencoders and spatiotemporal compression](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression), and the capstone [building with video generation: the playbook](/blog/machine-learning/video-generation/building-with-video-generation-the-playbook).
