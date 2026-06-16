---
title: "The Diffusion U-Net: Anatomy of the Network That Denoises"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Take the diffusion denoiser apart bolt by bolt: why a multi-scale U-Net matches the structure of images, how residual blocks and AdaGN inject the timestep, why self-attention is gated to 16x16, how cross-attention lets text steer pixels, and where Stable Diffusion's 860M parameters actually sit."
tags:
  [
    "image-generation",
    "diffusion-models",
    "u-net",
    "unet",
    "cross-attention",
    "stable-diffusion",
    "generative-ai",
    "deep-learning",
    "pytorch",
    "diffusers",
  ]
category: "machine-learning"
subcategory: "Image Generation"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/the-diffusion-unet-1.png"
---

Every diffusion post you have read so far has treated the denoiser as a black box. The forward process noises an image; we train a network $\epsilon_\theta(x_t, t)$ to predict the noise; we run it a few dozen times to walk from static back to a photograph. Fine. But "a network" is doing an enormous amount of quiet work in that sentence. The network has to look at a $64 \times 64 \times 4$ latent that is mostly random, decide which parts are signal and which are noise *at this particular noise level*, respect the global composition (one cat, two ears, the right number of legs) while also getting the local texture right (fur, whiskers, the glint in an eye), and do all of that while a text prompt is trying to steer it toward "a tabby cat wearing a tiny wizard hat." If you hand that job to a plain stack of convolutions, you get blurry mush. If you hand it to a plain transformer at full resolution, you run out of memory before the first epoch. The architecture that actually works — the one that powered diffusion from Ho et al.'s 2020 DDPM through Stable Diffusion 1.5 and SDXL — is a very specific, very battle-tested beast: the **time-conditioned U-Net with multi-resolution attention**.

This post takes that beast apart. By the end you will be able to read a `UNet2DConditionModel` config and know exactly what every field buys you, implement a residual block with timestep scale-shift and a cross-attention block from scratch in PyTorch, explain on a whiteboard *why* self-attention is affordable at $16 \times 16$ but ruinous at full resolution, count where Stable Diffusion 1.5's roughly 860M parameters sit, and articulate the precise limitation — the baked-in convolution prior and the fixed resolution ladder — that motivated the field to throw the whole thing out and replace it with a transformer (the [diffusion transformer, or DiT](/blog/machine-learning/image-generation/diffusion-transformers-dit), which gets its own post).

![A directed graph of the U-Net denoiser showing an encoder that compresses to an attention-rich bottleneck and a decoder linked back by skip connections](/imgs/blogs/the-diffusion-unet-1.png)

This is a *Track C* post — the architecture track. The previous track built the diffusion *engine*: the forward and reverse processes in [diffusion from first principles](/blog/machine-learning/image-generation/diffusion-from-first-principles), the full variational derivation in [the math of DDPM](/blog/machine-learning/image-generation/the-math-of-ddpm), and the conditioning machinery in [classifier-free guidance](/blog/machine-learning/image-generation/classifier-free-guidance). All of those treat $\epsilon_\theta$ as given. Here we open it up. We will keep tying back to the spine of the series — the **diffusion stack** (data → VAE latent → forward noising → *denoiser net* → sampler → guidance → image) — because the U-Net is the "denoiser net" box, the single most parameter-heavy and compute-heavy component in the whole stack, and the place where the next two posts ([latent diffusion and Stable Diffusion](/blog/machine-learning/image-generation/latent-diffusion-and-stable-diffusion) and DiT) make their biggest moves.

Let me set the running example up front and keep it the whole way through: **a Stable Diffusion 1.5 U-Net denoising a $64 \times 64 \times 4$ latent** (which the VAE will later decode to a $512 \times 512$ RGB image). Every channel count, resolution, and parameter number I quote will be anchored to that concrete model, with ImageNet/ADM numbers brought in where they make the science precise.

## Why a U-Net? The shape of the problem dictates the shape of the network

Before any code, the central question: of all possible architectures, why did the field converge on a U-Net for the denoiser? The answer is that the U-Net's shape mirrors the structure of the *task*, and when the architecture's inductive bias matches the problem, you need far less data and compute to reach a given quality. Three facts about the denoising task pin down the design.

**Fact one: the output is the same shape as the input.** The network takes a noisy image (or latent) of shape $H \times W \times C$ and must output a noise prediction $\hat\epsilon$ of *exactly* the same shape $H \times W \times C$ — one predicted noise value per input element. This is a dense, pixel-to-pixel prediction task, structurally identical to semantic segmentation, which is precisely the task the U-Net was invented for (Ronneberger et al., 2015, for biomedical segmentation). A classifier collapses an image to a single label and can throw away spatial resolution freely; a denoiser cannot — it must reconstruct a full-resolution map. So the architecture must *return* to the input resolution at the end. An encoder-decoder that goes down and then back up is the natural shape.

**Fact two: images are multi-scale, and so is the noise to be removed.** Look at what "remove the noise" means at different noise levels (we made this precise in the worked examples of the foundations post). At low noise levels, the image is almost intact and denoising is a *local, high-frequency* operation — find the faint speckle on top of an edge and subtract it. At high noise levels, the image is nearly gone and denoising is a *global, low-frequency* operation — the network has to decide what the overall composition should be, which requires integrating information across the entire image. A single fixed receptive field cannot do both well. You need a representation that captures fine local detail *and* coarse global structure simultaneously. A multi-resolution encoder gives you exactly that: the early, high-resolution stages see fine detail with small receptive fields; the deep, low-resolution stages see the whole image at once because each of their "pixels" summarizes a large patch of the original. The U-Net builds a feature pyramid, and the pyramid is the multi-scale representation the task demands.

**Fact three: you must not lose the fine detail you compressed away.** Here is the tension. To get a global view, you downsample — but downsampling throws away high-frequency information, and the output needs that information back to be sharp. A plain encoder-decoder (an autoencoder) that compresses to a bottleneck and decodes back is exactly the architecture that *blurs*, because the bottleneck is an information chokepoint. The U-Net's defining trick — the thing that makes it a U-Net and not just an autoencoder — is the **skip connection**: at every resolution, the encoder's feature map is handed *directly* across to the decoder stage at the same resolution, bypassing the bottleneck entirely. The decoder concatenates the skipped high-resolution features with its upsampled low-resolution features, so it has both the global plan (from below, through the bottleneck) and the local detail (from the side, through the skip). That is the whole reason it is shaped like a "U": down on the left, up on the right, and horizontal bridges connecting matching rungs.

![A residual-block diagram showing GroupNorm and SiLU before each convolution, a timestep scale-shift in the middle, and a residual add at the end](/imgs/blogs/the-diffusion-unet-2.png)

Let me make the skip-connection argument quantitative, because it is the load-bearing claim and it deserves more than hand-waving. Consider the information flow. The encoder maps the input $x$ to a hierarchy of feature maps $h_1, h_2, \dots, h_L$ at successively coarser resolutions, ending at a bottleneck $h_L$ that is small — say $8 \times 8$ for a $64 \times 64$ input, a $64\times$ spatial reduction. If the decoder had to reconstruct the full $64 \times 64$ noise map from *only* $h_L$, it would be trying to recover roughly 4096 spatial positions worth of high-frequency content from 64 positions worth of features. By the data-processing inequality, information destroyed by the downsampling cannot be recovered downstream from the bottleneck alone — whatever fine detail was averaged away in $h_L$ is gone. The skip connection routes around this: the decoder stage at resolution $r$ receives $h_r^{\text{enc}}$ directly, so the high-frequency content at resolution $r$ never has to survive the bottleneck. Formally, the decoder feature at level $r$ is $h_r^{\text{dec}} = f_r\big(\text{concat}(\text{up}(h_{r+1}^{\text{dec}}),\ h_r^{\text{enc}})\big)$ — a function of *both* the coarse plan from below and the fine detail from the side. The bottleneck carries the *semantics* (what to draw); the skips carry the *texture* (how to render it sharply). Remove the skips and FID collapses; this has been ablated repeatedly and it is not subtle.

This is also why the U-Net refines both global layout and fine texture in a single forward pass. The two jobs are handled by two different information pathways that meet in the decoder. When you later watch a diffusion sampler run and see the global composition lock in early (in the first few steps, at high noise) and the texture sharpen late (in the last steps, at low noise), you are watching the bottleneck and the skip pathways do their respective jobs across the noise schedule.

There is a deeper reason the multi-scale shape is the right bet, and it ties straight back to the spine of the series. Natural images live on a thin, curved manifold inside the astronomically large pixel space — that is the manifold hypothesis the foundations posts established. Crucially, that manifold has structure at *many* scales at once: coarse structure (a face has two eyes above a nose above a mouth) and fine structure (skin pores, hair strands, the specular highlight in an iris). A generative network that wants to stay on the manifold has to respect constraints at every scale simultaneously, and it cannot do that with a single fixed receptive field. The U-Net's feature pyramid is, in effect, a multi-scale coordinate system for the manifold: the coarse stages place the sample in the right *neighborhood* (this is a face, oriented this way), and the fine stages move it onto the manifold's *surface* (these exact textures). The denoiser's job at each step is to nudge an off-manifold noisy sample back toward the surface, and because the off-manifold error itself has structure at every scale (low-frequency error at high noise, high-frequency error at low noise), a multi-scale corrector is exactly what the geometry calls for. This is the architectural echo of the score-function view from the SDE post: the network estimates the direction back to high-density regions, and that direction is most naturally computed scale by scale.

A note on terminology, since the series promised to define jargon on first use. **Receptive field**: the region of the input that influences a given output element — small in early conv layers, large after downsampling. **Feature map**: the multi-channel tensor of activations at a given layer, shape $C \times H \times W$. **Inductive bias**: the assumptions baked into an architecture (a convolution assumes translation equivariance and locality; that assumption is the U-Net's strength here and, as we will see at the end, its eventual weakness).

## The building blocks: residual blocks, down/up stages, the bottleneck

Zoom in one level. A diffusion U-Net is built from a small number of repeated parts. Understanding the parts and how they nest is most of understanding the whole.

### The residual block — the workhorse unit

The atom of the U-Net is the **residual block** (ResBlock). The diffusion ResBlock is a specific recipe, and it differs in important ways from a ResNet classifier block (which uses BatchNorm and ReLU). The diffusion ResBlock uses **GroupNorm** (not BatchNorm) and **SiLU** (not ReLU), and it carries an extra input nothing in a classifier has: the timestep embedding. Figure 2 lays out its internals; here is the dataflow in words.

Given an input feature map $x$ of shape $C \times H \times W$ and a timestep embedding vector $t_\text{emb}$:

1. **GroupNorm → SiLU → Conv 3×3.** Normalize the channels (in groups), apply the smooth SiLU nonlinearity, then a $3 \times 3$ convolution. This is the "pre-activation" ordering: normalize and activate *before* the convolution, which trains more stably in deep nets.
2. **Inject the timestep** via a learned scale-and-shift (the AdaGN / FiLM modulation we derive in the next section). A small linear layer maps $t_\text{emb}$ to a per-channel scale and shift that modulate the activations.
3. **GroupNorm → SiLU → Conv 3×3** again, with the last convolution often **zero-initialized** so that at the start of training the whole block is the identity (the residual path dominates), which stabilizes early training.
4. **Residual add:** output $= x + F(x)$. If the block changes the channel count (the in and out channels differ), a $1 \times 1$ convolution on the skip path matches the dimensions.

Why GroupNorm instead of BatchNorm? Two reasons, both real. First, diffusion is trained with a *random timestep per sample in the batch*, so different elements of a batch are at wildly different noise levels — BatchNorm's batch statistics would mix a barely-noised image with a near-pure-noise one, which is meaningless. GroupNorm normalizes each sample independently, so it is immune to this. Second, BatchNorm behaves differently at train and inference time (running statistics), which is a notorious source of bugs; GroupNorm behaves identically, removing a whole class of train/test mismatch. Why SiLU (also called Swish, $x \cdot \sigma(x)$) instead of ReLU? It is smooth and non-monotonic, empirically gives slightly better generative quality, and avoids the dead-unit problem; the diffusion literature standardized on it (ADM, Stable Diffusion, and essentially everything since use SiLU).

Here is a faithful, runnable PyTorch implementation of the diffusion ResBlock with timestep scale-shift. This is close to what 🤗 `diffusers` calls a `ResnetBlock2D`.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """A diffusion residual block: GroupNorm-SiLU-Conv twice, with a
    timestep scale-shift injected in the middle (AdaGN / FiLM)."""
    def __init__(self, in_ch, out_ch, t_emb_dim, groups=32):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        # Project the time embedding to 2 * out_ch: a scale and a shift
        # per output channel. This is the FiLM / AdaGN modulation.
        self.time_proj = nn.Linear(t_emb_dim, 2 * out_ch)

        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv2.weight)   # zero-init last conv: block = identity at t=0 of training
        nn.init.zeros_(self.conv2.bias)

        # 1x1 skip to match channels if they differ
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))          # GroupNorm -> SiLU -> Conv

        # Inject timestep: split the projection into scale and shift,
        # broadcast over the spatial dims, and modulate the normalized features.
        scale, shift = self.time_proj(F.silu(t_emb)).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[..., None, None]) + shift[..., None, None]

        h = self.conv2(F.silu(h))                       # SiLU -> Conv (zero-init)
        return h + self.skip(x)                         # residual connection
```

Read the `forward`. The two convolutions do the spatial mixing; the `time_proj` line is the *only* place the timestep enters this block, and it enters as a per-channel `(1 + scale)` multiply and a `shift` add applied right after the second GroupNorm. That `* (1 + scale) + shift` is the FiLM/AdaGN modulation we will derive properly in a moment. The `+ self.skip(x)` is the residual highway that keeps gradients flowing through dozens of these blocks.

### Down stages, up stages, and the bottleneck

A diffusion U-Net stacks ResBlocks into **stages**, one per resolution level. The structure, top to bottom and back up:

- **Encoder (down path):** at each resolution, run a couple of ResBlocks (and, at the low resolutions, attention blocks — covered below), then **downsample** by a strided convolution (or average pool) that halves $H$ and $W$ and typically doubles the channel count. So as you go down, the spatial grid shrinks and the channels grow — the network trades spatial resolution for representational depth.
- **Bottleneck (mid block):** at the lowest resolution, one or two ResBlocks with a self-attention block sandwiched between them. This is where the most global, most abstract processing happens — the receptive field here covers the entire image.
- **Decoder (up path):** the mirror image. At each resolution, **upsample** (nearest-neighbor or transposed conv) to double $H$ and $W$ and halve channels, **concatenate the matching encoder skip**, then run ResBlocks (and attention at the low resolutions). The decoder climbs back to the input resolution.
- **Output head:** a final GroupNorm → SiLU → Conv that maps the last feature map to the output channel count — for SD 1.5 latents, 4 channels, the predicted noise $\hat\epsilon$ at $64 \times 64 \times 4$.

The channel-vs-resolution trade is worth internalizing because it is *the* lever for the attention-cost argument later. Concretely, Stable Diffusion 1.5's U-Net uses `block_out_channels = (320, 640, 1280, 1280)` across four resolution levels. Starting from a $64 \times 64$ latent:

| Stage | Resolution | Channels | Tokens (H·W) | Self-attn? |
|---|---|---|---|---|
| 1 (top) | 64×64 | 320 | 4096 | no |
| 2 | 32×32 | 640 | 1024 | yes |
| 3 | 16×16 | 1280 | 256 | yes |
| 4 (mid) | 8×8 | 1280 | 64 | yes |

![A matrix of U-Net stages showing channels rising and resolution falling together so attention runs only on short token sequences](/imgs/blogs/the-diffusion-unet-3.png)

Figure 3 is this table as a figure. The pattern — channels going $320 \to 640 \to 1280 \to 1280$ while resolution goes $64 \to 32 \to 16 \to 8$ — is the canonical diffusion U-Net shape, and the "no attention at 64×64, attention everywhere below" column is the single most important design decision in the whole architecture. The next two sections explain *why* that column reads the way it does, first the time/condition embedding, then the attention gating.

## Time and condition embedding: FiLM, AdaGN, and the scale-shift derivation

A classifier's network is a fixed function of its input. A diffusion U-Net is *not*: the same network must behave differently at different noise levels — like a gentle sharpener at low $t$, like a bold hallucinator at high $t$. So the timestep $t$ is a first-class input, and it must reach *every* block, because every block's job depends on the noise level. The mechanism is elegant and worth deriving rather than asserting.

![A graph showing the timestep becoming a global scale-shift into every block while text enters locally through cross-attention](/imgs/blogs/the-diffusion-unet-4.png)

### From a scalar timestep to a rich embedding

First, the scalar $t$ (an integer in $[0, 1000)$ for DDPM) is mapped to a high-dimensional vector with a **sinusoidal embedding**, exactly the positional encoding from the original Transformer. For embedding dimension $d$ and a chosen frequency base (10000 is standard):

$$
\text{emb}(t)_{2i} = \sin\!\left(\frac{t}{10000^{2i/d}}\right), \qquad
\text{emb}(t)_{2i+1} = \cos\!\left(\frac{t}{10000^{2i/d}}\right).
$$

Why sinusoidal rather than just feeding the raw scalar $t$? Because a single scalar is a terrible input for a network that needs to make *fine* distinctions between nearby noise levels and *coarse* distinctions between far-apart ones. The sinusoidal embedding spreads $t$ across many frequencies, so the network can read off both "roughly where in the schedule am I" (low-frequency components) and "exactly which step" (high-frequency components) with simple linear projections. It is the same reason transformers use sinusoidal position encodings, and it works for the same reason. The embedding then passes through a small MLP (two linear layers with a SiLU between them) to produce the final timestep embedding $t_\text{emb}$, typically 1280-dimensional for SD 1.5 (4× the base channel count). That $t_\text{emb}$ vector is computed *once per forward pass* and broadcast to every ResBlock.

### The FiLM / AdaGN modulation, derived

Now the key question: *how* should the timestep modulate a block's activations? The answer the field settled on is **Feature-wise Linear Modulation (FiLM)** — also called **adaptive group normalization (AdaGN)** in the diffusion context. The idea: condition the network by applying a per-channel affine transform to the normalized features, where the affine parameters are *predicted from the conditioning vector*.

Let $h \in \mathbb{R}^{C \times H \times W}$ be a block's features after GroupNorm. GroupNorm has already standardized $h$ to roughly zero mean and unit variance per group. FiLM then applies a learned, conditioning-dependent scale $\gamma$ and shift $\beta$, one per channel:

$$
\text{AdaGN}(h, t_\text{emb}) = \gamma(t_\text{emb}) \odot \text{GroupNorm}(h) + \beta(t_\text{emb}),
$$

where $\gamma(t_\text{emb}), \beta(t_\text{emb}) \in \mathbb{R}^{C}$ are produced by a single linear layer $W_{t} \, t_\text{emb} = [\gamma; \beta]$, and $\odot$ broadcasts the per-channel vector over the spatial dimensions. In the code above this is exactly the line `h = self.norm2(h) * (1 + scale) + shift`, with $\gamma = 1 + \text{scale}$ (the $1+$ means "at initialization the scale is identity").

Why does this particular form work so well, and why scale-*and*-shift rather than just one or the other? Here is the derivation that makes it click. After GroupNorm, every channel of $h$ has been forced to a standard distribution — the normalization has *removed* per-channel mean and variance information. A conditioning signal that needs to change the network's behavior has exactly two cheap, global knobs to turn: it can rescale a channel (amplify or suppress that feature — the $\gamma$ scale) and it can re-center it (push the feature's baseline up or down — the $\beta$ shift). Together, an affine map $\gamma h + \beta$ is the most general *per-channel linear* transform, and it is the natural inverse of what GroupNorm just did. So AdaGN is, precisely, "let the timestep decide how much to undo the normalization, per channel." This is far cheaper than, say, concatenating $t_\text{emb}$ as extra channels (which would need full convolutions to mix it in) and far more expressive than adding a single bias. The cost is just one linear layer per block producing $2C$ numbers.

There is a subtle but important point about *where* the modulation lands. Because GroupNorm normalizes per-group, and AdaGN's scale-shift is per-channel applied *after* the norm, the timestep gets to set the gain and bias of every feature channel independently at every block — a global, low-dimensional, but per-feature steering signal. The ADM paper (Dhariwal & Nichol, 2021) showed empirically that this adaptive group norm conditioning beats simply *adding* the timestep embedding to the features, which was the original DDPM approach. That one change — additive timestep injection to AdaGN scale-shift — is one of the concrete improvements we will tabulate in the ADM section.

#### Worked example: the parameter cost of AdaGN in one block

Let me put a number on it, because "it's cheap" should be defensible. Take a mid-resolution ResBlock with $C = 1280$ output channels and a timestep embedding of dimension $d_t = 1280$. The AdaGN projection is a linear layer from $d_t$ to $2C$, i.e. $1280 \to 2560$, which is $1280 \times 2560 + 2560 \approx 3.28$M parameters. Compare that to the two $3 \times 3$ convolutions in the same block: each is $C \times C \times 3 \times 3 = 1280^2 \times 9 \approx 14.7$M parameters, so $\approx 29.5$M for the pair. The timestep projection is about 10% of the block's parameters — non-trivial but small, and it buys the entire ability to condition on noise level. Across all the blocks in SD 1.5's U-Net, the timestep MLP and per-block projections together are on the order of a few percent of the total 860M parameters. The conditioning is genuinely cheap; the convolutions and attention are where the parameters live, which we will tally at the end.

### Text conditioning: cross-attention, not FiLM

The timestep is a *global* signal — it is the same everywhere in the image, so a per-channel scale-shift is the right tool. Text is different. A prompt like "a red ball on the left, a blue cube on the right" must steer *different spatial locations differently* — the left region toward "red ball," the right toward "blue cube." A global per-channel modulation cannot do that; it has no spatial selectivity. So text enters the U-Net through a fundamentally different mechanism: **cross-attention**, which is spatially adaptive by construction. We give cross-attention its own section because it is the heart of how text-to-image actually works, and it is what the [classifier-free guidance](/blog/machine-learning/image-generation/classifier-free-guidance) post manipulates at sampling time.

## Self-attention at low resolution: why it is gated, with the FLOP count

Convolutions are local. A $3 \times 3$ conv at the top resolution sees a $3 \times 3$ neighborhood; even after many layers, the receptive field grows only linearly with depth. But images have *long-range* structure — the two ears of a cat must match, the horizon must be level across the whole frame, a face must be symmetric. Pure convolution struggles to enforce global coherence; it tends to produce locally-plausible, globally-inconsistent results (three-armed people, mismatched eyes). The fix is **self-attention**, which lets every spatial position directly attend to every other position in a single layer — global mixing in one shot. But attention is expensive, and *how* expensive depends violently on the resolution. That dependence is the entire reason attention is gated to the low-resolution stages.

![A matrix showing self-attention cost scaling with the square of the token count so full-resolution attention is hundreds of times costlier](/imgs/blogs/the-diffusion-unet-5.png)

### The quadratic cost, made precise

Self-attention over a feature map of spatial size $H \times W$ treats each of the $N = H \cdot W$ spatial positions as a token. It computes an $N \times N$ attention matrix (every token's similarity to every other token), which is the crux: the attention matrix has $N^2$ entries, and computing it costs $O(N^2 d)$ FLOPs (the $QK^\top$ matmul plus the $\times V$ matmul), where $d$ is the channel dimension. The memory to store the attention scores is $O(N^2)$ — and *that* is usually the binding constraint on a GPU, because $N^2$ grows fast.

Now plug in the resolutions of an SD 1.5 latent U-Net. The number of tokens at each stage is the spatial grid size:

| Stage | Tokens $N$ | $N^2$ (attn scores) | Relative cost vs 16×16 |
|---|---|---|---|
| 64×64 | 4096 | 16,777,216 | 256× |
| 32×32 | 1024 | 1,048,576 | 16× |
| 16×16 | 256 | 65,536 | 1× |
| 8×8 | 64 | 4,096 | 0.06× |

The arithmetic is brutal and decisive. Self-attention at the top $64 \times 64$ stage would need a $4096 \times 4096$ attention matrix — about 16.8M entries *per attention head*, and SD 1.5 uses 8 heads. At fp16 that is hundreds of megabytes of attention scores for a *single* attention layer at a *single* timestep in a *single* batch element, before you even multiply by the number of attention layers. It is not that it is slow; it is that it does not fit. Drop to $16 \times 16$ and the attention matrix is $256 \times 256 = 65{,}536$ entries — 256× smaller, trivially affordable. This is the whole story of the "Self-attn?" column in the stage table: attention is **gated to the resolutions where $N$ is small enough that $N^2$ is affordable**. SD 1.5 puts attention at $32 \times 32$, $16 \times 16$, and the $8 \times 8$ bottleneck, but *not* at the $64 \times 64$ top stage, exactly because $4096^2$ is the cliff.

#### Worked example: attention memory at 64×64 vs 16×16

Concretely, in fp16 (2 bytes per score), the attention-score tensor for one head is $2 N^2$ bytes. At $64 \times 64$: $2 \times 4096^2 \approx 33.6$ MB per head, $\times 8$ heads $\approx 268$ MB for *one* attention op. At $16 \times 16$: $2 \times 256^2 \approx 131$ KB per head, $\times 8 \approx 1$ MB. The U-Net has several attention layers and you process a batch with classifier-free guidance doubling the effective batch — multiply the $64 \times 64$ figure out and you blow past a 24 GB consumer GPU on attention scores alone, while the $16 \times 16$ version is a rounding error. *That* is why you never see full-resolution self-attention in a classic diffusion U-Net. (Modern work — linear attention in SANA, the efficient attention in newer DiTs — attacks exactly this $N^2$ wall; it is the central efficiency frontier and we will return to it in the speed track.)

There is also a *quality* reason the gating is fine, not just a cost reason. The high-resolution stages handle local texture, which convolution already does well — they do not *need* global mixing. The global structural decisions (composition, symmetry, object coherence) happen in the low-resolution stages where the receptive field is large and a token already summarizes a big patch — which is exactly where attention is cheap. So the gating aligns with the work: put the expensive global operator where global decisions are made and where it happens to be affordable, and let convolution handle the cheap local detail at full resolution. The architecture is *efficient because the cost structure and the task structure point the same way*. (For deeper background on the attention mechanism itself — queries, keys, values, multi-head — the [large-language-model track](/blog/machine-learning/large-language-model/) covers the transformer in detail; here we only need the cost model and the cross-attention variant below.)

### The self-attention block, in code

What does a U-Net self-attention block actually compute? It is ordinary multi-head self-attention applied to the spatial positions of a feature map. The feature map $h \in \mathbb{R}^{C \times H \times W}$ is reshaped to a sequence of $N = H \cdot W$ tokens, each a $C$-dimensional vector; the block computes queries, keys, and values *all from those same image tokens* (this is the difference from cross-attention, where keys and values come from text); it attends; and it reshapes back to a feature map, with a residual add. Here is the block written out, the kind that runs at the $32 \times 32$, $16 \times 16$, and $8 \times 8$ stages of an SD U-Net:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention2D(nn.Module):
    """Multi-head self-attention over the spatial positions of a feature map.
    Lets every position attend to every other position in one layer (global mixing)."""
    def __init__(self, channels, heads=8, groups=32):
        super().__init__()
        self.heads, self.scale = heads, (channels // heads) ** -0.5
        self.norm = nn.GroupNorm(groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)   # one 1x1 conv produces Q, K, V
        self.proj = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)                  # zero-init: block starts as identity

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W                                         # number of tokens = spatial positions
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)  # each (B, C, H, W)
        # reshape to (B, heads, N, C // heads)
        def heads_view(t):
            return t.reshape(B, self.heads, C // self.heads, N).transpose(-2, -1)
        q, k, v = map(heads_view, (q, k, v))
        attn = (q @ k.transpose(-2, -1)) * self.scale     # (B, heads, N, N) <- the N^2 matrix
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(-2, -1).reshape(B, C, H, W)
        return x + self.proj(out)                         # residual add
```

The line `attn = (q @ k.transpose(-2, -1))` is *exactly* the $N \times N$ matrix from the cost analysis — you can see in the code why the memory is $O(N^2)$: that tensor has shape `(B, heads, N, N)`, and at $N = 4096$ it is the thing that does not fit. In real `diffusers` you would replace the explicit `softmax(q @ kᵀ) @ v` with `F.scaled_dot_product_attention(q, k, v)`, which computes the same result via a fused FlashAttention kernel that *never materializes* the full `(N, N)` matrix — it tiles the computation and keeps only $O(N)$ memory live. That single substitution is what lets modern SD U-Nets run attention at $32 \times 32$ ($N = 1024$) comfortably; without it the $1024 \times 1024$ score matrix at every mid-stage attention layer adds up fast.

A word on **multi-head** attention, since the ADM ablations cared about it. Instead of one attention over the full $C$-dimensional features, multi-head attention splits the channels into $h$ groups (heads), runs attention independently in each $C/h$-dimensional subspace, and concatenates. Why split? Because a single attention head can only express *one* pattern of "what attends to what" — one similarity geometry. Multiple heads let the network learn several relationship types in parallel: one head might match the cat's left ear to its right ear (symmetry), another might bind a color to a region, another might track the horizon line. ADM found that scaling the number of heads — or equivalently fixing the channels-per-head and letting heads grow with width — improved FID, because more heads means more simultaneous global relationships the network can enforce. In the code above, `heads=8` with $C = 320$ gives 40 channels per head; SD 1.5 uses 8 heads at the mid stages.

### A full forward pass, stage by stage

It helps to trace a single forward pass all the way through, because the skip bookkeeping is where implementations get fiddly and where the U-Net's structure becomes concrete. Take a batch of noisy latents $x_t$ of shape $(B, 4, 64, 64)$, a timestep $t$, and the CLIP text embedding $c$ of shape $(B, 77, 768)$.

1. **Embed the timestep.** Compute the sinusoidal embedding of $t$, push it through the MLP, get $t_\text{emb}$ of shape $(B, 1280)$. Compute it *once*; it will be handed to every ResBlock.
2. **Stem.** A $3 \times 3$ conv lifts the 4 input channels to 320: now $(B, 320, 64, 64)$.
3. **Down stage 1 (64×64, 320 ch, no attention).** Two ResBlocks (each gets $t_\text{emb}$). **Save both block outputs to the skip stack.** Then downsample to $(B, 640, 32, 32)$ and save that too. (The decoder will pop these in reverse.)
4. **Down stage 2 (32×32, 640 ch, attention).** Two ResBlocks, each followed by a transformer block (self-attention over $N=1024$ tokens, then cross-attention into $c$, then a feed-forward). Save outputs to the skip stack. Downsample to $(B, 1280, 16, 16)$.
5. **Down stage 3 (16×16, 1280 ch, attention).** Same pattern, $N=256$ tokens. Save skips. Downsample to $(B, 1280, 8, 8)$.
6. **Down stage 4 (8×8, 1280 ch).** ResBlocks, save skips. (In SD 1.5 the very bottom is plain ResBlocks; attention lives in the mid block.)
7. **Mid block (bottleneck, 8×8, 1280 ch).** ResBlock → self-attention ($N=64$) + cross-attention → ResBlock. This is the most global processing in the network; every spatial position sees every other and the full prompt.
8. **Up stages, mirror image.** At each resolution: **pop the matching encoder feature from the skip stack and concatenate it** to the current upsampled feature along the channel dimension (so a $1280$-channel decoder feature concatenated with a $1280$-channel skip becomes $2560$ channels going into the next ResBlock, which projects back down). Run ResBlocks (and transformer blocks at the attention resolutions). Upsample. Climb back: $8 \to 16 \to 32 \to 64$.
9. **Output head.** GroupNorm → SiLU → $3 \times 3$ conv mapping 320 channels back to 4: the predicted noise $\hat\epsilon$ of shape $(B, 4, 64, 64)$ — same shape as the input, as promised.

The skip stack is the part worth dwelling on. Notice the encoder *pushes* a skip after every block (not just every stage), and the decoder *pops* them in last-in-first-out order, so the decoder block at a given resolution always receives the encoder feature from the *same* resolution. The concatenation (not addition) is deliberate: concatenating preserves both the decoder's coarse plan and the encoder's fine detail as separate channels, letting the next ResBlock *learn* how to combine them, rather than forcing a fixed sum. This is why a `diffusers` U-Net forward threads a Python list of skip tensors through the whole call — the implementation detail that makes the "U" real.

#### Worked example: the channel count at the noisiest concatenation

Concretely, at the first up stage (16×16, mirroring down stage 3), the decoder arrives with a $(B, 1280, 16, 16)$ feature upsampled from the 8×8 bottleneck, and it pops a $(B, 1280, 16, 16)$ encoder skip. Concatenation along channels gives $(B, 2560, 16, 16)$ — *double* the channels — which the up-stage ResBlock's first conv ($2560 \to 1280$) projects back down to 1280. So the U-Net momentarily holds 2560-channel feature maps at the decoder, which is part of why the decoder side carries slightly more parameters than the encoder side: every up ResBlock's first conv is sized for the concatenated input. If you ever wondered why a U-Net is not perfectly symmetric in parameter count, this is it — the skips fatten the decoder's input convolutions.

## Cross-attention: how text actually steers the pixels

Self-attention mixes image positions with *each other*. **Cross-attention** mixes image positions with the *text* — it is the bridge between the prompt and the denoiser, and it is where "a tabby cat in a wizard hat" becomes a tabby cat in a wizard hat rather than random noise. The mechanism is a small twist on self-attention, and once you see it, the whole text-to-image pipeline clicks.

![A graph of cross-attention where image features form the queries and frozen text embeddings form the keys and values](/imgs/blogs/the-diffusion-unet-7.png)

In self-attention, the queries $Q$, keys $K$, and values $V$ all come from the *same* sequence (the image tokens). In cross-attention, they come from *two* sequences: the **queries come from the image features**, and the **keys and values come from the text embeddings**. Concretely, in a U-Net cross-attention block:

- The prompt is encoded once by a frozen text encoder (CLIP's text transformer for SD 1.5) into a sequence of token embeddings $c \in \mathbb{R}^{77 \times 768}$ — 77 tokens (CLIP's context length), each 768-dimensional.
- At a given U-Net stage, the image feature map $h \in \mathbb{R}^{C \times H \times W}$ is flattened to $N = H \cdot W$ tokens of dimension $C$.
- Queries: $Q = h W_Q$ (from the image — "what does this position need?").
- Keys and values: $K = c W_K$, $V = c W_V$ (from the text — "what does each word offer?").
- Attention: $\text{Attn}(Q, K, V) = \text{softmax}\!\left(\dfrac{QK^\top}{\sqrt{d_k}}\right) V$.

The result has shape $N \times d$ — one updated feature vector per image position, formed as a *text-weighted sum*. Each image position computes a similarity to every text token, softmaxes those similarities into weights, and pulls in a weighted blend of the text values. So the position that will become the cat's fur attends strongly to the "tabby" and "cat" tokens; the position that will become the hat attends to "wizard" and "hat." That is the spatial selectivity that FiLM cannot provide and that text conditioning requires.

Notice the cost is *not* the $N^2$ problem of self-attention. The cross-attention matrix is $N \times M$ where $M = 77$ is the number of text tokens — so it scales as $O(N \cdot M \cdot d)$, *linear* in the number of image tokens, not quadratic. That is why cross-attention can run at every attention stage cheaply alongside self-attention: a single text token is a tiny, fixed sequence. (In practice a U-Net "attention block" — `Transformer2DModel` in diffusers — bundles a self-attention layer, a cross-attention layer, and a feed-forward, in that order, mirroring a transformer block but with the cross-attention reaching into the text.)

Here is a faithful cross-attention block in PyTorch, the kind that lives at the $32 \times 32$, $16 \times 16$, and $8 \times 8$ stages of an SD U-Net:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    """Image features query frozen text embeddings (keys/values).
    query_dim = U-Net feature channels; context_dim = text encoder width (768 for CLIP)."""
    def __init__(self, query_dim, context_dim=768, heads=8, dim_head=64):
        super().__init__()
        inner = heads * dim_head
        self.heads, self.scale = heads, dim_head ** -0.5
        self.to_q = nn.Linear(query_dim, inner, bias=False)     # from image
        self.to_k = nn.Linear(context_dim, inner, bias=False)   # from text
        self.to_v = nn.Linear(context_dim, inner, bias=False)   # from text
        self.to_out = nn.Linear(inner, query_dim)

    def forward(self, x, context):
        # x: (B, N, query_dim) flattened image tokens; context: (B, M, context_dim) text tokens
        B, N, _ = x.shape
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)
        # split heads -> (B, heads, seq, dim_head)
        q, k, v = (t.view(B, -1, self.heads, t.shape[-1] // self.heads).transpose(1, 2)
                   for t in (q, k, v))
        # scaled dot-product attention; in production use F.scaled_dot_product_attention for FlashAttention
        attn = (q @ k.transpose(-2, -1)) * self.scale     # (B, heads, N, M)
        attn = attn.softmax(dim=-1)
        out = attn @ v                                    # (B, heads, N, dim_head)
        out = out.transpose(1, 2).reshape(B, N, -1)       # merge heads
        return self.to_out(out)
```

In real 🤗 `diffusers` code you would replace the explicit `softmax(q @ kᵀ) @ v` with `torch.nn.functional.scaled_dot_product_attention(q, k, v)`, which dispatches to FlashAttention or a memory-efficient kernel — the same numerics, far less memory, because it never materializes the full $N \times M$ score matrix. The point of writing it out longhand is to see that cross-attention is *literally* "image asks, text answers," and that the `context` tensor is the only place the prompt touches the denoiser.

How does classifier-free guidance fit here? At sampling time you run the U-Net twice — once with the real text `context` and once with an empty/unconditional `context` — and extrapolate between the two noise predictions. Cross-attention is exactly the layer that makes the two runs differ: with a real prompt, the keys and values carry content; with the empty prompt, they carry the null embedding. Everything CFG does to trade diversity for prompt adherence, it does *through* these cross-attention layers. The full derivation of why that extrapolation sharpens prompt alignment is in the [classifier-free guidance](/blog/machine-learning/image-generation/classifier-free-guidance) post; here the takeaway is structural — guidance is a sampling-time trick layered on top of the cross-attention conditioning the U-Net was trained with.

## The ADM improvements: how to make the U-Net beat GANs

The DDPM U-Net (Ho et al., 2020) worked, but it was Dhariwal and Nichol's 2021 paper — bluntly titled *"Diffusion Models Beat GANs on Image Synthesis"* — that turned the diffusion U-Net into a state-of-the-art architecture and produced the **ADM** (Ablated Diffusion Model) backbone that essentially everything since inherits. The paper is, in large part, a careful ablation of U-Net design choices. The improvements are concrete and each one moved FID, so they are worth enumerating precisely.

![A before and after comparison of a vanilla DDPM U-Net and the ADM U-Net with wider attention adaptive group norm and lower FID](/imgs/blogs/the-diffusion-unet-6.png)

**1. More attention, at more resolutions, with more heads.** DDPM used self-attention at a single resolution ($16 \times 16$). ADM added attention at $32 \times 32$, $16 \times 16$, *and* $8 \times 8$, and — crucially — used **multiple attention heads** (and explored fixing the number of channels *per head* rather than the number of heads, which scales attention capacity sensibly with width). More attention at more scales means more global coherence across more spatial resolutions, and it consistently improved FID in their ablations.

**2. Adaptive group normalization (AdaGN) for conditioning.** This is the scale-shift mechanism we derived above. DDPM injected the timestep (and class) by *adding* the embedding to the features; ADM replaced that with the AdaGN affine modulation $\gamma(\text{emb}) \cdot \text{GroupNorm}(h) + \beta(\text{emb})$. This was one of the larger single wins in their ablation table — adaptive normalization conditions far more effectively than additive injection, and it is the direct ancestor of the AdaLN-Zero conditioning that DiT later uses.

**3. BigGAN-style residual blocks for up/downsampling.** ADM borrowed the residual-block design for changing resolution from BigGAN — using the residual block itself to do the up/downsampling (with the skip path appropriately resampled) rather than a separate pooling/conv. This is a cleaner, better-behaved way to change resolution inside a residual network.

**4. Rescaling residual connections and deeper/wider nets with more compute.** They scaled model width and depth and rescaled residual connections by $1/\sqrt{2}$ in places for stability, then spent the compute where the ablations said it helped.

The headline result: on **ImageNet 256×256**, ADM (with classifier guidance) reached **FID ≈ 4.59**, beating BigGAN-deep's FID of around 6.95 — the first time a diffusion model decisively beat the best GAN on a hard, high-resolution conditional benchmark. (Without guidance the unconditional/lightly-guided numbers are higher; the often-quoted ADM-G figure around 4.6 includes their classifier guidance.) Vanilla DDPM on comparable settings sat well above that. Figure 6 contrasts the two backbones: same U-Net skeleton, but wider attention, AdaGN conditioning, and BigGAN res-blocks take it from "competitive" to "beats GANs."

| Design choice | DDPM (2020) | ADM (2021) | Effect |
|---|---|---|---|
| Attention resolutions | 16×16 only | 32, 16, 8 | better global coherence |
| Attention heads | 1 | multiple (fixed channels/head) | more attention capacity |
| Timestep conditioning | additive | AdaGN scale-shift | large FID win |
| Resampling blocks | plain conv/pool | BigGAN res-blocks | cleaner resolution change |
| ImageNet-256 FID | well above 10 | ≈ 4.59 (with guidance) | beats BigGAN-deep |

**Imagen's choices**, for contrast, since the scope asks for them. Google's Imagen (Saharia et al., 2022) kept the U-Net but made different bets at the edges: it used a **large frozen T5-XXL text encoder** instead of CLIP (finding that scaling the text encoder helped text-image alignment more than scaling the U-Net), introduced an **"Efficient U-Net"** variant that shifted parameters toward the lower resolutions and added more residual blocks there (where compute is cheaper per the resolution argument above), and used a **cascade** of U-Nets — a base $64 \times 64$ model plus super-resolution U-Nets to $256$ and $1024$ — rather than a single latent-space model. Imagen also leaned on **dynamic thresholding** at sampling time to allow high guidance weights without saturation. The throughline: the U-Net backbone is the same family; the design space is *where to attend, how to condition, what encodes the text, and whether to work in pixel space (Imagen, cascaded) or latent space (Stable Diffusion, single model)*.

## Where the parameters and the compute go

Engineers ask the practical question: SD 1.5's U-Net is about **860M parameters** — where do they sit, and where does the FLOP budget go at inference? Knowing this tells you what to optimize and what to leave alone.

**Parameter distribution.** The parameters are dominated by the convolutions and the linear projections inside attention, concentrated in the *deep, wide* stages. Recall channels go $320 \to 640 \to 1280 \to 1280$. A $3 \times 3$ conv has $C_\text{in} \cdot C_\text{out} \cdot 9$ weights, so a conv at the $1280$-channel stage has $1280^2 \cdot 9 \approx 14.7$M parameters — versus $320^2 \cdot 9 \approx 0.9$M at the top stage, a 16× difference *per conv*. Since parameter count scales with the *square* of the channel width, the deepest stages (the $16 \times 16$ and $8 \times 8$ blocks at 1280 channels) hold the lion's share of the weights even though they have the fewest spatial positions. The attention blocks' $W_Q, W_K, W_V, W_O$ projections at 1280 channels are similarly heavy. The skinny $64 \times 64$ top stage, despite touching the most pixels, holds relatively few parameters. So: **most of SD 1.5's 860M parameters live in the deep, low-resolution, high-channel ResBlocks and attention blocks**, with the timestep MLP, the input/output convs, and the cross-attention text projections taking a few percent each.

**Compute (FLOP) distribution is the opposite story** — and this contrast is the single most useful thing to understand for optimization. FLOPs for a conv scale as (parameters) × (spatial positions). The top $64 \times 64$ stage has few parameters but $4096$ positions; the bottom $8 \times 8$ stage has many parameters but only $64$ positions. These partly cancel, so the FLOP budget is spread more evenly across resolutions than the parameter budget — and the *attention* FLOPs at the mid resolutions ($32 \times 32$ with its $N^2 = 10^6$ scores) are a large, sometimes dominant, slice. The practical upshot: the parameters you would quantize to save *memory* (the deep wide blocks) are not the same as the operations you would optimize to save *latency* (the attention matmuls and the high-resolution convs). This is why diffusion-acceleration work splits into "shrink the weights" (quantization, covered in the [edge-AI track](/blog/machine-learning/edge-ai/)) and "cut the attention/feature compute" (linear attention, feature caching) — two different levers on two different parts of the budget.

#### Worked example: estimating one SD 1.5 forward pass

Order-of-magnitude, on an RTX 4090 (24 GB). One U-Net forward pass on a $64 \times 64 \times 4$ latent is on the order of a few hundred GFLOPs (the full SD 1.5 U-Net is roughly 0.5–0.9 TFLOPs per forward depending on how you count, and reports vary by measurement method, so treat this as approximate). A 4090 does on the order of 100+ fp16 TFLOP/s usefully, so a single forward is a handful of milliseconds in principle; in practice, with 25–50 sampling steps and classifier-free guidance doubling the batch (conditional + unconditional), a 512×512 SD 1.5 image lands around **1–3 seconds** on a 4090, dominated by the sequential step count, not any single forward. The lesson the rest of the series acts on: the U-Net forward is fast; *running it 50 times in sequence* is the cost, which is why fast samplers (DPM-Solver, [DDIM](/blog/machine-learning/image-generation/ddim-and-fast-deterministic-sampling)) and distillation (LCM, Turbo) attack the *step count*, while quantization attacks the *per-step* cost. Different levers, same goal.

Here is the parameter budget laid out by component, to make "where the 860M sit" concrete (these are approximate proportions for the SD 1.5 U-Net, not exact counts):

| Component | Where it lives | Rough share | Why |
|---|---|---|---|
| ResBlock convolutions | all stages, heaviest at 1280 ch | ~45–55% | conv params scale as channels² |
| Self + cross attention proj. | 32/16/8 stages | ~30–40% | Q/K/V/out at wide channels |
| Decoder skip-fattened convs | up stages | included above | concatenated input doubles in-channels |
| Timestep MLP + per-block proj. | global | ~2–4% | small linear layers, cheap |
| Input stem + output head | top resolution | ~1% | only 4↔320 channels |

The table makes the optimization map obvious: if you want to *shrink the model*, you target the conv and attention weights in the deep wide stages (quantize them); if you want to *speed up a step*, you target the attention matmuls at the mid resolutions (fuse them, linearize them, or cache them across steps). They are different parts of the network, which is the whole reason the speed track has multiple independent chapters.

## A problem-solving narrative: debugging a U-Net that produces gray mush

Let me reason through a concrete engineering failure end to end, the way you actually would at 2am, because the architecture only becomes *yours* once you have debugged it. The scenario: you trained a small diffusion U-Net from scratch on a dataset of faces, and at sampling time it produces **gray, low-contrast mush** — vaguely face-shaped blobs with no detail and washed-out color. Walk the stack.

**Hypothesis 1: the skip connections are broken.** Gray, detail-free output is the *exact* signature of an autoencoder bottleneck with no working skips — the global plan survives (face-shaped) but all high-frequency detail is lost (mush). First thing I check: are the encoder features actually reaching the decoder? A common bug is mismatched channel counts at the concatenation (you changed `block_out_channels` but the skip stack still has the old shapes) silently zero-padded, or skips popped in the wrong order. The test is cheap: log the norm of each skip tensor as it is concatenated; if a skip's contribution is near zero after the up-stage conv, the decoder is ignoring it. **If the skips are dead, you get mush.** This is the most likely culprit and the first to rule out.

**Hypothesis 2: the timestep is not actually conditioning the network.** If the AdaGN projection is mis-wired — say `time_proj` outputs the wrong dimension, or the `scale`/`shift` are added at the wrong place, or `t_emb` is all zeros because the sinusoidal embedding overflowed — then the network behaves *identically at every noise level*. The symptom: it cannot tell "almost clean, just sharpen" from "almost noise, hallucinate structure," so it hedges toward the mean of all noise levels, which is a blurry average. Test: feed two very different timesteps and confirm the AdaGN `scale`/`shift` differ; visualize the predicted $\hat\epsilon$ at $t$ near 0 vs near $T$ and confirm they look different. **Constant output across timesteps means the time embedding is dead.**

**Hypothesis 3: attention is silently disabled or NaN-ing.** If the attention blocks produce NaNs (a too-large `scale`, fp16 overflow in the softmax) and you have a `nan_to_num` somewhere swallowing them, the network loses its global-coherence mechanism and you get locally-okay, globally-incoherent output — often a specific kind of repetitive, structureless texture. Test: hook the attention block outputs and check for NaN/Inf; switch the softmax to fp32 (`scaled_dot_product_attention` does this internally, which is one more reason to use it).

**Now the stress tests** — the "what happens when" probes that pressure-test the architecture's limits, which is exactly the discipline the rest of the series applies to samplers and guidance:

- **What happens at the resolution boundary — too few attention stages?** If you train a U-Net with attention *only* at the $8 \times 8$ bottleneck (to save memory), global coherence degrades at the mid scales: you get the right overall layout but mismatched paired features (one eye blue, one brown; ears at different heights), because the $16 \times 16$ and $32 \times 32$ stages have no mechanism to enforce mid-scale consistency. This is the concrete cost of under-provisioning attention, and it is why ADM *added* attention scales.
- **What happens when you push attention to full resolution to "fix" detail?** You OOM (the $4096^2$ wall from the worked example) — and even if you had the memory, it would barely help, because full-resolution detail is a *local* problem convolution already handles. Spending attention there is the wrong lever; the detail problem is a skip-connection problem, not an attention problem.
- **What happens when GroupNorm's group count does not divide the channels?** A hard crash — `num_channels` must be divisible by `num_groups` (32). A subtle version: too few groups (e.g. 1, which is LayerNorm-like) or too many (per-channel, InstanceNorm-like) changes the normalization statistics and can destabilize training. 32 groups is the diffusion default for a reason; deviate deliberately, not accidentally.
- **What happens when you fine-tune only the cross-attention layers?** This is actually a *useful* trick, not a failure: because cross-attention is the *only* place text touches the network, fine-tuning just the cross-attention $W_K, W_V$ (or adding LoRA there) is a parameter-efficient way to teach the model new concepts or styles — the same insight that makes attention-only LoRA effective. Knowing *which* layers carry *which* responsibility, which is the whole point of this anatomy, is what makes such targeted fine-tuning possible.

The throughline of the debugging: every failure mode maps to a *specific component* — mush to skips, time-invariance to AdaGN, incoherence to attention, crashes to GroupNorm shapes. That is the practical payoff of taking the U-Net apart. You cannot debug a black box; you can debug an anatomy.

## Reading and building the real thing in 🤗 diffusers

Enough theory — let me connect every concept above to the actual `UNet2DConditionModel` you will use in practice. Here is a config walkthrough that maps each field to a section of this post.

```python
from diffusers import UNet2DConditionModel

# This is (close to) the Stable Diffusion 1.5 U-Net config.
unet = UNet2DConditionModel(
    sample_size=64,                 # latent spatial size (64x64 -> 512x512 image after VAE)
    in_channels=4,                  # VAE latent channels (noisy input)
    out_channels=4,                 # predicted noise eps, same shape as input
    layers_per_block=2,             # ResBlocks per resolution stage
    block_out_channels=(320, 640, 1280, 1280),   # channels per stage: the 320->1280 ramp
    down_block_types=(
        "CrossAttnDownBlock2D",     # 64x64 ... (attention CONFIGURED but see attention_head_dim)
        "CrossAttnDownBlock2D",     # 32x32  <- attention here
        "CrossAttnDownBlock2D",     # 16x16  <- attention here
        "DownBlock2D",              # 8x8 bottleneck approach: plain ResBlocks
    ),
    up_block_types=(
        "UpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
    ),
    cross_attention_dim=768,        # text encoder width: CLIP ViT-L/14 = 768
    attention_head_dim=8,           # channels per attention head
)

print(f"U-Net parameters: {sum(p.numel() for p in unet.parameters())/1e6:.0f}M")
# -> ~860M for SD 1.5
```

Map it to the post: `block_out_channels=(320, 640, 1280, 1280)` is the channel ramp from the stage table; `layers_per_block=2` is two ResBlocks per stage; `CrossAttnDownBlock2D` vs `DownBlock2D` is literally the "attention here / no attention here" gating we derived from the $N^2$ cost; `cross_attention_dim=768` is the CLIP text width that becomes the keys/values in our `CrossAttention` module; `in_channels=out_channels=4` is the VAE latent's 4 channels in and the predicted noise out. The `attention_head_dim` controls heads-vs-width exactly as ADM discussed. Running this config and counting parameters returns roughly 860M — the number we have been anchoring to.

Now drive the whole thing end to end, the way you actually would, to see the U-Net in the context of the full diffusion stack (text encoder → U-Net → VAE):

```python
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")
# swap to a fast solver so we need ~25 steps, not 1000 (see the samplers deep-dive)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()  # fit comfortably on a consumer GPU

image = pipe(
    "a tabby cat wearing a tiny wizard hat, studio photo",
    num_inference_steps=25,
    guidance_scale=7.5,            # classifier-free guidance: drives cross-attention conditioning
).images[0]
image.save("wizard_cat.png")
```

Everything in this post is what happens *inside* `pipe(...)`: the prompt is CLIP-encoded to the $77 \times 768$ `context`; for each of the 25 steps the scheduler hands the U-Net a noisy latent and a timestep; the U-Net runs the encoder → bottleneck → decoder with skips, with AdaGN injecting the timestep into every ResBlock and cross-attention injecting the text at the $32/16/8$ stages; `guidance_scale=7.5` means each step actually runs the U-Net twice (conditional and unconditional) and extrapolates. The VAE then decodes the final $64 \times 64 \times 4$ latent to a $512 \times 512$ image. If you want to *watch* the U-Net's intermediate predictions, you can register a forward hook on `pipe.unet` and inspect the predicted noise at each step — the global composition stabilizes in the first handful of steps (the bottleneck's work) and the texture sharpens in the last few (the skip pathway's work), exactly as the architecture predicts.

A minimal manual denoising loop makes the U-Net's role unmissable — this is the inner loop the pipeline wraps:

```python
import torch

latents = torch.randn(1, 4, 64, 64, device="cuda", dtype=torch.float16)
latents = latents * pipe.scheduler.init_noise_sigma
prompt_embeds = pipe._encode_prompt("a tabby cat in a wizard hat", "cuda", 1, True)  # cond+uncond

pipe.scheduler.set_timesteps(25, device="cuda")
for t in pipe.scheduler.timesteps:
    inp = torch.cat([latents] * 2)                       # CFG: duplicate for cond/uncond
    inp = pipe.scheduler.scale_model_input(inp, t)
    noise_pred = pipe.unet(inp, t, encoder_hidden_states=prompt_embeds).sample
    noise_uncond, noise_cond = noise_pred.chunk(2)       # split the two passes
    noise_pred = noise_uncond + 7.5 * (noise_cond - noise_uncond)   # CFG extrapolation
    latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample
```

The single line `pipe.unet(inp, t, encoder_hidden_states=prompt_embeds)` is the entire subject of this post — the time-conditioned, text-cross-attended, multi-resolution U-Net, called once per step. Everything else is scheduling and guidance bookkeeping.

## Case studies: real numbers from the U-Net era

Concrete, cited numbers anchor the architecture in reality. Four data points that matter.

**ADM on ImageNet 256 (Dhariwal & Nichol, 2021).** The redesigned U-Net with classifier guidance hit **FID ≈ 4.59** on ImageNet 256×256, beating BigGAN-deep (≈ 6.95) — the result that named the paper. The architectural deltas that got it there are exactly the four ADM improvements above: more attention at more scales, AdaGN conditioning, BigGAN res-blocks, and more compute. This is the empirical proof that *U-Net design choices*, not just the diffusion framework, were decisive.

**Stable Diffusion 1.5 (Rombach et al., LDM, 2022).** The roughly **860M-parameter** U-Net operates in the VAE's latent space rather than pixel space — a $512 \times 512$ image becomes a $64 \times 64 \times 4$ latent, an $8 \times$ spatial / $48 \times$ element reduction, which is what makes attention at $32/16/8$ tractable and the whole model fit on consumer GPUs. The U-Net architecture is the same family as ADM; the move to latent space (the subject of the [latent diffusion post](/blog/machine-learning/image-generation/latent-diffusion-and-stable-diffusion)) is what made it ship. SD 1.5 uses CLIP ViT-L/14 text encoding (768-dim, hence `cross_attention_dim=768`).

**SDXL (Podell et al., 2023) — the U-Net at its peak.** SDXL scaled the U-Net to about **2.6B parameters** — roughly 3× SD 1.5 — by widening the channels and, notably, *rebalancing where the transformer (attention) blocks sit*: SDXL removed attention at the highest feature resolution and put many more transformer blocks at the lower resolutions (a "heterogeneous" distribution), plus used two text encoders (CLIP ViT-L + OpenCLIP ViT-bigG, concatenated). This is the resolution-cost argument taken to its logical end: spend your attention budget where tokens are few. SDXL is roughly the largest a *pure U-Net* text-to-image model got before the field pivoted to transformers.

**The limitation that ended the U-Net's reign.** Three pressures pushed the field from U-Net to [DiT](/blog/machine-learning/image-generation/diffusion-transformers-dit) and the MM-DiT recipe behind SD3/FLUX:

1. **The convolution inductive bias is a ceiling as well as a floor.** Convolutions assume locality and translation equivariance — great priors at small scale and small data, but at large scale they *limit* the model. A transformer with global attention at every layer has a weaker prior and, given enough data and compute, learns a better one. The U-Net's bias helps early and hurts late.
2. **Fixed resolution stages are rigid.** The U-Net's down/up ladder bakes in a specific set of resolutions and a specific multi-scale processing pattern. A patchified transformer treats the image as a flat sequence of tokens and is far more flexible about resolution and aspect ratio.
3. **Scaling is messier than transformers'.** DiT (Peebles & Xie, 2023) showed a *clean* power-law: FID falls predictably as you add transformer compute (measured in Gflops), with no architectural surgery — just make the transformer bigger. The U-Net's heterogeneous mix of convs, attention, and resampling at different resolutions does not scale with anything like the same clean law, so as labs poured compute in, the transformer's predictability won.

![A timeline of the denoiser backbone from DDPM through ADM and Stable Diffusion to the transformer that scales more cleanly](/imgs/blogs/the-diffusion-unet-8.png)

Figure 8 is the arc: DDPM's single-attention U-Net (2020) → ADM's wider, AdaGN-conditioned U-Net that beat GANs (2021) → LDM/Stable Diffusion's latent-space U-Net that democratized it (2022) → SDXL's 2.6B U-Net peak (2023) → DiT and the transformer backbones that scale more cleanly (2023 onward). The U-Net did not *fail* — it powered the entire first wave of usable text-to-image and is still excellent for fine-tuning, LoRA, and consumer inference. It was *out-scaled*, which is a different and more interesting fate.

## When to reach for a U-Net (and when not to)

A decisive recommendation, because every architecture is a cost.

**Reach for the U-Net denoiser when:**

- **You are working in the Stable Diffusion 1.5 / SDXL ecosystem.** The overwhelming majority of community fine-tunes, LoRAs, ControlNets, and IP-Adapters target U-Net models. If you want to use that ecosystem (and it is enormous), you want a U-Net, full stop. The inductive bias and the pretrained weights are right there.
- **Compute and data are modest.** The convolution prior is a genuine advantage at smaller scale — a U-Net reaches good quality with less data and compute than a transformer of equal capacity, because it does not have to *learn* locality and translation equivariance. For a from-scratch model on a modest dataset, the U-Net is often the pragmatic choice.
- **You are fine-tuning or serving on consumer hardware.** SD 1.5's 860M-parameter U-Net runs comfortably on a 24 GB (or even 8–12 GB with offload) GPU, and the entire LoRA/ControlNet toolchain assumes it. The [personalization](/blog/machine-learning/image-generation/personalization-dreambooth-textual-inversion-lora) and control posts in this series live in U-Net land.

**Do NOT reach for a U-Net when:**

- **You are training a frontier model from scratch with serious compute.** The 2024–2026 frontier (SD3, FLUX, and the rest of the [MM-DiT recipe](/blog/machine-learning/image-generation/mmdit-and-the-modern-text-to-image-recipe)) is transformer-based for the scaling reasons above. If you are spending H100-months, the clean DiT scaling law is worth the weaker prior.
- **You need flexible resolution / aspect ratio at scale.** The U-Net's fixed resolution ladder fights you; a patchified transformer is more natural.
- **You are betting on the unified-multimodal future.** Architectures that share a transformer backbone across text and image (the convergence the [2026 showdown post](/blog/machine-learning/image-generation/autoregressive-vs-diffusion-the-2026-showdown) covers) want a transformer, not a U-Net.

The honest summary: the U-Net is the *right* denoiser for the existing open ecosystem and for compute-constrained work, and the *wrong* one for frontier-scale from-scratch training. Both halves of that sentence are true at once, which is why the field is currently bilingual — U-Nets for the LoRA-and-ControlNet world, transformers for the frontier.

## Key takeaways

- **The U-Net's shape mirrors the denoising task:** dense same-resolution output (encoder-decoder), multi-scale image structure (a feature pyramid), and the need to keep fine detail (skip connections that route high-frequency content around the bottleneck).
- **Skip connections are the defining feature, not a detail.** The bottleneck carries semantics (what to draw); the skips carry texture (how to render it sharply). Remove them and the model blurs — this is provable from the data-processing inequality and confirmed by ablation.
- **The ResBlock is the workhorse:** GroupNorm (not BatchNorm — diffusion's per-sample random timesteps break batch statistics) + SiLU + Conv, twice, with a residual add and a timestep scale-shift in the middle.
- **The timestep enters via AdaGN/FiLM** — a per-channel `γ·GroupNorm(h)+β` whose `γ,β` are predicted from a sinusoidal-then-MLP timestep embedding. It is the most general per-channel linear modulation and the natural inverse of normalization; ADM showed it beats additive injection.
- **Self-attention is gated to low resolution because its cost is $O(N^2)$ in tokens.** At $64 \times 64$ (4096 tokens) the attention matrix is 256× larger than at $16 \times 16$ (256 tokens) — it does not fit. Put the global operator where global decisions are made and where it is cheap.
- **Text steers pixels through cross-attention,** where image features are the queries and frozen text embeddings are the keys/values — spatially selective in a way the global timestep modulation can never be. Its cost is linear in image tokens, so it is cheap.
- **ADM made the U-Net beat GANs** (ImageNet-256 FID ≈ 4.59) via more attention at more scales, AdaGN conditioning, BigGAN res-blocks, and more compute. Stable Diffusion 1.5's ≈ 860M parameters sit mostly in the deep, wide, low-resolution blocks; the FLOPs spread more evenly and lean on attention.
- **The U-Net was out-scaled, not beaten:** its convolution prior, fixed resolution ladder, and messier scaling law motivated the move to DiT, which scales with a clean power law. Reach for a U-Net in the SD1.5/SDXL ecosystem and at modest compute; reach for a transformer at the frontier.

## Further reading

- **Ho, Jain, Abbeel — "Denoising Diffusion Probabilistic Models" (2020).** The original DDPM U-Net and the noise-prediction objective.
- **Dhariwal & Nichol — "Diffusion Models Beat GANs on Image Synthesis" (2021).** The ADM backbone: the U-Net ablations (attention scales, AdaGN, BigGAN res-blocks) that produced the architecture everything inherits.
- **Ronneberger, Fischer, Brox — "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015).** The original U-Net and the skip-connection idea, born in segmentation.
- **Rombach, Blattmann, Lorenz, Esser, Ommer — "High-Resolution Image Synthesis with Latent Diffusion Models" (2022).** Stable Diffusion's latent-space U-Net with cross-attention conditioning.
- **Podell et al. — "SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis" (2023).** The 2.6B-parameter U-Net at its peak, with rebalanced attention and dual text encoders.
- **Saharia et al. — "Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding" (Imagen, 2022).** T5-XXL text encoding, the Efficient U-Net, and cascaded super-resolution.
- **Peebles & Xie — "Scalable Diffusion Models with Transformers" (DiT, 2023).** The clean transformer scaling law that ended the U-Net's reign — the subject of the next post.
- **🤗 `diffusers` docs — `UNet2DConditionModel`.** The config fields (`block_out_channels`, `down_block_types`, `cross_attention_dim`, `attention_head_dim`) mapped to everything above.
- **Within this series:** [diffusion from first principles](/blog/machine-learning/image-generation/diffusion-from-first-principles), [the math of DDPM](/blog/machine-learning/image-generation/the-math-of-ddpm), [classifier-free guidance](/blog/machine-learning/image-generation/classifier-free-guidance), [latent diffusion and Stable Diffusion](/blog/machine-learning/image-generation/latent-diffusion-and-stable-diffusion), [diffusion transformers (DiT)](/blog/machine-learning/image-generation/diffusion-transformers-dit), and the capstone [building an image-generation stack](/blog/machine-learning/image-generation/building-an-image-generation-stack).
