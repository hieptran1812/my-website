---
title: "Video Diffusion Transformers: Spacetime Patches and the Scaling Thesis"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Take the same diffusion transformer that paints one image, feed it a clip cut into spacetime patches, and you have the architecture under Sora, CogVideoX, HunyuanVideo, and Wan — here is exactly what is reused, what is new, and why scaling it buys coherence."
tags:
  [
    "video-generation",
    "diffusion-models",
    "video-diffusion",
    "text-to-video",
    "diffusion-transformer",
    "spacetime-patches",
    "scaling-laws",
    "generative-ai",
    "deep-learning",
    "pytorch",
    "diffusers",
  ]
category: "machine-learning"
subcategory: "Video Generation"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/video-diffusion-transformers-1.png"
---

You are staring at a profiler trace at two in the morning. The model is a diffusion transformer — the exact architecture that, on a single image, gives you a clean 1024×1024 frame in under a second on an A100. You have fed it a five-second clip instead of one image. The forward pass is now taking 0.9 seconds *per denoising step*, the attention kernel is eating 70% of the wall-clock, and peak VRAM has climbed from 11 GB to 61 GB. Nothing about the network changed. You did not add a layer, a head, or a parameter. You changed exactly one thing: the input is no longer a grid of image tokens, it is a *sequence of spacetime patches* — little cuboids that each span a slab of time and a patch of space — and there are roughly nine hundred times more of them.

That single change is the entire subject of this post, and it is the architectural idea that quietly underwrites every frontier video model shipped between 2024 and 2026. Sora calls them "patches." CogVideoX, HunyuanVideo, and Wan call them tokens. They are the same thing: you take the compressed latent that the [causal 3D-VAE](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression) produces — a tensor of shape $T' \times H' \times W' \times C$ — and you slice it across space *and time* into a flat sequence, then you run a plain transformer over that sequence as if it were a paragraph of text. The transformer does not know it is looking at video. It sees a long list of vectors with positions, and it denoises them. The genius is in how little had to change from the image case, and the cost is in that nine-hundred-times number.

![Stacked diagram of the spacetime diffusion transformer pipeline running from the three dimensional VAE latent through spacetime patchify into the transformer blocks and back out through unpatchify to a denoised latent](/imgs/blogs/video-diffusion-transformers-1.png)

By the end of this post you will be able to do four concrete things. First, you will be able to *count the tokens* — given a clip's frames, resolution, VAE compression, and patch size, compute the exact sequence length and the attention FLOPs that follow, which is the number that decides whether your render OOMs at second six. Second, you will be able to *write a video-DiT block in PyTorch* from scratch: patchify a latent video into spacetime tokens, modulate them with AdaLN from the timestep and text embedding, run attention and an MLP, and unpatchify back. Third, you will know exactly *how 3D positions work* — how a single RoPE rotation encodes $(t, h, w)$ so the transformer can tell frame 3 from frame 30 and the top-left patch from the bottom-right. Fourth, you will understand the *scaling thesis* — Sora's central claim that more compute on this architecture does not merely sharpen frames but buys *emergent coherence*: object permanence, consistent identity, a camera that moves like a camera. And you will be able to call CogVideoX in 🤗 `diffusers` and reason about what its transformer is doing under each flag.

This is the post where the [image diffusion transformer](/blog/machine-learning/image-generation/diffusion-transformers-dit) grows a time axis. If you have read [from image diffusion to video diffusion](/blog/machine-learning/video-generation/from-image-diffusion-to-video-diffusion), you know the three classic ways to add temporal modeling to a U-Net — 3D conv, $(2{+}1)\text{D}$ conv, temporal attention. The DiT recipe is the limit of that line of thought: stop bolting temporal modules onto a convolutional backbone and instead make the *whole network* a transformer over spacetime tokens, where time is just another axis the patches span. The attention pattern over those tokens — full 3D versus factorized — is the subject of the sibling post on [spatiotemporal attention patterns](/blog/machine-learning/video-generation/spatiotemporal-attention-patterns); here we build the container that pattern lives in. And the whole thing only works because the 3D-VAE already crushed the clip by a factor of a few thousand — without that compression the token count is not nine hundred times an image, it is tens of thousands of times, and nothing fits. The spine of the [whole series](/blog/machine-learning/video-generation/why-video-generation-is-hard) holds: video is spatial generation times temporal coherence under a brutal compute budget, and the spacetime-DiT is how you spend that budget on a transformer.

## 1. What a spacetime patch actually is

Let us be concrete, because "spacetime patch" sounds more mystical than it is. Start from the output of the VAE, the thing we are denoising. Our running example throughout this post is a five-second clip at 720p — $1280 \times 720$, 24 frames per second, so 120 frames. A typical causal 3D-VAE (CogVideoX's, HunyuanVideo's, Wan's are all in this range) compresses by $4\times$ in time and $8\times \times 8\times$ in space, with 16 latent channels. So the latent is

$$
T' = \frac{120}{4} = 30 \text{ latent frames}, \qquad H' = \frac{720}{8} = 90, \qquad W' = \frac{1280}{8} = 160, \qquad C = 16.
$$

That latent tensor has shape $30 \times 90 \times 160 \times 16$. It is, abstractly, a little 4D block of numbers. The denoiser's job is to take a noisy version of this block and predict the noise (or the velocity, under flow matching). A diffusion transformer cannot eat a 4D block; transformers eat *sequences*. So we have to flatten the block into a list of tokens, and the only question is how big a chunk of the block each token represents.

A *spacetime patch* is one such chunk: a small cuboid spanning $p_t$ latent frames by $p_h$ rows by $p_w$ columns, across all $C$ channels. CogVideoX uses $p_t = 1, p_h = p_w = 2$ — a $1 \times 2 \times 2$ patch, so it patchifies space but not time. HunyuanVideo and many others use $p_t = 1, p_h = p_w = 2$ as well at the transformer's input, having already done the temporal compression in the VAE. Sora's framing, and the conceptually cleanest version, patchifies all three axes — a $2 \times 2 \times 2$ patch genuinely spans time. To turn a patch into a token vector you take the $p_t \times p_h \times p_w \times C$ values inside it, flatten them to a vector of length $p_t \, p_h \, p_w \, C$, and apply a learned linear projection to the model dimension $d$. That is the patch embedding, and it is *exactly* the image-DiT patch embedder with one more axis added to the kernel.

![Grid showing a latent video block sliced into spacetime patches where each patch spans two latent frames and a two by two spatial region and the patches flatten into one token sequence](/imgs/blogs/video-diffusion-transformers-2.png)

The sequence length is then the number of patches. With a $1 \times 2 \times 2$ patch on our $30 \times 90 \times 160$ latent:

$$
L = \frac{T'}{p_t} \cdot \frac{H'}{p_h} \cdot \frac{W'}{p_w} = \frac{30}{1} \cdot \frac{90}{2} \cdot \frac{160}{2} = 30 \cdot 45 \cdot 80 = 108{,}000 \text{ tokens}.
$$

That is the number from the opening paragraph. One hundred and eight thousand tokens for a five-second clip. The same DiT on a single 720p image latent ($90 \times 160$, patched $2 \times 2$) sees $45 \times 80 = 3{,}600$ tokens. The video is exactly $30\times$ longer in token count — the temporal-compression factor — but self-attention is quadratic, so the *attention* cost scales as $L^2$: $(108{,}000 / 3{,}600)^2 = 30^2 = 900\times$. There is the nine hundred. The token count went up linearly with frames; the attention bill went up quadratically. This is not a detail; it is the central fact that organizes every design decision in a video DiT, and we will return to it constantly.

What does a token *mean*? It is a learned summary of a tiny clip of a region. Token number 4,217 might be "the patch covering rows 30–31, columns 60–61, at latent frames 12–13" — a little $2 \times 2$ window of a region over a slab of about 160 milliseconds of real time (two latent frames at $4\times$ temporal compression and 24 fps is $2 \times 4 / 24 \approx 0.33$ s, actually). The transformer's self-attention then lets that token look at every other token: the same region earlier and later (temporal coherence), neighboring regions in the same frames (spatial structure), and distant regions across time (a hand reaching for a ball on the far side of the frame). That is the payoff. By flattening space and time into one sequence and running full self-attention, *every* spacetime relationship is expressible by the same mechanism. There is no separate "temporal module." Time is just more tokens.

#### Worked example: the token tax of resolution, length, and patch size

Hold the architecture fixed and vary the input to see where tokens come from. Our base clip is 5 s, 720p, 24 fps, VAE compression $4 \times 8 \times 8$, patch $1\times2\times2$ — that is $L = 108{,}000$ tokens. Now:

- **Double the duration** to 10 s. Frames go $120 \to 240$, latent frames $30 \to 60$, so $L \to 216{,}000$. Token count doubles (linear in time); attention cost *quadruples* ($4\times$). This is why long video is a quadratic-cost problem, not a linear one — the subject of [autoregressive long-video rollout](/blog/machine-learning/video-generation/why-video-generation-is-hard).
- **Go to 1080p** ($1920 \times 1080$). Spatial latent $\to 240 \times 135$, patched $\to 120 \times 68 \approx 8{,}160$ spatial tokens per latent frame, $\times 30 = 244{,}800$ tokens. Attention cost relative to 720p: $(244{,}800/108{,}000)^2 \approx 5.1\times$. Resolution is *brutal* because it hits two spatial axes at once.
- **Coarsen the patch** to $1 \times 4 \times 4$. Spatial tokens drop $4\times$ to $22.5 \times 40 = 900$ per latent frame, $L \to 27{,}000$, attention cost drops $16\times$. The price is detail: a $4\times4$ latent patch is a $32 \times 32$ pixel block, and fine texture inside it is summarized into one token. This is the real knob — patch size is the cheapest lever you have on the token count, and it trades directly against spatial fidelity.

The lesson: the VAE sets the floor on tokens, the patch size sets the multiplier, and duration and resolution are the two axes that blow the budget. A video DiT is, at the input, a machine for managing this one number.

## 2. The video-DiT block: what is reused, what is new

Here is the claim that should reframe how you think about this entire architecture: **a video DiT is an image DiT with a longer sequence and a slightly bigger positional encoding, and almost nothing else changes.** If you internalize one thing from this post, make it that. Let us prove it by walking the block.

![Matrix table contrasting the image diffusion transformer and the video diffusion transformer across patching, attention, conditioning, positions, and loss, marking which parts are reused unchanged and which are genuinely new](/imgs/blogs/video-diffusion-transformers-4.png)

Recall the image-DiT block from [the DiT post](/blog/machine-learning/image-generation/diffusion-transformers-dit). It is a standard pre-norm transformer block with one twist: instead of vanilla LayerNorm, it uses **adaptive layer norm (AdaLN-Zero)**, where the scale and shift parameters of the norm are not learned constants but are *predicted from the conditioning* (the diffusion timestep, and a pooled text embedding). The block does, in order: AdaLN-modulate the tokens, self-attention, residual add (gated by a predicted scale), AdaLN-modulate again, MLP, residual add (gated again). That is the whole block. Stack $N$ of them, sandwich between a patch embedder and an unpatchify head, and you have DiT.

Now the video version. The block math is *identical*. AdaLN works the same way — it takes the timestep embedding and a text embedding and produces per-block scale-and-shift vectors that modulate the normalized tokens. The MLP is the same. The residual gating is the same. The diffusion loss (or [flow-matching](/blog/machine-learning/image-generation/flow-matching-and-rectified-flow) loss) is the same per-voxel objective we inherit from the image series — see [from image to video diffusion](/blog/machine-learning/video-generation/from-image-diffusion-to-video-diffusion) for why the loss does not change at all when $x_0$ becomes a clip. Three things, and only three things, are new:

1. **Patchify spans time.** The patch embedder's kernel grows a temporal dimension ($1\times2\times2$ or $2\times2\times2$ instead of $2\times2$). One extra axis.
2. **Self-attention runs over spacetime tokens.** The attention is mechanically unchanged — it is still $\text{softmax}(QK^\top/\sqrt{d})V$ — but $Q, K, V$ now have $L \approx 108{,}000$ rows instead of $\approx 1{,}024$. Whether that attention is *full* (every token attends to every token) or *factorized* (spatial attention then temporal attention, cheaper) is the design choice covered in [spatiotemporal attention patterns](/blog/machine-learning/video-generation/spatiotemporal-attention-patterns). The block container is agnostic to which you use.
3. **Positional encoding has three axes.** The token at sequence position $i$ needs to know its $(t, h, w)$ coordinates, not just a 1D index, so the position encoding is 3D. We will derive this in §4.

That is the complete list of differences. Everything else — and "everything else" is most of the parameter count and most of the engineering — is shipped verbatim from the image DiT. This is *why* the field converged so fast: the moment Peebles and Xie showed in 2023 that a transformer beats a U-Net for image diffusion and scales cleanly, the path to video was obvious. Add a time axis to the patches, make attention span it, fix the positions, and scale. Sora, less than a year later, was the demonstration that this path went much further than anyone expected.

![Graph of a single video diffusion transformer block showing the conditioning branch where timestep and text are turned into adaptive layer norm scale and shift that modulate the tokens before spacetime attention and the MLP](/imgs/blogs/video-diffusion-transformers-3.png)

Look at the conditioning branch in the figure, because it is the part people get wrong. The timestep $t$ and the text embedding do *not* enter the sequence as tokens (that would add to the already-painful $L$). They enter through AdaLN: a small MLP maps $(\text{embed}(t) + \text{embed}(\text{text}))$ to six vectors per block — scale and shift for the attention norm, scale and shift for the MLP norm, and two residual gates. These modulate *every* token identically within a block. This is enormously cheaper than prepending text tokens and cross-attending, and it is why DiT conditioning is so clean. Some video models (CogVideoX in its "expert" variant, and MMDiT-style models) *do* add text tokens and let them attend jointly with the video tokens — that is the [MMDiT recipe](/blog/machine-learning/image-generation/mmdit-and-the-modern-text-to-image-recipe) generalized to video — but the timestep almost always goes through AdaLN. The distinction matters for cost: AdaLN conditioning is free in sequence length; joint text tokens add a few hundred tokens (negligible against 108k) but enable richer text-video binding.

Let me write the block, because reading it settles any remaining ambiguity. This is a self-contained, runnable video-DiT block with AdaLN conditioning and full spacetime attention.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def modulate(x, shift, scale):
    # AdaLN: per-token affine where shift/scale come from the conditioning
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class VideoDiTBlock(nn.Module):
    """One transformer block with AdaLN-Zero conditioning and full
    spacetime self-attention. This is the image-DiT block, unchanged,
    operating on a longer (spacetime) token sequence."""

    def __init__(self, dim, n_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, dim)
        )
        # AdaLN-Zero: predict 6 modulation vectors from the conditioning.
        # Zero-init so each block starts as the identity (stable training).
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, cond, rope=None):
        # x:    (B, L, dim)   spacetime tokens
        # cond: (B, dim)      timestep + text embedding
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = (
            self.ada(cond).chunk(6, dim=1)
        )
        # --- attention sublayer ---
        h = modulate(self.norm1(x), shift_a, scale_a)
        if rope is not None:
            h = rope(h)  # apply 3D RoPE to the (already-projected) tokens
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_a.unsqueeze(1) * attn_out
        # --- MLP sublayer ---
        h = modulate(self.norm2(x), shift_m, scale_m)
        x = x + gate_m.unsqueeze(1) * self.mlp(h)
        return x
```

Read that against the image-DiT block and the diff is: the tokens `x` happen to be spacetime patches, and there is an optional `rope` hook for 3D positions. The conditioning path, the AdaLN-Zero modulation, the gated residuals, the MLP — all identical. That is the point. (In production you would replace `nn.MultiheadAttention` with `F.scaled_dot_product_attention` or FlashAttention for the memory savings that 108k tokens demand, and you would likely apply RoPE to $Q$ and $K$ inside the attention rather than to the token stream — the listing keeps it simple to read.)

## 3. Patchify and unpatchify, in code

The two ends of the network — the patch embedder and the unpatchify head — are where the 4D latent meets the 1D sequence, and they are worth writing explicitly because a sign error here produces video that is spatially scrambled in a way that looks like a bug in attention but is not. Here is a complete patchify/unpatchify pair for a $1\times2\times2$ spacetime patch.

```python
class SpacetimePatchify(nn.Module):
    """Latent video (B, C, T, H, W) -> token sequence (B, L, dim)."""

    def __init__(self, in_ch=16, dim=1152, patch=(1, 2, 2)):
        super().__init__()
        self.patch = patch
        pt, ph, pw = patch
        # A 3D conv with stride=kernel IS the spacetime patch embedder.
        self.proj = nn.Conv3d(in_ch, dim, kernel_size=patch, stride=patch)

    def forward(self, z):
        # z: (B, C, T, H, W)
        z = self.proj(z)                      # (B, dim, T/pt, H/ph, W/pw)
        B, dim, t, h, w = z.shape
        z = rearrange(z, "b d t h w -> b (t h w) d")  # (B, L, dim)
        return z, (t, h, w)


class Unpatchify(nn.Module):
    """Token sequence (B, L, dim) -> latent video (B, C, T, H, W)."""

    def __init__(self, out_ch=16, dim=1152, patch=(1, 2, 2)):
        super().__init__()
        self.patch = patch
        pt, ph, pw = patch
        self.proj = nn.Linear(dim, out_ch * pt * ph * pw)
        self.out_ch = out_ch

    def forward(self, x, grid):
        t, h, w = grid
        pt, ph, pw = self.patch
        x = self.proj(x)  # (B, L, out_ch*pt*ph*pw)
        x = rearrange(
            x, "b (t h w) (c pt ph pw) -> b c (t pt) (h ph) (w pw)",
            t=t, h=h, w=w, c=self.out_ch, pt=pt, ph=ph, pw=pw,
        )
        return x  # (B, out_ch, T, H, W)
```

Two things to notice. First, the patch embedder is *just a strided 3D convolution* — kernel size equals stride equals the patch shape — which is the cleanest way to implement "cut into non-overlapping cuboids and linearly project each." This is the direct generalization of the image-DiT patch embedder, which is a strided 2D conv. Second, the unpatchify is the exact inverse: project each token back to $C \cdot p_t \cdot p_h \cdot p_w$ values and re-tile them into the latent grid. The `rearrange` pattern is the one place to be careful — the order of the patch dimensions in the einops string must match between patchify and unpatchify or you get a spatially permuted output.

Now wire the full model. This is a minimal but complete spacetime-DiT denoiser: embed timestep and text, patchify, run blocks, unpatchify.

```python
class VideoDiT(nn.Module):
    def __init__(self, in_ch=16, dim=1152, depth=28, n_heads=16, patch=(1, 2, 2)):
        super().__init__()
        self.patchify = SpacetimePatchify(in_ch, dim, patch)
        self.t_embed = nn.Sequential(
            nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.text_proj = nn.Linear(4096, dim)  # e.g. T5-XXL pooled
        self.blocks = nn.ModuleList(
            [VideoDiTBlock(dim, n_heads) for _ in range(depth)]
        )
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.unpatchify = Unpatchify(in_ch, dim, patch)

    def forward(self, z_noisy, t, text_emb):
        # z_noisy: (B, C, T, H, W); t: (B,); text_emb: (B, 4096)
        x, grid = self.patchify(z_noisy)                  # (B, L, dim)
        cond = self.t_embed(timestep_embedding(t, 256)) + self.text_proj(text_emb)
        for blk in self.blocks:
            x = blk(x, cond)                              # rope omitted for brevity
        x = self.norm_out(x)
        return self.unpatchify(x, grid)                   # predicted noise/velocity


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -torch.log(torch.tensor(max_period)) * torch.arange(half, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
```

That is a working video diffusion transformer in under a hundred lines. With `dim=1152, depth=28, n_heads=16` it is roughly DiT-XL's shape — about 675M parameters of transformer — and it is, modulo the 3D patch conv and the (omitted) 3D RoPE, byte-for-byte the image DiT-XL. CogVideoX-2B is this design at $\approx 2$B; CogVideoX-5B and HunyuanVideo's $\approx 13$B and Wan's $\approx 14$B are the same recipe scaled up in width and depth. You have just written the skeleton of the open video frontier.

#### Worked example: counting the parameters and where they live

Take the `dim=1152, depth=28` model. Per block: the attention has $4 \times d^2 = 4 \times 1152^2 \approx 5.3$M parameters ($Q,K,V,O$ projections), the MLP has $2 \times 4 d^2 = 8 \times 1152^2 \approx 10.6$M, and the AdaLN MLP has $6 d^2 \approx 8.0$M. So $\approx 24$M per block $\times 28 \approx 670$M, plus small embedders — about 675M total, exactly DiT-XL. Now notice the breakdown: the AdaLN conditioning is a *third* of each block's parameters. Conditioning is not a free add-on; it is a substantial fraction of the model. And critically: *not one of these parameters knows it is processing video.* The temporal modeling lives entirely in (a) the patch conv's time dimension and (b) the fact that the attention sequence happens to contain tokens from different frames. Scale this to 14B and you have Wan; the per-parameter story is unchanged.

There is one subtlety in AdaLN worth a paragraph, because it is the mechanism that makes a video DiT *trainable* at all at this scale. The "Zero" in AdaLN-Zero means the final linear layer of each block's conditioning MLP is initialized to zero, so at step zero every block's scale-and-shift output is zero and every residual gate is zero. The consequence is that the network starts as the *identity function* — each block passes its input straight through untouched. This matters enormously for a 108k-token sequence: a deep transformer that starts as a random nonlinear map over a sequence that long has gradients that explode or vanish before the loss can find signal, and you burn days of compute fighting instability. Starting from the identity, the network gently learns to deviate, block by block, and training is stable from the first step. This trick is inherited verbatim from the image DiT — Peebles and Xie found AdaLN-Zero was the single most important architectural choice for stable, scalable diffusion-transformer training — and it transfers to video without a single change. It is a small example of the larger thesis: the things that make the image DiT scale are exactly the things that make the video DiT scale, because they are the same network.

Why does AdaLN, specifically, beat the alternatives for diffusion conditioning? The competing options are (a) concatenating the conditioning to the input, (b) adding it as bias, and (c) cross-attention to conditioning tokens. AdaLN dominates (a) and (b) empirically because it lets the conditioning *gate* the network multiplicatively at every block — the timestep can tell layer 14 to "barely move" (small scale) when the input is nearly clean, and "move aggressively" (large scale) when it is nearly pure noise, per-block and per-channel. That is exactly the kind of timestep-dependent behavior diffusion needs, since the denoiser's job changes character across the noise schedule (coarse structure early, fine detail late). And AdaLN costs *zero* sequence length, which in the video regime — where every token is precious — is decisive. Cross-attention (c) is richer but adds tokens and FLOPs; the modern compromise, used by HunyuanVideo and MMDiT-style video models, is *both*: AdaLN for the timestep (free) and joint/cross attention for the text (richer binding), which is why those models bind prompts more tightly than a pure-AdaLN model at the cost of some extra compute.

It is worth pausing on *why coherence is a property of the joint distribution* and what that means for the architecture, because it is the deepest version of "video is hard." A real video lives on a thin manifold: given frame $t$, frame $t{+}1$ is almost determined — it is frame $t$ plus a small, physically-plausible motion. The probability distribution $p(x_{1:T})$ over clips is therefore wildly non-factorial; it is *not* the product $\prod_t p(x_t)$ of per-frame marginals. A model that scored perfectly on every per-frame marginal — every frame individually photorealistic — could still place almost all its probability mass off the video manifold, on sequences of beautiful but *incompatible* frames. That is the flickering slideshow, stated measure-theoretically. The denoising loss, as we noted, is a weighted score-matching objective for the *clip* distribution $p(x_{1:T})$, and the score of that distribution has enormous components pointing toward inter-frame consistency (the directions that pull an incoherent sample back onto the manifold). The spacetime-DiT's full self-attention is what lets the network *represent* those components — every token can see every other token, so the model can express "this patch at frame 30 must be consistent with that patch at frame 3." Take the attention away (process frames independently) and the network literally cannot represent the off-diagonal of the score, so it cannot learn coherence no matter how long you train. This is the precise, provable reason the architecture — not the loss — is where video's difficulty lives, and it is why §7's scaling thesis lands on *coherence*: the cross-frame score components are the high-dimensional, data-hungry part of the target, so they are exactly what more compute and more data improve.

## 4. Three-dimensional positions: factorized RoPE over (t, h, w)

A transformer is permutation-invariant — shuffle the tokens and self-attention gives the same answer up to the same shuffle. That is catastrophic for video: the model must know that token A is at frame 3 and token B is at frame 30, that one patch is top-left and another bottom-right. The fix is positional encoding, and the modern choice — used by HunyuanVideo, Wan, and most 2024–2026 models — is **3D rotary position embedding (RoPE)**.

Let me build it from the image case. In [the DiT lineage](/blog/machine-learning/image-generation/diffusion-transformers-dit), 2D RoPE encodes a token's $(h, w)$ position by *rotating* its query and key vectors by an angle proportional to position. The beautiful property of RoPE is that the dot product $q_i \cdot k_j$ after rotation depends only on the *relative* position $i - j$, so the model learns relative spatial relationships that extrapolate to resolutions it never saw. We want the same property in three dimensions.

The factorized recipe is simple: split the head dimension $d_h$ into three equal slices, one per axis, and apply a standard 1D RoPE to each slice with that axis's coordinate.

![Matrix laying out factorized three dimensional rotary position embedding where the head dimension is split into thirds for time, height, and width, each rotated by its own coordinate and frequency band](/imgs/blogs/video-diffusion-transformers-6.png)

Formally, a token at spacetime position $(t, h, w)$ has its query vector $q \in \mathbb{R}^{d_h}$ partitioned as $q = [q^{(t)}; q^{(h)}; q^{(w)}]$ with each piece of size $d_h/3$. Each piece is rotated by the rotation matrix $R$ for its coordinate:

$$
\tilde{q} = \big[\, R_{\Theta_t}(t)\, q^{(t)} \;;\; R_{\Theta_h}(h)\, q^{(h)} \;;\; R_{\Theta_w}(w)\, q^{(w)} \,\big],
$$

where $R_{\Theta}(p)$ is the standard RoPE block-diagonal rotation: for the $k$-th pair of channels it rotates by angle $p \cdot \theta_k$ with $\theta_k = b^{-2k/(d_h/3)}$ and base $b$ (often 10000). The key $k_j$ is rotated identically. Then the attention score between token $i$ and token $j$ is

$$
\tilde{q}_i \cdot \tilde{k}_j = \sum_{a \in \{t,h,w\}} q_i^{(a)} \cdot R_{\Theta_a}(p_j^{(a)} - p_i^{(a)}) \, k_j^{(a)},
$$

which depends only on the *relative* offsets $\Delta t, \Delta h, \Delta w$. That is the whole point: the model sees "this token is 5 frames ahead and 2 rows down from that one," not absolute indices, so it generalizes across clip lengths and resolutions it never trained on. This is *exactly* why a model trained on 720p can often do a passable 1080p, and why the temporal RoPE makes variable-length generation natural.

Here is a compact 3D RoPE implementation.

```python
def build_3d_rope(grid, dim_head, base=10000.0, device="cuda"):
    """Return cos/sin tables of shape (L, dim_head) for factorized 3D RoPE.
    grid = (t, h, w) in token (post-patchify) units."""
    t, h, w = grid
    d3 = dim_head // 3            # channels per axis (must be even)
    d3 -= d3 % 2

    def axis_freqs(length, d):
        theta = base ** (-torch.arange(0, d, 2, device=device).float() / d)
        pos = torch.arange(length, device=device).float()
        ang = torch.outer(pos, theta)            # (length, d/2)
        return ang

    # coordinates of every token, in (t, h, w) order matching patchify flatten
    ct = torch.arange(t, device=device).view(t, 1, 1).expand(t, h, w).reshape(-1)
    ch = torch.arange(h, device=device).view(1, h, 1).expand(t, h, w).reshape(-1)
    cw = torch.arange(w, device=device).view(1, 1, w).expand(t, h, w).reshape(-1)

    theta = base ** (-torch.arange(0, d3, 2, device=device).float() / d3)
    angles = torch.cat([
        torch.outer(ct.float(), theta),   # time slice
        torch.outer(ch.float(), theta),   # height slice
        torch.outer(cw.float(), theta),   # width slice
    ], dim=-1)                            # (L, 3*d3/2)
    cos = torch.cos(angles).repeat_interleave(2, dim=-1)
    sin = torch.sin(angles).repeat_interleave(2, dim=-1)
    return cos, sin                       # apply to q, k by rotate-half


def apply_rope(x, cos, sin):
    # x: (B, n_heads, L, dim_head)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rot = torch.stack([-x2, x1], dim=-1).flatten(-2)
    return x * cos + rot * sin
```

The crucial detail is that the coordinate tensors `ct, ch, cw` must be built in the *same flatten order* as patchify (`(t h w)` here). Get that order wrong and the model learns positions that do not match the tokens — it trains, slowly, to terrible quality, and the bug is invisible in the loss curve. Ask me how I know.

Why factorized RoPE and not a learned 3D position table? Two reasons, both about the variable-input flexibility that is the next section's subject. A learned absolute table has a fixed maximum $(T, H, W)$ baked in at training time; you cannot generate a longer or larger clip than the table supports without interpolation hacks. RoPE has no table — it is a closed-form rotation defined for any coordinate — so the model trained on 30 latent frames can be *asked* for 60 (with quality caveats) just by extending the position range. That extrapolation is shaky in practice (long-range position generalization is genuinely hard), but it is *possible*, which absolute embeddings make awkward. The second reason is the relative-position property: video has strong relative structure (motion is about frame-to-frame *offsets*), and RoPE encodes offsets natively.

## 5. The flexibility dividend: variable duration, resolution, and aspect

Here is a property of the spacetime-patch design that is easy to miss and turns out to be one of the most consequential things about it: **a transformer over patches does not care how many patches you give it.** The same weights process a sequence of 3,600 tokens (one image), 27,000 tokens (a short low-res clip), or 216,000 tokens (a long high-res clip). There is no architectural commitment to a fixed input shape. This is the property Sora's report leans on when it describes training on "videos and images of variable durations, resolutions and aspect ratios," and it is inherited directly from how transformers and patches compose.

Contrast this with a U-Net. A convolutional U-Net is not shape-agnostic in the same easy way: its downsampling stack assumes spatial dimensions divisible by $2^{\text{depth}}$, and while it tolerates a range of resolutions, training it on wildly mixed aspect ratios and durations is awkward — the batched tensors have to share a shape. A patch transformer sidesteps this entirely. You patchify each sample to *its own* token count and either (a) train one sample at a time, or (b) pack multiple variable-length samples into one batch with attention masks that prevent cross-sample attention. That packing trick is the NaViT idea (Native Resolution ViT) carried into video: instead of resizing every clip to a fixed grid and wasting compute on padding, you pack patches from clips of different shapes into one long sequence and mask the attention so each clip only attends to itself.

The payoff is not just convenience; it is *data efficiency and capability*. Training on native aspect ratios means the model sees vertical phone video as vertical, not letterboxed into a square — so it generates correctly-framed vertical video at inference. Training on mixed durations means the model learns short and long dynamics from the same weights, and you can ask for a 3-second or an 8-second clip from one checkpoint by simply choosing the token count. Sora's report explicitly credits this: sampling at native aspect ratios improves composition and framing. The flexibility falls out of "video is just a token sequence" for free.

```python
# Variable duration is just a different num_frames -> different token count.
# Same weights, no architecture change.
for num_frames in (13, 25, 49):           # short, medium, longer clips
    # latent frames after 4x temporal VAE compression (+1 causal frame)
    latent_t = (num_frames - 1) // 4 + 1
    L = latent_t * (H // 8 // 2) * (W // 8 // 2)
    print(f"{num_frames} frames -> {latent_t} latent frames -> {L} tokens")
# The model's forward() runs unchanged; only the sequence length differs.
```

There is a real cost discipline hiding here, though, and it is worth stating plainly so you do not over-promise the flexibility. The weights are shape-agnostic, but the *compute* is not — a longer sequence costs quadratically more attention, as we have hammered. And the model only generalizes *near* the distribution it trained on. A model trained on clips up to 5 seconds, asked for 30 seconds, does not gracefully produce 30 good seconds; it produces 5 good seconds and then drifts, because neither the data nor (often) the position encoding's effective range covered that regime. The flexibility is "any shape *within and slightly beyond* what training covered," not "any shape at all." This is precisely the limit that pushes the field toward [autoregressive and chunked long-video methods](/blog/machine-learning/video-generation/why-video-generation-is-hard), which trade the clean one-shot generation for the ability to roll out indefinitely.

#### Worked example: one checkpoint, three aspect ratios

Say you trained a spacetime-DiT on a data mix that was 50% landscape 16:9, 30% vertical 9:16, and 20% square 1:1, all at roughly constant token budget (~100k tokens) by trading resolution against aspect. At inference, the *same* weights generate:

- **16:9 landscape**, $1280 \times 720$ → $30 \times 45 \times 80 = 108{,}000$ tokens.
- **9:16 vertical**, $720 \times 1280$ → $30 \times 80 \times 45 = 108{,}000$ tokens.
- **1:1 square**, $960 \times 960$ → $30 \times 60 \times 60 = 108{,}000$ tokens.

All three are the same sequence length, so all three cost the same and the same checkpoint serves all three with correct framing — vertical subjects centered for vertical output, wide establishing shots for landscape. A fixed-grid model would have to letterbox two of these into the third's shape, wasting tokens on black bars and teaching the model that vertical video "looks like" a centered strip in a wide frame. This is the concrete dividend of patch-based variable-shape training.

## 6. The cost wall: why this only works on top of the 3D-VAE

Let me make the dependency on the VAE quantitative, because it is the reason the whole spacetime-DiT recipe is feasible at all, and it is the thing that breaks if you get greedy.

![Before and after comparison of an image diffusion transformer on one frame versus a video diffusion transformer on a five second clip, showing the token count rising from about one thousand to about one hundred thousand and the attention cost rising about nine hundred fold](/imgs/blogs/video-diffusion-transformers-5.png)

Suppose, for a horrifying moment, that there were *no* VAE — you ran the DiT directly on pixels. Our 5-second 720p clip is $120 \times 720 \times 1280 \times 3 \approx 3.3 \times 10^8$ values. Patchify at $1 \times 16 \times 16$ (a large pixel patch) and you still get $120 \times 45 \times 80 = 432{,}000$ tokens, four times more than the latent case, and at a finer patch you get millions. Attention over 432k tokens is $(432/108)^2 = 16\times$ the latent cost, and the pixel patches carry far less useful structure per token. Run the DiT on raw pixels and you are not OOMing at second six, you are OOMing at frame one. The causal 3D-VAE's $4 \times 8 \times 8 = 256\times$ volumetric compression is what drags the token count from "impossible" to "merely expensive." The spacetime-DiT is a passenger; the VAE is the vehicle.

This inverts a common intuition. People assume the denoiser is the expensive part of a video model — it has all the parameters, it runs $N$ times per sample. But the denoiser only runs because the VAE made its input small enough to run at all. And at inference, there is a second twist: the VAE *decode* is frequently the VRAM wall, not the denoiser. Decoding 30 latent frames back to 120 pixel frames at 720p, all in memory at once, can peak higher than any single denoising step — which is why every production pipeline uses VAE *tiling* and *slicing* (decode the latent in spatial tiles and temporal chunks, never the whole thing at once). We will see the flags for that in §8. The point for the architecture: the DiT's feasibility, its token count, its cost, and even its peak memory at decode are all downstream of the VAE. Change the VAE's compression and you move every number in this post.

Let me put the FLOPs on a firmer footing. For a transformer of dimension $d$, depth $N$, sequence length $L$, the forward cost is dominated by two terms: the **attention** term $\mathcal{O}(N \cdot L^2 \cdot d)$ and the **MLP/projection** term $\mathcal{O}(N \cdot L \cdot d^2)$. The crossover — where attention overtakes the linear layers — is at $L \approx d$. For an image DiT, $L \approx 1{,}000 \approx d$, so the two terms are comparable; the model is *not* attention-bound. For a video DiT, $L \approx 100{,}000 \gg d \approx 1{,}000$, so $L^2 d \gg L d^2$ by a factor of $L/d \approx 100$ — the model is *overwhelmingly* attention-bound. This single regime change, image-DiT being projection-bound and video-DiT being attention-bound, is *why* video models obsess over attention efficiency (factorized attention, sparse/windowed attention, FlashAttention) in a way image models never had to. The architecture is the same; the operating regime is not. And it is the regime that motivates the entire [spatiotemporal attention patterns](/blog/machine-learning/video-generation/spatiotemporal-attention-patterns) post — when $L^2$ dominates, *how* you reduce $L^2$ is the whole game.

## 7. The scaling thesis: compute buys coherence

Now the claim that made Sora a moment rather than a model. Peebles and Xie's [DiT result](/blog/machine-learning/image-generation/diffusion-transformers-dit) showed that for image diffusion, transformer quality scales *cleanly and predictably* with compute — more Gflops (deeper, wider, more tokens) lowers FID along a smooth curve, with no sign of saturation in the range they tested. This is the image-diffusion scaling story, and it is the foundation the video story inherits. Because a video DiT *is* an image DiT on a longer sequence, the same scaling logic should apply: pour more compute into the spacetime-DiT and quality should improve predictably.

Sora's central, striking claim is that for *video*, scaling does something qualitatively richer than lower a Fréchet distance. The technical report shows the same prompt generated by the same model at "base compute," "4× compute," and "32× compute," and the difference is not merely sharpness — it is *coherence*. At base compute the dog's body warps as it runs, objects flicker in and out, the background slides incoherently. At 32× compute the same model holds the dog's identity across the whole clip, keeps objects that leave and re-enter the frame, and moves the camera in a way that respects 3D geometry. Sora's report frames these as *emergent* capabilities: 3D consistency, long-range coherence, object permanence, and even rudimentary "interactions with the world" — none of which were explicitly trained for. They emerge from scaling a spacetime-DiT on enough video.

![Before and after comparison of a small compute and a thirty two times compute spacetime diffusion transformer, where per frame sharpness is similar but only the large model holds object permanence and consistency across time](/imgs/blogs/video-diffusion-transformers-8.png)

Why would *coherence specifically* be the thing that scales? Here is the mechanistic argument, and it follows from §6's regime change. Per-frame sharpness is a *local* property — it depends on getting each patch's texture right, which is mostly the spatial layers and the VAE, and which even a small model handles well (small video models make individually pretty frames). Coherence is a *global* property of the joint distribution over all $L \approx 100{,}000$ tokens — it requires the attention to correctly relate a token at frame 3 to a token at frame 30, across the full sequence. Capturing those long-range spacetime dependencies is exactly what more attention capacity (more heads, more layers, more parameters) and more data (more examples of coherent motion) buy. So the scaling curve's *spatial* component saturates early (frames look good fast) while its *temporal/coherence* component keeps improving with compute — which is precisely the qualitative picture the Sora report shows. The emergent-coherence claim is the video-specific reading of the same DiT scaling law: in the attention-bound video regime, the marginal compute goes disproportionately into the global, cross-frame structure that coherence lives in.

I want to be honest about the epistemic status, because this series does not do marketing. The compute multipliers (4×, 32×) in Sora's report are illustrative, not a published scaling-law fit with axes and exponents; OpenAI did not release a clean $\text{FVD} = a \cdot C^{-b}$ curve for video the way Kaplan-style laws exist for language. So "more compute → more coherence" is well-supported as a *direction* and as a qualitative demonstration, but the precise exponent is not public. What *is* independently corroborated is the open-model trend: CogVideoX-5B is meaningfully more coherent than CogVideoX-2B (same recipe, more parameters), and HunyuanVideo/Wan at $\approx 13$–$14$B push VBench's consistency and motion dimensions higher still. The direction is real and reproduced across labs. The exact law is, as of 2026, not pinned down in public. Treat the scaling thesis as a strong, repeatedly-confirmed *trend* — and as the reason every lab is building bigger spacetime-DiTs — rather than a precise formula.

## 8. The practical flow: CogVideoX in diffusers

Enough theory; let us run the architecture. CogVideoX is the cleanest open instantiation of everything above — causal 3D-VAE, spacetime-DiT with 3D RoPE, flow-matching-adjacent training — and it is a first-class citizen in 🤗 `diffusers`. Here is a real text-to-video call with the flags that matter.

```python
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16
)

# The two flags that decide whether this fits in 24 GB:
pipe.enable_model_cpu_offload()   # stream weights on/off GPU per submodule
pipe.vae.enable_tiling()          # decode the latent in tiles, not all at once

prompt = (
    "A golden retriever running across a sunlit field, camera tracking "
    "alongside, fur and grass moving naturally, cinematic, 24fps."
)

video = pipe(
    prompt=prompt,
    num_frames=49,            # -> ~6s at 8fps export; sets the token count
    num_inference_steps=50,   # denoising steps; cost is steps x per-step
    guidance_scale=6.0,       # CFG; trades prompt-fidelity vs diversity
    generator=torch.Generator("cuda").manual_seed(0),
).frames[0]

export_to_video(video, "dog.mp4", fps=8)
```

Every flag here connects to the architecture we built. `num_frames=49` is the single knob on sequence length — it sets $T'$, hence $L$, hence the attention cost and the VRAM. `num_inference_steps=50` multiplies the per-step cost (each step is one full spacetime-DiT forward pass over all $L$ tokens). `guidance_scale` is [classifier-free guidance](/blog/machine-learning/image-generation/classifier-free-guidance), inherited unchanged from image diffusion — it runs the DiT twice per step (conditional and unconditional) and extrapolates, doubling the per-step cost for sharper prompt adherence. And the two memory flags are the §6 story made real: `enable_model_cpu_offload()` keeps the giant transformer off the GPU except when it is running, and `vae.enable_tiling()` defuses the decode VRAM wall by decoding the latent in spatial tiles rather than materializing the whole pixel clip at once.

For image-to-video — usually the higher-quality choice when you can supply a first frame, because it removes the burden of inventing a coherent scene from text alone — the call swaps the pipeline and adds an image:

```python
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import load_image, export_to_video
import torch

pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    "THUDM/CogVideoX-5b-I2V", torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

first_frame = load_image("dog_standing.png")
video = pipe(
    image=first_frame,
    prompt="the dog starts running to the right, camera follows",
    num_frames=49,
    num_inference_steps=50,
    guidance_scale=6.0,
).frames[0]
export_to_video(video, "dog_i2v.mp4", fps=8)
```

The transformer is the same; the conditioning path now includes the encoded first frame (its VAE latent is concatenated or used to initialize the denoising), which anchors identity and scene so the model spends its capacity on *motion* rather than *invention*. The choice between T2V and I2V, and the conditioning machinery behind it, is the subject of the [conditioning post](/blog/machine-learning/video-generation/why-video-generation-is-hard); architecturally, I2V is "same spacetime-DiT, richer conditioning."

If you want to inspect the transformer directly — to confirm it is the DiT we built — you can reach into the pipeline:

```python
dit = pipe.transformer            # CogVideoXTransformer3DModel
print(type(dit).__name__)         # the spacetime-DiT
print(dit.config.num_layers)      # depth N (e.g. 42 for the 5B)
print(dit.config.num_attention_heads, dit.config.attention_head_dim)
print(dit.config.patch_size)      # spatial patch (time handled in VAE)
# The VAE is the causal 3D autoencoder that sets the token budget:
print(type(pipe.vae).__name__)    # AutoencoderKLCogVideoX
print(pipe.vae.config.temporal_compression_ratio)  # ~4
```

This is the architecture made tangible: a `CogVideoXTransformer3DModel` (our `VideoDiT`), an `AutoencoderKLCogVideoX` (the causal 3D-VAE that produced the latent and sets $L$), and a flow-matching scheduler driving the denoising. Every concept from §1–§7 has a line you can print.

## 9. Training the spacetime-DiT: the loop and the noise-schedule shift

Inference is the easy half. Training a spacetime-DiT is where the architecture's costs become visceral, and there are two video-specific wrinkles on top of the standard diffusion-transformer recipe that are worth making explicit: the **memory profile** and the **noise-schedule shift**. Both follow directly from the token-count story.

The training loop itself is, again, the image-DiT loop with a clip instead of an image. You encode a batch of video clips to latents with the frozen 3D-VAE, sample a timestep (or a flow-matching time) per clip, add noise / interpolate, run the DiT to predict noise or velocity, and take an MSE step. Here is the inner loop in PyTorch with flow-matching, the objective most 2024–2026 video models train with.

```python
import torch
import torch.nn.functional as F

def video_dit_train_step(dit, vae, batch, text_emb, optimizer):
    # batch: (B, 3, T_pixels, H, W) raw video clips, already on GPU
    with torch.no_grad():
        # frozen causal 3D-VAE -> latent clip (B, C, T', H', W')
        latents = vae.encode(batch).latent_dist.sample() * vae.config.scaling_factor

    B = latents.shape[0]
    # flow-matching: sample a time t in (0,1), interpolate clean->noise
    t = torch.rand(B, device=latents.device)
    noise = torch.randn_like(latents)
    t_b = t.view(B, 1, 1, 1, 1)
    x_t = (1 - t_b) * latents + t_b * noise          # straight path
    target = noise - latents                          # velocity target

    pred = dit(x_t, t, text_emb)                      # spacetime-DiT forward
    loss = F.mse_loss(pred.float(), target.float())   # per-voxel, summed

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(dit.parameters(), 1.0)
    optimizer.step()
    return loss.item()
```

Nothing here is video-specific except the *shape* of `latents` (five dimensions instead of four) and the fact that the VAE is a causal 3D autoencoder. The velocity target $u_t = x_1 - x_0 = \text{noise} - \text{latents}$ is computed per voxel exactly as in the [image flow-matching](/blog/machine-learning/image-generation/flow-matching-and-rectified-flow) case. This is the third time we have seen the same pattern, and it is the post's refrain: the new axis touches the *shapes*, not the *math*.

The first wrinkle is **memory**, and it is severe. The activations of a transformer scale with sequence length, and at $L \approx 100{,}000$ the attention activations alone — the $L \times L$ score matrices, even materialized per-head — are enormous. Three techniques are non-negotiable for training at this scale: (1) **FlashAttention**, which never materializes the full $L \times L$ matrix and so turns attention's *memory* from $\mathcal{O}(L^2)$ to $\mathcal{O}(L)$ (the compute stays $\mathcal{O}(L^2)$, but memory is the wall); (2) **gradient checkpointing**, which discards block activations on the forward pass and recomputes them on the backward, trading ~30% more compute for a large memory saving — essential when a single clip's activations would otherwise not fit; and (3) **sequence/context parallelism**, splitting the token sequence across GPUs so each holds only a slice of the 100k tokens. The launch looks like this:

```bash
accelerate launch \
  --multi_gpu --num_processes 8 \
  --mixed_precision bf16 \
  train_video_dit.py \
  --model_dim 3072 --depth 42 --heads 24 \
  --patch_t 1 --patch_h 2 --patch_w 2 \
  --num_frames 49 --resolution 480 \
  --gradient_checkpointing \
  --attn_impl flash \
  --sequence_parallel_size 4 \
  --learning_rate 1e-4 --warmup_steps 1000 \
  --batch_size_per_gpu 1 --grad_accum 8
```

Note `--batch_size_per_gpu 1`: at 100k tokens, *one clip per GPU* is often all that fits even with every trick on, and you reach an effective batch size through gradient accumulation and data parallelism. This is the operating reality the token count forces. The image DiT trains with batch sizes in the hundreds per GPU; the video DiT trains with one. Same architecture, utterly different logistics — and the difference is entirely the sequence length.

The second wrinkle is the **noise-schedule shift**, a genuinely video-relevant piece of science. Flow-matching and diffusion both have an implicit assumption about how much noise it takes to "destroy" the signal, and that assumption depends on the *resolution and token count*. The intuition: at high resolution and long duration, a given absolute noise level destroys *less* perceptual information than at low resolution, because there is more redundancy (neighboring pixels and frames are correlated, so noise averages out more). Stable Diffusion 3's paper formalized this for images as a **timestep shift** that pushes the sampling distribution toward higher noise levels for higher-resolution generation, and video models inherit and extend it: because a video has even *more* redundancy than an image (the temporal axis adds correlation), the schedule must shift even further. Concretely, models scale the flow-matching time by a *shift* factor $s$:

$$
t' = \frac{s \cdot t}{1 + (s - 1) \, t}, \qquad s > 1 \text{ for high-res / long video}.
$$

This warps the time distribution so the model spends more of its training and sampling budget at the high-noise end, where the coarse global structure (and, for video, the coarse *motion*) is determined. Get the shift wrong — use an image-tuned schedule on a long high-res clip — and the model under-trains the high-noise regime, producing clips with correct fine texture but incoherent global motion: the frames are sharp and the *trajectory* is wrong. This is a video-specific failure that the architecture alone does not cause and the schedule must fix. It is also why you cannot simply take an image-DiT checkpoint, feed it clips, and expect coherent motion for free even after adding temporal attention — the noise schedule has to move with the token count.

#### Worked example: the activation-memory wall at training time

Take the $d = 3072$, depth 42 model (roughly CogVideoX-5B's shape) training on 49-frame 480p clips, $L \approx 17{,}500$ tokens. Per layer, the naive attention score matrix is $L^2 = 17{,}500^2 \approx 3.1 \times 10^8$ entries *per head*; at 24 heads and bf16 (2 bytes) that is $3.1\times10^8 \times 24 \times 2 \approx 15$ GB for *one layer's* attention scores, and there are 42 layers. Materialize those naively and a single forward pass needs hundreds of GB — impossible on any single GPU. FlashAttention drops this to near-zero (it never stores the full matrix), and gradient checkpointing drops the stored *block* activations from 42 layers' worth to ~1 layer's worth at the cost of a recompute. Even then, one 480p clip per GPU is typical. Now scale the clip to 720p 129 frames ($L \approx 80$k) and you see why HunyuanVideo's training needed a large H100 cluster with sequence parallelism: the activation memory is the wall, and it scales with $L$, which scales with frames times resolution. The architecture is cheap to *describe* and brutally expensive to *train* — and the gap between those two is exactly the token count.

## 10. Case studies and real numbers

Let us ground the scaling thesis in measured results from shipped, documented models. Numbers below are from the respective technical reports and the public VBench leaderboard; where I am uncertain I mark it approximate, and I never invent a figure.

![Matrix comparing video diffusion transformer models across parameter count, frames and resolution, and VBench score, showing the open frontier sharing one recipe at different scales](/imgs/blogs/video-diffusion-transformers-7.png)

| Model | Params | Frames × res | Reported quality / notes |
|---|---|---|---|
| **CogVideoX-2B** | ~2B | 49f · 720×480 | VBench total ~80%; the clean open baseline; runs on a single 24 GB GPU with offload |
| **CogVideoX-5B** | ~5B | 49f · 720×480 | VBench ~81–82%; same recipe, more parameters → better motion & consistency |
| **HunyuanVideo** | ~13B | 129f · 720p | Strong open SOTA at release; VBench ~83%+; MMDiT-style joint text-video attention |
| **Wan 2.1** | ~14B (also 1.3B) | up to 81f · 720p | Top open VBench (~84%+); 1.3B variant runs on consumer GPUs |
| **Sora / Sora 2** | undisclosed | up to ~60s | No public VBench; the scaling-thesis demonstration; closed |

Three things to read out of this table. First, the **architecture is constant** — every open model here is causal-3D-VAE + spacetime-DiT + flow-matching. The variation is scale (2B → 14B) and details (patch size, full vs factorized attention, AdaLN vs joint text tokens). Second, **VBench rises with scale**, exactly as the scaling thesis predicts: CogVideoX-2B → 5B is a within-recipe controlled comparison, and the larger model wins on the motion and consistency dimensions. Third, **frames and resolution are bought with parameters and compute** — HunyuanVideo's 129 frames at 720p is a far larger token budget than CogVideoX's 49 frames at 480p, and it takes a $\sim 13$B model and serious infrastructure to denoise that sequence.

A measurement honesty note, because VBench is gameable and this series insists on it. VBench's "dynamic degree" dimension rewards *motion*, and its "subject consistency" dimension rewards *stability* — and these trade off. A model that barely moves scores high on consistency and low on dynamic degree; a model with frantic motion does the reverse. So a single "VBench total" can be inflated by a near-static model that wins the consistency dimensions while failing the one thing video is for: motion. When you compare models, read the *dimension breakdown*, not just the total, and weight dynamic degree against consistency for your use case. The right way to measure a video DiT is: fixed prompt set, fixed seeds, fixed frame count, report the per-dimension VBench plus seconds-per-clip and peak VRAM on a *named* GPU — and look at the motion-vs-stability frontier, not a scalar. The [metrics post](/blog/machine-learning/video-generation/the-metrics-of-video-generation) goes deep on exactly this gaming problem.

#### Worked example: cost and memory for a 6-second CogVideoX-5B clip

Concretely, here is what generating one 6-second clip (49 frames) with CogVideoX-5B costs on an A100 80GB, order-of-magnitude:

- **Tokens**: 49 frames → ~13 latent frames → with $720\times480$ latent ($90 \times 60$, patch $1\times2\times2$) → $13 \times 45 \times 30 \approx 17{,}500$ tokens. (CogVideoX's lower resolution and frame count keep $L$ well under our 720p 5-second example's 108k — a deliberate cost choice.)
- **Per-step cost**: one spacetime-DiT forward over ~17.5k tokens, ×2 for CFG, ×50 steps = 100 forward passes.
- **Wall-clock**: roughly 90–180 seconds on an A100 80GB at bf16, dominated by attention. On a 24 GB 4090 with `enable_model_cpu_offload()` and VAE tiling it still runs but slower, as weights stream on and off the GPU.
- **Peak VRAM**: the denoiser fits comfortably with offload; the *decode* of 13 latent frames → 49 pixel frames at 480p is the spike, which VAE tiling caps. Without tiling, decode can OOM a 24 GB card even when the denoiser was fine — the §6 wall, live.

The actionable trade-off: every quality lever (more frames, higher resolution, more steps, CFG on) multiplies into either $L$, $L^2$, or the step count. There is no free coherence — only tokens you pay for.

## 11. Stress-testing the design

A good architecture is defined as much by its failure modes as its features. Let us push the spacetime-DiT until it breaks, because knowing *where* it breaks is what lets you ship it.

**Past the VAE's trained clip length.** The VAE was trained to encode and decode clips up to some length — say 49 or 129 frames. Ask it to decode 300 frames in one shot and two things go wrong: VRAM (the §6 decode wall) and *quality drift* — causal 3D-VAEs accumulate small reconstruction errors over long temporal contexts, and the late frames degrade. The fix is temporal tiling in the VAE (decode in overlapping chunks) and, for the denoiser, chunked/sliding-window generation. The spacetime-DiT itself is shape-agnostic, but its *position encoding* and its *training distribution* are not — ask for far more frames than trained and the RoPE temporal range extrapolates poorly and coherence falls apart after the trained horizon. This is the hard ceiling that motivates [long-video and autoregressive methods](/blog/machine-learning/video-generation/why-video-generation-is-hard).

**When attention is factorized away.** Full spacetime attention couples every token to every token — maximum coherence, $L^2$ cost. Factorized attention (spatial-then-temporal) drops the cost dramatically but weakens long-range *diagonal* dependencies (a token at frame 3, top-left relating to a token at frame 30, bottom-right is now a two-hop path through the factorization rather than a direct edge). For most content this is invisible; for fast diagonal motion or objects that traverse the frame, factorized attention can produce subtle incoherence that full attention would catch. The trade is quantified in the [spatiotemporal attention post](/blog/machine-learning/video-generation/spatiotemporal-attention-patterns); the container here works with either.

**When motion is large between frames.** The VAE's temporal compression ($4\times$) assumes adjacent frames are *similar* — that is the redundancy that makes compression possible. With very fast motion (a fast pan, an explosion), adjacent frames differ a lot, the temporal redundancy the VAE relies on collapses, and the latent loses information the denoiser cannot recover. Symptoms: motion blur, ghosting, temporal aliasing. The spacetime-DiT is downstream of this; it can only denoise the latent it was given, and if the VAE threw away the fast motion, the DiT cannot invent it back coherently.

**When identity drifts over a long rollout.** Generate a 5-second clip and the dog stays the same dog. Roll out to 30 seconds via chunked generation and the dog's markings slowly mutate — error accumulates because each chunk conditions on the (slightly imperfect) last frames of the previous chunk, and the imperfections compound. This is not a flaw in the spacetime-DiT block; it is a flaw in *autoregressive rollout* on top of it, and it is the central open problem in long video. The one-shot spacetime-DiT is coherent *within* its trained horizon and only drifts when you stitch beyond it.

**When VAE decode, not the denoiser, is the VRAM wall.** This one surprises people the first time. You profile a generation that OOMs, you assume the 14B transformer is the culprit, you add more offload to the *denoiser* — and it still OOMs, at the very end, *after* all the denoising steps finished. The spike is the VAE decode: turning 30 latent frames into 120 pixel frames at 720p, all resident at once, is a single large allocation that can dwarf any individual denoising step's footprint. The fix is not in the DiT at all — it is `vae.enable_tiling()` and `vae.enable_slicing()`, which decode the latent in spatial tiles and temporal chunks. The architectural lesson: in a video pipeline the denoiser and the decoder are *separate* memory regimes, and the decoder is often the binding one. Diagnose the spike before you optimize the wrong component.

Here is the failure map as a table, because it is the kind of thing you want pinned above your desk while debugging a render that looks subtly wrong.

| Symptom | Root cause | Where it lives | Fix |
|---|---|---|---|
| Late frames blur / degrade | VAE reconstruction error over long context | 3D-VAE, not DiT | Temporal tiling; stay within trained length |
| Subtle incoherence on diagonal motion | Factorized attention misses long-range diagonal | Attention pattern | Full 3D attention (3× cost) for that content |
| Ghosting / motion blur on fast action | Temporal compression lost the fast motion | 3D-VAE | Lower temporal compression; higher fps source |
| Identity mutates over 30s | Error accumulation in chunked rollout | Rollout loop, not block | Anchor frames; overlap; identity conditioning |
| Sharp frames, wrong global motion | Noise schedule under-trains high-noise regime | Training schedule | Apply the resolution/length timestep shift |
| OOM after denoising finishes | VAE decode materializes full pixel clip | Decoder memory | `enable_tiling()` + `enable_slicing()` |

Read the table as a diagnostic flowchart: the *symptom* tells you which *component* to suspect, and almost none of the fixes are in the DiT block itself. That is the deepest practical truth about the spacetime-DiT — it is a remarkably robust, almost boring container, and the interesting failures live in the VAE that feeds it, the attention pattern inside it, the schedule that trains it, and the rollout loop that extends it.

## 12. When to reach for a spacetime-DiT (and when not to)

A decisive recommendation, because the kit demands one. The spacetime-DiT is the *default* architecture for video generation in 2026, and for good reason — but it is not always the right call.

**Reach for it when**: you want a single architecture that scales predictably, you can afford the attention cost (you have A100/H100-class hardware or you accept slow generation with offload), and you want the flexibility dividend (variable duration/resolution/aspect from one checkpoint). It is the right choice for any frontier-quality T2V/I2V system. Every open model worth self-hosting (CogVideoX, HunyuanVideo, Wan) is a spacetime-DiT, so reaching for it also means reaching for a mature, supported toolchain.

**Do not reach for full spacetime attention when factorized hits your coherence bar at a fraction of the FLOPs.** This is the single most important cost decision. For most content, factorized (spatial-then-temporal) attention reaches near-full-3D coherence at roughly a third of the attention FLOPs. Use full 3D attention only when you have measured that factorization costs you on your specific content (fast diagonal motion, long-range object tracking) — otherwise you are paying $3\times$ for coherence you cannot see.

**Do not over-scale resolution and frames in one shot.** The cost is multiplicative and quadratic. If you need a long, high-res result, generate at a moderate resolution and frame count and upscale/interpolate as a *separate* cheaper stage, rather than asking one giant DiT forward pass to do everything. The token budget is the constraint; spend it where it shows.

**Prefer I2V over T2V when you can supply a first frame.** It removes the scene-invention burden, anchors identity, and lets the model's capacity go to motion — usually higher quality for the same compute.

**Do not autoregress for a fixed short clip.** If you want exactly 5 seconds, generate it in one shot — the one-shot spacetime-DiT is coherent within its trained horizon. Autoregressive rollout, with its identity drift, is a tool for *unbounded* length, not a default. Pay its cost only when the length genuinely exceeds what one forward pass can hold.

**Reconsider a U-Net only for tiny budgets.** AnimateDiff-style temporal modules over a frozen image U-Net (see [SVD and AnimateDiff](/blog/machine-learning/video-generation/why-video-generation-is-hard)) are cheaper to train and run for short, low-res clips and reuse a pretrained image backbone. For frontier quality, scale, and flexibility, the DiT wins — but for a small project on consumer hardware, the U-Net recipe is still viable.

## 13. Key takeaways

- **A spacetime patch is a cuboid token.** Slice the 3D-VAE latent into small blocks spanning time and space, flatten and project each to a vector, and a plain transformer denoises the whole clip as one long sequence. Time is just more tokens.
- **A video DiT is an image DiT with a longer sequence.** The block math, AdaLN conditioning, residual gating, and diffusion/flow-matching loss are reused *unchanged*. Only three things are new: patching across time, attention over spacetime tokens, and 3D positions.
- **Token count is everything.** $L = (T'/p_t)(H'/p_h)(W'/p_w)$. It scales linearly with frames and quadratically into attention cost. A 5-second 720p clip is ~108k tokens and ~900× an image's attention bill.
- **The video DiT is attention-bound; the image DiT is not.** Because $L \gg d$ for video, attention's $L^2 d$ dwarfs the linear layers' $L d^2$. This regime change is *why* video models obsess over factorized and efficient attention.
- **3D RoPE encodes (t, h, w) with no learned table.** Split the head dimension into thirds, rotate each by its coordinate. Relative positions, table-free, with some extrapolation to longer/larger clips.
- **The patch design buys flexibility for free.** One checkpoint serves variable duration, resolution, and aspect ratio — Sora's "patches of variable size" property — because a transformer does not care how many tokens you give it (within the trained distribution).
- **It only works on top of the 3D-VAE.** The VAE's ~256× volumetric compression is what drags the token count from impossible to expensive. The decode is often the real VRAM wall — use tiling.
- **Compute buys coherence, not just sharpness.** Scaling a spacetime-DiT improves the global, cross-frame structure (object permanence, identity, 3D consistency) more than the local per-frame quality that small models already nail. The exact law is not public; the direction is reproduced across labs.

## Further reading

- **Brooks, Peebles, et al. — "Video generation models as world simulators" (OpenAI, Sora technical report, 2024).** The spacetime-patch framing and the compute → coherence scaling demonstration. The primary source for this post's central thesis.
- **Peebles & Xie — "Scalable Diffusion Models with Transformers" (DiT, 2023).** The image diffusion transformer and its clean compute-scaling law — the architecture this post grows a time axis on. See also our [DiT post](/blog/machine-learning/image-generation/diffusion-transformers-dit).
- **Yang et al. — "CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer" (2024).** The cleanest open spacetime-DiT: causal 3D-VAE, expert transformer, 3D RoPE — the model we ran in §8.
- **Kong et al. — "HunyuanVideo: A Systematic Framework for Large Video Generative Models" (2024).** A ~13B open spacetime-DiT with MMDiT-style joint text-video attention; a detailed engineering report.
- **Wan team — "Wan: Open and Advanced Large-Scale Video Generative Models" (2025).** Top open VBench at the 14B scale, plus a 1.3B consumer-GPU variant; the converged open recipe.
- **Lipman et al. — "Flow Matching for Generative Modeling" (2023).** The training objective most video DiTs use; we link out to [flow matching](/blog/machine-learning/image-generation/flow-matching-and-rectified-flow) rather than re-derive it.
- **Dehghani et al. — "Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution" (2023).** The native-resolution / variable-shape packing idea behind §5's flexibility dividend.
- **Within this series**: [why video generation is hard](/blog/machine-learning/video-generation/why-video-generation-is-hard) (the foundation), [from image to video diffusion](/blog/machine-learning/video-generation/from-image-diffusion-to-video-diffusion) and [video autoencoders](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression) (the inputs to this architecture), [spatiotemporal attention patterns](/blog/machine-learning/video-generation/spatiotemporal-attention-patterns) (how attention spans the tokens), and the capstone [building with video generation: the playbook](/blog/machine-learning/video-generation/building-with-video-generation-the-playbook).
