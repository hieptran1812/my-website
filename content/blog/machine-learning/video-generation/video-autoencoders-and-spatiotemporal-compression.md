---
title: "Video Autoencoders and Spatiotemporal Compression: The Real Bottleneck"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Why the causal 3D-VAE — not the denoiser — sets the cost and the clip length of every video diffusion model, and how to encode, decode, and stream long clips without OOMing."
tags:
  [
    "video-generation",
    "diffusion-models",
    "video-diffusion",
    "autoencoders",
    "3d-vae",
    "spatiotemporal-compression",
    "text-to-video",
    "generative-ai",
    "deep-learning",
  ]
category: "machine-learning"
subcategory: "Video Generation"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/video-autoencoders-and-spatiotemporal-compression-1.png"
---

The first time I tried to generate a five-second 720p clip with a freshly-trained denoiser, the model worked — and it was unusably slow. Each sampling step chewed through a latent tensor with hundreds of thousands of tokens, fifty steps took the better part of ten minutes on an A100, and the decode at the end OOMed at second four. My instinct, like everyone's, was to blame the denoiser: too many parameters, attention too expensive, sampler too greedy. That instinct was wrong. The denoiser was just paying a bill that something upstream had quietly set. The thing that decides whether a video diffusion model is cheap or ruinous, short or long, crisp or shimmering, is the **autoencoder** that turns pixels into latents — and specifically whether it compresses *time* as well as space.

This is the post where the series stops talking about pixels and starts talking about the *latent* that everything downstream actually operates on. In the [image-generation series we built the variational autoencoder from scratch](/blog/machine-learning/image-generation/variational-autoencoders-from-scratch) and saw how [latent diffusion](/blog/machine-learning/image-generation/latent-diffusion-and-stable-diffusion) made Stable Diffusion tractable by denoising in a compressed space instead of in pixels. Video raises the stakes by an order of magnitude, because now there is a whole extra axis — time — that is both the source of the cost (more frames = more data) and the source of the relief (adjacent frames are nearly identical, so they should compress hard). The piece of the stack that turns that relief into reality is the **3D video VAE**, and getting it right is the single highest-leverage decision in the entire pipeline. Figure 1 is the map of where it sits and what it does.

![A branching dataflow graph showing a pixel clip flowing into a 3D encoder that splits into a spatial 8x8 reduction branch and a causal temporal 4x reduction branch, both merging into a small latent tensor that a 3D decoder upsamples back to the original 49 frames](/imgs/blogs/video-autoencoders-and-spatiotemporal-compression-1.png)

The punchline of this entire post, stated up front so you can hold it the whole way through: **video latent diffusion lives or dies on the 3D-VAE.** The denoiser's cost scales with the number of latent tokens; the VAE's compression ratio sets that number; therefore the VAE — not the transformer everyone obsesses over — sets the budget for compute, for memory, and for how long a clip you can possibly generate. We will derive that claim rigorously, write the code to feel it, and then stress-test it against the failure modes that show up the moment you push past the VAE's comfort zone: flicker, detail loss, the first-frame boundary artifact, color drift, and the decode-time memory wall that ends most ambitious long-video runs. Throughout we keep one running example — a five-second 720p clip of a dog running, 49 frames at 720×1280 — and watch what the VAE does to it.

## 1. Why a per-frame image VAE is not enough

The lazy way to build a video model is to take a great image VAE — say the one from Stable Diffusion, which downsamples 8× in each spatial dimension — and run it on every frame independently. Encode frame 1, encode frame 2, and so on; stack the resulting latents into a tensor of shape `[T, C, H/8, W/8]`; denoise that; decode each latent frame back to pixels independently. It is the obvious first move, and it is wrong in two ways that matter enormously.

The first problem is **the latent is still T× too big**. An image VAE compresses each frame spatially by 8×8 = 64× in pixel area, and folds the 3 RGB channels into (say) 4 or 16 latent channels. That is a real win for a single image. But for a clip it does nothing about the *time* axis. Our dog clip has 49 frames; after a per-frame image VAE the latent is 49 frames deep. The denoiser must process all 49 of them, and — as we will prove in section 5 — its cost grows at least linearly and, where attention spans time, quadratically in that frame count. The image VAE throws away none of the gigantic redundancy *between* frames. Frame 24 and frame 25 of a dog running are almost the same image; a per-frame encoder spends a full latent frame on each of them anyway. We are paying to store the same wall, the same grass, the same sky, forty-nine times.

The second problem is **frames decode independently, so they flicker**. The VAE decoder is a lossy, slightly stochastic map: tiny differences in the latent — and there are always tiny differences, because the encoder is not perfectly deterministic and the denoiser injects its own noise — produce visible differences in the decoded pixels. When each frame is decoded with no knowledge of its neighbors, those small per-frame differences are uncorrelated across time. The result is the single most recognizable artifact in early video generation: **temporal flicker**. Fine textures shimmer — the dog's fur boils, the grass crawls, flat walls pulse with a faint moiré that changes every frame. The content is right; it just refuses to hold still. A human eye is brutally sensitive to this because our visual system evolved to notice motion, and flicker reads as motion that has no cause.

You can paper over flicker with post-hoc temporal smoothing, but that is treating the symptom. The disease is architectural: the decoder has no temporal receptive field, so it cannot enforce that "this patch of fur looked like *that* in the previous frame, so keep it consistent." The fix is to give the autoencoder a temporal receptive field — to make it a genuinely *3D* model that sees a stack of frames at once, compresses them jointly, and decodes them jointly. That is the causal 3D-VAE, and it solves both problems at the same stroke: it compresses the time axis (killing the T×-too-big problem) *and* it couples neighboring frames during decode (killing the flicker). Figure 6, later, shows the before/after directly.

There is a deeper, information-theoretic reason a per-frame VAE is leaving money on the table, and it is worth making explicit because it is the prior that the entire 3D-VAE exploits. Video is overwhelmingly *redundant along time*. Consider the raw bitrate: our 49-frame 720p clip at 3 bytes per pixel is about 135 megabytes of uncompressed pixels. But a standard video codec — H.264, H.265 — routinely compresses that to a few hundred kilobytes, a ratio in the hundreds-to-thousands, and the bulk of that compression comes not from intra-frame coding (JPEG-style spatial compression of single frames) but from *inter-frame* coding: motion-compensated prediction, where a frame is encoded as "the previous frame, with these blocks shifted by these motion vectors, plus a small residual." The codec wins because consecutive frames are almost the same image. A per-frame image VAE is, in this analogy, an all-intra codec — it codes every frame from scratch and captures none of the inter-frame redundancy. The 3D-VAE is the analog of an inter-frame codec: by convolving across time it learns to represent the *changes* between frames rather than re-encoding the static content of each one. This is exactly why the temporal compression factor ($f_t = 4$) is nearly free in quality terms — it is removing redundancy that was genuinely there, not throwing away signal. The [next post on representing video redundancy](/blog/machine-learning/video-generation/representing-video-redundancy-and-tokens) develops this bitrate argument in full; here the takeaway is that the 3D-VAE is the learned, differentiable analog of a video codec's inter-frame prediction, and that is *why* it can compress time so cheaply.

#### Worked example: the cost of doing nothing about time

Take our dog clip: 49 frames, 720×1280, 3 channels. A per-frame image VAE at 8× spatial gives latents of `[49, 16, 90, 160]`. Patchify that for a DiT at patch size 2×2 and you get `49 × (90/2) × (160/2) = 49 × 45 × 80 = 176,400` tokens, before you even add the temporal patching most video DiTs use. A 3D-VAE that *additionally* compresses time 4× turns the 49 frames into 13 latent frames, giving `13 × 45 × 80 = 46,800` tokens at the same spatial patching — a 3.77× reduction in token count from the temporal axis alone. Since attention is quadratic in tokens, that 3.77× fewer tokens is roughly a **14× reduction in attention FLOPs** for the dominant layers. You have not touched the denoiser. You changed the autoencoder, and the denoiser's bill dropped by more than an order of magnitude. That is the leverage we are talking about.

## 2. The causal 3D-VAE: joint spatial and temporal compression

A 3D-VAE is structurally the same idea as an image VAE — an encoder that maps pixels to a low-dimensional latent distribution, a decoder that maps latents back to pixels, trained to reconstruct — but every 2D convolution becomes a 3D convolution that spans `(time, height, width)`, and the downsampling happens along all three axes. Concretely, the modern open recipe (CogVideoX, HunyuanVideo, Wan, LTX-Video all converge here) compresses:

- **Spatially by 8× in each of height and width** — the same factor as a good image VAE, so the spatial latent of our 720×1280 clip is 90×160.
- **Temporally by 4×** — the new axis, achieved by strided 3D convolutions along time, so 49 input frames become roughly 13 latent frames.
- **Into a latent channel dimension of 16** (CogVideoX uses 16; HunyuanVideo uses 16; some variants use 4 or 8), folding the 3 RGB channels up to give the denoiser a richer per-token representation.

The total **value compression ratio** is what matters for cost, and it is the product of these factors. Let me make it precise. Let the input clip have $T$ frames, height $H$, width $W$, and 3 channels. Let the VAE compress time by $f_t$, height and width by $f_s$ each, and map to $C$ latent channels. The number of scalar values in the input is

$$
N_\text{px} = T \cdot H \cdot W \cdot 3,
$$

and the number in the latent is

$$
N_z = \frac{T}{f_t} \cdot \frac{H}{f_s} \cdot \frac{W}{f_s} \cdot C.
$$

The compression ratio is their quotient:

$$
R = \frac{N_\text{px}}{N_z} = \frac{T \cdot H \cdot W \cdot 3}{\frac{T}{f_t}\cdot\frac{H}{f_s}\cdot\frac{W}{f_s}\cdot C} = f_t \cdot f_s^2 \cdot \frac{3}{C}.
$$

The $T$, $H$, $W$ all cancel — the ratio is a property of the *architecture*, not the clip. Plug in the CogVideoX numbers ($f_t = 4$, $f_s = 8$, $C = 16$):

$$
R = 4 \cdot 8^2 \cdot \frac{3}{16} = 4 \cdot 64 \cdot 0.1875 = 48.
$$

So a CogVideoX-style 3D-VAE achieves about **48× value compression**. People sometimes quote "a couple hundred ×" for video VAEs, and that figure refers to the *spatiotemporal volume* compression $f_t \cdot f_s^2 = 4 \cdot 64 = 256\times$ — the number of latent *positions* per pixel position, ignoring the channel expansion. Both numbers are correct; they answer different questions. The 256× volume ratio is what determines the **token count** the denoiser sees (because tokens are positions, not raw scalars). The 48× value ratio is what determines the **storage and bandwidth**. Keep them straight: when we talk about denoiser cost, the volume ratio is the one that bites. Figure 2 stacks these stages with the actual value counts for our clip.

![A vertical stack diagram showing how raw pixel value count is reduced through spatial eightfold compression, temporal fourfold compression, and channel expansion, arriving at the final latent value count and total compression ratio](/imgs/blogs/video-autoencoders-and-spatiotemporal-compression-2.png)

### 3D convolutions and the shape of the receptive field

Why 3D convolutions and not, say, 2D convolutions plus a separate temporal mixing layer? You can do either — and the "(2+1)D" factorization that splits a 3D conv into a spatial 2D conv followed by a 1D temporal conv is a legitimate and cheaper alternative we will revisit when we discuss the [denoiser's own factorized attention](/blog/machine-learning/video-generation/from-image-diffusion-to-video-diffusion). But for the VAE the full 3D conv has a property that matters: its receptive field is a genuine spatiotemporal *block*, so a single latent value summarizes a small cube of `(time × height × width)` pixels. That is exactly the right inductive bias for compression, because the redundancy in video lives in spatiotemporal blocks — a patch of grass that barely changes over four frames should collapse to roughly one latent value, and a 3D conv with a 4-frame temporal stride does precisely that collapse.

The cost is memory. A 3D conv's activation tensor carries the time dimension through the whole encoder, so peak activation memory during a forward pass scales with clip length. This is the seed of the decode-time memory wall we hit in section 8 — the very mechanism that makes the VAE good at compression (carrying time through 3D activations) is what makes it expensive to run on long clips.

### The encoder/decoder anatomy and the (2+1)D shortcut

It helps to know the rough shape of what is inside the box. The encoder is a stack of residual blocks built from 3D convolutions, interleaved with downsampling layers that halve the spatial resolution (and, at a few specific layers, the temporal resolution). A typical CogVideoX-class encoder downsamples spatially three times (2×2×2 = 8× total) and temporally twice (2× twice = 4× total), with the temporal downsampling concentrated in the early layers so that most of the network operates on the already-compressed time axis — which keeps the bulk of the compute cheap. After the final block, a 1×1×1 convolution projects to `2C` channels: $C$ for the latent mean and $C$ for the log-variance of the Gaussian posterior, which is what `.latent_dist` exposes in the code from section 4. The decoder is the mirror image — residual 3D blocks interleaved with *upsampling* layers that reverse the spatial and temporal reductions — and, crucially, the decoder's upsampling is where the activation tensors balloon back toward pixel scale, which is the memory story of section 8.

The full 3D convolution is not the only way to mix across time, and the alternative is worth naming because it is the same trade-off that recurs in the denoiser. A **(2+1)D** block factorizes a single 3D convolution into a spatial 2D convolution (mixing $H, W$ within each frame) followed by a separate 1D temporal convolution (mixing across frames at each spatial location). The parameter and FLOP count of a (2+1)D block is much lower than a full 3D block — a $3\times3\times3$ kernel has 27 weights per channel pair, while a $1\times3\times3$ spatial plus $3\times1\times1$ temporal has $9 + 3 = 12$ — and the separation lets you initialize the spatial part from a pretrained *image* VAE and only train the temporal part from scratch, the "inflate a 2D VAE" trick that several video models use to save training compute. The cost is expressiveness: a factorized block cannot represent a feature that genuinely couples a diagonal spacetime direction (an edge moving diagonally across the frame) in a single layer; it has to compose one across multiple layers. For the *VAE*, where the job is compression rather than fine-grained generation, the full 3D conv is usually worth its extra cost, because the receptive-field cube is the right shape for collapsing spatiotemporal redundancy. For the *denoiser*, the factorization story flips — and that is the subject of the [image-diffusion-to-video-diffusion post](/blog/machine-learning/video-generation/from-image-diffusion-to-video-diffusion). The same factorization knob, two different optimal settings, because the VAE and the denoiser are solving different problems on the same tensor.

## 3. Causal temporal convolution: why the first frame stays crisp and the clip can stream

Here is the subtle, beautiful idea that separates a *video* VAE from a naive 3D autoencoder, and it is worth slowing down for. A plain 3D convolution is **symmetric in time**: to compute the output at latent frame $t$, it looks at input frames both before and after $t$ (a kernel of temporal size 3 centered at $t$ reads $t-1$, $t$, $t+1$). That seems harmless until you ask two questions. First: what about the very first frame, which has no past? A symmetric conv pads it with zeros on the left, and that zero-padding leaks into the reconstruction — the first frame comes out subtly wrong, washed-out or smeared, the notorious **first-frame artifact**. Second: how do you encode a clip of *arbitrary, unbounded* length, or stream a clip frame-by-frame as it arrives? A symmetric conv needs the future to compute the present, so you cannot — you are stuck processing fixed-size windows and stitching them, with seams.

The fix is to make the temporal convolution **causal**: pad only on the left (the past), never on the right (the future), so that the output at latent frame $t$ depends *only* on input frames at positions $\le t$. Formally, for a temporal kernel of size $k$, a causal conv computes

$$
y_t = \sum_{i=0}^{k-1} w_i \, x_{t - (k-1) + i},
$$

where any index $t-(k-1)+i < 0$ is handled by left-padding with replicated or zero frames. The output at $t$ is a function of $x_{t-(k-1)}, \dots, x_t$ — strictly the present and the past. Figure 3 shows this window: the output at $t$ reads $t$ and $t-1$, and the future frame $t+1$ is masked off entirely.

![A two-row grid showing input frames across time with the future frame marked blocked, and below it the causal output at time t reading only the present and past frames while the future frame is masked off](/imgs/blogs/video-autoencoders-and-spatiotemporal-compression-3.png)

This one design choice buys three things at once, and each is load-bearing:

1. **The first frame stays crisp.** Because the conv only looks left, the first frame is encoded and decoded as a special case — typically these VAEs treat the first latent frame as a standalone "image" latent (encoded with no temporal context) and only apply temporal compression to the frames *after* it. CogVideoX, for instance, maps $T$ input frames to $1 + (T-1)/f_t$ latent frames: one un-compressed first frame plus the temporally-compressed remainder. For our 49-frame clip that is $1 + 48/4 = 13$ latent frames, with the first one carrying a full-quality keyframe. This is why image-to-video models can take a sharp input image as the first frame and have it survive the round trip — the causal VAE protects it.

2. **You can encode arbitrary length.** Since latent frame $t$ never needs frames after $t$, you can feed frames in as they arrive and emit latents online. The clip length is no longer bounded by a fixed window; it is bounded only by how you manage memory (section 8). This is the architectural prerequisite for the long-video and streaming work in the [autoregressive-rollout post](/blog/machine-learning/video-generation/from-image-diffusion-to-video-diffusion) — without causal temporal conv, "generate a 60-second clip" is not even well-posed.

3. **Encode and decode become chunkable.** Causality means you can split the time axis into chunks, process each chunk with only a small left-context overlap carried from the previous chunk, and get *bit-identical* results to processing the whole clip at once (up to the overlap). That is what makes the constant-memory streaming decode of section 8 possible without seams.

The trade-off, to be honest about it: a causal conv has a strictly smaller receptive field than a symmetric one (it sees only the past, not the future), so in principle it has less information to reconstruct each frame and can be marginally worse on raw PSNR. In practice the difference is tiny — a few hundredths of a dB — and it is dwarfed by the gains in streaming, first-frame quality, and length-generalization. Every serious open video VAE since CogVideoX is causal. The symmetry you give up was never worth what it cost.

To make this concrete, here is what a causal temporal convolution looks like in PyTorch. The entire trick is the asymmetric padding: pad `kernel - 1` frames on the *left* of the time axis and zero on the right, so the convolution's output at frame $t$ can only have drawn from frames $\le t$. This is a building block you would stack inside the VAE's residual blocks in place of an ordinary `Conv3d`.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv3d(nn.Module):
    """A 3D conv that is causal along time: output at frame t sees only frames <= t."""
    def __init__(self, in_ch, out_ch, kernel=(3, 3, 3)):
        super().__init__()
        self.kt, self.kh, self.kw = kernel
        # No built-in time padding; we pad manually and asymmetrically below.
        self.conv = nn.Conv3d(in_ch, out_ch, kernel, padding=0)
        self.pad_h = self.kh // 2
        self.pad_w = self.kw // 2
        self.pad_t = self.kt - 1  # all temporal padding goes on the LEFT (past)

    def forward(self, x):  # x: [B, C, T, H, W]
        # F.pad order is (W_left, W_right, H_left, H_right, T_left, T_right).
        # Spatial padding is symmetric; temporal padding is left-only -> causal.
        x = F.pad(x, (self.pad_w, self.pad_w,
                      self.pad_h, self.pad_h,
                      self.pad_t, 0))   # T_right = 0 means no future leakage
        return self.conv(x)

# Sanity check: the output at frame t must not change when we alter frame t+1.
layer = CausalConv3d(4, 4).eval()
x = torch.randn(1, 4, 8, 16, 16)
y1 = layer(x)
x2 = x.clone(); x2[:, :, 5:] += 10.0      # perturb only frames 5..7 (the future)
y2 = layer(x2)
print("frames 0..4 unchanged:", torch.allclose(y1[:, :, :5], y2[:, :, :5], atol=1e-5))
```

The sanity check is the whole point: perturbing a *future* frame leaves every earlier output untouched, which is exactly the causality property that makes streaming and chunked decode exact rather than approximate. If that `allclose` ever prints `False`, your padding is leaking the future and your "streaming" decode will have seams.

## 4. Loading a real 3D-VAE and feeling the compression

Enough theory — let's encode our dog clip with an actual `diffusers` video VAE and print the shapes, so the ratio stops being algebra and becomes a tensor you can hold. We use `AutoencoderKLCogVideoX`, the causal 3D-VAE that ships with CogVideoX. The pattern below loads the VAE in `bfloat16`, encodes a clip to its latent distribution, samples a latent, and prints the compression ratio we derived.

```python
import torch
from diffusers import AutoencoderKLCogVideoX

# Load just the VAE from the CogVideoX-5b checkpoint.
vae = AutoencoderKLCogVideoX.from_pretrained(
    "THUDM/CogVideoX-5b",
    subfolder="vae",
    torch_dtype=torch.bfloat16,
).to("cuda")
vae.eval()

# A clip tensor: [batch, channels, frames, height, width], values in [-1, 1].
# 49 frames is the CogVideoX native length; 480x720 is a common training res.
clip = torch.randn(1, 3, 49, 480, 720, device="cuda", dtype=torch.bfloat16)

with torch.no_grad():
    # encode() returns a distribution; .sample() draws a latent, .mode() takes the mean.
    posterior = vae.encode(clip).latent_dist
    latent = posterior.sample()

print("pixel  clip shape:", tuple(clip.shape))      # (1, 3, 49, 480, 720)
print("latent      shape:", tuple(latent.shape))    # (1, 16, 13, 60, 90)

n_px = clip.numel()
n_z  = latent.numel()
print(f"value compression ratio R = {n_px / n_z:.1f}x")  # ~48x
print(f"latent frames: {latent.shape[2]} from {clip.shape[2]} input frames")  # 13 from 49
```

A few things to notice in the output. The frame axis went `49 → 13` — that is the $1 + 48/4 = 13$ causal temporal compression, with the leading 1 being the protected first frame. Each spatial axis went `480 → 60` and `720 → 90`, the 8× spatial factor. The channel axis went `3 → 16`. The value compression ratio prints at roughly 48×, exactly matching $R = f_t f_s^2 \cdot 3/C = 4 \cdot 64 \cdot 3/16$. The latent you now hold is `[1, 16, 13, 60, 90]` — about 1.1 million scalars, down from 50.8 million pixels. *That* tensor, not the pixel clip, is what the denoiser will spend its FLOPs on.

Now decode it back and confirm the round trip restores the original shape:

```python
with torch.no_grad():
    # CogVideoX latents are scaled by a constant before/after the denoiser;
    # for a pure VAE round-trip you decode the raw sampled latent directly.
    reconstructed = vae.decode(latent).sample

print("reconstructed shape:", tuple(reconstructed.shape))  # (1, 3, 49, 480, 720)
# In a real run you'd compute PSNR / LPIPS between `clip` and `reconstructed`
# on real video (random noise has no structure to reconstruct).
```

The decode restores `[1, 3, 49, 480, 720]` — same shape as the input. On real video (not the random tensor above, which has nothing to reconstruct) you would measure reconstruction quality with PSNR and [LPIPS](/blog/machine-learning/image-generation/variational-autoencoders-from-scratch) between input and output; a well-trained CogVideoX-class VAE lands around 31–33 dB PSNR on held-out clips, which is the "good but not perfect" band that the whole pipeline is built to tolerate. The reconstruction is slightly soft — that softness is the price of 48× compression, and the denoiser is trained to generate latents that decode to *plausible* video, not pixel-perfect video, so a little softness in the VAE is acceptable as long as it is *temporally stable*.

## 5. Why the VAE is THE bottleneck: deriving the denoiser's bill

Now the central claim, made provable. The denoiser is a diffusion transformer ([DiT](/blog/machine-learning/image-generation/diffusion-transformers-dit)) operating on patches of the latent. Its cost per sampling step is dominated by two terms: the per-token feed-forward / projection work, which is **linear** in the token count $N$, and the self-attention, which is **quadratic** in $N$ when attention spans the full spatiotemporal sequence. Write the per-step FLOPs as

$$
\text{FLOPs}_\text{step} \approx \underbrace{c_1 \, N \, d^2}_{\text{linear (FFN, proj)}} + \underbrace{c_2 \, N^2 \, d}_{\text{attention}},
$$

where $d$ is the hidden width and $c_1, c_2$ are constants. Total generation cost multiplies by the number of sampling steps $S$:

$$
\text{FLOPs}_\text{total} \approx S \big( c_1 N d^2 + c_2 N^2 d \big).
$$

Now connect $N$ to the VAE. The token count is the number of *latent patches*. With a DiT patch size of $p$ spatially and $p_t$ temporally, and a latent of shape $[C, T/f_t, H/f_s, W/f_s]$,

$$
N = \frac{T/f_t}{p_t} \cdot \frac{H/f_s}{p} \cdot \frac{W/f_s}{p} = \frac{T \, H \, W}{f_t f_s^2 \cdot p_t p^2}.
$$

The denominator contains $f_t f_s^2$ — the VAE's *volume* compression ratio. So $N$ is **inversely proportional to the VAE's volume compression**:

$$
N \propto \frac{1}{f_t f_s^2}.
$$

Substitute into the attention term, which dominates at the token counts video models use:

$$
\text{FLOPs}_\text{attn} \propto N^2 \propto \frac{1}{(f_t f_s^2)^2}.
$$

This is the whole argument in one line. **Doubling the VAE's volume compression cuts the token count in half and the attention FLOPs by 4×.** A more compressing VAE does not make the denoiser a little cheaper — it makes the *dominant* cost drop quadratically. Conversely, a weak VAE (or a per-frame image VAE with $f_t = 1$) inflates $N$, and the denoiser's attention cost explodes by the *square* of that inflation. The VAE sets the budget; the denoiser merely spends it. Figure 7 traces this dependency from the ratio through to seconds-per-clip.

![A branching graph where the VAE compression ratio sets the latent token count, which feeds both the quadratic attention cost and the per-step FLOPs, both flowing through the sampling step count into total denoiser seconds per clip](/imgs/blogs/video-autoencoders-and-spatiotemporal-compression-7.png)

#### Worked example: what 4× temporal compression saves on an A100

Take the per-frame image VAE baseline: our 49-frame clip at 8× spatial, DiT patch 2×2×1, gives $N_\text{2D} = 49 \cdot 45 \cdot 80 = 176{,}400$ tokens. Now switch to a 3D-VAE with $f_t = 4$ and temporal patch $p_t = 1$ on the 13 latent frames: $N_\text{3D} = 13 \cdot 45 \cdot 80 = 46{,}800$ tokens, a 3.77× reduction. The linear FFN term drops 3.77×; the quadratic attention term drops $3.77^2 \approx 14.2\times$. On an A100 80GB where the 2D-VAE configuration takes, say, 9 minutes for 50 steps, the attention-dominated portion shrinks roughly 14×; even with the linear terms and fixed overheads, real measurements on CogVideoX-class models put the 3D-VAE configuration in the **2–3 minute** range for the same clip — a 3–4× wall-clock speedup that came *entirely from the autoencoder*. The denoiser architecture, the sampler, the step count: all unchanged. This is why I keep saying the VAE is the lever. You can spend a month tuning the DiT and get 20%; you swap to a 4× temporal VAE and get 3×.

### How to measure a VAE honestly

The VAE's job is reconstruction, so its primary metric is reconstruction quality on *held-out* video — not the training clips, and not random tensors. The standard battery is **PSNR** (peak signal-to-noise ratio, a pixel-level fidelity number in decibels; higher is better, ~31–33 dB is the good band for a 48× video VAE), **LPIPS** (perceptual distance; lower is better), and a **reconstruction FVD** — the Fréchet Video Distance between a set of real clips and their VAE round-trips. FVD is the video analog of FID: it extracts features from a pretrained I3D action-recognition network for both sets of clips, fits a Gaussian to each feature set, and computes the Fréchet distance $\| \mu_r - \mu_g \|^2 + \mathrm{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$ between them. A reconstruction FVD isolates the VAE's contribution to the full pipeline's FVD — if the VAE round-trip alone already has a high FVD, no denoiser can recover; the VAE has put a floor under your achievable quality.

Three honesty traps to avoid when you report these numbers. First, **PSNR hides flicker**. A VAE can have excellent per-frame PSNR and still flicker, because PSNR averages over frames and does not penalize temporal *inconsistency*. Always pair PSNR with a temporal metric — frame-to-frame LPIPS, or a flicker score that measures the variance of the reconstruction error across time. The whole point of the temporal-consistency loss is invisible to PSNR. Second, **FVD is noisy and sample-size-sensitive**: it needs a few thousand clips for a stable estimate, it shifts with clip length and resolution, and small FVD differences (say, 5 points) are within the noise. Report the sample count, the clip length, and a fixed seed, or the number is not reproducible. Third, **measure the decode path you will actually ship**. If you deploy with `enable_tiling()`, measure reconstruction quality *with* tiling on, because the tile-seam blending changes the output slightly — a VAE that scores well in a full-frame decode can show faint seam artifacts in production tiled decode, and you want to catch that before users do.

#### Worked example: reading a reconstruction-FVD table

Suppose you are choosing between two VAE configurations for a 5-second 720p pipeline and you measure on 2,048 held-out clips at a fixed seed. Config A (4×8×8, 16 channels) scores reconstruction FVD 24, PSNR 31.8 dB, frame-to-frame LPIPS 0.018. Config B (8×16×16, 16 channels) scores reconstruction FVD 61, PSNR 28.1 dB, frame-to-frame LPIPS 0.041 — but its latent has 7× fewer tokens, so the denoiser runs ~3× faster and a 30-second clip fits where Config A's would OOM. The decision is not "A is better"; it is "A puts a quality floor of FVD 24 under the pipeline, B puts a floor of 61 but unlocks 6× longer clips at the same latency." If your product is short cinematic clips, A. If your product is long, real-time-ish generation where some softness is acceptable, B. The reconstruction-FVD table told you exactly where each VAE's quality floor sits, and the token count told you the cost — and that is the whole video-VAE decision rendered as two numbers you can act on.

## 6. The training losses: how you teach a VAE to compress 256× without ruining the video

A VAE that compresses spatiotemporal volume 256× is throwing away an enormous amount of information, and the *art* is in throwing away the right information — the redundant, imperceptible bits — while keeping what the eye cares about. You cannot get that from a single reconstruction loss; L2 alone gives you a blurry, over-smoothed decoder that minimizes mean-squared error by hedging toward the average of all plausible pixels. The modern video VAE is trained with a weighted sum of five terms, each pulling in a different direction. Figure 5 stacks them.

![A vertical stack of the five 3D-VAE training loss terms: pixel reconstruction, perceptual LPIPS, adversarial, KL regularizer, and a temporal consistency term, summing into a weighted total loss](/imgs/blogs/video-autoencoders-and-spatiotemporal-compression-5.png)

**1. Reconstruction loss (L1 or L2 on pixels).** The floor of the objective: decoded pixels should match input pixels. L1 (mean absolute error) is often preferred over L2 because it is less prone to the gray-mush averaging failure and tends to preserve edges. This term keeps the global structure — colors, layout, large shapes — faithful.

**2. Perceptual loss (LPIPS).** Pixel losses do not match human perception; two images can be far in L2 but perceptually identical (a one-pixel shift) or close in L2 but perceptually awful (a uniform blur). [LPIPS](/blog/machine-learning/image-generation/variational-autoencoders-from-scratch) measures distance in the feature space of a pretrained network (VGG/AlexNet), which correlates far better with what people notice. Adding LPIPS is what restores *texture* — fur that looks like fur, grass with high-frequency detail — that pure L1 would smooth away. In a video VAE, LPIPS is typically applied per-frame.

**3. Adversarial loss (a discriminator).** Even LPIPS leaves a slight softness; a GAN discriminator that tries to tell real frames from reconstructed ones pushes the decoder to produce *crisp, plausible* high-frequency detail. Crucially, video VAEs use a **3D (spatiotemporal) discriminator** — it judges short clips, not single frames — so it penalizes not just per-frame blur but *temporal* implausibility: a discriminator that sees a 4-frame window learns to spot flicker and unnatural motion, and the decoder learns to avoid them. This is one of the two main forces (the other is term 5) that kill flicker.

**4. KL regularizer.** This is the "variational" part: the encoder outputs a distribution (mean and variance), and a KL term pulls that distribution toward a standard Gaussian. The weight is *tiny* — on the order of $10^{-6}$ — because the goal is not to make the latent perfectly Gaussian (that would over-regularize and hurt reconstruction) but merely to keep the latent space smooth, bounded, and well-scaled so the downstream denoiser sees a sane distribution. Some video VAEs replace this with a vector-quantization / commitment term (a VQ-VAE flavor), but the continuous-KL formulation dominates the diffusion line because diffusion wants a continuous latent. The relationship to the diffusion prior is exactly the one we covered in the [latent-diffusion post](/blog/machine-learning/image-generation/latent-diffusion-and-stable-diffusion): the VAE provides the latent space, the diffusion model learns to sample in it.

**5. Temporal-consistency loss.** The term that exists *only* in video VAEs and is the heart of why they do not flicker. There are a few formulations: penalize the difference between consecutive *reconstructed* frames where the input frames are similar (a frame-difference or optical-flow-warped consistency loss), or add a term that matches the temporal gradient of the reconstruction to the temporal gradient of the input. The effect is to explicitly forbid the decoder from introducing per-frame variation that was not in the source. Together with the 3D discriminator, this is what turns a boiling, shimmering decode into a stable one.

The total objective is the weighted sum

$$
\mathcal{L}_\text{VAE} = \mathcal{L}_\text{rec} + \lambda_\text{p}\,\mathcal{L}_\text{LPIPS} + \lambda_\text{adv}\,\mathcal{L}_\text{adv} + \lambda_\text{KL}\,\mathcal{L}_\text{KL} + \lambda_\text{t}\,\mathcal{L}_\text{temporal},
$$

with the weights chosen so reconstruction and perceptual terms dominate, the adversarial term is annealed in after the VAE has learned basic reconstruction (turning the GAN on too early destabilizes training), the KL weight is microscopic, and the temporal weight is tuned to kill flicker without over-smoothing motion. Here is a compact PyTorch sketch of the loss assembly — not a full training loop, but the shape of the objective so you can see how the terms combine:

```python
import torch
import torch.nn.functional as F
import lpips  # pip install lpips

lpips_fn = lpips.LPIPS(net="vgg").cuda().eval()

def vae_loss(x, x_hat, posterior, discriminator, step,
             lam_p=1.0, lam_adv=0.5, lam_kl=1e-6, lam_t=1.0, adv_warmup=20000):
    # x, x_hat: [B, C, T, H, W] in [-1, 1]
    B, C, T, H, W = x.shape

    # 1. Reconstruction (L1 keeps edges, avoids gray mush).
    rec = F.l1_loss(x_hat, x)

    # 2. Perceptual: flatten frames into the batch and run per-frame LPIPS.
    xf  = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    xhf = x_hat.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    perceptual = lpips_fn(xhf, xf).mean()

    # 3. Adversarial: a 3D discriminator scores short clips; annealed in late.
    adv = torch.tensor(0.0, device=x.device)
    if step > adv_warmup:
        adv = -discriminator(x_hat).mean()  # generator wants high real-score

    # 4. KL: pull the latent posterior toward N(0, I), with a microscopic weight.
    kl = posterior.kl().mean()

    # 5. Temporal consistency: reconstructed frame-to-frame deltas should match
    #    the input's. This is the term that suppresses flicker.
    dx     = x[:, :, 1:] - x[:, :, :-1]
    dx_hat = x_hat[:, :, 1:] - x_hat[:, :, :-1]
    temporal = F.l1_loss(dx_hat, dx)

    total = rec + lam_p * perceptual + lam_adv * adv + lam_kl * kl + lam_t * temporal
    return total, {"rec": rec, "lpips": perceptual, "adv": adv,
                   "kl": kl, "temporal": temporal}
```

The temporal term (`dx_hat` vs `dx`) is doing the quiet heavy lifting: if the input's frame-to-frame change at a pixel is near zero (a static background), the loss punishes the decoder for putting *any* change there. That is flicker, defined and penalized.

## 7. Failure modes and trade-offs: what breaks when you compress hard

Every choice in a 3D-VAE is a trade-off between **compression** (cheaper denoiser, longer clips) and **reconstruction fidelity** (sharper, more stable video). Push compression up and four characteristic failures appear, in roughly this order. Knowing their signatures lets you diagnose a bad VAE from the decoded video alone.

**Temporal flicker** is the failure of insufficient temporal modeling — too-weak a temporal-consistency loss, no 3D discriminator, or a per-frame VAE. Signature: fine textures shimmer and boil; flat regions pulse. Fix: strengthen the temporal term and the 3D discriminator, or move from per-frame to genuine 3D. Figure 6 contrasts the two regimes side by side.

![A before-and-after comparison contrasting a per-frame 2D VAE that decodes each frame independently and flickers against a causal 3D-VAE that shares temporal context, removes the shimmer, and also cuts the latent frame count fourfold](/imgs/blogs/video-autoencoders-and-spatiotemporal-compression-6.png)

**Detail loss at high compression** is the fidelity cost of pushing the volume ratio up. At 256× you keep most of what matters; at 1024× (e.g. $f_t = 8$, $f_s = 16$) you start losing fine text, small faces, thin structures (fence wires, distant branches). Signature: the *content* is right but soft, edges blurred, small high-frequency objects smeared or hallucinated. This is fundamental — you cannot losslessly compress 1024× — and the only real fixes are to back off the compression or add latent channels (raise $C$, which lowers the *value* ratio while keeping the *volume* ratio, buying back fidelity at the cost of more denoiser channels).

**The boundary / first-frame artifact** is the failure mode that causal temporal conv was designed to fix, and it returns if you get the causal handling wrong. Signature: the first frame (or the first frame of each decode chunk) is subtly washed out, off-color, or smeared relative to the rest. With a properly causal VAE that protects the first latent frame, this disappears; with chunked decode it can reappear at chunk seams if you do not carry enough left-context overlap (section 8).

**Color drift** is the slow, insidious one: over a long clip the decoded colors gradually shift — a scene that started neutral creeps warm, or brightness ramps. It comes from tiny systematic biases in the decoder accumulating across many latent frames, and it is worst in autoregressive / chunked generation where each chunk inherits and amplifies the previous chunk's bias. Signature: no single frame looks wrong, but frame 1 and frame 200 have visibly different white balance. Fixes are partial — better temporal regularization, color-consistency losses, and re-anchoring to the conditioning image periodically in I2V.

The unifying trade-off, the thing to internalize: **more compression = cheaper denoiser but harder reconstruction.** Figure 4 lays this out as a matrix across three VAE regimes, and it is the single most useful artifact in this post for making the call.

![A matrix comparing three VAE variants across compression ratio, latent token count, reconstruction quality, and flicker, showing the per-frame 2D VAE, the standard 3D-VAE at four-by-eight-by-eight, and a high-compression variant](/imgs/blogs/video-autoencoders-and-spatiotemporal-compression-4.png)

Read the matrix as a Pareto frontier. The per-frame 2D VAE is dominated — it has the *worst* token count (397k after temporal patching considerations) *and* the worst flicker, with only a marginal PSNR edge that does not justify the cost. The standard 4×8×8 3D-VAE is the sweet spot the whole field converged on: ~45× compression, ~19k tokens, good PSNR, low flicker. The aggressive 8×16×16 variant is for when you are length- or latency-bound and can tolerate softer output — it cuts tokens another 7× but starts losing detail and showing edge ghosting. Where you sit on this frontier is *the* video-VAE decision, and it should be driven by your binding constraint: latency-bound → compress harder; quality-bound → compress less and pay the denoiser.

## 8. How clip length is gated by the VAE: chunked and causal decoding

Here is where the VAE stops being a theoretical concern and becomes the literal wall your run hits at 3am. Generating the *latent* for a long clip is cheap-ish — the denoiser cost scales with token count, which scales with length, but linearly. The killer is the **decode**. Recall from section 2 that the 3D decoder carries the time dimension through its activations, and it *upsamples* — the spatial 8× and temporal 4× compression run in reverse, so the activation tensors at the decoder's later stages are enormous, approaching full pixel resolution across all frames at once. Decoding our 13-latent-frame clip in one pass needs the decoder to hold activations for all 49 output frames at 480×720 simultaneously, and for a longer clip — say 30 seconds, ~750 frames — that is tens of gigabytes of activations. **The VAE decode, not the denoiser, is usually the first thing to OOM.**

I have watched this exact thing many times: the denoiser finishes, the progress bar hits 100%, and then the pipeline dies in `vae.decode()` with a CUDA out-of-memory error. The denoiser fit in 16GB; the decode wanted 30. Figure 8 is the diagnosis and the fix side by side.

![A before-and-after comparison showing a naive full decode that holds all frames at once and OOMs at second six on a 24-gigabyte GPU, versus a tiled and chunked decode that processes spatial patches and frame chunks to hold peak VRAM nearly constant at any clip length](/imgs/blogs/video-autoencoders-and-spatiotemporal-compression-8.png)

The fix is two complementary forms of chunking, both of which `diffusers` exposes as one-line toggles:

**Spatial tiling** (`vae.enable_tiling()`): decode the latent in overlapping spatial tiles — e.g. 256×256 patches with a small overlap — and blend the seams. Peak activation memory now scales with the *tile* size, not the full frame, so a 4K decode that would need 40GB fits in 8. The overlap-blend hides the seams; without overlap you get a visible grid.

**Temporal chunking** (`vae.enable_slicing()` / frame chunking): decode the latent frames in groups — say 8 latent frames at a time — carrying a small left-context overlap from the previous chunk so the causal conv has the past it needs. Because the VAE is *causal*, this is exact: chunk boundaries do not introduce seams as long as the overlap covers the temporal receptive field. Peak memory now scales with the *chunk* length, not the clip length — which is exactly what makes arbitrary-length decode possible. This is the payoff of the causal design from section 3: causality is what makes constant-memory streaming decode correct, not just cheap.

Here is the practical pattern. The two `enable_*` calls are the difference between a run that fits and a run that dies:

```python
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()      # stream weights; frees room for the decode
pipe.vae.enable_tiling()             # spatial tiles -> decode VRAM ~ tile, not frame
pipe.vae.enable_slicing()            # decode frames in chunks -> VRAM ~ chunk, not clip

video = pipe(
    prompt="a golden retriever running across a sunny field, cinematic",
    num_frames=49,                   # gated by the VAE's trained temporal range
    num_inference_steps=50,
    guidance_scale=6.0,
    generator=torch.Generator("cuda").manual_seed(0),
).frames[0]

export_to_video(video, "dog_running.mp4", fps=8)
```

The VRAM effect is dramatic and worth measuring on your own hardware. On an RTX 4090 (24GB), a 49-frame 480×720 CogVideoX-5b decode *without* tiling peaks well past 24GB and OOMs; *with* `enable_tiling()` and `enable_slicing()` it peaks around 8–10GB and completes, leaving headroom for the offloaded denoiser. The cost is a modest slowdown (tiles and chunks add overlap recomputation and reduce parallelism) — typically 10–25% more decode time — which is a trade every long-clip run gladly makes, because the alternative is no clip at all.

It is worth understanding *why* the decode is the wall, with the back-of-envelope accounting, because it makes the fix obvious. The decoder's memory is dominated by its widest activation tensor, which appears near the end where spatial resolution has been upsampled back toward full but the channel count is still in the hundreds. For a single full-resolution decoder stage at 480×720 with, say, 256 channels in `bfloat16` (2 bytes), one frame's activation is $720 \times 480 \times 256 \times 2 \approx 177$ MB. With 49 output frames held simultaneously that is ~8.7 GB for *one* stage — and the decoder has several such wide stages plus the autograd-free but still-resident intermediates, which is how you get to 25–30 GB. The key term is the frame count multiplying everything. Spatial tiling attacks the $720 \times 480$ factor (a 256×256 tile is ~7× smaller in area); temporal chunking attacks the 49-frame factor (an 8-frame chunk is ~6× smaller). Multiply those reductions and the peak collapses from tens of gigabytes to single digits, with the only residual cost being the overlap regions you recompute at each tile and chunk boundary.

Here is the explicit chunked-decode pattern when you want control beyond the one-line toggles — for example, to decode a very long latent that you generated in windows. The pattern carries a left-context overlap of latent frames into each chunk so the causal convolutions have the past they need, then trims the overlap from the output:

```python
import torch

@torch.no_grad()
def chunked_decode(vae, latent, chunk=4, overlap=1):
    """Decode a long latent [B, C, T_lat, H, W] in temporal chunks with left overlap.
    Causality makes this seam-free as long as `overlap` covers the temporal receptive field."""
    T = latent.shape[2]
    frames = []
    start = 0
    while start < T:
        # Pull a chunk plus left-context overlap from the previous chunk.
        lo = max(0, start - overlap)
        chunk_latent = latent[:, :, lo:start + chunk]
        decoded = vae.decode(chunk_latent).sample  # [B, 3, t_px, H*8, W*8]
        # Trim the pixel frames that correspond to the overlap region.
        trim = 0 if lo == start else (start - lo) * 4  # 4 = temporal upsample factor
        frames.append(decoded[:, :, trim:])
        start += chunk
    return torch.cat(frames, dim=2)

# Peak VRAM now scales with `chunk`, not with the full latent length T_lat,
# so a 200-latent-frame clip decodes in the same memory as a 4-frame one.
```

The `trim` arithmetic is the only fiddly part: because the temporal upsampling factor is 4, each overlapping *latent* frame expands to 4 *pixel* frames in the decode, and you drop exactly those to avoid emitting the overlap region twice. Get the trim wrong and you get either duplicated frames (overlap too large, not trimmed) or a seam-and-jump (overlap too small for the receptive field). With the trim correct, the concatenated output is indistinguishable from a monolithic decode — that indistinguishability is the payoff of causality, and it is what lets the peak memory be set by `chunk` instead of by the clip length.

#### Worked example: where the VRAM wall actually is

Budget a 6-second clip (≈73 frames after the causal `1 + 72/4 = 19` latent frames) at 480×720 on a 4090. The denoiser (CogVideoX-5b in bf16, with CPU offload) runs the 19-latent-frame sequence in roughly 14–16GB peak — fits. The *full* decode of 19 latent frames to 73 output frames at 480×720 wants, by activation accounting, north of 25GB — does **not** fit; this is the OOM-at-second-6 in the figure. Turn on tiling (256-px tiles) and slicing (8-frame chunks) and the decode peak drops to ~9GB. Net: the run that died now completes, the wall moved from the *decode* back to the *denoiser*, and the clip length is no longer gated by a single monolithic decode but by how patient you are with chunked decoding. The lesson generalizes: **before you blame the denoiser for your length limit, check whether it is actually the VAE decode that is OOMing — it usually is.**

There is one more length gate worth naming: the VAE's *trained* temporal range. A VAE trained on 49-frame clips has learned its temporal statistics for that length; pushing it to decode a 500-frame latent in one causal pass can drift (the color-drift failure mode) even if it fits in memory, because the model is extrapolating beyond its training distribution. The robust pattern for very long video is to generate in overlapping windows at the *latent* level too — the [autoregressive-rollout and long-video techniques](/blog/machine-learning/video-generation/from-image-diffusion-to-video-diffusion) — and let the causal VAE stitch them. Memory chunking and distributional length-generalization are two different walls; tiling solves the first, windowed generation solves the second.

## 9. Case studies: real VAEs, real numbers

Let's ground all of this in shipped models. These are the open, documented 3D-VAEs whose recipes you can actually read and run, with figures drawn from their papers and model cards. Where I state a number I am confident in, I state it; where a figure is approximate or load-dependent, I say so. Never trust a benchmark you cannot reproduce.

**CogVideoX (THUDM, 2024).** The reference open causal 3D-VAE. Compression $f_t = 4$, $f_s = 8$, $C = 16$ latent channels → ~48× value / 256× volume compression. The causal temporal conv with the protected first frame ($1 + (T-1)/4$ latent frames) is the design we coded against in section 4. CogVideoX-5b generates 6-second 720×480 clips at 8 fps; the VAE is the part that makes a 5B-parameter denoiser fit on a single 24GB consumer GPU with offload + tiling. Its `AutoencoderKLCogVideoX` is the most-copied video VAE in `diffusers`.

**HunyuanVideo (Tencent, 2024) and HunyuanVideo-1.5 (2025).** A 13B-parameter open model whose VAE also uses 4×8×8 spatiotemporal compression with a causal 3D design and 16 latent channels — the same converged recipe. HunyuanVideo's contribution to *this* topic is mostly engineering at scale: a carefully-tuned VAE training schedule (reconstruction → perceptual → adversarial annealing) and the demonstration that the recipe holds at 13B. HunyuanVideo-1.5 pushes toward long-clip generation (reports of ~75-second clips on a 4090-class GPU with aggressive chunked decode), which is *entirely* a story about causal VAE + chunked decode + windowed generation — exactly the section 8 machinery.

**Wan 2.x (Alibaba, 2025).** Another open model on the converged causal-3D-VAE + DiT + flow-matching recipe. Wan's VAE notably emphasizes the reconstruction↔compression trade-off, offering configurations that trade a bit of fidelity for cheaper denoising; it is a good case study in choosing your point on the Figure 4 frontier deliberately rather than by default.

**LTX-Video (Lightricks, 2024).** The interesting outlier: LTX-Video pushes the compression *much* harder (a higher overall spatiotemporal ratio) specifically to enable near-real-time generation, and accepts the corresponding softness in reconstruction. It is the living proof of the section 5 math — by shrinking $N$ aggressively at the VAE, it shrinks the denoiser's quadratic attention cost enough to generate faster than real time on a single high-end GPU. It sits at the "aggressive" end of Figure 4's frontier on purpose, and it is the model to study if your binding constraint is latency, not fidelity.

The table below collects the comparison. Treat the latency/VRAM figures as order-of-magnitude on the named hardware — they depend heavily on resolution, step count, offload settings, and driver/runtime, and I have rounded toward the conservative side.

| Model (VAE) | Temporal × spatial | Latent ch. | Value ratio | Clip (native) | Notes |
|---|---|---|---|---|---|
| Per-frame 2D (SD VAE) | 1× × 8×8 | 4–16 | ~12–48× | n/a (image) | flickers; latent T× too deep |
| CogVideoX-5b | 4× × 8×8 | 16 | ~48× | 6 s @ 720×480 | reference causal 3D-VAE |
| HunyuanVideo-13b | 4× × 8×8 | 16 | ~48× | ~5 s @ 720p | same recipe, 13B scale |
| HunyuanVideo-1.5 | 4× × 8×8 | 16 | ~48× | up to ~75 s* | long-clip via chunked decode |
| Wan 2.x | 4× × 8×8 | 16 | ~48× | ~5 s @ 480/720p | tunable fidelity/compression |
| LTX-Video | aggressive | — | higher | real-time class | softer recon, latency-first |

\*The 75-second figure is a reported headline result under aggressive chunked decode and windowed generation; sustained quality at that length is the open hard problem, not a solved one.

The pattern across all of them is the convergence itself: **4× temporal, 8×8 spatial, 16 latent channels, causal conv** is the recipe the open field settled on, because it is the point on the trade-off frontier where the denoiser becomes affordable without the reconstruction falling apart. When you see a new open video model, the first thing to check is its VAE's compression numbers — they tell you more about its cost and length envelope than the denoiser's parameter count does.

It is worth noting how the VAE's quality propagates into the *full-pipeline* metrics everyone quotes, because it explains a subtlety in reading benchmark tables. [VBench](/blog/machine-learning/video-generation/why-video-generation-is-hard), the standard video-generation benchmark, scores dimensions like subject consistency, background consistency, motion smoothness, dynamic degree, and imaging quality. The VAE contributes directly to several of these in ways that are easy to misattribute to the denoiser. *Imaging quality* is partly the VAE's reconstruction fidelity — a softer VAE caps your imaging-quality score no matter how good the denoiser is. *Motion smoothness* and *background consistency* are partly the VAE's temporal stability — a flickering VAE drags both down, and a strong temporal-consistency loss lifts them. So when one open model beats another on VBench imaging quality and background consistency, a meaningful part of that gap can be the VAE, not the headline denoiser. The lesson for anyone benchmarking: if you are comparing denoisers, hold the VAE fixed, or you are partly measuring the autoencoder. And if you are comparing whole pipelines, remember that swapping in a better VAE is often a cheaper path to a VBench gain than scaling the denoiser — another consequence of the leverage this whole post is about.

One more case worth calling out is the special handling of the **first frame in image-to-video** pipelines, because it is where the causal-VAE design pays off most visibly. In I2V (the [conditioning post](/blog/machine-learning/video-generation/from-image-diffusion-to-video-diffusion) covers the full mechanism), you supply a sharp first frame and the model continues it. That first frame is encoded by the causal VAE as the protected, un-temporally-compressed leading latent — which is exactly why I2V models can preserve a crisp input image while everything after it is generated. SVD, CogVideoX-I2V, and the I2V variants of the open models all lean on this. If the VAE were not causal, the input frame would be smeared by symmetric temporal padding the moment it entered the encoder, and the I2V promise — "start from *this* exact image" — would be impossible to keep. The causal design is not just a streaming convenience; it is what makes image-conditioned generation faithful to its condition.

## 10. When to reach for which VAE configuration (and when not to)

A decisive section, because the VAE choice is the one that propagates into every other cost in your pipeline. Here is how I make the call.

**Use the standard 4×8×8 causal 3D-VAE (CogVideoX-class) for almost everything.** It is the default for a reason: it sits at the knee of the trade-off curve. Unless you have a specific, measured constraint pulling you off it, this is the right choice and you should not overthink it.

**Compress harder (LTX-class, higher temporal/spatial ratio) only when latency or length is your binding constraint** and you have *confirmed* that the denoiser — not the VAE decode — is your wall, and that your content tolerates softer reconstruction. Real-time/interactive generation, very long clips, and serving at scale are the legitimate reasons. Talking-head close-ups and text-heavy scenes are the wrong content for aggressive compression — that is exactly where detail loss bites hardest.

**Compress less / add latent channels only when reconstruction fidelity is the hard requirement** — film-grade output, fine text, small faces — and you have the denoiser budget to pay for the extra tokens. Raising $C$ (latent channels) buys back fidelity at the volume ratio without inflating token count as much as lowering $f_s$ would, so it is often the better fidelity dial.

**Do NOT use a per-frame image VAE for video. Ever.** It is dominated on every axis that matters — worse flicker *and* worse cost. The only time it appears is in early prototypes or in models that deliberately keep a frozen image backbone and bolt on temporal layers elsewhere (the AnimateDiff line), and even those pay for it in flicker. If you find yourself stacking per-frame image latents into a video tensor, stop and reach for a real 3D-VAE.

**Do NOT skip tiling/chunking on long clips.** `enable_tiling()` and `enable_slicing()` cost you 10–25% decode time and save you from the OOM that ends the run. There is no clip long enough that you should decode it monolithically; turn them on by default for anything past a couple of seconds.

**Do NOT push a VAE far past its trained temporal length and expect stability.** Memory chunking lets a long latent *fit*; it does not make the VAE's learned temporal statistics *valid* at that length. For genuinely long video, generate in overlapping latent windows and let the causal VAE stitch — do not just feed it a 500-frame latent and hope.

## 11. Stress-testing the design

Let me close the technical core by pushing the 3D-VAE past its comfort zone the way a real run does, because the failure boundaries are where you actually learn what the thing is.

**What happens past the VAE's trained clip length?** If it fits in memory (you chunked the decode), the immediate symptom is color drift and a slow loss of temporal coherence — the VAE is extrapolating its temporal statistics beyond training. The clip does not crash; it *degrades*, which is more dangerous because it is easy to miss until you compare frame 1 to frame 300. The fix is windowed latent generation with overlap, not a bigger decode.

**What happens when motion is large between frames?** The temporal-consistency loss and the 4× temporal compression both assume adjacent frames are *similar*. A fast pan or a cut violates that. Symptom: the VAE either smears the fast-moving region (temporal averaging) or, at a hard cut, produces a ghost of the previous scene bleeding into the next. This is why high-dynamic-degree content is genuinely harder to compress well — the redundancy prior that makes the VAE efficient is weaker. There is a real tension here with the [dynamic-degree-vs-stability gaming problem in evaluation](/blog/machine-learning/video-generation/why-video-generation-is-hard): a VAE tuned for stability can quietly suppress the very motion that makes video worth generating.

**What happens when the VAE decode — not the denoiser — is the VRAM wall?** Covered in section 8, but worth stating as a diagnostic reflex: when a video pipeline OOMs *after* the denoiser finishes, it is the decode, and the fix is `enable_tiling()` + `enable_slicing()`, not a smaller denoiser. I have seen people downsize their model to fix an OOM that was entirely in the VAE — solving the wrong problem.

**What happens if you weaken the temporal modeling?** Drop the temporal-consistency loss or the 3D discriminator and you regress toward the per-frame regime: flicker returns, fur boils, flat regions pulse. This is the cleanest demonstration that flicker is an *autoencoder* property, fixed in the VAE, not something the denoiser can clean up afterward — by the time the denoiser sees latents, the temporal-coupling decision has already been made by the VAE's architecture and losses.

**What happens at the tension between stability and motion?** This is the deepest trade-off and the easiest to get wrong. The temporal-consistency loss rewards the decoder for *not changing* pixels frame-to-frame; pushed too hard, it produces a VAE that is beautifully stable on static scenes but *smears genuine motion* — a fast-moving limb gets temporally averaged into a blur, because the loss penalized the legitimate frame-to-frame change as if it were flicker. Tune it too weak and you flicker; too strong and you lose motion. The right setting threads the needle by making the temporal term *conditional* on the input's own motion (penalize reconstruction change only where the input was static), which is why the better formulations key off optical flow or the input frame-difference rather than a flat frame-to-frame penalty. There is no free lunch here: a single scalar weight cannot perfectly separate "flicker the decoder invented" from "motion the scene actually had," and the residual is part of why high-dynamic-degree video is genuinely harder to compress cleanly than slow, cinematic footage. When you see a model that is rock-stable but oddly low-energy in its motion, suspect the VAE's temporal term turned up too high — the stability you are admiring was bought with the motion you are missing.

## 12. Tying it back to the stack

Step back to the series' recurring frame: **video = (spatial generation) × (temporal coherence) under a brutal compute budget**, and the stack is **data → causal 3D-VAE → spacetime-patch DiT denoiser → flow-matching sampler → conditioning → frames**. This post is about the second box, and the argument has been that it is the *load-bearing* box. The causal 3D-VAE is where temporal coherence is first won or lost (the flicker decision), where the compute budget is set (the token count that the denoiser's quadratic attention spends), and where clip length is gated (the decode memory wall and the trained temporal range). Everything downstream inherits the VAE's choices. A great denoiser on a bad VAE gives you an expensive flickering model; a good VAE makes even a modest denoiser affordable and stable.

That is why, when I sit down to design or debug a video pipeline, the VAE is the *first* thing I look at, not the last. The denoiser gets the headlines and the parameter count; the VAE gets the leverage. Internalize the one-line law from section 5 — token count is inversely proportional to volume compression, attention cost inversely to its square — and you will never again be surprised by where your seconds-per-clip and your gigabytes went. They went to the autoencoder, and the autoencoder is where you get them back.

To bring our running example all the way home: that five-second 720p dog clip began as 135 MB of pixels across 49 frames — far too much for a denoiser to touch directly, and certain to flicker if decoded frame-by-frame. The causal 3D-VAE turned it into a `[16, 13, 60, 90]` latent of about a million values, a 48× value and 256× volume compression, which is what let a 5-billion-parameter denoiser run it on a single consumer GPU in minutes rather than the better part of an hour. The causal first frame kept the dog crisp from frame one; the temporal-consistency loss and 3D discriminator kept its fur from boiling; and when we asked for a longer clip, `enable_tiling()` and `enable_slicing()` kept the decode from OOMing where a monolithic decode would have died at second six. Every one of those wins traces to a single component. Change the denoiser and you change the headline; change the VAE and you change the whole economics of the run. That asymmetry — small box, enormous leverage — is the thing to carry into every other post in this series.

The next post in the foundations track, [representing video redundancy and tokens](/blog/machine-learning/video-generation/representing-video-redundancy-and-tokens), zooms out to *why* this compression is even possible — the temporal-redundancy and bitrate argument that the VAE exploits. After that, [from image diffusion to video diffusion](/blog/machine-learning/video-generation/from-image-diffusion-to-video-diffusion) takes the latent the VAE produces and builds the denoiser that operates on it. And the whole thing comes together in the [building-with-video-generation playbook](/blog/machine-learning/video-generation/building-with-video-generation-the-playbook), where the VAE configuration is one of the first decisions in the production pipeline. If you only remember one thing from this post: **before you tune anything else, get the VAE right — it is the real bottleneck, and the real lever.**

## Key takeaways

- **A per-frame image VAE is wrong for video on two counts**: the latent is still T× too deep (it ignores temporal redundancy) and frames decode independently (flicker). A causal 3D-VAE fixes both at once.
- **The compression ratio is architectural, not clip-dependent**: $R = f_t f_s^2 \cdot 3/C$. The CogVideoX recipe (4×, 8×8, 16 channels) gives ~48× value / 256× volume compression — the point the open field converged on.
- **The VAE sets the denoiser's bill.** Token count is inversely proportional to the VAE's volume compression; attention cost is inversely proportional to its *square*. Doubling compression cuts attention FLOPs ~4×. The VAE, not the DiT, is the budget.
- **Causal temporal convolution is the key enabler**: padding only on the past makes latent frame $t$ depend on frames $\le t$, which keeps the first frame crisp, allows arbitrary-length and streaming encode, and makes chunked decode exact.
- **Five losses balance the VAE**: reconstruction + LPIPS + adversarial (3D discriminator) + a microscopic KL + a temporal-consistency term. The temporal term and the 3D discriminator are what kill flicker.
- **More compression = cheaper denoiser but harder reconstruction.** Know the four failure signatures: flicker (weak temporal modeling), detail loss (too-high ratio), first-frame/boundary artifact (causal handling), and color drift (accumulated bias over long clips).
- **Clip length is gated by the VAE decode, not the denoiser.** The 3D decoder's upsampling activations OOM first; `enable_tiling()` + `enable_slicing()` hold peak VRAM nearly constant at the cost of 10–25% decode time. When a pipeline OOMs *after* sampling finishes, it is the decode.
- **Memory chunking and distributional length-generalization are different walls.** Tiling lets a long latent fit; only windowed latent generation keeps the VAE's temporal statistics valid past its trained length.

## Further reading

- Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models* (2022) — the latent-diffusion foundation the video VAE generalizes; pairs with our [latent diffusion post](/blog/machine-learning/image-generation/latent-diffusion-and-stable-diffusion).
- Yang et al. (THUDM), *CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer* (2024) — the reference open causal 3D-VAE and the source of the 4×8×8 / 16-channel recipe.
- Kong et al. (Tencent), *HunyuanVideo: A Systematic Framework for Large Video Generative Models* (2024) — the converged recipe at 13B scale, with VAE training details.
- *Wan 2.x technical report* (Alibaba, 2025) — a tunable causal-3D-VAE + DiT + flow-matching open model; good case study in choosing your compression point.
- *LTX-Video* (Lightricks, 2024) — aggressive compression for near-real-time generation; the latency-first end of the trade-off frontier.
- Zhang et al., *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (LPIPS)* (2018) — the perceptual loss that restores texture in VAE training.
- 🤗 `diffusers` documentation, `AutoencoderKLCogVideoX` and video-pipeline memory-optimization guides (`enable_tiling`, `enable_slicing`, `enable_model_cpu_offload`) — the official reference for the code in this post.
- Within series: [why video generation is hard](/blog/machine-learning/video-generation/why-video-generation-is-hard) (foundation), [representing video redundancy and tokens](/blog/machine-learning/video-generation/representing-video-redundancy-and-tokens) (sibling), [from image diffusion to video diffusion](/blog/machine-learning/video-generation/from-image-diffusion-to-video-diffusion), and the [building-with-video-generation playbook](/blog/machine-learning/video-generation/building-with-video-generation-the-playbook) (capstone).
