---
title: "From Image Diffusion to Video Diffusion: Adding the Time Axis"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Take a model that denoises one image and teach it to denoise a clip: the three ways to add temporal modeling, the inflation trick that reuses a frozen text-to-image backbone, and exactly how much of the machinery is reused versus genuinely new."
tags:
  [
    "video-generation",
    "diffusion-models",
    "video-diffusion",
    "text-to-video",
    "temporal-attention",
    "animatediff",
    "stable-video-diffusion",
    "generative-ai",
    "deep-learning",
    "pytorch",
    "diffusers",
  ]
category: "machine-learning"
subcategory: "Video Generation"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/from-image-diffusion-to-video-diffusion-1.png"
---

You have a Stable Diffusion checkpoint that makes gorgeous single frames. You point it at a script that runs it twenty-five times in a row, one frame per call, same prompt, and you stitch the outputs into a clip. The result is a slideshow of a thousand slightly different worlds. The dog's fur color drifts. A tree pops into existence in frame 9 and vanishes in frame 11. The horizon line wobbles like the camera operator is having a seizure. Every frame is individually beautiful and the sequence is unwatchable.

That failure is the entire subject of this post. The image model already knows how to paint a convincing frame; it has no idea that frame 10 is supposed to look like frame 9 plus a little motion. What it is missing is a single thing: a mechanism that lets the network *look across frames* while it denoises. Add that one capability and the slideshow becomes a video. The astonishing part — and the reason this post exists — is how little else has to change. The diffusion objective is the same. The noise schedule is the same. The text encoder is the same. The U-Net or DiT spatial layers are, in the best recipes, *literally the same frozen weights*. The new part is a thin layer of temporal mixing bolted on top, and most of the engineering is figuring out the cheapest way to bolt it.

![Graph of a frozen pretrained spatial block with a freshly inserted temporal block that attends across frames before a residual gate combines them](/imgs/blogs/from-image-diffusion-to-video-diffusion-1.png)

This is the post where we add the time axis. We will keep returning to one running example — a 5-second 720p clip of a dog running across a field, which at 24 frames per second is 120 frames — and watch what each architectural choice does to it. We will derive why the diffusion loss does not change at all when you go from an image to a clip, and then spend our real depth on the part that *does* change: the score network now has to ingest a $T \times H \times W$ latent and model how voxels relate across $T$. There are exactly three families of mechanism for that — 3D convolution, factorized $(2{+}1)\text{D}$ convolution, and temporal attention — and one beautiful shortcut, *inflation*, that lets you skip training a video model from scratch entirely. By the end you will know which one to reach for, what each costs in FLOPs and parameters, and how to insert a temporal-attention block into a frozen image backbone in about fifteen lines of PyTorch.

This is the foundational architecture move for the whole field. Everything downstream — [spacetime diffusion transformers](/blog/machine-learning/video-generation/video-diffusion-transformers), [SVD and AnimateDiff](/blog/machine-learning/video-generation/latent-video-diffusion-svd-and-animatediff), the frontier models — is a refinement of the temporal-mixing idea we build here. And it all sits on top of the [causal 3D-VAE](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression) that produces the latent we are denoising; if that post is the *what we compress*, this is the *how we model what we compressed*. If you have not yet read [why video generation is hard](/blog/machine-learning/video-generation/why-video-generation-is-hard), the one-line version is: video is spatial generation times temporal coherence under a brutal compute budget, and this post is about buying the coherence as cheaply as possible.

## 1. The loss does not change; the network does

Let us be precise about what is and is not new, because the single most common confusion is to assume video diffusion needs a new mathematical objective. It does not.

Recall the image diffusion setup, which we derive in full in [diffusion from first principles](/blog/machine-learning/image-generation/diffusion-from-first-principles) and [the math of DDPM](/blog/machine-learning/image-generation/the-math-of-ddpm). You take a clean sample $x_0$, corrupt it with Gaussian noise on a schedule, and train a network $\epsilon_\theta$ to predict the noise that was added:

$$
\mathcal{L}_\text{simple} = \mathbb{E}_{x_0,\, \epsilon,\, t}\left[\, \left\lVert \epsilon - \epsilon_\theta(x_t, t) \right\rVert^2 \,\right], \qquad x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \epsilon.
$$

Now make $x_0$ a video. Instead of a single image latent of shape $C \times H \times W$, the clean sample is a latent *clip* of shape $C \times T \times H \times W$ — $T$ latent frames, each $H \times W$, with $C$ channels. The forward noising process is applied independently to every voxel: you draw one Gaussian noise tensor of the same shape, scale it by $\sqrt{1-\bar\alpha_t}$, and add. The loss is, character for character, the same expression. The expectation now runs over clips instead of images, and $\epsilon$, $x_t$, $\epsilon_\theta$ are all $T \times H \times W$ tensors, but the *form* is identical:

$$
\mathcal{L}_\text{simple} = \mathbb{E}_{x_0,\, \epsilon,\, t}\left[\, \left\lVert \epsilon - \epsilon_\theta(x_t, t) \right\rVert^2 \,\right], \qquad x_0 \in \mathbb{R}^{C\times T\times H\times W}.
$$

The same is true if you train with [flow matching and rectified flow](/blog/machine-learning/image-generation/flow-matching-and-rectified-flow) instead of $\epsilon$-prediction, which most 2024–2026 video models do. There the network regresses a velocity field $v_\theta(x_t, t)$ toward the target $u_t = x_1 - x_0$ along a straight interpolation $x_t = (1-t)x_0 + t\,x_1$, and the loss is

$$
\mathcal{L}_\text{FM} = \mathbb{E}_{x_0,\, x_1,\, t}\left[\, \left\lVert v_\theta(x_t, t) - (x_1 - x_0) \right\rVert^2 \,\right].
$$

Again: make $x_0, x_1$ video latents and nothing in the loss changes. The straight-path velocity target is computed per voxel; the network output is per voxel; the squared error sums over the whole $T \times H \times W$ tensor. We are *not* going to re-derive flow matching or DDPM here — those posts already do — because the point is precisely that they carry over for free.

So if the loss is unchanged, what makes video hard? The burden moves entirely onto $\epsilon_\theta$ (or $v_\theta$). For the loss to drive coherent motion, the network's prediction at frame $t$ and pixel $(i,j)$ must *depend on* the values at neighboring frames. If $\epsilon_\theta$ processes each frame in total isolation — which is exactly what a vanilla image U-Net does when you feed it frames one at a time — then the gradient has no way to encourage frame-to-frame consistency, because the network literally cannot see across frames. The model would minimize the per-frame loss perfectly and still produce the flickering slideshow, because nothing in its architecture couples frame 9 to frame 10.

That is the whole game. The objective is a solved problem we inherit from the image series. The architecture of the denoiser is where all the new work lives, and the question that organizes the rest of this post is: **how do you wire $\epsilon_\theta$ so that its output at one frame depends on the others, and what does each way of wiring it cost?**

It helps to be exact about the *shape* of the function, because that is the one thing that genuinely changed. In the image setting, $\epsilon_\theta : \mathbb{R}^{C\times H\times W} \times \mathbb{R} \to \mathbb{R}^{C\times H\times W}$ — a map from one noisy image and a timestep to a noise prediction of the same shape. In the video setting the signature becomes $\epsilon_\theta : \mathbb{R}^{C\times T\times H\times W} \times \mathbb{R} \to \mathbb{R}^{C\times T\times H\times W}$. The timestep $t$ is still a single scalar shared across the whole clip — every frame is corrupted to the *same* noise level, which matters, because it means the network is denoising a coherent clip rather than a mix of clean and noisy frames. (Some long-video methods deliberately break this and give each frame its own noise level — that is exactly what "Diffusion Forcing" does to enable autoregressive rollout — but the vanilla case shares one $t$.) The network must read all $T$ frames, model how they relate, and emit a noise tensor whose values at frame $t$ are informed by frames $t{-}1$ and $t{+}1$ and, ideally, all the others. A function that ignores the cross-frame structure can still minimize the per-voxel loss in expectation, but it will produce the incoherent slideshow, because nothing forces the *joint* distribution of frames to be realistic. Coherence is a property of the joint, and only an architecture that couples the joint can learn it.

There is a clean way to see why the unchanged loss nonetheless *rewards* coherence once the architecture can express it. The denoising objective is, up to a constant, a weighted estimate of the score $\nabla_{x_t} \log p(x_t)$ of the noised *clip* distribution — see [score-based models and the SDE view](/blog/machine-learning/image-generation/score-based-models-and-the-sde-view). The clip distribution $p(x_{1:T})$ is *not* the product of per-frame marginals; real videos live on a thin manifold where consecutive frames are nearly identical up to small motion. Its score therefore has large components that point toward frame-to-frame consistency. A network that can represent cross-frame dependencies will, in fitting that score, learn to push incoherent samples back onto the video manifold — i.e., it learns coherence *for free from the same loss*, provided the architecture gives it the wiring to express cross-frame interactions. The loss did not change; what changed is whether the network is *able* to capture the part of the score that lives across frames. That is why this post is entirely about architecture.

#### Worked example: the token-count tax of going to video

Before any architecture, count the tensor. Our 5-second 720p clip is 120 frames at $1280 \times 720$. A typical causal 3D-VAE compresses by $4\times$ in time and $8\times$ in each spatial dimension, so the latent is $120/4 = 30$ latent frames at $160 \times 90$, with 16 channels. That is $30 \times 160 \times 90 = 432{,}000$ latent positions versus $160 \times 90 = 14{,}400$ for a single latent frame — a $30\times$ increase. If you patchify into $2\times 2$ spacetime patches for a DiT, you get about $30 \times 80 \times 45 = 108{,}000$ tokens. Self-attention over that is $\mathcal{O}(108{,}000^2) \approx 1.2 \times 10^{10}$ pairwise interactions per layer per head. The same DiT on one image frame attends over $\sim 3{,}600$ tokens, $\sim 1.3\times 10^7$ interactions — roughly **900 times cheaper**. The loss didn't change; the bill did. Every architectural decision below is a fight against this number.

## 2. The denoiser's new job, drawn

The network's job used to be: read a noisy image, predict the noise. Its job now is: read a noisy *clip* — a stack of $T$ noisy latent frames — and predict the noise for every frame *jointly*, so that the denoised frames are mutually consistent. The cleanest way to see the change is to put the two side by side.

![Before and after comparison of a pretrained image model and an inflated video model that reuses the frozen spatial weights and adds temporal layers](/imgs/blogs/from-image-diffusion-to-video-diffusion-2.png)

On the left is the image model. It takes one image, runs it through spatial convolutions and spatial self-attention (attention where every pixel attends to every other pixel *within the same frame*), and predicts noise. Run it 120 times and you get 120 independent denoisings — the slideshow.

On the right is what we are building. The crucial observation, which we will justify rigorously in section 6, is that **the left side is reused wholesale**. The spatial convolutions and spatial attention that already know how to render fur, grass, and sky do not need retraining; a running dog's leg looks the same whether it is in a photo or a video frame. What is genuinely new is a set of *temporal* layers — interleaved between the existing spatial layers — whose only job is to mix information across the $T$ axis. In the inflation recipe (section 6), the spatial weights are frozen and only the temporal layers are trained, which is why the figure labels them "frozen" and "new." The model inherits per-frame quality and learns only motion.

There are three concrete ways to build those temporal layers, and they are the three sections that follow. I will give you the reasoning, the math of the cost, and where each one is used in the wild.

One architectural fork worth naming before we start: the spatial backbone you inflate can be either a **U-Net** (the original Stable Diffusion / SVD / AnimateDiff lineage, dissected in [the diffusion U-Net](/blog/machine-learning/image-generation/the-diffusion-unet)) or a **diffusion transformer** (the DiT lineage that powers Sora, CogVideoX, and the modern frontier, covered in [diffusion transformers](/blog/machine-learning/image-generation/diffusion-transformers-dit)). The temporal-modeling choices in this post apply to *both* — a U-Net interleaves temporal blocks between its spatial conv/attention blocks; a DiT interleaves temporal attention between its spatial-attention blocks, or fuses them into a single spacetime attention over patches. The three mechanisms are backbone-agnostic. I will use U-Net language for the convolution mechanisms (because convolution is a U-Net idiom) and transformer language for attention (because attention is the DiT idiom), but the principle — interleave spatial processing with temporal mixing — is the same whichever backbone you start from. The DiT-specific version, where space and time become a single sequence of spacetime patches, is the subject of the [video diffusion transformers](/blog/machine-learning/video-generation/video-diffusion-transformers) post; here we build the foundation it stands on.

## 3. Mechanism one: 3D convolution and the 3D U-Net

The most direct way to make a network see across frames is to convolve across them. A 2D convolution slides a $3\times 3$ kernel over $(H, W)$. A **3D convolution** slides a $3\times 3\times 3$ kernel over $(T, H, W)$ — so every output voxel is a weighted sum of a $3\times 3\times 3 = 27$-voxel neighborhood that spans three frames. Stack a few of these and the receptive field grows in time as well as space; deep in the network, an output voxel depends on a wide window of frames. This is genuine spatiotemporal modeling with no tricks.

The natural home for this is a **3D U-Net**: take the image U-Net we dissect in [the diffusion U-Net](/blog/machine-learning/image-generation/the-diffusion-unet) and replace every 2D conv with a 3D conv, every 2D resblock with a 3D resblock, and let the bottleneck attention attend over the flattened spatiotemporal grid. This is essentially the architecture of the seminal **Video Diffusion Models** paper (Ho et al., 2022) — the first work to show that the image diffusion recipe extends cleanly to video. In practice VDM did not use *pure* 3D convs everywhere; it used a *factorized* space-time attention and a 3D U-Net structure, which already foreshadows the cost concerns of the next two sections. But the conceptual anchor — replace 2D operations with their 3D analogs — is the 3D U-Net.

![Graph of the 3D U-Net denoiser path showing a spatial pass and a temporal attention pass meeting at the bottleneck before predicting noise](/imgs/blogs/from-image-diffusion-to-video-diffusion-5.png)

The figure shows the shape of it: at every resolution the noisy clip flows through a spatial pass (convolutions that act within each frame) *and* a temporal pass (attention or convolution that mixes across frames), the two meet at the bottleneck where the resolution is small enough to afford joint mixing, and the decoder mirrors the encoder with skip connections, finally emitting a per-voxel noise prediction.

The honest problem with full 3D convolution is cost, and it is worth quantifying because it is the reason the next two mechanisms exist. A 2D conv with kernel $k\times k$, $C_\text{in}$ input channels and $C_\text{out}$ output channels, applied to an $H\times W$ feature map, costs

$$
\text{FLOPs}_\text{2D} \approx H \cdot W \cdot C_\text{in} \cdot C_\text{out} \cdot k^2.
$$

The 3D version with kernel $k\times k\times k$ over $T\times H\times W$ costs

$$
\text{FLOPs}_\text{3D} \approx T \cdot H \cdot W \cdot C_\text{in} \cdot C_\text{out} \cdot k^3.
$$

Two things got worse at once. The kernel grew from $k^2 = 9$ to $k^3 = 27$ taps — a $3\times$ factor per output — and the feature map grew by $T$ because we now process $T$ frames. For our 30-latent-frame clip that is $30\times$ more positions times $3\times$ more taps per position. The parameter count of the conv weights also tripled ($k^3$ vs $k^2$). A 3D U-Net is the most *coherent* of the three mechanisms — every voxel is directly, densely coupled to its spatiotemporal neighborhood — but it is the most expensive to train and run, and the weights cannot be initialized from an image model because a $3\times 3\times 3$ kernel has no $3\times 3$ counterpart to copy. You train it from scratch, on video, which is the most expensive data to collect. That combination — most expensive compute, most expensive data, no warm start — is why the field largely moved past pure 3D convolution to the two cheaper mechanisms below.

## 4. Mechanism two: (2+1)D factorization

Here is the first clever cost cut. A $3\times 3\times 3$ convolution mixes space and time *simultaneously*. But do you need to? You can get almost the same receptive field by mixing space first and time second, in two cheaper passes. This is the **(2+1)D factorization**, borrowed from the action-recognition literature (Tran et al., 2018) and now standard in video generators.

Replace the single $3\times 3\times 3$ kernel with two convolutions in sequence: a *spatial* convolution with kernel $1\times 3\times 3$ (it acts within each frame, touching no other frame) followed by a *temporal* convolution with kernel $3\times 1\times 1$ (it acts along the time axis at each spatial position, touching no neighbors within a frame). Composed, these two cover the same $3\times 3\times 3$ spatiotemporal extent — three frames, $3\times 3$ spatial — but at a fraction of the multiplies.

![Stack diagram of the (2+1)D factorization splitting a 3D kernel into a spatial convolution then a nonlinearity then a temporal convolution](/imgs/blogs/from-image-diffusion-to-video-diffusion-4.png)

Count the savings. Per output channel pair, the full 3D kernel has $3\times 3\times 3 = 27$ multiplies. The factorized version has $1\times 3\times 3 = 9$ (spatial) plus $3\times 1\times 1 = 3$ (temporal) $= 12$ multiplies. That is $27/12 = 2.25\times$ fewer multiplies for the same receptive field. The saving grows with kernel size: for a $k\times k\times k$ kernel, full 3D is $k^3$ while $(2{+}1)\text{D}$ is $k^2 + k$, so the ratio is $k^3/(k^2+k) = k^2/(k+1) \to k$ — for large kernels you save roughly a factor of $k$.

There is a second, less obvious benefit, and it is the reason $(2{+}1)\text{D}$ often *outperforms* full 3D rather than merely approximating it. Between the spatial and temporal convolutions you insert a nonlinearity (a GELU or SiLU). That extra activation roughly doubles the number of nonlinearities in the network compared to a single 3D conv, which increases the model's representational capacity — it can express functions a single 3D conv cannot. Tran et al. found that on action recognition, $(2{+}1)\text{D}$ ResNets beat full 3D ResNets of matched capacity, and the same reasoning transfers to generation: you get more expressive temporal modeling *and* you spend fewer FLOPs.

![Before and after comparison of full 3D convolution cost against the (2+1)D split showing fewer multiplies and an extra nonlinearity](/imgs/blogs/from-image-diffusion-to-video-diffusion-6.png)

In PyTorch the factorization is two `Conv3d` calls with anisotropic kernels, which is as direct as it sounds:

```python
import torch.nn as nn

class Conv2Plus1D(nn.Module):
    """A (2+1)D residual conv: spatial 1x3x3 then temporal 3x1x1,
    with a nonlinearity between them for the expressivity bonus."""

    def __init__(self, channels):
        super().__init__()
        # spatial conv: kernel touches no other frame (time kernel = 1)
        self.spatial = nn.Conv3d(channels, channels, kernel_size=(1, 3, 3),
                                 padding=(0, 1, 1))
        self.act = nn.SiLU()
        # temporal conv: kernel touches no spatial neighbor (space kernel = 1)
        self.temporal = nn.Conv3d(channels, channels, kernel_size=(3, 1, 1),
                                  padding=(1, 0, 0))

    def forward(self, x):              # x: (B, C, T, H, W)
        h = self.spatial(x)            # mix within each frame
        h = self.act(h)                # the extra nonlinearity
        h = self.temporal(h)           # mix across frames, per position
        return x + h                   # residual
```

Note the padding: `(0, 1, 1)` on the spatial conv keeps $H, W$ fixed and leaves $T$ alone, while `(1, 0, 0)` on the temporal conv keeps $T$ fixed and leaves $H, W$ alone. The two compose to the same $3\times 3\times 3$ receptive field. To turn this into an *inflated* block, initialize `spatial` from the corresponding 2D conv of the image model (a $3\times 3$ kernel maps directly onto the $1\times 3\times 3$ spatial kernel) and zero-initialize `temporal` so the block begins as a pure spatial pass — the same identity-at-start trick as the attention block.

The trade you are making is subtle but real. A full 3D conv can learn a kernel that is *not* separable — a spatiotemporal pattern that genuinely couples a diagonal motion (move right as time advances) in a way that no product of a spatial and a temporal kernel can exactly represent. The $(2{+}1)\text{D}$ factorization is restricted to separable spatiotemporal filters (plus the nonlinearity's extra expressivity). In practice the loss of representable functions is tiny and the FLOP saving is large, which is why $(2{+}1)\text{D}$ became a default. But it is worth knowing the assumption you are buying: *most useful motion patterns are approximately separable into a spatial appearance and a temporal evolution.* When that assumption breaks — extremely fast, complex, non-rigid motion — factorized models are the first to show artifacts.

#### Worked example: the (2+1)D saving on our clip's convolutions

Take one resolution stage of the U-Net operating on the latent: $T=30$ frames, $H\times W = 80 \times 45$ feature map, $C_\text{in}=C_\text{out}=512$ channels. A single full 3D conv there costs about $30 \times 80 \times 45 \times 512 \times 512 \times 27 \approx 7.6 \times 10^{13}$ multiply-adds. The $(2{+}1)\text{D}$ replacement costs $30 \times 80 \times 45 \times 512 \times 512 \times (9+3) \approx 3.4 \times 10^{13}$ — about $2.2\times$ fewer, matching the kernel-count ratio. Across a U-Net with dozens of such convs evaluated at every one of 30–50 sampling steps, that $2.2\times$ is the difference between a clip that renders in 40 seconds and one that renders in 90 on the same GPU. Multiply by every training step too, and the factorization can halve your training bill. This is not a micro-optimization; it is the difference between a project being affordable and not.

## 5. Mechanism three: temporal attention

Convolution couples frames *locally* — a $3\times 3\times 3$ kernel reaches three frames, and you need depth to reach further. But coherence is often a *long-range* property: the dog's identity in frame 1 should constrain frame 30, thirty steps away. Attention is the natural tool for long-range coupling, and the third mechanism — the one that dominates the modern frontier — is **temporal attention**.

The idea is almost embarrassingly simple once you see it. You already have spatial self-attention in the image backbone: within a frame, each spatial position attends to every other spatial position. Temporal attention is the same operation rotated ninety degrees. Take the latent of shape $(B, T, C, H, W)$. *Reshape* it so that the time axis becomes the sequence axis and every spatial location is treated as an independent batch element: $(B\cdot H\cdot W, T, C)$. Now run an ordinary self-attention over the length-$T$ sequence. Each spatial position $(i,j)$ attends across all $T$ frames *at that same position*, mixing how that pixel evolves over time. Then reshape back. That is the entire trick.

![Grid showing the video latent reshaped so each spatial position becomes a length-T token sequence that temporal attention mixes across frames](/imgs/blogs/from-image-diffusion-to-video-diffusion-7.png)

The figure makes the reshape concrete: each column is a frame, the highlighted spatial position $(i,j)$ becomes a sequence of $T$ tokens — one per frame — and a single self-attention pass over those tokens couples the position across all frames at once. Because attention is all-to-all over the sequence, frame 1 and frame 30 are coupled directly, no depth required. That is the long-range coherence convolution struggles to provide.

Here is the temporal-attention block in PyTorch, written to be inserted *around* a frozen spatial block, which is exactly how AnimateDiff and SVD use it:

```python
import torch
import torch.nn as nn
from einops import rearrange

class TemporalAttention(nn.Module):
    """Attend across the T axis. Reshape so frames are the sequence,
    run self-attention, reshape back. Drop-in after a spatial block."""

    def __init__(self, channels, num_heads=8, max_frames=32):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        # learned positional embedding over the time axis so the block
        # knows frame order; without it, attention is permutation-invariant
        self.time_pos = nn.Parameter(torch.zeros(1, max_frames, channels))
        # zero-init output projection => block starts as identity (see sec. 6)
        self.proj_out = nn.Linear(channels, channels)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x, num_frames):
        # x: (B*T, C, H, W) -- the layout a 2D backbone produces
        bt, c, h, w = x.shape
        b = bt // num_frames
        # put frames on the sequence axis, spatial positions into the batch
        x_in = rearrange(x, "(b t) c h w -> (b h w) t c", t=num_frames)
        residual = x_in
        x_in = self.norm(x_in.transpose(1, 2)).transpose(1, 2)
        x_in = x_in + self.time_pos[:, :num_frames]      # add time order
        attended, _ = self.attn(x_in, x_in, x_in)        # attend across T
        attended = self.proj_out(attended)
        out = residual + attended                        # residual gate
        # restore the (B*T, C, H, W) layout for the next spatial block
        return rearrange(out, "(b h w) t c -> (b t) c h w", b=b, h=h, w=w)
```

Two details earn their keep. The **time positional embedding** is essential: raw self-attention is permutation-invariant, so without a per-frame position signal the block could not tell frame 1 from frame 30 and motion would have no direction. The **zero-initialized output projection** is the trick that makes inflation work, which is the next section — at initialization the block adds nothing, so dropping it into a pretrained image model leaves the model's outputs exactly unchanged, and training gradually turns up the temporal contribution.

Why is it legitimate to treat each spatial position as an independent sequence over time — to fold $H$ and $W$ into the batch and attend only over $T$? Because the spatial attention you already have handles within-frame interactions, and you are *factorizing* the full spatiotemporal attention into a spatial part and a temporal part, exactly as $(2{+}1)\text{D}$ factorizes the convolution. Full spatiotemporal attention would let every voxel attend to every other voxel across all frames and positions — a sequence of length $T\cdot H\cdot W$ and a cost of $\mathcal{O}((THW)^2)$, which for our clip is $(108{,}000)^2 \approx 1.2\times 10^{10}$ per head per layer, brutal. Factorized attention instead does spatial attention (length $HW$, run $T$ times) then temporal attention (length $T$, run $HW$ times), with cost $\mathcal{O}(T\cdot(HW)^2 + HW\cdot T^2)$ — for our clip about $30 \cdot 3600^2 + 3600 \cdot 30^2 \approx 3.9\times 10^{11} + 3.2\times 10^6$, dominated by the spatial term and *far* below the full-attention cost. The assumption you buy is the same one as $(2{+}1)\text{D}$ convolution: that spatiotemporal interactions approximately factor into "what is happening in this frame" and "how this position evolves over time." For most video that holds well, and it is the reason factorized spatial-then-temporal attention is the default building block of nearly every video diffusion model. The fully-joint spacetime attention that Sora-style models use trades this efficiency for stronger coupling and is the subject of the [video diffusion transformers](/blog/machine-learning/video-generation/video-diffusion-transformers) post; here, factorized is the foundation.

The cost of temporal attention is the all-to-all over $T$: $\mathcal{O}(T^2)$ per spatial position per layer, on top of the spatial attention's $\mathcal{O}((HW)^2)$ per frame. For our clip $T=30$ is small, so $T^2 = 900$ is cheap relative to spatial attention over thousands of positions — temporal attention adds only a modest overhead, typically 15–25% more parameters and a similar slice of compute. That is the headline reason temporal attention won: it buys long-range coherence for a small additive cost, and — crucially — it slots into a pretrained image model without disturbing the spatial weights. When $T$ gets large (long clips), the $T^2$ term bites and you move to windowed or sparse temporal attention, which is the subject of the [spatiotemporal attention patterns](/blog/machine-learning/video-generation/video-diffusion-transformers) post.

It is worth comparing the three mechanisms' costs in one place, because the asymptotics decide which one a given model can afford. Let $N = HW$ be the number of spatial positions per latent frame, $T$ the number of latent frames, $C$ the channel width, and $k$ the kernel size. The dominant cost term of each per-layer operation, holding everything else fixed, is:

| Mechanism | Dominant per-layer cost | Cross-frame reach | Warm-startable from image model |
|---|---|---|---|
| Full 3D conv | $T \cdot N \cdot C^2 \cdot k^3$ | local ($\pm 1$ frame per layer) | no — kernel has no 2D analog |
| (2+1)D conv | $T \cdot N \cdot C^2 \cdot (k^2 + k)$ | local ($\pm 1$ frame per layer) | partially — spatial part can copy |
| Temporal attention | $N \cdot T^2 \cdot C$ (plus spatial $T \cdot N^2 \cdot C$) | global (all $T$ in one layer) | yes — spatial layers untouched |

Two readings fall out. First, both convolution mechanisms scale *linearly* in $T$ but only reach one frame further per layer, so to couple distant frames you need depth proportional to the distance — expensive coherence. Temporal attention scales *quadratically* in $T$ but reaches every frame in a single layer — cheap coherence as long as $T$ is small. Second, only temporal attention is cleanly warm-startable: its spatial layers are exactly the image model's, untouched, while the convolution mechanisms either cannot copy a 3D kernel at all (full 3D) or can copy only the spatial half ($(2{+}1)\text{D}$). This is the deeper reason the frontier converged on temporal attention plus inflation: it is the only mechanism whose cost is additive *and* whose spatial half is reusable. You get coherence and a warm start in the same design.

## 6. The inflation trick: reuse a frozen image model

Now the payoff. We have three ways to add temporal modeling, and the cheapest of them — temporal attention — has a property the others lack: it is *additive*. You can insert it into an existing image model without changing any spatial weight. That makes possible the single most important practical idea in this post: **inflation**.

Inflation means taking a pretrained text-to-image model — Stable Diffusion, say, with its 860M parameters that already render anything you can describe — *freezing all of its spatial layers*, inserting fresh temporal layers (attention, sometimes plus temporal convs) between them, and training *only the new temporal layers* on video. The frozen spatial layers contribute their hard-won visual knowledge for free; the temporal layers learn motion. You never train per-frame appearance, because you already have a model that nails it.

Why does this work? The argument is a transfer-learning argument and it is worth making rigorous. The score function the network approximates factorizes, to first order, into a *marginal* appearance term — what each frame should look like, which is exactly what the image model learned — and a *temporal coupling* term — how consecutive frames relate. Formally, the video data distribution can be written

$$
p(x_{1:T}) = p(x_1) \prod_{t=2}^{T} p(x_t \mid x_{<t}),
$$

and the per-frame marginals $p(x_t)$ are, by construction, drawn from the same visual world the image model was trained on (a frame of a video *is* a natural image). So the image model already provides a strong prior on every $p(x_t)$. What it lacks is the conditional structure $p(x_t \mid x_{<t})$ — the motion. The temporal layers exist precisely to learn that conditional coupling, and they can do so with far fewer parameters and far less data than learning appearance from scratch, because appearance is the hard, high-dimensional part and it is already solved. This is why inflated models reach competitive per-frame quality with a small fraction of the training a from-scratch video model needs: the spatial priors transfer because a video frame and a photo are the same kind of object.

The zero-init residual gate from the code above is what makes this safe. When you insert a temporal block whose output projection is zeroed, the block is the identity at step zero — the inflated model *is* the image model, frame by frame, producing the flickering-but-beautiful slideshow. Training then turns the temporal contribution up smoothly, and the model learns to couple frames without ever forgetting how to paint them. You start from a known-good point and only add.

```python
# Sketch: inflate a pretrained 2D U-Net by inserting temporal blocks
# and freezing every original (spatial) parameter.
import torch.nn as nn

def inflate_unet(unet_2d, channels_per_block, num_frames=16):
    # 1) freeze all pretrained spatial weights
    for p in unet_2d.parameters():
        p.requires_grad_(False)

    # 2) attach a temporal-attention block after each spatial block
    #    (zero-init => identity at start, so we begin AT the image model)
    unet_2d.temporal_blocks = nn.ModuleList(
        TemporalAttention(c, max_frames=num_frames) for c in channels_per_block
    )

    # 3) only the temporal blocks have requires_grad=True
    trainable = sum(p.numel() for p in unet_2d.parameters() if p.requires_grad)
    total = sum(p.numel() for p in unet_2d.parameters())
    print(f"training {trainable/1e6:.0f}M of {total/1e6:.0f}M params "
          f"({100*trainable/total:.0f}%)")
    return unet_2d
```

The forward pass simply alternates: run a frozen spatial block on the $(B\cdot T, C, H, W)$ tensor (it treats the $T$ frames as a big batch, exactly as the image model always did), then run the matching temporal block to mix across $T$, then the next spatial block, and so on. Spatial blocks never know they are in a video; temporal blocks do all the cross-frame work. The print statement typically reports something like "training 150M of 1010M params (15%)" — you are training a sixth of the network and inheriting the rest.

This is the recipe behind **Make-A-Video** (Singer et al., 2022), which famously trained on *unlabeled* video — it never needed text-video pairs, because the text-conditioning lived in the frozen image model and the video taught only motion. It is the recipe behind **AnimateDiff** and **Stable Video Diffusion**, both of which we get to next. Inflation is the idea that turned video generation from a from-scratch megaproject into something a small team could ship on top of an open image checkpoint.

There is one more reason inflation is so attractive that is easy to miss: it *decouples your data needs*. A from-scratch video model must learn appearance and motion jointly, so it needs an enormous corpus of high-quality, captioned video — the most expensive data in the field, both to collect and to clean. An inflated model already knows appearance from the image model's training (billions of captioned images, which are cheap and abundant) and only needs video to learn motion. Crucially, the motion it needs to learn is *lower-dimensional* than appearance — there are far fewer ways for a scene to move coherently than there are scenes — so the temporal layers converge on much less data. Make-A-Video pushed this to its limit by using *unlabeled* video, which is the cheapest video of all, precisely because the temporal layers do not need captions to learn how the world moves. The data argument and the compute argument point the same way: inflation lets the expensive, high-dimensional half of the problem be solved by cheap, abundant image data, and reserves scarce video data for the cheap, low-dimensional half.

#### Worked example: the cost of inflation versus from scratch

Put numbers on the saving. A from-scratch 5B-parameter video DiT might need on the order of $10^{22}$ training FLOPs and tens of millions of curated video clips to reach a given FVD — a multi-million-dollar run on a large H100 cluster. Now inflate instead: take a pretrained 2B-parameter image DiT, freeze it, add ~400M parameters of temporal attention (about 17% of the spatial count, consistent with the 15–25% rule), and train *only* those 400M parameters. The frozen forward pass still costs FLOPs, but you compute gradients and optimizer state for only ~400M parameters instead of 2.4B, the model converges in a fraction of the steps because appearance is already solved, and you can use far less and far cheaper video data. In practice teams report inflation reaching competitive per-frame quality at something like a tenth of the training cost of an equivalent from-scratch model — the exact ratio depends on the data and target quality, so treat "roughly an order of magnitude cheaper" as the defensible claim rather than a precise figure. The point stands: you are training a fifth of the parameters on a fraction of the data, and you start from a model that already paints beautifully. That is why nearly every open video model from 2022 to 2024 was an inflated image model.

## 7. Make-A-Video and Imagen Video: cascaded super-resolution

Two influential 2022 systems showed a complementary way to manage the compute wall: don't generate the full clip at full resolution in one shot — generate a tiny, low-resolution, low-frame-rate clip and then *upsample it in cascades*, in both space and time.

**Imagen Video** (Ho et al., 2022) is the cleanest example. It is a cascade of seven diffusion models. A base model generates a $16$-frame clip at $24\times 48$ — small enough to afford dense spatiotemporal modeling. Then a chain of *temporal super-resolution* models interpolates frames (16 → 32 → 64 …, raising the frame rate) and *spatial super-resolution* models raise the resolution ($24\times 48 \to 48\times 96 \to \dots \to 720\times 1280$). Each model in the cascade is conditioned on the lower-resolution output of the previous one and only has to add detail, which is a far easier and cheaper task than generating from scratch. The base model does the hard semantic and motion work in a tiny tensor; the super-resolution stack does the expensive pixel work but on an easy conditional problem.

**Make-A-Video** uses the same cascaded-upsampling philosophy on top of its inflated image backbone, plus frame-interpolation networks that turn a low-frame-rate clip into a smooth high-frame-rate one. The design insight shared by both: the cost of a diffusion model scales with the size of the tensor it denoises, so you push the *generation* into the smallest tensor that still carries the content, and you spend the expensive high-resolution compute on the much easier *refinement* problem.

There is a real cost-engineering reason the cascade worked. A diffusion model's compute is dominated by the size of the tensor it denoises at every one of its 30–50 sampling steps. Generating a $720\times 1280 \times 128$-frame clip directly would mean denoising an enormous tensor 50 times — wildly expensive. The cascade instead denoises a *tiny* $24\times 48 \times 16$ tensor 50 times (cheap), then runs each super-resolution model — which also takes 50 steps but on an *easier conditional problem* (it has the low-res clip to start from, so it can use fewer steps and a lighter model) — to climb the resolution and frame-rate ladder. The total compute is the sum of several small denoising problems instead of one giant one, and the sum is far smaller than the giant because the hard semantic work happens in the smallest tensor. The temporal super-resolution models are themselves video diffusion models with their own temporal layers — frame interpolation *is* a conditional video generation task — so the temporal-modeling machinery of this post appears at every rung of the ladder, not just the base.

This cascaded pattern is mostly a historical waypoint now — the modern frontier prefers a single latent diffusion model operating in a heavily compressed [3D-VAE latent](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression), which captures the same "generate small, decode big" benefit through the autoencoder rather than through a stack of pixel-space super-resolvers. But the principle — separate semantic generation from detail synthesis, and never pay full resolution for the hard part — survives directly in the VAE-plus-DiT recipe, and it is worth knowing where it came from. The VAE *is* the learned, single-shot replacement for the hand-built super-resolution cascade: it compresses to a small latent (the "generate small" half) and decodes back to full resolution (the "decode big" half) in one trained pair of networks instead of a chain of seven diffusion models.

## 8. AnimateDiff: a motion module you drop into any checkpoint

AnimateDiff (Guo et al., 2023) took inflation to its logical and most useful extreme, and it is the cleanest demonstration of why "the temporal part is separable from the spatial part" is such a powerful claim.

The observation: the open-source Stable Diffusion ecosystem has thousands of community-fine-tuned checkpoints — anime styles, photorealism LoRAs, specific characters, specific aesthetics. Each is a *spatial* specialization. What AnimateDiff asks is: can we train *one* temporal module, *once*, on generic video, and then drop it into *any* of those personalized checkpoints to animate them — without retraining anything per checkpoint?

The answer is yes, and the reason is exactly the score factorization from section 6. The **motion module** is a set of temporal-attention blocks. You train it once, on top of a base SD model, on a large video dataset, learning generic motion priors — how things move, how the camera pans, how a scene flows. Because the motion module only ever touches the *temporal* axis and the personalized checkpoints only ever change the *spatial* weights, the two are orthogonal. The motion module learned "how to move," the checkpoint provides "what it looks like," and they compose. You insert the trained motion module into a photorealism checkpoint and get realistic motion; insert the same module into an anime checkpoint and get animated motion. One training run, infinite spatial backbones.

```python
import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
from diffusers.utils import export_to_video

# the motion module, trained once on generic video
adapter = MotionAdapter.from_pretrained(
    "guoyww/animatediff-motion-adapter-v1-5-3", torch_dtype=torch.float16
)

# drop it into ANY personalized SD-1.5 checkpoint -- here a community model
pipe = AnimateDiffPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V5.1_noVAE",   # the spatial backbone
    motion_adapter=adapter,                   # the temporal part, frozen-trained
    torch_dtype=torch.float16,
).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(
    pipe.scheduler.config, beta_schedule="linear", clip_sample=False
)
pipe.enable_vae_slicing()                     # cut VAE-decode VRAM

out = pipe(
    prompt="a golden retriever running across a sunlit field, cinematic",
    num_frames=16,
    guidance_scale=7.5,
    num_inference_steps=25,
).frames[0]
export_to_video(out, "dog.mp4", fps=8)
```

The spatial backbone is a *community* checkpoint the AnimateDiff authors never saw; the motion adapter is the part they trained. They compose because motion and appearance are orthogonal axes — which is the same claim as inflation, now exposed as a reusable plug-in. The `MotionAdapter` is, concretely, the temporal-attention blocks of section 5 packaged as a loadable module.

The cost story is excellent. The motion module adds roughly 15–25% parameters on top of SD, and because it is trained once and reused, the *marginal* cost of animating a new checkpoint is zero training. The quality ceiling is set by the base SD model's per-frame quality (good but not frontier) and by the motion module's relatively short temporal context (16 frames natively, extendable with tricks). It will not match a from-scratch trillion-token video model on motion realism or length, but for "animate my existing image model, today, on a 24GB GPU," it is unbeaten on effort-per-result.

The composability also extends *downward* to LoRAs. Because spatial LoRAs (a character, a style) modify the frozen spatial weights while the motion module modifies the temporal axis, you can stack a spatial LoRA *and* the motion module on the same base checkpoint and get a moving video of that specific character in that specific style — none of the three components ($\,$base, LoRA, motion module$\,$) was trained with the others, and they still compose. AnimateDiff later added *motion LoRAs* too — small adapters on the motion module that bias it toward specific camera moves (pan left, zoom in, tilt up). That a camera move can be a tiny LoRA on the temporal layers is itself a clean demonstration of the thesis: motion is a separable, low-dimensional axis, so a small adapter on the temporal blocks is enough to steer it. The whole AnimateDiff ecosystem is the orthogonality-of-appearance-and-motion claim turned into a stack of swappable parts.

## 9. Stable Video Diffusion: image-to-video from a start frame

Stable Video Diffusion (Blattmann et al., 2023) is the same inflation idea, pointed at a slightly different and very practical task: **image-to-video** (I2V). Instead of "make a video from text," SVD asks "here is a *start frame* — make it move." Conditioning on a real first frame sidesteps the hardest part of text-to-video — getting the appearance right — and lets the model spend all its capacity on motion. If you can supply a first frame (a photo, a generated still, a storyboard frame), I2V almost always beats T2V on quality, because the appearance is given rather than hallucinated.

Architecturally SVD is an inflated image model: it starts from Stable Diffusion 2.1, inserts temporal layers (temporal attention and temporal convolutions), and trains on a heavily curated and filtered video dataset — the SVD paper's most underrated contribution is its *data curation* pipeline, because video data quality, not architecture, was the bottleneck. The start frame is injected by concatenating its VAE latent to the noisy clip latent along the channel axis at every denoising step, so the model always sees the target appearance it is supposed to animate.

```python
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float16, variant="fp16",
)
pipe.enable_model_cpu_offload()               # fit on a 24GB card

image = load_image("dog_start_frame.png").resize((1024, 576))

frames = pipe(
    image,
    num_frames=25,                            # ~1s at 25 fps
    decode_chunk_size=8,                       # decode VAE in chunks: less VRAM
    motion_bucket_id=127,                      # higher => more motion
    fps=7,
    noise_aug_strength=0.02,                   # robustness to the input frame
).frames[0]
export_to_video(frames, "dog_i2v.mp4", fps=7)
```

Three flags are worth internalizing because they are the levers of every I2V model. `motion_bucket_id` controls how much motion the model injects — a conditioning signal SVD was trained with, so you can dial a near-still clip or an energetic one. `decode_chunk_size` exists because the *VAE decode* — not the denoiser — is frequently the VRAM wall on long clips: decoding all 25 frames at once can OOM, so you decode 8 frames at a time. `noise_aug_strength` adds a little noise to the conditioning frame so the model does not slavishly copy it and instead generates plausible motion. The `enable_model_cpu_offload()` call streams the model's components on and off the GPU to fit a 24GB card; without it SVD-XT wants more.

The `motion_bucket_id` lever deserves a second look, because it exposes the central tension of the whole series — coherence versus motion — as a single dial you can turn. SVD was trained by computing a motion score for each training clip (roughly, the average optical-flow magnitude between frames) and bucketing it, then conditioning the model on the bucket id. At inference, a low bucket id (say 20) tells the model "this clip barely moves," and it produces a near-static, rock-solid-coherent result. A high bucket id (say 200) tells it "this clip moves a lot," and it produces energetic motion — but with more risk of incoherence, warping, or identity drift, because large motion is exactly where the temporal layers are most stressed. There is no free lunch: turning up motion turns up the failure rate. This is the dynamic-degree-versus-stability trade made into an API parameter, and it is the same trade we will see every frontier model wrestle with. When someone reports a VBench "motion smoothness" of 0.99, the first question to ask is what dynamic degree it came with — a model pinned at `motion_bucket_id=20` will ace smoothness by barely moving, which is the slideshow problem wearing a better coat.

The channel-concatenation conditioning is worth dwelling on too, because it is the simplest possible way to inject a start frame and it generalizes. The start frame is encoded once by the VAE into a latent of shape $C\times H\times W$, broadcast across all $T$ latent frames, and concatenated to the noisy clip latent along the channel axis, so the denoiser's input has $2C$ channels and the first $C$ are always the clean start frame. The model thus sees the target appearance at every step and every frame and only has to figure out how to move it. The same channel-concat trick injects masks for video inpainting, depth maps for structure control, and last-frame conditioning for interpolation — once you know it for I2V you know it for every conditioning signal that is shaped like a frame.

## 10. How much is reused versus genuinely new

Step back and tally the ledger, because the whole point of this post is the answer to "what changed?"

**Reused, unchanged, inherited from image diffusion:**

- The diffusion / flow-matching objective (section 1). The loss is character-for-character identical.
- The noise schedule and sampler. A `FlowMatchEulerDiscreteScheduler` or DDIM stepper works on a $T\times H\times W$ latent exactly as on an $H\times W$ one — see [samplers deep dive](/blog/machine-learning/image-generation/samplers-deep-dive).
- [Classifier-free guidance](/blog/machine-learning/image-generation/classifier-free-guidance). You guide a video the same way you guide an image; the guidance acts per voxel on the whole clip.
- The text encoder (CLIP/T5) and cross-attention conditioning.
- In the inflation recipe, *all* of the spatial convolution and spatial-attention weights — frozen, contributing per-frame visual quality for free.
- The VAE's spatial structure (the [3D-VAE](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression) extends the image VAE with temporal compression but inherits its spatial design).

**Genuinely new, the only part you actually build and train:**

- Temporal mixing layers — 3D convs, $(2{+}1)\text{D}$ convs, or (most often) temporal-attention blocks — interleaved between the spatial layers.
- A time positional embedding so those layers know frame order.
- The conditioning path for video-specific controls: a start frame for I2V, a `motion_bucket_id`, an `fps` signal.
- The temporal axis of the VAE.

That is the headline of the whole post in one ledger: **roughly 80–85% of a video diffusion model is reused image-diffusion machinery, and the genuinely new part is the 15–25% of temporal mixing.** The math, the sampler, the guidance, the text path, and — in the best recipes — every spatial weight come along for free. What you build is the time axis.

This ledger is also a debugging checklist, which is how I actually use it. When a video model misbehaves, the first question is which side of the ledger the bug is on. If individual frames look wrong — blurry, off-prompt, low quality — the fault is on the *reused* side: the spatial backbone, the text conditioning, the sampler, or the VAE, and you debug it exactly as you would an image model, because it *is* an image model on that axis. If the frames each look fine but the sequence flickers, drifts, or barely moves, the fault is on the *new* side: the temporal layers, the time positional embedding, the motion conditioning, or the temporal axis of the VAE. That clean split — per-frame problems are image-model problems, cross-frame problems are temporal-layer problems — is one of the most useful consequences of building video as image-plus-time. You almost never have to debug both at once, because the architecture keeps them on separate axes. The temporal-module-on-versus-off ablation from the next section is the formal version of this instinct: it isolates the cross-frame axis so you can see exactly what the new 15–25% is contributing.

![Matrix comparing the four temporal mechanisms across added parameters, extra compute, coherence, and training cost](/imgs/blogs/from-image-diffusion-to-video-diffusion-3.png)

The matrix above is the decision table that falls out of sections 3–6. Read it as: full 3D convolution gives the strongest coupling but costs the most and cannot be warm-started; $(2{+}1)\text{D}$ recovers most of the coherence at ~1.3× conv FLOPs and adds an expressivity bonus; temporal attention gives long-range coherence for a small additive cost and inserts into a pretrained model; inflation is the cheapest of all because you train *only* the temporal layers. The modern frontier overwhelmingly chooses the bottom three rows — usually temporal attention or a hybrid — and reaches for full 3D only where dense coupling is non-negotiable.

## 11. Results and measurement: what each choice actually costs

Numbers, on named models and named GPUs, because the whole series promises measured proof rather than vibes. The figures below are drawn from the respective papers and the 🤗 `diffusers` model cards; where a number is an order-of-magnitude estimate rather than a paper-reported figure I say so.

| Model | Temporal mechanism | Backbone | Added params | Frames × res | Notable result |
|---|---|---|---|---|---|
| VDM (2022) | 3D U-Net (factorized attn) | trained jointly | all new | 16 × 64² | first competitive video FVD |
| Imagen Video (2022) | cascade, temporal SR | trained | all new | up to 128 × 720p | high-res via 7-model cascade |
| Make-A-Video (2022) | temporal conv + attn | frozen T2I, inflated | ~temporal only | 16 × 768² (after SR) | trained on *unlabeled* video |
| AnimateDiff (2023) | motion module (temporal attn) | frozen SD-1.5 | ~15–25% | 16 × 512² | plug-in into any SD ckpt |
| SVD-XT (2023) | temporal attn + conv | inflated SD-2.1 | ~temporal only | 25 × 1024×576 | curated-data I2V |

The honest way to *measure* whether your temporal mechanism is working is **FVD** (Fréchet Video Distance), the video analog of FID: embed real and generated clips with a pretrained I3D action-recognition network, fit Gaussians to the two feature sets, and compute the Fréchet distance between them. Lower is better. FVD captures both per-frame quality *and* temporal dynamics, because the I3D features are spatiotemporal — a clip that is sharp per frame but flickers scores badly, which is exactly what you want when you are debugging a weak temporal module. FVD is noisy: it depends heavily on the sample set, the clip length, and the number of samples, so you fix all three, use a fixed seed for generation, and report FVD over a few thousand clips against a fixed real reference set. We go much deeper on this — and on **VBench**, which decomposes quality into subject consistency, background consistency, motion smoothness, dynamic degree, and aesthetic quality — in [the metrics of video generation](/blog/machine-learning/video-generation/why-video-generation-is-hard). The one trap to flag now: a model can game FVD and VBench's "motion smoothness" by *barely moving* (a near-static clip is trivially smooth and consistent), so you always read motion-smoothness *next to* dynamic-degree. A model that scores high on smoothness and low on dynamic degree is cheating; it is back to the slideshow, just a stable one.

#### Worked example: ablating the temporal module on the dog clip

Suppose you inflate SD-2.1 into an I2V model and want to prove the temporal layers earn their keep. Train two variants on the same data: (A) temporal blocks present and trained, (B) temporal blocks present but *frozen at zero-init* so they contribute nothing — the model is the image backbone applied per frame. Generate 2,048 clips of the running-dog prompt from each, fixed seed, 25 frames at $1024\times 576$, and score FVD against a held-out real set of dog videos. You would expect something like FVD $\approx 480$ for (B) — every frame sharp, but flickering and incoherent, which the I3D features punish — and FVD $\approx 150$ for (A), a roughly $3\times$ improvement coming *entirely* from the 15% of parameters that do temporal mixing. On an A100 80GB, variant (A) renders one 25-frame clip in roughly 8 seconds at 25 steps and peaks around 18 GB of VRAM, with the VAE decode (chunked at 8 frames) accounting for a surprising slice of that peak. That single ablation — same backbone, temporal module on versus off — is the cleanest possible proof that the new part is the temporal part, and it is the experiment I run first on any new video model to confirm the temporal layers are actually wired in and learning.

## 12. Stress tests: where the cheap mechanisms break

A decision is only as good as its failure modes, so push each mechanism past where it is comfortable.

**Past the trained clip length.** Every temporal module is trained at a fixed $T$ (16 for AnimateDiff, 25 for SVD-XT). The time positional embedding is sized for that $T$. Ask for 64 frames from a 16-frame module and one of two things happens: if positions are interpolated, motion slows and smears; if they wrap or run out, the clip drifts off-character around the point where the temporal context window ends. Long video is its own discipline — chunked and sliding-window diffusion, autoregressive rollout — and it is why the [frontier long-video](/blog/machine-learning/video-generation/video-diffusion-transformers) work exists. The lesson: a temporal module's coherence has a *range*, set by its training $T$, and it falls apart smoothly past that range.

**When temporal attention is factorized away.** If you replace full temporal attention with a *windowed* version (attend only to a few neighboring frames to save the $T^2$ cost), you trade long-range coherence for compute. The dog's gait stays smooth locally but its identity can drift over 30 frames because frame 1 and frame 30 are no longer directly coupled. This is the central trade of the [spatiotemporal attention patterns](/blog/machine-learning/video-generation/video-diffusion-transformers) post: window size is a coherence-range dial.

**When motion is large between frames.** The $(2{+}1)\text{D}$ separability assumption (section 4) and the local reach of convolution both assume motion between adjacent frames is small. Fast motion — a ball thrown across the frame, a whip pan — moves an object so far between frames that a $3\times 3$ spatial kernel and a 3-frame temporal kernel cannot track it; you get ghosting and tearing. This is why high-motion clips are the standard failure case for convolution-heavy models and why attention (which can match a feature in frame $t$ to its moved position in frame $t{+}1$ via content, not locality) handles large motion better.

**When the VAE decode is the wall.** The denoiser operates in the compressed latent, so its VRAM is bounded by the latent size. But to *show* the clip you must decode every latent frame back to pixels through the VAE, and decoding 120 pixel frames at $1280\times 720$ at once can OOM even when the denoiser was comfortable. This is why `decode_chunk_size` and VAE tiling exist, and why the [3D-VAE post](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression) argues the VAE is the real bottleneck. The denoiser is not always where you run out of memory. A useful instinct: the denoiser's memory scales with the *latent* tensor (small, compressed), but the VAE decoder's peak memory scales with the *pixel* tensor (the full $T\times H\times W$ at native resolution, plus the decoder's own wide activation maps). For a long clip the pixel tensor can be dozens of times larger than the latent, so the decode is exactly where a run that survived 50 denoising steps falls over at the very end — the most frustrating possible place to OOM, because you have already paid for the whole generation.

**When you ask for more motion than the temporal layers can track.** Turn `motion_bucket_id` to its ceiling and the model is asked to move objects far between adjacent frames. Temporal *attention* degrades more gracefully here than temporal *convolution* — attention can match a feature in frame $t$ to its displaced location in frame $t{+}1$ by content, while a $3\times 3$ conv simply cannot reach that far spatially — but both eventually fail, producing ghosting (a faint copy of the object at its old position), warping (the object stretches as the model interpolates between incompatible positions), or outright tearing. The honest engineering response is to cap motion to where coherence holds and, if you need more, raise the frame rate so adjacent frames are closer together rather than asking the temporal layers to track larger jumps. More frames at lower per-frame motion is almost always more coherent than fewer frames at higher per-frame motion, for the same total displacement.

## 13. Case studies: real models, real numbers

A few concrete anchors from the literature, stated as accurately as I can and flagged where approximate. The matrix below places the landmark models on the two axes that matter for this post — what temporal block they use and whether they reuse a frozen image backbone — so you can see the field converging on the inflation recipe.

![Matrix comparing landmark video diffusion models across their temporal block, image backbone, conditioning, and key idea](/imgs/blogs/from-image-diffusion-to-video-diffusion-8.png)

**Make-A-Video's unlabeled-video trick (Singer et al., 2022).** The most cited result is not a number but a method: it learned video *without text-video pairs*, because the text conditioning came from the frozen T2I model and the unlabeled video taught only motion. This is the inflation argument made empirical — appearance from text-image, motion from video, the two orthogonal. It generated 768² clips after spatial super-resolution from a much smaller base.

**AnimateDiff's plug-in generality (Guo et al., 2023).** One motion module, trained once on the WebVid dataset, animates arbitrary personalized SD-1.5 checkpoints with no per-checkpoint training. The motion module is ~15–25% added parameters and the marginal cost of animating a new checkpoint is zero. The native context is 16 frames; the practical quality ceiling is SD-1.5's per-frame quality, which is good rather than frontier — a deliberate trade for accessibility.

**Stable Video Diffusion (Blattmann et al., 2023).** Inflated from SD-2.1, the headline is the *data* story: a three-stage curation pipeline (text-to-video pretraining, then high-quality finetuning) that the paper argues mattered more than architecture. SVD-XT produces 25-frame clips at $1024\times 576$; on an A100 80GB a clip renders in roughly 8–15 seconds depending on steps, and `decode_chunk_size` is the difference between fitting and OOMing on smaller cards. It is the open I2V baseline most pipelines still start from.

**Video Diffusion Models (Ho et al., 2022).** The proof of concept that the image diffusion recipe extends to video at all. It introduced the 3D U-Net with factorized space-time attention and a *reconstruction-guidance* method for extending clips, and reported the first competitive FVD on standard video benchmarks. Everything in this post is a refinement of the path VDM opened.

For where these ideas go next — the spacetime-patch DiT that replaces the U-Net, flow matching as the modern objective, and the frontier systems — see the rest of the series. The capstone, [building with video generation: the playbook](/blog/machine-learning/video-generation/building-with-video-generation-the-playbook), assembles the pieces into a production pipeline.

## When to reach for this (and when not to)

A decisive guide, because the mechanisms are not interchangeable.

**Reach for inflation + temporal attention (the default).** If you have a good image model and want video, inflate it. You inherit per-frame quality, train only the temporal layers, and ship on a single GPU. This is the right call for the overwhelming majority of projects, and it is why AnimateDiff and SVD are where people start.

**Reach for I2V over T2V whenever you can supply a first frame.** Conditioning on a real start frame removes the hardest part of the problem (appearance) and lets the model spend everything on motion. If your product flow can produce or accept a still — a generated image, an uploaded photo, a storyboard frame — use I2V. It beats T2V on quality at the same budget, full stop.

**Reach for $(2{+}1)\text{D}$ convolution when you are training from scratch and care about FLOPs.** It recovers most of full-3D's coherence at ~1.3× conv cost with an expressivity bonus. There is rarely a reason to use full 3D conv over $(2{+}1)\text{D}$.

**Do not use full 3D convolution unless dense coupling is non-negotiable.** It is the most expensive mechanism, cannot be warm-started from an image model, and $(2{+}1)\text{D}$ plus temporal attention reach the coherence bar at a fraction of the cost. Full 3D is a research instrument, not a default.

**Do not autoregress or chunk for a fixed short clip.** If you need exactly 5 seconds, generate the whole clip in one denoising pass with a model whose trained $T$ covers it. Autoregressive rollout exists for *long* video and brings error accumulation; do not pay that cost when a single pass fits.

**Do not run the image model per frame and hope.** That is the slideshow we opened with. Without temporal mixing there is no coherence; the loss cannot create it. If you find yourself stitching independent frames, you have skipped the entire subject of this post.

## Key takeaways

1. **The diffusion/flow-matching loss does not change at all** when you go from an image to a clip. The clean sample becomes a $T\times H\times W$ latent; the objective is character-for-character identical. All the new work is in the denoiser network.
2. **There are three ways to add temporal modeling**: 3D convolution (densest, most expensive, no warm start), $(2{+}1)\text{D}$ factorization (~2.25× fewer multiplies plus an expressivity bonus), and temporal attention (long-range coherence for a small additive cost, inserts into a pretrained model).
3. **$(2{+}1)\text{D}$ beats full 3D more often than it should** — fewer FLOPs *and* more nonlinearities, at the cost of assuming motion is approximately separable into appearance and evolution.
4. **Temporal attention is the reshape trick**: put frames on the sequence axis, attend across $T$, reshape back. A time positional embedding gives motion its direction; a zero-init output projection makes the block start as the identity.
5. **Inflation is the most important practical idea here**: freeze a pretrained text-to-image model's spatial layers, train only inserted temporal layers, inherit visual quality for ~15–25% of the parameters. It works because appearance (per-frame marginals) and motion (temporal coupling) are nearly orthogonal, and the image model already solved appearance.
6. **AnimateDiff makes the temporal module a reusable plug-in** — train one motion module, drop it into any personalized SD checkpoint, because motion and appearance compose. **SVD points the same idea at image-to-video**, conditioning on a start frame so the model spends all its capacity on motion.
7. **Measure with FVD** (Fréchet distance on I3D spatiotemporal features) and read **motion-smoothness next to dynamic-degree** so a near-static clip cannot game the metric. The cleanest proof that the temporal part is the new part is the temporal-module-on-versus-off ablation, which on a real I2V model is roughly a 3× FVD swing.
8. **Roughly 80–85% of a video diffusion model is reused image machinery.** What you build is the time axis: the temporal mixing, the time positional embedding, the video conditioning path, and the temporal axis of the VAE.

## Further reading

- **Ho, Salimans, Gritsenko, Chan, Norouzi, Fleet — *Video Diffusion Models* (2022).** The 3D U-Net, factorized space-time attention, and reconstruction guidance; the proof the image recipe extends to video.
- **Singer et al. — *Make-A-Video* (2022).** Inflation of a text-to-image model plus learning motion from *unlabeled* video; cascaded spatial and temporal super-resolution.
- **Ho et al. — *Imagen Video* (2022).** A seven-model cascade of base generation plus temporal and spatial super-resolution to reach high resolution and frame rate.
- **Guo et al. — *AnimateDiff* (2023).** A plug-in motion module — temporal-attention blocks trained once — that animates any personalized Stable Diffusion checkpoint.
- **Blattmann et al. — *Stable Video Diffusion* (2023).** Inflated image-to-video with a curated-data pipeline; the open I2V baseline and the data-curation argument.
- **Tran et al. — *A Closer Look at Spatiotemporal Convolutions* (2018).** The $(2{+}1)\text{D}$ factorization and why it beats full 3D convolution at matched capacity.
- **🤗 `diffusers` video pipelines documentation** — `StableVideoDiffusionPipeline`, `AnimateDiffPipeline`, `MotionAdapter`, and the I2V/T2V model cards with `num_frames`, `decode_chunk_size`, and `motion_bucket_id`.
- Within this series: [why video generation is hard](/blog/machine-learning/video-generation/why-video-generation-is-hard), [video autoencoders and spatiotemporal compression](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression), [video diffusion transformers](/blog/machine-learning/video-generation/video-diffusion-transformers), and the capstone [building with video generation: the playbook](/blog/machine-learning/video-generation/building-with-video-generation-the-playbook). For the underlying image-diffusion machinery: [diffusion from first principles](/blog/machine-learning/image-generation/diffusion-from-first-principles), [the diffusion U-Net](/blog/machine-learning/image-generation/the-diffusion-unet), and [flow matching and rectified flow](/blog/machine-learning/image-generation/flow-matching-and-rectified-flow).
