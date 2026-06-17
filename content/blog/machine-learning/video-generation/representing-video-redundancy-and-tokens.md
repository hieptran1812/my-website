---
title: "Representing Video: Temporal Redundancy, Optical Flow, and the Token Budget"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Why a five-second clip is a third of a billion numbers, why adjacent frames are nearly identical, and how that redundancy — turned into a token budget — quietly decides whether a video model can be trained at all."
tags:
  [
    "video-generation",
    "diffusion-models",
    "video-diffusion",
    "temporal-redundancy",
    "optical-flow",
    "tokenization",
    "video-codecs",
    "generative-ai",
    "deep-learning",
    "text-to-video",
  ]
category: "machine-learning"
subcategory: "Video Generation"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/representing-video-redundancy-and-tokens-1.png"
---

Pick a clip you have on your phone right now: five seconds, shot at 720p, the kind of thing you would not think twice about texting to a friend. To the model that has to *generate* something like it, that clip is a brick of numbers — 120 frames, each 1280×720 pixels, each pixel three color channels — and if you multiply it out, it is **331,776,000 values**. A third of a billion numbers for five forgettable seconds. A still photo at the same resolution is 2.76 million values; the video is a hundred-and-twenty-fold heavier just by existing. Now remember that a diffusion model has to push a tensor *this size* through a heavy network not once but tens of times per sample, and millions of times during training. If you tried to run a transformer's self-attention over all of those positions, the cost would be quadratic in their count, which for a third of a billion positions is a number with seventeen digits. You would not OOM at second six. You would OOM before the first step finished allocating.

And yet video generation works. Sora makes minute-long clips, Veo renders 4K with synchronized audio, and you can run an open model like CogVideoX or Wan on a single 24 GB consumer GPU and get a coherent five-second clip in a couple of minutes. The thing that makes the impossible tensor tractable is not a cleverer network and not a faster GPU. It is a *prior* — a fact about video so obvious that it is easy to miss: **adjacent frames are nearly identical.** Frame 60 and frame 61 of your clip differ in a few thousand pixels around a moving hand or a swaying branch; the other 99% of the image is the same wall, the same sky, the same shirt. Almost all of the information in a video lives in the *first frame plus the changes*, and the changes are tiny. Every efficient video representation — every video codec since the 1990s, and every video generation model since 2022 — is built on monetizing that one fact.

![A layered stack showing raw pixels at a third of a billion values, a per-frame latent at 27 million values, a 3D latent at under 7 million values, and a token sequence of about 432 thousand tokens feeding a quadratic attention budget](/imgs/blogs/representing-video-redundancy-and-tokens-1.png)

This post is about how video is *represented* and why that choice is the most consequential decision in the whole stack. By the end you will be able to: compute the raw value count and bitrate of any clip from first principles, and feel in your bones why pixel-space training is hopeless past toy sizes; state the **temporal-redundancy argument** quantitatively — why the entropy of a frame *difference* is far below the entropy of the frame itself, and why that is the entire reason video is learnable; trace how classical **codecs** (I/P/B-frames, motion compensation, residual coding) already exploit redundancy and what generative models borrowed from them; understand **optical flow** as an explicit motion representation and why modern models mostly use it implicitly; and — the payoff — work the **token-budget** math: the formula $N \approx \frac{T}{c_t}\cdot\frac{H}{c_s}\cdot\frac{W}{c_s}$ that says how many tokens your clip becomes under compression factors $c_t$ (time) and $c_s$ (space), and why, because attention is $O(N^2)$, that single number decides whether training is even feasible. We will quantify everything, write runnable code to measure inter-frame differences and optical-flow magnitude, and tabulate the four representation choices side by side.

Where does this sit in the series' running frame? Recall the spine: **video = (spatial generation) × (temporal coherence) under a brutal compute budget**, with the stack **data → causal 3D-VAE → spacetime-patch DiT → flow-matching sampler → conditioning → frames**. This post is about the very first arrow — how you turn raw pixels into something the rest of the stack can afford to touch. It is the *why* behind the post that follows it, [video autoencoders and spatiotemporal compression](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression), which builds the 3D-VAE that does the compressing; and it sharpens the cost argument first raised in [why video generation is hard](/blog/machine-learning/video-generation/why-video-generation-is-hard). If the *latent* idea is new to you, the image series already built it: [latent diffusion and Stable Diffusion](/blog/machine-learning/image-generation/latent-diffusion-and-stable-diffusion) is the same move in one fewer dimension, and [variational autoencoders from scratch](/blog/machine-learning/image-generation/variational-autoencoders-from-scratch) is the encoder this whole argument stands on.

## A video is a tensor, and the tensor is enormous

Let us be completely concrete, because the entire post falls out of the arithmetic. A video clip, before any compression, is a four-dimensional array — a tensor — with axes

$$ V \in \mathbb{R}^{T \times H \times W \times C}, $$

where $T$ is the number of frames, $H$ and $W$ are the frame height and width in pixels, and $C$ is the number of color channels (3 for RGB). The total number of scalar values is the product

$$ \lvert V \rvert = T \cdot H \cdot W \cdot C. $$

Our running example for the whole post is a **5-second 720p clip at 24 fps**, which gives $T = 120$ frames, $H = 720$, $W = 1280$, $C = 3$. Plug in:

$$ \lvert V \rvert = 120 \cdot 720 \cdot 1280 \cdot 3 = 331{,}776{,}000 \approx 0.33\text{ billion values}. $$

Stored naively as 8-bit integers (one byte per value), that is **316 MB** for five seconds. Stored as the float16 tensors a model actually computes on, double it to **633 MB** — for one clip, before a single layer of computation. A training batch of even 8 such clips, materialized in pixel space, is 5 GB of activations before you add the network. This is why "just train on pixels" is a non-starter at video resolutions: you run out of memory holding the *data*, never mind the model.

It is worth pausing on the raw **bitrate** too, because it is the codec engineer's framing and it will matter when we talk about redundancy. Uncompressed, our clip's bitrate is

$$ \text{bitrate} = H \cdot W \cdot C \cdot 8 \text{ bits} \cdot \text{fps} = 1280 \cdot 720 \cdot 3 \cdot 8 \cdot 24 \approx 531 \text{ Mbit/s}. $$

Half a gigabit per second. A Blu-ray disc maxes out around 40 Mbit/s; a Netflix 1080p stream is 5–8 Mbit/s; a video call is often under 2 Mbit/s. The clip you would actually send your friend is something like **1 MB/s**, which is roughly **8 Mbit/s** — a *66-fold* reduction from the raw 531 Mbit/s, with no visible loss. That 66× did not come from a better entropy coder squeezing the last few percent out of each frame. It came almost entirely from exploiting the fact that **the frames barely change.** Hold onto that number; we will derive it.

The value count scales the way you would fear with every knob:

| Clip | T | H×W | Raw values | Raw bytes (uint8) |
|---|---|---|---|---|
| 1s 256×256, 8 fps (toy) | 8 | 256×256 | 1.57M | 1.6 MB |
| 5s 720p, 24 fps (our example) | 120 | 1280×720 | 331.8M | 316 MB |
| 5s 1080p, 24 fps | 120 | 1920×1080 | 746.5M | 712 MB |
| 10s 1080p, 30 fps | 300 | 1920×1080 | 1.87B | 1.8 GB |
| 5s 4K, 24 fps | 120 | 3840×2160 | 2.99B | 2.8 GB |

A ten-second 1080p clip is already **1.87 billion** values; a five-second 4K clip is nearly **3 billion**. These are not exotic targets — they are the *baseline* a commercial video model is judged against. The numbers explode multiplicatively because video has three large axes that all want to grow at once: longer ($T\uparrow$), bigger ($H,W\uparrow$), smoother ($\text{fps}\uparrow$). The compute does not just add up; it multiplies. This is the "quadratic-in-frames compute wall" the foundations post warned about, made of concrete numbers.

So the first thing to internalize is that **the data itself is the problem**, before the model. Every representation choice we discuss is, at heart, an answer to one question: *how do I avoid ever holding all of $\lvert V \rvert$ at once in the expensive part of the pipeline?* And the answer always routes through redundancy.

It is worth noting that this wall shows up even before training, in the *data pipeline*. A video dataset is stored compressed (H.264/H.265 `.mp4`), and the dataloader must *decode* it back to pixel tensors on the fly — which is itself expensive enough that production training pipelines pre-encode clips into latents once and stream the *latents*, not the pixels, to the GPU. The reason is the same arithmetic: streaming 633 MB of float16 pixels per clip across the PCIe bus at training throughput saturates the bus and starves the GPU, while streaming a 6.9-million-value latent (the number we will derive) is 48× lighter and keeps the accelerators fed. So "compress first" is not only a modeling decision; it is a *data-engineering* decision forced by the same value count. Teams that skip it discover their expensive GPUs sitting idle, waiting on a CPU video decoder. The redundancy you exploit in the model, you also exploit in the disk format, the network, and the dataloader — it is the same fact wearing different hats at every layer of the stack.

#### Worked example: storing a 1-million-clip training set

Suppose you have a training corpus of 1 million 5-second 720p clips — a modest size for a serious video model. Materialized as raw float16 pixels, that is $10^6 \cdot 633\text{ MB} = 633$ **petabytes** — obviously absurd; you cannot store it, let alone read it at training speed. Stored as the source `.mp4` files at ~1 MB/s, each clip is ~5 MB, so the corpus is ~5 TB — storable, but every training step pays a CPU video-decode tax to turn `.mp4` back into pixels, and at scale that decode throughput, not the GPU, becomes the bottleneck. Now pre-encode once into 3D latents: each clip's latent is 6.9M values at float16 ≈ 13.8 MB, so the latent corpus is ~14 TB. Larger than the `.mp4` corpus on disk, but it is *already in the model's input space* — no per-step decode, no VAE encode in the training loop, and the bytes you stream to the GPU are exactly the bytes the denoiser consumes. The trade is "more disk, but the GPU never waits," and at the scale where GPUs cost more than disk, that is always the right trade. The number that made all three options thinkable or unthinkable — 633 PB vs. 5 TB vs. 14 TB — is the same value count $\lvert V \rvert$ we computed in the first section, refracted through three representations. *This is the post in one example.*

## Temporal redundancy: most of the video is already in the first frame

Here is the load-bearing observation of the entire field. Take two adjacent frames of our clip — a dog mid-stride, say — and subtract them pixel by pixel. Call the result the **frame difference** or **residual**:

$$ R_t = V_t - V_{t-1}. $$

If the camera and most of the scene are still, $R_t$ is *almost entirely zero*. A patch of blue sky is identical frame to frame; subtract it from itself and you get black. The only non-zero entries cluster where something moved — the dog's legs, a blade of grass, a flicker of shadow. Visually, $R_t$ looks like a faint sketch of edges on a black field. Quantitatively, the *fraction of pixels that change meaningfully* between adjacent frames in ordinary footage is often under 10%, and the magnitude of even those changes is small.

![A before-and-after comparison contrasting storing every full 2.76 megabyte frame against storing one keyframe plus tiny residuals that compress to about 40 kilobytes each](/imgs/blogs/representing-video-redundancy-and-tokens-2.png)

The right way to make this precise is **entropy**. Entropy measures the average number of bits needed to encode a value drawn from some distribution — it is the information content. For a single frame $V_t$, the pixels are spread across the whole intensity range; the per-pixel entropy is high, on the order of 5–7 bits for natural images. For the *residual* $R_t$, the distribution is sharply peaked at zero — most entries are exactly or nearly zero — so its entropy is far lower, often **well under 1 bit per pixel** for slow-motion content, climbing toward the frame's own entropy only when motion is large or a scene cut occurs. The inequality that justifies every video codec and every video latent is

$$ H(R_t) \;=\; H(V_t \mid V_{t-1}) \;\ll\; H(V_t). $$

Read it aloud: the entropy of the *change* equals the entropy of the next frame *given the previous one*, and that conditional entropy is much smaller than the unconditional entropy of the frame on its own. Knowing $V_{t-1}$ tells you almost everything about $V_t$. In information-theoretic terms, adjacent frames share enormous **mutual information** $I(V_t; V_{t-1}) = H(V_t) - H(V_t \mid V_{t-1})$, which is large precisely because the conditional entropy is small.

This is the **prior** that makes video learnable. A model that had to independently learn the full distribution of every frame would be solving 120 separate image-generation problems and then somehow stitching them coherently — hopeless. Instead, the structure of the problem is: learn the distribution of the *first* frame (an image-generation problem, which we know how to do), and then learn the distribution of *changes* conditioned on what came before (a much lower-entropy problem). The whole reason a video model has any hope of fitting in memory and converging in finite compute is that the per-frame *new* information is tiny. Compression is not a trick we bolt on afterward — it is the *recognition that the information was never as large as the tensor*.

We can sanity-check the redundancy claim with a back-of-envelope bit count. If a raw frame carries $H \cdot W \cdot C \cdot 8 = 22.1$ Mbit, and a typical compressed stream spends about $8\text{ Mbit/s} / 24\text{ fps} = 0.33$ Mbit per frame *on average*, then the codec is representing each frame in roughly $0.33/22.1 \approx 1.5\%$ of its raw size. Keyframes cost more and predicted frames far less, but averaged out, **98.5% of the raw bits were redundant** — predictable from neighbors. That 1.5% is the *actual* information rate of the video, and it is the rate a good representation should aim for.

There is a clean way to state the *lower bound* this implies. The total information in a clip is, by the chain rule of entropy, the information in the first frame plus the conditional information of each subsequent frame given its predecessor:

$$ H(V_1, \dots, V_T) = H(V_1) + \sum_{t=2}^{T} H(V_t \mid V_{1:t-1}) \;\le\; H(V_1) + \sum_{t=2}^{T} H(V_t \mid V_{t-1}). $$

The first term is one image's worth of bits; each term in the sum is a *residual* worth of bits, which we established is a small fraction of a frame. So the total information in a 120-frame clip is approximately *one full frame plus 119 small residuals* — not 120 full frames. If a residual averages 1.5% of a frame, the clip's information content is about $1 + 119 \cdot 0.015 \approx 2.8$ frames' worth of bits, for 120 frames of video. That is a **43× redundancy factor** baked into the content, and it is the ceiling any representation is chasing. A 3D-VAE at 4×8×8 with a 16-channel latent (vs. 3 input channels) achieves a $4 \cdot 8 \cdot 8 \cdot 3 / 16 = 48\times$ value-count reduction — right at that information-theoretic ceiling, which is not a coincidence: the compression ratio is *engineered* to match the redundancy the content actually has.

#### Worked example: the entropy of a still vs. moving clip

Take two 5-second 720p clips. Clip A is a locked-off shot of a coffee cup on a desk — nothing moves but a wisp of steam. Clip B is the same duration but it is a fast pan across a crowd. Measure the mean absolute frame difference (we will code this below). For Clip A you might find a mean residual magnitude of ~1.2 (out of 255) and fewer than 2% of pixels changing by more than 8 levels; the residual entropy is perhaps 0.3 bit/pixel. For Clip B, a large camera pan makes *almost every* pixel change — the mean residual jumps to ~30, most pixels move, and the residual entropy climbs toward 4–5 bit/pixel, approaching the frame's own entropy. The lesson is sharp: **redundancy is not a constant; it is a property of the motion.** A still clip is ~95% redundant and compresses to almost nothing; a high-motion clip is far less redundant and is genuinely expensive to both code and generate. This is the same fact that, downstream, makes high-"dynamic-degree" clips harder for generative models to keep coherent — the prior you are leaning on is weaker exactly when motion is large.

## Codecs got here first: I-frames, motion compensation, and residuals

Before any of this was a machine-learning concern, video *codecs* — MPEG-2, H.264/AVC, H.265/HEVC, AV1 — were built entirely around temporal redundancy, and the generative-model designs we use today borrow their core idea almost verbatim. It is worth understanding the codec because it is the cleanest existing proof that "first frame + changes" is the right factorization of a video.

![A branching dataflow showing an I-frame coded in full, P-frames and B-frames predicting from it via motion vectors, residual coding of the leftover error, and a compressed stream roughly sixty times smaller](/imgs/blogs/representing-video-redundancy-and-tokens-3.png)

A codec splits frames into three roles. An **I-frame** (intra-coded, the "keyframe") is compressed entirely on its own, like a JPEG — no reference to other frames. It is expensive in bits but self-contained, and it is the anchor a decoder can seek to. A **P-frame** (predicted) is *not* stored as pixels at all; it is stored as a set of **motion vectors** plus a small residual. The encoder divides the frame into blocks (say 16×16), and for each block it searches the previous frame for the best-matching block and records only the *offset* — "this block is the same as the one 7 pixels left and 2 pixels up in the last frame." This is **motion compensation**: instead of re-describing the moving object, you describe *where it moved from*. Whatever the prediction gets wrong — lighting changes, newly revealed background, deformation — is captured in a small **residual** that is itself transform-coded and quantized. A **B-frame** (bi-directional) predicts from *both* a past and a future reference frame, which lets it interpolate motion even more cheaply; it is typically the smallest of the three.

The arithmetic of why this wins is exactly the redundancy inequality. An I-frame pays the full $H(V_t)$; a P-frame pays roughly $H(\text{motion vectors}) + H(R_t)$, and since the residual entropy $H(R_t)$ is tiny and motion vectors are a few hundred numbers per frame instead of millions of pixels, the P-frame is an order of magnitude cheaper. A typical GOP (group of pictures) might be one I-frame followed by a dozen P- and B-frames, so the *amortized* cost per frame is dominated by the cheap predicted frames. That is where the 66× we computed earlier comes from: one expensive anchor, many cheap deltas.

Make the GOP arithmetic concrete. Suppose a 24-frame GOP (one second of our clip) spends 200 KB on its single I-frame and 15 KB on each of the other 23 predicted frames. The GOP costs $200 + 23 \cdot 15 = 545$ KB for one second of video — versus $24 \cdot 2.76 = 66$ MB raw, a 121× reduction, with the I-frame alone accounting for 37% of the compressed bits despite being one of 24 frames. Shorten the GOP (more frequent I-frames) and you get better seek and error resilience but pay more bits; lengthen it and you save bits but a decode error propagates farther. That GOP-length knob is *exactly* the chunk-length knob in autoregressive video generation: a shorter chunk re-anchors more often (more robust to drift, more expensive), a longer chunk amortizes the anchor (cheaper, but identity drifts farther before it re-grounds). The codec engineer and the video-model engineer are tuning the same trade-off under different names, because they are exploiting the same redundancy with the same "anchor plus changes" factorization.

Three ideas from the codec carry straight into generative video, and it is worth naming them precisely because you will see each one reappear:

1. **Anchor plus changes.** The keyframe-and-residual factorization is exactly the structure of **image-to-video (I2V)** generation: condition on a first frame (the anchor), generate the motion (the changes). It is also the structure of autoregressive long-video rollout — generate a chunk, then generate the next chunk conditioned on the last frame of the previous one. The codec's GOP is the generative model's context window.
2. **Motion as a first-class object.** The codec represents motion *explicitly* as vectors. Generative models can do this too (motion conditioning, trajectory control, optical-flow guidance), but as we will see, most learn it *implicitly* inside temporal attention. The codec is the proof that motion is low-dimensional enough to be worth separating from appearance.
3. **Residual coding.** The idea that you should spend your bits (or your model capacity) on *what changed*, not on re-describing what stayed the same, is the whole game. A 3D-VAE that compresses time is, in effect, learning a residual code: it keeps full detail on the spatial content and a thin description of how it evolves.

There is one place generative models *diverge* from codecs, and it matters. A codec is *lossless about structure and lossy about detail in a hand-designed way* — it is optimized to be perceptually faithful to *this specific input video*. A generative latent is optimized to be a good *training signal for a model that will produce new videos*, which means it can throw away things a codec would keep (exact textures, film grain) as long as the decoder can plausibly re-synthesize them, and it must keep things a codec might quantize away (semantically important edges) because the denoiser needs them. So while the *idea* is borrowed, the *objective* is different: a codec minimizes reconstruction bits for a known video; a video VAE minimizes reconstruction error subject to a latent that a diffusion model can actually denoise. The next post is entirely about that objective.

## Optical flow: motion as an explicit field

The codec's motion vectors are a coarse, block-level version of a more fundamental object: **optical flow**. Optical flow is the per-pixel motion field — for every pixel in frame $t$, a 2D vector $(u, v)$ saying where that pixel went in frame $t+1$. Formally it rests on the **brightness-constancy assumption**: a point's intensity is the same before and after it moves, so

$$ I(x, y, t) = I(x + u, y + v, t + 1). $$

Taylor-expanding the right side to first order gives the classic **optical-flow constraint equation**:

$$ I_x u + I_y v + I_t = 0, $$

where $I_x, I_y$ are spatial image gradients and $I_t$ is the temporal gradient (the frame difference). This is one equation in two unknowns per pixel — the famous **aperture problem**, which is why flow methods add a regularizer (Horn–Schunck's smoothness term, Lucas–Kanade's local-constancy window, or, today, a learned prior as in RAFT). The output is a dense field $F \in \mathbb{R}^{T \times H \times W \times 2}$: two numbers per pixel per frame describing motion.

![A nine-cell grid laying out a latent video tensor across time and space, showing that the four axes of time, height, width, and channels all shrink under compression](/imgs/blogs/representing-video-redundancy-and-tokens-5.png)

Optical flow is the *explicit* answer to "what is the motion?", and historically it powered a whole generation of video methods: frame interpolation (warp a frame along the flow to invent an in-between frame), video super-resolution (align frames before fusing), and early video-generation attempts that warped a single image along a predicted flow to animate it. The appeal is that flow makes the temporal-redundancy prior *operational*: if you know the flow, you can *warp* frame $t$ to predict frame $t+1$, and the only thing left to generate is the small residual — exactly the codec's P-frame, but at pixel resolution.

So why do modern video diffusion models mostly *not* use explicit optical flow? Three reasons, and they are instructive about the field's whole trajectory:

- **Flow breaks at exactly the hard cases.** Brightness constancy fails under occlusion (a pixel disappears behind an object — where did it "go"?), lighting changes, reflections, transparency, and large fast motion (the linearization is only valid for small displacements). The places where flow is unreliable are precisely the places a generative model most needs to get right. A model that *depended* on flow would inherit all of flow's failure modes.
- **Learned temporal attention subsumes it.** A transformer with attention across the time axis can learn whatever motion relationship the data demands — including non-rigid deformation, appearance change, and long-range correspondence — without committing to the brightness-constancy assumption. The motion ends up represented *implicitly* in the attention patterns and the latent dynamics, not as an explicit field. You give up interpretability and gain robustness.
- **The latent already encodes change.** Once you compress a clip with a 3D-VAE, the temporal axis of the latent *is* a learned motion representation. Adjacent latent frames already factor appearance and change in whatever way minimized the reconstruction loss; bolting an explicit flow field on top is redundant.

That said, flow has not vanished — it has moved to the edges. It powers **conditioning and control** (give the model a flow or trajectory and it follows the motion you specify — motion-brush tools, drag-based animation, and camera-trajectory control all pass an explicit motion field into the model), it is used in **evaluation** (VBench's "motion smoothness" is computed from optical-flow consistency, and "dynamic degree" is essentially a thresholded flow magnitude), and it remains the right tool for classical interpolation and frame-rate upsampling. The pattern — *an explicit, hand-designed representation gets absorbed into a learned one but survives as a control and measurement signal* — is one you will see again and again across this series. We will still measure flow magnitude in code below, because even when a model uses motion implicitly, *you* the engineer want to know how much motion is in your data: it predicts how hard the clip will be to generate coherently.

The aperture problem deserves one more sentence because it explains a failure you will see in generated video. With only the single constraint equation $I_x u + I_y v + I_t = 0$ per pixel, you can recover only the motion *component along the image gradient* (perpendicular to edges) — motion *along* an edge is invisible locally, which is why a featureless moving wall or a rotating barber-pole looks ambiguous. Learned models inherit a softened version of this: they are most confident about motion where there is texture to track and least confident in flat regions, which is exactly where generated video tends to wobble or smear — large untextured surfaces (skies, walls, water) drift because the model, like classical flow, has no local signal pinning their motion. The aperture problem is not a quirk of an old algorithm; it is a property of the *information* in the pixels, so every motion representation — explicit or learned — pays for it somewhere.

## Measuring redundancy in code

Enough assertion — let us measure. The following loads a clip, computes the inter-frame difference, and quantifies how redundant the video actually is. This is the single most useful diagnostic you can run on a video dataset before training: it tells you how much temporal compression your data can tolerate and which clips will be hard.

```python
import torch
import numpy as np
from torchvision.io import read_video

# read_video returns frames as a (T, H, W, C) uint8 tensor in [0, 255]
# pts_unit="sec" keeps timestamps sane; we only need the frames here
frames, _, info = read_video("clip.mp4", pts_unit="sec", output_format="THWC")
frames = frames.float()                      # (T, H, W, C), 0..255
T, H, W, C = frames.shape
print(f"clip: T={T} frames, {W}x{H}, {C} channels")
print(f"raw values = {T*H*W*C:,}  ({T*H*W*C/1e9:.3f}B)")

# inter-frame residual: R_t = V_t - V_{t-1}, for t = 1..T-1
residual = frames[1:] - frames[:-1]          # (T-1, H, W, C)

mean_abs = residual.abs().mean().item()      # average pixel change, 0..255
# "active" pixels: those that changed by more than a small threshold (8 of 255)
active = (residual.abs().amax(dim=-1) > 8).float().mean().item()
print(f"mean |residual|      = {mean_abs:6.2f}  (out of 255)")
print(f"fraction of pixels moving > 8/255 = {active*100:5.1f}%")
```

Run this on a tripod-mounted talking-head clip and you will typically see a mean residual around 1–3 and an active fraction under 5%. Run it on a handheld action shot and the mean residual leaps past 20 with most of the frame "moving" (because the *camera* moved, so every pixel shifts). That single `active` number is a remarkably good predictor of compressibility and of generation difficulty.

We can go one step further and estimate the *entropy* of the residual versus the frame, which makes the $H(R_t) \ll H(V_t)$ inequality concrete:

```python
def shannon_entropy_bits(x, bins=256):
    """Empirical per-value entropy in bits, from a histogram."""
    x = x.flatten().cpu().numpy()
    hist, _ = np.histogram(x, bins=bins, range=(x.min(), x.max()), density=True)
    p = hist[hist > 0]
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())

# entropy of a single frame's pixels vs. the inter-frame residual
H_frame = shannon_entropy_bits(frames[0])
H_resid = shannon_entropy_bits(residual)
print(f"H(frame)    = {H_frame:.2f} bits/value")
print(f"H(residual) = {H_resid:.2f} bits/value")
print(f"redundancy factor = {H_frame / max(H_resid, 1e-6):.1f}x")
```

For ordinary footage you will see the frame entropy around 6–7 bits and the residual entropy a small fraction of that — a redundancy factor of anywhere from 3× on busy clips to 20× or more on calm ones. This is the empirical face of the conditional-entropy inequality: knowing the previous frame collapses the information you still need by that factor. **That factor is the headroom every video representation is trying to capture.**

Now the optical-flow magnitude, which tells you *how much motion* (not just how much pixel change — a global brightness flicker changes pixels without any motion). We use OpenCV's Farnebäck dense flow because it needs no model download; for quality you would reach for RAFT (`torchvision.models.optical_flow.raft_large`), but the magnitude statistic is what we want and Farnebäck gives it cheaply.

```python
import cv2
import numpy as np

def flow_magnitude(prev_gray, next_gray):
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, next_gray,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )                                        # (H, W, 2): per-pixel (u, v)
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return mag                               # pixels-of-motion per pixel

gray = frames.mean(dim=-1).to(torch.uint8).numpy()   # (T, H, W) grayscale
mags = [flow_magnitude(gray[t - 1], gray[t]) for t in range(1, T)]
mean_motion = float(np.mean([m.mean() for m in mags]))
p95_motion = float(np.percentile(np.concatenate([m.ravel() for m in mags]), 95))
print(f"mean optical-flow magnitude = {mean_motion:.2f} px/frame")
print(f"95th-percentile motion      = {p95_motion:.2f} px/frame")
```

Mean flow magnitude under ~1 px/frame is a near-static clip; 1–5 px/frame is gentle motion that models handle well; double-digit px/frame is fast motion where temporal coherence gets genuinely hard and where the brightness-constancy assumption (and any model leaning on it) starts to break. This number, paired with the residual statistics, is the profile of a clip's *difficulty*. I keep both in my dataset manifest and use them to balance training batches — too many calm clips and the model learns to barely move; too many violent ones and it learns to flicker.

### Profiling a dataset honestly

Two cautions about doing this measurement *honestly*, because both bit me in practice. First, **global photometric changes masquerade as motion in the residual but not in the flow.** A camera auto-exposure adjustment or a fade-to-black changes every pixel's *value* (so the frame-difference residual spikes) while nothing actually *moved* (so optical-flow magnitude stays near zero). If you profile compressibility by residual magnitude alone, you will mislabel an exposure ramp as a high-motion clip and over-allocate model capacity to it. Reporting *both* residual entropy and flow magnitude — and flagging clips where they disagree — separates "the lighting changed" from "things moved," which matters because a 3D-VAE compresses photometric change cheaply (it is low-rank) but genuine motion expensively.

Second, **measure on the representation you will actually train in, not on pixels.** The redundancy you care about is the redundancy *after* the VAE encodes the clip, because that is the signal the denoiser sees. A clip that looks high-motion in pixels (a slow pan across fine texture) may be highly redundant in latent space (the VAE's spatial compression already folded the texture away), and vice versa. The honest profiling pass encodes a sample of clips into latents once and measures inter-*latent*-frame differences — the same code above, run on `vae.encode(frames).latent_dist.mode()` instead of raw frames. It is one extra step and it is the difference between profiling the data and profiling what the model will actually struggle with.

Putting the diagnostics together gives a difficulty profile you can act on. Here is the kind of table I build for a candidate dataset before committing to a training run — three representative buckets, with the statistics that predict how each will behave:

| Clip type | Mean &#124;residual&#124; | Active pixels | Flow mag (px/frame) | Residual entropy | Compressibility |
|---|---|---|---|---|---|
| Locked-off talking head | ~1.5 / 255 | < 3% | ~0.4 | ~0.4 bit | very high (≥20× redundant) |
| Gentle handheld B-roll | ~8 / 255 | ~25% | ~2.5 | ~1.8 bit | high (~8× redundant) |
| Fast sports / whip pan | ~30 / 255 | > 80% | ~15 | ~4.5 bit | low (~1.5× redundant) |

The right-hand column is what you are buying with temporal compression, and it is *not uniform across your data*. A dataset that is mostly talking heads will tolerate aggressive $c_t$ (the redundancy is enormous); a dataset of sports footage will fight it (the redundancy is thin, and a high $c_t$ will blur the motion the VAE cannot represent). This is the empirical, measurable face of the entire post's thesis: the representation can only capture the redundancy that is *in the content*, and you should measure the content before you pick the compression.

## Three representation choices: pixel, latent, token

We have established that the raw tensor is enormous and that temporal redundancy means it *can* be compressed dramatically. The question a generative-model designer actually has to answer is: **in what space does the model do its work?** There are three answers, and the entire architecture of a video model follows from which one you pick.

![A taxonomy tree of representation choices branching from the raw clip into pixel space versus compress-first, with compress-first splitting into a continuous latent and discrete tokens](/imgs/blogs/representing-video-redundancy-and-tokens-7.png)

**Pixel space.** The model operates directly on $V \in \mathbb{R}^{T\times H\times W\times C}$. This is the most faithful — no information is thrown away before the model sees it — and the most expensive. Early pixel-space video diffusion models made it work only by going *small and cascading*: Make-A-Video and Imagen Video generate a low-resolution, low-frame-rate clip in pixel space and then run a stack of spatial and temporal **super-resolution** diffusion models to upsample resolution and interpolate frames. The cascade is itself a redundancy trick — it never holds the full-resolution full-frame-rate tensor in the expensive base model; it lets cheap upsamplers fill in the predictable high-frequency and in-between-frame detail. Pixel space survives today only for short, low-res stages or specialized models; at 720p-and-up it is a memory non-starter for the reasons we computed.

**Latent space.** Compress the clip with an autoencoder into a much smaller continuous tensor, run the diffusion there, and decode once at the end. This is the *dominant* choice — it is what CogVideoX, HunyuanVideo, Wan, Mochi, SVD, LTX-Video, and essentially every modern open and closed video diffusion model do. There are two flavors. A **per-frame latent** encodes each frame independently with an image VAE (8× spatial compression, the SD-VAE move), giving $\frac{T \cdot H \cdot W \cdot C}{64}$ values — it shrinks space but *not time*, so $T$ is untouched. A **3D (spatiotemporal) latent** compresses time *and* space jointly with a causal 3D-VAE — a typical ratio is $4\times8\times8$ (4× along time, 8× along each spatial axis), which shrinks the tensor by $4 \cdot 8 \cdot 8 = 256$ before accounting for channel changes. The 3D latent is the lever that makes long, high-res clips affordable, and it is the subject of the next post. The latent idea is exactly the one the image series builds in [latent diffusion and Stable Diffusion](/blog/machine-learning/image-generation/latent-diffusion-and-stable-diffusion); video just adds the time axis to the compression.

**Token space.** Quantize the (usually already-compressed) representation into a finite vocabulary of **discrete tokens** — like words — so the video becomes a *sequence* you can model autoregressively, the way a language model models text. A VQ-VAE or its successors (VQGAN, the magnitude-quantized tokenizers in MAGVIT/MAGVIT-v2) map each spatiotemporal latent patch to the nearest entry in a learned codebook, producing a grid of integer token IDs. An autoregressive transformer (or a masked-token model like MAGVIT/Phenaki) then predicts tokens one or many at a time. Token space is what makes *autoregressive* video generation and unified vision-language-action models possible — and it is the representation behind the world-model line (Genie tokenizes frames and predicts the next token grid conditioned on an action). The cost is quantization error (a finite codebook cannot represent every latent exactly) and the same sequence length you would have in latent space, now as a discrete budget.

To make the latent choice concrete in code, here is a real image-to-video call in 🤗 `diffusers` against CogVideoX, with the exact flags that exist *because* of the token budget. Notice `enable_model_cpu_offload`, `vae.enable_tiling`, and the way the pipeline encodes the conditioning image to a latent before the denoiser ever runs — every one of these is a concession to the value count we have been computing.

```python
import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    "THUDM/CogVideoX-5b-I2V", torch_dtype=torch.bfloat16
)
# the levers that exist BECAUSE of the token/value budget:
pipe.enable_model_cpu_offload()      # weights stream on/off GPU to fit 24 GB
pipe.vae.enable_tiling()             # decode the latent in spatial tiles, not all at once
pipe.vae.enable_slicing()            # decode frames in slices to cap VAE peak memory

image = load_image("first_frame.png")          # the I2V anchor (the "keyframe")
video = pipe(
    image=image,
    prompt="a golden retriever running across a sunlit field",
    num_frames=49,                   # latent length after 4x temporal compression: ~13
    num_inference_steps=50,
    guidance_scale=6.0,
).frames[0]

export_to_video(video, "out.mp4", fps=8)
```

The `num_frames=49` is not arbitrary: with the causal 3D-VAE's 4× temporal compression, 49 input frames become $\lceil 49/4 \rceil = 13$ latent frames, and 13 is what the DiT actually attends over along time. The `vae.enable_tiling()` and `enable_slicing()` calls exist because the VAE *decode* — turning the small latent back into full-resolution pixels at the very end — briefly reconstructs the large tensor, and that decode, not the denoiser, is frequently the peak-VRAM moment of the whole pipeline. (We flagged that "the VAE decode, not the denoiser, is the VRAM wall" as a stress test; here is where you would hit it.) Every flag in this snippet is the token-and-value budget made operational.

The crucial point — and the reason this whole post exists — is that **all three reduce to a count.** Pixels, continuous latents, discrete tokens: whatever you call them, the model's compute is governed by *how many positions it has to attend over*. That count is the token budget, and it is set entirely by the compression factors.

![A layered stack from the input clip through a causal 3D encoder and a latent tensor to a flattened token sequence feeding a spacetime DiT](/imgs/blogs/representing-video-redundancy-and-tokens-6.png)

## The token budget: why compression is the whole ballgame

Here is the formula that ties the post together. After compressing a clip by a temporal factor $c_t$ and a spatial factor $c_s$ (per axis), the number of tokens the model must process is

$$ N \;\approx\; \frac{T}{c_t} \cdot \frac{H}{c_s} \cdot \frac{W}{c_s}. $$

(Channels become the per-token embedding dimension, not extra tokens; and if you additionally patchify with patch size $p$, divide each spatial term by $p$ and the temporal term by the temporal patch — for clarity below we fold patchify into $c_s, c_t$.) The thing that makes $N$ matter so much is what it feeds: the dominant cost in a spacetime transformer is **self-attention**, whose compute and memory both scale **quadratically** in the sequence length:

$$ \text{Attention cost} \;\propto\; N^2. $$

That single exponent is the reason compression is not a nice-to-have but a *gate*. Halving the tokens quarters the attention cost. A 256× reduction in tokens is a **65,536× reduction in attention FLOPs and activation memory.** This is why the foundations post called video "quadratic in frames" — and why, of the entire stack, the compression ratio is the most important number you choose. Everything downstream (how long a clip you can make, how high a resolution, how big a batch, whether it fits on a 24 GB card or needs an H100 cluster) is set here.

Let us make $N^2$ concrete by computing it for our 5-second 720p clip under each representation. Recall $T=120$, $H=720$, $W=1280$.

#### Worked example: tokens and attention for a 5s 720p clip, 1× vs 4×8×8

**Pixel space ($c_t = 1$, $c_s = 1$).** Treating every pixel position as a token (the naive limit),

$$ N_{\text{pixel}} = 120 \cdot 720 \cdot 1280 = 110{,}592{,}000 \approx 110.6\text{M tokens}, $$

and attention cost $\propto N^2 = 1.22\times 10^{16}$. This is not "expensive," it is *physically impossible* — 110 million tokens through full attention would need petabytes of activation memory. Even with a 16×16 spatial patch (the DiT default), you are at $120 \cdot 45 \cdot 80 = 432{,}000$ tokens and $1.9\times10^{11}$ attention units *per layer*, still brutal for a deep model trained for hundreds of thousands of steps. Pixel space is out.

**Per-frame latent ($c_t = 1$, $c_s = 8$).** Compress space 8× but leave time alone:

$$ N_{\text{per-frame}} = 120 \cdot 90 \cdot 160 = 1{,}728{,}000 \approx 1.73\text{M tokens}, $$

attention cost $\propto N^2 = 2.99\times 10^{12}$. Better by a factor of 4,096 than naive pixels, but still 1.7 million tokens — a long way past what you would attend over jointly. (In practice per-frame-latent models add a spatial patchify and *factorize* attention to avoid full 3D attention; even so, $T=120$ untouched frames is a lot of temporal length.)

**3D latent ($c_t = 4$, $c_s = 8$).** Now compress time 4× as well:

$$ N_{\text{3D}} = \frac{120}{4} \cdot \frac{720}{8} \cdot \frac{1280}{8} = 30 \cdot 90 \cdot 160 = 432{,}000 \approx 432\text{k tokens}, $$

attention cost $\propto N^2 = 1.87\times 10^{11}$. Versus naive pixels, that is a **256× token reduction and a 65,536× attention reduction** — the difference between impossible and trainable. Add the standard $1\times2\times2$ latent patchify that real models (CogVideoX) apply on top, and you divide by another 4, landing at **108k tokens** and $1.2\times10^{10}$ attention units — comfortably in the range a DiT handles. *That* is the regime modern video models actually live in.

![A decision matrix of the worked example comparing 1x pixels, per-frame 8x, and 3D 4x8x8 compression across token count, quadratic attention cost, and the resulting verdict from out-of-memory to trainable](/imgs/blogs/representing-video-redundancy-and-tokens-8.png)

The table makes the cliff vivid:

| Representation | $c_t$ | $c_s$ | Tokens $N$ | Attention $\propto N^2$ | Verdict |
|---|---|---|---|---|---|
| Pixel (per-pixel) | 1 | 1 | 110.6M | $1.2\times10^{16}$ | impossible |
| Pixel (16×16 patch) | 1 | 16 | 432k | $1.9\times10^{11}$ | brutal |
| Per-frame latent | 1 | 8 | 1.73M | $3.0\times10^{12}$ | very costly |
| **3D latent 4×8×8** | **4** | **8** | **432k** | $1.9\times10^{11}$ | **feasible** |
| 3D latent + 1×2×2 patch | 4 | 16 | 108k | $1.2\times10^{10}$ | comfortable |

![A decision matrix comparing raw pixels, per-frame latent, 3D latent, and discrete tokens across value counts, attention cost, and training feasibility](/imgs/blogs/representing-video-redundancy-and-tokens-4.png)

Two things jump out. First, **temporal compression is doing real, independent work** — going from per-frame latent (1.73M tokens) to 3D latent (432k) is a 4× token cut and a 16× attention cut, purely from compressing the time axis the per-frame approach left alone. That 4× is the entire argument for paying the cost of a 3D-VAE over a cheap per-frame image VAE. Second, notice that "pixel with a 16×16 patch" and "3D latent" land at the *same* token count by coincidence — but they are not equivalent, because the pixel-patch representation threw away no temporal redundancy and is far worse at reconstruction per token. Compression buys you *quality per token*, not just fewer tokens.

Here is a tiny calculator you can keep around; it is the back-of-the-envelope I run before committing to any clip length or resolution.

```python
def token_budget(T, H, W, c_t=4, c_s=8, patch_t=1, patch_s=2):
    """Tokens and relative attention cost for a clip under given compression."""
    n_t = T // (c_t * patch_t)
    n_h = H // (c_s * patch_s)
    n_w = W // (c_s * patch_s)
    N = n_t * n_h * n_w
    raw_values = T * H * W * 3
    return {
        "raw_values": raw_values,
        "tokens": N,
        "attention_units": N * N,           # ~quadratic in N
        "token_reduction_vs_pixels": (T * H * W) / max(N, 1),
    }

# our running example, no compression vs the standard 4x8x8 + 1x2x2 patch
for label, kw in [("pixels", dict(c_t=1, c_s=1, patch_t=1, patch_s=1)),
                  ("per-frame 8x", dict(c_t=1, c_s=8, patch_t=1, patch_s=1)),
                  ("3D 4x8x8 +patch", dict(c_t=4, c_s=8, patch_t=1, patch_s=2))]:
    b = token_budget(120, 720, 1280, **kw)
    print(f"{label:18s} tokens={b['tokens']:>12,}  attn~{b['attention_units']:.2e}")
```

Run it and the cliff between the rows is the single most important fact about video model design. When the next post asks "why is the 3D-VAE — not the denoiser — the real bottleneck and the main length lever," *this* is the reason: the VAE sets $c_t$ and $c_s$, and those exponents in $N^2$ decide everything the denoiser can afford to do.

### Factorized attention: trading coherence for a smaller exponent

There is a second lever that the token budget makes visible, and it is worth deriving because it is the most common way real models tame the $N^2$ wall *without* compressing further. Full 3D (joint spatiotemporal) attention lets every token attend to every other token, so its cost is the full $N^2 = (T'H'W')^2$, where $T', H', W'$ are the compressed dimensions. **Factorized attention** splits this into a *spatial* attention that runs within each frame and a *temporal* attention that runs across frames at each spatial position. The spatial pass attends over $H'W'$ tokens, $T'$ times; the temporal pass attends over $T'$ tokens, $H'W'$ times. The combined cost is

$$ \text{full 3D}: \; (T'H'W')^2 \qquad\text{vs.}\qquad \text{factorized}: \; T'\,(H'W')^2 + H'W'\,(T')^2. $$

Plug in our 3D-latent dimensions $T'=30$, $H'W' = 90 \cdot 160 = 14{,}400$ (so $N = 432{,}000$). Full 3D attention costs $N^2 = 1.87\times10^{11}$ units. Factorized attention costs $30 \cdot 14400^2 + 14400 \cdot 30^2 = 6.2\times10^{9} + 1.3\times10^{7} \approx 6.2\times10^{9}$ units — a **30× reduction**, and it grows: the longer the clip (the bigger $T'$), the more factorization saves, because it never multiplies the time axis by the full spatial token count. The price is *coherence*: factorized attention cannot directly relate a pixel in frame 5 to a *different* pixel in frame 25 in one hop — it has to route through the spatial-then-temporal decomposition, which weakens long-range spatiotemporal correspondence and is one source of the flicker and identity-drift that plague factorized models. This is the central trade-off of the next track's architecture posts; I raise it here only to show that **the token budget is a knob, not a verdict** — you can attack the $N^2$ by shrinking $N$ (more compression, this post's lever) *or* by changing the attention pattern so you never pay the full $N^2$ (factorization, [spatiotemporal attention patterns](/blog/machine-learning/video-generation/spatiotemporal-attention-patterns)). Most production models do both.

The reason I emphasize compression *first* is that it is strictly upstream: factorization helps the denoiser, but the VAE's $c_t, c_s$ also shrink the *latent storage*, the *VAE decode cost*, and the *number of denoiser steps' worth of tokens you hold in memory*. Compression is a lever on the whole stack; attention factorization is a lever on one box. When you have a fixed compute budget and need both length and coherence, you spend your first move on the compression ratio and your second on the attention pattern — and you measure both against the token budget this section defined.

## Putting the latent tensor on a page

It helps to picture the compressed representation concretely, because "3D latent" can sound more mysterious than it is. After the causal 3D-VAE encodes our clip at $4\times8\times8$ with a 16-channel latent, the tensor is

$$ z \in \mathbb{R}^{\frac{T}{4} \times \frac{H}{8} \times \frac{W}{8} \times 16} = \mathbb{R}^{30 \times 90 \times 160 \times 16}, $$

which is $30 \cdot 90 \cdot 160 \cdot 16 = 6{,}912{,}000$ values — **6.9 million**, down from 331.8 million, a **48× reduction** in raw value count (and the *token* count, ignoring channels, is the 432k we computed). Each of the 30 latent "frames" summarizes 4 real frames; each latent spatial position summarizes an 8×8 pixel block; each position carries a 16-dimensional vector instead of 3 color channels. The temporal axis is shorter but each step is *denser* — it has folded four frames' worth of appearance-and-motion into one. The grid figure above lays out exactly this: time × height × width, with channels as the depth at each cell, every axis shrunk.

The reason this is "denser, not just smaller" is the redundancy argument coming full circle. Those 4 real frames the latent collapsed into one were ~95% redundant; the latent does not need 4× the capacity to represent them because there was never 4× the *information*. The 3D-VAE learned a code that spends its 16 channels on the *new* information per latent step — appearance plus the small change — which is exactly the residual-coding idea from the codec, now learned end-to-end rather than hand-designed. The temporal axis of the latent *is* the model's learned motion representation; we got the benefit of optical flow (motion factored out) without ever computing a flow field.

One practical consequence worth flagging now, because it surprises people the first time they profile a pipeline: the latent is 48× smaller, but the *decode* back to pixels is not free. The VAE decoder has to expand all 6.9M latent values back into 331.8M pixels, and it does so through a stack of convolutions and (in a causal 3D-VAE) temporal upsampling that briefly materializes the large tensor — so the peak memory of generation often lands not in the denoiser, which lives comfortably in the small latent, but in the *final decode*. This is why the tiling and slicing flags in the code earlier exist, and why "the VAE decode is the VRAM wall" is a real stress test rather than a curiosity. The compression that made training feasible has to be *paid back* at decode time, and a careful pipeline pays it back in tiles so the bill never comes due all at once.

## The four representations, side by side

Let us consolidate everything into the comparison a designer actually uses. The axes that matter are: how many values/tokens the representation produces, how faithful it is (reconstruction quality), what generative paradigm it enables, and where each is used in practice.

| Representation | Values for our 5s clip | Tokens (attn length) | Fidelity | Paradigm | Used by |
|---|---|---|---|---|---|
| **Pixel space** | 331.8M | 110.6M (or 432k @ p16) | perfect (no compression) | cascaded diffusion (low-res base + SR) | Make-A-Video, Imagen Video |
| **Per-frame latent** | 27.6M (16 ch) | 1.73M | high; no temporal compression | latent diffusion, factorized attention | early latent video, AnimateDiff over SD-VAE |
| **3D latent (4×8×8)** | 6.9M (16 ch) | 432k (108k @ patch) | high; joint spatiotemporal | latent diffusion + spacetime DiT | CogVideoX, HunyuanVideo, Wan, Mochi, LTX |
| **Discrete tokens** | ~432k codes | 432k | lossy (quantization) | autoregressive / masked-token | MAGVIT, Phenaki, Genie, VideoPoet |

A few reading notes. The **value count** and the **token count** are different quantities and both matter: value count drives VAE decode memory and storage; token count drives the denoiser's attention cost. The 3D latent wins on *both* because temporal compression cuts the time axis that the per-frame latent leaves at full length. **Fidelity** is where pixel space and continuous latents beat discrete tokens — quantization to a finite codebook is genuinely lossy, which is why the highest-quality video models today are continuous-latent diffusion models, not token-autoregressive ones (the token models win on *flexibility* — unified with language, action-conditionable — not on raw fidelity). And the **paradigm** column is the punchline: your representation choice is not cosmetic, it *selects your generative method*. Continuous latent → diffusion/flow-matching. Discrete tokens → autoregressive/masked prediction. Pixel → cascaded diffusion. You do not pick the method and then a representation; you pick a representation and it largely picks the method.

#### Worked example: what changes if you want a 10-second clip instead of 5

Suppose your 5-second 720p 3D-latent model fits on an A100 80GB at ~40 GB peak with 432k tokens. A product manager asks for 10 seconds. Naively you double $T$ to 240, which doubles tokens to 864k — but attention is quadratic, so the attention cost goes up **4×**, not 2×, and the activation memory for attention scales similarly. You blow past 80 GB and OOM. You have three levers, all of which trace back to this post: (1) increase temporal compression $c_t$ from 4 to 8 in the VAE — that halves tokens back to 432k and quarters the attention cost back to baseline, at some reconstruction cost; (2) keep $c_t$ but *factorize* attention so temporal and spatial attention run separately (spatial attention sees $H'W'$ tokens, temporal sees $T'$ — neither sees the full $N$), trading some coherence for a linear-ish cost; or (3) switch to **autoregressive chunked rollout** — generate two 5-second chunks, conditioning the second on the last frame of the first, keeping each chunk's token budget fixed (at the cost of error accumulation across the boundary). Every one of these is a *representation-and-budget* decision. The length problem is a token-budget problem, and you solve it by changing $c_t$, changing the attention pattern, or chunking the sequence — there is no free lunch hiding in a bigger GPU, because the wall is quadratic.

## Stress-testing the redundancy prior

The temporal-redundancy prior is powerful precisely because it is *usually* true. It is worth being honest about where it weakens, because those are the cases where video models — and the representations we have been praising — struggle.

**Scene cuts and shot changes.** At a hard cut, frame $t$ and frame $t-1$ are *unrelated*; the residual entropy spikes to the full frame entropy, $H(R_t) \approx H(V_t)$. The redundancy you were counting on vanishes for that one transition. Codecs handle this by inserting an I-frame at the cut; 3D-VAEs handle it by training on cut-free clips (datasets are scene-segmented before training), because a VAE asked to compress across a cut wastes capacity trying to find motion that is not there. The lesson for data engineering: **segment your training clips at shot boundaries**, or your compression model spends capacity on impossible predictions.

**Large, fast motion.** When motion between frames is large (high optical-flow magnitude), the linearization behind both optical flow and a fixed-window temporal model breaks down. A pixel can move farther than the model's temporal receptive field, so "where did it go" becomes genuinely ambiguous. This is the same regime where generative models flicker or smear fast-moving objects, and where the "dynamic-degree vs. stability" tension in evaluation comes from — a model can score well on motion *or* on consistency, and pushing one degrades the other, because high motion *is* low redundancy and low redundancy is hard.

**Past the VAE's trained temporal window.** A causal 3D-VAE is trained on clips of a fixed length (say up to 49 or 81 frames). Encode a clip longer than that and the temporal compression degrades or produces artifacts at the seams, because the convolutions never saw that temporal extent. This is one face of the "what happens past the VAE's trained clip length" stress test, and it is why long-video generation is *not* just "set $T$ bigger" — it requires chunking or a VAE explicitly trained for longer windows. The representation's redundancy assumption is calibrated to a length, and exceeding it breaks the calibration.

**Transparency, reflection, and non-rigid deformation.** Smoke, water, glass, and fluids violate brightness constancy and have no clean motion field — a reflection moves opposite to the surface, smoke has no rigid correspondence at all. Explicit-flow methods fail here outright; learned latents do better because they never assumed rigidity, but these remain the hardest content to compress *and* to generate coherently. When you see a video model produce beautiful rigid motion and mushy water, this is why: the redundancy prior is strong for rigid scenes and weak for fluid ones.

The throughline: **the redundancy prior is a property of the content, and the representation inherits its content's compressibility.** A representation cannot manufacture redundancy that is not there. This is why "quantify everything" is not a slogan — measuring residual entropy and flow magnitude *before* training tells you which clips your compression will handle and which will fight it.

## When to reach for each representation (and when not to)

A decisive recommendation section, because the whole point of understanding the token budget is to make this call confidently.

**Reach for a 3D latent (continuous, diffusion) when** you want the best quality-per-compute for clips up to a minute or so — which is almost always. It is the modern default for good reason: joint spatiotemporal compression gives you the smallest token budget at the highest fidelity, and it composes with the spacetime-DiT and flow-matching machinery the rest of this series covers. If you are building or fine-tuning a video generator in 2026, you are almost certainly working in a 3D latent. Start here.

**Reach for a per-frame latent only when** you are *inflating a pretrained image model* (AnimateDiff, early SVD-style work) and want to reuse a frozen image VAE plus a frozen T2I backbone, adding only a temporal module. You accept a larger token budget (time uncompressed) in exchange for not training a 3D-VAE from scratch and for inheriting a strong image prior. It is a pragmatic shortcut, not a quality ceiling — and you pay for it with that 4× token penalty on the time axis.

**Reach for discrete tokens when** you need *autoregressive* generation or unification with language and action: world models (Genie's action-conditioned next-token prediction), interactive/playable video, video-language-action models, or any setting where you want to reuse the transformer-decoder and KV-cache machinery from LLMs. Accept the quantization fidelity hit; it buys you flexibility and a clean interface to text and actions that continuous latents do not have. **Do not** reach for discrete tokens if your only goal is the highest-fidelity fixed-length clip — continuous-latent diffusion currently wins that contest.

**Reach for pixel space (cascaded) almost never, today** — only for the *base* stage of a cascade at very low resolution and frame rate, where the token count is small and you will upsample afterward, or for research that specifically needs no autoencoder in the loop. At any real resolution, the value count we computed at the top of this post makes pixel-space training a memory dead end. The historical pixel-space models (Make-A-Video, Imagen Video) all survived by *cascading*, which is itself a concession that you cannot afford full resolution in one model.

And the meta-rule: **the compression ratio is the first decision, not the last.** Pick $c_t$ and $c_s$ before you pick model size, attention pattern, or sampler, because the token budget they set bounds everything else. A model designer who treats the VAE as a preprocessing detail will discover, at second six of an OOMing render, that it was the most important choice they made.

## Case studies: the numbers real models actually use

These representation choices are not hypothetical — here is what shipped models do, with the figures they report.

**CogVideoX (Zhipu AI / Tsinghua, 2024)** uses a causal 3D-VAE with **4×8×8** compression and a 16-channel latent, then a 3D-DiT with a $1\times2\times2$ patchify on top, then full 3D attention over the resulting tokens. For its 49-frame 720×480 clips this lands the token budget in the low hundreds of thousands — exactly the regime our worked example identified as "comfortable." The reported recipe (causal 3D-VAE + expert-adaptive DiT + 3D full attention) is the canonical open instantiation of everything this post argued; the 5B model runs on a single high-memory consumer GPU with VAE tiling and CPU offload, which is *only* possible because of the 256× token reduction the 3D-VAE buys.

**HunyuanVideo (Tencent, 2024) and HunyuanVideo-1.5 (2025)** likewise use a 3D causal VAE with high spatiotemporal compression feeding a large DiT (HunyuanVideo's base is ~13B parameters). The 1.5 release is notable for generating long clips (reportedly up to ~75 seconds of coherent video) on a single 24 GB-class GPU — a length that is *only* reachable by aggressive temporal compression and chunked/sliding generation, i.e., by managing the token budget exactly as this post describes. The headline is not the parameter count; it is the token budget that lets a 24 GB card hold the sequence.

**Make-A-Video (Meta, 2022)** is the pixel-space counterexample that proves the rule. It generates in pixel space but *cascades*: a base model produces a low-resolution, low-frame-rate clip, then spatial super-resolution and frame-interpolation networks upsample to the final 768×768 and higher frame rate. It never holds the full-resolution full-frame-rate tensor in one expensive model — the cascade is its way of dodging the value-count wall we computed at the top. It worked, and it was state-of-the-art for its moment, but the entire field moved to *latent* compression within a year precisely because cascading is a more awkward way to manage the same redundancy that a 3D-VAE captures in a single learned encoder.

**Genie (DeepMind, 2024)** sits in token space: it tokenizes video frames and trains a model to predict the next frame's tokens *conditioned on a latent action*, yielding an interactive, playable environment learned from unlabeled video. It is the clearest demonstration of *why* you would accept the quantization fidelity hit of discrete tokens — the token interface is what makes the model action-conditionable and lets it reuse autoregressive-transformer machinery. You would not pick tokens for a cinematic T2V model; you pick them when the *interface* (to actions, to language) matters more than the last few points of fidelity. We pick this thread back up in the world-models posts.

The convergence is striking: across CogVideoX, HunyuanVideo, Wan, Mochi, and LTX-Video, the open recipe has settled on **causal 3D-VAE + spacetime DiT + flow matching**, and the *first* of those three is the 3D-VAE — the compression — because it sets the token budget that the other two have to live within. The representation is not one choice among many; it is the choice that constrains the rest.

## Key takeaways

- **A 5-second 720p clip is ~0.33 billion raw values** ($T\cdot H\cdot W\cdot C$); a 10-second 1080p clip is ~1.9 billion. The *data* is the problem before the model is, and pixel-space training is a memory dead end at real resolutions.
- **Temporal redundancy is the prior that makes video learnable.** Adjacent frames are nearly identical, so $H(R_t) = H(V_t \mid V_{t-1}) \ll H(V_t)$: almost all information is in the first frame plus small changes. Measure it — residual entropy and optical-flow magnitude profile a clip's compressibility and difficulty.
- **Classical codecs got here first.** I-frames + motion-compensated P/B-frames + residual coding is the "anchor plus changes" factorization that generative models reuse as I2V conditioning, motion representation, and the residual-coding intuition behind 3D-VAEs.
- **Optical flow is the explicit motion field**, but modern models use motion *implicitly* via temporal attention and the latent's time axis, because explicit flow breaks at occlusion, large motion, and non-rigid content. Flow survives in conditioning, control, and evaluation (VBench motion smoothness).
- **Three representations, three paradigms.** Pixel (cascaded diffusion), continuous latent (diffusion/flow — the modern default), discrete tokens (autoregressive/masked, the world-model interface). The representation largely *selects* the generative method.
- **The token budget $N \approx \frac{T}{c_t}\cdot\frac{H}{c_s}\cdot\frac{W}{c_s}$ governs everything**, and because attention is $O(N^2)$, a 256× token cut is a 65,536× attention cut. A 3D latent at 4×8×8 turns 110M impossible pixel tokens into ~432k feasible ones.
- **Temporal compression does independent work.** Going per-frame-latent → 3D-latent is a 4× token / 16× attention cut purely from compressing time — the entire justification for a 3D-VAE over a per-frame image VAE.
- **The compression ratio is the first design decision, not the last.** It bounds clip length, resolution, batch size, and which GPU you need. Pick $c_t, c_s$ before model size, attention pattern, or sampler — which is exactly why the next post argues the 3D-VAE, not the denoiser, is the real bottleneck.

## Further reading

- **Ho, Salimans, Chan, Fleet, Norouzi, et al. — *Video Diffusion Models* (2022).** The paper that brought diffusion to video with a 3D U-Net; the origin of the temporal-axis-added-to-image-diffusion lineage.
- **Singer, Polyak, Hayes, et al. — *Make-A-Video* (Meta, 2022).** Pixel-space cascaded text-to-video; the clearest example of dodging the value-count wall by generating small and upsampling.
- **Brooks, Peebles, et al. — *Video generation models as world simulators* (Sora technical report, OpenAI, 2024).** The spacetime-patch framing and the scaling thesis; representation-as-tokens at scale.
- **Yang, Teng, Zheng, et al. — *CogVideoX* (Zhipu AI / Tsinghua, 2024).** The canonical open recipe: causal 3D-VAE (4×8×8), expert DiT, 3D full attention — this post's argument, shipped.
- **Yu, Lezama, et al. — *MAGVIT / MAGVIT-v2* (2023–2024).** Magnitude-quantized video tokenization; the modern discrete-token representation and why a good tokenizer can rival continuous latents.
- **Bruce, Dennis, et al. — *Genie: Generative Interactive Environments* (DeepMind, 2024).** Token-space, action-conditioned next-frame prediction; the world-model case for discrete tokens.
- **Teed & Deng — *RAFT: Recurrent All-Pairs Field Transforms for Optical Flow* (2020).** The modern learned optical-flow method; the explicit-motion baseline that learned models absorb.
- **Within this series:** the foundation [why video generation is hard](/blog/machine-learning/video-generation/why-video-generation-is-hard), the payoff [video autoencoders and spatiotemporal compression](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression), and the [capstone playbook](/blog/machine-learning/video-generation/building-with-video-generation-the-playbook). For the latent idea in one fewer dimension, the image series' [latent diffusion and Stable Diffusion](/blog/machine-learning/image-generation/latent-diffusion-and-stable-diffusion) and [variational autoencoders from scratch](/blog/machine-learning/image-generation/variational-autoencoders-from-scratch).
