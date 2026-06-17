---
title: "Latent Video Diffusion: Stable Video Diffusion and AnimateDiff in Practice"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "The two open recipes that actually run on your GPU: how Stable Video Diffusion conditions on a start frame and earns its motion from curated data, and how AnimateDiff's plug-in motion module animates any personalized checkpoint without retraining the base."
tags:
  [
    "video-generation",
    "diffusion-models",
    "video-diffusion",
    "image-to-video",
    "stable-video-diffusion",
    "animatediff",
    "motion-module",
    "diffusers",
    "pytorch",
    "generative-ai",
    "deep-learning",
  ]
category: "machine-learning"
subcategory: "Video Generation"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/latent-video-diffusion-svd-and-animatediff-1.png"
---

You have a photo of a corgi standing in tall grass, and you want it to run. Not a new dog — *that* dog, the one in your photo, for two and a half seconds, fur and all. You also have a folder full of personalized Stable Diffusion checkpoints — a watercolor model, an anime model, a photoreal portrait model you fine-tuned on a friend's face — and you want each of them to *move* without paying to train a video model for every single one. These are the two most common things people actually want from open video generation, and they have two clean, runnable answers that both fit on a single consumer GPU: **Stable Video Diffusion** for the first, **AnimateDiff** for the second.

This post is about those two recipes, hands-on. They are the practical bridge between the architecture we built in [from image diffusion to video diffusion](/blog/machine-learning/video-generation/from-image-diffusion-to-video-diffusion) — where we derived the temporal-mixing block and the inflation trick in the abstract — and the DiT-video frontier that comes later in the series. Both are *latent* video diffusion: they denoise a compressed video latent produced by a [3D / temporal VAE](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression), not raw pixels, which is the only reason they run at all. And both are case studies in this series' recurring tension, **coherence × motion × length × cost** — they buy you coherent short clips cheaply, and the exact way each one falls short is what motivates everything after them.

![Graph showing two routes from an image diffusion backbone to video, one finetuning a single network for image-to-video and one inserting a reusable motion module into any frozen checkpoint](/imgs/blogs/latent-video-diffusion-svd-and-animatediff-1.png)

By the end you will be able to: call `StableVideoDiffusionPipeline` to turn your corgi photo into a clip and tune the `motion_bucket_id` / `fps` / `decode_chunk_size` knobs that actually matter; call `AnimateDiffPipeline` with a `MotionAdapter` over a community checkpoint and stack a MotionLoRA for a camera pan; reason precisely about *why* AnimateDiff's frozen-spatial-plus-trained-temporal design transfers across checkpoints when nothing else does; read the VRAM/decode trade-off so a 25-frame render stops OOMing at the very last step; and say plainly when each recipe is the right tool — and when their ~2-to-4-second ceiling, limited motion range, and identity drift mean you should reach for the DiT-video models instead. The single most important lesson is not architectural at all, and we will keep coming back to it: **SVD's quality came from data curation, not from a cleverer network.** That is the most transferable thing in this entire post.

## 1. The two problems, and why they want different recipes

Adding motion to an image model splits into two genuinely different products, and conflating them is the most common early mistake.

The first product is **image-to-video (I2V)**: you have a real first frame and you want the model to animate *it*. The appearance is given — the model does not have to invent what the corgi looks like, only how it moves. This is the easier of the two problems in a deep sense, because the hardest part of video generation (producing a coherent, high-quality frame at all) is handed to the model for free in frame zero. The model's entire job is to propagate that frame forward in time plausibly. Stable Video Diffusion is built for exactly this.

The second product is **text-to-video (T2V) from your own style**: you have a personalized checkpoint — a LoRA you trained, a community model from Civitai — and you want *that aesthetic* to move, driven by a text prompt, with no first frame. Here appearance is *not* given; the model has to generate the look and the motion together. AnimateDiff's insight is that you can decompose this: let the frozen personalized checkpoint own the look (it already does), and let a separately-trained motion module own the movement, so that the two compose without ever being trained together.

The reason these want different recipes comes straight from the inflation argument we made in the [previous post](/blog/machine-learning/video-generation/from-image-diffusion-to-video-diffusion): appearance (the per-frame marginal distribution) and motion (the temporal coupling between frames) are nearly *orthogonal*. A running dog's leg looks the same whether it is in a photo or a video frame; what video adds is the relationship *between* frames. Because the two are separable, you have a design choice about *where to put the seam*:

- **SVD puts no seam at all** — it inflates one backbone and finetunes the whole thing (spatial and temporal together) into a single image-to-video network. You get a strong, tightly-integrated model, but it is one frozen artifact: it animates *photos*, not your personalized styles, and you cannot swap its aesthetic.
- **AnimateDiff puts the seam between spatial and temporal** — it freezes the spatial weights of *any* checkpoint and inserts a motion module that was trained once, on video, against a fixed base. Because the spatial priors are shared across the whole SD-1.5 ecosystem, the motion module transfers to checkpoints it was never trained on. You get a reusable plug-in at the cost of being locked to the per-frame quality of SD-1.5-class models.

That single architectural fork — *finetune one integrated model* versus *train one transferable module* — is the whole story, and it determines everything downstream: input type, motion range, VRAM, and whether you can reuse your own art. Hold it in your head; the rest of the post fills it in.

One framing to carry through. Both of these are **latent** video diffusion. The denoiser never touches pixels — it operates on a video latent that a VAE has compressed, typically around $8\times$ spatially (and for true 3D-VAEs, also in time). This is not a footnote; it is the load-bearing reason these models fit on a 4090. We covered the compression math in the [video-autoencoders post](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression); here the consequence we will hit repeatedly is that the **VAE decode**, not the denoiser, is usually where you run out of memory, because decode happens at full pixel resolution. Keep that in mind when we get to `decode_chunk_size`.

## 2. Stable Video Diffusion: the architecture in one pass

Stable Video Diffusion (Blattmann et al., 2023) is, structurally, an inflated latent diffusion model. Start from Stable Diffusion 2.1's spatial U-Net — the same convolutions and spatial-attention blocks that already know how to render grass and fur — and interleave **temporal layers** (temporal attention and temporal convolution) between the existing spatial blocks, exactly as we built in the [previous post](/blog/machine-learning/video-generation/from-image-diffusion-to-video-diffusion). The result is a network whose input and output are a latent *clip* of shape $C \times T \times H \times W$ rather than a single latent frame. SVD-XT, the released checkpoint most people use, produces **25 frames at $1024 \times 576$**.

What makes SVD an *image-to-video* model rather than a text-to-video model is how it is conditioned, and this is where the interesting engineering lives. SVD does not take a text prompt at inference. It takes a **start frame** and three scalar knobs, and it injects the start frame in two different ways at once.

The pure image-diffusion machinery here — the latent VAE, the noise schedule, classifier-free guidance — we inherit wholesale from the image series; see [latent diffusion and Stable Diffusion](/blog/machine-learning/image-generation/latent-diffusion-and-stable-diffusion) and [classifier-free guidance](/blog/machine-learning/image-generation/classifier-free-guidance). We will not re-derive any of it. The genuinely video-specific parts are the temporal layers (covered last post) and the conditioning path (covered next). Let me draw the conditioning, because it is the crux of how SVD works.

![Graph of the Stable Video Diffusion conditioning path where the start frame is encoded both as a concatenated VAE latent and a CLIP image embedding while fps and motion-bucket enter as scalar embeddings into the video U-Net](/imgs/blogs/latent-video-diffusion-svd-and-animatediff-3.png)

There are two paths for the start frame and one path for the scalar knobs:

1. **The concatenated VAE latent (the appearance path).** SVD encodes the start frame through the VAE to a single latent frame, then *concatenates* that latent to the noisy video latent along the channel dimension, at every frame and every denoising step. Concretely: the noisy latent clip is $C \times T \times H \times W$; the start-frame latent is $C \times 1 \times H \times W$; you broadcast the start frame across all $T$ positions and stack it on the channel axis, so the U-Net's first conv sees $2C$ channels — the noise it is denoising *plus* a constant reminder of exactly what the first frame's pixels were. This is what pins the generated clip to the *specific* image you provided. Without it, the model would know roughly what to draw but not the exact corgi.

2. **The CLIP image embedding (the semantic path).** SVD also runs the start frame through a CLIP image encoder and feeds the resulting embedding into the U-Net's cross-attention layers — the same cross-attention slots that, in text-to-image SD, receive the *text* embedding. Here they receive an *image* embedding instead. This path carries the high-level semantics ("a corgi, in grass, daylight, this color palette") rather than the exact pixels. It is coarser than the concatenation path but global: every spatial position can attend to it.

The reason you want *both* is a division of labor. The concatenated latent is pixel-precise but local — it anchors low-level appearance. The CLIP embedding is semantic but global — it keeps the *meaning* of the scene stable as motion accumulates. Drop the concatenation and the clip drifts off the exact input image; drop the CLIP embedding and the high-level scene can wander even when local pixels match. Using both is why SVD stays on-subject for its full 25 frames.

3. **The micro-conditioning scalars (`fps_id` and `motion_bucket_id`).** These are two scalar values that get turned into sinusoidal embeddings and added to the timestep embedding, so the network is told, at every layer, *how fast* and *how much* to move. We will spend a whole section on them next, because they are the knobs you will actually touch.

That is the entire conditioning interface: one image in, two scalars, a video clip out. No text. The micro-conditioning idea — passing image properties as scalar embeddings — SVD borrows from SDXL, which conditioned on image size and crop coordinates to fix the aspect-ratio and cropping artifacts of earlier SD models. SVD repurposes the mechanism for *motion* properties, which turns out to be the cleanest way to give the user a motion dial.

#### Worked example: counting SVD's conditioning channels

Make this concrete. SVD operates at $1024 \times 576$. The VAE downsamples $8\times$ spatially, so the latent is $128 \times 72$ with $C = 4$ channels (SD-2.1's VAE is a 2D VAE, applied per frame — SVD does *not* use a temporal VAE; that comes with the DiT generation). The noisy video latent for a 25-frame clip is therefore $4 \times 25 \times 72 \times 128$. The start-frame latent is $4 \times 1 \times 72 \times 128$, broadcast to $4 \times 25 \times 72 \times 128$ and concatenated on the channel axis. So the U-Net's input convolution actually receives **8 channels**, not 4. That doubled first conv is a tiny parameter cost — one extra $3\times3$ conv worth of input channels — and it is *the* mechanism that makes SVD image-to-video rather than text-to-video. The CLIP image embedding, by contrast, is a $1024$-dim vector that enters through cross-attention and costs no change to the latent shape at all. Two start-frame paths, two very different insertion points.

### 2.1 The temporal layers, mechanically

It is worth being precise about what the "temporal layers" inside SVD's U-Net actually compute, because it is the operation that makes the 25 frames cohere, and because it is the same operation AnimateDiff trains as a standalone module. The clearest way to see it is the *reshape trick* we built in the [previous post](/blog/machine-learning/video-generation/from-image-diffusion-to-video-diffusion), restated in the concrete shapes SVD uses.

At a given U-Net layer, the activation tensor has shape $(B, T, C, H, W)$ — batch, frames, channels, height, width. A **spatial** block treats the time axis as part of the batch: it folds $T$ into $B$, giving $(B \cdot T, C, H, W)$, and runs ordinary 2D convolution and 2D self-attention, so each frame is processed in complete isolation. That is the inherited image machinery; it never looks across frames. A **temporal** block does the opposite fold: it reshapes to put $T$ on the *sequence* axis. For temporal *attention*, you reshape to $(B \cdot H \cdot W, T, C)$ — every spatial position becomes an independent sequence of length $T$ — and run self-attention over the $T$ axis, so the feature at position $(h, w)$ in frame $i$ can attend to the feature at the *same* position in every other frame. Then you reshape back. The temporal block sees only the time relationship; the spatial block sees only the within-frame relationship; alternating them is how the network couples space and time without paying for full joint $T \cdot H \cdot W$ attention.

Two details make this work in practice and both matter for the failure modes later. First, the temporal attention needs a **time positional embedding** — without it, the $T$ axis is a permutation-invariant set and the model has no notion of frame order, so motion would have no direction. SVD adds a sinusoidal position embedding along $T$ exactly as a transformer does along token position. Second, when you *insert* a temporal block into a pretrained spatial network, you want training to start from the pretrained image model's behavior, so the temporal block's output projection is **zero-initialized**: at step zero the block is the identity (it adds nothing), and the network behaves exactly like the image model. As training proceeds, the temporal block learns to add motion. This zero-init is why inflation training is stable — you never destroy the spatial prior you are trying to keep. AnimateDiff uses the identical trick on its motion module, which is part of why a freshly-plugged module degrades gracefully rather than catastrophically when it is slightly out of distribution.

The cost of temporal attention is worth a number, because it bounds clip length. Per spatial position, temporal self-attention is $\mathcal{O}(T^2 \cdot C)$ — quadratic in the number of frames. For SVD's $T = 25$ that is $625$ pairwise interactions per position, cheap. But this is *exactly* the term that makes longer clips expensive: doubling to $T = 50$ quadruples the temporal-attention cost, and it is also the term that has no trained meaning past the clip lengths the model saw. The quadratic-in-frames temporal attention is both the cost wall and the length wall, and it is the single most important reason SVD and AnimateDiff are short-clip models. The [spacetime-DiT post](/blog/machine-learning/video-generation/video-diffusion-transformers) is in large part about how to spend this $T^2$ budget more cleverly.

### 2.2 The EDM noise schedule and why SVD's sampler is different

One more piece of SVD's recipe is easy to get wrong when you adapt the code: SVD is trained and sampled under the **EDM** formulation (Karras et al., 2022), not the DDPM $\beta$-schedule that base Stable Diffusion uses. In EDM the noise level is parameterized directly as a continuous standard deviation $\sigma$, and the network is wrapped in *preconditioning* functions $c_\text{skip}(\sigma)$, $c_\text{out}(\sigma)$, $c_\text{in}(\sigma)$, $c_\text{noise}(\sigma)$ that scale the input, output, and skip connection so the network always sees a well-conditioned target regardless of noise level. The denoiser $D_\theta$ is

$$
D_\theta(x; \sigma) = c_\text{skip}(\sigma)\, x + c_\text{out}(\sigma)\, F_\theta\big(c_\text{in}(\sigma)\, x;\, c_\text{noise}(\sigma)\big),
$$

where $F_\theta$ is the raw U-Net. We will not re-derive EDM — the [samplers deep dive](/blog/machine-learning/image-generation/samplers-deep-dive) and [score-based view](/blog/machine-learning/image-generation/score-based-models-and-the-sde-view) in the image series cover the why. The *practical* consequence is the one that bites: SVD's sampling uses a continuous-$\sigma$ schedule with a small number of steps (the released default is around 25), and the maximum $\sigma$ is set *high* — SVD samples from a much larger top noise level than image SD, because a video latent is higher-dimensional and needs more noise to fully destroy structure. If you wire SVD into a vanilla DDPM scheduler with the wrong $\sigma_\text{max}$, you get washed-out or static clips. In `diffusers` the `StableVideoDiffusionPipeline` ships the correct `EulerDiscreteScheduler` configuration, which is why the code in this post does not touch the scheduler for SVD but *does* explicitly configure it for AnimateDiff — AnimateDiff inherits SD-1.5's $\beta$-schedule and is sensitive to getting it right. The lesson: a video model carries its noise schedule as part of its trained identity, and swapping it is not free.

There is also a guidance subtlety worth flagging. SVD uses **classifier-free guidance with a per-frame guidance scale that increases across the clip** — the released pipeline ramps the guidance from a low value on the first frame to a higher value on the last (the `min_guidance_scale` / `max_guidance_scale` arguments). The intuition: the first frame is strongly pinned by the concatenated start-frame latent, so it needs little guidance; later frames have drifted further from that anchor and benefit from stronger guidance to stay on-subject. This linear guidance ramp is a small but real piece of why SVD holds identity across its 25 frames, and it is invisible unless you read the pipeline source — most users never touch it, but it is doing quiet work.

## 3. The micro-conditioning knobs: motion_bucket_id and fps

Here are the two dials that change SVD's output the most, and the two that people most often set wrong.

**`motion_bucket_id`** is a learned conditioning signal in the range roughly 0–255 that tells the model *how much motion* to put in the clip. During training, SVD computed a motion score for each training clip (a measure of how much the frames change — derived from optical flow magnitude) and bucketed it, then conditioned the model on that bucket. At inference you set the bucket directly. Low values (≈ 20–60) produce subtle, gentle motion — a slight breeze, a small parallax. The default of **127** is a moderate amount. High values (≈ 180–255) ask for large, dramatic motion. Crucially, this is not a physically meaningful unit; it is "where on the training distribution of motion magnitudes do you want this clip to sit." Push it to the ceiling and you are asking the model to move objects far between adjacent frames, which is exactly where temporal coherence breaks (more on that in the stress test).

**`fps_id`** tells the model the *frame rate* the clip should represent, which — combined with the fixed 25-frame count — implicitly sets the *duration* and therefore the *per-frame displacement*. At 25 fps, 25 frames is one second; at 7 fps (a common SVD default), 25 frames is about 3.5 seconds. The subtlety: `fps` and `motion_bucket_id` *interact*. A high frame rate means adjacent frames are close in time, so even large total motion is small *per-frame motion* — easy to keep coherent. A low frame rate spreads the same 25 frames over more wall-clock time, so each frame jumps further from the last — harder to keep coherent. The practical instinct, which we will justify in the stress test, is: **if you want more motion, prefer raising it via the scene and fps rather than slamming `motion_bucket_id` to 255**, because more frames at lower per-frame displacement is almost always more coherent than fewer frames at higher displacement.

There is a third knob worth naming: the **noise augmentation strength** (`noise_aug_strength` in `diffusers`). SVD adds a small amount of noise to the *conditioning* start frame during training and inference. A little noise on the conditioning image loosens the model's grip on the exact input, which paradoxically *increases* motion and diversity — too tight a grip on frame zero and the model produces a near-frozen clip. Raising `noise_aug_strength` from its default (≈ 0.02) toward 0.1 is a second, more subtle, motion dial.

Here is the honest mental picture of these three knobs as a control surface. `motion_bucket_id` is the coarse "how much movement" request; `fps_id` reshapes that request into per-frame displacement; `noise_aug_strength` controls how literally the model treats your start frame. They are not independent — they all push on the same underlying quantity, *how far things move between adjacent frames* — which is precisely the quantity that governs whether the clip stays coherent.

```python
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

# Load SVD-XT in fp16; model CPU offload keeps peak VRAM down on a 24GB card.
pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe.enable_model_cpu_offload()          # stream weights; saves several GB
pipe.unet.enable_forward_chunking()      # chunk the temporal attention to cut activation memory

image = load_image("corgi_in_grass.png").resize((1024, 576))

generator = torch.manual_seed(42)
frames = pipe(
    image,
    num_frames=25,                       # SVD-XT native length
    fps=7,                               # implicit duration: 25/7 ~ 3.5s of footage
    motion_bucket_id=127,                # moderate motion; raise toward 180 for more
    noise_aug_strength=0.02,             # raise toward 0.1 to loosen the start-frame grip
    decode_chunk_size=8,                 # decode 8 latent frames per VAE pass (VRAM dial)
    num_inference_steps=25,
    generator=generator,
).frames[0]

export_to_video(frames, "corgi_run.mp4", fps=7)
```

Every flag in that call is a real `diffusers` argument and every one maps to something we just discussed. `num_frames=25` is the trained clip length — push past it and you leave the model's trained regime (stress test). `fps` and `motion_bucket_id` are the motion knobs. `decode_chunk_size` is the VRAM dial we are about to dissect. `enable_model_cpu_offload` and `enable_forward_chunking` are the two lines that move SVD from "needs an A100" to "runs on a 4090."

## 4. The decode_chunk_size trade-off: where SVD actually OOMs

This is the single most important practical detail in running SVD, and it is the cleanest illustration of the series-wide claim that the *VAE decode*, not the denoiser, is the memory wall.

Walk through the memory profile of a generation. The denoiser runs entirely in latent space: it iterates 25 denoising steps over a $4 \times 25 \times 72 \times 128$ latent. That tensor is small — a few hundred megabytes of activations even with the U-Net's intermediate feature maps. The denoiser is *comfortable*. Then, at the very end, you have a clean latent clip and you must turn it back into pixels: the VAE decoder takes each of the 25 latent frames and upsamples it $8\times$ in each spatial dimension to $1024 \times 576$ RGB. The decoder's activation maps live at *pixel* resolution, and there are wide intermediate channels. If you decode all 25 frames at once, you allocate the decoder's peak activation footprint times 25 frames simultaneously — and *that* is what OOMs, on a card that sailed through the entire denoising loop.

`decode_chunk_size` is the fix. Instead of decoding all 25 latent frames in one VAE forward pass, decode them in chunks of `decode_chunk_size` frames at a time, freeing memory between chunks. Set it to 25 and you decode in one pass at peak VRAM. Set it to 8 and you decode in four passes (8 + 8 + 8 + 1) at roughly a third of the peak. Set it to 1 and you decode frame by frame at the absolute minimum memory, at the cost of 25 separate decode passes.

![Matrix showing how lowering the SVD decode chunk size cuts peak decode VRAM while adding decode passes and changing whether the run fits on a 12GB card](/imgs/blogs/latent-video-diffusion-svd-and-animatediff-7.png)

The trade is memory for wall-clock time, and it is close to linear in the chunk size on the memory side. The decode is a relatively small fraction of total generation time (the denoising loop dominates), so even `decode_chunk_size=2` adds only modest overall latency while dramatically lowering the peak. This is why, on a 12GB card, the difference between a successful render and an OOM at the final step is one integer argument.

There is a second, orthogonal memory tool: **VAE tiling and slicing** (`pipe.vae.enable_tiling()` / `enable_slicing()`), which decode each frame in *spatial* tiles rather than all at once, trading a little seam-blending overhead for a large drop in per-frame peak. For very high resolutions you combine both — small `decode_chunk_size` (fewer frames at once) *and* tiling (smaller spatial extent at once). Together they make the decode footprint nearly independent of clip length and resolution, at the cost of more, smaller passes.

#### Worked example: the SVD memory wall on an RTX 4090 24GB

Concretely, on an RTX 4090 (24GB) running SVD-XT at $1024\times576$, 25 frames, 25 steps, fp16 with `enable_model_cpu_offload`: the denoising loop runs comfortably and a full clip takes on the order of **15–40 seconds** depending on steps and whether `torch.compile` is warmed up. With `decode_chunk_size=25` (decode all at once), peak VRAM during decode spikes hard and can approach or exceed the card's budget at this resolution; the run is fragile. Drop to `decode_chunk_size=8` and peak decode VRAM falls to a comfortable margin with negligible added time — call it under a second of extra decode. Drop to `decode_chunk_size=2` and the same clip will decode on a **12GB** card (e.g., a 3060) at the cost of a handful of extra decode passes. The denoiser never moved; only the decode did. The lesson generalizes to *every* latent video model in this series: budget your VRAM for the decode, not the denoise, and keep a frame-chunking dial in reach.

## 5. SVD's real lesson: data curation beat architecture

If you remember one thing from the Stable Video Diffusion paper, make it this: **the authors' central empirical finding was that data curation mattered more than the architecture.** They built a strong, conventional inflated latent diffusion model — but the thing that moved quality was a disciplined, multi-stage *data* pipeline, and they showed it with ablations. In a field obsessed with architecture, that is a genuinely important and under-internalized result.

SVD trains in three stages, and the staging is the point.

![Stack diagram of Stable Video Diffusion's three training stages from image pretraining through curated large-scale video pretraining to a small high-quality finetune, with curation filters as the decisive lever](/imgs/blogs/latent-video-diffusion-svd-and-animatediff-2.png)

**Stage 1 — image pretraining.** Start from a strong text-to-image model (SD-2.1). This gives the network its spatial prior — how to render a convincing frame — before it ever sees a video. This is the inflation starting point and it is free; you inherit it.

**Stage 2 — video pretraining on a *curated* large dataset.** This is the stage the paper is really about. The raw video data the authors collected was enormous and *filthy* — clips with cuts mid-scene, static slideshow-like footage with no real motion, motion-blurred garbage, clips whose captions did not match their content, and clips with low aesthetic quality. Training on the raw pile produced a weak model. The authors built a *filtering* pipeline and showed that a model trained on the curated subset beat a model trained on far more raw data. The filters were, roughly:

- **Cut detection and clip splitting.** Detect scene cuts (a hard transition between unrelated shots) and split videos at them, so each training clip is a single continuous shot. A clip that contains a cut teaches the model to *teleport* — exactly the discontinuity you do not want.
- **Motion filtering.** Compute an optical-flow-based motion score per clip and *remove the static ones*. A huge fraction of scraped "video" is effectively a still image with a watermark — train on it and your model learns to produce frozen clips. (This is also the score that became `motion_bucket_id`: once you have a motion score per clip, you can both filter on it *and* condition on it.)
- **Caption and text filtering.** Re-caption clips with a strong captioner and filter on caption quality and alignment; remove clips dominated by on-screen text (subtitles, logos) that the model would otherwise learn to hallucinate.
- **Aesthetic filtering.** Score frames with an aesthetic predictor and drop the ugly tail, the same idea LAION-aesthetics used for image models.

The result of stage 2 is the "video pretrained" model — competent motion, broad coverage, but not yet polished.

**Stage 3 — high-quality finetuning.** Take the pretrained model and finetune it on a *small*, hand-curated set of high-quality, high-resolution clips. This is the stage that sharpens the output — it is analogous to instruction-tuning a language model on a small clean set after pretraining on a large noisy one. The paper showed that this final finetune, on top of a *well-curated* pretraining set, was what produced the released quality. Finetuning on top of a *poorly*-curated pretrain did not recover the gap.

The deep lesson, and the reason it is the most transferable thing in this post: **video data is far dirtier than image data, and the dirt is specifically temporal.** A static clip, a clip with a cut, a clip with mismatched captions — these are failure modes that simply do not exist for single images, and they poison temporal learning in ways that no amount of architecture fixes. If you ever train or finetune a video model, your first and highest-leverage investment is the curation pipeline, not the network. SVD is the empirical proof.

#### Worked example: the curation ablation, in spirit

The paper's ablation runs like this (I am stating the *shape* of the result, which is robust, rather than quoting exact FVD digits I would have to reconstruct). Take the same architecture and the same compute budget. Train model A on the *full, uncurated* video pile. Train model B on a *curated subset* — smaller in raw clip count, but filtered for motion, cuts, captions, and aesthetics. Evaluate both with human preference and with FVD on a held-out set. Model B wins, and it wins *despite seeing fewer clips*, because the curated clips carry cleaner temporal signal per example. The takeaway you can act on: when your video model produces frozen or teleporting clips, suspect your data before your architecture, and audit specifically for static clips and undetected cuts — those two failure modes account for a disproportionate share of bad temporal behavior.

## 6. AnimateDiff: motion as a plug-in

Now the second recipe, and the more conceptually beautiful one. AnimateDiff (Guo et al., 2023) asks a question SVD does not: *can the motion be a separate, reusable component?* Can you train a temporal module **once**, against a fixed base model, and then drop it into **any** personalized checkpoint built on that base — without retraining either the module or the checkpoint?

The answer is yes, and the reason it works is the orthogonality of appearance and motion, stated sharply. Here is the setup. Take base Stable Diffusion 1.5. The community has produced *thousands* of personalized checkpoints by fine-tuning or LoRA-tuning SD-1.5 — watercolor models, anime models, photoreal portrait models. Critically, all of these checkpoints **share the same architecture and live near the same point in weight space**: they are SD-1.5 with the spatial weights nudged toward a style. Their *spatial* layers differ; their notion of *space* — the shape of the network, the resolution, the channel layout — is identical because they all inflated from the same base.

![Before and after comparison showing a frozen personalized checkpoint producing a single still on its own and the same checkpoint producing a coherent sixteen-frame clip once a trained motion module is inserted](/imgs/blogs/latent-video-diffusion-svd-and-animatediff-4.png)

AnimateDiff's move: train a **motion module** — a stack of temporal-attention layers, inserted between the frozen spatial blocks of *base* SD-1.5 — on a video dataset (WebVid-10M). During this training, the SD-1.5 spatial weights are **frozen**; only the motion module learns. The module's only job is to learn *general motion priors* — how things move in the world — in the shared SD-1.5 latent space.

Now the magic. Because the motion module was trained against the *shared* spatial representation, and because every personalized checkpoint speaks that same spatial language, you can **lift the spatial weights of the base out and drop a personalized checkpoint's spatial weights in**, keep the motion module exactly as trained, and it *still works*. The motion module never saw the watercolor model during training, but it does not need to — it learned how features in the SD-1.5 latent space evolve over time, and the watercolor model produces features in *that same space*. Motion transfers because the substrate is shared.

This is the "motion as a plug-in" idea, and it is genuinely powerful for the community, for one concrete reason: **the marginal cost of animating a new checkpoint is zero.** Someone trains a motion module once; everyone with any SD-1.5 checkpoint gets video for free, no per-checkpoint training, no GPUs spent. That is why AnimateDiff exploded in the open ecosystem in a way a monolithic model like SVD never could — it composes with the thousands of styles people had already built.

Here is the inference path, showing how the pieces stack:

![Graph of the AnimateDiff inference path where a frozen personalized backbone and an optional style LoRA carry appearance while a trained motion module and an optional MotionLoRA carry movement before fusing into the animated clip](/imgs/blogs/latent-video-diffusion-svd-and-animatediff-5.png)

Four components fuse into one U-Net at inference: the frozen personalized backbone (appearance), an optional style LoRA (more appearance), the motion module (movement), and an optional MotionLoRA (a specific *kind* of movement — we will get to it). Appearance components and motion components stack independently, which is the whole point.

The `diffusers` API mirrors this composition exactly. You load a `MotionAdapter` (the motion module) and a community SD-1.5 checkpoint separately, then combine them in an `AnimateDiffPipeline`:

```python
import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
from diffusers.utils import export_to_gif

# The motion module: trained once, on video, against base SD-1.5.
adapter = MotionAdapter.from_pretrained(
    "guoyww/animatediff-motion-adapter-v1-5-3",
    torch_dtype=torch.float16,
)

# ANY SD-1.5 community checkpoint provides the appearance. Swap this line to
# re-style the same motion module — the adapter does not change.
pipe = AnimateDiffPipeline.from_pretrained(
    "emilianJR/epiCRealism",             # a photoreal SD-1.5 checkpoint
    motion_adapter=adapter,
    torch_dtype=torch.float16,
)

# AnimateDiff wants a linear-beta DDIM schedule with these settings.
pipe.scheduler = DDIMScheduler.from_config(
    pipe.scheduler.config,
    clip_sample=False,
    beta_schedule="linear",
    timestep_spacing="linspace",
    steps_offset=1,
)
pipe.enable_vae_slicing()                # cut VAE-decode peak memory
pipe.enable_model_cpu_offload()

output = pipe(
    prompt="a corgi running through tall grass, golden hour, cinematic",
    negative_prompt="low quality, worst quality, blurry",
    num_frames=16,                       # AnimateDiff's native context length
    guidance_scale=7.5,
    num_inference_steps=25,
    generator=torch.manual_seed(0),
)
export_to_gif(output.frames[0], "corgi_animatediff.gif")
```

The line that carries the entire idea is the checkpoint name in `from_pretrained`. Swap `"emilianJR/epiCRealism"` for a watercolor or anime SD-1.5 checkpoint and you get the *same motion* in a completely different style, with no other change and no retraining. That swap is the plug-in property made literal. Note the scheduler configuration: AnimateDiff is sensitive to the noise schedule it was trained under (linear beta), and getting this wrong is a common cause of mushy or static output — a small reminder that these recipes carry trained-in assumptions you cannot freely change.

## 7. Why frozen-spatial-plus-trained-temporal transfers (the science)

Let me make the transfer argument rigorous, because "appearance and motion are orthogonal" is a slogan until you can say *why* the slogan licenses dropping in an unseen checkpoint.

Model the personalized checkpoints as points in weight space. Base SD-1.5 has spatial weights $\theta_0$. A personalized checkpoint is $\theta_0 + \Delta\theta$, where $\Delta\theta$ is the fine-tuning (or LoRA) update that bends the model toward a style. The empirical fact that makes the community work — and it is empirical, not guaranteed — is that these $\Delta\theta$ are *small relative to* $\theta_0$ and they live in the spatial layers. The network's *functional structure* — its resolution, its channel layout, the geometry of its latent activations at each layer — is preserved. A personalized checkpoint paints a different style *on the same canvas*: the activation at a given layer still represents the same kind of thing (a feature map of the same shape, encoding edges/textures/parts in the same coordinate system), just colored toward watercolor or anime.

The motion module is a function $M$ that takes the *temporal stack* of spatial activations and mixes them across frames. It was trained to take activations *produced by $\theta_0$* and predict how they should evolve over time. The key question is: does $M$ still behave well on activations produced by $\theta_0 + \Delta\theta$?

It does, to the extent that the activation *distribution* is approximately preserved by the fine-tuning. Because $\Delta\theta$ shifts style but not structure, the activations the motion module receives from a personalized checkpoint are *in-distribution* for $M$ — they live in the same activation space, with the same dimensionality and roughly the same statistics, that $M$ was trained on. $M$ learned a motion prior over *that space*, and a watercolor corgi's activations and a photoreal corgi's activations are both points in it. Motion is a property of how features move; the features themselves can be re-styled without changing how a "leg" feature ought to translate from frame to frame.

This also predicts exactly *when* the transfer breaks, which is the honest part. If a checkpoint's $\Delta\theta$ is *large* — a heavily fine-tuned model with a radically different aesthetic, or a model fine-tuned at a different resolution, or (especially) a checkpoint based on a *different base architecture* (SDXL motion modules do not work on SD-1.5 checkpoints and vice versa) — then its activations drift *out of distribution* for the motion module, and you get degraded or broken motion. The plug-in property is not magic; it is the statement that *small style shifts preserve the activation distribution the motion module was trained on*, and it holds exactly as well as that statement holds. This is why AnimateDiff motion adapters are versioned *per base model* (`animatediff-motion-adapter-v1-5-*` for SD-1.5, separate adapters for SDXL): the base defines the activation space, and the adapter is only valid within it.

There is a clean way to connect this to the inflation argument from the [previous post](/blog/machine-learning/video-generation/from-image-diffusion-to-video-diffusion). Inflation freezes spatial weights and trains temporal weights *for one model*. AnimateDiff takes that one step further: it freezes spatial weights, trains temporal weights against a *base*, and then observes that the trained temporal weights are valid for *any model sharing that base's activation space*. The first is "reuse appearance within a model"; the second is "reuse motion across a family of models." Both rest on the same orthogonality — appearance is the per-frame marginal, motion is the temporal coupling, and the coupling is learnable in a way that is largely indifferent to the style of the marginals.

## 8. MotionLoRA: steering the camera

The motion module gives you *generic* motion — things move, plausibly, in whatever way the WebVid prior suggests. But often you want a *specific* motion: a camera that pans left, zooms in, tilts up, or rolls. AnimateDiff's answer is **MotionLoRA**: a small LoRA applied *to the motion module* that biases it toward a particular camera movement.

The idea reuses [LoRA](/blog/machine-learning/image-generation/personalization-dreambooth-textual-inversion-lora) — the same low-rank adaptation that personalizes appearance in image models — but applied to the *temporal* layers instead of the spatial ones. You take a small set of clips exhibiting a specific camera motion (say, twenty clips of a leftward pan), and train a low-rank update on the motion module's weights so it leans toward that motion. Because it is a LoRA, it is tiny (a few megabytes), trains fast (on a handful of clips), and — crucially — *composes*: you can load multiple MotionLoRAs at weighted strengths and blend "pan left" with "zoom in" to get a diagonal dolly.

```python
# Stack a camera-motion MotionLoRA on top of the motion module.
pipe.load_lora_weights(
    "guoyww/animatediff-motion-lora-zoom-in",
    adapter_name="zoom_in",
)
pipe.load_lora_weights(
    "guoyww/animatediff-motion-lora-pan-left",
    adapter_name="pan_left",
)
# Blend two camera motions at chosen strengths -> a diagonal move.
pipe.set_adapters(["zoom_in", "pan_left"], adapter_weights=[0.8, 0.6])

output = pipe(
    prompt="a corgi running through tall grass, golden hour, cinematic",
    num_frames=16,
    guidance_scale=7.5,
    num_inference_steps=25,
    generator=torch.manual_seed(0),
)
```

The architecture here is worth appreciating as a layered control surface. The frozen checkpoint controls *what the scene looks like*. The motion module controls *that things move at all*. The MotionLoRA controls *how the camera moves*. Each is a separately-trained, separately-swappable component, and they stack at inference with simple weighted blending. This is the most modular point the open ecosystem reached before the DiT-video frontier folded everything back into a single end-to-end model: appearance, motion, and camera as three independent plug-ins. It is a beautiful design, and its limits — which we turn to next — are exactly what justified folding it back together.

### 8.1 Training a motion module (the "once" in "trained once")

The phrase "trained once on video" hides the only expensive step in the whole AnimateDiff story, so it is worth seeing what that training actually is — both because it demystifies the recipe and because the same loop, scaled down, is how you would *fine-tune* a motion module for your own domain (say, a module that animates medical imaging, or anime specifically, where the generic WebVid prior underperforms).

The setup is the inflation training from the [previous post](/blog/machine-learning/video-generation/from-image-diffusion-to-video-diffusion), made concrete. Freeze base SD-1.5 entirely. Insert temporal-attention blocks (the motion module) into the U-Net, with zero-initialized output projections so the network starts as the identity. Now train *only* the motion module on a video dataset, with the *same* diffusion objective base SD-1.5 was trained with — there is no new loss. For each training step: sample a 16-frame clip, encode every frame through the frozen VAE to a latent clip, sample one shared timestep $t$, add noise, and ask the inflated U-Net to predict it. The gradient flows only into the temporal blocks. That is the entire algorithm.

```python
# Sketch of the AnimateDiff motion-module training step (frozen base, train temporal only).
import torch
import torch.nn.functional as F

# Freeze everything, then unfreeze only the inserted temporal blocks.
for p in unet.parameters():
    p.requires_grad_(False)
for name, p in unet.named_parameters():
    if "temporal" in name or "motion" in name:   # the inserted motion module
        p.requires_grad_(True)

optimizer = torch.optim.AdamW(
    (p for p in unet.parameters() if p.requires_grad), lr=1e-4
)

for clip, prompt in dataloader:                   # clip: (B, T=16, 3, H, W)
    with torch.no_grad():
        b, t = clip.shape[:2]
        latents = vae.encode(
            clip.flatten(0, 1)                    # fold time into batch for the 2D VAE
        ).latent_dist.sample() * vae.config.scaling_factor
        latents = latents.unflatten(0, (b, t))    # back to (B, T, C, h, w)
        text = text_encoder(tokenize(prompt))     # frozen SD-1.5 text encoder

    noise = torch.randn_like(latents)
    timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (b,))
    noisy = scheduler.add_noise(latents, noise, timesteps)   # one shared t per clip

    pred = unet(noisy, timesteps, encoder_hidden_states=text).sample
    loss = F.mse_loss(pred, noise)                # the SAME epsilon-prediction loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

Three things in that loop are the whole point. The VAE encode folds time into the batch (`flatten(0, 1)`) because SD-1.5's VAE is a per-frame 2D VAE — same reason SVD's is. The timestep is *one scalar shared across the clip*, so every frame is corrupted to the same noise level and the network denoises a coherent clip rather than a mixture. And the loss is `F.mse_loss(pred, noise)` — character-for-character the image $\epsilon$-prediction loss, because, as we established at length last post, the loss does not change when you go to video; only the network does. The motion module is the *only* thing learning, and what it learns is the temporal prior over SD-1.5's activation space.

The data requirement is the expensive part, and it is where SVD's curation lesson reappears. The original AnimateDiff trained on WebVid-10M, a large scraped video-caption set — and WebVid is *notoriously* dirty: watermarks, static clips, mismatched captions. AnimateDiff's later versions and community retrains improved markedly by filtering for exactly the things SVD filtered for (motion, cuts, captions). If you fine-tune your own motion module, budget your effort on curating a few thousand *clean, moving, well-captioned* clips in your target domain rather than scraping more. A small clean motion dataset beats a large dirty one — the same conclusion, reached independently, that the SVD authors proved with ablations. The two recipes converge on the same lesson from opposite directions.

#### Worked example: the cost of a motion-module fine-tune

Concretely, fine-tuning a MotionLoRA (the cheapest case) on a specific camera motion takes on the order of *tens* of clips and minutes-to-an-hour on a single 24GB GPU, because the LoRA is a few-megabyte low-rank update and you are only nudging an already-trained module. Training a *full* motion module from scratch is a different scale: it is the inflation training above on a large curated video set, which is days on multiple GPUs — that is the "once" that someone in the community pays so everyone else gets video for free. The asymmetry is the entire economic argument for the plug-in design: one expensive training (the module), then unlimited zero-cost reuse (every checkpoint) and cheap specialization (MotionLoRA). It is the same shape as foundation-model pretraining followed by cheap LoRA fine-tunes, applied to *motion* instead of *language*.

## 9. SVD vs AnimateDiff: the decision

Time to put them side by side on the axes that decide which one you reach for. The split is clean enough that, once you know your inputs, the choice is almost forced.

![Matrix comparing Stable Video Diffusion and AnimateDiff across conditioning input, clip length, motion range, base-model reuse, and peak VRAM on a consumer GPU](/imgs/blogs/latent-video-diffusion-svd-and-animatediff-6.png)

Read it axis by axis:

- **Input.** SVD takes a *start frame* (image-to-video). AnimateDiff takes a *text prompt* plus *any checkpoint* (text-to-video in your style). This is the first fork: do you have a first frame, or do you have a style?
- **Clip length.** SVD-XT: 25 frames. AnimateDiff: 16 frames natively (extendable with context-window tricks, but 16 is the trained length). Both are short — roughly 2 seconds, occasionally pushed to 3–4. Neither makes long video; that is a different post.
- **Motion range.** SVD generally produces *larger, more coherent* motion — it was finetuned end-to-end for motion and has the `motion_bucket_id` dial. AnimateDiff's motion is *gentler* and more prone to drift, because the motion module is generic and the per-frame quality is capped at SD-1.5.
- **Base-model reuse.** This is AnimateDiff's decisive win: *any* SD-1.5 checkpoint, for free. SVD is a single frozen artifact — you animate photos, not your styles, and you cannot re-skin it.
- **VRAM.** AnimateDiff is lighter (SD-1.5-class, ~8–12GB with offload) and SVD is heavier (~14–20GB at $1024\times576$ with the decode dial set sensibly). Both fit a 24GB card comfortably; AnimateDiff fits smaller cards more easily.

| Property | Stable Video Diffusion (SVD-XT) | AnimateDiff (v3 + SD-1.5) |
| --- | --- | --- |
| Task | Image-to-video (start frame in) | Text-to-video, in your checkpoint's style |
| Native frames | 25 (~2.5s @ ~10fps) | 16 (~2s @ ~8fps) |
| Resolution | $1024\times576$ | $512\times512$-class (SD-1.5) |
| Per-frame quality | SD-2.1-class, strong | SD-1.5-class, capped by checkpoint |
| Motion range | Larger; `motion_bucket_id` dial | Gentler; generic prior |
| Camera control | Implicit via fps/motion bucket | Explicit via MotionLoRA |
| Base-model reuse | None (single frozen model) | Any SD-1.5 checkpoint, free |
| Conditioning | VAE-concat + CLIP image embed | Text + frozen spatial weights |
| Peak VRAM (RTX 4090) | ~14–20 GB (decode-dial dependent) | ~8–12 GB |
| Seconds per clip (4090) | ~15–40s | ~10–25s |
| Best when | You have a first frame | You have a personalized style |

The honest one-line decision: **if you can supply a first frame, use SVD** — handing the model frame zero removes the hardest part of the problem and yields stronger, larger motion. **If you must reuse a personalized style and only have a text prompt, use AnimateDiff** — it is the only one that composes with your existing checkpoints, at zero marginal training cost. The decision tree below captures the two questions that resolve it.

#### Worked example: an end-to-end product decision

Suppose you are building a feature that turns a user's uploaded product photo into a 3-second looping clip for an ad. Walk the decision. Do you have a first frame? Yes — the uploaded photo *is* frame zero, and it is the exact thing the user wants animated; inventing a new product would be a bug. That single fact eliminates AnimateDiff and text-to-video entirely: you would be throwing away the most valuable conditioning signal you have. Use SVD. Now the knobs: the product should move gently (a slow rotate, a subtle parallax), so `motion_bucket_id` lands around 60–100, not the default 127 and certainly not 200; `fps` around 8–10 to make the loop feel smooth; `noise_aug_strength` left near default so the product stays recognizable. On the serving side, you are on consumer GPUs at scale, so `decode_chunk_size=4` and VAE tiling keep you off the OOM cliff, and you accept ~20–30 seconds per clip. Total decision time: about thirty seconds of reasoning, all of it forced by "do you have a first frame."

Now change one fact: the user uploads *no* photo and instead picks a brand art-style you shipped as a fine-tuned SD-1.5 checkpoint, plus a text prompt ("our mascot waving"). Now you have a style and a prompt but no frame. AnimateDiff is the *only* recipe here that animates *your* checkpoint — SVD cannot use your style at all, and training a bespoke SVD per brand style is absurd. Plug your checkpoint into `AnimateDiffPipeline`, load the motion module, add a gentle MotionLoRA for a slow push-in, and generate 16 frames. Same problem shape (animate something for an ad), opposite recipe — and the fork was decided entirely by *what conditioning you have*, exactly as the tree says.

![Tree showing that the choice between image-to-video SVD, AnimateDiff, and a DiT-video model follows from whether you have a start frame and whether you need to reuse a personalized style checkpoint](/imgs/blogs/latent-video-diffusion-svd-and-animatediff-8.png)

## 10. Where they break: the limits that motivate the frontier

Both recipes are excellent at what they do and both hit walls in the same three places. Naming the walls precisely is what sets up the rest of the series, because the DiT-video frontier exists largely to climb them.

**The length wall: ~2–4 seconds, hard.** SVD trained on 25-frame clips; AnimateDiff on 16. The temporal layers learned positional relationships *within that window* and have no notion of what frame 40 should look like — there is no trained signal that far out. Push `num_frames` past the trained length and you do not get a longer coherent clip; you get repetition, degradation, or a clip that loops back on itself, because the temporal attention is operating outside the positions it ever saw. People extend AnimateDiff with **context-window / sliding-window** schemes (denoise overlapping 16-frame windows and blend the overlaps) and SVD with frame-conditioning chaining, but these are stopgaps that fight error accumulation. Genuinely long video — minutes, with identity held — is a different architecture problem, the subject of the long-video post later in the series.

The sliding-window idea is worth seeing concretely, because it is the standard community workaround and it makes the *cost* of fighting the length wall visible. You split the desired long sequence into overlapping windows of the trained length, denoise each window, and blend the overlap regions so the seams do not pop:

```python
# Sketch: extend an AnimateDiff/SVD-style model past its trained window by
# denoising overlapping context windows and averaging the overlaps each step.
context_len, stride = 16, 12          # 16-frame windows, 4-frame overlap (16-12)
windows = [(i, i + context_len)
           for i in range(0, total_frames - context_len + 1, stride)]

for t in scheduler.timesteps:         # one shared diffusion schedule for the whole sequence
    noise_pred = torch.zeros_like(latents)   # (1, total_frames, C, h, w)
    counts = torch.zeros(total_frames)       # how many windows touched each frame
    for (a, b) in windows:
        window_pred = unet(latents[:, a:b], t, encoder_hidden_states=text).sample
        noise_pred[:, a:b] += window_pred    # accumulate
        counts[a:b] += 1
    noise_pred /= counts.view(1, -1, 1, 1, 1)  # average overlapping predictions
    latents = scheduler.step(noise_pred, t, latents).prev_sample
```

Notice what this buys and what it costs. It produces a longer clip without retraining, but it pays for every extra frame with more U-Net forward passes per step (one per window), and — more importantly — it does *not* create genuine long-range coherence: two frames in non-overlapping windows are only coupled *indirectly* through the chain of overlaps, so identity still drifts across many windows. The averaging hides seams; it does not give the model a memory of frame 1 when it is denoising frame 60. That residual drift is exactly why the field moved to architectures with real long-context mechanisms rather than window-stitching, which the long-video post takes up.

**The motion-range wall.** Both models keep coherence only up to a certain per-frame displacement. Turn SVD's `motion_bucket_id` to its ceiling and you ask objects to move far between adjacent frames; the temporal layers cannot track motion that large and you get **ghosting** (a faint copy of the object at its old position), **warping** (objects stretch as the model interpolates between incompatible positions), or **tearing**. AnimateDiff degrades earlier still, because its motion prior is generic and its per-frame quality is lower. The fundamental cause is the one we have hit before: temporal *attention* can match a feature to its moved location by *content*, which helps, but past a point the displacement exceeds what the trained window supports and coherence collapses. The practical workaround — raise the frame rate so adjacent frames are closer in time rather than asking for larger per-frame jumps — buys you headroom but not unlimited motion.

**The identity-drift wall.** Over even a short clip, fine details can drift: the exact pattern of the corgi's fur, the precise shape of an ear, the color of a collar. SVD's CLIP-embedding path fights this by re-anchoring high-level semantics every step, and its concatenated start-frame latent pins low-level appearance — which is *why* SVD drifts less than AnimateDiff, whose only appearance anchor is the text prompt. But neither holds identity *perfectly*, and the drift compounds with clip length. This is the wall that becomes catastrophic in long-video rollout, where small per-frame drifts accumulate into a different-looking subject by second 20 — the error-accumulation problem the autoregressive-rollout post tackles head-on.

#### Worked example: pushing SVD past its trained length

Set `num_frames=48` on SVD-XT (nearly double its trained 25) and watch what happens. The first ~25 frames look fine — they are in the trained regime. Somewhere around frame 26–30 the clip starts to degrade: motion becomes incoherent, the subject can morph, and the later frames look progressively less like a continuation and more like noise resolving into something unrelated. There is no error message; the model dutifully produces 48 frames, they are just *bad* past 25, because the temporal positional structure has no learned meaning out there. The fix is not "ask for more frames" — it is sliding-window denoising (generate overlapping 25-frame chunks and blend) or, better, a model actually *trained* for longer clips. This single experiment is the most direct way to feel the length wall: the model does not refuse, it silently leaves its competence zone, which is exactly why you must know the trained clip length and respect it.

These three walls — length, motion range, identity drift — are not bugs in SVD or AnimateDiff. They are the inherent limits of *bolting a short, fixed temporal window onto an image model*. The DiT-video frontier (Sora, CogVideoX, HunyuanVideo, Wan) climbs them by going further: a [spacetime-patch DiT](/blog/machine-learning/video-generation/video-diffusion-transformers) that scales to longer sequences, a [true causal 3D-VAE](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression) that compresses time so longer clips fit, and [flow matching](/blog/machine-learning/image-generation/flow-matching-and-rectified-flow) as the objective. SVD and AnimateDiff are where the field proved you could get coherent short clips cheaply; the frontier is where it spent real compute to make them long, high-motion, and identity-stable.

## 11. Measuring these models honestly

Before the case studies, a word on *proof*, because the series rule is that claims get measured. How would you actually compare SVD and AnimateDiff, or compare two configurations of one of them, without fooling yourself?

The standard automatic metric is **FVD** (Fréchet Video Distance): you pass a set of real clips and a set of generated clips through a pretrained video feature extractor (an I3D network trained on Kinetics), fit a Gaussian to each set's features, and compute the Fréchet distance between the two Gaussians — lower is better. It is the video analog of FID. We dissect its noise and its failure modes in [the metrics of video generation](/blog/machine-learning/video-generation/the-metrics-of-video-generation); the one-line caution is that FVD is *extremely* protocol-sensitive — sample size, clip length, and the feature backbone all shift it independent of true quality — so an FVD number without a fixed protocol is not comparable across papers.

The structured alternative is **VBench**, which decomposes quality into named dimensions — subject consistency, background consistency, motion smoothness, *dynamic degree*, aesthetic quality, imaging quality, and more. The dimension that matters most for our two recipes is the interplay between **motion smoothness** and **dynamic degree**, and it exposes a gaming problem you must guard against: a near-*static* clip scores beautifully on motion smoothness and subject consistency (nothing moves, so nothing is inconsistent) while being a terrible *video*. This is exactly the failure mode SVD's motion curation and `motion_bucket_id` were designed to avoid, and it is why you must *always read motion-smoothness next to dynamic-degree* — a model that scores high on smoothness and low on dynamic degree is producing pretty slideshows, not video. A model can "win" on consistency by simply refusing to move; only reading the two together catches it.

The honest measurement protocol, if you wanted to put real numbers on the SVD-vs-AnimateDiff comparison: fix the seed set, fix the number of generated clips ($\geq 2048$ for FVD stability), fix the clip length and resolution across both models (resample if needed), name the FVD backbone, warm up the pipeline before timing, and report seconds-per-clip and peak VRAM on a *named* GPU at a *named* configuration. Report VBench per-dimension, not a single aggregate, and always show dynamic degree alongside the consistency scores. Anything less and you are comparing protocols, not models.

#### Worked example: the temporal-module ablation as the cleanest proof

The single most convincing measurement you can run on either model is the *temporal-module-on-versus-off* ablation, because it isolates exactly the thing this post is about. Take SVD (or any inflated model), generate a fixed set of clips with the temporal layers active, then generate the *same* set — same seeds, same start frames, same prompts — with the temporal layers forced to the identity (zero their output, so each frame denoises independently, the slideshow). Compute FVD against a real-clip reference set for both. The motion-on FVD is dramatically lower; the gap is roughly a **3× swing** on a real I2V model, and crucially it is driven almost entirely by the *motion-smoothness* and *temporal-flicker* failures of the off condition, not by per-frame quality (which is identical, since the spatial weights did not move). That decomposition — same per-frame quality, very different FVD — is the empirical signature that the temporal module is doing the work, and it is the honest way to *prove* a coherence claim rather than assert it. Pair it with VBench: motion-on should win on motion-smoothness and background-consistency while dynamic-degree stays sane, and if dynamic-degree *collapses* while smoothness *rises*, you have caught the model gaming consistency by refusing to move — exactly the failure the dual reading is designed to expose.

## 12. Case studies: the real numbers

A few concrete anchors from the literature and from running these models, stated as accurately as I can and flagged where approximate.

**Stable Video Diffusion (Blattmann et al., 2023).** Inflated from SD-2.1 into a latent video diffusion model; SVD-XT outputs **25 frames at $1024\times576$**. The paper's headline contribution is the *data* story — the three-stage curation pipeline (image pretrain → curated video pretrain → high-quality finetune) — and the ablation showing curation beat raw scale. On an A100 80GB a clip renders in roughly 8–15 seconds depending on step count; on a 4090 with offload, ~15–40 seconds. `decode_chunk_size` is the practical difference between fitting and OOMing on smaller cards. It remains the open image-to-video baseline most pipelines start from.

**AnimateDiff (Guo et al., 2023).** One motion module, trained once on WebVid-10M against frozen SD-1.5, animates *arbitrary* personalized SD-1.5 checkpoints with **no per-checkpoint training**. The motion module is on the order of 15–25% added parameters, and the marginal cost of animating a new checkpoint is zero — which is exactly why it dominated the open ecosystem. Native context is **16 frames**; the per-frame quality ceiling is SD-1.5's, which is good rather than frontier. MotionLoRA adds composable camera-motion control (pan, zoom, tilt, roll) as tiny stackable LoRAs on the motion module. Later versions (v2, v3, and the SDXL-based AnimateDiff) raised quality and resolution, but the plug-in principle is unchanged.

**The curation lesson, generalized.** The single most-cited *idea* from SVD is not a benchmark number — it is the empirical demonstration that for video, **data curation is higher-leverage than architecture**. The specific filters (cut detection, motion filtering, caption alignment, aesthetic scoring) target failure modes that are *temporal and do not exist for images*, which is why image-model intuitions undershoot how much curation video needs. Every open video model since — CogVideoX, HunyuanVideo, Wan — invests heavily in curation pipelines, and they cite this lineage. If you internalize one transferable result from this post, make it this one.

**The composition lesson, generalized.** AnimateDiff proved that *motion can be a separable, reusable, composable component* in the open ecosystem — appearance (checkpoint), motion (module), and camera (MotionLoRA) as three independent plug-ins. The frontier DiT-video models largely *fold this back* into one end-to-end model (you do not plug a motion module into CogVideoX), trading modularity for quality and length. But the conceptual proof — that the seam between appearance and motion is real and exploitable — is permanent, and it shows up again in control adapters and in how people fine-tune the modern models.

## 13. When to reach for this (and when not to)

A decisive guide, because the two recipes are not interchangeable and the frontier models are not always the right answer either. The table compresses the routing rule into the one signal that decides each case.

| Your situation | Reach for | Why |
| --- | --- | --- |
| You have a specific first frame to animate | SVD (I2V) | Frame zero hands the model the hardest part; spend capacity on motion |
| You have a personalized style + a text prompt | AnimateDiff | Only recipe that composes with your checkpoint, at zero marginal cost |
| You need a specific camera move | AnimateDiff + MotionLoRA | Composable pan/zoom/tilt as stackable low-rank updates |
| You need 5+ seconds, large motion, or top quality | DiT-video (CogVideoX/Hunyuan/Wan) | Both recipes here cap at ~2–4s and SD-class per-frame quality |
| You want a frozen 5s clip, not a stream | One full denoising pass | Do not autoregress or window-stitch a fixed short clip |
| Your clips come out static or teleporting | Fix your data, not your net | Audit for static clips and undetected cuts first (the SVD lesson) |

The prose version of the same routing:

**Reach for SVD when you have a first frame.** Image-to-video is the easier problem and SVD is built for it. If your product can produce or accept a still — a generated image, an uploaded photo, a storyboard frame — hand it to SVD and let it spend all its capacity on motion. It will give you stronger, larger, more coherent motion than a text-to-video model at the same budget. This is the right default for "animate this specific picture."

**Reach for AnimateDiff when you must reuse a personalized style and have only a prompt.** If your value is in a fine-tuned aesthetic — a brand style, a character LoRA, a community checkpoint — AnimateDiff is the *only* recipe here that composes with it at zero marginal cost. Train no video model; plug the motion module into your checkpoint and animate. Add MotionLoRA when you need explicit camera control.

**Reach for the DiT-video frontier when you need length, high motion, or top-tier quality.** Both recipes here cap at ~2–4 seconds, moderate motion, and SD-1.5/2.1-class per-frame quality. When you need a 10-second clip, large dynamic motion that stays coherent, or commercial-quality output, you have outgrown these recipes — go to CogVideoX / HunyuanVideo / Wan (the [open frontier](/blog/machine-learning/video-generation/video-diffusion-transformers)) or a hosted model. Do not fight SVD's length wall with sliding windows when a model trained for the length exists.

**Do not push `motion_bucket_id` to 255 to get more motion.** Past the model's coherent displacement range you get ghosting and tearing, not better video. Raise motion through the scene and the frame rate first; cap the bucket where coherence holds.

**Do not set `decode_chunk_size=num_frames` on a memory-constrained card.** That is the configuration that sails through denoising and OOMs at the final decode. Set the chunk to 8 (or 2 on a small card) and pay a second of decode time to never see that crash.

**Do not run an image model per frame and call it video.** It is the slideshow we have been fighting since the [previous post](/blog/machine-learning/video-generation/from-image-diffusion-to-video-diffusion). Without the temporal module there is no coherence; the loss cannot create it. If you are stitching independent frames, you have skipped the entire point of these recipes.

## 14. Key takeaways

1. **Two problems, two recipes.** Image-to-video (you have a first frame) wants SVD; text-to-video in your own style (you have a personalized checkpoint) wants AnimateDiff. Knowing which problem you have almost fully determines the recipe.
2. **SVD conditions on the start frame twice** — a concatenated VAE latent for pixel-precise appearance and a CLIP image embedding for global semantics — plus scalar `fps` and `motion_bucket_id` micro-conditioning. The two image paths divide labor: local pixels and global meaning.
3. **`motion_bucket_id`, `fps`, and `noise_aug_strength` are one control surface** for the same underlying quantity — per-frame displacement — which is exactly what governs coherence. More motion is safer as more frames at lower per-frame displacement than as a higher motion bucket.
4. **The VAE decode, not the denoiser, is the memory wall.** `decode_chunk_size` trades peak VRAM for decode passes nearly linearly; it is the difference between a successful render and an OOM at the final step on a smaller card. Combine with VAE tiling for high resolution.
5. **SVD's real lesson is data curation, not architecture.** A multi-stage pipeline (cut detection, motion filtering, caption alignment, aesthetic scoring) on dirty, temporally-specific video data beat raw scale — the most transferable result in this post, and the one every open video model since has adopted.
6. **AnimateDiff makes motion a reusable plug-in.** Train one motion module against frozen base SD-1.5 and it animates *any* SD-1.5 checkpoint for free, because personalized checkpoints share the base's activation space. The marginal cost of animating a new style is zero.
7. **The transfer holds exactly as far as the activation distribution is preserved.** Small style shifts keep the personalized checkpoint's activations in-distribution for the motion module; large shifts or a different base architecture break it — which is why motion adapters are versioned per base model.
8. **MotionLoRA adds composable camera control** — pan, zoom, tilt as tiny stackable LoRAs on the motion module — completing a three-layer plug-in surface: appearance (checkpoint), motion (module), camera (MotionLoRA).
9. **Both recipes hit three walls — length (~2–4s), motion range, identity drift** — that are inherent to bolting a short fixed temporal window onto an image model. These walls are precisely what the DiT-video frontier was built to climb.
10. **Measure with FVD under a fixed protocol and read VBench's motion-smoothness next to dynamic degree**, so a pretty static slideshow cannot game its way to a high score.

## 15. Further reading

- **Blattmann, Dockhorn, Kulal, et al. — *Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets* (2023).** The three-stage training, the curation pipeline and its ablation, and the start-frame + micro-conditioning interface.
- **Guo, Yang, Rao, et al. — *AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning* (2023).** The plug-in motion module, the frozen-spatial transfer argument, and MotionLoRA for camera motion.
- **Podell, English, Lacey, et al. — *SDXL* (2023).** The source of the size/crop micro-conditioning idea that SVD repurposes for motion.
- **Tran, Wang, Torresani, et al. — *A Closer Look at Spatiotemporal Convolutions* (2018).** Why factorized temporal operations work, background for the temporal layers both recipes use.
- **🤗 `diffusers` documentation** — `StableVideoDiffusionPipeline`, `AnimateDiffPipeline`, `MotionAdapter`, with the real flags (`num_frames`, `decode_chunk_size`, `motion_bucket_id`, `fps`, `noise_aug_strength`) and the MotionLoRA loading API.
- Within this series: [from image diffusion to video diffusion](/blog/machine-learning/video-generation/from-image-diffusion-to-video-diffusion) (the temporal module these recipes use), [video autoencoders and spatiotemporal compression](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression) (the latent these models denoise), [why video generation is hard](/blog/machine-learning/video-generation/why-video-generation-is-hard) (the coherence×motion×length×cost frame), and forward to [conditioning video: text, image, motion, camera](/blog/machine-learning/video-generation/conditioning-video-text-image-motion-camera), [efficient and real-time video generation](/blog/machine-learning/video-generation/efficient-and-real-time-video-generation), and the capstone [building with video generation: the playbook](/blog/machine-learning/video-generation/building-with-video-generation-the-playbook).
- Link out to the image series for the underlying machinery: [latent diffusion and Stable Diffusion](/blog/machine-learning/image-generation/latent-diffusion-and-stable-diffusion) and [personalization: DreamBooth, textual inversion, LoRA](/blog/machine-learning/image-generation/personalization-dreambooth-textual-inversion-lora).
