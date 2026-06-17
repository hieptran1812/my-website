---
title: "Long Video and Autoregressive Rollout: Beating the Length Wall"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Understand why video models cap at a few seconds, derive why autoregressive rollout drifts, and learn the chunked, diffusion-forcing, and streaming techniques that push to minutes."
tags:
  [
    "video-generation",
    "diffusion-models",
    "autoregressive",
    "long-video",
    "diffusion-forcing",
    "video-diffusion",
    "generative-ai",
    "deep-learning",
    "text-to-video",
  ]
category: "machine-learning"
subcategory: "Video Generation"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/long-video-and-autoregressive-rollout-1.png"
---

The first time I tried to make a video model produce a full minute of footage, I did the naive thing: I asked a model that was trained on 5-second clips to render 60 seconds in one shot. It OOMed before the first denoising step finished. So I did the second naive thing — I generated twelve 5-second chunks, each one starting from the last frame of the one before, and stitched them together. The result was mesmerizing in the wrong way. The first chunk was a crisp golden retriever running across a lawn. By chunk four the fur had taken on a slightly waxy sheen. By chunk seven the dog's snout had subtly changed shape, the lawn had shifted from green to a washed-out olive, and the lighting had drifted as if a cloud were permanently rolling in. By the last chunk it was a different animal in a different scene. The model had not crashed. It had done something worse: it had slowly, confidently lied its way off the rails.

That failure is the subject of this entire post, and it is one of the deepest open problems in video generation. There are two walls in your way when you want long video, and they are completely different in character. The **first wall is hard compute**: attention is quadratic in the number of spacetime tokens and the [causal 3D-VAE](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression) holds the whole clip's activations in memory, so you simply cannot fit a long clip into a single forward pass on any real GPU. The **second wall is statistical**: the moment you generate the clip in pieces and condition each piece on your own previous output, errors stop being independent and start to compound. The first wall caps you at a few seconds per shot. The second wall punishes you the instant you try to escape the first by rolling forward.

The tension this post lives inside is the series spine specialized to the time axis: **autoregressive rollout buys you unbounded length but accumulates error; parallel and chunked generation stays stable but is bounded and has seams.** Every technique that matters here — sliding windows with overlap, Diffusion Forcing, CausVid and self-forcing, hierarchical keyframe-then-interpolate — is a different bet on how to get the length of the first while paying as little of the drift of the second as possible. Figure 1 is the shape of the whole problem: a chain of chunks, each conditioned on the last, length growing without bound on one branch while error compounds on the other.

![Diagram of an autoregressive chunk chain where each chunk conditions on the previous chunk's last frames, one branch showing unbounded length and the other showing error compounding into a drifted output](/imgs/blogs/long-video-and-autoregressive-rollout-1.png)

This is part of the [Video Generation, From First Principles to the Frontier](/blog/machine-learning/video-generation/why-video-generation-is-hard) series, and it sits downstream of the architecture posts. I assume you already know how a video model [adds the time axis on top of image diffusion](/blog/machine-learning/video-generation/from-image-diffusion-to-video-diffusion) and how [spatiotemporal attention patterns](/blog/machine-learning/video-generation/spatiotemporal-attention-patterns) trade coherence for cost. By the end of this post you will be able to derive why a clip caps at a few seconds, formalize error accumulation in rollout and connect it to the exposure-bias problem you may have met in autoregressive language and [autoregressive image models](/blog/machine-learning/image-generation/autoregressive-image-models), implement a sliding-window rollout loop in PyTorch, chain an image-to-video model with 🤗 `diffusers`, and reason honestly about where current open models still fall apart past tens of seconds.

## 1. The length wall: why a clip caps at a few seconds

Let me anchor everything to one concrete target, the same clip I will return to for every number in this post: **one minute of 720p video at 24 fps** of a golden retriever running through a park. That is `$60 \times 24 = 1{,}440$` frames at `$720 \times 1280$`. The question is why you cannot just ask a model to generate all 1,440 frames in a single pass, and the answer has two independent parts that both bite.

### The compute part: quadratic attention in frames

A modern video VAE compresses by `$4 \times 8 \times 8$` (4× in time, 8× in each spatial dimension), so our minute-long clip becomes a latent of shape:

$$
T = \frac{1440}{4} = 360, \quad H = \frac{720}{8} = 90, \quad W = \frac{1280}{8} = 160.
$$

After patchifying with a `$2 \times 2$` spatial patch and temporal patch 1, the DiT attends over:

$$
N = T \cdot \frac{H}{2} \cdot \frac{W}{2} = 360 \cdot 45 \cdot 80 = 1{,}296{,}000 \text{ tokens.}
$$

Recall from the [attention-patterns post](/blog/machine-learning/video-generation/spatiotemporal-attention-patterns) that full 3D attention costs `$4 N^2 d$` FLOPs per layer and, naively implemented, materializes an `$N \times N$` score matrix. For a 5-second clip that `$N$` was 108,000; for our minute it is 1.296 million, a `$12\times$` longer sequence. Attention is quadratic, so the cost is `$12^2 = 144\times$` higher per layer than the 5-second clip. The score matrix alone, even in bf16 at 2 bytes, would be:

$$
N^2 \cdot 2 = (1.296 \times 10^6)^2 \cdot 2 \approx 3.4 \times 10^{12} \text{ bytes} \approx 3.4 \text{ TB.}
$$

That is per head, per layer, before activations, before the model weights. FlashAttention removes the materialized score matrix and brings attention memory down to `$O(N)$`, but it does not remove the FLOPs, which still scale as `$N^2$`. Factorized attention helps enormously on the spatial axis, but the temporal pass is now over `$T = 360$` frames, and the activations you must hold to backprop (or even just to run a long sampler) scale with `$N$`. Length grows linearly in time but the dominant costs grow super-linearly, so there is a clip length past which a single forward pass does not fit and does not finish.

### The memory part: the VAE was never trained that long

The second part of the length wall is subtler and it is the one people forget. Even if you could afford the attention, **the VAE itself has a trained clip length.** A causal 3D-VAE is trained on clips of some fixed temporal extent — say 49 or 121 frames — and its temporal convolutions and normalization statistics are tuned for that regime. Push a 1,440-frame clip through it and two things go wrong. First, the decoder must hold the entire spatiotemporal activation stack in VRAM to decode — and VAE decode, not the denoiser, is very often the actual VRAM wall, because the decoder upsamples back to full `$720 \times 1280 \times 1440$` pixel resolution and those activation tensors are enormous. Second, the temporal receptive field and the learned dynamics simply were not trained at that length, so even if it fit, the reconstruction quality degrades outside the trained window.

So the length wall is really two walls stacked: a quadratic compute wall in the denoiser and a fixed-trained-length plus decode-memory wall in the VAE. Both say the same thing: **you cannot make a long video in one shot.** You must generate it in pieces. And the instant you generate it in pieces, you inherit the second, statistical wall — which is where the rest of this post lives.

### Quantifying the VAE-decode memory wall

It is worth doing the decode-memory arithmetic explicitly, because almost everyone underestimates it. When the VAE decoder upsamples a latent back to pixels, it does so through a stack of convolutional blocks at progressively higher resolution, and the activations at the *highest* resolution dominate. For a single frame at `$720 \times 1280 \times 3$` channels in the final layer that is about 2.8 million values, but the decoder's penultimate blocks run at full spatial resolution with many channels — say 128 channels at `$720 \times 1280$`, which is `$720 \cdot 1280 \cdot 128 \approx 1.18 \times 10^8$` values, or 236 MB per frame in fp16 for *one* activation tensor, and the decoder holds several at once. Multiply by the temporal extent of the chunk you decode at once. For a 16-latent-frame chunk (64 pixel frames at 4× temporal compression) that is on the order of `$64 \cdot 236 \text{ MB} \cdot (\text{a few tensors}) \approx 45\text{–}90$` GB of transient activation just for the high-resolution decoder stages — which is why VAE tiling and chunked decode are not optional niceties but the only way the decode fits at all.

Now extend that to the full minute decoded at once: 1,440 pixel frames, `$\approx 22.5\times$` the 64-frame chunk, lands you in the multi-hundred-GB regime for transient decode activation. No single GPU has that. This is the precise mechanism behind "OOMed at second 6": the denoiser produced enough latents, the decode started, and the activation stack for the high-resolution upsampling blocks blew past VRAM partway through. The fix is to decode in temporal chunks — and decoding in temporal chunks *requires* the causal VAE of Section 8, because you must be able to decode chunk `$k$` without the future latents of chunk `$k+1$`. So the decode-memory wall and the streaming-VAE solution are two sides of the same coin, and we will close that loop later.

#### Worked example: where the minute-long clip dies

Take an H100 80GB. A well-tuned open model like Wan 2.1 or HunyuanVideo generates a 5-second 720p clip (≈108k tokens) in roughly 2–4 minutes with full sampling, peaking around 40–60 GB of VRAM with model CPU offload and VAE tiling on. Scale to the full minute in one pass: the attention FLOPs go up `$\approx 144\times$`, so a 3-minute render becomes a `$\approx 7$`-hour render even before memory; and the VAE decode of 1,440 frames at full resolution wants hundreds of GB of activation memory it does not have. In practice you OOM in the VAE decode long before the denoiser finishes — which is exactly the "OOMed at second 6" pattern. The honest conclusion: the single-shot ceiling on an 80 GB card for 720p is roughly **5–10 seconds**, and everything beyond that is a stitching or rollout problem.

## 2. Chunked and sliding-window generation: stable but seamed

The most conservative way past the wall is to generate **overlapping windows in parallel and stitch them.** You pick a window the model is comfortable with — say 16 latent frames — slide it forward with a stride smaller than the window, generate each window, and blend the overlapping regions so there is no visible seam. Crucially, in the pure chunked scheme each window is generated *independently* (or all conditioned on the same global text prompt), so there is no feedback loop and therefore no error accumulation. That is the whole appeal: it is statistically stable. The price is that two independently generated windows do not agree perfectly in their overlap, so you get a seam unless you blend, and even with blending you only get *local* coherence — the model has no mechanism to keep the dog the same dog across windows that never saw each other.

The blending itself is the easy part. In the overlap band you cross-fade: for a frame at relative position `$\tau \in [0, 1]$` within an overlap of length `$L$`, the stitched frame is `$x_\text{stitch} = (1-\tau) \, x_\text{prev} + \tau \, x_\text{next}$`, a linear ramp from the previous window's prediction to the next window's prediction. Figure 3 shows the scheme: three windows, two overlap bands, a linear blend in each.

![Stacked diagram of three overlapping generation windows with shared overlap bands between consecutive windows that are linearly blended into a single seam-free stitched clip](/imgs/blogs/long-video-and-autoregressive-rollout-3.png)

### The overlap consistency condition

Why does overlap help at all? Formalize it. Let window `$A$` predict the latent frames `$\{a_t\}$` and window `$B$` predict `$\{b_t\}$`, and suppose they share an overlap of `$L$` frames. At the seam frame, window `$A$` ends at value `$a_L$` and window `$B$` begins at value `$b_1$`. If you hard-cut from `$A$` to `$B$`, the visible discontinuity is the jump `$\|b_1 - a_L\|$`. A linear blend over the overlap makes the transition continuous, but it does not make it *consistent* — if `$a$` and `$b$` disagree by a constant offset across the whole band, the blend produces a ghosting cross-fade rather than a clean motion. The condition you actually want is that the two windows agree on the *content* in the overlap, not just at the endpoints:

$$
\frac{1}{L}\sum_{t=1}^{L} \| a_t - b_t \| < \epsilon.
$$

This is satisfied only if both windows are conditioned on enough shared information — the same text prompt, and ideally the same starting frame or latent for the overlap — that they converge on the same content. The deepest version of this is to **condition window `$B$` on the actual generated frames of window `$A$` in the overlap**, which is exactly the step that turns a stable chunked scheme into an autoregressive one. So chunked generation with shared conditioning is the stable, bounded extreme, and the moment you make the overlap a *causal dependency* — window `$B$` must reproduce window `$A$`'s last frames — you have crossed into rollout and inherited its drift. That continuum, from "independent windows blended" to "each window conditioned on the previous," is the central design space of this post.

### Blend in latent space, not pixel space

A practical detail that bites people: *where* you blend matters. The cheap thing is to decode both windows to pixels and cross-fade the pixel frames in the overlap. The problem is that a linear cross-fade in pixel space of two slightly-different frames produces a literal **double exposure** — you see both versions ghosted together, because pixels are not a space where averaging two plausible images gives a plausible image (the average of a dog one inch left and one inch right is a transparent two-headed dog). Blending in *latent* space, before the VAE decode, is meaningfully better: the VAE's decoder is trained to map latents to plausible images, so a blended latent decodes to something closer to a single coherent frame than to a ghost. It is still not perfect — the latent manifold is not perfectly linear either — but it is the difference between a soft transition and an obvious artifact. This is why the sliding-window code blends `full[:, :, -overlap:]` while everything is still latent, and only decodes at the very end. The general rule for any stitching in this whole space: **operate on latents as long as you can, decode once at the end**, both for blend quality and to avoid re-running the expensive VAE decode per window.

The deeper fix, when even latent blending shows a seam, is to not blend at all but to make window `$B$` *generate* the overlap frames conditioned on window `$A$`'s latents — the inpainting-forward trick from the code in Section 2. Then there is nothing to blend because the overlap is, by construction, a continuation the model produced. That eliminates the seam entirely at the cost of the causal dependency (and thus the drift). Blending is the stable-but-seamed choice; conditioned generation is the seamless-but-drifting choice; this is the same fundamental trade as the whole post, now visible at the granularity of a single overlap band.

### A sliding-window rollout loop in PyTorch

Here is a concrete sliding-window loop over a video diffusion model. It generates a window, takes the last `$k$` frames as the conditioning for the next window, generates the next window so that it *reproduces* those `$k$` frames and then continues, and blends the overlap. This is the autoregressive end of the continuum — each window depends causally on the last — written against a generic latent-space video pipeline.

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def sliding_window_rollout(
    pipe,                 # a latent video diffusion pipeline
    prompt: str,
    total_frames: int = 240,   # ~10s at 24fps in pixel terms
    window: int = 16,          # latent frames the model is trained on
    cond_frames: int = 4,      # how many trailing frames condition the next window
    overlap: int = 4,          # frames blended at the seam
    num_inference_steps: int = 40,
    guidance_scale: float = 6.0,
    generator=None,
):
    stride = window - overlap                 # how far we advance each step
    device = pipe._execution_device

    # 1) Seed window: a normal text-to-video generation.
    out = pipe(
        prompt=prompt,
        num_frames=window,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        output_type="latent",
        generator=generator,
    )
    latents = out.frames  # shape [1, C, window, H, W] (latent frames)
    full = latents.clone()

    # 2) Roll forward until we have enough latent frames.
    produced = window
    while produced < total_frames:
        # Last cond_frames latents become the conditioning anchor.
        anchor = full[:, :, -cond_frames:]            # [1, C, k, H, W]

        # Generate the next window, forcing it to start from the anchor.
        # The pipeline init-noises only the *new* frames and keeps the
        # anchor latents clean (noise level 0) -- this is the key trick.
        next_latents = generate_conditioned_window(
            pipe, prompt, anchor=anchor,
            window=window, cond_frames=cond_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale, generator=generator,
        )  # [1, C, window, H, W], first cond_frames == anchor

        # 3) Blend the overlap band linearly.
        new_part = next_latents[:, :, cond_frames:]   # genuinely new latents
        if overlap > 0:
            tail = full[:, :, -overlap:]
            head = next_latents[:, :, cond_frames - overlap: cond_frames]
            w = torch.linspace(0, 1, overlap, device=device).view(1, 1, -1, 1, 1)
            blended = (1 - w) * tail + w * head
            full[:, :, -overlap:] = blended

        full = torch.cat([full, new_part], dim=2)
        produced = full.shape[2]

    full = full[:, :, :total_frames]
    # 4) Decode all latents to pixels with VAE tiling to fit memory.
    pipe.vae.enable_tiling()
    video = pipe.decode_latents(full)
    return video
```

The piece that actually matters is `generate_conditioned_window`: it must noise only the *new* latents and keep the anchor latents at noise level zero throughout sampling, so the diffusion process is "inpainting forward in time." Here is a concrete implementation of that inner loop, written as a temporal inpainting over the scheduler's denoising trajectory — at every step it overwrites the anchor positions with the (re-noised) clean anchor so the sampler never gets to move them, exactly the way image inpainting clamps the known region.

```python
import torch

@torch.no_grad()
def generate_conditioned_window(
    pipe, prompt, anchor, window, cond_frames,
    num_inference_steps=40, guidance_scale=6.0, generator=None,
):
    # anchor: [1, C, cond_frames, H, W] clean latents to be reproduced.
    device = pipe._execution_device
    B, C, _, H, W = anchor.shape
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_inference_steps, device=device)

    # Start the whole window from noise...
    latents = torch.randn(B, C, window, H, W, device=device,
                          dtype=anchor.dtype, generator=generator)
    latents *= scheduler.init_noise_sigma

    text_emb = pipe.encode_prompt(prompt, device, 1, guidance_scale > 1.0)

    for t in scheduler.timesteps:
        # ...but at EVERY step, clamp the first cond_frames positions to the
        # anchor re-noised to the current level. The sampler denoises only
        # the new frames; the anchor is fixed context it must continue from.
        noised_anchor = scheduler.add_noise(
            anchor, torch.randn_like(anchor), t
        )
        latents[:, :, :cond_frames] = noised_anchor

        model_in = scheduler.scale_model_input(latents, t)
        noise_pred = pipe.unet(model_in, t, encoder_hidden_states=text_emb).sample
        if guidance_scale > 1.0:
            uncond, cond = noise_pred.chunk(2)
            noise_pred = uncond + guidance_scale * (cond - uncond)

        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # Final clamp so the returned anchor frames are exactly the clean input.
    latents[:, :, :cond_frames] = anchor
    return latents
```

The clamp `latents[:, :, :cond_frames] = noised_anchor` at every step is the entire mechanism. Without it, the sampler would happily redraw the anchor frames from the prompt and you would get a hard cut at the seam. With it, the new frames are generated *conditioned on* a fixed, continuing context, which is what makes the window flow out of the previous one. Notice the structural cost, though: every window after the first conditions on the model's own output. That single fact is the entire source of drift, and Section 4 makes it rigorous.

## 3. Autoregressive rollout: unbounded length, accumulating error

The sliding-window loop above, taken to its logical end with `overlap = cond_frames` and no independent windows, *is* autoregressive rollout: generate chunk `$1$`, condition chunk `$2$` on the last few frames of chunk `$1$`, condition chunk `$3$` on the last few frames of chunk `$2$`, and so on, forever. This is the only approach in the whole space that gives you **genuinely unbounded length** — you can keep rolling as long as you have GPU-seconds, because each step is a fixed-size window and the cost is constant per chunk. A 60-second clip costs `$12\times$` a 5-second clip; a 10-minute clip costs `$120\times$`; there is no quadratic blow-up, because you never hold more than one window in attention at a time.

The catastrophe is that each chunk conditions on its own imperfect output. The model was never asked to be perfect — it produces a sample from a learned distribution, and that sample has error relative to the "true" continuation. When that slightly-wrong chunk becomes the conditioning for the next, the next chunk is generated from a starting point that is already off-distribution, and it adds its own error on top. The errors do not cancel; they compound. This is why my golden retriever turned into a different animal: not one catastrophic mistake but a thousand small ones, each one inherited and amplified by the next chunk. Figure 2 contrasts the two regimes directly — a single in-window clip that stays sharp and on-identity against a long rollout that loses color, sharpness, and identity.

![Before-and-after comparison of a stable 5-second single-shot clip that holds identity and sharpness against a 60-second rollout whose identity, sharpness, and color drift](/imgs/blogs/long-video-and-autoregressive-rollout-2.png)

### Why "unbounded but drifting" is the fundamental trade

Hold the contrast in your head, because it is the axis everything else navigates. Chunked-parallel generation is **bounded but stable**: each window is independent so errors do not compound, but you cannot extend past the point where windows stop sharing enough context to stay coherent, and you fight seams. Autoregressive rollout is **unbounded but unstable**: you can extend forever at constant per-chunk cost, but error compounds and quality decays. There is no free lunch here — these are genuinely different points on a frontier, and the clever methods in Sections 5 and 6 do not eliminate the trade, they bend the decay curve so that the unbounded regime stays usable for longer. Before we can appreciate how they bend it, we need to know exactly *how fast* error grows, which is the science block.

### Stress test: when motion between frames is large

The drift bound assumes the per-step fresh error `$\delta$` is small, but `$\delta$` is not a constant — it depends on how hard the next chunk is to predict from the conditioning. The single worst case for rollout is **large motion between the conditioning frames and the new frames.** When the subject moves fast — a runner sprinting, a fast camera pan, a scene cut — the conditioning frames carry little information about where things will be a fraction of a second later, so the model's fresh error `$\delta$` spikes, and a spiked `$\delta$` early in the rollout seeds a larger accumulated error for the rest of it. This is why rollout demos overwhelmingly feature slow, smooth motion: slow motion keeps `$\delta$` small per step, which keeps the geometric sum manageable. Push the same rollout to fast action and it collapses far sooner than the `$\delta = 0.05$` worked example suggests, because the effective `$\delta$` might be `$0.15$` on the high-motion chunks.

There is a second failure mode hiding here: **a hard anchor fights large motion.** If you anchor strongly on the seed frame to resist identity drift (Section 9), you are simultaneously telling the model to stay close to the seed's *pose and position*, which directly opposes large motion. So on high-motion content you face a genuine dilemma — anchor hard and the subject cannot move enough (stutter, rubber-banding back toward the seed), or anchor loosely and identity drifts. The frontier systems resolve this with a *learned* memory that anchors identity (appearance) without anchoring pose, which is exactly what a fixed seed-frame latent cannot do. This is one of the clearest places where open models, limited to crude seed-frame anchoring, visibly fall short of the proprietary frontier on dynamic content.

## 4. The science: formalizing error accumulation

Let me make "errors compound" precise, because the precise version tells you exactly which knobs help. This is the same exposure-bias story that haunts autoregressive sequence models, transplanted to continuous video latents.

### The setup

Let `$x_t$` denote the true latent at chunk-step `$t$` of an ideal continuation, and let `$\hat{x}_t$` be the model's generated latent at that step. The model is a conditional generator `$\hat{x}_{t} = G(\hat{x}_{t-1})$` (it predicts the next chunk from the previous chunk's trailing frames). Define the per-step error as the deviation of the generated latent from the true latent:

$$
e_t = \hat{x}_t - x_t.
$$

At training time we use **teacher forcing**: the model always sees the *true* previous chunk `$x_{t-1}$` as conditioning, and we train it to produce `$x_t$`. So its one-step error when conditioned on the truth is some small residual `$\delta_t$` with `$\mathbb{E}\|\delta_t\| \le \delta$` — the model is good, one step from clean input it makes a small mistake. At inference time, though, the model conditions on its own previous output `$\hat{x}_{t-1} = x_{t-1} + e_{t-1}$`, which is *not* clean. That train-test mismatch is the **exposure bias** — the model is never exposed to its own errors during training, so it has no idea how to recover from them at inference. Figure 6 lays the mismatch out: teacher forcing conditions on ground truth; rollout conditions on the model's own drifting output.

![Before-and-after comparison of teacher-forced training conditioning on clean ground-truth frames versus inference rollout conditioning on the model's own compounding errors, illustrating exposure bias](/imgs/blogs/long-video-and-autoregressive-rollout-6.png)

### The compounding bound

Assume the generator is locally Lipschitz in its conditioning with constant `$L$`: a perturbation `$e_{t-1}$` in the input produces at most `$L \|e_{t-1}\|$` perturbation in the output, on top of the model's own fresh one-step error `$\delta_t$`. Then:

$$
\|e_t\| = \|\hat{x}_t - x_t\| \le \underbrace{L \|e_{t-1}\|}_{\text{inherited and amplified}} + \underbrace{\|\delta_t\|}_{\text{fresh error}}.
$$

Unrolling this recurrence from `$e_0 = 0$` (the seed chunk is given) gives a geometric series:

$$
\|e_t\| \le \delta \sum_{i=0}^{t-1} L^i = \delta \cdot \frac{L^t - 1}{L - 1} \quad (L \ne 1).
$$

This single inequality is the whole story of long-video drift, and it tells you three concrete things:

1. **If `$L > 1$` (the model amplifies perturbations), error grows *exponentially* in the number of chunks.** This is the regime of catastrophic collapse — the dog becomes a different animal within a dozen chunks. The base `$L$` is set by how sensitive the model is to off-distribution conditioning, which exposure bias makes large because the model never learned to damp its own mistakes.
2. **If `$L = 1$` (marginal stability), error grows *linearly*: `$\|e_t\| \le \delta t$`.** Slow, graceful degradation. This is the best you can hope for from a naive rollout and roughly what a well-behaved model shows before it falls off the cliff.
3. **If `$L < 1$` (the model is contractive — it pulls back toward the data manifold), error stays *bounded*: `$\|e_t\| \le \delta / (1 - L)$`.** This is the holy grail. A rollout that contracts can run forever without collapsing. Every stabilization technique in this post is, in effect, an attempt to push the effective `$L$` below 1 — to make the model self-correcting rather than self-amplifying.

That is the lever. The reason Diffusion Forcing, history conditioning, and self-forcing work is not magic; they all reduce the effective Lipschitz amplification of conditioning error, moving you from the exponential regime toward the contractive one. Now we can read each technique as a move in this single coordinate.

#### Worked example: how fast the cliff arrives

Put numbers on the recurrence so the cliff is not abstract. Suppose the model's fresh per-step error is `$\delta = 0.05$` (5% relative deviation per chunk — a good model) and you want to know when the accumulated error crosses a visible-collapse threshold of, say, `$\|e_t\| = 1.0$` (100% — the subject is unrecognizable). In the **marginally stable** case `$L = 1$`, error grows linearly: `$\|e_t\| = 0.05 \, t$`, so you cross the threshold at `$t = 20$` chunks. At a 16-latent-frame window (≈2.7s of pixels), 20 chunks is about 54 seconds — which is roughly where well-behaved open models do visibly fall apart, matching the Figure 7 curve. Now make the model slightly *amplifying*, `$L = 1.1$`. The geometric sum `$0.05 \cdot \frac{1.1^t - 1}{0.1}$` crosses 1.0 at `$t \approx 12$` chunks (≈32s) — the 10% amplification chops nearly a third off your usable length. Push to `$L = 1.3$` and you collapse by `$t \approx 7$` (≈19s). The lesson is brutal and quantitative: **a small change in how much the model amplifies its own error moves the collapse point dramatically,** which is exactly why the stabilization methods that nudge `$L$` from 1.1 down toward 0.95 are worth so much. Get `$L < 1$` and the sum converges — at `$L = 0.95$`, `$\|e_t\| \to 0.05 / 0.05 = 1.0$` only in the limit, and stays *bounded below collapse* forever.

### A variance reading of the same bound

There is a complementary, probabilistic way to see this that some find more intuitive. Treat each chunk's fresh error `$\delta_t$` as a zero-mean random perturbation with variance `$\sigma_\delta^2$`, independent across steps, and the conditioning amplification as a scalar gain `$L$`. The accumulated error variance after `$t$` steps is then `$\text{Var}(e_t) = \sigma_\delta^2 \sum_{i=0}^{t-1} L^{2i} = \sigma_\delta^2 \frac{L^{2t} - 1}{L^2 - 1}$`. For `$L < 1$` this converges to `$\sigma_\delta^2 / (1 - L^2)$` — a stationary noise floor the rollout fluctuates around without diverging. For `$L \ge 1$` the variance grows without bound. The point is the same as the deterministic bound, but it makes clear *why a contractive rollout looks like stable jitter rather than perfect stillness*: there is always a fresh-error injection `$\sigma_\delta$` each step, and contraction does not remove it, it just stops it from accumulating. A method that achieves perfect stillness has not reached `$L<1$`; it has set `$\sigma_\delta = 0$` by killing motion, which is the gaming failure mode Section 12 warns about.

### The exposure-bias parallel

If you have read the [autoregressive image models](/blog/machine-learning/image-generation/autoregressive-image-models) post or worked with autoregressive language models, this is the same disease in a continuous costume. There, the model predicts the next token conditioned on previous tokens; at training it sees ground-truth prefixes (teacher forcing), at inference it sees its own sampled prefixes, and a single bad token can derail the rest of the sequence. The classic fixes — scheduled sampling, where you sometimes feed the model its own predictions during training so it learns to recover — have a direct video analog: train the model on its own rollouts so it sees off-distribution conditioning and learns to contract. The continuous-latent twist is that video has a second axis the noise level can exploit, which is exactly what Diffusion Forcing does. But the core pathology is identical: **the gap between the clean inputs of training and the dirty inputs of inference.** Name it once and every technique becomes legible.

## 5. Diffusion Forcing: per-token noise levels that stabilize rollout

Diffusion Forcing (Chen et al., 2024) is the cleanest idea in this space, and once you see it you cannot unsee it. The standard diffusion training objective applies a *single* noise level to the whole sample — every frame in the clip gets the same `$t$` in the schedule. Diffusion Forcing breaks that assumption: **it gives every frame (every token, in the general formulation) its own independent noise level.** During training, frame `$i$` might be at noise level `$\sigma_i$` while frame `$j$` is at a completely different `$\sigma_j$`. The model learns to denoise a clip where different frames are corrupted to different degrees.

That sounds like a small change to the noise schedule. It is actually a profound change to what the model can do at inference, because it unifies two operations that were previously separate. A frame at noise level 0 is a *clean conditioning frame*. A frame at the maximum noise level is a *frame to be generated from scratch*. A frame in between is *partially known*. Once the model can denoise arbitrary per-frame noise configurations, you can set up the exact configuration you need for stable rollout: hold the history frames at noise level 0 (clean anchors), set near-future frames to low noise, far-future frames to high noise, and denoise the whole window jointly. Figure 5 shows this configuration.

![Graph of Diffusion Forcing where clean history frames at zero noise, low-noise near-future frames, and high-noise far-future frames are jointly denoised into an anchored output that rolls the window forward](/imgs/blogs/long-video-and-autoregressive-rollout-5.png)

### Why per-frame noise reduces the effective Lipschitz constant

Here is the science of *why* this stabilizes rollout, in the language of Section 4. In a naive rollout, the conditioning frames are passed in as fixed, clean inputs, and the model has no learned mechanism to treat them as possibly-erroneous — so a small error in the conditioning propagates with amplification `$L$`. Under Diffusion Forcing, the conditioning frames are presented at a noise level the model has seen during training across the *entire* range, including configurations where the "clean" anchor is itself slightly noisy. Two things follow.

First, because the model was trained to denoise frames whose neighbors carry arbitrary noise, it has learned to **reconcile** a partially-noisy context rather than blindly trust it. The denoising dynamics pull the whole window toward the data manifold jointly, which is exactly a contractive operation — it reduces the effective `$L$` toward the `$L < 1$` bounded-error regime. Second, you gain a control knob: by injecting a small amount of noise into the conditioning frames at rollout time and letting the model denoise it away, you actively *project the accumulated error back onto the manifold* at every step, rather than letting it ride forward. The Diffusion Forcing paper frames this as a "stabilizing" sampling scheme, and in our coordinates it is a direct intervention to make the rollout contractive.

### A sketch of the per-frame-noise training loss

The training change is small and worth seeing concretely. Standard video diffusion samples one timestep `$t$` per clip; Diffusion Forcing samples one per frame.

```python
import torch

def diffusion_forcing_loss(model, x0, noise_scheduler):
    # x0: clean latent clip, shape [B, C, T, H, W]
    B, C, T, H, W = x0.shape
    device = x0.device

    # Standard diffusion would draw ONE timestep per sample:
    #   t = torch.randint(0, num_train_timesteps, (B,))
    # Diffusion Forcing draws an INDEPENDENT timestep PER FRAME:
    t = torch.randint(
        0, noise_scheduler.config.num_train_timesteps,
        (B, T), device=device,
    )  # [B, T] -- one noise level per frame

    noise = torch.randn_like(x0)
    # Broadcast per-frame timesteps over C, H, W when adding noise.
    t_b = t.view(B, 1, T, 1, 1).expand(B, C, T, H, W)
    noisy = noise_scheduler.add_noise(x0, noise, t_b)

    # The model is conditioned on the per-frame noise levels so it knows
    # which frames are clean anchors and which are to be generated.
    pred = model(noisy, timesteps=t)            # predicts noise (or velocity)
    return torch.nn.functional.mse_loss(pred, noise)
```

At inference you choose the per-frame noise vector to encode your rollout: zeros for the clean history you want to keep, a ramp from low to high for the new frames you want to generate. Slide the window forward, set the newly-generated frames to "clean anchor" for the next step, and repeat. Because the model has seen every noise configuration in training, this rollout is in-distribution in a way naive teacher-forced rollout never is — which is the whole point.

### The stabilizing sampling schedule

The per-frame-noise capability gives you a control surface, and how you *use* it decides whether the rollout contracts. The naive use — clean history at noise 0, new frames at max noise, denoise once — already beats teacher-forced rollout, but the Diffusion Forcing line goes further with a **monotone noise schedule across the window**: arrange the new frames so that the noise level increases smoothly from the history edge outward, so the frame just past the clean anchor is at low noise (nearly determined by history) and the farthest future frame is at high noise (maximally free). Denoising this configuration lets information propagate gradually from the certain past into the uncertain future, frame by frame, rather than forcing the model to hallucinate the whole window at once. As you slide the window forward, the frame that was at low noise becomes the new clean anchor, and a fresh high-noise frame enters at the far end. This rolling, graded schedule is what keeps each new frame *close* to its conditioning — small per-step error `$\delta$` — and is the concrete mechanism behind the contractive `$L < 1$` behavior. A useful way to see it: the schedule turns one hard "generate 16 new frames from nothing" problem into 16 easy "generate one frame from a near-certain neighbor" problems, and easy problems have small fresh error.

A second knob is **how much noise to leave on the conditioning frames.** Setting the history to *exactly* noise 0 makes the model trust it completely, which propagates any accumulated error verbatim. Leaving a small residual noise on the conditioning — say a few steps from clean — tells the denoiser "this context is approximately right, clean it up," which actively projects accumulated error back toward the manifold each step. That residual is the dial between "trust the past" (low residual, faster drift) and "re-anchor the past" (higher residual, more correction but slower scene change). Tuning it is the practical art of stable rollout, and it is only *possible* because the model was trained on every noise level.

## 6. CausVid and self-forcing: causal, distilled, frame-by-frame streaming

Diffusion Forcing stabilizes rollout but still runs a full multi-step diffusion sampler per window. The next move — and the one that connects long video to the [efficient and real-time](/blog/machine-learning/video-generation/efficient-and-real-time-video-generation) frontier — is to make the model **causal and distilled** so it can emit frames in a continuous stream, ideally one (or a few) at a time, in near real time.

CausVid (causal video generation via distillation) and the broader **self-forcing** line attack two problems at once. The causal part replaces bidirectional spatiotemporal attention with **causal attention along the time axis**: each frame attends only to past frames, never future ones. This is what makes streaming possible — you do not need the whole clip to compute frame `$t$`, only the frames before it, so you can produce frames as fast as you can run the forward pass, exactly like an autoregressive language model emitting tokens. The distillation part collapses the many-step diffusion sampler into a few steps (or one) per frame, using distribution-matching distillation from a strong bidirectional teacher, so each frame is cheap.

The "self-forcing" insight is the deepest part and ties straight back to Section 4. Recall the exposure-bias diagnosis: the model drifts because at training it conditions on clean ground truth but at inference it conditions on its own output. **Self-forcing closes that gap by training the model on its own autoregressive rollouts.** During training, instead of always feeding ground-truth history, the model generates a rollout, sees its own (imperfect) generated frames as conditioning, and is trained to keep the rollout on-distribution. In our coordinates from Section 4, this directly attacks the Lipschitz constant: by exposing the model to its own errors during training and penalizing divergence, you teach it to contract — to pull a drifting rollout back toward the data manifold. It is scheduled sampling for video, done properly.

### Why causal attention enables a KV cache for frames

The mechanism that makes streaming cheap is worth spelling out, because it is the exact analog of how a language model streams tokens. In **bidirectional** spatiotemporal attention, frame `$t$`'s representation depends on *all* frames, future included, so adding a new frame at the end forces you to recompute attention for every earlier frame — there is no reuse. In **causal** temporal attention, frame `$t$` attends only to frames `$\le t$`, which means the keys and values of past frames are *fixed once computed*. You cache them. When you generate the next frame you compute its query, attend against the cached past keys/values, and append its own key/value to the cache — `$O(t)$` work for the new frame instead of `$O(t^2)$` to recompute the whole clip. This is a literal KV cache over the time axis:

```python
class CausalTemporalAttention:
    def __init__(self):
        self.k_cache = None   # [B, heads, t_so_far, d_head]
        self.v_cache = None

    def step(self, x_new):    # x_new: one (or a few) new frames' tokens
        q = self.to_q(x_new)
        k = self.to_k(x_new)
        v = self.to_v(x_new)
        # Append the new frame's keys/values to the running cache.
        self.k_cache = k if self.k_cache is None else torch.cat([self.k_cache, k], dim=2)
        self.v_cache = v if self.v_cache is None else torch.cat([self.v_cache, v], dim=2)
        # New frame attends over all cached past + itself -- O(t), not O(t^2).
        attn = torch.softmax(q @ self.k_cache.transpose(-1, -2) * self.scale, dim=-1)
        return self.to_out(attn @ self.v_cache)
```

This is precisely what lets CausVid-style models emit frames at a steady rate indefinitely: per-frame cost is constant (plus a slowly growing cache you can window or compress), not quadratic in the clip length so far. Bidirectional diffusion has no such cache, which is the deep reason it cannot stream and must regenerate whole windows. Causal attention is the price of admission to real-time, and self-forcing is what keeps the resulting cheap rollout from drifting.

### A streaming I2V chaining sketch with 🤗 diffusers

The simplest practical version of streaming rollout you can run today does not require a special causal model at all — it chains an off-the-shelf image-to-video model by feeding the **last frame of each clip as the first frame of the next.** This is the crudest autoregressive rollout (the conditioning is a single frame), it drifts the fastest, but it is the clearest illustration of the loop and it runs on real hardware right now.

```python
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float16, variant="fp16",
)
pipe.enable_model_cpu_offload()          # fit on a 24GB card
pipe.vae.enable_slicing()                # cut VAE-decode VRAM

def chain_i2v(first_image, num_clips=8, frames_per_clip=14, fps=7, seed=0):
    image = load_image(first_image)
    all_frames = []
    gen = torch.manual_seed(seed)

    for k in range(num_clips):
        out = pipe(
            image,
            num_frames=frames_per_clip,
            decode_chunk_size=4,         # decode in chunks to bound VRAM
            motion_bucket_id=110,        # how much motion to inject
            fps=fps,
            generator=gen,
        )
        clip = out.frames[0]             # list of PIL frames
        # Drop the duplicated first frame on all clips after the seed,
        # so the seam is not a stutter.
        all_frames.extend(clip if k == 0 else clip[1:])
        # The LAST frame of this clip seeds the NEXT clip -- the AR step.
        image = clip[-1]

    export_to_video(all_frames, "rolled_out.mp4", fps=fps)
    return all_frames
```

Run this and you will watch the drift from Section 4 happen in real time: the first clip is crisp, and by the sixth the colors have washed and the subject has wandered, because a single conditioning frame carries almost no identity anchor and the effective `$L$` is well above 1. It is the honest baseline — every stabilization technique is measured against how much further it pushes this loop before it collapses. To get from here to a usable minute, you need either Diffusion-Forcing-style per-frame noise, richer history conditioning (more than one frame, plus a persistent memory), or a self-forcing-trained causal model — which is exactly the frontier.

## 7. Hierarchical generation: keyframes first, then interpolate

There is a structurally different way to get length that sidesteps the rollout problem entirely: **generate sparse keyframes that span the whole duration first, then fill in the frames between them.** Instead of marching forward chunk by chunk and hoping identity survives, you plan the long-range structure globally — a handful of keyframes at, say, one every two seconds across the full minute — and then run a frame-interpolation or short-clip-conditioned model to densify each gap. This is the same coarse-to-fine idea that hierarchical methods use elsewhere, applied to the time axis.

The advantage is global coherence by construction. The keyframes are generated together (or by a model that sees the whole sparse timeline), so the dog at second 0 and the dog at second 58 are conditioned on the same global plan — identity does not have to survive 30 sequential rollout steps, it only has to be consistent across keyframes generated in one shot. The interpolation between keyframes is a *bounded* problem: each gap is short, the endpoints are fixed, so error cannot accumulate across the whole duration the way it does in pure rollout. Drift is replaced by interpolation quality, which is a far more tractable failure mode.

The cost is that interpolation between distant keyframes is hard when there is large motion or scene change — the model must invent plausible in-between motion, and if the keyframes are too far apart it either produces mushy morphs or implausible dynamics. So hierarchical generation trades the rollout drift problem for a keyframe-spacing problem: too sparse and the interpolation guesses badly, too dense and you are back to nearly per-frame generation. In practice the strong long-video systems combine ideas — a hierarchical plan for global structure plus a stabilized rollout (Diffusion Forcing or self-forcing) within each segment for local motion. Figure 4 puts all five approaches on one grid so you can read the trade directly.

The same trade-offs laid out as a decision table, with the effective-`$L$` reading from Section 4 and the typical practical horizon:

| Approach | Max length | Effective `$L$` | Drift behavior | Seams | Best for |
| --- | --- | --- | --- | --- | --- |
| Single-shot | ~5–10s | n/a (no rollout) | none | none | fixed short clips, sharpest output |
| Chunked-parallel | ~20–30s | n/a (independent) | none | blended overlaps | ambient b-roll, parallel wall-clock |
| Single-frame I2V chaining | unbounded | `$\gg 1$` | fast collapse | none | demos only, drifts by ~15s |
| Multi-frame AR rollout | unbounded | `$\approx 1$` | linear decay | none | 20–40s with anchoring |
| Diffusion Forcing | unbounded | `$\lesssim 1$` | bounded/slow | none | stable rollout past trained length |
| Self-forcing / causal | unbounded | `$< 1$` (trained) | contractive | none | real-time + long, needs special training |
| Hierarchical keyframe | minutes | bounded per gap | interp quality | low, anchored | persistent subject across minutes |

Read the `$L$` column as the lever: the methods that move down the table are the methods that push the effective Lipschitz constant below 1, which is exactly the difference between a rollout that collapses in seconds and one that holds for a minute.

![Matrix comparing single-shot, chunked window, autoregressive, diffusion forcing, and hierarchical approaches across maximum length, coherence, drift, and seams](/imgs/blogs/long-video-and-autoregressive-rollout-4.png)

#### Worked example: budgeting a one-minute clip three ways

Take the one-minute 720p target and budget three strategies on an H100. **Pure autoregressive** with a 16-latent-frame window (≈2.7s of pixels per window at our 4× temporal compression and 24 fps) needs about `$60 / 2.1 \approx 28$` rollout steps after accounting for overlap; at ~30s per window that is ~14 minutes of compute, and identity has measurably drifted by step ~10 (≈20s) on most open models. **Chunked-parallel** generates ~12 independent 5s windows you can run in parallel across GPUs — fast wall-clock, but you fight 11 seams and the windows do not share identity, so it is best when the prompt tolerates loose continuity (b-roll, ambient scenes). **Hierarchical** generates ~30 keyframes (one every 2s) in a few global passes, then interpolates 29 gaps; the keyframes lock global identity, and each gap is a bounded 2s interpolation — the most coherent of the three for a single subject across the full minute, at the cost of needing a good keyframe model and an interpolator. The decision rule falls out: single subject that must persist → hierarchical or stabilized rollout; loose ambient continuity → chunked-parallel; never pure single-frame I2V chaining for anything past ~15s.

## 8. The causal 3D-VAE: what actually makes streaming possible

Everything in Sections 5 and 6 assumes you can produce and consume frames *incrementally* — generate a new latent chunk and decode it to pixels without re-decoding the whole clip and without waiting for the end. That capability is not free; it comes from the **causal 3D-VAE**, and it is worth understanding why the causality of the VAE, specifically, is what unlocks streaming long video. (For the full treatment of the VAE itself, see the [video autoencoders post](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression); here I only need the streaming consequence.)

A non-causal 3D-VAE uses temporal convolutions that look both backward and forward in time — to decode frame `$t$` it needs frames around `$t$` on both sides. That is fine for a fixed clip you decode all at once, but it makes streaming impossible: you cannot decode the latest latent until you have the *future* latents, which you do not have yet in a rollout. A **causal** 3D-VAE replaces those with **causal temporal convolutions** — frame `$t$`'s decode depends only on frames `$\le t$`, never on the future. Combined with a small carried-over **causal cache** (the receptive-field state from past frames, exactly analogous to a KV cache in a causal transformer), this lets the decoder emit each chunk of frames the instant its latents are ready, using only past context. Figure 8 shows the streaming decode path: latent stream in, causal cache feeding the decode, frames streamed out, next latent appended forward — no future peek, no full re-decode.

![Graph of a causal 3D-VAE streaming decode where a chunked latent stream and a past-only causal cache feed a causal decode that streams out frames and appends the next latent forward](/imgs/blogs/long-video-and-autoregressive-rollout-8.png)

This is why the causal VAE is the quiet hero of long video. Without it, "long video" would mean "generate all the latents, then decode the whole thing at the end" — which slams straight back into the VAE decode-memory wall from Section 1, because you would have to hold the entire pixel-resolution activation stack at once. With it, you decode chunk by chunk, hold only one chunk's worth of decoder activations plus the small cache, and stream frames out continuously. The memory cost of decode goes from `$O(\text{total length})$` to `$O(\text{chunk length})$`, which is the difference between OOM and a render that runs forever. Causal attention in the denoiser (Section 6) and causal convolution in the VAE (here) are the same idea applied to the two halves of the stack, and together they are what make a truly streaming, unbounded-length pipeline physically possible.

### The honest limit: tiling and the cache are not free

Two caveats keep this honest. First, **VAE tiling** — the standard trick to fit decode in VRAM by decoding spatial tiles separately — interacts badly with temporal causality if done naively, because tile boundaries can introduce seams that the causal cache does not smooth. Production systems overlap tiles spatially (the same blend idea as Section 2, in space) to hide them. Second, the causal cache means the *first* few frames of a rollout have a smaller temporal receptive field than later frames (there is no past to attend to yet), so a streamed clip can have a brief warm-up where temporal stability is weaker. Neither is fatal, but both are the kind of detail that separates a demo from a shipped pipeline.

## 9. Keeping identity and scene coherent across minutes

The Lipschitz analysis says drift is inevitable unless you make the rollout contractive. The practical question is: *what do you feed the model so it can contract?* The answer is **memory and anchoring** — give the model enough persistent information about the subject and scene that each new chunk is pulled back toward the original rather than wandering from the immediately-previous chunk. There are three complementary mechanisms, in increasing strength.

**History conditioning beyond one frame.** The single-frame I2V chaining in Section 6 drifts fast precisely because one frame carries almost no identity — it specifies appearance at one instant but not the underlying object. Conditioning on the last `$k = 4$` to `$8$` frames (as the sliding-window loop in Section 2 does) gives the model motion context and a richer appearance anchor, which lowers the effective `$L$`. More history is better up to the point where it costs too much attention; this is the cheapest, most reliable improvement.

**A persistent identity anchor.** Stronger systems keep a *fixed* reference — the seed frame, a reference image, or a learned identity embedding — and condition *every* chunk on it, not just on the recent past. This is the difference between a Markov chain (only the last state matters) and an anchored chain (every state also sees a fixed origin). Anchoring directly fights the geometric error growth from Section 4: even if the recent chunk has drifted, the fixed anchor re-injects the original identity at every step, which is a contractive force toward the origin. The cost is that a hard anchor can fight legitimate change (the dog *should* be able to turn around), so the anchor is usually a soft conditioning the model can override.

**A memory bank or long-range context.** The frontier systems maintain an explicit memory — a compressed bank of past frames or latents that the model can attend to across the whole rollout, so an object that left the frame ten seconds ago and returns is still the same object. This is the video analog of long-context attention, and it is expensive, which is why most open models do not have it yet. It is also why the honest current limit is what it is.

Here is how anchoring slots into the rollout loop in practice — keep the recent history *and* a fixed seed anchor, and condition every chunk on both:

```python
@torch.no_grad()
def anchored_rollout_step(pipe, prompt, recent_history, seed_anchor,
                          anchor_weight=0.3, **kw):
    # recent_history: last k generated latent frames (Markov context).
    # seed_anchor:    the FIXED seed frame's latent (identity origin).
    # Blend the seed anchor into the conditioning so every chunk is pulled
    # back toward the original identity, not just the previous (drifted) chunk.
    cond = torch.cat([
        seed_anchor.expand_as(recent_history[:, :, :1]) * anchor_weight
        + recent_history[:, :, :1] * (1 - anchor_weight),  # anchored first slot
        recent_history[:, :, 1:],                          # raw recent motion
    ], dim=2)
    return generate_conditioned_window(pipe, prompt, anchor=cond, **kw)
```

The `anchor_weight` is the knob that trades drift-resistance against legitimate change: at `$0$` you have pure Markov rollout (fast drift), at `$1$` you have a hard anchor that fights any motion away from the seed (stutter, frozen identity). Somewhere around `$0.2$`–`$0.4$` usually buys most of the identity preservation while still letting the subject move and turn. This soft anchor is the cheapest contractive force you can add to a rollout — in Section 4's terms, it directly subtracts a fraction of the accumulated error each step by re-injecting the origin.

### The honest current limits

Let me be blunt about where open models stand in 2026, because the marketing reels hide it. **Most open models hold quality for the first 5–15 seconds and decay noticeably past 20–30 seconds.** Identity slips, fine detail washes out, motion gets sluggish or jittery, and colors drift. The proprietary frontier (Sora 2, Veo 3.1, Kling 3.0) is meaningfully better — coherent shots in the tens of seconds, and minute-plus with planning — because they combine stabilized rollout, strong memory/anchoring, hierarchical structure, and far more compute than any open release. But even there, true minute-long coherence with a persistent subject and complex motion is at the edge of what works, and "long video" demos are often carefully chosen prompts (slow motion, limited subject change) that flatter the method. Figure 7 plots the decay honestly: sharp and on-identity for the first window, coherent for the first 15 seconds, slow drift through 30, identity slipping by 60, collapse beyond — the shape every open rollout shows, just shifted right or left by how well it contracts.

![Timeline of rollout quality decay from sharp on-identity output in the first seconds, through coherent and slow-drift phases, to identity slipping and collapse past a minute](/imgs/blogs/long-video-and-autoregressive-rollout-7.png)

## 10. Case studies: real numbers from shipped systems

Concrete results, cited and honest. Treat the exact figures as approximate where I say so — I would rather be defensibly order-of-magnitude than precisely wrong.

**Diffusion Forcing (Chen et al., 2024).** The original paper introduced per-token independent noise levels and demonstrated stable, variable-length rollout on video and on planning/decision tasks, showing that the per-frame-noise formulation lets a model roll out *past its training horizon* without the immediate collapse that fixed-noise rollout shows. The headline is qualitative but important: it is the formulation that made "roll out and stay stable" a tractable training objective rather than a hope. Subsequent video systems (the "Diffusion Forcing Transformer" line and follow-ups) carried it into larger video DiTs.

**CausVid / self-forcing line (2024–2025).** Causal, distilled video models demonstrated frame-by-frame streaming generation at dramatically reduced step counts — distilling a strong bidirectional teacher into a few-step or single-step causal student. The reported wins are on *latency and length together*: streaming generation approaching interactive rates while extending well past a single window, with self-forcing training (rolling out on the model's own outputs) specifically credited for taming the drift that plain causal distillation leaves behind. This is the through-line to the [real-time post](/blog/machine-learning/video-generation/efficient-and-real-time-video-generation) — causal+distilled is simultaneously the long-video path and the real-time path.

**HunyuanVideo and the open frontier (2024–2026).** HunyuanVideo and its 1.5 update, alongside Wan 2.x and CogVideoX, converged on the causal-3D-VAE + DiT + flow-matching recipe and pushed open clip lengths up — HunyuanVideo-1.5 has been reported generating notably long clips on a single high-end consumer card (on the order of tens of seconds at lower resolution on a 24GB-class GPU, approximate). The pattern across all three: native single-shot length is bounded by VAE and attention, and longer outputs come from chunked or rollout extensions layered on top, with the expected quality decay past the trained window.

**Sora and the world-simulator framing (Brooks et al., 2024).** Sora's technical report leaned on spacetime patches and scale, and its longer coherent shots are widely attributed to a combination of strong global modeling and very large compute rather than a single rollout trick — consistent with the thesis here that length at quality is bought with global coherence (planning/memory) plus stabilized extension, not with naive autoregression. The "world simulator" claim is the subject of [the Sora post](/blog/machine-learning/video-generation/sora-and-the-world-simulator-thesis); here the relevant point is that even the strongest systems treat long coherence as a first-class problem, not a free byproduct.

To make the landscape concrete, here is how the main long-video approaches map onto representative systems and their honest, approximate practical horizons. Treat the numbers as order-of-magnitude — they shift with resolution, hardware, and the exact prompt — but the *shape* is what matters.

| System / method | Long-video mechanism | Approx. coherent horizon | Streaming? | Notes |
| --- | --- | --- | --- | --- |
| SVD I2V chaining | single-frame autoregressive | ~10–15s before drift | no | crudest rollout, fastest collapse |
| Diffusion Forcing | per-frame noise rollout | past trained length, stable | yes | the stabilizing formulation |
| CausVid / self-forcing | causal + distilled, trained on rollouts | tens of seconds, near real-time | yes | merges long + real-time path |
| HunyuanVideo / Wan / CogVideoX | bounded single-shot + chunked extend | ~5–15s native, more via stitching | partial | converged open recipe |
| Sora 2 / Veo 3.1 / Kling 3.0 | global modeling + planning + scale | tens of seconds, minute-plus curated | mostly no | proprietary, far more compute |

**The interpolation-anchored long-video line.** A distinct family worth naming gets length from hierarchical structure rather than rollout: generate a sparse set of keyframes spanning the whole duration with a model that sees the global timeline, then densify with a strong interpolator. The honest result here is that *global identity is excellent* (the keyframes were generated together) but *local motion can be soft* in the interpolated gaps, especially across large motion — the failure mode is mushy in-betweens, not identity collapse. This is the opposite trade from autoregression: hierarchical methods sacrifice fine local dynamics to buy global coherence, while rollout sacrifices global coherence to buy fine local dynamics. The strongest production pipelines combine both — hierarchical plan, stabilized rollout within segments — which is why the capstone playbook recommends layering them rather than picking one.

## 11. When to reach for each approach (and when not to)

A decisive recommendation section, because the trade-offs above only matter if they change what you do.

- **Want a fixed short clip (≤ 5–8s)? Do not autoregress at all.** Generate it single-shot with full bidirectional attention. Rollout only buys you length you do not need, at the cost of drift you do not want. This is the single most common mistake — reaching for a rollout loop when one forward pass would have been sharper and simpler.
- **Want loose, ambient continuity (b-roll, backgrounds, no persistent hero subject)? Use chunked-parallel.** Generate independent windows, blend the overlaps, run them in parallel across GPUs for fast wall-clock. Accept that the windows will not share a precise identity — for ambient content that is fine, and you get stability and speed for free.
- **Want a persistent subject across 20–60s? Use a stabilized rollout (Diffusion Forcing or self-forcing) plus strong anchoring,** or a hierarchical keyframe-then-interpolate plan. Plain single-frame I2V chaining will drift; richer history conditioning plus a fixed identity anchor is the minimum bar. If you have access to a causal/distilled model, that is the right tool — it gives you length and streaming together.
- **Want real-time or interactive (game-like, streamed) generation? You need a causal, distilled model.** Bidirectional diffusion cannot stream; only causal attention plus few-step distillation gets you frame-by-frame output. This is the same tooling as long video, which is why the two frontiers merged.
- **Need minute-plus with complex motion and scene change? Be honest that open models are not there yet.** Plan hierarchically, anchor hard, expect to curate, and reach for the proprietary frontier if quality at length is the product. Do not promise a minute of coherent complex action from an open 5-second model and a rollout loop — it will drift, and the worked examples above tell you roughly when.

## 12. How to measure long-video drift honestly

You cannot improve what you do not measure, and long-video quality is easy to measure dishonestly. A few principles, building on the [metrics post](/blog/machine-learning/video-generation/the-metrics-of-video-generation).

Report quality **as a function of time**, not a single number. A model that scores well on a 5-second VBench run can collapse at 30 seconds; a single aggregate hides exactly the failure you care about. Plot subject-consistency and background-consistency (the VBench dimensions most sensitive to drift) against rollout position, and you will see the decay curve from Figure 7 directly. Fix the seed, fix the prompt, and warm up the pipeline before timing, so the curve reflects the method and not run-to-run noise.

Watch for the **dynamic-degree-versus-stability gaming problem.** A trivially stable rollout is one that barely moves — a near-static clip will score high on subject consistency because nothing changes, but it is a bad video. So always report a motion/dynamic-degree metric alongside the consistency metrics: the honest claim is "stays coherent *while still moving*," and a method that buys stability by killing motion has not solved the problem, it has dodged it. The right summary is the Pareto point — consistency at a fixed dynamic degree, across rollout length — not either axis alone.

#### Worked example: a drift-versus-length measurement protocol

Concretely: pick 50 prompts with a clear persistent subject, fix one seed each, and roll each out to 60 seconds in 5-second increments. At each increment compute (a) CLIP-image similarity between the current frame and the seed frame as a cheap identity-drift proxy, (b) VBench subject-consistency over the local window, and (c) a dynamic-degree score so you can rule out the static-clip cheat. Plot all three against time. A good method holds CLIP-to-seed similarity roughly flat (identity preserved) while keeping dynamic degree above a floor; a drifting method shows CLIP-to-seed similarity falling monotonically — that falling curve *is* the `$\|e_t\|$` from Section 4 made visible. Report the time at which CLIP-to-seed crosses a threshold (say, drops 15% from the seed) as the method's practical coherence horizon. That single number — "coherent to ~X seconds at dynamic degree Y" — is the honest headline for a long-video method.

## Key takeaways

- **There are two length walls, not one.** Quadratic attention in frames plus a fixed-trained-length, decode-memory-bound VAE cap single-shot generation at roughly 5–10 seconds at 720p on an 80 GB card. You must generate long video in pieces.
- **Generating in pieces forces a choice between bounded-stable and unbounded-drifting.** Chunked-parallel windows are stable but bounded and seamed; autoregressive rollout is unbounded but accumulates error. This is a real frontier, not a solved problem.
- **Error accumulation is exponential when the model amplifies its own errors (`$L>1$`), linear at marginal stability (`$L=1$`), bounded only when the rollout is contractive (`$L<1$`).** Every stabilization technique is an attempt to push the effective Lipschitz constant below 1.
- **It is exposure bias.** The model trains on clean ground-truth conditioning and infers on its own dirty output; that train-test gap is the same disease as in autoregressive sequence models, and the fix is the same in spirit — train on your own rollouts (self-forcing, scheduled sampling).
- **Diffusion Forcing's per-frame noise levels unify conditioning and generation,** letting clean history, low-noise near-future, and high-noise far-future denoise jointly, which is a contractive, stabilizing rollout.
- **Causal models stream; bidirectional models cannot.** Causal temporal attention in the denoiser and causal temporal convolution in the VAE (plus its cache) are what make frame-by-frame, unbounded, streaming generation physically possible — and they are the same tooling as real-time generation.
- **Identity survives minutes only with memory and anchoring** — multi-frame history, a persistent identity anchor, or a long-range memory bank — because raw Markov rollout off the last frame drifts fastest.
- **Be honest about the limit.** Most open models hold quality for 5–15 seconds and decay past 20–30; minute-plus coherence with complex motion is at the edge of the proprietary frontier and is often demo-curated.
- **Measure drift as a function of time, at a fixed dynamic degree.** A single aggregate score hides the decay, and stability bought by killing motion is not a win.

## Further reading

- Chen, Boyuan, et al. "Diffusion Forcing: Next-Token Prediction Meets Full-Sequence Diffusion." 2024 — the per-token-noise formulation at the center of this post.
- Ho, Jonathan, et al. "Video Diffusion Models." 2022 — the foundational extension of diffusion to the time axis.
- Blattmann, Andreas, et al. "Stable Video Diffusion." 2023 — the open image-to-video model used in the chaining sketch.
- Brooks, Tim, et al. "Video generation models as world simulators" (Sora technical report). 2024 — spacetime patches and the scale thesis for long coherent shots.
- Peebles, William, and Saining Xie. "Scalable Diffusion Models with Transformers" (DiT). 2023 — the transformer backbone the video DiT inherits.
- The HunyuanVideo, Wan 2.x, and CogVideoX technical reports (2024–2026) — the converged open causal-VAE + DiT + flow-matching recipe and its real length limits.
- Within series: [why video generation is hard](/blog/machine-learning/video-generation/why-video-generation-is-hard) (the foundation), [video autoencoders and spatiotemporal compression](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression) (the causal VAE that enables streaming), [efficient and real-time video generation](/blog/machine-learning/video-generation/efficient-and-real-time-video-generation) (the sibling real-time path), and the capstone [building with video generation, the playbook](/blog/machine-learning/video-generation/building-with-video-generation-the-playbook).
- Link out: [autoregressive image models](/blog/machine-learning/image-generation/autoregressive-image-models) for the exposure-bias parallel in the image domain.
