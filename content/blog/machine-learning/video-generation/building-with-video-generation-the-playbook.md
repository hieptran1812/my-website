---
title: "Building With Video Generation: The End-to-End Playbook"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "The capstone of the series: a practitioner's decision tree across hosted and open video models, the full inference pipeline and the knob at each stage, when to use prompt versus first-frame versus a video-LoRA, a concrete speed-and-cost model, long-video rollout, the evaluation-and-safety layer, and three worked ship-a-product scenarios with real dollar-per-clip stacks."
tags:
  [
    "video-generation",
    "diffusion-models",
    "text-to-video",
    "video-diffusion",
    "image-to-video",
    "video-lora",
    "inference-optimization",
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
image: "/imgs/blogs/building-with-video-generation-the-playbook-1.png"
---

You have read the series. You know that video is image generation times the hardest extra axis — time — and you know the stack that the field converged on: a causal 3D-VAE that crushes a clip into a manageable latent, a spacetime diffusion transformer that denoises that latent, a flow-matching sampler that walks it from noise to signal, and conditioning that steers the whole thing with text, a first frame, motion, and sometimes audio. You know why each piece exists and roughly how it works. Now someone hands you a brief — "ship a tool that turns a product photo into a five-second marketing clip, on a budget, by Friday" — and every one of those pieces becomes a decision with a price tag.

This is the post where the series stops explaining and starts deciding. It is a playbook, which means it is opinionated on purpose. I will not tell you that "it depends" and leave you there; I will tell you what I would pick, why, and exactly what it costs, and then I will tell you the three or four cases where I would pick the opposite. The shape of every decision here is the same recurring tension the whole series has circled: coherence times motion times length times cost. You cannot maximize all four. The job of a practitioner is to know which one the product actually needs and to spend the other three buying it down.

I want to be precise about what "the series taught" means, because this post assumes it and builds on it. You learned *why* video is hard — that the time axis multiplies your token count and your sampling cost, and that temporal coherence is a property the model must actively maintain rather than get for free. You learned that the [causal 3D-VAE](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression), not the denoiser, is the real lever for both cost and length, because it sets the token count that everything downstream pays for. You learned how the [spacetime diffusion transformer](/blog/machine-learning/video-generation/video-diffusion-transformers) denoises that latent, why [flow matching](/blog/machine-learning/video-generation/flow-matching-for-video) is the sampler the modern stack trains under, and how [conditioning](/blog/machine-learning/video-generation/conditioning-video-text-image-motion-camera) steers it. You met the frontier — the open recipe, the hosted leaders, the distilled real-time line. This post does not re-teach any of that; it *uses* it, turning each piece of understanding into a lever you pull with a known cost. If a lever surprises you, the post that explains it is one click away.

![A vertical stack of the six video inference stages from text and image encoder through 3D-VAE encode, spacetime-DiT denoiser, flow-matching sampler, VAE decode, and frame plus audio mux, with the denoiser marked as the repeated cost and the decode marked as the VRAM wall](/imgs/blogs/building-with-video-generation-the-playbook-1.png)

The map above is the whole post in one picture, and it is the map of the whole series. Six stages: encode the prompt and any conditioning image, encode pixels to latent through the 3D-VAE, run the spacetime denoiser for N flow-matching steps, decode the clean latent back to frames, and mux in audio. Every architecture you have read about — Sora, Veo, Wan, HunyuanVideo, CogVideoX, LTX — is some instantiation of this exact pipeline. The differences that matter to you as a builder are not the research differences; they are which stages you control, which stages dominate your latency and your bill, and which knob at which stage buys you the quality, speed, or length your product needs. We will walk it end to end: choosing a model class, the inference pipeline knob by knob, customization from prompt to LoRA, speed and deployment to a latency target, long video, evaluation and safety before you ship, and three worked product scenarios with real dollar-per-clip numbers. If anything here references a mechanism you want re-derived, the series has a post for it; I will link as we go rather than repeat. The foundation is always [why video generation is hard](/blog/machine-learning/video-generation/why-video-generation-is-hard) — that post is the one-paragraph version of everything below.

## 1. The first decision is a class, not a model

The mistake almost everyone makes on day one is to open a benchmark leaderboard and pick the model with the best VBench score. That is the wrong first question. The right first question is which *deployment class* you are in, because the class — not the model — decides your cost structure, your control surface, and what you are even permitted to tune. Get the class right and the model choice inside it is a footnote; get the class wrong and no model can save you.

There are three classes, and they map cleanly onto three things you might value most.

**Hosted APIs** — Google's Veo, OpenAI's Sora, Kuaishou's Kling — are the class you choose when you value maximum quality and zero infrastructure above everything else. You send a prompt (and optionally a first frame) over HTTP, you get back an MP4. You do not own a GPU, you do not manage a queue, you do not debug a VAE decode that OOMs at second six. In exchange you give up control — you cannot fine-tune the model, you often cannot touch the sampler — and you pay per clip, which at scale is the most expensive path per generated second. This is the class for a marketing tool, a consumer app where quality is the product, or any case where native synchronized audio matters, because as of 2026 the hosted models are the only ones that generate sound and picture jointly. The [cinematic-quality bar these models set](/blog/machine-learning/video-generation/veo-and-cinematic-generation) is genuinely above what you can self-host today.

**Self-hosted open models** — Wan 2.x, HunyuanVideo 1.5, CogVideoX, LTX-Video — are the class you choose when you value control, cost-at-scale, and privacy. You run the weights on your own GPU. You can fine-tune them, you can swap the sampler, you can train a video-LoRA on your brand's footage, and your users' frames never leave your machine. Your marginal cost per clip is just electricity and amortized hardware, which at volume is one to two orders of magnitude below a hosted API. The price you pay is engineering: you own the VRAM wall, the latency, the queueing, and the quality gap to the frontier. The good news the series has hammered is that this gap is closing fast, because [the open frontier converged on one shared recipe](/blog/machine-learning/video-generation/the-open-video-frontier-wan-hunyuanvideo-cogvideox) — causal 3D-VAE plus DiT plus flow matching — so a 14B-class open model in 2026 sits within striking distance of last year's hosted quality.

**Distilled and real-time models** — step-distilled Wan/LTX students, CausVid, the few-step line — are a sub-class of open models you choose when latency is the binding constraint. A four-step LTX student produces a clip in seconds on a single GPU instead of a minute. You sacrifice some quality and some motion range for a ten-times speedup and a ten-times lower bill. This is the class for interactive tools, previews, and anything where a person is waiting and watching. The mechanics — consistency, distribution-matching, and adversarial distillation adapted to the spacetime denoiser — are the subject of [the efficient and real-time post](/blog/machine-learning/video-generation/efficient-and-real-time-video-generation).

![A decision tree rooting at need a video that branches into hosted API, self-hosted open, and distilled real-time classes, each splitting into concrete model families such as Veo and Sora, Wan and HunyuanVideo, and the LTX four-step student](/imgs/blogs/building-with-video-generation-the-playbook-2.png)

The tree above is the decision in one glance, and notice that the hard fork is at the top. The two questions that resolve almost everything are: *do you need maximum quality with zero infrastructure* (go hosted), and *do you need control, privacy, or cost-at-scale* (go open, then ask whether latency forces you down to a distilled student). Everything below those forks is a model-family detail you can change later without re-architecting.

#### Worked example: routing three briefs to a class

Take three real briefs and route each one. Brief A: "a consumer app that turns a selfie into a fun talking clip with a voice." Audio and quality are the product, you have no ML team, and volume is unknown — that is hosted, full stop; the native-audio requirement alone rules out self-hosting in 2026. Brief B: "an internal tool that renders 50,000 product clips a month from our catalog photos, frames must never leave our VPC." Privacy and per-clip cost dominate, quality needs to be good not perfect — that is self-hosted open, almost certainly an [image-to-video](/blog/machine-learning/video-generation/conditioning-video-text-image-motion-camera) model since you have the product photo as the first frame. Brief C: "a creative playground where users tweak a prompt and see the result update in under three seconds." Latency is everything — that is a distilled four-step model, and you will trade away the top of the quality range to hit the interactivity bar. Same three classes, three different products, and the model name inside each was the easy part.

The numbers under these classes are worth pinning down before we go further, because the cost gap is the whole reason the decision matters.

| Class | Example models | Quality | Cost / clip | Control | Native audio |
|---|---|---|---|---|---|
| Hosted API | Veo 3.1, Sora 2, Kling 3.0 | Top tier | \$0.30–0.75 | Prompt (+ frame) only | Yes |
| Self-hosted open | Wan 14B, HunyuanVideo 1.5, CogVideoX 5B | Near-hosted | \$0.02–0.08 | Full + LoRA | No |
| Distilled / real-time | LTX 4-step, CausVid, distilled Wan | Fair–good | \$0.003–0.01 | Full | No |

Those per-clip figures are for a roughly five-second clip and are order-of-magnitude estimates, not quotes; hosted prices move and open-model cost depends entirely on your GPU utilization. But the *ratios* are stable and they are the point: self-hosting is roughly ten times cheaper per clip than a hosted API at decent utilization, and a distilled student is another three-to-ten times cheaper than that. If your product generates a handful of clips a day, the hosted API's higher per-clip price is irrelevant and its zero-ops advantage wins easily. If it generates fifty thousand a month, that ten-times multiplier is the difference between a viable business and a GPU bill that eats your margin.

There is a subtler point hiding in that utilization caveat, and it is the thing that trips up most cost models. A self-hosted GPU costs the same whether it renders one clip an hour or thirty; the per-clip cost is the hourly price divided by clips-per-hour, so it collapses as utilization rises and explodes as the machine sits idle. A 4090 at \$0.50/hr rendering one clip a minute costs about \$0.008 per clip; the same 4090 rendering one clip an hour costs \$0.50 per clip — sixty times more, and now indistinguishable from a hosted API. This is why self-hosting only wins at *sustained* volume. A bursty workload — busy at noon, idle at night — wastes most of the hardware you are paying for around the clock, and unless you can autoscale the GPU fleet down to zero (rare, because cold-loading a 14B model takes tens of seconds), a hosted API's pay-per-use model may genuinely be cheaper despite its higher headline price. The break-even is not a clip count; it is a *utilization* threshold, and you should compute it for your actual traffic shape before committing to a fleet.

The other thing the table hides is that "cost per clip" is not the only cost. A hosted API has near-zero fixed cost and a linear marginal cost. A self-hosted stack has a large fixed cost — the engineering to build serving, the on-call burden, the GPU reservation — and a tiny marginal cost. The two cross at some volume. Below it, the API's zero fixed cost dominates and you should not self-host no matter how cheap the per-clip number looks. Above it, the self-hosted marginal cost dominates and the API's premium compounds into real money. Plotting your own version of that crossover, with your real engineering cost and your real traffic, is the single most valuable hour you can spend before writing any code.

## 2. Quality, cost, control, length, audio — read all five together

The class decision narrows you to a family. Picking the model inside the family is a five-axis trade-off, and the single most useful habit I can give you is to *read all five axes at once* and identify which one you can afford to lose. No model is best on all five. The right model is the one whose weakest axis is the axis your product does not care about.

![A matrix scoring five model approaches across quality, cost per clip, control, length, and audio, showing hosted APIs winning quality and audio while open models win control and cost](/imgs/blogs/building-with-video-generation-the-playbook-3.png)

Walk the matrix above column by column and the structure jumps out. **Quality** is led by hosted APIs, with 14B open models a close second and distilled students a clear third — distillation always costs you some fidelity and some motion range. **Cost per clip** runs exactly the other way: distilled students are cheapest, open models next, hosted APIs most expensive. **Control** is binary in practice — hosted gives you a prompt and maybe a first frame, open gives you the full surface including fine-tuning. **Length** is everyone's weak spot; almost every model trains on clips of five to ten seconds and anything longer is a rollout trick (more on that in section 7). **Audio**, as of 2026, is a hosted-only luxury — the open models generate silence and you bolt sound on afterward.

The discipline is to look at your product and cross out the axis you can sacrifice. A marketing tool sacrifices cost (clips are low-volume and high-value, so \$0.50 a clip is nothing). An internal batch renderer sacrifices audio and a little quality (it needs cost and control). A real-time playground sacrifices quality and length (it needs latency, which the matrix encodes as cost). Once you have crossed out one axis, the model usually picks itself.

There is a sixth axis the matrix does not show because it is not a model property but a *task* property, and it is the highest-leverage choice of all: text-to-video versus image-to-video.

### Why image-to-video usually wins

A text-to-video model is asked to do two hard things at once: invent the very first frame — every object, its appearance, the lighting, the composition — and then animate it coherently. An image-to-video model is handed the first frame and asked to do only the second thing. That single difference removes an entire degree of generative freedom, and it shows up directly in the metrics: I2V models reliably post higher subject-consistency and aesthetic scores than the same backbone run as T2V, because the appearance is *pinned* rather than hallucinated and then maintained.

![A before-and-after comparison contrasting text-to-video, which must invent frame zero and risks identity drift, with image-to-video, which is given frame zero and generates only motion for higher consistency](/imgs/blogs/building-with-video-generation-the-playbook-4.png)

The before-and-after above is the argument in one figure. On the left, T2V invents frame zero, the prompt is the only anchor, and identity tends to drift as the clip plays. On the right, I2V is given frame zero, generates motion only, and posts higher consistency. The practical rule that falls out: **if you can supply a first frame, supply it.** You almost always can — a product photo, a character sheet, a still rendered by a cheaper image model. The series' [conditioning post](/blog/machine-learning/video-generation/conditioning-video-text-image-motion-camera) goes deep on how that frame is injected (it is encoded by the same 3D-VAE and concatenated into the latent the denoiser sees), and the practical upshot is that an I2V pipeline built on a strong image generator front-end often beats a pure T2V model of the same size. If you have already built an [image generation stack](/blog/machine-learning/image-generation/building-an-image-generation-stack), you are most of the way to a high-quality video pipeline: generate the keyframe with your image model, animate it with I2V.

When does T2V still win? When you genuinely have no first frame and no cheap way to make one, when the brief is "surprise me" rather than "animate this," or when the model you want is T2V-only. But those are the exceptions. The default, on quality grounds, is I2V.

There is a third task mode worth naming: video-to-video (V2V), where you supply an existing clip and the model transforms it — restyling it, extending it, or editing its content while preserving its motion. V2V is the right mode when motion is the thing you want to keep and appearance is the thing you want to change: turn a real dance clip into the same dance in a different art style, or extend a 3-second clip to 8. Mechanically it is I2V's cousin — the input clip is encoded by the same 3D-VAE into a latent that conditions the denoiser — and it inherits I2V's quality advantage because, again, the model is handed structure instead of inventing it. The practical caution is that V2V quality is bounded by how faithfully the VAE round-trips your input clip; if your source is high-motion or off-distribution, the encode-decode loop can soften it before the model even starts. For most products the choice is simply: have a target image, use I2V; have a target clip, use V2V; have neither, fall back to T2V and accept the harder task.

## 3. The inference pipeline, knob by knob

Now we open the box. Whatever class and model you chose, generation runs the six-stage pipeline from figure 1, and at each stage there is a knob — sometimes several — that moves quality, speed, length, or VRAM. Knowing the knobs is what separates someone who runs the default `pipe()` call and accepts whatever comes out from someone who can hit a target.

![A dataflow graph showing the prompt embedding and an optional first-frame latent both feeding into the spacetime-DiT denoiser alongside the noise latent, merging and denoising to a clean latent that decodes to frames](/imgs/blogs/building-with-video-generation-the-playbook-5.png)

The dataflow above shows the heart of it: the text embedding and the optional first-frame latent enter on separate paths and *merge inside the denoiser*, which is mechanically why I2V is "free" quality — you are handing the denoiser a known starting point instead of asking it to invent one. Let me walk every stage and name its knob.

**Stage 1 — encode the conditioning.** Text goes through a frozen encoder (T5, CLIP, or an LLM-class encoder depending on the model). The knob here is the *prompt* itself and the *guidance scale*. Higher guidance pushes the output to follow the prompt harder at some cost to motion and naturalness; video models are more guidance-sensitive than image models because over-guiding tends to freeze motion. If you supply a first frame, it is encoded by the 3D-VAE in this stage too.

**Stage 2 — 3D-VAE encode (I2V/V2V only).** For pure T2V you skip straight to noise. For I2V or video-to-video, the conditioning image or clip is compressed to latent by the [causal 3D-VAE](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression). There is no quality knob here, but there is a *correctness* gotcha: the VAE's temporal compression ratio determines how your first frame maps to latent frames, and mismatches here are a common source of a flickery first half-second.

**Stage 3 — the spacetime-DiT denoiser, run N times.** This is the dominant cost and it has the most important knobs. **`num_frames`** sets the clip length (bounded by what the VAE was trained on — exceed it and coherence collapses). **Resolution** sets the spatial token count. **The sampler and `num_inference_steps`** set how many denoiser passes you pay for — this is your single biggest latency lever. **Motion/camera conditioning** (a `motion_bucket_id` in SVD, trajectory or camera-pose inputs in others) steers how much and how the scene moves. And **caching** — reusing denoiser features across adjacent steps — can cut the effective step count without retraining.

**Stage 4 — the flow-matching sampler.** Modern video models use [flow matching](/blog/machine-learning/video-generation/flow-matching-for-video) rather than the older DDPM formulation, which is why their straight-line probability paths sample well in few steps. The knob is the sampler class and its step count; `FlowMatchEulerDiscreteScheduler` at 30–50 steps is the quality default, and the few-step students push that to 4–8.

**Stage 5 — 3D-VAE decode.** Here is the knob nobody expects to matter until it OOMs: **tiled/chunked decode** (`enable_vae_tiling()`, `decode_chunk_size`). Decoding a full 5-second 720p latent back to pixels in one shot can need more VRAM than the entire denoiser. Tiling decodes in spatial or temporal chunks so peak VRAM stays bounded. On a distilled model where the denoiser is fast, the VAE decode can be the single largest term in your wall-clock time.

**Stage 6 — frames and audio mux.** Export frames to a container with `export_to_video`, and if you generated or sourced audio separately, mux it in. For open models this is also where you would run a talking-head or [joint audio-video](/blog/machine-learning/video-generation/audio-and-joint-av-generation) step if your stack includes one.

Here is the whole pipeline as code, with every knob exposed and commented, for a CogVideoX image-to-video run — the kind of call that is the backbone of a self-hosted product:

```python
import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

# Load in bf16; the 5B model needs ~16-20 GB before offload tricks.
pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    "THUDM/CogVideoX-5b-I2V",
    torch_dtype=torch.bfloat16,
)

# --- The deployment knobs that keep this on a 24 GB card ---
pipe.enable_model_cpu_offload()      # stream modules on/off the GPU
pipe.vae.enable_tiling()             # tiled VAE decode -> bounds peak VRAM
pipe.vae.enable_slicing()            # decode one frame-slice at a time

first_frame = load_image("product_photo.png")   # the I2V anchor

video = pipe(
    image=first_frame,                # stage 2: the pinned first frame
    prompt="a sleek perfume bottle slowly rotating on a marble surface, "
           "soft studio light, shallow depth of field",
    num_frames=49,                    # stage 3: ~6s at the model's fps
    num_inference_steps=50,           # stage 3/4: quality default
    guidance_scale=6.0,               # stage 1: prompt adherence vs motion
    generator=torch.Generator().manual_seed(42),  # reproducibility
).frames[0]

export_to_video(video, "out.mp4", fps=8)   # stage 6: mux to a container
```

Every line above is a decision. Drop `num_inference_steps` to 8 with a distilled checkpoint and you trade a little quality for a 6x speedup. Drop `enable_model_cpu_offload()` and you need a bigger GPU but run faster. Raise `guidance_scale` and the bottle follows the prompt harder but may rotate more stiffly. Raise `num_frames` past what the VAE was trained on and the clip drifts. This single call, with these five knobs understood, is most of what running a self-hosted video product *is*.

A word on the offloading trio, because it is the difference between "fits on a 24 GB card" and "OOMs immediately," and the three calls do genuinely different things that people conflate. `enable_model_cpu_offload()` keeps the model on the CPU and streams each submodule onto the GPU only while it runs, then evicts it — so peak VRAM is roughly the largest single submodule, not the whole model, at the cost of PCIe transfer time each step. `enable_sequential_cpu_offload()` is the more aggressive cousin that offloads at a finer granularity for even lower VRAM and even higher latency — reach for it only when model-level offload still does not fit. `enable_vae_tiling()` and `enable_vae_slicing()` are orthogonal: they attack the *decode* stage specifically, splitting the VAE's pixel reconstruction into spatial tiles or per-frame slices so the decode's activation memory — which for a full 720p clip can exceed the denoiser's — stays bounded. The reason all three matter is that the VRAM budget has two distinct peaks, one during denoising and one during decode, and they are reached by different parts of the pipeline; you need a lever for each. A 14B model in bf16 on a 24 GB card without these calls does not run. With model offload plus VAE tiling, it does, a little slower, and that "a little slower" is the price of admission to consumer hardware.

There is also a quality knob hiding in `num_frames` that is worth making explicit, because it is the most common self-inflicted wound. Every model's VAE and denoiser were trained on a specific clip length — often 49 or 81 latent-conditioned frames depending on the temporal compression. Request that length and you are in-distribution; request 200 frames and you have left the training regime, and the failure mode is not a crash but a *silent* degradation — the back half of the clip drifts, repeats, or freezes. The model will happily generate it and it will look plausible for the first few seconds, which is exactly what makes the bug pernicious. The rule: stay within the model's documented frame count for a single generation, and reach for the rollout techniques in section 7 when you need more. Do not push `num_frames` and hope.

### A latency, VRAM, and cost model you can compute in your head

Before you tune knobs you should be able to predict what they do. The wall-clock time for one clip on one warm GPU decomposes cleanly, and this is the most useful equation in the whole playbook:

$$
T_\text{clip} = c_\text{enc} + N_\text{steps} \cdot c_\text{denoiser} + c_\text{vae},
$$

where $c_\text{enc}$ is the (small, fixed) text-and-image encode cost, $c_\text{denoiser}$ is the cost of one forward pass through the spacetime denoiser over the full latent, and $c_\text{vae}$ is the one-time decode. The asymmetry is everything: the denoiser term is multiplied by $N_\text{steps}$, the VAE term is paid once. When you run 50 steps, the denoiser dominates and the decode is a rounding error. When you distill down to 4 steps, the fixed $c_\text{vae}$ does not shrink, the two terms cross, and the VAE decode becomes your bottleneck — which is exactly why the efficient-inference work cares so much about a fast, high-compression VAE.

The per-step cost is itself governed by the token count $L$, which the VAE sets. The denoiser is a transformer over $L$ spacetime tokens, costing roughly $O(L \cdot d^2)$ for the projections and MLPs plus $O(L^2 \cdot d)$ for full attention. For a 5-second 720p clip at 24 fps — 120 frames of $1280 \times 720$ — a causal 3D-VAE with $4 \times 8 \times 8$ compression and patch size 2 produces about

$$
L \approx \frac{120}{4} \cdot \frac{720}{16} \cdot \frac{1280}{16} = 30 \cdot 45 \cdot 80 = 108{,}000
$$

latent tokens. That six-figure token count is why each step is expensive, and it is why the VAE — by setting $L$ — is simultaneously the lever for cost (through the $L^2$ attention term) and for length (more frames means more tokens means more memory). Now you can reason about every knob quantitatively:

- **Image-to-video** does not change $L$ but improves quality at fixed compute, so it raises quality-per-dollar without touching the equation.
- **Distillation** attacks $N_\text{steps}$ directly — the biggest single lever, a 6–12x cut.
- **Quantization** (fp8, int4) shrinks $c_\text{denoiser}$ and the weight memory roughly proportionally to the bit-width, with a small quality cost.
- **Caching** reduces the *effective* $N_\text{steps}$ by skipping recomputation on steps where features barely change.
- **Tiled decode** does not speed up $c_\text{vae}$ but caps its peak VRAM, which is what lets a big clip fit at all.

And the dollar cost follows immediately: at a GPU price of \$$p$ per hour, one clip costs $\$p \cdot T_\text{clip} / 3600$. At \$2/hr for a rented 4090-class GPU and a 30-second clip render, that is about \$0.017 a clip — which is how the open-model column in your cost table gets to single-digit cents.

## 4. Customization: reach for the cheapest knob that works

"Customization" is where teams burn the most time and money for the least reason, because the instinct is to fine-tune when a prompt edit would have done the job. The rule is a ladder: start at the cheapest rung and climb only when the rung below genuinely cannot meet the need. Three of the four rungs are *free at inference* — they cost no training. Only the top rung, a video-LoRA, costs GPU-hours.

![A matrix mapping four customization knobs, prompt edit, first-frame I2V, camera or motion conditioning, and video-LoRA, against what each controls, its cost, and when to use it](/imgs/blogs/building-with-video-generation-the-playbook-6.png)

The matrix above is the ladder. **Rung one: the prompt.** It controls content and style, it is free, and you try it first, always. A surprising fraction of "we need to fine-tune" requests evaporate once someone writes a better prompt with concrete lighting, motion, and camera language. **Rung two: the first frame (I2V).** It controls the exact opening appearance, it is free, and you use it whenever you have or can cheaply make a frame — which, per section 2, you almost always can. **Rung three: camera and motion conditioning.** It controls trajectory, pan, zoom, and motion strength, it is free, and you use it when the brief needs specific camera moves the prompt cannot reliably evoke; the [camera-control post](/blog/machine-learning/video-generation/camera-control-and-4d-generation) covers how explicit trajectories are injected. **Rung four: the video-LoRA.** It controls a subject identity or a brand look the base model simply cannot produce, it costs GPU-hours, and you climb to it only when rungs one through three demonstrably fall short.

The honest test for whether you need a LoRA: generate twenty clips with the best prompt plus first frame you can manage. If the model gets your subject or style right most of the time, you do not need a LoRA — you need better prompting and a curated first-frame library. If it systematically misses — wrong logo, wrong character face, a brand aesthetic it has never seen — then a LoRA is the right tool.

It is worth dwelling on rung three, camera and motion conditioning, because it is the rung most teams skip straight past — jumping from "the prompt isn't enough" to "let's fine-tune" — when the free middle rung would have solved it. Modern open models expose explicit control over how the scene moves: SVD's `motion_bucket_id` dials the overall amount of motion from a near-still clip to vigorous action; camera-trajectory conditioning lets you specify a pan, a dolly, an orbit, or a crane move as an explicit path the generated camera follows; and motion-strength or optical-flow conditioning steers how much the subject itself moves versus the background. These are *inference-time* controls — no training, no GPU-hours — and they solve a huge fraction of "the motion is wrong" complaints. If the brief is "a slow cinematic push-in on the product," that is a camera-trajectory input, not a LoRA and not even a prompt; the [camera-control post](/blog/machine-learning/video-generation/camera-control-and-4d-generation) shows exactly how these trajectories are injected into the denoiser. The discipline is the whole point of the ladder: each rung solves a different *kind* of failure, and reaching past a rung that fits your failure mode wastes time and money. Prompt fixes content, first-frame fixes appearance, camera/motion fixes movement, and only a LoRA fixes a learned identity the base model has never seen.

### A real video-LoRA workflow

When you do climb to the top rung, treat it as a small data-and-evaluation project, not a training stunt. The launch command in the middle is the easy part; the curate and evaluate ends are where the quality lives.

![A timeline of the video-LoRA workflow running from curating twenty to fifty on-style clips through captioning and bucketing, a peft training run, a VBench and spot-check evaluation, and finally serving via fuse or hot-swap](/imgs/blogs/building-with-video-generation-the-playbook-7.png)

The timeline above is the whole lifecycle. **Curate** 20–50 clips that exemplify exactly the subject or style you want — quality and consistency of this set matters far more than quantity; ten clean on-style clips beat a hundred noisy ones. **Caption and bucket** them with consistent language and a unique trigger token, and group by resolution and length so the dataloader is not padding wildly. **Train** with `peft` — a LoRA of rank 32–64 on the denoiser's attention projections is the standard, leaving the VAE and text encoder frozen. **Evaluate** on held-out prompts with VBench subject-consistency plus a human spot-check before you trust it. **Serve** by either fusing the LoRA into the weights (faster inference, one style) or hot-swapping adapters at request time (slower, many styles).

Here is the launch sketch — a `diffusers`/`peft` LoRA on a CogVideoX-class backbone, the kind of command that, with a curated dataset, produces a usable brand-style adapter in a few GPU-hours:

```bash
accelerate launch train_cogvideox_lora.py \
  --pretrained_model_name_or_path THUDM/CogVideoX-5b \
  --instance_data_root ./brand_clips \
  --caption_column prompts.txt \
  --rank 64 \
  --learning_rate 1e-4 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 30 \
  --mixed_precision bf16 \
  --gradient_checkpointing \
  --enable_model_cpu_offload \
  --output_dir ./brand_lora
```

And loading the trained adapter back into the inference pipeline is one call:

```python
pipe.load_lora_weights("./brand_lora", adapter_name="brand")
pipe.set_adapters(["brand"], adapter_weights=[0.9])  # dial the strength
# ... then call pipe() exactly as before; the "brand" style now applies.
```

The `adapter_weights` knob deserves a note: a video-LoRA at full strength can overpower motion and bake in the training clips' framing. Dialing it to 0.7–0.9 usually keeps the style while preserving the base model's motion quality — a small tuning loop worth running before you ship.

#### Worked example: when a LoRA is and is not worth it

A team wants every generated clip to feature their mascot — a specific cartoon fox with an exact color palette and a particular smile. They try prompting ("an orange cartoon fox with a white-tipped tail, friendly expression"): the model produces *a* fox, but the face and palette drift clip to clip. They try I2V with a reference still: better, the opening frame is right, but as the fox moves its face morphs off-model by second three. This is the textbook LoRA case — a specific identity the base model cannot hold through motion. They curate 40 short clips of the mascot from existing assets, train a rank-64 LoRA in about four GPU-hours on a single H100 (call it \$8 of compute at \$2/hr), and now the fox stays on-model across the whole clip. Contrast that with a second team who wants "a cinematic, slightly desaturated look." They almost reach for a LoRA too — and they should not. That is a *style* the prompt and a color-grade post-step handle for free. The discipline is to climb the ladder, not jump to the top.

## 5. Speed and deployment: hit a target, then serve it

A product has a latency budget, a VRAM budget, and a cost budget, and your job is to compose the optimization levers until all three are met. The latency equation from section 3 tells you which lever moves which term; here is how to actually stack them.

Start from the target. Say the brief is "under 10 seconds per clip on a single 24 GB GPU, under one cent per clip." A vanilla Wan 14B at 50 steps blows all three budgets — it is a minute per clip and needs more than 24 GB to decode. So you stack levers:

1. **Distillation** cuts 50 steps to 4–8. This is the dominant move; it takes the denoiser term from dominant to small. Use a distilled checkpoint if one exists for your model, or a few-step LoRA/student.
2. **Quantization** to fp8 (or int4 for weights) shrinks both the per-step compute and the weight memory, which is often what gets a 14B model under 24 GB at all. `optimum-quanto` and `bitsandbytes` are the usual tools.
3. **Caching** (TeaCache-style feature reuse across steps) shaves another fraction off the effective step count for free, no retraining.
4. **Tiled VAE decode** caps the peak VRAM of stage 5 so the decode — which after distillation may be your *largest* term — fits and does not OOM.

Each lever costs a little quality; the art is spending the smallest amount of quality to hit the budget. The order matters too: distill first (biggest win), then quantize to fit memory, then cache and tile to clean up the remainder.

A note on what quantization actually buys, because it is the lever most often misunderstood. Two distinct things get quantized and they have different effects. Quantizing the *weights* to int4 or int8 shrinks the model's memory footprint — a 14B model in bf16 is roughly 28 GB of weights, in fp8 about 14 GB, in int4 about 7 GB — which is purely a memory win that lets a big model fit on a small card; it does not by itself make each step faster unless the hardware has fast low-bit math. Quantizing the *activations and the compute* to fp8 is the part that actually speeds up $c_\text{denoiser}$, because the matrix multiplies run on fp8 tensor cores at higher throughput. On an H100 or 4090 with fp8 support, weight-plus-activation fp8 gives you both the memory win and a real per-step speedup at a small quality cost; on older hardware without fast low-bit math, int4 weight quantization gives you the memory win but little speedup. Know which one your hardware supports before you assume quantization is a free lunch — the memory win is nearly always available, the speed win is hardware-gated. And test quality after: video is more sensitive to quantization noise than still images because the error is *temporally correlated*, so a quantization that looks fine on a single frame can introduce a subtle shimmer across the clip that only shows up in motion.

```python
import torch
from diffusers import LTXPipeline
from diffusers.utils import export_to_video

# A distilled, few-step model engineered for speed from the VAE up.
pipe = LTXPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()                 # cap decode VRAM (often the new wall)

video = pipe(
    prompt="a hummingbird hovering at a red flower, slow motion, sunlight",
    num_frames=121,
    num_inference_steps=8,               # few-step student, not 50
    guidance_scale=3.0,                  # distilled models like low guidance
).frames[0]
export_to_video(video, "fast.mp4", fps=24)
```

Once you can render one clip inside budget, serving is mostly a systems problem the series' [efficient-serving post](/blog/machine-learning/video-generation/efficient-video-inference-and-serving) covers in depth: a request queue, a warm model pinned in VRAM (cold-loading a 14B model per request is fatal), batching where the model supports it, and autoscaling on queue depth. A minimal serving endpoint looks like this — a FastAPI worker holding a warm pipeline:

```python
from fastapi import FastAPI
from pydantic import BaseModel
import torch, uuid
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

app = FastAPI()

# Load ONCE at startup; never per-request. This is the whole game.
pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    "THUDM/CogVideoX-5b-I2V", torch_dtype=torch.bfloat16
).to("cuda")
pipe.vae.enable_tiling()

class Job(BaseModel):
    prompt: str
    image_url: str
    num_frames: int = 49

@app.post("/generate")
def generate(job: Job):
    frame = load_image(job.image_url)
    with torch.inference_mode():
        video = pipe(
            image=frame, prompt=job.prompt,
            num_frames=job.num_frames, num_inference_steps=30,
        ).frames[0]
    out = f"/tmp/{uuid.uuid4().hex}.mp4"
    export_to_video(video, out, fps=8)
    return {"path": out}
```

The two load-bearing decisions in that endpoint are: the pipeline is loaded *once* at module level (a warm model), and decode is tiled so a burst of large-frame requests cannot OOM the worker. Everything else — auth, the queue, storage, the watermark step we will add in section 8 — wraps around this core. In production you would put a queue (Redis, SQS) in front, run several workers, and autoscale on queue depth, because a single GPU serializes requests and a video render is seconds-to-a-minute long.

The batching question deserves its own paragraph because video breaks the intuition you carry over from serving text or image models. With an LLM you batch aggressively — dozens of requests share a forward pass and throughput scales beautifully. With a video model, a single request already saturates the GPU: those 100,000-plus spacetime tokens fill the memory and the compute units on their own, so there is little headroom to batch a second request alongside it without OOM. The practical consequence is that a video GPU mostly serves requests *serially*, and your throughput is one-over-latency, not the batched throughput you might expect. This changes the serving math: to handle ten concurrent users you need closer to ten GPUs than the one-or-two an LLM team would provision, and your scaling unit is the whole GPU. It also means latency optimization (section 5) and throughput optimization are the *same* problem for video — making one clip faster is the only way to serve more users, since you cannot pack more clips onto the same pass. Teams coming from LLM serving routinely under-provision video fleets by assuming batching will save them; it will not.

For interactive authoring rather than a programmatic API, the other serving surface worth knowing is **ComfyUI** — a node-graph interface where the pipeline stages from figure 1 become draggable nodes (load model, encode, sample, decode, save), and artists wire them together and tune knobs visually without touching Python. It is the standard tool for the open-model community, it supports the same Wan/Hunyuan/CogVideoX/LTX backbones, and it is where a lot of the practical recipe-sharing happens. For a *product* you will usually want a programmatic endpoint like the one above; for *exploration* and for handing the knobs to non-engineers, a ComfyUI graph is often the faster path to a working pipeline.

#### Worked example: hitting a 10-second, 24 GB, sub-cent target

Concretely, on a single RTX 4090 (24 GB), a distilled LTX-class model at 8 steps with fp8 weights, tiled decode, and feature caching renders a 5-second clip in roughly 4–7 seconds of wall time and peaks around 18–20 GB of VRAM — inside the 24 GB budget with headroom for batching. At \$0.50/hr for a 4090 (spot pricing varies), 6 seconds of render is about \$0.0008 per clip — comfortably sub-cent. The same model un-distilled at 50 steps in bf16 without tiling would be roughly 45 seconds, peak past 24 GB (OOM on the decode), and \$0.006 a clip. Same hardware, same model family: the four stacked levers turned an out-of-budget render into an in-budget one and cut the cost nearly eight-fold. That is the deployment game in one comparison.

## 6. The full decision, end to end

Let me assemble the pieces into one walk-through, because in practice you make these decisions in sequence and each one constrains the next.

You start with the brief and pick a **class** (section 1): hosted if quality-and-audio-with-zero-ops, open if control-cost-privacy, distilled if latency-bound. Inside the class you pick a **model** by reading the five-axis matrix (section 2) and crossing out the axis you can lose. You pick **T2V or I2V** — and you pick I2V whenever you can supply a first frame, which is almost always, because it is free quality. You set the **pipeline knobs** (section 3): steps and sampler for the speed/quality trade, `num_frames` for length, guidance for adherence, tiling for VRAM. You decide **customization** (section 4) by climbing the ladder — prompt, then first frame, then camera/motion, and only then a LoRA, and only if twenty test clips prove you need one. You stack **speed levers** (section 5) until you hit your latency/VRAM/cost target — distill, quantize, cache, tile, in that order. Then you handle **length** (section 7) and **evaluation plus safety** (section 8) before a single clip reaches a user.

Notice that every decision is the same trade-off wearing a different hat: coherence times motion times length times cost. Choosing hosted buys coherence and audio with cost. Choosing I2V buys coherence for free. Distilling buys cost-and-latency by spending coherence and motion. A LoRA buys identity-coherence with GPU-hours. Rolling out long video buys length by spending coherence (drift). You are always moving along that four-way trade-off, and the playbook is just a disciplined order in which to move.

The order matters more than any individual choice, and here is why: each decision constrains the next, so a wrong early choice forces a cascade of expensive corrections downstream. Pick the wrong class and every later optimization is fighting the wrong cost structure — you cannot optimize a hosted API's per-clip price, and you cannot make a self-hosted model produce native audio no matter how many levers you pull. Pick T2V when I2V was available and you have given up free quality that you will then try, and fail, to buy back with a LoRA. Skip the camera/motion rung and you will train an adapter to fix a problem an inference-time knob already solved. This is why the playbook is a *sequence*, not a menu: class, then model, then task mode, then knobs, then customization, then speed, then length, then eval-and-safety. Each step is cheap to get right if the steps before it were right, and expensive to fix if they were not. The discipline of the order is most of the value; the individual picks are comparatively forgiving once the order is respected.

## 7. Long video: rollout buys length with coherence

Almost every model trains on five to ten seconds. The moment a brief says "a 30-second clip" or "a one-minute explainer," you have left the comfort zone of every model in the series and entered the land of *rollout*, where you stitch many short generations into one long one — and pay for the length in accumulated drift. The mechanics are the subject of [the long-video post](/blog/machine-learning/video-generation/long-video-and-autoregressive-rollout); here is the builder's version.

There are two rollout strategies. **Chunked / sliding-window**: generate a 5-second chunk, take its last frame (or last few frames) as the first frame of the next chunk via I2V, and continue. **Autoregressive / streaming**: models like CausVid and the Diffusion-Forcing line generate frames causally, conditioning each new segment on the generated past, which is more coherent but needs a model trained for it. Both share one enemy: *error accumulation*. Each chunk inherits the small imperfections of the last, and over many chunks they compound — the character's face drifts off-model, the lighting wanders, the scene slowly forgets itself.

The science of the drift is simple and worth internalizing. If each rollout step introduces a small identity error $\epsilon$ and the conditioning carries a fraction $\rho$ of the previous error forward, the error after $k$ chunks behaves like a geometric sum $\epsilon \cdot \frac{1 - \rho^k}{1 - \rho}$, which for $\rho$ near 1 grows roughly linearly in $k$ and is bounded only when $\rho < 1$ and small. In plain numbers: a model that holds identity beautifully for one 5-second chunk can visibly drift by the sixth chunk of a 30-second rollout, and there is no prompt that fixes it — it is structural. The builder's mitigations: re-inject the *original* reference frame (not just the previous chunk's last frame) periodically to re-anchor identity; keep chunks as long as the VAE allows to minimize the number of seams; and for anything past ~15–20 seconds, budget for either an autoregressive model built for rollout or a human-in-the-loop pass that catches drift. The honest rule: **do not promise minute-long coherent video from a 5-second model.** Promise 5–10 seconds clean, 10–20 with re-anchoring and some drift, and reach for a purpose-built long-video model or hosted API beyond that.

```python
# Chunked rollout: each chunk's last frame seeds the next via I2V.
frames_all = []
seed_frame = load_image("character_ref.png")
ref_frame = seed_frame                      # keep the original to re-anchor

for chunk in range(6):                       # 6 chunks ~ 30s
    out = pipe(image=seed_frame, prompt=prompt,
               num_frames=49, num_inference_steps=30).frames[0]
    frames_all.extend(out)
    seed_frame = out[-1]                      # continue from the last frame
    if chunk % 2 == 1:                        # re-anchor every 2 chunks
        seed_frame = ref_frame                # fights identity drift
```

That `re-anchor every 2 chunks` line is the single most important trick in long-video rollout, and it is the kind of thing you only learn by watching a face slowly stop being the right face at second eighteen.

There is a second-order failure in chunked rollout that re-anchoring alone does not fix: the *seam*. Each chunk transition is a point where the model switches from "continuing real generated motion" to "starting fresh from a handed-in frame," and even when identity is preserved the motion can stutter or reset at the boundary — a walk cycle hitches, a camera pan jumps. The mitigations are to overlap chunks (generate with a few frames of shared context and blend the overlap) and to keep chunks as long as the VAE allows so there are fewer seams to hide. Autoregressive models built for rollout — the CausVid and Diffusion-Forcing line — attack this at the root by training the model to condition each new segment on the generated past rather than on a single handed-in frame, which is why they produce smoother long video than naive chunking. The builder's decision is a length ladder: under 10 seconds, generate in one shot; 10 to 20 seconds, chunk with re-anchoring and accept minor seams; beyond 20 seconds, reach for an autoregressive model or a hosted API that was trained for length, because no amount of seam-blending makes a 5-second model into a coherent one-minute storyteller.

#### Worked example: budgeting drift for a 30-second explainer

A team needs 30-second product explainers from a 5-second open model. They do the arithmetic before they build. Six chunks of 5 seconds, each inheriting roughly a fraction $\rho \approx 0.8$ of the prior chunk's identity error: by the geometric-sum model, the error after six chunks is about $\epsilon \cdot \frac{1 - 0.8^6}{1 - 0.8} \approx 3.7\epsilon$ — nearly four times the single-chunk error, which in practice is the difference between "clearly our mascot" in chunk one and "a cousin of our mascot" in chunk six. They cut $\rho$ by re-anchoring to the original reference every second chunk, which resets the accumulation and keeps the worst-case error closer to $2\epsilon$. The clip is still not as clean as a single 5-second generation, so they set product expectations accordingly: the explainer ships as a sequence of clearly-bounded scenes rather than one continuous shot, which hides the seams *and* the residual drift inside deliberate cuts. The decision that saved the project was arithmetic done up front, not a fix discovered at second eighteen.

## 8. Evaluate and make safe before you ship

You do not ship a video model the way you ship a deterministic service. The output is a sample from a distribution, and "it looked good on my three test prompts" is how teams ship a model that subtly flickers, freezes motion, or drifts off-character at scale. Evaluate properly, and add the safety layer that any generative-video product in 2026 needs.

**Evaluation** has four layers, cheapest to most expensive. First, **FVD** (Fréchet Video Distance) on a fixed sample set against a reference distribution — it is the FID of video, a distance between I3D-feature statistics of generated and real clips, and like FID it is noisy and sensitive to sample size, so use a fixed seed, a fixed prompt set, and enough samples (hundreds) to be stable. The math is worth stating once: FVD models the I3D features of real and generated clips as two Gaussians with means $\mu_r, \mu_g$ and covariances $\Sigma_r, \Sigma_g$, and reports the Fréchet distance $\lVert \mu_r - \mu_g \rVert^2 + \mathrm{tr}\!\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)$ between them. The Gaussian assumption is a lie — feature distributions are not Gaussian — which is exactly why FVD is *relative*, useful for ranking model A against model B on the same eval set with the same sample count, and meaningless as an absolute number you quote out of context. Change the sample count and the number moves; that is not a bug, it is the estimator's variance, and it is why "our FVD is 180" means nothing without "on N samples against reference set X." Second, the **VBench dimensions** that matter for *your* product — subject consistency and background consistency for identity-critical tools, motion smoothness and dynamic degree for action, aesthetic and imaging quality for marketing. Third, a **dynamic-degree sanity check**, because the single most common way to game a benchmark is to make a near-static clip that scores high on consistency while barely moving — a model that "wins" on consistency by not moving is failing your users. This is the deepest trap in video eval: consistency and motion are in tension, so any single-number leaderboard can be climbed by sacrificing the axis it does not measure, and you must read consistency *and* dynamic degree together or be fooled. Fourth, a **human spot-check** on a held-out prompt set, which catches the failures metrics miss — the off-character face, the physically-impossible motion, the uncanny hand — that no I3D feature distance will ever flag. The full treatment is in [the metrics post](/blog/machine-learning/video-generation/the-metrics-of-video-generation) and [the evaluation-and-red-teaming post](/blog/machine-learning/video-generation/evaluating-and-red-teaming-video-generation); the builder's summary is the table below.

| Eval layer | What it catches | Cost | Gotcha |
|---|---|---|---|
| FVD | Overall distribution gap | Medium | Noisy; needs fixed seed + hundreds of samples |
| VBench dims | Per-axis quality (consistency, motion) | Medium | Pick the dims your product needs |
| Dynamic-degree check | Static clips gaming consistency | Cheap | High consistency + low motion = failure |
| Human spot-check | What metrics miss | Expensive | Use a fixed held-out prompt set |

#### Worked example: catching a model that games consistency

A team is choosing between two open checkpoints for an avatar product. Model X posts a VBench subject-consistency of 0.96 and aesthetic 0.62; Model Y posts consistency 0.91 and aesthetic 0.60. On the leaderboard X looks like the obvious pick — higher consistency, higher aesthetic. But the team adds the dynamic-degree check and finds X scores 0.18 and Y scores 0.51. That is the tell: X is "consistent" because its avatars barely move — they nod where they should gesture, they hold a pose where they should turn — and a consistency metric rewards a near-static clip. Y's lower consistency is the honest cost of real motion. In the human spot-check the team confirms it: X's clips are eerily frozen, Y's feel alive. They ship Y, the lower-leaderboard model, because the single number that looked decisive was the single number being gamed. The lesson generalizes — never choose a video model on one VBench dimension; the axis you ignore is the axis a model can cheat on, and dynamic degree against consistency is the cheat that bites avatar and action products hardest.

**Safety** is not optional for a shipped product, and it is three concrete components. A **safety classifier** on prompts and outputs to refuse disallowed content (the obvious categories plus, for video specifically, non-consensual likenesses). A **visible or invisible watermark** so generated clips are identifiable as synthetic. And **C2PA content credentials** — cryptographically signed provenance metadata embedded in the file — which is rapidly becoming the industry standard for "this was AI-generated, here is by what." The image series' [safety, watermarking, and provenance post](/blog/machine-learning/image-generation/safety-watermarking-and-provenance) derives the mechanisms; for video the same machinery applies per-frame plus at the container level. The serving endpoint from section 5 grows two steps — a classifier gate before generation and a watermark-plus-C2PA step after — and neither is optional if real users will touch the tool.

```python
def safe_generate(job):
    if not safety_classifier_allows(job.prompt):     # gate BEFORE compute
        return {"error": "prompt rejected"}
    video = run_pipeline(job)                          # the section-5 core
    video = apply_watermark(video)                     # mark as synthetic
    path = export_to_video(video, ...)
    sign_c2pa(path, generator="our-tool-v1")           # provenance credential
    return {"path": path}
```

The ordering is deliberate: classify *before* you spend GPU time, watermark and sign *after* you have the frames. Skipping the pre-gate wastes compute on prompts you will reject; skipping the post-step ships unattributed synthetic media, which is a legal and reputational liability you do not want.

## 9. Three worked ship-a-product scenarios

Abstract advice is cheap. Here are three concrete products, each routed through the whole playbook, with a recommended stack and a real per-clip cost. These are the three corners of the design space, and most real products are a variation on one of them.

![A matrix mapping four target scenarios, a 4090 self-hosted demo, a marketing-video SaaS, a brand-style pipeline, and a real-time tool, to a recommended model, approach, and per-clip cost ranging from sub-cent to most of a dollar](/imgs/blogs/building-with-video-generation-the-playbook-8.png)

The matrix above is the punchline of the post: four scenarios, four stacks, and a per-clip cost that ranges from a third of a cent to seventy-five cents — a 200x spread driven entirely by the class-and-optimization decisions we have walked through. Let me narrate three of them.

**Scenario A — a 4090 self-hosted image-to-video demo.** A startup wants to show investors a tool that animates product photos, running on a single rented 4090, with no hosted-API dependency. Route: open class (control, no per-call cost), Wan 14B or HunyuanVideo as the backbone, I2V because they have the product photo, fp8 quantization plus tiled decode to fit 24 GB, 30 steps for demo-quality output. Render is roughly 25–40 seconds per 5-second clip, peak VRAM around 20 GB, and at \$0.50–2/hr for a 4090 the cost lands around \$0.03–0.06 per clip. No audio, but the demo does not need it. This is the **control-and-cost** corner.

**Scenario B — a hosted-API marketing-video SaaS.** An agency wants a tool where clients type a product description and get a polished 8-second clip *with music and a voiceover*, and they have no ML team. Route: hosted class (quality and native audio, zero ops), Veo or Sora as the backbone, prompt-plus-optional-frame the only control surface, and the agency builds its value in the UI and the prompt templates, not the model. Cost is \$0.30–0.75 per clip — irrelevant, because the clips are high-value and low-volume and the zero-ops advantage is worth far more than the per-clip premium. This is the **quality-and-audio** corner, and the right call here is emphatically *not* to self-host; the native audio alone justifies the API.

**Scenario C — a custom brand-style LoRA pipeline.** A consumer brand wants every generated clip to carry its exact mascot and visual identity, at the scale of thousands of clips a month, on their own infrastructure for IP control. Route: open class (they must own and fine-tune the model), HunyuanVideo or CogVideoX backbone, I2V from brand stills, *plus* a video-LoRA trained on 40 curated mascot clips (the one scenario where the top customization rung is justified, per section 4's worked example). Cost is \$0.05–0.10 per clip — the LoRA adds a one-time training cost of a few GPU-hours, then inference is just slightly above the bare open-model rate. This is the **identity-control** corner, and it is the only one of the three where a LoRA earns its keep.

Here is the same comparison as a table, because side-by-side is how you actually choose:

| Scenario | Class | Model | Key technique | Audio | Cost / clip |
|---|---|---|---|---|---|
| 4090 self-hosted demo | Open | Wan 14B I2V | fp8 + tiled VAE | No | \$0.03–0.06 |
| Marketing-video SaaS | Hosted | Veo / Sora | Hosted + native audio | Yes | \$0.30–0.75 |
| Brand-style pipeline | Open | Hunyuan + LoRA | I2V + brand video-LoRA | No | \$0.05–0.10 |
| Real-time playground | Distilled | LTX 4-step | Distill + cache + tile | No | \$0.003–0.01 |

The 200x cost spread across these four rows is not noise — it is the direct, predictable consequence of the class decision and the optimization stack. Pick the wrong class for your product and you either overpay by 100x or ship a tool that cannot do what the brief asked. Pick the right one and the model name inside it barely matters.

## 10. Case studies: real numbers from shipped models

To ground the playbook, four real data points from the literature and shipped models that anchor the choices above. Where a figure is approximate I say so; never trust a round number you cannot trace.

**Sora's spacetime-patch scaling (Brooks et al., OpenAI technical report, 2024).** Sora's central claim is that treating video as sequences of spacetime patches and scaling transformer compute produces *emergent* coherence — 3D consistency and object permanence that were not explicitly trained. The report shows the same prompt rendered at increasing training compute getting visibly more coherent. The builder's takeaway: the scaling thesis is why hosted frontier models lead on quality, and why you cannot easily match them by self-hosting a smaller model. The deep version is [the Sora post](/blog/machine-learning/video-generation/sora-and-the-world-simulator-thesis).

**HunyuanVideo's open 13B recipe (Tencent, 2024–2025).** HunyuanVideo shipped a ~13B-parameter open model with a causal 3D-VAE, an MM-DiT denoiser, and flow matching — the converged recipe — and posted VBench scores competitive with closed models at the time. The 1.5 line pushed efficiency further; reports describe rendering relatively long clips on a single consumer 4090-class GPU with the right offloading and tiling. The takeaway: the open frontier is genuinely usable on accessible hardware, which is what makes scenarios A and C above viable.

**Stable Video Diffusion's I2V curation lesson (Blattmann et al., Stability AI, 2023).** SVD demonstrated that image-to-video quality is dominated by *data curation* and the conditioning recipe as much as by architecture — a carefully filtered, captioned video dataset and a clean I2V conditioning path produced strong motion from a frozen-ish image backbone. The takeaway that runs through this whole playbook: I2V plus good first frames plus curated data beats brute-force T2V, which is why the playbook defaults to I2V.

**LTX-Video's real-time push (Lightricks, 2024–2025).** LTX-Video is engineered for speed from a high-compression VAE up, with step-distilled students that render clips in a handful of steps and approach real-time on a single strong GPU. Reported few-step renders bring a clip to seconds rather than a minute, at a quality cost the [efficient-and-real-time post](/blog/machine-learning/video-generation/efficient-and-real-time-video-generation) quantifies honestly. The takeaway: the distilled-and-real-time corner of the matrix is real and shipped, and it is what powers scenario D's sub-cent cost.

**Veo's native-audio bar (Google, 2024–2025).** Veo's defining feature for a builder is not just resolution but *synchronized native audio* — the model generates picture and matched sound (ambient, effects, even speech) jointly rather than as a bolt-on. The [Veo post](/blog/machine-learning/video-generation/veo-and-cinematic-generation) details why joint audio-video is hard and why it currently sits behind a hosted API rather than in the open models. The builder's takeaway is the sharpest line in the model-selection matrix: if your product needs sound that matches the picture, in 2026 that is a hosted-API decision, full stop, because no open model generates it natively yet — you would otherwise be stitching a separate [audio-generation step](/blog/machine-learning/video-generation/audio-and-joint-av-generation) onto a silent clip and fighting lip-sync and timing by hand. This single capability is why scenario B routes to hosted despite the cost premium.

## 11. When to reach for each path — and when not to

A playbook earns its keep by being decisive, so here are the plain rules, including the anti-rules.

- **Reach for a hosted API** when quality and native audio are the product and ops bandwidth is zero, *or* volume is low enough that the per-clip premium does not matter. **Do not** self-host out of pride if a hosted API meets the brief at your volume — you will spend months of engineering to lose on quality.
- **Reach for an open self-hosted model** when you need control, privacy, fine-tuning, or cost-at-scale, and you have (or will build) the GPU-serving competence. **Do not** self-host if your monthly volume is a few hundred clips — the hosted API is cheaper all-in once you price your engineering time.
- **Reach for I2V** whenever you can supply a first frame, which is almost always. **Do not** default to T2V — it is the harder task and you pay for it in consistency on every clip.
- **Reach for distillation** when latency is the binding constraint and you can spend a little quality. **Do not** distill a model that already meets your latency budget — you would be throwing away quality for a speedup you do not need.
- **Reach for a video-LoRA** only when prompt, first frame, and camera/motion control demonstrably fail to hold a specific identity or look, proven by twenty test clips. **Do not** train a LoRA for a *style* the prompt and a color-grade handle for free.
- **Reach for long-video rollout** when the brief truly needs more than ~10 seconds, and budget for drift and re-anchoring. **Do not** promise minute-long coherent video from a 5-second model — re-anchor, use a purpose-built model, or split the brief.
- **Reach for the full eval-and-safety stack** before any real user touches the tool. **Do not** ship on "it looked good on three prompts" — FVD on a fixed set, the VBench dims you care about, a dynamic-degree check, a human spot-check, and a classifier-plus-watermark-plus-C2PA layer are the floor, not extras.

## 12. Key takeaways

- **Pick the class before the model.** Hosted versus open versus distilled decides cost, control, and what you can tune; the model name inside the class is a footnote you can change later.
- **Read all five axes together** — quality, cost, control, length, audio — and pick the model whose weak axis is the one your product can lose. No model wins them all.
- **I2V is free quality.** If you can supply a first frame, do; it removes the appearance-invention task and reliably lifts consistency over T2V at the same compute.
- **Know the latency equation:** $T_\text{clip} = c_\text{enc} + N \cdot c_\text{denoiser} + c_\text{vae}$. Distillation attacks $N$, quantization attacks $c_\text{denoiser}$, tiling caps the decode's VRAM, and after distillation the VAE decode is often your real bottleneck.
- **Climb the customization ladder.** Prompt, then first frame, then camera/motion — all free — and only then a video-LoRA, justified by twenty failing test clips, never for a style a prompt handles.
- **Stack speed levers in order:** distill (biggest win), quantize (to fit memory), cache, tile. Hit the latency/VRAM/cost target with the smallest quality spend.
- **Long video buys length with coherence.** Drift accumulates roughly linearly across chunks; re-anchor to the original reference frame and do not promise minute-long coherence from a 5-second model.
- **Evaluate and make safe before shipping.** FVD on a fixed set, the right VBench dims, a dynamic-degree sanity check, a human spot-check, and a classifier-plus-watermark-plus-C2PA layer are the floor.
- **The whole series is one trade-off:** coherence times motion times length times cost. Every decision in this playbook moves you along it; the skill is knowing which one your product actually needs.

## Further reading

- Brooks et al., **"Video generation models as world simulators"** (Sora technical report), OpenAI, 2024 — the spacetime-patch scaling thesis behind frontier quality.
- Blattmann et al., **"Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets"**, Stability AI, 2023 — the I2V curation and conditioning lesson.
- Ho et al., **"Video Diffusion Models"**, 2022, and the **Make-A-Video** / **Imagen Video** line — the origins of adding the time axis.
- Peebles & Xie, **"Scalable Diffusion Models with Transformers (DiT)"**, 2023, and Lipman et al., **"Flow Matching for Generative Modeling"**, 2023 — the denoiser and the sampler the video stack is built on.
- The **CogVideoX**, **HunyuanVideo**, **Wan**, and **Movie Gen** technical reports — the converged open recipe and its measured VBench/FVD numbers.
- The 🤗 `diffusers` **video pipelines documentation** — the real APIs (`CogVideoXImageToVideoPipeline`, `LTXPipeline`, `enable_vae_tiling`, `export_to_video`) used throughout this post.
- Within this series: start at [why video generation is hard](/blog/machine-learning/video-generation/why-video-generation-is-hard); the architecture in [video diffusion transformers](/blog/machine-learning/video-generation/video-diffusion-transformers), [the latent video stack with SVD and AnimateDiff](/blog/machine-learning/video-generation/latent-video-diffusion-svd-and-animatediff), and [from image diffusion to video diffusion](/blog/machine-learning/video-generation/from-image-diffusion-to-video-diffusion); and the image-series capstone, [building an image generation stack](/blog/machine-learning/image-generation/building-an-image-generation-stack), whose front-end feeds your I2V pipeline.
