---
title: "The Open Video Frontier: Wan, HunyuanVideo, and CogVideoX"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "The video models you can actually download all converged on one recipe — a causal 3D-VAE, an MM-DiT denoiser, flow matching, and an LLM text encoder — and this post shows you exactly how CogVideoX, HunyuanVideo, and Wan instantiate it and how to run a 13B model on a single 24GB card."
tags:
  [
    "video-generation",
    "diffusion-models",
    "video-diffusion",
    "text-to-video",
    "open-source",
    "cogvideox",
    "hunyuanvideo",
    "wan",
    "generative-ai",
    "deep-learning",
    "diffusers",
  ]
category: "machine-learning"
subcategory: "Video Generation"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/the-open-video-frontier-wan-hunyuanvideo-cogvideox-1.png"
---

In the summer of 2024 you could not download a serious video model. You could marvel at Sora demos on a launch page, you could pay for a closed API that returned a four-second clip and a watermark, and you could run ModelScope or Zeroscope — open, yes, but a 1.7-billion-parameter U-Net that produced two seconds of 256-pixel mush where a dog's legs swapped places between frames and the background boiled like a heat haze. The gap between what was published and what was downloadable was a chasm. Then, in the span of about eighteen months, it closed. By the start of 2026 you can `git clone` a 13-billion-parameter denoiser, pull its 3D-VAE and its text encoder from the same hub, and render a coherent five-second 720p clip on a single consumer GPU — and the result sits within a VBench point or two of a flagship you cannot download at any price.

The remarkable thing is not that this happened. It is that three different labs — Zhipu AI with CogVideoX, Tencent with HunyuanVideo, Alibaba with Wan — got there by building *the same model*. Not literally the same weights, but the same architecture, the same training objective, the same four-stage stack, down to the compression ratios and the choice of text encoder. They converged. If you learn how one of them works under the hood, you have learned all three, and you have learned the template that every open video model released after them will almost certainly follow. That convergence is the subject of this post.

![Stacked diagram of the converged open video recipe running from curated data through a causal three dimensional VAE then spacetime tokens then an MM-DiT denoiser then flow matching and an LLM text encoder out to a five second clip](/imgs/blogs/the-open-video-frontier-wan-hunyuanvideo-cogvideox-1.png)

By the end you will be able to do five concrete things. First, you will be able to *name the four layers of the converged recipe* — a causal 3D-VAE, an (MM-)DiT denoiser on spacetime tokens, flow-matching training, and a strong LLM-class text encoder — and say precisely where CogVideoX, HunyuanVideo, and Wan differ inside that shared skeleton. Second, you will be able to *do the VRAM math*: given a model's parameter count, its VAE's compression ratio, and a target resolution and length, compute roughly how much GPU memory the denoiser and the VAE decode will actually need, and decide whether it fits. Third, you will be able to *call CogVideoX in 🤗 `diffusers`* with the right offload, tiling, and decode-chunking flags to make a 13B-class model fit on a 24GB card. Fourth, you will be able to *sketch loading a quantized HunyuanVideo or Wan checkpoint* and reason about what each flag buys you. Fifth, you will be able to *pick the right open model* for your hardware and your goal, and to fine-tune it with a video LoRA on a handful of clips.

This post sits late in the [video generation series](/blog/machine-learning/video-generation/why-video-generation-is-hard). It assumes you have met the pieces individually: the [causal 3D-VAE](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression) that does the temporal compression, the [video diffusion transformer](/blog/machine-learning/video-generation/video-diffusion-transformers) that denoises the spacetime tokens, [flow matching for video](/blog/machine-learning/video-generation/flow-matching-for-video) that trains it, and the [conditioning machinery](/blog/machine-learning/video-generation/conditioning-video-text-image-motion-camera) that lets you steer it with text, an image, motion, or a camera path. What is new here is seeing all four assembled into shipped, downloadable systems, and confronting the one thing the papers gloss over: the brutal practical reality of running them. The spine of the series holds throughout — video is spatial generation times temporal coherence under a brutal compute budget — and in this post the budget is *your* GPU, not a training cluster.

## 1. The converged recipe, stated plainly

Let me state the shared recipe in one breath, then spend the rest of the post unpacking it. An open video model in 2026 is: a **causal 3D-VAE** that compresses a clip by roughly $4\times$ in time and $8\times \times 8\times$ in space into a 16-channel latent; a **diffusion transformer** — usually a multimodal DiT, or MM-DiT, that attends jointly over video tokens and text tokens — that denoises the spacetime tokens of that latent; trained with a **flow-matching** objective that regresses a straight-line velocity field instead of the older noise-prediction loss; conditioned by a **strong text encoder**, almost always an LLM-class model (T5-XXL, a decoder-only MLLM, or a multilingual UMT5) rather than the small CLIP text encoder that image models used to use. Data → 3D-VAE → MM-DiT → flow matching → text encoder → frames. That is the whole thing.

What makes this a *recipe* rather than a list is that the four pieces are not independent. They are coupled by a single resource: the token budget. The 3D-VAE's compression ratio decides how many spacetime tokens a clip becomes. The token count decides the DiT's attention FLOPs, which scale quadratically. The DiT's size and the token count together decide your VRAM and your seconds-per-clip. And the text encoder's quality decides how much of your prompt actually survives into the generation. Push any one dial and the others move. A more aggressive VAE (Wan-VAE compresses $16\times$ spatially instead of $8\times$) shrinks the token count by $4\times$ and the attention bill by $16\times$, which is what lets Wan run longer clips at a given VRAM — but it asks more of the VAE, which now has to reconstruct fine detail from a coarser latent. The recipe is a set of coupled trade-offs, and each lab tuned the dials differently.

Here is the converged stack as a single figure. Read it top to bottom: this is the pipeline that CogVideoX, HunyuanVideo, and Wan all instantiate, and the rest of the post walks each layer and notes where the three models diverge.

The reason this convergence is worth a whole post — rather than a paragraph saying "they're all DiTs" — is that the *details* of the convergence are exactly the knobs you turn when you self-host. You do not get to choose the architecture; the lab chose it. You get to choose the VAE tiling, the offload strategy, the quantization, the number of sampling steps, and the resolution-length trade. Every one of those choices is constrained by the recipe above. So we are going to be specific. Where I give a parameter count or a VRAM figure, I will name the model and the GPU; where I am estimating, I will say so. The numbers in this post are drawn from the CogVideoX, HunyuanVideo, and Wan technical reports and model cards, plus the 🤗 `diffusers` documentation, and where a figure is a defensible order-of-magnitude rather than a reported value I mark it approximate. Never trust a video-model VRAM number that does not name the resolution, the length, and the dtype — the same model can need 12GB or 60GB depending on those three.

### The one law that ties the four layers together

Before we walk the layers, let me write down the single quantitative relationship that makes the recipe a coupled system rather than four independent boxes, because every practical decision in the rest of the post is a consequence of it. Start from the clip you want: $T$ frames at $H \times W$ pixels. The VAE compresses by $(c_t, c_h, c_w)$ — temporal and the two spatial ratios — to a latent of $T/c_t \times H/c_h \times W/c_w$ cells with $C$ channels. The DiT patchifies that latent by $(p_t, p_h, p_w)$, so the token sequence length is

$$
L = \frac{T}{c_t\, p_t} \cdot \frac{H}{c_h\, p_h} \cdot \frac{W}{c_w\, p_w}.
$$

Self-attention over that sequence costs $O(L^2 d)$ per layer, and the DiT has $N$ layers each with width $d$, so the dominant compute per denoising step scales as $N\, d\, L^2$ for attention plus $N\, d^2 L$ for the MLPs. For a video DiT the $L^2$ term dominates because $L$ is enormous, so to first order **the per-step cost is governed by $L^2$, and $L$ is set entirely by the VAE compression and the patch size**. Now you can see why the VAE is the master lever: doubling the temporal compression $c_t$ halves $L$, which *quarters* the attention bill. Wan's choice of $c_h = c_w = 16$ instead of $8$ is a $16\times$ cut in $L^2$ for the same clip. Nothing else in the stack moves the needle that hard, which is exactly why the open labs spent their hardest engineering on the autoencoder, not the denoiser.

This same law is why "longer clip" is so much more expensive than "bigger model." Doubling $T$ doubles $L$ and *quadruples* the attention cost; doubling the denoiser's width $d$ only doubles the per-step cost. Length is the quadratic axis, model size is linear, and that asymmetry organizes every trade-off in the rest of this post.

## 2. The causal 3D-VAE: the lever for length and cost

Start with the layer that matters most and gets the least attention in casual write-ups: the autoencoder. We covered the [theory of spatiotemporal compression](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression) in its own post; here I want the *numbers* that the three open models actually shipped, because they directly set everything downstream.

A causal 3D-VAE takes a clip of shape $T \times H \times W \times 3$ — frames, height, width, RGB — and encodes it to a latent of shape $T' \times H' \times W' \times C$. "Causal" means the temporal convolutions only look backward in time, so the VAE can encode frame $t$ without seeing the future; this is what lets it handle the first frame specially and stream arbitrary lengths. The compression is the ratio of input voxels to latent voxels. CogVideoX and HunyuanVideo both use $4 \times 8 \times 8$ with $C = 16$: four frames collapse to one latent frame, an $8\times8$ pixel block collapses to one spatial latent cell. So a clip compresses by

$$
\frac{T \cdot H \cdot W \cdot 3}{T' \cdot H' \cdot W' \cdot C} = \frac{4 \cdot 8 \cdot 8 \cdot 3}{1 \cdot 1 \cdot 1 \cdot 16} = \frac{768}{16} = 48\times
$$

in raw voxel count. Wan's VAE (Wan-VAE) pushes the spatial ratio to $16\times$ in some configurations, giving $4 \times 16 \times 16$, which is a $4\times$ smaller spatial latent and therefore a much smaller token sequence — at the cost of asking the decoder to hallucinate more high-frequency detail back. This is the single most consequential design choice in the whole stack, and it is why the VAE ratio gets its own column in every comparison table in this post.

Why does the VAE, and not the giant denoiser, decide your maximum clip length? Because of the causal-frame arithmetic. A $4\times$ temporal VAE maps $T$ pixel frames to roughly $T' = T/4 + 1$ latent frames (the $+1$ is the special causal first frame). The denoiser then runs over $T'$ latent frames. If you want a longer clip you need more latent frames, which means more tokens, which means quadratically more attention. But there is a second, sneakier wall: the VAE itself was *trained* on clips of a certain latent length, and if you ask it to decode far past that it degrades or OOMs. The decode is the part most people forget to budget for, and it is frequently the actual VRAM ceiling.

#### Worked example: token count for a 5-second 720p clip on CogVideoX

Take the series' running example — a five-second clip at 720p, 24 fps, so 120 frames at $1280 \times 720$. Through CogVideoX's $4 \times 8 \times 8$ VAE:

$$
T' = \frac{120}{4} + 1 \approx 31, \qquad H' = \frac{720}{8} = 90, \qquad W' = \frac{1280}{8} = 160, \qquad C = 16.
$$

The latent is about $31 \times 90 \times 160 \times 16$. CogVideoX patchifies space by $2\times2$ but not time, so the token sequence length the DiT sees is

$$
L = T' \cdot \frac{H'}{2} \cdot \frac{W'}{2} = 31 \cdot 45 \cdot 80 \approx 111{,}600 \text{ tokens}.
$$

Now run the same clip through Wan-VAE's $4 \times 16 \times 16$: $H' = 720/16 = 45$, $W' = 1280/16 = 80$, and with a $2\times2$ patch the sequence is $31 \cdot 22 \cdot 40 \approx 27{,}300$ tokens — roughly a quarter of CogVideoX's, and the attention bill is $4^2 = 16\times$ smaller. That single VAE choice is why Wan can render longer at a given VRAM and why CogVideoX's higher-resolution latent can look slightly crisper at short lengths. The VAE is the lever. You can see all four models' choices side by side here.

![Matrix comparing CogVideoX HunyuanVideo HunyuanVideo 1.5 and Wan across denoiser size VAE compression ratio text encoder and minimum VRAM](/imgs/blogs/the-open-video-frontier-wan-hunyuanvideo-cogvideox-3.png)

You can feel the VAE's compression directly in a few lines of `diffusers`, and it is worth running once because the numbers make the rest of the post concrete. Load CogVideoX's VAE, encode a clip, and print the shapes:

```python
import torch
from diffusers import AutoencoderKLCogVideoX

vae = AutoencoderKLCogVideoX.from_pretrained(
    "THUDM/CogVideoX-5b", subfolder="vae", torch_dtype=torch.bfloat16
).to("cuda")
vae.enable_tiling()  # decode/encode in tiles so a 720p clip fits

# A fake clip: 1 batch, 3 channels, 49 frames, 480 x 720.
clip = torch.randn(1, 3, 49, 480, 720, dtype=torch.bfloat16, device="cuda")

with torch.no_grad():
    latent = vae.encode(clip).latent_dist.sample()
print("pixels :", clip.shape)     # [1, 3, 49, 480, 720]
print("latent :", latent.shape)   # ~[1, 16, 13, 60, 90]  (4x8x8 + causal frame)

# Compression ratio in raw element count:
ratio = clip.numel() / latent.numel()
print(f"compression: {ratio:.1f}x")   # ~48x in voxels
```

The `latent.shape` is the single most clarifying thing to see with your own eyes: 49 pixel frames become ~13 latent frames (the $4\times$ temporal compression plus the causal first frame), and $480 \times 720$ becomes $60 \times 90$. That ~13-frame latent is what the denoiser actually operates on — the denoiser never sees 49 frames, it sees 13 latent frames, and *that* is why the architecture is tractable. Run this and the whole "the VAE is the lever" argument stops being abstract.

One more property worth internalizing: the VAE is *frozen at inference and used twice*. It encodes your conditioning (for I2V, the input image; for T2V, just to set up the latent grid and noise) and it decodes the final latent to pixels. Everything in between — the 30 to 50 denoising steps — happens entirely in latent space, never touching pixels. This is the whole reason the architecture is tractable: you pay the pixel-space cost exactly twice, not once per step. The figure below shows that bracketing structure, with the frozen text encoder and the frozen VAE on the outside and the trained denoiser in the middle.

![Graph of the shared open pipeline showing a frozen text encoder and a three dimensional VAE both feeding one trained MM-DiT denoiser whose flow matching output the same VAE decodes to frames](/imgs/blogs/the-open-video-frontier-wan-hunyuanvideo-cogvideox-2.png)

## 3. The MM-DiT denoiser: where the three models diverge most

If the VAE is where the recipe is most shared, the denoiser is where it diverges most — not in kind, but in the details of how text and video tokens are fused. All three are diffusion transformers over spacetime tokens, the architecture from the [video-DiT post](/blog/machine-learning/video-generation/video-diffusion-transformers). The interesting question is: how does the text get in?

The image-generation world settled this with [MM-DiT, the modern text-to-image recipe](/blog/machine-learning/image-generation/mmdit-and-the-modern-text-to-image-recipe), where text tokens and image tokens are concatenated into one sequence and attend to each other jointly, with separate weight streams (separate QKV projections, separate MLPs) for each modality but a single shared attention. Video inherited this directly. But the three open models implement the fusion at three different points on a spectrum from "fully joint from the start" to "specialize first, fuse late."

**CogVideoX** uses what its paper calls an *expert-adaptive-LayerNorm* MM-DiT. Text and video tokens go into the same transformer blocks and attend jointly, but each modality gets its own AdaLN "expert" — its own learned scale-and-shift modulation conditioned on the timestep. The intuition is that text tokens and video tokens have very different statistics, so forcing them through one shared LayerNorm conditioning is wasteful; giving each its own expert lets the conditioning specialize. It is a lightweight way to get modality-specific behavior without doubling the whole network. CogVideoX-2B and CogVideoX-5B differ mainly in width and depth of these blocks.

**HunyuanVideo** uses a *dual-stream then single-stream* design, which is the more elaborate end of the spectrum and worth a figure. In the early blocks (dual-stream), video tokens and text tokens are processed in *fully separate* transformer streams — separate attention, separate MLP, no cross-talk — so each modality refines its own representation first. Then the two token sequences are *concatenated* into one long sequence and passed through single-stream blocks where a single full-attention operation fuses everything. The argument is that early specialization plus late fusion gets you the best of both: the text stream builds a rich prompt representation before the video tokens have to attend to it, and the late single-stream blocks do the heavy cross-modal binding. HunyuanVideo is roughly 13 billion parameters, the largest of the three, and this stream design is part of why.

![Graph of HunyuanVideo stream design showing video tokens and text tokens refined in separate dual stream blocks then concatenated into one sequence for single stream blocks with full joint attention](/imgs/blogs/the-open-video-frontier-wan-hunyuanvideo-cogvideox-5.png)

**Wan** sits in the middle and is the most aggressively engineered for scale. Wan 2.1 is a fairly standard MM-DiT; Wan 2.2 introduces a mixture-of-experts denoiser (the "A14B" naming means roughly 14B total parameters but only a subset — two experts — active per token), which lets it grow capacity without growing the per-step FLOPs proportionally. Wan also leans on umT5, a multilingual T5 variant, as its text encoder, which is part of why it handles non-English prompts better than its peers. Across the Wan 2.x line the constant is a strong VAE and a focus on I2V quality and first/last-frame control, which we will come back to.

Why did all three abandon the old *cross-attention* design, where text was injected through separate cross-attention layers (as in the original Stable Diffusion U-Net) rather than concatenated into the sequence? Because joint attention is strictly more expressive for the binding problem that video makes worse. In cross-attention, video tokens query text but text never updates from video — the information flows one way. In MM-DiT's joint attention, a text token like "left to right" can attend to the *actual spatial layout* of the video tokens and a video token can attend back, so the model can bind "the corgi" to a specific moving region across frames rather than re-deciding what "the corgi" looks like every frame. That two-way binding is exactly what long, compositional prompts need, and it is exactly what fails when identity drifts in a long clip. The cost is that the text tokens now live in the attention sequence and add to $L$ — but text is a few hundred tokens against a hundred thousand video tokens, so the overhead is negligible. The open models paid a tiny token cost for a large gain in prompt adherence, and that trade is the whole reason the field converged on MM-DiT.

The takeaway is not that one stream design is "best." It is that *all three are the same architecture* — a transformer over concatenated or jointly-attended spacetime-and-text tokens — and the differences are tuning choices on a known spectrum. If you have understood MM-DiT once, you can read any of these three papers in an afternoon and know exactly what you are looking at.

### Full 3D versus factorized attention inside the block

There is one more axis of divergence worth a paragraph, because it is where the per-step FLOPs actually go and where the [spatiotemporal attention post](/blog/machine-learning/video-generation/spatiotemporal-attention-patterns) lives in detail. Inside each MM-DiT block, the attention over the $L \approx 10^5$ video tokens can be *full 3D* — every token attends to every other token across all of space and time — or *factorized*, where the model alternates a spatial attention (within a frame, over $H'W'/p^2$ tokens) and a temporal attention (across frames, over $T'/p_t$ tokens at each spatial position). Full 3D costs $O(L^2)$; factorized costs $O\!\left(\frac{T'}{p_t}\left(\frac{H'W'}{p_h p_w}\right)^2 + \frac{H'W'}{p_h p_w}\left(\frac{T'}{p_t}\right)^2\right)$, which for a clip that is much wider in space than deep in time is dramatically cheaper — often by an order of magnitude. The cost is coherence: factorized attention cannot directly relate a token in the top-left at frame 3 to a token in the bottom-right at frame 30 in one hop; it has to route that relationship through the spatial-then-temporal alternation, which weakens long-range spacetime binding. The open models lean toward full 3D attention (within the budget the VAE compression leaves them) precisely because coherence is the thing video is hardest at, and they buy back the FLOPs with the aggressive VAE compression rather than by factorizing the attention. This is the recipe's internal logic: compress hard in the VAE so you can afford full attention in the DiT.

### The flow-matching objective they share

All three train with flow matching rather than the older $\epsilon$-prediction DDPM loss. We derived [flow matching for video](/blog/machine-learning/video-generation/flow-matching-for-video) in its own post, so I will state the objective briefly and link out for the full treatment. The model learns a velocity field $v_\theta(x_t, t, c)$ that, integrated from noise to data, transports a Gaussian sample to a clean latent. With the simplest linear (rectified-flow) interpolation between data $x_1$ and noise $x_0$,

$$
x_t = (1 - t)\, x_0 + t\, x_1, \qquad \frac{dx_t}{dt} = x_1 - x_0,
$$

and the training loss regresses the model's predicted velocity onto that constant target:

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, x_0, x_1, c}\left[\; \big\| v_\theta(x_t, t, c) - (x_1 - x_0) \big\|^2 \;\right].
$$

The single sentence on *why* this matters for video, rather than re-deriving it: flow matching's target is a *straight line* in latent space, so the learned ODE has gentle, nearly-constant curvature, and a coarse ODE solver — 30 to 50 steps, sometimes fewer — integrates it accurately. In the video regime, where each step is enormously expensive because the sequence is 100k+ tokens, halving the step count is worth more than it is for images. That is the practical reason the open models converged on flow matching: it is the objective that makes few-step sampling actually work at video scale. The high-resolution models also apply a *timestep shift* — they spend more sampling steps near the high-noise end where the global structure of the clip is decided — which is the video analog of the resolution-dependent shift the image flow-matching models use.

## 4. The text encoder: why an LLM, not CLIP

The least-discussed convergence is the text encoder, and it is the one that most changes what you can actually prompt. Early video models (and early image models) used the CLIP text encoder — a relatively small, ~123M-parameter transformer trained on image-caption contrastive learning. CLIP is fine for "a dog running on a beach" but falls apart on long, compositional prompts: "a corgi running left to right across a beach at sunset, the camera tracking it from the side, waves breaking behind, then the corgi stops and shakes off water." CLIP's representation simply does not have the capacity to keep all those clauses bound to the right objects and actions over a 100k-token video sequence.

So the open models all moved to LLM-class text encoders, and this is a genuine convergence with three flavors:

- **CogVideoX** uses **T5-XXL** — the ~4.8B-parameter encoder of the T5 text-to-text model, frozen. T5-XXL has been the workhorse text encoder of high-end image and video diffusion for years; it produces rich, long-context token embeddings that the MM-DiT can attend over.
- **HunyuanVideo** uses a **decoder-only MLLM** (a multimodal large language model) as its text encoder, taking the hidden states of an instruction-tuned LLM rather than a T5 encoder. The argument is that an instruction-tuned LLM has a far better grasp of compositional, instruction-like prompts, and its hidden states carry that understanding into the video model.
- **Wan** uses **umT5**, a multilingual T5, which is why it handles Chinese and other non-English prompts natively rather than relying on translation.

The practical consequence for you, the operator, is large and underappreciated: **the text encoder is often the second-largest weight you have to load**, after the denoiser. T5-XXL is ~5B parameters; an MLLM encoder can be 7B+. On a 24GB card, loading a 13B denoiser *and* a 5B text encoder *and* the VAE at full precision simply will not fit — and the reason `enable_model_cpu_offload()` is the single most important flag in the whole pipeline is that it lets you run the text encoder, move its output to where you need it, *evict it to CPU*, then load the denoiser. The text encoder runs once at the start; it has no business sitting in VRAM during the 50 denoising steps. We will see exactly that pattern in the code.

## 5. Running CogVideoX on a 24GB card: the real code

Enough architecture. Here is the actual code to generate a clip with CogVideoX-5B in 🤗 `diffusers`, with every flag you need to fit it on a 24GB consumer GPU (an RTX 3090 or 4090). This is the pipeline I reach for first when I want a quick, reliable open T2V baseline, because CogVideoX is the lightest of the three and the most thoroughly integrated into `diffusers`.

```python
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

# Load in bf16. The 5B model's weights are ~10GB at bf16; we will offload them.
pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b",
    torch_dtype=torch.bfloat16,
)

# The three flags that make a 13B-class model fit on 24GB.
# 1) Offload whole modules (text encoder, transformer, VAE) to CPU and
#    pull each onto the GPU only while it runs. The text encoder runs once
#    at the start, then is evicted before the 50 denoising steps begin.
pipe.enable_model_cpu_offload()
# 2) Tile the VAE: decode the latent in spatial tiles instead of all at once,
#    which is what keeps the VAE decode (often the real VRAM wall) from OOMing.
pipe.vae.enable_tiling()
# 3) Slice the VAE: process one frame (or a small group) at a time on top of tiling.
pipe.vae.enable_slicing()

prompt = (
    "A corgi runs left to right across a sunlit beach, the camera tracking "
    "from the side, waves breaking behind it, photorealistic, smooth motion."
)

video = pipe(
    prompt=prompt,
    num_frames=49,            # CogVideoX-5B: 49 frames ~= 6s at 8 fps latent
    num_inference_steps=50,   # flow-matching / DDIM steps
    guidance_scale=6.0,       # CFG; 6 is a good default for CogVideoX
    generator=torch.Generator(device="cuda").manual_seed(42),
).frames[0]

export_to_video(video, "corgi_beach.mp4", fps=8)
```

A few things to notice, because they are the whole point. `enable_model_cpu_offload()` is not the same as `enable_sequential_cpu_offload()` — the former moves whole modules (text encoder, transformer, VAE) on and off the GPU as a unit, which is fast; the latter moves individual *submodules* (down to layers) and is far slower but cuts VRAM even harder, for when even module-level offload OOMs. `vae.enable_tiling()` is the flag people forget, and it is frequently the difference between a successful render and an OOM at the very last step, because the VAE decode of a full 720p clip can momentarily need more VRAM than all 50 denoising steps combined. The `guidance_scale` here is classifier-free guidance, the same mechanism as in [image diffusion](/blog/machine-learning/image-generation/classifier-free-guidance) — for video, values around 6 are typical, and pushing it higher tends to oversaturate and *reduce* motion, a video-specific failure mode worth remembering.

### Image-to-video with CogVideoX

If you can supply a first frame — and you usually can, because generating one good image is far easier than generating one good clip — image-to-video beats text-to-video on coherence almost every time. You are giving the model a concrete anchor for identity, composition, and color, so it only has to solve the temporal problem, not the spatial one too. Here is the I2V pipeline:

```python
import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    "THUDM/CogVideoX-5b-I2V",
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

# The first frame is the anchor. Generate it however you like (an image model,
# a photo, a frame grab) and pass it in.
image = load_image("first_frame.png")

prompt = "The corgi shakes off water, droplets flying, slow motion, cinematic."

video = pipe(
    image=image,
    prompt=prompt,
    num_frames=49,
    num_inference_steps=50,
    guidance_scale=6.0,
    generator=torch.Generator(device="cuda").manual_seed(7),
).frames[0]

export_to_video(video, "corgi_shake.mp4", fps=8)
```

The only structural difference from T2V is the `image=` argument; under the hood the pipeline encodes that image with the same 3D-VAE, places it as the conditioning latent for the first frame, and the denoiser generates the rest conditioned on it. This is exactly the conditioning machinery from the [conditioning post](/blog/machine-learning/video-generation/conditioning-video-text-image-motion-camera), now in a single shipped model. In practice I2V is where the open models are most useful: a designer can art-direct the first frame precisely, then let the model animate it.

## 6. The VRAM math, made explicit

Let me make the memory budget concrete, because "it needs 24GB" is useless without a breakdown, and the breakdown is what tells you which flag to reach for when you OOM. For a forward pass of a video DiT plus its VAE, peak VRAM is roughly the sum of four terms:

$$
\text{VRAM} \approx \underbrace{P \cdot b_w}_{\text{weights}} \;+\; \underbrace{E \cdot b_w}_{\text{text encoder}} \;+\; \underbrace{A(L, d)}_{\text{activations}} \;+\; \underbrace{V_{\text{dec}}(T, H, W)}_{\text{VAE decode peak}}
$$

where $P$ is the denoiser parameter count, $E$ the text-encoder parameter count, $b_w$ the bytes per weight (2 for fp16/bf16, 1 for int8, 0.5 for int4), $A$ the activation memory (grows with sequence length $L$ and width $d$), and $V_{\text{dec}}$ the peak memory of the VAE decode (grows with the pixel-space clip size $T \times H \times W$).

Plug in HunyuanVideo's 13B at fp16 with a ~5B-class text encoder for a 720p clip:

$$
\text{VRAM} \approx \underbrace{13 \times 2}_{26\,\text{GB}} \;+\; \underbrace{5 \times 2}_{10\,\text{GB}} \;+\; \underbrace{\sim 6}_{\text{activations}} \;+\; \underbrace{\sim 18}_{\text{VAE decode}} \;\approx\; 60\,\text{GB (naive)}.
$$

Sixty gigabytes. That is why a naive `from_pretrained(...).to("cuda")` of HunyuanVideo OOMs on everything short of an A100 80GB or an H100. Now apply the three levers in order. **Offload** the text encoder: it runs once, then leaves the GPU, dropping its 10GB to roughly zero during the denoising steps. **Quantize** the denoiser to int8: $13 \times 1 = 13$GB instead of 26. **Tile and chunk** the VAE decode: instead of decoding the whole $T \times H \times W$ clip at once, decode spatial tiles and temporal chunks, dropping the ~18GB decode peak to ~4GB. The result:

$$
\text{VRAM} \approx \underbrace{13 \times 1}_{13\,\text{GB int8}} \;+\; \underbrace{\sim 0}_{\text{encoder offloaded}} \;+\; \underbrace{\sim 3}_{\text{activations}} \;+\; \underbrace{\sim 4}_{\text{tiled decode}} \;\approx\; 20\,\text{GB}.
$$

Now it fits on a 24GB card. The figure below lays out this same budget term by term, with the lever that saves each one.

![Matrix showing how a 13 billion parameter video model is brought from about forty five gigabytes to under twenty four gigabytes by quantization offload and VAE tiling across weights text encoder activations and decode](/imgs/blogs/the-open-video-frontier-wan-hunyuanvideo-cogvideox-7.png)

The single most important thing this math teaches is *which term is your wall*. New users assume the 13B weights are the problem, so they quantize aggressively and are confused when they still OOM. The wall is very often the **VAE decode**, because it happens in pixel space — the one place the latent compression does not protect you — and it spikes at the very end of the render after all the denoising has succeeded. There is nothing more demoralizing than an OOM at second six, after eight minutes of denoising, because you forgot `vae.enable_tiling()`. Budget the decode first.

#### Worked example: choosing resolution to fit a fixed VRAM ceiling

Suppose you have exactly 24GB and you want the longest clip you can render with Wan 2.2 at int8. The denoiser and offloaded encoder leave you roughly $24 - 13 - 1 = 10$GB for activations plus VAE decode. The activation and decode terms both scale with the pixel-space clip size $T \times H \times W$. Wan-VAE's $16\times$ spatial compression means its latent — and therefore its activations — are about $4\times$ smaller than CogVideoX's at the same resolution, so you have headroom. Concretely: at 480p ($832 \times 480$) you might fit ~5 seconds comfortably; at 720p the decode peak roughly triples and you may be limited to ~3 seconds or need to drop to fp8 weights and tighter tiling. The actionable rule: **at a fixed VRAM ceiling, dropping resolution buys length faster than dropping the model size**, because both activations and decode scale with pixel area, while weights are a fixed cost you pay regardless. When you OOM, lower the resolution before you reach for a smaller model.

## 7. Loading a quantized HunyuanVideo or Wan checkpoint

CogVideoX is well-integrated and easy. The bigger models — HunyuanVideo 13B, Wan 14B — usually need quantization to fit on a single card, and `diffusers` supports loading them through quantization backends. Here is the shape of loading HunyuanVideo with int8 weights via `optimum-quanto`, the pattern that brings the 13B denoiser from 26GB to ~13GB:

```python
import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers import QuantoConfig
from diffusers.utils import export_to_video

model_id = "hunyuanvideo-community/HunyuanVideo"

# Quantize the transformer (the 13B denoiser) to int8 on load.
# This halves its VRAM footprint vs bf16 with minimal quality loss.
quant_config = QuantoConfig(weights_dtype="int8")
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=quant_config,
    torch_dtype=torch.bfloat16,
)

pipe = HunyuanVideoPipeline.from_pretrained(
    model_id,
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()   # evict the MLLM text encoder after it runs
pipe.vae.enable_tiling()          # tile the decode — the real VRAM wall

output = pipe(
    prompt="A cat wearing sunglasses drives a convertible down a coastal road, cinematic.",
    height=720,
    width=1280,
    num_frames=61,
    num_inference_steps=30,        # HunyuanVideo needs fewer steps than older models
    guidance_scale=6.0,
).frames[0]

export_to_video(output, "cat_drive.mp4", fps=15)
```

The structure is identical to CogVideoX — load, offload, tile, call — with two differences worth flagging. First, you quantize the *transformer specifically* rather than the whole pipeline, because the VAE and text encoder are smaller and quantizing them buys little while risking decode-quality loss. Second, HunyuanVideo runs well at fewer steps (~30) than CogVideoX (~50), a direct benefit of its flow-matching training and timestep shift. For Wan the pattern is the same — `WanPipeline.from_pretrained(...)`, quantize the transformer, offload, tile — and Wan additionally exposes first-frame and last-frame conditioning arguments for its I2V and first/last-frame variants, which is one of its standout open features.

For genuinely long clips that do not fit even with these tricks, the move is **multi-GPU sequence parallelism**: split the long token sequence across GPUs so each holds a contiguous slab of spacetime tokens, with the attention computed across the split. This is the same idea as sequence/context parallelism in LLM training, applied to the video DiT's enormous sequence, and `accelerate` plus the models' own multi-GPU inference scripts support it. Mechanically each GPU holds $L / G$ of the tokens for $G$ GPUs, the per-GPU activation memory drops by $G\times$, and the attention is computed with a ring or all-gather communication so every token still attends to every other token — you trade GPU-to-GPU bandwidth for the ability to hold a sequence no single card could. The launch is a one-liner that scales the same script across cards:

```bash
# Render a long clip with the denoiser's sequence sharded across 4 GPUs.
# Each card holds ~L/4 of the ~100k+ spacetime tokens; attention is
# computed across the shards, so the full clip fits where one card OOMs.
accelerate launch --num_processes 4 --multi_gpu \
  generate_long_video.py \
  --model "hunyuanvideo-community/HunyuanVideo" \
  --num_frames 129 \
  --height 720 --width 1280 \
  --sequence_parallel_degree 4 \
  --num_inference_steps 30 \
  --output "long_clip.mp4"
```

The key flag is `--sequence_parallel_degree`: it tells the runtime to shard the token sequence across that many GPUs. The catch is that sequence parallelism scales the *attention sequence*, not the clip length arbitrarily — you are still bounded by the VAE's trained length, past which the decode degrades regardless of how many GPUs you throw at the denoiser. For clips longer than the VAE's envelope you need the genuinely different machinery of chunked or autoregressive rollout, which the [long-video post](/blog/machine-learning/video-generation/why-video-generation-is-hard) covers; sequence parallelism buys you *one long render that fits*, not *unbounded length*.

## 8. HunyuanVideo 1.5: the distillation that fits a 4090

The most consequential recent open release is **HunyuanVideo 1.5**, and it is worth its own section because it is the clearest demonstration of where the open frontier is heading: not bigger, but *smaller and faster at the same quality*. HunyuanVideo 1.5 is a re-engineered, substantially smaller model (roughly 8B-class rather than 13B) that, with step distillation and an improved VAE plus a super-resolution refinement pass, produces clips of comparable quality to the original 13B model while running on a *single RTX 4090*. The reported envelope — and I mark this approximate, drawn from the release notes — is on the order of generating multi-second clips in a couple of minutes on one 24GB card, where the 13B original needed an A100-class GPU.

How does it get there? Three techniques you have now seen the foundations of. **Step distillation**: a student model is trained to match the multi-step teacher's output in far fewer sampling steps — the video analog of the [few-step image models](/blog/machine-learning/image-generation/consistency-models-and-few-step-generation) — cutting the dominant cost, which is the number of expensive denoising passes. **A higher-compression VAE with a separate super-resolution stage**: generate at a smaller spatial latent (fewer tokens, cheaper denoising), then upsample with a lightweight SR pass, so the heavy DiT never has to operate at full resolution. **Aggressive offload and tiling baked into the reference pipeline**, so the out-of-the-box experience fits 24GB rather than requiring you to discover the flags yourself.

The lesson generalizes beyond Hunyuan. The open frontier's next year is not a race to 30B denoisers; it is a race to make the existing quality run on hardware people own. Distillation, higher-compression VAEs, and few-step samplers are the levers, and they are exactly the levers the [efficient and real-time generation post](/blog/machine-learning/video-generation/why-video-generation-is-hard) explores in depth. HunyuanVideo 1.5 is the first widely-usable proof that the 13B quality bar is reachable on a consumer card. When you are choosing a model in 2026, the distilled tier is often the right default, not a compromise.

## The self-hosting reality: the decode wall and the two ways to run

There are two practical ways people actually run these models, and it is worth naming both because the right choice depends on whether you are scripting or art-directing. The first is the `diffusers` Python path you have seen — best when you are building a pipeline, batching renders, or fine-tuning, because it is programmable and composes with the rest of your code. The second is **ComfyUI**, a node-graph interface where you wire the text encoder, the denoiser, the sampler, and the VAE decode as visual boxes and tune them interactively. ComfyUI is where most of the open-model community lives, because it makes the offload-and-tiling knobs visible and lets you swap a LoRA or a sampler without editing code. The trade is reproducibility: a Python script is a precise, version-controllable artifact; a Comfy graph is faster to explore but harder to pin down. My rule is explore in ComfyUI, then port the settled recipe to a `diffusers` script for production.

Whichever path you take, the same wall waits for you, and it is worth one focused pass because it is the single most common way a self-hosted render fails: **the 3D-VAE decode**. Here is the mechanics of why it is so memory-hungry. The denoising steps run in latent space, where the clip is the small $T' \times H' \times W'$ tensor — that is cheap. But the *final* step, decoding that latent back to pixels, has to materialize the full $T \times H \times W \times 3$ tensor, and for a 720p clip that is hundreds of millions of floats, plus the VAE decoder's own activations, which for a 3D causal decoder include temporal-convolution buffers that span several frames at once. The decode peak can therefore exceed the peak of *all* the denoising steps combined, and it arrives at the very end, after the expensive part has already succeeded. An OOM at the decode is the cruelest failure mode in this whole domain.

The fix is two complementary knobs. **VAE tiling** (`vae.enable_tiling()`) splits the spatial decode into overlapping tiles that are decoded one at a time and stitched, so the decoder never holds the full $H \times W$ frame in memory — it holds one tile. **Decode chunking** (the temporal analog, exposed in some pipelines as `decode_chunk_size` or handled by `enable_slicing()`) decodes a few frames at a time rather than the whole clip's frames at once, so the decoder never holds all $T$ frames' activations simultaneously. Together they convert the decode's memory from "full clip at once" to "one spatial tile of a few frames at a time," which is the difference between an 18GB spike and a 4GB one. The cost is a little speed (tiles are decoded sequentially) and, if the tiling is too aggressive, faint seams at tile boundaries — which is why the tiling defaults overlap the tiles and blend the seams. If you see grid-like artifacts in your output, your tiles are too small or the overlap is off; widen them or accept a higher decode peak.

#### Worked example: diagnosing an OOM-at-second-6

A concrete debugging story, because it is the one everyone hits. You launch a HunyuanVideo render at 720p, the progress bar marches through all 30 denoising steps over eight minutes, and then — `CUDA out of memory` — it dies at the final step. The instinct is "the model is too big, I need a smaller one." That instinct is wrong, and the VRAM-math figure tells you why: the denoising steps already *succeeded*, so the denoiser weights and activations clearly fit. The failure is the decode, which spikes *after* the steps. The fix is not a smaller model; it is `pipe.vae.enable_tiling()` and, if that is not enough, `pipe.vae.enable_slicing()` plus a smaller decode chunk. The diagnostic rule that saves the afternoon: **if it OOMs during the steps, the denoiser is too big — quantize or offload; if it OOMs after the steps, the decode is the wall — tile and chunk.** Where in the timeline the OOM lands tells you exactly which lever to pull.

## 9. The results: a comparison table you can act on

Now the proof. Here is a comparison of the four open models on the axes that decide whether you can use them, drawn from the technical reports, model cards, and VBench leaderboard standings, with VRAM figures for a single-GPU self-hosted setup. Where a figure is approximate I mark it; treat all VRAM numbers as "with the offload and tiling tricks from this post, at 720p, in the stated dtype," because without those qualifiers a VRAM number is meaningless.

| Model | Denoiser params | VAE ratio | Frames × res (typical) | Text encoder | VBench / notes | Min VRAM (self-host) |
|---|---|---|---|---|---|---|
| CogVideoX-2B | 2B | 4×8×8, 16ch | 49 × 720×480 | T5-XXL (~4.8B) | Solid open baseline; proved the recipe | ~5 GB (offload) |
| CogVideoX-5B | 5B | 4×8×8, 16ch | 49 × 720×480 | T5-XXL (~4.8B) | Noticeably better motion/detail than 2B | ~12 GB (offload) |
| HunyuanVideo | ~13B | 4×8×8, 16ch | 129 × 1280×720 | MLLM (decoder LLM) | Near-closed quality; top open VBench tier | ~45 GB fp16 / ~20 GB int8 |
| HunyuanVideo-1.5 | ~8B (distilled) | higher temporal + SR | multi-sec × 720p+ | MLLM (decoder LLM) | ~13B-class quality, runs on a 4090 | ~14 GB (single 4090) |
| Wan 2.1 14B | 14B | Wan-VAE 4×16×16 | up to ~5s × 720p | umT5 (multilingual) | Leads several open T2V/I2V benchmarks | ~24 GB quantized |
| Wan 2.2 (A14B MoE) | 14B total / ~2 experts | Wan-VAE 4×16×16 | up to ~5s × 720p | umT5 (multilingual) | MoE capacity at lower active FLOPs; strong I2V | ~24 GB quantized |

A few honest caveats about this table. VBench is a *noisy, gameable* benchmark — its "dynamic degree" dimension can be inflated by adding jittery motion that tanks the "motion smoothness" and "subject consistency" dimensions, so a single aggregate VBench number hides the quality↔motion trade-off. The right way to read it is dimension by dimension, and the right way to *measure* a model honestly is the protocol from the [video metrics post](/blog/machine-learning/video-generation/the-metrics-of-video-generation): a fixed prompt set, a fixed seed, a warm-up render discarded, and FVD computed on a held-out reference set with the same I3D features and the same number of samples. Never compare two models' FVD computed on different sample counts; FVD is biased by sample size. And the VRAM figures are *peak* figures on a self-hosted single card with the tricks applied — your mileage varies with resolution, length, and exact driver, which is precisely why every figure in the table is qualified.

When you benchmark these models yourself — and you should, because reported numbers rarely use your resolution and length — measure both the wall-clock and the *peak* VRAM honestly, discarding the first render so you do not pay the one-time compile and weight-load cost in your timing:

```python
import time
import torch

def benchmark(pipe, prompt, num_frames, steps, runs=3):
    # Warm-up: the first call pays compile + cudnn autotune + weight load.
    _ = pipe(prompt=prompt, num_frames=num_frames, num_inference_steps=steps).frames[0]
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = pipe(prompt=prompt, num_frames=num_frames, num_inference_steps=steps).frames[0]
        torch.cuda.synchronize()           # GPU work is async; sync before timing
        times.append(time.perf_counter() - t0)

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"median {sorted(times)[len(times)//2]:.1f}s/clip, peak {peak_gb:.1f} GB")
```

Two non-obvious correctness points here, both of which silently corrupt video-model benchmarks. First, `torch.cuda.synchronize()` is mandatory — CUDA work is asynchronous, so without the sync your timer measures *kernel-launch* time, not compute time, and you will report a model as ten times faster than it is. Second, the warm-up call is mandatory because the first render pays one-time costs (CUDA graph capture, autotune, weight load from disk) that have nothing to do with steady-state speed; report the *median* of warm runs, not the mean, so a single stutter does not skew the number. Apply this and your seconds-per-clip figures become reproducible and comparable; skip it and they are noise.

#### Worked example: seconds-per-clip and the cost of a render

Put real numbers on "is this fast enough?" On a single RTX 4090 (24GB), a CogVideoX-5B render of a 49-frame 480p clip at 50 steps lands in the neighborhood of 2–4 minutes wall-clock with offload (offload trades VRAM for speed, because modules shuttle across PCIe). HunyuanVideo 13B int8 on the same card, 30 steps, 720p, is slower — call it 8–15 minutes for a few seconds, and you are squarely in "render it overnight in a batch" territory, not "iterate interactively." HunyuanVideo 1.5's whole point is to collapse that to a couple of minutes. On the cost side, renting an A100 80GB at roughly \$2/hr to skip the offload penalty, a 5-second HunyuanVideo clip in ~5 minutes costs about \$0.17 of compute; a 4090 you already own costs only electricity. The decision rule: **if you iterate (try many prompts), use the smallest model that clears your quality bar and keep it resident in VRAM with no offload; if you batch (render a known list overnight), use the biggest model and eat the offload penalty.** Iteration latency, not peak quality, is what makes a model pleasant to work with.

## 10. The open-versus-closed gap: where it actually stands

It is worth being precise about the gap, because it is easy to either over- or under-state. The honest summary for early 2026: **at the top of the open tier, the gap to the best closed models is small and shrinking, but it is not zero, and it is largest exactly where closed labs invested most — long-clip coherence, synchronized audio, and the highest-fidelity 4K cinematic look.**

Concretely. On a five-second 720p text-to-video clip of an ordinary scene, a top open model (HunyuanVideo 13B, Wan 2.2) is genuinely competitive — a careful viewer might not reliably tell it from a closed flagship, and on some VBench dimensions the open models *lead*. But push to the closed models' home turf and the gap reappears: native synchronized audio (a Veo strength, which no open model matches as of this writing — covered in the [audio and joint AV post](/blog/machine-learning/video-generation/audio-and-joint-av-generation)); long clips that hold identity for thirty seconds without drift; native 4K; and the hardest-to-fake property, *physical plausibility* over complex motion. Those are where closed labs spent compute the open models have not. The figure below frames the trajectory: from a clear generation behind in 2023 to roughly a VBench point at the top of the tier in 2026.

![Before and after comparison showing the open video tier moving from flickery two second clips clearly behind closed models in 2023 to coherent five second 720p clips within about one VBench point in 2026](/imgs/blogs/the-open-video-frontier-wan-hunyuanvideo-cogvideox-4.png)

Why did the open gap close so fast when, say, the open-versus-closed LLM gap took longer? Two reasons specific to video. First, the recipe is *more discoverable*: a video model is a 3D-VAE plus a DiT plus flow matching plus a text encoder, and every one of those components had a public, well-understood image-generation ancestor, so the open labs were not inventing from scratch — they were assembling known parts on the time axis. Second, the *data* story is more tractable than for LLMs: high-quality video-caption pairs are scrapeable and curatable at the scale these models need, and the open labs published their curation recipes. The convergence we keep returning to is itself the reason the gap closed — when three labs independently arrive at the same architecture, that architecture is no longer a moat.

## 11. Case studies: real numbers from the three reports

Let me ground all of this in named, sourced results from the technical reports, so the claims are checkable rather than vibes.

**CogVideoX (Zhipu AI, 2024) — the model that proved the recipe.** CogVideoX shipped at 2B and 5B parameters with the $4\times8\times8$ causal 3D-VAE and the expert-adaptive-LayerNorm MM-DiT, with both T2V and I2V variants, all weights open. Its significance is historical: it was the first *fully open* model to demonstrate, end to end, that the causal-3D-VAE-plus-MM-DiT-plus-flow-matching recipe produces coherent multi-second clips you can download and run. The 5B model's improvement over the 2B in motion fidelity and detail was a clean demonstration of the scaling thesis at the open tier — more denoiser capacity over the same VAE buys coherence. Every open model after it is, in a real sense, a CogVideoX descendant.

**HunyuanVideo (Tencent, 2024) — open quality reaching the closed bar.** HunyuanVideo at ~13B was the first open model that a careful evaluation placed *competitive with closed flagships* on short clips, via its large 3D-VAE, the dual-stream-then-single-stream MM-DiT, and the MLLM text encoder. Its release was the moment "open video is a flickery toy" stopped being true. The follow-up, **HunyuanVideo 1.5**, then did the harder engineering work of compressing that quality onto a single 4090 through distillation, a higher-compression VAE, and a super-resolution pass — the clearest signal that the open frontier's next phase is efficiency, not raw scale.

**Wan (Alibaba, 2024–2026) — the benchmark leader and the I2V workhorse.** The Wan 2.x family (2.1, 2.2, and point releases) led several open T2V and I2V benchmarks on release, built on the aggressive Wan-VAE ($4\times16\times16$ spatial compression), umT5 multilingual text encoding, and — in Wan 2.2 — a mixture-of-experts denoiser that grows capacity without proportionally growing per-step FLOPs. Wan's standout *practical* features are its strong image-to-video quality and its first/last-frame control: you supply both endpoints and the model interpolates a coherent clip between them, which is exactly the control a designer wants and which the closed APIs often do not expose. For an open model you can self-host and direct precisely, Wan is frequently the right call.

The through-line across all three: same recipe, different dials. CogVideoX tuned for the lightest fully-open proof; HunyuanVideo tuned for top fidelity then for efficiency; Wan tuned for benchmark-leading quality with an aggressive VAE and strong I2V control. Learn the recipe and you can read any of their reports — and any future open report — and know precisely what knob they turned.

## The training recipe the reports actually converged on

The architecture converged, and so — less visibly — did the *training* recipe, in three moves that every one of the open reports describes and that you should know if you ever fine-tune or pretrain one of these. They matter because they are the difference between a model that produces a still image that happens to repeat and a model that produces *motion*.

**Move one: joint image-and-video training.** All three train on images and videos together, treating a single image as a one-frame clip. The reason is data economy and quality transfer: high-quality captioned images vastly outnumber high-quality captioned videos, and the per-frame visual fidelity the model learns from images transfers directly to the spatial quality of video frames. Practically, the training data loader mixes image batches and video batches, and the model's positional encoding handles a one-frame clip as a degenerate case of the general $(t, h, w)$ scheme. Skip this and the model's individual frames look worse than a dedicated image model's; include it and you get image-model frame quality with video-model motion.

**Move two: a multi-stage resolution and length curriculum.** None of these models trains at the final resolution and length from scratch — that would be ruinously expensive given the $L^2$ attention cost. They train low-resolution and short first (where $L$ is small and the model learns motion and semantics cheaply), then progressively increase resolution and clip length in later stages (where the model only has to *refine* spatial detail and extend coherence). This is the direct exploitation of the one law from section 1: spend your cheap early compute where $L$ is small to learn the hard temporal structure, then pay the expensive high-$L$ compute only to polish. The curriculum is why a 13B model can be trained at all on a finite cluster.

**Move three: aggressive caption re-writing.** Raw video captions scraped from the web are short and noisy ("dog on beach"). All three open labs run a captioning model — often a vision-language model — over their training clips to produce long, dense, structured captions ("a corgi runs from left to right across a sunlit beach, the camera tracking from the side, waves breaking behind it"). The model is then trained on those dense captions, which is *why* it responds to long compositional prompts at inference — it was trained on them. This is also why your prompts to these models should be *long and detailed*: the model saw dense captions in training, so a one-line prompt under-specifies relative to its training distribution and you leave quality on the table. The practical advice that falls out: write paragraph-length prompts, name the camera motion, name the lighting, and the open models reward you for it because that is the distribution they learned.

#### Worked example: why your short prompt underperforms

Take the difference concretely. Prompt the model "a cat" and you get a roughly static cat — the model has no instruction about motion, camera, or scene, so it defaults to the lowest-energy thing that satisfies the caption, which is minimal motion. Prompt it "a fluffy orange cat leaps from a kitchen counter to the floor, the camera following the arc, soft morning light through a window, slow motion, photorealistic" and the *same model* produces real motion with a real camera. Nothing changed but the prompt length, and the VBench "dynamic degree" can swing from near-zero to healthy. The model is not being lazy on the short prompt; it is doing exactly what its dense-caption training distribution implies, which is that an under-specified caption corresponds to an under-specified, low-motion scene. The actionable rule: **treat the prompt as a shot description, not a label.** This single habit improves open-model output more than any flag.

## 12. Fine-tuning an open model with a video LoRA

The final reason open weights matter: you can *teach them*. You cannot fine-tune a closed API, but you can take CogVideoX or Wan and, on twenty to fifty clips and a single 24GB card, teach it a specific subject (your product, your character) or a specific style (a particular animation look). The technique is LoRA — low-rank adaptation — the same [parameter-efficient fine-tuning](/blog/machine-learning/image-generation/diffusion-transformers-dit) that revolutionized image-model customization, now applied to the video DiT.

The idea is simple and the figure below makes it concrete: you *freeze* the entire giant denoiser, the VAE, and the text encoder, and you inject small trainable low-rank matrices into the DiT's attention layers. For a weight matrix $W \in \mathbb{R}^{d \times d}$ you learn $W + BA$ where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times d}$ with rank $r \ll d$ (typically $r = 16$ or $32$). The number of trainable parameters drops below one percent of the model, so you can fine-tune a 13B model on a single card, and the resulting adapter is a tiny 100–300MB file you load at inference on top of the frozen base.

![Stacked diagram of the video LoRA fine tuning flow freezing the denoiser VAE and text encoder while training rank sixteen adapters on the attention layers to produce a small adapter checkpoint](/imgs/blogs/the-open-video-frontier-wan-hunyuanvideo-cogvideox-8.png)

Here is the training launch with `accelerate` and `peft`, the standard `diffusers` LoRA training pattern adapted for CogVideoX:

```bash
accelerate launch train_cogvideox_lora.py \
  --pretrained_model_name_or_path "THUDM/CogVideoX-5b" \
  --instance_data_root "./my_clips" \
  --caption_column "prompt" \
  --video_column "video" \
  --rank 16 \
  --lora_alpha 16 \
  --mixed_precision "bf16" \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing \
  --learning_rate 1e-4 \
  --lr_scheduler "cosine_with_restarts" \
  --max_train_steps 2000 \
  --enable_model_cpu_offload \
  --output_dir "./cogvideox-lora-mysubject"
```

Note `--gradient_checkpointing` and `--enable_model_cpu_offload`: training a video model is even more VRAM-hungry than inference because you also store gradients and optimizer state, so the same offload-and-checkpoint levers from inference apply, harder. `--rank 16` is the dial that trades adapter capacity against trainable size; rank 16 is a good default for a subject, rank 32+ for a complex style. Then loading the adapter at inference is one call:

```python
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

# Load the tiny LoRA adapter on top of the frozen 5B base.
pipe.load_lora_weights("./cogvideox-lora-mysubject", adapter_name="mysubject")
pipe.set_adapters(["mysubject"], adapter_weights=[0.9])

video = pipe(
    prompt="mysubject corgi surfing a wave at sunset, cinematic, smooth motion.",
    num_frames=49,
    num_inference_steps=50,
    guidance_scale=6.0,
).frames[0]
export_to_video(video, "mysubject_surf.mp4", fps=8)
```

The `adapter_weights=[0.9]` scales the LoRA's influence — turn it down if the adapter overpowers the prompt, up if the subject is not coming through strongly enough. This single capability — teaching an open model your subject on a handful of clips — is, for many production teams, the entire reason to choose an open model over a closed API, and it is covered end to end in the series [capstone playbook](/blog/machine-learning/video-generation/building-with-video-generation-the-playbook).

## 13. When to reach for which open model (and when not to)

Decisions, stated plainly. The first fork is hardware, the second is goal, and the tree below captures it; the prose makes it actionable.

![Tree of which open video model to run starting from a hardware fork between a single twenty four gigabyte card and a multi GPU box then splitting by quality versus iteration speed into CogVideoX HunyuanVideo and Wan](/imgs/blogs/the-open-video-frontier-wan-hunyuanvideo-cogvideox-6.png)

**If you have one 24GB card and want to iterate**, start with **CogVideoX-5B**. It is the lightest, the best-integrated into `diffusers`, keeps resident in VRAM without aggressive offload at lower resolution, and gives the fastest prompt-to-clip loop. Do not start with HunyuanVideo 13B for iteration — the offload penalty makes every render a coffee break, and you will iterate ten times slower for a quality gain you do not need while exploring.

**If you have one 24GB card and want the best quality that still fits**, use **HunyuanVideo 1.5**. It is the distilled model designed for exactly this hardware, and it brings near-13B quality to a 4090 without the overnight-batch latency. This is the sweet spot for most solo developers and small teams in 2026.

**If you have a multi-GPU box or rent A100/H100 time and want the top of the open tier**, use **HunyuanVideo 13B** for fidelity or **Wan 2.2** for benchmark-leading quality and the best I2V and first/last-frame control. Choose Wan specifically when you need *multilingual prompts* (umT5) or *precise endpoint control* (supply first and last frames). Choose Hunyuan when raw single-clip fidelity is the priority.

**When NOT to self-host at all:** if you need synchronized audio, native 4K, or thirty-second clips that hold identity, no open model delivers that today — that is closed-model territory ([Veo and the cinematic tier](/blog/machine-learning/video-generation/why-video-generation-is-hard)), and you should pay the API rather than fight the open tools to a worse result. And **when NOT to use T2V**: if you can supply a first frame, always prefer I2V — it is more coherent, more controllable, and costs the same. Generating one good image then animating it beats generating a whole clip from text almost every time.

A stress test on the recommendation, in the series' spirit: *what breaks each choice?* CogVideoX-5B breaks on large, fast motion — its smaller capacity shows up as motion blur and identity wobble when the scene moves hard. HunyuanVideo 1.5's distillation can show as slightly reduced fine detail versus the 13B original on the most demanding scenes. Wan's aggressive $16\times$ VAE can soften high-frequency texture (the cost of compressing more into the latent). And *all three* break the moment you push past the VAE's trained clip length — the decode degrades or OOMs, the failure is in the autoencoder not the denoiser, and the fix is chunked decoding or the autoregressive long-video methods, not a bigger GPU. Knowing *which component* fails when you push a model past its envelope is the difference between a one-line fix and a wasted afternoon.

There is a second stress test worth running before you commit to a model: *what does it do at the edge of its prompt distribution?* Push a model with a prompt full of text-rendering ("a sign that says OPEN"), fine-grained counting ("exactly five birds"), or hard physics ("water pouring into a glass and filling it"), and all three open models degrade in the same characteristic ways — garbled text, miscounts, water that does not conserve volume — because none of them learned an explicit symbolic or physical model; they learned the statistics of clips. That shared failure profile is itself evidence of the convergence: three models that learned the same recipe on similar data fail in the same places. The practical consequence is that no amount of choosing *between* the open models fixes a physics or text-rendering failure — that is a frontier-of-the-field limit, not a model-selection question, and the honest move is to avoid those prompts or post-composite the text rather than to keep swapping models hoping one is secretly better at it. Spend your model-selection energy on the axes where the models genuinely differ — VRAM fit, I2V control, multilingual prompting, iteration latency — and accept the shared ceiling on the axes where they do not.

## 14. Key takeaways

- **Open video models converged on one recipe**: a causal 3D-VAE ($\sim 4\times8\times8$, 16 channels), an (MM-)DiT denoiser on spacetime tokens, flow-matching training, and an LLM-class text encoder. Learn it once; it reads all three of CogVideoX, HunyuanVideo, and Wan, and every model that follows.
- **The VAE compression ratio is the master lever.** It sets the token count, which sets the attention FLOPs ($\propto L^2$), which sets your VRAM and your clip length. Wan's $16\times$ spatial VAE is why it runs longer at a given memory budget than CogVideoX's $8\times$.
- **The denoisers differ only in how text and video tokens fuse** — CogVideoX's expert-AdaLN joint attention, HunyuanVideo's dual-stream-then-single-stream, Wan's MoE — all points on the known MM-DiT spectrum, not different architectures.
- **The text encoder is an LLM now, and it is your second-biggest weight.** Offload it after it runs; it has no business in VRAM during the denoising steps.
- **The VAE decode, not the denoiser weights, is usually your VRAM wall**, because it happens in pixel space and spikes at the very end. Reach for `vae.enable_tiling()` before you blame the model size.
- **At a fixed VRAM ceiling, drop resolution before model size** to buy length — activations and decode scale with pixel area, weights are a fixed cost.
- **The offload trio fits a 13B model on 24GB**: `enable_model_cpu_offload()` (evict the text encoder), int8 quantization of the transformer, and VAE tiling plus decode chunking take ~60GB naive down to ~20GB.
- **For iteration use the smallest model that clears your quality bar and keep it resident; for batch use the biggest and eat the offload penalty.** Latency, not peak quality, decides how pleasant a model is to use.
- **The open-vs-closed gap is small and shrinking on short clips** but remains real on synchronized audio, native 4K, and long-clip coherence — exactly where closed labs spent the most compute.
- **Open weights mean you can fine-tune.** A rank-16 video LoRA on 20–50 clips and one 24GB card teaches an open model your subject or style — the single capability no closed API offers.

## 15. Further reading

- **CogVideoX** — Yang et al., *CogVideoX: Text-to-Video Diffusion Models with an Expert Transformer* (Zhipu AI, 2024). The report for the model that proved the open recipe; read it for the expert-adaptive-LayerNorm MM-DiT and the causal 3D-VAE design.
- **HunyuanVideo** — Tencent, *HunyuanVideo: A Systematic Framework For Large Video Generative Models* (2024), and the HunyuanVideo 1.5 release notes. The dual-stream-then-single-stream MM-DiT, the MLLM text encoder, and the distillation that fits a 4090.
- **Wan** — Alibaba, *Wan: Open and Advanced Large-Scale Video Generative Models* (2024–2026). The Wan-VAE, umT5 encoding, the Wan 2.2 MoE denoiser, and the first/last-frame control.
- **Lipman et al.**, *Flow Matching for Generative Modeling* (2023) — the training objective all three share; pairs with the series' [flow matching for video](/blog/machine-learning/video-generation/flow-matching-for-video) post.
- **Peebles & Xie**, *Scalable Diffusion Models with Transformers (DiT)* (2023) — the denoiser backbone; the [video-DiT post](/blog/machine-learning/video-generation/video-diffusion-transformers) extends it to spacetime.
- **🤗 `diffusers` documentation** — the `CogVideoXPipeline`, `HunyuanVideoPipeline`, and `WanPipeline` references, the quantization and offload guides, and the LoRA training scripts used in this post.
- **Within this series** — the [foundations post on why video generation is hard](/blog/machine-learning/video-generation/why-video-generation-is-hard), the [3D-VAE compression post](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression), the [conditioning post](/blog/machine-learning/video-generation/conditioning-video-text-image-motion-camera), and the [capstone playbook](/blog/machine-learning/video-generation/building-with-video-generation-the-playbook) that turns all of this into a production pipeline.
- **For the image-diffusion foundations** these models inherit — [MM-DiT and the modern text-to-image recipe](/blog/machine-learning/image-generation/mmdit-and-the-modern-text-to-image-recipe), [classifier-free guidance](/blog/machine-learning/image-generation/classifier-free-guidance), and [few-step generation](/blog/machine-learning/image-generation/consistency-models-and-few-step-generation) in the image series.
