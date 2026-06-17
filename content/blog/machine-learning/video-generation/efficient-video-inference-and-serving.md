---
title: "Efficient Video Inference and Serving: Memory, Cost, and Throughput"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "The systems reality of running video generation in production — a peak-VRAM model that shows the 3D-VAE decode is the real wall, the levers that fit a 14B model on a 24GB card, sequence parallelism for long clips, fp8 and 4-bit quantization, a dollar-per-clip cost model, and a runnable FastAPI plus diffusers serving worker with the ops gotchas spelled out."
tags:
  [
    "video-generation",
    "diffusion-models",
    "inference-serving",
    "vram",
    "quantization",
    "video-diffusion",
    "text-to-video",
    "mlops",
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
image: "/imgs/blogs/efficient-video-inference-and-serving-1.png"
---

The first time I tried to serve a video model in anger, it died at second six. The render had been running for nearly four minutes, the denoiser loop had finished, the progress bar said one hundred percent, and then — right at the end, on the step that should have been the cheap one — the process threw `CUDA out of memory` and took the whole worker down with it. The GPU was a 24GB RTX 4090. The clip was a five-second, 720p prompt of a dog running through a field. The model had fit. The sampling had fit. The thing that did not fit was the very last operation: decoding the finished latent back into pixels.

That failure is the most important thing to understand about putting video generation into production, and it is the opposite of what your intuition from language-model serving tells you. In an LLM the memory you fight is the KV cache, and it grows with the conversation. In a video diffusion model the weights are a fixed cost you pay once, the denoiser activations are large but steady, and then the 3D-VAE decode arrives at the end and asks for a single enormous contiguous buffer to hold every frame at full resolution simultaneously. That one buffer is frequently the largest allocation in the entire pipeline. The denoiser was never the problem. The decode was the wall.

![A vertical stack showing peak VRAM split into model weights, a frozen text encoder, the spacetime denoiser activations, attention scratch, and a 3D-VAE decode spike that pushes the sum past 24GB and triggers an OOM](/imgs/blogs/efficient-video-inference-and-serving-1.png)

This post is about everything that surrounds that moment — the systems and deployment reality of running video generation as a service, as distinct from the algorithmic question of making the model itself fast. Its sibling, [efficient and real-time video generation](/blog/machine-learning/video-generation/efficient-and-real-time-video-generation), is about cutting the *number* of seconds per clip: step distillation, feature caching, the latency model. This post assumes you have done some of that and now have to *run* the thing reliably for real users on rented hardware, where the questions are different. How much VRAM does one request actually peak at, and why does it OOM at the end? How do I fit a 14-billion-parameter model with a 720p clip onto a card I can actually rent? When the clip is long enough that no single GPU holds it, how do I split it? What does one clip *cost*, in real dollars, and which levers cut that cost the most? And how do I wrap all of this in an API that does not fall over when ten requests arrive at once or when a model takes ninety seconds to load?

By the end you will be able to write a memory budget for any video pipeline and predict where it will OOM before you run it; apply the levers — VAE tiling and chunked decode, model offload, FlashAttention, quantization — in the right order to fit a big model on a small card; reason about sequence parallelism for long clips; compute a defensible dollar-per-clip on a named GPU; and stand up a FastAPI worker with a job queue that keeps models warm and survives the ops failure modes that actually happen. This sits squarely on the series spine — video is spatial generation times temporal coherence under a brutal compute budget — and it is the post about paying the *bill* for that budget in production. If you have not read [why video generation is hard](/blog/machine-learning/video-generation/why-video-generation-is-hard), the one-line version is that the time axis multiplies your token count, your activation memory, and your decode cost all at once, and serving is where those three multiplications come due.

## 1. The memory wall: why a video model OOMs where you do not expect

Start with the question that governs whether a request succeeds or dies: what is the *peak* VRAM, the single highest instantaneous allocation, over the whole forward pass? Not the average, not the resident weights — the peak. A GPU OOMs at its peak, and for a video model the peak lands somewhere surprising.

Write the budget as a sum of four terms:

$$
\text{VRAM}_\text{peak} \;\approx\; \underbrace{W}_{\text{weights}} \;+\; \underbrace{A_\text{denoise}(T,H,W)}_{\text{denoiser activations}} \;+\; \underbrace{S_\text{attn}}_{\text{attention scratch}} \;+\; \underbrace{D_\text{vae}(T,H,W)}_{\text{VAE decode spike}}.
$$

The weights term $W$ is fixed and easy: a 14B-parameter diffusion transformer in fp16 is about 28GB just for the denoiser, plus a frozen T5 text encoder that can be another 10GB if you load the large one, plus the VAE's own few hundred megabytes. Already, before a single activation, a naive load of a Wan-2.1-14B-class model in fp16 wants nearly 40GB resident. That alone is why it will never fit a 24GB card without intervention — and why offload and quantization, covered below, are not optional optimizations but the price of entry.

The denoiser activation term $A_\text{denoise}$ scales with the number of spacetime tokens, which is the crux of everything video. For a clip of $T$ latent frames, latent height $H$, and latent width $W$, the token count going into the diffusion transformer is roughly $N = T \cdot H \cdot W$ (times any patchification factor). Concretely, a five-second 720p clip at 16fps, after a causal 3D-VAE with $4\times8\times8$ compression, becomes about $T \approx 21$ latent frames of $H \approx 90 \times W \approx 160$ latent pixels — on the order of $300{,}000$ tokens, and with a finer patch grid you can push past a million. Activations for a transformer scale linearly in token count per layer, so the denoiser holds a steady, large footprint throughout sampling, proportional to $T \cdot H \cdot W$. This is the term that makes video fundamentally heavier than images: an image is one frame, a video is $T$ of them entangled by temporal attention.

The attention scratch term $S_\text{attn}$ is the one that *used* to be catastrophic and now usually is not, because of FlashAttention. Naive attention materializes an $N \times N$ score matrix, which for a million tokens is a trillion entries — instant death. FlashAttention (and PyTorch's `scaled_dot_product_attention`, which dispatches to a fused kernel) never materializes that matrix; it tiles the computation and keeps the memory at $O(N)$ instead of $O(N^2)$. For video this is not a nice-to-have, it is the difference between possible and impossible. We will treat SDPA/FlashAttention as always-on; the interesting memory fights are elsewhere.

And then the fourth term, $D_\text{vae}$, the one that killed my worker. The 3D-VAE decode takes the small clean latent and upsamples it by the compression factor in every dimension — $4\times$ in time, $8\times$ in height, $8\times$ in width — reconstructing the full-resolution RGB clip. If you decode all $T$ frames at once, the decoder's intermediate activations and the output buffer together hold a tensor on the order of $81 \times 720 \times 1280 \times 3$ in fp16 plus the convolutional feature maps at every upsampling stage, which are far larger than the final image because they carry many channels at intermediate resolutions. Empirically this single decode can spike VRAM by ten gigabytes or more on a 720p clip — a spike that arrives at the very end, after everything else is already resident, which is exactly why it is the term that pushes the sum over the cliff.

![A branching graph showing the small denoiser latent feeding both a steady low-VRAM denoiser loop and a 3D-VAE decode that either materializes a huge full-resolution buffer and OOMs or is tiled and chunked into blocks that fit on a 24GB card](/imgs/blogs/efficient-video-inference-and-serving-2.png)

The decode is the wall because of an asymmetry: the denoiser does fifty passes over a *small* latent, while the VAE does one pass that *inflates* that latent to full pixels. Against expectation, the cheap-looking final step is the memory peak. Once you internalize that, the fix is obvious — never decode all frames at full resolution in one shot. Decode in chunks along time, and tile along space. That is the single most important serving knob in video generation, and the next section is about it.

It is worth contrasting this memory profile against the LLM serving most engineers know better, because the differences are exactly the things that trip people up when they move from text to video. In an LLM, the weights are a fixed cost, the *activations* are tiny (one token's worth per decode step), and the memory that grows is the KV cache — it grows with sequence length, and serving an LLM well is largely about managing that cache (paged attention, eviction, prefix sharing). The memory pressure is dynamic, request-dependent, and grows over the life of a conversation. A video diffusion model is almost the mirror image. There is no KV cache that grows over time — diffusion is not autoregressive over tokens, it is iterative over a fixed-size latent — so the memory is *static within a request*: it is set the instant you fix the clip's $T$, $H$, and $W$, and it does not grow as sampling proceeds. The whole footprint is knowable in advance from the request's dimensions, which is a gift for capacity planning: you can compute peak VRAM from the request *before* you run it, route it to a worker that fits, and reject what does not. The catch is the decode spike, which has no analogue in LLM serving at all — there is nothing in text generation that suddenly inflates a small tensor into an enormous one at the very end. An engineer coming from LLM serving instinctively watches the growing cache and is blindsided by the one-shot decode that arrives after the "growth" phase is over. The right model for video is not "manage the cache" but "compute the static peak, and chunk the one spike that dominates it."

This static-peak property also changes how you think about safety margins. With an LLM you size for the *worst-case* sequence length because the cache grows unpredictably; with a video model you size for the *worst-case requested dimensions*, which you can cap and validate at the API boundary. If you forbid clips longer than $N$ frames and resolutions above some ceiling, you have an exact upper bound on per-request VRAM, and a worker sized for that bound will never OOM on a valid request. The OOMs that happen in video production are almost always one of three things: a request that slipped past validation with oversized dimensions, fragmentation accumulating over a worker's lifetime (section 8), or — the classic — forgetting to chunk the decode so the spike blows a budget that the *sampling* phase fit comfortably. All three are preventable with the memory model in hand. The model is not just an explanation; it is an operational tool you run on every request.

#### Worked example: predicting the OOM before you run it

Take Wan-2.1-14B on a 4090 (24GB), a 5-second 720p clip. Weights in fp16: denoiser ~28GB. We have already lost — it does not even fit resident. Suppose we quantize the denoiser to fp8, halving it to ~14GB, and offload the text encoder to CPU so it is not resident during sampling. Now resident weights are ~14GB. Denoiser activations with FlashAttention for ~300k tokens add roughly 3–4GB, so during sampling we sit around 18GB — comfortable. Then the VAE decode arrives. Naive full-clip decode adds ~10–12GB on top of the 14GB resident weights (the VAE is tiny but its decode activations are not), peaking near 26GB. Over the 24GB ceiling: OOM at second six, exactly as I saw. Now enable chunked temporal decode at `decode_chunk_size=4` and spatial tiling: the decode spike drops to ~2GB because we only ever hold a few frames' worth of decoder activations at a time. Peak ~18GB. Fits. We did not change the model's quality — only the *order* and *granularity* in which we materialized pixels.

## 2. Tiling and chunked decode: trading time for memory, on purpose

The VAE-decode fix is simple to state and worth stating precisely because it is the whole ballgame. A 3D-VAE decode is a stack of 3D (or factorized 2D-spatial + 1D-temporal) transposed convolutions. Convolutions are *local*: an output pixel depends only on a bounded neighborhood of inputs. That locality is what licenses decoding the clip in pieces. You can decode a temporal *chunk* of latent frames, then the next chunk, blending across the small overlap the temporal receptive field needs, and you never hold the whole clip's decoder activations at once. Identically, you can *tile* each frame spatially — decode the top-left tile, the top-right, and so on, overlapping by the spatial receptive field and feathering the seams — so the peak activation is one tile's worth, not one frame's worth. Combine both and the decode VRAM becomes nearly independent of clip length and resolution; it is set by your chunk and tile size, which you choose.

The cost is time, and it is real but modest. Tiling and chunking add overhead from the overlap regions (you recompute the seam pixels) and from kernel-launch and memory-copy overhead of many small decodes instead of one big one. In practice this is a 5–15% latency increase on the decode, which is itself a fraction of total render time. You are trading a sliver of latency for the difference between OOM and success — an excellent trade. In `diffusers` this is two flags plus a chunk size:

```python
import torch
from diffusers import WanPipeline
from diffusers.utils import export_to_video

pipe = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

# The three memory levers that matter most for the VAE-decode wall:
pipe.enable_vae_tiling()    # spatial tiling: decode each frame in overlapping tiles
pipe.enable_vae_slicing()   # batch slicing: decode one sample at a time, not the whole batch
# Chunked temporal decode is exposed per-pipeline; on SVD it is decode_chunk_size,
# on Wan/CogVideoX the VAE tiling config carries a temporal tile size.

frames = pipe(
    prompt="a golden retriever running through a sunlit field, cinematic",
    height=720,
    width=1280,
    num_frames=81,            # ~5s at 16fps
    num_inference_steps=40,
    guidance_scale=5.0,
).frames[0]

export_to_video(frames, "dog.mp4", fps=16)
```

For Stable Video Diffusion, the temporal chunk is the headline knob and it is explicit:

```python
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

image = load_image("first_frame.png")

# decode_chunk_size is THE memory knob for SVD's temporal VAE decode.
# Lower it until you fit; each lower value adds a little latency.
frames = pipe(
    image,
    num_frames=25,
    decode_chunk_size=4,      # decode 4 frames at a time, not all 25
    motion_bucket_id=127,
    fps=7,
).frames[0]
```

The mental model is a dial, not a switch: `decode_chunk_size` (or the VAE temporal tile size) is a continuous trade. Set it high for speed when you have headroom, low to survive on a small card. The rule I follow: decode the *largest* chunk that fits with a safety margin, because every chunk boundary costs a little overhead, and you want the fewest boundaries that still fit.

There is a quality footnote worth being honest about. Tiling introduces seams. A well-implemented tiled decode overlaps tiles and blends them, so seams are invisible; a lazy one can leave faint grid artifacts at tile boundaries, most visible on smooth gradients like skies. The `diffusers` implementations blend, so this is rarely an issue, but if you see a faint checkerboard on flat regions, your tile overlap is too small. It is a memory-for-quality trade only if you cut the overlap too aggressively; at default overlaps it is effectively free.

The reason the overlap matters, and the reason it has a *minimum* you cannot go below, is the decoder's receptive field. A 3D-VAE decoder is a stack of convolutions, and the output value at a given pixel depends on a neighborhood of input latents whose size is the cumulative receptive field of that stack. If your tile overlap is smaller than the receptive field, the pixels near a tile boundary are computed from an incomplete neighborhood — they are missing the context that lives in the adjacent tile — and that is precisely what produces a visible seam. So the overlap is not a free parameter you tune for quality; it is a hard floor set by the architecture, and good implementations compute it from the decoder's structure rather than guessing. The same logic applies to the *temporal* chunk: the causal 3D-VAE has a temporal receptive field (how many neighboring latent frames feed each output frame), and your chunk overlap along time must cover it or you get a temporal seam — a visible discontinuity, a little stutter, at the boundary between chunks. This is why `decode_chunk_size` is not arbitrary: too small and you pay overhead and risk temporal seams if the implementation does not overlap correctly; the right value is the largest chunk that fits, with the architecture-mandated overlap.

There is a deeper reason temporal chunking is special for *causal* video VAEs, the kind used in modern open models. A causal VAE is built so that each output frame depends only on the current and *past* latent frames, never future ones — this is what makes streaming and autoregressive generation possible, as the [3D-VAE post](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression) explains. The serving consequence is convenient: because the decode is causal, you can decode chunk by chunk in temporal order and flush each chunk as it completes, without ever needing future frames to finish the current one. That is the structural property that lets a streaming worker return the first second of a clip while the rest is still decoding — the causality that the model was designed for at training time pays off again at serving time, turning a memory optimization (chunked decode) into a latency optimization (early frame delivery) for free. A non-causal VAE cannot do this; it needs the whole clip to decode any of it, so chunking still saves memory but you cannot stream the output. When you choose a model to self-host, this is a serving property worth checking: a causal VAE is strictly easier to serve.

## 3. The full lever stack: fitting a big model on a small card

VAE decode is the dramatic wall, but it is one of several levers, and in production you stack them. Here is the complete kit, ordered by how much VRAM they buy and what they cost you.

![A matrix of memory levers against VRAM saved, latency cost, and quality cost, showing VAE tiling and offload buy large memory cuts at a latency price while fp8 and int4 quantization shrink the weights with fp8 nearly free and int4 risking a visible quality drop](/imgs/blogs/efficient-video-inference-and-serving-3.png)

**Model CPU offload.** The weights are the biggest *resident* cost. If they do not all need to be on the GPU at the same time, move the idle ones to CPU RAM and stream them in as needed. `diffusers` gives two granularities. `enable_model_cpu_offload()` keeps whole submodules (text encoder, transformer, VAE) on CPU and moves each to GPU only while it runs — so the text encoder is on GPU during prompt encoding, then evicted before the denoiser loads. `enable_sequential_cpu_offload()` is far more aggressive, offloading at the *layer* level, which can fit enormous models on tiny cards but pays a heavy PCIe-transfer tax on every forward pass. Model-level offload is the sweet spot: it cuts resident VRAM substantially for a 5–15% latency hit (the text-encoder and VAE swaps), and it is almost always worth enabling for serving on constrained hardware.

```python
# Offload, ordered from gentlest to most aggressive:
pipe.enable_model_cpu_offload()       # whole-module swap; modest latency cost, big VRAM win
# pipe.enable_sequential_cpu_offload()  # per-layer; fits huge models, heavy PCIe tax — last resort
```

A subtle gotcha: do not call `pipe.to("cuda")` *and* an offload method. Offload manages device placement itself; forcing everything to CUDA first defeats it and can OOM. Pick one.

**Quantization.** Section 4 is devoted to this, but in the lever stack: casting the denoiser weights to fp8 roughly halves $W$ with near-zero quality cost and often *faster* compute on Hopper/Ada tensor cores; int4 quarters the weights but starts to cost visible quality on a model this sensitive. Quantization attacks the one term offload can only relocate — it actually makes the weights smaller.

**FlashAttention / SDPA.** Always on. In modern `diffusers` + PyTorch 2.x, `scaled_dot_product_attention` is the default and dispatches to the fused FlashAttention kernel automatically; you rarely touch it. If you are on an older stack, `pipe.enable_xformers_memory_efficient_attention()` gets you the same $O(N)$ attention memory. Without it, a million-token video attention is simply not runnable.

**Activation savings at inference.** Gradient checkpointing is a training trick — recompute activations in the backward pass instead of storing them — but the *spirit* applies at inference too: anywhere you can recompute instead of cache, you trade compute for memory. The VAE chunking from section 2 is exactly this idea applied to the decoder. Some pipelines also expose attention slicing (`enable_attention_slicing`), which computes attention in chunks over the query dimension to cap its working set; on video it is usually unnecessary once SDPA is doing $O(N)$ attention, but it is a free fallback if a particular resolution still spikes.

The order I apply these for a fixed target card: (1) turn on SDPA/FlashAttention and VAE tiling/chunking — free or near-free, do them always; (2) enable model CPU offload — cheap, big win; (3) quantize the denoiser to fp8 — cheap, halves the biggest term; (4) only if still over budget, go to int4 or sequential offload, accepting the quality or latency hit. You almost never need all of them. Two or three usually clear a 14B model onto a 24GB card.

![A before-after diagram contrasting a naive fp16 load whose full-resolution decode spikes peak VRAM to about 40GB and OOMs on a 24GB card against a tiled plus offload plus fp8 configuration whose chunked decode keeps peak near 18GB and fits](/imgs/blogs/efficient-video-inference-and-serving-4.png)

#### Worked example: a 14B model on a 24GB 4090

Target: Wan-2.1-14B, 5s 720p, RTX 4090 (24GB). Baseline fp16 load wants ~28GB of denoiser weights alone — does not fit, full stop. Apply the stack: fp8 denoiser weights → ~14GB; `enable_model_cpu_offload()` so the 10GB T5 encoder is never resident during sampling; SDPA on (default); `enable_vae_tiling()` plus chunked temporal decode. Resident during sampling: ~14GB weights + ~3–4GB activations ≈ 18GB. VAE decode spike with chunking: +~2GB → peak ~20GB. Fits the 24GB card with a ~4GB safety margin for fragmentation. The same model that could not even *load* now serves clips. The cost was a few percent of latency from offload swaps and tiling overhead, plus the small fp8 quality delta we quantify next. That is the whole game: the levers turned an impossible request into a routine one.

## 4. Quantization for serving: fp8, int4, and what they cost

Quantization is the lever that does not just *move* memory around — it shrinks the model itself, and on the right hardware it speeds it up. The principle is the same one the image series derives in detail in [quantization, caching, and efficient inference](/blog/machine-learning/image-generation/quantization-caching-and-efficient-inference), and the LLM mechanics — weight-only versus weight-and-activation, per-channel scales, the activation-outlier problem — are covered in the edge-ai series posts on [weight-only quantization](/blog/machine-learning/edge-ai/llm-quantization-weight-only-gptq-awq) and [activation quantization and SmoothQuant](/blog/machine-learning/edge-ai/llm-quantization-activations-smoothquant-kv-cache). I will not re-derive those here; I will say what is *different* for a video diffusion model.

Two things are different. First, the dominant cost is a single huge transformer evaluated dozens of times, so weight-only quantization already buys most of the memory win — you halve (fp8) or quarter (int4) the 28GB of weights, which is the term that decides whether the model loads at all. Second, video models are *more sensitive* to quantization error than LLMs in one specific way: errors compound across frames through temporal attention, so a small per-frame degradation can read as flicker or texture-crawl across time, which the human eye is very good at catching. An int4 weight error that would be invisible on a still image can show up as a shimmering surface in motion. This is why fp8 is the comfortable serving default and int4 is a "measure it yourself before you ship" decision.

The practical landscape:

- **fp8 (e8m4 / e4m3) weights.** On Hopper (H100) and Ada (4090, L40S) the tensor cores have native fp8 matmul, so fp8 is not just smaller, it is *faster* — you get a memory win and a speed win together, at a quality cost that is typically a fraction of an FVD point and imperceptible on VBench. This is the one I reach for first. Tools: `optimum-quanto`, `torchao`'s float8 path, or `bitsandbytes` for the simpler cases.
- **int8 weights.** Universally supported, ~2× memory, near-lossless for diffusion weights in practice. A safe fallback when fp8 hardware is not available.
- **int4 weights.** ~4× memory — this is what lets a 14B model fit a 24GB card with room to spare — but the quality cost is now measurable. On a sensitive video DiT expect a small but real VBench drop (a point or two on motion-related dimensions) and watch for temporal shimmer. Worth it when memory is the binding constraint; not worth it when fp8 already fits.
- **Quantizing the VAE.** The VAE is small in *weights* but its decode is the memory spike. Quantizing VAE weights barely helps the spike (the spike is activations, not weights) — chunking is the right tool there. Quantizing VAE *activations* is risky because the decoder's job is to reconstruct fine pixel detail and it is unforgiving of precision loss. I leave the VAE in bf16/fp16 and chunk it; quantize the *denoiser*, where the win is large and the risk is managed.

As a decision table for the denoiser weights on a sensitive video DiT:

| Precision | Memory vs fp16 | Speed on Ada/Hopper | Quality cost | When to use |
| --- | --- | --- | --- | --- |
| bf16/fp16 | 1× (baseline) | baseline | none | fits already; no constraint |
| fp8 (e4m3) | ~2× smaller | faster (native fp8) | below FVD noise floor | default on 4090/H100/L40S |
| int8 | ~2× smaller | neutral (no fp8 cores) | near-lossless | fp8 hardware unavailable |
| int4 | ~4× smaller | neutral, dequant cost | visible motion-quality drop | only when fp8 still does not fit |

The pattern the table encodes: fp8 is the free lunch on modern cards, int8 the safe fallback elsewhere, and int4 a deliberate last resort you measure before shipping. Here is a serving-time quantized load with `optimum-quanto`, the path I have found most robust for `diffusers` video pipelines:

```python
import torch
from diffusers import WanTransformer3DModel, WanPipeline
from optimum.quanto import freeze, qfloat8, quantize

# Load the big transformer, quantize ONLY it to fp8, leave VAE/text-encoder alone.
transformer = WanTransformer3DModel.from_pretrained(
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
)
quantize(transformer, weights=qfloat8)   # fp8 weights: ~2x smaller, often faster on Ada/Hopper
freeze(transformer)

pipe = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()
pipe.enable_vae_tiling()

# Optional: compile the denoiser for a steady-state speedup after a one-time warmup cost.
pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune")
```

A `torch.compile` note for serving: the first call after compilation is *slow* — sometimes a minute or more — because it traces and tunes kernels. That cost is a cold-start problem, not a per-request one. You must warm the model (run one throwaway generation) at worker startup so the first real user does not eat the compile. More on cold starts in section 8. A second `torch.compile` gotcha specific to video: compilation specializes on input *shapes*, so if your worker serves multiple resolutions or clip lengths, each distinct shape triggers a *recompile* the first time it is seen — which is another reason to offer a small fixed menu of output dimensions rather than arbitrary ones. Warm up *each* shape you intend to serve, or pass `dynamic=True` to `torch.compile` to trade some peak speed for shape-flexibility and avoid the recompile stalls.

It is worth being concrete about *why* weight-only quantization buys most of the win for a video DiT, because it differs from the activation-quantization story that dominates LLM serving. In an LLM, the KV cache and activations are a large fraction of memory and a large fraction of the bandwidth cost, so quantizing activations and the cache (the SmoothQuant / KV-cache-quantization line) is where a lot of the LLM win lives. In a video diffusion model the picture is different: the weights are the term that decides whether the model *loads at all* (28GB of denoiser is the entire problem), and activations, while large, are kept tractable by FlashAttention's $O(N)$ behavior and by VAE chunking. So weight-only quantization — leave activations in bf16, quantize only the weight matrices — already gets you the load-feasibility win and most of the speed win on fp8 tensor cores, without the harder engineering and quality risk of activation quantization. Activation quantization for a video DiT is a real research frontier (the activations have outliers, just like LLMs, and the temporal structure compounds errors across frames), but for *serving today* the pragmatic recipe is fp8 weight-only on the denoiser, bf16 everywhere else, and chunked decode for the VAE. That combination is robust, well-supported in `optimum-quanto` and `torchao`, and clears the binding constraint with the least quality risk.

One more practical detail: the *granularity* of the quantization scale matters for quality. A single scale for an entire weight tensor (per-tensor) is the cheapest but the most lossy, because one outlier channel forces a coarse scale on all the others. Per-channel scales (one per output channel) cost a little more memory for the scales themselves but recover most of the quality, and are the sensible default for a sensitive video DiT. The `optimum-quanto` and `torchao` paths default to sane per-channel granularity, so you usually do not touch this — but if you see unexpected quality loss after quantizing, the granularity is the first thing to check, exactly as the edge-ai posts on [weight-only quantization](/blog/machine-learning/edge-ai/llm-quantization-weight-only-gptq-awq) detail for LLMs. The mechanics transfer directly; only the sensitivity-to-temporal-error is new.

#### Worked example: the fp8 quality delta, measured honestly

How would you actually know fp8 is safe? Fix a seed, fix a 32-prompt eval set, generate in bf16 and in fp8, and compute FVD against a reference set plus VBench on both. In my experience and the reported literature on diffusion fp8, the FVD delta is small — on the order of a fraction of a point to a couple of points (approximate; it depends on the model and the eval set) — and VBench's subject/background consistency barely moves, while the seconds-per-clip on a 4090 *drops* by 20–40% because the fp8 tensor cores are faster. Int4 on the same protocol shows a larger FVD increase and a visible dip in motion-smoothness and dynamic-degree scores. The honest report is a table, not an adjective: state the FVD numbers, the VBench dimensions, the seed, and the eval-set size, and let the reader judge whether the memory win is worth their quality bar. The dynamic-degree-versus-stability gaming problem from [the metrics of video generation](/blog/machine-learning/video-generation/the-metrics-of-video-generation) applies here too — a quantized model that moves *less* can score *better* on some consistency metrics while being worse, so do not trust a single number.

## 5. Sequence parallelism: when one GPU cannot hold the clip

Everything so far fits a clip on *one* GPU. But the activation term $A_\text{denoise}(T,H,W)$ grows with clip length, and past some length — a long or high-resolution clip, a 1080p render, a 10-second sequence — no single card holds the spacetime activations no matter how you chunk the decode, because the *denoiser itself* (not just the VAE) needs the whole sequence resident to do attention across it. At that point you split the sequence across GPUs. This is sequence parallelism (also called context parallelism), and it is the multi-GPU technique that matters most for long video.

The idea: the denoiser processes a sequence of $N = T \cdot H \cdot W$ spacetime tokens. Split that sequence into $G$ contiguous slices and put one slice on each of $G$ GPUs. Each GPU holds only $N/G$ tokens' worth of activations, so per-device activation memory drops nearly linearly with the device count. The non-trivial part is attention: a token on GPU 0 must be able to attend to a token on GPU 3, because video attention is global across space and time. The standard solution is an *all-to-all* communication at each attention layer (the Ulysses / DeepSpeed-style sequence parallelism, or ring-attention variants): the GPUs exchange the slices they need so that every device can compute attention over the full sequence for its tokens, then exchange back. The communication is the cost — you are trading interconnect bandwidth for memory — which is why this wants NVLink-connected GPUs, not cards talking over PCIe.

![A branching and merging graph showing a 1.2 million token spacetime sequence split into two GPU slices that each run local computation, an all-to-all attention exchange giving global context, then a gather back into a full long-clip latent that fits within per-GPU VRAM](/imgs/blogs/efficient-video-inference-and-serving-5.png)

The math is clean and worth stating because it tells you when to reach for this. Per-device activation memory is approximately

$$
A_\text{per-device} \;\approx\; \frac{A_\text{denoise}(T,H,W)}{G} \;+\; A_\text{comm-buffers},
$$

so with $G=4$ NVLink-connected GPUs you can hold a clip roughly four times longer (or four times the spatial resolution) than one card, minus the communication buffers and minus a sublinear efficiency loss from the all-to-all traffic. The weights are *replicated* on every GPU (each needs the full model to process its slice), so sequence parallelism does not help the weights term $W$ — for that you combine it with quantization, or with tensor/FSDP parallelism that shards the weights too. The clean recipe for long video is: shard the *sequence* with sequence parallelism to cut activations, and quantize the *weights* to cut $W$, because they attack different terms.

In practice you rarely hand-roll the all-to-all. `xDiT` (a community library for parallel diffusion inference) and the parallel paths increasingly landing in `diffusers` wrap sequence/context parallelism for video DiTs, and you drive them through `accelerate`. The launch looks like this:

```bash
# Launch a sequence-parallel video render across 4 GPUs on one NVLink node.
# xDiT exposes ulysses/ring sequence parallelism for diffusion transformers.
torchrun --nproc_per_node=4 generate_long_clip.py \
  --model Wan-AI/Wan2.1-T2V-14B-Diffusers \
  --height 720 --width 1280 --num_frames 161 \
  --ulysses_degree 4 \
  --steps 40
```

It is worth being precise about why sequence parallelism, and not the other parallelism strategies, is the right tool for *this* problem. There are four ways to split a model across GPUs, and they attack different terms. Data parallelism replicates the whole model and gives each GPU a different *request* — it raises throughput but does nothing for a single clip that does not fit, so it is useless for the long-clip memory problem (it is a throughput lever, covered by the worker pool in section 8). Tensor parallelism shards each weight matrix across GPUs — it cuts the *weights* term $W$, which is helpful when weights are the constraint, but it has heavy per-layer communication and it does not address the activation term that dominates long clips. Pipeline parallelism splits the *layers* across GPUs — it cuts both weights and activations but introduces pipeline bubbles and is awkward for a diffusion model that runs the same stack dozens of times. Sequence parallelism shards the *token sequence* — it cuts the activation term $A_\text{denoise}$, which is exactly the term that grows with clip length, with communication only at the attention layers. For long video, where the binding constraint is activations growing with $T\cdot H\cdot W$, sequence parallelism is the natural fit, and it composes cleanly with quantization (which handles $W$) so you cut both dominant terms at once. In practice the production recipe for very long or high-res clips is sequence-parallel activations plus fp8 weights, possibly with FSDP sharding the weights too if a single card cannot even hold the fp8 model.

The decision rule: reach for sequence parallelism only when the *denoiser* activations — not the VAE decode — are what exceed a single card, which means long or high-res clips. For a standard 5-second 720p clip on a 24GB card, the levers in sections 2–4 are enough and multi-GPU is wasted money and complexity. For a 15-second 1080p clip you have no choice. And note the honest limit from [long video and autoregressive rollout](/blog/machine-learning/video-generation/long-video-and-autoregressive-rollout): sequence parallelism lets you *fit* a longer clip's activations, but it does not solve the *quality* problem of long video — identity drift and error accumulation are a modeling issue, not a memory one. Splitting the sequence across GPUs buys you the memory to attempt a long clip; whether that clip stays coherent is a different fight. There is also a cost subtlety: sequence parallelism uses $G$ GPUs to render *one* clip, so it raises the GPU-seconds-per-clip in absolute terms (you are paying for $G$ cards for the duration), but it does not raise wall-clock proportionally — the clip finishes faster because the work is split. Whether it lowers cost-per-clip depends on the communication efficiency: if the all-to-all overhead is small, $G$ GPUs for $1/G$ the time is roughly cost-neutral while delivering a clip that no single card could produce at all; if the interconnect is slow (PCIe rather than NVLink), the overhead eats the speedup and you pay for $G$ cards while getting much less than $G\times$ the throughput. This is why sequence parallelism wants NVLink — not just for feasibility, but for cost-efficiency.

## 6. Throughput, batching, and time-to-first-frame

So far the lens has been a single request's memory. Production is about *many* requests, and there the governing trade is latency versus throughput — the same tension as any inference service, with a video-specific twist.

Throughput is clips per hour; latency is seconds a single user waits. Batching trades one for the other. If you process clips one at a time (batch size 1), each user gets the minimum possible latency, but the GPU sits partly idle between the compute-heavy and memory-bound phases, so clips-per-hour is low and the cost-per-clip is high. If you batch multiple clips into one forward pass, the GPU stays saturated, throughput rises and cost-per-clip falls — but every user in the batch waits for the *slowest* member, so per-request latency rises. The right batch size depends entirely on what you are optimizing: a queue of offline render jobs wants large batches (maximize throughput, nobody is watching in real time); an interactive product wants small batches (minimize the latency a person feels).

![A matrix comparing batch sizes one, four, and eight against per-clip latency, clips per hour, and cost per clip, showing batch one gives the lowest latency but worst throughput while batch eight maximizes throughput and minimizes cost at the price of the longest wait](/imgs/blogs/efficient-video-inference-and-serving-7.png)

The video-specific twist is memory. Batching multiplies the activation term — a batch of $B$ clips holds $B$ times the spacetime activations and, critically, $B$ times the VAE-decode spike. So batch size is bounded by VRAM, and it is bounded *by the decode spike specifically* unless you chunk it. This is why `enable_vae_slicing()` exists: it decodes the batch one sample at a time so the decode spike does not multiply by $B$. With slicing on, you can run a larger denoiser batch (which shares the compute-heavy part) while keeping the decode memory flat. The pattern for a throughput-oriented worker is: batch the denoiser as large as activations allow, then decode the batch sliced and chunked so the decode does not blow the budget.

A nuance that catches people: dynamic batching, which works beautifully for LLMs, is awkward for video. An LLM server can add a new request to an in-flight batch mid-generation (continuous batching) because each request advances one token at a time and the steps are cheap and uniform. A video diffusion request runs dozens of heavy steps over a *fixed-size* latent, and you cannot easily splice a new clip of different dimensions into a batch already mid-denoise — the tensors do not align, and the step is too coarse to interrupt cheaply. So video batching is mostly *static*: you collect requests of *compatible dimensions* into a batch, run them together start to finish, and the batch is fixed for its lifetime. This means a batching policy needs a small time window — wait up to, say, two seconds to collect same-resolution requests, then fire the batch — trading a little queueing latency for batch efficiency. The window length is itself a latency-versus-throughput knob layered on top of the batch-size knob. For a render queue you can use a generous window (throughput is everything); for an interactive product you use a tiny window or none (the queueing wait is latency the user feels). Requests of *different* resolutions cannot share a batch at all, which is a practical argument for offering a small fixed menu of output resolutions rather than arbitrary dimensions — every distinct resolution fragments your batching and lowers utilization.

Time-to-first-frame is the metric that matters for *streaming* and interactive generation, and it is different from total latency. Most video pipelines are not streaming — they produce all frames then return the clip, so the user sees nothing until the whole render finishes, including the VAE decode. But the causal and autoregressive models from the [real-time post](/blog/machine-learning/video-generation/efficient-and-real-time-video-generation) produce frames in temporal order, which means you *can* stream: decode and ship the first chunk of frames while the model is still generating later ones. For a product where the user watches the clip grow, time-to-first-frame — not total render time — is the latency they perceive, and a streaming decode that returns the first second in a few hundred milliseconds feels real-time even if the full ten-second clip takes much longer to finish. The serving implication: if your model supports causal generation, structure the worker to flush frames to the client as chunks complete, rather than buffering the whole clip.

#### Worked example: choosing a batch size for a render queue

You run an offline render queue on an A100 80GB. Nobody waits live; you want maximum clips per hour to minimize cost. With VAE slicing on, you find batch 8 fits in VRAM and saturates the GPU. Measured (approximate, illustrative): batch 1 gives ~40s/clip and ~90 clips/hour; batch 8 gives ~190s wall per *batch* but that is ~24s/clip amortized and ~165 clips/hour — nearly double the throughput, so nearly half the cost-per-clip. For this queue, batch 8 is obviously right: the per-request latency of 190s is irrelevant because no human is watching, and the cost-per-clip nearly halves. Now swap the use case to an interactive demo where users iterate on prompts: batch 1 (or dynamic batching with a tiny window) is right, because a user who waits three minutes per iteration stops iterating. Same hardware, opposite decision — driven entirely by whether latency or throughput is the thing you are paid to optimize.

## 7. The cost model: what one clip actually costs

Now the dollars. The cost of one clip on rented hardware is almost embarrassingly simple to write down, and the simplicity is the point — it tells you exactly which lever moves the bill:

$$
\text{cost per clip} \;=\; \frac{\text{GPU-seconds per clip}}{3600} \;\times\; \text{rate per GPU-hour}.
$$

That is it. Every optimization in this post and its sibling either cuts GPU-seconds per clip (distillation, caching, quantization speedups, batching efficiency) or lets you use a cheaper GPU (the memory levers, which let a job that needed an A100 run on a 4090). There is no third term. Cost is seconds times rate, and the levers multiply *through the seconds*.

Plug in numbers. Rent an A100 80GB at roughly \$2 per GPU-hour (approximate, varies by provider and commitment). A naive 5-second 720p render at 50 steps takes ~90 GPU-seconds. Cost: $90/3600 \times \$2 \approx \$0.05$ per clip. Five cents sounds trivial until you serve a million clips a month — that is \$50,000 a month, real money that a serving optimization directly attacks. Now compound the levers, and note they multiply: step distillation from 50 to ~4 steps cuts the denoiser term roughly 6×; feature caching across steps adds maybe another 1.5×; fp8 plus `torch.compile` another ~1.4× from faster tensor cores. The seconds-per-clip drops from ~90 to ~7. Cost: $7/3600 \times \$2 \approx \$0.004$ per clip — an order of magnitude cheaper, the same million clips now costing ~\$4,000 a month instead of \$50,000.

![A vertical stack showing cost per clip starting from a naive ninety-second render at five cents and compounding step distillation, feature caching, and fp8 plus compile multiplicatively down to a seven-second render at under half a cent](/imgs/blogs/efficient-video-inference-and-serving-8.png)

The reason a *single* 5-second 720p clip can cost real money is now visible in the model: it is GPU-seconds, and video is heavy in GPU-seconds because each of the (many) sampling steps is a full forward pass over hundreds of thousands of spacetime tokens, followed by a non-trivial decode. An image is one step's worth of tokens decoded once; a video is fifty steps over $T$ frames' worth of tokens. The cost ratio between a video clip and a still image is roughly $(\text{steps ratio}) \times (\text{tokens ratio})$, which is why a clip costs cents while an image costs a fraction of a cent — and why the levers that cut steps and tokens (distillation, caching, the memory tricks that allow batching) compound into the order-of-magnitude savings above.

#### Worked example: the full-stack cost collapse on a million clips

Suppose you are pricing a product that generates a million 5-second 720p clips a month, and you want to know what the serving stack is worth in dollars. Start naive: 50-step fp16 render, ~90 GPU-seconds on an A100 at \$2/hr, full-clip decode. Per clip: $90/3600 \times \$2 \approx \$0.05$. Monthly: \$50,000 — and that is *if* it runs, which on a 4090 it does not (the decode OOMs), so naive also forces the expensive card. Now walk the stack. fp8 weights cut the per-step compute ~1.3× and let you drop to a 4090 at, say, \$0.40/hr (approximate spot pricing) — already the *rate* term fell 5×. Step distillation 50→4 cuts steps ~6×. Feature caching adds ~1.4×. `torch.compile` and the fp8 tensor-core speedup fold in another ~1.3×. Compounded on the seconds term: $90 / (6 \times 1.4 \times 1.3) \approx 8$ GPU-seconds per clip. On the 4090 at \$0.40/hr: $8/3600 \times \$0.40 \approx \$0.0009$ per clip. Monthly: ~\$900. The stack took a \$50,000/month workload to under \$1,000/month — a 50× reduction — by attacking the seconds term multiplicatively *and* the rate term by right-sizing the card. The two biggest single levers were step distillation (the 6×) and the card downgrade (the 5× on rate), and the card downgrade was *only possible* because the memory levers in this post fit the model on the cheaper card. That is the punchline: the serving optimizations did not just save memory, they unlocked a cheaper rate tier, and the rate tier was half the total saving.

A practical cost-control checklist that falls straight out of the model:

- **Cut steps first.** Distillation is the single biggest multiplier on the seconds term. A 4-step distilled student is ~6× cheaper than a 50-step teacher before any other lever.
- **Cache across steps.** Feature/block caching reuses computation between adjacent diffusion steps for a further 1.3–1.6× at small quality cost.
- **Quantize for the speedup, not just the memory.** fp8 on Ada/Hopper is faster *and* smaller — a rare lever that helps both terms.
- **Right-size the GPU.** The memory levers let a job run on a cheaper card. If fp8 + tiling fits the clip on a 4090 (cheaper per hour than an A100), and throughput is acceptable, the cheaper card cuts the *rate* term directly.
- **Keep utilization high.** A GPU idle between requests is GPU-seconds you pay for and do not use. Batching and a warm worker pool (section 8) keep utilization up, which lowers effective cost-per-clip even with the same per-clip compute.

## 8. A reference serving setup, and the ops gotchas

Now assemble it into something that serves real traffic. The architecture is standard async-inference, with the video-specific constraints baked in: models are huge and slow to load (so keep them warm), requests are long (so never block the API on generation), and OOM is a real per-request risk (so isolate and guard it).

![A serving topology graph showing a client request hitting a FastAPI endpoint that returns a job id and enqueues to a Redis or SQS job queue, with multiple warm autoscaled GPU workers pulling jobs, running generation on an A100 or 4090, and writing the finished mp4 and status to an object store](/imgs/blogs/efficient-video-inference-and-serving-6.png)

The shape: a FastAPI endpoint that does *not* generate — it validates the request, enqueues a job, and immediately returns a job ID. A pool of GPU workers, each with the model loaded and *warm*, pulls jobs off the queue, generates, and writes the result (the mp4 and a status) to an object store. The client polls or gets a webhook when the job is done. This decoupling is the whole point: generation takes 10–90 seconds, far too long to hold an HTTP request open, and model loading takes another 30–90 seconds, which must happen at worker startup, *off* the request path, never per-request.

Here is the FastAPI side — a generate endpoint with a job queue, deliberately minimal:

```python
import uuid
from fastapi import FastAPI
from pydantic import BaseModel
from redis import Redis
import json

app = FastAPI()
redis = Redis(host="localhost", port=6379)

class GenerateRequest(BaseModel):
    prompt: str
    num_frames: int = 81
    height: int = 720
    width: int = 1280
    steps: int = 40

@app.post("/generate")
def generate(req: GenerateRequest):
    job_id = str(uuid.uuid4())
    # Enqueue, return immediately. The API never runs the model.
    redis.lpush("video_jobs", json.dumps({"job_id": job_id, **req.dict()}))
    redis.hset(f"job:{job_id}", mapping={"status": "queued"})
    return {"job_id": job_id, "status": "queued"}

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    data = redis.hgetall(f"job:{job_id}")
    return {k.decode(): v.decode() for k, v in data.items()}
```

And the worker — the part that holds the warm model and pulls jobs. The critical structure is that the pipeline is built once, at startup, with all the memory levers and a warmup pass, then loops forever pulling jobs:

```python
import json
import torch
from redis import Redis
from diffusers import WanPipeline
from diffusers.utils import export_to_video

redis = Redis(host="localhost", port=6379)

def build_pipeline():
    pipe = WanPipeline.from_pretrained(
        "Wan-AI/Wan2.1-T2V-14B-Diffusers", torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_tiling()
    pipe.enable_vae_slicing()
    pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune")
    return pipe

def warmup(pipe):
    # Pay the torch.compile and CUDA-graph cost ONCE, off the request path,
    # so the first real user does not eat a 60s compile.
    _ = pipe(prompt="warmup", num_frames=9, height=320, width=320,
             num_inference_steps=2).frames[0]
    torch.cuda.empty_cache()

def run_worker():
    pipe = build_pipeline()
    warmup(pipe)
    while True:
        _, raw = redis.brpop("video_jobs")     # block until a job arrives
        job = json.loads(raw)
        job_id = job["job_id"]
        redis.hset(f"job:{job_id}", "status", "running")
        try:
            frames = pipe(
                prompt=job["prompt"],
                num_frames=job["num_frames"],
                height=job["height"], width=job["width"],
                num_inference_steps=job["steps"],
                guidance_scale=5.0,
            ).frames[0]
            path = f"/data/{job_id}.mp4"
            export_to_video(frames, path, fps=16)
            redis.hset(f"job:{job_id}", mapping={"status": "done", "path": path})
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()            # recover, do not crash the worker
            redis.hset(f"job:{job_id}", mapping={"status": "failed", "error": "oom"})

if __name__ == "__main__":
    run_worker()
```

And the VRAM-profiling snippet you should run on every new configuration *before* you deploy it, so you know your peak and your margin:

```python
import torch

torch.cuda.reset_peak_memory_stats()
frames = pipe(prompt="a dog running", num_frames=81,
              height=720, width=1280, num_inference_steps=40).frames[0]
peak_gb = torch.cuda.max_memory_allocated() / 1e9
print(f"peak VRAM: {peak_gb:.1f} GB")   # the number that decides whether you OOM in prod
```

Run that, note the peak, and size your card with a safety margin above it — because the measured peak is the *floor*, and fragmentation pushes the real ceiling higher. Which brings us to the ops gotchas, the failures that actually take down video workers in production:

**Cold start.** Loading a 14B model from disk or network storage, moving it to GPU, and running the `torch.compile` warmup is a 60–120 second operation. If a request triggers a cold load, that user waits two minutes for nothing. The fix is the warm worker pool above: load and warm up at startup, keep workers resident, and scale the *pool* up before traffic arrives, not in response to a request. Autoscaling video workers is autoscaling *warm capacity*, and because warming is slow, you scale predictively (ahead of expected load) rather than reactively.

**VRAM fragmentation.** PyTorch's caching allocator can fragment over a worker's lifetime — many allocations of different sizes (different clip lengths, resolutions, batch sizes) leave the GPU's memory pool checkerboarded, so a request that *should* fit fails to find a contiguous block and OOMs even though the *total* free memory is sufficient. Symptoms: a worker that ran fine for hours suddenly OOMs on a request identical to ones it served earlier. Mitigations: set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to let the allocator grow segments and fragment less; call `torch.cuda.empty_cache()` between jobs (it returns cached-but-unused blocks to the driver, defragmenting); and pin each worker to a *bounded* range of resolutions/lengths so allocation sizes are predictable. The most robust fix is recycling workers — restart a worker process after N jobs to reset the allocator from scratch, the way you'd recycle a leaky web worker.

**OOM on long requests.** A worker sized for 5-second clips will OOM on a 15-second request even though it is the same model, because activations scale with length. The fix is to *route by size*: validate `num_frames` and resolution at the API, and send oversized requests to a different worker pool (bigger cards, or sequence-parallel multi-GPU workers from section 5), rather than letting them land on a worker that cannot hold them. An API that accepts any clip length and hopes every worker can serve it is an API that OOMs unpredictably. Cap the request, route by the memory model, and reject what truly does not fit with a clear error instead of a crash.

**Recovering from OOM without crashing the worker.** When an OOM does happen — and it will — catch it, call `empty_cache()`, mark the job failed, and *keep the worker alive* to serve the next job, as the worker loop above does. A worker that dies on one bad request takes its warm model with it and triggers a cold start to replace it, turning one failed clip into two minutes of lost capacity. Catch, clean, continue. One caveat: a CUDA OOM does not always leave the context in a fully recoverable state — sometimes the allocator is wedged and the cleanest recovery is to drain the worker and recycle the process. A robust worker tracks consecutive OOMs and, past a small threshold, restarts itself deliberately (a controlled recycle is far better than a hard crash mid-job). The pattern is: catch and retry-clean for the common transient OOM, but recycle the worker if OOMs cluster, because a clustering pattern usually means fragmentation has wedged the pool and only a fresh process will fix it.

**Storage and the export step.** A subtle one that bites in production: after generation, you encode frames to an mp4 and write it somewhere. On a busy worker, the encode (ffmpeg under `export_to_video`) and the upload to object storage are *not* GPU work — they are CPU and network — but they hold the worker's job slot while they run, so a slow upload to a distant bucket idles the GPU you are paying for. The fix is to hand the finished frames to a *separate* CPU-side encode-and-upload task (a thread pool or a second queue) so the GPU worker can immediately pull the next job. This is the same decoupling logic as the API-versus-worker split, applied one level down: keep the expensive resource (the GPU) doing only the work that needs it, and push I/O off its critical path. On a high-throughput render farm this single change can lift GPU utilization several points, which translates directly into lower cost-per-clip via the utilization term from section 7.

**Health checks and the warm-model liveness trap.** A naive liveness probe that just checks the process is alive will keep routing jobs to a worker whose GPU has silently faulted (an uncorrectable ECC error, a driver hang) — the process is up but the model is dead. A proper readiness check runs a *tiny* generation periodically and verifies it produces frames, so a wedged GPU is detected and the worker is pulled from the pool before more jobs land on it. The cost is a few GPU-seconds per health interval, cheap insurance against a black-hole worker silently failing every request routed to it.

## 9. Putting it together: a before-after on named hardware

Here is the consolidated result — the same 5-second 720p, 81-frame Wan-2.1-14B-class workload, on two named GPUs, as you stack the levers. Numbers are approximate and illustrative of the *shape* of the trade (exact figures depend on model build, driver, and drivers' allocator behavior), but the relationships are what matter and they are robust.

| Configuration | Peak VRAM | Seconds/clip (A100 80GB) | \$/clip (A100 @ \$2/hr) | Fits 4090 24GB? |
| --- | --- | --- | --- | --- |
| Naive fp16, full-clip decode | ~40 GB | OOM on 4090; ~90s on A100 | ~\$0.050 | No (OOM at decode) |
| + VAE tiling/chunked decode | ~30 GB | ~95s | ~\$0.053 | No (weights too big) |
| + model CPU offload | ~30 GB resident lower | ~100s | ~\$0.056 | Marginal |
| + fp8 denoiser weights | ~18 GB | ~70s | ~\$0.039 | Yes |
| + step distillation (50→4) | ~18 GB | ~12s | ~\$0.007 | Yes |
| + caching + torch.compile | ~18 GB | ~7s | ~\$0.004 | Yes |

Read the table top to bottom as the production journey. The naive load OOMs on the affordable card and costs five cents on the expensive one. VAE tiling stops the decode OOM but the weights still do not fit a 4090. Offload helps resident memory but not the fundamental weight size. fp8 is the inflection point: it halves the weights, fits the 4090, *and* speeds up the A100 — the rare lever that improves every column at once. Then the speed levers (distillation, caching, compile) — which are the [real-time post](/blog/machine-learning/video-generation/efficient-and-real-time-video-generation)'s domain — collapse the seconds and therefore the dollars by an order of magnitude. The serving levers in *this* post (tiling, offload, quantization-for-memory, routing) are what made the model *runnable* and *cheap-to-host*; the speed levers are what made it *fast*. You need both, and they live in different posts because they are different engineering problems with different failure modes.

The throughput dimension overlays on top: every row's seconds/clip is the batch-1 figure, and batching with VAE slicing roughly halves the cost-per-clip again on a throughput-oriented queue, at the price of per-request latency, exactly as section 6's trade describes. The fully-optimized, batched, distilled configuration is what makes a video product economically viable — sub-cent clips at high throughput — and every lever you drop adds back a multiple of cost.

## 10. Case studies and real numbers

A few grounded data points from shipped models and the literature, to anchor the earlier claims in something real. Where I am not certain of an exact figure I say so.

**HunyuanVideo on consumer hardware.** HunyuanVideo's 13B model, in its original fp16 form, needs on the order of 60GB+ of VRAM for a high-res clip — firmly data-center territory. The community quantized and offloaded path (GGUF/fp8 builds plus block-swap offload in ComfyUI) is what brought it onto 24GB consumer cards at all, and the reported HunyuanVideo-1.5 figure of a ~75-second generation on a 4090 (approximate, for a short clip at reduced settings) is *only* reachable with the memory levers in this post — without quantization and tiling the model simply does not load on the card, regardless of how long you are willing to wait. This is the cleanest real-world illustration that serving levers are the price of entry, not an optimization.

**Wan 2.1 14B and the offload reality.** Wan-2.1-14B in fp16 wants ~28GB for the transformer alone, so the official and community guidance for running it on anything smaller than an A100/H100 leans on `enable_model_cpu_offload`, fp8 quantization via `optimum-quanto` or GGUF, and VAE tiling — exactly the stack in sections 2–4. The reported result is that a 720p clip becomes feasible on a 24GB 4090 with these enabled, at a latency cost from the offload swaps. This matches the worked example in section 3 closely.

**The VAE-decode spike is a documented, common failure.** The `diffusers` docs ship `enable_vae_tiling`, `enable_vae_slicing`, and `decode_chunk_size` precisely because the VAE decode OOM is the single most common issue users hit with video pipelines — the library's own troubleshooting points at the decode as the memory peak for high-resolution or long clips. The existence of these dedicated flags is itself the evidence that the decode, not the denoiser, is the wall. (See the [3D-VAE post](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression) for *why* the decode is so heavy: it is reconstructing the full spatial and temporal resolution the encoder threw away.)

**fp8 on Ada/Hopper as a speed win.** The reported behavior of fp8 weight quantization on Ada (4090) and Hopper (H100) tensor cores is a simultaneous ~2× memory reduction and a meaningful latency *reduction* (often 20–40%, approximate and workload-dependent) because the hardware does native fp8 matmul, with a quality cost that is typically below the noise floor of FVD on a sensible eval set. This is why fp8 is the recommended serving default for video DiTs on those cards rather than int8 — you get the memory win and a speed win together, where int8 gives memory but no speed advantage on these tensor cores.

These all point at the same conclusion: in 2024–2026, the difference between "this model is a research artifact that needs 8×H100" and "this model serves users on a single rentable card" is almost entirely the serving stack — quantization, offload, tiling, and the queue-plus-warm-worker infrastructure — not the model architecture. The architecture is shared and open; the serving is where the engineering lives.

## 11. When to reach for each lever (and when not to)

A decisive section, because the failure mode here is over-engineering — applying every lever to a workload that needed two of them.

- **Always enable VAE tiling/chunking and SDPA/FlashAttention.** They are free or near-free and they prevent the most common OOM. There is no workload where you should turn them off. Start here.
- **Enable model CPU offload when resident weights are your binding constraint** — i.e. the model does not fit in fp16/bf16 on your card. Skip it if the model already fits comfortably; the PCIe swaps add latency for no benefit when you have VRAM to spare.
- **Quantize to fp8 by default on Ada/Hopper.** It cuts the biggest memory term and usually speeds things up; the quality cost is below most product bars. Measure it once on your eval set and then trust it.
- **Go to int4 only when fp8 still does not fit** and you have verified the quality cost is acceptable for your use case. Do *not* reach for int4 reflexively for the memory — on a sensitive video DiT it can cost visible motion-smoothness, and that flicker is exactly what users notice. fp8-fits beats int4-just-because.
- **Use sequence parallelism only when the denoiser activations exceed one card** — long or high-res clips on NVLink-connected GPUs. For a standard 5-second 720p clip it is wasted money and complexity; the single-card levers are enough. And remember it solves *memory*, not the long-video *quality* problem.
- **Batch large for throughput queues, batch small for interactive products.** Do not default to large batches on a latency-sensitive product — every user waits for the slowest in the batch. Do not default to batch-1 on a render farm — you are leaving half your throughput (and money) on the table.
- **Stream frames only if your model is causal/autoregressive.** Time-to-first-frame is a real win for interactive UX, but a non-causal diffusion model produces all frames at once and cannot stream — do not try to bolt streaming onto a model that does not generate in temporal order.
- **Right-size the GPU before you optimize compute.** The cheapest lever is often "the memory tricks let this run on a 4090 instead of an A100", cutting the *rate* term directly. Check whether you can drop to a cheaper card before you spend engineering time shaving seconds.

The meta-rule: the levers attack different terms of the budget — weights, activations, decode spike, rate, seconds — and the right configuration is the *minimal* set that clears your binding constraint. Profile first (the `max_memory_allocated` snippet), find which term is the wall, and apply the lever that hits *that* term. Do not apply all of them; apply the ones the memory model tells you to.

## 12. Key takeaways

- **The VAE decode is the memory wall, not the denoiser.** Peak VRAM is weights plus $T\cdot H\cdot W$ activations plus a one-shot 3D-VAE decode spike, and that decode — which materializes every frame at full resolution at once — is what OOMs a 24GB card at the very end of a render.
- **Chunked temporal decode and spatial tiling are mandatory, not optional.** They make the decode VRAM nearly independent of clip length and resolution, trading a few percent of latency for the difference between OOM and success. Turn them on for every video workload.
- **The lever stack fits a 14B model on a 24GB card:** SDPA/FlashAttention (always), VAE tiling (always), model CPU offload (when weights are the constraint), fp8 quantization (cuts the biggest term, often speeds things up). Apply the minimal set the memory model demands.
- **fp8 is the serving default on Ada/Hopper.** It halves the weights *and* speeds up the matmul, at a quality cost below the noise floor of FVD. Reserve int4 for when fp8 still does not fit and you have measured the motion-quality cost.
- **Sequence parallelism is for long clips only.** When the denoiser's own activations exceed one GPU, split the spacetime token sequence across NVLink-connected cards; it cuts activations near-linearly but replicates weights, so combine it with quantization. It solves memory, not the long-video coherence problem.
- **Batching is the latency-versus-throughput dial.** Large batches with VAE slicing maximize clips-per-hour and minimize cost-per-clip for render queues; small batches minimize the latency interactive users feel. Pick by what you are paid to optimize.
- **Cost per clip is GPU-seconds times rate, full stop.** Distillation, caching, quantization, and batching all multiply *through the seconds* term; compounded, they take a 5-second 720p clip from ~5 cents to under half a cent. Right-sizing the GPU cuts the rate term directly.
- **Serve with a queue and warm workers.** Never run the model on the API request path; enqueue and return a job ID, let warm GPU workers (model loaded and `torch.compile`-warmed at startup) pull jobs. Cold start, VRAM fragmentation, and OOM-on-long-requests are the real production failures — warm pools, `expandable_segments`/worker recycling, and size-based routing are the fixes.

If you are assembling all of this into a real product — choosing models, fine-tuning, picking samplers and serving infra end to end — the [capstone playbook](/blog/machine-learning/video-generation/building-with-video-generation-the-playbook) ties the serving stack here to the model-selection and pipeline decisions, and the [open-frontier models post](/blog/machine-learning/video-generation/the-open-video-frontier-wan-hunyuanvideo-cogvideox) is where the self-hosting hardware reality this post operationalizes gets its model-by-model detail.

## Further reading

- Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (2022) — why attention memory is $O(N)$ not $O(N^2)$, the lever that makes million-token video attention runnable at all.
- Blattmann et al., *Stable Video Diffusion* (2023) — the `decode_chunk_size` temporal-decode knob in context; the I2V serving baseline.
- Wan team, *Wan 2.1 / 2.2 technical report* (2025) — a 14B open video DiT with the causal-3D-VAE + flow-matching recipe, and the offload/quantization reality of self-hosting it.
- Hunyuan team, *HunyuanVideo / HunyuanVideo-1.5 report* (2024–2025) — a 13B model and the quantization-plus-offload path that brought it to 24GB consumer cards.
- Jacob et al. and the SmoothQuant / GPTQ / AWQ line — the quantization mechanics; see also the edge-ai series posts on [weight-only](/blog/machine-learning/edge-ai/llm-quantization-weight-only-gptq-awq) and [activation quantization](/blog/machine-learning/edge-ai/llm-quantization-activations-smoothquant-kv-cache).
- 🤗 `diffusers` documentation — *Reduce memory usage* and the video-pipeline guides: `enable_model_cpu_offload`, `enable_vae_tiling`, `enable_vae_slicing`, `decode_chunk_size`, the canonical serving-memory reference.
- The `xDiT` project — sequence/context-parallel inference for diffusion transformers, the practical path to multi-GPU long-video rendering.
- Within this series: the foundation [why video generation is hard](/blog/machine-learning/video-generation/why-video-generation-is-hard), the [3D-VAE compression post](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression) (why the decode is so heavy), the sibling [efficient and real-time video generation](/blog/machine-learning/video-generation/efficient-and-real-time-video-generation) (the seconds-per-clip war), [spatiotemporal attention patterns](/blog/machine-learning/video-generation/spatiotemporal-attention-patterns) (where the activation memory comes from), and the [capstone playbook](/blog/machine-learning/video-generation/building-with-video-generation-the-playbook).
