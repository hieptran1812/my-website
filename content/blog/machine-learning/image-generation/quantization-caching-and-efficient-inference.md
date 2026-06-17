---
title: "Quantization, Caching, and Efficient Diffusion Inference"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Make a 12-billion-parameter diffusion model fit and fly on a single consumer GPU: the activation-outlier problem and SVDQuant's low-rank fix for 4-bit weights and activations, why naive int8 breaks diffusion when it barely scratches LLMs, DeepCache and TeaCache feature reuse, attention and VAE optimizations, and the torch.compile/offload runtime tricks — with measured VRAM and latency on an RTX 4090."
tags:
  [
    "image-generation",
    "diffusion-models",
    "quantization",
    "svdquant",
    "feature-caching",
    "efficient-inference",
    "fp8",
    "int4",
    "generative-ai",
    "deep-learning",
    "pytorch",
  ]
category: "machine-learning"
subcategory: "Image Generation"
author: "Hiep Tran"
featured: true
readTime: 58
image: "/imgs/blogs/quantization-caching-and-efficient-inference-1.png"
---

You have a FLUX.1-dev checkpoint, a single RTX 4090 with 24 GB of memory, and a product manager who wants 1024×1024 images served at under a second each. You load the model in fp16, watch nvidia-smi climb to 23.4 GB the instant the transformer and the two text encoders and the VAE are all resident, and discover that one image takes 6.1 seconds at 50 steps. The card is one cache allocation away from an out-of-memory crash, the latency is 6× over budget, and you haven't even handled a second concurrent request. This is the gap between "the model works in the paper" and "the model ships," and closing it is a systems problem, not a modeling one.

There are exactly four levers you can pull to close that gap, and we laid out the frame in [why diffusion is slow and how to fix it](/blog/machine-learning/image-generation/why-diffusion-is-slow-and-how-to-fix-it): take **fewer steps**, use a **smaller or faster network**, compress the **latent more aggressively**, or **reuse computation** you already did. The previous two posts in this track — [consistency models and few-step generation](/blog/machine-learning/image-generation/consistency-models-and-few-step-generation) and [distribution matching and adversarial distillation](/blog/machine-learning/image-generation/distribution-matching-and-adversarial-distillation) — attacked the first lever, cutting 50 steps down to 1–4. This post attacks the other three, the ones that are *orthogonal* to step count: you can quantize a model to 4 bits, cache its features across steps, and fuse its attention kernels, and every one of those wins *stacks on top of* a 4-step Turbo model. They are the difference between a demo and a service.

![A before and after comparison showing FLUX in fp16 using about twenty-three gigabytes of memory at six seconds per image versus an SVDQuant four-bit version using about seven gigabytes at under two and a half seconds.](/imgs/blogs/quantization-caching-and-efficient-inference-1.png)

Figure 1 is the destination. On the left, the fp16 baseline: roughly 23 GB of VRAM, 6.1 seconds per image, one allocation from disaster. On the right, the same FLUX.1-dev quantized to 4-bit weights *and* activations with SVDQuant and served with feature caching: roughly 7 GB, 2.4 seconds, comfortable headroom for a second request. Same model, same prompt, a quality delta you'd struggle to see in a blind test. By the end of this post you will be able to: explain *why* activation outliers make diffusion quantization hard and how SVDQuant's low-rank branch absorbs them to enable true 4-bit inference; explain why a naive int8 that LLMs shrug off can wreck a diffusion model; wire up DeepCache and TeaCache feature caching with a real cache hook; turn on SDPA/FlashAttention, VAE tiling and slicing, and `AutoencoderTiny`; and apply `torch.compile`, CUDA graphs, channels-last, and CPU offload — then read a before→after table on a named GPU and know which lever to pull first. We'll keep tying back to the series spine: the **generative trilemma** (quality × diversity × speed) and the **diffusion stack** (data → VAE latent → noising → denoiser → sampler → guidance → image). Quantization and caching change the *cost* of running that stack without changing what it computes.

A note on scope and on borrowing. Quantization theory — what a scale and zero-point are, the difference between symmetric and asymmetric, per-tensor versus per-channel, the int4 numerics and the NF4 datatype — is a deep topic in its own right, and the Edge AI track on this blog already derives it from first principles. Rather than re-derive it, I'll link out where it earns a full treatment ([quantization from first principles](/blog/machine-learning/edge-ai/quantization-from-first-principles) for the numerics, [LLM quantization: activations, SmoothQuant, and KV cache](/blog/machine-learning/edge-ai/llm-quantization-activations-smoothquant-kv-cache) for the outlier story in language models, [below 8 bits](/blog/machine-learning/edge-ai/sub-8-bit-int4-ternary-and-binary-networks) for the int4 accuracy cliff) and focus here on what's *specific to diffusion*: the activation-outlier problem in a denoiser that you run dozens of times, the low-rank trick that fixes it, and the timestep redundancy that makes caching almost free.

## 1. Where the time and memory actually go

Before you optimize anything, measure where the cost lives, because the wrong optimization is worse than none. A modern text-to-image pipeline has four cost centers, and they do not contribute equally.

The **denoiser** — the U-Net or DiT — is the dominant cost, and it dominates twice over. On memory: FLUX.1-dev's transformer is 12 billion parameters, which at fp16 (2 bytes each) is 24 GB of weights *by itself* before you account for activations, the KV-style attention buffers, or the optimizer state (we're doing inference, so no optimizer, but the weights alone overshoot a 4090). On compute: you run the denoiser *once per sampling step*, so a 50-step generation invokes the 12B-parameter network 50 times. If a single forward pass is 120 ms, that's 6 seconds, and the denoiser is essentially the whole bill. This is the structural fact that makes diffusion expensive and that makes both step-reduction and per-step-reduction worthwhile: **the denoiser runs in a loop**, so any saving on a single forward pass multiplies by the step count.

The **text encoders** — CLIP and, for SD3/FLUX, a large T5-XXL (4.7B parameters) — run *once* at the start, not per step. They cost memory while resident (T5-XXL is ~9.5 GB in fp16) but contribute little to latency because they fire a single time. This asymmetry matters for your optimization strategy: quantize or offload the text encoders to *save memory*, but don't expect them to move the latency needle.

The **VAE decoder** runs once at the end to turn the final latent into pixels, and it has a sneaky cost profile: cheap in FLOPs but brutal in *peak activation memory* at high resolution, because decoding a 128×128×16 latent to a 1024×1024×3 image inflates the tensor by the spatial-upsampling factor, and at 2K or 4K the decode activations alone can exceed your entire denoiser's footprint. The VAE is the reason your 4090 OOMs at 4K even though the latent is tiny — and it's why VAE tiling exists (section 7).

The **sampler and guidance** logic — the scheduler step, the classifier-free-guidance combination — is essentially free arithmetic on the latent. But note one thing: classifier-free guidance runs the denoiser *twice per step* (once conditional, once unconditional) unless you've distilled the guidance away, so CFG silently doubles your denoiser bill. This is the most under-appreciated 2× in the whole pipeline. The CFG prediction is $\hat\epsilon = \epsilon_\text{uncond} + w\,(\epsilon_\text{cond} - \epsilon_\text{uncond})$, which needs *both* a conditional and an unconditional forward pass through the 12B network at every step — so a "50-step" generation with CFG is really *100* denoiser invocations. Two ways to reclaim it: batch the two passes together (run cond and uncond as a batch of 2, which is what `diffusers` does, so they share a kernel launch but still do 2× the matmul work), or use a *guidance-distilled* model where the guidance is baked into a single forward pass — most Turbo/LCM/distilled checkpoints do exactly this, which is part of why they're so much faster than the step-count reduction alone would suggest. When you see a distilled model claim "4 steps," remember it's also usually killed the CFG 2×, so the real speedup over a 50-step *guided* baseline is closer to 25× than 12×.

So the priority order writes itself. Attack the denoiser's per-pass cost (quantization, attention fusion, compilation) because it multiplies by step count. Attack the step count (distillation, caching) because it multiplies the per-pass cost. Manage the VAE's peak memory (tiling) only when you push resolution. And offload the text encoders to reclaim VRAM when you're memory-bound. The rest of this post takes those in turn, starting with the heaviest hammer: quantization.

It's worth putting a back-of-the-envelope number on the FLUX bill so the rest of the post has a concrete anchor. The transformer is 12B parameters; a single forward pass at 1024×1024 processes roughly 4,096 image tokens plus ~512 text tokens through 57 transformer blocks, and the dominant cost is the two big matmuls per block (the attention projections and the MLP) plus the attention itself. The MLP alone is on the order of $2 \cdot N_\text{params} \cdot T$ FLOPs for sequence length $T$, so a single pass is in the low tens of TFLOPs and the attention adds a quadratic-in-$T$ term that grows fast at high resolution. A 4090 delivers ~80 TFLOPs of usable bf16 throughput in practice (well below its peak, because attention and the small matmuls are memory-bandwidth-bound, not compute-bound), which lands a single forward pass around 100–130 ms — and you do that 50 times for ~6 s. Two facts jump out of that arithmetic, and they steer everything that follows. First, the model is *partly memory-bandwidth-bound*, not purely compute-bound, which is why weight-only quantization (less data to move) helps latency more than the FLOP count alone predicts, and why fusing kernels (fewer round-trips to HBM) is a real win. Second, the per-pass cost is large enough that *anything* that lets you skip or cheapen a pass — caching, distillation — pays off enormously. The whole optimization game is a fight over those two quantities: bytes moved per pass, and passes per image.

One more structural note before we dig in, because it changes which levers even apply: the *architecture* of the denoiser matters. A U-Net (SD1.5, SDXL) has skip connections and a clear deep/shallow split, which is exactly what DeepCache exploits. A DiT (SD3, FLUX, PixArt) is a flat stack of identical transformer blocks with no skip connections and no deep/shallow asymmetry — so U-Net caching tricks don't transfer, and the DiT-native caching story (TeaCache, residual caching) is different. Quantization, by contrast, applies to both because both are matmul-dominated. I'll flag the architecture dependence as we go, because reaching for a U-Net trick on a DiT (or vice versa) is a common and frustrating way to waste an afternoon.

## 2. Quantization, and why diffusion is harder than it looks

Quantization is the single biggest memory lever, and the idea is simple: store and compute the weights (and ideally the activations) in fewer bits. A weight tensor in fp16 takes 2 bytes per element; in int8, 1 byte (2× smaller); in int4, half a byte (4× smaller). The mechanism is a uniform affine map. For a tensor with values in $[\beta_\min, \beta_\max]$, pick a scale $s$ and zero-point $z$ so that a real value $w$ maps to an integer

$$
q = \mathrm{round}\!\left(\frac{w}{s}\right) + z, \qquad \hat{w} = s\,(q - z),
$$

where $\hat w$ is the dequantized approximation. For symmetric int4 you drop the zero-point and pick $s = \max|w| / 7$ (since signed int4 spans $[-8, 7]$). The quantization error per element is bounded by half a step, $|w - \hat w| \le s/2$, so the error scales with the *range* of the tensor divided by the number of representable levels. That last clause is the whole story of why diffusion quantization is hard. I'm going to lean on the Edge AI track for the full numerics — see [quantization from first principles](/blog/machine-learning/edge-ai/quantization-from-first-principles) — and focus on the part that bites diffusion specifically.

Here's the trap. **Weight-only** quantization (keep activations in fp16, quantize just the weights, dequantize on the fly before each matmul) is relatively easy and well-understood: GPTQ and AWQ do it for LLMs at 4-bit with tiny quality loss, and it cuts your *memory* by ~4×. But weight-only quantization does *not* speed up compute much, because the matmul still runs in fp16 — you saved storage and bandwidth, not arithmetic. To get real *speed*, you must also quantize the **activations** so the matmul itself runs in low-precision integer or FP8 arithmetic on the tensor cores. And activation quantization is where diffusion falls off a cliff.

The reason is **outliers**. When you quantize a tensor, the scale $s$ is set by the *maximum* magnitude — a few extreme values stretch the range, and since you have only 16 levels in int4, every other value gets squeezed into a coarse grid. To see why this is so destructive, write the quantization error as a fraction of the signal. If the signal of interest has typical magnitude $\sigma$ and the outliers have magnitude $M \gg \sigma$, then the int4 step is $s = M / 7$ and the error on a typical value is up to $s/2 = M/14$. The *relative* error on the signal is therefore $\sim M / (14\sigma)$ — it grows linearly with how much bigger the outliers are than the signal. With $M = 20\sigma$ (a mild outlier), the relative error is already ~140%: the signal is *gone*, buried in rounding noise, because the few levels you have are all spent covering the gap up to the outlier. This is the entire problem in one inequality, and it explains why granularity (per-channel, per-group scales) helps — finer scales let outlier channels have their own large $s$ while signal channels keep a small $s$ — but doesn't fully solve it when the outliers and signal share a channel.

In language models this problem is real but localized: outliers appear in specific, consistent channels, and tricks like SmoothQuant (migrate the activation outlier scale into the weights) or keeping a handful of channels in fp16 largely fix it. We covered exactly that in [LLM quantization: activations, SmoothQuant, and KV cache](/blog/machine-learning/edge-ai/llm-quantization-activations-smoothquant-kv-cache). Diffusion models are worse for three compounding reasons.

First, the activation distributions in a diffusion denoiser are *heavier-tailed* than in an LLM, especially in the modulation/normalization paths (AdaLN, the time and text conditioning) where a few channels carry very large values — the AdaLN modulation produces scale and shift parameters that can swing wide, and those large values propagate. Second — and this is the part people miss — diffusion runs the network across a *range of noise levels* $t$, and the activation statistics shift dramatically with $t$. The activations at $t \approx 1000$ (almost pure noise) look nothing like the activations at $t \approx 0$ (almost clean): the input latent's variance, the magnitude of the predicted noise, even which channels light up all change with the noise level. A single static quantization scale, calibrated on activations averaged over timesteps, has to cover all of them, so it's mis-calibrated at most individual timesteps — too coarse where the activations are small, clipping where they're large. (This is why some diffusion-quantization methods like Q-Diffusion and PTQD use *timestep-aware* calibration, collecting separate statistics across the noise schedule, but that adds complexity and still doesn't fully solve the outlier problem.) Third, errors *accumulate across steps*. An LLM quantization error affects one token's logits and stops there; a diffusion quantization error gets fed back into the next denoising step as the input latent, and over 20–50 steps small per-step errors compound. The mechanism is a feedback loop: the error in $\hat x_{t-1}$ becomes part of the input to the network at step $t-1$, which produces a slightly-wrong $\hat x_{t-2}$, and so on — a small bias per step integrates into a visible drift (color shift, washed saturation, smeared fine texture) by the final image. The combination — heavier tails, timestep-varying statistics, and error accumulation through a feedback loop — is why naive int8 weight-and-activation quantization, which an LLM barely notices, can push a diffusion model's FID up by several points and produce visibly degraded images.

There's a deeper reason FP8 specifically rescues this where int8 cannot, and it's worth deriving because it's the cleanest argument in the section. Integer quantization spaces its levels *uniformly*: the gap between adjacent representable values is constant, $s$, everywhere. Floating-point quantization spaces its levels *logarithmically*: the gap between adjacent values is proportional to the value's magnitude (a constant *relative* precision set by the mantissa bits). For a distribution with a few large outliers and a mass of small signal values, the floating-point grid is exactly the right shape — it puts *dense* levels near zero where the signal lives and *sparse* levels out near the outliers where you only need to represent magnitude, not fine distinctions. FP8 `e4m3` has 4 exponent bits, giving it a dynamic range of roughly $2^{-9}$ to $448$ — wide enough to hold an outlier and a small value *simultaneously* with reasonable relative precision on both. Int8 has to choose: either set $s$ small (clip the outliers) or set $s$ large (lose the signal). It cannot win at a heavy-tailed distribution, and a diffusion denoiser's activations are exactly that. This is the formal version of "format beats bit-count at 8 bits": the floating-point grid's *shape* matches the data's *shape*, and int8's uniform grid does not.

![A four by four matrix comparing fp16, FP8, naive int8, and SVDQuant int4 across weight bits, activation bits, memory savings, and quality delta, showing naive int8 breaks while FP8 and SVDQuant stay near baseline.](/imgs/blogs/quantization-caching-and-efficient-inference-3.png)

Figure 3 makes the contrast concrete. The fp16 baseline is the reference. **FP8** (the `e4m3` format, 4 exponent bits and 3 mantissa bits) keeps a *floating-point* representation, so it has a much wider dynamic range than int8 at the same bit width — it represents large outliers gracefully because the exponent stretches, and small values keep relative precision. That's why FP8 weight-and-activation quantization holds diffusion quality close to baseline (a tiny FID delta) at 2× memory savings, while *integer* int8 with the same 8 bits but a *linear* grid gets destroyed by the same outliers (a large FID jump — "broken" in the figure). The lesson is sharp and counterintuitive: at 8 bits, the *format* matters more than the *bit count* for diffusion. FP8's floating-point grid is forgiving where int8's linear grid is not. And then the last row — SVDQuant int4 — is the surprise: it gets *4-bit* weights and activations to a quality delta smaller than naive int8 manages at 8 bits, by handling the outliers structurally rather than hoping the format absorbs them. That's the next section.

#### Worked example: why int8 hurts diffusion but not your LLM

Take a Linear layer whose activation has 1,022 values in $[-2, 2]$ and two outlier values at $+40$. Symmetric int8 over the whole tensor sets $s = 40/127 \approx 0.315$. The 1,022 "normal" values now snap to a grid spaced 0.315 apart — so a value of 0.5 and a value of 0.7 both round to the same integer (1), and you've lost the distinction. The signal-carrying values, which live in $[-2, 2]$, are represented with effectively *7 levels* instead of 256, because the outliers ate the range. In an LLM this Linear feeds one token's logits and the damage is bounded. In a diffusion denoiser this Linear's output feeds the next ResBlock, whose output feeds the sampler, whose output is the latent for step $t-1$, which goes back through the *same* layer — so the rounding error is injected 50 times and integrated. The fix is not a better int8 scale; it's to stop the two outliers from setting the range at all. Hold that thought.

## 3. SVDQuant: absorbing outliers into a low-rank branch

The key 2024 result for diffusion quantization — the one that finally made 4-bit weights *and* activations practical — is **SVDQuant** (Li et al., 2024), shipped as the **Nunchaku** inference engine. The core move is elegant and worth understanding deeply because it's the cleanest demonstration of "structure the problem so the hard part lives somewhere cheap." 

The setup: a Linear layer computes $Y = X W$ where $X$ is the activation and $W$ the weight. We want both $X$ and $W$ in 4-bit so the matmul runs on int4 tensor cores. The obstacle, from section 2, is the outliers — in both $X$ and $W$ — that blow up the quantization range. SVDQuant's idea: don't try to quantize the outliers, *remove* them into a separate, cheap, high-precision path, and quantize only what's left.

It happens in two stages. **Stage one — smoothing.** First, migrate the activation outliers into the weights, the SmoothQuant move: rescale $X \to X \cdot \mathrm{diag}(\lambda)^{-1}$ and $W \to \mathrm{diag}(\lambda)\, W$ for a per-channel factor $\lambda$ chosen so the products are unchanged ($X W$ is invariant) but the activation's outlier channels are tamed and the burden shifts to $W$. Now the *weight* carries the concentrated outliers — but a weight matrix's outliers are static and predictable, which is exactly the structure the next stage exploits.

**Stage two — low-rank absorption.** Decompose the smoothed weight as

$$
W \;\approx\; \underbrace{L_1 L_2}_{\text{16-bit, rank } r} \;+\; \underbrace{W_\text{res}}_{\text{4-bit}},
$$

where $L_1 \in \mathbb{R}^{d \times r}$ and $L_2 \in \mathbb{R}^{r \times k}$ form a low-rank (rank $r = 16$ or $32$) branch kept in 16-bit precision, and $W_\text{res} = W - L_1 L_2$ is the residual that gets quantized to 4-bit. The decomposition is computed via SVD: take the top-$r$ singular components of $W$ (after smoothing) into $L_1 L_2$. The magic is that **the outliers live in the top singular components**. The dominant directions of the weight — the ones with the largest singular values, which are also where the large-magnitude outlier energy concentrates — get captured by the rank-$r$ branch. What remains in $W_\text{res}$ is the bulk of the matrix with the spikes shaved off: a *flat, well-conditioned* residual whose values cluster tightly, so its int4 grid (16 levels) is spent on the signal instead of being stretched by spikes. The residual quantizes cleanly because the hard part — the outliers — was lifted out into the 16-bit low-rank branch.

![A dataflow graph showing an activation with outliers being smoothed, then split into a sixteen-bit low-rank branch carrying the outliers and a four-bit residual path, recombined into the full-quality output.](/imgs/blogs/quantization-caching-and-efficient-inference-2.png)

Figure 2 is the data flow. The activation enters with outliers; smoothing migrates the outlier scale; the smoothed weight splits into a 16-bit rank-32 low-rank branch ($L_1 L_2$) that carries the outlier energy and a 4-bit residual that's now flat enough to quantize. At inference, you compute *both* paths and sum: $Y = X (L_1 L_2) + X_\text{q} W_{\text{res,q}}$. The 4-bit GEMM is the main, expensive path and runs fast on int4 tensor cores; the 16-bit low-rank GEMM is *tiny* — its cost is $O(d r)$ and $O(r k)$ with $r = 32$, a few percent of the full matmul — so you pay a few-percent overhead for the branch and get the 4-bit speedup on everything else. The output is full-quality because the outliers were never forced through the 4-bit grid.

Why does this preserve quality where naive int4 destroys it? Because quantization error is bounded by $s/2$ and $s$ is set by the *range* of what you're quantizing. By construction, the residual $W_\text{res}$ has had its largest components removed, so its range is dramatically smaller than $W$'s — the same 16 int4 levels now cover a tight distribution, the per-element error shrinks proportionally, and the part you *couldn't* afford to lose (the outlier directions) sits safely in 16-bit. It's the same insight as keeping a few channels in fp16, but done *structurally* via SVD rather than by hand-picking channels, and crucially it handles *both* weight and activation outliers because the smoothing stage moves the activation outliers into the weight first. That's why the result holds: SVDQuant reported FID on par with fp16 (a fraction of a point) for SDXL, PixArt-Σ, and FLUX at W4A4 (4-bit weights, 4-bit activations), where every prior W4A4 attempt had failed.

There's a beautiful subtlety in *why the SVD specifically* lifts out the outliers, and it rewards a careful look. The Eckart–Young theorem says the best rank-$r$ approximation of a matrix, in the Frobenius (and spectral) norm, is exactly its top-$r$ singular components. So $L_1 L_2$ = top-$r$ SVD is *provably* the rank-$r$ matrix that captures the most of $W$'s energy. Now, why does "most energy" coincide with "the outliers"? Because in these weight matrices the large-magnitude structure is *low-rank*: the outlier directions are a handful of dominant modes, not spread uniformly across the spectrum. When a few directions carry disproportionate energy (large singular values), the SVD's top components grab exactly those, and the residual is what's left after you subtract the dominant modes — a matrix whose singular values have all been knocked down to the tail of the spectrum, hence flat and well-conditioned. The method works *because* outliers in trained weights are empirically low-rank; if the outliers were full-rank (spread evenly), no small low-rank branch could capture them and the trick would fail. They aren't, so it doesn't. This is the load-bearing empirical fact, and it's worth internalizing: SVDQuant trades a *little* rank (a 16-bit branch of rank 32) for a *lot* of range reduction, and that trade is favorable precisely because the hard-to-quantize part of a weight matrix is concentrated in a few directions.

#### Worked example: the residual range after low-rank removal

Suppose the smoothed weight $W$ has singular values $\sigma_1 = 50, \sigma_2 = 30, \sigma_3 = 8, \sigma_4 = 7, \dots$ tapering down, with the top two carrying the outlier energy. Take $r = 2$: $L_1 L_2$ captures $\sigma_1, \sigma_2$, and $W_\text{res}$ has effective spectral norm $\sigma_3 = 8$. The residual's dynamic range dropped from $\sim 50$ to $\sim 8$, a **6× tighter range**, so the int4 step $s = \max|W_\text{res}|/7$ is ~6× smaller and the per-element error is ~6× smaller too. The 16-bit branch costs $2 \cdot d \cdot 2 / (d \cdot k)$ of the full matmul FLOPs — for $d = k = 3072$ and $r = 2$ that's $\sim 0.13\%$ overhead; even at the practical $r = 32$ it's only ~2%. You bought a 6× cleaner residual for a 2% compute tax. That trade is why the method works and why it's the diffusion-quantization result of 2024–2025.

Here's how you'd actually load and run a Nunchaku/SVDQuant 4-bit FLUX in practice. A practical detail that trips people up the first time: not every layer should be quantized to 4-bit, and good quantization tooling knows this. The very first and very last layers of the network (the patch embedding that ingests the latent, the final projection that produces the noise prediction) and the normalization-adjacent modulation layers are *sensitivity hotspots* — quantizing them aggressively does outsized damage because errors there propagate through every block, and they're a tiny fraction of the parameters anyway. The standard recipe (in SVDQuant, in mixed-precision quantization generally — see [mixed precision and sensitivity analysis](/blog/machine-learning/edge-ai/mixed-precision-and-sensitivity-analysis)) keeps these few sensitive layers in higher precision (FP8 or bf16) and quantizes only the bulk transformer blocks to int4. You give up almost no memory (the sensitive layers are small) and you keep the quality that a uniform "quantize everything to 4-bit" would throw away. This is the same lesson as the low-rank branch at a coarser grain: spend your precision budget where the data is sensitive, and be aggressive everywhere else. A blanket "int4 the whole model" is the naive move that gives quantization a bad name; selective precision is what makes it work.

The Nunchaku project distributes pre-quantized checkpoints; you swap the transformer for the quantized one and the rest of the `diffusers` pipeline is unchanged:

```python
import torch
from diffusers import FluxPipeline
from nunchaku import NunchakuFluxTransformer2dModel

# Load the SVDQuant 4-bit transformer (weights + activations int4,
# low-rank branch kept in bf16). int4 == "svdq-int4".
transformer = NunchakuFluxTransformer2dModel.from_pretrained(
    "mit-han-lab/svdq-int4-flux.1-dev"
)

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    transformer=transformer,          # the only swapped component
    torch_dtype=torch.bfloat16,
).to("cuda")

image = pipe(
    "a photorealistic red fox in a snowy forest, golden hour",
    num_inference_steps=50,
    guidance_scale=3.5,
    height=1024, width=1024,
).images[0]
image.save("fox.png")
# Transformer VRAM drops from ~24 GB (bf16) to ~6.5 GB (int4 + low-rank).
```

If you don't have a pre-quantized checkpoint and want weight-only 4-bit (memory win without the activation-quantization speedup), `bitsandbytes` NF4 or `optimum-quanto` plug straight into `diffusers`:

```python
import torch
from diffusers import FluxTransformer2DModel, FluxPipeline
from diffusers import BitsAndBytesConfig

# NF4: 4-bit NormalFloat weights, activations stay bf16 (weight-only).
# Saves ~4x memory on the weights; matmul still runs in bf16.
nf4 = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
transformer = FluxTransformer2DModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    subfolder="transformer",
    quantization_config=nf4,
    torch_dtype=torch.bfloat16,
)
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()   # text encoders + VAE shuttle on demand
```

For the middle ground — the *easy 2×* before you reach for SVDQuant's machinery — FP8 with `optimum-quanto` is the lowest-friction option on Ada/Hopper hardware, and it needs no calibration or low-rank branch:

```python
import torch
from diffusers import FluxPipeline
from optimum.quanto import freeze, qfloat8, quantize

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
)

# FP8 e4m3 weights (and optionally activations) on the heavy modules.
# qfloat8 == e4m3; the floating-point grid absorbs outliers (section 2).
quantize(pipe.transformer, weights=qfloat8)
freeze(pipe.transformer)
quantize(pipe.text_encoder_2, weights=qfloat8)   # the big T5-XXL
freeze(pipe.text_encoder_2)

pipe.to("cuda")
# transformer memory roughly halves vs bf16; quality delta is typically tiny.
img = pipe("a koi pond at dawn", num_inference_steps=28).images[0]
```

The distinction across those three snippets is the crux of the whole quantization story: NF4 weight-only buys you the *memory* (fits the model); FP8 buys you ~2× memory *plus* a modest speedup on FP8 tensor cores with almost no setup; SVDQuant W4A4 buys you the *most* memory *and* the *most* speed (the matmuls run in int4) at the cost of needing a pre-quantized checkpoint and the low-rank machinery. If you're VRAM-bound but latency is acceptable, NF4 is a one-line win. If your hardware has FP8 tensor cores and 2× is enough, FP8 is the easy default. If you need the full 4× of both memory and speed, you want true activation quantization, and SVDQuant is currently the only path that holds diffusion quality at 4-bit.

## 4. The redundancy diffusion leaves on the table: feature caching

Quantization makes each forward pass cheaper. Caching makes you do *fewer* forward passes' worth of work — without changing the step count or the sampler. The lever is a redundancy that sits inside the denoiser and that the iterative structure of diffusion practically hands you for free.

The observation is empirical and robust: **the deep, low-resolution features of the denoiser change very slowly between adjacent timesteps.** In a U-Net, the bottleneck features (the coarse, semantic, low-spatial-resolution maps in the middle) at step $t$ and step $t-1$ are nearly identical — cosine similarity above 0.99 across most of the trajectory — because adjacent denoising steps make only a small refinement to the latent, and the *high-level* content (what the image *is*) is settled early while only the *details* (what the high-frequency texture *looks like*) keep changing. The shallow, high-resolution features that produce those details do change step to step; the deep features that encode the gist do not. So recomputing the deep features every single step is wasted work.

**DeepCache** (Ma et al., 2024) exploits exactly this. The recipe: on a "full" step, run the entire U-Net and *cache the deep-block (bottleneck) features*. On the next $N-1$ "cached" steps, run only the *shallow* encoder/decoder blocks and *reuse* the cached deep features instead of recomputing them, splicing the cached bottleneck back into the U-Net via its skip connections. Because the skip connections are exactly the wires that carry the deep features up to the shallow decoder, the U-Net's architecture makes this splice clean. You run the expensive deep blocks once every $N$ steps and the cheap shallow blocks every step. With $N = 3$ (recompute deep features every third step) you cut roughly two-thirds of the deep-block FLOPs for a 1.5–2× wall-clock speedup at a quality cost that's barely measurable.

![A dataflow graph showing a latent split into a cheap shallow path run every step and a heavy deep path that is cached and reused for two to three adjacent steps before being combined.](/imgs/blogs/quantization-caching-and-efficient-inference-4.png)

Figure 4 shows the split. The latent at step $t$ feeds two paths: the shallow blocks, which are cheap and run every step, and the deep blocks, which are heavy and run once, then sit in the cache for the next two or three steps. The "combine" node splices the freshly computed shallow features with the cached deep features and produces the noise prediction. The win is structural: you've identified the part of the computation that's redundant (the slowly-changing deep features) and stopped repeating it.

The science of *why this is safe* deserves a moment, because "it just works" is not an engineering answer. Let $f_t$ be the deep feature at step $t$. The claim is $\|f_t - f_{t-1}\| \ll \|f_t\|$ for most $t$, so reusing $f_t$ in place of $f_{t-1}$ introduces an error $\|f_t - f_{t-1}\|$ that's small relative to the signal. This holds because the denoiser's deep features are a function of the *current estimate of the clean image* (the model's running guess of $x_0$), and that estimate stabilizes early — by the time you're a third of the way through sampling, the coarse content is largely decided and later steps mostly sharpen. The error you inject by caching is concentrated in the high-frequency details, which the *shallow* path (run every step) is still computing fresh — so the caching error lands precisely where it does the least damage. That's the difference between feature caching and just doing fewer steps: fewer steps degrades the *whole* image (coarse and fine); caching degrades only the part you're still computing anyway, which is to say almost nothing.

There's a subtlety that the naive "cache every $N$ steps" recipe gets wrong, and fixing it is what **TeaCache** (Liu et al., 2024) contributes. A *uniform* cache schedule (recompute every 3rd step, always) is suboptimal because the *rate* at which features change is not constant across the trajectory — features change fast early (when the latent is mostly noise and the model is rapidly committing to content) and slowly late (when it's just polishing). A uniform schedule over-caches early (where it hurts) and under-caches late (where you could cache more aggressively for free). TeaCache makes the cache **timestep-aware**: it uses a cheap, precomputed indicator — the rate of change of the *timestep-modulated inputs* to the transformer block, which correlates strongly with the rate of change of the *outputs* — to decide *adaptively* when the accumulated change since the last full compute has crossed a threshold, and only then does it recompute. Between recomputes it reuses the cached output. The effect is that the model recomputes densely in the volatile early region and sparsely in the stable late region, getting a better speed/quality trade than any fixed schedule. On DiT-based models (where there are no U-Net skip connections to splice into, so DeepCache doesn't directly apply), TeaCache caches the *residual* of the full transformer output between steps, and reports 1.5–2× speedups on FLUX, PixArt, and Open-Sora at a quality delta the authors measure as negligible on standard benchmarks.

Here's a minimal TeaCache-style hook for a `diffusers` transformer — the idea is to register a forward wrapper that decides, per step, whether to recompute or reuse the cached output based on an accumulated-change estimate:

```python
import torch

class TeaCacheWrapper:
    """Timestep-aware caching: reuse last output while the accumulated
    modulated-input change stays under a threshold; recompute when it
    crosses. Wraps a diffusers transformer's forward."""
    def __init__(self, transformer, rel_l1_thresh=0.4):
        self.transformer = transformer
        self.thresh = rel_l1_thresh
        self.accumulated = 0.0
        self.prev_mod_input = None
        self.cached_residual = None

    def __call__(self, hidden_states, timestep, **kwargs):
        # cheap proxy for how much the output will change this step
        mod_in = self.transformer.time_embed(timestep)  # modulation signal
        if self.prev_mod_input is not None:
            delta = (mod_in - self.prev_mod_input).abs().mean()
            self.accumulated += (delta / (self.prev_mod_input.abs().mean() + 1e-6)).item()
        self.prev_mod_input = mod_in

        if self.cached_residual is not None and self.accumulated < self.thresh:
            # REUSE: skip the heavy forward, apply the cached residual
            return hidden_states + self.cached_residual

        # RECOMPUTE: run the real transformer, refresh the cache, reset
        out = self.transformer(hidden_states, timestep, **kwargs)
        self.cached_residual = out.sample - hidden_states
        self.accumulated = 0.0
        return out

# usage: pipe.transformer.forward = TeaCacheWrapper(pipe.transformer)
# higher rel_l1_thresh -> more caching -> faster but more quality drift
```

The single knob is `rel_l1_thresh`: raise it to cache more aggressively (faster, more drift), lower it for fidelity. In practice you tune it to the largest value where a fixed-seed batch of generations is visually indistinguishable from the uncached run, then back off 20% for safety margin. That's the honest way to set any caching threshold — eyeball a fixed-seed grid, find the breaking point, and leave headroom.

The architecture dependence I flagged earlier shows up sharply here, and it's the single most common reason a caching trick silently does nothing. **DeepCache needs a U-Net** — its entire mechanism is splicing cached *bottleneck* features back in through the *skip connections*, and a DiT has neither bottleneck nor skips. If you try to apply DeepCache to FLUX you'll find there's no deep/shallow boundary to cut at; the blocks are uniform. **TeaCache works on both** because it doesn't depend on internal structure — it caches the *residual of the whole block's output* ($\Delta = \text{output} - \text{input}$) and reuses that residual while the timestep-modulation hasn't changed much. On a DiT, that residual-caching is the natural form: you skip the entire transformer's recompute on cached steps and just re-apply the last residual. There's a family of related DiT methods — block caching, $\Delta$-caching, FORA — that all exploit the same residual-redundancy in slightly different ways, but the principle is identical: the per-step *change* in the network's output is small and slowly varying, so you can hold it constant for a few steps. Pick the caching method that matches your backbone; reaching for DeepCache on a DiT is the afternoon-waster I warned about in section 1.

One honest caveat about caching that the papers underplay: caching interacts badly with *very few steps*. The whole premise is "features barely change between *adjacent* steps," but if you're already at 4 steps (Turbo/LCM), adjacent steps are *far apart* in noise level — each step makes a big jump — so the slowly-varying assumption breaks and caching introduces visible error. Caching is a lever for the *many-step* regime (28–50 steps), where there's redundancy to harvest; in the few-step regime, distillation already removed the redundancy and there's little left for caching to take. The two levers are substitutes at the low-step end, not complements. Don't stack aggressive caching on top of a 4-step model expecting a free 2×; you'll mostly get artifacts. This is the kind of interaction the "everything composes" slogan glosses over, and it's exactly the sort of thing you only learn by measuring.

## 5. Attention and VAE: the two specialized optimizations

Two components of the stack have their own well-developed optimization stories that don't reduce to general quantization or caching, and skipping them leaves easy wins on the table.

**Attention** is the dominant per-block cost in a DiT and a significant one in the attention layers of a U-Net. The naive implementation materializes the full $T \times T$ attention matrix in memory — quadratic in sequence length $T$ — which at high resolution (thousands of tokens) is both a memory hog and a bandwidth bottleneck. The fix is **FlashAttention**: compute attention in tiles, never materializing the full matrix, fusing the softmax and the matmuls into one kernel that streams through SRAM. In PyTorch you get it for free through `scaled_dot_product_attention` (SDPA), which dispatches to a FlashAttention or memory-efficient backend automatically:

```python
import torch
import torch.nn.functional as F

# diffusers uses SDPA by default on recent versions; this is the path it hits.
# q, k, v: (batch, heads, seq_len, head_dim)
out = F.scaled_dot_product_attention(q, k, v)   # fused, no T x T matrix

# force the FlashAttention backend explicitly if you want to be sure:
from torch.nn.attention import sdpa_kernel, SDPBackend
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    out = F.scaled_dot_product_attention(q, k, v)
```

Modern `diffusers` already routes attention through SDPA, so on a recent PyTorch you likely have FlashAttention without doing anything — but it's worth verifying, because an old `xformers` path or a custom attention processor can silently fall back to the quadratic-memory implementation. The complementary trick when you're *memory*-bound rather than speed-bound is **attention slicing** (`pipe.enable_attention_slicing()`), which computes attention in chunks over the head or query dimension to cap peak memory at the cost of some speed. You reach for slicing only when SDPA still OOMs, because SDPA's memory-efficient backend already does most of what slicing does, better.

The quantitative reason FlashAttention matters so much for diffusion at high resolution is the same quadratic that makes high-res expensive in the first place. Naive attention materializes a $T \times T$ score matrix: at 1024×1024 with $T \approx 4{,}096$ tokens and 24 attention heads, that's $24 \cdot 4096^2 \approx 400$M entries *per layer*, which in fp16 is ~800 MB of attention scores you write to and read back from HBM, *per layer*, ×57 layers. The matrix isn't just a memory hog — writing and reading hundreds of MB to global memory is a *bandwidth* disaster, and attention is bandwidth-bound, not compute-bound, exactly in this regime. FlashAttention never materializes that matrix: it tiles the computation so the scores live in fast on-chip SRAM and are consumed immediately, turning $O(T^2)$ HBM traffic into $O(T)$. At 1024² the speedup is large; at 2K (where $T$ quadruples to ~16K and the naive score matrix would be ~13 GB *per layer*) it's the difference between running and not running at all. This is why FlashAttention/SDPA is non-negotiable for high-resolution diffusion, and why it's the *first* lossless lever to confirm is on before you optimize anything else.

**The VAE** has the opposite problem: it's cheap in FLOPs but explodes in *peak activation memory* at high resolution, because the decoder upsamples a small latent into a large image and the intermediate feature maps at full resolution are enormous. Decoding a 1024×1024 image is fine on a 4090; decoding 2048×2048 or 4096×4096 can OOM even though the latent is tiny and the denoiser already finished. Two complementary tricks fix this, and they're the reason high-res generation is feasible at all on consumer cards.

**VAE slicing** (`pipe.enable_vae_slicing()`) decodes a batch of latents one image at a time instead of all at once, capping memory at single-image cost — relevant when you generate many images per call. **VAE tiling** (`pipe.enable_vae_tiling()`) is the bigger hammer for a *single large* image: it splits the latent into overlapping spatial tiles, decodes each tile separately so the peak activation memory is bounded by *tile* size not *image* size, and blends the overlapping seams back together with a feathered weight to hide tile boundaries. The cost is a small compute overhead (overlap is decoded twice) and the risk of faint seams if the blend is too narrow, but it's what lets you decode a 4K image on a 24 GB card.

![A dataflow graph showing a large latent split into overlapping tiles, each decoded with low peak memory, then blended at the seams into a full four-K image that fits on a twenty-four gigabyte card.](/imgs/blogs/quantization-caching-and-efficient-inference-7.png)

Figure 7 shows the tiling path: the big latent splits into overlapping tiles, each decodes with bounded peak memory, and the seams are feathered together into the full image. The reason this works is that VAE decoding is *mostly local* — a pixel's value depends mostly on the nearby latent, so decoding a tile with a modest overlap reconstructs the interior of the tile almost perfectly, and only the boundary needs blending. The overlap has to be at least as wide as the decoder's receptive field for the interior to be seamless; in practice `diffusers` uses a 25% overlap, which is comfortably enough for the typical VAE, and feathers the overlap with a linear ramp so two adjacent tiles' predictions average smoothly across the seam.

#### Worked example: why the VAE, not the transformer, OOMs at 4K

Take FLUX at 4096×4096. The latent (8× spatial compression, 16 channels) is 512×512×16 — that's 4M elements, 8 MB in bf16. The transformer processes it as $512^2/4 = 65{,}536$ tokens; its activations are large but the SVDQuant-or-not transformer fits because you already optimized it. Now the VAE decoder upsamples that latent toward 4096×4096×3 = 50M output pixels, and the *intermediate* feature maps in the decoder's full-resolution stages are the killer: a single conv feature map at 4096×4096 with, say, 128 channels is $4096^2 \cdot 128 = 2.1$B elements = ~4.3 GB in bf16, and the decoder holds *several* of these live at once during the forward pass — easily 15–20 GB of transient activations, on top of everything else, and you OOM. Tiling into, say, 16 tiles of 1024×1024 caps each tile's decode peak at the 1024² activation size (~64× smaller per tile), so peak memory drops from ~15 GB to ~1 GB of transient VAE activation. *That's* the difference between a 4K render that crashes and one that fits — and notice it had nothing to do with the transformer or quantization. Wrong-lever diagnosis here (quantizing the transformer harder to "make room") wastes effort; the fix is structurally in the VAE.

```python
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
).to("cuda")

pipe.enable_vae_tiling()    # bound VAE peak memory -> high-res decode fits
pipe.enable_vae_slicing()   # decode batched images one at a time

img = pipe("a sweeping mountain vista, ultra detailed",
           height=2048, width=2048, num_inference_steps=28).images[0]
```

There's a third VAE option that trades quality for raw speed: **`AutoencoderTiny`** (TAESD), a tiny distilled autoencoder that decodes ~10× faster than the full VAE with a small fidelity loss. It's the right tool for *previews* — show the user a fast approximate decode while the full VAE runs in the background, or use it during interactive prompt iteration where speed beats fidelity. Swap it in with one line:

```python
from diffusers import AutoencoderTiny

# Tiny distilled VAE: ~10x faster decode, slightly lower fidelity.
pipe.vae = AutoencoderTiny.from_pretrained(
    "madebyollin/taesd", torch_dtype=torch.bfloat16
).to("cuda")
# great for live previews; switch back to the full VAE for the final render.
```

## 6. The runtime layer: compilation, CUDA graphs, channels-last, offload

You've quantized the weights, cached the features, and fused the attention. The last layer of wins is in the *runtime* — how the kernels are scheduled, laid out in memory, and where tensors live. These are the cheapest optimizations to apply (often one line) and they compound with everything above.

**`torch.compile`** is the biggest single runtime win. It traces the denoiser, fuses pointwise operations into combined kernels, removes Python overhead, and autotunes. For a diffusion model — which calls the *same* graph dozens of times in the sampling loop — the one-time compilation cost amortizes immediately, and you typically get a 1.3–2× speedup with zero quality change:

```python
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
).to("cuda")

# channels-last memory layout: better tensor-core utilization for conv/attn
pipe.transformer.to(memory_format=torch.channels_last)
pipe.vae.to(memory_format=torch.channels_last)

# compile the hot module; "max-autotune" searches kernels (slow first call)
pipe.transformer = torch.compile(
    pipe.transformer, mode="max-autotune", fullgraph=True
)

# first call compiles (30-120 s); subsequent calls run the fused graph
_ = pipe("warmup prompt", num_inference_steps=4)         # triggers compile
img = pipe("a tiger in tall grass", num_inference_steps=28).images[0]
```

Two details earn their keep. **`channels-last`** memory format reorders the tensor layout so that convolutions and attention hit the tensor cores more efficiently — a free 5–15% on conv-heavy models, essentially free on the VAE. **CUDA graphs** (which `torch.compile`'s `"reduce-overhead"` mode captures, or which you can capture manually) record the entire sequence of GPU kernel launches once and replay them as a single unit, eliminating per-launch CPU overhead — this matters most when your per-step kernel work is small relative to launch overhead, i.e. exactly the few-step distilled regime where you've already cut the compute and launch overhead becomes the tax. The interaction is worth naming: the more you optimize the compute (quantization, distillation, caching), the *larger* the relative share of CPU launch overhead, so CUDA graphs become *more* valuable the more of the other levers you've pulled. They compound.

The final runtime lever is for when you're flatly out of VRAM: **CPU offload**. `pipe.enable_model_cpu_offload()` keeps each model component (text encoders, transformer, VAE) on the CPU until it's needed, moves it to GPU for its turn, then evicts it. For FLUX, that means the 9.5 GB T5 encoder runs, gets evicted, *then* the transformer loads — so peak VRAM is bounded by the single largest component, not the sum. The cost is the PCIe transfer time per component (hundreds of milliseconds), which is why offload *adds latency* — it's a memory-for-time trade, the one lever in this whole post that makes you *slower*. The more aggressive `enable_sequential_cpu_offload()` offloads at the *module* level (sub-components shuttle on demand), cutting VRAM further at a larger speed penalty. The decision rule: use model-level offload when you're memory-bound and can tolerate the transfer latency; never use it when you have the VRAM, because it's pure overhead. It exists so a 12B model *runs at all* on a card that can't hold it — correctness over speed.

```python
# memory-bound recipe: fit a model that doesn't otherwise fit
pipe.enable_model_cpu_offload()        # component-level: peak = largest model
# or, even tighter (slower):
# pipe.enable_sequential_cpu_offload() # module-level: minimal peak VRAM

# combine with quantization for the smallest possible footprint:
# NF4 transformer + model offload runs FLUX in well under 12 GB.
```

There's one runtime trap worth calling out because it silently un-does `torch.compile`: **dynamic shapes**. `torch.compile` specializes the compiled graph to the input shapes it first sees. If your batch size, resolution, or sequence length changes between calls, PyTorch *recompiles* — and recompilation costs the same tens of seconds every time, which obliterates the speedup if it happens per request. The fixes are either to pin your shapes (always generate at one resolution and batch size, padding if needed), or to compile with `dynamic=True` (which produces a more general but slightly slower graph that handles a range of shapes without recompiling). For a service that generates at a fixed 1024×1024 with a fixed batch, pin the shapes and enjoy the fully-specialized graph. For an interactive tool where users pick arbitrary resolutions, `dynamic=True` is the safer default. Diagnose a "my compiled model is slow" complaint by logging recompilation events (`TORCH_LOGS=recompiles`); nine times out of ten it's a shape that changed.

## 7. A consolidated serving script

It helps to see the levers assembled into one script you can actually run, rather than as a pile of independent snippets. Here is a realistic "fit-and-fly on a 4090" configuration: FP8 weights (the easy memory win), SDPA attention (on by default, confirmed), `torch.compile` with pinned shapes, VAE tiling for headroom, and a distilled few-step scheduler. This is the shape of code that turns the fp16/50-step baseline into the sub-second service from figure 1.

```python
import torch
from diffusers import FluxPipeline, FlowMatchEulerDiscreteScheduler
from optimum.quanto import freeze, qfloat8, quantize

torch.set_float32_matmul_precision("high")   # let matmuls use TF32 fastpaths

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
)

# 1) FP8 the two heavy modules: transformer and the big T5 text encoder.
for module in (pipe.transformer, pipe.text_encoder_2):
    quantize(module, weights=qfloat8)
    freeze(module)

pipe.to("cuda")

# 2) channels-last + compile the transformer with PINNED shapes (no recompiles)
pipe.transformer.to(memory_format=torch.channels_last)
pipe.transformer = torch.compile(
    pipe.transformer, mode="max-autotune", fullgraph=True
)

# 3) cap VAE peak memory so high-res decodes never OOM
pipe.enable_vae_tiling()
pipe.enable_vae_slicing()

# 4) fewer steps via a distilled LoRA (Turbo/LCM-style); guidance baked in
pipe.load_lora_weights("path/to/flux-turbo-lora")   # e.g. an 8-step adapter
pipe.fuse_lora()

GEN = torch.Generator("cuda").manual_seed(0)

def generate(prompt, steps=8):
    return pipe(prompt, num_inference_steps=steps, guidance_scale=0.0,
                height=1024, width=1024, generator=GEN).images[0]

# warm up to trigger compilation, then time the steady state
_ = generate("warmup", steps=4)
torch.cuda.synchronize()

img = generate("a photorealistic red fox in a snowy forest, golden hour")
img.save("fox.png")
print("peak VRAM (GB):", torch.cuda.max_memory_allocated() / 1e9)
```

Read that script as the *policy* the rest of the post argues for, made concrete: lossless levers (SDPA, compile, channels-last) unconditionally; the easy 2× (FP8) by default; VAE tiling for headroom; and the step-count collapse (distilled LoRA) as the biggest single latency win. If FP8's 2× isn't enough memory, swap the `quantize(..., qfloat8)` calls for the SVDQuant int4 transformer from section 3. If 8 steps isn't fast enough, drop to 4 and re-measure quality. Every knob in this script maps to a lever and a constraint, and that mapping is the actual skill — not memorizing the API calls, but knowing which one your bottleneck demands.

## 8. Stacking the levers: the full efficient-inference recipe

The reason this post is worth reading as a whole, rather than cherry-picking one trick, is that **the levers are orthogonal and they compound**. Quantization shrinks the per-pass memory and (with W4A4) compute; caching cuts the number of expensive passes; attention fusion and compilation make each kernel faster; distillation cuts the step count; VAE tiling caps the decode peak. None of them conflict — you can quantize *and* cache *and* compile *and* distill the same model — and because each multiplies a different factor of the latency, stacking them is multiplicative, not additive.

![A vertical stack showing the fp16 fifty-step baseline at the bottom and each added optimization, attention fusion, quantization, compilation, caching, and few-step distillation, multiplying toward a roughly ten-times faster stacked total at the top.](/imgs/blogs/quantization-caching-and-efficient-inference-6.png)

Figure 6 stacks them in the order you'd typically apply them, each layer multiplying onto the last. Start from fp16, 50 steps, 6.1 s. Fuse attention with SDPA/FlashAttention: shave ~25% off the per-step cost. Quantize to 4-bit (SVDQuant W4A4): 3.3× less VRAM and ~2× faster matmuls. Compile with `torch.compile`: another ~1.4×. Add feature caching (TeaCache): skip redundant steps for ~1.6×. Finally, distill to 4 steps (Turbo/LCM): ~5× from the step-count collapse. Multiply the surviving factors and you land near 0.5 s/image — a 10×-plus end-to-end speedup over the baseline, on the same 4090, with a quality delta you tune to be imperceptible.

The order matters for a practical reason: apply the *lossless* levers first (attention fusion, compilation, channels-last cost you nothing in quality), then the *near-lossless* ones (SVDQuant, conservative caching), and only then the *quality-trading* ones (aggressive caching, few-step distillation) — and stop as soon as you hit your latency budget so you spend the least quality. There's no point distilling to 1 step if 4-bit + caching already met your SLA; you'd be burning quality you didn't need to.

![A five by three matrix comparing SVDQuant, feature caching, torch.compile, VAE tiling, and CPU offload across speedup, VRAM saved, and quality cost, showing each lever pays a different currency.](/imgs/blogs/quantization-caching-and-efficient-inference-5.png)

Figure 5 is the lever cheat-sheet. Read it as "what currency does each lever pay in." SVDQuant pays in VRAM *and* speed at a small quality cost. Caching pays in speed only, no VRAM, small quality cost. `torch.compile` pays in speed at zero quality cost — always on. VAE tiling pays in peak memory at high resolution at no quality cost — on whenever you push resolution. CPU offload pays *negative* speed (it's slower) to buy huge VRAM — only when you're out of memory. The recipe falls out of matching the lever's currency to your binding constraint, which is the next section's whole point.

#### Worked example: serving FLUX.1-dev on one RTX 4090 under a 2 s SLA

Constraint: 1024×1024, p95 latency under 2 seconds, single 24 GB 4090. Baseline fp16/50-step is 6.1 s at 23 GB — fails on both. Plan: (1) load the SVDQuant W4A4 transformer → VRAM drops to ~7 GB, matmuls ~2× faster, per-step ~70 ms. (2) `torch.compile` the transformer → per-step ~50 ms. (3) Distill the sampler to an 8-step schedule with an LCM-LoRA → 8 steps × 50 ms = 0.4 s of denoiser. (4) VAE decode with the full VAE ~120 ms, text encode (cached per prompt) amortized. Total ~0.55 s/image, comfortably under the 2 s SLA, at ~7 GB — leaving room for a second concurrent stream. We *didn't* need feature caching or 1-step distillation; we stopped at the budget and kept the quality. That's the discipline: stack until you clear the SLA, then stop.

## 9. Case studies: the real numbers

Numbers in this section are drawn from the source papers and from `diffusers` documentation/community benchmarks; where a figure is a representative measurement rather than a headline result from the paper, I mark it approximate. Latencies are hardware- and software-version-dependent — treat them as order-of-magnitude points that show the *shape* of the trade, not exact constants.

**SVDQuant / Nunchaku on FLUX.1-dev (Li et al., 2024).** The headline result: 4-bit weights and activations (W4A4) on a 12B FLUX transformer, with the transformer memory dropping from ~22–24 GB (bf16) to ~6.5 GB (int4 + low-rank branch) — roughly a 3.5× model-memory reduction — and a reported ~3× latency speedup over the bf16 baseline on a single GPU, at an image-quality delta the authors measure as on-par with fp16 (a fraction of a FID point, and human-preference scores statistically tied). The result that mattered to the field: *every prior* W4A4 attempt on diffusion had produced visibly broken images; SVDQuant was the first to hold quality, and the low-rank-branch insight is *why*.

**FP8 inference on FLUX and SD3 (community / diffusers).** FP8 (`e4m3`) weight-and-activation inference roughly halves transformer memory versus bf16 with a quality delta typically reported as negligible, and gives a modest speedup on FP8-tensor-core hardware (Ada/Hopper). The practical takeaway from the field: FP8 is the *easy* 2× — it needs no calibration or low-rank machinery, just hardware support — and it's the first quantization step most teams take. SVDQuant int4 is the *hard* 4× you reach for when FP8's 2× isn't enough.

**DeepCache (Ma et al., 2024).** On Stable Diffusion v1.5, DeepCache with a cache interval $N = 3$ reported roughly a 2× wall-clock speedup with a CLIP-score change within noise and FID essentially unchanged on COCO captions — the canonical demonstration that U-Net deep features are reusable. On SDXL the relative win is smaller (SDXL's architecture has proportionally more shallow compute) but still meaningful, ~1.5×.

**TeaCache on FLUX / Open-Sora (Liu et al., 2024).** Timestep-aware caching reported ~1.5–2× speedups on FLUX.1-dev and on video DiTs (Open-Sora, CogVideoX) with negligible quality change at the conservative threshold, scaling to larger speedups (and larger but still-acceptable quality drift) as the threshold rises. The video result is notable because video diffusion is even more redundant across steps (and across *frames*), so caching pays off more there than in images.

**SANA's deep-compression autoencoder (Xie et al., 2024).** A different attack on the same problem, worth contrasting because it pulls the *latent compression* lever rather than quantization or caching. SANA uses a 32× autoencoder (versus the standard 8×) so the latent is 16× smaller in element count, which slashes the transformer's token count and makes high-resolution generation cheap *before* any of the inference tricks in this post apply. Combined with linear attention, SANA reported generating 1024×1024 images dramatically faster than comparable models — an existence proof that the four levers are genuinely independent: you can win on latent compression (SANA), step count (Turbo), per-pass compute (SVDQuant), and caching (TeaCache) *separately and together*. We cover the architecture in [the modern text-to-image recipe](/blog/machine-learning/image-generation/mmdit-and-the-modern-text-to-image-recipe); the relevant point here is that aggressive AE compression is the *resolution-bound* lever, complementary to everything else.

**Stacking on a 4090 (representative, approximate).** A FLUX.1-dev pipeline taken from fp16/50-step (~23 GB, ~6 s) through SVDQuant W4A4 + `torch.compile` + an 8-step distilled schedule lands at roughly 7 GB and well under 1 s/image on a single RTX 4090 — a >6× speed and >3× memory improvement, with the quality tuned to imperceptible. These are representative integration numbers, not a single published benchmark; the point is the *compounding*, which is robust even if your exact constants differ. The reason I keep flagging "approximate" is that these constants move month to month — a new PyTorch release improves the compiler, a new FlashAttention kernel lands, a better-distilled LoRA ships — so the *exact* latency you measure next quarter will differ from the table. What *doesn't* move is the structure: the denoiser dominates, the levers are orthogonal, format beats bit-count at 8 bits, and the VAE OOMs you at high resolution. Internalize the structure; re-measure the constants on your own hardware.

| Configuration (FLUX.1-dev, 1024², RTX 4090) | VRAM | Steps | Latency / image | Quality (FID/CLIP delta) |
| --- | --- | --- | --- | --- |
| fp16 baseline | ~23 GB | 50 | ~6.1 s | reference |
| FP8 (e4m3) W+A | ~12 GB | 50 | ~4.5 s | ≈ baseline (approx) |
| NF4 weight-only + offload | <10 GB | 50 | ~7 s (offload tax) | ≈ baseline (approx) |
| SVDQuant int4 W4A4 | ~7 GB | 50 | ~2.4 s | +<1 FID (approx) |
| SVDQuant + torch.compile | ~7 GB | 50 | ~1.9 s | +<1 FID (approx) |
| SVDQuant + compile + TeaCache | ~7 GB | 50 eff. | ~1.2 s | +~1 FID (approx) |
| SVDQuant + compile + 8-step LCM | ~7 GB | 8 | ~0.55 s | +~2 FID (approx) |

Read the table as a Pareto frontier, not a leaderboard. Each row buys something — the int4 rows buy memory, the compile/cache/distill rows buy latency — and your job is to walk down it until you hit your constraint and stop. The NF4-offload row is the cautionary one: it's the *smallest* memory but *slowest* latency because offload transfers dominate, which is exactly why you only reach for offload when you're memory-bound and latency-tolerant.

## 10. Honest measurement: how not to fool yourself

A speedup table is only as trustworthy as its methodology, and efficiency work is unusually easy to fake yourself out on. A few rules I hold to.

**Warm up before you time.** The first call to a `torch.compile`d model pays the compilation cost (tens of seconds); the first CUDA kernel launch pays initialization. Always run 2–3 warm-up iterations and discard them, then time the median of 10+ runs — report the median, not the mean (one stray GC pause skews the mean). Synchronize the GPU (`torch.cuda.synchronize()`) before stopping the timer, or you're timing the kernel *launch*, not the kernel *execution*.

**Measure VRAM at peak, not at rest.** `torch.cuda.max_memory_allocated()` after a full generation tells you the peak — which is what determines whether you OOM — not `memory_allocated()` at an arbitrary moment. The VAE decode is often the peak even though the transformer is the bigger model, which surprises people and is exactly why you measure rather than assume.

**For quality, fix the seed and compare apples to apples.** Generate the same prompts with the same seed under fp16 and under the optimized config, and compare side by side — a fixed-seed grid catches the subtle degradations (slightly washed color, lost fine texture) that an FID number aggregates away. For an FID number, use a fixed reference set (e.g. 10K–30K COCO images), a fixed sample count, and the *same* count for both configs — FID is biased by sample size, so 5K-vs-30K is not a fair comparison. Report FID-DINOv2 alongside classic FID if you can; it correlates better with human judgment and is less gameable. And state your numbers as deltas from the baseline you measured *yourself* on *your* hardware, not the paper's — your VAE, scheduler, and step count differ, and absolute numbers won't transfer.

**Watch for the quality cost that doesn't show up in FID.** Aggressive caching and low-bit quantization sometimes preserve FID (distributional similarity) while degrading *prompt adherence* or *fine detail* — the kind of thing CLIP-score or a human eval catches but FID misses. If your use case is compositional (three objects, specific colors, text rendering), test on *that*, because the average-case metric will lie to you about the cases you care about. A concrete failure I've watched bite teams: a model quantized to int4 that scores fine on FID but can no longer render legible text in the image, because text is high-frequency, spatially precise structure that the quantization error smears just enough to turn letters into squiggles — and FID, computed on global Inception features, is blind to it. If "render a sign that says OPEN" is in your product, that's the eval, not COCO FID.

**Separate the levers when you attribute a speedup.** When you stack five optimizations and measure 6× faster, you don't actually know which lever did what unless you ablate — turn each one off in isolation and re-measure. This matters because some levers *interact* (CUDA graphs help more after quantization; caching helps less after distillation), and a number you can't decompose is a number you can't reason about when something regresses. Keep a small ablation grid — baseline, +quant, +quant+compile, +quant+compile+cache — so when a future PyTorch upgrade changes one row you can see exactly which lever moved. The discipline pays for itself the first time a "mysterious slowdown" turns out to be a silent recompilation that your ablation grid localizes in one run.

## 11. When to reach for each lever (and when not to)

Every optimization is a cost, and the engineering judgment is matching the lever to the constraint. The decision is almost always dominated by a single binding constraint — figure out which, and the lever picks itself.

![A decision tree branching on whether the bottleneck is VRAM, latency, or resolution, routing to quantization plus offload, distillation plus caching plus compile, or VAE tiling plus a tiny preview decoder.](/imgs/blogs/quantization-caching-and-efficient-inference-8.png)

Figure 8 is the routing logic. Ask one question — what's the bottleneck? — and follow the branch.

**If you're VRAM-bound (the model won't fit):** quantize first. FP8 if your hardware supports it and 2× is enough; NF4 weight-only if you just need it to *fit* and latency is fine; SVDQuant int4 if you need memory *and* speed. Add `enable_model_cpu_offload()` if quantization alone doesn't fit it. Do *not* reach for caching or distillation here — they don't save memory.

**If you're latency-bound (the model's too slow):** the biggest lever is *step count*, so distill (Turbo/LCM/DMD — see the [consistency](/blog/machine-learning/image-generation/consistency-models-and-few-step-generation) and [distillation](/blog/machine-learning/image-generation/distribution-matching-and-adversarial-distillation) posts) before anything else; one 4-step model beats every caching trick combined. Then stack the free wins: `torch.compile`, SDPA, channels-last. Then feature caching. SVDQuant W4A4 also helps latency (faster matmuls), so it's a two-for-one if you're *both* memory- and latency-bound.

**If you're resolution-bound (OOM at the VAE decode):** this is a VAE problem, not a denoiser problem, so quantizing the transformer won't help. Turn on `enable_vae_tiling()` and `enable_vae_slicing()`; use `AutoencoderTiny` for previews. The denoiser at high resolution also explodes in attention cost (quadratic in tokens), so this is where deep latent compression — a more aggressive VAE like SANA's 32× autoencoder, covered in [the modern text-to-image recipe](/blog/machine-learning/image-generation/mmdit-and-the-modern-text-to-image-recipe) — earns its keep by shrinking the token count before the transformer ever sees it.

A word on *combining* the branches, because real systems are usually bound by more than one thing at once. The common case is "VRAM-bound *and* latency-bound on a consumer card" — you want it to fit *and* be fast. There the recipe is unambiguous: SVDQuant W4A4 (which uniquely buys both memory and speed), then `torch.compile`, then a distilled few-step schedule, and reach for caching only if you're still over budget at many steps. The "resolution-bound *and* latency-bound" case (fast 4K) is harder, because tiling adds compute and high-res attention is quadratic — there your biggest lever is a deeper-compression autoencoder (fewer tokens before the transformer ever runs), then tiling for the decode, then the usual compile/quant stack. The point of the decision tree isn't that constraints come one at a time; it's that you should *name* your binding constraints explicitly, in priority order, and let that ordering pick the levers — rather than reaching reflexively for whichever optimization you read about most recently.

And the *when-nots*, which are where teams waste effort. **Don't quantize to int4 when FP8 met your budget** — you spent quality you didn't need to. **Don't use CPU offload when you have the VRAM** — it's pure latency overhead for nothing. **Don't cache aggressively on a compositional prompt** without testing prompt adherence — FID will hide the regression. **Don't `torch.compile` if your input shapes change every call** (different resolutions, dynamic batch) — recompilation will eat the speedup; compile only when shapes are stable, or use dynamic shapes carefully. **Don't distill to 1 step when 4 steps clears your SLA** — the marginal quality loss from 4→1 is the steepest part of the curve. And **don't quantize the VAE casually** — the VAE is small (a few hundred MB) so quantizing it saves little memory, and its decode quality is highly visible; keep it in fp16/bf16 unless you've measured no degradation.

## Key takeaways

- **The denoiser runs in a loop, so per-pass savings multiply by step count.** Any optimization to a single forward pass — quantization, attention fusion, compilation — is amplified by the number of sampling steps. That's why diffusion is worth optimizing hard.
- **Activation quantization is the hard part, and diffusion is harder than LLMs.** Weight-only 4-bit is easy and saves memory; activation 4-bit is what saves *compute*, and diffusion's heavy-tailed, timestep-varying activations with cross-step error accumulation break naive int8 where an LLM shrugs it off.
- **At 8 bits, format beats bit-count.** FP8's floating-point grid absorbs outliers gracefully; int8's linear grid does not. FP8 is the easy 2×.
- **SVDQuant is the 4-bit diffusion result.** Smooth the activation outliers into the weights, lift the weight outliers into a tiny 16-bit low-rank branch via SVD, and the flat residual quantizes cleanly to int4 — W4A4 at near-fp16 quality, the first method to hold the line.
- **Feature caching is near-free speed.** Deep denoiser features barely change between adjacent steps; DeepCache reuses U-Net bottleneck features, TeaCache caches the transformer residual *adaptively* by timestep. The error lands in the high-frequency details the shallow path recomputes anyway.
- **The VAE, not the denoiser, OOMs you at high resolution.** Tiling and slicing cap its peak activation memory; `AutoencoderTiny` gives a 10× faster preview decode.
- **The runtime layer is cheap free speed.** `torch.compile` (1.3–2×, zero quality cost), channels-last, and CUDA graphs compound with everything; CUDA graphs matter *more* the more you've already optimized the compute.
- **The levers are orthogonal and compound multiplicatively.** Quantize *and* cache *and* compile *and* distill — apply lossless levers first, stop at your SLA, and spend the least quality.
- **Match the lever to the binding constraint.** VRAM-bound → quantize/offload; latency-bound → distill then compile then cache; resolution-bound → VAE tiling and deeper latent compression. The constraint picks the lever.
- **Measure honestly.** Warm up, synchronize, take the median, fix the seed, compare same-sample-count FID on your own hardware, and test prompt adherence — because the average metric hides the cases you care about.

## Further reading

- **SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models** — Li et al., 2024. The low-rank outlier-absorption method and the Nunchaku engine; the core 4-bit diffusion result this post is built around.
- **SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models** — Xiao et al., 2022. The activation-outlier migration trick SVDQuant's smoothing stage builds on.
- **DeepCache: Accelerating Diffusion Models for Free** — Ma et al., 2024. Caching U-Net deep-block features across timesteps.
- **TeaCache: Timestep Embedding Aware Cacheing** — Liu et al., 2024. Adaptive, timestep-aware caching for diffusion transformers and video DiTs.
- **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** — Dao et al., 2022. The tiled, fused attention kernel SDPA dispatches to.
- **🤗 `diffusers` — Memory and inference optimization docs.** The canonical reference for `enable_model_cpu_offload`, `enable_vae_tiling/slicing`, `torch.compile`, and `AutoencoderTiny` in practice.
- **Within this series:** [why diffusion is slow and how to fix it](/blog/machine-learning/image-generation/why-diffusion-is-slow-and-how-to-fix-it) (the four-lever frame), [consistency models and few-step generation](/blog/machine-learning/image-generation/consistency-models-and-few-step-generation) and [distribution matching and adversarial distillation](/blog/machine-learning/image-generation/distribution-matching-and-adversarial-distillation) (the step-count lever this stacks with), [the modern text-to-image recipe](/blog/machine-learning/image-generation/mmdit-and-the-modern-text-to-image-recipe) (SANA's deep-compression AE), and the capstone [building an image generation stack](/blog/machine-learning/image-generation/building-an-image-generation-stack).
- **Quantization fundamentals:** [quantization from first principles](/blog/machine-learning/edge-ai/quantization-from-first-principles), [LLM quantization: activations, SmoothQuant, and KV cache](/blog/machine-learning/edge-ai/llm-quantization-activations-smoothquant-kv-cache), and [below 8 bits: int4, ternary, and binary networks](/blog/machine-learning/edge-ai/sub-8-bit-int4-ternary-and-binary-networks) in the Edge AI track.
