---
title: "Why Diffusion Is Slow, and the Four Levers That Fix It"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Build the cost model that explains why diffusion sampling is slow, profile a real diffusers run, and learn the four levers — fewer steps, a faster denoiser, deeper latent compression, and feature caching — that turn a four-second image into a sub-second one."
tags:
  [
    "image-generation",
    "diffusion-models",
    "inference-optimization",
    "step-distillation",
    "latency",
    "sana",
    "generative-ai",
    "deep-learning",
    "diffusers",
  ]
category: "machine-learning"
subcategory: "Image Generation"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/why-diffusion-is-slow-and-how-to-fix-it-1.png"
---

Here is a number that quietly shapes the entire economics of generative imaging. You load Stable Diffusion XL on an RTX 4090, type "a golden retriever puppy in a field of wildflowers, soft morning light," set `num_inference_steps=50`, and press go. Roughly **3.9 seconds** later you have a gorgeous 1024×1024 image. Now imagine you are running a product that generates millions of these a day, or you want a slider in your app that updates the image in real time as the user drags it, or you simply want to put image generation on a phone. Suddenly that 3.9 seconds is a wall. It is the difference between an interactive experience and a loading spinner, between a profitable inference bill and a ruinous one, between something that runs on-device and something that needs a datacenter GPU.

The frustrating part is that the model is not doing 3.9 seconds of useful "thinking." It is doing the *same* forward pass fifty times in a row, each one nudging a noisy latent a little closer to a clean image. The network itself — a U-Net or a diffusion transformer — runs in about 75 milliseconds. If you could run it once, you would have your answer in a fraction of a second. The slowness is not a property of the network; it is a property of the **algorithm** we wrap around it. Diffusion turns generation into an iterative, sequential walk, and that walk is where almost all the latency lives.

This post is the intro to the **Speed track** of the series, and its job is to give you the map. By the end you will have a precise cost model for diffusion latency — a single equation you can profile against — and you will understand the **four levers** that the rest of this track pulls to make diffusion fast: (1) take **fewer steps** by distilling the sampler, (2) use a **smaller, faster denoiser**, (3) compress the **latent** more aggressively so each call is cheaper, and (4) **cache features** so you do not recompute everything every step. We will measure where the time actually goes, profile a real `diffusers` run, and build the **speed↔quality Pareto frontier** that is the recurring frame for everything that follows. This ties directly back to the series' spine — the generative trilemma of quality × diversity × *speed* — and this is the track where we finally attack that third axis head-on.

![A vertical stack showing the diffusion sampling cost split into a one-time text encode, fifty sequential denoiser forwards that dominate, a one-time VAE decode, and the final image](/imgs/blogs/why-diffusion-is-slow-and-how-to-fix-it-1.png)

If you have not yet internalized *why* diffusion samples by iterating, read [the foundations post on diffusion from first principles](/blog/machine-learning/image-generation/diffusion-from-first-principles) — this post assumes you know that sampling means walking from noise to image one denoising step at a time. Here we treat that walk as a cost to be optimized.

## 1. The cost model: where the seconds actually go

Let us write down the latency of a single text-to-image generation as an equation, because once you have it, every optimization in this track becomes obvious — each one is just an attack on a specific term.

A standard latent-diffusion text-to-image pipeline does exactly three things:

1. **Encode the prompt** once, with a text encoder (CLIP, or CLIP+T5 for SD3/FLUX). Cost: $c_\text{text}$.
2. **Denoise** by calling the denoiser network $N$ times, where $N$ is `num_inference_steps`. Each call costs $c_\text{net}$ (one forward pass of the U-Net or DiT, including the cross-attention to the text embeddings). Total: $N \cdot c_\text{net}$.
3. **Decode** the final latent back to pixels once, with the VAE decoder. Cost: $c_\text{vae}$.

So the total wall-clock latency is

$$
\text{latency} = c_\text{text} + N \cdot c_\text{net} + c_\text{vae}.
$$

That is the whole story, and it is worth staring at. The text encode and the VAE decode are **fixed, one-time overheads** — they do not depend on $N$. The denoiser term is **multiplied by $N$**, the step count. With $N = 50$ and $c_\text{net} \approx 75$ ms, the loop alone is $50 \times 75 = 3750$ ms, which is about 96% of the 3.9-second total. The text encode (~25 ms) and VAE decode (~120 ms) together are under 4%.

This is the single most important fact in diffusion inference: **at typical step counts, latency is dominated by $N \cdot c_\text{net}$, and $N$ is the dial with the most leverage.** Halving $N$ nearly halves latency. Halving $c_\text{net}$ nearly halves latency. But shaving the text encode or VAE decode does almost nothing — until you have shrunk $N$ so far that those fixed costs finally start to matter (which, as we will see, is exactly what happens once you distill to four steps).

One subtlety the bare equation hides: with **classifier-free guidance** (CFG), each denoising step actually runs the network *twice* — once conditioned on the prompt, once unconditioned — and combines them. So the effective per-step cost is $2 c_\text{net}$ unless you batch the two passes together (which you should; more on that later). For the rest of this post I will fold the CFG factor into $c_\text{net}$ and note it explicitly when it matters. If CFG is new to you, [the classifier-free guidance post](/blog/machine-learning/image-generation/classifier-free-guidance) derives why that second forward pass exists and what it buys.

#### Worked example: profiling the SDXL latency budget

Let us make the numbers concrete for SDXL at 1024×1024 on an RTX 4090 (24 GB), fp16, with the default 50-step Euler sampler and CFG on (batched, so one forward per step covers both conditional and unconditional via a batch of 2).

- Text encode (CLIP-L + OpenCLIP-bigG, run once): **~25 ms**.
- Denoiser: SDXL's U-Net is ~2.6B parameters; one forward at 1024² (batch 2 for CFG) is **~75 ms**. Over 50 steps: **~3,750 ms**.
- VAE decode (the SDXL VAE, one pass): **~120 ms**.

Total: $25 + 3750 + 120 \approx 3{,}895$ ms, i.e. **~3.9 s/image**. The denoiser loop is **96.3%** of the budget. If a vendor tells you their "model" is slow, this is what they mean — not that one forward is slow, but that you do fifty of them in a strict sequence. The fix is never "buy a faster matrix multiply"; it is "do fewer, cheaper, or reused forwards."

These are representative figures for SDXL on a 4090 in fp16; exact numbers vary with driver, attention backend, and resolution, so treat them as order-of-magnitude. The *ratios* — loop dominates, overheads are small — are robust across pipelines and GPUs.

### 1.1 Is one forward compute-bound or memory-bound?

There is a deeper question hiding inside $c_\text{net}$ that decides *which* systems lever helps: is a single denoiser forward limited by raw arithmetic throughput (compute-bound) or by how fast the GPU can move weights and activations through memory (memory-bound)? The answer tells you whether quantization (which moves fewer bytes) or a faster matmul kernel (more FLOPs/s) is the win.

The tool here is the **roofline model** and its single most useful number, **arithmetic intensity**: FLOPs performed per byte of memory traffic. Every GPU has a ridge point — the arithmetic intensity at which it transitions from memory-bound (below) to compute-bound (above). An RTX 4090 does roughly 165 TFLOP/s of fp16 tensor-core compute against roughly 1 TB/s of memory bandwidth, so its ridge point sits around 165 FLOPs/byte. A kernel with lower intensity than that is starved waiting for memory; a kernel above it is genuinely crunching numbers.

For diffusion denoisers at the batch sizes typical of single-image generation (batch 1, or batch 2 for CFG), large parts of the network are **memory-bound**. The convolutions and linear layers process relatively few tokens, so the GPU spends much of its time reading the ~2.6B fp16 weights of SDXL (≈5.2 GB) from memory rather than multiplying. This is *why quantization helps so much*: dropping weights from fp16 to int8 halves the bytes you must read every forward, and on a memory-bound layer that nearly doubles throughput — with essentially no quality cost. It is also why batching helps: a larger batch reuses each weight read across more samples, raising arithmetic intensity and pushing the kernel toward the compute-bound regime where the hardware is fully utilized.

The attention blocks behave differently. Self-attention is $\mathcal{O}(n^2)$ in token count $n$, and at high resolution (many tokens) the attention matrix dominates and the layer becomes compute-bound *and* memory-heavy (materializing the $n \times n$ matrix). That is precisely the regime FlashAttention targets (avoid materializing the matrix) and the regime SANA's linear attention targets (make the cost $\mathcal{O}(n)$ instead of $\mathcal{O}(n^2)$). So the same network has memory-bound parts (the weight-heavy linear/conv layers at low batch) and compute-bound parts (attention at high resolution), and the levers split along exactly that seam: quantization and batching for the memory-bound parts, FlashAttention and linear attention for the compute-bound parts.

The practical upshot: before you optimize, know which regime you are in. At batch 1, single image, 1024² — the common interactive case — you are largely memory-bound, so quantization and a tiny VAE buy more than you might expect, and a raw FLOPs increase (a faster matmul) buys less. At large batch (serving), you tip compute-bound, and FlashAttention plus `torch.compile` fusion matter more. This is the kind of measurement that turns a guess into a plan.

## 2. Why you cannot just parallelize the steps

The obvious engineer's instinct, faced with "we call the network 50 times," is: *run them in parallel.* GPUs are massively parallel; surely we can fan out the 50 calls across the hardware and finish in the time of one?

You cannot, and the reason is structural. Diffusion sampling is a **sequential dependency chain**. The input to denoising step $t-1$ is the *output* of denoising step $t$. You cannot compute step 49 until you have finished step 50, because step 49's input latent $x_{49}$ is literally produced by step 50's scheduler update. There is no parallelism across the time axis to exploit; each link in the chain feeds the next.

![A branching dataflow graph showing a noisy latent feeding both the denoiser net and a residual carry, both merging in the scheduler step that produces the next latent, which feeds the next call](/imgs/blogs/why-diffusion-is-slow-and-how-to-fix-it-2.png)

To see this precisely, recall the DDIM/Euler update. The network predicts the noise $\epsilon_\theta(x_t, t, c)$ in the current latent $x_t$ (conditioned on text $c$), and the scheduler combines that prediction with $x_t$ to produce the next, slightly cleaner latent:

$$
x_{t-1} = \sqrt{\bar\alpha_{t-1}}\,\hat{x}_0 + \sqrt{1 - \bar\alpha_{t-1}}\;\epsilon_\theta(x_t, t, c),
\qquad
\hat{x}_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t,t,c)}{\sqrt{\bar\alpha_t}}.
$$

Notice $x_{t-1}$ depends on $\epsilon_\theta(x_t, \cdot)$, which depends on $x_t$, which was produced by the previous step. It is a recurrence. You can no more parallelize it than you can parallelize a long chain of `x = f(x)` assignments — each needs the previous result.

The image-embedded figure above shows exactly this: the noisy latent $x_t$ feeds *both* the denoiser network (which predicts the noise) and a residual carry term, and the scheduler step merges them to produce $x_{t-1}$, which becomes the input to the next call. The merge is the point where the chain re-forms; there is no way to break it without changing the math of sampling itself.

This is why every speed lever in this track attacks the problem differently than "parallelize." You cannot make the chain wider, so you make it **shorter** (fewer steps), make each **link cheaper** (faster net, fewer tokens), or **reuse work** between adjacent links (caching). What you *can* parallelize is *within* a single step — the matrix multiplies of one forward pass are already running across thousands of CUDA cores, and you can batch multiple *images* together (one big batch, all denoising in lockstep). But the depth of the chain — the $N$ in our cost model — is irreducible without changing the algorithm. Step distillation, which we cover next, is precisely the art of changing the algorithm so the chain can be short without the image turning to mush.

A natural question: if it is sequential, why is one image not just one long-running kernel? Because between forwards there is a tiny scheduler computation (the update equation above) on the CPU/GPU boundary, plus the data dependency forces a sync. Those inter-step gaps are small compared to the forward itself, but they are why naively chasing kernel-fusion inside one step gives you a 10–20% win, while cutting the step count gives you a 10× win. The leverage is in $N$.

## 3. The four levers: a map of the whole track

Given the cost model $\text{latency} = c_\text{text} + N \cdot c_\text{net} + c_\text{vae}$, there are exactly four places to push, and the rest of this track is one post per lever. Here is the map.

![A tree rooted at cutting sampling latency, branching into fewer steps, a faster denoiser, deeper compression, and feature caching, each with a concrete technique leaf](/imgs/blogs/why-diffusion-is-slow-and-how-to-fix-it-3.png)

**Lever 1 — Fewer steps (shrink $N$).** This is the biggest single win, because $N$ is the multiplied term. The technique is **step distillation**: train a student model that reproduces what the 50-step teacher produces, but in 1–4 steps. Consistency models, LCM, SDXL-Turbo, and DMD all live here. Going from 50 to 4 steps is a >10× reduction in the dominant term. This is covered in [consistency models and few-step generation](/blog/machine-learning/image-generation/consistency-models-and-few-step-generation) and [distribution matching and adversarial distillation](/blog/machine-learning/image-generation/distribution-matching-and-adversarial-distillation).

**Lever 2 — A faster denoiser (shrink $c_\text{net}$).** Make each forward pass cheaper: a more efficient architecture (linear attention instead of quadratic, as in SANA), pruning, or distilling a big teacher into a smaller student net. This multiplies with lever 1 — a 2× cheaper net on top of 10× fewer steps is 20×. The architecture side is covered in [diffusion transformers (DiT)](/blog/machine-learning/image-generation/diffusion-transformers-dit) and [the modern MM-DiT recipe](/blog/machine-learning/image-generation/mmdit-and-the-modern-text-to-image-recipe).

**Lever 3 — Deeper latent compression (fewer tokens per call).** Diffusion in latent space already shrinks the image (an 8× VAE turns 1024² pixels into a 128² latent). Compress harder — SANA's deep-compression autoencoder is **32×**, giving a 32² latent — and each token the transformer processes shrinks by 16× in count. Since attention is quadratic in token count, that is a large cut to $c_\text{net}$ on the attention side. This sits at the boundary of levers 2 and 3 and is part of [the modern frontier recipe](/blog/machine-learning/image-generation/mmdit-and-the-modern-text-to-image-recipe).

**Lever 4 — Feature caching (reuse across steps).** Adjacent denoising steps produce very similar intermediate features, especially in the deep layers and at low-to-mid noise levels. **DeepCache** and **TeaCache** detect when features barely change and reuse the previous step's computation instead of redoing it, skipping a large fraction of the network's work on many steps. This buys ~1.5–2.5× with near-zero quality cost and composes with everything else. Covered in [quantization, caching, and efficient inference](/blog/machine-learning/image-generation/quantization-caching-and-efficient-inference).

On top of these four **algorithmic** levers sit the **orthogonal systems levers** — things that speed up the *implementation* without changing the math at all: **quantization** (run the net in int8/fp8 instead of fp16 — fewer bytes moved, faster matmuls), **FlashAttention / SDPA** (a memory-efficient attention kernel), **`torch.compile`** (kernel fusion and graph optimization), and **batching** (amortize fixed costs over many images). These are nearly free quality-wise and stack with the four big levers. They are the "always turn these on first" wins, and we cover them in section 7 and in the [edge-AI quantization deep-dives](/blog/machine-learning/edge-ai/quantization-from-first-principles).

The mental model to carry forward: latency is a *product* of factors, and each lever cuts a different factor. They multiply. A model that is 10× fewer steps × 2× faster net × 2× from caching × 1.5× from int8 is **60×** faster than the baseline — which is roughly how a 4-second image becomes a sub-100-ms one on the same GPU.

### 3.1 The "faster denoiser" lever, unpacked

Lever 2 is the broadest of the four because "make each forward cheaper" has several distinct sub-techniques, and they are easy to conflate. It is worth separating them, because they have different costs and combine differently with the others.

**Architecture.** Choose a cheaper backbone. The big lever here is the *attention mechanism*: standard softmax self-attention is $\mathcal{O}(n^2)$ in tokens, while linear-attention variants are $\mathcal{O}(n)$. SANA's denoiser uses linear attention precisely to make $c_\text{net}$ scale gracefully with resolution. Other architectural choices — fewer transformer blocks, narrower hidden dimension, more efficient convolutions — trade capacity for speed and must be validated against quality. Architecture changes are a *training-time* decision: you cannot bolt linear attention onto a pretrained softmax model and expect it to work, because the weights were trained for the old attention. This is why lever 2 mostly lives in new model designs (SANA, efficient DiT variants), not in post-hoc tweaks.

**Pruning.** Remove weights or whole structures (heads, channels, blocks) that contribute little, then fine-tune to recover quality. Structured pruning (removing entire channels or blocks) gives real speedups because it shrinks the actual matmul dimensions; unstructured pruning (zeroing individual weights) gives memory savings but needs sparse-kernel hardware support to translate into speed. Pruning a diffusion denoiser is delicate — over-prune and you lose the fine detail that makes images look good — but a modest structured prune plus a short fine-tune can cut $c_\text{net}$ by 20–40% at small quality cost. The [edge-AI pruning fundamentals post](/blog/machine-learning/edge-ai/pruning-fundamentals) covers the mechanics that transfer directly here.

**Distillation into a smaller net.** This is where lever 2 meets lever 1: you can distill not just to *fewer steps* but to a *smaller student network*. A compact student trained to match a large teacher's outputs gives a cheaper $c_\text{net}$ *and*, if you distill for few steps simultaneously, a smaller $N$ — the two wins compound. The risk is that a too-small student cannot represent the teacher's full distribution, so quality and diversity drop; the art is finding the smallest student that still clears your quality bar. [The knowledge-distillation fundamentals post](/blog/machine-learning/edge-ai/knowledge-distillation-fundamentals) covers the general recipe.

The reason to separate these is that they sit at different points in your workflow. Architecture is decided when you choose or design the model. Pruning and small-net distillation are post-training compression you apply to a checkpoint you already have. And all three are *orthogonal to* the systems levers (quantization, compile) — a pruned, distilled, linear-attention net still benefits from int8 and `torch.compile` on top. The factors keep multiplying.

## 4. The science: why fewer steps is the highest-ROI lever

It is not arbitrary that step distillation is lever number one. There is a quantitative reason: **quality saturates with step count**. Past a fairly low number of steps, more steps buy you almost nothing in image quality but cost you linearly in latency. So the marginal value of a step collapses, while its marginal cost stays flat. That asymmetry is exactly what makes "use fewer steps" the highest return-on-investment lever.

Here is the why, grounded in the [ODE view of sampling](/blog/machine-learning/image-generation/ddim-and-fast-deterministic-sampling). DDIM and its descendants treat sampling as numerically integrating an ordinary differential equation — the probability-flow ODE — from pure noise at $t=T$ to a clean image at $t=0$. The step count $N$ is the number of points at which you evaluate that ODE. A numerical integrator's error shrinks as you add steps, but with **diminishing returns**: the global truncation error of a first-order method like Euler scales like $\mathcal{O}(1/N)$, so going from 25 to 50 steps roughly halves an already-small error, while going from 4 to 8 steps fixes a *large* error. The curve is steep at the low end and flat at the high end.

Empirically, for a well-tuned sampler like DPM-Solver++ or UniPC, FID drops sharply from 4 to ~20 steps and then is essentially flat from 20 to 50. The image at 30 steps and the image at 50 steps are perceptually identical; you paid 20 extra forward passes — ~1.5 extra seconds on a 4090 — for a FID change in the noise.

![A matrix of step counts against latency, FID, and marginal gain, showing FID falling steeply from 4 to 20 steps then flattening](/imgs/blogs/why-diffusion-is-slow-and-how-to-fix-it-6.png)

So why not just always run 20 steps with a good sampler and call it done? Because there is a hard floor that *no sampler* can cross: even the best ODE solver still needs enough function evaluations to integrate the trajectory accurately, and below ~8–10 steps the discretization error becomes visible as blur, color shift, and structural mush. A 4-step image from a vanilla SDXL + DPM-Solver++ is genuinely bad.

This is the gap that **step distillation** fills, and why it is a separate, more powerful idea than "just use a better sampler." A distilled student is not solving the same ODE more cleverly — it is **learning a new map** that jumps from noise to image (or across large chunks of the trajectory) in one shot. It is trained so that one forward pass of the student equals many forward passes of the teacher. That breaks the $\mathcal{O}(1/N)$ floor entirely, because the student is not doing numerical integration anymore; it has *amortized* the integration into its weights. That is how Turbo and LCM hit 1–4 steps with quality that a non-distilled model needs 30+ steps to match. The science says: spend your engineering on distillation, not on squeezing two more steps out of your solver.

### 4.1 The consistency property, made precise

It is worth seeing the actual mechanism, because it explains *why* one forward can replace many — and the math is clean. Recall that deterministic sampling follows the probability-flow ODE, which defines a trajectory $\{x_t\}_{t \in [0, T]}$ from a clean image $x_0$ at $t=0$ to (nearly) pure noise $x_T$ at $t=T$. For any point on a given trajectory, there is a single endpoint $x_0$ it flows to.

A **consistency function** $f_\theta(x_t, t)$ is defined to map *any* point on a trajectory directly to that trajectory's clean endpoint:

$$
f_\theta(x_t, t) \approx x_0 \quad \text{for all } t \text{ on the trajectory ending at } x_0.
$$

Two properties define it. First, a **boundary condition**: at $t \to 0$ the function is the identity, $f_\theta(x_0, 0) = x_0$, because a clean image maps to itself. Second, **self-consistency**: any two points $x_t$ and $x_{t'}$ on the *same* trajectory must map to the *same* output, $f_\theta(x_t, t) = f_\theta(x_{t'}, t')$. If a network truly satisfies this, then sampling is trivial — draw noise $x_T$, evaluate $f_\theta(x_T, T)$ once, and you land on the clean image in a single forward pass. That is the one-step dream, and it is not a heuristic; it is a direct consequence of the definition.

How do you train a network to satisfy self-consistency? **Consistency distillation** uses the teacher's ODE solver to take a single small step. Sample a clean image, noise it to a level $x_{t_{n+1}}$, run one ODE-solver step of the *teacher* backward to get a slightly cleaner $\hat{x}_{t_n}$, and then enforce that the student maps both points to the same place:

$$
\mathcal{L}_\text{CD} = \mathbb{E}\!\left[\, d\big(\, f_\theta(x_{t_{n+1}}, t_{n+1}),\; f_{\theta^-}(\hat{x}_{t_n}, t_n)\,\big)\,\right],
$$

where $d$ is a distance (e.g. squared error or LPIPS), and $\theta^-$ is an exponential-moving-average "target" copy of the student's own weights (a stop-gradient target, like in self-distillation, to keep training stable). The loss says: *adjacent points on a trajectory must agree.* If every adjacent pair agrees, then by transitivity *all* points on the trajectory agree — including the noisy endpoint and the clean one. The network has internalized the whole integration into a single evaluation.

This is the rigorous answer to "why can one forward replace fifty?" It is not that the network suddenly computes faster; it is that you have trained it on a *different objective* — endpoint consistency rather than per-step noise prediction — so that one evaluation is mathematically the answer the iterative solver was converging to. Latent Consistency Models apply exactly this in latent space (so the distance is computed on cheap latents), which is why LCM made the technique practical for SD/SDXL. The cost you pay is upfront training compute and a small quality gap from imperfect consistency; the cost you save is 90%+ of inference forwards, forever. That is the trade the rest of the Speed track quantifies in detail.

#### Worked example: the amortization math of distillation

Distillation moves a one-time training cost into a permanent inference saving, so the break-even is worth computing. Suppose distilling an LCM-LoRA for SDXL costs, roughly, a few hundred A100-GPU-hours of training (representative order of magnitude). Say it is 300 GPU-hours — about \$600 at \$2/hr.

Now the saving: each generated image drops from 50 forwards to 4, saving 46 forwards × ~75 ms ≈ 3.45 s of GPU time per image. At \$2/hr (≈ \$0.00056/s), that is about **\$0.0019 saved per image**. The distillation pays for itself after $600 / 0.0019 \approx 316{,}000$ images — which, for any product generating images at scale, is a *single day* of traffic. After that, every image is pure saving, and the latency win (8.7×) is there from the very first inference. This is the financial shape of lever 1: a bounded, one-time training cost amortized against an unbounded stream of inference savings. It is why distillation is not a research curiosity but the default first move for anyone serving diffusion at volume — the ROI is not close.

#### Worked example: the diminishing-returns arithmetic

Suppose, for SDXL with a good sampler on a 4090, FID-vs-reference behaves roughly like this (representative, not exact): 4 steps → FID ~22, 8 → ~14, 20 → ~10, 30 → ~9.6, 50 → ~9.4. Latency is linear: ~0.075 s/step plus ~0.145 s of fixed overhead.

- Going **4 → 20 steps**: FID improves by ~12 points; latency rises from ~0.45 s to ~1.65 s (+1.2 s). Cost per FID point: ~0.1 s. **Worth it.**
- Going **20 → 50 steps**: FID improves by ~0.6 points; latency rises from ~1.65 s to ~3.9 s (+2.25 s). Cost per FID point: ~3.75 s. **Not worth it** — you are paying 37× more latency per unit of quality.

Now compare to a distilled 4-step model (LCM/Turbo) that hits FID ~12–14 at 4 steps: it matches the *8-step* non-distilled quality at *4 steps*, and it does so with a sampler that was *trained* to be stable at 4 steps. You get 30-step-class images in ~0.45 s. That is the ROI argument for lever 1 in one paragraph: distillation moves the whole quality-vs-steps curve left, so you operate at a step count that was previously a quality cliff.

## 5. Practical flow: profiling a diffusers run

Theory is cheap; let us measure. Here is how to profile a real `diffusers` sampling run and split the time into the three terms of the cost model. The key is to use **CUDA events** for timing (not Python's `time.time()`), because GPU kernels are launched asynchronously — the CPU returns immediately while the GPU is still working, so wall-clock timing on the CPU lies. CUDA events record on the GPU stream and give you true device time, with `torch.cuda.synchronize()` to force completion before you read the clock.

```python
import torch
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
).to("cuda")

prompt = "a golden retriever puppy in a field of wildflowers, soft morning light"

def cuda_time(fn, warmup=2, iters=5):
    # Warm up: first calls pay one-time compile/alloc costs we do not want to measure.
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # milliseconds
    return sum(times) / len(times)

# Full pipeline at a fixed step count, fixed seed for reproducibility.
gen = torch.Generator("cuda").manual_seed(0)
full_ms = cuda_time(lambda: pipe(prompt, num_inference_steps=50, generator=gen).images[0])
print(f"full 50-step pipeline: {full_ms:.0f} ms")
```

To attribute the time to each term, profile the **pieces** separately. The cleanest way is to time the text encoder, one U-Net forward, and the VAE decode in isolation:

```python
# 1. Text encode (run once per image).
with torch.no_grad():
    text_ms = cuda_time(lambda: pipe.encode_prompt(prompt, device="cuda",
                                                    num_images_per_prompt=1,
                                                    do_classifier_free_guidance=True))

# 2. One U-Net forward at the working resolution (batch 2 for CFG: cond + uncond).
latent = torch.randn(2, 4, 128, 128, dtype=torch.float16, device="cuda")
t = torch.tensor([500], device="cuda")
emb, _, pooled, _ = pipe.encode_prompt(prompt, device="cuda",
                                        num_images_per_prompt=1,
                                        do_classifier_free_guidance=True)
added = {"text_embeds": pooled.repeat(2, 1),
         "time_ids": torch.zeros(2, 6, dtype=torch.float16, device="cuda")}
with torch.no_grad():
    unet_ms = cuda_time(lambda: pipe.unet(latent, t, encoder_hidden_states=emb,
                                          added_cond_kwargs=added).sample)

# 3. VAE decode (run once at the end).
with torch.no_grad():
    z = torch.randn(1, 4, 128, 128, dtype=torch.float16, device="cuda")
    vae_ms = cuda_time(lambda: pipe.vae.decode(z / pipe.vae.config.scaling_factor).sample)

N = 50
print(f"text encode:   {text_ms:7.1f} ms  (x1)")
print(f"unet forward:  {unet_ms:7.1f} ms  (x{N})  -> loop {N*unet_ms:.0f} ms")
print(f"vae decode:    {vae_ms:7.1f} ms  (x1)")
print(f"model => {text_ms + N*unet_ms + vae_ms:.0f} ms  (matches full within scheduler overhead)")
```

On a 4090 you will see something like `text encode ~25 ms`, `unet forward ~75 ms` (so the 50-step loop is ~3,750 ms), `vae decode ~120 ms`. The reconstructed total matches the full-pipeline number to within the small per-step scheduler overhead. Now you have *proof*, on your own hardware, of where the seconds go — and you can re-run this after each optimization to see exactly which term you moved.

To see the step-count scaling directly, sweep $N$:

```python
for n in [4, 8, 20, 30, 50]:
    g = torch.Generator("cuda").manual_seed(0)
    ms = cuda_time(lambda: pipe(prompt, num_inference_steps=n, generator=g).images[0],
                   warmup=1, iters=3)
    print(f"{n:3d} steps -> {ms:6.0f} ms")
```

You will see latency rise almost perfectly linearly in $N$ with a constant offset (the fixed $c_\text{text} + c_\text{vae}$ term), which is the cost model made visible. The slope is $c_\text{net}$; the intercept is the overhead. This single sweep is the most useful diffusion-profiling experiment you can run, and it takes thirty seconds.

A good habit is to plot the sweep so the linear-in-$N$ relationship is undeniable. Here is the minimal plotting code — note the escaped dollar signs in the mathtext label, which is the one place a stray `$` will bite you in a Matplotlib script:

```python
import matplotlib.pyplot as plt

steps = [4, 8, 20, 30, 50]
latency_ms = [450, 750, 1650, 2400, 3900]   # fill from your own cuda_time sweep

plt.figure(figsize=(5, 3.2))
plt.plot(steps, latency_ms, "o-")
plt.xlabel("num_inference_steps (N)")
plt.ylabel("latency (ms)")
# Paired dollar signs in mathtext MUST be escaped as \\$ inside a Python string.
plt.title(r"Latency $\\approx c_\\mathrm{text} + N\\,c_\\mathrm{net} + c_\\mathrm{vae}$")
plt.tight_layout()
plt.savefig("latency_sweep.png", dpi=140)
```

The fitted slope of that line is your measured $c_\text{net}$, and the y-intercept is your measured $c_\text{text} + c_\text{vae}$. If the line is *not* straight — if it curves upward at high $N$ — you have a memory issue (activations spilling, recompilation, or thermal throttling), which is itself useful to know.

### 5.1 Measuring VRAM, the other budget

Latency is one axis; **memory** is the other, and it is the one that decides whether a model runs *at all* on a given card. Measuring peak VRAM is as important as measuring latency, and `torch.cuda` gives it to you directly:

```python
torch.cuda.reset_peak_memory_stats()
g = torch.Generator("cuda").manual_seed(0)
_ = pipe(prompt, num_inference_steps=20, generator=g).images[0]
peak_gb = torch.cuda.max_memory_allocated() / 1e9
print(f"peak VRAM: {peak_gb:.2f} GB")
```

For SDXL in fp16 you will see roughly 8–10 GB at 1024² (weights ~5 GB plus activations and the VAE), which is why it fits a 12 GB card but is tight on 8 GB. The memory levers attack this number: int8 quantization roughly halves the weight contribution (~2.5 GB saved), `enable_vae_slicing()` cuts the VAE decode's activation spike, and `enable_model_cpu_offload()` trades latency to keep only the active submodule resident. When someone says "I can't run FLUX on my card," they are hitting *this* budget, not the latency one — FLUX's ~12B-parameter transformer is ~24 GB in fp16, which is why 4-bit quantization (SVDQuant) is the difference between "runs on a 4090" and "does not."

### 5.2 Swapping the sampler is a free step-count win

Before touching distillation, swap the scheduler — it is a one-line change that moves you left on the step-count curve for free. A higher-order solver like DPM-Solver++ or UniPC reaches the same quality as Euler in roughly half the steps, because it uses the network evaluations more efficiently (it fits a higher-order local model of the ODE):

```python
from diffusers import DPMSolverMultistepScheduler, UniPCMultistepScheduler

# Swap the default Euler/DDIM scheduler for a higher-order multistep solver.
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
img = pipe(prompt, num_inference_steps=20).images[0]   # ~Euler-50 quality at 20 steps
```

This is not distillation — it does not break the $\mathcal{O}(1/N)$ floor — but it is the cheapest possible left-shift, and you should do it before anything else. [The samplers deep dive](/blog/machine-learning/image-generation/samplers-deep-dive) derives exactly why a second-order solver halves the step count for the same error, and where even good solvers hit the cliff (around 8–10 steps) that only distillation crosses.

For quick reference, here is roughly where the common schedulers land on the step-count-for-equal-quality axis (representative; tune for your model):

| Scheduler | Order | Steps for "good" SDXL | Notes |
|---|---|---|---|
| `DDIMScheduler` | 1 | ~30–50 | Deterministic, simple, the baseline. |
| `EulerDiscreteScheduler` | 1 | ~30–50 | Default in many pipelines; first-order. |
| `DPMSolverMultistepScheduler` | 2 (multistep) | ~20–25 | High-order; reuses past evals — strong default. |
| `UniPCMultistepScheduler` | up to 3 | ~15–20 | Often the fewest steps before distillation. |
| `LCMScheduler` | — (distilled) | ~4 | Requires an LCM-distilled model/LoRA; lever 1. |

The pattern is clear: moving down the table buys fewer steps for the same quality, but the first four rows are all still *numerical solvers* bounded by the $\mathcal{O}(1/N)$ floor — they cluster between 15 and 50 steps. Only the last row, which requires a *distilled model*, breaks through to single-digit steps. That visual break in the table is exactly the lever-1-vs-everything-else distinction the whole track turns on: better solvers optimize *within* the floor; distillation removes the floor.

## 6. The before/after that motivates the whole track: 50 → 4 steps

Let us look at the payoff that lever 1 delivers, because it reframes the entire optimization problem. When you distill a 50-step model to a 4-step one, you collapse the dominant loop term by an order of magnitude — and that has a surprising second-order effect.

![A before-after comparison of a 50-step pipeline dominated by the denoiser loop versus a 4-step pipeline where the fixed VAE decode and text encode become visible](/imgs/blogs/why-diffusion-is-slow-and-how-to-fix-it-4.png)

At 50 steps, the loop is ~3,750 ms out of ~3,895 ms total — 96% of the budget — and the text encode and VAE decode are rounding error. At 4 steps, the loop is ~300 ms (4 × 75 ms), and the **fixed overheads suddenly matter**: the VAE decode (~120 ms) and text encode (~25 ms) are now ~33% of a ~450-ms total. The bottleneck has *moved*.

This is the most important strategic insight for anyone optimizing a diffusion stack: **the bottleneck shifts as you optimize.** Profiling once at the start is not enough. After you distill to few steps, the VAE decode becomes a real fraction of latency, so suddenly **lever 3 (a cheaper decoder / deeper AE)** and a **tiny VAE** (like `AutoencoderTiny`, which decodes ~10× faster than the full SDXL VAE at a small quality cost) move from "irrelevant" to "worth doing." Likewise the text encode — negligible at 50 steps — becomes worth caching or skipping (reuse the same prompt embedding across a batch). The order in which you pull the levers is itself an optimization: pull the biggest one (steps) first, *re-profile*, then attack whatever became dominant.

#### Worked example: the moving bottleneck

Start: SDXL, 50 steps, ~3,895 ms. Loop 96%, VAE 3%, text 0.6%.

1. **Distill to 4 steps** (SDXL-Turbo / LCM): loop → ~300 ms. New total ~450 ms (~8.7× faster). Loop now 67%, VAE 27%, text 6%.
2. Re-profile. VAE decode is now the second-biggest term. **Swap to `AutoencoderTiny`** (TAESD): decode ~12 ms. New total ~340 ms. Loop 88%, VAE 4%.
3. Re-profile. Loop dominates again at a *much* smaller absolute size. **Add a faster/quantized net or caching**: net 75 → ~45 ms with int8 + a cheaper attention. Loop → ~180 ms. Total ~225 ms (~17× over baseline).

Each step re-profiled, each lever applied where it now bites. That discipline — measure, pull the biggest lever, re-measure — is the difference between a 2× hack and a 17× rebuild.

## 7. The orthogonal systems levers (turn these on first)

Before you do anything fancy, there are four implementation-level wins that change *no math* and cost *no quality* (or nearly none). They are the table stakes of fast diffusion, and you should enable them before reaching for distillation.

**Memory offload, when VRAM-bound.** If your model does not fit comfortably in VRAM, `enable_model_cpu_offload()` keeps only the currently-active submodule on the GPU and streams the rest from CPU. This trades a little latency for a lot of headroom — it is how you run SDXL or FLUX on a 12 GB card. On a card with ample VRAM, leave it off (it adds transfer latency).

```python
# Systems wins, in the order you should try them:
pipe.enable_model_cpu_offload()          # only if VRAM-constrained; saves memory, costs a little time
pipe.enable_vae_slicing()                # decode the VAE in slices -> lower peak memory at large res
pipe.enable_vae_tiling()                 # tile the VAE for very high resolutions (e.g. 2048+)

# Attention: PyTorch SDPA picks FlashAttention/mem-efficient kernels automatically on recent GPUs.
# It is the default in modern diffusers; this is just to be explicit.
pipe.unet.set_attn_processor  # (modern diffusers uses SDPA by default; no manual xformers needed)
```

**FlashAttention / SDPA.** Attention is a big chunk of $c_\text{net}$, and the naive implementation materializes the full $n \times n$ attention matrix in memory. `torch.nn.functional.scaled_dot_product_attention` (SDPA) dispatches to FlashAttention or a memory-efficient kernel that never materializes that matrix, cutting both memory and time. Modern `diffusers` uses SDPA by default, so on a recent PyTorch + GPU you already have this — but verify, because an old `xformers` path or a forced eager attention will silently cost you 20–40%.

**`torch.compile`.** This fuses kernels and optimizes the graph, typically buying **1.2–2×** on the denoiser with zero quality change. The catch is a one-time compile cost (tens of seconds) and that it recompiles if input shapes change — so compile once with fixed resolution and batch size.

```python
import torch
# Compile the hot path. mode="max-autotune" searches for the best kernels (longer compile, faster runtime).
pipe.unet = torch.compile(pipe.unet, mode="max-autotune", fullgraph=True)
# First call compiles (slow); subsequent calls at the same shape are fast.
_ = pipe(prompt, num_inference_steps=4)   # warm-up / compile
img = pipe(prompt, num_inference_steps=4) # fast
```

**Quantization.** Running the net in int8 or fp8 instead of fp16 moves half (or a quarter) of the bytes and uses faster low-precision matmul units, buying ~1.5–3× on memory-bound parts with near-zero quality loss for int8 (more care needed for 4-bit). For diffusion this is covered in [the caching-and-quantization post](/blog/machine-learning/image-generation/quantization-caching-and-efficient-inference) and, for the underlying numerics, the [edge-AI quantization series](/blog/machine-learning/edge-ai/post-training-quantization-ptq). The short version: `optimum-quanto` or `bitsandbytes` will quantize a `diffusers` transformer in a few lines, and SVDQuant-style 4-bit diffusion is the 2025 frontier for fitting big models on consumer cards.

**Batching.** The fixed costs ($c_\text{text}$, and the per-launch kernel overhead) amortize over a batch. Generating 8 images in one batch is far cheaper per-image than 8 separate calls, because the GPU is better utilized and the overheads are shared. If you are serving, batch aggressively up to the point where you saturate VRAM.

These five — offload (when needed), SDPA, compile, quantization, batching — are the "free" tier. Turn them on, *then* decide whether you need the algorithmic levers. For a real-time or high-volume product, you will need both: systems wins for the constant factors, distillation for the order-of-magnitude.

### 7.1 Why these are "free" and the levers are not

It is worth being precise about what "free" means, because it is the property that lets you reach for these first without thinking. The systems levers are **mathematically transparent**: they compute the same function, just with a faster implementation. `torch.compile` fuses kernels but produces (up to floating-point rounding) the same outputs. SDPA computes the same attention as the naive path, just without materializing the score matrix. int8 quantization introduces a tiny numerical error that, for well-behaved weight distributions, is below perceptual threshold for image quality. Batching changes nothing about the math at all. So you can stack all of them and the *only* thing that changes is the clock — there is no quality knob to tune, no retraining, no new failure mode (beyond the 4-bit caveat). That is what "free" means and why they are always step one.

The four algorithmic levers are *not* free in this sense, because each one **changes the function being computed** in service of speed:

- Distillation replaces the iterative solver with a learned map — a *different* function that approximates the teacher. It can lose diversity and has a guidance sweet spot.
- A smaller/faster denoiser is a *different, lower-capacity* network — it can lose fine detail unless re-distilled carefully.
- A deeper AE means the denoiser operates in a *different, more lossy* latent space — capped by reconstruction quality.
- Caching reuses stale features — an *approximation* that is wrong by however much the features actually changed.

This is the cleanest way to decide order of operations: do every transparent (free) optimization first, measure, and only then spend quality budget on the algorithmic levers in proportion to how much speed you still need. If the free tier already meets your latency SLO, you never touch the quality-costing levers at all — which is the right outcome for quality-critical workloads.

### 7.2 The attention term, concretely

Because attention is the part of $c_\text{net}$ that scales worst with resolution, it is worth seeing the numbers. Self-attention over $n$ tokens costs $\mathcal{O}(n^2 d)$ FLOPs and, naively, $\mathcal{O}(n^2)$ memory to hold the score matrix. At 1024² with an 8× VAE and patch size 2, a DiT processes on the order of a few thousand tokens; the $n^2$ term is large enough that attention is a meaningful slice of each forward, and it grows *quadratically* if you push to 2048². This is why three different optimizations all target attention: FlashAttention removes the $\mathcal{O}(n^2)$ *memory* (it tiles the computation so the score matrix never fully exists), linear attention removes the $\mathcal{O}(n^2)$ *compute* (it reformulates attention as $\mathcal{O}(n)$ via a kernel feature map, at some quality cost), and deep latent compression removes the *tokens themselves* (fewer $n$ means less of everything). They are complementary: SANA uses both linear attention and a 32× AE, so it attacks the quadratic term from two directions at once. For a workload that pushes high resolution, the attention term is where the marginal engineering pays off most, and knowing it is quadratic tells you that doubling resolution roughly *quadruples* the attention cost — a fact that should shape your resolution choices, not just your kernel choices.

## 8. Where the levers attach, and why they compose

A crucial property of the four levers is that they act on **different stages of the pipeline**, which is exactly why they multiply instead of overlapping.

![A graph showing the sampler loop, denoiser net, deep AE latent, and feature cache attaching at distinct stages and merging into a fast pipeline whose speedups multiply](/imgs/blogs/why-diffusion-is-slow-and-how-to-fix-it-8.png)

- **Fewer steps** acts on the *sampler loop* — it changes $N$.
- **A faster denoiser** and **deeper latent compression** act on the *per-call cost* $c_\text{net}$ — one shrinks the network, the other shrinks the number of tokens the network processes.
- **Feature caching** acts *across* calls — it skips redundant computation between adjacent steps.

Because each cuts a different factor in $\text{latency} = c_\text{text} + N \cdot c_\text{net} + c_\text{vae}$, their effects compound. Concretely: 50 → 4 steps is ~12× on the loop; a 2× cheaper net (deeper AE + linear attention) doubles that to ~24×; caching adds ~1.5× to ~36×; int8 adds ~1.5× to ~54×. You do not get all of these for free — distillation costs training compute and a little quality, deeper AE is bounded by the autoencoder's own reconstruction quality — but the *structure* is multiplicative, and that is why the frontier models stack all of them.

This composition is also why **latent compression** is such a leveraged move. The denoiser's attention cost scales with the **square** of the token count (self-attention is $\mathcal{O}(n^2)$ in sequence length $n$). An 8× VAE turns a 1024² image into a 128² latent = 16,384 tokens at patch size 1 (fewer after patchify). SANA's 32× deep-compression AE turns it into a 32² latent = 1,024 tokens — a 16× reduction in token count, which is up to ~256× less attention compute (before the linear-attention trick that SANA also uses). That is why deep compression is not a marginal tweak; it attacks the quadratic term directly.

![A stack showing a pixel image compressed by an 8x VAE into 16384 tokens versus a 32x deep autoencoder into 1024 tokens, with attention cost scaling as the square of token count](/imgs/blogs/why-diffusion-is-slow-and-how-to-fix-it-7.png)

The trade-off, and there is always one: the deeper you compress, the harder the autoencoder's job, and the more its reconstruction quality bounds the *whole pipeline's* quality. A 32× AE that reconstructs poorly will cap your image quality no matter how good the denoiser is, because the denoiser only ever sees and produces latents — the VAE decoder is the last word on pixels. SANA's contribution was training a 32× AE that *also* reconstructs well, which is genuinely hard. So lever 3 is bounded by autoencoder research in a way that lever 1 is not. This is the kind of trade-off the rest of the track quantifies.

### 8.1 Why feature caching works: temporal redundancy across steps

Lever 4 deserves its own mechanism, because it is the one that feels like a free lunch and very nearly is. The observation behind DeepCache is empirical but robust: across two *adjacent* denoising steps, the deep, low-resolution features of the U-Net change very little. The model is making a small refinement to a latent that is already mostly formed, so the high-level "what is in this image" representation is nearly identical from step $t$ to step $t-1$, even though the final output changes.

Concretely, in a U-Net the encoder compresses to a low-resolution bottleneck, and the decoder upsamples back. DeepCache computes the expensive bottleneck (the deepest, most parameter-heavy block) only on a subset of steps — say every other step — and on the in-between steps it *reuses* the cached bottleneck features, recomputing only the cheap high-resolution decoder layers that the skip connections feed. Since the deep block is where most of the FLOPs live, skipping it on half the steps gives close to a 2× speedup, and because the deep features genuinely barely changed, the quality cost is small.

The honest caveat is *when* the cached features stop being valid. Early in sampling (high noise) the latent is changing rapidly step-to-step, so caching there hurts more; late in sampling the features are stable and caching is nearly free. So good caching schemes are *adaptive*: TeaCache estimates how much the model's output is changing (a timestep-aware signal) and caches more aggressively when the change is small, less when it is large. That adaptivity is why TeaCache extends cleanly to DiT models, which do not have the U-Net's explicit bottleneck but do still exhibit slow-changing features across steps. The lever is pure inference-time computation reuse — no retraining, no new weights — which is what makes it stack on top of distillation, quantization, and everything else: it is the one lever you can add to a *finished* pipeline and still get a multiplier.

There is a natural ceiling, though, and it is worth stating because it bounds when lever 4 is worth it. Caching saves work by *reusing across steps* — so the fewer steps you have, the less there is to reuse. On a 50-step model, caching the deep block on 25 of those steps is a big win. On a 4-step distilled model, there are barely any adjacent steps to cache across, and the per-step latents are changing a lot (each distilled step covers a big chunk of the trajectory), so caching buys little. This is the general pattern with the levers: lever 1 (fewer steps) partially *consumes* the headroom that lever 4 (caching) needs. They still compose, but the caching multiplier shrinks as the step count drops — which is one more reason to re-profile after distilling rather than assuming the levers add up arithmetically.

## 9. The speed↔quality Pareto frontier (the recurring frame)

Everything in this track lives on one chart in your head: **quality on one axis, latency on the other, and a frontier of the best achievable trade-offs.** Every technique is a point or a curve on this plane, and the question is never "is this fast?" but "does this push the frontier — is it faster *at the same quality*, or better *at the same latency*?"

The baseline frontier is the **step-count curve**: 50-step SDXL is high-quality and slow; drop steps and you slide down-and-left (faster, worse) along a curve that falls off a cliff below ~8 steps. A better *sampler* (DPM-Solver++ vs Euler) shifts that curve left — same quality at fewer steps. **Step distillation** does something more dramatic: it creates a *new* curve far to the left, because the distilled model hits good quality at 1–4 steps. A **faster net** or **deeper AE** shifts every point down (lower latency at each step count). **Caching** shifts points down with almost no quality cost.

So the levers are not competing — they are each a different transformation of the frontier:

| Lever | Effect on the speed↔quality plane | Typical magnitude | Quality cost |
|---|---|---|---|
| Fewer steps (distillation) | New curve far left; good quality at 1–4 steps | 10–50× on the loop | Low–moderate (improving fast) |
| Faster denoiser (arch/prune) | Shifts every point down | 1.5–3× per call | Low if re-distilled |
| Deeper latent compression | Shifts every point down (attention term) | 2–4× tokens, more on attention | Bounded by AE quality |
| Feature caching | Shifts points down, tiny quality cost | 1.5–2.5× | Very low |
| Quantization (int8/fp8) | Shifts points down (memory-bound parts) | 1.5–3× memory | Near-zero (int8) |
| `torch.compile` / SDPA | Shifts points down (constant factor) | 1.2–2× | None |

![A matrix of the six levers against typical speedup, quality cost, and which post covers each, showing step distillation as the highest-leverage row](/imgs/blogs/why-diffusion-is-slow-and-how-to-fix-it-5.png)

The frontier framing tells you *which* lever to reach for given *your* constraint. If you are latency-bound (real-time UI), distillation is mandatory — nothing else gets you to sub-second. If you are quality-critical (final renders), stay at 20–30 steps with a good sampler and instead pull the free systems levers. If you are memory-bound (consumer GPU, big model), quantization and offload come first. If you are throughput-bound (serving millions), batching plus a distilled few-step model maximizes images-per-GPU-hour. The Pareto frontier is not a theoretical nicety; it is the concrete decision tool for picking levers, and we will return to it in every post of this track.

## 10. Case studies: real numbers from the frontier

Let us ground the levers in named, published results. These are the points the rest of the track derives in detail; here we just plot them on the frontier so you see the magnitudes.

**LCM and LCM-LoRA (Luo et al., 2023).** Latent Consistency Models distill a latent-diffusion teacher into a student satisfying the consistency property, enabling **4-step** (and even 1–2 step) sampling. LCM-LoRA packages the distillation as a LoRA you can attach to many SD/SDXL checkpoints, turning a 50-step model into a 4-step one *without retraining the base*. The reported result: SDXL-class quality at ~4 steps, roughly an order of magnitude fewer forwards than the 30–50 the base needs. This is lever 1 as a plug-in. Detailed in [the consistency-models post](/blog/machine-learning/image-generation/consistency-models-and-few-step-generation).

**SDXL-Turbo / ADD (Sauer et al., 2023).** Adversarial Diffusion Distillation adds a GAN-style discriminator loss to the distillation objective, pushing SDXL to **1–4 step** generation with surprisingly strong quality. SDXL-Turbo can produce a 512² image in a *single* forward pass — sub-200-ms on a good GPU — making it the first widely-used real-time text-to-image model. The trade-off is some diversity loss and a quality ceiling below the full 50-step model, but for interactive use it is transformative. Covered in [the distribution-matching post](/blog/machine-learning/image-generation/distribution-matching-and-adversarial-distillation).

**DMD / DMD2 (Yin et al., 2023–2024).** Distribution Matching Distillation trains a one-step generator by matching the *distribution* of the teacher's outputs (via a score-distillation gradient), and DMD2 removes the regression loss and adds a GAN loss for stability. The headline: **one-step** text-to-image with FID competitive with the multi-step teacher — DMD2 reported one-step SDXL FID in the same ballpark as the multi-step base on COCO-class benchmarks. One forward pass, full-resolution. This is lever 1 taken to its limit.

**SANA (Xie et al., 2024).** SANA combines lever 2 and lever 3: a **32× deep-compression autoencoder** (16× fewer tokens than the usual 8× VAE) plus **linear attention** in the DiT (replacing the quadratic softmax attention). The result is a model that generates **1024² images dramatically faster** than comparable DiT models — the paper reports large throughput gains and the ability to run high-resolution generation on a laptop-class GPU. SANA is the clearest single example of attacking $c_\text{net}$ from both the token-count and the attention-complexity side at once. Its architecture is covered in [the modern frontier-recipe post](/blog/machine-learning/image-generation/mmdit-and-the-modern-text-to-image-recipe).

**DeepCache / TeaCache (2023–2024).** DeepCache observes that the U-Net's high-level features change slowly across adjacent denoising steps, so it caches and reuses them, recomputing only the cheap high-resolution parts on most steps — reported ~2× speedups with negligible quality change on SD-class models. TeaCache extends the idea to DiT-based models (timestep-aware caching). Pure lever 4: no retraining, no quality regression worth mentioning, stacks on top of everything. Covered in [the caching post](/blog/machine-learning/image-generation/quantization-caching-and-efficient-inference).

These are representative published figures; exact numbers depend on benchmark, resolution, and hardware, and I have framed magnitudes as approximate where the precise value depends on the setup. The pattern across all of them is the same: each pushes the speed↔quality frontier by attacking one term of the cost model, and the frontier models combine several.

### 10.1 A worked optimization, end to end (and where it breaks)

Let me walk one concrete engineering problem from start to finish, because the levers only mean something when you sequence them against a real constraint — and because the instructive part is where each step *stops* paying off.

**The problem.** You are building an interactive design tool. The product requirement is a *live preview*: as the user edits the prompt, the image should refresh in under 300 ms on an A100 so it feels responsive. Your starting point is FLUX.1-dev at 1024², 50 steps, which on an A100 takes roughly 10–12 seconds per image (FLUX's transformer is ~12B parameters, so $c_\text{net}$ is large). You are about 40× too slow. Where do you start?

**Step 1 — profile.** Run the CUDA-event sweep from section 5. You confirm the denoiser loop is ~95% of the time and that one forward is ~200 ms. The cost model says: attack $N$ first, then $c_\text{net}$.

**Step 2 — turn on the free systems levers.** `torch.compile` the transformer (one-time compile, then ~1.4× faster forwards), confirm SDPA is active, and quantize to fp8 (FLUX tolerates fp8 well). The forward drops from ~200 ms to ~110 ms. You are now at ~5.5 s — still 18× too slow, but you spent zero quality and an hour of work.

**Step 3 — pull the biggest lever: fewer steps.** Apply a few-step distilled variant (a FLUX-schnell-style or LCM-LoRA-style 4-step model). The loop goes from 50 × 110 ms to 4 × 110 ms ≈ 440 ms. Total ~600 ms. You are within 2× of target now, having gained ~9× from this single lever.

**Step 4 — re-profile (the bottleneck moved).** At 4 steps, the loop is ~440 ms but the VAE decode (~90 ms at 1024² for the FLUX VAE) and text encode (T5 is heavy, ~60 ms) are now ~25% of the budget. Swap to a tiny/taesd-style VAE (decode ~15 ms) and cache the T5 embedding across edits where the prompt prefix is unchanged. Total ~480 ms.

**Step 5 — squeeze $c_\text{net}$ further.** You are close. A linear-attention or pruned denoiser would help, but you cannot retrain FLUX. Instead, drop to 512² for the *live preview* and render 1024² only on release of the slider. At 512² the token count drops 4× and the forward is ~45 ms. Loop ~180 ms, total ~270 ms. **Target met.**

**Where it breaks — the stress tests.** Now push each decision until it fails, because that is how you learn the real boundaries.

- *What happens at 1 step?* The 4-step distilled model run at 1 step produces noticeably worse structure — counting fails, fine textures smear. The 4→1 step jump is the steepest part of the remaining quality cliff; only a model *distilled specifically for 1 step* (DMD2-style) holds up, and even then with some diversity loss. Lesson: do not assume a 4-step model degrades gracefully to 1 step.
- *What happens when CFG is high?* If you crank `guidance_scale` to 12 to "sharpen" the preview, distilled few-step models often over-saturate and posterize badly — they were distilled at a specific guidance setting, and pushing past it breaks the distillation assumption. Lesson: distilled models have a narrow guidance sweet spot; respect it.
- *What happens when the VAE is the bottleneck?* You swapped to a tiny VAE for speed, but on the *final* 1024² render the tiny VAE's reconstruction is visibly softer than the full VAE. Lesson: use the fast VAE for the preview, the full VAE for the deliverable — the levers can differ by stage.
- *What happens at batch 8 (a sudden traffic spike)?* Your single-image-tuned pipeline is memory-bound at batch 1; at batch 8 you tip compute-bound and per-image latency rises, but throughput climbs. If your autoscaler does not account for this, p95 latency spikes under load. Lesson: the regime (memory- vs compute-bound) shifts with batch, so size your batch to your latency SLO, not just your throughput goal.
- *What happens when you fine-tune the distilled model on 5 brand images?* Few-step distilled models are *fragile* to fine-tuning — a small LoRA on top can break the consistency property and reintroduce the need for more steps. Lesson: fine-tune the base model and re-distill, or use a distillation method (LCM-LoRA) designed to stack with content LoRAs.

That narrative is the whole track in miniature: profile, pull the biggest lever, re-profile because the bottleneck moves, and know the failure mode of each decision so you stop before quality cracks.

## 11. The economics: latency, throughput, and cost per image

The reason this track matters in dollars is that latency and throughput translate directly into the inference bill. Two numbers govern serving cost: **latency** (how long one image takes, which sets the user experience) and **throughput** (images per GPU-hour, which sets the cost). They are related but not identical — batching can raise throughput without lowering single-image latency, and distillation lowers both.

Let us do the arithmetic that a serving engineer actually cares about. Suppose an A100 80GB rents for about \$2.00 per hour (representative on-demand cloud pricing; spot and committed-use are cheaper). One GPU-hour is 3,600 seconds.

#### Worked example: cost per image, 50-step vs 4-step

- **50-step SDXL, batch 1, ~3.9 s/image.** Throughput ≈ $3600 / 3.9 \approx 923$ images/GPU-hour. At \$2/hr, that is about **\$0.0022 per image** — roughly a fifth of a cent. Sounds cheap, until you serve ten million a day: that is ~\$22,000/day in raw GPU cost, and your p95 latency is ~4 seconds, which feels sluggish in a UI.
- **4-step distilled SDXL (Turbo/LCM), batch 1, ~0.45 s/image.** Throughput ≈ $3600 / 0.45 \approx 8000$ images/GPU-hour — about **8.7×** more. Cost drops to about **\$0.00025 per image**, and the same ten million a day now costs ~\$2,500/day. The latency is sub-half-second, which feels instant.

That single lever — distillation — cut the inference bill by ~8.7× *and* turned a sluggish experience into a snappy one. Now layer on **batching for throughput**: at batch 8 on an A100, you do not get 8× because you become compute-bound and the per-image time stops falling linearly, but you might get ~4–5× more images/GPU-hour at the cost of higher per-batch latency. For an asynchronous workload (generate-and-store, not interactive), you batch aggressively and chase throughput; for an interactive workload, you keep batch small and chase latency. The cost model and the four levers are what let you make that trade deliberately instead of by trial and error.

The general lesson: **a step is not just milliseconds, it is money and user experience at once.** A 10× latency win from distillation is simultaneously a ~10× cost win and the difference between a spinner and a live preview. That is why, of all the levers, the step count is the one the whole industry raced to cut first — and why this track opens with it.

One more economic wrinkle worth flagging: the *memory* budget feeds back into the *cost* budget through batch size. A model that fits twice as comfortably in VRAM (because you quantized it) lets you run a larger batch, which raises throughput, which lowers cost per image — so quantization pays you twice, once in faster memory-bound forwards and once in headroom for bigger batches. This coupling between the latency budget and the memory budget is why the strongest serving setups quantize *and* distill *and* batch: each lever unlocks more of the others. The four algorithmic levers plus the systems tier are not a menu to pick one from; for a production stack you pull most of them, in the order the cost model dictates, re-profiling at each step. The rest of this Speed track is the detailed how-to for each lever — and the capstone, [building an image generation stack](/blog/machine-learning/image-generation/building-an-image-generation-stack), assembles them into one real pipeline.

## 12. When to reach for each lever (and when not to)

A decisive guide, because every lever is a cost and the wrong one wastes your time.

**Reach for fewer steps (distillation) when** you are latency- or throughput-bound and need sub-second generation or maximum images-per-GPU-hour. It is the only lever that gives an order of magnitude. **Do not** reach for it first if you need the absolute highest quality and diversity for final renders — distilled models trade a little of both, and at 30 steps with a good sampler a non-distilled model is still the quality king. Also do not distill if you cannot afford the training compute and a stock LCM-LoRA does not cover your base model — though increasingly one does.

**Reach for a faster denoiser / deeper AE when** you are building or fine-tuning a model from a strong base and can absorb an architecture change. This is a model-design decision (SANA-style), not a post-hoc tweak — you generally cannot bolt a 32× AE onto an existing SDXL without retraining the denoiser to live in the new latent space. **Do not** expect to swap the autoencoder under a pretrained model for free; the denoiser was trained on the old latent distribution.

**Reach for feature caching when** you want a near-free 1.5–2.5× on an existing model with no retraining and you can tolerate a tiny, usually-imperceptible quality change. It is the easiest algorithmic win. **Do not** rely on it for huge speedups — it is a multiplier on top of the big levers, not a replacement for them, and at very low step counts (where there are few adjacent steps to cache across) its benefit shrinks.

**Reach for the systems levers (quantization, SDPA, compile, batching) always, first.** They are nearly free and they stack. **Do not** skip them and jump straight to distillation — you would be optimizing the algorithm while leaving an easy 2–4× of constant-factor speedup on the table. And do not over-quantize to 4-bit without measuring quality; int8 is usually safe, 4-bit needs care (SVDQuant-style methods exist precisely because naive 4-bit hurts).

**Do not** chase parallelizing the step loop — it is structurally impossible (section 2). **Do not** profile once and stop — the bottleneck moves as you optimize (section 6). **Do not** add steps past ~20–30 with a good sampler expecting quality — it saturates (section 4).

The meta-rule: **measure, pull the biggest lever for your constraint, re-measure, repeat.** Latency optimization is iterative because the cost model's dominant term changes as you shrink it.

## 13. Key takeaways

- **Latency $= c_\text{text} + N \cdot c_\text{net} + c_\text{vae}$.** At 50 steps the denoiser loop is ~96% of the cost; the step count $N$ is the dial with the most leverage.
- **You cannot parallelize the steps.** Sampling is a sequential dependency chain — each step's input is the previous step's output. You make it shorter, cheaper, or reuse work; you cannot make it wider.
- **Four algorithmic levers**: fewer steps (distillation), a faster denoiser, deeper latent compression, and feature caching. They act on different terms and **multiply**.
- **Fewer steps is the highest-ROI lever** because quality saturates with step count — FID is flat from 20 to 50 steps, so those steps buy latency, not quality. Distillation moves the whole quality-vs-steps curve left.
- **The bottleneck moves as you optimize.** After distilling to 4 steps, the VAE decode and text encode become a real fraction of latency — re-profile and attack whatever became dominant.
- **Latent compression attacks the quadratic attention term.** A 32× AE gives 16× fewer tokens, up to ~256× less attention compute — bounded only by the autoencoder's reconstruction quality.
- **Turn on the systems levers first** (SDPA/FlashAttention, `torch.compile`, int8 quantization, batching, offload-if-VRAM-bound) — they are nearly free and stack with the big levers for the constant-factor wins.
- **Profile with CUDA events, not wall-clock**, with warm-up and a fixed seed; sweep `num_inference_steps` to see the cost model's linear-in-$N$ shape on your own GPU.
- **Everything lives on the speed↔quality Pareto frontier.** Pick the lever that pushes the frontier in the direction your constraint (latency / quality / memory / throughput) demands.

## 14. Further reading

- Ho, Jain & Abbeel, **"Denoising Diffusion Probabilistic Models"** (2020) — the iterative sampling process whose step count this whole track attacks.
- Song, Meng & Ermon, **"Denoising Diffusion Implicit Models"** (2021) — DDIM and the ODE view that explains why steps have diminishing returns. See also [the DDIM and fast deterministic sampling post](/blog/machine-learning/image-generation/ddim-and-fast-deterministic-sampling).
- Song, Dhariwal, Chen & Sutskever, **"Consistency Models"** (2023) — the foundation of single-and-few-step generation (lever 1).
- Sauer, Lorenz, Blattmann & Rombach, **"Adversarial Diffusion Distillation"** (2023) — SDXL-Turbo, real-time 1–4 step text-to-image.
- Yin et al., **"One-step Diffusion with Distribution Matching Distillation"** (DMD, 2023; DMD2, 2024) — one-step generation by matching the teacher's distribution.
- Xie et al., **"SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers"** (2024) — the 32× deep-compression AE + linear attention (levers 2 and 3).
- Ma et al., **"DeepCache: Accelerating Diffusion Models for Free"** (2023) — feature reuse across steps (lever 4).
- 🤗 `diffusers` documentation, **"Optimization"** and **"Speed up inference"** guides — the practical `enable_*`, SDPA, `torch.compile`, and scheduler-swap APIs used in this post.
- Within this series: [diffusion from first principles](/blog/machine-learning/image-generation/diffusion-from-first-principles) (why sampling iterates), [the samplers deep dive](/blog/machine-learning/image-generation/samplers-deep-dive) (the step-count Pareto in detail), and the capstone [building an image generation stack](/blog/machine-learning/image-generation/building-an-image-generation-stack) (where all the levers come together in a real pipeline).
