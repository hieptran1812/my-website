---
title: "GEMM for decode: the skinny-matrix problem"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "Your projection layers are matmuls, and at batch 1 they are the wrong shape for a GPU: a matrix-vector product that reads a whole weight matrix to do one multiply-add per element. Derive why the tensor cores sit idle, compute the critical batch size where they wake up, and write a Triton decode GEMV that gets honest about bandwidth."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "gpu",
    "cuda",
    "triton",
    "gemm",
    "batching",
    "quantization",
    "throughput",
    "latency",
    "ml-systems",
    "pytorch",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 46
---

Here is a number that should stop you. When Llama-3.1-8B generates one token, the down-projection inside a single MLP layer does about 117 million floating-point operations. To do them, it reads about 117 million *bytes* from HBM — the entire down-projection weight matrix, all 112 mebibytes of it, streamed across the memory bus so that each weight can participate in exactly one multiply-add and then be discarded. That is one FLOP per byte moved. The RTX 4090 those weights are sitting on can do roughly 165 trillion floating-point operations per second on its tensor cores and can read roughly one trillion bytes per second from memory. Feed it a ratio of one FLOP per byte and it spends all its time waiting on memory, and the tensor cores — the expensive silicon you actually paid for — run at about half a percent of their rated throughput.

This is not a bug in your code or a missing flag. It is the shape of the problem. The projection and MLP layers of a transformer are matrix multiplies, and at decode time, with batch size 1, every one of them is the wrong shape for a GPU. A matrix multiply `[M, K] x [K, N]` is a beautiful, tensor-core-saturating operation when M is large. At decode, M is the number of tokens you are computing this step, and at batch 1 that is exactly one. The matmul degenerates into a matrix-*vector* product — a GEMV — and a GEMV reads the whole weight matrix to produce one output row. The arithmetic per byte collapses to nearly nothing, and no kernel, however clever, can raise it, because the ratio is a property of the operation, not the implementation.

![Side-by-side comparison of a fat prefill GEMM that saturates the tensor cores against a skinny decode GEMV that leaves them idle](/imgs/blogs/gemm-for-decode-the-skinny-matrix-problem-1.webp)

The figure above is the whole post in one picture. The same weight matrix, run two ways: as a fat GEMM during prefill (many prompt tokens at once, tensor cores lit, compute-bound) and as a skinny GEMV during decode (one token, tensor cores idle, bandwidth-bound). The difference between them is not the weights and not the kernel. It is the value of M. This post is about what that single integer does to your hardware, and what you can — and cannot — do about it.

By the end you will be able to derive, for any linear layer in any model, its arithmetic intensity at any batch size; compute the exact batch size at which a decode GEMV stops being memory-bound and becomes a proper compute-bound GEMM (the "critical batch size," one of the most useful numbers in this whole series); explain why tokens-per-second is a misleading yardstick for a decode kernel and what to report instead; and write a real Triton matrix-vector kernel with split-K, autotuned, tested against `torch.mm`, that measures its own achieved bandwidth. This is `nanoserve/kernels/gemv.py`, and it is the last piece of the kernel layer before we go back up to precision and compression. If you have not read [what inference engineering is](/blog/machine-learning/inference-engineering/what-inference-engineering-is), that post frames the scoreboard — TTFT, TPOT, tokens per second, memory, goodput — that everything here moves. And if you want to see this decode step in the full company of the other kernels that run alongside it, [the inference kernel landscape](/blog/machine-learning/inference-engineering/the-inference-kernel-landscape-what-actually-runs) is the map.

Two promises up front, restated from the series introduction. First: **I have no GPU and have run none of this.** Every number below is derived from arithmetic I show you, cited from a public source with a link, or framed as something you should reproduce yourself with a named script on named hardware. The results tables carry a `Source` column. Second: this is a derivation post before it is a kernel post. The kernel is fun, but the number that matters is the one you can compute in your head about your own model, and I would rather you leave able to re-derive the critical batch size than able to paste my Triton.

---

## 1. Two shapes of the same matmul

Let me fix vocabulary, because the entire argument rides on three letters. A general matrix multiply — GEMM, the "GE" is "general" — computes `Y = X @ W` where `X` is `[M, K]`, `W` is `[K, N]`, and `Y` is `[M, N]`. In a transformer's linear layers, `W` is a weight matrix, fixed at load time; `X` is the stack of hidden vectors you are pushing through the layer this step; `K` is the input feature dimension and `N` the output feature dimension. The interesting letter is **M**: the number of rows of `X`, which is the number of tokens whose hidden states you are transforming right now.

During **prefill** — the pass that ingests the prompt — M is the prompt length. A 512-token prompt gives you a `[512, K]` activation matrix, and the projection is a genuine GEMM with a fat M. During **decode** — the autoregressive loop that emits one token at a time — you have already computed and cached everything about the prompt, so each step transforms exactly the hidden states of the *new* tokens. At batch size 1, that is one token: M equals 1. The activation is a single row vector `[1, K]`, and `Y = x @ W` is a matrix multiplied by a vector. That special case has its own name, GEMV, the "V" for "vector," and it behaves nothing like its fat cousin even though the weight matrix `W` is byte-for-byte identical.

Why does M matter so much? Because M is the only dimension over which the weight matrix gets *reused*. Look at what happens element by element. Every weight `W[k, n]` must be read from memory once. In a GEMM it is multiplied against `X[m, k]` for all M rows — so one weight read feeds M multiply-adds. In a GEMV, M is 1, so one weight read feeds exactly one multiply-add, and then that weight is never touched again for the rest of the step. The weight matrix is the overwhelmingly dominant thing you move across the bus — for an 8B model it is gigabytes, while the activation vector is kilobytes — and at batch 1 you pay the full cost of reading it to extract a single token's worth of work from it.

That is the reframe, and it reorganizes everything you thought you knew about inference performance. Prefill is compute-bound: you have plenty of arithmetic to hide the memory traffic behind, and the tensor cores are the bottleneck. Decode at batch 1 is memory-bound: there is almost no arithmetic, the tensor cores are idle, and the HBM bus is the bottleneck. These are two different machines wearing the same GPU. The kernels that win at one lose at the other, the metrics that describe one mislead about the other, and the optimizations that help one do nothing for the other. Hold that split in your head; the rest of the post is consequences of it.

### Where the bytes actually go

Before the formula, let me make the traffic concrete, because "you read the weights" is abstract until you count them. Consider one decode token flowing through one Llama-3.1-8B layer. Its `config.json` gives the shapes: hidden size 4096, intermediate size 14336, 32 layers, 32 attention heads, 8 key-value heads, head dimension 128. A single token is a length-4096 vector, and inside each layer it hits four weight matrices.

![A single decode token forking into four projection reads that merge into a per-layer byte total and then the full sixteen-gigabyte weight sweep](/imgs/blogs/gemm-for-decode-the-skinny-matrix-problem-2.webp)

Picture that one vector fanning out, as in the figure, into the four projections and the bytes each one costs, all in bf16 (2 bytes per weight):

- **QKV projection.** Query is `[4096, 4096]`; with grouped-query attention the 8 KV heads make key and value each `[4096, 1024]`. Total parameters `4096*4096 + 2*(1024*4096)` = about 25.2 million, so about **50 MB**.
- **Output projection.** `[4096, 4096]`, about 16.8 million parameters, about **34 MB**.
- **Gate and up projections (SwiGLU).** Two matrices, each `[4096, 14336]`, about 58.7 million parameters apiece, about **235 MB** together.
- **Down projection.** `[14336, 4096]`, about 58.7 million parameters, about **117 MB**.

Add them: about **436 MB of weights read per layer, per token.** Multiply by 32 layers and add the roughly 1 GB language-model head at the end, and one decode token drags on the order of **16 GB** across HBM — which is, not coincidentally, the entire bf16 weight footprint of the model. This is the single most important sentence about batch-1 decode: *you read the whole model, once, per token.* We derived exactly this in [the memory math of the KV cache](/blog/machine-learning/inference-engineering/the-memory-math-of-the-kv-cache) from the other direction; here it is the numerator of everything.

---

## 2. Arithmetic intensity, derived

Now the formula that makes it rigorous. **Arithmetic intensity** (AI) is the ratio of useful floating-point work to bytes moved from memory: FLOPs per byte. It is the x-axis of the roofline model, and it is the number that decides whether an operation is compute-bound or memory-bound. Let me derive it for a general GEMM.

A GEMM `[M, K] x [K, N]` computes `M*N` output elements, each a dot product of length `K`, and each dot product is `K` multiply-adds — `2K` FLOPs counting the multiply and the add separately. So total work is:

$$\text{FLOPs} = 2 \cdot M \cdot N \cdot K$$

The bytes moved, assuming everything is read from and written to HBM once at `b` bytes per element, are the input `X` (`M*K` elements), the weights `W` (`K*N` elements), and the output `Y` (`M*N` elements):

$$\text{bytes} = b \cdot (M K + K N + M N)$$

Divide, and the `b` in front of the FLOPs (which is `2`, for two operations per element pair) and the `b` in the denominator do not cancel cleanly in general, so keep them explicit. With `b = 2` for bf16 the arithmetic intensity is:

$$\text{AI} = \frac{2 M N K}{2\,(M K + K N + M N)} = \frac{M N K}{M K + K N + M N}$$

Stare at the denominator. When M is small, the term `K N` — the weight matrix — dominates it, because `K N` is enormous (millions of elements) while `M K` and `M N` are small (M is tiny). So the whole expression is governed by the weights. Let me take the two limits that matter.

**Decode, M = 1.** Substitute:

$$\text{AI}_{M=1} = \frac{N K}{K + K N + N} \approx \frac{N K}{K N} = 1$$

The `K` and `N` terms in the denominator are negligible next to `K N`, so the arithmetic intensity of a batch-1 GEMV is **almost exactly 1 FLOP per byte**, independent of how large the matrix is. Read a weight, do one multiply-add, throw it away. This is the arithmetic reason the tensor cores are idle: they can consume hundreds of FLOPs per byte, and you are handing them one.

**Batched decode, weight-dominated.** As long as M stays modest so that `M K` and `M N` remain small next to `K N`, the denominator is approximately `2 K N` (weights read plus a same-shaped write is not the point — the weights are the read that dominates), and:

$$\text{AI} \approx \frac{2 M N K}{2 K N} = M$$

This is the cleanest and most useful result in the post: **in the weight-dominated regime, arithmetic intensity equals the batch size.** Batch 1 gives AI near 1; batch 32 gives AI near 32; batch 128 gives AI near 128. Every token you add to the step is one more multiply-add extracted from each weight read, so intensity climbs linearly with M until the small-M approximation breaks. Remember this: **AI is M.** It turns the abstract roofline into a concrete lever you control from a config file.

![A comparison grid of the down projection at batch one versus batch five hundred twelve showing FLOPs bytes intensity and percent of peak compute](/imgs/blogs/gemm-for-decode-the-skinny-matrix-problem-3.webp)

#### Worked example: the Llama-3.1-8B down projection

Let me put real numbers on the down projection, the matrix from the opening paragraph. It is `[K, N] = [14336, 4096]`.

At **M = 1** (decode, batch 1):

- FLOPs: `2 * 1 * 4096 * 14336` = 117,440,512, about **117 MFLOP**.
- Weight bytes (bf16): `2 * 14336 * 4096` = 117,440,512, about **117 MB**. The input vector is `2 * 14336` = about 29 KB and the output `2 * 4096` = about 8 KB — both rounding error.
- Arithmetic intensity: `117,440,512 / 117,477,376` = **0.9997**, i.e. essentially **1.0 FLOP/byte**.

At **M = 512** (prefill-scale):

- FLOPs: `2 * 512 * 4096 * 14336` = about **60.1 GFLOP**.
- Bytes: weights 117 MB, input `2 * 512 * 14336` = about 14.7 MB, output `2 * 512 * 4096` = about 4.2 MB, total about **136 MB**.
- Arithmetic intensity: `60.1e9 / 136e6` = about **441 FLOP/byte**.

Same matrix, same kernel, and the arithmetic intensity moved by a factor of over four hundred. The comparison figure above lays the two side by side. Now translate intensity into achieved throughput on an RTX 4090, whose relevant specs — I am citing NVIDIA's *Ada GPU Architecture* whitepaper — are 1,008 GB/s of GDDR6X bandwidth and 165.2 dense FP16 tensor TFLOPS with FP32 accumulation (the mode real inference uses for numerical stability; the FP16-accumulate path is quoted at double that but is rarely used for LLMs).

For the M = 1 case, the operation is bandwidth-bound. A good kernel moves the 117 MB at, say, 85% of the 1,008 GB/s spec — about 857 GB/s — so it takes `117.4e6 / 857e9` = about **137 microseconds**. In that time it did 117.4 MFLOP, which is `117.4e6 / 137e-6` = **0.86 TFLOP/s**. Against the 165.2 TFLOPS tensor-core peak, that is **0.52% of peak compute.** For the M = 512 case, the operation is compute-bound: even at a realistic 60% of the 165.2 TFLOPS peak the tensor cores do about 99 TFLOP/s, and the GEMM finishes in about 600 microseconds while genuinely using the hardware it runs on.

Half a percent versus sixty percent. That is the skinny-matrix problem, quantified, and it is why the decode loop is the hardest part of inference to make fast: the operation is structurally starved of the arithmetic that GPUs are built to devour.

| Shape | M | FLOPs | Weight bytes | AI (FLOP/byte) | Bound by | % of peak compute | Source |
|---|---|---|---|---|---|---|---|
| Down proj, decode | 1 | 117 MFLOP | 117 MB | ~1.0 | bandwidth | ~0.5% | derived |
| Down proj, prefill | 512 | 60.1 GFLOP | 117 MB | ~441 | compute | ~60% | derived |
| Whole model / token | 1 | ~16.1 GFLOP | ~16 GB | ~1.0 | bandwidth | ~0.5% | derived |
| Whole model / token | 64 | ~1.03 TFLOP | ~16 GB | ~64 | mixed | rising | derived |

---

## 3. The roofline and the critical batch size

The roofline model draws two ceilings on a plot of achievable FLOP/s against arithmetic intensity. On the left, a diagonal: when you are memory-bound, achievable throughput is `AI * bandwidth`, so it rises with intensity. On the right, a flat line: the compute peak, which you cannot exceed no matter how much arithmetic you pile on. The two meet at the **ridge point**, the arithmetic intensity at which a perfectly efficient kernel would just saturate the compute peak using all of the bandwidth. Below the ridge you are memory-bound; above it, compute-bound. If the roofline is new to you, [the roofline model post](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) builds it from scratch; here I only need its punchline.

The ridge-point intensity is peak compute divided by peak bandwidth:

$$\text{AI}^{*} = \frac{\pi_{\text{peak}}}{\beta_{\text{peak}}}$$

where `π_peak` is FLOP/s and `β_peak` is bytes/s. Now combine this with the result from the last section — that in the weight-dominated regime `AI ≈ M`. Setting the operating intensity equal to the ridge gives the batch size at which decode stops being memory-bound and becomes a real GEMM:

$$M^{*} = \text{AI}^{*} = \frac{\pi_{\text{peak}}}{\beta_{\text{peak}}}$$

This is the **critical batch size**. Below it, adding tokens to the step is free throughput — you are on the diagonal, and the weight read you were paying for anyway now serves more tokens. Above it, adding tokens no longer helps per-token latency because you have run out of tensor cores; you are on the flat ceiling. `M*` is the hinge of the entire economics of serving, and it is a property of the *hardware*, computable from two datasheet numbers.

<figure class="blog-anim">
<svg viewBox="0 0 640 320" role="img" aria-label="A roofline chart whose operating point climbs the bandwidth-bound diagonal as batch grows, then flattens onto the compute ceiling past the ridge point" style="width:100%;height:auto;max-width:760px">
<style>
.rl-axis{stroke:var(--border,#d1d5db);stroke-width:1.5}
.rl-roof{fill:none;stroke:var(--text-secondary,#6b7280);stroke-width:2.5}
.rl-ridge{stroke:var(--border,#d1d5db);stroke-width:1;stroke-dasharray:4 4}
.rl-lbl{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.rl-sub{font:500 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.rl-dot{fill:var(--accent,#6366f1)}
.rl-halo{fill:var(--accent,#6366f1);opacity:.18}
@keyframes rl-climb{0%{transform:translate(0px,0px)}8%{transform:translate(0px,0px)}50%{transform:translate(280px,-170px)}58%{transform:translate(280px,-170px)}92%{transform:translate(530px,-170px)}100%{transform:translate(530px,-170px)}}
.rl-mv{animation:rl-climb 9s ease-in-out infinite alternate}
@media (prefers-reduced-motion:reduce){.rl-mv{animation:none;transform:translate(280px,-170px)}}
</style>
<line class="rl-axis" x1="70" y1="40" x2="70" y2="250"/>
<line class="rl-axis" x1="70" y1="250" x2="610" y2="250"/>
<polyline class="rl-roof" points="70,250 350,80 610,80"/>
<line class="rl-ridge" x1="350" y1="80" x2="350" y2="250"/>
<text class="rl-sub" x="60" y="36" text-anchor="end">FLOP/s</text>
<text class="rl-sub" x="600" y="270" text-anchor="end">arithmetic intensity ~ batch M</text>
<text class="rl-lbl" x="150" y="185">bandwidth-bound</text>
<text class="rl-lbl" x="420" y="70">compute ceiling</text>
<text class="rl-sub" x="350" y="245" text-anchor="middle">ridge M* ~ 164</text>
<circle class="rl-halo rl-mv" cx="70" cy="250" r="14"/>
<circle class="rl-dot rl-mv" cx="70" cy="250" r="7"/>
<text class="rl-sub" x="78" y="248">M=1</text>
</svg>
<figcaption>The operating point starts bandwidth-bound at batch 1, climbs the diagonal as batch raises arithmetic intensity, then flattens onto the compute ceiling once it passes the ridge point near M=164.</figcaption>
</figure>

The animation traces exactly this. The operating point begins at batch 1, deep in the memory-bound region at AI near 1, then climbs the diagonal as you add tokens (because AI is M), and flattens onto the compute ceiling the instant it passes the ridge. That climb is what continuous batching buys you, and the flattening is where it stops buying.

#### Worked example: critical batch on a 4090 versus an A100

Plug in real datasheet numbers. All specs cited; the division is mine.

- **RTX 4090.** Peak 165.2 TFLOPS (Ada whitepaper, FP16 tensor / FP32 accumulate), bandwidth 1,008 GB/s. `M* = 165.2e12 / 1.008e12` = **164**.
- **A100 80GB SXM.** NVIDIA's A100 datasheet lists 312 BF16 tensor TFLOPS (624 with sparsity) and 2,039 GB/s of HBM2e. `M* = 312e12 / 2.039e12` = **153**.
- **H100 80GB SXM.** The H100 datasheet lists 3.35 TB/s of HBM3 and 1,979 BF16 tensor TFLOPS with sparsity, so about 990 dense. `M* = 989.4e12 / 3.35e12` = **295**.
- **L4.** NVIDIA's L4 datasheet lists about 121 FP16 tensor TFLOPS and 300 GB/s. `M* = 121e12 / 0.30e12` = **403**.

Read that table twice, because the ordering is the opposite of intuition. The consumer 4090 and the datacenter A100 have nearly identical critical batch sizes — about 160 and 150 — despite the A100 being far more expensive. And the *most* capable inference chips need the *largest* batches to become compute-bound: the H100's tensor cores grew faster than its bandwidth relative to the A100, so its ridge sits near 295, and the cost-optimized L4 has such thin bandwidth relative to its compute that you need over 400 concurrent tokens before its tensor cores wake up. The better the GPU, the harder you have to work to feed it — which is precisely why the scheduler and the batching loop, not the kernel, are where decode throughput is won.

| GPU | Peak bf16 TFLOPS | Bandwidth | Critical batch M* | Source |
|---|---|---|---|---|
| RTX 4090 | 165.2 (dense) | 1,008 GB/s | ~164 | derived; specs cited: Ada whitepaper |
| A100 80GB SXM | 312 (dense) | 2,039 GB/s | ~153 | derived; specs cited: A100 datasheet |
| H100 80GB SXM | ~990 (dense) | 3,350 GB/s | ~295 | derived; specs cited: H100 datasheet |
| L4 | ~121 | 300 GB/s | ~403 | derived; specs cited: L4 datasheet |

---

## 4. The honest metric: achieved bandwidth, not TFLOP/s

Now a methodological point that matters more than any kernel trick, because getting it wrong makes you optimize the wrong thing. If you benchmark a decode GEMV and report its TFLOP/s, you have reported a number that is *guaranteed* to look terrible and that you *cannot* improve. It will always be about 0.5% of peak, because the arithmetic intensity is fixed at 1 by the operation. Reporting it invites a manager or a well-meaning colleague to conclude that the kernel is broken and someone should "optimize the matmul," which is a doomed errand: there is no arithmetic to un-waste.

The right yardstick for a bandwidth-bound operation is **achieved bandwidth as a fraction of peak.** You moved `W` bytes of weights and it took `t` seconds; your achieved bandwidth is `W / t`, and the question is how close that is to the GPU's spec. A good decode kernel reaches **80–90% of peak HBM bandwidth** — the last 10–20% is lost to imperfect coalescing, tail effects on the final tile, and the small activation and output traffic. If your GEMV hits 88% of a 4090's 1,008 GB/s, it is an excellent kernel and there is essentially nothing left to win on that operation; the only levers left are the ones in section 8, which change how many bytes you move rather than how fast you move them.

So the honest report for a decode kernel is a bandwidth-utilization table, not a FLOP/s table:

| Kernel / shape | Bytes moved | Achieved BW | % of 4090 peak | Verdict | Source |
|---|---|---|---|---|---|
| Down proj GEMV, bf16 | 117 MB | ~880 GB/s | ~87% | healthy | reproduce: `bench_bandwidth` |
| Whole-model decode step, bf16 | ~16 GB | ~850 GB/s | ~84% | healthy | reproduce: `bench_bandwidth` |
| Naive strided GEMV | 117 MB | ~430 GB/s | ~43% | fix coalescing | reproduce: `bench_bandwidth` |
| cuBLAS big-tile at M=1 | 117 MB | ~600 GB/s | ~60% | wrong kernel | reproduce vs `torch.mv` |

I have labeled these `reproduce` rather than derived or cited, because they are what you should *expect to see* when you run the harness in section 6 on a 4090 — not measurements I made. The point of the table is the *shape* of the reasoning: a healthy decode kernel lives in the mid-to-high 80s as a percentage of bandwidth peak, and anything down in the 40s is a coalescing or kernel-selection bug, not a fundamental limit.

### How to measure it without lying to yourself

The measurement itself has traps, and they are the same ones from [the reproducible-benchmark discipline](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark). Three rules. **Warm up first** — the first launch of a Triton kernel triggers autotuning and JIT compilation that can take seconds and has nothing to do with steady-state speed; discard the first dozens of iterations. **Synchronize around the timed region with CUDA events, not Python clocks** — kernel launches are asynchronous, so `time.time()` measures how fast Python enqueued work, not how fast the GPU did it; you need `torch.cuda.Event` and a `torch.cuda.synchronize()`. **Time many iterations and divide** — a single decode GEMV is about a hundred microseconds, below the resolution of anything but CUDA events, and dominated by launch overhead unless you amortize it.

And the deepest trap: **tok/s at batch 1 tells you almost nothing about your server.** It measures the memory-bound diagonal at its far-left end. A server runs at whatever batch the scheduler assembles, which is somewhere up the diagonal or on the compute ceiling entirely. Reporting batch-1 tok/s as "the model's speed" is like reporting a truck's fuel economy while it idles in the driveway. The number you want is the throughput-versus-batch curve, and building it honestly is what the batch sweep in section 6 does.

---

## 5. Why cuBLAS is the wrong kernel for M = 1

You might assume the vendor library always picks the best kernel. For fat GEMMs it does; for M = 1 it often does not, and understanding why is the bridge to writing your own. A GEMM library like cuBLAS ships a portfolio of kernels, each built around a *tile*: a small `[BM, BN]` block of the output that one thread block computes, holding operands in shared memory and registers and feeding the tensor cores with dense `[BM, BK] x [BK, BN]` micro-multiplies. Those tiles are typically 64, 128, or 256 rows tall in the M dimension, because that is what saturates a tensor core's systolic array. The library's heuristic picks a tile and launches enough thread blocks to cover the `[M, N]` output.

At M = 1, every one of those M-tiles is a catastrophe of waste. A kernel built for BM = 128 loads and computes a `[128, BN]` output tile, but only the first row exists — the other 127 rows are padding, computed and discarded. The tensor-core micro-multiply runs at `1/128` useful occupancy. Worse, with M = 1 you have very few output tiles to distribute (only along N), so you may not even launch enough thread blocks to fill all the streaming multiprocessors, leaving whole chunks of the GPU idle while the few active blocks trudge through the long K reduction alone. In practice modern cuBLAS and cuBLASLt do detect small M and dispatch to dedicated GEMV or thin-M kernels — this is not a library bug — but the dispatch is heuristic, it does not always fire on unusual shapes, and it cannot do the fused tricks a hand kernel can. When you profile a decode step and see a GEMM kernel pulling 60% of bandwidth where a GEMV pulls 87%, this is what you are looking at.

![A decision tree selecting a GEMM kernel by the M and N dimensions with branches to GEMV small tile split-K and big-tile paths](/imgs/blogs/gemm-for-decode-the-skinny-matrix-problem-6.webp)

### Split-K: parallelizing the reduction

The tree above shows the kernel choice as a function of shape, and the interesting branch is the small-M, small-N corner, because that is where the GPU starves for parallelism. Here is the problem. A GEMV `[1, K] x [K, N]` has `N` independent output elements, and the natural parallelization gives each thread block a slab of the N dimension. If N is large — the gate and up projections have N = 14336 — you get plenty of blocks and the machine fills nicely. But if N is small relative to the GPU's block count, you launch too few blocks, each grinds serially through the entire length-K reduction (K = 14336 for the down projection), and most of the SMs sit idle.

**Split-K** fixes this by parallelizing the reduction itself. Instead of one block summing the whole K dimension for its output slab, you split K into, say, 8 chunks and give each chunk to a different block; each block computes a *partial* dot product over its slice, and then a second step sums the partials into the final output. You have traded a longer, serial reduction for more, shorter, parallel ones plus a combine. The combine is the cost: either a second kernel launch that reduces the partials, or atomic adds into the output from all the split blocks (which serialize on contention). Split-K is a win when you are parallelism-starved — small N, so few output tiles, so idle SMs — and a loss when you already have enough blocks, because then the combine is pure overhead. This is exactly the trade-off the vLLM team describes for their Triton attention decode kernel, which splits the KV traversal and runs a second reduction kernel to combine partials, gated by a heuristic because, in their words, no single configuration dominates ([vLLM Triton Attention Backend Deep Dive](https://vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive), 2026-03-04).

### The production path

For real serving you rarely hand-roll the base GEMV; you reach for a specialized library that has already solved the small-M problem and fused the surrounding work. On Blackwell, the vLLM team reports that the production stack leans on **FlashInfer's fast FP8 and FP4 GEMMs and MoE kernels**, together with fused all-reduce plus RMSNorm plus quantization, to deliver up to 4x higher throughput at similar latency versus Hopper on the SemiAnalysis InferenceMAX suite ([vLLM + NVIDIA Blackwell](https://vllm.ai/blog/2025-10-09-blackwell-inferencemax), 2025-10-09) — note this is a Pareto framing of throughput against per-token latency, not a single peak number, and I cite it as their result, not mine. And the fusion idea reaches its logical end in the Tencent Hunyuan HPC-Ops backend, whose fused FP8 MoE **Gate-Up GEMM reads its tokens directly through the routing index, skipping the gather that a naive MoE would do first**, and fuses the activation and FP8 quantization into the same kernel; they report the fused GEMM at 42.0 microseconds versus 56.4 for a Triton baseline and 74.5 for CUTLASS on an H20 with 192 experts and top-8 routing ([vLLM HPC-Ops](https://vllm.ai/blog/2026-07-06-vllm-hpc-ops), 2026-07-06). The lesson from both: for a decode GEMV, the wins are in *moving fewer bytes and skipping redundant passes*, never in finding more FLOPs.

---

## 6. Writing a decode GEMV in Triton

Enough theory. Let me write the kernel, because the act of writing it makes the bandwidth argument physical: you will see that every design choice is about how the weight bytes stream from memory, and none of it is about arithmetic. This is `nanoserve/kernels/gemv.py`. It builds on the Triton basics from earlier in this track — the `@triton.jit` decorator, `tl.load`/`tl.store`, program IDs, and autotuning.

Start with a helper that just prints the roofline verdict for a shape, so the numbers from section 2 are code you can run rather than arithmetic you have to trust:

```python
# nanoserve/kernels/roofline.py
def roofline(M, K, N, bytes_per_elem=2,
             peak_tflops=165.2, peak_bw_gbps=1008.0):
    """Predict whether Y = X[M,K] @ W[K,N] is compute- or memory-bound."""
    flops = 2 * M * N * K
    weight_bytes = bytes_per_elem * K * N
    io_bytes = bytes_per_elem * (M * K + M * N)
    bytes_moved = weight_bytes + io_bytes
    ai = flops / bytes_moved
    ridge = (peak_tflops * 1e12) / (peak_bw_gbps * 1e9)   # FLOP per byte
    regime = "compute-bound" if ai > ridge else "memory-bound"
    return dict(ai=ai, ridge=ridge, regime=regime,
                flops=flops, bytes=bytes_moved)

if __name__ == "__main__":
    for M in (1, 8, 64, 164, 512):
        r = roofline(M, K=14336, N=4096)
        print(f"M={M:4d}  AI={r['ai']:7.1f}  ridge={r['ridge']:.0f}  {r['regime']}")
```

Running this prints the transition happening right around M = 164 on a 4090, exactly where the ridge-point arithmetic said it would. That is the whole post as a five-line function.

Now the kernel. Each Triton program computes a `BLOCK_N`-wide slab of the output vector, streaming the corresponding columns of `W` and reducing over K in chunks of `BLOCK_K`:

```python
# nanoserve/kernels/gemv.py
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 64,  "BLOCK_K": 512}, num_warps=4),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 512}, num_warps=8),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 256}, num_warps=4),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 256}, num_warps=8),
    ],
    key=["N", "K"],
)
@triton.jit
def gemv_kernel(x_ptr, w_ptr, y_ptr, N, K,
                stride_wk, stride_wn,
                BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_n = tl.program_id(axis=0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)      # output columns
    n_mask = offs_n < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)          # fp32 accumulate
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        x = tl.load(x_ptr + offs_k, mask=k_mask, other=0.0)          # [BLOCK_K]
        w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
        w = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        acc += tl.sum(x[:, None].to(tl.float32) * w.to(tl.float32), axis=0)
    tl.store(y_ptr + offs_n, acc.to(y_ptr.dtype.element_ty), mask=n_mask)


def gemv(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """y[N] = x[K] @ w[K, N]."""
    K, N = w.shape
    assert x.shape == (K,), f"expected x of shape ({K},), got {tuple(x.shape)}"
    y = torch.empty(N, device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
    gemv_kernel[grid](x, w, y, N, K, w.stride(0), w.stride(1))
    return y
```

Two design notes that are the whole game. First, we accumulate in `float32` even though the inputs are bf16, because a length-14336 dot product summed in bf16 loses precision badly; this is a numerical requirement, not a speed one, and it is why the tensor-core FP32-accumulate peak is the right peak to compare against. Second, notice there is no `tl.dot` — no tensor-core instruction at all. At M = 1 there is nothing for a tensor core to do; this is a plain fused-multiply-add reduction that lives entirely on the load-store units and the memory system. The kernel is, by design, a bandwidth pump.

For the small-N, parallelism-starved case, here is the split-K variant. A second grid dimension slices K, each program accumulates its partial, and `tl.atomic_add` combines them into a float32 output:

```python
# nanoserve/kernels/gemv.py  (continued)
@triton.jit
def gemv_splitk_kernel(x_ptr, w_ptr, y_ptr, N, K, SPLIT_K,
                       stride_wk, stride_wn,
                       BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_n = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)                         # which K-slice
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    k_per_split = tl.cdiv(K, SPLIT_K)
    k_lo = pid_k * k_per_split
    k_hi = tl.minimum(k_lo + k_per_split, K)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(k_lo, k_hi, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < k_hi
        x = tl.load(x_ptr + offs_k, mask=k_mask, other=0.0)
        w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
        w = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        acc += tl.sum(x[:, None].to(tl.float32) * w.to(tl.float32), axis=0)
    tl.atomic_add(y_ptr + offs_n, acc, mask=n_mask)        # combine partials


def gemv_splitk(x, w, split_k=8, block_n=128, block_k=512):
    K, N = w.shape
    y = torch.zeros(N, device=x.device, dtype=torch.float32)   # atomics need fp32
    grid = (triton.cdiv(N, block_n), split_k)
    gemv_splitk_kernel[grid](x, w, y, N, K, split_k,
                             w.stride(0), w.stride(1),
                             BLOCK_N=block_n, BLOCK_K=block_k)
    return y.to(x.dtype)
```

Now the part that keeps you honest — a correctness test against a trusted reference. bf16 matmul carries real rounding error against an fp32 reference, so the tolerance is deliberately loose; the point is to catch a wrong kernel, not to demand bit-exactness (the determinism story is its own post, [sampling numerics and batch invariance](/blog/machine-learning/inference-engineering/sampling-numerics-determinism-and-batch-invariance)):

```python
# tests/test_gemv.py
import torch
from nanoserve.kernels.gemv import gemv, gemv_splitk

def test_gemv_matches_torch():
    torch.manual_seed(0)
    K, N = 14336, 4096                       # Llama-3.1-8B down proj
    x = torch.randn(K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    ref = (x.float() @ w.float())            # fp32 reference
    for out in (gemv(x, w).float(), gemv_splitk(x, w).float()):
        torch.testing.assert_close(out, ref, rtol=2e-2, atol=1e-1)
    print("max abs err:", (gemv(x, w).float() - ref).abs().max().item())
```

And the honest measurement harness. Weight bytes over time, with a warmup and CUDA events, reported as a fraction of the 4090 spec:

```python
# nanoserve/kernels/bench.py
import torch

def bench_bandwidth(fn, weight_bytes, peak_gbps=1008.0, iters=200, warmup=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters                 # ms per call
    gbps = weight_bytes / (ms * 1e-3) / 1e9
    return ms, gbps, gbps / peak_gbps

if __name__ == "__main__":
    K, N = 14336, 4096
    x = torch.randn(K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    wbytes = w.numel() * w.element_size()                # ~117 MB
    ms, gbps, frac = bench_bandwidth(lambda: gemv(x, w), wbytes)
    print(f"{ms*1e3:6.1f} us/call   {gbps:5.0f} GB/s   {frac*100:4.0f}% of peak")
```

Run this on a 4090 and you should land somewhere around 80–90% of the 1,008 GB/s spec for the well-tuned `gemv` — roughly 800–900 GB/s, roughly 130–150 microseconds per call — and rather less for a deliberately strided or mis-tiled variant. I am telling you the range you should *expect to reproduce*, on named hardware, from the script above; I have not run it. If yours comes in near 40% of peak, you have a coalescing problem in the `w_ptrs` access pattern, not a fundamental limit.

Finally, the batch sweep that draws the real throughput curve — this is the number that actually describes your server, walking M up the roofline diagonal:

```python
# nanoserve/kernels/sweep.py
import torch
from nanoserve.kernels.bench import bench_bandwidth

K, N = 14336, 4096
w = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
wbytes = w.numel() * w.element_size()
for M in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    ms, gbps, frac = bench_bandwidth(lambda: x @ w, wbytes)   # torch GEMM
    print(f"M={M:4d}  {ms*1e3:7.1f} us/step  "
          f"{M/(ms*1e-3):9.0f} tok/s  {frac*100:4.0f}% peak BW")
```

On a 4090 you should see the microseconds-per-step stay almost flat while M climbs from 1 toward the critical batch — because you are re-reading the same 117 MB regardless of M — so tokens-per-second rises almost linearly. Past M near 164 the step time starts climbing with M, tokens-per-second flattens, and the bandwidth fraction falls away from peak as the operation goes compute-bound. Watching that curve bend at the ridge is the most instructive thing you can do with a GPU and ten lines of Python.

---

## 7. Where batching lands you

The last section's sweep is the payoff, so let me derive the curve it draws and name where it stops helping. In the memory-bound regime — batch below the critical size — the weights are read once per step regardless of M, so step time is roughly constant at `weight_bytes / bandwidth`, and each step now produces M tokens. Throughput is therefore:

$$\text{tok/s} \approx \frac{M \cdot \beta_{\text{eff}}}{W_{\text{bytes}}}$$

which rises linearly in M. This is the entire economic argument for [continuous batching](/blog/machine-learning/inference-engineering/writing-a-continuous-batching-loop): the weight read is a fixed cost you pay every step, and every additional token you can pack into the step is amortized against it for free. A decode step serving one token and a decode step serving sixty-four tokens cost almost the same in weight traffic; the second just divides that cost across sixty-four useful outputs.

![Comparison of a batch-one decode reading sixteen gigabytes for one token against a batch-sixty-four decode reading the same sixteen gigabytes for sixty-four tokens](/imgs/blogs/gemm-for-decode-the-skinny-matrix-problem-5.webp)

The figure makes the reuse literal: the same 16 GB weight read, one token riding it versus sixty-four riding it, cores idle versus cores busy. But the linear rise cannot continue forever, and it stops at two distinct ceilings.

**Ceiling one: the compute peak.** Once M passes the critical batch, arithmetic intensity has climbed past the ridge and you are compute-bound. Now step time grows linearly with M (more tokens, more arithmetic, and the tensor cores are already full), so tokens-per-second *saturates*. The ceiling is peak compute divided by the FLOPs per token, and the FLOPs per token for the linear layers are about `2P` where `P` is the parameter count (each parameter is one multiply-add per token):

$$\text{tok/s}_{\max} \approx \frac{\pi_{\text{peak}}}{2 P}$$

**Ceiling two: the KV cache read comes back.** The derivation above counted only weight traffic, which is fixed per step. But the attention step also reads the KV cache, and *that* read grows with both batch size and context length — every sequence in the batch, at every step, re-reads its entire key-value history. At small batch and short context the KV read is a rounding error next to the 16 GB of weights. At large batch and long context it can overtake the weight read entirely, and then you are bandwidth-bound *again* — but now on the cache rather than the weights, and the throughput ceiling bends back down. This is why long-context serving at high concurrency does not scale the way the compute-peak formula alone would predict, and it is the deep reason the paged-attention kernel and KV quantization exist.

![Timeline of throughput as batch grows from bandwidth-bound at batch one through the ridge point to the compute ceiling and back to bandwidth-bound when the KV read dominates](/imgs/blogs/gemm-for-decode-the-skinny-matrix-problem-4.webp)

The timeline above walks the whole journey: bandwidth-bound at batch 1, linear gains as batch grows, the ridge crossing near M = 164, the compute-bound plateau, and the eventual return to bandwidth-bound territory once the KV read dominates.

#### Worked example: the compute-bound throughput ceiling on an A100

Take Llama-3.1-8B (about 8.03 billion parameters) on an A100 at its 312 dense BF16 TFLOPS. The linear-layer ceiling is `312e12 / (2 * 8.03e9)` = about **19,400 tok/s**. That is the absolute upper bound on decode throughput from the projection and MLP GEMMs alone, ignoring attention — a ceiling you approach only at batch sizes well past the critical 153, and one you never quite reach because attention, sampling, and Python overhead all take their cut. On a 4090 the same formula gives `165.2e12 / (2 * 8.03e9)` = about **10,300 tok/s**. Compare those to the batch-1 numbers: the whole 16 GB weight read at 85% of a 4090's bandwidth is `16.06e9 / (0.85 * 1.008e12)` = about 18.7 ms per token, or about **53 tok/s**; on the A100 at 85% of 2,039 GB/s it is about 9.3 ms, or about **108 tok/s**. Batching is the difference between 53 and 10,000. No kernel is.

| Regime | 4090 | A100 | Source |
|---|---|---|---|
| Batch-1 decode, weights only | ~53 tok/s | ~108 tok/s | derived (85% of peak BW) |
| Compute-bound ceiling (linear layers) | ~10,300 tok/s | ~19,400 tok/s | derived (peak / 2P) |
| Critical batch to reach the ceiling | ~164 | ~153 | derived |

---

## 8. Stress tests

A model is only as trustworthy as its behavior at the edges, so push the arithmetic-intensity story into three places where it bends. Each one changes the *bytes* in the denominator, which is the only thing that moves a bandwidth-bound operation.

![A stack of the three levers on decode throughput quantize batch and sparse MoE reads above the non-lever of faster tensor cores](/imgs/blogs/gemm-for-decode-the-skinny-matrix-problem-7.webp)

The stack above names the levers, and the crucial one at the bottom is the non-lever: buying a GPU with faster tensor cores does nothing for batch-1 decode, because the tensor cores are already idle. Everything that helps changes how many weight bytes you move or how many tokens share the move.

**Int4 weights (forward link: [dequant-fused GEMM](/blog/machine-learning/inference-engineering/dequant-fused-gemm-int4-weights-on-the-fly)).** Quantize the weights to 4 bits and the weight bytes drop by 4x: the 16 GB decode read becomes about 4 GB, and the batch-1 throughput ceiling on a 4090 rises from about 53 to about **213 tok/s**. The arithmetic intensity at M = 1 rises correspondingly: FLOPs are unchanged (you must dequantize back to bf16 to do the math, because the tensor cores multiply in bf16), but bytes are quartered, so `AI ≈ 4M` and the critical batch *falls* by 4x to about **41**. This is the whole thesis of weight-only quantization in one line: it is a **bandwidth win, not a FLOP win.** It speeds up the memory-bound regime by moving fewer bytes, and it pushes you into the compute-bound regime sooner — after which it gives you *nothing*, because at large batch you are limited by the bf16 arithmetic the dequantized weights still have to do. If your server runs at batch 256, int4 weight quantization will not speed up your steady-state throughput at all; it only buys memory and helps the low-batch tail.

**MoE, where only some experts are read (forward link: Track G).** A mixture-of-experts model like Qwen3-30B-A3B has about 30 billion total parameters but activates only about 3 billion per token (the "A3B" in the name). At batch 1, decode reads only the active experts' weights — on the order of 6 GB rather than 60 GB — so the per-token bandwidth bill, and thus the batch-1 latency, is set by the *active* parameter count, not the total. This is why MoE models can be so fast to decode at low batch. But the amortization story inverts under batching: as batch grows, different tokens route to different experts, so the *union* of experts touched grows toward all of them, and the effective weight read per step climbs back toward the dense-model figure. MoE trades cheap low-batch decode for worse batching efficiency — the exact opposite of the dense-model curve — which is why MoE serving is so sensitive to expert-parallel layout and routing balance.

**Huge batch and long context, where the KV read overtakes the weights.** Push batch and context far enough and the KV-cache read, which grows as batch times context, overtakes the fixed 16 GB weight read. At that point the down-projection GEMV is no longer what you are waiting on; you are waiting on attention streaming gigabytes of cached keys and values. The whole GEMV analysis still holds — the projections are still bandwidth-bound at their own arithmetic intensity — but they are no longer the bottleneck, and optimizing them further is wasted effort. This is the moment to stop reading this post and go quantize the cache instead.

---

## 9. Case studies and real numbers

Four public results that ground the argument. Every number below is cited to its source with its setup; none is mine.

**vLLM's Triton attention backend: a hand-written kernel matching the vendor.** The vLLM team reports that their pure-Triton attention decode kernel reaches 100.7% of FlashAttention-3's throughput on an H100 (Llama-3.1-8B, batch 1, 500-token input, long decode) in about 800 lines of Triton against FlashAttention-3's roughly 70,000, using the split-KV decode strategy — splitting the KV traversal across a 3D grid and combining partials in a second reduction kernel, heuristic-gated ([vLLM Triton Attention Backend Deep Dive](https://vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive), 2026-03-04). The lesson for our GEMV: at batch 1 the operation is so thoroughly memory-bound that a readable Triton kernel can tie a heroic hand-CUDA one, because both are pinned to the same bandwidth ceiling and neither can use the tensor cores. Cleverness cannot beat the memory wall; it can only reach it.

**HPC-Ops: fusing the gather away.** The Tencent Hunyuan HPC-Ops backend's fused FP8 MoE Gate-Up GEMM reads tokens through the routing index rather than gathering them into a contiguous buffer first, and fuses activation and FP8 quantization into the same pass, using Programmatic Dependent Launch to erase the bubble between kernel stages. They report the fused GEMM at 42.0 microseconds versus 56.4 for a Triton baseline and 74.5 for CUTLASS (H20, 192 experts, top-8, TP8/EP1, batch 4), and roughly 24% lower TTFT and 17% lower TPOT end-to-end on 8xH20 ([vLLM HPC-Ops](https://vllm.ai/blog/2026-07-06-vllm-hpc-ops), 2026-07-06). The win is not more arithmetic — it is one fewer pass over memory. That is the only kind of win available in this regime.

**Blackwell InferenceMAX: the batch-1 versus high-batch framing made explicit.** The vLLM team's Blackwell writeup frames performance as a Pareto curve of throughput against per-token latency, not a single peak, precisely because batch-1 latency and high-batch throughput are different points on the roofline. They report up to 4x higher throughput at similar latency versus Hopper, with gpt-oss-120B at 1K/1K chat up to 4.3x and Llama-3.3-70B at 1K/8K reasoning up to 3.7x, powered by FlashInfer's FP8/FP4 GEMMs and full compute-communication overlap via async scheduling; B200 brings 192 GB of HBM3e at 8 TB/s and native FP4 ([vLLM + NVIDIA Blackwell](https://vllm.ai/blog/2025-10-09-blackwell-inferencemax), 2025-10-09). Note the direction of travel: the hardware fix for the skinny-matrix problem is more bandwidth (8 TB/s) and cheaper bytes (native FP4), not more FLOPs.

**GB300 and the batch effect, quantified.** For a concrete look at how much batch changes the picture, the vLLM team's GB300 numbers for DeepSeek-V3.2 (NVFP4, TP2) report 7,360 tokens per GPU per second on a prefill-heavy shape (ISL 2K, OSL 1) at batch 64, versus 2,816 on a mixed shape (ISL 2K, OSL 1K) — and note that throughput plateaus beyond very large token batches as the sparse-attention indexer overhead (2.7x the kernel time of standard MLA) starts to bite ([DeepSeek-V3.2 on GB300](https://vllm.ai/blog/2026-02-13-gb300-deepseek), 2026-02-13). The plateau is the compute ceiling and the KV-read ceiling of this post, showing up in someone else's benchmark.

---

## 10. When to reach for this, and when not to

Here is the decisive guidance, because the temptation after reading this is to go write kernels, and that is usually the wrong move.

**Do not hand-write a decode GEMV for production.** cuBLASLt, FlashInfer, and the Triton kernels inside vLLM already dispatch competent small-M kernels for the standard shapes, and they get 80–90% of bandwidth without your help. If you profile a decode step (start with [the kernel landscape](/blog/machine-learning/inference-engineering/the-inference-kernel-landscape-what-actually-runs) post's nsys walkthrough) and the projection GEMVs are already in the high 80s as a fraction of peak bandwidth, there is nothing to win and you should close the editor.

**Do write your own — or reach for a specialized library — when the fusion is the point.** The place a custom kernel earns its keep is not the bare GEMV; it is a *fused* one: unpacking int4 weights in registers and feeding the dequantized values straight into the reduction without a separate dequantize pass (the subject of the next post), or reading MoE tokens through a routing index to skip a gather. Those fuse away an entire trip to memory, and in a bandwidth-bound regime one fewer memory pass is the whole game.

**When in doubt, use vLLM.** The honest framing of this whole series is that `nanoserve` exists to teach you what vLLM is doing, not to replace it. For the decode GEMV specifically, the gap between a good hand kernel and the production library is small — both hit the bandwidth wall — so the return on writing your own is low unless you are doing research on a genuinely novel shape or precision. Understand the arithmetic-intensity argument deeply; let someone else maintain the kernel.

**And above all, fix the batch before you fix the kernel.** If your server runs at batch 1, no kernel will save you — you are at 0.5% of compute and roughly 53 tok/s on a 4090 no matter how good your GEMV is. The single highest-leverage change is the continuous-batching loop that raises M, because M is the arithmetic intensity, and arithmetic intensity is the only thing standing between you and the tensor cores you already paid for.

---

## Key takeaways

- A transformer's projection and MLP layers are matmuls `[M, K] x [K, N]`; **M is the number of tokens you compute this step**, and at decode with batch 1, M = 1, turning every GEMM into a GEMV.
- **Arithmetic intensity of a GEMM is `MNK / (MK + KN + MN)`**, which collapses to about **1 FLOP/byte at M = 1** and, in the weight-dominated regime, simplifies to the beautiful identity **AI ≈ M**.
- At batch 1 you **read the entire model — about 16 GB for an 8B model in bf16 — once per token**, and do one multiply-add per byte, so the tensor cores run at roughly **0.5% of peak**. This is structural, not a bug.
- The **critical batch size** where decode becomes compute-bound is `M* = peak_FLOPs / peak_bandwidth`: about **164 on a 4090, 153 on an A100, 295 on an H100, 403 on an L4**. Counterintuitively, faster GPUs need *bigger* batches to be fed.
- For a decode kernel, **report achieved bandwidth as a fraction of peak, not TFLOP/s.** TFLOP/s is fixed near 0.5% by the operation and misleads everyone; a healthy kernel hits **80–90% of peak HBM bandwidth**.
- **Batching is the only lever that turns a GEMV back into a GEMM.** Throughput rises linearly with M (`M * BW / weight_bytes`) until the ridge, then saturates at the compute ceiling `peak / 2P`, then bends back down when the KV read overtakes the weight read.
- **Weight-only quantization (int4) is a bandwidth win, not a FLOP win**: it quarters the bytes and quarters the critical batch, but gives nothing once you are compute-bound.
- **cuBLAS big-tile kernels waste the M dimension at M = 1**; use a GEMV or split-K kernel. Split-K parallelizes the long K reduction and helps when small N leaves the SMs idle — at the cost of a combine step.

## Further reading

- [What inference engineering is](/blog/machine-learning/inference-engineering/what-inference-engineering-is) — the series intro and the TTFT / TPOT / tok-s / memory / goodput scoreboard this post moves.
- [The inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook) — the capstone that benchmarks `nanoserve` against vLLM and tallies what it still loses.
- [The inference kernel landscape](/blog/machine-learning/inference-engineering/the-inference-kernel-landscape-what-actually-runs) — one decode step down to its kernel list, prefill-GEMM versus decode-GEMV, with nsys.
- [Dequant-fused GEMM: int4 weights on the fly](/blog/machine-learning/inference-engineering/dequant-fused-gemm-int4-weights-on-the-fly) — the next post: unpacking 4-bit weights in registers, why it is a bandwidth win, and where it stops paying.
- [Writing a continuous batching loop](/blog/machine-learning/inference-engineering/writing-a-continuous-batching-loop) — the loop that raises M, which is the arithmetic intensity, which is everything.
- [The roofline model](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) — compute-bound versus memory-bound, the ridge point, and where an operation sits.
- [The Triton language documentation](https://triton-lang.org/main/index.html) — `@triton.jit`, `tl.load`/`tl.store`, autotuning, and the reduction primitives used in `gemv.py`.
