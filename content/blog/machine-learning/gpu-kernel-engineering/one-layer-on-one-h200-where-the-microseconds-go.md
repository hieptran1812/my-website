---
title: "One Layer on One H200: Where the Microseconds Go"
date: "2026-10-24"
publishDate: "2026-10-24"
description: "Draw one Muse Spark Open decoder layer as a timeline whose block widths are microseconds, then use the challenge's own numbers to see which block dominates at decode and at prefill, what Amdahl says each problem is worth, and why the whole layer is not the sum of its parts."
tags: ["gpu-kernel-engineering", "open-kernel-challenge", "h200", "moe", "decode", "roofline", "amdahl", "sglang", "muse-spark-open"]
category: "machine-learning"
subcategory: "GPU Kernel Engineering"
author: "Hiep Tran"
featured: false
readTime: 13
---

Before writing a kernel for the Open Kernel Challenge, know how a decode step on one H200 divides up. Dense-model intuition ("attention is the expensive part") is wrong twice here. At one request the layer is mostly weight bytes for dense projections. Averaged over the scored loads, it is mostly routed experts. Attention dominates only one specific corner.

The figure below is the tool this whole series reuses: **the layer strip**. One decoder layer of Muse Spark Open, on one GPU, drawn as a horizontal timeline in fixed order (qkv_gate projection, attention, output projection, router, pooler, routed experts, unpooler), where each block's width is its time in microseconds.

![The layer strip: one Muse Spark Open decoder layer at decode 1, sliding window, each block drawn at its speed-of-light floor, widths in microseconds](/imgs/blogs/one-layer-on-one-h200-where-the-microseconds-go-1.webp)

These are the challenge's published **floors**: per shape, the time to move the data at 4.8 TB/s or do the math at 989 TFLOP/s, whichever is longer. Real kernels are slower; the floor shows what the layer is made of.

## 1. What the numbers are, exactly

Every number here comes from the challenge's public problem API (cached 2026-10-24) or is derived from it with the arithmetic shown. Two kinds matter.

**Floors are per call, per shape.** The task file for `muse_bf16_proj` lists, for example, "decode 1 | m 1, layer qkv_gate | weight 62 | floor 7.9 µs". That is one call of one projection at one token.

**`sglang_us` is per decode step, aggregated.** Each problem defines six loads (decode with 1, 16, 32 requests; 8-token speculative verify blocks for 1, 16, 32 requests). A load's time is the sum of its shapes' times, each multiplied by its weight, which is "how many times one step runs it". The problem's headline SGLang time is then the **geometric mean over the six loads**.

| problem | what one load sums | weights per step | `sglang_us` per decode step |
|---|---|---|---|
| `muse_bf16_proj` | qkv_gate, o, pooler, unpooler, router | 62 each | 4,830.8 |
| `muse_attention_decode` | sliding + full attention | 46 + 16 | 3,766.3 |
| `muse_moe_w4a16` | routed experts | 60 | 13,448.5 |
| `muse_decode_layer` | whole layer, sliding + full | 46 + 16 | 22,620.3 |
| `muse_prefill_layer` | whole layer at prefill shapes | varies | 387,403.9 per prefill load |

The 46 + 16 split is the model's layer mix: three of every four layers attend over a 2,048-token sliding window, every fourth is full attention. The experts problem weights its shapes 60 rather than 62; the task file states the weight without explaining it, so we take it as given. Communication is excluded everywhere: each all-reduce and all-gather is the identity on this GPU's partial result.

So the layer's `sglang_us` divided by 62 gives ${22{,}620.3 / 62 = 364.8}$ µs per layer: a geometric-mean blend of six very different loads, not the time of any one.

## 2. Reading the strip at one request

At one request on a sliding layer, the layer's floor is 40.0 µs, and the seven blocks sum to 40.4 µs. The arithmetic behind the blue blocks is pure weight streaming. Each projection reads its whole weight matrix to produce one output row, so time is bytes over bandwidth. This illustrative script recomputes the published floors from the task file's weight shapes (it runs; real output below):

```python
# Floors from the challenge task.md files (us per call): decode 1, sliding layer
floor = {"qkv_gate": 7.9, "attention": 0.1, "o": 3.5, "router": 1.8,
         "pooler": 14.0, "experts": 11.4, "unpooler": 1.7}
HBM = 4.8e12  # H200 HBM3e, bytes per second
weight_bytes = {
    "qkv_gate": 2304 * 8192 * 2,   # BF16
    "o":        8192 * 1024 * 2,   # BF16
    "pooler":   4096 * 8192 * 2,   # BF16, replicated on every GPU
    "unpooler": 1024 * 4096 * 2,   # BF16, this GPU's 1/8 slice up to 8 tokens
    "router":   256 * 8192 * 4,    # FP32
}
for name, b in weight_bytes.items():
    print(f"{name:9s} {b / 1e6:6.2f} MB -> {b / HBM * 1e6:5.2f} us  (published {floor[name]})")
total = sum(floor.values())
proj = sum(floor[k] for k in weight_bytes)
print(f"strip {total:.1f} us, projections {proj:.1f} us = {proj / total:.0%}")
print(f"62-layer step floor: {46 * 40.0 + 16 * 44.2:.1f} us")
```

```
qkv_gate   37.75 MB ->  7.86 us  (published 7.9)
o          16.78 MB ->  3.50 us  (published 3.5)
pooler     67.11 MB -> 13.98 us  (published 14.0)
unpooler    8.39 MB ->  1.75 us  (published 1.7)
router      8.39 MB ->  1.75 us  (published 1.8)
strip 40.4 us, projections 28.9 us = 72%
62-layer step floor: 2547.2 us
```

Three things jump out. First, **the BF16 projections are 72% of the floor**. The released NVFP4 checkpoint quantizes the experts but keeps every attention projection, the pooler and unpooler in BF16, and the router in FP32. The pooler alone (67 MB) costs more than the routed experts.

Second, **attention is a sliver**. A sliding layer's KV for one request is 2,048 tokens × 2 KV heads × 64 dims × 2 (K and V) × 1 byte of FP8 = 524 KB, which is 0.11 µs at 4.8 TB/s. Even a full layer at 80,000 tokens of context is only 20.5 MB, 4.3 µs.

Third, **the experts are cheap per token, not free**. One expert's NVFP4 weights (gate and up rows, down rows, plus FP8 block scales) come to about 28.3 MB, 5.9 µs to stream. The 11.4 µs floor is roughly two experts' worth; routing is relabelled so this GPU is always the busiest, at about 2.6 experts per token on average.

Multiply out the 62 layers and the floor for a whole one-request decode step on this GPU is ${46 \times 40.0 + 16 \times 44.2 = 2{,}547.2}$ µs, about 2.5 ms, before any communication.

## 3. The dominant block moves

Now change the load. The next figure stacks three strips on one shared microsecond axis. Each row is the sum of the per-problem floors, with the layer's own published floor marked as a tick.

![Three layer strips on one 0 to 240 microsecond axis: decode 1 sliding, decode 32 full attention, and verify 32 sliding, with the dominant block outlined in each](/imgs/blogs/one-layer-on-one-h200-where-the-microseconds-go-2.webp)

What to notice: the dominant block is different in every row.

| load | projections | attention | experts | sum of parts | layer floor | dominant |
|---|---|---|---|---|---|---|
| decode 1, sliding | 28.9 | 0.1 | 11.4 | 40.4 | 40.0 | projections (72%) |
| decode 32, full | 41.7 | 126.4 | 70.8 | 238.9 | 208.1 | attention (53%) |
| verify 32 (256 tokens), sliding | 51.8 | 3.7 | 108.9 | 164.4 | 143.8 | experts (66%) |

All values µs per call at the floor. Each row is a different byte stream:

- **Projections barely grow with batch.** From 1 to 256 tokens, qkv_gate goes from 7.9 to 9.8 µs. Weights are read once however many rows multiply them. The jump in the unpooler (1.7 to 17.4 µs) is a rule of the serving recipe, not physics: up to 8 tokens SGLang computes only this GPU's 1/8 slice of the output and all-gathers it, above 8 it runs the full matrix on every GPU.
- **Attention grows linearly with requests on full layers.** Thirty-two requests each with roughly 80K tokens of FP8 KV is about 32 × 20.5 MB of bytes nobody shares, so decode 32 on a full layer is the one place attention owns the strip. On sliding layers it stays tiny at every batch, because the window caps it at 2,048 tokens.
- **Experts grow with the number of distinct experts touched.** One token touches about 2.6 local experts on the busiest GPU; 256 tokens produce about 390 (token, expert) pairs here, and with 32 local experts most of them get touched. Every newly touched expert is another 28 MB of NVFP4 weights to stream.

This is also the honest version of the claim in the experts problem's description: "With speculative decoding on (SGLang's default), the experts are the largest share of a decode step from 16 requests up." Verify blocks multiply tokens by 8, and tokens are what spread routes across experts.

At 32 requests the sum of parts overshoots the layer's own floor by about 30 µs on both layer types. The task files do not itemize the layer floor; the visible difference is that the standalone experts problem draws worst-case routing while the layer routes with its own router. Treat per-problem floors as an upper bound on the layer's floor, not a decomposition.

## 4. Where SGLang actually spends it

Floors say what the layer is made of. `sglang_us` says where the real microseconds go. Both are aggregated per decode step the same way (the floors by me, with the challenge's weights and geometric mean; the SGLang numbers by the challenge).

![SGLang time versus floor per decode step for projections, attention, routed experts and the whole layer, with the ratio for each](/imgs/blogs/one-layer-on-one-h200-where-the-microseconds-go-3.webp)

| block | SGLang µs/step | floor µs/step | SGLang / floor | µs above floor |
|---|---|---|---|---|
| projections | 4,830.8 | 2,391.1 | 2.0× | 2,439.7 |
| attention | 3,766.3 | 559.8 | 6.7× | 3,206.5 |
| routed experts | 13,448.5 | 3,055.3 | 4.4× | 10,393.2 |
| whole layer | 22,620.3 | 5,843.9 | 3.9× | 16,776.4 |

The projections are the closest to their floor: SGLang's dense GEMMs run at half of speed of light, averaged over loads. Attention is the furthest in ratio (6.7×) but carries the smallest absolute time. The experts are both large and slow relative to their floor: over 10 ms of the 22.6 ms step is experts time above what their bytes require. The mechanics behind those gaps are in [GEMM for decode: the skinny matrix problem](/blog/machine-learning/inference-engineering/gemm-for-decode-the-skinny-matrix-problem) and [dequant-fused GEMM](/blog/machine-learning/inference-engineering/dequant-fused-gemm-int4-weights-on-the-fly); here we only need the budget.

Drawn as a strip of SGLang time rather than floor time, the picture is lopsided:

![The SGLang strip: 22,620.3 microseconds per decode step split into routed experts 59.5 percent, projections 21.4 percent, attention 16.7 percent, and 2.5 percent in no single problem](/imgs/blogs/one-layer-on-one-h200-where-the-microseconds-go-4.webp)

Routed experts are 59.5% of the layer's SGLang time; projections 21.4%; attention 16.7%. The remaining 2.5% is the gap between the layer and the sum of its parts, which section 6 takes apart. Dividing one geometric mean by another is an approximation, but a sound one for ranking.

The one-request picture differs, in the challenge's own words: in SGLang's H200 profile, the dense projections are "about a third of GPU time with one request", and the glue between kernels "alone is a quarter of a decode step with one request". Compare that with the floor, where projections are 72% and glue is near zero. At one request, SGLang's excess time is not in the big GEMMs; it is in the many small kernels around them.

## 5. Amdahl across the five problems

The overall kernel board ranks entrants by "how much your best runs together would speed up a decode step". The exact combination formula is not published in the cached sources, but Amdahl's law gives the shape of the answer. If a block is a fraction $p$ of the step and you make it $s$ times faster, the step speeds up by

$$
S(s) = \frac{1}{(1 - p) + p / s}, \qquad S(\infty) = \frac{1}{1 - p}.
$$

![Amdahl curves for routed experts, projections and attention, whole-step speedup versus speedup on one block, with dashed asymptotes](/imgs/blogs/one-layer-on-one-h200-where-the-microseconds-go-5.webp)

With the shares from section 4, a 2× win on each block is worth very different amounts:

| block | share $p$ | 2× on block | 4× on block | limit at ∞ |
|---|---|---|---|---|
| routed experts | 0.595 | 1.42× | 1.81× | 2.47× |
| projections | 0.214 | 1.12× | 1.19× | 1.27× |
| attention | 0.167 | 1.09× | 1.14× | 1.20× |

Read it against section 4's ratio-to-floor column. Projections cannot go much past 2× before hitting physics, so their realistic ceiling is 1.12×. Attention has 6.7× of headroom, but even an infinitely fast attention kernel buys only 1.20×. The experts have 4.4× of headroom on 59.5% of the step: a 4× experts kernel alone moves the step 1.81×. That asymmetry is why the series spends a whole track on routed experts.

Two caveats. The shares are six-load averages; at one request the experts are a smaller slice and the projections larger. And each problem board takes a geometric mean of per-load speedups, so 3× on one load and 1× on five scores ${3^{1/6} \approx 1.20}$×, not 3×.

## 6. Why `decode_layer` is not the sum of its parts

Add the three component problems: ${4{,}830.8 + 3{,}766.3 + 13{,}448.5 = 22{,}045.6}$ µs. The whole layer is 22,620.3 µs. The gap is 574.7 µs per step, about 9.3 µs per layer, 2.5%. Yet the profile says glue is a quarter of the step at one request. Both are correct; the coverage grid shows why.

![Coverage grid of fifteen layer ops by four problems: which problem times which op, with four ops only the whole-layer problem covers](/imgs/blogs/one-layer-on-one-h200-where-the-microseconds-go-6.webp)

**Much of the glue is already inside the parts.** `muse_attention_decode` starts from the fused projection output and includes QK-norm, RoPE, the FP8 cache write and the gated output norm. `muse_moe_w4a16` includes the per-expert RMSNorm. Only four ops are timed nowhere but the whole layer: the two fused norm-residual boundaries, the router's sigmoid, bias and top-8 selection, and the pre-expert norm. Those are the 2.5%.

**The aggregation does not add.** Summed geometric means are not the geometric mean of summed loads, and the experts problem weights 60 layers, not 62.

**The inputs differ.** The standalone experts problem relabels routing so this GPU is the busiest. The layer computes its routing from its own router on its own residual stream, with scored shapes that give keys "a shared direction, as Muse's real keys have". The task file notes that SGLang's FP8 attention path reroutes 9 to 46% of tokens relative to the FP32 reference. Different routes mean different experts touched.

**The timing boundary differs.** Every timed replay starts from a cold L2, and the judge subtracts a graph replay's fixed cost (about 4 µs) from both sides. Three standalone problems pay three cold starts per layer's worth of work; the whole layer pays one, and its kernels can hand activations through L2 to the next kernel.

The last point is the opening `muse_decode_layer` offers. The challenge's own description calls it "the problem that rewards fusing across ops". At one request, the floor for a whole layer is 40 µs, and each separate small kernel carries a fixed cost that does not shrink with the work. Fusing the norm-residual boundaries into neighbors, and top-8 into the router GEMM's epilogue, removes kernels instead of speeding them up. The CUDA-graph mechanics that set that per-kernel floor are in [CUDA graphs and torch.compile for the decode loop](/blog/machine-learning/inference-engineering/cuda-graphs-and-torch-compile-for-the-decode-loop).

## 7. Prefill: same layer, different physics

`muse_prefill_layer` is the same layer at prefill shapes: a 753-token turn on an 80K conversation, four such turns batched, and a first turn in SGLang's 16,384-token chunks. SGLang takes 387,403.9 µs per prefill load (geometric mean over three loads), on its own board, outside the overall kernel board.

![Two layer strips on separate axes: decode 1 sliding at a 40 microsecond floor set by bytes, and a 16,384-token prefill chunk at about 5,000 microseconds set by FLOPs](/imgs/blogs/one-layer-on-one-h200-where-the-microseconds-go-7.webp)

At 16,384 tokens the weights are amortized and the floor becomes FLOPs. Counting multiply-adds per token on one GPU (2 FLOPs each):

| part | FLOPs per token | at 16,384 tokens and 989 TFLOP/s |
|---|---|---|
| dense GEMMs (qkv_gate 37.7M, o 16.8M, pooler 67.1M, unpooler 67.1M, router 4.2M) | 192.9M | 3,196 µs |
| experts, balanced (one local pair per token, 100.7M per pair) | 100.7M | 1,668 µs |
| sliding attention (16 heads, 2,048 keys, QK and PV) | 8.4M | 139 µs |
| total | 302.0M | 5,003 µs |

The published floor for that chunk is 5,663.4 µs. Our count with perfectly balanced routing lands about 12% below it, which is consistent with the busiest GPU carrying more than its 1/8 of the routes (the experts problem quotes 390 pairs at 256 tokens against a balanced 256). At the floor, dense GEMMs outweigh the experts. Yet the challenge describes SGLang's actual prefill layer as "mostly the routed experts as one large grouped NVFP4 GEMM". The gap between those two sentences is the prefill opportunity: a grouped GEMM over dequantized 4-bit weights runs far below the dense tensor-core rate that the floor assumes.

Aggregated the same way, the floors give 121,862 µs per prefill load: SGLang sits at 3.2× its floor, and every win here is a compute-bound GEMM win.

## 8. How to use the strip

Every later post redraws the strip with one block highlighted. Place each technique on it and ask, in order:

1. **Which load does it help?** A win at decode 1 lives in the projections and in kernel count; a win at verify 32 lives in the experts; a win on full layers at decode 32 lives in attention.
2. **What is the block's share there?** Amdahl caps the payoff at $1/(1-p)$, whatever the kernel does.
3. **How far is the block from its floor?** The projections at 2.0× have little left; the experts at 4.4× have the most absolute time left.

Next: [the roofline for decode on H200](/blog/machine-learning/gpu-kernel-engineering/roofline-for-decode-on-h200-bytes-not-flops) derives every scored floor; [inside the judge](/blog/machine-learning/gpu-kernel-engineering/inside-the-judge-cuda-graph-timing-tolerances-and-reward-hacking) covers the paired-ratio timing; [Muse Spark Open for kernel engineers](/blog/machine-learning/gpu-kernel-engineering/muse-spark-open-for-kernel-engineers) is the anatomy behind each block; [the Open Kernel Challenge decoded](/blog/machine-learning/gpu-kernel-engineering/the-open-kernel-challenge-decoded) has the rules.

> **On the board**
>
> **Problem / track:** all four decode problems and the overall kernel board; `muse_prefill_layer` on its own board. This post moves nothing directly; it decides where your first ten submissions go.
>
> **Lever:** spend effort by share × headroom. Routed experts: 59.5% of the step at 4.4× from floor (`muse_moe_w4a16`, about 10.4 ms per step above floor). Whole-layer fusion of the four ops no component problem times (`muse_decode_layer`), which matters most at one request where glue is a quarter of SGLang's step. Projections are already at 2.0× of floor; attention has 6.7× headroom but caps the step at 1.20×.
>
> **Rule fence:** communication is the identity in every problem, so all-reduce tricks score nothing here. Weights keep their precision (NVFP4 stays NVFP4, BF16 stays BF16; the router may run as a BF16 hi + lo pair). Queries no coarser than FP8 E4M3, KV stays FP8. Every key in the window or context is attended: no top-k or sparse attention. Every call replays inside a CUDA graph on fresh inputs from a cold L2: no host syncs, no caching anything derived from inputs.
