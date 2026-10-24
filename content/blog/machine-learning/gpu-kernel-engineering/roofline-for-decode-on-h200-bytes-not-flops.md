---
title: "Roofline for Decode on H200: Bytes, Not FLOPs"
date: "2026-10-24"
publishDate: "2026-10-24"
description: "Compute the bytes and FLOPs of every scored shape in the Open Kernel Challenge, derive the H200 speed of light for each problem, and read off how far SGLang sits from it before you write a single kernel."
tags: ["gpu-kernel-engineering", "open-kernel-challenge", "roofline", "h200", "llm-inference", "decode", "memory-bandwidth", "moe", "sglang", "arithmetic-intensity"]
category: "machine-learning"
subcategory: "GPU Kernel Engineering"
author: "Hiep Tran"
featured: true
readTime: 13
---

Every problem page in the Open Kernel Challenge has a column most people scroll past. Next to each scored shape, after the weight, sits a number called the **floor**: "the time to move the data at 4.8 TB/s or do the math at 989 TFLOP/s, whichever is longer." That one sentence is a roofline model, and it is the most useful thing on the page. It tells you, before you open Nsight or write a line of CUDA, how much of each problem is left on the table and where it is hiding. For Muse Spark Open at decode, the answer is the same almost everywhere: the time goes into moving bytes, and the math is a rounding error.

![One decode layer at batch 1 moves 212 MB in 44.2 µs and does its math in about 0.6 µs](/imgs/blogs/roofline-for-decode-on-h200-bytes-not-flops-1.webp)

The diagram above is the mental model: one Muse decoder layer on one H200, one request, full attention at 80K context. Reading 212.2 MB of weights and cache at 4.8 TB/s takes 44.2 µs. The arithmetic it feeds takes well under one microsecond at 989 TFLOP/s. The floor is the longer bar, and at decode the longer bar is always bytes. This post recomputes the floor for every scored shape of all five problems from the shapes in the challenge's own sources, checks it against the judge's published column wherever the inputs are public, and then puts SGLang's measured time next to it.

The sibling posts set the stage: [the challenge itself](/blog/machine-learning/gpu-kernel-engineering/the-open-kernel-challenge-decoded), [the model as a kernel engineer sees it](/blog/machine-learning/gpu-kernel-engineering/muse-spark-open-for-kernel-engineers), and [where the microseconds go in one layer](/blog/machine-learning/gpu-kernel-engineering/one-layer-on-one-h200-where-the-microseconds-go). The roofline itself is derived in [the roofline model post](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound); here we only apply it.

## Why decode is different

| Assumption | Naive view | What the shapes say |
| --- | --- | --- |
| GPUs are FLOP machines | A faster GEMM means better tiling for tensor cores | At m=1 a BF16 projection does 1 FLOP per byte; the tensor cores idle |
| Bigger batch, compute-bound | 32 requests is "a real batch" | 32 tokens is 31 FLOP/byte, still 6.6× left of the ridge |
| Attention is the expensive part | FlashAttention math dominates | Decode attention is 16 FLOP/byte at any batch: it is a memory copy with a softmax |
| The floor is a fixed number | Bytes are bytes | The MoE floor depends on how many experts the router touches |

## 1. Two numbers define the roof

**Rule of thumb: divide peak FLOP/s by peak bytes/s and you get the only threshold that matters.** NVIDIA's [H200 page](https://www.nvidia.com/en-us/data-center/h200/) lists 4.8 TB/s of HBM3e and 1,979 BF16 tensor TFLOPS, footnoted "with sparsity", so dense BF16 is about 989 TFLOP/s. The ratio is

$$
\text{ridge} = \frac{989 \times 10^{12}\ \text{FLOP/s}}{4.8 \times 10^{12}\ \text{B/s}} \approx 206\ \text{FLOP/byte}.
$$

A kernel whose arithmetic intensity (FLOPs per byte of HBM traffic) is below 206 cannot finish faster than its bytes divided by 4.8 TB/s, no matter how clever its math. Above 206, the tensor cores set the floor.

Two details of the harness make this cleaner than usual. First, every timed replay starts with a cold L2: the judge reads a buffer larger than L2 just before each replay, so weights cannot hide in cache between iterations. Second, the judge subtracts a graph replay's fixed cost (about 4 µs) from both sides, so the floor is compared against kernel time, not launch overhead.

![The H200 roofline with every problem's representative shapes: decode lives on the slope, prefill on the flat roof](/imgs/blogs/roofline-for-decode-on-h200-bytes-not-flops-2.webp)

The roofline card above places each problem's shapes by intensity. Everything you will be timed on at decode sits on the slanted part of the roof. Only three groups make it to the flat part: the largest verify projections, and both prefill loads.

### Second-order: the judge's peak is BF16

The judge uses 989 TFLOP/s for every problem, including attention, whose FP8 path in FA3 casts the query to E4M3. If you use FP8 tensor cores, the compute roof doubles (the page lists 3,958 FP8 TFLOPS with sparsity) and the ridge moves to about 412. That makes the decode shapes even more firmly memory-bound, not less.

## 2. Projections: intensity is just m

**Rule of thumb: for a BF16 GEMM with BF16 weights, arithmetic intensity is roughly the number of tokens.** The problem computes `out[m, n] = x[m, k] @ W[n, k]^T`. FLOPs are $2mnk$. Bytes are the weight $2nk$ plus the activations $2mk$ and the output $2mn$. When $m$ is small next to $n$ and $k$, intensity is $2mnk / 2nk = m$. The ridge sits at $m \approx 206$, so of the six scored token counts (1, 8, 16, 32, 128, 256), only 256 crosses it.

![The same qkv_gate weight is memory-bound at m=1 and compute-bound at m=256](/imgs/blogs/roofline-for-decode-on-h200-bytes-not-flops-3.webp)

The worked example above is the fused `qkv_gate` projection, 2304 × 8192 in BF16 (16 query, 2 key, 2 value and 16 gate heads of 64 on this GPU under TP8). At m=1 it reads 37.77 MB and needs 7.87 µs; its 37.7 MFLOP would take 0.04 µs. At m=256 the bytes grow only to 43.12 MB (8.98 µs), but the math grows 256× to 9.66 GFLOP, which is 9.77 µs. The floor flips sides. The pooler and unpooler (4096 × 8192 and 8192 × 4096) cross at m=256 too, which is why their judge floors jump from 14.6 µs at m=128 to 17.4 µs.

Here is the arithmetic as a script. It is illustrative, restated from the formulas the judge's `reference.py` uses, and the output below is its real output under `python3 -I`:

```python
import math

BW, PEAK = 4.8e12, 989e12   # H200 SXM: HBM3e bytes/s, dense BF16 FLOP/s
H, Q, KV, DE = 8192, 1024, 128, 4096   # hidden, q width, kv width per GPU (TP8), MoE width

def floor_us(bytes_, flops):
    return 1e6 * max(bytes_ / BW, flops / PEAK), flops / bytes_

# Dense projection: out[m, n] = x[m, k] @ W[n, k]^T, weights read once.
def proj(m, n, k, wbytes=2):
    return floor_us(n * k * wbytes + m * k * 2 + m * n * wbytes, 2 * m * n * k)

# Attention decode: read the FP8 K and V of every attended token once.
def attn(batch, ctx, q_len, sliding, share):
    toks = batch * (min(ctx, 2048) if sliding else ctx) * share
    io = batch * q_len * (2 * 16 + 2 * 2) * 64 * 2 * 2
    return floor_us(toks * KV * 2 + io, 4 * toks * q_len * Q)

print(f"ridge = {PEAK / BW:.1f} FLOP/byte")
for m in (1, 8, 32, 128, 256):
    us, ai = proj(m, 2304, H)
    print(f"qkv_gate m={m:3d}: {us:5.2f} us  AI {ai:6.1f}  {'compute' if ai > PEAK / BW else 'memory'}")
for b, ql, sl, sh in ((1, 1, True, 1.0), (1, 1, False, 1.0), (32, 1, False, 0.925), (32, 8, False, 0.925)):
    us, ai = attn(b, 80000, ql, sl, sh)
    print(f"attn batch={b:2d} q_len={ql} {'sliding' if sl else 'full   '}: {us:6.1f} us  AI {ai:5.1f}")

# Ceiling on the score: geomean over loads of (SGLang time / floor time).
loads = {"bf16_proj": (4830.8, [1791.8, 2573.0, 2585.4, 1798.0, 2715.6, 3211.6]),
         "attention_decode": (3766.3, [73.4, 1084.8, 2174.2, 73.4, 1101.8, 2199.0]),
         "moe_w4a16": (13448.5, [684.0, 3150.0, 4248.0, 2238.0, 6078.0, 6534.0]),
         "decode_layer": (22620.3, [2547.2, 5728.2, 7216.6, 3928.2, 8842.8, 10890.0]),
         "prefill_layer": (387403.9, [19769.2, 79548.2, 1150766.8])}
for name, (sglang, fl) in loads.items():
    g = math.exp(sum(map(math.log, fl)) / len(fl))
    print(f"{name:17s} floor {g:9.1f} us  SGLang {sglang:9.1f} us  at {100 * g / sglang:4.1f}% of SOL, ceiling {sglang / g:4.2f}x")
```

```console
ridge = 206.0 FLOP/byte
qkv_gate m=  1:  7.87 us  AI    1.0  memory
qkv_gate m=  8:  7.90 us  AI    8.0  memory
qkv_gate m= 32:  8.00 us  AI   31.4  memory
qkv_gate m=128:  8.42 us  AI  119.5  memory
qkv_gate m=256:  9.77 us  AI  224.1  compute
attn batch= 1 q_len=1 sliding:    0.1 us  AI  15.7
attn batch= 1 q_len=1 full   :    4.3 us  AI  16.0
attn batch=32 q_len=1 full   :  126.4 us  AI  16.0
attn batch=32 q_len=8 full   :  126.8 us  AI 127.5
bf16_proj         floor    2391.1 us  SGLang    4830.8 us  at 49.5% of SOL, ceiling 2.02x
attention_decode  floor     559.8 us  SGLang    3766.3 us  at 14.9% of SOL, ceiling 6.73x
moe_w4a16         floor    3055.3 us  SGLang   13448.5 us  at 22.7% of SOL, ceiling 4.40x
decode_layer      floor    5843.9 us  SGLang   22620.3 us  at 25.8% of SOL, ceiling 3.87x
prefill_layer     floor  121862.2 us  SGLang  387403.9 us  at 31.5% of SOL, ceiling 3.18x
```

A fuller version of the same formulas over all 30 `bf16_proj` shapes and all 12 attention shapes matches the judge's published floor to the 0.1 µs it is printed at. The load floors in the `loads` table are the judge's per-shape floors times each shape's weight, summed per load.

### Second-order: the router and the unpooler

Two projections break the pattern. The router's weights are FP32 (256 × 8192, 8.39 MB), so its intensity is at most half of m: 0.5 at m=1, 83.6 at m=256, and it never crosses. The unpooler at up to 8 tokens computes only this GPU's 1/8 of the output (rows 0 to 1023), so its floor is 1.7 µs at m=1 and 14.1 µs at m=16. That 8× step is a property of the serving recipe, not a kernel inefficiency.

## 3. Attention decode: 16 FLOPs per byte, forever

**Rule of thumb: decode attention intensity is set by the GQA ratio, not by the batch.** Each KV head serves 8 query heads here (16 query, 2 KV on this GPU). Per attended token, the kernel reads 256 bytes of FP8 cache and does $4 \times 1024$ FLOPs for one query token (QK and PV, 2 FLOPs each, across 1024 query dimensions). That is $4096 / 256 = 16$ FLOP/byte, at batch 1 and at batch 32 alike.

![One token's KV row is 256 bytes; at batch 32 the full layer's cache is 606 MB, more than all the dense weights](/imgs/blogs/roofline-for-decode-on-h200-bytes-not-flops-4.webp)

The cells above build the cache up from one token. A sliding layer reads 2,048 tokens per request, 524 KB. A full layer reads all 80,000, 20.5 MB per request. At batch 32 the judge's lower bound uses 0.925 × 80,000 tokens per request, because "agentic" lengths are drawn uniformly from 80,000 ± 7.5% and the bound must stay below the mean. That is 606 MB and 126.4 µs, more than three times the layer's 188.7 MB of dense BF16 weights.

Verify blocks raise intensity eight-fold, to 127.5 FLOP/byte at q_len 8, because 8 query tokens reuse every byte of cache read. That is still left of the ridge, and the floor barely moves: 126.4 µs becomes 126.8 µs.

### Second-order: the sliding floor is fiction

The sliding layer at batch 1 has a floor of 0.1 µs (524 KB at 4.8 TB/s is 0.11 µs). It carries weight 46, so in the decode 1 load it contributes 4.6 µs of floor against 68.8 µs from the 16 full layers. No kernel finishes in 0.1 µs: filling 132 SMs with enough loads in flight to reach peak bandwidth takes longer than that. The judge already removes the ~4 µs replay cost, but a kernel's own ramp and tail remain. For tiny shapes, the honest floor is "one well-shaped kernel launch," and the work is fusing it with its neighbours rather than optimizing it alone.

## 4. Routed experts: the floor depends on the router

**Rule of thumb: an MoE floor counts experts touched, not tokens.** Each local expert is 3 × 4096 × 4096 NVFP4 weights (w13 is 8192 × 4096 and w2 is 4096 × 4096). NVFP4 stores 4 bits per weight plus one E4M3 scale byte per 16 weights, so 0.5625 bytes per weight, and one expert is 28.31 MB. The judge's `speed_of_light_ns` reads only experts that receive at least one token, and counts 0.75 × the mean number of distinct local experts in its routing (the exact means live in a task file that is not public). Inverting the published floors gives the expert count each one credits:

| Load | bs | Judge floor | Bytes at 4.8 TB/s | Experts credited |
| --- | --- | --- | --- | --- |
| decode 1 | 1 | 11.4 µs | 54.7 MB | 1.9 |
| verify 1 | 8 | 37.3 µs | 178.9 MB | 6.3 |
| decode 16 | 16 | 52.5 µs | 251.7 MB | 8.9 |
| decode 32 | 32 | 70.8 µs | 339.3 MB | 12.0 |
| verify 16 | 128 | 101.3 µs | 484.1 MB | 17.1 |
| verify 32 | 256 | 108.9 µs | 518.5 MB | 18.3 |

![Of this GPU's 32 experts, the floor credits 1.9 at one token, 12.0 at 32 and 18.3 at 256](/imgs/blogs/roofline-for-decode-on-h200-bytes-not-flops-6.webp)

The cells above make the shape visible. Even at 256 tokens, with 2,048 routes per step, the floor credits only 18.3 of 32 experts, because the generator gives experts a heavy-tailed popularity and the busiest GPU is chosen on purpose. Is any of it compute-bound? Per expert, intensity is $2t / 0.5625 \approx 3.56\,t$ FLOP/byte for $t$ tokens on that expert, so crossing 206 needs about 58 tokens per expert. At bs=256 the busiest GPU has at least 256 routes (one per token on average) spread over roughly 18 experts, about 14 each. Memory-bound by 4×.

## 5. The whole layer: weights at batch 1, cache at batch 32

**Rule of thumb: the dominant byte changes with the load, so the optimization changes too.** `muse_decode_layer` adds the pieces: dense BF16 projections, the FP32 router, touched experts, KV cache, and activations. Subtract every known term from the judge's floor and what remains is the expert term, and it lines up with the MoE problem's counts (about 1.9 experts at batch 1).

![Four decode-layer loads stacked by what the floor reads: weights dominate at batch 1, KV cache at batch 32](/imgs/blogs/roofline-for-decode-on-h200-bytes-not-flops-5.webp)

At decode 1 the layer reads 192 to 212 MB, and 130.0 MB of it is dense weights. At decode 32 with full attention it reads 998.9 MB, and 606.2 MB of that is KV cache. At verify 32 the expert term grows to 454.1 MB (about 16 experts) while the cache stays put. A fused layer kernel that wins at batch 1 by streaming weights perfectly can still lose at batch 32 if its attention reads the cache at 60% of bandwidth.

One more check falls out for free. At decode 1, the three parts' load floors sum to 1,791.8 + 73.4 + 684.0 = 2,549.2 µs, against 2,547.2 µs for the layer. Fusion cannot beat the floor of the bytes; it can only remove the launch gaps and round trips between kernels that the floors never counted. The [layer-strip post](/blog/machine-learning/gpu-kernel-engineering/one-layer-on-one-h200-where-the-microseconds-go) shows where SGLang loses that time.

## 6. Prefill crosses the ridge

**Rule of thumb: at hundreds of tokens per step, everything is a GEMM again.** The prefill loads run 753 tokens per request (one turn) or 16,384-token chunks of a long first prompt. At 753 tokens a whole sliding layer sits at roughly 515 FLOP/byte; at 16,384 it reaches about 2,500. Every prefill shape is compute-bound, and the judge's floors grow with tokens: 257.0 µs per sliding layer at turn 1, 5,663.4 µs per sliding chunk in the first turn. The first-turn load alone has a floor of 1,150,766.8 µs, and since the score is a geometric mean over the three loads, it counts exactly as much as the 19,769.2 µs turn-1 load.

My lower-bound version of the formula lands 5 to 12% under the judge's prefill floors, because it assumes one local route per token; the judge uses the measured route count. The direction of the bound is what matters: prefill is a tensor-core problem, and grouped GEMM efficiency, not bandwidth, is the lever.

## 7. The ceiling on your score

**Rule of thumb: SGLang time divided by floor time is the best speedup any kernel can post.** The score is the geometric mean over loads of SGLang's load time over yours. The `sglang_us` on each problem page is SGLang's time per step averaged over loads the same way. So the geometric mean of the load floors, divided into `sglang_us`, is the speedup you would post by hitting the floor on every shape.

![SGLang runs at 15 to 50% of the H200 speed of light, which caps each problem's possible speedup](/imgs/blogs/roofline-for-decode-on-h200-bytes-not-flops-7.webp)

| Problem | Floor per step (geomean) | SGLang `sglang_us` | SGLang at % of floor | Ceiling |
| --- | --- | --- | --- | --- |
| `muse_bf16_proj` | 2,391.1 µs | 4,830.8 µs | 49.5% | 2.02× |
| `muse_attention_decode` | 559.8 µs | 3,766.3 µs | 14.9% | 6.73× |
| `muse_moe_w4a16` | 3,055.3 µs | 13,448.5 µs | 22.7% | 4.40× |
| `muse_decode_layer` | 5,843.9 µs | 22,620.3 µs | 25.8% | 3.87× |
| `muse_prefill_layer` | 121,862.2 µs | 387,403.9 µs | 31.5% | 3.18× |

No kernel sustains 4.8 TB/s, so treat these as ceilings, not targets. If a kernel reaches 90% of spec bandwidth on every shape, the projection ceiling becomes 0.9 × 2.02 = 1.82×.

> The floor does not tell you how to go fast. It tells you where going fast is still possible.

## Case studies from the board

### 1. The projections are already half-way there

`muse_bf16_proj` has the least headroom: SGLang is at 49.5% of the floor. Every shape but three is pure weight streaming, so the work is a bandwidth-saturating skinny GEMM for m between 1 and 128 (the [skinny GEMM post](/blog/machine-learning/inference-engineering/gemm-for-decode-the-skinny-matrix-problem) covers the mechanics), and real tensor-core tiling only for m=256. A 2× ceiling means a 1.3× result is a good result here.

### 2. Attention has the most room, in the least obvious place

At 14.9% of floor, attention looks like the jackpot. But the decode 1 and verify 1 loads have floors of 73.4 µs, of which 4.6 µs is sliding layers that will never run near their 0.1 µs floor. The load floors with real bytes are the batched full layers: 63.2 µs and 126.4 µs per shape of streaming FP8 cache. That is a split-KV problem across 132 SMs with only 2 KV heads per request, so the lever is parallelism over the sequence, not math.

### 3. The MoE floor moves with the data

The MoE ceiling of 4.40× is computed on a floor that credits 18.3 experts at bs=256. A kernel that streams all 32 local experts regardless of routing reads 32 / 18.3 = 1.75× the bytes the floor counts at that load. The floor rewards kernels that skip untouched experts cheaply, and the held-out reruns punish kernels tuned to one popularity profile; the [load-imbalance post](/blog/machine-learning/inference-engineering/moe-inference-routing-expert-parallel-and-the-load-imbalance-problem) covers why that skew exists.

## When to trust the floor, and when not to

Reach for the floor when:

- Choosing which problem to work on: the ceiling column ranks the headroom before you spend one of the 10 scored runs a day.
- Deciding between a bandwidth fix and a math fix: left of 206 FLOP/byte, only bytes and parallelism matter.
- Sanity-checking a result: a time below the floor means the timing or the correctness check is broken.
- Comparing loads within one problem: the geometric mean makes a 73 µs load count as much as a 2,199 µs one.

Distrust the floor when:

- The shape is tiny (sliding attention at batch 1, routers): launch, ramp and tail dominate.
- The shape is MoE: the floor follows the routing, which changes on held-out reruns.
- You are reading prefill floors as a bandwidth story: they are FLOP floors.

## On the board

> **On the board.** *Moves:* all five kernel-track problems, as the map of where headroom lives (projections 2.02×, attention 6.73×, MoE 4.40×, decode layer 3.87×, prefill 3.18×). *Lever:* know which side of the 206 FLOP/byte ridge each shape sits on before choosing between bandwidth and tensor-core work. *Rule fence:* you cannot lower the floor by changing precision. NVFP4 weights stay NVFP4 and BF16 weights stay BF16 (lossless compression aside, and the router's BF16 hi + lo split is allowed), and every replay starts from a cold L2, so caching weights across calls is not a byte you get to skip.
