---
title: "Muse Spark Open for Kernel Engineers: One Layer, One H200, Every Shape That Matters"
date: "2026-10-24"
publishDate: "2026-10-24"
description: "Read Muse Spark Open the way the Open Kernel Challenge makes you read it: the exact per-GPU shapes under TP8/EP8, where the bytes live, what the router, norms and gated attention force on a kernel, and what SGLang runs today."
tags: ["gpu-kernel-engineering", "open-kernel-challenge", "mixture-of-experts", "muse-spark-open", "nvfp4", "gqa", "sliding-window-attention", "h200", "sglang", "llm-inference"]
category: "machine-learning"
subcategory: "GPU Kernel Engineering"
author: "Hiep Tran"
featured: true
readTime: 14
---

Most model write-ups read like a real-estate brochure: rooms, square footage, a nice view. A kernel engineer needs the plumbing diagram. How wide is each pipe on *this* GPU, which ones are shared with the neighbours, and where does the water pool? The [Open Kernel Challenge](/blog/machine-learning/gpu-kernel-engineering/the-open-kernel-challenge-decoded) hands us exactly that plumbing for Muse Spark Open, an 816B-parameter mixture-of-experts model with 42B active, served on 8×H200 under 8-way tensor parallelism and 8-way expert parallelism (TP8/EP8). The judge's `reference.py` files spell out every shape one GPU sees. This post reads them.

![Width ribbon of one Muse Spark Open layer on one GPU: 8192 narrows to 2304, 1024, widens to 8192, narrows to 4096 through the experts, widens to 8192](/imgs/blogs/muse-spark-open-for-kernel-engineers-1.webp)

The ribbon above is the mental model: one decoder layer on one GPU is a stream 8,192 wide that pinches down twice, once to a 1,024-wide attention slice and once to a 4,096-wide MoE block, and widens back each time. Everything below is a consequence of where those pinches sit, which matrices make them, and in what precision.

## 1. One layer, one GPU: the shapes under TP8/EP8

The hidden size is 8,192 and the challenge never lets you forget it. The residual stream travels between layers in FP32, `[T, 8192]`, where `T` is tokens per step (batch times query length: 1 to 32 at decode, 8 to 256 for speculative verify blocks). The attention input is that stream RMS-normalized and cast to BF16.

Here is what this GPU holds for one layer, straight from the challenge's `LAYERS` table and `generate_weights`:

| Weight | Per-GPU shape `[N, K]` | Dtype | How it is split |
| --- | --- | --- | --- |
| `qkv_gate` | 2304 × 8192 | BF16 | column-parallel: q 16×64, k 2×64, v 2×64, gate 16×64 |
| `o` | 8192 × 1024 | BF16 | row-parallel, all-reduce after |
| `router` | 256 × 8192 | FP32 | replicated |
| `pooler` | 4096 × 8192 | BF16 | **replicated on every rank** |
| `w13` per expert | 8192 × 4096 | NVFP4 | 32 local experts of 256 (EP8) |
| `w2` per expert | 4096 × 4096 | NVFP4 | 32 local experts of 256 (EP8) |
| `unpooler` | 8192 × 4096 | BF16 | replicated; at `T <= 8` only this rank's 1,024 output rows run, then all-gather |

Three things stand out.

First, attention is tiny per GPU. Sixteen query heads and two KV heads of dimension 64 means the attention working set on this GPU is 1,024 wide for queries and 128 wide for each of K and V. Across the node that is 128 query heads and 16 KV heads.

Second, the MoE block does not run at the hidden width. A `pooler` projects 8,192 down to 4,096, every expert works at 4,096 in and 4,096 intermediate, and an `unpooler` projects back. That is a latent MoE: the experts never touch the 8,192-wide stream directly. The price is two dense BF16 matrices that every GPU holds in full.

Third, the step weights in each problem's shape table quietly tell you the layer count. The projections run with weight 62, attention with 46 sliding plus 16 full, and the routed experts with weight 60. So a decode step is 62 layers, 60 of which run routed experts. The challenge sources do not say what the other two run instead, so we will not guess.

## 2. Where the bytes are

At decode, every kernel in this layer is a memory problem, and the [roofline post](/blog/machine-learning/gpu-kernel-engineering/roofline-for-decode-on-h200-bytes-not-flops) makes that rigorous. Here we only need the inventory. The script below is illustrative arithmetic over the reference shapes. It needs no GPU, runs under `python3 -I`, and its output is pasted unedited.

```python
# Per-GPU byte budget for one Muse Spark Open layer under TP8/EP8.
# Shapes from the challenge's reference.py; illustrative arithmetic, not a benchmark.
HIDDEN, D_EXPERT, HEAD_DIM, HQ, HKV = 8192, 4096, 64, 16, 2
HBM = 4.8e12  # H200 SXM bytes/s

dense = {  # name: (rows, cols, bytes per element)
    "qkv_gate": (2 * HQ * HEAD_DIM + 2 * HKV * HEAD_DIM, HIDDEN, 2),
    "o":        (HIDDEN, HQ * HEAD_DIM, 2),
    "router":   (256, HIDDEN, 4),
    "pooler":   (D_EXPERT, HIDDEN, 2),
    "unpooler": (HIDDEN, D_EXPERT, 2),
}
for name, (n, k, b) in dense.items():
    mb = n * k * b / 1e6
    print(f"{name:9s} {n:5d} x {k:5d}  {mb:6.1f} MB  {mb * 1e6 / HBM * 1e6:5.2f} us")

# NVFP4: 0.5 B per weight + one 1-byte E4M3 scale per 16 weights.
w13, w2 = 2 * D_EXPERT * D_EXPERT, D_EXPERT * D_EXPERT
expert = (w13 + w2) * (0.5 + 1 / 16)
print(f"one expert          {expert / 1e6:6.1f} MB  ({(w13 + w2) / 1e6:.1f}M params)")
print(f"32 local experts    {32 * expert / 1e6:6.1f} MB")

# Parameter count over the layers the step weights imply: 62 layers, 60 with routed experts.
dense_params = (HIDDEN * (8 * 2304 + 8 * 1024) + 2 * D_EXPERT * HIDDEN + 256 * HIDDEN) * 62
print(f"total  {(256 * 60 * (w13 + w2) + dense_params) / 1e9:5.1f}B   "
      f"active {(8 * 60 * (w13 + w2) + dense_params) / 1e9:5.1f}B")

kv = 2 * HKV * HEAD_DIM  # K and V, FP8 = 1 byte each
print(f"KV per token per layer {kv} B; window 2048 -> {2048 * kv / 1e6:.2f} MB, "
      f"80K full -> {80000 * kv / 1e6:.1f} MB per request")
```

```console
qkv_gate   2304 x  8192    37.7 MB   7.86 us
o          8192 x  1024    16.8 MB   3.50 us
router      256 x  8192     8.4 MB   1.75 us
pooler     4096 x  8192    67.1 MB  13.98 us
unpooler   8192 x  4096    67.1 MB  13.98 us
one expert            28.3 MB  (50.3M params)
32 local experts     906.0 MB
total  790.9B   active  42.0B
KV per token per layer 256 B; window 2048 -> 0.52 MB, 80K full -> 20.5 MB per request
```

The microsecond column is bytes over 4.8 TB/s, the H200 SXM's HBM3e bandwidth on [NVIDIA's spec page](https://www.nvidia.com/en-us/data-center/h200/), and it reproduces the challenge's own floors for `muse_bf16_proj` (7.9, 3.5, 14.0 and 1.8 µs at one token). The parameter line is a sanity check, not an audit: the routed experts and the per-layer dense weights alone land on 42.0B active, matching the "A42B" in the checkpoint name `Muse-Spark-Open-816B-A42B-NVFP4`, and the remaining ~25B of the 816B (embeddings, the two layers without routed experts) is outside what the challenge publishes.

![Horizontal bars of MB read per GPU per layer: dense matrices from 8.4 to 67.1 MB, experts touched at batch 1 and 32, and the 906 MB local expert bank](/imgs/blogs/muse-spark-open-for-kernel-engineers-2.webp)

Notice two things in the chart. The pooler alone, 67.1 MB, is heavier than `qkv_gate` and `o` together. Because it is replicated, all eight GPUs read the full matrix every layer, every step, which makes it one of the largest dense reads in the model and a prime target for `muse_bf16_proj`. And the expert bank is 4.6 times all the dense matrices combined, but decode touches only a slice of it. The `muse_moe_w4a16` floors let us back out how much: 11.4 µs at batch 1 is 54.7 MB at 4.8 TB/s, about 2 experts; 70.8 µs at batch 32 is about 12 experts. The busiest GPU reads 6% of its bank at batch 1 and 37% at batch 32. That slice is where the step's time goes.

## 3. Attention: gated, normalized, and two kinds of layer

The attention computation, as the `muse_attention_decode` task states it:

```python
q, k, v, gate = split(qkv[t])        # the fused projection output
q = rmsnorm(q) * 5.473;  k = rmsnorm(k)
q, k = rope(q, p), rope(k, p)        # sliding layers only
cache[cache_loc[t]] = fp8(k), fp8(v)
a = softmax(q @ K.T / 8) @ V         # up to p: window or all
out = rmsnorm(a) * sigmoid(gate)
```

Every line has a kernel consequence.

**QK-norm without a gain.** Both q and k get a per-head RMSNorm with no learned weight, so cached keys have unit RMS per head. The query is then multiplied by a fixed 5.473 and the softmax scale is the usual ${1/\sqrt{64} = 1/8}$. Since both vectors have unit RMS over 64 dimensions, the largest possible logit is ${64 \times 5.473 / 8 \approx 43.8}$. That bounded logit range is what makes FP8 safe here, and the rules exploit it: the cache stays FP8 E4M3, and the query may be rounded to FP8 E4M3 (FA3's FP8 path folds the 5.473 into the softmax scale) but no coarser.

**Two layer kinds, one of them without RoPE.** Sliding layers use interleaved (GPT-J style) RoPE with theta 500,000 and a 2,048-token window; SGLang passes FA3 `window=(2047, 0)`. Full layers attend to the whole context and apply no RoPE at all. A kernel that branches on `config["window"]` is fine, since branching on config is legal inside a CUDA graph.

**GQA at 8:1 with head_dim 64.** Eight query heads share each KV head. With head_dim 64, every KV byte loaded feeds eight dot products of only 64 elements, which keeps arithmetic intensity low even when you batch the group.

![Left: 16 query heads bracketed into two groups of 8 over 2 KV heads. Right: two equal strips of an 80K context, sliding fills the last 2048 tokens, full fills all](/imgs/blogs/muse-spark-open-for-kernel-engineers-4.webp)

The strips are drawn to scale, and they are the whole story of the attention problem. The cache costs 256 bytes per token per layer on this GPU (2 KV heads × 64 dims × K and V × 1 byte). A sliding layer at 80K context reads the last 2,048 tokens, 0.52 MB per request. A full layer reads all 80,000, 20.5 MB per request, 39 times more. At batch 32 the full layer is moving about 655 MB of cache, 137 µs at 4.8 TB/s before any weights. That is why the `muse_decode_layer` floor jumps from 84.5 µs on a sliding layer to 208.1 µs on a full one at the same batch. Sliding layers are 46 of the 62, so the cheap kind is the common kind, and its time is dominated by everything except attention.

**The output gate.** After attention, each head's 64-wide output is RMS-normalized and multiplied elementwise by `sigmoid(gate)`, where `gate` is the last 1,024 columns of the fused projection. Cheap math, but the gate columns must survive until after attention, which matters if you fuse projection and attention.

## 4. The router: sigmoid, a bias that only picks, and a renormalize

![256 experts drawn as 8 blocks of 32 cells, one per GPU, with one token's 8 experts lit: 3 on GPU 0, 2 on GPU 6, one each on GPUs 1, 3, 4](/imgs/blogs/muse-spark-open-for-kernel-engineers-3.webp)

Routing is where a "faster" kernel most easily turns into a disqualified one, so read it precisely. From the reference:

1. Logits are `h @ router.T` with the router stored in **FP32**, `[256, 8192]`.
2. Scores are `sigmoid(0.17678 * logits)`. The multiplier is ${1/\sqrt{32}}$.
3. Selection takes the top 8 of `score + router_bias`. The bias is a load-balancing knob: it decides *who* is picked.
4. The weights use the **unbiased** scores of the 8 picked experts, renormalized to sum to 1 (epsilon 1e-15, route scale 1.0).

This is the pattern DeepSeek-V3 popularized as [auxiliary-loss-free balancing](/blog/machine-learning/frontier-models/deepseek-v3-fp8-mtp-loss-free-balancing), and it has a kernel trap: the bias must enter the comparison and leave the arithmetic. A fused routing kernel that adds the bias once and reuses the biased value for the weights produces plausible-looking, wrong outputs.

Precision is fenced too. The FP32 router may run as a BF16 hi plus lo pair with FP32 accumulation (SGLang does exactly that above 8 tokens, splitting `hi = W.bf16()` and `lo = (W - hi).bf16()`), and nothing coarser. Because near-ties are inevitable, the judge checks routing with tolerances instead of exact equality: every expert you pick must score within 0.05 of the reference's 8th best, every reference expert that beats the 9th by more than 0.03 (0.04 above 256 tokens) must be in your set, and this GPU must keep at least the reference's routes minus 8% and 4.

The figure's uneven columns are the other half of the story. Under EP8, experts 32g through 32g+31 live on GPU g, and the challenge's routing is deliberately skewed, "as in the real model; a few experts per layer take most tokens." The judge picks the router seed so that the timed GPU is the busiest one, because a step waits for its slowest rank. For background on why that imbalance is structural, see [MoE routing and the load-imbalance problem](/blog/machine-learning/inference-engineering/moe-inference-routing-expert-parallel-and-the-load-imbalance-problem).

## 5. The experts and their norms

The `muse_moe_w4a16` computation, per local route:

```python
h = x[t] @ dequant(w13[e]).T              # [8192]: gate | up
y = (silu(gate) * up) @ dequant(w2[e]).T  # [4096]
out[t] += g * rmsnorm(y * (1 + post_norm_w))
```

Before this loop, the pooler output gets a pre-expert RMSNorm with a `(1 + w)` gain. Inside it, two NVFP4 GEMMs with a SwiGLU between them. After it, the part that makes Muse different from most MoE models you have optimized: **every expert's output is normalized before it is weighted and summed.** The post-expert norm scales by `(1 + w)` first, then applies RMSNorm with epsilon 1e-8, per token, per expert.

That one line forbids the classic trick of accumulating expert outputs straight into a shared buffer from the down-projection's epilogue. Each (token, expert) row needs a full 4,096-wide reduction for its RMS before it can be scaled by its routing weight. SGLang handles it with a separate Triton kernel, `fused_post_expert_norm_weighted_sum`, that reads all 8 expert outputs back. A kernel that computes the row's sum of squares inside the w2 GEMM's epilogue, across the tiles that own that row, saves a full round trip of `T × 8 × 4096` values.

![NVFP4 block of 16 four-bit E2M1 cells plus one E4M3 scale byte and an FP32 tensor scale; below, one cached token as 4 cells of 64 FP8 bytes totaling 256 B](/imgs/blogs/muse-spark-open-for-kernel-engineers-7.webp)

The weights are NVFP4: E2M1 values (magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6), two codes per byte, one E4M3 scale per 16-element block, one FP32 scale per tensor, so `value = e2m1 × block_scale × global_scale`. That is 9 bytes per 16 weights, 0.5625 bytes per weight, 3.6 times smaller than BF16. The rule fence: NVFP4 weights stay NVFP4. You may repack them in `prepare()` (untimed, up to 300 s), compress losslessly, and dequantize inside the kernel, but you may not store them in another precision. Activations stay BF16, which is why the problem is called W4A16 and why SGLang's baseline is Marlin's `moe_wna16_marlin_gemm`. For what Hopper does and does not accelerate here, see [FP8 and FP4 inference](/blog/machine-learning/inference-engineering/fp8-and-fp4-inference-what-the-hardware-actually-gives-you): H200 has no FP4 tensor cores, so every FP4 weight is dequantized in registers before a BF16 MMA.

One more subtlety for the decode path: up to 8 tokens, the unpooler computes only this GPU's 1,024 output rows and the all-gather fills in the rest. The judge models the all-gather as the identity, so the other 7,168 columns are zeros. Above 8 tokens the full 67.1 MB matrix runs. That is why the unpooler's floor is 1.7 µs at one token and 14.1 µs at sixteen.

## 6. A residual stream that keeps its norm

The stream between sublayers is not the usual `s + branch`. The reference's `residual_coefficients` reads:

$$
w = \frac{g}{0.3},\quad \beta = \sigma(w),\quad \alpha = \sqrt{\max\big(\sigma(-w)\,(1+\beta),\ 10^{-3}\big)},\quad s' = \alpha\, s + \beta\, \mathrm{rmsnorm}(\text{branch})
$$

where `g` is a learned per-channel gate (8,192 values per sublayer). Expand $\alpha^2 + \beta^2 = (1-\beta)(1+\beta) + \beta^2 = 1$ whenever the clamp is inactive. If the stream and the normalized branch have unit RMS and are roughly orthogonal, the new stream keeps unit RMS. The architecture is norm-preserving by construction.

![Line chart of alpha falling and beta rising as the residual gate g goes from -1.5 to 1.5, crossing at g = 0.26 where both equal 0.71, with alpha squared plus beta squared flat at 1](/imgs/blogs/muse-spark-open-for-kernel-engineers-5.webp)

The curves cross at $g \approx 0.26$, where both coefficients equal ${1/\sqrt{2} \approx 0.71}$. For a kernel engineer the takeaway is the shape of the boundary, not the math. Every sublayer exit is the same fused chain: post-norm the branch, gate-mix into the FP32 stream, then pre-norm for the next consumer. Count the RMSNorms in one layer: q and k norms, the attention output norm, post-attention, pre-FFN, pre-expert, the per-expert post norm, post-FFN, and the next layer's pre-attention norm. Nine reductions, most of them over 8,192 or 4,096 elements. Each moves little data and costs a launch, which at batch 1 is the whole cost.

## 7. What SGLang runs today

![Kernel sequence of SGLang's stock layer: 18 main-stream launches in three rows from qkv_gate GEMM to the final fused norm, with the router on a side stream joining at moe_align_block_size](/imgs/blogs/muse-spark-open-for-kernel-engineers-6.webp)

The baseline you are timed against is SGLang's own `MuseSpark1xDecoderLayer.forward` at commit `a55ab7d5`, reproduced module by module in `baseline.py`. Read the figure as a launch count, not a timeline (block widths are not time). Some of it is already fused: `FusedQKNorm` folds RoPE into the q/k norm, `FusedRMSNormGate` does the output gate, and `FusedPostNormResidualPreNorm` collapses the post-norm, residual mix and next pre-norm into one JIT kernel at each boundary. The router runs on a side stream that overlaps the pooler GEMM.

What is not fused is just as informative:

| Seam in the stock layer | Why it is a seam | What a submission could do |
| --- | --- | --- |
| `qkv_gate` GEMM → QK-norm | the norm reads the GEMM output back from HBM | norm in the GEMM epilogue (16+2 heads of 64 fit a tile) |
| q to FP8 + `store_cache` → FA3 | two passes over the new tokens | write the cache from the norm kernel |
| Marlin w13 → `silu_and_mul` → Marlin w2 | intermediate `[routes, 8192]` round-trips | SwiGLU in the w13 epilogue |
| Marlin w2 → post-expert norm sum | per-expert outputs round-trip | per-row sum of squares in the w2 epilogue |
| `pooler` and `unpooler` via cuBLAS | generic skinny GEMMs at `m` from 1 to 256 | split-K or stream-K tuned for `m <= 32` |

These are hypotheses, not results. The [layer strip](/blog/machine-learning/gpu-kernel-engineering/one-layer-on-one-h200-where-the-microseconds-go) measures how much each seam is worth, and the [judge post](/blog/machine-learning/gpu-kernel-engineering/inside-the-judge-cuda-graph-timing-tolerances-and-reward-hacking) explains why every one of them must survive CUDA-graph capture, a cold L2 and fresh inputs on every replay.

## Reading the model as a checklist

What to keep in your head when you open `submission.py`:

| Property | Value per GPU | Kernel consequence |
| --- | --- | --- |
| Hidden / MoE width | 8,192 / 4,096 | pooler and unpooler are big, replicated BF16 GEMMs |
| Attention heads | 16 q, 2 KV, head_dim 64 | low intensity, GQA 8:1 |
| Layer mix | 46 sliding (2,048, RoPE) + 16 full (no RoPE) | full layers dominate cache bytes at batch 32 |
| KV cache | FP8 E4M3, 256 B/token/layer | queries may go FP8, nothing coarser |
| Router | FP32 `[256, 8192]`, sigmoid × 1/√32, selection bias | BF16 hi+lo allowed; tolerant top-8 check |
| Experts | 32 local of 256, top-8, NVFP4, SwiGLU | W4A16; norm each expert before summing |
| Residual | FP32, norm-preserving α/β mix | 9 RMSNorms per layer, fuse the boundaries |

Trust this map for anything shape-dependent: tile sizes, split factors, which kernel owns which norm. Do not trust it for time. Nothing here was measured; the bytes are floors, and turning them into microseconds is the layer strip's job.

## On the board

> **On the board.** *Moves:* all five kernel problems indirectly, by fixing their shapes in your head: `muse_bf16_proj` (the replicated 67.1 MB pooler and unpooler), `muse_attention_decode` (2 KV heads, sliding versus full), `muse_moe_w4a16` (per-expert post norm, ~2 to ~12 experts touched per decode step), and the two layer problems. *Lever:* fuse along the seams the model forces (QK-norm in the projection epilogue, SwiGLU and the post-expert norm in the Marlin epilogues, the residual boundaries), and spend effort where the bytes are: the expert slice and the pooler. *Rule fence:* NVFP4 stays NVFP4 and BF16 stays BF16 (lossless repacks only; the FP32 router may become BF16 hi+lo with FP32 accumulation), activations stay BF16 where SGLang keeps them, the KV cache stays FP8 and queries go no coarser than FP8 E4M3, the router keeps 8 experts with the bias used only for selection, every norm stays, and attention covers the whole 2,048-token window or the whole context.
