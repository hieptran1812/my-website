---
title: "Grouped GEMM: The Kernel That Makes Mixture-of-Experts Fast"
date: "2026-07-18"
publishDate: "2026-07-18"
description: "How one kernel turns the ragged, per-expert matmuls of a Mixture-of-Experts layer into a single launch — the intuition, the CUTLASS/cuBLAS/Triton implementations, and the persistent, cache-aware, TMA-driven tricks that buy 1.4–2.6× on H100."
tags:
  [
    "grouped-gemm",
    "mixture-of-experts",
    "moe",
    "triton",
    "cuda-kernels",
    "cublas",
    "cutlass",
    "gpu-performance",
    "deepseek",
    "model-serving",
    "tma",
    "llm",
  ]
category: "machine-learning"
subcategory: "Model Serving"
author: "Hiep Tran"
featured: true
readTime: 50
image: "/imgs/blogs/grouped-gemm-moe-kernel-1.webp"
---

Open the profiler on a Mixture-of-Experts layer and you will find something that offends everything you know about GPU efficiency. The layer does *less* arithmetic than the dense feed-forward it replaced — a token touches 8 experts out of 256, not the whole network — and yet on a naive implementation it runs *slower per token*. The tensor cores that should be pegged at 90% utilization sit at 30%. The trace is a picket fence: hundreds of tiny matmul kernels, each a few microseconds long, separated by gaps of idle silicon where the GPU is doing nothing but waiting for the next launch.

The arithmetic is cheap. The *shape* of the arithmetic is the problem. A dense FFN is one big, fat, beautiful matrix multiply — the exact workload a GPU was built for. An MoE FFN is a pile of small, uneven matrix multiplies, one per expert, with a different number of rows each, and their sizes are not known until the router has run. Feed that to PyTorch the obvious way — a Python `for` loop over experts — and you launch one kernel per expert, pay the launch latency every time, and leave the machine starved between launches. The grouped GEMM kernel is the fix: it takes that whole pile of ragged matmuls and computes them in **one launch**, keeping every streaming multiprocessor fed from start to finish.

This post is a tour of that kernel — what it is, why MoE needs it specifically, and the three families of implementation (cuBLAS's drop-in API, CUTLASS's device-side problem visitor, and the Triton persistent kernel PyTorch shipped for DeepSeek-V3). We will build the intuition first, in pictures, then go all the way down to the persistent-kernel, cache-aware, device-side-TMA tricks that turn "one launch" into a 1.4–2.6× end-to-end speedup on an H100. If you have not yet internalized *why* MoE serving is memory-bound and strange, read [serving Mixture-of-Experts models at scale](/blog/machine-learning/model-serving/serving-moe-models-at-scale) first — this post is the kernel-level companion to it, and assumes you know what a router and an expert are.

## The mental model: one router, many ragged matmuls, one launch

![A graph showing a token batch feeding a router that fans out to four experts of unequal token counts, which fan into a single grouped GEMM producing the layer output](/imgs/blogs/grouped-gemm-moe-kernel-1.webp)

The diagram above is the mental model, and the whole post is a tour of it. A batch of tokens hits the router, a small learned layer that picks a few experts for each token (top-2 of 4 in the toy figure; top-8 of 256 in DeepSeek-V3). Routing scatters the batch: 112 tokens land on expert 0, 40 on expert 1, only 8 on expert 2, 96 on expert 3. Each expert is a matrix — its FFN weight — and "run expert *e*" means "multiply that expert's tokens by that expert's weight." So one MoE layer is not one matmul; it is *four independent matmuls with four different row-counts*. The grouped GEMM is the green box: it swallows all four, in one kernel launch, and hands back all four results, which the layer then combines with the router's gating weights.

Hold onto three facts from that picture, because every technique below is in service of one of them:

1. **The matmuls are independent.** Expert 0's output does not depend on expert 3's. There is nothing to serialize; the only reason to run them one-at-a-time is that we did not know how to run them together.
2. **The row-counts are ragged and data-dependent.** 112, 40, 8, 96 — different every batch, unknown until the router fires. This is the single fact that rules out the batched-GEMM primitives you already have.
3. **The weights are the expensive operand.** Each expert weight is large (for DeepSeek-V3, roughly ${44 \times 10^6}$ parameters per expert per FFN matrix) and gets re-read for every tile of output. How you schedule the reads against the cache decides whether you are compute-bound or bandwidth-bound.

Let us take the naive approach first, see exactly where it bleeds, and then earn each optimization.

### Why grouped GEMM is different from what your intuition expects

Before the mechanics, a table — because the assumptions that make dense-model kernels fast are precisely the ones MoE violates.

| Your assumption (from dense models) | The naive MoE view | The reality grouped GEMM exploits |
| --- | --- | --- |
| One layer is one matmul | One layer is E matmuls in a loop | One layer is E matmuls in **one launch** |
| Bigger matmul = better utilization | Small per-expert matmuls waste the machine | Fusing small matmuls recovers utilization |
| Problem shapes are known at compile time | Shapes depend on data, so recompile per batch | Shapes are passed as **runtime data** (an offsets array), no recompile |
| Kernel launch cost is negligible | E launches per layer × L layers = thousands | 1 launch per layer amortizes the cost E-fold |
| The weights fit in cache and stay hot | Each expert reloads its weight from HBM | Tile ordering keeps a weight band hot in L2 |
| More FLOPs is the cost | Padding to tile boundaries adds wasted FLOPs | You trade a little wasted compute for one clean launch |

Every row of the right-hand column is a section below. The recurring theme of this series — the tension between latency, throughput, and cost — shows up here as a single question: *how do you keep the tensor cores busy when the work arrives in small, uneven, unpredictable pieces?*

## 1. The problem: what routing does to your matmul

Start from the dense baseline so the damage is legible. A dense FFN takes the batch's hidden states ${X \in \mathbb{R}^{T \times H}}$ (T tokens, hidden size H), multiplies by a single weight ${W \in \mathbb{R}^{H \times F}}$, applies a nonlinearity, multiplies by a second weight, and returns. It is two matmuls, each of shape ${T \times H \times F}$. With T in the thousands and H, F in the thousands, those are large, well-shaped GEMMs. The GPU eats them at near-peak throughput because there is enough arithmetic per byte loaded to hide memory latency, and enough tiles of output to keep all 132 SMs of an H100 busy for many waves.

MoE replaces that single ${W}$ with E expert weights ${W_0, \dots, W_{E-1}}$, and routing sends each token to a subset of them. Group the tokens by expert and the layer becomes: for each expert *e*, take the ${T_e}$ tokens routed to it, and compute ${X_e W_e}$. The total arithmetic is *smaller* than the dense layer (each token flows through a few experts, not the whole width), but it is chopped into E pieces of size ${T_e \times H \times F}$, and the ${T_e}$ are wildly uneven.

The obvious PyTorch implementation writes exactly what the math says — a loop:

```python
import torch
import torch.nn.functional as F

def moe_ffn_loop(x, w1, w2, expert_ids, num_experts):
    # x:          [T, H]            hidden states for the batch
    # w1:         [E, H, F]         gate/up projection, one per expert
    # w2:         [E, F, H]         down projection, one per expert
    # expert_ids: [T]               which expert each token was routed to
    out = torch.zeros_like(x)
    for e in range(num_experts):
        mask = expert_ids == e
        if not mask.any():
            continue
        xe = x[mask]                 # gather this expert's tokens -> [T_e, H]
        h  = torch.matmul(xe, w1[e]) # GEMM launch #1  (T_e x H x F)
        h  = F.silu(h)               # elementwise launch
        ye = torch.matmul(h, w2[e])  # GEMM launch #2  (T_e x F x H)
        out[mask] = ye               # scatter back
    return out
```

This is correct, readable, and slow. Count the kernel launches: two matmuls, one activation, one gather, one scatter — call it five kernels — *per expert*, times E experts, times L transformer layers. For DeepSeek-V3's 256 experts across 58 MoE layers, a single forward pass launches on the order of tens of thousands of kernels. Each launch costs a few microseconds of CPU-to-GPU handoff, and at small ${T_e}$ the kernel itself runs for only a few microseconds, so you spend as much time *starting* work as *doing* it. This is the launch-bound regime, the same one that motivates [CUDA graphs and torch.compile in the decode path](/blog/machine-learning/model-serving/kernel-fusion-cuda-graphs-torch-compile) — except here the culprit is not autoregression, it is the expert loop.

Put numbers on it, because the arithmetic is damning. Take a modest MoE with 64 experts across 32 layers and a batch that routes, on average, 96 tokens to each expert. The two matmuls per expert are the only real work; at ${T_e = 96}$, ${H = 4096}$, ${F = 14336}$ each is roughly ${2 \times 96 \times 4096 \times 14336 \approx 11.3}$ GFLOP, which an H100 at ~600 BF16 TFLOP/s runs in about 19 µs — if it were running at peak, which at 96 rows it is not, because 96 rows barely fills one row-tile. Now the overhead: five kernel launches per expert at ~5 µs of dispatch each is 25 µs of pure launch cost *per expert*, before the GPU computes anything. Multiply by 64 experts and 32 layers and the launch overhead alone is ${25 \times 64 \times 32 \approx 51}$ ms per forward pass — time the GPU spends idle, waiting on the CPU to hand it the next scrap of work. The grouped kernel replaces those 320 launches per layer with one, turning 51 ms of dispatch into well under a millisecond. That is the gap the rest of this post closes.

![A before/after comparison: the per-expert loop pays eight kernel launches and eight tail-wave bubbles with idle SMs, while one grouped GEMM pays a single launch and single tail wave for 1.4 to 2.6 times the throughput](/imgs/blogs/grouped-gemm-moe-kernel-2.webp)

The figure names the two costs the loop pays that the grouped kernel pays only once. The first is **launch latency** — the CPU-side cost of dispatching a kernel, roughly 5 µs of overhead that the GPU cannot hide when the kernel itself is short. Eight experts, eight launches, and if each GEMM runs for 3 µs, you have spent more wall-clock launching than computing. The second, subtler cost is the **tail wave**. A GPU runs a kernel in *waves*: it fills all SMs with tiles, they finish roughly together, and if the number of tiles is not a clean multiple of the SM count, the final wave is half-empty — some SMs work, the rest idle until the kernel ends. A small per-expert GEMM might produce only a handful of tiles, so its *entire* execution is one ragged, half-empty wave. Loop over eight experts and you have eight half-empty tail waves back to back, each one leaving most of the machine dark.

> The naive MoE loop does not fail because MoE is expensive. It fails because it turns one fat, cache-friendly, SM-saturating matmul into a stutter of tiny ones, and a GPU hates nothing more than a stutter.

Grouped GEMM collapses the stutter. All E matmuls become one kernel launch, the tiles from *all* experts are scheduled together so the machine stays full, and there is exactly one tail wave for the whole layer instead of one per expert. That is the entire value proposition in one sentence — the rest of the post is how you actually build a kernel that does it, and why the naive fusions you might reach for first do not work.

### Second-order: the loop also destroys overlap and cache reuse

There is a third cost that the picture cannot show. Because each expert's matmul is a separate kernel with a barrier between iterations (the mask, the gather, the scatter), the loop cannot overlap expert *e*'s compute with expert ${e{+}1}$'s memory loads. In a fused kernel, while one SM finishes the arithmetic for one tile, the memory subsystem is already prefetching the operands for the next — the classic latency-hiding that makes GEMM fast. The loop throws that away at every iteration boundary. And because each expert weight is loaded fresh from HBM by its own kernel, there is zero reuse of anything across experts, even though consecutive experts' weights may sit adjacent in memory. Hold these two — overlap and reuse — in mind; the persistent kernel and the cache-aware ordering below are precisely the machinery to win them back.

## 2. Why you can't just use batched GEMM

The instinct of anyone who has written CUDA is: "I don't need a new kernel, I need *batched* GEMM." cuBLAS has shipped `cublasGemmBatchedEx` and `cublasGemmStridedBatchedEx` for years, and they do exactly what the name says — run many matmuls in one launch. Reach for them and you hit a wall on the second line of the docs.

![A matrix comparing batched, strided-batched, and grouped GEMM across per-group shapes, memory layout, single-launch, and MoE fit — only grouped GEMM allows M to differ per group and fits MoE](/imgs/blogs/grouped-gemm-moe-kernel-3.webp)

Batched GEMM requires **every problem in the batch to have identical dimensions**. All M, all N, all K the same; only the data pointers differ. That is a perfect fit for, say, multi-head attention, where every head is the same shape. It is a total non-fit for MoE, where the whole point is that ${T_e}$ — the M dimension — is different for every expert. Strided-batched GEMM is even more rigid: it assumes the problems are not just the same shape but laid out at a *fixed stride* in memory, so it can address them with pure arithmetic instead of a pointer array. Same fatal constraint: uniform shapes only.

The figure lays out the three families side by side. Batched and strided-batched both answer "yes" to *one launch* — they will happily run 256 matmuls in a single kernel — but both answer "no (ragged M)" to *fits MoE*, because you cannot express "112 rows here, 8 rows there" in an API that demands one M for the whole batch. Grouped GEMM is the family that answers "yes" to both. Its defining feature is exactly the one the other two lack: **each problem in the group carries its own M, N, K.** In practice, for MoE, N and K are shared (all experts have the same hidden and FFN dimensions) and only M varies, but the generality is there.

Mechanically, grouped GEMM replaces the "one shape for the whole batch" assumption with a small **descriptor array**: a list of per-group problem sizes and a list of per-group data offsets. The kernel reads that array at runtime to figure out which rows belong to which expert and which weight to multiply them by. That is the crux, and it is worth saying slowly because it is the thing that makes grouped GEMM *possible* on a GPU at all: **the shapes are data, not code.** You do not recompile the kernel when the router's decisions change from batch to batch; you hand the same compiled kernel a different offsets array. A dense matmul bakes M, N, K into the launch configuration; a grouped GEMM reads them from a buffer the router just wrote.

### The workaround that isn't: pad-to-max batched GEMM

Before grouped GEMM was widely available, teams faked it with batched GEMM plus padding: pick the largest ${T_e}$ in the batch, pad every expert up to that size with zero rows, and run a uniform batched GEMM of shape ${E \times T_{\max} \times H \times F}$. It works, and it is one launch. It is also catastrophically wasteful when the load is skewed, which — because routers are never perfectly balanced — it always is. If one hot expert draws 512 tokens and a cold one draws 8, you pad the cold expert to 512 and do ${64\times}$ the necessary arithmetic on it, and you do that for every under-loaded expert. Pad-to-max turns a load-balancing problem into a compute-multiplication problem. Grouped GEMM's whole reason to exist is to handle the ragged shapes *without* padding to the max — it pads only to the tile boundary, which as we will see in §8 costs a few percent, not a few hundred.

## 3. Plumbing: permute, group offsets, and unpermute

Here is a subtlety that trips up everyone the first time. The router does not hand you tokens neatly sorted by expert. It hands you a batch in *token order*, and a parallel array saying "token 0 → expert 2, token 1 → expert 0, token 2 → expert 3, …". But a grouped GEMM wants each expert's rows *contiguous* in memory, so that "expert *e*'s tokens" is a simple slice `[offset_e : offset_e + m_e]` it can point a tile at. Bridging those two representations is the permute/unpermute dance, and it is as important to MoE performance as the matmul itself.

![A grid showing scattered routed tokens in arrival order being stable-sorted by permute into contiguous per-expert bands, with an m_sizes array of per-expert row counts and offsets](/imgs/blogs/grouped-gemm-moe-kernel-4.webp)

The figure walks the transformation. The top row is the batch as the router produced it — token order, expert ids interleaved: E2, E1, E0, E2, E1, E2. `permute()` performs a **stable sort by expert id**, gathering all of expert 0's tokens, then all of expert 1's, then expert 2's, into one contiguous buffer: E0, E1, E1, E2, E2, E2. Now every expert owns a solid band of rows. The only extra bookkeeping you need is the bottom bar: `m_sizes = [1, 2, 3]`, the number of tokens per expert, and its prefix-sum `offsets = [0, 1, 3]`, the starting row of each band. That tiny array *is* the grouped GEMM's problem descriptor. The kernel does not need to understand routing; it needs `m_sizes` and a pointer, and it can find every group.

The [`grouped_gemm`](https://github.com/fanshiqing/grouped_gemm) library (the CUTLASS-backed one that Megatron-Core builds on) exposes exactly these three primitives, and the end-to-end MoE FFN reads almost like the math:

```python
from grouped_gemm import ops
import torch.nn.functional as F

# indices: [T, top_k] int32, the expert(s) each token was routed to
# w1:      [E, H, F]  gate/up projection weights, one per expert
# w2:      [E, F, H]  down projection weights, one per expert
# probs:   [T, top_k] the router's gating weights for the weighted combine

# 1. gather: sort tokens into contiguous per-expert bands
permuted_x, row_id_map = ops.permute(x, indices)

# 2. build the problem descriptor: rows per expert (int64, on CPU)
m_sizes = torch.bincount(indices.flatten(), minlength=E).cpu()

# 3. two grouped GEMMs with the activation between them
h = ops.gmm(permuted_x, w1, m_sizes, trans_b=False)   # [sum(m), F]
h = F.silu(h)
y = ops.gmm(h, w2, m_sizes, trans_b=False)             # [sum(m), H]

# 4. scatter back to token order and apply the gating combine
out = ops.unpermute(y, row_id_map, probs)              # [T, H]
```

Two things about that snippet earn their own paragraph. First, `m_sizes` lives **on the CPU** and is `int64`. That is not an accident — the kernel launch needs the group boundaries to configure its work, and in the CUTLASS grouped path those sizes are read host-side to build the schedule. (The Triton persistent kernel in §7 reads them device-side instead, which is one of the things that makes it special.) Second, `unpermute()` folds two operations into one: it scatters the rows back to their original token positions *and* multiplies by the gating weights `probs`, doing the router's weighted combine as part of the un-sort. Fusing the combine into the scatter saves a full read-modify-write pass over the activation, which at MoE batch sizes is not nothing.

### The `gmm` signature, precisely

The core call is `ops.gmm(a, b, batch_sizes, trans_b)`, and getting its shapes right is where people lose an afternoon:

- **`a`** is `[total_tokens, K]` — *all* the permuted tokens stacked, every expert's band concatenated. Not `[E, ...]`; the expert boundaries live only in `batch_sizes`.
- **`b`** is `[E, K, N]` — a genuine 3-D stack, one weight matrix per expert. This is the one operand that *is* indexed by expert.
- **`batch_sizes`** is `[E]`, `int64`, **on CPU**, and must sum to `total_tokens`. Element *e* is how many rows of `a` belong to expert *e*.
- **`trans_b`** transposes each expert's weight; you set it `True` when your weight is stored `[E, N, K]` instead of `[E, K, N]`.

The mental compression: `a` is one tall stack of tokens, `b` is a shelf of per-expert weights, and `batch_sizes` is the ruler that tells the kernel where each expert's slice of the stack begins and ends. That is the entire interface, and it is the same conceptual interface whether the backend is CUTLASS, cuBLAS, or Triton — only the internals differ.

### Second-order: permutation is not free, so fuse it

The gather and scatter are memory-bandwidth operations over the full activation, and at large batch they can rival the matmul in wall-clock if you implement them as separate PyTorch ops. This is why production stacks *fuse* them. Megatron-Core, for instance, treats permutation fusion as a first-class optimization alongside the grouped GEMM itself — the permute is folded into the epilogue of the preceding op or the prologue of the GEMM so the tokens are reordered on the fly rather than in a standalone kernel. The lesson generalizes: in MoE, the data movement around the matmul is a peer of the matmul, not an afterthought. If you profile a grouped-GEMM MoE and the GEMM looks fast but the layer is slow, look at the permute.

### The backward pass is grouped too

Everything above is the forward pass, but MoE is usually a *training* workload, and the backward pass doubles the grouped-GEMM count in a way worth understanding — because it is where most of the FLOPs and most of the subtle bugs live. A GEMM ${Y = X W}$ has two gradients. The **input gradient** ${\partial X = \partial Y \, W^\top}$ is itself a grouped GEMM: each expert's output-gradient band multiplied by that expert's transposed weight, exactly the same ragged shape as the forward, so you call `gmm` with `trans_b=True`. The **weight gradient** ${\partial W = X^\top \, \partial Y}$ is grouped in a *different* way: for each expert it contracts that expert's input band against its output-gradient band to produce a per-expert weight update, so the reduction dimension is the (ragged) token count and the output is a fixed-size ${H \times F}$ per expert. That second one is the tricky case, because the contraction dimension varies per group while the output shape does not — the grouped-GEMM library has to handle "same output shape, different K per group," which is the transpose of the forward's "same K, different M per group."

The practical consequences: a single MoE layer runs *three* grouped GEMMs per forward-backward step (one forward, two backward per weight matrix, so six total across `w1` and `w2`), and all three must agree on the *same* `m_sizes` and the *same* permutation, or the gradients land on the wrong tokens. This is why the `row_id_map` returned by `permute()` is saved for the backward pass — `unpermute`'s gradient re-permutes using the identical map so a token's gradient flows back to exactly the row it came from. Get the map out of sync between forward and backward and the model trains, slowly diverges, and gives you no stack trace — the exact failure mode of case study #8, one level up.

### Why the sizes live on the CPU as int64

One detail from the `gmm` signature deserves its own paragraph because it surprises people and occasionally deadlocks them: `batch_sizes` is `int64` and on the **CPU**. The reason is that, in the CUTLASS grouped path, the host needs the group boundaries to configure the kernel's grid and problem visitor *before* the launch — the launch configuration is a host-side decision, so the sizes must be readable on the host. That creates a hazard: `m_sizes` is derived from the router's output, which lives on the GPU, so computing it (`torch.bincount(...).cpu()`) forces a **device-to-host copy**, which is a synchronization point that stalls the pipeline until the GPU has finished routing. Naively placing that `.cpu()` call blocks the CPU from launching *anything* until routing completes, serializing what could overlap. Production code hides the sync — precompute the counts a step ahead, use pinned memory for the copy, or (as the Triton persistent kernel does) read the sizes *device-side* so no host round-trip happens at all. The device-side approach is one more reason the persistent kernel is serving-friendly: it never asks the CPU what the router decided.

## 4. Grouped GEMM, three ways

There is not one grouped GEMM kernel; there are three families, and they occupy different points on the tradeoff between "drop it in and move on" and "extract the last 20% on this specific GPU." Choosing among them is a real engineering decision, not a detail.

![A matrix comparing three grouped-GEMM implementations — cuBLAS grouped API, CUTLASS grouped, and Triton persistent — across ease of use, precision coverage, peak throughput, and where each shines](/imgs/blogs/grouped-gemm-moe-kernel-5.webp)

| Implementation | How you call it | Precision | Where it wins |
| --- | --- | --- | --- |
| **cuBLAS grouped** | `cublasGemmGroupedBatchedEx` | FP16, BF16, FP32/TF32, FP64 | Drop-in; few groups; you want NVIDIA to own the tuning |
| **CUTLASS grouped** | `grouped_gemm.ops.gmm` (PyTorch binding) | FP16, BF16, FP8 | Many experts; fuses all groups into one kernel; the training default |
| **Triton persistent** | a `@triton.jit` kernel you can read and edit | BF16 (FP8 in progress) | H100-class hardware; you want to hack the schedule and use TMA |

**cuBLAS grouped GEMM** is the newest and the easiest. NVIDIA added a true grouped API in cuBLAS 12.5 — `cublasGemmGroupedBatchedEx` for the half/single/double-precision `Ex` path, and `cublasXgemmGroupedBatched` for the fixed-precision variants. It handles per-group shapes, transpositions, and scaling in one launch, and NVIDIA owns the autotuning so you do not. In pseudocode the call is a batched call with *arrays* where the old API took scalars:

```c
// cuBLAS 12.5+ grouped GEMM: each group carries its own shape.
cublasGemmGroupedBatchedEx(
    handle,
    transa_array, transb_array,        // per-group transpose flags
    m_array, n_array, k_array,         // per-group problem sizes (host arrays)
    alpha_array,
    A_array, CUDA_R_16BF, lda_array,   // per-group A pointers + leading dims
    B_array, CUDA_R_16BF, ldb_array,   // per-group B pointers + leading dims
    beta_array,
    C_array, CUDA_R_16BF, ldc_array,   // per-group C pointers + leading dims
    group_count,                       // number of independent GEMMs
    CUBLAS_COMPUTE_32F);               // accumulate in FP32
```

NVIDIA reports roughly a **1.2× speedup over naive looping with the batched-GEMM API** for MoE-shaped workloads at group counts of 8 and 64 in FP16 — and, tellingly, they got that speedup using only *warp-level* MMA instructions, competitive with kernels that lean on the newer warp-*group* MMA. The number is modest compared to what the Triton kernel gets end-to-end, because 1.2× is measured against an already-batched baseline, not against the Python loop. If your baseline is the loop, the win is much larger; if it is batched GEMM with padding, 1.2× is the marginal gain from dropping the padding.

**CUTLASS grouped GEMM** is the workhorse of MoE *training*. This is what [`fanshiqing/grouped_gemm`](https://github.com/fanshiqing/grouped_gemm) wraps and what Megatron-Core reaches for when the expert count is large. CUTLASS fuses *all* the groups into a single kernel using a device-side "problem visitor": the kernel walks the `m_sizes` array on the GPU, and each threadblock figures out which group and which tile it owns by binary-searching the offsets. Because everything is one kernel, there is no per-group launch and the scheduler can balance tiles across all groups at once — which is why CUTLASS pulls ahead of cuBLAS precisely when the number of GEMMs is large. It also carries FP8 support, which matters for anyone training in low precision. If you are training an MoE with Megatron-Core today, you are almost certainly running this kernel whether you know it or not.

**The Triton persistent kernel** is the newest and the most interesting, and it is the subject of the rest of this post. PyTorch's team wrote a BF16 grouped GEMM in Triton specifically for DeepSeek-V3-class models and reported up to **2.62× over the PyTorch loop** on an H100. Its advantage is not that Triton is magic — it is that the kernel is *readable and editable*, so its authors could layer in three Hopper-specific tricks (persistent scheduling, cache-aware tile ordering, and device-side TMA descriptors) that the black-box libraries either do not expose or had not shipped. It is the implementation you study when you want to understand *why* grouped GEMM is fast, and the one you fork when you need a schedule the libraries do not offer. For the general case of when to write your own kernel at all, see [custom CUDA kernels for inference](/blog/machine-learning/model-serving/custom-cuda-kernels-for-inference); grouped GEMM is a textbook example of the "the library shape does not fit my problem" trigger.

### How to choose

The decision rule is short. If you are doing inference and want zero fuss, use cuBLAS grouped or whatever your serving engine ([vLLM](/blog/machine-learning/model-serving/vllm-distributed-architecture-anatomy), SGLang) already wires up. If you are training an MoE at scale with many experts, you are on the CUTLASS path via your framework, and the thing to tune is the permutation fusion around it, not the GEMM. If you are chasing peak on H100/H200, need a schedule the libraries do not offer, or are prototyping FP8/MXFP4 support ahead of the libraries, fork the Triton kernel. And if you are just trying to understand the machine — read the Triton kernel regardless, because it is the only one whose source explains itself.

## 5. Inside the Triton persistent kernel

Everything so far is about *interface* — how you express "many ragged matmuls, one launch." Now we go inside the kernel that executes it fastest, and the three ideas that make it fast. They are independent wins that stack: a persistent grid to kill wave quantization, a cache-aware tile order to keep weights hot in L2, and device-side TMA descriptors to load the right expert's weight without recompiling. Each one is a general GPU technique that happens to pay off spectacularly on the grouped-GEMM shape.

### 5a. Persistent kernels and the wave-quantization tax

Recall the tail wave from §1: a GPU runs a kernel in waves of tiles, and the last wave is usually half-empty, idling SMs. For a normal large GEMM this is a rounding error — one ragged wave out of dozens. For grouped GEMM it is a disaster, because *every group is small enough to be mostly tail*. A cold expert with 8 tokens might produce a single row-tile; its entire contribution to the kernel is one half-empty wave. Sum that over experts and over layers and wave quantization becomes the dominant inefficiency.

![A before/after comparison: non-persistent scheduling launches tiles in waves and idles SMs in the half-empty tail, while a persistent grid of one program per SM sweeps every tile in a single wave with no relaunch](/imgs/blogs/grouped-gemm-moe-kernel-6.webp)

The **persistent kernel** design dissolves the problem by inverting the relationship between programs and tiles. Normally you launch one program (one CTA, one threadblock) per output tile and let the hardware scheduler stream them onto SMs in waves. Instead, you launch *exactly as many programs as there are SMs* — 132 on an H100 — and each program stays resident and loops over its share of tiles until every tile is done. There is no second wave, because there was only ever one grid, sized to the machine. The tiles are distributed across the persistent programs, so the last tiles finish nearly together and the tail idle shrinks to almost nothing.

In Triton it is startlingly little code. On the host you size the grid to the SM count:

```python
# host: one program per streaming multiprocessor
NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count  # 132 on H100
grid = (NUM_SMS, 1, 1)
_grouped_gemm_persistent[grid](
    a_ptr, b_ptr, c_ptr, m_sizes_ptr,
    N, K, NUM_EXPERTS,
    BLOCK_M=128, BLOCK_N=256, BLOCK_K=64, GROUP_SIZE_M=8,
    NUM_SMS=NUM_SMS,
)
```

and inside the kernel, the outer loop is a strided range over all tiles, so each of the 132 programs picks up tile `start_pid`, then `start_pid + 132`, then `start_pid + 264`, and so on until the tiles run out:

```python
@triton.jit
def _grouped_gemm_persistent(a_ptr, b_ptr, c_ptr, m_sizes_ptr, N, K, NUM_EXPERTS,
                             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                             BLOCK_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
                             NUM_SMS: tl.constexpr):
    start_pid = tl.program_id(0)                 # 0 .. NUM_SMS-1, fixed for this program
    num_tiles = compute_total_padded_tiles(m_sizes_ptr, NUM_EXPERTS, N, BLOCK_M, BLOCK_N)
    # each resident program strides through the whole tile space
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS):
        expert, m_tile, n_tile = map_tile_grouped(tile_id, m_sizes_ptr, ...)
        # ... load A[m_tile], load W[expert][:, n_tile], accumulate, store C ...
```

The `tl.range(start_pid, num_tiles, NUM_SMS)` is the whole trick: it keeps every program alive, feeding it new tiles until the grouped GEMM is complete, so the kernel is one continuous single-wave sweep instead of a sequence of relaunches. The measured effect of persistence-plus-scheduling — combined with the next two tricks — is up to **1.5× over the baseline (non-persistent) Triton kernel**, before you even compare against the Python loop.

A worked example makes the tail concrete. Suppose the grouped GEMM's padded output has 400 tiles and the H100 has 132 SMs. A non-persistent launch runs this in ${\lceil 400 / 132 \rceil = 4}$ waves: three full waves of 132 tiles, then a final wave of only ${400 - 396 = 4}$ tiles. That last wave occupies 4 SMs and idles 128 — the machine runs at 3% utilization for the entire duration of the tail wave, and because a wave takes as long as its slowest tile, that tail costs a full tile's worth of wall-clock at 3% efficiency. Across many small groups the effect compounds: each group contributes its own ragged remainder. The persistent grid erases the discretization entirely — 132 programs each take ${\lceil 400 / 132 \rceil = 4}$ tiles (a few take 3), they all finish within one tile of each other, and there is no distinct "tail wave" to idle in. The 400 tiles are simply dealt out like cards to 132 hands. Wave quantization is, at bottom, the cost of rounding your tile count up to a multiple of your SM count; persistence stops rounding.

There is a real cost to be honest about: a persistent kernel is harder to write and reason about, because you are now hand-managing the tile-to-program mapping that the hardware used to do for you. The `map_tile_grouped` helper has to translate a flat `tile_id` into "which expert, which row-tile within that expert, which column-tile," which means walking the padded `m_sizes` offsets — the same offsets from §3, now consumed on the device. Get that mapping wrong and you silently compute the wrong expert's output. This is exactly the kind of code you validate against the simple loop (which is why the PyTorch team kept the loss-curve comparison in their writeup — a numerical regression here is a scheduling bug, not a math bug).

Walk the mapping concretely, because it is the heart of the kernel and it is the same idea CUTLASS calls a "problem visitor." Say the padded per-expert tile counts (row-tiles × column-tiles) are `[8, 4, 12, 6]` for four experts, giving cumulative offsets `[0, 8, 12, 24, 30]` and 30 total tiles. A program holding `tile_id = 15` needs to know which expert owns tile 15. It searches the cumulative offsets: 15 falls in `[12, 24)`, so it belongs to expert 2, and its local index within expert 2 is `15 - 12 = 3`. From that local index and expert 2's row-tile count it recovers the row-tile and column-tile within the expert's own output block, and from the expert id it recovers the weight base pointer `b_ptr + 2 * N * K`. That is the whole translation: a search into a small offsets array plus two integer divisions. Every one of the 132 persistent programs runs it for every `tile_id` it is dealt, and because the offsets array is tiny and hot, the search is nearly free. The elegance is that the *only* thing distinguishing a grouped GEMM from a plain one is this per-tile lookup — swap `map_tile_grouped` for "expert 0 always" and you have an ordinary persistent GEMM. All of MoE's raggedness is compressed into one binary search over an offsets array.

### 5b. Cache-aware tile ordering: keep the weight band hot in L2

Persistence fixed *when* tiles run. This trick fixes *in what order*, and it is the one with the cleanest intuition and the most surprising payoff. The insight is about the L2 cache and which operand gets reused.

In a matmul ${C = A \times B}$, output tile ${C_{i,j}}$ needs row-band *i* of A and column-tile *j* of B. If you compute tiles in the obvious row-major order — ${C_{0,0}, C_{0,1}, C_{0,2}, \dots}$ across a full row before dropping to the next — then between ${C_{0,0}}$ and ${C_{1,0}}$ you have touched *every* column-tile of B. By the time you come back to column-tile 0 for ${C_{1,0}}$, it has been evicted from L2 by all the other B tiles you streamed through in between. You reload it from HBM. You reload *all* of them, every row. B is the expensive operand — for grouped GEMM it is the expert weight — so this is the difference between bandwidth-bound and compute-bound.

<figure class="blog-anim">
<svg viewBox="0 0 760 410" role="img" aria-label="Two tile-traversal orders side by side: linear row-major reloads a different B weight tile every step and misses L2, while grouped band order keeps one B tile resident and hits L2." style="width:100%;height:auto;max-width:820px">
<style>
.gg-tile{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.gg-lbl{font:600 13px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.gg-ttl{font:700 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.gg-blue{fill:var(--accent,#6366f1);opacity:.36;stroke:var(--accent,#6366f1);stroke-width:2}
.gg-amber{fill:#f59e0b;opacity:.34;stroke:#f59e0b;stroke-width:2}
.gg-green{fill:#22c55e;opacity:.34;stroke:#22c55e;stroke-width:2}
.gg-miss{font:600 13px ui-sans-serif,system-ui;fill:#d97706;text-anchor:middle}
.gg-hit{font:600 13px ui-sans-serif,system-ui;fill:#16a34a;text-anchor:middle}
@keyframes gg-sweepx{0%{transform:translateX(0);opacity:1}82%{transform:translateX(240px);opacity:1}88%{transform:translateX(240px);opacity:0}99%{transform:translateX(0);opacity:0}100%{transform:translateX(0);opacity:1}}
@keyframes gg-sweepy{0%{transform:translateY(0);opacity:1}82%{transform:translateY(180px);opacity:1}88%{transform:translateY(180px);opacity:0}99%{transform:translateY(0);opacity:0}100%{transform:translateY(0);opacity:1}}
@keyframes gg-pulse{0%,100%{opacity:.24}50%{opacity:.46}}
.gg-mx{animation:gg-sweepx 9s linear infinite}
.gg-my{animation:gg-sweepy 9s linear infinite}
.gg-pl{animation:gg-pulse 9s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.gg-mx,.gg-my{animation:none;opacity:.36}.gg-pl{animation:none;opacity:.34}}
</style>
<text class="gg-ttl" x="190" y="30">linear (row-major) order</text>
<rect class="gg-tile" x="40"  y="52" width="60" height="40" rx="6"/>
<rect class="gg-tile" x="120" y="52" width="60" height="40" rx="6"/>
<rect class="gg-tile" x="200" y="52" width="60" height="40" rx="6"/>
<rect class="gg-tile" x="280" y="52" width="60" height="40" rx="6"/>
<text class="gg-lbl" x="70"  y="77">B0</text>
<text class="gg-lbl" x="150" y="77">B1</text>
<text class="gg-lbl" x="230" y="77">B2</text>
<text class="gg-lbl" x="310" y="77">B3</text>
<rect class="gg-amber gg-mx" x="40" y="52" width="60" height="40" rx="6"/>
<text class="gg-lbl" x="190" y="118">weight tiles B (per column)</text>
<rect class="gg-tile" x="40"  y="150" width="60" height="58" rx="6"/>
<rect class="gg-tile" x="120" y="150" width="60" height="58" rx="6"/>
<rect class="gg-tile" x="200" y="150" width="60" height="58" rx="6"/>
<rect class="gg-tile" x="280" y="150" width="60" height="58" rx="6"/>
<text class="gg-lbl" x="70"  y="184">C0</text>
<text class="gg-lbl" x="150" y="184">C1</text>
<text class="gg-lbl" x="230" y="184">C2</text>
<text class="gg-lbl" x="310" y="184">C3</text>
<rect class="gg-blue gg-mx" x="40" y="150" width="60" height="58" rx="6"/>
<text class="gg-lbl" x="190" y="234">output row i sweeps left to right</text>
<text class="gg-miss" x="190" y="266">every step needs a new B tile</text>
<text class="gg-miss" x="190" y="286">evicted before reuse -&gt; L2 miss</text>
<line x1="380" y1="44" x2="380" y2="372" stroke="var(--border,#d1d5db)" stroke-width="1.5" stroke-dasharray="5 5"/>
<text class="gg-ttl" x="574" y="30">grouped (band) order</text>
<rect class="gg-tile" x="544" y="52" width="60" height="40" rx="6"/>
<rect class="gg-green gg-pl" x="544" y="52" width="60" height="40" rx="6"/>
<text class="gg-lbl" x="574" y="77">B col j</text>
<text class="gg-hit" x="574" y="118">one tile, resident in L2</text>
<rect class="gg-tile" x="544" y="150" width="60" height="52" rx="6"/>
<rect class="gg-tile" x="544" y="210" width="60" height="52" rx="6"/>
<rect class="gg-tile" x="544" y="270" width="60" height="52" rx="6"/>
<rect class="gg-tile" x="544" y="330" width="60" height="52" rx="6"/>
<text class="gg-lbl" x="574" y="181">C0j</text>
<text class="gg-lbl" x="574" y="241">C1j</text>
<text class="gg-lbl" x="574" y="301">C2j</text>
<text class="gg-lbl" x="574" y="361">C3j</text>
<rect class="gg-blue gg-my" x="544" y="150" width="60" height="52" rx="6"/>
<text class="gg-hit" x="680" y="266">band of A rows</text>
<text class="gg-hit" x="680" y="286">reuses B -&gt; L2 hit</text>
</svg>
<figcaption>Linear order marches across a row of output tiles, so the shared weight tile B changes every step and is evicted before it can be reused; grouped band order marches down a column, so one B tile stays hot in L2 for the whole band — the source of the +60% L2 hit rate and 1.33x speedup.</figcaption>
</figure>

The animation shows the fix. **Grouped launch order** (the right panel) traverses column-major within a band: hold a band of A rows, fix one column-tile of B, and sweep *down* the column — ${C_{0,j}, C_{1,j}, C_{2,j}, \dots}$ — computing every output tile that uses that same B tile before you ever move to the next column. While the left panel's shared weight tile flips on every step and gets evicted before reuse (L2 miss), the right panel's B tile stays resident in L2 for the entire band (L2 hit). Consecutive threadblocks in the persistent grid reuse the same weight tile in quick succession, and the `GROUP_SIZE_M` parameter controls how tall a band you hold — how many A row-tiles share one B tile before advancing.

The measured payoff is the most satisfying number in the whole post. On a representative grouped-GEMM shape — ${m = 4096}$, ${k = 2048}$, ${n = 7168}$, 8 groups — switching from linear to grouped launch order alone delivers a **1.33× speedup** and raises the **L2 cache hit rate by 60 percentage points**. No new arithmetic, no precision change, no different hardware — just visiting the output tiles in an order that respects the cache. If you ever want a crisp demonstration that memory access pattern, not FLOPs, governs GPU performance, this is it. (For the tooling to *see* an L2 hit-rate change like this, the [Nsight Compute kernel deep-dive](/blog/machine-learning/performance-engineering/nsight-compute-kernel-deep-dive) is the companion — the `l2_tex_read_hit_rate` counter is exactly what moved here.)

#### The `GROUP_SIZE_M` knob

`GROUP_SIZE_M` is the one tuning parameter worth understanding intuitively. It sets the height of the A-row band you keep resident against a fixed B column-tile. Set it to 1 and you are back to row-major (no reuse). Set it too large and the A band no longer fits in cache alongside the B tile, and you start thrashing on A instead of B. The sweet spot — typically 4 to 16 — is where the working set of "this many A row-tiles plus one B column-tile" just fits in L2. It is a classic cache-blocking parameter, and like all of them it is best found by autotuning over a few values for your exact ${m, n, k}$ and hardware rather than reasoned about in the abstract.

### 5c. Device-side TMA descriptors for data-dependent experts

The third trick is the most Hopper-specific, and it exists to solve a problem the first two created. The H100's Tensor Memory Accelerator (TMA) is a dedicated engine that streams a tile of memory from HBM to shared memory asynchronously, freeing the threads to compute. It is a big part of why Hopper GEMMs are fast. But TMA has a catch: it loads through a **descriptor** — a small structure describing the tile's base address, shape, and strides — and the standard, fast way to use TMA is to build that descriptor on the *host*, ahead of time, when you already know what you are loading.

For MoE, you do not know what you are loading until the kernel is running. Which expert's weight a given persistent program needs depends on which tile it just picked up, which depends on the `m_sizes` array, which depends on the router, which ran microseconds ago. The base address of the weight to load — ${\text{b\_ptr} + \text{expert\_idx} \times N \times K}$ — is a *data-dependent* quantity. A host-built descriptor cannot express "whichever expert this tile turns out to belong to." You would have to build one descriptor per expert and select among them, or worse, recompile.

![A timeline of the device-side TMA descriptor sequence: host allocates a per-SM workspace, the kernel reads the data-dependent expert index, builds a 2D TMA descriptor on-device pointing at that expert's weight, fences it to make it visible, then issues the TMA load](/imgs/blogs/grouped-gemm-moe-kernel-8.webp)

The kernel builds the descriptor **on the device, at runtime**, and the timeline shows the four-step dance. First, on the host, allocate a scratch workspace with one descriptor-sized slot per persistent program — one per SM — so there is no contention:

```python
# host: one TMA-descriptor slot per persistent program
TMA_SIZE = 128  # bytes per 2-D descriptor
workspace = torch.empty(NUM_SMS * TMA_SIZE, device=x.device, dtype=torch.uint8)
```

Then, inside the kernel, once a program knows the `expert_idx` for the tile it is working on, it writes a fresh 2-D descriptor into its own slot, pointing at that expert's weight band; fences to make the write visible to the TMA engine; and issues the load:

```python
# device (inside the persistent kernel), for the tile's chosen expert:
desc_ptr = workspace + start_pid * TMA_SIZE            # this program's private slot

tl.extra.cuda.experimental_device_tensormap_create2d(  # build the descriptor on-device
    desc_ptr=desc_ptr,
    global_address=b_ptr + expert_idx * N * K + n_start * K,  # data-dependent base
    load_size=[BLOCK_N, BLOCK_K],
    global_size=[NUM_EXPERTS * N, K],
    element_ty=tl.bfloat16)

tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(desc_ptr)  # make it visible

w_tile = tl.extra.cuda.experimental_descriptor_load(  # TMA streams the weight tile
    desc_ptr, [0, k_offset], [BLOCK_N, BLOCK_K], tl.bfloat16)
```

The **fence** is the subtle, load-bearing step people skip and then spend a day debugging. Building the descriptor is a normal memory write from the SM; the TMA engine reads descriptors through a separate "proxy" path. Without `experimental_tensormap_fenceproxy_acquire`, the TMA engine can read a *stale* descriptor — the previous expert's — because the SM's write has not yet propagated to where the TMA engine looks. The fence forces memory consistency between the SM and the TMA engine before the load issues, so you always stream the correct expert's weights. When a program moves to a tile belonging to a different expert, it overwrites its descriptor slot with the new base address, fences again, and reloads. This is how one compiled kernel serves 256 different experts without ever recompiling: the "which weight" decision is a runtime pointer, not a compile-time constant.

That last property — no recompilation as the routing changes — is exactly what makes the Triton kernel viable in a serving loop where the token-to-expert distribution shifts every batch. Contrast that with the trap in §9's case studies, where a naively `torch.compile`-d MoE recompiles every time the per-expert token counts change.

## 6. The padding tax: M-alignment and contiguous grouping

There is one honest cost that grouped GEMM cannot fully escape, and pretending otherwise would be dishonest engineering. It comes from a hard correctness constraint: **a tile must belong to exactly one expert.** A tile computes ${\text{BLOCK\_M} \times \text{BLOCK\_N}}$ outputs using one weight matrix. If a row-tile straddled the boundary between expert 0's rows and expert 1's rows, its top rows would need ${W_0}$ and its bottom rows ${W_1}$ — but a tile loads one weight. It cannot serve two experts. So every expert's row-band must start on a tile boundary, which means padding each ${m_e}$ up to a multiple of `BLOCK_M`.

![A grid showing three experts with 112, 40, and 8 rows each padded up to BLOCK_SIZE_M multiples of 128, 64, 64, with the small 8-row expert wasting the most — 96 of 256 rows, 37 percent padding on a hot batch](/imgs/blogs/grouped-gemm-moe-kernel-9.webp)

The figure makes the tax concrete with `BLOCK_M = 64`. Expert 0's 112 rows round up to 128 (16 wasted). Expert 1's 40 rows round up to 64 (24 wasted). Expert 2's 8 rows round up to 64 — **56 wasted rows to compute 8 real ones**. Sum it: 96 padding rows across a padded batch of 256, so 37% of the arithmetic in this (deliberately skewed) example is computing zeros. And notice *who* pays: the small experts. A cold expert with 8 tokens padded to 64 does ${8\times}$ the necessary work; a hot expert with 112 padded to 128 barely notices. Padding overhead is inversely proportional to load, so it is worst exactly when the router is imbalanced — the same condition that already hurt you.

The code to compute the padded layout is short, and it is the "contiguous grouped GEMM" prologue that production kernels run before the matmul:

```python
def pad_group_sizes(m_sizes, block_m):
    # m_sizes: [E] int tensor, tokens routed to each expert
    # round each expert's row count up to a BLOCK_M multiple so no
    # tile ever straddles two experts.
    padded = ((m_sizes + block_m - 1) // block_m) * block_m
    offsets = torch.cumsum(padded, dim=0) - padded   # start row of each group
    total_padded = int(padded.sum())
    return padded, offsets, total_padded
```

The design tension is now explicit and it is a real dial, not a free lunch. A **larger** `BLOCK_M` gives you fatter, more efficient tiles (better tensor-core utilization per tile, fewer tiles to schedule) but more padding waste, because you round up to a coarser boundary. A **smaller** `BLOCK_M` cuts the padding waste but gives you skinnier, less efficient tiles and more scheduling overhead. There is no globally correct choice; it depends on how skewed your routing is. Well-balanced routing (an auxiliary load-balancing loss doing its job) tolerates a large `BLOCK_M` because every expert is near-full anyway. Skewed routing wants a smaller `BLOCK_M` to avoid multiplying the cold experts' waste.

> Grouped GEMM does not make the padding cost zero; it makes it *bounded*. Pad-to-max scales the waste with the largest expert; pad-to-tile scales it with the tile size. Trading an unbounded tax for a few-percent one is the whole game.

### Second-order: padding interacts with load balancing and precision

Two consequences fall out of the padding math that are easy to miss. First, it changes how you should think about the router's [auxiliary load-balancing loss](/blog/machine-learning/scaling-laws/moe-scaling-laws): balanced routing is not just about not overloading one expert's memory — it directly reduces the grouped GEMM's padding waste, so a better-balanced router is a *faster* router, not only a more stable one. Second, the padded rows are real reads and writes of zeros through the memory system, so at low precision (FP8, where you are already bandwidth-limited on the weights) the relative cost of padding *rises* — the wasted rows are a larger fraction of a smaller compute budget. Teams pushing [FP8 MoE](/blog/machine-learning/model-serving/fp8-fp4-low-precision-serving-deep-dive) therefore care more about `BLOCK_M` tuning and routing balance than BF16 teams do.

## 7. What the numbers say

Time to assemble the wins into one table, because the point of all three tricks is that they *stack*, and the end-to-end number is what pays your GPU bill.

| Measurement | Baseline | Result | Setup |
| --- | --- | --- | --- |
| Grouped launch order vs linear | linear tile order | **1.33×**, +60 pts L2 hit | ${m{=}4096, k{=}2048, n{=}7168}$, 8 groups |
| Optimized vs baseline Triton | non-persistent Triton | **1.50×** | persistence + ordering + TMA, H100 |
| cuBLAS grouped vs batched loop | `cublasGemmBatchedEx` loop | **1.20×** | FP16, 8 and 64 groups |
| End-to-end training throughput | PyTorch expert loop | **1.42–2.62×** | 16B DeepSeek-V3, 8×H100, BF16, torchtitan + FSDP2 |
| Convergence (loss curve) | PyTorch expert loop | **matches** | no accuracy regression |

Read the last two rows together. The **2.62×** headline is measured end-to-end, on a real 16B-parameter DeepSeek-V3 trained with torchtitan and FSDP2 on eight H100s in BF16, against the honest baseline (the Python loop most people start with). The range — 1.42× to 2.62× — is because the win depends on batch size: at small batch the launch overhead the loop pays is proportionally larger, so the grouped kernel wins more; at large batch the loop's GEMMs are big enough to amortize their own launches, so the margin narrows. And the loss curve *matches* the loop's exactly, which is the essential control: a fast kernel that changes your numerics is not a speedup, it is a bug you have not found yet. The Triton kernel is bit-faithful enough to training that convergence is unchanged.

### Second-order: FSDP2 is the wrong parallelism, and that is a kernel problem

The PyTorch writeup is candid about a limitation that reaches beyond the kernel. FSDP2 — fully-sharded data parallelism — is a poor fit for MoE, because MoE has a high parameter-to-FLOP ratio (that is the whole point: lots of weights, few active per token), so FSDP2 spends enormous bandwidth all-gathering expert weights it barely uses. The natural fit is **expert parallelism**: place experts statically on GPUs and route tokens to them, so a weight never moves. But expert parallelism reintroduces the ragged-shape problem *across devices* — each GPU gets a dynamic, data-dependent number of tokens per step — which means the grouped GEMM's per-expert counts change every step, which means a `torch.compile`-d kernel wants to recompile every step. Solving that (dynamic kernel compilation, or a kernel that reads counts as pure data like the persistent Triton kernel does) is the frontier. It is why the device-side, no-recompile descriptor trick in §5c is not a micro-optimization but an enabler: it is what lets one compiled kernel survive a token distribution that changes every step. For the parallelism side of this story, [tensor, pipeline, and expert parallelism for serving](/blog/machine-learning/model-serving/tensor-pipeline-expert-parallelism-for-serving) is the companion.

## Case studies from production

The techniques above stay theoretical until you watch them decide a real system's performance. Here are eight incidents — some triumphs, some traps — where grouped GEMM was the pivot.

### 1. DeepSeek-V3 on torchtitan: the 2.62× that started this post

The PyTorch team's target was training-time throughput on a 16B DeepSeek-V3 variant under torchtitan with FSDP2 on 8×H100 in BF16. The starting point was the honest baseline: a Python loop over experts, two matmuls each. Profiling showed the expected pathology — tensor cores under-utilized, the trace a picket fence of short GEMMs separated by launch gaps, wave quantization eating the tail of every small per-expert matmul. The fix was the persistent, cache-aware, TMA-driven Triton kernel described in §5, layered in one optimization at a time so each win could be measured. Grouped launch ordering alone bought 1.33×; the full kernel bought 1.5× over baseline Triton; and end-to-end the throughput rose 1.42–2.62× depending on batch size, with the loss curve unchanged. The lesson that generalizes: the win came not from a cleverer matmul but from *scheduling* — persistence for the tail, ordering for the cache, device-side descriptors for the routing — and it was validated by keeping the numerics bit-faithful to the loop. When your speedup changes your loss, stop and find the bug.

### 2. MegaBlocks and "dropless" MoE: grouped GEMM as a correctness fix

Before grouped GEMM was a performance story, it was a *correctness* one. Early MoE training used a fixed "expert capacity" — a hard cap on tokens per expert — and simply dropped tokens that overflowed a hot expert, because the batched-GEMM primitives of the day demanded uniform shapes and capacity made the shapes uniform. Dropping tokens is a silent accuracy leak: the model never learns from the routed-but-discarded tokens, and the loss curve quietly suffers. MegaBlocks reframed the expert computation as **block-sparse matrix multiplication** — the direct ancestor of today's grouped GEMM — which let it process the true, ragged per-expert token counts *without* a capacity cap and *without* dropping. "Dropless MoE" was the headline, and grouped GEMM was the mechanism. The `fanshiqing/grouped_gemm` library and its `permute`/`gmm`/`unpermute` interface descend directly from this line of work. The lesson: the ragged-shape problem is not a performance nuisance to be padded away; solving it properly *removes a source of silent quality loss*, which is worth more than the speedup.

### 3. cuBLAS 12.5: NVIDIA blesses the pattern

For years grouped GEMM lived in CUTLASS templates and third-party libraries; the vendor BLAS had only batched and strided-batched. cuBLAS 12.5 changed that with `cublasGemmGroupedBatchedEx` and `cublasXgemmGroupedBatched` — first-class grouped APIs that take per-group shapes and pointers and run them in one launch. NVIDIA's own benchmark reported ~1.2× over a naive batched-GEMM loop on MoE-shaped workloads at 8 and 64 groups in FP16, notable because they achieved it with warp-level MMA rather than the newer warp-group MMA, showing the grouped scheduling itself was carrying the win. The strategic significance outweighs the 1.2×: once a pattern is in cuBLAS, every framework gets it for free, tuned per-architecture by the people who built the architecture. The lesson for infra engineers: watch what the vendor BLAS absorbs, because the day grouped GEMM landed in cuBLAS was the day "should we hand-roll this?" became "only if we need a schedule cuBLAS does not offer" — which, as the Triton kernel shows, is still a real case, but a narrower one.

### 4. DeepGEMM: FP8 grouped GEMM for DeepSeek inference

The Triton kernel in this post is BF16, with FP8 listed as in-progress, and that gap is exactly where DeepSeek's own DeepGEMM library lives. DeepGEMM is a from-scratch FP8 GEMM library, JIT-compiled, that includes grouped-GEMM variants specifically for MoE, targeting Hopper (and later Blackwell) with fine-grained block scaling to keep FP8 accurate. It is what serves DeepSeek-V3 in FP8 at the throughput the model was designed for. The interesting engineering detail is that FP8 grouped GEMM makes the padding tax from §6 *worse* in relative terms — FP8 is already bandwidth-bound on the weights, so wasted padded rows are a larger fraction of a tighter budget — which pushed DeepGEMM toward tighter tile scheduling and careful `BLOCK_M` choices. The lesson: precision and the grouped-GEMM schedule are coupled. You cannot pick an FP8 recipe and a tile size independently; the padding math changes under your feet when the bytes-per-element drops.

### 5. vLLM fused MoE: wiring grouped GEMM into a serving engine

The Triton persistent kernel did not stay a training curiosity; there is an open effort (vLLM PR #19443) to bring it into vLLM's fused-MoE path for *inference*. Serving raises constraints training does not. In training you can afford a permute/gmm/unpermute with generous workspace; in serving you are latency-sensitive, batch sizes shift request-to-request, and the token-to-expert distribution is never the same twice. This is precisely where the device-side TMA descriptor (§5c) earns its keep — because the kernel reads expert counts as runtime data and builds its weight descriptors on-device, it does not recompile when the batch's routing changes, so it survives the serving loop's churn. The lesson: a kernel's *recompilation behavior* is a first-class serving property. A kernel that is 2.62× faster but recompiles on every new batch shape can be slower in a serving loop than a slower kernel that never recompiles. Grouped GEMM's "shapes are data" design is what makes it serving-safe.

### 6. Megatron-Core GroupedMLP: the cuBLASLt-vs-CUTLASS crossover

Production MoE training frameworks do not commit to one grouped-GEMM backend; they choose per-workload. Megatron-Core's GroupedMLP offers two: a **multi-stream cuBLASLt** path that launches per-expert GEMMs into several CUDA streams so they overlap (cheap, flexible on precision and scaling — BF16, per-tensor FP8, blockwise FP8, MXFP8, NVFP4), and a **CUTLASS grouped** path that fuses all experts into one kernel (higher peak, but wins mainly when the GEMM count is large). The crossover is real and shape-dependent: with a handful of experts per rank, the multi-stream approach's flexibility and low overhead win; with many experts, CUTLASS's single-kernel fusion pulls ahead because it schedules all groups' tiles together and never pays a per-expert launch. The lesson: "use grouped GEMM" is not a single decision. The right backend depends on expert count per device, precision, and whether you need exotic scaling modes — and a serious framework exposes the choice rather than hard-coding it.

### 7. The recompilation trap under expert parallelism

A team moved their MoE from FSDP2 to expert parallelism to stop all-gathering expert weights, saw their all-to-all traffic drop as expected — and watched step time get *worse*. The culprit was recompilation. Under expert parallelism each GPU receives a dynamic, data-dependent number of tokens per step, so the local grouped GEMM's per-expert counts changed every step, and their `torch.compile`-d kernel treated each new count vector as a new shape and recompiled, paying compilation cost that dwarfed the communication they had saved. The fix had two parts: bucket the token counts to a small set of padded sizes so shapes repeated and the compile cache hit, and, for the hot path, use a kernel that consumes counts as pure runtime data (the persistent Triton kernel's device-side offsets) so it never recompiles at all. The lesson connects §5c and §7: the "shapes are data, not code" property is not academic. Violate it under dynamic routing and recompilation silently eats your parallelism win. This is the same class of failure as a [graph break in torch.compile](/blog/machine-learning/performance-engineering/debugging-graph-breaks), just triggered by data-dependent shapes instead of control flow.

### 8. The tile-straddle bug: when M-alignment is not optional

A kernel author writing their first grouped GEMM skipped the padding step — the routing looked well-balanced, every expert had a comfortable few hundred tokens, so why round up? The kernel ran, was fast, and produced subtly wrong outputs that only showed up as a slow divergence in the loss. The bug: without M-alignment, a row-tile at the boundary between two experts spanned rows belonging to both, but the tile loaded exactly one expert's weight, so the straddling rows were multiplied by the *wrong* expert's matrix. It was invisible on a single tile-aligned expert and only appeared when an expert's token count was not a multiple of `BLOCK_M` — which, with real routing, is almost always. The fix was the eight-line `pad_group_sizes` from §6: round every expert up to a tile boundary so no tile ever crosses an expert. The lesson is the whole reason padding exists, stated as a war story: the padding tax is not an efficiency choice, it is a *correctness* requirement. You are not paying it to go faster; you are paying it so the answer is right, and the fact that it is bounded to a few percent is the gift, not the cost.

## When to reach for grouped GEMM, and when not to

Grouped GEMM is the right tool far more often than not for MoE, but it is not free and it is not universal. Here is the decision boundary.

**Reach for grouped GEMM when:**

- **You are running an MoE FFN of any real size.** More than a few experts, or a batch large enough that per-expert launch overhead matters — which is essentially every production MoE. This is the default, and the burden of proof is on *not* using it.
- **Your per-expert token counts are ragged and data-dependent.** This is the exact shape batched and strided-batched GEMM cannot express, and the exact shape grouped GEMM was built for.
- **You are launch-bound or wave-quantization-bound.** If the profiler shows a picket fence of short GEMMs with idle gaps, or SMs idling in tail waves, grouped GEMM's single persistent launch is the fix.
- **You are on Hopper/Blackwell and can use the persistent + TMA path.** The cache-aware ordering and device-side descriptors were designed for these architectures and deliver their largest wins there.
- **You need a kernel that survives dynamic routing without recompiling** — expert parallelism, a serving loop with shifting batch shapes. The "shapes are data" property is a serving-safety feature, not just a speed one.

**Skip grouped GEMM (or don't over-invest) when:**

- **The model is dense.** A single fat matmul is already the ideal GPU workload; grouped GEMM adds machinery for a raggedness you do not have. Use cuBLAS/CUTLASS plain GEMM.
- **You have exactly one or two experts, or perfectly uniform token counts.** With no raggedness, a batched GEMM (or even two plain GEMMs) is simpler and gives up nothing. Grouped GEMM's descriptor overhead is pure cost here.
- **The GEMM is not your bottleneck.** If the profiler says the permute/unpermute, the all-to-all, or the router is the hot spot, a faster grouped GEMM buys you nothing. Fuse the permutation, fix the communication, and measure again before touching the matmul. In MoE, the data movement around the GEMM is frequently the real cost.
- **You cannot yet get the precision you need.** If you require FP8 today and your only grouped path is BF16, either use a purpose-built FP8 grouped library (DeepGEMM) or accept BF16 — do not hand-roll FP8 into a kernel that does not support it and hope.
- **Routing is so skewed that padding dominates.** If one expert draws 90% of tokens, you have a load-balancing problem that grouped GEMM will *expose* (via padding waste) but not solve. Fix the router's balance first; the kernel is downstream of that.

The through-line of this whole post is a single reframing. A GPU is a machine that wants one big, regular, cache-friendly matrix multiply, and MoE hands it a pile of small, irregular, unpredictable ones. Grouped GEMM is the adapter between those two facts — not by making the machine tolerate irregularity, but by *gathering* the irregular pieces into one launch, *scheduling* them so the machine stays full, *ordering* their memory accesses so the cache stays hot, and *reading the shapes as data* so nothing has to recompile when the router changes its mind. Every trick in it — persistence, grouped ordering, device-side descriptors, tile padding — is in service of that one adaptation. Get it right and MoE is the best deal in modeling: frontier quality at a fraction of the active compute. Get the kernel wrong and you pay for a rack of H100s to watch their tensor cores idle behind a Python loop.

## Further reading

- [Serving Mixture-of-Experts Models at Scale](/blog/machine-learning/model-serving/serving-moe-models-at-scale) — the system-level companion: routers, experts, all-to-all, and why MoE is memory-capacity-bound.
- [Custom CUDA Kernels for Inference](/blog/machine-learning/model-serving/custom-cuda-kernels-for-inference) — when to drop below PyTorch ops at all, with a real Triton fused kernel.
- [Tensor, Pipeline, and Expert Parallelism for Serving](/blog/machine-learning/model-serving/tensor-pipeline-expert-parallelism-for-serving) — the parallelism that makes expert weights stop moving, and the ragged-across-devices problem it creates.
- [Kernel Fusion, CUDA Graphs, and torch.compile](/blog/machine-learning/model-serving/kernel-fusion-cuda-graphs-torch-compile) — the launch-overhead machinery grouped GEMM leans on.
- [PyTorch blog: Accelerating MoEs with a Triton persistent cache-aware grouped GEMM kernel](https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/) — the primary source for §5.
- [NVIDIA: Introducing grouped GEMM APIs in cuBLAS](https://developer.nvidia.com/blog/introducing-grouped-gemm-apis-in-cublas-and-more-performance-updates/) — the cuBLAS 12.5 grouped API.
- [`fanshiqing/grouped_gemm`](https://github.com/fanshiqing/grouped_gemm) — the CUTLASS-backed PyTorch binding with `permute`/`gmm`/`unpermute`.
