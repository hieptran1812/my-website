---
title: "Attention Backends Deep Dive: FlashAttention, FlashAttention-3, and FlashInfer"
date: "2026-07-07"
publishDate: "2026-07-07"
description: "A principal engineer's guide to the attention kernels that decide LLM-serving performance — the online-softmax math that makes FlashAttention exact, the I/O-complexity derivation, what FA2 vs FA3 changed, why FlashInfer wins for ragged decode, and how to pick and tune the backend in vLLM."
tags:
  [
    "model-serving",
    "inference",
    "flashattention",
    "flashinfer",
    "attention",
    "gpu-kernels",
    "vllm",
    "paged-attention",
    "fp8",
    "ml-infrastructure",
  ]
category: "machine-learning"
subcategory: "Model Serving"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/attention-backends-deep-dive-flashattention-flashinfer-1.webp"
---

The regression was three lines in a config diff. We had upgraded a Llama-3-70B serving fleet from an older vLLM build, and someone had pinned `VLLM_ATTENTION_BACKEND=XFORMERS` months earlier "to be safe" — a value copied from a 2023 runbook that nobody had revisited. On H100s. The service met its SLA, so nothing paged. But when we ran a capacity review, the numbers were embarrassing: prefill throughput was roughly 40% below what the same hardware delivered with the default FlashAttention backend, and decode was leaving a chunk of GPU memory bandwidth on the floor because the paged-attention path was falling back to a slower kernel. We were paying for H100s and running an A100-era attention kernel on them. One environment-variable change bought back nearly half the prefill throughput and let us delete two nodes from the fleet.

That episode is the reason this post exists. Attention is not just one operation among many in a transformer — for LLM serving it is *the* operation whose kernel implementation most directly sets your latency and your bill. Every other layer (the MLP, the norms, the projections) is a plain matmul or elementwise op that the compiler handles well. Attention is different: it has an $N \times N$ intermediate that you must never write to memory, a softmax that couples every element of a row, and two completely different performance regimes depending on whether you are in prefill or decode. The kernels that solve these problems — FlashAttention and its descendants, and serving-focused engines like FlashInfer — are where the largest, cheapest wins in LLM inference hide, and where the wrong default silently taxes you.

The figure below is the one idea the whole post orbits. On the left, "naive" attention computes the score matrix $S = QK^\top$, writes it to high-bandwidth memory (HBM), reads it back to apply softmax, writes the probabilities, reads them again to multiply by $V$. On the right, FlashAttention fuses all of that into one kernel that streams tiles of $K$ and $V$ through on-chip SRAM and *never writes the score matrix at all*. That single refusal — do not materialize $S$ — is worth tens of times less HBM traffic, and it converts a kernel that was memory-bound and left the tensor cores idle into one that can actually keep them busy.

![Before-after comparison of naive attention materializing the N by N score matrix in HBM at about 512 MB of traffic versus FlashAttention streaming tiles through SRAM at about 8 MB, roughly 64 times less HBM traffic](/imgs/blogs/attention-backends-deep-dive-flashattention-flashinfer-1.webp)

By the end of this post you will be able to: derive why FlashAttention's HBM traffic scales as $O(N^2 d^2 / M)$ instead of $O(N^2)$; explain the online-softmax recurrence well enough to implement it; state precisely what FlashAttention-2 changed (parallelism and work partitioning) and what FlashAttention-3 changed (Hopper asynchrony and FP8); describe why decode needs a completely different, paged kernel than prefill; read vLLM's backend-selection logic and override it with `VLLM_ATTENTION_BACKEND` when — and only when — you should; and benchmark two backends against each other across sequence lengths with `do_bench`. This sits one level below the [custom CUDA kernels for inference](/blog/machine-learning/model-serving/custom-cuda-kernels-for-inference) post: there we surveyed the whole kernel landscape; here we go all the way down into the single kernel that matters most. It is also the attention-specific companion to [roofline analysis for LLM inference](/blog/machine-learning/model-serving/roofline-analysis-for-llm-inference) and [GPU-architecture-specific tuning for LLM serving](/blog/machine-learning/model-serving/gpu-architecture-specific-tuning-for-llm-serving).

Throughout, keep the series' spine in mind: every serving technique is a trade on the latency-throughput-cost triangle. Attention backends are unusual in that a good one often improves all three corners at once — which is exactly why leaving the wrong one pinned is such an expensive mistake.

## Why attention is the kernel that decides your serving bill

Start with the arithmetic. For a single attention head with sequence length $N$ and head dimension $d$, attention computes

$$O = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V, \qquad Q, K, V \in \mathbb{R}^{N \times d}.$$

The two matmuls — $S = QK^\top$ (shape $N \times N$) and $O = PV$ (where $P$ is the softmax of $S$) — each cost about $2N^2 d$ floating-point operations. So the compute is $\Theta(N^2 d)$. The *inputs and outputs*, however, are only $\Theta(Nd)$: three matrices in, one out. The problem child is the intermediate $S$, which is $\Theta(N^2)$ and grows quadratically in the sequence length. At $N = 8192$ and fp16, a single head's score matrix is $8192^2 \times 2$ bytes $= 128$ MiB. With 32 heads that is 4 GiB — for one layer, for one forward pass, of a tensor you are about to throw away.

The reason this matters is the memory wall. A modern data-center GPU has enormously more compute throughput than memory bandwidth. An H100 SXM delivers about 989 TFLOP/s of dense fp16 tensor-core compute but only about 3.35 TB/s of HBM bandwidth. The ratio — roughly 295 fp16 FLOPs per byte at the "ridge point" of the roofline — means that any kernel doing fewer than ~295 FLOPs per byte it moves is *memory-bound*: its speed is set by how fast it can read and write HBM, not by how fast it can multiply. Naive attention, which writes and re-reads a 128 MiB score matrix to do a comparatively small amount of arithmetic per byte, sits deep in the memory-bound regime. The tensor cores starve. (If the roofline framing is new, the dedicated [roofline analysis](/blog/machine-learning/model-serving/roofline-analysis-for-llm-inference) post derives the ridge point in full; here we just use it.)

This is the entire motivation for FlashAttention. The insight is not to reduce FLOPs — FlashAttention does the same $\Theta(N^2 d)$ arithmetic as naive attention, it is *exact*, not an approximation — but to reduce *HBM traffic* by refusing to materialize $S$. If you can hold a tile of the computation in SRAM (the on-chip scratchpad, ~228 KB per streaming multiprocessor on H100) and finish everything you need to do with that tile before writing it back, you never pay the quadratic memory cost. The catch is softmax: it seems to need the whole row of $S$ at once to compute the normalizing denominator. Breaking that dependency is the mathematical trick at the heart of the algorithm, and it is worth understanding in detail.

The [why LLM serving is different](/blog/machine-learning/model-serving/why-llm-serving-is-different) post frames the memory wall at the level of the whole model and the KV cache. This post zooms into the one kernel where the wall is most acute, because the score matrix is quadratic where everything else is linear.

## The online-softmax recurrence: exact softmax you never fully see

Softmax over a vector of scores $s_1, \dots, s_N$ is

$$p_i = \frac{e^{s_i}}{\sum_{j=1}^{N} e^{s_j}}, \qquad o = \sum_{i=1}^{N} p_i v_i.$$

Numerically you never exponentiate raw scores — a single $s_i = 90$ overflows fp16 — so the standard "safe softmax" subtracts the row maximum $m = \max_j s_j$ first: $p_i = e^{s_i - m} / \sum_j e^{s_j - m}$. This is exact and stable, but it appears to require two passes over the row: one to find $m$ and the denominator $\ell = \sum_j e^{s_j - m}$, and one to compute the weighted sum. Two passes means you must have all of $S$ available, which is exactly what we are trying to avoid.

The online-softmax trick, due to Milakov and Gimelshein and adapted into FlashAttention by Dao et al., collapses this into a *single* streaming pass by carrying a small running state and correcting it as new blocks arrive. Process the scores in blocks. Maintain three running quantities: the running max $m$, the running denominator $\ell$, and the running (unnormalized) output accumulator $o$. When a new block with local max $\tilde{m}$ and local contributions arrives, you update:

$$m^{\text{new}} = \max(m, \tilde{m}),$$
$$\ell^{\text{new}} = e^{\,m - m^{\text{new}}}\,\ell \;+\; \sum_{i \in \text{block}} e^{\,s_i - m^{\text{new}}},$$
$$o^{\text{new}} = e^{\,m - m^{\text{new}}}\,o \;+\; \sum_{i \in \text{block}} e^{\,s_i - m^{\text{new}}}\, v_i.$$

The correction factor $e^{\,m - m^{\text{new}}}$ is the whole idea. Whenever a later block raises the maximum, everything you accumulated so far was exponentiated against a *smaller* max and is therefore too large by exactly the factor $e^{\,m^{\text{old}} - m^{\text{new}}}$. You rescale the running denominator and the running output by that factor to bring them onto the new reference, then add the new block's contribution. After the last block, the final output is $o / \ell$. The result is bit-for-bit the same safe-softmax answer you would get with two passes, but you only ever held one block of scores in SRAM at a time, plus $O(d)$ of running state per query row. The figure below traces one iteration of this recurrence.

![Timeline of the online-softmax recurrence: read a KV block into SRAM, compute block scores, update the running max, rescale the running sum and output by exp of the max difference, accumulate, and normalize after the last block](/imgs/blogs/attention-backends-deep-dive-flashattention-flashinfer-2.webp)

Here is a compact reference implementation. It is not the GPU kernel — it is the math, in NumPy, so you can convince yourself the recurrence is exact by comparing against a plain softmax.

```python
import numpy as np

def online_softmax_attention(Q, K, V, block=256):
    # Q,K,V: (N, d). Returns (N, d), exact softmax attention, one streaming pass over KV.
    N, d = Q.shape
    scale = 1.0 / np.sqrt(d)
    O = np.zeros((N, d), dtype=np.float64)
    for qs in range(0, N, block):                 # outer loop: Q tiles (FA2 order)
        q = Q[qs:qs + block]                       # (Bq, d)
        m = np.full((q.shape[0], 1), -np.inf)      # running max per query row
        l = np.zeros((q.shape[0], 1))              # running denominator
        o = np.zeros((q.shape[0], d))              # running unnormalized output
        for ks in range(0, N, block):              # inner loop: stream KV tiles
            k = K[ks:ks + block]                    # (Bk, d)
            v = V[ks:ks + block]
            s = (q @ k.T) * scale                   # (Bq, Bk) block scores — never leaves SRAM on GPU
            m_new = np.maximum(m, s.max(axis=1, keepdims=True))
            corr = np.exp(m - m_new)                # rescale factor for prior state
            p = np.exp(s - m_new)                    # block probabilities vs new max
            l = corr * l + p.sum(axis=1, keepdims=True)
            o = corr * o + p @ v                     # accumulate rescaled + new contribution
            m = m_new
        O[qs:qs + block] = o / l                    # normalize once, at the end
    return O

# sanity check: matches a plain two-pass softmax to floating-point precision
Q, K, V = (np.random.randn(1024, 64) for _ in range(3))
ref = (lambda s: (np.exp(s - s.max(1, keepdims=True)) /
       np.exp(s - s.max(1, keepdims=True)).sum(1, keepdims=True)) @ V)((Q @ K.T) / 8.0)
assert np.allclose(online_softmax_attention(Q, K, V), ref, atol=1e-10)
```

The GPU kernel does exactly this, but with the block matmuls on tensor cores, the running state in registers, and the tiles streamed from HBM into SRAM. The `assert` passing is the point: FlashAttention is not a fast approximation of attention, it is a memory-efficient *exact* computation of it.

Three properties of this recurrence are worth drawing out, because each one is load-bearing later in the post.

**The running state is associative — partial results merge.** Nothing in the update requires the blocks to arrive in order, or even for a single thread to see all of them. If thread A processes KV blocks 0–3 into a partial state $(m_A, \ell_A, o_A)$ and thread B independently processes blocks 4–7 into $(m_B, \ell_B, o_B)$, a third step can combine them with the exact same rescaling logic: take $m = \max(m_A, m_B)$, then $\ell = e^{m_A - m}\ell_A + e^{m_B - m}\ell_B$ and $o = e^{m_A - m}o_A + e^{m_B - m}o_B$, and divide $o/\ell$ at the end. This merge is what makes *split-KV decoding* (below) possible: you can shard one query's long KV cache across many streaming multiprocessors, each computing a partial softmax over its shard, and reduce the partials correctly afterward. Hold onto this — it is the single most important structural fact for decode performance, and it is the same rescaling you already saw within one thread, applied one level up.

Here is the merge as code, which is all a flash-decoding reduction kernel does after the split:

```python
def merge_states(m_a, l_a, o_a, m_b, l_b, o_b):
    # Combine two partial online-softmax states into one, exactly.
    m = np.maximum(m_a, m_b)
    a, b = np.exp(m_a - m), np.exp(m_b - m)   # rescale each partial onto the shared max
    l = a * l_a + b * l_b
    o = a * o_a + b * o_b
    return m, l, o                            # caller divides o / l once, at the very end
```

**Precision: accumulate in fp32 even when the inputs are fp16.** The running $\ell$ and $o$ are sums over potentially tens of thousands of positive terms, and the rescaling multiplies them repeatedly by factors at or below one. Doing that arithmetic in fp16 would bleed accuracy — the denominator can span many orders of magnitude across a long context, and fp16 has only ~3 decimal digits of precision. Every production FlashAttention kernel keeps $m$, $\ell$, and the output accumulator in fp32 registers regardless of the fp16/bf16 input dtype, and only casts the final $o/\ell$ back down. This is why "FlashAttention is exact" is a precise claim about the *algorithm*: the online recurrence introduces no approximation, and the implementation preserves that by accumulating the reduction in higher precision. The one deliberate exception is FP8 attention (FA3), where the *matmul inputs* are quantized — a separate, opt-in precision trade discussed later, not a property of online softmax itself.

**The log-sum-exp is the one extra scalar worth storing.** At the end of a row, FlashAttention can write out $L = m + \log \ell$ — the log of the softmax denominator — alongside the output. A serving forward pass can drop it, but it is exactly what the backward pass needs to recompute the probabilities $P$ tile by tile without ever having stored them, and it is also the quantity that split-KV partials carry between the split and reduce kernels. It is $O(N)$ bytes against the $O(Nd)$ output, so storing it is nearly free.

#### Worked example: HBM traffic of naive vs FlashAttention for an 8k prefill

Take one attention head, one layer, $N = 8192$, $d = 128$, fp16. Naive attention writes and reads the $8192 \times 8192$ score matrix (${128}$ MiB) four times over the course of the two matmuls and the softmax: write $S$, read $S$ to compute $P$, write $P$, read $P$ for $PV$. That is $4 \times 128 = 512$ MiB of HBM traffic for the quadratic temporary, plus a negligible $4 \times Nd \times 2 = 8$ MiB for $Q, K, V, O$. Total: about ${520}$ MiB.

FlashAttention reads $Q, K, V$ and writes $O$ (plus a tiny log-sum-exp vector for the backward pass), and holds every score tile in SRAM. The dominant HBM traffic is the $4 \times Nd \times 2 \approx 8$ MiB of inputs and outputs. The ratio is roughly $520 / 8 \approx 65\times$ less HBM traffic — the "~64×" on the figure above. Now scale it: a full Llama-3-8B prefill has 32 layers and 32 query heads. Per layer, naive attention shuttles about $32 \times 512$ MiB $= 16$ GiB of score-matrix traffic through HBM; FlashAttention shuttles about $32 \times 8$ MiB $= 256$ MiB. On an H100 at 3.35 TB/s, the naive score traffic alone across all 32 layers is $32 \times 16 = 512$ GiB, or about ${153}$ ms of pure memory movement per prefill; FlashAttention's is about ${8}$ GiB, or ${2.4}$ ms. That two-order-of-magnitude gap in memory traffic is why no serious serving stack has materialized the score matrix since 2022.

## FlashAttention: tiling and the I/O-complexity derivation

The online-softmax recurrence tells you *what* to compute; tiling tells you *how* to schedule it on the GPU so the traffic bound is actually achieved. The core theorem from the original FlashAttention paper (Dao, Fu, Ermon, Rudra, Ré, 2022) is a statement about I/O complexity: the number of HBM accesses, not FLOPs.

Let $M$ be the size of on-chip SRAM in elements (a few tens of thousands of fp16 values per SM). FlashAttention chooses block sizes $B_c$ (for the KV dimension) and $B_r$ (for the Q dimension) so that a $K$ tile, a $V$ tile, a $Q$ tile, and the running state all fit in SRAM simultaneously — which forces $B_c = \Theta(M/d)$. The outer loop runs over KV blocks (in FA1) or Q blocks (in FA2); the inner loop streams the other. Counting HBM accesses: $Q, K, V$ are each read, and $O$ written, but $K$ and $V$ get re-read once per Q block. The number of Q blocks is $N / B_r$, and with the block sizes set by SRAM, the total HBM access count works out to

$$\Theta\!\left(\frac{N^2 d^2}{M}\right)$$

versus $\Theta(N d + N^2)$ for standard attention. Compare the dominant terms: standard attention pays $\Theta(N^2)$; FlashAttention pays $\Theta(N^2 d^2 / M)$. The ratio is $d^2 / M$. With $d = 64$–${128}$ and $M$ on the order of $10^5$ elements, $d^2/M$ is well below one — the paper reports roughly $9\times$ fewer HBM accesses for GPT-2-scale settings, and the gap widens as $M/d^2$ grows. Crucially, the reduction requires $M \geq \Theta(d^2)$, which every real GPU satisfies comfortably. This is the derivation that turns the hand-wavy "keep it in SRAM" into a concrete asymptotic win.

To see where the $d^2/M$ ratio comes from mechanically, count the re-reads rather than trusting the asymptotics. FA2's outer loop is over Q tiles and its inner loop streams the full $K$ and $V$. With a Q-tile height $B_r$, there are $N/B_r$ Q tiles, and each one reads all of $K$ and $V$ once — so $K$ and $V$ are each re-read $N/B_r$ times over the whole kernel. Total input traffic is therefore about $\frac{N}{B_r}\cdot 2Nd = \frac{2N^2 d}{B_r}$ elements. The block-size constraint sets $B_r = \Theta(M/d)$ — a Q tile, a KV tile, and the running state must co-reside in SRAM — and substituting gives $\Theta(N^2 d^2 / M)$. The bound falls straight out of "how many times is $K$ re-read," with no hand-waving.

#### Worked example: the re-read factor at 8k context, and why it is nearly free

Take $N = 8192$, $d = 128$, and a Q-tile height $B_r = 128$. There are $N/B_r = 64$ Q tiles, so $K$ and $V$ are each streamed through the kernel ${64}$ times. That is a lot of reads — naively it sounds worse than naive attention, which reads $K$ and $V$ once. The resolution, and the real reason FlashAttention wins, is *what those reads overlap with*. FlashAttention's $K,V$ re-reads happen inside a tile loop that is running tensor-core matmuls the whole time; on a compute-bound prefill the memory system prefetches the next $K,V$ tile while the tensor cores chew on the current one, so the re-read latency hides almost entirely under compute. Naive attention's score-matrix traffic is the opposite: writing ${512}$ MiB of $S$ and $P$ to HBM and reading it back is pure memory movement with no arithmetic to hide it under — it *is* the bottleneck. So the honest comparison is not "8 MiB versus 520 MiB of raw bytes"; it is "hideable, compute-overlapped input re-reads versus un-hideable quadratic-temporary traffic." The asymptotic $d^2/M \ll 1$ ratio captures the first-order win; the overlap is why the realized speedup is larger still. Memory traffic you can hide behind compute versus memory traffic you cannot — that distinction is the through-line of every kernel in this post, and it is why prefill (lots of compute to hide behind) and decode (almost none) end up needing different kernels entirely.

The scheduling that realizes this is a grid of Q tiles by KV tiles. In FlashAttention-2, each Q tile is owned by one thread block (a warp group), which streams every KV block left to right, updates its running $(m, \ell, o)$ in registers, and at the end writes only its $B_r \times d$ slice of the output. The figure shows the schedule: three Q tiles, each a separate thread block, each walking the KV blocks and writing one output tile. No cell of the conceptual $N \times N$ score grid is ever written to HBM.

![Grid of FlashAttention-2 tiling showing three Q tiles as rows, each streaming KV blocks 0 through 2 across the columns while maintaining running max and sum in SRAM, then writing one output tile per row](/imgs/blogs/attention-backends-deep-dive-flashattention-flashinfer-3.webp)

Two subtleties make the difference between a textbook description and a production kernel. First, **causal masking**: in decoder attention, query $i$ only attends to keys $j \le i$. A tiled kernel exploits this by skipping entire KV blocks that lie strictly above the diagonal for a given Q block — for long sequences this halves the attention FLOPs, and the kernel must handle the partial (diagonal) blocks with an element-wise mask while skipping the fully-masked ones outright. Second, **the backward pass** (for training, less relevant to serving but worth noting) recomputes $S$ and $P$ tiles on the fly rather than storing them, trading a modest amount of extra compute for the same memory savings. For serving you only run the forward pass, so the win is pure.

A practical note on how you actually call this. In PyTorch, `torch.nn.functional.scaled_dot_product_attention` (SDPA) dispatches to a FlashAttention kernel automatically when the inputs qualify (fp16/bf16, supported head dim, a compatible mask). You rarely call the raw kernel yourself in a serving stack, but knowing what SDPA is choosing under it matters:

```python
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

q, k, v = (torch.randn(4, 32, 8192, 128, device="cuda", dtype=torch.bfloat16)
           for _ in range(3))  # (batch, heads, seq, head_dim)

# Let PyTorch pick the fastest available backend (usually FlashAttention on Ampere/Hopper).
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

# Or pin the FlashAttention kernel explicitly and fail loudly if it is unavailable,
# so a silent fallback to the math backend never hides in a benchmark:
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

The equivalent when you use the reference `flash-attn` package directly (the one that ships the actual FlashAttention-2/3 kernels) is `flash_attn_func`, which takes tensors in `(batch, seq, heads, head_dim)` layout and returns the attention output without ever exposing a score matrix:

```python
from flash_attn import flash_attn_func  # pip install flash-attn

# q, k, v: (batch, seqlen, nheads, head_dim), fp16/bf16, on CUDA
out = flash_attn_func(q, k, v, causal=True, softmax_scale=None)  # scale defaults to 1/sqrt(d)
```

## FlashAttention-2 vs FlashAttention-3: what actually changed

FlashAttention has had three major versions, and the differences are not cosmetic — each one chased a specific inefficiency that the previous version left on the table. Getting this right matters because the version you get depends on your GPU and your library, and the speedups are large.

**FlashAttention-1 (2022)** established the tiling + online-softmax algorithm and the I/O bound. It was already a big win over naive attention, but its GPU utilization was mediocre — around 25–40% of peak on A100 — because of how it partitioned work across warps.

**FlashAttention-2 (2023)**, from Tri Dao, kept the algorithm and attacked the scheduling. Three changes: (1) **better parallelism** — FA2 parallelizes the forward pass over the sequence-length dimension in addition to batch and heads, so long sequences with small batch (exactly the serving decode-adjacent regime) keep all the SMs busy; (2) **better work partitioning between warps** — FA2 reworked how a thread block's warps split the Q and KV tiles to cut the shared-memory reads and writes that FA1 spent synchronizing partial results; and (3) **fewer non-matmul FLOPs** — this one is subtle and important. On a tensor-core GPU, matmul throughput dwarfs non-matmul throughput: an A100 does ~312 TFLOP/s of fp16 matmul but only ~19.5 TFLOP/s of fp32 non-matmul arithmetic, a 16× gap. The online-softmax rescaling involves non-matmul work (the exponentials and the $e^{m-m^{\text{new}}}$ corrections), so FA2 restructured the algebra to minimize those operations — for instance, deferring the division by $\ell$ to the very end rather than rescaling every block. The net result: about $2\times$ over FA1, reaching up to ~225 TFLOP/s on A100, or roughly 72% of the fp16 matmul peak.

**FlashAttention-3 (2024)**, from Shah, Bikshandi, Zhang, Dao and collaborators, is **Hopper-specific** — it exploits hardware features that only exist on the H100/H200 (SM90) architecture. Three techniques: (1) **producer-consumer asynchrony via warp specialization** — some warps become dedicated "producers" that issue asynchronous TMA (Tensor Memory Accelerator) copies to move tiles from HBM to SRAM, while other warps are "consumers" that run the tensor-core matmuls, so data movement and compute overlap instead of taking turns; (2) **interleaving matmul and softmax** (a ping-pong schedule) so the low-throughput softmax exponentials of one block hide under the high-throughput matmul of the next, keeping the tensor cores fed; and (3) **FP8 attention with block quantization and incoherent processing** — FA3 can run the attention matmuls in FP8, roughly doubling tensor-core throughput, and uses per-block scaling plus a Hadamard-transform trick ("incoherent processing") to keep the numerical error low despite the tiny dynamic range of FP8. The reported results: FA3 in fp16 is about ${1.5}$–$2.0\times$ faster than FA2 on H100, reaching up to ~740 TFLOP/s (about 75% of the fp16 peak), and FA3 in FP8 approaches ~1.2 PFLOP/s with about $2.6\times$ lower numerical error than a baseline FP8 attention.

| Property | FlashAttention-1 | FlashAttention-2 | FlashAttention-3 |
|---|---|---|---|
| Year | 2022 | 2023 | 2024 |
| Target GPUs | Ampere+ (sm80) | Ampere+ (sm80) | Hopper only (sm90) |
| Key idea | tiling + online softmax | parallelism + work partitioning | async TMA + warp specialization + FP8 |
| fp16 throughput | ~1× baseline | ~225 TFLOP/s A100 (72%) | ~740 TFLOP/s H100 (75%) |
| FP8 support | no | no | yes (~1.2 PFLOP/s) |
| Best use | historical | A100 serving, any sm80/sm89 | H100 prefill, FP8 attention |

The three FA3 techniques rest on three specific Hopper hardware primitives, and it is worth naming them because they recur across every modern Hopper kernel, not just attention:

- **TMA (Tensor Memory Accelerator)** is a dedicated copy engine that moves a whole tile between HBM and SRAM from a single descriptor, asynchronously, without tying up the threads that will consume it. FA3's "producer" warps issue TMA loads for the next $K,V$ tile and move on; the copy proceeds in the background. On Ampere, the same copy had to be driven by the compute warps themselves (`cp.async` at best), which stole issue slots from the matmuls.
- **WGMMA (warp-group matrix-multiply-accumulate)** is Hopper's asynchronous tensor-core instruction: a whole warp group (${128}$ threads) launches a large matmul that retires asynchronously, so the warps can issue the next instruction — a TMA prefetch, or the softmax exponentials — while the MMA is still in flight. This asynchrony is what the ping-pong schedule exploits: while one warp group's WGMMA runs, the other warp group runs softmax, and they trade roles every tile.
- **Enough register file and shared memory to double-buffer**, so a producer can fill buffer B while the consumer drains buffer A, and no warp ever waits at a tile boundary.

Put together, these turn attention from a "load, then compute, then load" sequence into a software pipeline where loads and computes run concurrently across warp groups — the difference between a kernel that stalls at every tile boundary and one that keeps the tensor cores saturated. It is the same pipelining idea as CPU instruction pipelining, applied at the tile granularity by hand, because the compiler will not do it for you. FA2 could not exploit any of this because sm80 has no TMA and no async WGMMA; the FA2 kernel is genuinely a different program, not a recompile.

You should confirm which FlashAttention version your stack actually resolved to, because "I installed flash-attn" does not tell you whether the FA3 Hopper path is live:

```python
import torch

cap = torch.cuda.get_device_capability()          # (9, 0) on H100, (8, 0) on A100
print("compute capability:", cap)                  # FA3 kernels require sm90 (Hopper)

try:
    import flash_attn
    print("flash-attn version:", flash_attn.__version__)   # FA3 ships in the 3.x hopper builds
    from flash_attn import flash_attn_interface     # present only when the Hopper FA3 path is built
    print("FA3 Hopper interface available:", True)
except Exception as e:                              # noqa: BLE001 — illustrative probe
    print("FA3 interface not available:", e)
```

If you are on `sm90` but the FA3 interface is missing, you built the sm80 wheel and are silently running FA2 kernels on a Hopper card — the exact "A100-era kernel on H100 hardware" failure from the opening story, one layer further down the stack.

The FP8 path deserves a code note because it is the newest and least familiar. In a serving stack you rarely call FA3's FP8 attention directly; you enable it through the engine. In vLLM, FP8 attention on Hopper is gated by the KV-cache dtype and the backend, roughly:

```python
from vllm import LLM

# FP8 KV cache + an FP8-capable attention backend (FlashAttention-3 / FlashInfer) on H100.
# The engine quantizes the KV cache to fp8_e4m3 and runs the attention matmuls in FP8
# where the backend supports it — halving KV-cache bytes and lifting attention throughput.
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    kv_cache_dtype="fp8_e4m3",     # store K/V in FP8; ~2x KV-cache capacity
    dtype="bfloat16",              # weights/activations stay bf16
    max_model_len=32768,
    gpu_memory_utilization=0.90,
)
```

Two honest caveats. FP8 attention trades a little accuracy for throughput; validate perplexity and task metrics before shipping it, because "close to fp16" is a claim about aggregate error, not about your specific eval. And FP8 attention is only a win on Hopper (and Blackwell) — on an A100 there is no FP8 tensor-core path for attention, so the flag either errors or falls back.

#### Worked example: what FP8 attention buys and costs on Hopper

Concretely, on H100 the fp16 tensor-core peak is about ${989}$ TFLOP/s and the FP8 (e4m3) peak is about ${1979}$ TFLOP/s — a clean ${2}\times$ on the matmuls. FA3's fp16 attention already reaches ~${740}$ TFLOP/s (about ${75}$% of fp16 peak); its FP8 path reaches ~${1.2}$ PFLOP/s, so the realized attention speedup from turning FP8 on is roughly ${1.6}\times$ — less than the ${2}\times$ raw ratio because the softmax exponentials and the rescaling stay in higher precision and now take a proportionally larger share of the kernel time (Amdahl's law, at the granularity of one attention kernel). The cost side is numerical. FP8's e4m3 format has ${3}$ mantissa bits, so a naive cast of the attention inputs would lose precision badly on any outlier channel. FA3 controls this with two tricks: per-block scaling (each tile carries its own scale factor, so the quantization range tracks the local magnitudes) and *incoherent processing* — multiplying $Q$ and $K$ by a random orthogonal (Hadamard) matrix before quantizing, which spreads the energy of any single outlier dimension across all dimensions so no quantization bucket has to cover a huge dynamic range. Because the transform is orthogonal it cancels inside the $QK^\top$ product, leaving the scores exact up to FP8 rounding. The reported result is about ${2.6}\times$ lower RMS error than plain FP8 attention. The engineering takeaway: FP8 attention is a real ${\sim}1.6\times$ on Hopper prefill, but it is a *numerical* change — gate it behind a perplexity and task-accuracy check on your own model, because the error, small in aggregate, can concentrate on exactly the rare tokens you care about most.

## Prefill vs decode: two workloads, two kernels

Here is the fork in the road that shapes every serving attention kernel. A transformer serves in two phases, and they hit opposite ends of the roofline.

**Prefill** processes the whole prompt at once. For a prompt of $N$ tokens, attention is the full $N \times N$ computation — a big, batched matmul-heavy operation with high arithmetic intensity. Prefill is **compute-bound**: it can saturate the tensor cores, and it is exactly what FlashAttention-2/3's high-throughput kernels are optimized for. A longer prompt means quadratically more attention work, and the kernel's job is to keep the tensor cores at 70%+ utilization.

**Decode** generates one token at a time. At each step there is exactly *one* new query row, which must attend over the entire KV cache accumulated so far — potentially tens of thousands of keys and values. This is a matrix-vector product (GEMV), not a matrix-matrix product: a single query against a long $K$ and $V$. The arithmetic per step is tiny, but you must *read the entire KV cache from HBM* to do it. Decode is therefore **memory-bound**, and its speed is set by HBM bandwidth, not tensor-core throughput. The FlashAttention-2 kernel tuned for prefill is a poor fit here — it assumes enough query rows to fill the machine, and one query row leaves the GPU almost idle.

The figure shows the split: a batched attention step forks into a compute-bound prefill path and a memory-bound decode path, each dispatched to a different kernel, converging only at the token sampler.

![Graph showing an attention step splitting into a compute-bound prefill path with an S by S batched GEMM dispatched to an FA3 kernel and a memory-bound decode path with a single token over a 32k paged KV dispatched to a paged split-KV kernel, both reaching the token sampler](/imgs/blogs/attention-backends-deep-dive-flashattention-flashinfer-4.webp)

This is why "which attention backend" is really "which attention *kernels*," plural. A serving engine like vLLM or SGLang selects a prefill kernel and a decode kernel, and the best choices can differ. FlashAttention-3 is often the prefill winner on Hopper; FlashInfer's paged decode kernel is often the decode winner for real serving workloads, which we will get to. The [continuous batching and PagedAttention](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention) post covers how the scheduler mixes prefill and decode steps into one batch; here we care about the kernels each phase runs.

#### Worked example: the arithmetic-intensity gulf between prefill and decode

The prefill/decode split is a roofline story, and the numbers make it stark. Arithmetic intensity is FLOPs per byte of HBM traffic; the H100 ridge point is about ${295}$ FLOP/byte, so anything below that is memory-bound and anything above is compute-bound.

*Prefill*, one layer, batch ${1}$, $N = 8192$, $d = 128$, ${32}$ query heads: attention does about $2 \times 2 \times \frac{N^2}{2} \times \text{heads} \times d \approx 5.5 \times 10^{11}$ FLOP (the factor $\frac{1}{2}$ is causal masking), while its dominant, un-hideable traffic is on the order of the ${8}$ MiB of $Q,K,V,O$. Even charging it a generous ${256}$ MiB of realized traffic including re-reads, that is $5.5\times10^{11} / (256\times10^6) \approx 2100$ FLOP/byte — far *above* the ridge point. Prefill is compute-bound, and a good kernel runs it near the tensor-core peak.

*Decode*, one layer, batch ${1}$, one new token against an ${8192}$-token KV cache, GQA with ${8}$ KV heads: attention does about $2 \times 2 \times 8192 \times 8 \times 128 \approx 3.4 \times 10^{7}$ FLOP, but must read the entire per-layer KV slice — $8192 \times 2 \times 8 \times 128 \times 2$ bytes $\approx 32$ MiB. That is $3.4\times10^7 / (32\times10^6) \approx 1$ FLOP/byte — three orders of magnitude *below* the ridge point. Decode is deeply memory-bound. Same operation, same layer, opposite regime, purely because prefill has $N$ query rows to amortize each KV read while decode has exactly one.

The one lever that moves decode's intensity is **batch size with sharing**. Batch ${B}$ decode requests that share nothing and each still reads its own KV, so intensity stays ~${1}$; but batch ${B}$ requests that share a prefix (a system prompt, few-shot examples) and the shared KV is read once for all ${B}$ queries, so intensity rises toward $B$. That is the whole economic argument for prefix caching and for [continuous batching](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention): decode is memory-bound per request, so the only way to feed the tensor cores is to make more query rows share each byte of KV you read.

**Chunked prefill** is the scheduling trick that ties the two phases together. Rather than run a long prompt's prefill as one giant compute-bound kernel that blocks every decode step behind it — a latency spike for everyone else in the batch — vLLM's V1 scheduler splits the prefill into fixed-size token chunks and interleaves those chunks with decode steps in the same batch. The attention kernel then sees a *mixed* batch: some sequences contributing a chunk of many prefill tokens, others contributing a single decode token — which is exactly the ragged, heterogeneous workload that FlashInfer's scheduler (next) is built to balance. Chunked prefill is why the clean "two kernels" picture blurs in production: one fused kernel launch often serves both phases at once, and the backend has to be good at both, not just the phase you had in mind when you pinned it.

The whole path, from a batched request down to the tensor cores, is a stack of layers the serving engine manages for you. The backend-dispatch layer near the top is where the prefill/decode kernel choice is made per GPU and dtype; the layers below it are where the Hopper-specific machinery (warp groups, TMA async copies) lives.

![Stack diagram of the attention execution path from batched requests down through backend dispatch, the attention kernel, warp groups with TMA async copy to SRAM, tensor cores and SRAM, and HBM at 3.35 TB per second](/imgs/blogs/attention-backends-deep-dive-flashattention-flashinfer-5.webp)

## Paged-KV decode attention: the memory-bound kernel

Decode has a second problem beyond being memory-bound: *where the KV cache lives*. In the naive layout, you reserve a contiguous buffer sized to the maximum sequence length for each sequence in the batch. A request that will generate 200 tokens but *could* generate 32,768 reserves the full 32k up front, and most of that reservation is wasted. Worse, contiguous per-sequence buffers fragment GPU memory, so you cannot pack as many concurrent sequences as the raw capacity would allow. That fragmentation directly caps your batch size, and batch size is what amortizes the weight reads that dominate decode cost.

PagedAttention (the vLLM innovation) fixes this by storing the KV cache in fixed-size blocks — typically 16 tokens each — that need not be contiguous. A per-sequence *block table* maps logical token positions to physical blocks, exactly like virtual memory maps pages. The paged-attention kernel must therefore gather its $K$ and $V$ tiles through the block table rather than reading a contiguous stride, and it runs the same online-softmax accumulation over the gathered blocks. The figure contrasts the two.

![Before-after comparison of contiguous KV decode reserving one 4 GB tensor with about 40 percent waste versus paged-KV decode using 16-token blocks gathered through a block table with split-KV parallelism, tripling batch size at under 4 percent waste](/imgs/blogs/attention-backends-deep-dive-flashattention-flashinfer-7.webp)

The paged kernel has one more trick it must pull off, and it is the key to decode performance: **split-KV**, also called *flash-decoding*. A single decode query row cannot fill 132 SMs — there simply is not enough parallel work in one GEMV. So the kernel splits the long KV dimension into chunks, assigns each chunk to a different thread block (a different SM), and each block computes a *partial* attention output over its chunk: a partial max $m$, partial denominator $\ell$, and partial output $o$. A second, cheap reduction kernel then combines the partials using the very same online-softmax rescaling — because online softmax is associative, you can merge partial states just as you merged blocks within one thread. Splitting the KV dimension turns a latency-bound, low-occupancy GEMV into a bandwidth-bound operation that many SMs attack in parallel, which is exactly what you want when the bottleneck is reading bytes from HBM.

The number of KV splits is a tuning knob with a real trade-off. More splits means more SMs working in parallel — lower latency for a single sequence — but also more partial states to reduce and more redundant loads of the query, so past a point the reduction overhead and extra launch cost eat the parallelism gain. Flash-decoding and FlashInfer both pick the split count from the batch shape: few or no splits when the batch already has enough sequences to fill the machine, many splits when a handful of long sequences would otherwise leave most SMs idle. This is the decode analogue of FA2's "parallelize over sequence length" insight — when there are not enough query rows to occupy the GPU, manufacture parallelism by splitting the KV dimension instead.

Structurally the paged split-KV kernel is two kernels. The first (the "split" or "partial" kernel) launches a grid of (sequence × KV-chunk × head), and each block gathers its chunk of $K,V$ through the block table, runs the online-softmax accumulation over just that chunk, and writes a partial $(m, \ell, o)$ to a scratch buffer. The second (the "reduce" or "combine" kernel) reads the partials for each (sequence × head) and merges them with the associative rule from earlier. That is the whole flash-decoding structure, and it is why the merge property mattered:

```python
# Sketch of paged split-KV decode (one query, one head). Real kernels fuse this
# onto the GPU; the shape of the computation is what matters here.
def paged_decode_split(q, block_table, kv_cache, page_size, n_splits):
    # 1) SPLIT: each KV chunk -> a partial online-softmax state, in parallel across SMs.
    partials = []
    for chunk in np.array_split(block_table, n_splits):          # block_table: logical->physical pages
        k = gather_pages(kv_cache["k"], chunk, page_size)        # (chunk_len, d) via the block table
        v = gather_pages(kv_cache["v"], chunk, page_size)
        s = (q @ k.T) / np.sqrt(q.shape[-1])
        m = s.max()
        l = np.exp(s - m).sum()
        o = np.exp(s - m) @ v
        partials.append((m, l, o))
    # 2) REDUCE: merge partial states with the exact associative rule (see merge_states above).
    m, l, o = partials[0]
    for pm, pl, po in partials[1:]:
        m, l, o = merge_states(m, l, o, pm, pl, po)
    return o / l                                                 # final attention output for this query
```

The `gather_pages` step is the paged part: the block table turns a sequence's logical token range into a list of physical block ids, and the kernel reads $K,V$ block by block through that indirection instead of a single contiguous stride. The cost of the indirection is one extra pointer chase per block — cheap next to the KV bytes themselves — and the payoff is the near-zero fragmentation and 16-token allocation granularity that lets you pack far more concurrent sequences into the same HBM.

#### Worked example: decode paged-attention over a 32k KV cache

Llama-3-8B on an H100: 32 layers, 32 query heads, 8 KV heads (grouped-query attention, more on that next), $d = 128$, fp16. The KV cache per token per layer is $2 \times 8 \times 128 \times 2 = 4096$ bytes = 4 KiB (factor of 2 for K and V). Across 32 layers that is 128 KiB per token. For a 32,768-token context, one sequence's KV cache is $32768 \times 128\text{ KiB} = 4$ GiB — the "4 GB reserved" on the figure above.

Each decode step must read that entire 4 GiB once (all layers, all heads) to attend. At the H100's 3.35 TB/s peak, that is $4 \times 10^9 / 3.35 \times 10^{12} = 1.19$ ms of pure KV reads per generated token — a hard floor on time-per-output-token (TPOT) for a single 32k-context sequence, before you count anything else. The attention *compute* for that step is about $4 \times L \times n_\text{heads} \times d \approx 17$ GFLOP across all layers, which at 989 TFLOP/s is only ~17 µs. Compute is ~17 µs, memory is ~1.19 ms: a ~70× ratio confirming decode is deeply memory-bound, with an arithmetic intensity of about 4 FLOP/byte — far below the ridge point. Per layer, the paged-decode kernel reads $32768 \times 4\text{ KiB} = 128$ MiB; at a realistic ~50–60% of peak HBM bandwidth a well-tuned kernel lands around ~70–80 µs per layer (the "~80 µs/step" on the figures is this per-layer kernel time). The lesson: at long context, decode latency is dominated by KV-cache bandwidth, split-KV is what lets you approach the bandwidth floor, and shrinking the KV cache (via GQA, quantized KV, or MLA) attacks the floor itself. See [KV-cache optimization](/blog/machine-learning/model-serving/kv-cache-optimization) for the cache-management side of this.

## GQA, MQA, and MLA: how head structure reshapes the kernel

The head structure of the model changes the attention kernel's tiling and its memory traffic, and modern models are designed specifically to make decode cheaper. Three variants matter.

**Multi-head attention (MHA)** is the original: every query head has its own K and V head. The KV cache is $2 \times n_\text{heads} \times d$ per token — the largest and most bandwidth-hungry option.

**Multi-query attention (MQA)** and **grouped-query attention (GQA)** shrink the KV side. MQA uses a single KV head shared by all query heads; GQA uses a small number of KV heads, each shared by a *group* of query heads (Llama-3-8B uses 8 KV heads for 32 query heads, a 4:1 group ratio). Fewer KV heads means a proportionally smaller KV cache — the 4:1 ratio is why our 32k example above was 4 GiB rather than 16 GiB — and therefore proportionally less HBM traffic per decode step. For the kernel, GQA changes the tiling: instead of one query head against one KV head, you load a KV tile *once* and reuse it across the whole group of query heads that share it, packing the query group into the "rows" of the tile. This raises the kernel's arithmetic intensity (more FLOPs per byte of K/V read) and is why FlashAttention and FlashInfer both ship GQA-aware kernels that batch the query group rather than looping over query heads.

**Multi-head latent attention (MLA)**, introduced by DeepSeek, is the most aggressive and the most kernel-disruptive. Instead of caching per-head K and V, MLA caches a single low-rank *latent* vector per token (for DeepSeek-V2/V3, a 512-dimensional compressed latent plus a small 64-dimensional decoupled RoPE part, so 576 dimensions total), and reconstructs the per-head K and V via learned up-projection matrices. The cache shrinks dramatically — roughly an order of magnitude smaller than MHA. The critical kernel trick is **projection absorption**: during decode you do not actually up-project the latent back to full K and V (that would defeat the memory savings). Instead the up-projection matrices are algebraically folded ("absorbed") into the query and output projections, so attention is computed *directly in the compressed latent space*. The effect is that MLA decode behaves like MQA over an unusually large head dimension (576), which needs a specialized kernel — **FlashMLA**, a Hopper-optimized kernel DeepSeek open-sourced that handles the 576-dim head and variable-length paged latent KV. This is why vLLM has a separate family of MLA backends (`FLASHMLA`, `TRITON_MLA`, `CUTLASS_MLA`, `FLASHINFER_MLA`) distinct from the dense-attention backends: the math is different enough that the dense kernels do not apply. If you serve DeepSeek models, the MLA decode backend choice is the single biggest attention-performance lever you have.

#### Worked example: KV-cache bytes per token across head structures

Head structure is, above everything else, a KV-cache-size decision, and the KV cache is what sets decode's memory floor. Take a ${70}$B-class model with ${64}$ query heads, $d = 128$, ${80}$ layers, fp16, and compare the per-token KV cache under each scheme (both $K$ and $V$, all layers):

| Scheme | KV heads | Bytes/token/layer | Bytes/token (80 layers) | Relative to MHA |
|---|---|---|---|---|
| MHA | 64 | $2\times64\times128\times2 = 32$ KiB | 2.56 MiB | 1.0× |
| GQA (8:1) | 8 | $2\times8\times128\times2 = 4$ KiB | 320 KiB | 0.125× |
| MQA | 1 | $2\times1\times128\times2 = 512$ B | 40 KiB | 0.016× |
| MLA (576-dim latent) | — | $576\times2 = 1152$ B | 90 KiB | 0.035× |

At a ${32}$k context, MHA would demand $32768 \times 2.56$ MiB $\approx 82$ GiB of KV cache for a *single* sequence — more than an H100's ${80}$ GB of HBM, before you load one byte of weights. GQA at 8:1 brings that to ~${10}$ GiB; MLA to ~${2.8}$ GiB. This is not a micro-optimization; it is the difference between "this context length is servable" and "it is not." Every byte you cut from the per-token KV cache both raises decode's arithmetic intensity (fewer bytes read per token generated) and frees HBM for a larger batch, which is the other way to raise intensity. GQA is the near-universal default now precisely because 8:1 captures most of the memory win with almost no quality loss; MLA pushes an order of magnitude further at the cost of a bespoke kernel.

For the kernel, the tiling consequence is direct. Under MHA the kernel loops one query head against one KV head. Under GQA it loads a KV tile once and runs the whole group of query heads that share it against that resident tile — turning what would be $G$ separate small matmuls (for group size $G$) into one wider matmul, which both raises arithmetic intensity and cuts KV bandwidth by the group factor. This is why a GQA-unaware kernel — one that redundantly re-loads the shared KV per query head — leaves a large fraction of GQA's benefit on the floor, and why FlashAttention and FlashInfer both ship group-packed tilings that put the query group in the rows of the tile. Under MLA the "head" is a single ${576}$-dimensional absorbed latent, so the kernel looks like MQA over an unusually wide head — a shape neither the MHA nor the GQA kernel handles well, which is precisely the gap FlashMLA fills.

## Choosing and setting the backend in vLLM

Now the practical core: how vLLM picks an attention backend, and when you should override it. vLLM (and SGLang, similarly) maintains a *priority-ordered* list of backends per GPU compute capability. At startup, for each of prefill and decode, it walks the list and selects the first backend that is compatible with your model's dtype, head dimension, and KV-cache configuration. The relevant environment variable is `VLLM_ATTENTION_BACKEND`; the current backend names include `FLASH_ATTN`, `FLASHINFER`, `TRITON_ATTN`, `FLEX_ATTENTION`, and the legacy `XFORMERS`, plus the MLA family `FLASHMLA`, `TRITON_MLA`, `CUTLASS_MLA`, and `FLASHINFER_MLA`. (Exact spellings drift across vLLM releases — always check `vllm.attention.backends` or the design docs for your pinned version rather than trusting a runbook.)

The auto-selection priority, roughly: on **Ampere and Hopper** (sm80–sm90), the order is `FLASH_ATTN` first, then `FLASHINFER`, then `TRITON_ATTN`. On **Blackwell** (sm100), `FLASHINFER` is prioritized *ahead* of `FLASH_ATTN`, because FlashInfer's Blackwell kernels are further along. For MLA models the engine routes to the MLA backends entirely. The capability matrix below summarizes which backend wins where; it is the same information the auto-selector encodes, made explicit so you can sanity-check its choice.

![Matrix comparing FlashAttention-2, FlashAttention-3, FlashInfer, xFormers, and FlashMLA across best phase, paged KV support, FP8 attention, GPU and dtype requirements, and when to pick each](/imgs/blogs/attention-backends-deep-dive-flashattention-flashinfer-6.webp)

| Backend | Best phase | Paged KV | FP8 attention | GPU / dtype | When to pick it |
|---|---|---|---|---|---|
| FlashAttention-2 | prefill + decode | via engine wrapper | no (fp16/bf16) | Ampere+ (sm80) | A100, general dense models |
| FlashAttention-3 | prefill (Hopper) | yes | yes (~1.2 PFLOP/s) | H100 only (sm90) | H100, maximum prefill throughput |
| FlashInfer | decode + ragged batches | yes (BSR unified) | yes | sm80–sm100 | serving with dynamic/variable lengths |
| Triton attention | prefill + decode | yes | partial | sm80+ (portable) | NVIDIA/AMD/Intel portability, fallback |
| xFormers | prefill (memory-efficient) | limited | no | sm70+ (broad) | fallback / older GPUs |
| FlashMLA | MLA decode | paged latent KV | yes (Hopper) | H100, DeepSeek models | MLA models (DeepSeek-V2/V3) |
| FlashInfer-MLA | MLA decode | paged latent KV | yes | sm90+ | MLA under the FlashInfer engine |

Setting the backend is a one-liner, but *how* you set it matters. Prefer the explicit flag over an environment variable buried in a container image, and log the resolved backend so a silent fallback never hides:

```bash
# As an environment variable (affects the whole process; simplest for a container):
export VLLM_ATTENTION_BACKEND=FLASHINFER
vllm serve meta-llama/Llama-3.1-70B-Instruct --tensor-parallel-size 4

# Or, on newer vLLM, as an explicit engine flag (V1 engine):
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 --attention-backend FLASHINFER
```

```python
import os
# Set BEFORE importing vllm — the backend is resolved at engine construction.
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"

from vllm import LLM, SamplingParams
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", max_model_len=32768)
# vLLM logs the selected backend at startup, e.g.
#   "Using FlashInfer backend."  <-- confirm this matches what you set.
out = llm.generate(["Explain online softmax."], SamplingParams(max_tokens=128))
```

My standing advice: **do not override the auto-selection unless you have measured a reason to.** The default is chosen by people who benchmark these kernels for a living, per GPU generation, and it is right the large majority of the time. Override when (a) you have a specific workload — very ragged batches, or MLA models — where a different backend measurably wins, (b) you hit a correctness or stability bug in the default and need a known-good fallback, or (c) you are running on hardware where the default's kernels are immature. Every other pin is a future 40%-regression waiting to be discovered in a capacity review, exactly as in the opening story.

A few operational specifics that save real debugging time. First, **the backend is resolved once, at engine construction**, from the environment variable or flag, the GPU compute capability, the model's head dim and dtype, and whether the KV cache is FP8. Setting `VLLM_ATTENTION_BACKEND` after the engine is built does nothing. Second, **prefill and decode can resolve to different backends** — vLLM may pick an FA3 prefill kernel and a paged decode kernel from the same or a different backend — so "the backend" is really a pair, and a benchmark that only stresses one phase can hide a bad choice in the other. Third, **an incompatible pin does not always error loudly**; depending on the release it may fall back to a slower compatible backend with only a log line to tell you. That log line is the thing to assert on in a startup smoke test:

```python
import os
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"   # must be set BEFORE importing vllm

from vllm import LLM
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", max_model_len=32768)

# Confirm what actually resolved. The exact attribute path drifts across releases,
# so the portable check is to grep the startup log for the "Using ... backend" line;
# programmatically, the platform selector is the source of truth for capability:
from vllm.platforms import current_platform
print("device capability:", current_platform.get_device_capability())  # (9, 0) => Hopper
# In a deploy smoke test, fail hard if the resolved backend is not the one you pinned,
# rather than discovering a silent fallback in a capacity review six months later.
```

The chunked-prefill interaction is the subtle one. When chunked prefill is on (the V1 default for many configs), a single batch mixes prefill chunks and decode tokens, so the backend must handle a *unified* attention call over ragged inputs — one reason FlashInfer, built for ragged batches, is increasingly the default on newer hardware, and why a decode-only or prefill-only kernel is not sufficient on its own. If you pin a backend that handles one phase well but the other poorly, chunked prefill surfaces that weakness on every mixed step, not just on the phase you were thinking about when you set the pin. The failure mode to internalize from all of this: the dangerous state is not "running the default," it is "running a pin that was correct on last year's GPU." Put the resolved backend in your startup logs, assert on it in a smoke test, and re-run that assertion on every hardware or vLLM-version migration.

## FlashInfer: the serving-optimized attention engine

FlashAttention is a kernel library optimized for the attention *operation*. FlashInfer (Ye et al., MLSys 2025 best paper) is an attention *engine* optimized for the attention *serving workload* — which is a different and harder problem, because serving means ragged batches, mixed prefill and decode, paged and shared KV caches, and dynamic request lengths, all while staying compatible with CUDA graphs. Three ideas make it the decode-and-serving winner in vLLM and SGLang.

First, **a unified block-sparse representation**. FlashInfer observes that every KV-cache layout a serving system uses — contiguous, paged, radix-tree prefix sharing, tree-attention masks for speculative decoding — is a special case of a block-sparse-row (BSR) matrix. By representing them all in one BSR format with composable block sizes, FlashInfer serves the whole zoo of layouts with one family of kernels instead of a bespoke kernel per layout. This is why it slots so cleanly under vLLM's paged cache and SGLang's [RadixAttention prefix cache](/blog/machine-learning/model-serving/kv-cache-optimization).

Second, **JIT-compiled customizable templates**. Rather than shipping a fixed kernel, FlashInfer exposes an attention template (customizable for the mask, the position encoding, the group size, the dtype) and JIT-compiles a specialized kernel for the exact configuration at hand. You get a kernel tuned to your head dim and mask without the library maintainer having pre-written it.

Third, and most important for serving, **load-balanced scheduling**. In a real batch, sequence lengths vary wildly — one request has 200 KV tokens, another has 30,000. A naive kernel assigns one thread block per sequence and the long sequences straggle while the short ones' SMs sit idle, so the batch runs at the speed of its longest member. FlashInfer's scheduler plans a work decomposition (how to split each sequence's KV across thread blocks, split-KV style) that balances the load *across the whole batch* — and it does this in a way compatible with CUDA graphs, which require a static launch configuration even though the request mix is dynamic. That combination — dynamic balancing under a static graph — is the hard part, and it is why FlashInfer's decode throughput on ragged production batches often beats FlashAttention's even on the same H100.

Here is a paged-decode call through FlashInfer's Python API. The two-phase `plan`/`run` structure is exactly the load-balanced scheduling: `plan` inspects the batch's ragged structure and computes the work decomposition once; `run` executes the balanced kernel, and can be replayed under a CUDA graph.

```python
import torch
import flashinfer

num_qo_heads, num_kv_heads, head_dim = 32, 8, 128   # GQA 4:1, like Llama-3-8B
page_size = 16                                       # 16 tokens per KV block

# Ragged batch: three sequences with very different KV lengths.
kv_lens = torch.tensor([200, 4096, 30000], dtype=torch.int32)

# Paged KV cache laid out as [num_pages, 2, page_size, num_kv_heads, head_dim].
# kv_page_indices / kv_page_indptr / kv_last_page_len describe each sequence's pages
# (the block table), exactly as vLLM's BlockSpaceManager tracks them.
workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD")

# plan(): inspect the ragged batch once and compute a load-balanced split-KV schedule.
decode_wrapper.plan(
    kv_page_indptr, kv_page_indices, kv_last_page_len,
    num_qo_heads, num_kv_heads, head_dim, page_size,
    data_type=torch.bfloat16,
)
# run(): one decode query per sequence attends over its paged KV; CUDA-graph replayable.
q = torch.randn(len(kv_lens), num_qo_heads, head_dim, device="cuda", dtype=torch.bfloat16)
out = decode_wrapper.run(q, paged_kv_cache)   # (batch, num_qo_heads, head_dim)
```

You almost never write this by hand in production — vLLM and SGLang call it for you when you select the FlashInfer backend — but seeing the `plan`/`run` split makes concrete *why* FlashInfer is a serving engine and not just a kernel: the scheduling intelligence lives in `plan`, and that is the part FlashAttention does not have.

Two more FlashInfer capabilities matter for real serving and are worth knowing by name. **Cascade attention** handles the shared-prefix case directly: when many requests in a batch share a long common prefix — a system prompt, a few-shot preamble — FlashInfer computes attention over the shared prefix once, as a single dense operation across all queries, and attention over each request's unique suffix separately, then merges the two with the same online-softmax combine used in split-KV. This turns a redundant per-request re-read of the shared prefix into one amortized read, a large win for high-fan-out prompts. **Prefill-append** (the `BatchPrefillWithPagedKVCacheWrapper`) handles chunked prefill: appending a chunk of new query tokens to an existing paged KV cache, the exact operation chunked prefill needs. The same `plan`/`run` split applies:

```python
import flashinfer

# Prefill/append wrapper: a chunk of new tokens attends over the existing paged KV + itself.
prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")
prefill_wrapper.plan(                       # inspect the ragged chunk lengths once
    qo_indptr, kv_page_indptr, kv_page_indices, kv_last_page_len,
    num_qo_heads, num_kv_heads, head_dim, page_size,
    causal=True,                            # causal mask within the appended chunk
)
out = prefill_wrapper.run(q_chunk, paged_kv_cache)   # CUDA-graph replayable, like decode
```

The pattern across all three wrappers — decode, prefill-append, cascade — is identical: a `plan` phase that inspects the batch's ragged structure and computes a balanced, CUDA-graph-static schedule, and a `run` phase that executes it. That uniform interface over a single block-sparse representation is what lets one engine serve contiguous, paged, prefix-shared, and tree-masked (speculative-decoding) layouts without a bespoke kernel for each — the practical payoff of the block-sparse framing, and the reason NVIDIA now co-develops FlashInfer rather than treating it as an external add-on.

## Benchmarking two backends against each other

You should never take a backend claim — including the ones in this post — on faith for your workload. The right tool is `triton.testing.do_bench`, which handles warmup, cache flushing, and robust timing (it reports the median, not a noisy single run). Here is a harness that compares two attention paths across sequence lengths, which is exactly the sweep you want because backend rankings *cross over* as sequence length changes.

```python
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from triton.testing import do_bench

def bench_attention(seqlens, batch=8, heads=32, head_dim=128, dtype=torch.bfloat16):
    rows = []
    for N in seqlens:
        q, k, v = (torch.randn(batch, heads, N, head_dim, device="cuda", dtype=dtype)
                   for _ in range(3))

        def flash():
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                return F.scaled_dot_product_attention(q, k, v, is_causal=True)

        def mem_efficient():   # xFormers-style memory-efficient kernel, for contrast
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                return F.scaled_dot_product_attention(q, k, v, is_causal=True)

        t_flash = do_bench(flash)          # median ms over many runs, warmup handled
        t_mem   = do_bench(mem_efficient)
        # attention FLOPs (causal): ~2 * 2 * (N^2/2) * heads * head_dim * batch
        flops = 2 * 2 * (N * N / 2) * heads * head_dim * batch
        rows.append((N, t_flash, t_mem,
                     flops / (t_flash * 1e-3) / 1e12,   # TFLOP/s for the flash kernel
                     t_mem / t_flash))                  # speedup of flash over mem-efficient
        print(f"N={N:6d}  flash={t_flash:7.3f}ms  mem_eff={t_mem:7.3f}ms  "
              f"flash={rows[-1][3]:6.1f} TFLOP/s  speedup={rows[-1][4]:.2f}x")
    return rows

bench_attention([512, 2048, 8192, 32768])
```

Two things this harness teaches you that a single benchmark never will. First, the crossover: at short sequences the kernels are launch-bound and nearly tie; the FlashAttention advantage widens with $N$ because the quadratic score-matrix traffic it avoids grows quadratically. Second, always report throughput in the units of the decision you are making — TFLOP/s for a compute-bound prefill sweep, but tokens/s and GB/s of achieved HBM bandwidth for a decode sweep, because a decode kernel at "low TFLOP/s" might be perfectly optimal if it is saturating memory bandwidth. Measuring decode with a prefill metric is how people talk themselves into the wrong backend.

## Measurement: backends across phase, sequence length, and GPU

The table below is a representative picture assembled from the FlashAttention-2/3 papers, the FlashInfer paper, and public vLLM backend benchmarks. Treat the numbers as *representative orders of magnitude on the named hardware*, not as a guarantee for your model — kernel performance moves with every library release, and your head dim, group size, and batch shape shift the ranking. The point of the table is the *shape* of the answer: where each backend wins, and by roughly how much.

| Phase | Seq len | GPU | Backend | Throughput | Notes |
|---|---|---|---|---|---|
| Prefill | 8k | A100 80GB | FlashAttention-2 | ~200–225 TFLOP/s | ~72% of fp16 peak; the sm80 workhorse |
| Prefill | 8k | H100 SXM | FlashAttention-2 | ~450–550 TFLOP/s | works, but leaves Hopper features unused |
| Prefill | 8k | H100 SXM | FlashAttention-3 (fp16) | ~650–740 TFLOP/s | ~1.5–2× over FA2; ~75% of fp16 peak |
| Prefill | 8k | H100 SXM | FlashAttention-3 (FP8) | ~0.9–1.2 PFLOP/s | validate accuracy before shipping |
| Decode | 8k | A100 80GB | FlashAttention (paged) | ~1.6–1.9 TB/s HBM used | bandwidth-bound; report GB/s, not TFLOP/s |
| Decode | 32k, ragged | H100 SXM | FlashAttention (paged) | baseline | bandwidth-bound; ~70–80 µs/layer |
| Decode | 32k, ragged | H100 SXM | FlashInfer (paged) | ~1.1–1.3× vs FA on ragged | load-balanced scheduling wins the tail |
| Decode | 32k, ragged | any | Triton attention | ~0.8–0.95× vs FA | trades peak for NVIDIA/AMD/Intel portability |
| Decode | 32k | H100 SXM | FlashMLA (DeepSeek MLA) | ~10× smaller KV traffic | different math; MLA models only |

The two rows worth internalizing: on Hopper prefill, moving from FA2 to FA3 is a free ~1.5–2× if your library exposes it; on ragged decode, FlashInfer's scheduler beats a plain FlashAttention decode kernel by a tail-latency margin that grows with how uneven your batch is. Neither win requires touching your model — they are pure kernel-selection dividends.

#### Worked example: what the opening regression actually cost

Back to the story that opened this post. The fleet served Llama-3-70B on H100s with `XFORMERS` pinned. xFormers' memory-efficient attention is a fine kernel — it is the "broad fallback" row of the matrix — but it is an sm80-era design: no Hopper TMA async pipeline, no warp-specialized producer-consumer overlap, no FP8, and a weaker paged-decode path than FlashInfer. On H100 prefill at 8k context, that is the difference between roughly ~500 TFLOP/s (a good sm80-style kernel) and ~700 TFLOP/s (FA3) — about a 40% prefill throughput gap, matching what our capacity review found. At a fleet cost on the order of tens of thousands of dollars per month, "delete two nodes" translated to a five-figure annual saving from a one-line change. The lesson is not "always use FA3" — it is "never let a stale backend pin outlive the hardware it was written for," and "re-benchmark the default on every GPU migration."

## Case studies and benchmarks

**FlashAttention-1/2/3 (Dao et al., 2022–2024).** The three papers are the primary source for every throughput claim here. FA1 established the $O(N^2 d^2 / M)$ I/O bound and the tiling algorithm; FA2 reported ~2× over FA1 and up to ~225 TFLOP/s on A100 (72% MFU) from better parallelism and work partitioning; FA3 reported ~1.5–2× over FA2 on H100, up to ~740 TFLOP/s fp16 (75%) and ~1.2 PFLOP/s FP8, from Hopper asynchrony (warp specialization + TMA) and FP8 with incoherent processing. Read FA3 for the clearest modern statement of why non-matmul FLOPs and data-movement overlap dominate a well-tuned attention kernel.

**FlashInfer (Ye et al., MLSys 2025, best paper).** The paper's contribution is the *serving* framing: a block-sparse unified KV representation, JIT-customizable templates, and load-balanced scheduling that stays CUDA-graph-compatible. The reported wins are concentrated exactly where you would expect from the design — inter-token latency reductions on the order of tens of percent versus prior serving kernels on skewed, ragged batches, and near-flat performance as the batch's length distribution gets more uneven, which is the property a load-balanced scheduler is supposed to buy. It has been adopted into vLLM, SGLang, and MLC-Engine, and NVIDIA now co-develops it. If you serve dynamic, ragged workloads, this is the paper that explains why a decode kernel's *scheduler* matters as much as its inner loop — the inner loop sets your best case, but the scheduler sets your tail, and in serving the tail is what pages you.

**FlashMLA (DeepSeek, 2025).** DeepSeek open-sourced a Hopper-optimized MLA decode kernel alongside DeepSeek-V3. It handles the 576-dimensional absorbed-latent head and variable-length paged latent KV, and reported very high bandwidth utilization on H800/H100. It is the reference for why MLA models need their own backend family rather than reusing dense-attention kernels — the absorbed-projection math changes the kernel's shape.

**vLLM backend benchmarks (2024–2026).** vLLM's own blog posts and issue tracker are a running, honest record of backend crossovers: FlashInfer beating FlashAttention on some decode workloads and losing on others; the Triton attention backend trading peak throughput for portability across NVIDIA, AMD, and Intel; and the migration to per-GPU priority ordering in the V1 engine. The recurring theme is that no backend is a universal winner, which is exactly why the auto-selector is per-GPU and per-dtype — and why you benchmark rather than assume.

## When to use this (and when not to)

The decisive version. First, the meta-rule: **the auto-selected default is right most of the time — measure before you override.** The figure encodes the decision as a two-question filter (which GPU generation, then which model and dtype), which is the same tree vLLM's selector walks internally.

![Decision tree for picking an attention backend: root splits by GPU into H100 Hopper, A100 Ampere, and B200 Blackwell, then leaves recommend FlashMLA, FA3, FLASH_ATTN, and FlashInfer by model and dtype](/imgs/blogs/attention-backends-deep-dive-flashattention-flashinfer-8.webp)

**Use FlashAttention-3 when** you are on H100/H200, prefill-heavy, and want maximum throughput — and especially when you can validate FP8 attention for another near-doubling. Do *not* reach for it on A100: it is Hopper-only and there is no sm80 path.

**Use FlashInfer when** you serve dynamic, ragged production traffic — highly variable sequence lengths, mixed prefill and decode, prefix sharing — where its load-balanced scheduling wins the tail latency. It is also the sensible default on Blackwell today. Do *not* assume it beats FlashAttention on uniform, batch-1, fixed-length microbenchmarks — on those it often ties or slightly loses, because its scheduling advantage has nothing to balance.

**Use FlashMLA (or the MLA backend family) when and only when** your model uses multi-head latent attention (DeepSeek-V2/V3 and derivatives). For any non-MLA model these backends do not apply.

**Use xFormers or the Triton backend when** you need broad hardware compatibility (older GPUs, or a single kernel across NVIDIA/AMD/Intel) more than you need peak throughput, or as a known-good fallback while you debug a correctness issue in a faster backend. Do *not* leave xFormers pinned on Hopper — that is the exact mistake this post opened with.

**Do not hand-pick a backend at all when** the auto-selection already resolves to the right one — which, on a current vLLM release matched to your GPU, it usually does. The failure mode is not "using the default"; it is "pinning a non-default in a runbook and never revisiting it." And do not tune attention kernels while your real bottleneck is elsewhere: if your service is scheduler-bound or network-bound, a faster attention kernel buys you nothing. Profile first ([roofline analysis](/blog/machine-learning/model-serving/roofline-analysis-for-llm-inference) tells you which wall you are hitting) before you reach for a kernel change.

## Key takeaways

- **Attention is special because of the $N \times N$ score matrix you must never materialize.** The whole game is refusing to write $S$ to HBM; everything else follows from that.
- **Online softmax makes single-pass exact softmax possible** by carrying a running max, denominator, and output, and rescaling prior state by $e^{m - m^{\text{new}}}$ when the max grows. FlashAttention is exact, not approximate.
- **The I/O bound is $O(N^2 d^2 / M)$, not $O(N^2)$** — a factor $d^2/M \ll 1$ fewer HBM accesses. That derivation, not a FLOP reduction, is why FlashAttention wins.
- **FA2 improved scheduling** (parallelism, work partitioning, fewer non-matmul FLOPs) for ~2× over FA1; **FA3 is Hopper-only** (async TMA, warp specialization, FP8) for another ~1.5–2× and up to ~1.2 PFLOP/s in FP8.
- **Prefill is compute-bound; decode is memory-bound.** They need different kernels: high-throughput matmul kernels for prefill, paged split-KV (flash-decoding) kernels for decode.
- **Paged-KV decode replaces reserved contiguous buffers with gathered 16-token blocks**, cutting waste and roughly tripling batch size; split-KV parallelizes the memory reads across SMs to approach the bandwidth floor.
- **GQA/MQA shrink the KV cache and raise kernel arithmetic intensity; MLA absorbs projections to attend in a compressed latent space** and needs its own kernel family (FlashMLA).
- **FlashInfer is a serving engine, not just a kernel** — unified block-sparse layouts, JIT templates, and load-balanced scheduling under CUDA graphs — which is why it wins ragged decode in vLLM and SGLang.
- **Set the backend with `VLLM_ATTENTION_BACKEND` (or `--attention-backend`), but trust the per-GPU auto-selection unless you have measured a reason not to.** A stale pin is a silent tax.
- **Benchmark across sequence lengths with `do_bench`**, and report the metric that matches the phase (TFLOP/s for prefill, tokens/s and GB/s for decode).

## Further reading

- Dao, Fu, Ermon, Rudra, Ré, "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (NeurIPS 2022) — the I/O-complexity theorem and tiling algorithm.
- Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" (2023, arXiv:2307.08691) — the scheduling rework.
- Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao, "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision" (2024, arXiv:2407.08608) — Hopper asynchrony and FP8.
- Ye et al., "FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving" (MLSys 2025, best paper) — the serving-engine design.
- Milakov, Gimelshein, "Online normalizer calculation for softmax" (2018, arXiv:1805.02867) — the running-max recurrence FlashAttention builds on.
- DeepSeek-AI, "DeepSeek-V2 / V3 Technical Reports" and the open-source FlashMLA kernel — multi-head latent attention and its decode kernel.
- vLLM documentation, "Attention Backends" design page — the authoritative, version-matched list of backend names and the auto-selection priority.
- Within this series: [custom CUDA kernels for inference](/blog/machine-learning/model-serving/custom-cuda-kernels-for-inference), [KV-cache optimization](/blog/machine-learning/model-serving/kv-cache-optimization), [continuous batching and PagedAttention](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention), [roofline analysis for LLM inference](/blog/machine-learning/model-serving/roofline-analysis-for-llm-inference), [GPU-architecture-specific tuning for LLM serving](/blog/machine-learning/model-serving/gpu-architecture-specific-tuning-for-llm-serving), and [why LLM serving is different](/blog/machine-learning/model-serving/why-llm-serving-is-different).
