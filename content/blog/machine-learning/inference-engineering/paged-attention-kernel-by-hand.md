---
title: "Paged attention kernel by hand: online softmax over scattered blocks"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "Write the decode attention kernel the way a real engine does it: one query attending over a paged KV cache, with online softmax derived from scratch, a split-K variant that fills the card, and a Triton implementation you can diff against PyTorch."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "paged-attention",
    "triton",
    "cuda",
    "kv-cache",
    "flash-attention",
    "gpu",
    "pytorch",
    "ml-systems",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 46
---

At decode time your model produces one token, and to produce it the attention layers read the entire KV cache. Not a slice of it, not a window — all of it, every cached key and every cached value for the sequence, at every one of the 32 layers, on every single step. For an 8k-token conversation on Llama-3.1-8B that is one gigabyte of reads to emit one token, and then you do it again for the next token, and again. The matmul in there is trivial; a single query vector dotted against the keys is a handful of FLOPs per byte. The kernel that does this is not compute-limited and it is not clever. It is a memory-streaming loop, and the only thing that matters is whether it reads the cache at the speed the card can read memory.

Most people never write this kernel. They call `scaled_dot_product_attention`, or they let vLLM's paged-attention CUDA kernel do it, and they are right to — those kernels are excellent and you will not beat them on a Tuesday afternoon. But if you have followed this series into building [`nanoserve`](/blog/machine-learning/inference-engineering/what-inference-engineering-is), you now have a [paged KV cache](/blog/machine-learning/inference-engineering/paged-kv-cache-implementing-blocks-and-a-block-table) — fixed-size blocks, a free pool, a per-request block table — and a [gather kernel that appends into it](/blog/machine-learning/inference-engineering/the-kv-cache-append-and-gather-kernel). The one thing missing is the kernel that reads through the block table and actually computes attention over those scattered blocks. This post writes it. By the end you will have `nanoserve/kernels/paged_attention.py`: a `@triton.jit` decode kernel that gathers through the block table, an online-softmax accumulation that never materializes the score row, a split-K variant that saturates the GPU when a single long sequence cannot, and a test that diffs the whole thing against a PyTorch reference within tolerance.

![Dataflow graph in which one query vector reads scattered physical blocks through a block table and reduces them into a single output vector](/imgs/blogs/paged-attention-kernel-by-hand-1.webp)

This is where two tracks of the series meet. Track B built the memory layout — blocks and a block table — and Track E has been building kernels. The decode attention kernel is the seam: it is the code that reads the paged layout and does the math, and it is the crown-jewel kernel of an inference engine because it runs more often than anything else and it touches the most memory. Get it right and the engine is fast. Get the rescaling wrong and it produces plausible garbage that passes a smoke test and fails a benchmark.

One standing promise, the same one from [the introduction](/blog/machine-learning/inference-engineering/what-inference-engineering-is): **I have no GPU and I have run none of this.** Every number below is derived from arithmetic I show you, cited from a paper or an official post with a link, or framed as something you will reproduce yourself with a named script and an expected range. The results tables carry a `Source` column. The bandwidth ceilings are the easy case — they are arithmetic on published datasheet specs. The achieved fractions are the hard case, and those stay cited or framed as reproduce-it-yourself.

---

## 1. The decode step is one query against the whole cache

Start with what attention computes, stripped to the decode case. During prefill you have many query positions and you compute a full attention matrix. During decode you have exactly one new query position per sequence — the token you are about to generate reads the tokens before it. For one attention head, with query vector $\mathbf{q} \in \mathbb{R}^{d}$ and cached keys and values $\mathbf{k}_1, \dots, \mathbf{k}_S$ and $\mathbf{v}_1, \dots, \mathbf{v}_S$ over a context of length $S$, the output is:

$$
s_j = \frac{\mathbf{q} \cdot \mathbf{k}_j}{\sqrt{d}}, \qquad
p_j = \frac{e^{s_j}}{\sum_{k=1}^{S} e^{s_k}}, \qquad
\mathbf{o} = \sum_{j=1}^{S} p_j \, \mathbf{v}_j
$$

That is the whole computation. A dot product per key to get a score, a softmax over the scores, a weighted sum of the values. The figure above is exactly this, drawn against the paged layout: the query walks the block table, which tells it that logical block 0 lives in physical block 7, logical block 1 in physical block 2, and so on; it gathers the keys and values from those scattered physical blocks; and it reduces them into a single output vector. The block table is the indirection from [the paging post](/blog/machine-learning/inference-engineering/paged-kv-cache-implementing-blocks-and-a-block-table), and the gather is the read half of [the append-and-gather kernel](/blog/machine-learning/inference-engineering/the-kv-cache-append-and-gather-kernel). This kernel is where they get used to do the math.

Two properties of this computation drive everything that follows, and they are worth stating before any code.

**It reads every cached token, once, per step.** There is no way around it: token $S+1$ attends to all $S$ prior tokens, so all $S$ keys and all $S$ values must be read. As the sequence grows the read grows linearly. This is why the KV cache is the memory hog of LLM inference, and it is why a decode step gets slower as the conversation gets longer even though you are always emitting exactly one token.

**The arithmetic per byte read is tiny.** Each key contributes $d$ multiply-adds to its score; each value contributes $d$ multiply-adds to the output. That is a constant amount of work per element read, and it is a small constant. A single query dotted against a matrix of keys is a matrix-vector product — a GEMV — and a GEMV is the canonical memory-bound operation. Contrast prefill, where a whole block of query rows shares the same keys, turning the operation into a GEMM that reuses each loaded key many times. Decode has no such reuse for a single sequence. That distinction is the entire reason decode needs its own kernel, and it is the subject of the next section.

---

## 2. Why decode attention is bandwidth-bound

The word "bandwidth-bound" gets used loosely. Let us make it precise, because the precise version tells you exactly what a good kernel is allowed to achieve, and therefore whether yours is any good.

The tool is the **roofline** ([the roofline model](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) has the full treatment). A kernel's **arithmetic intensity** is the ratio of the floating-point work it does to the bytes it moves from memory:

$$
\text{AI} = \frac{\text{FLOPs}}{\text{bytes read and written}}
$$

Every GPU has a **ridge point**: the arithmetic intensity at which its peak compute rate and its peak memory bandwidth are balanced. Below the ridge you are memory-bound — you finish the math before the data arrives, and your speed is set by bandwidth. Above it you are compute-bound. For an H100 SXM, NVIDIA's datasheet lists roughly 989 teraFLOP/s of bf16 tensor-core throughput and 3.35 TB/s of HBM3 bandwidth, so the ridge sits near $989 / 3.35 \approx 295$ FLOPs per byte. Anything with an intensity far below 295 is firmly memory-bound on that card.

Now compute the intensity of decode attention for one KV head. Reading the keys and values costs $2 \cdot S \cdot d \cdot b$ bytes, where $b$ is bytes per element (2 for bf16). The work: the score GEMV is $S$ dot products of length $d$, about $2 S d$ FLOPs; the output weighted sum is another $2 S d$ FLOPs; softmax adds $O(S)$ cheap exponentials. With grouped-query attention, a group of $G$ query heads shares one KV head, so you read the KV once and do the work $G$ times. The intensity is:

$$
\text{AI} = \frac{G \cdot 4 S d}{2 S d \cdot b} = \frac{2G}{b}
$$

The context length $S$ and the head dimension $d$ cancel — they scale work and bytes together. For Llama-3.1-8B in bf16, the group size is $G = 32 / 8 = 4$ and $b = 2$, so the intensity is about 4 FLOPs per byte. That is roughly seventy times below the H100 ridge point of 295. Decode attention is not "somewhat" memory-bound; it lives at the bottom of the roofline, and no amount of kernel cleverness changes the intensity. The only lever is to move the bytes at the full rate the card allows.

![Before-and-after comparison contrasting compute-bound prefill against memory-bound decode attention](/imgs/blogs/paged-attention-kernel-by-hand-2.webp)

So the target is concrete: a decode attention kernel should read the KV cache at close to peak HBM bandwidth, and the time it takes is, to first order, the bytes divided by the bandwidth. That gives a floor no kernel can beat, and it is the number you measure yours against.

#### Worked example: the read floor for one decode step

From [the memory-math post](/blog/machine-learning/inference-engineering/the-memory-math-of-the-kv-cache), Llama-3.1-8B keeps 128 KiB of KV per token (2 for key and value, 32 layers, 8 KV heads, head dim 128, 2 bytes: $2 \times 32 \times 8 \times 128 \times 2 = 131{,}072$ bytes). A single decode step reads the whole cache once for the attention ops across all layers. At 8k context that is:

$$
8192 \text{ tokens} \times 128 \text{ KiB} = 1.0 \text{ GiB read per step}
$$

Divide by bandwidth to get the floor. On an H100 at 3.35 TB/s:

$$
\frac{1.0 \times 1024^3 \text{ bytes}}{3.35 \times 10^{12} \text{ bytes/s}} \approx 320 \text{ }\mu\text{s}
$$

That 320 microseconds is the fastest the attention reads of one decode step can possibly complete at 8k context on that card, GQA already accounted for in the 128 KiB figure. A perfect kernel hits it; a bad one takes multiples of it. Here is the same floor across the hardware and context matrix, all derived the same way:

| Context | KV read/step | H100 3.35 TB/s | A100 2.0 TB/s | RTX 4090 1.01 TB/s | Source |
| --- | --- | --- | --- | --- | --- |
| 8k | 1.0 GiB | 320 µs | 537 µs | 1.06 ms | derived |
| 32k | 4.0 GiB | 1.28 ms | 2.15 ms | 4.25 ms | derived |
| 128k | 16.0 GiB | 5.13 ms | 8.59 ms | 17.0 ms | derived |

Bandwidth specs cited from the NVIDIA H100, A100, and RTX 4090 datasheets; the times are arithmetic on those specs.

There is a second reading here that matters for the whole engine. A decode step also reads the model weights once — about 15 GiB for Llama-3.1-8B in bf16. At 8k context the 1.0 GiB of KV is small next to that; attention is a minor slice of the step. But at 128k the KV read is 16 GiB, larger than the weights, and attention becomes the single most expensive thing the decode step does. The efficiency of this one kernel matters more the longer your contexts get, which is precisely the regime everyone is racing toward.

#### How to measure it honestly

You cannot trust a decode-attention timing without ceremony, because the operation is so short that everything else dominates if you are careless. The recipe:

```python
import torch

def bench_decode_attn(fn, *args, warmup=25, iters=200):
    # Warm up: trigger Triton autotune / compilation and let clocks settle.
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()

    ms_per_iter = start.elapsed_time(end) / iters
    return ms_per_iter
```

Warm up first — the first call pays autotuning and lazy allocation. Use CUDA events, not `time.time()`, because the launch is asynchronous and the wall clock measures the launch, not the work. Call `torch.cuda.synchronize()` before you read the timer. Run enough iterations that the fixed launch overhead (a few microseconds per launch) is amortized against a step that may itself be only hundreds of microseconds. Then convert your measured time to achieved bandwidth — bytes moved divided by time — and compare to the datasheet peak. If you are reading 1.0 GiB in 400 µs you are at 2.7 TB/s, about 80 percent of an H100's peak, which is a good decode kernel. If you are at 900 µs you are leaving more than half the card on the floor. **On an H100 at 8k context you should expect a well-tuned decode-attention kernel to land somewhere in the 350–500 µs range** — run `bench.py` in `nanoserve` and report what you see; the exact number depends on your Triton version and your block sizes.

---

## 3. Online softmax from first principles

The naive way to compute the softmax in section 1 is to compute all $S$ scores, find their max, exponentiate, sum, divide, and take the weighted sum of values. That requires holding the whole score row of length $S$ in memory, and it requires reading the keys once to compute scores and the values once afterward — and if you want the numerically stable version you also need the max before you can exponentiate anything, which means a first pass over the scores just to find it. For a decode step at 128k context that score row is 128k floats per head. Materializing it defeats the purpose of a fused kernel, and on a real GPU it will not fit in the fast on-chip memory where you want to keep it.

**Online softmax** removes the score row entirely. It is the mathematical heart of FlashAttention, and it is the single idea that makes a fused, memory-bound attention kernel possible. The trick: process the keys and values in **tiles**, and carry only three running quantities — a running maximum, a running denominator, and a running weighted-sum of values — updating them as each tile arrives, rescaling the accumulator whenever a new tile contains a larger score. No full row is ever held.

<figure class="blog-anim">
<svg viewBox="0 0 760 300" role="img" aria-label="Online softmax walks four key-value tiles left to right, carrying a running max and running sum, and rescales the accumulator when tile one raises the max, then divides once at the end to produce the exact output" style="width:100%;height:auto;max-width:820px">
<style>
.osm-tile{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.osm-tl{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.osm-h{font:600 16px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.osm-led{font:400 13.5px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.osm-acc{fill:var(--accent,#6366f1);font:600 13.5px ui-sans-serif,system-ui}
.osm-sweep{fill:var(--accent,#6366f1);opacity:.18}
@keyframes osm-sweep{0%,22%{transform:translateX(0)}25%,47%{transform:translateX(170px)}50%,72%{transform:translateX(340px)}75%,97%{transform:translateX(510px)}100%{transform:translateX(0)}}
@keyframes osm-r0{0%,3%{opacity:0}9%,100%{opacity:1}}
@keyframes osm-r1{0%,28%{opacity:0}34%,100%{opacity:1}}
@keyframes osm-r2{0%,53%{opacity:0}59%,100%{opacity:1}}
@keyframes osm-r3{0%,78%{opacity:0}84%,100%{opacity:1}}
.osm-mv{animation:osm-sweep 12s ease-in-out infinite}
.osm-l0{animation:osm-r0 12s ease-in-out infinite}
.osm-l1{animation:osm-r1 12s ease-in-out infinite}
.osm-l2{animation:osm-r2 12s ease-in-out infinite}
.osm-l3{animation:osm-r3 12s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.osm-mv{animation:none}.osm-l0,.osm-l1,.osm-l2,.osm-l3{animation:none;opacity:1}}
</style>
<text class="osm-h" x="40" y="30">Walk the KV cache one tile at a time; carry only max and sum</text>
<rect class="osm-tile" x="60"  y="52" width="150" height="60" rx="8"/>
<rect class="osm-tile" x="230" y="52" width="150" height="60" rx="8"/>
<rect class="osm-tile" x="400" y="52" width="150" height="60" rx="8"/>
<rect class="osm-tile" x="570" y="52" width="150" height="60" rx="8"/>
<rect class="osm-sweep osm-mv" x="60" y="52" width="150" height="60" rx="8"/>
<text class="osm-tl" x="135" y="88">K,V tile 0</text>
<text class="osm-tl" x="305" y="88">K,V tile 1</text>
<text class="osm-tl" x="475" y="88">K,V tile 2</text>
<text class="osm-tl" x="645" y="88">K,V tile 3</text>
<text class="osm-led osm-l0" x="40" y="160">after tile 0:  max = 2.1   sum = 3.0   <tspan class="osm-acc">acc built from 16 tokens</tspan></text>
<text class="osm-led osm-l1" x="40" y="192">after tile 1:  bigger score, max 2.1 to 3.4   <tspan class="osm-acc">rescale acc by e^(2.1-3.4) = 0.27</tspan></text>
<text class="osm-led osm-l2" x="40" y="224">after tile 2:  max = 3.4   sum = 6.2   acc keeps accumulating in place</text>
<text class="osm-led osm-l3" x="40" y="256">after tile 3:  <tspan class="osm-acc">divide acc by sum 8.4  gives the exact softmax output</tspan></text>
</svg>
<figcaption>Online softmax processes the cache tile by tile with constant state; when tile 1 raises the running max it rescales the accumulator by e^(old max minus new max), and one final division yields the exact result.</figcaption>
</figure>

Let me derive it so you can see it is *exact*, not an approximation. Define the running state after processing the first $i$ tokens, using the running max $m_i = \max_{j \le i} s_j$:

$$
\ell_i = \sum_{j=1}^{i} e^{s_j - m_i}, \qquad
\mathbf{o}_i = \sum_{j=1}^{i} e^{s_j - m_i}\, \mathbf{v}_j
$$

Note that both are defined relative to the *current* running max $m_i$. When token $i+1$ (or a whole tile) arrives, we get a new max $m_{i+1} = \max(m_i, s_{i+1})$. Every previously-accumulated term was scaled by $e^{-m_i}$ and must now be scaled by $e^{-m_{i+1}}$ instead. The correction is a single multiply, because:

$$
e^{s_j - m_i} \cdot e^{m_i - m_{i+1}} = e^{s_j - m_{i+1}}
$$

So define the rescale factor $\alpha = e^{m_i - m_{i+1}}$, which is at most 1 (the new max is at least the old one, so the exponent is at most 0, so $\alpha \in (0, 1]$ and there is no overflow). The updates are:

$$
m_{i+1} = \max(m_i, s_{i+1})
$$
$$
\ell_{i+1} = \alpha\, \ell_i + e^{s_{i+1} - m_{i+1}}
$$
$$
\mathbf{o}_{i+1} = \alpha\, \mathbf{o}_i + e^{s_{i+1} - m_{i+1}}\, \mathbf{v}_{i+1}
$$

At the very end, after all $S$ tokens, divide once:

$$
\mathbf{o} = \frac{\mathbf{o}_S}{\ell_S}
= \frac{\sum_j e^{s_j - m_S}\, \mathbf{v}_j}{\sum_j e^{s_j - m_S}}
= \frac{\sum_j e^{s_j}\, \mathbf{v}_j}{\sum_j e^{s_j}}
= \sum_j p_j\, \mathbf{v}_j
$$

The offset $m_S$ cancels between numerator and denominator, so the result is the exact softmax-weighted sum, identical to the naive computation, but computed in one streaming pass with $O(1)$ state per head instead of $O(S)$. This is the running-max rescaling identity, and it is the thing a reviewer should check line-by-line in any attention kernel, because getting the rescale wrong — forgetting to multiply the accumulator by $\alpha$, or using the old max in the exponent — produces an answer that is smooth and plausible and wrong. The online-normalizer formulation was published by Milakov and Gimelshein (2018), "Online normalizer calculation for softmax"; FlashAttention (Dao et al., 2022) fused it into the attention kernel to beat the memory wall, which the [kernel fusion and FlashAttention post](/blog/machine-learning/high-performance-computing/kernel-fusion-and-flashattention-beating-the-memory-wall) covers in depth.

Here is the whole algorithm in plain Python — no GPU, no Triton, just the loop, so you can read it and convince yourself it matches the derivation. This is the reference we will translate to Triton and, later, test against.

```python
import numpy as np

def online_softmax_attention(q, K, V, tile=16):
    # q: [d]   K, V: [S, d].  Returns o: [d], the exact attention output.
    d = q.shape[0]
    scale = 1.0 / np.sqrt(d)
    m = -np.inf                 # running max of the scores
    l = 0.0                     # running denominator
    acc = np.zeros(d)           # running weighted sum of values (unnormalized)

    for start in range(0, K.shape[0], tile):
        Kt = K[start:start + tile]          # [t, d]
        Vt = V[start:start + tile]          # [t, d]
        s = (Kt @ q) * scale                # [t] scores for this tile

        m_new = max(m, s.max())
        alpha = np.exp(m - m_new)           # rescale factor for prior state
        p = np.exp(s - m_new)               # [t] tile weights at the new max

        acc = acc * alpha + p @ Vt          # rescale, then add this tile
        l = l * alpha + p.sum()
        m = m_new

    return acc / l                          # one division at the very end
```

You can check it against a dense reference right now, on the CPU, and it should agree to floating-point tolerance:

```python
def dense_reference(q, K, V):
    scale = 1.0 / np.sqrt(q.shape[0])
    s = (K @ q) * scale
    p = np.exp(s - s.max())
    p /= p.sum()
    return p @ V

rng = np.random.default_rng(0)
q = rng.standard_normal(128)
K = rng.standard_normal((500, 128))
V = rng.standard_normal((500, 128))
o1 = online_softmax_attention(q, K, V)
o2 = dense_reference(q, K, V)
print(np.max(np.abs(o1 - o2)))   # ~1e-15, exact up to float rounding
```

The maximum absolute difference is on the order of $10^{-15}$ — the two are the same computation, and the online version simply never held the length-500 score row. That is the property we carry into the kernel.

---

## 4. Writing the paged decode kernel in Triton

Now the real one. We write it in Triton, and that choice deserves a sentence: this kernel is memory-bound, its shapes are dynamic (every sequence has a different length), and it wants autotunable tile sizes — exactly the case where Triton's Python-level control and autotuner earn their keep over hand-written CUDA, as [the Triton post](/blog/machine-learning/inference-engineering/triton-for-inference-kernels-and-when-to-stop-writing-cuda) argues. We are not chasing the last five percent that a CUDA expert would extract; we are writing something readable and fast that lives on top of the paged layout.

The layout, fixed by the paging post: `k_cache` and `v_cache` are `[num_blocks, block_size, num_kv_heads, head_dim]`, one flat pool of physical blocks. The `block_table` is `[num_seqs, max_blocks_per_seq]`, mapping each sequence's logical block index to a physical block id. The `seq_lens` tensor holds each sequence's current length so the kernel knows where the cache ends and how much of the last block is real.

The grid is the key design decision, and we take it from vLLM's Triton attention backend: **one program per (sequence, KV head)**. The vLLM team's [Triton attention backend deep dive](https://vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive) (2026-03-04) describes the kernel as looping over query tokens and query heads with the paged KV cache in the innermost loop, with the launch grid set to batch times KV heads. Assigning by KV head rather than query head is what lets a program read each KV tile once and reuse it across the whole group of query heads that share it — the GQA amortization we will return to in section 8. For a first, readable version we serve a single query head per program and add the group loop afterward.

![Grid layout showing one program instance per sequence and KV head, each reused across its query-head group](/imgs/blogs/paged-attention-kernel-by-hand-4.webp)

The grid, drawn above, is a rectangle: KV heads down one axis, sequences across the other, one independent program per cell. Here is the kernel.

```python
import triton
import triton.language as tl

@triton.jit
def paged_decode_kernel(
    q_ptr,               # [num_seqs, num_q_heads, head_dim]
    k_cache_ptr,         # [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache_ptr,         # [num_blocks, block_size, num_kv_heads, head_dim]
    out_ptr,             # [num_seqs, num_q_heads, head_dim]
    block_table_ptr,     # [num_seqs, max_blocks_per_seq]
    seq_lens_ptr,        # [num_seqs]
    scale,
    stride_q_seq, stride_q_head,
    stride_kc_blk, stride_kc_tok, stride_kc_head,
    stride_bt_seq,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GROUP: tl.constexpr,          # num_q_heads // num_kv_heads
):
    seq_idx = tl.program_id(0)
    kv_head = tl.program_id(1)
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    n_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE

    d = tl.arange(0, HEAD_DIM)                       # [HEAD_DIM]
    tok = tl.arange(0, BLOCK_SIZE)                   # [BLOCK_SIZE]

    # First query head of this KV group (single-head version).
    q_head = kv_head * GROUP
    q = tl.load(q_ptr + seq_idx * stride_q_seq + q_head * stride_q_head + d)

    m_i = -float("inf")                              # running max
    l_i = 0.0                                        # running denominator
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)     # running weighted sum

    bt = block_table_ptr + seq_idx * stride_bt_seq
    for blk in range(0, n_blocks):
        phys = tl.load(bt + blk)                     # block-table indirection
        pos = blk * BLOCK_SIZE + tok
        mask = pos < seq_len                         # last block is partial

        k_off = (phys * stride_kc_blk + tok[:, None] * stride_kc_tok
                 + kv_head * stride_kc_head + d[None, :])
        k = tl.load(k_cache_ptr + k_off, mask=mask[:, None], other=0.0)  # [BLK, D]

        s = tl.sum(q[None, :] * k, axis=1) * scale   # [BLK] scores (GEMV)
        s = tl.where(mask, s, -float("inf"))         # mask padding to -inf

        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        alpha = tl.exp(m_i - m_new)                  # rescale prior state
        p = tl.exp(s - m_new)                        # [BLK] tile weights

        v = tl.load(v_cache_ptr + k_off, mask=mask[:, None], other=0.0)  # [BLK, D]
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    out = acc / l_i
    tl.store(out_ptr + seq_idx * stride_q_seq + q_head * stride_q_head + d, out)
```

Read it against the derivation and it is the same six lines of update logic, wrapped in the paged-cache addressing. The three things that make it a *paged* kernel rather than a textbook one:

**The block-table indirection.** `phys = tl.load(bt + blk)` is the whole point of paging. The logical block index `blk` counts up 0, 1, 2, …, but the physical block id it maps to is scattered anywhere in the pool. Every address into the cache is computed from `phys`, not from `blk`. This is a gather, and it is why the read is not a single clean strided slice; the [gather kernel post](/blog/machine-learning/inference-engineering/the-kv-cache-append-and-gather-kernel) built the write side of this same indirection.

**The last-block mask.** `mask = pos < seq_len` handles the partial final block. A sequence of 70 tokens with block size 16 fills four blocks and uses 6 of the 16 slots in the fifth. Loading the missing slots with `other=0.0` and then forcing their scores to $-\infty$ with `tl.where` makes their softmax weight exactly zero, so they contribute nothing. Forget this mask and you attend to whatever stale garbage was in those cache slots — a classic and maddening correctness bug that only shows up when a sequence length is not a multiple of the block size.

**Online softmax in the loop body.** `acc = acc * alpha + ...` and `l_i = l_i * alpha + ...` are the running-max rescale from section 3. This is the line to stare at in code review. The accumulator must be rescaled by the same `alpha` as the denominator, using the *new* max in both exponents.

The launcher wires up the strides and the grid:

```python
import torch

def paged_decode(q, k_cache, v_cache, block_table, seq_lens, scale):
    num_seqs, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[2]
    out = torch.empty_like(q)
    grid = (num_seqs, num_kv_heads)          # batch x KV heads
    paged_decode_kernel[grid](
        q, k_cache, v_cache, out, block_table, seq_lens, scale,
        q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        block_table.stride(0),
        HEAD_DIM=head_dim,
        BLOCK_SIZE=k_cache.shape[1],
        GROUP=num_q_heads // num_kv_heads,
    )
    return out
```

To serve the whole GQA group and read each KV tile only once, wrap the per-head state in the group dimension and load `k` and `v` a single time per block. The change is small and it is exactly where the bandwidth win lives:

```python
    # GQA-amortized inner body (replaces the single-head version):
    #   keep GROUP running states, load K/V once, reuse across the group.
    m_i = tl.full([GROUP], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([GROUP], dtype=tl.float32)
    acc = tl.zeros([GROUP, HEAD_DIM], dtype=tl.float32)
    qg = tl.load(q_ptr + seq_idx * stride_q_seq
                 + (q_head + tl.arange(0, GROUP))[:, None] * stride_q_head
                 + d[None, :])                       # [GROUP, HEAD_DIM]
    # ... inside the block loop, after loading k, v ONCE:
    s = tl.dot(qg, tl.trans(k)) * scale              # [GROUP, BLK] one GEMM
    # rescale acc/l_i/m_i per row of the group, same identity, vectorized.
```

That `tl.dot(qg, k^T)` is the vLLM "Q blocks" idea: packing multiple query heads into the rows of a small matrix turns the per-head GEMV into one GEMM, which uses the tensor cores better and, more importantly here, reads each K and V tile exactly once for the whole group. On Llama-3.1-8B that is a 4x reduction in KV traffic versus reading per query head, and KV traffic is the entire cost.

---

## 5. Split-K: filling the card on a lonely sequence

The kernel in section 4 has a problem that only appears in a specific but common situation: one sequence, or a few, with a very long context. Suppose batch size 1 at 128k context. The grid is `(1 sequence, 8 KV heads)` — eight programs. An H100 has 132 streaming multiprocessors. Eight programs occupy eight of them and leave 124 idle, and those eight programs each have to stream 2 GiB of KV serially. You are reading 16 GiB of memory using six percent of the card's ability to read memory. The card is not the bottleneck; your grid is.

The fix is **split-K** (vLLM calls it "parallel tiled softmax," and the classic reference is Flash-Decoding, from the FlashAttention authors). Split each sequence's KV range into chunks, give each chunk its own program, and let those programs run in parallel across the idle SMs. Each program computes a *partial* result over its chunk — a local max, a local denominator, and a local unnormalized output — and then a second, tiny reduction kernel combines the partials into the exact answer.

![Dataflow graph splitting one sequence's KV range across parallel partial kernels that merge through a reduction kernel](/imgs/blogs/paged-attention-kernel-by-hand-5.webp)

The reason this is safe is the same rescaling identity from section 3, now applied across chunks instead of tiles. The online-softmax merge is **associative and commutative**: combining the partials of two disjoint score ranges gives the same result as processing them in one pass. Given two partials $(m^{(1)}, \ell^{(1)}, \mathbf{o}^{(1)})$ and $(m^{(2)}, \ell^{(2)}, \mathbf{o}^{(2)})$, the merge is:

$$
m = \max(m^{(1)}, m^{(2)})
$$
$$
\ell = e^{m^{(1)} - m}\,\ell^{(1)} + e^{m^{(2)} - m}\,\ell^{(2)}, \qquad
\mathbf{o} = e^{m^{(1)} - m}\,\mathbf{o}^{(1)} + e^{m^{(2)} - m}\,\mathbf{o}^{(2)}
$$

Because it is associative you can split into any number of chunks and reduce them in any order. In practice the partial kernel stores its normalized output and a single scalar per split — the **log-sum-exp**, $\text{lse} = m + \log \ell$ — which is all the reduction needs to weight the partials correctly. The partial kernel is the section-4 loop restricted to a range of logical blocks:

```python
@triton.jit
def paged_decode_splitk_kernel(
    q_ptr, k_cache_ptr, v_cache_ptr,
    partial_out_ptr,     # [num_seqs, num_q_heads, num_splits, head_dim]
    partial_lse_ptr,     # [num_seqs, num_q_heads, num_splits]
    block_table_ptr, seq_lens_ptr, scale,
    stride_q_seq, stride_q_head,
    stride_kc_blk, stride_kc_tok, stride_kc_head,
    stride_bt_seq, stride_po_seq, stride_po_head, stride_po_split,
    stride_pl_seq, stride_pl_head,
    HEAD_DIM: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    GROUP: tl.constexpr, BLOCKS_PER_SPLIT: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    kv_head = tl.program_id(1)
    split = tl.program_id(2)                         # third grid axis
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    n_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE

    blk_start = split * BLOCKS_PER_SPLIT
    blk_end = tl.minimum(blk_start + BLOCKS_PER_SPLIT, n_blocks)
    if blk_start >= n_blocks:
        return                                       # this split has no work

    d = tl.arange(0, HEAD_DIM)
    tok = tl.arange(0, BLOCK_SIZE)
    q_head = kv_head * GROUP
    q = tl.load(q_ptr + seq_idx * stride_q_seq + q_head * stride_q_head + d)

    m_i = -float("inf"); l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    bt = block_table_ptr + seq_idx * stride_bt_seq
    for blk in range(blk_start, blk_end):            # only this split's range
        phys = tl.load(bt + blk)
        pos = blk * BLOCK_SIZE + tok
        mask = pos < seq_len
        off = (phys * stride_kc_blk + tok[:, None] * stride_kc_tok
               + kv_head * stride_kc_head + d[None, :])
        k = tl.load(k_cache_ptr + off, mask=mask[:, None], other=0.0)
        s = tl.where(mask, tl.sum(q[None, :] * k, axis=1) * scale, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new)
        v = tl.load(v_cache_ptr + off, mask=mask[:, None], other=0.0)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    po = partial_out_ptr + (seq_idx * stride_po_seq + q_head * stride_po_head
                            + split * stride_po_split + d)
    tl.store(po, acc / l_i)                           # normalized partial output
    pl = partial_lse_ptr + seq_idx * stride_pl_seq + q_head * stride_pl_head + split
    tl.store(pl, m_i + tl.log(l_i))                   # log-sum-exp for the merge
```

The reduction kernel loads all splits' partials and log-sum-exps for one (sequence, query head), finds the max lse, and takes the lse-weighted average of the normalized partials:

```python
@triton.jit
def splitk_reduce_kernel(
    partial_out_ptr, partial_lse_ptr, out_ptr,
    stride_po_seq, stride_po_head, stride_po_split,
    stride_pl_seq, stride_pl_head,
    stride_o_seq, stride_o_head,
    HEAD_DIM: tl.constexpr, NUM_SPLITS: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    q_head = tl.program_id(1)
    s = tl.arange(0, NUM_SPLITS)
    lse = tl.load(partial_lse_ptr + seq_idx * stride_pl_seq
                  + q_head * stride_pl_head + s)      # [NUM_SPLITS]
    m = tl.max(lse, axis=0)
    w = tl.exp(lse - m)                               # weights proportional to l*e^m
    denom = tl.sum(w, axis=0)

    d = tl.arange(0, HEAD_DIM)
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    for i in range(0, NUM_SPLITS):
        po = tl.load(partial_out_ptr + seq_idx * stride_po_seq
                     + q_head * stride_po_head + i * stride_po_split + d)
        wi = tl.load(partial_lse_ptr + seq_idx * stride_pl_seq
                     + q_head * stride_pl_head + i)
        acc += tl.exp(wi - m) * po
    tl.store(out_ptr + seq_idx * stride_o_seq + q_head * stride_o_head + d,
             acc / denom)
```

The weight on split $i$ is $e^{\text{lse}_i - m} \propto \ell_i e^{m_i}$, which is exactly the total softmax mass that split accumulated, so the weighted average reconstructs the global softmax. It is the merge rule above, written as a normalized average.

**When to split, and the cost.** Splitting is not free: it launches a second kernel, and it writes and re-reads the partials through HBM. The vLLM Triton post is explicit that the split path is heuristic-gated and that "no single configuration dominates." The rule is simple to state: split only when there is idle hardware to fill. If your grid already has more programs than the card has SMs — many sequences, or long batches — splitting adds overhead for nothing. If your grid is starved — few sequences, long context — splitting is the difference between six percent and full utilization.

```python
def choose_num_splits(num_seqs, num_kv_heads, max_seq_len, block_size,
                      num_sms, min_blocks_per_split=8, max_splits=16):
    base_programs = num_seqs * num_kv_heads
    if base_programs >= num_sms:
        return 1                                    # grid already fills the card
    n_blocks = (max_seq_len + block_size - 1) // block_size
    # enough splits to fill the card, but not so fine each split is trivial
    want = -(-num_sms // base_programs)              # ceil(num_sms / base_programs)
    cap = max(1, n_blocks // min_blocks_per_split)
    return max(1, min(max_splits, want, cap))
```

#### Worked example: batch 1 at 128k on an H100

Base grid: 1 sequence times 8 KV heads is 8 programs, against 132 SMs. The card is 94 percent idle. At 128k tokens with block size 16 there are 8,192 logical blocks. `choose_num_splits` wants `ceil(132 / 8) = 17` splits, capped by `max_splits = 16`, and each split then covers 512 blocks — plenty of work. The grid becomes `(1, 8, 16)` = 128 programs, now filling essentially the whole card, each streaming 1 GiB instead of 16 GiB serially. The read floor from section 2 was 5.13 ms on an H100; without splitting you cannot get near it because you are using 8 SMs; with 16 splits you can. **The expected win is dramatic at batch 1 and long context and shrinks to zero — then negative — as batch size rises**, because a full batch already saturates the card and the extra launch is pure overhead. This is derived from the grid arithmetic and the SM count; the exact speedup is workload-dependent and you should measure it with the benchmark harness from section 2.

---

## 6. The four decode strategies, compared

We now have three kernels — naive-per-head, flash-decode (online softmax, one pass), and split-K — and there is a fourth worth knowing because it is where the production frontier is: the **persistent kernel**. Here is the whole space.

![Matrix comparing naive, flash-decode, split-K, and persistent decode-attention strategies across memory, utilization, and launches](/imgs/blogs/paged-attention-kernel-by-hand-6.webp)

| Strategy | Score row in VRAM | Fills card at batch 1 | Launches | Where you see it | Source |
| --- | --- | --- | --- | --- | --- |
| Naive, one program per head | full row, $O(S)$ | no, a few SMs | 1 | textbook / explainer | derived |
| Flash-decode (online softmax) | none, $O(1)$ | partly, one SM per head | 1 | FlashAttention, FlashInfer | cited: Dao et al. |
| Split-KV (parallel tiled softmax) | none, $O(1)$ per split | yes, splits fill SMs | 2 | vLLM Triton backend | cited: vLLM |
| Persistent kernel | none, $O(1)$ | yes, fixed launch waves | 1 | vLLM Triton, HPC-Ops | cited: vLLM, Tencent |

The **persistent kernel** attacks a subtler problem than utilization: launch overhead and CUDA-graph replay. A split-K grid varies its shape with the workload — the number of splits depends on batch size and context length, so the launch dimensions change from step to step. That is a problem in the decode loop because production engines capture the decode step into a **CUDA graph** — a recorded, replayable sequence of kernel launches that removes per-launch CPU overhead, which matters enormously when the step itself is only hundreds of microseconds. A graph bakes in fixed launch dimensions; a kernel whose grid changes every step cannot be replayed cleanly and forces re-capture or falls back to eager launches. The vLLM Triton post names this directly: variable launch grids "replay badly under CUDA graphs."

The persistent-kernel design fixes it by launching a **fixed** number of program instances — one per unit of compute the card has, decided once — and having each instance loop, pulling its next unit of work from a small metadata table in GPU memory. The grid never changes shape, so it captures into a CUDA graph and replays every step with no CPU involvement, while the work-stealing over the metadata table keeps every instance busy regardless of how the sequences are distributed. The trade is complexity: you are now scheduling work inside the kernel. A later post in this series covers CUDA graphs in the decode loop in full; the connection to hold onto here is that **your attention kernel's launch shape is part of your CUDA-graph story**, and a kernel that is fast in isolation but forces graph re-capture every step can be a net loss.

Tencent's Hunyuan team pushed the persistent idea further for mixed-length batches in their [HPC-Ops backend](https://vllm.ai/blog/2026-07-06-vllm-hpc-ops) (2026-07-06). Their attention kernel runs in three stages: an assign kernel slices the KV into uniform 64-token tiles bucketed across the CTAs, a persistent compute-and-combine stage writes partials and reduces them, and a fused prologue folds QK-norm, RoPE, and the KV-cache write together. The uniform-tile bucketing is the direct answer to the failure mode we hit in section 5's worked example — one long sequence monopolizing a CTA while short ones finish and leave their CTAs idle. We will price that failure mode next.

---

## 7. Correctness: an SDPA reference and the rescale bug

A decode attention kernel is dangerous precisely because a wrong one looks right. The output is a smooth vector of plausible magnitude; the model keeps generating fluent text; nothing crashes. The only way to know it is correct is to diff it against a reference you trust, element by element, within a tolerance you chose on purpose. The reference is PyTorch's own attention, driven through the same gather so it sees the same paged data. This is the reference implementation the series has leaned on since [the forward-pass post](/blog/machine-learning/inference-engineering/what-inference-engineering-is) — you check your fast path against the slow, obviously-correct one.

```python
import torch
import torch.nn.functional as F

def paged_decode_reference(q, k_cache, v_cache, block_table, seq_lens):
    # q: [num_seqs, num_q_heads, head_dim].  Gathers the paged cache into a
    # dense [S, num_kv_heads, head_dim] per sequence, then runs SDPA. Slow,
    # obviously correct, GQA-aware. This is ground truth.
    num_seqs, num_q_heads, head_dim = q.shape
    block_size = k_cache.shape[1]
    num_kv_heads = k_cache.shape[2]
    group = num_q_heads // num_kv_heads
    out = torch.empty_like(q)
    scale = 1.0 / (head_dim ** 0.5)

    for i in range(num_seqs):
        S = int(seq_lens[i])
        n_blocks = (S + block_size - 1) // block_size
        phys = block_table[i, :n_blocks]                       # logical -> physical
        k = k_cache[phys].reshape(-1, num_kv_heads, head_dim)[:S]   # [S, Hkv, d]
        v = v_cache[phys].reshape(-1, num_kv_heads, head_dim)[:S]
        # expand KV heads to query heads for GQA, then SDPA over the S keys
        k = k.repeat_interleave(group, dim=1)                  # [S, Hq, d]
        v = v.repeat_interleave(group, dim=1)
        qi = q[i].unsqueeze(1)                                 # [Hq, 1, d]
        ki = k.permute(1, 0, 2)                                # [Hq, S, d]
        vi = v.permute(1, 0, 2)
        o = F.scaled_dot_product_attention(qi, ki, vi, scale=scale)  # [Hq,1,d]
        out[i] = o.squeeze(1)
    return out
```

The test compares the Triton kernel to this reference and picks a tolerance that reflects bf16 accumulation, not float64 exactness:

```python
def test_paged_decode_matches_reference():
    torch.manual_seed(0)
    num_seqs, num_kv_heads, group, head_dim, block_size = 3, 8, 4, 128, 16
    num_q_heads = num_kv_heads * group
    seq_lens = torch.tensor([70, 512, 4096], device="cuda")     # note 70: partial block
    max_blocks = (int(seq_lens.max()) + block_size - 1) // block_size
    num_blocks = num_seqs * max_blocks + 8
    q = torch.randn(num_seqs, num_q_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim,
                          device="cuda", dtype=torch.bfloat16)
    v_cache = torch.randn_like(k_cache)
    # scatter each sequence's logical blocks to shuffled physical ids
    block_table = torch.zeros(num_seqs, max_blocks, dtype=torch.int32, device="cuda")
    free = torch.randperm(num_blocks, device="cuda")
    cursor = 0
    for i in range(num_seqs):
        n = (int(seq_lens[i]) + block_size - 1) // block_size
        block_table[i, :n] = free[cursor:cursor + n].to(torch.int32)
        cursor += n

    got = paged_decode(q, k_cache, v_cache, block_table, seq_lens, 1.0 / head_dim ** 0.5)
    ref = paged_decode_reference(q, k_cache, v_cache, block_table, seq_lens)
    torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)   # bf16 tolerance
```

Two deliberate choices in that test. The sequence lengths include 70, which is not a multiple of 16, so the partial-block mask gets exercised — the single most common source of a kernel that passes on round lengths and fails in production. And the block table is *shuffled*, so physical ids are scattered, which exercises the indirection rather than a lucky contiguous layout that would hide an addressing bug.

Now the rescale bug, because it is worth seeing fail. Suppose you forget to rescale the accumulator — you update `l_i` correctly but write `acc = acc + tl.sum(p[:, None] * v, axis=0)` without the `* alpha`. The denominator is then correct but the numerator mixes terms scaled to different maxima, so the output is a weighted sum with the wrong weights on every tile before the last max update.

```python
# WRONG: accumulator not rescaled when the running max grows.
acc = acc + tl.sum(p[:, None] * v, axis=0)   # missing  acc * alpha
l_i = l_i * alpha + tl.sum(p, axis=0)        # denominator still rescaled
```

The failure has a signature: it vanishes when scores are small and uniform (then `alpha` is near 1 and the missing rescale barely matters) and grows with the score range and the context length (more max-updates, each mis-weighting more history). So it sails through a tiny unit test with a 32-token sequence and blows up at 8k, which is exactly the kind of bug that ships. The `assert_close` above with a real 4096-token sequence catches it; a 32-token smoke test does not. Test at length, with shuffled blocks, and with a non-multiple sequence length, or you are not testing the kernel you are going to run.

---

## 8. Where it breaks: GQA, skewed batches, and fat heads

A decode attention kernel meets four stress cases in production. Each one is a place where the simple version quietly does the wrong thing or the slow thing.

**Grouped-query attention is the read multiplier.** Modern models share KV heads across query heads to shrink the cache: Llama-3.1-8B has 32 query heads but only 8 KV heads, so each KV head serves a group of 4. The kernel must respect this, and how it does so is the difference between a fast kernel and a 4x-too-slow one.

![Stacked hierarchy showing grouped-query attention reading each KV head once and reusing it across four query heads](/imgs/blogs/paged-attention-kernel-by-hand-7.webp)

If you launch one program per query head, you read each KV head's cache four times — once per query head in the group — and since the cache read is the entire cost, you have quadrupled the cost for nothing. Launching one program per KV head and looping the group inside, reusing the loaded K and V tiles, reads the cache once. That is the whole reason the grid is batch times KV heads and not batch times query heads. For a model with no GQA (multi-head attention, group size 1) the two are identical; the wider the group, the bigger the amortization. Multi-query attention (group size equal to the number of query heads, one KV head total) is the extreme, and there the amortization is the entire head count.

**Skewed batches starve the card.** Section 5's split-K fills the card for a uniformly long batch, but real batches are ragged: one request at 128k context sitting next to 31 requests at 4k. A static split — the same number of splits for every sequence — either under-splits the long one (leaving it serial while its CTA runs long after the short ones finish) or over-splits the short ones (launch overhead on work that did not need it). This is the CTA-monopoly problem, and it is exactly what Tencent's HPC-Ops assign-stage solves by slicing all sequences into **uniform 64-token tiles** and bucketing those tiles across CTAs, so a long sequence's tiles spread across many CTAs instead of one CTA owning the whole thing. The HPC-Ops team reports, on an H20 with a Hunyuan FP8 model, up to 2.95x over static split-KV and an average 2.25x over FlashInfer and FlashAttention, and on the exact skewed shape — one sequence at 128k with 31 at 4k — 0.063 ms with their dynamic scheme versus 0.186 ms static. Those are their measurements on their hardware with their model; treat the 2.95x as a cited ceiling for what dynamic tiling buys on a pathologically skewed batch, not a number you will see on a uniform one.

**Fat heads blow up on-chip memory.** The kernel keeps its running accumulator in registers and its tiles in shared memory. Both scale with `head_dim`. At `head_dim = 128` (Llama, Qwen) this is comfortable. At `head_dim = 256` — which Gemma-family models use — the accumulator doubles, the K and V tiles double, and you can run out of registers or shared memory, forcing the compiler to spill or the occupancy to drop, either of which slows the kernel. The vLLM FP8 KV-cache work notes in passing that `head_dim = 256` makes even prefill slower for related reasons. The practical consequence: the block size that autotunes best on a 128-dim model is not the one that works on a 256-dim model, which is why the kernel exposes `BLOCK_SIZE` as an autotuned constant rather than hard-coding it. This is the "no single configuration dominates" caveat the vLLM Triton post makes, in one concrete instance.

**Prefill and decode want different kernels.** This kernel is a decode kernel: it assumes one query position per sequence. Prefill has many query positions and a genuinely different arithmetic profile — it is compute-bound, it reuses each key across many queries, and it wants a GEMM-shaped kernel (a real FlashAttention forward), not a GEMV-shaped one. Engines run different kernels for the two phases, or a unified kernel that special-cases them, precisely because the roofline position is different. Do not try to serve prefill with the decode kernel; at long prompt lengths it will be dramatically slower than a proper prefill attention, because it throws away the query-side reuse that makes prefill compute-bound in the first place.

#### Worked example: GQA read amortization on Llama-3.1-8B

At 8k context, one decode step reads 1.0 GiB of KV with the correct KV-head grid (the 128 KiB/token from section 2 already assumes 8 KV heads). Launch per query head instead, and you read each KV head's slice 4 times: the effective read becomes 4.0 GiB, and the read floor on an H100 jumps from 320 µs to about 1.28 ms — a 4x regression from a single wrong grid choice, with identical math and identical output. There is no accuracy difference and no crash; the only symptom is that your tokens-per-second is a quarter of what the hardware allows. Source: derived from the per-token KV bytes and the H100 bandwidth spec.

---

## 9. Case studies: real numbers with provenance

Three public results anchor the claim that a readable kernel can be competitive, and one anchors the ceiling.

**vLLM's Triton attention backend reached parity with FlashAttention 3.** In the [Triton attention backend deep dive](https://vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive) (2026-03-04), the vLLM team reports their Triton kernel hitting **100.7 percent of FlashAttention 3** on an H100 — for Llama-3.1-8B, batch 1, a 500-token input, and a long decode — in roughly **800 lines of Triton versus FlashAttention 3's roughly 70,000 lines**. On AMD MI300 they report about 5.8x over earlier implementations, and the Triton backend is the default on AMD/ROCm. The setup matters: batch 1 and long decode is the memory-bound, split-K-friendly regime this whole post is about, which is exactly where a bandwidth-saturating Triton kernel has nothing to prove against hand-tuned CUDA — both are reading memory as fast as the card allows. The lesson is not that Triton beats CUDA in general; it is that for a memory-bound kernel, readable and fast are not in tension.

**HPC-Ops shows what dynamic tiling buys on skewed batches.** As covered in section 8, Tencent's [HPC-Ops backend](https://vllm.ai/blog/2026-07-06-vllm-hpc-ops) (2026-07-06) reports up to 2.95x over static split-KV on an H20 with a Hunyuan FP8 model, from bucketing uniform 64-token tiles across CTAs. It is Hopper-only and tuned for their model family, so it is a ceiling for the technique, not a portable number.

**Flash-Decoding is the origin of decode split-K.** The split-K-over-sequence idea for decode was introduced by the FlashAttention authors as Flash-Decoding, and it is the direct ancestor of vLLM's parallel tiled softmax. FlashAttention itself (Dao et al., 2022) is the fused online-softmax kernel that this entire post rests on; the [kernel fusion and FlashAttention post](/blog/machine-learning/high-performance-computing/kernel-fusion-and-flashattention-beating-the-memory-wall) covers the memory-wall argument in depth.

| Result | Setup | Number | Source |
| --- | --- | --- | --- |
| Triton vs FlashAttention 3 | H100, Llama-3.1-8B, batch 1, 500-tok input, long decode | 100.7% of FA3 | cited: vLLM (2026-03-04) |
| Triton kernel size | same backend | ~800 lines vs ~70,000 | cited: vLLM (2026-03-04) |
| Dynamic tiling vs static split-KV | H20, Hunyuan FP8 | up to 2.95x | cited: Tencent HPC-Ops (2026-07-06) |
| Skewed batch (1x128k + 31x4k) | H20, Hunyuan FP8 | 0.063 ms vs 0.186 ms | cited: Tencent HPC-Ops (2026-07-06) |

Every number here is someone else's measurement on named hardware with a named model. None is mine. Reproduce them on your card and your model and you will get different absolutes and, if the technique is real, the same direction.

---

## When to reach for this (and when not to)

Write this kernel to understand it. There is no substitute for deriving online softmax and watching your own kernel match SDPA to tolerance for grasping why the KV cache is the thing that governs LLM inference speed. If you are building an engine to learn how engines work — which is the entire premise of `nanoserve` — this is the most instructive kernel you will write.

Do not write it to ship. In production, use FlashAttention, FlashInfer, or vLLM's paged-attention kernel. They handle FP8 KV caches, every head dimension, sliding-window and hybrid layers, the prefill and decode paths, and a decade of edge cases you have not thought of, and they are maintained by people whose full-time job is this kernel. Your hand-written Triton kernel will match them in the easy batch-1-long-decode regime — that is genuinely the point vLLM's own results make — and lose in every regime that has an edge case, because the edge cases are the work.

The line is sharp. Reach for your own kernel when you are learning, prototyping a genuinely novel attention variant that no library implements yet, or targeting hardware the libraries do not support well. Reach for the library the moment correctness across the full matrix of shapes and dtypes matters more than your understanding of the internals — which, for anything user-facing, is immediately. And if you do write your own, the split-K heuristic and the CUDA-graph launch-shape interaction from section 6 are not optional polish; they are the difference between a kernel that is fast on a microbenchmark and one that is fast in a decode loop.

---

## Key takeaways

- **Decode attention is one query reading the entire KV cache.** It reads all of K and V once per step, does $O(1)$ work per element, and its arithmetic intensity is about $2G/b$ — roughly 4 for Llama-3.1-8B in bf16, seventy times below the H100 ridge. It is bandwidth-bound, always.
- **The target is peak HBM bandwidth.** A decode step's attention read time is bytes divided by bandwidth: about 320 µs for 8k context on an H100, derived. A good kernel lands within a small multiple of that floor; measure achieved GB/s against the datasheet peak.
- **Online softmax is exact, not approximate.** Carry a running max, denominator, and accumulator; rescale the accumulator by $e^{m_{\text{old}} - m_{\text{new}}}$ on each new max; divide once at the end. The offset cancels in the ratio. This is the FlashAttention core and the heart of the kernel.
- **The rescale is the bug.** Forgetting to multiply the accumulator by the same factor as the denominator produces smooth, plausible, wrong output that grows with score range and context length. Test at length, with a non-multiple sequence length and a shuffled block table.
- **Grid is batch times KV heads.** One program per (sequence, KV head), looping the query-head group and reading each KV tile once, amortizes the GQA group — a 4x KV-traffic reduction on Llama-3.1-8B. Launching per query head silently quarters your throughput.
- **Split-K fills the card when the batch cannot.** Partition the KV range across instances, each producing a partial, then merge with a second reduction kernel using the associative online-softmax rule. Split only when the base grid has fewer programs than SMs; otherwise the extra launch is pure loss.
- **Launch shape is part of your CUDA-graph story.** Variable split-K grids replay badly under CUDA graphs; the persistent-kernel design uses a fixed grid with in-kernel work-stealing so the decode step captures and replays cleanly.
- **Build it to learn; ship the library.** Your Triton kernel can match FlashAttention 3 in the batch-1-long-decode regime (vLLM shows 100.7 percent in ~800 lines). It will lose everywhere the edge cases live, and the edge cases are the job.

---

## Further reading

- [What inference engineering is](/blog/machine-learning/inference-engineering/what-inference-engineering-is) — the series introduction and the `nanoserve` project.
- [The inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook) — the capstone that ties the kernels, cache, scheduler, and decoding layer together.
- [Paged KV cache: implementing blocks and a block table](/blog/machine-learning/inference-engineering/paged-kv-cache-implementing-blocks-and-a-block-table) — the memory layout this kernel reads through.
- [The KV cache append and gather kernel](/blog/machine-learning/inference-engineering/the-kv-cache-append-and-gather-kernel) — the write side of the same block-table indirection.
- [Triton for inference kernels, and when to stop writing CUDA](/blog/machine-learning/inference-engineering/triton-for-inference-kernels-and-when-to-stop-writing-cuda) — why Triton is the right tool for this memory-bound kernel.
- [Kernel fusion and FlashAttention: beating the memory wall](/blog/machine-learning/high-performance-computing/kernel-fusion-and-flashattention-beating-the-memory-wall) — the memory-wall argument and the fused online-softmax kernel in depth.
- [The roofline model: compute-bound vs memory-bound](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) — the arithmetic-intensity framing used in section 2.
- vLLM, ["Triton Attention Backend Deep Dive"](https://vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive) (2026-03-04) — grid design, parallel tiled softmax, the persistent kernel, and the FlashAttention 3 parity result.
- Tencent Hunyuan, ["HPC-Ops attention and MoE backends"](https://vllm.ai/blog/2026-07-06-vllm-hpc-ops) (2026-07-06) — the three-stage persistent attention kernel and uniform-tile bucketing for skewed batches.
- Milakov and Gimelshein, ["Online normalizer calculation for softmax"](https://arxiv.org/abs/1805.02867) (2018), and Dao et al., ["FlashAttention"](https://arxiv.org/abs/2205.14135) (2022) — the origins of online softmax and its fusion into attention.
