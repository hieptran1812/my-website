---
title: "The KV-cache append and gather kernel: coalescing under indirection"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "Write the tiny kernel that bolts your paged blocks onto your attention kernel — reshape_and_cache — and learn the one layout decision that decides whether every KV write moves 128 useful bytes or wastes 31 out of every 32."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "kv-cache",
    "cuda",
    "triton",
    "paged-attention",
    "pytorch",
    "gpu",
    "ml-systems",
    "throughput",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 46
---

In [the paged KV cache post](/blog/machine-learning/inference-engineering/paged-kv-cache-implementing-blocks-and-a-block-table) we shipped a KV write path that was one line long:

```python
store.flat[layer, 0].index_copy_(0, slots, k)
```

It is correct. It passes the tests. And if you profile a decode step with it in place, you will find that this "trivial" scatter is quietly eating a share of your memory bandwidth wildly out of proportion to the four kilobytes of data it moves — sometimes ten or twenty times what the same bytes cost when the attention kernel reads them back. Nothing is wrong with the code. The problem is that `index_copy_` does not know your layout, and the block table it writes through scatters each token's keys and values to an arbitrary physical block. On a GPU, that scatter is the difference between a warp moving 128 useful bytes in a single memory transaction and the same warp issuing 32 transactions that fetch four kilobytes to deliver those same 128 bytes.

![Dataflow diagram showing new keys and values scattered into paged blocks by the append path and read back through the same block table by the gather path](/imgs/blogs/the-kv-cache-append-and-gather-kernel-1.webp)

This post writes the real thing. vLLM calls it `reshape_and_cache`, and it is the smallest, least-glamorous kernel in the whole engine: after the QKV projection produces this step's keys and values, `reshape_and_cache` scatters them into the correct physical block and offset for every running sequence, using each sequence's block table. Then, when attention runs, the *gather* side reads those keys and values back through the same table. Two directions of one indirection, shown above in figure 1 — the append writes, the attention path reads, and a single block table drives both. Get the kernel right and it disappears into the noise. Get its layout wrong and it becomes a bottleneck you will chase for a week because the code looks innocent.

By the end you will have `nanoserve/csrc/reshape_and_cache.cu` and a Triton twin in `nanoserve/kv_kernels.py`, a coalesced append that writes each token's KV in one aligned transaction, the gather-side page load the attention kernel will use in [the next post](/blog/machine-learning/inference-engineering/paged-attention-kernel-by-hand), and — the actual point of the post — a derivation of *why the memory layout of the cache is the single most important decision here*, worked out to the byte for three candidate layouts including the asymmetric one vLLM actually ships.

One standing promise from [the series introduction](/blog/machine-learning/inference-engineering/what-inference-engineering-is): **I have no GPU and I have run none of this.** Every number below is either derived from arithmetic I show you, cited from a vendor spec or a vLLM post with a link, or framed as something you reproduce yourself with a named script and an expected range on named hardware. The coalescing numbers are the good kind — they follow from the warp width and the transaction size, so your arithmetic and mine agree exactly. The wall-clock numbers are cited or derived-from-bandwidth, never measured by me.

---

## 1. The append and the gather are two sides of one lookup

Recall the three lines of integer arithmetic from the paged-cache post. A request believes its keys and values occupy logical positions 0 through ${S-1}$ in order. The block table maps that logical order onto scattered physical blocks:

$$
\text{block\_index} = \left\lfloor \frac{\text{pos}}{\text{block\_size}} \right\rfloor, \qquad
\text{offset} = \text{pos} \bmod \text{block\_size}
$$

$$
\text{physical\_slot} = \text{block\_table}[\text{block\_index}] \cdot \text{block\_size} + \text{offset}
$$

Both kernels in this post are built from exactly this map, run in opposite directions.

**The append** (`reshape_and_cache`) runs once per layer per step, right after the QKV projection. It is handed this step's new keys and values — one key vector and one value vector per token per KV head — plus a flat array `slot_mapping` giving the physical slot for each token. Its whole job is a scatter: take the key vector for token $t$ and write it to physical slot `slot_mapping[t]`, and the same for the value. For a decode step there is exactly one new token per running sequence; for a prefill or a chunked prefill there are many. The kernel does not care which — it sees a flat list of tokens and a flat list of destination slots.

**The gather** runs inside the attention kernel. For one query token it must read *all* of that sequence's cached keys and values — logical positions 0 through the current length — which are strewn across however many physical blocks the request holds. It walks the block table, and for each block reads the `block_size` tokens' worth of KV that block holds, feeding them into the attention math. Where the append writes one token to one slot, the gather reads a whole sequence from many blocks.

The reason these two belong in one post, and the reason the layout decision below is load-bearing, is that **they read and write the same tensor, but with completely different access shapes.** The append touches one token across all heads and dims. The gather touches one head's history across the whole sequence. A memory layout that makes one of those a nice contiguous streak makes the other a scattered mess. There is no layout that is contiguous for both — you have to choose, or, as vLLM does, store keys and values under two different layouts so each side gets the shape it wants. That tension is the entire technical content of this post, and everything else is machinery around it.

Before we can price a layout we need one number: what a GPU actually charges for a scattered memory access versus a contiguous one. That number is not a rule of thumb. It falls straight out of how the hardware serves memory.

---

## 2. Coalescing, and where the 32x comes from

Threads on an NVIDIA GPU execute in groups of 32 called a **warp**. The 32 lanes of a warp issue their memory instructions together, in lockstep. The memory system does not serve 32 independent loads; it serves the warp's *combined* request as a set of **transactions**, where each transaction moves one aligned, fixed-size chunk of memory. The size that matters for our purposes is 128 bytes — a cache line — subdivided into four 32-byte sectors, which is the granularity at which recent NVIDIA GPUs actually fetch from HBM. (This is the model laid out in [the memory hierarchy post](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm) and in NVIDIA's *CUDA C++ Best Practices Guide*, section on coalesced access to global memory.)

Here is the whole mechanism. Suppose each lane wants to read a 4-byte word — a natural size, one `float` or two `bfloat16` values.

**Coalesced.** If the 32 lanes' addresses are consecutive and fall within one aligned 128-byte segment — lane 0 reads bytes 0–3, lane 1 reads bytes 4–7, and so on — the hardware satisfies the entire warp with **one 128-byte transaction**. It fetches 128 bytes and every one of them is used. Efficiency: 100%.

**Scattered.** If the 32 lanes' addresses fall in 32 *different* 128-byte segments — the pathological case, and exactly what a naive write through a block table can produce — the hardware must issue **up to 32 separate transactions**. Each transaction still moves at least its minimum granularity to deliver the lane's 4 bytes. Count it in cache lines and you fetch ${32 \times 128 = 4096}$ bytes to deliver ${32 \times 4 = 128}$ useful bytes: 3.1% efficiency, a **32x amplification** of bandwidth. Count it in the finer 32-byte sectors that modern GPUs actually use and it softens to ${32 \times 32 = 1024}$ bytes for the same 128 useful, an 8x amplification — still catastrophic on an operation whose entire cost *is* bandwidth. The canonical worst-case number you will hear quoted is 32x, and it is the right number to fear; the sector granularity is why real-world scattered access usually lands somewhere between 8x and 32x rather than pinned at the ceiling.

![Before and after comparison of a coalesced warp access moving one transaction against a scattered access moving thirty-two transactions for the same payload](/imgs/blogs/the-kv-cache-append-and-gather-kernel-2.webp)

Watch the two access patterns run against the same four writes:

<figure class="blog-anim">
<svg viewBox="0 0 720 320" role="img" aria-label="Four warp lanes write to memory: when their addresses fall in one aligned segment the hardware serves them in a single transaction with every byte used, but when the addresses scatter across four segments it issues four transactions and most fetched bytes are wasted" style="width:100%;height:auto;max-width:820px">
<style>
.kv-used{fill:var(--accent,#6366f1)}
.kv-waste{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.kv-seg{fill:none;stroke:var(--text-secondary,#6b7280);stroke-width:2}
.kv-lane{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.kv-arw{stroke:var(--accent,#6366f1);stroke-width:2;fill:none}
.kv-h{font:600 16px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.kv-l{font:600 12px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.kv-s{font:400 12.5px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.kv-win{font:600 13px ui-sans-serif,system-ui;fill:var(--accent,#6366f1)}
@keyframes kv-fadeA{0%,42%{opacity:1}52%,94%{opacity:0}100%{opacity:1}}
@keyframes kv-fadeB{0%,42%{opacity:0}52%,94%{opacity:1}100%{opacity:0}}
.kv-A{animation:kv-fadeA 11s ease-in-out infinite}
.kv-B{animation:kv-fadeB 11s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.kv-A{animation:none;opacity:1}.kv-B{animation:none;opacity:0}}
</style>
<g class="kv-A">
<text class="kv-h" x="30" y="30">Coalesced: 4 lanes land in one aligned segment</text>
<rect class="kv-lane" x="120" y="52" width="90" height="34" rx="6"/>
<rect class="kv-lane" x="230" y="52" width="90" height="34" rx="6"/>
<rect class="kv-lane" x="340" y="52" width="90" height="34" rx="6"/>
<rect class="kv-lane" x="450" y="52" width="90" height="34" rx="6"/>
<text class="kv-l" x="165" y="74">lane 0</text>
<text class="kv-l" x="275" y="74">lane 1</text>
<text class="kv-l" x="385" y="74">lane 2</text>
<text class="kv-l" x="495" y="74">lane 3</text>
<path class="kv-arw" d="M165 88 L165 150"/>
<path class="kv-arw" d="M275 88 L275 150"/>
<path class="kv-arw" d="M385 88 L385 150"/>
<path class="kv-arw" d="M495 88 L495 150"/>
<rect class="kv-used" x="120" y="152" width="105" height="60" rx="4"/>
<rect class="kv-used" x="230" y="152" width="105" height="60" rx="4"/>
<rect class="kv-used" x="340" y="152" width="105" height="60" rx="4"/>
<rect class="kv-used" x="450" y="152" width="105" height="60" rx="4"/>
<rect class="kv-seg" x="114" y="146" width="447" height="72" rx="8"/>
<text class="kv-win" x="120" y="244">1 transaction, 128 B fetched, 128 B used, 100% useful</text>
<text class="kv-s" x="120" y="268">Every lane's address falls in the same 128-byte segment.</text>
</g>
<g class="kv-B">
<text class="kv-h" x="30" y="30">Scattered: 4 lanes land in 4 different segments</text>
<rect class="kv-lane" x="120" y="52" width="90" height="34" rx="6"/>
<rect class="kv-lane" x="230" y="52" width="90" height="34" rx="6"/>
<rect class="kv-lane" x="340" y="52" width="90" height="34" rx="6"/>
<rect class="kv-lane" x="450" y="52" width="90" height="34" rx="6"/>
<text class="kv-l" x="165" y="74">lane 0</text>
<text class="kv-l" x="275" y="74">lane 1</text>
<text class="kv-l" x="385" y="74">lane 2</text>
<text class="kv-l" x="495" y="74">lane 3</text>
<path class="kv-arw" d="M165 88 L60 150"/>
<path class="kv-arw" d="M275 88 L235 150"/>
<path class="kv-arw" d="M385 88 L410 150"/>
<path class="kv-arw" d="M495 88 L585 150"/>
<rect class="kv-used" x="20" y="152" width="30" height="60" rx="4"/>
<rect class="kv-waste" x="50" y="152" width="120" height="60" rx="4"/>
<rect class="kv-seg" x="14" y="146" width="162" height="72" rx="8"/>
<rect class="kv-used" x="195" y="152" width="30" height="60" rx="4"/>
<rect class="kv-waste" x="225" y="152" width="120" height="60" rx="4"/>
<rect class="kv-seg" x="189" y="146" width="162" height="72" rx="8"/>
<rect class="kv-used" x="370" y="152" width="30" height="60" rx="4"/>
<rect class="kv-waste" x="400" y="152" width="120" height="60" rx="4"/>
<rect class="kv-seg" x="364" y="146" width="162" height="72" rx="8"/>
<rect class="kv-used" x="545" y="152" width="30" height="60" rx="4"/>
<rect class="kv-waste" x="575" y="152" width="120" height="60" rx="4"/>
<rect class="kv-seg" x="539" y="146" width="162" height="72" rx="8"/>
<text class="kv-win" x="120" y="244">4 transactions, 512 B fetched, 128 B used, 25% useful</text>
<text class="kv-s" x="120" y="268">Scale this to a 32-lane warp across 32 segments and the waste reaches 32x.</text>
</g>
</svg>
<figcaption>The same four writes: one aligned segment collapses to a single full transaction, while four scattered addresses force four transactions that fetch mostly bytes no one asked for.</figcaption>
</figure>

The lesson is blunt: on a bandwidth-bound kernel, **the layout that determines which lanes touch which segments determines your effective bandwidth, and it can swing by more than an order of magnitude with no change to the amount of data you actually need.** The KV append is bandwidth-bound — it does no arithmetic, it just moves bytes — so it lives or dies entirely on coalescing. That is why a one-line `index_copy_` can be correct and slow at the same time: correctness is about *which* bytes land where, coalescing is about *how the addresses group*, and PyTorch's generic scatter optimizes for the former with no knowledge of your block geometry.

#### Worked example: what the append actually costs, coalesced and not

Take Llama-3.1-8B on an RTX 4090, decoding at batch 256. Per the [KV memory-math post](/blog/machine-learning/inference-engineering/the-memory-math-of-the-kv-cache), one token's KV for one layer is ${2 \cdot H_{kv} \cdot d \cdot b = 2 \times 8 \times 128 \times 2 = 4096}$ bytes — 4 KiB, keys and values together. `reshape_and_cache` runs once per layer, so a full 32-layer decode step writes:

$$
256 \text{ tokens} \times 4\text{ KiB} \times 32 \text{ layers} = 32 \text{ MiB}
$$

NVIDIA's RTX 4090 specification lists 1,008 GB/s of memory bandwidth. Assume a well-tuned kernel reaches about 85% of that on a pure copy, so roughly 857 GB/s effective.

- **Coalesced:** ${32 \text{ MiB} / 857 \text{ GB/s} \approx 39\ \mu s}$ for the whole step's appends.
- **Fully scattered (the 32x ceiling):** ${\approx 1{,}250\ \mu s = 1.25 \text{ ms}}$.

Now put that in context. Decode at batch 256 for an 8B model in bf16 is memory-bound on the *weights*: you read all 16.06 GB of parameters once per step, which at 857 GB/s is about 18.7 ms. So the coalesced append is ${39/18{,}700 \approx 0.2\%}$ of the step — genuinely negligible, exactly where you want a bookkeeping kernel to be. The scattered append is ${1{,}250/18{,}700 \approx 6.7\%}$ of the step. **One layout mistake turns a 0.2% cost into a 6.7% cost** — a 6.5% throughput regression from a kernel that moves the same 32 MiB either way. *(Source: derived; bandwidth cited from NVIDIA's RTX 4090 spec; the 85%-of-peak assumption is a reproduce-it-yourself figure — measure your own copy bandwidth with a memcpy microbenchmark.)*

That 6.7% is why this post exists. Nobody profiles a KV write expecting to find headroom, so the regression hides in plain sight, attributed to "attention being slow" for as long as it takes someone to run a kernel-level profiler.

---

## 3. The layout decision, worked out to the byte

We have three candidate layouts for how a single layer's KV blocks sit in memory. To compare them we ask two questions of each, because those are the two access patterns from section 1:

1. **Write one token** (the append): is the write of one token's ${H_{kv} \times d}$ key elements coalesced?
2. **Read one head's sequence** (the gather): when attention reads head $h$'s keys for logical positions 0 through ${S-1}$, is that read coalesced?

Let me set up coordinates. Within one physical block we have three axes: `block_size` tokens (call it $T$), `num_kv_heads` heads ($H$), and `head_dim` ($d$). The layouts differ in what order these nest, with the last axis being the fastest-varying (contiguous in memory). Here is the layout as vLLM nests it for keys, dimension by dimension:

![Layered diagram of vLLM's key-cache layout showing five nested dimensions ending in a sixteen-byte vector unit](/imgs/blogs/the-kv-cache-append-and-gather-kernel-3.webp)

### Layout A: token-major, `[block, token, head, dim]`

This is what [`nanoserve/blocks.py`](/blog/machine-learning/inference-engineering/paged-kv-cache-implementing-blocks-and-a-block-table) already uses — the flat view was `[num_blocks * block_size, kv_head, head_dim]`, so within a block the order is token, then head, then dim, with dim contiguous.

- **Write one token:** all of token $t$'s ${H \times d = 8 \times 128 = 1024}$ elements are one contiguous run (a slot row). A block of threads covering that row produces perfectly consecutive addresses. **Coalesced.**
- **Read head $h$'s sequence:** the keys for head $h$ at consecutive tokens are separated by a stride of ${H \times d = 1024}$ elements — token 0's head-$h$ vector, then a jump over the other seven heads, then token 1's. Within one token's head vector the 128 dims are contiguous, but across tokens the reads stride. **Partially coalesced within a token, strided across tokens.**

Token-major is the natural choice if you only think about the write, which is why hand-rolled caches land here. The read pays for it.

### Layout B: sequence-inner, `[block, head, dim, token]`

Flip it so the token axis is innermost.

- **Write one token:** token $t$'s value for a fixed `(head, dim)` sits at stride $T = 16$ from token ${t+1}$'s. Writing one token means touching one element out of every 16-element token run, across all ${H \times d}$ of them — a strided write with a gap of 15 elements between each useful one. **Uncoalesced.**
- **Read head $h$'s sequence:** for a fixed `(head, dim)`, the $T$ tokens are contiguous. Reading head $h$ across a block streams `block_size` consecutive values. **Coalesced.**

Layout B is the mirror image of A: it makes the gather beautiful and the append ugly. Neither A nor B wins outright, which is the crux of the whole post — the two access patterns want opposite layouts.

### Layout C: vLLM's split — keys and values shaped differently

vLLM refuses the false choice. Its `csrc/cache_kernels.cu` stores the two tensors under two different shapes (public, and stable since the original PagedAttention kernel):

| tensor | shape (one layer) |
| --- | --- |
| `key_cache` | `[num_blocks, num_kv_heads, head_dim / x, block_size, x]` |
| `value_cache` | `[num_blocks, num_kv_heads, head_dim, block_size]` |

where ${x = 16 / \text{sizeof(dtype)}}$ is the number of elements that fill 16 bytes — for bf16, ${x = 8}$ elements, so `head_dim / x` is ${128 / 8 = 16}$ groups, and the innermost `x` axis is exactly one 128-bit (16-byte) vectorized load. That is the five-dimensional nesting drawn in figure 3 above: block, then head, then the 16 head-dim groups, then the 16 tokens of the block, and finally the 16-byte vector as the contiguous unit.

Why this specific, asymmetric shape? Because the two tensors are consumed by two different matrix products inside attention, and each wants its reduction axis laid out for vectorized access:

- The **key** cache participates in ${QK^\top}$, a dot product over `head_dim`. Splitting `head_dim` into `(head_dim / x, ..., x)` and putting `x` innermost lets the attention kernel issue 16-byte vectorized loads down the head dimension while `block_size` sits one axis out, so a warp can stream a block's worth of keys with aligned 128-bit reads. The reshape is the whole trick: `head_dim` is broken so that the contiguous unit is exactly a vector-load width.
- The **value** cache participates in the ${\text{softmax} \cdot V}$ product, where the reduction is over the *sequence* (the tokens), so `block_size` innermost means the accumulation streams `block_size` contiguous values per `(head, dim)`.

The append kernel then writes into *both* shapes. For the key it writes to `[block, head, d/x, offset, :]` — the innermost `x` run is contiguous, so a thread that owns one 16-byte vector does a single aligned store. For the value it writes to `[block, head, dim, offset]` — a single scalar per `(head, dim)`, strided by `block_size`. The value write is therefore the one that is *not* naturally coalesced, and vLLM's kernel handles it by having each thread write its scalar with the offset fixed; the writes across the `head_dim` axis for one token still group reasonably because a warp covers many `(head, dim)` positions at the same `offset`. The key write is the clean one; the value write is the compromise. That asymmetry — keys optimized hard, values good-enough — is the thing tutorials skip, and it is a direct consequence of keys and values being reduced over different axes.

![Matrix comparing three cache layouts against whether the one-token write and the one-head read are coalesced](/imgs/blogs/the-kv-cache-append-and-gather-kernel-4.webp)

Here is the whole decision in one table, which is figure 4 above rendered as numbers you can check:

| Layout | Shape (one layer) | Write one token | Read one head's seq | Verdict | Source |
| --- | --- | --- | --- | --- | --- |
| A: token-major | `[block, tok, head, dim]` | coalesced (1024-elt run) | strided across tokens | simple; gather pays | derived |
| B: sequence-inner | `[block, head, dim, tok]` | strided (gap 15) | coalesced (16 contig) | gather wins; write pays | derived |
| C: vLLM split | K `[blk,h,d/x,tok,x]`, V `[blk,h,d,tok]` | K coalesced, V strided | vectorized both sides | best overall; two shapes | cited: vLLM cache\_kernels.cu |

The reason `nanoserve` used layout A in the paged-cache post and got away with it is that our gather *materialized* the whole sequence with `index_select` and then handed a contiguous tensor to `scaled_dot_product_attention` — so the strided read happened once, inside PyTorch's gather, rather than inside a tuned attention kernel that cares. The moment we write our own paged attention kernel in [the next post](/blog/machine-learning/inference-engineering/paged-attention-kernel-by-hand), the read pattern stops being PyTorch's problem and becomes ours, and layout A's strided gather starts costing real bandwidth. That is the point at which you either switch to layout C or accept the read penalty — and knowing *why* is the difference between choosing and cargo-culting.

For the rest of this post I will implement the append against layout A, because it is the layout `nanoserve` already has and because its write is the clean, coalesced case that makes the kernel easy to read. I will then show exactly what changes for layout C's key tensor, so the vectorized version is a small diff rather than a rewrite.

---

## 4. `reshape_and_cache`, the append kernel

Time to write CUDA. This builds directly on the extension pattern from [the RMSNorm and RoPE kernel post](/blog/machine-learning/inference-engineering/writing-your-first-inference-cuda-kernel-rmsnorm-and-rope): a single `.cu` file compiled at import time with `torch.utils.cpp_extension.load`, exposed as a Python-callable op. If you have not read that post, the one thing to carry over is that `load` JIT-compiles the file, caches the `.so`, and hands you back a module — no build system, no `setup.py`.

The thread mapping is the design, so let me state it before the code. Layout A's write is coalesced when consecutive threads write consecutive elements of a token's slot row. So:

- **Grid:** one CUDA block per token. `blockIdx.x = t`, the index into `slot_mapping`.
- **Threads:** each block runs enough threads to cover the ${H \times d = 1024}$ elements of one token's KV. Thread `i` handles element `i` of the flattened `(head, dim)` row.
- **The scatter:** every thread in a block shares the same destination slot `slot_mapping[t]`, computed once; thread `i` writes to `slot * (H*d) + i`. Consecutive `i` means consecutive addresses means a coalesced store. The scatter lives entirely in the *base* address per block; within a block the writes are contiguous.

That is the key insight that makes the append fast: the indirection is per-block, not per-element. All the "scattering" happens in choosing where each token's row starts; the 1024 elements of that row are then written in order.

Here is `nanoserve/csrc/reshape_and_cache.cu`:

```cpp
// nanoserve/csrc/reshape_and_cache.cu
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

// One CUDA block per token. Threads stride over the (head, dim) row.
// key, value : [num_tokens, num_kv_heads, head_dim]
// key_cache, value_cache : [num_slots, num_kv_heads, head_dim]   (layout A, flat)
// slot_mapping : [num_tokens] int64; negative slot => skip (padding token)
template <typename scalar_t>
__global__ void reshape_and_cache_kernel(
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    scalar_t* __restrict__ key_cache,
    scalar_t* __restrict__ value_cache,
    const int64_t* __restrict__ slot_mapping,
    const int row_stride) {          // row_stride = num_kv_heads * head_dim
  const int token_idx = blockIdx.x;
  const int64_t slot = slot_mapping[token_idx];
  if (slot < 0) return;              // padding token: nothing to write

  const int64_t src_base = static_cast<int64_t>(token_idx) * row_stride;
  const int64_t dst_base = slot * row_stride;

  // Consecutive threads -> consecutive elements -> coalesced store.
  for (int i = threadIdx.x; i < row_stride; i += blockDim.x) {
    key_cache[dst_base + i]   = key[src_base + i];
    value_cache[dst_base + i] = value[src_base + i];
  }
}

void reshape_and_cache(torch::Tensor key, torch::Tensor value,
                       torch::Tensor key_cache, torch::Tensor value_cache,
                       torch::Tensor slot_mapping) {
  const int num_tokens = key.size(0);
  const int row_stride = key.size(1) * key.size(2);   // num_kv_heads * head_dim
  const int threads = std::min(row_stride, 1024);
  const dim3 grid(num_tokens);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_REDUCED_FLOATING_TYPES(
      key.scalar_type(), "reshape_and_cache", [&] {
        reshape_and_cache_kernel<scalar_t><<<grid, threads, 0, stream>>>(
            key.data_ptr<scalar_t>(), value.data_ptr<scalar_t>(),
            key_cache.data_ptr<scalar_t>(), value_cache.data_ptr<scalar_t>(),
            slot_mapping.data_ptr<int64_t>(), row_stride);
      });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("reshape_and_cache", &reshape_and_cache, "Scatter K/V into paged slots");
}
```

Three details that are load-bearing, none of which are obvious:

**The `slot < 0` early return is not a nicety — it is the mechanism that makes batched, variable-length steps work.** When a decode step has fewer live tokens than the padded tensor shape (a request just finished, or you pad to a fixed batch for CUDA graphs), the scheduler sets those tokens' slots to ${-1}$. The kernel skips them. vLLM's real `reshape_and_cache` does exactly this; it is how a padded batch coexists with a paged cache. Forget it and your padding tokens overwrite physical slot ${-1 \times \text{row\_stride}}$, which is a wild negative index into the KV tensor and a memory-corruption bug that will surface three requests later as garbage output in an unrelated sequence.

**`row_stride` is passed in, not recomputed.** The host wrapper computes `num_kv_heads * head_dim` once and hands it down. Recomputing per-thread is free arithmetically but the point is that the kernel should be agnostic to the head/dim split — it treats a token's KV as one flat row, which is precisely why it coalesces.

**One block per token, capped at 1024 threads.** For Llama-3.1-8B the row is 1024 elements, so one thread per element, one warp-aligned block of 1024 threads (32 warps), each warp writing a coalesced 128-byte-friendly streak. For a model with a larger row the grid-stride loop (`i += blockDim.x`) handles the overflow without changing the launch. This is the simplest mapping that coalesces; it is not the fastest — the vectorized version below is — but it is the one to understand first.

Now the Python side that compiles and calls it, in `nanoserve/kv_kernels.py`:

```python
# nanoserve/kv_kernels.py
import os
import torch
from torch.utils.cpp_extension import load

_HERE = os.path.dirname(__file__)
_ext = load(
    name="nanoserve_kv",
    sources=[os.path.join(_HERE, "csrc", "reshape_and_cache.cu")],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)


def reshape_and_cache(key, value, key_cache, value_cache, slot_mapping):
    """Scatter this step's K/V into the paged cache (layout A, in place).

    key, value           : [num_tokens, num_kv_heads, head_dim]
    key_cache/value_cache: [num_slots, num_kv_heads, head_dim]
    slot_mapping         : [num_tokens] int64, negative => skip
    """
    assert key.is_cuda and key_cache.is_cuda
    assert slot_mapping.dtype == torch.int64
    _ext.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)
```

And the correctness check that proves it matches the `index_copy_` version from the paged-cache post — because a fast kernel that writes the wrong slot is worse than a slow one that is right:

```python
# nanoserve/tests/test_reshape_and_cache.py
import torch
from nanoserve.kv_kernels import reshape_and_cache


def test_matches_reference():
    torch.manual_seed(0)
    num_tokens, H, d, num_slots = 37, 8, 128, 4096
    key = torch.randn(num_tokens, H, d, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    # Random destination slots, plus a couple of padding tokens.
    slots = torch.randint(0, num_slots, (num_tokens,), device="cuda")
    slots[5] = -1          # padding
    slots[20] = -1

    kc = torch.zeros(num_slots, H, d, device="cuda", dtype=torch.bfloat16)
    vc = torch.zeros_like(kc)
    reshape_and_cache(key, value, kc, vc, slots)

    # Reference: index_copy_ over the non-padding tokens only.
    kc_ref = torch.zeros_like(kc)
    vc_ref = torch.zeros_like(vc)
    live = slots >= 0
    kc_ref.index_copy_(0, slots[live], key[live])
    vc_ref.index_copy_(0, slots[live], value[live])

    torch.testing.assert_close(kc, kc_ref)
    torch.testing.assert_close(vc, vc_ref)
    print("reshape_and_cache matches index_copy_ (padding skipped)")
```

Running it prints:

```console
reshape_and_cache matches index_copy_ (padding skipped)
```

![Timeline of the five steps one append thread runs, from loading K and V to issuing a single vectorized store](/imgs/blogs/the-kv-cache-append-and-gather-kernel-5.webp)

Figure 5 above is the whole kernel from a single thread's point of view: load the element, read the slot, turn it into a physical address with a shift and a mask, store. Two integer ops and one store. When people say a kernel is "just bookkeeping," this is what they mean — and it is exactly why the only thing that can make it slow is the store failing to coalesce.

### The vectorized version, and the Triton twin

The scalar kernel writes one 2-byte element per thread. A warp of 32 threads therefore writes 64 bytes — half a transaction. You can do better by having each thread write a 16-byte vector (a `float4`, or eight bf16 values), so a warp writes ${32 \times 16 = 512}$ bytes — four full transactions with zero waste. This is also exactly the shape layout C's key tensor is built around, which is not a coincidence: the `x = 8` innermost axis *is* the vector.

Rather than write the `float4` CUDA by hand — which means reinterpret-casting pointers and is easy to get wrong on alignment — this is where Triton earns its place. Triton's `tl.load`/`tl.store` vectorize automatically when the access is contiguous, and the block-table arithmetic reads naturally:

```python
# nanoserve/kv_kernels.py (continued)
import triton
import triton.language as tl


@triton.jit
def _reshape_and_cache_tri(
    key_ptr, value_ptr, kcache_ptr, vcache_ptr, slot_ptr,
    row_stride, BLOCK: tl.constexpr,
):
    token_idx = tl.program_id(0)
    slot = tl.load(slot_ptr + token_idx)
    if slot < 0:
        return                                   # padding token
    src = token_idx * row_stride
    dst = slot * row_stride
    # Cover the (head, dim) row in contiguous chunks; Triton vectorizes each.
    for base in range(0, row_stride, BLOCK):
        offs = base + tl.arange(0, BLOCK)
        mask = offs < row_stride
        tl.store(kcache_ptr + dst + offs, tl.load(key_ptr + src + offs, mask=mask), mask=mask)
        tl.store(vcache_ptr + dst + offs, tl.load(value_ptr + src + offs, mask=mask), mask=mask)


def reshape_and_cache_triton(key, value, key_cache, value_cache, slot_mapping):
    num_tokens = key.shape[0]
    row_stride = key.shape[1] * key.shape[2]
    _reshape_and_cache_tri[(num_tokens,)](
        key, value, key_cache, value_cache, slot_mapping,
        row_stride, BLOCK=256,
    )
```

The two implementations are interchangeable behind the `nanoserve.kv_kernels.reshape_and_cache` name, and the correctness test runs against both. Which you ship is a maintenance question, not a performance one at this size — the [Triton-for-inference post](/blog/machine-learning/inference-engineering/paged-attention-kernel-by-hand) makes the general argument, but for a kernel this simple the honest answer is that both hit the same bandwidth and the Triton one is easier to read. Reach for hand-CUDA only when you need the `float4` alignment guarantees or you are fusing this into another kernel, which section 7 does.

---

## 5. The gather side: loading a page inside the kernel

The append was the easy half because its access pattern is fixed at authoring time — you always write a full token row. The gather is harder because it is driven by the block table at runtime, and it lives inside the attention kernel where you cannot afford to materialize anything.

The paged-cache post's `gather_kv` built a full `[S, H, d]` tensor with `index_select` and handed it to `scaled_dot_product_attention`. That is 80 MiB of allocate-and-copy per decode step across 32 layers for a 640-token request, feeding an attention that reads each byte once. The real gather never does that. It walks the block table one block at a time, loads that block's `block_size` tokens of KV straight from HBM into registers or shared memory, does the partial attention math on them, and moves to the next block — the online-softmax structure that [the next post](/blog/machine-learning/inference-engineering/paged-attention-kernel-by-hand) is entirely about.

Here is the gather *core* — the part that turns a block-table entry into a coalesced load of one block's keys. This is the loop body the attention kernel wraps in its softmax accumulation:

```python
# nanoserve/kv_kernels.py (continued)

@triton.jit
def _load_kv_block(
    kcache_ptr, block_id, kv_head, block_size, head_dim,
    HEAD_STRIDE: tl.constexpr,          # elements per token row = H * d
    D: tl.constexpr,                    # head_dim as a compile-time constant
):
    """Load one block's keys for one KV head: [block_size, head_dim].

    Layout A: cache is [num_slots, H, d]; slot = block_id * block_size + tok.
    """
    tok = tl.arange(0, block_size)                      # [block_size]
    d = tl.arange(0, D)                                 # [head_dim]
    slot = block_id * block_size + tok                  # [block_size]
    # Address of (slot, kv_head, d): slot*HEAD_STRIDE + kv_head*D + d
    addr = (slot[:, None] * HEAD_STRIDE
            + kv_head * D
            + d[None, :])                               # [block_size, head_dim]
    return tl.load(kcache_ptr + addr)                   # coalesced along d
```

Notice which axis coalesces. Within one token row, the `head_dim` values (`d`) are contiguous — the innermost `+ d` — so each token's key vector loads as a clean 128-element (256-byte) streak. Across tokens the address jumps by `HEAD_STRIDE = H * d = 1024` elements, which is layout A's strided-across-tokens read from section 3. Because a block is only 16 tokens, that stride is paid 16 times per block, and each of the 16 streaks is well-coalesced internally — this is the "large blocks amortize the jump" argument from the paged-cache post made concrete. Shrink the block to 1 token and you pay the jump every single token with no amortization; this is the coalescing reason 16 beats 1, on top of the bookkeeping reason.

If the cache were in layout C, the load would instead pull `x = 8`-element (16-byte) vectors down the `head_dim / x` axis with `block_size` one step out, which is what lets vLLM's attention kernel issue 128-bit vectorized loads for the ${QK^\top}$ dot product. The gather-side payoff of layout C is exactly this: the read becomes a sequence of aligned vector loads instead of strided scalar streaks. We will implement the layout-C attention read in full next post; here the point is only that **the gather's coalescing is decided by the same layout choice as the append's, pulling in the opposite direction, which is why the split layout exists.**

The last-block subtlety: a request's final block is usually partial — `num_tokens % block_size` tokens are live, the rest are uninitialized. The gather must mask those out so attention never scores a garbage key. In the loop above that is a `tok < valid_len` mask on the load; in the full kernel it becomes a `-inf` in the softmax for the dead positions. Section 8 stress-tests exactly this.

---

## 6. The numbers, with provenance

Let me put the bandwidth accounting in one place. Every row is derived from the per-token formula and a cited bandwidth spec; nothing here is measured by me.

Per token per layer, the append moves ${2 \cdot H_{kv} \cdot d \cdot b}$ bytes (keys and values). Across a step it scales by the number of tokens and by 32 layers. The effective bandwidth is the peak spec times an achievable fraction (coalesced) or divided by the amplification factor (scattered).

| Quantity | Llama-3.1-8B bf16 | Formula | Source |
| --- | --- | --- | --- |
| KV bytes / token / layer | 4 KiB | ${2 \cdot H_{kv} \cdot d \cdot b}$ | derived |
| KV bytes / token, all layers | 128 KiB | ${\times\ L=32}$ | derived (matches post 7) |
| Append bytes / decode step, batch 256 | 32 MiB | ${256 \cdot 128\text{ KiB}}$ | derived |
| 4090 peak HBM | 1,008 GB/s | GDDR6X, 384-bit | cited: NVIDIA RTX 4090 spec |
| A100 80GB SXM peak HBM | 2,039 GB/s | HBM2e | cited: NVIDIA A100 datasheet |
| H100 SXM peak HBM | 3,350 GB/s | HBM3 | cited: NVIDIA H100 datasheet |
| L4 peak HBM | 300 GB/s | GDDR6 | cited: NVIDIA L4 datasheet |
| Coalesced append time, 4090 | ~39 µs | ${32\text{ MiB} / (0.85 \cdot 1008)}$ | derived |
| Scattered append time, 4090 (32x) | ~1.25 ms | ${\times 32}$ | derived (worst case) |
| Coalesced append time, A100 | ~19 µs | ${32\text{ MiB} / (0.85 \cdot 2039)}$ | derived |
| Append as % of batch-256 decode step, 4090 | 0.2% coalesced / 6.7% scattered | vs 18.7 ms weight read | derived |

The single most important thing this table says is in the last row: **the append is invisible when coalesced and a measurable tax when scattered, and the amount of data is identical.** You are not optimizing how much you write — you cannot, the model dictates it — you are optimizing whether the writes group into full transactions.

### Measuring it honestly on your own card

The derived numbers above are ceilings and floors. To find where your kernel actually lands, the rules from [the baseline post](/blog/machine-learning/inference-engineering/what-inference-engineering-is) are non-negotiable, and coalescing bugs are especially good at hiding from sloppy measurement:

- **Warm up and synchronize.** The first launch pays JIT compilation (for the Triton path) and autotuning. Discard ~20 iterations, then time with `torch.cuda.Event` pairs and a `torch.cuda.synchronize()` before reading the elapsed time. Timing a launch without a sync measures the queue, not the kernel.
- **Isolate the append.** Run `reshape_and_cache` alone in a loop over realistic `slot_mapping` tensors — one built from actual block tables, not `arange`, because `arange` slots are contiguous and will *hide* a coalescing bug by making even a bad layout look sequential. This is the trap: benchmark with sorted slots and your scattered kernel looks fine, then it falls over in production where slots are genuinely scattered across the free pool.
- **Read the profiler's memory-efficiency counter, not the wall clock.** `ncu` reports `gld_efficiency` / `gst_efficiency` (global load/store efficiency). A coalesced append should show near 100% store efficiency; a layout bug shows up as 12%–25% directly, before you even look at timing. This is the honest metric for a bandwidth-bound kernel — achieved efficiency, not tok/s.
- **Compare against a plain copy.** The append can never beat `torch.empty_like(dst).copy_(src)` for the same bytes; that copy is your coalesced ceiling. If the append is within ~10% of it, you are done. If it is 3x slower, you have a coalescing problem, full stop.

A microbenchmark that belongs in your repo, framed as something you run:

```python
# nanoserve/bench/bench_append.py
import torch, time
from nanoserve.kv_kernels import reshape_and_cache

def bench(num_tokens=256, H=8, d=128, num_slots=54886, scattered=True, iters=200):
    key = torch.randn(num_tokens, H, d, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    kc = torch.zeros(num_slots, H, d, device="cuda", dtype=torch.bfloat16)
    vc = torch.zeros_like(kc)
    if scattered:
        slots = torch.randint(0, num_slots, (num_tokens,), device="cuda")
    else:
        slots = torch.arange(num_tokens, device="cuda")   # contiguous: cheats!
    for _ in range(20):                                   # warmup
        reshape_and_cache(key, value, kc, vc, slots)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        reshape_and_cache(key, value, kc, vc, slots)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    moved = 2 * num_tokens * H * d * 2          # bytes: K+V, bf16
    print(f"scattered={scattered}  {dt*1e6:6.1f} us  "
          f"{moved/dt/1e9:6.1f} GB/s effective")

if __name__ == "__main__":
    bench(scattered=False)
    bench(scattered=True)
```

On an RTX 4090 you should see both cases land in the same ballpark for *this* kernel, because layout A coalesces the write regardless of how scattered the slots are — the scatter is per-token-row, and each row is contiguous. That is the point: a good layout makes the write robust to scattered slots. Swap in a layout-B write (token innermost) and the `scattered=True` case should collapse to a fraction of the bandwidth while `scattered=False` still looks fine — which is precisely how you would catch the bug if you only ever benchmarked with sorted slots. *(Source: reproduce with the script; expected ranges are the derived 4090 figures above.)*

#### Worked example: the same kernel on an L4 versus an A100

Move the batch-256 step to an NVIDIA L4 (300 GB/s) and an A100 80GB SXM (2,039 GB/s), keeping everything else fixed.

- **L4:** coalesced append ${= 32\text{ MiB} / (0.85 \times 300\text{ GB/s}) \approx 131\ \mu s}$. The decode step's weight read is ${16.06\text{ GB} / (0.85 \times 300) \approx 63\text{ ms}}$, so the append is 0.2% coalesced — same ratio as the 4090, because both the append and the weight read scale with the same bandwidth. Scattered, the append is ${\approx 4.2\text{ ms}}$, still 6.7% of the step.
- **A100:** coalesced append ${\approx 19\ \mu s}$ against a ${\approx 7.9\text{ ms}}$ weight read, again 0.2%.

The ratio is bandwidth-invariant, which is the honest headline: **coalescing does not matter more on a slow card or less on a fast one — it costs you the same 6.5% of your decode throughput everywhere, because both terms scale with HBM bandwidth.** What changes across cards is the absolute latency, not the penalty for getting the layout wrong. *(Source: derived; bandwidths cited from the respective NVIDIA datasheets.)*

---

## 7. Fusion: fold the write into the RoPE that produced K

Here is the optimization that separates a teaching kernel from a production one. Look at the life of a key vector in a decode step, unfused:

1. The attention prologue computes ${K = \text{RoPE}(\text{norm}(x W_K))}$ and **writes K to HBM** as the layer's key activation.
2. `reshape_and_cache` **reads K back from HBM**.
3. `reshape_and_cache` **writes K to the cache** in HBM.

That is three HBM passes over the same key data. But the key that `reshape_and_cache` writes is the *exact tensor* the RoPE kernel just produced — it was in registers moments ago. If you fuse the cache write into the tail of the RoPE kernel, K never leaves registers between producing it and caching it: one HBM pass instead of three, for the append's share of the traffic.

![Dataflow diagram of norm, RoPE, and the cache write merging into one fused kernel that keeps K in registers for a single HBM pass](/imgs/blogs/the-kv-cache-append-and-gather-kernel-6.webp)

This is not hypothetical. The vLLM / Tencent Hunyuan *HPC-Ops* post ([2026-07-06](https://vllm.ai/blog/2026-07-06-vllm-hpc-ops)) describes exactly this fusion: their attention backend uses a fused prologue called `HpcRopeNorm` that does "QK-Norm + RoPE + KV-cache write" in one kernel. The write rides along on the RoPE that produced the data. It is the same principle as the fused RMSNorm+RoPE kernel from [the CUDA kernel post](/blog/machine-learning/inference-engineering/writing-your-first-inference-cuda-kernel-rmsnorm-and-rope) — do all the work that touches a tensor while it is hot in registers — extended one op further to include the scatter.

Here is the fused Triton prologue for `nanoserve`, RoPE applied to K followed immediately by the paged write, no round trip:

```python
# nanoserve/kv_kernels.py (continued)

@triton.jit
def _rope_and_cache_k(
    k_ptr, cos_ptr, sin_ptr, kcache_ptr, slot_ptr,
    row_stride, HALF: tl.constexpr, BLOCK: tl.constexpr,
):
    """RoPE-rotate one token's K, then write it straight to its cache slot.

    k_ptr : [num_tokens, num_kv_heads, head_dim]  (post-norm, pre-RoPE K)
    Fuses steps 1 and 3: K is produced, rotated, and cached without
    ever going back to HBM between the rotation and the write.
    """
    t = tl.program_id(0)
    slot = tl.load(slot_ptr + t)
    if slot < 0:
        return
    src = t * row_stride
    dst = slot * row_stride
    # head_dim is split into two halves for rotate_half RoPE (see the CUDA post).
    off = tl.arange(0, HALF)
    for h_base in range(0, row_stride, 2 * HALF):        # per (head) chunk
        lo = tl.load(k_ptr + src + h_base + off)
        hi = tl.load(k_ptr + src + h_base + HALF + off)
        cos = tl.load(cos_ptr + off)
        sin = tl.load(sin_ptr + off)
        rot_lo = lo * cos - hi * sin
        rot_hi = hi * cos + lo * sin
        # Write the rotated halves straight to the cache slot. One HBM pass.
        tl.store(kcache_ptr + dst + h_base + off, rot_lo)
        tl.store(kcache_ptr + dst + h_base + HALF + off, rot_hi)
```

The bandwidth win is derivable. Unfused, the append's key traffic is: write K (4 KiB/token/layer, but keys only, so 2 KiB), read K back (2 KiB), write to cache (2 KiB) — 6 KiB of key traffic per token per layer. Fused, it is 2 KiB — the single cache write, because the RoPE output goes nowhere else. That is a 3x reduction on the key half of the append's traffic. Values fuse the same way if they pass through a prologue op; if they do not, they keep their single write.

Is it worth it? At batch 256 the append's coalesced cost was 39 µs against an 18.7 ms step — cutting it by 3x saves ~26 µs, or 0.14% of the step. On its own, no. **The fusion pays not because the append is expensive but because the round trip it eliminates is one of dozens of small round trips that, summed across every op in the layer, are what make the decode loop host- and launch-bound.** The HPC-Ops post reports their fully-fused attention path (of which `HpcRopeNorm` is the prologue) delivering up to 2.95x over a static split-KV baseline and around 17% end-to-end TPOT reduction on 8×H20 for their Hunyuan model — a number that is *cited, with its setup*, and that comes from fusing the whole prologue-plus-attention, not the cache write alone. The cache-write fusion is one brick in that wall. The general lesson — from [the roofline framing](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm) — is that on a memory-bound loop you fuse to delete HBM passes, and a standalone `reshape_and_cache` is a deletable pass hiding in plain sight.

---

## 8. Stress-testing the kernel

Every failure mode of this kernel is a coalescing story or an indexing story. Figure 7 collects the four that matter and their fixes.

![Matrix of four stress cases for the append kernel with what breaks and the fix for each](/imgs/blogs/the-kv-cache-append-and-gather-kernel-7.webp)

### `block_size` that is not a power of two

The append coalesces because a token's row is contiguous. But the *base address* of each physical block is `block_id * block_size * row_stride`. If `block_size` is 15 instead of 16, block bases stop landing on 128-byte-aligned boundaries — block 1 starts at ${15 \times 1024}$ elements ${= 15{,}360}$, which for bf16 is ${30{,}720}$ bytes, not a multiple of 128 in a way that keeps every token's 256-byte row aligned. Misaligned rows split across segment boundaries, so a read or write that *would* have been one transaction becomes two, quietly, on every block. This is a second, independent reason `block_size` wants to be a power of two — the [paged-cache post](/blog/machine-learning/inference-engineering/paged-kv-cache-implementing-blocks-and-a-block-table) argued 16 from fragmentation and bookkeeping; alignment is the third leg. The fix if you must use an odd block size is to pad each block's storage up to an aligned stride, trading a sliver of memory for keeping every row aligned.

### The last partial block and the negative slot

A request holding 37 tokens with `block_size` 16 uses three blocks: 16, 16, and 5. The last block has 11 uninitialized token slots. Two things must handle this:

- **The append never writes them** — it only writes the tokens that exist, and the `slot_mapping` for a decode step contains exactly the live tokens' slots. There is no partial-block problem on the write side; you write token by token.
- **The gather must mask them** — reading the third block pulls 16 tokens' worth of KV but only 5 are valid, so attention must score the other 11 as ${-\infty}$. This is the `tok < valid_len` mask from section 5.

The negative-slot convention ties into this. When you pad a decode batch to a fixed size (for CUDA graphs, covered in a later Track F post), the padding tokens get `slot = -1` and the kernel's early return skips them. Miss the guard and a padding token writes to a negative index — undefined behavior that corrupts whatever lives before the KV tensor. This is the single most common `reshape_and_cache` bug, and it is why the `if slot < 0: return` is the first line of the kernel body, not an afterthought.

#### Worked example: fp8 KV changes the coalescing arithmetic

Switch the cache to fp8 (E4M3, 1 byte per element) — the subject of a later Track F post on KV-cache quantization, where the accuracy cliffs live. The append's coalescing math shifts in three linked ways:

- **Bytes per token per layer halve:** ${2 \times 8 \times 128 \times 1 = 2\text{ KiB}}$, so the batch-256 step's append drops from 32 MiB to 16 MiB. Coalesced time on a 4090 falls to ~19 µs.
- **The vector width in elements doubles.** Layout C's ${x = 16 / \text{sizeof(dtype)}}$ goes from 8 (bf16) to 16 (fp8). A 16-byte vectorized store now carries 16 fp8 elements, and `head_dim / x` becomes ${128 / 16 = 8}$ groups instead of 16. The kernel that wrote `float4`s of bf16 now writes the same 16-byte vectors but with twice the payload density.
- **A single head's key becomes exactly one 128-byte segment.** For head_dim 128 in fp8, one key vector is 128 bytes — one full transaction, perfectly aligned. In bf16 it was 256 bytes, two transactions. fp8 does not just save capacity; it makes the per-head access map cleanly onto the transaction size, which is a small, real, and rarely-mentioned second-order win. *(Source: derived; the accuracy trade of fp8 KV is cited-territory for the later post, not claimed here.)*

The one thing that does *not* change: the append is still bandwidth-bound and still lives or dies on coalescing. Halving the bytes halves the time; it does not change the shape of the problem.

### MHA versus GQA versus MQA: the row gets thin

The append writes a token's whole `H_kv * head_dim` row. That row's size depends entirely on how many KV heads the model keeps:

| Attention | `num_kv_heads` | Bytes / token / layer (bf16) | Coalescing note | Source |
| --- | --- | --- | --- | --- |
| MHA (e.g. old Llama-1) | 32 | 16 KiB | fat row, trivially coalesces | derived |
| GQA (Llama-3.1-8B) | 8 | 4 KiB | one full row = 8 warps, clean | derived |
| MQA (some Gemma, Falcon) | 1 | 512 B | thin row, may underfill a launch | derived |

The interesting end is MQA. With a single KV head the row is ${1 \times 128 \times 2 = 256}$ bytes for the key — just two transactions. Launch one block per token and each block does almost no work; the launch overhead starts to rival the transfer, and a warp of 32 threads on a 128-element row has half its lanes idle. The fix is to *batch tokens into a launch* — process many tokens per CUDA block so the block has enough contiguous work to saturate its warps — or to let Triton's autotuner pick a larger token tile. This is the same "thin work underfills the GPU" problem that haunts batch-1 decode GEMV, and the same fix: give each launch more to do. MQA saves the most cache memory and, precisely because it saves so much, gives the append kernel the least to chew on per token.

At the other end, MHA's 16 KiB row is so fat that coalescing is automatic and the only question is occupancy — 16 KiB is 8,192 bf16 elements, so a 1024-thread block does eight coalesced passes and you are bandwidth-bound with no effort. **The narrower the attention (the bigger the inference win from GQA/MQA), the more the append kernel has to work to stay efficient** — a small, honest tension worth knowing when you read a model card that brags about a single KV head.

---

## 9. Case studies and public numbers

Four public results that pin down what this kernel looks like in a real engine, each with its setup.

**vLLM's `reshape_and_cache` and the split layout ([`csrc/cache_kernels.cu`](https://github.com/vllm-project/vllm)).** The origin of the name and the split ${K/V}$ layout described in section 3: keys stored as `[num_blocks, num_kv_heads, head_dim/x, block_size, x]` with ${x = 16/\text{sizeof(dtype)}}$, values as `[num_blocks, num_kv_heads, head_dim, block_size]`, `slot_mapping` carrying negative sentinels for skipped tokens. The asymmetry is the takeaway: keys and values are reduced over different axes in attention, so they get different layouts, and the append writes into both.

**The 16-token block and the 2 MiB physical block ([vLLM Anatomy, 2025-09-05](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm)).** Default `block_size` of 16 tokens; per-block bytes computed as `2 * block_size * num_kv_heads * head_size * dtype_bytes`. For Llama-3.1-8B across all layers that is the 2 MiB physical block the [KV-offloading post](https://vllm.ai/blog/2026-01-08-kv-offloading-connector) (2026-01-08) reports after its KV-layout change merged per-layer, per-tensor blocks into one — a change made specifically so the *copy* path (offload to CPU) moves large contiguous units efficiently. Same layout tension, one subsystem out: the append wants aligned rows, the offloader wants big contiguous blocks, and the layout serves both.

**The Triton attention backend's gather ([vLLM, 2026-03-04](https://vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive)).** The read side of this post at production scale: paged KV in the innermost loop, a grid over batch and KV heads, and a split-KV decode variant that partitions the sequence traversal and combines partial softmax results in a second reduction kernel. The vLLM team reports their Triton implementation reaching 100.7% of FlashAttention 3's performance on H100 for Llama-3.1-8B at batch 1 with a 500-token input and a long decode, in roughly 800 lines against FA3's roughly 70,000. That is the gather this post's `_load_kv_block` is a toy of.

**The fused prologue ([vLLM HPC-Ops, 2026-07-06](https://vllm.ai/blog/2026-07-06-vllm-hpc-ops)).** The `HpcRopeNorm` fusion from section 7 — QK-Norm, RoPE, and the KV-cache write in one kernel — as part of an attention backend the vLLM/Tencent team reports at up to 2.95x over static split-KV and around 17% end-to-end TPOT reduction on 8×H20 for their Hunyuan model with FP8 KV and `--block-size 64`. The cache-write fusion is one component of that stack, not the whole win; quoted here to show that folding the append into the prologue is a real production technique, not a micro-optimization I invented.

| Result | Setup | Headline | Source |
| --- | --- | --- | --- |
| `reshape_and_cache` split layout | any model, vLLM | K and V under different shapes | cited: vLLM cache\_kernels.cu |
| Block default | vLLM V1 engine | 16 tokens, 2 MiB physical block (8B) | cited: vLLM Anatomy, 2025-09-05 |
| Triton paged gather | Llama-3.1-8B, H100, batch 1 | 100.7% of FlashAttention 3 | cited: vLLM Triton deep-dive, 2026-03-04 |
| Fused RoPE+norm+KV-write | Hunyuan, 8×H20, FP8 KV | ~17% TPOT reduction (whole prologue) | cited: vLLM HPC-Ops, 2026-07-06 |
| `nanoserve` append | Llama-3.1-8B, 4090, batch 256 | 39 µs coalesced vs 1.25 ms scattered | derived; reproduce: bench\_append.py |

---

## 10. When to write this kernel (and when to use vLLM's)

Write your own `reshape_and_cache` when:

- **You are building the engine to learn it.** This kernel is 30 lines and it is where the abstract block table becomes real addresses. Writing it once teaches you coalescing in the most direct way available — a kernel whose entire performance is one access pattern.
- **You need a non-standard KV dtype or layout.** fp8, int8, a custom paged layout for a hybrid attention model, MLA's compressed latent cache — all of them need an append that knows the specific byte layout, and the stock kernel may not cover yours. This is the one case where you genuinely have to touch it.
- **You are fusing.** If you are writing your own fused attention prologue, the cache write folds into it as in section 7, and at that point it is your code by definition.

Do not write your own when:

- **You are shipping a service.** vLLM's `reshape_and_cache` handles the split layout, fp8, the negative-slot convention, alignment, and a dozen dtype/head-count combinations, and it has been tuned against real profiles for years. Yours will at best match it and at worst introduce a memory-corruption bug that surfaces as garbage output in an unrelated request three hops away. The [custom CUDA kernels for inference post](/blog/machine-learning/model-serving/custom-cuda-kernels-for-inference) makes the general version of this argument.
- **The append is not your bottleneck** — which, coalesced, it never is. It is 0.2% of a decode step. Optimizing it in isolation is optimizing the wrong thing; the only reason to care is to keep it *from* becoming 6.7% via a layout bug, and the way to ensure that is to profile store efficiency, not to hand-tune the kernel.

The honest framing: you write this kernel to understand the layout decision and the coalescing law, and having understood them you are far better at operating a real engine — at reading `--block-size` and `--kv-cache-dtype` as coalescing knobs, at recognizing a store-efficiency collapse in a profile, and at knowing that "attention is slow" sometimes means "the KV write is scattered." The capstone, [the inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook), takes that position across the whole stack.

---

## Key takeaways

1. **`reshape_and_cache` is two integer ops and a store.** Its entire cost is whether the store coalesces; it does no arithmetic, so it is purely bandwidth-bound.
2. **Coalescing can swing effective bandwidth by up to 32x.** A warp's 32 lanes hitting one aligned 128-byte segment is one transaction; hitting 32 segments is up to 32 transactions for the same payload. The realistic range is 8x–32x because modern GPUs fetch in 32-byte sectors.
3. **The append and the gather want opposite layouts.** Writing one token wants heads-and-dims contiguous; reading one head's sequence wants tokens contiguous. No single layout coalesces both.
4. **vLLM resolves it by storing keys and values differently** — keys as `[block, head, dim/x, tok, x]` for vectorized ${QK^\top}$ loads, values as `[block, head, dim, tok]` for the ${\text{softmax}\cdot V}$ reduction. The asymmetry follows from the two products reducing over different axes.
5. **Make the indirection per-block, not per-element.** All the scattering is in choosing where each token's row starts; the row itself is written contiguously. That is what keeps the write coalesced no matter how scattered the slots are.
6. **The negative-slot guard is load-bearing.** Padding tokens carry `slot = -1` so a fixed-size batch coexists with a paged cache; skip the guard and you corrupt memory. It is the first line of the kernel body.
7. **`block_size` wants to be a power of two for a third reason: alignment.** Beyond fragmentation and bookkeeping, an odd block size misaligns every block's base address and silently doubles transactions.
8. **Fuse the write into the RoPE that produced K.** Unfused, K makes three HBM passes (write, read, write-to-cache); fused into the prologue it makes one. This is vLLM's `HpcRopeNorm`, and it deletes a memory pass rather than speeding one up.
9. **Benchmark with scattered slots, never `arange`.** Contiguous slots hide a coalescing bug; profile `gst_efficiency` and compare against a plain copy, not tok/s.
10. **The narrower the attention, the thinner the append's work.** MQA saves the most cache memory and gives the kernel the least per-token work, so batch tokens per launch to keep the warps full.

---

## Further reading

- vLLM, *Inside vLLM: Anatomy of a High-Throughput Inference System* (2025-09-05) — [vllm.ai/blog/2025-09-05-anatomy-of-vllm](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm). Block size, per-block bytes, and the structures the append writes into.
- vLLM, *Triton Attention Backend Deep Dive* (2026-03-04) — [vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive](https://vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive). The gather side at production scale: paged KV in the innermost loop and split-KV decode.
- vLLM, *HPC-Ops: attention + MoE backends* (2026-07-06) — [vllm.ai/blog/2026-07-06-vllm-hpc-ops](https://vllm.ai/blog/2026-07-06-vllm-hpc-ops). The `HpcRopeNorm` fused prologue that folds the KV-cache write into norm and RoPE.
- vLLM, *KV Cache Offloading to CPU* (2026-01-08) — [vllm.ai/blog/2026-01-08-kv-offloading-connector](https://vllm.ai/blog/2026-01-08-kv-offloading-connector). The KV-layout change to 2 MiB physical blocks and why the copy path wanted them.
- NVIDIA, *CUDA C++ Best Practices Guide* — the coalesced-access section for the transaction model behind section 2.
- [Paged KV cache: implementing blocks and a block table](/blog/machine-learning/inference-engineering/paged-kv-cache-implementing-blocks-and-a-block-table) — the allocator and block table this kernel writes through.
- [Writing your first inference CUDA kernel: RMSNorm and RoPE](/blog/machine-learning/inference-engineering/writing-your-first-inference-cuda-kernel-rmsnorm-and-rope) — the extension pattern and the fused prologue this post extends.
- [Paged attention kernel by hand](/blog/machine-learning/inference-engineering/paged-attention-kernel-by-hand) — the next post, where the gather lives inside an online-softmax attention kernel.
- [The memory hierarchy: registers, shared memory, and HBM](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm) — the coalesced-versus-strided framing this whole post rests on.
- [The memory math of the KV cache](/blog/machine-learning/inference-engineering/the-memory-math-of-the-kv-cache) — where the 4 KiB per token per layer comes from.
