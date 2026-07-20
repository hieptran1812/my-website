---
title: "Paged KV cache: implementing blocks and a block table"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "Turn the KV cache from a fragmentation nightmare into an allocator problem: derive exactly where a contiguous cache throws memory away, then build a block allocator, a per-request block table, and an attention path that gathers through it."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "kv-cache",
    "paged-attention",
    "memory-management",
    "vllm",
    "pytorch",
    "gpu",
    "ml-systems",
    "throughput",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 45
---

Your engine reports 6.7 GiB of KV cache and refuses to admit a fourteenth request. `nvidia-smi` agrees that the memory is allocated. The scheduler agrees that thirteen requests are running. And yet, if you sum the tokens those thirteen requests actually hold, you get 8,320 — out of a budget that could store 54,886. Six sevenths of the most expensive buffer on the card is sitting there allocated, untouched, and unavailable to anyone.

Nothing is broken. That is what a contiguous KV cache does. It reserves `max_seq` per request because it has to: the attention kernel wants one base pointer and a uniform stride, so the request's keys and values must live in one unbroken run of memory, and you cannot know at admission time how long the run needs to be. So you reserve for the worst case, every time, for every request, and then almost every request stops early and you eat the difference.

![Stacked breakdown of a 24 GiB card showing weights, activations, and a KV budget of which most is reserved but never written](/imgs/blogs/paged-kv-cache-implementing-blocks-and-a-block-table-1.webp)

This post is where that stops. We take the single idea that made vLLM famous — borrowed wholesale from operating-system virtual memory — and build it into `nanoserve` ourselves: fixed-size **blocks** of KV, a **free-block pool**, a per-request **block table** that maps logical token positions to physical block ids, and an attention path that gathers through that table instead of slicing a contiguous tensor. By the end you will have `nanoserve/blocks.py`, a paged write path, a readable paged-attention gather in PyTorch, and a script that counts — not times, *counts* — how many more requests fit. The count goes from 13 to 85 on the same 6.7 GiB, and every step of that arithmetic is on the page.

One standing promise from [the series introduction](/blog/machine-learning/inference-engineering/what-inference-engineering-is): **I have no GPU and I have run none of this.** Every number below is derived from arithmetic I show you, cited from a paper or an official vLLM post with a link, or framed as something you will reproduce yourself with a named script. The results tables carry a `Source` column. Capacity claims are the easy case — they are counting problems, so a script that counts gives you the same integers I get. Throughput claims are the hard case, and those stay cited.

---

## 1. What a contiguous cache actually costs

[The memory math of the KV cache](/blog/machine-learning/inference-engineering/the-memory-math-of-the-kv-cache) gave us the per-token bill. For a model with $L$ layers, $H_{kv}$ key/value heads, head dimension $d$, and $b$ bytes per element:

$$
\text{KV bytes per token} = 2 \cdot L \cdot H_{kv} \cdot d \cdot b
$$

The leading 2 is for keys and values. For Llama-3.1-8B in bf16 — 32 layers, 8 KV heads after grouped-query attention, head dim 128, 2 bytes — that is:

$$
2 \times 32 \times 8 \times 128 \times 2 = 131{,}072 \text{ bytes} = 128\text{ KiB per token}
$$

Now the budget. An RTX 4090 has 24 GiB. Llama-3.1-8B in bf16 is 8.03B parameters at 2 bytes, which is 16.06 GB, or 15.0 GiB. Add roughly 2.3 GiB for the CUDA context, activation workspace, and the logits tensor at a realistic batch size, and you are left with:

$$
24.0 - 15.0 - 2.3 = 6.7 \text{ GiB for KV}
$$

At 128 KiB per token that is $6.7 \times 1024 \times 1024 / 128 = 54{,}886$ tokens of KV capacity. Call this the **memory-math ceiling**: the number of live token positions the card can physically hold. It is the number your allocator should be trying to reach.

A contiguous allocator does not get close, and it fails in three distinct ways that are worth naming separately, because paging fixes them in different ways.

![Taxonomy tree splitting wasted KV bytes into reserved-not-written, scattered free space, and last-partial-block tail waste](/imgs/blogs/paged-kv-cache-implementing-blocks-and-a-block-table-2.webp)

**Reserved waste (internal fragmentation).** You allocate a slot sized for `max_seq` and the request uses a fraction of it. With `max_seq = 4096` each slot costs $4096 \times 128\text{ KiB} = 512\text{ MiB}$, so 6.7 GiB holds exactly 13 slots. If the average request is a 512-token prompt plus 128 generated tokens — 640 tokens, a reasonable chat shape — then each slot uses 640 of 4,096 positions. That is 15.6% occupancy. The other 84.4%, or 3,456 token positions per request, is memory you paid for, cannot lend to anyone, and will never write a byte into.

**Scattered waste (external fragmentation).** Requests finish out of order. Slot 2 frees, then slot 4 frees. You now have 1.0 GiB free and a request that wants two slots' worth — and it does not fit, because your free space is two separated 0.5 GiB holes, not one 1.0 GiB run. The memory exists. The geometry does not. This is the failure mode that makes people say "the engine OOMs at 57% utilization," which is a sentence I have heard more than once and which is exactly true and exactly this.

**Tail waste.** Whatever granularity you allocate at, the last unit is partly empty. In a contiguous cache with token granularity this is zero; in a paged cache it is at most one block minus one token. It is the only kind of waste paging *cannot* remove, and section 7 is where we price it.

Add the first two up and the accounting is brutal. Thirteen resident requests, 640 live tokens each, is 8,320 live token positions against a ceiling of 54,886. **Your allocator is delivering 15.2% of the concurrency your memory can support.** Not 90%, not 60% — fifteen. And the effect compounds: concurrency is what feeds the batch, and batch size is what pulls decode off the memory-bandwidth floor. A serving loop stuck at 13 concurrent requests on hardware that could hold 85 is not slightly slower, it is running the GPU in a completely different regime.

The vLLM paper (Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023, [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)) measured the same pathology on production engines of the day and reports that existing systems waste roughly 60–80% of KV cache memory to fragmentation and over-reservation, while their block-based manager keeps waste under about 4% — and that this alone is worth 2–4× throughput at comparable latency. The 15.2% utilization we just derived sits inside their band. That is a reassuring sign that the arithmetic above is describing the real thing and not a strawman.

#### Worked example: the fourteenth request

You are serving Llama-3.1-8B on a 4090 with a contiguous cache and `max_seq = 4096`. Thirteen requests are resident, each with a 512-token prompt, each about 100 tokens into generation. A fourteenth arrives with a 200-token prompt.

- Bytes it needs, honestly: $200 \times 128\text{ KiB} = 25\text{ MiB}$.
- Bytes the allocator demands: $4096 \times 128\text{ KiB} = 512\text{ MiB}$.
- Bytes free: $6.7\text{ GiB} - 13 \times 512\text{ MiB} = 6.7 - 6.5 = 0.2\text{ GiB} = 205\text{ MiB}$.

The request is rejected or queued. It needed 25 MiB. You have 205 MiB. The allocator asked for 512 MiB because it does not know the future, and the geometry says no. *(Source: derived.)*

That is the whole problem, and it is not a performance problem. It is an allocator problem wearing a performance problem's clothes.

---

## 2. Logical positions, physical blocks, and a table in between

The fix is the one operating systems landed on in the 1960s and it is worth stating precisely, because the loose version of the analogy leads people astray.

A process believes it owns a contiguous address space. The hardware does not give it one. Instead, memory is carved into fixed-size **pages**, the process's contiguous *virtual* addresses are chopped into page-sized chunks, and a **page table** records which physical frame each virtual page currently lives in. Contiguity becomes a property of the *index*, not of the *storage*. Physical frames can be handed out in any order, from anywhere, and reclaimed independently.

Transplant that to KV. A request believes its keys and values occupy positions 0 through $S-1$ in order — because the attention math genuinely needs them in order, with position $t$ attending to everything at position $\le t$. The cache does not have to store them in order. Carve the KV memory into fixed-size **blocks** holding `block_size` tokens each. Give every request a **block table**: a list where entry $i$ holds the physical block id storing logical positions $[i \cdot \text{block\_size}, (i+1) \cdot \text{block\_size})$. Then the mapping from a logical position to a physical storage slot is two lines of integer arithmetic:

$$
\text{block\_index} = \left\lfloor \frac{\text{pos}}{\text{block\_size}} \right\rfloor, \qquad
\text{offset} = \text{pos} \bmod \text{block\_size}
$$

$$
\text{physical\_slot} = \text{block\_table}[\text{block\_index}] \cdot \text{block\_size} + \text{offset}
$$

That is it. That is PagedAttention's core, and everything else in this post is consequences of those three expressions.

![Grid showing four logical blocks of a request mapping down to scattered physical block ids seven, two, nineteen and five](/imgs/blogs/paged-kv-cache-implementing-blocks-and-a-block-table-3.webp)

Here is where the analogy stops being helpful, and it matters. In an OS, address translation happens in hardware, on every memory access, transparently, via the MMU and the TLB. Nothing in your program changes. In an inference engine there is no MMU for tensor indices: **the translation is your code, and it happens inside the attention kernel.** Every consumer of the KV cache — the write path, the attention read, the eviction logic, the prefix-cache lookup, the swap-to-host path — has to be taught the indirection explicitly. Paging is not free the way virtual memory is nearly free; it is free in *memory* and it costs you in *kernel complexity and indirection*. That trade is overwhelmingly worth it, as the numbers below show, but pretending it is a pure win is how people end up surprised by their gather kernel.

The second thing the analogy hides: OS pages are 4 KiB and chosen to match hardware TLB and disk-transfer granularity. KV block sizes are chosen to balance tail waste against block-table length and gather efficiency, and the answer lands somewhere entirely different — 16 tokens, which for Llama-3.1-8B works out to a 2 MiB physical block. Section 3 derives that.

Watch the two layouts hold the same live tokens:

<figure class="blog-anim">
<svg viewBox="0 0 720 330" role="img" aria-label="A contiguous KV cache holds mostly reserved-but-unwritten space and leaves two separated holes that reject an incoming request, while the paged layout stores the same live tokens in fixed-size blocks and admits it" style="width:100%;height:auto;max-width:840px">
<style>
.p8-live{fill:var(--accent,#6366f1)}
.p8-resv{fill:var(--border,#d1d5db);opacity:.5}
.p8-free{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.p8-hole{fill:none;stroke:var(--border,#d1d5db);stroke-width:1.5;stroke-dasharray:6 5}
.p8-bad{fill:none;stroke:#dc2626;stroke-width:2.5;stroke-dasharray:7 5}
.p8-badt{font:600 13px ui-sans-serif,system-ui;fill:#dc2626}
.p8-okt{font:600 13px ui-sans-serif,system-ui;fill:var(--accent,#6366f1)}
.p8-h{font:600 16px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.p8-s{font:400 12.5px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.p8-m{font:600 11px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
@keyframes p8-fadeA{0%,40%{opacity:1}50%,92%{opacity:0}100%{opacity:1}}
@keyframes p8-fadeB{0%,40%{opacity:0}50%,92%{opacity:1}100%{opacity:0}}
@keyframes p8-land{0%,55%{opacity:0}70%,100%{opacity:1}}
.p8-A{animation:p8-fadeA 12s ease-in-out infinite}
.p8-B{animation:p8-fadeB 12s ease-in-out infinite}
.p8-N{animation:p8-land 12s ease-in-out infinite}
.p8-N2{animation-delay:.35s}
.p8-N3{animation-delay:.7s}
.p8-N4{animation-delay:1.05s}
.p8-N5{animation-delay:1.4s}
@media (prefers-reduced-motion:reduce){.p8-A{animation:none;opacity:1}.p8-B{animation:none;opacity:0}.p8-N{animation:none;opacity:1}}
</style>
<g class="p8-A">
<text class="p8-h" x="30" y="34">Contiguous: one 4,096-token slot per request</text>
<rect class="p8-live" x="30"  y="56" width="25"  height="46" rx="4"/>
<rect class="p8-resv" x="55"  y="56" width="135" height="46" rx="4"/>
<rect class="p8-hole" x="200" y="56" width="160" height="46" rx="4"/>
<rect class="p8-live" x="370" y="56" width="25"  height="46" rx="4"/>
<rect class="p8-resv" x="395" y="56" width="135" height="46" rx="4"/>
<rect class="p8-hole" x="540" y="56" width="160" height="46" rx="4"/>
<text class="p8-m" x="110" y="122">req A · 640 of 4,096 used</text>
<text class="p8-m" x="280" y="122">free slot</text>
<text class="p8-m" x="450" y="122">req B · 640 of 4,096 used</text>
<text class="p8-m" x="620" y="122">free slot</text>
<rect class="p8-bad" x="200" y="160" width="330" height="46" rx="4"/>
<text class="p8-badt" x="214" y="188">incoming request needs 2 adjacent slots</text>
<text class="p8-badt" x="214" y="232">1.0 GiB free, largest run 0.5 GiB, rejected</text>
<text class="p8-s" x="30" y="270">Two holes exist and neither is big enough; 84% of every reserved slot was never written.</text>
<text class="p8-s" x="30" y="292">Resident requests: 13. Live tokens: 8,320 of a possible 54,886.</text>
</g>
<g class="p8-B">
<text class="p8-h" x="30" y="34">Paged: 16-token blocks from one free pool</text>
<rect class="p8-live" x="30"  y="56" width="34" height="30" rx="4"/>
<rect class="p8-free" x="72"  y="56" width="34" height="30" rx="4"/>
<rect class="p8-live" x="114" y="56" width="34" height="30" rx="4"/>
<rect class="p8-live" x="156" y="56" width="34" height="30" rx="4"/>
<rect class="p8-free" x="198" y="56" width="34" height="30" rx="4"/>
<rect class="p8-live" x="240" y="56" width="34" height="30" rx="4"/>
<rect class="p8-free" x="282" y="56" width="34" height="30" rx="4"/>
<rect class="p8-live" x="324" y="56" width="34" height="30" rx="4"/>
<rect class="p8-live" x="366" y="56" width="34" height="30" rx="4"/>
<rect class="p8-free" x="408" y="56" width="34" height="30" rx="4"/>
<rect class="p8-live" x="450" y="56" width="34" height="30" rx="4"/>
<rect class="p8-free" x="492" y="56" width="34" height="30" rx="4"/>
<rect class="p8-live" x="534" y="56" width="34" height="30" rx="4"/>
<rect class="p8-live" x="576" y="56" width="34" height="30" rx="4"/>
<rect class="p8-free" x="618" y="56" width="34" height="30" rx="4"/>
<rect class="p8-free" x="660" y="56" width="34" height="30" rx="4"/>
<text class="p8-m" x="362" y="106">physical blocks 0 to 15 · 2 MiB each · order means nothing</text>
<rect class="p8-live p8-N"    x="72"  y="150" width="34" height="30" rx="4"/>
<rect class="p8-live p8-N p8-N2" x="198" y="150" width="34" height="30" rx="4"/>
<rect class="p8-live p8-N p8-N3" x="282" y="150" width="34" height="30" rx="4"/>
<rect class="p8-live p8-N p8-N4" x="408" y="150" width="34" height="30" rx="4"/>
<rect class="p8-live p8-N p8-N5" x="492" y="150" width="34" height="30" rx="4"/>
<text class="p8-m" x="362" y="200">incoming request takes any 5 free blocks</text>
<text class="p8-okt" x="30" y="232">block_table = [1, 4, 6, 9, 11] — logical order restored by the table</text>
<text class="p8-s" x="30" y="270">Nothing is reserved and nothing has to be adjacent; only the last block is partly empty.</text>
<text class="p8-s" x="30" y="292">Resident requests: 85. Live tokens: 54,400 of a possible 54,886.</text>
</g>
</svg>
<figcaption>The same live tokens under both allocators: contiguous slots reserve for the worst case and reject a request that would fit, while fixed-size blocks let the block table restore logical order from scattered physical storage.</figcaption>
</figure>

Notice what disappeared between the two states. Reserved waste is gone because nothing is allocated until a token needs it. Scattered waste is gone because every free unit is interchangeable — there is no such thing as a hole that is the wrong shape when every hole is exactly one block. What remains is the tail, and the tail is bounded by one block.

---

## 3. Sizing a block

Two questions decide the block size: how many bytes is one block, and how many tokens should it hold.

The bytes follow from the per-token formula, scaled by the block size. Per the vLLM team's *Inside vLLM: Anatomy of a High-Throughput Inference System* post ([2025-09-05](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm)), vLLM's per-block KV bytes are computed as `2 (key/value) * block_size * num_kv_heads * head_size * dtype_bytes`, with a default `block_size` of 16 tokens. Written our way, and taken across all layers so that one block id addresses a request's whole KV footprint for those 16 positions:

$$
\text{bytes per block} = \text{block\_size} \cdot 2 \cdot L \cdot H_{kv} \cdot d \cdot b
$$

For Llama-3.1-8B at `block_size = 16`:

$$
16 \times 131{,}072 = 2{,}097{,}152 \text{ bytes} = 2 \text{ MiB}
$$

That number is checkable against a published one. vLLM's *KV Cache Offloading to CPU* post ([2026-01-08](https://vllm.ai/blog/2026-01-08-kv-offloading-connector)) lists the physical block sizes after a KV-layout change that merged the per-layer, per-tensor blocks into one: Llama-3.1-8B goes from 32 KB to **2 MB**, Llama-3.2-1B from 16 KB to **0.5 MB**, and Llama-3.1-70B from 8 KB to **1.25 MB**. Our formula reproduces all three, which is the best sanity check a derivation can get:

| Model | $L$ | $H_{kv}$ | $d$ | KV bytes/token (bf16) | Block at 16 tokens | vLLM's published figure | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Llama-3.1-8B | 32 | 8 | 128 | 131,072 (128 KiB) | 2.00 MiB | 2 MB | derived + cited: vLLM offloading post |
| Llama-3.2-1B | 16 | 8 | 64 | 32,768 (32 KiB) | 0.50 MiB | 0.5 MB | derived + cited: vLLM offloading post |
| Llama-3.1-70B | 80 | 8 | 128 | 327,680 (320 KiB) | 5.00 MiB whole model | 1.25 MB | derived; cited figure is per-rank |

The 70B row is the interesting one. Our whole-model block is 5 MiB and the published figure is 1.25 MB — exactly a quarter. That is not a contradiction, it is tensor parallelism: at $\text{TP} = 4$ each rank holds 2 of the 8 KV heads, so each rank's block is ${5 / 4 = 1.25}$ MiB. Reproducing a published number *and* explaining the factor you are off by is how you know you have the model right. The `32 KB → 2 MB` framing tells the same story from the other side: the pre-change block was one layer's keys only ($16 \times 8 \times 128 \times 2 = 32$ KiB), and merging across 32 layers and both K and V is a factor of 64, taking 32 KiB to 2 MiB.

Now, how many tokens per block? Three forces pull:

**Smaller blocks reduce tail waste.** If a request holds $S$ tokens, it occupies $\lceil S / \text{block\_size} \rceil$ blocks and wastes $(-S) \bmod \text{block\_size}$ token slots in the last one. For sequence lengths spread uniformly over block boundaries the expectation is $(\text{block\_size} - 1)/2$ tokens. At 16 that is 7.5 tokens, or 960 KiB, per request. At 256 it is 127.5 tokens, or about 16 MiB.

**Larger blocks reduce bookkeeping.** A 640-token request needs 40 table entries at `block_size = 16` and 3 at 256. Forty `int32` entries is 160 bytes of block table against 80 MiB of actual KV — a rounding error, but that ratio is per request, and the table has to be materialized on the GPU every step so the kernel can read it. vLLM's *Model Runner V2* post ([2026-03-24](https://vllm.ai/blog/2026-03-24-mrv2)) describes building the ordered block tables with a per-step gather and constructing input tensors on-GPU via Triton kernels precisely because this per-step assembly is not free at scale.

**Larger blocks help the kernel and hurt the cache.** A gather of 16 contiguous tokens' worth of KV is one reasonably-sized contiguous read; a gather of one token's worth is a scattered 2 KiB read per layer that no memory controller will love. But large blocks also coarsen prefix-cache granularity — the Anatomy post notes that only *complete* blocks are cacheable, so a shared prefix of length `long_prefix_len` recomputes `long_prefix_len % block_size` tokens no matter how good your hit rate is. That is a tax that grows linearly with block size, and it is why [prefix sharing](/blog/machine-learning/inference-engineering/prefix-sharing-radix-trees-and-copy-on-write) cares about this knob as much as the allocator does.

Sixteen sits near the knee of all three curves, which is why it is the default. We will use it, and section 7 stress-tests the choice at both extremes.

The pool size follows immediately. With 6.7 GiB of KV budget and 2 MiB blocks:

$$
\text{num\_blocks} = \left\lfloor \frac{6.7 \times 1024}{2} \right\rfloor = 3{,}430 \text{ blocks}
$$

3,430 blocks × 16 tokens = 54,880 token positions, which is the memory-math ceiling to within one block. The pool *is* the ceiling now. That is the whole point.

---

## 4. `nanoserve/blocks.py`: the allocator

Here is the first real file. It has no PyTorch in it at all, which is deliberate: the allocator is an integer bookkeeping problem, it should be testable on a laptop, and keeping tensors out of it means you can fuzz it in a loop without a GPU.

```python
# nanoserve/blocks.py
"""Fixed-size KV block allocation: a free pool plus per-request block tables."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


class OutOfBlocks(RuntimeError):
    """Raised when the free pool cannot satisfy an allocation."""

    def __init__(self, wanted: int, available: int):
        super().__init__(f"wanted {wanted} blocks, {available} free")
        self.wanted = wanted
        self.available = available


class BlockAllocator:
    """A pool of interchangeable physical KV blocks.

    Physical blocks are identified by an integer id in [0, num_blocks). The id
    is the only thing a request ever holds; where that block sits in the KV
    tensor is a multiplication, not a pointer.
    """

    def __init__(self, num_blocks: int, block_size: int = 16):
        self.num_blocks = num_blocks
        self.block_size = block_size
        # FIFO, not LIFO. See the note below on why this ordering matters.
        self._free: deque[int] = deque(range(num_blocks))
        # Reference counts exist for post 9's copy-on-write sharing. For now
        # every live block has exactly one owner.
        self._ref: list[int] = [0] * num_blocks

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_used(self) -> int:
        return self.num_blocks - len(self._free)

    def can_allocate(self, n: int) -> bool:
        return n <= len(self._free)

    def allocate(self, n: int) -> list[int]:
        if n > len(self._free):
            raise OutOfBlocks(n, len(self._free))
        out = [self._free.popleft() for _ in range(n)]
        for b in out:
            self._ref[b] += 1
        return out

    def free(self, block_ids: list[int]) -> None:
        for b in block_ids:
            if self._ref[b] <= 0:
                raise RuntimeError(f"double free of block {b}")
            self._ref[b] -= 1
            if self._ref[b] == 0:
                self._free.append(b)

    def ref_count(self, block_id: int) -> int:
        return self._ref[block_id]
```

Three design notes that are not obvious from the code.

**The pool is a queue, not a stack.** `popleft` / `append` gives FIFO reuse: a block that was freed long ago is handed out before one freed a moment ago. The Anatomy post describes vLLM's pool as a `free_block_queue` holding "hundreds of thousands" of blocks depending on VRAM, and the queue ordering is not incidental. Once prefix caching lands, a freed block still holds valid cached content, and FIFO reuse means that content survives as long as possible before being overwritten — the pool doubles as an LRU cache for free-but-still-useful blocks. A LIFO stack would clobber the most recently freed block first, which is exactly the one most likely to be hit again. Free ordering is a caching policy in disguise.

**Reference counts are here on day one even though nothing shares yet.** Adding refcounting later means auditing every free path in the engine for the case where a block has two owners. Adding it now costs one array and one comparison, and post 9 lights it up for copy-on-write. Allocators are the wrong place to be clever later.

**Allocation cannot fail silently.** `OutOfBlocks` is a real exception with the numbers in it, because the scheduler is going to catch it and make a policy decision — preempt someone, queue the request, or reject it. That decision belongs to post 10, but the *signal* has to exist here and it has to be precise. An allocator that returns `None` on failure produces a `NoneType has no attribute` traceback three call frames away from the actual problem.

Now the per-request side.

```python
# nanoserve/blocks.py (continued)

@dataclass
class PagedSequence:
    """One request's view of the cache: a block table and a token count."""

    req_id: str
    allocator: BlockAllocator
    block_table: list[int] = field(default_factory=list)
    num_tokens: int = 0

    @property
    def block_size(self) -> int:
        return self.allocator.block_size

    def blocks_needed_for(self, n_new: int) -> int:
        """How many NEW blocks appending n_new tokens would require."""
        bs = self.block_size
        have = len(self.block_table) * bs
        return max(0, -(-(self.num_tokens + n_new) // bs) - len(self.block_table)) \
            if self.num_tokens + n_new > have else 0

    def append(self, n_new: int = 1) -> list[int]:
        """Reserve room for n_new tokens; return their flat physical slots.

        A flat slot is `physical_block_id * block_size + offset`, i.e. an index
        into the KV store viewed as one long array of token slots. This is the
        only number the write kernel needs.
        """
        need = self.blocks_needed_for(n_new)
        if need:
            self.block_table.extend(self.allocator.allocate(need))
        bs = self.block_size
        slots = []
        for i in range(n_new):
            pos = self.num_tokens + i
            phys = self.block_table[pos // bs]
            slots.append(phys * bs + pos % bs)
        self.num_tokens += n_new
        return slots

    def release(self) -> None:
        self.allocator.free(self.block_table)
        self.block_table = []
        self.num_tokens = 0
```

The engine keeps a `dict[str, PagedSequence]`, which is the same structure the Anatomy post calls `req_to_blocks`: a map from request id to its list of blocks. That dictionary plus the free queue is the entire memory manager. There is no compaction, no coalescing, no best-fit search, no buddy allocator. Every free unit is the same size, so allocation is `popleft` and freeing is `append`, both $O(1)$.

![Branching diagram of the append_token decision writing in place, popping a free block, or failing to the preemption path](/imgs/blogs/paged-kv-cache-implementing-blocks-and-a-block-table-4.webp)

The branch structure above is worth internalizing because it is where the engine's behaviour changes character. Fifteen out of every sixteen decode steps take the left branch and touch no shared state at all. One in sixteen pops from the pool. And when the pool is empty, `append` raises, and that raise is the moment your engine has to have an opinion about fairness, priorities and preemption. A paged engine does not degrade smoothly into swap the way an OS does — it hits a wall and calls a policy. That policy is post 10.

Two more pieces: computing the pool size from a real memory budget, and allocating the physical store.

```python
# nanoserve/blocks.py (continued)
import torch


def kv_bytes_per_token(cfg, dtype_bytes: int = 2) -> int:
    """2 (K and V) * layers * kv_heads * head_dim * bytes."""
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    return 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * head_dim * dtype_bytes


def plan_pool(cfg, budget_bytes: int, block_size: int = 16, dtype_bytes: int = 2):
    """Turn a memory budget into a block count. Prints the arithmetic."""
    per_tok = kv_bytes_per_token(cfg, dtype_bytes)
    per_block = per_tok * block_size
    num_blocks = budget_bytes // per_block
    print(f"KV bytes/token : {per_tok:,} ({per_tok / 1024:.0f} KiB)")
    print(f"bytes/block    : {per_block:,} ({per_block / 2**20:.2f} MiB)")
    print(f"budget         : {budget_bytes / 2**30:.2f} GiB")
    print(f"blocks         : {num_blocks:,}")
    print(f"token capacity : {num_blocks * block_size:,}")
    return num_blocks


class PagedKVStore:
    """The physical KV memory: one tensor, addressed only by flat slot index."""

    def __init__(self, cfg, num_blocks: int, block_size: int = 16,
                 dtype=torch.bfloat16, device="cuda"):
        self.block_size = block_size
        self.num_blocks = num_blocks
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.head_dim = head_dim
        self.n_kv = cfg.num_key_value_heads
        # [layer, K/V, block, slot-in-block, kv_head, head_dim]
        self.data = torch.empty(
            cfg.num_hidden_layers, 2, num_blocks, block_size, self.n_kv, head_dim,
            dtype=dtype, device=device,
        )
        # A flat view: [layer, K/V, num_blocks * block_size, kv_head, head_dim].
        # Every write and read below indexes THIS, using flat slot ids.
        self.flat = self.data.view(
            cfg.num_hidden_layers, 2, num_blocks * block_size, self.n_kv, head_dim
        )

    def nbytes(self) -> int:
        return self.data.numel() * self.data.element_size()
```

Running `plan_pool` against a real Llama-3.1-8B config with a 6.7 GiB budget prints:

```console
KV bytes/token : 131,072 (128 KiB)
bytes/block    : 2,097,152 (2.00 MiB)
budget         : 6.70 GiB
blocks         : 3,430
token capacity : 54,880
```

Those five lines are the derivation from section 1 and section 3, executed. If your engine cannot print this table on startup, you do not know your own capacity, and every OOM you hit later will feel like a mystery instead of a subtraction.

The single-tensor layout deserves a word. Allocating one 6.7 GiB tensor up front, once, and never calling the caching allocator again for KV is not an optimization detail — it *is* the design. PyTorch's caching allocator would otherwise fragment the same way our contiguous cache did, just one level down, and `torch.cuda.empty_cache()` would become a load-bearing part of your serving loop. Take the memory once, address it with integers forever.

---

## 5. Reading and writing through the table

The write path is the easier half. After the forward pass computes new keys and values for the tokens in this step, they have to land in the right physical slots. This is the function vLLM calls `reshape_and_cache` and implements as a CUDA kernel; ours is one `index_copy_` per tensor.

```python
# nanoserve/blocks.py (continued)

def write_kv(store: PagedKVStore, layer: int,
             k: torch.Tensor, v: torch.Tensor, slots: torch.Tensor) -> None:
    """Scatter this step's K/V into their physical slots.

    k, v   : [T, n_kv_heads, head_dim] for the T new tokens in this step
    slots  : [T] int64, flat slot ids from PagedSequence.append()
    """
    store.flat[layer, 0].index_copy_(0, slots, k)
    store.flat[layer, 1].index_copy_(0, slots, v)
```

That is the entire difference between a contiguous write and a paged write: the destination is a gathered index rather than a slice. Note that `slots` is built once per step for the whole batch — every request's new tokens concatenated — and reused across all 32 layers. Building it 32 times would be 32 host-to-device transfers per decode step, which on a loop that is already host-bound is a genuine regression. Build it once, keep it on the GPU.

The read path is where the indirection actually costs something.

```python
# nanoserve/blocks.py (continued)

def gather_kv(store: PagedKVStore, layer: int,
              block_table: torch.Tensor, seq_len: int):
    """Materialize a request's K/V in logical order.

    block_table : [num_blocks_used] int64 physical block ids, logical order
    returns k, v of shape [seq_len, n_kv_heads, head_dim]
    """
    bs = store.block_size
    # block id -> the bs consecutive flat slots it owns
    base = block_table.unsqueeze(1) * bs                       # [B, 1]
    offs = torch.arange(bs, device=block_table.device)         # [bs]
    idx = (base + offs).reshape(-1)[:seq_len]                  # [seq_len]
    k = store.flat[layer, 0].index_select(0, idx)
    v = store.flat[layer, 1].index_select(0, idx)
    return k, v
```

And the attention itself, for one request at decode time:

```python
# nanoserve/attention_paged.py
import torch
import torch.nn.functional as F

from nanoserve.blocks import gather_kv


@torch.inference_mode()
def paged_decode_attention(store, layer, q, block_table, seq_len, n_heads):
    """One query token attending over a paged KV history.

    q : [n_heads, head_dim] for the single new token
    """
    k, v = gather_kv(store, layer, block_table, seq_len)   # [S, n_kv, D]
    n_kv = k.shape[1]
    rep = n_heads // n_kv                                  # GQA group size
    k = k.repeat_interleave(rep, dim=1)                    # [S, n_heads, D]
    v = v.repeat_interleave(rep, dim=1)
    # SDPA wants [B, heads, seq, dim]
    q_ = q.unsqueeze(0).unsqueeze(2)                       # [1, n_heads, 1, D]
    k_ = k.permute(1, 0, 2).unsqueeze(0)                   # [1, n_heads, S, D]
    v_ = v.permute(1, 0, 2).unsqueeze(0)
    # Decode attends to everything already in the cache: no mask needed.
    out = F.scaled_dot_product_attention(q_, k_, v_, is_causal=False)
    return out.squeeze(0).squeeze(1)                       # [n_heads, D]
```

![Two-column comparison of contiguous attention reading one strided slice against paged attention gathering forty blocks through a lookup table](/imgs/blogs/paged-kv-cache-implementing-blocks-and-a-block-table-5.webp)

Be honest about what this code is and is not. It is **correct** and it is **readable** and it is **not the kernel you ship**. Three specific costs are hiding in it:

**It materializes.** `index_select` allocates a fresh `[S, n_kv, D]` tensor for K and another for V, every layer, every step. For a 640-token request in bf16 that is $640 \times 8 \times 128 \times 2 = 1.25$ MiB per tensor per layer — 80 MiB of allocation and copying per decode step across 32 layers, to feed an attention that reads each byte exactly once. A real paged-attention kernel never materializes; it walks the block table inside the kernel and streams KV straight from the blocks into registers and shared memory as it computes the online softmax.

**It loses coalescing at block boundaries.** Within a block, the 16 tokens are contiguous, so a warp reading them gets well-coalesced access. Across blocks, the address jumps arbitrarily. With 16-token blocks and bf16 GQA, one block's keys for one layer are $16 \times 8 \times 128 \times 2 = 32$ KiB — comfortably large enough that the jump between blocks amortizes, which is another argument for 16 over 1. This is exactly the tension [the memory hierarchy post](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm) frames as coalesced versus strided access, and it is why the KV layout inside a block (`[block, slot, head, dim]` versus interleaving heads outermost) is a real decision with real bandwidth consequences rather than a matter of taste.

**It does one request at a time.** A batched version needs a padded `[B, max_blocks]` block table and a mask, or a variable-length layout with `query_start_loc`-style offsets. Both are straightforward and both add host work per step.

Writing that kernel properly — online softmax, split-K over the sequence, partial reductions, and getting close to peak HBM bandwidth — is post 25 of this series. What matters here is that the *interface* is now right: attention consumes a block table and a length, not a base pointer and a stride. Once that boundary exists, replacing the Python gather with a CUDA kernel is a drop-in change that touches one function. If you want to see what production versions of this look like before we get there, the [continuous batching and PagedAttention post](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention) covers how vLLM's kernel and scheduler fit together.

For a batch, the block tables need to be one tensor. Here is the assembly, which runs once per step:

```python
# nanoserve/blocks.py (continued)

def build_batch_tables(seqs: list[PagedSequence], device="cuda"):
    """Pad per-request block tables into one [B, max_blocks] int64 tensor."""
    max_blocks = max(len(s.block_table) for s in seqs)
    table = torch.zeros(len(seqs), max_blocks, dtype=torch.int64)
    for i, s in enumerate(seqs):
        table[i, : len(s.block_table)] = torch.tensor(s.block_table, dtype=torch.int64)
    lens = torch.tensor([s.num_tokens for s in seqs], dtype=torch.int64)
    return table.to(device, non_blocking=True), lens.to(device, non_blocking=True)
```

Padding here is cheap in a way that padding the KV itself never was: an unused table entry costs 8 bytes, not 128 KiB per token. The padding tax moved from the cache to a table 10,000 times smaller. That is the same trick under a different name, and it is worth noticing because [static batching](/blog/machine-learning/inference-engineering/why-recompute-is-fatal-writing-a-kv-cache) suffers from precisely the version of this problem that we have now made irrelevant.

---

## 6. What it buys, counted rather than claimed

Now the payoff, and it is a counting exercise, which is the most honest kind of number available to someone without a GPU.

Under paging, a request holding $S$ tokens consumes $\lceil S / 16 \rceil$ blocks and nothing more. For our 640-token average request that is exactly 40 blocks. The pool holds 3,430. So:

$$
\left\lfloor \frac{3430}{40} \right\rfloor = 85 \text{ concurrent requests}
$$

against 13 for the contiguous allocator. Live tokens go from 8,320 to $85 \times 640 = 54{,}400$, which is 99.1% of the 54,880-token ceiling.

| Metric | Contiguous, `max_seq` 4096 | Paged, `block_size` 16 | Source |
| --- | --- | --- | --- |
| KV budget | 6.7 GiB | 6.7 GiB | derived |
| Allocation unit | 4,096 tokens (512 MiB) | 16 tokens (2 MiB) | derived |
| Units available | 13 slots | 3,430 blocks | derived |
| Units per 640-token request | 1 slot | 40 blocks | derived |
| Concurrent requests | 13 | 85 | derived |
| Live tokens resident | 8,320 | 54,400 | derived |
| Ceiling utilization | 15.2% | 99.1% | derived |
| Waste per request | 3,456 tokens (432 MiB) | 7.5 tokens avg (0.94 MiB) | derived |
| Rejects a request when free bytes suffice | yes | no | derived |
| Throughput effect | baseline | 2–4× reported | cited: vLLM paper, arXiv:2309.06180 |

Read the last two rows together and carefully. The **6.5× concurrency** is derived arithmetic and you can check every digit. The **2–4× throughput** is *not* mine and is not the same claim: it is what Kwon et al. measured against the engines of 2023 on their workloads, and it is smaller than the concurrency ratio because throughput saturates — beyond some batch size decode stops being starved for parallelism and starts being limited by bandwidth and by the attention work itself, which grows with total resident tokens. Anyone quoting a concurrency ratio as a throughput ratio is selling something.

Here is the script that produces the counting half. It is deliberately pure Python with no GPU and no timing, so your run and my arithmetic agree exactly.

```python
# nanoserve/fragsim.py
"""Count how many requests fit under a contiguous vs a paged allocator.

Deterministic: no timing, no GPU, no randomness unless you pass a seed.
Run:  python -m nanoserve.fragsim
"""
import argparse
import random
from math import ceil

from nanoserve.blocks import BlockAllocator, PagedSequence


def contiguous_capacity(budget_tokens: int, max_seq: int, lengths: list[int]) -> dict:
    slots = budget_tokens // max_seq
    admitted = lengths[:slots]
    return {
        "admitted": len(admitted),
        "live_tokens": sum(admitted),
        "reserved_tokens": len(admitted) * max_seq,
        "utilization": sum(admitted) / budget_tokens,
    }


def paged_capacity(budget_tokens: int, block_size: int, lengths: list[int]) -> dict:
    alloc = BlockAllocator(budget_tokens // block_size, block_size)
    admitted, live = 0, 0
    for i, n in enumerate(lengths):
        seq = PagedSequence(req_id=str(i), allocator=alloc)
        if not alloc.can_allocate(ceil(n / block_size)):
            break
        seq.append(n)
        admitted, live = admitted + 1, live + n
    return {
        "admitted": admitted,
        "live_tokens": live,
        "reserved_tokens": alloc.num_used * block_size,
        "utilization": live / budget_tokens,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget-gib", type=float, default=6.7)
    p.add_argument("--kv-kib-per-token", type=float, default=128.0)
    p.add_argument("--max-seq", type=int, default=4096)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--mean-len", type=int, default=640)
    p.add_argument("--spread", type=int, default=0, help="uniform +/- around mean")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    budget_tokens = int(a.budget_gib * 1024 * 1024 / a.kv_kib_per_token)
    rng = random.Random(a.seed)
    lengths = [
        max(1, a.mean_len + (rng.randint(-a.spread, a.spread) if a.spread else 0))
        for _ in range(10_000)
    ]

    print(f"budget: {budget_tokens:,} token slots")
    for name, res in (
        ("contiguous", contiguous_capacity(budget_tokens, a.max_seq, lengths)),
        ("paged", paged_capacity(budget_tokens, a.block_size, lengths)),
    ):
        print(
            f"{name:>11}: {res['admitted']:>5} requests | "
            f"{res['live_tokens']:>7,} live tokens | "
            f"{res['utilization'] * 100:5.1f}% of ceiling"
        )


if __name__ == "__main__":
    main()
```

With the defaults it prints:

```console
budget: 54,886 token slots
 contiguous:    13 requests |   8,320 live tokens |  15.2% of ceiling
      paged:    85 requests |  54,400 live tokens |  99.1% of ceiling
```

Those are the numbers in the table. They are counting, not measuring, so *this is one of the very few places in inference engineering where "run it and you will get exactly this" is a true statement.* Anything involving a clock is not like this, which is why the rest of the series is so careful about it.

![Timeline of one request's block usage from admission through decode growth to the moment EOS frees all forty blocks](/imgs/blogs/paged-kv-cache-implementing-blocks-and-a-block-table-6.webp)

#### Worked example: what changes on an A100 80 GB

Same model, bigger card. Llama-3.1-8B in bf16 still costs 15.0 GiB of weights; on an 80 GiB A100 you can afford a larger activation workspace, say 4 GiB, leaving roughly 61 GiB for KV.

- Token ceiling: $61 \times 1024 \times 1024 / 128 = 499{,}712$ tokens.
- Blocks at 2 MiB: $61 \times 512 = 31{,}232$ blocks.
- Contiguous with `max_seq = 4096`: $61 \times 1024 / 512 = 122$ slots, so 122 concurrent requests, 78,080 live tokens, 15.6% of ceiling.
- Paged, 640-token requests: $31{,}232 / 40 = 780$ concurrent requests, 499,200 live tokens, 99.9% of ceiling.

The ratio is the same 6.4×, because the ratio depends only on `max_seq / avg_len`, not on the card. Bigger GPUs do not dilute the fragmentation problem; they scale it. *(Source: derived. Reproduce with `python -m nanoserve.fragsim --budget-gib 61`.)*

That last observation generalizes into a formula worth remembering. Under a contiguous allocator with fixed `max_seq`, ceiling utilization is:

$$
U_{\text{contig}} = \frac{\overline{S}}{\text{max\_seq}}
$$

where $\overline{S}$ is the average live sequence length. It does not involve the GPU at all. Under paging:

$$
U_{\text{paged}} = \frac{\overline{S}}{\overline{S} + \frac{\text{block\_size} - 1}{2}}
$$

which for ${\overline{S} = 640}$ and block size 16 is $640 / 647.5 = 98.8\%$ — the slight difference from the 99.1% above is because 640 divides evenly by 16 in the simulation, so its tail waste is zero rather than average. The improvement factor is the ratio of those two expressions, and for realistic chat workloads with a generous `max_seq` it is between 4× and 10×. Set `max_seq` to 32,768 to advertise long context and serve 600-token chats, and the contiguous allocator hands you 1.8% utilization. That is not a hypothetical configuration; that is the default on a lot of deployments.

### Measuring it honestly on real hardware

The counting numbers stand on their own. The moment you want a *throughput* number, the rules from [the baseline post](/blog/machine-learning/inference-engineering/what-inference-engineering-is) apply and they are non-negotiable:

- **Warm up.** The first paged decode step pays for `torch.empty` on 6.7 GiB, kernel autotuning, and the first block-table transfer. Discard at least 20 steps.
- **`torch.cuda.synchronize()` before every timestamp**, or use `torch.cuda.Event` pairs. The block-table build is host work and the gather is device work; without a sync you will time the queue, not the kernel.
- **Measure at steady state under open-loop load.** A closed-loop harness that sends the next request only after the last one finishes can never demonstrate a concurrency improvement, because it never has 85 requests to offer. This is the single most common way people fail to reproduce a paging speedup: they change the allocator and keep the load generator, and the load generator was the bottleneck.
- **Report the resident set, not just tok/s.** `allocator.num_used * block_size` is your live token count. If it is not climbing toward the ceiling under load, the allocator is not your limiter and paging will not help you.
- **Lock clocks** (`nvidia-smi -lgc`) if you want run-to-run comparability, and say in your writeup whether you did.

A reasonable smoke test that does belong in your repo, framed as something you run rather than something I claim:

```python
# nanoserve/tests/test_pool_bytes.py
import torch
from transformers import AutoConfig

from nanoserve.blocks import PagedKVStore, plan_pool

def test_pool_matches_derivation():
    cfg = AutoConfig.from_pretrained("meta-llama/Llama-3.1-8B")
    budget = int(6.7 * 2**30)
    n = plan_pool(cfg, budget, block_size=16)
    torch.cuda.reset_peak_memory_stats()
    store = PagedKVStore(cfg, n, block_size=16)
    # The tensor should account for essentially all of the budget we planned.
    assert store.nbytes() <= budget
    assert budget - store.nbytes() < 2 * 2**20          # under one block of slack
    assert torch.cuda.max_memory_allocated() >= store.nbytes()
```

If that assertion fails on your machine, the interesting question is *why* — a different `head_dim`, an fp8 KV dtype, a config where `num_key_value_heads` equals `num_attention_heads` because the checkpoint is not GQA. All of those change the arithmetic and all of them should change the printed plan. A capacity planner that cannot be wrong is not telling you anything.

---

## 7. Stress-testing the design

A design you have not tried to break is a hypothesis. Here are the four ways this one bends.

![Matrix comparing block sizes of one, sixteen, sixty-four and two hundred fifty-six tokens on pool size, tail waste, and table length](/imgs/blogs/paged-kv-cache-implementing-blocks-and-a-block-table-7.webp)

### Tiny blocks: the overhead comes back as bookkeeping

Set `block_size = 1` and tail waste is exactly zero. Utilization hits 100%. And everything else gets worse:

- The pool holds 54,886 blocks instead of 3,430. The `deque` is fine; the `_ref` list is fine; but every request's block table is now 640 entries instead of 40, and that table is rebuilt and shipped to the GPU every decode step. For 85 concurrent requests that is a $[85, 640]$ int64 tensor — 435 KiB per step, versus 27 KiB at block size 16. At 50 decode steps per second that is the difference between 1.4 MB/s and 22 MB/s of pure metadata traffic, plus the host-side Python that builds it.
- The gather degenerates. Each "block" holds one token, so a single layer's key for one block is $8 \times 128 \times 2 = 2$ KiB and every consecutive logical token can be anywhere. You have converted a mostly-sequential read into a fully random one at 2 KiB granularity.
- Prefix caching, when it arrives, gets *better* granularity but pays hash cost per token instead of per 16 tokens.

Block size 1 is a legitimate design point — it is what a purely token-indexed cache with an indirection table looks like — and it is the right choice essentially never for dense transformer attention.

### Huge blocks: fragmentation walks back in through the front door

Set `block_size = 256` and the tail waste is $(256-1)/2 = 127.5$ tokens per request on average, which is 15.9 MiB. Now the counting changes:

| `block_size` | Pool size | Blocks per 640-tok request | Concurrent requests | Avg tail waste | Source |
| --- | --- | --- | --- | --- | --- |
| 1 | 54,886 | 640 | 85 | 0 tokens | derived |
| 16 | 3,430 | 40 | 85 | 7.5 tok (0.94 MiB) | derived |
| 64 | 857 | 10 | 85 | 31.5 tok (3.9 MiB) | derived |
| 256 | 214 | 3 | 71 | 127.5 tok (15.9 MiB) | derived |
| 1,024 | 53 | 1 | 53 | 511.5 tok (64 MiB) | derived |

At 1,024 tokens per block you have reinvented the contiguous allocator with a slightly better `max_seq`, and concurrency has fallen 38% from its peak. The lesson generalizes: **the block size is the granularity of your fragmentation.** Paging did not eliminate internal fragmentation, it made the unit small enough that the fragmentation stopped mattering. Choose a unit that is a meaningful fraction of a request and it starts mattering again immediately.

### The last partial block is a real bill, especially for prefixes

For a single request, 7.5 wasted tokens out of 640 is noise. Where it stops being noise is prefix sharing. The Anatomy post is explicit that only *complete* blocks are cacheable and that a shared prefix of `long_prefix_len` tokens forces recomputation of `long_prefix_len % block_size` tokens. That modulo is doing real work.

#### Worked example: the prefix tail on a RAG deployment

You serve a RAG endpoint with a 2,000-token system prompt shared by every request, at 20 requests per second, `block_size` 16.

- Complete blocks in the prefix: $\lfloor 2000/16 \rfloor = 125$, covering 2,000 tokens exactly? No — $125 \times 16 = 2000$, so this prefix happens to align and the remainder is 0.
- Now edit the prompt and it becomes 2,007 tokens. Complete blocks: 125, covering 2,000. Remainder: $2007 \bmod 16 = 7$ tokens recomputed on every single request.
- At 20 rps that is 140 tokens per second of prefill work you cannot cache — trivial.
- Change `block_size` to 256 and the same prompt gives $2007 \bmod 256 = 215$ tokens recomputed per request, or 4,300 tokens per second of uncacheable prefill. On a card doing perhaps 10,000–20,000 prefill tokens per second for an 8B model, that is a meaningful slice of your prefill budget spent on the same seven-word difference, forever.

*(Source: derived from the block-alignment rule cited in vLLM's Anatomy post; the prefill-rate range is an order-of-magnitude framing, not a measurement.)*

The moral is that `block_size` is not only a memory knob. It is simultaneously the fragmentation granularity, the block-table length, the gather granularity, and the prefix-cache granularity, and those four want different things. Sixteen is a compromise that has held up well enough to be a default across engines.

### Long contexts, short contexts, and the batch-1 case

At **batch 1 with a 128k context**, paging buys you approximately nothing on memory — one request, one allocation, no fragmentation to eliminate — and costs you the gather indirection. A single 128k-token request needs 8,192 blocks and 16 GiB of KV for Llama-3.1-8B, which does not fit in our 6.7 GiB budget at all, so the real conversation there is about KV quantization and attention variants, not allocation. Paging is a *concurrency* technology.

At **batch 64 with short contexts**, paging is at its best: many small requests, high variance in length, constant churn as requests finish. Every one of those properties is poison for a contiguous allocator and irrelevant to a block pool.

On an **L4 with 24 GB and roughly 300 GB/s of bandwidth versus an A100 at about 2 TB/s**, the memory arithmetic is identical — capacity is capacity — but the *value* of the extra concurrency differs. On the bandwidth-starved card, more concurrent requests amortize the weight read across more tokens and the win is large. On a card that is already closer to compute-bound at your batch size, the marginal request adds attention work without amortizing much. Paging always improves capacity; how much that capacity is worth is a roofline question, which is exactly what [the roofline model](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) is for.

---

## 8. What this design does not solve yet

Three doors we have deliberately left closed, each of which is a post of its own.

**Sharing.** Two requests with the same system prompt currently allocate two disjoint sets of blocks holding byte-identical content. The block table makes fixing this almost trivial — point both tables at the same physical block id and bump `_ref` — but "almost" hides the interesting part: what happens when one of them writes. That is copy-on-write, it needs a hash over block contents to find candidates in the first place, and the data structure that makes lookup fast is a radix tree. All of that is [prefix sharing, radix trees and copy-on-write](/blog/machine-learning/inference-engineering/prefix-sharing-radix-trees-and-copy-on-write). The refcount array in `BlockAllocator` is the hook it plugs into. Note also that `n > 1` sampling from one prompt is the same problem in miniature: the prefill blocks are shared and each sample diverges from the moment it picks a different token.

**Running out.** `OutOfBlocks` is currently an exception with nowhere to go. The scheduler has three options when the pool empties mid-generation — recompute (free a victim's blocks and re-prefill it later), swap (copy its blocks to host memory and bring them back), or reject — and each has a completely different cost curve. vLLM's offloading connector post reports that with a high CPU-side hit rate the swap path lifted concurrent throughput substantially in their setup, and also documents the DMA-versus-custom-kernel trade for the copy itself. The failure mode to fear is preemption thrash: evict A to admit B, then evict B to make room for A, forever. That is post 10.

**The kernel.** The gather in section 5 materializes tensors it does not need. The production version reads the block table inside the attention kernel, streams KV from HBM into shared memory block by block, and runs an online softmax so it never holds the full score vector. The vLLM team's *Triton Attention Backend Deep Dive* ([2026-03-04](https://vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive)) describes exactly this shape — paged KV in the innermost loop, a grid over batch and KV heads, and a split-KV decode variant that partitions the sequence traversal and combines partial softmax results in a second reduction kernel. They report their Triton implementation reaching 100.7% of FlashAttention 3's performance on H100 for Llama-3.1-8B at batch 1 with a 500-token input and a long decode, in roughly 800 lines against FA3's roughly 70,000. That is the target we are building toward in post 25.

---

## 9. Case studies and public numbers

Four public results worth knowing, each with its setup, because a number without a setup is a rumour.

**vLLM / PagedAttention (Kwon et al., SOSP 2023, [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)).** The origin. Reports that prior serving systems waste roughly 60–80% of KV memory to internal and external fragmentation, that block-based management brings waste under about 4%, and that this yields 2–4× throughput at comparable latency, with larger gains for longer sequences, larger models, and more complex decoding algorithms. That last clause matters: beam search and `n > 1` sampling benefit disproportionately because sharing, not just packing, is what saves memory there.

**Block sizing in production vLLM ([Anatomy post, 2025-09-05](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm)).** Default block size 16 tokens; per-block bytes computed as `2 * block_size * num_kv_heads * head_size * dtype_bytes`; free blocks held in a `free_block_queue` numbering "hundreds of thousands" depending on VRAM; `req_to_blocks` mapping request id to block list; prefix-cache lookup via `find_longest_cache_hit` over block hashes chained as a function of the previous block's hash and the current block's tokens. Every structural choice in this post has a counterpart there, which is the point of building it yourself.

**The KV-layout change ([offloading post, 2026-01-08](https://vllm.ai/blog/2026-01-08-kv-offloading-connector)).** Physical block sizes moved from 32 KB to 2 MB for Llama-3.1-8B, 16 KB to 0.5 MB for Llama-3.2-1B, and 8 KB to 1.25 MB for Llama-3.1-70B. The post frames this as making the copy path efficient — larger contiguous units are far better for `cudaMemcpyAsync` and GPUDirect DMA, and the same post reports DMA sustaining 83.4 GB/s bidirectional against 68.5 GB/s for a custom 16-byte-word copy kernel in their setup. This is a nice example of an allocator decision being driven by a completely different subsystem: the offload path wanted bigger contiguous blocks, so the layout changed underneath the allocator without changing the 16-token logical block size at all.

**Freed memory converts to throughput super-linearly ([distributed inference post, 2025-02-17](https://vllm.ai/blog/2025-02-17-distributed-inference)).** The vLLM team reports that going from TP=1 to TP=2 gave 13.9× more KV cache blocks and 3.9× more token throughput — super-linear in the GPU count, because the second GPU's memory relieved a KV bottleneck rather than adding compute. Same causal chain as ours: blocks are concurrency, concurrency is batch, batch is throughput. It is also a useful calibration for the gap between a capacity ratio and a throughput ratio — 13.9× of one produced 3.9× of the other.

| Result | Setup | Headline | Source |
| --- | --- | --- | --- |
| PagedAttention | Llama family vs FasterTransformer and Orca, 2023 | 2–4× throughput; waste under ~4% | cited: arXiv:2309.06180 |
| Block default | vLLM V1 engine | 16 tokens; `free_block_queue` pool | cited: vLLM Anatomy, 2025-09-05 |
| Physical block size | Llama-3.1-8B after KV-layout change | 32 KB → 2 MB | cited: vLLM offloading, 2026-01-08 |
| Memory to throughput | TP=1 → TP=2 | 13.9× blocks → 3.9× throughput | cited: vLLM distributed inference, 2025-02-17 |
| Our allocator | Llama-3.1-8B, 4090, 6.7 GiB, 640-tok requests | 13 → 85 concurrent | derived; reproduce: `nanoserve.fragsim` |

---

## 10. When to reach for this (and when not)

Build a paged allocator when:

- You are serving **many concurrent requests with variable, unpredictable lengths**. This is the entire case. Variance is what kills contiguous allocation and paging is indifferent to it.
- Your **advertised `max_seq` is much larger than your typical request**, which is essentially every chat and RAG deployment. The utilization formula $\overline{S}/\text{max\_seq}$ tells you your loss in one division.
- You intend to add **prefix sharing, preemption, or KV offload** later. All three are far easier on top of a block table than on top of contiguous slots, and retrofitting the block table afterwards means rewriting the attention path anyway.
- You are **learning how engines work**. This is 200 lines and it is the single highest-leverage data structure in LLM serving. Writing it once changes how you read every engine's source afterwards.

Do not bother when:

- You run **batch 1 on a local machine**. One request cannot fragment against itself. `llama.cpp` and friends do fine with simpler schemes for this case, and the indirection is pure cost.
- Your workload is **fixed-length by construction** — a classifier, an embedding service, a batch job where every input is padded to the same 512 tokens anyway. Contiguous is simpler and marginally faster and nothing is wasted.
- Your bottleneck is **not memory**. If your resident set is 8 requests because your arrival rate is 8 requests, a bigger pool changes nothing. Measure the resident token count under load before you touch the allocator; if it is far below the ceiling *and* the queue is empty, paging is not your problem.

And the honest version of the whole recommendation: **if you are shipping a service to users, run vLLM or SGLang.** Their allocators handle hybrid attention layouts, sliding-window layers, FP8 KV, cascade attention, multi-modal encoder caches, TP-sharded blocks and half a dozen things this post has not mentioned, and their attention kernels are five years of tuning ahead of anything you will write this quarter. Write `nanoserve/blocks.py` because understanding the allocator makes you dramatically better at operating theirs — at reading `--block-size`, `--gpu-memory-utilization` and `--max-num-seqs` as the three numbers that jointly determine your pool, and at recognizing an out-of-blocks preemption storm in a latency graph before it becomes an incident. The capstone, [the inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook), takes exactly that position across the whole stack.

---

## Key takeaways

1. **Fragmentation is an allocator problem, not a performance problem.** A contiguous KV cache with a fixed `max_seq` delivers utilization $\overline{S}/\text{max\_seq}$ — 15.2% for 640-token requests at `max_seq` 4096 — and no amount of kernel tuning recovers it.
2. **Three wastes, two of which paging kills outright.** Reserved-but-unwritten space vanishes because nothing is allocated before it is needed; non-adjacent free space vanishes because every free unit is interchangeable; only the last-partial-block tail remains, bounded by `block_size - 1` tokens.
3. **The block table is three lines of integer arithmetic.** Logical position divided by block size gives the table index, modulo gives the offset, and the table entry times the block size plus the offset gives the physical slot. Everything else is consequences.
4. **Per-block bytes are `block_size × 2 × L × H_kv × d × b`.** For Llama-3.1-8B at 16 tokens that is 2 MiB, which reproduces vLLM's published physical block size — and the 70B figure only matches once you divide by the tensor-parallel degree.
5. **Block size is four knobs at once**: fragmentation granularity, block-table length, gather granularity, and prefix-cache granularity. Sixteen is where those four stop fighting. One reinvents random access; 1,024 reinvents the contiguous allocator.
6. **Free the pool FIFO, not LIFO.** A freed block still holds valid content, and queue ordering turns the free pool into an LRU cache for free-but-useful blocks the moment prefix caching arrives.
7. **Refcount from day one.** Copy-on-write, `n > 1` sampling and beam search all need it, and retrofitting refcounts into a live allocator means auditing every free path in the engine.
8. **Capacity ratios are not throughput ratios.** Ours is a derived 6.5× more concurrent requests; the vLLM paper reports 2–4× throughput, and vLLM's own TP=1 to TP=2 result turned 13.9× more blocks into 3.9× more throughput. Quote the one you actually have.
9. **The indirection is real and it moves into the kernel.** A Python gather that materializes `[S, H, D]` per layer per step is correct and readable and roughly 80 MiB of pointless copying per decode step. The interface — block table plus length, not pointer plus stride — is what matters now; the kernel comes later.
10. **Print your pool plan on startup.** Bytes per token, bytes per block, budget, block count, token capacity. Five lines that turn every future OOM into a subtraction instead of a mystery.

---

## Further reading

- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023 — [arXiv:2309.06180](https://arxiv.org/abs/2309.06180). The primary source for everything in this post.
- vLLM, *Inside vLLM: Anatomy of a High-Throughput Inference System* (2025-09-05) — [vllm.ai/blog/2025-09-05-anatomy-of-vllm](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm). Block size, `free_block_queue`, `req_to_blocks`, `find_longest_cache_hit`, and the block-alignment rule for prefix caching.
- vLLM, *KV Cache Offloading to CPU* (2026-01-08) — [vllm.ai/blog/2026-01-08-kv-offloading-connector](https://vllm.ai/blog/2026-01-08-kv-offloading-connector). Physical block sizes per model and why the copy path wanted them larger.
- vLLM, *Triton Attention Backend Deep Dive* (2026-03-04) — [vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive](https://vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive). What the gather looks like when it lives inside the kernel.
- [The memory math of the KV cache](/blog/machine-learning/inference-engineering/the-memory-math-of-the-kv-cache) — where the 128 KiB per token comes from, and the capacity tables for the rest of the model matrix.
- [Why recompute is fatal: writing a KV cache](/blog/machine-learning/inference-engineering/why-recompute-is-fatal-writing-a-kv-cache) — the contiguous cache this post replaces, and the layout choices that made it awkward.
- [Prefix sharing, radix trees and copy-on-write](/blog/machine-learning/inference-engineering/prefix-sharing-radix-trees-and-copy-on-write) — what the refcount array in `BlockAllocator` is for.
- [Continuous batching and PagedAttention](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention) — how the allocator and the scheduler compose in a production engine.
- [The roofline model: compute-bound vs memory-bound](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) — for deciding how much the extra concurrency is actually worth on your card.
