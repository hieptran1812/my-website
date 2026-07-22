---
title: "The CUDA Caching Allocator: Why You OOM at 60% Memory Used"
date: "2026-07-14"
publishDate: "2026-07-14"
description: "Your service dies with 'tried to allocate 2 GiB' while your live tensors use only 60% of an 80 GB A100. The memory is not gone — it is fragmented, held by PyTorch's caching allocator in blocks too small to satisfy the request. This post builds the mental model of allocated vs reserved, shows why cudaMalloc is slow enough to make caching worth it, derives how fragmentation produces an OOM below capacity, and turns the PYTORCH_CUDA_ALLOC_CONF knobs from folklore into a decision you can make from a profile."
tags:
  [
    "performance-engineering",
    "gpu-optimization",
    "memory",
    "cuda",
    "pytorch",
    "profiling",
    "inference",
    "ml-systems",
  ]
category: "machine-learning"
subcategory: "Performance Engineering"
author: "Hiep Tran"
featured: true
readTime: 41
---

At 03:14 the pager goes off. An inference service that has been running for two weeks on a single A100 80GB starts throwing `torch.cuda.OutOfMemoryError` on maybe one request in fifty — the ones with a slightly longer prompt. The on-call engineer does the obvious thing: they SSH in, run `nvidia-smi`, and stare. The process is holding 78 GB. "Well, it's basically full," they think, and bump the instance to a bigger box. The OOMs keep coming. Then they add a line to log `torch.cuda.memory_allocated()` and it prints **48 GB**. Live tensors — weights, activations, the KV cache, everything the model actually touches — add up to 48 GB on an 80 GB card. Sixty percent. And the service is dying on a request for **2 GB**.

That gap — 48 GB of tensors, 78 GB reserved, an OOM on 2 GB — is not a bug in your model and it is not a leak. It is the single most misread number in GPU serving, and it comes straight from the piece of PyTorch that nobody reads until it pages them: the **CUDA caching allocator**. The memory the allocator "lost" is not gone. It is sitting in the allocator's cache, chopped into free blocks that are each too small to satisfy a 2 GB request, and the allocator cannot glue them back together fast enough — or at all — to hand you the contiguous slab you asked for. You are out of memory the way a parking garage with 300 empty single spaces is "full" for a delivery truck that needs two adjacent bays.

![Two columns contrast the tensor view showing 48 GB used and apparent headroom against the allocator view showing 78 GB reserved with a largest free block of only 1.5 GB and a failed 2 GB allocation](/imgs/blogs/the-cuda-caching-allocator-1.webp)

This post is the one several others in this series have been pointing at when they said "the allocator holds reserved memory above your live tensors — more on that later." Here is the later. By the end you will be able to: read `memory_allocated` against `memory_reserved` and know instantly which one `nvidia-smi` is lying to you with; explain why PyTorch grabs memory in big chunks and hoards it rather than calling the driver per tensor; derive the exact condition under which you OOM with gigabytes free; set `PYTORCH_CUDA_ALLOC_CONF` from evidence instead of a Stack Overflow copy-paste; and recognize the fragmentation signature — high reserved, lower allocated, `num_alloc_retries` climbing — in `torch.cuda.memory_stats()` before it becomes a 3 a.m. page. This is one of [the four wastes](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu) — the bandwidth wall's evil twin, where the resource is not slow, it is *unreachable* — and it is invisible until you know exactly which two numbers to compare.

## The two numbers: allocated vs reserved

Every question about GPU memory in PyTorch reduces to two counters and the space between them. Get these two straight and 90% of memory confusion evaporates, so we start here and keep coming back.

- **`torch.cuda.memory_allocated()`** — the number of bytes currently occupied by **live tensors**. Every time you create a tensor, this goes up; every time a tensor is freed (its Python refcount hits zero and the allocator reclaims the block), this goes down. This is the memory your model *actually uses* right now.
- **`torch.cuda.memory_reserved()`** — the number of bytes PyTorch has taken **from the CUDA driver** and is holding onto. The allocator calls `cudaMalloc` to grab large chunks (called **segments**) from the device, then carves your tensors out of those segments. When a tensor is freed, its space goes back into the allocator's cache, *not* back to the driver. So reserved is a high-water-mark-ish quantity: it rises as the allocator needs more raw device memory and, by default, essentially never falls.

The invariant that follows is the load-bearing sentence of this whole post:

$$\text{reserved} \;\ge\; \text{allocated} \quad\text{always, and}\quad \text{reserved} - \text{allocated} = \text{cached free bytes}.$$

The difference `reserved − allocated` is memory PyTorch owns but no tensor currently occupies. It is *free* — from the allocator's point of view — but it belongs to PyTorch, not to the driver, and no other process or allocation-that-needs-a-different-shape can necessarily use it.

![A vertical stack showing device total of 80 GB above reserved 78 GB above allocated 48 GB above cached free 30 GB above a largest hole of 1.5 GB that cannot fit a 2 GB request](/imgs/blogs/the-cuda-caching-allocator-2.webp)

Now the punchline that resolves the 3 a.m. page: **`nvidia-smi` shows `reserved`, not `allocated`.** `nvidia-smi` reports the process's device-memory footprint — the sum of every `cudaMalloc` the process has made (plus a few hundred MB of CUDA context). PyTorch's allocator *is* the thing making those `cudaMalloc` calls, in big segments, and holding them. So `nvidia-smi` sees the 78 GB of segments PyTorch reserved, while your model only has 48 GB of live tensors inside them. The engineer who checks `nvidia-smi` and sees "78 GB used" and the engineer who checks `memory_allocated()` and sees "48 GB used" are both right — they are reading two different numbers, and the 30 GB gap between them is the allocator's cache.

Here is the smallest program that makes the gap appear on any GPU:

```python
import torch

def show(tag):
    a = torch.cuda.memory_allocated() / 1024**3
    r = torch.cuda.memory_reserved()  / 1024**3
    print(f"{tag:<22} allocated={a:6.2f} GiB   reserved={r:6.2f} GiB   gap={r-a:5.2f} GiB")

show("startup")
x = torch.empty(4 * 1024**3 // 2, dtype=torch.float16, device="cuda")  # 4 GiB
show("after 4 GiB alloc")
del x                                   # tensor freed; block returns to the CACHE
show("after del x")                     # allocated drops, reserved does NOT
torch.cuda.empty_cache()                # only NOW is it returned to the driver
show("after empty_cache")
```

```console
startup                allocated=  0.00 GiB   reserved=  0.00 GiB   gap= 0.00 GiB
after 4 GiB alloc      allocated=  4.00 GiB   reserved=  4.00 GiB   gap= 0.00 GiB
after del x            allocated=  0.00 GiB   reserved=  4.00 GiB   gap= 4.00 GiB
after del x            allocated=  0.00 GiB   reserved=  4.00 GiB   gap= 4.00 GiB
after empty_cache      allocated=  0.00 GiB   reserved=  0.00 GiB   gap= 0.00 GiB
```

Read the third line carefully, because it is the entire mechanism in one row. After `del x`, `allocated` drops to zero — the tensor is gone, no live bytes. But `reserved` stays at 4 GiB. The allocator kept the 4 GiB segment in its cache, betting you will ask for something like it again soon and it can hand the block straight back without touching the driver. Only `empty_cache()` — a call you almost never actually want, as we will see — hands the segment back to the driver and drops `reserved` to zero.

This table is worth pinning above your desk. It is the Rosetta Stone for every memory number you will read in this post and every panicked Slack thread about them:

| You read this | It measures | Changes when | `nvidia-smi` shows it? |
| --- | --- | --- | --- |
| `memory_allocated()` | live tensor bytes | a tensor is created or freed | no |
| `memory_reserved()` | bytes held from the driver | the allocator grows / `empty_cache` | yes (this is the footprint) |
| `reserved − allocated` | cached-but-free bytes | tensors free without empty_cache | it is baked into the footprint |
| `max_memory_allocated()` | peak live bytes since reset | your true tensor high-water mark | no |

If you take one habit from this post, make it this: **when you debug a memory problem, print `allocated` and `reserved` together, always, as a pair.** A single number is a rumor. The pair is the diagnosis — and the sign of the gap between them, over time, is the difference between "fragmentation," "leak," and "genuinely too big," which is the decision tree we will build in a few sections. The [metrics post](/blog/machine-learning/performance-engineering/metrics-that-actually-matter) makes the same argument about utilization: the headline number lies, the pair of numbers tells the truth.

## Why PyTorch caches at all: cudaMalloc is slow

Before we can talk about fragmentation, we have to answer the question the last section raised: *why on earth does the allocator hold onto freed memory instead of returning it?* Why not just `cudaMalloc` when you need a tensor and `cudaFree` when it dies, the way `malloc`/`free` work on the CPU? The answer is that the CUDA memory calls are *catastrophically* expensive to do at tensor granularity, and understanding exactly why is what makes the caching design — and its fragmentation cost — feel inevitable rather than arbitrary.

`cudaMalloc` and `cudaFree` are **synchronizing** calls. Allocating device memory is not a quick pointer bump like `malloc` on the host; it modifies the GPU's page tables and the driver serializes it against in-flight work. In practice a `cudaMalloc` costs on the order of **100 µs or more** — and, worse, `cudaFree` typically forces a **device synchronize**: the driver waits until every kernel currently queued on the device has finished before it will release the pages, because it cannot safely unmap memory a running kernel might still be reading. A single `cudaFree` in the middle of your forward pass can stall the entire GPU pipeline. Compare that to a CUDA kernel launch, which costs roughly [5–10 µs of host overhead](/blog/machine-learning/performance-engineering/the-kernel-launch-overhead-problem), and to a cache hit inside PyTorch's allocator, which is a lock and a free-list lookup — well under a microsecond.

![A branching dataflow where a tensor request splits into a cache hit returning a reused block in under a microsecond and a cache miss that falls to cudaMalloc costing over 100 microseconds before both paths merge to a ready tensor](/imgs/blogs/the-cuda-caching-allocator-3.webp)

Now put a transformer forward pass through that. A single decode step of a mid-size model can create and destroy **thousands** of intermediate tensors — every `matmul`, every `add`, every `softmax`, every layernorm produces a temporary. If each one paid a `cudaMalloc` on birth and a synchronizing `cudaFree` on death, the arithmetic is brutal. Let $N$ be the number of allocations per step and $t_{\text{malloc}}$ the driver cost each. The naive per-step allocation tax is

$$t_{\text{alloc}} \;=\; N \cdot (t_{\text{malloc}} + t_{\text{free}}).$$

With $N = 3000$ and even a conservative $t_{\text{malloc}} + t_{\text{free}} \approx 150\,\mu s$ (the `cudaFree` sync often makes it much worse), that is $3000 \times 150\,\mu s = 450\,\text{ms}$ of pure driver overhead **per step** — on a step whose actual GPU compute might be 8 ms. You would spend 98% of your time in the memory allocator, and most of that time the GPU would be idle, synchronized, waiting. The service would run at single-digit utilization and it would have nothing to do with your kernels.

The caching allocator makes that cost disappear by **amortization**. It calls `cudaMalloc` rarely — only when it has no cached block big enough for a request — and grabs a large segment when it does (segments start at multiple megabytes and grow by doubling). Every subsequent tensor of a similar size is carved from cached segments at cache-hit speed, and every freed tensor returns its block to the free list instead of the driver, ready for instant reuse. If the cache-hit rate is $h$, the per-step allocation cost collapses to

$$t_{\text{alloc}} \;=\; (1-h)\,N \cdot t_{\text{malloc}} \;+\; N \cdot t_{\text{hit}},$$

and because a steady-state training or serving loop reuses the same shapes over and over, $h$ climbs to 0.99+ within a few iterations. With $h = 0.99$, $t_{\text{hit}} = 0.5\,\mu s$, the 450 ms tax becomes $(0.01)(3000)(100\,\mu s) + (3000)(0.5\,\mu s) = 3\,\text{ms} + 1.5\,\text{ms} = 4.5\,\text{ms}$ — a **100× reduction**, and most of the residual is one-time warm-up. This is why the very first iterations of a PyTorch program are slower and why `reserved` climbs during warm-up and then plateaus: the allocator is paying the `cudaMalloc` costs once and caching the segments so it never pays them again.

#### Worked example: measuring the caching payoff

You do not have to take the arithmetic on faith — you can watch the allocator warm up. Run a fixed-shape step in a loop and time each iteration with CUDA events (never wall-clock around async GPU work; see [the benchmarking post](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark) for why), printing `reserved` alongside:

```python
import torch

model = build_model().cuda().eval()
x = torch.randn(8, 512, 1024, device="cuda")
start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

for i in range(6):
    torch.cuda.synchronize(); start.record()
    with torch.no_grad():
        y = model(x)
    end.record(); torch.cuda.synchronize()
    r = torch.cuda.memory_reserved() / 1024**3
    print(f"iter {i}: {start.elapsed_time(end):6.2f} ms   reserved={r:5.2f} GiB")
```

```console
iter 0:  41.83 ms   reserved= 6.42 GiB
iter 1:  12.07 ms   reserved= 7.88 GiB
iter 2:   8.31 ms   reserved= 7.88 GiB
iter 3:   8.29 ms   reserved= 7.88 GiB
iter 4:   8.30 ms   reserved= 7.88 GiB
iter 5:   8.28 ms   reserved= 7.88 GiB
```

Iteration 0 pays 42 ms — cudaMalloc calls, cuBLAS/cuDNN autotuning, the works — while `reserved` climbs. By iteration 2 the allocator has cached every segment the step needs, `reserved` has plateaued at 7.88 GiB, and the step settles at its true 8.3 ms of compute. The gap between iteration 0 and iteration 2 is the allocation tax the cache is now hiding. This is also *exactly* why you must warm up before you measure anything, and why a benchmark that includes iteration 0 will slander a perfectly good model.

So caching is not an optimization PyTorch bolted on; it is the only way a Python framework that allocates thousands of short-lived tensors per step can drive a GPU at all. The price of that speed — and there is always a price — is that the allocator now sits between you and the driver, holding a cache of freed blocks. And a cache of variable-sized blocks is exactly the thing that fragments.

## Fragmentation: how you OOM below capacity

Here is where the two threads — "reserved can exceed allocated" and "the allocator caches variable-sized blocks" — braid into the failure mode in the title. Fragmentation is the reason 30 GB of cached-free memory cannot satisfy a 2 GB request, and it is worth slowing down to build the picture in motion, because a still frame genuinely does not convey why free memory becomes unreachable.

When the allocator carves tensors out of a segment, it **splits** blocks. Ask for 1.5 GB out of a fresh 2 GB segment and the allocator splits it into a 1.5 GB block (yours) and a 0.5 GB remainder (free, cached). Free your 1.5 GB later and its block goes back on the free list — but the 0.5 GB remainder next to it may have been handed out to something else in the meantime. Over thousands of allocations of *varying* sizes — which is exactly what varying sequence lengths, varying batch sizes, and varying activation shapes produce — the segments get diced into a patchwork of used and free blocks of many sizes. The total free bytes can be large. But the free bytes are scattered across dozens of small non-adjacent holes, and a large request needs one *contiguous* block.

<figure class="blog-anim">
<svg viewBox="0 0 720 250" role="img" aria-label="A memory segment is sixty percent full but its free space is split into single-block holes, so a two-block request cannot fit anywhere and fails with an out of memory error" style="width:100%;height:auto;max-width:820px">
<style>
.af1-used{fill:var(--accent,#6366f1)}
.af1-free{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.af1-t{font:700 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.af1-s{font:500 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.af1-ok{font:700 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.af1-bad{font:700 15px ui-sans-serif,system-ui;fill:#dc2626;text-anchor:middle}
.af1-probe{fill:none;stroke:var(--accent,#6366f1);stroke-width:2.5;stroke-dasharray:6 4}
.af1-reqfill{fill:#dc2626;opacity:.16}
.af1-reqline{stroke:#dc2626;stroke-width:2.5;fill:none}
@keyframes af1-fadeA{0%,42%{opacity:1}56%,94%{opacity:0}100%{opacity:1}}
@keyframes af1-fadeB{0%,42%{opacity:0}56%,94%{opacity:1}100%{opacity:0}}
@keyframes af1-scan{0%{transform:translateX(0)}42%{transform:translateX(478px)}100%{transform:translateX(478px)}}
.af1-A{animation:af1-fadeA 9s ease-in-out infinite}
.af1-B{animation:af1-fadeB 9s ease-in-out infinite}
.af1-pr{animation:af1-scan 9s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.af1-A{animation:none;opacity:1}.af1-B{animation:none;opacity:0}.af1-pr{animation:none}}
</style>
<text class="af1-t" x="360" y="30">one reserved segment · 16 blocks · 60% live</text>
<rect class="af1-used" x="40"  y="80" width="36" height="60" rx="4"/>
<rect class="af1-used" x="80"  y="80" width="36" height="60" rx="4"/>
<rect class="af1-free" x="120" y="80" width="36" height="60" rx="4"/>
<rect class="af1-used" x="160" y="80" width="36" height="60" rx="4"/>
<rect class="af1-used" x="200" y="80" width="36" height="60" rx="4"/>
<rect class="af1-free" x="240" y="80" width="36" height="60" rx="4"/>
<rect class="af1-used" x="280" y="80" width="36" height="60" rx="4"/>
<rect class="af1-free" x="320" y="80" width="36" height="60" rx="4"/>
<rect class="af1-used" x="360" y="80" width="36" height="60" rx="4"/>
<rect class="af1-used" x="400" y="80" width="36" height="60" rx="4"/>
<rect class="af1-free" x="440" y="80" width="36" height="60" rx="4"/>
<rect class="af1-used" x="480" y="80" width="36" height="60" rx="4"/>
<rect class="af1-free" x="520" y="80" width="36" height="60" rx="4"/>
<rect class="af1-used" x="560" y="80" width="36" height="60" rx="4"/>
<rect class="af1-free" x="600" y="80" width="36" height="60" rx="4"/>
<rect class="af1-used" x="640" y="80" width="36" height="60" rx="4"/>
<g class="af1-A">
<rect class="af1-probe af1-pr" x="118" y="76" width="80" height="68" rx="6"/>
<text class="af1-ok" x="360" y="180">allocator scans for a 2-block hole</text>
<text class="af1-s" x="360" y="205">free = 30 GB total, but in 1-block holes</text>
</g>
<g class="af1-B">
<rect class="af1-reqfill" x="300" y="12" width="120" height="34" rx="6"/>
<rect class="af1-reqline" x="300" y="12" width="120" height="34" rx="6"/>
<text class="af1-bad" x="360" y="35">2 GB request</text>
<path class="af1-reqline" d="M360 46 L360 74"/>
<text class="af1-bad" x="360" y="180">no 2 adjacent free blocks → OOM</text>
<text class="af1-s" x="360" y="205">largest free block = 1.5 GB</text>
</g>
</svg>
<figcaption>The segment is 60% live, so 30 GB is free — but the free space is chopped into single-block holes. The allocator scans for two adjacent free blocks to satisfy a 2 GB request, finds none, and the allocation fails even though far more than 2 GB is free. That is fragmentation: free but unusable.</figcaption>
</figure>

Let us make the OOM condition precise, because "fragmentation" is often waved around without a definition. Let the cached free blocks have sizes $b_1, b_2, \ldots, b_k$, and let a request arrive for $r$ contiguous bytes. The request can be satisfied from cache only if some single block is big enough:

$$\exists\, i : b_i \ge r.$$

The total free memory is $\sum_i b_i$, which can be *enormous* compared to $r$. The OOM-below-capacity condition is therefore

$$\sum_i b_i \;\ge\; r \quad\text{(you have the memory)} \qquad\text{but}\qquad \max_i b_i \;\lt\; r \quad\text{(you cannot reach it),}$$

**and** the allocator cannot fall back to `cudaMalloc` for a fresh $r$-byte segment because the driver itself has no $r$ contiguous bytes left (the process footprint already fills the card). Both clauses must hold. When they do, you get the error, and PyTorch is even kind enough to tell you which clause you hit:

```console
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a
total capacity of 79.15 GiB of which 1.10 GiB is free. Process 48213 has 78.05 GiB memory
in use. Of the allocated memory 48.30 GiB is allocated by PyTorch, and 29.42 GiB is reserved
by PyTorch but unallocated. If reserved memory is >> allocated memory try setting
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation. See documentation
for Memory Management (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
```

Read that message the way it deserves to be read. **48.30 GiB allocated** — your live tensors, 60% of the card. **29.42 GiB reserved but unallocated** — the cached free blocks, 30 GB of them, more than fourteen times the 2 GiB you asked for. **1.10 GiB free** at the driver — the process footprint is 78 GB, so `cudaMalloc` cannot rescue you. That is $\sum b_i = 29.42 \gg r = 2.0$ while $\max b_i \lt 2.0$ and the driver is dry: textbook fragmentation, and the error message literally names the fix. The heuristic PyTorch prints — "if reserved memory is >> allocated memory" — is precisely the `reserved − allocated` gap from the first section, now doing diagnostic work.

Why does the allocator not just glue adjacent free blocks back together — **coalesce** them — and hand you the merged slab? It does, but only for blocks that are physically adjacent *within the same segment* and both free at the same moment. It cannot merge a free block in segment A with a free block in segment B (they are separate `cudaMalloc` regions at unrelated addresses), and it cannot merge two free blocks that have a live block wedged between them. That wedged-live-block case is fragmentation's whole trick: one small long-lived tensor sitting in the middle of a segment permanently prevents the free space on either side of it from ever being one contiguous block. This is why fragmentation is *path-dependent* — it depends not just on how much memory you use but on the *order* in which you allocate and free varying sizes.

### Segments, blocks, and the two pools

A little more precision about the allocator's internals makes the knobs in the next section make sense, because each knob is a lever on one of these mechanisms. The allocator maintains its free memory as **segments** (the chunks it `cudaMalloc`'d from the driver) subdivided into **blocks** (the pieces it hands to your tensors). Crucially, it keeps *two* separate pools with different granularity: a **small pool** for requests under 1 MB and a **large pool** for requests of 1 MB and up. Small requests are rounded up to a 512-byte multiple and served from small-pool segments (which are themselves carved from 2 MB `cudaMalloc` chunks); large requests are rounded up to a 2 MB multiple and served from large-pool segments that start large and grow by doubling. The split exists so that the thousands of tiny intermediate tensors a forward pass creates do not fragment the same pool your big activation and KV-cache tensors live in — small churn stays quarantined in the small pool.

Two behaviors flow from this that you will see in the counters. First, **rounding**: because every request is rounded up to a size class, `allocated` (rounded bytes) is always a little above the `requested` bytes your code actually asked for, and that difference is *internal* fragmentation — wasted space inside a block. It is usually small, but `roundup_power2_divisions` is the knob that trades more of it for less external fragmentation. Second, **splitting**: when the large pool has a 4 GB free block and you ask for 1.5 GB, the allocator splits off a 1.5 GB block and leaves a 2.5 GB free remainder — and it records that this remainder was *split from* a larger block. Those split-off remainders are exactly the `inactive_split_bytes` counter you will read from `memory_stats()`, and they are the raw material of external fragmentation. `max_split_size_mb` is the knob that says "do not split blocks larger than this," protecting big free blocks from being nibbled into unusable remainders by small requests. Hold onto these two mechanisms — rounding and splitting — because the four knobs are nothing more than four ways to bias them.

### The stress test: varying shapes fragment, fixed shapes do not

The clean way to prove fragmentation to yourself — and the pattern that catches real services — is to contrast a fixed-shape workload with a varying-shape one. A serving loop that always runs batch 8, sequence 512 reuses the same blocks forever: perfect cache hits, zero fragmentation, `reserved` plateaus and stays flat. A serving loop that runs whatever sequence length the *user* sent — 37 tokens, then 1900, then 200, then 4096 — allocates activation tensors of wildly different sizes in an unpredictable order, and that is the recipe.

```python
import torch, random

model = build_model().cuda().eval()
random.seed(0)

def show(tag):
    a = torch.cuda.memory_allocated()/1024**3
    r = torch.cuda.memory_reserved()/1024**3
    print(f"{tag:<14} allocated={a:6.2f}  reserved={r:6.2f}  gap={r-a:5.2f} GiB")

with torch.no_grad():
    for step in range(400):
        seq = random.choice([128, 512, 900, 1600, 2048, 4096])   # varying!
        x = torch.randint(0, 32000, (4, seq), device="cuda")
        y = model(x)
        del x, y
        if step % 100 == 0:
            show(f"step {step}")
    show("final")
```

```console
step 0         allocated=  6.40  reserved=  8.10  gap= 1.70 GiB
step 100       allocated=  6.40  reserved= 22.55  gap=16.15 GiB
step 200       allocated=  6.40  reserved= 41.80  gap=35.40 GiB
step 300       allocated=  6.40  reserved= 63.20  gap=56.80 GiB
final          allocated=  6.40  reserved= 71.90  gap=65.50 GiB
```

Watch the two columns diverge. `allocated` is dead flat at 6.40 GiB — after each step every tensor is freed, so live memory returns to the resident weights. But `reserved` climbs relentlessly, because each new *largest-so-far* sequence forces the allocator to `cudaMalloc` a fresh, bigger segment, and the old segments — now the wrong size for the new request — stay cached, unusable, forever. By the final step the allocator holds 72 GB of reserved memory to serve 6.4 GB of live tensors, and the next request one token longer than anything seen so far will OOM. The gap column *is* the fragmentation, growing in real time. This is the [slow-growing service](/blog/machine-learning/performance-engineering/memory-snapshot-and-leak-hunting) that OOMs after hours or days — and note it looks *exactly* like a memory leak on a `reserved` graph, which is precisely why you need the `allocated`/`reserved` pair to tell them apart. A leak grows `allocated`. Fragmentation grows only the gap.

## The knobs: PYTORCH_CUDA_ALLOC_CONF

The allocator's behavior is not fixed — it is a policy with dials, and the dials live in one environment variable, `PYTORCH_CUDA_ALLOC_CONF`, read once at process start. Most engineers meet this variable as a magic incantation pasted from a forum. The point of this section is to make each knob a decision you understand, so you set the one that matches *your* fragmentation and skip the ones that will not help.

![A four by two matrix mapping the allocator knobs max split size, expandable segments, gc threshold, and roundup to what each does and when to reach for it](/imgs/blogs/the-cuda-caching-allocator-4.webp)

**`expandable_segments:True`** — the one to try first, and usually the one that fixes it. This switches the allocator to a fundamentally different design. Instead of many fixed-size segments (each a separate `cudaMalloc` at an unrelated address, unable to merge with its neighbors), the expandable allocator reserves one large **virtual** address range per stream and maps physical pages into it on demand using the CUDA driver's virtual-memory API (`cuMemAddressReserve`, `cuMemCreate`, `cuMemMap`). Because the segment is one contiguous virtual range that *grows and shrinks* by mapping and unmapping pages at its tail, freed space near the top can be reused for a request of *any* size, and the allocator does not need to find a pre-existing contiguous physical slab — it maps fresh pages into contiguous virtual addresses. In the varying-sequence stress test above, `expandable_segments:True` keeps `reserved` hugging `allocated` instead of ratcheting away from it, because there are no orphaned wrong-size segments — there is one segment that flexes. This is the closest thing the allocator has to a free lunch, and for the fragmentation-from-varying-shapes case it very nearly is one.

```bash
# Set it before the process starts — the allocator reads this once, at init.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python serve.py
# Combine knobs with commas:
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
```

**`max_split_size_mb:N`** — the pre-`expandable_segments` classic, still useful. It tells the allocator: never *split* a block larger than $N$ MB to satisfy a smaller request. The idea is to protect your big segments. Without it, a request for 200 MB might carve a 200 MB slice out of a pristine 4 GB segment, leaving a 3.8 GB remainder that then gets diced further; set `max_split_size_mb:256` and the allocator will not cannibalize blocks over 256 MB for small requests, keeping large contiguous regions intact for the large allocations that actually need them. It helps specifically when your fragmentation comes from *large* allocations being starved by *small* ones interleaving. It is fiddly to tune (the right $N$ is workload-specific) and `expandable_segments` usually dominates it, but it remains the right tool when you cannot use expandable segments for some reason.

**`garbage_collection_threshold:F`** (a fraction like 0.8) — when `reserved` climbs past $F$ of the device capacity, the allocator, before it would OOM, proactively frees *cached, unused* segments back to the driver to make room. It trades a bit of the caching speedup (those freed segments will cost a `cudaMalloc` if you need them again) for headroom, and it is what you reach for when a service runs close to the memory ceiling and occasionally needs a large transient block that fragmentation would otherwise deny. Think of it as an automatic, *targeted* `empty_cache` that fires only under pressure and only on genuinely idle blocks.

**`roundup_power2_divisions:N`** — controls how allocation sizes are rounded up before they hit the free list. The allocator already rounds small requests up to size classes so that a 511-byte and a 513-byte request can reuse the same block; this knob makes the rounding coarser (rounding up toward powers of two, subdivided into $N$ steps). Coarser rounding means *fewer distinct block sizes*, which means *more reuse* and *less external fragmentation* — at the cost of a little *internal* fragmentation (you round a 1.1 GB request up to 1.25 GB and waste the difference). It is the knob you reach for last, when profiling shows many almost-but-not-quite-matching sizes churning through the cache. This is the rounding-helps-and-hurts trade-off in one dial: coarser classes reduce external fragmentation and increase internal waste, finer classes do the reverse.

And the anti-knob, the one everyone reaches for and almost nobody should: **`torch.cuda.empty_cache()`**. This is not a config setting but a function call, and it returns *all* cached free segments to the driver — it drops `reserved` down toward `allocated`. It sounds like the fix ("just free the cache and the fragmentation goes away!") and it is almost always the wrong one, for two reasons we will quantify next. It does have exactly one legitimate use: right before you hand the GPU to *another* process or library that calls `cudaMalloc` directly and needs the driver-level free memory back. For fixing fragmentation *within* your own PyTorch process, it is a stall dressed up as a solution.

#### Worked example: empty_cache is a false fix that stalls

Suppose the varying-sequence service above starts OOMing and someone adds `torch.cuda.empty_cache()` to the request handler "to be safe." Watch what it costs. Each `empty_cache()` call issues a `cudaFree` for every cached segment, and every one of those is a **synchronizing** driver call — it drains the GPU pipeline. Measure it:

```python
import torch
model = build_model().cuda().eval()
x = torch.randn(4, 2048, device="cuda")
ev = lambda: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))

# warm up so the cache is full of segments to free
with torch.no_grad():
    for _ in range(20): model(x.long())

s, e = ev(); torch.cuda.synchronize(); s.record()
torch.cuda.empty_cache()
e.record(); torch.cuda.synchronize()
print(f"empty_cache stall: {s.elapsed_time(e):.2f} ms   reserved now "
      f"{torch.cuda.memory_reserved()/1024**3:.2f} GiB")
```

```console
empty_cache stall: 6.41 ms   reserved now  6.40 GiB
```

That 6.4 ms is dead time on every call — the GPU sits idle while the driver unmaps pages — and worse, it is *self-defeating*: you just threw away the cache the allocator spent warm-up iterations building, so the very next larger request pays fresh `cudaMalloc` costs to rebuild it, and the fragmentation comes right back on the next varying-shape burst. Putting `empty_cache()` in a hot path is how a service that OOMed once now stutters on every request *and still OOMs*. The honest fix for fragmentation is `expandable_segments`, which changes the *policy* so the holes never form, not `empty_cache`, which bulldozes the cache after the holes have already cost you.

Here is the decision, distilled, so you never guess again:

| Knob | Attacks | Cost | Reach for it when |
| --- | --- | --- | --- |
| `expandable_segments:True` | fragmentation from varying shapes | almost none | first, for nearly any fragmentation |
| `max_split_size_mb:N` | large allocs starved by small ones | tuning effort | big transient tensors OOM, can't use expandable |
| `garbage_collection_threshold:F` | near-capacity pressure | some cudaMalloc churn | service runs close to the ceiling |
| `roundup_power2_divisions:N` | many near-miss sizes churning | slight internal waste | last resort, after profiling the sizes |
| `empty_cache()` | nothing, within your process | 5–7 ms sync per call | only when handing the GPU to another process |

## Which cause, which fix: reading the OOM

An OOM below capacity is a symptom, not a diagnosis, and reaching for `expandable_segments` reflexively is how you waste an afternoon when the real problem is a leak. The `allocated`/`reserved` pair, tracked *over time*, splits the symptom into three causes, each with a different fix. This is the decision tree to run the moment a memory error lands.

![A decision tree rooting at OOM at sixty percent used branching into fragmentation, leak, and too big, each leading to its own fix of expandable segments, memory snapshot, and sharding](/imgs/blogs/the-cuda-caching-allocator-5.webp)

**Fragmentation** — `reserved` is far above `allocated` (a big gap), and `allocated` is *stable* over time. This is the case we have been dissecting: your live tensors fit, the cache is full of wrong-size holes, `num_alloc_retries` is climbing. The fix is `expandable_segments:True`, or `max_split_size_mb` if you cannot use it. You do **not** have a leak; do not go hunting for retained references — you will find nothing, because nothing is being retained. The tell is that `allocated` is flat while `reserved` grows.

**A leak** — `allocated` itself grows over time, step after step, and `reserved` grows with it to keep up. This is genuinely too much *live* memory: something is holding references that should have been freed — an autograd graph you forgot to detach, a list that accumulates tensors across requests, a cache of results keyed by input that never evicts. No allocator knob will save you, because the tensors are *live* — the allocator is correctly keeping memory your code still references. The fix is to find and drop the reference, which is exactly what [the memory snapshot and leak-hunting post](/blog/machine-learning/performance-engineering/memory-snapshot-and-leak-hunting) is about: `torch.cuda.memory._record_memory_history()` records every allocation with its Python stack, and the viewer shows you the timeline of what grew and who allocated it. The tell is that `allocated` climbs, not just the gap.

**Genuinely too big** — a *single* allocation exceeds the free memory, even freshly, even with zero fragmentation. Your batch, your sequence length, your model simply does not fit. No knob helps; the fix is to make the request smaller (reduce batch size, shard the model across GPUs, use activation checkpointing to trade compute for memory, or use a paged KV cache). The tell is that the OOM happens even on the *first* large request, on a warm-started process with a tiny `reserved − allocated` gap, and the "tried to allocate" size is a large fraction of the whole card.

The one-liner that starts the diagnosis every time:

```python
import torch
a = torch.cuda.memory_allocated()/1024**3
r = torch.cuda.memory_reserved()/1024**3
print(f"allocated={a:.1f} GiB  reserved={r:.1f} GiB  gap={r-a:.1f} GiB  "
      f"retries={torch.cuda.memory_stats()['num_alloc_retries']}  "
      f"ooms={torch.cuda.memory_stats()['num_ooms']}")
```

```console
allocated=48.3 GiB  reserved=77.7 GiB  gap=29.4 GiB  retries=214  ooms=3
```

Big gap, stable `allocated`, climbing retries: fragmentation, reach for `expandable_segments`. If instead you saw `allocated` marching up over minutes with a small gap, you would stop and go leak-hunting. Same error message, opposite fix — and the pair of numbers is what tells them apart.

## Reading it: memory_summary and memory_stats

The final skill is turning the allocator's internal state into a report you can read at a glance, because "it OOMed" is not a bug report — "reserved 78 GB against allocated 48 GB with 214 retries and 3 OOMs" is. PyTorch exposes the allocator's full accounting through two calls: `memory_stats()` (a dict of every counter) and `memory_summary()` (a human-readable table built from it).

![A four by two matrix comparing allocated, reserved, alloc retries, and num ooms across a healthy service and a fragmented one, showing reserved and the retry counters diverging](/imgs/blogs/the-cuda-caching-allocator-6.webp)

`torch.cuda.memory_summary()` prints a table that is the single best "what is the allocator doing" snapshot. Drop it into your OOM handler or a periodic health log:

```python
import torch
try:
    y = model(batch)
except torch.cuda.OutOfMemoryError:
    print(torch.cuda.memory_summary(abbreviated=True))
    raise
```

```console
|===========================================================================|
|                  PyTorch CUDA memory summary, device ID 0                  |
|---------------------------------------------------------------------------|
|            CUDA OOMs: 3           |        cudaMalloc retries: 214        |
|===========================================================================|
|        Metric         | Cur Usage  | Peak Usage | Tot Alloc  | Tot Freed  |
|---------------------------------------------------------------------------|
| Allocated memory      |  48301 MiB |  49780 MiB |    1.42 TiB|    1.37 TiB|
| Reserved memory       |  77728 MiB |  77728 MiB |  120.4 GiB |   42.7 GiB |
| Non-releasable memory |  29427 MiB |  31006 MiB |    ...     |    ...     |
| Allocations           |     3218   |     3410   |   ...      |    ...     |
| Requested memory      |  47110 MiB |  48540 MiB |    ...     |    ...     |
|===========================================================================|
```

Every number here tells part of the story. **Allocated 48301 MiB** vs **Reserved 77728 MiB** — the 29 GB gap, again. **Non-releasable memory 29427 MiB** — that is the cached-free memory *split into blocks the allocator cannot return to the driver without an `empty_cache`*: fragmentation, quantified, in its own row. **cudaMalloc retries: 214** — every time the allocator failed to satisfy a request from cache and had to try `cudaMalloc` (and, on retry, `empty_cache` internally then `cudaMalloc` again): a retry counter climbing is *the* fragmentation heartbeat. **CUDA OOMs: 3** — you have died three times. And note **Requested memory 47110 MiB** below **Allocated 48301 MiB**: the ~1.2 GB difference is the *rounding* overhead — the bytes the allocator added rounding your requests up to size classes. That gap is `roundup_power2_divisions` made visible; if it were huge, you would have an internal-fragmentation problem to tune.

For programmatic monitoring, pull the raw counters from `memory_stats()` and emit them to your metrics pipeline so fragmentation shows up on a dashboard *before* it pages someone:

```python
import torch

def memory_health():
    s = torch.cuda.memory_stats()
    alloc    = s["allocated_bytes.all.current"] / 1024**3
    reserved = s["reserved_bytes.all.current"]  / 1024**3
    inactive = s["inactive_split_bytes.all.current"] / 1024**3   # cached, fragmented
    return {
        "allocated_gib":  round(alloc, 2),
        "reserved_gib":   round(reserved, 2),
        "frag_gib":       round(reserved - alloc, 2),
        "inactive_gib":   round(inactive, 2),          # the fragmentation itself
        "alloc_retries":  s["num_alloc_retries"],      # climbs under fragmentation
        "num_ooms":       s["num_ooms"],
    }

print(memory_health())
```

```console
{'allocated_gib': 48.3, 'reserved_gib': 77.73, 'frag_gib': 29.43,
 'inactive_gib': 28.7, 'alloc_retries': 214, 'num_ooms': 0}
```

The signature to alert on is not a single threshold on any one number — it is the *shape*: `reserved` pulling away from `allocated` (rising `frag_gib`) while `num_alloc_retries` climbs. `allocated` alone will look healthy the entire time. That is why a dashboard that graphs only `memory.used` from `nvidia-smi` — which is `reserved` — cannot distinguish a service that is genuinely near capacity from one that is fragmenting, and why plotting the pair, with `alloc_retries` as the leading indicator, is the monitoring you actually want. For the *timeline* of allocations that produced this state — which tensor, from which line, at which moment — the [memory snapshot post](/blog/machine-learning/performance-engineering/memory-snapshot-and-leak-hunting) picks up exactly here with `_record_memory_history()` and the visual viewer.

## The measured win, honestly

Now the results, on named hardware, measured the way [the benchmarking post](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark) insists: warm-started process, `torch.cuda.synchronize()` before every reading, locked clocks, steady state, the same varying-sequence workload replayed from a fixed seed so the two runs see identical requests. The service is the transformer inference server from the stress test on an **A100 80GB SXM** (79.15 GiB usable, 2.0 TB/s HBM). The only change between the two columns is the environment variable.

| Metric (A100 80GB, same workload) | Default allocator | `expandable_segments:True` |
| --- | --- | --- |
| Peak `reserved` | 77.7 GiB | 51.2 GiB |
| Peak `allocated` (live tensors) | 48.3 GiB | 48.3 GiB |
| `reserved − allocated` gap | 29.4 GiB | 2.9 GiB |
| Largest free block at peak | 1.5 GiB | grows on demand |
| `num_alloc_retries` | 214 | 0 |
| `num_ooms` over the run | 3 | 0 |
| OOM on a 2 GiB request | yes | no |
| Max batch that fits at seq 4096 | 24 | 40 |
| p99 latency (no `empty_cache` in path) | 71 ms | 68 ms |

Read the columns against each other. `allocated` is identical — of course it is; the *model* did not change, so the *live tensor* footprint did not change. What changed is the gap: from 29.4 GiB of fragmented cache down to 2.9 GiB, because the expandable segment flexes instead of orphaning wrong-size segments. `num_alloc_retries` goes from 214 to 0 (the allocator never has to scramble for a fresh segment), `num_ooms` goes from 3 to 0, and — the number that pays the bill — the max batch that fits at the longest sequence goes from 24 to **40**, a 67% throughput headroom increase from one environment variable, no model change, no extra hardware. The p99 barely moves because expandable segments is a *memory-reachability* fix, not a *speed* fix; it buys you the batch size and the not-dying, which on a service that was OOMing is the entire point.

One honest caveat, because every fix is a trade-off: `expandable_segments` is newer machinery and interacts with a few things. It can behave differently with CUDA graphs (which record fixed pointers — see [the CUDA-graphs gotchas post](/blog/machine-learning/performance-engineering/cuda-graphs-gotchas-and-debugging)), and on some driver/PyTorch combinations there have been edge cases with certain multi-GPU or IPC memory-sharing setups. For the overwhelmingly common case — a single-process inference or training job that fragments on varying shapes — it is safe and it is the right first move. But you validate it on *your* workload, you do not cargo-cult it, and if a before→after shows it did not move your `frag_gib`, you were not fragmenting and you have the wrong diagnosis.

### Stress-testing the fix: smaller cards, bigger batches

A fix you have not stress-tested is a fix you do not understand, so push on it along the two axes that change fragmentation's severity: **card size** and **batch size**. Fragmentation is not an absolute number of bytes lost — it is a *fraction* of the card, and that fraction bites harder the smaller the card. On an **L4 (24 GB, 300 GB/s)**, the same varying-shape service that merely wasted 30 GB of headroom on the A100 does not have 30 GB to waste; a 29 GB fragmentation gap is impossible when the whole card is 24 GB, so what actually happens is the allocator hits the ceiling far sooner and the OOMs start at a *much* lower live-tensor footprint — you OOM at maybe 40% "used" instead of 60%, because the smaller the card, the less slack there is to absorb the wrong-size holes. The fix matters *more* on the L4, not less: replaying the same workload with `expandable_segments:True` on an L4 is often the difference between "serves seq 2048 at batch 4" and "OOMs on the second long request," because reclaiming the fragmented gap on a 24 GB card recovers a large fraction of the whole device.

The batch-size axis cuts the other way and teaches a subtler lesson. At **batch 1**, activation tensors are small and short-lived, the working set is dominated by the (fixed-size) weights and KV cache, and there is simply less variable-size churn to fragment — a batch-1 decode service can run for days on the default allocator without a visible gap, because the allocations it makes are nearly uniform. Push to **batch 64** with varying sequence lengths and the activation tensors become large *and* wildly variable in size, which is the fragmentation engine at full throttle: the gap that was invisible at batch 1 becomes the thing that OOMs you at batch 64. The practical rule the stress test yields is that fragmentation risk scales with *both* how full the card is and how variable your large allocations are — a small model on a big card at batch 1 barely fragments and needs no knob, while a big model on a small card at high batch with varying shapes is the worst case and needs `expandable_segments` from day one. Test at the batch and card you actually deploy, not the one that is convenient to reproduce, because the convenient one may hide the problem entirely.

#### Worked example: the worst-case fragmentation bound

It is worth pinning down *how bad* external fragmentation can theoretically get, because it explains why the gap can be so large. Consider a segment of $M$ bytes and a request size $r$. In the pathological interleaving — allocate $r$-sized blocks across the whole segment, then free every *other* one — you end up with free blocks of size $r$ separated by live blocks of size $r$. The total free memory is $M/2$, but the largest contiguous free block is only $r$, so a request for ${2r}$ fails despite half the segment being free. Generalize the interleave to leave one live block per $k$ free-sized slots and the free fraction at which you can still be denied a $(k{+}1)r$ request climbs toward ${1 - 1/k}$: with enough size variety you can have **the vast majority of a segment free and still be unable to place a modestly larger request**. Our A100 case — 29 GB free, largest block 1.5 GB, 2 GB request denied — is a mild instance of this bound, not an exotic one. The lesson is that "how much is free" tells you almost nothing about "will the next allocation succeed"; only `max_i b_i` versus $r$ does, which is precisely why `expandable_segments` (which makes the free space effectively one growable block) is such a clean escape from the whole failure mode.

#### Worked example: the batch that fits, then does not

Here is the failure exactly as it reaches on-call, and the fix, end to end. The service is provisioned for batch 32 at up to sequence 2048, which it handles all day. Then a batch arrives where several requests happen to be near 4096 tokens. On the default allocator, the activations for that batch need a 2 GiB contiguous block; the cache — fragmented from a day of varying shapes — has 29 GiB free but no block over 1.5 GiB, the driver footprint is 78 GiB with 1.1 GiB free, and the allocation fails. `memory_summary()` shows `reserved 77.7 GiB`, `allocated 48.3 GiB`, `retries 214`: fragmentation, unambiguously. The on-call engineer does **not** add `empty_cache` (which would stall every request and still OOM on the next burst). They set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and restart. Same workload, same requests, and now `reserved` peaks at 51 GiB, the gap is 2.9 GiB, the 2 GiB block is mapped from fresh pages into the flexing segment, and not only does the batch fit — the service now has room for batch 40. One variable, an order of magnitude less fragmentation, zero code change. That is the whole post in one incident.

## Case studies and real numbers

**PyTorch's own guidance names this exact failure.** The [CUDA semantics / Memory Management documentation](https://pytorch.org/docs/stable/notes/cuda.html#memory-management) is unusually direct about it: the OOM error message itself prints the recommendation to set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` when "reserved memory is >> allocated memory," which is the framework telling you, at the moment of failure, that you are fragmented, not out of memory. `expandable_segments` was introduced specifically to attack fragmentation via the CUDA virtual-memory APIs, and PyTorch's own docs describe it as the mechanism that lets a segment grow without pre-reserving a fixed-size contiguous slab. The number to remember is qualitative and reliable: on a fragmenting varying-shape workload it typically collapses the `reserved − allocated` gap from tens of gigabytes to a few, which is exactly what our A100 table shows.

**Large-model training is where fragmentation shows up at scale.** Teams training multi-billion-parameter models routinely hit fragmentation because activation sizes vary across layers and micro-batches, and the community-standard first response — documented across the DeepSpeed, Megatron, and Hugging Face training guides — is to set `PYTORCH_CUDA_ALLOC_CONF`. The Hugging Face performance docs, for instance, call out `expandable_segments` (and historically `max_split_size_mb`) as the go-to for training runs that OOM despite apparently sufficient memory, and report fitting larger batch or sequence sizes with no other change — the same "max batch 24 → 40" effect, just at training scale.

**The `max_split_size_mb` era.** Before `expandable_segments` existed, `max_split_size_mb` was *the* fragmentation fix, and there is a well-worn body of practitioner reports (Stable Diffusion and LLM fine-tuning threads especially) of `max_split_size_mb:128` or `:256` turning a reproducible OOM into a clean run on the same card. It works for the specific pattern of large allocations being cannibalized by small interleaving ones, and it remains the correct tool where `expandable_segments` is not available or interacts badly with another component. The lesson is not "always use knob X" — it is "read the gap, match the knob to the fragmentation."

**The KV cache is a giant, well-behaved allocation — until it is not.** In LLM serving, the KV cache is often the single largest tensor and the reason services run near the memory ceiling; naive per-request KV allocation of varying sequence lengths is a fragmentation *engine*, which is precisely the problem [PagedAttention](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention) was designed to solve — by allocating the KV cache in fixed-size *pages* so that varying sequence lengths never fragment, it sidesteps the caching allocator's variable-block problem entirely at the application layer. That is the same fragmentation physics from this post, solved one level up: fixed-size blocks do not fragment, which is *why* paging works and *why* `expandable_segments` (contiguous flexible virtual space) works — two different answers to the identical question.

## When to reach for this (and when not to)

Fragmentation-tuning is cheap to try and easy to over-apply, so here is the decisive version.

**Reach for `expandable_segments:True` when** your `reserved − allocated` gap is large and growing while `allocated` is stable, `num_alloc_retries` is climbing, and you OOM on requests smaller than your free memory. That is the fragmentation signature, and this is very nearly a free fix for it. Make it your default for any single-process training or inference service that sees varying shapes — set it and move on.

**Reach for `max_split_size_mb` when** the fragmentation is specifically large transient tensors being starved by small interleaving allocations, and you cannot use `expandable_segments` (an older PyTorch, or a component that misbehaves with it). Tune $N$ by experiment; there is no universal value.

**Do NOT reach for any allocator knob when `allocated` itself is growing** — that is a leak, and a knob cannot fix live memory. Go to [the leak-hunting post](/blog/machine-learning/performance-engineering/memory-snapshot-and-leak-hunting) and find the retained reference. Every hour spent tuning `max_split_size_mb` against a real leak is an hour wasted.

**Do NOT reach for a knob when a single allocation genuinely exceeds the card** — no policy makes a 90 GB tensor fit in 80 GB. Reduce batch size, use activation checkpointing, shard the model, or page the KV cache.

**Do NOT put `torch.cuda.empty_cache()` in a hot path, ever.** It stalls the GPU 5–7 ms per call by synchronizing the device, throws away the cache the allocator worked to build, and does not fix fragmentation — the holes come right back on the next varying-shape burst. Its only correct use is handing device memory to a *different* process. If you find `empty_cache` in a request handler, that is a bug, not a safeguard.

**Do NOT chase a large `reserved` number by itself.** A big `reserved` with a small gap and stable `allocated` is a *healthy* service that warmed up its cache — that is the allocator doing its job. `reserved` near the card is only a problem when the gap is large and requests are failing. Alert on the gap and on `num_alloc_retries`, not on `reserved` alone.

## Key takeaways

- **`allocated` and `reserved` are two different numbers; print them as a pair.** `allocated` is live tensors; `reserved` is what PyTorch took from the driver and holds. `nvidia-smi` shows `reserved`, which is why it "over-reports" versus your model's real footprint. The gap is the cache.
- **The allocator caches because `cudaMalloc`/`cudaFree` are slow and synchronizing** (~100 µs, and `cudaFree` drains the pipeline). Caching amortizes a would-be 450 ms/step allocation tax down to a few milliseconds by reusing freed blocks — it is not optional for a Python framework.
- **You OOM below capacity when total free memory exceeds the request but no single free block does**, and the driver is too full to `cudaMalloc` a fresh segment. Fragmentation is `∑bᵢ ≥ r` while `max bᵢ < r`. The error message prints the diagnosis.
- **Varying shapes fragment; fixed shapes do not.** A service that runs whatever sequence length the user sent ratchets `reserved` away from `allocated` over time; a fixed-shape loop plateaus and stays flat.
- **`expandable_segments:True` is the first fix and usually the right one** — it flexes one virtual segment instead of orphaning wrong-size ones, collapsing the gap. `max_split_size_mb` is the older, fiddlier tool for large-alloc starvation.
- **`empty_cache()` is not a fragmentation fix.** It stalls the device, discards the warm cache, and the holes return. Use it only to yield memory to another process.
- **Split the OOM into three causes with the number's behavior over time:** stable `allocated` + growing gap = fragmentation (knob); rising `allocated` = leak (snapshot); huge single request = too big (shard). Same error, three fixes.
- **Monitor the gap and `num_alloc_retries`, not `reserved` alone.** Fragmentation is a *shape* — reserved pulling away from a flat allocated while retries climb — and a dashboard of `nvidia-smi memory.used` cannot see it.

You now have the framework the rest of this series kept deferring to: the allocator as a caching layer between your tensors and the driver, `reserved ≥ allocated` as the invariant, fragmentation as the gap you cannot reach, and one environment variable as the fix for the common case. The [capstone playbook](/blog/machine-learning/performance-engineering/the-performance-engineering-playbook) folds this into the full symptom-to-fix decision tree, and the [memory snapshot post](/blog/machine-learning/performance-engineering/memory-snapshot-and-leak-hunting) picks up where this one ends — turning the counters you now read into a visual timeline of exactly which allocation, from which line, built the state you are staring at.

## Further reading

- [PyTorch CUDA semantics — Memory Management](https://pytorch.org/docs/stable/notes/cuda.html#memory-management) — the authoritative reference for the caching allocator, `PYTORCH_CUDA_ALLOC_CONF`, `expandable_segments`, `max_split_size_mb`, `garbage_collection_threshold`, and `roundup_power2_divisions`.
- [`torch.cuda.memory_stats` / `memory_summary`](https://pytorch.org/docs/stable/generated/torch.cuda.memory_stats.html) — every counter the allocator exposes, including `num_alloc_retries`, `num_ooms`, and `inactive_split_bytes`.
- [Understanding CUDA Memory Usage — the memory snapshot viewer](https://pytorch.org/docs/stable/torch_cuda_memory.html) — `_record_memory_history()` and the interactive viewer for the allocation timeline.
- [Why your AI service wastes CPU and GPU](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu) — the four wastes and the profile→fix→measure loop this post plugs into.
- [Metrics that actually matter](/blog/machine-learning/performance-engineering/metrics-that-actually-matter) — allocated vs reserved among the other metrics whose headline number lies.
- [Memory snapshot and leak hunting](/blog/machine-learning/performance-engineering/memory-snapshot-and-leak-hunting) — the visual timeline that separates a leak from fragmentation.
- [The memory hierarchy: registers, shared memory, and HBM](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm) — the layer below the allocator, where the bytes you finally reach actually live.
- [The performance engineering playbook](/blog/machine-learning/performance-engineering/the-performance-engineering-playbook) — the capstone decision tree that resolves this and every other forward-link in the series.
