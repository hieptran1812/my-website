---
title: "Implementing a two-cache engine: KV blocks plus recurrent state"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "Build the allocator a hybrid model actually needs — a paged KV pool that grows with context sitting beside a fixed-size state pool that never does — and derive the page size that keeps one physical arena from stranding thirty-five gigabytes."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "kv-cache",
    "mamba",
    "state-space-models",
    "hybrid-models",
    "memory-management",
    "batching",
    "pytorch",
    "gpu",
    "ml-systems",
    "vllm",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 54
---

There is a specific way the first hybrid port goes wrong, and it is worth naming before we write any code, because it looks like success for about a day.

You take `nanoserve`, which already has a paged KV cache, a block table, a free pool and a scheduler that admits requests by counting blocks. You add Mamba-2 layers. The forward pass needs somewhere to keep the recurrent state, so you do the obvious thing: allocate `torch.zeros(max_num_seqs, n_layers, ...)` at startup, index it by the request's position in the batch, and move on. It works. Your smoke test passes. Then two things happen in production. First, a request finishes and its slot gets recycled, and the next request that lands in that slot starts generating with somebody else's conversation compressed into its state — no crash, no traceback, just answers that are subtly, confidently wrong. Second, you discover that on a workload of short prompts your server saturates at 320 concurrent requests while `nvidia-smi` reports 35 GiB of the cache pool sitting completely idle, because the thing that ran out was not blocks.

Both failures have the same root. Your engine now manages **two caches with two different index formats, two different growth laws, and two different lifetimes**, and it has an allocator for exactly one of them. The vLLM team put the index-format half of this precisely in their post on [disaggregated serving for hybrid SSM models (2026-04-21)](https://vllm.ai/blog/2026-04-21-hybrid-ssm-disagg): full-attention layers use a uniform per-token KV layout, while state-space layers hold a fixed-size convolution state plus a temporal state that has *no per-token dimension at all*, so a single descriptor format cannot address both. That is not a transfer-layer detail. It is the shape of the whole memory manager.

![Stacked breakdown of an eighty gigabyte GPU showing weights, workspace, and a cache budget split into a growing KV region and a fixed state region](/imgs/blogs/implementing-a-two-cache-engine-kv-blocks-plus-recurrent-state-1.webp)

Figure 1 is the object we are going to build, drawn as a budget. An H100 80GB SXM gives you about 74.5 GiB of usable device memory. An 8B model in bf16 takes 14.9 GiB of it. Activations, CUDA context and kernel workspace take roughly 4 GiB more. What is left — 55.6 GiB — is the cache budget, and on a hybrid it does not go into one pool. It splits into a KV region whose occupancy tracks how long the conversations are, and a state region whose occupancy tracks only *how many* conversations there are. Two regions, two counters, two ways to run out. Every design decision in this post is about where to put the line between them, or how to avoid drawing it at all.

By the end you will have written `nanoserve/state.py` — a `StateCache` that preallocates fixed-shape conv and SSM tensors and hands out integer slots — plus the `HybridRequest` object that carries both a block table and a state slot, the layer router that sends each layer to the right cache, an admission predicate that is now two inequalities instead of one, and a capacity planner that tells you, before you start the server, exactly how many requests fit and which pool will run out first. Then we will throw the split-pool design away and derive the better one: a single arena of 2 MiB pages that both caches draw from, which on the same GPU serves 862 concurrent 1k-token requests where the tuned split served 320. That unified design is the thing vLLM describes in one sentence in its [Qwen3-Next post (2025-09-11)](https://vllm.ai/blog/2025-09-11-qwen3-next), and we are going to derive why the sentence is true.

Two promises, inherited from [the series introduction](/blog/machine-learning/inference-engineering/what-inference-engineering-is) and binding here as everywhere. **I have no GPU and I have run nothing.** Every figure below is derived from arithmetic I show you, cited from a paper, model card or vendor post with a link, or framed as something you should reproduce yourself with an expected range on named hardware. Results tables carry a `Source` column. **And I do not name a model whose layer census I could not check against a primary source.** Several plausible candidates are missing for that reason alone.

This post assumes you have read [hybrid models and the end of the KV-cache assumption](/blog/machine-learning/inference-engineering/hybrid-models-and-the-end-of-the-kv-cache-assumption), which derives the two-term memory law and builds the shape model we will import. That post answered "how much memory does a hybrid need?". This one answers "where does the engine put it, and who frees it?".

---

## 1. Two caches, two index formats

Start with the thing you already have. In [the paged KV cache post](/blog/machine-learning/inference-engineering/paged-kv-cache-implementing-blocks-and-a-block-table) we built an allocator whose entire vocabulary is the integer. A physical block is an id in `[0, num_blocks)`. A request holds a `block_table: list[int]`. A token at logical position `pos` lives at flat slot `block_table[pos // 16] * 16 + pos % 16`. The address of a piece of KV is therefore a pair — *which block, which offset inside it* — and every kernel, every eviction policy, every prefix-sharing hash and every transfer descriptor in the engine is written against that pair.

A recurrent layer's state has no second component. There is no offset, because there is no position. Request 7's Mamba-2 state at layer 12 is one tensor of shape `(n_heads, head_dim, d_state)`, and its address is the single integer 7 plus the single integer 12. It was that shape when the request had processed one token and it will be that shape when it has processed 128,000, because every step overwrites it in place rather than appending to it.

That difference propagates into six places, and it is worth listing them all before we start writing code, because each one is a place where a naive port silently does the wrong thing:

1. **Allocation unit.** KV is allocated in blocks of 16 tokens, many per request, on demand. State is allocated as one whole slot, exactly one per request, at admission.
2. **Growth.** The KV block count grows monotonically as the request generates. The state slot count per request is constant at 1, forever.
3. **Lifetime.** A KV block can be freed the moment the tokens it holds fall out of the window — sliding-window layers do exactly this. A state slot cannot be partially freed at all, because there is no part of it that corresponds to old tokens.
4. **Sharing.** Two requests with a common prefix can point their block tables at the same physical KV blocks and bump a refcount. Two requests cannot share a state slot, because the first decode step of either one mutates it.
5. **Reclaim under pressure.** Evicting a request's KV blocks is recoverable: recompute the prefill and you get the same bytes back. Evicting a request's state is recoverable only by replaying every token from the beginning, because the state at step $t$ is a function of the entire history and there is no checkpoint.
6. **Zeroing.** A freshly popped KV block will be fully overwritten by the write kernel before anything reads it, so its previous contents are irrelevant. A freshly popped state slot is *read* by the first decode step before it is written. Stale bytes there are not garbage — they are a previous user's conversation.

Point 6 is failure number one from the opening paragraph, and it is the reason your smoke test passed: a single request against a fresh server never recycles a slot.

None of these six is exotic. Each is one line of code. The problem is that if you carry the mental habits of a block allocator into the state allocator, you get all six wrong in the same direction, and only two of them fail loudly.

---

## 2. What the state tensor actually is, in bytes

Before allocating anything we need the exact shapes, because the whole capacity argument rests on them. Take `nvidia/Nemotron-H-8B-Base-8K`, whose config we parsed in the previous post: 52 layers, of which 4 are self-attention, 24 are Mamba-2 and 24 are FFN. The attention layers use 8 KV heads of 128 dimensions. The Mamba-2 layers have `d_inner` 8192, `d_state` 128, 8 groups, a convolution kernel width of 4, and 128 heads of dimension 64. Everything in bf16, so 2 bytes per element.

A Mamba-2 layer carries two tensors per request.

The **temporal state** — the thing that plays the role attention's KV cache plays — is one matrix per head:

$$
s_{\text{ssm}} = H_m \cdot d_p \cdot N \cdot b = 128 \cdot 64 \cdot 128 \cdot 2 = 2{,}097{,}152 \text{ bytes}
$$

Exactly 2.00 MiB. Note what is *not* in that product: sequence length. There is no $S$ anywhere on the right-hand side, which is the entire point.

The **convolution state** is the short causal-conv window that Mamba-2 runs before the recurrence. It keeps the last $k-1$ positions of the projected input across a channel dimension that is the concatenation of three sub-projections — the main input path $x$ of width `d_inner`, plus the $B$ and $C$ projections of width `n_groups * d_state` each:

$$
C_{\text{conv}} = d_{\text{inner}} + 2 \cdot G \cdot N = 8192 + 2 \cdot 8 \cdot 128 = 10{,}240
$$

$$
s_{\text{conv}} = (k-1) \cdot C_{\text{conv}} \cdot b = 3 \cdot 10{,}240 \cdot 2 = 61{,}440 \text{ bytes}
$$

60.0 KiB. That three-way decomposition is not a detail I invented for tidiness — it is load-bearing in production. The vLLM disaggregation post describes storing the conv state as its three sub-projections separately, and the temporal state in a layout they call `DS`, shaped as `(dim, state_len)`, specifically so that each decode rank under tensor parallelism can read only its own $1/\text{TP}$ slice over RDMA and never transfer the padding. They report that avoiding the padding transfer saves roughly 50 MB per request on a bf16 setup, per their Figure 1. Hold onto that number; it is the same order of magnitude as an entire Nemotron-H-8B request state, which tells you how much of the transfer cost in a hybrid is bookkeeping rather than data.

So one recurrent layer costs $2{,}097{,}152 + 61{,}440 = 2{,}158{,}592$ bytes, or 2.0586 MiB. Twenty-four of them:

$$
\sigma = 24 \cdot 2{,}158{,}592 = 51{,}806{,}208 \text{ bytes} = 49.41 \text{ MiB}
$$

And the attention side, from the standard per-token law $\kappa = 2 \cdot L_{\text{attn}} \cdot H_{kv} \cdot d_h \cdot b$ derived in [the memory math of the KV cache](/blog/machine-learning/inference-engineering/the-memory-math-of-the-kv-cache):

$$
\kappa = 2 \cdot 4 \cdot 8 \cdot 128 \cdot 2 = 16{,}384 \text{ bytes/token} = 16 \text{ KiB/token}
$$

Total per-request memory is the two-term law:

$$
M(S) = \sigma + \kappa S
$$

![Comparison table of cache kinds showing how each is addressed, its byte size, whether it grows with sequence length, and when it is freed](/imgs/blogs/implementing-a-two-cache-engine-kv-blocks-plus-recurrent-state-2.webp)

Figure 2 is the contract sheet for the rest of the post. Read the "addressed by" column first: a KV block needs two integers, a state needs one. Then read the "lifetime" column, because that is the one your allocator gets wrong. A KV block table shrinks — a sliding-window layer drops blocks off the front, a preemption drops the whole table, a finished request returns all of them. A state slot has exactly two events in its life: allocated, and freed. Nothing in between touches it.

#### Worked example: one Nemotron-H-8B request, byte by byte

An 8,192-token conversation on this model holds:

- 24 SSM states at 2,097,152 bytes = 50,331,648 bytes (48.00 MiB)
- 24 conv states at 61,440 bytes = 1,474,560 bytes (1.41 MiB)
- KV for 8,192 tokens at 16,384 bytes/token = 134,217,728 bytes (128.00 MiB)

Total 186,023,936 bytes, or 177.41 MiB. The state is 27.8% of it. Now run the same arithmetic at 1,024 tokens: state 49.41 MiB, KV 16.00 MiB, total 65.41 MiB — the state is **75.5%** of the request's memory. And at 131,072 tokens: state 49.41 MiB, KV 2,048 MiB, total 2,097.41 MiB — the state is **2.4%**.

That single sweep is the reason a fixed split between two regions is a trap. The correct ratio between them is not a property of the model. It is a property of your traffic, and it moves by a factor of thirty across the context lengths a normal chat endpoint sees in a day.

| Context $S$ | State bytes | KV bytes | Total | State share | Source |
| --- | --- | --- | --- | --- | --- |
| 1,024 | 49.41 MiB | 16.00 MiB | 65.41 MiB | 75.5% | derived |
| 4,096 | 49.41 MiB | 64.00 MiB | 113.41 MiB | 43.6% | derived |
| 8,192 | 49.41 MiB | 128.00 MiB | 177.41 MiB | 27.8% | derived |
| 32,768 | 49.41 MiB | 512.00 MiB | 561.41 MiB | 8.8% | derived |
| 131,072 | 49.41 MiB | 2,048.00 MiB | 2,097.41 MiB | 2.4% | derived |

---

## 3. Two allocators, two lifetimes

Now watch a single request move through the engine, and pay attention to which pool it touches at each moment.

![Timeline of a single request showing pages taken at admission, pages added during prefill and decode, and both caches released at completion](/imgs/blogs/implementing-a-two-cache-engine-kv-blocks-plus-recurrent-state-3.webp)

Figure 3 walks it. At admission the request takes **one** state slot and **zero** KV blocks, because nothing has been computed yet. Prefill of an 8k prompt then takes 512 blocks in one shot and writes the state 8,192 times — once per token, in place, each write overwriting the last. Every decode step after that writes the state again, exactly once, and appends one token's worth of KV, which turns into a new block every sixteenth step. At EOS both are released, but through different calls into different structures.

The asymmetry is easier to feel in motion than in a table, so here it is running:

<figure class="blog-anim">
<svg viewBox="0 0 660 250" role="img" aria-label="KV pages accumulate one after another on the left while a single fixed state slot on the right stays the same size and is rewritten each step" style="width:100%;height:auto;max-width:820px">
<style>
.h1-lbl{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.h1-sub{font:400 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.h1-box{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.h1-pg{fill:var(--accent,#6366f1);opacity:.85}
.h1-st{fill:none;stroke:var(--accent,#6366f1);stroke-width:2.5}
.h1-sweep{fill:var(--accent,#6366f1);opacity:.22}
@keyframes h1-p1{0%,4%{opacity:0}9%,86%{opacity:.85}93%,100%{opacity:0}}
@keyframes h1-p2{0%,16%{opacity:0}21%,86%{opacity:.85}93%,100%{opacity:0}}
@keyframes h1-p3{0%,28%{opacity:0}33%,86%{opacity:.85}93%,100%{opacity:0}}
@keyframes h1-p4{0%,40%{opacity:0}45%,86%{opacity:.85}93%,100%{opacity:0}}
@keyframes h1-p5{0%,52%{opacity:0}57%,86%{opacity:.85}93%,100%{opacity:0}}
@keyframes h1-p6{0%,64%{opacity:0}69%,86%{opacity:.85}93%,100%{opacity:0}}
@keyframes h1-hold{0%,3%{opacity:0}8%,88%{opacity:1}94%,100%{opacity:0}}
@keyframes h1-rewrite{0%{transform:translateX(0)}100%{transform:translateX(126px)}}
.h1-a1{animation:h1-p1 14s ease-in-out infinite}
.h1-a2{animation:h1-p2 14s ease-in-out infinite}
.h1-a3{animation:h1-p3 14s ease-in-out infinite}
.h1-a4{animation:h1-p4 14s ease-in-out infinite}
.h1-a5{animation:h1-p5 14s ease-in-out infinite}
.h1-a6{animation:h1-p6 14s ease-in-out infinite}
.h1-hd{animation:h1-hold 14s ease-in-out infinite}
.h1-rw{animation:h1-rewrite 2.4s steps(3,end) infinite}
@media (prefers-reduced-motion:reduce){.h1-a1,.h1-a2,.h1-a3,.h1-a4,.h1-a5,.h1-a6,.h1-hd{animation:none;opacity:.85}.h1-rw{animation:none}}
</style>
<text class="h1-lbl" x="24" y="30">KV pages</text>
<text class="h1-sub" x="24" y="48">one more every 512 tokens</text>
<text class="h1-lbl" x="410" y="30">state slot</text>
<text class="h1-sub" x="410" y="48">rewritten in place, never resized</text>
<rect class="h1-box" x="20" y="66" width="340" height="120" rx="10"/>
<rect class="h1-pg h1-a1" x="34"  y="82" width="46" height="88" rx="6"/>
<rect class="h1-pg h1-a2" x="88"  y="82" width="46" height="88" rx="6"/>
<rect class="h1-pg h1-a3" x="142" y="82" width="46" height="88" rx="6"/>
<rect class="h1-pg h1-a4" x="196" y="82" width="46" height="88" rx="6"/>
<rect class="h1-pg h1-a5" x="250" y="82" width="46" height="88" rx="6"/>
<rect class="h1-pg h1-a6" x="304" y="82" width="46" height="88" rx="6"/>
<rect class="h1-box" x="400" y="66" width="230" height="120" rx="10"/>
<g class="h1-hd">
<rect class="h1-st" x="414" y="82" width="202" height="88" rx="6"/>
<rect class="h1-sweep h1-rw" x="418" y="86" width="64" height="80" rx="4"/>
<text class="h1-lbl" x="515" y="132" text-anchor="middle">49.4 MiB</text>
</g>
<text class="h1-sub" x="24" y="212">16 KiB per token, freed page by page</text>
<text class="h1-sub" x="400" y="212">constant at token 1 and token 128,000</text>
</svg>
<figcaption>The left pool grows one page at a time as the conversation lengthens; the right slot is claimed once, overwritten every decode step, and released whole.</figcaption>
</figure>

Two allocators, two lifetimes. The left-hand pool is a *counter that goes up*. The right-hand pool is a *set of slots that are either taken or not*. If you have ever written a fixed-slot session table for a network server, the right-hand structure is that, and it has that structure's characteristic failure mode: a hard ceiling that no amount of free memory elsewhere can relieve.

That ceiling has a name in your metrics, and you should decide now what to call it, because a scheduler that reports "out of memory" for both conditions is a scheduler you cannot debug at 3am. `nanoserve` will report them separately: `kv_pages_exhausted` and `state_slots_exhausted`.

---

## 4. `nanoserve/state.py`: the fixed-size allocator

Here is the file. It is deliberately boring, and boring is the goal: the interesting decisions are all in the constructor and in `allocate`.

```python
# nanoserve/state.py
"""Fixed-size recurrent state allocation: preallocated tensors plus a slot pool."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch


class OutOfStateSlots(RuntimeError):
    """Raised when every recurrent state slot is occupied."""

    def __init__(self, capacity: int):
        super().__init__(f"all {capacity} state slots in use")
        self.capacity = capacity


@dataclass(frozen=True)
class Mamba2StateSpec:
    """Everything needed to size one recurrent layer's per-request state."""

    n_heads: int          # mamba heads, e.g. 128
    head_dim: int         # per-head channel width, e.g. 64
    d_state: int          # SSM state dim N, e.g. 128
    d_inner: int          # e.g. 8192
    n_groups: int         # e.g. 8
    conv_kernel: int      # e.g. 4
    dtype: torch.dtype = torch.bfloat16

    @property
    def conv_channels(self) -> int:
        """x, B and C sub-projections concatenated along the channel axis."""
        return self.d_inner + 2 * self.n_groups * self.d_state

    @property
    def conv_width(self) -> int:
        return self.conv_kernel - 1

    def bytes_per_layer(self) -> int:
        el = torch.finfo(self.dtype).bits // 8
        ssm = self.n_heads * self.head_dim * self.d_state * el
        conv = self.conv_channels * self.conv_width * el
        return ssm + conv
```

Nothing surprising yet — this is the shape model from the previous post, restated so the allocator can use it directly. Now the pool itself.

```python
# nanoserve/state.py (continued)


class StateCache:
    """A fixed number of interchangeable recurrent-state slots.

    Unlike the KV pool, capacity here is a count of REQUESTS, not a count of
    tokens. Every slot is the same size and that size never changes, so the
    entire allocator is a free-slot queue over two preallocated tensors.
    """

    def __init__(self, spec: Mamba2StateSpec, recurrent_layers: list[int],
                 n_slots: int, device: str = "cuda"):
        self.spec = spec
        self.layers = list(recurrent_layers)          # global layer indices
        self.layer_pos = {l: i for i, l in enumerate(self.layers)}
        self.n_slots = n_slots
        self.device = device

        n_rec = len(self.layers)
        # [recurrent_layer, slot, heads, head_dim, d_state]
        self.ssm = torch.zeros(
            n_rec, n_slots, spec.n_heads, spec.head_dim, spec.d_state,
            dtype=spec.dtype, device=device,
        )
        # [recurrent_layer, slot, conv_channels, conv_width]
        self.conv = torch.zeros(
            n_rec, n_slots, spec.conv_channels, spec.conv_width,
            dtype=spec.dtype, device=device,
        )
        self._free: deque[int] = deque(range(n_slots))

    @property
    def num_free(self) -> int:
        return len(self._free)

    def nbytes(self) -> int:
        return (self.ssm.numel() * self.ssm.element_size()
                + self.conv.numel() * self.conv.element_size())

    def bytes_per_slot(self) -> int:
        return self.nbytes() // self.n_slots

    def allocate(self) -> int:
        """Claim a slot AND clear it. The clearing is not optional."""
        if not self._free:
            raise OutOfStateSlots(self.n_slots)
        slot = self._free.popleft()
        self.ssm[:, slot].zero_()
        self.conv[:, slot].zero_()
        return slot

    def free(self, slot: int) -> None:
        self._free.append(slot)

    def views(self, layer: int, slot: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The (ssm, conv) tensors a recurrent kernel mutates in place."""
        i = self.layer_pos[layer]
        return self.ssm[i, slot], self.conv[i, slot]
```

Four things in that class are decisions, not defaults.

**`allocate` zeroes.** This is the fix for failure number one, and it belongs in `allocate` rather than in `free` for a reason worth stating: zeroing on free is a cost you pay for every request, including the ones whose slot is never reused before the server shuts down, and it happens on the completion path where you are trying to get a response out the door. Zeroing on allocate happens on the admission path, where you are already about to run a prefill that costs a thousand times more. The cost itself is small — writing 51,806,208 bytes at the H100 SXM's published 3.35 TB/s of HBM3 bandwidth takes about 15.5 µs, derived, and even at 100 admissions per second that is 0.16% of one second of GPU time.

There is a faster option, and you should know it exists before you decide you need it: instead of zeroing, pass a `first_step=True` flag into the recurrent kernel so that step zero *writes* the state rather than accumulating into it. That eliminates the memset entirely. It also means every kernel you ever write has to honour the flag, and a single kernel that forgets it reintroduces the cross-request contamination bug in a form that no test without slot recycling will catch. Zero on allocate. Take the 15 µs.

**Slots are a `deque`, not a bitmask or a counter.** You need the actual identity of the free slot, not just a count, and you need `free` to be $O(1)$ without a scan. FIFO rather than LIFO here is not a caching policy the way it was for KV blocks — nothing useful survives in a freed state slot — but it does spread wear evenly and, more usefully, it makes a use-after-free bug reproducible rather than intermittent, because a slot does not come straight back to the next request.

**The layer axis is compacted.** `self.ssm` has one entry per *recurrent* layer, not per model layer. On Nemotron-H that is 24 rows instead of 52, and `layer_pos` does the translation. Allocating 52 rows and leaving 28 of them empty would waste 28/52 of a 15 GiB tensor, which is 8 GiB of nothing.

**Slot comes before the head axes, layer comes first.** The layout `[layer, slot, heads, head_dim, d_state]` means one request's state for one layer is contiguous, which is what the kernel wants, and it means all slots for one layer are adjacent, which is what a batched kernel wants when it gathers a batch of slots. The alternative — slot-major — would make a single request's 24 layers contiguous, which is what a *swap* wants. You cannot have both; pick the one your hot path uses and eat the strided copy on the cold path.

Let us see the sizes.

```python
NEMOTRON_H_8B_MAMBA = Mamba2StateSpec(
    n_heads=128, head_dim=64, d_state=128,
    d_inner=8192, n_groups=8, conv_kernel=4,
)

recurrent = [i for i, c in enumerate("M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-")
             if c == "M"]

print("recurrent layers :", len(recurrent))
print("bytes per layer  :", NEMOTRON_H_8B_MAMBA.bytes_per_layer())
print("bytes per request:", len(recurrent) * NEMOTRON_H_8B_MAMBA.bytes_per_layer())

cache = StateCache(NEMOTRON_H_8B_MAMBA, recurrent, n_slots=320, device="cpu")
print("ssm tensor       :", tuple(cache.ssm.shape))
print("conv tensor      :", tuple(cache.conv.shape))
print("total GiB        :", round(cache.nbytes() / 2**30, 3))
```

```console
recurrent layers : 24
bytes per layer  : 2158592
bytes per request: 51806208
ssm tensor       : (24, 320, 128, 64, 128)
conv tensor      : (24, 320, 10240, 3)
total GiB        : 15.439
```

Fifteen and a half gigabytes, allocated at startup, before a single request has arrived, and it does not shrink when the server is idle. That number is the whole argument of sections 7 and 8, and it is worth sitting with: on a 24 GB RTX 4090 running the same model, 15.4 GiB of state slots would not fit alongside 14.9 GiB of weights. The slot count is not a tuning knob you set to "big enough". It is a hard commitment of memory made before you know your traffic.

### The recycled-slot bug, made visible

Since the zeroing is the part everyone skips, here is the test that catches it. It needs no GPU.

```python
def test_slot_is_clean_on_reuse():
    cache = StateCache(NEMOTRON_H_8B_MAMBA, [0, 1], n_slots=2, device="cpu")

    a = cache.allocate()
    ssm_a, conv_a = cache.views(layer=0, slot=a)
    ssm_a.fill_(0.5)                    # request A "decodes" for a while
    conv_a.fill_(0.25)
    cache.free(a)

    b = cache.allocate()                # request B lands on the same slot
    ssm_b, _ = cache.views(layer=0, slot=b)
    assert b == a, "this test only means something if the slot is reused"
    assert ssm_b.abs().max().item() == 0.0, "state slot carried over stale bytes"
```

Delete the two `zero_()` calls from `allocate` and that assertion fires. Keep them and it passes. It is four lines and it is the difference between a correct engine and one that occasionally answers a question about last week's invoice with somebody else's Python traceback.

---

## 5. The request object, and the layer router

A hybrid request is not a `PagedSequence` with an extra field bolted on; it is a pair. Making that explicit in the type is the cheapest way to stop the rest of the engine from assuming there is only one cache.

```python
# nanoserve/hybrid_request.py
from __future__ import annotations

from dataclasses import dataclass

from nanoserve.blocks import BlockAllocator, PagedSequence
from nanoserve.state import StateCache, OutOfStateSlots


@dataclass
class HybridRequest:
    """One request's claim on BOTH caches."""

    req_id: str
    kv: PagedSequence          # block table, grows with tokens
    state_slot: int            # one integer, constant for the request's life

    @property
    def num_tokens(self) -> int:
        return self.kv.num_tokens

    def append(self, n_new: int = 1) -> list[int]:
        """Only the KV side has anything to append. The state is mutated
        in place by the recurrent kernels, so it has nothing to reserve."""
        return self.kv.append(n_new)
```

The comment on `append` is the API's whole personality. There is no `state.append`. There will never be a `state.append`. Every time someone asks why the state does not have one, the answer is section 1, point 2.

![Two-column comparison of a naive design that stores the recurrent state as fake KV blocks against a design with a separate slot allocator](/imgs/blogs/implementing-a-two-cache-engine-kv-blocks-plus-recurrent-state-4.webp)

Figure 4 shows the design you should not build, next to the one you should. The tempting shortcut is to keep exactly one allocator by expressing the state as KV blocks — 49.41 MiB is 198 blocks of 256 KiB, so "just" hand the request 198 extra blocks and pretend. It fails on the third row. Anything in your engine that shrinks a block table — a sliding-window layer dropping stale blocks, a preemption path freeing half a request, a bug in `blocks_needed_for` — will happily return blocks that were holding a recurrent state, and the corruption surfaces as bad tokens rather than as an exception. The unit is also wrong by two orders of magnitude: a 16-token block is 1.2% of one state, so the "state" is a run of 198 blocks that must stay together and must never be individually reclaimed, which is precisely the constraint a block allocator exists to avoid.

Now the manager that owns both pools.

```python
# nanoserve/hybrid_request.py (continued)


class HybridCacheManager:
    """Owns the KV block pool and the recurrent state pool together."""

    def __init__(self, blocks: BlockAllocator, state: StateCache):
        self.blocks = blocks
        self.state = state
        self.live: dict[str, HybridRequest] = {}
        # Why a request was refused, so the scheduler can report a cause.
        self.refused = {"kv_pages_exhausted": 0, "state_slots_exhausted": 0}

    def can_admit(self, prompt_len: int) -> tuple[bool, str]:
        """TWO inequalities now, and the caller needs to know which failed."""
        need = -(-prompt_len // self.blocks.block_size)
        if self.state.num_free < 1:
            return False, "state_slots_exhausted"
        if not self.blocks.can_allocate(need):
            return False, "kv_pages_exhausted"
        return True, "ok"

    def admit(self, req_id: str, prompt_len: int) -> HybridRequest:
        ok, why = self.can_admit(prompt_len)
        if not ok:
            self.refused[why] += 1
            raise OutOfStateSlots(self.state.n_slots) if why.startswith("state") \
                else RuntimeError(why)
        slot = self.state.allocate()             # zeroed here
        seq = PagedSequence(req_id=req_id, allocator=self.blocks)
        seq.append(prompt_len)
        req = HybridRequest(req_id=req_id, kv=seq, state_slot=slot)
        self.live[req_id] = req
        return req

    def release(self, req_id: str) -> None:
        req = self.live.pop(req_id)
        req.kv.release()                          # blocks back to the queue
        self.state.free(req.state_slot)           # slot back to the deque

    def describe(self) -> str:
        b, s = self.blocks, self.state
        return (f"kv {b.num_used}/{b.num_blocks} blocks used  "
                f"({b.num_used * b.block_size:,} token slots)  |  "
                f"state {s.n_slots - s.num_free}/{s.n_slots} slots used")
```

Note the ordering inside `can_admit`: the slot check comes first. That is not arbitrary. Checking slots first means the cheap, exact, integer-comparison constraint short-circuits before the block arithmetic, and — more importantly — it means that when both constraints fail, the cause you report is the one that is *structurally* harder to relieve. You can free KV blocks by preempting somebody. Freeing a state slot requires an entire request to finish or be evicted wholesale.

### Routing each layer to its own cache

The forward pass now has to ask, per layer, which cache it is talking to. Build that table once at load time rather than branching on a string in the inner loop.

```python
# nanoserve/router.py
from enum import Enum


class Mixer(str, Enum):
    ATTENTION = "attention"
    RECURRENT = "recurrent"
    NONE = "none"          # FFN / MoE: nothing persists across positions


class LayerRouter:
    """Which cache each layer talks to. Built once, read every step."""

    def __init__(self, pattern: str):
        table = {"*": Mixer.ATTENTION, "M": Mixer.RECURRENT, "-": Mixer.NONE}
        self.mixers = [table[c] for c in pattern]
        self.attention_layers = [i for i, m in enumerate(self.mixers)
                                 if m is Mixer.ATTENTION]
        self.recurrent_layers = [i for i, m in enumerate(self.mixers)
                                 if m is Mixer.RECURRENT]

    def __len__(self) -> int:
        return len(self.mixers)
```

And the dispatch itself, which is the smallest piece of code in this post and the one that most changes the engine's shape:

```python
# nanoserve/model_hybrid.py
import torch


@torch.inference_mode()
def hybrid_forward(model, router, mgr, req, hidden, positions):
    """One step for one request. Batching is elided to keep the dispatch clear."""
    slots = torch.tensor(req.append(hidden.shape[0]), dtype=torch.long,
                         device=hidden.device)

    for layer_idx, block in enumerate(model.layers):
        mixer = router.mixers[layer_idx]

        if mixer is Mixer.ATTENTION:
            k, v = block.project_kv(hidden)
            write_kv(mgr.kv_store, layer_idx, k, v, slots)          # paged path
            hidden = hidden + block.attend(hidden, mgr.kv_store,
                                           layer_idx, req.kv.block_table)

        elif mixer is Mixer.RECURRENT:
            ssm, conv = mgr.state.views(layer_idx, req.state_slot)  # state path
            hidden = hidden + block.recur(hidden, ssm, conv)        # MUTATES both

        hidden = hidden + block.mlp(block.norm(hidden))

    return hidden
```

Three details in twenty lines. The attention branch passes a **block table** — a list of physical block ids — because the kernel has to gather scattered pages. The recurrent branch passes **two tensors** obtained from a single slot index, and the kernel writes through them; there is no gather because there is nothing scattered. And `req.append` is called once at the top for the whole layer stack, because the KV slot assignment is a property of the *token*, not of the layer — all four attention layers write token $t$ into the same logical position.

That last point matters when you get to the page-pool design in section 8, because it means the four attention layers can either share one block table (with the physical KV store carrying a leading layer axis, which is what `PagedKVStore` already does) or hold four independent ones. Sharing is simpler and it is what we keep.

---

## 6. Admission control is now two inequalities

The single most consequential line in the whole port is the one that used to read `if free_blocks >= needed`. Here is what it becomes, and what it does to your scheduler's behaviour.

![Branching admission diagram where a request is checked against the block pool and the slot pool and both must pass before it is admitted](/imgs/blogs/implementing-a-two-cache-engine-kv-blocks-plus-recurrent-state-5.webp)

Figure 5 draws the fork and the join. The request splits into two independent questions, and the *and* of their answers is the admission decision. What the diagram does not show, and what you have to reason about, is that the two branches fail under completely different conditions:

- The KV branch fails when **the sum of all live requests' lengths** is too large. Its pressure is proportional to $\sum_i S_i$.
- The slot branch fails when **the count of live requests** is too large. Its pressure is proportional to $n$, and it does not care about lengths at all.

A workload of 400 requests at 1,000 tokens each and a workload of 40 requests at 10,000 tokens each put identical pressure on the KV pool — 400,000 tokens either way — and pressure differing by a factor of ten on the slot pool. Your old capacity intuition, which was entirely about total tokens, is now wrong in one of those two directions depending on which pool you provisioned generously.

There is a clean way to state when each binds. With $N$ slots and a KV region holding $T$ token-slots, the KV pool binds when the mean context $\bar{S}$ exceeds $T/N$, and the slot pool binds below it. Call that ratio the **design context** of the deployment:

$$
S^{*} = \frac{T}{N}
$$

Provision for $S^{*}$ and you are balanced. Serve traffic at $\bar{S}$ far from $S^{*}$ and one pool sits idle while the other throttles you. Section 7 puts numbers on how much idle.

The scheduler also has to handle the case where a *running* request needs a new block and there is none — the preemption path from [request scheduling and preemption](/blog/machine-learning/model-serving/request-scheduling-and-preemption). On a dense transformer, preempting a victim returns $\lceil S_v/16 \rceil$ blocks, and you choose a victim big enough to cover the shortfall. On a hybrid, preempting a victim returns its blocks *and* its slot — and if the shortfall was blocks, the slot you just freed is worthless to you, while the state it held is gone in a way that costs a full prefill to rebuild. Preemption on a hybrid is strictly more expensive per unit of memory recovered than on a dense model, which is a genuinely counterintuitive consequence of using *less* memory overall.

```python
# nanoserve/scheduler_hybrid.py

def admit_batch(mgr, waiting, max_batch: int) -> list:
    """FCFS admission with both constraints, reporting the binding one."""
    admitted, blocked_by = [], None
    while waiting and len(mgr.live) < max_batch:
        req_id, prompt_len = waiting[0]
        ok, why = mgr.can_admit(prompt_len)
        if not ok:
            blocked_by = why
            break
        waiting.popleft()
        admitted.append(mgr.admit(req_id, prompt_len))
    if blocked_by:
        print(f"[sched] head-of-line blocked by {blocked_by}; {mgr.describe()}")
    return admitted
```

Printing the binding constraint on every stall is not a debugging nicety you add later. It is the only signal that tells you whether the fix is "buy more memory", "lower `--max-num-seqs`", or "you provisioned the wrong split", and those three have nothing to do with each other.

---

## 7. Capacity math when half your layers ignore sequence length

Now the numbers that decide the deployment. All of this is derived; the arithmetic is shown so you can check it.

Take an H100 80GB SXM. NVIDIA's datasheet lists 80 GB of HBM3 at 3.35 TB/s. In binary units that device memory is 74.5 GiB. Subtract 14.9 GiB for 8B parameters in bf16 (16.0 GB), and roughly 4.0 GiB for activations, the CUDA context, kernel workspace and allocator slack. Cache budget: **55.6 GiB**.

For a model with the two-term law $M(S) = \sigma + \kappa S$, the number of concurrent requests at context $S$ is:

$$
n(S) = \left\lfloor \frac{M_{\text{budget}}}{\sigma + \kappa S} \right\rfloor
$$

Instantiate it across interleave ratios. Hold the skeleton fixed at 52 layers, 8 KV heads, 128 head dimensions, bf16, with recurrent layers costing 2.0586 MiB each, and vary the fraction $f$ of layers that are full attention:

| Interleave | Attn layers | $\kappa$ per token | $\sigma$ fixed | $M$ at 8k | Concurrent at 8k | Source |
| --- | --- | --- | --- | --- | --- | --- |
| all attention | 52 | 208 KiB | 0 | 1,664 MiB | 34 | derived (synthetic sweep) |
| 1 in 2 | 26 | 104 KiB | 53.5 MiB | 885.5 MiB | 64 | derived (synthetic sweep) |
| 1 in 6 | 9 | 36 KiB | 88.5 MiB | 376.5 MiB | 151 | derived (synthetic sweep) |
| Nemotron-H-8B | 4 | 16 KiB | 49.41 MiB | 177.4 MiB | 320 | derived from config |
| Llama-3.1-8B | 32 | 128 KiB | 0 | 1,024 MiB | 55 | derived from config |

Read the last two rows together, because those are the two real models. A pure transformer of comparable size fits 55 concurrent 8k conversations on this GPU. The hybrid fits 320. That is 5.8×, and it is not a benchmark — it is a division. The synthetic middle rows exist to show the shape of the curve: the win is roughly hyperbolic in the attention fraction, and even a modest 1-in-2 interleave nearly doubles capacity.

#### Worked example: 320 slots that strand 35 GiB

Provision the split-region design for that 8k design point. You want 320 concurrent requests, so:

- State region: $320 \times 51{,}806{,}208 = 16{,}577{,}986{,}560$ bytes = **15.44 GiB**
- KV region: $55.6 - 15.44 = 40.16$ GiB = 41,124 MiB, which at 256 KiB per 16-token block is **164,497 blocks**, or 2,631,952 token-slots.

Check the balance: at 8,192 tokens a request needs 512 blocks, so the KV region supports $164{,}497 / 512 = 321$ requests while the slots support 320. Balanced, as designed. $S^{*} = 2{,}631{,}952 / 320 = 8{,}225$ tokens.

Now run the traffic you actually get. Suppose your endpoint is a chat product and the mean conversation is 1,024 tokens:

- Blocks needed per request: 64. The KV region supports 2,570 requests.
- Slots: 320.
- **You serve 320.** KV blocks in use: $320 \times 64 = 20{,}480$ of 164,497, which is 12.4%.
- Idle KV memory: $40.16 \times 0.876 =$ **35.2 GiB**.

Thirty-five gigabytes of an eighty-gigabyte GPU, allocated, resident, and touched by nothing. And the failure is silent — the server is not erroring, it is just refusing to go above 320 while `nvidia-smi` shows the memory as used, because it *is* used, by an empty pool.

The mirror image is just as bad. Point the same server at a document-analysis workload with 32,768-token contexts:

- Blocks per request: 2,048. KV region supports 80.
- Slots: 320.
- **You serve 80.** State slots in use: 80 of 320. Idle state memory: $240 \times 49.41$ MiB = **11.58 GiB**.

![Grid comparing concurrency at three context lengths for a split two-region pool against a single unified page pool](/imgs/blogs/implementing-a-two-cache-engine-kv-blocks-plus-recurrent-state-6.webp)

Figure 6 has the third column, which we have not derived yet — that is section 8. For now read only the first two: a split pool tuned for one context length delivers its designed capacity at that length and throws away between a quarter and two thirds of the cache budget everywhere else.

#### Worked example: an RTX 4090 cannot afford the slots at all

Repeat the exercise on a 24 GB RTX 4090: 22.35 GiB usable, minus 14.9 GiB of weights, minus about 2.0 GiB of workspace, leaves **5.45 GiB** of cache budget. Now ask for the 320 slots that the H100 configuration wanted: 15.44 GiB. It does not fit — not "it fits but leaves nothing for KV", it does not fit at all, by a factor of nearly three.

Solve the other way instead. On 5.45 GiB, what balanced configuration does a 4090 support at 8k? $5{,}580.8 / 177.41 = 31$ requests: 31 slots consuming 1.50 GiB and a KV region of 3.95 GiB. Against Llama-3.1-8B on the same card, which fits $5{,}580.8 / 1{,}024 = 5$ concurrent 8k requests, the hybrid gets you 31. Six times the concurrency on consumer hardware, from arithmetic, with no quantization involved.

### The cost nobody puts in the capacity table: bandwidth

Memory is not the only thing a cache spends. Every decode step *reads* the cache and, for the recurrent half, writes it back. And here the fixed state stops looking free.

Per request per decode step:

- KV traffic: read $\kappa S$ bytes = $16{,}384 \cdot S$, plus a negligible 16 KiB append.
- State traffic: read *and* write the whole thing = $2\sigma = 103{,}612{,}416$ bytes, or 98.8 MiB, **regardless of $S$**.

Set them equal:

$$
16{,}384 \cdot S = 103{,}612{,}416 \quad \Rightarrow \quad S \approx 6{,}324 \text{ tokens}
$$

Below roughly 6,300 tokens of context, **the fixed state moves more bytes per decode step than the entire KV cache does.** The architecture that saved you 87% of the memory is spending more bandwidth than the thing it replaced, on every short request, forever. That is not an argument against hybrids — at 8k the state is 44% of the KV traffic and at 128k it is 2.8% — but it is the reason a hybrid's decode step does not get 5.8× faster just because 5.8× more requests fit.

Put concrete numbers on it at batch 32, context 8,192, on an H100 at 3.35 TB/s:

| Traffic component | Bytes per step | Time at 3.35 TB/s | Source |
| --- | --- | --- | --- |
| Weights (8B bf16) | 16.00 GB | 4.78 ms | derived from datasheet bandwidth |
| KV read (32 × 128 MiB) | 4.29 GB | 1.28 ms | derived |
| State read + write (32 × 98.8 MiB) | 3.32 GB | 0.99 ms | derived |
| Total memory-bound floor | 23.61 GB | 7.05 ms | derived |

The state is 14% of the step's memory traffic at 8k and batch 32. At batch 128 and 1k context it becomes 45%, because the state term scales with batch while the KV term scales with batch *and* length. This is exactly the roofline reasoning from [the roofline model](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound), applied to a cache instead of a kernel: a term that is constant in $S$ is not constant in $n$.

---

## 8. One pool, two views: getting the 35 GiB back

The split-region design has one flaw and it is fatal: the boundary is a startup constant and the workload is not. The fix is the one production engines converged on, and vLLM states it in a single sentence in the Qwen3-Next post, worth quoting exactly:

> "vLLM automatically tunes the 'logical' block size of the full attention layers to ensure that the state for the full attention layers and linear attention layers occupy the same amount of 'physical' GPU memory."

Unpack what that buys. If one unit of full-attention allocation and one unit of linear-attention allocation are the *same number of bytes*, then there is no boundary to draw. There is one pool of identically-sized physical pages, and each page is handed to whichever cache asks for it next. A short-context workload takes many state pages and few KV pages; a long-context workload takes the reverse; nothing is stranded either way because nothing was reserved either way.

Now derive the page size for our model, which is where it gets pleasing.

We want a page $P$ such that (a) one page holds exactly one recurrent layer's SSM state, and (b) one page holds a whole number of tokens of one attention layer's KV, and (c) that token count is a multiple of the attention kernel's granularity. Take the SSM state first: it is 2,097,152 bytes, exactly 2 MiB. Set $P = 2$ MiB and check the others.

**KV per page.** One attention layer stores $2 \cdot H_{kv} \cdot d_h \cdot b = 2 \cdot 8 \cdot 128 \cdot 2 = 4{,}096$ bytes per token. So:

$$
\text{tokens per page} = \frac{2{,}097{,}152}{4{,}096} = 512
$$

Exactly 512 tokens, and 512 is a multiple of 16. That matters because attention kernels impose their own granularity — the vLLM disaggregation post cites FlashInfer's requirement that full-attention blocks be subdivided to a 16-token boundary, and describes bridging physical and logical blocks precisely so that a physical allocation unit can be subdivided into whatever the kernel demands while the state layers, having no token dimension, use the logical blocks directly. Our 2 MiB page subdivides into 32 logical 16-token blocks per attention layer, and the block table we hand the kernel is that subdivision.

That post also mentions the cost of getting this wrong: their hybrid memory allocator's padding can inflate a full-attention block to 400 tokens. Large logical blocks are not free, and section 8's honest accounting has to include their tail waste.

**Conv states.** All 24 layers' conv states are $24 \times 61{,}440 = 1{,}474{,}560$ bytes — 1.41 MiB, which fits in a single 2 MiB page with 0.59 MiB to spare. One page holds the entire convolution state of the whole request.

So a request's page bill is:

$$
\text{pages}(S) = \underbrace{24}_{\text{SSM, one per layer}} + \underbrace{1}_{\text{all conv states}} + \underbrace{4 \cdot \left\lceil S/512 \right\rceil}_{\text{KV, one set per attention layer}}
$$

<figure class="blog-anim">
<svg viewBox="0 0 660 240" role="img" aria-label="Pages leave a single free pool and are claimed alternately by the block-table view and the state-slot view" style="width:100%;height:auto;max-width:820px">
<style>
.h2-box{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.h2-lbl{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.h2-sub{font:400 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.h2-pg{fill:var(--accent,#6366f1)}
@keyframes h2-up{0%{transform:translate(0,0);opacity:0}12%{opacity:1}82%{opacity:1}100%{transform:translate(268px,-62px);opacity:0}}
@keyframes h2-dn{0%{transform:translate(0,0);opacity:0}12%{opacity:1}82%{opacity:1}100%{transform:translate(268px,62px);opacity:0}}
.h2-u{animation:h2-up 6s linear infinite}
.h2-d{animation:h2-dn 6s linear infinite}
.h2-t1{animation-delay:0s}
.h2-t2{animation-delay:1.5s}
.h2-t3{animation-delay:3s}
.h2-t4{animation-delay:4.5s}
@media (prefers-reduced-motion:reduce){.h2-u,.h2-d{animation:none;opacity:1}}
</style>
<rect class="h2-box" x="20" y="76" width="180" height="92" rx="10"/>
<text class="h2-lbl" x="110" y="112" text-anchor="middle">free page pool</text>
<text class="h2-sub" x="110" y="134" text-anchor="middle">28,467 pages of 2 MiB</text>
<rect class="h2-box" x="440" y="20" width="200" height="82" rx="10"/>
<text class="h2-lbl" x="540" y="52" text-anchor="middle">block-table view</text>
<text class="h2-sub" x="540" y="74" text-anchor="middle">512 tokens per page</text>
<rect class="h2-box" x="440" y="146" width="200" height="82" rx="10"/>
<text class="h2-lbl" x="540" y="178" text-anchor="middle">state-slot view</text>
<text class="h2-sub" x="540" y="200" text-anchor="middle">one layer per page</text>
<rect class="h2-pg h2-u h2-t1" x="206" y="112" width="26" height="20" rx="4"/>
<rect class="h2-pg h2-d h2-t2" x="206" y="112" width="26" height="20" rx="4"/>
<rect class="h2-pg h2-u h2-t3" x="206" y="112" width="26" height="20" rx="4"/>
<rect class="h2-pg h2-d h2-t4" x="206" y="112" width="26" height="20" rx="4"/>
<text class="h2-sub" x="20" y="204">no reserved boundary, so nothing is stranded</text>
</svg>
<figcaption>A single arena of identically-sized pages feeds both caches on demand; the split between them is whatever the live traffic happens to need.</figcaption>
</figure>

The animation is the design in one loop: one queue, two consumers, and the ratio between them decided at runtime instead of at startup.

Here is the pool and the planner.

```python
# nanoserve/pages.py
"""One arena of equal-sized pages, addressed by two different views."""

from collections import deque


class PagePool:
    """Physical pages. Neither cache owns any of them until it asks."""

    def __init__(self, num_pages: int, page_bytes: int = 2 * 1024**2):
        self.num_pages = num_pages
        self.page_bytes = page_bytes
        self._free: deque[int] = deque(range(num_pages))

    @property
    def num_free(self) -> int:
        return len(self._free)

    def allocate(self, n: int) -> list[int]:
        if n > len(self._free):
            raise RuntimeError(f"wanted {n} pages, {len(self._free)} free")
        return [self._free.popleft() for _ in range(n)]

    def free(self, pages: list[int]) -> None:
        self._free.extend(pages)


class HybridPagePlan:
    """Turns a model's shapes into a page bill, for both caches."""

    def __init__(self, page_bytes, kv_bytes_per_token_per_layer,
                 n_attention_layers, ssm_bytes_per_layer,
                 conv_bytes_per_layer, n_recurrent_layers):
        self.page_bytes = page_bytes
        self.tokens_per_page = page_bytes // kv_bytes_per_token_per_layer
        self.n_attn = n_attention_layers
        self.n_rec = n_recurrent_layers
        self.ssm_pages = n_recurrent_layers * -(-ssm_bytes_per_layer // page_bytes)
        self.conv_pages = -(-(conv_bytes_per_layer * n_recurrent_layers) // page_bytes)

    @property
    def fixed_pages(self) -> int:
        return self.ssm_pages + self.conv_pages

    def kv_pages(self, seq_len: int) -> int:
        return self.n_attn * -(-seq_len // self.tokens_per_page)

    def pages_for(self, seq_len: int) -> int:
        return self.fixed_pages + self.kv_pages(seq_len)

    def concurrency(self, num_pages: int, seq_len: int) -> int:
        return num_pages // self.pages_for(seq_len)
```

Drive it against Nemotron-H-8B on the H100 budget:

```python
MIB = 1024 ** 2
plan = HybridPagePlan(
    page_bytes=2 * MIB,
    kv_bytes_per_token_per_layer=2 * 8 * 128 * 2,   # 4096
    n_attention_layers=4,
    ssm_bytes_per_layer=128 * 64 * 128 * 2,         # 2,097,152
    conv_bytes_per_layer=(8192 + 2 * 8 * 128) * 3 * 2,  # 61,440
    n_recurrent_layers=24,
)

budget_pages = int(55.6 * 1024 * MIB) // (2 * MIB)
print("tokens per page :", plan.tokens_per_page)
print("fixed pages/req :", plan.fixed_pages)
print("pool pages      :", budget_pages)
for s in (1024, 8192, 32768, 131072):
    print(f"  S={s:>7,}  pages={plan.pages_for(s):>5}  "
          f"concurrent={plan.concurrency(budget_pages, s):>4}")
```

```console
tokens per page : 512
fixed pages/req : 25
pool pages      : 28467
  S=  1,024  pages=   33  concurrent= 862
  S=  8,192  pages=   89  concurrent= 319
  S= 32,768  pages=  281  concurrent= 101
  S=131,072  pages= 1049  concurrent=  27
```

Compare against the split design tuned for 8k:

| Context $S$ | Split pool, tuned for 8k | Unified 2 MiB pages | Gain | What the split wasted | Source |
| --- | --- | --- | --- | --- | --- |
| 1,024 | 320 | 862 | 2.69× | 35.2 GiB of KV region idle | derived |
| 8,192 | 320 | 319 | 1.00× | nothing — the design point | derived |
| 32,768 | 80 | 101 | 1.26× | 11.6 GiB of state region idle | derived |
| 131,072 | 20 | 27 | 1.35× | 14.5 GiB of state region idle | derived |

The unified pool matches the split at its design point — one request worse, from page rounding — and beats it everywhere else, by a factor of nearly three where it hurts most. That is the entire argument for the trick vLLM describes in one sentence, and now you can see why the sentence is about *equalizing physical bytes*: equal-sized units are what makes a single free queue legal.

#### Worked example: what the unified pool costs

Nothing is free. The unified design pays in three places, and the numbers are small but you should know them.

**Conv-page slack.** The conv page holds 1.41 MiB of a 2 MiB page. Waste: 0.59 MiB per request, or 0.66% of an 8k request's 178 MiB.

**KV tail waste.** With 16-token blocks the last partial block wastes at most 15 tokens of KV, which across 4 attention layers is 240 KiB. With 512-token pages it wastes at most 511 tokens, which is $511 \times 4 \times 4{,}096 = 8.0$ MiB. On an 8k request that is 4.5% in the worst case; on a 300-token request it is a whole extra page set — 4 pages, 8 MiB, against a state cost of 50 MiB. Live with it, or size the page down to 512 KiB and accept that a single SSM state now spans four pages (which is fine, as long as they are contiguous *per layer* — see below).

**Kernel-facing indirection.** The attention kernel wants 16-token logical blocks. Each 2 MiB page is 32 of them, so the block table you build per step is 32× longer than the page list, and you build it by expansion:

```python
def logical_blocks(pages: list[int], tokens_per_page: int,
                   kernel_block: int = 16) -> list[int]:
    """Subdivide physical pages into the granularity the attention kernel wants."""
    per_page = tokens_per_page // kernel_block          # 512 // 16 = 32
    return [p * per_page + i for p in pages for i in range(per_page)]


pages = [7, 12, 3]
print(logical_blocks(pages, 512)[:8], "...", len(logical_blocks(pages, 512)))
```

```console
[224, 225, 226, 227, 228, 229, 230, 231] ... 96
```

Three physical pages become 96 logical blocks. That expansion runs once per step per request and is pure integer arithmetic; in a real engine you build it on the GPU, which is exactly what vLLM's [Model Runner V2 post (2026-03-24)](https://vllm.ai/blog/2026-03-24-mrv2) describes doing with Triton kernels that construct `input_ids`, `positions`, `query_start_loc` and `seq_lens` on-device rather than on the host.

### The constraint the page pool must not violate

One rule makes the whole thing work: **a single recurrent layer's state must live in exactly one contiguous page.** The Mamba-2 kernel reads and writes `(n_heads, head_dim, d_state)` as one dense tensor; it does not have a page table and you do not want to give it one. Choosing $P$ = 2 MiB satisfies this by construction, because that is exactly one layer's SSM state.

What you must *not* do is page the state at a granularity finer than one layer. A state scattered across four 512 KiB pages needs either a gather before the kernel and a scatter after — doubling its already-significant bandwidth cost from section 7 — or a kernel that walks a page table on every element access. Both are worse than the tail waste you were trying to avoid. Per-layer granularity is the sweet spot: the request's 24 layers can live on 24 arbitrary, non-adjacent pages, because the kernel is invoked per layer anyway and gets a fresh base pointer each time.

---

## 9. Stress-testing the design

Now break it. Each subsection below is a condition that a design validated only on the happy path gets wrong.

### Preemption: swap the state, do not recompute it

When the pool runs dry, the scheduler picks a victim. On a dense transformer the standard advice is recompute-over-swap, because prefill is fast and PCIe is slow and the swapped bytes compete with the decode stream. On a hybrid, run the numbers again, because the ratio has moved.

Recomputing an 8,192-token prefill on an 8B model costs roughly $2 \cdot N_{\text{params}} \cdot S = 2 \cdot 8 \times 10^9 \cdot 8192 = 1.31 \times 10^{14}$ FLOPs. NVIDIA's H100 datasheet lists 1,979 TFLOPS of bf16 tensor throughput with sparsity, so about 989 TFLOP/s dense. At a realistic 40% model-FLOPs utilization for a prefill, that is 396 TFLOP/s, giving **331 ms**.

Swapping instead: the whole request is 177.4 MiB at 8k. Over PCIe Gen5 x16 at a practical 50 GB/s, that is $186 \times 10^6 / 50 \times 10^9 =$ **3.7 ms** out and 3.7 ms back.

| Reclaim strategy | Cost at 8k | Cost at 128k | Source |
| --- | --- | --- | --- |
| Recompute prefill (hybrid or dense) | 331 ms | 5.3 s | derived from datasheet FLOPS at 40% MFU |
| Swap hybrid request (177 MiB / 2.05 GiB) | 3.7 ms out + 3.7 in | 43 ms out + 43 in | derived at 50 GB/s |
| Swap dense request (1.0 GiB / 16 GiB) | 21 ms out + 21 in | 336 ms out + 336 in | derived at 50 GB/s |

Swapping wins for both architectures on paper, but it wins by 89× for the hybrid and by 15× for the dense model, and the reason is structural: recompute cost is set by FLOPs and does not change when you swap attention layers for recurrent ones, while swap cost is set by bytes and drops by the same factor your memory did. If your engine's reclaim policy was tuned on dense models, revisit it — and revisit it knowing the caveat that these are bandwidth-limit derivations, not measurements, and that real swap contends with the decode stream for the same PCIe link.

There is a second reason to prefer swapping on a hybrid, and it is not about speed. Recomputing the KV of the four attention layers is exact. Recomputing the recurrent state is *also* exact, but it requires replaying every token, and if your engine implements preemption-by-recompute as "drop the blocks, re-prefill the prompt", it will silently do the wrong thing for a request that has already generated 400 tokens: those tokens are in the prompt for KV purposes but the state must also absorb them. Get that wrong and the request continues from a state that never saw its own output.

### Prefix sharing: copy, do not refcount

Two requests hitting the same 2,000-token system prompt can share KV blocks with a refcount bump, as built in the prefix-sharing machinery of the paged cache. They cannot share a state slot, because the first decode step of either mutates it.

What they *can* share is a **snapshot**. Run the shared prefix once, copy the resulting 49.41 MiB state into a cache keyed by the prefix hash, and let every subsequent request start by copying it back into its own slot. Costs, derived at 3.35 TB/s (a device-to-device copy reads and writes, so 2σ of traffic):

- Snapshot restore: $103.6 \times 10^6 / 3.35 \times 10^{12} =$ **31 µs**
- Re-prefilling those 2,000 tokens instead: $2 \cdot 8\times10^9 \cdot 2000 / 396\times10^{12} =$ **81 ms**

A 2,600× saving, for a storage cost of one full state per cached prefix point. Twenty cached system prompts is 20 × 49.41 MiB = **0.97 GiB**, which is real memory you must subtract from the page pool — 494 pages of the 28,467.

Note the asymmetry with KV prefix caching: KV sharing is free in memory (refcount) and free in time (no copy). State sharing costs memory per cached prefix and 31 µs per hit. It is still overwhelmingly worth it, but it is a different economic object, and it explains why vLLM's Qwen3-Next post lists prefix caching for the hybrid path as a roadmap gap rather than a shipped feature.

```python
# nanoserve/state.py (continued)

class StateSnapshotStore:
    """Prefix-hash -> a frozen copy of a state, for warm-starting requests."""

    def __init__(self, cache: StateCache, capacity: int = 20):
        self.cache = cache
        self.capacity = capacity
        self.entries: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def save(self, key: str, slot: int) -> None:
        if len(self.entries) >= self.capacity:
            self.entries.pop(next(iter(self.entries)))     # FIFO, keep it simple
        self.entries[key] = (self.cache.ssm[:, slot].clone(),
                             self.cache.conv[:, slot].clone())

    def restore(self, key: str, slot: int) -> bool:
        hit = self.entries.get(key)
        if hit is None:
            return False
        self.cache.ssm[:, slot].copy_(hit[0])
        self.cache.conv[:, slot].copy_(hit[1])
        return True

    def nbytes(self) -> int:
        return len(self.entries) * self.cache.bytes_per_slot()
```

### Speculative decoding: the state has no undo

Draft $k$ tokens, verify them, accept the first $j$. On a dense transformer, rejecting the tail is trivial: you truncate the block table and the KV of the rejected positions is simply never read again. On a hybrid, those $k$ tokens each advanced the recurrent state in place. There is no position to truncate — the state at step $j$ is gone, overwritten $k-j$ times.

Two ways out. Checkpoint the state before each draft round: 31 µs per request per step at H100 bandwidth, plus 49.41 MiB of checkpoint storage per in-flight request. At batch 32 that is 1.54 GiB of checkpoints and 1.0 ms per decode step — against a 20 ms TPOT target, a 5% tax before the speculation has bought you anything. Or recompute the state forward from the last committed position, which costs a $j$-token replay through the recurrent layers only.

The vLLM disaggregation post is candid about this being unfinished territory: it lists speculative decoding interaction with hybrid models as "not extensively validated". Treat that as the honest state of the art rather than a gap in your own implementation, and read the [speculative decoding core idea](/blog/machine-learning/speculative-decoding/speculative-decoding-core-idea-draft-and-verify) post with the rollback question in mind.

### Parallel hybrids: when one layer needs both caches

Everything above assumes a layer talks to exactly one cache, which is what a layer-wise interleave gives you. It is not the only architecture. The [Falcon-H1 technical report (arXiv 2507.22448)](https://arxiv.org/abs/2507.22448) describes a *parallel* hybrid: attention heads and Mamba-2 heads run concurrently inside the same mixer block and their outputs are concatenated before the output projection, with the ratio of attention heads to SSM heads adjustable independently.

For that family, `Mixer.ATTENTION` and `Mixer.RECURRENT` are not alternatives — a single layer index maps to both a block-table read and a state-slot read. If your router is an `if/elif`, you are stuck. If it returns a *set* of cache handles per layer, you are not:

```python
# nanoserve/router.py (continued)
from dataclasses import dataclass, field


@dataclass
class LayerCaches:
    """Which caches a layer touches. A set, because parallel hybrids need both."""
    attention: bool = False
    recurrent: bool = False


class GeneralRouter:
    def __init__(self, per_layer: list[LayerCaches]):
        self.per_layer = per_layer

    def needs_kv(self, layer: int) -> bool:
        return self.per_layer[layer].attention

    def needs_state(self, layer: int) -> bool:
        return self.per_layer[layer].recurrent
```

Two extra booleans, written on day one, and the parallel-hybrid family costs you a config parser instead of a rewrite. Note also that a parallel hybrid changes the capacity arithmetic: $f$ is no longer a fraction of *layers* but of *heads within every layer*, so $\kappa$ and $\sigma$ are both nonzero for all $L$ layers.

### Tensor parallelism and disaggregation

Under tensor parallelism the KV cache shards cleanly by head: rank $r$ of TP-$T$ holds $H_{kv}/T$ heads and the block table is identical on every rank. The recurrent state shards by the head axis too, but the *conv* state shards along its channel axis, and that axis is the concatenation of three differently-sized sub-projections. Splitting $10{,}240 = 8{,}192 + 1{,}024 + 1{,}024$ into $T$ equal parts does not put each rank's share of $x$, $B$ and $C$ in one contiguous run unless you lay the tensor out for it.

This is exactly what the vLLM disaggregation post's `DS` layout is for. They store the conv state as its three sub-projections separately and the temporal state as `(dim, state_len)`, so that a decode rank pulling state over RDMA reads only its own $1/T$ slice and never transfers padding — roughly 50 MB per request saved on their bf16 configuration, per their Figure 1. Their concrete worked deployment is `Nemotron-3-Nano-30B-A3B-FP8` at TP=2, described as 52 layers alternating Mamba and full attention, with five hybrid-memory-allocator groups producing six shared KV tensors, on vLLM v0.20.0 or later with `VLLM_SSM_CONV_STATE_LAYOUT=DS` and a `NixlConnector` in the `kv_both` role.

For `nanoserve` the lesson is smaller and still useful: lay out `conv` as three separate tensors from the start if you ever intend to shard it, because reshaping a 15 GiB allocation later is a migration, not a patch.

---

## 10. Measuring it honestly

You cannot tune a two-pool allocator from a single "GPU memory used" number, because both pools are allocated up front and both look identical to `nvidia-smi`. Here is the minimum instrumentation.

```python
# nanoserve/metrics_hybrid.py

def pool_metrics(mgr) -> dict:
    b, s = mgr.blocks, mgr.state
    kv_used = b.num_used / b.num_blocks
    slot_used = (s.n_slots - s.num_free) / s.n_slots
    return {
        "kv_pages_used_frac": round(kv_used, 4),
        "state_slots_used_frac": round(slot_used, 4),
        "binding": "kv" if kv_used > slot_used else "state",
        "headroom_ratio": round(max(kv_used, slot_used)
                                / max(min(kv_used, slot_used), 1e-9), 2),
        "refused_kv": mgr.refused["kv_pages_exhausted"],
        "refused_state": mgr.refused["state_slots_exhausted"],
    }
```

`headroom_ratio` is the one to alert on. A well-provisioned deployment keeps it near 1.0; a value of 8 means one pool is eight times more loaded than the other and you are wasting most of the slack pool. In the 1k-token scenario from section 7 it would read 8.06, and that single number is the 35.2 GiB.

The timing rules from [setting up a reproducible benchmark](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark) apply unchanged, and three of them bite harder here:

**Warm up past the allocator.** The first `allocate()` on a fresh `StateCache` touches 15 GiB of pages that CUDA has not faulted in. Discard at least the first 20 steps, and call `torch.cuda.synchronize()` before you start the clock — a `zero_()` on 50 MB is asynchronous and will otherwise be attributed to whatever you time next.

**Time with CUDA events, not the wall clock.** `torch.cuda.Event(enable_timing=True)` around the decode step, with a synchronize before reading. Wall-clock timing of an async queue measures your Python loop.

**Load-test open-loop.** The two-pool failure mode only appears when arrivals are independent of completions. A closed-loop harness that keeps exactly 64 requests in flight will never exhaust slots, because it never tries to. Use Poisson arrivals and let the queue build; the metric you want is goodput under an SLO, not peak throughput.

And the reproduce-it-yourself framing: on an A100 80GB at 2.0 TB/s, an 8B hybrid at batch 32 and 8k context should land in the neighbourhood of 10–14 ms per decode step from the memory-traffic floor alone — scale the 7.05 ms H100 figure from section 7 by the bandwidth ratio and add scheduler overhead. Run your own step timer and report what you get; if you are far above that range the bottleneck is not the cache.

---

## 11. Case studies and public numbers

Four public data points about *engines*, not architectures, each with its setup and its caveat.

**vLLM's equal-physical-memory block sizing.** The [Qwen3-Next post (2025-09-11)](https://vllm.ai/blog/2025-09-11-qwen3-next) states that vLLM "automatically tunes the 'logical' block size of the full attention layers to ensure that the state for the full attention layers and linear attention layers occupy the same amount of 'physical' GPU memory", implemented with a hybrid KV-cache manager and Triton kernels from Flash Linear Attention. That is the mechanism section 8 derives from first principles. The caveat is that the post gives no throughput numbers and does not state the model's interleave ratio, so treat it as evidence about design, not about performance.

**vLLM's dual descriptor views.** The [hybrid SSM disaggregation post (2026-04-21)](https://vllm.ai/blog/2026-04-21-hybrid-ssm-disagg) reports maintaining two descriptor lists over the same physical memory — one indexing full-attention blocks, one indexing SSM blocks — so that transfers work across heterogeneous tensor-parallel degrees without reshuffling. It reports a `Nemotron-3-Super-120B-A12B-FP8` deployment on 8×H200 with prefill TP4 and decode TP4 that "Pareto-dominates the co-located baseline at higher batch sizes". Read the limits section as carefully as the results: Mamba1 unsupported, gated DeltaNet pending, mixed block sizes in the hybrid allocator unsupported, speculative decoding not extensively validated.

**Granite 4.0 (IBM).** The [`ibm-granite/granite-4.0-h-small` model card](https://huggingface.co/ibm-granite/granite-4.0-h-small) lists the architecture as a decoder-only MoE transformer with GQA, Mamba-2, MoE with shared experts, and gives the layer census directly: **4 attention layers and 36 Mamba-2 layers**. Forty layers, one attention layer in ten. Our section-7 arithmetic says a 1-in-10 hybrid should behave much like Nemotron-H's 1-in-13, and it is a useful independent confirmation that production model families are converging on single-digit attention-layer counts. The card itself makes no quantified memory claim, so do not attach one to it.

**Kimi Linear (Moonshot AI).** [arXiv 2510.26692](https://arxiv.org/abs/2510.26692) reports a layerwise hybrid of Kimi Delta Attention with periodic full-attention MLA at 48B total and 3B activated parameters, "reducing KV cache usage by up to 75%" with up to 6× higher decoding throughput at 1M context. The 75% is the structural number our two-term law predicts for a 3:1 interleave, which is a good sign it is arithmetic rather than a benchmark artifact. The 6× is a long-context result and does not transfer to short prompts, for exactly the bandwidth reason in section 7.

One more data point worth knowing about, in the negative: vLLM's [Model Runner V2 post (2026-03-24)](https://vllm.ai/blog/2026-03-24-mrv2) lists linear-attention models among the architectures its new runner did not support at v0.18.0. Two-cache support is not a solved checkbox even in the engine that leads on it — which is a decent argument for understanding the allocator yourself rather than assuming your framework has it.

---

## 12. When to build this, and when not to

![Decision tree for sizing a hybrid cache showing when a two-region split is adequate and when a single page pool is worth the extra machinery](/imgs/blogs/implementing-a-two-cache-engine-kv-blocks-plus-recurrent-state-7.webp)

Figure 7 is the decision, and it turns on one question: **how wide is your context distribution?**

**Build the two-region split if** your traffic is narrow — a code-completion endpoint where 95% of requests land between 2k and 6k tokens, or a batch pipeline with a fixed prompt template. Tuned to $S^{*}$, the split gives you the same capacity as the unified pool with a quarter of the code, two `torch.zeros` calls and a `deque`, and it is far easier to reason about when a kernel misbehaves. It is what you should write first, and it is what `nanoserve` ships as the default.

**Build the unified page pool if** your traffic spans an order of magnitude in context length, which describes essentially every general-purpose chat or agent endpoint. The 2.7× at short contexts is not a micro-optimization; it is the difference between one GPU and three.

**Do not build either if** you are not actually serving a hybrid. Everything in this post is dead weight on a dense transformer, and the routing indirection costs you a branch per layer for nothing.

**And use vLLM instead of your own code if** you need any of: heterogeneous tensor parallelism, prefill-decode disaggregation, FP8 KV cache on the attention layers, or a hybrid allocator that has been tested against more than one model family. The gap between the design in this post and a production hybrid allocator is not the design — it is the hundred edge cases each of the vLLM limitations lists implies. Write this to understand it. Run theirs when it is somebody's paycheck.

There is one more case worth naming explicitly. If you are memory-bound at *short* contexts and cannot fit the state slots at all — the RTX 4090 case from section 7, where 320 slots would have wanted 15.4 GiB out of a 5.45 GiB budget — the answer is not a cleverer allocator. It is fewer concurrent requests or a smaller model, and the fixed state has turned from an asset into the binding constraint. A hybrid is a bet that you are serving long contexts. Serve short ones and you paid 49 MiB per request for a compression you never used.

---

## Key takeaways

1. **A hybrid engine has two caches with two index formats.** KV is addressed by (block, offset) and grows with tokens; recurrent state is addressed by (slot, layer) and never grows. One descriptor format cannot express both, which is the design constraint the vLLM disaggregation post names explicitly.
2. **State slots must be zeroed on allocate.** A recycled slot is read before it is written, so stale bytes are a previous request's conversation, not garbage. It costs about 15 µs on an H100 and it is the cheapest correctness guarantee in the engine.
3. **Admission is two inequalities, and you must report which one failed.** KV pressure scales with $\sum_i S_i$; slot pressure scales with the request count. A single "out of memory" counter makes the two indistinguishable and undebuggable.
4. **A fixed split between the two regions strands memory whenever traffic moves off its design point.** Tuned for 8k on an 80 GB card, it leaves 35.2 GiB idle at 1k contexts and caps you at 80 requests at 32k.
5. **Equal-sized pages fix it.** Size the page to exactly one recurrent layer's state — 2 MiB for Nemotron-H-8B — and it holds exactly 512 tokens of one attention layer's KV, a clean multiple of the kernel's 16-token granularity. One free queue, two views, nothing reserved.
6. **The fixed state is memory-cheap and bandwidth-expensive.** It moves $2\sigma$ bytes per request per step regardless of context, so below roughly 6,300 tokens it costs more bandwidth per decode step than the entire KV cache.
7. **Preemption economics invert.** Recompute cost is unchanged by the architecture; swap cost falls with the bytes. On a hybrid, swapping beats recomputing by roughly 89× at 8k, against 15× for the dense equivalent.
8. **State cannot be refcount-shared, only snapshot-copied.** A prefix snapshot restores in about 31 µs against 81 ms to re-prefill 2,000 tokens, but each cached prefix costs a full state in storage.
9. **Route layers through a set of cache handles, not an if/else.** Parallel hybrids like Falcon-H1 put attention and SSM heads in the same block, and a boolean pair on day one saves a rewrite later.
10. **Never let a state span pages within a layer.** Per-layer page granularity gives you paging without a gather; anything finer doubles the state's already-significant bandwidth cost.

---

## Further reading

- vLLM, [*Disaggregated Serving for Hybrid SSM Models* (2026-04-21)](https://vllm.ai/blog/2026-04-21-hybrid-ssm-disagg) — dual descriptor views over one physical pool, the `DS` conv-state layout, physical-to-logical block bridging, and a candid list of what is still unsupported.
- vLLM, [*Qwen3-Next support* (2025-09-11)](https://vllm.ai/blog/2025-09-11-qwen3-next) — the one-sentence statement of the equal-physical-memory block-size trick that section 8 derives.
- vLLM, [*Inside vLLM: Anatomy of a High-Throughput Inference System* (2025-09-05)](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) — the single-cache baseline: `free_block_queue`, `req_to_blocks`, and the 16-token default block.
- Dao and Gu, [*Transformers are SSMs* (arXiv 2405.21060)](https://arxiv.org/abs/2405.21060) — the Mamba-2 / SSD formulation that fixes the state shapes we allocate.
- [*Falcon-H1* (arXiv 2507.22448)](https://arxiv.org/abs/2507.22448) and the [`granite-4.0-h-small` model card](https://huggingface.co/ibm-granite/granite-4.0-h-small) — a parallel hybrid and a 4-attention-of-40-layers interleave, the two ends of the routing problem.
- Within this series: [what inference engineering is](/blog/machine-learning/inference-engineering/what-inference-engineering-is), [the memory math of the KV cache](/blog/machine-learning/inference-engineering/the-memory-math-of-the-kv-cache), [paged KV cache: implementing blocks and a block table](/blog/machine-learning/inference-engineering/paged-kv-cache-implementing-blocks-and-a-block-table), [hybrid models and the end of the KV-cache assumption](/blog/machine-learning/inference-engineering/hybrid-models-and-the-end-of-the-kv-cache-assumption), and the capstone [inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook).
- Architecture-side companion: [Nemotron-H: how NVIDIA swaps most attention for Mamba-2](/blog/machine-learning/large-language-model/nemotron-h-hybrid-mamba-transformer).
</content>
</invoke>
