---
title: "Eviction, preemption and KV swapping: what happens when the block pool runs dry"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "Your block allocator has a fixed pool and sixteen requests that keep growing. Learn exactly what an engine must do the moment a running request needs a block that does not exist, why swapping to host memory is twenty-three times cheaper than recomputing and still often the wrong answer, and how to build the swap path without corrupting the cache."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "kv-cache",
    "paged-attention",
    "scheduling",
    "preemption",
    "memory",
    "pytorch",
    "gpu",
    "ml-systems",
    "vllm",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 55
---

The block allocator you built two posts ago has a beautiful property: it never fragments. Every request takes exactly the blocks it needs, one sixteen-token block at a time, and returns them when it finishes. The block table indirection means logical position 1,847 can live anywhere in physical memory. It is the single data structure that made continuous batching possible.

It also has a fixed size. On an RTX 4090 serving Llama-3.1-8B in bf16 you get about 6 GiB of headroom after weights and workspace, which is 3,072 blocks, which is 49,152 tokens of key-value cache across every request on the machine. That is not a lot. Sixteen concurrent requests running a reasoning workload will eat it in about half an hour of wall-clock decoding, and the last thing that happens before they do is the interesting part: a request that has already produced 4,000 tokens, that a user is watching stream into their browser right now, asks for one more block, and there isn't one.

![A timeline showing a block pool of 3072 blocks draining by one block per decode step until it reaches zero and the next append has nowhere to go](/imgs/blogs/eviction-preemption-and-kv-swapping-1.webp)

`allocate()` cannot return `None` and let the caller shrug. There is no graceful degradation available inside a decode step — the attention kernel needs somewhere to write $K$ and $V$ for the token it just computed, and if there is nowhere, the request is dead. So the engine has to take something from someone. This post is about that theft: who gets robbed, how much it costs, how to give it back, and the pathological state where the engine spends every millisecond taking blocks from one request to give to another and produces almost no tokens at all.

By the end you will have three new files in `nanoserve` — `nanoserve/evict.py`, `nanoserve/swap.py` and a rewritten `step()` in `nanoserve/engine.py` — plus a CPU-only simulator you can run without a GPU to watch a preemption policy thrash. You will also have a formula that tells you, for your specific GPU and host link, whether recomputing a victim is cheaper than copying it, and the answer is going to surprise you in both directions.

The usual promise, restated from [the series introduction](/blog/machine-learning/inference-engineering/what-inference-engineering-is): **I have no GPU and I have run none of this.** Every number below is derived from arithmetic I show you, cited from a public source with a link, or framed as something you should expect when you run the script yourself. The results table carries a `Source` column.

---

## 1. A full pool is a countdown, not a margin

Start with the drain rate, because it is the number that makes the rest of the post feel urgent.

During decode, every running request appends exactly one token per step. With a block size of $B_{\text{tok}}$ tokens, a request consumes one new block every $B_{\text{tok}}$ steps. With $R$ requests in the running set, the pool drains at

$$\frac{\Delta \text{blocks}}{\Delta \text{step}} = \frac{R}{B_{\text{tok}}}$$

blocks per step. This is embarrassingly simple and almost nobody has it written on a dashboard. At the vLLM default block size of 16 tokens — which the vLLM team documents in [Inside vLLM: Anatomy of a High-Throughput Inference System](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) (2025-09-05) — and 16 running requests, the pool loses one block per step. Exactly one. The arithmetic is a coincidence but a useful one: it makes the countdown legible.

So take the 4090 configuration. The pool is 3,072 blocks. Sixteen requests arrive with 1,000-token prompts, each taking $\lceil 1000/16 \rceil = 63$ blocks, so 1,008 blocks go out immediately and 2,064 remain. From that moment the engine has **2,064 decode steps** before the pool is empty. At a decode step time of roughly 22 ms (derived in section 9), that is 45 seconds. Forty-five seconds after a batch of sixteen medium prompts is admitted, an engine with no preemption logic crashes.

Two things make this worse than it sounds, and one makes it better.

Worse, first: prefill is bursty. A single new admission with an 8,000-token prompt takes 500 blocks in one step — 500 steps' worth of drain, instantly. The pool does not drain smoothly; it drains in a sawtooth with steep cliffs wherever the scheduler admits someone. Second, output length is unknown at admission time. You cannot compute the working set in advance because you do not know whether a request will emit 40 tokens or 40,000. Every capacity plan is a bet on an output-length distribution.

Better: requests finish, and when they do they return everything at once. A request that completes releases all 63-plus blocks in a single step. In a healthy steady state, completions and admissions balance and the free-block count wanders around a mean. Memory pressure is what happens when the mean drifts down and the wander takes it to zero.

### The headroom rule

You can turn the drain rate into a scheduling constraint. If you want to survive $H$ steps without a preemption event — say, long enough for the average request to finish and release its blocks — you need

$$F_{\min} = \frac{R \cdot H}{B_{\text{tok}}}$$

free blocks at all times. With $R = 16$, $H = 64$ and $B_{\text{tok}} = 16$: 64 free blocks. That is a 2% reserve on a 3,072-block pool, which sounds trivially cheap and is exactly the kind of reserve every naive engine forgets to keep. Set $H$ to 512 steps and the reserve becomes 512 blocks — 17% of the pool, permanently unavailable. The headroom you keep is a direct trade against the concurrency you can run, and it is the first real policy knob in this post.

Hold onto $F_{\min}$. It comes back in section 10 as the thing that actually prevents thrash, when the eviction policy cannot.

---

## 2. Three responses, and only one of them is free

When `allocate()` finds an empty free list there are exactly three places a block can come from, and they cost wildly different amounts.

![A tree showing an empty block pool branching into cached blocks live blocks and the waiting queue with eviction recompute swap and deferral as the four leaf outcomes](/imgs/blogs/eviction-preemption-and-kv-swapping-2.webp)

**Evict a cached block.** After the prefix-sharing work in [prefix sharing, radix trees and copy-on-write](/blog/machine-learning/inference-engineering/prefix-sharing-radix-trees-and-copy-on-write), a block whose refcount drops to zero does not go straight back on the free list. It stays in the hash index so a future request with the same prefix can reuse it. Those blocks are a *cache*: their contents are recoverable by recomputation, nobody currently depends on them, and dropping one costs precisely zero microseconds. If any exist, take one and stop reading this post.

**Reclaim a live block.** A block with refcount greater than zero belongs to a request that is decoding right now. Its contents are not recoverable from anywhere except the model itself. You can free it, but only by first deciding what happens to the request that owns it: either you copy its keys and values somewhere else (swap), or you accept that they are gone and will have to be computed again (recompute). Either way the owning request stops making progress.

**Refuse work.** Do not admit the request in the waiting queue that would have needed the blocks. This costs queue time and nothing else, and it is by far the most under-used of the three. Most memory-pressure emergencies are actually admission-control failures that arrived twenty seconds late; the later post on [admission control, backpressure and latency collapse](/blog/machine-learning/inference-engineering/admission-control-backpressure-and-latency-collapse) is where that argument gets its own 10,000 words.

The critical distinction — and the one that trips up every first implementation — is that **eviction and preemption are different operations on different resources**, even though both end with a block on the free list.

![A matrix of four block classes showing refcount presence in the cache index and what happens to each one when the pool is exhausted](/imgs/blogs/eviction-preemption-and-kv-swapping-3.webp)

Eviction operates on the prefix cache. It is a pure optimization: the cache exists to save future prefill work, and shrinking it degrades hit rate but breaks nothing. Preemption operates on live state. It is a scheduling decision that changes which requests are making progress, and getting it wrong produces user-visible latency spikes or, if you are careless with refcounts, silently corrupted attention outputs.

The rule that keeps them straight: **a block is evictable if and only if its refcount is zero and it carries a hash.** Refcount zero without a hash means it is already free. Refcount above zero means somebody is reading it — either a decoding request or a sibling that shares the prefix — and touching it is a correctness bug, not a policy choice.

That last case deserves a moment. A shared system-prompt block with refcount 3 cannot be evicted no matter how much pressure you are under, because three requests are attending over it this very step. The pool can be 100% "in use" with zero preemptable blocks if enough of it is pinned by sharing. The only way to free a pinned block is to preempt every request that references it, which drops the refcount to zero and moves it into the evictable set. The pinning is not a bug; it is the price of the sharing win, and it means your eviction policy must be able to reason about the *ownership graph*, not just a list of blocks.

---

## 3. Writing the eviction path

Here is the block pool from posts 8 and 9 with the eviction machinery filled in. The important structural choice is that `release()` does not free — it demotes.

```python
# nanoserve/blocks.py
from collections import OrderedDict, deque
from dataclasses import dataclass, field


class OutOfBlocks(RuntimeError):
    """Raised when the pool cannot satisfy an allocation even after eviction."""


@dataclass
class Block:
    bid: int                      # physical block index into the KV tensor
    ref: int = 0                  # how many sequences currently read this block
    hash: int | None = None       # prefix hash, set only when the block is full
    seq_id: int | None = None     # owner while live; None once cached


class BlockPool:
    def __init__(self, num_blocks: int, block_size: int = 16):
        self.block_size = block_size
        self.blocks = [Block(bid=i) for i in range(num_blocks)]
        self.free: deque[int] = deque(range(num_blocks))
        # prefix cache index: hash -> bid, for blocks that are complete
        self.hash_to_bid: dict[int, int] = {}
        # LRU order over refcount-0 hashed blocks; oldest is first
        self.evictable: OrderedDict[int, None] = OrderedDict()
        self.stats = {"evicted": 0, "cache_hits": 0, "allocs": 0}

    # ---- the two operations everything else is built from -----------------

    def allocate(self, seq_id: int) -> int:
        self.stats["allocs"] += 1
        if self.free:
            bid = self.free.popleft()
        elif self.evictable:
            bid = self._evict_lru()
        else:
            raise OutOfBlocks(
                f"pool exhausted: {len(self.blocks)} blocks, "
                f"0 free, 0 evictable"
            )
        b = self.blocks[bid]
        b.ref, b.hash, b.seq_id = 1, None, seq_id
        return bid

    def release(self, bid: int) -> None:
        """Drop one reference. A complete, hashed block becomes CACHED, not free."""
        b = self.blocks[bid]
        b.ref -= 1
        assert b.ref >= 0, f"refcount underflow on block {bid}"
        if b.ref > 0:
            return
        b.seq_id = None
        if b.hash is not None:
            # keep the contents; a future prefix match can reuse them for free
            self.evictable[bid] = None
            self.evictable.move_to_end(bid)
        else:
            # partial block: contents are worthless to anyone else
            self.free.append(bid)

    # ---- eviction ---------------------------------------------------------

    def _evict_lru(self) -> int:
        bid, _ = self.evictable.popitem(last=False)   # oldest first
        b = self.blocks[bid]
        assert b.ref == 0 and b.hash is not None
        del self.hash_to_bid[b.hash]
        b.hash = None
        self.stats["evicted"] += 1
        return bid

    def num_available(self) -> int:
        """Blocks obtainable without preempting anyone."""
        return len(self.free) + len(self.evictable)
```

Three details in there are load-bearing.

**`release()` demotes rather than frees.** This is what makes the prefix cache a cache. If you free straight to the list you lose every reuse opportunity the moment a request finishes, and your hit rate on a repeated system prompt goes to zero. The cost is that `len(self.free)` is now a lie about capacity — the honest number is `num_available()`, and every capacity check in the engine must use it.

**Partial blocks skip the cache.** A block that is not full has no stable hash, because hashing it would produce a key that can never be matched by a longer prefix. The vLLM team makes the same choice and documents the consequence in the anatomy post: only complete blocks are cacheable, so a prefix that is not a multiple of the block size always recomputes `prefix_len % block_size` tokens. Partial blocks therefore go straight back to `free`, which is both simpler and correct.

**Eviction unlinks the hash before returning the block.** Forget this line and you get the worst bug in this entire post: a stale hash pointing at a physical block that now holds a different request's keys. The next prefix lookup "hits", the block table is populated with garbage, and the model produces fluent, confident, completely unrelated text. There is no crash, no assertion, no OOM — just a request that answers a question nobody asked. Put the `del self.hash_to_bid[b.hash]` and the `b.hash = None` on adjacent lines and never separate them.

### Is LRU the right policy?

LRU is the default for the same reason it is the default everywhere: it is cheap, it is hard to reason your way past, and the access pattern it assumes — recently used prefixes are likely to be used again — is exactly true for LLM serving. A system prompt that ten requests shared in the last minute will very likely be shared again in the next.

But LRU over *blocks* has a specific failure mode with radix-tree prefix caches. A long shared prefix occupies many blocks, and they are all touched together on a hit. Under pure block-level LRU, evicting the middle of a 40-block prefix chain destroys the value of the other 39 — a lookup that walks the trie stops at the first missing block, so the tail blocks are now unreachable and merely wasting pool space until they age out themselves. Two fixes, both cheap:

- **Evict leaf-first.** Walk the radix tree from the leaves inward, so the blocks you drop are always the deepest and least-shared. A shared parent stays resident until all of its children are gone. This is the structurally correct policy for a trie-shaped cache.
- **Evict in prefix-chain order.** When you must evict from a chain, take the whole tail rather than a random member, so the surviving prefix stays a valid, matchable prefix.

```python
# nanoserve/evict.py
def evict_leaf_first(pool: "BlockPool", trie, n: int) -> list[int]:
    """Evict n blocks, always taking the deepest evictable node in the trie.

    Falls back to plain LRU order among nodes at equal depth, so the policy is
    'oldest of the deepest' rather than 'oldest overall'.
    """
    freed = []
    while len(freed) < n:
        node = trie.deepest_evictable()      # ref == 0 and no evictable children
        if node is None:
            break
        for bid in reversed(node.block_ids):  # tail of the chain first
            if pool.blocks[bid].ref == 0:
                pool.evictable.pop(bid, None)
                del pool.hash_to_bid[pool.blocks[bid].hash]
                pool.blocks[bid].hash = None
                pool.free.append(bid)
                freed.append(bid)
                if len(freed) == n:
                    return freed
        trie.unlink(node)
    return freed
```

Whether this beats plain LRU depends entirely on your traffic shape. With a handful of dominant system prompts, both policies converge to "keep the hot prefixes" and the difference is noise. With a long tail of semi-shared prefixes — the agentic pattern, where every conversation turn extends a slightly different context — leaf-first is meaningfully better because it preserves the shared trunk. The Mooncake team's published trace statistics for agentic workloads make that shape concrete: in [Mooncake Store: Distributed KV Cache for Agentic Workloads](https://vllm.ai/blog/2026-05-06-mooncake-store) (2026-05-06) they report an input-to-output token ratio of 131 to 1 across 610 Codex and SWE-bench traces, a median of 33 turns, and roughly 2,242 tokens of context growth per turn reaching a median of about 80K tokens by turn 30. That is a trunk-and-branches access pattern, and it is the case where eviction order matters.

---

## 4. Preemption: choosing who loses

Eviction has run, the evictable set is empty, and the pool still cannot satisfy an append. Now somebody who is currently decoding has to stop.

![A branching diagram showing a failed append leading to victim selection and then to either a swap out to pinned host memory or a block drop marked for recompute both of which converge on a resume path](/imgs/blogs/eviction-preemption-and-kv-swapping-4.webp)

### Who is the victim?

The candidate policies, in roughly increasing order of sophistication:

| Policy | Rule | Why it works | What it breaks |
| --- | --- | --- | --- |
| Last-admitted-first | Preempt the newest request in the running set | The newest request has the least sunk cost and the least user-visible progress | Nothing, mostly — this is the sane default |
| Largest-footprint-first | Preempt whoever holds the most blocks | Frees the most blocks per preemption event, so you preempt less often | Systematically punishes long-context requests; a 100k-token RAG request becomes permanently unschedulable |
| Lowest-priority-first | Preempt by an explicit tier | Lets a paid tier survive pressure that a free tier does not | Needs a starvation guard or the low tier never completes |
| Least-progress-first | Preempt whoever has generated fewest tokens | Minimizes wasted work under a recompute policy | Requests that just started keep getting hit; combine with an age term |
| Longest-remaining-first | Preempt whoever is predicted to run longest | Shortest-job-first in disguise, optimal for mean latency | Requires an output-length predictor you do not have |

Last-admitted-first wins on the argument that matters at 3 a.m.: it is the policy whose failure mode is easiest to explain to a user. The request that just started gets delayed. Nobody's half-finished 3,000-token answer evaporates because a newcomer showed up.

The single most important addition to any of these is a **starvation guard**. Without it, "last admitted first" under sustained pressure means the last request admitted is preempted, requeued at the head, re-admitted, preempted again — forever, while the other fifteen requests run to completion. Track a per-request preemption counter and make a request ineligible after $k$ preemptions; if the eligible set becomes empty, preempt the request with the *lowest* counter and reset the ceiling. That guarantees every request eventually runs.

```python
# nanoserve/preempt.py
from dataclasses import dataclass


@dataclass
class PreemptionStats:
    count: int = 0
    swapped_blocks: int = 0
    recomputed_tokens: int = 0


def pick_victim(running, *, max_preemptions: int = 3):
    """Last-admitted-first with a starvation guard.

    `running` is the scheduler's ordered running set (oldest admission first).
    Returns None when every candidate has hit its preemption ceiling AND the
    caller should therefore stop admitting rather than preempt again.
    """
    eligible = [r for r in running if r.preempt.count < max_preemptions]
    if eligible:
        return max(eligible, key=lambda r: r.admit_seq)   # newest admission
    if not running:
        return None
    # everyone is at the ceiling: fall back to fairest available choice and
    # raise the ceiling so the guard cannot deadlock the engine
    victim = min(running, key=lambda r: r.preempt.count)
    return victim
```

Notice what `pick_victim` does *not* do: it never selects a request whose blocks are all shared with someone else, because preempting it frees nothing. Add that filter if your workload has heavy sharing:

```python
def freeable_blocks(req, pool) -> int:
    """Blocks that actually return to the pool if this request is preempted."""
    return sum(1 for bid in req.block_ids if pool.blocks[bid].ref == 1)
```

A request that shares a 2,000-token system prompt with fifteen siblings and has generated 40 tokens of its own holds 128 blocks but can only free 3. Preempting it is close to pointless, and a policy that does not know this will preempt it, discover it freed nothing, and preempt again.

### The two exits

Once a victim is chosen, its keys and values leave the pool by one of two doors.

**Recompute.** Drop every block the victim exclusively owns, set `num_computed_tokens = 0` (or to the length of whatever prefix survives in the cache), and put the request back at the head of the waiting queue. Its generated tokens are *not* lost — the token ids are in the request object, and on resume they become part of the prompt. What is lost is the $K$ and $V$ tensors for all of them, which must be recomputed by re-running prefill over prompt-plus-generated-so-far.

**Swap.** Copy the victim's blocks to host memory, free the physical blocks, and record where the copies went. On resume, allocate fresh physical blocks, copy back, rewrite the block table. The token index does not move; the request continues from exactly where it stopped.

Both free the same blocks. They differ in what they cost, and the arithmetic is not what most people expect.

---

## 5. The arithmetic: recompute versus swap

This is the mechanism block. Two formulas, one ratio, and a conclusion that flips depending on three factors nobody puts in the model.

### Cost of recompute

Recomputing a victim with $n$ tokens of context means running prefill over those $n$ tokens. For a dense transformer, prefill FLOPs are approximately ${2 P n}$ where $P$ is the parameter count — two FLOPs per parameter per token, the standard forward-pass estimate. If the machine sustains a fraction $\eta$ of its peak dense throughput $F$:

$$T_{\text{recompute}}(n) = \frac{2 P n}{\eta F}$$

For Llama-3.1-8B ($P = 8.03 \times 10^9$) on an RTX 4090, whose bf16 tensor throughput NVIDIA specifies as 165.2 TFLOP/s dense, at a realistic prefill MFU of $\eta = 0.4$:

$$T_{\text{recompute}}(2048) = \frac{2 \times 8.03 \times 10^{9} \times 2048}{0.4 \times 165.2 \times 10^{12}} = \frac{3.29 \times 10^{13}}{6.61 \times 10^{13}} \approx 0.498\ \text{s}$$

Just under half a second, or **243 µs per token**.

### Cost of swap

Swapping $n$ tokens means moving $n \cdot b_{\text{kv}}$ bytes out and the same amount back in, where $b_{\text{kv}}$ is KV bytes per token. From the cache math established earlier in this track, for Llama-3.1-8B in bf16 with 32 layers, 8 KV heads and a head dimension of 128:

$$b_{\text{kv}} = 2 \times L \times H_{kv} \times d \times \text{bytes} = 2 \times 32 \times 8 \times 128 \times 2 = 131{,}072\ \text{bytes} = 128\ \text{KiB}$$

A 16-token block therefore holds $16 \times 128\ \text{KiB} = 2\ \text{MiB}$ of keys and values across the whole model. That figure is worth pausing on, because it independently reproduces something the vLLM team published: in [KV Cache Offloading to CPU](https://vllm.ai/blog/2026-01-08-kv-offloading-connector) (2026-01-08) they report that after a KV-layout change the physical block size for Llama-3.1-8B moved from 32 KB to **2 MB** (and for Llama-3.1-70B from 8 KB to 1.25 MB). Their 32 KB is a per-layer, per-tensor figure; multiply by two for $K$ and $V$ and by 32 for the layers and you land on the same 2 MB whole-model block. Two different routes, one number.

With an effective host-link bandwidth $B$:

$$T_{\text{swap}}(n) = \frac{2 n\, b_{\text{kv}}}{B}$$

PCIe 4.0 x16 has a theoretical 32 GB/s per direction; pinned-memory transfers in practice land around 25 GB/s after protocol overhead, which is the number to plan with on a 4090 or an A100. So:

$$T_{\text{swap}}(2048) = \frac{2 \times 2048 \times 131072}{25 \times 10^{9}} = \frac{5.37 \times 10^{8}}{2.5 \times 10^{10}} \approx 0.0215\ \text{s}$$

**21.5 ms** round trip, or 10.5 µs per token.

![A side by side comparison of recompute at 498 milliseconds against a swap round trip at 21.5 milliseconds for the same 2048-token victim on a 4090](/imgs/blogs/eviction-preemption-and-kv-swapping-5.webp)

### The ratio, and why there is no crossover in context length

Divide:

$$\frac{T_{\text{recompute}}(n)}{T_{\text{swap}}(n)} = \frac{2Pn/(\eta F)}{2 n\, b_{\text{kv}}/B} = \frac{P}{\eta F} \cdot \frac{B}{b_{\text{kv}}}$$

The $n$ cancels. **Both costs are linear in context length, so the ratio is a constant property of the hardware and the model — it does not cross over as prompts get longer.** This is the first result that contradicts the folk wisdom. People say "short prefix, recompute; long prefix, swap," and the naive arithmetic says nothing of the kind: on this hardware swap is cheaper by the same factor at 200 tokens as at 200,000.

Plug in the four GPUs in the series matrix. Peak dense bf16 numbers are from the NVIDIA product specifications; the host-link column is the effective pinned-transfer bandwidth you should expect on that platform's PCIe generation.

| GPU | Dense bf16 (TFLOP/s) | Effective host link | ${T_{\text{rc}}}/{T_{\text{sw}}}$ | Reading |
| --- | --- | --- | --- | --- |
| RTX 4090 | 165.2 | PCIe 4.0, ~25 GB/s | 23.2 | swap far cheaper |
| L4 | 121.0 | PCIe 4.0, ~25 GB/s | 31.7 | swap far cheaper |
| A100 80GB SXM | 312 | PCIe 4.0, ~25 GB/s | 12.3 | swap much cheaper |
| H100 80GB SXM | 989 | PCIe 5.0, ~50 GB/s | 7.7 | swap still cheaper |

The trend is the interesting part. **The faster the GPU relative to its host link, the more competitive recompute becomes.** An H100 can regenerate keys and values almost eight times slower than it can fetch them, but a 4090 is twenty-three times slower. On a Grace Hopper system, where NVIDIA specifies NVLink-C2C at 900 GB/s between the CPU and the GPU, the ratio would move by more than an order of magnitude in swap's favour again. The hardware you are on genuinely changes the right answer.

### So why do production engines lean on recompute?

Because the model above prices a *cold, isolated, serialized* recompute against an *ideal, uncontended* copy, and both of those are wrong in a real engine. Three corrections, each of which pushes toward recompute.

**Correction 1: the prefix-cache discount.** A preempted request's prompt is very often still in the prefix cache — especially if the pressure came from many requests sharing a system prompt. If a fraction $h$ of the victim's tokens hit the cache on resume, only $(1-h)$ of them are actually recomputed. With a 90% hit rate the recompute bill drops by 10×.

**Correction 2: the marginal-cost discount.** This is the big one, and it comes straight from the decode floor derived in the baseline post. A decode step at batch 16 on a 4090 is memory-bound: it drags 16.06 GB of weights across HBM and leaves the tensor cores idle for the overwhelming majority of the step. Chunked prefill exploits exactly that hole — recompute tokens ride along inside decode steps that were going to happen anyway, using compute that was otherwise wasted. Call the fraction of the isolated cost you actually pay $\alpha$; on a decode-heavy batch it is plausibly in the 0.2 to 0.5 range, because the first slice of prefill work is nearly free and only later slices contend.

**Correction 3: swap does not overlap the stall.** An asynchronous copy on a side stream overlaps beautifully with *other* work — the vLLM offloading post reports that because their offload is asynchronous, cache misses have minimal TTFT effect. But the allocation that triggered the preemption is blocked until the swap-out *completes*, because those exact physical blocks are what it needs. The copy is on the critical path of the step that provoked it. Recompute has no such stall: dropping blocks is instantaneous.

Fold corrections 1 and 2 into the comparison. Recompute wins when

$$(1-h)\,\alpha \cdot t_{\text{tok}} \lt \frac{2 b_{\text{kv}}}{B} \quad\Longleftrightarrow\quad (1-h)\,\alpha \lt \frac{2 b_{\text{kv}}}{B \cdot t_{\text{tok}}}$$

where $t_{\text{tok}} = 2P/(\eta F)$ is the cold per-token prefill time. On the 4090, with $t_{\text{tok}} = 243$ µs and $B = 25$ GB/s:

$$\frac{2 \times 131072}{25 \times 10^{9} \times 243 \times 10^{-6}} = \frac{262144}{6.075 \times 10^{6}} = 0.0431$$

Recompute wins when the effective recompute cost falls below **4.3%** of a cold full prefill. With $h = 0.9$ and $\alpha = 0.4$ you get $(1-h)\alpha = 0.04$ — just under the line. On an A100 the threshold is 8.2%; on an H100, 12.9%. So the honest answer is: **on a well-tuned engine with a warm prefix cache and chunked prefill, recompute and swap are within a factor of two of each other, and swap's advantage on paper mostly evaporates in practice.** That is a much less satisfying conclusion than "swap is 23× better", and it is the true one.

#### Worked example: a 512-token victim versus a 32k-token victim

Take two requests on the 4090 and price both exits, cold, no cache hits.

The short one, 512 tokens of context, 32 blocks, 64 MiB of KV:

- Recompute: $512 \times 243\ \mu s = 124$ ms.
- Swap round trip: $2 \times 512 \times 131072 / 25\text{e}9 = 5.4$ ms.
- Blocks freed: 32. Ratio 23.2, as predicted.

The long one, 32,768 tokens, 2,048 blocks, 4 GiB of KV:

- Recompute: $32768 \times 243\ \mu s = 7.96$ s.
- Swap round trip: $2 \times 32768 \times 131072 / 25\text{e}9 = 344$ ms.
- Blocks freed: 2,048 — two thirds of the entire 4090 pool in one victim.
- Ratio 23.2 again.

Two things are different at 32k that the ratio does not capture. First, the swap needs 4 GiB of pinned host buffer for one request; a swap space sized for four such victims is 16 GiB of page-locked RAM permanently unavailable to the operating system. Second, an 8-second recompute is not a latency blip, it is a timeout — the user's stream goes silent for eight seconds and the client library gives up. Long-context victims are exactly where swap earns its keep, not because the ratio changes, but because the *absolute* recompute cost crosses a user-perception threshold and the absolute swap cost does not. That is the real "long prefix favours swap" result, and it is about latency SLOs rather than about total work.

---

## 6. Building the swap path

Now the code. This is the part with the sharp edges: pinned memory, side streams, events, and one race that produces silent corruption.

### Layout decides whether swap is one DMA or sixty-four

The KV cache tensor's shape determines whether a logical block is contiguous in memory. Two candidate layouts:

```python
# Layout A: layer-major. Natural for the attention kernel, terrible for swap.
#   [num_layers, 2, num_blocks, block_size, num_kv_heads, head_dim]
#   One logical block = 2 * 32 = 64 disjoint 32 KiB regions.
#
# Layout B: block-major. One logical block is one contiguous 2 MiB region.
#   [num_blocks, num_layers, 2, block_size, num_kv_heads, head_dim]
kv = torch.empty(
    (num_blocks, num_layers, 2, block_size, num_kv_heads, head_dim),
    dtype=torch.bfloat16, device="cuda",
)
```

Under layout A, swapping a 128-block request issues $128 \times 64 = 8{,}192$ separate copies of 32 KiB each. At a per-transfer dispatch overhead in the single-digit microseconds, the overhead alone is tens of milliseconds — more than the transfer itself. Under layout B it is 128 copies of 2 MiB, or one copy if the block ids happen to be contiguous.

This is not a hypothetical trade-off. It is precisely the change the vLLM team describes in their offloading post: after PR #27743 changed the KV layout, the physical block size grew from 32 KB to 2 MB, and they report that the DMA path using `cudaMemcpyAsync` (which benefits from GPUDirect and imposes low core overhead) sustains **83.4 GB/s bidirectional**, versus **68.5 GB/s** for a custom CUDA kernel that copies 16-byte words — while noting that the custom kernel is the better choice for *small* blocks and that DMA is weak there. Big contiguous blocks make DMA viable; small scattered ones do not.

Layout B costs you something in the attention kernel, where per-layer access is now strided across the block dimension. Whether that is acceptable depends on your kernel; the honest framing is that layout is a global decision the swap path gets a vote in, not a free win.

### Pinned host buffers

Host memory must be page-locked for the copy to be genuinely asynchronous. This is not a performance nicety — a `copy_(non_blocking=True)` from pageable host memory silently becomes a synchronous copy, and your carefully overlapped swap turns into a stall you will spend a day finding in a profiler.

```python
# nanoserve/swap.py
import torch
from collections import deque


class HostSwapSpace:
    """Page-locked host staging area for preempted KV blocks.

    One host slot mirrors exactly one GPU block, so a swap is a slot-for-slot
    copy with no reshaping. Allocation happens once at startup because
    cudaHostAlloc is slow and fragments the OS page tables.
    """

    def __init__(self, gpu_cache: torch.Tensor, num_slots: int):
        # gpu_cache is layout B: [num_blocks, L, 2, block_size, H_kv, D]
        block_shape = gpu_cache.shape[1:]
        self.bytes_per_block = gpu_cache[0].numel() * gpu_cache.element_size()
        self.buf = torch.empty(
            (num_slots, *block_shape),
            dtype=gpu_cache.dtype,
            device="cpu",
            pin_memory=True,               # cudaHostAlloc: page-locked
        )
        self.free_slots: deque[int] = deque(range(num_slots))
        self.stream = torch.cuda.Stream()  # copies never block the compute stream
        self.gpu_cache = gpu_cache

    def capacity_gib(self) -> float:
        return self.buf.numel() * self.buf.element_size() / (1 << 30)
```

Sizing rule: the swap space must hold the largest set of victims you are ever willing to have outstanding. A reasonable default is one full pool's worth — 3,072 slots at 2 MiB is 6 GiB of pinned RAM to mirror a 6 GiB GPU pool. Going much beyond that buys you nothing, because a request whose blocks are on host is not making progress; a swap space large enough to hold everyone is a swap space large enough to make no progress at all.

Allocating 6 GiB of pinned memory takes on the order of a second and will show up as a mysterious startup pause. Do it once, at engine construction, and log it.

### The copy, and the race

```python
    def swap_out(self, gpu_bids: list[int]) -> tuple[list[int], torch.cuda.Event]:
        """Copy GPU blocks to host slots. Returns (host_slots, completion event).

        The caller MUST NOT return gpu_bids to the free pool until the returned
        event has been recorded as complete. Freeing early lets a new request
        overwrite blocks the DMA engine is still reading, and the corruption is
        silent: no crash, just wrong keys and values on resume.
        """
        if len(self.free_slots) < len(gpu_bids):
            raise RuntimeError("host swap space exhausted")
        slots = [self.free_slots.popleft() for _ in gpu_bids]

        # The blocks were just written by the compute stream; the copy stream
        # must not start reading before those writes retire.
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            for slot, bid in zip(slots, gpu_bids):
                self.buf[slot].copy_(self.gpu_cache[bid], non_blocking=True)
            done = torch.cuda.Event()
            done.record(self.stream)
        return slots, done

    def swap_in(self, host_slots: list[int], gpu_bids: list[int]) -> torch.cuda.Event:
        """Copy host slots back into freshly allocated GPU blocks."""
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            for slot, bid in zip(host_slots, gpu_bids):
                self.gpu_cache[bid].copy_(self.buf[slot], non_blocking=True)
            done = torch.cuda.Event()
            done.record(self.stream)
        # the compute stream must wait for the data before the next forward pass
        torch.cuda.current_stream().wait_event(done)
        for slot in host_slots:
            self.free_slots.append(slot)
        return done
```

The `wait_stream` / `wait_event` pairs are the whole correctness story, and they are easy to omit because omitting them works fine on a lightly loaded machine and fails under load. The failure mode has a name in the vLLM issue tracker — their offloading post notes a race condition fixed in PR #31341 — which is reassuring in the way that only "a production engine hit this too" can be.

A subtlety worth calling out: `swap_out` returns an event rather than synchronizing. If you call `done.synchronize()` inside `swap_out`, you have turned an asynchronous copy into a 10 ms stall in the middle of your scheduler, and the entire point of the side stream evaporates. The engine must instead track in-flight swaps and only reuse the blocks once the event has fired. That is what the deferred-free queue in the next section does.

### Shared blocks must not be copied

A victim's block list may contain blocks with refcount above one — shared prefix blocks that other running requests are still attending over. Copying those to host is pure waste: they are not going anywhere, because the other readers keep them pinned. The correct behaviour is to decrement the refcount, remember the *hash* rather than the contents, and on resume try the prefix cache first.

```python
def partition_victim_blocks(req, pool):
    """Split a victim's blocks into (copy_these, just_release_these).

    Shared blocks stay resident for their other readers; on resume we re-look
    them up by hash and only recompute the ones that were evicted meanwhile.
    """
    exclusive, shared = [], []
    for bid in req.block_ids:
        (exclusive if pool.blocks[bid].ref == 1 else shared).append(bid)
    return exclusive, [pool.blocks[b].hash for b in shared]
```

This is where the two Track B ideas compose. Prefix sharing reduces how much a preemption can free; it also reduces how much a preemption has to copy. A request sitting on a heavily shared 2,000-token system prompt with 200 tokens of its own conversation copies 13 blocks, not 138.

---

## 7. Putting it in the step loop

Here is the engine step with exhaustion handling. The structure is: try to grow everyone, catch the failure, resolve it, and retry — with a hard bound on retries so a pathological state cannot spin.

```python
# nanoserve/engine.py
from nanoserve.blocks import OutOfBlocks
from nanoserve.preempt import pick_victim, partition_victim_blocks


class Engine:
    def __init__(self, pool, swap_space, model, policy="swap"):
        self.pool = pool
        self.swap = swap_space
        self.model = model
        self.policy = policy            # "swap" | "recompute"
        self.running = []               # ordered by admission
        self.waiting = []               # preempted requests go to the HEAD
        self.pending_free = []          # [(event, [bids])] awaiting DMA completion
        self.metrics = {"preemptions": 0, "swap_bytes": 0, "recompute_tokens": 0}

    # ---- block accounting -------------------------------------------------

    def _reap_pending_free(self):
        """Return blocks whose swap-out DMA has completed."""
        still = []
        for event, bids in self.pending_free:
            if event.query():                     # non-blocking completion check
                for bid in bids:
                    self.pool.free.append(bid)
                    self.pool.blocks[bid].hash = None
                    self.pool.blocks[bid].seq_id = None
            else:
                still.append((event, bids))
        self.pending_free = still

    def _need_blocks(self, req) -> int:
        """Blocks this request must acquire to append one more token."""
        used = req.num_tokens % self.pool.block_size
        return 1 if used == 0 else 0

    # ---- the step ---------------------------------------------------------

    def step(self):
        self._reap_pending_free()

        demand = sum(self._need_blocks(r) for r in self.running)
        attempts = 0
        while self.pool.num_available() < demand:
            if attempts >= len(self.running):
                # nothing left to take: refuse to admit and run a smaller batch
                self._shed_to_fit(demand)
                break
            if not self._preempt_one():
                self._shed_to_fit(demand)
                break
            attempts += 1
            demand = sum(self._need_blocks(r) for r in self.running)

        for req in self.running:
            if self._need_blocks(req):
                req.block_ids.append(self.pool.allocate(req.seq_id))

        out = self.model.decode_step(self.running)
        self._finish_and_admit(out)
        return out

    def _preempt_one(self) -> bool:
        victim = pick_victim(self.running)
        if victim is None:
            return False
        self.running.remove(victim)
        victim.preempt.count += 1
        self.metrics["preemptions"] += 1

        exclusive, shared_hashes = partition_victim_blocks(victim, self.pool)
        for bid in victim.block_ids:
            if self.pool.blocks[bid].ref > 1:
                self.pool.release(bid)          # shared: just drop our reference

        if self.policy == "swap" and exclusive:
            slots, event = self.swap.swap_out(exclusive)
            victim.host_slots = slots
            victim.shared_hashes = shared_hashes
            victim.state = "SWAPPED"
            # blocks are NOT freed here; the DMA is still reading them
            self.pending_free.append((event, exclusive))
            self.metrics["swap_bytes"] += len(exclusive) * self.swap.bytes_per_block
        else:
            for bid in exclusive:
                self.pool.release(bid)
            victim.block_ids = []
            victim.num_computed_tokens = 0
            victim.state = "RECOMPUTE"
            self.metrics["recompute_tokens"] += victim.num_tokens

        self.waiting.insert(0, victim)          # head of queue: it had priority
        return True
```

And the resume path, which is where the two policies diverge again:

```python
    def _resume(self, req) -> bool:
        """Bring a preempted request back. Returns False if there is no room yet."""
        if req.state == "SWAPPED":
            need = len(req.host_slots)
            if self.pool.num_available() < need:
                return False
            gpu_bids = [self.pool.allocate(req.seq_id) for _ in range(need)]
            self.swap.swap_in(req.host_slots, gpu_bids)
            # physical addresses changed; the block table must be rewritten
            req.block_ids = self._rebuild_block_table(req, gpu_bids)
            req.host_slots = []
        else:  # RECOMPUTE
            # try the prefix cache before paying for a single token of prefill
            hit_blocks, hit_tokens = self.pool.lookup_longest_prefix(req.all_token_ids)
            if self.pool.num_available() < req.blocks_needed() - len(hit_blocks):
                return False
            req.block_ids = list(hit_blocks)
            req.num_computed_tokens = hit_tokens
            # the scheduler will now feed the remaining tokens through chunked
            # prefill over the next few steps
        req.state = "RUNNING"
        self.running.append(req)
        return True
```

Three things about this code that matter more than the syntax.

**Swapping preserves contents, not addresses.** `_rebuild_block_table` is not optional. The physical block ids after a swap-in are whatever the allocator handed out, which will almost never match what the request had before. Every kernel that reads the block table must be given the new one, and any cached device-side copy of the table — a captured CUDA graph, for example — is now stale. This is the single biggest reason CUDA graphs and preemption interact badly, and it is why engines that capture graphs per batch-size bucket must re-capture or use indirection when the block table moves.

**Recompute resume goes through the prefix cache.** `lookup_longest_prefix` is the correction-1 discount made real. If the victim's prompt is still cached — very likely if the pressure came from sharing — the "full re-prefill" is a handful of tokens, not thousands. An engine that drops blocks and then blindly re-prefills from token zero is leaving the entire discount on the table.

**The retry loop is bounded.** `attempts >= len(self.running)` is the guard that turns an infinite preemption spiral into a load-shed. If preempting everyone still does not produce enough blocks, the demand is structurally unsatisfiable and the right answer is `_shed_to_fit` — run a smaller batch this step and leave the rest waiting — not another lap around the loop.

---

## 8. Watching a victim leave and come back

The static figures cannot show the part that matters: the pool has a *rhythm*. Blocks fill, a victim's blocks vacate to the host lane, the free count jumps, and later they come back and the free count collapses again.

<figure class="blog-anim">
<svg viewBox="0 0 660 280" role="img" aria-label="A GPU block pool fills until it is nearly full, then three of a victim's blocks move down to a pinned host memory lane and later return" style="width:100%;height:auto;max-width:800px">
<style>
.e1-lane{fill:none;stroke:var(--border,#d1d5db);stroke-width:1.5;stroke-dasharray:5 4}
.e1-slot{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.4}
.e1-blk{fill:var(--accent,#6366f1);opacity:.85}
.e1-lbl{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.e1-sub{font:500 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.e1-cnt{font:700 15px ui-sans-serif,system-ui;fill:var(--accent,#6366f1)}
@keyframes e1-move{0%,26%{transform:translateY(0)}44%,72%{transform:translateY(118px)}90%,100%{transform:translateY(0)}}
@keyframes e1-full{0%,30%{opacity:1}42%,76%{opacity:0}88%,100%{opacity:1}}
@keyframes e1-freed{0%,30%{opacity:0}42%,76%{opacity:1}88%,100%{opacity:0}}
.e1-mv{animation:e1-move 11s ease-in-out infinite}
.e1-f0{animation:e1-full 11s ease-in-out infinite}
.e1-f3{animation:e1-freed 11s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.e1-mv{animation:none}.e1-f0{animation:none;opacity:1}.e1-f3{animation:none;opacity:0}}
</style>
<text class="e1-lbl" x="20" y="30">GPU block pool</text>
<text class="e1-sub" x="152" y="30">8 slots · 2 MiB each</text>
<rect class="e1-slot" x="20"  y="46" width="66" height="66" rx="7"/>
<rect class="e1-slot" x="96"  y="46" width="66" height="66" rx="7"/>
<rect class="e1-slot" x="172" y="46" width="66" height="66" rx="7"/>
<rect class="e1-slot" x="248" y="46" width="66" height="66" rx="7"/>
<rect class="e1-slot" x="324" y="46" width="66" height="66" rx="7"/>
<rect class="e1-slot" x="400" y="46" width="66" height="66" rx="7"/>
<rect class="e1-slot" x="476" y="46" width="66" height="66" rx="7"/>
<rect class="e1-slot" x="552" y="46" width="66" height="66" rx="7"/>
<rect class="e1-blk" x="20"  y="46" width="66" height="66" rx="7"/>
<rect class="e1-blk" x="96"  y="46" width="66" height="66" rx="7"/>
<rect class="e1-blk" x="172" y="46" width="66" height="66" rx="7"/>
<rect class="e1-blk" x="248" y="46" width="66" height="66" rx="7"/>
<rect class="e1-blk" x="324" y="46" width="66" height="66" rx="7"/>
<rect class="e1-blk e1-mv" x="400" y="46" width="66" height="66" rx="7"/>
<rect class="e1-blk e1-mv" x="476" y="46" width="66" height="66" rx="7"/>
<rect class="e1-blk e1-mv" x="552" y="46" width="66" height="66" rx="7"/>
<text class="e1-cnt e1-f0" x="20" y="140">free blocks: 0 — the next append has nowhere to go</text>
<text class="e1-cnt e1-f3" x="20" y="140">free blocks: 3 — reclaimed from the victim</text>
<rect class="e1-lane" x="14" y="164" width="612" height="88" rx="10"/>
<text class="e1-lbl" x="20" y="192">Pinned host DRAM</text>
<text class="e1-sub" x="172" y="192">PCIe 4.0 · 25 GB-s · 84 us per block</text>
</svg>
<figcaption>The victim's three blocks are copied down to page-locked host memory, which is what turns a full pool back into a pool with room; the same copy runs in reverse on resume.</figcaption>
</figure>

The reduced-motion frame shows the pool full with zero free blocks, which is the state the whole post is about, so freezing there loses nothing.

Two details the motion makes obvious that prose does not. First, the freed blocks are the *victim's specific blocks*, not "some blocks" — which is why a victim that shares most of its prefix frees almost nothing. Second, the round trip is symmetric: whatever the swap-out cost, resume pays it again. A request that is preempted and resumed three times pays six one-way transfers.

---

## 9. The thrash spiral

Everything so far assumed preemption is an occasional event. The failure mode that actually takes services down is when it stops being occasional.

### The condition

Define the **working-set demand** $W$ as the number of blocks the running set needs to keep decoding, and $C$ as the pool capacity. If $W \gt C$ persistently, then every step ends the same way: an append fails, a victim is chosen, blocks are freed, the step completes, and on the *next* step the demand is right back where it was because the victim's replacement has grown by a token too. The engine preempts once per step forever.

That is the spiral, and its signature is unmistakable once you know to look: **preemptions per second approaches steps per second, while output tokens per second collapses toward zero.** GPU utilization stays high — the copy engines and the SMs are both busy — so every dashboard that watches utilization reports a healthy machine.

<figure class="blog-anim">
<svg viewBox="0 0 660 250" role="img" aria-label="A victim's blocks oscillate between the GPU pool and host memory four times while the tokens-produced bar barely grows" style="width:100%;height:auto;max-width:800px">
<style>
.e2-lane{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.4}
.e2-blk{fill:var(--accent,#6366f1);opacity:.85}
.e2-lbl{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.e2-sub{font:500 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.e2-track{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.4}
.e2-bar{fill:var(--accent,#6366f1);transform-box:fill-box;transform-origin:left center}
@keyframes e2-osc{0%,6%{transform:translateY(0)}16%,23%{transform:translateY(84px)}33%,39%{transform:translateY(0)}49%,56%{transform:translateY(84px)}66%,72%{transform:translateY(0)}82%,89%{transform:translateY(84px)}100%{transform:translateY(0)}}
@keyframes e2-grow{0%{transform:scaleX(.02)}100%{transform:scaleX(.16)}}
.e2-mv{animation:e2-osc 12s ease-in-out infinite}
.e2-gr{animation:e2-grow 12s linear infinite}
@media (prefers-reduced-motion:reduce){.e2-mv{animation:none}.e2-gr{animation:none;transform:scaleX(.16)}}
</style>
<text class="e2-lbl" x="20" y="26">GPU pool</text>
<text class="e2-sub" x="104" y="26">demand exceeds capacity every step</text>
<rect class="e2-lane" x="20" y="38" width="600" height="58" rx="8"/>
<rect class="e2-blk e2-mv" x="34"  y="46" width="58" height="42" rx="6"/>
<rect class="e2-blk e2-mv" x="100" y="46" width="58" height="42" rx="6"/>
<rect class="e2-blk e2-mv" x="166" y="46" width="58" height="42" rx="6"/>
<text class="e2-lbl" x="20" y="146">Host swap space</text>
<text class="e2-sub" x="146" y="146">21.5 ms round trip · one full decode step</text>
<rect class="e2-lane" x="20" y="158" width="600" height="12" rx="6"/>
<text class="e2-lbl" x="20" y="204">Tokens produced</text>
<text class="e2-sub" x="150" y="204">the engine is busy and the users are not served</text>
<rect class="e2-track" x="20" y="214" width="600" height="20" rx="10"/>
<rect class="e2-bar e2-gr" x="20" y="214" width="600" height="20" rx="10"/>
</svg>
<figcaption>Four preemption round trips in the time the output bar creeps forward, which is what a thrash spiral looks like from the outside: full utilization, no goodput.</figcaption>
</figure>

### The cost, in numbers

Derive the decode step time on the 4090 so the thrash cost is concrete. A decode step at batch 16 must read the weights once and the live KV once:

- Weights: 8.03B parameters $\times$ 2 bytes = 16.06 GB. NVIDIA specifies the RTX 4090 at 1,008 GB/s of memory bandwidth, so 15.9 ms.
- KV: 16 requests $\times$ 3,000 tokens $\times$ 128 KiB = 6.29 GB, so 6.2 ms.
- Total, bandwidth-bound: **about 22 ms per step**, or 727 output tokens per second across the batch.

Now add one swap round trip per step. If the copies fully overlap the compute they add nothing to the step time — but they do not, because the allocation that provoked the preemption waits for the swap-out to land. Realistically a serialized 10.7 ms swap-out plus a 10.7 ms swap-in of the previous victim turn a 22 ms step into something in the 33 to 43 ms range. **Throughput halves.** Meanwhile the victim itself produces zero tokens, so the effective batch is 15, not 16.

#### Worked example: what thrash costs in dollars

Take an A100 80GB at a rented rate of \$1.80 per GPU-hour. Derive the healthy throughput first. Weights: 16.06 GB / 2,039 GB/s (NVIDIA's A100 80GB SXM bandwidth spec) = 7.9 ms. KV at batch 64 and 3,000 tokens each: 64 $\times$ 3000 $\times$ 128 KiB = 25.2 GB / 2,039 GB/s = 12.3 ms. Step time about 20.2 ms, so 64 tokens every 20.2 ms is roughly 3,168 tok/s.

At \$1.80 per hour, one second of GPU time costs \$0.0005. A million tokens takes 1,000,000 / 3,168 = 316 seconds, so the cost is **\$0.158 per million output tokens** — derived, not measured. Halve the throughput with a thrash spiral and the same million tokens costs **\$0.32**. The spiral does not just make users wait; it doubles the unit economics of the service while every utilization dashboard shows green.

### The fix is not a better eviction policy

This is the section's real point. No eviction policy fixes thrash, because thrash is not caused by choosing the wrong victim — it is caused by admitting more work than the pool can hold. Every policy in the table in section 4 produces the same spiral when $W \gt C$; they differ only in *whose* latency is destroyed.

The fixes, in order of how much they help:

1. **Cap the running set.** Never admit a request unless the projected working set stays below $C - F_{\min}$, with $F_{\min}$ from section 1. This is the durable fix and it belongs in the scheduler, not the allocator. The later post on [admission control, backpressure and latency collapse](/blog/machine-learning/inference-engineering/admission-control-backpressure-and-latency-collapse) develops the queueing-theory version.
2. **Add hysteresis to resume.** Do not resume a preempted request the instant one block frees up. Require a margin — say, enough free blocks for the request plus $F_{\min}$ — before bringing it back. Without hysteresis you get the classic oscillation of a control system with no deadband.
3. **Reserve for the running set, not the waiting queue.** When free blocks drop below a watermark, stop admitting new requests entirely and let the running set drain. Growth of an existing request must always outrank admission of a new one, because the existing request has sunk cost and a user watching.
4. **Cap preemptions per request.** The starvation guard from section 4 does double duty here: a request that has been preempted three times is pinned into the running set, which forcibly reduces $W$ by making one request non-preemptable.

Here is the detector, which is a dozen lines and belongs in every engine:

```python
# nanoserve/engine.py (continued)
class ThrashDetector:
    """Flags the state where the engine spends its time moving blocks.

    The signal is the ratio of preemptions to completed decode steps over a
    sliding window. Healthy engines sit near zero; anything above ~0.2 means
    one request in five steps is being taken apart and put back together.
    """

    def __init__(self, window: int = 200, alarm_ratio: float = 0.2):
        self.window, self.alarm_ratio = window, alarm_ratio
        self.steps = 0
        self.preemptions = 0

    def observe(self, preempted_this_step: int) -> bool:
        self.steps += 1
        self.preemptions += preempted_this_step
        if self.steps < self.window:
            return False
        ratio = self.preemptions / self.steps
        self.steps, self.preemptions = 0, 0
        return ratio >= self.alarm_ratio
```

Wire `observe()` into `step()`, export the ratio as a gauge, and alert on it. Preemptions per second is the leading indicator that predicts the latency collapse; queue depth and p99 are the lagging ones that tell you it already happened.

---

## 10. The numbers, with provenance

Everything quantitative in this post, in one table. `derived` means the arithmetic is shown above. `cited` means a public source with a link. `reproduce` means run the named script.

| Quantity | Value | Source |
| --- | --- | --- |
| KV bytes per token, Llama-3.1-8B bf16 | 128 KiB | derived (§5) |
| Physical block, 16 tokens, all 32 layers | 2 MiB | derived (§5) |
| vLLM physical block after KV-layout change, Llama-3.1-8B | 2 MB | cited: [vLLM KV Cache Offloading to CPU](https://vllm.ai/blog/2026-01-08-kv-offloading-connector) (2026-01-08) |
| Block pool on a 4090, 6 GiB free after weights | 3,072 blocks / 49,152 tokens | derived (§1) |
| Pool drain rate, 16 requests, block size 16 | 1 block per decode step | derived (§1) |
| Swap of one 2 MiB block over PCIe 4.0 at 25 GB/s | 84 µs | derived (§5) |
| Swap round trip, 2,048-token victim | 21.5 ms | derived (§5) |
| Cold re-prefill, 2,048 tokens, 4090 at 40% MFU | 498 ms | derived (§5) |
| Recompute / swap ratio, 4090 | 23.2 | derived (§5) |
| Recompute / swap ratio, H100 | 7.7 | derived (§5) |
| Effective-recompute threshold where recompute wins, 4090 | 4.3% of a cold prefill | derived (§5) |
| DMA offload bandwidth, `cudaMemcpyAsync` path | 83.4 GB/s bidirectional | cited: [vLLM KV Cache Offloading to CPU](https://vllm.ai/blog/2026-01-08-kv-offloading-connector) |
| Custom-kernel offload bandwidth, 16-byte words | 68.5 GB/s bidirectional | cited: same post |
| Single-request TTFT reduction from CPU KV offload | ×2 to ×22 by prompt size, Llama-3.1-8B on H100 | cited: same post |
| Concurrent throughput gain at high CPU hit rate | ×9, 10,000 prefills of 512 tokens | cited: same post |
| Remote DRAM read bandwidth, one-sided RDMA | 194 GB/s average, 261.6 GB/s peak | cited: [PegaFlow](https://vllm.ai/blog/2026-05-18-pegaflow) (2026-05-18) |
| Local SSD tier bandwidth via io_uring | ~6.9 GB/s | cited: same post |
| Decode step time, 4090, batch 16, 3k context | ~22 ms | derived (§9) |
| Cost per million output tokens, A100 at \$1.80/hr | \$0.158 healthy, \$0.32 thrashing | derived (§9) |
| Preemption ratio alarm threshold | 0.2 preemptions per step | reproduce: `sim_preempt.py` |
| Policy comparison under a fixed arrival trace | see below | reproduce: `sim_preempt.py` |

### How to measure this honestly

Preemption is harder to benchmark than a kernel, because it is a property of a *trace*, not of a single call. Four rules.

**Use open-loop load.** A closed-loop client that waits for a response before sending the next request cannot produce memory pressure, because the running set is bounded by the client's concurrency. Memory pressure needs Poisson arrivals that keep coming while the engine is already saturated. If your load generator has a `--concurrency` flag and no `--qps` flag, it cannot reproduce any bug in this post.

**Report preemptions, not just latency.** Instrument `preemptions_total`, `preempted_requests_total`, `swap_bytes_total`, `recomputed_tokens_total` and the free-block gauge. A p99 spike with a preemption counter next to it is a diagnosis; a p99 spike alone is a mystery.

**Do not time a swap with wall-clock Python.** `time.perf_counter()` around a `copy_(non_blocking=True)` measures the enqueue, not the transfer. Use `torch.cuda.Event` on the copy stream, or `torch.cuda.synchronize()` before and after if you are willing to destroy the overlap you are trying to measure. The honest metric for the swap path is achieved bandwidth: bytes moved divided by event-to-event elapsed time, compared against the PCIe generation's practical ceiling.

**Separate queue time from compute time.** A preempted request's latency has three parts: time decoding, time waiting after preemption, and time re-establishing state. Report them separately. A service where p99 doubled because of preemption looks identical, in an end-to-end histogram, to one where p99 doubled because the model got slower.

### A simulator you can actually run

You do not need a GPU to study preemption policy, because the policy is pure bookkeeping. This script models the pool, the arrival process and the policies, and prints goodput and preemption counts. Run it and vary `--cap`.

```python
# nanoserve/sim_preempt.py — pure Python, no GPU, no torch
import argparse
import random
from collections import deque


def simulate(seed=0, pool_blocks=3072, block_size=16, qps=4.0, steps=20000,
             cap=None, prompt_mean=1000, out_mean=1200, step_ms=22.0):
    rng = random.Random(seed)
    free = pool_blocks
    running, waiting = [], deque()
    done = tokens = preemptions = 0
    next_id = 0

    for t in range(steps):
        # open-loop Poisson arrivals: qps requests per second, step_ms per step
        if rng.random() < qps * step_ms / 1000.0:
            p = max(1, int(rng.expovariate(1.0 / prompt_mean)))
            o = max(1, int(rng.expovariate(1.0 / out_mean)))
            waiting.append({"id": next_id, "n": p, "budget": o, "blocks": 0,
                            "preempts": 0})
            next_id += 1

        # admission, subject to an optional cap on the running set
        while waiting and (cap is None or len(running) < cap):
            r = waiting[0]
            need = -(-r["n"] // block_size) - r["blocks"]
            if need > free:
                break
            free -= need
            r["blocks"] += need
            running.append(waiting.popleft())

        # growth: every running request appends one token
        for r in list(running):
            r["n"] += 1
            if r["n"] % block_size == 1 and r["n"] > 1:
                if free == 0:
                    # preempt the newest admission (recompute policy)
                    victim = running[-1]
                    free += victim["blocks"]
                    victim["blocks"] = 0
                    victim["preempts"] += 1
                    preemptions += 1
                    running.remove(victim)
                    waiting.appendleft(victim)
                    if victim is r:
                        continue
                free -= 1
                r["blocks"] += 1
            r["budget"] -= 1
            tokens += 1
            if r["budget"] <= 0:
                free += r["blocks"]
                running.remove(r)
                done += 1

    secs = steps * step_ms / 1000.0
    print(f"cap={cap} qps={qps}")
    print(f"  completed        {done}")
    print(f"  output tok/s     {tokens / secs:.0f}")
    print(f"  preemptions      {preemptions}")
    print(f"  preempt per step {preemptions / steps:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=None)
    ap.add_argument("--qps", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    simulate(cap=ap.parse_args().cap, qps=ap.parse_args().qps,
             seed=ap.parse_args().seed)
```

What you should expect when you run it: with `--cap` unset and a high enough `--qps`, `preempt per step` climbs toward and past 0.2 while `output tok-s` falls, which is the spiral. Add `--cap 16` and the preemption count should drop to near zero while completed requests *rise*, because the engine stops destroying work it already did. The exact numbers depend on your seed and the arrival draw; the shape of the curve is the reproducible part, and that shape — throughput improving when you admit *less* — is the entire argument of section 9 in twenty lines of output.

---

## 11. Where this goes at scale: tiered memory

Host DRAM is not the last tier, and swapping for preemption is not the only reason to move keys and values off the GPU. Production systems have generalized both ideas into a memory hierarchy.

![A layered stack showing GPU HBM at 1008 gigabytes per second above a PCIe gap then pinned host DRAM remote DRAM over RDMA and local SSD with the rule that a tier pays only when reuse beats copy cost](/imgs/blogs/eviction-preemption-and-kv-swapping-6.webp)

The tiers, with the bandwidths that define them:

| Tier | Medium | Bandwidth | Role |
| --- | --- | --- | --- |
| L0 | GPU HBM | 1,008 GB/s (4090) to 3,350 GB/s (H100) | live blocks, the only tier attention reads from |
| L1 | Pinned host DRAM | ~25 GB/s (PCIe 4.0) to ~50 GB/s (PCIe 5.0) | preemption swap space and a spillover prefix cache |
| L2 | Remote DRAM over RDMA | 194 GB/s average reported | cross-node prefix reuse |
| L3 | Local SSD | ~6.9 GB/s reported | cold long-tail prefixes |

The L2 and L3 numbers come from the vLLM team's [PegaFlow](https://vllm.ai/blog/2026-05-18-pegaflow) post (2026-05-18), which describes a Rust daemon with exactly this three-tier structure: L1 as local pinned DRAM, L2 as remote DRAM reached by one-sided RDMA reads, and L3 as local SSD via `io_uring`, with CUDA IPC on the data path and gRPC on the control path. They report, for Qwen3-8B on a single host, a 56% throughput improvement (11.97 versus 7.68 req/s), a 36% TTFT reduction and a prefix hit rate of 52.35% versus 11.77%; for DeepSeek-V3.2 with MLA at TP8, 72% higher throughput, 41% lower TTFT and a hit rate of 97.23% versus 65.18%. They are explicit that effectiveness depends on workload shape, which is the correct caveat: a tier only pays when the reuse it enables exceeds the cost of the copy that put the data there.

[Mooncake Store](https://vllm.ai/blog/2026-05-06-mooncake-store) (2026-05-06) takes the L2 idea further, using GPUDirect RDMA between HBM and CPU memory with no staging buffers and a dedicated I/O thread, with a master server tracking block hashes and client health. Their reported numbers on agentic traces are striking — for Kimi-2.5 in NVFP4 across twelve GB200s in a 1-prefill-1-decode configuration, a hit rate of 92.2% versus 1.7%, 3.8× throughput, 46× lower P50 TTFT and 8.6× lower end-to-end latency, scaling from 12 to 60 GB200s with better than 95% hit rate near-linearly.

Two limits are worth naming, because they are exactly the kind of thing a tiered-memory pitch omits.

**These are prefill-side caches, not preemption swap.** The Mooncake team states plainly that **decode does not currently read from the pool** — loads happen on the prefill path only. That means a distributed KV store, however fast, does not solve the problem this post opened with. When a decoding request needs one more block and the pool is empty, an L2 tier that only serves prefill has nothing to offer. Preemption swap and cache offloading share a mechanism and a bandwidth budget, but they are different features with different critical paths.

**Single-instance caches saturate and evict.** Mooncake's own framing is that the motivation for a distributed pool is that per-instance caches fill up and start evicting, which destroys hit rate exactly when load is highest. Tiering pushes the eviction boundary outward; it does not remove it. Somewhere in every hierarchy there is still a last tier with a fixed size and an LRU list, and everything in this post applies to it.

The same distinction explains the vLLM `OffloadingConnector` numbers cited in the table. Their ×2 to ×22 TTFT reduction and ×9 concurrent throughput gain are *prefill* wins from a bigger prefix cache, achieved because the offload is asynchronous and misses therefore cost little. They are not measurements of preemption swap latency, and quoting them as such would be a category error. The transferable fact is the bandwidth: 83.4 GB/s bidirectional over the DMA path is the ceiling any host-tier design has to plan against.

---

## 12. Stress tests

Four scenarios that break a naive implementation, and what each one teaches.

![A matrix of four memory-pressure failure modes with the symptom the tempting response and the durable fix for each](/imgs/blogs/eviction-preemption-and-kv-swapping-7.webp)

**A burst that exceeds the pool.** Twenty requests with 3,000-token prompts arrive in one second. Each needs 188 blocks; twenty of them need 3,760 blocks against a pool of 3,072. There is no eviction, no swap and no policy that makes this fit — the total is larger than the resource. What a naive engine does is admit them all, discover the shortfall on the first append, and enter a spiral immediately. What a correct engine does is admit sixteen and leave four queued. The lesson: **admission is the only control that can refuse work, so it is the only control that can bound the working set.**

**A single request that alone cannot fit.** A 128,000-token RAG prompt needs $128000 \times 128\ \text{KiB} = 16$ GiB of KV — more than the entire 6 GiB pool on a 4090. Preempting every other request frees the whole pool and it is still not enough. The engine must reject this request at admission with a clear error, not accept it and then thrash trying. Compute the requirement from the prompt length before admitting: $\lceil n / B_{\text{tok}} \rceil$ blocks against the pool size, and if it exceeds a configured per-request ceiling, refuse. The lesson: **some requests are structurally infeasible and no runtime policy can rescue them.** This is the memory-pressure half of the long-context failure that gets its own case study later in the series.

**The thrash spiral.** Covered in section 9. The tell is preemptions per step above 0.2 with flat goodput. The tempting response — allocate more host swap space — makes it worse, because a bigger swap space lets more requests be outstanding-but-not-progressing, which raises $W$ without raising $C$.

**A pinned prefix under pressure.** Sixteen requests share a 2,000-token system prompt: 125 blocks with refcount 16. The pool is exhausted. Every one of those 125 blocks is unevictable and unswappable. `freeable_blocks()` returns 3 or 4 for each candidate victim, so preempting one request frees almost nothing and you preempt again, and again. The only way to reclaim the shared prefix is to preempt *all sixteen readers*, which is a catastrophic response to a modest shortage. The correct handling is to recognize the situation — a high ratio of pinned blocks to pool size — and stop admitting rather than keep preempting. The lesson: **prefix sharing changes the shape of your pool, and an eviction policy that does not know about refcounts will grind against blocks it can never have.**

**What about batch 1?** Worth a sentence, because it is the case where none of this fires. A single request on a 3,072-block pool can decode 49,152 tokens before exhaustion. Every mechanism in this post is a consequence of concurrency; at batch 1 you have a memory ceiling and nothing else. This is why single-stream benchmarks tell you nothing about whether an engine handles memory pressure — and why the baseline numbers from the naive decode loop post are necessary but wildly insufficient.

**What about an L4 instead of an A100?** The L4's 300 GB/s of memory bandwidth against the A100's 2,039 GB/s makes decode steps roughly seven times slower, which means the pool drains seven times more slowly in wall-clock time — memory pressure arrives later but hits just as hard. Meanwhile the L4's compute-to-host-link ratio pushes the recompute/swap crossover to 31.7 in swap's favour, the most extreme in the matrix. Slow GPUs on fast host links are the configuration where swapping is most clearly correct.

---

## 13. Case studies and public numbers

Four results from the literature that bear directly on the choices above.

**PagedAttention's original preemption design.** The vLLM paper (Kwon et al., SOSP 2023, [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)) introduced both recovery mechanisms — swapping blocks to CPU memory and recomputing them — and made the all-or-nothing observation that matters: because every block of a sequence is needed together, preemption operates on whole sequences rather than individual blocks. That constraint is why `pick_victim` returns a request, not a block, and it is easy to get wrong if you come at the problem from an operating-system paging background where per-page eviction is the norm.

**Asynchronous offload makes misses cheap.** The vLLM team's [KV Cache Offloading to CPU](https://vllm.ai/blog/2026-01-08-kv-offloading-connector) post reports that because the offload path is asynchronous, cache misses have minimal effect on TTFT — the copy happens off the critical path. The corollary, which they do not need to state and which this post does, is that a *preemption* swap is not off the critical path, because the blocks are needed now. Same mechanism, opposite latency profile, entirely because of when the data is required.

**Block size determines whether DMA is the right transport.** The same post's DMA-versus-custom-kernel comparison — 83.4 GB/s versus 68.5 GB/s bidirectional, with the custom kernel preferred for small blocks — is the empirical version of the layout argument in section 6. Their PR #27743 layout change, which grew the Llama-3.1-8B physical block from 32 KB to 2 MB, is a production engine making exactly the layout-B choice for exactly this reason.

**Freed memory is worth more than it looks.** In [Distributed Inference with vLLM](https://vllm.ai/blog/2025-02-17-distributed-inference) (2025-02-17) the vLLM team report that moving from TP=1 to TP=2 yielded 13.9× more KV-cache blocks and 3.9× more token throughput — super-linear, and attributable to the freed memory rather than the added compute. That is the clearest public evidence for the thesis running underneath this whole post: **on a serving engine, block-pool capacity is a throughput knob, and everything that wastes blocks — fragmentation, a bloated cache, requests parked in swap — costs tokens per second directly.**

---

## 14. When to reach for this (and when not to)

**Build the eviction path always.** Any engine with a prefix cache needs LRU over refcount-zero blocks. It is fifty lines, it cannot make anything worse, and without it the cache grows until it starves the live requests it was supposed to help.

**Build the recompute preemption path second.** It is simpler than swap — drop blocks, requeue, resume through the prefix cache — it needs no pinned memory, no side stream and no event bookkeeping, and per section 5 it is within a small factor of swap once you account for the prefix-cache and chunked-prefill discounts. If you build exactly one preemption mechanism, build this one.

**Build swap when you have long-context victims and an SLO.** The case for swap is not throughput, it is tail latency: a 32k-token victim costs 8 seconds to recompute and 344 ms to swap, and the difference between those two numbers is the difference between a stream that pauses and a stream that dies. If your traffic has long contexts and your users have timeouts, swap earns its complexity. If your traffic is short chat turns, it does not.

**Do not build tiered memory yourself.** L2 and L3 tiers are distributed systems with hash consistency, health checking, RDMA transports and their own failure modes. PegaFlow and Mooncake are real engineering efforts and the right move is to use one, or to use vLLM's offloading connector, rather than to grow your own. `nanoserve` exists to teach you what they are doing and why the numbers come out the way they do — not to compete with them.

**Use vLLM instead of your own code when** you need any of: prefix caching that is correct under concurrent CoW, preemption that composes with chunked prefill and CUDA graphs, or a KV offloading tier. Every one of those is a place where the simple version in this post is a teaching artifact and the production version has absorbed years of race-condition fixes. The value of writing your own allocator is that you can now read theirs, predict its behaviour under pressure, and know which knob to turn — which is exactly what [the capstone](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook) benchmarks. For the operational view of the same problem from the outside, the model-serving series covers [request scheduling and preemption](/blog/machine-learning/model-serving/request-scheduling-and-preemption) as a serving-platform concern rather than an allocator one.

---

## 15. Key takeaways

1. **A full block pool is a countdown clock.** With $R$ running requests and block size $B_{\text{tok}}$, the pool drains at $R/B_{\text{tok}}$ blocks per step. Keep $F_{\min} = R H / B_{\text{tok}}$ blocks free to survive $H$ steps without preempting.
2. **Eviction and preemption are different operations.** Eviction reclaims refcount-zero cached blocks for free. Preemption reclaims live blocks and always costs either a copy or a re-prefill. Blocks with refcount above zero are untouchable at any pressure.
3. **Unlink the hash in the same breath as the eviction.** A stale hash pointing at a reallocated block produces fluent, confident, wrong output with no crash to warn you.
4. **Recompute and swap are both linear in context length, so there is no crossover in $n$.** The ratio $\frac{P}{\eta F}\cdot\frac{B}{b_{\text{kv}}}$ is a hardware constant: 23.2 on a 4090, 7.7 on an H100.
5. **The prefix cache and chunked prefill close most of that gap.** Recompute wins once its effective cost falls below $2 b_{\text{kv}} / (B \cdot t_{\text{tok}})$ of a cold prefill — 4.3% on a 4090, 12.9% on an H100 — which a warm cache plus piggybacked prefill can reach.
6. **Swap wins on tail latency, not on total work.** A 32k victim is 8 seconds of recompute versus 344 ms of copy; the second number fits inside a client timeout and the first does not.
7. **Layout decides whether swapping is one DMA or thousands.** Block-major layout makes a block one contiguous 2 MiB region; layer-major makes it 64 scattered pieces and dispatch overhead swallows the transfer.
8. **Never free a swapped-out block before its event fires**, and never assume physical block ids survive a round trip — rebuild the block table on every resume.
9. **Thrash is an admission-control failure, not an eviction-policy failure.** When working-set demand exceeds pool capacity, every policy produces the same spiral. Cap the running set, add resume hysteresis, and alert on preemptions-per-step above 0.2.
10. **Instrument preemptions before you instrument latency.** Preemption counters are the leading indicator; p99 is the lagging one. And test with open-loop Poisson load, because a closed-loop client physically cannot create the pressure that triggers any of this.

---

## Further reading

- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — Kwon et al., SOSP 2023. The original block allocator, and the paper that introduced both swap and recompute as preemption recovery mechanisms.
- [Inside vLLM: Anatomy of a High-Throughput Inference System](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) — the free-block queue, the `req_to_blocks` mapping, block hashing and the scheduler's waiting/running structure, described by the people who maintain it.
- [KV Cache Offloading to CPU](https://vllm.ai/blog/2026-01-08-kv-offloading-connector) — the DMA-versus-custom-kernel comparison, the KV-layout change that grew physical blocks to 2 MB, and the asynchronous-offload TTFT results.
- [PegaFlow: External KV Cache Service](https://vllm.ai/blog/2026-05-18-pegaflow) — the L1/L2/L3 tiering design, RDMA and SSD bandwidths, and hit-rate ceiling estimation.
- [Mooncake Store: Distributed KV Cache for Agentic Workloads](https://vllm.ai/blog/2026-05-06-mooncake-store) — GPUDirect RDMA, agentic trace statistics, and the honest limitation that decode does not read from the pool.
- [NVIDIA CUDA C++ Programming Guide — page-locked host memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#page-locked-host-memory) — why `pin_memory=True` is the difference between an asynchronous copy and a stall.
- [Paged KV cache: implementing blocks and a block table](/blog/machine-learning/inference-engineering/paged-kv-cache-implementing-blocks-and-a-block-table) — the allocator this post drains.
- [Prefix sharing, radix trees and copy-on-write](/blog/machine-learning/inference-engineering/prefix-sharing-radix-trees-and-copy-on-write) — where refcounts come from, and why some blocks can never be evicted.
- [What inference engineering is](/blog/machine-learning/inference-engineering/what-inference-engineering-is) — the series map and the honesty rule.
