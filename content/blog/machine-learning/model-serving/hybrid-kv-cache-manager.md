---
title: "The Hybrid KV Cache Manager: Serving Models That Mix Attention Types"
date: "2026-07-18"
publishDate: "2026-07-18"
description: "Modern LLMs mix full attention, sliding-window, and Mamba layers in one model — and the old assumption that every layer needs the same amount of KV cache falls apart. This is a deep dive into vLLM's Hybrid KV Cache Manager: KV cache groups, one shared block pool, per-type prefix-cache rules, the 'full attention + X' restriction, and how SGLang solves the same problem a different way."
tags:
  [
    "model-serving",
    "inference",
    "kv-cache",
    "hybrid-models",
    "sliding-window-attention",
    "mamba",
    "vllm",
    "sglang",
    "prefix-caching",
    "llm-serving",
    "memory-optimization",
    "gemma",
  ]
category: "machine-learning"
subcategory: "Model Serving"
author: "Hiep Tran"
featured: true
readTime: 50
image: "/imgs/blogs/hybrid-kv-cache-manager-1.webp"
---

For the first few years of production LLM serving, one assumption held so universally that nobody bothered to state it: every layer of a transformer needs the same amount of KV cache. A 32-layer model with a 4,096-token context caches key and value tensors for all 4,096 tokens at all 32 layers. The [KV cache](/blog/machine-learning/large-language-model/kv-cache) grew linearly with sequence length, uniformly across depth, and the whole machinery of paged allocation — [PagedAttention's block table](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention), the free-block queue, the eviction policy — was built on the quiet premise that a "block" of KV cache meant the same thing for layer 3 as it did for layer 30.

That premise is now false, and it broke on contact with the models people actually want to serve. Gemma 3 interleaves sliding-window attention layers with a few full-attention layers. Llama 4 alternates chunked-local attention with full causal layers. Jamba, Bamba, Nemotron-H, and Granite 4.0 stitch Mamba state-space layers next to attention. Qwen3-Next runs Gated DeltaNet linear attention for three of every four layers. gpt-oss bands most of its attention into a 128-token window and keeps a handful of full-attention layers with learned sinks. In all of these, the memory a layer needs depends on *which kind of layer it is* — and a serving engine that reserves full-attention-sized memory for a sliding-window layer is wasting the exact resource that caps how many requests fit on a GPU.

The **Hybrid KV Cache Manager** is vLLM's answer. It is the subsystem that lets a single model contain layers with wildly different memory footprints, groups those layers by type, and packs them into *one* shared pool of physical blocks so that memory freed by a bounded sliding-window layer is immediately reusable by an unbounded full-attention layer. SGLang solves the same underlying conflict with a different philosophy — separate, elastically resized pools per layer type — and comparing the two is the clearest way to understand what the problem actually is.

![The mental model: a request descends four control layers before two per-group managers converge on one shared block pool](/imgs/blogs/hybrid-kv-cache-manager-1.webp)

The diagram above is the mental model, and the rest of this article is a tour of it. A scheduler, on every step, calls a `KVCacheManager`. That manager holds a `KVCacheCoordinator`, which owns one `SingleTypeKVCacheManager` per group of layers — a `FullAttentionManager` for the full-attention group, a `SlidingWindowManager` for the sliding-window groups, a `MambaManager` for state-space layers. Each of those per-group managers decides how many blocks its layers need and asks the same, single `BlockPool` for them. The whole design turns on one property: because every group is engineered to have an identical page size, that shared block pool holds blocks that are *interchangeable* — a block does not belong to full attention or to sliding window; it belongs to whichever group needs it next. This post is about how that interchangeability is engineered, why it matters, where it breaks, and what happens in production when it does.

## Why hybrid serving is different

Before the mechanism, the mismatch. Here is what changes when a model stops being a homogeneous stack of full-attention layers.

| Aspect | The old assumption | The reality for hybrid models |
|---|---|---|
| Per-layer memory | Every layer caches $O(N)$ tokens | Full attention is $O(N)$; sliding window is $O(W)$; Mamba is $O(1)$ state |
| What a "block" means | The same shape for every layer | Depends on the layer's `KVCacheSpec` — full, sliding, chunked-local, or Mamba |
| Sizing the KV pool | `pool / (bytes-per-token × N)` uniformly | Different groups consume different block counts for the same request |
| Freeing blocks | Only when a request finishes | Also mid-request, when tokens leave a sliding window |
| Prefix-cache hit rule | A prefix hits if all its tokens are cached | Full attention needs all tokens; sliding window needs only the last $W$ |
| The allocator | One block table, one manager | One coordinator over multiple per-type managers, one shared pool |
| Model coverage | Any transformer | Exactly "full attention + one other type" (for now) |

Every row of that table is a design decision the Hybrid KV Cache Manager had to make, and every one of them has a failure mode when it is made naively. A serving engineer who understands only the left column will, sooner or later, watch a Gemma-3 deployment run out of memory at a context length a full-attention model of the same size handles comfortably — and will not understand why, because the model is *smaller* in every way that the old mental model can see. The answer is in the right column, and it starts with the memory demand of the layers themselves.

## 1. The problem: heterogeneous memory demand

**Senior rule of thumb: in a hybrid model, the cheapest layer and the most expensive layer can differ in memory by three orders of magnitude — and a uniform allocator sizes everything for the most expensive one.**

Consider the three kinds of "efficient" layer that show up in production, and what each actually needs to cache per request.

A **full-attention** layer must let every future token attend to every past token, so it caches a key and a value vector for all $N$ tokens in the sequence. Its footprint is $O(N)$ and grows without bound as the conversation continues. This is the layer type the entire pre-hybrid stack was built around.

A **sliding-window attention** layer (Gemma, Ministral, gpt-oss's banded layers) restricts each token to attend only to the previous $W$ tokens. Once the sequence passes $W$ tokens, the oldest keys and values will never be read again, so they can be freed. Its footprint is bounded at $O(W)$ regardless of how long the sequence gets. A 100,000-token conversation on a $W = 4096$ layer has the same KV footprint as a 4,096-token one. This is the same idea covered from the model-architecture side in [KV cache optimization and management](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management) and from the serving side in [KV cache optimization](/blog/machine-learning/model-serving/kv-cache-optimization) — but here it is not one property of one model; it is one layer type living next to another inside the same forward pass.

A **Mamba / state-space** layer does not cache per-token keys and values at all. It maintains a fixed-size recurrent state — a convolution state and an SSM temporal state — that is updated in place at each step. Its footprint is $O(1)$ in sequence length: one state per request, no matter whether the request is 200 tokens or 2,000,000. It is large in absolute terms (megabytes, because the state dimension is big) but it does not grow.

![Naive uniform allocation reserves full-attention-sized memory for layers that never use it; per-type allocation gives each layer only what it needs](/imgs/blogs/hybrid-kv-cache-manager-2.webp)

The figure makes the waste concrete. Suppose we serve a model with a mix of these layers at a 32,768-token context. A uniform allocator — the behavior of vLLM's V1 engine before the hybrid manager, which modeled every layer with a `FullAttentionSpec` — reserves $O(N) = 32\text{K}$ slots for *every* layer, including the sliding-window and Mamba layers that will never use more than a tiny fraction of it. The right column is what a per-type allocator does instead: the full-attention layers still get their $32\text{K}$ slots, but the sliding-window layers get $O(W) = 4\text{K}$ slots and the Mamba layers get a single state each. For a model that is mostly efficient layers — Qwen3-Next is 75% linear attention, MiniMax-Text-01 is 87.5% lightning attention — the difference between these two columns is the difference between fitting a handful of long-context requests on a GPU and fitting many times more.

The uniform allocator is not merely inefficient; it inverts the entire economic argument for building a hybrid model. The whole reason Google shipped Gemma 3 with five sliding-window layers for every full-attention layer, and the whole reason MiniMax shipped seven lightning-attention layers for every softmax layer, was to make long-context serving cheap. Serve that model with an allocator that reserves full-attention memory everywhere, and you have paid the full memory price of a dense transformer while getting none of the memory savings the architecture was designed to deliver. You have, in effect, un-optimized the model in the serving layer.

So the problem statement is precise: **let a single model contain layers whose per-request memory footprint differs by orders of magnitude, give each layer only the memory its type requires, and do it without fragmenting the GPU into rigid per-type partitions that waste memory a different way.** That last clause is the hard part, and it is what the next three sections are about.

## 2. KV cache groups: clustering layers that behave alike

**Senior rule of thumb: the unit of management is not the layer and not the model — it is the group, a set of layers that share one KV-cache spec and therefore need one identical block.**

vLLM's first move is to stop thinking about individual layers and start thinking about **KV cache groups**. A group is a set of layers that share the same `KVCacheSpec` — same attention type, same head count, same head dimension, same block size — and therefore require exactly the same amount of memory per block and the same number of blocks per request. Every layer in a group is, from the allocator's point of view, indistinguishable.

The spec hierarchy is worth naming precisely, because it is the vocabulary the whole system speaks. `KVCacheSpec` is the abstract base; its concrete subclasses are `FullAttentionSpec`, `SlidingWindowSpec`, `ChunkedLocalAttentionSpec`, `MambaSpec`, and `CrossAttentionSpec` (for encoder-decoder models). Each spec knows how to compute one crucial number: `page_size_bytes`, the number of bytes one block occupies for one layer of that type. A `KVCacheGroupSpec` bundles a list of layers with their shared spec, and a `KVCacheConfig` is the top-level object listing all the groups the model was carved into, each backed by a physical `KVCacheTensor`.

Here is the shape of these specs, lightly paraphrased from vLLM's `kv_cache_interface`:

```python
# vllm/v1/kv_cache_interface.py (paraphrased)
class KVCacheSpec:
    block_size: int          # tokens per block, e.g. 16
    @property
    def page_size_bytes(self) -> int:
        """Bytes for one block, for one layer of this type."""
        raise NotImplementedError

class FullAttentionSpec(KVCacheSpec):
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
    @property
    def page_size_bytes(self) -> int:
        # 2 (key + value) * block_size * heads * head_dim * dtype_bytes
        return 2 * self.block_size * self.num_kv_heads * self.head_size * self.dtype.itemsize

class SlidingWindowSpec(FullAttentionSpec):
    sliding_window: int      # the only difference at the byte level is *how many* blocks

class MambaSpec(KVCacheSpec):
    shapes: tuple            # conv-state and ssm-state shapes; one page holds a whole state
    @property
    def page_size_bytes(self) -> int:
        return sum(prod(s) for s in self.shapes) * self.dtype.itemsize
```

Notice that a `SlidingWindowSpec` has the *same* per-block byte size as a `FullAttentionSpec` with the same heads and head dimension. The difference between them is not how big a block is — it is how many blocks a request needs. A full-attention layer needs $\lceil N / \text{block\_size} \rceil$ blocks; a sliding-window layer needs only about $\lceil W / \text{block\_size} \rceil$. That distinction — same block, different count — is exactly what makes the shared-pool trick in the next section possible.

Now, how are layers assigned to groups? The rule vLLM uses is deliberately simple, and it is easiest to see on a real model.

![Gemma-3-27B's 62 layers split into groups of 10, with the last group padded so every group holds the same block count](/imgs/blogs/hybrid-kv-cache-manager-3.webp)

Take Gemma-3-27B, whose 62 layers follow a 5:1 pattern — five sliding-window layers for every full-attention layer — giving 52 sliding-window layers and 10 full-attention layers. The grouping algorithm picks the group size as the *smallest layer count among the attention types present* — here, $\min(10, 52) = 10$ — and then carves each type into groups of that size, padding the last group when a type's count is not an even multiple:

- **Group 0** — the 10 full-attention layers, `full.0` through `full.9`.
- **Groups 1 through 5** — 10 sliding-window layers each: `sw.0`–`sw.9`, `sw.10`–`sw.19`, and so on through `sw.40`–`sw.49`.
- **Group 6** — the last 2 sliding-window layers, `sw.50` and `sw.51`, plus 8 padding slots so the group still holds exactly 10 layers' worth of blocks.

Seven groups, each holding the memory of exactly 10 layers. Why force them all to hold ten? Because that is what makes their `page_size_bytes` identical, and identical page sizes are the precondition for a single shared pool. If Group 0 held 10 layers and Group 6 held 2, the two groups would allocate blocks of different physical sizes, and a block freed by one could not be reused by the other without a fragmentation nightmare. The padding is the price of interchangeability: eight layers' worth of wasted memory in one group, in exchange for a pool where every free block fits every group. On a 62-layer model, that is a rounding error; the payoff is the entire shared-pool architecture.

The validator that builds these groups lives in `kv_cache_utils.py`, and the core check is exactly the one you would expect:

```python
# vllm/v1/core/kv_cache_utils.py (paraphrased)
def _get_kv_cache_groups(kv_cache_spec: dict[str, KVCacheSpec]):
    # Every group must end up with the same page_size_bytes so one BlockPool
    # can serve all of them. The uniform-type path pads the shorter type's
    # last group; a genuinely uniform model (all full attention) takes the
    # fast path with a single group.
    page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}
    if len(page_sizes) == 1:
        return _get_kv_cache_groups_uniform_spec(kv_cache_spec)
    # ...otherwise split by type, then pad so max_page_size is shared by all
    max_page_size = max(spec.page_size_bytes for spec in kv_cache_spec.values())
    # ... pad each group's spec up to max_page_size ...
```

### Second-order optimization: grouping is a scheduling decision, not just a memory one

There is a subtlety that only bites at scale. Because a request's block *table* is per-group, the number of groups affects how much bookkeeping the scheduler does per step. A model split into 7 groups maintains 7 block tables per request; a model split into 30 groups maintains 30. vLLM keeps the group count small by using the largest group size the padding rule allows, which is why it picks the *minimum* layer count as the group size rather than, say, one layer per group. One-layer-per-group would minimize padding waste but explode the per-request metadata and the per-step scheduling cost. The chosen rule trades a little padding memory for a lot fewer block tables — the right trade for a system that runs the scheduler thousands of times per second.

## 3. One unified block pool

**Senior rule of thumb: the win is not that each layer type gets its own memory — it is that they all share the same memory, and a block freed anywhere can be allocated anywhere.**

This is the heart of the design, and it is worth being slow about. A tempting but wrong way to serve a hybrid model is to give each attention type its own pre-sized memory region: carve the GPU into a "full-attention pool" and a "sliding-window pool," size each one at startup, and manage them independently. This is essentially SGLang's approach (which we will see has its own machinery to make it work), and the failure mode of doing it naively is **static-partition fragmentation**. If you guess wrong about the ratio of full-attention to sliding-window demand — and the right ratio depends on the workload's context-length distribution, which shifts hour to hour — one pool runs dry while the other sits half-empty, and you cannot move memory between them.

vLLM avoids that trap by making all groups share a **single** `BlockPool`. Because §2 forced every group to the same `page_size_bytes`, a physical block is a fungible unit: it does not carry a type. The pool holds `num_gpu_blocks` identical blocks and hands them out to whoever asks.

![One pool of interchangeable blocks and one free queue: a block freed by a sliding-window layer returns to the shared queue and is reused by the full-attention group](/imgs/blogs/hybrid-kv-cache-manager-4.webp)

The figure shows the consequence. At the top, the shared pool holds eight physical blocks; at this instant, four are held by the full-attention group, two by a sliding-window group, and two are free. Below, the lifecycle: the `SlidingWindowManager` frees a block the moment a token leaves its window; that block returns to a single `free_block_queue` — one LRU queue shared by every group — and the `FullAttentionManager` then takes it for the next token of a growing sequence. There is no "full-attention memory" and "sliding-window memory." There is memory, and it flows to demand.

The `BlockPool` interface is small and the important methods are exactly these:

```python
# vllm/v1/core/block_pool.py (paraphrased)
class BlockPool:
    def __init__(self, num_gpu_blocks: int, enable_caching: bool):
        self.blocks = [KVCacheBlock(i) for i in range(num_gpu_blocks)]
        # One LRU queue of free blocks, shared across ALL kv cache groups.
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
        # Block-hash -> cached block, for prefix caching (see §5).
        self.cached_block_hash_to_block = BlockHashToBlockMap()

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        # Pop from the front of the shared LRU. If the popped block still
        # carries a prefix-cache hash, evict that cache entry now.
        return self.free_block_queue.popleft_n(num_blocks)

    def free_blocks(self, blocks: list[KVCacheBlock]) -> None:
        # A block freed here — whether the request finished or a token left a
        # sliding window — goes back to the SAME queue any group draws from.
        self.free_block_queue.append_n(blocks)
```

The single `free_block_queue` is the whole game. When any group frees a block — because a request finished, or because a sliding-window layer's oldest block fell out of the window — that block goes back onto the one queue that every group draws from. A long-running streaming session on a sliding-window model is constantly recycling blocks back into the pool, and those recycled blocks become available to a brand-new request's full-attention layers on the very next step. The memory "breathes" between layer types automatically, sized by actual demand rather than a startup guess. This is the property a static partition can never have.

There is a second, quieter benefit: **one eviction policy**. Because there is one LRU queue, the engine has a single, global notion of which cached block is least recently used. It does not have to arbitrate between a "full-attention LRU" and a "sliding-window LRU," deciding whose cold blocks to evict first. Every freed block — from any group — sits in one order, and the coldest one is the next to be reused or evicted. That is a genuine simplification over any multi-pool design, and it is the direct dividend of paying the padding tax in §2.

### Second-order optimization: page_size and the block-count identity

The identity that makes this work is worth stating as an equation. For a group of $L$ layers with per-layer, per-block byte size $b$, the group's page size is $\text{page\_size} = L \times b$. The shared pool is a flat buffer of $P$ total bytes divided into blocks of that page size, so the pool holds $P / (L \times b)$ blocks. Because §2 guarantees every group has the same $L \times b$, every group sees the *same number of blocks* in the pool — which is why one block index means the same thing to every group, and why the free queue can be group-agnostic. Break the equal-page-size invariant and this identity collapses; the padding in §2 exists precisely to preserve it. (The Jenga paper, "Effective Memory Management for Serving LLM with Heterogeneity," formalizes this and is the research lineage behind vLLM's implementation — see Further reading.)

### Worked example: what the shared pool buys on one H100

It helps to put numbers on the savings, because they are large enough to change a capacity plan. Take Gemma-3-27B — 10 full-attention layers, 52 sliding-window layers, and a sliding window of 1,024 tokens — served at a 64K context. Count KV in units of per-layer token-slots (a slot being one token's key-plus-value at one layer), which cancels out the head-dimension and dtype details and isolates the architectural effect.

A uniform allocator reserves $O(N)$ for every layer: $62 \times 64\text{K} \approx 3.97\text{M}$ token-slots per fully-extended request. The hybrid manager reserves $O(N)$ only for the 10 full-attention layers and $O(W)$ for the 52 sliding-window layers: $10 \times 64\text{K} + 52 \times 1024 \approx 640\text{K} + 53\text{K} \approx 693\text{K}$ token-slots. The ratio is about $3.97\text{M} / 693\text{K} \approx 5.7\times$. On the same H100, with the same weights loaded and the same leftover memory for KV, the hybrid manager fits roughly **5.7× more** long-context requests than the uniform allocator — not because it compressed anything, but because it stopped reserving full-attention memory for layers that were architecturally bounded. That multiple is the entire reason the subsystem exists, and it grows as the sliding-window share of the model grows: on a 7:1 model like MiniMax-Text-01, the equivalent ratio is larger still.

The reason this shows up as a hard OOM rather than gentle slowdown is that KV reservation is up-front, not lazy: the engine must know a request *can* reach its declared `max_model_len` before it admits it. Reserve $5.7\times$ too much per request and your admission control rejects requests — or the pool runs dry mid-flight and preempts them — at roughly one-sixth the concurrency the hardware could actually sustain. The hybrid manager does not make any single request faster; it makes far more of them fit at once.

## 4. Per-type allocation and freeing

**Senior rule of thumb: full attention only ever grows; sliding window grows and shrinks within a single request — and the manager that handles shrinking mid-request is where most of the interesting logic lives.**

Each group is driven by a `SingleTypeKVCacheManager` subclass, and the family is larger than you might guess. The verified set in vLLM's `single_type_kv_cache_manager.py` is `FullAttentionManager`, `SlidingWindowManager`, `ChunkedLocalAttentionManager`, `MambaManager`, `CrossAttentionManager`, plus two specializations that matter for real models: `SinkFullAttentionManager` (for attention-sink models like gpt-oss) and `RSWAManager` (a rotating-sliding-window variant that subclasses `FullAttentionManager`). Each implements the same interface — how many blocks does this request need, which blocks can be freed — but with type-specific logic.

The `FullAttentionManager` is the simple case: blocks only ever get allocated, never freed until the request ends, because every token remains attendable forever. Its "which blocks can I free" method returns nothing mid-request.

The `SlidingWindowManager` is where it gets interesting, because it frees blocks *during* a request.

<figure class="blog-anim">
<svg viewBox="0 0 720 260" role="img" aria-label="A sliding window sweeps left to right across a token sequence; blocks behind the window are freed back to the shared pool while blocks ahead are allocated" style="width:100%;height:auto;max-width:820px">
<title>Sliding-window KV cache: allocate ahead, free behind</title>
<style>
.a5-cell{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.a5-live{fill:var(--accent,#6366f1)}
.a5-lbl{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.a5-key{font:600 15px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.a5-win{fill:none;stroke:var(--accent,#6366f1);stroke-width:3;rx:8}
@keyframes a5-slide{0%{transform:translateX(0)}100%{transform:translateX(360px)}}
@keyframes a5-fillA{0%,18%{opacity:1}42%,100%{opacity:.12}}
@keyframes a5-fillB{0%,52%{opacity:.12}74%,100%{opacity:1}}
.a5-mv{animation:a5-slide 9s ease-in-out infinite alternate}
.a5-freed{animation:a5-fillA 9s ease-in-out infinite}
.a5-alloc{animation:a5-fillB 9s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.a5-mv,.a5-freed,.a5-alloc{animation:none}.a5-alloc{opacity:1}.a5-freed{opacity:.12}}
</style>
<text class="a5-key" x="360" y="40">window size W = 4 blocks — only blocks inside the frame are kept</text>
<rect class="a5-cell" x="40"  y="90" width="60" height="72" rx="8"/>
<rect class="a5-cell" x="110" y="90" width="60" height="72" rx="8"/>
<rect class="a5-cell" x="180" y="90" width="60" height="72" rx="8"/>
<rect class="a5-cell" x="250" y="90" width="60" height="72" rx="8"/>
<rect class="a5-cell" x="320" y="90" width="60" height="72" rx="8"/>
<rect class="a5-cell" x="390" y="90" width="60" height="72" rx="8"/>
<rect class="a5-cell" x="460" y="90" width="60" height="72" rx="8"/>
<rect class="a5-cell" x="530" y="90" width="60" height="72" rx="8"/>
<rect class="a5-cell" x="600" y="90" width="60" height="72" rx="8"/>
<rect class="a5-live a5-freed" x="40"  y="90" width="60" height="72" rx="8"/>
<rect class="a5-live a5-freed" x="110" y="90" width="60" height="72" rx="8"/>
<rect class="a5-live a5-alloc" x="530" y="90" width="60" height="72" rx="8"/>
<rect class="a5-live a5-alloc" x="600" y="90" width="60" height="72" rx="8"/>
<rect class="a5-win a5-mv" x="176" y="82" width="288" height="88" rx="10"/>
<text class="a5-lbl" x="70"  y="200">freed</text>
<text class="a5-lbl" x="140" y="200">freed</text>
<text class="a5-lbl" x="560" y="200">alloc</text>
<text class="a5-lbl" x="630" y="200">alloc</text>
<text class="a5-key" x="360" y="238">blocks behind the window return to free_block_queue; blocks ahead are allocated on demand</text>
</svg>
<figcaption>The window sweeps forward: trailing blocks are freed back to the shared pool while new blocks are allocated ahead, so a sliding-window layer's footprint stays bounded at W.</figcaption>
</figure>

The animation shows the mechanism. As decoding advances and the window slides forward, blocks behind the window hold tokens no future step will attend to — so the moment the window's trailing edge passes them, they are freed back to the shared `free_block_queue`. Blocks ahead are allocated on demand as new tokens arrive. The window's *width* is fixed at $W$, so the number of live blocks a sliding-window layer holds is bounded at roughly $\lceil W / \text{block\_size} \rceil$ no matter how long the sequence runs. This is the single most important behavioral difference from full attention: a full-attention layer's block count is monotonically increasing; a sliding-window layer's block count plateaus and then holds steady, recycling one block into the pool for every new block it takes out.

Here is the shape of that logic:

```python
# vllm/v1/core/single_type_kv_cache_manager.py (paraphrased)
class SlidingWindowManager(SingleTypeKVCacheManager):
    def get_num_blocks_to_allocate(self, request, num_tokens) -> int:
        # Only need blocks covering the last `sliding_window` tokens.
        needed = cdiv(min(num_tokens, self.sliding_window), self.block_size)
        return max(0, needed - len(self.req_to_blocks[request.id]))

    def remove_useless_blocks(self, request, num_computed_tokens) -> list[KVCacheBlock]:
        # Blocks entirely behind the window's trailing edge will never be read
        # again — free them mid-request so the pool can reuse them.
        first_useful_token = num_computed_tokens - self.sliding_window + 1
        first_useful_block = first_useful_token // self.block_size
        freed = self.req_to_blocks[request.id][:first_useful_block]
        # ... detach freed blocks and hand them back to the BlockPool ...
        return freed
```

The `ChunkedLocalAttentionManager` is a close cousin of the sliding-window manager, and the difference is worth pinning down because it trips people up. A sliding window is *continuous*: every token attends to the previous $W$ tokens, a window that moves one position per step. Chunked-local attention — Llama 4's local layers — is *discrete*: the sequence is partitioned into fixed, non-overlapping chunks (8,192 tokens in Llama 4), and a token attends only within its own chunk. The memory consequence differs subtly. A sliding-window layer's live block set slides smoothly; a chunked-local layer's live block set resets at each chunk boundary — when decoding crosses from one chunk into the next, the entire previous chunk becomes unreadable at once and its blocks are freed in a batch. Both are bounded, both recycle blocks into the shared pool, but the freeing *cadence* is different, and a capacity model that assumes smooth sliding will mis-predict the sawtooth footprint of a chunked-local model. This is why the two get separate manager subclasses rather than sharing one with a parameter.

The `MambaManager` is different again: a Mamba layer holds exactly one state per request, so it allocates one "block" (sized to hold the full conv + SSM state) at the start and holds it for the request's lifetime. There is no per-token growth and nothing to free mid-request. The interesting complication is that a Mamba state is updated *in place* — which, as we will see in §5 and §7, makes prefix caching genuinely hard, because you cannot "roll back" an in-place state the way you can drop the tail of a KV cache. In current vLLM, Mamba prefix caching is a work in progress; once it lands, "Mamba layer + full-attention layer" models slot into the same "full attention + X" machinery as sliding-window models. The [Nemotron-H hybrid Mamba-Transformer](/blog/machine-learning/large-language-model/nemotron-h-hybrid-mamba-transformer) architecture is exactly this kind of model — mostly Mamba with a few attention layers — and it is legal precisely because it *has* at least one full-attention layer.

The two specializations round out the picture. `SinkFullAttentionManager` extends the full-attention manager to pin a handful of sink tokens that must never be evicted — the mechanism gpt-oss relies on, discussed in case study 2. `RSWAManager` ("rotating sliding-window attention") subclasses `FullAttentionManager` rather than `SlidingWindowManager`, a hint that some rotating-window variants are handled by keeping the full set of blocks and rotating within them rather than freeing behind a moving edge. The taxonomy exists because real models keep inventing new bounded-attention variants, and each needs a manager that knows exactly which blocks its layers will and will not read again.

### Second-order optimization: the window boundary is not the block boundary

A gotcha that produces confusing off-by-one memory behavior: the sliding window is measured in *tokens*, but freeing happens in *blocks*. A block can only be freed when *all* of its tokens are behind the window's trailing edge, not when the window's edge merely enters it. With `block_size = 16` and `sliding_window = 4096`, a layer holds not ${4096 / 16 = 256}$ blocks but ${257}$ — the 256 blocks fully inside the window plus the one straddling its trailing edge, which cannot be freed until the window advances past its last token. It is a small overhead, but if you are doing capacity math by hand and your measured block count is consistently one higher than your formula, this is why. The manager is conservative on purpose: freeing a block one token too early would corrupt the attention computation.

## 5. Prefix caching with mixed rules

**Senior rule of thumb: a hybrid cache hit is not a single number — it is the reconciliation of two different rules about what "cached" means, and the reconciliation can be shorter than either rule alone would allow.**

[Prefix caching](/blog/machine-learning/model-serving/prefix-caching-and-radixattention) is the optimization where two requests sharing a prefix — a common system prompt, a shared few-shot preamble, a RAG document reused across questions — reuse the same cached KV blocks instead of recomputing them. For a homogeneous full-attention model the rule is simple: a prefix of length $L$ hits if the blocks holding tokens ${0}$ through $L-1$ are all still in the cache. Hybrid models break this simplicity, because different layer types have different rules for what "still cached" means.

The two rules are:

- **Full attention:** a prefix of length $L$ is a cache hit only if *every* token from ${0}$ to $L-1$ is still cached. Full attention reads all of history, so it needs all of history present.
- **Sliding window:** a prefix of length $L$ is a cache hit if the last $W - 1$ tokens before position $L$ are still cached. A sliding-window layer at position $L$ only ever reads back $W$ tokens, so it does not care whether tokens far in the past were evicted — it never would have read them.

This asymmetry has a beautiful and non-obvious consequence.

![Full attention needs every prefix token cached; sliding window needs only the last W, so a shared prefix still hits even after the sliding-window layer freed its early blocks](/imgs/blogs/hybrid-kv-cache-manager-6.webp)

The figure walks through it. A previous request cached a 10-block prefix `t0`–`t9`. Its full-attention layers, which never free anything mid-request, still hold all ten blocks. Its sliding-window layers (with $W$ covering the last four blocks) *freed* the early blocks `t0`–`t5` while that request was running — they left the window and went back to the pool. Now a new request arrives sharing the same 10-token prefix. Can it hit the cache?

For the full-attention group: yes, all ten tokens are cached, so it hits at length 10. For the sliding-window group: it needs only the last few blocks present at position 10, and `t6`–`t9` are still cached — so it *also* hits at length 10, even though it dropped `t0`–`t5` long ago. The reconciled hit is the full ten tokens. The sliding-window layer registers a full-length hit *despite having freed most of the prefix*, because it only ever needed the tail. This is the payoff of the per-type rule: a naive engine that demanded "all layers have all prefix tokens cached" would have reported a *miss* here (the sliding-window layer is missing `t0`–`t5`) and needlessly recomputed the whole prefix.

The reconciliation is done by `HybridKVCacheCoordinator.find_longest_cache_hit`, described in vLLM's source as "an iterative fixed-point algorithm." It computes the full-attention hit length, then checks the sliding-window rule within that length, and settles on the longest prefix where *both* groups can hit simultaneously:

```python
# vllm/v1/core/kv_cache_coordinator.py (paraphrased)
class HybridKVCacheCoordinator(KVCacheCoordinator):
    def find_longest_cache_hit(self, block_hashes, max_len):
        # 1. Longest prefix the FULL-ATTENTION group can serve: every block present.
        full_hit = self.full_manager.longest_cache_hit(block_hashes, max_len)
        # 2. Within that, the sliding-window group only needs the last W-1 tokens
        #    of the candidate prefix present. Scan right-to-left from full_hit.
        sw_hit = self.sw_manager.longest_cache_hit(block_hashes, full_hit)
        # 3. The reusable prefix is where BOTH hold — the intersection.
        reconciled = min(full_hit, sw_hit)
        return reconciled
```

The subtle production consequence: because the two groups can hit at *different* lengths, your observability needs to think in terms of a reconciled hit, not a single boolean. When a dashboard shows a fractional or surprising cache-hit rate on a hybrid model, it is almost always because one group hit and another did not, and the reconciled number sits between them (see case study 3).

### How block hashes stay correct across groups

There is a correctness question hiding under the prefix cache: if all groups draw from one pool and the pool's `cached_block_hash_to_block` map is shared, what stops a full-attention block that happens to hold tokens `t0`–`t15` from being mistakenly handed to a sliding-window layer looking for the same tokens? The two blocks hold the same *token ids* but represent KV computed under different attention masks — reusing one for the other would be silently wrong.

The answer is that a block's cache key — its `BlockHash` — is not just a hash of the token ids. It incorporates the block's position in the sequence and the group's identity, so that "tokens `t0`–`t15` as computed by a full-attention layer" and "tokens `t0`–`t15` as computed by a sliding-window layer" hash to *different* keys and never collide in the shared `BlockHashToBlockMap`. Prefix caching then proceeds per group: the `find_longest_cache_hit` reconciliation asks each group's manager to look up *its own* block hashes, and only blocks that match a group's own keys are reused for that group. The pool is shared at the level of physical blocks, but the cache index is partitioned by group at the level of hashes. This is what lets one `BlockPool` serve heterogeneous groups without ever cross-wiring their caches — the memory is fungible, but the meaning of what is stored in it is not, and the hash carries that meaning.

```python
# Conceptual: a block's cache identity includes more than its tokens.
def block_hash(token_ids, block_position, group_id, prev_block_hash):
    # Chaining prev_block_hash makes the key depend on the WHOLE prefix, and
    # group_id keeps a full-attention block distinct from a sliding-window
    # block that holds the same tokens under a different mask.
    return hash((prev_block_hash, block_position, group_id, tuple(token_ids)))
```

### Second-order optimization: why Mamba prefix caching is genuinely hard

Sliding-window prefix caching works because a sliding-window layer's state at position $L$ is fully determined by the last $W$ tokens — it is a pure function of a bounded, cached suffix. A Mamba layer's state is *not* a function of a bounded suffix; it is the accumulated result of every update from token 0 to the current position, applied in place. You cannot reconstruct it from a cached window, and you cannot "un-apply" the tail to roll back to an earlier prefix. This is why vLLM lists Mamba prefix caching as work in progress, and it is exactly the problem SGLang solves with checkpointing in §7. The takeaway: sliding-window is "stateless given a window," Mamba is "stateful and irreversible," and that difference is why one had prefix caching on day one and the other did not.

## 6. The coordinator hierarchy and the "full attention + X" restriction

**Senior rule of thumb: the number of distinct attention types in your model decides which coordinator you get — and if you have more than two, you silently lose prefix caching.**

Sitting above the per-group managers is the `KVCacheCoordinator`, and vLLM ships three of them. Which one a model gets is a function of how many distinct attention types it has.

![The count of distinct attention types picks the coordinator: one type routes to Unitary, two to Hybrid, more than two to the no-prefix-cache fallback](/imgs/blogs/hybrid-kv-cache-manager-7.webp)

The decision, as the tree shows:

- **One spec type** (a homogeneous transformer — all full attention, or all sliding window) routes to `UnitaryKVCacheCoordinator`. One group, one block table, the classic fast path. This is what every model got before hybrid support existed.
- **Two types — full attention plus exactly one other** (`full + X`, where X is sliding window, chunked-local, or Mamba) routes to `HybridKVCacheCoordinator`. This is the case the whole design is built for, and it carries the two-group reconciliation logic from §5.
- **More than two types, or no full-attention layer at all** routes to `KVCacheCoordinatorNoPrefixCache`. The model still runs — memory is allocated per type — but prefix caching is turned off entirely.

That last branch is the "full attention + X" restriction stated as code, and it is the single most important operational fact in this whole article. vLLM's own design doc is blunt: the algorithm "doesn't support models without full attention layers, and models with more than 2 types of attention." The `HybridKVCacheCoordinator.__init__` even asserts it: it "requires at least two attention groups" and is built for exactly the two-type case. If your model has three attention types — say full attention *and* sliding window *and* Mamba, all in one stack — you fall off the supported path and land in `KVCacheCoordinatorNoPrefixCache`, where prefix caching is simply gone.

Why the restriction? Because the reconciliation in §5 is a two-way intersection, and generalizing it to an arbitrary number of interacting cache-hit rules — each with its own notion of "what must be present" — is a genuinely harder algorithm that the current implementation does not attempt. Two types cover the overwhelming majority of shipping hybrid models (full + sliding window, full + Mamba), so the restriction is pragmatic rather than fundamental. But it is a cliff, not a slope: you either have a supported topology and get prefix caching, or you do not and lose it entirely, with no middle ground. Knowing which side of that cliff your model sits on is the difference between a healthy cache-hit rate and a mysterious flatline (case study 10).

```python
# vllm/v1/core/kv_cache_coordinator.py (paraphrased factory)
def get_kv_cache_coordinator(kv_cache_config, ...):
    if not enable_caching:
        return KVCacheCoordinatorNoPrefixCache(...)
    if len(kv_cache_config.kv_cache_groups) == 1:
        return UnitaryKVCacheCoordinator(...)   # homogeneous
    return HybridKVCacheCoordinator(...)         # full + X, exactly two groups
```

### Second-order optimization: the manager subclasses encode real architectures

The `SingleTypeKVCacheManager` family is not abstract taxonomy — each subclass exists because a real model needed it. `SinkFullAttentionManager` exists because gpt-oss keeps learned attention sinks that must never be evicted even as its banded window slides (case study 2). `ChunkedLocalAttentionManager` exists because Llama 4's local layers attend in fixed 8,192-token chunks rather than a smoothly sliding window (case study 4). `CrossAttentionManager` exists for encoder-decoder models whose cross-attention KV is fixed to the encoder output. When you serve a new hybrid model, the first question is "which manager subclass claims each of its layer types" — and if the answer is "none of them," that is your signal that the model needs new support before it will serve efficiently.

## 7. A different design: SGLang's separate pools

**Senior rule of thumb: vLLM unifies heterogeneous layers into one pool; SGLang isolates them into separate elastic pools — same conflict, opposite philosophy, and the tradeoffs are instructive.**

vLLM's answer to heterogeneous memory demand is unification: force every group to one page size, pool everything, let one free queue arbitrate. SGLang takes the opposite path — it keeps the pools *separate* and makes them elastic. Understanding the contrast sharpens what each is really buying.

![vLLM unifies heterogeneous layers into one pool; SGLang isolates them into separate elastic pools — two philosophies for the same storage conflict](/imgs/blogs/hybrid-kv-cache-manager-8.webp)

The matrix lays out the two designs across the dimensions that matter. SGLang, as described in its team's "Hybrid Models Meet SGLang" work, maintains physically separate memory pools: a token-level paged KV pool for full-attention layers, and a request-level pool for Mamba states. Two data structures make this work. `HybridReqToTokenPool` binds a Mamba state to a request so their lifetimes align — the state is allocated when the request starts and freed when it ends, request-granularity rather than token-granularity. `HybridLinearKVPool` maps a model's logical layer indices to actual indices in the KV pool, so that linear (Mamba) layers are simply skipped when allocating token-level KV — no wasted KV allocation for a layer that does not have per-token keys and values. The ratio between the two pools is set at startup with `--mamba-full-memory-ratio`.

The obvious objection to separate pools is the static-partition fragmentation from §3 — guess the ratio wrong and one pool starves while the other idles. SGLang's answer is an **elastic** pool built on CUDA virtual memory management: it pre-reserves an oversized virtual address space for each pool and maps physical GPU pages to it on demand, with a centralized controller that shrinks an underutilized pool and grows a saturated one while holding the total GPU budget fixed. It is a more moving parts than vLLM's single-pool design, but it recovers the "memory breathes between types" property that a naive static partition lacks — achieving dynamically what vLLM achieves structurally.

The place the two designs differ most sharply is Mamba prefix caching. Where vLLM lists it as work in progress, SGLang ships `MambaRadixCache` — a [radix-tree](/blog/machine-learning/model-serving/prefix-caching-and-radixattention) prefix cache adapted to state-space models by **checkpointing**. Because a Mamba state updates in place and cannot be rolled back, SGLang does not try to. Instead it snapshots the state:

```python
# SGLang MambaRadixCache (conceptual)
def match_prefix(self, token_ids):
    node = self.tree.longest_matching_node(token_ids)
    # KV blocks are referenced directly; the Mamba state, being in-place and
    # irreversible, is COPIED out of the tree as a fresh checkpoint.
    mamba_state = node.mamba_state.clone()   # forced snapshot
    return node.kv_blocks, mamba_state

def insert(self, token_ids, kv_blocks, mamba_state):
    # Fork a checkpoint of the current request's state INTO the tree.
    self.tree.insert(token_ids, kv_blocks, mamba_state.clone())

def evict(self):
    # Two independent LRU lists: KV cache evicts leaf->root; Mamba states
    # can evict from any node (no ordering constraint).
    self.evict_kv_lru()
    self.evict_mamba_lru()
```

The design has three moving parts worth naming: matching **copies** a checkpoint of the Mamba state for the new request (you cannot share an in-place state), insertion **forks** a checkpoint into the tree, and eviction runs **two separate LRU lists** — one for KV blocks, one for Mamba states — because the two have different lifetimes and different eviction constraints. It is more machinery than vLLM's single global LRU, but it is what buys Mamba prefix caching today rather than "work in progress."

The same isolation philosophy extends to speculative decoding, which is otherwise fundamentally incompatible with in-place SSM state (a rejected draft token cannot be rolled back). SGLang gives each candidate draft token its own **isolated Mamba cache slot** — a physical state sandbox — and on verification promotes the accepted slot to be the new main state rather than attempting a rollback. On an H200 serving Qwen3-Next-80B-A3B-Instruct-FP8, this pushes speculative decoding to 324.57 tokens/sec with an average accepted length of 4.231 at an MTP window of 4 with top-8 drafts, and prefix matching cuts time-to-first-token to 57.63% of the no-cache baseline on a shared-system-prompt workload.

The isolation also pays off in [prefill–decode disaggregation](/blog/machine-learning/model-serving/prefill-decode-disaggregation), where prefill and decode run on separate instances and the KV cache computed during prefill must be shipped to the decode instance. For a pure full-attention model this is a stream of block-indexed KV pages. For a hybrid model, the Mamba state has to travel too — and because a state is a single contiguous per-request object rather than a growing sequence of pages, SGLang transfers it *atomically* over a dedicated state channel, distinct from the incremental page-by-page KV transfer. The decode instance pre-allocates both the KV page slots and the Mamba state slot before receiving anything, then drops the state into its reserved location. The general recipe SGLang documents for adding a new hybrid pool is three steps: expose the state buffers (pointers, sizes, item lengths) so the transfer layer can register them; define the state-index logic (a single index per request for Mamba, page indices for sliding-window, full sequences for others); and register the state type in the KV manager. That recipe is the separate-pool philosophy taken to its conclusion — every layer type is a first-class citizen with its own storage, its own transfer path, and its own lifecycle — and it is the mirror image of vLLM's "make everything look the same so one pool can hold it all."

Neither design is strictly better. vLLM's single pool is simpler, has one eviction policy, and gets sliding-window prefix caching for free — but it pays a padding tax, is limited to two attention types, and is still catching up on Mamba. SGLang's separate pools handle Mamba prefix caching and speculative decoding today and flex per-type memory dynamically — but at the cost of more infrastructure (CUDA VMM, dual LRUs, a central controller) and more configuration surface. If you are choosing an engine for a specific hybrid model, the question is not "which is better" but "which one already supports *my* model's exact topology, today, with prefix caching on."

## 8. Operating it: configuration, flags, and observability

**Senior rule of thumb: the hybrid manager is mostly automatic, and the few knobs you have are about turning it off, not tuning it — so your job is to read what it decided, not to hand-configure it.**

The Hybrid KV Cache Manager is not something you configure layer by layer. When you serve a hybrid model, vLLM reads the model's config — Gemma 3's `sliding_window` and `sliding_window_pattern`, Llama 4's chunked-attention settings, a Mamba model's state shapes — builds the `KVCacheSpec` for each layer, forms the groups, and picks the coordinator, all automatically. What you can do is observe and, if necessary, disable.

```bash
# Serve Gemma-3-27B; the hybrid manager forms 7 groups automatically.
vllm serve google/gemma-3-27b-it \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.90

# The escape hatch: if a model lands on an unsupported topology (>2 attention
# types) or you hit a hybrid-allocator bug, disabling prefix caching forces
# the KVCacheCoordinatorNoPrefixCache path — correct, just without reuse.
vllm serve <some-3-attention-type-model> \
  --no-enable-prefix-caching
```

There is no `--full-attention-pool-size` or `--sliding-window-pool-size` to get wrong, and that is by design — the whole point of the single shared pool is that you do *not* pre-partition it. The one place a flag matters is the escape hatch: `--no-enable-prefix-caching` routes you to `KVCacheCoordinatorNoPrefixCache`, which is the correct thing to reach for when a model's topology is unsupported or you have hit an allocator bug and need a known-good fallback while you file the issue.

For observability, the numbers to watch are the ones the old mental model does not have. On a hybrid model, track the reconciled prefix-cache hit rate (not a per-layer boolean), the number of KV cache groups the model was carved into (a sanity check that grouping did what you expected), and the block-pool utilization over time (which, on a sliding-window model, should plateau rather than climb, because the sliding-window groups recycle blocks). A climbing block-pool utilization on a model you *thought* was mostly sliding-window is a strong signal that something forced full-attention specs onto layers that should have been bounded — exactly the legacy-allocator failure of case study 1.

```python
# What to log per model at startup — the grouping decision is the thing to verify.
# (Pulled from the KVCacheConfig the engine built.)
for i, group in enumerate(kv_cache_config.kv_cache_groups):
    spec = group.kv_cache_spec
    print(f"group {i}: {type(spec).__name__}  "
          f"layers={len(group.layer_names)}  "
          f"page_size_bytes={spec.page_size_bytes}")
# A healthy Gemma-3 log shows one FullAttentionSpec group and several
# SlidingWindowSpec groups, ALL with identical page_size_bytes.
```

### Second-order optimization: verify the grouping matches the architecture

The most valuable thirty seconds you can spend on a new hybrid deployment is reading that startup log and confirming the group count and spec types match the model card. If Gemma-3-27B logs a single `FullAttentionSpec` group instead of one full group plus several sliding-window groups, the hybrid path did not engage — you are serving it as a dense transformer and paying full-attention memory for every layer. If a Mamba model logs a `KVCacheCoordinatorNoPrefixCache`, you have silently lost prefix caching. These are not runtime errors; the model serves correctly either way. They are *efficiency* regressions that only the startup log reveals, and catching them there is far cheaper than discovering them as a capacity shortfall under load.

## Case studies from production

The following incidents are composites drawn from the failure modes that recur when hybrid models meet serving engines. Model names, layer counts, and flags are real; the specific timelines are illustrative of the class of problem.

### 1. Gemma-3-27B — the OOM that started it all

A team migrated a Gemma-3-27B deployment from a 32K context limit to 128K to support long-document summarization. On the older vLLM build, every request over roughly 60K tokens hit an out-of-memory error, even though the GPU had ample headroom for a 27B model at that length. The wrong first hypothesis was weight bloat or an activation-memory spike from the longer sequences. The actual root cause was the pre-hybrid allocator: it modeled all 62 layers with a `FullAttentionSpec`, reserving $O(N)$ KV memory for the 52 sliding-window layers that should have been bounded at $O(W)$. At 128K context, that meant reserving full-attention memory for layers that would never cache more than a 1,024-token window — roughly a fivefold overshoot on the dominant layer type. Upgrading to a build with the Hybrid KV Cache Manager carved the model into one full-attention group and six sliding-window groups; the sliding-window groups' footprint stopped scaling with context, and the same GPU that OOM'd at 60K served comfortably past 128K. The lesson: on a sliding-window-heavy model, your context ceiling is set by the *full-attention* layers, and an allocator that does not know the difference will quietly quintuple your memory bill.

### 2. gpt-oss-120B — the vanishing attention sink

A deployment of gpt-oss-120B produced fluent output for short prompts but degraded into repetitive garbage once conversations passed a few thousand tokens. The model alternates full-attention layers with 128-token banded (sliding-window) layers, and — critically — uses *learned attention sinks*: the first few tokens carry a bias the softmax relies on to park probability mass. The first hypothesis was a sampling bug or a bad chat template. The real cause was that the banded layers' sliding-window eviction was dropping the sink tokens the moment the window advanced past them, and without the sink the attention distribution in the banded layers destabilized exactly as the StreamingLLM work predicted. The fix was routing those layers through `SinkFullAttentionManager`, which pins the sink tokens in the cache and slides the window over the rest. The lesson: sliding-window and attention-sink are two different properties, and a manager that handles the window but not the sink will produce a model that works until precisely the moment the first real token falls out of the window.

### 3. Ministral-8B — the "partial hit" dashboard mystery

An on-call engineer was paged because a Ministral-8B service's prefix-cache-hit metric had dropped from a comfortable 0.9 to a baffling 0.55 overnight, with no deploy and no traffic change. The team burned an hour looking for a cache-invalidation bug. There was none. Ministral interleaves sliding-window and full-attention layers, and the metric was averaging two different hit rates: the full-attention group was hitting a shared 8K system prompt at nearly 100%, while the sliding-window group — which only needs the last $W$ tokens present and had freed the rest — was "hitting" at a lower reconciled length for a subset of requests whose prompts exceeded the window in a particular way. The number was not wrong; it was *two numbers averaged into one*. The fix was observability, not code: split the dashboard into per-group reconciled hit rates. The lesson: on a hybrid model, a single scalar cache-hit rate is a lie of composition — you need to see the groups separately or you will chase phantom regressions.

### 4. Llama-4 Scout — the 10M-context ceiling that wasn't the chunks

A team deploying Llama-4 Scout to exploit its advertised 10M-token context found requests failing well short of that — around 500K tokens — with OOM. Because Scout's headline feature is chunked-local attention (8,192-token chunks via its iRoPE design), the natural assumption was that the chunked layers were the memory hog. They were not: chunked-local layers, like sliding-window layers, are bounded and cheap. The culprit was the NoPE layers — every fourth layer is a full-attention layer with no positional encoding, precisely so it can see the entire context — and *those* layers grow $O(N)$. At 500K tokens, the one-in-four full-attention layers, not the three-in-four chunked layers, were consuming the memory. The `ChunkedLocalAttentionManager` was doing its job perfectly; the ceiling was the full-attention group. The fix was capacity math that counted the full-attention layers correctly and, for the longest requests, KV quantization on the full-attention group. The lesson is the same shape as case study 1 from a different architecture: in a hybrid model, the unbounded layers set the ceiling, and they are often not the layers the marketing talks about.

### 5. Jamba-1.5 — RAG with zero prefix-cache benefit

A retrieval-augmented deployment of Jamba-1.5 (AI21's Mamba-plus-attention MoE) was designed around a large shared system prompt reused across thousands of queries — the textbook case for prefix caching. But measured cache hit rates were near zero, and time-to-first-token never improved on repeated prefixes. The team suspected their prompt-construction code was accidentally varying the prefix. It wasn't. The model contains Mamba layers, and Mamba prefix caching was, on that engine build, a work in progress — so the presence of Mamba layers routed the model down a path where prefix reuse for the state-space layers was not available, and the reconciled benefit collapsed. The options were: accept no prefix caching on that engine, switch to an engine with `MambaRadixCache`-style checkpointing (SGLang), or wait for Mamba prefix-cache support to land in vLLM. The lesson: "this model supports prefix caching" is a per-*engine*, per-*layer-type* claim, not a property of the model — and Mamba's in-place state is the specific thing that makes it hard.

### 6. Bamba-9B — the starved pool

A team running Bamba-9B (a Mamba-2 plus full-attention hybrid) on SGLang saw throughput collapse under a specific workload: many short requests interleaved with a few very long ones. Because SGLang splits memory into a request-level Mamba pool and a token-level KV pool sized by `--mamba-full-memory-ratio`, and the team had tuned that ratio for a long-context workload, the many short requests exhausted the Mamba-state pool (one state per request, and lots of requests) while the KV pool sat half-empty. The wrong hypothesis was a scheduler fairness bug. The root cause was static pool sizing meeting a workload whose request-shape distribution differed from the tuning assumption. Enabling SGLang's elastic pool (CUDA-VMM remapping) let the Mamba pool grow into the KV pool's slack, and throughput recovered. The lesson: separate-pool designs move the fragmentation risk from "within a pool" to "between pools," and the fix is either dynamic remapping or a ratio tuned to the *actual* request-shape distribution — which is exactly the fragmentation that vLLM's single shared pool sidesteps structurally.

### 7. Nemotron-H-56B — mostly-Mamba, still legal

A team bringing up NVIDIA's Nemotron-H-56B — an architecture that is overwhelmingly Mamba-2 with a small number of interspersed full-attention layers — worried during capacity planning that a "mostly Mamba" model would fall outside the "full attention + X" support and lose prefix caching. It did not, and the reason is instructive: the restriction is not "mostly full attention," it is "*has* a full-attention layer and *exactly one* other type." Nemotron-H has full-attention layers (few, but present) and Mamba layers (many) — two types, one of which is full — so it routes to `HybridKVCacheCoordinator`, not the no-prefix-cache fallback. The full-attention group is small, so it contributes little to the total KV footprint, and the Mamba groups dominate. The contrast that makes the point: a *pure* Mamba model, with no full-attention layer at all, would fall into `KVCacheCoordinatorNoPrefixCache`, because the reconciliation logic is anchored on the full-attention group. The lesson: what matters for support is the *set* of attention types, not their proportions — one full-attention layer is enough to keep you on the supported path.

### 8. Qwen3-Next-80B-A3B — the SGLang counter-design in production

A team chose Qwen3-Next-80B-A3B specifically for cheap long-context serving — the model runs Gated DeltaNet linear attention for three of every four layers and Gated (full) attention for the fourth, an ultra-sparse MoE with only 3B active parameters. On SGLang, they leaned on the full counter-design from §7: `MambaRadixCache`-style checkpointing gave them prefix caching on the linear-attention layers, dual-LRU eviction kept KV and state lifetimes separate, and per-draft-token state sandboxes made speculative decoding work despite the in-place linear-attention state. The measured payoff on an H200 with the FP8 build: speculative decoding reached 324.57 tokens/sec at 4.231 average accepted length (MTP window 4, top-8), and prefix matching cut time-to-first-token to 57.63% of baseline on a shared-prompt workload. The lesson: when a model is *dominated* by an efficient layer type, the engine's handling of that type — its prefix caching, its speculative-decoding compatibility — is not a detail, it is the entire performance story, and it is worth choosing the engine around it.

### 9. MiniMax-Text-01 — when padding is the price of uniformity

A team serving MiniMax-Text-01 — 7 lightning (linear) attention layers for every 1 softmax (full) attention layer, an extreme 7:1 ratio — on vLLM's unified-pool design noticed a small but persistent memory overhead they could not explain from the model config alone. The cause was the group-padding rule from §2. With one full-attention layer per 8-layer block and seven linear-attention layers, the group sizes are lopsided, and forcing every group to the same page size means the smaller type's last group carries padding. On a 7:1 model that padding is more visible than on Gemma-3's 5:1, because the type imbalance is larger. The overhead was real but modest — a few percent — and it is the deliberate trade the unified pool makes: waste a sliver of memory to padding so that every free block fits every group and one LRU governs everything. The lesson: the padding tax scales with how imbalanced your layer-type ratio is, and on the most extreme hybrid models it is worth measuring — though it is almost always smaller than the fragmentation a static partition would cost instead.

### 10. The three-attention-type model — silent prefix-cache death

An experimental internal model combined full attention, sliding-window attention, *and* Mamba layers — three distinct attention types in one stack — because the research team wanted the long-context properties of all three. It served correctly, benchmarks looked fine in isolation, but under production traffic with heavy prefix sharing the cache-hit rate sat flat at zero and nobody could find the invalidation bug. There was none. Three attention types exceed the "full attention + X" two-type limit, so the model routed to `KVCacheCoordinatorNoPrefixCache` — prefix caching was not broken, it was *off by construction*. The model was on the wrong side of the support cliff from §6. The fixes were architectural (drop to two attention types), engine-level (wait for or contribute multi-type reconciliation), or operational (accept no prefix caching and provision for the recompute cost). The lesson: the two-type restriction is not a performance knob, it is a binary — cross it and you lose prefix caching entirely, silently, with a model that still passes every functional test.

### 11. The sliding-window prefix-reuse cliff

A chat service on a sliding-window model expected near-perfect prefix reuse for its long shared system prompt, but reuse fell off sharply once the system prompt grew *longer than the window*. The team assumed a cache-size problem and threw more GPU memory at it, which did nothing. The real mechanism is the sliding-window cache-hit rule from §5: a sliding-window layer can only register a hit for the last $W - 1$ tokens before the query position. When the shared prefix is longer than the window, the portion of it beyond the window's reach on the sliding-window layers simply cannot be reused *by those layers* — the full-attention group still hits the whole prefix, but the reconciled hit is bounded by what the sliding-window group can serve for the specific query offsets in play. More memory does not help because the limit is the window, not the cache capacity. The fix was understanding the rule and setting expectations: on a sliding-window model, prefix reuse for the sliding-window layers is inherently capped by $W$, and the reconciled benefit reflects that. The lesson: the window is not just a memory bound, it is a *cache-reuse* bound, and no amount of memory buys back reuse the architecture forbids.

### 12. The many-group model — death by block tables

A team profiled a hybrid deployment expecting a decode-throughput win from the hybrid manager's memory savings and instead measured a *regression* in tokens-per-second at high concurrency, even though the model now fit far more requests than before. Memory had improved exactly as promised; latency had not. The wrong hypothesis was a kernel or attention-backend regression. The root cause was the group count. Because a request maintains one block table per KV cache group, and the scheduler walks those tables every step for every running request, a model carved into many groups pays more per-step scheduling overhead — and at high concurrency, with hundreds of running requests, that per-step cost is on the critical path of every decode token. The memory win was real; it had simply been partly eaten by metadata bookkeeping the team had not accounted for. The mitigations were the ones §2 anticipates: keep the group count as small as the padding rule allows (vLLM already does this by choosing the largest group size), and, where a workload is latency-sensitive, weigh the memory savings against the scheduling cost rather than assuming the hybrid path is free. The lesson: the hybrid manager buys concurrency, not per-token speed, and on a model with many groups the per-step cost of more block tables is a real, measurable line item — one that only shows up when you profile decode latency separately from memory footprint.

## When to reach for the Hybrid KV Cache Manager — and when not to

You do not usually "reach for" the hybrid manager the way you reach for a config flag; you inherit it the moment you serve a hybrid model. But there is a real set of decisions around it.

**Lean into the hybrid path when:**

- You are serving a **sliding-window model** (Gemma 2/3, Ministral, gpt-oss) at long context — this is where the memory savings are largest and the payoff over a uniform allocator is a multiple, not a margin.
- Your model is **dominated by efficient layers** (Qwen3-Next at 3:1, MiniMax at 7:1) — the more of the model is bounded or $O(1)$, the more the hybrid manager saves versus reserving full-attention memory everywhere.
- You run **long-lived streaming sessions** where sliding-window recycling keeps footprint flat — the single shared pool turns freed window blocks straight back into capacity for new requests.
- Your model is **"full attention + exactly one other type"** — the supported topology — and you want prefix caching, which the hybrid coordinator gives you for sliding-window layers automatically.

**Be cautious, or look at alternatives, when:**

- Your model has **more than two attention types** — you will land in `KVCacheCoordinatorNoPrefixCache` and lose prefix caching entirely; either simplify the architecture or budget for the recompute.
- You need **Mamba prefix caching or Mamba-compatible speculative decoding today** — vLLM lists these as work in progress; SGLang's `MambaRadixCache` and state-sandbox designs ship them now, so the engine choice matters.
- Your workload has a **highly variable request-shape distribution** across a Mamba-plus-attention model — a separate-pool engine with elastic remapping (SGLang) may adapt better than a static ratio, though vLLM's single pool avoids the between-pool fragmentation entirely.
- You are serving a **pure, homogeneous model** — a plain full-attention transformer — in which case there is nothing hybrid to manage; you get the `UnitaryKVCacheCoordinator` fast path and none of this machinery applies.

The through-line of every case study and every decision above is a single shift in mental model: a KV cache block is no longer a uniform thing that means the same for every layer. It is a fungible unit of memory that flows between layer types whose demands differ by orders of magnitude, governed by per-type rules about allocation, freeing, and reuse. The engines that serve hybrid models well are the ones that made that shift completely — and the failures are, almost without exception, the old uniform assumption leaking through somewhere it no longer holds.

## Further reading

- [Hybrid KV Cache Manager — vLLM design docs](https://docs.vllm.ai/en/stable/design/hybrid_kv_cache_manager/): the primary source for the group / coordinator / manager design and the "full attention + X" restriction.
- **Jenga: Effective Memory Management for Serving LLM with Heterogeneity**: the research lineage behind the shared-pool, equal-page-size approach.
- [Hybrid Models Meet SGLang: More than Full Attention](https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/): SGLang's separate-pool philosophy, `MambaRadixCache`, and the speculative-decoding state sandboxes.
- [KV cache optimization: managing the memory that caps LLM serving](/blog/machine-learning/model-serving/kv-cache-optimization) — the byte-level KV equation and the levers this design builds on.
- [Prefix caching and RadixAttention](/blog/machine-learning/model-serving/prefix-caching-and-radixattention) — the prefix-cache machinery the per-type hit rules extend.
- [Continuous batching and PagedAttention](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention) — the block-allocator foundation the shared pool sits on.
- [vLLM deep dive](/blog/machine-learning/model-serving/vllm-deep-dive) — where the KV cache manager fits in the larger engine.
