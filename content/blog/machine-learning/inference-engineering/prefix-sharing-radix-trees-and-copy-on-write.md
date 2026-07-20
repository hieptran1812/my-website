---
title: "Prefix sharing, radix trees, and copy-on-write: making system prompts nearly free"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "Two requests that start with the same tokens compute the same keys and values twice. Hash the blocks, share the physical memory, count the references, and copy only when a shared block is about to be written."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "kv-cache",
    "prefix-caching",
    "radix-tree",
    "copy-on-write",
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
readTime: 50
---

Your chat product has a 600-token system prompt. Every single request pays to compute keys and values for those 600 tokens, and every single request computes exactly the same numbers as the request before it. Your RAG endpoint stuffs the same 3,500-token policy document into every prompt. Same story. Your agent loop sends a 33-turn conversation back to the model on every turn, where turn 30 re-prefills roughly 67,000 tokens that turns 1 through 29 already computed, verbatim, and threw away.

That is not a rounding error. It is the dominant cost of most production LLM traffic. And the fix is not a new kernel or a bigger GPU — it is a dictionary.

![Two requests with different total lengths merging onto a shared prefix of 248 blocks and then branching into short private tails](/imgs/blogs/prefix-sharing-radix-trees-and-copy-on-write-1.webp)

This post writes `nanoserve/kv/prefix.py`: a hash-indexed prefix cache layered on top of the paged block allocator from [the paged KV cache post](/blog/machine-learning/inference-engineering/paged-kv-cache-implementing-blocks-and-a-block-table), plus reference counting, plus `find_longest_cache_hit`, plus the copy-on-write path that keeps it correct when two sequences that share memory start to diverge. Then it builds the radix-tree variant, derives exactly how much you save as a function of the shared fraction, and — the part most write-ups skip — derives how much you *lose* when nothing is shared, so you can decide honestly whether to turn it on.

As always in this series, and per the promise made in [the series introduction](/blog/machine-learning/inference-engineering/what-inference-engineering-is): **I have no GPU and have run none of this.** Every number below is derived from arithmetic I show you, cited from a paper or an engineering post with a link, or framed as a range you should expect when you run the script yourself. The results tables carry a `Source` column for that reason.

---

## 1. Why two identical prefixes produce identical keys and values

Start with the property that makes all of this legal, because if you do not believe the property you will not trust the cache.

A decoder-only transformer with causal masking computes, at layer $\ell$ and position $i$:

$$
\mathbf{k}^{(\ell)}_i = W_K^{(\ell)} \, \mathbf{x}^{(\ell)}_i, \qquad
\mathbf{v}^{(\ell)}_i = W_V^{(\ell)} \, \mathbf{x}^{(\ell)}_i
$$

where $\mathbf{x}^{(\ell)}_i$ is the hidden state entering layer $\ell$ at position $i$. The causal mask guarantees that $\mathbf{x}^{(\ell)}_i$ is a function of $\mathbf{x}^{(\ell-1)}_0, \dots, \mathbf{x}^{(\ell-1)}_i$ only — never of anything to the right. Unroll that all the way down to the embedding layer and you get the statement that matters:

$$
\bigl(\mathbf{k}^{(\ell)}_i,\ \mathbf{v}^{(\ell)}_i\bigr) = f^{(\ell)}\bigl(t_0, t_1, \dots, t_i\bigr)
$$

The key and value at position $i$ depend on the token ids at positions ${0}$ through $i$, on the weights, and on nothing else. Not on the tokens that follow. Not on what the other sequences in the batch are doing. Not on the sampling parameters — temperature and top-p touch logits at the very end, long after K and V are written. Not on how many tokens you eventually generate.

So: if request A and request B agree on their first $S$ tokens, then for every layer and every position $i \lt S$, A's and B's keys and values are the same tensor. Computing them twice is pure waste, and storing them twice is pure waste on top of that.

There is a caveat worth stating up front because it will bite someone. "The same tensor" here means *mathematically* the same. Floating-point kernels are not guaranteed to produce bit-identical results across different batch shapes — the vLLM team's write-up on [bitwise-consistent train/inference determinism](https://vllm.ai/blog/2025-11-10-bitwise-consistent-train-inference) (2025-11-10) traces this to batch-size-dependent kernel selection, noting that "kernels for high batch sizes parallelize on the batch dimension, kernels for low batch sizes parallelize within a single instance." A cached prefix was produced under whatever batch shape happened to be running when it was first computed. So reusing it can give you a last-bit-different key from what a fresh recompute would have produced, which can occasionally flip a sampled token. Enabling prefix caching therefore changes outputs slightly, in the same way that changing batch size does. That is a real property of the system, not a bug in your allocator, and you should know it before someone files a ticket titled "same prompt, different answer."

### The unit of sharing is the block, not the token

We are building on the paged allocator, so the KV cache is not one contiguous tensor per sequence. It is a pool of fixed-size **physical blocks**, each holding `block_size` token positions' worth of keys and values for every layer, plus a per-request **block table** mapping logical block index to physical block id. That indirection is what makes sharing almost free: to share a prefix, two block tables simply list the same physical block id. The attention kernel does not know or care.

For the running example — Llama-3.1-8B, 32 layers, 8 key-value heads (grouped-query attention), head dimension 128, bf16 — the per-token cost is:

$$
2 \times 32 \times 8 \times 128 \times 2\ \text{bytes} = 131{,}072\ \text{bytes} = 128\ \text{KB per token}
$$

The leading 2 is K and V; the trailing 2 is bytes per bf16 element. With `block_size = 16` (the default block size in vLLM, per the [Inside vLLM anatomy post](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm), 2025-09-05), one physical block across all layers is:

$$
16 \times 128\ \text{KB} = 2\ \text{MB}
$$

Hold on to that 2 MB figure — it is the unit of every copy we make later. It is also independently corroborated: vLLM's [KV cache offloading post](https://vllm.ai/blog/2026-01-08-kv-offloading-connector) (2026-01-08) reports that after a KV-layout change, Llama-3.1-8B physical block sizes moved from 32 KB to 2 MB. The 32 KB figure is one of K or V for one layer ($16 \times 8 \times 128 \times 2 = 32{,}768$ bytes); multiply by 2 for both tensors and by 32 layers and you land on exactly 2 MB.

---

## 2. Naming a block so that a name proves a prefix

To look up "have I already computed the keys and values for these tokens?", you need a key. The naive key is the token list itself, and it is wrong in an interesting way.

Suppose you hash just the 16 token ids inside block 3 and find a match. That tells you some other sequence had those same 16 tokens at *some* logical position. It does not tell you that the 48 tokens before them were the same — and if they were not, the cached keys and values are for a different context and are garbage. Reusing them would silently corrupt the output. You would not crash. You would just get subtly wrong answers, which is worse.

The fix is to fold the parent into the child. Per the [Inside vLLM anatomy post](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm), the block hash is a function of the previous block's hash, the current block's tokens, and optional metadata, computed with Python's builtin `hash` or with SHA-256. Written out:

$$
h_0 = H(\varnothing,\ t_{0:B},\ m), \qquad
h_j = H(h_{j-1},\ t_{jB:(j+1)B},\ m)
$$

Now $h_j$ is a fingerprint of the entire token prefix of length $(j+1)B$, not just of one block. A match on $h_j$ is a claim about everything to the left of it. That single design choice turns an $O(P)$ prefix comparison into an $O(1)$ dictionary probe, and it is the reason a hash map can compete with a trie at all.

![A left to right chain of four complete blocks each folding the previous hash into the next, ending with a nine-token tail that carries no hash](/imgs/blogs/prefix-sharing-radix-trees-and-copy-on-write-2.webp)

### Only complete blocks are cacheable

The second rule falls straight out of the first. A block hash covers exactly `block_size` tokens. A block that is half full does not have a stable identity — the next token appended to it changes what it contains, and therefore what its hash should be. So partial blocks are never inserted into the cache.

The consequence is a rounding loss the anatomy post states plainly: a prefix of length $S$ yields $\lfloor S/B \rfloor$ cacheable blocks, and the remaining $S \bmod B$ tokens are recomputed on every hit. If your shared prefix length is uniformly distributed relative to the block boundary, the expected loss is

$$
\mathbb{E}[\text{tokens lost}] = \frac{B-1}{2}
$$

With $B = 16$ that is 7.5 tokens — noise. With $B = 256$ it is 127.5 tokens, which for a 600-token system prompt means you lose over a fifth of your sharing to alignment. Block size is not a free knob: it trades hash-table pressure and metadata size against sharing granularity, and long-context configurations that push `--block-size 64` or `--block-size 256` for kernel efficiency are quietly paying for it in cache resolution.

Here is the hashing code, dropped into `nanoserve`:

```python
# nanoserve/kv/prefix.py
from __future__ import annotations

import hashlib
from typing import Iterable, NamedTuple, Sequence

BLOCK_SIZE = 16


class BlockKey(NamedTuple):
    """Identity of one complete block of tokens, chained to its parent."""

    block_hash: int
    token_ids: tuple[int, ...]


def _h(parent: int | None, tokens: tuple[int, ...], extra: object) -> int:
    """Fast path: Python's builtin tuple hash. 64-bit, process-local."""
    return hash((parent, tokens, extra))


def _h_sha256(parent: int | None, tokens: tuple[int, ...], extra: object) -> int:
    """Stable path: identical across processes and machines. ~10x slower."""
    m = hashlib.sha256()
    m.update(b"\x00" if parent is None else parent.to_bytes(32, "big", signed=False))
    for t in tokens:
        m.update(t.to_bytes(4, "big"))
    m.update(repr(extra).encode())
    return int.from_bytes(m.digest()[:16], "big")


def block_keys(
    token_ids: Sequence[int],
    block_size: int = BLOCK_SIZE,
    extra: object = None,
    stable: bool = False,
) -> list[BlockKey]:
    """Chain-hash every COMPLETE block of `token_ids`.

    The trailing `len(token_ids) % block_size` tokens produce no key: a
    partial block has no stable identity and can never be cached.
    """
    fn = _h_sha256 if stable else _h
    keys: list[BlockKey] = []
    parent: int | None = None
    n_complete = len(token_ids) // block_size
    for j in range(n_complete):
        chunk = tuple(token_ids[j * block_size : (j + 1) * block_size])
        parent = fn(parent, chunk, extra)
        keys.append(BlockKey(parent, chunk))
    return keys
```

Running it on a 73-token prompt:

```python
>>> keys = block_keys(list(range(73)))
>>> len(keys)                       # 73 // 16 == 4 complete blocks
4
>>> 73 - len(keys) * BLOCK_SIZE     # tokens that will always recompute
9
```

Four blocks cacheable, nine tokens permanently on the recompute path. That is the whole rounding story in two lines of output.

### Two hashes, two failure modes

`hash` versus SHA-256 is not a style preference; the two have different blast radii.

Python's builtin `hash` on a tuple of ints is deterministic within a process, fast (order of hundreds of nanoseconds for a 16-element tuple), and 64 bits wide. It is *not* stable across processes when strings are involved (`PYTHONHASHSEED` randomizes those), and it is not a cryptographic hash. For a single-process engine with a purely local cache, it is the right default.

The 64-bit width has a birthday bound. With $n$ distinct cached blocks, the probability that some pair collides is approximately

$$
P_{\text{coll}} \approx \frac{n^2}{2^{65}}
$$

At one million live blocks that is about $2.7 \times 10^{-8}$ — ignorable. At one hundred million blocks, which is the scale of a shared cluster-wide KV store, it rises to roughly $2.7 \times 10^{-4}$: a one-in-four-thousand chance that somewhere in your fleet, one request is served another request's keys and values. There is no crash and no log line. There is just a wrong answer.

Two defenses, and you should take at least one:

1. **Verify tokens on hit.** Store the block's token ids alongside the cached entry and compare them before accepting the match. That costs 16 integer comparisons and removes collisions as a correctness concern entirely. My implementation does this, and it is why `BlockKey` carries `token_ids`.
2. **Use a wider, stable hash** (SHA-256 truncated to 128 bits, or BLAKE3) the moment the cache crosses a process boundary. A distributed store needs the hash to mean the same thing on every node anyway, so this is forced. It is exactly why the vLLM anatomy post lists SHA-256 as an option, and why a cross-node design like [Mooncake Store](https://vllm.ai/blog/2026-05-06-mooncake-store) (2026-05-06) has a master server tracking block hashes as its central index.

---

## 3. The allocator, extended: reference counts and a hash map

Now the implementation. The paged allocator from the previous post owns a pool of physical blocks and hands them out. It needs three additions:

1. A `ref_count` per physical block — how many sequences currently list it in their block table.
2. A `cached_block_hash_to_block` map from block hash to physical block id.
3. `find_longest_cache_hit`, which walks a request's block keys and stops at the first miss.

Everything else follows from those three.

```python
# nanoserve/kv/prefix.py  (continued)
from collections import OrderedDict
from dataclasses import dataclass, field


@dataclass
class PhysicalBlock:
    pid: int
    ref_count: int = 0
    block_hash: int | None = None            # None until the block is sealed
    token_ids: tuple[int, ...] = ()          # collision guard on lookup


class PrefixCachingAllocator:
    """Paged block allocator with hash-indexed prefix reuse.

    Invariants:
      * ref_count > 0  =>  block is in some request's table, never reallocated
      * ref_count == 0 =>  block sits in `free_q` but KEEPS its hash entry,
                           so it can still be re-hit until it is reused
      * block_hash is not None  =>  block is complete and immutable
    """

    def __init__(self, num_blocks: int, block_size: int = BLOCK_SIZE):
        self.block_size = block_size
        self.blocks = [PhysicalBlock(pid=i) for i in range(num_blocks)]
        # OrderedDict as an LRU queue: oldest freed block is evicted first.
        self.free_q: "OrderedDict[int, None]" = OrderedDict(
            (i, None) for i in range(num_blocks)
        )
        self.cached: dict[int, int] = {}          # block_hash -> pid
        self.tables: dict[str, list[int]] = {}    # req_id -> [pid, ...]
        self.stat_hit_tokens = 0
        self.stat_req_tokens = 0
```

The subtle part is `free_q`. A block whose reference count drops to zero is *free* — it may be handed to anyone — but it is still *cached*. Its hash entry survives. That means a request arriving thirty seconds after the one that created a prefix can still hit it, as long as memory pressure has not forced the block to be reused. Free and cached are the same pool, distinguished only by whether anyone currently points at the block. This is the single most important structural idea in the whole design, and it is easy to get wrong by eagerly purging the hash entry on free.

### Looking up the longest hit

```python
    def find_longest_cache_hit(self, keys: list[BlockKey]) -> list[int]:
        """Walk block keys in order; stop at the first miss.

        Returns the physical block ids for the matched prefix. A hit MUST be
        a prefix: block j is only reusable if blocks 0..j-1 also matched,
        because the attention for later tokens reads all of them.
        """
        hit: list[int] = []
        for key in keys:
            pid = self.cached.get(key.block_hash)
            if pid is None:
                break
            blk = self.blocks[pid]
            if blk.block_hash != key.block_hash or blk.token_ids != key.token_ids:
                break                     # stale entry or 64-bit collision
            hit.append(pid)
        return hit
```

Note the early `break` rather than a scan. You cannot skip a missing block and reuse a later one, for two independent reasons. Mathematically, block $j$'s keys and values are defined by all tokens ${0..(j+1)B-1}$, so a gap invalidates everything after it. Physically, decode attention for a query at position $p$ reads *every* KV position $\le p$, so a hole in the block table is not a slow path — it is a wrong answer or an out-of-bounds read.

The walk is $O(\text{number of blocks})$ dictionary probes, each $O(1)$. For a 4,096-token prompt at `block_size = 16`, that is 256 probes. We will price that in section 6.

### Admission

```python
    def _incref(self, pid: int) -> None:
        blk = self.blocks[pid]
        if blk.ref_count == 0:
            self.free_q.pop(pid, None)     # was free-but-cached; now pinned
        blk.ref_count += 1

    def _alloc_one(self) -> int:
        """Pop the least-recently-freed block, evicting its cache entry."""
        if not self.free_q:
            raise MemoryError("out of KV blocks")
        pid, _ = self.free_q.popitem(last=False)      # LRU end
        blk = self.blocks[pid]
        if blk.block_hash is not None:
            # This block was still serving as a cache entry. Reusing it
            # means the entry is gone; drop it so nobody finds a stale pid.
            self.cached.pop(blk.block_hash, None)
            blk.block_hash = None
            blk.token_ids = ()
        return pid

    def admit(self, req_id: str, token_ids: list[int], salt: object = None):
        """Reserve blocks for a new request, reusing any cached prefix.

        Returns (block_table, num_computed_tokens). The engine prefills only
        token_ids[num_computed_tokens:].
        """
        keys = block_keys(token_ids, self.block_size, extra=salt)
        hit = self.find_longest_cache_hit(keys)

        table: list[int] = []
        for pid in hit:
            self._incref(pid)
            table.append(pid)

        n_blocks = -(-len(token_ids) // self.block_size)   # ceil
        for _ in range(len(hit), n_blocks):
            pid = self._alloc_one()
            self._incref(pid)
            table.append(pid)

        self.tables[req_id] = table
        num_computed = len(hit) * self.block_size
        self.stat_hit_tokens += num_computed
        self.stat_req_tokens += len(token_ids)
        return table, num_computed
```

Two things to notice. First, `num_computed` is a multiple of `block_size` by construction — the partial-tail rounding is baked into the return value, not applied later. Second, the engine's prefill call becomes a one-line change: instead of `model(token_ids)` you run `model(token_ids[num_computed:])` with `positions` starting at `num_computed`, and the attention kernel reads the cached blocks through the block table exactly as it would read blocks the request computed itself. There is no special "cached block" code path in the kernel. That is the payoff of paging.

### Sealing a block once it is full

Blocks become cacheable at the moment they fill up. After prefill, or after any decode step that completes a block, the engine tells the allocator:

```python
    def seal(self, req_id: str, logical_idx: int, key: BlockKey) -> None:
        """Publish a now-complete block into the cache index."""
        pid = self.tables[req_id][logical_idx]
        blk = self.blocks[pid]
        if blk.block_hash is not None:
            return                                  # already cached (was a hit)

        dup = self.cached.get(key.block_hash)
        if dup is not None and dup != pid:
            return                                  # someone published first
        blk.block_hash = key.block_hash
        blk.token_ids = key.token_ids
        self.cached[key.block_hash] = pid
```

The `dup` branch is the concurrent-prefill race, and it is worth a paragraph because it is invisible until you look for it. Two requests carrying the same system prompt arrive in the same scheduler step. Neither finds a cache hit at admission, because nothing has been computed yet. Both allocate their own blocks, both prefill, both compute the same numbers, and both try to seal. You get duplicate physical blocks holding identical keys and values — wasted memory, though never a wrong answer.

You have three options. Accept it (the duplicates are transient; one of them loses the index race and its blocks return to the free pool when its request finishes). Reserve the hash at admission time and make the second request wait for the first to finish prefilling — which converts wasted memory into head-of-line blocking, a trade that is usually worse. Or admit at most one request per distinct prefix per step, which is a scheduler policy, not an allocator concern. `nanoserve` takes the first option; production engines mostly do too, because the duplicate window is one scheduler step wide.

### Freeing, in reverse

```python
    def free(self, req_id: str) -> None:
        """Release a finished request's blocks, tail first."""
        for pid in reversed(self.tables.pop(req_id)):
            blk = self.blocks[pid]
            blk.ref_count -= 1
            if blk.ref_count == 0:
                self.free_q[pid] = None       # free, but still cached
```

`reversed` is not cosmetic. Freeing tail-first puts the *last* block at the front of the LRU queue, so when memory runs short the allocator reclaims the request-specific tail before it reclaims the shared system-prompt head. Free in forward order and you evict the most valuable block in the cache first, every time. This is a two-character bug that costs a large fraction of your hit rate, and there is no error message for it.

![Two request block tables where logical blocks zero through two map to the same physical blocks with reference count two and only the tails differ](/imgs/blogs/prefix-sharing-radix-trees-and-copy-on-write-3.webp)

Putting it together in a trace:

```python
alloc = PrefixCachingAllocator(num_blocks=1024, block_size=16)
sys_prompt = list(range(1000, 1096))       # 96 tokens == 6 full blocks

a = sys_prompt + list(range(50, 90))       # 136 tokens
b = sys_prompt + list(range(70, 118))      # 144 tokens

ta, na = alloc.admit("A", a)
print("A:", na, "cached tokens,", len(a) - na, "to prefill")
for j, key in enumerate(block_keys(a)):
    alloc.seal("A", j, key)

tb, nb = alloc.admit("B", b)
print("B:", nb, "cached tokens,", len(b) - nb, "to prefill")
print("shared pids:", [p for p in tb if p in ta])
print("refcount of first block:", alloc.blocks[ta[0]].ref_count)
```

```console
A: 0 cached tokens, 136 to prefill
B: 96 cached tokens, 48 to prefill
shared pids: [0, 1, 2, 3, 4, 5]
refcount of first block: 2
```

Request B prefilled 48 tokens instead of 144. The six system-prompt blocks are physically shared, each with a reference count of two.

---

## 4. The other data structure: a radix tree over the token path

The hash map is not the only way to answer "what is the longest cached prefix of these tokens?". The alternative — used by SGLang under the name RadixAttention, and described in this repo's [prefix caching and RadixAttention post](/blog/machine-learning/model-serving/prefix-caching-and-radixattention) — is a **radix tree**: a trie over token ids with path compression, where each edge carries a run of tokens and each node owns the physical blocks covering that run.

![A prefix tree rooted at a shared system prompt splitting into two few-shot branches and then into three per-request tails](/imgs/blogs/prefix-sharing-radix-trees-and-copy-on-write-4.webp)

```python
# nanoserve/kv/radix.py
from dataclasses import dataclass, field


@dataclass
class RadixNode:
    key: tuple[int, ...]                       # tokens on the edge INTO this node
    blocks: list[int] = field(default_factory=list)
    parent: "RadixNode | None" = None
    children: dict[int, "RadixNode"] = field(default_factory=dict)
    ref_count: int = 0
    last_used: float = 0.0


def _common(a: tuple[int, ...], b: Sequence[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class RadixCache:
    def __init__(self, block_size: int = BLOCK_SIZE):
        self.block_size = block_size
        self.root = RadixNode(key=())

    def match(self, tokens: Sequence[int]) -> tuple[list[int], int]:
        """Longest cached prefix. Returns (physical blocks, matched tokens)."""
        node, i, blocks = self.root, 0, []
        while i < len(tokens):
            child = node.children.get(tokens[i])
            if child is None:
                break
            n = _common(child.key, tokens[i:])
            # A partial edge match is only usable up to a block boundary.
            n -= n % self.block_size
            blocks.extend(child.blocks[: n // self.block_size])
            i += n
            if n < len(child.key):
                break                          # diverged mid-edge
            node = child
        return blocks, i
```

Insertion is the part that makes it a *radix* tree rather than a plain trie: when a new sequence diverges in the middle of an existing edge, you split that edge into two nodes.

```python
    def _split(self, node: RadixNode, at: int) -> RadixNode:
        """Split `node`'s incoming edge after `at` tokens. `at` must be a
        multiple of block_size so the block list divides cleanly."""
        assert at % self.block_size == 0 and 0 < at < len(node.key)
        head = RadixNode(
            key=node.key[:at],
            blocks=node.blocks[: at // self.block_size],
            parent=node.parent,
        )
        node.parent.children[node.key[0]] = head
        node.key = node.key[at:]
        node.blocks = node.blocks[at // self.block_size :]
        node.parent = head
        head.children[node.key[0]] = node
        head.ref_count = node.ref_count
        return head

    def insert(self, tokens: Sequence[int], blocks: list[int]) -> None:
        node, i = self.root, 0
        while i < len(tokens):
            child = node.children.get(tokens[i])
            if child is None:
                leaf = RadixNode(
                    key=tuple(tokens[i:]),
                    blocks=blocks[i // self.block_size :],
                    parent=node,
                )
                node.children[tokens[i]] = leaf
                return
            n = _common(child.key, tokens[i:])
            n -= n % self.block_size
            if n < len(child.key):
                child = self._split(child, n) if n else child
                if n == 0:
                    break                      # cannot split below a block
            i += n
            node = child
```

That `assert at % self.block_size == 0` is the honest constraint, and it is the thing glossy diagrams of radix trees over KV caches tend to hide. The tree can match at token granularity, but the *cache* is paged, so any split has to land on a block boundary. A radix tree over a paged cache is still block-quantized. It buys you structure, not resolution.

### What the tree actually buys

The tree's real advantage is **eviction**. In the hash-map design, entries are independent: nothing stops you from evicting block 3 of a prefix while blocks 0-2 and 4-9 stay cached. Blocks 4-9 are then unreachable — `find_longest_cache_hit` stops at the hole and never looks past it. They are not *wrong*, just dead weight occupying index entries until they too get reused. The LRU free queue makes this rare in practice (parents are touched more often, so they sit deeper in the queue), but nothing structurally prevents it.

In the tree, you evict a **leaf**. A node with children cannot be evicted, so a parent is structurally protected by its descendants, and unreachable blocks cannot exist. Eviction becomes: collect leaves with `ref_count == 0`, drop the least recently used, and repeat — parents become leaves as their children disappear.

The tree is also the natural structure when one prompt fans out into many branches, which is exactly the parallel-sampling and beam-search case from the next section, and exactly the shape of an agent exploring a tree of tool calls.

| Property | Hash map over block hashes | Radix tree over tokens |
| --- | --- | --- |
| Lookup for a $P$-token prompt | $O(P/B)$ dictionary probes | $O(\text{depth})$ pointer hops + token compares |
| Match granularity | block (`floor(S/B)` blocks) | block (splits must be block-aligned) |
| Orphaned entries possible | yes, if a middle block is evicted | no, eviction always picks a leaf |
| Eviction policy | LRU over the free-block queue | LRU over zero-reference leaves |
| Insert cost | one dict write per sealed block | may split a node (allocation + relink) |
| Sharding across nodes | trivial, hash is the shard key | hard, tree is global state |
| Lines of code in `nanoserve` | about 70 | about 130 |
| Used by | vLLM | SGLang |

My honest read: the data structure is not what determines your hit rate. The workload determines your hit rate, and the eviction policy determines how much of it you keep under pressure. Pick the hash map if your cache will ever leave the process — a distributed store wants the hash as its shard key, which is precisely what Mooncake's master server indexes. Pick the tree if you need explicit control over what survives eviction, or if your workload is branch-heavy and you want the shared structure to be inspectable.

---

## 5. Copy-on-write: the correctness heart

Everything so far shares blocks that are **complete**, and a complete block is immutable — nobody appends to a block that is already full. So where does mutation come from?

From **forking**. A sequence forks when one prompt produces several continuations that must each keep their own keys and values:

- `n > 1` sampling: one request, four independent samples from the same prompt.
- Beam search: the beam branches at every step, and beams are pruned and re-parented.
- Agentic tree search or best-of-$n$ reranking: one context, many speculative continuations.

A fork happens at an arbitrary token position, and arbitrary positions are usually not block boundaries. So the child inherits the parent's block table, including the parent's **partial** final block — the one both of them are still writing into. Its reference count is now 2. The next token each branch generates goes into slot $p \bmod B$ of that block, and they are different tokens.

If you let both write, you get a race with no winner: whichever branch writes last defines the keys and values that *both* branches read, and one of them is now conditioning on a token it never generated. Silent corruption, non-deterministic, and essentially impossible to debug from the output.

Copy-on-write is the fix, and it is the same fix the operating system uses after `fork()`: a block with more than one reference is read-only. Before writing, check the count; if it exceeds one, allocate a fresh block, copy the old contents in, drop your reference to the original, and point your own block table at the copy.

<figure class="blog-anim">
<svg viewBox="0 0 560 300" role="img" aria-label="Two block tables share the same physical blocks until one request writes into the shared tail block and a private copy is made for it" style="width:100%;height:auto;max-width:760px">
<style>
.cw-cell{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.cw-hot{fill:var(--accent,#6366f1);opacity:.20;stroke:var(--accent,#6366f1);stroke-width:1.5}
.cw-lbl{font:600 13px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.cw-sub{font:500 11px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.cw-row{font:600 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.cw-link{stroke:var(--accent,#6366f1);stroke-width:2.5;fill:none}
.cw-cut{stroke:var(--border,#d1d5db);stroke-width:2.5;fill:none;stroke-dasharray:5 5}
@keyframes cw-fadeA{0%,36%{opacity:1}50%,94%{opacity:0}100%{opacity:1}}
@keyframes cw-fadeB{0%,36%{opacity:0}50%,94%{opacity:1}100%{opacity:0}}
@keyframes cw-slide{0%,36%{opacity:0;transform:translateX(46px)}50%,94%{opacity:1;transform:translateX(0)}100%{opacity:0;transform:translateX(46px)}}
.cw-s1{animation:cw-fadeA 9s ease-in-out infinite}
.cw-s2{animation:cw-fadeB 9s ease-in-out infinite}
.cw-in{animation:cw-slide 9s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.cw-s1{animation:none;opacity:0}.cw-s2{animation:none;opacity:1}.cw-in{animation:none;opacity:1;transform:translateX(0)}}
</style>
<text class="cw-sub" x="260" y="34">blocks 0 to 2 · complete · immutable</text>
<text class="cw-sub" x="452" y="34">block 3 · partial</text>
<text class="cw-row" x="60" y="84">Req A</text>
<rect class="cw-hot" x="120" y="54" width="88" height="50" rx="7"/>
<rect class="cw-hot" x="216" y="54" width="88" height="50" rx="7"/>
<rect class="cw-hot" x="312" y="54" width="88" height="50" rx="7"/>
<rect class="cw-cell" x="408" y="54" width="88" height="50" rx="7"/>
<text class="cw-lbl" x="164" y="84">P7</text>
<text class="cw-lbl" x="260" y="84">P8</text>
<text class="cw-lbl" x="356" y="84">P9</text>
<text class="cw-lbl" x="452" y="84">P12</text>
<path class="cw-link" d="M164 104 L164 194"/>
<path class="cw-link" d="M260 104 L260 194"/>
<path class="cw-link" d="M356 104 L356 194"/>
<path class="cw-link cw-s1" d="M452 104 L452 194"/>
<path class="cw-cut cw-s2" d="M452 104 L452 194"/>
<text class="cw-row" x="60" y="224">Req B</text>
<rect class="cw-hot" x="120" y="194" width="88" height="50" rx="7"/>
<rect class="cw-hot" x="216" y="194" width="88" height="50" rx="7"/>
<rect class="cw-hot" x="312" y="194" width="88" height="50" rx="7"/>
<text class="cw-lbl" x="164" y="224">P7</text>
<text class="cw-lbl" x="260" y="224">P8</text>
<text class="cw-lbl" x="356" y="224">P9</text>
<rect class="cw-cell cw-s1" x="408" y="194" width="88" height="50" rx="7"/>
<text class="cw-lbl cw-s1" x="452" y="224">P12</text>
<rect class="cw-hot cw-in" x="408" y="194" width="88" height="50" rx="7"/>
<text class="cw-lbl cw-in" x="452" y="220">P37</text>
<text class="cw-sub cw-in" x="452" y="236">copy · 2 MB</text>
<text class="cw-sub cw-s1" x="280" y="278">P12 refcount 2 — both requests read the same tail block</text>
<text class="cw-sub cw-s2" x="280" y="278">B writes a token: copy P12 into P37, refcounts drop to 1 and 1</text>
</svg>
<figcaption>Reference counts make the shared tail block read-only; the moment request B appends a token, the allocator copies the block and rewires only B's table, leaving A's view untouched.</figcaption>
</figure>

### The code

```python
# nanoserve/kv/prefix.py  (continued)
    def fork(self, parent_id: str, child_id: str) -> list[int]:
        """Branch a sequence: the child shares every block, physically."""
        table = list(self.tables[parent_id])
        for pid in table:
            self._incref(pid)
        self.tables[child_id] = table
        return table

    def prepare_write(self, req_id: str, logical_idx: int):
        """Call BEFORE appending K/V into logical block `logical_idx`.

        Returns (pid_to_write, copy_op) where copy_op is None if the block
        was already private, else (src_pid, dst_pid) that the engine must
        memcpy on the device before the append.
        """
        table = self.tables[req_id]
        pid = table[logical_idx]
        blk = self.blocks[pid]
        if blk.ref_count == 1:
            return pid, None                    # sole owner: write in place

        dst = self._alloc_one()
        self._incref(dst)
        blk.ref_count -= 1                      # we no longer point at `pid`
        if blk.ref_count == 0:
            self.free_q[pid] = None
        table[logical_idx] = dst
        return dst, (pid, dst)
```

And the device-side copy, which is the only place any bytes actually move:

```python
# nanoserve/kv/copy.py
import torch


@torch.inference_mode()
def copy_blocks(kv_cache: list[torch.Tensor], ops: list[tuple[int, int]]) -> None:
    """Copy whole physical blocks, all layers, on device.

    kv_cache[layer] has shape
        [2, num_blocks, block_size, num_kv_heads, head_dim]
    where dim 0 selects key vs value.
    """
    if not ops:
        return
    src = torch.tensor([s for s, _ in ops], device=kv_cache[0].device)
    dst = torch.tensor([d for _, d in ops], device=kv_cache[0].device)
    for layer in kv_cache:
        layer[:, dst] = layer[:, src]
```

Batching the ops matters: one `index_copy`-style call per layer beats a Python loop per block, and in a real engine this is a single fused CUDA kernel launched once per step alongside the swap-in and swap-out operations.

### Where `prepare_write` gets called

Exactly once per sequence per decode step, on the logical block that is about to receive the new token:

```python
    def append_slot(self, req_id: str, seq_len: int):
        """Make room for the token at position `seq_len`. Returns (pid, slot, copy)."""
        logical = seq_len // self.block_size
        table = self.tables[req_id]
        if logical == len(table):                 # crossed a block boundary
            pid = self._alloc_one()
            self._incref(pid)
            table.append(pid)
            return pid, 0, None
        pid, copy_op = self.prepare_write(req_id, logical)
        return pid, seq_len % self.block_size, copy_op
```

Note the branch. If the sequence has just crossed a block boundary, there is nothing to copy — the new block is fresh and private. Copy-on-write only fires when a sequence writes into a *shared partial* block, which happens at most once per fork per branch. Fork at a block boundary and the copy never happens at all.

### What copy-on-write actually costs

One block copy for Llama-3.1-8B moves 2 MB in and 2 MB out. On an A100 80GB SXM, whose HBM2e bandwidth NVIDIA's [datasheet](https://www.nvidia.com/en-us/data-center/a100/) lists at 2.039 TB/s, that is:

$$
\frac{4 \times 10^{6}\ \text{bytes}}{2.039 \times 10^{12}\ \text{bytes/s}} \approx 2.0\ \mu s
$$

Compare against the alternative — not forking, and instead re-prefilling those 16 tokens from scratch. A 16-token prefill for an 8B model is nowhere near compute-saturated, so it is bounded by streaming the weights: 16.1 GB of bf16 weights divided by 2.039 TB/s is about 7.9 ms. Copy-on-write is roughly **4,000× cheaper** than the recompute it replaces, and both numbers are derived from the same two specs.

#### Worked example: `n = 4` sampling on a 4,096-token prompt

A request asks for four independent samples from one 4,096-token prompt. Llama-3.1-8B, bf16, `block_size = 16`.

- The prompt occupies ${4096 / 16 = 256}$ blocks; 255 are complete, and the 256th holds all 16 tokens only if 4,096 divides evenly — it does, so let us make it realistic: a 4,090-token prompt gives 255 complete blocks and one partial block holding 10 tokens.
- Without sharing, four sequences each store their own copy: $4 \times 4090 \times 128\ \text{KB} = 2.00\ \text{GB}$.
- With sharing, one copy of the prompt plus three copies of the single partial block: $4090 \times 128\ \text{KB} + 3 \times 2\ \text{MB} = 511\ \text{MB} + 6\ \text{MB} = 517\ \text{MB}$.
- Memory saved: about **1.48 GB**, which on a 24 GB RTX 4090 with roughly 6 GB of KV budget after weights is the difference between running the request and rejecting it.
- Copy cost: three block copies, about **6 microseconds** total on an A100.
- Prefill saved: three full 4,090-token prefills.

`Source: derived` for all five figures, from the 128 KB-per-token formula in section 1 and the A100 bandwidth spec above.

That ratio — megabytes saved per microsecond spent — is why parallel sampling and beam search were the original motivating cases for copy-on-write in the PagedAttention design, and why an engine without it has to either forbid `n > 1` or pay for it in full.

### The refcount bugs you will write

Three, in the order I expect you to hit them.

1. **Forgetting to decrement on the copy path.** `prepare_write` must drop the original's reference. Miss it and the block never returns to the free pool: a slow leak that shows up as "we OOM after four hours" and nothing sooner.
2. **Incrementing after popping from the free queue instead of before.** `_incref` pulls the block out of `free_q` when the count goes zero-to-one. If you allocate and forget to increment, a block can be handed to two requests simultaneously. That one is not a leak; it is corruption.
3. **Sealing a block you obtained from a cache hit.** A hit block already has `block_hash` set and is already in the index. Sealing it again with a recomputed hash — or worse, overwriting `self.cached[h]` with the same pid but stale `token_ids` — creates entries that fail the collision guard forever. The `if blk.block_hash is not None: return` guard in `seal` exists solely for this.

A cheap invariant check, worth running in tests:

```python
    def check(self) -> None:
        counted = {}
        for table in self.tables.values():
            for pid in table:
                counted[pid] = counted.get(pid, 0) + 1
        for blk in self.blocks:
            assert blk.ref_count == counted.get(blk.pid, 0), f"refcount {blk.pid}"
            assert (blk.ref_count == 0) == (blk.pid in self.free_q), f"freeq {blk.pid}"
        for h, pid in self.cached.items():
            assert self.blocks[pid].block_hash == h, f"stale index {h}"
```

That function has found more bugs for me, in the abstract, than any print statement would. Run it after every scheduler step in your test harness and delete it in production.

---

## 6. What it saves, and what it costs when it saves nothing

Now the arithmetic that decides whether any of this is worth doing.

### The savings, derived

Let $P$ be the prompt length, $S$ the shared prefix length, $B$ the block size. The number of cached tokens is $C = B \lfloor S/B \rfloor$, so the request prefills $P - C$ tokens.

**Linear layers.** Every projection and MLP is per-token, so their cost scales as the token count. For a model with $N$ parameters, prefill FLOPs are about ${2NP}$; with a hit they are about ${2N(P - C)}$. Ratio:

$$
R_{\text{linear}} = \frac{P - C}{P}
$$

**Attention.** This one improves *more* than linearly, and the reason is nice. Causal prefill computes a query-key pair for every $(i, j)$ with $j \le i$, so the full count is $\sum_{i=0}^{P-1}(i+1) = P(P+1)/2$. With a cached prefix of length $C$, the queries at positions ${0..C-1}$ never run at all, but the surviving queries at positions $C..P-1$ still attend to *every* key to their left, including the cached ones. So the count becomes

$$
\sum_{i=C}^{P-1}(i+1) = \frac{P(P+1)}{2} - \frac{C(C+1)}{2}
$$

giving

$$
R_{\text{attn}} = 1 - \frac{C(C+1)}{P(P+1)} \approx 1 - f^2, \qquad f = C/P
$$

The linear part scales as ${1 - f}$ and the attention part as $1 - f^2$. At $f = 0.9$, linear work drops to 10% and attention work to 19% — attention keeps proportionally *more* work, because the uncached queries still have to look at all the cached keys. Anyone who tells you a 90% prefix hit gives a flat 10× on prefill is quoting only half the model. It gives 10× on the linear half and about 5.3× on the quadratic half, and which one dominates depends on how long your context is.

![Prefill work with no sharing on the left and with prefix sharing on the right showing token counts and query-key pair counts](/imgs/blogs/prefix-sharing-radix-trees-and-copy-on-write-5.webp)

#### Worked example: a RAG request on Llama-3.1-8B

Prompt of 4,096 tokens, of which 3,972 are a shared system prompt plus a fixed retrieved document. Block size 16.

| Quantity | Value | Source |
| --- | --- | --- |
| Cacheable prefix $C = 16\lfloor 3972/16\rfloor$ | 3,968 tokens (4 lost to rounding) | derived |
| Tokens actually prefilled | 128 | derived |
| Linear-layer work ratio | 3.1% | derived |
| Query-key pairs, no cache | 8,390,656 | derived |
| Query-key pairs, with cache | 516,160 | derived |
| Attention work ratio | 6.2% | derived |
| Full prefill FLOPs, ${2NP}$ at $N = 8.03 \times 10^{9}$ | 65.8 TFLOP | derived |
| Full prefill time at 50% of A100 bf16 peak (312 TFLOP/s) | about 420 ms | derived + cited: NVIDIA A100 datasheet |
| Cached prefill, weight-bandwidth bound at 16.1 GB / 2.039 TB/s | about 8 ms floor | derived + cited: NVIDIA A100 datasheet |
| TTFT improvement, order of magnitude | roughly 30× | derived, order-of-magnitude |

The 50% model-FLOPs-utilization assumption is a stand-in, and you should replace it with your own measured value before quoting the 420 ms to anyone. The point of the row is the shape, not the digits: a large prefix hit converts a compute-bound prefill into something close to the weight-streaming floor, which is a change of kind, not degree.

#### Worked example: the agentic loop, where this stops being an optimization

vLLM's [Mooncake Store post](https://vllm.ai/blog/2026-05-06-mooncake-store) publishes statistics from 610 Codex and SWE-bench agent traces: an input-to-output token ratio of **131:1**, a median of **33 turns** per trace, about **2,242 tokens of context growth per turn**, and a median context of roughly **80K tokens by turn 30**. `Source: cited: vLLM Mooncake Store post, 2026-05-06.`

Take that at face value and derive the workload. If context grows by $g = 2242$ tokens per turn, turn $k$ carries about $gk$ tokens of prompt.

- **Without prefix sharing**, every turn re-prefills its whole context. Total prefill over 33 turns:

$$
\sum_{k=1}^{33} gk = g \cdot \frac{33 \cdot 34}{2} = 2242 \times 561 = 1{,}257{,}762 \ \text{tokens}
$$

- **With perfect prefix sharing**, turn $k$ only prefills the $g$ new tokens appended since turn $k-1$:

$$
33 \times 2242 = 73{,}986 \ \text{tokens}
$$

- Ratio: **17.0×** fewer prefill tokens. `Source: derived from the cited trace statistics.`

Sanity check the model against the source: $2242 \times 30 = 67{,}260$, against a cited median of about 80K at turn 30 — the same order, so the linear-growth assumption is not crazy. And the 131:1 input-to-output ratio is the tell: in agentic serving, prefill *is* the workload. Decode is a rounding error. An engine that re-prefills every turn is doing seventeen times the necessary work, which is why the same post reports a hit rate of **92.2% versus 1.7%**, a **3.8× throughput** gain and **46× lower P50 TTFT** for Kimi-2.5 in NVFP4 on a 1-prefill-1-decode deployment of 12 GB200s driven by those Codex traces. `Source: cited: vLLM Mooncake Store post.`

Note the setup attached to those numbers. They are not "prefix caching gives you 46× TTFT." They are what a *distributed* KV store gives a *specific agentic trace* on a *specific 12-GPU deployment*, compared against a baseline whose hit rate was 1.7%. Quote the setup or do not quote the number.

### The cost when nothing is shared

This is the question that decides the default. If prefix caching only helps sometimes, it must cost approximately nothing the rest of the time, or you cannot leave it on.

Per request you pay:

1. $\lceil P/B \rceil$ hash computations. For $P = 4096$, $B = 16$: 256 hashes of a 16-element integer tuple.
2. $\lceil P/B \rceil$ dictionary probes in `find_longest_cache_hit` — though on a total miss you stop at the first one.
3. One dictionary write per sealed block.
4. Host memory for the index: roughly a 64-bit hash, a 16-element token tuple, and dict overhead per block. Call it 200 bytes. A pool of 100,000 blocks — which for Llama-3.1-8B at 2 MB per block is 200 GB of KV, more than a single node has — costs about **20 MB of host RAM**. Ignorable.

The one that could matter is (1). A Python `hash` over a 16-element tuple is on the order of a few hundred nanoseconds; call it 1 microsecond to be pessimistic, giving 256 microseconds against the ~420 ms prefill derived above — about **0.06%**. In compiled code with a fast non-cryptographic hash at roughly 50 ns per block, it is 13 microseconds, or **0.003%**. Both are order-of-magnitude estimates; measure yours with:

```python
# nanoserve/bench_hash.py
import timeit

setup = "from nanoserve.kv.prefix import block_keys; toks=list(range(4096))"
n = 200
secs = timeit.timeit("block_keys(toks)", setup=setup, number=n) / n
print(f"{secs * 1e6:.1f} us per 4096-token prompt "
      f"({secs * 1e6 / 256:.2f} us per block)")
```

On a modern x86 core you should expect somewhere in the range of 100 to 400 microseconds per 4,096-token prompt with the builtin hash, and roughly 10× that with the SHA-256 path. `Source: reproduce: bench_hash.py.` Run it and report yours.

This derivation is corroborated by the design bar vLLM set for itself. Per the [vLLM V1 announcement](https://vllm.ai/blog/2025-01-27-v1-alpha-release) (2025-01-27), V1 implements "zero-overhead" prefix caching with **less than 1% throughput decrease even at 0% hit rate**, and turns it on by default for that reason. `Source: cited: vLLM V1 post.` That is the number to hold yourself to: if your prefix cache costs more than 1% on a workload that never hits, you have implemented it wrong — most likely by doing the hashing inside the hot scheduler loop instead of once at admission, or by hashing token *strings* rather than ids.

### When it genuinely does nothing

Set $f = 0$ and every ratio above collapses to 1. Three workloads land there:

- **Unique long documents.** Batch summarization of a corpus where each request carries a different document and no shared instruction block. Every hash misses. You pay the 0.06% and get nothing.
- **Short shared prefixes.** If the shared prefix is under `block_size` tokens, $\lfloor S/B \rfloor = 0$ and you get *exactly* zero. A 12-token system prompt at `block_size = 16` shares nothing at all.
- **Per-request personalization at the front.** Putting a user id, a timestamp, or a session token at the *start* of the prompt destroys sharing for everything after it, because the chain hash makes block 0 differ and the walk stops immediately. Move volatile content to the end of the prompt. This is the single highest-leverage prompt-engineering change for serving cost, and it is invisible unless you understand the chain hash.

That last one deserves emphasis, because it is a prompt-design decision with a serving-cost consequence of an order of magnitude. Prefix caching rewards prompts built as *fixed preamble, then variable suffix*. Any template that interpolates a request-specific value near the top forfeits the entire mechanism.

---

## 7. Eviction: a free block is still a cached block

Prefix caching and eviction are the same subsystem viewed from two angles, which is why the next post in this series covers [eviction, preemption, and KV swapping](/blog/machine-learning/inference-engineering/eviction-preemption-and-kv-swapping) in depth. Three rules connect them.

**A block with a positive reference count cannot be evicted.** It is in someone's block table right now; reusing it would corrupt a live sequence. In `nanoserve` that invariant is enforced structurally: `_alloc_one` only pops from `free_q`, and `_incref` removes a block from `free_q` the moment its count goes from zero to one. There is no code path that can allocate a referenced block, which is much stronger than a check.

**A block with zero references stays cached until it is reused.** This is where the hit rate comes from. A conversation that pauses for twenty seconds between turns has zero live references during the pause, yet its blocks are still in the index. If the next turn arrives before memory pressure recycles them, it hits. The residency time of your cache is therefore *pool size divided by allocation rate*, and it is the number that actually governs your hit rate — not the size of your index.

vLLM's [streaming and realtime API post](https://vllm.ai/blog/2026-01-31-streaming-realtime) (2026-01-31) makes the same point from the other side: session streaming deliberately keeps KV blocks from eviction while waiting for the next chunk, avoiding a recompute of the trailing 16-token block. `Source: cited.` The mechanism there is pinning; the underlying problem is residency.

**Eviction order should follow the tree, even without a tree.** Evicting a middle block leaves its descendants unreachable, as discussed. Freeing tail-first — the `reversed()` in `free` — approximates leaf-first eviction in a hash-map design and gets most of the benefit for two characters of code. If you want the guarantee rather than the approximation, that is the radix tree's argument.

There is one more interaction worth flagging, because it is a trap. Under memory pressure a scheduler may **preempt** a running request — evict its blocks and restart it later. If your preemption path frees blocks the normal way, the preempted request's prefix stays in the cache, and when it is rescheduled it hits on its own blocks and recovers cheaply. If your preemption path *purges* the index for those blocks (a plausible-looking "clean up after yourself"), the restart pays a full recompute. Preemption plus prefix caching is nearly free; preemption plus eager purging is the thrash pattern. Same code, one wrong line.

---

## 8. What belongs in the cache key

The chain hash covers token ids and the parent. Everything *else* that changes the stored keys and values must go into the `extra` metadata slot, or you will serve one request's cache to another.

![Layers of a block hash from model identity down through adapter and tenant salt to the parent hash and token ids](/imgs/blogs/prefix-sharing-radix-trees-and-copy-on-write-6.webp)

Go down the list of things $f^{(\ell)}$ actually depends on:

- **The model and its weights.** Within one process this is constant and can be omitted. The moment the cache is shared across processes or persisted to disk, the model identity and revision must be in the key. A cache keyed only on tokens, served across a rolling deploy where half the fleet has new weights, is a correctness bug that survives every unit test.
- **The dtype and KV quantization scheme.** A block stored as bf16 and a block stored as FP8 E4M3 are not interchangeable, and if you use per-tensor or per-head FP8 scales those scales are part of the stored state. vLLM's [FP8 KV cache post](https://vllm.ai/blog/2026-04-22-fp8-kvcache) (2026-04-22) documents per-tensor uncalibrated scaling as the default with per-head scales as an option; a cache that mixes them is silently wrong.
- **The LoRA adapter id.** A LoRA modifies $W_K$ and $W_V$, so the same tokens under two adapters produce different keys and values. If you serve multiple adapters against one base model — and vLLM's [multi-LoRA post](https://vllm.ai/blog/2026-02-26-multi-lora) (2026-02-26) describes serving eight adapters in parallel — the adapter id must be in the key. Otherwise adapter B inherits adapter A's prefix.
- **Multimodal inputs.** An image occupying $n$ token positions contributes keys and values that depend on the image pixels, not on any token id. The V1 announcement notes that multimodal support adds image-hash prefix caching for exactly this reason. Hash the image, put the hash in the metadata for the blocks it spans.
- **Position offsets.** RoPE rotates keys by their *absolute* position. A prefix that starts at position 0 in one request and position 40 in another has different keys even for identical tokens. This is why prefix caching matches prefixes and not substrings: a "middle match" is not reusable without re-rotating, and re-rotating costs about as much as recomputing.
- **The tenant, if you want isolation.** vLLM exposes a per-request **cache salt**: an opaque string mixed into the hash so that two tenants with byte-identical prompts land in disjoint hash spaces and never share a block. It is a deliberate trade — you give up all cross-tenant sharing to gain isolation.

Things that are *not* in the key, and should not be: temperature, top-p, top-k, `max_tokens`, stop strings, `logprobs`, the seed, and every other sampling parameter. They act on logits after the keys and values exist. Two requests with the same prompt and wildly different sampling settings share their entire prefix, correctly.

In `nanoserve` the whole set travels through one argument:

```python
CacheScope = tuple                     # (model_rev, kv_dtype, lora_id, salt)


def scope_for(req) -> CacheScope:
    return (
        MODEL_REVISION,                # e.g. "meta-llama/Llama-3.1-8B@a1b2c3"
        str(KV_DTYPE),                 # "torch.bfloat16" / "fp8_e4m3"
        req.lora_id,                   # None for the base model
        req.cache_salt,                # None => shared, else per-tenant
    )


table, num_computed = alloc.admit(req.id, req.token_ids, salt=scope_for(req))
```

One argument, four fields, and the entire class of "why did this request get someone else's answer" bugs goes away.

### The security preview: sharing is a timing channel

Here is the part that is easy to leave out of an engineering post and very hard to leave out of a production design review.

A cache hit is *observably faster* than a miss. The RAG example above derives roughly 420 ms versus roughly 8 ms — a ratio no amount of network jitter hides. So an attacker who can submit prompts and measure time-to-first-token has an oracle: "does the cache already contain a block whose tokens are exactly these?" With that oracle, a prefix can be recovered token by token — guess the next token, submit prefix-plus-guess, keep the guess whose TTFT drops. Each block boundary crossed confirms 16 tokens at once.

What is at risk is any secret that lives in a shared prefix: a proprietary system prompt, a retrieved document another tenant uploaded, the leading tokens of another user's conversation. Published work auditing prompt caching in commercial LLM APIs has demonstrated that response-time differences do leak whether a prompt was cached, and by extension whether it was cached *for someone else*. This is not theoretical, and it is the reason the cache-salt mechanism exists.

The mitigations, in order of how much throughput they cost:

| Mitigation | What it stops | Cost |
| --- | --- | --- |
| Per-tenant cache salt | All cross-tenant inference | All cross-tenant sharing |
| Per-tenant cache partitions | Cross-tenant inference; keeps intra-tenant sharing | Fragmented pool, lower total hit rate |
| Share only operator-owned prefixes | Leaking user content; system prompt still probeable | Small — usually the right default |
| Constant-time TTFT (pad to worst case) | The timing signal itself | Destroys the entire benefit |

My recommendation for a multi-tenant service: share aggressively *within* a tenant, salt *across* tenants, and treat your system prompt as public — because with a shared cache and a stopwatch, it effectively is. A full case study of this attack class lands later in the series under the operations and case-studies track; treat this section as the warning label.

---

## 9. Measuring it without lying to yourself

Prefix caching is unusually easy to benchmark dishonestly, because the most natural benchmark is the one that lies the most.

**The trap.** You write a load script that sends the same prompt 500 times. First request misses; the other 499 hit at nearly 100%. You report a 30× TTFT improvement. It is real, and it is meaningless — you measured your load generator's lack of imagination, not your service.

Six rules that make the number mean something:

1. **Reset the cache between runs, and say whether you did.** A cold run and a warm run are different experiments. vLLM exposes a `reset_prefix_cache` operation for exactly this — the [sleep mode post](https://vllm.ai/blog/2025-10-26-sleep-mode) (2025-10-26) notes that waking from level-2 sleep requires `reload_weights` plus `reset_prefix_cache`. `Source: cited.` Use the same discipline in your own harness.
2. **Report hit rate token-weighted, not request-weighted.** "80% of requests hit" and "80% of prompt tokens hit" are wildly different claims. A request that hits 16 tokens of a 4,000-token prompt counts as a hit under the first metric and as nothing under the second. Only the second predicts your compute. My allocator tracks `stat_hit_tokens / stat_req_tokens` for this reason.
3. **Replay a real trace, or model one.** Hit rate is a property of the arrival pattern. If your production traffic is 40% multi-turn conversations with a 5-second median inter-turn delay — a figure the Mooncake trace statistics put at a median of 5.2 seconds with a P99 of 81.4 seconds — then your benchmark needs those gaps, because during them the blocks are unreferenced and eligible for reuse.
4. **Use open-loop arrivals.** A closed-loop harness that waits for each response before sending the next one self-limits concurrency, which keeps the cache small and the hit rate artificially high. Poisson arrivals at a target rate, and measure at steady state after the cache has filled.
5. **Time the GPU correctly.** Warm up, `torch.cuda.synchronize()` before and after the timed region or use `torch.cuda.Event` pairs, and discard the first several iterations. This is the same harness discipline established in [the baseline post](/blog/machine-learning/inference-engineering/the-naive-decode-loop-and-your-first-baseline); prefix caching does not get an exemption.
6. **Report TTFT p50 and p99, not tok/s.** Prefix caching changes prefill, and prefill owns TTFT. It barely touches TPOT. A tok/s headline averages the two and hides the effect you actually produced.

A minimal honest harness:

```python
# nanoserve/bench_prefix.py
import argparse, random, time


def run(engine, trace, warmup=20):
    """trace: list of (arrival_time_s, token_ids). Open-loop replay."""
    ttfts, hits, total = [], 0, 0
    t0 = time.perf_counter()
    for i, (at, toks) in enumerate(trace):
        while time.perf_counter() - t0 < at:
            engine.step()                        # keep the engine running
        start = time.perf_counter()
        req = engine.submit(toks)
        engine.wait_first_token(req)
        if i >= warmup:
            ttfts.append((time.perf_counter() - start) * 1e3)
            hits += req.num_cached_tokens
            total += len(toks)
    ttfts.sort()
    print(f"requests        {len(ttfts)}")
    print(f"token hit rate  {hits / total:.1%}")
    print(f"TTFT p50        {ttfts[len(ttfts) // 2]:.1f} ms")
    print(f"TTFT p99        {ttfts[int(len(ttfts) * 0.99)]:.1f} ms")
```

Expected output shape, not a claim about any machine:

```console
requests        480
token hit rate  61.4%
TTFT p50        74.3 ms
TTFT p99        410.8 ms
```

Run it twice — once with sharing enabled and once disabled — on the *same* replayed trace, and report both hit rate and TTFT from the same run. A hit-rate number without a TTFT number, or vice versa, is half an experiment.

One more measurement worth taking: the **ceiling**. Before optimizing your eviction policy, find out what hit rate the workload could possibly deliver with infinite memory. PegaFlow computes this with a HyperLogLog sketch, estimating $r^{*} = (N-U)/N$ where $N$ is total blocks referenced and $U$ is unique blocks, in under 1 MiB with about 0.8% error, per vLLM's [PegaFlow post](https://vllm.ai/blog/2026-05-18-pegaflow) (2026-05-18). `Source: cited.` If your achieved hit rate is 50% and the ceiling is 52%, stop tuning eviction and go buy memory or change the workload. If the ceiling is 95%, your policy is the problem.

---

## 10. Case studies: what this is worth in production

Four public results, each with its setup, because the setup is where the number lives.

**vLLM V1 — the overhead bar.** The [V1 announcement](https://vllm.ai/blog/2025-01-27-v1-alpha-release) reports "zero-overhead" prefix caching: less than 1% throughput decrease even at 0% hit rate, achieved with hash lookup plus LRU, and enabled by default. The engineering claim underneath is that constant-factor bookkeeping must be small enough to leave on unconditionally. That is the bar for your own implementation.

**Mooncake Store — the agentic ceiling.** Running Kimi-2.5 in NVFP4 on a 1-prefill-1-decode deployment of 12 GB200s against 610 Codex and SWE-bench agent traces, [the post](https://vllm.ai/blog/2026-05-06-mooncake-store) reports a hit rate of 92.2% versus 1.7% without the distributed store, 3.8× throughput, 46× lower P50 TTFT, and 8.6× lower end-to-end latency, with near-linear scaling to over 95% hit rate as the deployment grows from 12 to 60 GB200s. The acknowledged limit matters as much as the number: decode instances do not currently read from the pool (loads are prefill-only), and single-instance caches saturate and evict — which is the whole reason a distributed pool exists.

**PegaFlow — the chat-versus-long-context split.** vLLM's [PegaFlow post](https://vllm.ai/blog/2026-05-18-pegaflow) describes a Rust daemon with three tiers (local pinned DRAM, remote DRAM over one-sided RDMA READ, local SSD via io_uring) and reports, for Qwen3-8B on a single host, a hit rate of 52.35% versus 11.77%, a 56% throughput gain (11.97 versus 7.68 requests per second) and 36% lower TTFT; and for DeepSeek-V3.2 with MLA at TP8, a hit rate of 97.23% versus 65.18% with 72% higher throughput and 41% lower TTFT. Two things to take from the pair. First, chat workloads sit around 50% and long-context workloads sit in the high 90s — the *workload*, not the cache, sets the ceiling. Second, for tensor parallelism the design stores the logical latent KV once rather than per rank, which is the multi-GPU version of the same de-duplication idea this post is built on. The post's own caveat: effectiveness depends on workload shape.

**MoRIIO — where you must turn it off.** The [MoRIIO KV connector post](https://vllm.ai/blog/2026-04-07-moriio-kv-connector) (2026-04-07) reports 2.5× higher goodput than a collocated baseline for Qwen3-235B-A22B-FP8 on 8 MI300X split 4 prefill plus 4 decode, under an SLO of TTFT under 1 second and inter-token latency under 50 ms. The relevant detail here is a limitation, not a win: that configuration requires prefix caching to be **disabled** (`--no-enable-prefix-caching`). Prefix caching is not universally compatible with every disaggregation scheme, because a connector that streams KV from a prefill node to a decode node has its own idea of who owns which blocks. When two mechanisms both want to control block lifetime, one of them has to yield.

| Result | Setup | Source |
| --- | --- | --- |
| <1% throughput loss at 0% hit rate | vLLM V1, hash + LRU, default on | cited: vLLM V1 post |
| 92.2% vs 1.7% hit rate; 3.8× throughput; 46× lower P50 TTFT | Kimi-2.5 NVFP4, 1P1D on 12 GB200, Codex traces | cited: Mooncake Store post |
| 52.35% vs 11.77% hit; +56% throughput; −36% TTFT | Qwen3-8B, single host | cited: PegaFlow post |
| 97.23% vs 65.18% hit; +72% throughput; −41% TTFT | DeepSeek-V3.2 MLA, TP8 | cited: PegaFlow post |
| 2.5× goodput, prefix caching disabled | Qwen3-235B-A22B-FP8, 8×MI300X, 4P+4D | cited: MoRIIO post |
| 17.0× fewer prefill tokens over a 33-turn trace | 2,242 tok/turn growth, perfect sharing | derived from Mooncake trace stats |
| 1.48 GB saved, ~6 µs copy cost, n=4 sampling | Llama-3.1-8B bf16, 4,090-token prompt, A100 | derived |

---

## 11. When to reach for this, and when not

![A comparison of five workload types against shared prefix size, expected hit rate, and whether prefix sharing is worth enabling](/imgs/blogs/prefix-sharing-radix-trees-and-copy-on-write-7.webp)

**Turn it on, always, if you have any fixed preamble.** The overhead is derived above at well under a percent and independently bounded by vLLM's under-1% design target. Any system prompt longer than a couple of blocks pays for it immediately. There is no threshold to cross and no tuning to do.

**Build it yourself only to learn it.** If you are running vLLM or SGLang, prefix caching is already on by default and better than what you will write in an afternoon — theirs handles distributed stores, multimodal hashes, LoRA scoping, and the eviction interactions with preemption. Write `nanoserve/kv/prefix.py` because it makes you able to *debug* theirs, to reason about your hit rate, and to know which prompt-template change just cost you 30% of your prefill savings. That is a much better return than the code itself.

**Restructure your prompts before you tune your cache.** The highest-leverage action is not an engine flag. It is moving the timestamp, the user id, and the session token out of the top of your prompt template. A fixed preamble followed by variable content is the shape the whole mechanism rewards, and no amount of allocator tuning recovers what a leading variable destroys.

**Do not build a distributed KV store as your first move.** Mooncake and PegaFlow exist because single-node caches saturate and evict, but that is a problem you have only after single-node caching is working and measured. Measure your ceiling with a sketch, measure your achieved rate, and only reach across the network when the gap is capacity rather than policy.

**Turn it off, or salt it, when isolation beats throughput.** In a multi-tenant service where prompts may contain another customer's data, cross-tenant sharing is a timing channel, and the honest options are per-tenant salts or per-tenant partitions. Both cost hit rate. Take the cost knowingly rather than discovering the channel from a disclosure report.

**Expect nothing from it on unique-document batches.** Corpus-wide summarization with no shared instruction block, one-off document extraction, and any workload where every prompt begins differently will sit at a near-zero hit rate forever. Leave the feature on (it is nearly free), but do not budget for a speedup that the workload cannot produce.

**Check compatibility with your disaggregation setup.** As the MoRIIO result shows, some prefill-decode disaggregation connectors require prefix caching to be disabled. Verify against your actual connector before assuming the two compose.

---

## 12. Key takeaways

1. Keys and values at position $i$ depend only on tokens ${0..i}$, so any two requests sharing a prefix share those tensors exactly. Prefix caching is a consequence of causal masking, not a heuristic.
2. Hash a block as a function of its parent's hash plus its own tokens. That chain makes a single dictionary probe a proof about the entire prefix to the left. Verify token ids on hit, or accept a birthday-bound collision risk that becomes real at hundreds of millions of blocks.
3. Only complete blocks are cacheable. You always recompute $S \bmod B$ tokens, so larger block sizes cost you sharing granularity — an expected $(B-1)/2$ tokens per hit.
4. Reference counts do the work: positive means pinned, zero means free-but-still-cached. A freed block that keeps its index entry is where the hit rate actually comes from. Free tail-first so the shared head survives eviction longest.
5. Copy-on-write fires only when a sequence writes into a *shared partial* block, which happens on forks — `n > 1` sampling, beam search, agent branching. For Llama-3.1-8B that copy is 2 MB, roughly 2 microseconds on an A100, and about 4,000× cheaper than recomputing the block.
6. Savings are ${1 - f}$ on linear layers and $1 - f^2$ on attention, where $f$ is the cached fraction. Attention keeps proportionally more work, because uncached queries still read every cached key.
7. The overhead at 0% hit rate is a few hundred microseconds of hashing per long prompt — under 0.1% of prefill. vLLM V1's design bar of under 1% is what makes default-on defensible.
8. The cache key must contain everything that changes K and V: model revision, dtype and quantization scheme, LoRA adapter, multimodal input hashes, and — if you need tenant isolation — a per-tenant salt. Sampling parameters must not.
9. Hit rate is a property of the workload. Agentic traces with a 131:1 input-to-output ratio reach 92%+; unique-document batches reach zero. Measure the ceiling before tuning the policy.
10. A shared cache plus a stopwatch is an oracle for whether a prefix was cached. Treat cross-tenant sharing as a security decision, not a performance one.

---

## Further reading

- [Inside vLLM: Anatomy of a High-Throughput Inference System](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) — the block hash construction, `find_longest_cache_hit`, the free-block queue, and the partial-block recompute rule, stated by the people who wrote them.
- [vLLM V1: A Major Upgrade to vLLM's Core Architecture](https://vllm.ai/blog/2025-01-27-v1-alpha-release) — the zero-overhead prefix-caching claim and the scheduler design that makes prefix caching, chunked prefill, and speculative decoding one policy.
- [Mooncake Store: Distributed KV Cache for Agentic Workloads](https://vllm.ai/blog/2026-05-06-mooncake-store) — hit rates and trace statistics for agentic serving, plus the honest limitation that decode does not read from the pool.
- [PegaFlow: External KV Cache Service](https://vllm.ai/blog/2026-05-18-pegaflow) — three-tier KV storage, the HyperLogLog hit-rate ceiling estimator, and chat-versus-long-context hit rates side by side.
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — the original paper that introduced block sharing with reference counting and copy-on-write for parallel sampling and beam search.
- [Prefix caching and RadixAttention](/blog/machine-learning/model-serving/prefix-caching-and-radixattention) — the operator's view of the same mechanism as shipped in vLLM and SGLang.
- [KV cache optimization](/blog/machine-learning/model-serving/kv-cache-optimization) — the broader menu of cache-size reductions this technique sits alongside.
- Within this series: [what inference engineering is](/blog/machine-learning/inference-engineering/what-inference-engineering-is) for the layer map, [the paged KV cache](/blog/machine-learning/inference-engineering/paged-kv-cache-implementing-blocks-and-a-block-table) for the allocator this post extends, [eviction, preemption, and KV swapping](/blog/machine-learning/inference-engineering/eviction-preemption-and-kv-swapping) for what happens when the pool runs dry, and [the inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook) for how every piece fits together.
