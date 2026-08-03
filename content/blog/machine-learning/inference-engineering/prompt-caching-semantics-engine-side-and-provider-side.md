---
title: "Prompt caching semantics at the engine and provider: Prefixes, TTLs, and cacheable prompts"
date: "2026-08-03"
publishDate: "2026-08-03"
description: "Build a nanoserve prefix cache, reason about provider prompt-cache keys and TTLs, and design prompts that stay reusable without making correctness or privacy accidental."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "prompt-caching",
    "prefix-caching",
    "kv-cache",
    "latency",
    "throughput",
    "cost-model",
    "ml-systems",
    "vllm",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 37
---

A prompt can be identical to you and still be a cache miss.

That sentence is the source of a surprising number of expensive production incidents. An inference engine may have the first 8,192 tokens resident as GPU KV blocks, while the upstream model provider charges you for every input token. Or the provider may report cached input tokens while your local engine recomputes the prefix because the request landed on a different replica. The word *cache* hides two different ownership boundaries.

![A request crosses an engine-side KV prefix cache and a provider-side prompt cache with different keys and lifetimes](/imgs/blogs/prompt-caching-semantics-engine-side-and-provider-side-1.webp)

The diagram is the mental model: an engine cache answers “can this worker reuse already-prefilled KV tensors?”, while provider caching answers “can this provider reuse a prefix representation for this model, organization, and cache policy?” The two decisions can compose, but neither is a proof of the other. By the end of this post we will add a content-addressed prefix index to `nanoserve`, instrument hit and miss reasons, derive the cost of a hit, and build a prompt layout that remains stable when user text, timestamps, tools, and retrieval results change.

This post is #47 in [Inference Engineering — Build Your Own LLM Inference Engine](/blog/machine-learning/inference-engineering/what-inference-engineering-is). It connects the local KV work in [prefix sharing and radix trees](/blog/machine-learning/inference-engineering/prefix-sharing-radix-trees-and-copy-on-write) to the API edge. The capstone [Inference Engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook) will use the same accounting vocabulary across local and hosted engines.

> A cache key is a correctness boundary with a speed benefit, not a speed trick with a hash attached.

## 1. Two caches, two owners, two questions

The local cache lives beside the model weights and KV allocator. During prefill, the model turns input token IDs into per-layer keys and values. A later request with the same token prefix can attach to those blocks and start attention at the first new token. The cache stores tensors, not merely text. It is tied to model weights, tokenizer behavior, dtype, attention layout, device placement, and the exact worker that owns the blocks.

The provider cache is outside the process. A provider receives a serialized request, canonicalizes enough of it to identify a reusable prefix, and may reuse internal representations. The representation might be KV tensors, a compiled prompt artifact, or another private form. Your application sees usage counters and latency; it does not own the storage or the eviction policy. Provider docs describe the matching and billing contract, not the implementation.

| Property | Engine-side prefix cache | Provider-side prompt cache | Source |
| --- | --- | --- | --- |
| Owner | `nanoserve` worker or engine cluster | Model provider | derived from ownership boundary |
| Reusable object | KV blocks for a token prefix | Provider-managed prefix representation | derived from API boundary |
| Key material | model revision, tokenizer, dtype/layout, token IDs, prior block hash | provider request fields, model, organization/project, explicit breakpoints or provider policy | cited: [OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching), 2026-08-03; [Anthropic prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching), 2026-08-03 |
| Visibility | exact block IDs and hit length are observable locally | cached-token usage fields and provider headers are observable; physical placement is not | cited: OpenAI and Anthropic docs, 2026-08-03 |
| Eviction | local memory pressure, LRU/LFU/admission policy, model reload | provider TTL, inactivity, capacity, policy, or undocumented eviction | cited: Anthropic docs, 2026-08-03; otherwise provider-owned |
| Invalidation | token or model identity changes; block is evicted or overwritten | any provider-defined key input or TTL expiry; exact rules differ by provider | cited: provider docs, 2026-08-03 |
| Main win | lower TTFT and less prefill compute | lower billed input tokens and often lower TTFT | derived from work avoided |

The first engineering rule follows: record both counters. `engine_prefix_hit_tokens` measures local work avoided. `provider_cached_input_tokens` measures what the provider says it reused. A single “cache hit rate” metric is ambiguous and will mislead an incident review.

### The same prompt is not necessarily the same byte sequence

At the engine boundary, we should key on token IDs, not the source string. Unicode normalization, chat-template whitespace, special tokens, and tokenizer version can map visually similar strings to different IDs. A model can also have different tokenization after a tokenizer upgrade. If the IDs differ, reusing KV tensors is incorrect even if the screen rendering looks identical.

At the provider boundary, the serialized request matters. OpenAI says cache hits require exact prefix matches and that static instructions, images, and tools should come first; it also exposes `cached_tokens` in usage. Anthropic describes a hierarchy of `tools`, `system`, then `messages`, with an explicit `cache_control` breakpoint defining the prefix. Those are similar design principles, not one shared protocol.

## 2. The engine key: cumulative identity, not a string hash

The simplest local prefix key is a hash of the whole token prefix. That works functionally, but it makes every lookup touch a growing byte array and gives the cache no natural way to share blocks between requests. The paged design from the earlier series posts uses fixed-size blocks and a cumulative hash.

Let a request tokenize to blocks $B_0, B_1, \ldots, B_{m-1}$. Define an engine block identity as

$$
h_i = H(h_{i-1} \mathbin{\Vert} B_i \mathbin{\Vert} M), \qquad h_{-1}=H(\text{namespace}),
$$

where $H$ is a cryptographic or collision-resistant hash, $\mathbin{\Vert}$ is concatenation, and $M$ is the model and representation namespace. `B_i` must include the token IDs in order; `M` must include every fact that changes the meaning of the KV tensor. In a practical `nanoserve`, that includes a model revision, tokenizer revision, dtype, block size, attention implementation version, and tensor-parallel rank.

The cumulative term is important. If block 2 changes, then $h_2$ and every later hash changes. A request can still reuse blocks 0 and 1, but it must not reuse block 3 from the old branch. This gives a clean longest-common-prefix algorithm: compare cumulative identities from the start until the first mismatch.

### A runnable key implementation

The following file is intentionally boring. The cache should be boring. It is a security and correctness component, so explicit bytes and explicit namespaces beat clever object serialization.

```python
# nanoserve/prefix_cache.py
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct
from typing import Iterable


def _pack_tokens(tokens: Iterable[int]) -> bytes:
    values = tuple(int(token) for token in tokens)
    if any(token < 0 or token >= 2**32 for token in values):
        raise ValueError("token IDs must fit uint32")
    return struct.pack(f">{len(values)}I", *values)


def block_digest(namespace: bytes, parent: bytes, tokens: Iterable[int]) -> bytes:
    payload = namespace + b"\0" + parent + b"\0" + _pack_tokens(tokens)
    return hashlib.blake2b(payload, digest_size=32).digest()


@dataclass(frozen=True)
class CacheNamespace:
    model_revision: str
    tokenizer_revision: str
    dtype: str
    block_size: int
    layout_revision: str

    def bytes(self) -> bytes:
        fields = (
            self.model_revision,
            self.tokenizer_revision,
            self.dtype,
            str(self.block_size),
            self.layout_revision,
        )
        return "|".join(fields).encode("utf-8")


def prefix_digests(tokens: list[int], namespace: CacheNamespace) -> list[bytes]:
    block = namespace.block_size
    parent = hashlib.blake2b(namespace.bytes(), digest_size=32).digest()
    result: list[bytes] = []
    for start in range(0, len(tokens), block):
        current = tokens[start : start + block]
        parent = block_digest(namespace.bytes(), parent, current)
        result.append(parent)
    return result


if __name__ == "__main__":
    ns = CacheNamespace("llama-3.1-8b@a", "llama-tokenizer@b", "bf16", 4, "kv-v1")
    left = prefix_digests([10, 11, 12, 13, 14, 15, 16], ns)
    right = prefix_digests([10, 11, 12, 13, 99, 15, 16], ns)
    common = sum(a == b for a, b in zip(left, right))
    print(f"common blocks={common}; left={len(left)} right={len(right)}")
```

Expected output is `common blocks=1; left=2 right=2`. That output is derived from the four-token block size and the one changed token; it is not a GPU benchmark.

### Why a provider cache key must not be copied into the engine

A provider may expose a `prompt_cache_key` or an explicit breakpoint. That identifier is useful for routing and grouping requests at the provider boundary, but it is not a digest of the local KV tensor. The provider can include organization, model deployment, region, safety configuration, tool schema, and internal versioning in its namespace. Conversely, a local key must include the exact representation details needed to interpret the bytes on this worker.

Do not use a user-controlled provider key as the only local cache namespace. Two tenants could deliberately choose the same label, or an application could reuse a label after a model deployment. Treat it as an input to routing and observability, never as proof that tensors are interchangeable.

## 3. Prefix stability: put entropy after the breakpoint

Cacheability is mostly prompt layout. A prefix is stable when every token before the cache boundary remains identical across the requests that should share it. The practical ordering is:

1. model and version selection outside the prompt;
2. stable system instructions;
3. stable tool definitions and schemas;
4. stable few-shot examples or reference corpus;
5. session-specific but stable context;
6. retrieved documents, user message, current time, and nonce at the suffix.

![A decision tree places stable instructions before variable retrieval and user content so the cache breakpoint stays reusable](/imgs/blogs/prompt-caching-semantics-engine-side-and-provider-side-4.webp)

The common mistake is to place a timestamp or request ID inside the largest “system prompt” string and then declare the whole string cacheable. One changed byte changes the token sequence after tokenization, which changes the cumulative identity at that point. If the cache was only written at the end of that string, there may be no earlier local block to reuse. Anthropic explicitly warns that a breakpoint on changing content causes repeated writes; OpenAI’s current guidance likewise recommends a stable prefix followed by variable content.

### Stable does not mean semantically immutable forever

A system prompt that changes once per deployment can be stable for a request cohort but not across deployments. Make the deployment version explicit in the namespace. This is better than silently reusing old blocks because the content “usually” looks the same.

The same applies to tool schemas. Changing a parameter description can change tool serialization and therefore the provider prefix. Changing the available tool list can invalidate all later material. Anthropic documents this hierarchy directly: changing tool definitions invalidates tool, system, and message caches. In an engine, the safe equivalent is to put a canonical tool-schema digest into the namespace and into the prompt’s stable prefix.

### Canonicalization code for the API edge

Canonical JSON is not cosmetic. Dictionary insertion order, whitespace, number formatting, and Unicode normalization can make logically equivalent tool definitions produce different bytes.

```python
# nanoserve/prompt_identity.py
from __future__ import annotations

import hashlib
import json
import unicodedata
from typing import Any


def canonical_json(value: Any) -> bytes:
    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return unicodedata.normalize("NFC", text).encode("utf-8")


def stable_prompt_namespace(
    *, model: str, deployment: str, tokenizer: str, tools: list[dict[str, Any]]
) -> str:
    digest = hashlib.sha256(canonical_json(tools)).hexdigest()
    return f"{model}@{deployment}|tokenizer={tokenizer}|tools={digest}"


if __name__ == "__main__":
    tools_a = [{"name": "search", "parameters": {"type": "object"}}]
    tools_b = [{"parameters": {"type": "object"}, "name": "search"}]
    print(stable_prompt_namespace(model="llama-3.1-8b", deployment="blue", tokenizer="v1", tools=tools_a))
    print(stable_prompt_namespace(model="llama-3.1-8b", deployment="blue", tokenizer="v1", tools=tools_a)
          == stable_prompt_namespace(model="llama-3.1-8b", deployment="blue", tokenizer="v1", tools=tools_b))
```

The second line prints `True`: key order changes, but the canonical byte representation does not. This is a reader-runnable correctness check, not a claim about provider canonicalization. You own canonicalization before the request leaves your service; you do not own the provider’s internal serialization.

## 4. The animated boundary: fill, hit, change, expire

![An animated sequence shows a prefix filling the local KV cache, a later suffix hitting it, then a changed prefix and TTL expiry forcing prefill again](/imgs/blogs/prompt-caching-semantics-engine-side-and-provider-side-2.webp)

<figure class="blog-anim">
<svg viewBox="0 0 760 230" role="img" aria-label="A stable prefix fills a local cache, a later request reuses it, then a changed suffix and expiry cause a miss" style="width:100%;height:auto;max-width:900px">
<style>
.pc-stage{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:2}
.pc-label{font:600 17px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.pc-small{font:500 14px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.pc-block{fill:var(--accent,#6366f1);opacity:.82}
.pc-suffix{fill:var(--border,#d1d5db)}
@keyframes pc-fill{0%,15%{transform:translateX(0);opacity:.15}35%,55%{transform:translateX(180px);opacity:.95}70%,100%{transform:translateX(360px);opacity:.18}}
@keyframes pc-hit{0%,40%{opacity:.1}50%,68%{opacity:1}78%,100%{opacity:.1}}
@keyframes pc-miss{0%,70%{opacity:.1}82%,100%{opacity:1}}
.pc-moving{animation:pc-fill 9s ease-in-out infinite}
.pc-hit{animation:pc-hit 9s ease-in-out infinite}
.pc-miss{animation:pc-miss 9s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.pc-moving,.pc-hit,.pc-miss{animation:none}.pc-moving{transform:translateX(180px);opacity:.95}.pc-hit{opacity:1}.pc-miss{opacity:.1}}
</style>
<rect class="pc-stage" x="28" y="55" width="150" height="92" rx="12"/>
<rect class="pc-stage" x="305" y="55" width="150" height="92" rx="12"/>
<rect class="pc-stage" x="582" y="55" width="150" height="92" rx="12"/>
<text class="pc-label" x="103" y="92">request</text>
<text class="pc-small" x="103" y="119">stable + suffix</text>
<text class="pc-label" x="380" y="92">engine KV</text>
<text class="pc-small" x="380" y="119">prefix blocks</text>
<text class="pc-label" x="657" y="92">next request</text>
<text class="pc-small" x="657" y="119">hit or miss</text>
<rect class="pc-block pc-moving" x="178" y="82" width="125" height="38" rx="8"/>
<rect class="pc-block pc-hit" x="333" y="82" width="94" height="38" rx="8"/>
<rect class="pc-suffix pc-miss" x="430" y="82" width="22" height="38" rx="5"/>
<text class="pc-small" x="380" y="181">same prefix → reuse; changed token or expired entry → prefill again</text>
</svg>
<figcaption>The moving prefix is reusable only until a changed suffix or expiry breaks the cache contract.</figcaption>
</figure>

The motion makes one lifecycle visible: a first request writes a prefix; a second request reuses it and computes only the suffix; a changed token breaks the cumulative chain; an expired entry is a miss even if the text is identical again. The provider’s cache may follow a different lifecycle at the same time.

For an engine, “TTL” is often an application policy layered over memory eviction. An entry can be younger than its TTL and still disappear because GPU blocks are needed by a live request. A local TTL is therefore a freshness upper bound, not a residency guarantee. For a provider, the documented phrase may have a different contract. Anthropic’s default prompt cache lifetime is five minutes and use refreshes it at no additional cost; its one-hour duration is a separately priced option. OpenAI’s current GPT-5.6 documentation describes a 30-minute minimum lifetime that may be retained longer, while older model families expose different in-memory and extended-retention controls. Gemini distinguishes implicit caching from explicit cache objects: its documentation says implicit caching is enabled on newer models, while explicit caches have a configurable TTL defaulting to one hour on the referenced page.

Do not normalize those statements into “all provider caches last five minutes.” Put the provider, model family, date, and policy in your runbook.

### Local lookup with explicit miss reasons

Counters without reasons create false optimism. A low hit rate because prompts are unstable is a product problem; a low hit rate because the cache is full is a capacity problem; a low hit rate because the tokenizer revision changed is a deployment problem.

```python
# nanoserve/prefix_index.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import time


class Miss(str, Enum):
    EMPTY = "empty"
    NAMESPACE = "namespace"
    TOKEN = "token"
    EVICTED = "evicted"
    EXPIRED = "expired"


@dataclass
class Entry:
    digest: bytes
    block_ids: tuple[int, ...]
    created_at: float
    last_used_at: float


class PrefixIndex:
    def __init__(self, ttl_seconds: float = 300.0):
        self.ttl_seconds = ttl_seconds
        self.entries: dict[bytes, Entry] = {}
        self.hits = 0
        self.misses: dict[Miss, int] = {reason: 0 for reason in Miss}

    def lookup(self, digests: list[bytes], now: float | None = None) -> int:
        now = time.monotonic() if now is None else now
        longest = 0
        for digest in digests:
            entry = self.entries.get(digest)
            if entry is None:
                self.misses[Miss.EMPTY] += 1
                break
            if now - entry.last_used_at > self.ttl_seconds:
                self.entries.pop(digest, None)
                self.misses[Miss.EXPIRED] += 1
                break
            entry.last_used_at = now
            longest += 1
        if longest:
            self.hits += 1
        return longest

    def insert(self, digest: bytes, block_ids: tuple[int, ...], now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        self.entries[digest] = Entry(digest, block_ids, now, now)
```

Production code must distinguish “no entry” from “entry evicted” by retaining metadata or a bounded tombstone set. The compact example reports expiry and empty separately; a real allocator should attach the exact reason when an eviction frees block IDs.

### TTL and prewarming are workload decisions

If a shared system prefix is used every 30 seconds, a five-minute provider cache can refresh naturally. If an agent resumes after 12 minutes, a one-hour cache may be worth its write premium. Prewarming a cache is not free: it consumes provider input tokens, local prefill capacity, and possibly rate-limit budget. Prewarm only when the expected reuse exceeds the write and storage cost.

## 5. What invalidates each side

![A side-by-side invalidation matrix shows local KV invalidation and provider prompt-cache invalidation cascading from different inputs](/imgs/blogs/prompt-caching-semantics-engine-side-and-provider-side-3.webp)

![A timeline separates local eviction, provider TTL expiry, reuse refresh, and the recorded miss reason](/imgs/blogs/prompt-caching-semantics-engine-side-and-provider-side-5.webp)

The two invalidation graphs overlap but are not identical. A model revision change invalidates both in a correctly isolated system. A local GPU eviction invalidates only the local copy. A provider region change may invalidate the provider entry while leaving local blocks untouched. A prompt serialization change may invalidate the provider cache while the local token IDs remain identical if serialization happens outside the engine’s tokenizer path.

| Change | Local KV prefix | OpenAI-style provider prefix | Anthropic-style provider prefix | Source |
| --- | --- | --- | --- | --- |
| New model revision | invalid | model/request namespace differs; treat as miss | model differs; treat as miss | derived safe rule |
| New tokenizer or chat template | invalid | serialized input differs | message content differs | derived from exact prefix requirement |
| One token before boundary changes | that block and descendants invalid | exact prefix no longer matches | cached prefix and later hierarchy invalid | cited: provider docs, 2026-08-03 |
| Tool schema changes | namespace digest invalid | tools are part of cacheable prefix | tool, system, and message caches invalid | cited: [Anthropic invalidation table](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching), 2026-08-03 |
| User suffix changes after boundary | earlier blocks remain reusable | reusable prefix can remain; suffix is fresh | earlier breakpoint can remain; later message region changes | cited: provider docs, 2026-08-03 |
| Local block eviction | invalid on this worker | no effect | no effect | derived from ownership |
| Provider TTL expiry | no effect | miss | miss | cited: provider docs, 2026-08-03 |
| Different tenant or organization | local namespace must isolate tenants if data is private | provider caches are organization-isolated per docs | provider caches are organization/workspace-isolated per docs | cited: provider docs, 2026-08-03 |

The privacy row deserves emphasis. Sharing a prefix is a data-sharing decision even when the bytes are “only KV.” If tenant A’s private retrieval document can influence tenant B’s continuation through a reused tensor, the cache is a data leak. The safest default is tenant-scoped namespaces and explicit opt-in for cross-tenant public prefixes.

### Never infer invalidation from a miss alone

A miss proves only that this request did not reuse the entry. It does not prove why. A deployment can record a structured `CacheMissReason` locally, but provider APIs generally expose counters rather than internal reason codes. If a provider’s cached-token count drops, compare model, region, tools, request shape, breakpoint, TTL, and inter-request gap before changing prompt text.

## 6. Cost model: a hit must pay for its own existence

Prompt caching changes both token economics and latency. The local cache saves prefill work but consumes VRAM and may cause evictions or admission pressure. A provider cache can reduce the rate for repeated input tokens but may charge for cache writes and storage. The right decision is a break-even calculation, not “caching is cheaper.”

Let:

- $P_u$ be uncached input price per token;
- $P_r$ be cached-read price per token;
- $P_w$ be cache-write price per token;
- $S$ be the stable prefix token count;
- $N$ be the number of requests in the cache lifetime;
- $C_s$ be storage cost for the cached prefix over that lifetime.

Ignoring output tokens, uncached cost is $C_0 = N S P_u$. A single write followed by $N-1$ reads costs

$$
C_1 = S P_w + (N-1)S P_r + C_s.
$$

Caching wins when $C_1 < C_0$. Rearranging gives

$$
N > \frac{P_w - P_r + C_s/S}{P_u - P_r}.
$$

This is an explanatory abstraction, not a universal provider formula: providers can add minimums, model tiers, storage rounding, or different billing locations. It is still useful because it forces us to name the variables.

![A cost break-even chart compares uncached input cost with one write plus repeated cached reads as request reuse increases](/imgs/blogs/prompt-caching-semantics-engine-side-and-provider-side-6.webp)

### Worked example: a cited price multiplier

Anthropic’s pricing page states that a five-minute cache write is 1.25 times base input price, a one-hour write is 2 times base input price, and a cache read is 0.1 times base input price for the listed models. Use the five-minute case and temporarily set storage cost to zero to isolate token pricing. Then $P_w=1.25P_u$ and $P_r=0.1P_u$:

$$
N > \frac{1.25P_u - 0.1P_u}{P_u - 0.1P_u}
  = \frac{1.15}{0.9}
  = 1.278\ldots
$$

So two requests are enough to beat uncached token pricing in this simplified example, before considering minimum token thresholds, latency, or the cost of keeping a local cache warm. The rates are cited from [Anthropic pricing](https://docs.anthropic.com/en/docs/about-claude/pricing), accessed 2026-08-03; the arithmetic is derived here. Do not copy this threshold to OpenAI or Gemini. Their current controls and price schedules differ.

### Local cost: VRAM is not free capacity

For Llama-3.1-8B, using the series’ fixed example of 32 layers, 8 KV heads, head dimension 128, and bf16, KV bytes per token are

$$
2 \times 32 \times 8 \times 128 \times 2 = 131{,}072\text{ bytes} = 128\text{ KiB}.
$$

The `2` at the front is K plus V; the final `2` is bytes per bf16 value. A 4,096-token reusable prefix therefore occupies

$$
4{,}096 \times 131{,}072 = 536{,}870{,}912\text{ bytes} = 512\text{ MiB}
$$

per replica before allocator metadata. Ten such prefixes are about 5 GiB. The number is derived from the model dimensions and dtype, not measured on a GPU. On a 24 GB RTX 4090, reserving 5 GiB for shared prefixes can reduce the headroom available for active continuations; the cache is beneficial only if the saved prefill work and hit frequency justify that trade.

## 7. Designing the `nanoserve` cache interface

The cache belongs between tokenization and scheduling. Tokenization produces IDs; the prefix index returns reusable block IDs; the scheduler admits the remaining suffix as prefill work. The scheduler must still account for the full logical sequence length because attention over the reused prefix participates in decode, even if prefill was skipped.

```python
# nanoserve/engine.py
from dataclasses import dataclass

from .prefix_cache import CacheNamespace, prefix_digests


@dataclass(frozen=True)
class PrefixMatch:
    block_ids: tuple[int, ...]
    hit_tokens: int
    miss_block: int


def find_prefix_match(
    tokens: list[int],
    namespace: CacheNamespace,
    index,
    block_table: dict[bytes, tuple[int, ...]],
) -> PrefixMatch:
    digests = prefix_digests(tokens, namespace)
    block_count = index.lookup(digests)
    reused: list[int] = []
    for digest in digests[:block_count]:
        reused.extend(block_table[digest])
    hit_tokens = min(len(tokens), block_count * namespace.block_size)
    return PrefixMatch(tuple(reused), hit_tokens, block_count)


def prefill_slice(tokens: list[int], match: PrefixMatch, block_size: int) -> list[int]:
    return tokens[match.hit_tokens:]
```

The critical invariant is that `hit_tokens` ends on a complete reusable block. Partial blocks are dangerous: a block may contain capacity beyond the logical prefix, and its unused positions may later be written by another request. Either pad and hash only complete blocks, or implement copy-on-write for partial blocks. The earlier [paged KV cache implementation](/blog/machine-learning/inference-engineering/paged-kv-cache-implementing-blocks-and-a-block-table) makes this physical/logical distinction explicit.

### Admission policy: not every prefix deserves residency

Caching every prefix turns the cache into a graveyard of one-hit wonders. A useful admission policy tracks a reuse score:

$$
\text{value} = \text{expected future hits} \times \text{prefill tokens saved} \times \text{prefill cost per token} - \text{VRAM opportunity cost}.
$$

This reuse-score equation is an explanatory model, not a formula claimed by a specific engine. The implementation can approximate it with TinyLFU admission, a minimum reuse count, or a weighted LRU. Public vLLM work on external KV stores similarly treats workload shape and hit rate as first-order factors; the vLLM team reports workload-dependent gains for PegaFlow, including a Qwen3-8B setup with 52.35% hit rate and a DeepSeek-V3.2 MLA setup with 97.23% hit rate, in its [PegaFlow post](https://vllm.ai/blog/2026-05-18-pegaflow) dated 2026-05-18. Those are cited results with their setup, not predictions for `nanoserve`.

### Copy-on-write is mandatory for branching

If two requests share a prefix and one request appends a token, the shared blocks must become immutable. The first writer allocates a private tail block. Without copy-on-write, a generation branch can corrupt another request’s context. This is the same rule as a filesystem snapshot: sharing saves storage only while shared pages remain read-only.

```python
# nanoserve/cow.py
from dataclasses import dataclass


@dataclass
class Block:
    block_id: int
    refcount: int = 1
    tokens: list[int] | None = None


def acquire(block: Block) -> Block:
    block.refcount += 1
    return block


def writable(block: Block, allocator) -> Block:
    if block.refcount == 1:
        return block
    block.refcount -= 1
    clone = allocator.allocate()
    clone.tokens = list(block.tokens or [])
    return clone


if __name__ == "__main__":
    class Allocator:
        def allocate(self):
            return Block(9)
    original = Block(4, tokens=[10, 11])
    acquire(original)
    private = writable(original, Allocator())
    private.tokens.append(12)
    print(original.block_id, original.tokens, private.block_id, private.tokens)
```

Expected output is `4 [10, 11] 9 [10, 11, 12]`. The example demonstrates correctness, not throughput.

## 8. Provider adapters: observe, do not guess

A provider adapter should normalize request construction and usage fields into an internal record while preserving provider-specific semantics. Do not make the adapter pretend that OpenAI’s `cached_tokens`, Anthropic’s `cache_read_input_tokens`, and Gemini’s `total_cached_tokens` are interchangeable guarantees. They answer related but not identical questions.

```python
# nanoserve/provider_usage.py
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProviderCacheUsage:
    provider: str
    cached_input_tokens: int
    cache_write_tokens: int
    ttl_policy: str | None
    raw: dict[str, Any]


def parse_usage(provider: str, usage: dict[str, Any]) -> ProviderCacheUsage:
    if provider == "openai":
        details = usage.get("prompt_tokens_details", usage.get("input_tokens_details", {}))
        return ProviderCacheUsage(provider, details.get("cached_tokens", 0), usage.get("cache_write_tokens", 0), None, usage)
    if provider == "anthropic":
        return ProviderCacheUsage(provider, usage.get("cache_read_input_tokens", 0), usage.get("cache_creation_input_tokens", 0), None, usage)
    if provider == "gemini":
        return ProviderCacheUsage(provider, usage.get("total_cached_tokens", 0), 0, None, usage)
    raise ValueError(f"unsupported provider: {provider}")
```

The parser intentionally keeps `raw`. Billing dashboards need the original response, request model, region, and timestamp to reconcile changes in provider schema. Store a hash of sensitive prompt material for correlation, not the prompt itself, unless your retention policy permits it.

### Provider-specific operational notes

OpenAI’s current prompt-caching guide, accessed 2026-08-03, says eligible requests use exact prompt prefixes, static content should come first, and tools and images must be identical for a hit. It documents a strict 1,024-token minimum for GPT-5.6 and later, and describes explicit breakpoints and `prompt_cache_key` for controlling stable prefixes. Its current GPT-5.6 section says `prompt_cache_options.ttl` is a 30-minute minimum, not a maximum retention promise. Earlier models expose different retention controls, so an adapter must be model-aware.

Anthropic’s current guide, accessed 2026-08-03, defines the prefix order `tools → system → messages`, explicit `cache_control` breakpoints, a 20-block lookback for finding prior writes, and five-minute default or one-hour extended TTL. It reports cache-read and cache-creation input-token fields. The 20-block lookback is particularly operational: adding a breakpoint to a stable region can matter more than merely keeping the text stable.

Gemini’s current context-caching documentation, accessed 2026-08-03, separates implicit caching from explicit cache objects. The page says implicit caching is available by default on Gemini 2.5 and newer models, recommends large common content at the beginning, and exposes cached token counts in usage metadata. Its explicit cache flow has a user-controlled TTL, with the referenced generate-content documentation giving one hour as the default when omitted. The exact product surface and price table can change, so store the API version and documentation date with your pricing configuration.

## 9. Worked request flows and failure modes

#### Worked example: one local hit, one provider miss

Request A contains 2,048 stable tokens and 64 user tokens. `nanoserve` tokenizes with tokenizer revision `v7`, writes eight 256-token blocks, and forwards the full request to a provider. Request B has the same local token IDs but the application adds a current timestamp to the provider’s system string before sending it. Locally, the engine sees the same stable prefix and reuses eight blocks; its derived local prefill saving is $2{,}048$ tokens. Provider-side, the serialized prefix differs at the timestamp, so the provider may report zero cached input tokens. The local and remote counters correctly disagree.

#### Worked example: provider hit, local miss after a replica move

Request C and D use the same stable prompt at a hosted API. The provider reports 2,048 cached input tokens on D. Between calls, a local gateway routes D to a different `nanoserve` replica whose GPU cache is empty. The engine recomputes 2,048 tokens, while the provider may reuse its own representation. The provider hit saved provider-side billing and perhaps provider prefill; it did not save the local GPU’s work. If your latency dashboard only tracks provider usage, it will miss the gateway regression.

### Failure mode: cache key collision or namespace omission

Cryptographic collisions are not the practical threat; namespace omission is. If `dtype` is absent, a bf16 KV block can be presented to a fp16 path. If `layout_revision` is absent, a kernel update can reinterpret strides. If tenant identity is absent, private context can be shared. Hashes make bad inputs look authoritative. Construct the namespace first, test it in a compatibility matrix, and fail closed when a required field is missing.

### Failure mode: chat-template drift

Two clients may send the same visible conversation but use different templates: one inserts a generation marker, one adds a trailing newline, and one emits different role tokens. The provider sees different serialized messages; the engine sees different IDs. Pin the template revision and log token counts plus a short non-reversible prefix digest. A full prompt log is often a privacy incident waiting to happen.

### Failure mode: volatile retrieval in the stable region

Putting retrieved documents before the cache breakpoint is attractive because it feels like “context.” It is also a cache killer when retrieval ordering, chunk IDs, or metadata change per request. If the corpus is genuinely reused, canonicalize chunk ordering and version the corpus snapshot. If it is user-specific, keep it after the shared system and tool region. The goal is not maximum cached tokens; it is maximum *reusable* cached tokens without stale data.

### Failure mode: retries that double-write

A timeout after the provider accepted a cache write can cause a retry. If the retry has a different request ID or timestamp before the breakpoint, it writes again and misses. Make request IDs part of the volatile suffix, not the stable prefix, and use idempotency where the provider supports it. Locally, deduplicate in-flight writes by digest so ten simultaneous requests do not all prefill the same absent prefix.

```python
# nanoserve/singleflight.py
from concurrent.futures import Future
from threading import Lock


class SingleFlight:
    def __init__(self):
        self._lock = Lock()
        self._work: dict[bytes, Future] = {}

    def run(self, key: bytes, fn):
        with self._lock:
            future = self._work.get(key)
            if future is None:
                future = Future()
                self._work[key] = future
                owner = True
            else:
                owner = False
        if not owner:
            return future.result()
        try:
            value = fn()
            future.set_result(value)
            return value
        except BaseException as error:
            future.set_exception(error)
            raise
        finally:
            with self._lock:
                self._work.pop(key, None)
```

This prevents duplicate local work for the same digest while the first fill is in progress. It does not make a remote provider write idempotent; that remains an adapter concern.

## 10. Measuring the boundary honestly

The minimum dashboard has four rates:

| Metric | Definition | Why it matters | Source |
| --- | --- | --- | --- |
| Engine hit tokens | reused complete local prefix tokens / requested input tokens | local prefill work avoided | derived |
| Engine hit requests | requests with at least one reused block / requests | admission and routing health | derived |
| Provider cached tokens | provider-reported cached input tokens / provider input tokens | billing and remote reuse | cited: provider usage fields, docs accessed 2026-08-03 |
| Prefix divergence position | first differing block or token index | prompt-layout diagnosis | derived |
| Effective prefill saved | local hit tokens × measured prefill time per token under matching load | latency impact | reproduce: `bench_prompt_cache.py` |

Measure local hit latency with the same model, dtype, batch shape, and scheduler pressure as misses. A cache hit can reduce prefill but leave decode unchanged. A long cached prefix can still increase attention work during every decode step because the model attends over the full sequence. Report TTFT, TPOT, queue time, and provider usage separately.

```python
# bench_prompt_cache.py
import time
import torch


def timed(fn, warmup: int = 10, steps: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(steps):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / steps


def report(engine, uncached, cached):
    cold_ms = timed(lambda: engine.generate(uncached))
    warm_ms = timed(lambda: engine.generate(cached))
    print({"cold_ms": cold_ms, "warm_ms": warm_ms, "saved_ms": cold_ms - warm_ms})


if __name__ == "__main__":
    print("Run with a CUDA engine and two tokenized prompts; no expected value is fabricated here.")
```

The script synchronizes before timing and uses CUDA events, but it does not claim a number without hardware. A reader can run it on an RTX 4090, L4, A100, or H100 and report a range after fixing clocks and recording model, context, batch, block size, and output length. For a service, use open-loop load to test arrival-rate collapse and closed-loop load to test per-client throughput. A single batch-one timing cannot answer whether cache reuse improves goodput under concurrency.

### A useful synthetic prompt suite

Build four workloads from the series’ shared suite:

- chat: stable system instructions, changing short user turns;
- RAG: stable instructions and tool schema, changing retrieved documents;
- code completion: stable repository rules, changing file and cursor;
- translation: stable style guide, changing source text.

For each, generate a control where volatile content is intentionally moved before the breakpoint. Compare divergence position, engine hit tokens, provider cached tokens, TTFT, and input cost. The expected result is directional: putting changing material before the breakpoint lowers reuse; the exact hit percentage is reader-reproducible and provider-dependent, not a number to invent.

## 11. Routing, privacy, and the cache boundary in a fleet

One worker can have a perfect prefix hit rate while the fleet has a terrible one. If a gateway assigns requests randomly, a session’s second turn may land on a worker that has never seen its system prefix. The provider can have the opposite behavior: it may route identical prefixes to its own cache domain even when your gateway is unaware of that placement. Cache-aware routing is therefore a third identity problem, between prompt construction and block allocation.

For local KV, the useful routing key is not the entire user prompt. It is a stable prefix namespace plus a session or corpus identifier. Consistent hashing can pin a session to a worker, but it also creates hot spots. A large public system prompt may be broadly shareable; a private conversation should normally be pinned by tenant and conversation ID. If a worker is draining, the router must decide whether to move the request and accept a cold local cache or wait and risk queueing.

| Routing choice | Local cache effect | Failure mode | Source |
| --- | --- | --- | --- |
| Random replica | simplest; low expected hit rate for repeated sessions | every retry can be cold | derived |
| Session affinity | high hit probability for a conversation | hot session or worker failure | derived |
| Prefix digest affinity | shares public prefixes across sessions | privacy boundary must be explicit | derived |
| Provider-key affinity | groups requests for provider reuse when supported | does not guarantee local KV residency | derived |
| Two-level affinity | route by tenant/prefix, then choose least-loaded worker | more state and rebalancing | explanatory design |

The public vLLM router material cited in the series kit uses consistent hashing on a session or user key to maximize KV reuse. Its reported Llama 3.1 8B setup used eight prefill and eight decode pods and reported higher throughput than the comparison baselines; that is a cited system result, not a promise for a small local engine. The reusable lesson is architectural: the scheduler cannot recover a prefix that routing discarded.

### Tenant isolation is part of the key

Consider two users who submit the same public system instructions followed by different private documents. Sharing the public blocks is safe only if the boundary is before the private content. If a retrieval result is accidentally included in a block that the digest marks as public, a later user can attach to it. Namespace the cache by tenant unless a security review has proved that every block before the boundary is public.

```python
# nanoserve/tenant_key.py
import hashlib


def tenant_namespace(tenant_id: str, public_prefix: bool) -> bytes:
    if not tenant_id and not public_prefix:
        raise ValueError("private prefixes require a tenant ID")
    scope = "public" if public_prefix else f"tenant:{tenant_id}"
    return hashlib.sha256(scope.encode("utf-8")).digest()


def assert_cache_scope(*, tenant_id: str, cached_tenant: str, public: bool) -> None:
    if not public and tenant_id != cached_tenant:
        raise PermissionError("private KV block crossed a tenant boundary")
```

This is not a complete authorization system. It is a deliberately loud guardrail. Cache lookups should happen only after authentication and policy classification, not before. A digest is not an authorization token.

### The provider boundary changes the threat model

When using a hosted provider, the prompt and provider-managed cache cross your network boundary. A local engine can zero or overwrite a block on eviction; your application cannot assume that a provider’s cache has the same deletion timing. Read the provider’s retention and data-control documentation, choose the appropriate organization or project scope, and record whether extended caching is compatible with your data-retention requirements. OpenAI’s current data-controls documentation notes that extended prompt caching stores KV tensors in GPU-local storage and can affect Zero Data Retention eligibility. That is a policy fact to validate with the provider, not a reason to avoid all caching.

## 12. A test matrix that catches semantic drift

The cache needs tests at three levels: token identity, allocator correctness, and API observability. A unit test that checks only `digest_a == digest_b` will miss stale blocks, tenant leaks, provider usage changes, and routing-induced cold misses.

### Identity tests

For every namespace field, make one request pair that changes only that field. The expected result is a local miss. Make another pair that changes only suffix content after a complete block boundary. The expected result is a hit for the shared blocks and a miss for the suffix. Test Unicode NFC and NFD forms, JSON key order, whitespace, special tokens, and tokenizer revisions.

```python
# tests/test_prefix_semantics.py
from nanoserve.prefix_cache import CacheNamespace, prefix_digests


def test_suffix_change_keeps_complete_block():
    ns = CacheNamespace("model-a", "tok-a", "bf16", 4, "layout-a")
    first = prefix_digests([1, 2, 3, 4, 5, 6, 7, 8], ns)
    second = prefix_digests([1, 2, 3, 4, 5, 6, 7, 99], ns)
    assert first[0] == second[0]
    assert first[1] != second[1]


def test_dtype_is_not_interchangeable():
    bf16 = CacheNamespace("model-a", "tok-a", "bf16", 4, "layout-a")
    fp16 = CacheNamespace("model-a", "tok-a", "fp16", 4, "layout-a")
    assert prefix_digests([1, 2, 3, 4], bf16) != prefix_digests([1, 2, 3, 4], fp16)
```

These tests are reader-reproducible and expected to pass on any Python 3.11 environment. They test the key law, not the provider’s implementation.

### Allocator tests

Allocate a fixed number of blocks, admit two requests sharing a prefix, branch one request, evict the least valuable entry, and then verify that no live request points at a freed block. Run the test under concurrent lookups and fills. A race where a lookup returns a block just as eviction frees it is worse than a miss: it is memory corruption or wrong output.

The simplest safe protocol is pinning. A block is not evictable while a scheduler request holds a reference. Prefix entries add references when attached; finished requests release them. An entry that is eligible for eviction can still have a block reference from a live continuation, in which case the allocator skips it. Reference counts are a correctness mechanism; LRU is only a policy.

### Provider contract tests

Use a small fixture rather than live calls in unit tests. For each provider adapter, feed a recorded usage object with cached reads, cache writes, no cache fields, and an unknown schema version. Assert that unknown fields are retained in `raw` and that missing counters become zero only when the provider contract says zero is meaningful. Never silently classify an absent provider field as a hit.

At integration time, send two requests with a stable prefix and a changing suffix. Record the full request shape, model, provider, region, breakpoint, TTL, and timestamps. The expected assertion should be qualitative unless you control the provider: the second response should report a nonzero cached-token field when the documented minimum and eligibility conditions are satisfied. If it does not, store the request pair for diagnosis rather than automatically rewriting production prompts.

### Regression snapshots

Cache behavior changes when you change a chat template, tool schema, model alias, provider API version, or serialization library. Keep a redacted snapshot containing token count, first divergence position, namespace digest, and provider usage fields. A deployment can then answer whether the hit-rate drop came from content, identity, routing, or provider policy.

## 13. Prompt patterns that survive real applications

![A before-and-after prompt layout moves stable instructions before the breakpoint and volatile request data into the suffix](/imgs/blogs/prompt-caching-semantics-engine-side-and-provider-side-7.webp)

The stable-prefix rule needs to survive product requirements, not just toy strings. A useful pattern is a three-layer prompt builder:

```python
# nanoserve/prompt_layout.py
from dataclasses import dataclass
import json


@dataclass(frozen=True)
class PromptParts:
    system: str
    tools: list[dict]
    examples: list[dict]
    retrieved: list[str]
    user_text: str
    request_time: str


def build_messages(parts: PromptParts) -> list[dict]:
    stable = {
        "role": "system",
        "content": parts.system + "\nTOOLS=" + json.dumps(parts.tools, sort_keys=True),
    }
    examples = [{"role": item["role"], "content": item["content"]} for item in parts.examples]
    volatile_context = {
        "role": "user",
        "content": (
            "REFERENCE_DOCUMENTS\n"
            + "\n---\n".join(parts.retrieved)
            + "\nREQUEST_TIME=" + parts.request_time
            + "\nQUESTION=" + parts.user_text
        ),
    }
    return [stable, *examples, volatile_context]
```

This builder is only an arrangement example. The provider’s actual message schema and cache-breakpoint API still need an adapter. The design choice is the important part: examples and tool definitions remain before per-request content; retrieval results are allowed to vary without destroying the shared instruction prefix. If retrieval documents are a stable corpus snapshot, they can be promoted into a separately versioned cache region.

### Keep timestamps out of instructions

Current time is often needed for correct answers, but it is rarely an instruction. Put it in a request field or a suffix block. If a model must see it near the system policy, repeat a compact time value after the stable prefix rather than changing the policy text itself. The additional few tokens are usually cheaper than invalidating thousands of stable tokens. This is an engineering tradeoff to measure, not a provider guarantee.

### Keep request IDs out of semantics

Request IDs belong in tracing metadata. If the model does not need the ID, do not put it in the prompt. If it needs the ID for a tool call, put it in the volatile suffix. A UUID is high entropy and guarantees a different token sequence; it is an especially effective cache breaker with no model-quality benefit.

### Retrieval ordering must be deterministic when it is meant to cache

Stable retrieval is a data pipeline property. Sort by a versioned rank and document ID, canonicalize metadata, pin the embedding model, and include a corpus snapshot ID. If the top five chunks fluctuate because of a tie, the prompt prefix fluctuates. If you want a cache across queries, cache the corpus or system context, not accidental retrieval output. If the retrieved evidence is user-specific, preserve privacy and accept that the prefix is volatile.

## Case studies / real numbers

### 1. OpenAI’s breakpoint change

OpenAI’s current prompt-caching documentation describes a behavior change for GPT-5.6 and later families: exact prefixes are cached at cache breakpoints, with an implicit breakpoint at the latest user or tool message by default. If changing content lands at that breakpoint, shared earlier content may not be reused. The documented fix is to place an explicit breakpoint at the end of the stable prefix, use a stable `prompt_cache_key`, and optionally make caching explicit so a volatile suffix is not repeatedly written. This is a good example of why “the same 4,000 instructions are present” is weaker than “the same 4,000 instructions end at a cacheable breakpoint.” Source: [OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching), accessed 2026-08-03.

### 2. Anthropic’s 20-block lookback

Anthropic documents cache writes only at breakpoints and reads by walking backward through previously written positions, up to a 20-block lookback. A growing conversation can therefore hit an older breakpoint as long as the new request remains within that window. A breakpoint on a changing timestamp does not cause the service to discover and write the stable content behind it. This is a concrete failure mode for applications that mark the end of a whole request instead of the end of the reusable region. Source: [Anthropic prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching), accessed 2026-08-03.

### 3. PegaFlow’s workload dependence

The vLLM team’s [PegaFlow post](https://vllm.ai/blog/2026-05-18-pegaflow), dated 2026-05-18, reports workload-specific results for an external KV cache service: Qwen3-8B on one host had 52.35% hit rate in the cited comparison, while a DeepSeek-V3.2 MLA TP8 setup had 97.23%. The point is not that an external store delivers a fixed percentage. The point is that prompt recurrence, representation size, and workload shape determine whether remote KV storage helps. A provider prompt cache and a remote engine KV cache face the same economic question: how often does reusable context recur before its storage and movement cost dominate?

### 4. The “cache hit but no latency win” incident

This is a derived incident pattern rather than a public benchmark. Suppose a request reuses 8,192 local prefix tokens but then generates 256 output tokens. The engine avoids the initial prefill for 8,192 tokens, but every decode step still attends over the full 8,448-token sequence. If decode attention and network queueing dominate, TTFT improves while TPOT barely changes. If an operator watches only end-to-end latency, the cache may appear ineffective; if they watch only cached tokens, it may appear successful. The correct diagnosis is a phase split: prefill saved, decode unchanged, queue time separately accounted.

## When to reach for this, and when not to

Use engine-side prefix caching when the same worker sees repeated, long prefixes and the saved prefill work is material. Shared system prompts, repeated repository context, and agent tool definitions are strong candidates. Use a provider cache when the provider’s documented billing and TTL contract matches your reuse interval and you cannot or do not want to own GPU KV storage.

Do not cache private retrieval context across tenants by default. Do not add a cache layer to a 200-token prompt just because the API advertises caching; minimum thresholds and write overhead can make it irrelevant. Do not treat provider cached-token counters as a local latency metric. Do not use a long TTL to hide a prompt invalidation bug. Do not prewarm a cache without measuring the arrival pattern that will consume it.

If you need a production engine rather than a teaching artifact, use vLLM, SGLang, or a managed provider and adopt their documented prefix-cache behavior. `nanoserve` is valuable here because it makes the ownership boundary visible: tokenization, cumulative digests, block allocation, copy-on-write, and metrics are small enough to inspect. It is not a replacement for the admission, routing, failure recovery, and security work in a mature serving system.

## Key takeaways

- Engine prefix caching reuses local KV tensors; provider prompt caching reuses a provider-owned representation. They are separate counters.
- Hash token IDs and representation namespace, not the original prompt string, for local correctness.
- Put stable instructions, tools, and examples before the cache boundary; put time, request IDs, retrieval results, and user text after it.
- A cache write at a changing boundary does not automatically make earlier stable content reusable.
- TTL is a residency or eligibility policy, not a guarantee that a local block remains in VRAM or a provider keeps an entry forever.
- Cache writes, cache reads, storage, VRAM, and saved prefill all belong in the cost model.
- Copy-on-write protects shared prefixes when sampling branches or concurrent continuations append tokens.
- Record miss reasons and the first divergence position; a single hit-rate number cannot explain a miss.
- Treat tenant, model, tokenizer, tool schema, dtype, layout, and deployment revision as cache identity.
- Provider semantics are versioned facts. Store the provider, model, API surface, and documentation date with the metric.

## Further reading

- [OpenAI prompt caching guide](https://developers.openai.com/api/docs/guides/prompt-caching), accessed 2026-08-03.
- [Anthropic prompt caching guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching), accessed 2026-08-03.
- [Gemini context caching](https://ai.google.dev/gemini-api/docs/generate-content/caching), accessed 2026-08-03.
- [PegaFlow: External KV Cache Service](https://vllm.ai/blog/2026-05-18-pegaflow), 2026-05-18.
- [Prefix sharing, radix trees, and copy-on-write](/blog/machine-learning/inference-engineering/prefix-sharing-radix-trees-and-copy-on-write).
- [The cost model of inference dollars per million tokens](/blog/machine-learning/inference-engineering/the-cost-model-of-inference-dollars-per-million-tokens).
- [Observability for inference: goodput, not throughput](/blog/machine-learning/inference-engineering/observability-for-inference-goodput-not-throughput).
