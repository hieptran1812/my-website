---
title: "Case study: The prefix cache that leaked across tenants"
date: "2026-08-04"
publishDate: "2026-08-04"
description: "Build a tenant-safe prefix cache for nanoserve, reproduce its timing leak, and quantify the reuse and latency you give up for isolation."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "kv-cache",
    "prefix-caching",
    "multi-tenancy",
    "security",
    "latency",
    "pytorch",
    "vllm",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 28
---

In a synthetic reproduction of the incident, tenant B's time-to-first-token was 11 ms lower when its prompt began with a particular 128-token string. Treat that value as an illustrative, reader-reproducible scenario rather than a measurement from this environment. The request body was never logged. The model output was identical. No memory address, cache key, or physical GPU block crossed the API boundary.

The latency difference was still an answer to a private question: “Has another tenant recently sent this prefix?” A customer-support assistant, a retrieval chunk, or a system prompt can be sensitive even when the generated answer is not. Once we made prefix caching global, the GPU became a timing oracle.

![A shared prefix moves from request tokens through hashed blocks into a reusable physical KV pool while hit and miss paths expose different timing](/imgs/blogs/case-study-the-prefix-cache-that-leaked-across-tenants-1.webp)

The diagram above is the mental model: a prefix cache is not merely a dictionary from text to speed. It is a shared resource with a lookup result, a lifetime, an eviction policy, and an observable service time. In this case study we will build the bug in a small `nanoserve`-style cache, write the attack harness, add isolation keys and hash salting, and then measure the performance price honestly.

The goal is not to claim that one particular deployment had these exact timings. This is a controlled engineering incident: the numbers marked `derived` come from stated arithmetic, numbers marked `cited` come from public documentation or papers, and numbers marked `reproduce` are expected outputs from code you can run on a named setup. The scenario is synthetic; the failure mode is real.

## 1. The incident in one request path

Prefix caching removes repeated prefill work. During prefill, the model reads all prompt tokens and writes key/value tensors for every transformer layer. If later requests begin with the same token sequence, the engine can map their logical prefix blocks to the old physical blocks and start computation at the first uncached token.

In `nanoserve`, use a fixed block size of 16 tokens. That is also the block-size example used in the vLLM grounding material. A complete block is eligible for caching; a partial block is not. For a Llama-3.1-8B-shaped configuration with 32 layers, 8 key/value heads, head dimension 128, bf16 values, the KV storage per token is derived as

$$
2 \times 32 \times 8 \times 128 \times 2 = 131{,}072\text{ bytes} = 128\text{ KiB/token}.
$$

The first factor is key plus value, the last factor is two bytes per bf16 value. One full 16-token block is therefore $16 \times 128\text{ KiB} = 2\text{ MiB}$ of KV storage. That arithmetic is a capacity fact, not a benchmark result.

The production design described in [vLLM's automatic prefix-caching documentation](https://docs.vllm.ai/en/v0.14.1/design/prefix_caching/) hashes the tokens in a block together with the parent prefix hash. The same document explains that only full blocks are cached and that extra metadata, including cache salts, can be added to the identity. The [vLLM architecture post](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) describes the corresponding free-block pool and request-to-block mapping.

Our first implementation copied the useful data structure and missed the security boundary. It keyed the cache by model ID and a hash chain over token IDs:

```python
from dataclasses import dataclass
import hashlib

BLOCK_SIZE = 16

@dataclass(frozen=True)
class BlockKey:
    model_id: str
    parent_hash: bytes
    token_ids: tuple[int, ...]

def block_hash(parent: bytes, token_ids: tuple[int, ...]) -> bytes:
    payload = parent + b"|" + b" ".join(str(t).encode() for t in token_ids)
    return hashlib.sha256(payload).digest()

def prefix_keys(model_id: str, token_ids: list[int]) -> list[BlockKey]:
    parent = b"ROOT"
    keys = []
    for start in range(0, len(token_ids), BLOCK_SIZE):
        block = tuple(token_ids[start:start + BLOCK_SIZE])
        if len(block) < BLOCK_SIZE:
            break
        keys.append(BlockKey(model_id, parent, block))
        parent = block_hash(parent, block)
    return keys
```

This code is safe against ordinary accidental hash collisions far better than a process-randomized integer hash. It is not safe against an untrusted caller who shares the same namespace. Tenant A and tenant B produce the same `BlockKey` for the same model and token prefix. That is exactly what prefix reuse needs and exactly what the attacker needs.

### The first misleading fix

The on-call engineer initially proposed replacing SHA-256 with a faster hash. That addresses collision probability, not information flow. A cryptographically strong digest can make a key collision harder to cause while leaving a cache hit just as observable. The security question is not “Can two different prompts accidentally have the same digest?” It is “Which principal is allowed to reuse this digest's physical block?”

The second proposal was to hide cache metrics. That also misses the channel. A client does not need a `cache_hit=true` header if the hit avoids a large prefill kernel. The channel is the end-to-end request time, including queueing, tokenization, GPU synchronization, and output delivery.

## 2. The side channel

The attacker does not need to read a victim's KV tensor. They need a candidate prefix. Suppose a victim might have submitted one of several known document headers. The attacker sends a request beginning with candidate $c$, records time to first token, and repeats it under slightly different arrival times. If the candidate is already resident, the engine can skip the full-block prefill for that prefix. If it is absent, the engine must compute and allocate it.

![Tenant A fills a prefix block, tenant B guesses the same tokens, and the hit path skips prefill while the miss path computes it](/imgs/blogs/case-study-the-prefix-cache-that-leaked-across-tenants-2.webp)

The diagram captures membership, not plaintext exfiltration. A hit says that some request in the shared trust domain caused the block to be present recently. It does not by itself prove which user sent it. That is still a privacy violation when the candidate set is small or the victim is the only plausible source.

### Why one timing sample is not enough

A server is noisy. The request can wait behind another decode step. The tokenizer can be scheduled on a different CPU core. CUDA launches can overlap. The cache can be evicted between the attacker's warm-up and probe. A single 11 ms observation is weak evidence.

Repeated probes turn the question into a statistical test. Let $T$ be TTFT and let $H_1$ mean “candidate prefix is cached.” The attacker estimates the difference

$$
\Delta = \operatorname{median}(T \mid H_0) - \operatorname{median}(T \mid H_1).
$$

If $\Delta$ is larger than the noise floor, membership becomes distinguishable. This is an explanatory statistical abstraction, not a formula claimed by a serving paper. It is useful because it tells us what defenses must do: eliminate the cross-tenant hit, or make hit and miss service times indistinguishable enough that the residual signal is not useful.

The simplest reliable defense is to make $H_1$ false across tenants. Tenant B's candidate may be present in the GPU pool, but B's lookup must not be allowed to see it. The cache should behave as if the block does not exist in B's namespace.

### The attacker's candidate set

The attack is strongest when the attacker can guess structured prefixes. Examples include a shared system prompt with one secret clause, a known legal document with a private appendix, a template containing an account identifier, or a retrieval corpus whose public metadata narrows the possibilities. The attacker does not need arbitrary prompt reconstruction. A one-bit membership oracle can confirm a candidate that was already suspected.

This is why “we do not expose prompts in logs” is insufficient. The prompt can remain encrypted in application logs while its derived execution path remains observable. Cache sharing changes compute, and compute changes time.

## 3. The cache key is a security boundary

The fix starts before hashing. We need a precise answer to “same for whom?” A cache key that contains only token IDs is a content key. A tenant-safe key is a content key inside a security namespace.

![Request metadata branches into canonicalization, namespace construction, and salted block hashing before a cache lookup can reuse a physical block](/imgs/blogs/case-study-the-prefix-cache-that-leaked-across-tenants-3.webp)

At minimum, the namespace should bind the properties that affect the meaning and authorization of a KV block:

| Key component | Why it belongs in identity | Failure if omitted | Source |
| --- | --- | --- | --- |
| Tenant or trust-group ID | Defines who may observe reuse | Cross-tenant timing signal | `derived` from threat model |
| Model revision | Weights change hidden states | Wrong-model KV reuse | `derived` from cache correctness |
| Tokenizer and chat-template revision | Same text can tokenize differently | Token-position mismatch | `derived`; [tokenizer boundary](/blog/machine-learning/inference-engineering/the-tokenizer-boundary-and-incremental-detokenization) |
| Adapter or LoRA ID | Adapter changes activations | Cross-adapter contamination | `derived`; vLLM cache metadata docs |
| Policy namespace | Public, group, and private data have different owners | Public/private boundary collapse | `derived` from authorization policy |
| Salt or barrier ID | Prevents equality across namespaces | Timing oracle remains | [vLLM prefix-cache docs](https://docs.vllm.ai/en/v0.14.1/design/prefix_caching/) |

Do not confuse an authentication identity with a raw user-provided string. The request should carry an authenticated principal from the gateway. The engine should derive a short-lived cache namespace from policy, not trust an arbitrary `tenant_id` field sent by the client.

### Canonicalization is part of security

The key must be computed from the exact token IDs that enter the model, after the chat template and special-token policy are applied. “Same text” is not a sufficient definition. Whitespace, Unicode normalization, role markers, image hashes, adapter selection, and tokenizer version can all change the model input.

The cache key should be versioned. A practical namespace string is a length-delimited encoding of fields, followed by a digest:

```python
import base64
import hashlib

def _field(value: str) -> bytes:
    raw = value.encode("utf-8")
    return len(raw).to_bytes(4, "big") + raw

def namespace_key(*, tenant_scope: str, model_revision: str,
                  tokenizer_revision: str, adapter_id: str,
                  policy_revision: str, salt: bytes) -> bytes:
    fields = (
        _field(tenant_scope),
        _field(model_revision),
        _field(tokenizer_revision),
        _field(adapter_id),
        _field(policy_revision),
    )
    return hashlib.sha256(b"nanoserve/ns/v1" + b"".join(fields) + salt).digest()

def public_namespace(model_revision: str, tokenizer_revision: str) -> bytes:
    return namespace_key(
        tenant_scope="public",
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
        adapter_id="base",
        policy_revision="public-v1",
        salt=b"public-cache-v1",
    )
```

The `public` namespace is not automatically safe. It is safe only for prefixes that the product owner has explicitly classified as public. A common system prompt may be shared within an organization while a customer-uploaded retrieval document must stay private. The right policy is usually more granular than “prefix caching on” or “prefix caching off.”

## 4. Hash salting and barrier placement

Hash salting gives us isolation without changing model inputs. The salt is metadata used in the cache identity, never a token passed through the transformer. For a chain of full blocks $B_0, B_1, \ldots$, use

$$
h_0 = H(s \mathbin\| B_0), \qquad h_i = H(h_{i-1} \mathbin\| B_i).
$$

This is the normal parent-chained structure with a salt inserted at the first barrier. If tenant A and tenant B use different salts, their first hashes differ. Because every later block includes its parent's hash, all descendants differ too. A later block cannot accidentally rejoin the unsalted chain unless the implementation independently drops the parent hash, which would be a correctness bug.

![The same token blocks produce one unsalted chain, while tenant-specific salts create separate descendant chains that still share within each tenant](/imgs/blogs/case-study-the-prefix-cache-that-leaked-across-tenants-4.webp)

### Single barrier versus multi-barrier

A single barrier at the first block is easy to reason about. One salt partitions the entire request prefix. If the organization shares a salt, its users can reuse blocks; if each tenant receives a different salt, they cannot. The [vLLM cache-salting RFC](https://github.com/vllm-project/vllm/issues/16016) describes this model and also discusses a multi-barrier design for finer-grained sharing.

The choice is a product policy, not a hash implementation detail:

| Barrier design | Reuse boundary | Main benefit | Main cost | Recommended use |
| --- | --- | --- | --- | --- |
| Global unsalted | Every caller | Maximum hit rate | Cross-tenant timing leakage | Public-only service |
| One salt per organization | Organization | Shared system prompts remain hot | Members can test one another's private prefixes | Strong internal trust |
| One salt per tenant | Tenant | Prevents tenant membership probes | Recomputes overlap across tenants | Default for private data |
| Salt per request | No reuse | Strongest separation | Prefix caching is effectively disabled | High-risk or regulated prompts |
| Multi-barrier | Public then tenant-private | Share approved prefix, isolate suffix | More metadata and policy complexity | Mixed public/private templates |

### Salt rotation and cache lifetime

The salt must have an ownership and rotation story. If a tenant is deleted, its namespace must become unreachable immediately, and its blocks should be evicted or marked inaccessible. If the salt is rotated but old blocks remain addressable under the old namespace, rotation did not provide deletion.

A TTL is not a replacement for an isolation key. TTL limits how long an observation remains useful; it does not stop a tenant from observing a hit during that interval. TTL is still useful for reducing the window of exposure and controlling memory, but authorization must happen on every lookup.

### Why SHA-256 is not the isolation mechanism

The vLLM documentation says SHA-256 is used to reduce collision risk in multi-tenant setups and reports a hashing overhead of about 75 ns per token, or less than 4 ms for 50,000 context tokens, in its documented setup. That is a cited implementation number, not a promise for `nanoserve`. It is also orthogonal to tenant isolation. Use a collision-resistant hash for correctness and a namespace/salt for authorization.

## 5. Reproducing the leak safely

We do not need a real customer prompt for a useful test. Create two synthetic tenants and a candidate prefix. Tenant A warms the candidate. Tenant B probes it. Then repeat with a different prefix that A never sends. The experiment should report distributions, not a single impressive minimum.

![A randomized timing experiment warms a candidate, interleaves cold and warm probes, synchronizes GPU work, and ends with a membership decision](/imgs/blogs/case-study-the-prefix-cache-that-leaked-across-tenants-5.webp)

The following harness is deliberately backend-agnostic. It assumes an async client with a `generate` method that returns a first-token timestamp. Replace the client adapter with your OpenAI-compatible endpoint or a direct `nanoserve` call.

<figure class="blog-anim">
<svg viewBox="0 0 760 250" role="img" aria-label="An attacker probe alternates between a cold miss and a warm hit, then crosses a timing threshold" style="width:100%;height:auto;max-width:860px">
<style>
.pc1-box{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:2}.pc1-t{font:600 18px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}.pc1-s{font:14px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}.pc1-line{stroke:var(--border,#d1d5db);stroke-width:3}.pc1-dot{fill:var(--accent,#6366f1)}.pc1-th{stroke:var(--accent,#6366f1);stroke-width:3;stroke-dasharray:8 7}.pc1-ok{fill:var(--accent,#6366f1);opacity:.18}.pc1-bad{fill:var(--text-secondary,#6b7280);opacity:.18}
@keyframes pc1-probe{0%,12%{transform:translate(0,0)}42%,56%{transform:translate(210px,0)}88%,100%{transform:translate(420px,0)}}
@keyframes pc1-warm{0%,12%{opacity:0}30%,100%{opacity:1}}
.pc1-m{animation:pc1-probe 8s ease-in-out infinite}.pc1-w{animation:pc1-warm 8s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.pc1-m{animation:none;transform:translate(420px,0)}.pc1-w{animation:none;opacity:1}}
</style>
<rect class="pc1-box" x="30" y="55" width="170" height="90" rx="12"/><text class="pc1-t" x="115" y="92">cold miss</text><text class="pc1-s" x="115" y="120">prefill 16 tokens</text>
<rect class="pc1-box" x="295" y="55" width="170" height="90" rx="12"/><text class="pc1-t" x="380" y="92">warm hit</text><text class="pc1-s" x="380" y="120">reuse KV block</text>
<rect class="pc1-box" x="560" y="55" width="170" height="90" rx="12"/><text class="pc1-t" x="645" y="92">decision</text><text class="pc1-s" x="645" y="120">membership signal</text>
<line class="pc1-line" x1="200" y1="100" x2="295" y2="100"/><line class="pc1-line" x1="465" y1="100" x2="560" y2="100"/>
<line class="pc1-th" x1="590" y1="190" x2="705" y2="190"/><text class="pc1-s" x="648" y="218">timing threshold</text>
<rect class="pc1-bad" x="55" y="165" width="120" height="25" rx="6"/><rect class="pc1-ok" x="320" y="165" width="120" height="25" rx="6"/>
<circle class="pc1-dot pc1-m" cx="115" cy="177" r="10"/><circle class="pc1-dot pc1-m" cx="115" cy="177" r="10"/><circle class="pc1-dot pc1-m" cx="115" cy="177" r="10"/>
<text class="pc1-s" x="115" y="155">probe stream</text>
</svg>
<figcaption>The moving probe visits cold and warm paths; the attacker learns only when the timing shift crosses a decision threshold.</figcaption>
</figure>

The animation is intentionally placed beside the reproducibility harness: motion carries the transition from a cold miss to a warm hit, while the static timeline above carries the measurement protocol. With reduced motion enabled, it freezes on the decision state rather than disappearing.

```python
from __future__ import annotations
from dataclasses import dataclass
import random
import statistics
import time

@dataclass
class Sample:
    label: str
    ttft_ms: float

def clock_ms() -> float:
    return time.perf_counter_ns() / 1_000_000

def probe(client, tenant: str, token_ids: list[int], label: str) -> Sample:
    started = clock_ms()
    # The adapter must await the first streamed token, not the full response.
    client.generate(tenant=tenant, token_ids=token_ids, max_new_tokens=1)
    return Sample(label, clock_ms() - started)

def run_membership_test(client, warm_prefix, cold_prefix,
                        rounds: int = 40) -> list[Sample]:
    client.generate(tenant="victim", token_ids=warm_prefix, max_new_tokens=1)
    labels = ["candidate"] * rounds + ["control"] * rounds
    random.Random(7).shuffle(labels)
    samples = []
    for label in labels:
        token_ids = warm_prefix if label == "candidate" else cold_prefix
        samples.append(probe(client, "attacker", token_ids, label))
    return samples

def summarize(samples: list[Sample]) -> None:
    for label in ("candidate", "control"):
        values = [s.ttft_ms for s in samples if s.label == label]
        print(label, len(values), round(statistics.median(values), 2),
              round(statistics.mean(values), 2))
    candidate = [s.ttft_ms for s in samples if s.label == "candidate"]
    control = [s.ttft_ms for s in samples if s.label == "control"]
    print("median gap ms", round(statistics.median(control) -
                                  statistics.median(candidate), 2))
```

For a CUDA implementation, put `torch.cuda.synchronize()` immediately before and after the timed operation, or use CUDA events around the specific prefill work. For an HTTP endpoint, measure TTFT from request submission to the first streamed token and separately record queue time. The [inference benchmark protocol](/blog/machine-learning/inference-engineering/an-experiment-protocol-for-inference-benchmarks) has the series' warm-up, open-loop, percentile, and synchronization conventions.

Do not run this against another customer's data. The expected result is qualitative: in the intentionally vulnerable build, the candidate distribution should shift lower than the control distribution when the warm block survives. In the salted build, candidate and control should overlap within measurement noise because the attacker's namespace cannot reuse the victim's block. The exact milliseconds depend on model, GPU, scheduler load, block size, and transport.

### Avoiding false positives

Randomize probe order. If all candidate probes run first, the cache and GPU are in a different state than during controls. Keep output length at one token so decode time does not dominate. Run a warm-up before collecting samples. Record queue time and active batch size. Repeat the test at low and moderate load; a defense that appears effective only because the queue is noisy is not a defense.

The right acceptance test is not “the average got higher.” It is “cross-tenant probes do not produce a useful classifier.” A CI test can use a fixed synthetic setup and fail if a two-sample effect size exceeds a chosen threshold. A production security review should additionally test traffic shaping, eviction churn, retries, and concurrent victim requests.

## 6. What safety costs

The price of isolation is lost reuse. If 40% of prompts begin with a prefix that is shared only across tenants, global caching can skip that prefill while per-tenant caching cannot. The engine still has the same physical capacity; it simply cannot map one tenant's logical block to another tenant's physical block.

![Global, organization, tenant, and request namespaces trade prefix reuse and memory efficiency for progressively stronger cross-tenant isolation](/imgs/blogs/case-study-the-prefix-cache-that-leaked-across-tenants-6.webp)

Let $p$ be the fraction of prompt tokens that would have hit only because another tenant filled the block. Let $C_p$ be the prefill compute avoided per token and $C_d$ be the decode compute per output token. A rough derived prefill-work increase from disabling that reuse is $p$ times the eligible prefix tokens. It is not a universal latency increase because batching can hide some compute and the GPU may be decode-bound.

For a concrete derived example, suppose 100 requests each contain a 1,024-token shared prefix, the block size is 16, and the prefix is shared uniformly across four tenants. Global sharing computes that prefix once, then reuses 64 full blocks for the remaining 99 requests. Tenant-scoped sharing computes it once per tenant, for four copies. The extra prefill is 3,072 tokens, or 192 blocks. With 2 MiB per 16-token block in the Llama-shaped bf16 example, those four copies occupy 8 MiB while the one global copy occupies 2 MiB. This is a capacity and work derivation; it is not a measured throughput claim.

The extra physical memory may be temporary. An LRU cache can evict older tenant blocks, and a shared pool can still reuse free physical slots after a block becomes unreachable. What changes is resident reuse, not the allocator's ability to recycle bytes. Under high tenant cardinality, however, the same public prefix can be replicated many times until eviction.

### Worked example: reuse versus isolation

Assume 1,000 requests per minute, 512 eligible prefix tokens per request, and a 30% cross-tenant-only overlap. The avoided work under global sharing is

$$
1{,}000 \times 512 \times 0.30 = 153{,}600\text{ token-prefill equivalents per minute}.
$$

If each block is 16 tokens, that is 9,600 block-prefill equivalents per minute. Under tenant isolation, those equivalents become recomputation unless the same tenant repeats the prefix. The result says nothing about milliseconds until we know the model, GPU, batch mix, and scheduler. That is why the benchmark must report cache scope and hit rate alongside TTFT.

### The latency cost is workload-dependent

Prefix caching primarily helps prompt processing. It does not make the model decode the next token faster once the request reaches the same decode state. A chat workload with short prompts and long answers may see little end-to-end benefit even with a high prefix hit rate. A RAG workload with long repeated headers and short answers can be prefill-dominated and benefit substantially.

The [prefix-caching and RadixAttention post](/blog/machine-learning/model-serving/prefix-caching-and-radixattention) is useful background on the optimization. The safety decision must add a fourth dimension to the usual hit-rate table: who is allowed to share the hit.

| Workload | Global cache benefit | Tenant cache benefit | Likely safety choice |
| --- | --- | --- | --- |
| Public documentation bot | High if documents are public | Similar | Public namespace with explicit allowlist |
| Enterprise RAG | High across one organization | High within organization | Organization salt plus private tenant barrier |
| Consumer chat | Unpredictable overlap | Usually lower | Tenant salt by default |
| Code assistant with private repositories | Potentially high but sensitive | Useful within one tenant | Per-tenant salt; no global sharing |
| Regulated records | Reuse may be valuable | Reuse is secondary to boundary | Per-request or tightly scoped group salt |

### Hashing overhead is the easy cost

Adding a salt to SHA-256 changes a few bytes in the hash input. The digest cost is generally small compared with a transformer prefill, but the exact ratio is hardware- and implementation-dependent. The cited vLLM documentation reports less than 4 ms for hashing 50,000 context tokens in its setup. That is not the dominant safety cost here.

The material costs are lost cross-tenant hits, replicated resident blocks, lower effective cache capacity, more prefill work, and more policy metadata on the scheduling path. If a team says “salting is free,” ask whether it means hash CPU time or lost reuse. Those are different budgets.

## 7. Implementing the `nanoserve` fix

The safe implementation has three properties. First, the authenticated request is converted into a policy scope before lookup. Second, the scope is included in the first block hash and therefore in every descendant. Third, metrics distinguish local hits, approved group hits, and denied cross-scope candidates without exposing the raw hash or prompt.

![The hardened request path authenticates a principal, selects a policy namespace, derives a salted cache identity, and records an auditable hit or miss](/imgs/blogs/case-study-the-prefix-cache-that-leaked-across-tenants-7.webp)

Here is a minimal policy object. In a real service, the gateway would pass a signed identity context or an internal authorization token; this example keeps the boundary visible.

```python
from dataclasses import dataclass
from enum import Enum
import secrets

class CacheScope(str, Enum):
    PUBLIC = "public"
    ORGANIZATION = "organization"
    TENANT = "tenant"
    REQUEST = "request"

@dataclass(frozen=True)
class AuthContext:
    tenant_id: str
    organization_id: str
    data_class: str
    model_revision: str
    tokenizer_revision: str
    adapter_id: str = "base"

@dataclass(frozen=True)
class CachePolicy:
    scope: CacheScope
    salt: bytes
    policy_revision: str

def policy_for(auth: AuthContext) -> CachePolicy:
    if auth.data_class == "public":
        return CachePolicy(CacheScope.PUBLIC, b"public-v1", "policy-v1")
    if auth.data_class == "organization":
        salt = hashlib.sha256(
            b"org:" + auth.organization_id.encode()).digest()
        return CachePolicy(CacheScope.ORGANIZATION, salt, "policy-v1")
    salt = hashlib.sha256(
        b"tenant:" + auth.tenant_id.encode()).digest()
    return CachePolicy(CacheScope.TENANT, salt, "policy-v1")

def request_policy(auth: AuthContext) -> CachePolicy:
    # Call this for records that must not be reusable after the request.
    return CachePolicy(CacheScope.REQUEST, secrets.token_bytes(32), "policy-v1")
```

The policy must be deterministic for scopes that should share and random for request-private work. The `REQUEST` salt is generated server-side, not accepted from the caller. A client-provided salt can be useful as an explicit cooperative cache group, but it should be authenticated and authorized before it can join another group.

The cache key then combines the policy namespace with the parent chain:

```python
def salted_block_hash(namespace: bytes, parent: bytes,
                      token_ids: tuple[int, ...]) -> bytes:
    token_bytes = b"".join(t.to_bytes(4, "big", signed=False)
                            for t in token_ids)
    return hashlib.sha256(
        b"nanoserve/block/v1" + namespace + parent + token_bytes
    ).digest()

def safe_prefix_keys(auth: AuthContext, token_ids: list[int],
                     policy: CachePolicy) -> list[bytes]:
    namespace = namespace_key(
        tenant_scope=policy.scope.value,
        model_revision=auth.model_revision,
        tokenizer_revision=auth.tokenizer_revision,
        adapter_id=auth.adapter_id,
        policy_revision=policy.policy_revision,
        salt=policy.salt,
    )
    parent = b"ROOT"
    result = []
    for offset in range(0, len(token_ids), BLOCK_SIZE):
        block = tuple(token_ids[offset:offset + BLOCK_SIZE])
        if len(block) != BLOCK_SIZE:
            break
        current = salted_block_hash(namespace, parent, block)
        result.append(current)
        parent = current
    return result
```

Notice what this function does not do: it does not put a tenant ID into model input, it does not log the candidate text, and it does not return a boolean that the client can use as a cache oracle. It derives the identity inside the trusted engine and uses it for authorization and lookup.

### Lookup and reference counting

The physical KV pool still needs reference counting. A shared block can be mapped by several requests inside one namespace. When a request finishes, decrement its references; when the count reaches zero, retain the block only if the cache policy permits it. The allocator from [paged KV cache blocks](/blog/machine-learning/inference-engineering/paged-kv-cache-implementing-blocks-and-a-block-table) can remain unchanged below this layer.

```python
from collections import defaultdict

class ScopedBlockPool:
    def __init__(self):
        self.by_key = {}
        self.refcount = defaultdict(int)
        self.next_physical = 0

    def lookup_or_allocate(self, key: bytes, scope: CacheScope):
        entry = self.by_key.get(key)
        if entry is not None:
            self.refcount[entry] += 1
            return entry, True
        physical = self.next_physical
        self.next_physical += 1
        self.by_key[key] = physical
        self.refcount[physical] = 1
        return physical, False

    def release(self, key: bytes, keep_cached: bool = True):
        physical = self.by_key[key]
        self.refcount[physical] -= 1
        if self.refcount[physical] == 0 and not keep_cached:
            del self.by_key[key]
            del self.refcount[physical]

    def clear_scope(self, namespace_prefix: bytes):
        # Production code stores namespace metadata beside each entry.
        doomed = [key for key in self.by_key if key.startswith(namespace_prefix)]
        for key in doomed:
            del self.by_key[key]
```

The illustrative `clear_scope` method is intentionally incomplete: a cryptographic digest does not expose a namespace prefix. Production metadata should store an internal scope ID separately and use it for revocation. Never attempt to delete a tenant by guessing which digest bytes came from its salt.

### Metrics without a new oracle

Operators need to know whether salting is hurting capacity. Export aggregate counters by policy class, not per-request hit decisions to the untrusted caller. Useful internal metrics include `prefix_lookup_total`, `prefix_hit_total`, `prefix_recompute_tokens_total`, `prefix_denied_scope_total`, and resident blocks by scope. Protect dashboards and sample labels so an attacker cannot query the same oracle through a metrics endpoint.

```python
class CacheMetrics:
    def __init__(self):
        self.counters = defaultdict(int)

    def lookup(self, scope: CacheScope, hit: bool, tokens: int):
        self.counters["prefix_lookup_total"] += 1
        self.counters[f"prefix_lookup_scope_{scope.value}"] += 1
        self.counters["prefix_hit_total" if hit else
                       "prefix_miss_total"] += 1
        if not hit:
            self.counters["prefix_recompute_tokens_total"] += tokens

    def snapshot(self) -> dict[str, int]:
        return dict(self.counters)
```

## 8. Case studies and public grounding

This incident is a composite, but it sits on top of well-established mechanisms.

1. **PagedAttention and shared KV blocks.** The [PagedAttention paper](https://arxiv.org/abs/2309.06180) frames KV memory as a paged allocation problem: logical sequences map through block tables to physical blocks, reducing fragmentation and enabling sharing. That is the performance foundation. The security lesson is that a physical page can be shared only within an authorized namespace.

2. **Automatic prefix caching.** The [vLLM documentation](https://docs.vllm.ai/en/v0.14.1/design/prefix_caching/) states the parent-hash construction, full-block rule, SHA-256 option, and cache-salt mechanism. It explicitly describes salting as a way to prevent timing-based attacks while retaining reuse for callers with the same salt. That is the direct grounding for our barrier design.

3. **A public vLLM salting discussion.** The [cache-salting RFC](https://github.com/vllm-project/vllm/issues/16016) motivates salting because an attacker can guess popular inputs and compare latency. It proposes 256-bit salts and discusses per-organization, per-user, and multi-barrier choices. We use it as a design reference, not as evidence that the synthetic incident's timings occurred in vLLM.

4. **LLM prefix-cache side-channel research.** [The Early Bird Catches the Leak](https://arxiv.org/abs/2409.20002) studies timing side channels in LLM serving systems and connects shared KV or semantic caches to inference about sensitive inputs. [CacheSolidarity](https://arxiv.org/abs/2603.10726) studies the same problem and reports that selective isolation can retain more reuse than isolating every user. Those results motivate measuring the reuse/security frontier instead of assuming that “disable everything” is the only option; their experimental numbers do not transfer to our hardware.

5. **General cloud side channels.** The [cross-VM attack work](https://www.usenix.org/system/files/conference/usenixsecurity09/ristenpart.pdf) established the broader multi-tenant pattern: a shared machine can reveal co-residency or activity through timing even when tenants cannot read one another's memory. Prefix caching is a specialized version where the shared state is intentionally created by the serving engine.

The common thread is uncomfortable but useful: a cache is a security boundary whenever its state affects an observable outcome. The fact that the underlying bytes are protected by process isolation does not erase timing, eviction, queue, or power channels.

## 9. Testing and hardening the engine

The fix is incomplete until it has adversarial tests. Add a unit test for key separation, an integration test for physical-block non-reuse, a timing regression test, and a revocation test.

```python
def test_tenant_salts_separate_identical_prefixes():
    a = AuthContext("tenant-a", "org-1", "private", "llama-8b-r1", "tok-r1")
    b = AuthContext("tenant-b", "org-1", "private", "llama-8b-r1", "tok-r1")
    tokens = list(range(32))
    ka = safe_prefix_keys(a, tokens, policy_for(a))
    kb = safe_prefix_keys(b, tokens, policy_for(b))
    assert ka and kb
    assert set(ka).isdisjoint(kb)

def test_same_tenant_can_reuse():
    a1 = AuthContext("tenant-a", "org-1", "private", "llama-8b-r1", "tok-r1")
    a2 = AuthContext("tenant-a", "org-1", "private", "llama-8b-r1", "tok-r1")
    tokens = list(range(32))
    assert safe_prefix_keys(a1, tokens, policy_for(a1)) == \
           safe_prefix_keys(a2, tokens, policy_for(a2))
```

Then fuzz the metadata combinations. Change the tokenizer revision, model revision, adapter, policy revision, and salt one at a time. Every change that affects hidden states or authorization should change the namespace. Randomize token IDs and verify that identical input under identical authorized metadata remains stable.

### Worked example: the capacity budget after a rollout

Suppose an RTX 4090 deployment has 8 GiB reserved for KV blocks. Using the derived 2 MiB per full block from the Llama-shaped example, the theoretical upper bound is $8{,}192 / 2 = 4{,}096$ blocks, before allocator metadata, fragmentation, activations, and safety headroom. That corresponds to $4{,}096 \times 16 = 65{,}536$ token slots if every slot is usable by one logical prefix.

Now suppose the cache contains four tenant namespaces and a common 512-token public system prompt. If that prompt is approved for the public namespace, its 32 blocks can have one physical copy. If it is treated as tenant-private, each tenant can retain a copy: 128 blocks. The difference is 96 blocks, or $96 \times 2 = 192$ MiB of derived resident KV capacity. That number is not a prediction of the deployment's hit rate; it is the cost of four copies of one exact prefix under the stated block and dtype assumptions.

At high concurrency, this difference can change admission decisions. The scheduler may reject a long prompt earlier, evict a useful block, or preempt a request and recompute its prefix. This is why the isolation rollout belongs in the same capacity review as [admission control and backpressure](/blog/machine-learning/inference-engineering/admission-control-backpressure-and-latency-collapse) and [eviction and KV swapping](/blog/machine-learning/inference-engineering/eviction-preemption-and-kv-swapping). Security changes the scheduler's feasible state space.

The safe rollout is staged. First, add the namespace to internal metrics while keeping the old global lookup disabled for no one; compare expected hit scope with application ownership. Second, shadow-compute salted keys and count how many global hits would become same-tenant hits. Third, enable organization salts for a small set of explicitly trusted workloads. Fourth, make tenant salts the default for private data and set a rollback that changes policy, not code paths or physical memory ownership. Finally, expire old global entries so the previous namespace cannot be observed after the cutover.

Do not roll back by silently returning to global sharing after detecting a capacity regression. A rollback should select a documented namespace policy, page the owner, and preserve the security invariant. If capacity is insufficient, lower prefix-cache residency, reduce maximum context, add a GPU, or accept more recomputation. Security is not an emergency feature flag that should disappear under load.

### What to log during a security review

Keep a short-lived, access-controlled trace with request ID, policy scope class, cache-key version, number of full blocks hit, number of tokens recomputed, and scheduler queue time. Do not log raw token IDs or digests by default. A digest is not plaintext, but it can still become a stable identifier for a sensitive prompt.

The most useful dashboard is a matrix: hit rate by authorized scope, recompute tokens by workload, resident blocks by scope, and TTFT percentiles split into queue and compute. The [observability for inference](/blog/machine-learning/inference-engineering/observability-for-inference-goodput-not-throughput) post explains why a single throughput number hides this kind of failure.

### Timing equalization is a secondary defense

One can pad hits, add jitter, or delay responses until a fixed schedule. These measures are hard to tune. Padding every hit to the worst miss wastes the same prefill savings that caching was meant to recover. Random jitter reduces signal per sample, but repeated probes average it away unless the noise is large enough to damage the service-level objective.

Use equalization for defense in depth, especially at a public API boundary, but do not use it as a substitute for namespace isolation. Authorization removes the high-confidence cross-tenant hit. Timing controls reduce residual information from other shared resources.

## 10. When to share and when to isolate

My default for a multi-tenant inference endpoint is conservative: public data may use a public namespace; organization-approved system prompts may use an organization salt; tenant-provided context uses a tenant salt; highly sensitive or deletion-sensitive requests use a request-private salt. The policy should be explicit in configuration and visible in deployment review.

Use global sharing only when all callers are in the same trust domain or when the prefix corpus is genuinely public. “Everyone receives the same model” is not a trust-domain definition. Two tenants can share weights and kernels while remaining unauthorized to learn whether the other submitted a prompt.

Use per-organization sharing when the organization owns the prompt corpus and members are allowed to infer membership. This can preserve large system-prompt wins without turning one customer into an oracle for another customer. If organization members are not mutually trusted, use per-tenant sharing instead.

Use per-request isolation when the data has a strict lifetime, when deletion must be immediate, or when a request mixes public instructions with private records and the implementation cannot place a reliable barrier between them. It is expensive, but its behavior is easy to explain and test.

Do not build a custom security-sensitive cache merely to recover a few percent of throughput. [vLLM's deep dive](/blog/machine-learning/inference-frameworks/vllm-deep-dive) and its current prefix-cache documentation are better starting points for a production engine. Build the `nanoserve` version to understand the boundary, to test a novel policy, or to integrate a workload where you can own the full threat model.

## Key takeaways

- A prefix-cache hit is observable through request timing even when KV bytes never leave GPU memory.
- Collision resistance protects correctness; tenant isolation protects authorization. They are separate properties.
- Compute the cache namespace from authenticated policy context, not from an untrusted tenant field.
- Bind model revision, tokenizer/template revision, adapter, and policy revision into cache identity.
- Salt the first block hash; parent chaining propagates the separation to every descendant block.
- Share a salt only across principals that are intentionally in the same trust group.
- Measure cross-tenant timing distributions with randomized probes, synchronization, and queue-time accounting.
- The material safety cost is lost cross-tenant reuse and replicated resident blocks, not the few bytes added to a hash input.
- Keep cache metrics internal and aggregate; a public hit/miss counter simply recreates the oracle.
- Treat TTL, jitter, and response padding as defense in depth, never as the primary isolation boundary.

## Further reading

- [Automatic Prefix Caching](https://docs.vllm.ai/en/v0.14.1/design/prefix_caching/) — block hashes, full-block reuse, SHA-256, and cache salts.
- [Cache Salting for Secure and Flexible Prefix Caching](https://github.com/vllm-project/vllm/issues/16016) — isolation scopes and barrier designs.
- [The Early Bird Catches the Leak](https://arxiv.org/abs/2409.20002) — timing side channels in LLM serving.
- [CacheSolidarity](https://arxiv.org/abs/2603.10726) — selective defenses for multi-tenant prefix caching.
- [PagedAttention](https://arxiv.org/abs/2309.06180) — the paged KV memory foundation.
- [Prompt caching semantics](/blog/machine-learning/inference-engineering/prompt-caching-semantics-engine-side-and-provider-side) — how engine-side reuse differs from provider-side caching.
- [The inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook) — the series capstone and build-versus-buy decision.
