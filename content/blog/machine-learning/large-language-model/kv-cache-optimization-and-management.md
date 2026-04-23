---
title: "Optimizing and Managing the KV Cache: The Complete Production Guide"
publishDate: "2026-04-18"
category: "machine-learning"
subcategory: "Large Language Model"
tags:
  [
    "llm",
    "kv-cache",
    "inference",
    "serving",
    "optimization",
    "vllm",
    "sglang",
    "tensorrt-llm",
    "paged-attention",
    "radix-attention",
    "deep-learning",
  ]
date: "2026-04-18"
author: "Hiep Tran"
featured: true
aiGenerated: true
excerpt: "The KV cache is the single largest memory consumer in LLM serving — and the single biggest lever for throughput and latency. This guide explains how to manage it (PagedAttention, RadixAttention, block managers) and how to optimize it (GQA, MLA, FP8 quantization, eviction, offloading, prefix sharing), with the exact detail you need for staff-level LLM infra interviews."
---

## Why KV Cache Management Matters

![KV cache management techniques: PagedAttention, prefix caching, MHA->GQA->MLA architecture, KV quantization + eviction](/imgs/blogs/kv-cache-optimization-and-management-diagram.png)

In a modern LLM serving system, three things consume GPU memory: **model weights**, **activations**, and the **KV cache**. Weights are fixed once the model is loaded. Activations are short-lived and small. The KV cache, however, is **dynamic, per-request, and grows linearly with sequence length** — and it is, by far, the hardest resource to manage.

Consider a single Llama-3-70B serving node with an 80 GB H100:

- Weights in FP16: ~140 GB → already exceeds one GPU, needs tensor parallelism
- On 2×H100 (160 GB total), after weights: ~20 GB left
- KV cache for a single 8K-token request: **~2.5 GB**
- Maximum concurrent 8K requests fitting in the leftover memory: **~8**

Eight concurrent requests on $240K of hardware. The math is brutal. Everything about modern LLM serving — PagedAttention, continuous batching, prefix caching, quantization, GQA, MLA — exists because KV cache memory is the binding constraint on throughput.

This guide walks through both halves of the problem:

1. **Management** — how frameworks like vLLM, SGLang, and TensorRT-LLM physically lay out, allocate, share, and reclaim KV cache memory.
2. **Optimization** — how to shrink the KV cache (per-token and per-sequence) and how to reuse it across requests so the same GPU serves far more users.

By the end, you should be able to answer any LLM infrastructure interview question about the KV cache, from first principles up to production trade-offs.

## Part 1 — The KV Cache Budget: Know Your Numbers

Before any optimization, internalize this formula. It is the most important number in LLM serving:

$$
\text{KV cache size (bytes)} = 2 \times L \times H_{kv} \times d_{head} \times T \times B \times \text{dtype\_bytes}
$$

Where:

- $2$ — one tensor for K, one for V
- $L$ — number of transformer layers
- $H_{kv}$ — number of **KV heads** (equal to attention heads in MHA, fewer in GQA/MQA)
- $d_{head}$ — per-head dimension
- $T$ — sequence length in tokens
- $B$ — batch size (concurrent sequences)
- $\text{dtype\_bytes}$ — 2 for FP16/BF16, 1 for FP8/INT8, 0.5 for INT4

### Worked Example: Llama-3-70B

$L=80$, $H_{kv}=8$ (GQA with 64 query heads → 8 KV groups), $d_{head}=128$, dtype=BF16 (2 bytes).

Per-token KV cache:

$$
2 \times 80 \times 8 \times 128 \times 2 = 327{,}680 \text{ bytes} \approx 320 \text{ KB/token}
$$

For a single 8192-token sequence: $320 \text{ KB} \times 8192 \approx 2.5 \text{ GB}$. For 100 concurrent 8K requests: **250 GB**. This is why 70B models need tensor parallelism across multiple GPUs and why every byte of KV cache matters.

### The Two-Phase Nature of Inference

The KV cache behaves completely differently in the two phases of LLM inference:

| Phase | What happens | KV cache behavior | Bottleneck |
|-------|-------------|-------------------|------------|
| **Prefill** | Process the entire prompt in parallel | **Written once**, in large chunks | Compute (matmul-bound) |
| **Decode** | Generate tokens one at a time | **Read for every new token**, grows by 1 entry per step | Memory bandwidth (KV-load-bound) |

In prefill, the GPU is doing massive matrix multiplications and the KV cache is a byproduct. In decode, the GPU is mostly **streaming the entire KV cache out of HBM** to compute attention for the one new query. Decode is memory-bandwidth-bound; this is the single most important fact for understanding KV cache optimization.

The arithmetic intensity of decode attention is approximately:

$$
I = \frac{\text{FLOPs}}{\text{Bytes loaded}} \approx \frac{2 \cdot T \cdot d}{2 \cdot T \cdot d \cdot \text{dtype\_bytes}} = \frac{1}{\text{dtype\_bytes}}
$$

On an H100 with 989 TFLOPS BF16 and 3.35 TB/s HBM, the roofline crossover is at arithmetic intensity ≈ 295. Decode attention with BF16 KV cache has $I \approx 0.5$ — three orders of magnitude below the roofline. **Every KV cache optimization is fundamentally about making decode less memory-bandwidth-bound.**

## Part 2 — The Management Problem: Memory Fragmentation

![Fragmentation: naive contiguous allocation reserves max length per request (60-80% waste) vs PagedAttention which uses 16-token blocks with per-request block tables (95%+ utilization)](/imgs/blogs/kv-cache-opt-01-fragmentation.png)

### The Naive Approach (What Frameworks Did Before 2023)

The naive approach pre-allocates a contiguous buffer of shape `[max_seq_len, num_layers, ...]` for each request. Two fatal problems:

**Problem 1: Internal fragmentation.** If `max_seq_len = 4096` but the sequence only uses 200 tokens, you waste 95% of the allocation. Serving frameworks had to pick `max_seq_len` conservatively, dramatically reducing batch size.

**Problem 2: External fragmentation.** As sequences finish and free memory, the GPU memory becomes a Swiss cheese of free and used regions. New long sequences can't fit even though there's enough total free memory.

Empirically, pre-2023 frameworks wasted **60–80% of KV cache memory** to fragmentation. This is the problem that PagedAttention (the vLLM paper) solved.

### PagedAttention: Virtual Memory for the KV Cache

[Kwon et al. 2023] observed that the OS already solved exactly this problem decades ago via **paged virtual memory**. PagedAttention ports that design directly onto the GPU:

1. Divide GPU KV cache memory into **fixed-size blocks** (typically 16 tokens per block).
2. Allocate blocks **on demand** as sequences grow.
3. Maintain a **block table** per sequence — a list of block IDs in logical order, analogous to a page table.
4. The attention kernel is rewritten to gather K and V from non-contiguous blocks via the block table.

```
Logical view (what the model sees):
  Seq A: [tok0, tok1, ..., tok47]

Physical view (GPU memory):
  Block table for A: [block_7, block_2, block_19]
  block_7 : [tok0..tok15]
  block_2 : [tok16..tok31]
  block_19: [tok32..tok47]
```

**Internal fragmentation** is bounded by one block per sequence (≤15 wasted slots). **External fragmentation** is zero — any free block can be used by any sequence.

In the vLLM paper, PagedAttention reduced KV cache waste from ~60–80% to under 4%, enabling 2–4× throughput improvement over HuggingFace Transformers and FasterTransformer at the time.

### Copy-on-Write for Beam Search and Parallel Sampling

When a parent sequence forks (beam search, parallel sampling with `n > 1`), PagedAttention shares blocks between children via **reference counting**. Only the last, partially-filled block is copied on write. This is exactly the `fork()` semantics from Unix — complete conceptual reuse of OS techniques.

For beam search with 4 beams on a 2K-token prompt, naive KV cache duplication costs 4× the prompt cache. PagedAttention costs ~1× the prompt cache plus one block per beam — a 4× memory saving on the shared prefix.

### The Block Manager in vLLM

vLLM's `BlockManager` is the concrete implementation of this design. Key operations:

- `allocate(seq)` — find enough free blocks to hold the prompt, populate the block table.
- `append_slot(seq)` — when decoding produces a new token, either use space in the last block or allocate a new one.
- `fork(parent_seq, child_seq)` — share blocks via ref counting.
- `free(seq)` — decrement ref counts; blocks with count=0 return to the free pool.
- `can_allocate(seq)` — admission control check.

When free blocks run out, the scheduler **preempts** sequences (more on this below).

## Part 3 — Continuous Batching and KV Cache Interaction

![Continuous batching: iteration-level scheduler mixes prefill and decode each GPU step, frees KV blocks as requests finish, keeps GPU saturated](/imgs/blogs/kv-cache-opt-02-continuous-batching.png)

Continuous batching is the second half of the vLLM throughput story. It's worth understanding separately because it drives how KV cache is allocated in practice.

### Static vs Continuous Batching

**Static batching** (old style): accumulate $B$ requests, pad them to the same length, run them together, return all when the slowest finishes. GPU sits idle while short requests wait for the longest one.

**Continuous batching** (vLLM, SGLang, TGI): after every decode step, add newly-arrived requests and remove finished ones from the batch. No padding. The batch composition changes every step.

The interaction with KV cache: continuous batching means **the set of active sequences changes every step**, which means the KV cache memory must be allocatable and freeable per-sequence with tiny overhead. Paged allocation makes this essentially free (O(1) per step).

### Preemption and Recomputation

When new requests arrive and free blocks run out, vLLM has two preemption strategies:

1. **Swap** — move victim sequences' KV cache to CPU memory, bring them back when blocks free up. Transfer cost: a few hundred MB over PCIe (~20 GB/s), usually 10–100 ms.
2. **Recompute** — drop the victim's KV cache entirely and recompute it (prefill) when it's scheduled again. Free up-front, but pays the full prefill cost on resume.

Default policy: recompute for short sequences (cheap to redo), swap for long ones (expensive to redo, but large to transfer). This policy matters for P99 latency under pressure; always monitor preemption rate in production.

### Chunked Prefill

A single long-prompt prefill (e.g. 10K tokens) monopolizes the GPU for seconds. During that time, all concurrent decoders are blocked. Their Time-Per-Output-Token (TPOT) spikes.

**Chunked prefill** splits the long prefill into $k$-token chunks (typical: 512 or 1024) and interleaves them with decode steps from other sequences. The total prefill wall time increases slightly (~5–10% overhead) but P99 TPOT drops dramatically.

```
Without chunked prefill:
  Time: |──── 10K prefill (2s) ────||dec|dec|dec|...
  Concurrent requests: stuck for 2s

With chunked prefill (chunk=1024):
  Time: |c1|dec|c2|dec|c3|dec|...|c10|dec|
  Concurrent requests: regular decode slots, ~10ms gaps
```

vLLM, SGLang, and TensorRT-LLM all support this. It is essentially mandatory for any production workload mixing long prompts with streaming chat.

## Part 4 — Prefix Caching: Reusing KV Across Requests

Two users send "You are a helpful assistant. What is the capital of France?" and "You are a helpful assistant. What is the capital of Germany?". The first 7 tokens are identical — and so are the first 7 K and V vectors at every layer. Re-running prefill is pure waste.

**Prefix caching** reuses the KV cache from previous requests when a new request shares a prefix. Two designs dominate: hash-based (vLLM) and radix-tree-based (SGLang).

### vLLM: Hash-Based Block Matching

Since KV cache is already stored in 16-token blocks, vLLM hashes each full block and keeps a table mapping `hash(block_tokens, parent_hash) → block_id`. When a new prompt arrives, hash each 16-token window; on hit, reuse the block; on first miss, fall through to prefill for the rest.

Trade-offs:

- Block-granularity only: a prefix of 23 tokens hits 16 tokens (block 1), misses the remaining 7 (block 2 is only partial, can't share).
- Requires exact match at block boundaries.
- Simple, fast, integrates cleanly with PagedAttention.
- Opt-in via `--enable-prefix-caching`.

### SGLang: RadixAttention

SGLang maintains a **radix tree** (a prefix tree of token sequences) over all cached KV blocks. When a request arrives:

1. Walk the tree from the root, matching tokens one at a time.
2. Stop at the deepest common prefix — everything above that point is cache-hit.
3. Fork a new branch for the request-specific suffix; prefill only that suffix.

```
Radix tree after serving 3 requests:

              [root]
                │
     ["You are a helpful"]
          /         \
 [" assistant."]   [" AI tutor."]
      /
["What is"]
   /     \
["France?"]  ["Germany?"]
```

Advantages over hash-based:

- **Arbitrary-depth matching** — not constrained to block boundaries.
- **Automatic** — every request's KV cache is inserted into the tree for free; subsequent requests can reuse any suffix without user hints.
- **Any-shape sharing** — branches at any depth, perfect for multi-turn chat where each turn extends the conversation.

Eviction is **LRU** over tree nodes: unused leaves are evicted first. Critical detail: eviction must respect active sequences — you can't evict a block still held by a live request.

### When Prefix Caching Matters Most

- **Chatbots with system prompts**: the system prompt is 500–2000 tokens and identical across users. Prefix caching eliminates prefill for it entirely after the first request.
- **RAG with recurring documents**: when the retrieved chunks overlap across queries, the overlapping prefix is cached.
- **Multi-turn conversations**: turn $N+1$'s prompt is turn $N$'s prompt plus a few tokens; prefix cache hit approaches 100%.
- **Few-shot prompting**: the few-shot examples are reused across queries.

Empirically, prefix caching gives **3–10× throughput on real chat workloads** and is essentially free (small metadata overhead). Every serious serving system should have it on.

### What Breaks Prefix Caching

- **Different system prompts per user.** If the system prompt includes `{user_name}` or `{current_time}`, the prefix hash changes and nothing is cached. Solution: put user-specific content at the end of the prompt.
- **Different sampling parameters affecting logits-level state.** (Actually fine — KV cache is independent of sampling.)
- **RoPE scaling changes.** If positional encoding depends on absolute position (classical RoPE), a cached prefix is only valid at the position it was computed. This is why SGLang carefully stores positional offsets with tree nodes.
- **Tokenizer changes.** Obvious but easy to miss in multi-model deployments.

## Part 5 — Architectural Optimizations: MHA → MQA → GQA → MLA

![Evolution of attention for KV cost: MHA (full KV per head) to MQA (single KV head) to GQA (grouped KV, Llama-3/Qwen default) to MLA (low-rank latent KV, DeepSeek-V3)](/imgs/blogs/kv-cache-opt-03-mha-mqa-gqa-mla.png)

The single largest lever is the attention architecture itself. The goal of this evolution: reduce $H_{kv}$ (the KV head count) in the cache size formula.

### Multi-Head Attention (MHA) — Baseline

Standard transformer: $H_q = H_{kv} = H$. Each query head has its own K and V. For $H=64$, $d_{head}=128$: per-token KV cache = $2 \cdot 64 \cdot 128 = 16{,}384$ bytes (BF16) per layer.

### Multi-Query Attention (MQA) — Aggressive

[Shazeer 2019]: $H_{kv} = 1$. All query heads share one K and one V. For $H=64$: per-token KV cache = $2 \cdot 1 \cdot 128 = 256$ bytes per layer — **64× reduction**.

Problem: quality degrades noticeably on complex tasks. Falcon-180B used MQA and was widely criticized for it. MQA is too aggressive for most modern models.

### Grouped-Query Attention (GQA) — The Standard

[Ainslie et al. 2023]: $H_{kv} = H_q / g$ for some group size $g$ (typical: 4, 8, 16). Query heads within a group share one K and V. Llama-3-70B uses $H_q=64$, $H_{kv}=8$ (g=8): **8× cache reduction vs MHA** with negligible quality loss.

GQA is now the de facto standard. Every production-scale model after 2023 (Llama 2/3, Mistral, Gemma, Qwen, DeepSeek) uses it. When serving a model, check `config.json` for `num_key_value_heads` — if it's less than `num_attention_heads`, the model uses GQA.

### Multi-Head Latent Attention (MLA) — DeepSeek-V2/V3

[DeepSeek 2024]: instead of caching $H_{kv}$ K/V heads, cache a **low-dimensional latent vector** $c_{kv} \in \mathbb{R}^{d_c}$ (typically $d_c = 512$) per token per layer. During attention, expand the latent into K and V via learned matrices.

$$
c_{kv} = W_{DKV} \, h, \quad K = W_{UK} c_{kv}, \quad V = W_{UV} c_{kv}
$$

The cache only stores $c_{kv}$. For DeepSeek-V2: $d_c = 512$ vs. full K+V dimension ~2560 → **~5× smaller cache than GQA, ~93% smaller than MHA**.

The catch: the up-projection $W_{UK}, W_{UV}$ can be absorbed into $W_Q$ and $W_O$ during inference via matrix multiplication associativity — so there's no extra compute at decode time. This is a genuinely beautiful engineering trick.

DeepSeek-V3 (671B MoE) is tractable to serve largely because of MLA. Without it, the KV cache alone would blow the memory budget.

### Hybrid Attention: Local + Global

Some architectures mix sliding-window attention (fixed-cache-per-layer) with full attention (growing cache) on alternating layers.

- **Mistral** (sliding window 4K): every layer uses sliding window; cache is bounded by window size.
- **Gemma 2**: interleaves local (sliding) and global (full) attention layers.
- **Character.ai architecture**: most layers sliding, few layers full — they reported 33× memory reduction on their serving stack.

For a 32K-context model with 32 layers and sliding window 4K on 28 of them: effective cache is $4 \cdot 32000 + 28 \cdot 4000 = 240$K slots instead of $32 \cdot 32000 = 1.024$M slots — ~4× cache reduction.

### Linear Attention and State-Space Models

An entirely different approach: replace attention with a mechanism whose state is $O(1)$ per step rather than growing with $T$. Examples: RetNet, Mamba, RWKV, GLA.

- **Trade-off**: fixed-size recurrent state means no "infinite memory"; quality on long-context recall tasks tends to be weaker.
- **Benefit**: KV cache is eliminated entirely. Throughput and memory are independent of sequence length.
- **Current state**: hybrid architectures (e.g. Jamba, Zamba) combine a few attention layers with many state-space layers, getting most of the quality of attention with much of the efficiency of SSMs.

For most production LLMs today, standard attention + GQA + KV cache management is still the winning stack. But the landscape will continue to evolve.

## Part 6 — KV Cache Quantization

If architectural changes reduce the number of cached entries, quantization reduces the bytes per entry. The formula has a `dtype_bytes` factor — halve it and you halve the cache.

### FP8 KV Cache

FP8 (e4m3 or e5m2) halves memory vs BF16 with negligible quality loss if done correctly. Native support on H100 (and later). Typical behavior:

- Quality drop on standard benchmarks: < 0.5%.
- Memory reduction: 2×.
- Compute: K and V are loaded in FP8, dequantized on-the-fly in the attention kernel.

Available in vLLM (`--kv-cache-dtype fp8`), SGLang (`--kv-cache-dtype fp8_e5m2` or `fp8_e4m3`), and TensorRT-LLM (first-class).

Trap: if the model was not trained to be robust to FP8 KV (most aren't), you'll need **per-tensor or per-channel scales**. FP8 e4m3 (narrower range, more precision) is generally safer for KV than e5m2.

### INT8 KV Cache

Requires quantization scales computed from a calibration set. Per-token per-head scales are common. Quality drop slightly larger than FP8 but still under 1% on most tasks.

### INT4 KV Cache and KVQuant

[KVQuant, Hooper et al. 2024]: INT4 (or even mixed-precision 2-bit/4-bit) KV cache using several orthogonal tricks:

1. **Per-channel quantization for K** (K has large per-channel outliers — quantizing per-channel preserves them).
2. **Per-token quantization for V** (V outliers are per-token).
3. **Pre-RoPE quantization** for K (RoPE rotation mixes channels, destroying the per-channel outlier structure).
4. **Dense-and-sparse** decomposition — store outlier values in a sparse format at full precision.

Result: **4× memory reduction** vs FP16 with < 1% perplexity increase on long-context tasks. Enables 10M-token contexts on a single GPU.

Trade-off: the quantization/dequantization kernel is complex and slower than FP8. Use when memory is truly the binding constraint (very long contexts or very large models).

### The Interaction with Prefix Caching

Quantization is orthogonal to prefix caching. A quantized KV cache block is exactly as shareable as a full-precision one. The only subtlety: the scales must be consistent. Modern frameworks handle this by either storing global scales or making scales part of the block metadata.

## Part 7 — KV Cache Eviction: Keep Only What Matters

![KV cache eviction, offloading, and compression strategies: H2O heavy hitters, StreamingLLM attention sinks + window, SnapKV clustering, CPU/NVMe offload, INT8/FP8/INT4 quantization](/imgs/blogs/kv-cache-opt-04-eviction.png)

Quantization shrinks each entry; eviction deletes entries entirely. This is lossier than quantization (you lose exact tokens) but can be dramatic.

### StreamingLLM / Sink Attention

[Xiao et al. 2023]: surprising empirical finding — the first few tokens of a sequence (the "attention sinks") receive huge attention weights regardless of their content. Dropping them breaks the model. Dropping middle tokens often doesn't.

**Streaming attention**: keep `sink_size` (e.g., 4) initial tokens + the most recent `window_size` (e.g., 4K) tokens. Evict everything in between. The model keeps producing coherent text essentially forever.

Use case: infinite-length chat where you don't need exact recall of old context. Memory is bounded by `sink + window` regardless of total tokens generated.

### H2O: Heavy-Hitter Oracle

[Zhang et al. 2023]: across many generations, only a small subset of tokens (~20%) receive high attention weights consistently — the "heavy hitters". Evict the bottom 80% based on accumulated attention scores.

- Keep a running sum of attention weights each token has received.
- After each decode step, evict low-score tokens.
- Retains ~80% of quality with 5× KV cache reduction.

### SnapKV and Adaptive Eviction

[SnapKV 2024]: improves on H2O by using the last few queries to identify which past tokens will matter most, then evicting the rest. Works particularly well for long-context tasks.

More recent variants (PyramidKV, AdaKV) allocate different cache budgets to different layers — middle layers keep more, early/late layers keep less.

### When Eviction Is Safe

- **Summarization, classification, generation from long context** — yes, most heavy-hitter methods work.
- **Needle-in-haystack / exact recall** — no, eviction destroys the signal. These tasks need full cache or carefully tuned eviction.
- **Multi-turn chat** — risky, because the model may need to recall a fact from earlier that wasn't a heavy hitter at the time.

In production, eviction is still less common than quantization because the quality risks are task-dependent and hard to monitor. It shines in specific long-context offline workloads.

## Part 8 — KV Cache Offloading

Sometimes the KV cache simply doesn't fit on the GPU, and you want to serve it anyway. Offload to CPU RAM or NVMe.

### Memory Hierarchy

| Tier | Capacity | Bandwidth | Latency |
|------|----------|-----------|---------|
| GPU HBM | 80–192 GB | 3.35 TB/s (H100) | ~100 ns |
| CPU DRAM | 1–2 TB | 400–500 GB/s | ~1 μs via PCIe 5.0 |
| NVMe SSD | 10s of TB | 7 GB/s (per drive) | ~100 μs |

Offloading KV cache to CPU is viable because PCIe 5.0 delivers ~64 GB/s — slow compared to HBM but fast enough to overlap with compute if you're careful.

### vLLM CPU Offloading

vLLM supports CPU swap space (`--swap-space N` where N is GB). Preempted sequences have their KV cache swapped to CPU instead of dropped. When the sequence resumes, the KV cache is swapped back. Typical overhead: 50–200 ms for a full sequence, much less than recomputing prefill.

### CacheGen and Hierarchical Caching

[CacheGen 2023]: treats the KV cache as a compressible artifact that can be streamed from storage. Applies delta encoding and entropy coding to KV cache bytes, storing long conversation histories on disk and streaming them back with < 100 ms latency.

For chat products with very long user histories, this is the only way to preserve full state without burning GPU memory on idle users.

### Speculative Prefetching

If you know a sequence will be scheduled next, start transferring its KV cache from CPU to GPU before the scheduler picks it. Overlaps PCIe transfer with other sequences' decode steps. Worth the complexity in very large-scale serving.

## Part 9 — Framework Comparison

### vLLM

- **Core**: PagedAttention + continuous batching.
- **Prefix cache**: opt-in hash-based (`--enable-prefix-caching`).
- **Quantization**: FP8 (e4m3, e5m2), INT8, AWQ/GPTQ weights.
- **Preemption**: recompute (default for short) / swap (default for long).
- **Chunked prefill**: yes (`--enable-chunked-prefill`).
- **Attention kernel**: FlashAttention, FlashInfer, PagedAttention v1/v2, Triton.
- **Best for**: general-purpose serving, broad model support, the easiest starting point.

### SGLang

- **Core**: RadixAttention + continuous batching.
- **Prefix cache**: automatic via radix tree (no config needed).
- **Quantization**: FP8 (e4m3/e5m2), INT8.
- **Preemption**: LRU eviction of tree nodes.
- **Chunked prefill**: yes.
- **Attention kernel**: FlashInfer, Triton.
- **Frontend DSL**: first-class support for structured LLM programs (branching, constrained decoding, tool use).
- **Best for**: multi-turn chat, agentic workflows, anywhere prefix reuse is high.

### TensorRT-LLM

- **Core**: paged KV blocks + custom CUDA kernels.
- **Prefix cache**: supported (`enable_block_reuse`).
- **Quantization**: FP8 (native on H100), INT8, INT4 (AWQ, SmoothQuant), W4A8.
- **Preemption**: request queuing.
- **Chunked prefill**: yes ("chunked context").
- **Attention kernel**: hand-written CUDA, often the fastest per-GPU.
- **Best for**: squeezing maximum throughput out of NVIDIA hardware; requires more engineering effort to deploy.

### TGI (Text Generation Inference, HuggingFace)

- **Core**: paged attention (via vLLM's implementation).
- **Prefix cache**: supported.
- **Quantization**: FP8, AWQ, GPTQ, EETQ.
- **Strong points**: production-tested deployment, tight HuggingFace ecosystem integration.

### MLC-LLM / llama.cpp / ExLlamaV2

Edge/consumer-focused. Less relevant for large-scale serving but important for on-device LLM applications. They typically implement simpler KV cache management (contiguous per-sequence buffers) because concurrency is low.

## Part 10 — Advanced Topics

### Disaggregated Prefill and Decode

Prefill is compute-bound; decode is memory-bandwidth-bound. They stress different parts of the GPU. Running them together causes interference — prefill starves decode of compute, decode starves prefill of memory bandwidth.

**Disaggregated serving** separates them onto different GPU pools:

- Prefill nodes: run only prefill; optimize for compute throughput.
- Decode nodes: run only decode; optimize for memory bandwidth.
- When prefill finishes, transfer the KV cache to the decode node via NVLink or high-speed network.

Papers: Splitwise, DistServe, Mooncake. Reported gains: 2–3× cost/throughput improvement on mixed workloads. Adopted by the largest commercial LLM providers.

The KV cache transfer is the interesting part: for a 70B model with 8K context, that's ~2.5 GB of KV cache to transfer per request. At 400 GB/s NVLink, that's 6 ms — faster than a prefill chunk. On multi-node, RDMA (GPUDirect, NCCL, UCX) is used.

### Mooncake: KV Cache as a First-Class Distributed Resource

[Mooncake, Moonshot AI]: treats the KV cache as a distributed object store, separate from compute. Advantages:

- Decouples cache lifetime from request lifetime.
- Cache blocks can be served from any node with GPU, CPU, or NVMe.
- Prefix caching becomes globally available across a whole cluster.

Conceptually similar to how modern databases separate compute and storage. Likely the direction large-scale LLM serving is heading.

### Speculative Decoding and KV Cache

Speculative decoding uses a small draft model to propose $k$ tokens, which the target model verifies in a single forward pass. If $m$ tokens are accepted, you get $m$ tokens per forward pass instead of 1.

Interaction with KV cache:

- KV cache is **still written** for all $k$ drafted tokens during verification — but only the accepted prefix is kept.
- This means speculation temporarily needs $k$ extra KV cache slots per sequence.
- Biggest win in memory-bandwidth-bound regimes (small batch, long sequences): amortizes KV cache loading across multiple accepted tokens.
- Less effective when already compute-bound (large batch): the verification is the same as $k$ decode steps of compute, so you only win if acceptance is high.

Medusa, EAGLE, and self-speculation avoid the separate draft model by training extra heads on the target model itself — reduces deployment complexity but adds a training step.

### YOCO: You Only Cache Once

[YOCO, Microsoft 2024]: redesign the architecture so only the final few layers produce a cached KV; the early layers are "encoder-like" and processed once. Result: KV cache size reduced by ~$L$ (the number of layers) — for a 70B model with 80 layers, almost 80× reduction.

Requires a new architecture (not compatible with pretrained Llama-style models). Promising direction; not yet widely adopted.

### Tensor / Pipeline / Expert Parallelism and KV Cache

- **Tensor parallelism (TP)**: K and V heads are sharded across GPUs. Per-GPU KV cache is $1/\text{TP}$ of the total. Latency friendly; requires high-bandwidth interconnect (NVLink).
- **Pipeline parallelism (PP)**: layers are split across GPUs. Per-GPU KV cache is $1/\text{PP}$ of layers. Better for multi-node when interconnect is weaker.
- **Expert parallelism (EP)**: MoE experts are distributed. KV cache unaffected since attention is dense.

When serving with TP, always ensure the KV cache sharding matches the query/key/value computation sharding — otherwise you need cross-GPU communication in every attention step.

## Part 11 — Production Monitoring and Diagnosis

### Metrics to Track

| Metric | What it tells you | Typical threshold |
|--------|-------------------|-------------------|
| KV cache utilization (%) | How full the cache pool is | Sustained > 90% → add capacity |
| Preemption rate (per-min) | Cache pressure causing evictions | > 1% of requests → problem |
| Prefix cache hit rate (%) | Reuse effectiveness | 40–80% on real chat workloads |
| TTFT (Time To First Token) | Prefill latency | SLO-dependent, often < 500 ms |
| TPOT (Time Per Output Token) | Decode latency | SLO-dependent, often < 50 ms |
| Throughput (tokens/sec) | Aggregate output rate | Depends on batch size |
| GPU memory breakdown | Weights / KV / activations | KV should be 50–80% of free mem |

### Diagnostic Playbook

**Symptom: High TPOT (> 100ms)**

1. Check batch size: too low → raise it (may need KV quantization to fit more).
2. Check decode arithmetic intensity: if < 1, the GPU is idle waiting for HBM. Enable FP8 KV cache (2× effective bandwidth).
3. Check if prefills are blocking decode: enable chunked prefill.
4. Check if the model is MHA (not GQA/MLA): if yes, consider switching to a GQA variant.

**Symptom: Preemption rate > 1%**

1. Reduce `max_seq_len`.
2. Enable FP8 KV cache.
3. Scale out (more GPUs).
4. Check for requests that are much longer than the average — impose a hard cap and reject oversize.

**Symptom: High TTFT**

1. Long prompts + no chunked prefill → enable chunked prefill.
2. Enable prefix caching.
3. Increase TP for better prefill compute.
4. Consider disaggregated prefill/decode.

**Symptom: Low prefix cache hit rate**

1. Check if user-specific variables are at the start of the prompt — move them to the end.
2. Verify system prompts are fully deterministic (no timestamps, no UUIDs).
3. Confirm the framework has prefix caching enabled (vLLM requires `--enable-prefix-caching`; SGLang is always on).

**Symptom: OOM on model load**

1. Reduce `gpu_memory_utilization` (vLLM default 0.9 → try 0.85).
2. Reduce `max_seq_len` (controls pre-reserved KV pool size).
3. Increase TP size.
4. Enable weight quantization (AWQ, GPTQ, FP8).

## Part 12 — Interview Question Bank

### Fundamentals

**Q: What is the KV cache and why is it needed?**

In autoregressive generation, attention at step $t$ needs K and V for all tokens $1..t$. Without caching, you'd recompute K and V for tokens $1..t-1$ at every step — quadratic total cost. With caching, you compute K and V once per token, then reuse. The trade-off is memory: the cache grows linearly with sequence length and consumes the majority of GPU memory outside weights.

**Q: Compute the KV cache size for Llama-3-70B at 8K context and batch size 32.**

$L=80$, $H_{kv}=8$, $d_{head}=128$, BF16 (2 bytes), $T=8192$, $B=32$.

$$
2 \times 80 \times 8 \times 128 \times 2 \times 8192 \times 32 \approx 86 \text{ GB}
$$

Doesn't fit on a single H100. Needs TP=2 minimum just for KV, plus space for weights.

**Q: Why is decode memory-bandwidth-bound but prefill compute-bound?**

In prefill, you process $T$ tokens × batch in parallel. The matmul has $O(T^2)$ compute but loads each weight tile only once — arithmetic intensity is high, roofline-bound by compute.

In decode, you process 1 token per sequence but must load the entire KV cache ($O(T)$ bytes) to compute attention for just one new query. Arithmetic intensity is approximately $1 / \text{dtype\_bytes}$ ≈ 0.5 for BF16 — three orders of magnitude below the H100 roofline crossover. So decode is bound by HBM bandwidth, not FLOPs.

### Memory Management

**Q: Explain PagedAttention. What problem does it solve?**

Pre-vLLM, frameworks pre-allocated contiguous KV cache buffers per sequence with size `max_seq_len × layers × heads × d`. This wasted 60–80% of memory to internal fragmentation (unused slots in each buffer) and external fragmentation (Swiss-cheese memory as sequences finished).

PagedAttention divides GPU KV memory into fixed-size blocks (typically 16 tokens each), allocates blocks on demand, and maintains a per-sequence block table listing block IDs in logical order. The attention kernel is rewritten to gather K/V via block table indirection. Fragmentation drops to under 4%, enabling 2–4× higher throughput.

**Q: How does copy-on-write work in PagedAttention for beam search?**

When a sequence forks (beam search, parallel sampling), children share the parent's blocks via reference counting. Only when a child writes into the last, partially-filled block is the block actually copied. Shared blocks (which are the vast majority — the entire prompt prefix) are never duplicated. This is identical to Unix `fork()` / copy-on-write.

**Q: When does vLLM preempt a sequence? What are the two strategies?**

When new requests arrive and the block manager can't allocate enough blocks for them. Strategies:

- **Swap**: move the victim's KV cache to CPU memory, restore it when blocks are free. Cost: PCIe transfer (tens of ms).
- **Recompute**: drop the victim's KV cache entirely, rerun prefill when resumed. Cost: full prefill time.

Default: recompute for short sequences, swap for long ones. Preemption rate is a critical production metric.

### Reuse / Prefix Caching

**Q: Compare vLLM's hash-based prefix caching with SGLang's RadixAttention.**

vLLM hashes 16-token blocks and keeps `(hash, parent_hash) → block_id`. Simple, fast, but block-granular — can't share a prefix of 23 tokens (only 16 of the 23 get reused).

SGLang maintains a radix tree over all cached sequences. Arbitrary-depth prefix matching is automatic; every request's output is inserted into the tree for future reuse. Any two sequences sharing any prefix share the corresponding KV blocks automatically, without user configuration.

Result: SGLang tends to have higher cache hit rates on multi-turn chat (where each turn extends the conversation and the tree structure is natural), while vLLM is simpler and integrates more cleanly with its paged attention.

**Q: What workload patterns break prefix caching?**

- User-specific tokens at the start of the prompt (names, IDs, timestamps).
- Randomized system prompts (for A/B testing).
- RoPE scaling with absolute position dependence (the cached block's positional encoding is invalid at a different position).
- Different model / tokenizer / LoRA adapter (obviously).

Fix: put invariant content first, variant content last.

### Architecture

**Q: Compare MHA, MQA, GQA, and MLA.**

- **MHA**: $H_{kv} = H_q$. Baseline. Largest cache, best quality.
- **MQA**: $H_{kv} = 1$. 64× smaller cache for $H_q=64$. Noticeable quality loss.
- **GQA**: $H_{kv} = H_q / g$. Typical $g=4$ or $g=8$. 4–8× cache reduction, negligible quality loss. Industry standard since 2023.
- **MLA**: cache a low-dim latent $c_{kv}$ instead of K/V directly; re-expand at attention. ~5× smaller than GQA, ~93% smaller than MHA. Slightly more compute per attention step, but the up-projections can be absorbed into $W_Q$ and $W_O$, so in practice the overhead is negligible. DeepSeek-V2/V3.

**Q: What's the mathematical trick that makes MLA essentially free at inference?**

The up-projection matrices $W_{UK}, W_{UV}$ can be merged into $W_Q$ and $W_O$ respectively via matrix associativity:

$$
(Q \cdot W_Q)(K \cdot W_{UK} \cdot c_{kv})^T = Q \cdot W_Q W_{UK}^T \cdot c_{kv}
$$

So at inference you multiply by the precomputed $W_Q W_{UK}^T$ and operate directly on the cached $c_{kv}$. No actual up-projection happens at decode time — the cache stays small and attention still works.

**Q: How does sliding-window attention reduce KV cache?**

Each layer attends only to the last $W$ tokens. The effective KV cache per layer is bounded at $W$ regardless of total sequence length. For a $T=32$K context with $W=4$K, sliding-window layers cache $W$ entries instead of $T$ — an 8× reduction per sliding layer.

Global receptive field is achieved by stacking: with $L$ layers of window $W$, the model can attend to $L \cdot W$ tokens through the layer stack.

### Quantization and Compression

**Q: How does FP8 KV cache work, and what are the quality trade-offs?**

Store K and V in 8-bit floating point (e4m3 or e5m2), dequantize on-the-fly in the attention kernel. Memory halves, quality drops under 0.5% on standard benchmarks. Native H100 hardware support.

e4m3 has narrower range but more precision — preferred for KV.
e5m2 has wider range — sometimes needed for extreme outliers.

Requires per-tensor or per-channel scales. Most frameworks compute these from a short calibration run or use sensible defaults.

**Q: What is KVQuant and when would you use it?**

KVQuant pushes KV cache to INT4 (or mixed 2/4-bit) using per-channel quantization for K (which has channel-wise outliers), per-token quantization for V, pre-RoPE quantization for K (since RoPE rotation mixes channels and destroys the per-channel outlier structure), and a sparse side-channel for outlier values.

4× memory reduction vs FP16 with < 1% perplexity loss. Use when memory is the binding constraint — very long contexts (millions of tokens) or very large models on constrained hardware.

**Q: Explain heavy-hitter eviction (H2O). When is it safe?**

Track accumulated attention weights per token; evict tokens with the lowest scores. Surprisingly, ~20% of tokens account for most attention weight, so you can evict 80% and keep quality.

Safe: generation, summarization, classification from long context.

Unsafe: needle-in-haystack recall (the target token might have low accumulated attention until it's needed), multi-turn chat where arbitrary past facts might become relevant.

### Systems / Serving

**Q: Why is continuous batching tightly coupled with paged KV cache?**

Continuous batching changes the batch composition every decode step. That means KV cache must be allocatable and freeable per-sequence, at every step, with essentially zero overhead. Paged allocation with O(1) block alloc/free makes this cheap; pre-allocated contiguous buffers would make it impossible.

**Q: Explain chunked prefill and why it helps P99 TPOT.**

A single 10K-token prefill monopolizes the GPU for ~2s. During that time, all decoding sequences are blocked — their TPOT spikes to ~2s. Chunked prefill splits the prefill into ~1K-token chunks and interleaves with decode steps. Total prefill wall time increases slightly, but decoders get regular slots (~10ms gaps), so P99 TPOT stays healthy.

**Q: How does disaggregated prefill/decode serving work?**

Separate GPU pools for prefill (compute-bound) and decode (memory-bandwidth-bound). Prefill runs on one pool; when it finishes, transfer the KV cache to the decode pool via NVLink / RDMA. Each pool is tuned for its bottleneck (prefill wants FLOPs; decode wants HBM bandwidth and large batches).

Papers: Splitwise, DistServe, Mooncake. Typical gain: 2–3× cost/throughput vs. colocated serving on mixed workloads.

**Q: What is Mooncake's approach to KV cache?**

Treat the KV cache as a distributed, persistent object store — separate from compute. Cache blocks can live on GPU, CPU, or NVMe, across any node in a cluster. Prefix caching becomes globally available (a request routed to any node can pull the relevant prefix). Parallels the compute-storage separation in modern databases.

### Advanced

**Q: How does speculative decoding interact with the KV cache?**

During verification, the target model produces K and V for all $k$ drafted tokens in a single forward pass. Those $k$ entries occupy cache temporarily; after verification, only the accepted prefix's KV entries are kept.

Speculation helps most in memory-bandwidth-bound regimes (small batch, long context) because it amortizes KV cache loads across multiple accepted tokens. It helps less in compute-bound regimes because verification compute is similar to $k$ decode steps.

**Q: Design an LLM architecture to minimize KV cache overhead.**

1. GQA with $H_{kv} = H_q / 8$ or MLA with $d_c = 512$.
2. Hybrid attention: sliding window on most layers, full attention on a few (e.g., 1 in 4).
3. Support native FP8 KV (train with FP8 rounding noise if possible).
4. Avoid RoPE pathologies for prefix cache compatibility.
5. Design for prefix sharing: consistent positional encoding, tokenizer stability.
6. Consider YOCO-style "cache once, reuse many layers" if quality permits.
7. Offload intermediate cache to CPU / NVMe for cold sequences.

**Q: You're told TPOT is 150ms but should be 30ms. What's your diagnostic process?**

1. Check decode batch size. If < 16, increase it (arithmetic intensity too low).
2. Check KV dtype. If BF16, enable FP8 → 2× effective bandwidth.
3. Check attention architecture. If MHA, consider a GQA variant.
4. Check if prefills are interleaved. If yes, enable chunked prefill.
5. Check GPU utilization. If > 95% and TPOT is still high, you're compute-bound — scale out with TP.
6. Check for memory pressure causing eviction/recomputation. If yes, reduce max_seq_len or quantize.
7. If none of the above helps, measure actual HBM bandwidth utilization. Anything < 70% means the attention kernel is not optimal — try FlashAttention / FlashInfer.

**Q: What's the single most impactful decision you'd make as a new infra lead at a company serving LLMs?**

Enable prefix caching (3–10× throughput on real chat workloads). If it's already on, enable FP8 KV cache (2× memory → 2× batch → ~1.5× throughput). If both are on, add disaggregated prefill/decode (2–3×). Together, these three routinely yield 10–30× cost improvement over naive serving, and none of them require changing the model.

## Part 13 — Summary

The KV cache is both the biggest memory consumer and the biggest throughput lever in LLM serving. Managing and optimizing it is a multi-layer problem:

| Layer | Tools |
|-------|-------|
| **Architecture** | GQA, MLA, sliding window, hybrid attention |
| **Representation** | FP8/INT8/INT4 quantization, KVQuant |
| **Pruning** | H2O, StreamingLLM, SnapKV, PyramidKV |
| **Allocation** | PagedAttention, block managers, copy-on-write |
| **Reuse** | Prefix caching (hash / radix tree), continuous batching |
| **Tiering** | CPU offload, NVMe caches, CacheGen |
| **Systems** | Chunked prefill, disaggregated prefill/decode, Mooncake, distributed KV |
| **Algorithmic** | Speculative decoding, Medusa, EAGLE |

The best production systems stack these. A real-world deployment of Llama-3-70B on H100 at a competent serving company looks something like:

- GQA model (architecture-level, fixed).
- FP8 KV cache (2× memory).
- PagedAttention with 16-token blocks.
- Prefix caching on by default.
- Chunked prefill with 1K-token chunks.
- Continuous batching.
- Disaggregated prefill/decode on large deployments.
- CPU swap for overflow.

Any single layer gets you 1.5–4×. All layers together get you 20–50× over a naive implementation — the difference between "this is economically infeasible" and "this is a product."

## References

1. Vaswani et al., "Attention Is All You Need" (2017)
2. Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need" (2019) — MQA
3. Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (2023)
4. DeepSeek-AI, "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model" (2024) — MLA
5. Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (2023) — vLLM
6. Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs" (2024) — RadixAttention
7. NVIDIA, "TensorRT-LLM" documentation — FP8, chunked context, KV cache reuse
8. Dao et al., "FlashAttention" / "FlashAttention-2" / "FlashAttention-3" (2022–2024)
9. Xiao et al., "Efficient Streaming Language Models with Attention Sinks" (2023) — StreamingLLM
10. Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs" (2023)
11. Li et al., "SnapKV: LLM Knows What You Are Looking for Before Generation" (2024)
12. Hooper et al., "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization" (2024)
13. Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023)
14. Cai et al., "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads" (2024)
15. Patel et al., "Splitwise: Efficient Generative LLM Inference Using Phase Splitting" (2023)
16. Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving" (2024)
17. Qin et al., "Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving" (2024)
18. Sun et al., "You Only Cache Once: Decoder-Decoder Architectures for Language Models" (2024) — YOCO
19. Pope et al., "Efficiently Scaling Transformer Inference" (2023)
20. Liu et al., "CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving" (2023)
