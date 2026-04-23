---
title: "Efficient LLM Inference Techniques: A Complete Guide"
publishDate: "2026-04-17"
category: "machine-learning"
subcategory: "Large Language Model"
tags: ["llm", "inference", "optimization", "kv-cache", "quantization", "speculative-decoding", "batching", "vllm", "sglang", "deep-learning"]
date: "2026-04-17"
author: "Hiep Tran"
featured: false
aiGenerated: true
excerpt: "LLM inference is slow and expensive because of two fundamental bottlenecks: memory bandwidth and sequential decoding. This guide walks through every major technique used to beat them — KV cache, continuous batching, PagedAttention, quantization, speculative decoding, FlashAttention, and more — with intuition and clear explanations."
---

## Why LLM Inference Is Hard

![LLM inference techniques mapped to bottlenecks: memory-bandwidth (decode) vs compute (prefill/batch)](/imgs/blogs/efficient-llm-inference-techniques-diagram.png)

Training an LLM is expensive but finite. You do it once (or a few times), it finishes, you have a model. **Inference** is the other story: every single token served to a user costs GPU time. A popular model serving millions of queries a day can easily spend more on inference in a month than training cost in total.

Worse, LLM inference is *inherently* slow. A model with 70 billion parameters needs to generate tokens **one at a time**, and each token requires reading all ~140 GB of weights from GPU memory. There's no way to skip a token, no way to parallelize within a single response.

The entire field of efficient inference is an attempt to work around these facts. This article is a map of what the techniques actually do, why they work, and when to reach for each.

## The Two Bottlenecks (Know These Before Everything Else)

![Prefill vs decode: prefill is compute-bound (large GEMMs, high arithmetic intensity) while decode is memory-bound (reads full weights + KV per step) — each needs different optimizations](/imgs/blogs/eff-inf-01-bottlenecks.png)

Every LLM inference technique exists to attack one of two problems. Understanding which is the dominant bottleneck in your setup tells you which techniques will help.

### Bottleneck 1: Memory Bandwidth (The Decode Phase)

During **decode** — generating tokens one at a time after the prompt is processed — the GPU spends most of its time **moving data**, not computing.

For each new token, the GPU must:

1. Read all model weights from HBM (high-bandwidth memory) into the compute units
2. Read the KV cache (stored past keys and values for all previous tokens)
3. Do relatively little actual math
4. Write the result back

For a 70B model in FP16, that's ~140 GB of weights streamed from HBM for *every single token*. On an A100 with ~2 TB/s of memory bandwidth, the theoretical floor is:

$$
\frac{140 \text{ GB}}{2000 \text{ GB/s}} = 70 \text{ ms per token}
$$

This is a **bandwidth-bound** regime. The GPU's math units (which can do 312 TFLOPS on A100) sit mostly idle because they're waiting on memory. Classic sign: GPU utilization looks high in `nvidia-smi`, but you're nowhere near peak FLOPS.

### Bottleneck 2: Compute (The Prefill Phase)

The **prefill** phase — processing the user's prompt before generating the first response token — is different. Here, you process many tokens at once (the whole prompt in parallel), so each weight read is amortized across many FLOPs.

Prefill is **compute-bound**. You're actually using the GPU's math units, and throughput is limited by FLOPS, not bandwidth.

This distinction matters a lot:

| Phase | What's happening | Bottleneck | Latency impact |
|---|---|---|---|
| **Prefill** | Process whole prompt at once | Compute | Time to First Token (TTFT) |
| **Decode** | Generate one token at a time | Memory bandwidth | Time Per Output Token (TPOT) |

When an engineer says "decoding is memory-bound, prefill is compute-bound," this is what they mean. **Most efficient-inference techniques target decode, because decode dominates total latency for any response longer than a few tokens.**

## The Techniques, Grouped by What They Attack

![Taxonomy of efficient LLM inference techniques: memory attack (KV cache + quant + eviction), compute attack (FlashAttention, speculative, fusion), scheduling (continuous batching, PagedAttention, prefix cache), parallelism (TP/PP/EP)](/imgs/blogs/eff-inf-02-taxonomy.png)

Here's the map. We'll walk through each group.

```
Memory bottleneck        Compute bottleneck         Both / Serving-level
─────────────────        ──────────────────         ────────────────────
KV Cache                  FlashAttention             Continuous batching
Quantization              Fused kernels              PagedAttention
MQA / GQA                 Tensor parallelism         Chunked prefill
KV cache compression      Compiled graphs            Prefill-decode disaggregation
Speculative decoding      Tensor cores (BF16/FP8)    Prefix caching
Parallel decoding                                    RadixAttention
Early exit                                           Request scheduling
```

## Part 1: KV Cache — The Single Most Important Optimization

Before anything else, every modern inference stack uses a **KV cache**. Without it, the rest of the optimizations don't matter.

The idea: attention for token $t$ requires the keys and values of all previous tokens. Without a cache, you'd recompute those $K$ and $V$ vectors for every previous token at every new step — pure waste. With a cache, you compute each token's $K$ and $V$ exactly once, store them, and reuse them forever.

This turns generation from $O(n^2)$ per-token work into $O(n)$ per-token work. It's a 100× speedup on long sequences.

The downside is memory: the KV cache grows linearly with sequence length. For a 70B model at 8K context and bf16:

$$
\text{KV bytes} = 2 \times L \times H_{\text{kv}} \times d_h \times T \times \text{batch}
$$

That easily blows past 20 GB for a single 8K conversation. The cache can be larger than the model itself for long contexts — which is why almost every other technique below is designed around the cache.

See the dedicated [KV Cache deep-dive](/blog/machine-learning/large-language-model/kv-cache) for the full walkthrough.

### MQA and GQA: Shrinking the KV Cache at Its Source

**Multi-Query Attention (MQA)** and **Grouped-Query Attention (GQA)** attack the KV cache memory problem architecturally. Instead of each attention head having its own key/value projections, multiple heads *share* keys and values.

- **MHA** (original): $H$ query heads, $H$ key heads, $H$ value heads
- **MQA**: $H$ query heads, **1** key head, **1** value head
- **GQA**: $H$ query heads, $G$ key heads, $G$ value heads (where $G$ divides $H$)

GQA is the modern default (used by Llama 3, Qwen, Mistral) because it hits the sweet spot: near-MHA quality with MQA-sized cache. For Llama 3-70B, GQA with 8 KV heads (vs. 64 query heads) shrinks the KV cache by 8×.

**Takeaway:** if you're picking a base model for a long-context application, GQA or MQA architecture is a big deal. You can't retrofit it after training.

### KV Cache Quantization and Compression

You can also compress the cache that's already in memory:

- **KV cache quantization** — store K and V in int8 or int4 instead of bf16. Typical loss: tiny quality regression, 2-4× memory savings.
- **KV cache offloading** — move older parts of the cache to CPU RAM or disk. Useful for very long contexts where not every past token is relevant every step.
- **Dynamic KV cache eviction / H2O / StreamingLLM** — drop unimportant past tokens entirely. Works because attention sinks (first few tokens) and recent tokens carry most of the signal.

## Part 2: Quantization — Make Every Byte Smaller

Quantization reduces the precision of model weights (and sometimes activations and KV cache). Instead of storing each weight as a 16-bit number, store it as 8, 4, or even 2 bits.

Why this helps inference so much:

1. **Less memory to move per token.** Decode is bandwidth-bound. A 4-bit model needs a quarter of the bandwidth of a bf16 model → roughly 4× faster decode, at the same batch size.
2. **More room for KV cache, longer context, bigger batch.** Smaller weights leave more HBM free for everything else.
3. **Cheaper GPUs become viable.** A 70B bf16 model needs 2× A100-80GB. A 4-bit quantized one fits in a single 48GB GPU.

Common quantization formats in 2026:

| Format | Bits | Where used | Quality impact |
|---|---|---|---|
| FP8 (E4M3) | 8 | H100, H200, B200 (native) | Negligible |
| INT8 | 8 | Broad GPU support | Small |
| GPTQ / AWQ (INT4) | 4 | Most open deployments | Small-to-moderate |
| NF4 (QLoRA style) | 4 | Fine-tuning + serving | Small |
| 1.58-bit / 2-bit | <2 | Research / extreme-budget | Significant |

The dedicated [Quantization in LLMs](/blog/machine-learning/large-language-model/quantization-in-llm) article walks through GPTQ, AWQ, SmoothQuant, and the tradeoffs.

**Rule of thumb:** bf16 → int8 is essentially free in quality. int8 → int4 needs careful calibration (GPTQ/AWQ). Below int4, you're in research territory.

## Part 3: FlashAttention — Smarter Attention Computation

Even with a KV cache, attention is still a large fraction of inference cost, especially for long contexts.

The standard attention formula is:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
$$

A naive implementation materializes the full $N \times N$ attention matrix in HBM. For $N = 8192$, that's a 64M-entry matrix written and read multiple times. HBM bandwidth is the bottleneck — again.

**FlashAttention** (and its descendants FlashAttention-2, FlashAttention-3) reorganizes this computation to **never materialize the full attention matrix in HBM**. Instead, it tiles the computation, keeps intermediate results in SRAM (fast on-chip memory), and uses an online softmax trick to fuse the operations.

The result is:

- **2-4× faster** attention kernels
- **10× less memory** for attention (linear in sequence length instead of quadratic)
- **Longer context** becomes feasible

FlashAttention is not something you implement yourself — it's a CUDA kernel shipped in PyTorch, vLLM, SGLang, and every modern inference engine. But you should know it exists because:

- It's on by default in modern stacks, but older ones might not have it
- Version matters: FA-2 and FA-3 each bring substantial further speedups
- Custom attention patterns (sliding window, etc.) may or may not have FA kernels — worth checking

## Part 4: Continuous Batching — The Serving-Layer Breakthrough

Single-request speedups are great. But production inference is about serving **many requests at once** on the same GPU. This is where continuous batching (also called **in-flight batching** or **iteration-level scheduling**) comes in.

### The Problem With Naive Batching

In a naive batched server, you wait for $B$ requests to arrive, stack them into a batch, run them together until all are done, then serve the next batch. Problems:

- **Head-of-line blocking.** If one request generates 500 tokens and another generates 50, you wait for the long one. The GPU sits at batch-of-1 efficiency while finishing it.
- **Idle waiting.** If requests arrive at different times, early arrivals wait.
- **Padding waste.** Different prompt lengths are padded to the longest — wasted compute.

### The Continuous Batching Idea

Instead of batching at the *request* level, batch at the *iteration* level. Each forward pass, the scheduler picks whichever active requests are ready to generate their next token, runs them together, and streams outputs back. A request that finishes is dropped from the batch; a new request can be added mid-generation.

```
Time →
Req A:  [prefill][t1][t2][t3][t4][done]
Req B:          [prefill][t1][t2][t3][t4][t5][t6][done]
Req C:                  [prefill][t1][t2][t3][done]
Req D:                              [prefill][t1][t2]...

Each vertical slice is one GPU forward pass with whichever requests are active.
```

Effective batch size stays high throughout. GPU utilization goes from ~20-40% (static batching) to 60-90% (continuous batching). This single technique often brings a 3-10× throughput improvement.

**Implementations:** vLLM pioneered this pattern. TGI, SGLang, TensorRT-LLM, Triton — they all have it. If you're running LLMs in production with anything else, you're leaving massive throughput on the table.

See the [vLLM Inference](/blog/machine-learning/large-language-model/vllm-inference) and [SGLang Inference](/blog/machine-learning/large-language-model/sglang-inference) articles for hands-on guides.

## Part 5: PagedAttention — Virtual Memory for the KV Cache

Continuous batching raises a new problem: **KV cache memory management**. Different requests have different sequence lengths, they grow unpredictably, and they come and go. Naively allocating a contiguous max-length buffer for each request wastes 60-80% of GPU memory on padding.

**PagedAttention** (also introduced by vLLM) applies the classic OS trick of **paging**: break the KV cache into fixed-size blocks (pages), and maintain a page table per request that maps logical sequence positions to physical pages.

Benefits:

- **No pre-allocation waste.** Each request only holds the pages it actually needs.
- **Non-contiguous allocation.** Fragmented GPU memory becomes usable.
- **Easy copy-on-write sharing** for prefix caching and parallel sampling.

The implementation requires a custom attention kernel that understands the page table, but from the outside it's transparent. Memory utilization for KV cache goes from ~30% (naive) to >90%. That translates to ~3× larger batch sizes on the same hardware — meaning ~3× throughput.

## Part 6: Prefix Caching and RadixAttention — Remember What You've Seen

In real applications, many requests share a common prefix: the same system prompt, the same few-shot examples, the same retrieved documents. Recomputing the KV cache for those shared prefixes every request is pure waste.

**Prefix caching** (or **prompt caching**) stores the KV cache of common prefixes on the GPU and reuses it across requests. For a system prompt of 2000 tokens shared across 1000 requests, that's a ~1000× savings on prefill compute for those tokens.

**RadixAttention** (SGLang's contribution) takes this further: it organizes cached prefixes in a radix tree so that *arbitrary* shared prefixes — not just the exact same system prompt — can be reused. Two requests sharing the first 1500 tokens but diverging at token 1501 automatically share a 1500-token cache.

Impact: for workloads with repeated prompts (RAG, agents, multi-turn chat), prefix caching is frequently the single biggest throughput win after continuous batching.

## Part 7: Speculative Decoding — Parallelizing the Sequential

Decode is sequential by nature — token $t$ depends on token $t-1$. **Speculative decoding** breaks this by using a small, fast "draft" model to guess several tokens ahead, then uses the big model to verify all of them in a single forward pass.

If $k$ tokens are drafted and accepted, you've generated $k$ tokens in the time of one big-model forward pass — a $k$× speedup. If some are rejected, you keep the accepted prefix and retry.

Crucially, this is **mathematically lossless**: with the right acceptance rule (rejection sampling), the output distribution is identical to sampling from the big model directly.

Variants:

- **Classical speculative decoding** — small draft model + big target model
- **Medusa / EAGLE** — predict multiple future tokens from the target model itself, with extra heads
- **Self-speculation / lookahead decoding** — no draft model; use algorithmic tricks on the target model
- **Prompt lookup decoding** — copy n-grams from the prompt as drafts (great for summarization/editing)

Typical speedup: 2-3× for general chat, up to 5× for structured outputs where drafts are easy. Read the dedicated [Speculative Decoding guide](/blog/machine-learning/large-language-model/speculative-decoding) for the full math.

## Part 8: Parallelism Strategies — Splitting the Model

![Parallelism strategies: Tensor Parallel (shard weights per layer, all-reduce, NVLink), Pipeline Parallel (stages across GPUs, micro-batching), Expert Parallel (distribute MoE experts, all-to-all routing)](/imgs/blogs/eff-inf-03-parallelism.png)

At scale, a single GPU isn't enough. Different parallelism strategies distribute the model across GPUs:

- **Tensor parallelism (TP)** — split each weight matrix across GPUs, use all-reduce after each layer. Low latency, requires fast interconnect (NVLink). Default for serving 70B+ models on a single node.
- **Pipeline parallelism (PP)** — place different layers on different GPUs. Higher latency (stages wait for each other) but lower interconnect needs. Useful for multi-node setups.
- **Expert parallelism (EP)** — for MoE models, place different experts on different GPUs. Routes tokens to the right expert's GPU.
- **Data parallelism (DP)** — replicate the full model on each GPU, shard the requests. Trivially scales throughput at the cost of memory.

In practice, production serving uses combinations: e.g., **TP within a node + DP across nodes**. The right choice depends on interconnect speed, model size, and whether you're latency- or throughput-bound.

## Part 9: Advanced Serving Tricks

### Chunked Prefill

A 32K-token prompt can hog the GPU for hundreds of milliseconds during prefill, blocking decode steps for other in-flight requests. **Chunked prefill** splits long prefills into smaller chunks (e.g., 512 tokens) that interleave with decode steps, keeping latency low for other users.

### Prefill-Decode Disaggregation

Prefill is compute-bound; decode is memory-bound. They want different hardware. **Disaggregated serving** (Mooncake, DistServe) runs prefill on one pool of GPUs and decode on another, transferring KV cache between them. This improves utilization on both sides — prefill nodes run at high FLOPS, decode nodes run at high bandwidth — and lets you scale the two pools independently as your workload's prompt-to-output ratio shifts.

### Request Scheduling and Priority

Production servers often need:

- **Priority lanes** — interactive chat prioritized over batch jobs
- **Max-tokens-per-batch limits** — cap decode memory to avoid OOM
- **SLA-aware scheduling** — TTFT vs. throughput tradeoffs per tenant
- **Fair sharing** — prevent one user's long requests from starving others

Modern engines (SGLang, TensorRT-LLM, vLLM) expose these as first-class features. They matter more than algorithmic tricks for multi-tenant production systems.

### Sampling Optimizations

Even after computing logits, generating a token isn't free:

- Top-k / top-p / min-p sampling all involve sorting or scanning the full vocabulary (often 128K-256K tokens).
- Repetition penalties, frequency penalties, and logit bias each touch the logits.
- Constrained decoding (JSON schema, regex) adds per-step overhead.

Fused sampling kernels and optimized constrained decoders (like SGLang's compressed FSM, Outlines' regex compilation) significantly reduce per-token overhead — especially for structured outputs.

## Part 10: Putting It Together — What Actually Matters in Production

If you're starting from a naive PyTorch `generate()` loop, here's roughly the order of impact:

1. **Pick a modern serving engine** (vLLM, SGLang, TensorRT-LLM). Gets you continuous batching + PagedAttention + FlashAttention out of the box. **Biggest single jump — often 5-10× throughput.**
2. **Quantize the model** — bf16 → int8 or int4. Near-free quality loss, 2-4× more throughput and bigger context.
3. **Enable prefix caching / RadixAttention.** If your workload has shared prefixes (RAG, agents, multi-turn), this can double throughput again.
4. **Turn on speculative decoding** for latency-sensitive paths.
5. **Consider MQA/GQA base models** for long-context workloads if you haven't already.
6. **Tune parallelism** (TP degree, chunked prefill chunk size, max batch size) to match your hardware and workload.
7. **Profile and find the actual bottleneck.** Memory? Compute? Tokenization? Network? Most teams have a non-obvious bottleneck that no optimization paper describes. Measure before optimizing further.

### Choosing an Engine

- **vLLM** — most popular, best community, great continuous batching + PagedAttention, solid general-purpose choice.
- **SGLang** — strong on structured outputs, RadixAttention prefix caching, and complex request patterns. Often fastest on agent/RAG workloads.
- **TensorRT-LLM** — NVIDIA's engine, best raw throughput on NVIDIA hardware, more complex to operate. Often the answer for large-scale production on H100/H200.
- **llama.cpp / Ollama** — CPU + consumer GPU, great for local deployment, worse at multi-user throughput.
- **TGI (Text Generation Inference)** — Hugging Face's engine, good HF ecosystem integration.

## Closing Thoughts

Efficient LLM inference isn't a single algorithm — it's a stack of techniques, each attacking a specific bottleneck:

- **KV cache + MQA/GQA** — make the per-token memory work smaller
- **Quantization + FlashAttention** — make the remaining memory movement faster
- **Continuous batching + PagedAttention** — keep the GPU saturated
- **Prefix caching** — don't pay twice for the same prefix
- **Speculative decoding** — break the one-token-per-step barrier
- **Parallelism + disaggregation** — scale out when one GPU isn't enough

The techniques compose. The best deployments layer all of them. And the performance gap between a naive implementation and a well-tuned one is often **two orders of magnitude** — the difference between "this is too expensive to ship" and "this is the cheapest component of our system."

Start with a good engine, measure what your workload actually does, and optimize against the bottleneck that's actually hurting you — not the one a paper says is interesting.

## Further Reading

Deep dives on specific topics:
- [KV Cache in LLMs: A Complete Guide](/blog/machine-learning/large-language-model/kv-cache)
- [Quantization in LLMs](/blog/machine-learning/large-language-model/quantization-in-llm)
- [Speculative Decoding: A Complete Guide](/blog/machine-learning/large-language-model/speculative-decoding)
- [vLLM Inference Guide](/blog/machine-learning/large-language-model/vllm-inference)
- [SGLang Inference Guide](/blog/machine-learning/large-language-model/sglang-inference)

Foundational papers:
- *FlashAttention* (Dao et al., 2022) and *FlashAttention-2/3* follow-ups
- *Efficient Memory Management for Large Language Model Serving with PagedAttention* (Kwon et al., 2023) — vLLM
- *Orca: A Distributed Serving System for Transformer-Based Generative Models* (Yu et al., 2022) — continuous batching
- *Fast Inference from Transformers via Speculative Decoding* (Leviathan et al., 2023)
- *EAGLE / Medusa / Lookahead Decoding* — speculative decoding variants
- *SGLang: Efficient Execution of Structured Language Model Programs* (Zheng et al., 2023) — RadixAttention
