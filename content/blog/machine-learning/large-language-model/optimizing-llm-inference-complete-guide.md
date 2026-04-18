---
title: "Optimizing LLM Inference: A Complete, Detailed Guide to Making Models Run Fast"
publishDate: "2026-04-17"
category: "machine-learning"
subcategory: "Large Language Model"
tags:
  [
    "llm",
    "inference",
    "optimization",
    "deployment",
    "kv-cache",
    "quantization",
    "flash-attention",
    "batching",
    "speculative-decoding",
  ]
date: "2026-04-17"
author: "Hiep Tran"
featured: false
aiGenerated: true
excerpt: "Training an LLM is a one-time, well-funded event. Serving it is a forever problem that has to be cheap, fast, and correct under load. This article walks through every major inference optimization — prefill vs decode, KV cache, FlashAttention, PagedAttention, continuous batching, quantization, speculative decoding, and prompt caching — from first principles, with diagrams, arithmetic, and concrete numbers so the tradeoffs become obvious."
---

## Why This Article Exists

Most introductions to LLM inference start with a single sentence — "the model predicts the next token" — and then skip straight to code. That sentence is technically true and practically useless. The reason inference is hard isn't the math. It's the **memory**, the **hardware**, and the fact that the two halves of inference (prefill and decode) have opposite bottlenecks.

This article is a long, deliberate walk through the entire optimization stack for running an LLM fast. We'll build up from *why* naive inference is slow, through every major technique in use today, and end with a clear picture of which techniques fix which bottleneck.

If you've read my earlier posts on the [KV cache](/blog/machine-learning/large-language-model/kv-cache) and [LLM quantization](/blog/machine-learning/large-language-model/quantization-in-llm), this article sits above them — it's the map that shows where each of those techniques fits.

The companion article, [Serving LLMs at Scale](/blog/machine-learning/large-language-model/serving-llms-at-scale-production-systems), takes everything here and adds the system layer — schedulers, routing, autoscaling, cost.

## Part 1: What Actually Happens During Inference

### The Two Phases Nobody Tells You Are Different

Every LLM request has two very different phases:

```
User: "Write me a poem about rain."
       │
       │  [PREFILL PHASE]
       │  Feed the whole prompt into the model at once.
       │  All tokens processed in parallel.
       │  → produces the first output token.
       ▼
"The"     ─┐
            │  [DECODE PHASE]
"rain"      │  Generate one token at a time.
            │  Each token depends on all previous ones.
"falls"     │  Done sequentially.
            │
"softly"   ─┘
  ...
```

Prefill and decode *look* like the same operation (a forward pass through the transformer), but from a hardware perspective they are completely different workloads:

| Property | Prefill | Decode |
| --- | --- | --- |
| Tokens per forward pass | hundreds to thousands | exactly **one** |
| Parallelism | high — all prompt tokens at once | low — sequential |
| Bottleneck | compute (matrix multiplies) | memory bandwidth |
| Time per token | tiny (amortized) | larger (full pass per token) |
| What gets cached | builds the KV cache | reads the KV cache |

This distinction is the foundation of every optimization below. The rest of this article is essentially "what techniques help prefill?" and "what techniques help decode?" and they are mostly different techniques.

### The Metrics That Actually Matter

Three numbers describe inference performance:

- **TTFT — Time To First Token.** How long until the user sees *anything*. Dominated by prefill.
- **TPOT / ITL — Time Per Output Token / Inter-Token Latency.** How fast tokens arrive after the first. Dominated by decode.
- **Throughput.** Total tokens per second across all users. Dominated by batching.

A system can have great throughput and terrible TTFT (cheap but feels slow), or great TTFT and terrible throughput (fast for one user, bankrupting for many). Tuning inference is tuning the balance between these three.

## Part 2: Why Inference Is Memory-Bound (And Why That's Everything)

The most important fact about modern LLM inference is that **decode is bandwidth-limited, not compute-limited**. Once you internalize this, every optimization becomes intuitive.

### The Arithmetic Intensity Argument

GPUs are monsters. An H100 can do ~990 TFLOPS at BF16 — roughly a trillion multiply-adds per second. But it can only *read* memory at ~3.3 TB/s.

This ratio — **compute per byte read** — decides whether your workload is compute-bound or memory-bound.

During decode, to generate a single token, the GPU has to read:

- All the model weights (~140 GB for a 70B model in BF16).
- The entire KV cache for that sequence (grows with context length).

…and performs relatively few matrix multiplies (one token's worth).

The result: the GPU spends most of its time **waiting for memory**, not doing math. Adding more FLOPS would change nothing. The only thing that moves the needle is **moving less data** or **doing more useful work per byte moved**.

That single observation drives:

- KV cache design (smaller cache = less to read)
- Quantization (fewer bytes per weight)
- Batching (reuse the weights you just read for many sequences)
- FlashAttention (fewer memory round-trips)
- Speculative decoding (more tokens per decode pass)

Everything below is a variation on "reduce memory traffic."

## Part 3: The KV Cache — Inference's Most Important Data Structure

I covered this in detail in [a dedicated post on KV cache](/blog/machine-learning/large-language-model/kv-cache). Here's the short version needed to follow the rest of this article.

### Why It Exists

Naively, to generate token N+1, the model would re-compute the attention for all N prior tokens. That's quadratic work per token — unusable.

The fix: during the forward pass, for each layer's attention, we keep the **K and V tensors** for every token we've already seen. When a new token arrives, we only compute its new Q, K, V; then we attend against the cached K/V from all prior tokens. The cost per token becomes linear in sequence length instead of quadratic.

```
Without KV cache:             With KV cache:
gen token 1: O(1)             gen token 1: O(1), cache K,V
gen token 2: O(2)             gen token 2: O(2), append K,V
gen token 3: O(3)             gen token 3: O(3), append K,V
...                           ...
Total for N tokens: O(N²)    Total for N tokens: O(N²) — but wall time is O(N)
                              (we removed redundant recomputation)
```

Wait — the total is still O(N²). What did we actually save?

We saved the *redundant* recomputation. Without the cache, the model recomputes the attention for tokens 1..k at every step k+1, k+2, … When you sum that, it's O(N²) *of recomputation*. With the cache, each token pair is computed exactly once.

### Why It Hurts

The KV cache is huge. For a 70B model at context 8192, the cache is *tens of GB per sequence*. On a single GPU, this means:

- You can only fit a few concurrent requests.
- Long contexts become GPU-expensive even if compute is light.
- Evicting and re-loading the cache is catastrophically slow.

Almost every inference engine built in the last two years is fundamentally about managing this one data structure well.

## Part 4: FlashAttention — Fixing Attention's Memory Problem

Naive attention has an obvious problem: it materializes the full `N × N` attention matrix in GPU memory. For long sequences, this matrix is gigantic — and, worse, the GPU writes it out to high-bandwidth memory (HBM) and then reads it back to apply softmax and multiply by V.

### The Insight

The softmax-over-QKᵀ-times-V operation can be computed **without ever writing the full attention matrix to HBM**. Instead:

- Split Q, K, V into blocks that fit in on-chip SRAM (fast memory, tiny capacity).
- Stream through them, computing partial softmax statistics as you go.
- Use an online softmax formulation to combine partial results correctly.
- Write only the final output to HBM.

```
Naive attention:        FlashAttention:
Q ──▶ HBM               Q ──▶ SRAM ──┐
K ──▶ HBM               K ──▶ SRAM ──┤
      │                 V ──▶ SRAM ──┤──▶ fused
QKᵀ ──▶ HBM (N×N !!)                 │    compute
softmax ──▶ HBM                       │    in SRAM
× V ──▶ HBM                           └──▶ output ──▶ HBM
```

The HBM traffic drops from O(N²) to O(N). The wall-clock speedup is large — 2–4× on long contexts — and crucially, it enables contexts that previously OOM'd.

### Why It Works Everywhere Now

FlashAttention doesn't change the math, so there's no accuracy tradeoff. That's why it's been adopted universally. FlashAttention-2 and 3 further optimize for specific hardware (Hopper, Blackwell) and specific patterns (variable-length sequences, sparse attention).

**Takeaway:** FlashAttention is the default. If an inference engine isn't using it (or an equivalent), it's wrong.

## Part 5: PagedAttention — The Memory Manager for KV Cache

FlashAttention fixes attention *computation*. PagedAttention fixes KV cache *storage*.

### The Problem It Solves

Before PagedAttention, inference engines reserved a contiguous block of KV cache memory per sequence, sized to the max possible context length.

This is wasteful in two specific ways:

1. **Internal fragmentation.** A sequence that generates 200 tokens is given space for, say, 4096. The unused 3896 tokens' worth of memory is locked up until the sequence finishes.
2. **External fragmentation.** When sequences finish at different times, they leave holes of different sizes — like badly fragmented disk space — that new sequences can't use even though total free memory is plenty.

On a single H100, these two effects can mean the GPU is "full" while 40% of HBM is actually wasted. That directly caps your concurrent users.

### The Insight: Pages, Just Like an OS

Operating systems solved this decades ago with virtual memory: divide memory into fixed-size pages, let processes request non-contiguous pages, and use a page table to hide the fragmentation from the program.

PagedAttention applies the exact same idea to the KV cache:

```
Logical view (what the attention kernel sees):
  Sequence A: [tok0, tok1, tok2, ..., tok99]

Physical view (how memory is actually laid out):
  Page table for sequence A:
    block 0 ──▶ GPU page 17
    block 1 ──▶ GPU page 4
    block 2 ──▶ GPU page 42
```

- Each page is a fixed block (say, 16 tokens' worth of KV).
- A sequence owns a list of pages, not a contiguous region.
- When it needs another page, it grabs the next free one.
- When it finishes, pages return to the pool immediately.

The attention kernel is modified to read KV through the page table, with minimal overhead.

### The Payoff

- Near-zero fragmentation. Memory utilization climbs from ~40% to >90% in many workloads.
- **Copy-on-write sharing** between sequences that share a prefix (e.g., same system prompt, or multiple samples from the same prompt). This is how beam search, parallel sampling, and shared prompts get cheap.
- **Prefix caching** (Part 9) is almost free to build on top of it.

vLLM pioneered this; almost every modern engine (TGI, SGLang, TensorRT-LLM) has a paged KV manager now.

## Part 6: Batching — Turning Memory Traffic Into Real Work

Remember: decode is memory-bound. The model weights are already being read from HBM every step. **If you have to read them anyway, you may as well use them for as many sequences as possible.**

That's batching. But the naive form barely works for LLMs, and the non-obvious form — continuous batching — is one of the most impactful inventions in LLM serving.

### Static Batching (Doesn't Work for LLMs)

Classic batching: collect N requests, pad them to the same length, run them together, return all results.

Problems specific to LLMs:

- **Sequence lengths vary wildly.** Padding wastes compute.
- **Completion times vary wildly.** The whole batch waits for the slowest request.
- **New requests wait outside the batch.** If 9 requests finish in 0.5s but one takes 30s, the next arrivals wait 30s before they even start.

In practice, static batching leaves the GPU idle most of the time.

### Dynamic Batching (Slightly Better)

Collect requests for a small time window (say, 50ms), then run them as a batch. Better GPU utilization, worse TTFT for early arrivals.

Still has the problem that all requests must finish together.

### Continuous Batching (The Real Answer)

The key insight: **prefill and decode can be mixed arbitrarily in the same batch.** Each forward pass is free to include:

- Some sequences in prefill (processing their first batch of prompt tokens).
- Some sequences in decode (generating their next token).
- New sequences that just arrived.
- Sequences that are finishing and leaving the batch.

```
t=0: [A:prefill] [B:decode] [C:decode]
t=1: [A:decode]  [B:decode] [C:decode] [D:prefill]   ← D joined mid-batch
t=2: [A:decode]  [B:done — evicted] [C:decode] [D:decode]
t=3: [A:decode]  [C:decode] [D:decode] [E:prefill]   ← E joined
...
```

Every step, the scheduler reshuffles which sequences are in the batch. A sequence that finishes in 5 tokens is out of the batch in 5 steps; one that generates 1000 tokens stays for 1000 steps, sharing batch slots with hundreds of newer arrivals over its lifetime.

Effect on the metrics:

- **Throughput** goes up dramatically, because the GPU is always busy with a full batch.
- **TTFT** stays low, because new requests join on the next step instead of waiting for the current batch to finish.

Continuous batching is table stakes in any modern inference engine. If you're running vanilla HuggingFace `model.generate()` in a loop, you're leaving 5–10× throughput on the floor.

### Chunked Prefill

A refinement on top of continuous batching. Long prompts in prefill hurt the decode sequences sharing the batch (because prefill is compute-heavy and delays the whole step). The fix: **split prefill into chunks**, processing, say, 512 prompt tokens per step. This keeps each step short enough that co-running decode sequences don't notice the long prompt.

## Part 7: Quantization — Smaller Weights, Less Memory Traffic

Covered in depth in [my post on LLM quantization](/blog/machine-learning/large-language-model/quantization-in-llm). The short version:

The model weights are stored in some numeric format — BF16 (2 bytes), FP8 (1 byte), INT4 (0.5 bytes). Quantization converts high-precision weights to lower-precision formats.

Two reasons this helps inference:

1. **Smaller memory footprint.** A 70B BF16 model is ~140 GB. Quantized to INT4, it's ~35 GB — the difference between needing two H100s and one.
2. **Less memory traffic.** Decode reads weights every step. Fewer bytes to read = faster decode directly. This is often a *larger* win than the first.

### The Precision Spectrum

| Format | Bytes/weight | Typical quality loss | Where it's used |
| --- | --- | --- | --- |
| FP32 | 4 | reference | training |
| BF16 / FP16 | 2 | negligible | default inference |
| FP8 | 1 | small | modern H100/B200 inference |
| INT8 | 1 | small | general inference |
| INT4 (AWQ, GPTQ) | 0.5 | small-to-moderate | cost-sensitive inference |
| INT2 / 1-bit | 0.25 / 0.125 | noticeable | research, extreme compression |

### What Gets Quantized

- **Weights only.** Most common. Activations stay in higher precision. Minimal quality loss.
- **Weights + activations.** Higher gains, harder to do without quality loss. Needs calibration data.
- **KV cache quantization.** Often overlooked but hugely valuable: the KV cache is half the memory traffic in long-context decode. Dropping KV cache from FP16 to FP8 or INT8 gives a large free speedup.

### Where Quality Actually Drops

For most chat/assistant workloads, 4-bit weight quantization with a good algorithm (AWQ, GPTQ) is indistinguishable from BF16. Where it does hurt:

- Very long generations (errors compound)
- Code generation (one-wrong-token matters more)
- Reasoning-heavy tasks
- Multilingual or low-resource languages

The right evaluation is *your* task, not a generic benchmark.

## Part 8: Speculative Decoding — More Tokens Per Memory Read

Decode is memory-bound. Every forward pass reads all the model weights to produce *one* token. What if we could produce several tokens per pass?

That's speculative decoding. The idea is simple and the implementation is elegant.

### The Setup

- A **draft model** (small, fast — maybe a 1B model) proposes the next K tokens.
- The **target model** (large, slow — the one users actually want) verifies all K proposals in parallel in a single forward pass.

### How Verification Works

The target model, given the prompt, computes probabilities for the next token. Whether it accepts the draft's first proposal or not is determined probabilistically in a way that **provably produces samples from the target distribution** — i.e., the output is identical in distribution to what the target model would have produced alone.

- If accepted, move to the second proposal, and so on.
- If rejected at position i, use a corrected sample at position i, discard i+1..K, and restart.

```
Draft proposes: [A, B, C, D, E]
Target verifies:
  accept A ✓
  accept B ✓
  reject C ✗ → target samples C' instead
  discard D, E
  produce: [A, B, C']        (3 tokens from 1 target forward pass)
```

On average, if the draft model is reasonably good, you get 2–4× more target tokens per forward pass — which means 2–4× faster decode, for *no loss in output quality*.

### The Variants

- **Draft model.** The classic setup: separate small model. Works well if drafts are cheap.
- **Medusa.** Add multiple prediction heads to the target model so it proposes its own drafts.
- **EAGLE / EAGLE-2.** Use the target model's hidden states as input to a tiny draft module. Very high acceptance rates.
- **n-gram / retrieval-based drafting.** Use literal n-gram lookup from the prompt or a database for drafts. Free, surprisingly effective for repetitive content (code completion).

### When It Doesn't Help

- Very small target models (the draft is a large fraction of the target's cost).
- Low-batch-size regimes only; at huge batch sizes, the target forward pass is already dense, so the "free parallelism" of verification isn't actually free.
- Highly unpredictable outputs (low draft acceptance).

## Part 9: Prompt / Prefix Caching — Don't Redo Work You've Already Done

If many requests share the same prefix — a long system prompt, a few-shot example set, a large document — recomputing the prefill for that prefix on every request is pure waste. Prefix caching makes that work reusable.

### What's Actually Cached

Not the text, not the tokens — the **KV cache entries** for the shared prefix. Once computed, they can be reused across any request whose prompt starts with the same tokens.

```
Request 1: [system prompt (shared)] [user question 1]
Request 2: [system prompt (shared)] [user question 2]
Request 3: [system prompt (shared)] [user question 3]

Without caching: prefill cost × 3
With caching:    prefill cost × 1  + reuse × 2
```

For typical assistant deployments with long system prompts (10K+ tokens is common for tool-using agents), this is a massive win on both TTFT and cost.

### How It's Implemented

Matching a new request's prefix against cached prefixes has to be fast. Most systems:

- Hash token blocks (e.g., 16-token chunks) and keep a table from block-hash → KV-cache page.
- Walk the new request's tokens, matching block-by-block. The longest prefix match wins.
- Reuse matched blocks' KV pages directly; compute the rest normally.

PagedAttention makes this nearly free to build — the pages are already the right abstraction.

### When to Invalidate

- The model changed (different weights, different KV cache).
- The sampling config changes in ways that affect prefill (temperature does not; some system-level tokens might).
- Pressure from the cache eviction policy (usually LRU on pages).

A well-tuned prompt cache can turn a 10K-token system prompt from "adds 200ms to TTFT" into "adds 5ms." That's the difference between a usable agent and a sluggish one.

## Part 10: Parallelism — When One GPU Isn't Enough

Big models don't fit on one GPU. Even when they do, splitting them across multiple GPUs can be faster per request. Two main axes:

### Tensor Parallelism (TP)

Split the matrices *inside* a layer across GPUs. Each GPU owns a slice of every weight matrix and of the KV cache.

```
Layer weight W (huge)
   ├── GPU 0 owns columns 0..k
   ├── GPU 1 owns columns k..2k
   ├── GPU 2 owns columns 2k..3k
   └── GPU 3 owns columns 3k..4k

Forward pass: each GPU does its slice, then all-reduce the results.
```

- **Pros.** Linear memory reduction per GPU; lower latency per request (parallelism inside a layer).
- **Cons.** Needs fast interconnect (NVLink) — across network it's too slow.

Typically TP=2, 4, or 8 within one server.

### Pipeline Parallelism (PP)

Split *which layers* each GPU owns. GPU 0 runs layers 0–15, GPU 1 runs 16–31, etc.

```
GPU 0: layers 0..15
GPU 1: layers 16..31
GPU 2: layers 32..47
GPU 3: layers 48..63
```

- **Pros.** Much lower interconnect requirements — only hidden states cross GPU boundaries. Can scale across nodes.
- **Cons.** Pipeline bubbles (GPUs idle waiting for work). Mostly used in training; inference uses it for very large models.

### Expert Parallelism (EP)

For Mixture-of-Experts models: different experts live on different GPUs. Requests are routed to the GPUs owning their experts. Adds a whole new set of load-balancing concerns.

### Which To Use

- **Fits on one GPU?** Don't shard. Run replicas for throughput instead.
- **Needs 2–8 GPUs on one node?** TP. Simple, fast.
- **Doesn't fit on one node?** TP within node + PP across nodes, or use a framework like Megatron / vLLM's distributed mode.

## Part 11: A Practical Checklist

For most real deployments, stacking these in order gets you most of the win:

1. **Use a real inference engine** (vLLM, TGI, SGLang, TensorRT-LLM). Don't write your own loop.
2. **Make sure FlashAttention + PagedAttention are on.** They almost always are in these engines, but verify.
3. **Use continuous batching.** Again, default in modern engines.
4. **Quantize to FP8 or INT4** if memory/cost matters.
5. **Turn on prompt caching** for shared prefixes.
6. **Enable speculative decoding** if latency (TPOT) matters more than throughput.
7. **Quantize the KV cache too**, especially for long contexts.
8. **Pick the right TP degree.** Not bigger than needed.
9. **Use chunked prefill** if long prompts co-exist with short-latency decode sequences.
10. **Measure TTFT, TPOT, and throughput separately.** A single "latency" number hides which optimization is paying off.

## Part 12: Which Technique Fixes Which Bottleneck

One table to take away from the whole article:

| Optimization | Fixes prefill? | Fixes decode? | Fixes memory? | Fixes concurrency? |
| --- | --- | --- | --- | --- |
| FlashAttention | ✅ | small | ✅ | — |
| PagedAttention | — | small | ✅ | ✅ |
| Continuous batching | ✅ | ✅ | — | ✅ |
| Chunked prefill | ✅ | indirect | — | ✅ |
| Weight quantization | ✅ | ✅ | ✅ | ✅ |
| KV cache quantization | — | ✅ | ✅ | ✅ |
| Speculative decoding | — | ✅ | — | — |
| Prompt / prefix caching | ✅ | — | — | ✅ |
| Tensor parallelism | ✅ | ✅ | ✅ | — |

If you understand every row of this table and which column your current bottleneck is in, you know what to try next.

## Closing

The honest summary of LLM inference optimization is: **the model wants to read less memory per useful output token.** Almost everything above is a reformulation of that single goal.

- FlashAttention: read less memory per attention op.
- PagedAttention: waste less of the memory you have.
- Continuous batching: get more output per memory read.
- Quantization: make each byte read contain more model.
- Speculative decoding: get more output tokens per memory-read pass.
- Prompt caching: don't read memory for work you've already done.
- Parallelism: split the memory bandwidth problem across more GPUs.

Once you see inference this way, picking the right optimization for your workload stops being guesswork.

The next article, [Serving LLMs at Scale](/blog/machine-learning/large-language-model/serving-llms-at-scale-production-systems), takes this foundation and builds the production system on top — schedulers, routing, autoscaling, observability, and the cost model that decides which of these techniques actually matter for your traffic.

---

**Related reading**

- [KV Cache: The Data Structure That Makes LLM Inference Tractable](/blog/machine-learning/large-language-model/kv-cache)
- [Quantization in LLMs: Making Big Models Fit Small Hardware](/blog/machine-learning/large-language-model/quantization-in-llm)
- [Serving LLMs at Scale](/blog/machine-learning/large-language-model/serving-llms-at-scale-production-systems)
