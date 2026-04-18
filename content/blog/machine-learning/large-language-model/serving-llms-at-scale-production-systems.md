---
title: "Serving LLMs at Scale: Production Systems Beyond the Model"
publishDate: "2026-04-17"
category: "machine-learning"
subcategory: "Large Language Model"
tags:
  [
    "llm",
    "serving",
    "production",
    "deployment",
    "scheduling",
    "autoscaling",
    "caching",
    "observability",
    "cost",
    "infrastructure",
  ]
date: "2026-04-17"
author: "Hiep Tran"
featured: false
aiGenerated: true
excerpt: "Inference optimization makes the model fast. Serving is what turns that fast model into a product that stays fast under real traffic. This article walks through the production systems problems you only hit at scale — request scheduling, prefill–decode disaggregation, multi-LoRA serving, multi-layer caching, autoscaling LLMs (which is genuinely harder than it looks), SLO-driven observability, and the cost model that decides what's actually worth doing."
---

## Why Serving Is a Different Problem

The [previous article](/blog/machine-learning/large-language-model/optimizing-llm-inference-complete-guide) was about making a single model fast. This one is about keeping thousands of requests fast, all the time, while not going bankrupt.

The difference matters. A benchmark shows "3000 tokens/s on an H100." A product ships "p95 TTFT under 800ms for 50 concurrent users on a $40/hr instance." Those are different problems, and going from the first to the second is where most of the real engineering lives.

Serving an LLM at scale means solving:

- **Scheduling:** which request runs on which GPU, when, in what batch.
- **Routing:** which replica, which region, which model variant.
- **Caching:** at how many layers, with what invalidation rules.
- **Autoscaling:** how to add and remove GPU capacity without melting latency.
- **Observability:** knowing what's broken before users do.
- **Cost:** the $/million-tokens curve, which drives every product decision.

Every one of these has at least one non-obvious failure mode.

## Part 1: The Serving Mental Model

### The Request's Life

A single request flows through many systems before the model ever sees it:

```
User ──▶ Edge / Auth ──▶ API gateway ──▶ Router
                                            │
                                            ▼
                                     ┌─────────────┐
                                     │  Scheduler  │
                                     └──────┬──────┘
                                            │
                              ┌─────────────┼─────────────┐
                              ▼             ▼             ▼
                          Replica A     Replica B     Replica C
                           (GPU)         (GPU)         (GPU)
                              │
                              ▼
                     KV cache / prompt cache / batching
                              │
                              ▼
                        stream tokens back to user
```

Each hop can add latency, fail independently, and must be observable.

### The Metrics That Define the Service

| Metric | What it means | Who cares |
| --- | --- | --- |
| TTFT p50 / p95 | time to first token | user perception |
| TPOT p50 / p95 | time per output token after first | stream smoothness |
| End-to-end latency | full request duration | SLA |
| Throughput (tokens/s) | total output volume | capacity planning |
| GPU utilization | fraction of GPU actually computing | cost efficiency |
| Concurrency | in-flight requests | scaling decisions |
| $/M tokens | cost per million tokens served | CFO |

A serving system is "good" when it has a **defended SLO on a latency metric** (e.g., "TTFT p95 < 1s") and a **target $/M tokens** below some number. Every lever in this article is a way to trade one for the other.

## Part 2: The Scheduler — The Heart of Modern LLM Serving

A modern inference engine is mostly a scheduler with a forward pass attached. Understanding what it's doing is the single most important thing for running one well.

### What the Scheduler Does, Every Step

On every forward pass (every few tens of milliseconds), the scheduler answers:

1. **Which active sequences stay in the batch this step?**
2. **Which new requests, if any, get admitted?**
3. **Which sequences need more prefill chunks, and how big?**
4. **Is anything over memory budget? Who gets evicted?**
5. **Which sequences should be preempted (swapped to CPU or dropped)?**

```
for each step:
    select batch = decode_ready + prefill_ready + newly_admitted
    enforce memory budget (KV cache + activations)
    run one forward pass
    stream any new tokens to clients
    retire finished sequences
    repeat
```

### The Three Scheduling Objectives

A scheduler has to balance three things that pull in different directions:

- **Fairness.** No request should starve.
- **Throughput.** Keep the batch full so the GPU isn't wasted.
- **Latency SLO.** Meet TTFT and TPOT targets per request.

Every real scheduler has knobs for each, and the defaults are wrong for most workloads.

### Policies You'll Actually Configure

- **Max batch size.** Bigger = more throughput, but higher TPOT. Tune against your SLO.
- **Max tokens per step (prefill budget).** Caps how much prefill work co-runs with decode. Lower = better TPOT, worse TTFT.
- **Preemption policy.** When KV cache fills up, does the scheduler drop the newest requests (good for ongoing SLA), oldest (fair), lowest-priority (SLO-aware)?
- **Priority classes.** Interactive chat vs. background batch. Should never share the same queue without weights.

### The Quiet Killer: Head-of-Line Blocking

A long request in the batch slows every other sequence in that batch because each forward pass has to run to completion for the slowest sequence.

Mitigations:

- **Chunked prefill.** So a 32K-token prompt doesn't block a bunch of chat replies.
- **Length-aware routing.** Route likely-long requests to a separate replica or queue.
- **Priority scheduling.** Short, interactive requests beat long batch jobs to the GPU.

## Part 3: Prefill–Decode Disaggregation

This is the biggest architectural shift in LLM serving in the last two years, and it's worth understanding even if you don't deploy it yet.

### The Core Tension

Prefill is **compute-bound**. Decode is **memory-bound**. The [previous article](/blog/machine-learning/large-language-model/optimizing-llm-inference-complete-guide) established this.

When they run on the same GPU, they compete:

- A large prefill step stalls decode sequences sharing the batch (hurts TPOT).
- A full decode batch means prefill waits for a slot (hurts TTFT).

Continuous batching mitigates this but doesn't eliminate it. The two phases genuinely want different hardware configurations:

| Phase | Likes | Hates |
| --- | --- | --- |
| Prefill | many FLOPS, moderate memory bandwidth | being interrupted |
| Decode | huge memory bandwidth, few FLOPS | being interrupted |

### The Architecture

**Disaggregated serving** runs prefill and decode on different pools of GPUs.

```
User request
    │
    ▼
Router
    │
    ├──▶ PREFILL POOL (compute-heavy GPUs)
    │     - runs prompt through model
    │     - produces KV cache
    │     - emits first token
    │
    │ [transfer KV cache over fast network]
    │
    └──▶ DECODE POOL (memory-bandwidth GPUs)
          - receives KV cache
          - streams tokens
```

### The Benefits

- **Each phase gets the hardware and batching policy it likes.** Prefill can batch aggressively for throughput; decode can keep batches small for latency.
- **TTFT and TPOT decouple.** You can scale them independently.
- **Hardware utilization goes up** — no phase is starved by the other.

### The Cost

- **KV cache transfer.** A long prompt's KV cache is large, and moving it between GPUs is slow unless you have fast interconnect (NVLink, InfiniBand, or specialized topologies).
- **Operational complexity.** Two pools, two scheduling systems, a transfer layer between them.
- **Failure modes.** A failed transfer is a new class of bug.

Disaggregation is usually worth it at scale (high traffic, strict SLOs). It's usually not worth it for small deployments, where running both phases on the same GPU is simpler and fine.

## Part 4: Multi-LoRA and Adapter Serving

Modern deployments rarely serve one model. They serve one **base model** plus many **LoRA adapters** — small parameter patches that specialize the base for different tasks, customers, or languages.

### The Naive Approach That Doesn't Work

Serve each fine-tuned variant as a separate replica. For N adapters, you need N GPUs minimum.

For anything beyond a handful of adapters, this is ruinous.

### Multi-LoRA Inference

Keep the base model loaded once. Keep the much smaller LoRA weights (typically <1% of base model size) in a pool in GPU memory. At request time, the scheduler routes each sequence with a tag saying "use adapter X."

The forward pass computes `Wx + (AB)x` where `A, B` are adapter matrices — looked up per-sequence. Frameworks like SLoRA, Punica, and vLLM's LoRA support do exactly this.

### The Scheduling Wrinkle

Within a single batch, different sequences may use different adapters. The kernel has to handle this — a trick called **heterogeneous LoRA** in the literature. Modern engines support it, but it adds throughput overhead (usually 5–15%).

### Capacity Planning

- Base model: 1x memory cost.
- Each LoRA: tiny (~MB), cheap to keep in memory.
- In-flight adapter cache: how many can you keep hot vs. load on demand?

You can realistically serve hundreds to thousands of adapters from a single deployment if you get the caching right — making per-customer fine-tuning economically feasible.

## Part 5: Caching, Done Properly

"Use a cache" is one of those recommendations that hides a dozen decisions. At serving scale, you typically want **three distinct caches**, each solving a different problem.

### Cache Layer 1: Prefix (KV) Cache

Covered in the previous article. Reuses prefill computation for shared prompt prefixes — system prompts, few-shot examples, shared documents.

- **Hit rate is often 60–95%** for agent workloads with long system prompts.
- **Savings are real:** TTFT drops 5–10×, compute cost drops proportionally.
- **Default policy:** PagedAttention + block-hash matching + LRU eviction.

### Cache Layer 2: Full-Response Cache

For identical requests, return the cached full response. Classic HTTP cache, except the "key" is a hash of (prompt, model, sampling config).

- **Useful for:** deterministic requests, chatbot FAQs, common code-completion queries, cached tool-use results.
- **Hit rate varies wildly:** 0% for creative chat, 30–70% for search/RAG, 90%+ for specific workloads.
- **Gotchas:**
  - Sampling temperature > 0 breaks cacheability unless you cache per-sample.
  - Tools with side effects must never be cached.
  - User-specific personalization breaks cache unless keyed per-user.

### Cache Layer 3: Semantic Cache

Same idea as full-response cache, but the key is a **vector embedding** of the request rather than an exact hash. Similar-but-not-identical requests hit the cache.

- **Useful for:** FAQ-like workloads where users phrase the same question differently.
- **Risk:** semantic collisions. Two requests that embed close can actually want different answers. Production systems usually require a similarity threshold *and* an LLM-based or rule-based sanity check before returning a cached answer.
- **Don't use for:** anything where correctness matters more than latency savings.

### Where Each Cache Belongs

```
  ┌────────────────────┐
  │  Full-response     │   hash(prompt) ──▶ saved completion
  │  cache (Redis/etc) │   checked FIRST; return directly on hit
  └──────────┬─────────┘
             │ miss
             ▼
  ┌────────────────────┐
  │  Semantic cache    │   embed(prompt) ──▶ nearest saved
  │  (optional)        │   completion within threshold
  └──────────┬─────────┘
             │ miss
             ▼
  ┌────────────────────┐
  │  Prefix / KV cache │   token-block-level; always on, on GPU
  │  (inside engine)   │
  └──────────┬─────────┘
             │
             ▼
  Actual forward pass
```

Caches compose. A system that does full-response + prefix caching well will see its compute cost per request drop by an order of magnitude on many workloads.

## Part 6: Routing and Load Balancing

Traditional load balancers (round-robin, least-connections) are wrong for LLMs. Why? Because requests aren't fungible.

### Why Naive Balancing Hurts

A round-robin balancer sends the next request to the next replica without knowing:

- Replica A has a warm prefix cache for "system prompt v3," replica B does not.
- Replica A has its KV cache 95% full; adding a long request will cause eviction.
- Replica A has a long batch job running; TTFT there is 2s, replica B is 200ms.

### Routing Strategies That Actually Work

- **Cache-aware routing.** Hash the request prefix; send it to the replica most likely to have that prefix cached. Dramatic win on prefix-cache hit rate.
- **Load-aware routing.** Prefer replicas with lower pending-token counts (sum of output tokens in flight), not lower connection counts.
- **Length-aware routing.** If you can predict request length (from prompt classifier or history), route long jobs away from interactive replicas.
- **Session affinity.** If a user has a cached KV state on replica A, keep them on A for the next turn. Especially valuable for chat.

Modern serving stacks (vLLM production deployments, SageMaker, Anyscale) have these built in. If you're writing your own, it's worth the effort.

## Part 7: Autoscaling LLMs — Genuinely Harder Than It Looks

Web-service autoscaling is a solved problem. LLM autoscaling is not, and many teams discover this painfully in production.

### Why It's Hard

- **Cold starts are enormous.** Loading a 70B model onto a new GPU takes minutes, not milliseconds. You can't scale to zero and back quickly.
- **Load signals are misleading.** "CPU utilization" is meaningless. "GPU utilization" is subtle — 100% utilization can mean "serving 50 users great" or "serving 1 user and hallucinating busy work."
- **Scaling up takes longer than a traffic spike.** By the time a new replica is ready, the surge is over — or the old replicas are on fire.
- **Downscaling is risky.** Evicting a replica kills in-flight streaming requests and warms up cache misses on the remaining replicas.

### Signals to Scale On

Better than GPU utilization:

- **Queue depth** (requests waiting to be admitted).
- **Pending-tokens** (total in-flight output tokens).
- **TTFT p95** exceeding SLO threshold.
- **Batch admission latency** (how long a new request sits before joining the batch).

A good scaling policy uses a combination: "scale up when queue depth > X *and* TTFT p95 is trending up."

### Patterns That Work

- **Warm pools.** Keep a few spare replicas pre-loaded, idle. Scale in and out of the warm pool is instant; cold scaling only happens at the pool boundary.
- **Predictive scaling.** Time-of-day and day-of-week patterns are strong enough to pre-warm capacity before the spike.
- **Burstable capacity** (spot/preemptible GPUs) for overflow. Accept interruption for non-critical traffic.
- **Graceful draining.** A replica being scaled down stops accepting new requests but finishes in-flight ones. Streaming users don't get cut off.
- **Per-model autoscaling.** If you serve many models, scale each independently — their traffic patterns diverge.

### The Hard Truth

At LLM scale, some over-provisioning is non-negotiable. The question is how much. A well-instrumented system can usually run at 60–75% average utilization without SLO pain; anything higher is asking for it.

## Part 8: Observability — Knowing What's Broken Before Users Do

LLM serving observability is **not the same** as generic microservice observability. The failure modes are different.

### What to Log Per Request

Minimum:

- Request ID, user ID, model, adapter tag.
- Prompt length (tokens), output length (tokens).
- TTFT, TPOT, end-to-end latency.
- Prefix cache hit? Full-response cache hit?
- Which replica served it?
- Sampling params.
- Stop reason (length, EOS, user disconnect, error).

### Metrics to Dashboard

- TTFT p50 / p95 / p99 by model, region, user tier.
- TPOT distribution.
- Batch size distribution (are you batching as well as you think?).
- KV cache utilization per replica (how close to OOM?).
- Prefix cache hit rate over time (a drop often means your system prompt changed and you don't know).
- Queue depth.
- Cost per thousand tokens (live).

### The Silent Failures You Have to Instrument

These don't throw errors but kill your product:

- **Token quality regressions** after a model or config change. Instrument periodic evals on a held-out set; alert on drift.
- **Prompt cache hit-rate collapse** after someone edits a shared system prompt.
- **Slow tokens**: a model that responds normally but whose TPOT quietly doubled. Happens with GPU thermal throttling, driver issues, noisy neighbors.
- **Long-tail request storms.** A handful of 100K-token prompts can wreck a whole replica; alert on individual request durations past a threshold.
- **KV cache pressure.** If the scheduler is constantly preempting sequences, you're over capacity; alert on preemption rate > 0.

### Tracing Beats Logs

For complex request paths (router → scheduler → prefill → decode), you want distributed traces, not just logs. OpenTelemetry with LLM-specific span attributes (tokens in, tokens out, batch size at admission time) is the default now.

## Part 9: Hardware and Deployment Choices

### The GPU Ladder

| GPU | Memory | Memory BW | Good for |
| --- | --- | --- | --- |
| A10 / A100 40GB | 24–40 GB | 600–1555 GB/s | small-to-mid models, cost-sensitive |
| A100 80GB | 80 GB | 2 TB/s | mid-size models, general workhorse |
| H100 80GB | 80 GB | 3.3 TB/s | 70B class, FP8 inference, default serious serving |
| H200 | 141 GB | 4.8 TB/s | larger models single-GPU, long context |
| B200 (Blackwell) | 192 GB | 8 TB/s | next-gen, big memory, biggest models |

Memory bandwidth is more important than raw FLOPS for decode. An H200's extra bandwidth over H100 is more meaningful for decode-heavy workloads than the compute bump alone.

### When Alternative Hardware Makes Sense

- **TPUs (v5p/v6e).** Excellent for very large models with good software support; weaker ecosystem than NVIDIA for bespoke optimizations.
- **AMD MI300X.** More memory per chip than H100; ecosystem catching up.
- **Inferentia / Trainium.** Cheap for specific workloads; requires porting.
- **CPU inference.** Viable for very small models, embeddings, or quantized mid-size models serving low QPS. Not generally competitive above ~7B.

### Single-GPU vs. Multi-GPU

Rule of thumb:

- Model fits on one GPU? Run replicas, not TP.
- Model needs 2–8 GPUs on a node? TP within the node.
- Larger than one node? TP intra-node + PP or sharding across nodes.
- MoE? Expert parallelism plus one of the above.

More parallelism is not more better. Every additional axis of parallelism adds communication overhead and operational complexity.

## Part 10: The Cost Model and How to Actually Reduce It

Under every serving decision is $/M tokens. Here's the decomposition:

```
cost per token ≈ (GPU cost per hour) / (tokens per hour per GPU)
```

The denominator is what optimization is for. Every technique from the previous article and this one either increases tokens-per-hour or reduces GPU-hours needed.

### The Cost Levers, Ordered by Effort-to-Impact

**Low effort, high impact:**

- Turn on prefix caching. Immediate 2–5× on TTFT and compute for shared-prompt workloads.
- Continuous batching — make sure your engine is using it.
- Use FP8/INT4 quantization where quality allows.
- Cache-aware routing.

**Medium effort, high impact:**

- Speculative decoding for decode-heavy workloads.
- KV cache quantization for long-context workloads.
- Multi-LoRA consolidation.
- Full-response cache layer in front of the model.

**High effort, situational impact:**

- Prefill–decode disaggregation.
- Custom routing logic tuned to your traffic.
- Running on multiple hardware types (spot fallback, cheaper GPUs for overflow).
- Building a semantic cache with quality guards.

### What to Avoid

- **Chasing GPU utilization as the only metric.** It can be 100% while costing you more per token than a 60%-utilized cache-hit-heavy setup.
- **Building everything yourself.** vLLM, SGLang, TGI, TensorRT-LLM, and their managed equivalents exist. Building your own scheduler is a multi-engineer-year project you don't need.
- **Running a frontier model for a task that a small fine-tuned one could handle.** Model routing — a classifier that picks the cheapest capable model per request — is often a 10× cost improvement.

### A Representative Cost Breakdown

For a typical production chat deployment (rough numbers, vary by a lot):

| Technique | Cost reduction vs. naive |
| --- | --- |
| Continuous batching | 3–5× |
| + Prefix caching (long system prompt) | additional 2–3× |
| + FP8 quantization | additional 1.5–2× |
| + Speculative decoding (if applicable) | additional 1.5–2× |
| + Model routing to smaller model for easy queries | additional 2–5× |

Stacked, this is the difference between $30/M tokens and under $1/M tokens for the same model on the same hardware. None of it is magic — each piece is one of the techniques discussed.

## Part 11: Putting It All Together

A defensible production LLM serving stack, in one diagram:

```
                     ┌────────────────────────┐
                     │  Client / SDK / App    │
                     └───────────┬────────────┘
                                 │
                      ┌──────────▼──────────┐
                      │ API gateway + Auth  │
                      └──────────┬──────────┘
                                 │
                      ┌──────────▼──────────┐
                      │  Full-response &    │  ◀── optional
                      │  semantic caches    │
                      └──────────┬──────────┘
                                 │ miss
                      ┌──────────▼──────────┐
                      │  Model router       │  ◀── picks model size
                      │  (small vs big)     │
                      └──────────┬──────────┘
                                 │
                      ┌──────────▼──────────┐
                      │  Cache-aware LB     │  ◀── routes on prefix hash
                      └──────────┬──────────┘
                                 │
            ┌────────────────────┼────────────────────┐
            ▼                    ▼                    ▼
       Replica pool 1      Replica pool 2      Overflow (spot)
    (in-engine scheduler, PagedAttention, continuous batching,
     prompt cache, optional disaggregated prefill/decode)
            │
            ▼
      Streaming tokens back up through the chain.

  All layers: metrics + traces + evals + SLO alerts.
```

Almost every production LLM system has a version of this. The pieces are modular; you add them as your scale demands.

## Closing: What Changes With Scale

Scale changes which problems matter:

- At **10 requests per minute**, any of the major inference engines with default settings is fine. Don't overthink it.
- At **100 QPS**, scheduling, batching, and prefix caching become the dominant levers. SLO tracking becomes necessary.
- At **1000 QPS**, cache-aware routing, autoscaling, and per-model cost tracking become non-negotiable. Disaggregation starts to pay.
- At **10,000+ QPS**, multi-region deployments, predictive scaling, deep observability, and custom scheduling logic tuned to your specific traffic patterns are where the next wins are.

Every band above introduces concerns that didn't exist in the one below. Most teams build for the band they're in, plus a little headroom — not the band they imagine being in two years.

The optimizations from the [previous article](/blog/machine-learning/large-language-model/optimizing-llm-inference-complete-guide) make the model fast. The systems in this article make that fast model into a product that stays fast.

---

**Related reading**

- [Optimizing LLM Inference: A Complete, Detailed Guide](/blog/machine-learning/large-language-model/optimizing-llm-inference-complete-guide) — the model-side half of this picture.
- [KV Cache](/blog/machine-learning/large-language-model/kv-cache) — deep dive on the data structure every serving decision revolves around.
- [Quantization in LLMs](/blog/machine-learning/large-language-model/quantization-in-llm) — the memory-reduction half of cost optimization.
- [Scaling Managed Agents](/blog/machine-learning/ai-agent/scaling-managed-agents-decoupling-brain-from-hands) — what these production systems look like once you add durable sessions and long-horizon agents on top.
