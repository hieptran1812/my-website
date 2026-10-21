---
title: "Inside DeepInfer: How DeepSeek Serves V4.1-Flash in Production"
date: "2026-10-21"
publishDate: "2026-10-21"
description: "A guided tour of DeepInfer, the engine behind DeepSeek's API: the model it was built for, the 890-byte KV cache, two-stage sparse attention, bounded replay, confidence-scheduled speculative decoding, zero-overhead scheduling and expert load balancing."
tags: ["deepinfer", "deepseek", "llm-inference", "inference-engine", "speculative-decoding", "mixture-of-experts", "kv-cache", "sparse-attention", "expert-parallelism", "scheduling"]
category: "machine-learning"
subcategory: "Inference Frameworks"
author: "Hiep Tran"
featured: true
readTime: 13
---

Most open-source inference engines are generalists. vLLM and SGLang serve hundreds of architectures, so every optimization has to survive contact with all of them. DeepInfer, released by DeepSeek on 2026-10-20, is the opposite: it is the engine that runs DeepSeek's own production traffic, and it supports exactly one model, DeepSeek-V4.1-Flash. That narrowness is the interesting part. When an engine and a model are designed together, the engine can lean on facts about the model that a general engine cannot assume: that only four layers own a KV cache, that half the network can skip most of the prompt, that a draft head can predict its own acceptance rate.

![The 40-layer V4.1-Flash stack as DeepInfer sees it: four KV-owning layers, a producer/consumer split at layer 20, and a three-layer drafter reading the top of the stack](/imgs/blogs/deepinfer-deepseek-production-inference-engine-1.webp)

The diagram above is the mental model: a 40-layer mixture-of-experts transformer where KV is shared in four groups, the top half reuses the KV of layer 20, and a small drafter rides on layers 37 to 39. Nearly every technique in DeepInfer exploits one of those three facts. Below: what each piece does, why it exists, and where the published numbers need an asterisk.

The README is candid about lineage: DeepInfer "was originally built primarily on vLLM", and the Worker → ModelRunner → CacheEngine shape is recognizably [vLLM's](/blog/machine-learning/inference-frameworks/vllm-deep-dive). But the scheduler and block manager are now C++, paged attention is gone, and input preparation left Python entirely.

## 1. The model it was built for

V4.1-Flash has 40 layers, hidden size 5,120, 64 query heads and a single 512-dimensional KV head (448 latent dims plus 64 RoPE dims, an MLA-style latent cache). Every layer is an MoE block with 384 routed experts, top-6 routing and one shared expert. The routed experts are stored in FP4, everything else in FP8 with 32×32 blocks.

The attention is where it gets unusual. A per-layer `compress_ratios` list decides each layer's attention type:

| Layers | Attention | KV owned by |
| --- | --- | --- |
| 0–1 | 128-token sliding window (SWA) only | itself |
| 2–19 | SWA + 2:1 compressed KV, sparsely selected | layers 2, 8, 14 (groups of six) |
| 20–39 | SWA + uncompressed KV, sparsely selected | layer 20, for all twenty |
| 3 draft layers | SWA only | themselves |

Two more ingredients appear in the code and matter later. **mHC** widens the residual stream into four parallel streams mixed by a Sinkhorn-normalized matrix. **Engram** is a hashed n-gram memory: at layers 1 and 14, 2-to-4-gram hashes index 16-million-entry embedding tables that are sharded across GPUs and fetched with DeepEP. The [V4 architecture post](/blog/machine-learning/frontier-models/deepseek-v4-million-token-context-moe) covers the modeling side. For the engine, three facts matter: KV lives in only four places, layers 20 and up are pure *consumers* of layer 20's KV, and the drafter reads the last three layers.

## 2. A KV cache on a diet

KV cache size decides how many concurrent users fit on a GPU, and V4.1-Flash attacks it from every side.

![Per-token KV bytes drawn to scale: about 35 KB for a V3-style MLA cache versus about 890 B for V4.1-Flash, plus a fixed 2.83 MB sliding-window ring per sequence](/imgs/blogs/deepinfer-deepseek-production-inference-engine-2.webp)

Notice that the bar that grows with context almost disappears, and what replaces it is a constant. DeepInfer's `cache_engine.py` lays out five tensors per layer, but only the source layers allocate them. The main KV is FP4 (256 bytes of payload plus 32 bytes of per-16 E4M3 scales, 288 B per entry, "like NVFP4 but without the global scale", per the repo's Q&A doc). A 2:1 compressed layer stores one entry per two tokens, so 144 B per token. The sparse indexer keys add 68 B per entry. Summing four sources and the indexer caches gives roughly:

$$
\underbrace{3 \times 144 + 288}_{\text{main KV}} + \underbrace{3 \times 34 + 68}_{\text{indexer K}} \approx 890 \text{ bytes per token}
$$

For contrast, a DeepSeek-V3-style cache (576 bytes per token per layer in FP8 across 61 layers) is about 35 KB per token. That comparison is my arithmetic, not a DeepSeek claim, but roughly 40× is the headline.

The sliding window does not grow with context, so it lives outside the paged pool as a per-sequence **state slot**: a ring of 134 entries (128-token window plus 6 slots of rollback room for rejected drafts) at 528 bytes each (MXFP8). Across 40 layers that is 2.83 MB per sequence regardless of whether the prompt is 1K or 1M tokens. Only the producer half (layers 0–19) is ever persisted: the Q&A gives the on-disk state as exactly 20 × 134 × 528 = 1,415,040 bytes.

## 3. Two-stage sparse attention

A small cache is useless if decoding still reads all of it. V4.1-Flash never does.

![A log-scale funnel: up to 1,048,576 context tokens, 16,384 candidates kept by layer 20, 512 entries selected per layer, plus the 128-token window](/imgs/blogs/deepinfer-deepseek-production-inference-engine-3.webp)

The funnel has two stages. First, layer 20 runs a lightweight indexer (32 heads of 128 dims, scores computed as `relu(q·k)·w` in FP4) over every KV entry, max-pools the scores into 8-token blocks, and keeps the top 2,048 blocks: 16,384 candidate tokens. Second, every later indexer searches *only inside* that candidate set and keeps its own top 512. Attention then reads those 512 entries plus the 128-token sliding window, 640 entries in total, at any context length.

Top-k over hundreds of thousands of scores per query is a real kernel, so it goes to DeepSeek's DeepSelect library. Logical indices become physical slots through block tables, and FlashMLA's decode kernel takes two index lists at once (SWA slots and compressed-cache slots) with `is_fp8_kvcache=True`. One comment in `index_attn.py` caps the prefill indexer chunk at 8,192 tokens "because mqa downclocks when the chunk size is large": the kind of fact you only learn by profiling in production.

## 4. YOCO bounded replay: skip most of the prompt in half the network

Layers 20 to 39 do not project their own main KV; they read layer 20's. That is the YOCO ("you only cache once") pattern. So during prefill, what do layers 20 to 39 actually need to compute? Their main KV comes for free from layer 20's hidden state. The only thing they still own is their 128-token sliding-window cache, and the only queries whose outputs matter for the next token are the ones at the end.

![A layer-by-token grid for an 8,192-token prompt: layers 0–19 compute every token, layers 20–39 compute only the final 140 tokens](/imgs/blogs/deepinfer-deepseek-production-inference-engine-4.webp)

DeepInfer therefore runs the consumer half on a **bounded suffix** of the prompt. At the YOCO boundary, `compact_yoco_boundary` gathers only the last 140 query rows. The engine still writes full-length main and indexer KV from layer 20's hidden state, then runs consumer attention for the compacted rows only. The 140 is not arbitrary: it must cover the 128-token window plus the 6-token rollback budget, and `environ.py` asserts that constraint. The Q&A states the result plainly: this nearly halves total prefill computation, which tracks with the grid above, since half the layers now touch 140 tokens instead of 8,192.

The cost is approximation. In replay, query *i* only sees keys from position `max(s, i − 127)`, so consumer-layer SWA state is not bit-identical to a full prefill. DeepSeek accepts that because the result "is used for decode only and never enters the prefix cache". Two modes exist (`mask` narrows visibility, `compact` drops the rows), and `examples/yoco/` verifies they agree numerically. Enabling it is two environment variables:

```bash
export DEEPINFER_ACCELERATOR=nvidia
export DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=compact
export DEEPINFER_YOCO_PREFILL_KEEP_TOKENS=140      # >= 128 window + 6 rollback slots
export DEEPINFER_ENABLE_MTP_VARLEN=1               # confidence-scheduled verification
export DEEPINFER_USE_NEXTN_DECODE=5                # max draft length
python examples/offline_inference.py --ep 8
```

## 5. DSpark drafting and confidence-scheduled verification

DeepInfer's speculative decoding has two halves: a drafter that is unusually cheap, and a verifier that refuses to verify drafts that are not worth it. If the accept/reject math is new to you, the [rejection sampling post](/blog/machine-learning/speculative-decoding/speculative-decoding-token-acceptance-rejection-sampling) is the prerequisite.

**The drafter (DSpark, in `models/mtp/dflash.py`)** produces all five drafts in a *single* forward pass of three small layers. The input block is `[real token, noise, noise, noise, noise]`, and the drafter fills in the noise positions in parallel, conditioned on keys projected from the concatenated hidden states of target layers 37, 38 and 39. Parallel drafting loses the left-to-right dependency between drafts, so a rank-256 **Markov head** adds an embedding of the previous draft token to each position's logits: cheap, sequential glue. A **confidence head** then predicts, for each draft, the probability it will be accepted.

**The verifier** uses those predictions. Verifying a draft costs a query row in the next forward pass. At low load that row is almost free, but at high load every extra row makes the CUDA graph bigger and the step slower. `csrc/engine/mtp_varlen.hpp` turns this into an optimization: pool every draft in the batch, sort by predicted acceptance, and keep adding drafts while expected tokens per millisecond still rises.

![Expected tokens per millisecond against average drafts verified per sequence: the curve peaks before five, and measured gains rise with load from −1.5% at 16 sequences to +11% at 96](/imgs/blogs/deepinfer-deepseek-production-inference-engine-5.webp)

The peak sits left of "verify all five" because the last drafts in a block are the least likely to be accepted, yet each one costs the same row. Here is the scheduler distilled to Python. The real version adds an online slope estimator and per-batch-size sampler timings:

```python
from bisect import bisect_left

def schedule_drafts(rates_per_seq, perf_curve, slope_ms=0.0, sampler_ms=0.3):
    """rates_per_seq: per sequence, predicted P(accept) for each draft position.
    perf_curve: sorted (cuda_graph_size, forward_ms) pairs profiled at startup.
    Returns query rows per sequence (1 golden token + chosen drafts)."""
    num_seqs = len(rates_per_seq)
    pool = sorted(((r, i) for i, rates in enumerate(rates_per_seq) for r in rates),
                  key=lambda p: (-p[0], p[1]))
    sizes = [g for g, _ in perf_curve]

    def tokens_per_ms(total_q, expected):
        idx = bisect_left(sizes, total_q)          # smallest graph that fits
        if idx == len(sizes):
            return -1.0
        graph, base_ms = perf_curve[idx]
        return expected / (base_ms + slope_ms * graph + sampler_ms)

    expected, best_k = float(num_seqs), 0
    best = tokens_per_ms(num_seqs, expected)       # golden tokens only
    for k, (rate, _) in enumerate(pool, start=1):
        expected += rate
        tput = tokens_per_ms(num_seqs + k, expected)
        if tput > best:
            best, best_k = tput, k

    rows = [1] * num_seqs
    for _, seq in pool[:best_k]:                   # highest-confidence drafts win
        rows[seq] += 1
    return rows
```

Two details are worth stealing. The `perf_curve` is measured per CUDA-graph size at startup (graphs exist at sizes 1, 2, 4, then every 8), so the cost model is the real hardware, not a formula. And the C++ version uses acceptance rates from round N−2 to size the batch but rates from N−1 to assign rows, so that one draft's confidence cannot influence whether an *earlier* draft gets verified.

The measured payoff, from DeepSeek's Q&A: production drafts five tokens, dynamic verification checks 3.2 to 4.4 of them, accepted length barely moves (3.29 to 3.20), and throughput changes by −1.5% at 16 sequences per rank, +5.4% at 60 and +11.0% at 96. At light load it is a slight loss; under real load it is free throughput.

## 6. Zero-overhead scheduling

Every engine has a CPU loop: read sampled tokens, update sequences, decide the next batch, build input tensors. In a naive loop the GPU waits for all of that between steps.

![CPU and GPU lanes on one time axis: the naive loop leaves a GPU idle gap every step, while ZOS overlaps post-processing and scheduling with the running forward pass](/imgs/blogs/deepinfer-deepseek-production-inference-engine-6.webp)

DeepInfer's zero-overhead scheduler (ZOS) reorders the worker loop in `worker.py`: launch forward(N), *then* post-process step N−1 while N runs, then launch sampling for N. The catch is that step N+1's batch is built before step N's tokens are known, so the CPU cannot know how many drafts each sequence had accepted. DeepInfer resolves this on the GPU: a small kernel uploads a row remap and fixes sequence lengths from the previous step's `num_accepted_tokens_gpu`, and input ids are gathered on device. The confidence scheduler above splits the same way: the CPU fixes only the *total* query count, and a TileLang kernel recomputes per-sequence lengths on the GPU.

The guardrail I like most is `forbid_cuda_sync()`, which wraps forward and sampling and raises if anything triggers a hidden CPU-GPU synchronization. One innocent `.item()` would silently serialize the pipeline; here it fails loudly. The C++ scheduler also never mixes prefill and decode in one step: data-parallel ranks vote for a prefill round, by default every 70 decode steps.

## 7. Wide EP, EPLB and Mega-MoE

Production decode runs at EP64 to EP128, meaning each GPU holds a handful of experts and tokens travel across the NVLink domain to reach them. The [expert-parallel load imbalance post](/blog/machine-learning/inference-engineering/moe-inference-routing-expert-parallel-and-the-load-imbalance-problem) explains why that is fragile: a step takes as long as its busiest GPU.

![Per-GPU MoE load for a slice of EP64 before and after EPLB: a skewed load with max/mean near 1.6 flattens to the measured 1.09](/imgs/blogs/deepinfer-deepseek-production-inference-engine-7.webp)

EPLB's policy in `eplb/policy.py` is two greedy passes. First, choose replicas: for each redundant slot, copy the logical expert whose duplication most reduces the per-step maximum load, summed over a history window. Second, place them: sort replicas by load and put each on the rank that minimizes the summed per-step maximum. At EP64, 384 logical experts round up to 448 physical slots (7 per GPU, 64 replicas). The measured imbalance ratio (max over mean) is 1.090 for decode and 1.045–1.074 for prefill.

The MoE layer itself is one kernel when the EP group fits in an NVLink domain: DeepGEMM's `fp8_fp4_mega_moe` fuses dispatch, both expert GEMMs, SwiGLU and combine through a symmetric buffer, falling back to DeepEP dispatch/combine otherwise. The [open-infra post](/blog/machine-learning/frontier-models/deepseek-open-infra-deepep-deepgemm-flashmla) covers those libraries.

The repo also implements **balanced context parallelism** as a zigzag split: each sequence is cut into 2×cp chunks and rank *r* owns chunks *r* and 2cp−1−*r*, so every rank gets one cheap early chunk and one expensive late chunk under causal attention. With cp = 4 the ranks hold {0,7}, {1,6}, {2,5}, {3,4}. The Q&A notes that production does not use CP or TP at all, which tells you how far wide EP plus data parallelism carries this model.

## 8. Determinism and structured outputs

Two features look minor and are not. **Batch-invariant inference** (`DEEPINFER_BATCH_INVARIANT=1`) keeps a sequence's logits identical regardless of what else is in the batch. The repo does this by flipping deterministic modes in DeepGEMM, TileKernels and FlashMLA, and by forbidding DeepEP from reducing expert replicas in arbitrary order. The kernel internals live in those libraries, not here. **Deterministic sampling** derives each random draw from SplitMix64 over the request's `seq_seed` and token position, then inverts an fp64 CDF, so the result is independent of batch size and row order. Full determinism also needs `DEEPINFER_ENABLE_MTP_VARLEN=0`, since confidence scheduling changes which drafts get verified.

| Feature | Mechanism | Cost |
| --- | --- | --- |
| Batch invariance | deterministic reduction modes in every kernel library | NVIDIA only, some fused kernels disabled |
| Deterministic sampling | SplitMix64(seq_seed, position) + fp64 inverse CDF | requires fixed-length drafts |
| Structured output | xgrammar bitmask per draft step, matcher rollback of up to 7 tokens | grammar compile on a 4-thread pool, 20 s timeout |

Structured output uses xgrammar for regex, JSON schema and EBNF. The speculative twist is rollback: the grammar matcher advances through draft tokens and rewinds on rejection. A `trigger_token_ids` option enables the grammar only after a trigger sequence (KMP-matched), which is what you want for a JSON tool call embedded in free text.

## 9. What is not in the box, and reading the benchmarks honestly

The open-source release is the *engine*, not the *service*. Missing are the KV cache system (DeepSeek's runs on 3FS with 10+2 erasure coding and roughly 100 PiB), prefill/decode/vision-encoder disaggregation, and the data-parallel load balancers. The hooks are visible: prompt-only and completion-only engine types, KV save stubs, and the on-disk state region from section 2.

The README's charts are real, but read them carefully:

| Chart | Setup | Read-off |
| --- | --- | --- |
| Prefill | EP4 DP4, 4 GPUs, 131K context, 140-token replay | 4,096 tok/rank: ~60K tok/s/GPU; 65,536 tok/rank: ~89.5K |
| Decode | EP64 DP64, 64 GPUs | 64 req/rank: ~246 tok/s/user at ~15.7K tok/s/GPU |
| Wide EP | EP64 vs EP8 at 64 req/rank | EP64 ~24% ahead on *both* axes |

The asterisk is in `plot_pareto.py`: decode throughput is measured steps per second multiplied by an **assumed** 75% per-draft acceptance (three drafts, τ = 3.25), and the script refuses data collected with confidence scheduling on. The decode curve is a model of production, not a measurement of it. Reasonable for reproducibility, but not the same claim as "246 tokens per second per user".

## Case studies from the source

**1. The downclocking indexer.** The prefill indexer chunk is capped at 8,192 tokens because the MQA kernel downclocks at larger chunks. No roofline model predicts that. It comes from watching clocks in production, and it is the kind of constant every serving team ends up with.

**2. The missing SWA snapshot.** A prefix-cache hit is only usable if the sliding-window state at that position exists too. The Q&A describes the backoff: hits are counted in 128-token blocks, and if no SWA snapshot covers the hit point, the engine backs off one block and recomputes those 128 tokens, capping any hit at the prompt length minus 134. A hybrid attention stack turns "cache hit" from a boolean into a small negotiation.

**3. Profiling with forced imbalance.** `DEEPINFER_FORCE_IMBALANCE_RATIO` makes one rank receive a chosen multiple of the average MoE load, so the cost of imbalance can be measured on demand instead of on a skewed traffic day.

## When to reach for DeepInfer, and when not to

Reach for it if you serve DeepSeek-V4.1-Flash on SM100 GPUs or Ascend 950 at a scale where wide EP pays off, or if you build inference engines and want production-grade reference implementations of draft-confidence scheduling, bounded replay and zero-sync async scheduling.

Do not reach for it as a general server. It supports one model, it needs CUDA 13 and GCC 13 and DeepSeek's kernel libraries, and without a KV store, disaggregation and load balancers you are running a strong engine with no service around it. The real value of DeepInfer is what it shows: co-designing a model and its engine moves the bottleneck out of attention and into scheduling, where the remaining wins are counted in single-digit percentages and fought for with profiled cost curves.
