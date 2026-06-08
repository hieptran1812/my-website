---
title: "Mooncake: A KVCache-Centric Disaggregated Architecture for LLM Serving"
publishDate: "2026-06-05"
date: "2026-06-05"
category: "paper-reading"
subcategory: "Large Language Model"
tags:
  - llm-serving
  - kv-cache
  - prefill-decode-disaggregation
  - inference-optimization
  - distributed-systems
  - scheduling
  - long-context
description: "How Kimi's production serving stack pools idle DRAM, SSD, and RDMA into one KVCache tier, splits prefill from decode, and schedules every request around where its cache lives — for up to 525% more long-context throughput under SLO."
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/mooncake-1.png"
readTime: 31
---

If you run a large language model service at any real scale, the bill is not dominated by the model weights you bought once. It is dominated by the GPU-seconds you burn re-reading the same prefixes over and over, and by the awkward fact that one GPU cluster has to do two completely different jobs at once. The first job, **prefill**, ingests a prompt and is compute-bound: throw 64k tokens at it and the work scales super-linearly with input length. The second job, **decode**, emits tokens one at a time and is memory-bound: it lives or dies on how many sequences you can keep batched in VRAM and how fast you can stream their key-value cache. These two jobs want opposite hardware, opposite parallelism, and they answer to opposite service-level objectives. Couple them in one cluster and you are perpetually mis-provisioned for one of them.

Mooncake is the serving platform that Moonshot AI built to run **Kimi**, their long-context LLM product, and it is an unusually honest systems paper because it was written under genuine duress. The authors are explicit that they confront "severe overload problems, especially during peak times," driven by rapid user growth against a constrained GPU supply. That single sentence reframes the whole design. Most academic serving work assumes you have enough GPUs and asks how to use them efficiently. Mooncake assumes you do **not** have enough GPUs and asks a harder question: when you cannot serve everyone, how do you decide who to serve, how do you avoid wasting compute on requests you will eventually drop, and how do you squeeze the cluster's idle DRAM and SSD into a cache big enough to stop recomputing prefixes? The answer they arrive at is to make the **KVCache** the center of gravity of the entire system.

![Mooncake routes every request around where its KVCache lives](/imgs/blogs/mooncake-1.png)

The diagram above is the mental model: a request does not get routed to "the least busy GPU." It gets routed by a global scheduler called **Conductor** that first predicts where the request's prefix already lives in a cluster-wide KVCache store, predicts the time-to-first-token that each candidate prefill instance would deliver, and — critically — decides whether to accept the request at all before spending a single FLOP on it. Prefill and decode run in separate pools so each can be tuned for its own latency target, and a distributed KVCache store woven from idle CPU DRAM, SSD, and RDMA fabric stitches the two pools together. Everything downstream of Conductor is in service of one philosophy the paper states plainly: trade "more storage for less computation."

> [!tldr] TL;DR
> - **What it claims:** A KVCache-centric, prefill/decode-disaggregated serving architecture delivers up to **525%** more throughput on simulated 128k-token long-context workloads under SLO, and serves roughly **75% more real Kimi requests** under production SLOs than a vLLM baseline — by pooling idle CPU/DRAM/SSD/RDMA into a distributed KVCache and scheduling around cache locality.
> - **Why it matters:** It is a production-validated blueprint (Best Paper at FAST 2025, the serving stack behind Kimi) showing that prefix-cache reuse plus stage disaggregation is the dominant lever for long-context, overload-constrained LLM serving — and the Transfer Engine and Mooncake Store have since been open-sourced and wired into vLLM and SGLang.
> - **Most surprising finding:** Under overload, the biggest win is not a faster kernel; it is **rejecting requests earlier**. A prediction-based early-rejection policy that forecasts decode load cut rejected requests from 4,183 to 3,589 (**14.2%** fewer) by damping a load oscillation that naive rejection actually makes worse.
> - **Where it fails:** The cache hit ratio is workload-dependent (only ~50% reuse in their general workload), the prefill/decode instance ratio is **preset manually** with no dynamic rebalancing, request-level output-length prediction is unsolved, and the cache-reuse threshold is a hand-tuned constant.

## Context: what came before

The lineage here runs through three threads that Mooncake fuses. The first is **continuous batching** — the technique behind Orca and popularized by vLLM — which decouples the batch boundary from the request boundary so that a finished sequence can be evicted and a new one slotted in mid-flight. Continuous batching is what makes decode throughput tolerable, and Mooncake's decode pool uses it directly. But continuous batching alone treats prefill and decode as phases of the same engine on the same GPU, and that is the coupling Mooncake attacks.

The second thread is **PagedAttention and prefix caching**. vLLM's PagedAttention chopped the KVCache into fixed-size blocks so that VRAM no longer had to be reserved contiguously per sequence, which in turn made it cheap to **share** a prefix's KVCache blocks across requests that begin with the same tokens. That sharing is gold for a service like Kimi where many requests re-send the same long system prompt or the same uploaded document. Mooncake takes prefix caching and asks: why stop at one node's VRAM? If a prefix's KVCache can be reused, store it once in a cluster-wide pool and let any prefill instance pull it, rather than recomputing it.

The third thread is **disaggregated inference** — the idea, explored concurrently by systems like DistServe and SplitWise, that prefill and decode should run on physically separate instances precisely because they have different bottlenecks. The diagram below makes the argument concrete. A coupled cluster is forced to pick a single parallelism configuration and a single hardware profile that must serve both a compute-bound prefill and a memory-bound decode; when a long-context request lands, its enormous prefill stalls the decode of every sequence sharing that GPU, blowing the time-between-tokens budget.

![Disaggregation lets prefill and decode pick opposite configs](/imgs/blogs/mooncake-2.png)

The gap Mooncake fills is the **integration** of all three at production scale, plus two things the prior work largely punted on. First, prior disaggregation work tended to assume resources were ample; Mooncake is overload-first, which forces **scheduling and early rejection** to be primary design concerns rather than afterthoughts. Second, prior prefix-caching work kept the cache in VRAM or a single node; Mooncake builds a genuine **distributed KVCache pool** out of the cluster's otherwise-idle DRAM, SSD, and RDMA bandwidth, and then makes the global scheduler treat *cache locality* — not GPU utilization — as the primary routing signal. That inversion is the paper's signature idea.

It is worth dwelling on *why* the coupling is so painful, because the disaggregation only pays off if the underlying asymmetry is real. Prefill cost scales with the *square* of sequence length in the attention term — every token attends to every prior token — so a 128k-token prompt does not cost twice a 64k-token prompt, it costs closer to four times in the attention work, and the FLOP budget is enormous. Decode cost, by contrast, scales with the *batch size* and the per-step memory traffic: each decode step reads the entire KVCache of every sequence in the batch to produce exactly one token per sequence. The arithmetic intensity is low, so decode is starved for memory bandwidth, not compute. Put a giant compute-bound prefill on the same GPU as a batch of memory-bound decodes and the prefill monopolizes the SMs for hundreds of milliseconds while the decodes wait, and every waiting decode blows its TBT budget. This is not a tuning problem you can schedule your way out of inside a single engine; the two workloads are physically contending for the same silicon. Disaggregation is the structural fix: give prefill its own GPUs running its own parallelism (TP for short prompts, CPP for long ones), give decode its own GPUs running continuous batching, and the contention disappears by construction.

The other piece of lineage that matters is the economic framing. The paper's stated philosophy — trade "more storage for less computation" — is only rational because of a specific hardware reality: in a GPU serving cluster, the GPUs are the scarce, expensive, fully-utilized resource, while CPU DRAM, SSD, and RDMA bandwidth are comparatively abundant and frequently idle. Recomputing a prefix burns the scarce resource; caching it burns the abundant ones. Mooncake's entire architecture is a bet that this price ratio holds, and the bet is what justifies building a distributed cache instead of just buying more GPUs (which, the paper notes, they could not, given supply constraints).

## Contributions

1. **A KVCache-centric disaggregated architecture.** Prefill and decode run as two independently scalable GPU pools, joined by a distributed KVCache store. The scheduler routes by predicted cache hits and predicted TTFT rather than by instance load alone.
2. **Conductor, a global KVCache-aware scheduler.** For each request it selects a prefill/decode instance pair, predicts KVCache usage and TTFT, orchestrates KVCache replication, swapping, and transfer, and makes the accept/reject decision before prefill begins.
3. **Chunked Pipeline Parallelism (CPP)** for long-context prefill, which pipelines a single request's prefill across nodes with cross-node communication only at pipeline-stage boundaries — far cheaper than the per-layer cross-node traffic of sequence or tensor parallelism.
4. **Layer-wise prefill** with asynchronous, overlapped per-layer KVCache load/store, which decouples prefill VRAM cost from total KVCache size so the scheduler can "disregard the available VRAM size in prefill scheduling, as long as it can contain a single request."
5. **A distributed KVCache pool (Mooncake Store)** built from CPU DRAM, SSD, and RDMA, organized as paged blocks with prefix hashing for deduplication and prefix-match reuse, with LRU/LFU eviction.
6. **Overload-oriented scheduling with prediction-based early rejection**, which sheds load before prefill and forecasts short-term decode load to dampen the prefill/decode load oscillation that naive rejection induces.
7. **A production validation and an open-sourced Transfer Engine.** The whole stack runs Kimi, won Best Paper at FAST 2025, and the RDMA Transfer Engine plus Mooncake Store have been open-sourced and integrated with vLLM and SGLang.

## Method

The architecture has five moving parts: Conductor (the brain), the prefill pool, the decode pool, the KVCache store, and the Messenger/Transfer Engine that physically moves blocks between them. We will define each, define every symbol on first use, and walk a single request through the system end to end. Throughout, the model itself is a black box — the experiments use "a dummy model that follows the same architecture as LLaMA2-70B" because the real Kimi model is proprietary, and the design deliberately does not depend on model internals.

### Prefill/decode disaggregation

The first decision is to physically split inference into two GPU clusters. The **prefill pool** is optimized for **TTFT** (time-to-first-token), the latency from a request's arrival to the moment the first output token is produced. The **decode pool** is optimized for **TBT** (time-between-tokens), the latency between successive output tokens, and runs continuous batching so a finished sequence is evicted and a new one admitted without a batch barrier.

Because the pools are separate, each can be sized and parallelized independently. The paper notates a deployment as `[xP + yD]`: `[3P+1D]` means three prefill instances and one decode instance, `[10P+10D]` means ten of each. The ratio is a knob, and — important for the critique later — it is a **manually preset** knob. The cost of a wrong ratio is real: in the ablations, `[2P+2D]` underperforms `[3P+1D]` on TTFT purely because the prefill side becomes the bottleneck for their workload, so the split is not free and not self-tuning.

Here is the shape of the request lifecycle across the two pools, written as pseudocode against the abstractions the paper describes. Note the comments live inside the function body so they never sit at column zero.

```python
def serve(request, conductor, prefill_pool, decode_pool, kv_store):
    """End-to-end path of one request through the disaggregated pools.

    request carries: input_len, output_len_hint, hash_ids (block prefix hashes).
    Conductor owns every accept/route decision; the pools just execute.
    """
    plan = conductor.plan(request)          # predict TTFT, cache hit, pick pair
    if plan.reject:                          # overload-oriented early rejection
        return Rejection(reason="SLO infeasible under predicted load")

    prefill = prefill_pool[plan.prefill_id]
    decode = decode_pool[plan.decode_id]

    # Reuse the matched prefix; only recompute the uncached suffix.
    cached_blocks = kv_store.fetch(request.hash_ids[: plan.cache_hit_blocks])
    new_blocks = prefill.run(
        request,
        reuse=cached_blocks,
        method="cpp" if plan.uncached_tokens > PREFILL_CHUNK else "tp",
    )
    kv_store.put(new_blocks)                 # write fresh KV back to the pool

    first_token = decode.attach(request, cached_blocks + new_blocks)
    yield first_token                        # TTFT clock stops here
    for tok in decode.continuous_batch_loop(request):
        yield tok                            # each iteration must satisfy TBT
```

The `PREFILL_CHUNK` threshold and the `cpp` versus `tp` choice are the subject of the next two subsections. The `kv_store.fetch` / `kv_store.put` calls are the subject of the subsection after that. Everything Conductor decides — `plan` — is the subject of the scheduling subsection.

### KVCache-centric global scheduler (Conductor)

Conductor is the scheduler, and the name is apt: it does not play any instrument, it tells the instruments when to come in. For each incoming request it performs four jobs. It **selects** a prefill/decode instance pair. It **predicts** the request's KVCache usage and the TTFT each candidate prefill instance would deliver. It **orchestrates** KVCache replication, swapping, and transfer so that the chosen prefill instance has the prefix it needs. And it **decides acceptance versus rejection** based on predicted system load.

The defining property is that routing is driven by *where the KVCache lives*. Conductor estimates a prefill instance's execution time as a function of the request's total length and its prefix-cache-hit length on that instance, adds the instance's current queuing delay, and picks the instance that minimizes **predicted TTFT**. When an instance has a deep prefix match, recomputing locally is cheap; when it does not, Conductor weighs the cost of transferring the prefix from a node that *does* have it against the cost of recomputing it. That trade-off is governed by a **reusability threshold**: the heuristic "prefers to compute the input tokens if the best remote prefix match length is no larger than the current local reusable prefix multiplied by a threshold." In plain terms — if a remote node only barely beats your local match, don't bother shipping the cache over the network; just recompute.

The scheduler also does **cache load balancing**. Hot KVCache blocks — a popular system prompt, a frequently-referenced document — get replicated across nodes "when the estimated additional prefill time is shorter than the transfer time." This is a clean cost model: replicate a block if and only if having a local copy saves more prefill time than the replication costs to move. The threshold that governs this is, again, currently manual. We can sketch the cache-aware selection as follows.

```python
def select_prefill_instance(request, instances, kv_store, reuse_threshold):
    """Pick the prefill instance with the lowest predicted TTFT.

    predicted_ttft = queue_delay + prefill_time(total_len, hit_len).
    Cache locality enters through hit_len: a deeper local match means
    fewer tokens to recompute, hence lower predicted prefill_time.
    """
    best, best_ttft = None, float("inf")
    for inst in instances:
        local_hit = kv_store.prefix_match_len(inst, request.hash_ids)
        remote_hit = kv_store.best_remote_match_len(request.hash_ids)

        # Only consider a remote transfer if it materially beats local reuse.
        if remote_hit > local_hit * reuse_threshold:
            hit_len = remote_hit
            transfer_cost = kv_store.transfer_time(remote_hit)
        else:
            hit_len, transfer_cost = local_hit, 0.0

        recompute = request.total_len - hit_len
        ttft = inst.queue_delay() + prefill_time(recompute) + transfer_cost
        if ttft < best_ttft:
            best, best_ttft = inst, ttft
    return best, best_ttft
```

The honest caveat the paper makes is that this whole calculus rests on a **TTFT prediction**, and predictions are imperfect — which is exactly why the overload machinery later has to be prediction-aware too, to avoid stacking one bad estimate on another.

A worked example makes the cost model tangible. Suppose a request arrives whose prefix is a 30,000-token document that two prefill instances have seen before. Instance A holds the first 28,000 tokens of that prefix locally (`local_hit = 28000`); instance B holds the full 30,000 (`remote_hit = 30000`) but B is busy and would have to ship those blocks to A over the fabric. With `reuse_threshold = 1.5`, the guard `remote_hit > local_hit * reuse_threshold` evaluates to `30000 > 42000`, which is **false** — so Conductor declines the remote transfer and simply recomputes the missing 2,000 tokens locally on A. The intuition the threshold encodes is that recomputing a small uncached suffix is almost always cheaper than dragging a marginally-deeper prefix across the network. Now flip the numbers: if A had only `local_hit = 5000` and B had `remote_hit = 30000`, the guard becomes `30000 > 7500`, which is **true**, and Conductor pays the transfer because shipping 30k tokens' worth of KVCache beats recomputing 25,000 tokens of prefill. The whole point of the threshold is to keep the system on the right side of that crossover, and the limitation the paper flags is that the crossover constant is currently set by hand rather than learned from measured transfer-versus-recompute times.

### Chunked Pipeline Parallelism (CPP)

Long inputs are the regime Mooncake actually cares about, and they break naive parallelism. If you tensor-parallelize a 128k-token prefill across nodes, you pay cross-node communication at *every layer*, and that traffic dominates. If you sequence-parallelize, you again pay per-layer cross-node traffic. CPP sidesteps this. When a request's uncached token count exceeds the threshold `prefill_chunk` — which the paper says is "typically larger than 1000 tokens" — Conductor splits that single request's prefill into chunks and **pipelines** the chunks across multiple prefill nodes.

![Chunked pipeline parallelism splits a long prefill across nodes](/imgs/blogs/mooncake-3.png)

The figure above traces it: match the prefix in the store, find the uncached suffix is over the chunk threshold, split it into chunks across prefill nodes, run them as pipeline stages, and — the key property — incur "cross-node communication only at the boundaries of each pipeline stage." Boundary-only communication is what makes CPP cheap relative to TP or sequence parallelism for long contexts; you move activations between stages a handful of times, not once per layer. Short requests, where the uncached suffix is below the chunk threshold, skip all of this and use plain tensor parallelism, because for short inputs the pipeline bubble would cost more than it saves.

It is worth being precise about what CPP is and is not. It is **intra-request** pipelining: one long prompt's prefill is spread over several nodes to bring its TTFT down. It is not the same as pipelining different requests through different stages. The win is latency on the single long request, paid for by the pipeline's stage-boundary transfers, which are amortized because there are far fewer layer-boundaries than there are model layers.

### Layer-wise prefill

The second prefill-side trick decouples VRAM pressure from cache size. Naively, to prefill a request you need enough VRAM to hold its entire KVCache while you compute and before you can ship it off to the store. For a 128k-token request that is a lot of VRAM, and it would force the scheduler to reason about VRAM headroom on every routing decision. **Layer-wise prefill** removes that constraint: KVCache loading and storing happen **asynchronously per model layer, overlapped with computation**. As soon as layer $\ell$ finishes producing its slice of KVCache, that slice is asynchronously streamed out to the store while the GPU computes layer $\ell+1$; conversely, cached prefix blocks needed for layer $\ell$ are prefetched while earlier layers compute.

The payoff is stated as a scheduling simplification: it lets Conductor "disregard the available VRAM size in prefill scheduling, as long as it can contain a single request." That is a strong statement — it means the prefill VRAM budget is set by the *largest single request*, not by the *sum of all cached prefixes*, which is what makes a cluster-wide cache feasible without each prefill GPU needing to hold the whole thing.

Here is the overlap structured as a loop. The async handles are what hide the transfer latency behind compute.

```python
def layerwise_prefill(request, layers, kv_store):
    """Per-layer KV store/load overlapped with the next layer's compute.

    Each layer's freshly computed KV is shipped out asynchronously while the
    next layer computes; required prefix blocks are prefetched the same way.
    Peak VRAM is bounded by a single layer's working set + one request, not
    by the total cached-prefix size.
    """
    pending_stores = []
    prefetch = kv_store.async_load(request.hash_ids, layer=0)
    for l, layer in enumerate(layers):
        cached = prefetch.result()                  # block until layer-l prefix ready
        if l + 1 < len(layers):
            prefetch = kv_store.async_load(request.hash_ids, layer=l + 1)

        kv_l = layer.forward(request, prefix_kv=cached)   # compute overlaps stores
        pending_stores.append(kv_store.async_store(kv_l, layer=l))

    for h in pending_stores:                          # drain before handoff
        h.wait()
```

### Distributed KVCache pool (Mooncake Store)

The store is where the "more storage for less computation" philosophy becomes hardware. The cluster has a large amount of CPU DRAM, SSD, and RDMA bandwidth that sits idle while the GPUs are pinned. Mooncake aggregates all of it into a single paged, disaggregated KVCache store.

![The Mooncake Store pools idle cluster memory into one KV tier](/imgs/blogs/mooncake-5.png)

The store is organized as **paged KVCache blocks** with **prefix hashing**. Each block's hash is "determined by both its own hash and its prefix." That detail matters: hashing a block by its own content *and* its prefix means two requests that share a common prefix produce identical block hashes for the shared portion, which gives you deduplication and prefix-match reuse for free. If request A sent a 30k-token document and request B sends the same document plus a question, B's first blocks hash-match A's, and the store serves them without recomputation. Eviction is **LRU/LFU**, so cold prefixes age out and hot ones stay resident. From Conductor's vantage point, all of this — the DRAM, the SSD spillover, the RDMA fabric, the eviction policy — is one logical store it can query for prefix-match length and ask to fetch or replicate blocks.

The block hashing is simple enough to write down. The recurrence is the whole trick: a block's identity folds in the running prefix hash, so identity is path-dependent.

```python
def block_hashes(token_blocks):
    """Compute prefix-aware hashes for a request's KV blocks.

    h(block_i) folds the running prefix hash into the block's own content
    hash, so any two requests sharing a token prefix share block hashes for
    that prefix and the store dedups + reuses them.
    """
    running = 0  # rolling prefix hash, seeded per-model/per-config upstream
    out = []
    for blk in token_blocks:
        own = content_hash(blk)              # hash of this block's tokens
        running = combine(running, own)      # fold in -> own-and-prefix hash
        out.append(running)
    return out
```

### Messenger and the Transfer Engine

Moving KVCache blocks between GPU memory, CPU memory, and across machines is the physical layer, and it is handled by a per-node **Messenger** process built on **(GPUDirect) RDMA**. It handles high-speed cross-machine block movement, congestion, and the concurrent asynchronous loading that decode needs while it streams tokens. The open-sourced **Transfer Engine** supports TCP, RDMA, NVLink, NVMe-oF, EFA, and custom transports, which is why it could later be lifted out and reused as a backend in other serving stacks.

The bandwidth numbers are the reason this works at all. On a 40 GB transfer, the Transfer Engine hits up to **87 GB/s** on a 4×200 Gbps RoCE configuration (about 2.4× faster than TCP) and up to **190 GB/s** on an 8×400 Gbps RoCE configuration (about 4.6× faster than TCP). When integrated into vLLM, swapping TCP for the Transfer Engine dropped mean TTFT from **1414.05 ms** to **1056.76 ms**, roughly a 25% improvement, purely from faster KVCache movement. If transferring a cached prefix were slow, the entire "ship the cache instead of recomputing it" bet would lose; the RDMA fabric is what makes the bet pay.

To make the component responsibilities concrete, here is how the five parts divide the labor.

| Component | Optimizes for | Owns | Key mechanism | Cost it trades |
| --- | --- | --- | --- | --- |
| Conductor | Global SLO attainment | Routing, accept/reject | Predicted-TTFT, cache-aware selection | Prediction error risk |
| Prefill pool | TTFT | Incremental prefill | TP (short) / CPP (long), layer-wise KV | Pipeline bubbles on long inputs |
| Decode pool | TBT | Token generation | Continuous batching, async KV load | Memory-bound batch ceiling |
| KVCache store | Reuse / capacity | Cached prefixes | Paged blocks, prefix hash, LRU/LFU | Workload-dependent hit ratio |
| Messenger / TE | Transfer bandwidth | Block movement | GPUDirect RDMA, up to 190 GB/s | RDMA fabric requirement |

### Overload-oriented scheduling and early rejection

Now the part that only an overload-first paper would foreground. When the cluster is saturated, the worst thing you can do is run a full prefill and then discover at decode time that you have no capacity, because that prefill compute is now pure waste. So Mooncake rejects requests **before** prefill when they cannot meet their SLOs. The basic strategy evaluates both prefill and decode instance load and rejects requests that cannot be served in time.

But naive early rejection has a subtle failure mode the paper calls out: **load oscillation**. There is a time lag between when prefill finishes a request and when that request shows up as decode load. If you reject based only on *current* decode load, you will over-admit during the lag, then the admitted requests will all hit decode at once, decode will spike, you will over-reject, decode will drain, you will over-admit again — a sawtooth that wastes capacity at both extremes.

The fix is **prediction-based early rejection**: forecast the *short-term* decode load instead of reading the instantaneous load, "assuming that each request's decoding stage takes a uniform time $t_d$." With a per-request decode-time estimate $t_d$, Conductor can project how much decode load the currently-in-flight prefills will become, and admit or reject against the *projected* load rather than the lagging *current* load. That single change is what stabilizes the prefill/decode balance.

![Conductor splits scheduling from overload control](/imgs/blogs/mooncake-6.png)

The taxonomy above separates the two concerns Conductor juggles. On the routing side, the policy ladder runs random → load-balancing → cache-aware, and cache-aware wins. On the overload side, the ladder runs baseline → early rejection → predictive early rejection, and predictive wins. They are orthogonal knobs: one decides *which good instance* gets a request you are keeping; the other decides *whether to keep it at all*.

The predictive rejection logic, in sketch form:

```python
def admit(request, prefill_load, in_flight_prefills, t_d, slo):
    """Prediction-based early rejection.

    Reject before prefill if the PROJECTED near-term decode load (not the
    lagging instantaneous load) would push this request past its TBT/TTFT
    SLO. t_d is the assumed uniform per-request decode duration.
    """
    # Project decode load: requests already prefilling will land on decode soon.
    projected_decode = current_decode_load() + len(in_flight_prefills)
    projected_pressure = projected_decode * t_d

    if prefill_load > slo.prefill_ceiling:
        return False
    if projected_pressure > slo.decode_ceiling:
        return False                     # would violate TBT once it lands
    return True
```

## Experiments

The evaluation runs on nodes of **8× NVIDIA A800-SXM4-80GB GPUs with NVLink** (the v1 text quotes A800; a later extraction renders A100 — essentially the same silicon, the A800 being the China-market variant), connected by an RDMA network with cross-node bandwidth up to **800 Gbps**, on testbeds of 8, 16, or 20 nodes depending on the experiment. The model is the dummy LLaMA2-70B-shaped network. The workloads are three kinds: a real production trace of **23,608 entries** from a 1-hour sample (each entry carrying `timestamp, input_length, output_length, hash_ids`); two public datasets, **ArXiv Summarization** (avg 8,088 in / 229 out) and **L-Eval** (avg 19,019 in / 72 out); and simulated prompts of 16k / 32k / 64k / 128k tokens with 512 output tokens.

The SLOs are defined relative to a single-request baseline. End-to-end experiments cap the P90 latencies at **TTFT_P90 = 10×** and **TBT_P90 = 5×** the single-request baseline. The real deployment uses fixed thresholds: TTFT monitored at **30 seconds** and TBT at **0.1 seconds per token** (100 ms/token). Those production thresholds are worth internalizing — a 30-second TTFT ceiling is generous because Kimi serves very long contexts, and a 100 ms/token TBT is the human-readable-streaming budget.

![Throughput gains grow with context length and on real traces](/imgs/blogs/mooncake-4.png)

The headline result is that Mooncake's advantage **grows with context length**, which is exactly what you want from a long-context-oriented design. The throughput increase over the baseline climbs from ~50% at 16k, to ~150% at 32k, to ~300% at 64k, and to **up to 525%** at 128k, all while adhering to the SLOs. The mechanism is intuitive: the longer the context, the more prefill compute there is to save through cache reuse and the more the coupled baseline suffers from long prefills stalling decode. Here are the numbers in one place.

| Benchmark / scenario | Metric | Mooncake | Baseline |
| --- | --- | --- | --- |
| ArXiv Summarization, `[3P+1D]` | Throughput gain under SLO | **~20%** | vLLM `[4M]` |
| L-Eval, `[3P+1D]` | Throughput gain under SLO (prefix-cache benefit) | **~40%** | vLLM `[4M]` |
| Simulated 16k | Throughput increase | **~50%** | baseline |
| Simulated 32k | Throughput increase | **~150%** | baseline |
| Simulated 64k | Throughput increase | **~300%** | baseline |
| Simulated 128k | Throughput increase | **up to 525%** | baseline |
| Real workload, `[10P+10D]` vs vLLM `[20M]` | More requests under SLO | **~75% more** | vLLM `[20M]` |
| Real workload | TBT SLO attainment | **100%** | vLLM **57%** |
| Real workload | TTFT SLO attainment | **~100%** | vLLM **~100%** |
| Real traces (abstract v4) | Effective request-capacity increase | **59% – 498%** | baseline methods |

The real-workload result is the one I trust most, because it is the actual product. On ~23k real requests, Mooncake `[10P+10D]` served roughly **75% more requests** under SLO than vLLM `[20M]`. The detail that makes that number meaningful is the breakdown: both systems hit ~100% TTFT SLO attainment, but on **TBT** SLO attainment Mooncake hit **100%** versus vLLM's **57%**. That gap is the disaggregation thesis vindicated — the coupled baseline meets first-token latency fine, but it cannot keep token-to-token latency in budget because long prefills keep interrupting decode. Splitting the pools is precisely what fixes TBT.

The two ablations isolate the two scheduler knobs. On scheduling policy (8P/8D, ~23k requests), random scheduling sets the TTFT baseline, load-balancing scheduling cuts TTFT substantially over random, and **cache-aware** scheduling beats both on TTFT and on SLO attainment — isolating the value of KVCache-locality-aware routing. On overload control (8 prefill + 8 decode, 2× replay speed), the rejection ladder is:

| Strategy | Requests rejected | Improvement vs baseline |
| --- | --- | --- |
| Baseline strategy | **4,183** | — |
| Early Rejection | **3,771** | **9.8%** fewer |
| Early Rejection w/ Prediction | **3,589** | **14.2%** fewer |

Two things are load-bearing in this table and might not transfer. First, the absolute rejection counts are a function of running at **2× replay speed** — they deliberately overload the system to make the rejection logic matter, so the *ratios* generalize but the *counts* do not. Second, the prediction-based win rests on the uniform-$t_d$ assumption, which is a coarse approximation; in a workload where decode lengths are wildly bimodal, a single $t_d$ would predict load poorly and the 14.2% could shrink. The provenance arc — internal stack, open Transfer Engine, FAST 2025 best paper, vLLM/SGLang integration — is summarized below.

![From Kimi production stack to open-sourced KV ecosystem](/imgs/blogs/mooncake-7.png)

What is most load-bearing across the whole evaluation is the **cache hit ratio**, and the paper is refreshingly direct that it is workload-dependent. Only "~50% of the KVCache can be reused in our current workloads," while specialized scenarios like a "chat-to-paper service" reach "90%." The throughput numbers are downstream of this ratio. If your workload has near-zero prefix overlap — say, every request is a unique short query with no shared system prompt — then the store has nothing to reuse, CPP and layer-wise prefill still help with long inputs, but the dramatic "525% at 128k" figure assumes there is cacheable structure to exploit. That is the single biggest "might not transfer" caveat.

To see how the hit ratio multiplies through to throughput, walk the cost of a single 64k-token request. At a 50% hit ratio, 32k tokens are served from the cache and 32k must be prefilled; at the chat-to-paper service's 90%, only 6.4k tokens need fresh prefill — a 5× reduction in prefill compute for the *same* request. Because prefill attention cost is super-linear, halving the uncached token count more than halves the work, so the hit ratio is a force multiplier rather than a linear discount. This is exactly why the gains widen with context length in the matrix above: a longer prompt has more absolute tokens to potentially cache, and the cached fraction is recomputed for free on every repeat. It is also why the two public-dataset numbers differ so much — ArXiv Summarization shows ~20% throughput gain while L-Eval shows ~40%, and the paper attributes the L-Eval gap specifically to its higher prefix-cache benefit. The lesson for a practitioner is blunt: before you adopt this architecture, *measure your workload's prefix overlap*, because that single number sets the ceiling on what disaggregation plus caching can buy you. A workload that looks like L-Eval will see Mooncake-scale wins; a workload of unique, short, prefix-disjoint queries will see the architecture's overhead with little of its upside.

There is a second, quieter dependency worth surfacing: the SLO definitions themselves shape the throughput numbers. The end-to-end experiments cap P90 latencies at 10× (TTFT) and 5× (TBT) the single-request baseline, which are *relative* multipliers. A stricter SLO — say 3× TBT — would shrink the admissible batch size on both systems and compress the gap, while a looser SLO would widen it. The production thresholds (30 s TTFT, 100 ms/token TBT) are absolute and generous because Kimi serves long contexts where users already expect a wait. If you port these results to a latency-sensitive product with a 2-second TTFT budget, the relative ordering should hold but the magnitudes will not, because the baseline and Mooncake will both be operating in a much tighter feasible region. None of this undercuts the result; it just bounds where the specific percentages apply.

## Critique

**What is strong.** The paper is production-grounded, and that shows in the choices. Overload is treated as the common case, not an edge case, and the prediction-based rejection result — a non-obvious win from forecasting decode load to break an oscillation — is the kind of finding you only get from running a real service. The disaggregation-plus-cache thesis is cleanly validated by the TBT breakdown (100% vs 57%), which pinpoints exactly *why* the coupled baseline fails rather than just reporting that it does. And the engineering is honest about its cost models: replicate a hot block only if the prefill time saved exceeds the transfer cost; ship a remote prefix only if it beats local reuse by a threshold. Those are falsifiable, inspectable rules.

**What is weak or unfalsifiable.** The model is a "dummy LLaMA2-70B," and the real Kimi architecture is proprietary, which limits reproducibility of the *absolute* numbers — you can reproduce the system but not necessarily the exact 525%, because that depends on the real model's KVCache geometry and the real workload's hit ratio. The "thousands of nodes, over 100 billion tokens daily" framing appears in the abstract summary but the paper's own body emphasizes the 75% real-workload and 525% simulated figures; treat the operational-scale claim as context, not a measured result. The "59%–498%" effective-capacity range is from a later revision (v4) abstract while the v1 body leans on the 75%/525% figures — version drift to be aware of when citing.

**What ablation is missing.** The most important missing study is a **dynamic P/D ratio**. The paper presets the prefill/decode split manually and shows `[2P+2D]` underperforming `[3P+1D]`, which demonstrates the ratio matters but does not show what an *adaptive* rebalancer would buy. Given that workloads shift over a day (more long-context uploads at some hours, more short chat at others), a static ratio is leaving throughput on the table, and we have no measurement of how much. Second, the **reusability threshold** that governs recompute-versus-transfer is hand-tuned; there is no sensitivity sweep showing how fragile the results are to that constant. Third, there is no ablation that varies the **cache capacity** itself to map the throughput-versus-DRAM curve — given that the whole bet is "more storage for less compute," the marginal value of the next gigabyte of cache is the most interesting curve in the design and it is not plotted.

**What would change my mind.** If someone showed that on a workload with sub-20% prefix-cache reuse, Mooncake's advantage over a well-tuned coupled vLLM collapses to single digits, I would downgrade the architecture from "default for serving" to "default for *long-context, high-reuse* serving." The cache hit ratio is the hinge; the paper's own ~50%/90% spread already hints that the gains are conditional on cacheable structure, and a clean low-reuse counter-experiment would set the boundary of where this design earns its complexity.

## What I'd build with this

1. **An adaptive P/D rebalancer.** The obvious next system is a controller that watches the live prefill and decode queue depths and migrates instances between pools on a slow timescale (minutes, not seconds, to avoid thrashing). The paper hands you the cost signals already — projected decode load via $t_d$, prefill queue delay — so the controller has its inputs; what is missing is the actuation. Even a coarse "if prefill queue P90 exceeds decode queue P90 by 2× for five minutes, convert one decode instance to prefill" rule would beat a static ratio.

2. **A learned reusability threshold.** Replace the hand-tuned recompute-versus-transfer constant with a tiny online model that predicts, per request class, whether shipping a remote prefix will actually beat recomputing it, using measured transfer times and prefill times as labels. This directly attacks the "manual threshold" limitation and is cheap because the labels are generated for free by every served request.

3. **Tiered cache eviction tuned to prefix shape.** LRU/LFU is generic. A service like Kimi has structure — system prompts and uploaded documents are hot and long, ad-hoc queries are cold and short. An eviction policy that is aware of *block prefix depth* (keep deep, widely-shared prefixes; evict shallow, request-private suffixes first) should lift the hit ratio above the generic 50% on mixed workloads.

4. **Output-length-aware admission.** The paper concedes request-level output-length prediction is "challenging due to high costs or low accuracy," so it uses a uniform $t_d$. A lightweight classifier that bins requests into short/medium/long output buckets — even at modest accuracy — would let the predictive-rejection logic use a *per-bucket* $t_d$ instead of one global constant, which should sharpen the load projection and push the 14.2% rejection improvement higher.

5. **A speculative-prefetch layer.** Because block hashes are prefix-derived and Conductor sees the request before prefill, you can speculatively prefetch the predicted prefix blocks from SSD to DRAM (or DRAM to GPU) the instant a request arrives, overlapping the transfer with the routing decision itself. The Transfer Engine's bandwidth makes this nearly free to attempt and cheap to be wrong about.

## When to reach for Mooncake (and when not to)

Reach for this architecture when three conditions hold together: your workloads are **long-context**, your requests have **shared prefix structure** (system prompts, uploaded documents, repeated context), and you are **capacity-constrained** so that throughput-under-SLO and graceful overload behavior actually matter. That is precisely the Kimi regime, and it is increasingly the regime of any serious document-QA, coding-assistant, or agent-backend service. In that world, the disaggregation buys you a clean TBT (the 100%-vs-57% result), the distributed KVCache buys you the compute you do not have to spend, and the prediction-based rejection buys you a system that degrades gracefully instead of collapsing at peak.

Do **not** reach for it when your workload is short-context, low-reuse, and comfortably over-provisioned. If every request is a unique two-sentence query, there is no prefix to cache, so the store sits empty and you have paid the full operational cost of a distributed KVCache, an RDMA fabric, and a global scheduler for a benefit that mostly evaporates. A well-tuned single-cluster vLLM with continuous batching and PagedAttention will be simpler and competitive there. The same caution applies if you cannot supply the RDMA fabric: the entire "transfer the cache instead of recomputing it" bet depends on the Transfer Engine's 87–190 GB/s, and on commodity TCP networking the transfer cost can erase the recompute savings — recall that even within the paper, swapping TCP for RDMA was a 25% TTFT swing.

The deeper lesson, independent of whether you adopt the exact stack, is the inversion at the heart of the design: stop scheduling LLM requests by GPU utilization and start scheduling them by **where their KVCache lives**. Once you accept that prefill is a tax you can avoid paying twice, the cache stops being an implementation detail and becomes the thing the whole system is organized around. Mooncake is the most complete published argument that this inversion is the right one for the workloads that are eating production inference budgets today.

## References

- **Paper (arXiv abstract):** [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving — arXiv:2407.00079](https://arxiv.org/abs/2407.00079)
- **Code (GitHub):** [kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake)
- **FAST 2025 (peer-reviewed, Best Paper):** [fast25-qin.pdf](https://www.usenix.org/system/files/fast25-qin.pdf)
- Related reading on the Kimi/Moonshot stack and serving:
  - [Kimi K2: Open Agentic Intelligence](/blog/paper-reading/large-language-model/kimi-k2)
  - [Kimi Linear: An Expressive, Efficient Attention Architecture](/blog/paper-reading/large-language-model/kimi-linear)
  - [MoBA: Mixture of Block Attention for Long-Context LLMs](/blog/paper-reading/large-language-model/moba)
  - [Kimi K2 Thinking: An Open-Source Reasoning Model Built on K2](/blog/paper-reading/large-language-model/kimi-k2-thinking)
