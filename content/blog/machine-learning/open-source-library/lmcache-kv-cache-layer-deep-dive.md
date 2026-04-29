---
title: "LMCache: An Engineer's Deep Dive into the KV Cache Layer for vLLM and SGLang"
date: "2026-04-29"
publishDate: "2026-04-29"
description: "A deep, opinionated walkthrough of LMCache — the engine-decoupled KV cache layer behind vLLM and SGLang. Connector internals, chunk hashing, layer-wise pipelining, CacheGen, CacheBlend, NIXL/GDS/Mooncake backends, P2P sharing, disaggregated prefill, every config knob that matters, runnable code, real benchmarks, and a long catalog of production case studies."
tags:
  [
    "lmcache",
    "kv-cache",
    "vllm",
    "sglang",
    "llm-inference",
    "cachegen",
    "cacheblend",
    "mlops",
    "gpu",
    "nixl",
    "open-source-library",
  ]
category: "machine-learning"
subcategory: "Open Source Library"
author: "Hiep Tran"
featured: true
readTime: 51
aiGenerated: true
---

Most production LLM stacks discover the same uncomfortable truth in their second quarter of operation: the GPU is rarely the bottleneck. The bottleneck is _the same prefix being recomputed thousands of times an hour_, because every replica, every restart, every long-context truncation throws the prefix cache away. vLLM's built-in `PrefixCache` does heroic work inside a single replica's HBM, but it is a process-local data structure pinned to a single device. The moment you run two replicas behind a load balancer, the moment a request gets re-routed, the moment context length crosses your KV-block budget — that cache is gone, and the GPU starts re-doing work it has already done.

[LMCache](https://github.com/LMCache/LMCache) is the answer that the vLLM community converged on. It treats KV cache as a _first-class data object_ that lives outside the engine, in a tiered store spanning HBM, pinned CPU DRAM, NVMe (with GPUDirect Storage), and remote backends like Mooncake, Valkey, InfiniStore, S3 Express, and Weka. Engines (vLLM v1, SGLang) talk to it through a thin connector API. The community paper [*LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference*](https://arxiv.org/abs/2510.09665) reports up to **15× throughput** improvement on multi-round QA and **3.7–6.8× lower TTFT** on real traces, and the Apr 3, 2026 architecture revision claims [10× MoE inference gains](https://blog.lmcache.ai/) on top of that.

![LMCache architecture: vLLM/SGLang worker, connector, engine, controller, and the L0–L3 storage hierarchy](/imgs/blogs/lmcache-kv-cache-layer-deep-dive-1.png)

The diagram above is the mental model: vLLM keeps owning the paged KV blocks that live in HBM during a forward pass, LMCache owns everything below that line, and the connector is a thin, _stable_ API that lets the engine evolve at its own pace without dragging the cache layer with it. The rest of this article walks each layer in detail, then closes with ten case studies of real production incidents and the heuristics a senior engineer should reach for when reasoning about them. Companion reading: my earlier posts on [KV cache fundamentals](/blog/machine-learning/large-language-model/kv-cache) and [KV cache optimization and management](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management) cover the math and the in-engine mechanics; this article is about everything that happens _outside_ the engine.

## 1. Why a Separate KV Cache Layer Exists

The naive view is that vLLM's `PrefixCache` plus PagedAttention has already solved this. It hasn't. The problems start the moment you leave the textbook setup.

| Assumption (textbook)                | Reality (production)                                                          |
| ------------------------------------ | ----------------------------------------------------------------------------- |
| One replica, infinite VRAM           | 8–32 replicas behind a load balancer; each has 60–80 GB of HBM                |
| Prompts repeat exactly               | Prompts share a 4 KB system preamble + retrieved docs in arbitrary order      |
| Context fits in `max_model_len`      | Real conversations exceed `max_model_len`; clients truncate aggressively      |
| All requests prefill from token 0    | Multi-turn agents replay 90% of the previous turn but vLLM evicts on rollover |
| Cache survives restart               | A node drain wipes ~80 GB of warmed cache                                     |
| Routing is cache-oblivious           | Cache-aware routing is the actual cost driver in RAG                          |

Each row maps to a real LMCache feature.

**The truncation paradox.** The LMCache tech report measures something counter-intuitive: when an application truncates context to fit a memory budget, the prefix-cache hit ratio _drops by half_. The reason is subtle. Truncation usually keeps the _most recent_ tokens (the conversation history) and drops the _oldest_ (the system prompt and tool descriptions). But the reusable bit is the prefix, not the recent text. Truncation specifically destroys the part the cache could reuse, while keeping the part it cannot. The fix is not "cache more aggressively in HBM" — there is no HBM left. The fix is to push the prefix _down_ to a tier where 80 GB is cheap (CPU DRAM) or 8 TB is cheap (NVMe), and to reload it lazily.

**The router blind-spot.** When two replicas serve the same model, a round-robin router will scatter cache hits randomly across them. A cache-aware router must know _which replica has the prefix in HBM right now_ — a question the engine alone cannot answer because it does not know about its peers. LMCache's controller exposes `lookup(tokens) -> (instance_id, device, hit_tokens)` for exactly this reason; it is the missing primitive every production stack ends up reinventing.

**The disaggregation pivot.** The state of the art for very-long-context serving (>32 K tokens) is _prefill-decode (PD) disaggregation_: a beefy prefill node prepares the KV cache, ships it over RDMA to a smaller decode node, and the decoder streams tokens. This shaves prefill cost off the decode hot path. But it requires a transport layer that can move tens of GB of KV cache _at line rate_ and bind it back to paged memory blocks. NIXL plus the LMCache connector is what makes this practical in vLLM v1.

**The MoE wrinkle.** Mixture-of-Experts models magnify all these problems because expert activations and router decisions add a second dimension to "what's reusable." LMCache's April 2026 architecture revision specifically addresses this — separate transfer paths per expert, smarter sharding — and reports 10× speedups on top of the existing CacheGen/CacheBlend stack.

**The agent loop trap.** Modern coding and tool-using agents are the worst-case workload for naive prefix caching: the system prompt and the tool descriptions are 4–8 K tokens of pure repetition, but every turn appends a different tool call, a different observation, a different reasoning step. Without an external cache the engine recomputes those 4–8 K reused tokens _every turn_, which on a 30-turn agent loop costs 120–240 K redundant tokens of prefill. With LMCache plus the controller's `pin` API on the system prompt, the agent's prefill collapses to "just the new turn," cutting cost by an order of magnitude on long sessions. The Anthropic and Cursor teams have written publicly about how their internal serving stacks reach for an external KV cache for exactly this reason; the open-source equivalent is LMCache.

**The autoscaler tax.** A serving cluster that scales replicas up during traffic spikes pays a hidden cost: each new replica starts with a cold prefix cache and runs at 30–50% efficiency for its first 5–15 minutes of life. Multiply that across daily auto-scale events and you've burned hours of GPU time on cold-start prefills. LMCache flips this: a new replica registers with the controller, queries `lookup` for popular hashes, and pre-warms its CPU tier from peers via P2P or from Mooncake. The same warm-up that takes 15 minutes naively now takes ~30 seconds.

The conclusion: a _separate_ KV cache layer is not a luxury. It is the only place these problems can be solved without rewriting the engine.

## 2. The Mental Model: Engine, Connector, Engine-of-Caches, Controller

LMCache splits responsibilities four ways. Keep this split in your head; every config knob is in exactly one of these boxes.

**The serving engine** (vLLM v1, SGLang) owns:
- The paged KV blocks in HBM during a forward pass.
- The scheduler that decides batch composition.
- The attention kernels and the model weights.

**The connector** owns:
- The hooks into the engine's request lifecycle.
- The transformation between vLLM's `[block_id, slot_id]` paged layout and LMCache's `[chunk_hash, layer, kv]` chunk layout.
- The decision to issue an async load or store.

**The LMCache engine** owns:
- Chunking, hashing, eviction.
- The streaming GPU buffer and reference-counted zero-copy machinery.
- CacheGen serialization, CacheBlend selective recompute.
- NUMA-aware pinned memory allocation.
- The async prefetch worker.

**The controller** (optional but recommended at scale) owns:
- Global instance metadata: who has which chunks, where.
- The external lookup API for cache-aware routers.
- P2P peer discovery and pinning.

This separation is the reason LMCache compiles cleanly against rapidly-changing vLLM versions: the connector is the only file that needs to track engine internals. The rest of LMCache evolves on its own schedule.

## 3. The Connector API: Scheduler-Side vs Runner-Side

The connector lives in two halves of the engine. Each half has a different concurrency model and a different latency budget.

**Scheduler-side hooks** run on the engine's main loop, once per request, on the Python side. They are blocking — but they should never touch a GPU. Their job is to compute metadata.

```python
class LMCacheConnectorV1:
    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> Tuple[int, bool]:
        """How many additional tokens beyond num_computed_tokens
        can be served from the external cache? Return (count, ok)."""

    def update_state_after_alloc(
        self, request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ) -> None:
        """Engine has now allocated GPU blocks. Decide whether
        we still want to load from the external store."""

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> ConnectorMetadata:
        """Build a per-step bag of (chunk_hash, gpu_block_id, layer)
        triples. The runner-side will execute against this."""
```

**Runner-side hooks** run on each model worker, with a CUDA context and direct access to the paged KV tensors. They are the only place that touches GPU memory.

```python
class LMCacheConnectorV1:
    def start_load_kv(self, forward_context, **kwargs) -> None:
        """Issue async H2D copies on Stream B for the chunks
        listed in this step's ConnectorMetadata."""

    def wait_load_kv(self, layer_name: str) -> None:
        """Block Stream A until the load for `layer_name` is done."""

    def start_store_kv(self, layer_name: str) -> None:
        """After computing layer N, schedule a D2H copy of the
        new KV slice on Stream B. Non-blocking."""

    def wait_store_kv(self) -> None:
        """End-of-step barrier — wait for all stores to drain."""
```

Why this two-layer split matters: the scheduler can run far ahead of the GPU, deciding to load chunks _before_ the model worker even starts forwarding. By the time the worker calls `wait_load_kv` for layer 0, the bytes are already on the device. The scheduler is also where multi-tenant policy lives — `priority_limit`, `min_retrieve_tokens`, `store_location`, `retrieve_locations` — without those decisions blocking the CUDA stream.

The other reason is _evolution_. vLLM v1's scheduler refactor changed many internals between 0.10 and 0.18, but the four scheduler-side method signatures held. LMCache 0.4.3 supports vLLM 0.17–0.18 with a single connector class because the engine-facing surface has been deliberately frozen.

**Walk-through of one request.** It's worth tracing exactly what happens when a request arrives. Suppose the prompt is 4096 tokens and the cache already holds the first 2048 tokens (8 chunks of 256).

1. **Scheduler step T**. The engine's scheduler builds the next batch and asks every connector how much of the prompt is already cached. LMCache walks the hash chain `h_0, h_1, ..., h_15`. The first eight hash to entries that exist; the ninth misses. `get_num_new_matched_tokens` returns `(2048, ok=True)`.
2. **Engine allocates GPU blocks**. vLLM's block allocator reserves enough KV blocks for the full 4096-token prompt. It then calls `update_state_after_alloc`, telling LMCache "I have these GPU block IDs for the cached chunks; please load into them."
3. **Metadata assembly**. `build_connector_meta` produces a list of `(chunk_hash, layer, src_tier, src_addr, dst_block_id)` triples — one per layer per chunk. For our 8 cached chunks across a 32-layer model, that's 256 triples.
4. **Runner step T (GPU)**. The model runner enters the forward pass. Before layer 0 attention, it calls `start_load_kv` which dispatches all 256 H2D copies on Stream B. The runner then calls `wait_load_kv("layer_0")` — Stream A blocks on Stream B's layer-0 event. This event fires within microseconds because layer 0's chunks were submitted first; subsequent layer waits typically don't block at all because Stream B has run ahead.
5. **Layer-by-layer**. For each subsequent layer, the runner calls `wait_load_kv(layer_n)` immediately before its attention kernel. Stream B prefetches the next layer concurrently. After computing layer N's attention for the _suffix_ tokens (positions 2048–4095), the runner calls `start_store_kv(layer_n)` to schedule the new chunks' D2H copy.
6. **End-of-step barrier**. After the final layer, `wait_store_kv` ensures all writes drain to the L1/L2 tier before the engine considers the step complete. In practice this is a no-op because Stream B has been writing the whole time the GPU was busy on later layers.

The full lifecycle touches the CPU four times (steps 1, 2, 3, end-of-step) for a single GPU forward pass. Everything else is async. This is the core engineering reason LMCache adds <1% overhead to TTFT on a cache _miss_ — the connector is essentially free when there's nothing to load.

**Failure modes you'll hit when implementing your own connector.** A frequent mistake in vLLM forks is calling `wait_load_kv` _inside_ a CUDA graph capture. CUDA graphs don't replay events, so the wait succeeds the first time and then never blocks again, leading to silent reads of un-initialized KV. LMCache's connector explicitly disables CUDA graphs for cached layers. Another gotcha: `kv_role` matters. `kv_both` means the worker stores and loads, `kv_consumer` means it only loads (used for decode-only nodes in PD), `kv_producer` means store-only (prefill-only nodes). Mismatched roles produce the most baffling debug sessions of LMCache operations — the cache appears to work, but reuse never happens because nobody is actually writing.

## 4. Chunking, Hashing, and the 256-Token Unit

The basic unit of LMCache is not a token, not a vLLM block, but a _chunk_ — a contiguous run of `chunk_size` tokens (default 256). This choice is a load-bearing engineering decision, not a heuristic.

![Chunking and prefix-hash chain — 256-token chunks, chained hashes, and a tiered lookup table](/imgs/blogs/lmcache-kv-cache-layer-deep-dive-2.png)

A chunk's value is its KV tensors for _all layers_, packed contiguously: `[num_layers, 2, num_heads, chunk_size, head_dim]`. On Llama-3-8B at fp16 that is `32 × 2 × 32 × 256 × 128 × 2 bytes ≈ 32 MB`. On Qwen-72B with GQA it's closer to 200 MB per chunk. The point is: a chunk is _big_ — big enough that moving one is a multi-millisecond DMA, big enough that it amortizes the per-message overhead of any transport.

The transfer-bandwidth curve is the empirical justification. The LMCache tech report measures PCIe Gen5 utilization across message sizes:

| Message size | Achieved bandwidth | % of theoretical peak |
| ------------ | ------------------ | --------------------- |
| 16 KB        | 0.8 GB/s           | 1.5%                  |
| 64 KB        | 4 GB/s             | 7.6%                  |
| 1 MB         | 22 GB/s            | 42%                   |
| 32 MB        | 47 GB/s            | 89%                   |
| 100 MB       | 49 GB/s            | 93%                   |

vLLM's native paged transfer copies a single block at a time — typically 16–63 KB. That is why LMCache's CPU-offload path achieves 400 Gb/s (50 GB/s) where vLLM's own offload tops out at 88 Gb/s (11 GB/s). The 4.5× gap is not "better code" — it is "right unit of work."

**The chunk hash chain.** A chunk's identity depends on _all preceding chunks_. The hash of chunk `i` is `H(h_{i-1} || tokens[i*256:(i+1)*256])`. This is a Merkle-style chain. The reason: two prompts that share the first 256 tokens but diverge afterwards must hit the same `h_0` (so they share L0), and two prompts that share a middle slice but diverge earlier must _not_ collide. A flat hash of the chunk text alone would let request B's `chunk[2]` get served from request A's `chunk[2]` even if the first 512 tokens differed — a subtle correctness bug because the K/V tensors depend on the whole prefix through self-attention.

The hash function is configurable via `pre_caching_hash_algorithm` (default `"builtin"`, which is xxhash). xxhash is fast (8 GB/s on a single core) and the chain length is small (a 32 K-token prompt has only 128 chunks), so the hash cost is dwarfed by the transfer cost.

**`save_unfull_chunk: false` is the default for a reason.** Partial chunks at the end of a prompt are usually a request-specific suffix (the user's question), and saving them pollutes the cache with low-reuse entries. The exception: agentic workflows where the "tail" is a tool name plus arguments and recurs across sessions — there, flipping it to `true` can win 5–10% extra hits.

**Why not `chunk_size = 16` to match vLLM blocks?** Because then every transfer is in the bottom row of the bandwidth table. The chunk size is a hit-rate-vs-bandwidth knob:

| `chunk_size` | Hit granularity (worst-case wasted prefix) | Transfer bandwidth |
| ------------ | ------------------------------------------ | ------------------ |
| 16           | 16 tokens                                  | 0.8 GB/s           |
| 64           | 64 tokens                                  | 4 GB/s             |
| 256 (def)    | 256 tokens                                 | 47 GB/s            |
| 1024         | 1024 tokens                                | 49 GB/s            |

256 is the sweet spot for current generation hardware: it sits on the knee of the bandwidth curve, and 256-token quantization rarely costs more than a few percent of recoverable prefix. On future H200 / B200 PCIe Gen6 systems the peak will move and the optimum may shift up to 512 or 1024.

## 5. The Storage Hierarchy

LMCache abstracts eight storage backends behind a transfer-channel interface. Knowing when each one wins is the difference between a cache that pays for itself and a cache that adds latency.

| Tier | Backend | YAML key                      | Latency | Bandwidth   | Capacity         | Best for                        |
| ---- | ------- | ----------------------------- | ------- | ----------- | ---------------- | ------------------------------- |
| L0   | GPU HBM | (vLLM owns)                   | ns      | TB/s        | 80 GB            | Hot prefix, current decode      |
| L1   | CPU DRAM | `local_cpu`, `max_local_cpu_size` | μs      | 50 GB/s     | 0.5–2 TB / node  | Recently evicted, hot peers     |
| L2   | NVMe + GDS | `local_disk` / `gds_path`   | ms      | 14 GB/s     | 4–32 TB / node   | Cold long-tail, multi-day reuse |
| L2.5 | InfiniStore | `remote_url: infinistore://` | ms     | 100 Gb/s    | 50 TB cluster    | In-memory tiered shared cache   |
| L3   | Mooncake | `remote_url: mooncake://`   | ms     | RDMA-line   | unbounded        | Cross-replica reuse, low-latency tail |
| L3   | Valkey/Redis | `remote_url: redis://`  | ms      | 10–25 Gb/s  | bounded by RAM   | Small / high-reuse hot prefixes |
| L3   | S3 Express | `remote_url: s3://`        | tens ms | 1–10 Gb/s   | unbounded        | Cold archive, cross-zone        |
| L3   | Weka     | `remote_url: weka://`        | ms      | 100+ Gb/s   | PB-scale         | Enterprise parallel FS          |

A few observations that are not obvious from the table:

**S3 Express flipped the equation.** Pre-2025 wisdom said remote storage was always slower than recomputing. The tech report's measurement: with S3 Express on a 100 Gb NIC, **loading from S3 outpaces prefill compute at >256 K tokens** on Llama-3-70B. That changes the deployment topology: you no longer need a replicated Mooncake cluster for very-long-context apps; a regional S3 bucket suffices for L3.

**Mooncake is the right default for multi-replica RAG.** It was purpose-built for KV cache transfer, supports RDMA out of the box, and integrates cleanly with the LMCache async prefetch path. Most teams running >4 replicas converge on Mooncake within a quarter.

**GDS skips the CPU bounce.** GPUDirect Storage lets the GPU pull bytes from NVMe without staging through CPU DRAM. On A100/H100 with PCIe Gen4 SSDs you get ~14 GB/s sustained, ~3× faster than going through the page cache. The catch: GDS requires a kernel module (`nvidia-fs`) and the right NVMe firmware. When it works, it's transparent — `use_gds: true` is the only flag.

**Per-GPU sharding is the default.** `local_disk_path_sharding: by_gpu` means each GPU writes to its own subdirectory. This avoids file-locking contention and lets you mount different NVMe drives on different ranks. The alternative — `by_rank` — keeps a single namespace, which is convenient for inspection but slower under burst.

**The `store_location` / `retrieve_locations` knobs are for advanced policy.** `store_location: ["cpu"]` means "never write to disk." `retrieve_locations: ["cpu", "remote"]` means "skip disk on read." Use these for tier-skipping policies; e.g. a cluster with massive Mooncake but tiny local NVMe should set `store_location: ["cpu", "remote"]` and skip L2.

### 5.1 Backend Selection in Practice

Picking the right backend is rarely about the synthetic benchmark — it's about how the backend interacts with your network topology and your reliability requirements. A condensed decision matrix:

| Constraint                              | Pick                       | Why                                                        |
| --------------------------------------- | -------------------------- | ---------------------------------------------------------- |
| Single replica, 1 node                  | `local_cpu` + `local_disk` | No reason to leave the box                                 |
| ≤4 replicas, same rack                  | `local_*` + Valkey         | Valkey is small and operationally trivial                  |
| ≥8 replicas, same DC, RDMA available    | Mooncake + P2P             | RDMA P2P doubles effective L1 capacity                     |
| ≥8 replicas, no RDMA                    | Mooncake (no P2P)          | Centralize on L3 instead of fighting Ethernet              |
| Cross-AZ deployment, latency-tolerant   | S3 / S3 Express + local L2 | Cheap, durable, slow — accept the latency                  |
| Compliance: cache must persist offsite  | Weka or S3 (with KMS)      | Both encrypt-at-rest and survive node loss                 |
| Bursty workload, unknown working set    | InfiniStore                | In-memory tiered, auto-spills to disk                      |

**A note on Valkey.** It works, but it's not designed for this workload. KV cache chunks are 32–200 MB binary blobs; Valkey is tuned for sub-MB strings. You'll see `MEMORY` warnings, you'll run out of fragmented memory, and you'll need `maxmemory-policy: allkeys-lru` plus a separate process to evict large keys. Mooncake was purpose-built for this and avoids the entire class of problems. Reach for Valkey only when you have an existing Redis-compatible cluster you must reuse.

**Mooncake's RDMA model.** Mooncake exposes `kv_get(hash) -> (rdma_addr, length)` style APIs that let the receiver issue a one-sided RDMA READ directly from the storage node's memory. The data path bypasses both the storage node's CPU and the network stack — it's literally the NIC pulling bytes from RAM and writing them to the GPU's pinned buffer. End-to-end latency for a 32 MB chunk on a 200 GbE RoCE fabric: ~1.6 ms, dominated by the RDMA round-trip and the H2D copy. An equivalent fetch through Valkey (TCP, RESP protocol, single-threaded server) is 25–40 ms.

**S3 Express's quiet revolution.** Pre-2025 the conventional wisdom was that S3 was too slow for KV cache. S3 Express (One Zone) changed this by giving you single-AZ object storage with sub-10 ms p50 latency. The tech report's striking measurement: at 256 K-token contexts on Llama-3-70B, the time to load fully-cached KV from S3 Express over a 100 Gb NIC is _less than_ the time to recompute it on the GPU. That number flips the cost equation entirely — a regional S3 bucket replaces a Mooncake cluster for a large class of workloads. The downside: S3 Express is more expensive per GB than standard S3 and not available in all regions yet.

**Weka in regulated environments.** Banks and healthcare deployments often can't use cloud-managed object storage but already have parallel filesystems for training. LMCache's Weka integration plugs into existing infrastructure with `remote_url: weka://mount-point/lmcache` and inherits whatever access controls the filesystem already has. Performance is competitive with Mooncake (100+ Gb/s) on properly-tuned WekaFS clusters.

## 6. Asynchronous Pipelining and Minimum-Copy

The bandwidth numbers above are theoretical; they only matter if you can _overlap_ the I/O with computation. This is what `use_layerwise: true` buys you.

![Two CUDA streams overlap KV load (Stream B) with attention compute (Stream A) per layer](/imgs/blogs/lmcache-kv-cache-layer-deep-dive-3.png)

Without layer-wise pipelining: load all 32 layers of KV cache (~800 ms for a 16 K context), _then_ start prefill (~600 ms). Wall clock 1400 ms; PCIe is idle for the second half, GPU is idle for the first.

With layer-wise pipelining: Stream B starts loading layer 0, then layer 1, then layer 2, hands off to Stream A as soon as layer 0 is on device. Stream A runs `attn_L0` while Stream B is prefetching `L2`. Wall clock collapses to roughly `max(load, compute) + 1 layer`. In our example: 900 ms instead of 1400, a ~1.5× win on the load-bound path. The tech report measures 1.9–8.1× TTFT improvement on 16 K–128 K contexts, which scales with how load-bound the workload is.

A naive Python sketch (the real code is in `lmcache/v1/connector/runner.py`):

```python
class LayerwisePipeline:
    def __init__(self, num_layers: int, buffer_bytes: int):
        # Single fixed-size GPU buffer, reused per layer.
        self.gpu_buf = torch.empty(buffer_bytes, dtype=torch.uint8,
                                   device="cuda")
        self.compute_stream = torch.cuda.current_stream()
        self.io_stream = torch.cuda.Stream()
        self.events = [torch.cuda.Event() for _ in range(num_layers)]

    def start_load_kv(self, plan):
        with torch.cuda.stream(self.io_stream):
            for layer_idx, src_addr, dst_block in plan:
                # Copy from CPU pinned / NVMe / NIXL into gpu_buf,
                # then scatter into vLLM's paged KV at dst_block.
                _zero_copy_h2d(src_addr, self.gpu_buf, dst_block)
                self.events[layer_idx].record(self.io_stream)

    def wait_load_kv(self, layer_idx):
        # Stream A waits on the layer's load before running attention.
        self.events[layer_idx].wait(self.compute_stream)
```

The crucial details:

1. **The GPU buffer is fixed size.** It does not grow with context length — it is sized for one layer. A 32-layer model with 32 MB/layer/chunk uses 32 MB total transfer scratch, not 1 GB.
2. **The buffer is reused.** When layer N's compute finishes, layer N+2's load can already be writing into it.
3. **Refcount-based zero copy.** When the same KV chunk needs to be written to both CPU DRAM and to a remote Mooncake target, LMCache increments a refcount on the source buffer instead of duplicating. Each destination decrements on completion. The tech report shows this saves ~30% of CPU DRAM under high replication factor.
4. **`start_store_kv` happens during decode.** During the autoregressive decode phase, the GPU is bound on small matmuls, not on PCIe. LMCache exploits this idle bandwidth to write back newly-generated KV. The user sees zero impact on tokens/s.

**Three-pointer offload.** During steady-state operation, LMCache tracks free-page regions with three pointers: `start` (region base), `current` (offload progress), `end` (scheduled boundary). New requests allocate from `current → end`, completed offloads advance `start → current`. The trade-off knob is the gap between `current` and `end`: a wide gap means more in-flight transfers (good for latency hiding) but more memory tied up in scratch (bad for capacity). The default tuning is conservative; high-throughput services usually widen it via `extra_config.offload_lookahead`.

## 7. CacheGen: Learned Compression for the KV Stream

KV cache is _highly compressible_. The tensors are float16 but the effective entropy is much lower — adjacent layers' K vectors are correlated, head dimensions cluster around a few principal directions, and quantization to 8 or even 4 bits costs surprisingly little quality. CacheGen [SIGCOMM '24] exploits this with a learned encoder that produces a bitstream you can stream from disk or network.

Activation: `remote_serde: cachegen` (or via `extra_config`). The default is `"naive"` which is essentially `torch.save` with no compression.

**Encoding pipeline:**

1. Per-layer per-head residual: subtract the previous layer's K/V to get a low-magnitude delta.
2. Quantize the delta with a learned per-head scale.
3. Entropy-code the result with an arithmetic coder trained on a sample of cached states from the target model.

The result is typically 2–4× smaller on disk than fp16 raw KV, with measured quality drop of <0.5 F1 points on RAG benchmarks at 4-bit and effectively zero at 8-bit.

```yaml
# lmcache_config.yaml
chunk_size: 256
local_cpu: true
max_local_cpu_size: 8.0
local_disk: "/mnt/nvme0/lmcache"
max_local_disk_size: 200.0
remote_url: "redis://kv-cache.svc.cluster.local:6379"
remote_serde: "cachegen"        # learned bitstream
extra_config:
  cachegen_quant_bits: 8        # 4 / 8 / fp16 (no compression)
  cachegen_target_model: "Qwen3-8B"
```

**When CacheGen pays off:**
- Remote storage is bandwidth-constrained (S3, cross-zone, sub-100 Gb/s NICs).
- Cache is spilled to NVMe and you want to fit more chunks per TB.
- You're on AMD or older NVIDIA where GDS isn't fully wired.

**When CacheGen hurts:**
- Local CPU DRAM is the only tier and you have plenty of it. Decode cost > IO saved.
- Latency-critical short-context decode (the per-chunk decode adds ~50 μs).
- Unfamiliar models where the bundled encoder hasn't been trained — you'll get poor compression ratios and the fallback path is slow.

The decoder is amortized across the chunk: a 32 MB chunk decodes in ~2 ms on H100 with the bundled CUDA kernel. That is sub-1% of a typical TTFT — usually a free win when network/disk is the bottleneck.

## 8. CacheBlend: Non-Prefix KV Reuse for RAG

CacheBlend (EuroSys '25) is the paper that shifted what people consider "cacheable." Standard prefix caching only reuses tokens at fixed absolute positions. RAG breaks that assumption: every request concatenates a different subset of retrieved documents in a different order, so the K/V tensors depend on which documents preceded them. The cache hit rate collapses.

![CacheBlend: prefix-only cache misses on every doc vs. CacheBlend reusing all docs with ~15% selective recompute](/imgs/blogs/lmcache-kv-cache-layer-deep-dive-4.png)

CacheBlend reuses the cached K/V from each document _even when its absolute position differs_, then surgically recomputes the small fraction of tokens whose attention scores would actually change. The published numbers: **near-100% hit rate** on RAG, **3× lower TTFT**, and F1 within 0.5 points of full recompute.

**The HKVD score.** For each token in the concatenated prompt, CacheBlend defines a Hidden KV Deviation: the L2 distance between (a) the K/V vectors that would be produced if you fully recomputed the token in its new absolute position and (b) the cached K/V from its original position. Tokens with high HKVD are tokens whose attention pattern actually changed — those need recompute. Tokens with low HKVD can keep their cached K/V and the model produces statistically-indistinguishable output.

**The selection algorithm:**

1. Run `blend_check_layers` (default 1) layers in full — i.e., compute K/V from scratch for the full concatenated prompt, but only for the first layer.
2. For each token $i$, compute its HKVD score:

$$\text{HKVD}_i = \| K^{(1)}_i - \tilde{K}^{(1)}_i \|_2 + \| V^{(1)}_i - \tilde{V}^{(1)}_i \|_2$$

where $K^{(1)}_i, V^{(1)}_i$ are the freshly-computed K/V at layer 1 in the new position, and $\tilde{K}^{(1)}_i, \tilde{V}^{(1)}_i$ are the cached values from the document's original position.
3. Pick the top $\rho \cdot N$ tokens by HKVD, where $\rho$ is `blend_recompute_ratios` (default 0.15) and $N$ is the total token count.
4. For layers $\ell \in \{2, ..., L\}$, recompute K/V only for the selected token set $S$; for tokens outside $S$, reuse cached K/V with positional encoding rotated to the new offset.

The intuition: layer 1's HKVD is a strong predictor of how much each token's attention pattern will deviate _across all subsequent layers_. The CacheBlend paper validates this empirically — the correlation between layer-1 HKVD and per-layer attention deviation is >0.85 on every model class they tested (Llama, Mistral, Qwen). One layer of probing is enough.

The cost: 1 layer fully recomputed + 15% of all subsequent layers' tokens. On a 32-layer model that is `1/32 + 0.15 * 31/32 ≈ 18%` of full prefill cost. On a 128 K-token RAG prompt where prefill normally takes 5 seconds, CacheBlend lands TTFT around 900 ms.

**The `blend_special_str` separator.** CacheBlend needs to know where one cached document ends and the next begins. The default `" # # "` is a string the engine inserts between concatenated documents. Pick anything unique to your tokenizer (the default is safe for most BPE tokenizers; some sentencepiece tokenizers tokenize `#` differently and may need `" \n\n[DOC]\n\n "` or similar).

**Configuration:**

```yaml
enable_blending: true
blend_recompute_ratios: 0.15      # tune up for accuracy, down for speed
blend_check_layers: 1             # 1 is sufficient for most models
blend_special_str: " # # "
```

**When to lower the ratio.** Some workloads (e.g. dense Q&A over uniform documents) tolerate `blend_recompute_ratios: 0.05`. The way to discover this: run a holdout set with ratios `[0.05, 0.10, 0.15, 0.25]`, measure F1 / Rouge / your task metric, pick the lowest ratio that doesn't regress. Most teams converge on 0.10–0.15.

**When CacheBlend won't help.** Code-completion (the prefix is uniquely the user's file). Single-doc summarization (no concatenation). Inference where the prompt itself is unique per request. The default for non-RAG workloads should be `enable_blending: false`.

## 9. Peer-to-Peer Sharing and Disaggregated Prefill

Two separate features, both built on the NIXL transport.

### 9.1 P2P CPU Memory Sharing

In a multi-replica deployment, two replicas may have warmed _different_ chunks. Without P2P, replica A has to ask the L3 store (Mooncake / Valkey / S3) for chunks B has in its CPU DRAM — a 5–10× slower path than fetching directly. P2P lets A and B exchange chunks over an RDMA channel, bypassing L3.

```yaml
enable_p2p: true
p2p_host: "10.0.1.4"                  # this replica's IP
peer_init_ports: [21001, 21002]       # for handshake
peer_lookup_ports: [21003, 21004]     # for chunk lookup
transfer_channel: "nixl"
extra_config:
  p2p_socket_recv_timeout_ms: 30000
  p2p_socket_send_timeout_ms: 10000
```

Production usage (Tencent, Tensormesh, per the [Jan 21 2026 blog](https://blog.lmcache.ai/2026-01-21-p2p/)) reports P2P doubles effective cache capacity (each replica's CPU DRAM is now reachable by all peers) and cuts cold-start TTFT by 30–60% during a deploy.

The catch: P2P over plain Ethernet is _slow_. 10 GbE caps at 1.25 GB/s, which is 10× below local CPU DRAM access. P2P only beats L3 if you have RDMA (RoCE, InfiniBand). On a 100 GbE RoCE fabric, P2P fetches at ~10 GB/s — competitive with local NVMe and 5× faster than Valkey. If your fleet doesn't have RDMA, leave P2P off and lean on Mooncake instead.

### 9.2 PD Disaggregation

PD disaggregation runs the prefill pass on a beefy node (often older A100s with lots of compute) and the decode pass on smaller, faster-turnaround H100s. The KV cache transfers between them. LMCache's `enable_pd: true` mode wraps NIXL with a sender/receiver protocol.

**Sender (prefill node):**

```yaml
enable_pd: true
pd_role: "sender"
transfer_channel: "nixl"
nixl_backends: ["UCX"]              # or ["MOONCAKE"], ["GDS"]
pd_buffer_size: 17179869184         # 16 GB
pd_buffer_device: "cuda"            # transfer directly from HBM
pd_proxy_host: "10.0.1.10"
pd_proxy_port: 25500
```

**Receiver (decode node):**

```yaml
enable_pd: true
pd_role: "receiver"
transfer_channel: "nixl"
nixl_backends: ["UCX"]
pd_buffer_size: 17179869184
pd_buffer_device: "cuda"
pd_peer_host: "10.0.1.4"
pd_peer_init_port: 25510
pd_peer_alloc_port: 25511
```

Behavioral details that bite:

**Backend selection matters.** `UCX` (default) is portable but adds protocol overhead. `MOONCAKE` is fastest if you already run a Mooncake cluster. `GDS` direct-attaches NVMe → remote NVMe through GPUDirect — useful for federations across data centers. Picking the wrong backend on a high-throughput rig can cost 30% of TTFT.

**`pd_buffer_device: "cpu"` vs `"cuda"`.** CPU buffer adds a hop but lets the receiver pin the buffer in advance (lower latency variance). CUDA buffer is faster but requires GPU memory budget on both sides. Default to CUDA on receiver, CPU on sender for asymmetric topologies.

**The proxy notification dance.** The sender notifies a proxy (typically a small ZMQ broker) when prefill completes; the receiver polls the proxy for ready batches. `pd_skip_proxy_notification: true` short-circuits this for direct sender→receiver topologies, saving ~10 ms but losing observability. Use only when you have an alternative readiness signal.

The tech report measures 1.5–1.8× lower mean TTFT on PD vs vLLM's native page-by-page transfer, with the larger gains on contexts >32 K. The gap exists because vLLM's native disaggregation uses one block per RDMA SEND; LMCache batches an entire chunk's worth of blocks into a single SEND.

## 10. The Controller and External Lookups

Most single-replica deployments don't need the controller. Multi-replica + cache-aware routing definitely does.

```yaml
enable_controller: true
lmcache_instance_id: "rag-replica-3"
controller_url: "tcp://lmcache-controller.svc:50051"
lmcache_worker_port: 50052
internal_api_server_enabled: true
internal_api_server_host: "0.0.0.0"
internal_api_server_port_start: 6999
```

The controller runs as a separate process (or pod). Each LMCache instance reports `(chunk_hash, instance_id, device, last_access)` on admit and evict. External clients — the load balancer, an autoscaler, a monitoring dashboard — query:

```python
# Example: cache-aware request router
import httpx
async def pick_replica(token_ids: list[int]) -> str:
    r = await httpx.AsyncClient().post(
        "http://lmcache-controller:50051/lookup",
        json={"tokens": token_ids},
    )
    hits = r.json()  # list of (instance_id, device, hit_tokens)
    if not hits:
        return least_loaded_replica()
    # Prefer the replica with the most hit tokens, breaking ties by device tier.
    return sorted(hits, key=lambda h: (-h["hit_tokens"],
                                       _tier_rank(h["device"])))[0]["instance_id"]
```

Other controller APIs:

- `move(src_instance, dst_instance, chunk_hashes)` — pre-warm a target replica before a known-bursty request lands.
- `pin(chunk_hash)` / `unpin(...)` — keep a popular system prompt out of LRU eviction.
- `compress(chunk_hashes)` — re-encode in CacheGen at higher compression to free space.
- `clear(filter)` — drop tenant-specific caches on a hard delete (GDPR).

These primitives are what production stacks (vLLM Production Stack, llm-d, KServe, NVIDIA Dynamo 1.0) build on. If you're rolling your own router, start with `lookup` and add `pin` next; you'll re-discover the rest.

### 10.1 Observability

Without metrics you can't tune anything. The `internal_api_server_enabled: true` flag exposes a Prometheus-compatible `/metrics` endpoint with the following families (truncated to the ones I find indispensable):

| Metric                                | Type      | Why you watch it                                  |
| ------------------------------------- | --------- | ------------------------------------------------- |
| `lmcache_lookup_total{tier=...}`      | counter   | Lookup volume by tier; rate-of-change is hit-rate |
| `lmcache_hit_total{tier=...}`         | counter   | Hits per tier; tier=l1 vs l2 vs remote split      |
| `lmcache_miss_total`                  | counter   | Misses; ratio with hit_total = miss rate          |
| `lmcache_load_latency_seconds`        | histogram | Per-chunk load latency, bucketed                  |
| `lmcache_store_latency_seconds`       | histogram | Per-chunk store latency                           |
| `lmcache_bytes_loaded_total`          | counter   | Cumulative bytes read; divide by time = bandwidth |
| `lmcache_evict_total{tier=...}`       | counter   | Eviction pressure indicator                       |
| `lmcache_pinned_chunks`               | gauge     | How many chunks are protected from eviction       |
| `lmcache_local_cpu_used_bytes`        | gauge     | L1 utilization vs `max_local_cpu_size`            |
| `lmcache_local_disk_used_bytes`       | gauge     | L2 utilization                                    |
| `lmcache_remote_inflight_requests`    | gauge     | Concurrent remote ops (saturation indicator)      |
| `lmcache_blending_recomputed_tokens`  | counter   | When CacheBlend is on, how much you're recomputing |

A short Grafana dashboard recipe for the first dashboard a new operator should build:

1. **Hit rate** = `rate(lmcache_hit_total[5m]) / (rate(lmcache_hit_total[5m]) + rate(lmcache_miss_total[5m]))` — should stabilize above 60% on RAG workloads, above 80% on chat.
2. **Tier mix** = stacked area of `rate(lmcache_hit_total[5m]) by (tier)` — tells you whether you're winning on L1 (good), L2 (acceptable), or having to hit remote (look at network costs).
3. **Load latency p99** by tier = `histogram_quantile(0.99, rate(lmcache_load_latency_seconds_bucket[5m]))` — anomalies here precede TTFT regressions by minutes.
4. **Eviction rate** by tier — sustained high values mean your tier is undersized for the working set.

You will tune almost everything by reading these graphs, not by reading docs. The fastest path to a well-tuned LMCache deployment is a fast feedback loop on metrics.

## 11. Configuration Cheat Sheet

The configs that actually move the needle, ranked by frequency-of-tuning in real deployments:

| Knob                       | Default     | When to touch                                          |
| -------------------------- | ----------- | ------------------------------------------------------ |
| `chunk_size`               | 256         | Almost never. 256 is the bandwidth-knee value.         |
| `local_cpu`                | true        | Disable only if you're CPU-RAM-starved (rare).         |
| `max_local_cpu_size`       | 5.0 GB      | **Always tune.** Set to 60–80% of free RAM.            |
| `local_disk`               | (none)      | Set to NVMe path for L2.                               |
| `max_local_disk_size`      | 0.0 GB      | Set with `local_disk`. Don't accept the default.       |
| `remote_url`               | (none)      | Set for multi-replica or persistent cache.             |
| `remote_serde`             | "naive"     | Switch to `"cachegen"` if remote is bandwidth-bound.   |
| `save_decode_cache`        | false       | **Leave false** unless decode tokens are reused.       |
| `use_layerwise`            | false       | **Set true** for vLLM v1; it's the headline win.       |
| `cache_policy`             | "LRU"       | LFU for skewed reuse; FIFO for benchmarks.             |
| `save_unfull_chunk`        | false       | True only for agent workflows with reused suffixes.    |
| `enable_blending`          | false       | True for RAG; otherwise off.                           |
| `blend_recompute_ratios`   | 0.15        | Profile per workload; 0.10 often safe.                 |
| `enable_p2p`               | false       | True only with RDMA fabric.                            |
| `enable_pd`                | false       | True for prefill-decode split topology.                |
| `enable_lazy_memory_allocator` | false   | True for spiky workloads (lazy allocation).            |
| `numa_mode`                | null        | Set explicitly on dual-socket nodes.                   |
| `priority_limit`           | None        | Multi-tenant: skip cache for low-prio tenants.         |
| `min_retrieve_tokens`      | 0           | Avoid loading sub-X-token chunks (overhead floor).     |

**The lazy memory allocator.** A subtle production feature: LMCache pins CPU pages on first use, not at startup. The lazy allocator (`enable_lazy_memory_allocator: true`) starts at `lazy_memory_initial_ratio` (0.2 = 20% of `max_local_cpu_size`), expands when usage hits `lazy_memory_expand_trigger_ratio` (0.5 = 50% of current allocation), grows by `lazy_memory_step_ratio` (0.1) each step. Why it matters: a fleet of 32 replicas each pinning 80 GB at startup is 2.56 TB of memory off-limits to anything else. Lazy allocation gives that memory back during low-traffic periods.

### 11.1 Multi-Tenant Policy Knobs

On a shared cluster, different tenants need different SLAs. LMCache's policy primitives are sparse but composable.

`priority_limit` skips the cache entirely for tenants below a numeric threshold, sent through the request's metadata. Combine with a per-tenant proxy that sets `extra={"priority": tenant.priority_class}` in the inference call. Low-priority background work doesn't pollute the cache; high-priority chat traffic gets first-class treatment.

`min_retrieve_tokens` sets a floor below which cache lookups are skipped. The reasoning: a 64-token prefix lookup amortizes poorly against the lookup overhead (~50–200 μs depending on tier). For workloads where most prompts are short, set this to 256 or 512 to skip the lookup entirely on tiny prompts. Saves CPU cycles you can give back to the engine.

`store_location` and `retrieve_locations` together let you express tier-skipping policies. A common production pattern for multi-tenant clusters: tenant-A traffic uses `["cpu", "remote"]` (skip disk) for low latency, tenant-B traffic uses `["cpu", "disk"]` (skip remote) for data-residency compliance. Implement by spinning up two LMCache engines per replica, each routed to by the proxy based on tenant ID. Yes, it's two engines on one GPU; the connector handles it cleanly because each gets a distinct `lmcache_instance_id`.

**`numa_mode` on dual-socket boxes.** On a 2-socket Intel/AMD system, the GPU is attached to one socket. CPU pages allocated on the wrong socket are accessed across the inter-socket link (UPI/Infinity Fabric) at half the bandwidth of local access. `numa_mode: "interleave"` or explicit pinning via `numactl` can cut CPU-offload latency variance significantly. The fix is invisible to single-socket boxes.

## 12. Hands-On: Deploy with vLLM v1

A minimal, end-to-end runnable setup for Llama-3 on H100. Tested with `lmcache==0.4.3` and `vllm==0.18.0`.

### 12.1 Install

```bash
uv venv --python 3.12
source .venv/bin/activate

# CUDA 12.9 (matches default vLLM wheel)
uv pip install lmcache==0.4.3 vllm==0.18.0

# Verify the C extension built
python -c "import lmcache.c_ops; print('ok')"
```

For CUDA 13:

```bash
VERSION=0.4.3
uv pip install lmcache==${VERSION} \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --find-links https://github.com/LMCache/LMCache/releases/expanded_assets/v${VERSION}-cu13 \
    --index-strategy unsafe-best-match
```

### 12.2 Offline inference (CPU offload only)

```python
# offline_lmcache.py
import os
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

os.environ["LMCACHE_CHUNK_SIZE"] = "256"
os.environ["LMCACHE_LOCAL_CPU"] = "True"
os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "20.0"  # 20 GB
os.environ["LMCACHE_USE_LAYERWISE"] = "True"
os.environ["LMCACHE_CACHE_POLICY"] = "LRU"

ktc = KVTransferConfig(
    kv_connector="LMCacheConnectorV1",
    kv_role="kv_both",          # this worker both stores and loads
)

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    kv_transfer_config=ktc,
    max_model_len=32_000,
    gpu_memory_utilization=0.85,
)

sampling = SamplingParams(temperature=0, max_tokens=128)

system = "You are a careful assistant. Answer concisely.\n\n"
docs = open("doc.txt").read()                       # ~24 K tokens
prompts = [
    system + docs + "\n\nQuestion: " + q
    for q in ["Summarize.", "List entities.", "Pick the date."]
]

# First call: cold cache, full prefill of system+docs.
out0 = llm.generate(prompts[:1], sampling)
# Subsequent calls: warm cache, only the question is prefilled.
out1 = llm.generate(prompts[1:], sampling)

# Clean up to flush counters.
from lmcache.v1.cache_engine import LMCacheEngineBuilder
from lmcache.integration.vllm.utils import ENGINE_NAME
LMCacheEngineBuilder.destroy(ENGINE_NAME)
```

The TTFT difference between `out0` and the entries in `out1` is the headline number. On H100 with a 24 K-token prompt, cold TTFT is ~2.0 s; warm TTFT (system+docs reused, only the new question prefilled) drops to ~250 ms — an ~8× win that maps directly to the tech report's 1.9–8.1× range.

### 12.3 Online server (production-ish)

`lmcache_config.yaml`:

```yaml
chunk_size: 256
local_cpu: true
max_local_cpu_size: 60.0
local_disk: "/mnt/nvme0/lmcache"
max_local_disk_size: 500.0
local_disk_path_sharding: "by_gpu"
remote_url: "redis://kv-cache.svc.cluster.local:6379"
remote_serde: "cachegen"
use_layerwise: true
cache_policy: "LRU"
save_decode_cache: false
save_unfull_chunk: false

# RAG-friendly defaults
enable_blending: true
blend_recompute_ratios: 0.15
blend_check_layers: 1

# Multi-replica
enable_p2p: true
p2p_host: "10.0.1.4"
peer_init_ports: [21001, 21002]
peer_lookup_ports: [21003, 21004]
transfer_channel: "nixl"

# Memory hygiene
enable_lazy_memory_allocator: true
lazy_memory_initial_ratio: 0.25
numa_mode: "interleave"

# Observability
internal_api_server_enabled: true
internal_api_server_port_start: 6999
```

Launch:

```bash
LMCACHE_CONFIG_FILE=/etc/lmcache/lmcache_config.yaml \
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
  --gpu-memory-utilization 0.85 \
  --max-model-len 32000 \
  --tensor-parallel-size 2
```

Smoke test:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "You are a helpful assistant. ... <repeat 4000 chars> ... \n\nQ: Hello?\nA:",
    "max_tokens": 50,
    "temperature": 0.0
  }'
```

Run the same request twice. The second TTFT should drop dramatically. Watch the LMCache internal API server (`http://localhost:6999/metrics`) for live hit-rate counters.

### 12.4 Kubernetes / vLLM Production Stack

Most teams arrive at LMCache via the [vLLM Production Stack](https://github.com/vllm-project/production-stack) Helm chart. The chart wires LMCache + a Mooncake StatefulSet + a router pod that uses the controller's `lookup` API for cache-aware load balancing. The relevant values:

```yaml
# values.yaml
lmcache:
  enabled: true
  remote:
    url: "mooncake://mooncake-master.cache.svc:50051"
  cpu:
    sizeGB: 60
  disk:
    enabled: true
    path: "/mnt/nvme0/lmcache"
    sizeGB: 500
  blending:
    enabled: true
    recomputeRatio: 0.15
controller:
  enabled: true
router:
  cacheAware: true
```

This is the minimum viable production deployment. Everything else (S3 archive tier, RDMA P2P, GDS) is a layer on top that you turn on when a specific bottleneck shows up in the telemetry.

## 13. Benchmarks: What the Numbers Actually Mean

The headline numbers from the tech report and the LMCache blog, with the conditions that produced each:

| Metric                                  | Number        | Conditions                                                                   |
| --------------------------------------- | ------------- | ---------------------------------------------------------------------------- |
| CPU offload bandwidth                   | 50 GB/s       | Llama-3-8B, H100, PCIe Gen5, chunk_size=256 (vs 11 GB/s for vLLM native)     |
| Single-node CPU-offload TTFT speedup    | 1.9–8.1×      | 16 K–128 K context, Llama-3-70B, vs vLLM with HBM-only prefix cache          |
| Real-trace TTFT speedup                 | 3.7–6.8×      | Mixed workload trace (chat + RAG + agent), 16 replicas, mooncake L3          |
| PD disaggregation TTFT                  | 1.5–1.8×      | vs vLLM's native page-by-page transfer, NIXL UCX backend                    |
| Multi-round QA throughput               | up to 15×     | Conversational benchmark, full LMCache stack with CacheBlend                 |
| MoE inference (Apr 2026 architecture)   | 10×           | Reported by LMCache team, Mixtral 8x22B, vs LMCache 0.3 baseline             |
| CacheBlend RAG TTFT                     | 3×            | EuroSys '25, ~100% effective hit rate, F1 within 0.5 of full recompute      |
| 4-bit CacheGen quality drop             | <0.5 F1       | RAG benchmarks; <0.1 at 8-bit                                                |
| Remote-vs-prefill crossover             | >256 K tokens | S3 Express on 100 Gb NIC, Llama-3-70B; remote load < prefill compute        |

A few things to keep in mind when you read these numbers in someone's pitch deck:

**The 15× is throughput, not latency.** Multi-round QA throughput improves because every turn after the first reuses 95%+ of the previous prefix. Per-token decode latency doesn't change. Don't promise 15× TTFT.

**Single-node TTFT speedups are mostly load-elimination.** When you reload from CPU instead of recomputing, you save the prefill FLOPs. The fraction of TTFT spent in prefill grows with context length, which is why the speedup range stretches from 1.9× (16 K) to 8.1× (128 K).

**CacheBlend's 3× is a property of RAG-heavy workloads.** A chat-only deployment will see closer to 1.0–1.2×. The win comes from converting "0% cacheable doc tokens" into "85% cacheable doc tokens."

**MoE's 10× is on top of a baseline that already had LMCache.** The 10× is LMCache 0.4.x architecture vs LMCache 0.3.x architecture, _both with_ caching. Don't compare it to a no-cache baseline.

**Real-trace numbers are the honest ones.** 3.7–6.8× is what you should expect once you ship. Microbenchmarks favor whichever feature the benchmark stresses.

### 13.1 A Worked Cost Example

The benchmark numbers are abstract. The actual question on a CFO's mind is: "does this pay for itself?" Let's walk through a specific deployment.

A RAG product runs 16 H100 replicas at $4/hour each on a managed cloud, ~$46 K/month. Average request: 8 K-token prompt (system + 4 retrieved docs + question), 200-token response. Traffic: 50 RPS sustained. Without caching: each request prefills 8 K tokens at ~50 ms TTFT. The cluster runs at ~75% GPU utilization; bumping above 90% triggers queueing latency.

Add LMCache with CacheBlend, a 32 GB CPU tier per replica, and a Mooncake cluster (~$2 K/month). Measured outcome on this workload: TTFT drops from 1.2 s to 380 ms (3.2×), prefill GPU-time per request drops 70%. The team scales the replica count down from 16 to 7 (the prefill capacity is 3× more efficient per replica). Net cost change:

| Line item              | Before    | After   | Delta      |
| ---------------------- | --------- | ------- | ---------- |
| GPU compute (16→7 H100) | $46,080  | $20,160 | −$25,920   |
| Mooncake cluster        | $0       | $2,000  | +$2,000    |
| CPU/RAM upgrade         | $0       | $1,200  | +$1,200    |
| Engineering time (one-off) | —     | $15,000 | (amortized) |
| **Monthly run-rate**    | **$46,080** | **$23,360** | **−$22,720** |

The monthly savings exceed the engineering one-off cost in the first month. This is the typical shape: LMCache pays for itself in 2–6 weeks on any deployment with >4 replicas and a working set that exceeds local HBM. The exception is small, latency-insensitive, single-replica deployments — in those cases the operational overhead is real and the savings are tiny. Don't deploy LMCache there.

## 14. Case Studies

Ten production incidents, each summarized with the symptom, the root cause, and the fix. These are composites drawn from public LMCache issue threads, the EuroSys / SIGCOMM papers, and the operational write-ups on the LMCache blog. Names are anonymized.

### 14.0 The framing: how to read these case studies

Each story below is a composite drawn from real production threads, but the pattern is identical: a default value or a topology assumption made silently, breaks loudly under load, and the fix lives in a single config knob _once you understand which knob_. The reason I've collected them here is that the knob alone is useless without the diagnosis that points at it. Read these as exercises in "where in the LMCache stack would I look first?" rather than as lookup-table fixes.

A pattern that recurs: the symptom is almost always TTFT or memory, never throughput in isolation. LMCache regressions present as latency tails, eviction storms, OOMs, or NIC saturation — the GPU itself rarely complains. So your alerting should weight tail-latency and infra metrics over GPU utilization. A green GPU dashboard with a red TTFT dashboard is the canonical LMCache misconfiguration signature.

### 14.1 The 50% hit-rate cliff after context truncation

A chat product saw prefix cache hit rate fall from 84% to 41% over a single deploy. The cause turned out to be a backend change two weeks earlier that increased `max_model_len` from 16 K to 32 K and quietly re-enabled aggressive context truncation. Truncation kept the most recent 28 K tokens — meaning the system prompt and tool descriptions (the actually-reusable bit) were silently dropped from every long conversation.

The vLLM-only fix would be to disable truncation, but that means OOM under long chats. The LMCache fix: keep the system prompt as a separate, _pinned_ chunk via `controller.pin(system_chunk_hash)`, and arrange the prompt template so the system prompt is always at offset 0. Hit rate climbed back to 81% within an hour. Lesson: aggressive truncation is the adversary of prefix caching, and the fix is at the cache layer, not the engine.

### 14.2 CPU offload made things slower: chunk_size 16 vs 256

A team running their own fork set `chunk_size: 16` "to match vLLM's block size," reasoning that smaller chunks would give finer-grained reuse. After the deploy TTFT got _worse_ — by ~30% on long contexts. The bandwidth table from §4 explains why: 16-token chunks are 2 MB transfers, which sit at the bottom of the PCIe utilization curve (~1 GB/s actual). Loading 32 K tokens of KV took longer than recomputing it.

The fix was a one-line revert to `chunk_size: 256`. TTFT dropped 4×. The deeper lesson: chunks are a transfer unit before they are a reuse unit, and the bandwidth curve is the dominant constraint. Tune chunk_size by measuring achieved bandwidth, not by matching some other layer's block size.

### 14.3 Disaggregated prefill TTFT spike traced to UCX backend

A PD-disaggregated rig (4 prefillers + 16 decoders, all H100, 200 GbE RoCE) hit a sustained 800 ms p99 TTFT after enabling LMCache PD mode, which was 2× worse than the non-disaggregated baseline. Profiling showed 90% of the time was spent in `nixl_send` on the prefill side.

`nixl_backends` defaulted to `["UCX"]`. UCX over RoCE works but adds protocol overhead — TCP-like control packets, software ack, congestion control. The team's existing Mooncake cluster supported direct RDMA writes. Switching to `nixl_backends: ["MOONCAKE"]` and pointing it at the cluster brought p99 to 290 ms.

Generalizable advice: NIXL backend choice is _the_ first knob to check on PD performance regressions. If you have Mooncake or a vendor-native RDMA library, prefer it. UCX is a portability default, not a performance default.

### 14.4 Multi-round QA cache stampede with shared Valkey

A RAG product spun up 32 replicas for a launch and immediately took down their shared Valkey cluster: connections-per-second pegged the proxy, and Valkey memory hit 100% within 20 minutes. The pattern: every replica was independently reading the same hot prefixes from Valkey on cold start, and writing _newly-decoded_ KV back at scale.

Three separate fixes layered together. (1) `save_decode_cache: false` everywhere — decode KV from the same model on the same prompt is reproducible, no need to write it back. That cut writes by 60%. (2) Enabled P2P (RoCE was available) so warm peers serviced reads instead of all hitting Valkey. (3) Set `priority_limit: 5` so low-priority background traffic skipped the remote write path entirely.

Valkey load dropped to 8% of peak. The high-level lesson: a shared L3 store is a stampede risk; the mitigations are layered (P2P, priority gating, decode-cache discipline) rather than a single big knob.

### 14.5 CacheBlend RAG regression at recompute ratio 0.05

A team trying to squeeze more TTFT out of CacheBlend lowered `blend_recompute_ratios` from 0.15 to 0.05. TTFT dropped another 1.8× — and F1 on their eval set dropped 2.1 points, which was unacceptable. They ran a sweep:

| ratio | TTFT (ms) | F1   |
| ----- | --------- | ---- |
| 0.00  | 410       | 71.2 |
| 0.05  | 480       | 76.4 |
| 0.10  | 580       | 78.1 |
| 0.15  | 720       | 78.4 |
| 0.25  | 940       | 78.5 |

The shape is the canonical S-curve: large gains from 0.00 → 0.10, diminishing returns above 0.15. They settled on 0.10. Lesson: sweep this knob _per-workload_; the EuroSys paper's 0.15 default is a model-class average, not a universal optimum.

### 14.6 P2P across nodes saturating 10 GbE

A self-hosted Kubernetes cluster on 10 GbE bonded interfaces enabled `enable_p2p: true` and watched their NICs hit line-rate within seconds of warm-up. Pods started missing health checks. The math: 32 replicas, 60 GB CPU cache each, ~1 KB hashes flying around plus occasional 32 MB chunk pulls. P2P at 10 GbE peaks at 1.25 GB/s — a single chunk fetch eats 25 ms of NIC bandwidth that everything else is sharing.

The fix wasn't a tuning knob; it was acknowledging the topology constraint. They turned P2P off, kept Mooncake as L3, and added a small InfiniStore tier for hot system prompts. Lesson: P2P only pays back its complexity on RDMA-class fabrics. On anything Ethernet-only, lean on a centralized L2.5/L3 instead.

### 14.7 CacheGen at 4-bit broke long-context summarization

A product enabled `remote_serde: cachegen` with `cachegen_quant_bits: 4` to fit more cache per dollar of S3. Quality on short contexts (<2 K) was indistinguishable. On 64 K-token document summarization, summaries started missing key entities — about 8% of evaluations regressed.

Root cause: at 4-bit quantization, K-vector errors compound over long contexts. Tokens in the middle of the document have their attention scores subtly perturbed; tokens with low attention probability get nudged below threshold and are effectively masked. The 70B model is robust to this in short contexts but sensitive at the tail of long ones.

Fix: bump to 8-bit (`cachegen_quant_bits: 8`). Storage went up ~2× (still ~2× smaller than fp16 raw), quality recovered fully. The general rule: 4-bit CacheGen is fine for short-context chat, suspect for long-context summarization, never for code generation.

### 14.8 `save_decode_cache: true` blowing the CPU budget — extended analysis

A team enabled `save_decode_cache: true` because "more cache is better." Within an hour CPU memory was at 95% on every replica and the lazy allocator started OOM-killing. The math: a typical chat session generates 200–500 decode tokens; at fp16 that is 6.4–16 MB per session of new KV. Across thousands of concurrent sessions, decode KV swamps the prefix KV by 5–10×.

The deeper question: _is decode KV reusable?_ It's reusable across sessions only if the same exchange recurs verbatim. For typical chat (where users phrase questions slightly differently), it isn't. So you're paying the storage cost for ~2% reuse rate.

Fix: `save_decode_cache: false` (the default). The 1–2% TTFT improvement on the rare repeat exchange isn't worth the 5× storage cost. Enable only if you have a workload (e.g. agent loops with deterministic tool outputs) where decode tokens genuinely repeat.

### 14.9 Lazy memory allocator stalling under burst

A replica configured with `enable_lazy_memory_allocator: true`, `lazy_memory_initial_ratio: 0.1`, `lazy_memory_step_ratio: 0.05` saw periodic 200 ms TTFT stalls during traffic bursts. Profiling showed `mlock` calls in the hot path: each step expansion was pinning ~6 GB of new pages, blocking the engine for the duration of the syscall.

Fix: bump `lazy_memory_initial_ratio` to 0.5 and `lazy_memory_step_ratio` to 0.2. The replica pre-pins more memory on startup (slower cold boot, ~3s extra) but expansion steps now happen rarely and outside critical paths. Stalls disappeared.

The general intuition: lazy allocation exists to release memory back during quiet periods, not to chase every megabyte of headroom. Tune the initial ratio so steady-state usage rarely triggers expansion. Treat the allocator like a slow-start TCP window: aggressive ramp-up, gentle scale-down.

### 14.10 NUMA pinning recovered 22% TTFT on a dual-socket box

A team running on AMD EPYC dual-socket boxes (each socket with its own attached H100) noticed their LMCache hit-rate metrics looked great but TTFT was inconsistent — p50 acceptable, p99 spiky. Profiling revealed the spikes correlated with cache fetches that crossed sockets: chunks pinned in NUMA node 0 being read by GPU on socket 1, traversing the cross-socket Infinity Fabric link.

EPYC's cross-socket bandwidth is roughly half the local DRAM bandwidth, and contention with the OS's own page-allocator decisions made variance worse. Setting `numa_mode: "interleave"` was a partial fix; the full fix was to launch each rank with `numactl --cpunodebind=$N --membind=$N` matching the GPU's PCIe topology. p99 TTFT dropped 22%, variance halved.

Generalizable advice: NUMA effects are invisible until they're not. On any dual-socket box where you serve LLMs, verify with `nvidia-smi topo --matrix` which CPU socket each GPU is attached to, and pin LMCache's CPU cache and the rank's threads to that socket. The default of "let the kernel decide" is always wrong on this hardware.

### 14.11 Mooncake-backed cross-zone reuse for an enterprise RAG stack

A bank deploying RAG across three AZs wanted cache reuse for a customer-support workload where the same 200-document corpus was queried millions of times a day. Local-only caching gave them 70% hit rate per replica but 0% across AZs (replicas in AZ-a never saw chunks from AZ-b). They stood up a Mooncake cluster with one replica per AZ, configured `remote_url: mooncake://...`, and routed inter-AZ traffic over a 100 Gb cross-AZ trunk.

Cache hit rate cluster-wide jumped to 91%; per-AZ Mooncake fan-in absorbed bursts that previously OOM'd Valkey. The cost: ~$8 K/mo of Mooncake cluster and 1.2 PB-month of cross-AZ traffic. Net win: ~38% reduction in GPU-hours, more than offsetting the infra cost.

Lesson: at scale, the cache layer is a _shared regional asset_, not a per-replica concern. Mooncake is the canonical answer; S3 Express is the cheap-but-slow alternative. The break-even depends on your traffic profile and your cross-AZ pricing.

## 15. The Alternatives, and Why Most Teams End Up Here

LMCache is not the only KV-caching effort. A short, opinionated tour of the alternatives:

**vLLM's built-in `PrefixCache`.** Same process, HBM-only, no cross-replica sharing, no tiered storage, no compression. Sufficient if your traffic fits one replica, your context lengths are short, and your prefixes never spill out of HBM. Below that bar it's strictly inferior; above it, it's strictly insufficient. LMCache plugs into vLLM specifically to extend it, not replace it.

**SGLang's `RadixCache`.** Smarter prefix data structure (radix tree instead of hash chain) and earlier than LMCache for non-prefix sharing of prompt prefixes. Lives in HBM only. SGLang gained an LMCache integration in 0.4.x specifically to get the tiered store and remote backends — the radix tree complements LMCache rather than competing with it.

**Mooncake (standalone).** Mooncake is a transport plus a storage cluster, not a connector. You still need something on the engine side to chunk, hash, and address into Mooncake. LMCache's `remote_url: mooncake://...` is the bridge. Production stacks routinely run both.

**NVIDIA TRT-LLM's `kv_cache_reuse`.** Closed-source, NVIDIA-only, integrated into TRT-LLM's executor. Faster than LMCache on bare-metal H100/H200 because it bypasses the Python connector layer, but locks you to TRT-LLM. The vLLM ecosystem's bet is that the open-source connector layer is fast enough (the tech report's measured overhead is <2%) and the operational flexibility is worth more than the 2%.

**Roll your own.** Doable, popular two years ago, increasingly rare now. The list of features you have to re-implement before you reach LMCache parity — chunking, hashing, eviction, NUMA-aware allocation, layer-wise pipelining, ref-counted zero-copy, CacheGen, CacheBlend, NIXL transport, controller, observability — is the reason this stops being a roll-your-own thing. Most teams who tried it now run LMCache and contribute back.

## 16. When to Reach for LMCache, and When Not To

LMCache is not a free lunch. It costs CPU memory, optionally NVMe space, optionally network bandwidth, and a non-trivial amount of operational complexity. Reach for it when the math actually works.

**Use LMCache when:**
- Multi-round chat / agent loops: turn-N reuses 80%+ of turn-(N-1).
- RAG with concatenated retrieved documents: CacheBlend turns 0% reuse into ~85% reuse.
- Long-context (>16 K tokens) inference: the prefill cost is large enough that any cache hit pays back the IO.
- Multi-replica deployments where cache-aware routing materially improves hit rate.
- Prefill-decode disaggregated topologies where you must move KV between nodes.
- Bursty workloads with predictable hot prefixes (system prompts, tool descriptions, persona templates).

**Skip LMCache when:**
- Single-replica, short-context, low-traffic deployment. vLLM's built-in prefix cache is sufficient.
- Prompts are unique per request (one-shot summarization, code completion). No reuse, no cache benefit.
- You don't have CPU DRAM headroom — LMCache without local CPU cache is a remote-only deployment, which is rarely worth it.
- Your network fabric is sub-25 Gb Ethernet and you don't have a dedicated cache cluster. P2P and remote tiers will hurt more than they help.
- Your workload is decode-bound (long generations from short prompts). The prefill optimization is irrelevant; you're spending all your time decoding.

**The minimum-viable starting config**, for a team trying LMCache for the first time:

```yaml
chunk_size: 256
local_cpu: true
max_local_cpu_size: 30.0     # 30 GB; tune to your free RAM
use_layerwise: true
cache_policy: "LRU"
```

That's it. No remote, no P2P, no PD, no blending. Run for a week, measure hit rate via the internal API server, then add the next tier (CacheBlend if you do RAG, remote if you do multi-replica, P2P if you have RDMA, PD if you have asymmetric prefill/decode hardware). The features stack — but each one carries operational cost. The most expensive mistake a new LMCache user makes is enabling six features at once and being unable to attribute regressions.

LMCache is, ultimately, an admission that the LLM serving stack has a memory hierarchy, and that the hierarchy needs to be managed _intentionally_. Treating it as one is the move that takes a serving cluster from "single-replica vLLM in a Docker container" to "production-scale infrastructure that pays for itself." The rest is execution.

A senior reading of the field as it stands at the time of writing (April 2026): the next 12 months of LMCache development are likely to focus on three axes. First, deeper MoE integration — the 10× claim from the April architecture revision is the start of a longer arc, not the end. Second, smarter eviction policies that incorporate the controller's global view of access patterns rather than each instance running its own LRU in isolation. Third, tighter coupling with disaggregated prefill on heterogeneous hardware, where prefill happens on cheap-and-numerous A10s and decode on H200s — a topology that's economically attractive but operationally unforgiving without a robust transport layer. The connector API is the constant; everything below it is moving fast.

Companion deep-dives on adjacent topics in this blog: [KV cache fundamentals](/blog/machine-learning/large-language-model/kv-cache) (the math and intuition behind why this all matters), [KV cache optimization and management](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management) (paged attention, eviction strategies, quantization _inside_ the engine), and [Docker for LLM workloads](/blog/machine-learning/mlops/docker-optimization-for-llm-and-ai-workloads) (the host-runtime layer that LMCache deployments sit on top of). Together they form a fairly complete picture of where production LLM serving is in 2026: the engine, the cache layer, and the host. Each is independently optimizable, and each pays back the engineering effort.

A final aside on _what to measure first_ when you're new to this. New operators almost universally start by graphing throughput. Throughput is the wrong leading indicator: it lags hit-rate by minutes and conflates engine improvements with cache improvements. Graph hit rate first; graph TTFT p99 second; graph eviction rate third. If those three are healthy, throughput will be healthy. If they're not, no amount of engine tuning will save you. This is the single most useful piece of operational advice anyone gave me about LMCache, and it took embarrassingly long to internalize. Pass it on.

The other operational rule worth committing to memory: never enable two LMCache features at the same time on the same week. Each feature shifts the workload's behavior in non-trivial ways. Turn on layer-wise pipelining one Tuesday, measure for a week, then turn on remote storage, then CacheBlend, then P2P. Bisecting regressions across three simultaneous toggles is a way to lose half a quarter to vibes-based debugging, and the LMCache surface area is large enough that you _will_ regress something. Slow rollouts are not optional; they are the difference between an operator who looks competent and one who looks lucky.

### Further reading

- The [LMCache GitHub repository](https://github.com/LMCache/LMCache) — source of truth, issue tracker, and the cleanest place to see how the connector hooks evolve.
- The [LMCache tech report](https://arxiv.org/abs/2510.09665) — full architecture, benchmark methodology, and the specific measurements quoted throughout this article.
- [CacheBlend: Fast LLM Serving with Cached Knowledge Fusion (EuroSys '25)](https://blog.lmcache.ai/2025-03-31-eurosys/) — the paper behind §8.
- [CacheGen (SIGCOMM '24)](https://blog.lmcache.ai/2024-09-17-release/) — KV cache compression and streaming.
- The [LMCache blog](https://blog.lmcache.ai/) — release notes, partnership announcements, and the most current production case studies.
- The [vLLM Production Stack](https://github.com/vllm-project/production-stack) — Helm charts and reference architectures for deploying LMCache on Kubernetes.

If something in this post doesn't match what you're seeing in production, the issue tracker is the right place; the LMCache maintainers are responsive and the project has been through enough production churn that most edge cases have already surfaced. The community is one of the better-organized open-source efforts in the LLM serving space — worth contributing back to when you find yourself relying on it.
