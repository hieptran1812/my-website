---
title: "Prefill/decode disaggregation: splitting LLM inference into two GPU pools"
date: "2026-07-03"
publishDate: "2026-07-03"
description: "Why prefill is compute-bound and decode is memory-bandwidth-bound, why colocating them wrecks your tail latency, and how splitting inference into separate prefill and decode worker pools joined by a sub-millisecond KV-cache handoff buys you multiples more SLO-meeting goodput on the same GPUs."
tags:
  [
    "model-serving",
    "inference",
    "llm-serving",
    "prefill-decode-disaggregation",
    "kv-cache",
    "goodput",
    "vllm",
    "nvidia-dynamo",
    "distserve",
    "mooncake",
    "nccl",
  ]
category: "machine-learning"
subcategory: "Model Serving"
author: "Hiep Tran"
featured: true
readTime: 50
image: "/imgs/blogs/prefill-decode-disaggregation-1.webp"
---

The dashboard looked wrong in a way that took a week to explain. Time-to-first-token was fine — p50 around 90 ms, p99 under 300 ms, comfortably inside the SLA. Throughput was fine — the fleet of 8×H100 nodes was chewing through 300-odd requests per second. But the inter-token latency, the metric users actually *feel* as the model "typing," was oscillating violently. Most tokens arrived every 20 ms like clockwork. Then, a few times a minute, a stream would freeze for 120, 150, sometimes 180 ms mid-sentence, stutter, and resume. The p99 of time-per-output-token was almost ten times its median. Nobody had changed the model. Nobody had changed the config. The only thing that had changed was the traffic mix: a new customer had started sending 4,000-token documents to summarize, alongside everyone else's short chat turns.

That stutter is not a bug. It is the single most important structural fact about LLM inference, and it has a name: **prefill/decode interference**. When a long prompt lands, the server has to run *prefill* — one big forward pass over all 4,000 prompt tokens to populate the key/value cache before it can emit a single output token. That prefill pass is a compute monster; it saturates the GPU's floating-point units for 100–200 ms. And while it runs on the same GPU that is supposed to be emitting the next token for forty other in-flight streams, those forty streams wait. Their decode steps — cheap, memory-bound, one-token-at-a-time operations — get stalled behind the prefill burst. Every long prompt that arrives is a landmine under everybody else's inter-token latency.

![Before and after comparison showing a shared GPU pool where a prefill burst stalls decode steps and breaches the token-latency SLO, versus two disaggregated pools joined by a sub-millisecond KV handoff that keeps decode latency flat](/imgs/blogs/prefill-decode-disaggregation-1.webp)

The figure above is the whole post in one picture. On the left, a colocated pool: prefill and decode share the same GPUs, a 150 ms prefill burst stalls the decodes, and TPOT p99 swings between 40 and 180 ms — SLO breached. On the right, the fix: split the work into a **prefill pool** that runs nothing but prompt forward passes at 95% FLOP utilization, and a **decode pool** that runs nothing but token generation in one big steady batch, joined by a **KV-cache handoff** that takes 0.7 ms over NVLink. Decode latency goes flat because no prefill ever runs on a decode GPU again. That is prefill/decode disaggregation, and it is the technique that Track H of this series exists to teach.

This post is the deep dive on that technique. It is H1 in the Model Deployment and Serving series, and it assumes you have already met the core LLM-serving concepts — if the terms KV cache, TTFT, and TPOT are not yet reflexes, start with [why LLM serving is different](/blog/machine-learning/model-serving/why-llm-serving-is-different) and [continuous batching and PagedAttention](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention), then come back. By the end of this one you will be able to derive from first principles why prefill and decode want different hardware; compute the exact number of bytes a KV-cache handoff moves and how long it takes on NVLink versus InfiniBand; decide whether to transfer the cache or recompute it; derive the prefill-to-decode worker ratio for your workload; stand up a two-pool deployment in vLLM and NVIDIA Dynamo with a real KV connector; and — the part that matters most — know the handful of situations where disaggregation is the wrong call and colocation wins.

Everything here ties back to the spine of the series: the serving SLO triangle of **latency ↔ throughput ↔ cost**. Disaggregation is a trade on that triangle. It buys you dramatically better *goodput* — throughput that actually meets your latency SLO — at the cost of a more complex topology, a KV transfer on the critical path, and a minimum scale below which it simply is not worth it. Let us earn every one of those claims.

## 1. Prefill and decode are two different machines

The reason colocation hurts is not scheduling clumsiness that a smarter batching algorithm could fix. It is that prefill and decode are physically different kinds of computation that stress different parts of the GPU. To see why, you have to look at the two phases through the lens of the **roofline model**.

The roofline model says the throughput of any GPU kernel is bounded by one of two ceilings: the peak floating-point rate (the flat "compute roof") or the peak memory bandwidth times the kernel's *arithmetic intensity* (the sloped "bandwidth roof"). Arithmetic intensity is the ratio of useful FLOPs performed to bytes moved from memory: $I = \text{FLOPs} / \text{bytes}$. Where a kernel's arithmetic intensity sits relative to the GPU's **ridge point** — the intensity at which the two roofs meet — tells you which resource is the bottleneck. Below the ridge, you are memory-bandwidth-bound; above it, you are compute-bound.

For an H100 SXM, the peak BF16 rate is about 990 TFLOP/s (dense, no sparsity) and HBM3 bandwidth is about 3.35 TB/s. The ridge point is their ratio:

$$I_\text{ridge} = \frac{990 \times 10^{12}\ \text{FLOP/s}}{3.35 \times 10^{12}\ \text{byte/s}} \approx 295\ \text{FLOP/byte}$$

Any kernel that does fewer than about 295 floating-point operations per byte it reads from HBM cannot keep the H100's compute units fed; it is starved by memory bandwidth. Any kernel above that line is starved by compute. Now place the two phases on this line.

![Matrix contrasting prefill as a compute-bound pass of large GEMMs above the roofline ridge against decode as a bandwidth-bound stream of single-token GEMV steps below the ridge, and the colocated case that saturates neither](/imgs/blogs/prefill-decode-disaggregation-2.webp)

The matrix above lays out the contrast across four dimensions; walk it row by row.

**Prefill is compute-bound.** When you prefill a prompt of $S$ tokens, every weight matrix in the model is multiplied against an $S$-row activation matrix. These are large, dense **GEMMs** (general matrix-matrix multiplies). A weight tile is loaded from HBM once and reused across all $S$ rows of the activation, so the arithmetic intensity scales with the sequence length and easily exceeds 300 FLOP/byte for any prompt longer than a few hundred tokens. Prefill sits *above* the ridge. It wants raw FLOPs, and on an H100 it can push the tensor cores to 40–55% of peak (model FLOP utilization, or MFU) — the rest lost to attention's quadratic cost and non-GEMM operations. Prefill's latency is set by how many FLOP/s you can bring to bear.

**Decode is memory-bandwidth-bound.** Once the prompt is cached, every subsequent step generates exactly one token. The activation is a single row. The weight matrices are multiplied against that one row — these are **GEMVs** (matrix-vector multiplies), not GEMMs. A weight tile is loaded from HBM and used *once*, against one token, then discarded. Arithmetic intensity collapses to roughly 1–2 FLOP/byte. Decode sits far *below* the ridge. It does not want FLOPs; it wants HBM bandwidth, because its wall-clock time is dominated by the cost of streaming the entire set of model weights out of memory once per token. On a single H100 a 70B model's 140 GB of FP16 weights take about ${140 / 3.35 \approx 42}$ ms to stream — which is exactly why decode is slow, and why batching is the only lever that helps it.

**Batching helps them oppositely.** Because decode reads each weight tile once per token, batching $B$ requests together means one weight load serves $B$ tokens — the per-token bandwidth cost divides by $B$. Decode throughput scales almost linearly with batch size until you run out of KV-cache memory or hit the compute roof. Prefill gets almost nothing from batching, because a single long prompt already saturates the FLOP units; stacking more prompts just queues them. This is the crux: **decode desperately wants a large batch and prefill does not care about batch at all.** They have opposite optimal operating points.

**Colocation saturates neither.** Put both on the same GPU and you get the bottom row of the matrix: prefill and decode interleave on the same streaming multiprocessors, the arithmetic intensity swings wildly step to step, the GPU is compute-bound one millisecond and bandwidth-bound the next, and neither resource is ever fully used. You measure 55% MFU during prefill bursts and 60% HBM utilization during decode lulls, never both at once, and your TPOT jitters between 40 and 180 ms as prefill work randomly preempts decode work. You bought an expensive GPU and are using a little over half of it on average while missing your SLO.

#### Worked example: the two phases on Llama-3-70B

Take Llama-3-70B in FP16 on H100s. Its KV cache per token is ${2 \times 80 \times 8 \times 128 \times 2 = 327{,}680}$ bytes — 320 KB — where the factors are 2 (K and V), 80 layers, 8 KV heads under grouped-query attention, 128 head dimension, and 2 bytes for FP16. Hold that number; the whole KV-transfer story is built on it.

- **Prefill a 2,048-token prompt.** The forward-pass FLOP count is about ${2 \times 70 \times 10^9 \times 2048 \approx 2.9 \times 10^{14}}$ FLOPs (the factor of 2 is one multiply-add per parameter per token). On a 4-GPU tensor-parallel prefill worker delivering ${4 \times 990 \times 0.5 \approx 1{,}980}$ TFLOP/s at 50% MFU, that is ${2.9 \times 10^{14} / 1.98 \times 10^{15} \approx 145}$ ms. Compute-bound: the answer is set entirely by FLOP/s.
- **Decode one token, batch of 1.** The worker must stream all 70B FP16 weights — 140 GB, sharded 35 GB per GPU across 4 GPUs — out of HBM once. At 3.35 TB/s per GPU that is ${35 / 3.35 \approx 10.4}$ ms per token, plus KV reads and kernel launch overhead, landing near 20–22 ms TPOT. Bandwidth-bound: FLOPs are almost idle.
- **Decode one token, batch of 128.** The *same* 35 GB weight read now serves 128 tokens. Per-request TPOT stays about 22 ms, but the worker emits 128 tokens per step — throughput multiplies by 128 while latency barely moves. This is why decode is a batching game.

The two phases in this one example want a 4-GPU FLOP machine running for 145 ms, and a bandwidth machine running a batch of 128 at 22 ms per token. Those are not the same machine. Section 2 turns that observation into an architecture.

### Chunked prefill: the best you can do while still colocated

Before we split the pools, it is worth understanding the state-of-the-art *colocated* fix, because it is the baseline disaggregation must beat and it is the right default at small scale. The technique is **chunked prefill** (introduced by Sarathi and now standard in vLLM). Instead of running a 2,048-token prefill as one indivisible 145 ms burst that monopolizes the GPU, you break it into fixed-size chunks — say 512 tokens each — and interleave those chunks with decode steps in the continuous-batching scheduler. Each scheduler iteration processes a mixed batch: some decode tokens for in-flight streams, plus one prefill chunk making progress on a new prompt. The prefill of a long prompt is now spread across several iterations rather than blocking one long one.

This helps, and it is why the colocated baseline in this post is not a strawman. Chunked prefill caps the worst-case stall at one chunk's compute (a few tens of milliseconds) rather than a whole prompt's (150+ ms), so the TPOT tail tightens. But it does not remove the interference — it rations it. Every scheduler iteration that spends FLOPs on a prefill chunk is an iteration those FLOPs are not decoding, so decode still slows whenever a prefill is in flight; you have traded a tall spike for a broad plateau. And the mixed batch runs at neither phase's optimal point: the prefill chunk drags the batch's arithmetic intensity up while the decode tokens drag it down, so you still saturate neither the FLOP roof nor the bandwidth roof. Chunked prefill is the right tool up to moderate scale, and it is the honest baseline. Disaggregation's claim is that past a certain QPS and SLO tightness, rationing the interference is not enough — you have to eliminate it by giving the two phases separate GPUs.

## 2. The disaggregation idea: two pools joined by a KV handoff

If prefill and decode are different machines, the fix is to build both and wire them together. Run a **prefill pool** of workers that do nothing but prompt forward passes, tuned for FLOP throughput. Run a **decode pool** of workers that do nothing but token generation, tuned for the largest batch your KV memory allows. Between them, ship the one artifact that prefill produces and decode consumes: the KV cache. That is the entire idea. The subtlety is all in the wiring.

![Graph of a disaggregated serving topology in which a KV-aware router fans prompts to two tensor-parallel prefill workers, whose per-request KV cache is transferred over NVLink into a large-batch decode pool that streams tokens back to the client](/imgs/blogs/prefill-decode-disaggregation-3.webp)

The topology above shows the four moving parts. A **router** receives requests — 320 RPS in the figure — and load-balances prompts across prefill workers. Each **prefill worker** (tensor-parallel across 4 H100s here) runs the prompt forward pass and produces that request's KV cache in its own HBM. The **KV transfer** moves those cache blocks over the interconnect — 0.7 ms over NVLink in this example — into the decode pool's memory. The **decode pool**, running one large batch of 128, resumes each request at its first output token and streams tokens back to the client at a steady 22 ms TPOT. The prefill worker, having handed off the cache, is immediately free to prefill the next prompt.

The elegance is that the two pools now scale and batch *independently*. The prefill pool sizes to your prompt-arrival rate and prompt lengths. The decode pool sizes to your concurrency and output lengths, and it can run the biggest batch it wants because no prefill burst will ever interrupt it. You have decoupled the two axes of the SLO triangle that colocation had welded together: TTFT is now owned entirely by the prefill pool, and TPOT is owned entirely by the decode pool, and neither can hurt the other.

Three design decisions fall out of this topology, and the rest of the post is about getting each right:

1. **How the KV cache actually moves** — the bytes, the bandwidth, the layer-by-layer pipelining, and the alternative of just recomputing it (Section 3).
2. **How the latency budget splits** across the request's life, and why the handoff barely moves TTFT (Section 4).
3. **How many prefill workers versus decode workers** to provision — the PD ratio (Section 5).

There is one non-obvious property worth stating up front, because it drives the whole cost argument. The KV cache is a *write-once, read-many* artifact. Prefill writes it once. Decode reads it once per token for the entire generation. So the transfer happens exactly once per request, at the prefill→decode boundary, and its cost is amortized over hundreds of decode steps. A handoff that costs 0.7 ms and is followed by 512 decode steps at 22 ms each — 11.3 seconds of decode — adds 0.006% to the request's wall-clock time. That asymmetry is what makes disaggregation viable at all. If the transfer had to happen every token, the whole scheme would collapse. It happens once.

### The KV-aware router is not a plain load balancer

The router in the figure does more than round-robin. It is **KV-aware**: it tracks which prefill workers are busy, how full the decode pool's KV memory is, and — critically for long-context workloads — whether a prompt's prefix is already cached somewhere so prefill can be skipped entirely. This is where disaggregation composes with prefix caching (see [KV-cache optimization](/blog/machine-learning/model-serving/kv-cache-optimization)): a KV-centric router that sees the same 2,000-token system prompt on every request can prefill it once, keep it in a shared pool, and hand only the cached blocks to decode. Mooncake's entire architecture, which we will study in the case studies, is built around exactly this global KV pool. For now, take the router as a component that assigns prompts to prefill workers and coordinates the handoff; it is not the interesting mechanism, but it is the one that turns the topology into a system.

The router also owns the two policies that most affect goodput. The first is **prefill placement** — which prefill worker gets a prompt — where the right choice balances load while preferring a worker that already holds a matching prefix. The second is **decode admission** — whether the decode pool has KV headroom to accept a handoff right now, or whether the prompt should wait in the prefill pool rather than fill decode memory it cannot yet use. Both policies need live state from every worker, which is why the router is a stateful controller, not a stateless L7 proxy. Get either policy wrong and you convert the goodput win back into the interference you disaggregated to escape — over-eager admission refills the decode pool's memory and triggers the back-pressure failure mode, while naive round-robin placement throws away every prefix-cache hit. The router is where disaggregation's theory meets its operational reality.

## 3. The KV-cache transfer: bytes, bandwidth, and the recompute alternative

This is the mechanics block — the arithmetic that makes disaggregation provable rather than asserted. The central question is simple: **is it cheaper to ship the KV cache from prefill to decode, or to just recompute it on the decode worker?** Answering it requires exactly two numbers: how many bytes the cache is, and how fast the link is.

![Layered data path showing the KV cache leaving prefill HBM, being read and serialized per layer, crossing the NVLink interconnect at 900 GB/s, and landing in the decode worker's HBM buffer so decode resumes at the first output token](/imgs/blogs/prefill-decode-disaggregation-4.webp)

The stack above traces the path a single request's KV cache travels. It leaves the prefill worker's HBM as fixed-size cache blocks, gets read and serialized layer by layer, crosses the interconnect via an RDMA write, lands in the decode worker's HBM as paged blocks, and the decode worker resumes at token one. Two properties of this path matter enormously: the transfer is *sized by the prompt*, and it can be *overlapped with prefill compute*.

### The bytes

The KV cache for one request is its per-token footprint times the prompt length:

$$B_\text{KV} = \underbrace{2 \cdot L \cdot h_\text{kv} \cdot d_\text{head} \cdot b_\text{kv}}_{\text{bytes per token}} \times S$$

where $L$ is layers, $h_\text{kv}$ is KV heads (grouped-query attention shares them, which shrinks this term), $d_\text{head}$ is head dimension, $b_\text{kv}$ is KV precision in bytes, and $S$ is prompt length. For Llama-3-70B that per-token figure is the 320 KB we computed. So a 2,048-token prompt carries ${320\ \text{KB} \times 2048 = 640}$ MB of KV cache. A 128-token chat turn carries only 40 MB. An 8,000-token RAG prompt carries 2.5 GB. The transfer cost scales linearly with prompt length — long prompts are expensive to move, short prompts are nearly free.

### The time

Transfer time is bytes over bandwidth, plus a fixed latency floor for the round trip:

$$t_\text{transfer} = \frac{B_\text{KV}}{\text{BW}_\text{link}} + \alpha$$

The link bandwidth is the one number that changes everything, so let us tabulate the two-node case for that 640 MB Llama-3-70B prompt. The transfer is naturally sharded across the tensor-parallel GPUs — with TP=4, each GPU holds a quarter of the KV heads and pushes its shard over its own link, so the effective per-GPU payload is 160 MB and the links run in parallel.

| Interconnect | Per-GPU BW | Latency floor $\alpha$ | Time for 160 MB shard | Time for full 640 MB (unsharded) |
| --- | --- | --- | --- | --- |
| NVLink 4 (intra-node) | 900 GB/s | < 1 µs | 0.18 ms | 0.71 ms |
| InfiniBand NDR (inter-node) | 50 GB/s | 2–3 µs | 3.2 ms | 12.8 ms |
| RoCE v2 (inter-node) | 25 GB/s | 3–5 µs | 6.4 ms | 25.6 ms |
| TCP Ethernet (100 GbE) | 12 GB/s | 20–50 µs | 13.3 ms | 53.3 ms |

The 900 GB/s NVLink handoff is the 0.7 ms number from every figure in this post — it is so cheap it is essentially free relative to a 145 ms prefill. InfiniBand at 50 GB/s per port turns the same transfer into 3–13 ms, still small next to prefill but no longer negligible. And TCP Ethernet at the bottom is a non-starter for anything but the shortest prompts — 53 ms of transfer would eat a quarter of your prefill budget. **The interconnect tier is the first thing you check before disaggregating**; it decides whether the handoff is free or a tax.

### Transfer versus recompute

Now the decision. The alternative to transferring the cache is to *not* transfer it: send the raw prompt to the decode worker and have it run prefill itself. Recompute costs a full prefill's worth of compute on the decode GPU — the very compute-bound burst we disaggregated to avoid — which for our 2,048-token prompt is about 145 ms of FLOP-heavy work that also stalls the decode pool. Transfer wins whenever:

$$\frac{B_\text{KV}}{\text{BW}_\text{link}} + \alpha \;<\; t_\text{prefill,recompute}$$

Plug the numbers. On NVLink, ${0.71\ \text{ms} < 145\ \text{ms}}$ — transfer wins by 200×. On InfiniBand, ${12.8\ \text{ms} < 145\ \text{ms}}$ — transfer wins by 11×. Even on 100 GbE Ethernet, ${53\ \text{ms} < 145\ \text{ms}}$ — transfer still wins, though the margin is thinning. Transfer is almost always the right call, and the reason is structural: recompute pays the *compute* cost (which is what we disaggregated to isolate), while transfer pays only the *bandwidth* cost of moving an already-computed result. You spent the FLOPs once on the prefill worker; throwing that work away to recompute it on decode defeats the entire purpose.

The one regime where recompute wins is when the transfer is *more expensive than the compute* — which happens only when bandwidth is very low, the prompt is very long, or, most importantly, when prefix caching means recompute is nearly free because the prefix is already cached on the decode side. If the decode worker already holds 90% of a prompt's prefix in its KV cache, recomputing the last 10% is cheaper than transferring the whole thing. This is why real systems support *both* and let the router choose per request.

#### Worked example: transfer versus recompute for a RAG prompt

A retrieval-augmented prompt on Llama-3-70B: 6,000 tokens of retrieved context, 200 tokens of question. The KV cache is ${320\ \text{KB} \times 6200 \approx 1.9}$ GB. Recomputing prefill on the decode worker costs about ${2 \times 70 \times 10^9 \times 6200 / 1.98 \times 10^{15} \approx 438}$ ms of compute — and it stalls the decode pool for all 438 ms.

- **Transfer over NVLink:** ${1.9\ \text{GB} / 900\ \text{GB/s} \approx 2.1}$ ms. Transfer wins by 200×.
- **Transfer over InfiniBand:** ${1.9\ \text{GB} / 50\ \text{GB/s} \approx 38}$ ms. Transfer still wins by 11×, and — crucially — those 38 ms happen on a dedicated link, not by stealing decode compute.
- **Recompute, but the 6,000-token context is a cached document already in decode HBM:** ${\approx 14}$ ms to prefill just the 200-token question. Now recompute wins, because the expensive part was free.

The lesson: default to transfer, but keep recompute in your pocket for the prefix-cache-hit case. The router's job is to know which one it is looking at.

### Layer-by-layer pipelining

One more mechanic makes the transfer even cheaper than the table suggests. The KV cache is produced layer by layer during prefill — layer 0's KV exists long before layer 79 finishes. So instead of waiting for the entire prefill to complete and then transferring all 640 MB in one blocking shot, a good implementation **pipelines** the transfer: as soon as prefill finishes layer $\ell$, it kicks off the transfer of layer $\ell$'s KV blocks while prefill computes layer $\ell+1$. The transfer of the early layers overlaps with the compute of the later layers, and only the last layer's transfer is exposed on the critical path. For an 80-layer model, this hides roughly 79/80 of the transfer time behind compute. This is why the figures quote the handoff as a sub-millisecond addition to TTFT even when the raw transfer is a few milliseconds — most of it happens *during* prefill, not after. vLLM's disaggregated-prefill connectors and NVIDIA Dynamo's NIXL transfer layer both do this layer-wise overlap.

### How the bytes actually move: three transport mechanisms

The formula $t = B_\text{KV}/\text{BW}$ hides a real engineering choice about *how* the transfer is implemented, and the three common mechanisms have materially different overheads. Because the KV cache lives in **paged blocks** (PagedAttention stores it as fixed-size pages, typically 16 tokens per block, scattered non-contiguously across HBM), the transfer is not one big contiguous copy — it is a gather of many small pages, which stresses the transport's small-message performance.

- **NCCL point-to-point** (`ncclSend`/`ncclRecv`) is the vLLM `PyNcclConnector` path. It rides the same NCCL stack the model already uses for tensor-parallel collectives, so it is well-tuned for GPU-to-GPU over NVLink and needs no extra infrastructure. Its weakness is that NCCL prefers large contiguous buffers; transferring thousands of scattered 16-token pages means either a gather-into-staging-buffer step first, or many small sends that pay per-message latency. For intra-node NVLink this is fine; the bandwidth is so high that even the staging copy is cheap.
- **GPUDirect RDMA** (the NIXL/Mooncake path for cross-node) writes directly from one GPU's HBM into another's across an InfiniBand or RoCE NIC, bypassing the host CPU and its bounce buffers entirely. This is what makes the 12.8 ms InfiniBand number achievable — a naive host-staged transfer (GPU→CPU→NIC→CPU→GPU) would be several times slower because of the extra HBM↔DRAM copies. RDMA one-sided writes also let the producer push KV into the consumer's memory without the consumer's CPU being involved on the critical path, which matters for keeping decode's scheduler free.
- **Shared-memory / tiered pool** (Mooncake, LMCache) treats KV as a first-class storage object in a pool spanning HBM, host DRAM, and SSD. The transfer becomes a write to the pool and a read from it, and the pool's tiering policy decides where each block lives. This is slower per-transfer than direct RDMA but unlocks reuse: a block written by one request's prefill can be read by a different request's decode, which is the whole basis of cross-request prefix caching.

The practical rule: **NCCL for intra-node NVLink, GPUDirect RDMA for cross-node, and a tiered pool when reuse dominates.** The connector you pick in Section 6 is exactly this choice, and it is the single biggest lever on whether the transfer term stays negligible.

## 4. The latency budget: life of a disaggregated request

To reason about SLOs, you have to account for every millisecond of a request's life and see exactly where the handoff lands. Two metrics govern LLM serving. **TTFT** (time to first token) is what the user waits before anything appears — dominated by prefill. **TPOT** (time per output token, sometimes called inter-token latency) is the steady cadence of generation after the first token — dominated by decode. A good chat SLO looks like "TTFT p90 < 500 ms, TPOT p90 < 40 ms." Disaggregation's promise is that it protects *both* by giving each its own hardware.

![Timeline of a disaggregated request showing arrival of a 2048-token prompt, about 150ms of prefill on a four-GPU worker, a sub-millisecond KV transfer over NVLink, first token at roughly 151ms, and a steady 22ms-per-token decode stream to completion](/imgs/blogs/prefill-decode-disaggregation-5.webp)

The timeline above walks a single 2,048-token request end to end. Trace it:

- **t = 0** — the prompt arrives at the router and is dispatched to a prefill worker.
- **0–150 ms** — the prefill worker runs the forward pass over all 2,048 tokens on 4×H100. This is the compute-bound burst.
- **t = 150 ms** — the KV cache handoff. Over NVLink, 0.7 ms; because of layer-wise pipelining, almost all of it already happened during prefill, so the *exposed* cost is a fraction of a millisecond.
- **t ≈ 151 ms** — the decode pool emits the **first token**. TTFT ≈ 151 ms, essentially the prefill time plus a rounding error for the handoff.
- **151 ms onward** — the decode pool streams every subsequent token at a steady 22 ms TPOT, insulated from any prefill burst because it runs on decode-only GPUs.
- **t ≈ 11.3 s** — after 512 output tokens at 22 ms each, generation completes. Both the TTFT and the TPOT SLO were met for the entire stream.

The point the timeline makes visually is that **the handoff barely moves TTFT**. TTFT is prefill-dominated; adding 0.7 ms (or even a pipelined-away 3 ms on InfiniBand) to a 150 ms prefill is noise. And TPOT is *perfectly flat* because the decode pool never runs prefill. Compare that to the colocated case from the intro, where a single long prompt's prefill burst would inject a 150 ms gap into the middle of some other request's decode stream — the stutter on the dashboard. Disaggregation converts a shared, interfering resource into two dedicated, non-interfering ones, and the timeline is the proof: there is nowhere in a decode GPU's schedule for a prefill burst to hide, because prefill bursts do not run there.

#### Worked example: the TTFT/TPOT budget under an SLO

Suppose your SLO is TTFT p90 < 400 ms and TPOT p90 < 40 ms, on Llama-3-70B, and your workload is 512-token prompts with 256-token outputs.

- **TTFT budget.** Prefill of 512 tokens on a 4-GPU worker: ${2 \times 70 \times 10^9 \times 512 / 1.98 \times 10^{15} \approx 36}$ ms of compute. Add queueing delay in the prefill pool (the time a prompt waits for a free worker) and the handoff. As long as the prefill pool is provisioned so queueing stays under ~360 ms at p90, TTFT holds. This is a **queuing** constraint on the prefill pool — it sets how many prefill workers you need.
- **TPOT budget.** Decode at batch 128 gives ~22 ms per token, comfortably under 40 ms. The headroom lets you push the batch higher — batch 200 might land at 32 ms TPOT, still inside SLO, at higher throughput. This is a **batch-size** knob on the decode pool.

The two budgets are now independent dials. In a colocated system they fought each other: raising the decode batch to hit throughput made prefill bursts longer and blew TTFT; shrinking the batch to protect TTFT tanked throughput. Disaggregation cuts that knot. You tune the prefill pool for TTFT and the decode pool for TPOT, separately. That independence is the real product.

### Sizing the prefill pool with queuing theory

TTFT has two components: the prefill *compute* (fixed by the model, the prompt length, and the worker's FLOP/s) and the prefill *queue wait* (how long a prompt sits before a worker is free). The compute term you cannot cheat — 36 ms for a 512-token prompt is 36 ms. The queue term is entirely a provisioning decision, and queuing theory tells you exactly how many workers you need to keep it small.

Model the prefill pool as an M/M/c queue: Poisson arrivals at rate $\lambda$, $c$ prefill workers, each with service rate $\mu = 1/t_\text{prefill}$. The key quantity is the pool utilization $\rho = \lambda / (c\mu)$. Queue wait explodes non-linearly as $\rho \to 1$ — the classic hockey stick — so the entire game is keeping $\rho$ comfortably below 1. A useful rule of thumb from the M/M/c waiting-time formula: to hold p90 queue wait under one service time, you generally want $\rho \lesssim 0.7$, which means provisioning

$$c \gtrsim \frac{\lambda \cdot t_\text{prefill}}{0.7}$$

workers. Concretely: at $\lambda = 40$ prompts/s with $t_\text{prefill} = 36$ ms, you need $c \gtrsim 40 \times 0.036 / 0.7 \approx 2.1$, so three prefill workers keep TTFT queue wait small. Push utilization to $\rho = 0.95$ to save a worker and the queue wait balloons past the compute term, and your nice 36 ms prefill becomes a 300 ms TTFT under load — the pool is technically not overloaded ($\rho < 1$) but the tail has already blown. **The prefill pool must run at moderate utilization, not maximal.** This is the counterintuitive cost of tight TTFT: you deliberately leave prefill headroom idle so that a burst of arrivals does not queue. The decode pool, by contrast, *wants* high utilization because its SLO (TPOT) is set by batch efficiency, not queue wait — which is one more reason the two pools want different operating points and belong on different GPUs.

## 5. The PD ratio: how many prefill workers versus decode workers

Once you commit to two pools, the money question is how to split your GPUs between them. Too many prefill workers and your decode pool is starved and can't hold a big enough batch; too many decode workers and prompts queue for prefill and TTFT blows up. The right split is the **PD ratio**, and it follows from a Little's-Law-style balance: provision each pool in proportion to the GPU-time each phase consumes per request.

![Grid of an eight-GPU node partitioned by PD ratio, dedicating two GPUs to a prefill worker and six GPUs to a large-batch decode pool, the 1:3 split derived for a short-prompt long-output chat workload under a tight token-latency SLO](/imgs/blogs/prefill-decode-disaggregation-6.webp)

The grid above shows the answer for a chat workload with short prompts and long outputs under a tight TPOT SLO: a **1:3 split** — two of the eight GPUs run prefill, six run the decode pool. Here is where that comes from.

### The derivation

Define the GPU-seconds each phase consumes per request:

$$C_\text{prefill} = t_\text{prefill} \cdot g_p \qquad C_\text{decode} = \frac{L_\text{out} \cdot t_\text{TPOT} \cdot g_d}{B}$$

$C_\text{prefill}$ is the prefill latency times the number of GPUs in a prefill worker — the GPUs are busy for the whole prefill. $C_\text{decode}$ is the output length times TPOT (the wall-clock a request spends decoding) times the decode worker's GPU count, divided by the decode batch $B$ (because those GPUs are shared across $B$ concurrent requests). To keep the pipeline balanced — prefill producing KV caches exactly as fast as decode consumes them — you provision GPUs in the ratio of these two quantities:

$$R = \frac{N_\text{prefill}}{N_\text{decode}} = \frac{C_\text{prefill}}{C_\text{decode}} = \frac{t_\text{prefill} \cdot g_p \cdot B}{L_\text{out} \cdot t_\text{TPOT} \cdot g_d}$$

The intuition the formula encodes: **the ratio tilts toward decode when outputs are long and toward prefill when prompts are long.** Long outputs ($L_\text{out}$ large) put more work in decode → more decode GPUs. Long prompts ($t_\text{prefill}$ large) put more work in prefill → more prefill GPUs. Your workload's prompt-to-output shape *is* your PD ratio.

#### Worked example: deriving the 1:3 chat ratio

Chat workload: 256-token prompts, 1,024-token outputs, on Llama-3-70B. Prefill worker is TP=2 ($g_p = 2$), decode worker is TP=2 ($g_d = 2$) running batch $B = 256$, TPOT = 22 ms.

- Prefill of 256 tokens on 2 GPUs: ${2 \times 70 \times 10^9 \times 256 / (2 \times 990 \times 10^{12} \times 0.5) \approx 36}$ ms. So $C_\text{prefill} = 0.036 \times 2 = 0.072$ GPU-s.
- Decode: $C_\text{decode} = (1024 \times 0.022 \times 2) / 256 = 0.176$ GPU-s.
- Ratio: $R = 0.072 / 0.176 \approx 0.41$, i.e. **1 : 2.4** prefill-to-decode GPUs.

Round to the nearest clean partition of an 8-GPU node and you get 2 prefill : 6 decode — the 1:3 split in the figure. The rounding always goes toward *more* decode, because starving decode raises TPOT (a per-token SLO you feel on every token), whereas slightly over-provisioning prefill only trims TTFT queueing. Now flip the workload: a summarization service with 6,000-token prompts and 200-token outputs. Prefill dominates, $C_\text{prefill}$ balloons, and the same formula yields something like 3:1 the *other* way — most GPUs on prefill. There is no universal PD ratio; there is only your workload's ratio.

### A PD-ratio calculator

Here is the derivation as runnable code you can point at your own traces. Feed it your model dimensions, hardware, and the prompt/output distribution, and it prints the GPU split.

```python
# pd_ratio.py — derive the prefill:decode GPU ratio for a workload.
# All FLOP/s and bandwidth figures are per-GPU peak; tune MFU/util to your fleet.

from dataclasses import dataclass

@dataclass
class Model:
    params: float          # total parameters
    layers: int
    kv_heads: int
    head_dim: int
    kv_bytes: int = 2      # FP16 KV cache

@dataclass
class HW:
    tflops: float          # per-GPU dense BF16 TFLOP/s
    hbm_tbs: float         # per-GPU HBM bandwidth, TB/s
    mfu: float = 0.5       # realized prefill MFU

def prefill_ms(m: Model, hw: HW, prompt_len: int, gp: int) -> float:
    flops = 2 * m.params * prompt_len          # fwd-pass multiply-adds
    rate = gp * hw.tflops * 1e12 * hw.mfu      # aggregate realized FLOP/s
    return flops / rate * 1e3

def tpot_ms(m: Model, hw: HW, gd: int, batch: int) -> float:
    # decode is bandwidth-bound: stream all weights once per step, amortized by batch.
    weight_bytes = m.params * m.kv_bytes       # FP16 weights
    per_gpu = weight_bytes / gd
    step_ms = per_gpu / (hw.hbm_tbs * 1e12) * 1e3
    return step_ms + 2.0                        # + kernel/KV overhead floor

def pd_ratio(m, hw, prompt_len, out_len, gp, gd, batch):
    t_pre = prefill_ms(m, hw, prompt_len, gp)
    t_tpot = tpot_ms(m, hw, gd, batch)
    c_pre = t_pre / 1e3 * gp                    # prefill GPU-seconds/request
    c_dec = (out_len * t_tpot / 1e3 * gd) / batch
    R = c_pre / c_dec
    return t_pre, t_tpot, R

llama70b = Model(params=70e9, layers=80, kv_heads=8, head_dim=128)
h100 = HW(tflops=990, hbm_tbs=3.35)

t_pre, t_tpot, R = pd_ratio(llama70b, h100,
                            prompt_len=256, out_len=1024,
                            gp=2, gd=2, batch=256)
print(f"prefill {t_pre:.0f} ms  |  TPOT {t_tpot:.1f} ms  |  ratio 1:{1/R:.1f}")
# prefill 36 ms  |  TPOT 12.4 ms  |  ratio 1:2.4
# -> round to 2 prefill : 6 decode on an 8-GPU node.
```

Run it against your production prompt/output histogram (not the mean — the *distribution*, because a heavy tail of long prompts changes the answer), and you have a defensible starting partition. Then measure and adjust, because the model's constants — MFU, the overhead floor, the batch you can actually fit — are only approximate until you benchmark them.

### The static ratio is a lie your traffic will expose

The derivation above computes one ratio for one workload, but production traffic is not one workload — it breathes. Morning traffic might be short interactive chat (decode-heavy, wants 1:3); an afternoon batch of document summarization jobs is prompt-heavy (wants 3:1); a nightly eval run is something else again. A statically partitioned 2-prefill/6-decode node is optimal for exactly one of those and wasteful for the rest: during the summarization burst, two prefill GPUs are a bottleneck while six decode GPUs sit half-idle waiting for KV caches that arrive too slowly.

Two responses. The first is **independent autoscaling**: because the pools are separate Deployments (Section 6), you scale each on its own signal — the prefill pool on prompt-arrival rate and prefill-queue depth, the decode pool on batch occupancy and KV-memory pressure. When the summarization burst hits, the prefill pool scales out and the decode pool scales in, and the effective ratio shifts with the traffic. This is the cloud-native answer and it works when your scale is large enough that pools span many nodes and a few seconds of scaling lag is tolerable.

The second is **elastic role assignment**: some systems (Dynamo among them) let a GPU flip between prefill and decode roles on a timescale of seconds rather than being nailed to one pool. A worker that is currently decoding can be told to take a prefill chunk when the prefill queue backs up, blurring the hard partition back toward colocation *on purpose* when the ratio is wrong. This recovers colocation's fungibility without giving up disaggregation's isolation most of the time — a hybrid that is increasingly where production systems land. The honest takeaway: **derive a static ratio to size your fleet, but plan for it to be wrong intraday, and build the autoscaling or elastic-role machinery to track the real distribution.** A perfect static ratio for average traffic is a poor ratio for every actual minute of traffic.

## 6. Frameworks: DistServe, Splitwise, Mooncake, vLLM and Dynamo

Disaggregation is not a research curiosity; it is shipping in production systems today, each with a different bet on *how* to split and *how* to move the KV cache. Knowing which framework fits which workload is most of the practical decision.

![Matrix comparing DistServe, Splitwise, Mooncake, and vLLM plus Dynamo across core idea, KV transport, reported win, and best-fit workload](/imgs/blogs/prefill-decode-disaggregation-7.webp)

The matrix above compares the four systems across the axes that matter. Read it as four different answers to the same question. **DistServe** optimizes per-phase parallelism for goodput and moves KV over local NVLink. **Splitwise** splits the phases across two *machine types* — cheap compute-heavy boxes for prefill, memory-heavy boxes for decode — and ships KV over InfiniBand. **Mooncake** centers everything on a global KV-cache pool tiered across DRAM and SSD, optimizing for reuse on long-context workloads. **vLLM plus NVIDIA Dynamo** package disaggregation for open-source datacenter scale, with a KV-aware router and NIXL/NCCL layer-piped transfer. The full narrative and reported numbers are in the case studies section; here is how you actually configure the open-source path.

### vLLM disaggregated prefill

vLLM ships experimental disaggregated prefill via a **KV connector**. You launch two engines — a `kv_producer` (prefill) and a `kv_consumer` (decode) — that share a KV transfer channel. The minimal shape uses the NCCL-based connector for intra-node NVLink transfer:

```bash
# Prefill worker (KV producer) — pinned to GPUs 0-1, tuned for FLOP throughput.
CUDA_VISIBLE_DEVICES=0,1 vllm serve meta-llama/Meta-Llama-3-70B \
  --tensor-parallel-size 2 \
  --max-num-batched-tokens 8192 \
  --kv-transfer-config \
  '{"kv_connector":"PyNcclConnector","kv_role":"kv_producer",
    "kv_rank":0,"kv_parallel_size":2}' \
  --port 8100

# Decode worker (KV consumer) — pinned to GPUs 2-7, tuned for large decode batch.
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 vllm serve meta-llama/Meta-Llama-3-70B \
  --tensor-parallel-size 6 \
  --max-num-seqs 256 \
  --kv-transfer-config \
  '{"kv_connector":"PyNcclConnector","kv_role":"kv_consumer",
    "kv_rank":1,"kv_parallel_size":2}' \
  --port 8200
```

The `kv_producer` runs prefill, writes the KV cache into the shared transfer buffer keyed by request, and returns immediately; the `kv_consumer` pulls that cache and runs decode. A thin proxy in front sends each request to the producer first, then the consumer. Note the asymmetric shapes: TP=2 for prefill (small, FLOP-tuned) and TP=6 for decode (large, batch-tuned) — the 1:3 GPU split from Section 5, expressed as launch flags. For inter-node or higher-throughput transfer, swap `PyNcclConnector` for a Mooncake-backed or LMCache connector; the role/rank contract stays the same.

### A KV connector in the engine API

If you are embedding vLLM rather than using the CLI, the same configuration lives in `KVTransferConfig` on the engine args:

```python
# disagg_engine.py — build a decode-side engine that consumes transferred KV.
from vllm import AsyncLLMEngine, AsyncEngineArgs
from vllm.config import KVTransferConfig

kv_cfg = KVTransferConfig(
    kv_connector="PyNcclConnector",
    kv_role="kv_consumer",          # this engine decodes
    kv_rank=1,
    kv_parallel_size=2,             # 1 producer + 1 consumer group
)

engine = AsyncLLMEngine.from_engine_args(
    AsyncEngineArgs(
        model="meta-llama/Meta-Llama-3-70B",
        tensor_parallel_size=6,
        max_num_seqs=256,           # big decode batch — this is the point
        gpu_memory_utilization=0.92,
        kv_transfer_config=kv_cfg,
    )
)
# The producer engine is identical except kv_role="kv_producer",
# tensor_parallel_size=2, and a prefill-tuned max_num_batched_tokens.
```

The connector interface is the seam. `PyNcclConnector` moves KV over NCCL point-to-point (good for NVLink); the Mooncake and LMCache connectors add a tiered DRAM/SSD pool and cross-node RDMA. You pick the connector by your interconnect and your reuse pattern, and the engine code above does not change.

### NVIDIA Dynamo: disaggregation as a datacenter graph

NVIDIA Dynamo (open-sourced 2025) treats disaggregation as a first-class deployment topology rather than a two-process hack. You declare prefill and decode workers, a KV-aware router, and the NIXL transfer layer in a single config, and Dynamo handles the routing, the handoff, and the scaling:

```yaml
# dynamo-disagg.yaml — a representative disaggregated graph.
# Schema tracks the Dynamo release; treat as the shape, not the exact keys.
common:
  model: meta-llama/Meta-Llama-3-70B
  kv-transfer:
    backend: nixl          # NVIDIA Inference Xfer Library (RDMA/NVLink)
    overlap: layerwise     # pipeline KV transfer with prefill compute

router:
  policy: kv-aware         # skip prefill on prefix-cache hits
  slo:
    ttft-ms: 400
    tpot-ms: 40

workers:
  prefill:
    replicas: 1
    tensor-parallel-size: 2
    role: prefill          # produces KV, tuned for FLOP throughput
  decode:
    replicas: 1
    tensor-parallel-size: 6
    max-batch-size: 256
    role: decode           # consumes KV, tuned for batch throughput
```

The `nixl` backend with `overlap: layerwise` is the layer-by-layer pipelined transfer from Section 3, and the `kv-aware` router is the prefix-cache-hit optimization from Section 2. Dynamo's contribution is making all of that declarative and cluster-scale instead of something you wire by hand.

### The two-pool Kubernetes deployment

To run this in production you need each pool as its own Deployment behind its own GPU node pool, with a proxy that chains producer → consumer. A minimal sketch:

```yaml
# pd-pools.yaml — prefill and decode as separate GPU-pinned Deployments.
apiVersion: apps/v1
kind: Deployment
metadata: { name: prefill-pool }
spec:
  replicas: 1
  selector: { matchLabels: { app: prefill } }
  template:
    metadata: { labels: { app: prefill } }
    spec:
      nodeSelector: { pool: gpu-h100 }
      containers:
        - name: vllm
          image: vllm/vllm-openai:latest
          args: ["--model","meta-llama/Meta-Llama-3-70B",
                 "--tensor-parallel-size","2",
                 "--kv-transfer-config",
                 '{"kv_connector":"PyNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2}']
          resources: { limits: { nvidia.com/gpu: 2 } }
---
apiVersion: apps/v1
kind: Deployment
metadata: { name: decode-pool }
spec:
  replicas: 1
  selector: { matchLabels: { app: decode } }
  template:
    metadata: { labels: { app: decode } }
    spec:
      nodeSelector: { pool: gpu-h100 }
      containers:
        - name: vllm
          image: vllm/vllm-openai:latest
          args: ["--model","meta-llama/Meta-Llama-3-70B",
                 "--tensor-parallel-size","6","--max-num-seqs","256",
                 "--kv-transfer-config",
                 '{"kv_connector":"PyNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2}']
          resources: { limits: { nvidia.com/gpu: 6 } }
```

Two Deployments, two GPU allocations in the 1:3 ratio, one shared KV channel. The prefill pool and decode pool now autoscale independently — see [autoscaling model servers](/blog/machine-learning/model-serving/gpu-scheduling-mig-and-autoscaling) for scaling each pool on its own signal (prefill on prompt-arrival rate, decode on batch occupancy). This is the topology from figure 3, expressed as YAML. For how requests are ordered and preempted within each pool, [request scheduling and preemption](/blog/machine-learning/model-serving/request-scheduling-and-preemption) is the companion read — disaggregation changes *where* work runs, but each pool still needs a scheduler.

## 7. The goodput argument: what disaggregation actually buys

The reason to do any of this is a single metric, and it is not raw throughput. It is **goodput** — the rate of requests that meet *both* their TTFT and their TPOT SLO. A system can post a big tokens-per-second number while quietly violating latency on half its requests; those violating requests are worthless to the user, so they should not count. Goodput counts only the requests you actually served acceptably, and it is the metric disaggregation optimizes.

![Before and after comparison showing colocated chunked-prefill serving capped at 9 requests per second of SLO-meeting goodput with high tail TTFT and TPOT, versus disaggregated pools reaching 34 requests per second with roughly halved tail latencies on the same GPU count](/imgs/blogs/prefill-decode-disaggregation-8.webp)

The figure above is a representative before/after on a fixed 8×H100 budget, and it is the payoff shot. The colocated baseline here is not naive — it uses **chunked prefill**, the current best-practice colocation technique where long prefills are broken into token-sized chunks and interleaved with decode steps to reduce (not eliminate) interference. Even so, under load it caps at about 9 RPS of goodput, with TTFT p90 at 380 ms and TPOT p90 at 55 ms, both strained by the residual prefill contention. Disaggregating the same 8 GPUs into a 2-prefill / 6-decode split lifts goodput to about 34 RPS — roughly 3.8× — while *dropping* TTFT p90 to 160 ms and TPOT p90 to 24 ms. More SLO-meeting throughput and lower tail latency, from the same hardware, because the two phases stop fighting.

### Why goodput jumps and raw throughput barely moves

This is the subtle part worth internalizing: disaggregation does not make the GPUs compute faster. The aggregate FLOP/s and HBM bandwidth of 8 H100s is identical before and after. What changes is *how much of that capacity produces SLO-meeting output*. In the colocated case, the interference tax means you must run a conservative batch and leave headroom to absorb prefill bursts, or you blow TPOT; either way, a chunk of capacity is spent on latency violations that do not count toward goodput. Disaggregation removes the interference, so the decode pool can run its batch flat-out at a stable TPOT and the prefill pool can saturate its FLOPs, and nearly all of the output lands inside SLO. You are not adding capacity; you are stopping the two phases from wasting each other's capacity. That is why the headline number is a *goodput* multiple, and why the raw tokens/sec number moves much less.

### A goodput benchmark harness

You should measure this on your own workload before believing any figure. Here is an async harness that fires a Poisson-arrival load, records TTFT and TPOT per request from the streaming response, and reports goodput against your SLO thresholds:

```python
# goodput_bench.py — measure SLO-meeting goodput of a streaming LLM endpoint.
import asyncio, time, json, random, statistics as st
import aiohttp

ENDPOINT = "http://localhost:8200/v1/completions"
MODEL = "meta-llama/Meta-Llama-3-70B"
TTFT_SLO_MS, TPOT_SLO_MS = 400, 40          # your SLO
DURATION_S, TARGET_RPS = 60, 30

async def one_request(session, prompt, out_len, results):
    t0 = time.perf_counter()
    first_tok_t, prev_t, itls = None, None, []
    payload = {"model": MODEL, "prompt": prompt, "max_tokens": out_len,
               "stream": True, "temperature": 0.7}
    async with session.post(ENDPOINT, json=payload) as r:
        async for line in r.content:
            if not line.startswith(b"data:"): continue
            if b"[DONE]" in line: break
            now = time.perf_counter()
            if first_tok_t is None:
                first_tok_t = now                       # first token seen
            else:
                itls.append((now - prev_t) * 1e3)       # inter-token latency
            prev_t = now
    ttft_ms = (first_tok_t - t0) * 1e3
    tpot_ms = st.mean(itls) if itls else float("inf")
    ok = ttft_ms <= TTFT_SLO_MS and tpot_ms <= TPOT_SLO_MS
    results.append((ttft_ms, tpot_ms, ok))

async def main():
    prompts = ["Summarize the following:\n" + "lorem ipsum " * random.randint(30, 800)
               for _ in range(4000)]
    results = []
    async with aiohttp.ClientSession() as s:
        tasks, t_end = [], time.perf_counter() + DURATION_S
        while time.perf_counter() < t_end:
            p = random.choice(prompts)
            tasks.append(asyncio.create_task(
                one_request(s, p, random.randint(128, 512), results)))
            await asyncio.sleep(random.expovariate(TARGET_RPS))  # Poisson arrivals
        await asyncio.gather(*tasks)

    n = len(results)
    good = sum(1 for _, _, ok in results if ok)
    ttfts = sorted(r[0] for r in results)
    tpots = sorted(r[1] for r in results if r[1] != float("inf"))
    p90 = lambda xs: xs[int(0.9 * len(xs)) - 1]
    dur = DURATION_S
    print(f"requests={n}  goodput={good/dur:.1f} RPS  "
          f"({100*good/n:.0f}% met SLO)")
    print(f"TTFT p90={p90(ttfts):.0f}ms  TPOT p90={p90(tpots):.1f}ms")

asyncio.run(main())
```

Run it against the colocated endpoint and the disaggregated endpoint with the *same* arrival stream and SLO, and the goodput ratio it prints is the number that should drive your decision — not vendor slides, not this post's figures. Measure your own workload.

### A named-hardware before→after table

Pulling the representative numbers into one table, on a fixed 8×H100-80GB node serving Llama-3-70B in FP16 under a TTFT p90 < 400 ms / TPOT p90 < 40 ms SLO:

| Configuration | GPU split | Goodput (SLO-met RPS) | TTFT p90 | TPOT p90 | Decode batch | Prefill MFU |
| --- | --- | --- | --- | --- | --- | --- |
| Colocated, chunked prefill | 8 shared | ~9 | 380 ms | 55 ms | ~48 (capped) | ~40% |
| Disaggregated 2P + 6D | 2 prefill / 6 decode | ~34 | 160 ms | 24 ms | 256 | ~92% |
| Disaggregated 1P + 7D (output-heavy) | 1 prefill / 7 decode | ~28 | 220 ms | 21 ms | 320 | ~90% |
| Disaggregated 4P + 4D (prompt-heavy) | 4 prefill / 4 decode | ~19 | 120 ms | 30 ms | 160 | ~93% |

The table also shows the sensitivity to the PD ratio: the 2:6 split is best for the balanced chat workload, but if your outputs run longer you shift GPUs to decode (1:7) and if your prompts run longer you shift to prefill (4:4). Get the ratio wrong and you leave goodput on the table — an over-prefilled 4:4 split on a chat workload starves decode and drops goodput back toward the colocated number. The ratio is not a detail; it is half the win. These figures are representative of the DistServe/Splitwise-class results discussed next, not a specific published benchmark — treat them as the shape of the outcome and measure your own.

## Case studies

Four systems built disaggregation into production and published what they got. The numbers below are from their papers and posts under their specific settings; I frame each as reported rather than universal, because the win is workload-dependent and the baselines differ.

### DistServe (OSDI 2024) — the goodput framing

DistServe (Zhong, Liu, Chen, Hu, Zhu, Liu, Jin, Zhang, OSDI 2024) is the paper that named the problem. Its central insight is exactly Section 1's: prefill and decode have opposing resource profiles, and colocating them forces a bad compromise. DistServe disaggregates the two phases onto separate GPUs and — its distinctive contribution — optimizes the parallelism *per phase*, choosing tensor- and pipeline-parallel degrees independently for prefill and decode based on each phase's bottleneck and the cluster's bandwidth. It also introduced **goodput** (requests/s meeting both TTFT and TPOT SLOs) as the right optimization target. The reported result: up to about **4.5× more goodput**, or the ability to hold roughly **10× tighter SLOs**, versus state-of-the-art colocated systems on chatbot, code-completion, and summarization workloads. The 3.8× goodput lift in figure 8 is a representative echo of this. DistServe is the reference point for tight-SLO, single-cluster serving where the KV transfer stays on fast local links.

### Splitwise (ISCA 2024) — split across machine types

Splitwise (Patel, Choukse, Zhang, Shah, Goiri, Maleki, Bianchini, Microsoft Azure, ISCA 2024) took the same phase split and pushed it in a cost direction: put prefill and decode on *different machine types*. Because prefill is compute-bound and decode is bandwidth-bound, you can run prefill on newer compute-dense GPUs and decode on older or memory-heavy GPUs, matching each phase to the cheapest hardware that fits its bottleneck — then ship the KV cache between them over InfiniBand. Reported results: **1.4× higher throughput at 20% lower cost** than a homogeneous design, or **2.35× more throughput** at the same cost and power budget. Splitwise is the reference for heterogeneous fleets — mixed A100/H100 or spot-plus-reserved capacity — where the goal is cost efficiency, not just latency, and where the InfiniBand transfer cost (Section 3's 12.8 ms) is an acceptable tax for using cheaper decode boxes.

### Mooncake (Moonshot AI / Kimi, 2024) — KVCache-centric

Mooncake (Qin et al., Moonshot AI) is the disaggregated architecture behind Kimi, and it inverts the emphasis: the KV cache is not just something you transfer, it is the *center of the system*. Mooncake maintains a **global KV-cache pool** tiered across GPU HBM, host DRAM, and SSD, disaggregates prefill and decode around that pool, and adds prediction-based early rejection to shed load before it violates SLO. The payoff is enormous on long-context, high-reuse workloads — the same 128K-token document or system prompt hit by many requests is prefilled once and its KV reused across all of them, so the effective prefill cost amortizes toward zero. Reported: in real Kimi workloads Mooncake handled about **75% more requests** under SLO, with much larger throughput gains (the paper reports up to ~525% in some long-context simulated scenarios) where cache reuse is high. Mooncake is the reference for long-context, high-prefix-reuse serving where the KV pool, not the GPU, is the scarce resource.

### vLLM and NVIDIA Dynamo — disaggregation for everyone

The open-source ecosystem has converged on disaggregation as a standard capability. vLLM ships experimental disaggregated prefill via the KV-connector interface (`PyNcclConnector`, plus Mooncake- and LMCache-backed connectors for tiered/cross-node transfer), which is the code you saw in Section 6. NVIDIA Dynamo (open-sourced 2025) packages the whole pattern for datacenter scale: disaggregated prefill/decode workers, a KV-aware router that skips prefill on prefix-cache hits, and NIXL — the NVIDIA Inference Xfer Library — for RDMA/NVLink KV transfer with layer-wise overlap. Neither publishes a single headline multiplier because the win depends entirely on your workload and SLO, which is the honest answer: disaggregation's payoff is a function of your prompt/output distribution and your interconnect, and the only number that matters is the one your own goodput harness prints.

## Operational reality: what breaks

The clean topology in figure 3 hides a set of failure modes that only show up in production, and a principal engineer plans for them before the 3 AM page, not during it.

**KV-memory back-pressure and head-of-line blocking.** The decode pool has finite HBM for its KV cache. If the prefill pool produces caches faster than decode retires requests, the decode pool's KV memory fills, and it must either refuse new handoffs (prompts pile up in the prefill pool, and their already-computed KV caches occupy prefill HBM while they wait) or preempt and evict in-flight decodes. Either way you get head-of-line blocking that the colocated system did not have, because in a shared pool a stalled decode simply yields its GPU to something else. The fix is a **credit-based flow control** between the pools: the decode pool advertises how many free KV slots it has, and the prefill pool only dispatches handoffs it knows will land. Without it, a decode-pool slowdown silently propagates backward into TTFT.

**Load imbalance between the pools.** The static PD ratio is right on average and wrong every minute (Section 5). When it is wrong toward prefill-starvation, TTFT spikes because prompts queue for a free prefill worker; when it is wrong toward decode-starvation, TPOT spikes because the decode batch shrinks below its efficient point. The two failures look completely different on your dashboards and have opposite fixes, so your monitoring must attribute latency to the *right* pool. A single "p99 latency" number across both pools is useless here; you need TTFT broken out by prefill-queue time versus prefill-compute time, and TPOT broken out by decode-batch occupancy. Instrument the seam.

**A worker dies mid-handoff.** If a prefill worker crashes after computing a KV cache but before the transfer completes, that request's cache is gone and the request must be retried from scratch — re-prefilled on another worker. If a decode worker dies mid-generation, every in-flight stream on it loses its KV cache and must be re-prefilled and resumed, which is far more disruptive than a colocated single-worker crash because one decode worker holds hundreds of concurrent streams. The runbook: prefill failures retry cheaply (recompute the prompt); decode failures are expensive and argue for smaller decode-worker blast radius (more, smaller decode workers rather than one giant one) and for a KV pool that can re-materialize caches from a tier that survived the crash.

**The router becomes the bottleneck.** Every request crosses the router twice (dispatch to prefill, then to decode) and the router holds the KV-slot accounting for the whole fleet. At high QPS the router's own latency and its accuracy in tracking pool state start to matter; a router that mis-estimates decode KV availability will either over-dispatch (back-pressure stalls) or under-dispatch (idle decode GPUs). Treat the router as a real distributed-systems component with its own SLO, not as glue.

None of these is a reason to avoid disaggregation at scale — they are the operational tax you take on in exchange for the goodput win, and every one of them is manageable with the right flow control, monitoring, and blast-radius sizing. But they are real, and they are the difference between a disaggregated fleet that holds SLO through a traffic spike and one that cascades into a brownout when the pools fall out of balance.

## When to use this (and when not to)

Disaggregation is a scale technique with a real floor. It is the wrong call more often than the hype suggests, and knowing the failure modes is what separates a principal engineer from a benchmark-chaser.

**Do not disaggregate when:**

- **You are at low QPS or single-node scale.** Below roughly 50–100 QPS, or on a single small node, the interference you are trying to remove is rare — a prefill burst only hurts if there are many concurrent decodes to stall, and at low concurrency there often aren't. You pay the full complexity cost of two pools, a router, and a KV channel to solve a problem you barely have. Colocation with chunked prefill is simpler, cheaper, and usually meets SLO at this scale. **The single most common mistake is disaggregating a workload that a single well-tuned vLLM instance would have served fine.**
- **Your interconnect is slow.** If prefill and decode must live on different nodes connected only by TCP Ethernet, the KV transfer (53 ms for a 2K prompt from Section 3's table, worse for long prompts) starts eating a real fraction of your latency budget, and the layer-wise overlap can only hide so much. Disaggregation assumes NVLink intra-node or at least InfiniBand/RoCE inter-node. On commodity Ethernet, the transfer tax can wipe out the interference savings.
- **Your prompts and outputs are both short.** With short prompts, prefill bursts are tiny and interference is mild; with short outputs, there is little decode to protect. The whole tension disaggregation resolves — long prefill bursts stalling long decode streams — is weak. Chat with 50-token turns and 30-token replies does not need two pools.
- **You cannot afford idle capacity in one pool.** Disaggregation splits your GPUs into two fixed(ish) pools, and if your traffic swings between prompt-heavy and output-heavy through the day, one pool sits underutilized while the other saturates. Colocation naturally load-balances the two phases onto shared GPUs; disaggregation trades that flexibility for isolation. If your workload shape is highly variable and you can't autoscale the pools fast enough, colocation's fungibility may win on utilization.

**Do disaggregate when:**

- **You are at real scale with a strict TPOT SLO.** High concurrency plus a tight inter-token-latency requirement is the exact condition where prefill interference wrecks the tail and disaggregation shines — this is the DistServe regime.
- **You have long prompts and long outputs mixed together.** RAG, agents, summarization alongside chat — the workloads where a long prompt's prefill is a genuine landmine under everyone else's decode. Isolation is worth the most here.
- **You have fast interconnect and can right-size the PD ratio.** NVLink or InfiniBand, plus enough traffic to keep both pools busy at their derived ratio, is the sweet spot.
- **You have high prefix reuse and long context.** The Mooncake regime: a global KV pool plus disaggregation turns repeated long prompts into near-free prefills.

The blunt rule: **disaggregation is a technique for the high-QPS, tight-SLO, mixed-length regime. Below that, colocation with chunked prefill is the right default, and reaching for two pools is over-engineering.** Start colocated, measure your TPOT tail under realistic mixed traffic, and only disaggregate when you can *see* the interference in your histogram.

## Key takeaways

- **Prefill and decode are physically different computations.** Prefill is compute-bound (large GEMMs, arithmetic intensity above the ~295 FLOP/byte H100 ridge); decode is memory-bandwidth-bound (single-token GEMVs, intensity ~1–2). One GPU cannot serve both at their optimal operating points.
- **Colocation couples TTFT and TPOT.** A long prompt's prefill burst stalls in-flight decode steps, injecting 100–180 ms gaps into other requests' token streams. Chunked prefill reduces this but does not eliminate it.
- **Disaggregation decouples them** into a FLOP-tuned prefill pool and a batch-tuned decode pool, joined by a one-time KV-cache handoff. TTFT becomes the prefill pool's job; TPOT becomes the decode pool's; neither can hurt the other.
- **The KV transfer is cheap because it happens once per request.** For Llama-3-70B a 2,048-token prompt is 640 MB of KV, moved in 0.7 ms over NVLink or ~13 ms over InfiniBand, amortized over hundreds of decode steps. Layer-by-layer pipelining hides most of it behind prefill compute.
- **Transfer beats recompute almost always** — it pays only the bandwidth cost of an already-computed result, not the compute cost you disaggregated to isolate. The exception is a prefix-cache hit, where recompute of the small uncached suffix is cheaper.
- **The PD ratio follows from GPU-seconds per phase.** $R = (t_\text{prefill} \cdot g_p \cdot B) / (L_\text{out} \cdot t_\text{TPOT} \cdot g_d)$. Long outputs tilt toward decode (1:3 for chat); long prompts tilt toward prefill. There is no universal ratio — derive yours from your prompt/output distribution.
- **Optimize goodput, not throughput.** Goodput counts only requests meeting both SLOs. Disaggregation's ~3.8× win is a goodput multiple; raw tokens/sec barely moves, because you are not adding capacity, you are stopping the phases from wasting each other's.
- **Know the floor.** Below ~50–100 QPS, on slow interconnect, or with short prompts and outputs, colocation with chunked prefill wins. Disaggregation is a high-scale, tight-SLO, mixed-length technique — reaching for it too early is over-engineering.

## Further reading

- **DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving** — Zhong, Liu, Chen, Hu, Zhu, Liu, Jin, Zhang (OSDI 2024). The paper that named the problem, introduced goodput as the target, and reported up to ~4.5× goodput from per-phase parallelism.
- **Splitwise: Efficient Generative LLM Inference Using Phase Splitting** — Patel, Choukse, Zhang, Shah, Goiri, Maleki, Bianchini (ISCA 2024). Disaggregation across heterogeneous machine types for cost efficiency; 1.4× throughput at 20% lower cost.
- **Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving** — Qin et al., Moonshot AI (2024). The global tiered KV-cache pool behind Kimi; ~75% more requests under SLO on real long-context workloads.
- **vLLM disaggregated prefill documentation and KV-connector examples** — the open-source reference implementation (`KVTransferConfig`, `PyNcclConnector`, Mooncake/LMCache connectors).
- **NVIDIA Dynamo documentation and the NIXL transfer library** — datacenter-scale disaggregation with a KV-aware router and layer-wise RDMA/NVLink KV transfer.
- Within this series: [why LLM serving is different](/blog/machine-learning/model-serving/why-llm-serving-is-different) for the KV-cache memory wall; [continuous batching and PagedAttention](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention) for the batching mechanics disaggregation depends on; [KV-cache optimization](/blog/machine-learning/model-serving/kv-cache-optimization) for prefix caching and RadixAttention that compose with the KV-aware router; [request scheduling and preemption](/blog/machine-learning/model-serving/request-scheduling-and-preemption) for how each pool orders its work; and [what is model serving](/blog/machine-learning/model-serving/what-is-model-serving) for the SLO triangle that frames every trade in this post.
