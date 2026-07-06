---
title: "Running vLLM distributed in production: from one node to many, with the exact commands and the decisions behind them"
date: "2026-07-06"
publishDate: "2026-07-06"
description: "A hands-on, best-practice guide to actually running vLLM across GPUs and nodes: how to pick tensor, pipeline, and data parallelism, the exact serve flags, Ray and native multi-node bootstraps, the NCCL environment that makes or breaks bring-up, a Kubernetes LeaderWorkerSet, and how to verify it all works."
tags:
  [
    "model-serving",
    "inference",
    "vllm",
    "distributed-inference",
    "tensor-parallelism",
    "pipeline-parallelism",
    "data-parallelism",
    "ray",
    "kubernetes",
    "nccl",
    "multi-node",
    "llm-serving",
  ]
category: "machine-learning"
subcategory: "Model Serving"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/running-vllm-distributed-in-production-1.webp"
---

Someone on the platform team read that vLLM scales to many GPUs, so when the 70B model started missing its p99 latency budget they did the reasonable-sounding thing: they grabbed two 8-GPU nodes and launched `vllm serve --tensor-parallel-size 16`. The engine came up. Health check passed. And throughput was *worse* than the single node it replaced — roughly twelve tokens per second per request, GPUs sitting at thirty percent utilization, the interconnect saturated with tiny messages. They had doubled the hardware bill and made the service slower. The rollback ticket said "vLLM doesn't scale across nodes," which was exactly the wrong lesson.

vLLM scales across nodes beautifully. What does not scale is stretching a single tensor-parallel group across the boundary between two physical servers, because a tensor-parallel all-reduce fires on *every* transformer layer, and the moment that all-reduce has to cross from one node's NVLink fabric to another over the data-center network, you have taken the single most latency-sensitive collective in the whole forward pass and put it on the slowest wire in the building. The fix was not more hardware. It was one flag: keep tensor parallelism inside the node (`--tensor-parallel-size 8`) and cross the node boundary with pipeline parallelism (`--pipeline-parallel-size 2`) instead. Throughput went to fifty-five tokens per second per request and utilization to eighty-five percent, on the same two nodes.

![Two 8-GPU nodes forming one 16-GPU vLLM engine, with tensor-parallel shards inside each node on NVLink and a single pipeline stage boundary crossing between nodes](/imgs/blogs/running-vllm-distributed-in-production-1.webp)

This post is the hands-on companion to the architecture write-up. Where the [vLLM distributed architecture anatomy](/blog/machine-learning/model-serving/vllm-distributed-architecture-anatomy) explains what the executor, the workers, and the coordinator *are*, this one is about *running the thing*: which parallelism to pick and why, the exact `vllm serve` flags for one node and for many, both ways of standing up a multi-node cluster (a Ray cluster and vLLM's native data-parallel launcher), the container flags that keep NCCL from deadlocking, a Kubernetes `LeaderWorkerSet` that launches a multi-node replica as one logical unit, and a verification routine that tells you the engine is actually healthy rather than merely running. Every technique lands back on the serving SLO triangle — **latency ↔ throughput ↔ cost** — because every parallelism knob is a trade on that triangle. By the end you will be able to look at a model size and a hardware inventory and write down the `TP × PP × DP` layout that fits, respects the NVLink boundary, and hits your SLO, then launch and verify it from a cold cluster.

## 1. The three axes, and where each one is allowed to live

Distributed inference in vLLM composes three orthogonal ways to split work across GPUs. You have almost certainly seen the names; what matters for *running* it is not the definitions but the *placement rule* each one implies.

**Tensor parallelism (TP)** shards every weight matrix across GPUs. A linear layer of shape `[hidden, 4·hidden]` is split column-wise into `TP` slices; each GPU multiplies its slice, and then an all-reduce sums the partial results back into the full activation before the next layer can run. The defining property for placement: TP does a collective **every single layer**, twice per transformer block (once after attention, once after the MLP). For an 80-layer model that is 160 all-reduces per forward step. Those collectives are on the critical path — nothing downstream can start until the all-reduce finishes — so TP is only viable where the interconnect is fast. That means inside a node, on NVLink.

**Pipeline parallelism (PP)** splits the model by *layers*. Stage 0 holds layers 0 to k, stage 1 holds layers k+1 to the end. A request flows through stage 0, and the resulting activation is handed to stage 1 with a single point-to-point send/recv. The defining property: PP communicates **once per stage boundary**, not once per layer, and it moves one activation tensor rather than doing a global reduction. That is a couple of orders of magnitude less traffic than TP, and it tolerates a slower link. That is why PP is the axis you use to cross the node boundary.

**Data parallelism (DP)** replicates the whole model and load-balances independent requests across the replicas. There is no per-token communication between replicas at all; the only coordination is a lightweight scheduler deciding which replica gets the next request (and, for mixture-of-experts models, an all-to-all within the expert layers, which is its own topic). DP is how you add throughput once one replica already meets your latency target.

The single most important operational fact in this entire post is the **interconnect hierarchy**, because it is what turns those three definitions into a placement rule:

| Link | Typical bandwidth | Typical latency | Scope |
|---|---|---|---|
| NVLink / NVSwitch (H100) | ~900 GB/s per GPU | sub-microsecond | inside one node |
| PCIe Gen5 | ~64 GB/s | low microseconds | inside one node |
| InfiniBand NDR (400 Gb/s) | ~50 GB/s | single-digit microseconds | between nodes |
| Ethernet (RoCE / TCP) | ~12–25 GB/s effective | tens of microseconds | between nodes |

NVLink is roughly eighteen times the *bandwidth* of a 400 Gb/s InfiniBand link and dramatically lower latency, and the gap against ordinary Ethernet is larger still. Read the table as a physical constraint, not a preference: **TP wants NVLink, so TP must stay inside a node; PP and DP can tolerate the inter-node fabric, so they are the axes that cross the boundary.** vLLM does not enforce this for you — it will happily let you set `--tensor-parallel-size 16` across two 8-GPU nodes — which is exactly how the team in the intro shot themselves in the foot.

Put a number on why. A ring all-reduce of a tensor of `S` bytes across `T` GPUs moves about $2S(T-1)/T$ bytes on each GPU's link. During decode with batch `B` and hidden size `h` in BF16, the activation being reduced is roughly $B \cdot h \cdot 2$ bytes. Take `B = 32`, `h = 8192`: that is `32 × 8192 × 2 ≈ 0.5 MB` per all-reduce, and the ring moves about `0.9 MB` per GPU. On NVLink at 900 GB/s that transfer is on the order of a microsecond; over a 400 Gb/s InfiniBand link (~50 GB/s effective) it is closer to eighteen microseconds, before you even add the higher fixed network latency. Now multiply by the collective count: an 80-layer model does two all-reduces per block, 160 per token. On NVLink that is roughly `160 × 1 µs ≈ 0.16 ms` of communication per token — a rounding error against the ~10–20 ms of compute. Across nodes it is `160 × 18 µs ≈ 3 ms` of *pure* communication per token, on the critical path, stalling every GPU while the network crawls. That single multiplication is the entire reason TP must not cross the node boundary, and why the intro's `TP = 16` deployment ran at a crawl.

Pipeline parallelism, by contrast, sends one activation tensor of about `B · h · 2` bytes once per stage boundary — for `PP = 2` that is a single `~0.5 MB` send/recv per token, not 160 global reductions. Even over InfiniBand that is well under a microsecond of transfer plus one network hop's latency, which the pipeline can overlap with compute on the other stage. The communication asymmetry between the two axes is not marginal; it is two orders of magnitude, and it is what makes the placement rule non-negotiable.

If you want the fuller derivation of the all-reduce tax and the pipeline bubble from first principles — the queuing and bandwidth math behind these axes — that is the subject of the sibling post on [tensor, pipeline, and expert parallelism for serving](/blog/machine-learning/model-serving/tensor-pipeline-expert-parallelism-for-serving). Here we take the primitives as given and focus on composing and launching them.

## 2. Choosing the layout: a decision you can make on a napkin

Every distributed vLLM deployment starts with two questions, answered in order. First: **does the model fit in one node's aggregate HBM, with room left for KV cache?** Second: **does one replica meet the target QPS?** The answers determine your layout with almost no remaining freedom.

![Decision flow from model weights and GPUs-per-node to a final TP times PP times DP layout, branching on whether the model fits one node and whether one replica meets QPS](/imgs/blogs/running-vllm-distributed-in-production-2.webp)

Walk the branches:

- **Fits one node.** Set `TP = GPUs per node` and stop. A 70B model in BF16 is 140 GB of weights; an 8×H100 node has 640 GB of HBM. It fits with enormous room to spare, so you serve it with `--tensor-parallel-size 8` and nothing crosses the node boundary. This is the common case, and the most common *mistake* is reaching for multi-node when a single node would do.
- **Does not fit one node.** Now you must cross the boundary, and you cross it with pipeline parallelism. Keep `TP` at the number of GPUs per node (so each tensor-parallel group stays on NVLink) and set `PP` to the number of nodes you need. A 405B model in BF16 is 810 GB, larger than one 640 GB node, so it needs at least two nodes: `--tensor-parallel-size 8 --pipeline-parallel-size 2`.
- **One replica is too slow for your QPS.** Whether the replica is one node or several, if a single copy cannot keep up with request volume, add data-parallel replicas: `--data-parallel-size N`. Each replica is an independent `TP × PP` engine; the front end load-balances across them.

The three cases compose. A large model that also needs high throughput becomes `TP` (intra-node) `× PP` (across nodes, to fit) `× DP` (more replicas, for QPS). The layout is a product, and the constraint `TP ≤ GPUs_per_node` is inviolable. We will do the arithmetic that turns a model spec into these numbers in the next section.

One nuance worth stating plainly, because it saves a lot of grief: **prefer TP over PP whenever the model fits, and prefer PP over TP for crossing nodes.** Within a node, TP gives lower single-request latency than PP because it parallelizes each layer rather than pipelining across them, and it has no pipeline bubble. Across nodes, PP wins because its communication volume is tiny compared to TP's per-layer all-reduce. The layout that violates this — TP across nodes — is the one that looks reasonable and performs terribly.

As a starting-point cheat sheet for BF16 dense models on 8×H100 nodes (640 GB/node), before you do the precise arithmetic:

| Model (BF16) | Weights | Nodes | Layout |
|---|---|---|---|
| 7B–13B | 14–26 GB | 1 (partial) | `--tp 2` or `--tp 4` |
| 34B | ~68 GB | 1 | `--tp 4` (packs 2 replicas/node) |
| 70B | ~140 GB | 1 | `--tp 8` |
| 405B | ~810 GB | 2 | `--tp 8 --pp 2` |
| 671B | ~1.3 TB | 3–4 | `--tp 8 --pp 3` or `--pp 4` |

The pattern is mechanical: keep `TP` at 8 (or the largest divisor of the GPUs-per-node that the model needs), and grow `PP` by whole nodes until the weights fit. When you set `PP`, remember that vLLM splits *layers* as evenly as it can across stages, so a model whose layer count divides cleanly by `PP` gives a balanced split; an awkward layer count leaves one stage slightly heavier, which is one source of the uneven-memory signal you will check at verification time. You do not normally tune the split point by hand — vLLM's default even division is right almost always — but knowing it exists explains why a `PP = 3` engine on a 126-layer model lands 42 layers per stage cleanly while an oddly-sized model does not.

## 3. The placement math: turning a model spec into TP × PP × DP

Here is the mechanics block. A serving deployment must hold four things in GPU memory at once, and their sum, per GPU, must fit under the usable HBM:

$$M_\text{per GPU} = \frac{M_\text{weights} + M_\text{KV} + M_\text{act}}{\text{TP} \times \text{PP}} + M_\text{overhead} \;\le\; \gamma \cdot H$$

where `H` is the physical HBM per GPU (80 GB on an H100), and $\gamma$ is `--gpu-memory-utilization` (the fraction vLLM is allowed to claim, typically 0.90). The weights, KV, and activations are divided by `TP × PP` because those axes shard them across GPUs; overhead (CUDA context, NCCL buffers, allocator fragmentation, CUDA-graph capture — a few GB) does not shard and is charged per GPU.

The weight term is the floor you cannot argue with:

$$M_\text{weights} = P \cdot b_w$$

`P` is the parameter count and $b_w$ is bytes per parameter: 2 for BF16/FP16, 1 for FP8, 0.5 for INT4. The KV cache term, per token in flight, is

$$m_\text{kv} = 2 \cdot L \cdot h_\text{kv} \cdot d_\text{head} \cdot b_\text{kv}$$

with the leading 2 for the separate K and V tensors, `L` layers, $h_\text{kv}$ key/value heads (grouped-query attention shares KV heads across query heads, so this is smaller than the full head count), $d_\text{head}$ per-head dimension, and $b_\text{kv}$ the KV precision in bytes. Total KV memory is $m_\text{kv}$ times your concurrency in tokens — which is the knob that trades throughput for the memory you have left after weights.

The number of GPUs you need is the smallest `N = TP × PP` such that the inequality holds, and then you factor `N` respecting the NVLink boundary: `TP` is at most the GPUs per node, and `PP` is `N / TP` distributed one stage per node (or a few stages per node if a node holds more than one). Data parallelism multiplies the whole thing by the replica count and does not change the per-replica arithmetic.

#### Worked example: Does Llama-3.1-70B need one node or two?

Llama-3.1-70B has roughly `P = 70 × 10^9` parameters, `L = 80` layers, `h_kv = 8` KV heads, `d_head = 128`. In BF16, weights are `70e9 × 2 = 140 GB`. One 8×H100 node has `8 × 80 = 640 GB`, and at $\gamma = 0.90$ you can use about 576 GB. Weights take 140 GB, overhead maybe `8 × 3 = 24 GB`, leaving about **412 GB for KV cache**. Per-token KV is `2 × 80 × 8 × 128 × 2 = 327,680` bytes, about `0.31 MB`. So the node holds roughly `412 GB / 0.31 MB ≈ 1.3 million` KV tokens in flight — enough for thousands of concurrent long-context requests. The verdict is unambiguous: **70B fits one node with room to spare. Do not reach for a second node.** Serve it with `--tensor-parallel-size 8` and you are done. If you later need more throughput than one node can push, you add a *second replica* with data parallelism, not a second slice of the same model.

#### Worked example: Laying out Llama-3.1-405B on 2×8 H100

Now the model that actually forces the boundary. Llama-3.1-405B is about 810 GB in BF16, larger than one 640 GB node, so `N ≥ 2` nodes. With `TP = 8` (one tensor-parallel group per node, on NVLink) and `PP = 2` (one pipeline stage per node), each GPU holds `810 / 16 ≈ 50.6 GB` of weights. Add overhead and you are near 55 GB before any KV; on an 80 GB H100 at $\gamma = 0.90$ (72 GB usable) that leaves roughly 17 GB per GPU for KV — about 55 GB per node across the eight ranks, comfortably enough for a healthy batch. Every one of the sixteen ranks lands near the same 74 GB of the 80 GB it owns, so no single GPU is the odd one out that OOMs first.

![A 405B model placed across sixteen H100 GPUs with TP equals 8 and PP equals 2, showing two 63-layer pipeline stages each loaded to about 74 of 80 GB per rank](/imgs/blogs/running-vllm-distributed-in-production-8.webp)

That even loading is not cosmetic — it is a health signal. When you verify a multi-node engine later in this post, the first thing you check is that GPU memory is roughly equal across all ranks. A rank that is 6 GB heavier than its peers is a misconfiguration (an uneven layer split, a stray KV allocation) that will OOM under load while the other fifteen sit idle.

#### Worked example: quantizing to stay on one node

Before you accept a node boundary, price the alternative. That same 405B model in FP8 is `810 / 2 ≈ 405 GB` of weights, and it now fits a single 8×H100 node (640 GB, ~576 GB usable at $\gamma = 0.90$) with about 170 GB left for KV — enough for a real batch. So `vllm serve ... --tensor-parallel-size 8 --quantization fp8` on *one* node replaces the two-node BF16 layout entirely, at the cost of a small, usually-acceptable accuracy delta. The lesson generalizes: multi-node is the *last* resort, after "does a quantized checkpoint fit one node?" The order of preference — fit natively, quantize to fit, replicate for throughput, and only then span nodes — is not a stylistic choice; each step you avoid removes a fault domain and a slice of network latency from the critical path. FP8 on H100 is nearly free in latency because the tensor cores run it natively; INT4 (AWQ/GPTQ) squeezes further at a larger accuracy cost. The trade-offs are the subject of [quantization for LLM serving](/blog/machine-learning/model-serving/quantization-for-llm-serving); here the point is only that it belongs *above* multi-node in your decision order.

One more term deserves a word, because it is the one that causes surprise OOMs on a layout that looked like it fit: **activations**. During decode the activation footprint is small — roughly `batch × hidden × bytes`, a few hundred megabytes. During *prefill* of a long prompt it spikes to `batch × seq × hidden × bytes`, which for a 16k-token prompt is orders of magnitude larger and transient. This is why $\gamma = 0.90$ rather than 0.98: the ~10% headroom absorbs prefill spikes that the steady-state arithmetic does not show. Chunked prefill (below) caps how much of that spike lands at once, which is part of why it is a near-mandatory flag on long-context workloads.

## 4. Single node, multiple GPUs: the base case done right

Start where most services actually live: one node, several GPUs. This is the simplest distributed case and the one you should master before touching multi-node.

```bash
# Serve a 70B model tensor-parallel across all 8 GPUs of one node.
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 8 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --port 8000
```

A few things are happening here that are worth understanding rather than copying blind.

**vLLM auto-picks the executor.** When `TP × PP > 1` on a single node, vLLM selects a multiprocessing executor: it forks one worker process per GPU, each pinned to its device, and drives them over shared memory and NCCL. You do not have to ask for it. When `TP × PP == 1` it uses a single in-process executor (`UniProcExecutor`) — no forking, no NCCL. The multi-GPU single-node path is `MultiProcExecutor`, and it is the right default for one node: lower startup overhead than Ray and no external dependency. If you ever need to force it explicitly, that is `--distributed-executor-backend mp`.

**`--max-model-len` is a memory decision, not just a capability.** It caps the KV cache each request can consume. Set it to the longest context you actually serve, not the model's theoretical maximum. Every extra token of `max-model-len` that you will never use is KV budget you cannot spend on concurrency. Matching it to your real workload is one of the highest-leverage tuning moves in serving.

**`--gpu-memory-utilization` pins the KV pool.** vLLM computes how much HBM is left after weights and CUDA overhead, multiplies by $\gamma$, and pre-allocates that as the paged KV cache. Setting it too high (0.97, 0.98) leaves no room for the transient spikes of long-context prefill and gets you a mid-traffic OOM; too low wastes cache. 0.90 is a sane starting point; tune down if you see OOM under prefill-heavy load.

**Prefix caching and chunked prefill are nearly free wins.** Prefix caching reuses the KV of shared prompt prefixes (system prompts, few-shot examples) across requests; chunked prefill interleaves prefill chunks with decode steps so a long prompt does not stall everyone else's token generation. Both are toggles, both help almost every workload, and they are covered in depth in the [vLLM deep dive](/blog/machine-learning/model-serving/vllm-deep-dive).

The whole deployment, from the HTTP entry point down to the silicon, is the same six layers regardless of scale. Only the orchestration layer — the thing that spawns and connects the worker processes — changes as you go from one node to many.

![The distributed serving stack in six layers from the OpenAI-compatible API down through the vLLM engine, executor backend, orchestration, container, and GPUs](/imgs/blogs/running-vllm-distributed-in-production-3.webp)

That stack is the map for the rest of the post. The API, engine, and GPU layers are constant. The executor is `mp` on one node and `ray` across nodes. The orchestration layer is nothing on one node, a Ray cluster or vLLM's native launcher across nodes, and a `LeaderWorkerSet` on Kubernetes. The container layer wraps all of it. We will walk up from single-node to each of those in turn.

## 5. Pipeline parallelism, and mp versus ray

When a model does not fit one node, you add pipeline parallelism to cross the boundary. Syntactically it is one more flag:

```bash
# Serve a 405B model across two 8-GPU nodes: TP=8 inside each node, PP=2 across.
# (Run this once the Ray cluster from the next section is up; vLLM sees all 16 GPUs.)
vllm serve meta-llama/Llama-3.1-405B-Instruct \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --distributed-executor-backend ray \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --enable-chunked-prefill \
  --port 8000
```

The product `TP × PP = 8 × 2 = 16` must equal the total GPU count across the cluster. vLLM shards each pipeline stage's weights tensor-parallel across its node's 8 GPUs, and connects the two stages with a point-to-point activation transfer over the inter-node fabric.

The new decision here is the **executor backend**, and the rule is simple: **`mp` for a single node, `ray` for multi-node.** The multiprocessing executor can only spawn workers on the machine it runs on; it has no way to reach GPUs on another host. Ray does — it treats the two nodes as one logical GPU pool and places one worker actor per GPU wherever they physically are. So the moment `PP` or `TP` needs GPUs that are not all on one box, you need `--distributed-executor-backend ray` (and a running Ray cluster underneath it, which is the next section).

Here is the comparison you will actually reason from:

| Property | `mp` (MultiProcExecutor) | `ray` (RayDistributedExecutor) |
|---|---|---|
| Scope | single node only | single or multi-node |
| External dependency | none (built into vLLM) | requires a Ray cluster |
| Startup latency | lower (fork workers directly) | higher (Ray actor scheduling) |
| Elasticity / fault handling | none | Ray can reschedule actors on node loss |
| Best for | one node, `TP ≤ GPUs/node` | multi-node, or elastic/large fleets |
| How to select | `--distributed-executor-backend mp` | `--distributed-executor-backend ray` |

The practical takeaway: do not pay Ray's startup and operational cost on a single node — `mp` is faster to boot and has one less moving part. But do not try to bend `mp` across nodes; it cannot go there. Multi-node is Ray's job (or the native data-parallel launcher, for the DP-only case in section 8).

The failure mode that motivates all of this — putting the wrong axis across the boundary — is worth seeing side by side, because it is the single most common way a first multi-node deployment underperforms.

![Before and after comparison: TP equals 16 stretched across two nodes forces an all-reduce over InfiniBand every layer at about 12 tokens per second, while TP equals 8 intra-node with PP equals 2 keeps all-reduce on NVLink at about 55 tokens per second](/imgs/blogs/running-vllm-distributed-in-production-4.webp)

On the left, `TP = 16` means the per-layer all-reduce spans both nodes, so it rides InfiniBand. With 80 layers and two all-reduces per block, that is 160 inter-node collectives per token, each paying microseconds of network latency on the critical path; the GPUs spend most of their time waiting for the network and utilization collapses to around thirty percent. On the right, `TP = 8` keeps every all-reduce on the 900 GB/s NVLink inside each node, and `PP = 2` crosses the boundary with a single activation send/recv per token. Same sixteen GPUs, same model, roughly 4.5× the per-request throughput. The only thing that changed is *which collective was allowed to cross the wire*.

### Pipeline parallelism is not free: the bubble

PP has its own cost, and you should know it before you reach for it. When a request flows through a pipeline, only one stage works on it at a time — while stage 1 processes the request, stage 0 is idle with respect to that request. That idle time is the **pipeline bubble**. With naive single-request scheduling and `P` stages, a fraction $(P-1)/(P-1+m)$ of the pipeline's capacity is wasted, where `m` is the number of in-flight units keeping the stages fed. For `P = 2` and a single request (`m = 1`), that is a 50% bubble: each stage sits idle half the time, so a single-request latency benchmark on a `PP = 2` engine looks *worse* than the same model on one node, not better.

The rescue is that serving is never single-request. vLLM's continuous batching keeps many requests in flight simultaneously, and those requests are exactly the `m` units that fill the pipeline: while stage 1 decodes request A's next token, stage 0 is already decoding request B's, and so on. At a concurrency of a few dozen requests the bubble shrinks toward zero and both stages stay busy. So the operational truth about PP is a two-sided one: **PP is a capacity axis that pays off when the server is busy, and it adds latency at low load.** If your traffic is bursty with long idle troughs, a `PP = 2` engine will show higher TTFT during the quiet periods; if it is a steadily-loaded service, the bubble is a non-issue. This is the opposite of TP's profile (TP lowers single-request latency but has a hard per-layer communication cost), which is another reason the two axes belong in different places: TP inside the node for latency, PP across nodes for the capacity to fit and serve a model that a single node cannot hold.

## 6. Multi-node the first way: a Ray cluster

To run one engine across several nodes with tensor and pipeline parallelism, vLLM needs a Ray cluster spanning those nodes. You start Ray on a head node, join the workers, confirm the pool, then launch `vllm serve` on the head — vLLM discovers all the GPUs through Ray and places its workers across them.

```bash
# --- On node 0 (the head) ---
# Bind Ray to the NIC that carries your fast fabric, not the management NIC.
export VLLM_HOST_IP=10.0.0.1
ray start --head \
  --port=6379 \
  --node-ip-address=10.0.0.1 \
  --num-gpus=8

# --- On node 1 (each worker) ---
export VLLM_HOST_IP=10.0.0.2
ray start \
  --address=10.0.0.1:6379 \
  --node-ip-address=10.0.0.2 \
  --num-gpus=8

# --- Back on node 0: confirm the pool, then serve ---
ray status          # expect 16 GPUs total (2 nodes x 8), all "ACTIVE"
vllm serve meta-llama/Llama-3.1-405B-Instruct \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --distributed-executor-backend ray \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --port 8000
```

The order matters, and it matters in a way that bites silently rather than loudly. If you launch `vllm serve` before every worker has joined, vLLM will try to build a world of 16 ranks, find fewer, and either hang in NCCL initialization waiting for peers that never arrive or fail with a rank-count mismatch. The bring-up sequence is fixed, and the discipline of following it — check `ray status` shows all 16 GPUs *before* serving — is what turns a flaky launch into a reliable one.

![Multi-node bring-up timeline from ray start head through workers joining, ray status showing 16 GPUs, vllm serve, NCCL init across all 16 ranks, even GPU memory, and a health-checked test completion](/imgs/blogs/running-vllm-distributed-in-production-5.webp)

In production you do not run these commands by hand on each box — you bake them into a container and launch it identically everywhere, passing a role and the head address as arguments. The community `run_cluster.sh` pattern from the vLLM docs is exactly this: the same image runs on every node, and an argument decides whether it starts the Ray head or joins as a worker.

```bash
#!/usr/bin/env bash
# run_cluster.sh — start one node of a vLLM Ray cluster inside a container.
# Usage:
#   ./run_cluster.sh <image> <head_ip> <--head|--worker> <hf_cache_dir> [extra_env...]
IMAGE="$1"; HEAD_IP="$2"; ROLE="$3"; HF_CACHE="$4"; shift 4

if [ "$ROLE" = "--head" ]; then
  RAY_CMD="ray start --head --port=6379 --num-gpus=8 --block"
else
  RAY_CMD="ray start --address=${HEAD_IP}:6379 --num-gpus=8 --block"
fi

docker run \
  --gpus all \
  --network host \
  --shm-size 10.24gb \
  --ipc host \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -e VLLM_HOST_IP="$(hostname -i)" \
  -e NCCL_SOCKET_IFNAME=eth0 \
  -e GLOO_SOCKET_IFNAME=eth0 \
  "$@" \
  "$IMAGE" \
  /bin/bash -c "$RAY_CMD"
```

Four container flags in there are load-bearing, and skipping any of them produces a failure that looks like a vLLM bug but is really a Docker configuration bug:

- `--gpus all` exposes the node's GPUs to the container. Without it the container sees no CUDA devices and vLLM dies at import.
- `--network host` gives the container the host's network stack so Ray and NCCL can reach the other nodes at their real IPs. Bridge networking hides the container behind NAT and NCCL's peer discovery breaks.
- `--shm-size 10.24gb` enlarges `/dev/shm`. vLLM's workers move tensors between processes through shared memory; the default 64 MB is far too small and you get cryptic "Bus error" crashes under load.
- `--ipc host` shares the host IPC namespace so those shared-memory segments are visible across the worker processes. Together with `--shm-size`, this is what makes intra-node inter-process tensor transfer work.

You run this on node 0 with `--head` and on every other node with `--worker`, pointing all of them at the head IP. Once the cluster is up, `vllm serve` runs inside the head container exactly as before.

Two operational details about this pattern earn their keep. First, `ray start ... --block` keeps the container's main process alive; without `--block`, `ray start` daemonizes and returns, the container's entrypoint exits, and your orchestrator (Docker, Kubernetes) treats the pod as finished and kills it — taking the Ray node with it. The `--block` is what makes the container a long-lived cluster member. Second, the head node is special: it runs the Ray GCS (the cluster's control store) *and* the vLLM API server and the engine's rank-0 worker. Workers are interchangeable; the head is not. If the head dies, the cluster is gone; if a worker dies, the cluster is degraded and the engine on it is broken, but the GCS survives. This asymmetry is why, on Kubernetes, the head becomes the LWS *leader* and the workers become LWS *workers* — the API precisely mirrors the head/worker split Ray already has.

When something is wrong, `ray status` is your first probe. A healthy two-node cluster shows `16.0` total GPUs with both nodes listed and no `pending` or `failed` entries. A worker that failed to join shows fewer GPUs than expected — and critically, `vllm serve` launched against that degraded cluster will *not* error clearly; it will try to form a 16-rank world, block waiting for the missing ranks, and hang. That is the single most common multi-node symptom, and the discipline that prevents it is the one from the bring-up timeline: confirm `ray status` shows every GPU *before* you serve, never the other way around.

## 7. Multi-node the second way: native data parallelism

Ray is the answer when a single replica spans multiple nodes (you need `TP × PP` to reach GPUs on more than one box). But there is a second multi-node shape that does not need Ray at all: when each replica fits on part of a node and you want *many replicas* across nodes for throughput. vLLM has a native data-parallel launcher for exactly this, and it is the setup the anatomy post's two-node example uses.

The idea: each node runs some number of data-parallel replicas *locally*, one node hosts the API server and a coordinator, and the other nodes run "headless" (workers only, no HTTP front end). A `DPCoordinator` fans incoming requests to local and remote replicas over an RPC port and merges their token streams back to the caller.

![Native data-parallel topology across two nodes: an API server and DPCoordinator on node 1 fan requests over the RPC port to local replicas and to headless replicas on node 0, then merge the token streams back to the client](/imgs/blogs/running-vllm-distributed-in-production-6.webp)

Here is the exact launch, matching the vLLM anatomy reference: four data-parallel replicas total, two local to each node, each replica tensor-parallel across four GPUs. Node 0 runs headless; node 1 runs the API server. The `--data-parallel-start-rank` offsets which global replica ranks live on each node.

```bash
# --- On node 0 (headless: workers only, no API server) ---
# This node hosts data-parallel ranks 0 and 1.
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 4 \
  --data-parallel-size 4 \
  --data-parallel-size-local 2 \
  --data-parallel-start-rank 0 \
  --data-parallel-address 10.0.0.1 \
  --data-parallel-rpc-port 13345 \
  --headless

# --- On node 1 (the API server; hosts DP ranks 2 and 3) ---
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 4 \
  --data-parallel-size 4 \
  --data-parallel-size-local 2 \
  --data-parallel-start-rank 2 \
  --data-parallel-address 10.0.0.1 \
  --data-parallel-rpc-port 13345 \
  --port 8000
```

Decode the flags, because each one is doing real work:

- `--data-parallel-size 4` — four replicas total across the cluster.
- `--data-parallel-size-local 2` — of those four, two live on *this* node. Two nodes × two local = four. This is how vLLM knows how many worker groups to spawn per host.
- `--data-parallel-start-rank 0` / `2` — the global index of this node's first replica. Node 0 owns ranks 0–1, node 1 owns ranks 2–3. Get this wrong and two nodes claim the same rank and the coordinator rejects the duplicate.
- `--data-parallel-address 10.0.0.1` — the IP the coordinator binds to and every node dials. It is the same on both nodes (it names the coordinator, not the local host).
- `--data-parallel-rpc-port 13345` — the coordinator's RPC port for the control channel between replicas. Same on both nodes; must be open in any firewall between them.
- `--headless` — on node 0, run workers only, no HTTP server. Requests enter through node 1's API server and are dispatched to node 0's replicas over the RPC channel.

The picture is the figure above: one API server, one coordinator, four independent engines, no per-token cross-node collectives — only the lightweight RPC that assigns each request to a replica and streams tokens back. Because the replicas are independent, this scales throughput almost linearly with replica count, and it survives a replica dying with far less drama than a tensor-parallel group does (losing one replica costs you `1/N` of capacity, not the whole engine).

The coordinator does not round-robin blindly. It balances by *load* — it tracks each replica's queue depth and in-flight work and steers a new request to the least-loaded replica, which matters for LLM serving because request costs are wildly uneven (a 50-token completion and a 4,000-token one land in the same queue). Round-robin would routinely pile a long generation behind another long generation on one replica while a neighbor sits idle; least-loaded routing keeps the batches balanced and the tail latency down. This is also why data parallelism composes cleanly with the KV-cache and prefix-caching optimizations: each replica manages its own paged cache independently, so a prefix cache hit on one replica is not invalidated by traffic on another. The one caveat is prefix-cache locality — if you rely heavily on shared-prefix reuse, session-affinity routing (pinning a conversation to the replica that already holds its prefix) can beat pure least-loaded, at the cost of less even load. It is a genuine trade, not a free lunch.

A note for mixture-of-experts models: DP and EP interact. In a large MoE served with expert parallelism, the "data-parallel" replicas may themselves share an expert-parallel all-to-all, so the clean independence above is partial — the all-to-all is another latency-sensitive collective that wants the fast fabric. The placement rule still holds (keep the all-to-all on NVLink where you can), but the fault-independence of DP is reduced. That composition is the subject of [serving MoE models at scale](/blog/machine-learning/model-serving/serving-moe-models-at-scale); for dense models the replicas are fully independent as described.

#### Worked example: a data-parallel throughput setup

Suppose your 70B service (one 8×H100 node, `TP = 8`) meets latency at 20 requests per second but marketing is about to drive 60 RPS. One replica is fast enough per-request; you just need three of them. Rather than one big engine, stand up `--data-parallel-size 3` — but note each replica here still wants a full node if you keep `TP = 8`, so this is three nodes of one replica each, front-ended by one coordinator. If instead your model is 34B (fits on 4 GPUs at `TP = 4`), you can pack two replicas per 8-GPU node and get four replicas on two nodes with the native launcher above — `4 × ~20 = ~80 RPS` from two nodes, no Ray required. The choice between "one big TP engine per node" and "several small TP engines per node" is a latency-versus-throughput trade: bigger TP lowers single-request latency; more DP replicas raise aggregate throughput. Size the replica to hit latency, then multiply with DP to hit throughput.

## 8. The environment that makes or breaks bring-up

More multi-node vLLM launches die in NCCL initialization than anywhere else, and almost always for the same reason: the collectives library picked the wrong network interface. A node usually has several NICs — a management interface, a storage interface, and the fast InfiniBand or RoCE fabric — and if NCCL or Gloo binds to the management NIC instead of the fabric, you get either a hang (peers cannot reach each other) or catastrophic slowness (traffic routed over a 1 Gb/s management link). You must tell them explicitly which interface to use.

```bash
# The environment block every multi-node vLLM node should export before serving.

# The IP this node advertises to peers — must be on the fast fabric's subnet.
export VLLM_HOST_IP=10.0.0.2

# Force NCCL and Gloo onto the fabric NIC. Find yours with `ip addr`; it is the
# interface on the 10.0.0.0/24 fabric subnet, NOT eth0/the mgmt NIC.
export NCCL_SOCKET_IFNAME=ens6np0
export GLOO_SOCKET_IFNAME=ens6np0

# On an InfiniBand cluster, name the HCA(s) so NCCL uses RDMA, not TCP.
# List them with `ibv_devices`; omit this block entirely on Ethernet/RoCE.
export NCCL_IB_HCA=mlx5_0,mlx5_1

# Observability while bootstrapping. Turn NCCL_DEBUG down to WARN once stable.
export NCCL_DEBUG=INFO
export VLLM_LOGGING_LEVEL=DEBUG
```

What each variable buys you:

- `VLLM_HOST_IP` — the address this vLLM process advertises to its peers. If it defaults to a management-NIC address, workers exchange the wrong IPs and NCCL rendezvous fails. Pin it to the fabric subnet.
- `NCCL_SOCKET_IFNAME` — the interface NCCL uses for its bootstrap and (on Ethernet) its data. This is the single most common fix for a hanging multi-node launch. Set it to the fabric NIC name.
- `GLOO_SOCKET_IFNAME` — the same, for the Gloo backend vLLM uses for some CPU-side control collectives. Set it alongside `NCCL_SOCKET_IFNAME`; forgetting it causes a subset of the coordination to hang while NCCL itself looks fine.
- `NCCL_IB_HCA` — on InfiniBand, restricts NCCL to the named host channel adapters so it uses RDMA verbs rather than falling back to slow TCP. Wrong or missing HCA names are why an "InfiniBand" cluster sometimes runs at Ethernet speeds.
- `NCCL_DEBUG=INFO` — prints the rings and channels NCCL builds and, crucially, *which transport it chose per pair of ranks*. During bring-up this is your ground truth for whether the fabric is actually being used. Turn it to `WARN` in steady state so logs stay quiet.
- `VLLM_LOGGING_LEVEL=DEBUG` — surfaces vLLM's own worker-startup and rank-assignment logs. Invaluable the first time; noisy forever after.

Set this block identically on every node (adjusting only `VLLM_HOST_IP`), and set it *before* `ray start` and `vllm serve`, because NCCL reads it at process launch. Getting this right once, in your container image, is the difference between a bring-up that works every time and one that hangs on a coin flip.

The symptoms map to causes with surprising reliability, so keep this table near the runbook:

| Symptom | Most likely cause | Fix |
|---|---|---|
| Launch hangs at "initializing NCCL", no error | NCCL bound to the wrong NIC; peers cannot rendezvous | set `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` to the fabric NIC |
| Engine runs but decode is 10× too slow | NCCL fell back to TCP instead of RDMA | set `NCCL_IB_HCA` to the real HCA names; confirm with `NCCL_DEBUG=INFO` |
| "Bus error" / crash under load | `/dev/shm` too small for inter-worker transfer | raise `--shm-size` and pass `--ipc host` |
| Rank count mismatch at startup | served before all workers joined | wait for `ray status` to show every GPU, then serve |
| Worker cannot reach coordinator | RPC port blocked or wrong `--data-parallel-address` | open `--data-parallel-rpc-port`; verify the fabric IP |
| One rank OOMs, others fine | uneven layer split or a rank missing its KV pool | check even GPU memory; verify NIC/HBM on the heavy rank |

For the systematic version of diagnosing these hangs — reading the NCCL logs, isolating the bad NIC, catching a firewall drop on the RPC port — see the sibling runbook on [debugging vLLM distributed serving](/blog/machine-learning/model-serving/debugging-vllm-distributed-serving).

## 9. Containerizing for reproducibility

Everything above should live in a container so that every node runs bit-identical software and the launch is one command. Start from NVIDIA's CUDA base image (or vLLM's official image, which is built on it), because it ships the matched CUDA, cuDNN, and NCCL that the vLLM wheels expect — mismatched CUDA/NCCL between host and container is a rich source of "works on one node, hangs on another" incidents.

```dockerfile
# A minimal, reproducible vLLM serving image.
FROM nvcr.io/nvidia/pytorch:24.10-py3

# Pin the vLLM version — never float it across a fleet, or two nodes end up on
# different NCCL protocol versions and rendezvous fails.
ARG VLLM_VERSION=0.6.3
RUN pip install --no-cache-dir "vllm==${VLLM_VERSION}"

# Bake the env defaults that are the same everywhere. Per-node values
# (VLLM_HOST_IP) and the fabric NIC name are passed at `docker run` time.
ENV NCCL_DEBUG=WARN \
    VLLM_LOGGING_LEVEL=INFO \
    HF_HOME=/root/.cache/huggingface

EXPOSE 8000
ENTRYPOINT ["vllm", "serve"]
```

And the run command carries the flags that are about the *container*, not the model:

```bash
docker run --rm \
  --gpus all \
  --network host \
  --ipc host \
  --shm-size 10.24gb \
  -v /data/hf-cache:/root/.cache/huggingface \
  -e VLLM_HOST_IP=10.0.0.2 \
  -e NCCL_SOCKET_IFNAME=ens6np0 \
  -e GLOO_SOCKET_IFNAME=ens6np0 \
  my-registry/vllm-serve:0.6.3 \
  meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 8 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --port 8000
```

The `--gpus all`, `--ipc host`, and `--shm-size` flags are the same load-bearing ones from the Ray script and for the same reasons — GPU access, cross-process shared memory visibility, and a `/dev/shm` large enough for inter-worker tensor transfer. Mount the Hugging Face cache as a volume so you download the checkpoint once per node and reuse it across restarts; on a 405B model that is the difference between a 90-second restart and a 30-minute one. The broader Docker-for-GPU discipline — multi-stage builds to shrink the image, pinning the CUDA minor version, not baking secrets into layers — is its own subject, but the four flags here are the ones specific to distributed vLLM.

## 10. Kubernetes: one multi-node replica as a LeaderWorkerSet

On Kubernetes, a multi-node vLLM replica is not a normal `Deployment`. A `Deployment` gives you N interchangeable, independent pods; a multi-node engine is the opposite — a *group* of pods that must come up together, discover each other, form one NCCL world, and be treated as a single unit for scaling and rolling updates. The `LeaderWorkerSet` (LWS) API exists precisely for this "one logical replica made of several pods" shape. One leader pod runs the Ray head and the vLLM API server; a fixed number of worker pods join the Ray cluster. LWS guarantees they schedule and restart as a group.

```yaml
apiVersion: leaderworkerset.x-k8s.io/v1
kind: LeaderWorkerSet
metadata:
  name: vllm-405b
spec:
  replicas: 1                 # one multi-node engine; bump for more DP replicas
  leaderWorkerTemplate:
    size: 2                   # 1 leader + 1 worker = 2 nodes
    restartPolicy: RecreateGroupOnPodRestart   # a dead pod recreates the group
    leaderTemplate:
      metadata:
        labels: { role: leader }
      spec:
        containers:
          - name: vllm-leader
            image: my-registry/vllm-serve:0.6.3
            command: ["/bin/bash", "-c"]
            args:
              - |
                ray start --head --port=6379 --num-gpus=8 --block &
                # Wait for the worker to join before serving (see verify script).
                until [ "$(ray status 2>/dev/null | grep -c 'GPU')" -ge 1 ]; do sleep 2; done
                vllm serve meta-llama/Llama-3.1-405B-Instruct \
                  --tensor-parallel-size 8 --pipeline-parallel-size 2 \
                  --distributed-executor-backend ray \
                  --max-model-len 16384 --gpu-memory-utilization 0.90 --port 8000
            env:
              - name: NCCL_SOCKET_IFNAME
                value: "ens6np0"
              - name: GLOO_SOCKET_IFNAME
                value: "ens6np0"
            resources:
              limits: { nvidia.com/gpu: 8 }
            ports:
              - containerPort: 8000
            readinessProbe:
              httpGet: { path: /health, port: 8000 }
              initialDelaySeconds: 120     # weight load on 405B is slow; be patient
              periodSeconds: 10
    workerTemplate:
      spec:
        containers:
          - name: vllm-worker
            image: my-registry/vllm-serve:0.6.3
            command: ["/bin/bash", "-c"]
            args:
              - |
                ray start --address=$(LWS_LEADER_ADDRESS):6379 --num-gpus=8 --block
            env:
              - name: NCCL_SOCKET_IFNAME
                value: "ens6np0"
            resources:
              limits: { nvidia.com/gpu: 8 }
```

Three details make this work in practice. First, `LWS_LEADER_ADDRESS` is injected by LWS into every worker pod, so workers find the leader without you hard-coding an IP — that is the whole reason to use LWS rather than assembling pods by hand. Second, `restartPolicy: RecreateGroupOnPodRestart` means that if any pod in the group dies, the *entire* group is recreated; this is correct, because a surviving half of a NCCL world is useless and would otherwise wedge. Third, the `readinessProbe` on `/health` with a long `initialDelaySeconds` keeps the pod out of the Service's endpoints until vLLM has finished loading weights and initializing all ranks — without it, the load balancer sends traffic to an engine that is still allocating and every request times out. To add throughput, raise `replicas` (more independent DP replicas of the whole two-node engine); to change the engine size, change `size` and the `TP × PP` flags together.

Two scheduling concerns lurk beneath this YAML and are worth naming, because they cause "the pods are Pending forever" incidents. The first is **gang scheduling**: a multi-node engine needs *all* its pods running at once or none are useful, but the default Kubernetes scheduler places pods one at a time and can deadlock — it schedules the leader, then finds no room for the worker, and the leader sits idle holding 8 GPUs while the worker waits for a node that never frees up. On a busy cluster you want a gang-aware scheduler (Volcano, or Kueue) that admits the whole group atomically or not at all. The second is **topology**: the two pods of one replica should land on nodes that share the same high-speed fabric (the same InfiniBand leaf switch, the same NVLink domain where relevant), not on nodes chosen at random across the cluster. Node affinity and topology-aware placement (matching on a rack or switch label) keep the pipeline handoff on the fast path instead of routing it across the data center. Both of these — plus the GPU resource-limit and node-labeling mechanics — live in the autoscaling and GPU-scheduling story in [GPU scheduling, MIG, and autoscaling](/blog/machine-learning/model-serving/gpu-scheduling-mig-and-autoscaling), and the operational depth of running such engines across nodes is the subject of [multi-node LLM serving for 100B+ models](/blog/machine-learning/model-serving/multi-node-llm-serving-100b-plus).

## 11. Verifying it actually works

An engine that is "running" is not the same as an engine that is *healthy*. A multi-node launch can pass its own startup and still be quietly broken: one rank on the wrong NIC, an uneven memory split, a coordinator that never saw a worker. Four checks, in order, catch essentially all of it.

```bash
#!/usr/bin/env bash
# verify-vllm.sh — confirm a distributed vLLM engine is genuinely healthy.
set -euo pipefail
HOST=${1:-localhost}; PORT=${2:-8000}

# 1. The Ray pool has every GPU you expect (run on the head).
echo "== ray status =="
ray status | grep -E "GPU|node_" || true   # expect e.g. 16.0 GPU total, all ACTIVE

# 2. The HTTP server is up and the engine reports ready.
echo "== /health =="
curl -fsS "http://${HOST}:${PORT}/health" && echo "  health OK"

# 3. GPU memory is EVEN across all ranks — the key multi-node health signal.
#    A rank far heavier or lighter than its peers is a misconfiguration.
echo "== per-GPU memory (should be ~equal across ranks) =="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader

# 4. A real completion actually returns tokens end to end.
echo "== test completion =="
curl -fsS "http://${HOST}:${PORT}/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3.1-405B-Instruct",
       "prompt":"The capital of France is","max_tokens":8,"temperature":0}' \
  | python -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['text'])"
```

Read the results the way an on-call engineer would. `ray status` must show the full GPU count with every node `ACTIVE`; a missing GPU means a worker did not join and the engine is running degraded or hung. `/health` returning 200 means the API server and engine finished initialization. The **GPU memory check is the one people skip and should not** — on a correctly laid-out engine, every rank sits within a gigabyte or two of its peers (recall the ~74 GB per rank in the 405B example); a rank that is several GB heavier or lighter tells you the pipeline split is uneven or a NIC problem left one rank without its KV pool, and that rank will OOM first under load. Finally, a real completion proves the whole path — scheduler, all ranks, the pipeline handoff, detokenization — actually produces tokens, which a health check alone does not.

### Warm up before you take traffic

An engine that just finished loading is *cold*: its CUDA graphs are not yet captured, its prefix cache is empty, and its allocator has not settled. The first real requests pay for all of that, so if the load balancer routes production traffic to a freshly-ready pod, those users eat multi-second first tokens and the pod looks like it is failing its SLO when it is merely warming. The fix is to warm it yourself and only signal readiness once it responds.

```bash
#!/usr/bin/env bash
# warmup.sh — drive a few synthetic requests so CUDA graphs capture and the
# allocator settles before the pod is marked ready. Exit non-zero on failure so
# a readiness gate keeps the pod out of rotation until warmup succeeds.
set -euo pipefail
HOST=${1:-localhost}; PORT=${2:-8000}; MODEL=${3:?model name required}
for len in 8 128 1024; do          # short, medium, and a longer prefill path
  curl -fsS "http://${HOST}:${PORT}/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL}\",\"prompt\":\"warmup\",\"max_tokens\":${len},\"temperature\":0}" \
    >/dev/null
done
echo "warmup complete"
```

Run this after `/health` first returns 200 and before you flip the pod into the Service's endpoints — in Kubernetes, gate it with a `startupProbe` or an init step so the `readinessProbe` only passes post-warmup. On a large model the warmup exercises the full pipeline (both stages, all ranks, the cross-node handoff) at a few sequence lengths, which also doubles as a final correctness check: if warmup hangs, the engine has a rank problem you want to catch before users do, not after.

Once it is healthy and warm, benchmark it under realistic load before you call it done, because steady-state throughput and tail latency are what your SLO is written against:

```bash
# Drive realistic load and report TTFT, TPOT, throughput, and p99.
vllm bench serve \
  --backend vllm \
  --model meta-llama/Llama-3.1-405B-Instruct \
  --base-url http://localhost:8000 \
  --dataset-name sharegpt \
  --num-prompts 1000 \
  --request-rate 20 \
  --percentile-metrics ttft,tpot,itl,e2el
```

The benchmark reports time-to-first-token (TTFT), time-per-output-token (TPOT), and end-to-end latency percentiles at your chosen request rate. Run it at a few request rates to find the knee where p99 latency starts climbing — that knee is your per-replica capacity, and it is the number you feed the autoscaler. For a full treatment of turning these numbers into a capacity plan, see [load testing and capacity planning](/blog/machine-learning/model-serving/load-testing-and-capacity-planning) in this series.

The whole decision surface — scenario to flags to executor to expected behavior — collapses into one table you can pin above your desk.

![Config recipe matrix mapping five scenarios from a 7B on a single GPU to a 405B on two nodes to their GPUs, parallelism flags, executor backend, and expected throughput](/imgs/blogs/running-vllm-distributed-in-production-7.webp)

Here is the same recipe set as text, which is the version you will actually copy from:

| Scenario | GPUs | Flags | Executor | Expected |
|---|---|---|---|---|
| 7B, single GPU | 1×A100 80GB | (none) | `UniProc` (auto) | ~2.5k tok/s aggregate |
| 13B, one node | 2×A100 | `--tp 2` | `mp` (auto) | ~1.9k tok/s |
| 70B, one node | 8×H100 | `--tp 8` | `mp` (auto) | ~1.8k tok/s |
| 405B, two nodes | 2×8 H100 | `--tp 8 --pp 2` | `ray` | ~30 tok/s per request |
| 70B, max QPS | 2×8 H100 | `--tp 4 --dp 4` | `ray` (native DP) | ~4× replica throughput |

The throughput figures are order-of-magnitude, workload-dependent guides (they move with sequence length, batch, and precision), not guarantees — treat them as "what good looks like," and measure your own with the benchmark above.

## 12. Observability: the metrics that tell you the layout is right

Verification is a point-in-time check; observability is the continuous version, and on a distributed engine it is what tells you the difference between "healthy" and "one bad rank away from a page." vLLM exposes Prometheus metrics on `/metrics`, and a handful of them are the ones you actually alert on.

```bash
# The distributed-serving metrics worth scraping and alerting on.
curl -s http://localhost:8000/metrics | grep -E \
  'vllm:num_requests_(running|waiting)|vllm:gpu_cache_usage_perc|vllm:time_to_first_token_seconds|vllm:time_per_output_token_seconds'
```

Read them as a small dashboard:

- **`vllm:num_requests_waiting`** — the scheduler's queue depth. A queue that rises and does not drain means one replica cannot keep up: you are past the p99 knee and need another DP replica, not a bigger TP group. This is your primary autoscaling signal, far better than CPU or raw GPU utilization.
- **`vllm:gpu_cache_usage_perc`** — how full the paged KV cache is. Sustained readings near 100% mean the engine is about to start preempting (evicting and recomputing) requests, which shows up as latency cliffs. The fix is more KV budget: lower `--max-model-len`, add a replica, or (if weights are the squeeze) quantize. This metric is the early warning that the memory math from section 3 is being violated by real traffic.
- **`vllm:time_to_first_token_seconds`** and **`vllm:time_per_output_token_seconds`** — TTFT and TPOT histograms. Alert on their p99 against your SLO. On a `PP > 1` engine, a TTFT that climbs specifically at low load is the pipeline bubble from section 5 showing up in production, not a bug.
- **`vllm:num_requests_running`** — the current batch size. If it is pinned at your `max-num-seqs` while the queue grows, you are capacity-bound; if it is far below and the queue is empty, you are over-provisioned and paying for idle GPUs.

The distributed-specific discipline is to scrape these **per replica and per rank**, not just aggregate. An aggregate TPOT can look fine while one DP replica is silently slow because its worker landed on a throttled NIC; only the per-replica view shows the outlier. Pair the Prometheus metrics with the per-GPU `nvidia-smi` memory check on a short interval, and a single dashboard tells you at a glance whether every rank is carrying its share. The full treatment — dashboards, trace propagation across the API server and workers, and the alert rules — is in [observability for LLM serving](/blog/machine-learning/model-serving/observability-for-llm-serving); the point here is that a distributed engine needs per-rank visibility, because the failures that matter are the asymmetric ones.

## 13. When a node dies: failure domains by axis

A 3 AM page for a dead GPU has very different blast radius depending on which axis that GPU was part of, and designing for that is the difference between a degraded service and an outage. The rule is that **communication coupling equals fault coupling**: the tighter an axis binds GPUs together, the larger the unit that fails together.

| Axis | What one lost GPU takes down | Blast radius | Recovery |
|---|---|---|---|
| Tensor parallel | the whole TP group (weights are sharded; a missing shard cannot run) | one node | restart the replica |
| Pipeline parallel | the whole engine (you cannot skip layers) | all nodes of the replica | recreate the pod group |
| Data parallel | one replica only | `1/N` of capacity | reschedule that replica; peers keep serving |

Tensor parallelism has the tightest coupling: every GPU in the group holds a slice of every layer, so if one dies, the group cannot complete a single forward pass — the whole node's worth of GPUs is down. Pipeline parallelism is no gentler across the replica: a stage with a dead GPU cannot produce the activations the next stage needs, so the entire multi-node replica halts. Data parallelism is the only axis that fails *gracefully*: replicas are independent, so losing one drops throughput by `1/N` while the survivors keep answering.

This has a direct design consequence. A single multi-node replica (`TP × PP`, one copy) is a single fault domain — one GPU failure anywhere is a full outage. **For any service with an availability SLO, run at least two DP replicas**, so a node loss degrades capacity rather than taking the service to zero. On Kubernetes, this is why the `LeaderWorkerSet` uses `RecreateGroupOnPodRestart`: a half-alive NCCL world cannot heal in place, so the correct response to any pod loss is to recreate the whole group, and meanwhile a *second* LWS replica keeps traffic served. Ray can reschedule actors when a node returns, but it cannot stitch a live request's NCCL communicators back together mid-flight — in-flight requests on the failed replica are lost and must be retried against a healthy one, which is why a retry-with-backoff at the gateway is part of every serious deployment. The operational depth of these recovery flows — draining, re-registration, the node-death runbook — is covered in [multi-node LLM serving for 100B+ models](/blog/machine-learning/model-serving/multi-node-llm-serving-100b-plus); the takeaway here is that your replica count, not your cleverness, is what buys availability.

## 14. Best practices, consolidated

The rules that survive contact with production, in one place:

- **Keep `TP ≤ GPUs per node.** This is the cardinal rule. Tensor parallelism must stay on NVLink; cross the node boundary with pipeline or data parallelism, never tensor parallelism.
- **Do not multi-node a model that fits one node.** A second node adds network latency, a fault domain, and operational cost. If the weights plus a healthy KV budget fit one node, serve it on one node and scale out with DP replicas only when QPS demands it.
- **Use `mp` on one node, `ray` for multi-node.** Do not pay Ray's overhead on a single box; do not try to stretch `mp` across boxes.
- **Match `--max-model-len` to your real workload.** Every unused token of context is KV budget you cannot spend on concurrency.
- **Pin `--gpu-memory-utilization` around 0.90.** Leave headroom for prefill spikes; tune down if you OOM under load, not up to squeeze the last gigabyte.
- **Set the NCCL/Gloo interface variables explicitly.** `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, `VLLM_HOST_IP`, and (on IB) `NCCL_IB_HCA` are not optional on multi-node; the default NIC selection is the number-one cause of hangs.
- **Enable prefix caching and chunked prefill.** Near-free wins on almost every workload.
- **Warm up before you take traffic.** The first requests trigger CUDA-graph capture and cache population; send a few synthetic completions and only mark the pod ready (via the `/health` readiness probe) once it responds, so the load balancer never sees a cold engine.
- **Verify memory is even across ranks after every launch.** It is the fastest signal that the layout is correct and no rank will OOM first.
- **Pin your vLLM and CUDA/NCCL versions across the whole fleet.** Version skew between nodes breaks NCCL rendezvous in ways that look like network problems.

## Case studies

**vLLM + Ray, official multi-node docs.** The `run_cluster.sh` pattern in section 6 is drawn directly from vLLM's own distributed-serving documentation, which recommends the same shape: one image on every node, Ray to unify the GPU pool, `--tensor-parallel-size` equal to GPUs-per-node and `--pipeline-parallel-size` equal to node count, and the `--gpus all --shm-size --ipc host --network host` container flags. The docs are explicit that tensor parallelism should not span nodes and that pipeline parallelism is the axis for the boundary — the same rule we derived from the interconnect table. Treat the version numbers and exact flag spellings in this post as accurate to vLLM 0.6.x; always cross-check against the docs for the release you deploy, since flag names occasionally change across minor versions.

**Serving Llama-3.1-405B.** The 810 GB BF16 checkpoint is the canonical "genuinely needs multiple nodes" model, and the community-standard recipe is the `TP = 8 × PP = 2` on two 8×H100 nodes we laid out in section 3 — roughly 50 GB of weights per GPU, even loading near 74 of 80 GB per rank, and per-request decode in the tens of tokens per second range depending on context length and batch. An FP8 quantized variant halves the weight memory and can fit an 8×H100 node with `TP = 8` alone, eliminating the node boundary entirely — a reminder that quantization is often the cheaper alternative to going multi-node, covered in [quantization for LLM serving](/blog/machine-learning/model-serving/quantization-for-llm-serving). The specific throughput numbers here are approximate and hardware- and workload-dependent; the layout and the memory arithmetic are the durable part.

**DeepSeek-scale and mixture-of-experts.** Very large MoE models (DeepSeek-V3-class, in the hundreds of billions of total parameters) add a fourth axis, expert parallelism, which shards the experts across GPUs and introduces an all-to-all collective in the MoE layers. The placement logic extends naturally: keep the latency-sensitive collectives (TP all-reduce, and ideally EP all-to-all) on the fastest fabric, and use PP or DP to cross slower boundaries. vLLM and its ecosystem support these layouts, and the serving-specific treatment lives in [serving MoE models at scale](/blog/machine-learning/model-serving/serving-moe-models-at-scale). The general principle from this post — which collective is allowed to cross which wire — is what governs an MoE layout too.

**High-concurrency SLO serving.** The largest production deployments (chat services fielding many thousands of concurrent conversations) lean hard on the data-parallel axis: many replicas behind a load-balancing coordinator, each sized to hit the latency SLO, scaled horizontally on queue depth. The lesson from these systems is the same one this post has repeated — size the *replica* for latency, then multiply with DP for throughput — but at scale it comes with an SLO-aware scheduling layer that preempts and prioritizes across replicas. That is enough of its own discipline to warrant a separate treatment in [high-concurrency SLO management](/blog/machine-learning/model-serving/high-concurrency-slo-management); the connection to this post is that the DP replica is the atomic unit of scaling, and everything above the replica (routing, priority, autoscaling) treats each `TP × PP` engine as one black box that either has queue headroom or does not.

## When to use this (and when not to)

**Use single-node multi-GPU (`--tensor-parallel-size = GPUs/node`, `mp`)** for any model whose weights plus a workable KV budget fit one node. This is most 7B–70B serving. It is the simplest, lowest-latency, most reliable configuration, and you should exhaust it before considering anything else.

**Use Ray multi-node (`TP` intra-node `× PP` across nodes)** only when the model genuinely does not fit one node — 100B+ dense models in full precision, the largest MoE checkpoints. Crossing the boundary buys you the ability to serve the model at all; it costs you network latency on the pipeline handoff, a larger fault domain, and Ray's operational surface. Do not cross it to chase throughput on a model that already fits — that is what data parallelism is for.

**Use native data parallelism (`--data-parallel-size`)** when one replica already meets your latency SLO and you need more aggregate throughput. It scales nearly linearly, needs no Ray, and degrades gracefully (a lost replica costs `1/N`, not the engine). When replicas are small enough to pack several per node, the native launcher in section 7 is the least-overhead way to run many of them.

**Do not go multi-node** if a smaller model, a quantized checkpoint (FP8/AWQ/GPTQ), or a single bigger node would meet the requirement. Every node boundary you avoid is latency you keep and a fault domain you do not have to operate. The order of preference is always: fit one node → quantize to fit one node → data-parallel replicas for throughput → and only then, tensor-plus-pipeline across nodes for a model that truly overflows a single box.

## Key takeaways

- The interconnect hierarchy is the whole game: **TP wants NVLink (intra-node), PP and DP tolerate the inter-node fabric.** Every placement rule follows from this.
- **`TP ≤ GPUs per node`, always.** Stretching a tensor-parallel group across nodes puts a per-layer all-reduce on the slow wire and collapses throughput — the single most common multi-node mistake.
- Choose the layout with two questions: **does it fit one node?** (if not, add PP across nodes) and **does one replica meet QPS?** (if not, add DP replicas). The layout is the product `TP × PP × DP`.
- **`mp` for one node, `ray` for multi-node.** vLLM auto-selects `UniProc`/`mp` on a single host; you opt into `ray` (plus a Ray cluster) to span nodes.
- vLLM's **native data-parallel launcher** (`--data-parallel-size`, `--data-parallel-size-local`, `--data-parallel-start-rank`, `--data-parallel-address`, `--data-parallel-rpc-port`, `--headless`) runs many replicas across nodes without Ray.
- The **NCCL/Gloo interface variables** (`NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, `VLLM_HOST_IP`, `NCCL_IB_HCA`) are mandatory on multi-node; wrong NIC selection is the number-one cause of hangs.
- Container flags **`--gpus all --network host --ipc host --shm-size`** are load-bearing; each maps to a specific failure if omitted.
- On Kubernetes, a multi-node replica is a **`LeaderWorkerSet`**, not a `Deployment` — the pods form one NCCL world and must scale and restart as a group.
- **Verify health, do not assume it:** full Ray GPU count, `/health` 200, **even GPU memory across ranks**, and a real completion. Then benchmark for the p99 knee that sets your capacity.

## Further reading

- [Anatomy of vLLM](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) — the vLLM team's walk through the engine, executor, workers, and the native data-parallel multi-node example whose exact flags this post uses.
- vLLM documentation, "Distributed Inference and Serving" — the canonical reference for `--tensor-parallel-size`, `--pipeline-parallel-size`, the Ray `run_cluster.sh` pattern, and the container flags.
- Ray documentation, "Ray Clusters" — how `ray start --head` / `--address` form the pool vLLM places workers into.
- LeaderWorkerSet (`sigs.k8s.io/lws`) — the Kubernetes API for multi-pod, single-replica workloads like a multi-node inference engine.
- [vLLM distributed architecture anatomy](/blog/machine-learning/model-serving/vllm-distributed-architecture-anatomy) — the companion post on what the executor, workers, and coordinator are and how a request flows through them.
- [Tensor, pipeline, and expert parallelism for serving](/blog/machine-learning/model-serving/tensor-pipeline-expert-parallelism-for-serving) — the first-principles derivation of the all-reduce tax, the pipeline bubble, and the layout math.
- [Debugging vLLM distributed serving](/blog/machine-learning/model-serving/debugging-vllm-distributed-serving) — the runbook for when bring-up hangs: reading NCCL logs, isolating the bad NIC, catching firewall drops.
