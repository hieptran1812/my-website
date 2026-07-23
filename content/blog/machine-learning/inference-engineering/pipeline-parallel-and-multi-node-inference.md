---
title: "Pipeline Parallelism and Multi-Node Inference: When the Model Spans Nodes and the Wire Is Slow"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "Split a model by layer instead of by tensor, hand one small activation across the node boundary, derive the pipeline bubble, and wire a real send/recv pipeline into nanoserve's continuous-batching loop."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "pipeline-parallelism",
    "tensor-parallelism",
    "multi-node",
    "distributed-inference",
    "kv-cache",
    "pytorch",
    "gpu",
    "ml-systems",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 46
---

You put a 70B model on two nodes, wired tensor parallelism across all sixteen GPUs the way the tutorial said, and the thing crawled. Batch 1 latency was worse than a single node running a model half the size. `nvidia-smi` showed the GPUs mostly idle. `nsys` showed why: every transformer layer was blocking on an `all_reduce` that had to cross the Ethernet link between the two nodes, and that link — 25 GbE, maybe 100 GbE — was three orders of magnitude slower than the NVLink fabric the all-reduce assumed. The GPUs weren't computing. They were waiting for the network, sixty-four times per token.

The fix is not a faster all-reduce. It is a different *axis of splitting*. Tensor parallelism cuts every matrix in half and glues the halves back with a collective on every layer; that collective is fine over NVLink at 900 GB/s and catastrophic over a slow cross-node link. Pipeline parallelism cuts the model the other way — layers 0 through 15 on node A, layers 16 through 31 on node B — so the *only* thing that ever crosses the node boundary is a single hidden-state vector handed forward once per stage, roughly `hidden × bytes` per token, and nothing else. For an 8B-class model that is 8 KB per token instead of nearly a megabyte. The figure below is the whole post in one frame: same two nodes, same slow link, two completely different wire bills.

![Two-column comparison showing tensor parallelism paying a blocking all-reduce every layer against pipeline parallelism paying one small activation send per stage boundary](/imgs/blogs/pipeline-parallel-and-multi-node-inference-1.webp)

By the end of this post you will have written `nanoserve/pipeline.py`: a real pipeline-parallel forward pass where each stage owns a contiguous range of layers, activations move between stages over `torch.distributed` point-to-point sends on NCCL, micro-batches fill and drain the pipeline, and the whole thing plugs into the `step()` function from [writing a continuous batching loop](/blog/machine-learning/inference-engineering/writing-a-continuous-batching-loop). You will be able to derive, from the interconnect bandwidth alone, when pipeline parallelism beats tensor parallelism; you will be able to compute the pipeline *bubble* — the idle time baked into any pipeline — and you will understand the nasty inference-specific twist that makes the bubble much worse at decode than it ever is at training. If you have not read [what inference engineering is](/blog/machine-learning/inference-engineering/what-inference-engineering-is), that post frames the scoreboard — TTFT, TPOT, tokens per second, memory, goodput — that this one moves. This post is the sibling of [tensor-parallel inference by hand](/blog/machine-learning/inference-engineering/tensor-parallel-inference-by-hand); read that one for the within-node axis, this one for the across-node axis, and the two together for the layout that actually ships.

## 1. When even tensor parallelism runs out of wire

Start with why you are here at all. A single GPU holds a model when the weights, the activations, and the KV cache all fit in HBM. An RTX 4090 has 24 GB; an A100 or H100 has 80 GB. Llama-3.1-8B in bf16 is about 16 GB of weights, so it fits on one 24 GB card with room for a modest KV cache. A 70B model in bf16 is 140 GB of weights — it does not fit on any single GPU, full stop. You need more than one device, and the question is only *how* you spread the model across them.

There are two clean answers, and they cut the model along perpendicular axes. **Tensor parallelism** (TP) splits every weight matrix. The attention heads are dealt out across ranks, the MLP's hidden dimension is sharded column-wise then row-wise, and each rank computes its slice of every layer. Because each rank produces only a partial result for the layer's output, the ranks must sum their partials before the next layer can start — that is the `all_reduce`, and there are two of them per transformer layer (one after attention, one after the MLP). TP is the right tool inside a node, where the ranks talk over NVLink and an all-reduce of an 8 KB vector is almost free. The full derivation of TP's shapes and collectives is the sibling post, [tensor-parallel inference by hand](/blog/machine-learning/inference-engineering/tensor-parallel-inference-by-hand); here we only need one fact about it — its per-layer synchronization.

**Pipeline parallelism** (PP) splits the model by *depth*. If the model has 32 layers and you have two nodes, you put layers 0–15 on node A and layers 16–31 on node B. A token's hidden state flows through node A's sixteen layers, then the resulting hidden vector is handed to node B, which runs it through the remaining sixteen and produces the logits. The node boundary is crossed exactly once per token, and what crosses it is one hidden-state vector. Nothing is summed across the boundary; nothing is gathered. One send, one receive, forward only.

The reason this matters is bandwidth. Let me put a number on "the wire is slow." NVLink on an H100 SXM node moves about 900 GB/s between GPUs. A fast cross-node fabric — InfiniBand NDR, ConnectX-7 — is around 400 Gb/s, so roughly 50 GB/s, already 18× slower. A commodity data-center Ethernet link is 25–100 Gb/s, so 3–12 GB/s: two orders of magnitude below NVLink. The instant a collective that was designed to run over NVLink is forced to traverse that link on every layer, your GPUs stop computing and start waiting on packets.

The official guidance is blunt about this. vLLM's [Distributed Inference with vLLM](https://blog.vllm.ai/2025/02/17/distributed-inference.html) (2025-02-17) states the rule verbatim: *"Use pipeline parallelism across nodes and tensor parallelism within nodes when interconnects are slow."* That single sentence is the design of every large deployment you will ever run. TP stays inside the NVLink island where its per-layer chatter is cheap; PP spans the slow boundary where its once-per-token handoff is the only affordable thing to send. The rest of this post derives *why* that rule is true, works out what it costs, and builds it.

### The wire bill, derived

Here is the arithmetic that turns "the wire is slow" into a decision. Take our spine model, Llama-3.1-8B: hidden dimension `d_model = 4096`, `L = 32` layers, bf16 so `b = 2` bytes per number. During decode you process one token per sequence per step, so the hidden state for one token is a single vector of 4096 numbers.

$$
\text{activation bytes per token} = d_{\text{model}} \cdot b = 4096 \times 2 = 8192\ \text{bytes} = 8\ \text{KB}
$$

That 8 KB is the fundamental unit of pipeline-parallel communication. It is what one stage hands the next. Now count what tensor parallelism sends across the same boundary. Each transformer layer does two all-reduces, so there are `2L = 64` collectives per token. A ring all-reduce of a message of size `M` across `P` ranks moves, per rank, on the wire:

$$
\text{all-reduce bytes per rank} = \frac{2(P-1)}{P} \cdot M
$$

That factor is the bandwidth-optimal cost of a ring all-reduce (reduce-scatter plus all-gather, each moving `(P-1)/P · M`). Plug in an eight-way tensor-parallel group whose ring is forced to cross the node boundary, `P = 8`, `M = 8` KB:

$$
\frac{2(8-1)}{8} \cdot 8\ \text{KB} = 1.75 \times 8\ \text{KB} = 14\ \text{KB per collective}
$$

Multiply by the 64 collectives per token and you get about 896 KB crossing the fabric for every single token. Pipeline parallelism, splitting the same model across two nodes, crosses the boundary once and sends 8 KB. The ratio is 112×: tensor parallelism asks the slow link to carry more than a hundred times the traffic pipeline parallelism does, and it asks for it in 64 tiny blocking round-trips rather than one forward send. On NVLink that 896 KB is nothing. On a 25 GbE link it is the whole ballgame — it is why your two-node TP deployment crawled.

## 2. The split axis decides the wire bill

Sit with the contrast in a table, because it is the entire justification for the technique. The two axes move wildly different amounts of data across the boundary you can least afford to stress, and they synchronize differently — one blocks the compute 64 times per token, the other never blocks it at all.

![Comparison matrix of tensor parallel versus pipeline parallel across cross-node bytes per token, synchronization operations, and how each splits the key-value cache](/imgs/blogs/pipeline-parallel-and-multi-node-inference-2.webp)

The middle column is the one people miss. Tensor parallelism's all-reduce is a *blocking* collective: rank 0 cannot start the next layer until every rank has contributed its partial sum, so the slowest link in the ring sets the pace of every layer. When that link is a cross-node hop, the entire tensor-parallel group runs at cross-node latency 64 times per token. Pipeline parallelism's send is *non-blocking* in the pipelined sense — while node B computes stage 1 for micro-batch `m`, node A is already computing stage 0 for micro-batch `m+1`. The handoff is one directed edge in a dataflow graph, not a barrier.

The third column previews section 4: the two axes cut the KV cache differently too. Tensor parallelism shards the KV cache *by head* — each rank holds the K and V for its slice of the attention heads, across all layers. Pipeline parallelism holds the KV cache *by layer range* — each stage keeps K and V for only the layers it owns, across all heads. That difference decides where a request's memory lives, and it turns out to complicate eviction in a way we will have to reason about carefully.

When does PP win, then? Purely from the wire bill: PP wins the moment the cross-node link is slow enough that TP's 896 KB of per-token, per-layer collective traffic cannot be hidden behind compute. On NVLink, TP wins — its collectives are cheap and it keeps every token's latency low because all ranks work on the same token simultaneously. Across a slow link, PP wins — it trades a small, honest latency penalty (the hops) for keeping the GPUs busy instead of network-bound. The rule from section 1 is not a preference; it falls directly out of these two numbers.

#### Worked example: a 70B model on 2×8 H100

You have two H100 nodes, eight GPUs each, joined by 400 Gb/s InfiniBand (50 GB/s). Llama-3.1-70B in bf16 is 140 GB of weights, `d_model = 8192`, `L = 80` layers. It does not fit on one GPU (80 GB) and does not fit on one node's worth of aggregate HBM comfortably once you add KV cache, so you need both nodes.

Option A, pure TP=16 across both nodes. Activation per token is `8192 × 2 = 16` KB. All-reduces per token: `2 × 80 = 160`. Ring cost at P=16: `2(15/16) × 16 KB ≈ 30 KB` per collective. Per token across the fabric: `160 × 30 KB ≈ 4.8` MB. At 50 GB/s that is about 96 microseconds of pure cross-node transfer per token, on the critical path, blocking — and that is the ideal-bandwidth figure, ignoring the 160 separate latency hits.

Option B, hybrid TP=8 within each node, PP=2 across nodes. Each node runs an 8-way tensor-parallel group over NVLink; the two nodes form a 2-stage pipeline. Cross-node traffic per token: one activation send, `16` KB. At 50 GB/s that is about 0.3 microseconds of transfer plus one InfiniBand latency hop of a few microseconds. Same hardware, and the cross-node data volume dropped by 300×. Source for both rows: **derived** from the ring all-reduce formula and `d_model`, `L` from the Llama-3.1-70B config. This is the layout section 5 builds toward.

## 3. The bubble, derived: the price pipelines always pay

Pipeline parallelism is not free. It has one structural cost, and it has a name: the **bubble**. Understanding it — and especially how it behaves differently at inference than at training — is what separates a pipeline that hits its throughput target from one that leaves half the cluster idle.

Here is the mechanism. A pipeline of `P` stages is like a `P`-station assembly line. When the line starts empty, station 2 has nothing to do until station 1 finishes the first item and passes it along; station 3 waits even longer. That startup period, when the downstream stages are idle because the work has not reached them yet, is the *fill* bubble. Symmetrically, when the last item leaves station 1, the upstream stations go idle while the item finishes its journey down the line — the *drain* bubble. To keep all stages busy you push many items through at once, staggered, so that at steady state every station is working on a different item. Those items are called **micro-batches**.

![Grid schedule of three pipeline stages over three timesteps showing the lower stages idle during the first two timesteps as the fill bubble before the first token emerges](/imgs/blogs/pipeline-parallel-and-multi-node-inference-3.webp)

Read the grid as a schedule: rows are stages, columns are timesteps, and a cell is what that stage does at that instant. At timestep 1, only stage 0 is busy — it runs micro-batch 1 through layers 0–9. Stages 1 and 2 have nothing yet; those are bubble cells. At timestep 2, stage 0 runs micro-batch 2 while stage 1 runs micro-batch 1; stage 2 still waits. Only at timestep 3 is the whole pipeline busy, and the first token finally emerges from stage 2. Those idle cells in the lower-left triangle are wasted GPU-time, and they are exactly what the bubble formula counts.

### The formula

Let `P` be the number of stages and `M` the number of micro-batches in flight. Doing the accounting in units of "one micro-batch through one stage" (call it a *slot*): the total useful work is `M × P` slots — every micro-batch must pass through every stage. But the wall-clock time to run the pipeline is `M + P − 1` slots wide: it takes `P − 1` slots to fill the pipeline before micro-batch 1 exits, then one slot for each of the `M` micro-batches to stream out. During that wall-clock window there are `P` stages running, so the total *available* slots are `P × (M + P − 1)`. The fraction that is idle — the bubble — is:

$$
B = 1 - \frac{M \cdot P}{P \cdot (M + P - 1)} = \frac{P - 1}{M + P - 1}
$$

That is the whole thing. The bubble shrinks as you add micro-batches (`M` large) and grows as you add stages (`P` large). Some concrete values, all **derived** from the formula:

- `P = 2, M = 8`: `1/9 = 11.1%` idle.
- `P = 4, M = 8`: `3/11 = 27.3%` idle.
- `P = 4, M = 32`: `3/35 = 8.6%` idle.
- `P = 2, M = 32`: `1/33 = 3.0%` idle.
- `P = 4, M = 4`: `3/7 = 42.9%` idle — the pipeline spends more time filling and draining than working.
- `P = 4, M = 1`: `3/4 = 75%` idle — three of four stages are always waiting.

The lesson from those numbers: deep pipelines demand *many* micro-batches to amortize the fill and drain. That is easy in training, where a "micro-batch" is just a slice of your big training batch and you can pick `M` to be as large as you like. In inference it is not easy at all, and that is the twist.

### The inference twist: micro-batches are your live requests

In training you *own* the batch. You have, say, 1024 examples in a global batch, you split them into 32 micro-batches of 32, you feed them through the pipeline, and the bubble is `(P-1)/(M+P-1)` with `M = 32` — small. You choose `M`.

At inference decode, you do not own the batch. Each decode step produces exactly one token per sequence, so the natural unit of work flowing through the pipeline is *one request's one-token forward pass*. The micro-batches that fill the pipeline are the requests in your running set — the live, concurrent requests the [continuous-batching loop](/blog/machine-learning/inference-engineering/writing-a-continuous-batching-loop) is currently serving. So `M` is not a knob you turn; it is set by how many requests happen to be in flight, which is set by your traffic. Serve a burst of 64 concurrent requests and `M = 64`, the bubble is tiny, the pipeline hums. Serve a trickle of 2 requests through a 4-stage pipeline and `M = 2, P = 4`, the bubble is `3/5 = 60%` — most of your expensive multi-node cluster is idle, and the two users see latency no better than a single GPU could give them, because their tokens still have to walk all four stages one hop at a time.

That is the defining property of pipeline parallelism for serving: **it is a throughput technique, not a latency technique, and its efficiency is a direct function of your concurrency.** A single request in flight gets no speedup from PP whatsoever — its token still traverses every stage sequentially. PP earns its keep only when there are enough concurrent requests to keep every stage fed. The animation below is the mechanism in motion: watch the micro-batches descend the stages as a diagonal wavefront, and watch the idle triangles at the start (fill) and end (drain) that never go away when the running set is thin.

<figure class="blog-anim">
<svg viewBox="0 0 740 300" role="img" aria-label="Micro-batches descend a four-stage pipeline as a diagonal wavefront; early on the lower stages sit idle as the fill bubble and late on the upper stages sit idle as the drain bubble" style="width:100%;height:auto;max-width:840px">
<style>
.pp-lane{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.pp-lbl{font:600 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.pp-note{font:600 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.pp-chip rect{fill:var(--accent,#6366f1)}
.pp-chip text{font:700 13px ui-sans-serif,system-ui;fill:#fff;text-anchor:middle}
@keyframes pp-stair{0%,2%{transform:translate(0px,0px);opacity:0}4%{opacity:1}22%{transform:translate(0px,0px);opacity:1}24%{transform:translate(140px,60px)}45%{transform:translate(140px,60px)}47%{transform:translate(280px,120px)}68%{transform:translate(280px,120px)}70%{transform:translate(420px,180px)}90%{transform:translate(420px,180px);opacity:1}100%{transform:translate(550px,180px);opacity:0}}
.pp-c1{animation:pp-stair 12s ease-in-out infinite;animation-delay:0s}
.pp-c2{animation:pp-stair 12s ease-in-out infinite;animation-delay:-2s}
.pp-c3{animation:pp-stair 12s ease-in-out infinite;animation-delay:-4s}
.pp-c4{animation:pp-stair 12s ease-in-out infinite;animation-delay:-6s}
.pp-c5{animation:pp-stair 12s ease-in-out infinite;animation-delay:-8s}
.pp-c6{animation:pp-stair 12s ease-in-out infinite;animation-delay:-10s}
@media (prefers-reduced-motion:reduce){.pp-c1{animation:none;transform:translate(420px,180px)}.pp-c2{animation:none;transform:translate(280px,120px)}.pp-c3{animation:none;transform:translate(140px,60px)}.pp-c4{animation:none;transform:translate(0px,0px)}.pp-c5{animation:none;opacity:0}.pp-c6{animation:none;opacity:0}}
</style>
<text class="pp-lbl" x="14" y="66">stage 0</text>
<text class="pp-lbl" x="14" y="126">stage 1</text>
<text class="pp-lbl" x="14" y="186">stage 2</text>
<text class="pp-lbl" x="14" y="246">stage 3</text>
<rect class="pp-lane" x="140" y="40" width="560" height="44" rx="8"/>
<rect class="pp-lane" x="140" y="100" width="560" height="44" rx="8"/>
<rect class="pp-lane" x="140" y="160" width="560" height="44" rx="8"/>
<rect class="pp-lane" x="140" y="220" width="560" height="44" rx="8"/>
<text class="pp-note" x="196" y="248">fill bubble</text>
<text class="pp-note" x="560" y="66">drain bubble</text>
<g class="pp-chip pp-c1"><rect x="150" y="47" width="62" height="30" rx="7"/><text x="181" y="67">mb1</text></g>
<g class="pp-chip pp-c2"><rect x="150" y="47" width="62" height="30" rx="7"/><text x="181" y="67">mb2</text></g>
<g class="pp-chip pp-c3"><rect x="150" y="47" width="62" height="30" rx="7"/><text x="181" y="67">mb3</text></g>
<g class="pp-chip pp-c4"><rect x="150" y="47" width="62" height="30" rx="7"/><text x="181" y="67">mb4</text></g>
<g class="pp-chip pp-c5"><rect x="150" y="47" width="62" height="30" rx="7"/><text x="181" y="67">mb5</text></g>
<g class="pp-chip pp-c6"><rect x="150" y="47" width="62" height="30" rx="7"/><text x="181" y="67">mb6</text></g>
</svg>
<figcaption>Six micro-batches walk a four-stage pipeline. Each descends one stage per timestep, so at any instant the busy stages form a diagonal. Before the wavefront reaches the bottom the lower stages are idle (the fill bubble); after the last micro-batch passes the top the upper stages are idle (the drain bubble). A small running set never fills the diagonal, so the bubble dominates.</figcaption>
</figure>

There is a second-order subtlety worth naming. Training pipelines schedule fill and drain once per iteration, so the bubble is a fixed overhead you pay at the boundaries of a long forward-backward pass. A decode pipeline runs the fill-and-drain shape *continuously*, because requests arrive and finish at arbitrary times — the running set is churning every step. When a request finishes and the running set drops below `P`, you fall into a bubble immediately; when a burst arrives, you climb out of it. The bubble is not a one-time startup cost at decode; it is a live function of the running-set depth, moment to moment. That is why admission control and the running-set size — the subject of the [scheduler](/blog/machine-learning/inference-engineering/writing-a-continuous-batching-loop) — matter so much once a pipeline is involved.

## 4. Where the KV cache lives when the model is sliced by layer

Now the memory. In [the memory math of the KV cache](/blog/machine-learning/large-language-model/kv-cache) the fundamental formula is that a single token of context costs, across the whole model:

$$
\text{KV bytes per token} = 2 \cdot L \cdot H_{kv} \cdot d_{\text{head}} \cdot b
$$

The 2 is for K and V; `L` is the number of layers; `H_kv` is the number of key/value heads (8 for Llama-3.1-8B's grouped-query attention); `d_head` is 128; `b` is 2 bytes for bf16. Plug it in for the full model:

$$
2 \times 32 \times 8 \times 128 \times 2 = 131072\ \text{bytes} = 128\ \text{KB per token}
$$

So 8k tokens of context for one request costs `128 KB × 8192 ≈ 1` GB of KV cache. That is on a single device. The moment you go pipeline-parallel, that KV cache does not sit in one place. **Each stage holds the KV for only the layers it owns.** Split the 32 layers across two stages, and each stage holds K and V for 16 layers:

$$
2 \times 16 \times 8 \times 128 \times 2 = 65536\ \text{bytes} = 64\ \text{KB per token per stage}
$$

A single request's KV cache is now physically spread across both nodes: 64 KB per token on node A for layers 0–15, 64 KB per token on node B for layers 16–31. The figure makes the split concrete — one model, one request, memory living in two places, and only the 8 KB activation ever crossing between them.

![Layered view of one request whose key-value cache is split across two nodes by layer range, with only an eight-kilobyte activation crossing the link between them](/imgs/blogs/pipeline-parallel-and-multi-node-inference-4.webp)

The good news in this picture: pipeline parallelism *divides* your KV-cache memory pressure across nodes, and it does so cleanly, because layer ranges are disjoint. The full-model 128 KB/token becomes 64 KB/token on each of two nodes; more stages, less per node. Combined with the fact that freeing memory buys throughput super-linearly — vLLM's [distributed inference post](https://blog.vllm.ai/2025/02/17/distributed-inference.html) reports that going from TP=1 to TP=2 yields 13.9× more KV-cache blocks and 3.9× more token throughput, because the extra memory lets you hold far more concurrent sequences — the memory division is a real win, not just a way to make a big model fit.

The bad news is what it does to eviction. In [eviction, preemption and KV swapping](/blog/machine-learning/inference-engineering/eviction-preemption-and-kv-swapping) the whole mechanism assumes a request's KV cache is a set of blocks you can free on one device to relieve pressure on that device. Under pipeline parallelism a request's blocks live on *every* stage. If node B is under memory pressure and wants to preempt a request, freeing that request's node-B blocks alone is incoherent — the request still holds node-A blocks for the same tokens, and you cannot resume it later unless *both* halves are recoverable. So preemption becomes a pipeline-wide decision: to evict request `r`, every stage must drop `r`'s blocks together, and to swap it out you must move (or recompute) the KV for all layers, not just the ones on the crowded node. The block table you built earlier now has a per-stage dimension, and the scheduler's free-block accounting has to be the *minimum* free capacity across stages, because a request cannot be admitted unless every stage can house its share.

This is exactly the problem that dedicated cross-node KV systems exist to solve. vLLM's [Mooncake Store](https://blog.vllm.ai/2026/05/06/mooncake-store.html) (2026-05-06) pools KV cache across a cluster with GPUDirect RDMA between HBM and CPU memory and a master server that tracks block hashes and client health, so KV can be transferred and reused across engines rather than pinned to one device's fate; the [MoRIIO KV connector](https://blog.vllm.ai/2026/04/07/moriio-kv-connector.html) (2026-04-07) pushes KV across an RDMA link between prefill and decode roles. Both are cited here as the production-grade version of the problem this section raises — you will not build them into nanoserve, but you should know the cross-node KV transfer path is the seam where multi-node serving gets genuinely hard.

#### Worked example: per-node memory budget for PP=4

Serve Llama-3.1-70B (`L = 80`, `H_kv = 8`, `d_head = 128`, bf16) on four pipeline stages, 20 layers each. Full-model KV per token: `2 × 80 × 8 × 128 × 2 = 320` KB. Per stage: `2 × 20 × 8 × 128 × 2 = 80` KB per token per node. Weights per stage: 140 GB / 4 = 35 GB. On an 80 GB H100 stage that leaves ~45 GB for KV after weights (before activation scratch and fragmentation). At 80 KB/token/stage, 45 GB houses about `45 × 10^9 / (80 × 10^3) ≈ 560,000` token-slots on that stage — say 68 concurrent requests at 8k context each. Because every stage holds the same *number* of tokens (the layer count differs but here it is even), the pipeline's concurrency ceiling is that 68, and it is set by the tightest stage. Source: **derived** from the KV formula and the Llama-3.1-70B config; treat the 45 GB as an order-of-magnitude budget, since real activation scratch and allocator overhead eat into it.

## 5. The hybrid layout: tensor parallel inside, pipeline parallel across

Everything so far points at one layout, and it is the layout large deployments actually run: **tensor parallelism within a node, pipeline parallelism across nodes.** Two H100 nodes of eight GPUs each become an 8-way tensor-parallel group per node (all-reduces over NVLink, cheap) stitched into a 2-stage pipeline (one activation hop over the slow link, cheap). TP=8 × PP=2 = 16 GPUs, and each GPU holds `1/16` of the model. The figure traces one token's path through the hybrid.

![Dataflow diagram of a hidden state fanning out to two tensor-parallel ranks, merging through an all-reduce over fast NVLink, then a single activation crossing the slow link to the next pipeline stage](/imgs/blogs/pipeline-parallel-and-multi-node-inference-5.webp)

Follow the token. Its hidden vector fans out to the tensor-parallel ranks within the node — each rank owns half the heads (in a 2-way sketch; eight in practice). The ranks compute their slices and merge back through an all-reduce over NVLink at 900 GB/s, which produces the complete hidden state for that stage's layers. That merged activation — and *only* that merged activation, 8 KB — is what crosses the InfiniBand link to node B, which runs its own tensor-parallel group over its layers and finally emits the logits. The slow link sees one small vector per token; the fast link inside each node absorbs all the per-layer collective chatter where bandwidth is plentiful.

This is why the axis assignment is not arbitrary. You put the *chatty* parallelism (TP, with its 64 collectives per token) where bandwidth is cheap, and the *quiet* parallelism (PP, with its one send per token) where bandwidth is scarce. Invert it — TP across nodes, PP within a node — and you get the disaster from the intro: cheap NVLink carrying a single 8 KB hop it could do a thousand of, while the precious slow link chokes on 896 KB of collectives. The layout is a direct consequence of matching each axis's communication pattern to the bandwidth available on each tier of the fabric.

The same logic generalizes. Expert parallelism for MoE models is a third axis, and at very large scale you combine all three; vLLM's [Expert Parallelism at Scale](https://blog.vllm.ai/2025/12/17/large-scale-serving.html) (2025-12-17) reports DeepSeek-V3 hitting 2.2k tok/s per H200 on a CoreWeave cluster by combining wide expert parallelism with data parallelism, with the all-to-all traffic carried by DeepEP kernels — the point being that every axis gets matched to the fabric tier that can afford its traffic pattern. Expert parallelism, which routes each token to the ranks holding the experts it activates, is a topic of its own; here, keep the two-axis mental model: chatty inside, quiet across.

## 6. Building it in nanoserve

Time to write `nanoserve/pipeline.py`. We build three things: a `PipelineStage` that owns a contiguous layer range, the point-to-point send/recv that moves activations between stages over NCCL, and a scheduled pipelined `step()` that fills and drains micro-batches and plugs into the continuous-batching engine. Everything here is real PyTorch using `torch.distributed`; none of it is pseudocode.

### Splitting the model into stages

A pipeline stage is just a slice of the layer stack plus, on the ends, the embedding and the LM head. The rank owns which layers by its position in the pipeline. We compute a contiguous, balanced partition.

```python
# nanoserve/pipeline.py
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn


def partition_layers(num_layers: int, pp_size: int) -> list[tuple[int, int]]:
    """Contiguous, balanced [start, end) layer range for each of pp_size stages.

    32 layers over 4 stages -> [(0,8), (8,16), (16,24), (24,32)].
    Remainder goes to the earliest stages so no stage is more than one layer
    heavier than any other -- balance matters because the slowest stage sets
    the whole pipeline's token rate (section 9).
    """
    base, extra = divmod(num_layers, pp_size)
    ranges, start = [], 0
    for s in range(pp_size):
        n = base + (1 if s < extra else 0)
        ranges.append((start, start + n))
        start += n
    return ranges


class PipelineStage(nn.Module):
    """One stage: a contiguous slice of transformer layers, plus the
    embedding on stage 0 and the final norm + lm_head on the last stage."""

    def __init__(self, model_config, weights, pp_rank: int, pp_size: int):
        super().__init__()
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.is_first = pp_rank == 0
        self.is_last = pp_rank == pp_size - 1

        lo, hi = partition_layers(model_config.num_hidden_layers, pp_size)[pp_rank]
        self.layer_lo, self.layer_hi = lo, hi
        # Build ONLY this stage's layers -- the other stages' weights never
        # touch this GPU's memory. That is the whole point.
        self.layers = nn.ModuleList(
            build_decoder_layer(model_config, weights, i) for i in range(lo, hi)
        )
        self.embed = build_embedding(model_config, weights) if self.is_first else None
        self.norm = build_final_norm(model_config, weights) if self.is_last else None
        self.lm_head = build_lm_head(model_config, weights) if self.is_last else None
        self.hidden_size = model_config.hidden_size

    def forward(self, x, kv, batch):
        # x: on stage 0 these are token ids [T]; elsewhere a hidden state [T, H].
        h = self.embed(x) if self.is_first else x
        for local_idx, layer in enumerate(self.layers):
            global_idx = self.layer_lo + local_idx
            h = layer(h, kv.view(global_idx), batch)  # KV for THIS layer only
        if self.is_last:
            h = self.norm(h)
            return self.lm_head(h)     # logits [T, vocab]
        return h                       # hidden state [T, H] to hand forward
```

The key line is that each stage constructs only its own layers. The 70B model never fully materializes on any GPU; stage `r` loads `weights[layer]` for `layer in [lo, hi)` and nothing else. The `kv.view(global_idx)` call reaches into a per-layer KV store indexed by the *global* layer number, which is how a stage addresses its slice of the request's spread-out cache from section 4.

### Handing activations across the boundary

Between stages, one tensor moves: the hidden state, shape `[T, H]` where `T` is the number of tokens in this step's flattened batch and `H` is the hidden size. NCCL supports point-to-point `send`/`recv` between any two ranks, which is precisely the primitive we want — no collective, no all-reduce, just a directed edge.

```python
# nanoserve/pipeline.py  (continued)

def send_activation(h: torch.Tensor, dst_rank: int, group=None):
    """Blocking P2P send of a hidden state to the next stage.

    NCCL requires the receiver to post a matching recv with the SAME shape and
    dtype, so we send the token count first as a tiny header, then the tensor.
    """
    header = torch.tensor([h.shape[0]], device=h.device, dtype=torch.long)
    dist.send(header, dst=dst_rank, group=group)
    dist.send(h.contiguous(), dst=dst_rank, group=group)


def recv_activation(src_rank: int, hidden_size: int, device, dtype, group=None):
    """Blocking P2P recv: read the header, allocate, receive the hidden state."""
    header = torch.empty(1, device=device, dtype=torch.long)
    dist.recv(header, src=src_rank, group=group)
    num_tokens = int(header.item())
    h = torch.empty((num_tokens, hidden_size), device=device, dtype=dtype)
    dist.recv(h, src=src_rank, group=group)
    return h
```

Two things earn their place here. First, the *header*: NCCL's `recv` needs to know the tensor's shape before it can allocate the receive buffer, and at decode the number of tokens `T` changes every step as the running set churns. Sending a one-element header first — a few bytes — lets the receiver size its buffer correctly. Second, `.contiguous()`: the hidden state coming out of the last layer may be a non-contiguous view, and NCCL sends raw bytes, so you must guarantee a contiguous layout or you will silently send garbage. That is the kind of bug that produces plausible-looking-but-wrong tokens and costs an afternoon.

For real throughput you replace the blocking pair with `dist.batch_isend_irecv`, which posts the send for micro-batch `m` and the receive for micro-batch `m+1` together and lets NCCL overlap them with compute. The blocking version above is the one to understand first; the overlapped version is the one to ship.

### The naive forward shows you the bubble

Before scheduling micro-batches, run the pipeline the dumb way — one micro-batch at a time, each stage waiting for the previous — and you can *see* the bubble as wall-clock idle. This is the version you write first to convince yourself the plumbing works, then throw away.

```python
# nanoserve/pipeline.py  (continued)

@torch.inference_mode()
def naive_pipeline_forward(stage: PipelineStage, batch, kv):
    """Run ONE micro-batch straight through the pipeline. Every stage but one
    is idle at every instant -- this is a P-stage pipeline running at 1/P
    utilization. Correct, and exactly the thing micro-batching fixes."""
    world = dist.get_world_size()
    prev, nxt = stage.pp_rank - 1, stage.pp_rank + 1

    if stage.is_first:
        x = batch.input_ids                       # [T]
    else:
        x = recv_activation(prev, stage.hidden_size,
                            batch.device, torch.bfloat16)

    out = stage(x, kv, batch)                      # this stage's compute

    if not stage.is_last:
        send_activation(out, nxt)                  # hand forward, then go idle
        return None
    return out                                     # logits, only on last stage
```

Time this with CUDA events and a `dist.barrier()` before you start the clock (section 8), and on a 4-stage pipeline you will measure roughly `4×` the single-stage compute time for one micro-batch's logits, with each GPU busy only `1/4` of the wall clock. That is the 75% bubble from the formula at `M = 1, P = 4`. The whole job of the scheduler is to overlap micro-batches so those idle slots get filled.

### Scheduling micro-batches: fill, steady state, drain

Now the real loop. At decode, the micro-batches are the requests in the running set. We split the running set into groups (one group per "in-flight slot" of the pipeline) and stagger them so that while stage 1 works on group `A`, stage 0 is already working on group `B`. Each stage runs a small state machine: receive from upstream (unless first), compute, send downstream (unless last).

```python
# nanoserve/pipeline.py  (continued)

@torch.inference_mode()
def pipelined_step(stage: PipelineStage, microbatches: list, kv):
    """One decode step across the whole pipeline, overlapping micro-batches.

    microbatches: list of Batch objects, one per in-flight group of requests.
    Every stage executes the SAME number of slots; the diagonal wavefront
    (see the animation in section 3) falls out of who sends/recvs when.
    Returns logits per micro-batch on the last stage, else None.
    """
    world = dist.get_world_size()
    prev, nxt = stage.pp_rank - 1, stage.pp_rank + 1
    M = len(microbatches)
    P = stage.pp_size

    outputs = [None] * M
    # A stage that owns pipeline position r starts working on micro-batch m
    # at slot (m + r): that offset IS the fill bubble made explicit.
    total_slots = M + P - 1
    inbox = {}  # micro-batch index -> received hidden state

    for slot in range(total_slots):
        m = slot - stage.pp_rank        # which micro-batch this stage handles now
        if m < 0 or m >= M:
            continue                    # bubble slot: this stage idles

        mb = microbatches[m]
        if stage.is_first:
            x = mb.input_ids
        else:
            x = inbox.pop(m)            # arrived from upstream on an earlier slot

        out = stage(x, kv, mb)

        if stage.is_last:
            outputs[m] = out            # logits for this micro-batch
        else:
            send_activation(out, nxt)   # next stage will recv it at slot+1

        # Receive the NEXT micro-batch's input from upstream, if any is coming.
        if not stage.is_first:
            nm = (slot + 1) - stage.pp_rank
            if 0 <= nm < M:
                inbox[nm] = recv_activation(prev, stage.hidden_size,
                                            mb.device, torch.bfloat16)
    return outputs if stage.is_last else None
```

The `m = slot - stage.pp_rank` line is the entire schedule in one expression. Stage 0 starts micro-batch 0 at slot 0; stage 1 starts it at slot 1; stage `r` starts it at slot `r`. That per-stage offset *is* the diagonal wavefront from the animation, and the slots where `m` falls outside `[0, M)` are exactly the fill and drain bubble cells. `total_slots = M + P − 1` is the wall-clock width the bubble formula predicted. The code and the formula are the same object.

### Wiring it into the continuous-batching step()

The [continuous-batching engine](/blog/machine-learning/inference-engineering/writing-a-continuous-batching-loop) had a `step()` that called `model.forward(batch)` once. Under pipeline parallelism, "the model" is now a distributed object, and one forward is a pipelined pass across stages. The engine change is surgical: group the running set into micro-batches, call `pipelined_step`, and let only the last stage sample and stream. The other stages produce no user-visible output — they exist to compute hidden states — so they run the same loop but skip sampling.

```python
# nanoserve/engine.py  (pipeline-parallel variant of step())

@torch.inference_mode()
def step(self) -> list[tuple[str, int, str | None]]:
    plan = self.sched.schedule()                      # admit + plan (unchanged)
    if not plan.num_tokens:
        self.step_idx += 1
        return []

    scheduled = [r for r in self.sched.running if r.req_id in plan.num_tokens]
    # Chunk the running set into P-ish micro-batches so the pipeline stays fed.
    groups = chunk_into_microbatches(scheduled, self.n_microbatches)
    batches = [build_batch(g, plan.num_tokens, self.kv, device=self.stage.device)
               for g in groups]

    logits_per_mb = pipelined_step(self.stage, batches, self.kv)

    events = []
    if self.stage.is_last:                            # only the tail samples
        for g, logits in zip(groups, logits_per_mb):
            next_ids = self.model.sample(logits, g)
            events += self._retire_or_continue(g, next_ids)   # from Track C
        # Sampled tokens must reach stage 0, which owns request state + streaming.
        broadcast_sampled_tokens(events, src=self.stage.pp_size - 1)
    else:
        events = recv_sampled_tokens(src=self.stage.pp_size - 1)
        self._apply_tokens(events)                    # advance local request state

    self.step_idx += 1
    return events
```

One subtlety the last block handles: the tokens are *sampled* on the final stage, but the request bookkeeping — appending to `output_ids`, checking stop conditions, streaming to the client — conventionally lives with stage 0, which owns the front of the pipeline and talks to the scheduler. So after sampling, the last stage broadcasts the sampled token ids back so every stage advances the same request state in lockstep. Skip that and your stages disagree about how long each sequence is, which desynchronizes the KV cache lengths and corrupts attention on the very next step. Distributed inference is full of these "everyone must agree" invariants; naming them in the code is how you keep them.

### Launching across nodes

Nothing runs until you start one process per GPU across both nodes with a shared rendezvous. `torchrun` does this directly for a static cluster:

```bash
# On node A (the rendezvous host), 8 GPUs:
torchrun --nnodes=2 --nproc-per-node=8 --node-rank=0 \
         --rdzv-backend=c10d --rdzv-endpoint=node-a:29500 \
         serve.py --pp-size 2 --tp-size 8 --model llama-3.1-70b

# On node B, same command with --node-rank=1:
torchrun --nnodes=2 --nproc-per-node=8 --node-rank=1 \
         --rdzv-backend=c10d --rdzv-endpoint=node-a:29500 \
         serve.py --pp-size 2 --tp-size 8 --model llama-3.1-70b
```

Inside `serve.py`, each process computes its `pp_rank` and `tp_rank` from the global rank (`pp_rank = global_rank // tp_size`, `tp_rank = global_rank % tp_size`), builds its `PipelineStage` with its tensor-parallel sub-group, and initializes NCCL with `dist.init_process_group(backend="nccl")`. The InfiniBand transport is picked up automatically by NCCL when the fabric is present; set `NCCL_IB_HCA` and `NCCL_SOCKET_IFNAME` if it guesses the wrong interface, which on multi-NIC nodes it often does. For dynamic or elastic clusters you drive the same processes with Ray instead — `ray.init()`, a placement group pinning one GPU per worker, and Ray's actors standing in for the `torchrun` ranks — which is the path vLLM takes for multi-node, and the same path its elastic expert-parallelism work builds on. Ray buys you fault tolerance and rescheduling; `torchrun` is simpler for a fixed two-node box. Either way the `PipelineStage` and `pipelined_step` code above does not change.

## 7. The network is the new wall

On a single GPU your bottleneck is HBM bandwidth — decode is memory-bound, and time per token is roughly weight-bytes over HBM-bandwidth. Go multi-node and a second wall appears behind the first: the *network*. Every stage boundary a token crosses adds a hop, and the hop's cost is set by the interconnect. The timeline traces one token's full journey through a four-stage pipeline and makes the hops visible as segments on the critical path.

![Timeline of a single token traversing four pipeline stages with a network hop between each, showing the token pays every stage's decode time plus every hop's latency](/imgs/blogs/pipeline-parallel-and-multi-node-inference-6.webp)

Read the timeline as one token's latency budget. It pays every stage's decode time — four stages of ~2 ms each here — *plus* every hop it crosses. And the hops are not equal. An InfiniBand hop is a few microseconds; a commodity Ethernet hop can be 30–50 microseconds, ten times more, before you even count congestion. For a latency-sensitive single stream those hops are pure tax: pipeline parallelism does not shorten the path any one token travels, it lengthens it by `P − 1` hops. What PP buys is *capacity* — the ability to have many tokens in flight across the stages at once — not a faster single token.

This is the throughput-versus-latency fork, and it decides your `M`. A **throughput-oriented** deployment (batch inference, offline scoring, high-concurrency chat) wants `M` large: pack the running set deep, fill the pipeline, drive the bubble toward zero, and accept that any individual token's latency includes the hops. A **latency-oriented** deployment (an interactive endpoint with strict TTFT and TPOT budgets, low concurrency) is exactly where pipeline parallelism hurts most: `M` is small because concurrency is low, the bubble is large, and every token eats the hop penalty. If your SLO is a tight time-per-output-token and your traffic is thin, adding pipeline stages can make latency *worse* while adding throughput you are not using. That is not a bug in PP; it is PP being a throughput technique used against a latency problem.

The interconnect choice follows directly. InfiniBand's few-microsecond hop is negligible against a 2 ms stage, so on IB the hops essentially disappear and only the bubble matters. On Ethernet the hop is a meaningful fraction of the stage time, so on slow Ethernet you want *fewer* stage boundaries (shallower pipelines, `P` small) to minimize hops, which fights against the deeper-pipeline instinct you would have on IB. The fabric is not a detail; it changes the optimal `P`.

## 8. The numbers, with provenance

Every number below is derived from the formulas in this post or cited to a named source. None of it is a first-hand benchmark — I have no cluster and have run nothing. The point of the table is that you can reproduce every derived row with a calculator, and reproduce the cited rows by reading the linked source.

### Bubble fraction as a function of stages and micro-batches

| Stages `P` | Micro-batches `M` | Bubble `(P-1)/(M+P-1)` | Reading | Source |
|---|---|---|---|---|
| 2 | 32 | 3.0% | deep concurrency, shallow pipe: ideal | derived |
| 2 | 8 | 11.1% | modest concurrency, 2 stages: fine | derived |
| 4 | 32 | 8.6% | deep concurrency hides a 4-stage pipe | derived |
| 4 | 8 | 27.3% | thin concurrency starts to bite | derived |
| 4 | 4 | 42.9% | running set = stages: half idle | derived |
| 4 | 1 | 75.0% | one request through 4 stages: catastrophic | derived |
| 8 | 8 | 47.0% | deep pipe, matched running set: still bad | derived |

### Cross-node bytes per token, TP versus PP

| Layout | Cross-node bytes/token | Sync ops/token | Blocking? | Source |
|---|---|---|---|---|
| TP=8 across nodes, 8B model | ~896 KB | 64 all-reduces | yes, every layer | derived |
| PP=2 across nodes, 8B model | 8 KB | 1 P2P send | no, forward only | derived |
| TP=16 across nodes, 70B model | ~4.8 MB | 160 all-reduces | yes, every layer | derived |
| PP=2 across nodes, 70B model | 16 KB | 1 P2P send | no, forward only | derived |

The TP rows use the ring all-reduce cost `2(P-1)/P · M` with `M = d_model × 2` bytes and `2L` collectives per token; the PP rows are one `d_model × 2`-byte send. The 300× gap on the 70B row is the whole argument for the hybrid layout.

### A cited anchor for the memory-buys-throughput effect

| Effect | Reported result | Source |
|---|---|---|
| TP=1 → TP=2 frees KV memory | 13.9× more KV-cache blocks, 3.9× more token throughput | cited: [vLLM Distributed Inference](https://blog.vllm.ai/2025/02/17/distributed-inference.html) |
| Wide-EP DeepSeek-V3 on H200 | 2.2k tok/s per GPU (CoreWeave, IB ConnectX-7) | cited: [vLLM Expert Parallelism at Scale](https://blog.vllm.ai/2025/12/17/large-scale-serving.html) |

### How to measure a pipeline honestly

If you build this and want real numbers, the honest measurement discipline for a distributed decode loop is stricter than for a single GPU:

- **Barrier before you start the clock.** Call `dist.barrier()` so every rank begins timing together; otherwise a late-starting rank makes stage 0 look slow when it is really waiting.
- **CUDA events, not wall-clock `time.time()`.** Record `torch.cuda.Event(enable_timing=True)` on each stage's stream and `torch.cuda.synchronize()` before reading, because kernel launches are asynchronous and `time.time()` around a launch measures the launch, not the work.
- **Warm up, then steady state.** The first few steps pay CUDA graph capture, NCCL channel setup, and allocator warm-up. Throw them away and measure a steady running set, not the fill transient.
- **Measure the bubble directly.** Instrument each stage's idle slots (the `continue` branch in `pipelined_step`) and divide by total slots — that empirical bubble should match `(P-1)/(M+P-1)` for your running-set depth. If it does not, your micro-batch chunking is wrong.
- **Open-loop load, not closed-loop.** Drive the server with Poisson arrivals at a fixed rate and measure TTFT/TPOT/goodput under that arrival process. A closed-loop harness that fires the next request only after the last finishes hides exactly the queueing behavior that pipeline bubbles create.
- **tok/s at batch 1 tells you nothing.** A single request through a pipeline runs at single-stream latency plus hops — worse than one GPU. The pipeline's number is throughput at your real concurrency, and it only looks good once `M` is large.

## 9. Stress tests and failure modes

A pipeline that works in the demo breaks in production in four specific ways. Reason through each before it happens to you.

**The running set drops below the stage count.** Traffic thins out at 3 a.m., the running set falls to 2 requests, and your 4-stage pipeline is now running at `M = 2, P = 4` — a 60% bubble. Two-thirds of your expensive multi-node cluster is idle, and the two live users see latency no better than a single GPU would give, plus the hops. There is no clever scheduling fix; the bubble is structural. The real fixes are operational: scale the pipeline *in* under low load (collapse to fewer stages or a single node), or accept the idle cost as the price of keeping a big model resident. This is why elastic layouts matter — the layout that is optimal at peak concurrency is wasteful at trough, and the running-set depth, not a static config, should drive it.

**A slow stage stalls the whole pipeline.** The pipeline's token rate is set by its *slowest* stage — it is a synchronous assembly line, and no station can outrun the one in front of it. If one stage got an uneven layer partition (19 layers while others got 15), or one node has a slightly slower GPU, or one stage's node has a busier NIC, that stage becomes the pacer and every other stage waits on it. This is why `partition_layers` balances to within one layer, and why in production you monitor per-stage step time and alert on divergence. A stage that is 20% slower makes the *entire* pipeline 20% slower, no matter how fast the others are — the cost is not amortized, it is imposed.

**A long generation flushed across many steps.** A request generating 4,000 tokens sits in the running set for 4,000 decode steps, and across those steps the running set churns as short requests arrive and finish. Every time the set dips below `P`, that long request is stuck in a bubble; every time a burst arrives, it climbs out. Its per-token latency is therefore *not* constant — it wobbles with the ambient concurrency. If you promise a flat TPOT SLO on a pipeline under variable load, you will miss it during the quiet stretches. The honest SLO on a pipeline is conditional on concurrency, and admission control (holding some minimum running-set depth, or shedding to keep the pipeline fed) is what makes it hold.

**Topology mistakes: TP and PP crossed wrong.** The single most expensive misconfiguration in multi-node serving is putting tensor parallelism across the node boundary and pipeline parallelism inside the node — the exact inversion of the rule. It "works" in that it produces correct tokens, so it passes your smoke test, and then it runs at a fraction of the throughput because the slow link is carrying 896 KB of collectives per token instead of an 8 KB send. Always verify the mapping: tensor-parallel ranks must share a node (check that each TP group's ranks report the same hostname), pipeline stages must be the thing that spans nodes. A one-line assertion at startup — every tensor-parallel group is co-located on one host — catches the disaster before it costs you a week of "why is our cluster slow."

## When to reach for this (and when not to)

Pipeline parallelism is a specific tool for a specific constraint, and using it outside that constraint is how you make things worse. The matrix summarizes the decision.

![Decision matrix mapping interconnect tier to the right parallel axis, from tensor parallel within a node to hybrid across nodes to a caution on small running sets](/imgs/blogs/pipeline-parallel-and-multi-node-inference-7.webp)

**Reach for pipeline parallelism when:** the model does not fit within a single node's worth of GPUs, *and* the link between nodes is slow enough that cross-node tensor parallelism would be network-bound, *and* you have enough concurrency to keep the pipeline fed. That is the intersection where PP is not just useful but the only sane choice — a 70B-plus model on Ethernet-joined nodes serving a real request load.

**Do not reach for it when:** the model fits in one node — then use pure tensor parallelism over NVLink and skip the bubble and the hops entirely. Or when your traffic is thin and latency-sensitive — a low-concurrency interactive endpoint gets the bubble and the hop tax with none of the throughput payoff, and you are better off with a smaller model, quantization ([weight-only quantization in your engine](/blog/machine-learning/inference-engineering/weight-only-quantization-in-your-engine-gguf-awq-gptq-at-load-time)), or a single node. Or when the interconnect is fast — on an NVLink-connected multi-GPU box, or InfiniBand-joined nodes where the hop is negligible, tensor parallelism's per-layer collectives are cheap and it keeps single-stream latency low, which PP cannot.

And the honest meta-recommendation: **you should almost certainly not write your own multi-node pipeline for production.** vLLM, SGLang, and TensorRT-LLM implement TP×PP with overlapped communication, elastic scaling, fault tolerance, and cross-node KV transfer that took teams years to harden. Build the nanoserve version to *understand* the mechanism — the bubble, the hybrid layout, the spread KV cache, the send/recv plumbing — so that when vLLM's `--pipeline-parallel-size` and `--tensor-parallel-size` flags misbehave, you know exactly what they are doing and why your cluster is slow. Understanding is the deliverable; the production engine is theirs. For the production picture at this scale, see [multi-node LLM serving for 100B-plus models](/blog/machine-learning/model-serving/multi-node-llm-serving-100b-plus), which covers the operational side this post's mechanism underpins.

## Case studies: real numbers from public sources

**vLLM's distributed inference guidance.** The [Distributed Inference with vLLM](https://blog.vllm.ai/2025/02/17/distributed-inference.html) post (2025-02-17) is the primary source for the rule this whole post derives — pipeline parallelism across nodes, tensor parallelism within nodes when interconnects are slow — and for the memory-buys-throughput anchor: TP=1 to TP=2 gives 13.9× more KV-cache blocks and 3.9× more token throughput, a super-linear effect that comes purely from the freed memory letting the engine hold more concurrent sequences. That super-linearity is why the KV-cache division in section 4 is a genuine win and not just a way to fit the weights.

**Expert parallelism at DeepSeek scale.** vLLM's [Expert Parallelism at Scale](https://blog.vllm.ai/2025/12/17/large-scale-serving.html) (2025-12-17) reports DeepSeek-V3 (37B of 671B active per forward) reaching 2.2k tok/s per H200 GPU on a CoreWeave cluster with InfiniBand ConnectX-7, using wide expert parallelism combined with data parallelism, DeepEP all-to-all kernels, and dual-batch overlap of compute and communication. The relevant lesson for this post: at scale you stack a *third* axis (expert parallelism) on top of TP and PP, and again the rule is to match each axis's communication pattern to the fabric tier that can afford it.

**Cross-node KV as its own system.** The [Mooncake Store](https://blog.vllm.ai/2026/05/06/mooncake-store.html) (2026-05-06) post reports pooling KV cache across a cluster with GPUDirect RDMA, hitting a 92.2% cache hit rate versus 1.7% on agentic Codex traces and 3.8× throughput on a Kimi-2.5 deployment — evidence that once KV cache is spread across nodes (section 4), moving and reusing it across the cluster becomes a first-class system, not an afterthought. The [MoRIIO KV connector](https://blog.vllm.ai/2026/04/07/moriio-kv-connector.html) (2026-04-07) reports 2.5× higher goodput than a collocated baseline by pushing KV over RDMA between prefill and decode roles. Both are the production form of the eviction problem this post raises.

## Key takeaways

- **Split by layer, not by tensor, when the wire is slow.** Tensor parallelism's per-layer all-reduce is fine over NVLink and murderous over a slow cross-node link; pipeline parallelism sends one hidden-state vector per stage boundary, roughly `d_model × bytes` per token. For an 8B model that is 8 KB versus ~896 KB — a 112× difference on the link you can least afford to stress.
- **The rule is not a preference, it is arithmetic.** Pipeline parallelism across nodes, tensor parallelism within nodes, when interconnects are slow. It falls directly out of matching each axis's communication volume to each fabric tier's bandwidth.
- **Every pipeline has a bubble:** `(P-1)/(M+P-1)` idle. It shrinks with micro-batches and grows with stages.
- **At inference, micro-batches are your live requests.** You do not choose `M`; your concurrency does. A thin running set through a deep pipeline is mostly bubble, and there is no scheduling trick that removes it.
- **Pipeline parallelism is a throughput technique, not a latency one.** A single request gets no speedup — its token still walks every stage sequentially and pays every hop. PP buys capacity for many concurrent streams.
- **The KV cache spreads across stages, one layer range per node.** This divides memory pressure cleanly but makes preemption a pipeline-wide decision, because a request's blocks live on every stage.
- **The slowest stage sets the token rate.** Balance layer partitions to within one layer, and monitor per-stage step time — a 20% slow stage makes the whole pipeline 20% slower.
- **Do not cross the axes.** TP across nodes and PP within a node produces correct tokens and terrible throughput; assert co-location of every tensor-parallel group at startup.
- **Build it to understand it; run vLLM to serve it.** The nanoserve pipeline teaches the mechanism so you can debug the real one.

## Further reading

- [What inference engineering is](/blog/machine-learning/inference-engineering/what-inference-engineering-is) — the scoreboard (TTFT, TPOT, tok/s, memory, goodput) this post moves.
- [Tensor-parallel inference by hand](/blog/machine-learning/inference-engineering/tensor-parallel-inference-by-hand) — the within-node axis: sharding matrices and the per-layer all-reduce this post routes around.
- [Writing a continuous batching loop](/blog/machine-learning/inference-engineering/writing-a-continuous-batching-loop) — the `step()` and running set that the pipelined step plugs into.
- [Eviction, preemption and KV swapping](/blog/machine-learning/inference-engineering/eviction-preemption-and-kv-swapping) — why a KV cache spread across stages complicates preemption.
- [The inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook) — the capstone that ties parallelism back to the full serving stack.
- [Multi-node LLM serving for 100B-plus models](/blog/machine-learning/model-serving/multi-node-llm-serving-100b-plus) — the production and operational picture behind this mechanism.
- [Distributed Inference with vLLM](https://blog.vllm.ai/2025/02/17/distributed-inference.html) — the primary source for the across-nodes/within-nodes rule and the 13.9×/3.9× memory-throughput anchor.
- [Expert Parallelism at Scale](https://blog.vllm.ai/2025/12/17/large-scale-serving.html) and [Mooncake Store](https://blog.vllm.ai/2026/05/06/mooncake-store.html) — the third parallel axis and cross-node KV as production systems.
