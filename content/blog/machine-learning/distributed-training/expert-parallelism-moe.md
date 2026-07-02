---
title: "Expert Parallelism: Training Mixture-of-Experts and the All-to-All Bottleneck"
date: "2026-07-02"
publishDate: "2026-07-02"
description: "Split the experts across GPUs, derive the all-to-all byte law that decides whether a Mixture-of-Experts scales or stalls, and learn to fix load imbalance and dropped tokens before they wreck your throughput."
tags:
  [
    "distributed-training",
    "multi-gpu",
    "mixture-of-experts",
    "expert-parallelism",
    "all-to-all",
    "pytorch",
    "deepspeed",
    "megatron",
    "deep-learning",
    "ml-systems",
    "gpu",
  ]
category: "machine-learning"
subcategory: "Distributed Training"
author: "Hiep Tran"
featured: true
readTime: 37
---

You have a 47-billion-parameter model that only ever runs 13 billion parameters per token, and you cannot figure out why eight H100s are barely beating four. The forward pass looks cheap on paper — a Mixtral-style block does the FLOPs of a 13B dense model, not a 47B one. But when you profile a single training step across the eight ranks, a third of the wall-clock is a single collective you never call in a dense model, a collective that lights up every GPU-to-GPU link at once and then sits there, waiting, while one overloaded GPU finishes a pile of tokens the router dumped on it. That collective is the **all-to-all**, and it is the price of admission for training a Mixture-of-Experts (MoE) model at scale.

This post is about the parallelism you reach for when your model is *sparse*: you have far more parameters than you have FLOPs, because most of those parameters are **experts** — independent feed-forward networks — and each token only ever touches a couple of them. The parameters are too numerous and too big to replicate on every GPU the way ordinary data parallelism does. So you do the obvious thing: you put different experts on different GPUs. That is **expert parallelism** (EP), and it changes the shape of your communication completely. Data parallelism costs one gradient all-reduce per step. Tensor parallelism costs an all-reduce inside every layer. Expert parallelism costs two all-to-alls per MoE layer in the forward pass — and two more in the backward pass — because tokens have to physically travel to whichever GPU owns their assigned expert, then travel back.

![an anatomy of one Mixture-of-Experts layer where a router picks experts, a dispatch all-to-all sends each token to the GPU holding its expert, the experts compute, and a combine all-to-all returns the results for a weighted sum](/imgs/blogs/expert-parallelism-moe-1.webp)

By the end you will be able to: explain, from a systems lens, why MoE decouples parameter count from FLOPs and why that forces experts onto separate GPUs; derive the all-to-all byte volume $k \cdot T \cdot (E-1)/E \cdot d$ per GPU and predict whether it will be free (intra-node NVLink) or brutal (cross-node InfiniBand); reason about **capacity factor** and **dropped tokens** and why the buffers have to be a fixed size at all; diagnose and fix **load imbalance** — the war-story failure where a collapsed router overloads a few "hot" experts and the whole step stalls on the slowest GPU — with the auxiliary load-balancing loss and expert-choice routing; lay out an EP-by-DP device grid and know which gradients all-reduce over which group; and write a working MoE layer with a top-k router, the two `all_to_all` calls, capacity-drop logic, and the aux loss, plus the DeepSpeed-MoE and Megatron flags that do it for you in production. This is a post in the [Distributed Training in the Trenches](/blog/machine-learning/distributed-training/why-distributed-training) series, and it slots into the same frame as every other: you pick a parallelism to knock down one of the four walls, and you pay for it in collectives over the interconnect.

## Mixture-of-experts from a systems lens: parameters without the FLOPs

Start with the dense Transformer block you already know. Attention, then a feed-forward network (FFN): a linear up-projection from the model dimension $d$ to some larger hidden dimension (typically $4d$), a nonlinearity, and a linear down-projection back to $d$. That FFN is where most of a Transformer's parameters and most of its FLOPs live. If you want a bigger, more capable model, the crude lever is to make that FFN wider — but a wider FFN costs more compute *for every single token*, and compute is the thing you are trying to conserve.

Mixture-of-experts breaks the link between "how many parameters the model has" and "how many parameters run per token." Instead of one FFN, you install $N$ separate FFNs — the **experts** — and a small **router** (also called the gating network): a linear layer that, for each token, produces a score over the $N$ experts. The router picks the **top-k** experts for that token (usually $k=1$ or $k=2$), and only those $k$ experts run. Their outputs are combined, weighted by the router's scores, and that is the layer's output. A token that scores expert 5 and expert 41 highest gets processed by experts 5 and 41 and *nobody else*. The other $N-2$ experts, for that token, do nothing.

The arithmetic is the whole point. The model *stores* all $N$ experts, so the parameter count scales with $N$. But the model only *runs* $k$ of them per token, so the FLOPs scale with $k$. Set $N=8$ and $k=2$ and you get a model with roughly $8/2 = 4$ times the FFN parameters of a dense model at the same per-token cost. This is exactly Mixtral 8x7B: eight experts, top-2 routing, about 47 billion total parameters but only about 13 billion active for any given token. Switch Transformer went to top-1 and scaled the expert count to thousands, reaching 1.6 trillion parameters while keeping the per-token compute of a far smaller model. GLaM ran 64 experts per MoE layer at top-2 and 1.2 trillion parameters. DeepSeek-V3 pushed to 256 routed experts plus a shared one, top-8, 671 billion total parameters with only about 37 billion active per token. The pattern is always the same: **a lot of parameters, a little compute.**

Figure 1 above is one MoE layer drawn as a dataflow. Tokens arrive; the router scores them and picks top-k; a dispatch step routes each token to its chosen experts; the experts compute; a combine step brings the results back; a weighted sum produces the output. In a single-GPU toy model, "dispatch" and "combine" are just indexing operations — you gather the tokens for each expert, run the expert, scatter the results back. There is no communication. The trouble starts the moment the experts do not all fit on one GPU.

And they usually do not. This is the systems consequence that defines the rest of the post. Experts are full FFNs — each one is as big as the dense FFN it replaced — and there are many of them. Storing $N=64$ experts, each the size of a $4d$-wide FFN, is $64$ times the FFN memory of a dense model. You cannot replicate all of that on every GPU; that is the entire reason MoE is attractive (parameters you *couldn't* afford to replicate) and the entire reason it is hard (parameters you now have to *split*). So you split them: **different experts live on different GPUs.** That is expert parallelism, and now "dispatch" and "combine" are not indexing operations. They are network transfers.

## Expert parallelism: put different experts on different GPUs

Expert parallelism is almost embarrassingly simple to state. You have $N$ experts and an **EP degree** $E$ — the number of GPUs you spread the experts across, forming an **expert-parallel group**. Each GPU in that group holds $N/E$ experts and nothing else's experts. If $N=64$ and $E=8$, every GPU stores 8 experts. If $N=8$ and $E=8$, every GPU stores exactly one expert. The router, the attention, the embeddings, the layer norms — the *non-expert* parameters — are small and are handled separately (replicated or data-parallel-sharded, which we will get to). Only the big, numerous expert parameters are partitioned by EP.

Contrast this with data parallelism, which is the reflex most engineers reach for first. In pure data parallelism (DDP), *every* GPU holds a full copy of *every* parameter, and the only communication is a gradient all-reduce at the end of the backward pass. That works beautifully until the parameters stop fitting. For a sparse MoE model, replicating all $N$ experts on every GPU defeats the purpose — you would need a GPU big enough to hold the whole 47B or 671B parameter set, which is exactly the wall you were trying to avoid. Expert parallelism is the answer to the "model won't fit" wall specifically for the expert parameters: each GPU only has to hold its $N/E$ slice, so the per-GPU expert memory drops by a factor of $E$.

Here is the memory picture for one MoE layer, using the Mixtral-ish numbers. Say $d = 4096$ and the expert hidden dimension is $4d = 16384$. One expert's two matrices are about $d \times 4d + 4d \times d = 2 \cdot 4 d^2 = 8 \cdot 4096^2 \approx 1.34 \times 10^8$ parameters, so roughly 134 million parameters per expert, or about 0.27 GB in bf16 (2 bytes each), plus optimizer state on top. With $N=64$ experts, the full expert set for one layer is about 8.6 billion parameters — over 17 GB in bf16 just for weights, before Adam's momentum and variance nearly triple it. No single 80 GB GPU wants to hold that for every layer *and* the activations *and* everything else. Split across $E=8$ GPUs, each GPU holds 8 experts, about 2.1 GB of weights per layer — comfortable. That factor-of-8 memory relief is why you do EP at all.

But every parallelism is a trade, and EP's bill comes due in communication. The tokens do not know or care which GPU owns their expert. A token sitting on GPU 0 might get routed to expert 41, which lives on GPU 5. To be processed, that token's activation vector has to *travel* from GPU 0 to GPU 5, get transformed by expert 41, and travel back. Multiply that by every token on every GPU going to experts scattered across the whole EP group, and you have the defining communication pattern of MoE training. It is not an all-reduce. It is an all-to-all.

## The all-to-all: the beating heart of MoE communication

If you have read [Collectives From Scratch](/blog/machine-learning/distributed-training/collectives-from-scratch) you already met the all-to-all as the odd one out — the collective where *every rank sends a different slice of its data to every other rank*. In an all-reduce, every rank ends up with the same result. In an all-gather, every rank ends up with the same concatenation. In an all-to-all, every rank ends up with something *different*: the collection of pieces that the other ranks each addressed specifically to it. It is a distributed transpose. Rank $i$ chops its send buffer into $E$ chunks and sends chunk $j$ to rank $j$; simultaneously it receives one chunk from every rank and concatenates them.

![a send matrix for an all-to-all across three GPUs where every GPU sends a distinct slice to every GPU, the diagonal stays local and the off-diagonal slices cross the interconnect](/imgs/blogs/expert-parallelism-moe-2.webp)

Figure 2 draws it as a send matrix. Row $i$ is what GPU $i$ sends; column $j$ is what GPU $j$ receives. The diagonal cells (GPU $i$ to GPU $i$) never leave the GPU — those are the tokens whose expert happens to live locally. Every off-diagonal cell is a slice that has to cross the interconnect. With $E$ GPUs there are $E^2$ cells, $E$ of them local and $E^2 - E = E(E-1)$ of them remote. That $E(E-1)$ scaling is why all-to-all is the most bandwidth-hungry collective in the toolbox: it saturates every link in the fabric at once, in both directions, with no reuse. A ring all-reduce is clever about moving each byte the minimum number of hops; an all-to-all has no such trick, because by definition every rank needs unique data from every other rank.

Now trace exactly where the two all-to-alls appear in one MoE layer. After the router assigns each token to its top-k experts, the tokens are grouped by destination GPU and a **dispatch all-to-all** sends each token to the GPU that owns its expert. The experts run locally on the tokens they received. Then a **combine all-to-all** sends every result back to the GPU the token came from, where it is scaled by the router weight and summed. Two all-to-alls, forward. The backward pass runs the mirror image — gradients flow back through the same routing, so there are two more all-to-alls in the backward pass. Four all-to-alls per MoE layer per step. In a model with, say, 30 MoE layers, that is 120 all-to-alls per training step, each one a full-fabric synchronizing collective.

### Deriving the byte volume

Let each GPU in the EP group process $T$ tokens locally per micro-batch. Each token is a vector of dimension $d$; in bf16 that is $2d$ bytes. With top-k routing, each token produces $k$ copies to dispatch (one per selected expert). So each GPU has $k \cdot T$ token-copies to send in the dispatch all-to-all. Of those, on average a fraction $1/E$ are destined for the local GPU's own experts (they never leave), and a fraction $(E-1)/E$ leave for other GPUs. The bytes that *depart* each GPU in one dispatch all-to-all are therefore

$$V_\text{dispatch} = k \cdot T \cdot \frac{E-1}{E} \cdot 2d \quad \text{bytes per GPU.}$$

The combine all-to-all sends the same-shaped results back, so it moves the same volume. The forward pass moves $2 V_\text{dispatch}$ per GPU per MoE layer; the backward roughly doubles that again. The time this takes is the volume divided by the *achieved* per-GPU all-to-all bandwidth $B$:

$$T_\text{a2a} = \frac{V_\text{dispatch}}{B}.$$

Two facts fall right out of this formula, and they are the two facts that decide whether EP is a joy or a nightmare. First, the volume is basically fixed by your token count, hidden size, and top-k — you cannot shrink it much without changing the model. Second, everything hangs on $B$, the achieved bandwidth, and $B$ swings by an order of magnitude depending on whether the all-to-all stays inside one node (over NVLink and NVSwitch) or has to cross nodes (over InfiniBand or RoCE). Intra-node, an H100's NVLink4 fabric moves roughly 900 GB/s aggregate per GPU and NVSwitch gives full all-to-all bandwidth, so achieved all-to-all bandwidth is in the hundreds of GB/s. Cross-node, a DGX H100 node typically has eight NDR400 InfiniBand NICs — about 400 GB/s of aggregate injection per node, or roughly 50 GB/s per GPU. That is the cliff. The same all-to-all that is nearly free on one node becomes the dominant cost the instant it has to leave the node.

The formula also explains why all-to-all is *sensitive to the token distribution* in a way an all-reduce never is. An all-reduce moves a fixed number of bytes regardless of the data. An all-to-all's per-link volume depends on how many tokens each GPU sends to each other GPU — and if the router sends a disproportionate pile of tokens to the experts on one GPU, that GPU's incoming links become the bottleneck and everyone waits for it. Hold that thought; it is the whole load-imbalance war story two sections down.

#### Worked example: a 64-expert MoE on 8 GPUs

Take a concrete layer. $N=64$ experts, EP degree $E=8$ (so 8 experts per GPU), top-1 routing ($k=1$) for a clean number, $d=4096$, bf16. Each GPU processes $T=8192$ tokens per micro-batch. A token vector is $2d = 8192$ bytes, or 8 KB.

Dispatch volume per GPU:

$$V_\text{dispatch} = 1 \cdot 8192 \cdot \frac{7}{8} \cdot 8192 \approx 5.87 \times 10^{7} \text{ bytes} \approx 56 \text{ MB.}$$

Combine moves another 56 MB, so the forward all-to-all is about 112 MB per GPU for this one layer. Now price it on two fabrics.

*All eight GPUs on one DGX H100 node (NVLink/NVSwitch).* Take an achieved all-to-all bandwidth of roughly 450 GB/s per GPU (well below the 900 GB/s aggregate peak, which is realistic for all-to-all). Each dispatch takes $56 \times 10^6 / 450 \times 10^9 \approx 124$ microseconds; the forward pair is about 250 microseconds. Compare that to the expert compute: 8192 tokens through an FFN with $d=4096$, $4d=16384$ is about $2 \cdot 2 \cdot T \cdot d \cdot 4d = 16 T d^2 \approx 2.2 \times 10^{12}$ FLOPs, and on an H100 sustaining, say, 500 bf16 TFLOP/s that is about 4.4 milliseconds. The all-to-all is roughly 250 microseconds against 4.4 milliseconds of compute — about 5% overhead. Intra-node EP is essentially free.

*The same eight GPUs split four-and-four across two nodes (InfiniBand).* Now much of the all-to-all traffic crosses the node boundary, and the achieved per-GPU bandwidth for the cross-node portion collapses to roughly 50 GB/s. Each dispatch now takes about $56 \times 10^6 / 50 \times 10^9 \approx 1.1$ milliseconds; the forward pair is about 2.2 milliseconds. Against the same 4.4 milliseconds of compute, the all-to-all is now a third of the step, and once you add the backward all-to-alls it can approach half. Same model, same math, and the interconnect turned a 5% tax into a 40% tax. This is the single most important number to internalize about EP: **all-to-all is cheap inside a node and expensive across nodes, and the gap is roughly a factor of ten.**

## Capacity and dropped tokens: why the buffers are a fixed size

There is a subtlety in that all-to-all that I glossed over, and it turns out to be central to how MoE training actually behaves. An all-to-all needs *statically known tensor shapes*. To send a chunk to every other rank, the collective has to know exactly how many bytes each chunk is — you cannot easily do a ragged, variable-length all-to-all where GPU 0 sends 900 tokens to expert 5 and GPU 1 sends 3 tokens to it. The tensors have to be rectangular. So MoE implementations give every expert a fixed-size buffer — its **capacity** — and pad or drop to fit.

Capacity is defined relative to the *expected* load. If tokens were routed perfectly uniformly, each expert would receive $k \cdot T_\text{tot} / N$ tokens, where $T_\text{tot}$ is the total tokens across the EP group. The capacity is that expected number scaled by a **capacity factor** $f$:

$$C = f \cdot \frac{k \cdot T_\text{tot}}{N}.$$

A capacity factor of 1.0 means "give each expert exactly enough room for its fair share." Switch Transformer used factors around 1.0 to 2.0; GShard used 2.0. The buffer is fixed at $C$ tokens per expert, and now two things can go wrong at the edges.

![a capacity buffer for one hot expert where arrivals exceed the fixed number of slots, the slots that fit are processed, and the overflow tokens are dropped and skip the layer](/imgs/blogs/expert-parallelism-moe-3.webp)

If *more* than $C$ tokens route to an expert, the overflow has nowhere to go. The standard behavior is to **drop** those tokens: they skip the expert entirely for this layer, and their output is just the residual connection carrying them through unchanged (no expert contribution). Some implementations instead reroute overflow to the token's second-choice expert, but plain dropping is common and simplest. Dropped tokens hurt model quality — a dropped token got no expert compute at that layer, as if the MoE layer briefly became the identity for it. If *fewer* than $C$ tokens route to an expert, the buffer is padded with zeros up to $C$, and the expert wastes compute processing padding, and the all-to-all wastes bandwidth moving it.

Figure 3 draws the overflow for one hot expert. The trade is now visible and sharp. **Low capacity factor** means small buffers: little padding waste, little all-to-all volume, cheap — but any imbalance immediately drops tokens and dents quality. **High capacity factor** means big buffers: fewer drops, higher quality — but proportionally more padding compute and more all-to-all bytes, because $V_\text{dispatch}$ scales directly with $C$. Doubling the capacity factor doubles the volume in the byte law above. You cannot buy your way out of a badly balanced router by cranking capacity: to guarantee a 4x-overloaded expert never drops, you would need $f \geq 4$, which quadruples the all-to-all cost and the padding compute *for every expert*, most of which are not overloaded. That is untenable. The real fix for drops is not a bigger buffer; it is a router that spreads tokens evenly in the first place. Which brings us to the failure that pages you at 3am.

## Load imbalance: the war story

Here is the scenario, and it is extremely common early in an MoE run. You launch a 64-expert model on 8 GPUs. The loss is coming down. But your throughput is a fraction of what the FLOP math promised, and when you profile, the GPUs are wildly uneven: GPU 5 is pinned at 100% utilization while GPUs 0 through 4 are idling at 20%, waiting. You dig into the router statistics and find the problem. The router has **collapsed**: it learned early that a couple of experts were slightly better and started sending most tokens to them, which trained those experts faster, which made them better still, which sent them even more tokens. A runaway feedback loop. Now expert 41 is receiving four times its fair share of tokens and expert 12 is receiving almost none.

![a before and after comparison of expert load where a collapsed router overloads one expert and idles the rest, then an auxiliary loss balances the load and restores throughput](/imgs/blogs/expert-parallelism-moe-4.webp)

Two things go wrong at once, and figure 4 contrasts the broken and the fixed run. First, quality: the overloaded expert's buffer overflows and drops thousands of tokens every step, so a chunk of your batch gets no expert compute. Second — and this is the throughput killer — the all-to-all is a *synchronizing collective*. Every GPU has to finish its part before any GPU can proceed, so the step time is set by the slowest GPU, which is the one whose local experts got dumped on. GPU 5 is grinding through 4x the tokens while everyone else finished long ago and sits at a barrier. Your eight-GPU job is running at the speed of its most overloaded GPU. The straggler here is not a slow *device*; it is a device the router *made* slow by handing it too much work. You can have perfect hardware and still get straggler behavior purely from a lopsided routing distribution.

### The auxiliary load-balancing loss

The fix that made large MoE training practical is an **auxiliary load-balancing loss**, added to the main training loss, that pushes the router toward uniform assignment. The Switch Transformer formulation is the canonical one. For a batch, let $f_i$ be the fraction of tokens dispatched to expert $i$ (a hard count, the actual load), and let $P_i$ be the average router probability assigned to expert $i$ across the batch (a soft, differentiable quantity). The auxiliary loss is

$$\mathcal{L}_\text{aux} = \alpha \cdot N \sum_{i=1}^{N} f_i \, P_i.$$

The intuition: $f_i$ measures how overloaded expert $i$ actually is, and $P_i$ is the differentiable knob the optimizer can turn to change that. Multiplying them and summing penalizes the router most for confidently ($P_i$ high) piling tokens onto already-loaded experts ($f_i$ high). The whole sum is minimized when the load is perfectly uniform, $f_i = P_i = 1/N$ for all $i$, at which point $\mathcal{L}_\text{aux} = \alpha \cdot N \cdot N \cdot (1/N)^2 = \alpha$. The coefficient $\alpha$ is small — around 0.01 — because you want just enough pressure to keep the router honest without overwhelming the actual language-modeling objective. Turn $\alpha$ too low and the router collapses; turn it too high and you force uniformity so hard the experts never specialize and quality suffers. It is a genuine knob you tune.

An alternative that sidesteps the aux loss entirely is **expert-choice routing** (Zhou et al., 2022). Flip the assignment around: instead of each *token* choosing its top-k experts, each *expert* chooses its top-$C$ tokens. By construction every expert receives exactly $C$ tokens — perfect load balance, guaranteed, no auxiliary loss and no drops in the usual sense. The catch is that a given token might be chosen by zero experts (it gets no compute that layer) or by many, and the variable per-token expert count complicates autoregressive decoding, which is why token-choice with an aux loss remains the mainstream approach for decoder LLMs. DeepSeek-V3 took yet another route: an **auxiliary-loss-free** strategy that adds a per-expert bias to the routing scores and nudges those biases up or down based on recent load, achieving balance without the aux loss's gradient interference with the main objective. The idea in all three is identical — get the tokens spread evenly so no GPU becomes the straggler — and only the mechanism differs.

#### Worked example: a hot expert at four times its share

Concrete numbers. Total tokens per step across the EP group: 8 GPUs times 8192 tokens is 65,536. With $N=64$ experts and top-1, the fair share per expert is $65{,}536 / 64 = 1024$ tokens. Set the capacity factor to 1.0, so each expert's buffer is 1024 slots.

*Before (collapsed router).* Expert 41 receives 4096 tokens — four times its share. Its buffer holds 1024, so $4096 - 1024 = 3072$ tokens are dropped at that expert every step, about 4.7% of the whole batch gone from that layer. Meanwhile GPU 5, which owns expert 41, must run its experts on far more tokens than the others; because the all-to-all synchronizes, the step time stretches to match GPU 5. The effective compute is spread so unevenly that MFU (model FLOPs utilization) sits around 12%: seven of eight GPUs spend most of the step idle at a barrier. Throughput is roughly 22,000 tokens/second for the layer's share of the step.

*After (aux loss, $\alpha = 0.01$).* Over a few thousand steps the auxiliary loss trains the router toward uniform. The ratio of the busiest expert's load to the mean load — the metric you actually watch — falls from about 4.0 to about 1.1. Drops fall below 1% of the batch. Now every GPU processes close to its fair 1024 tokens, the all-to-all no longer stalls on one overloaded rank, and MFU recovers to about 38%. Throughput climbs to roughly 58,000 tokens/second — about 2.6x faster — purely from balancing the router, without touching the hardware, the batch size, or the capacity factor. That multiplier is why the aux loss is not optional; a collapsed MoE is a very expensive dense model.

## Composing expert parallelism with data (and tensor) parallelism

Expert parallelism rarely rides alone. It handles the expert parameters, but a Transformer is mostly *not* experts — attention, embeddings, the router itself, the layer norms — and those non-expert parameters want ordinary data parallelism (or FSDP sharding) so you can scale the batch. So the real layout is EP *composed with* DP, and getting the composition right is where MoE training gets its reputation for being fiddly.

![a device layout for expert parallelism composed with data parallelism where two replicas each hold the full expert set, tokens all-to-all within a replica, expert gradients all-reduce across the two replicas, and non-expert gradients all-reduce across all sixteen GPUs](/imgs/blogs/expert-parallelism-moe-5.webp)

Figure 5 lays out a 16-GPU example: EP degree 8, DP degree 2. Read it as two **replicas**, each an 8-GPU expert-parallel group that together hold the complete set of 64 experts exactly once. So the full expert set exists twice across the cluster, once per replica. That gives you three distinct communication scopes, and confusing them is a classic MoE bug:

- **Token all-to-all** happens *within* each replica's 8-GPU EP group. Tokens never cross replica boundaries during routing; each replica is self-contained for the forward and backward expert compute.
- **Expert gradients** all-reduce across the **expert-data-parallel group** — the small set of GPUs, one per replica, that hold the *same* expert. With DP degree 2, each expert lives on exactly 2 GPUs (one per replica), so expert gradients all-reduce over just those 2. Not over all 16.
- **Non-expert gradients** (attention, router, embeddings) all-reduce across the *full* 16-GPU data-parallel group, because those parameters are replicated everywhere.

Mix these up — for instance, all-reducing expert gradients over all 16 GPUs instead of the expert-DP group of 2 — and you either average the wrong things or hang on a mismatched collective. DeepSpeed-MoE and Megatron-Core build these process groups for you precisely so you do not hand-roll them, but you still have to understand the scoping to read a profiler trace or debug a stall.

When a single expert is itself too big for one GPU, you add a third dimension: **tensor parallelism inside the expert**, splitting each expert's FFN matmul across GPUs the way [the map of parallelism](/blog/machine-learning/distributed-training/the-map-of-parallelism) lays out for dense layers. That gives a full three-dimensional EP-by-TP-by-DP grid, which is exactly the territory the [3D parallelism](/blog/machine-learning/distributed-training/3d-parallelism) post covers; MoE just adds the expert dimension on top. The rule of thumb for placement is the same one the byte law implies: keep the all-to-all inside a node. Put the EP group on the eight NVLink-connected GPUs of one node whenever you can, and let DP be the dimension that crosses nodes, because DP's all-reduce tolerates the slower inter-node fabric far better than EP's all-to-all does. When the expert count forces EP across nodes, the mitigation the big labs use is **node-limited (or device-limited) routing**: cap the number of nodes any single token's experts may span (DeepSeek-V3 limits each token to experts on at most four nodes), which bounds the cross-node all-to-all volume at the cost of a slightly constrained router.

## The code: a minimal MoE layer end to end

Enough theory; here is the machinery. This is a simplified but structurally correct expert-parallel MoE layer — top-1 routing, fixed capacity, one all-to-all each way — written against `torch.distributed`. The point is to make the dispatch/combine collectives and the drop logic concrete; production layers add top-2, expert-choice, and fused kernels, but the skeleton is this.

First, the router and the auxiliary loss. The router is just a linear layer to $N$ logits; the aux loss is the Switch formulation derived above.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class Router(nn.Module):
    def __init__(self, d_model, num_experts, aux_alpha=0.01):
        super().__init__()
        self.num_experts = num_experts
        self.aux_alpha = aux_alpha
        # Route in fp32 for stability; the router is tiny and precision here matters.
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x):                      # x: [T, d_model] on this rank
        logits = self.gate(x.float())          # [T, num_experts]
        probs = F.softmax(logits, dim=-1)      # differentiable soft assignment
        gate_weight, expert_idx = probs.max(dim=-1)   # top-1: [T], [T]

        # Switch auxiliary load-balancing loss.
        # f_i = fraction of tokens dispatched to expert i (hard count).
        # P_i = mean router probability for expert i (soft, differentiable).
        f = F.one_hot(expert_idx, self.num_experts).float().mean(dim=0)  # [E_total]
        P = probs.mean(dim=0)                                            # [E_total]
        aux_loss = self.aux_alpha * self.num_experts * torch.sum(f * P)

        return gate_weight, expert_idx, aux_loss
```

Next, the capacity and drop logic. For each token we compute its **position within its expert's buffer** using a cumulative sum over the one-hot assignment; any token whose position lands at or beyond the capacity is dropped.

```python
def assign_slots(expert_idx, num_experts, capacity):
    """Return per-token (slot, keep) where slot is the token's index within its
    expert's fixed-size buffer, and keep is False for overflow (dropped) tokens."""
    one_hot = F.one_hot(expert_idx, num_experts)               # [T, E_total]
    # cumulative count of tokens seen so far for each expert -> this token's slot
    position = one_hot.cumsum(dim=0) - 1                        # [T, E_total]
    slot = position.gather(1, expert_idx.unsqueeze(1)).squeeze(1)   # [T]
    keep = slot < capacity                                     # drop the overflow
    return slot, keep
```

Now the heart of it: building the dense dispatch buffer indexed by *global* expert, doing the dispatch all-to-all, running the local experts, and doing the combine all-to-all back. With a fixed capacity, every tensor shape is static, which is exactly why the all-to-all works.

```python
class ExpertParallelMoE(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, ep_group, capacity_factor=1.25):
        super().__init__()
        self.ep_group = ep_group
        self.ep_size = dist.get_world_size(ep_group)
        assert num_experts % self.ep_size == 0
        self.num_experts = num_experts
        self.local_experts = num_experts // self.ep_size      # experts on THIS rank
        self.capacity_factor = capacity_factor
        self.router = Router(d_model, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
            for _ in range(self.local_experts)
        ])

    def forward(self, x):                                     # x: [T, d]
        T, d = x.shape
        gate_weight, expert_idx, aux_loss = self.router(x)

        # capacity = f * (global tokens) / num_experts, per expert
        capacity = int(self.capacity_factor * (T * self.ep_size) / self.num_experts)
        slot, keep = assign_slots(expert_idx, self.num_experts, capacity)

        # Scatter kept tokens into a dense buffer indexed by [global_expert, slot].
        dispatch = x.new_zeros(self.num_experts, capacity, d)
        e_keep = expert_idx[keep]
        s_keep = slot[keep]
        dispatch[e_keep, s_keep] = x[keep]

        # Dispatch all-to-all: reshape so dim 0 is the EP rank axis, then exchange.
        # After this, recv[j] holds the tokens rank j sent for MY local experts.
        dispatch = dispatch.view(self.ep_size, self.local_experts * capacity * d)
        recv = torch.empty_like(dispatch)
        dist.all_to_all_single(recv, dispatch, group=self.ep_group)
        recv = recv.view(self.ep_size, self.local_experts, capacity, d)

        # Run each local expert on all the tokens it received from every source rank.
        out = torch.empty_like(recv)
        for le in range(self.local_experts):
            out[:, le] = self.experts[le](recv[:, le])        # [ep_size, capacity, d]

        # Combine all-to-all: send the results back the way they came.
        out = out.view(self.ep_size, self.local_experts * capacity * d)
        combined = torch.empty_like(out)
        dist.all_to_all_single(combined, out, group=self.ep_group)
        combined = combined.view(self.num_experts, capacity, d)

        # Scatter results back to token positions, scaled by the router weight.
        y = torch.zeros_like(x)
        y[keep] = combined[e_keep, s_keep] * gate_weight[keep].unsqueeze(-1)
        return y, aux_loss
```

The training loop then adds the aux loss to the main loss before the backward pass, so the router gets its balancing gradient:

```python
y, aux_loss = moe_layer(hidden)
loss = task_loss(y, target) + aux_loss     # aux_loss already scaled by alpha
loss.backward()
```

In production you would not hand-write this. **DeepSpeed-MoE** wraps the whole pattern behind one module — you hand it an expert template and it builds the process groups, the capacity logic, and both all-to-alls:

```python
from deepspeed.moe.layer import MoE

expert = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
self.moe = MoE(
    hidden_size=d_model,
    expert=expert,
    num_experts=64,
    ep_size=8,                 # expert-parallel degree
    k=1,                       # top-k
    capacity_factor=1.25,      # training capacity
    eval_capacity_factor=2.0,  # more slack at eval (drops hurt more there)
    use_residual=False,
)
```

**Megatron-Core** exposes the same knobs as launch flags, and this is what a real large-scale MoE run looks like on the command line:

```bash
torchrun --nproc_per_node=8 --nnodes=4 pretrain_gpt.py \
  --num-experts 64 \
  --expert-model-parallel-size 8 \
  --moe-router-topk 2 \
  --moe-router-load-balancing-type aux_loss \
  --moe-aux-loss-coeff 0.01 \
  --moe-expert-capacity-factor 1.25 \
  --moe-token-dropping \
  --moe-grouped-gemm \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1
```

The `--expert-model-parallel-size` is the EP degree; `--moe-grouped-gemm` fuses the many small per-expert matmuls into one batched kernel (a real throughput win, since dozens of tiny GEMMs otherwise leave the GPU starved); and the load-balancing and capacity flags map one-to-one onto the mechanics above.

## Measuring an MoE run honestly

You cannot tune what you do not measure, and MoE has failure modes that a dense-training dashboard will not show you. Three numbers belong on your MoE dashboard, in addition to the usual tokens/s and MFU.

**All-to-all time as a fraction of the step.** Wrap the collectives in CUDA events or read them off a `torch.profiler` / Nsight trace, and watch the ratio $T_\text{a2a} / T_\text{step}$. If it is under 10%, EP is nearly free and you are almost certainly intra-node. If it is 30% or more, you are paying the cross-node tax and should look hard at your placement and at node-limited routing. As always, measure in steady state: run a dozen warm-up steps first, call `torch.cuda.synchronize()` before and after the region you time, and average over many steps, because the first steps include allocator warm-up and NCCL's connection setup and will lie to you.

**Load imbalance: the max-to-mean ratio.** Log, per step, the maximum expert load divided by the mean expert load. Perfectly balanced is 1.0; a collapsing router climbs toward $N$ in the worst case. Watch this like a hawk in the first few thousand steps — it is your early warning that the aux loss coefficient is too low. A healthy run settles around 1.05 to 1.2.

**Drop rate.** The fraction of tokens that overflowed capacity and were dropped. A percent or two is usually tolerable in training; if it is climbing into double digits you are either imbalanced (fix the router) or running the capacity factor too low (raise it, but only after the router is balanced). The two knobs interact, which is why you read them together.

Here is a before-and-after table for the 64-expert layer from the worked examples, on a single DGX H100 node (8x H100 SXM, NVLink4/NVSwitch), all numbers per the layer's share of the step and rounded:

| Metric | Collapsed router | After aux loss | After node-limited placement |
|---|---|---|---|
| Max/mean expert load | 4.0 | 1.1 | 1.1 |
| Token drop rate | 4.7% | 0.8% | 0.7% |
| All-to-all share of step | 34% (cross-node) | 33% | 6% (intra-node) |
| MFU | 12% | 24% | 41% |
| Throughput (tokens/s) | ~22k | ~40k | ~62k |

Read the columns as the two independent fixes, because they are independent: the aux loss cures the *imbalance* (max/mean and drops), and moving the EP group onto one node's NVLink fabric cures the *all-to-all cost*. A collapsed, cross-node MoE gets hit by both problems at once, which is why the naive setup looks so bad and why fixing one alone leaves throughput on the table. You need both — a balanced router *and* the all-to-all kept inside a node — to get MoE's promised efficiency. The numbers here are illustrative of the shape of the improvement rather than a specific published run, but the *directions* and *rough magnitudes* are exactly what you see in practice.

## Debugging and tuning an MoE run at scale

The good news is that the tuning is a short, ordered loop, and once you have run it a few times it becomes mechanical.

![an ordered tuning loop that measures load imbalance, adds an auxiliary loss, tunes the capacity factor, limits routing across nodes, overlaps the all-to-all with compute, and re-measures a recovered MFU](/imgs/blogs/expert-parallelism-moe-6.webp)

Figure 6 walks the loop. **Measure the imbalance first** — max/mean load and drop rate — because if the router has collapsed, nothing else matters and every other optimization is polishing a stalled run. **Add or raise the auxiliary loss** ($\alpha$ around 0.01, up if the ratio stays high) and watch max/mean fall toward 1.1 over a few thousand steps. **Then tune the capacity factor**: with a balanced router you can usually *lower* it from 2.0 toward 1.25 or even 1.0, which cuts both the padding compute and the all-to-all volume, buying back throughput you were spending on slack you no longer need. **Then attack the all-to-all cost itself**: get the EP group onto one node if at all possible, and if you must span nodes, turn on node-limited routing to bound the cross-node volume. **Finally, overlap the all-to-all with compute** — the frameworks increasingly pipeline the dispatch of the next micro-batch's tokens against the current one's expert compute, the way DeepSeek-V3's DualPipe schedule hides almost all of the all-to-all behind computation. Overlap is the same lever that makes DDP scale (comms hidden under the backward pass); MoE just has more comms to hide. **Re-measure**, and repeat if any number is still off.

The stress tests are where you learn whether your configuration is robust. *What happens at 64 GPUs across 8 nodes?* The all-to-all now spans the whole cluster and the cross-node bandwidth is your ceiling; node-limited routing and overlap stop being nice-to-haves and become mandatory. *What happens with a tiny per-GPU batch?* The all-to-all becomes latency-bound rather than bandwidth-bound — each message is small, so you are paying fixed per-message overhead $E(E-1)$ times, and MoE efficiency craters; MoE wants large token batches per GPU to amortize the all-to-all. *What happens when one node is a genuine straggler* (a throttling GPU, a flaky NIC)? Because the all-to-all synchronizes, that one node sets the pace for the entire EP group, exactly as an overloaded expert did — the difference is you cannot fix a bad NIC with an aux loss; you find it (per-rank timing) and evict it. *What happens when the optimizer state won't fit?* Expert parallelism shards the expert *parameters* but the optimizer state for the experts each GPU owns still lives on that GPU; you compose EP with ZeRO/FSDP on the non-expert parameters and expert-data-parallel sharding on the expert optimizer state, which is precisely the composition figure 5 sketched.

## Case studies and real numbers

The MoE literature is unusually candid about the systems trade-offs, because they are unavoidable.

**Switch Transformer** (Fedus, Zoph, Shazeer, 2021) is the reference implementation of everything above: top-1 routing (they showed a single expert is enough and cuts the all-to-all in half versus top-2), a fixed capacity factor, the auxiliary load-balancing loss with coefficient 0.01, and a careful detail — routing computed in fp32 even when the rest of the model is bf16, because the router's argmax is sensitive to small numerical differences that can flip an expert assignment and desync ranks. They scaled to 1.6 trillion parameters and reported roughly a 7x pretraining speedup over a dense T5 baseline at matched compute, which is the entire promise of sparsity made concrete.

**GShard** (Lepikhin et al., 2020) trained a 600-billion-parameter multilingual translation MoE and is where the all-to-all-based expert dispatch and the capacity-factor mechanism were first laid out at scale, across TPU pods where the inter-node fabric made the all-to-all cost front and center.

**GLaM** (Du et al., 2022) ran 64 experts per MoE layer at top-2 for 1.2 trillion total parameters and reported training to better quality than GPT-3 while using roughly a third of the training energy — the sparse-compute argument stated as a power bill.

**Mixtral 8x7B** (Jiang et al., 2024) is the one most engineers can actually run: 8 experts, top-2, about 47B total and 13B active, matching or beating much larger dense models at the inference cost of a 13B. Its small expert count makes it a friendly first MoE — with only 8 experts, EP degree 8 puts one expert per GPU and the all-to-all stays trivially intra-node on a single 8-GPU box.

**DeepSeek-V3** (2024) is the current master class in taming the all-to-all: 256 routed experts plus a shared expert, top-8, 671B total and 37B active, trained on a 2048-GPU H800 cluster. It combines the auxiliary-loss-free balancing (bias-adjusted routing, so the main objective is never distorted by a balancing gradient), node-limited routing (each token's experts confined to at most four nodes, bounding cross-node all-to-all), and the DualPipe schedule that overlaps almost all of the all-to-all communication behind computation. The reported result is a very high MFU for a sparse model at that scale — the payoff of doing every lever in this post at once. If you want the inference-time counterpart to all of this, the [optimizing MoE training and inference](/blog/machine-learning/large-language-model/optimizing-moe-training-and-inference) post picks up where serving becomes the concern.

The through-line across all five: the model-quality story (sparsity buys parameters cheaply) is settled, and the *entire* remaining engineering effort is the systems story — balancing the router and taming the all-to-all. That is the work.

## When to reach for expert parallelism (and when not)

Expert parallelism is not a general-purpose scaling lever like data parallelism. It is specifically the tool for **sparse models with many large experts**, and reaching for it in the wrong situation buys you the all-to-all tax for nothing.

![a comparison of data, tensor, pipeline, and expert parallelism by what each splits, its main communication cost, and when it pays off](/imgs/blogs/expert-parallelism-moe-7.webp)

Figure 7 puts EP next to its siblings. Data parallelism splits the batch and costs one gradient all-reduce; it is where you always start and it should saturate before you add anything else. Tensor parallelism splits a layer's matmul and costs an all-reduce per layer; it earns its keep when a single layer is too big to fit and you keep it intra-node. Pipeline parallelism splits the layers into stages and costs point-to-point transfers plus a bubble; it pays for very deep models across nodes. Expert parallelism splits the *experts* and costs two all-to-alls per layer; it pays when — and only when — you are training a sparse MoE with enough experts that they cannot be replicated.

Concretely, reach for EP when your parameter count is dominated by experts you cannot afford to replicate, and when you can keep the expert-parallel group inside a single node's NVLink domain (or you have very fat inter-node InfiniBand *and* you are willing to run node-limited routing and overlap). Do *not* reach for it when your model is dense — there are no experts to split, and you should be using DP, TP, or PP. Do not reach for it when the model already fits comfortably and DP saturates your interconnect — you would be adding all-to-all cost for no memory relief. Do not naively push EP across nodes on thin interconnect and expect it to scale; the byte law guarantees the all-to-all will dominate and your expensive sparse model will run like a slow dense one. And do not turn on MoE without an aux loss (or an equivalent balancing scheme) and a plan to watch the max/mean load, because a collapsed router will quietly convert your throughput advantage into a straggler nightmare. MoE is a genuine superpower for parameter efficiency, but only after you have paid the two systems tolls — balance and bandwidth — that this entire post has been about. When you are composing it with everything else, the [distributed training playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook) capstone ties EP back to the full decision tree.

## Key takeaways

- **MoE trades parameters for FLOPs.** Store $N$ experts, run only $k$ per token; parameters scale with $N$, compute with $k$. That is why MoE gives you a 47B-parameter model at 13B-parameter cost — and why the experts, being big and numerous, must be split across GPUs.
- **Expert parallelism puts different experts on different GPUs.** Each of $E$ GPUs holds $N/E$ experts, cutting per-GPU expert memory by a factor of $E$. That is the memory win that makes sparse models trainable at all.
- **The all-to-all is the defining cost.** Tokens route to experts on other GPUs, so every MoE layer does a dispatch all-to-all and a combine all-to-all (and two more in the backward pass). It saturates every link at once and has no reuse trick.
- **The byte law is $V = k \cdot T \cdot (E-1)/E \cdot 2d$ per GPU, and everything hangs on the achieved bandwidth.** Intra-node NVLink makes the all-to-all a ~5% overhead; cross-node InfiniBand can make it 40% of the step. Keep the EP group inside a node.
- **Capacity is a fixed-size buffer because the all-to-all needs static shapes.** Overflow tokens are dropped (they skip the expert); low capacity is cheap but drops, high capacity is wasteful padding. You cannot fix imbalance by cranking capacity.
- **A collapsed router is a straggler you built yourself.** Hot experts overload their GPUs, and because the all-to-all synchronizes, the whole step stalls on the slowest. The auxiliary load-balancing loss ($\mathcal{L}_\text{aux} = \alpha N \sum_i f_i P_i$, $\alpha \approx 0.01$) pushes the router toward uniform; expert-choice and bias-adjusted routing are alternatives.
- **Watch three numbers: all-to-all share of step, max/mean expert load, and drop rate.** They diagnose the two independent failure modes — bandwidth and balance — and you fix them separately.
- **Compose EP with DP (and TP).** Expert grads all-reduce over the small expert-DP group; non-expert grads all-reduce over the full DP group; token all-to-alls stay within a replica. Mixing the scopes is a classic bug.
- **EP is for sparse models only.** If the model is dense or already fits with DP saturating the interconnect, EP adds all-to-all cost for no benefit.

## Further reading

- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) — Fedus, Zoph, Shazeer (2021). Top-1 routing, the auxiliary load-balancing loss, capacity factor, selective fp32 routing.
- [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668) — Lepikhin et al. (2020). The all-to-all expert dispatch and capacity mechanism at pod scale.
- [Mixture-of-Experts with Expert Choice Routing](https://arxiv.org/abs/2202.09368) — Zhou et al. (2022). Inverting the assignment for perfect load balance without an aux loss.
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088) — Jiang et al. (2024). The friendly 8-expert, top-2 model most engineers can actually run.
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — auxiliary-loss-free balancing, node-limited routing, and DualPipe all-to-all overlap at 2048-GPU scale.
- [DeepSpeed-MoE](https://arxiv.org/abs/2201.05596) and the [PyTorch `all_to_all` docs](https://pytorch.org/docs/stable/distributed.html) — the toolchain that builds the process groups and collectives for you.
- Within this series: [why distributed training](/blog/machine-learning/distributed-training/why-distributed-training) for the four-walls frame, [the map of parallelism](/blog/machine-learning/distributed-training/the-map-of-parallelism) for where EP sits, [collectives from scratch](/blog/machine-learning/distributed-training/collectives-from-scratch) for the all-to-all primitive, [3D parallelism](/blog/machine-learning/distributed-training/3d-parallelism) for composing EP with TP and DP, and [the distributed training playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook) for the full decision checklist.
