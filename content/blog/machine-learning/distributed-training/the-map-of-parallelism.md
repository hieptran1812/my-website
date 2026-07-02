---
title: "The Map of Parallelism: Data, Tensor, Pipeline, Expert, Sequence"
date: "2026-07-02"
publishDate: "2026-07-02"
description: "A single map for every distributed-training decision — the five axes you can split work along, exactly what each one splits, replicates, and costs in communication, and how to name the one axis your run actually needs."
tags:
  [
    "distributed-training",
    "multi-gpu",
    "data-parallelism",
    "tensor-parallelism",
    "pipeline-parallelism",
    "deep-learning",
    "ml-systems",
    "pytorch",
    "fsdp",
    "megatron",
  ]
category: "machine-learning"
subcategory: "Distributed Training"
author: "Hiep Tran"
featured: true
readTime: 41
---

Here is a conversation that happens on every team that decides to train something bigger than the last thing they trained. An engineer walks over and says, "The model doesn't fit anymore. I added GPUs. It's still not working — now it either OOMs on rank 0, or it runs, but eight GPUs give me barely more throughput than one." And then the question that has no single answer: "So how do I make this train across the cluster?"

The reason that question is hard is that "make it train across the cluster" is not one decision. It is a family of decisions, and they are not interchangeable. You can split the *batch* and give every GPU a full copy of the model. You can split the *weight matrices* so a layer too big for one card lives across eight. You can split the *layers* into stages and stream micro-batches through them. You can split the *experts* of a mixture-of-experts model. You can split the *sequence* so a context that overflows attention on one GPU is shared across several. Each of these is a different axis of the same cube of work, each replicates something different, and — this is the part that bites people — each one bills you in a different communication currency: an all-reduce here, an all-to-all there, a point-to-point handoff somewhere else. Pick the axis that doesn't match your interconnect and your eight GPUs really will run barely faster than one.

This post is the map. Not a deep manual for any single technique — the sibling posts in this series do that — but the atlas you keep open on the other monitor so that when the model doesn't fit, you can *name the axis you need* in thirty seconds instead of flailing across a config file. By the end you will be able to look at a model, a cluster, and an interconnect and say: "this is a tensor-parallel problem," or "this is just data parallelism with sharding," or "this needs pipeline stages," and know *why* — what each choice splits, what it duplicates, and the exact collective it will cost you on every step. Figure 1 is the whole map on one page; everything after it is that map, drawn in detail.

![A branching diagram that starts from the single question of which axis of the work to split and fans out into the five parallelism strategies, each labeled with the communication collective it costs](/imgs/blogs/the-map-of-parallelism-1.webp)

This is the second stop in the series. The [intro post on why we train across many GPUs at all](/blog/machine-learning/distributed-training/why-distributed-training) frames the four walls that force your hand — the model won't fit, the data won't finish, the run is too slow, the cost is too high. This post answers the very next question those walls raise: *along which axis do I split the work to knock the wall down?* And the [capstone playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook) turns the whole series into a checklist. Keep those two open; we will hand off to the deep-dive posts as each axis earns its own chapter.

## 1. The one question every decision answers

Strip away the framework names, the config flags, and the vendor slides, and every distributed-training design reduces to a single question: **which axis of the work do you split, and which do you replicate?**

A training step is a cube. One dimension is the *batch* — the independent examples you process together. Another is the *model* — the parameters, arranged in layers, each layer a stack of matrix multiplies. A third, for a Transformer, is the *sequence* — the tokens within each example. Every parallelism strategy is a choice of how to slice that cube across your GPUs, and the choice is forced by which dimension is too big to fit or too slow to finish on one device.

There are exactly five useful slices, and Figure 1 places all of them. Two of them slice the *batch* side (you keep a whole model on each GPU and feed it different data); three of them slice the *model* side (you keep the batch and cut the parameters apart). Here they are in one breath, and then we spend the rest of the post making each one concrete:

- **Data parallelism** splits the *batch*. Every GPU holds a full copy of the model and processes a different shard of the batch. Because the copies must agree, you **all-reduce the gradients** once per step. This is the workhorse — reach for it first, always.
- **Tensor parallelism** splits individual *weight matrices* across GPUs. A single linear layer's matmul is carved into pieces, and the pieces are stitched back together with an **all-reduce inside every layer**, in both the forward and the backward pass. It is bandwidth-hungry and wants a fat interconnect.
- **Pipeline parallelism** splits the *layers* into stages, one contiguous block of layers per GPU. Activations flow forward and gradients flow back as **point-to-point handoffs between stages**, and the price is a scheduling *bubble* — idle time while the pipeline fills and drains.
- **Expert parallelism** splits the *experts* of a mixture-of-experts model. Each GPU owns a subset of experts, and every token is routed to the GPUs that hold its chosen experts with an **all-to-all**.
- **Sequence (context) parallelism** splits the *sequence* dimension. The tokens of one long example are spread across GPUs, and attention — which every token needs against every other token — is completed by **communicating the keys and values** around the group.

That is the entire vocabulary. Five axes, five collectives. The rest of large-scale training is a matter of choosing which ones to use, in what combination, and on which links. Notice the structure the diagram makes visible: the axes on the "split the batch" branch replicate the model and cost you a gradient all-reduce; the axes on the "split the model" branch replicate the batch and cost you comms *inside* the computation. That single distinction — do you replicate the model or the batch — is the first fork of every design.

A quick note on why the collective matters so much, because it is the thread that runs through the whole post. A GPU computes fast and communicates slow. An A100 does roughly 312 dense bf16 TFLOP/s of math but its fastest link to a neighbor GPU moves data at hundreds of gigabytes per second, and its link to a GPU in another node moves data at tens of gigabytes per second. So the moment your chosen axis forces a collective onto a slow link, that collective — not the math — sets your step time. Choosing a parallelism strategy is, in the end, choosing which collective you can afford on the wires you actually have. We will make that trade quantitative in Section 8. First, the router.

## 2. A first router: which lever, roughly

Before the five deep dives, here is the decision in its shortest form, because most of the time you do not need all five — you need to identify the *one* wall you have hit and pull the matching lever. Figure 2 is that decision as a short tree, and it is worth committing to memory because it collapses a scary design space into three or four yes/no questions.

![A decision tree that starts from whether the model fits on one GPU and routes to data parallelism, to sharding, to tensor parallelism, or to pipeline parallelism depending on what fits](/imgs/blogs/the-map-of-parallelism-2.webp)

Read it top to bottom. The very first question is the only one that matters most of the time: **does the model fit on one GPU** — parameters, gradients, optimizer states, and a step's worth of activations, all of it?

- **Yes, it fits.** Then you are done thinking about model parallelism. Use **data parallelism** (plain DDP), split the batch across your GPUs, and all-reduce the gradients. Do not add anything cleverer until DDP stops scaling. The most common mistake in this whole field is reaching for tensor or pipeline parallelism when the model fits and DDP would have saturated the interconnect just fine.
- **No, it doesn't fit — but a single layer does.** Then you do not need to cut layers apart. You need to stop *replicating* the parts of the model that don't have to be replicated. That is **sharding**: FSDP or ZeRO shards the parameters, gradients, and optimizer states across the data-parallel group so each GPU stores only its slice, gathering the full layer only for the moment it computes it. Alternatively, **pipeline parallelism** puts different layers on different GPUs so no GPU holds the whole model at once.
- **No, and a single layer is itself too big for one GPU.** Now you have no choice: you must split a matrix. That is **tensor parallelism** — carve the weight matrices across GPUs and all-reduce inside the layer.
- **Special cases.** A sparse mixture-of-experts model routes to **expert parallelism**; a context so long that the attention matrix alone overflows a GPU routes to **sequence parallelism**.

That router gets you 80% of the way. The remaining 20% — when you compose several axes at once because a frontier model triggers all of these at the same time — is Section 8. But you cannot compose axes you don't understand individually, so we take them one at a time, and for each we answer the same three questions: *what does it split, what does it replicate, and what does it cost?*

## 3. Data parallelism: split the batch

Data parallelism is the axis you already half-know, because it is what "just add more GPUs" means when it works. Every GPU gets a **full, identical copy of the model**. You take the global batch, cut it into equal shards, and hand one shard to each GPU. Every GPU runs the forward and backward pass on its own shard and computes gradients — but those gradients are only correct for its slice of the data. To take a single optimizer step that reflects the *whole* batch, all the copies must average their gradients so they stay bit-for-bit identical. That average is an **all-reduce**, and it is the entire communication cost of data parallelism: one all-reduce of the gradient tensor, once per step.

What it splits: the batch. What it replicates: the model — every parameter, on every GPU. What it costs: an all-reduce of the gradients each step.

Here is the minimal PyTorch that expresses it. This is the code you should reach for first for any model that fits.

```python
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

def main():
    # torchrun sets RANK, LOCAL_RANK, and WORLD_SIZE for every process it spawns.
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)

    model = build_model().to(local_rank)
    # DDP registers backward hooks that all-reduce each gradient bucket
    # the instant it is ready, overlapping comms with the rest of backward.
    model = DDP(model, device_ids=[local_rank], gradient_as_bucket_view=True)

    # Every rank must see a DIFFERENT, non-overlapping shard of the data.
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=32, sampler=sampler,
                        num_workers=4, pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # reshuffle consistently across ranks each epoch
        for x, y in loader:
            x, y = x.to(local_rank), y.to(local_rank)
            loss = model(x, y)
            loss.backward()            # gradients all-reduced here, overlapped
            opt.step()
            opt.zero_grad(set_to_none=True)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

You launch it with `torchrun`, which spawns one process per GPU and wires up the rendezvous. Terms, defined once, since the whole series leans on them: a **rank** is the global index of a process, `world_size` is how many there are in total, and `local_rank` is the index *within a node* (which physical GPU to use). A **collective** like all-reduce is a communication primitive that every rank in a group participates in together — if one rank skips it, the others hang forever waiting.

```bash
# One node, 8 GPUs.
torchrun --standalone --nproc_per_node=8 train.py

# Two nodes, 8 GPUs each (16 ranks total). Run on each node with its own --node_rank.
torchrun \
  --nnodes=2 --nproc_per_node=8 --node_rank=0 \
  --rdzv_backend=c10d --rdzv_endpoint=host0:29500 \
  train.py
```

The one line worth staring at is `loss.backward()`, because that is where the magic and the cost both live. DDP does not wait until the whole backward pass finishes and then send one giant gradient. It groups parameters into *buckets* and fires the all-reduce for a bucket the moment its gradients are ready, while the backward pass is still computing gradients for earlier layers. This **overlap of communication with computation** is the reason DDP scales at all — the gradient sync hides underneath the math instead of adding to it. When you see DDP get 90%+ efficiency, that overlap is why; when you see it fall apart, broken overlap is usually the reason. The mechanics of bucketing, overlap, and the `no_sync()` accumulation path are the whole subject of the dedicated [DDP from first principles](/blog/machine-learning/distributed-training/ddp-from-first-principles) post.

### The mechanism: what one all-reduce actually costs

Let the gradient tensor be `S` bytes and let there be `N` GPUs. The all-reduce that DDP performs is, on modern NCCL, a **ring all-reduce**, and its cost is not `N` times anything — it is beautifully independent of `N` in bandwidth terms. The ring algorithm runs in two phases, a reduce-scatter and an all-gather, and in each phase every GPU sends and receives `(N-1)/N` of the data. So each GPU moves a total of

$$ V = 2 \cdot \frac{N-1}{N} \cdot S \text{ bytes} $$

and the time is that volume over the link bandwidth `B`, roughly $T = 2(N-1)/N \cdot S / B$. For a 1.3-billion-parameter model in bf16 the gradient is about 2.6 GB, so at large `N` each GPU shoves roughly `2 * 2.6 = 5.2` GB across its link every step. On an intra-node NVLink fabric at hundreds of GB/s that is a few milliseconds and hides under the backward pass. Across nodes on InfiniBand at tens of GB/s it is an order of magnitude slower — which is exactly why data-parallel scaling *within* a node looks near-perfect and *across* nodes starts to sag. The derivation of the ring, and why $2(N-1)/N \cdot S$ is bandwidth-optimal, is done properly in [collectives from scratch](/blog/machine-learning/distributed-training/why-distributed-training); here the takeaway is only this: data parallelism's cost is one all-reduce whose per-GPU volume flattens with scale but whose *wall-clock* is set by the slowest link it has to cross.

#### Worked example: 1.3B model, 1 to 64 GPUs

Take a 1.3B-parameter Transformer in bf16 on A100 80GB cards. The whole model — weights, gradients, and Adam states — is about 21 GB, so it fits comfortably on one card, which makes this a clean data-parallelism story. Here are representative (illustrative, not a published benchmark) throughput numbers as you scale:

| GPUs | Fabric | Tokens/s | Scaling efficiency | Notes |
| --- | --- | --- | --- | --- |
| 1 | — | 12.0k | baseline | single-card reference |
| 8 | 1 node, NVLink | 90k | 94% | all-reduce hides under backward |
| 64 | 8 nodes, InfiniBand | 560k | 73% | cross-node all-reduce starts to show |

Scaling efficiency is just $E_N = (\text{tokens/s at } N) / (N \cdot \text{tokens/s at } 1)$. The 94% inside a node says NVLink is fast enough that the gradient all-reduce almost entirely overlaps the backward pass. The drop to 73% across eight nodes is the InfiniBand all-reduce poking out from under the compute — the same $2(N-1)/N \cdot S$ volume, now on a link an order of magnitude slower. That sag is the signal that you are approaching the limit of *pure* data parallelism, and it is the doorway to everything else in this post.

### How to measure scaling without fooling yourself

Those efficiency numbers are only worth anything if you measure them correctly, and the number of scaling reports that are quietly wrong is high, because a GPU makes it easy to time the wrong thing. Four disciplines separate a real scaling number from a fantasy:

- **Synchronize before you stop the clock.** CUDA kernels are asynchronous — `loss.backward()` returns to Python before the GPU has finished the work. If you time a step with a bare wall clock you measure kernel-launch latency, not compute. Call `torch.cuda.synchronize()` immediately before reading the timer, or use CUDA events, or you are benchmarking the Python interpreter and calling it throughput.
- **Throw away the warm-up.** The first steps of any run are unrepresentative: cuDNN is still autotuning kernels, the caching allocator is still growing its memory pool, and NCCL is still establishing its rings and buffers. Discard the first twenty to fifty steps and average over a few hundred, or your "throughput" is dominated by one-time costs that never recur.
- **Profile the data loader before you blame the fabric.** The single most common cause of "adding GPUs didn't help" is not the network — it is an input pipeline that cannot feed the GPUs. If `num_workers` is too low or preprocessing is CPU-bound, every GPU stalls waiting for the next batch, and the gradient-overlap machinery is irrelevant because the GPU was idle regardless. This confound is subtle because it looks exactly like a comms problem in aggregate metrics; the data pipeline at scale is its own discipline, covered later in the series.
- **Beware thermal and clock throttling.** A GPU under sustained load can down-clock, so a ten-second benchmark may report a throughput the full run never sustains. Measure long enough for clocks to settle, and cross-check tokens/s against MFU — if your MFU comes out implausibly high, you are almost certainly measuring a burst, not a steady state.

The honest north-star metric under all of this is **MFU (model FLOPs utilization)**: the ratio of the useful floating-point work your model performs to what the hardware could theoretically do. It is honest precisely because it resists gaming — a bigger batch or a warm cache inflates tokens/s but not MFU. For a dense Transformer it is roughly `(6 * params * tokens_per_s) / (peak_flops * num_gpus)`, and its behavior as you scale is the single most diagnostic number you have: if MFU falls while you add GPUs, communication is eating your gains, and no amount of raw tokens/s should reassure you.

Data parallelism has exactly one hard limit: **the model, its gradients, and its optimizer states must all fit on one GPU**, because every GPU holds a full copy. And that limit arrives sooner than people expect. With mixed-precision Adam, each parameter costs about 16 bytes of resident state — 2 bytes for the bf16 weight, 2 for the bf16 gradient, and 12 for the fp32 master weight plus the two fp32 optimizer moments. That is the famous `(2 + 2 + 12)Ψ` accounting, and it means a 7B model needs roughly 112 GB of state before you have stored a single activation — already over an 80GB card. So the very first thing that breaks as models grow is not speed; it is the *replication* in data parallelism. The fix is to stop replicating what you don't have to, which is sharding — the subject of the next posts on [ZeRO and FSDP](/blog/machine-learning/distributed-training/why-distributed-training) — or to stop putting the whole model on one GPU at all, which is model parallelism, where we go next.

## 4. Tensor parallelism: split the matrices

When a single layer is too big to fit — or when its matmul is simply too slow on one GPU and you want more math throughput per layer — you split the layer itself. Tensor parallelism (also called intra-layer model parallelism, and the technique Megatron-LM made standard) carves the individual **weight matrices** across GPUs so that a matrix multiply that was one big GEMM becomes several smaller GEMMs running in parallel, then stitched back together.

What it splits: the weight matrices, along one dimension. What it replicates: the batch — every GPU in the tensor-parallel group sees the same tokens. What it costs: an all-reduce *inside every layer*, on both the forward and the backward pass.

The trick that makes it efficient is choosing *how* to split so that consecutive layers cancel out each other's communication. A Transformer block is essentially two matmuls back to back (an up-projection then a down-projection in the MLP; the analogous pair in attention). Megatron splits the first one **column-wise** — each GPU computes a slice of the output features with no communication at all — and the second one **row-wise** — each GPU computes a partial sum over its slice of the input features, and a single all-reduce adds the partials back together. One all-reduce per matmul pair in the forward, one in the backward. Here is the shape of it:

```python
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

# Column-parallel: each rank owns a slice of the OUTPUT features.
# Forward needs NO communication; the slices are independent.
class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, tp_group):
        super().__init__()
        self.tp_group = tp_group
        world = dist.get_world_size(tp_group)
        self.weight = nn.Parameter(torch.empty(out_features // world, in_features))

    def forward(self, x):
        return F.linear(x, self.weight)          # local matmul, no comms

# Row-parallel: each rank owns a slice of the INPUT features and produces a
# PARTIAL sum. One all-reduce completes the result -- this is the price of TP.
class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, tp_group):
        super().__init__()
        self.tp_group = tp_group
        world = dist.get_world_size(tp_group)
        self.weight = nn.Parameter(torch.empty(out_features, in_features // world))

    def forward(self, x):
        y = F.linear(x, self.weight)             # partial sum on each rank
        dist.all_reduce(y, group=self.tp_group)  # <-- the cost of tensor parallelism
        return y
```

That `all_reduce` inside `forward` is the whole story, and it is a very different animal from the data-parallel all-reduce. The data-parallel all-reduce happens **once per step** and syncs *gradients*, and it overlaps beautifully with the backward pass. The tensor-parallel all-reduce happens **twice per layer per pass** — so for an 80-layer model, hundreds of times per step — and it syncs *activations*, which sit squarely on the critical path: the next layer literally cannot start until the all-reduce that finishes this layer completes. There is far less room to hide it. That is why tensor parallelism has one iron rule: **keep it inside a single node, on NVLink.** The full mechanics of column/row splitting, the backward-pass collectives, and Megatron's sequence-parallel refinement are the subject of the dedicated [tensor parallelism with Megatron](/blog/machine-learning/distributed-training/tensor-parallelism-megatron) post; the point here is the *shape* of the cost — frequent, activation-sized, on the critical path — and the interconnect it therefore demands.

#### Worked example: is tensor parallelism worth it on this cluster?

This is the calculation that decides whether TP helps or destroys you, and it is worth doing once by hand. Take a hidden size of 8192 and a micro-batch of 4096 tokens. The activation that a row-parallel layer must all-reduce is roughly `tokens * hidden * 2` bytes in bf16 = `4096 * 8192 * 2` ≈ 67 MB. With a tensor-parallel group of 8, the ring all-reduce moves about $2 \cdot 7/8 \cdot 67 \approx 117$ MB per GPU, per all-reduce.

- **On NVLink** at ~600 GB/s: `117e6 / 600e9` ≈ **0.2 ms** per all-reduce. With two per layer in the forward pass, that is a small tax on top of the matmul, and TP buys you real headroom.
- **On PCIe** at ~32 GB/s: `117e6 / 32e9` ≈ **3.7 ms** per all-reduce — roughly *18 times* slower, and now the communication for a single layer dwarfs the matmul that layer performs. Multiply by hundreds of all-reduces per step and your "parallel" model runs slower than one GPU would have.

That factor of ~18 is the entire reason "tensor parallelism needs NVLink" is a rule and not a preference. The comms are frequent, they are on the critical path, and they are sized like activations, so the only place they hide is the fattest link you own. Put a tensor-parallel group across a slow link and you have built a machine for turning GPUs into idle heaters. Keep the group inside one node — which is exactly why tensor-parallel degree so often equals the number of GPUs per node, 8 on a DGX box — and it sings.

## 5. Pipeline parallelism: split the layers

There is a second way to model-parallelize that pays a completely different communication bill. Instead of splitting each layer across GPUs, **pipeline parallelism** splits the *layers themselves* into contiguous stages — layers 1 to 40 on GPU group A, layers 41 to 80 on group B — and streams the batch through the stages like a factory line. Stage A computes its layers on a chunk of data, hands the resulting activations to stage B, and moves on to the next chunk while B works.

What it splits: the layers, into stages. What it replicates: nothing — the stages are disjoint, which is what makes pipelining so memory-efficient. What it costs: **point-to-point handoffs** of activations between adjacent stages — a small, cheap communication compared to any all-reduce — plus a scheduling cost that has no analog in the other axes: the **bubble**.

The bubble is the idle time at the start and end of every batch. When the pipeline first fills, stage B has nothing to do until stage A produces its first output; when the batch drains, stage A is finished while B is still processing the last chunks. During fill and drain, some stages sit idle. To shrink the bubble you cut each batch into many **micro-batches** so the stages stay fed — the more micro-batches in flight, the smaller the fraction of time spent filling and draining.

### The mechanism: the bubble fraction

This is one of the cleanest laws in distributed training, and it tells you exactly when pipelining pays. With `p` pipeline stages and `m` micro-batches per batch, the pipeline needs `p - 1` steps to fill before all stages are busy and `p - 1` to drain at the end. The useful work spans `m + p - 1` time slots, of which `p - 1` are bubble. So the fraction of time wasted is

$$ \text{bubble fraction} = \frac{p - 1}{m + p - 1}. $$

Read what that equation demands. With `p = 4` stages and only `m = 4` micro-batches, the bubble is $3 / 7 \approx 43\%$ — nearly half your GPUs idle. Push to `m = 32` micro-batches and it falls to $3 / 35 \approx 8.6\%$. That is the whole reason pipeline parallelism only pays when you can afford **many micro-batches per stage**: the bubble is a fixed tax of `p - 1` slots that you amortize over `m`. Too few micro-batches and pipelining is a way to buy idle GPUs. The scheduling refinements that shrink the bubble further — GPipe versus the 1F1B schedule versus interleaved stages — are the subject of the dedicated pipeline post; the load-bearing fact here is that pipelining's cost is *idle time you control with the micro-batch count*, not bandwidth.

#### Worked example: the bubble at four stages

Say you have a model too deep for tensor parallelism alone and you split it into `p = 8` pipeline stages. If your global batch only gives you `m = 8` micro-batches, the bubble is $7 / 15 \approx 47\%$ — you are paying for 8 GPUs and getting the work of about 4. That is a terrible deal and a common one. The fix is not more GPUs; it is more micro-batches: raise `m` to 64 and the bubble drops to $7 / 71 \approx 9.9\%$. This is why pipeline parallelism lives at the *outer* levels of a large training job, where the global batch is huge and micro-batches are plentiful, and almost never as the only axis on a small run. The point-to-point activation handoff, meanwhile, is cheap enough to cross a slower link than tensor parallelism can tolerate — which is exactly why, when we compose axes, pipeline stages get to span nodes while tensor groups stay glued inside them.

## 6. Expert parallelism: split the experts

The three axes so far all assume a *dense* model — every parameter participates in every token. A **mixture-of-experts (MoE)** model breaks that assumption: it replaces the dense feed-forward block with many parallel "expert" networks and a small router that sends each token to just one or two of them. The model has enormous total parameter count but each token only touches a sliver of it, so you get the capacity of a huge model at the compute of a small one. That sparsity opens a fourth axis.

**Expert parallelism** places different experts on different GPUs — expert 0 and 1 on GPU 0, experts 2 and 3 on GPU 1, and so on. What it splits: the experts. What it replicates: the shared parts — attention and the router — which are usually data-parallel across the same GPUs. What it costs: an **all-to-all**, and this is a genuinely different collective from anything above.

Here is why all-to-all is the natural cost. After the router decides, on each GPU, which expert every local token wants, the tokens have to *travel* to the GPU that owns their expert — and tokens from every GPU are heading to experts on every other GPU, all at once. That simultaneous everyone-sends-to-everyone shuffle is precisely an all-to-all: the first one dispatches tokens to their experts, the experts compute, and a second all-to-all gathers the results back to where each token started. Two all-to-alls wrap every MoE layer.

All-to-all has a nasty failure mode that the other collectives don't: **load imbalance**. If the router sends far more tokens to the experts on GPU 3 than to those on GPU 7, then GPU 3 becomes a straggler that the whole group waits on, and GPU 7 sits idle. Real MoE training fights this with a **capacity factor** — a hard cap on how many tokens any one expert will accept — and simply **drops** the overflow tokens (they skip the expert and pass through). Set the capacity factor too low and you drop signal; too high and you waste memory and bandwidth padding for tokens that never arrive. The full treatment of routing, capacity, and the dropped-token trade-off belongs to the dedicated expert-parallelism post. For the map, the essentials are: expert parallelism splits the experts, costs a pair of all-to-alls per MoE layer, and is uniquely sensitive to how *evenly* the router spreads its load.

## 7. Sequence parallelism: split the sequence

The last axis appears when neither the parameters nor the batch is the problem — the *sequence* is. Attention costs memory that grows with the square of the sequence length, so at long enough context the attention computation for a single example overflows one GPU even though the model's weights fit fine. When the thing that won't fit is a function of sequence length, you split along the sequence.

**Sequence (or context) parallelism** spreads the tokens of one long example across GPUs — GPU 0 holds tokens 1 to 8192, GPU 1 holds 8193 to 16384, and so on. What it splits: the sequence dimension. What it replicates: the weights — every GPU has the full model, it just holds a slice of the tokens. What it costs: **communicating the keys and values**, because attention is the one operation where every token must interact with every other token, including the ones now living on a different GPU.

Two designs dominate, and they trade the same way as always — which link, which collective. **Ring Attention** streams keys and values around the group in a ring: each GPU computes attention against its local keys and values, then passes them to its neighbor and receives the next block, so over `p` steps every GPU has attended to the whole sequence without ever materializing the full attention matrix. It overlaps the key/value transfer with the attention compute, which makes it tolerant of ordinary interconnects. **Ulysses** instead uses all-to-all to re-partition the tensors so that each GPU holds all tokens for a subset of attention heads, turning the sequence split into a head split for the duration of attention. Ring favors bandwidth-overlap; Ulysses favors fewer, larger collectives. The details are the sequence-parallelism post's job; the map only needs you to recognize the trigger — *attention OOMs on context length, not on parameters* — and the cost: moving keys and values around the group.

## 8. The cheat sheet, and how to read a run in one glance

We have now walked all five axes. Figure 3 puts them side by side so you can read across any row and recite what that axis splits, replicates, costs, and demands. This is the table to screenshot.

![A comparison matrix with one row per parallelism axis and columns for what it splits, what it replicates, the communication collective it costs, the interconnect it needs, and the situation that triggers it](/imgs/blogs/the-map-of-parallelism-3.webp)

The same content in prose, because the relationships between the columns are the real lesson:

| Axis | What it splits | What it replicates | Comms per step | Interconnect need | Reach for it when |
| --- | --- | --- | --- | --- | --- |
| **Data (DP)** | the batch | the whole model | one gradient all-reduce | any; tolerant | model fits, you want throughput |
| **Tensor (TP)** | weight matrices | the batch | 2 all-reduce per layer, each pass | NVLink, intra-node | a single layer is too big or too slow |
| **Pipeline (PP)** | layers into stages | nothing | point-to-point activations | modest; tolerates IB | model is deep, many micro-batches |
| **Expert (EP)** | the experts | attention and router | two all-to-alls per MoE layer | fat all-to-all fabric | you are training a sparse MoE |
| **Sequence (SP)** | the sequence | the weights | gather keys and values | ring-friendly or IB | context OOMs attention itself |

Two patterns in this table are worth naming because they drive every real design. First, **frequency times size equals interconnect demand.** Data parallelism's all-reduce is large but *rare* (once per step) and *overlappable*, so it tolerates a slow link. Tensor parallelism's all-reduce is activation-sized but *frequent* and *on the critical path*, so it demands the fastest link you own. That single difference — rare-and-hideable versus frequent-and-exposed — is why DP crosses nodes happily and TP must not. Second, **what you replicate is what you pay to keep in sync.** DP replicates the model, so it syncs gradients; the model-splitting axes replicate the batch, so they sync activations. Knowing which thing is replicated tells you immediately which collective you will be staring at in the profiler.

## 9. Composing the axes: 3D parallelism

No frontier model is trained on a single axis. A 70B or 400B model trips *several* walls at once — a layer too big for tensor parallelism alone, a model too deep for the memory even after that, a batch you still want to spread for throughput — so real recipes **compose** two or three axes into what the field calls 3D parallelism: tensor times pipeline times data. Figure 4 shows the composition as a nesting, and the nesting order is not arbitrary — it is dictated exactly by the interconnect demands we just derived.

![A nested diagram showing sixty-four GPUs organized as four data-parallel replicas, each replica split into two pipeline stages, and each stage running eight-way tensor parallelism inside a single node](/imgs/blogs/the-map-of-parallelism-4.webp)

The rule for *where each axis goes on the cluster* falls straight out of the cheat sheet: put the chattiest axis on the fastest link, and the most tolerant axis on the slowest. Figure 5 stacks the axes by exactly that ranking.

![A layered stack ordering the parallelism axes by communication intensity from tensor parallelism on the fastest link down to data parallelism on the most tolerant one](/imgs/blogs/the-map-of-parallelism-5.webp)

Reading the stack top to bottom: **tensor parallelism** is the innermost ring, glued to NVLink inside a single node, because its all-reduce is frequent and on the critical path. **Pipeline parallelism** sits one level out — its point-to-point activation handoffs are cheap enough to cross the InfiniBand links *between* nodes. **Data parallelism** is the outermost ring, spanning the whole cluster, because its once-per-step gradient all-reduce is the easiest cost to overlap and can tolerate the slowest fabric. Expert and sequence parallelism slot in by the same logic: expert parallelism wants a fat all-to-all fabric near the top; sequence parallelism's ring rides the local links. You are, in every case, matching a collective to the link that can afford it.

To put real numbers on why that ordering is forced rather than chosen, here are the links the stack refers to:

| Link | Bandwidth (approx) | Scope | Axis it can carry |
| --- | --- | --- | --- |
| NVLink 3 / 4, NVSwitch | ~600–900 GB/s aggregate per GPU | inside a node | tensor parallelism |
| PCIe Gen4 x16 | ~32 GB/s per direction | inside a node, no NVLink | at most data parallelism |
| InfiniBand HDR | ~25 GB/s (200 Gb/s) per link | between nodes | pipeline, data parallelism |

The two orders of magnitude between the top row and the bottom are the entire reason the axes stack the way they do. A collective that is a rounding error on NVLink becomes the dominant cost on InfiniBand, so the placement is not a style choice — it is the only arrangement that keeps every collective on a link fast enough to hide it. The physics behind these numbers — why NVLink is a switched on-package fabric while InfiniBand needs RDMA to move bytes without stealing CPU cycles — is the subject of the [interconnects and RDMA deep-dive](/blog/machine-learning/high-performance-computing/interconnects-nvlink-nvswitch-infiniband-and-rdma).

To make the composition concrete in code, PyTorch gives you a **device mesh** — a named grid that lets you address the same set of GPUs along multiple axes at once. Figure 6 shows the smallest useful mesh, a 2D grid of data-parallel by tensor-parallel.

![A grid of GPUs where each row is a tensor-parallel group that together holds one model copy and each column is a data-parallel group that averages gradients](/imgs/blogs/the-map-of-parallelism-6.webp)

Read the mesh both ways, because that double reading is the entire point of a device mesh. Along a **row**, the GPUs form a tensor-parallel group: they collectively hold one copy of the model and all-reduce activations inside each layer. Down a **column**, the GPUs form a data-parallel group: they hold the same model shard on different data and all-reduce gradients once per step. The mesh names both groupings so your framework can issue each collective on the right sub-group automatically. Here is how you build and use one:

```python
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    parallelize_module, ColwiseParallel, RowwiseParallel,
)
from torch.distributed.fsdp import fully_shard

# A 2D mesh: 4-way data parallel x 8-way tensor parallel = 32 GPUs.
mesh = init_device_mesh("cuda", (4, 8), mesh_dim_names=("dp", "tp"))
tp_mesh = mesh["tp"]   # the 8 GPUs that together hold one model copy
dp_mesh = mesh["dp"]   # the 4 GPUs holding the same shard on different data

# Tensor-parallelize each block over the tp sub-mesh: column-parallel then
# row-parallel, so consecutive matmuls cancel each other's communication.
parallelize_module(model, tp_mesh, {
    "attn.qkv": ColwiseParallel(),
    "attn.out": RowwiseParallel(),
    "mlp.up":   ColwiseParallel(),
    "mlp.down": RowwiseParallel(),
})

# ...and shard the parameters over the dp sub-mesh with FSDP (ZeRO-style),
# so no GPU stores a full copy of even its own tensor-parallel shard.
for block in model.transformer_blocks:
    fully_shard(block, mesh=dp_mesh)
fully_shard(model, mesh=dp_mesh)
```

That single mesh expresses two axes at once, and each collective lands on the sub-group that can afford it: the tensor-parallel all-reduce stays inside the 8-GPU NVLink group, and the data-parallel gradient sync spans the 4 replicas across the slower fabric. FSDP itself is the sharded form of data parallelism — it splits parameters, gradients, and optimizer states across the DP group so replication stops being the memory wall. Its mechanics, and how it composes with tensor parallelism through exactly this mesh, are the subject of the [3D parallelism deep-dive](/blog/machine-learning/distributed-training/3d-parallelism) and the DeepSpeed and Megatron internals in the [ZeRO and 3D-parallelism library post](/blog/machine-learning/open-source-library/deepspeed-zero-3d-parallelism-deep-dive).

If you drive the composition through a framework like Megatron-LM instead of hand-rolling the mesh, the same three axes are just launch flags:

```bash
# Megatron-LM 3D parallelism for a 70B-class model on 8 nodes x 8 GPUs = 64.
# TP=8 stays inside each node; PP=2 spans nodes; DP is inferred: 64/(8*2)=4.
torchrun --nproc_per_node=8 --nnodes=8 pretrain_gpt.py \
  --tensor-model-parallel-size 8 \
  --pipeline-model-parallel-size 2 \
  --num-layers 80 --hidden-size 8192 --num-attention-heads 64 \
  --micro-batch-size 1 --global-batch-size 1024 \
  --sequence-parallel --use-distributed-optimizer \
  --bf16
```

One more axis often joins this composition, and it is worth naming so you recognize it when a config sprouts a fourth degree. Even after TP, PP, and DP, the **activations** — the intermediate tensors a forward pass must keep around for the backward pass — can dominate memory at long sequence lengths, because attention's memory grows with the square of the context. That is exactly the trigger for sequence parallelism from Section 7, and Megatron's `--sequence-parallel` flag in the launch above is doing precisely this: splitting the parts of each layer that operate along the sequence across the same tensor-parallel group, so the activation memory drops without adding a new collective on a slow link. The result is sometimes called 4D parallelism, but it is not a new idea — it is the same discipline applied a fourth time: find the dimension that overflows, split it, and pay its collective on a link that can afford it.

#### Worked example: sizing TP by PP by DP for a 70B on 64 GPUs

Now the payoff — turning the map into an actual layout. You have a 70B-parameter model and 64 A100 80GB GPUs, arranged as 8 nodes of 8 GPUs each, NVLink inside a node and InfiniBand between nodes. Why does a plausible layout come out to **TP=8, PP=2, DP=4**? Reason through it axis by axis, and watch each number get forced by the physics, not chosen by taste.

Start with the memory problem. A 70B model needs about 140 GB just for bf16 weights, and with Adam states the `(2 + 2 + 12)Ψ` accounting pushes the full training state to roughly 1.1 TB. No single 80GB card comes close. Figure 7 is the whole predicament and its resolution on one page.

![A before and after comparison showing a seventy billion parameter model that cannot fit when each GPU holds a full copy, then fits and stays busy when the model is split sixteen ways across each replica](/imgs/blogs/the-map-of-parallelism-7.webp)

Now size each axis:

- **Tensor parallelism first, and pin it to the node.** The model's individual layers, at hidden size 8192, are large enough that splitting them helps both memory and per-layer math throughput. Because TP's all-reduce is frequent and critical-path, it must live on NVLink, so **TP = 8** — exactly one node. This is the single most important sizing rule in the whole example: tensor-parallel degree equals GPUs-per-node, so the chattiest collective never leaves the fastest link.
- **Pipeline parallelism next, across a few nodes.** Even split 8 ways, the 80-layer model is still too much state for one node's worth of memory once you add activations, so cut the layers into stages. Its cheap point-to-point handoffs tolerate the inter-node InfiniBand, so **PP = 2** — layers 1 to 40 on one node, 41 to 80 on the next. One model replica now spans 2 nodes = 16 GPUs and holds the whole model, split 16 ways.
- **Data parallelism fills the rest.** You have 64 GPUs and each replica uses 16, so you have room for `64 / 16 = 4` replicas. **DP = 4**, spanning the cluster, each replica chewing a different shard of the batch and syncing gradients with the tolerant once-per-step all-reduce.

Check it: `TP * PP * DP = 8 * 2 * 4 = 64`. Each GPU holds about `1/16` of the model's parameters and their optimizer states, so instead of an impossible 1.1 TB per card you are down to something that fits — and with the whole thing tuned, a 70B run on this shape reaches roughly the 40–55% MFU (model FLOPs utilization) band that well-run large training achieves. The layout was not a guess. Every dimension was set by matching a collective to the link that can afford it: TP on NVLink because its comms are frequent, PP across nodes because its comms are cheap, DP across the cluster because its comms are rare and hideable. That is the map paying off.

Stress-test the choice, because the whole point of a map is that it tells you what changes when the terrain does:

- **On PCIe instead of NVLink?** TP=8 collapses — recall the ~18x slowdown from the earlier worked example. You would drop to TP=1 or TP=2 and lean far harder on pipeline and sharded-data parallelism.
- **With a tiny batch?** PP=2 is fine, but if you pushed to PP=8 with a small global batch, the bubble fraction $(p-1)/(m+p-1)$ would eat you alive — you'd need many micro-batches to justify the stages.
- **When one node is a straggler?** DP's synchronous all-reduce means the slowest replica sets the pace, so a single sick node halves your throughput — a failure mode the [debugging DDP and multi-GPU](/blog/machine-learning/debugging-training/debugging-ddp-and-multi-gpu) post is entirely about.
- **When the optimizer state still won't fit?** Add ZeRO-style sharding *on top of* the DP axis (offloading optimizer states, or sharding them across the 4 replicas), which is exactly why the code above shards with FSDP over the `dp` sub-mesh rather than plain-replicating.

## 10. Case studies and real numbers

The map is not folklore; it is how the largest published training runs were actually laid out. A few anchors, cited so you can check them and framed as approximate where the sources are.

**Megatron-LM's 3D parallelism.** Narayanan et al. (2021), "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM," is the canonical demonstration of the composition in Section 9. They trained models up to a trillion parameters on 3072 A100 GPUs, sustaining about 502 petaFLOP/s — roughly 52% of theoretical peak, or about 163 of the A100's 312 bf16 TFLOP/s per GPU. The layout is exactly the one we derived: tensor parallelism kept *inside* each 8-GPU node on NVLink, pipeline parallelism *across* nodes, and data parallelism on the outside. The paper's own ablations show tensor parallelism falling off a cliff the moment it is forced across the slower inter-node links — the empirical version of our PCIe worked example.

**GPT-3 and the cost of getting the axes wrong.** The original GPT-3 175B training (Brown et al., 2020) is widely cited as achieving only low-20s-percent hardware utilization on its cluster, and a large part of the gap versus later runs is parallelism layout — the Megatron line of work roughly doubled effective utilization on comparable hardware largely by placing collectives on the right links. Same GPUs, same model size, very different throughput, because of the map.

**PaLM 540B.** Chowdhery et al. (2022) reported about 46.2% model FLOPs utilization training PaLM on 6144 TPU v4 chips, a number they highlight precisely because sustaining high MFU at that scale is *hard* and depends on getting the parallelism and the collectives right. It is a useful sanity anchor: when someone quotes 45–55% MFU on a giant model, they are in the neighborhood of the best published dense runs, not exaggerating.

**Mixture-of-experts and all-to-all.** GShard (Lepikhin et al., 2020) and Switch Transformer (Fedus et al., 2021) are the reference points for expert parallelism at scale, and both papers spend real ink on the all-to-all cost and on load balancing and dropped tokens — the exact failure mode Section 6 flagged. They are the proof that the fourth axis is not exotic: it is how models with hundreds of billions to trillions of sparse parameters are trained at the compute of far smaller dense ones.

The through-line across all four is the thesis of this whole post: throughput at scale is not mostly about the GPUs. It is about matching each axis's collective to a link that can afford it. Get that right and you sit in the 40–55% MFU band; get it wrong and you buy a cluster to run at the speed of a workstation.

## 11. When to reach for each axis, and when not to

Every axis is a cost, and the discipline is refusing to pay a cost you don't owe. The decisions, stated bluntly:

- **Reach for data parallelism first, always.** If the model, gradients, and optimizer states fit on one GPU, DDP is the answer, full stop. Do not add tensor or pipeline parallelism to a model that fits — you would be paying critical-path all-reduces or a pipeline bubble to solve a problem you don't have, and DDP would have saturated your interconnect anyway.
- **Reach for sharding (FSDP/ZeRO) before model-splitting.** When the model stops fitting, the first move is usually not to cut layers apart but to stop *replicating* the optimizer states across the data-parallel group. Sharding often buys you the room you need with far less complexity than tensor or pipeline parallelism, because it keeps the simple data-parallel dataflow and only changes what each rank stores.
- **Reach for tensor parallelism only when a single layer is the problem, and only on NVLink.** It is the highest-communication axis, and its comms are on the critical path. If you cannot keep the group inside one node, do not use it — the ~18x penalty from crossing to PCIe or IB is not a tax, it is a wall. Tensor-parallel degree greater than the GPUs-per-node is almost always a mistake.
- **Reach for pipeline parallelism when the model is deep and the batch is large.** It is memory-efficient and link-tolerant, but the bubble makes it worthless without many micro-batches. Below a healthy micro-batch count, pipelining just buys idle GPUs; the `(p-1)/(m+p-1)` law is not optional reading.
- **Reach for expert parallelism only if the model is actually sparse (MoE).** It is not a general scaling tool; it is the native parallelism of a specific architecture, and it drags in all-to-all and load-balancing headaches you do not want unless the model demands them.
- **Reach for sequence parallelism only when context length is the thing that OOMs.** If your attention fits, you do not need it. When it doesn't fit, it is the *only* axis that helps, because none of the others touch the sequence dimension.
- **Do not go multi-node until you have saturated one node.** Almost every "eight GPUs barely beat one" story is a within-node problem — a broken overlap, a data-loader starving the GPUs, a tensor-parallel group where it shouldn't be. Fix single-node efficiency first; the cross-node fabric only makes a badly-laid-out job slower.

The meta-rule under all of these: **add an axis only when a specific wall forces it, and place it on the link its collective can afford.** The [decision framework post](/blog/machine-learning/distributed-training/the-distributed-training-playbook) turns this into a full flowchart; the map above is the compressed version you can hold in your head.

## 12. Key takeaways

- **Every distributed-training decision answers one question: which axis of the work do you split, and which do you replicate?** There are exactly five axes, and each bills you in a different collective.
- **Data parallelism** splits the batch, replicates the model, and costs one gradient all-reduce per step. It is the workhorse; use it first and until it stops scaling.
- **Tensor parallelism** splits weight matrices, replicates the batch, and costs an all-reduce inside every layer on the critical path. It demands NVLink and must stay inside a node.
- **Pipeline parallelism** splits layers into stages, replicates nothing, and costs cheap point-to-point handoffs plus a bubble of `(p-1)/(m+p-1)` that only many micro-batches can shrink.
- **Expert parallelism** splits the experts of an MoE and costs a pair of all-to-alls per layer, uniquely sensitive to router load balance. **Sequence parallelism** splits the sequence and costs the communication of keys and values; it is the only axis that helps when context length is what OOMs.
- **The interconnect decides everything.** Frequency times size equals interconnect demand: rare, overlappable collectives (DP) tolerate slow links; frequent, critical-path collectives (TP) require the fastest link you own.
- **Real large models compose axes** — TP times PP times DP — and the composition order is forced by the links: TP on NVLink inside a node, PP across nodes, DP across the cluster. A 70B on 64 GPUs lands naturally at TP=8, PP=2, DP=4.
- **Add an axis only when a wall forces it.** The commonest expensive mistake is model-parallelizing a model that fits, where plain data parallelism would have done the job.

## 13. Further reading

- [Why distributed training](/blog/machine-learning/distributed-training/why-distributed-training) — the series intro and the four-walls frame this map hangs on.
- [DDP from first principles](/blog/machine-learning/distributed-training/ddp-from-first-principles) — the gradient all-reduce, bucketing, and compute/comms overlap in full.
- [Tensor parallelism with Megatron](/blog/machine-learning/distributed-training/tensor-parallelism-megatron) — column/row splitting and the in-layer all-reduce, derived.
- [3D parallelism](/blog/machine-learning/distributed-training/3d-parallelism) — composing TP, PP, and DP with a device mesh, in depth.
- [The distributed-training playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook) — the capstone decision and debugging checklist for the whole series.
- [DeepSpeed ZeRO and 3D-parallelism deep-dive](/blog/machine-learning/open-source-library/deepspeed-zero-3d-parallelism-deep-dive) — the framework internals behind sharded data parallelism.
- Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism" (2019), and Narayanan et al., "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM" (2021) — the reference for tensor and 3D parallelism.
- Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (2020); Huang et al., "GPipe" (2019); Fedus et al., "Switch Transformer" (2021) — the primary sources for sharding, pipelining, and expert parallelism.
