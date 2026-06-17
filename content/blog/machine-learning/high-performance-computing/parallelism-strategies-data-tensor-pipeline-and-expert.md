---
title: "Parallelism Strategies: Data, Tensor, Pipeline, and Expert"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "A practitioner's map of every way to split a model across GPUs — data, tensor, pipeline, sequence, and expert parallelism — with the math, the code, and the measured scaling numbers to choose well."
tags:
  [
    "high-performance-computing",
    "gpu",
    "distributed-training",
    "model-parallelism",
    "tensor-parallelism",
    "pipeline-parallelism",
    "mixture-of-experts",
    "deep-learning",
    "ml-systems",
    "megatron",
  ]
category: "machine-learning"
subcategory: "High Performance Computing"
author: "Hiep Tran"
featured: true
readTime: 53
image: "/imgs/blogs/parallelism-strategies-data-tensor-pipeline-and-expert-1.png"
---

You add a second GPU to your training run, expecting it to finish in half the time. It finishes in 0.9× the time. You add eight, hoping for an 8× speedup, and you get 5.5× on a good day and 3× on a bad one. Somewhere a colleague is training a 70-billion-parameter model on 512 GPUs and reporting that each GPU is doing 50% of the work it theoretically could, and you cannot even figure out why your two GPUs are slower than one. This is the moment most AI engineers meet distributed training, and it is almost always a disappointment, because nobody told them that putting a model on many GPUs is not one technique — it is at least five, each solving a different problem, each with its own communication bill, and each catastrophic if you pick the wrong one for your situation.

This post is the map. By the end you will be able to look at a model — its parameter count, its activation memory, the GPUs you have, and the wires between them — and say with confidence: this one is pure data parallelism; this one needs tensor parallelism inside the node and pipeline parallelism across nodes; this one is a mixture-of-experts and wants expert parallelism with an all-to-all you had better have InfiniBand for. We will build each strategy from a single intuition and a single number before any formula appears, derive the costs that actually matter (the all-reduce volume, the two collectives per tensor-parallel layer, the pipeline **bubble** fraction), write the code in real PyTorch and Megatron-style sketches, and ground every claim in measured results from the papers that introduced these ideas: Megatron-LM, GPipe, GShard, and Switch Transformer.

The whole thing fits one frame, which is the spine of this entire series: training large models is a fight against three walls — **compute**, **memory bandwidth**, and **communication**. On a single GPU you fight the first two with precision, kernels, and fusion (covered in [why HPC is the bottleneck for modern AI](/blog/machine-learning/high-performance-computing/why-hpc-is-the-bottleneck-for-modern-ai) and [the memory hierarchy from registers to HBM](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm)). Parallelism is what happens when you cannot win on one GPU anymore and the third wall — communication — becomes the thing that decides whether your cluster is fast or just expensive. Everything below is about spending the least communication for the memory and compute you need.

![A decision tree that starts by asking whether a model fits on one GPU and escalates from data parallelism to tensor, pipeline, and expert parallelism as memory pressure grows](/imgs/blogs/parallelism-strategies-data-tensor-pipeline-and-expert-1.png)

Read that tree top to bottom, because the order is the whole strategy. The first question is never "which fancy parallelism should I use" — it is "does the model even fit on one GPU?" If it fits, you are done thinking about model splitting; you replicate and scale the batch, and the only cost is averaging gradients. Only when the model busts the 80 GB of an A100 or H100 do you start splitting the model itself, and even then you escalate carefully: tensor parallelism first (cheap within a node), then pipeline parallelism (cheap across nodes), then expert parallelism (only if you built a mixture-of-experts). We will walk this tree one branch at a time.

## The one running example: a Transformer from 1 to many GPUs

Let me fix a concrete model so every number below means something. Take a decoder-only Transformer with hidden size $d = 4096$, $L = 32$ layers, a vocabulary of 50,000, and a sequence length of 2048 — roughly a 6.7-billion-parameter model, the size of the original LLaMA-7B and GPT-J's bigger cousins. In bf16 (2 bytes per number) the parameters alone are about $6.7 \times 10^9 \times 2 = 13.4$ GB. That fits on one A100 80GB with room to spare, so you might think one GPU is fine. It is not, and the reason is the rest of the memory budget.

Training memory is not just parameters. With the Adam optimizer you also store, for every parameter: a gradient (2 bytes in bf16), a first-moment estimate, and a second-moment estimate (typically 4 bytes each in fp32 for stability), plus a fp32 master copy of the weights (4 bytes). That is the classic ZeRO accounting: roughly $2\Psi$ bytes for bf16 params, $2\Psi$ for grads, and $12\Psi$ for the fp32 master weight plus the two Adam moments, where $\Psi$ is the parameter count. For our 6.7B model: $16 \times 6.7 = 107$ GB of optimizer-and-weight state before a single activation is stored. That already does not fit on 80 GB. Add activation memory — the intermediate tensors you must keep for the backward pass, which scale with batch size times sequence length times hidden size times layers — and you are far over. We cover the memory side in depth in [memory optimization: ZeRO, FSDP, activation checkpointing, and offload](/blog/machine-learning/high-performance-computing/memory-optimization-zero-fsdp-activation-checkpointing-and-offload); here the point is simpler.

Here is the full budget laid out, because seeing it once makes every later decision obvious. For our 6.7B model in bf16 mixed-precision training with Adam:

| Memory component | Bytes per parameter | Total for 6.7B | What it is |
| --- | --- | --- | --- |
| bf16 parameters | 2 | 13.4 GB | the weights you compute with |
| bf16 gradients | 2 | 13.4 GB | one per parameter, per step |
| fp32 master weights | 4 | 26.8 GB | the precise copy the optimizer updates |
| Adam first moment (fp32) | 4 | 26.8 GB | the running mean of gradients |
| Adam second moment (fp32) | 4 | 26.8 GB | the running variance of gradients |
| **Subtotal (state)** | **16** | **107 GB** | fixed, before any activation |
| Activations (varies) | — | tens of GB | scales with batch × seq × layers |

Read the subtotal: 107 GB of fixed state on a GPU that has 80 GB, and we have not stored a single activation. That is the number that forces distribution. It also tells you *which* parallelism to reach for first — the state is dominated by the optimizer, so sharding the optimizer (FSDP/ZeRO) across replicas is often the lightest fix, while splitting the model itself (tensor/pipeline) is what you need when even the bf16 parameters do not fit.

So even a "7B" model — which sounds small — does not train comfortably on one GPU once you count the optimizer. This is why parallelism is not a frontier-lab problem. The instant you do real training of anything past a couple of billion parameters, you are distributing. Let me define the five strategies in one breath, then spend the rest of the post on each:

- **Data parallelism** — replicate the whole model on every GPU, give each a different slice of the batch, and average gradients with an **all-reduce**. Splits work, not the model.
- **Tensor parallelism** — split a single matrix multiply across GPUs (the Megatron column/row split), reconstruct the result with an all-reduce. Splits the model inside a layer.
- **Pipeline parallelism** — put different layers on different GPUs, stream **micro-batches** through, and eat a startup **bubble**. Splits the model across layers.
- **Sequence / context parallelism** — split the sequence dimension so each GPU holds part of a very long context. Splits the activations along time.
- **Expert parallelism** — in a **mixture-of-experts** (MoE), route each token to a few experts that live on different GPUs via all-to-all. Splits a sparse layer across experts.

A quick vocabulary pass so nothing below is mysterious. An **all-reduce** is a collective operation where every GPU contributes a tensor and every GPU ends up holding the sum (or average) of all of them — it is how replicas agree on a gradient. A **micro-batch** is a sub-slice of your batch that you push through a pipeline so the stages are not all idle at once. The **bubble** is the wasted time at the start and end of a pipeline step when some stages have no work. A **device mesh** is the logical N-dimensional grid you arrange your GPUs into so each collective knows exactly which group of GPUs it talks to. And **MoE**, mixture-of-experts, is a layer that holds many parallel feed-forward sub-networks ("experts") but activates only a few per token. Keep these five words handy.

## Data parallelism: replicate, split the batch, all-reduce

Start with the one everybody reaches for first, because it is the simplest and, for most models, the right answer. **Data parallelism** keeps the model whole. Every GPU holds an identical, complete replica of the model. You take your global batch — say 256 samples — and hand each of your 4 GPUs a different 64. Each GPU runs a full forward and backward pass on its own 64 samples, producing a full set of gradients. Then comes the only coordination step: an all-reduce that averages those 4 gradient sets into one, which every GPU applies. After the step, all four replicas are byte-for-byte identical again, ready for the next batch.

![A dataflow diagram showing a global batch split across four full model replicas that each compute gradients and then average them with one all-reduce so every replica stays identical](/imgs/blogs/parallelism-strategies-data-tensor-pipeline-and-expert-2.png)

Here is the intuition before the math: because every replica trains on different data, they compute different gradients, and you want all of them to take the *same* optimizer step — the step they would have taken if one giant GPU had seen all 256 samples. Averaging the gradients gives you exactly that. The all-reduce is the price of staying in sync. And critically, that price depends only on the size of the model, not on the size of the batch — you are exchanging gradients, and there is exactly one gradient per parameter.

Now the number that governs everything. Per training step, a ring all-reduce of a gradient tensor of $\Psi$ parameters moves about $2\Psi$ elements' worth of data per GPU on the wire (the ring algorithm sends roughly $2(N-1)/N \cdot S$ bytes, which approaches $2S$ for large $N$, where $S$ is the tensor size in bytes). For our 6.7B model in bf16, $S = 13.4$ GB, so each GPU pushes and pulls on the order of $2 \times 13.4 = 26.8$ GB of gradient traffic every step. That is the data-parallel communication bill: **roughly $2 \cdot \text{params} \cdot \text{bytes}$ per step**, independent of how big your batch is. We derive the ring all-reduce byte volume carefully in [collective communication and NCCL: all-reduce from scratch](/blog/machine-learning/high-performance-computing/collective-communication-and-nccl-all-reduce-from-scratch); here the consequence is what matters.

Why does this make data parallelism so forgiving of slow links? Because that $26.8$ GB is fixed and, crucially, it **overlaps** with computation. Modern frameworks fire the all-reduce for each layer's gradients the moment that layer's backward pass finishes, while the next layer is still computing. If your backward pass takes 200 ms and your all-reduce takes 150 ms, the all-reduce hides entirely behind compute and you pay almost nothing. This is why data parallelism scales beautifully even over InfiniBand or, at small scale, even PCIe — the communication is bounded, batched, and hideable.

It is worth being precise about *why* the volume is $2\Psi$ and not just $\Psi$, because the factor of two is not arbitrary. A ring all-reduce runs in two phases. In the **reduce-scatter** phase, the $N$ GPUs are arranged in a logical ring; each GPU splits its gradient tensor into $N$ chunks and, over $N-1$ steps, passes a chunk to its neighbor and adds the chunk it receives, so that at the end each GPU owns the fully-summed version of exactly one chunk. In the **all-gather** phase, those summed chunks are passed around the ring again over $N-1$ steps until every GPU holds every summed chunk. Each phase moves $(N-1)/N \cdot S$ bytes per GPU, so the total is $2(N-1)/N \cdot S$, which for large $N$ approaches $2S$. The beautiful property the ring buys you is that this per-GPU volume is *independent of $N$* in the limit — going from 8 to 64 to 512 GPUs does not increase the bytes each GPU moves, only the number of small steps. That is the deep reason data parallelism can scale to thousands of GPUs at all, and it is why the ring (and its tree variants for latency-bound small messages) is the algorithm NCCL reaches for. The full derivation, with the latency-versus-bandwidth crossover that makes NCCL switch between ring and tree, is in the collectives post linked above.

There is one more lever that interacts with all of this: **gradient accumulation**. If your global batch will not fit in memory even split across replicas, you run several forward-backward passes locally, summing gradients, and only fire the all-reduce and optimizer step after the last one. This is the same idea as a micro-batch but for the data-parallel axis, and it has a lovely side effect: it *amortizes* the all-reduce. If you accumulate over 4 sub-steps before reducing, you do one all-reduce per 4 backward passes instead of one per backward pass, so the fixed $26.8$ GB is now hidden behind 4× as much compute. On a thin interconnect, cranking up gradient accumulation is often the cheapest way to rescue scaling efficiency — you trade a slightly staler gradient (it is mathematically identical to a bigger batch) for a much smaller communication-to-compute ratio. In PyTorch you express it by skipping `opt.step()` until the accumulation boundary and using DDP's `no_sync()` context manager on the non-boundary steps so DDP does not waste an all-reduce on partial gradients.

#### Worked example: when does the all-reduce stop hiding?

Suppose you train the 6.7B model on 8 A100 80GB GPUs in one DGX node connected by NVLink at roughly 600 GB/s of bidirectional bandwidth per GPU. One step's gradient all-reduce moves about 26.8 GB per GPU. At 600 GB/s that is $26.8 / 600 \approx 45$ ms of communication. If a forward-plus-backward step takes, say, 400 ms of compute at this model size and batch, then 45 ms of all-reduce hides comfortably behind it — scaling efficiency stays above 90%. Now move the same job to 8 nodes connected by a single 200 Gb/s (25 GB/s) InfiniBand link. The cross-node all-reduce of 26.8 GB now takes $26.8 / 25 \approx 1.07$ s, which does *not* hide behind a 400 ms step. Your "8× more GPUs" run is now communication-bound and might deliver 3–4× instead of 8×. The fix is more bandwidth (multi-rail IB), gradient bucketing, or — when the model is too big anyway — switching to a model-splitting strategy. The lesson: data parallelism scales until the all-reduce stops hiding, and that boundary is set by your interconnect, not your GPUs.

The code is mercifully short. PyTorch's `DistributedDataParallel` (DDP) wraps your model, registers backward hooks that bucket and launch the all-reduce automatically, and you launch it with `torchrun`:

```python
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    model = build_transformer().to(device)
    # DDP buckets gradients and overlaps all-reduce with backward.
    model = DDP(model, device_ids=[local_rank], gradient_as_bucket_view=True)

    sampler = DistributedSampler(train_dataset)  # each rank sees a disjoint shard
    loader = DataLoader(train_dataset, batch_size=64, sampler=sampler,
                        num_workers=8, pin_memory=True, prefetch_factor=4)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # reshuffle shards each epoch
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = model(x, y)        # full forward on this rank's 64 samples
            loss.backward()           # backward + overlapped grad all-reduce
            opt.step()                # identical step on every replica
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

You launch it with one command per node. On a single 8-GPU box:

```bash
torchrun --standalone --nproc_per_node=8 train.py
```

And across, say, 4 nodes of 8 GPUs each, with a rendezvous so the 32 ranks find each other:

```bash
# run on every node; NODE_RANK differs per node, MASTER_ADDR is node 0
torchrun \
  --nnodes=4 --nproc_per_node=8 \
  --node_rank=$NODE_RANK \
  --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 \
  train.py
```

Two practical notes that cost people days. First, use a `DistributedSampler` so each rank gets a *disjoint* shard of the data — without it every GPU trains on the same samples and your 8 GPUs do the work of 1. Second, call `sampler.set_epoch(epoch)` each epoch or every rank reshuffles identically and you lose the variety. Beyond these, DDP is close to free to adopt and is the correct default for any model that fits.

When you need gradient accumulation to grow the effective batch or to amortize the all-reduce on a thin link, the idiom is to suppress DDP's per-step all-reduce on the accumulation steps and only let it fire on the boundary step:

```python
accum_steps = 4
for i, (x, y) in enumerate(loader):
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    is_boundary = (i + 1) % accum_steps == 0
    # no_sync() skips the gradient all-reduce; only the boundary step reduces.
    ctx = model.no_sync() if not is_boundary else contextlib.nullcontext()
    with ctx:
        loss = model(x, y) / accum_steps   # scale so the sum is a mean
        loss.backward()
    if is_boundary:
        opt.step()                          # one all-reduce already happened here
        opt.zero_grad(set_to_none=True)
```

Without `no_sync()`, DDP would launch an all-reduce on every one of the four backward passes — three of them wasted, because you have not stepped the optimizer yet. With it, you pay the fixed gradient-traffic bill once per four passes, which on a 25 GB/s link can be the difference between 55% and 85% scaling efficiency. The natural next step when the *optimizer state* is what does not fit — not the model — is FSDP / ZeRO, which shards that state across the same data-parallel group; that is its own post, linked above.

## Tensor parallelism: split a single matmul

Data parallelism breaks the moment the model itself — not just the optimizer — does not fit on one GPU. A single 70B model in bf16 is 140 GB of parameters; there is no replica to make. Now you must split the model, and the first and cheapest place to split is *inside a layer*, across the math of a single matrix multiply. This is **tensor parallelism**, and the canonical scheme is from Megatron-LM.

The intuition first, with a number. A Transformer's feed-forward block is two big matmuls: it projects the hidden size $d$ up to $4d$, applies a nonlinearity, and projects back down to $d$. For $d = 4096$ that first weight matrix is $4096 \times 16384$ — about 67 million parameters, 134 MB in bf16, and the bulk of the layer's compute. The insight of Megatron is that you do not have to replicate this matrix; you can *cut it into vertical strips* and give each GPU one strip. Each GPU computes part of the output, in parallel, and then you stitch the parts back together with one communication.

![A dataflow diagram showing an input activation fed to two GPUs that each hold half the weight columns, compute partial outputs, and combine them with one all-reduce into the full output](/imgs/blogs/parallelism-strategies-data-tensor-pipeline-and-expert-3.png)

Concretely, Megatron does a **column split** on the first matmul and a **row split** on the second, chosen so the two combine to need exactly one all-reduce in the forward pass and one in the backward pass per block. Take the up-projection $Y = XW$ where $W$ is $d \times 4d$. Split $W$ by columns into $[W_1, W_2]$ across 2 GPUs. Each GPU has the full input $X$ and its half of $W$, so GPU 0 computes $Y_1 = XW_1$ and GPU 1 computes $Y_2 = XW_2$ — these are the left and right halves of the output, computed independently, no communication needed yet. Now the down-projection $Z = YW^O$ where $W^O$ is $4d \times d$. Split $W^O$ by *rows* into $[W^O_1; W^O_2]$ matching the column split above. GPU 0 computes $Z_1 = Y_1 W^O_1$ and GPU 1 computes $Z_2 = Y_2 W^O_2$, and now $Z = Z_1 + Z_2$ — a partial sum that needs **one all-reduce** to complete. That is the whole trick: column-then-row split turns a two-matmul block into parallel local work plus a single all-reduce.

Here is the byte cost and why it dictates hardware. That all-reduce is over the *activation* $Z$, whose size is batch × sequence × $d$ — for a microbatch of, say, 8 sequences of 2048 tokens at $d=4096$ in bf16, that is $8 \times 2048 \times 4096 \times 2 = 134$ MB. You pay this **twice per Transformer layer** (the attention block needs the same column/row split, so it is two all-reduces in forward and two in backward per layer). For a 32-layer model that is on the order of $32 \times 2 = 64$ all-reduces of activation tensors *per forward pass*, and they are on the critical path — unlike data parallelism's gradient all-reduce, these cannot hide behind compute, because the next matmul literally needs the all-reduced result as its input.

This is the entire reason tensor parallelism demands NVLink and lives *inside* a node. The all-reduce is synchronous, frequent, and blocking. Over NVLink at 600 GB/s a 134 MB all-reduce takes well under a millisecond; over a 25 GB/s InfiniBand link it takes 5–10× longer and, multiplied by 64+ blocking all-reduces per step, destroys your throughput. The Megatron-LM paper is explicit about this: tensor parallelism is kept *within* a node where NVLink/NVSwitch provides the bandwidth, and pipeline or data parallelism is used *across* nodes where only InfiniBand is available. Violate this and your scaling efficiency falls off a cliff. We unpack the interconnect hierarchy and these bandwidth numbers in [collective communication and NCCL: all-reduce from scratch](/blog/machine-learning/high-performance-computing/collective-communication-and-nccl-all-reduce-from-scratch).

There is a symmetry in the backward pass that is easy to miss and important to get right. The forward pass of the column-then-row split needs one all-reduce (to complete the row-parallel partial sum) and an identity at the column split (each GPU just keeps its slice). In the backward pass these *swap*: the gradient flowing back through the column-parallel layer needs an all-reduce (because each GPU's input gradient is a partial contribution that must be summed), while the row-parallel layer's backward is the identity. Megatron encodes this as two conjugate operators, conventionally called $f$ and $g$: $f$ is identity in forward and all-reduce in backward, $g$ is all-reduce in forward and identity in backward. The upshot is that a single attention-plus-MLP Transformer layer costs **four all-reduces per training step** — two in the forward pass (one for attention, one for the MLP) and two in the backward — every one of them on the critical path and every one of them over the tensor-parallel group. For a 32-layer model at TP-8 that is $32 \times 4 = 128$ blocking all-reduces per step across 8 GPUs, which is exactly why the link under them must be the fastest thing in your cluster.

The activation-memory benefit is the flip side and the reason you tolerate all that communication. Because each GPU holds only $1/t$ of the weights and computes only $1/t$ of the intermediate $4d$ activations, the per-GPU activation footprint of the MLP block also shrinks by the tensor-parallel degree. Combined with **sequence parallelism** (covered below), which Megatron pairs with tensor parallelism to also split the layer-norm and dropout activations, the total activation memory per GPU drops by roughly $t$× — turning a model whose *activations* would not fit into one that does. So tensor parallelism is not only a parameter-fitting tool; it is one of the strongest activation-memory levers you have, which is part of why it earns its place inside the node despite the communication.

A column-split linear in PyTorch, sketched to show the mechanics:

```python
import torch
import torch.distributed as dist

class ColumnParallelLinear(torch.nn.Module):
    """Y = X @ W, with W split by columns across the TP group.
    Each rank computes a column-slice of Y; no comms in forward (output is sharded)."""
    def __init__(self, in_features, out_features, tp_group):
        super().__init__()
        self.tp_size = dist.get_world_size(tp_group)
        assert out_features % self.tp_size == 0
        local_out = out_features // self.tp_size
        # each rank holds W shard of shape (in_features, local_out)
        self.weight = torch.nn.Parameter(torch.empty(local_out, in_features))
        torch.nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        # x is replicated across the TP group; output is column-sharded
        return torch.nn.functional.linear(x, self.weight)

class RowParallelLinear(torch.nn.Module):
    """Z = Y @ W, with W split by rows. Input Y is already column-sharded,
    so each rank produces a PARTIAL sum; one all-reduce completes it."""
    def __init__(self, in_features, out_features, tp_group):
        super().__init__()
        self.tp_group = tp_group
        self.tp_size = dist.get_world_size(tp_group)
        assert in_features % self.tp_size == 0
        local_in = in_features // self.tp_size
        self.weight = torch.nn.Parameter(torch.empty(out_features, local_in))
        torch.nn.init.normal_(self.weight, std=0.02)

    def forward(self, y_shard):
        z_partial = torch.nn.functional.linear(y_shard, self.weight)
        # the one all-reduce per block that completes the row-parallel matmul
        dist.all_reduce(z_partial, op=dist.ReduceOp.SUM, group=self.tp_group)
        return z_partial
```

That `dist.all_reduce` in `RowParallelLinear.forward` is the single blocking collective per matmul block — the one that has to be fast, hence NVLink. In real code you would use Megatron-LM's or `torch.distributed.tensor`'s implementations, which also handle the backward-pass all-reduce (an identity in forward becomes an all-reduce in backward and vice versa), but the shape above is exactly what they do.

#### Worked example: how far should you push tensor parallelism?

Say your 70B model needs to be split because 140 GB of params will not fit on 80 GB. With tensor parallelism degree 2, each GPU holds 70 GB of params plus its share of optimizer state — still too much. At TP degree 8 (all 8 GPUs in a DGX node), each holds $140/8 = 17.5$ GB of params, which leaves room for optimizer state and activations. Good — the model fits. But now every layer does its two all-reduces across all 8 GPUs, and the per-step communication is 8× more frequent than TP-2. Megatron-LM's own measurements show tensor parallelism scaling well up to 8 GPUs (one NVLink node) and then degrading sharply past the node boundary, because the all-reduce must cross InfiniBand. The rule that falls out: **cap tensor parallelism at the NVLink domain — typically 8 GPUs — and use other axes beyond that.** Pushing TP to 16 or 32 across nodes is almost always slower than TP-8 plus pipeline or data parallelism on top.

## Pipeline parallelism: split layers into stages

Tensor parallelism cuts inside a layer; **pipeline parallelism** cuts between layers. The intuition is an assembly line. Put layers 1–8 on GPU 0, layers 9–16 on GPU 1, 17–24 on GPU 2, and 25–32 on GPU 3. A batch flows through: GPU 0 processes it and hands the activations to GPU 1, which hands them to GPU 2, and so on. The communication is tiny — you send only the activations crossing a stage boundary, point-to-point, once per micro-batch — which is why pipeline parallelism happily spans nodes over InfiniBand where tensor parallelism cannot.

But there is a catch that defines this whole strategy, and it is the **bubble**. If you push one big batch through naively, then while GPU 0 works, GPUs 1, 2, and 3 sit idle waiting for it; when the batch reaches GPU 3, GPUs 0, 1, 2 are idle. Three-quarters of your hardware is doing nothing at any instant. The fix, from the GPipe paper, is to chop the batch into **micro-batches** and stream them so the stages overlap — GPU 0 starts micro-batch 2 the moment it finishes micro-batch 1 and passes it forward.

![A timeline of a pipeline schedule showing stages filling and draining one micro-batch at a time so the idle bubble shrinks as more micro-batches are added](/imgs/blogs/parallelism-strategies-data-tensor-pipeline-and-expert-4.png)

The science here is a clean, provable formula, and it is worth deriving because it tells you exactly how many micro-batches to use. With $p$ pipeline stages and $m$ micro-batches, the pipeline takes $p - 1$ time units to fill (the startup, while micro-batch 1 marches from stage 0 to stage $p-1$) and $p - 1$ to drain at the end. The steady-state, fully-utilized portion processes all $m$ micro-batches. So the total time, in units of one micro-batch-on-one-stage, is $(p - 1) + m + (p - 1) = m + 2(p-1)$ for the forward... but the standard accounting counts the bubble as the idle fraction: the pipeline does useful work for $m$ units but the wall-clock length is $m + (p - 1)$ for a one-directional schedule. The **bubble fraction** — the fraction of time wasted — is:

$$\text{bubble fraction} = \frac{p - 1}{m + p - 1}$$

Stare at that and the design rule jumps out: the bubble shrinks as $m$ grows. With $p = 4$ stages and $m = 4$ micro-batches, the bubble is $3 / (4 + 3) = 3/7 \approx 43\%$ — you waste almost half your hardware. Increase to $m = 32$ micro-batches and the bubble is $3 / (32 + 3) = 3/35 \approx 8.6\%$. Push to $m = 64$ and it is $3 / 67 \approx 4.5\%$. The bubble never reaches zero, but you can drive it down by feeding more micro-batches per step. That is the single most important knob in pipeline parallelism: **micro-batches must substantially exceed pipeline stages** ($m \gg p$) or the bubble eats your speedup.

#### Worked example: sizing micro-batches against the bubble

You split the 6.7B model across $p = 8$ pipeline stages on 8 GPUs. You want the bubble under 10%. Solve $\frac{p-1}{m + p - 1} < 0.10$, that is $\frac{7}{m + 7} < 0.10$, giving $m + 7 > 70$, so $m > 63$. You need at least 64 micro-batches per step to keep the bubble under 10% with 8 stages. If your global batch is 512 sequences, that means a micro-batch of 8 sequences each — small enough that each stage's matmuls might be inefficient (low arithmetic intensity), which is the trade-off: more micro-batches shrink the bubble but shrink each matmul. This is why real systems blend pipeline with tensor and data parallelism rather than pushing pipeline stages high — past about 8–16 stages the bubble math and the tiny micro-batches fight you. GPipe and later 1F1B (one-forward-one-backward) schedules in Megatron and DeepSpeed reduce the *memory* cost of keeping many micro-batches in flight, but the bubble fraction formula above is the law you plan around.

The code: PyTorch's newer `torch.distributed.pipelining` API (and the older `torch.distributed.pipeline.sync.Pipe`) handle the schedule for you. A GPipe-style sketch:

```python
import torch
import torch.nn as nn
from torch.distributed.pipelining import pipeline, SplitPoint, ScheduleGPipe

# A sequential stack of Transformer blocks; we split it into pipeline stages.
model = nn.Sequential(*[TransformerBlock(d=4096) for _ in range(32)])

# Tell the pipeline where to cut: 4 stages => split before blocks 8, 16, 24.
split_spec = {
    "8":  SplitPoint.BEGINNING,
    "16": SplitPoint.BEGINNING,
    "24": SplitPoint.BEGINNING,
}

example_input = torch.randn(8, 2048, 4096)  # one micro-batch shape
pipe = pipeline(model, mb_args=(example_input,), split_spec=split_spec)

stage = pipe.build_stage(stage_index=rank, device=device, group=pp_group)

# GPipe schedule: split the batch into n_microbatches and stream them.
schedule = ScheduleGPipe(stage, n_microbatches=64, loss_fn=loss_fn)

for batch_x, batch_y in loader:
    if rank == 0:
        schedule.step(batch_x)               # first stage feeds inputs
    elif rank == world_size - 1:
        losses = schedule.step(target=batch_y)  # last stage computes loss
    else:
        schedule.step()                       # middle stages just relay
```

Notice `n_microbatches=64` — that is the bubble knob from the worked example, chosen so $\frac{p-1}{m+p-1}$ is small. The framework handles the point-to-point sends between stages (over NCCL, which routes cross-node sends over InfiniBand). The only communication is activations at the 3 stage boundaries, point-to-point, which is why pipeline parallelism is the cheap way to span nodes.

There is a second cost the bubble formula hides: **memory**. The naive GPipe schedule runs *all* forward passes for the $m$ micro-batches first, then *all* the backward passes — which means every stage must keep the activations for all $m$ micro-batches in flight simultaneously, because the backward pass needs them. With $m = 64$ micro-batches that is 64× the per-micro-batch activation memory pinned on the first stage, which can itself cause an out-of-memory. The fix is the **1F1B** (one-forward-one-backward) schedule used by Megatron and DeepSpeed: once the pipeline is full, each stage alternates a forward and a backward, so a micro-batch's activations are freed as soon as its backward completes. 1F1B has the *same* bubble fraction as GPipe but caps the in-flight activations to roughly the number of stages, not the number of micro-batches — a far smaller memory footprint. There is also an **interleaved** 1F1B variant (Megatron's "virtual pipeline") that gives each GPU several non-contiguous chunks of layers, which shrinks the bubble below $\frac{p-1}{m+p-1}$ at the cost of more frequent, smaller activation sends. The practical takeaway: GPipe's formula tells you the bubble; 1F1B tells you the memory is survivable; interleaving buys you a smaller bubble when you can afford the extra communication.

This is why you almost never run pipeline parallelism alone — it pairs naturally with tensor parallelism inside each stage's node and data parallelism across pipeline replicas, which brings us to the mesh. One more honest caveat: pipeline parallelism assumes you can *balance* the stages so each holds roughly equal compute. A Transformer is convenient here because its layers are near-identical, but the embedding and the final language-model head are heavier than a middle block, so a careful split gives the first and last stages slightly fewer Transformer layers. An imbalanced pipeline has a stage that is always the straggler, and the whole pipeline runs at the speed of its slowest stage — a different failure mode from the bubble, and one you diagnose by watching per-stage step times rather than the bubble math.

## Sequence and context parallelism: splitting along time

Before the mesh, one more axis that matters more every year: **sequence parallelism** (and its long-context cousin, context parallelism). The motivation is a number that has been climbing relentlessly — context length. Attention and the activations around it scale with sequence length, and the activation memory for a long sequence can dwarf the parameters. A 32K-token context with $d=4096$ over 32 layers holds enormous activation tensors that no single GPU can store, even when the *parameters* fit fine.

Sequence parallelism splits the *sequence* dimension across GPUs. In Megatron's sequence-parallel mode, the parts of a Transformer layer that operate token-wise and do not mix across positions — layer norm, dropout, the residual adds — are split so each GPU owns a slice of the tokens, which cuts the activation memory for those operations by the parallel degree without adding communication (it actually replaces some all-reduces with all-gather and reduce-scatter that move the same total bytes). Context parallelism goes further and splits attention itself across the sequence, using ring-attention-style communication so each GPU computes attention over its token slice while exchanging keys and values with neighbors. The headline: when your bottleneck is activation memory from a long context rather than parameter memory, splitting the batch (data parallel) or the weights (tensor parallel) does not help — you must split the sequence. This is increasingly standard in long-context training and pairs with the other axes as yet another dimension of the mesh.

The cost profile sits between data and tensor parallelism: the communication is real (all-gather/reduce-scatter or ring exchanges of keys and values) but lighter than tensor parallelism's two full all-reduces per layer, and it directly relieves the activation-memory pressure that activation checkpointing only partly solves. If you have never hit a context-length wall you can defer learning this; the moment you train at 32K+ tokens, it becomes essential, and it composes cleanly with everything else here.

A number makes the motivation concrete. The attention score matrix is $T \times T$ for a sequence of length $T$ — at $T = 32{,}768$ that is over a billion entries *per head per layer*, and even with FlashAttention (which avoids materializing the full matrix) the activations you must retain for the backward pass grow linearly in $T$ across every layer. For a long-context fine-tune, activation memory can easily be 3–5× the parameter memory, so the GPU that comfortably held a 7B model's weights drowns in activations the moment you stretch the context. Neither replication (data parallel keeps full activations on each GPU) nor weight splitting (tensor parallel shrinks weight-derived activations but not the attention activations themselves at full degree) solves it. Splitting the sequence does: with sequence-parallel degree $s$, each GPU owns $T/s$ tokens' worth of the token-wise activations, cutting that dominant term by $s$. **Ring attention** is the elegant version for the attention operator itself — each GPU holds its slice of queries and streams keys and values around a ring, computing the partial attention contributions blockwise and accumulating them, so no GPU ever holds the full $T \times T$ interaction yet the result is exact. The communication is a ring of key-value blocks, which overlaps well with the attention compute. This is how frontier labs train at 100K-plus token contexts without the activation memory exploding, and it is increasingly a fourth standard axis of the mesh rather than an exotic add-on.

## The device mesh: 3D parallelism

Now combine. The reason frontier models use three or four axes at once is that no single axis scales far enough alone: data parallelism is capped by the all-reduce, tensor parallelism is capped by the NVLink domain (8 GPUs), and pipeline parallelism is capped by the bubble and the tiny micro-batches. So you compose them into **3D parallelism** — tensor inside the node, pipeline across nodes, and data parallelism replicating the whole pipeline — and you arrange the GPUs into a **device mesh** so every collective knows its group.

![A grid of GPUs arranged as a logical device mesh with a data-parallel axis across rows and a combined tensor-and-pipeline axis across columns so every collective knows its group](/imgs/blogs/parallelism-strategies-data-tensor-pipeline-and-expert-5.png)

A device mesh is the bookkeeping that makes this tractable. Each physical GPU gets a coordinate — its data-parallel rank, its tensor-parallel rank, and its pipeline-parallel rank. From those coordinates, the framework derives the *process groups*: all GPUs sharing the same data and pipeline rank but differing in tensor rank form one tensor-parallel group (they do the per-layer all-reduces together over NVLink); all GPUs sharing the same data and tensor rank but differing in pipeline rank form a pipeline group (they pass activations stage-to-stage); and all GPUs sharing the same tensor and pipeline rank but differing in data rank form a data-parallel group (they all-reduce gradients). The mesh is what lets a single GPU participate in three different collectives, each with the right partners, without you hand-wiring 512 ranks.

The arithmetic of a mesh is just multiplication. If you have TP degree $t$, PP degree $p$, and DP degree $d$, you use $t \times p \times d$ GPUs. A common 70B-class configuration: $t = 8$ (one NVLink node), $p = 4$ (four nodes deep), $d = 16$ (sixteen pipeline replicas), totaling $8 \times 4 \times 16 = 512$ GPUs. The tensor all-reduces stay on NVLink inside each node; the pipeline sends cross node boundaries over IB; the gradient all-reduces happen across the 16 data-parallel replicas and overlap with backward. Every axis is placed where its communication pattern matches the available bandwidth — that placement *is* the art of 3D parallelism.

PyTorch's `DeviceMesh` and `DTensor` make this concrete and far less error-prone than raw `dist.new_group` calls:

```python
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_module, distribute_tensor, Shard, Replicate

# A 512-GPU job arranged as (data=16, pipeline=4, tensor=8).
mesh = init_device_mesh(
    "cuda",
    mesh_shape=(16, 4, 8),
    mesh_dim_names=("dp", "pp", "tp"),
)

# Sub-meshes give you the right process group for each collective:
tp_group = mesh["tp"].get_group()   # 8-way tensor all-reduce, on NVLink
dp_group = mesh["dp"].get_group()   # 16-way gradient all-reduce
pp_group = mesh["pp"].get_group()   # 4-stage pipeline sends

# Shard a weight across the tensor dimension as a DTensor:
W = torch.empty(4096, 16384, device="cuda")
W_tp = distribute_tensor(W, mesh["tp"], placements=[Shard(dim=1)])  # column split
# DTensor tracks the sharding and inserts the right collectives automatically.
```

`DTensor` carries the sharding metadata (a `Shard(dim=...)` or `Replicate()` placement per mesh axis) so that when you do a matmul, the framework inserts the necessary all-reduce or all-gather automatically — you describe *where the data lives* and it computes *what communication is needed*. This is the modern, less bug-prone way to express what Megatron-LM historically did with hand-written collectives, and it is what FSDP2 and the latest training stacks build on.

The placement decisions inside the mesh are where the real engineering lives, and they all reduce to one principle: **put the chattiest axis on the fastest link**. Tensor parallelism is the chattiest (four blocking all-reduces per layer), so its group must lie entirely within the NVLink domain — you never let a tensor-parallel group straddle a node boundary. Pipeline parallelism is the least chatty (a couple of point-to-point sends per micro-batch), so it is the natural axis to stretch across nodes over InfiniBand. Data parallelism is in between — its gradient all-reduce is large but hideable — so it sits on whatever is left, usually spanning nodes alongside pipeline. When you also use **FSDP** to shard optimizer state, you typically run it *along the data-parallel axis* (so the parameter all-gather and gradient reduce-scatter happen within the data-parallel group), composing cleanly with tensor and pipeline parallelism on the other two axes. Getting this wrong — say, a tensor-parallel group that crosses two nodes because someone set the mesh dimension order carelessly — is one of the most common causes of a 3D run sitting at 20% MFU when it should be near 50%. The mesh abstraction does not protect you from a bad placement; it just makes a good placement expressible.

A subtle but important property of the mesh is that the *order* of the mesh dimensions in `mesh_shape` determines the physical-to-logical mapping, which in turn determines which GPUs end up in the same NVLink domain. The convention `(dp, pp, tp)` with `tp` as the last (fastest-varying) dimension places consecutive GPU ranks into the same tensor-parallel group — and consecutive ranks on a DGX node share NVLink — so the tensor-parallel all-reduces land on NVLink by construction. Reorder the dimensions and you can accidentally scatter a tensor-parallel group across nodes. This is exactly the kind of bug that does not throw an error and does not corrupt results; it just silently halves your throughput, which is why reading the achieved MFU against the Megatron reference is the check that catches it.

## Expert parallelism: routing in a mixture-of-experts

The fifth axis is different in kind, because it changes the *model*, not just how you split it. A **mixture-of-experts** (MoE) replaces a dense feed-forward block with many parallel feed-forward "experts" — 8, 64, even 256 of them — plus a small **gate** (a router) that, for each token, picks the top-$k$ experts (usually $k = 1$ or $2$) to actually run. The payoff is dramatic: you get the *parameter count* of a huge model but the *compute* of a small one, because each token only touches $k$ of the experts. Switch Transformer scaled to 1.6 trillion parameters this way while keeping per-token compute modest.

![A dataflow diagram showing a batch of tokens scored by a top-2 gate and routed to experts on different GPUs via all-to-all, then combined by a weighted sum](/imgs/blogs/parallelism-strategies-data-tensor-pipeline-and-expert-6.png)

**Expert parallelism** is how you split this across GPUs: put different experts on different GPUs. Expert 0 lives on GPU 0, expert 1 on GPU 1, and so on. But now a token that the gate routes to expert 2 might be sitting on GPU 0 — so before the experts run, you must *shuffle every token to the GPU holding its chosen expert*, and after they run, shuffle the results back. That shuffle is an **all-to-all** collective: every GPU sends a (different) chunk of its tokens to every other GPU. You pay two all-to-alls per MoE layer — one to dispatch tokens to experts, one to combine results back. This is the GShard and Switch Transformer routing mechanism.

The all-to-all is the defining cost and the defining headache. Unlike an all-reduce, where every GPU sends the same-sized contribution, an all-to-all's volume depends on the *routing* — if the gate happens to send most tokens to experts on one GPU, that GPU is overloaded and everyone waits for it. This is the **load-balancing** problem, and MoE training fights it with an auxiliary load-balancing loss that nudges the gate toward even expert usage, plus a per-expert **capacity factor** that caps how many tokens an expert accepts (dropping the overflow). Get the balance wrong and either you drop too many tokens (hurting quality) or one expert becomes a straggler (hurting throughput). The all-to-all also wants fast, uniform bandwidth — it stresses the interconnect differently from an all-reduce, and at scale across nodes it is often the bottleneck that decides MoE training speed.

The **capacity factor** deserves a number, because it is the lever you actually tune. If you have $E$ experts and a batch of $T$ tokens routed top-1, perfect balance sends $T/E$ tokens to each expert. But routing is never perfectly balanced, so you set a capacity $C = \lceil \text{capacity\_factor} \cdot T/E \rceil$ — say `capacity_factor = 1.25`, giving each expert 25% headroom. Tokens beyond an expert's capacity are *dropped* (their hidden vector passes through the residual connection unchanged, contributing nothing from the MoE layer that step). A higher capacity factor drops fewer tokens but inflates the all-to-all buffers (you pad every expert to capacity $C$, so the communication volume is $E \cdot C$ regardless of actual load), wasting bandwidth on padding. A lower capacity factor saves bandwidth but drops more tokens, hurting quality. Switch Transformer found that a capacity factor around 1.0–1.25 with a well-tuned load-balancing loss is the sweet spot for top-1 routing. This is the central systems trade-off of MoE: you are constantly balancing dropped-token quality loss against all-to-all padding waste, and the gate's load-balancing loss is what keeps both small.

One more reason MoE all-to-all is uniquely painful at scale: it is a *bursty, latency-sensitive* collective on the critical path, and it does not overlap with compute the way a data-parallel all-reduce does — the experts cannot run until the dispatch all-to-all finishes, and the combine all-to-all cannot start until the experts finish. So you pay two synchronous all-to-alls per MoE layer with the experts' compute sandwiched between them, and across many MoE layers this serializes into a large fraction of the step. Frameworks fight back with hierarchical all-to-all (group the intra-node exchange separately from the inter-node one), expert-parallelism degrees chosen to keep most traffic on NVLink, and overlapping the all-to-all of one layer with the compute of another. The deeper treatment of these serving and training optimizations is in [optimizing MoE training and inference](/blog/machine-learning/large-language-model/optimizing-moe-training-and-inference).

A top-$k$ router, the heart of the MoE layer, is short enough to write in full:

```python
import torch
import torch.nn.functional as F

class Top2Router(torch.nn.Module):
    def __init__(self, d_model, n_experts, capacity_factor=1.25):
        super().__init__()
        self.gate = torch.nn.Linear(d_model, n_experts, bias=False)
        self.n_experts = n_experts
        self.capacity_factor = capacity_factor

    def forward(self, x):                      # x: (tokens, d_model)
        logits = self.gate(x)                  # (tokens, n_experts)
        probs = F.softmax(logits, dim=-1)
        # top-2 experts per token and their gate weights
        topw, topi = probs.topk(2, dim=-1)     # (tokens, 2)
        topw = topw / topw.sum(dim=-1, keepdim=True)  # renormalize the 2 weights

        # load-balancing aux loss (Switch Transformer): encourage uniform routing
        tokens = x.shape[0]
        # fraction of tokens dispatched to each expert (top-1 for the aux term)
        me = F.one_hot(topi[:, 0], self.n_experts).float().mean(0)   # actual load
        ce = probs.mean(0)                                            # router prob mass
        aux_loss = self.n_experts * (me * ce).sum()  # minimized when both are uniform

        # capacity: max tokens an expert will accept this step
        capacity = int(self.capacity_factor * tokens * 2 / self.n_experts)
        return topi, topw, aux_loss, capacity
```

In a real MoE layer, `topi` drives the dispatch all-to-all (each token's hidden vector is sent to the GPUs hosting its two chosen experts, up to `capacity`), the experts run their feed-forward locally, and a second all-to-all combines the results, each weighted by its `topw`. Frameworks like DeepSpeed-MoE, Megatron's MoE support, and Tutel implement the all-to-all and the overflow handling; the router above is the part you actually tune. We go much deeper on the serving and training trade-offs in [optimizing MoE training and inference](/blog/machine-learning/large-language-model/optimizing-moe-training-and-inference). The strategic point: expert parallelism only exists if you chose an MoE architecture in the first place — it is not a way to split a dense model. It is a different bet on how to spend parameters, and the scaling-laws case for it connects to [Chinchilla compute-optimal scaling](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling).

## Putting it together: the decision framework

You now have five axes. The hard part is not knowing them — it is choosing. Here is the decision procedure, in order, and it follows the tree from the intro exactly.

First, **does the model fit on one GPU, with its optimizer state and a usable batch?** If yes, you are done with model splitting: use data parallelism (DDP), and if only the *optimizer state* overflows, shard it with FSDP/ZeRO across the data-parallel group. Most models up to ~10B with a modest batch land here, and the answer is gloriously simple. Do not reach for tensor or pipeline parallelism if data parallelism already saturates your NVLink — the extra communication makes it slower, not faster. This is the most common mistake: adding model parallelism a model does not need.

Second, if the model does not fit, **how badly, and what is your interconnect?** The order of escalation is dictated by communication cost matched to bandwidth. Add **tensor parallelism first, capped at the NVLink domain** (typically 8 GPUs in a node), because its frequent blocking all-reduces need the fastest link you have. If TP-8 still does not fit the model, add **pipeline parallelism across nodes**, because its only communication is cheap point-to-point activation sends that tolerate InfiniBand. Then wrap the whole tensor-pipeline unit in **data parallelism** for throughput — that is full 3D parallelism. Layer in **sequence/context parallelism** only when activation memory from long context is the specific wall. And use **expert parallelism** only if you committed to an MoE architecture.

Third, **what is your activation memory doing?** Parameters are a fixed cost, but activations scale with batch × sequence × layers and can dominate. If activations are the problem, your levers are activation checkpointing (recompute instead of store), sequence parallelism (split the time axis), and smaller micro-batches — not more weight splitting. Diagnosing *which* memory is overflowing (params vs optimizer vs activations) is the first move, and it is covered in [memory optimization: ZeRO, FSDP, activation checkpointing, and offload](/blog/machine-learning/high-performance-computing/memory-optimization-zero-fsdp-activation-checkpointing-and-offload).

The communication patterns make this concrete, and they are the real reason each strategy belongs where it does:

![A matrix comparing data, tensor, pipeline, and expert parallelism by communication per step, link required, and when to use each, showing data and pipeline tolerate slower links while tensor and expert demand fast ones](/imgs/blogs/parallelism-strategies-data-tensor-pipeline-and-expert-7.png)

That matrix is the cheat sheet. Read the "Link needed" column and the placement strategy falls out mechanically: tensor parallelism's NVLink requirement pins it inside a node; data and pipeline parallelism's tolerance for InfiniBand lets them span nodes; expert parallelism's all-to-all wants uniform fast bandwidth and careful load balancing. Here is the same comparison as a table you can keep, with the key quantities spelled out:

| Strategy | What it splits | Comms per step | Critical-path? | Belongs | Use when |
| --- | --- | --- | --- | --- | --- |
| Data parallel | The batch (model replicated) | 1 all-reduce of $\approx 2\Psi$ bytes | No (hides behind backward) | Anywhere, incl. IB/PCIe | Model fits on 1 GPU |
| Tensor parallel | A single matmul | 2 all-reduces of activations per layer | Yes (blocking) | Inside a node, NVLink | A layer is too big to fit |
| Pipeline parallel | Layers into stages | Point-to-point activations at boundaries | Partly (the bubble) | Across nodes, IB | Many layers span nodes |
| Sequence/context | The sequence dimension | All-gather/reduce-scatter or ring KV | Partly | Mixes with the above | Long-context activations overflow |
| Expert parallel | Experts in an MoE | 2 all-to-alls per MoE layer | Yes (load-imbalanced) | Fast uniform links | You chose an MoE model |

And the second table that earns its place: the same five strategies against the memory they relieve, because "which wall am I hitting" is the question that actually selects the strategy.

| Strategy | Param memory | Optimizer memory | Activation memory | Throughput effect |
| --- | --- | --- | --- | --- |
| Data parallel | No relief (replicated) | No relief (FSDP/ZeRO needed) | No relief | Scales batch, near-linear if comms hide |
| Tensor parallel | Splits by TP degree | Splits by TP degree | Splits by TP degree | Costs blocking all-reduces |
| Pipeline parallel | Splits by stages | Splits by stages | Per-stage only | Costs the bubble |
| Sequence/context | No relief | No relief | Splits by SP degree | Light comms, relieves long-context |
| Expert parallel | Splits experts across GPUs | Splits with experts | Routing-dependent | Costs all-to-all, load balancing |

## Case studies / real numbers

Now the measured results, because the whole point of this post is that you can predict and verify scaling, not just hope for it. All figures below are from the cited papers and vendor reports; where I give a round number it is approximate and labeled as such — do not quote these to four significant figures.

**Megatron-LM tensor parallelism and 3D parallelism MFU.** The Megatron-LM line of work is the reference for model parallelism on NVIDIA hardware. In the Megatron-LM scaling study (Narayanan et al., 2021), the authors trained models up to a trillion parameters on thousands of A100 GPUs by combining tensor, pipeline, and data parallelism, and reported **achieved aggregate throughput corresponding to roughly 50% of theoretical peak — about 502 petaFLOP/s on 3072 A100s for a 1T-parameter model, an MFU in the ballpark of 50%.** That ~50% **model FLOPs utilization** (MFU — the fraction of the GPU's peak FLOP/s your training actually realizes) is the number to internalize: with careful 3D parallelism, half of peak is the realistic, very good outcome at scale; the other half is lost to the all-reduces, the bubble, and the all-to-alls we have been counting. Tensor parallelism alone scaled well to 8 GPUs (the NVLink node) and degraded beyond it, exactly as the bandwidth argument predicts.

**The bubble in practice and why micro-batches matter.** GPipe (Huang et al., 2019) introduced micro-batching and reported near-linear speedup for an AmoebaNet and a Transformer across up to 8 accelerators *once the number of micro-batches comfortably exceeded the number of stages* — directly the $\frac{p-1}{m+p-1}$ relationship. The paper's central demonstration is that splitting a batch into more micro-batches drives the bubble down without changing the model's math, which is why every serious pipeline implementation since (Megatron's 1F1B, DeepSpeed's pipeline engine) is fundamentally a scheduling refinement on top of GPipe's micro-batch idea.

**GPT-3 and Llama parallelism configurations.** GPT-3 (175B parameters, Brown et al., 2020) was trained with a mix of model and data parallelism on a V100 cluster; the paper reports the model would not remotely fit on one device and required splitting both within and across layers. The Llama and Llama-2 training (Touvron et al., 2023) used data and model parallelism with activation checkpointing, and Meta reported training Llama-2-70B with high GPU utilization on a large A100 fleet — the published figures put the larger models in the rough range of **~38–50% MFU**, consistent with the Megatron numbers and a good reminder that even the best-run frontier jobs leave roughly half of peak on the table to communication and memory traffic. The exact configurations vary, but the shape is always the same: tensor parallelism inside the node, pipeline or sharding across nodes, data parallelism for throughput.

**Switch Transformer and MoE scaling.** Switch Transformer (Fedus et al., 2021) pushed mixture-of-experts to **1.6 trillion parameters with top-1 routing**, demonstrating that expert parallelism lets you grow parameters roughly 7× over a dense T5 baseline at *comparable compute per token*, and reported large pre-training speedups (the paper cites up to ~7× faster pre-training to a target quality for some configurations) precisely because each token activates only one expert. The catch they document is exactly the all-to-all and load-balancing cost we derived: the auxiliary load-balancing loss and capacity factor are load-bearing, and the all-to-all communication is the dominant systems cost of MoE at scale. GShard (Lepikhin et al., 2020) is the companion reference for the all-to-all dispatch/combine mechanism, and it is where the capacity-factor and expert-sharding machinery was first laid out at trillion-parameter scale.

#### Worked example: how to measure scaling honestly

Before you trust any of the numbers above for *your* run, you have to measure them without fooling yourself, because distributed timing is full of traps. Suppose you want to report scaling efficiency from 1 to 8 GPUs on the 6.7B model. The honest procedure: (1) **Warm up** for 10–20 steps and discard them — the first steps pay one-time costs (CUDA context creation, NCCL ring setup, autotuning of `torch.compile` or cuDNN kernels, allocator warm-up) that are not representative of steady state. (2) **Synchronize before timing**: call `torch.cuda.synchronize()` before reading the clock, because CUDA kernel launches are asynchronous and the CPU races ahead of the GPU — without the sync you are timing the launch queue, not the work. (3) **Time many steady-state steps** and take the median, not the mean, so a single slow step (a checkpoint write, a thermal throttle, an OS jitter event) does not skew the result. (4) **Control the data-loader confound**: if your `DataLoader` cannot keep the GPUs fed, you will measure the loader's speed, not the model's — check that GPU utilization stays near 100% with `nvidia-smi dmon` or the profiler, and raise `num_workers` and `prefetch_factor` until the loader is not the bottleneck. (5) **Watch for clock throttling**: a GPU at 78°C runs at a lower boost clock than one at 50°C, so an 8-GPU node that heats up will show *worse per-GPU* throughput than a single cool GPU — report the sustained, thermally-settled number. Do all five and your "8 GPUs gave 7.2× = 90% efficiency" is a number you can defend; skip them and you will misattribute a data-loader stall to a communication problem and waste a week tuning NCCL.

The before-and-after that ties the memory and communication trade-off together — pure data parallelism's memory ceiling versus 3D parallelism's higher wire cost — is worth seeing as a picture, because it is the decision in one frame:

![A before-and-after comparison showing pure data parallelism replicating 140 GB per GPU and running out of memory versus 3D parallelism sharding to 40 GB per GPU at higher communication cost and reaching about 50 percent MFU](/imgs/blogs/parallelism-strategies-data-tensor-pipeline-and-expert-8.png)

That is the entire trade you are making when you leave data parallelism behind. On the left, replication is communication-cheap but hits a hard memory wall — 140 GB of a 70B model simply will not fit on an 80 GB GPU, and there is no batch size small enough to save you because the *parameters* are the problem. On the right, 3D parallelism shards the model down to a fitting ~40 GB per GPU but spends far more on the wire, which is only viable if your interconnect can absorb it — hence the obsession with NVLink inside the node and InfiniBand across nodes. The ~50% MFU you reach on the right is not a failure; it is the realistic ceiling once communication enters the budget, and matching the Megatron reference is a sign you did it right.

#### Worked example: predicting 8 vs 64 vs 512 GPU scaling

Put numbers on the scaling curve so you can sanity-check your own runs. For pure data parallelism on a model that fits, scaling efficiency at 8 GPUs in one NVLink node is typically **>90%** (the 26.8 GB gradient all-reduce hides behind a ~400 ms step, as computed earlier). Move to 64 GPUs across 8 nodes on InfiniBand and the cross-node all-reduce grows; with good gradient bucketing and overlap you might hold **~80–85%** efficiency, but a thin interconnect can drop you to 50–60%. At 512 GPUs, pure data parallelism with a model that fits can still work if the per-GPU batch stays large enough to hide the all-reduce, but you are usually past the point where the model fits at all — so you are in 3D parallelism, where the relevant metric is MFU, and **~50% MFU (the Megatron figure)** is the target. The contrast is the headline: data parallelism is about *scaling efficiency* (how close to N× you get) and works while the model fits and the all-reduce hides; 3D parallelism is about *MFU* (how close to peak FLOP/s you get) and ~50% is excellent. If your 64-GPU data-parallel run is at 40% efficiency, your all-reduce is not hiding — fix the interconnect or the bucketing before blaming the GPUs. If your 512-GPU 3D run is at 25% MFU, your tensor parallelism probably crossed the node boundary or your pipeline bubble is too big — check the mesh placement and the micro-batch count.

## When to reach for this (and when not to)

Every axis of parallelism is a cost, and the discipline is refusing to pay a cost you do not need. The blunt rules:

**Do not add tensor parallelism if the model fits and DDP already saturates NVLink.** Tensor parallelism's two blocking all-reduces per layer are pure overhead when a replica would have worked. People reach for it because it sounds sophisticated; it makes a fitting model *slower*. Use it only when a single layer's weights genuinely do not fit, and cap it at the NVLink domain.

**Do not push pipeline parallelism past the point where micro-batches get tiny.** The bubble formula is friendly up to a point, but $m \gg p$ requires either a large global batch or micro-batches so small your matmuls lose efficiency. Past roughly 8–16 stages, you are usually better off spending GPUs on tensor or data parallelism. Pipeline parallelism *pays* mainly when you need to span nodes cheaply and have a big enough batch to feed the schedule.

**Do not adopt MoE for the parameters if you cannot pay the all-to-all.** Expert parallelism delivers huge parameter counts cheaply in compute, but the all-to-all communication and the load-balancing fragility are real operational costs. If your interconnect is thin or your serving latency budget is tight, a dense model may be the better engineering choice even at fewer parameters. MoE is a bet, not a free lunch.

**Do not chase 3D parallelism for a model that fits on 8 GPUs.** The full machinery — device meshes, mixed collectives, careful placement — is justified at hundreds of GPUs and a model that busts a node. For a 7B model on one DGX box, DDP plus activation checkpointing (or FSDP for the optimizer state) is simpler, less bug-prone, and often just as fast. Reach for complexity only when the simpler tier provably fails.

**Always diagnose which wall you are hitting before choosing.** Is it parameter memory (split the model: TP then PP), optimizer memory (shard it: FSDP/ZeRO), activation memory (checkpoint, or sequence-parallel for long context), or communication (more bandwidth, better overlap, or move the chatty axis onto NVLink)? The strategy follows the wall, and the wall is something you measure with a profiler and `nvidia-smi`, not something you guess. This is the whole spine of the series, and the capstone ties it together: [the HPC playbook for AI engineers](/blog/machine-learning/high-performance-computing/the-hpc-playbook-for-ai-engineers).

## Key takeaways

- **Start with the fit question, always.** If the model fits on one GPU with its optimizer and a usable batch, the answer is data parallelism (DDP), full stop. Model splitting is a response to memory pressure, not a default.
- **Data parallelism's cost is fixed and hideable.** One gradient all-reduce of roughly $2 \cdot \text{params} \cdot \text{bytes}$ per step, independent of batch size, overlapping with the backward pass. It scales until that all-reduce stops hiding behind compute — a boundary set by your interconnect.
- **Tensor parallelism is fast but lives on NVLink.** Two blocking activation all-reduces per layer (Megatron's column-then-row split) put it on the critical path, so cap it at the NVLink domain of ~8 GPUs and never push it across the node boundary.
- **Pipeline parallelism trades a bubble for cheap node-spanning.** The bubble fraction is $\frac{p-1}{m+p-1}$; drive it down by making micro-batches far exceed stages ($m \gg p$). It pays when you need to cross nodes and have a batch big enough to feed.
- **Sequence/context parallelism is the long-context answer.** When activation memory from a long sequence is the wall, split the time axis — neither data nor tensor parallelism relieves it.
- **Expert parallelism is an architecture choice, not a splitting trick.** MoE buys huge parameter counts at small per-token compute via a top-$k$ router, paid for with two all-to-alls per layer and a real load-balancing problem.
- **3D parallelism is composition with placement.** Tensor inside the node, pipeline across nodes, data parallel for throughput, organized by a device mesh so every collective finds its group. ~50% MFU (the Megatron figure) is an excellent real-world ceiling at scale.
- **Measure the wall before you spend complexity.** Parameters, optimizer state, activations, and communication each have a different fix; the profiler tells you which one is hurting, and the strategy follows the diagnosis.

## Further reading

- **Megatron-LM** — Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism" (2019), and Narayanan et al., "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM" (2021), for tensor/pipeline parallelism and the ~50% MFU 3D-parallel results.
- **GPipe** — Huang et al., "GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism" (2019), for micro-batching and the bubble.
- **GShard** — Lepikhin et al., "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding" (2020), for the MoE all-to-all dispatch/combine.
- **Switch Transformer** — Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity" (2021), for top-1 routing, load-balancing loss, and capacity factor.
- **PyTorch docs** — `DistributedDataParallel`, `torch.distributed.pipelining`, `DeviceMesh`/`DTensor`, and FSDP guides for the production APIs sketched above.
- Within this series: [why HPC is the bottleneck for modern AI](/blog/machine-learning/high-performance-computing/why-hpc-is-the-bottleneck-for-modern-ai), [the memory hierarchy from registers to HBM](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm), [collective communication and NCCL: all-reduce from scratch](/blog/machine-learning/high-performance-computing/collective-communication-and-nccl-all-reduce-from-scratch), [memory optimization: ZeRO, FSDP, activation checkpointing, and offload](/blog/machine-learning/high-performance-computing/memory-optimization-zero-fsdp-activation-checkpointing-and-offload), and the capstone [the HPC playbook for AI engineers](/blog/machine-learning/high-performance-computing/the-hpc-playbook-for-ai-engineers).
