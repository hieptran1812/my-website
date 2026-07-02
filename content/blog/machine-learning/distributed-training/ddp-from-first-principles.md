---
title: "DDP From First Principles: Gradient All-Reduce, Bucketing, and Overlap"
date: "2026-07-02"
publishDate: "2026-07-02"
description: "Build DistributedDataParallel from the ground up — why gradients are averaged, how one all-reduce prices the whole strategy, and how bucketing and overlap hide that cost so eight GPUs run almost eight times faster."
tags:
  [
    "distributed-training",
    "multi-gpu",
    "ddp",
    "all-reduce",
    "pytorch",
    "nccl",
    "gradient-accumulation",
    "deep-learning",
    "ml-systems",
    "gpu",
  ]
category: "machine-learning"
subcategory: "Distributed Training"
author: "Hiep Tran"
featured: true
readTime: 42
---

You have a model that fits on one GPU, a dataset that would take three weeks to grind through on that one GPU, and a machine with eight of them sitting idle. So you do the obvious thing: wrap the model in `DistributedDataParallel`, launch with `torchrun --nproc_per_node=8`, and watch the tokens-per-second number. It goes up. Not by eight. Maybe by seven and a half if your fabric is good, maybe by three and a half if it is not. Somewhere between "wrap the model" and "watch the number," a machine that has exactly eight times the compute delivered somewhere between four and seven and a half times the work — and the gap is not a mystery, it is a *mechanism*, and once you can see the mechanism you can close the gap.

That mechanism has three moving parts, and this post is about all three. First: every one of the eight GPUs holds a complete copy of the model, each processes a different slice of the batch, and before any of them can take an optimizer step they must **average their gradients** so all eight copies stay identical. That averaging is a single collective operation — an all-reduce — and it is the *entire* communication cost of data-parallel training. Second: that all-reduce moves gigabytes across the wire every single step, and on the wrong interconnect it is slower than the compute it is trying to keep up with. Third, and this is the part that makes DDP actually scale: PyTorch does not wait for the backward pass to finish and *then* start communicating. It **overlaps** the two, firing the all-reduce for the earliest-ready gradients while the backward pass is still churning through the rest of the network, so the communication hides underneath compute that was going to happen anyway.

![a data parallel training loop where each rank runs forward and backward locally and a single gradient all-reduce averages before the synchronized optimizer step](/imgs/blogs/ddp-from-first-principles-1.webp)

By the end of this post you will be able to derive from scratch why gradients are averaged and not summed, and why that makes eight GPUs mathematically equivalent to one machine with an eight-times-larger batch; price the per-step all-reduce in gigabytes and milliseconds on named hardware and predict whether it will hide under your backward pass or stall your whole job; write and launch a complete DDP training loop with the two knobs that matter most (`bucket_cap_mb` and `gradient_as_bucket_view`); use `no_sync()` to accumulate gradients across micro-steps for a larger effective batch without extra communication; apply the linear learning-rate scaling rule when you go from eight to sixty-four GPUs; and recognize the three situations where overlap breaks down and your beautiful 95%-efficient job collapses to 50%. This is the sixth post in the [Distributed Training in the Trenches](/blog/machine-learning/distributed-training/why-distributed-training) series and the workhorse of the whole thing — nearly every large model you have heard of was trained with data parallelism as at least one of its dimensions.

## The whole loop in four lines

Strip away the tooling and data-parallel training is four steps repeated a few hundred thousand times. Let me state them precisely, because every optimization later in this post is a way of making one of these four steps cheaper without changing what it computes.

One: **replicate.** At startup, every rank builds the same model and loads (or is broadcast) the same initial weights. A *rank* is one process in the job, almost always bound to exactly one GPU, so "rank 3" is shorthand for "the process driving GPU 3." The *world size* is the number of ranks — eight, on a single DGX node. After replication, all eight GPUs hold byte-for-byte identical parameters. This is the invariant DDP must preserve forever: **the replicas must never drift apart.** Everything DDP does is in service of that one promise.

Two: **split the batch.** Each rank pulls a *different* slice of the global batch. If the global batch is 128 sequences and the world size is 8, each rank gets 16 sequences. A `DistributedSampler` handles this: it deterministically partitions the dataset indices so that on any given step, no two ranks see the same example and together they cover a disjoint 128-example slice. The forward pass runs entirely locally on each GPU — no communication, no coordination. Each rank computes a loss on its own 16 sequences.

Three: **average the gradients.** Each rank runs its backward pass entirely locally and ends up with a full gradient for every parameter — but a *different* gradient, because each rank saw different data. If they each stepped their optimizer right now, the eight replicas would diverge and the invariant from step one would be broken. So before the optimizer step, all eight ranks exchange and average their gradients. Every rank contributes its local gradient, they are summed and divided by eight, and every rank walks away with the identical averaged gradient. That is the definition of an **all-reduce**, and it is the only communication in the entire loop.

Four: **step.** Every rank applies the identical averaged gradient with its own local optimizer. Because the gradients are identical and the weights started identical, the weights stay identical. The invariant holds, and the loop repeats. The figure above draws exactly this: the batch fans out to per-rank local compute, the local gradients merge into a single all-reduce, and the averaged result flows into a synchronized step. Notice that only *one* box in that diagram is colored as a cost — the all-reduce. Forward is free (local). Backward is free (local). The optimizer step is free (local). The bill for scaling data parallelism is that one caution-colored collective, and the rest of this post is about reading and reducing that bill.

This is the frame of the whole series, applied to its most common case. You add GPUs to knock down one of the four walls — here, the wall is "the data won't finish in time" — and the strategy you chose (data parallelism) costs you exactly one collective per step over whatever interconnect you have. If you can price the collective, you can price the strategy. So the first thing to understand is not the code; it is why the gradients are *averaged*, because that one word carries the entire correctness argument.

## Why average, not sum

Here is a question that trips up almost everyone the first time: when the eight ranks combine their gradients, should they *sum* them or *average* them? The all-reduce primitive itself just sums (NCCL's default reduction op is sum). So why does DDP divide by the world size afterward? The answer is not a convention — it is a correctness requirement, and deriving it tells you exactly how DDP relates to single-GPU training.

Consider ordinary mini-batch SGD on one GPU with a batch of size $B$. The loss you minimize is the *average* loss over the batch:

$$L(\theta) = \frac{1}{B} \sum_{i=1}^{B} \ell(x_i; \theta)$$

and the gradient you step with is the gradient of that average:

$$g = \nabla_\theta L = \frac{1}{B} \sum_{i=1}^{B} \nabla_\theta \ell(x_i; \theta)$$

The key word is *average*: the per-example gradients are summed and then divided by $B$. This is what your single-GPU code already does — a standard loss like `nn.CrossEntropyLoss()` uses `reduction='mean'` by default, so `loss.backward()` produces the mean gradient over the batch, not the sum.

Now split that same batch of $B$ examples across $N$ ranks, so each rank holds $B/N$ examples. Rank $r$ computes the mean loss over *its own* slice, and `loss.backward()` gives it the mean gradient over its slice:

$$g_r = \frac{1}{B/N} \sum_{i \in \text{slice}_r} \nabla_\theta \ell(x_i; \theta) = \frac{N}{B} \sum_{i \in \text{slice}_r} \nabla_\theta \ell(x_i; \theta)$$

Each local gradient already has a factor of $N/B$ baked in, because each rank divided by its *local* batch size $B/N$, not the global $B$. Now watch what happens when we average the local gradients across ranks:

$$\frac{1}{N} \sum_{r=1}^{N} g_r = \frac{1}{N} \sum_{r=1}^{N} \frac{N}{B} \sum_{i \in \text{slice}_r} \nabla_\theta \ell(x_i; \theta) = \frac{1}{B} \sum_{i=1}^{B} \nabla_\theta \ell(x_i; \theta) = g$$

The two factors of $N$ cancel, the per-rank sums stitch back into the full sum over all $B$ examples, and you are left with *exactly* the single-GPU gradient $g$. This is the whole point: **averaging the per-rank mean gradients reproduces the gradient of a single batch of size $B$.** Eight GPUs each running a local batch of 16, gradients averaged, is mathematically identical to one impossibly-large GPU running a batch of 128. Not approximately — identically, up to floating-point associativity.

If you *summed* the local gradients instead of averaging, you would get $N \cdot g$ — the gradient scaled up by the world size. Your effective learning rate would silently be $N$ times too large, and the run would diverge the moment you added GPUs. That is a real and common bug: someone writes their own `all_reduce(grad)` call, forgets the division, and cannot understand why the loss explodes on multi-GPU but is fine on one. DDP handles the division for you — it uses `ReduceOp.SUM` and then scales — but if you ever roll your own gradient sync, this factor of $N$ is the first thing to check.

This derivation also nails down the single most important fact about what data parallelism *is*: it is a way to run a **larger effective batch**, nothing more. Eight GPUs with a local batch of 16 is a global batch of 128. Sixty-four GPUs is a global batch of 1024. You are not changing the math of training; you are computing the same large-batch gradient faster by splitting the sum across machines. This is why data parallelism does not, by itself, let you train a bigger *model* — every rank still holds the full model, so if it did not fit on one GPU it does not fit on eight. Data parallelism buys you throughput, not capacity. (For capacity you shard the model itself, which is [ZeRO and FSDP: the memory model](/blog/machine-learning/distributed-training/zero-and-fsdp-the-memory-model), a sibling post.) And because a larger batch changes the optimization dynamics, it forces you to re-tune the learning rate — a subtlety we will make precise in the second worked example.

## The all-reduce is the bill

We have established that the *only* communication in DDP is one all-reduce of the gradients per step. Now let us price it, because this number decides whether your job scales.

The gradient buffer is the same size as the model's parameters, in whatever precision the gradients are stored. For a 1-billion-parameter model with bf16 gradients (2 bytes each), that buffer is $2 \times 10^9 \times 2 = 4$ gigabytes if you keep fp32 gradients, or 2 gigabytes if the gradients are bf16. Take the bf16 case: every step, each rank must participate in an all-reduce over a 2 GB buffer. That is not a one-time cost. It is 2 GB of gradient reduction *every single step*, hundreds of thousands of times over a training run.

How many bytes actually cross each GPU's wire to complete that all-reduce? This is derived in full in [Collectives From Scratch](/blog/machine-learning/distributed-training/collectives-from-scratch), and the result is the single most useful number in distributed training: a bandwidth-optimal ring all-reduce moves

$$2 \cdot \frac{N-1}{N} \cdot S$$

bytes across each GPU's link, where $S$ is the buffer size and $N$ is the world size. The factor of two is because a ring all-reduce is two passes — a reduce-scatter followed by an all-gather — each moving $(N-1)/N \cdot S$ bytes. For our 2 GB buffer on 8 GPUs, that is $2 \times (7/8) \times 2\ \text{GB} = 3.5$ GB moved per GPU, per step. As $N$ grows, the $(N-1)/N$ factor approaches 1, so the per-GPU traffic converges to ${2S}$ regardless of how many GPUs you add — which is the beautiful property of the ring: **the communication volume per GPU does not grow with the world size.** Eight GPUs or eight hundred, each GPU moves about ${2S}$ bytes to complete the all-reduce.

That flat-in-$N$ property is why data parallelism scales at all. But "does not grow with $N$" is not the same as "free." Moving 3.5 GB per GPU takes *time*, and that time depends entirely on the bandwidth of the wire.

![a comparison matrix of NVLink, PCIe, and InfiniBand showing all-reduce bandwidth, the time to reduce a two gigabyte gradient, and the resulting eight-GPU scaling efficiency](/imgs/blogs/ddp-from-first-principles-2.webp)

NCCL, the library PyTorch uses for these collectives, reports an *effective all-reduce bandwidth* — usually called "busbw" in the `nccl-tests` benchmark — that is defined so that the wall-clock time of an all-reduce is approximately the buffer size divided by busbw. That definition already folds in the ${2(N-1)/N}$ ring factor, which is convenient: you can estimate the all-reduce time as just $S / \text{busbw}$. On a DGX A100 node with NVLink, a well-tuned NCCL all-reduce sustains a busbw in the neighborhood of 200 GB/s (this is approximate and depends on message size and topology). So our 2 GB gradient all-reduce takes roughly $2\ \text{GB} / 200\ \text{GB/s} = 10$ milliseconds. On PCIe without NVLink, where GPUs talk through the host and NCCL cannot use peer-to-peer links, the effective all-reduce bandwidth collapses to single-digit GB/s — call it 8 GB/s — and the same 2 GB gradient now takes $2\ \text{GB} / 8\ \text{GB/s} = 250$ milliseconds. Same code, same model, same math; a twenty-five-fold difference in communication time, driven entirely by the wire. The matrix above tabulates it, and [The Interconnect Physics](/blog/machine-learning/distributed-training/the-interconnect-physics) explains why the gap is so brutal.

Here is why that time is the whole ballgame. If the backward pass for this model takes, say, 90 milliseconds, then a 10-millisecond all-reduce is a rounding error — you can hide all of it and the step is essentially just compute. But a 250-millisecond all-reduce is nearly three times the backward pass; there is no way to hide it, and it stalls every GPU while they wait for gradients to finish crossing the PCIe bus. The all-reduce is the bill, and whether you can afford it depends on the ratio of communication time to compute time — a ratio that gets *worse* for smaller models (less compute to hide behind) and thinner interconnects (slower comms). Hold that ratio in your head; it is the single most predictive quantity in this post.

## Bucketing and overlap: hiding the bill

So far the picture is a strict sequence: run the full backward pass, and *then* all-reduce the gradients. That naive schedule is correct but wasteful, because during the all-reduce every GPU's compute units sit idle waiting for the network, and during the backward pass the network sat idle waiting for compute. Two expensive resources, each idle exactly when the other is busy. The fix is to run them at the same time.

![a two-column comparison of a serial schedule where backward finishes before communication starts against an overlapped schedule where all-reduce fires during the backward pass](/imgs/blogs/ddp-from-first-principles-3.webp)

This is the single optimization that makes DDP fast, and it rests on one observation about how the backward pass works. Backpropagation computes gradients in *reverse* layer order: the gradient for the last layer is ready first, then the second-to-last, and so on down to the first layer. The gradient of layer 24 exists a long time before the gradient of layer 1. So there is no reason to wait for *all* the gradients before starting to communicate — the moment the last layer's gradient is computed, you can begin all-reducing it while the backward pass keeps churning through layers 23, 22, 21. By the time backward reaches layer 1, most of the network's gradients have already been reduced and are sitting averaged, ready for the optimizer step. The communication has been **overlapped** with the backward compute, hiding underneath it. That is the difference between the two columns in the figure: on the left, communication is a serial phase that adds to the step time; on the right, communication happens *during* backward and adds almost nothing.

But there is a tension. If you fired a separate all-reduce for every single parameter tensor the instant its gradient was ready, you would launch thousands of tiny collectives per step. Every collective has a fixed launch overhead — kernel launch latency, NCCL handshake, the cost of getting the ring spun up — and for a small tensor that overhead dwarfs the actual data transfer. A thousand tiny all-reduces is far slower than a few big ones, even though the total bytes are identical. This is a bandwidth-versus-latency problem: small messages are latency-bound and waste the wire; large messages are bandwidth-bound and saturate it.

DDP's answer is **bucketing.** Instead of one all-reduce per parameter, DDP groups consecutive parameters into buckets of a target size — the default is about 25 MB, controlled by `bucket_cap_mb` — and fires the all-reduce for a whole bucket at once, as soon as every gradient in that bucket is ready. A 25 MB bucket is large enough to be bandwidth-bound (the launch overhead is amortized over a meaningful transfer) but small enough that many of them fill during a single backward pass, giving overlap plenty of opportunities to fire. It is the Goldilocks size between "one giant all-reduce at the end, no overlap" and "a thousand tiny all-reduces, all latency."

### Gradients fill buckets in reverse

The interaction between bucketing and reverse-order backward is worth drawing explicitly, because it is where the overlap actually comes from.

![a four-row grid showing gradients becoming ready in reverse layer order and packing into buckets, with each full bucket launching its all-reduce while earlier layers still compute](/imgs/blogs/ddp-from-first-principles-4.webp)

DDP assigns parameters to buckets in roughly the reverse of the order their gradients will be produced — which, since backward runs in reverse, means roughly the *forward* order of the layers gets split into buckets that fill from the back. As the figure shows, the gradient for layer 24 is ready first and drops into bucket 0. A few more layers' gradients arrive, bucket 0 fills to 25 MB, and DDP immediately launches its all-reduce — while the backward pass is still down in layer 18. That all-reduce runs on the NCCL stream, on the GPU's copy engines and network, entirely in parallel with the compute stream still doing backward for the earlier layers. Bucket 1 fills next and fires; bucket 2 fills and fires. By the time backward finishes layer 1, buckets 0 and 1 have long since completed their all-reduces, and only the final bucket's communication is left to finish. The exposed communication — the part that is *not* hidden under backward — is just that last bucket, a small fraction of the total.

There is a mechanical detail that makes this work: DDP registers an **autograd hook** on every parameter. When a parameter's gradient is computed, its hook fires, DDP marks that gradient "ready," and checks whether the parameter's bucket is now complete. When a bucket's last gradient arrives, DDP launches that bucket's `all_reduce` asynchronously and moves on. No explicit orchestration in your training loop — you write a normal `loss.backward()`, and the hooks do all the overlap underneath. This is why DDP is nearly a drop-in wrapper: the entire bucketing-and-overlap machinery lives inside the autograd hooks, invisible from your code.

One more knob compounds the benefit: `gradient_as_bucket_view=True`. Normally each parameter has its own `.grad` tensor, and DDP has to *copy* each gradient into the bucket's contiguous buffer before the all-reduce, then copy the reduced result back out. With `gradient_as_bucket_view=True`, DDP instead makes each parameter's `.grad` a *view* into the bucket buffer, so the gradient is written directly into the bucket during backward — no copy in, no copy out. This saves both memory (you do not keep a second copy of every gradient) and time (you skip two large copies per bucket per step). It is essentially free correctness-wise and should be on for almost every job. We will turn it on in the code below.

### The reducer, and why the first iteration is special

The component inside DDP that owns all of this is called the **reducer**. It is a C++ object created when you wrap your model, and it does three things: it assigns every parameter to a bucket, it registers the autograd hook that marks gradients ready, and it launches each bucket's `all_reduce` when the bucket is complete. Once per iteration it also resets its internal "ready" counters so the next step starts clean. Almost none of this is visible from Python — you see a `DDP` wrapper and a normal `backward()`, and the reducer does the rest.

Two subtleties from the reducer earn their keep in practice. First, the **bucket order is decided once, from the reverse order of parameters as registered**, and DDP assumes the gradients will keep arriving in roughly that order every step. If your model's backward order changes step to step — because of data-dependent control flow, say — the reducer's overlap gets less effective, and this is one reason `static_graph=True` exists: it tells DDP the graph (and therefore the gradient-ready order) is fixed, so it can commit to an ordering and even skip re-checking for unused parameters after the first step. Second, the **first iteration is special.** On step one, DDP has to rebuild buckets to match the *actual* order gradients arrive (which it cannot know until it sees one backward), and with `find_unused_parameters` it traverses the graph to find parameters that got no gradient. So the first step is slower and does extra work — which is exactly why your benchmark harness must warm up before timing. If you ever see step one take five times as long as step two, that is the reducer doing its one-time setup, not a bug.

There is also a correctness reason the reducer must see *every* parameter's gradient each step: it waits for all buckets before returning from `backward()`, and if some parameter never receives a gradient (because a branch of the model did not run), its bucket never completes and DDP would hang forever waiting for it. That is the failure mode `find_unused_parameters=True` prevents — at the cost of a per-step graph traversal — and it is why the flag exists at all. The right fix is usually to not have unused parameters, but when you genuinely do (a conditionally-executed expert, a frozen-then-unfrozen head), the flag is your escape hatch. The full taxonomy of these hangs lives in [DDP Internals and Gotchas](/blog/machine-learning/distributed-training/ddp-internals-and-gotchas).

## Gradient accumulation with no_sync()

There is a situation where you want a *larger* effective batch than your GPUs' memory allows — say your global batch of 128 is not enough for stable large-model training and you want 512, but you cannot fit a bigger local batch on each GPU and you do not have more GPUs. The answer is **gradient accumulation**: run several forward-backward passes on different micro-batches, letting the gradients add up in `.grad`, and only step the optimizer after $K$ of them. Since `.grad` accumulates by default (backward *adds* to existing gradients unless you zero them), $K$ micro-batches of size $B$ produce the same accumulated gradient as one batch of size $KB$ — an effective batch $K$ times larger, at the cost of $K$ times the compute per step.

The subtlety in DDP is communication. If you naively ran $K$ backward passes, DDP would fire its all-reduce on *every single one* of them, because each `backward()` triggers the autograd hooks. That is $K$ full gradient all-reduces to produce one optimizer step — a $K$-fold waste, since you only need the *final* accumulated gradient to be averaged across ranks. The intermediate all-reduces average partial sums that you are just going to add to anyway.

![a timeline of gradient accumulation where the first micro-steps run under no_sync with the all-reduce skipped and only the final micro-step synchronizes before the optimizer step](/imgs/blogs/ddp-from-first-principles-5.webp)

PyTorch gives you a context manager for exactly this: `model.no_sync()`. Inside `no_sync()`, DDP disables its gradient hooks, so `backward()` accumulates gradients locally *without* any all-reduce. You run the first $K-1$ micro-steps inside `no_sync()` — pure local accumulation, zero communication — and then run the last micro-step *outside* it, where the hooks fire normally and a single all-reduce averages the fully accumulated gradient across all ranks before you step. The figure lays out the schedule: three micro-steps of silent local accumulation, then one micro-step that triggers the all-reduce, then the optimizer step. You paid for $K$ times the compute and exactly *one* all-reduce, giving an effective batch of $K$ times the global batch for the price of one gradient sync. Here is the pattern in code:

```python
accum_steps = 4
optimizer.zero_grad(set_to_none=True)
for i, (x, y) in enumerate(loader):
    x, y = x.to(device), y.to(device)
    is_last_micro = (i % accum_steps) == (accum_steps - 1)
    # Skip the all-reduce on every micro-step except the last one.
    ctx = model.no_sync() if not is_last_micro else nullcontext()
    with ctx:
        loss = model(x, y).loss / accum_steps  # scale so the sum is a mean
        loss.backward()
    if is_last_micro:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

Two details that bite people. First, the `/ accum_steps` on the loss: because gradients *sum* across the $K$ micro-steps, and each micro-step's loss is already a mean over its own micro-batch, you must divide by $K$ so the accumulated gradient is the mean over all $KB$ examples rather than $K$ times too large. This is the same factor-of-$N$ logic as the average-not-sum derivation, one level down. Second, the gradient clipping and optimizer step go *inside* the `is_last_micro` branch — you clip and step once per $K$ micro-batches, not once per micro-batch. Get either wrong and your effective learning rate is off by a factor of $K$, and you will spend an afternoon wondering why accumulation "hurts" your loss curve.

## The full training loop, in code

Enough mechanism — here is a complete, runnable DDP training script with the pieces we have discussed wired together. This is the shape of essentially every data-parallel training loop; you copy it and change the model, the dataset, and the optimizer.

```python
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

def main():
    # torchrun sets these env vars for every process it spawns.
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Bind this process to its GPU BEFORE init_process_group so NCCL picks
    # the right device, then initialize the process group over NCCL.
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)

    model = build_model().to(device)              # your model, full copy per rank
    model = DDP(
        model,
        device_ids=[local_rank],
        bucket_cap_mb=25,                          # the overlap bucket size
        gradient_as_bucket_view=True,              # grads written into the bucket, no copy
        static_graph=True,                         # allow graph-structure optimizations
    )

    dataset = build_dataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(
        dataset, batch_size=16, sampler=sampler,
        num_workers=4, pin_memory=True, prefetch_factor=2, drop_last=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)                   # reshuffle differently each epoch
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = torch.nn.functional.cross_entropy(out, y)
            loss.backward()                        # hooks fire the overlapped all-reduces
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

Every line earns its place. `torch.cuda.set_device(local_rank)` must come *before* `init_process_group` so NCCL binds each process to a distinct GPU; skip it and multiple ranks pile onto GPU 0. The `DDP` wrap is where all the machinery lives: `device_ids=[local_rank]` tells DDP which GPU this replica lives on, `bucket_cap_mb=25` sets the overlap bucket size, `gradient_as_bucket_view=True` eliminates the gradient copies, and `static_graph=True` tells DDP the network structure does not change between iterations, which unlocks a faster code path and lets it skip some bookkeeping. The `DistributedSampler` is what makes each rank see a disjoint slice — and `sampler.set_epoch(epoch)` is *not* optional: without it, every epoch reshuffles identically and every rank sees the same order every epoch, quietly hurting your training. You launch it with `torchrun`, which spawns one process per GPU and sets the `RANK`, `LOCAL_RANK`, and `WORLD_SIZE` environment variables the script reads:

```bash
# Single node, 8 GPUs.
torchrun --standalone --nproc_per_node=8 train.py

# Two nodes, 8 GPUs each (run on each node, with matching rendezvous endpoint).
torchrun \
  --nnodes=2 --nproc_per_node=8 --node_rank=0 \
  --rdzv_backend=c10d --rdzv_endpoint=node0:29500 \
  train.py
```

That is a working data-parallel trainer. The `--standalone` flag on the single-node form spins up a local rendezvous so you do not have to specify an endpoint; the multi-node form uses the `c10d` rendezvous backend with an explicit endpoint that all nodes agree on. Getting rank, local rank, and world size right is covered end to end in [Your First Multi-GPU Run](/blog/machine-learning/distributed-training/your-first-multi-gpu-run); here the point is that the training loop itself is *almost identical to single-GPU code* — the only additions are the process-group setup, the `DDP` wrap, and the `DistributedSampler`. Everything else, including the overlap that makes it fast, happens inside `backward()` where you cannot see it.

Before we measure, one comparison worth internalizing: the difference between the knobs, and when each matters.

| Knob | What it does | When to change it |
|---|---|---|
| `bucket_cap_mb` (default 25) | Sets the all-reduce bucket size | Larger for very big models on fast fabric; smaller if overlap starts too late |
| `gradient_as_bucket_view=True` | Grads are views into the bucket, no copy | Almost always on; saves memory and two copies per bucket |
| `static_graph=True` | Assumes fixed graph across steps | On when the model structure is fixed; off if control flow changes per step |
| `find_unused_parameters=True` | Waits for params that got no gradient | Only when some params are conditionally unused; it is *slow*, avoid otherwise |

That last row is a trap worth flagging now and dissecting in [DDP Internals and Gotchas](/blog/machine-learning/distributed-training/ddp-internals-and-gotchas): `find_unused_parameters=True` makes DDP traverse the autograd graph every step to discover which parameters received no gradient, so it knows not to wait for their all-reduce. It is correctness insurance for models with conditional branches, but it is expensive, and people enable it reflexively to silence an error message without realizing they just taxed every step. Leave it off unless you actually have unused parameters.

## What DDP is not

It is worth being precise about what DDP replaced, because the alternatives still show up in old code and in well-meaning-but-wrong hand-rolled scripts, and understanding why they are worse cements why DDP is built the way it is.

The first alternative is `torch.nn.DataParallel` — note the missing "Distributed." It is the older, single-process API: one Python process holds the model, and on every forward pass it *scatters* the input across GPUs, replicates the model to each, runs them, and *gathers* the outputs back to GPU 0, which computes the loss. It looks convenient — one line, no `torchrun` — but it is slow for reasons that are structural, not tunable. It runs all GPUs from one process, so the Python GIL serializes the per-GPU launches. It re-broadcasts the model to every GPU *every single forward pass* instead of keeping a persistent replica. It funnels all outputs and the loss through GPU 0, creating a memory and compute hotspot that unbalances the GPUs. And it has no overlap of gradient communication with backward at all. `DataParallel` is deprecated for exactly these reasons; if you see it in a codebase, replacing it with DDP is usually a free speedup.

The second alternative is rolling your own gradient sync: after `loss.backward()`, loop over parameters and call `dist.all_reduce(p.grad)` yourself, then divide by the world size. This is a great *teaching* exercise — it makes the average-not-sum math concrete — but as production code it leaves all the performance on the table. It fires one all-reduce per parameter (thousands of tiny latency-bound collectives), it does zero overlap (the whole sync happens after backward completes, serially), and it is easy to get the division wrong. Everything DDP's reducer does — bucketing the tiny collectives into 25 MB bandwidth-bound ones, overlapping them with backward via autograd hooks, handling the division — is precisely the work a hand-rolled loop skips. Here is the comparison in one table:

| Approach | Model replicas | Gradient sync | Overlap with backward | Verdict |
|---|---|---|---|---|
| `nn.DataParallel` | Re-broadcast every forward | Implicit, via gather to GPU 0 | None | Deprecated; GIL-bound, GPU-0 hotspot |
| Hand-rolled `all_reduce` | Persistent, one per rank | One per parameter, after backward | None | Fine for learning, slow in production |
| `DistributedDataParallel` | Persistent, one per rank | Bucketed, ~25 MB, in hooks | Yes, during backward | The default; near-linear on fast fabric |

The pattern is that DDP wins on the two axes that matter — persistent replicas (no re-broadcast) and overlapped bucketed communication (no serial comms phase, no tiny-message latency). If you understand why the two alternatives are slow, you understand why DDP's design is not arbitrary: every piece of it exists to eliminate a specific inefficiency in the naive approaches.

## Measuring it honestly

You cannot claim a speedup you have not measured correctly, and multi-GPU timing is full of ways to fool yourself. Here is the honest way to time a DDP step, and the confounds that will lie to you if you skip it.

```python
import time
import torch

def benchmark_step(model, batch, optimizer, warmup=10, iters=50):
    device = next(model.parameters()).device
    x, y = batch
    # Warm-up: the first steps pay for cuDNN autotuning, NCCL ring setup,
    # and allocator growth. Never time them.
    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize(device)                 # finish all queued GPU work
    t0 = time.perf_counter()
    for _ in range(iters):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize(device)                 # wait for the LAST step to finish
    dt = (time.perf_counter() - t0) / iters
    return dt
```

The two `torch.cuda.synchronize()` calls are the crux. CUDA kernels and NCCL collectives are launched *asynchronously* — the Python call returns immediately while the GPU works in the background. If you time without synchronizing, you are timing how fast Python can *launch* work, not how fast the GPU *finishes* it, and you will report absurd throughput. You synchronize once before starting the clock (so all warm-up work is truly done) and once after the loop (so the last step's backward and all-reduce have actually completed). The warm-up matters too: the first few steps pay for cuDNN's autotuner picking convolution algorithms, NCCL establishing its ring topology and buffers, and the caching allocator growing to its steady-state footprint. Time those and you will blame your fabric for a cost that vanishes after step ten.

Three confounds will still lie to you. The **data loader**: if your `DataLoader` cannot feed the GPU fast enough, your "training step" time silently includes the wait for the next batch, and you will misattribute loader stalls to compute or comms. Fix by benchmarking on a single pre-loaded batch (as above) to isolate the model, then separately check that the loader keeps up. **Thermal and clock throttling**: a GPU that boosts to a high clock for the first few seconds and then throttles under sustained load will show a decaying throughput; measure in steady state, after the clocks settle. And **the all-reduce itself is a synchronization point** — a straggler rank that is slow for any reason (a slow disk, a hot GPU, a noisy neighbor) makes *every* rank wait at the all-reduce, so a per-rank average can hide the fact that one rank is dragging the whole job. Always look at the slowest rank, not the mean.

To actually *see* whether your all-reduce is overlapping, you need a trace, and `torch.profiler` gives you one with the NCCL collectives on their own timeline row:

```python
from torch.profiler import profile, schedule, ProfilerActivity, tensorboard_trace_handler

prof_schedule = schedule(wait=5, warmup=5, active=5, repeat=1)  # skip warm-up, capture 5 steps
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=prof_schedule,
    on_trace_ready=tensorboard_trace_handler("./trace"),
    record_shapes=True,
) as prof:
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        prof.step()                                # advance the profiler schedule
```

Open the resulting trace in TensorBoard or Chrome's `chrome://tracing`, and look for the `nccl:all_reduce` kernels. In a *healthy* overlapped run, those all-reduce kernels sit on the NCCL stream *underneath* the backward-pass compute kernels — they run concurrently, and the step's wall-clock time is barely longer than the compute alone. In a *broken* run, the all-reduce kernels sit in a solid block *after* the last backward kernel, with the compute stream idle beside them — that gap is your exposed communication, drawn to scale. This single visual is the fastest way to confirm the mechanism this whole post describes is actually happening on your hardware. Aligning these traces *across* ranks — so you can see which rank finishes backward first and who waits longest at the barrier — is the subject of a dedicated profiling post; for a single-node sanity check, one rank's trace is enough to tell overlap from no-overlap at a glance.

![a two-column result comparison of the same eight-GPU job on PCIe versus NVLink showing exposed all-reduce, step time, and achieved speedup and efficiency for each](/imgs/blogs/ddp-from-first-principles-6.webp)

With honest measurement in hand, the payoff of overlap is stark, and the figure above is the result we have been building toward. Same code, same 1B model, same 8 GPUs — only the interconnect changes. On NVLink the all-reduce is ~10 ms, fully hidden under a ~90 ms backward, the step is ~135 ms, and you get 7.6x throughput at 95% efficiency. On PCIe the all-reduce is ~250 ms, most of it exposed because it dwarfs the backward, the step balloons to ~300 ms, and the speedup collapses to ~3.7x at 46% efficiency. The mechanism did not change and the code did not change; the *ratio of comms time to compute time* changed, and that ratio is the whole story. Let us make it a worked example.

#### Worked example: the comms cost and how much overlap hides it

Take a 1-billion-parameter transformer, bf16 gradients, on 8x A100 80GB connected by NVLink. Walk the numbers:

- **Gradient buffer:** $2 \times 10^9$ params $\times$ 2 bytes = 2 GB.
- **Ring all-reduce volume per GPU:** $2 \cdot (N-1)/N \cdot S = 2 \times (7/8) \times 2\ \text{GB} = 3.5$ GB moved per GPU per step.
- **All-reduce time on NVLink:** with NCCL busbw ≈ 200 GB/s, time ≈ $S/\text{busbw} = 2\ \text{GB} / 200\ \text{GB/s} \approx 10$ ms (busbw folds in the ring factor, so you divide by $S$, not by the 3.5 GB).
- **Backward compute:** at roughly 50% MFU on this model and batch, call it ~90 ms; forward ~45 ms.
- **With overlap:** the 10 ms of comms fits entirely inside the 90 ms backward window, so exposed comms ≈ 0. Step ≈ forward 45 + backward 90 ≈ 135 ms.
- **Scaling:** 8 GPUs process 8x the data in the same ~135 ms a single GPU takes for its slice, so ideal speedup is 8x; after loader jitter and slight load imbalance, ≈ 7.6x, i.e. **~95% efficiency.** This is the good regime.

Now the stress test — the *same job on PCIe* (no NVLink, GPUs talking through the host):

- **All-reduce time on PCIe:** NCCL busbw collapses to roughly 6–10 GB/s without peer-to-peer links, so the 2 GB all-reduce takes ~200–330 ms. Call it ~250 ms.
- **With overlap:** overlap can hide at most one backward's worth of comms (~90 ms); the remaining ~160 ms is *exposed*, stalling every GPU. Step ≈ 45 + 90 + 160 ≈ ~300 ms.
- **Scaling:** 8 GPUs now take ~300 ms per step versus a single GPU's ~135 ms for its slice, so speedup ≈ $8 \times 135/300 \approx 3.6\text{–}3.7\text{x}$, i.e. **~46% efficiency.** Half your hardware is idle, waiting on the PCIe bus. No code change fixes this; you need a better wire, a bigger model (more compute to hide behind), or a different parallelism strategy.

The lesson is a ratio, not a rule: overlap hides communication only up to the length of the backward pass. When comms time is well under backward time, you win big; when comms time approaches or exceeds backward time, the excess is exposed and efficiency falls off a cliff. Bigger models help (more compute per byte of gradient); faster fabrics help (fewer milliseconds per byte); tiny models on thin fabric are the worst case.

#### Worked example: effective batch and the linear LR scaling rule

Now the second thing scaling changes: the learning rate. Suppose each GPU runs a local batch of 16 sequences at sequence length 2048, so 32,768 tokens per GPU per step.

- **On 8 GPUs:** global batch = $8 \times 16 = 128$ sequences = 262,144 tokens per step.
- **On 64 GPUs:** global batch = $64 \times 16 = 1024$ sequences = 2,097,152 tokens per step — an 8x larger batch.

From the average-not-sum derivation, we know an 8-GPU run is exactly a batch-128 run and a 64-GPU run is exactly a batch-1024 run. Larger batches produce lower-variance gradient estimates, which means you can — and to keep training dynamics comparable, *should* — take proportionally larger steps. The **linear scaling rule** (Goyal et al., 2017, *Accurate, Large Minibatch SGD*) says: when you multiply the batch size by $k$, multiply the base learning rate by $k$. If batch-128 trains well at a peak learning rate of $3 \times 10^{-4}$, then batch-1024 (8x larger) should use roughly $8 \times 3 \times 10^{-4} = 2.4 \times 10^{-3}$.

Two caveats that keep this from blowing up. First, the rule needs a **warmup**: you cannot start at the full scaled learning rate from step zero, because the early gradients are large and noisy; you ramp the learning rate linearly from near zero over the first few hundred to few thousand steps, then apply your normal schedule. Second, the rule breaks down at *very* large batches — beyond some critical batch size the returns diminish and pure linear scaling over-shoots — which is why the largest runs use square-root scaling or careful tuning past a point. But for the common 8x-to-64x range, linear scaling with warmup is the right default, and forgetting it is a top reason a run that was healthy at 8 GPUs diverges at 64. And if you want the effective batch of 64 GPUs but only have 8, combine this with the `no_sync()` accumulation from earlier: 8 micro-steps of accumulation on 8 GPUs gives you the batch-1024 gradient, at 8x the compute per step and exactly one all-reduce.

## When overlap breaks down

The 95%-efficient case is the happy path, and it is genuinely common on a well-provisioned NVLink node. But DDP has failure modes, and knowing them is the difference between a job that scales and a night spent staring at a profiler. Each of these is a case where the comms-to-compute ratio goes wrong, and each is dissected further in [DDP Internals and Gotchas](/blog/machine-learning/distributed-training/ddp-internals-and-gotchas).

**The model is too small.** Overlap hides communication under the backward pass, so if the backward pass is short — a small model, a tiny hidden size — there is not enough compute to hide the all-reduce behind, and the comms is exposed. A 100-million-parameter model on 8 GPUs can easily be comms-bound even on NVLink, because its backward is only a few milliseconds and its gradient all-reduce, while small in absolute terms, is a large fraction of that. The fix is counterintuitive: make each GPU do *more* work per step (larger local batch, longer sequences) so there is more backward to hide behind, or accept that tiny models simply do not scale well across many GPUs and use fewer of them. Data parallelism rewards big models with big backward passes.

**The interconnect is thin.** This is the PCIe case from the worked example, and it is the most common real-world disappointment. People rent 8-GPU instances assuming NVLink and get PCIe-connected GPUs, or they go multi-node over a slow Ethernet fabric instead of InfiniBand, and their all-reduce time jumps by an order of magnitude. The tell is that single-node NVLink scaling is great but adding a second node tanks efficiency — the cross-node hop is now the bottleneck. You diagnose it by measuring achieved all-reduce bandwidth with `nccl-tests` and comparing to the spec; a 10x gap means NCCL fell back to a slow transport, which you chase down with `NCCL_DEBUG=INFO`:

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET \
  torchrun --standalone --nproc_per_node=8 train.py 2>&1 | grep -E "NET/|Ring|via"
```

That dump tells you which transport NCCL chose (NVLink peer-to-peer, PCIe, IB, or — the horror — TCP sockets over Ethernet), and reading it is a skill in its own right. If you see `NET/Socket` where you expected `NET/IB`, your InfiniBand is not being used and every all-reduce is crawling over TCP. That specific autopsy — and the full `NCCL_DEBUG` grammar — is the subject of a later post; for now, know that a surprising efficiency drop almost always traces to the transport NCCL actually selected, not the one you assumed.

**A straggler.** Because the all-reduce is a synchronization barrier, the whole job runs at the speed of the *slowest* rank. One GPU running hot and throttling its clocks, one node with a slower disk starving its loader, one rank stuck doing extra work — any of these makes all seven other GPUs wait at the all-reduce every single step. The insidious part is that per-rank averages hide it: seven ranks at 100% and one at 60% looks like "88% average utilization," but the job runs at the stragger's 60%. You catch it by looking at the *distribution* of per-rank step times, not the mean, and by watching for one rank whose "waiting at all-reduce" time is near zero while everyone else's is high (that near-zero rank is the straggler — it is the one everyone waits for). The straggler is common enough at scale that it gets its own war-story post; here, just know that a mysterious efficiency loss with no obvious comms or compute cause is very often one slow rank.

The through-line: DDP's overlap is a bet that your backward pass is long enough and your fabric fast enough to hide the all-reduce, and that every rank runs at the same speed. When any of those three assumptions breaks, the exposed communication or the straggler wait shows up directly in your step time. The `debugging-ddp-and-multi-gpu` companion post, [Debugging DDP and Multi-GPU](/blog/machine-learning/debugging-training/debugging-ddp-and-multi-gpu), walks the diagnostic tree for each.

## Case studies and real numbers

A few results from the literature and from measured hardware, to anchor the mechanism in reality. Where a number is approximate or version-dependent, I say so — do not treat any single figure as a spec.

**The PyTorch DDP paper.** The design we have been describing is documented in *PyTorch Distributed: Experiences on Accelerating Data Parallel Training* (Li et al., VLDB 2020). It is the primary source for bucketing and overlap, and its measurements are the reference point: they show that gradient bucketing with computation-communication overlap lifts DDP from poor scaling to near-linear scaling on models like ResNet and BERT across dozens of GPUs, and they quantify how the bucket size trades off launch overhead against overlap opportunity — exactly the 25 MB Goldilocks argument. If you read one paper alongside this post, read that one; every knob in the `DDP` constructor traces to a decision in it.

**NCCL all-reduce bandwidth, NVLink vs the rest.** The `nccl-tests` benchmark reports "busbw" for all-reduce, and the qualitative result is robust across generations: on an intra-node NVLink/NVSwitch fabric, all-reduce busbw lands in the low hundreds of GB/s (order 200+ GB/s on 8x A100, higher on H100/NVLink4); over PCIe without peer-to-peer it drops to single-digit-to-low-tens of GB/s; over InfiniBand HDR the per-node injection bandwidth is around 25 GB/s (200 Gb/s), so multi-node all-reduce is fundamentally slower than intra-node. The exact number depends on message size, NCCL version, and topology, but the *ratios* are what drive your scaling, and they are stable: intra-node NVLink is roughly an order of magnitude faster than PCIe for these collectives, which is precisely why single-node scaling is easy and multi-node scaling is hard.

**Large-model training MFU.** The headline runs that use data parallelism as one dimension report model FLOPs utilization (MFU) in a characteristic band. Megatron-LM's tensor+pipeline+data-parallel runs reported roughly 50% MFU on large clusters; the PaLM training report described about 46% MFU (57% including rematerialization) at massive scale; well-tuned LLaMA-class runs sit in a similar 40–55% range. Data parallelism alone, on a single well-connected node with a big model, can push MFU higher than that because it has the least communication of any parallelism dimension — one all-reduce per step, fully overlapped. The reason those big runs sit at ~50% rather than ~95% is that they are *also* paying for tensor and pipeline parallelism communication, which DDP-only jobs avoid. That is a useful north star: if your single-node DDP job is far below 50% MFU, the problem is usually not the all-reduce — it is your kernels, your data loader, or your batch size, and the profiler will tell you which.

**The 8x-GPUs-same-speed war story.** The canonical DDP failure — eight GPUs running barely faster than one — is almost always one of three things we have covered: `find_unused_parameters=True` taxing every step, a PCIe/Ethernet fabric exposing the all-reduce, or a straggler rank. The debugging companion post catalogs the full set, but the meta-lesson is that the *mechanism* tells you where to look: if adding GPUs does not add throughput, the exposed communication or a synchronization stall is eating the gain, and the profiler's "time spent in all-reduce / time spent waiting at all-reduce" split points straight at the cause.

## When to reach for DDP (and when not to)

DDP is the first tool you reach for and the last one you should abandon, but it is not universal. Here is the decision, drawn as a tree, followed by the plain-language version.

![a decision tree for whether DDP is the right lever based on whether the model fits one GPU and whether the interconnect can hide the all-reduce](/imgs/blogs/ddp-from-first-principles-7.webp)

**Reach for DDP when the model fits on one GPU and you want to go faster.** This is the overwhelmingly common case, and DDP is the right answer: it is nearly a drop-in wrapper, it has the least communication of any parallelism strategy (one overlapped all-reduce per step), and on a fast intra-node fabric it delivers 90%+ efficiency with almost no tuning. If your model fits and your fabric is NVLink, stop reading about fancier parallelism and just use DDP — adding tensor or pipeline parallelism to a job that DDP already scales well is pure overhead, more communication for no benefit.

**Do not reach for DDP when the model does not fit on one GPU.** DDP replicates the *full* model on every rank, so it saves you exactly zero memory — if a 13B model in fp16 plus its Adam optimizer states does not fit in 80 GB on one card, it does not fit on eight cards running DDP either. That is a capacity problem, and the answer is to *shard* the model with ZeRO or FSDP, which is the subject of [ZeRO and FSDP: the memory model](/blog/machine-learning/distributed-training/zero-and-fsdp-the-memory-model). DDP and FSDP are not competitors; FSDP is what you graduate to when replication no longer fits, and it reuses the exact same all-reduce identity (reduce-scatter plus all-gather) we priced here.

**Be wary of DDP when your interconnect is thin or you are going multi-node prematurely.** If your all-reduce is exposed — PCIe-only GPUs, a slow Ethernet fabric, a model too small to hide the comms — DDP will scale poorly, and no knob fixes a fundamental bandwidth shortage. The decision tree's danger branch is exactly this: model fits, but the fabric cannot hide the all-reduce, so the job stalls. Before you go multi-node, *saturate one node* — get single-node 8-GPU DDP to 90%+ efficiency first, because if you cannot make intra-node NVLink scale, cross-node InfiniBand (an order of magnitude slower) will only be worse. Multi-node is a last resort you take when one node's eight GPUs genuinely are not enough, not a default.

**And do not enable `find_unused_parameters` reflexively.** It is the single most common self-inflicted DDP slowdown. Enable it only when you truly have parameters that receive no gradient on some steps (conditional model branches, some multi-task setups); otherwise it taxes every step for nothing.

## Key takeaways

- **DDP is one all-reduce per step, and nothing else.** Forward, backward, and the optimizer step are all local; the only communication is averaging the gradients. Price that one collective and you have priced the whole strategy.
- **Average, not sum.** Averaging per-rank mean gradients reproduces exactly the single-GPU gradient of a batch $N$ times larger. Summing instead scales your learning rate by $N$ and diverges — the top self-inflicted multi-GPU bug.
- **Data parallelism buys throughput, not capacity.** Every rank holds the full model, so DDP does not help a model that does not fit on one GPU. For that, shard with FSDP/ZeRO.
- **The all-reduce moves $2(N-1)/N \cdot S$ bytes per GPU, flat in $N$.** Its wall-clock time is $S/\text{busbw}$, and busbw depends entirely on the interconnect — ~200 GB/s on NVLink, single-digit on PCIe. That ratio decides your scaling.
- **Bucketing plus overlap is what makes DDP fast.** Gradients ready in reverse layer order fill ~25 MB buckets, each firing its all-reduce during the still-running backward pass, so communication hides under compute and the step time falls to roughly the backward time.
- **Turn on `gradient_as_bucket_view=True` and `static_graph=True`; leave `find_unused_parameters` off** unless you genuinely have unused parameters.
- **Use `no_sync()` for gradient accumulation** to get a larger effective batch for the price of one all-reduce — and remember to divide the loss by the accumulation count.
- **When you scale the batch, scale the learning rate.** Linear scaling with warmup keeps 64-GPU training as stable as 8-GPU training.
- **Overlap breaks down three ways:** the model is too small (not enough backward to hide behind), the fabric is too thin (exposed all-reduce), or a straggler makes every rank wait at the barrier. A mysterious efficiency loss is almost always one of these.
- **Measure honestly:** warm up, `torch.cuda.synchronize()` around the timed region, isolate the loader, watch the slowest rank, not the mean.

## Further reading

- *PyTorch Distributed: Experiences on Accelerating Data Parallel Training* (Li et al., VLDB 2020) — the primary source for DDP bucketing and computation-communication overlap.
- *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour* (Goyal et al., 2017) — the linear learning-rate scaling rule and warmup, derived and validated.
- The PyTorch DDP documentation and the `DistributedDataParallel` API reference — the authoritative list of constructor knobs (`bucket_cap_mb`, `gradient_as_bucket_view`, `static_graph`, `find_unused_parameters`).
- The NCCL documentation and `nccl-tests` — how to measure achieved all-reduce bandwidth (busbw) on your own hardware and read the ring/tree it built.
- [Collectives From Scratch](/blog/machine-learning/distributed-training/collectives-from-scratch) — the full derivation of the ring all-reduce byte law this post rests on.
- [ZeRO and FSDP: the memory model](/blog/machine-learning/distributed-training/zero-and-fsdp-the-memory-model) — what you graduate to when the model no longer fits on one GPU.
- [DDP Internals and Gotchas](/blog/machine-learning/distributed-training/ddp-internals-and-gotchas) — unused parameters, seeding, `DistributedSampler`, and the subtle correctness traps.
- [Debugging DDP and Multi-GPU](/blog/machine-learning/debugging-training/debugging-ddp-and-multi-gpu) — the diagnostic tree for the 8x-GPUs-same-speed trap and gradient-sync bugs.
- [The Distributed Training Playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook) — the capstone that ties DDP, FSDP, tensor, and pipeline parallelism into one decision checklist.
