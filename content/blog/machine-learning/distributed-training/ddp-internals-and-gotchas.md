---
title: "DDP Internals and Gotchas: The Bugs That Only Appear on Eight GPUs"
date: "2026-07-02"
publishDate: "2026-07-02"
description: "The subtle DDP correctness and performance traps that pass on one GPU and only surface on eight: the unused-parameter hang, static_graph, gradient-as-bucket-view, the DistributedSampler coverage bug, the two jobs of seeding, and SyncBatchNorm — each with runnable fixes and an assertion that proves your ranks agree."
tags:
  [
    "distributed-training",
    "multi-gpu",
    "ddp",
    "pytorch",
    "nccl",
    "distributed-sampler",
    "batchnorm",
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

Your training script works. It works on one GPU, it works on two, it passed code review, and the loss goes down. So you launch it on all eight GPUs of the DGX node, and one of three things happens. Either it hangs at the very first step — no error, no output, just eight processes sitting at 100% GPU utilization forever, until the job scheduler kills them at the wall-clock limit. Or it runs, the loss goes down beautifully, you ship the model, and a week later someone notices the eight-GPU model is measurably worse than the one-GPU model you trained last month on the exact same data. Or it runs and the loss looks *identical* to single-GPU — suspiciously identical — and you slowly realize you spent eight GPUs of electricity to train on one eighth of your dataset, eight times over.

None of these bugs exists on one GPU. That is what makes them so expensive. Single-GPU training has no ranks, no collectives, no shards, no cross-process synchronization — so an entire class of correctness and performance bugs simply cannot occur, and every habit you built training on one card is silent about them. The moment you wrap your model in `DistributedDataParallel` and launch with `torchrun`, you inherit a distributed system, and distributed systems fail in ways that a single process never does: they deadlock when two ranks disagree about what to do next, they silently diverge when each rank quietly does something slightly different, and they waste hardware when the work is duplicated instead of divided. DDP hides almost all of this behind a one-line wrapper, which is exactly why the failures feel like they come from nowhere.

![how a single parameter with no gradient leaves its bucket incomplete so the all-reduce never launches and every rank blocks in the collective](/imgs/blogs/ddp-internals-and-gotchas-1.webp)

This post is a field guide to that class of bug. By the end you will be able to: explain precisely *when* DDP decides to communicate, and why an unused parameter hangs the whole job; use `find_unused_parameters`, `static_graph`, and `gradient_as_bucket_view` correctly and know what each one costs; set up a `DistributedSampler` so your eight ranks actually cover the whole dataset instead of one shard of it; get seeding right — which is two different requirements that everyone conflates into one — so your ranks start from identical weights but see decorrelated randomness; convert BatchNorm layers to `SyncBatchNorm` when the per-GPU batch gets small; and, most importantly, write an assertion that *proves* all eight of your ranks hold the identical model after a step, so the "subtly worse" bug can never ship silently again. This is the seventh post in the [Distributed Training in the Trenches](/blog/machine-learning/distributed-training/why-distributed-training) series; it assumes you have seen a `torchrun` launch and the gradient all-reduce, which we build in [DDP from first principles](/blog/machine-learning/distributed-training/ddp-from-first-principles) and [your first multi-GPU run](/blog/machine-learning/distributed-training/your-first-multi-gpu-run).

## First, what DDP actually does during backward

Every gotcha in this post is a consequence of one design decision, so we have to be precise about it. When you wrap a module in `DistributedDataParallel`, DDP does two things at construction time and one thing on every backward pass, and if you hold those three facts in your head the rest of the post is obvious.

At construction, DDP does two setup steps. First, it **broadcasts** the module's parameters and buffers from rank 0 to every other rank. This is `_sync_module_states`, and it is the reason — which we will return to when we get to seeding — that all your ranks start from bit-identical weights even if you did nothing to make that happen. Second, it walks the model's parameters *in reverse order of registration* and packs them into **buckets**: contiguous chunks of memory, 25 MB each by default, that group many parameters' gradients together so they can be all-reduced in one shot instead of one tiny collective per parameter. Reverse order is deliberate, because backward produces gradients roughly last-layer-first, so the parameters that will be ready soonest are packed into the first bucket to be reduced.

Then, on every backward pass, DDP does the one thing that matters. It registers an **autograd hook** on every parameter — specifically on the `AccumulateGrad` node that writes into `param.grad`. When backward produces the gradient for a parameter, that parameter's hook fires and DDP marks it *ready*, decrementing a pending counter on its bucket. **When a bucket's counter hits zero — meaning every parameter assigned to that bucket has produced its gradient — DDP launches that bucket's all-reduce asynchronously**, while the rest of backward is still running for the earlier layers. That overlap of gradient communication with gradient computation is the entire performance story of DDP, and we derive it in [DDP from first principles](/blog/machine-learning/distributed-training/ddp-from-first-principles). Here we care about the flip side: the all-reduce for a bucket does not launch until *every* parameter in it is ready.

That is the load-bearing sentence. DDP is *waiting*. Each bucket is a rendezvous, counting down to zero, and it will not fire the collective until the count reaches zero. Now ask the question that breaks everything: what if one parameter's hook never fires?

### The counter that never reaches zero

Look again at the figure above. The healthy path is the top row: backward produces gradients for the parameters, a bucket fills, its counter reaches zero, and the all-reduce launches. The broken path is the bottom row: some parameter — a frozen head you are not training this step, a branch of the model that this particular batch did not exercise, an auxiliary loss you commented out — never receives a gradient. Its hook never fires. The bucket that parameter lives in never reaches a count of zero. The all-reduce for that bucket never launches. And because the final step of DDP's backward *waits for every bucket to finish reducing* before it returns control to your training loop, the whole thing stalls.

On a single GPU this is a non-event: there is no all-reduce, no bucket, no counter. The gradient for that parameter is simply `None`, the optimizer skips it, and you never notice. On eight GPUs it is a hang or a hard error, and which one you get depends on a subtlety we will make precise in the next section. This is the archetype of every bug in this post: a difference that is *invisible and harmless* on one device becomes *fatal* the moment there are multiple ranks that must stay in lockstep.

There is a clean quantitative law hiding in the mechanism, and it is worth stating because it predicts the performance of DDP directly. A bucket of size `B` bytes holds gradients that finish computing over some window of backward. DDP launches the collective at the instant the *last* parameter in the bucket becomes ready — call that time `t_ready` — and the collective takes about `2(N-1)/N · B / BW` seconds to complete on a ring of `N` GPUs over an interconnect of bandwidth `BW`. DDP wins when that collective finishes before backward finishes, so its cost is hidden. It loses — and the eight-GPU job crawls — when the collective for the *first* bucket cannot start until backward is nearly done, which is exactly what happens when an unused parameter lands in an early bucket and stalls it. The mechanism that makes DDP fast is the same mechanism that makes it hang. Let us fix the hang.

## Gotcha 1: the unused parameter that hangs the job

Here is a model that trains perfectly on one GPU and hangs on eight. It has two output heads — a main head and an auxiliary head — and a config flag that turns the auxiliary loss on and off. You are running an ablation with the auxiliary loss off.

```python
import torch
import torch.nn as nn

class TwoHeadModel(nn.Module):
    def __init__(self, d_model=1024, vocab=32000):
        super().__init__()
        self.backbone = nn.Linear(d_model, d_model)
        self.main_head = nn.Linear(d_model, vocab)
        self.aux_head = nn.Linear(d_model, vocab)   # trained only when use_aux=True

    def forward(self, x, use_aux=False):
        h = torch.relu(self.backbone(x))
        logits = self.main_head(h)
        aux = self.aux_head(h) if use_aux else None
        return logits, aux

# ... in the training step, with use_aux=False:
logits, aux = model(x, use_aux=False)
loss = loss_fn(logits, targets)   # aux_head never touched -> no gradient
loss.backward()
```

On one GPU this is fine. `aux_head.weight.grad` is `None`, the optimizer sees no gradient for it and leaves it alone, and you never think about it again. On eight GPUs, `aux_head`'s parameters are registered with DDP, they were packed into a bucket, and their autograd hooks are armed and waiting. This backward pass never produces their gradients, so their bucket never completes. What you observe depends on the PyTorch version and on whether the unused set is the same on every rank, and it is worth being exact about the two failure modes because they look different in the logs.

**The hard error.** In recent PyTorch, DDP detects on the *next* iteration that the previous backward did not finish reducing every bucket, and it raises a pointed `RuntimeError` telling you exactly what happened:

```console
RuntimeError: Expected to have finished reduction in the prior iteration
before starting a new one. This error indicates that your module has
parameters that were not used in producing loss. You can enable unused
parameter detection by passing the keyword argument find_unused_parameters=True
to torch.nn.parallel.DistributedDataParallel, and by making sure all forward
function outputs participate in calculating loss.
Parameters which did not receive grad for rank 0: aux_head.weight, aux_head.bias
```

**The true hang.** The nastier version happens when different ranks disagree about which parameters are used — for example a data-dependent branch where rank 3's batch triggers the auxiliary path and rank 5's does not. Now rank 3 fills and reduces the auxiliary bucket while rank 5 never does. NCCL collectives are matched positionally across ranks: rank 3 calls an all-reduce that rank 5 never calls, so rank 3 blocks forever waiting for rank 5 to join, and rank 5 blocks at the end of backward waiting for a bucket that will never complete. No error is raised. Both processes sit at full GPU utilization, spinning, and the only symptom is that the step counter stops advancing. This is the deadlock the opening figure draws, and it is the single most common "my job hangs and I don't know why" report in distributed training.

### The fix, and the fix behind the fix

The one-line fix is to tell DDP that some parameters may legitimately go ungradiented:

```python
model = torch.nn.parallel.DistributedDataParallel(
    model,
    device_ids=[local_rank],
    find_unused_parameters=True,   # tolerate params that get no gradient
)
```

With this flag, DDP walks the autograd graph at the end of each forward pass, starting from the outputs and traversing backward through the graph to discover which parameters are actually reachable from the loss this iteration. Any parameter that is *not* reachable is marked ready immediately, so its bucket can complete without waiting for a gradient that is never coming. The hang goes away.

But it is not free, and the cost is the reason this flag is a gotcha rather than a default. That graph traversal runs *every single step*, because in principle the set of used parameters could change from step to step — that is the whole reason you would need the flag. The traversal is `O(number of nodes in the autograd graph)`, which for a deep model is thousands of nodes, walked in Python-adjacent C++ on the critical path between forward and backward. On top of the traversal, DDP cannot rebuild its buckets into the optimal order, because it does not know until runtime which parameters participate, so you also lose some of the compute-communication overlap.

![the default deadlock on an unused parameter versus the flag that walks the autograd graph every step to keep the job running at a few percent overhead](/imgs/blogs/ddp-internals-and-gotchas-2.webp)

The figure contrasts the two states: on the left, the default DDP deadlock; on the right, the traversal that keeps the job alive at a few percent overhead. And that framing points at the better fix. `find_unused_parameters=True` treats a *structural* fact about your model — that `aux_head` is not trained in this ablation — as if it were a *dynamic* fact that could change every step. It pays a per-step tax to rediscover something that is actually constant. The better fix, whenever the set of unused parameters is the same on every step, is to **fix the model** so there are no unused parameters: do not register the auxiliary head as a submodule when you are not training it, or freeze it explicitly and exclude it, or — the cleanest option for a genuinely static graph — use `static_graph=True`, which we get to next. Reach for `find_unused_parameters` only when your model has *genuinely dynamic* control flow, where different steps really do use different parameters and there is no static answer.

#### Worked example: the find_unused_parameters tax

Let us put a number on it. Take a 350M-parameter encoder on eight A100 80GB GPUs connected by NVLink, per-GPU batch tuned so a training step takes 180 ms in steady state. Turn on `find_unused_parameters=True` even though the model has *no* unused parameters — a common defensive habit — and measure the step time again.

The graph traversal for a model this size walks on the order of a few thousand autograd nodes. On its own that is roughly 4 to 7 ms of overhead per step, plus a smaller penalty from the less-optimal bucket order that costs a bit of overlap. Call it about 8 ms total, which on a 180 ms step is roughly a 4 to 5 percent slowdown. Over a training run that is a week of wall-clock time, 5 percent is eight hours of A100 time — on eight cards at roughly \$2 per GPU-hour, on the order of \$130 of compute burned to defensively enable a flag the model never needed. On a much larger 7B model where the step is 1.5 seconds, the same fixed traversal cost of a few milliseconds is under half a percent and genuinely does not matter — which is why the advice depends on your step time. The rule that falls out: enable `find_unused_parameters` only when you actually have dynamic unused parameters, and when you do and the *set* is static, prefer `static_graph=True`, which pays the traversal cost once instead of every step.

## Static graph: telling DDP the shape never changes

Most training loops run the *exact same computation* every step. Same layers, same order, same set of parameters touched, same points where gradients become ready. If that is true of your model, you can tell DDP so, and it can stop re-deriving on every step what it could have learned once:

```python
model = torch.nn.parallel.DistributedDataParallel(
    model,
    device_ids=[local_rank],
    static_graph=True,     # the graph is identical every iteration
)
```

With `static_graph=True`, DDP records the structure of the autograd graph and the order in which gradients become ready during the *first* iteration (technically it uses the first one or two iterations to observe and lock in the plan). After that, it replays a fixed schedule. Three concrete things become possible once the graph is known to be static.

![a timeline where DDP records the graph shape on the first iterations then replays a fixed bucket plan on every later step with no per-step search](/imgs/blogs/ddp-internals-and-gotchas-3.webp)

First, DDP can detect unused parameters *once*, on the first iteration, and reuse that knowledge — so a model with a *statically* unused parameter (like our ablation with the auxiliary head permanently off) no longer needs `find_unused_parameters` at all, and no longer pays the per-step traversal. Second, DDP can rebuild its buckets into the *optimal* order based on the actually-observed gradient-ready sequence, which recovers the compute-communication overlap that `find_unused_parameters` gives up. Third — and this is the reason many large-model recipes turn it on — `static_graph` makes DDP compatible with **activation checkpointing** that reruns the forward pass during backward, and with parameters that are *used more than once* per forward (weight tying, a shared encoder called on two inputs), both of which otherwise confuse the hook-counting because a parameter's gradient arrives in multiple pieces. The timeline in the figure shows the shape of it: a costlier first couple of iterations that trace and plan, then a long tail of cheap replayed steps.

The measured effect is a real speedup, not a rounding error. On models with unused parameters or reused parameters, switching from `find_unused_parameters=True` to `static_graph=True` typically buys back the few-percent traversal tax *and* improves overlap, for something in the 5 to 10 percent throughput range on communication-bound steps. It is one of the highest-leverage one-word changes in a DDP training script — when it is safe.

### When static graph is a landmine

`static_graph=True` is a *promise*, and DDP believes you. The promise is: every iteration executes the same set of parameters in the same order, forever. If that promise is false, DDP happily replays the plan it locked in on step one, and the plan is now wrong — it may reduce the wrong buckets, or skip a parameter that this step actually used, and it will do so **silently**. There is no error. The gradients are subtly incorrect, the loss looks plausible, and you are back in the worst category of bug: the model that trains fine and comes out worse. The danger node on the right of the timeline is exactly this: a graph that changes after the plan is frozen produces wrong gradients with no warning.

So `static_graph` is unsafe precisely when your forward pass has **data-dependent control flow**: a mixture-of-experts router that sends different tokens to different experts each step; an early-exit network that stops at a different layer depending on the input; a curriculum that switches loss terms partway through training; any `if` in `forward` whose branch depends on the data rather than on a fixed config. If your forward is a straight-line computation that does the same thing to every batch — which describes the overwhelming majority of standard transformer and CNN training — `static_graph=True` is safe and you should use it. If it is not, do not, and reach for `find_unused_parameters` instead, paying the per-step traversal precisely because the graph genuinely is dynamic. The two flags are mutually exclusive by design, and the choice between them is really a single question: *is the set of used parameters the same every step?* Static graph for yes, find-unused for no.

## gradient_as_bucket_view: free memory you can quietly corrupt

Here is a flag that is almost always a pure win, with one sharp edge. Recall that DDP packs gradients into flat 25 MB bucket buffers to all-reduce them. By default, backward writes each gradient into that parameter's own `.grad` tensor, and DDP then *copies* the gradient out of `.grad` into the bucket buffer before the all-reduce, and copies the averaged result back afterward. That is two extra copies per gradient per step, and the bucket buffers are duplicate memory: you are holding each gradient twice, once in `.grad` and once in the bucket.

`gradient_as_bucket_view=True` removes both:

```python
model = torch.nn.parallel.DistributedDataParallel(
    model,
    device_ids=[local_rank],
    gradient_as_bucket_view=True,   # .grad points into the bucket, no copy
)
```

With it, each parameter's `.grad` is not a separate tensor — it is a **view into the bucket buffer**. Backward writes the gradient directly into the exact bytes that the all-reduce will send, and reads the averaged result from those same bytes, with no copy in either direction and no duplicate allocation. On a 7B model whose gradients are on the order of 14 GB in fp32, eliminating the duplicate gradient buffer is 14 GB of GPU memory recovered — often the difference between fitting and not fitting. It is close to free performance and memory, and for large models it is a near-default.

![a memory stack where each parameter gradient is a view into the flat bucket buffer so backward writes straight into the all-reduce buffer with no extra copy](/imgs/blogs/ddp-internals-and-gotchas-4.webp)

The stack in the figure shows the aliasing: parameters at the top, their `.grad` tensors as views, the flat bucket buffer they point into, the all-reduce reading straight from that buffer — and the trap node, because there is exactly one. Since `.grad` is now a *view into a shared buffer that DDP owns and reuses*, you can no longer treat it as your own private tensor. Two habits break. First, **do not mutate `.grad` in place** expecting it to behave like an independent tensor — if you write custom gradient surgery like `p.grad.mul_(mask)` or an in-place gradient-clipping variant that assumes `.grad` is yours, you are now editing the bytes inside the bucket, and depending on timing you may be corrupting the buffer the all-reduce is about to send, or that it just wrote. Use the out-of-place forms (`torch.nn.utils.clip_grad_norm_` is safe; it scales via the optimizer's view correctly), or `clone()` the gradient before you do anything exotic to it. Second, **do not hold references to `.grad` across iterations** and expect the values to persist — DDP may reuse or resize the bucket, and your saved reference can point at stale or overwritten data. If you need to stash a gradient (for logging, for a custom EMA of gradients), `clone()` it out first. The clone-before-mutate rule is the entire gotcha, and it appears in the summary table at the end because it is easy to violate accidentally inside a custom optimizer.

## The sampler bug that trains on one eighth of your data

Now we leave the DDP wrapper and move to the data loader, which is where the *correctness* bugs live — the ones that do not hang and do not error, that just quietly make your model worse or waste your compute. Start with the most common one, and the most expensive, because it can run to completion and produce a plausible model that is trained on a fraction of your dataset.

The mechanism is simple. Each of your eight ranks is a separate Python process running your whole training script. Each one constructs its own `DataLoader` over the *same* dataset. If you do nothing to coordinate them, all eight loaders iterate the dataset in the same order and hand each rank the same batches. Now DDP's gradient all-reduce averages eight *identical* gradients, which is the same as one gradient — so your "effective batch size" of eight times the per-GPU batch is a lie; the gradient statistics are those of a single per-GPU batch, and you are paying for eight GPUs to compute the same thing eight times. That is the pure-waste version.

The fix is `DistributedSampler`, which shards the dataset so each rank sees a distinct, non-overlapping slice: rank 0 gets indices 0, 8, 16, …; rank 1 gets 1, 9, 17, …; and so on, so that across the eight ranks the union is the whole dataset and each rank does one eighth of the work. That is what makes the effective batch size real: eight ranks, each on a different eighth, all-reduced into one gradient over the full global batch.

```python
from torch.utils.data import DataLoader, DistributedSampler

sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,          # <-- THE line everyone forgets
    shuffle=True,
    drop_last=True,
)
loader = DataLoader(
    dataset,
    batch_size=per_gpu_batch,
    sampler=sampler,    # note: sampler replaces shuffle=; do not pass both
    num_workers=8,
    pin_memory=True,
    drop_last=True,
)
```

And here is the bug that is worse than doing nothing, the one that trains you on one eighth of your data. You *did* add a `DistributedSampler` — you knew you were supposed to — but you constructed it without passing `rank` (or you passed the wrong variable, or you let it default). `DistributedSampler` defaults `rank` and `num_replicas` by reading them from the current process group, so *usually* an omitted `rank` still works — but if the sampler is built before `init_process_group`, or in a context where the default group is not what you think, `rank` can resolve to 0 on every process. Now every rank believes it is rank 0, so every rank takes shard 0, and all eight processes train on the same one eighth of the dataset — the other seven eighths are never loaded, never seen, never learned from.

![a grid contrasting the broken case where every rank reads the same shard with the correct case where each rank reads a distinct shard of the dataset](/imgs/blogs/ddp-internals-and-gotchas-5.webp)

The grid draws both states: the top row is the bug, every rank pinned to shard 0; the bottom row is correct, each rank on its own shard. The reason this is so dangerous is that it *looks fine*. The loss goes down — in fact it often goes down *faster*, because the model is overfitting to a small slice of data it sees over and over. Training metrics are healthy. Only your validation loss, on data the model was supposed to have seen a representative sample of, tells you something is wrong — and only if you are watching it against a single-GPU baseline. Always construct `DistributedSampler` after `init_process_group`, and pass `rank` and `num_replicas` explicitly so there is no ambiguity.

### set_epoch: the shuffle that never changes

There is a second, subtler sampler bug sitting right next to the first. `DistributedSampler` shuffles by seeding its random permutation with an internal epoch counter that starts at 0 and *never advances on its own*. If you do not tell it the epoch changed, it produces the **identical shuffle every epoch** — rank 0 sees the same indices in the same order in epoch 5 as in epoch 0. Your model sees the same batch sequence over and over, which reduces the stochasticity that stochastic gradient descent relies on and can measurably hurt generalization. The fix is one line, at the top of your epoch loop, before you create the iterator:

```python
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)   # re-seeds the shuffle; MUST be called each epoch
    for batch in loader:
        train_step(batch)
```

`set_epoch` also matters for correctness across ranks: it ensures all ranks re-shuffle *consistently* with the same epoch seed, so the shards stay disjoint after reshuffling. Skip it and your shuffle is frozen; call it and every epoch is a fresh, coordinated permutation.

### drop_last and the duplicated tail

The last sampler subtlety is about what happens when your dataset size is not divisible by the world size. `DistributedSampler` needs every rank to get exactly the same number of samples, because DDP's all-reduce requires every rank to run the same number of steps — if one rank runs out of data a step early, it stops calling the collective and every other rank hangs waiting for it (the same deadlock as an unused parameter, from a different cause). To keep the counts even, `DistributedSampler` by default *pads* the dataset: it repeats a few samples from the start so the total is divisible by the world size. During training that padding is harmless noise. During **evaluation** it is a real bug: those repeated samples get counted twice in your metrics, so your reported accuracy or loss is computed over a slightly wrong denominator. For training, `drop_last=True` on both the sampler and the loader drops the ragged tail and keeps batches uniform. For evaluation, you want to *not* drop data but also *not* double-count it — which usually means gathering per-sample results across ranks and de-duplicating the padded indices before you compute the metric. The general rule: `drop_last=True` for training loaders, and handle the padding explicitly when you evaluate.

#### Worked example: eight ranks, one shard

Put concrete numbers on the coverage bug. Say your dataset is 1,000,000 examples and you train on eight GPUs with a per-GPU batch of 32, so the global batch is 256, and one epoch is `1,000,000 / 256 ≈ 3,906` steps.

In the **correct** setup, each rank's sampler hands it a disjoint 125,000-example shard. Over one epoch the eight ranks collectively touch all 1,000,000 examples exactly once, and each optimizer step averages gradients computed over 256 distinct examples. Everything is as intended.

In the **broken** setup — sampler built without `rank`, every process pinned to shard 0 — all eight ranks draw from the *same* 125,000 examples. The other 875,000 examples are never loaded. Worse, at each step all eight ranks are likely to be reading overlapping or identical batches from that one shard, so your true effective batch is far smaller than 256 and your gradient diversity collapses. The model sees 12.5 percent of the data, cycles through it eight times faster than you think, and overfits. If you were judging progress by training loss you would conclude the run is going *great*. The only signals that catch it are (a) validation loss diverging from a single-GPU baseline, and (b) the assertion we build at the end of this post, which would show the ranks are computing suspiciously correlated gradients. This is why the discipline of always comparing an eight-GPU run against a one-GPU baseline for the first few hundred steps is not optional — it is the cheapest insurance in distributed training.

## Seeding: the two requirements everyone conflates

Seeding in distributed training is where careful engineers most often go wrong, because the intuitive move — "set the same seed everywhere so the run is reproducible" — is *half right and half wrong*, and the two halves pull in opposite directions. There are genuinely two separate requirements, and the reason people get it wrong is that they collapse them into one.

**Requirement one: model initialization must be identical across ranks.** All eight ranks must start from bit-for-bit the same weights. If they do not, then rank 0 and rank 3 are training *different models*, DDP's gradient all-reduce averages gradients that were computed against different parameters, and the whole premise of data parallelism — that every rank holds a replica of one model — is violated. The model diverges or trains garbage.

**Requirement two: data-side randomness must *differ* across ranks.** Dropout masks, data augmentation (random crops, flips, color jitter, mixup), any stochastic layer — these should be *decorrelated* between ranks. If every rank uses the identical RNG stream, then every rank applies the *same* dropout mask to its (different) data and the *same* augmentation choices, which wastes the regularizing diversity that these techniques are supposed to provide. You wanted eight independent noisy views; you got the same noise pattern applied eight times. It is not a correctness bug in the way the sampler bug is — the gradients are still valid — but it quietly weakens your regularization and gives you a slightly worse, slightly more overfit model than you should have gotten from eight GPUs.

The naive "same seed everywhere" satisfies requirement one (identical init) but *violates* requirement two (it correlates all the noise). The equally-naive "different seed per rank" satisfies requirement two but appears to *violate* requirement one — which is why people are scared to do it. The resolution is the fact we set aside at the very start of this post.

![a before and after of seeding showing one seed everywhere giving correlated dropout versus a rank-offset data seed with DDP broadcasting the initial weights](/imgs/blogs/ddp-internals-and-gotchas-6.webp)

**DDP already handles requirement one for you.** When you wrap the model, DDP's constructor *broadcasts* the parameters and buffers from rank 0 to every other rank. So even if rank 3 initialized its weights with a different seed and they started out different, the instant you wrap in `DistributedDataParallel`, rank 0's weights are copied over everyone else's, and all ranks are identical from step zero. That means you are *free* to use a different seed on each rank for everything else, because the one thing that had to be identical is made identical by the wrapper, not by your seed. The before/after figure shows the resolution: on the left, one seed everywhere buys you correlated dropout and a wasted ensemble; on the right, DDP broadcasts the init while a rank-offset seed decorrelates the data-side randomness.

The correct pattern is therefore: offset the seed by rank, and let DDP synchronize the weights.

```python
import os, random, numpy as np, torch

def seed_everything(base_seed: int, rank: int):
    # Rank-offset seed: decorrelates dropout / augmentation / sampling across ranks.
    seed = base_seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Order matters:
seed_everything(base_seed=42, rank=rank)     # per-rank RNG
model = build_model().to(local_rank)         # ranks init to DIFFERENT weights -- that's fine
model = torch.nn.parallel.DistributedDataParallel(
    model, device_ids=[local_rank],
)   # <-- broadcast here makes all ranks identical, satisfying requirement one

# DataLoader workers need per-worker, per-rank seeds too, or all workers
# on a rank share an RNG stream and produce correlated augmentation:
def worker_init_fn(worker_id):
    worker_seed = 42 + rank * 1000 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

loader = DataLoader(dataset, sampler=sampler, num_workers=8,
                    worker_init_fn=worker_init_fn, pin_memory=True)
```

Two details make this robust. First, the DataLoader's *worker* processes each need their own seed, because augmentation usually runs inside the workers; without `worker_init_fn`, all workers on a rank can share the same NumPy seed and produce correlated augmentation even within a rank. Second, if you genuinely need bit-exact reproducibility of the whole run (for debugging a divergence), that is a *third* concern — full determinism — which trades throughput for exactness and is its own topic; we cover it in [determinism across ranks](/blog/machine-learning/distributed-training/determinism-across-ranks) and in the debugging series' [reproducibility and determinism in training](/blog/machine-learning/debugging-training/reproducibility-and-determinism-in-training). For normal training you want *decorrelated* randomness, not *identical* randomness, and the rank-offset pattern above is what you want.

## BatchNorm only sees one eighth of the batch

There is one more silent-correctness trap, and it is specific to models with **BatchNorm** — which is to say CNNs, detectors, segmentation networks, and anything from the vision world, though not transformers, which use LayerNorm and are immune to this. BatchNorm normalizes each activation using the mean and variance *computed over the current batch*. On one GPU with a batch of 256, those statistics are estimated over 256 samples. On eight GPUs with DDP, each rank runs its own forward pass over its own per-GPU batch of 32, and BatchNorm on each rank computes statistics over only *its* 32 samples — one eighth of the global batch. DDP synchronizes gradients, but it does *not* synchronize BatchNorm's forward-pass statistics, so each rank is normalizing against a noisier, smaller-sample estimate.

Why this matters is a straightforward statistics fact. The variance of an estimated mean over `m` samples scales as ${1/m}$, so an estimate over `B/N` samples has variance proportional to $N/B$ — eight times noisier at eight ranks than the single-GPU estimate over `B`. When the per-GPU batch is comfortably large (say 64 or more), that extra noise is tolerable and plain BatchNorm is fine. When the per-GPU batch is *small* — 2, 4, 8, which is exactly the regime you land in for high-resolution detection and segmentation, where memory forces tiny per-GPU batches — the per-rank statistics are so noisy that BatchNorm becomes actively harmful, and models train worse on eight GPUs than on one.

The fix is `SyncBatchNorm`, which replaces each BatchNorm layer with a version that all-reduces the batch statistics across ranks in the forward pass, so every rank normalizes against the *global* batch of `B` samples exactly as if it had computed them on one enormous GPU:

```python
import torch

# Convert every BatchNorm in the model to its synchronized version.
# Do this BEFORE wrapping in DDP.
model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
```

`SyncBatchNorm` is not free: it adds an all-reduce of the mean and variance for every BN layer in both the forward and the backward pass, which is real communication on the critical path. So it is a genuine trade-off, and the decision is entirely about per-GPU batch size.

| Per-GPU batch | Plain BatchNorm | SyncBatchNorm |
|---|---|---|
| 64+ | Fine — statistics are stable | Unnecessary comms overhead |
| 16–32 | Usually fine; watch val loss | Small quality gain, small cost |
| 2–8 | Statistics too noisy; hurts | Recommended — recovers global-batch stats |
| 1 | Degenerate — variance undefined | Required if you use BN at all |

For transformers, which use LayerNorm (a per-sample normalization that needs no cross-batch statistics), none of this applies — LayerNorm on one eighth of the batch is identical to LayerNorm on the whole batch, because it never looks across samples. This is one of the underrated reasons transformers are so much friendlier to distribute than convnets: their normalization has no hidden cross-rank dependency to get wrong.

### broadcast_buffers: the running stats that sync behind your back

There is a related DDP internal that surprises people the first time they trip on it, and it is a natural companion to the BatchNorm story. DDP has a constructor argument `broadcast_buffers` that defaults to `True`, and what it does is re-broadcast the module's *buffers* — not parameters, buffers, which includes BatchNorm's `running_mean` and `running_var` — from rank 0 to every other rank at the *start of every forward pass*. The intent is benign: it keeps non-parameter state in sync so all ranks agree on the running statistics used at inference. But it has two consequences worth knowing.

First, it means BatchNorm's running statistics on ranks 1 through 7 are *thrown away and overwritten by rank 0's* on every forward. Only rank 0's running estimate actually accumulates across the run; the others are cosmetic. That is usually what you want, but it interacts with `SyncBatchNorm` (which computes the batch statistics jointly and does not need this rebroadcast) and with any custom buffer you register expecting it to hold per-rank state — a per-rank counter stored as a buffer will be silently reset to rank 0's value every step. Second, the rebroadcast is a real collective on the critical path of every forward. For models with many buffers it is a small but non-zero cost, and if you have no buffers that need syncing (a pure-LayerNorm transformer with no running statistics anywhere), you can set `broadcast_buffers=False` and reclaim it. The rule: leave it on for BatchNorm models, and consider turning it off for transformers where nothing in the model carries running state that needs synchronizing.

## The all-reduce you forgot to suppress: gradient accumulation

One more trap belongs with the internals, because it is pure wasted communication and it is invisible until you look at your scaling numbers. When your global batch will not fit even split across ranks, you accumulate gradients over several micro-steps before stepping the optimizer — run backward `k` times, then step once. On a single GPU that is free; you just do not call `optimizer.step()` until the last micro-step. On DDP it is a landmine, because DDP's hooks fire an all-reduce *at the end of every backward*, so a naive `k`-step accumulation does `k` all-reduces when it only needed one — the intermediate `k-1` reductions are averaging partial gradients you are about to add to anyway, which is both wrong-in-spirit and a straight `k`-fold inflation of your communication cost.

The fix is `model.no_sync()`, a context manager that tells DDP's hooks *not* to fire the collective. You wrap every micro-step except the last one in it, and let the final backward — outside the context — do the single all-reduce over the fully accumulated gradient:

```python
import contextlib

accum_steps = 4
for i, batch in enumerate(loader):
    is_last_micro = (i + 1) % accum_steps == 0
    # Suppress the all-reduce on all micro-steps except the last of the group.
    ctx = model.no_sync() if not is_last_micro else contextlib.nullcontext()
    with ctx:
        loss = loss_fn(model(batch.x), batch.y) / accum_steps
        loss.backward()
    if is_last_micro:
        optimizer.step()      # one all-reduce happened on this backward only
        optimizer.zero_grad(set_to_none=True)
```

Get this wrong and nothing crashes and the loss looks fine — you simply pay four times the communication for a four-step accumulation, which shows up as mysteriously poor scaling efficiency and a job that is communication-bound for no visible reason. It is the same bucket-and-hook machinery from the top of this post: `no_sync()` just leaves the hooks disarmed until the step where you actually want to synchronize. One more subtlety: you still divide the loss by `accum_steps` so the accumulated gradient is the *mean* over the whole effective batch, not the sum, or your effective learning rate silently scales with `accum_steps`.

## The loss looks fine but the model is worse: how to catch it

We have now met a whole family of bugs — the sampler coverage bug, correlated seeding, per-rank BatchNorm, a silently-wrong `static_graph`, an in-place edit through `gradient_as_bucket_view` — that share one terrifying property: **they do not crash, and the training loss looks reasonable.** They produce a model that is subtly worse than it should be, and if you are only watching training loss you will ship it. The gotcha table gathers them in one place.

![a table of six multi-GPU bugs listing the visible symptom the underlying cause and the one-line fix for each](/imgs/blogs/ddp-internals-and-gotchas-7.webp)

The table is a map of the whole post: each row is a bug that is invisible on one GPU, its symptom, its cause, and its fix. What ties them together — and what tells you how to *catch* them — is that every one of them is a violation of an invariant that is easy to state and easy to check. In correct data-parallel training, three things must hold, and each is a cheap assertion you can add to your training loop.

**Invariant one: all ranks hold identical parameters after every optimizer step.** This is the master invariant — if it holds, DDP is doing its job. It is also the one that catches the widest range of bugs, because divergence between ranks is the downstream symptom of almost everything: a bad all-reduce, a rank that skipped a step, a `static_graph` replaying the wrong plan. You check it by computing a checksum of all parameters on each rank and asserting they match:

```python
import torch
import torch.distributed as dist

@torch.no_grad()
def assert_params_in_sync(model, rtol=0, atol=0):
    """Assert every rank holds bit-identical parameters. Cheap enough to run
    every N steps as a tripwire. Compares each rank's checksum against rank 0."""
    flat = torch.cat([p.detach().reshape(-1).double()
                      for p in model.parameters()])
    checksum = flat.sum()                      # one scalar per rank
    # Gather rank 0's checksum onto every rank via broadcast, then compare.
    ref = checksum.clone()
    dist.broadcast(ref, src=0)                 # ref == rank 0's checksum everywhere
    max_abs = (checksum - ref).abs()
    dist.all_reduce(max_abs, op=dist.ReduceOp.MAX)
    if max_abs.item() > atol:
        raise RuntimeError(
            f"[rank {dist.get_rank()}] parameters diverged across ranks: "
            f"max checksum delta = {max_abs.item():.3e}. DDP is not keeping "
            f"replicas in sync -- check gradients, static_graph, or a manual "
            f".grad edit.")
```

Run this every few hundred steps — it is a single reduction of a scalar, so it costs essentially nothing — and it converts the "subtly worse" bug from something you discover a week later on the validation set into something that fails loudly at step 500. A checksum is not a perfect hash (two different parameter sets could in principle sum to the same value), so for a stricter check you can reduce a norm or an XOR of the raw bytes; in practice a double-precision sum catches every real divergence, because real divergence moves many parameters by large amounts.

**Invariant two: the gradient your ranks all-reduce is the average of genuinely different per-rank gradients.** If the sampler bug has struck and every rank sees the same data, then every rank's pre-all-reduce gradient is *identical*, which you can detect: gather each rank's gradient norm before the all-reduce and confirm they differ. Identical gradient norms across ranks at every step is the fingerprint of the "every rank sees the same shard" bug.

```python
@torch.no_grad()
def log_pre_reduce_grad_norms(model):
    """Call inside no_sync() or before backward's reduction completes to see
    per-rank gradients. If all ranks report identical norms, your ranks are
    seeing identical data -- the DistributedSampler is misconfigured."""
    local_norm = torch.sqrt(sum((p.grad.detach()**2).sum()
                                 for p in model.parameters()
                                 if p.grad is not None))
    world = dist.get_world_size()
    gathered = [torch.zeros_like(local_norm) for _ in range(world)]
    dist.all_gather(gathered, local_norm)
    if dist.get_rank() == 0:
        norms = [g.item() for g in gathered]
        spread = max(norms) - min(norms)
        print(f"per-rank grad norms: {[f'{n:.4f}' for n in norms]} "
              f"(spread={spread:.2e})")
        if spread < 1e-6:
            print("WARNING: per-rank gradients are identical -- "
                  "ranks are seeing the same data. Check DistributedSampler(rank=...).")
```

**Invariant three: the eight-GPU loss curve matches the one-GPU loss curve (after accounting for batch size).** This is the integration test that catches everything the assertions miss. Train the *same* configuration on one GPU and on eight GPUs for a few hundred steps, holding the *global* batch size and learning rate schedule constant (so the one-GPU run uses gradient accumulation to match the eight-GPU global batch). The two loss curves should track each other closely — not bit-identical, because the per-rank randomness differs, but statistically indistinguishable. If the eight-GPU curve is systematically lower (overfitting to a data shard) or systematically higher (broken gradients, wrong BatchNorm), you have a distributed bug, and you have caught it in an hour of debugging instead of a week of wasted training. This one-GPU-versus-N-GPU comparison is the single most valuable habit in this entire post, and the broader debugging workflow around it is covered in [debugging DDP and multi-GPU](/blog/machine-learning/debugging-training/debugging-ddp-and-multi-gpu).

These three invariants are the difference between hoping your distributed run is correct and knowing it. The parameter checksum catches divergence, the per-rank gradient-norm spread catches the data-coverage bug, and the one-GPU baseline catches everything else. Wire all three into a debug mode you can flip on for the first few hundred steps of any new multi-GPU configuration, and the entire "subtly worse" family of bugs stops being able to reach production. The `no_sync()` performance trap from the previous section belongs to the same family — a fourth thing that does not crash and does not error, caught by watching scaling efficiency rather than loss — which is why the honest measurement of tokens per second, not just the shape of the loss curve, is the north star this series keeps returning to.

## Case studies and real numbers

These are not hypothetical footguns. They show up in the design of the tools and in published training recipes, and it is worth grounding the post in a few real references.

**The DDP design itself.** The canonical description of how DDP buckets gradients and overlaps the all-reduce with backward is Li et al., *"PyTorch Distributed: Experiences on Accelerating Data Parallel Training"* (VLDB 2020) — the paper behind `torch.nn.parallel.DistributedDataParallel`. It documents the bucketing, the gradient-ready hooks, and the reverse-order bucket assignment we described, and it reports that with bucketing and overlap, DDP sustains high scaling efficiency (in the ~90 percent range on well-connected nodes) where a naive all-reduce-after-backward does not. The unused-parameter machinery and `find_unused_parameters` exist precisely because the hook-counting design cannot, on its own, tell the difference between "a gradient is late" and "a gradient is never coming." That is a design consequence, not a bug, and the paper is the primary source for the mechanism this whole post depends on.

**SyncBatchNorm in detection.** The value of synchronizing BatchNorm across GPUs was made concrete by Peng et al., *"MegDet: A Large Mini-Batch Object Detector"* (CVPR 2018), which introduced cross-GPU BatchNorm ("Cross-GPU Batch Normalization") to train detectors with large effective batches across many GPUs where each GPU holds only one or two images. With per-GPU batches that small, per-rank BatchNorm statistics are hopeless, and synchronizing them across the whole batch was a measurable accuracy improvement and a training-time reduction. This is the empirical backing for the small-per-GPU-batch row of the SyncBatchNorm table: it is not a theoretical nicety, it is the difference between a detector that trains and one that does not.

**The sampler and set_epoch contract.** The PyTorch `DistributedSampler` documentation states outright that `set_epoch` must be called at the start of each epoch or the shuffling order will be the same across epochs, and the reference training scripts in torchvision call it in exactly the place we showed. The coverage bug — every rank on the same shard — is common enough that it is a recurring item in the PyTorch forums and issue tracker; the fix is always the same, construct the sampler after `init_process_group` and pass `rank` explicitly.

**Hardware for the numbers.** The worked-example timings above assume A100 80GB SXM cards (roughly 312 dense bf16 TFLOP/s, about 2.0 TB/s of HBM2e bandwidth) connected by NVLink (on the order of 600 GB/s of aggregate GPU-to-GPU bandwidth on an A100 node). The exact millisecond figures depend on your model, kernel efficiency, and NCCL version, so treat them as order-of-magnitude illustrations rather than benchmarks you can quote — the *ratios* (a few percent for the traversal tax, 5 to 10 percent for `static_graph`, an all-reduce per BN layer for SyncBatchNorm) are the durable part, and they are what should drive your decisions. When a headline number matters, measure it on your own hardware with `torch.cuda.synchronize()` before and after timing, a warm-up loop to reach steady state, and enough steps to average out the data-loader jitter.

## When to reach for each fix, and when not to

Every flag in this post is a cost, and the discipline is to turn on exactly the ones your model needs and no more. Here is the decisive version.

| Setting | Turn it on when | Leave it off when |
|---|---|---|
| `find_unused_parameters=True` | Forward has genuinely dynamic control flow — the set of used parameters changes per step | The graph is static; fix the model or use `static_graph` instead |
| `static_graph=True` | The computation is identical every step (most transformer/CNN training); or you use activation checkpointing or reused/tied weights | Forward has data-dependent branches (MoE routing, early exit, curriculum switches) |
| `gradient_as_bucket_view=True` | Almost always — especially large models where the duplicate gradient buffer is expensive | You do custom in-place `.grad` surgery you cannot refactor |
| `DistributedSampler` | Always, for multi-rank training | Never — if you skip it your ranks see the same data |
| `sampler.set_epoch(epoch)` | Every epoch, always | Never skip it — the shuffle freezes otherwise |
| Rank-offset data seed | Always — decorrelates dropout and augmentation | Never use the same seed on every rank for data |
| `SyncBatchNorm` | Model has BatchNorm *and* per-GPU batch is small (roughly under 16) | Transformers (LayerNorm), or large per-GPU batches where the comms cost is not worth it |
| `model.no_sync()` | During gradient-accumulation micro-steps | On the final accumulation step, and when not accumulating |

Two "when not to" points deserve emphasis because they are where people over-engineer. First, do not reach for `find_unused_parameters=True` as a reflexive safety blanket — it is a per-step tax, and if your model is static (the common case) you are paying it for nothing; the correct move for a static graph with genuinely unused parameters is `static_graph=True`, which pays the cost once. Second, do not enable `SyncBatchNorm` on a transformer or on a model with a comfortable per-GPU batch — it adds real communication for a statistical problem you do not have, and on LayerNorm models it does nothing at all because there is no cross-sample statistic to synchronize. The whole art here is matching the fix to the actual failure, which is why the assertions from the previous section matter so much: they tell you *which* invariant is broken, so you can apply the one fix that addresses it instead of turning on every flag and hoping.

And the meta-point that ties back to the series frame: none of these are exotic large-scale problems. They bite at *eight* GPUs — one node, the smallest multi-GPU setup there is — long before you ever touch multi-node, tensor parallelism, or FSDP. Data parallelism is the foundation every other parallelism strategy is layered on top of, so a DDP correctness bug does not stay contained; it corrupts everything built above it. Getting these eight things right is the price of admission to distributed training, and the assertion that all ranks hold identical parameters is the tripwire that keeps you honest about it. The full decision-and-debugging checklist that composes these with the rest of the series lives in [the distributed training playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook).

## Key takeaways

- **DDP fires a bucket's all-reduce only when every parameter in it has produced a gradient.** A parameter that gets no gradient — a frozen head, an untaken branch — leaves its bucket forever incomplete, and the job hangs or errors. This one mechanism explains most DDP mysteries.
- **`find_unused_parameters=True` fixes the hang by walking the autograd graph every step**, which costs a few percent of step time. Prefer fixing the model, or `static_graph=True` when the unused set is static.
- **`static_graph=True` records the graph once and replays it**, buying back the traversal tax and improving overlap — but it is *unsafe* with data-dependent control flow, where it silently produces wrong gradients.
- **`gradient_as_bucket_view=True` aliases `.grad` into the bucket buffer**, saving a copy and a duplicate allocation. The one rule: never mutate `.grad` in place or hold references across steps — clone first.
- **Without `DistributedSampler(rank=...)` your ranks train on the same data.** The worst version — sampler built without `rank` — pins every rank to shard 0, so you train on one eighth of your dataset and overfit while the training loss looks great.
- **Always call `sampler.set_epoch(epoch)`**, or the shuffle is identical every epoch.
- **Seeding is two requirements: identical init, decorrelated data randomness.** DDP broadcasts the weights at wrap time, so init is handled for free — which frees you to offset the data seed by rank. Same seed everywhere correlates dropout and augmentation and wastes the ensemble.
- **BatchNorm computes statistics over only the per-GPU batch.** When that batch is small, convert to `SyncBatchNorm`. Transformers use LayerNorm and are immune.
- **Assert that all ranks hold identical parameters after a step.** A double-precision parameter checksum, reduced across ranks every few hundred steps, turns the "subtly worse" bug into a loud failure at step 500 instead of a bad model a week later.
- **Compare the eight-GPU loss curve against a one-GPU baseline** for the first few hundred steps. It is the cheapest and most powerful distributed-correctness test there is.

## Further reading

- [Why distributed training](/blog/machine-learning/distributed-training/why-distributed-training) — the four walls and the map of the whole series.
- [DDP from first principles](/blog/machine-learning/distributed-training/ddp-from-first-principles) — the gradient all-reduce, bucketing, and compute-communication overlap that this post debugs.
- [Your first multi-GPU run](/blog/machine-learning/distributed-training/your-first-multi-gpu-run) — rank, local rank, world size, and the `torchrun` launch these bugs sit on top of.
- [Determinism across ranks](/blog/machine-learning/distributed-training/determinism-across-ranks) — when you need bit-exact reproducibility instead of merely decorrelated randomness.
- [The distributed training playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook) — the capstone decision-and-debugging checklist for the whole series.
- [Debugging DDP and multi-GPU](/blog/machine-learning/debugging-training/debugging-ddp-and-multi-gpu) and [reproducibility and determinism in training](/blog/machine-learning/debugging-training/reproducibility-and-determinism-in-training) — the broader debugging workflows these correctness checks plug into.
- Li et al., *"PyTorch Distributed: Experiences on Accelerating Data Parallel Training"* (VLDB 2020) — the DDP design paper: bucketing, gradient hooks, overlap.
- Peng et al., *"MegDet: A Large Mini-Batch Object Detector"* (CVPR 2018) — the empirical case for synchronizing BatchNorm across GPUs at small per-GPU batch.
- The PyTorch `DistributedDataParallel` and `DistributedSampler` documentation — the authoritative reference for `find_unused_parameters`, `static_graph`, `gradient_as_bucket_view`, and the `set_epoch` contract.
