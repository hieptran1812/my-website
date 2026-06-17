---
title: "Debugging DDP and Multi-GPU: Gradient Sync, Unused Params, and the 8x-GPUs-Same-Speed Trap"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Localize and fix the bugs that only appear once one GPU becomes eight: silent gradient-sync errors, the find_unused_parameters hang, rank desync deadlocks, duplicated data, BatchNorm across ranks, and the run that uses eight GPUs at the speed of one."
tags:
  [
    "debugging",
    "model-training",
    "distributed-training",
    "ddp",
    "multi-gpu",
    "pytorch",
    "finetuning",
    "deep-learning",
    "nccl",
    "throughput",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/debugging-ddp-and-multi-gpu-1.png"
---

You added seven GPUs to a run that worked on one, and something is wrong. Maybe the run hangs at the first `loss.backward()` and sits there, silent, until a NCCL watchdog kills it half an hour later. Maybe it trains fine but the final validation accuracy is two points *worse* than the single-GPU baseline you were trying to speed up. Or maybe — the most demoralizing one — it runs perfectly, every GPU pegged at 100% in `nvidia-smi`, and the wall-clock time per epoch is *exactly the same* as it was on one card. You paid for eight GPUs and got the throughput of one.

Welcome to distributed data-parallel (DDP) debugging, where an entire new class of bugs exists that simply cannot happen on a single device. On one GPU, a gradient is a gradient; there is no sync, no rank, no sharding, no collective operation that can deadlock. The moment you wrap your model in `DistributedDataParallel` and launch it with `torchrun --nproc_per_node=8`, you have signed up for at least eight new ways for the run to lie to you: the data each rank sees, the seeds each rank uses, whether every parameter gets a gradient, whether BatchNorm sees a meaningful batch, whether your logging and checkpointing corrupt each other, and whether the communication that glues the ranks together is overlapping with compute or strangling it.

![A vertical stack showing how a single-GPU baseline gains eight new failure layers under DDP including data sharding, per-rank seeds, gradient all-reduce, unused parameters, local BatchNorm, rank-0 logging, and a comm bottleneck](/imgs/blogs/debugging-ddp-and-multi-gpu-1.png)

This post is a field guide to those bugs. We will keep the same spine the rest of this series uses: a training bug hides in one of six places — **data, optimization, model code, numerics, systems, evaluation** — and you *bisect* to the right one before you touch code, using two master tools, **make-it-fail-small** and **read the instruments**. DDP is squarely a *systems* problem, but it has a sharp twist: the single most important DDP debugging move is to **reproduce on one GPU first**. If your run fails the same way on a single GPU with the same data, it was never a DDP bug — it is an ordinary data, optimization, or model bug wearing a distributed costume, and you should go debug it on one device where the instruments are clean. Only when a bug *appears or changes* when you go from one GPU to N is it truly a distributed bug worth chasing here. That one branch cuts your search space in half before you have read a single line of NCCL code.

By the end you will be able to take a multi-GPU run that hangs, runs slow, or quietly loses accuracy, and localize the cause in minutes: confirm gradient sync by comparing a one-GPU and N-GPU step on identical data; catch an unused parameter before it hangs the reducer; prove your data is sharded and not duplicated; measure how much of each step is communication versus compute; decide whether you need SyncBatchNorm; and guard your logging and checkpointing so eight ranks do not stomp on each other. We will derive the *why* — data-parallel SGD is gradient averaging, which is mathematically a bigger batch, and that single fact explains the learning-rate scaling, the BatchNorm trap, and the duplicated-data bug — write the *runnable diagnostics*, and show *before→after numbers* for each fix.

## 1. The science: data-parallel SGD is gradient averaging

Before any debugging, you need the one equation that makes every DDP bug predictable. Strip away the engineering and DDP is doing one thing: it computes the gradient of your loss on several disjoint slices of the batch in parallel, then *averages those gradients* so every replica takes the same step. That average is what makes N GPUs behave (almost) like one big batch.

Write the mini-batch loss on a batch $B$ as the mean per-example loss:

$$
L(\theta; B) = \frac{1}{|B|} \sum_{i \in B} \ell(\theta; x_i).
$$

Its gradient is the mean of the per-example gradients, $g(B) = \frac{1}{|B|}\sum_{i\in B} \nabla_\theta \ell(\theta; x_i)$. Now split $B$ into $R$ disjoint shards $B_1, \dots, B_R$ of equal size $b = |B|/R$, one per rank. The gradient that rank $r$ computes locally is $g_r = \frac{1}{b}\sum_{i \in B_r} \nabla_\theta \ell(\theta; x_i)$. Averaging the local gradients across ranks gives:

$$
\frac{1}{R}\sum_{r=1}^{R} g_r = \frac{1}{R}\sum_{r=1}^{R} \frac{1}{b}\sum_{i \in B_r} \nabla_\theta \ell = \frac{1}{R b}\sum_{i \in B} \nabla_\theta \ell = g(B).
$$

This is the whole game. **The average of per-rank mean-gradients equals the gradient of the full batch**, *provided* (1) each shard has the same number of examples $b$, and (2) the shards are disjoint and together cover $B$ exactly once. DDP implements that average with a collective communication operation called **all-reduce**: every rank sends its local gradient, the values are summed across ranks and divided by $R$, and the result is written back to every rank, so all replicas end up holding the identical averaged gradient and therefore take the identical optimizer step. Models stay in lockstep — they start from the same weights, apply the same gradient, and never drift apart.

![A branching dataflow graph showing four per-rank local gradients flowing into an all-reduce that averages them into one shared gradient applied identically on every rank](/imgs/blogs/debugging-ddp-and-multi-gpu-2.png)

Hold onto the two provisos, because almost every DDP correctness bug is a violation of one of them:

- **Equal-size shards.** If one rank has fewer examples (the last uneven shard, a rank that dropped a bad batch, a rank that broke out of the loop early), the average is no longer the full-batch gradient — and worse, a rank with *zero* steps left will never join the next all-reduce, and the others will block forever.
- **Disjoint, full coverage.** If you forget the `DistributedSampler`, every rank reads the *entire* dataset. The shards are not disjoint; they are eight identical copies. The all-reduce still runs, the math still averages, but it is averaging eight copies of the same gradient — so you do `8×` the compute to get the gradient signal of one pass over the data. The model trains, the loss goes down, and you quietly waste seven GPUs and overcount your epochs by `8×`.

There is one more consequence that trips up everyone, and it follows directly from the equation: **the effective batch size is `per_gpu_batch × num_gpus`**. If each GPU processes 32 examples and you have 8 GPUs, your effective batch is 256, not 32. Your optimizer is taking steps as if it saw 256 examples per update, because that is exactly what the averaged gradient represents. This is why moving a single-GPU recipe to 8 GPUs unchanged usually *underperforms*: your learning rate was tuned for batch 32, and you are now silently training at batch 256 with the same LR, which is too small for the larger batch. The standard fix is **linear LR scaling** — multiply the base LR by the number of GPUs (with warmup) — a rule we will revisit and which the sibling post on [gradient accumulation and effective batch](/blog/machine-learning/debugging-training/gradient-accumulation-and-effective-batch-bugs) treats in full, because accumulation and data-parallelism are the same "bigger batch" mechanism through two different doors.

Keep this equation in your head as the lens. When accuracy drops, ask: did I break disjointness (duplicated data) or equal-size (uneven shards)? When the run hangs, ask: which rank failed to reach the collective? When it is slow, ask: how much time is the all-reduce taking relative to the compute that produced the gradient? Every section below is an instance of this one mechanism going wrong.

### The cost of all-reduce (why communication is bounded, not free)

The averaging is correct, but it is not instantaneous, and to debug the throughput problems in section 8 you need a rough model of what the all-reduce *costs*. A naive all-reduce — every rank sends its full gradient to one rank, which sums and broadcasts back — would push $O(R)$ copies of the gradient through a single link and scale terribly. Modern implementations (NCCL's ring and tree algorithms) are far smarter. In a **ring all-reduce**, the $R$ ranks form a logical ring, the gradient is split into $R$ chunks, and the algorithm does $R-1$ steps where each rank simultaneously sends one chunk to its neighbor and receives another, accumulating partial sums, then another $R-1$ steps to propagate the final sums around. The beautiful property: the total data each rank sends is approximately $2 \cdot \frac{R-1}{R} \cdot M$ where $M$ is the gradient size in bytes — *independent of $R$ for large $R$* (it approaches $2M$). Every link is busy every step; nothing bottlenecks on a single node.

The practical consequence for debugging: the all-reduce time is dominated by $2M / \text{bandwidth}$ plus a per-collective latency term that grows slowly (logarithmically for tree, linearly-but-small for ring) with $R$. So the *gradient size* $M$ (your parameter count times bytes-per-grad) is the dominant factor in communication cost, and the per-GPU *compute* is what you have to hide it under. A 7-billion-parameter model in fp16 has roughly $M \approx 14$ GB of gradient to reduce *every step*; on a 600 GB/s NVLink fabric that is on the order of tens of milliseconds of pure communication per step, which is fine if each step's compute is hundreds of milliseconds and catastrophic if it is not. This is the quantitative version of "8 GPUs, same speed": when $2M/\text{bandwidth}$ approaches the compute time, you are comm-bound, and the cure is to grow the compute (bigger per-GPU batch) or shrink/overlap the communication — never "add more GPUs," which leaves $M$ unchanged while making compute-per-step smaller.

This also explains why **gradient bucketing** exists (section 4): rather than waiting for the entire gradient to be ready and then doing one giant all-reduce after the backward finishes, DDP slices the parameters into buckets (default ~25 MB each) and fires each bucket's all-reduce as soon as that bucket's gradients are computed, *during* the backward. The later layers' gradients reduce while the earlier layers are still computing. That overlap is the difference between communication adding to every step and communication being nearly free — and it is exactly what an unused parameter, or the wrong `find_unused_parameters` setting, breaks.

## 2. The first move: reproduce on ONE GPU

If you remember one sentence from this post, make it this: **before you debug anything in DDP, reproduce it on a single GPU with the same data and the same seed.** This is the multi-GPU version of the series' make-it-fail-small principle, and it is the highest-leverage step by a wide margin, because it answers the only question that matters first — *is this even a distributed bug?*

![A decision tree branching on whether the run fails on one GPU, splitting ordinary training bugs from true distributed bugs that only appear on N GPUs](/imgs/blogs/debugging-ddp-and-multi-gpu-8.png)

The logic is a clean bisection. There are two possibilities:

1. **It fails the same way on one GPU.** Then it is *not* a DDP bug. It is a data, optimization, model, numerics, or evaluation bug that exists regardless of device count, and you have been blaming NCCL for a bug in your loss function. Go debug it on one GPU, where you have `pdb`, clean grad-norm instruments, and no collective deadlocks to confuse you. The series' single-device tools all apply — overfit one batch, read the loss curve, print grad norms — and you will find it ten times faster.
2. **It only fails on N GPUs, or fails *differently*.** Now you have isolated a true distributed bug, and the rest of this post is your map. The bug lives in the new layers DDP introduced: sharding, seeds, gradient sync, unused params, BatchNorm, logging, or communication.

Running on one GPU under the same launcher is trivial: `torchrun --nproc_per_node=1 train.py`. Your DDP wrapper still works (a world size of 1 is a degenerate, valid DDP group — the all-reduce of one tensor is itself), so you do not have to rip out the distributed code. You just shrink the world to one rank and see if the symptom survives.

```python
# repro_on_one_gpu.py — run with: torchrun --nproc_per_node=1 repro_on_one_gpu.py
import os, torch, torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

def setup():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def main():
    rank = setup()
    torch.manual_seed(0)                     # SAME seed as the N-GPU run
    model = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 10)).cuda(rank)
    model = DDP(model, device_ids=[rank])
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    # One fixed batch, identical bytes on every world size you test.
    g = torch.Generator().manual_seed(0)
    x = torch.randn(32, 128, generator=g).cuda(rank)
    y = torch.randint(0, 10, (32,), generator=g).cuda(rank)

    for step in range(5):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        if rank == 0:
            print(f"step {step}  loss {loss.item():.6f}")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

Run this with `--nproc_per_node=1` and record the five loss values. Then run the *exact same script* with `--nproc_per_node=8` (the fixed batch is replicated, so each rank sees the identical 32 examples for this test). If the bug is a deadlock that shows up only at 8, you have proven it is distributed. If the bug — say, an immediate NaN — shows up at 1 too, you have proven it is not. We will reuse this skeleton as the grad-sync correctness probe in section 4.

A practical note on the launcher. `torchrun` is the modern entry point (it replaced `python -m torch.distributed.launch`); it sets `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, and the rendezvous environment variables for you. `LOCAL_RANK` is the GPU index *on this node* (0–7 on an 8-GPU box); `RANK` is the *global* index across all nodes; `WORLD_SIZE` is the total number of processes. On a single node these distinctions do not bite, but they matter the instant you go multi-node, and confusing `RANK` for `LOCAL_RANK` when you call `torch.cuda.set_device` is a classic way to put two processes on the same GPU and OOM. Always `set_device(LOCAL_RANK)`.

## 3. Gradient-sync correctness: getting the average right

The all-reduce is correct by construction — but only if *you* feed it correctly. Two mistakes break gradient sync silently, meaning the run trains and the loss goes down but the gradients are subtly wrong, which is the worst kind of bug because nothing crashes.

**Mistake 1: a non-mean loss reduction.** Recall the averaging equation depended on each rank computing a *mean* per-example gradient. If you write your loss with `reduction="sum"` instead of `reduction="mean"`, each rank's gradient is the *sum* over its $b$ examples, and the all-reduce then averages the sums: $\frac{1}{R}\sum_r \sum_{i\in B_r}\nabla\ell = \frac{1}{R}\sum_{i\in B}\nabla\ell$. That is the full-batch *sum* divided by $R$, not by $|B|$. Your effective gradient magnitude is off by a factor of $b$ (the per-GPU batch size) compared to what you would get on one GPU with a mean loss. The model still trains, but your effective learning rate is scaled by the per-GPU batch, so a recipe that was stable on one GPU can diverge on 8, or crawl, depending on which way the factor cuts. The fix is to use mean reduction consistently and let LR scaling handle the batch-size change explicitly. This is a cousin of the reduction bugs in the [loss-function bugs](/blog/machine-learning/debugging-training/loss-function-bugs) post; under DDP the consequence is amplified by the rank count.

**Mistake 2: skipping `backward()` on some ranks.** DDP installs the all-reduce as a side effect of the backward pass — the gradient hooks that trigger the collective fire *during* `loss.backward()`. So if one rank conditionally skips the backward (e.g., `if batch_is_bad: continue`), that rank never launches its all-reduce, the other ranks block waiting for it, and you deadlock. Every rank must call `backward()` every step, on a real loss, even if that means feeding a dummy batch or multiplying a bad loss by zero so it contributes nothing while still flowing through the graph. Symmetry of collectives is non-negotiable: **every rank must execute the same collective operations in the same order.**

#### Worked example: confirming gradient sync by comparing 1-GPU and N-GPU steps

Here is the test I run whenever I suspect the average is wrong. Take the same fixed batch, replicate it across all ranks, do one step, and compare the post-step weights to a single-GPU step on the *same total batch*. If the averaging is correct, the gradient each rank applies must equal the gradient a single GPU would compute on the concatenation of all the shards.

Concretely: on 4 GPUs, give rank $r$ the slice `B[r*8:(r+1)*8]` of a 32-example batch (per-GPU batch 8, effective batch 32). After `loss.backward()`, every rank's `model.module.fc.weight.grad` should be *identical* (the all-reduce made them so) and should equal the grad you get from a single GPU forward/backward on all 32 examples with a mean loss. I add this assertion and run it once:

```python
# After loss.backward() on each rank, before opt.step():
g = model.module[0].weight.grad
# 1) all ranks must agree (all-reduce worked):
g_list = [torch.empty_like(g) for _ in range(dist.get_world_size())]
dist.all_gather(g_list, g)
for other in g_list:
    assert torch.allclose(g, other, atol=1e-6), "grads differ across ranks -> sync broken"
# 2) the averaged grad must match a single-GPU full-batch grad (computed offline):
ref = torch.load("single_gpu_fullbatch_grad.pt").cuda()
print("max |ddp_grad - single_gpu_grad| =", (g - ref).abs().max().item())
```

When sync is healthy the printed difference is ~`1e-7` (floating-point noise from a different summation order). When someone has used a sum reduction, the difference is a clean factor: the DDP grad is `b×` the reference, here `8×`, an unmistakable signature. I have caught a "DDP slows convergence" complaint this way in five minutes — the difference was exactly `4.0×` on a 4-GPU run, which immediately said *sum reduction*, not *NCCL*. That is the value of a numeric test over staring at loss curves: the *ratio* names the bug.

A subtler grad-sync issue is **gradient accumulation interacting with DDP**. If you accumulate gradients over `k` micro-steps, you do *not* want DDP to all-reduce on every micro-step — that is `k×` the communication for no benefit, since you only step the optimizer once per `k` micro-steps. DDP gives you `model.no_sync()` to suppress the all-reduce on the intermediate micro-steps and sync only on the last one:

```python
for i, (x, y) in enumerate(loader):
    is_last_micro = (i % k) == (k - 1)
    ctx = model.no_sync() if not is_last_micro else nullcontext()
    with ctx:
        loss = criterion(model(x), y) / k     # scale so the sum over k micro-steps is a mean
        loss.backward()
    if is_last_micro:
        opt.step(); opt.zero_grad()
```

Forgetting `no_sync()` does not break correctness, but it can double or triple your communication cost — it is a throughput bug hiding inside a correctness-shaped API, and we will return to it in section 8.

One more grad-sync subtlety that surprises people: **DDP broadcasts the model's initial weights from rank 0 at construction time.** When you wrap a model in `DDP(...)`, it broadcasts rank 0's parameters and buffers to all other ranks, so every replica starts identical even if you seeded the ranks differently. This is why a per-rank seed for *augmentation* (section 6) does not desync the model — the weights are forced identical at wrap time, and they stay identical because every step applies the same averaged gradient. But it also means a parameter you create or modify *after* the DDP wrap (a late-initialized embedding, a buffer you reset by hand) will *not* be broadcast and can differ across ranks, silently breaking the lockstep. The rule: finalize your model's parameters and buffers *before* wrapping in DDP, and if you must touch them after, broadcast them yourself with `dist.broadcast(tensor, src=0)`. A diagnostic for "are my weights actually identical across ranks?" is to all-gather a hash of the flattened parameters once at step 0 and assert they match — a five-line check that catches a whole category of "the model drifted apart" bugs at their source.

## 4. The find_unused_parameters hang

This is the single most common DDP support question, and once you understand the mechanism you will never be confused by it again. The symptom is brutal and unmistakable: the run reaches `loss.backward()`, sits there with GPUs idle or spinning, and after roughly 30 minutes dies with a NCCL timeout, often with a message like `Watchdog caught collective operation timeout` or, if you are lucky, a clear error: `Expected to have finished reduction in the prior iteration before starting a new one ... you can enable unused parameter detection by passing find_unused_parameters=True`.

![A branching graph showing a forward pass that skips an auxiliary head, leaving one reducer bucket waiting for a gradient that never arrives until the NCCL timeout fires](/imgs/blogs/debugging-ddp-and-multi-gpu-3.png)

Here is what is actually happening. To overlap communication with computation, DDP does not wait for the whole backward pass to finish before it starts all-reducing. It groups parameters into **buckets** and launches the all-reduce for a bucket the moment *all* parameters in that bucket have received their gradients during backprop. This is the comm-compute overlap that makes DDP fast: while the later layers are still computing gradients, the earlier layers' gradients are already flying over the wire. The reducer keeps a count of how many parameters it is still waiting on per bucket.

Now suppose a parameter does *not* get a gradient on this forward pass — because the forward took a conditional branch that skipped a module, or because an output head was not used for this batch, or because a parameter is genuinely dead code. That parameter's gradient never arrives. The reducer's bucket count never reaches zero. The bucket's all-reduce never launches. And because every *other* rank may have used that parameter (or may also be stuck), the collective never completes on any rank. Everyone waits. The watchdog fires. Half an hour of GPU time, gone.

There are two fixes, and the difference between them is the difference between a junior and a senior debugging this.

**The blunt fix: `find_unused_parameters=True`.** This tells DDP to do an extra traversal of the autograd graph at the start of each backward to *find* which parameters will not get gradients, mark their buckets as "ready" preemptively, and proceed. It works. It is also a real cost: that graph traversal runs every iteration and adds overhead (often a few percent, sometimes more for large models), and it disables some bucketing optimizations. It is a band-aid, not a cure. Use it to *confirm* the diagnosis and to keep a run alive, not as a permanent answer.

**The real fix: make every parameter used, or remove it.** An unused parameter is almost always a bug or a design smell. If you have an auxiliary head that is only used for some batches, either always run it (and zero out its loss contribution when not needed, so it still gets a gradient of zero through a live path) or move it out of the DDP-wrapped module. If a branch is conditional, restructure so the conditional happens *inside* the loss, not by skipping a module. If a parameter is truly dead code, delete it. When every parameter participates in every forward, the reducer's counts always reach zero, the all-reduces always launch, and you get both correctness and full overlap performance with no `find_unused_parameters` tax.

#### Worked example: localizing the unused parameter in 60 seconds

Do not guess which parameter is unused. Ask PyTorch. After a backward pass (run it on **one GPU** first, per section 2 — an unused param is a model-code bug, not a distributed one, so it reproduces on a single device), print every parameter whose `.grad` is `None`:

```python
# Run on ONE GPU. After loss.backward(), before opt.step():
unused = [name for name, p in model.named_parameters()
          if p.requires_grad and p.grad is None]
print(f"{len(unused)} params got no gradient this step:")
for name in unused:
    print("   ", name)
```

On a real run that hung, this printed:

```bash
2 params got no gradient this step:
    aux_classifier.weight
    aux_classifier.bias
```

Instantly the diagnosis was clear: an auxiliary classifier head, added months ago for a deep-supervision experiment, was no longer wired into the loss but still lived in the module. On one GPU it was harmless — a parameter with no gradient just does not update. On 8 GPUs it hung the reducer every step. The fix was four characters: delete the unused head. Throughput went from "dead at 30 minutes" to a clean run, and we *did not* pay the `find_unused_parameters` overhead. Compare that to the lazy path — set `find_unused_parameters=True`, ship it, and silently carry a 3–8% perpetual slowdown forever.

The general rule: `find_unused_parameters=True` is a diagnostic and a crutch; `named_parameters()` with a `grad is None` check is the actual diagnosis; and "every parameter participates in every forward" is the cure.

There is a related optimization worth knowing for the case where your graph *is* fixed across iterations: **`static_graph=True`**. If every forward pass uses exactly the same set of parameters (no data-dependent branching), passing `static_graph=True` to the DDP constructor lets the reducer learn the graph structure on the first iteration and reuse it, which both removes the need for `find_unused_parameters` traversal *and* enables extra optimizations (it can re-order buckets optimally and supports some activation-checkpointing patterns that otherwise break). The catch is in the name: it is *static*, so if your graph actually varies between iterations, `static_graph=True` will produce wrong gradients or errors — it is a promise you are making to DDP. Use it only when the graph is genuinely fixed, which is the common case for a plain feed-forward model. The decision table:

| Situation | Setting | Cost | When |
|---|---|---|---|
| All params always used, fixed graph | defaults (`find_unused=False`) | none | the goal — restructure to get here |
| All params used, fixed graph, want max speed | `static_graph=True` | none, faster | feed-forward models, no branching |
| Some params conditionally unused | `find_unused_parameters=True` | per-iter traversal, 3–8% | temporary diagnosis or unavoidable branching |
| Param genuinely dead | delete it | none | the real fix |

The ranking is clear: delete or always-use the parameter (best), `static_graph=True` if the graph is fixed (fast), and `find_unused_parameters=True` only as a diagnostic or when conditional branching is genuinely required. Reaching for the flag first, and leaving it on forever, is the lazy default that quietly taxes every run you will ever launch from that script.

## 5. Rank desync and the NCCL deadlock

Section 3 covered one rank skipping a collective. The broader failure is **rank desync**: ranks executing *different sequences* of collective operations, which deadlocks because a collective only completes when every rank in the group has called it. NCCL collectives are barriers in disguise — `all_reduce`, `broadcast`, `all_gather`, `barrier` all require every peer to show up. If rank 3 calls its 413th all-reduce while ranks 0–2 are calling their 414th, the group is mismatched and everyone blocks until the watchdog (default ~30 minutes, set by `TORCH_NCCL_BLOCKING_WAIT` / the timeout you pass to `init_process_group`) kills it.

![A left-to-right timeline showing ranks in sync until uneven shards make rank 3 exit its loop early, after which the remaining ranks block on a collective until the NCCL watchdog kills the job](/imgs/blogs/debugging-ddp-and-multi-gpu-5.png)

The most common cause is **uneven data across ranks causing a different number of steps**. If your dataset has 1000 batches and you shard it across 3 GPUs without care, one rank may get 334 batches and the others 333. The rank with 334 calls one extra all-reduce on the last step; the others have already exited their loop; deadlock. This is exactly the "equal-size shards" proviso from section 1 biting at the systems level. The `DistributedSampler` with `drop_last=True` (or its built-in padding behavior) is what prevents it — it ensures every rank gets the same number of samples, padding by repeating a few examples if needed so the per-rank batch count is identical. We will set it up properly in the next section.

Other desync causes, each a mismatch in the *sequence* of collectives:

- **A rank that `break`s early** on a per-rank condition (early stopping evaluated independently on each rank, a rank that hit an exception and exited its loop). The decision to stop must be *collective* — compute it on rank 0 and broadcast it, or all-reduce a "should I stop" flag so every rank agrees.
- **Conditional collectives.** A `dist.barrier()` or logging-related all-gather inside an `if rank == 0:` block, or inside a data-dependent branch that differs across ranks. Every collective must be reached by every rank, unconditionally, in the same order.
- **An exception on one rank.** If rank 5 throws (a CUDA OOM, a bad batch) and dies, ranks 0–4 do not know; they block on the next collective forever. This is why a single OOM on one rank manifests as a *hang*, not a clean crash — the dead rank cannot tell the others it is gone.

The diagnostic for a hang is to figure out *which rank is where*. The cheapest tool is a per-rank heartbeat log: have every rank print its step number and a timestamp, prefixed with its rank, and flush. When it hangs, the last lines tell you instantly that rank 3 stopped logging at step 412 while ranks 0–2 reached 413 — that gap *is* the desync.

```python
import logging, sys
rank = dist.get_rank()
logging.basicConfig(
    format=f"[rank {rank}] %(asctime)s %(message)s",
    level=logging.INFO, stream=sys.stdout, force=True)
# inside the loop:
logging.info(f"step {step} start")     # flush=default for logging to stdout
```

For production multi-node runs, PyTorch ships heavier instrumentation: setting `TORCH_NCCL_DEBUG=INFO` and enabling **flight recorder** (`TORCH_NCCL_TRACE_BUFFER_SIZE`) dumps the in-flight collective state when a timeout fires, so you can see exactly which collective each rank was stuck on and which rank failed to arrive. But on a single 8-GPU box, the per-rank heartbeat finds 90% of desyncs in one read.

One configuration knob worth setting deliberately is the **collective timeout** itself. By default it is large (on the order of 30 minutes) so that legitimately slow collectives — a giant all-reduce, a slow checkpoint barrier — are not killed prematurely. But during *debugging*, a 30-minute wait per hang is brutally slow feedback. Pass a short timeout to `init_process_group` while you are hunting a deadlock so it fails fast and tells you which rank was missing:

```python
import datetime
dist.init_process_group(
    "nccl",
    timeout=datetime.timedelta(seconds=60),   # short while debugging; raise for prod
)
```

With a 60-second timeout, a desync surfaces in a minute instead of half an hour, and the error message names the collective and the participating ranks. Once the bug is fixed, raise the timeout back to the default for production so a legitimately slow step does not get killed. This is a small thing, but the difference between a 60-second and a 30-minute debug loop is the difference between fixing a desync before lunch and fixing it tomorrow.

#### Worked example: an early-break deadlock

A team reported a finetune that "randomly hangs around 80% of the way through the epoch, but only sometimes." The randomness was the tell — a deterministic bug hangs every time; a *data-dependent* one hangs only on certain batches. The per-rank heartbeat showed rank 6 stopped at step 1487 while the others reached 1488. Looking at step 1487 on rank 6: a per-rank `if loss.item() > threshold: break` guard, added to bail on bad batches, fired on rank 6's shard and not the others. Rank 6 exited the loop; ranks 0–5, 7 blocked on the next all-reduce; the watchdog killed it 30 minutes later. The fix was to make the bail collective: `bad = torch.tensor(float(loss.item() > threshold)).cuda(); dist.all_reduce(bad); if bad > 0: handle_uniformly()`. Either all ranks bail or none do. No more hangs. The lesson: **any decision that changes the control flow must be made collectively, or it will desync the group.**

## 6. Per-rank seeds and data sharding

This section contains the two highest-impact *correctness* bugs in DDP, both of which let the run train happily while silently destroying your effective data. They follow directly from the section-1 provisos: disjointness and equal-size.

![A two-column before-and-after showing that without DistributedSampler every rank reads the full dataset producing duplicated gradients and lower accuracy, while sharded data with set_epoch restores a true bigger-batch signal and higher accuracy](/imgs/blogs/debugging-ddp-and-multi-gpu-4.png)

**Bug 1: no `DistributedSampler` — every rank sees the same data.** This is the duplicated-data bug from section 1, and it is shockingly common because the code *works*. You wrap your model in DDP, you keep your existing `DataLoader`, and you launch on 8 GPUs. Without a sampler, each rank's `DataLoader` iterates the *full* dataset in the same order. All 8 ranks compute gradients on the same examples. The all-reduce averages 8 identical gradients (modulo per-rank augmentation randomness), which is just *one* gradient with extra steps. You are doing `8×` the FLOPs for the signal of `1×` the data, you overcount your epochs by `8×` (your "10 epochs" is really 80 effective passes by wall-clock but 10 by unique data), and your final accuracy is *worse* than a properly-sharded run because you have effectively trained on `1/8` of the data diversity per wall-clock hour.

The fix is the `DistributedSampler`, which assigns each rank a disjoint slice of the indices:

```python
from torch.utils.data import DataLoader, DistributedSampler

sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
loader = DataLoader(dataset, batch_size=per_gpu_batch, sampler=sampler,
                    num_workers=8, pin_memory=True)
# 'sampler' and 'shuffle=True' on DataLoader are mutually exclusive:
# the sampler owns the shuffling now, so DataLoader(shuffle=...) stays default False.
```

**Bug 2: forgetting `sampler.set_epoch(epoch)`.** This is the sneaky follow-on. `DistributedSampler` shuffles by deriving its permutation from `epoch + seed`. If you never call `set_epoch`, the epoch stays 0 forever, so *every epoch sees the same shuffle in the same order on every rank*. Your data is sharded (good — disjoint) but never re-shuffled across epochs (bad — each rank sees its same fixed shard every epoch, in the same order). The model sees less effective variety, and on small datasets you can watch validation accuracy plateau early. The fix is one line at the top of each epoch:

```python
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)     # MUST be called; reshuffles + keeps ranks consistent
    for x, y in loader:
        ...
```

**Bug 3: identical augmentation randomness across ranks.** Even with a correct sampler, if every rank uses the same RNG seed, then a random crop or color jitter applied to a sample produces the *same* augmented view on whichever rank touches it — and more importantly, your dataloader workers may reseed identically, collapsing augmentation diversity. You generally want the *data order* to be coordinated (the sampler handles that) but the *augmentation randomness* to differ per rank, so the model sees a richer set of augmented views. The clean pattern is to offset the base seed by the rank for augmentation while keeping the sampler's shuffle seed shared:

```python
def setup_seeds(base_seed: int):
    rank = dist.get_rank()
    # Sampler uses a SHARED seed internally (it must agree across ranks);
    # augmentation / dropout get a PER-RANK seed for diversity:
    torch.manual_seed(base_seed + rank)
    np.random.seed(base_seed + rank)
    import random; random.seed(base_seed + rank)
```

This is the multi-GPU face of the determinism discussion in [reproducibility and determinism in training](/blog/machine-learning/debugging-training/reproducibility-and-determinism-in-training): you want *coordinated* determinism for the things that must agree across ranks (data sharding) and *diverse* randomness for the things that should differ (augmentation, dropout). Getting both backwards — shared augmentation seeds and unshared sampler seeds — is a real and subtle bug.

#### Worked example: the +7-point accuracy fix from adding a sampler

A ResNet-50 finetune on a 50k-image dataset was being trained on 8 GPUs and topping out at 71% validation accuracy, two points below the single-GPU baseline of 73% — the opposite of what scaling should do. The diagnostic was direct: print the number of unique samples each rank draws per epoch and the total unique samples across the run.

```python
seen = set()
for x, y, idx in loader:           # dataset returns its index as a third item for this probe
    seen.update(idx.tolist())
print(f"[rank {dist.get_rank()}] saw {len(seen)} unique samples this epoch")
```

Every rank reported `saw 50000 unique samples` — the smoking gun. All 8 ranks were iterating the full dataset; there was no `DistributedSampler`. We added the sampler with `drop_last=True` and the `set_epoch` call, and re-ran. Now each rank reported `saw 6250 unique samples` (`50000 / 8`), the effective batch became a true `8×` larger batch, we scaled the LR by 8 with a short warmup, and validation accuracy climbed to 78% — five points over the broken run and above the single-GPU baseline, in *less* wall-clock time per effective epoch. The before→after is the kind of evidence that ends the argument: same code, same hyperparameters except LR, one missing sampler.

| Metric | Before (no sampler) | After (sampler + set_epoch) |
|---|---|---|
| Unique samples per rank / epoch | 50,000 (full dataset) | 6,250 (1/8 shard) |
| Effective gradient signal | 1× duplicated | 8× true bigger batch |
| Learning rate | 0.01 (unchanged) | 0.08 (linear scale + warmup) |
| Wall-clock per *effective* epoch | 8× wasted | 1× |
| Final val accuracy | 71% | 78% |

## 7. BatchNorm across ranks: the SyncBN question

Here is a correctness bug that is not a crash and not a deadlock — it is a quiet accuracy regression that depends on a number most people never think about: the *per-GPU* batch size. It only matters for models that use **BatchNorm** (BN), which normalizes activations using the mean and variance of the current batch.

![A two-column before-and-after showing local BatchNorm normalizing on a tiny per-GPU batch with noisy statistics and lower accuracy, versus SyncBatchNorm computing statistics across all ranks to recover global-batch behavior and higher accuracy](/imgs/blogs/debugging-ddp-and-multi-gpu-6.png)

The science first. BatchNorm computes, per channel, the mean $\mu$ and variance $\sigma^2$ *over the examples in the current batch* and normalizes activations as $\hat{x} = (x - \mu)/\sqrt{\sigma^2 + \epsilon}$. The quality of that estimate depends on how many examples it is averaged over. The variance of a sample-mean estimate scales as $1/n$ where $n$ is the batch size, so the BN statistics from a batch of 4 are roughly $\sqrt{32/4} \approx 2.8\times$ noisier than from a batch of 32. BN was designed and tuned assuming a reasonably large batch.

Now the DDP twist: **each rank computes BN statistics on its own local batch, independently.** There is no all-reduce of BN statistics by default. So if your effective batch is 256 across 8 GPUs, each BN layer is *not* seeing 256 examples — it is seeing 32 (the per-GPU batch). And if you went to many GPUs to keep memory low and dropped the per-GPU batch to 4 or 8, your BN is now estimating statistics on 4–8 examples, which is noisy enough to hurt. This is why a model can train fine on one GPU with batch 32, then lose accuracy when "scaled" to 8 GPUs with per-GPU batch 4 — the *effective* batch went up but the *BN* batch went down by `8×`, and BN got noisier, not better. It is a direct collision between the section-1 "bigger batch" intuition (true for the gradient) and the reality that BN never saw the bigger batch (it saw the per-GPU slice).

The fix when per-GPU batch is small is **SyncBatchNorm (SyncBN)**, which all-reduces the BN statistics across ranks so every BN layer normalizes using the *global* batch's mean and variance. PyTorch makes the conversion one line:

```python
# Convert all BatchNorm layers to SyncBatchNorm BEFORE wrapping in DDP:
model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
model = DDP(model, device_ids=[local_rank])
```

SyncBN is not free: it adds an extra all-reduce per BN layer per forward pass (to gather statistics), which is communication overhead. The trade-off is clear and you should reason about it explicitly:

| Per-GPU batch | BN statistics quality | Recommendation |
|---|---|---|
| ≥ 32 | Plenty for stable BN | Plain BN; SyncBN is wasted comm |
| 8–32 | Marginal; task-dependent | Measure both; SyncBN if val drops |
| ≤ 8 | Noisy, hurts accuracy | SyncBN usually worth it |
| Detection / segmentation, small batch | Notoriously BN-sensitive | SyncBN standard practice |

The decision rule: **if your per-GPU batch is small (≤ 8) and you use BatchNorm, suspect BN before anything else when accuracy lags the single-GPU baseline.** It is also worth knowing the alternatives that sidestep the problem entirely: GroupNorm and LayerNorm compute statistics per-example (not across the batch), so they are *batch-size-independent* and immune to this whole class of bug — which is one reason transformers, which use LayerNorm, never have a "BN across ranks" problem. If you are designing a model for distributed training with small per-GPU batches, normalization-by-design (GroupNorm/LayerNorm) avoids the SyncBN tax. This connects to the broader BN train/eval and small-batch failure modes in [initialization and normalization bugs](/blog/machine-learning/debugging-training/initialization-and-normalization-bugs); DDP adds the across-ranks dimension on top.

#### Worked example: SyncBN recovers 4 points at per-GPU batch 4

A semantic segmentation model trained on one GPU at batch 16 reached 77% mean IoU. To fit a larger backbone, the team moved to 8 GPUs with per-GPU batch 4 (effective batch 32) — and mean IoU *fell* to 73%, despite the larger effective batch. Segmentation is BN-sensitive and the per-GPU batch was tiny (4), so BN was the prime suspect. The confirming test was to log the running variance of an early BN layer: on the single-GPU run it was stable across steps; on the 8-GPU run it was visibly noisier step-to-step, consistent with 4-example statistics. Converting to SyncBN (`convert_sync_batchnorm` before the DDP wrap) recovered the global-batch statistics and mean IoU climbed to 77%, matching the single-GPU baseline, at the cost of roughly 6% slower steps from the extra BN all-reduces. The 4-point IoU recovery was worth the 6% throughput.

## 8. The 8-GPUs-same-speed trap: communication, batch, and the dataloader

Now the demoralizing one: the run is *correct* — accuracy matches, no hangs — but you added 7 GPUs and the wall-clock per epoch barely moved. You have a throughput bug, and there are three usual suspects. The discipline is the same as always: **read the instruments** before you guess. The key instrument here is the split of each step into *compute* time versus *communication* time versus *data-wait* time.

The first thing to internalize is **Amdahl's law for DDP**. Each training step is roughly compute (forward + backward) plus the all-reduce communication, with the two overlapping as much as possible. If the all-reduce time is not hidden under compute, it adds to every step. Define the communication-to-computation ratio. If your per-GPU compute per step is $T_c$ and the gradient all-reduce takes $T_r$, the best speedup over one GPU is bounded by how much of $T_r$ overlaps with $T_c$. When $T_r$ approaches or exceeds $T_c$ — which happens when the model has many parameters (big gradients to reduce) but each GPU does little compute (small per-GPU batch) — communication dominates and adding GPUs buys you almost nothing. That is the math of "8 GPUs, 1.1× speed."

The three concrete causes:

**Cause 1: per-GPU batch too small.** If each GPU processes only a handful of examples per step, the compute $T_c$ is tiny, so the all-reduce $T_r$ cannot hide under it and becomes the bottleneck. Worse, tiny batches mean more steps per epoch, and each step pays the fixed all-reduce launch overhead. **The fix is usually to increase the per-GPU batch** (use the memory you have) so compute grows relative to communication — this is the most common single fix for poor DDP scaling. Bigger per-GPU batch also improves BN (section 7) and reduces the *number* of all-reduces per epoch.

**Cause 2: the dataloader is the bottleneck, not the GPU.** This is insidious because `nvidia-smi` shows high utilization in bursts and you conclude the GPUs are busy — but they are busy *waiting* between batches. With 8 GPUs each demanding batches `8×` faster than one, your CPU-side data pipeline (disk reads, decode, augmentation, collate) may not keep up, and every GPU stalls waiting for the next batch. The all-reduce and compute are both fast; the *data* is slow. The fix is more `num_workers`, `pin_memory=True`, prefetching, faster decode (e.g., decode on GPU, use a faster image backend), and caching. This is exactly the dataloader bottleneck covered in depth in [the GPU is idle: throughput debugging](/blog/machine-learning/debugging-training/the-gpu-is-idle-throughput-debugging); under DDP it is `8×` more likely because the data demand is `8×` higher.

**Cause 3: lost comm-compute overlap.** DDP's whole speed story is overlapping the all-reduce with the backward pass. Things that break the overlap: a tiny model where there is no backward to hide behind, `find_unused_parameters=True` (it serializes part of the reduction), gradient checkpointing arranged so gradients all arrive at once at the end (no early buckets to start reducing), and forgetting `no_sync()` during accumulation (you reduce on every micro-step). Reordering so gradients become available progressively, and using `no_sync()` correctly, restores the overlap.

The diagnostic is the PyTorch profiler, which shows you the timeline of compute kernels and NCCL all-reduce kernels and whether they overlap:

```python
from torch.profiler import profile, ProfilerActivity, schedule

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=3),
    record_shapes=True, with_stack=False,
) as prof:
    for step, (x, y) in enumerate(loader):
        opt.zero_grad()
        loss = criterion(model(x.cuda()), y.cuda())
        loss.backward()
        opt.step()
        prof.step()
        if step >= 5: break

if dist.get_rank() == 0:
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    prof.export_chrome_trace("ddp_trace.json")   # open in chrome://tracing
```

In the trace, look for: (a) large gaps between steps where the GPU is idle — that is the dataloader bottleneck; (b) `nccl:all_reduce` kernels that run *after* the backward kernels finish rather than overlapping them — that is lost overlap; (c) the wall-clock fraction spent in NCCL versus in compute — that is your comm-to-compute ratio. The table's `cuda_time_total` for NCCL ops, divided by total step time, tells you immediately whether communication is your bottleneck.

#### Worked example: from 31% GPU-util to 89% by fixing the dataloader

An 8-GPU language-model finetune was getting only `1.3×` the throughput of a single GPU — a catastrophic scaling efficiency of 16%. The profiler trace on rank 0 showed each GPU was active for ~290 ms then *idle for ~640 ms* per step, with the idle gap before the forward pass — the classic dataloader-starvation signature. GPU utilization averaged 31%. The all-reduce was not the problem; it overlapped cleanly. The problem was the data pipeline: `num_workers=2`, no `pin_memory`, and a slow on-the-fly tokenization that could not feed 8 hungry GPUs. We raised `num_workers` to 8 per process, set `pin_memory=True`, pre-tokenized and cached the dataset to disk, and enabled prefetching. The idle gap collapsed; GPU utilization rose to 89%; throughput went from `1.3×` to `6.8×` over one GPU. The fix touched *zero* lines of model or DDP code — it was a `data` bug surfaced by `systems` scaling, found by reading the profiler instead of guessing about NCCL.

| Instrument | Before | After |
|---|---|---|
| GPU utilization (avg) | 31% | 89% |
| Per-step active / idle | 290 ms / 640 ms | 305 ms / 40 ms |
| Bottleneck (from trace) | dataloader stall | compute-bound |
| Speedup vs 1 GPU | 1.3× | 6.8× |
| Scaling efficiency | 16% | 85% |

## 9. Logging and checkpointing: rank-0 discipline

The last bug class is the least scientific and the most embarrassing, because it corrupts your outputs even when training is perfect. Every rank is a separate Python process running the *same code*. So unless you guard it, every rank will: print every log line (8× duplicate logs, or worse, interleaved garbage), write to the same TensorBoard/W&B run (8 writers, corrupted event files), and save a checkpoint to the same path (8 processes writing the same file simultaneously, producing a truncated or corrupt checkpoint).

The rule is simple and absolute: **side effects that should happen once must be guarded with `rank == 0`.** Logging, progress bars, metric reporting, checkpoint saving, and writing summary files all belong to rank 0 only. Computation that must happen on every rank (the forward, backward, all-reduce, optimizer step) is *not* guarded.

```python
rank = dist.get_rank()

def is_main():
    return rank == 0

# Logging: only rank 0.
if is_main():
    wandb.log({"loss": loss.item(), "lr": sched.get_last_lr()[0]}, step=step)

# Checkpointing: save on rank 0, barrier so others wait, then everyone loads.
def save_checkpoint(model, opt, sched, scaler, step, path):
    if is_main():
        torch.save({
            "model": model.module.state_dict(),   # .module unwraps DDP
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict() if scaler else None,
            "step": step,
        }, path)
    dist.barrier()   # non-zero ranks WAIT until the file exists before continuing
```

Three subtleties that bite even people who know the rank-0 rule:

- **Save `model.module.state_dict()`, not `model.state_dict()`.** A DDP-wrapped model prefixes every parameter name with `module.`. If you save the wrapped state dict and later load it into a bare (non-DDP) model for inference, every key is mismatched (`module.fc.weight` vs `fc.weight`) and the load fails or silently loads nothing. Save the *unwrapped* `.module` state dict so the checkpoint is portable to single-GPU inference. This portability/key-prefix issue is the DDP slice of the broader resume problems in [the taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs).
- **Barrier after saving.** If rank 0 is busy writing a large checkpoint while ranks 1–7 race ahead into the next epoch, and then the script tries to *load* that checkpoint on all ranks (e.g., for an eval that reloads), the non-zero ranks may try to read a half-written file. `dist.barrier()` after the rank-0 save makes everyone wait until the file is complete. Equally, `barrier()` before a load makes sure the save finished.
- **Metrics must be reduced before logging.** If rank 0 logs only *its own* loss, you are reporting the loss on `1/8` of the batch, which is noisier and can mislead. To log the true batch loss, all-reduce the per-rank losses and divide by world size *before* the rank-0 log: `dist.all_reduce(loss_tensor); loss_tensor /= world_size`. Note that this all-reduce is a collective — **every** rank must call it (not just rank 0), then only rank 0 logs the result. Guarding the all-reduce itself with `if rank == 0` would deadlock (section 5).

#### Worked example: the corrupted-checkpoint mystery

A run "completed successfully" but the saved checkpoint could not be loaded — `torch.load` threw `RuntimeError: PytorchStreamReader failed reading zip archive`. The training logs looked clean (because the print statements were, by luck, guarded). The cause: the *checkpoint save* was not guarded, so all 8 ranks called `torch.save(..., "best.pt")` at the same instant. Eight processes writing the same file produced a corrupt archive — whichever process finished last left a truncated file mixing fragments from the others. The fix was the `if is_main(): torch.save(...)` guard plus the barrier. The tell that it was a rank issue and not a disk issue: the file size was *smaller* than a single valid checkpoint, indicating an interrupted write, and it only happened on multi-GPU runs (the single-GPU baseline saved fine). One `if rank == 0:` and the corruption was gone.

## 10. The full bisection: a worked multi-GPU debug

Let me put the whole method together on a realistic failing run, because the value of this post is not any single fix — it is the *order* in which you apply the tests. The diagnostic table below is the map; the narrative below it walks the path.

![A matrix mapping five DDP symptoms to a confirming test, a root cause, and a fix, covering backward hangs, low accuracy, same-speed scaling, NaN losses, and corrupt checkpoints](/imgs/blogs/debugging-ddp-and-multi-gpu-7.png)

**The report:** "Our BERT finetune works on one GPU and reaches 91% on the eval set. On 8 GPUs it reaches only 86%, and each epoch takes almost as long as one GPU. Help."

Two distinct symptoms here — *lower accuracy* and *poor speedup* — and they may have different causes, so I will bisect each.

**Step 1 — reproduce on one GPU (section 2).** Run `torchrun --nproc_per_node=1` with the same config. It reaches 91% and runs at the expected one-GPU speed. So neither symptom reproduces on one GPU: both are *true distributed bugs*. Good — the search space is now the DDP-specific layers, not the model or loss.

**Step 2 — attack the accuracy gap first (it is correctness; speed is just money).** The suspects for "accuracy lower than one-GPU" are: duplicated data (no sampler), missing `set_epoch`, sum-vs-mean reduction, BN across ranks, or wrong LR scaling. Bisect by elimination:

- *Data sharding?* Add the unique-samples probe from section 6. Every rank reports the full dataset → **no `DistributedSampler`.** That is almost certainly the accuracy bug: 8 ranks training on duplicated data, overcounting epochs by 8×, with an LR tuned for the wrong effective batch. Add the sampler + `set_epoch`, scale LR by 8 with warmup. Re-run: accuracy climbs to 90.5%. Most of the gap closed.
- *Remaining 0.5%?* BERT uses LayerNorm, not BatchNorm, so SyncBN is irrelevant (LayerNorm is batch-size-independent — section 7). The small residual is within run-to-run noise. Accuracy bug solved.

**Step 3 — attack the speed (now correctness is fixed).** With the sampler in place, each epoch now does `1/8` the *unique* data per rank, so it should already be faster. Profile (section 8). The trace shows GPUs idle ~50% of the time before each step → **dataloader bottleneck**, amplified because 8 ranks now demand data 8× faster. Raise `num_workers`, `pin_memory=True`, pre-tokenize and cache. Re-profile: utilization from 44% to 86%, speedup from 1.4× to 6.5×.

**Step 4 — confirm no regressions and stress-test.** Run the grad-sync assertion from section 3 (ranks agree, matches single-GPU full-batch grad → sync correct). Run the unused-param check from section 4 (no params with `grad is None` → no hang risk). Add a per-rank heartbeat (section 5) and the `rank == 0` guards on logging and checkpointing (section 9). Final: 8 GPUs, 90.5% accuracy (matching one GPU), 6.5× throughput.

The point is the *sequence*: reproduce-on-one-GPU split distributed from non-distributed; then for each symptom you bisected through a short, ordered list of DDP-specific suspects, each with a one-line confirming test, fixing correctness before chasing speed. No NCCL source code was read. No guessing. The two symptoms had two different root causes (no sampler; dataloader starvation), which is common — *do not assume one fix explains all symptoms.*

#### Worked example: deciding comm-bound vs compute-bound with one number

When a run scales poorly, the single most useful number is the **comm-to-compute ratio**: the fraction of each step spent in NCCL all-reduce versus in compute. You do not need a full profiler trace to get a first estimate — you can isolate the communication cost directly. Run the step once normally (compute + comm = $T_{\text{full}}$), then run it with the all-reduce suppressed via `model.no_sync()` on *every* step (compute only = $T_{\text{compute}}$, though you must not actually train this way — it is a measurement). The difference $T_{\text{full}} - T_{\text{compute}}$ is the non-overlapped communication time.

On a poorly-scaling 8-GPU ResNet run, I measured $T_{\text{full}} = 142$ ms and $T_{\text{compute}} = 138$ ms per step — only 4 ms of non-overlapped comm, a comm-to-compute ratio of about 3%. That is *not* a communication problem; the all-reduce was overlapping almost perfectly, so adding GPUs should scale near-linearly and the real bottleneck had to be elsewhere (it was the dataloader). Contrast a 32-GPU multi-node run of the same model: $T_{\text{full}} = 95$ ms but $T_{\text{compute}} = 58$ ms — 37 ms of exposed communication, a ratio of 39%. *That* is comm-bound, and the cause was the slow inter-node link (section 14); no amount of dataloader tuning would help. One number — measured, not guessed — routed the two runs to two completely different fixes. The rule: if non-overlapped comm is under ~10% of the step, stop blaming NCCL and look at data or compute; if it is over ~30%, communication is your wall and you need a bigger per-GPU batch, a faster fabric, or fewer/larger collectives.

## 11. When it is (and isn't) a DDP bug

A decisive section, because half of DDP debugging is *not* debugging DDP — it is recognizing the bug lives elsewhere.

**It is probably NOT a DDP bug when:**

- **It reproduces identically on one GPU.** This is the master test. A NaN at step 412 on 8 GPUs that also NaNs at step 412 on 1 GPU is a numerics/LR/data bug — go to [hunting NaNs and infs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) territory, not here. DDP did not cause it; DDP just ran your bug 8 times.
- **The loss curve shape is wrong in a way that is not about data volume.** A smooth-then-spike divergence is an optimization/numerics signature (LR too high, a bad batch), independent of rank count.
- **Accuracy is bad on one GPU too.** If the single-GPU baseline never reached the target either, your problem is the model/data/recipe, not distribution.

**It IS a DDP bug when:**

- **The symptom appears or changes between 1 and N GPUs.** A hang that only happens on N → unused param or rank desync. Accuracy that is *worse* on N than on 1 → sharding, BN, or LR scaling. Speed that does not scale with N → comm/dataloader/overlap.
- **It hangs at a collective.** Hangs at `backward()` or `barrier()` with no Python traceback are distributed by definition — single-GPU code does not deadlock on a collective.
- **Outputs are corrupted or duplicated.** Corrupt checkpoints, 8× log lines, mangled event files → rank-0 guarding.

**The tricky middle ground — bugs that *change* between 1 and N:**

- **A NaN that only appears on N GPUs** is usually about the *effective batch and LR*: at effective batch 256 with un-scaled LR you are stable, but scale LR by 8 without warmup and you spike. The DDP part is real (the effective batch changed), but the fix is LR scheduling, not NCCL.
- **Accuracy slightly worse on N** with LayerNorm models and correct sharding is often just the larger-batch generalization gap — a known phenomenon where very large batches generalize slightly worse, fixable with LR warmup and tuning, not a bug at all. Know the difference between "I have a bug" and "large-batch training behaves differently," or you will chase a non-bug for days.

The honest summary: **reproduce on one GPU first, and let the answer route you.** If it survives the single GPU, you are in the wrong post — go debug the underlying training bug. If it needs N GPUs to appear, you are in the right place, and the suspect list is short.

## 12. Case studies and real-world signatures

These are well-known patterns and representative results; where a number is approximate I say so, and the mechanisms are exactly as documented in the PyTorch DDP and NCCL documentation.

**The `find_unused_parameters` hang is the #1 DDP issue.** Across the PyTorch forums and issue tracker, the most common DDP question is some variant of "my training hangs at the first backward." The documented mechanism is exactly section 4: the reducer waits for a bucket whose parameter got no gradient. PyTorch's own guidance is to use `find_unused_parameters=True` to *diagnose* but to prefer ensuring all parameters participate, because the flag adds a per-iteration graph traversal whose cost the docs describe as non-trivial for large models. The fix-by-restructuring is the recommended cure.

**The duplicated-data bug is silent and common.** Forgetting `DistributedSampler` does not error — it trains on duplicated data. The signature is "more GPUs, same-or-worse accuracy, epochs that finish suspiciously fast in terms of unique data," and the unique-samples probe (section 6) confirms it in one read. This is one of the first things every DDP tutorial warns about precisely because the failure is invisible without instrumentation.

**SyncBN for small-batch dense prediction is standard practice.** In object detection and segmentation, where per-GPU batch is often 2–4 due to image size and memory, plain BN's per-GPU statistics are too noisy, and the field standardized on SyncBN (and BN alternatives like GroupNorm) to recover accuracy. The documented effect — a few points of mAP/IoU recovered by syncing BN statistics across GPUs — matches the section-7 worked example's order of magnitude; the exact gain is task- and batch-dependent, so treat the "+4 IoU" as representative, not universal.

**Poor scaling efficiency at small per-GPU batch is the textbook comm-bound regime.** When the model is large (big gradients) and the per-GPU batch is small (little compute), the all-reduce cannot hide under the backward and communication dominates. This is the well-understood reason large-model training uses gradient accumulation and large per-GPU batches to raise the compute-to-comm ratio, and why naive "just add GPUs" often yields sublinear speedup. The cure — bigger per-GPU batch, fix the dataloader, preserve comm-compute overlap — is consistent across the distributed-training literature; for the sharding-based approach to fitting bigger batches and models, see the sibling [FSDP and sharding bugs](/blog/machine-learning/debugging-training/fsdp-and-sharding-bugs) post, and the systems-level scaling story in the [multi-node LLM training recipe and troubleshooting](/blog/machine-learning/mlops/multi-node-llm-training-recipe-troubleshooting) writeup.

**Left-over outputs from unguarded ranks are a recurring footgun.** Corrupt checkpoints from concurrent writes and 8× duplicated logs are common enough that every mature training framework (Lightning, HF `accelerate`, the HF `Trainer`) guards them automatically with an `is_main_process` check. When people roll their own DDP loop, they rediscover this bug; the fix is the rank-0 guard plus barrier from section 9. If you use `accelerate` or the `Trainer`, much of section 9 (and the sampler, and the `.module` unwrapping) is handled for you — which is itself a strong argument for using them rather than hand-rolling.

## 13. Tooling shortcut: let accelerate or Trainer do the bookkeeping

Most of the *mechanical* DDP bugs — missing sampler, unguarded logging, `.module` unwrapping, `set_epoch` — disappear if you use a framework that handles the bookkeeping. Hugging Face `accelerate` wraps the model, sampler, and dataloader so the sharding and rank-0 guarding are automatic:

```python
from accelerate import Accelerator
accelerator = Accelerator()          # reads world size / rank from torchrun env
model, optimizer, loader, sched = accelerator.prepare(model, optimizer, loader, sched)

for epoch in range(epochs):
    for batch in loader:             # already sharded; set_epoch handled
        with accelerator.accumulate(model):     # no_sync handled for accumulation
            loss = model(**batch).loss
            accelerator.backward(loss)          # all-reduce handled
            optimizer.step(); sched.step(); optimizer.zero_grad()
    if accelerator.is_main_process:             # rank-0 guard provided
        accelerator.save_state("ckpt/")         # save handled, no .module footgun
```

The HF `Trainer` goes further — pass `--nproc_per_node` to `torchrun` and the `Trainer` shards data, scales nothing automatically (you still set LR), guards logging/checkpointing, and unwraps the model on save. The catch: frameworks hide the bugs but *do not eliminate the science*. You still must reason about effective batch and LR scaling (the framework will not scale your LR for you), about per-GPU batch and BN (it will not switch you to SyncBN automatically unless told), and about the dataloader bottleneck (it will not magically make your data pipeline fast). The framework removes the *bookkeeping* bugs (sections 4, 6, 9); the *science* bugs (sections 1, 3, 7, 8) are still yours. So even on `accelerate`, the diagnostic toolkit — reproduce on one GPU, compare grad sync, probe unique samples, profile comm-vs-compute — is exactly the same. Frameworks change *who writes the guard*, not *what the bug is*.

A final framework note for finetuning: when you combine DDP with parameter-efficient finetuning (LoRA), the unused-parameter trap (section 4) gets sharper, because most base-model parameters are frozen (`requires_grad=False`) and only the adapter trains — so the set of parameters that *should* get gradients is small and specific. If your `target_modules` are wrong, the adapter is unused and DDP can hang or train nothing; the section-4 `grad is None` probe is the fastest way to confirm the adapter actually entered the graph. That intersection is covered in the PEFT-specific debugging posts; the DDP lesson is that frozen-parameter finetuning makes "is every trainable parameter used?" a question you must answer explicitly.

## 14. Multi-node: when the bug crosses the network

Everything so far assumed a single node with 8 GPUs talking over the fast intra-node fabric (NVLink/PCIe). The moment you span *multiple* nodes — say 4 machines of 8 GPUs each for a world size of 32 — a new layer of bugs appears, and they are nastier because they involve the network, environment variables, and rendezvous, none of which exist on a single box. The mechanism (gradient averaging) is unchanged; what changes is *where the collective can fail*.

**The rendezvous and environment.** On a single node, `torchrun` sets everything up. Across nodes, every process must agree on a *master address and port* to rendezvous, and you must launch `torchrun` on each node with consistent `--nnodes`, `--node_rank`, and `--rdzv_endpoint`. The classic multi-node bug is a `--node_rank` mismatch (two nodes both think they are node 0, so global ranks collide) or a master address that one node cannot reach (firewall, wrong interface). The symptom is a hang *at* `init_process_group` — the run never even starts training, because the ranks never find each other. The diagnostic is to check that every node can reach the master port (`nc -zv master_host 29500`) and that `RANK`/`WORLD_SIZE` printed by each process are globally unique and sum correctly.

```bash
# Node 0 (master):
torchrun --nnodes=4 --node_rank=0 --nproc_per_node=8 \
         --rdzv_id=job123 --rdzv_backend=c10d \
         --rdzv_endpoint=10.0.0.1:29500 train.py
# Node 1 (same command, --node_rank=1), Node 2 (=2), Node 3 (=3).
# A wrong node_rank or unreachable rdzv_endpoint hangs at init_process_group.
```

**The slow-link bottleneck is real across nodes.** Intra-node NVLink might be 600 GB/s; inter-node networking is often 10–100× slower (even fast InfiniBand is ~25–50 GB/s per link). So the all-reduce cost model from section 1 ($2M/\text{bandwidth}$) gets a much smaller bandwidth for the inter-node hops, and a run that scaled fine within one node can fall off a cliff at the node boundary. The signature is "great scaling up to 8 GPUs, terrible past 8" — the eighth-to-ninth GPU crossed a node and hit the slow link. The diagnostic is to compare scaling efficiency at world size 8 (one node) versus 16 (two nodes); a sharp drop at the boundary points at the network, and the fixes are gradient compression, larger per-GPU batch to amortize the slow all-reduce, or a faster interconnect — not anything in your model code.

**Network-interface selection.** NCCL must pick the right network interface for inter-node traffic, and on machines with multiple NICs (a management interface and a fast data interface) it sometimes picks the slow one, silently. The result is a *correct but slow* multi-node run. Setting `NCCL_SOCKET_IFNAME` (or letting NCCL auto-detect InfiniBand via `NCCL_IB_HCA`) pins the fast interface. The diagnostic is `NCCL_DEBUG=INFO`, which logs exactly which interface and transport (IB, socket) NCCL chose at startup — read that line and confirm it is the fast fabric, not `eth0`.

**Asymmetric hardware desync.** If one node has slightly slower GPUs, a thermal-throttled card, or a noisy neighbor, that node's ranks fall behind, and because every all-reduce waits for the slowest participant, the *whole job* runs at the speed of the slowest rank — a "straggler." The signature is utilization that is high on most ranks but the step time matching the slow node. The flight recorder (`TORCH_NCCL_TRACE_BUFFER_SIZE`) and per-rank step timing identify the straggler; the fix is to exclude the bad hardware. This is the multi-node face of section 5's "every collective waits for every peer": across nodes, the peer that is merely *slow* (not dead) silently caps your throughput.

The debugging discipline scales with the topology: reproduce on one GPU (is it distributed at all?), then one node (is it a within-node DDP bug?), then multi-node (is it a network/rendezvous bug?). Each boundary is a bisection point. A bug that only appears at multi-node and not single-node is, by construction, about the network, the launcher, or the rendezvous — a tiny suspect list. For the heavier production patterns (elastic training, fault tolerance, checkpoint-on-preemption across nodes), the [multi-node LLM training recipe and troubleshooting](/blog/machine-learning/mlops/multi-node-llm-training-recipe-troubleshooting) post goes deeper into the operational side.

## Key takeaways

- **Reproduce on ONE GPU first.** If the bug survives a single GPU with the same data and seed, it is not a DDP bug — go debug the underlying data/optimization/model issue where the instruments are clean. This one branch halves your search.
- **Data-parallel SGD is gradient averaging, which is a bigger batch.** The all-reduce averages per-rank mean-gradients into the full-batch gradient. Effective batch = per_gpu_batch × num_gpus, so scale your LR (linearly, with warmup) when you scale GPUs.
- **A backward that hangs ~30 minutes then NCCL-times-out is almost always an unused parameter.** Confirm with `[n for n,p in model.named_parameters() if p.grad is None]`; fix by using or removing the parameter, not by permanently setting `find_unused_parameters=True`.
- **No `DistributedSampler` means every rank trains on the full dataset** — duplicated gradients, overcounted epochs, worse accuracy. Probe unique-samples-per-rank; add the sampler plus `sampler.set_epoch(epoch)` every epoch.
- **Any control-flow decision must be collective.** A rank that `break`s early, skips `backward()`, or runs a different number of steps desyncs the group and deadlocks. Per-rank heartbeat logs find which rank stalled.
- **BatchNorm sees only the per-GPU batch.** At small per-GPU batch (≤ 8), plain BN statistics are too noisy and accuracy drops; SyncBatchNorm restores the global statistics at a small comm cost. LayerNorm/GroupNorm sidestep the problem entirely.
- **8 GPUs at 1× speed is comm-bound, dataloader-bound, or lost overlap.** Profile the step into compute vs all-reduce vs data-wait; the usual fixes are bigger per-GPU batch, more dataloader workers, and preserving comm-compute overlap (use `no_sync()` during accumulation).
- **Guard side effects with `rank == 0`.** Logging, progress bars, and `torch.save` must run on one rank (with a barrier and `model.module.state_dict()`); collectives like reducing metrics must run on *every* rank, then only rank 0 logs.
- **Frameworks remove bookkeeping bugs, not science bugs.** `accelerate`/`Trainer` handle the sampler, guards, and `.module` unwrapping, but you still own LR scaling, the BN decision, and the dataloader — the diagnostic toolkit is identical.

## Further reading

- **PyTorch DDP documentation and design notes** — `torch.nn.parallel.DistributedDataParallel`, the bucketing/reducer design, `find_unused_parameters`, and `no_sync`; the authoritative reference for the all-reduce overlap mechanism (pytorch.org docs).
- **"PyTorch Distributed: Experiences on Accelerating Data Parallel Training"** — Li et al., 2020 (VLDB) — the paper behind DDP's gradient bucketing and computation–communication overlap; explains *why* the reducer works the way it does.
- **"Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"** — Goyal et al., 2017 — the linear LR scaling rule and warmup for large effective batches; the reference for section 1's batch/LR mechanism.
- **NCCL documentation** — collective semantics (all-reduce, broadcast, barrier), environment variables, and the timeout/flight-recorder debugging tools for diagnosing hangs (docs.nvidia.com/deeplearning/nccl).
- **PyTorch profiler and `torch.profiler`** — the tool for splitting a step into compute vs NCCL vs data-wait and confirming comm-compute overlap; the instrument behind section 8.
- **Hugging Face `accelerate` documentation** — how the framework handles sharding, rank-0 guarding, and accumulation, so you can stop hand-rolling them.
- **Within this series:** the master decision tree in [a taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs); the [training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook) capstone; the closely related [gradient accumulation and effective batch bugs](/blog/machine-learning/debugging-training/gradient-accumulation-and-effective-batch-bugs), [reproducibility and determinism in training](/blog/machine-learning/debugging-training/reproducibility-and-determinism-in-training), [the GPU is idle: throughput debugging](/blog/machine-learning/debugging-training/the-gpu-is-idle-throughput-debugging), and [FSDP and sharding bugs](/blog/machine-learning/debugging-training/fsdp-and-sharding-bugs).
