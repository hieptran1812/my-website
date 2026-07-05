---
title: "Determinism Across Ranks: Making a Distributed Run Reproducible"
date: "2026-07-02"
publishDate: "2026-07-02"
description: "Two 8-GPU runs launched with the same seed drift apart by step 50, and now you cannot bisect a single change against your own baseline. Here is every source of nondeterminism a distributed run adds on top of one GPU, the exact knob for each, and the honest cost of turning them all off."
tags:
  [
    "distributed-training",
    "multi-gpu",
    "reproducibility",
    "determinism",
    "nccl",
    "pytorch",
    "debugging",
    "deep-learning",
    "ml-systems",
  ]
category: "machine-learning"
subcategory: "Distributed Training"
author: "Hiep Tran"
featured: true
readTime: 41
---

Here is a bug that cost me two days, and the reason it cost two days is that it was not really a bug — it was the *absence* of a property I had assumed I had for free.

I was chasing a small regression. A config change had nudged the final validation loss up by about 1%, and I wanted to know which of four changes in the commit was responsible. This is the most routine debugging task in all of machine learning: change one thing, hold everything else fixed, compare. So I launched the baseline twice — same commit, same 8×A100 node, same seed, same data, same everything — intending to establish the noise floor before I started bisecting. The two runs should have produced the same loss curve. They diverged at step 50. By step 200 they were 0.08 apart in loss, which is *larger* than the 1% regression I was trying to chase. My measuring instrument had more noise than the thing I was measuring.

That is the moment you discover, viscerally, that reproducibility is not a nicety. It is the ground you stand on to debug anything at all. A run you cannot reproduce cannot be bisected, because "before" and "after" are both moving. A "silently wrong" bug — the kind that does not crash, just quietly makes your model 1% worse — cannot be isolated, because you cannot tell its effect apart from run-to-run wobble. Every debugging technique in this series, from `git bisect` on your training code to A/B-testing a kernel, secretly assumes that if you run the same thing twice you get the same answer. On a single GPU that assumption is *almost* true by default. On eight GPUs, across two nodes, it is false in more ways than most engineers realize.

![A taxonomy tree showing four single-GPU sources of nondeterminism on one branch and five distributed-only sources on another branch below a shared root](/imgs/blogs/determinism-across-ranks-1.webp)

The figure above is the map for the whole post. A distributed run inherits every single-GPU source of nondeterminism — RNG seeds, nondeterministic CUDA kernels, cuDNN autotuning, floating-point accumulation order — and then adds its own: the order NCCL sums gradients across ranks, the way data is sharded and shuffled per rank, the two-faced problem of seeding (model init must be *identical* across ranks, but augmentation and dropout must be *decorrelated*), the fact that changing the GPU count changes the math, and dynamic control flow. By the end of this post you will be able to name every one of these, reach for the exact knob that controls it, write a five-line test that proves two runs match bit-for-bit for K steps, and — just as importantly — decide *which* of these you actually need to turn off, because determinism is not free and full determinism is genuinely slower.

This sits on the [four walls](/blog/machine-learning/distributed-training/why-distributed-training) frame that runs through the series: a large run is too slow and too expensive to debug by guesswork, so the ability to reproduce it exactly is what makes it debuggable at all. If you have not read the [intro](/blog/machine-learning/distributed-training/why-distributed-training), it sets up the vocabulary — rank, world size, all-reduce, shard — that this post leans on. The single-GPU half of this story is covered in depth in [reproducibility and determinism in training](/blog/machine-learning/debugging-training/reproducibility-and-determinism-in-training); here I recap it briefly and spend the bulk of our time on what *distributed* adds.

## 1. Why reproducibility is a debugging superpower

Before the mechanisms, let me make the case for why you should care enough to pay the cost, because the cost is real and you will be tempted to skip it.

A training run is a long chain of computation: forward pass, loss, backward pass, all-reduce, optimizer step, repeat, tens of thousands of times. Somewhere in that chain something is wrong — the loss plateaus too early, a checkpoint resume spikes, the multi-node run is slower than single-node, one config is 1% worse than another. Debugging is the process of localizing "where" and "what." Every localization technique is a controlled experiment: you vary one thing and hold the rest fixed.

Now suppose the run is nondeterministic. You change the learning rate, rerun, and the loss moves. Did it move because of the learning rate, or because the run was going to move anyway? You genuinely cannot tell. The signal you are trying to read is buried under noise that you introduced yourself by not controlling the seed and the reduction order. This is not a subtle statistical point — it is the difference between a debugging session that converges in an hour and one that never converges at all.

There is a second, sharper reason. Some bugs are *silent*. A miswired gradient hook that drops 0.1% of updates, a data-augmentation seed that is accidentally identical across all eight ranks so every GPU sees the same "random" crop, an all-reduce that is subtly wrong on one node — none of these crash. They produce a model that trains fine and is slightly worse than it should be. The *only* way to catch a silent correctness bug is to have a reference you trust and compare against it bit-for-bit. If you cannot reproduce your own reference, you have no way to know the silent bug is even there. Reproducibility is what turns "the model seems a bit off" from an unfalsifiable feeling into a testable claim.

It is worth being concrete about how expensive the *absence* of this property is, because it is easy to treat determinism as an academic nicety. My two-day regression hunt is a small example. The larger ones are worse: a team that ships a model 0.5% below where it should be because a seeding bug halved their augmentation diversity, and never finds out because they have no reproducible reference to compare against; an A/B test between two kernels that reports a "win" that is actually run-to-run noise, sending the team down a months-long optimization path built on a mirage; a research result that cannot be reproduced by the next person and quietly poisons the literature. Every one of these is a determinism failure wearing a different costume, and every one of them is far more expensive than the 15 minutes and the 10–30% short-run slowdown it takes to prevent.

So the thesis of this post is blunt: **reproducibility is the debugging superpower, and distributed training is where it is hardest to keep.** The rest is a catalog of where it leaks and how to plug each leak, followed by the honest accounting of what plugging them costs.

## 2. The single-GPU sources, recapped

To understand what distribution *adds*, we need the baseline. On a single GPU there are four classic sources of nondeterminism, and they are worth naming precisely because the distributed sources are variations and amplifications of them.

**RNG seeds.** Python's `random`, NumPy's `np.random`, and PyTorch's `torch.manual_seed` each drive a separate pseudo-random stream. Model initialization, dropout masks, data shuffling, and augmentation all draw from these streams. If you do not seed them, every run starts from different weights and sees data in a different order, and the loss curve is different by construction. Seeding all three is the price of admission and costs nothing at runtime.

**Nondeterministic CUDA kernels.** This is the sneaky one. Many GPU kernels — especially in the *backward* pass — use atomic accumulation for performance. When many threads add their partial results into the same memory location with `atomicAdd`, the hardware does not guarantee the *order* of those additions. Because floating-point addition is not associative (more on this in a moment), a different order gives a different sum in the last bit. Scatter operations, some pooling backwards, `index_add`, embedding-bag backward, and several others are nondeterministic for exactly this reason. The same code, same input, same GPU, run twice, can give bitwise-different gradients.

**cuDNN autotuning.** When `torch.backends.cudnn.benchmark = True`, cuDNN times several convolution algorithms on the first call for each input shape and picks the fastest. That choice depends on transient machine state — what else is running, thermal conditions, timing jitter — so two runs can select different algorithms, and different algorithms produce different rounding. Benchmark mode is a throughput win and a reproducibility hazard in the same switch.

**Accumulation order in reductions.** Even a plain `tensor.sum()` over a large tensor is computed in parallel by many threads whose partial sums are combined in a hardware-dependent order. Same non-associativity, same consequence: the last few bits of a reduction are not guaranteed run-to-run unless you force a deterministic algorithm.

Notice the common root under three of these four: **floating-point addition is not associative, and parallel hardware does not fix the order of additions.** Hold that thought, because it is the single most important idea in the distributed half of this post.

## 3. The floating-point fact that makes distributed determinism hard

Let me make the non-associativity concrete, because everything downstream depends on it and it is easy to wave at without believing.

A 32-bit float has a 23-bit mantissa, so its relative precision — machine epsilon — is about $2^{-23} \approx 1.19 \times 10^{-7}$. When you add two numbers of very different magnitudes, the smaller one's low bits fall off the end of the mantissa and are rounded away. That makes addition *order-dependent*:

$$(a + b) + c \neq a + (b + c) \quad \text{in general.}$$

A worked instance in fp32: let $a = 1.0$, $b = 2^{-24}$, $c = 2^{-24}$. Compute left to right: $a + b$ rounds to exactly 1.0 because $2^{-24}$ is below half an ULP of 1.0, then $(a+b)+c = 1.0$ again. Now regroup: $b + c = 2^{-23}$, which *is* representable relative to 1.0, so $a + (b + c) = 1.0000001$. Two ways of adding the same three numbers, two different answers, differing in a real bit. Scale this up to summing eight gradient tensors of wildly different magnitudes and the "last bit" disagreements compound into differences you can see in the loss after a few dozen steps, because training is a chaotic dynamical system where tiny perturbations grow.

Why does this matter more across ranks than within one GPU? Because a distributed gradient reduction *sums numbers that live on different GPUs*, and the order in which they are summed is decided by the collective algorithm and the network topology — both of which can change from run to run, and even within a run. A [ring all-reduce](/blog/machine-learning/distributed-training/collectives-from-scratch) accumulates the eight ranks' contributions in ring order; a tree all-reduce accumulates them pairwise up a tree; and NCCL is free to *pick* between ring, tree, and protocol variants (`Simple`, `LL`, `LL128`) based on message size and measured bandwidth. Two runs of the identical code can therefore reduce the same gradients in two different orders and land on bitwise-different values — which then flow into the optimizer, perturb the weights, and diverge the trajectories.

Make it concrete with eight ranks. Suppose the same gradient element on the eight GPUs has values roughly $g_0 = 3.0$ and $g_1 \ldots g_7 = 2^{-24}$ each — one big contribution and seven tiny ones. A ring reduction that starts its accumulator at $g_0$ and adds the tiny values one at a time will *lose every one of them*: each $2^{-24}$ is below half an ULP of 3.0 and rounds away, so the ring answer is 3.0 exactly. A tree reduction pairs the tiny values first — $2^{-24} + 2^{-24} = 2^{-23}$, then combines those partial sums up the tree — so seven of them accumulate to roughly $7 \times 2^{-24} \approx 4.2 \times 10^{-7}$ *before* meeting the big value, and that combined tiny sum is now large enough to survive the addition to 3.0. Same eight numbers, ring gives 3.0 and tree gives 3.0000004. That difference is small, but the optimizer applies it every step to millions of parameters, and training's sensitivity to initial conditions amplifies it: within a few dozen steps the two trajectories are visibly apart. This is not a hardware defect — both answers are "correct" to within floating-point tolerance. It is just that "correct to within tolerance" is not the same as "bit-for-bit identical," and reproducibility demands the latter.

This is the crux. On one GPU, nondeterminism is mostly about kernels you can force deterministic. Across ranks, nondeterminism is baked into the *communication* itself, and controlling it means controlling the collective algorithm — a knob most people have never touched.

## 4. The distributed-only sources, one at a time

With the floating-point fact in hand, here is the catalog of what distribution adds. Each of these is absent on a single GPU and each needs its own control.

**Source 1: the all-reduce reduction order.** Covered above — this is the heart of it. NCCL does not guarantee bit-for-bit identical results across runs unless you pin the algorithm and protocol, because it may sum gradients in a different order. The knob is the `NCCL_ALGO` environment variable (`Ring` or `Tree`) plus `NCCL_PROTO` (`Simple`), which stop NCCL from switching strategies underneath you.

**Source 2: data order.** On one GPU, `DataLoader(shuffle=True)` with a fixed seed gives a fixed order. Across ranks, order is set by `DistributedSampler`, which deterministically partitions indices into per-rank shards and shuffles them using a seed *and the epoch number*. The trap: `DistributedSampler` only re-shuffles between epochs if you call `sampler.set_epoch(epoch)`. Forget it and every epoch replays the identical order — deterministic, but wrong. And a *streaming* dataset (an `IterableDataset` reading sharded files) has an order that depends on how shards are assigned to ranks and how each worker interleaves them, which is a whole additional surface to pin down.

Streaming deserves a closer look because it is where large-scale runs actually live — you do not fit a trillion-token corpus in a map-style dataset. With an `IterableDataset`, three separate decisions each inject order: which files each *rank* owns (usually `files[rank::world_size]`, which is stable only if the file list is sorted deterministically before slicing), which files each *worker* within a rank owns (a second modular split by `worker_id`), and how a shuffle buffer interleaves records as they stream. Every one of those needs its own seed, and the shuffle-buffer seed in particular must be derived from the epoch so that a resumed run picks up the same interleaving. The failure mode is subtle: if your file globbing returns paths in filesystem order rather than sorted order, two launches on two machines can assign different shards to rank 0, and now "same seed" produces different data — a reproducibility break that has nothing to do with kernels or NCCL and everything to do with a missing `sorted()`.

**Source 3: per-rank seeding — the two-faced requirement.** This one trips up almost everyone, so it gets its own section (§5). In short: model initialization must be *identical* on every rank, or the ranks start from different weights and DDP's assumption that all replicas are the same is violated from step zero. But data augmentation and dropout must be *decorrelated* across ranks, or every GPU applies the same "random" crop and the same dropout mask to different data, which wastes the entropy that stochastic regularization is supposed to provide. Identical here, different there — from the same base seed. Getting this backwards is a silent bug that costs you a fraction of a percent of final accuracy and never announces itself.

**Source 4: the world size changes the math.** This is not a bug you fix; it is a fact you accept. Eight GPUs with local batch size 16 give a global batch of 128. Run the same code on 16 GPUs and the global batch is 256 — a *different optimization problem* with a different effective learning rate and different gradient noise. Even with everything else pinned, a 64-GPU run and an 8-GPU run are not expected to match, because the number of ranks is a hyperparameter. Reproducibility is always "same result at the same world size."

**Source 5: dynamic control flow and variable-length inputs.** If your model has data-dependent branches (mixture-of-experts routing, early exits) or your batches have variable sequence lengths that get bucketed differently depending on what else is in the batch, then the *computation graph itself* changes run to run. Padding, sorting-by-length, and bucketing decisions all interact with data order, and any of them can desynchronize two runs even when every seed is pinned.

Mixture-of-experts is the sharpest example. In an MoE layer, a router sends each token to its top-k experts, and the experts live on different ranks, so the routing decision determines an [all-to-all](/blog/machine-learning/distributed-training/collectives-from-scratch) communication pattern. If token routing is even slightly different between two runs — because a tie in the router's top-k was broken differently, or because a capacity-factor overflow dropped a different token — then the all-to-all moves different data, the experts see different batches, and the two runs diverge in a way no amount of kernel-level determinism can fix. The routing *is* the nondeterminism. The controls here are narrower: fix the tie-breaking rule, pin the capacity factor, and accept that some MoE implementations are simply not bitwise-reproducible without disabling the dynamic dispatch entirely. For MoE runs I target statistical, not bitwise, reproducibility and lean hard on resumability instead. Variable sequence length is the milder cousin: if you sort a batch by length to minimize padding, the sort must be stable and seeded, or two runs pad differently and the matmul shapes — and therefore the last bits — differ.

## 5. The two-seed rule: identical init, decorrelated noise

The seeding trap deserves a picture, because the requirement genuinely pulls in two directions from a single starting point.

![A branching and merging graph where one base seed forks into a model-init path that stays identical across ranks and an augmentation path that is offset by rank, then both merge into a reproducible and decorrelated run](/imgs/blogs/determinism-across-ranks-2.webp)

The figure traces the correct scheme. One base seed, chosen once. It forks. Down the left path is model construction: you seed with the *base seed, unmodified*, on every rank, and build the model. Because every rank runs `torch.manual_seed(1234)` and then constructs the same architecture, every rank gets bit-for-bit identical initial weights — which is exactly what DDP requires, since DDP broadcasts rank 0's weights at construction and then trusts that all replicas stay in lockstep. (Belt and suspenders: DDP *does* broadcast the initial state so init would be synced even if you seeded differently, but relying on that is fragile, and FSDP's sharded init is less forgiving. Seed identically and do not depend on the broadcast.)

Down the right path is everything stochastic that should differ per rank: the data-augmentation RNG, the dropout RNG, any sampling in the data pipeline. Here you seed with `base_seed + offset(rank)` — a per-rank offset so that rank 0 and rank 3 draw *different* random crops and dropout masks. If you skip this and every rank shares the augmentation seed, all eight GPUs apply the identical transform to their (different) data, and your effective augmentation diversity collapses by a factor of the world size. The two paths merge into a run that is both reproducible (rerun it and the same seeds produce the same everything) and correctly decorrelated (the ranks are not mirror images).

Here is the setup code that implements the two-seed rule. The *ordering* of operations matters: seed identically, build the model, then reseed with the rank offset before touching the data pipeline.

```python
import os
import random
import numpy as np
import torch
import torch.distributed as dist


def seed_identical_for_init(base_seed: int) -> None:
    """Call BEFORE constructing the model. Same on every rank."""
    random.seed(base_seed)
    np.random.seed(base_seed % (2**32))
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed_all(base_seed)


def seed_decorrelated_for_data(base_seed: int, rank: int) -> None:
    """Call AFTER the model exists, BEFORE building the dataloader.
    Offset per rank so augmentation/dropout differ across GPUs."""
    aug_seed = base_seed + 1000 * rank
    random.seed(aug_seed)
    np.random.seed(aug_seed % (2**32))
    torch.manual_seed(aug_seed)
    torch.cuda.manual_seed_all(aug_seed)


rank = dist.get_rank()
seed_identical_for_init(base_seed=1234)     # identical init on all ranks
model = build_transformer().cuda()          # bit-for-bit identical weights
model = torch.nn.parallel.DistributedDataParallel(model)
seed_decorrelated_for_data(base_seed=1234, rank=rank)   # now decorrelate
loader = build_dataloader(...)              # per-rank augmentation differs
```

One subtlety worth flagging: dropout inside the model draws from the *current* torch RNG at forward time, not at construction time. After you call `seed_decorrelated_for_data`, the torch RNG is per-rank, so dropout masks decorrelate too — which is what you want. If you had left the torch RNG at the base seed, every rank would draw identical dropout masks, a classic silent regularization bug. The [DDP internals and gotchas](/blog/machine-learning/distributed-training/ddp-internals-and-gotchas) post digs deeper into how `DistributedSampler` and seeding interact with DDP's bucketing; the seeding half of that post and this one are two views of the same machinery.

#### Worked example: the correlated-dropout bug

A vision team I worked with had a ResNet-style model training on 8 GPUs that plateaued about 0.4% below the single-GPU baseline at the same global batch. No error, no crash, just a persistent small gap that survived every hyperparameter sweep. The tell, once we looked, was almost invisible: their `worker_init_fn` seeded every worker with the *same* constant, and their model seeding used one seed for the whole job with no rank offset. Every one of the eight ranks was drawing the identical sequence of augmentation transforms and the identical dropout masks. Eight GPUs, one stream of "randomness."

The effect: their augmentation diversity was one-eighth of what they thought, and dropout — which is supposed to decorrelate the co-adaptation of features across the batch — was applying the same mask everywhere, weakening its regularization. The fix was four lines: offset the augmentation and worker seeds by `1000 * rank` and `base_seed + worker_id`. The gap closed to within run-to-run noise. The lesson is the shape of the whole post: a determinism bug does not have to break your run to *cost* you — this one quietly taxed every experiment the team ran for months. The prevention is the two-seed rule, applied deliberately, with the init seed identical and the noise seed decorrelated.

The `worker_init_fn` that makes dataloader workers both reproducible and decorrelated:

```python
def worker_init_fn(worker_id: int) -> None:
    # torch.initial_seed() already includes the per-rank base seed plus
    # a per-worker offset PyTorch adds internally; derive Python/NumPy
    # streams from it so all three libraries are reproducible and distinct.
    base = torch.initial_seed() % (2**32)
    np.random.seed(base + worker_id)
    random.seed(base + worker_id)


loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=16,
    sampler=torch.utils.data.distributed.DistributedSampler(
        dataset, shuffle=True, seed=1234, drop_last=True
    ),
    num_workers=8,
    worker_init_fn=worker_init_fn,
    persistent_workers=True,
)
```

And the piece everyone forgets, `set_epoch`, without which every epoch replays the same order:

```python
for epoch in range(num_epochs):
    loader.sampler.set_epoch(epoch)   # reshuffle deterministically per epoch
    for batch in loader:
        train_step(batch)
```

## 6. Same seed, two runs diverge — the mechanism you can see

Now let me walk the failure from the intro all the way to the fix, because it is the canonical distributed-determinism war story and every piece of the diagnosis generalizes.

![A before-and-after comparison showing two same-seed runs drifting apart at step 50 under default settings on the left and staying identical for 500 steps with a pinned reduction order and deterministic kernels on the right](/imgs/blogs/determinism-across-ranks-3.webp)

**The symptom.** Two launches, identical seed, identical code, same 8×A100 node. Loss at step 1 matches to the bit. By step 50 they differ in the fifth decimal. By step 200 they are 0.08 apart. The divergence *grows*, which immediately rules out a one-time data-order blip (that would be a single-step spike, not a growing gap) and points at something injected every step.

**The bisection.** There are only two things that touch every step and can differ run to run once the seed is pinned: nondeterministic kernels and the all-reduce reduction order. So I tested them one at a time. First I set `torch.use_deterministic_algorithms(True)` and reran twice. Still diverged — so the dominant source was not a local kernel. Then I pinned the collective with `NCCL_ALGO=Tree` and `NCCL_PROTO=Simple` and reran twice. The two runs now matched bit-for-bit for 500 steps. The all-reduce order was the culprit: NCCL had been choosing different algorithms or protocols across the two launches based on transient bandwidth measurements, summing the gradients in different orders, and the last-bit differences grew into 0.08 of loss over 200 steps.

The evidence was right there in the NCCL debug log, which I had not thought to read until the bisection pointed at the collective. Running with `NCCL_DEBUG=INFO` on the two launches and diffing the tuner lines showed NCCL selecting a different algorithm on each:

```console
# launch A, NCCL_DEBUG=INFO (grepped for the tuner decision)
NCCL INFO Channel 00/0 : 0 1 2 3 4 5 6 7
NCCL INFO Trees [0] 1/-1/-1->0->-1 [1] ...
NCCL INFO Connected all rings, using Ring/LL128 for allReduce
NCCL INFO comm 0x... rank 0 nranks 8 : chose Ring protocol LL128

# launch B, same code, same seed, minutes later
NCCL INFO Connected all trees, using Tree/Simple for allReduce
NCCL INFO comm 0x... rank 0 nranks 8 : chose Tree protocol Simple
```

Two launches of the identical program, and NCCL picked `Ring/LL128` on one and `Tree/Simple` on the other, purely because its internal tuner measured slightly different bandwidth at init time. Different algorithm, different summation order, different bits — exactly the mechanism from the previous section, caught red-handed. Pinning `NCCL_ALGO=Tree` and `NCCL_PROTO=Simple` forces both launches down the same path.

**The fix**, in the launch environment, set before the CUDA context is created:

```bash
# Pin every source of cross-run nondeterminism before launch.
export CUBLAS_WORKSPACE_CONFIG=:4096:8   # deterministic cuBLAS GEMMs
export NCCL_ALGO=Tree                    # pin the collective algorithm
export NCCL_PROTO=Simple                 # avoid LL / LL128 protocol switching
export PYTHONHASHSEED=0                   # stable Python hashing

torchrun --nproc_per_node=8 train.py --deterministic --seed 1234
```

And in the code, the matching in-process flags:

```python
def enable_determinism(seed: int) -> None:
    # cuBLAS workspace must be set in the env BEFORE the first CUDA call;
    # we assert it here so a forgotten export fails loudly.
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") in (":4096:8", ":16:8"), \
        "set CUBLAS_WORKSPACE_CONFIG=:4096:8 in the launch environment"

    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False          # no autotune roulette
    torch.backends.cuda.matmul.allow_tf32 = False   # TF32 varies across HW
    torch.backends.cudnn.allow_tf32 = False
```

**The stress test.** A fix you have not stress-tested is a guess. What happens to this recipe under harsher conditions?

- *At 64 GPUs across 8 nodes.* Now the all-reduce spans InfiniBand, not just NVLink, and NCCL has even more algorithm and protocol choices (it may go hierarchical: NVLink within a node, IB across nodes). Pinning `NCCL_ALGO` still works, but you may also need `NCCL_PROTO=Simple` to stop protocol switching on the inter-node hop, and the bitwise guarantee holds only if the *tree topology itself* is stable — which it is for a fixed set of nodes but changes if you reschedule onto different hardware. Bitwise determinism is a same-hardware, same-topology property.
- *On PCIe instead of NVLink.* The reduction *order* is still what matters, not the medium, so pinning the algorithm gives the same guarantee — you just pay more for the collective because the [interconnect is slower](/blog/machine-learning/distributed-training/collectives-from-scratch). Determinism and bandwidth are orthogonal.
- *With a tiny batch.* Smaller messages push NCCL toward the `LL`/`LL128` low-latency protocols, which is *exactly* the switching you are trying to suppress — so `NCCL_PROTO=Simple` matters *more* at small message sizes, at some latency cost.
- *When `use_deterministic_algorithms(True)` throws.* Some ops have no deterministic implementation and will raise an error under this flag rather than silently run nondeterministically. That is a feature: it tells you precisely which op is unpinnable. Your options are to restructure the model to avoid it, set the `CUBLAS_WORKSPACE_CONFIG` it demands, or accept `warn_only=True` and lose bit-for-bit at that op.

#### Worked example: two runs that diverge at step 50

To put numbers on the whole arc: two launches of a 1.3B model on 8×A100, same seed `1234`, same code. Step-1 loss matched to 2.4e-7 (already not bitwise, but within kernel wobble). By step 50 the gap was 3e-4; by step 200 it was 0.08 — bigger than the 1% regression I was hunting. Enabling `use_deterministic_algorithms(True)` alone dropped the step-200 gap to about 9e-3: better, because it killed the local atomic kernels, but not fixed, because the all-reduce order still varied. Adding `NCCL_ALGO=Tree` and `NCCL_PROTO=Simple` dropped it to 0.0 — bit-for-bit identical for the full 500-step check. The cost of buying that identity was measured separately (§9): about 21% throughput on this model. The prevention rule that falls out: never trust a baseline you have not proven reproducible with a two-run assert, and when it fails, read the `NCCL_DEBUG=INFO` tuner line *before* you start guessing — it names the algorithm, and the algorithm is usually the answer.

## 7. The three levels of reproducibility — and what each costs

Here is the part most treatments skip, and it is the part that saves you the most time: **you almost never want full bitwise determinism.** Reproducibility comes in levels, each with a different cost, and picking the wrong level either wastes throughput or fails to give you what you needed.

![A vertical ladder of reproducibility levels rising from a run with no guarantee through resumable and statistical match up to single-node and multi-node bitwise identity with the cost climbing at each rung](/imgs/blogs/determinism-across-ranks-4.webp)

The ladder, from cheapest to costliest:

**Level 0 — no guarantee.** Seeds unset, nothing pinned. Every run is different. Fine for a one-off, useless for debugging. You are here by default if you do nothing.

**Resumable.** A checkpoint resumes onto the *same trajectory* the uninterrupted run would have followed. This does not require bitwise determinism across a fresh launch — it requires that you save and restore *all* the state: model, optimizer, LR scheduler, sampler epoch, and every RNG stream (Python, NumPy, torch CPU, and per-device CUDA). Cost: a bit more checkpoint size and code. This is the level that prevents the [loss spike after resume](/blog/machine-learning/distributed-training/the-loss-spike-after-resume) — go read that post for the full checkpoint-state catalog, because "resumable" is a determinism property that most people get wrong by omitting the optimizer or RNG state.

**Statistical match.** Two runs produce the *same loss curve within noise* — not bit-for-bit, but indistinguishable at the level of your metrics. This is the level you want for almost all debugging and for reporting results honestly. It costs essentially nothing at runtime: seed everything (the two-seed rule), use `DistributedSampler` with `set_epoch`, fix the data order. You leave nondeterministic kernels and the exact all-reduce order *alone*, accept that the last bits wobble, and rely on the fact that a correct run is robust to last-bit wobble. If your two "statistically identical" runs diverge by more than sampling noise, that itself is a bug signal.

**Bitwise identical (single node).** Two runs match to the last bit for K steps. This needs everything in the statistical level *plus* `torch.use_deterministic_algorithms(True)`, `cudnn.benchmark=False`, `CUBLAS_WORKSPACE_CONFIG`, and a pinned `NCCL_ALGO`. Cost: 10–30% slower (measured below) and sometimes more memory. Worth it when you are hunting a *silent correctness* bug and need to compare two builds exactly, or when you are validating that a refactor changed nothing.

**Bitwise identical (multi-node).** The same, but the reduction now spans the network, so you also pin `NCCL_PROTO` and accept that the guarantee is tied to a fixed node set and topology. This is the hardest and rarest, and I have needed it maybe twice — both times to prove a specific node was corrupting an all-reduce.

The single most common mistake is aiming for bitwise when you needed statistical, eating the 10–30% slowdown for no debugging benefit, or aiming for a fresh-launch match when what you actually needed was *resumable*. Match the level to the job.

| Level | What it guarantees | Runtime cost | When you need it |
|---|---|---|---|
| No guarantee | Nothing | 0% | A throwaway run |
| Resumable | Resume = uninterrupted run | ~0% (bigger ckpt) | Every real long run |
| Statistical | Same loss curve ± noise | ~0% | Almost all debugging, honest reporting |
| Bitwise (1 node) | Same bits for K steps | 10–30% slower | Silent-correctness bug hunts, refactor validation |
| Bitwise (multi-node) | Same bits across ranks | 10–30% + topology-locked | Proving a bad node / corrupt all-reduce |

## 8. The complete determinism recipe

Let me assemble the full setup into one place you can copy. This targets the *bitwise single-node* level — the strictest you would normally reach for during a bug hunt. Strip out the deterministic-algorithms lines to drop to the statistical level for production.

```python
import os
import random
import numpy as np
import torch
import torch.distributed as dist


def full_determinism(base_seed: int, rank: int, strict: bool = True) -> None:
    """Configure a distributed run for reproducibility.
    strict=True  -> bitwise (slower). strict=False -> statistical (fast).
    """
    # (a) cuBLAS workspace: must be in the env before the first CUDA op.
    if strict:
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") in (":4096:8", ":16:8")

    # (b) Two-seed rule, part 1: identical init on every rank.
    #     Call this BEFORE constructing the model.
    random.seed(base_seed)
    np.random.seed(base_seed % (2**32))
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed_all(base_seed)

    # (c) Deterministic kernels + no autotune (the costly, strict-only part).
    if strict:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def reseed_for_data(base_seed: int, rank: int) -> None:
    """Two-seed rule, part 2: decorrelate augmentation/dropout per rank.
    Call AFTER the model is built, BEFORE the dataloader."""
    aug_seed = base_seed + 1000 * rank
    random.seed(aug_seed)
    np.random.seed(aug_seed % (2**32))
    torch.manual_seed(aug_seed)
    torch.cuda.manual_seed_all(aug_seed)
```

The launch script that pairs with it:

```bash
#!/usr/bin/env bash
# Pin cross-run and cross-rank nondeterminism BEFORE any CUDA context.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NCCL_ALGO=Tree
export NCCL_PROTO=Simple
export PYTHONHASHSEED=0
# Optional: surface what NCCL actually chose, to confirm the pin took.
export NCCL_DEBUG=WARN

torchrun \
  --nproc_per_node=8 \
  --nnodes=1 \
  --rdzv_backend=c10d \
  train.py --seed 1234 --strict-determinism
```

Now the part that turns all of this from hope into a *guarantee*: the correctness test. Determinism you have not tested is determinism you do not have. The test is simple — run K steps twice under identical settings, record the per-step loss, and assert the two signatures agree to a tolerance. At the bitwise level the tolerance is `1e-6` or tighter; at the statistical level you widen it to your metric's noise floor.

```python
def loss_signature(steps: int = 200, seed: int = 1234) -> torch.Tensor:
    """Train `steps` steps deterministically; return the per-step losses."""
    full_determinism(base_seed=seed, rank=dist.get_rank(), strict=True)
    model = build_transformer().cuda()
    model = torch.nn.parallel.DistributedDataParallel(model)
    reseed_for_data(seed, dist.get_rank())
    loader = build_dataloader(seed=seed)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    losses = []
    for step, batch in zip(range(steps), loader):
        loss = model(batch).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        # Reduce the scalar so every rank checks the SAME number.
        dist.all_reduce(loss, op=dist.ReduceOp.AVG)
        losses.append(loss.detach().float().cpu())
    return torch.stack(losses)


def assert_reproducible(tol: float = 1e-6) -> None:
    a = loss_signature()
    b = loss_signature()
    max_delta = (a - b).abs().max().item()
    if dist.get_rank() == 0:
        print(f"max |Δloss| over 200 steps = {max_delta:.2e}")
    assert max_delta < tol, (
        f"NONDETERMINISTIC: max |Δloss| = {max_delta:.2e} > {tol:.0e}. "
        "Check NCCL_ALGO, use_deterministic_algorithms, and seeding."
    )
```

I run `assert_reproducible()` in CI on a two-GPU runner for a short K, and as a manual pre-flight before any bug hunt on the full node. The first time it fails you will be glad it is a five-line assert and not a two-day mystery like the one that opened this post. When it fails, the *magnitude* of `max_delta` is a clue: `1e-7`-ish means last-bit kernel wobble (turn on deterministic algorithms), while a jump to `1e-3` by step 200 means the all-reduce order is unpinned (set `NCCL_ALGO`).

A passing and a failing run look like this — the failing one immediately tells you which knob is loose by the size of the delta:

```log
# strict=True, NCCL_ALGO pinned  -> PASS
max |Δloss| over 200 steps = 4.17e-07
PASSED: runs are bitwise-reproducible within tol=1e-06

# strict=True but NCCL_ALGO unset -> FAIL, and the magnitude names the cause
max |Δloss| over 200 steps = 7.90e-03
AssertionError: NONDETERMINISTIC: max |Δloss| = 7.90e-03 > 1e-06.
Check NCCL_ALGO, use_deterministic_algorithms, and seeding.
```

### Saving RNG state for a reproducible resume

The recipe above buys reproducibility for a *fresh* launch. The other half of the story — the resumable level of the ladder — needs the checkpoint to carry every RNG stream so that resuming continues the same trajectory. Weights and optimizer state are the pieces everyone remembers; the RNG streams are the pieces everyone forgets, and forgetting them means the data order and dropout masks restart from a different point, quietly biasing the rest of training. Save all of it:

```python
def save_checkpoint(path, model, optimizer, scheduler, epoch, step):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,               # so sampler.set_epoch resumes right
            "step": step,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),  # per-device streams
        },
        path,
    )


def load_checkpoint(path, model, optimizer, scheduler):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    random.setstate(ckpt["python_rng"])
    np.random.set_state(ckpt["numpy_rng"])
    torch.set_rng_state(ckpt["torch_rng"])
    torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
    return ckpt["epoch"], ckpt["step"]
```

Note `get_rng_state_all` / `set_rng_state_all` — the *plural* — because each rank has its own CUDA RNG stream and a distributed resume must restore each rank's stream on its own device. The full catalog of what a checkpoint must carry, and the precise shape of the loss spike you get when you omit the optimizer state, is the subject of [the loss spike after resume](/blog/machine-learning/distributed-training/the-loss-spike-after-resume); this snippet is the RNG-specific slice of that checklist.

## 9. The cost of determinism, measured

Determinism is slower, and you should know by how much before you flip the switch in production by accident. Here are representative before→after numbers from a 1.3B-parameter GPT-style model on a single 8×A100 80GB SXM node with NVLink, bf16 autocast, sequence length 2048, global batch 128. Treat the absolute tokens/s as illustrative of *my* setup and the *relative* slowdowns as the portable lesson; your exact percentages depend on the model and the ops it hits.

| Configuration | Tokens/s (8×A100) | Rel. throughput | MFU | Reproducibility |
|---|---|---|---|---|
| Default (benchmark=True, no det) | ~118,000 | 100% | ~42% | none |
| Statistical (seed + set_epoch only) | ~118,000 | 100% | ~42% | same curve ± noise |
| + cudnn.benchmark=False | ~112,000 | 95% | ~40% | fewer sources |
| + use_deterministic_algorithms | ~97,000 | 82% | ~34% | bitwise (1 node) |
| + NCCL_ALGO=Tree, PROTO=Simple | ~93,000 | 79% | ~33% | bitwise (multi-run) |

The headline: statistical reproducibility is *free*, and full bitwise reproducibility cost about 21% throughput here — squarely in the 10–30% band the community reports. The two biggest bites are `use_deterministic_algorithms` (which forces slower deterministic kernels and disables some fused paths) and pinning `NCCL_ALGO` (which forbids NCCL from picking the fastest collective for each message size). `cudnn.benchmark=False` costs little on a transformer with stable shapes but can cost more on convnets with many input sizes.

There is a second cost that the throughput column hides: **memory.** Some deterministic kernels need a larger workspace than their nondeterministic counterparts, and `CUBLAS_WORKSPACE_CONFIG=:4096:8` reserves a fixed cuBLAS workspace per stream. On a run that is already close to the memory ceiling, flipping on strict determinism can be the difference between fitting and an out-of-memory crash — one more reason to reserve it for short validation runs on models that have headroom, not the production job that is packed to 78 GB of an 80 GB card. If a determinism flag OOMs you, that is not a determinism bug, it is the workspace tax, and the fix is a smaller micro-batch for the validation run rather than abandoning the reproducibility check.

**How to measure this honestly**, because a careless benchmark will lie to you:

- *Warm up.* The first 10–20 steps include CUDA context creation, cuDNN autotuning (in benchmark mode), and allocator warmup. Throw them away and time steady state.
- *Synchronize before you read the clock.* CUDA kernels are asynchronous; `torch.cuda.synchronize()` before every `time.perf_counter()` or you are timing kernel *launches*, not kernel *execution*.
- *Watch the data loader.* If the loader can't keep the GPU fed, you are measuring the loader, not the model — and `worker_init_fn` plus deterministic settings can change loader behavior. Confirm GPU utilization is pinned near 100% with `nvidia-smi dmon` during the timed window.
- *Hold the clocks steady.* Thermal throttling and dynamic clocks make two identical runs report different tokens/s for reasons that have nothing to do with determinism. Lock clocks with `nvidia-smi -lgc` for a benchmark, or at least run long enough to reach thermal steady state.

#### Worked example: the determinism budget on a real bug hunt

I once needed to prove that a fused-attention kernel swap changed *nothing* mathematically — that the new kernel was a drop-in. Statistical match was not enough: "the loss curve looks the same" cannot distinguish "identical" from "different in the third decimal," and a subtle attention bug is exactly the kind that hides in the third decimal. So I paid for the bitwise level. I ran `assert_reproducible(tol=1e-6)` with the old kernel to establish that my harness was actually deterministic (it was, after I pinned `NCCL_ALGO`), then swapped in the new kernel and compared the two loss signatures. They matched to `4e-7` over 200 steps — proof the swap was mathematically neutral. The bitwise run was 21% slower, but I only needed 200 steps, so the whole validation cost about ninety seconds of wall-clock and a couple of dollars of GPU time. That is the right trade: pay the determinism tax on a *short* validation run, never on the 100,000-step production job. Turn `strict` back off and the production run reclaims its 21%.

## 10. Reproduce before you debug — the workflow

Everything above composes into a single discipline, and it is worth stating as a workflow because the *order* is what makes it work.

![A left-to-right timeline of the reproduce-first workflow that fixes seeds and determinism, runs twice, asserts the losses match, bisects any divergence to its source, pins it, re-verifies, and only then bisects the real bug](/imgs/blogs/determinism-across-ranks-6.webp)

The principle is *reproduce before you debug*. Before you go hunting the bug you actually care about, you first establish that your harness is deterministic to the level you need. The steps: fix all seeds and enable determinism to your chosen level; run the identical config twice for K steps; assert the loss signatures match within tolerance. If they *do not* match, you have found a bug in your *harness* first — bisect the divergence to its source (a `1e-7` wobble is kernels, a growing gap is the all-reduce order, a step-1 mismatch is data order or seeding), pin it, and re-verify. Only once two runs provably agree do you have a baseline you can trust — and *now* bisecting the real bug (a config change, a code change, a kernel swap) is a clean controlled experiment, because the baseline holds still while you vary one thing.

The failure mode this prevents is the two-day one from the intro: charging straight into bisecting a regression against a baseline that was never stable, and burning a day before realizing the noise floor was above the signal. Fifteen minutes of `assert_reproducible` up front would have saved that day. The discipline is cheap and the payoff is the entire premise of debugging: a controlled experiment requires a controlled control.

## 11. The sources-and-controls checklist

Here is the whole catalog condensed to a paste-into-your-runbook form: every source, its symptom, its knob, its cost.

![A five-row matrix mapping each nondeterminism source to how it bites a distributed run, the specific PyTorch or NCCL knob that controls it, and the honest cost of that control](/imgs/blogs/determinism-across-ranks-5.webp)

| Source | How it bites | The knob | Cost |
|---|---|---|---|
| RNG seeds | Different init/masks per run and per rank | `manual_seed` all libs; rank offset for aug | Free — always do it |
| Nondet kernels | Atomics in backward → bitwise drift every step | `use_deterministic_algorithms(True)` | 10–30% slower; debug only |
| cuDNN autotune | Benchmark picks a different algo per run | `cudnn.benchmark=False` | Small; loses autotune on convnets |
| All-reduce order | FP non-associative; ranks summed in different order | `NCCL_ALGO`, `NCCL_PROTO` pinned | Small–moderate; forbids fastest collective |
| Data order | Shard + shuffle drift; forgotten `set_epoch` | `DistributedSampler` + `set_epoch(e)` | Free — but must get right |
| Checkpoint state | Resume loses RNG/optimizer → trajectory jumps | Save & restore all RNG + optimizer state | Bigger checkpoint |
| TF32 / precision | TF32 differs across GPU generations | `allow_tf32=False` for cross-HW repro | Slower matmul |
| World size | More ranks → bigger global batch → different math | Accept it; compare at fixed world size | N/A — it's a hyperparameter |

The matrix figure carries the top five — the ones with a specific PyTorch or NCCL switch — and the table adds the checkpoint, precision, and world-size rows that are facts to accept rather than flags to flip.

The one row people most often misread is TF32. TF32 is not nondeterministic *within* a run — it does not wobble run to run on the same GPU. But it makes results differ *across GPU generations* (an A100 and an H100 round TF32 matmuls differently), so if your definition of reproducible includes "same result on different hardware," you must disable it, at a real matmul-speed cost. If your reproducibility target is same-hardware, leave TF32 on and keep the speed.

## 12. Case studies and real numbers

A few grounding points from the literature and the tools themselves, so this is not just my war stories.

**PyTorch's own reproducibility contract.** The PyTorch documentation is explicit that "completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms," and that even on one platform you must set the seeds, call `torch.use_deterministic_algorithms(True)`, disable `cudnn.benchmark`, and set `CUBLAS_WORKSPACE_CONFIG=:4096:8` for deterministic cuBLAS GEMMs. That `:4096:8` value is not folklore — it is the documented setting that forces cuBLAS to use a fixed workspace and therefore a deterministic reduction split. The docs also list which operations have no deterministic implementation and will raise under the flag, which is your map of what you cannot pin without restructuring.

**NCCL and cross-run reduction.** NCCL does not promise bit-for-bit identical all-reduce results across runs by default, precisely because it may select a different algorithm (ring vs. tree) or protocol (`Simple`, `LL`, `LL128`) based on message size and measured bandwidth, and those sum in different orders. Pinning `NCCL_ALGO` and `NCCL_PROTO` restores order-stability at the cost of forbidding the autotuned fastest path. This is the distributed-specific knob that has no single-GPU analog, and it is the one most people have never touched — which is why the "same seed, diverges anyway" bug is so common and so confusing the first time.

**Large-model reproducibility statements.** Big pretraining efforts — the Megatron-LM, OPT, and BLOOM writeups among them — publish their exact seeds, data order, and framework versions specifically so a run can be reproduced, and several document loss-curve *statistical* reproducibility rather than bitwise, because at scale, across a changing cluster with occasional node replacement, bitwise is not a realistic target. The community norm at scale is: pin what you can, seed everything, save full checkpoint state for resumability, and target statistical rather than bitwise reproducibility. The OPT logbook in particular is a candid record of how many restarts and hardware swaps a long run actually endures — which is exactly why "resumable" is the load-bearing reproducibility level for large runs, not "bitwise."

**Floating-point non-associativity is textbook.** The order-dependence of floating-point addition is not a PyTorch quirk; it is covered in Goldberg's "What Every Computer Scientist Should Know About Floating-Point Arithmetic." The distributed twist is only that the *order* is now decided by a collective algorithm running across a network, which is more variable and less visible than the order inside one GPU's reduction tree.

**The broader reproducibility crisis in ML is real and mundane.** Surveys of published deep-learning results repeatedly find that a large fraction cannot be reproduced from the paper alone, and when researchers dig in, the culprits are almost never exotic — they are unreported seeds, unpinned library versions, undocumented data order, and exactly the nondeterministic-kernel and reduction-order effects cataloged here. The distributed setting makes this worse, not better, because there are more moving parts and fewer people who know to pin the collective. The practical antidote is unglamorous and entirely within your control: log the seed, the world size, the framework and CUDA versions, and the NCCL settings into every run's metadata, and keep the five-line reproducibility assert in CI. A result you can reproduce is a result you can defend; a result you cannot is a story.

## 13. When to reach for determinism, and when not to

Every knob in this post is a cost, so the decisive question is *which level for which job*. The answer follows almost mechanically from what you are trying to do.

![A decision tree that routes a debugging task to statistical reproducibility, a regression or correctness hunt to bitwise, and a production run to resumable-only](/imgs/blogs/determinism-across-ranks-7.webp)

- **Debugging most things → statistical match.** Seed everything, use `DistributedSampler` with `set_epoch`, fix the data order, and *stop there*. Leave kernels and NCCL alone. You get a run whose loss curve reproduces within noise, at zero runtime cost, which is enough to bisect the overwhelming majority of bugs. Do *not* pay the bitwise tax here.
- **Silent-correctness hunt or refactor validation → bitwise, briefly.** When you must prove two builds are mathematically identical — a kernel swap, a refactor, a "did this change anything?" question — pay for bitwise, but only on a *short* K-step validation, never the full run. The 21% slowdown on 200 steps is ninety seconds; the same slowdown on 100,000 steps is real money and real calendar time.
- **Production run → resumable only.** The full-length job needs one property above all: it must survive interruptions without a [loss spike on resume](/blog/machine-learning/distributed-training/the-loss-spike-after-resume). Save and restore all RNG and optimizer state, target statistical reproducibility for your metrics, and leave the throughput-eating determinism flags *off* so you get full MFU. Turning on `use_deterministic_algorithms` for a production pretraining run is a common and expensive mistake — you pay 10–30% for a guarantee the run does not need.

And the honest "don't bother" cases, because determinism has diminishing returns:

- **Don't chase bitwise across GPU generations or framework versions.** It is not guaranteed and rarely worth the fight; target statistical and pin your versions instead.
- **Don't expect two different world sizes to match.** Eight and sixteen GPUs are different optimization problems. Reproducibility is always "same result at the same world size."
- **Don't pay for bitwise when statistical answers your question.** This is the single most common over-spend. If "the curves overlap within noise" settles it, you are done.
- **Don't skip resumability to save checkpoint bytes.** Weights-only checkpoints are the classic false economy — the [resume spike](/blog/machine-learning/distributed-training/the-loss-spike-after-resume) costs far more compute than the optimizer state costs disk.

The [debugging distributed jobs](/blog/machine-learning/distributed-training/debugging-distributed-jobs) post is the sibling that uses this reproducibility foundation to actually localize hangs and crashes, and the [capstone playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook) folds the determinism checklist into the end-to-end decision flow for a real run.

## 14. Key takeaways

- **Reproducibility is the debugging superpower.** A run you cannot reproduce cannot be bisected, and a silent correctness bug cannot be isolated. Everything else in distributed debugging assumes it.
- **Distributed adds sources single-GPU does not have:** the all-reduce reduction order (floating-point addition is non-associative, and NCCL may sum ranks in different orders), per-rank data order, per-rank seeding, world-size effects, and dynamic control flow.
- **The two-seed rule:** model init must be *identical* across ranks (same base seed, before construction); augmentation and dropout must be *decorrelated* (base seed plus a per-rank offset, before the dataloader). Getting it backwards is a silent regularization bug.
- **`set_epoch` is not optional.** Without `sampler.set_epoch(epoch)`, `DistributedSampler` replays the same order every epoch — deterministic and wrong.
- **Pin the collective for cross-run bitwise identity.** `NCCL_ALGO` and `NCCL_PROTO` are the distributed-only knobs with no single-GPU analog; they are why "same seed, diverges anyway" happens.
- **Reproducibility is a ladder:** no guarantee → resumable → statistical → bitwise (1 node) → bitwise (multi-node). Match the level to the job; almost everything needs only statistical plus resumable.
- **Full determinism costs 10–30% throughput.** Enable it for short bug-hunt validations, not for production. Statistical reproducibility is free.
- **Test it or you don't have it.** Two runs, same seed, K steps, assert the loss signatures match within tolerance — five lines that save two days.
- **Reproduce before you debug.** Establish a stable baseline first; only then vary one thing. A controlled experiment needs a controlled control.

## 15. Further reading

- **PyTorch reproducibility notes** — the canonical list of seeds, `use_deterministic_algorithms`, `cudnn.benchmark`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and the operations with no deterministic implementation.
- **NCCL documentation** — `NCCL_ALGO`, `NCCL_PROTO`, and the environment variables that pin collective behavior; read alongside [collectives from scratch](/blog/machine-learning/distributed-training/collectives-from-scratch) for the ring-vs-tree byte-volume math.
- **Goldberg, "What Every Computer Scientist Should Know About Floating-Point Arithmetic"** — why $(a+b)+c \neq a+(b+c)$ and why order controls the last bits.
- **The single-GPU companion** — [reproducibility and determinism in training](/blog/machine-learning/debugging-training/reproducibility-and-determinism-in-training) for the non-distributed half of this story in depth.
- **[DDP internals and gotchas](/blog/machine-learning/distributed-training/ddp-internals-and-gotchas)** — how `DistributedSampler`, seeding, and bucketing interact inside DDP.
- **[The loss spike after resume](/blog/machine-learning/distributed-training/the-loss-spike-after-resume)** — the resumability level of the ladder, and the complete checkpoint-state catalog.
- **[Why distributed training](/blog/machine-learning/distributed-training/why-distributed-training)** and the **[distributed training playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook)** — the series intro and capstone this post plugs into.
- **The OPT-175B and BLOOM training logbooks** — candid records of restarts, hardware swaps, and why "resumable" and "statistical" are the reproducibility levels that survive a real large-scale run.
