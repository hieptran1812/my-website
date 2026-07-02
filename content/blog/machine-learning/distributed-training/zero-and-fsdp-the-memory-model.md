---
title: "ZeRO and FSDP: The Memory Model That Lets a 70B Model Fit"
date: "2026-07-02"
publishDate: "2026-07-02"
description: "Derive the famous sixteen-bytes-per-parameter memory law by hand, then watch ZeRO and FSDP shard it away stage by stage until a 70B model fits on commodity 80GB cards — with the exact comms bill you pay for it."
tags:
  [
    "distributed-training",
    "multi-gpu",
    "fsdp",
    "deepspeed",
    "zero",
    "memory",
    "pytorch",
    "ddp",
    "deep-learning",
    "ml-systems",
  ]
category: "machine-learning"
subcategory: "Distributed Training"
author: "Hiep Tran"
featured: true
readTime: 41
---

You load a 7-billion-parameter model onto a fresh A100 with 80 GB of memory, wrap it in `DistributedDataParallel`, call `AdamW`, and launch. Before the first forward pass even completes, the process dies: `CUDA out of memory. Tried to allocate 2.00 GiB`. You stare at the traceback. Seven billion parameters in bf16 is fourteen gigabytes. You have eighty. The model is one-sixth the size of the card. How is it possible that you are out of memory?

The answer is that the fourteen gigabytes of weights you were counting are the *smallest* thing on the card. Mixed-precision training with Adam keeps five separate copies of every parameter, most of them in fp32, and the total comes to sixteen bytes per parameter — not two. Seven billion parameters therefore need 112 GB of pure optimizer and gradient state before you have stored a single activation, and 112 GB does not fit in 80. This is not a bug in your code. It is the arithmetic of the optimizer, and until you can do that arithmetic on the back of an envelope you will keep being surprised by OOMs you cannot explain.

![a vertical stack showing bf16 weights and gradients at two bytes each plus three fp32 optimizer copies at four bytes each summing to sixteen bytes per parameter](/imgs/blogs/zero-and-fsdp-the-memory-model-1.webp)

This post is about the single most important piece of arithmetic in large-model training and the two systems built to defeat it. First we derive the sixteen-bytes-per-parameter law term by term, so you know exactly where every gigabyte goes. Then we notice the waste: `DistributedDataParallel` replicates all sixteen bytes on *every* GPU, so eight GPUs store eight identical copies of state that only needs to exist once. ZeRO — the Zero Redundancy Optimizer from DeepSpeed — and its PyTorch-native twin FSDP fix that by **sharding** the state so each of `N` GPUs holds only `1/N` of it. By the end you will be able to compute the exact per-GPU memory of DDP, ZeRO-1, ZeRO-2, and ZeRO-3 for any model and cluster; explain the all-gather-compute-free mechanism that lets a parameter live sharded and still be used; price the extra communication that ZeRO-3 charges (it is exactly 1.5x DDP, and we will prove it); write a working FSDP wrap and a DeepSpeed ZeRO config; and decide, for a given model and cluster, which stage is the cheapest one that fits. This is the eighth post in the [Distributed Training in the Trenches](/blog/machine-learning/distributed-training/why-distributed-training) series, and it is the one that turns "the model won't fit" — the first of the [four walls](/blog/machine-learning/distributed-training/why-distributed-training) — from a dead end into a knob you can turn.

## Where every byte of memory actually goes

Let `Ψ` (psi) be the number of parameters in your model — 7 billion for a 7B model, 70 billion for a 70B one. We are going to account for every byte of memory a standard mixed-precision Adam training step consumes per parameter, and the sum will be the famous ${(2 + 2 + 12)\Psi = 16\Psi}$ bytes. The figure above is the whole derivation as a stack; here is the same thing in words, one term at a time.

**The bf16 weights: 2Ψ bytes.** In mixed-precision training the forward and backward passes run in bf16 (or fp16) because the tensor cores are two to four times faster in 16-bit than in fp32, and because half the bytes means half the memory traffic. bf16 is two bytes per number. So the copy of the weights that the GPU actually multiplies with costs ${2\Psi}$ bytes. For a 7B model that is 14 GB — the number you were counting when you were surprised by the OOM.

**The bf16 gradients: 2Ψ bytes.** The backward pass produces one gradient per parameter, and it lives in the same precision as the weights it flows through, so gradients are also bf16: another ${2\Psi}$ bytes, another 14 GB at 7B. These exist for the duration of the backward pass and the optimizer step; you cannot free them until the optimizer has consumed them.

Now the part that surprises people — the optimizer states, all of which live in **fp32** (four bytes), and there are three of them.

**The fp32 master weights: 4Ψ bytes.** Here is the subtlety that trips up everyone the first time. You cannot actually keep your *canonical* weights in bf16. bf16 has only about 7 bits of mantissa, so once your weights are large and your updates are small, the update `w -= lr * grad` rounds to zero: `1.0000` in bf16 simply cannot represent `1.0000 + 0.0001`, so the addition is a no-op and training silently stalls. The fix, which every serious mixed-precision recipe uses, is to keep a **master copy of the weights in fp32** and apply the optimizer update to *that*, then round it down to bf16 for the next forward pass. The bf16 weights are a disposable view; the fp32 master is the truth. That master copy is ${4\Psi}$ bytes — 28 GB at 7B, already twice the size of the model you thought you were training.

**The fp32 Adam momentum: 4Ψ bytes.** Adam keeps a running exponential average of the gradient — the first moment, `m`. One fp32 number per parameter: ${4\Psi}$ bytes, 28 GB at 7B.

**The fp32 Adam variance: 4Ψ bytes.** Adam also keeps a running average of the *squared* gradient — the second moment, `v`. Another fp32 number per parameter: ${4\Psi}$ bytes, another 28 GB at 7B.

Add the three fp32 optimizer states and you get ${4\Psi + 4\Psi + 4\Psi = 12\Psi}$ bytes. This is the `12` in the famous formula, and it is the reason large-model training is a memory problem before it is a compute problem. The ZeRO paper writes it as `KΨ` with `K = 12` for Adam; other optimizers change `K` but not the structure. Plain SGD with momentum keeps only `m`, so `K = 8` (master + momentum, both fp32, plus... actually master 4 + momentum 4 = 8). SGD without momentum keeps only the master, `K = 4`. Adam's `K = 12` is the expensive common case, and it is what almost everyone trains transformers with.

The grand total per parameter:

$$
M_\text{per-param} = \underbrace{2}_{\text{bf16 weights}} + \underbrace{2}_{\text{bf16 grads}} + \underbrace{4}_{\text{fp32 master}} + \underbrace{4}_{\text{fp32 momentum}} + \underbrace{4}_{\text{fp32 variance}} = 16 \text{ bytes}
$$

Written the way the ZeRO paper does it, that is ${(2 + 2 + 12)\Psi = 16\Psi}$ bytes. Memorize the grouping, not just the sum: `2` for the working weights, `2` for the gradients, `12` for the fp32 optimizer trio. When we start sharding, we will shard those three groups independently and in that order, and the grouping is the whole map.

#### Worked example: why a 7B model OOMs on an 80GB card

Take ${\Psi = 7 \times 10^9}$. The state alone is ${16 \times 7 \times 10^9 = 1.12 \times 10^{11}}$ bytes = **112 GB**. That already exceeds an 80 GB A100 by 40%, and we have not counted activations yet. Activations — the intermediate tensors the forward pass saves so the backward pass can compute gradients — scale with batch size, sequence length, and model depth, and for a 7B transformer at a reasonable sequence length they add tens of gigabytes more (this is what [activation checkpointing](/blog/machine-learning/distributed-training/the-memory-budget) exists to fight). So the real footprint of naive 7B Adam training is closer to 130–160 GB per GPU. On an 80 GB card it is not close. That `CUDA out of memory` in the intro was not mysterious at all — it was 112 GB of state plus activations trying to fit in 80 GB, and the arithmetic said no before you ran a single step.

The lesson generalizes to a rule you should carry everywhere: **the model's parameter count times sixteen is your Adam mixed-precision floor, in bytes, per GPU, before activations.** A 1.5B model: 24 GB — fits comfortably. A 13B model: 208 GB — needs sharding across at least three 80 GB cards. A 70B model: 1120 GB — needs at least fourteen 80 GB cards just for the state. That last number is why you cannot train a 70B model the way you trained your first CNN, and it is the problem the rest of this post solves.

### The formula generalizes: other optimizers, other precisions

The `16` is specific to bf16-plus-Adam, which is the common case, but the *structure* — `(weights) + (grads) + (optimizer)` — holds for everything, and it pays to know how the number moves so you are never surprised.

**Change the optimizer, change the `12`.** Adam's `K = 12` comes from three fp32 states (master, momentum, variance). Plain SGD keeps no moments, so `K = 4` (just the fp32 master), and mixed-precision SGD is `2 + 2 + 4 = 8` bytes per parameter — half of Adam. SGD with momentum keeps one moment, `K = 8`, total `2 + 2 + 8 = 12`. This is one reason some very large models were trained with plain SGD or memory-frugal optimizers: the optimizer state is often the single largest term, and halving it is the cheapest memory win there is.

**Use an 8-bit optimizer, and the `12` nearly halves.** Libraries like bitsandbytes implement 8-bit Adam, which stores the momentum and variance in quantized 8-bit form (one byte each) instead of fp32, blockwise-dequantized on the fly for the update. That turns the momentum-plus-variance cost from `4 + 4 = 8` bytes down to `1 + 1 = 2` bytes, leaving roughly `2 + 2 + (4 + 1 + 1) = 10` bytes per parameter — and if you keep the master in bf16 too, less. It is a memory lever *orthogonal* to sharding: you can stack 8-bit Adam on top of ZeRO to push the per-GPU floor down even further. For a model that just barely does not fit, an 8-bit optimizer sometimes saves you from having to climb a ZeRO stage at all.

**Train in full fp32, and everything doubles.** Pure fp32 training keeps `4` byte weights and `4` byte grads with no separate master copy, so it is `4 + 4 + 8 = 16` bytes as well but with a *different* structure (there is no bf16/fp32 split, and Adam's two moments are `4 + 4`). The reason mixed precision still wins on memory despite matching this number is the compute: fp32 matmuls run at a fraction of the tensor-core throughput, and the activations — which we have been ignoring — are twice as large in fp32. Precision is a memory *and* a speed lever; we give it a whole post in the [memory-budget](/blog/machine-learning/distributed-training/the-memory-budget) track.

The habit to build: when someone quotes you a model size, immediately multiply by the right `K + 4` to get the state floor, divide by your GPU memory, and you know instantly how many GPUs you must shard across before you have written a line of code. That single multiplication is the most useful reflex in large-model training.

## The waste: DDP replicates all sixteen bytes on every GPU

Here is the thing about `DistributedDataParallel`, the workhorse of [data-parallel training](/blog/machine-learning/distributed-training/collectives-from-scratch): it makes eight GPUs faster by having each one process a different slice of the batch, but it does so by giving each GPU a **complete, independent copy** of the model. Every GPU holds the full bf16 weights. Every GPU holds the full bf16 gradients. Every GPU holds the full fp32 master, momentum, and variance. Eight GPUs running DDP on a 7B model are storing 112 GB of state *eight times* — 896 GB of aggregate memory to train a model whose state only needs to exist once.

That is the redundancy the Zero Redundancy Optimizer is named after. Think of it this way: DDP replicates the model so that each GPU can do its forward and backward pass entirely locally, communicating only once per step to average the gradients. The replication buys communication simplicity — one all-reduce and you are done — but it wastes memory catastrophically, because `N-1` out of every `N` GPUs are storing a copy of state that is byte-for-byte identical to a copy on some other GPU. If you could keep *one* copy of the state, split across the GPUs so each holds `1/N` of it, you would cut per-GPU memory by a factor of `N` and the wasted `896 - 112 = 784` GB would simply vanish.

![a before and after comparison contrasting DDP holding a full seventy billion parameter replica per GPU against ZeRO-3 holding one shard per GPU and gathering a layer on demand](/imgs/blogs/zero-and-fsdp-the-memory-model-2.webp)

The figure draws the two worlds side by side. On the left, DDP: a 70B model is ${70 \times 16 = 1120}$ GB per GPU, replicated across all 64 cards, nothing shared — and it OOMs an 80 GB card by a factor of fourteen. On the right, ZeRO-3: the same 1120 GB of total state, cut into 64 shards of 17.5 GB each, one shard per GPU. When a GPU needs a full layer to compute with, it gathers that layer from the other GPUs just in time, uses it, and throws it away. The memory that used to be replicated 64 times now exists exactly once, spread thin. That is the entire idea. Everything else is mechanism.

The obvious objection: if each GPU only holds `1/N` of the parameters, how does it run a forward pass? A matmul needs the *whole* weight matrix, not a shard of it. The answer — and this is the mechanism we will spend the middle of this post on — is that you reconstruct the full weight matrix on demand with an all-gather, right before you need it, and free it right after. But before the mechanism, let us be precise about *what* gets sharded, because ZeRO comes in three stages and they shard progressively more.

## The three stages of ZeRO

The `(2 + 2 + 12)` grouping from the derivation is the map. ZeRO shards those three groups in order of how much they cost and how little they hurt to shard, and each stage adds the next group on top of the previous one.

![a matrix comparing DDP and the three ZeRO stages across params grads optimizer per-GPU memory and communication volume showing memory falling toward one over N](/imgs/blogs/zero-and-fsdp-the-memory-model-3.webp)

**ZeRO-1 shards the optimizer states** — the fat `12Ψ` group. The bf16 weights and gradients stay fully replicated on every GPU, exactly as in DDP, but the fp32 master, momentum, and variance are partitioned: rank 0 holds the optimizer state for the first `1/N` of the parameters, rank 1 the next `1/N`, and so on. Per-GPU memory becomes:

$$
M_1 = 2\Psi + 2\Psi + \frac{12\Psi}{N} = 4\Psi + \frac{12\Psi}{N}
$$

The genius of ZeRO-1 is that it costs *nothing* in communication relative to DDP. The implementation reduce-scatters the gradients so each rank receives the averaged gradient shard for the parameters whose optimizer state it owns (`Ψ` of traffic), each rank updates its slice of the master weights, and then an all-gather distributes the updated bf16 parameters back to everyone (`Ψ`) — reduce-scatter plus all-gather is ${2\Psi}$, exactly DDP's bill. The only difference from ZeRO-2 is *memory*, not comms: ZeRO-1 still allocates a full gradient buffer on every rank (it shards only the optimizer state), whereas ZeRO-2 additionally frees the non-owned gradient bytes. So you cut the biggest memory term — the `12Ψ` optimizer states — by a factor of `N` and your communication bill does not move an inch. This is why ZeRO-1 is nearly always worth turning on: it is free.

**ZeRO-2 additionally shards the gradients** — the second `2Ψ` group. Now, since each rank only needs the gradient for the `1/N` of parameters whose optimizer state it owns, why keep the full gradient on every rank? ZeRO-2 replaces the gradient all-reduce with a **reduce-scatter**: gradients are summed across ranks, but the sum is scattered so each rank keeps only its `1/N` shard of the averaged gradient. Per-GPU memory:

$$
M_2 = 2\Psi + \frac{2\Psi + 12\Psi}{N} = 2\Psi + \frac{14\Psi}{N}
$$

Again, the communication volume is unchanged from DDP, and it is worth doing the accounting exactly because "free" sounds too good. DDP's gradient all-reduce moves ${2\Psi}$ (reduce-scatter `Ψ` + all-gather `Ψ`). ZeRO-2 replaces it with just the reduce-scatter half — `Ψ` — because each rank only needs the gradient shard for the parameters it owns; there is no need to all-gather the full averaged gradient back, since no rank updates parameters it does not own. But after the optimizer step, each rank has a fresh copy of *its* `1/N` slice of the weights, and every rank needs the *full* updated weights for the next forward — so ZeRO-2 does an all-gather of the updated parameters, `Ψ`. Reduce-scatter (`Ψ`) plus parameter all-gather (`Ψ`) equals ${2\Psi}$ — identical to DDP. The traffic just moved from "after backward" to "before the next forward"; the total did not change. ZeRO-2 is genuinely free comms-wise, and it drops per-GPU memory further because the `2Ψ` of gradients now scales with `1/N` too.

**ZeRO-3 additionally shards the parameters** — the last `2Ψ` group, the bf16 weights themselves. This is the big one, and it is exactly what FSDP's `FULL_SHARD` does. Now *nothing* is replicated: params, grads, and optimizer all live as `1/N` shards. Per-GPU memory:

$$
M_3 = \frac{2\Psi + 2\Psi + 12\Psi}{N} = \frac{16\Psi}{N}
$$

That is near-linear memory scaling: double the GPUs, halve the per-GPU memory, all the way down. It is what makes a 70B model fit on 80 GB cards. But it is the first stage that is *not* free — because if the parameters live sharded, you have to gather them before every forward and every backward, and that gathering is extra communication DDP never paid. The matrix figure shows the whole progression: memory falls from 112 GB (DDP) to ~38 GB (ZeRO-1) to ~26 GB (ZeRO-2) to 14 GB (ZeRO-3) for a 7B model on 8 GPUs, while the comms column stays at 1.0x until ZeRO-3, where it jumps to 1.5x. That jump is the price of the last, biggest memory savings, and understanding exactly why it is 1.5x — not 2x, not 3x — is the mechanism we turn to now.

#### Worked example: the four stages on a 7B model, 8 GPUs

Let ${\Psi = 7 \times 10^9}$ and ${N = 8}$. Reading the formulas as bytes and converting to GB (${\times 10^9}$ bytes ≈ 1 GB, close enough for a memory budget):

- **DDP:** ${16\Psi = 112}$ GB per GPU. Does not fit on 80 GB. Dead on arrival.
- **ZeRO-1:** ${4\Psi + 12\Psi/8 = 28 + 10.5 = 38.5}$ GB. Fits comfortably, with tens of GB to spare for activations. And it cost nothing in comms.
- **ZeRO-2:** ${2\Psi + 14\Psi/8 = 14 + 12.25 = 26.25}$ GB. Even more headroom, still free comms.
- **ZeRO-3:** ${16\Psi/8 = 14}$ GB. Enormous headroom — but you are now paying 1.5x the communication of DDP.

The punchline the table makes obvious: for a 7B model on 8 GPUs, you do **not** need ZeRO-3. ZeRO-1 already fits it at 38.5 GB with room to spare, at zero comms cost. Reaching for ZeRO-3 here would shard the parameters — and pay the 1.5x comms tax — to buy memory headroom you did not need. This is the single most common mistake people make with FSDP: turning on full sharding for a model that would have fit under ZeRO-1, and then wondering why their scaling efficiency is worse than plain DDP. Shard only as far as the model forces you to. We will make that a hard rule at the end.

## The mechanism: gather, compute, free

Everything above is bookkeeping — which bytes live where. The interesting question is the one the objection raised: if rank 3 holds only `1/N` of the weights, how does it multiply an input by a full weight matrix? The answer is a three-beat cycle that runs for every sharded layer, and once you see it, ZeRO-3 and FSDP stop being magic.

The key identity is one you may already know from the [collectives post](/blog/machine-learning/distributed-training/collectives-from-scratch): **an all-reduce is exactly a reduce-scatter followed by an all-gather.** DDP does one all-reduce per step to average gradients. ZeRO-3 takes that same all-reduce, splits it into its two halves, and places them at different points in the step — the all-gather to *reconstruct parameters* before compute, and the reduce-scatter to *average and shard gradients* after.

![a branch and merge graph showing a DDP all-reduce decomposed into a reduce-scatter and an all-gather with the all-gather placed in the forward and backward and the reduce-scatter in the backward summing to three units of traffic](/imgs/blogs/zero-and-fsdp-the-memory-model-4.webp)

Here is the cycle for one layer, in order. Suppose the model has been wrapped so that each transformer block is its own FSDP unit — a "flat parameter" that is sharded across ranks. The figure above traces the decomposition; the timeline below traces it in time.

**Forward pass, one layer at a time.** Before the layer runs, FSDP issues an **all-gather**: every rank contributes its shard of this layer's parameters, and every rank receives the full, reconstructed parameter tensor for the layer. Now every rank momentarily holds the complete weights for *this one layer* — but only this one layer, not the whole model. The layer's forward compute runs normally. The instant it finishes, FSDP **frees** the full parameters, dropping each rank back to holding only its `1/N` shard. Then it does the same for the next layer. At any moment, the full-size resident parameters are just one layer's worth, plus the permanent shards of everything else. That is why the peak stays near `16Ψ/N` rather than `16Ψ`.

![a left to right timeline of all-gather then forward compute then free then all-gather again then backward compute then reduce-scatter then free showing peak memory held to one layer](/imgs/blogs/zero-and-fsdp-the-memory-model-5.webp)

**Backward pass, one layer at a time.** The backward pass needs the full weights again — you cannot compute a gradient through a matmul without the matrix — but they were freed after forward. So FSDP issues a **second all-gather** for each layer's parameters as the backward reaches it. Compute the gradient. Now the local gradient for the full layer exists on every rank, but each rank only needs to *keep* the `1/N` shard corresponding to the optimizer state it owns. So FSDP issues a **reduce-scatter**: the per-rank gradients are summed across ranks (that is the averaging DDP also does) and the sum is scattered so each rank walks away with only its shard. Then it frees the full parameters again. The optimizer step then updates each rank's shard of the master weights locally, with no further communication.

The timeline figure lays out the seven beats: all-gather, forward, free, all-gather again, backward, reduce-scatter, free. Read it and you can see exactly why peak parameter memory is one layer, not the model, and exactly where every byte of communication happens.

### Why the extra comms is exactly 1.5x

Now we can price it, and this is the mechanism block — a derivation, not an assertion. Measure communication in units of `Ψ` (one unit = moving one copy of the parameters' worth of bytes across the interconnect; the exact per-GPU byte count carries a `(N-1)/N` factor, but it is identical for every collective here so it cancels out of the *ratio*).

**DDP** does one gradient all-reduce per step. An all-reduce is a reduce-scatter (`1` unit) plus an all-gather (`1` unit), so DDP moves ${2\Psi}$ per step. Call that **2 units — the 1.0x baseline.**

**ZeRO-3** moves:

- **Forward all-gather of parameters:** `1` unit. The params are sharded, so to run the forward you must gather them.
- **Backward all-gather of parameters:** `1` unit. They were freed after forward, so you must gather them again for the backward.
- **Backward reduce-scatter of gradients:** `1` unit. Average the gradients and scatter the shards.

Total: ${3\Psi}$ per step. **3 units — which is exactly 1.5x the DDP baseline of 2 units.**

$$
\frac{\text{ZeRO-3 comms}}{\text{DDP comms}} = \frac{3\Psi}{2\Psi} = 1.5
$$

That is the whole trade, and it is beautifully clean: you pay one extra all-gather of the parameters (the one in the backward, because you had to gather them twice) and in exchange the parameters, gradients, *and* optimizer states all shrink to `1/N`. The figure's caution-colored node — "params gathered 2x, 3 units = 1.5x" — is this derivation in one box. ZeRO-1 and ZeRO-2, by contrast, never gather parameters (they keep params replicated), so they stay at the 2-unit DDP baseline: their memory savings really are free, and only ZeRO-3's parameter sharding costs the extra half.

One important refinement: that 1.5x assumes you re-gather parameters in the backward. If you are using [activation checkpointing](/blog/machine-learning/distributed-training/the-memory-budget) — recomputing the forward during the backward to save activation memory — you gather parameters a *third* time (once for the real forward, once for the recomputed forward inside checkpointing, once implicitly for the backward), and the comms can climb toward 2x. Memory and comms trade against each other at every turn; there is no free lunch, only a menu of prices. We will price activation checkpointing properly in its own post.

### Overlap turns 1.5x comms into almost no slowdown

Here is the distinction that separates people who understand FSDP from people who merely turned it on: **1.5x communication volume is not 1.5x wall-clock.** Volume is bytes moved; wall-clock is time you actually wait. If the all-gather for the next layer happens *while the current layer is still computing*, the bytes move in the shadow of compute you were going to do anyway, and the extra communication costs you nothing on the clock. This overlap is the entire reason ZeRO-3 is practical rather than a 50%-slower curiosity.

The trick is **prefetch**. In the forward pass, as soon as layer `L` starts computing, FSDP kicks off the all-gather for layer `L+1` on a separate CUDA stream. By the time layer `L` finishes and the forward reaches `L+1`, its weights are already resident — the gather ran under the compute. FSDP's `BackwardPrefetch.BACKWARD_PRE` does the same for the backward: it starts gathering layer `L-1`'s parameters while layer `L`'s backward is still running. DeepSpeed exposes the same behavior through `overlap_comm` and the `stage3_prefetch_bucket_size`. When overlap is working, a `torch.profiler` trace shows the NCCL all-gather kernels sitting *underneath* the matmul kernels on the timeline, not in the gaps between them, and your step time is bounded by `max(compute, comms)` rather than `compute + comms`.

Overlap has a memory cost, and this is the tension you tune. To gather `L+1` while computing `L`, both layers' full parameters are briefly resident — so aggressive prefetch raises the temporary spike. That is exactly what DeepSpeed's `stage3_max_live_parameters` caps: it bounds how many gathered parameters may co-exist, trading a little overlap for a lower peak. The right setting depends on your headroom: if you sharded down to 14 GB on an 80 GB card, spend the headroom on deep prefetch and hide all the comms; if you sharded down to 70 GB and are tight, throttle prefetch to stay alive. When overlap fully hides the comms, ZeRO-3 and ZeRO-1 run at nearly the same tokens/s despite the 1.5x volume — which is the whole reason the 70B model is trainable at a reasonable MFU and not just *fit-able*.

The failure mode to recognize: on a **slow interconnect** — GPUs talking over PCIe instead of NVLink, or a thin inter-node network — the all-gather is so slow that it *cannot* finish in the shadow of one layer's compute, so `comms > compute`, overlap saturates, and the 1.5x volume becomes a real 1.3–1.5x wall-clock hit. This is the single most common reason "FSDP is slow" reports turn out to be true: not the algorithm, but the wire. Before you blame ZeRO-3, profile the interconnect — a topic [the interconnect physics](/blog/machine-learning/distributed-training/the-interconnect-physics) post covers, and one we will diagnose directly when a [multi-node run comes out slower than a single node](/blog/machine-learning/distributed-training/multinode-slower-than-single-node).

## How the shard actually lives on each rank

It helps to see the steady state — what a rank is holding when it is *not* mid-gather. Under ZeRO-3, each rank permanently owns exactly one `1/N` slice of every state tensor, and never the whole thing except momentarily during a gather.

![a three by three grid where each row is a rank and each column is params grads or optimizer showing rank i owning slice i of every state type](/imgs/blogs/zero-and-fsdp-the-memory-model-6.webp)

The grid figure shows three ranks (rows) and the three state types (columns). Rank 0 owns `params[0:1/3]`, `grads[0:1/3]`, and `opt[0:1/3]`. Rank 1 owns the middle third of each. Rank 2 owns the last third. Read down any column and you see one state tensor cut into three contiguous shards; read across any row and you see everything a single rank is responsible for. This is the resting state ZeRO-3 returns to after every gather-compute-free cycle. The all-gather temporarily fills in a full row's worth of one layer's params; the free operation drops back to exactly this picture.

A subtlety worth flagging: FSDP shards a *flattened* view of each wrapped unit's parameters, not each individual weight matrix. It concatenates all the parameters in a unit into one long 1-D "FlatParameter," pads it to be divisible by `N`, and hands each rank a contiguous `1/N` chunk of that flat buffer. This is why the wrapping policy matters so much — it decides how big each gather is and therefore how big your temporary memory spike and your communication messages are. Wrap the whole model as one unit and you gather the entire model at once (huge spike, defeats the purpose). Wrap each transformer block as its own unit and you gather one block at a time (small spike, the sweet spot). We tune that in the [FSDP-in-practice](/blog/machine-learning/distributed-training/fsdp-in-practice) post; here, know that "one FSDP unit per transformer block" is the sane default and the reason the peak is "one layer," not "the model."

## The code: FSDP and DeepSpeed, made concrete

Two mainstream implementations. PyTorch-native **FSDP** (`torch.distributed.fsdp`) and **DeepSpeed ZeRO**. They implement the same math; they differ in ergonomics and in which knobs they expose. Here is enough of each to make the stages concrete — we defer the real tuning to [fsdp-in-practice](/blog/machine-learning/distributed-training/fsdp-in-practice) and [deepspeed-zero-and-offload](/blog/machine-learning/distributed-training/deepspeed-zero-and-offload).

### FSDP at ZeRO-3 equivalence

FSDP's `ShardingStrategy` is a direct dial onto the ZeRO stages: `NO_SHARD` is DDP, `SHARD_GRAD_OP` is ZeRO-2 (shard grads and optimizer, replicate params), and `FULL_SHARD` is ZeRO-3 (shard everything). Here is a minimal ZeRO-3 wrap with the settings that matter:

```python
import functools
import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

model = build_transformer()  # your nn.Module on CPU or meta device

# Wrap EACH transformer block as its own FSDP unit, so the all-gather
# gathers one block at a time and frees it — the peak stays one block.
auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={TransformerBlock},
)

# bf16 compute + bf16 gradient reduce-scatter; fp32 master stays fp32.
mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,   # == ZeRO-3
    auto_wrap_policy=auto_wrap_policy,
    mixed_precision=mp_policy,
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # overlap next gather
    device_id=torch.cuda.current_device(),
    use_orig_params=True,  # keeps param names/optimizer groups sane
)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))
```

Three lines carry the whole post. `sharding_strategy=FULL_SHARD` is the choice to shard params, grads, and optimizer — ZeRO-3, `16Ψ/N`. `transformer_auto_wrap_policy` with your block class is what makes the gather granularity "one block," so the temporary spike is one layer, not the model. `backward_prefetch=BACKWARD_PRE` starts the all-gather for layer `L-1` while layer `L`'s backward is still computing, overlapping the extra communication with compute so the 1.5x comms does not become 1.5x wall-clock — overlap is the entire game, and we spend a [whole post on it](/blog/machine-learning/distributed-training/overlapping-compute-and-communication). To drop to ZeRO-2 — replicate params, shard only grads and optimizer — you change exactly one argument:

```python
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,  # == ZeRO-2
    auto_wrap_policy=auto_wrap_policy,
    mixed_precision=mp_policy,
    device_id=torch.cuda.current_device(),
    use_orig_params=True,
)
```

That single-word change — `FULL_SHARD` to `SHARD_GRAD_OP` — is the difference between `16Ψ/N` at 1.5x comms and `2Ψ + 14Ψ/N` at 1.0x comms. It is the most consequential one-line edit in a training script, and now you can predict exactly what it does to both your memory and your comms bill.

You launch it with `torchrun`, which spawns one process per GPU and sets the `RANK`/`LOCAL_RANK`/`WORLD_SIZE` environment variables the code reads:

```bash
torchrun --standalone --nproc_per_node=8 train_fsdp.py \
    --model 7b --seq-len 4096 --micro-batch 4
```

### DeepSpeed ZeRO via config JSON

DeepSpeed drives the same three stages from a JSON config, which is convenient because you can sweep stages without touching Python. The `stage` field is literally 1, 2, or 3:

```json
{
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 8,
  "bf16": { "enabled": true },
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 500000000,
    "stage3_prefetch_bucket_size": 500000000,
    "stage3_param_persistence_threshold": 1000000,
    "stage3_max_live_parameters": 1000000000,
    "stage3_gather_16bit_weights_on_model_save": true
  }
}
```

`"stage": 3` shards params, grads, and optimizer (the FSDP `FULL_SHARD` equivalent); set it to `2` for grads-and-optimizer or `1` for optimizer-only. `overlap_comm` is DeepSpeed's version of prefetch — overlap the all-gather with compute. `stage3_max_live_parameters` caps how many full (gathered) parameters may be resident at once, which is the direct knob on your temporary memory spike; lower it and the peak drops but you gather more often. `stage3_gather_16bit_weights_on_model_save` reassembles the full model on save so your checkpoint is not left in shards. You launch with the `deepspeed` runner:

```bash
deepspeed --num_gpus=8 train.py --deepspeed ds_config.json
```

For the ZeRO-3-plus-offload configuration — pushing the fp32 optimizer states and even the parameters to CPU RAM or NVMe when you have run out of GPUs to shard across — see the dedicated [DeepSpeed ZeRO and 3D-parallelism deep-dive](/blog/machine-learning/open-source-library/deepspeed-zero-3d-parallelism-deep-dive), which walks the offload config and the bandwidth math end to end.

### FSDP or DeepSpeed?

They implement the same three-stage math, so the choice is about ergonomics and ecosystem, not capability. The short version:

| Dimension | FSDP (PyTorch-native) | DeepSpeed ZeRO |
|---|---|---|
| Where it lives | In-tree `torch.distributed.fsdp`; no extra dependency | Separate library; its own engine and launcher |
| Stage control | `ShardingStrategy` enum in code | `stage` field in a JSON config, swappable without code |
| Offload to CPU/NVMe | Limited (CPU offload of params/grads) | First-class (ZeRO-Infinity: optimizer, params, NVMe) |
| Composability | Native with `torch.compile`, tensor parallelism via DeviceMesh | 3D parallelism via Megatron-DeepSpeed integration |
| Sweet spot | You live in pure PyTorch and want minimal moving parts | You want offload, config-file sweeps, or DeepSpeed's ecosystem |

The honest guidance: **if you are already in a plain PyTorch training loop and your model fits under `FULL_SHARD` across the GPUs you have, use FSDP** — it is in-tree, composes cleanly with `torch.compile`, and there is one fewer thing to install and debug. **Reach for DeepSpeed when you need offload** (the model does not fit even at ZeRO-3 across every GPU), when you want to sweep stages from a config without editing code, or when you are integrating with a stack that already speaks DeepSpeed. Both are correct; neither is magic; the memory and comms arithmetic in this post is identical for both, because it is a property of the algorithm, not the library. FSDP2 — the newer `fully_shard` API that shards per-parameter rather than via a flattened buffer — is the direction PyTorch is heading and is worth adopting for new code; we cover its differences in [fsdp-in-practice](/blog/machine-learning/distributed-training/fsdp-in-practice).

## Measuring it honestly

Formulas predict the floor; the profiler tells you the truth, because activations, fragmentation, and framework overhead all live on top of the `16Ψ/N` state. Measure peak memory the honest way — reset the peak counter, run a few steps to reach steady state, synchronize, then read the max:

```python
import torch

torch.cuda.reset_peak_memory_stats()
for step, batch in enumerate(loader):
    loss = model(batch).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if step == 10:   # let it reach steady state first
        break

torch.cuda.synchronize()
peak_gb = torch.cuda.max_memory_allocated() / 1e9
reserved_gb = torch.cuda.max_memory_reserved() / 1e9
print(f"rank {dist.get_rank()}: peak alloc {peak_gb:.1f} GB, reserved {reserved_gb:.1f} GB")
```

Two gotchas that bite everyone. First, `max_memory_allocated` is live tensor bytes; `max_memory_reserved` is what the caching allocator has grabbed from the driver, which is larger and is the number `nvidia-smi` shows — always compare like with like. Second, memory grows for the first several steps as the allocator warms up and the optimizer states get lazily materialized on their first `.step()`, so a reading at step 0 will under-report; take it at steady state. And run it on *every* rank, not just rank 0 — under FSDP the shards are balanced, but a bad wrap or an unsharded buffer can leave one rank fatter than the others, and the job OOMs on the fattest rank while rank 0 looks fine.

Here is a representative before-and-after for a 7B model on a single 8x A100 80GB SXM node, the kind of table you should be able to reproduce and predict:

| Configuration | Params | Grads | Optim | State/GPU | Fits 80GB? | Comms |
|---|---|---|---|---|---|---|
| DDP | full | full | full | 112 GB | No (OOM) | 1.0x |
| ZeRO-1 | full | full | 1/8 | 38.5 GB | Yes | 1.0x |
| ZeRO-2 (`SHARD_GRAD_OP`) | full | 1/8 | 1/8 | 26.3 GB | Yes | 1.0x |
| ZeRO-3 (`FULL_SHARD`) | 1/8 | 1/8 | 1/8 | 14.0 GB | Yes | 1.5x |

Read the "State/GPU" column as the floor from the formulas; the real peak-allocated number will sit tens of GB above it once activations are counted (which is exactly why ZeRO-1's 38.5 GB floor still needs activation checkpointing at a long sequence length even though the state fits). The "Comms" column is the prediction from the 1.5x derivation, and a `torch.profiler` trace filtered to NCCL kernels will confirm ZeRO-3 spends about 50% more time in collectives per step than DDP does — which turns into wall-clock only to the extent it fails to overlap with compute.

#### Worked example: fitting a 70B model on a 64-GPU cluster

This is the headline the title promises. Take ${\Psi = 70 \times 10^9}$ and a cluster of ${N = 64}$ A100 80GB SXM cards (eight 8-GPU nodes on InfiniBand).

- **DDP:** ${16\Psi = 1120}$ GB per GPU. There is no such card. Impossible, full stop — this is the wall.
- **ZeRO-1:** ${4\Psi + 12\Psi/64 = 280 + 13.1 = 293}$ GB per GPU. Still impossible — sharding only the optimizer leaves the `4Ψ` of replicated params and grads at 280 GB.
- **ZeRO-2:** ${2\Psi + 14\Psi/64 = 140 + 15.3 = 155}$ GB per GPU. Still does not fit — the `2Ψ` of replicated params alone is 140 GB.
- **ZeRO-3:** ${16\Psi/64 = 17.5}$ GB per GPU of state. **Fits**, with ~60 GB left for activations and the temporary one-layer gather spike.

Notice the shape of that list: for a 70B model, *only* ZeRO-3 fits, because any stage that leaves the parameters replicated is stuck with a `2Ψ = 140` GB floor that no amount of sharding-the-rest can move. The moment your model's `2Ψ` — its bf16 weights alone — exceeds a single GPU's memory, you have no choice but to shard the parameters, which means ZeRO-3 or FSDP `FULL_SHARD`, which means paying the 1.5x comms. For a 70B model, `2Ψ` is 140 GB, wildly over 80 — so ZeRO-3 is mandatory, not optional. That is the crisp rule: **if `2Ψ` exceeds one GPU's memory, you must shard parameters; otherwise you should not.** The 7B model (`2Ψ = 14` GB, fits on one card) does not need it; the 70B model (`2Ψ = 140` GB) has no alternative. And notice the payoff of near-linear scaling: 17.5 GB per GPU means you have massive headroom, and you could even train this 70B model on *fewer* than 64 GPUs — at 32 GPUs it would be 35 GB per GPU, still a comfortable fit.

## A war story: the 13B that OOM'd at step 137

Here is the kind of problem this arithmetic lets you solve in your head instead of by trial and error. A colleague brings you a 13B model on an 8x A100 80GB node. It is wrapped in FSDP `FULL_SHARD`. It starts training fine — step 0, step 1, step 10 all green — and then at step 137 it dies: `CUDA out of memory`. Intermittent OOMs are the worst kind, because the obvious question ("does it fit?") apparently answers yes for 136 steps and then no. Let us reason it out.

First, the state floor. 13B under ZeRO-3 on 8 GPUs is ${16 \times 13 / 8 = 26}$ GB of state per GPU. That is *constant* — it does not grow with the step number, because params, grads, and optimizer are all fixed-size. So whatever is OOMing at step 137 is **not** the state. That single deduction eliminates the entire subject of this post as the culprit and points at the two things that *do* vary step to step: activations and the temporary gather spike.

Second, activations scale with the *tokens in the batch*, and if the data loader is packing variable-length sequences, some batches are much longer than others. Step 137 was probably the first batch to hit the maximum sequence length, and at max length the activations spiked past the headroom. The tell: `26` GB state on an `80` GB card leaves `54` GB for activations and the gather spike, and a 13B model at a long sequence length with no activation checkpointing can absolutely blow through 54 GB. The fix is not more sharding — the state already fits with room — it is [activation checkpointing](/blog/machine-learning/distributed-training/the-memory-budget) to trade some recompute for a much smaller activation footprint, plus capping or bucketing sequence length so batch size in *tokens* is bounded, not just in sequences.

Third, there is a subtler contributor the arithmetic warns you about: the optimizer states materialize **lazily**. Adam allocates its momentum and variance buffers on the *first* `optimizer.step()`, not at construction — so your peak actually steps up after the first optimizer update, not at step 0. If your headroom was thin, the run can survive the forward/backward of step 0 and die shortly after when the fp32 moments get allocated. A memory reading taken at step 0 would have under-reported the true peak and hidden the problem. This is exactly why the honest measurement recipe reads the peak at *steady state*, several steps in — a step-0 reading here would have said "62 GB, plenty of room" and lied to you.

So the reasoning chain, start to finish: intermittent OOM → state is constant so it is not the state → the variable costs are activations and the gather spike → activations spike on long batches and the peak steps up after lazy optimizer allocation → fix with activation checkpointing and sequence-length bounding, not with a higher ZeRO stage. **Stress-test the conclusion:** what if it had OOM'd at step 0 every single time, deterministically? Then it *would* be the state or the gather spike — and the fix would be a smaller FSDP unit (wrap per block so the gather is one layer, not the whole model) or, if the state genuinely does not fit, more GPUs to shard across. Same arithmetic, different branch. The value of knowing where every byte goes is that an OOM stops being a mystery and becomes a short deduction.

## Case studies and real numbers

The stages are not a whiteboard idealization; they are what the papers and the production systems actually report.

**The original ZeRO paper (Rajbhandari et al., 2020).** DeepSpeed's ZeRO paper is where the ${(2 + 2 + 12)\Psi}$ decomposition and the three stages come from, and it reports fitting and training models up to 100B+ parameters by combining ZeRO-3 with data parallelism, at a time when the largest densely-trained models were an order of magnitude smaller. Its central table is exactly the memory-per-stage progression we derived; its headline is that ZeRO-3 makes per-device memory scale as `1/N`, so you can trade GPUs for model size almost linearly. The paper is also careful about the communication analysis — it is the source of the "ZeRO-1 and ZeRO-2 match DDP comms, ZeRO-3 is 1.5x" result, and it is worth reading for the exact accounting.

**PyTorch FSDP (Zhao et al., 2023).** The FSDP paper documents the PyTorch-native implementation and its FlatParameter/wrapping design, and reports training GPT-style models up to 175B parameters on clusters of A100s, with near-linear memory scaling matching the theory and TFLOP/s-per-GPU numbers competitive with DeepSpeed. The practical contribution is the `auto_wrap_policy` and the prefetch/overlap machinery that keeps the extra all-gather from showing up as wall-clock — the difference between the 1.5x comms *volume* and a 1.5x *slowdown*.

**Fitting a 13B model on consumer-ish hardware.** A widely reproduced result: a 13B model, which needs ${16 \times 13 = 208}$ GB of Adam state and therefore cannot come close to fitting on a single 80 GB card even before activations, trains comfortably under FSDP `FULL_SHARD` on a single 8x A100 node — ${208/8 = 26}$ GB of state per GPU, well within budget. The same model under DDP is a non-starter. This is the everyday version of the 70B story and the one most readers will actually run.

**The offload extreme.** When you have run out of GPUs to shard across, ZeRO-3 combined with CPU/NVMe offload (ZeRO-Infinity) has been reported training models with hundreds of billions to trillions of parameters on very modest GPU counts by parking the `12Ψ` optimizer states and even the parameters in CPU RAM and NVMe, gated by bus bandwidth rather than GPU memory. This trades a large slowdown (you are now bottlenecked on PCIe and SSD bandwidth, not NVLink) for the ability to fit at all — a price worth paying only when the alternative is not training the model. The [DeepSpeed deep-dive](/blog/machine-learning/open-source-library/deepspeed-zero-3d-parallelism-deep-dive) covers offload's bandwidth math.

The through-line: every one of these results is the same `16Ψ/N` formula with a different `Ψ` and `N`, and the only real variable is whether you paid the extra comms of parameter sharding, and whether you overlapped it away.

## When to reach for each stage (and when not to)

Here is the decisive part. Every stage past ZeRO-1 is a cost, and the discipline is to climb only as far as memory forces you.

![a decision tree that asks whether the model fits as a full replica then how much to shard leading to DDP ZeRO-1 ZeRO-2 ZeRO-3 or offload](/imgs/blogs/zero-and-fsdp-the-memory-model-7.webp)

The tree encodes the rule. Start at the top and stop at the first stage that fits.

**Does the full state (`16Ψ`) fit on one GPU with room for activations?** If yes, use plain DDP or ZeRO-1 and go no further. A 1.5B model (24 GB) or a 3B model (48 GB) on 80 GB cards fits with activations to spare; sharding parameters here would add comms and complexity to solve a problem you do not have. Turn on ZeRO-1 regardless because it is free — it shards the `12Ψ` optimizer states at zero comms cost and gives you headroom for a bigger batch — but do not go past it.

**If `16Ψ` does not fit but `2Ψ` (the params) does, and `2Ψ + 14Ψ/N` fits?** Use ZeRO-2 (`SHARD_GRAD_OP`). This is the sweet spot for mid-size models — say 7B to 13B on a full node — where sharding grads and optimizer gets you under budget while keeping parameters replicated, so you stay at the 1.0x DDP comms baseline and enjoy the *lowest* communication of any stage that fits. ZeRO-2 is underused precisely because people jump straight to `FULL_SHARD`; if your model fits under it, it is the fastest option.

**If even `2Ψ` (the bf16 params alone) exceeds one GPU's memory?** You have no choice: shard the parameters. This is ZeRO-3 / FSDP `FULL_SHARD`, and it is where 70B and up live. You pay the 1.5x comms; overlap it with compute via prefetch and it costs you little wall-clock on a fast interconnect ([NVLink or InfiniBand](/blog/machine-learning/distributed-training/collectives-from-scratch)); on a slow interconnect (PCIe-only, or a thin network between nodes) the extra all-gather can dominate and you will feel it. Wrap per transformer block so the gather granularity is one layer.

**If ZeRO-3 across every GPU you have still does not fit?** Now, and only now, reach for offload — push the optimizer states to CPU, then the parameters to CPU, then to NVMe — accepting the bandwidth-bound slowdown as the price of fitting at all. This is a last resort, not a default.

The anti-patterns are as important as the patterns. **Do not use `FULL_SHARD` for a model that fits under ZeRO-1 or ZeRO-2** — you will pay 1.5x comms and get worse throughput than DDP for zero memory benefit you needed; this is the most common FSDP mistake. **Do not wrap the whole model as one FSDP unit** — the all-gather then reconstructs the entire model at once, spiking memory back toward `16Ψ` and defeating the sharding; wrap per block. **Do not turn on ZeRO-3 on a slow interconnect without measuring** — the 1.5x comms is fine on NVLink and can be crippling on a machine where GPUs talk over PCIe; profile before you commit. And **do not forget activations** — the `16Ψ/N` formula is *state only*; a model whose state fits at ZeRO-3 can still OOM on activations at a long sequence length, which is a job for [activation checkpointing](/blog/machine-learning/distributed-training/the-memory-budget), a different lever entirely.

#### Worked example: the wrong stage costs you throughput

Concrete stress test. You have a 7B model on 8 A100 80GB cards and you reflexively set `FULL_SHARD`. State per GPU is 14 GB — lovely, tons of headroom. But your all-reduce-equivalent comms is now 3 units per step instead of 2, a 50% increase, and on a node where the eight GPUs share NVLink at ~600 GB/s aggregate that might be tolerable if it overlaps — but if you had instead used ZeRO-1, state per GPU would be 38.5 GB (still fits fine on 80) at 2 units of comms, and your step time would be shorter because you never gather parameters at all. The measured difference on a real 8x A100 node for a 7B model is typically single-digit-percent to low-double-digit-percent throughput — small, but you paid it for *nothing*, because both configs fit. Multiply a 10% throughput loss across a two-week training run and you have burned a couple of days of an 8-GPU node — call it ${8 \times 24 \times 14 \times 0.10 \approx 270}$ GPU-hours, or a few hundred dollars at \$1–2 per GPU-hour — to solve a memory problem you did not have. The stage is not free; pick the cheapest one that fits.

## Key takeaways

- **Mixed-precision Adam costs 16 bytes per parameter, not 2:** ${(2 + 2 + 12)\Psi}$ — 2 for bf16 weights, 2 for bf16 grads, 12 for the fp32 master, momentum, and variance. A 7B model is 112 GB of state before activations; a 70B model is 1120 GB. That arithmetic is your fit-or-not floor.
- **DDP replicates all 16 bytes on every GPU;** `N` GPUs store `N` identical copies of state that only needs to exist once. ZeRO and FSDP shard that state so each GPU holds `1/N`.
- **The three stages shard progressively:** ZeRO-1 shards the optimizer (`4Ψ + 12Ψ/N`), ZeRO-2 also shards gradients (`2Ψ + 14Ψ/N`), ZeRO-3 / FSDP `FULL_SHARD` also shards parameters (`16Ψ/N`, near-linear).
- **ZeRO-1 and ZeRO-2 are free in communication** (same as DDP); only **ZeRO-3 costs extra — exactly 1.5x** — because it must all-gather the parameters in both the forward and the backward.
- **The mechanism is gather → compute → free:** params live sharded, an all-gather reconstructs one layer just-in-time, compute runs, the full weights are freed, and the backward adds a reduce-scatter of gradients. An all-reduce is a reduce-scatter plus an all-gather, split across the layer boundary.
- **`FULL_SHARD` in FSDP equals ZeRO-3; `SHARD_GRAD_OP` equals ZeRO-2; `NO_SHARD` equals DDP** — a one-word change with fully predictable memory and comms consequences.
- **The decision rule:** if `16Ψ` fits, use DDP/ZeRO-1; if only `2Ψ` fits, use ZeRO-2; if even `2Ψ` overflows one GPU, you must shard parameters (ZeRO-3/FSDP) and pay 1.5x comms; if that still overflows, offload. Stop at the cheapest stage that fits.
- **Overlap is what turns 1.5x comms into ~0% slowdown:** prefetch the next layer's all-gather during the current layer's compute, and measure on your actual interconnect before assuming ZeRO-3 is free.

## Further reading

- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) — Rajbhandari et al., 2020. The source of the `(2+2+12)Ψ` decomposition, the three stages, and the 1.5x comms analysis.
- [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://arxiv.org/abs/2304.11277) — Zhao et al., 2023. The FlatParameter, wrapping, and prefetch design behind `torch.distributed.fsdp`.
- [ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://arxiv.org/abs/2104.07857) — the CPU/NVMe offload extension for when GPU count runs out.
- [PyTorch FSDP documentation](https://pytorch.org/docs/stable/fsdp.html) and the [FSDP tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) — the API, the sharding strategies, and FSDP2's `fully_shard`.
- [Why Distributed Training](/blog/machine-learning/distributed-training/why-distributed-training) — the four walls and the series map; this post is the "model won't fit" wall.
- [Collectives From Scratch](/blog/machine-learning/distributed-training/collectives-from-scratch) — the all-reduce = reduce-scatter + all-gather identity that ZeRO-3 splits across the layer.
- [The Memory Budget](/blog/machine-learning/distributed-training/the-memory-budget) — where activations, not state, become the wall, and how activation checkpointing fights them.
- [FSDP in Practice](/blog/machine-learning/distributed-training/fsdp-in-practice) — wrapping policy, sharding strategy, mixed precision, and FSDP2, tuned for real runs.
- [DeepSpeed ZeRO and 3D Parallelism Deep-Dive](/blog/machine-learning/open-source-library/deepspeed-zero-3d-parallelism-deep-dive) — the config, the offload bandwidth math, and composing ZeRO with tensor and pipeline parallelism.
- [The Distributed Training Playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook) — the capstone decision and debugging checklist that ties the whole series together.
