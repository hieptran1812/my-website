---
title: "FSDP in Practice: Wrapping, Sharding Strategy, and Not Shooting Yourself in the Foot"
date: "2026-07-02"
publishDate: "2026-07-02"
description: "Configure PyTorch FSDP correctly from the wrap policy up: sharding strategy, mixed precision, activation checkpointing, prefetch overlap, FSDP2, and the checkpoint gotchas that silently OOM your run."
tags:
  [
    "distributed-training",
    "multi-gpu",
    "fsdp",
    "pytorch",
    "multi-node",
    "mixed-precision",
    "activation-checkpointing",
    "deep-learning",
    "ml-systems",
    "gpu",
  ]
category: "machine-learning"
subcategory: "Distributed Training"
author: "Hiep Tran"
featured: true
readTime: 41
---

You have a 13-billion-parameter transformer and eight A100 80GB cards. On paper it fits: the model, its gradients, and the Adam optimizer state total about 208 GB, and eight cards give you 640 GB of HBM. So you wrap the model in `FullyShardedDataParallel`, launch with `torchrun`, and watch the first step. It OOMs. Not by a little — the allocator dies trying to reserve 26 GB on top of a card that is already nearly full. You add gradient checkpointing. Still OOM. You cut the batch size to one sequence. Still OOM. Eight cards, 640 GB, a model that needs 208 GB, and you cannot fit a single forward pass.

The problem is not the hardware and it is not the batch size. It is one line you never wrote: the **wrapping policy**. You handed FSDP the whole model as a single shardable unit, so at the moment of the forward pass FSDP dutifully reconstructs *every* parameter on *every* GPU at once — all 26 GB of bf16 weights, materialized simultaneously, defeating the entire point of sharding. Sharding only helps if you shard at the right granularity, and the granularity is exactly the granularity of the units you wrap. Wrap the whole model as one unit and you have written a very slow, very memory-hungry way of doing nothing.

This post is the hands-on companion to the memory model. It assumes you already understand *why* FSDP saves memory — the `(2 + 2 + 12)Ψ` optimizer-state math, the idea of sharding parameters, gradients, and optimizer states across ranks — from [ZeRO and FSDP: the memory model](/blog/machine-learning/distributed-training/zero-and-fsdp-the-memory-model). Here we do the *practice*: every knob you actually set, what it does to memory and comms, and the specific ways each one can silently ruin your run. By the end you will be able to write an FSDP config for a real transformer — wrap policy, sharding strategy, mixed precision, activation checkpointing, prefetch — reason about its peak memory and its throughput before you launch, and know exactly which checkpoint API will OOM your CPU and which one will not.

We will keep returning to the same running example: a 13B transformer on 8 A100 80GB cards (one node), then the same model on 2 nodes of 8. That example is the spine. Everything else — the strategy matrix, the overlap timeline, FSDP2 — hangs off it. This is the fourth of the [four walls](/blog/machine-learning/distributed-training/why-distributed-training): the model that won't fit. FSDP is the sharpest tool we have for it, and like every sharp tool it cuts both ways.

## The wrapping policy: the one decision that makes or breaks FSDP

Start with the most important sentence in this entire post: **FSDP shards, gathers, and frees parameters at the granularity of the wrapped unit.** Every FSDP instance owns a set of parameters — its *flat shard* — and three things happen to that set on every step. Before the unit runs its forward, FSDP issues an `all_gather` so that every rank temporarily holds the *full*, unsharded parameters of that unit. The unit computes. Then FSDP frees the gathered parameters back down to the local shard. The same gather-compute-free dance happens again in the backward pass, followed by a `reduce_scatter` that averages gradients and leaves each rank with only its shard.

The size of that temporary "full parameters" balloon is set entirely by how big the wrapped unit is. If the unit is one transformer block of a 13B model — roughly 325 million parameters, about 0.65 GB in bf16 — then the balloon is 0.65 GB, and only one block is inflated at a time as the model rolls forward layer by layer. If the unit is the *whole model*, the balloon is all 13 billion parameters, 26 GB, materialized on every GPU at once. Same model, same eight GPUs, same sharding math — a 40x difference in transient memory, driven by one policy object.

![A before and after comparison of wrapping the whole model as one unit versus wrapping each transformer block, showing one giant all gather against a rolling per block all gather](/imgs/blogs/fsdp-in-practice-1.webp)

The figure above is the whole ballgame. On the left, one giant unit: a single `all_gather` reconstructs the entire model, peak transient memory is the full parameter set, and you get no memory win — you have paid all the communication cost of sharding and kept none of the memory benefit. On the right, per-block units: the gather is a *rolling* operation, one block inflated and freed before the next, so peak transient memory is a single block. The right picture is the one you want, every time.

### How to actually set the policy

FSDP takes an `auto_wrap_policy` argument. The single most useful one for transformers is `transformer_auto_wrap_policy`, which walks the module tree and wraps every instance of the block class you name. You tell it "wrap each `LlamaDecoderLayer`" and it does exactly that — each decoder layer becomes its own FSDP unit, sharded and gathered independently.

```python
import functools
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision, BackwardPrefetch
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# Wrap EACH transformer block as its own FSDP unit. This is the line that
# turns FSDP from "no memory win" into "fits a 13B model on 8x80GB".
auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={LlamaDecoderLayer},
)

model = FSDP(
    model,
    auto_wrap_policy=auto_wrap_policy,
    sharding_strategy=ShardingStrategy.FULL_SHARD,   # ZeRO-3: shard params, grads, optim
    device_id=torch.cuda.current_device(),
    use_orig_params=True,                            # keep param names/groups intact
)
```

The alternative, when you do not have a clean block class to name — a bespoke model, a fused stack, something from a research repo — is `size_based_auto_wrap_policy`, which wraps any submodule whose parameter count crosses a threshold:

```python
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

# Wrap any submodule with >= 100M parameters as its own unit.
auto_wrap_policy = functools.partial(
    size_based_auto_wrap_policy,
    min_num_params=int(1e8),
)
```

The size-based policy is a blunter instrument — it can split a block awkwardly or leave a large embedding unwrapped — but it is a safe default when you cannot name the class. The one thing you must not do is pass `auto_wrap_policy=None` and expect sharding to help. With no policy, FSDP creates exactly one unit for the root module, and you are back to the left side of the figure: the whole model gathered at once. This is the single most common FSDP mistake, and it is *silent*. Nothing errors. The model trains. It just uses far more memory than it should and gives you none of the fit you paid for, and you discover it only when a slightly larger model or a slightly longer sequence OOMs on hardware that should have handled it easily.

### The mechanism: peak memory as a function of wrap granularity

Let us make the 40x concrete. Denote the model's parameter count $\Psi$ and, for a single wrapped unit, $\psi$. In bf16, parameters and gradients are 2 bytes each; the fp32 Adam state — master weights, first moment, second moment — is 12 bytes per parameter. On $N$ ranks with `FULL_SHARD`, the *resident* per-rank memory for model state is the sharded portion:

$$M_\text{resident} = \frac{(2 + 2 + 12)\,\Psi}{N} = \frac{16\,\Psi}{N}$$

That part does not depend on the wrap policy — the shards are the shards. What the wrap policy controls is the *transient* balloon on top of it: the fully gathered parameters of the units that are live at any instant. With per-block wrapping and prefetch depth one, that is one block, maybe two:

$$M_\text{transient} \approx 2 \cdot 2\psi \quad \text{(two bf16 blocks: current + prefetched)}$$

With whole-model wrapping, the transient term is the entire model, $2\Psi$, plus the full unsharded gradient $2\Psi$ in the backward pass. For our 13B model on 8 GPUs those numbers are:

- Resident model state: $16 \times 13\text{B} / 8 = 26$ GB per GPU.
- Per-block transient: about $2 \times 2 \times 0.325\text{B} = 1.3$ GB.
- Whole-model transient: about $2 \times 13\text{B} = 26$ GB of params, plus 26 GB of full gradients during backward.

Add resident 26 GB + whole-model transient 26 GB + backward gradients before reduce-scatter, and you are already over 78 GB before a single activation exists. That is the OOM from the intro. Swap to per-block wrapping and the transient term collapses from 26 GB to 1.3 GB, leaving roughly 53 GB of an 80 GB card free for activations. The model fits. Nothing changed but the granularity of one policy object.

#### Worked example: the 13B that OOMs one way and fits the other

Concrete accounting for LLaMA-13B (40 decoder layers, `d_model` 5120), 8x A100 80GB, `FULL_SHARD`, Adam, bf16 compute, sequence length 2048, micro-batch 1:

| Term | Whole-model wrap | Per-block wrap |
|---|---|---|
| Resident sharded state ($16\Psi/N$) | 26 GB | 26 GB |
| Transient gathered params | 26 GB (all layers) | ~0.7 GB (one layer) |
| Transient full gradients (backward) | 26 GB | ~0.7 GB |
| Activation working set | needs ~15 GB | ~15 GB |
| **Peak per GPU** | **~93 GB → OOM** | **~42 GB → fits** |

The resident term is identical. The entire difference — the difference between a job that runs and a job that dies on step one — is the transient term, and the transient term is the wrap policy. If you take one thing from this post, take this table. Wrap per block.

### What actually gets wrapped, and the use_orig_params knob

The auto-wrap policy wraps each block, but the model has more than blocks. The token embedding, the final norm, and the language-model head are not decoder layers, so `transformer_auto_wrap_policy` leaves them in the *root* FSDP unit — the outermost wrapper around the whole model. That is usually fine: the embedding and head of a 13B model are a few hundred million parameters, gathered once at the start of the forward and once at the end, and they shard along with everything else. But if your embedding is enormous — a large vocabulary times a wide hidden dimension — it can dominate the root unit's transient gather, and you may want to wrap it separately or tie it to the head. The habit worth building is to print the wrapped structure once (`print(model)` after wrapping shows the nested `FullyShardedDataParallel` units) and confirm that no single unit is unexpectedly large. A giant unwrapped embedding is a quieter cousin of the whole-model-wrap mistake.

The other knob every snippet above carries is `use_orig_params=True`, and it deserves a sentence of its own because leaving it at the FSDP1 default of `False` causes real pain. With `use_orig_params=False`, FSDP replaces your parameters with the opaque flat parameters, so `model.named_parameters()` returns FSDP's internal buffers rather than your original weights — which breaks optimizer parameter groups (you cannot give the embedding a different weight decay than the blocks, because you can no longer name them), breaks any code that inspects parameters by name, and makes the state dict harder to reason about. With `use_orig_params=True`, FSDP keeps your original parameter names and shapes visible even though the storage underneath is sharded, so optimizer param groups, weight-decay exclusions for norms and biases, and name-based inspection all work normally. Set it to `True` on every FSDP1 job; FSDP2 makes this behavior the only behavior, which is one more reason it is cleaner.

A subtle consequence: because `use_orig_params=True` exposes original parameters, you can pass fine-grained parameter groups to the optimizer — for instance, no weight decay on `LayerNorm` weights and biases, standard decay on the linear layers — exactly as you would on a single GPU. Under `use_orig_params=False` that pattern is impossible, and people silently apply weight decay to their norms, which subtly hurts convergence. The knob that looks like a compatibility detail is really a correctness detail for your optimizer.

## Sharding strategy: how much to shard, and what it costs in comms

Once units are wrapped correctly, the next knob is the `ShardingStrategy`: *how much* of the model state each unit shards, and therefore how much it must communicate to reconstruct what it sharded. There are four, and they trace a clean line from "shard everything, communicate the most" to "shard nothing, communicate the least" — which is just plain DDP.

![A matrix comparing four FSDP sharding strategies across what they shard, memory per GPU, communication cost, and when to use each](/imgs/blogs/fsdp-in-practice-2.webp)

- **`FULL_SHARD`** (equivalent to ZeRO stage 3) shards parameters, gradients, and optimizer states. After the forward pass it *reshards* — frees the gathered parameters back to shards — so it must `all_gather` them again in the backward pass. Lowest memory, highest comms.
- **`SHARD_GRAD_OP`** (roughly ZeRO stage 2) shards gradients and optimizer states, and it shards parameters at rest, but it sets `reshard_after_forward=False`: after the forward gather it *keeps* the full parameters resident through the backward pass. That saves the second `all_gather` at the cost of holding one unit's full parameters longer. Middle memory, middle comms.
- **`HYBRID_SHARD`** does `FULL_SHARD` *within* a node and replicates *across* nodes — the subject of the next section, and the right answer for most multi-node jobs.
- **`NO_SHARD`** shards nothing; it is ordinary `DistributedDataParallel` wearing an FSDP jacket. One `all_reduce` per step, full memory. Useful only so that a single code path can toggle between DDP and sharding by changing an enum.

### The mechanism: why FSDP moves about 1.5x the bytes of DDP

This is the derivation that justifies the whole matrix. Recall from [DDP from first principles](/blog/machine-learning/distributed-training/ddp-from-first-principles) that a ring `all_reduce` moves, per GPU, a volume

$$V_\text{AR} = 2\,\frac{N-1}{N}\,S$$

bytes, where $S$ is the message size — it is a `reduce_scatter` followed by an `all_gather`, each moving $\frac{N-1}{N}S$. DDP does exactly one `all_reduce` of the full gradient per step, so its per-step comms volume is $V_\text{AR}$ with $S$ equal to the gradient size.

`FULL_SHARD` does three collectives per step, each moving $\frac{N-1}{N}P$ where $P$ is the parameter (or gradient) size: an `all_gather` in the forward, an `all_gather` in the backward, and a `reduce_scatter` of gradients. So its per-step volume is

$$V_\text{FSDP} = 2 \cdot \frac{N-1}{N}P \;+\; \frac{N-1}{N}P \;=\; 3\,\frac{N-1}{N}P.$$

Because bf16 parameters and bf16 gradients are the same size, $P = S$, and the ratio is

$$\frac{V_\text{FSDP}}{V_\text{AR}} = \frac{3\,\frac{N-1}{N}P}{2\,\frac{N-1}{N}S} = \frac{3}{2} = 1.5.$$

`FULL_SHARD` moves 50% more bytes than DDP. That is the price of the memory win, and it is why FSDP is not free even when the model would have fit under DDP: you are buying memory with bandwidth. `SHARD_GRAD_OP` drops the backward `all_gather`, so it moves $2\frac{N-1}{N}P$ — the *same* volume as DDP — while still sharding gradients and optimizer states. That is the sweet spot when you have memory headroom: ZeRO-2-level memory savings at DDP-level communication.

Here is the trade-off as a table, for the 13B model on one 8-GPU node:

| Strategy | Shards | Resident model state / GPU | Per-step comms vs DDP | Reach for it when |
|---|---|---|---|---|
| `FULL_SHARD` | params + grads + optim | ~26 GB | ~1.5x | Model won't fit any other way |
| `SHARD_GRAD_OP` | grads + optim | ~29 GB* | ~1.0x | You have a few GB of headroom and want speed |
| `HYBRID_SHARD` | full-shard in node, replica across | ~26 GB | ~1.5x intra, tiny inter | Multi-node with fast intra-node links |
| `NO_SHARD` (DDP) | nothing | ~208 GB → won't fit | ~1.0x | Model already fits on one GPU |

\* `SHARD_GRAD_OP` holds one unit's full parameters through the backward pass, so its transient peak is a few GB higher than `FULL_SHARD` even though its resident state is similar; the exact number depends on block size.

The decision rule that falls out: if the model fits under DDP, use DDP — do not pay FSDP's 1.5x comms for memory you do not need. If it does not fit but you have headroom once optimizer and gradients are sharded, use `SHARD_GRAD_OP` and enjoy DDP-level comms. Only when you are genuinely tight on memory do you want `FULL_SHARD` and its extra `all_gather`.

## HYBRID_SHARD: shard inside the node, replicate across nodes

Now cross the node boundary, and the physics changes. Inside a DGX-class node, eight GPUs are connected by NVLink and NVSwitch at roughly 600–900 GB/s of aggregate bandwidth per GPU. Between nodes you have InfiniBand — HDR at 200 Gb/s is about 25 GB/s, and after protocol and topology overhead the *effective* per-GPU cross-node bandwidth is often closer to 10–15 GB/s. That is a 40x-plus gap between the fat pipe inside the node and the thin pipe between nodes. `FULL_SHARD` across all 16 GPUs of a 2-node job is blind to that gap: it stripes every `all_gather` and `reduce_scatter` across all 16 ranks, so a large fraction of every parameter gather crosses the thin InfiniBand link, every layer, every step. Your 26 GB/step of gathers now partly traverse a 12 GB/s wire, and throughput falls off a cliff.

`HYBRID_SHARD` is the fix, and it is the single most important multi-node knob in FSDP. It splits the world into two orthogonal process groups. *Within* each 8-GPU node it does `FULL_SHARD` — parameters, gradients, and optimizer states are sharded across the eight local ranks, and every `all_gather` and `reduce_scatter` for parameter reconstruction stays on NVLink. *Across* the two nodes it replicates: each node holds a full copy of the sharded model, and the only cross-node traffic is a single `all_reduce` of the already-reduce-scattered gradient shards, to average across the two replicas. The expensive, frequent parameter gathers ride the fat intra-node pipe; the thin inter-node pipe carries only the small, once-per-step gradient sync.

![A topology graph showing eight GPUs inside each of two nodes doing intra node all gather on NVLink and a thin inter node gradient all reduce on InfiniBand](/imgs/blogs/fsdp-in-practice-3.webp)

The figure shows the two-level structure: dense NVLink meshes inside each node carrying the parameter gathers, and a single thin InfiniBand edge between the nodes carrying the gradient all-reduce. The design matches the hardware — put the heavy traffic on the fast link, the light traffic on the slow one.

Enabling it is a one-line change to the strategy plus a device mesh that tells FSDP the shape of your cluster:

```python
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import ShardingStrategy

# 2 nodes x 8 GPUs. Mesh dim 0 = replicate across nodes, dim 1 = shard within node.
mesh = init_device_mesh("cuda", (2, 8), mesh_dim_names=("replicate", "shard"))

model = FSDP(
    model,
    auto_wrap_policy=auto_wrap_policy,
    sharding_strategy=ShardingStrategy.HYBRID_SHARD,
    device_mesh=mesh,
    device_id=torch.cuda.current_device(),
    use_orig_params=True,
)
```

There is a memory cost to be aware of, and it is the reason `HYBRID_SHARD` is a choice rather than a free lunch. Because each node holds a *full replica* of the sharded model, the model state is sharded only across the 8 GPUs of one node, not across all 16. Under `FULL_SHARD` across 16 ranks, each GPU holds $16\Psi/16$ of the model state; under `HYBRID_SHARD`, each GPU holds $16\Psi/8$ — twice as much resident model state, because the sharding group is half the size. For our 13B model that is 26 GB resident per GPU under hybrid versus 13 GB under full-shard-16. You are spending memory to buy back the interconnect. That trade is almost always worth it when the model fits comfortably within one node's sharding group, and it stops being an option the moment the model is so large it *needs* all 16 ranks to shard across — at which point `FULL_SHARD` across nodes is forced and you must have the inter-node bandwidth to survive it, or move to 3D parallelism. The decision, again, is "the least sharding that fits": shard within the node if one node can hold a replica, across nodes only if it cannot.

There is also `_HYBRID_SHARD_ZERO2`, which does `SHARD_GRAD_OP` within the node instead of `FULL_SHARD` — keep parameters resident through the backward pass locally, replicate gradients across nodes. Reach for it when a single node has enough memory to hold one replica's parameters comfortably and you want to shave the intra-node backward `all_gather`.

#### Worked example: HYBRID_SHARD vs FULL_SHARD on 2 nodes

Take the 13B model, 2 nodes of 8 A100s, per-GPU parameter-gather volume of roughly 24 GB/step ($\frac{N-1}{N}\cdot 26$ GB for the forward gather, similar for backward). Numbers are order-of-magnitude and depend on topology, but the ratio is the point.

Under `FULL_SHARD` across all 16 ranks, a large share of those 24 GB gathers crosses InfiniBand at an effective ~12 GB/s. Even if only half the volume is inter-node, that is ~12 GB over 12 GB/s ≈ 1.0 s per gather, and with two gathers plus a reduce-scatter you are looking at multiple seconds of communication per step — comms utterly dominates, and your 8-GPU-node throughput does not improve when you add the second node; it often gets *worse*.

Under `HYBRID_SHARD`, those 24 GB of gathers stay on NVLink at ~600 GB/s: 24 / 600 ≈ 0.04 s. The only inter-node traffic is the gradient `all_reduce` of the sharded gradient — about 3.25 GB per rank (26 GB / 8) reduced across 2 replicas, so $2\cdot\frac{2-1}{2}\cdot 3.25 \approx 3.25$ GB over 12 GB/s ≈ 0.27 s. Total comms drops from *seconds* to roughly a third of a second. The parameter gathers, which were killing you, are back on the fast wire where they belong. This is why HYBRID_SHARD is the default recommendation the moment you go multi-node with fat intra-node links: it makes the second node actually add throughput instead of subtracting it. If you have ever seen multi-node come out *slower* than single-node, an unqualified `FULL_SHARD` across nodes is one of the top two suspects, along with an interconnect that silently fell back to TCP.

## Mixed precision, done right

FSDP's `MixedPrecision` policy controls three dtypes independently, and getting them right is the difference between a stable run and a loss curve that drifts or NaNs. The three are:

- `param_dtype`: the dtype the parameters are cast to for the forward and backward compute. bf16 for the compute win.
- `reduce_dtype`: the dtype gradients are reduced (`reduce_scatter` / `all_reduce`) in. **fp32** for stability.
- `buffer_dtype`: the dtype for non-parameter buffers such as normalization running statistics and rotary caches. Usually fp32.

```python
from torch.distributed.fsdp import MixedPrecision

bf16_policy = MixedPrecision(
    param_dtype=torch.bfloat16,     # compute in bf16
    reduce_dtype=torch.float32,     # reduce gradients in fp32 -> stable
    buffer_dtype=torch.float32,     # keep buffers in fp32
)

model = FSDP(
    model,
    auto_wrap_policy=auto_wrap_policy,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    mixed_precision=bf16_policy,
    device_id=torch.cuda.current_device(),
    use_orig_params=True,
)
```

Why fp32 for the reduce, specifically? Because a gradient reduction is a *sum* across $N$ ranks, and summation in low precision loses small-magnitude contributions. In bf16, with only 8 bits of mantissa, adding a small gradient to a large running sum can round the small one away entirely — the classic large-plus-small cancellation. Across 8, 64, or 512 ranks that error compounds, and it manifests not as a crash but as a subtly wrong average gradient, a slightly-off optimizer step, and a loss curve that trains *almost* right and then diverges a few thousand steps in. Reducing in fp32 keeps the sum accurate; the parameters and activations can stay bf16. Set `reduce_dtype=torch.bfloat16` only when you have measured that your model tolerates it and you need the bandwidth — it roughly halves the gradient communication volume, which is tempting at scale, but it is the first thing to revert when a large run starts drifting for no visible reason.

bf16 versus fp16 matters here too. bf16 has the same 8-bit exponent as fp32, so it has fp32's dynamic range and does not need loss scaling; fp16 has a 5-bit exponent, a narrow range, and needs a `GradScaler` to avoid underflowing small gradients to zero. On A100 and H100, bf16 is the right default for training from scratch — no scaler, no overflow drama. Use fp16 only when you must (older hardware, a checkpoint trained in fp16), and when you do, remember the scaler. For the deeper treatment of why fp16 NaNs and bf16 does not, see [debugging FSDP and sharding bugs](/blog/machine-learning/debugging-training/fsdp-and-sharding-bugs).

## Stacking the second memory lever: activation checkpointing

Sharding attacks *model-state* memory — parameters, gradients, optimizer states. It does nothing for *activation* memory, the intermediate tensors the forward pass stashes so the backward pass can compute gradients. For a deep transformer with long sequences, activations can rival or exceed the model state, and they scale with batch size and sequence length in ways sharding cannot touch. So FSDP composes with a second, orthogonal lever: **activation checkpointing** (also called gradient checkpointing or recomputation). Instead of stashing every activation, you stash only the inputs to each checkpointed block and *recompute* the interior activations during the backward pass. You trade compute — one extra forward per block — for memory.

![A vertical stack of the per GPU memory budget under FSDP showing sharded parameters, sharded gradients, sharded optimizer state, and the activation working set shrinking with checkpointing](/imgs/blogs/fsdp-in-practice-4.webp)

The figure stacks the per-GPU budget so you can see the two levers acting on different bricks: sharding shrinks the three model-state bricks by dividing them across ranks, and checkpointing shrinks the activation brick by recomputing instead of storing. They are independent, and they multiply — that is what makes a 13B, or a 70B, fit on cards that could never hold it densely.

Composing the two is a wrapping operation, just like FSDP itself, and the order matters: apply activation checkpointing to the *same* block boundaries you gave FSDP, so each transformer block is both an FSDP unit and a recomputation unit.

```python
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    apply_activation_checkpointing,
    CheckpointImpl,
)

# Recompute each decoder layer's interior activations in the backward pass.
check_fn = lambda submodule: isinstance(submodule, LlamaDecoderLayer)

apply_activation_checkpointing(
    model,
    checkpoint_wrapper_fn=functools.partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,   # the modern, safer variant
    ),
    check_fn=check_fn,
)
```

Two practical notes. First, prefer `NO_REENTRANT` checkpointing (the non-reentrant autograd implementation); the older reentrant version has sharp edges with FSDP, with `requires_grad` toggling, and with anything that runs code between forward and backward. Second, the compute cost is real: full activation checkpointing adds roughly one extra forward pass, so about a 33% throughput hit for a transformer (backward is ~2x forward, so recomputing the forward adds ~1 of ~3 units). That is often a bargain — a third slower but able to run at all, or able to double the sequence length — but it is not free, and *selective* activation checkpointing (recompute only the cheap-to-recompute, memory-heavy ops such as attention, keep the rest) is the modern refinement that recovers much of the throughput. The full treatment of the recompute trade-off, selective policies, and how to measure the break-even lives in [activation checkpointing](/blog/machine-learning/distributed-training/activation-checkpointing).

## Prefetching: the overlap that makes FSDP fast

Everything so far has been about *fitting*. This section is about *speed*, and it hinges on a single idea: the parameter `all_gather` for the next block can run on the network *while the current block is still computing on the GPU*. If it does, communication hides behind computation and FSDP approaches DDP throughput. If it does not — if every block waits for its own gather to finish before it starts, then waits for the next gather before the next block — you serialize compute and comms end to end, and FSDP crawls.

![A left to right timeline of two transformer blocks under FSDP showing the next block all gather overlapping the current block compute while gradients reduce scatter in the background](/imgs/blogs/fsdp-in-practice-5.webp)

The timeline shows the overlap: while block $i$ computes, FSDP is already gathering block $i+1$'s parameters; when block $i$ finishes, block $i+1$'s parameters are already resident and it starts immediately, no stall. In the backward pass the same overlap runs in reverse, and the gradient `reduce_scatter` of a finished block streams out while the previous block recomputes. The network is never idle and the GPU is never waiting — that is the entire performance game.

Three knobs control it:

- **`backward_prefetch`**: when to kick off the next-needed `all_gather` during the backward pass. `BackwardPrefetch.BACKWARD_PRE` (prefetch the next block's parameters *before* the current block's backward computation) gives the best overlap at a slightly higher memory cost — two blocks' parameters are briefly resident. `BACKWARD_POST` prefetches after, using less memory but overlapping less. Default to `BACKWARD_PRE` unless you are memory-bound.
- **`forward_prefetch`**: prefetch the next block's parameters in the forward pass too. It helps CPU-bound forward passes where kernel launch latency would otherwise delay the gather; enable it (`forward_prefetch=True`) and measure.
- **`limit_all_gathers`** (the "rate limiter"): caps how far ahead FSDP prefetches so that it does not run so many gathers ahead of compute that the transient memory balloon blows the budget. It trades a little overlap for a memory ceiling. Leave it on (the default) unless you have profiled and have memory to spare.

```python
model = FSDP(
    model,
    auto_wrap_policy=auto_wrap_policy,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    mixed_precision=bf16_policy,
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # best backward overlap
    forward_prefetch=True,                            # overlap forward gathers too
    limit_all_gathers=True,                           # cap prefetch depth for memory
    device_id=torch.cuda.current_device(),
    use_orig_params=True,
)
```

### The mechanism: why overlap is the whole game

Model each step as compute time $T_\text{compute}$ and communication time $T_\text{comm}$. Without overlap, the step takes $T_\text{compute} + T_\text{comm}$ — every gather stalls the GPU. With perfect overlap, the step takes $\max(T_\text{compute}, T_\text{comm})$ — comms hides entirely behind compute as long as compute is the longer of the two. The whole design of FSDP prefetching is to move you from the sum to the max. On NVLink, where $T_\text{comm}$ for a per-block gather is tens of microseconds against block compute of hundreds of microseconds, overlap makes FSDP's 1.5x byte volume nearly invisible: the extra bytes fit inside the compute you were doing anyway, and measured throughput lands within a few percent of DDP. On a thin inter-node link, $T_\text{comm}$ can *exceed* $T_\text{compute}$, the max is now the comms term, and no amount of prefetching saves you — which is exactly why HYBRID_SHARD, by keeping gathers on NVLink, is the multi-node answer. Prefetch turns the sum into a max; HYBRID_SHARD makes sure the max is the compute term.

## FSDP1 vs FSDP2: per-parameter sharding and DTensor

Everything above is FSDP1 — the `FullyShardedDataParallel` wrapper class that has been in PyTorch since 1.11. Starting in PyTorch 2.4 and stabilizing through 2.6, there is a second, cleaner implementation: **FSDP2**, exposed as the `fully_shard` function. It is worth understanding what changed, because FSDP2 is where the ecosystem is heading and it removes a class of sharp edges.

![A before and after comparison of FSDP1 flattening a unit into one flat parameter versus FSDP2 sharding each parameter individually as a DTensor](/imgs/blogs/fsdp-in-practice-6.webp)

The difference the figure draws is the internal representation. FSDP1 takes all parameters in a unit, flattens them into a single 1D `FlatParameter`, and shards *that* flat buffer across ranks. It is efficient for communication — one contiguous buffer to gather — but it fuses parameters that were logically separate. FSDP2 shards each parameter *individually* as a `DTensor` (distributed tensor) sharded on dimension 0, so every parameter keeps its own identity, dtype, and gradient.

That per-parameter representation fixes several FSDP1 pain points at once:

- **Mixed requires_grad within a unit.** In FSDP1, if some parameters in a block are frozen and others are trainable, the FlatParameter cannot represent both cleanly and you fight it. FSDP2 shards each parameter separately, so frozen and trainable parameters coexist in one block with no ceremony — a big win for parameter-efficient finetuning and for partially-frozen models.
- **Per-parameter dtypes.** Different parameters can carry different dtypes without the FlatParameter forcing them into one buffer.
- **Cleaner state_dict.** Because each parameter is a DTensor, saving and loading is a matter of gathering or redistributing DTensors — no opaque flat buffer to reconstruct, and native compatibility with distributed checkpointing.
- **Composability with tensor parallelism.** FSDP2's DTensor sharding composes with tensor-parallel DTensor sharding on a 2D device mesh, so 2D parallelism (shard *and* tensor-parallel) is a mesh construction rather than a framework fight.
- **Lower and more predictable memory**, because there is no flat-parameter padding to a common size and dtype.

The API is different in shape: instead of one wrapper call with an `auto_wrap_policy`, you call `fully_shard` on each module you want to be a unit, bottom-up — typically each transformer block, then the root — and you pass the mesh and per-call options directly.

```python
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import init_device_mesh

mesh = init_device_mesh("cuda", (torch.distributed.get_world_size(),))
mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

# Apply fully_shard to each block first (bottom-up), then the root module.
for layer in model.model.layers:
    fully_shard(layer, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=True)
fully_shard(model, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=True)
```

Notice `reshard_after_forward` is now a per-call boolean rather than a global enum: `True` gives `FULL_SHARD` behavior for that unit, `False` gives `SHARD_GRAD_OP` behavior, and you can mix them per block if you want to keep the last few blocks resident. That per-unit control, expressed as an ordinary function argument, is the ergonomic heart of FSDP2. For a new project on PyTorch 2.6+, start with FSDP2; for an existing FSDP1 codebase that works, there is no urgency to migrate, but know that the two are not drop-in identical — the wrapping model and state_dict format differ.

## Saving and loading under sharding

Here is where a working FSDP job most often falls over in production, and it has nothing to do with training. Under sharding, the model state is spread across every rank — each rank holds only its shard of parameters and optimizer state. When you call for a checkpoint, FSDP has to decide how to assemble that distributed state into something on disk, and the choice you make there decides whether checkpointing scales or detonates.

There are two shapes of state dict, selected via `FSDP.state_dict_type`:

- **`FULL_STATE_DICT`** reconstructs the *entire* unsharded model on one rank (rank 0), as if it were never sharded, and saves a single file. Convenient — the checkpoint looks exactly like a normal single-GPU checkpoint, loadable anywhere. Dangerous — rank 0 must *materialize the whole model in memory* to build it. For a 70B model that is 140 GB of bf16 parameters gathered onto one host. If you do not offload to CPU, it OOMs the GPU; if you do offload but the host does not have 140 GB of RAM free, it OOMs the CPU. Either way, the run that trained fine for 20 hours dies at the first checkpoint.
- **`SHARDED_STATE_DICT`** keeps the state sharded: each rank saves its own shard, in parallel, through `torch.distributed.checkpoint` (DCP). No rank ever holds the full model. It scales to any model size, saves faster because every rank writes concurrently, and is the correct default for large models. The one cost is that the on-disk format is sharded — you load it back with DCP, and resharding to a different world size is handled by the checkpoint library rather than being free.

```python
import torch.distributed.checkpoint as dcp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig, ShardedStateDictConfig

# SCALES: sharded checkpoint via DCP -- every rank writes its own shard, in parallel.
with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT,
                          ShardedStateDictConfig(offload_to_cpu=True)):
    dcp.save({"model": model.state_dict()}, checkpoint_id="ckpt/step_10000")

# CONVENIENT BUT DANGEROUS: single consolidated file on rank 0 only.
# offload_to_cpu + rank0_only are mandatory or you OOM the GPU; you still
# need enough HOST RAM to hold the whole model, or you OOM the CPU.
full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_cfg):
    cpu_state = model.state_dict()
    if torch.distributed.get_rank() == 0:
        torch.save(cpu_state, "ckpt/step_10000_full.pt")
```

Loading a sharded checkpoint back is the mirror image, and it has one property that saves you at scale: DCP loads *in place* into the already-sharded model, so no rank ever holds more than its shard, and — critically — a sharded checkpoint can be *resharded* on load to a different world size. Save on 8 GPUs, resume on 16, and DCP redistributes the shards for you; a full state dict cannot do this without a manual gather-and-rescatter. You must construct the FSDP model and optimizer first, then load their state dicts into the live objects:

```python
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

# Build model + optimizer (already FSDP-wrapped), THEN load into them in place.
model_sd, optim_sd = get_state_dict(model, optimizer)
state = {"model": model_sd, "optim": optim_sd}
dcp.load(state, checkpoint_id="ckpt/step_10000")   # resharding-aware
set_state_dict(model, optimizer, model_state_dict=state["model"],
               optim_state_dict=state["optim"])
```

The optimizer half is the part people forget, and forgetting it is the classic cause of the loss spike right after a resume: you restore the parameters but reinitialize Adam's moments to zero, so the first post-resume step takes a wrongly-scaled jump before the moments warm back up. Always checkpoint and restore the optimizer state alongside the model, and always through the sharded path so its size does not OOM anyone. For very large runs, DCP also supports *asynchronous* saves — the save is staged to CPU and written in the background while training continues — so a checkpoint costs you a brief stall instead of a full stop. The correctness of all of this under sharding, plus resharding across a changed world size, is the subject of a dedicated post; here the rule is simply: save sharded, load sharded, and never leave the optimizer state behind.

The rule of thumb: **train with `SHARDED_STATE_DICT`, export with `FULL_STATE_DICT`.** During training, checkpoint sharded — it is fast, it scales, it never OOMs. Only at the very end, when you want a single portable file for inference or for sharing, consolidate to a full state dict once, on a host you have confirmed has the RAM. And do not forget the optimizer: the Adam state is sharded too, and it is *larger* than the parameters (12 bytes/param vs 2), so a full-state-dict save of the optimizer is the biggest OOM risk of all. DCP handles the optimizer state alongside the model when you save sharded. This is a foreshadowing of a whole post — [distributed checkpointing](/blog/machine-learning/distributed-training/the-distributed-training-playbook) covers async saves, resharding across a changed world size, and correctness under sharding — but the one-line takeaway is: if your checkpoint OOMs, you almost certainly saved a full state dict when you should have saved sharded.

## A complete, runnable FSDP recipe

Here is the whole thing assembled — wrap policy, `FULL_SHARD`, bf16 mixed precision with fp32 reduce, activation checkpointing, backward prefetch — the config you would actually launch for the 13B model on 8 GPUs.

```python
import functools
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy, MixedPrecision, BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper, apply_activation_checkpointing, CheckpointImpl,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

def main():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    model = build_llama_13b()  # your model, on CPU or meta device

    # 1. Wrap EACH decoder layer as its own FSDP unit -- the load-bearing line.
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LlamaDecoderLayer},
    )

    # 2. bf16 compute, fp32 gradient reduction for stability.
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32,
    )

    model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,   # HYBRID_SHARD for multi-node
        mixed_precision=mp_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        forward_prefetch=True,
        limit_all_gathers=True,
        device_id=local_rank,
        use_orig_params=True,
    )

    # 3. Compose the second memory lever: recompute each block in backward.
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=functools.partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
        check_fn=lambda m: isinstance(m, LlamaDecoderLayer),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
    for batch in dataloader:
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        loss.backward()
        model.clip_grad_norm_(1.0)   # FSDP-aware grad clipping across shards
        optimizer.step()

    dist.destroy_process_group()

if __name__ == "__main__":
    import os
    main()
```

Two details that trip people up. Use `model.clip_grad_norm_(...)` — FSDP's own method — not `torch.nn.utils.clip_grad_norm_`, because the gradient norm has to be computed across shards on all ranks; the free function only sees the local shard and clips wrong. And prefer building the model on the `meta` device and letting FSDP materialize shards, so you never instantiate the full model on one GPU before sharding — for a 13B model, instantiating densely on rank 0 first can OOM before FSDP even runs.

The launch is ordinary `torchrun`. Single node, 8 GPUs:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 train_fsdp.py
```

Two nodes of 8 (switch the strategy to `HYBRID_SHARD` in code first), with a rendezvous so the ranks find each other:

```bash
# On BOTH nodes; set NODE_RANK=0 on the first, 1 on the second.
torchrun \
  --nnodes=2 --node_rank=$NODE_RANK --nproc_per_node=8 \
  --rdzv_backend=c10d --rdzv_endpoint=$HEAD_NODE_IP:29500 \
  train_fsdp.py

# NCCL env that matters on multi-node: name the InfiniBand HCA and the right NIC,
# and turn on debug the first time you bring up a new cluster.
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5
export NCCL_SOCKET_IFNAME=eth0
```

If that multi-node launch runs *slower* than the single-node one, the first two things to check are (1) that you switched to `HYBRID_SHARD`, and (2) that NCCL is actually using InfiniBand and not silently falling back to TCP over Ethernet — an `NCCL_DEBUG=INFO` log will tell you which transport each rank chose.

## Measuring FSDP honestly: tokens/s, MFU, and peak memory

You cannot tune what you cannot measure, and FSDP is full of knobs whose effect is only visible in numbers. Three numbers matter: **peak memory per GPU** (does it fit, and how much headroom), **tokens per second** (throughput), and **MFU** — model FLOPs utilization, the fraction of the GPU's peak floating-point rate you actually achieve. MFU is the honest north star because it is hardware-normalized: a run at 45% MFU on A100 is doing real work, a run at 12% MFU is leaving three-quarters of the silicon idle no matter how many GPUs you throw at it.

Measuring these correctly is easy to get wrong. Four rules:

- **Warm up first.** The first several steps include CUDA context creation, cuDNN autotuning, allocator growth, and NCCL ring setup. Throw away the first 10–20 steps and measure steady state.
- **Synchronize before you time.** CUDA kernels are asynchronous; without `torch.cuda.synchronize()` you are timing kernel *launches*, not kernel *execution*, and your numbers are fiction.
- **Measure peak, not current, memory.** Use `torch.cuda.max_memory_allocated()` after a `reset_peak_memory_stats()`, over a full step including the backward pass — the transient gather balloon peaks mid-backward, not at rest.
- **Watch the data-loader confound.** If the GPU is starving because the loader cannot keep up, you will measure low throughput and blame FSDP when the fix is `num_workers` and `prefetch_factor`. Confirm the loader is not the bottleneck before attributing a slowdown to communication.

```python
import time, torch

def measure_step(model, batch, optimizer, warmup=15, iters=50):
    torch.cuda.reset_peak_memory_stats()
    for i in range(warmup + iters):
        if i == warmup:
            torch.cuda.synchronize(); t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        loss.backward()
        model.clip_grad_norm_(1.0)
        optimizer.step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters              # seconds/step, steady state
    peak_gb = torch.cuda.max_memory_allocated() / 1e9    # peak, includes backward balloon
    tokens = batch["input_ids"].numel()
    return tokens / dt, peak_gb                           # tokens/s/GPU, peak GB
```

The MFU calculation closes the loop. A dense transformer costs about $6N$ FLOPs per token for a forward-plus-backward pass, where $N$ is the parameter count; full activation checkpointing adds one extra forward, about $2N$, for roughly $8N$ per token. Divide achieved FLOPs/s by the GPU's peak rate — 312 dense bf16 TFLOP/s on an A100 SXM — to get MFU. That single number tells you whether your next hour is better spent on the wrap policy, the prefetch config, or the interconnect.

#### Worked example: the FSDP throughput ladder on 8x A100 80GB

The 13B model, A100 80GB SXM, sequence length 2048. Numbers are approximate and illustrative of the *relationships* — exact MFU depends on the model, the kernels, and the PyTorch version — but the ordering and the magnitudes are representative of real runs.

| Config | Fits on 8 GPU? | Peak mem/GPU | ~MFU | Note |
|---|---|---|---|---|
| DDP (`NO_SHARD`) | No — needs ~208 GB | OOM | — | Model state alone exceeds one card |
| FSDP `FULL_SHARD`, whole-model wrap | No | ~93 GB | — | The intro OOM; transient balloon kills it |
| FSDP `FULL_SHARD`, per-block wrap | Yes | ~42 GB | ~45–50% | The configuration that works |
| + activation checkpointing | Yes, at longer seq | ~30 GB | ~38–42% | ~25–33% slower, fits longer sequences |
| + bf16 reduce (bandwidth-bound only) | Yes | ~30 GB | ~40–44% | Halves grad comms; verify stability first |

Read the table top to bottom as the journey the intro started: DDP cannot fit, whole-model FSDP cannot fit, per-block FSDP fits at healthy MFU, and activation checkpointing trades some of that MFU for the memory to run longer sequences or bigger batches. Every row is one knob from this post.

Now cross the node boundary and measure the multi-node effect, 2 nodes of 8 A100s with HDR InfiniBand between them:

| Config | ~throughput vs 8-GPU | ~scaling efficiency | Why |
|---|---|---|---|
| `FULL_SHARD` across all 16 | ~1.1x | ~55% | Param gathers cross InfiniBand every layer |
| `HYBRID_SHARD` (shard in node) | ~1.8x | ~90% | Gathers stay on NVLink, only grads cross IB |

The `FULL_SHARD`-across-16 row is the multi-node-slower-than-you-hoped trap made numeric: you doubled the GPUs and got 10% more throughput, a scaling efficiency of 55% that makes the second node almost pure waste. `HYBRID_SHARD` recovers it — 1.8x throughput at 90% efficiency — for the cost of one changed enum and a device mesh. That single measurement is the strongest argument in this post for reaching for the right strategy instead of the default one.

## Which strategy should you reach for?

Put the whole decision in one place. The question is always "what is the smallest amount of sharding that makes the model fit, on the interconnect I actually have?" — because every increment of sharding costs communication, and communication is the thing that erodes your scaling efficiency.

![A decision tree routing from whether the model fits on one GPU through single node versus multi node to the four FSDP sharding strategies](/imgs/blogs/fsdp-in-practice-7.webp)

The tree above routes the decision: if the model plus its Adam state fits on one GPU, do not use FSDP at all — use DDP (`NO_SHARD` or plain `DistributedDataParallel`) and keep DDP's simpler, cheaper single `all_reduce`. If it does not fit, ask whether it fits *sharded* on a single node: if you are on one node, use `FULL_SHARD` when memory is tight or `SHARD_GRAD_OP` when you have a few GB of headroom and want DDP-level comms. If you need more than one node, use `HYBRID_SHARD` so the parameter gathers stay on NVLink and only the gradient sync crosses InfiniBand.

### When to reach for FSDP, and when not to

- **Do not use FSDP if the model fits under DDP.** DDP moves fewer bytes (one `all_reduce` versus FSDP's gather-gather-scatter), it is simpler, and it has fewer knobs to misconfigure. FSDP is a memory tool; if you are not memory-bound, it is pure overhead. The one exception is `NO_SHARD` as a code-path toggle so a single script can flip between DDP and sharded without a rewrite.
- **Do not wrap the whole model as one unit.** Ever. It is the silent no-win from the first section — full comms cost, zero memory benefit. Always pass an `auto_wrap_policy`.
- **Do not use `FULL_SHARD` across nodes without thinking.** Unqualified `FULL_SHARD` stripes parameter gathers across the thin inter-node link and can make multi-node slower than single-node. Reach for `HYBRID_SHARD` the moment you cross a node boundary with fast intra-node links.
- **Do not reduce gradients in bf16 by default.** Keep `reduce_dtype=torch.float32` unless you have measured that your model tolerates bf16 reduction and you need the bandwidth. A wrong reduce dtype does not crash — it drifts, and drift at step 5,000 is expensive to debug.
- **Do not save a full state dict during training.** Save `SHARDED_STATE_DICT` through DCP; consolidate to a full state dict once at the end, on a host you have confirmed has the RAM. The optimizer state is the biggest OOM risk here — it is six times larger than bf16 parameters.
- **Do reach for FSDP** the moment the model, gradients, and optimizer state do not fit under DDP and you have a fast interconnect. It is the most bandwidth-efficient way to fit a large model on commodity multi-GPU nodes, and with per-block wrapping and prefetch it lands within a few percent of DDP throughput while fitting models DDP cannot touch.

## Case studies and real numbers

**PyTorch FSDP fitting large models.** The FSDP paper and the PyTorch team's reports (Zhao et al., 2023, *"PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel"*) demonstrate FSDP training models from 611M up to 175B parameters, and show that with per-block wrapping and prefetch, FSDP achieves near-linear scaling on GPT-family models across hundreds of A100s. The headline practical result matches this post's spine: models that do not fit under DDP fit under `FULL_SHARD`, and the throughput cost of the extra communication is small when it overlaps with compute on a fast interconnect. The paper explicitly documents the transient-memory dependence on wrap granularity — the 40x effect from the first section is not folklore, it is the measured behavior.

**HYBRID_SHARD on multi-node.** The PyTorch team's HYBRID_SHARD results show that for models in the tens of billions of parameters on clusters with fast intra-node NVLink and slower inter-node InfiniBand, hybrid sharding recovers a large fraction of the throughput lost by full sharding across nodes, precisely because it confines the frequent, large parameter gathers to the intra-node fabric. This is the single most impactful multi-node knob and the reason the strategy exists.

**ZeRO, the parent idea.** FSDP is PyTorch's native realization of the sharding scheme introduced by ZeRO (Rajbhandari et al., 2020, *"ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"*). ZeRO's three stages — shard optimizer states (stage 1), add gradients (stage 2), add parameters (stage 3) — map onto FSDP's `SHARD_GRAD_OP` (stage 2) and `FULL_SHARD` (stage 3). The `(2 + 2 + 12)Ψ` memory accounting this post uses comes straight from that paper's analysis of mixed-precision Adam. For the DeepSpeed implementation of the same ideas, including CPU and NVMe offload beyond what FSDP offers, see the [DeepSpeed ZeRO and 3D parallelism deep dive](/blog/machine-learning/open-source-library/deepspeed-zero-3d-parallelism-deep-dive).

**bf16 versus fp16 at scale.** The large-model training literature (the OPT-175B logbook, the BLOOM training reports) is a catalogue of fp16 instability — loss spikes, NaNs, and the operational pain of loss scaling across hundreds of ranks — and a strong argument for bf16, whose fp32-range exponent removes the underflow class of failures entirely. This is why every knob in this post defaults to bf16 compute with fp32 reduction: it is the configuration that large runs converged on after paying for the alternatives.

## The gotchas that bite everyone

A consolidated list, because each of these has cost someone a run:

- **Forgetting the wrap policy** → full comms cost, zero memory win, and a silent OOM later on a model that "should" fit. Always pass `auto_wrap_policy` (FSDP1) or call `fully_shard` per block (FSDP2).
- **Wrong mixed-precision reduce dtype** → a subtly wrong averaged gradient that trains almost-right and then diverges. Keep `reduce_dtype=torch.float32` until proven otherwise.
- **Saving a full state dict** → rank 0 gathers the whole model (and the even-larger optimizer state) onto one host and OOMs the CPU. Save `SHARDED_STATE_DICT`; consolidate once at the end.
- **`FULL_SHARD` across nodes** → parameter gathers cross the thin link every layer, multi-node comes out slower than single-node. Use `HYBRID_SHARD`.
- **`torch.nn.utils.clip_grad_norm_` instead of `model.clip_grad_norm_`** → the norm is computed on the local shard only and clipping is wrong. Use FSDP's method.
- **Instantiating the dense model on rank 0** before sharding → OOM before FSDP even runs. Build on `meta` device and let FSDP materialize shards.
- **NCCL falling back to TCP** on multi-node → InfiniBand unused, throughput tanks, and no error. Check `NCCL_DEBUG=INFO`, set `NCCL_IB_HCA` and `NCCL_SOCKET_IFNAME`.

## Key takeaways

1. **The wrap policy is the decision.** FSDP shards at the granularity of the wrapped unit; wrap per transformer block, never the whole model. This one line is the difference between a 40x transient-memory balloon and a rolling per-block gather.
2. **Pick the least sharding that fits.** `NO_SHARD` (DDP) if it fits, `SHARD_GRAD_OP` if you have headroom, `FULL_SHARD` when tight — every step up the ladder costs communication.
3. **`FULL_SHARD` moves 1.5x the bytes of DDP.** Gather-gather-scatter versus one all-reduce. That is the price of the memory win; overlap hides it on a fast interconnect.
4. **`HYBRID_SHARD` is the multi-node answer.** Shard within the node on NVLink, replicate across nodes on InfiniBand, so the heavy parameter gathers never touch the thin link.
5. **bf16 compute, fp32 reduce.** bf16 has fp32's range and needs no loss scaling; reducing gradients in fp32 keeps the cross-rank sum accurate and the loss stable.
6. **The two memory levers stack.** Sharding shrinks model state; activation checkpointing shrinks activations. They are independent and they multiply.
7. **Prefetch turns a sum into a max.** Overlap the next block's gather with the current block's compute; without it FSDP serializes and crawls.
8. **FSDP2 shards per parameter as DTensors.** Cleaner state dicts, frozen-parameter support, tensor-parallel composability, per-unit `reshard_after_forward`. Start new projects here.
9. **Train sharded, export full.** `SHARDED_STATE_DICT` through DCP during training; consolidate to a full state dict once at the end on a host with the RAM.
10. **Most FSDP failures are silent.** No wrap policy, wrong reduce dtype, full state dict, `FULL_SHARD` across nodes — none of them throw; they just cost you memory, stability, or throughput. Configure deliberately.

## Further reading

- [ZeRO and FSDP: the memory model](/blog/machine-learning/distributed-training/zero-and-fsdp-the-memory-model) — the `(2 + 2 + 12)Ψ` math and the sharding intuition this post builds on.
- [DDP from first principles](/blog/machine-learning/distributed-training/ddp-from-first-principles) — the gradient all-reduce and the 1.5x baseline FSDP is measured against.
- [Activation checkpointing](/blog/machine-learning/distributed-training/activation-checkpointing) — the second memory lever in full, including selective recomputation and the break-even.
- [Why distributed training](/blog/machine-learning/distributed-training/why-distributed-training) — the four walls; FSDP is the tool for the wall where the model won't fit.
- [The distributed training playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook) — the capstone decision and debugging checklist, including distributed checkpointing.
- [Debugging FSDP and sharding bugs](/blog/machine-learning/debugging-training/fsdp-and-sharding-bugs) — when mixed precision, wrapping, or state dicts go wrong at scale.
- [DeepSpeed ZeRO and 3D parallelism deep dive](/blog/machine-learning/open-source-library/deepspeed-zero-3d-parallelism-deep-dive) — the other major implementation of the same sharding ideas, with CPU/NVMe offload.
- Zhao et al., *PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel* (2023); Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* (2020); the [PyTorch FSDP documentation](https://pytorch.org/docs/stable/fsdp.html) and the FSDP2 `fully_shard` API notes.
