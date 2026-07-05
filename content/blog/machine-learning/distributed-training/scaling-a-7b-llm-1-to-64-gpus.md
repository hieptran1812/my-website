---
title: "Scaling a 7B LLM From 1 to 64 GPUs: An MFU Journey"
date: "2026-07-02"
publishDate: "2026-07-02"
description: "Follow one 7B model from a single GPU that won't even fit it to a 64-GPU cluster running at 87% scaling efficiency — hitting a wall at every step, diagnosing it with real numbers, fixing it, and scoring the result in MFU."
tags:
  [
    "distributed-training",
    "multi-gpu",
    "multi-node",
    "fsdp",
    "mfu",
    "pytorch",
    "nccl",
    "gpu",
    "deep-learning",
    "ml-systems",
  ]
category: "machine-learning"
subcategory: "Distributed Training"
author: "Hiep Tran"
featured: true
readTime: 30
---

Here is the moment that starts most of these stories. You have a 7-billion-parameter model, a fresh reservation on an eight-GPU A100 node, and a deadline. You wrap the model in `DistributedDataParallel`, launch it across all eight cards, watch the first log line print — and the throughput is barely faster than the single-GPU prototype you ran last week. Eight times the hardware, roughly the same tokens per second. The bill is eight times larger and the model is not learning any faster. Somewhere between one GPU and eight, you gave away almost all of the compute you paid for, and the training log is not telling you where.

This post is the whole journey out of that hole, told as one continuous run. We take a single 7B model and scale it from **1 GPU → 8 GPUs (one node) → 64 GPUs (eight nodes)**, and at every jump we hit a wall, form a hypothesis, measure, fix it, and write down the one number that tells the truth: **Model FLOPs Utilization**, or **MFU** — the fraction of the GPU's theoretical peak arithmetic that your model's math actually consumes. MFU is the score. Everything else — tokens per second, scaling efficiency, memory headroom — is in service of pushing that one percentage up. The figure below is the entire arc in one strip: a wall, a fix, a number; a bigger cluster, a new wall, a new fix, a new number.

![a left to right strip of five milestones showing the model going from one GPU that runs out of memory through eight and sixty four GPU stages with the efficiency at each](/imgs/blogs/scaling-a-7b-llm-1-to-64-gpus-1.webp)

By the end you will be able to do the arithmetic that predicts whether a model fits before you launch; compute MFU honestly from a training log; recognize the four classic walls in order (won't fit, GPU starved, comms exposed, batch too big to converge); and name the single highest-leverage fix at each scale. This is the fifth track of the [Distributed Training in the Trenches](/blog/machine-learning/distributed-training/why-distributed-training) series — the war-story track — and it is where the techniques from every earlier post get applied to one run and scored. The [four walls](/blog/machine-learning/distributed-training/why-distributed-training) from the intro are the spine of the whole trip: the model won't fit, the data won't feed it, the run is too slow, and every wasted percent of MFU is money. We are going to hit all four, in order, and knock each one down.

## Step 0: one GPU, and the wall you hit before the first step

Start where everyone starts: one GPU, `model.cuda()`, a training loop. Except a 7B model trained with AdamW does not fit on one 80 GB A100, and it never will. Before we can talk about throughput we have to get past *fitting at all* — the first wall.

The arithmetic is short and you should be able to do it on the back of an envelope. Let $\Psi$ be the parameter count, here $\Psi = 7 \times 10^{9}$. A mixed-precision AdamW step keeps four things resident on the card:

- **Weights** in bf16: $2\Psi$ bytes = 14 GB.
- **Gradients** in bf16: $2\Psi$ bytes = 14 GB.
- **Optimizer states**: AdamW keeps an fp32 master copy of the weights plus fp32 first- and second-moment estimates, so $4\Psi + 4\Psi + 4\Psi = 12\Psi$ bytes = 84 GB.
- **Activations**: everything the forward pass saves for the backward pass — the one term that scales with batch and sequence length, not with $\Psi$.

The three fixed terms already sum to the number that decides the run:

$$
M_\text{state} = \underbrace{2\Psi}_{\text{weights}} + \underbrace{2\Psi}_{\text{grads}} + \underbrace{12\Psi}_{\text{optimizer}} = 16\Psi = 112\ \text{GB}.
$$

The figure below is that budget as a stack. The 84 GB of optimizer state — not the weights everyone quotes — is what pushes the total past the 80 GB card. You could delete the entire model from memory and the *optimizer* alone would still overflow.

![a vertical stack of four memory consumers for a seven billion parameter run showing optimizer state as the tallest bar and the total overflowing an eighty gigabyte card](/imgs/blogs/scaling-a-7b-llm-1-to-64-gpus-2.webp)

This derivation is the whole subject of [the memory budget](/blog/machine-learning/distributed-training/the-memory-budget); here it is the wall. There are exactly two ways past it on a single GPU: shrink the state, or shard it. Sharding needs more than one GPU (that is Step 1), so on one card we shrink.

#### Worked example: squeezing a 7B run onto one card

To get an honest single-GPU baseline at all, we lean on three memory levers at once:

- **8-bit AdamW** (bitsandbytes `paged_adamw_8bit`): the two moment estimates are block-wise quantized to 8-bit, cutting optimizer state from roughly 8 bytes per parameter to about 2, while the fp32 master stays. Optimizer memory drops from 84 GB to roughly 42 GB.
- **Full activation checkpointing**: save only layer boundaries and recompute the rest in the backward pass. Activation memory for a 7B transformer at sequence length 2048 falls from tens of GB to a couple of GB — at the cost of recomputing the forward pass, which we will pay for in throughput. This is the trade dissected in [activation checkpointing](/blog/machine-learning/distributed-training/activation-checkpointing).
- **Micro-batch of 1**: the smallest batch that runs.

The new ledger: 14 (weights) + 14 (grads) + 42 (8-bit optimizer) + ~2 (checkpointed activations) + ~2 (CUDA context and allocator slack) ≈ **74 GB**. It fits on an 80 GB card — barely, with a fragmentation cushion so thin that a slightly longer sequence tips it into `CUDA out of memory`. This is the run that *technically* trains 7B on one GPU, and it is miserable: a tiny batch, recompute overhead on every step, and no second card to hide anything behind.

Measured on a single A100 80GB SXM: about **1,500 tokens/second**. To know whether that is good or a disaster, we need the score.

## The one number that scores everything: MFU

Tokens per second is a raw rate; it tells you nothing about whether the GPU is working hard or loafing. The honest score is **Model FLOPs Utilization** — the ratio of the useful arithmetic your model performs to the arithmetic the hardware could perform at its rated peak.

A dense transformer does approximately $6\Psi$ floating-point operations per token: the forward pass is about $2\Psi$ (each parameter participates in one multiply-add, which counts as two FLOPs) and the backward pass is about twice that, $4\Psi$, because it computes gradients with respect to both activations and weights. (The attention score computation adds a term that grows with sequence length; for a 7B model at moderate context it is a single-digit percentage, and the standard MFU convention omits it, so we will too and note that real MFU is a hair higher.)

Let $T$ be aggregate tokens per second across the job, $D$ the number of GPUs, and $P$ the per-GPU peak throughput in the training precision — for an A100 in bf16, $P \approx 312$ TFLOP/s. Then:

$$
\text{MFU} = \frac{6\,\Psi\,T}{D \cdot P}.
$$

The single-GPU baseline: $6 \times 7\times10^{9} \times 1500 = 6.3\times10^{13}$ model FLOP/s, against $1 \times 312\times10^{12} = 3.12\times10^{14}$ FLOP/s of peak. That is an MFU of **20%**. Four-fifths of the card's arithmetic is going to waste — some to the recompute we bought with checkpointing, most to the tiny batch that leaves the tensor cores under-fed. Twenty percent is a floor to climb from, not a failure; it is what a cramped single-GPU 7B run looks like.

Two distinctions keep this number honest:

- **MFU vs HFU.** *Hardware* FLOPs Utilization counts every FLOP the GPU actually executes, including the recomputed forward pass from activation checkpointing. *Model* FLOPs Utilization counts only the FLOPs the model logically requires ($6\Psi$ per token). Checkpointing *raises* HFU (more real work) while it can *lower* MFU per unit time — so always quote which one you mean. MFU is the one that maps to "how fast is my loss actually dropping per dollar," which is why it is the score in this post.
- **Measuring it without lying to yourself.** MFU derived from a bad measurement is worse than none. Warm up for a few dozen steps so cuDNN autotuning and the allocator settle. Call `torch.cuda.synchronize()` before you read the clock — CUDA kernels are asynchronous and an un-synced timer measures the Python launch, not the GPU. Time steady state, not step 0. And watch the loader confound: if the GPU is idling waiting for data, your tokens/second is a *data-pipeline* number wearing an MFU costume. We will walk straight into that trap in Step 1.

#### Worked example: reading MFU off a log line

A training log prints `step 400 | 2.05 s/step | global tokens 33,600`. Aggregate throughput is $33{,}600 / 2.05 = 16{,}390$ tokens/second. On 8 A100s:

$$
\text{MFU} = \frac{6 \times 7\times10^{9} \times 16{,}390}{8 \times 312\times10^{12}} = \frac{6.88\times10^{14}}{2.496\times10^{15}} = 0.276 \approx 28\%.
$$

Twenty-eight percent on eight GPUs is the number that opens Step 1 — and it is the number that should make you suspicious, because a well-fed 7B FSDP run on A100s should be in the low-to-mid forties. Ten-plus points of MFU are missing. Let's find them.

## Step 1: eight GPUs, and the OOM that comes back

Eight GPUs, one DGX-class node, all cards wired together with NVLink through an NVSwitch. The obvious move is `DistributedDataParallel`: replicate the model on every GPU, split the batch across them, and average gradients with an all-reduce each step. DDP is the right default for models that fit, and it is the subject of [DDP from first principles](/blog/machine-learning/distributed-training/ddp-from-first-principles). But DDP *replicates* — every GPU holds the full 112 GB of state. We already know 112 GB does not fit in 80 GB. So the first thing that happens on eight GPUs is the exact same OOM we hit on one, eight times in parallel.

The fix is to stop replicating the state and start **sharding** it. Fully Sharded Data Parallel (FSDP) splits the parameters, gradients, and optimizer states across the ranks — each of the eight GPUs owns one-eighth of the model's state and materializes a full layer's parameters only for the moment it is computing that layer, by **all-gathering** the shards from its peers. (An all-gather is the collective where every rank contributes its piece and every rank ends up with the whole.) With FULL_SHARD across eight GPUs, resident state per card drops to ${112 / 8 = 14}$ GB, and the run fits with tens of GB to spare. The contrast is the difference between a run that dies and a run that breathes.

![a two column comparison of naive data parallel replicating one hundred twelve gigabytes per GPU and running out of memory versus fully sharded data parallel holding fourteen gigabytes per GPU and fitting](/imgs/blogs/scaling-a-7b-llm-1-to-64-gpus-3.webp)

Here is the FSDP wrap. This is the real API — `ShardingStrategy.FULL_SHARD`, a transformer auto-wrap policy so each block becomes its own sharded unit, and (for now) fp32 compute so we can isolate one variable at a time:

```python
import functools
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# init_process_group("nccl") and set_device(local_rank) already done by the launcher.
wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={LlamaDecoderLayer},
)

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # shard params + grads + optimizer
    auto_wrap_policy=wrap_policy,
    device_id=torch.cuda.current_device(),
)
```

It fits. We launch. And the score is **28% MFU** — the very log line from the worked example above. Eight GPUs, mid-twenties MFU, when the single-node ceiling should be the mid-forties. The model fits, the run is stable, and it is leaving nearly half its compute on the floor. Wall number two.

### The GPU is starving

When MFU is low but the math is sound, the first suspect is always the same: the GPU is idle, waiting for something. Watch `nvidia-smi dmon` during a step and you see it — GPU utilization sawtoothing between 95% and 30%, a rhythmic stall. The tensor cores burn through a batch faster than the data loader can assemble the next one, so the card sits waiting on the CPU. This is the loader confound from the MFU section, now costing real money. It is the whole subject of [the data pipeline at scale](/blog/machine-learning/distributed-training/the-data-pipeline-at-scale), and at eight GPUs it is the single most common reason a fit run is slow.

The default `DataLoader` uses too few workers and no prefetch, so tokenization and collation happen on the critical path. The fix is to move that work off the critical path entirely:

```python
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
loader = DataLoader(
    dataset,
    batch_size=micro_batch,
    sampler=sampler,
    num_workers=8,          # was 2 — parallelize tokenization across CPU cores
    prefetch_factor=4,      # each worker stages 4 batches ahead of the GPU
    pin_memory=True,        # page-locked host memory → faster H2D copy
    persistent_workers=True # don't respawn workers every epoch
)
```

`num_workers=8` puts eight CPU processes to work assembling batches in parallel; `prefetch_factor=4` means each keeps four batches staged ahead, so a full pipeline of 32 batches is always ready; `pin_memory=True` lets the host-to-device copy run asynchronously over DMA; and `persistent_workers=True` stops the per-epoch respawn stall. The sawtooth flattens, GPU utilization pins near 95%, and MFU climbs from 28% to **33%** with no change to the model at all. Five points of MFU that were never a model problem — they were a plumbing problem. This is why the loader confound is the first thing to rule out, not the last.

## Step 2: the levers that took eight GPUs from 28% to 44%

Feeding the GPU got us to 33%. The rest of the climb to the single-node ceiling is a stack of independent levers, each worth a few points, and the discipline is to apply them **one at a time and re-measure** — never bundle three changes and guess which one helped. The figure below is the ladder we are about to climb; no single rung is dramatic, but they compound.

![a five row table showing MFU rising from twenty eight to forty four percent as the loader fix bf16 activation checkpointing and communication overlap are added one at a time](/imgs/blogs/scaling-a-7b-llm-1-to-64-gpus-5.webp)

**bf16 mixed precision (33% → 39%).** Our isolating baseline ran compute in fp32. Switching the forward and backward matmuls to bf16 through FSDP's `MixedPrecision` config roughly doubles tensor-core throughput — A100 tensor cores run bf16 at 312 TFLOP/s versus 156 for fp32-accumulate paths — and halves the bytes moved in every collective. bf16 is the right choice over fp16 here because its wider exponent range makes loss scaling unnecessary and NaNs far rarer at scale, the trade-off laid out in [mixed precision at scale](/blog/machine-learning/distributed-training/mixed-precision-at-scale). Six points.

**Activation checkpointing to grow the batch (39% → 42%).** With state at 14 GB per card we have ~60 GB of headroom, and the way to convert headroom into MFU is a bigger batch — bigger matmuls run closer to peak because they amortize kernel-launch and memory-latency overhead over more arithmetic. Activation checkpointing frees the activation memory that would otherwise cap batch size, letting us push the micro-batch from 1 to 8. Yes, checkpointing recomputes the forward pass (that is HFU spent to buy MFU), but the larger, more efficient matmuls more than pay it back. Three points.

**Overlapping communication with compute (42% → 44%).** This is the lever that matters most as you scale, so it is worth seeing clearly. FSDP has to all-gather each layer's parameters before it can compute that layer. Done naively, the GPU stalls at every layer boundary waiting for the gather to finish — comms on the critical path. The fix is to run the all-gather for layer $k+1$ on a separate CUDA stream *while* the matmul for layer $k$ is still running, so the parameters arrive just before they are needed and the communication hides behind computation. This prefetch is the entire game of [overlapping compute and communication](/blog/machine-learning/distributed-training/overlapping-compute-and-communication), and the figure shows the shape of it: the step branches into a compute stream and a prefetch stream that reconverge with the comms already paid for.

![a branching dataflow where a training step forks into an all gather stream a matmul stream and a prefetch stream that merge into the layer output with communication hidden](/imgs/blogs/scaling-a-7b-llm-1-to-64-gpus-4.webp)

Here is the tuned FSDP setup with all three levers on — mixed precision, activation checkpointing, and the prefetch flags that turn on overlap:

```python
import functools
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision, BackwardPrefetch
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing, checkpoint_wrapper, CheckpointImpl,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

bf16 = MixedPrecision(
    param_dtype=torch.bfloat16,      # compute params in bf16
    reduce_dtype=torch.bfloat16,     # collectives in bf16 → half the bytes
    buffer_dtype=torch.bfloat16,
)
wrap_policy = functools.partial(
    transformer_auto_wrap_policy, transformer_layer_cls={LlamaDecoderLayer}
)

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    mixed_precision=bf16,
    auto_wrap_policy=wrap_policy,
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # gather next shard early
    forward_prefetch=True,                            # overlap in forward too
    limit_all_gathers=True,                           # cap in-flight gathers, avoid OOM
    device_id=torch.cuda.current_device(),
)

# Recompute each transformer block's forward during backward to free activation memory.
apply_activation_checkpointing(
    model,
    checkpoint_wrapper_fn=functools.partial(
        checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
    ),
    check_fn=lambda m: isinstance(m, LlamaDecoderLayer),
)
```

The scorecard for the single node, isolating each lever:

| Lever added | MFU | tok/s per GPU | What it fixed |
|---|---|---|---|
| FSDP FULL_SHARD (baseline) | 28% | 2,080 | fits, but GPU starves |
| + data loader (workers/prefetch/pin) | 33% | 2,450 | GPU stays fed |
| + bf16 mixed precision | 39% | 2,900 | 2× tensor-core rate, half the comms bytes |
| + activation ckpt, micro-batch 8 | 42% | 3,120 | bigger, more efficient matmuls |
| + comms overlap (prefetch) | 44% | 3,270 | all-gather hidden behind compute |

Forty-four percent MFU at 8 GPUs, ~26,000 tokens/second aggregate. That is a healthy single-node 7B run — in the same band as published FSDP and Megatron numbers on A100s — and it is the reference we will measure the cluster against. **The biggest single lever was the pair that kept the GPU busy: feeding it (the loader) and hiding its comms (overlap).** Everything else was incremental. Remember that ranking; it inverts at the next scale.

## Step 3: 64 GPUs, and why multi-node was slower per card

Eight nodes, 64 A100s, connected by an InfiniBand fabric. We keep the exact tuned FSDP config that hit 44% on one node, scale `--nnodes` to 8, and launch. The aggregate throughput goes up — of course it does, there are eight times as many GPUs — but the *per-GPU* rate collapses. Each card, which did 3,270 tokens/second inside one node, now does about 2,050. We are getting 64 GPUs' worth of hardware and 40 GPUs' worth of work.

The number that names this is **scaling efficiency**: the per-GPU throughput at scale divided by the per-GPU throughput of your reference configuration.

$$
\text{efficiency}(D) = \frac{T_D / D}{T_\text{ref} / D_\text{ref}}.
$$

Against the 8-GPU reference of 3,270 tokens/second per card, the naive 64-GPU run does $2{,}047 / 3{,}270 = 0.63$ — **63% scaling efficiency**, and MFU drops right back to 28%. We climbed all the way to 44% on one node and gave a third of it back the instant we crossed a node boundary. Wall number three, and it is the one that separates people who can run multi-node from people who can't.

### The interconnect is the whole story

The cause is physical, and it lives in the gap between two numbers. Inside a node, the eight GPUs are wired through an NVSwitch and exchange data over NVLink at roughly 600 GB/s of usable bandwidth per GPU. Between nodes, traffic crosses InfiniBand HDR at about 200 Gb/s — which is **25 GB/s**, a factor of roughly 24 slower. Every byte FSDP's all-gather moves within a node was nearly free; the same byte moved between nodes is 24× more expensive. The physics of this gap is the subject of [the interconnect physics](/blog/machine-learning/distributed-training/the-interconnect-physics); here is what it looks like across the cluster:

![a topology where a sixty four GPU cluster branches into node zero and node seven each with eight GPUs on an NVSwitch connected internally by fast NVLink and between each other by a far slower InfiniBand spine](/imgs/blogs/scaling-a-7b-llm-1-to-64-gpus-6.webp)

With FULL_SHARD across all 64 GPUs, the parameter shards for any given layer are scattered across all eight nodes. To materialize a layer, a GPU must all-gather pieces from GPUs on the *other seven nodes* — over InfiniBand. The all-gather that hid perfectly behind compute at 600 GB/s inside one node cannot hide behind the same compute at 25 GB/s across nodes. The communication comes out from behind the compute and stands exposed on the critical path, exactly the stall we spent Step 2 eliminating, reintroduced by geography.

#### Worked example: why the exposed all-gather costs so much

The full parameter set in bf16 is 14 GB. A ring all-gather moves $(D-1)/D \cdot S \approx S$ bytes per GPU, so reconstructing all the layers over a step moves on the order of 14 GB per GPU just for one gather (and FSDP does this in both forward and backward).

- **Confined to NVLink** (intra-node, 600 GB/s): $14\ \text{GB} / 600\ \text{GB/s} \approx 23\ \text{ms}$. Small enough to hide behind tens of ms of matmul.
- **Forced across InfiniBand** (25 GB/s): $14\ \text{GB} / 25\ \text{GB/s} \approx 560\ \text{ms}$. There is no per-step compute budget on a 7B model that hides half a second of gather. It stands exposed, and the GPU waits.

These are bandwidth-only lower bounds — they ignore latency, protocol overhead, and NCCL's hierarchical ring construction, so treat them as order-of-magnitude, not stopwatch. But the 24× ratio is real and it is the entire diagnosis: *the same collective, moved from NVLink to InfiniBand, went from hideable to fatal.*

### Confirming it with the profiler

Never fix a bottleneck you have only reasoned about — confirm it. `torch.profiler` with the right activities shows exactly where the step's wall-clock goes, and on the naive 64-GPU run it shows NCCL all-gather kernels sitting on the critical path with the compute stream idle behind them:

```python
import torch
from torch.profiler import profile, ProfilerActivity, schedule

prof_schedule = schedule(wait=1, warmup=2, active=3, repeat=1)
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=prof_schedule,
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./trace"),
    record_shapes=True,
    with_stack=True,
) as prof:
    for step, batch in enumerate(loader):
        train_step(model, batch)
        prof.step()
        if step >= 6:
            break

# In the trace: NCCL AllGather kernels on the critical path, compute stream idle
# behind them, and a per-step gap that lines up with the ~560 ms IB estimate.
```

The trace confirms the arithmetic: the gap between compute kernels lines up with the InfiniBand gather estimate. Now we fix it.

### The fix: shard within the node, replicate across nodes

The insight is that we do not have to shard across all 64 GPUs. We only *needed* sharding to fit the model, and the model fits comfortably in one node's worth of eight GPUs (14 GB per card). So we shard **within** each node and **replicate** across nodes — FSDP's `HYBRID_SHARD`. Now the parameter all-gather stays entirely on NVLink (fast, hideable), and the only traffic that crosses InfiniBand is the gradient **all-reduce** between the eight node-level replicas — which is a smaller, lower-cadence collective (you can accumulate gradients over several micro-batches before it fires) and it overlaps with the backward pass. This is exactly the data-parallel gradient all-reduce from [DDP from first principles](/blog/machine-learning/distributed-training/ddp-from-first-principles), now operating between nodes instead of between GPUs.

```python
from torch.distributed.fsdp import ShardingStrategy
# ...
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.HYBRID_SHARD,  # shard in node, replicate across
    mixed_precision=bf16,
    auto_wrap_policy=wrap_policy,
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    forward_prefetch=True,
    device_id=torch.cuda.current_device(),
)
```

Two more things make the InfiniBand path actually perform. First, tell NCCL which interface to use and confirm it picks the InfiniBand HCA rather than silently falling back to slow TCP over Ethernet — the single most common multi-node misconfiguration, and the subject of a whole later post on the NCCL log. Second, give the cross-node all-reduce bigger buckets so it sends fewer, larger messages and amortizes InfiniBand's per-message latency:

```bash
export NCCL_DEBUG=INFO             # print the rings/trees NCCL builds; confirm IB, not TCP
export NCCL_IB_HCA=mlx5            # use the InfiniBand cards
export NCCL_SOCKET_IFNAME=ib0      # bind the control plane to the IB interface
export NCCL_ALGO=Ring              # ring all-reduce for large bf16 gradient tensors
# In the FSDP/DDP wrap, a larger bucket cap batches gradients into fewer IB messages.
```

And the multi-node launch itself — `torchrun` with a rendezvous so all 64 processes find each other. Run this on every node with the same `--rdzv_endpoint` (or under SLURM's `srun`, which fills in the node rank for you):

```bash
torchrun \
  --nnodes=8 \
  --nproc_per_node=8 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=node0:29500 \
  --rdzv_id=7b_run \
  train.py --sharding hybrid --micro_batch 8 --seq_len 2048
```

The result: the all-gather goes back on NVLink, InfiniBand carries only the overlappable gradient all-reduce, and per-GPU throughput recovers from 2,050 to about 2,850 tokens/second — **87% scaling efficiency**, MFU back up to 38%.

![a two column comparison of full shard across sixty four GPUs with the all gather exposed on InfiniBand at sixty three percent efficiency versus hybrid shard keeping the gather on NVLink at eighty seven percent efficiency](/imgs/blogs/scaling-a-7b-llm-1-to-64-gpus-7.webp)

Note that 64-GPU MFU (38%) sits below the single-node 44%: crossing eight nodes is never free, and 87% scaling efficiency means we still pay a real 13% tax for the InfiniBand hop. But 87% is the difference between a cluster that earns its cost and one that wastes a third of it. **The biggest lever at 64 GPUs was not tuning the kernels — it was choosing where the communication happens.** At one node the lever was feeding the GPU; at eight nodes it is localizing the comms. That inversion is the lesson of the whole trip.

## Step 4: the global batch grew 64×, so does it still learn?

There is a wall here that has nothing to do with hardware, and skipping it means shipping a fast run that trains a worse model. Every time we added GPUs and grew the per-GPU batch, the **global batch** — the number of tokens whose gradients get averaged into one optimizer step — exploded. From one card at micro-batch 1 to 64 cards at micro-batch 8, with the same sequence length, the global batch grew by well over 64×. A gradient averaged over a huge batch is a lower-variance, more accurate estimate of the true gradient, which means the *same* learning rate now takes an over-cautious step. The optimizer under-shoots, and convergence per step slows even though throughput soared.

The classic correction is the **linear scaling rule**: when you multiply the global batch by $k$, multiply the base learning rate by $k$, and add a warmup of a few hundred steps so the large early steps don't destabilize the run before the statistics settle. The intuition is that $k$ times as much averaging supports a $k$ times larger step.

#### Worked example: scaling the learning rate from 8 to 64 GPUs

Say the 8-GPU run used base learning rate $3\times10^{-4}$ at a global batch of about 0.5M tokens. Moving to 64 GPUs with the same per-GPU batch multiplies the global batch by 8 to ~4M tokens. The linear rule says scale the learning rate by 8 as well — but a bare $8\times$ lands at $2.4\times10^{-3}$, which for a 7B transformer is well into the unstable range. This is the honest part: **the linear rule holds only up to a "critical batch size,"** beyond which returns diminish and stability breaks. In practice, past a few million tokens per batch, teams switch to square-root scaling ($\sqrt{k}$ instead of $k$) or simply tune the peak learning rate directly and lengthen warmup. A defensible choice here is to scale by $\sqrt{8} \approx 2.8$ to about $8\times10^{-4}$, with warmup stretched from 500 to ~2,000 steps. The compute-optimal framing of how batch and tokens trade off lives in [Chinchilla compute-optimal scaling](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling).

```python
import math
from torch.optim.lr_scheduler import LambdaLR

def warmup_then_cosine(step, warmup_steps, total_steps, peak_lr, min_lr):
    if step < warmup_steps:
        return peak_lr * step / max(1, warmup_steps)     # linear warmup
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))  # decay to min_lr
    return min_lr + (peak_lr - min_lr) * cosine

# 8-node run: sqrt-scaled peak LR, warmup stretched for the larger global batch.
peak_lr, min_lr = 8e-4, 8e-5
warmup, total = 2000, 100_000
scheduler = LambdaLR(
    optimizer,
    lr_lambda=lambda s: warmup_then_cosine(s, warmup, total, peak_lr, min_lr) / peak_lr,
)
```

The convergence check is non-negotiable: run the 64-GPU config for a few thousand steps and overlay its loss curve on the 8-GPU baseline at matched *tokens seen* (not matched steps — the big-batch run takes fewer steps to see the same tokens). If the curves track within noise, the larger batch is training the same model faster. If the big-batch curve sits persistently above the baseline, the batch has outrun its critical size and you back off the learning rate or the batch. Throughput you cannot converge is not throughput; it is a faster way to waste a cluster.

And the MFU computation you use to score all of this, dropped straight into the training loop:

```python
def mfu(tokens_per_sec, n_params, num_gpus, peak_flops_per_gpu=312e12):
    """Model FLOPs Utilization for a dense transformer (6*N FLOPs/token)."""
    model_flops = 6 * n_params * tokens_per_sec
    return model_flops / (num_gpus * peak_flops_per_gpu)

# steady-state, after torch.cuda.synchronize() and warmup:
print(f"MFU = {mfu(182_000, 7e9, 64):.1%}")   # -> 38%
```

## The scorecard: 1 → 8 → 64

Here is the whole journey in one table — the payoff of the trip. Read it top to bottom as a story: fit at all, feed the GPU, hide the comms, localize the comms, keep it converging.

| Stage | GPUs | Key configuration | Aggregate tok/s | Per-GPU tok/s | MFU | Scaling eff. | Biggest lever |
|---|---|---|---|---|---|---|---|
| 0 | 1 | 8-bit AdamW + full ckpt, mb=1 | ~1,500 | 1,500 | 20% | — (AdamW won't fit) | fit at all → 8-bit optimizer |
| 1 | 8 | FSDP FULL_SHARD, naive loader | ~16,600 | 2,080 | 28% | — | shard the state → FSDP |
| 2 | 8 | + loader + bf16 + ckpt + overlap | ~26,000 | 3,270 | 44% | 100% (ref) | feed & hide comms |
| 3 | 64 | FSDP FULL_SHARD, multi-node | ~131,000 | 2,050 | 28% | 63% | (regressed: exposed IB gather) |
| 4 | 64 | HYBRID_SHARD + buckets + IB tuned | ~182,000 | 2,850 | 38% | 87% | localize comms → HYBRID_SHARD |

Aggregate throughput went from 1,500 to 182,000 tokens/second — a 121× speedup on 64× the hardware, which is *more* than linear only because the single-GPU baseline was so crippled by not fitting. The number that matters is the honest one at the right, scaling efficiency: at the end, 64 GPUs deliver 87% of the per-card work that eight GPUs did. The walls, in order, were: **won't fit → GPU starved → comms exposed → batch too big to converge blindly.** Different lever at every wall. That ordering — memory, then feeding, then comms, then convergence — is the general shape of scaling almost any model, and it is the checklist the [distributed training playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook) makes reusable.

## Case studies: what MFU is actually achievable

Our numbers should sit inside the band that the field reports, and they do. A few anchors, cited so you can calibrate your own runs:

- **Megatron-LM tensor + pipeline parallelism** (Narayanan et al., 2021) reported sustained MFU in the roughly 45–52% range training GPT-scale models across thousands of A100s, using 3D parallelism tuned so that tensor-parallel all-reduces stay on NVLink within a node and pipeline/data-parallel traffic crosses the slower fabric — the same "keep the heavy collective on the fast link" principle that HYBRID_SHARD applies here.
- **PyTorch FSDP** has been reported by the PyTorch team fitting and training models from 13B up to 70B+ on A100 clusters at MFU in the high-30s to mid-40s percent, closely matching our 8-GPU 44% and 64-GPU 38% — FSDP is a data-parallel technique, so its MFU is very sensitive to exactly the interconnect exposure we diagnosed.
- **PaLM** (Chowdhery et al., 2022) is the high-water mark to respect, reporting ~46% MFU (and higher HFU) at 540B on TPU v4 — a reminder that even a world-class team with a custom interconnect lands in the mid-40s, so if your 7B run is at 44%, you are not leaving a fortune on the table; you are near the practical ceiling for dense data-parallel training.
- **NCCL bandwidth**, NVLink vs InfiniBand: NVIDIA's own microbenchmarks put NVLink-generation intra-node all-reduce in the hundreds of GB/s per GPU and InfiniBand HDR inter-node at ~25 GB/s, the ~20–24× gap that made our exposed all-gather a half-second stall. This is not a subtle effect; it is the dominant term in multi-node scaling.

The takeaway from the literature is the same as the takeaway from our run: **mid-40s MFU is a good dense-model result, low-40s at multi-node is very good, and anything under ~30% at scale is a signal that a collective is exposed on the slow link.**

## When to reach for each lever (and when not)

Every lever in this post is a cost, and the discipline is knowing when *not* to pull it.

| Lever | Reach for it when | Skip it when |
|---|---|---|
| FSDP FULL_SHARD | The model + optimizer state won't fit on one GPU | It fits — DDP is simpler and saturates NVLink |
| Activation checkpointing | Activation memory caps your batch size | You have memory headroom — recompute is wasted HFU |
| bf16 mixed precision | Always, on Ampere or newer | fp32-only hardware, or a model that needs fp16 loss scaling |
| Comms overlap (prefetch) | Always at scale — it is free MFU | Never skip; it is the highest-leverage single flag |
| HYBRID_SHARD | Multi-node, and the model fits within one node | Single node (no inter-node hop to protect), or the model is too big for one node's memory |
| Linear/√ LR scaling | The global batch grew materially | The batch is unchanged — don't touch a tuned LR |

The two decisions that trip people up most:

**Don't go multi-node until you've saturated one node.** A 63%-efficient 64-GPU run is *slower per dollar* than a 44%-MFU 8-GPU run. If your model fits in one node, prove you are at the single-node ceiling before you reach for a second node — most of the MFU you can win is inside one node, where the interconnect is free. Cross a node boundary only when the model or the batch genuinely demands it.

**Don't shard more than you must.** FULL_SHARD across 64 GPUs was strictly worse than HYBRID_SHARD for a model that fits in eight, because it dragged the parameter all-gather onto InfiniBand for no memory benefit. Shard exactly enough to fit, and replicate the rest. The corollary for genuinely large models — where even one node cannot hold the state — is that you *do* need FULL_SHARD across nodes (or tensor/pipeline parallelism), and then the interconnect tax is unavoidable and you plan the cluster around it.

## Stress-testing the result

The 87% number is not universal; it holds under the conditions we built for. Push on them:

- **On PCIe instead of NVLink**, the intra-node all-gather that took 23 ms would take ~5–10× longer, and even single-node overlap would struggle — HYBRID_SHARD's whole premise (fast intra-node link) weakens, and you would fall back to smaller shard groups or accept lower MFU.
- **When the batch is tiny** (memory-starved or a small model), the matmuls are too short to hide any comms behind, and even perfect overlap leaves the GPU stalling — the fix shifts from overlap to *growing* the batch, back to Step 2's lever.
- **When one node is a straggler** — a bad NIC, thermal throttling, an ECC-scrubbing GPU — the whole 64-GPU step waits for the slowest rank at every collective, and your 87% quietly becomes 60% with no config change at all. That failure has its own diagnosis (it is the subject of the straggler post in this track), and the first symptom is scaling efficiency that sags over hours, not at launch.
- **When the optimizer state won't fit even sharded across the whole cluster** — a much larger model — data parallelism alone runs out, and you compose tensor and pipeline parallelism on top, which is where 3D parallelism and the device mesh come in.

Each of those is a different wall, and each has a different lever. The method is always the same: symptom, hypothesis, measurement, fix, re-measure.

## Key takeaways

- **MFU is the score.** Tokens per second flatters big clusters; $\text{MFU} = 6\Psi T / (D P)$ tells you what fraction of the hardware you actually bought. Mid-40s is a good dense result; under 30% at scale means a collective is exposed.
- **The walls come in a fixed order**: won't fit → GPU starved → comms exposed → batch too big to converge. Fix them in that order; a later fix cannot help while an earlier wall stands.
- **The biggest lever changes with scale.** At one node it is feeding the GPU and hiding its comms. At eight nodes it is *where* the comms happen — keep the heavy collective on NVLink, send only the overlappable one over InfiniBand.
- **Shard exactly enough to fit, then replicate.** FULL_SHARD across nodes for a model that fits in one node is a self-inflicted InfiniBand tax; HYBRID_SHARD recovered 24 points of scaling efficiency for free.
- **Measure honestly**: warm up, `torch.cuda.synchronize()` before timing, steady state only, and rule out the data loader before you blame the model.
- **Overlap is never optional at scale** — the prefetch flags are the single highest-leverage line of config in the whole run.
- **Throughput you cannot converge is not throughput.** When the global batch grows, scale the learning rate, lengthen warmup, and check the loss curve at matched tokens seen.
- **Don't go multi-node until you've saturated one node** — most of your MFU lives where the interconnect is free.

## Further reading

- [Why distributed training](/blog/machine-learning/distributed-training/why-distributed-training) — the four walls this whole journey climbs, and the map of the series.
- [The memory budget](/blog/machine-learning/distributed-training/the-memory-budget) — the $16\Psi$ derivation behind Step 0's OOM.
- [FSDP in practice](/blog/machine-learning/distributed-training/fsdp-in-practice) — sharding strategies, wrap policies, and the FSDP2 API in depth.
- [Overlapping compute and communication](/blog/machine-learning/distributed-training/overlapping-compute-and-communication) — the prefetch mechanism that hid the all-gather.
- [The interconnect physics](/blog/machine-learning/distributed-training/the-interconnect-physics) — why NVLink and InfiniBand differ by 24×, and why placement decides scaling.
- [The data pipeline at scale](/blog/machine-learning/distributed-training/the-data-pipeline-at-scale) — the loader confound that cost the first five points of MFU.
- [The distributed training playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook) — the capstone checklist that turns this trip into a repeatable procedure.
- Megatron-LM (Narayanan et al., 2021), *Efficient Large-Scale Language Model Training on GPU Clusters* — the tensor/pipeline MFU results and the "keep the heavy collective on the fast link" principle.
- PyTorch FSDP documentation and the *Getting Started with FSDP* tutorial — the reference for `ShardingStrategy`, `MixedPrecision`, and `HYBRID_SHARD`.
