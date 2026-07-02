---
title: "DeepSpeed ZeRO and Offload: Training Models Bigger Than Your GPU Memory"
date: "2026-07-02"
publishDate: "2026-07-02"
description: "Use DeepSpeed's ds_config.json to shard optimizer state with ZeRO and offload it to CPU RAM or NVMe, so a single 24GB GPU can train a 13B model — and know exactly what throughput you pay for the privilege."
tags:
  [
    "distributed-training",
    "multi-gpu",
    "deepspeed",
    "zero",
    "offload",
    "fsdp",
    "pytorch",
    "gpu",
    "ml-systems",
    "deep-learning",
  ]
category: "machine-learning"
subcategory: "Distributed Training"
author: "Hiep Tran"
featured: true
readTime: 41
---

You have a 13-billion-parameter model and a single GPU with 24 GB of memory. You do the arithmetic once and it is grim: the weights alone in fp16 are 26 GB, already more than the card holds, and that is before a single gradient, before the Adam optimizer's momentum and variance buffers, before the fp32 master copy of the weights, before one activation. The honest total to train this model with Adam is about 182 GB. Your GPU has 24. By every reasonable measure, this model cannot be trained on this hardware. And yet — with one JSON file and about ten lines of Python — it can. It will be slow, but it will train, and it will converge to exactly the same loss it would on a cluster.

That JSON file is a DeepSpeed `ds_config.json`, and the trick it pulls is the subject of this post. DeepSpeed implements the same ZeRO memory-sharding math that PyTorch's FSDP does, but it leans hard into one capability that FSDP treats as an afterthought: **offload**. ZeRO-Offload pushes the fp32 optimizer states and the master weights out of GPU memory and into CPU RAM, and lets the CPU run the Adam step. ZeRO-Infinity goes one tier further and streams parameters and optimizer state to NVMe SSD. The result is a memory hierarchy — fast-and-small GPU HBM on top, big-and-slow NVMe at the bottom — that you can spill state down when you run out of room up top.

![A vertical stack of four memory tiers from GPU HBM down to NVMe showing capacity rising and bandwidth falling at each step](/imgs/blogs/deepspeed-zero-and-offload-1.webp)

The figure above is the picture to hold for the whole post. Each tier down is bigger and slower. Offload is the art of deciding which slice of your training state lives on which tier — and paying, in throughput, for every gigabyte you push downhill. This is the tenth post in the *Distributed Training in the Trenches* series. It assumes you already know [why we distribute at all](/blog/machine-learning/distributed-training/why-distributed-training) and, critically, the [ZeRO memory model — how sharding params, gradients, and optimizer state saves memory](/blog/machine-learning/distributed-training/zero-and-fsdp-the-memory-model). This post is DeepSpeed's specific take on that model, plus the offload superpower, plus the loud warning that comes with it: offload buys capacity, never speed, and if your model already fits on the GPUs you have, turning it on just makes your training slower for nothing.

By the end you will be able to read and write a real `ds_config.json`, pick the right ZeRO stage for your situation, turn on CPU or NVMe offload deliberately, launch the job with `deepspeed`, and — most important — estimate ahead of time both whether a model will fit and roughly how much throughput the offload will cost you.

## The same ZeRO math, a different door

Let us refresh the memory arithmetic, because every decision in a `ds_config.json` traces back to it. For a model with $\Psi$ parameters trained in mixed precision with the Adam optimizer, the state you must hold is:

- **fp16 parameters**: $2\Psi$ bytes (2 bytes per parameter).
- **fp16 gradients**: $2\Psi$ bytes.
- **fp32 optimizer state**: the fp32 master copy of the weights (4 bytes), plus Adam's fp32 momentum (4 bytes) and fp32 variance (4 bytes), for $12\Psi$ bytes total.

Add those and you get the famous coefficient: $(2 + 2 + 12)\Psi = 16\Psi$ bytes just for model and optimizer state, before activations. For a 7.5B model that is $16 \times 7.5 \times 10^9 \approx 120$ GB — which will not fit on any single GPU made today. For the 13B model in the opening, it is $16 \times 13 \times 10^9 \approx 208$ GB of state, and with the fp16 parameters counted once that headline "182 GB to train" is the same order of magnitude.

ZeRO's insight, covered in depth in [the memory-model post](/blog/machine-learning/distributed-training/zero-and-fsdp-the-memory-model), is that in plain data parallelism every one of your $N_d$ GPUs holds a *full, redundant copy* of all $16\Psi$ bytes. That is enormously wasteful. ZeRO shards the state across ranks so each GPU holds only $1/N_d$ of it, gathering the pieces on demand. It comes in three stages, each sharding one more slice:

- **ZeRO-1** ($P_{os}$): shard the optimizer state ($12\Psi$). Per-GPU memory becomes $4\Psi + 12\Psi/N_d$.
- **ZeRO-2** ($P_{os+g}$): also shard the gradients ($2\Psi$). Per-GPU becomes $2\Psi + 14\Psi/N_d$.
- **ZeRO-3** ($P_{os+g+p}$): also shard the parameters ($2\Psi$). Per-GPU becomes $16\Psi/N_d$ — it falls almost linearly with GPU count.

The ZeRO paper's worked numbers for a 7.5B model on 64 GPUs are the ones to memorize: baseline data parallelism needs 120 GB per GPU; ZeRO-1 brings that to 31.4 GB; ZeRO-2 to 16.6 GB; ZeRO-3 to 1.9 GB. That is a 64× reduction from full ZeRO-3 sharding, at the cost of extra all-gather communication to reconstruct parameters during the forward and backward passes.

DeepSpeed and FSDP implement exactly this. The difference is the *door* you walk through. In FSDP you express sharding in Python code — you wrap modules, pass a `ShardingStrategy` enum, and set a `MixedPrecision` policy, as covered in [FSDP in practice](/blog/machine-learning/distributed-training/fsdp-in-practice). In DeepSpeed you express it in a JSON config file that your training script barely has to know about. Same math; a config-file interface instead of a code interface. And DeepSpeed adds the fourth column that FSDP under-sells: the ability to move that sharded state *off the GPU entirely*, onto CPU RAM or NVMe.

![A four by four matrix showing which of params gradients and optimizer each ZeRO stage shards and the resulting per GPU memory](/imgs/blogs/deepspeed-zero-and-offload-2.webp)

The matrix above lays out what each stage shards and where offload takes the last slice. Read it as the decision surface for the rest of the post: as you move down the rows, more state is sharded and per-GPU memory falls, until the bottom row moves the optimizer off the GPU altogether — the step that lets a single card train a 10B-plus model.

## The ds_config.json is the interface

Here is the thing that surprises people coming from FSDP: with DeepSpeed, your training loop is almost provider-agnostic, and *the config file carries the intelligence*. You write a plain training script, hand DeepSpeed a JSON file, and the same script runs on one GPU or a 64-GPU cluster depending on what the JSON says. So the JSON is where we start.

This is a complete, real ZeRO-3 configuration with CPU offload turned on — the exact file that makes the 13B-on-24GB trick work:

```json
{
  "train_batch_size": 32,
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 8,
  "gradient_clipping": 1.0,
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6,
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 1e-4,
      "betas": [0.9, 0.95],
      "eps": 1e-8,
      "weight_decay": 0.1
    }
  },
  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "warmup_num_steps": 1000,
      "total_num_steps": 100000
    }
  },
  "steps_per_print": 50,
  "wall_clock_breakdown": false
}
```

Every field here earns its keep. Let us walk the ones that matter, because these are the knobs you will actually turn.

**The batch-size trio.** `train_batch_size` is the effective global batch across all GPUs and all accumulation steps. It must equal `train_micro_batch_size_per_gpu` × `gradient_accumulation_steps` × number of GPUs. DeepSpeed will error at startup if the three do not multiply out, which is a feature — it catches the classic "my effective batch is 8× what I thought" bug before it wastes a run. Set any two and let DeepSpeed infer the third.

**Precision.** `"bf16": {"enabled": true}` trains in bfloat16, which on A100 and H100 is the right default — it has the same exponent range as fp32, so it does not need the loss-scaling dance that fp16 requires. If you are on older V100 hardware without bf16, you switch this block to `"fp16": {"enabled": true, "loss_scale": 0, "initial_scale_power": 16}`, and DeepSpeed manages dynamic loss scaling for you.

**The `zero_optimization` block** is the whole point. `"stage": 3` selects full ZeRO-3 sharding. The two `offload_*` blocks are the offload switches, discussed at length in the next section. The rest are tuning knobs, and they deserve their own table because getting them wrong is the difference between offload that overlaps cleanly and offload that stalls the GPU on every step.

| Knob | What it controls | Practical guidance |
|---|---|---|
| `reduce_bucket_size` | Bytes of gradients coalesced before a reduce-scatter | Larger = fewer, bigger collectives = better bus utilization; `5e8` (500M) is a sane default. Too large wastes memory. |
| `allgather_bucket_size` | (ZeRO-1/2) Bytes gathered per all-gather of params | Same idea for stages 1 and 2; `5e8` is standard. |
| `overlap_comm` | Overlap the reduce-scatter and all-gather with backward compute | Almost always `true` — this is what hides comms behind compute. |
| `contiguous_gradients` | Copy grads into one contiguous buffer to avoid fragmentation | `true` for large models; prevents memory spikes and speeds the reduce. |
| `stage3_prefetch_bucket_size` | Bytes of the *next* layer's params to all-gather ahead of time | Prefetch hides the gather latency; set near `reduce_bucket_size`. Too big spikes memory. |
| `stage3_param_persistence_threshold` | Params smaller than this stay resident (never sharded) | Small tensors — LayerNorm weights, biases — cost more to shard than to keep. `1e6` keeps sub-million-param tensors put. |
| `stage3_max_live_parameters` | Cap on params gathered (un-sharded) at once | Bounds peak memory during forward/backward; lower it if you still OOM. |

The intuition for these numbers: ZeRO-3 is a constant dance of *gather a layer's parameters just before you need them, use them, then free the gathered copy*. The bucket and prefetch sizes control how far ahead the gather runs and how big each chunk is. Tuned well, the gather for layer $L+1$ happens while the GPU computes layer $L$, and you never see the latency. Tuned badly, the GPU finishes layer $L$ and then sits idle waiting for layer $L+1$'s parameters to arrive. That idle time is pure waste, and the profiling post in this series shows you how to see it.

A note on ergonomics: you do not have to write these numbers from scratch. Set them to `"auto"` and, when you launch through Hugging Face's `Trainer` or `accelerate`, the framework fills in values derived from your model's hidden size. For raw DeepSpeed, the defaults above are a good starting point that you tune only if profiling shows a comms stall.

### The stage decision, in one paragraph

Which stage do you pick? The rule is: **use the lowest stage that fits, because each stage adds communication.** ZeRO-1 shards only optimizer state and adds essentially no communication over plain data parallelism — it is nearly free, and you should basically always use at least ZeRO-1. ZeRO-2 also shards gradients and still adds no extra communication beyond the reduce-scatter you were already doing. ZeRO-3 shards parameters too, which forces an all-gather of each layer's weights on every forward *and* every backward — roughly a 1.5× increase in communication volume — so you reach for it only when ZeRO-2 does not fit. Offload sits below all of them: turn it on only when even ZeRO-3 across all your GPUs does not fit.

### The communication ZeRO-3 adds, derived

The "roughly 1.5×" is not a hand-wave; it falls out of counting bytes. In plain data parallelism, the only collective is a gradient all-reduce at the end of the backward pass, which — implemented as reduce-scatter followed by all-gather over a ring — moves about $2\Psi$ bytes of traffic per GPU (the ring all-reduce cost of $2(N-1)/N \cdot S$ approaches ${2S}$ as $N$ grows, with $S = 2\Psi$ bytes of fp16 gradients). Call that the baseline: one unit of $2\Psi$.

ZeRO-1 and ZeRO-2 do the exact same total data movement, just split into a reduce-scatter (each GPU ends up owning its shard of the reduced gradients, $\approx\Psi$ of traffic) and, for the parameter update, they need no gather because each rank updates only its own shard. So stages 1 and 2 preserve the $2\Psi$ communication volume of the baseline — this is why they are "free."

ZeRO-3 is different because the parameters themselves are sharded, so no single GPU has a whole layer's weights sitting in memory. To run the forward pass, each GPU must **all-gather** the full parameters of a layer just before computing it (that gather is $\approx\Psi$ of traffic across the pass), then free them. The backward pass needs the parameters again to compute gradients, so it all-gathers them a *second* time ($\approx\Psi$ more), and finishes with the reduce-scatter of gradients ($\approx\Psi$). Sum it: two forward-and-backward gathers plus the gradient reduce-scatter give roughly $3\Psi$ against the baseline's $2\Psi$ — the 1.5× you were promised. That extra gather is real bandwidth, and it is why ZeRO-3 wants a fast interconnect (NVLink, not PCIe) between GPUs to stay efficient. Offload then rides *on top* of this: the sharded state that ZeRO-3 keeps on each GPU is what offload relocates to CPU or NVMe, adding the PCIe traffic derived later to the NVLink traffic derived here.

## Offload: the superpower FSDP under-sells

Now the headline. Sharding across $N_d$ GPUs divides your state by $N_d$ — but if $N_d = 1$, or if even the sharded state is too big for the GPUs you have, sharding has done all it can. Offload attacks the problem from a different direction: it moves state off the GPU entirely, onto a bigger, slower tier.

**ZeRO-Offload** targets the single most memory-hungry slice: the fp32 optimizer state and master weights, that $12\Psi$-byte block. It parks them in CPU RAM. And here is the clever part — it does not just *store* them there and shuttle them back to the GPU for the update. It runs the entire Adam optimizer step *on the CPU*. The GPU computes the forward and backward and produces fp16 gradients; those gradients are copied over PCIe to the CPU; the CPU, using a hand-optimized multi-threaded Adam implementation, updates the fp32 master weights in place; the updated weights are cast to fp16 and copied back to the GPU for the next forward pass.

![A dataflow diagram showing gradients crossing PCIe to a CPU Adam step that reads fp32 optimizer state and returns fp16 parameters to the GPU](/imgs/blogs/deepspeed-zero-and-offload-3.webp)

The figure traces one training step with optimizer offload. The way this works is a careful division of labor drawn from the compute characteristics of each device. The forward and backward passes are dense matrix multiplies — throughput-bound work the GPU is built for, so they stay on the GPU. The Adam update is a memory-bound, element-wise operation over the optimizer state — it has almost no arithmetic intensity, so running it on the CPU (which sits right next to that huge, cheap RAM) barely hurts, and it frees a massive chunk of HBM. The GPU never has to hold the $12\Psi$-byte optimizer state at all. For a 13B model that is 156 GB of memory that simply leaves the GPU.

**ZeRO-Infinity** goes one tier deeper. In addition to putting the optimizer on CPU RAM, it can stream the *parameters* themselves — and, if RAM is not enough, the optimizer state too — to NVMe SSD. The parameters live on disk; when the forward pass reaches layer $L$, ZeRO-Infinity reads layer $L$'s parameters from NVMe (or CPU RAM) into GPU HBM just in time, uses them, and evicts them. This is what lets a single node fit a model whose state runs into the hundreds of gigabytes or terabytes — the model's parameters never all reside on the GPU, or even in RAM, at once. The config change is a one-word edit: `"device": "nvme"` with an `nvme_path`.

Here is the NVMe-offload variant of the config. Note the two `_config` blocks that tell DeepSpeed which SSD to use and how to size the asynchronous I/O:

```json
{
  "train_micro_batch_size_per_gpu": 2,
  "gradient_accumulation_steps": 16,
  "bf16": { "enabled": true },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "nvme",
      "nvme_path": "/local_nvme/deepspeed_offload",
      "pin_memory": true,
      "buffer_count": 4,
      "fast_init": false
    },
    "offload_param": {
      "device": "nvme",
      "nvme_path": "/local_nvme/deepspeed_offload",
      "pin_memory": true,
      "buffer_count": 5,
      "buffer_size": 1e9
    },
    "aio": {
      "block_size": 1048576,
      "queue_depth": 8,
      "thread_count": 1,
      "single_submit": false,
      "overlap_events": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "stage3_max_live_parameters": 1e9,
    "stage3_prefetch_bucket_size": 5e8
  }
}
```

The `aio` block configures libaio-based asynchronous disk I/O — `queue_depth`, `block_size`, and `thread_count` should be tuned to your specific SSD, and DeepSpeed ships an `aio_bench` utility to find good values. The critical, non-negotiable detail: `nvme_path` **must point at a real, local NVMe drive**. Point it at a network filesystem or a spinning disk and your training will grind to a crawl that makes CPU offload look instantaneous.

### The mechanism: why offload costs throughput, derived

Offload's cost is not vague. It is a bandwidth calculation you can do on the back of an envelope, and doing it is the single most useful skill in this post.

Consider ZeRO-Offload with the optimizer on CPU. Every training step, two transfers cross the PCIe bus: the fp16 gradients go from GPU to CPU (that is $2\Psi$ bytes), and the updated fp16 parameters come back from CPU to GPU (another $2\Psi$ bytes). So per step you move $4\Psi$ bytes across PCIe. A PCIe 4.0 x16 link runs at roughly 25 GB/s in each direction in practice (its 32 GB/s theoretical peak minus overhead). The time to move that data is therefore:

$$T_\text{pcie} = \frac{4\Psi}{B_\text{pcie}} = \frac{4 \times 13 \times 10^9 \text{ bytes}}{25 \times 10^9 \text{ bytes/s}} \approx 2.1 \text{ s}$$

for a 13B model. On top of that, the CPU has to actually run the Adam update over 13 billion parameters — a memory-bound sweep over the $12\Psi = 156$ GB of optimizer state — which on a fast multi-socket server takes a further second or more. So the offload path costs on the order of 2–4 seconds per step.

Now the redeeming subtlety, and it is why offload is usable at all: **these transfers and the CPU compute overlap with the GPU's forward and backward.** DeepSpeed uses a one-step-delayed parameter update and pinned-memory asynchronous copies so that while the GPU is busy computing step $t+1$'s forward and backward, the CPU is simultaneously finishing step $t$'s optimizer update and shuttling data over PCIe. The effective per-step time is therefore not the *sum* of GPU compute and offload cost, but roughly the *maximum* of the two:

$$T_\text{step} \approx \max(T_\text{gpu}, \; T_\text{pcie} + T_\text{cpu-adam})$$

This is the whole game. If your GPU compute per step already takes longer than the offload path — because you have a big micro-batch, a long sequence, lots of activation recomputation — then the offload is completely hidden and you pay *nothing* for it. If your GPU compute is fast relative to the offload path — small batch, short sequences — then the PCIe-plus-CPU path is the bottleneck and your throughput collapses to whatever that bus can sustain. The design implication is direct and counterintuitive: **when you offload, you want a large micro-batch**, because a larger micro-batch increases $T_\text{gpu}$ and gives the offload path more compute to hide behind. This is the opposite of what memory pressure usually pushes you toward, and it is why offload configs often pair a big micro-batch with aggressive activation checkpointing.

For ZeRO-Infinity on NVMe the arithmetic is far harsher. Now the parameters *and* optimizer state stream from a ~6 GB/s SSD rather than a ~25 GB/s bus, and you are reading and writing the full $12\Psi$-plus of optimizer state from disk each step. Moving 156 GB of optimizer state over a 6 GB/s NVMe drive is ~26 seconds of I/O — an order of magnitude slower than the PCIe path. NVMe offload is the tier you reach for when the answer to "can I fit this at all?" is otherwise no, and you have made peace with a step time measured in tens of seconds.

### Is offload numerically correct?

A fair worry: if the CPU runs the optimizer step and the update overlaps with the next forward pass, are you training on *stale* parameters? Does offload change the loss curve? The answer is that ZeRO-Offload is designed to be numerically identical to non-offloaded Adam, and the overlap is arranged so that it stays that way.

The subtlety is the ordering. DeepSpeed does not let the forward pass of step $t+1$ read parameters that step $t$'s optimizer has not finished updating — that would be a genuine correctness bug. Instead, the overlap it exploits is between the CPU's Adam computation for the *current* step and the GPU's gradient computation, arranged with a one-step pipelining that respects data dependencies. The gradients for step $t$ are copied to CPU as they are produced during the backward pass (not all at once at the end), so the CPU starts working on early-layer gradients while the GPU is still computing late-layer ones. The parameter update completes before those parameters are needed again. Where a true one-step *delayed* update is used as an optimization for extra overlap, it is an opt-in mode with a documented, bounded effect on convergence — not the default. The default, plain CPU offload, produces the same weights bit-for-similar as on-GPU Adam, up to the ordinary non-associativity of floating-point summation that you already accept in any distributed reduction.

Two practical correctness notes follow. First, **pin your host memory** (`"pin_memory": true`, as in the configs above). Pinned (page-locked) CPU memory lets the GPU DMA data across PCIe without the CPU staging it through pageable memory first, which both speeds the copy and makes the asynchronous overlap actually asynchronous. Unpinned offload can be 2–3× slower and can serialize where you expected overlap. Second, the CPU Adam is DeepSpeed's own hand-written, AVX-vectorized, multi-threaded implementation (`DeepSpeedCPUAdam`), not a naive Python loop — it is what keeps $T_\text{cpu-adam}$ down to the "a second or two" figure rather than minutes. If you see the CPU step dominating, check that this fused optimizer is actually being used and that your host has enough cores to feed it.

### Offload and activations must work together

There is a memory category offload does *not* touch by default, and forgetting it is how people OOM even with aggressive offload turned on: **activations**. The $16\Psi$ state — params, grads, optimizer — is what ZeRO and offload manage, but the activations you save during the forward pass to compute gradients in the backward pass are a separate budget, and for large batches and long sequences they can be the dominant consumer of HBM. Offloading the optimizer to CPU frees 156 GB of state, and then the activations of a big micro-batch quietly refill the card.

This is why offload and **activation checkpointing** are not independent choices — they are two halves of one plan. Activation checkpointing (recomputation) discards most activations during the forward pass and recomputes them on the fly during the backward pass, trading roughly 30% more compute for a large drop in activation memory. DeepSpeed exposes it through `deepspeed.checkpointing` (or you use `torch.utils.checkpoint` in your model), and it composes directly with offload: checkpointing frees the HBM that offload's larger micro-batch wants to use.

The interaction is a virtuous loop when you set it up deliberately. Offload wants a *large* micro-batch, because a larger micro-batch grows $T_\text{gpu}$ and hides the PCIe-and-CPU path behind it. But a large micro-batch needs a lot of activation memory. Activation checkpointing supplies that memory by recomputing instead of storing. So the canonical offload recipe is: ZeRO-3 to shard, offload to relocate the optimizer, activation checkpointing to shrink the activation footprint, and a micro-batch sized as large as the freed HBM allows — each piece enabling the next. If you turn on offload and leave the micro-batch tiny and checkpointing off, you get the worst of both worlds: an exposed, unhidden offload path *and* wasted HBM. DeepSpeed can even offload the checkpointed activations themselves to CPU (`"cpu_checkpointing": true`) for the most extreme cases, though that adds yet more PCIe traffic and should be a last resort.

## Launching a DeepSpeed job

The training loop is refreshingly boring, and that is the point. You wrap your model with `deepspeed.initialize()`, which returns a `model_engine` that has swallowed your optimizer, your ZeRO sharding, your offload, and your gradient accumulation. Then you call `.backward()` and `.step()` on the engine instead of on loss and optimizer directly.

```python
import argparse
import deepspeed
import torch
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)  # adds --deepspeed, --deepspeed_config
    return parser.parse_args()

def main():
    args = parse_args()
    model = build_model()          # your plain nn.Module, un-sharded
    train_ds = build_dataset()

    # deepspeed.initialize reads ds_config.json and returns a wrapped engine.
    # It builds the optimizer, applies ZeRO sharding, and sets up offload.
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
    )

    # The engine knows its own micro-batch; a standard DataLoader is fine.
    loader = DataLoader(train_ds, batch_size=model_engine.train_micro_batch_size_per_gpu())

    device = model_engine.local_rank  # cuda device index for this rank
    model_engine.train()
    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model_engine(input_ids, labels=labels)
        loss = outputs.loss

        model_engine.backward(loss)   # ZeRO reduce-scatter + (offloaded) grad copy
        model_engine.step()           # (offloaded) Adam step + param update

        if step % 50 == 0 and model_engine.global_rank == 0:
            print(f"step {step} loss {loss.item():.4f}")

if __name__ == "__main__":
    main()
```

Three things to notice. First, `model` going into `deepspeed.initialize` is your ordinary, un-sharded `nn.Module` — DeepSpeed does the sharding internally, and for ZeRO-3 you should ideally construct the model under a `deepspeed.zero.Init()` context so the parameters are never fully materialized on one device even at construction time. Second, `model_engine.backward(loss)` replaces `loss.backward()`; it is where the gradient reduce-scatter and, under offload, the gradient copy to CPU happen. Third, `model_engine.step()` replaces `optimizer.step()` and `optimizer.zero_grad()` — it performs the (possibly CPU-side) Adam update and clears gradients. Gradient accumulation is automatic: the engine tracks micro-steps and only performs the real optimizer step every `gradient_accumulation_steps` micro-batches.

The launch is a single command. DeepSpeed ships its own launcher that plays the role `torchrun` plays for vanilla PyTorch:

```bash
# Single node, 8 GPUs:
deepspeed --num_gpus=8 train.py \
  --deepspeed \
  --deepspeed_config ds_config.json

# The 13B-on-one-GPU case — a single card:
deepspeed --num_gpus=1 train.py \
  --deepspeed \
  --deepspeed_config ds_config_cpu_offload.json

# Multi-node, 2 nodes of 8 GPUs, via a hostfile:
deepspeed --hostfile=hostfile.txt --num_nodes=2 --num_gpus=8 train.py \
  --deepspeed \
  --deepspeed_config ds_config.json
```

The `--deepspeed` flag and `--deepspeed_config` path are what `deepspeed.add_config_arguments` added to your parser. For multi-node, `hostfile.txt` lists your nodes and their slot counts (`node1 slots=8`), and DeepSpeed uses `pdsh` or SSH to launch the ranks — the same rendezvous story as `torchrun`, wearing a different coat. You can also launch DeepSpeed under `torchrun` or SLURM's `srun` if you prefer to keep one launcher across frameworks; the config file is unchanged.

### Knowing before you launch: the memory estimator

The best time to discover a model does not fit is *before* you burn an hour of GPU time watching it OOM. DeepSpeed ships an estimator that reports, for each offload configuration, how much GPU and CPU memory ZeRO-3 will need:

```python
from deepspeed.runtime.zero.stage3 import (
    estimate_zero3_model_states_mem_needs_all_live,
)
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf")

# Report per-GPU and CPU memory for 1 GPU, no offload / CPU offload / etc.
estimate_zero3_model_states_mem_needs_all_live(
    model,
    num_gpus_per_node=1,
    num_nodes=1,
)
```

This prints a small table: how much GPU memory and CPU memory ZeRO-3 needs with no offload, with CPU offload of the optimizer, and with CPU offload of both optimizer and parameters. Run it, read the row that fits your hardware, and set your config accordingly. There is an equivalent `estimate_zero2_model_states_mem_needs_all_live` for stage 2. Ten seconds of estimation saves an hour of trial and error.

#### Worked example: a 13B model on a single 24GB GPU

Let us make the opening concrete with real numbers. The hardware is one NVIDIA RTX 4090-class consumer card: 24 GB of GDDR6X, no NVLink, connected to the host over PCIe 4.0 x16. The host has 256 GB of DDR5 RAM. The model is a 13B-parameter LLaMA-style transformer. We train with AdamW in bf16.

**The budget without offload.** In fp16/bf16 the parameters are $2 \times 13\text{B} = 26$ GB — already past the 24 GB card. Gradients add another 26 GB. The Adam optimizer state (fp32 master + momentum + variance) adds $12 \times 13\text{B} = 156$ GB. Total model-plus-optimizer state: 208 GB, against 24 GB of HBM. It is not close. On a single GPU, sharding does nothing for you — there is no second rank to shard against. Offload is the only lever.

**The budget with ZeRO-3 + full CPU offload.** Now we push the optimizer state (156 GB) and the master weights and even the parameters into the 256 GB of host RAM. What stays on the GPU is: the currently-active layer's parameters gathered for compute, the gradients for that layer, the activations for the micro-batch, and DeepSpeed's working buffers. With `stage3_max_live_parameters` capping how many parameters are un-sharded at once and activation checkpointing on, that footprint fits in roughly 18–22 GB of the 24 GB card. The 208 GB of state lives in CPU RAM; the GPU holds only a rolling window of it.

![A before and after comparison showing naive fp16 training needing 182 GB and failing versus ZeRO-3 with CPU offload fitting in 20 GB](/imgs/blogs/deepspeed-zero-and-offload-4.webp)

The figure contrasts the two budgets: the naive path cannot even load the model, while the offload path streams state through a 24 GB window. This is not a trick or an approximation — the model trains to the same loss it would on a DGX. What you pay is throughput.

**The throughput you pay.** Per step, ZeRO-3 with parameter offload must, in the worst case, stream all 26 GB of fp16 parameters from CPU to GPU (once for forward, and the gathered params can be reused into backward with a good `max_reuse_distance`), copy gradients back, and let the CPU run Adam over 156 GB of state. The PCIe path alone, moving on the order of $4\Psi = 52$ GB of gradients-and-params traffic, is ~2 seconds; the CPU Adam step is another second or two; parameter streaming for forward adds more. In practice, expect a step time of several seconds and single-digit MFU. Concretely, the ZeRO-Offload paper reports training a 13B model on a single 32 GB V100 at roughly 40 TFLOPS — on an A100 with 312 bf16 TFLOPS of peak, 40 TFLOPS is about 13% MFU. That is slow. But 13% of a GPU you can afford beats 0% of a cluster you cannot. The model trains overnight instead of never.

The lesson of this example is the whole thesis of the post in miniature: offload converts an *impossible* run into a *slow* run. If slow-but-possible is what you need — you are a researcher with one card, or you are fine-tuning something once — it is a spectacular deal. If you have eight cards that would fit the model sharded, offload is throwing away throughput for no reason.

## The throughput you pay, quantified

Let us put numbers on the tiers so the trade-off is not hand-waved. The table below is a controlled comparison: the *same* 6.7B model, trained on the same 8× A100 80GB node, under four configurations. The point is to isolate what each choice costs. I have normalized throughput to the no-offload ZeRO-3 baseline and flagged the absolute figures as approximate order-of-magnitude — the exact numbers depend on sequence length, micro-batch, and your PCIe generation, and you should measure your own.

| Config | Fits? | Per-GPU HBM | Rel. throughput | Bottleneck |
|---|---|---|---|---|
| ZeRO-3, no offload | yes (13 GB/GPU) | ~13 GB | **1.0×** | HBM / NVLink comms |
| ZeRO-3 + CPU optimizer offload | yes | ~9 GB | ~0.6–0.8× | PCIe + CPU Adam |
| ZeRO-3 + CPU param & optim offload | yes | ~6 GB | ~0.4–0.6× | PCIe (param streaming) |
| ZeRO-Infinity (NVMe offload) | yes | ~4 GB | ~0.15–0.3× | NVMe bandwidth |

![A vertical stack of throughput tiers from no offload at full speed down to NVMe offload at a quarter speed](/imgs/blogs/deepspeed-zero-and-offload-5.webp)

The stack figure is the same story as the table, drawn as a descent. Read the first two rows together and the headline jumps out: **on this hardware, offload is pure loss.** The 6.7B model fits comfortably in 13 GB per GPU with plain ZeRO-3 sharding — there is nothing to offload. Turning on CPU offload here does not fit anything new; it just drags 20–40% of your throughput onto the PCIe bus for no reason. This is the single most common offload mistake: someone copies an offload config from a tutorial written for a smaller GPU, runs it on a big cluster, and wonders why their expensive A100s are running at 60% speed. The fix is to *delete the offload block*.

The bottom two rows are where offload earns its place: they fit models that the top row cannot. If your model needs 40 GB per GPU of optimizer state that you do not have, CPU offload's 0.5× throughput is infinitely better than not running. And if it needs more RAM than you have, NVMe's 0.2× is the only door left. The rule crystallizes: **offload is a capacity lever, never a speed lever. Turn it on when — and only when — the model does not fit without it.**

### How to measure this honestly

If you are going to report a throughput number, measure it correctly, because offload makes several measurement traps worse. Warm up for a dozen steps before timing anything — the first steps pay one-time costs (CUDA kernel compilation, NVMe file allocation, pinned-buffer setup) that are not representative. Call `torch.cuda.synchronize()` before you read the clock, because the GPU runs asynchronously and an un-synchronized timer measures Python, not compute. Measure steady-state, not the first epoch. And watch the data loader: under offload your step is slow, which paradoxically makes it *easy* to hide a slow data loader — but the moment you fix the offload bottleneck, a starved loader becomes your new wall. Report tokens/second and achieved MFU, not just "it works," so the next person can see what they are trading.

#### Worked example: ZeRO-Infinity on a single node

Now the extreme case the goal of "fit at all costs" is built for. The hardware is a single DGX-class node: 8× A100 80GB (640 GB aggregate HBM), 2 TB of host RAM, and a fast local NVMe array. The goal is to fit a 70B-parameter model — whose training state is $16 \times 70\text{B} \approx 1.12$ TB, well past the 640 GB of aggregate HBM even with perfect ZeRO-3 sharding across all 8 GPUs.

**Where the state goes.** With ZeRO-3 across 8 GPUs, the $16\Psi$ state is divided by 8, giving ~140 GB per GPU of sharded state — still nearly double the 80 GB card. Sharding alone does not fit it. So ZeRO-Infinity offloads the optimizer state (the $12\Psi = 840$ GB block) to the 2 TB of CPU RAM, and streams parameters between NVMe, RAM, and HBM as each layer is needed. The GPUs now hold only the active layers plus activations. The 1.12 TB of state is spread across the hierarchy: hot parameters in HBM, warm state in RAM, cold state on NVMe.

**The bandwidth reality check.** This is where you must be sober. Each optimizer step, the CPU must run Adam over 840 GB of fp32 state. If that state lives in RAM, at ~50 GB/s it is ~17 seconds just to read-modify-write it once. If parts of it spill to NVMe at ~6 GB/s, that portion is slower still. Parameter streaming for the forward pass reads 140 GB of fp16 params from somewhere in the hierarchy each step. Realistically, step times here are measured in tens of seconds and MFU is in the low single digits. The published ZeRO-Infinity results demonstrated fitting models with a *trillion* parameters on modest node counts, and even fine-tuning models with on the order of a trillion parameters on a single DGX-2 node — a genuinely remarkable capability — but the throughput was correspondingly low. This is a "make the impossible possible" tool, not a "train fast" tool.

**When this is the right call.** You reach for ZeRO-Infinity when the alternative is *not doing the work at all*: you need to fine-tune a model that is simply larger than your cluster's aggregate HBM, you cannot get more GPUs, and a slow run is acceptable because it is a one-time job or an experiment. If you have a real training budget and a real cluster, the answer is almost always to add GPUs and use tensor and pipeline parallelism instead — the [3D-parallelism deep-dive on DeepSpeed](/blog/machine-learning/open-source-library/deepspeed-zero-3d-parallelism-deep-dive) walks through how DeepSpeed composes ZeRO with tensor and pipeline parallelism to train huge models *fast* rather than merely *at all*.

## A war story: the offload that made an 8-GPU job crawl

Here is the failure that taught me to distrust copied configs, told the way it actually happened. A teammate was fine-tuning a 6.7B model on a fresh 8× A100 80GB node. The loss was going down, the job was stable — and it was running at roughly half the throughput of a comparable job I had run the month before on the same hardware. Half. On identical GPUs, identical model size, identical sequence length. Something was quietly eating four GPUs' worth of compute, and the run "worked," so nothing was on fire and nobody had noticed.

The first move was not to guess but to *measure*. I turned on `wall_clock_breakdown` in the config, which makes DeepSpeed print a per-step timing breakdown: forward, backward, optimizer step, and the communication inside each. The optimizer step was taking almost as long as the forward and backward combined — a huge red flag, because for a model that fits in HBM the Adam step should be a small fraction of the step, a quick element-wise sweep over on-GPU state. An optimizer step that rivals the forward pass means the update is not happening on the GPU.

That pointed straight at the config, and there it was: an `offload_optimizer` block with `"device": "cpu"`, inherited from a tutorial written for a 24 GB consumer card. On that card the offload was necessary. On an 80 GB A100 it was catastrophic and pointless — the 6.7B model's entire $16\Psi \approx 107$ GB of state fits in 13–14 GB per GPU once sharded across 8 cards, with tens of gigabytes of HBM to spare. There was nothing to offload. But DeepSpeed dutifully did what the config said: every step, it copied gradients over PCIe to the host, ran Adam on the CPU, and copied parameters back — a two-second detour bolted onto a step that should have taken under a second on the GPU. And because PCIe traffic does not overlap perfectly with a compute-light step, the GPUs sat idle waiting for the host, half the time.

The fix was to *delete four lines*. Remove the `offload_optimizer` block, keep `stage: 3`, relaunch. Throughput doubled instantly, MFU jumped from the low twenties to the low forties, and the optimizer step shrank back to a rounding error in the breakdown. Same loss curve, twice the speed, zero code change. The whole diagnosis took fifteen minutes once I looked at the timing breakdown instead of the loss.

The lesson generalizes into a habit. **A config that runs is not a config that is right.** Offload fails silently in the direction that hurts most — it does not crash, it does not NaN, it just quietly halves your throughput and lets the run limp along looking healthy. So two rules stuck with me after that day: always sanity-check whether your model actually needs the offload you configured (run the memory estimator, look at `nvidia-smi` — if the card is half empty, you are offloading for nothing), and always look at the per-step timing breakdown on a new run, not just the loss. The loss will tell you the model is learning; only the breakdown will tell you it is learning as fast as the hardware allows.

The stress-test version of this story is worth stating too. *What if the model had genuinely needed offload on this node?* Then the fix would not be to delete the block but to make the offload hide — enlarge the micro-batch so the GPU forward-backward is long enough to overlap the CPU Adam, add activation checkpointing to afford that larger micro-batch, and pin host memory so the copies are truly asynchronous. The failure mode and the tuning both come back to the same equation, $T_\text{step} \approx \max(T_\text{gpu}, T_\text{offload})$: you either remove $T_\text{offload}$ entirely (delete the block) or grow $T_\text{gpu}$ until it hides $T_\text{offload}$ (bigger micro-batch). What you must never do is leave a small step exposed to an unhidden offload path, which is exactly the state that copied config left the job in.

## DeepSpeed vs FSDP: same math, different ergonomics

You now have two ways to shard the same state: DeepSpeed and PyTorch's native FSDP. They implement the identical ZeRO algorithm — FSDP's `FULL_SHARD` strategy *is* ZeRO-3 — so the memory savings are the same. The choice is about ergonomics, offload depth, and ecosystem.

![A matrix comparing DeepSpeed and FSDP across interface offload sharding kernels and best use cases](/imgs/blogs/deepspeed-zero-and-offload-6.webp)

The matrix above lines them up on the axes that actually decide the pick. Let me expand each:

| Dimension | DeepSpeed | FSDP / FSDP2 |
|---|---|---|
| **Interface** | JSON `ds_config.json`; framework-agnostic training loop | Python code; wrap modules, pass `ShardingStrategy` and `MixedPrecision` |
| **Sharding** | ZeRO stages 1/2/3, selectable per config | `SHARD_GRAD_OP` (= ZeRO-2), `FULL_SHARD` (= ZeRO-3), `HYBRID_SHARD` |
| **Offload** | CPU **and NVMe** (ZeRO-Infinity), deeply engineered | CPU offload only, and less mature; no first-class NVMe |
| **Kernels** | Fused CPU Adam, fused transformer kernels, custom collectives | PyTorch-native; relies on `torch.compile` and stock kernels |
| **Ecosystem** | Its own launcher, Megatron-DeepSpeed for 3D parallelism, MoE support | Native PyTorch; composes cleanly with `DeviceMesh`, `torch.compile`, DTensor |
| **Best when** | You need deep offload, MoE, or 3D parallelism; you like config files | You want a PyTorch-native stack, tight `torch.compile` integration, minimal deps |

The practical decision comes down to two questions. First: **do you need NVMe offload or the most aggressive CPU offload?** If yes, DeepSpeed, no contest — this is where it is years ahead. Second: **do you want to stay entirely inside PyTorch, with `torch.compile`, DTensor, and `DeviceMesh` composing natively?** If yes, FSDP2, which is the modern, cleaner rewrite of FSDP and integrates beautifully with the rest of the PyTorch ecosystem. For most people training a model that fits with sharding on a real cluster, FSDP2 is the lower-friction default. For people who need to fit something on hardware that is too small, DeepSpeed's offload is the reason to switch. Both are covered from the FSDP side in [FSDP in practice](/blog/machine-learning/distributed-training/fsdp-in-practice); this post is the DeepSpeed complement.

One more honest note: you can mix. It is common to use FSDP for the sharding on a cluster and reach for DeepSpeed only when a specific job needs offload. They are tools, not tribes.

The good news for migration is that you rarely touch training-loop code to switch. Hugging Face's `Trainer` and `accelerate` both abstract the two behind a single flag: you pass `--deepspeed ds_config.json` to run under DeepSpeed, or an FSDP config to `accelerate` to run under FSDP, and your model and data code are unchanged. `accelerate config` will even interview you and generate either config. So the pragmatic workflow is to develop with whichever your team defaults to, and only hand-write a raw `ds_config.json` when you need a knob the abstraction does not expose — most often the NVMe offload block, or fine control over the `stage3_*` bucket sizes. Treat the framework choice as reversible, because on the sharding math it genuinely is.

## Case studies and real numbers

Concrete, published results anchor the intuition. Here are four, cited so you can check them.

**ZeRO on a 7.5B model (Rajbhandari et al., "ZeRO," 2020).** The paper's memory table is the canonical reference: a 7.5B-parameter model in mixed-precision Adam needs 120 GB per GPU under plain data parallelism, dropping to 31.4 GB (ZeRO-1), 16.6 GB (ZeRO-2), and 1.9 GB (ZeRO-3) as each stage shards more state across 64 GPUs. ZeRO-3's 64× reduction is what makes trillion-parameter training feasible at all. These are the numbers behind the matrix figure early in this post.

**ZeRO-Offload on a single GPU (Ren et al., "ZeRO-Offload," 2021).** The headline result: a single 32 GB V100 trained a **13B-parameter** model — roughly 10× larger than the ~1.4B that fit with PyTorch alone on the same card — sustaining about **40 TFLOPS**. On a DGX-2 (16 V100s) ZeRO-Offload scaled to 70B parameters with near-linear throughput scaling. This is the origin of the "13B on one GPU" claim, and the 40 TFLOPS figure is exactly the low-double-digit MFU that offload's PCIe-and-CPU path implies.

**ZeRO-Infinity's capacity (Rajbhandari et al., "ZeRO-Infinity," 2021).** By adding the NVMe tier, ZeRO-Infinity demonstrated training models with over a **trillion parameters** on a few hundred GPUs, and showed that a single DGX-2 node could hold and fine-tune models on the order of a trillion parameters by spilling state to NVMe — capacity that is otherwise impossible without thousands of GPUs. The throughput was low, as the bandwidth math predicts, but the *capability* was unprecedented.

**The FSDP equivalence (PyTorch FSDP paper / docs).** PyTorch's Fully Sharded Data Parallel re-implemented ZeRO-3 natively and reported training multi-hundred-billion-parameter models with sharding and optional CPU offload, at throughput competitive with DeepSpeed for the no-offload case. The takeaway is not that one framework wins, but that the *ZeRO algorithm* — sharding params, grads, and optimizer state, gathering on demand — is now the industry-standard memory model, available through two mature front doors.

If you want the deeper systems view of how DeepSpeed composes ZeRO with tensor and pipeline parallelism to reach the very largest models efficiently, the [DeepSpeed ZeRO and 3D-parallelism deep-dive](/blog/machine-learning/open-source-library/deepspeed-zero-3d-parallelism-deep-dive) is the companion to this practical post.

## When to reach for offload (and when not)

Every technique in distributed training is a cost, and offload's cost is throughput. So the decision has to be disciplined. Walk it as a tree.

![A decision tree for whether to offload starting from whether the model fits and branching to more GPUs or CPU and NVMe offload](/imgs/blogs/deepspeed-zero-and-offload-7.webp)

The decision tree above is the one to internalize. Start at the top: does the model fit on the GPUs you have?

- **If it fits with plain ZeRO-2 or ZeRO-3 sharding — do not offload.** This is the loud one. Offload on hardware that does not need it is a pure throughput tax. Delete the `offload_*` blocks and use the fastest stage that fits. If someone hands you a config with offload on and the model fits without it, that is a bug, not a feature.
- **If it does not fit, and you can add GPUs — add GPUs first.** ZeRO-3 divides state by GPU count. Going from 4 to 8 GPUs halves per-GPU state and *increases* throughput. Sharding across more GPUs is strictly better than offloading to a slower tier, whenever more GPUs are available. Only when you have saturated the GPUs you can get does offload enter the picture.
- **If the hardware is truly fixed and it still does not fit — offload the optimizer to CPU first.** ZeRO-Offload of the optimizer state removes the biggest slice ($12\Psi$) at the gentlest throughput cost, because the CPU Adam step overlaps with GPU compute. Pair it with a large micro-batch and activation checkpointing so the offload hides behind compute.
- **If CPU RAM is not enough — offload parameters to CPU too, then reach for NVMe.** Each tier down costs more throughput. NVMe (ZeRO-Infinity) is the floor: it fits almost anything but runs at a fraction of the speed. Reach for it only when the alternative is not running.
- **If even NVMe does not fit or the throughput is unusable — shrink the model or the sequence length, or get a bigger cluster.** Offload is not magic; there is a point where the honest answer is that this model does not belong on this hardware.

The stress tests sharpen the rule. *What at 64 GPUs?* At 64 GPUs, a model that needed offload on 8 GPUs very likely fits with pure ZeRO-3 sharding — check before you offload, because the offload you needed at small scale is dead weight at large scale. *What on PCIe instead of NVLink?* Offload rides PCIe regardless of your inter-GPU interconnect, so a slow PCIe generation (3.0 at ~16 GB/s vs 4.0 at ~25 GB/s vs 5.0 at ~50 GB/s) directly caps your offload throughput — know your PCIe generation before you promise a timeline. *What when the batch is tiny?* Tiny batches are the worst case for offload, because $T_\text{gpu}$ shrinks and the fixed PCIe-plus-CPU cost stops being hidden — if you must offload, make the micro-batch as large as memory allows. *What when the optimizer state won't fit even in RAM?* That is precisely the ZeRO-Infinity NVMe case; accept the tens-of-seconds step time or reduce the model.

### The offload pitfalls checklist

When offload is the right call and it is still slow or still OOMing, the cause is almost always one of a short list. Run down it before you file a bug:

- **You offloaded when the model fit.** Check `nvidia-smi` — if the card is half empty, delete the `offload_*` block. This is the war story, and it is the most common one.
- **Host memory is not pinned.** Without `"pin_memory": true`, copies go through pageable memory and the overlap you counted on serializes. Turn it on unless you are genuinely short on host RAM.
- **The micro-batch is too small to hide the offload.** Grow it (with activation checkpointing to afford it) until the per-step timing breakdown shows the optimizer step tucked behind compute.
- **`nvme_path` points somewhere slow.** It must be a real local NVMe device, not a network filesystem, not a spinning disk, not a tmpfs that silently eats your RAM. Benchmark it with DeepSpeed's `aio_bench` first.
- **You forgot activations.** Offload frees $16\Psi$ of state and then a big batch's activations refill HBM. Turn on activation checkpointing; the two are one plan.
- **The CPU is the bottleneck, not PCIe.** If the optimizer step dominates even with pinned memory, confirm `DeepSpeedCPUAdam` is in use and your host has enough cores; a small host CPU cannot feed a big optimizer.

Most "offload does not work" reports resolve to one of these six, and all six are visible in either `nvidia-smi` or the `wall_clock_breakdown` timing dump. Look there first.

## Key takeaways

- **DeepSpeed and FSDP run the same ZeRO math.** Sharding params, gradients, and optimizer state across $N_d$ ranks drops per-GPU state toward $16\Psi/N_d$. The difference is ergonomics (config file vs code) and offload depth.
- **The `ds_config.json` is the interface.** Your training loop stays plain; the JSON carries the ZeRO stage, offload targets, and the bucket/prefetch knobs. Learn to read it and you can change your whole memory strategy without touching Python.
- **Use the lowest ZeRO stage that fits.** ZeRO-1 is nearly free; ZeRO-2 adds no comms; ZeRO-3 adds an all-gather per forward and backward. Do not pay for a stage you do not need.
- **Offload is a capacity lever, never a speed lever.** ZeRO-Offload moves the fp32 optimizer to CPU RAM and runs Adam on the CPU; ZeRO-Infinity streams params and optimizer to NVMe. Each fits a bigger model and runs slower.
- **The cost is a bandwidth calculation.** CPU offload moves $4\Psi$ bytes over ~25 GB/s PCIe per step; NVMe streams the full optimizer state over ~6 GB/s. Do the arithmetic before you commit to a timeline.
- **Offload overlaps with compute — so use a large micro-batch.** Effective step time is roughly $\max(T_\text{gpu}, T_\text{offload})$. A bigger micro-batch grows $T_\text{gpu}$ and hides the offload behind it.
- **Do not offload if the model fits.** On a cluster where sharding already fits the model, offload just taxes throughput 20–60% for nothing. Add GPUs before you offload; offload only on fixed, too-small hardware.
- **Estimate before you launch.** `estimate_zero3_model_states_mem_needs_all_live` tells you in seconds whether a config fits, saving an hour of trial-and-error OOMs.

## Further reading

- [Why we distribute training at all](/blog/machine-learning/distributed-training/why-distributed-training) — the four walls (model too big, data too big, run too slow, cost too high) that motivate every technique in this series, including offload.
- [ZeRO and FSDP: the memory model](/blog/machine-learning/distributed-training/zero-and-fsdp-the-memory-model) — the $(2+2+12)\Psi$ derivation and how sharding stages 1/2/3 divide it; the prerequisite for this post.
- [FSDP in practice](/blog/machine-learning/distributed-training/fsdp-in-practice) — the PyTorch-native front door to the same ZeRO-3 sharding, with wrapping policies, sharding strategies, and mixed precision.
- [The memory budget](/blog/machine-learning/distributed-training/the-memory-budget) — where every gigabyte goes (params, grads, optimizer, activations) and the fit-or-not calculation offload exists to change.
- [DeepSpeed ZeRO and 3D parallelism, a deep dive](/blog/machine-learning/open-source-library/deepspeed-zero-3d-parallelism-deep-dive) — how DeepSpeed composes ZeRO with tensor and pipeline parallelism to train the largest models fast, not just at all.
- [The distributed training playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook) — the capstone decision-and-debugging checklist that ties the whole series together, including where offload sits in the lever order.
- Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (2020); Ren et al., "ZeRO-Offload: Democratizing Billion-Scale Model Training" (2021); Rajbhandari et al., "ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning" (2021) — the three source papers, and the DeepSpeed configuration JSON docs for the exact knob semantics.
