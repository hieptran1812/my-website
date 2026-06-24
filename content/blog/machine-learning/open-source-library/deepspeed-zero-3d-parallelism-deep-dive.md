---
title: "DeepSpeed Deep-Dive: ZeRO, 3D Parallelism, and the Tradeoffs Behind Trillion-Parameter Training"
date: "2026-05-19"
publishDate: "2026-05-19"
description: "A principal-engineer tour of how DeepSpeed actually trains models bigger than your GPUs — ZeRO partitioning, offload, 3D parallelism, the communication costs nobody budgets for, and thirteen production case studies."
tags: ["deepspeed", "zero", "distributed-training", "3d-parallelism", "model-parallelism", "gpu-memory", "mixed-precision", "mlops", "large-language-model", "pytorch"]
category: "machine-learning"
subcategory: "Open Source Library"
author: "Hiep Tran"
featured: true
readTime: 50
---

Here is the assumption that gets teams into trouble: *training a bigger model is a procurement problem.* You want to train something that doesn't fit, so you buy more GPUs, wrap the model in `DistributedDataParallel`, and expect the memory to spread out. It does not. Plain data parallelism replicates the *entire* model — parameters, gradients, and optimizer states — on every single GPU. Adding a 64th GPU to a 63-GPU job buys you exactly zero additional bytes of model capacity. It buys throughput, not headroom.

DeepSpeed exists because the memory wall and the compute wall are two different walls, and the distributed-training tools most engineers reach for first only knock down one of them. [DeepSpeed](https://github.com/deepspeedai/DeepSpeed) is the library that made it routine to train a 10B-parameter model on hardware that, on paper, cannot hold a 10B-parameter model — and then made it possible to train 100B+ and trillion-scale models on clusters that would otherwise need to be 8× larger. It powered BLOOM-176B, Megatron-Turing NLG 530B, and a long tail of models you have used without knowing it.

This is a deep-dive in the literal sense. We will not stop at "ZeRO saves memory." We will walk the actual byte arithmetic, the communication volume each optimization adds, the bandwidth tiers it exploits, and — most importantly — the places where DeepSpeed makes your training *slower* and you have to decide whether the memory was worth it. Every optimization here is a trade. The engineers who get burned are the ones who enabled a feature for the memory win and never measured what they paid for it.

## Why DeepSpeed is different

Start with the mismatch. Most engineers carry a mental model of distributed training that was correct in 2018 and has been quietly wrong ever since.

| Question | The naive assumption | The reality DeepSpeed is built around |
|---|---|---|
| What limits model size? | GPU FLOPs | GPU **memory**, and specifically optimizer-state memory |
| What does data parallelism do? | Splits the model across GPUs | **Replicates** the entire model on every GPU |
| Where does training memory go? | Mostly the weights | Optimizer states (3× the weights) + activations (often more) |
| How do I train a model 4× too big? | Buy 4× the GPUs | Buy 4× the GPUs **and partition the state**, or you gain nothing |
| Is the optimizer step expensive? | No, it's cheap vs. matmuls | In FLOPs yes; in **memory and bandwidth** it dominates |
| What slows a big training job down? | Slow GPUs | **Communication** — all-reduce, all-gather, reduce-scatter volume |

Every row of that table is a thing DeepSpeed's design responds to directly. The library is, at its core, a set of answers to one question: *given that memory is the wall and communication is the cost, how do we place the bytes?*

DeepSpeed is not a model library and not a trainer in the Hugging Face `Trainer` sense. It is an engine you wrap around an ordinary PyTorch module. You hand it a model, an optimizer, and a JSON config; it hands you back objects that look like a model and an optimizer but whose memory and communication behavior have been completely re-plumbed. That JSON config is where all the decisions live, and most of this article is really an annotated tour of it.

## The mental model: where training memory actually goes

Before any optimization, you have to know what you are optimizing. Ask a room of ML engineers "what uses GPU memory during training" and most will say "the model." The model is frequently the *smallest* of the four tenants.

![Stacked breakdown of GPU training memory showing activations and optimizer states dominating over parameters and gradients](/imgs/blogs/deepspeed-zero-3d-parallelism-deep-dive-1.png)

The diagram above is the mental model: training memory has four tenants, and the parameters — the thing you actually ship — are the cheapest two. Let $N$ be the parameter count. In standard mixed-precision training with the Adam optimizer:

- **Parameters**, FP16 working copy: $2N$ bytes. This is the model you forward through.
- **Gradients**, FP16: $2N$ bytes. One per parameter, produced by backward.
- **Optimizer states**: an FP32 *master copy* of the weights ($4N$), plus Adam's first moment $m$ ($4N$) and second moment $v$ ($4N$). That is $12N$ bytes — **six times the FP16 weights**.
- **Activations**: the intermediate tensors backward needs. This one does not scale with $N$ at all; it scales with batch size, sequence length, and depth, and for long-context training it is routinely the single largest line item.

Sum the first three and you get the famous number: **$16N$ bytes per parameter** of model state, before a single activation. A 7B model is therefore $7 \times 10^9 \times 16 \approx 112$ GB of state. That does not fit on an 80 GB H100. Not close. And note what dominates: the $12N$ of optimizer state, the part you never think about because it is invisible at inference time.

This is the crux. Inference engineers reason about $2N$. Training engineers who still reason about $2N$ are off by a factor of eight. DeepSpeed's entire ZeRO family is an attack on that $12N$ first, then the $2N + 2N$, then — with offload and checkpointing — the activations.

> The parameters are the tip of the iceberg. The optimizer states are the iceberg. Plan your memory budget around $16N$, not $2N$, or your job will OOM at step zero.

A worked number to anchor everything that follows. Take a 13B-parameter model, Adam, mixed precision:

- Model state: $13 \times 10^9 \times 16 = 208$ GB.
- On 8× 80 GB GPUs with plain DDP: each GPU needs all 208 GB. **OOM.**
- The aggregate cluster memory is $8 \times 80 = 640$ GB. The model state is 208 GB. It *fits in aggregate* — three times over. The problem was never total memory. The problem was that DDP refuses to spread it.

That gap — "fits in aggregate, OOMs per-GPU" — is the entire opportunity. Everything DeepSpeed does is a way to convert aggregate memory into usable memory.

It is worth pausing on *why* the four tenants have the sizes they do, because the ratios are not arbitrary and they drive every later decision. Parameters and gradients are $2N$ each because BF16/FP16 is two bytes and there is exactly one gradient per parameter. The optimizer's $12N$ is three FP32 quantities per parameter: the master weight and Adam's two moments. Swap the optimizer and the constant changes — plain SGD with momentum keeps one FP32 momentum buffer plus the master copy, roughly $8N$; SGD without momentum is lighter still; 8-bit optimizers like `bitsandbytes`' quantized Adam cut the moment storage dramatically. This is a real lever: if you are memory-bound and have not questioned the optimizer, you are leaving an easy win unclaimed. DeepSpeed's ZeRO attacks the $12N$ structurally by partitioning it; choosing a lighter optimizer attacks the constant itself, and the two compose. The point of the taxonomy is that each tenant has a different owner, a different lever, and a different cost — and a memory plan that treats "GPU memory" as one undifferentiated pool will mis-budget every time.

## 1. ZeRO: partitioning the redundancy away

**Senior rule of thumb: data parallelism replicates state; ZeRO shards it. If you are running DDP on a model that barely fits, you are leaving 7/8ths of your cluster's memory on the floor.**

ZeRO — Zero Redundancy Optimizer — is the observation that DDP's replication is *pure waste*. If GPU 3 is only ever going to update its own slice of the optimizer states, why does it store the other seven slices? It does not need them at the moment of the update. ZeRO partitions the redundant state across the data-parallel group and reconstructs only the pieces a GPU needs, only when it needs them.

ZeRO comes in three stages, and they are strictly nested: each stage does everything the previous one did, plus partitions one more class of memory.

![Matrix showing ZeRO stages 1, 2, 3 and which of optimizer states, gradients, and parameters each one partitions](/imgs/blogs/deepspeed-zero-3d-parallelism-deep-dive-2.png)

Let $d$ be the data-parallel degree (the number of GPUs in the group).

- **ZeRO-1** partitions the **optimizer states**. The $12N$ becomes $12N/d$ per GPU. Parameters and gradients are still replicated. This is free in the sense that it adds *no* communication beyond what DDP already does — DDP already all-reduces gradients; ZeRO-1 just reorganizes who owns what afterward. If you run DDP today and want a memory win for zero throughput cost, ZeRO-1 is the answer. It is the most under-used setting in the library.
- **ZeRO-2** additionally partitions the **gradients**. Now $2N$ of gradient memory becomes $2N/d$. The all-reduce of gradients is replaced by a **reduce-scatter** — each GPU ends up with only its slice of the summed gradient. Same communication *volume* as DDP's all-reduce, just a different collective. Still essentially free on throughput.
- **ZeRO-3** partitions the **parameters** themselves. This is the big one and the expensive one. Each GPU permanently holds only $2N/d$ of the weights. To run a layer's forward pass, it must **all-gather** that layer's full weights from the other GPUs, compute, then throw the gathered copy away. This adds communication that DDP never had. We will spend all of section 5 on that cost.

Here is the memory arithmetic side by side.

![Before-after comparison of per-GPU memory under plain data parallelism versus ZeRO-3](/imgs/blogs/deepspeed-zero-3d-parallelism-deep-dive-3.png)

| Mode | Optimizer states / GPU | Gradients / GPU | Parameters / GPU | Total / GPU |
|---|---|---|---|---|
| DDP | $12N$ | $2N$ | $2N$ | $16N$ |
| ZeRO-1 | $12N/d$ | $2N$ | $2N$ | $4N + 12N/d$ |
| ZeRO-2 | $12N/d$ | $2N/d$ | $2N$ | $2N + 14N/d$ |
| ZeRO-3 | $12N/d$ | $2N/d$ | $2N/d$ | $16N/d$ |

For the 13B model on 8 GPUs: DDP needs 208 GB/GPU (OOM); ZeRO-1 needs $4(26) + 12(26)/8 = 104 + 39 = 143$ GB (still OOM); ZeRO-2 needs $26 + 14(26)/8 \approx 26 + 45.5 = 71.5$ GB (fits!); ZeRO-3 needs $208/8 = 26$ GB (fits comfortably, room for activations and a bigger batch). The stage you pick is the smallest one that fits — because each stage upward costs more communication.

Here is the config that turns it on. DeepSpeed is configured almost entirely through a JSON file:

```json
{
  "train_batch_size": 256,
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 8,
  "bf16": { "enabled": true },
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6,
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true
  },
  "gradient_clipping": 1.0,
  "steps_per_print": 10
}
```

And the Python side is deliberately small — DeepSpeed wants to be a thin wrapper:

```python
import deepspeed
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")

engine, optimizer, _, lr_scheduler = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config="ds_config.json",
)

for step, batch in enumerate(dataloader):
    batch = {k: v.to(engine.device) for k, v in batch.items()}
    loss = engine(**batch).loss
    engine.backward(loss)        # NOT loss.backward() — engine owns the graph
    engine.step()                # optimizer step + zero_grad + lr schedule
```

Three things to internalize from that loop. You call `engine.backward(loss)`, not `loss.backward()` — DeepSpeed needs to interleave reduce-scatter and parameter-free operations into the backward pass, so it owns it. You call `engine.step()` once and it does the optimizer step, the gradient zeroing, and the LR schedule together. And `gradient_accumulation_steps` is handled *inside* the engine — you do not write an accumulation loop, you just feed micro-batches and DeepSpeed counts.

### Second-order optimization: the mixed-precision master copy

The single most misunderstood part of the $16N$ figure is *why the optimizer states are FP32 when everything else is FP16*. It is not an accident and you cannot just delete it.

In mixed-precision training the forward and backward run in FP16 or BF16 for speed and memory. But the optimizer update $w \leftarrow w - \eta \cdot \hat{m}/(\sqrt{\hat{v}} + \epsilon)$ involves adding a tiny number (the scaled gradient) to a large number (the weight). In FP16, with ~3 significant decimal digits, that addition silently rounds to a no-op once the update is small relative to the weight. Your model stops learning and the loss curve flatlines for no visible reason.

The fix is the **FP32 master copy**: the optimizer keeps weights in FP32, applies the update there with full precision, and casts down to the FP16 working copy each step. That master copy is $4N$ of the $12N$. It is load-bearing. When you see ZeRO-3 partition the optimizer states, the most valuable thing it is partitioning is that FP32 master copy — the part you would never think to shard by hand.

This is also why BF16 changed the calculus. BF16 has the same exponent range as FP32, so loss scaling is usually unnecessary, but it still has only 8 mantissa bits — *fewer* than FP16's 10 — so the FP32 master copy is still required. BF16 buys you stability, not a smaller optimizer. Anyone who tells you BF16 removes the master copy is wrong; check `optimizer.fp32_partitioned_groups` in any ZeRO job and you will find it.

### The three collectives ZeRO is built on

You cannot reason about ZeRO's cost without a working grasp of three collective operations, because every stage is defined by which collective it swaps in. They are worth nailing down precisely.

**All-reduce** takes a tensor that every GPU holds a copy of, sums (or averages) them element-wise, and leaves the *full result* on every GPU. This is what plain DDP does with gradients: every GPU computed gradients on its own micro-batch, all-reduce sums them so every GPU ends with the global gradient. Cost: each GPU sends and receives roughly $2\times$ the tensor size in the standard ring implementation.

**Reduce-scatter** also sums across GPUs, but instead of leaving the whole result everywhere, it leaves each GPU with only *its slice* of the sum. ZeRO-2 swaps DDP's gradient all-reduce for a reduce-scatter — which is exactly right, because under ZeRO-2 each GPU only owns and updates one slice of the gradients, so delivering the whole summed gradient everywhere would be wasted bandwidth. Cost: roughly half an all-reduce.

**All-gather** is the inverse: every GPU starts with one slice of a tensor, and the operation assembles the full tensor on every GPU. ZeRO-3 uses all-gather to reconstruct a layer's full parameters from the per-GPU shards just before that layer computes. Cost: roughly half an all-reduce, per gather.

Now the stages fall out cleanly. An all-reduce is, in fact, *equivalent to* a reduce-scatter followed by an all-gather — that is how ring all-reduce is implemented. So DDP's one all-reduce ≈ reduce-scatter + all-gather. ZeRO-2 keeps the reduce-scatter (for gradients) and, because parameters are still replicated, skips the parameter all-gather entirely — same total volume as DDP. ZeRO-3 keeps the gradient reduce-scatter *and* adds a parameter all-gather in forward *and another* in backward. That is the precise origin of the "1.5× DDP communication" figure: DDP does the equivalent of one reduce-scatter + one all-gather; ZeRO-3 does one reduce-scatter + two all-gathers. Three half-units instead of two. Internalize this and the throughput numbers in section 5 stop being surprising.

## 2. ZeRO-Offload: trading PCIe bandwidth for GPU memory

**Senior rule of thumb: offload moves the memory-bound work off the GPU and keeps the compute-bound work on it. If you offload the wrong half, you turn a GPU job into a CPU job.**

ZeRO-3 partitions across GPUs. But what if you only have *one* GPU, or eight GPUs and a model that needs eighty? You have run out of GPUs to partition across. ZeRO-Offload's answer: the GPU is not the only memory in the box. The host has hundreds of gigabytes of DRAM sitting mostly idle during training. Use it.

The non-obvious part is *what* to offload. The naive instinct is "offload the weights" — but the weights are touched constantly by every forward and backward. The genius of ZeRO-Offload is the split it actually chooses.

![ZeRO-Offload pipeline showing forward and backward on GPU, optimizer step on CPU, with PCIe transfers between](/imgs/blogs/deepspeed-zero-3d-parallelism-deep-dive-4.png)

Forward and backward are **compute-bound** — they are matmuls, the thing the GPU is built for, and they touch every weight. Those stay on the GPU. The **optimizer step**, by contrast, is *memory-bound*: it streams the $12N$ of optimizer state through a handful of cheap element-wise operations (a couple of multiplies and an add per parameter). The arithmetic intensity is near zero. A CPU, which has far less compute but plenty of memory bandwidth relative to that tiny FLOP count, can do the optimizer step *acceptably* — and doing it on the CPU means the entire $12N$ of optimizer state never has to live in GPU memory at all.

So ZeRO-Offload's split is: **FP16 forward/backward on GPU, FP32 Adam step on CPU, FP32 optimizer states in host DRAM.** Per step, the GPU ships FP16 gradients to the host over PCIe, the CPU runs Adam, and the updated FP16 parameters come back. DeepSpeed ships a hand-written, AVX-vectorized, OpenMP-parallelized **CPU Adam** kernel for exactly this — a naive PyTorch CPU optimizer would be 4–10× slower and make the whole scheme pointless.

```json
{
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    }
  }
}
```

`pin_memory: true` matters more than it looks. Pinned (page-locked) host memory lets the PCIe DMA engine transfer without the OS staging through a pageable bounce buffer, and it lets the copy overlap with GPU compute. Without it the gradient transfer is synchronous and serializes against your backward pass. Always pin when you offload.

The tradeoff you are signing up for, stated plainly:

| You gain | You pay |
|---|---|
| $12N$ of optimizer state leaves the GPU | Every step, $2N$ of gradients + $2N$ of params cross PCIe |
| Single-GPU training of models 5–10× too big | Optimizer step now bounded by ~16–25 GB/s PCIe, not 2 TB/s HBM |
| No extra GPUs needed | CPU Adam competes with the dataloader for CPU cores and DRAM bandwidth |

The rule: offload when you are **memory-desperate**, not when you are throughput-sensitive. ZeRO-Offload turns an impossible job into a slow job. A slow job that finishes beats a fast job that OOMs — but if the job already fit, offload is a pure throughput regression. Measure tokens/sec before and after; if memory headroom was not the binding constraint, turn it back off.

### Second-order optimization: the CPU is now on your critical path

Once you offload, the host CPU is a training resource, not just a babysitter. Two things bite teams here. First, CPU Adam is multi-threaded and *will* contend with your PyTorch dataloader workers — if you run `num_workers=32` on a 32-core box and offload, the optimizer step and data loading fight for the same cores and both get slower. Leave headroom. Second, host DRAM bandwidth is finite and shared; on a dual-socket box, if the optimizer state lands in the NUMA node *far* from the CPU running CPU Adam, every optimizer step pays a cross-socket penalty. `numactl --membind` on the training process is a real, measurable win that nobody remembers to try.

## 3. ZeRO-Infinity: when even host memory runs out

**Senior rule of thumb: ZeRO-Infinity makes any model trainable on any hardware — and the price is that your SSD is now in the training loop. Treat NVMe as a memory tier with a bandwidth number, not as infinite free space.**

ZeRO-Offload spills to CPU DRAM. ZeRO-Infinity goes one tier further: it spills to **NVMe SSD**. The framing that makes it click is to stop thinking of storage as a separate thing and see it as the bottom of a memory pyramid, where each tier trades bandwidth for capacity.

![Three-tier memory pyramid: GPU HBM, CPU DRAM, NVMe SSD, ordered by bandwidth and capacity](/imgs/blogs/deepspeed-zero-3d-parallelism-deep-dive-5.png)

The numbers on that pyramid are the whole design:

| Tier | Bandwidth | Typical capacity | Holds, under ZeRO-Infinity |
|---|---|---|---|
| GPU HBM | ~2–3 TB/s | 40–80 GB | Only the layer currently computing |
| CPU DRAM | ~50–150 GB/s | 0.5–2 TB | Warm optimizer states, prefetched params |
| NVMe SSD | ~3–7 GB/s | 10–60 TB | Cold parameters + optimizer-state overflow |

ZeRO-Infinity moves data up the pyramid just in time for compute and back down right after. It exploits a fact the diagram makes obvious: GPU HBM only ever needs to hold *one layer at a time* during forward and backward. The other 95% of the model can sit on NVMe and be streamed in. The library overlaps these NVMe reads with compute on the previous layer — if a layer's compute takes longer than streaming the next layer's weights off the SSD, the offload is *free*. If it does not, you are NVMe-bound and the SSD bandwidth number above is your training speed.

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "nvme",
      "nvme_path": "/local_nvme",
      "pin_memory": true,
      "buffer_count": 4
    },
    "offload_param": {
      "device": "nvme",
      "nvme_path": "/local_nvme",
      "buffer_count": 5,
      "buffer_size": 1e8
    }
  },
  "aio": {
    "block_size": 1048576,
    "queue_depth": 8,
    "thread_count": 1,
    "single_submit": false,
    "overlap_events": true
  }
}
```

That `aio` block is the part everyone ignores and then files a bug about. DeepSpeed talks to NVMe through Linux **libaio** asynchronous I/O, and the `block_size`, `queue_depth`, and `thread_count` are tuning knobs for *your specific SSD*. The defaults are conservative. DeepSpeed ships `ds_io` and the `aio_bench` utility precisely so you can sweep these and find the combination that saturates your drive — and the difference between tuned and untuned is routinely 2–3×. We will see this go wrong in case study 6.

The honest verdict on ZeRO-Infinity: it removes the model-size limit entirely — the original paper trained a 1-trillion-parameter model on a single DGX-2 node — but it converts your training job into something whose speed is dictated by storage hardware. Use it to make the impossible possible, to fit a model for a one-off experiment, or to fine-tune something huge on modest hardware. Do not use it as the backbone of a job you will run for three weeks; at that duration, renting GPUs that hold the model in HBM is almost always cheaper than the wall-clock you lose to the SSD.

### Second-order optimization: overlap depth and the bandwidth budget

ZeRO-Infinity's ability to hide NVMe latency rests entirely on having *enough work queued ahead* to cover the read. That is what `buffer_count` and `buffer_size` control — they size the staging buffers DeepSpeed uses to prefetch parameter and optimizer-state tiles from NVMe while the GPU computes on the current layer. Too few buffers and the prefetcher cannot run far enough ahead; the GPU drains its work queue and stalls on a synchronous read. Too many and the buffers themselves consume the GPU and CPU memory you were trying to free. The tuning loop is concrete: enable the `flops_profiler`, watch for per-step gaps where GPU utilization dips, and raise `buffer_count` until the gaps close or memory runs out.

There is also a bandwidth budget to respect that nobody writes down. A single PCIe Gen4 x16 link is ~25 GB/s, and on a multi-GPU box *every* GPU's offload traffic, *plus* the dataloader, *plus* the NVMe reads may all be contending for the same PCIe root complex and the same DRAM channels. Eight GPUs each streaming parameters from NVMe through host DRAM is not eight independent 7 GB/s pipes — it is eight consumers of one shared, finite bandwidth tree. The practical consequence: per-GPU offload throughput on a fully-loaded 8-GPU node is often a fraction of what a single-GPU benchmark suggested. Benchmark offload at the concurrency you will actually run, not with one GPU and an idle box, or your capacity plan will be optimistic by a large factor.

One more knob worth knowing: `offload_optimizer` and `offload_param` are independent. A common middle-ground configuration offloads only the optimizer states (the $12N$, the largest and least latency-sensitive tenant) to NVMe while keeping parameters on GPU or in CPU DRAM. That keeps the latency-critical per-layer parameter gathers off the SSD and confines NVMe traffic to the once-per-step optimizer update, which is far easier to overlap. Reach for full parameter offload only when even that is not enough.

## 4. 3D parallelism: data, pipeline, and tensor combined

**Senior rule of thumb: ZeRO-3 is one way to scale; 3D parallelism is the other. They partition different things, and at extreme scale you use both.**

ZeRO-3 partitions *state*. 3D parallelism partitions *computation* along three independent axes. For models past ~100B parameters, the largest training runs combine all three plus ZeRO.

![Matrix of the three parallelism axes — data, pipeline, tensor — showing what each splits, its communication pattern, and when to use it](/imgs/blogs/deepspeed-zero-3d-parallelism-deep-dive-6.png)

The three axes:

- **Data parallelism** splits the *batch*. Every replica has the full model and processes different samples; gradients are all-reduced (or reduce-scattered under ZeRO) once per step. Communication is once-per-step and tolerant of slower interconnect.
- **Pipeline parallelism** splits the model's *layers* into sequential stages on different GPUs. GPU 0 holds layers 1–8, GPU 1 holds 9–16, and so on. Only activations cross the stage boundary — a small, point-to-point transfer. This is how you train a model too *deep* to fit, and it is interconnect-friendly because the transfers are small and local.
- **Tensor parallelism** (also called intra-layer or Megatron-style) splits *each individual layer's matmul* across GPUs — half the attention heads here, half there. It needs an all-reduce *inside every layer*, twice per transformer block. That is enormous communication volume, so tensor parallelism is only viable across GPUs joined by very fast links — NVLink within a single node. Never span tensor parallelism across a slow network; you will spend more time in all-reduce than in matmul.

The reason all three coexist is that they fail at different scales. Tensor parallelism is limited to a node (8 GPUs) by interconnect. Pipeline parallelism is limited by the bubble (section 5) and by how few stages you can tolerate. Data parallelism scales arbitrarily but does nothing for model size on its own. So the canonical recipe for a 530B-class model is: tensor-parallel within each node (degree 8), pipeline-parallel across a handful of nodes, data-parallel across the rest, and ZeRO-1 layered on top to shard the optimizer states of each data-parallel replica.

```python
from deepspeed.pipe import PipelineModule, LayerSpec

layers = (
    [LayerSpec(EmbeddingLayer, vocab, hidden)]
    + [LayerSpec(TransformerBlock, hidden, heads) for _ in range(num_layers)]
    + [LayerSpec(LMHead, hidden, vocab)]
)

model = PipelineModule(
    layers=layers,
    num_stages=4,                       # pipeline depth
    partition_method="parameters",      # balance stages by param count
    activation_checkpoint_interval=1,   # checkpoint every layer
)

engine, _, _, _ = deepspeed.initialize(model=model, config="ds_config.json")
```

The hard constraint pipeline parallelism imposes: your model has to be *expressible as a sequence of layers*. A clean stack of transformer blocks is fine. A model with skip connections that jump across many layers, or control flow that depends on the data, fights the `PipelineModule` abstraction. This is the friction point — and it is exactly why the field has partly migrated toward FSDP/ZeRO-3 style sharding, which imposes no structural requirement on the model. If you want the contrast, the [TorchTitan deep-dive](/blog/machine-learning/open-source-library/torchtitan-pytorch-native-pretraining-deep-dive) walks through the PyTorch-native FSDP approach that competes directly with this.

### Second-order optimization: partition_method is not cosmetic

`partition_method="parameters"` balances stages by parameter count. That sounds right and is often wrong. Pipeline stages must be balanced by *time*, not parameters. An embedding layer has enormous parameter count but trivial FLOPs; a transformer block has the reverse. If you balance by parameters, the stage holding the embedding finishes its micro-batch instantly and then idles, waiting for the FLOP-heavy stages. The whole pipeline runs at the speed of its slowest stage. `partition_method="type:transformer"` or a manual partition that equalizes *wall-clock per stage* is what you actually want. We will see this exact failure in case study 5.

## 5. The communication cost nobody budgets for

**Senior rule of thumb: every byte of memory ZeRO-3 saves you is a byte that has to travel over the network later. The memory is free; the bandwidth is not.**

This is the section that separates engineers who *use* DeepSpeed from engineers who *understand* it. ZeRO-3's headline — $16N/d$ memory per GPU — is real. But the parameters it shed do not vanish. They have to be reassembled, on demand, every forward and every backward.

![Timeline of ZeRO-3 per-layer communication: all-gather weights, compute, free, all-gather again, backward, reduce-scatter gradients](/imgs/blogs/deepspeed-zero-3d-parallelism-deep-dive-7.png)

Walk the timeline for a single layer under ZeRO-3. Before the forward pass of layer $L$ can run, the GPU must **all-gather** layer $L$'s parameters — collect the $d-1$ shards it does not own from its peers — to briefly reconstruct the full layer. It computes the forward. Then it **frees** the gathered copy, keeping only its own $1/d$ shard, so memory drops back down. In the backward pass it has to **all-gather the same weights a second time** (it threw them away), compute the gradient, and finally **reduce-scatter** the gradients so each GPU keeps only its slice.

Count the communication. Plain DDP communicates the gradients exactly once per step: one all-reduce of $2N$ bytes. ZeRO-3 does an all-gather of the parameters in forward ($\approx 2N$), an all-gather again in backward ($\approx 2N$), and a reduce-scatter of gradients ($\approx 2N$). That is roughly **3× the communication volume of DDP**. The standard accounting: ZeRO-1 and ZeRO-2 keep DDP's communication volume; ZeRO-3 multiplies it by 1.5×.

On a cluster with 400–800 Gb/s InfiniBand and NVSwitch, that extra communication is largely *hidden* — DeepSpeed prefetches layer $L+1$'s weights while layer $L$ computes, and if compute is slower than the gather, the gather is free. The settings that make this overlap happen:

```json
{
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6,
    "reduce_scatter": true,
    "contiguous_gradients": true
  }
}
```

`overlap_comm: true` lets communication run concurrently with compute. `stage3_prefetch_bucket_size` controls how far ahead DeepSpeed gathers the next layers — too small and you cannot hide the latency, too large and the prefetch buffer eats the memory you were trying to save. `stage3_param_persistence_threshold` says "parameters smaller than this stay un-partitioned and replicated" — layer norms, biases, small embeddings; partitioning them costs more in collective-launch overhead than it saves in bytes.

But on a cluster with only PCIe between GPUs and slow Ethernet between nodes, that 3× communication cannot be hidden, and ZeRO-3 becomes *communication-bound*. The GPUs sit idle waiting for all-gathers. This is the single most common DeepSpeed disappointment, and it is case study 2.

#### A worked overlap calculation

The question that decides whether ZeRO-3 is free or ruinous on a given cluster is simple: *for each layer, is the compute time longer than the all-gather time?* If yes, the gather hides and ZeRO-3 costs nothing; if no, you are bandwidth-bound and the deficit is pure idle.

Take one transformer block of a 13B model — call it ~0.4B parameters in the block, so ~0.8 GB of BF16 weights. On 8 GPUs the all-gather has to move roughly $(d-1)/d \approx 0.875$ of that, ~0.7 GB, across the GPU interconnect. On NVSwitch at ~300 GB/s effective, that gather takes ~2.3 ms. The forward compute for that block at a reasonable batch and sequence length is on the order of 8–15 ms on an H100. Compute dominates the gather by 4–6×: the prefetcher hides it completely, ZeRO-3 is free, ship it.

Now run the *same* model on a node where the only GPU-to-GPU path is PCIe Gen4 at ~25 GB/s effective. The same 0.7 GB gather now takes ~28 ms — *longer than the compute itself*. The prefetcher cannot win a race it starts 13 ms behind. Every layer, the GPU finishes its 8–15 ms of compute and then stalls ~15–20 ms waiting for the next layer's weights. Your effective throughput is roughly halved, and no amount of `stage3_prefetch_bucket_size` tuning fixes it, because the problem is not scheduling — it is that the wire is too slow. That is the arithmetic behind "check `nvidia-smi topo -m` before you pick a stage." The crossover is not subtle; it is a 10× swing in interconnect bandwidth, and it flips ZeRO-3 from free to fatal.

The activations are the other half of the memory story, and the lever there is **activation checkpointing** (gradient checkpointing): instead of storing every layer's activations for the backward pass, store only a few checkpoints and *recompute* the rest during backward. It trades compute for memory — roughly a 33% increase in forward FLOPs (one extra forward) to cut activation memory from $O(\text{layers})$ to $O(\sqrt{\text{layers}})$. For long-context training it is not optional; activations dominate and checkpointing is the only thing that fits them.

```json
{
  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": false,
    "contiguous_memory_optimization": true,
    "number_checkpoints": 4,
    "synchronize_checkpoint_boundary": false
  }
}
```

`partition_activations: true` additionally shards the *stored* checkpoints across the tensor-parallel group — a second-order win that only applies when you have tensor parallelism. `cpu_checkpointing: true` pushes the checkpoints to host DRAM, which you reach for only when GPU memory is so tight that even the $O(\sqrt{n})$ checkpoints do not fit.

### The pipeline bubble

Pipeline parallelism has its own communication-shaped tax, and it is not bandwidth — it is *idle time*. When you split the model into $p$ stages, the first micro-batch has to traverse all $p$ stages before stage $p$ has anything to do. While it travels, stages $2 \ldots p$ sit idle. Symmetrically, at the end of the step the early stages idle while the tail drains.

![Before-after of pipeline scheduling: GPipe fill-drain with large bubbles versus 1F1B interleaved with smaller bubbles](/imgs/blogs/deepspeed-zero-3d-parallelism-deep-dive-8.png)

That idle wedge is the **bubble**. With $p$ stages and $m$ micro-batches, the naive GPipe fill-drain schedule wastes a fraction $(p-1)/(m + p - 1)$ of all device-time. With $p = 8$ and $m = 8$, that is $7/15 \approx 47\%$ — you bought 8 GPUs and 4 of them are, on average, doing nothing.

Two levers shrink it. Crank $m$: with $p = 8$ and $m = 64$, the bubble falls to $7/71 \approx 10\%$. And use a better schedule — DeepSpeed implements **1F1B** (one-forward-one-backward), which interleaves forward and backward micro-batches so stages stay busy and, as a bonus, bounds peak activation memory per stage instead of letting it grow with $m$. The bubble formula is unchanged, but 1F1B reaches the steady state faster and holds it. The practical rule: **pipeline parallelism only pays off when you can run many micro-batches per step.** If your global batch size is small, the bubble eats you alive and ZeRO-3 is the better tool.

## 6. DeepSpeed-MoE: scaling parameters without scaling FLOPs

**Senior rule of thumb: a Mixture-of-Experts model multiplies your parameter count and your communication bill in the same breath. DeepSpeed-MoE is mostly a fight against that communication bill.**

There is a second way to make a model bigger that is orthogonal to everything above. Instead of making every layer wider, you replace a dense feed-forward block with $E$ parallel feed-forward "experts" and a small router that sends each token to the top-$k$ of them — usually $k = 1$ or $k = 2$. The parameter count grows by roughly $E\times$, but the FLOPs per token barely move, because each token still only visits $k$ experts. You get the representational capacity of a much larger model at the compute cost of a small one. This is the bet behind every large MoE model, and the broader tradeoffs are covered in [optimizing MoE training and inference](/blog/machine-learning/large-language-model/optimizing-moe-training-and-inference) and [MoE LLM architecture](/blog/machine-learning/large-language-model/moe-llm-architecture-training-finetuning-case-studies).

The catch is that those $E$ experts have to live *somewhere*, and they are far too large to replicate. So MoE introduces a fourth kind of parallelism — **expert parallelism** — where each GPU physically holds a different subset of the experts. And now the problem becomes routing: a token that lands on GPU 3 but is routed to an expert on GPU 7 has to *travel there*, be processed, and travel back. That round trip is an **all-to-all** collective — every GPU sends a different slice of its tokens to every other GPU — and all-to-all is the most bandwidth-hungry, most latency-sensitive collective in the entire distributed-training vocabulary. A naive MoE layer can spend more wall-clock in two all-to-alls than in the expert matmuls they bracket.

DeepSpeed-MoE is, in practice, a collection of engineering moves to make that all-to-all survivable:

- **Capacity factor and token dropping.** Experts are given a fixed buffer size — `capacity_factor` × (tokens / experts). If the router sends an expert more tokens than its buffer holds, the overflow tokens are *dropped* (skip the expert, pass through via the residual). This keeps every all-to-all a fixed, predictable shape — variable-size collectives are a nightmare to overlap — at the cost of some tokens silently not being processed by their chosen expert. Tune `capacity_factor` between 1.0 and 1.5: too low drops too many tokens and hurts quality, too high wastes compute and memory on empty buffer slots.
- **Hierarchical all-to-all.** A flat all-to-all across 256 GPUs crosses the slow inter-node network for most of its traffic. DeepSpeed-MoE decomposes it into an intra-node all-to-all over fast NVLink followed by a smaller inter-node all-to-all, cutting the volume that touches the slow tier.
- **Expert parallelism composed with the rest.** Expert parallelism is a *separate* communication group from data, pipeline, and tensor parallelism. DeepSpeed lets you set the expert-parallel degree independently — typically you want it equal to the number of experts, or a divisor of it, and aligned to node boundaries so the bulk of routing stays on NVLink.
- **PR-MoE and Mixture-of-Students.** DeepSpeed-MoE also ships architectural ideas — Pyramid-Residual MoE puts more experts in later layers where they help most, and a distillation recipe ("Mixture-of-Students") compresses the trained MoE for inference.

```python
import deepspeed
import torch.nn as nn

moe_layer = deepspeed.moe.layer.MoE(
    hidden_size=4096,
    expert=nn.Sequential(nn.Linear(4096, 16384), nn.GELU(),
                         nn.Linear(16384, 4096)),
    num_experts=64,
    ep_size=8,
    k=1,
    capacity_factor=1.25,
    eval_capacity_factor=2.0,
    use_residual=False,
)
```

The honest tradeoff table for MoE:

| You gain | You pay |
|---|---|
| 5–10× parameters at ~constant FLOPs per token | Two all-to-all collectives per MoE layer |
| Higher quality at fixed inference compute | Memory for all $E$ experts, even idle ones |
| Cheap capacity scaling | Token dropping when the router is unbalanced |
| Strong results on knowledge-heavy tasks | A load-balancing auxiliary loss you must tune |

The load-balancing loss deserves its own warning. Left to itself, a router will collapse — it learns to send almost every token to a handful of favorite experts, leaving the rest untrained. DeepSpeed adds an auxiliary load-balancing loss that penalizes imbalance; its coefficient (often ~0.01) is a real hyperparameter. Too small and the router collapses; too large and it dominates the language-modeling objective and quality suffers. Watch per-expert token counts in your logs the way you watch loss — a healthy MoE has experts within ~2× of each other, a sick one has a 20× spread.

### Second-order optimization: MoE memory is not free even when FLOPs are

The seductive MoE pitch — "more parameters, same compute" — quietly omits memory. All $E$ experts occupy GPU (or offloaded) memory whether or not a given step routes tokens to them. A 64-expert model is a 64-expert model in the memory budget even if top-1 routing means 63 of them are idle for any single token. This is why expert parallelism is mandatory rather than optional at scale, and why DeepSpeed-MoE composes with ZeRO: ZeRO shards the *non-expert* parameters and optimizer states while expert parallelism shards the experts. Get the two communication groups wrong — overlap them, or misalign expert parallelism with node boundaries — and the all-to-all spills onto the slow network and the whole model crawls.

## 7. DeepSpeed for inference

**Senior rule of thumb: DeepSpeed has two inference engines for two unrelated problems. Picking the wrong one means either leaving latency on the table or OOMing a model that could have fit.**

Training is where DeepSpeed earns its reputation, but it has a serious inference story — and it is genuinely two different products under one name.

![Decision graph: model fits in GPU memory leads to DeepSpeed-Inference; does not fit leads to ZeRO-Inference](/imgs/blogs/deepspeed-zero-3d-parallelism-deep-dive-9.png)

**DeepSpeed-Inference** is the latency-first path, for when the model *fits* in aggregate GPU memory. It does two things: it injects **fused, hand-optimized transformer kernels** (fused QKV projection, fused attention, fused bias+activation) that cut kernel-launch overhead and memory round-trips, and it applies **tensor-slicing** parallelism across GPUs to split a model that fits in aggregate but not on one card. The newer evolution of this is **DeepSpeed-Inference / FastGen**, which adds dynamic batching and SplitFuse-style scheduling.

```python
import deepspeed
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-70B")

engine = deepspeed.init_inference(
    model,
    tensor_parallel={"tp_size": 4},   # slice across 4 GPUs
    dtype=torch.bfloat16,
    replace_with_kernel_inject=True,  # swap in fused kernels
)
output = engine.generate(input_ids, max_new_tokens=256)
```

**ZeRO-Inference** is the opposite problem: the model does *not* fit, and you care about throughput, not per-token latency. It reuses the ZeRO-Infinity machinery — parameters live on CPU or NVMe and are streamed layer-by-layer into the GPU as the forward pass walks the model. One GPU can run a 100B+ model. It is slow per token, but it lets you run offline batch inference, evals, or distillation data generation on hardware that could never hold the model. For where this fits among the broader serving landscape, see [serving LLMs at scale](/blog/machine-learning/large-language-model/serving-llms-at-scale-production-systems) and [efficient LLM inference techniques](/blog/machine-learning/large-language-model/efficient-llm-inference-techniques).

| | DeepSpeed-Inference | ZeRO-Inference |
|---|---|---|
| Problem it solves | Latency, when the model fits | Capacity, when it does not |
| Where weights live | GPU HBM | CPU DRAM / NVMe, streamed in |
| Optimizes for | Tokens/sec/user, low latency | Total throughput, batch jobs |
| Key mechanism | Fused kernels + tensor slicing | Layer-by-layer weight streaming |
| Use it for | Online serving | Offline eval, batch generation |

A note on the honest competitive picture: for pure online LLM serving in 2026, dedicated inference engines like [vLLM](/blog/machine-learning/large-language-model/vllm-inference) and [SGLang](/blog/machine-learning/large-language-model/sglang-inference) — with PagedAttention, continuous batching, and prefix caching — generally beat DeepSpeed-Inference on throughput per dollar. DeepSpeed's inference path is most compelling when you are *already* a DeepSpeed shop and want one toolchain, or when ZeRO-Inference's stream-from-NVMe trick is the only way to run the model at all. Pick tools per workload, not per vendor.

## 8. Launching a DeepSpeed job: the config, the launcher, and multi-node

**Senior rule of thumb: the DeepSpeed config file is the program. The Python is glue. If a behavior surprises you, the answer is in the JSON, not the traceback.**

Everything above is configured through one JSON file, and it is worth understanding how its pieces compose because the most baffling DeepSpeed bugs are config-arithmetic bugs. The three batch-size fields are tied by an identity DeepSpeed enforces at startup:

$$\texttt{train\_batch\_size} = \texttt{train\_micro\_batch\_size\_per\_gpu} \times \texttt{gradient\_accumulation\_steps} \times \texttt{world\_size}$$

You may specify any two and let DeepSpeed infer the third; specify all three inconsistently and it aborts immediately. This is a feature — it makes the global batch size explicit and reproducible across cluster sizes — but it trips up everyone the first time, because changing the GPU count silently changes the global batch size unless you also adjust accumulation. A reproducible recipe pins `train_batch_size` and `train_micro_batch_size_per_gpu` and lets `gradient_accumulation_steps` float with the cluster.

The launcher is `deepspeed`, a thin wrapper over a process-spawning mechanism similar to `torchrun`. For a single node:

```bash
deepspeed --num_gpus 8 train.py \
    --deepspeed --deepspeed_config ds_config.json
```

For multi-node, you give it a hostfile — an MPI-style list of hosts and their GPU slot counts — and DeepSpeed handles the SSH fan-out, rank assignment, and rendezvous. The `hostfile` itself is plain text:

```
node-0 slots=8
node-1 slots=8
node-2 slots=8
node-3 slots=8
```

```bash
deepspeed --hostfile=hostfile --master_addr=node-0 train.py \
    --deepspeed --deepspeed_config ds_config.json
```

DeepSpeed picks up `NCCL_*` environment variables and passes them to every rank, which matters because multi-node NCCL tuning — `NCCL_IB_HCA` to select the right InfiniBand device, `NCCL_SOCKET_IFNAME` to pin the control-plane interface, `NCCL_NET_GDR_LEVEL` for GPUDirect RDMA — is frequently the difference between a job that scales linearly and one that plateaus at 60% efficiency past one node. If you add a node and throughput per GPU drops, the first suspect is always NCCL routing, not DeepSpeed.

Two more config blocks earn their keep on real jobs. **Autotuning** lets DeepSpeed sweep ZeRO stage, bucket sizes, and micro-batch for you:

```json
{
  "autotuning": {
    "enabled": true,
    "fast": false,
    "metric": "throughput",
    "num_tuning_micro_batch_sizes": 3
  }
}
```

It launches a series of short trial runs and reports the fastest config it found — genuinely useful for a new model on new hardware, and far better than guessing bucket sizes. And the **`flops_profiler`** block prints a per-module FLOPs, latency, and parameter breakdown, which is how you actually find the straggler in case study 5 rather than guessing:

```json
{
  "flops_profiler": {
    "enabled": true,
    "profile_step": 10,
    "module_depth": -1,
    "top_modules": 3,
    "detailed": true
  }
}
```

### Second-order optimization: the config drift problem

A DeepSpeed config tends to accumulate cruft. Someone bumps a bucket size to debug an OOM and never reverts it (case study 8). Someone copies a config from a 70B job to a 7B job and inherits offload settings the smaller model never needed. Treat the config as code: keep it in version control, comment *why* each non-default value is set (a `_comment` key is ignored by DeepSpeed), and re-run autotuning whenever the model size or hardware changes. A config that was optimal for the job it was born in is rarely optimal for the job it was copied into.

## DeepSpeed vs FSDP vs Megatron-LM: choosing the engine

DeepSpeed does not exist in a vacuum, and in 2026 the honest question is not "how do I use DeepSpeed" but "which sharding engine fits this job." Three serious options compete.

**PyTorch FSDP / FSDP2** is PyTorch's native Fully Sharded Data Parallel. Conceptually it *is* ZeRO-3 — it shards parameters, gradients, and optimizer states, and all-gathers parameters per module just-in-time. The appeal is that it is in-tree: no extra dependency, tracks new PyTorch features immediately, composes cleanly with `torch.compile` and the native device mesh. FSDP2 (the `fully_shard` API) improved the memory model and per-parameter sharding. For a greenfield job that needs ZeRO-3-style sharding and nothing exotic, FSDP is now the default-reasonable choice.

**Megatron-LM** is NVIDIA's library, built around best-in-class **tensor parallelism** and pipeline parallelism with hand-tuned CUDA kernels. If your bottleneck is intra-layer parallelism on NVLink-rich nodes, Megatron's tensor-parallel implementation is the reference. It is less ergonomic than DeepSpeed for the ZeRO use case and historically required more model-side integration.

**DeepSpeed** still wins decisively in two areas. First, **offload depth**: nothing else has ZeRO-Infinity's mature CPU+NVMe three-tier offload. If you must train a model that does not fit even in aggregate GPU memory, DeepSpeed is effectively the only option. Second, **all-in-one breadth**: ZeRO + offload + 3D parallelism + MoE + compression + inference under one config-driven interface, battle-tested at 530B+ scale.

| Need | Best fit |
|---|---|
| ZeRO-3-style sharding, greenfield, PyTorch ≥ 2.x | FSDP2 |
| Train a model too big for aggregate GPU memory | DeepSpeed (ZeRO-Infinity) |
| Maximum tensor-parallel throughput on NVLink nodes | Megatron-LM (or Megatron-DeepSpeed) |
| 3D parallelism at 100B–1T scale, one toolchain | DeepSpeed |
| Mixture-of-Experts training with expert parallelism | DeepSpeed-MoE or Megatron-DeepSpeed |
| Tightest `torch.compile` integration | FSDP2 |

The pragmatic synthesis the largest labs reached: **Megatron-DeepSpeed**, a fork that uses Megatron's tensor/pipeline parallelism *and* DeepSpeed's ZeRO and offload together. It is not a cop-out — it is the recognition that these libraries optimize different axes and the axes compose. Pick per workload; the worst outcome is religious attachment to one engine for a job it does not fit.

## Case studies from production

Every one of these is a pattern I have watched a real team hit. Names and exact numbers are composited, but the symptom, the wrong first guess, and the fix are faithful.

### 1. OOM at step zero despite "it fits on paper"

A team sized a 20B model for an 8× A100-80GB node. Their math: $20\text{B} \times 16 = 320$ GB of state, $/8 = 40$ GB per GPU under ZeRO-3, half the card free. They launched and it OOMed before the first optimizer step completed. The wrong hypothesis was a ZeRO bug — "the partitioning isn't working." It was working perfectly. They had budgeted the $16N$ model state and *completely omitted activations*. Their config was sequence length 8192, micro-batch 8, no activation checkpointing — and the activation memory for that configuration was over 45 GB per GPU on its own. The fix was two lines: enable `activation_checkpointing` and drop the micro-batch from 8 to 2, leaning on `gradient_accumulation_steps` to keep the global batch size. The job ran with room to spare. The lesson is the mental-model figure: the $16N$ is the part you can compute in your head, which is exactly why you forget the part you cannot — activations do not scale with $N$, they scale with what you feed the model.

### 2. ZeRO-3 three times slower than ZeRO-2

A team moved a 7B fine-tune from ZeRO-2 to ZeRO-3 expecting "more memory, same speed." Throughput fell from 3,100 to 1,050 tokens/sec — a 3× regression. They suspected a misconfigured prefetch bucket and spent two days tuning `stage3_prefetch_bucket_size` with no effect. The actual cause was the interconnect. The job ran on a cloud instance type where the 8 GPUs were connected by **PCIe only — no NVLink, no NVSwitch**. ZeRO-3's per-layer all-gathers (section 5) need fast GPU-to-GPU bandwidth to hide behind compute; on PCIe they could not be hidden, and the GPUs spent two-thirds of every step idle waiting for parameters. The "3× the communication of DDP" tax came due in full. The fix was not a config tweak — it was recognizing that on that hardware, ZeRO-2 (which keeps DDP's communication volume) was the *correct* stage, and the 7B model fit under ZeRO-2 anyway. They reverted to ZeRO-2 and got their throughput back. The lesson: the right ZeRO stage is a function of your interconnect, not just your model size. Check `nvidia-smi topo -m` before you pick a stage.

### 3. ZeRO-Offload thrashing the host

A single-GPU fine-tune of a 13B model used `offload_optimizer: cpu` and ran, but at a quarter of the expected speed, and the box's load average was pinned. The first guess was "the GPU is the bottleneck, it's just a big model." It was not — `nvidia-smi` showed the GPU at 30% utilization. The CPU Adam kernel was the bottleneck, and specifically it was contending with the dataloader. The team had set `num_workers=16` on a 16-core machine; CPU Adam wanted those same 16 cores for the optimizer step, and each step the two fought, context-switched, and stalled. On a dual-socket box the optimizer state had also landed on the wrong NUMA node. The fix was a combination: drop `num_workers` to 8 to leave cores for CPU Adam, pin the process with `numactl --membind=0 --cpunodebind=0`, and confirm `pin_memory: true` was set so the gradient transfer could overlap. Throughput more than doubled. The lesson: when you offload, the CPU joins the critical path — you now have to capacity-plan the host the way you capacity-plan the GPU.

### 4. Loss spikes after switching to FP16

A pretraining run that was stable in FP32 started spiking — loss would jump by 5–10× every few hundred steps, sometimes recovering, sometimes diverging — after the team enabled `"fp16": {"enabled": true}` for speed. The wrong hypothesis was a bad data shard. The cause was FP16's narrow exponent range: gradients underflowed to zero in the small-magnitude tail and occasionally a large activation overflowed to `inf`. DeepSpeed's dynamic loss scaler was reacting, but the team had set an aggressive `initial_scale_power` and a short `loss_scale_window`, so the scaler oscillated instead of settling. Two fixes, either of which worked: tune the loss scaler (`"initial_scale_power": 16`, `"loss_scale_window": 1000`, a higher `min_loss_scale`) so it adapts smoothly; or — far simpler on A100/H100 — switch to `"bf16": {"enabled": true}`. BF16 has FP32's exponent range, so underflow and overflow essentially stop being a concern, at the cost of fewer mantissa bits (handled by the FP32 master copy, section 1). They moved to BF16 and the spikes vanished. The lesson: FP16 is a precision *and* a range gamble; BF16 trades a precision concern you have already solved for a range concern you have not.

### 5. Pipeline parallelism with one straggler stage

A 4-stage pipeline-parallel job ran at roughly half the throughput a back-of-envelope FLOP estimate predicted. Profiling each stage showed stage 0 finishing its micro-batch in a fraction of the time of stages 1–3, then idling. The team had used `partition_method="parameters"`, and stage 0 held the token embedding — huge parameter count, almost no FLOPs — plus only a couple of transformer blocks, while the other stages held many FLOP-heavy blocks each. Balanced by parameters, wildly imbalanced by *time*. Because a pipeline runs at the speed of its slowest stage and stage 0 was now permanently waiting, half the cluster was idle. The fix was to repartition by compute: move more transformer blocks onto stage 0 to equalize wall-clock per stage, which DeepSpeed supports via `partition_method="type:TransformerBlock"` or an explicit manual partition. Throughput climbed back to the estimate. The lesson from section 4's second-order note, paid for in production: balance pipeline stages by time, never by parameter count.

### 6. NVMe offload far slower than the SSD spec sheet

A ZeRO-Infinity job offloading to NVMe ran at a third of the throughput the team expected from their drive's rated 7 GB/s. They assumed the SSD was throttling thermally. It was not — the drive was cool and the `aio` configuration was simply untuned. They had left the default `block_size` of 256 KB and `queue_depth` of 4, which on their particular enterprise NVMe drive left most of the device's parallelism unused. They ran DeepSpeed's `aio_bench` sweep and found the drive saturated at `block_size: 1 MB`, `queue_depth: 8`, `thread_count: 2`. There was a second problem: `nvme_path` pointed at a *network-attached* volume, not local disk, so every "NVMe" read was actually a network round-trip. Repointing to genuinely local NVMe and applying the tuned `aio` block roughly tripled throughput. The lesson: under ZeRO-Infinity the SSD is a first-class memory tier — you must benchmark and tune it like one, and you must verify `nvme_path` is local physical storage, not a mount that looks local.

### 7. Checkpoint size explosion and a load that would not load

A ZeRO-3 job wrote checkpoints fine but they were enormous — and worse, a teammate could not load one for inference; `from_pretrained` choked on the format. Two confusions in one. First, ZeRO-3 saves a **sharded** checkpoint: each rank writes only its parameter and optimizer-state shard, in DeepSpeed's own layout. That is correct and necessary for resuming training, but it is not a Hugging Face model directory. To get a normal consolidated FP32 model you run the `zero_to_fp32.py` script DeepSpeed writes *into the checkpoint directory* for exactly this purpose:

```bash
python checkpoints/global_step10000/zero_to_fp32.py \
    checkpoints/global_step10000 \
    consolidated/pytorch_model.bin
```

Second, the "explosion": their checkpoint included full optimizer states ($12N$) because they were saving a *training* checkpoint, which is correct for resumption but is 8× the size of the model. For a checkpoint meant only for inference or release, they set `stage3_gather_16bit_weights_on_model_save: true` and used `engine.save_16bit_model()`, which writes just the $2N$ of weights. The lesson: ZeRO-3 has two kinds of checkpoint — the sharded full-state one you resume training from, and the consolidated weights-only one you ship — and conflating them produces either an unusable file or a 200 GB artifact where a 25 GB one would do.

### 8. Lost overlap from a mistuned prefetch bucket

A well-provisioned cluster — NVLink, fast InfiniBand — ran a ZeRO-3 job that still showed periodic GPU idle gaps in the profiler, sawtoothing utilization between 95% and 40%. ZeRO-3's communication *should* have hidden completely behind compute on this hardware. The team had, while debugging an unrelated OOM, dropped `stage3_prefetch_bucket_size` to a very small value to shave a little memory. That starved the prefetcher: DeepSpeed could no longer gather the next layers far enough ahead, so each layer's compute occasionally had to *wait* for its all-gather instead of finding the weights already present. The memory the small bucket saved was a few hundred MB; the throughput it cost was 15%. They raised `stage3_prefetch_bucket_size` and `reduce_bucket_size` back to ~5e8, confirmed `overlap_comm: true`, and the sawtooth flattened. The lesson: the ZeRO-3 bucket-size knobs are an overlap-versus-memory trade, and shrinking them to win back a sliver of memory can silently cost you far more in throughput. Tune them with the profiler open, not in isolation.

### 9. Throughput plateau the moment a second node joined

A ZeRO-3 job scaled beautifully to 8 GPUs on one node — near-linear — then fell off a cliff when extended to 4 nodes: 32 GPUs delivered the throughput of roughly 19. The first hypothesis was the classic "ZeRO-3 communication does not hide" of case study 2. But the interconnect was 400 Gb/s InfiniBand — plenty. Profiling the NCCL collectives showed inter-node all-gathers running at a fraction of the line rate. The cause was NCCL routing: the cluster had multiple network interfaces, and NCCL had auto-selected a slow management Ethernet interface for some of its traffic instead of the InfiniBand HCAs. Nothing in DeepSpeed was wrong. The fix was environment variables passed through the launcher: `NCCL_IB_HCA` to name the InfiniBand devices explicitly, `NCCL_SOCKET_IFNAME` to pin the bootstrap interface, and `NCCL_NET_GDR_LEVEL=2` to enable GPUDirect RDMA so the NIC could DMA straight from GPU memory. Throughput returned to near-linear. The lesson: when scaling past one node, the suspect is almost always NCCL network selection, not the training framework — verify with `NCCL_DEBUG=INFO` that the collectives are riding the fast fabric before you touch anything in the DeepSpeed config.

### 10. The MoE router that collapsed onto four experts

A 64-expert DeepSpeed-MoE pretraining run showed validation loss plateauing well above where a dense model of equivalent active-parameter count sat. The team suspected the data mix. The actual problem surfaced in the per-expert token-count logs, which they had not been watching: of 64 experts, four were receiving roughly 70% of all tokens and a dozen were receiving almost none. The router had collapsed — early in training it found a few experts marginally better, sent them more tokens, which trained them faster, which made them better still, a runaway feedback loop. The load-balancing auxiliary loss was in the config but its coefficient had been set an order of magnitude too low. Most of the model's parameters were effectively untrained dead weight. The fix was to raise the load-balancing loss coefficient to ~0.01, restart, and watch the per-expert counts converge to within ~2× of each other over the first few thousand steps. The lesson: an MoE's expert utilization is a first-class training metric — log it, alert on it, and treat a widening spread the way you treat a diverging loss.

### 11. Autotuning picked a config that OOMed during evaluation

A team ran DeepSpeed autotuning, took its recommended config — a micro-batch size that maximized training throughput — and launched a multi-day job. It trained fine for hours, then OOMed during the periodic in-loop evaluation. The wrong hypothesis was a memory leak, since training had been stable. The cause was that autotuning had measured only the *training* step's memory and picked a micro-batch that left little headroom. Evaluation ran with a longer maximum sequence length and `eval_capacity_factor` set higher than the training capacity factor — both legitimate, both raising activation and expert-buffer memory above what autotuning had ever exercised. The config was optimal for the step autotuning measured and unsafe for the step it never saw. The fix was to autotune against the *worst-case* sequence length and batch shape the job would ever encounter, not the common-case training shape, and to leave an explicit memory margin. The lesson: autotuning optimizes the workload you show it. If evaluation, the longest document in your corpus, or a curriculum that ramps sequence length will stress memory harder than the training step you profiled, autotune against that, or you have tuned for a job you are not actually running.

### 12. A checkpoint that would not resume on a resized cluster

A job checkpointed under ZeRO-3 on 64 GPUs. A week later the team had only 48 GPUs free and tried to resume — and DeepSpeed refused to load the checkpoint cleanly. The instinct was that the checkpoint was corrupt. It was not. A ZeRO-3 sharded checkpoint is partitioned *across a specific world size*: 64 shards of optimizer state, each written by one rank. Resuming on 48 ranks means the shards no longer map one-to-one onto processes. DeepSpeed supports **universal checkpoints** precisely for this — a conversion step that consolidates the world-size-specific sharded checkpoint into a world-size-agnostic format, which can then be re-sharded onto any cluster size on load. The team ran the universal-checkpoint conversion, resumed on 48 GPUs, and continued training with optimizer state intact. The lesson: a ZeRO sharded checkpoint is coupled to the cluster shape that wrote it. If there is any chance you will resume on a different number of GPUs — and on shared or spot infrastructure there always is — either convert to a universal checkpoint at save time or budget the conversion step into your resume runbook.

### 13. Activation checkpointing that made the job slower without saving memory

A long-context fine-tune (sequence length 32k) enabled `activation_checkpointing` to fit, and it did fit — but throughput dropped 35% and the team accepted it as the cost of the long context. Months later a profiler pass showed something worse: the job was checkpointing *every* layer, recomputing the full forward of all of them in backward, paying the maximum recompute tax. And it did not need to. The model had comfortable GPU headroom once the global batch was set — they had copied `number_checkpoints` and the every-layer interval from a config built for a far tighter memory budget. Activation checkpointing is a *dial*, not a switch: checkpoint every layer and you cut activation memory hardest but pay a full extra forward; checkpoint every fourth layer and you keep more activations resident but recompute far less. They re-tuned to checkpoint roughly every fourth transformer block — still fitting the 32k context with margin — and recovered most of the lost throughput. The lesson: `activation_checkpointing` is not free memory, it is memory bought with compute, and the exchange rate is yours to set. Checkpoint exactly as aggressively as the memory budget forces and no more; an every-layer checkpoint config on a job with headroom is burning ~33% of forward FLOPs for nothing.

## When to reach for DeepSpeed — and when not to

DeepSpeed is a precision instrument. It is the right tool with a clear shape.

**Reach for DeepSpeed when:**

- Your model state ($16N$ bytes) does not fit on a single GPU and you need to *train*, not just infer. This is the core use case and nothing does it more maturely.
- You are training at the 10B–1T parameter scale and need to combine ZeRO with pipeline and tensor parallelism — the 3D parallelism story is battle-tested at the largest scales that exist.
- You must train or fine-tune a model that genuinely cannot fit even in aggregate GPU memory, and ZeRO-Infinity's CPU/NVMe offload is the only path. Slow-but-possible beats impossible.
- You want a memory win for *free*: ZeRO-1 on top of an existing DDP job shards the optimizer states with zero communication overhead. Almost nobody turns this on; almost everybody should.
- You are already invested in the DeepSpeed config-driven workflow and want one engine for training and inference.

**Skip DeepSpeed when:**

- Your model fits comfortably on one GPU. Plain DDP or single-GPU training is simpler, has fewer failure modes, and will not be slower. ZeRO's machinery is pure overhead when there is no redundancy to remove.
- You are doing pure online LLM *serving*. Reach for [vLLM](/blog/machine-learning/large-language-model/vllm-inference) or [SGLang](/blog/machine-learning/large-language-model/sglang-inference) — purpose-built inference engines beat DeepSpeed-Inference on throughput per dollar for that workload.
- You are starting fresh on PyTorch ≥ 2.x and want native, well-supported sharding. PyTorch's own FSDP (and FSDP2) covers the ZeRO-3 use case without a separate dependency — see the [TorchTitan deep-dive](/blog/machine-learning/open-source-library/torchtitan-pytorch-native-pretraining-deep-dive). DeepSpeed still wins on offload depth and 3D-parallelism maturity, but for plain sharding the gap has narrowed.
- Your interconnect is slow (PCIe-only nodes, Ethernet between hosts) and your model fits under ZeRO-2. ZeRO-3's extra communication will not hide and you will regress throughput for memory you did not need — case study 2.
- Your global batch size is small. Pipeline parallelism's bubble will dominate and the parallelism you paid for will sit idle.

The thread running through all of it: DeepSpeed does not give you anything for free. Every feature is a trade — memory for communication, GPU memory for PCIe bandwidth, GPU memory for SSD bandwidth, compute for activation memory, throughput for the ability to run at all. The engineers who succeed with DeepSpeed are the ones who name the trade before they flip the flag, then measure that they got the side of it they wanted. The ones who get burned enable a feature for the headline number and never look at what it cost.

## Further reading

- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) — the original paper; the byte arithmetic in section 1 comes straight from it.
- [ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://arxiv.org/abs/2104.07857) — the NVMe-tier design of section 3.
- [DeepSpeed documentation and tutorials](https://www.deepspeed.ai/) — the canonical config reference.
- [TorchTitan: PyTorch-native pretraining](/blog/machine-learning/open-source-library/torchtitan-pytorch-native-pretraining-deep-dive) — the FSDP-native alternative to ZeRO-3.
- [Optimizing MoE training and inference](/blog/machine-learning/large-language-model/optimizing-moe-training-and-inference) — where DeepSpeed-MoE fits.
- [Choosing a GPU for LLM serving](/blog/machine-learning/large-language-model/choosing-gpu-for-llm-serving-cost-throughput-latency) — the interconnect and memory tradeoffs that decide your ZeRO stage.
