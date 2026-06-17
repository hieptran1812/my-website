---
title: "Memory Optimization: ZeRO, FSDP, Activation Checkpointing, and Offload"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Learn exactly where the 80 GB on a GPU goes and how to fit a model that does not, by deriving the 16-bytes-per-parameter rule and then sharding it away with ZeRO, FSDP, checkpointing, and offload."
tags:
  [
    "high-performance-computing",
    "gpu",
    "zero",
    "fsdp",
    "activation-checkpointing",
    "offload",
    "distributed-training",
    "deep-learning",
    "ml-systems",
    "pytorch",
    "deepspeed",
  ]
category: "machine-learning"
subcategory: "High Performance Computing"
author: "Hiep Tran"
featured: true
readTime: 53
image: "/imgs/blogs/memory-optimization-zero-fsdp-activation-checkpointing-and-offload-1.png"
---

You launch a fine-tune of a 13-billion-parameter model on a node of eight 40 GB A100s, the run gets through exactly one step, and then `torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB` fills your terminal. You drop the batch size to 1. Same error. You enable gradient accumulation. Same error. You start to suspect the hardware is broken, because surely 320 GB of total GPU memory is enough for a 13B model that is "only" 26 GB in bf16. It is not broken, and the gap between "the model is 26 GB" and "the run needs 226 GB" is the entire subject of this post. The arithmetic that closes that gap, and the four techniques that bend it back under the line, are the difference between an engineer who shrugs at an OOM and one who knows, before launching, exactly how many GPUs a model needs.

Here is the one-sentence version of everything below. Training a model in mixed precision with the Adam optimizer costs **16 bytes for every parameter** before you store a single activation — 2 bytes for the bf16 parameter, 2 for its gradient, and 12 for the optimizer's fp32 master copy plus its two running averages — so a 7B model needs about **112 GB just for those states**, which is why it will not fit on an 80 GB GPU no matter how small your batch is. The fix is not a bigger GPU; it is to stop replicating that 16-bytes-per-parameter cost on every device and instead **shard it** across your data-parallel ranks. That single idea, plus recomputing activations instead of storing them and spilling cold state to the CPU, is the whole toolkit. Once you can do the 16Ψ arithmetic in your head, every memory decision — which ZeRO stage, whether to checkpoint, whether to offload — falls out of it.

![layered stack showing parameters at 2 bytes grads at 2 bytes the three Adam states at 4 bytes each and activations on top summing to 16 bytes per parameter plus activations](/imgs/blogs/memory-optimization-zero-fsdp-activation-checkpointing-and-offload-1.png)

The figure above is the memory budget for one parameter, stacked. The blue band at the bottom is what most people think of as "the model" — the bf16 weights and their gradients, 4 bytes total. The amber bands above it are the part nobody pictures until they hit an OOM: the optimizer's fp32 master copy and Adam's two moment estimates, another 12 bytes. The red band on top is activations, the one piece that scales with your batch and sequence length rather than your parameter count. By the end of this post you will be able to read that stack, multiply it by your parameter count, divide by your GPU count under each ZeRO stage, and predict your peak memory before you ever type `torchrun`. This is the memory wall in the three-walls frame from [why HPC is the bottleneck for modern AI](/blog/machine-learning/high-performance-computing/why-hpc-is-the-bottleneck-for-modern-ai): the model is a schedule of arithmetic over bytes, and when the bytes do not fit, no amount of compute helps.

We will build it in layers. First the exact 16Ψ derivation, term by term, so the number is provable and not memorized. Then **ZeRO** — the idea of sharding optimizer states, then gradients, then parameters across the data-parallel group, in three stages. Then **FSDP**, PyTorch's native implementation of the same idea, with real wrapping code. Then **activation checkpointing**, which trades a little compute to recover the one piece sharding does not touch. Then **CPU and NVMe offload** for when even a 1/N shard is too big. Throughout, our running example is the Transformer from earlier posts — a 7B and a 13B LLM — and we end by fitting that 13B model onto eight 40 GB GPUs with numbers you can reproduce.

## Where the 80 GB actually goes: deriving the 16Ψ rule

Let me define the four consumers of GPU memory precisely before deriving anything, because the whole post depends on telling them apart.

**Parameters** are the model's learnable weights — the matrices in every linear layer, the embedding table, the layernorm gains and biases. We will call the total parameter count $\Psi$ (the Greek letter psi, following the ZeRO paper's notation). A 7B model has $\Psi = 7 \times 10^9$. **Gradients** are the derivative of the loss with respect to each parameter, computed during the backward pass; there is exactly one gradient per parameter, so gradients also number $\Psi$. **Optimizer states** are the extra per-parameter numbers the optimizer keeps between steps — for plain SGD that is nothing, but for **Adam**, the optimizer that trains essentially every modern Transformer, it is three numbers per parameter, which we will derive below. **Activations** are the intermediate tensors produced during the forward pass — the output of every attention block, every feed-forward layer, every layernorm — that must be kept around because the backward pass needs them to compute gradients. Activations are the odd one out: they scale with batch size and sequence length, not with $\Psi$, and they are the piece checkpointing attacks.

Now the derivation. We are training in **mixed precision**, the standard recipe since 2017: the forward and backward passes run in bf16 (or fp16) to use the Tensor Cores at full speed, but the optimizer maintains an fp32 "master copy" of the weights to avoid the slow accumulation of rounding error that would otherwise corrupt training. (If you have not internalized why bf16 is safe for the compute but fp32 is needed for the weight update, the companion post on [numerical formats and mixed precision](/blog/machine-learning/high-performance-computing/numerical-formats-and-mixed-precision-fp32-tf32-bf16-fp16-fp8) derives exactly that from the bit layout.) Count the bytes per parameter:

- The **bf16 parameter** the forward pass reads: 2 bytes.
- The **bf16 gradient** the backward pass writes: 2 bytes.
- The **fp32 master copy** the optimizer updates: 4 bytes.
- Adam's **first moment** $m$ (the running mean of the gradient, fp32): 4 bytes.
- Adam's **second moment** $v$ (the running mean of the squared gradient, fp32): 4 bytes.

Sum: $2 + 2 + 4 + 4 + 4 = 16$ bytes per parameter. The model state for mixed-precision Adam is

$$M_\text{states} = (2 + 2 + 12)\,\Psi = 16\,\Psi \;\text{bytes},$$

where the 12 is the optimizer's share: 4 (master) + 4 ($m$) + 4 ($v$). This is the famous **16Ψ rule**, and it is the single most useful number in distributed training. Notice that *three quarters of it* — the 12 bytes — live in the optimizer, which the forward pass never touches and which you only need once per step at update time. That observation is the entire seed of ZeRO: the most expensive state is also the least frequently used, so it is the first thing you should shard or offload.

#### Worked example: a 7B model will not fit an 80 GB GPU

Plug $\Psi = 7 \times 10^9$ into $16\Psi$:

$$M_\text{states} = 16 \times 7 \times 10^9 = 1.12 \times 10^{11}\ \text{bytes} = 112\ \text{GB}.$$

That is 112 GB of fixed model state, before a single activation, before the CUDA context's own overhead (PyTorch reserves roughly 1 GB just to exist, and the NCCL communication buffers want a few hundred MB more). An A100 80GB or an H100 80GB has 80 GB. So a 7B model trained with Adam in mixed precision **cannot fit on a single 80 GB GPU at any batch size**, including batch size zero. People are routinely surprised by this because the bf16 weights are only 14 GB and they reason "14 GB fits in 80 GB easily." The weights are the cheapest eighth of the bill. The optimizer is 84 GB of it. If you have ever tried to fine-tune a 7B model on a single A100 and watched it OOM before the first step, this is why, and the fix is structural, not a knob.

Compare that to the same model under ZeRO-3 across an 8-GPU node, which is where we are headed:

$$\frac{16 \times 7 \times 10^9}{8} = 14\ \text{GB per GPU}.$$

Now it fits with room to spare for activations. Same model, same optimizer, same precision — the only thing that changed is that we stopped storing all 112 GB on every GPU and instead gave each GPU a one-eighth slice. That is the whole game.

A few honest caveats on the 16 number. If you use a memory-leaner optimizer — Adafactor, 8-bit Adam (`bitsandbytes`), or Lion — the 12-byte optimizer share shrinks, and the rule changes accordingly. 8-bit Adam stores $m$ and $v$ in 1 byte each instead of 4, taking the optimizer share from 12 to 4 + 1 + 1 = 6 and the total from 16Ψ to 10Ψ — a real and often-overlooked win that composes with everything below. If you train in pure fp32 (rare now), parameters and gradients become 4 bytes each and there is no separate master copy, giving 4 + 4 + 4 + 4 + 4 = 20Ψ. And if you use plain SGD with momentum, the optimizer keeps only one extra buffer, not two. Throughout this post we assume the dominant case — mixed-precision bf16 with fp32 Adam — and the 16Ψ rule. When you change the recipe, recompute; the *method* is what transfers.

### Activations: the piece that scales with the batch, not the model

The 16Ψ covers everything that scales with parameter count. Activations are governed by a different formula. For a Transformer with $L$ layers, hidden size $h$, batch size $b$, and sequence length $s$, the activation memory stored for the backward pass is approximately

$$M_\text{act} \approx L \cdot b \cdot s \cdot h \cdot c \ \text{bytes},$$

where $c$ is a per-layer constant (counting the activations a Transformer block keeps — the attention scores, the residual-stream tensors, the feed-forward intermediates) that lands somewhere around 30–70 bytes per token-channel for a standard block in bf16, depending on whether you store attention scores and how the implementation fuses things. The crucial structural fact is the linear dependence on $L \cdot b \cdot s$: doubling the depth, the batch, or the sequence length doubles the activation memory, while the 16Ψ state does not budge.

This is why activation memory is the dominant term for long-context and large-batch training even when the parameter state is comfortably sharded, and why it gets its own technique (checkpointing) entirely separate from ZeRO. A concrete feel for the magnitude: a 13B-class model (40 layers, $h = 5120$) at batch 1, sequence 2048, in bf16, stores very roughly $40 \times 1 \times 2048 \times 5120 \times 40\ \text{bytes} \approx 17\ \text{GB}$ of activations *per microbatch* if you keep everything. That 17 GB is on top of the sharded state, on every GPU, and it is exactly the term that turns a "fits on paper" plan into an OOM. Hold both formulas — $16\Psi$ for state, $L\,b\,s\,h\,c$ for activations — because every technique below attacks one or the other.

### Where the parameters actually live, layer by layer

It is worth opening up $\Psi$ once, because "7 billion parameters" is an abstraction that hides where the bytes physically sit, and knowing the distribution tells you what sharding can and cannot help. For a standard decoder-only Transformer with hidden size $h$, $L$ layers, vocabulary $V$, and feed-forward expansion factor 4, the parameter count breaks down as follows. Each attention block has four projection matrices (query, key, value, output), each $h \times h$, for $4h^2$ parameters. Each feed-forward block has an up-projection $h \times 4h$ and a down-projection $4h \times h$, for $8h^2$ parameters. So each Transformer layer is about $12h^2$ parameters, and the $L$ layers together are $12Lh^2$. On top of that sits the embedding table, $V \times h$, which the output head usually ties or duplicates. For a 7B model with $h = 4096$, $L = 32$, $V = 32000$: the layers are $12 \times 32 \times 4096^2 \approx 6.4 \times 10^9$, and the embedding is $32000 \times 4096 \approx 1.3 \times 10^8$ — so the layers are the overwhelming majority and the embedding is a couple of percent.

Why does this matter for memory? Because every one of those $12Lh^2$ layer parameters carries the full 16-byte tax, and the count grows *quadratically* in $h$. Double the hidden size and the per-layer state quadruples; that is why widening a model is so much more expensive in memory than deepening it (deepening is linear in $L$). The embedding table is the one piece that scales with vocabulary rather than $h^2$, and for very large vocabularies (multilingual models with 256k tokens) it can become a non-trivial slice — which is why some training setups shard the embedding separately. But for the canonical English-vocab LLM, the rule of thumb is simple: the parameters are almost all in the layer matmuls, they grow as $h^2$, and the 16-byte tax applies uniformly to all of them. Sharding cuts that tax by $N$ regardless of where the parameter sits; it does not care whether a byte is an attention weight or an embedding.

#### Worked example: estimating state from a checkpoint file size

You can sanity-check the 16Ψ rule against a number you already have: the size of a saved model checkpoint on disk. Suppose you download a 7B model and its bf16 weights file is 14 GB. That 14 GB is exactly $2\Psi$ — the bf16 parameters, 2 bytes each — which immediately tells you $\Psi = 14\text{ GB} / 2 = 7\text{ billion}$, confirming the parameter count. Now multiply by 8 (the ratio of full training state to bf16 weights, since $16\Psi / 2\Psi = 8$): $14\text{ GB} \times 8 = 112\text{ GB}$ of training state. So a back-of-envelope you can do in your head at the download prompt is: **training state $\approx 8\times$ the bf16 checkpoint size.** A 14 GB checkpoint means 112 GB of state; a 26 GB checkpoint (13B) means 208 GB of state; a 140 GB checkpoint (70B) means 1.12 TB of state. This $8\times$ heuristic is the fastest way to know, before you allocate any hardware, roughly how many GPUs a fine-tune will need. Divide the $8\times$ figure by your per-GPU memory and round up — that is your minimum GPU count for the state alone, before activations.

## ZeRO: shard the 16Ψ instead of replicating it

Standard data-parallel training — PyTorch's `DistributedDataParallel`, or **DDP** — gives every GPU a *complete replica* of the model: full parameters, full gradients, full optimizer states, the whole 16Ψ. The only thing that differs across GPUs is the data; each processes a different microbatch, computes gradients locally, and then an **all-reduce** averages the gradients across all ranks so every replica stays in sync. DDP is simple and fast and scales beautifully on throughput — but it is monstrously wasteful on memory. With 8 GPUs you are storing the same 112 GB of state eight times over, for an aggregate 896 GB to hold one 7B model's state. You bought eight GPUs and used their memory as if you had one.

**ZeRO** — Zero Redundancy Optimizer, from Rajbhandari et al. 2020 (the DeepSpeed paper) — is the observation that this replication is pure waste, and that you can *partition* the 16Ψ across the data-parallel ranks so each GPU holds only $1/N$ of it, reconstructing the full tensors only momentarily when an operation needs them. **Sharding** is exactly this: splitting a tensor into $N$ contiguous slices and giving slice $i$ to rank $i$, so the aggregate is stored exactly once across the group rather than $N$ times. ZeRO does this in three stages, peeling off the most wasteful state first.

![matrix showing ZeRO stages zero through three sharding optimizer states then gradients then parameters with per-GPU state shrinking from 16 Psi to 16 Psi over N](/imgs/blogs/memory-optimization-zero-fsdp-activation-checkpointing-and-offload-2.png)

The matrix above is the whole ZeRO progression on one page. Read it top to bottom. Stage 0 is plain DDP: everything is full, per-GPU state is the whole 16Ψ. Each stage down shards one more column, and the per-GPU state in the last column shrinks accordingly. Think of it as draining the redundancy out of the system one reservoir at a time, biggest reservoir first.

**ZeRO Stage 1** shards only the optimizer states — the 12Ψ that we just established is three-quarters of the bill and the least-used piece. Each rank keeps the full bf16 parameters (2Ψ) and full gradients (2Ψ) but only its $1/N$ slice of the fp32 master copy, $m$, and $v$. Per-GPU state drops from $16\Psi$ to $4\Psi + 12\Psi/N$. At $N = 8$ that is $4\Psi + 1.5\Psi = 5.5\Psi$, already a 2.9× reduction. The optimizer update now happens in pieces: each rank updates only its slice of the parameters, then an all-gather circulates the updated bf16 weights so everyone is whole again for the next forward.

**ZeRO Stage 2** additionally shards the gradients. Once a parameter's gradient has been reduced and consumed by that parameter's optimizer slice (which lives on exactly one rank under Stage 1), there is no reason for every rank to keep a full copy of every gradient. Stage 2 replaces the gradient all-reduce with a **reduce-scatter**: instead of every rank ending up with the full averaged gradient, each rank ends up with only the averaged gradient *slice* it owns. Per-GPU state drops to $2\Psi + 14\Psi/N$. At $N = 8$ that is $2\Psi + 1.75\Psi = 3.75\Psi$, a 4.3× reduction, and critically it costs **no extra communication** over Stage 1 — reduce-scatter moves the same bytes as all-reduce's first half. Stage 2 is the sweet spot for many setups: large memory savings, DDP-equal communication volume.

**ZeRO Stage 3** shards the parameters themselves. Now nothing is fully replicated — each rank permanently holds only its $1/N$ slice of the bf16 parameters too, and the full weights of a layer exist on a GPU *only* during the brief window when that layer is computing. Per-GPU state drops to the theoretical floor:

$$M_\text{ZeRO-3 per GPU} = \frac{16\Psi}{N}.$$

This is the line we computed earlier: a 7B model goes from 112 GB to 14 GB on 8 GPUs. The cost is the extra communication of gathering parameters on the fly, which we quantify below. Stage 3 is what lets you train models *larger than any single GPU can hold* — the aggregate model lives across the cluster and no GPU ever sees the whole thing at rest.

Here is the savings table for our 7B running example, $N = 8$ GPUs, in GB, computed straight from the formulas above:

| Stage | Per-GPU formula | 7B at N=8 | Reduction vs DDP | Extra comm vs DDP |
|---|---|---|---|---|
| 0 (DDP) | $16\Psi$ | 112 GB | 1.0× | 1.0× |
| 1 | $4\Psi + 12\Psi/N$ | 38.5 GB | 2.9× | 1.0× |
| 2 | $2\Psi + 14\Psi/N$ | 26.25 GB | 4.3× | 1.0× |
| 3 | $16\Psi / N$ | 14 GB | 8.0× | 1.5× |

The pattern is worth stating plainly: each stage trades a little more communication complexity for a lot more memory savings, and Stage 3's savings scale with $N$ — the more GPUs you have, the smaller each one's slice. That is the property that makes trillion-parameter training possible: throw enough ranks at it and the per-GPU state goes to zero. We will see the communication cost is the catch, but the memory math is unambiguous.

### Why ZeRO-3 costs about 1.5× the communication of DDP

The science behind that "1.5×" in the table is worth deriving, because it is the price of Stage 3 and you should know what you are buying. DDP communicates exactly once per step: an all-reduce of the gradients. On a ring, an all-reduce of $S$ bytes moves $2(N-1)/N \cdot S \approx 2S$ bytes per GPU (the [collective-communication post](/blog/machine-learning/high-performance-computing/collective-communication-and-nccl-all-reduce-from-scratch) derives that ring-all-reduce cost from scratch). Call DDP's per-step communication volume $2S$ where $S$ is the model size in bytes.

ZeRO-3 communicates three times per step. In the forward pass, it **all-gathers** the parameters layer by layer — total volume $\approx S$ across the step (each rank receives the $(N-1)/N$ of every layer it does not own). In the backward pass it all-gathers the parameters *again* (they were freed after the forward to save memory, so they must be re-materialized) — another $\approx S$. And it does a **reduce-scatter** of the gradients — $\approx S$. That is $3S$ of communication versus DDP's $2S$, a ratio of exactly 1.5×. The DeepSpeed paper reports this same factor. So ZeRO-3 moves 50% more bytes than DDP, and whether that 50% *costs* you wall-clock time depends entirely on whether you can hide it behind compute — which is the overlap problem we take up later. On a fat NVLink/NVSwitch interconnect with good overlap, the 1.5× is nearly free; on PCIe or across slow inter-node links, it can dominate and erase the win. The interconnect is the deciding variable, every time.

## FSDP: PyTorch's native ZeRO-3

ZeRO lives in DeepSpeed, a separate library you configure with a JSON file. PyTorch has its own first-class implementation of the same Stage-3 idea, built into `torch.distributed`: **FSDP**, Fully Sharded Data Parallel. The mechanics are identical to ZeRO-3 — shard everything, all-gather a layer's parameters just before computing it, free them right after — but it is native PyTorch, composes with `torch.compile` and the rest of the ecosystem, and is what most new training code reaches for. Conceptually FSDP and ZeRO-3 are the same algorithm; the differences are API and library, not principle.

![graph showing FSDP holding a one over N shard at rest all-gathering full layer weights for forward and backward compute then freeing back to the shard and reduce-scattering gradients](/imgs/blogs/memory-optimization-zero-fsdp-activation-checkpointing-and-offload-3.png)

The graph above traces one layer through FSDP. At rest, the rank holds only its $1/N$ shard. When it is time to compute layer $i$, FSDP issues an **all-gather** to assemble the full layer weights from all ranks. It runs the forward (and later the backward) on the now-complete layer. Then — this is the key memory move — it immediately **frees** the full weights, dropping back to the $1/N$ shard, so the *peak* memory holds only one layer's full weights at a time, not the whole model's. Gradients flow out through a **reduce-scatter** so each rank keeps only its slice. The whole model never coexists in full on any GPU; only the layer currently executing does. That is how FSDP fits a model far larger than one GPU.

Here is the actual wrapping code. The unit of sharding is a *wrapping policy* — you tell FSDP which submodules to treat as the all-gather/free unit. For a Transformer, you wrap each Transformer block as its own FSDP unit so that exactly one block's weights materialize at a time:

```python
import functools
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed import init_process_group

# torchrun sets RANK, LOCAL_RANK, WORLD_SIZE for us.
init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

model = build_transformer(num_layers=40, hidden=5120)  # ~13B params, on CPU

# Shard at the granularity of one Transformer block: each block becomes
# one all-gather / free unit, so peak holds one block's full weights.
auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={TransformerBlock},
)

# bf16 compute, bf16 communication, fp32 reduction for numerical safety.
mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,   # reduce-scatter grads in fp32
    buffer_dtype=torch.bfloat16,
)

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # = ZeRO-3
    auto_wrap_policy=auto_wrap_policy,
    mixed_precision=mp_policy,
    device_id=torch.cuda.current_device(),
    limit_all_gathers=True,   # backpressure so prefetch does not OOM
)
```

The two lines that matter most for memory are `sharding_strategy=ShardingStrategy.FULL_SHARD` and the `auto_wrap_policy`. `FULL_SHARD` is the ZeRO-3 mode — parameters, gradients, and optimizer states all sharded — and it is the strategy you want for fitting large models. FSDP also offers `SHARD_GRAD_OP` (shard gradients and optimizer states but keep parameters replicated, i.e. ZeRO-2) for when the model fits but you want the optimizer savings, and `NO_SHARD`, which is plain DDP. The `transformer_auto_wrap_policy` is what makes the per-layer all-gather/free pattern happen: without a sensible wrap policy, FSDP shards the whole model as one giant unit, all-gathers everything at once, and your peak memory is no better than DDP. **The wrap granularity is the single most common FSDP mistake** — wrap too coarse and you get no memory savings; wrap too fine (every linear layer) and you drown in tiny all-gather launches that kill throughput. One Transformer block per unit is the right granularity for almost every LLM.

Note `MixedPrecision` with `reduce_dtype=torch.float32`. This is the FSDP equivalent of the fp32-master-copy discipline from mixed precision: the parameters and the all-gather move in bf16 (fast, half the bytes), but the gradient reduce-scatter accumulates in fp32 to avoid the precision loss that summing thousands of small bf16 gradients would incur. Keeping `param_dtype` bf16 but `reduce_dtype` fp32 is the safe default; flipping the reduction to bf16 saves a little communication but is a known source of subtle training instabilities at scale.

The launch is ordinary `torchrun`:

```bash
torchrun \
  --nproc_per_node=8 \
  --nnodes=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:29500 \
  train_fsdp.py --model 13B --seq 2048
```

### Reading peak memory honestly

You cannot optimize memory you do not measure, and the number that matters is not what `nvidia-smi` shows (that includes the caching allocator's reserved-but-unused blocks and the CUDA context). The number that matters is the **peak allocated** bytes — the high-water mark of actually-live tensors — which PyTorch tracks for you:

```python
import torch

torch.cuda.reset_peak_memory_stats()
loss = model(batch).loss
loss.backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)
torch.cuda.synchronize()  # let all kernels finish before reading

peak_gb = torch.cuda.max_memory_allocated() / 1e9
reserved_gb = torch.cuda.max_memory_reserved() / 1e9
print(f"peak allocated: {peak_gb:.1f} GB   reserved: {reserved_gb:.1f} GB")
```

Two disciplines make this trustworthy. First, the `torch.cuda.synchronize()` before reading: CUDA kernels are asynchronous, so without a sync you might read the high-water mark before the backward's allocations have actually happened. Second, measure over a *steady-state* step, not the first one — the first step pays one-time costs (the allocator warming up, cuDNN/cuBLAS picking algorithms, NCCL allocating its buffers) that inflate the peak. Run a couple of warm-up steps, call `reset_peak_memory_stats()`, then measure. The gap between `max_memory_allocated` (live tensors) and `max_memory_reserved` (what the allocator holds from the driver) is fragmentation; if reserved is much larger than allocated, setting the env var `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` often reclaims it.

#### Worked example: FSDP cuts a 13B model from OOM to 32 GB per GPU

Take the 13B model, eight 40 GB A100s, batch 1, sequence 2048. Under plain DDP the per-GPU state is $16 \times 13\text{B} = 208$ GB — five times the GPU's 40 GB, an instant OOM before step one. Under FSDP `FULL_SHARD` the state becomes $208 / 8 = 26$ GB per GPU. Add the activation term. At batch 1, seq 2048, this 40-layer model stores roughly 17 GB of activations per microbatch if everything is kept — pushing the total to $26 + 17 = 43$ GB, still over the 40 GB line. This is the moment most people give up and ask for 80 GB GPUs. But the 17 GB of activations is exactly what checkpointing attacks: with activation checkpointing the activation term drops to roughly 6 GB (we derive the $\sqrt{}$ saving next), bringing the total to $26 + 6 = 32$ GB — comfortably under 40 GB. So the recipe that fits 13B on eight 40 GB GPUs is **FSDP `FULL_SHARD` plus activation checkpointing**, and the budget is 32 GB per GPU with margin for the CUDA context and NCCL buffers. We will lay this out as a full table at the end; for now, notice that *two* techniques were needed — sharding alone got the state down but the activations still blew the budget. Memory optimization is almost always a combination, not a single switch.

### FSDP versus DeepSpeed: same idea, different ergonomics

Because FSDP and DeepSpeed ZeRO-3 implement the same algorithm, the choice between them is about ergonomics and ecosystem, not capability. FSDP is native to PyTorch — it lives in `torch.distributed`, composes with `torch.compile`, plays cleanly with the standard PyTorch training loop (`loss.backward()`, `optimizer.step()`), and is configured entirely in Python with no separate config file. DeepSpeed is a heavier external library, configured through a JSON file, that owns the training loop via its `model_engine` and brings a broader feature set: NVMe offload (which FSDP does not natively do as richly), ZeRO-Infinity's parameter streaming, a built-in curriculum of optimizers, and the most battle-tested offload path. The practical guidance most teams converge on: **reach for FSDP first** if you are in pure PyTorch and your offload needs stop at the optimizer-to-CPU level, because it keeps your stack simpler and composes with the rest of the PyTorch ecosystem; **reach for DeepSpeed** when you need aggressive NVMe offload, ZeRO-Infinity-scale parameter streaming, or you are already in a DeepSpeed-based training framework. Both fit the same models to the same memory floor of $16\Psi/N$; the difference is the operational surface area, and you should not over-think it — pick the one your framework already uses.

One subtlety worth flagging: the two libraries can produce slightly different *peak* memory even at the same logical sharding, because the all-gather temporaries, the prefetch depth, and the reduce buffering differ in implementation. FSDP's `limit_all_gathers` and DeepSpeed's `stage3_prefetch_bucket_size` are the knobs that govern how much transient memory the prefetch is allowed to hold; if you are within a gigabyte or two of the line, tuning these can be the difference between fitting and OOM, independent of the steady-state $16\Psi/N$ math. Always measure the real peak with `max_memory_allocated`, do not trust the formula to the last gigabyte — the formula gives the steady state, the transients are implementation detail.

### A named-hardware results table

To make the savings concrete on real silicon, here is the per-GPU memory for our two running models under each strategy, on the two most common datacenter GPUs (A100/H100 come in 40 GB and 80 GB variants), batch 1, sequence 2048, with activation checkpointing on. State is computed from the formulas; activations are the checkpointed ~6 GB (13B) and ~4 GB (7B) estimates; "fits 40 GB" and "fits 80 GB" mark whether the total clears that GPU's wall. These are estimates from the memory model, not microbenchmarks, and I mark them approximate.

| Model · strategy (N=8) | State/GPU | + ckpt act | Total/GPU | Fits 40 GB | Fits 80 GB |
|---|---|---|---|---|---|
| 7B · DDP | 112 GB | ~9 GB | 121 GB | no | no |
| 7B · ZeRO-2 | 26 GB | ~4 GB | 30 GB | yes | yes |
| 7B · ZeRO-3 / FSDP | 14 GB | ~4 GB | 18 GB | yes | yes |
| 13B · DDP | 208 GB | ~17 GB | 225 GB | no | no |
| 13B · ZeRO-2 | 45 GB | ~6 GB | 51 GB | no | yes |
| 13B · ZeRO-3 / FSDP | 26 GB | ~6 GB | 32 GB | yes | yes |
| 13B · ZeRO-3 + CPU offload | ~7 GB | ~6 GB | ~13 GB | yes | yes |

Read the 13B rows as a ladder. DDP does not fit any GPU. ZeRO-2 fits an 80 GB GPU but not a 40 GB one (45 GB of state alone overshoots). ZeRO-3 is the first strategy that fits 40 GB. And ZeRO-3 with CPU offload of the optimizer drops the state to ~7 GB per GPU (the $4\Psi/N$ of params and grads stays on-GPU while the $12\Psi/N$ optimizer shard moves to host), leaving enormous headroom for a bigger batch or longer sequence — at the throughput cost we will quantify. The table is the whole post in one grid: pick the lowest-cost strategy whose total clears your GPU's wall, and apply offload only if even ZeRO-3 overshoots.

## Activation checkpointing: trade compute for memory

Sharding (ZeRO/FSDP) attacks the $16\Psi$ state. It does **nothing** for activations, because activations are not replicated across ranks — each rank already stores only its own microbatch's activations. To shrink activation memory you need a different lever entirely, and the lever is **activation checkpointing** (also called gradient checkpointing or activation recomputation): during the forward pass, store only a sparse set of activations, and during the backward pass, *recompute* the ones you dropped by re-running the forward for that segment. You pay extra compute to avoid storing memory. It is the cleanest compute-for-memory trade in the whole stack.

![before and after comparison showing storing all activations costing memory linear in depth versus checkpointing keeping square root of depth boundaries and recomputing the rest with one extra forward](/imgs/blogs/memory-optimization-zero-fsdp-activation-checkpointing-and-offload-4.png)

The before/after above frames the trade. On the left, the naive path: store every layer's activations, memory grows linearly with depth $L$, one forward and one backward per step. On the right, checkpointing: keep only a sparse set of *checkpoint boundaries*, recompute the activations between them on demand during the backward, paying one extra forward pass. The reason this is such a good deal is the asymmetry between how memory and compute scale, which we can make precise.

### The √L derivation

Suppose you have $L$ layers and you place a checkpoint every $k$ layers. Between two checkpoints, during the backward, you re-run the forward for those $k$ layers to regenerate their activations. Two costs:

- **Memory.** You permanently store the activations at the $L/k$ checkpoint boundaries, plus, transiently, the activations of the *single segment* currently being recomputed during the backward, which is $k$ layers' worth. Total stored activation memory is proportional to $L/k + k$.
- **Compute.** Every layer's forward runs once normally, and once again during the backward when its segment is recomputed — so the forward work roughly doubles for the recomputed portion, adding about one extra forward pass overall.

Minimize the memory term $L/k + k$ over $k$. Taking the derivative and setting it to zero: $-L/k^2 + 1 = 0$, so $k = \sqrt{L}$. At the optimum, stored activation memory is proportional to $\sqrt{L} + \sqrt{L} = 2\sqrt{L}$ instead of $L$. **Checkpointing turns linear activation memory into square-root activation memory.** For a 40-layer model that is a factor of $40 / (2\sqrt{40}) \approx 3.2\times$ less activation memory; for a 100-layer model it is $5\times$ less. The compute cost is one extra forward — and since the backward pass already costs about twice a forward, an extra forward adds roughly $1/3$ to the step's compute, a ~33% slowdown in the worst case (less in practice, because the recompute overlaps with other work and the matmuls are already efficient). Trade a third of your compute for a several-fold cut in activation memory; on a memory-bound fit, that is an obvious yes.

In practice, PyTorch and FSDP do not let you tune $k$ to exactly $\sqrt{L}$ — the common pattern is to checkpoint *every Transformer block* ($k = 1$), which stores only the block boundaries and recomputes each block's internals during the backward. That is more aggressive than the $\sqrt{L}$ optimum (it recomputes everything, not just segments), costing the full ~33% but giving the maximum activation saving, and it is the right default for a tight memory fit. Selective checkpointing — recomputing only the cheap-to-recompute, expensive-to-store ops (like attention) and keeping the rest — is the refinement when you have a little memory headroom and want to claw back some of the 33%.

The plain PyTorch API wraps a function or module call in `torch.utils.checkpoint.checkpoint`:

```python
import torch
from torch.utils.checkpoint import checkpoint

class TransformerBlock(torch.nn.Module):
    def forward(self, x):
        # Normally: store attn + mlp activations for the backward.
        # With checkpoint: store only x, recompute attn+mlp in backward.
        return checkpoint(self._forward_impl, x, use_reentrant=False)

    def _forward_impl(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
```

The `use_reentrant=False` flag selects the newer, safer implementation (the reentrant version has rough edges with some autograd features and is being deprecated). For an FSDP model, you do not hand-wrap every block; you use the composable `checkpoint_wrapper`, which applies checkpointing to whole module classes and composes cleanly with FSDP's wrapping:

```python
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper, apply_activation_checkpointing, CheckpointImpl,
)

# Apply activation checkpointing to every TransformerBlock, then wrap in FSDP.
check_fn = lambda m: isinstance(m, TransformerBlock)
apply_activation_checkpointing(
    model,
    checkpoint_wrapper_fn=lambda m: checkpoint_wrapper(
        m, checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    ),
    check_fn=check_fn,
)
# ... then FSDP(model, ...) as before. Order: checkpoint first, then shard.
```

The ordering matters: apply activation checkpointing **before** wrapping in FSDP so that the recomputed forward during the backward correctly triggers FSDP's re-all-gather of that block's parameters. Get the order wrong and you either lose the memory savings or hit subtle correctness issues.

#### Worked example: checkpointing turns a 17 GB activation bill into 6 GB

Back to our 13B model, 40 layers, batch 1, seq 2048. We estimated ~17 GB of activations if everything is stored — that is the $L \cdot b \cdot s \cdot h \cdot c$ term with $L = 40$. Checkpoint every block ($k = 1$): you now store only the 40 block-boundary tensors (each is just the residual stream, $b \cdot s \cdot h \cdot 2$ bytes $= 1 \times 2048 \times 5120 \times 2 \approx 21$ MB, times 40 $\approx 0.84$ GB) plus the transient internals of whichever single block is recomputing during the backward (a few GB at peak). The stored activation memory drops from ~17 GB to roughly 6 GB — close to a 3× cut, consistent with the $\sqrt{L}$ math for $L = 40$. The cost is one extra forward per step, about a 30% throughput hit. For a model that was 3 GB over the 40 GB line, paying 30% throughput to claw back 11 GB of activation memory and *actually run the job* is not a close call. This is the second of the two techniques that fit 13B on eight 40 GB GPUs in our running example.

## CPU and NVMe offload: when even 1/N is too big

Sharding gets the state down to $16\Psi/N$. Checkpointing gets activations down to $\sqrt{}$. But what if even $16\Psi/N$ does not fit — you are fine-tuning a 70B model on a single 8-GPU node, where $16 \times 70\text{B} / 8 = 140$ GB per GPU, still far over even an 80 GB GPU? You have two options: add more GPUs (raise $N$), or move the cold state off the GPU entirely. **Offload** is the second: park the optimizer states (and optionally the parameters and gradients) in CPU DRAM or on NVMe SSD, and stream them back to the GPU only when needed.

![graph showing optimizer states evicted from GPU HBM to CPU DRAM spilling to NVMe when DRAM is full and prefetched back over PCIe for the Adam step then evicted again](/imgs/blogs/memory-optimization-zero-fsdp-activation-checkpointing-and-offload-5.png)

The graph above is the offload data path. The hot working set — the parameters and activations the forward and backward need *right now* — stays in GPU HBM. The cold 12Ψ of optimizer states gets evicted to CPU DRAM (which on a DGX node is 1–2 TB, more than ten times the GPU memory). If even DRAM is not enough, it spills further to NVMe SSD. At update time, the states are prefetched back over the PCIe bus, the Adam step runs (often *on the CPU* itself, using all the host cores, so the states never have to come back to the GPU at all), and the updated weights are evicted again. The recurring word is **prefetch**: you overlap the slow transfer back with the compute that precedes the update, so the PCIe latency hides.

The catch is bandwidth, and it is a brutal one. GPU HBM runs at 2–3 TB/s. The PCIe Gen4 bus between GPU and CPU runs at about 25 GB/s — roughly **100× slower**. NVMe is slower still, a few GB/s. So every byte you offload is a byte you might have to drag across a link that is 100× slower than where it started. Offload only wins when the offloaded state is *cold* — touched rarely relative to compute — so that the slow transfer is amortized over a lot of fast GPU work. Optimizer states are the ideal offload candidate precisely because they are touched **once per step**, at update time, while the forward and backward (which never touch them) do thousands of matmuls. That single property is why "offload the optimizer" is almost always the first offload you reach for and the last you regret.

DeepSpeed exposes offload through its ZeRO config JSON. Here is a ZeRO-3 config that shards everything and offloads the optimizer states to the CPU:

```json
{
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 16,
  "bf16": { "enabled": true },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6
  },
  "activation_checkpointing": {
    "partition_activations": true,
    "contiguous_memory_optimization": true
  }
}
```

The `offload_optimizer.device: "cpu"` line is the one that moves the 12Ψ of states off the GPU; `pin_memory: true` uses pinned host memory so the prefetch DMA runs at full PCIe bandwidth instead of going through a staging copy. `overlap_comm: true` is what makes the prefetch hide behind compute. To go further and offload the parameters too — necessary for the truly enormous models — you add `"offload_param": {"device": "cpu"}` or, for models bigger than host DRAM, `"device": "nvme"` with an `nvme_path`. That is the configuration ZeRO-Infinity uses to train models with hundreds of billions of parameters on a handful of GPUs: the parameter state lives on NVMe and streams through the GPU layer by layer. It is slow — you are gated by SSD bandwidth — but it *runs* a model that otherwise could not exist on your hardware at all.

To use this config, you let DeepSpeed wrap the model and own the training loop's backward/step:

```python
import deepspeed

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config="ds_zero3_offload.json",
)

for batch in dataloader:
    loss = model_engine(batch).loss
    model_engine.backward(loss)   # reduce-scatter + grad offload
    model_engine.step()           # Adam on CPU, then prefetch weights back
```

`model_engine.backward` and `.step` replace the usual `loss.backward()` / `optimizer.step()` because DeepSpeed needs to interpose the sharding, offload, and prefetch logic around them.

### The throughput cost of offload, quantified

Offload is the technique you reach for last, because its throughput cost is real and often severe. Consider the optimizer step under CPU offload. The gradients (sharded, $2\Psi/N$ bytes per GPU) must be copied to the host, the Adam update runs on CPU cores (which are far slower at the elementwise math than the GPU, though there are many of them), and the updated parameters copied back. On a 7B model with the optimizer offloaded, the step time can grow by 2–4× compared to keeping the optimizer on-GPU, depending on host CPU strength, PCIe generation, and how well the transfer overlaps. The DeepSpeed team's own benchmarks show CPU-offload throughput landing meaningfully below on-GPU ZeRO-3 — you trade tokens/second for the ability to fit at all. The decision rule is simple: **offload only the coldest state you must, and only when adding GPUs is not an option.** If you can raise $N$ instead — borrow more GPUs, go multi-node — do that first, because more ranks shrink the shard *and* keep everything on fast HBM. Offload is the fallback for when you are memory-bound on a fixed, small GPU count.

It is worth being precise about *why* the optimizer is the right thing to offload and the parameters are the wrong thing, because the asymmetry is the entire reason offload works at all. The optimizer states are touched exactly once per training step — at update time — while the forward and backward, which together do thousands of matmuls over the parameters and activations, never touch them. So the ratio of compute-per-byte-transferred is enormous for the optimizer: you drag the 12Ψ across PCIe once and then do a whole step's worth of GPU work that does not need it. The parameters are the opposite — the forward and backward read every parameter on every microbatch, so offloading them means dragging them across PCIe constantly, and the transfer cannot be amortized over much compute. That is why CPU-offload-of-optimizer (ZeRO-Offload) is a moderate slowdown while CPU-offload-of-parameters (part of ZeRO-Infinity) is a severe one reserved for models that cannot fit any other way. The general principle, which transfers to every offload decision you will ever make: **offload the state with the highest compute-to-access ratio first**, because that is the state whose slow transfer hides best behind GPU work.

Stress-testing the offload decision exposes its failure modes. What happens with a weak host CPU? The CPU-side Adam becomes the bottleneck and the step time balloons far past the 2–4× — offload assumes a beefy multi-core host, and on a node with a modest CPU it can be a disaster. What happens on PCIe Gen3 instead of Gen4? The transfer bandwidth halves to ~12 GB/s and every offloaded byte costs twice as much wall-clock; the technique that was marginal on Gen4 becomes unusable. What happens when you offload to NVMe and the SSD is a consumer drive rather than a datacenter NVMe? The few-GB/s sequential bandwidth and the write-endurance limits make parameter streaming crawl, and you may wear the drive out. The honest framing is that offload's cost is *bimodal*: with a strong host CPU, fast PCIe, and pinned memory it is a manageable 2–4× and lets you fit the impossible; with any of those weak, it degrades sharply. Always benchmark offload on your actual hardware before committing a long run to it — the formula tells you it will fit, but only a measured step time tells you whether the run will finish this decade.

| Technique | Attacks | Per-GPU effect | Cost | Reach for when |
|---|---|---|---|---|
| ZeRO-1 / shard optimizer | 12Ψ optimizer | $4\Psi + 12\Psi/N$ | none extra | always, basically free |
| ZeRO-2 / + shard grads | 2Ψ gradients | $2\Psi + 14\Psi/N$ | none extra | always, DDP-equal comm |
| ZeRO-3 / FSDP full shard | 2Ψ params | $16\Psi/N$ | 1.5× comm | model bigger than 1 GPU |
| Activation checkpointing | activations | $L \to \sqrt{L}$ | ~33% compute | long seq or big batch |
| CPU offload | optimizer to host | states leave HBM | 2–4× slower step | fixed small GPU count |
| NVMe offload | params to SSD | params leave host | SSD-bandwidth bound | model bigger than DRAM |

The order in that table is the order you should apply the techniques: sharding first (it is nearly free and scales with $N$), checkpointing next (cheap compute for big activation savings), offload last (expensive, the fallback). Most real fits need the first two; offload enters only when the GPU count is fixed and small.

## Overlapping communication with compute

We have said twice now that sharding's communication cost is "free if you can hide it behind compute." That hiding is not automatic, and understanding it is what separates an FSDP run at 80% of DDP throughput from one at 40%. The mechanism is **prefetch overlap**: while the GPU computes layer $i$, it simultaneously issues the all-gather for layer $i+1$'s parameters on a separate CUDA stream, so by the time layer $i$ finishes, layer $i+1$'s weights are already assembled and compute continues without stalling on communication.

![timeline showing FSDP all-gathering layer L0 then computing each layer while prefetching the next layer's weights so communication overlaps with compute and the reduce-scatter runs in the backward](/imgs/blogs/memory-optimization-zero-fsdp-activation-checkpointing-and-offload-6.png)

The timeline above shows the overlap. The startup all-gather for layer L0 is exposed — there is no prior compute to hide it behind, so you eat that latency once at the start of the forward. But from L0 onward, each layer's compute overlaps with the prefetch of the next layer's all-gather, so the communication disappears into the shadow of the matmuls. The same overlap happens in reverse during the backward, where the reduce-scatter of gradients overlaps with the gradient compute of the previous layer. The only exposed communication is the startup all-gather and a tail flush at the very end — everything in between hides.

The condition for the overlap to fully hide the communication is an arithmetic-intensity argument straight out of the [roofline frame](/blog/machine-learning/high-performance-computing/why-hpc-is-the-bottleneck-for-modern-ai): a layer's compute time must exceed the time to all-gather the next layer's parameters. Compute time for a Transformer block scales with $b \cdot s \cdot h^2$ (the matmul FLOPs); the all-gather time scales with $h^2 / B_\text{net}$ (the parameter bytes over network bandwidth). So overlap succeeds when $b \cdot s$ is large enough relative to the compute-to-bandwidth ratio. **This is why small batches and short sequences are bad for FSDP**: with too little compute per layer, the matmuls finish before the next all-gather does, the prefetch cannot hide, and communication becomes exposed. Conversely, large batches and long sequences give the overlap plenty of compute to hide behind, and FSDP throughput approaches DDP. If your FSDP run is slow, the first thing to check is whether each layer has enough compute to cover its communication — often the fix is a bigger microbatch or gradient accumulation, not a different sharding strategy.

The FSDP knobs that control this are `forward_prefetch` and `backward_prefetch` (which schedule the next all-gather earlier) and `limit_all_gathers` (which throttles how many all-gathers can be in flight, trading a little overlap for a guarantee that prefetched-but-not-yet-used weights do not blow your memory budget). DeepSpeed's equivalent is `overlap_comm: true` plus the prefetch-bucket sizes. The defaults are usually good; reach for them when a profiler trace shows exposed communication.

#### Worked example: why an 8-GPU FSDP run can be barely faster than 1 GPU

Suppose you wrap a 1.3B model in FSDP `FULL_SHARD` on 8 GPUs, run it at batch 1, sequence 512, and measure throughput barely above a single GPU. The model fits easily on one GPU, so you did not even need sharding — but more importantly, at batch 1 seq 512 each Transformer block's matmul takes only a few hundred microseconds, while the all-gather of that block's bf16 parameters across the node (even on NVLink) also takes a few hundred microseconds. The compute cannot hide the communication; every layer stalls waiting for its weights. You are paying ZeRO-3's 1.5× communication with none of the overlap to hide it, and the sharding bought you nothing because the model already fit. The fix is one of: (a) drop to `NO_SHARD` (DDP) since the model fits, (b) raise the batch and sequence so compute covers communication, or (c) use `SHARD_GRAD_OP` (ZeRO-2) which keeps parameters replicated and skips the forward/backward all-gathers entirely. The lesson — and it is the whole "when not to" of this post — is that **sharding is a tool for models that do not fit; applying it to a model that fits, at a small batch, makes things slower, not faster.**

## Gradient accumulation: the free way to shrink activations

There is one more memory lever that costs no communication and no recomputation, and it is the one to try before anything fancy: **gradient accumulation**. The activation term $L \cdot b \cdot s \cdot h \cdot c$ scales linearly with the microbatch size $b$. If you want an effective batch of 32 but each microbatch of 32 would OOM on activations, you instead run 32 *microbatches of 1*, accumulating the gradients across them, and step the optimizer once after all 32. The activation memory is now governed by the microbatch of 1, not the effective batch of 32 — a 32× reduction in the activation term — while the gradient and optimizer state (which accumulate in place) are unchanged. You trade nothing but a little throughput (the optimizer step is amortized over more microbatches, and very small microbatches underutilize the GPU) for a linear cut in the one memory term that scales with batch.

```python
accum_steps = 16
optimizer.zero_grad(set_to_none=True)
for i, micro in enumerate(microbatches):
    loss = model(micro).loss / accum_steps   # scale so the sum averages
    loss.backward()                          # grads accumulate in .grad
    if (i + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

The crucial detail is dividing the loss by `accum_steps` so that summing the gradients over the microbatches produces the *average* gradient, matching what a single large batch would have computed — otherwise you have effectively scaled your learning rate by `accum_steps`. With FSDP there is a second subtlety: by default FSDP reduce-scatters gradients on *every* backward, which under accumulation means $N$ unnecessary reduce-scatters per optimizer step. The `no_sync()` context manager disables the gradient reduction for all but the final microbatch, so you accumulate locally and reduce once — a real throughput win under accumulation. Reach for gradient accumulation first when activations are your bottleneck and you have not yet maxed out the cheaper levers, because it is genuinely free of the communication and recompute costs that sharding and checkpointing carry.

## Saving and loading a sharded model

A practical wrinkle that bites everyone the first time: when your model is sharded across 8 GPUs under FSDP or ZeRO-3, **no single rank holds the full model**, so you cannot just call `torch.save(model.state_dict())` and get a usable checkpoint. Each rank's `state_dict` holds only its $1/N$ slice. There are two ways to handle this, and choosing wrong either OOMs your save or produces a checkpoint you cannot load elsewhere.

The first is a **full state dict**: FSDP all-gathers the entire model onto rank 0 (or onto CPU, with `offload_to_cpu=True`), which reconstitutes the unsharded weights so you can save one consolidated checkpoint compatible with a non-FSDP model. The catch is that all-gathering a 13B model onto one rank momentarily needs the full 26 GB of bf16 weights on that rank — which can OOM the very GPU whose memory pressure made you shard in the first place. So you offload the gather to CPU:

```python
from torch.distributed.fsdp import FullStateDictConfig, StateDictType

save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
    cpu_state = model.state_dict()      # gathered to CPU on rank 0 only
if rank == 0:
    torch.save(cpu_state, "model_full.pt")   # one portable checkpoint
```

The second is a **sharded state dict**, where each rank saves its own slice into a distributed checkpoint directory (PyTorch's `torch.distributed.checkpoint`). No rank ever materializes the full model, so the save is memory-cheap and fast, and it can reload onto a *different* number of GPUs (resharding on load) — exactly what you want for resuming a large training run on a different cluster size. The trade is that the resulting checkpoint is a directory of shards, not a single portable file, so you typically save sharded during training (for resumption) and do one final full-state-dict export at the end (for downstream use and sharing). Getting this right is the difference between a job that can resume after a node failure and one whose checkpoints silently only hold $1/N$ of the weights — a failure mode that does not surface until you try to load and find seven-eighths of your model is missing.

## Case studies and real numbers

Numbers from the literature, to ground the techniques in reported results. Where a figure is approximate or recipe-dependent I say so; I do not invent precise numbers.

**ZeRO enabling trillion-parameter training.** The original ZeRO paper (Rajbhandari et al., 2020) showed that by sharding the 16Ψ across hundreds of GPUs, the per-GPU state shrinks proportionally, removing the per-device memory ceiling as the barrier to model size. The paper reported training models up to 100B+ parameters and projected the path to a trillion: at ZeRO-3 the per-GPU state is $16\Psi/N$, so a trillion-parameter model ($16 \times 10^{12} = 16$ TB of state) sharded across 1024 GPUs is $\approx 16$ GB per GPU of state — back under the line. ZeRO-Infinity (the follow-up adding NVMe offload) pushed this further, reporting the ability to train models with tens of trillions of parameters by spilling parameters to NVMe. The structural point is the one we derived: ZeRO-3's per-GPU memory falls as $1/N$, so model size stops being bounded by a single device and becomes bounded by aggregate cluster memory plus communication.

**FSDP for Llama-scale training.** Meta's FSDP, which PyTorch open-sourced, is the workhorse behind much large-model training in the PyTorch ecosystem and was used in the kind of large-scale runs that produced Llama-family models. The FSDP design paper and PyTorch docs report that `FULL_SHARD` with per-Transformer-block wrapping and activation checkpointing fits multi-billion to tens-of-billions-parameter models on standard 8×A100/H100 nodes, with throughput within a modest margin of DDP when the batch is large enough for the prefetch to overlap communication — exactly the overlap condition we derived. The reported recipe for fitting the largest models on a fixed node count is the stack we built: `FULL_SHARD` + `transformer_auto_wrap_policy` + activation checkpointing + bf16 mixed precision, with offload added only when the GPU count cannot grow.

**The throughput cost of offload, measured.** DeepSpeed's own benchmarks and the ZeRO-Offload paper (Ren et al., 2021) report that CPU offload of the optimizer lets a single GPU train models an order of magnitude larger than it otherwise could (the paper headlines a 10B+ model on a single V100-class GPU), at a throughput cost — the offloaded step is bound by PCIe transfer and CPU-side Adam, landing the run well below an on-GPU ZeRO-3 configuration on the same model when on-GPU would have fit. The consistent finding across these reports is the trade we framed: offload buys *feasibility on small hardware* at the cost of *tokens per second*, so it is the technique of last resort, reached for only when raising the GPU count is not on the table. As always, treat the exact multipliers as hardware- and recipe-dependent; the *direction* (offload is slower but fits more) is the robust claim.

**8-bit optimizers as a composable win.** Worth noting alongside the big three: switching from fp32 Adam to 8-bit Adam (Dettmers et al., `bitsandbytes`) takes the optimizer share from 12 bytes to ~6, dropping the rule from 16Ψ to ~10Ψ with reported negligible accuracy impact across many tasks. It composes with everything above — you can run 8-bit Adam *and* ZeRO-3 *and* checkpointing — and is often the cheapest GB-per-effort lever available, because it changes no parallelism and adds no communication. When you are a little over the line, try the 8-bit optimizer before reaching for offload.

## Fitting the 13B model on eight 40 GB GPUs

Let us assemble the running example into one budget you can reproduce. The model: 13B parameters, 40 layers, hidden 5120. The hardware: a single DGX-class node, eight A100 40GB SXM GPUs connected by NVLink/NVSwitch. The goal: fit a training step, batch 1 per GPU, sequence 2048, mixed-precision bf16 with Adam.

![matrix comparing DDP ZeRO-1 ZeRO-2 and ZeRO-3 plus checkpointing per-GPU state activations and total for a 13B model showing only the last row fits under 40 GB](/imgs/blogs/memory-optimization-zero-fsdp-activation-checkpointing-and-offload-7.png)

The matrix above is the budget. Read down the rows: plain DDP needs 208 GB of state per GPU — a five-fold overshoot, instant OOM. ZeRO-1 (shard optimizer) brings state to $4\Psi + 12\Psi/8 = 4 \times 26 + 1.5 \times 26 \approx$ 78 GB — still nearly double the GPU. ZeRO-2 (also shard gradients) gives $2\Psi + 14\Psi/8 \approx$ 45 GB — tantalizingly close but still over. ZeRO-3 (shard everything) gives $16\Psi/8 = 26$ GB of state; add the checkpointed activation term of ~6 GB and the total is **32 GB**, under the 40 GB wall with 8 GB of headroom for the CUDA context, NCCL buffers, and allocator fragmentation. Only the last row fits, and it took *both* full sharding *and* activation checkpointing to get there.

#### Worked example: the full 13B-on-8×40GB budget, line by line

Here is the arithmetic explicitly, in bf16-Adam bytes, $N = 8$:

- **Parameters (bf16):** $2 \times 13\text{B} = 26$ GB total; sharded $/8 = 3.25$ GB per GPU.
- **Gradients (bf16):** $2 \times 13\text{B} = 26$ GB total; sharded $/8 = 3.25$ GB per GPU.
- **Optimizer (fp32 master + m + v):** $12 \times 13\text{B} = 156$ GB total; sharded $/8 = 19.5$ GB per GPU.
- **State subtotal:** $3.25 + 3.25 + 19.5 = 26$ GB per GPU. (This is exactly $16\Psi/N = 16 \times 13\text{B}/8 = 26$ GB, as the formula promises.)
- **Activations, no checkpointing:** ~17 GB per GPU → total would be 43 GB → **OOM by 3 GB**.
- **Activations, checkpoint every block:** ~6 GB per GPU → **total 32 GB → fits.**

The headroom (40 − 32 = 8 GB) absorbs the ~1 GB CUDA context, the few hundred MB of NCCL buffers, the FSDP all-gather temporaries (one block's full weights, transiently materialized — about 0.65 GB for a 13B block), and allocator fragmentation. If you wanted more margin — say to push sequence length to 4096, which doubles the activation term — you would add CPU offload of the optimizer (moving the 19.5 GB optimizer shard off the GPU, freeing it for activations) at the cost of a slower step. That is the escalation ladder in one example: shard, then checkpoint, then offload, applied in that order, each one only as far as you need.

The PyTorch recipe that realizes this budget, assembled from the snippets above:

```python
# 1. Build model on CPU (meta device for huge models to avoid the build OOM).
model = build_transformer(num_layers=40, hidden=5120)

# 2. Activation checkpoint every block FIRST.
apply_activation_checkpointing(
    model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=check_fn,
)

# 3. Then FSDP full-shard with per-block wrap + bf16 mixed precision.
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    auto_wrap_policy=auto_wrap_policy,
    mixed_precision=mp_policy,
    device_id=torch.cuda.current_device(),
    limit_all_gathers=True,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, fused=True)
```

And the before/after that summarizes the whole post: plain DDP versus ZeRO-3/FSDP per-GPU memory for our running model.

![before and after comparison of plain DDP replicating the full 16 Psi per GPU versus ZeRO-3 keeping a one over N slice trading replication for 1.5x communication](/imgs/blogs/memory-optimization-zero-fsdp-activation-checkpointing-and-offload-8.png)

The figure above is the one-line summary of memory optimization. On the left, DDP: every GPU holds the full 16Ψ, so a 7B model is 112 GB per GPU and will not fit 80 GB — and the only communication is a single gradient all-reduce. On the right, ZeRO-3: every GPU holds one $1/N$ slice, so the same 7B model is 14 GB per GPU on 8 ranks, at the cost of 1.5× the communication (the all-gathers plus the reduce-scatter). DDP optimizes for throughput and wastes memory; ZeRO-3 optimizes for memory and pays a manageable communication tax. Knowing which side of that trade you need — does the model fit, or not? — is the first decision in any training-scale plan.

## When to reach for each technique (and when not to)

Every technique here has a cost, and the engineering is in applying the *minimum* set that fits your model, in the right order. The decision rules:

**Reach for ZeRO-1/2 (or FSDP `SHARD_GRAD_OP`) almost always.** Sharding the optimizer states and gradients is nearly free — no extra communication over DDP — and cuts memory 3–4×. There is little reason *not* to shard the optimizer on a multi-GPU job; it is the cheapest memory win in the stack. If your model fits under DDP but you want headroom for a bigger batch or longer sequence, ZeRO-1/2 is the move.

**Reach for ZeRO-3 / FSDP `FULL_SHARD` when the model does not fit one GPU.** Full parameter sharding is the only thing that lets a model larger than a single GPU exist, but it costs 1.5× communication, and that cost only stays hidden if each layer has enough compute to overlap the all-gather. **Do not** use `FULL_SHARD` on a model that already fits under DDP at a small batch — you will pay the communication with no memory benefit and likely run *slower*, as the small-model worked example showed. The question "does the model fit one GPU?" decides this entirely.

**Reach for activation checkpointing when activations dominate** — long sequences, large batches, deep models. It is a clean ~33% compute cost for a √-scale activation saving, and on a memory-bound fit it is almost always worth it. **Do not** checkpoint if you are already throughput-bound and have activation headroom; you would be paying compute for memory you do not need. Selective/per-op checkpointing is the refinement when you want some of the 33% back.

**Reach for offload last, only when the GPU count is fixed and small.** CPU/NVMe offload is the technique that lets a 70B model train on a single node, but it costs 2–4× step time because PCIe is 100× slower than HBM. **Do not** offload if you can add GPUs instead — more ranks shrink the shard *and* keep everything on fast HBM, which is strictly better than dragging cold state over PCIe. Offload is the fallback for hardware-constrained fits, not a default.

And before any of the above, **try the cheap composable wins**: 8-bit Adam (16Ψ → 10Ψ, no communication change), a smaller microbatch with gradient accumulation (cuts the activation term linearly), and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reclaim fragmentation. Often a job that OOMs by a few GB is fixed by one of these without touching the parallelism at all.

## Key takeaways

- **Mixed-precision Adam costs 16 bytes per parameter** — 2 (bf16 param) + 2 (bf16 grad) + 12 (fp32 master + Adam $m$ + Adam $v$) — and three quarters of that is the optimizer, which is the first thing to shard or offload because it is the most expensive and least frequently used state.
- **A 7B model needs ~112 GB of state and cannot fit an 80 GB GPU at any batch size.** The bf16 weights are only 14 GB; the optimizer is the other 84 GB. The fix is structural (sharding), not a knob.
- **ZeRO stages 1/2/3 shard optimizer states, then gradients, then parameters** across the data-parallel group; per-GPU state falls from $16\Psi$ to $16\Psi/N$. Stages 1 and 2 are nearly free; Stage 3 costs 1.5× the communication of DDP.
- **FSDP is PyTorch's native ZeRO-3** — `ShardingStrategy.FULL_SHARD` plus a `transformer_auto_wrap_policy` that wraps each block, so only one layer's full weights materialize at a time. Wrap granularity is the most common FSDP mistake.
- **Activation checkpointing turns linear activation memory into √-scale** by recomputing dropped activations in the backward, at a cost of one extra forward (~33%). It is the only lever for activations, which sharding does not touch.
- **Offload moves cold optimizer state to CPU or NVMe**, fitting models far larger than the GPUs, but PCIe is ~100× slower than HBM, so the step slows 2–4×. Reach for it last, only when you cannot add GPUs.
- **FSDP's communication hides behind compute only if each layer has enough work** to overlap the next layer's all-gather. Small batches and short sequences expose the communication and make FSDP slow.
- **The order of application is sharding → checkpointing → offload**, each only as far as you need. Fitting 13B on eight 40 GB GPUs takes the first two (FULL_SHARD + checkpointing) for a 32 GB-per-GPU budget; offload enters only if you push sequence length or the GPU count is fixed.

## Further reading

- Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* (2020) — the paper that derives the 16Ψ split and the three sharding stages.
- Ren et al., *ZeRO-Offload: Democratizing Billion-Scale Model Training* (2021) and *ZeRO-Infinity* (2021) — CPU and NVMe offload, the throughput trade.
- The PyTorch FSDP documentation and the *PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel* design paper — wrapping policies, sharding strategies, and the overlap mechanics.
- The DeepSpeed documentation — the ZeRO config JSON, offload options, and activation-checkpointing settings.
- Within this series: [why HPC is the bottleneck for modern AI](/blog/machine-learning/high-performance-computing/why-hpc-is-the-bottleneck-for-modern-ai) for the three-walls frame; [numerical formats and mixed precision](/blog/machine-learning/high-performance-computing/numerical-formats-and-mixed-precision-fp32-tf32-bf16-fp16-fp8) for the bf16/fp32-master discipline the 16Ψ rule rests on; [the memory hierarchy](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm) for why HBM versus PCIe bandwidth governs offload; [collective communication and NCCL all-reduce](/blog/machine-learning/high-performance-computing/collective-communication-and-nccl-all-reduce-from-scratch) for the ring-collective cost behind ZeRO's 1.5×; [parallelism strategies](/blog/machine-learning/high-performance-computing/parallelism-strategies-data-tensor-pipeline-and-expert) for how sharding composes with tensor and pipeline parallel; and the capstone [HPC playbook for AI engineers](/blog/machine-learning/high-performance-computing/the-hpc-playbook-for-ai-engineers).
- Cross-links out: [memory is the real constraint](/blog/machine-learning/edge-ai/memory-is-the-real-constraint) for the on-device view of the same memory wall, and [optimizing MoE training and inference](/blog/machine-learning/large-language-model/optimizing-moe-training-and-inference) for how expert sharding extends these ideas.
