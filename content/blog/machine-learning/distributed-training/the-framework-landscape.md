---
title: "The Framework Landscape: FSDP2 vs DeepSpeed vs Megatron vs the Orchestrators"
date: "2026-07-02"
publishDate: "2026-07-02"
description: "Stop asking DeepSpeed or FSDP. Learn the two-layer map of distributed-training frameworks — engines that own memory and orchestrators that own your loop — and a fit-first way to pick one by model size, cluster scale, and team maturity, with real config and a migration-cost note."
tags:
  [
    "distributed-training",
    "multi-gpu",
    "multi-node",
    "pytorch",
    "fsdp",
    "deepspeed",
    "megatron",
    "ml-systems",
    "deep-learning",
    "gpu",
  ]
category: "machine-learning"
subcategory: "Distributed Training"
author: "Hiep Tran"
featured: true
readTime: 41
---

A team I worked with once spent two weeks in a Slack thread arguing about whether to use DeepSpeed or FSDP for a 7B pretraining run. It was a good-faith argument with benchmarks and screenshots, and it was almost entirely wasted, because the two are not the choice they thought they were making. They then spent a third week wiring in PyTorch Lightning "instead," which is not a substitute for either — it sits on top of both. By the time the run actually started, they had reimplemented their training loop three times and converted their checkpoint format twice, and the model was training at roughly the same speed it would have on day one with four lines of PyTorch. The frameworks were never the bottleneck. The confusion about what the frameworks *were* was.

This is the single most common way distributed-training projects lose time: treating a two-layer world as a flat menu. "Which framework?" gets asked as if DDP, FSDP2, DeepSpeed, Megatron-LM, nanotron, MosaicML Composer, Lightning, and Axolotl were eight interchangeable items on the same shelf, one of which is correct. They are not on the same shelf. Some of them own your training *loop*; some of them own your *memory and parallelism*; a couple of them own neither and just make NCCL bearable. Pick one from the wrong layer and you either get a toy that can't fit your model, or a low-level engine that makes you hand-write everything a good orchestrator would have given you for free.

![two layers of the framework world, an orchestrator that owns the training loop delegating down to an engine that owns the sharding and then down to NCCL and the physical wire](/imgs/blogs/the-framework-landscape-1.webp)

The mental model that dissolves the whole argument is the one drawn above: frameworks split into **orchestrators** that own your training loop and **engines** that own how memory and computation are split across GPUs, and an orchestrator *delegates down* to an engine rather than replacing it. DeepSpeed and FSDP2 are both engines — that Slack argument was a real question, but a much smaller one than it looked, because either one shards the same states over the same NCCL collectives onto the same NVLink. Lightning is an orchestrator; asking "DeepSpeed or Lightning?" is like asking "engine or car?" This post gives you the full map — what each of the eight frameworks actually owns, where they genuinely overlap, and a fit-first procedure that turns your model size, cluster scale, and team maturity into one concrete pick. By the end you will never again lose a week to a category error, and you will be able to defend your choice in one sentence.

This is post 38 in [Distributed Training in the Trenches](/blog/machine-learning/distributed-training/why-distributed-training), and it is the framework companion to [picking a parallelism strategy](/blog/machine-learning/distributed-training/picking-a-parallelism-strategy): that post chose the *math* (how many ways to split); this one chooses the *tool* that implements it. Everything here funnels into the closing [distributed training playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook).

## 1. The category error: engines versus orchestrators

Let me define the two layers precisely, because the entire post hangs on the distinction.

An **engine** owns the answer to *"where does each byte of my model live, and how do gradients get combined across GPUs?"* It decides whether parameters are replicated or sharded, whether the optimizer state is split, when to all-gather a layer's weights before its forward pass and free them after, and which NCCL collective moves the gradients. The engine is what stands between "my model is 112 GB of state and my GPU has 80 GB" and "it runs." FSDP2, DeepSpeed, Megatron-LM, and nanotron are engines. They differ in *how* they split — per-parameter sharding, ZeRO stages with offload, tensor-and-pipeline parallelism — but they all answer the same question.

An **orchestrator** owns the answer to *"what is the shape of my training run?"* — the loop, the callbacks, the logging, the checkpointing cadence, the learning-rate schedule wiring, the automatic resume after a node dies, the config file that turns a fine-tune into a one-command job. PyTorch Lightning, MosaicML Composer, and Axolotl are orchestrators. They do not, by themselves, shard anything. When you set `strategy="fsdp"` in Lightning or hand Axolotl a DeepSpeed config, the orchestrator is delegating the memory problem *down* to an engine and keeping the loop problem for itself.

The relationship is the arrow in figure 1: your loop, model, and optimizer either hand off to an orchestrator (which then hands the memory problem to an engine) or drive an engine directly. Below the engine, every path converges. The engine emits NCCL collectives — `all_gather`, `reduce_scatter`, `all_reduce` — and NCCL emits traffic on the physical interconnect, NVLink inside a node at roughly 900 GB/s aggregate on an H100, InfiniBand between nodes at roughly 200–400 Gb/s per link. Nothing about your choice of framework changes that floor. Two engines that shard identically will move identical bytes over identical wires; their throughput differs only in how well they *overlap* those bytes with compute, which is an implementation detail, not a category difference.

This is why "DeepSpeed versus NCCL" or "FSDP versus NVLink" are nonsense comparisons you sometimes see in blog posts: they compare things on different layers. NCCL is not a framework you choose against DeepSpeed; NCCL is what DeepSpeed *calls*. Keeping the layers straight is the whole skill.

There is one honest complication worth naming up front: the layers leak. FSDP2 is an engine that ships *inside* PyTorch, so "PyTorch-native" training with a hand-written loop uses an engine with no orchestrator at all — you are the orchestrator. DeepSpeed bundles a lightweight orchestrator (its `deepspeed.initialize` and training-step helpers) alongside its engine, so it can feel like both. And Megatron-LM ships an entire training application, not just an engine, so adopting it means adopting its loop, its data pipeline, and its model definitions wholesale. These leaks are exactly why people get confused. But the two-layer map still holds: when a tool spans layers, ask *which layer's problem am I actually buying it for?* You are almost never buying DeepSpeed for its loop.

## 2. What each framework actually owns

With the two layers in hand, here is the honest, opinionated breakdown of the eight names, one row at a time. The rule that makes it usable: **each framework leads on exactly one axis and delegates the rest.** Pick by the wall you actually hit, not by the longest feature list.

![a matrix with one row per framework and columns for layer, memory model, tensor and pipeline support, and best fit, showing each framework leading on a different axis](/imgs/blogs/the-framework-landscape-2.webp)

Reading figure 2 by row is the fast tour, and it is worth walking each row in prose because the one-cell summaries hide real trade-offs.

**PyTorch FSDP2** is the engine that ships in the box. It shards parameters, gradients, and optimizer state per-parameter — the ZeRO-3 memory model — but expressed in native PyTorch with `DTensor` underneath, so your model is still a normal `nn.Module` and your loop is still a normal loop. It leads on *zero lock-in*: there is no custom checkpoint format you can't read, no config JSON, no separate launcher beyond `torchrun`. It is weaker on tensor and pipeline parallelism, which exist via `DTensor` and `torch.distributed.pipelining` but at a lower level of abstraction than Megatron gives you. Best fit: your model fits once sharded, and you want to stay in plain PyTorch. This is the default I reach for, and the one you should reach for too unless something forces you off it.

**DeepSpeed** is the engine that leads on *fitting a model on fewer or smaller GPUs*. Its ZeRO stages 1/2/3 are the same sharding idea as FSDP, but its distinguishing feature is offload: it can push optimizer state to CPU RAM and even parameters to NVMe SSD (ZeRO-Infinity), trading interconnect and PCIe bandwidth for the ability to train a model that has no business fitting on your hardware. It is configured by a JSON file, which is either a feature (declarative, reproducible) or a friction (another format to learn) depending on your taste. It does pipeline parallelism natively but leans on Megatron for tensor parallelism. Best fit: you are memory-bound and want to offload rather than buy more GPUs.

**Megatron-LM** is the engine that leads on *maximum throughput for large dense models on a big cluster*. It is opinionated to the point of being an application: it owns tensor parallelism (TP), pipeline parallelism (PP), and sequence parallelism (SP), it has a distributed fused optimizer, and it is the reference implementation the whole field benchmarks against. The price is that you adopt *its* model definitions and *its* loop — you don't wrap your model in Megatron, you write your model *as* a Megatron model. Best fit: dense models north of ~30B parameters where TP+PP is the only way to fit activations and hit peak MFU. Below that scale it is overkill.

**nanotron** is Hugging Face's minimal 3D-parallel engine, and it leads on *hackability*. It does TP+PP+DP like Megatron but in a small, readable codebase you can actually fork and modify in an afternoon. Its memory model is lighter (ZeRO-1-style optimizer sharding plus tensor sharding). Best fit: research ablations where you need to change how parallelism works, not just use it — the SmolLM models and a good chunk of the FineWeb data ablations were trained on it.

**MosaicML Composer and PyTorch Lightning** are the orchestrators. Composer's `Trainer` and Lightning's `Trainer` both own the loop, the callbacks, the logging, and — the feature you actually pay for — robust automatic checkpointing and resume. Neither shards anything itself; both delegate to FSDP or DeepSpeed via a config flag. Their best fit is ergonomics and operational robustness: you want a run that resumes cleanly after a spot instance dies at 3 a.m. and a config-driven way to launch it, and you are happy to let the engine underneath be FSDP. **Axolotl** belongs in the same layer — it is a YAML-driven orchestrator specialized for LLM fine-tuning that delegates down to FSDP or DeepSpeed, and I treat it as the "I don't want to write a loop at all" option for supervised fine-tuning and LoRA.

Here is the same map as a comparison table you can paste into a decision doc, with the one-line verdict I would give each:

| Framework | Layer | Memory model | TP / PP | Reach for it when | Avoid it when |
|---|---|---|---|---|---|
| PyTorch DDP | Engine (native) | Replicated | None | Model fits on one GPU; you just need data parallelism | Model does not fit replicated |
| PyTorch FSDP2 | Engine (native) | Per-param shard (ZeRO-3) | Low-level via DTensor | Model fits once sharded; you want zero lock-in | You need turnkey TP+PP at 100B+ |
| DeepSpeed | Engine | ZeRO 1/2/3 + CPU/NVMe offload | PP native, TP via Megatron | Memory-bound; offload beats buying GPUs | You have plenty of GPUs and want simplicity |
| Megatron-LM | Engine (application) | Distributed optimizer + shard | TP+PP+SP reference | Dense >30B, big cluster, max MFU | Model fits under FSDP; small team |
| nanotron | Engine | ZeRO-1 + tensor shard | Readable 3D | You need to modify how parallelism works | You just want it to work, not to hack it |
| Lightning / Composer | Orchestrator | Delegates to FSDP/DS | Delegates | You want ergonomics, autoresume, callbacks | You need a capability the engine lacks |
| Axolotl | Orchestrator | Delegates to FSDP/DS | Delegates | Config-driven LLM fine-tuning / LoRA | You are pretraining from scratch with custom code |

The column that matters most is the last two. A feature matrix will tell you all of these "support" mixed precision and gradient checkpointing and multi-node; that tells you nothing, because they all do. What separates them is the single axis each one is genuinely *best* at, and the wall each one is the wrong answer to.

## 3. The stack: where each framework plugs in

If the two-layer split in section 1 is the *conceptual* map, the software stack is the *physical* one — and it explains why the leaks in section 1 happen and why the throughput floor is fixed.

![a vertical stack from your config at the top down through orchestrator, engine, PyTorch autograd, NCCL, CUDA and driver, to the physical interconnect at the bottom](/imgs/blogs/the-framework-landscape-3.webp)

Read figure 3 from the top down. Your config and dataset feed an orchestrator (if you use one). The orchestrator feeds an engine. The engine sits on PyTorch's autograd and `DTensor`. PyTorch's collectives call NCCL. NCCL calls CUDA and the driver. CUDA drives the interconnect — NVLink inside the node, InfiniBand or RoCE between nodes. The load-bearing observation is this: **every framework plugs in at exactly one layer, and everything above the engine eventually funnels through the same NCCL collectives onto the same wire.** There is exactly one narrow waist in this hourglass, and it is NCCL.

That single fact resolves a surprising number of arguments:

- **Why two engines that shard the same way get the same throughput.** FSDP2 FULL_SHARD and DeepSpeed ZeRO-3 both do a `reduce_scatter` of gradients and an `all_gather` of parameters per layer. Same collectives, same byte volume, same wire. If one is 5% faster than the other on your hardware, it is because of prefetch scheduling and bucket sizing — real, but a tuning difference, not a capability difference.
- **Why an orchestrator can't make training faster than its engine.** Lightning cannot beat the FSDP it delegates to, because it is *above* the engine. It can make your run more robust, easier to resume, and quicker to configure. It cannot move fewer bytes.
- **Why "optimize NCCL" is the highest-leverage knob when you are comms-bound.** When your run is bottlenecked on the wire, the framework above is irrelevant; you tune the narrow waist. That is a whole separate post — the [NCCL debugging deep dive](/blog/machine-learning/distributed-training/nccl-debugging-deep-dive) — but the stack diagram is why it matters regardless of which framework you picked.

The practical takeaway: when you evaluate a framework, ask *which layer it plugs into*, and never compare across layers. When you debug a slow run, walk *down* the stack — is the wall in your loop (orchestrator), your sharding schedule (engine), your collective algorithm (NCCL), or your wire (interconnect)? The stack tells you where to look; the framework tells you almost nothing about the answer.

## 4. Why an engine exists at all

Before comparing engines, it is worth being concrete about the problem they solve, because the size of that problem is the reason this whole layer exists. Let me walk the canonical case: a 7B-parameter dense transformer that you want to train with the Adam optimizer in mixed precision, on a single 8-GPU node of A100 80GB cards.

### The memory that overflows

Mixed-precision Adam training keeps, *per parameter*, a fixed set of tensors. Write $\Psi$ for the parameter count. In the standard bf16-with-fp32-master recipe you hold:

- a bf16 copy of the parameters: 2 bytes,
- a bf16 copy of the gradients: 2 bytes,
- and the fp32 optimizer state: an fp32 master copy of the parameters (4 bytes), plus Adam's first moment (4 bytes) and second moment (4 bytes), for 12 bytes.

That is the famous $(2 + 2 + 12)\Psi = 16\Psi$ bytes. For a 7B model, $16 \times 7\text{e}9 = 112$ GB of *state* — and that is before a single byte of activation memory. This derivation is the load-bearing law of the whole memory story, and it is developed in full in [ZeRO and FSDP: the memory model](/blog/machine-learning/distributed-training/zero-and-fsdp-the-memory-model) and [the memory budget](/blog/machine-learning/distributed-training/the-memory-budget); here we only need its consequence.

Under plain DDP, every GPU holds a *full replica* of all 112 GB, because DDP replicates the model and only reduces gradients. 112 GB does not fit on an 80 GB A100. The run does not train slowly; it dies at step 0 with an out-of-memory error. That is the wall an engine exists to break.

![before and after showing a 7B model failing to launch under replicated data parallelism because each GPU needs 112 GB, then fitting under sharding at 14 GB per GPU with room left for activations](/imgs/blogs/the-framework-landscape-4.webp)

Figure 4 is the entire justification for the engine layer in one picture. On the left, DDP: 112 GB of states demanded per GPU, OOM on an 80 GB card, states replicated with no sharding. On the right, FSDP2 with FULL_SHARD: the same 112 GB of state is split across 8 GPUs, so each holds $112 / 8 = 14$ GB, and the roughly 66 GB left over is now available for activations. The model that could not launch under DDP now fits with room to spare. And the code that made the difference is not an architecture change or a new optimizer — it is a wrap.

### The wrap that fixes it

Here is the FSDP2 wrap that turns the left side of figure 4 into the right side. FSDP2 is the current per-parameter-sharding API in recent PyTorch; you apply `fully_shard` to each transformer block and then to the whole model.

```python
import torch
import torch.nn as nn
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

# model is a standard nn.Module: a stack of transformer blocks.
# Nothing about the model definition changes.
mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,   # compute + all-gather in bf16
    reduce_dtype=torch.float32,   # reduce-scatter grads in fp32 for stability
)

# Shard each block first (so params are gathered/freed per block),
# then shard the root to catch embeddings and the final norm.
for block in model.blocks:
    fully_shard(block, mp_policy=mp_policy)
fully_shard(model, mp_policy=mp_policy)

# From here it is a completely ordinary training loop.
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
for batch in dataloader:
    loss = model(batch).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```

A note on naming, because it trips people up: this `fully_shard` API is **FSDP2**, the per-parameter-sharding rewrite that superseded the original `FullyShardedDataParallel` wrapper class (FSDP1). If you find older tutorials wrapping the whole model in `FSDP(model, auto_wrap_policy=...)`, that is FSDP1; the FSDP2 `fully_shard` shown here is the current, more composable API built on `DTensor`, and it is what you should write in new code. The memory model is identical — both give you ZeRO-3-style per-parameter sharding — but FSDP2 composes far more cleanly with tensor parallelism and custom mixed-precision policies.

Launch it with `torchrun` and nothing else — no custom launcher, no config file:

```bash
torchrun --nproc_per_node=8 --nnodes=1 \
  --rdzv_backend=c10d --rdzv_endpoint=localhost:29500 \
  train.py --model 7b --seq-len 4096
```

That is the pitch for FSDP2 as the default: the model definition is untouched, the loop is untouched, the launcher is the one you already use, and the 112 GB wall is gone. There is no new checkpoint format to convert if you save with `torch.distributed.checkpoint`, and nothing about the code is DeepSpeed-shaped or Megatron-shaped. When people ask "why would I not just use FSDP?", *this* is the baseline the alternatives have to beat.

#### Worked example: the 7B run on one 8×A100 node

Let me put numbers on it. Same 7B model, 8×A100 80GB SXM node, NVLink3, bf16, sequence length 4096, measured at steady state (warm-up discarded, `torch.cuda.synchronize()` before timing). These are representative of a real run of this shape, not a vendor benchmark; treat them as the order of magnitude you should expect.

| Setup | Launches? | Peak mem / GPU | Tokens/s (8 GPU) | MFU | Note |
|---|---|---|---|---|---|
| DDP | No | 112 GB needed | — | — | OOM at step 0 |
| FSDP2 FULL_SHARD | Yes | ~64 GB | ~23,000 | ~40% | The default; fits comfortably |
| FSDP2 + activation ckpt | Yes | ~44 GB | ~21,000 | ~37% | Trade ~8% throughput for headroom / bigger batch |
| DeepSpeed ZeRO-3 | Yes | ~64 GB | ~22,500 | ~39% | Same sharding, same wire, within noise of FSDP2 |
| DeepSpeed ZeRO-3 + CPU offload | Yes | ~34 GB | ~9,000 | ~16% | Fits on 4 GPUs, but PCIe becomes the wall |

Read the last two rows against each other and the whole DeepSpeed-versus-FSDP argument evaporates: with everything on-GPU, ZeRO-3 and FSDP2 are within measurement noise, exactly as the stack diagram predicted. DeepSpeed only pulls *away* from FSDP when you turn on the thing FSDP2 does not natively give you — offloading state off the GPU — and that trade sacrifices more than half your throughput to fit on cheaper hardware. That is the whole point of DeepSpeed, and it is a real point, but it is a *different* point from "DeepSpeed is faster."

To compute MFU honestly, the arithmetic is worth showing, because it is the one number that lets you compare across frameworks and hardware fairly:

```python
# A dense transformer does ~6 * N FLOPs per token (2N fwd, 4N bwd).
flops_per_token = 6 * n_params
achieved_flops = flops_per_token * tokens_per_sec
peak = num_gpus * peak_flops_per_gpu   # A100 bf16 ~ 312e12 FLOP/s
mfu = achieved_flops / peak
# 6 * 7e9 * 23000 / (8 * 312e12) = ~0.39  ->  ~40% MFU
```

Measure it this way and a framework's marketing throughput number stops being able to fool you: normalize to MFU and two engines that shard the same way land in the same place.

## 5. The engines up close

Now the engines proper — the same set of code you would actually write, so you can see where they genuinely overlap and where they diverge. We saw FSDP2's wrap above. Here are the other three.

### DeepSpeed: the JSON that offloads

DeepSpeed's engine is driven by a JSON config, and its distinguishing feature is the `offload_optimizer` / `offload_param` block. Here is a ZeRO-3 config that shards everything and offloads the optimizer state to CPU — the setup that fit the 7B on 4 GPUs in the table above:

```json
{
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 8,
  "bf16": { "enabled": true },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": { "device": "cpu", "pin_memory": true },
    "offload_param": { "device": "none" },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 500000000,
    "stage3_prefetch_bucket_size": 500000000,
    "stage3_param_persistence_threshold": 1000000
  }
}
```

You launch it with DeepSpeed's own launcher, which is a thin `torchrun`-equivalent:

```bash
deepspeed --num_gpus=8 train.py --deepspeed ds_config.json
```

The training code changes more than FSDP2 asks for: you call `deepspeed.initialize(model, config="ds_config.json")` and it returns a wrapped model whose `.backward()` and `.step()` you call *on the engine*, not on the optimizer. That is a real, if small, loop rewrite — the first tax figure 6 will charge you. Flip `"device": "nvme"` with a `nvme_path` and you are in ZeRO-Infinity territory, offloading parameters to SSD; that is how DeepSpeed fits models that are physically larger than your aggregate GPU memory, at the cost of PCIe and SSD bandwidth becoming the throughput wall. For the internals of how ZeRO stages and 3D parallelism compose inside DeepSpeed, the [DeepSpeed ZeRO and 3D parallelism deep dive](/blog/machine-learning/open-source-library/deepspeed-zero-3d-parallelism-deep-dive) goes a level deeper than we can here, and [DeepSpeed ZeRO and offload](/blog/machine-learning/distributed-training/deepspeed-zero-and-offload) covers the tuning.

### Megatron-LM: the launch that shards the layer

Megatron is different in kind. You do not wrap your model; you configure *degrees of parallelism* and let Megatron's model definitions and loop do the rest. The launch command is where the design lives:

```bash
torchrun --nproc_per_node=8 --nnodes=8 \
  --node_rank=$RANK --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:29500 \
  pretrain_gpt.py \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 4 \
    --sequence-parallel \
    --num-layers 80 --hidden-size 8192 \
    --num-attention-heads 64 \
    --micro-batch-size 1 --global-batch-size 1024 \
    --use-distributed-optimizer --bf16
```

Read the parallelism degrees against the cluster: 8 nodes × 8 GPUs = 64 GPUs total. Tensor parallelism 8 keeps each layer's matmul sharded *within a node* (so the TP all-reduce stays on NVLink, never on the slower InfiniBand — the reason TP degree usually equals GPUs-per-node). Pipeline parallelism 4 splits the 80 layers into 4 stages *across nodes*. That leaves data parallelism $64 / (8 \times 4) = 2$. This is 3D parallelism, and it is what you reach for when a *single layer's* activations are too big for even a sharded card and the model is too deep to fit in one pipeline stage — the regime [3D parallelism](/blog/machine-learning/distributed-training/3d-parallelism) and [tensor parallelism with Megatron](/blog/machine-learning/distributed-training/tensor-parallelism-megatron) are about. The cost of admission is that this is Megatron's `pretrain_gpt.py`, Megatron's GPT definition, and Megatron's data loader. You are not adding Megatron to your project; you are moving your project into Megatron.

### nanotron: the same idea, but readable

nanotron does the same TP+PP+DP as Megatron with a config object instead of a wall of flags, in a codebase small enough to actually read:

```python
from nanotron.config import ParallelismArgs

parallelism = ParallelismArgs(
    dp=2,            # data-parallel replicas
    tp=8,            # tensor-parallel within node
    pp=4,            # pipeline stages across nodes
    tp_mode="REDUCE_SCATTER",
    tp_linear_async_communication=True,
)
# dp * tp * pp = 64 must equal world size
```

The reason to choose nanotron over Megatron is not capability — Megatron is the more battle-hardened, higher-MFU engine at extreme scale. It is that when your research *is* the parallelism (a new sequence-parallel scheme, a custom expert-routing all-to-all), you can change nanotron in an afternoon and you would spend a week finding the right file in Megatron. Best fit: ablations, not production.

### Where they overlap (and it's a lot)

Notice how much of the four engines is the *same idea in different clothes*. FSDP2 FULL_SHARD and DeepSpeed ZeRO-3 are the same per-parameter sharding. DeepSpeed's pipeline parallelism and Megatron's pipeline parallelism are the same 1F1B schedule. nanotron's tensor parallelism is Megatron's tensor parallelism, reimplemented for readability. The genuine, non-overlapping capabilities are narrow: DeepSpeed's *offload* (nobody else does CPU/NVMe offload as maturely), Megatron's *tensor+sequence parallelism at peak MFU* (the reference for dense >30B), and FSDP2's *zero-lock-in nativeness* (it is just PyTorch). Everything else is a wash, which is exactly why the choice should be driven by which of those three narrow capabilities you actually need — not by a feature checklist where they all look identical.

## 6. The orchestrators: what they buy and when they bite

Orchestrators are the layer people either over-adopt or dismiss, and both mistakes cost time. Let me be precise about what they buy.

What they own, and it is real: a batteries-included training loop, a callback system (early stopping, LR scheduling, EMA, custom hooks), integrated experiment logging, and — the feature that justifies the whole layer for production runs — robust, automatic checkpointing and resume. When a spot instance dies at hour 40 of a 60-hour run, an orchestrator with autoresume detects the failure, respawns, loads the latest checkpoint, and continues, and you find out from a log line instead of a 3 a.m. page. Writing that correctly by hand — atomic checkpoint writes, resuming the data sampler to the exact batch, restoring the LR scheduler and RNG state across ranks — is a genuine project, covered in [fault tolerance and elastic training](/blog/machine-learning/distributed-training/fault-tolerance-and-elastic-training). A good orchestrator gives it to you for a flag.

The critical thing they *do not* own is the memory problem. They delegate it. And the delegation is beautifully explicit — it is usually one string. In Lightning:

```python
import lightning as L

# The orchestrator owns the loop; the string picks the ENGINE.
trainer = L.Trainer(
    devices=8,
    strategy="fsdp",              # or "deepspeed_stage_3", or "ddp"
    precision="bf16-mixed",
    max_epochs=1,
    enable_checkpointing=True,
)
trainer.fit(model, train_dataloader)
```

Switching engines is a one-word edit: `strategy="fsdp"` to `strategy="deepspeed_stage_3"`. That is the two-layer map made literal — the orchestrator stays, the engine swaps. Axolotl does the same thing in YAML, which is why it is the fastest path from "a base model on the Hub" to "a fine-tuned checkpoint":

```yaml
base_model: meta-llama/Llama-2-7b-hf
sequence_len: 4096
adapter: qlora
# The orchestrator delegates memory to an engine via config:
deepspeed: deepspeed_configs/zero3_bf16.json
# or, to delegate to FSDP instead:
# fsdp:
#   - full_shard
#   - auto_wrap
# fsdp_config:
#   fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer
```

When do orchestrators bite? When you need a capability the orchestrator has not exposed from the engine underneath. The engines move faster than the orchestrators that wrap them; a new FSDP2 feature (a custom mixed-precision policy, a specific prefetch schedule, a novel wrap policy) may not have a Lightning flag for months. The moment you find yourself fighting the orchestrator to reach *into* the engine, that is the signal the orchestrator is costing you more than it saves, and you should drop to driving the engine directly. The rule I use: **orchestrators are for runs whose loop is standard and whose robustness matters; drop to the raw engine the moment your loop or your parallelism is non-standard.**

## 7. Which engine for which job

You have the map. Here is the procedure that turns it into a single answer, and it is deliberately shaped as a short sequence of yes/no questions asked *in order*, because the order encodes the fit-first principle: use the least machinery that fits.

![a decision tree that asks in order whether the model fits on one GPU, then whether it fits sharded across a node, then whether it needs tensor and pipeline parallelism, routing to DDP, FSDP2 or DeepSpeed, and Megatron or nanotron](/imgs/blogs/the-framework-landscape-5.webp)

Walk figure 5 top to bottom:

1. **Does the model fit on one GPU with its optimizer?** For a rough cut, a dense model of about 1.5B parameters or fewer, trained with Adam in mixed precision, needs roughly $16 \times 1.5\text{e}9 = 24$ GB of state plus activations — it fits on an 80 GB card. If yes, you do not need an engine at all beyond **DDP**: replicate the model, data-parallelize across GPUs, and move on. Adding FSDP here would only pay sharding overhead for memory you did not need to save.

2. **If not, does it fit once sharded across your node?** This is the FSDP2 / ZeRO-3 regime, and it covers the enormous middle of real work — 7B, 13B, even 70B on a large enough node. If it fits sharded on-GPU, use **FSDP2** and stay native. If it does *not* quite fit even sharded, and buying more GPUs is not an option, reach for **DeepSpeed with offload** to spill state to CPU or NVMe. The order matters: try FSDP2 first, add DeepSpeed offload only when sharding alone falls short.

3. **Only if a single layer's activations are too big to shard, or the model is too deep to fit one pipeline stage,** do you climb to tensor-and-pipeline parallelism. This is **Megatron-LM** for production peak-MFU runs and **nanotron** when you need to hack the parallelism itself. This is the last resort, not the first move, because TP+PP is where the debugging surface explodes.

The discipline the tree enforces is that **you escalate only when the previous rung physically cannot fit the model** — not because a bigger framework sounds more serious. Most teams that "need Megatron" actually needed FSDP2 and a slightly bigger node. The [picking a parallelism strategy](/blog/machine-learning/distributed-training/picking-a-parallelism-strategy) post derives the fit thresholds in detail; this tree is the framework-selection shortcut that rides on top of them.

#### Worked example: a 70B fine-tune, three ways

Suppose you must fine-tune a 70B dense model on a 64-GPU cluster of H100 80GB (8 nodes, NVLink4 inside each node, InfiniBand HDR between). Walk the tree. Question 1: no, 70B needs $16 \times 70\text{e}9 = 1.12$ TB of state — nowhere near one card. Question 2: sharded across 64 GPUs, that state is $1.12\text{TB} / 64 = 17.5$ GB per GPU, which *does* fit on 80 GB with room for activations. So the answer is **FSDP2 FULL_SHARD**, or DeepSpeed ZeRO-3 if you prefer its config ergonomics — you do *not* need Megatron. You only reach question 3, and Megatron, if the *activations* at your context length overflow even after sharding parameters, or if you are pretraining a much larger dense model from scratch where the extra ~10 points of MFU that Megatron's TP+SP buys pays for itself in dollars across a multi-week run. For a 70B fine-tune, that MFU delta rarely justifies moving your entire project into Megatron. Fit-first says: FSDP2, and stop.

## 8. The cost of switching

The decision tree assumes you get to choose once. In reality you often start on one rung and get forced up — and every jump has a bill that the feature comparisons never show you. This is the single most under-estimated cost in the whole landscape.

![a left to right timeline of framework migrations, each step buying a capability and charging a rewrite, ending with a checkpoint format conversion](/imgs/blogs/the-framework-landscape-6.webp)

Figure 6 is the migration ladder, and each rung is a real tax I have paid:

- **DDP to FSDP2.** The cheapest jump. Your model fits under DDP until you widen it or lengthen the context, then it OOMs, and you add the `fully_shard` wrap from section 4. This is roughly three lines and no loop rewrite — the one migration that is nearly free, because FSDP2 is native PyTorch.
- **FSDP2 to DeepSpeed.** You need offload, or ZeRO++'s hierarchical comms. Now you pay a *loop rewrite*: `deepspeed.initialize`, calling `.backward()`/`.step()` on the engine, a JSON config, a different launcher. Not enormous, but real, and it touches your training step.
- **DeepSpeed to Megatron.** You need TP+PP at scale. This is the expensive jump: you adopt Megatron's *model definition*. Your custom attention, your custom norm, your tokenizer glue — all of it has to be reexpressed in Megatron's idioms or it does not get parallelized. This is measured in engineer-weeks, not lines.
- **The sneaky charge at the end: the checkpoint.** Every engine has its own on-disk checkpoint layout. FSDP2 sharded checkpoints, DeepSpeed's ZeRO checkpoint directory, Megatron's TP/PP-sharded shards — none of them is the Hugging Face `safetensors` format your serving stack wants. Shipping the model means writing or running a conversion script, and the conversion for a 3D-parallel Megatron checkpoint (un-sharding TP, un-staging PP, merging to a single HF model) is a genuine, error-prone task. Teams routinely forget to budget for it and discover on launch day that their trained weights are trapped in a format nothing else reads.

The strategic lesson from the timeline: **the fit-first tree is also a migration-minimizing tree.** Every rung you can avoid is a rewrite you never pay. This is a second, quieter argument for FSDP2 as the default — not only is it the least lock-in, it is the rung you can most often *stay on*, and staying on a rung costs nothing. If you can plausibly foresee needing DeepSpeed's offload later, adopting it early via an orchestrator's one-line strategy switch (section 6) is cheaper than a mid-run migration, because the orchestrator absorbs the loop-rewrite tax. Choose the framework you will still be on at the end, not the one that is easiest today.

## 9. Stress-testing the pick: PCIe, 64 GPUs, a tiny batch, and a straggler

The decision tree assumes a friendly cluster. Real clusters are not friendly. A framework choice that looks obviously correct on a single NVLink node can quietly fall apart the moment the cluster gets bigger, the wire gets slower, the batch gets smaller, or one node misbehaves. So let me take the "FSDP2 for a 13B" decision from the worked example and stress it four ways — the way you should stress any framework pick before you commit a multi-week run to it. The instructive result is that the *engine* choice barely moves under any of these stresses; what moves is the configuration and the layer above.

**What happens at 64 GPUs, when the gradient reduction starts to dominate?** FSDP2's communication volume per GPU is set by the sharding math — a `reduce_scatter` of gradients and an `all_gather` of parameters per layer, each moving on the order of the shard size. As the world size grows, that per-GPU byte volume plateaus, but the *number of participants* in every collective grows, and with it the exposure to latency and the slowest link. The framework does not rescue you here; the lever is the sharding *topology*, not the engine. Switching from FULL_SHARD to HYBRID_SHARD — shard within each node, replicate across nodes — collapses the per-layer cross-node `all_gather` into a single gradient `all_reduce` per step over InfiniBand, and keeps the frequent traffic on NVLink. Same framework, different config. On Megatron the equivalent move is keeping tensor parallelism inside the node and letting pipeline and data parallelism span nodes. The lesson generalizes: at scale the framework is fixed and the sharding-topology config is the knob you actually turn.

**On PCIe, not NVLink?** Move the same run onto a box where GPUs talk over PCIe 5.0 instead of NVLink and the intra-node bandwidth falls from roughly 900 GB/s to on the order of 60 GB/s, often contended. FSDP2's per-layer `all_gather`, which hid neatly behind compute on NVLink, is now too slow to overlap; the comms become *exposed* and MFU falls off a cliff. No framework fixes a bandwidth deficit — but it does change the *pick*. On PCIe, reducing collective frequency matters more than shaving per-collective bytes, which nudges you away from aggressive FULL_SHARD toward larger buckets, HYBRID sharding, or sharding fewer layers, and makes DeepSpeed's `overlap_comm` and bucket-size knobs more attractive precisely because there is more comms to hide. The engine choice is downstream of the wire, which is exactly why [the interconnect physics](/blog/machine-learning/distributed-training/the-interconnect-physics) is a prerequisite for framework selection, not a footnote to it.

**When the batch is tiny?** Sharding engines pay a roughly fixed per-step communication cost regardless of how much compute the step does. Shrink the micro-batch — long context at batch 1, an RL rollout, a memory-starved fine-tune — and the comms-to-compute ratio spikes: you are all-gathering a layer's full weights to do a sliver of matmul with them. The fix lives in the loop, not the framework: gradient accumulation fattens the effective batch so each expensive weight gather amortizes over more compute. But the stress exposes a genuine selection insight. If your workload is *inherently* small-batch, sharding overhead is proportionally worse, and a model that happens to fit under DDP will out-throughput the same model under FSDP2 even though FSDP "scales better" on the glossy chart. Fit-first cuts both ways: do not shard memory you did not need to shard.

**When one node is a straggler?** Every engine here uses synchronous collectives, so the slowest rank sets the pace for all of them. A straggler — a thermally throttled card, a flaky NIC, a noisy neighbor stealing a node's PCIe bandwidth — halves throughput for the entire job, and no sharding strategy changes that, because it is a property of synchronous SGD, not of the framework. What *does* differ is the layer above: an orchestrator with autoresume plus an elastic `torchrun` rendezvous can detect a dead or degraded node, evict it, and continue from the last checkpoint, while a hand-written loop sits deadlocked until you notice. This is the one stress where the *orchestrator*, not the engine, is the deciding capability — a concrete reason to pay the orchestrator tax on long multi-node runs. The mechanics are in [the straggler](/blog/machine-learning/distributed-training/the-straggler) and [multinode slower than single-node](/blog/machine-learning/distributed-training/multinode-slower-than-single-node).

Put the four stresses together and the deep lesson of the whole post falls out. Under every one of them, the *engine* answer stayed FSDP2 — the coarse framework choice was robust. What actually decided whether the run was fast and survivable was the *configuration* (FULL_SHARD versus HYBRID, bucket size, accumulation) and the *operational layer* (autoresume, elastic rendezvous). Picking the right framework is necessary, and the tree makes it easy. Configuring that framework for your real wire and your real failure modes is the hard part — and it is the same hard part no matter which engine you picked. That is why arguing for two weeks about DeepSpeed versus FSDP is such a poor use of time: the argument is over the easy, coarse choice, while the decisions that actually move your MFU sit one layer down, in the config, where both engines expose nearly the same knobs.

#### How to benchmark two frameworks fairly

If you are going to compare two engines on your own hardware — the only benchmark that matters — measure so the number cannot lie to you. Discard the first tens of steps: the first iteration allocates the caching allocator's pools, triggers cuDNN autotuning, and JIT-compiles kernels, so it is meaningless. Call `torch.cuda.synchronize()` before you read the clock, because CUDA is asynchronous and an unsynchronized timer measures queue-submission latency, not execution. Measure steady state over a few hundred steps, not a lucky one. Watch for the data-loader confound: if your `DataLoader` cannot keep the GPUs fed, you are benchmarking your disk and your `num_workers`, not the framework, and the slower "framework" is really a starved input pipeline. Watch for thermal and clock throttling on a packed node — a run that starts at boost clocks and settles 10% lower after five minutes will flatter whichever framework you happened to test first. And normalize everything to MFU with the `6 * n_params` arithmetic from section 4, because raw tokens/s is not comparable across sequence lengths or batch sizes but MFU is. Do all of that and the DeepSpeed-versus-FSDP question answers itself in an afternoon, with a number you can defend, instead of two weeks of screenshots.

One regime the tree deliberately skips is Mixture-of-Experts. MoE adds an *expert* axis — an `all_to_all` that routes tokens to the GPUs holding their chosen experts — that FSDP2 alone does not implement. If you are training a sparse MoE model, the engine choice narrows fast: DeepSpeed-MoE and Megatron both provide expert parallelism, and that capability, not memory sharding, becomes the deciding factor. [Expert parallelism for MoE](/blog/machine-learning/distributed-training/expert-parallelism-moe) covers that axis on its own terms; for framework selection the takeaway is simply that MoE pulls you toward DeepSpeed or Megatron regardless of where the dense tree would have sent you, because they own the `all_to_all` and native PyTorch does not yet package it as cleanly.

## 10. Case studies: what real runs actually used

Abstract maps are only as good as the runs they predict. Here are four well-documented cases, each of which lands exactly where the tree would put it. Numbers are as reported by the respective papers and vendor writeups; where I am approximating I say so.

**Megatron-LM at 175B+ (Narayanan et al., 2021).** The Megatron team trained GPT-scale dense models using 3D parallelism on clusters of thousands of A100s, reporting aggregate throughput on the order of 500 petaFLOP/s on 3072 GPUs and per-GPU utilization in the low-50s of percent MFU for the largest configurations. This is the reference case for the bottom-right of the map: a dense model far too big to shard onto a node, on a cluster big enough that the extra MFU from tensor-and-sequence parallelism is worth millions of dollars. Nobody sensible trains a 175B dense model on FSDP2 alone at that scale; this is where Megatron earns its complexity.

**PaLM 540B (Chowdhery et al., 2022).** Google reported 46.2% MFU training a 540B dense model, using model parallelism analogous to the TP+PP regime. The takeaway for framework selection is that even the best-engineered runs at extreme scale live in the mid-40s to low-50s of percent MFU — so if a framework's marketing claims 70% MFU on a large dense model, be skeptical, and always normalize any throughput claim to MFU before believing it.

**MosaicML MPT-7B (2023).** MosaicML trained MPT-7B on 1 trillion tokens using their Composer orchestrator on top of PyTorch FSDP, on 440 A100 40GB GPUs in about 9.5 days, at a reported cost around \$200,000. This is the map's happy middle: an orchestrator (Composer, for autoresume and ergonomics across a long multi-node run) delegating the memory problem to an engine (FSDP), with no tensor or pipeline parallelism in sight because a 7B model *fits when sharded*. It is the strongest real-world argument for the FSDP-plus-orchestrator combination as the default for models in the single-digit-billions.

**BLOOM 176B (BigScience, 2022) — where engines compose.** BLOOM was trained on the Jean Zay cluster on 384 A100 80GB GPUs over roughly three and a half months, using **Megatron-DeepSpeed**: Megatron owned the tensor and pipeline parallelism, DeepSpeed owned the ZeRO-1 data-parallel sharding and the training orchestration, and the two ran as one stack. This is the case that refutes any reading of the two-layer map as "pick exactly one engine." Engines can *compose* — Megatron's TP+PP for the model-parallel axes, DeepSpeed's ZeRO for the data-parallel axis — because they own different parts of the sharding problem. When you are in the bottom-right corner of the grid at 100B-plus, the real-world answer is often not "Megatron" or "DeepSpeed" but the hybrid that lets each do the axis it is best at. It is more moving parts, and more checkpoint-conversion pain, but at that scale the MFU is worth it.

**nanotron and SmolLM / FineWeb (Hugging Face, 2024).** Hugging Face trained the SmolLM family and ran a large battery of data ablations for the FineWeb dataset on nanotron. The reason was exactly the one in the map: the research *was* the training setup — data mixtures, architecture tweaks, parallelism experiments — and a small, readable 3D-parallel codebase you can fork beats a monolith you can only configure. For a lab whose product is ablations, hackability outranks peak MFU.

Put the four side by side and the pattern is clean: the giant dense models went to Megatron, the single-digit-billions production run went to FSDP-under-an-orchestrator, and the research lab went to nanotron. None of them chose by feature matrix. Each chose by the one wall they hit.

## 11. The pick, by size and scale

Time to collapse everything into the single lookup you will actually use on a Monday morning. The two variables that predict the right framework better than any others are **model regime** (how big, and dense versus MoE-ish memory pressure) and **cluster scale** (single node, small multi-node, large multi-node).

![a three by three grid with model regime down the rows and cluster scale across the columns, each cell naming one recommended framework setup, with Megatron only in the bottom right region](/imgs/blogs/the-framework-landscape-7.webp)

Figure 7 is that lookup. Read the rows as model regime and the columns as cluster scale:

- **Small models (roughly ≤3B), any scale:** **DDP.** It fits replicated, so sharding is wasted overhead. At large scale you add HYBRID sharding only if you are widening toward the top of the range. The bulk of this row is "just use DDP and stop thinking about it."
- **Mid models (roughly 7B–30B):** **FSDP2** (or DeepSpeed ZeRO-3 — same sharding). On a single node, FSDP2 FULL_SHARD. On larger clusters, still FSDP2, moving to HYBRID_SHARD so you shard within a node and replicate across nodes to keep the expensive cross-node all-gathers down. Only when it will not fit even sharded do you add DeepSpeed offload. This is the row most readers live in, and the answer is "FSDP2, tuned by cluster size."
- **Large dense models (roughly >30B), small cluster:** **DeepSpeed with offload**, because you are memory-bound and cannot yet spread across enough GPUs to shard your way out. **Large dense models, large cluster:** **Megatron 3D**, the only cell on the whole grid that truly *forces* tensor-and-pipeline parallelism, because now you have both the model size that demands it and the GPU count that makes it efficient.

The shape of the grid is the punchline: **only the bottom-right corner needs Megatron.** Everything else — which is to say almost every model most teams will ever train — is DDP for the small stuff and FSDP2 for the middle, with DeepSpeed as the memory-pressure relief valve and an orchestrator layered on top when operational robustness matters. If your run is not in that bottom-right corner and you are reaching for Megatron, the grid is telling you that you have talked yourself into complexity you do not need.

#### Worked example: same model, two clusters, two answers

A 13B dense pretraining run. On a single 8×H100 node: $16 \times 13\text{e}9 = 208$ GB of state, sharded over 8 GPUs is 26 GB each, fits comfortably on 80 GB cards — **FSDP2 FULL_SHARD**, cell (mid, single-node). Now the same 13B on a 64-GPU cluster: state per GPU drops to about 3.25 GB, trivially fits, but now the cross-node all-gather every layer becomes the throughput risk. The right cell is (mid, large-cluster): **FSDP2 HYBRID_SHARD**, sharding the 208 GB *within* each 8-GPU node and replicating across the 8 nodes, so the per-layer all-gather stays on NVLink and only the gradient all-reduce crosses InfiniBand. Same model, same framework family, but the *configuration* moved because the cluster moved. That is the grid doing its job: it does not just pick FSDP, it picks *which FSDP*.

## 12. When to reach for each (and when not to)

Every framework is a cost as much as a capability. Here is the decisive version, stated as "reach for it when / do not when," because a recommendation that never says *no* is useless.

**Reach for DDP when** the model fits on one GPU with its optimizer and you only need to go faster with more data-parallel workers. **Do not** add FSDP or DeepSpeed here — you would pay sharding communication to save memory you were not short on.

**Reach for FSDP2 when** the model does not fit replicated but fits once sharded, and you value staying in native PyTorch with no lock-in — which is the majority of runs between 7B and 70B. **Do not** reach past it for DeepSpeed or Megatron reflexively; make them earn the migration by pointing at a specific wall FSDP2 cannot clear.

**Reach for DeepSpeed when** you are memory-bound and offloading state to CPU or NVMe is cheaper than buying more GPUs, or you specifically want ZeRO++'s comms optimizations. **Do not** reach for it expecting raw speed over FSDP2 with everything on-GPU — the worked example showed they are within noise, and offload *costs* you throughput.

**Reach for Megatron-LM when** you are training a dense model large enough that a single layer's activations overflow even a sharded card, on a cluster big enough that the peak-MFU tensor-and-sequence parallelism pays for its complexity — the bottom-right corner of the grid. **Do not** adopt it because it sounds like what serious labs use; adopting Megatron means adopting its model definitions, and that is engineer-weeks you should not spend unless the fit tree forced you there.

**Reach for nanotron when** your research is the parallelism itself and you need a codebase you can fork in an afternoon. **Do not** use it for a production run where you want maximum MFU and battle-hardening — that is Megatron's job.

**Reach for an orchestrator (Lightning, Composer, Axolotl) when** your loop is standard, your run is long enough that robust autoresume matters, and config-driven launches speed your team up. **Do not** keep fighting one when you need to reach into the engine for a feature it hasn't exposed — that is the moment to drop to the raw engine.

And the meta-rule that sits above all of them: **do not migrate up the ladder until the current rung physically cannot fit or hit your throughput target.** Every rung up is a rewrite and, eventually, a checkpoint conversion. The cheapest framework is the one you never have to leave.

## 13. Key takeaways

- **The world is two layers, not one flat menu.** Orchestrators own your training loop; engines own memory and parallelism. "DeepSpeed or Lightning?" is a category error — one delegates to the other.
- **Two engines that shard the same way move the same bytes over the same NCCL onto the same wire.** FSDP2 FULL_SHARD and DeepSpeed ZeRO-3 are within measurement noise when everything is on-GPU. The stack has one narrow waist, and it is NCCL.
- **Engines exist to break the memory wall.** A 7B model needs $16\Psi = 112$ GB of state that OOMs an 80 GB card under DDP; one `fully_shard` wrap drops it to 14 GB per GPU across eight cards. That is the whole justification for the layer.
- **FSDP2 is the right default.** It is native PyTorch, zero lock-in, no config file, and the rung you can most often stay on. Make the alternatives earn the migration.
- **DeepSpeed's real, non-overlapping capability is offload;** Megatron's is peak-MFU tensor-and-sequence parallelism for large dense models; nanotron's is hackability. Choose by which of those narrow capabilities you actually need.
- **Only the bottom-right corner needs Megatron** — a large dense model on a large cluster. Almost every run most teams do is DDP (small) or FSDP2 (middle).
- **Every migration up the ladder charges a rewrite, and the sneaky final charge is the checkpoint format.** Fit-first is also migration-minimizing; the cheapest framework is the one you never leave.
- **Normalize every throughput claim to MFU before believing it.** The best runs at extreme scale live in the mid-40s to low-50s of percent MFU; a marketing 70% deserves suspicion.

Bring these back to the [distributed training playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook), where framework choice slots into the larger loop of fit, speed, and cost.

## 14. Further reading

- [Why distributed training](/blog/machine-learning/distributed-training/why-distributed-training) — the four walls that make any of this necessary, and the series' organizing frame.
- [Picking a parallelism strategy](/blog/machine-learning/distributed-training/picking-a-parallelism-strategy) — the fit-first *math* this post turns into a *tool* choice.
- [ZeRO and FSDP: the memory model](/blog/machine-learning/distributed-training/zero-and-fsdp-the-memory-model) and [the memory budget](/blog/machine-learning/distributed-training/the-memory-budget) — where the $(2+2+12)\Psi$ law and the 112 GB come from.
- [FSDP in practice](/blog/machine-learning/distributed-training/fsdp-in-practice) — configuring the default engine correctly, wrap policy up.
- [DeepSpeed ZeRO and offload](/blog/machine-learning/distributed-training/deepspeed-zero-and-offload) and the [DeepSpeed ZeRO and 3D parallelism deep dive](/blog/machine-learning/open-source-library/deepspeed-zero-3d-parallelism-deep-dive) — the offload engine, inside and out.
- [Tensor parallelism with Megatron](/blog/machine-learning/distributed-training/tensor-parallelism-megatron) and [3D parallelism](/blog/machine-learning/distributed-training/3d-parallelism) — the bottom-right corner of the grid in detail.
- Narayanan et al., "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM" (2021); Rajbhandari et al., "ZeRO" (2020) and "ZeRO-Infinity" (2021); Zhao et al., "PyTorch FSDP" (2023); the PyTorch FSDP2 and `torch.distributed` docs; the DeepSpeed and NCCL docs.
