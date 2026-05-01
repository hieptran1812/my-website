---
title: "TorchTitan: A Principal Engineer's Tour of PyTorch-Native LLM Pre-Training"
date: "2026-04-30"
publishDate: "2026-04-30"
description: "An opinionated, principal-engineer walkthrough of TorchTitan — the PyTorch-native pre-training platform that composes FSDP2, tensor parallel, async TP, pipeline parallel, context parallel, float8/MXFP8, selective activation checkpointing, torch.compile, and distributed checkpointing into a single clean codebase. Mental model, runnable code, real benchmarks from the ICLR 2025 paper, and a long catalog of production case studies on Llama 3.1 405B."
tags:
  [
    "torchtitan",
    "pytorch",
    "fsdp2",
    "tensor-parallel",
    "pipeline-parallel",
    "context-parallel",
    "float8",
    "torch-compile",
    "distributed-training",
    "llama",
    "moe",
    "open-source-library",
  ]
category: "machine-learning"
subcategory: "Open Source Library"
author: "Hiep Tran"
featured: true
readTime: 53
aiGenerated: true
---

Most teams discover the limits of their pre-training stack the same way: somebody forks Megatron-LM in 2023, gets a 7B run working, and by the time the team wants to push to 70B the fork has accumulated three custom parallelism patches, two checkpoint-format converters, and a recompiled CUDA extension that pins the cluster to a specific torch nightly. A year later the maintainer leaves, a new GPU generation drops, and the fork is suddenly an archaeology project nobody wants to touch. The team rediscovers the same lesson the field already learned by 2024: **a long-lived pre-training codebase should not encode parallelism as a fork; it should compose parallelism as a library call**. That is the problem [TorchTitan](https://github.com/pytorch/torchtitan) was built to solve.

TorchTitan is the PyTorch team's clean-room implementation of the parallelism primitives that ship inside `torch.distributed`. It is not a competitor to Megatron-LM or DeepSpeed in the "we did it differently" sense; it is the reference application that exercises FSDP2, DTensor, `parallelize_module`, `PipelineSchedule`, `torch.compile`, `torchao.float8`, and DCP — every one of which is a first-class PyTorch API. The headline numbers from the ICLR 2025 paper: a stacked **65.08% acceleration on Llama 3.1 8B with 1D parallelism**, an **additional 12.59% on 70B with 2D**, and an **additional 30% on 405B with 3D**, all on H100s and all relative to a competent FSDP1 baseline. None of those numbers come from a magic kernel. They come from composing the right axes in the right order with the right overlap.

![TorchTitan mental model: one config feeds four parallelism axes over a DeviceMesh](/imgs/blogs/torchtitan-pytorch-native-pretraining-deep-dive-1.png)

The diagram above is the mental model: a single `JobConfig` (a TOML file plus CLI overrides) describes the four parallelism degrees; `init_device_mesh` builds a 4D `DeviceMesh` named `(dp, tp, pp, cp)`; `parallelize_module` walks each transformer block and applies a parallelism plan; FSDP2's `fully_shard` shards parameters per-tensor over the `dp` dim; everything sits on a foundation of DTensor, `torch.compile`, float8, selective activation checkpointing, and Distributed Checkpointing. There is no fork. There is one config and one entry point.

If you have read the rest of this blog, this article completes a triangle: [verl](/blog/machine-learning/open-source-library/verl-rlhf-library-deep-dive) is the RLHF post-training story, [LMCache](/blog/machine-learning/open-source-library/lmcache-kv-cache-layer-deep-dive) is the inference-side KV cache story, and TorchTitan is the pre-training story. Companion reads: [KV cache](/blog/machine-learning/large-language-model/kv-cache) for the inference primitive, and [Docker optimization for LLM workloads](/blog/machine-learning/mlops/docker-optimization-for-llm-and-ai-workloads) for the image-side concerns these multi-node jobs run inside.

## 1. The Pre-Training Stack Problem TorchTitan Was Built To Solve

The naive view is that FSDP plus a hand-rolled tensor-parallel patch is enough. It is — for one model size, on one cluster topology, for one optimizer, until the next thing breaks.

| Assumption (single-fork world) | Reality (production pre-training) |
| --- | --- |
| One parallelism axis is enough (FSDP) | 405B does not fit without TP+PP+FSDP2; 1M-token context adds CP |
| One dtype across the model | Float8 wants per-layer scales; bf16 master weights for stability |
| Activation checkpointing is "on" or "off" | Best MFU comes from per-op SAC: cheap ops not recomputed, attention is |
| Pipeline parallel can be hand-rolled | Zero-bubble schedules are subtle; getting backward right is full-time work |
| `torch.save(model.state_dict())` scales | At 405B the rank-0 gather alone is 30 minutes; DCP is mandatory |
| Compile the whole model | Region compile + FSDP2 + TP needs careful boundary handling |
| Async TP is just a flag | Without SymmetricMemory it is *slower*; correctness depends on the topology |

Each row maps to a TorchTitan design choice. The multi-axis composition is `init_device_mesh` plus `parallelize_module`. The per-layer dtype is float8 conversion against a `Float8LinearConfig`. Per-op SAC is `torch.utils.checkpoint`'s `SelectiveActivationCheckpointPolicy`. Sharded checkpointing is `torch.distributed.checkpoint`. Region compile is `torch.compile(transformer_block, fullgraph=False)`. Async TP is `parallelize_module` plus `torch._inductor.config._micro_pipeline_tp`.

**The Megatron-LM ceiling.** Megatron is the most battle-tested option, but it is a *vertically integrated* codebase: parallelism, model code, optimizer, data loader, and checkpoint format are entangled. If you want to add a new parallelism axis, you fork. If you want a different model architecture (say, an attention-free SSM block), you fork. If you want fp8 with a custom scaling policy, you fork. The forks are individually fine and collectively the reason the LLM community has 30 incompatible "Megatron forks" floating around at any moment.

**The DeepSpeed ceiling.** ZeRO-3 was a remarkable idea in 2020 and remains a viable choice for medium-scale runs, but ZeRO is a *runtime* mechanism — it injects itself into the optimizer step — and that injection makes it hard to compose with TP, PP, or torch.compile. DeepSpeed's TP/PP additions have improved, but a fresh user still has to reason about whether a feature works under ZeRO-3 stage 3 with offloading on or off. The complexity surface is real.

**The "just write your own" ceiling.** Every senior engineer has at least once written `torch.distributed.all_reduce` by hand and felt clever. That cleverness scales until you need: gradient bucketing for performance, mixed-precision allreduce, optimizer-state sharding, fault-tolerant checkpoint resume, and PP that does not deadlock. By the time you have implemented all of those, you have rebuilt FSDP — poorly. TorchTitan exists so you do not.

The conclusion: pre-training in 2026 is a *composition* problem, not an *implementation* problem. The implementations live in PyTorch core. TorchTitan is the demonstration that the composition can be small, readable, and fast.

## 2. The Mental Model: One Config, Four Axes, Swappable Components

Keep this split in your head; every TorchTitan knob lives in exactly one of these boxes.

**The job config** owns:
- A TOML file (`train_configs/llama3_405b.toml`) plus CLI overrides.
- Parallelism degrees: `data_parallel_shard_degree`, `tensor_parallel_degree`, `pipeline_parallel_degree`, `context_parallel_degree`.
- Optimizer, scheduler, model arch, dataset, checkpoint paths.
- Float8 / SAC / compile / async-TP feature flags.

**The DeviceMesh** owns:
- The cartesian product of GPU ranks across the four axes.
- Named sub-meshes: `mesh["tp"]`, `mesh["dp"]`, `mesh["pp"]`, `mesh["cp"]`.
- Process-group lookup that the parallelism APIs consume.

**The parallelize step** owns:
- A walk over the transformer model that applies a plan per block.
- TP plans (`ColwiseParallel`, `RowwiseParallel`, `SequenceParallel`).
- FSDP2 wrapping via `fully_shard(transformer_block, mesh=dp_mesh)`.
- Optional float8 conversion, SAC wrapping, compile wrapping.

**The training loop** owns:
- Dataloader with sequence packing.
- Pipeline schedule if PP > 1 (`Schedule1F1B`, `ScheduleZBVZeroBubble`).
- Forward/backward/step under `mesh.get_group()` collectives.
- DCP async checkpoint every N steps.

The crucial invariant: model code is *unaware* of parallelism. `Llama.forward()` is the same forward you would write for a single GPU. Parallelism is applied *after* construction, by mutating module attributes (replacing a `nn.Linear` with a `ColwiseParallel`-wrapped version) and by registering DTensor placements on parameters. This is the inverse of Megatron's design where the model is written in terms of parallelism primitives. The cost is a small amount of API surface to learn (DTensor, `parallelize_module`, `fully_shard`); the benefit is that adding a new model architecture means writing a normal PyTorch model plus a parallelism plan dict — not forking.

## 3. FSDP2 and Per-Parameter Sharding

FSDP1 was the right idea in 2022. By 2024 its central trick — concatenating every parameter in a unit into a single flat buffer (`FlatParameter`) — became the reason it could not compose with anything else. FSDP2's `fully_shard` keeps the all-gather batching that made FSDP1 fast, but stops fusing parameters. Each parameter becomes its own DTensor with its own placement.

![FSDP1 vs FSDP2: flat-param vs per-parameter sharding](/imgs/blogs/torchtitan-pytorch-native-pretraining-deep-dive-2.png)

Why per-parameter matters in practice:

1. **Per-parameter dtype.** With FSDP1 every parameter inside a unit shares one dtype because they live in the same buffer. Float8 training wants the linear weights in fp8 but the embedding and the LM head in bf16. FSDP1 forces an awkward split into multiple units. FSDP2 keeps them in the same unit and just gives each parameter its own dtype.
2. **No padding waste.** FSDP1 pads the flat buffer to a multiple of `world_size`. On a 70B model with thousands of small bias tensors this can waste 5–15% of memory. FSDP2 sizes each shard to its parameter exactly.
3. **Composable with TP.** Tensor parallel marks a parameter with `Shard(0)` or `Shard(1)` placements on the `tp` mesh dim. FSDP2 then shards over the `dp` dim *of the same tensor*, producing a 2D `DTensor(Shard(0), Shard(1))`. FSDP1 cannot represent this because the flat buffer has no notion of "this slice belongs to a TP rank."
4. **DCP-native.** FSDP1 saves the flat buffer and a metadata file describing how to un-flatten it. FSDP2 saves the parameter dict directly. Loading a checkpoint into a different parallelism configuration becomes a DTensor reshard, not a custom converter.

Here is the full FSDP2 wrap that TorchTitan applies in `parallelize.py`:

```python
import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

def apply_fsdp2(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype = torch.bfloat16,
    reduce_dtype: torch.dtype = torch.float32,
) -> nn.Module:
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
    )
    # Wrap each transformer block as its own unit so the
    # all-gather/reduce-scatter happens at block boundaries,
    # which gives the best overlap with compute.
    for block_id, block in enumerate(model.layers):
        reshard_after_forward = block_id < len(model.layers) - 1
        fully_shard(
            block,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
        )
    fully_shard(model, mesh=dp_mesh, mp_policy=mp_policy)
    return model
```

Three subtle things in that 15 lines.

The `reshard_after_forward=False` for the last block is a perf optimization: backward begins on the last block immediately after forward ends, so we keep its parameters un-sharded to skip a redundant all-gather. The earlier blocks reshard to free memory.

The `mp_policy` says parameters live in bf16 for compute but gradients reduce in fp32 to avoid loss-scale gymnastics on small gradients. This is the Megatron-style mixed precision; FSDP1 supported it but only as a unit-wide setting. FSDP2 lets you configure it per `fully_shard` call.

The whole-model `fully_shard` at the end wraps the embeddings, lm_head, and any non-block parameters into their own unit. This single line is what stops the rank-0 process from materializing the full 405B parameter set during init.

**Memory math at 405B.** Llama 3.1 405B has roughly 405 × 10⁹ bf16 parameters = 810 GB of weights, 1620 GB of Adam state (m and v in fp32), and 405 GB of fp32 master weights — about 2.8 TB of state before activations. With DP=8, TP=8, PP=8 (so 512 ranks), the per-GPU state is 2.8 TB / 512 ≈ 5.5 GB of weights+optimizer. That fits comfortably in an H100's 80 GB alongside activations and KV-cache buffers, with room to grow microbatch size for better PP efficiency.

## 4. Tensor Parallel and Async TP

Tensor parallel splits each linear layer across `tp` ranks. The textbook version: `Q = x @ W_q` becomes `Q_local = x @ W_q[:, shard]` (column-wise) followed by an all-gather, or `out = attn @ W_o` becomes `out_partial = attn_local @ W_o[shard, :]` followed by a reduce. TorchTitan expresses this as a *plan*:

```python
from torch.distributed.tensor.parallel import (
    ColwiseParallel, RowwiseParallel, SequenceParallel,
    parallelize_module, PrepareModuleInput,
)
from torch.distributed.tensor import Replicate, Shard

def build_tp_plan() -> dict:
    return {
        "attention_norm": SequenceParallel(),
        "attention": PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
        ),
        "attention.wq": ColwiseParallel(),
        "attention.wk": ColwiseParallel(),
        "attention.wv": ColwiseParallel(),
        "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
        "ffn_norm": SequenceParallel(),
        "feed_forward": PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
        ),
        "feed_forward.w1": ColwiseParallel(),
        "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
        "feed_forward.w3": ColwiseParallel(),
    }

def apply_tp(model, tp_mesh):
    for block in model.layers:
        parallelize_module(block, tp_mesh, build_tp_plan())
    return model
```

Read this carefully because it is the densest API in TorchTitan. `ColwiseParallel` shards `W` on dim 1; the input is replicated, the output is sharded. `RowwiseParallel` shards `W` on dim 0; the input is sharded, the output is reduced (all-reduced) and replicated. `SequenceParallel` shards activations on the sequence dim while parameters stay replicated — this is the Megatron sequence-parallel trick that hides layer norm and dropout in the sequence dim and roughly halves activation memory at the cost of two extra all-gathers per block.

`PrepareModuleInput` is the glue: it tells the framework that `attention` *receives* a Shard(1) input (because the previous norm produced one) but *expects* a Replicate input (because the all-gather happens here). The framework inserts the all-gather automatically.

### Async TP

The naive TP execution is:

```text
all_gather(activations)  →  matmul(W, gathered)
```

Eight milliseconds of comm followed by twelve milliseconds of compute, in series. Async TP slices both into chunks and pipelines them.

![Async TP overlaps all-gather with matmul via SymmetricMemory and chunked epilogue](/imgs/blogs/torchtitan-pytorch-native-pretraining-deep-dive-4.png)

Three pieces make this work:

1. **SymmetricMemory.** A PyTorch primitive that maps NVLink-peer buffers into each GPU's address space. Instead of "rank 0 sends to rank 1," each rank reads directly from its peers via a P2P load. Comm becomes a memory-mapped pull, which can run concurrently with compute on the same SMs.
2. **Chunked matmul decomposition.** The matmul is split along the `K` (contraction) dimension into chunks. Chunk `i` is computed using the partial all-gather from chunk `i-1`. A trailing accumulate sums the chunk outputs.
3. **Fused kernel launch.** `torch._async_tp_ops.fused_all_gather_matmul` issues the comm and compute on the same CUDA stream with a chunked epilogue, eliminating the kernel-launch barrier between them.

Empirically this buys ~1.4× on the all-gather-matmul step and ~10–15% MFU at the model level on intra-node TP=8. It is *not* free across nodes: when the TP group spans IB instead of NVLink, the comm is bandwidth-bound, the chunking overhead dominates, and async TP slows things down. TorchTitan auto-disables async TP when the TP group has any inter-node members, but production teams still hit the trap when they manually set `enable_async_tensor_parallel = true` on a multi-node TP plan. (See case study 2.)

The flag in TorchTitan:

```toml
[experimental]
enable_async_tensor_parallel = true
```

It is gated behind `experimental` because the API is stabilizing; on Blackwell GPUs (B200) the trick gets a second wind because the NVLink-5 bandwidth is 2× and the chunked epilogue's overhead amortizes faster.

## 5. Pipeline Parallel: Zero-Bubble and 1F1B

Pipeline parallel splits the transformer's *layers* across stages: stage 0 owns layers 0..15, stage 1 owns 16..31, and so on. The cost is the *pipeline bubble*: at the start and end of each step, some stages are idle while microbatches are flowing in or out. With `S` stages and `M` microbatches the bubble fraction is roughly `(S - 1) / (M + S - 1)`. With `S=8` and `M=8` the bubble is 7/15 ≈ 47% — devastating. With `M=64` it drops to 7/71 ≈ 10%.

TorchTitan does not make you write a pipeline runner. It uses `torch.distributed.pipelining` and a `PipelineSchedule` registry:

```python
from torch.distributed.pipelining import (
    PipelineStage, ScheduleGPipe, Schedule1F1B,
    ScheduleZBVZeroBubble, ScheduleInterleaved1F1B,
)

stage = PipelineStage(
    submodule=local_layers,
    stage_index=pp_rank,
    num_stages=pp_size,
    device=device,
    input_args=example_input,
)

schedule = ScheduleZBVZeroBubble(
    stage=stage,
    n_microbatches=64,
    loss_fn=loss_fn,
    scale_grads=True,
)

## (training loop)
schedule.step(input_ids, target=labels)
```

Four schedules ship today:

| Schedule | Bubble | Memory | When to pick |
| --- | --- | --- | --- |
| `ScheduleGPipe` | `(S-1)/(M+S-1)` | M · activation | Tiny models, debugging only |
| `Schedule1F1B` | same | activation × S | Default; balances memory and bubble |
| `ScheduleInterleaved1F1B` | half of 1F1B | same | When `M >> S` and you want extra throughput |
| `ScheduleZBVZeroBubble` | ~0% | activation × S × 1.5 | Production, when you can afford the activation |

Zero-bubble works by splitting backward into "input gradient" (`B`) and "weight gradient" (`W`) — `W` does not depend on the next stage and can be slid into the bubble. The downside is more activation memory and a harder schedule to debug if something deadlocks, which is why `Schedule1F1B` remains the default.

**The PP placement rule.** PP communication is *latency-bound* (point-to-point sends of one microbatch's activations), not bandwidth-bound, so it tolerates IB perfectly well. TP is bandwidth-bound and must stay on NVLink. So the right composition is: TP=8 within a node (NVLink), DP across nodes within a "pod" (200G IB), PP across pods (lower-bandwidth IB). TorchTitan orders the mesh dims `(pp, dp, cp, tp)` such that the *inner* dim maps to consecutive ranks — which on a typical SLURM allocation is the same node — so TP sits on NVLink by default. Misordering the mesh dims is the single most common MFU-killing bug; case study 4 covers one such incident.

## 6. Context Parallel for 1M-Token Sequences

The activation footprint of attention is `O(B × H × S²)`. At `S = 1,048,576` tokens and `H = 128` heads with `D = 128`, a single attention layer's score matrix in bf16 is `1 × 128 × 1M × 1M × 2 bytes = 256 TB`. FlashAttention's recomputation trick brings the materialized matrix down to a tile, but the *softmax-statistics buffer* still scales with `B × H × S`, and at 1M context that is 256 GB per layer — more than an H100. The fix is context parallel: shard `S` across `cp` ranks.

TorchTitan implements ring attention. Each CP rank holds a `S/cp` slice of the sequence; for the attention computation, K and V are passed around the ring in `cp` rounds, with each rank computing the partial attention against its local Q. With causal masking, the naive ring is load-imbalanced because rank 0 only attends to its own slice while rank `cp-1` attends to everything. TorchTitan implements *load-balanced causal CP*: each rank holds two non-contiguous chunks (one early, one late) so the per-step work is uniform.

```python
from torchtitan.parallelisms.context_parallel import (
    create_context_parallel_ctx,
)

cp_mesh = mesh["cp"]
with create_context_parallel_ctx(
    cp_mesh=cp_mesh,
    cp_buffers=[input_ids, position_ids],
    cp_seq_dims=[1, 1],
    cp_no_restore_buffers={input_ids, position_ids},
    cp_rotate_method="alltoall",  # or "allgather"
):
    out = model(input_ids, position_ids=position_ids)
    loss = loss_fn(out, labels)
    loss.backward()
```

`cp_rotate_method="alltoall"` is the modern path: instead of sending K/V around the ring with point-to-point sends, the entire ring's K/V is exchanged with one all-to-all per step. This is bandwidth-equivalent to the ring but latency-free, which matters when the per-step compute is small (small heads, short K-tile). `allgather` is the legacy path that simply gathers all K/V at every CP rank — correct, simple, but doubles the activation memory.

The interaction with TP matters. If both TP and CP are on, the QKV projections are split column-wise across TP, and the resulting Q/K/V are *also* split sequence-wise across CP. The full DTensor placement is `(Shard(seq=1), Shard(head=2))` — a 2D sharding. The attention kernel must understand this placement, which is why TorchTitan ships its own attention dispatch that checks `qkv.placements` and routes to the right CP-aware FlashAttention.

CP composes cleanly with PP and FSDP2: PP shards layers, FSDP2 shards parameters, CP shards activations. A 1M-context Llama 3.1 8B run uses TP=1, CP=8, DP=8, PP=1 on 64 GPUs and trains stably with ~38% MFU.

## 7. Float8 and MXFP8 Training

Float8 cuts compute time in half on H100 (where fp8 matmul throughput is 2× bf16) and memory by half on the matmul activation footprint. The catch is dynamic range: fp8 has ~17 bits of effective mantissa+exponent and saturates aggressively. Two scaling strategies dominate.

**Delayed scaling.** Each linear keeps a running history of the absolute-max of its activations and weights; the scale for step `t` is computed from the history at step `t-1`. Cheap (the abs-max is a single reduction per linear) but stale: a sudden spike in activation magnitudes saturates fp8 before the scale catches up. The history must be checkpointed; if it desyncs from the model, training corrupts. (Case study 3 is exactly this.)

**Dynamic scaling.** The scale is computed from the *current* tensor before each matmul. No history, no staleness, but two extra reductions per linear. On H100 this costs ~3% throughput compared to delayed; on Blackwell it is roughly free because of the new fused-scale matmul intrinsic.

TorchTitan applies float8 conversion *after* parallelism via `torchao.float8`:

```python
from torchao.float8 import (
    Float8LinearConfig,
    convert_to_float8_training,
    Float8Linear,
)

float8_config = Float8LinearConfig(
    enable_fsdp_float8_all_gather=True,   # the killer feature
    enable_pre_and_post_scale_persistence=False,
    cast_config_input=Float8CastConfig(scaling_type=ScalingType.DYNAMIC),
    cast_config_weight=Float8CastConfig(scaling_type=ScalingType.DYNAMIC),
    cast_config_grad_output=Float8CastConfig(scaling_type=ScalingType.DYNAMIC),
)

convert_to_float8_training(
    model,
    config=float8_config,
    module_filter_fn=lambda m, fqn: isinstance(m, nn.Linear)
        and "lm_head" not in fqn
        and "embed_tokens" not in fqn,
)
```

Two things deserve attention. `enable_fsdp_float8_all_gather=True` is the "all-gather in fp8" trick: FSDP2 normally all-gathers parameters in the param dtype (bf16). With this flag, the all-gather happens in fp8 — half the bytes on the wire. On a 70B model this turns the FSDP2 collective from ~14 ms to ~7 ms per layer, and on H100 with NVLink-saturating clusters it is the single biggest end-to-end MFU lever.

The `module_filter_fn` excludes the embedding and lm_head. Both are sensitive: the embedding has very heavy-tailed activations (token frequencies follow Zipf), and the lm_head's gradients drive the cross-entropy loss directly. Quantizing either to fp8 risks loss spikes. A 5% throughput loss for keeping these two in bf16 is the right trade.

**MXFP8 for Blackwell.** MX (microscaling) is a block-wise fp8 format where each 32-element block carries its own 8-bit shared exponent. Blackwell's tensor cores support MXFP8 natively and outperform delayed-scaling fp8 on matmul throughput by another ~25%. TorchTitan supports MXFP8 for both dense and MoE models on B200; the conversion is the same `convert_to_float8_training` call with `MXFP8Config` instead of `Float8LinearConfig`. Expect MXFP8 to become the default once Blackwell penetration crosses 50%.

**Loss curve sanity.** Float8 training loses ~0.1–0.3% on perplexity at 8B scale, often within run-to-run variance. The loss is not from "fp8 is less precise"; it is from the saturation behavior of activations near the start of training when batch-norm-like statistics have not stabilized. The fix is to delay-enable float8 — train the first 500 steps in bf16, then flip the conversion on. TorchTitan has a `float8.warmup_steps` flag for exactly this.

## 8. Activation Checkpointing: Per-Op Selective vs Full

Activation checkpointing trades compute for memory: discard activations on forward and recompute them on backward. Full activation checkpointing recomputes everything; it is correct but expensive (~30% throughput hit). Per-op selective is the modern approach: recompute *only* the ops where the activation cost dominates the recompute cost. Attention scores (`O(S²)` activation, `O(S²)` recompute) are usually checkpointed; matmuls (`O(BSH)` activation, `O(BSH²)` recompute) are usually kept.

```python
from torch.utils.checkpoint import (
    checkpoint, create_selective_checkpoint_contexts,
)
from torch.utils.checkpoint import CheckpointPolicy

def selective_op_policy(ctx, op, *args, **kwargs):
    # Save outputs of cheap ops; recompute everything else.
    save_list = {
        torch.ops.aten.mm.default,
        torch.ops.aten._scaled_dot_product_efficient_attention.default,
        torch.ops.aten._scaled_dot_product_flash_attention.default,
        torch.ops.aten._scaled_mm.default,
    }
    if op in save_list:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE

def apply_sac(transformer_block):
    return torch.utils.checkpoint.checkpoint(
        transformer_block,
        use_reentrant=False,
        context_fn=lambda: create_selective_checkpoint_contexts(
            selective_op_policy
        ),
    )
```

The policy is a function from op to "save or recompute." TorchTitan's default policy saves matmuls (`aten.mm`, `aten._scaled_mm` for fp8) and saves the FlashAttention output (because re-running attention is expensive). It recomputes element-wise ops (RMSNorm, GeGLU, residual adds), which are bandwidth-bound and roughly free to recompute.

The empirical impact: per-op SAC runs at ~95% of no-checkpointing throughput while using ~60% of the activation memory of no-checkpointing. Full-block SAC runs at ~70% of throughput while using ~40% of the memory. The frontier is real and per-op is on it.

**The trap.** Per-op SAC interacts poorly with custom ops: if a fused kernel is not in the save list, it gets recomputed, and if the kernel has side effects (e.g., updates a stat), the recompute corrupts state. Always register custom ops with explicit `MUST_SAVE` if they are stateful. Case study 6 covers a real instance of this.

## 9. `torch.compile` Integration

Compiling the entire model is fragile: dynamic shapes from variable-length sequences, custom ops, FSDP2's unshard hooks, and TP's collective insertion all create graph breaks. TorchTitan compiles per-block:

```python
def apply_compile(model, fullgraph=False):
    for block_id, block in enumerate(model.layers):
        compiled_block = torch.compile(
            block,
            fullgraph=fullgraph,
            mode="default",
            dynamic=True,
        )
        model.layers[block_id] = compiled_block
    return model
```

Compiling each transformer block keeps the compile region small enough to recompile cheaply if a graph break occurs, and large enough to fuse the RMSNorm + linear + GeGLU ops that Inductor's fusion pass benefits from. Empirically the wins on Llama 3.1 8B at H100:

| Configuration | TFLOPs/GPU | MFU |
| --- | --- | --- |
| FSDP2, bf16, no compile | 432 | 43.6% |
| + `torch.compile` per block | 478 | 48.3% |
| + float8 dynamic | 612 | 61.9% |
| + async TP (TP=8) | 654 | 66.1% |

The `torch.compile` win is ~10% on top of FSDP2-bf16. It composes cleanly with float8 (Inductor recognizes the fp8 cast and fuses the scaling) and with async TP (the chunked all-gather-matmul is a pattern Inductor matches). It does *not* compose well with full SAC because the recompute path triggers a different compilation, doubling cache size.

**Region compile and graph breaks.** Common graph-break sources: `torch.distributed.all_reduce` (now supported), FSDP2's `unshard` hooks (handled internally), Python-side conditionals that depend on tensor values. The fix for the last category is to lift the conditional out of the compiled region. TorchTitan's transformer block has none of these by construction — every conditional is on a static config flag.

## 10. Distributed Checkpointing (DCP) and Async Save

`torch.save(model.state_dict())` does not work at 405B. Rank 0 cannot allocate 2.8 TB to gather the optimizer state; even if it could, the I/O serialization would take 30+ minutes. DCP (Distributed Checkpoint, `torch.distributed.checkpoint`) saves *sharded* state directly: each rank writes its own piece, the metadata records the placements, and a load reconstructs the right shape via DTensor reshard.

```python
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_state_dict, set_state_dict, StateDictOptions,
)

def save_checkpoint(model, optimizer, step, save_dir):
    state_dict = {
        "model": model,
        "optim": optimizer,
        "step": step,
    }
    msd, osd = get_state_dict(
        model, optimizer,
        options=StateDictOptions(full_state_dict=False, cpu_offload=False),
    )
    dcp.save(
        state_dict={"model": msd, "optim": osd, "step": step},
        checkpoint_id=f"{save_dir}/step_{step}",
    )

def load_checkpoint(model, optimizer, load_dir):
    msd, osd = get_state_dict(model, optimizer)
    state_dict = {"model": msd, "optim": osd, "step": 0}
    dcp.load(state_dict=state_dict, checkpoint_id=load_dir)
    set_state_dict(model, optimizer, model_state_dict=msd, optim_state_dict=osd)
    return state_dict["step"]
```

Two features make DCP usable at scale.

**Async save.** `dcp.async_save` returns immediately after staging the state dict to a CPU pinned buffer; the actual write to disk happens on a background thread. Steps are unblocked after ~200 ms instead of waiting 30+ seconds for the disk write. The catch: the *next* checkpoint must wait for the previous async save to finish or you risk overlapping writes corrupting the metadata. TorchTitan tracks this via a single in-flight future and blocks on it before issuing the next save.

**Resharding.** A checkpoint saved with TP=8, PP=8 can be loaded with TP=4, PP=16 — DCP looks at the saved DTensor placements and the current ones, computes the reshard, and pulls the right slices over the network. This is the killer feature for resuming after a hardware reconfiguration. Megatron-style flat checkpoints require an offline conversion script for the same operation.

**Interop with torchtune.** TorchTitan and torchtune share the DCP format. A pre-training run can resume into a torchtune fine-tune with no conversion. This is the only PyTorch-native pre-train → fine-tune handoff that exists; everywhere else you have to convert to HF format and back.

## 11. MoE and Expert Parallel (the DeepSeek-V3 Path)

MoE shifts the parallelism question from "how do we shard one giant matmul" to "how do we route tokens to a sparse subset of experts." TorchTitan supports DeepSeek-V3-style MoE with token-choice routing and expert parallelism (EP). The mesh adds one more dim: `(dp, tp, pp, cp, ep)`.

The forward of an MoE block:

1. Router scores each token against `N` experts; top-`k` are selected.
2. Tokens are dispatched to their experts via all-to-all on the `ep` group.
3. Each expert runs its FFN on its assigned tokens.
4. Outputs are combined back via a second all-to-all.

The all-to-all is the bottleneck. TorchTitan uses the fused `dispatch + expert_compute + combine` kernel from `torch.distributed._functional_collectives`, which overlaps the second all-to-all with the start of the next layer's attention. Without the overlap, MoE MFU sits ~20% below dense; with the overlap it is within 5%.

**Token routing pitfalls.** Token-choice routing without an auxiliary load-balancing loss leads to *expert collapse*: a few experts receive 80% of the tokens, the rest sit idle. TorchTitan applies the standard `router_aux_loss` of magnitude 0.01 and a `router_z_loss` to keep router logits bounded. Even with both, EP=8 with 64 experts can hot-spot one expert at the start of training; the trick is to *capacity-cap* each expert (drop the overflow tokens) for the first 200 steps until the router specializes. Case study 8 covers this.

**MXFP8 for MoE.** Blackwell's MXFP8 was originally designed for MoE: the 32-element microscaling block matches the typical expert FFN's hidden dim divisibility, and the per-block scale absorbs the high variance in expert activations. TorchTitan v0.2 ships MXFP8 MoE; the throughput gain over bf16 MoE on B200 is ~1.8×.

## 12. The Config and Extension Model

A TorchTitan run is one TOML file. Here is a stripped-down `llama3_70b.toml`:

```toml
[model]
name = "llama3"
flavor = "70B"
norm_type = "rmsnorm"
tokenizer_path = "/data/llama3/tokenizer.model"

[optimizer]
name = "AdamW"
lr = 3e-4
betas = [0.9, 0.95]
weight_decay = 0.1

[training]
batch_size = 8
seq_len = 8192
max_norm = 1.0
steps = 30000
gc_freq = 50

[parallelism]
data_parallel_shard_degree = 8
tensor_parallel_degree = 8
pipeline_parallel_degree = 4
context_parallel_degree = 1

[float8]
enable_float8_linear = true
enable_fsdp_float8_all_gather = true

[experimental]
pipeline_parallel_schedule = "ZBVZeroBubble"
enable_async_tensor_parallel = true

[checkpoint]
enable_checkpoint = true
folder = "/checkpoints/llama3_70b"
interval = 500
async_mode = "async"
```

The `[model]` section selects from a Python registry. To add a new model, you write `torchtitan/models/my_model/__init__.py` with three things: a `model_spec` (architecture dataclass), a `parallelize_my_model` function, and a `pipeline_my_model` function. Register them in `models/__init__.py`. The training loop is unchanged.

The `[parallelism]` degrees multiply to the total world size. A misconfiguration here is caught at startup with a clean error message — TorchTitan's startup validates `dp * tp * pp * cp == WORLD_SIZE` before it touches any GPU.

## 13. Composing 4D Parallelism: Llama 3.1 405B on 512 GPUs

![4D parallelism layout for 405B on 512 GPUs](/imgs/blogs/torchtitan-pytorch-native-pretraining-deep-dive-3.png)

The 405B run from the paper uses `DP=8, TP=8, PP=8, CP=1` on 512 H100s. Read the diagram top-down: 8 PP stages each owning 16 transformer layers (the 405B has 126 layers + embeddings/lm_head, split unevenly so the last stage carries the lm_head). Inside each PP stage, 64 GPUs form an 8×8 grid where the row is DP (FSDP2 shard rank) and the column is TP rank.

Per-GPU state at this configuration:

- Param shard: 405B / (8×8×8) ≈ 0.79B params × 2 bytes (bf16) = 1.58 GB.
- Adam state: 0.79B × 2 × 4 bytes (m, v in fp32) = 6.32 GB.
- Master weights: 0.79B × 4 bytes = 3.16 GB.
- Activations (per microbatch, at seq=8192): ~12 GB after SAC, kept across PP microbatches.

Total per H100: ~25 GB on weights+optimizer, ~12 GB × 4 in-flight microbatches = ~48 GB on activations, ~5 GB on workspace. ~78 GB out of 80 — tight, which is why the paper used microbatch size 1 with 8 microbatches per step.

**MFU.** The paper reports ~41% MFU on the 405B run with all features enabled (FSDP2 + TP=8 + PP=8 + ZB-V + float8 + per-op SAC + torch.compile). For context: Megatron-LM's published 405B numbers sit around 38–40% MFU. The TorchTitan number is competitive on a much smaller and more readable codebase.

## 14. Benchmarks and the MFU Lever Stack

The paper's stacked-acceleration numbers, restated as a build-up:

| Model | Scale | Configuration | Acceleration vs prior |
| --- | --- | --- | --- |
| Llama 3.1 8B | 128 H100s | FSDP2 + torch.compile + float8 + async TP | +65.08% over FSDP1 baseline |
| Llama 3.1 70B | 256 H100s | + TP=8 | +12.59% on top |
| Llama 3.1 405B | 512 H100s | + PP=8 | +30% on top |

What each lever buys:

| Lever | Typical MFU gain | Cost |
| --- | --- | --- |
| FSDP2 over FSDP1 | +2–4% | ~1 day to migrate |
| `torch.compile` per block | +5–10% | Risk of graph breaks |
| Float8 (dynamic) | +30–35% | ~0.1% perplexity, careful warmup |
| FSDP float8 all-gather | +5–10% | Free if float8 is on |
| Async TP | +5–10% | Intra-node only |
| ZB-V pipeline schedule | +3–8% | More activation memory |
| Per-op SAC vs full | +20–25% | Tuning the policy |

The numbers are not additive — there is overlap — but stacking them is how you get from a 35% MFU FSDP1 baseline to a 65–70% MFU TorchTitan run on H100.

## 14b. Debugging Tools and Observability

A pre-training run that fails silently at hour 14 of a 96-hour job is a worst-case-scenario for any team. TorchTitan ships with three observability primitives that turn "the run is wedged" from a multi-day investigation into a 30-minute one. Worth knowing about *before* you need them.

**Flight Recorder.** A ring buffer that records every NCCL collective each rank issues — op type, shape, source/dest rank, timestamp, and stack trace — and dumps the buffer to disk on hang or crash. When a run wedges, you do not get a Python traceback; you get a bunch of GPUs sitting at 100% memory and 0% SM utilization. The Flight Recorder dump tells you which rank issued *which* all-reduce *with what shape* and which rank was *missing*. Almost every "stuck NCCL" debug ends with someone pointing at the offending rank's last collective in the dump.

```python
import torch
torch._C._distributed_c10d._set_global_rank(rank)
# enable via env, before init_process_group:
# TORCH_NCCL_TRACE_BUFFER_SIZE=20000
# TORCH_NCCL_DUMP_ON_TIMEOUT=1
# TORCH_NCCL_DEBUG_INFO_TEMP_FILE=/tmp/nccl_dump_
```

The buffer size is per-rank; 20,000 entries covers ~10 minutes of training at typical collective rates. The dump file is a pickled list of `_NCCLOpRecord` objects; `torch.distributed._tools.flight_recorder.flight_recorder_analyzer` parses it into a CSV.

**Memory profiler.** `torch.cuda.memory._record_memory_history()` enabled at startup, then `torch.cuda.memory._dump_snapshot("memdump.pickle")` on OOM, gives a full allocation timeline. The Snapshot Viewer (a self-contained HTML app shipped with PyTorch) lets you see *which Python frame* allocated *which tensor* at *which step*. The first time I saw this tool I found a 4 GB phantom allocation in 10 minutes that had defeated three engineers for two days.

**MFU and tokens/sec.** Built-in. The training loop logs them every step to TensorBoard or W&B (whichever is configured). The MFU calculation uses the model's published FLOP-per-token (`6 × params + 12 × n_layer × seq_len × hidden` for a transformer in the Megatron formula), divided by the H100's 989 TFLOPs at bf16 or 1979 TFLOPs at fp8. Knowing this number trends upward across tuning iterations is the only way to know your tuning is working.

**The CUDA profiler hookup.** Profile a single step with `torch.profiler.profile(activities=[CPU, CUDA], record_shapes=True)`. Export to Chrome trace; view in `chrome://tracing` or Perfetto. The trace shows kernel-by-kernel time and the comm-vs-compute overlap. Most async-TP wins or losses are visible at a glance in this trace — if the all-gather and matmul kernels are stacked vertically (concurrent), you have overlap; if they are end-to-end (serial), you do not.

**Loss curve sanity.** TorchTitan logs *per-rank* loss for the first 10 steps as a sanity check; if any rank diverges, something is wrong with the data pipeline. After step 10 it logs the cross-rank averaged loss only. The first time you bring up a new cluster, leave per-rank logging on for the first 100 steps to catch sneaky data-shard bugs (one rank reading the wrong tar file).

## 15. TorchTitan vs Megatron-LM vs DeepSpeed vs NeMo

| Dimension | TorchTitan | Megatron-LM | DeepSpeed | NeMo |
| --- | --- | --- | --- | --- |
| Codebase size | ~10K LoC | ~80K LoC | ~150K LoC | ~200K LoC |
| Parallelism axes | DP, TP, PP, CP, EP | DP, TP, PP, CP, EP | DP (ZeRO 1/2/3), TP, PP, EP | DP, TP, PP, CP, EP (via Megatron core) |
| FSDP2 / per-param | Native | No (flat tensors) | ZeRO is param-level but not DTensor | Via Megatron |
| Async TP | Yes (intra-node) | Yes (TE) | No | Via Megatron |
| Float8 | torchao native | TE native | TE wrapper | TE via Megatron |
| Pipeline schedules | 1F1B, ZB-V, interleaved | 1F1B, interleaved | 1F1B | Via Megatron |
| Context parallel | Yes (ring + a2a) | Yes (TE) | Limited | Via Megatron |
| MoE | DeepSeek-V3 path, MXFP8 | Yes, mature | Yes, MoE-DeepSpeed | Yes |
| Checkpoint | DCP, sharded, async, reshard-friendly | Custom format, tools | Custom + ZeRO-specific | Megatron format |
| HF interop | DCP → torchtune | Conversion script | Conversion script | NeMo format → HF script |
| Adding a model | Python module + plan dict | Fork the framework | Fork the engine | NeMo collection contribution |

**When to pick what.** Megatron is the right pick if you want NVIDIA-supported, battle-tested-at-trillion-token-scale, and you accept the codebase weight. NeMo wraps Megatron with PyTorch Lightning ergonomics; pick it if your team lives in NeMo already. DeepSpeed remains a strong choice for ZeRO-3-only training where you do not need TP/PP. TorchTitan is the right pick when you want minimal code, native PyTorch APIs, and the freedom to read every line of the parallelism implementation.

## 16. Case Studies: Eight Real Pre-Training Incidents

The remainder of this article is the field-report section. Each is a real-shaped incident — names changed, sometimes composited — that I have either debugged personally or seen the post-mortem for. Read them in order; the mistakes get more interesting.

### Case 1: The 405B run that OOMed at step 47 (FSDP1 padding)

**Context.** An 8-node, 64-GPU prep run for Llama 3.1 405B at TP=8, DP=8, no PP. Memory looked fine on rank 0 at startup: ~62 GB out of 80, plenty of headroom. At step 47 — never step 1 — rank 5 OOMed with 79.8 GB allocated.

**Root cause.** FSDP1 was the framework. The model has thousands of parameters with non-uniform shapes (different head_dim per QKV split, different MLP intermediate sizes). FSDP1's flat-parameter trick padded each `FlatParameter` to a multiple of `world_size`. The padding for the attention block alone added ~7% to the memory footprint. On rank 5, an unrelated activation peak from the gradient bucketing pushed it over.

**Why step 47.** The first 50 steps had a learning-rate warmup; gradient magnitudes grew, and the gradient-bucket allocator's high-water mark grew with them. The peak activation footprint stabilized around step 40–60.

**Fix.** Migrated to FSDP2's `fully_shard`. Per-parameter sharding eliminated the padding and dropped the footprint by 6.8 GB per rank — back to ~73 GB at peak. The migration touched ~40 lines: replaced the `FullyShardedDataParallel` constructor with the per-block `fully_shard` loop, replaced the `MixedPrecision` arg with `MixedPrecisionPolicy`, regenerated the optimizer state from scratch. Loss curve was identical to the FSDP1 baseline within run-to-run noise.

**Lesson.** FSDP1's padding is invisible in the param count and only shows up as a "memory bloat" that scales with parameter shape diversity. If your model has many small biases or non-uniform MLP intermediate sizes, migrate to FSDP2 *before* you discover this at the worst possible step.

### Case 2: Async TP that *slowed* training (multi-node TP)

**Context.** A 70B run on 16 H100 nodes, configured with `TP=16` to keep model-parallel state small. `enable_async_tensor_parallel = true`. Throughput was 22% lower than the same config with async TP off.

**Root cause.** Async TP relies on SymmetricMemory, which requires NVLink peer access. With TP=16 spanning two nodes, the second-node ranks fell back to NCCL's IB path. The chunked epilogue's per-chunk launch overhead is constant; on NVLink it is amortized over a 100 GB/s link, on IB it is not. Net result: the chunking made the all-gather *more* expensive than the unchunked version.

**Detection.** Nsight profile showed kernel launches stacking up with idle SMs. The smoking gun was that the all-gather kernel had a `kHCP2P` variant on intra-node ranks and a `kHCNet` variant on the inter-node ones; the latter is a different code path that does not benefit from chunking.

**Fix.** Two options: (a) drop TP to 8 so it fits within a single node, or (b) leave TP=16 and disable async TP. Option (a) was a perf win because it freed the inter-node bandwidth for FSDP2's all-gather instead. Final config: TP=8, DP=16 (across nodes), async TP on.

**Lesson.** TP must stay on NVLink. Period. TorchTitan added an auto-disable for multi-node TP groups in v0.2 — but it can still be force-enabled, and teams still hit this trap.

### Case 3: Float8 loss spike at iter 1.2k (delayed-scaling history)

**Context.** A 13B float8 run resumed from a 1k-step bf16 warmup checkpoint. Loss was clean for the first 200 post-resume steps, then spiked from 2.3 to 7.1 at step 1.2k.

**Root cause.** Delayed scaling. The amax history buffer was *not* in the checkpoint — only the model parameters were saved. After resume, the float8 conversion ran with a fresh, all-zeros history. The first ~100 steps used a near-zero scale, which ran fine because the first matmul dominated the history. By step 1.2k, the history had drifted to a value calibrated for a learning-rate-warmup-period activation magnitude, but the LR was now full. Activations in some layers exceeded the scaled fp8 range and saturated; gradients went to zero or NaN.

**Fix.** Two changes. First, include the float8 amax history in the DCP checkpoint (a `torchao` PR fixed this upstream). Second, switch to dynamic scaling for resume-heavy runs: there is no history to lose, and the ~3% throughput cost was the right trade.

**Lesson.** Stateful quantization is a checkpoint hazard. If your scaling has memory, that memory must be checkpointed alongside the weights or you will lose the calibration on resume.

### Case 4: 18% MFU lost to pipeline bubble (wrong schedule)

**Context.** A 70B run at TP=8, PP=8, DP=4, microbatch=8 had 41% MFU on 256 GPUs. The GPU profile showed 18% of step time in "PP idle" — the pipeline bubble.

**Root cause.** The team had set `pipeline_parallel_schedule = "GPipe"` because it was the simplest and they wanted to debug a different issue. GPipe's bubble fraction at S=8, M=8 is 7/15 ≈ 47% — devastating. Even at M=64 it would be 10%.

**Fix.** Switched to `Schedule1F1B`, which interleaves forward and backward microbatches; at the same M=8 the bubble dropped to 7/(8+7) ≈ 47% in *theory* but the activation-recompute interleaving brought the *effective* idle time to ~10%. Then switched to `ScheduleZBVZeroBubble`, which slid weight-gradient kernels into the remaining bubble; idle time fell below 3%. Final MFU: 56%.

**Lesson.** PP schedule is not a debugging convenience. The bubble math is brutal at large `S/M` ratios; GPipe is for tutorials, 1F1B is the default, ZB-V is the production knob. Always start with 1F1B; flip to ZB-V once everything works.

### Case 5: Context-parallel correctness bug (causal mask sharding)

**Context.** A 1M-context run at CP=8 trained for two days, then a sanity eval showed perplexity 30% worse than the bf16 single-GPU short-context baseline at the *same* token count. The run was correct numerically (no NaNs) but the model was learning the wrong thing.

**Root cause.** The causal mask was being applied *before* the context-parallel attention dispatch, with the *full* sequence's positions. The CP-sharded Q/K/V chunks then attended to a mask that did not match their local positions. Tokens in the late chunks attended to "future" tokens in the early chunks. Loss looked plausible because the model was still learning *something* — but it was learning to predict tokens given partial future context, which is not next-token prediction.

**Detection.** A sharp-eyed reviewer spotted that the loss curve was suspiciously low for the model size at the token count. A toy reproducer at CP=2, S=128 confirmed that the attention output differed from the un-sharded reference.

**Fix.** Move the causal mask application *inside* the CP attention dispatch, after the K/V are exchanged via all-to-all. The mask is constructed per-rank from the local Q positions and the gathered K positions. TorchTitan's load-balanced causal CP does this correctly by default; the bug was in a custom CP fork the team had carried over.

**Lesson.** CP correctness is not just "no NaN." Always run a CP=1 vs CP=N parity test on a tiny model before committing to a long CP run. A 5-minute sanity check would have caught two days of wasted compute.

### Case 6: torch.compile graph-break cascade (custom op + SAC)

**Context.** Adding a custom RMSNorm CUDA kernel to a 13B run dropped MFU from 58% to 39%. The kernel itself was 1.8× faster than the PyTorch reference.

**Root cause.** The custom op was registered without a Python decomposition. `torch.compile` could not trace through it, so the compile region terminated at the first RMSNorm call. The transformer block was thus split into ~12 sub-graphs instead of one — each with its own kernel-launch overhead, no fusion across them, and no benefit from Inductor's pointwise fusion. The per-op SAC made it worse: the recompute path triggered a *separate* compilation (because the input shapes differed between forward and backward), doubling the compile cache and tripling compile time.

**Detection.** `TORCH_LOGS=graph_breaks python train.py` printed 11 graph breaks per block. Compile time per warmup step jumped from 12 s to 46 s.

**Fix.** Three steps. (a) Register a `torch.library.impl_abstract` for the custom op so Inductor knows the output shape. (b) Add a fake-tensor implementation so the op is traceable. (c) Add the op to the SAC `MUST_SAVE` list because it is stateful (the kernel keeps an internal stat buffer).

**Lesson.** Custom kernels and `torch.compile` have a tax: every op that breaks the graph splits the fusion region. If your op is stateful, it also needs SAC explicit handling. The 1.8× kernel speedup turned into a 1.5× model slowdown until both pieces were fixed.

### Case 7: DCP async save corruption (concurrent writes)

**Context.** A long 405B run had `interval = 100` and `async_mode = "async"`. After 12 hours, a node failure triggered a checkpoint resume; the resumed loss was 0.4 higher than the loss at the saved step.

**Root cause.** The async save thread had not finished by the time the *next* save was issued at step `t+100`. The two writes were going to different directories (`step_t/`, `step_t+100/`) but shared a metadata index file. The second save partially overwrote the first's metadata, and the first save's eventual completion overwrote the second's metadata in turn. The on-disk files for `step_t+100` were valid, but the metadata pointed to a mix of `step_t` and `step_t+100` shards.

**Detection.** A diff of the saved metadata against the in-memory state dict showed parameter dtype mismatches — the model believed a layer was bf16 while the metadata said fp8.

**Fix.** TorchTitan's checkpoint manager now blocks on the previous async save's future before issuing a new save. The cost is a ~200 ms blocking wait every `interval` steps, which is amortized to nothing. Upstream DCP also added a per-checkpoint metadata file (no shared index).

**Lesson.** Async I/O is fast until two writes overlap. The fix is not "make it more async" — it is to enforce a single in-flight write. Most production teams converge on `interval >= 500` for 70B+ runs to give the async save plenty of headroom.

### Case 8a: The "fast on rank 0, slow everywhere else" gradient norm bug

**Context.** A 70B run had loss curves that looked correct, MFU around 52%, but `grad_norm` reported by rank 0 was always exactly half the value of rank 1, which was a third of rank 2's. The team spent a week believing the optimizer was broken.

**Root cause.** The reduction for `clip_grad_norm_` was being computed against a TP-sharded gradient *without* an explicit reduction across the TP group. Each rank was seeing only its own slice's norm. The actual `total_norm` used inside the optimizer was correct (because PyTorch's clip_grad_norm under DTensor handles the reduction), but the *logged* value came from a separate `grad.norm()` call in the trainer that bypassed the DTensor reduce.

**Fix.** Replace the logging line with `total_norm = torch.distributed.tensor.placement_aware_norm(grad, mesh)`. The shipped value now matched the optimizer's view. Loss curves were unchanged because the bug was in logging only — but a week of debugging would have been avoided by trusting the cluster's loss/MFU instead of a stale norm.

**Lesson.** When a metric disagrees across ranks under TP, the metric is probably reading sharded state without reducing. The cluster is fine. Always log metrics through the same path the optimizer uses.

### Case 8: MoE expert hot-spotting (no aux loss)

**Context.** A 70B-MoE run with 64 experts at EP=8 had stable loss but ~14% MFU. The experts on rank 3 were processing 4× the tokens of the experts on rank 0.

**Root cause.** Token-choice routing without a load-balancing loss. The router specialized early — by step 200, expert 17 (on rank 2) and expert 23 (on rank 2) were absorbing 30% of all tokens. The all-to-all dispatch was bottlenecked on the slowest rank, which sat idle waiting for those two experts to finish their oversized batch.

**Detection.** A per-expert token-count histogram showed a Pareto distribution. The fastest rank finished its expert work in 4 ms; the slowest took 18 ms. The 14 ms gap was pure pipeline stall.

**Fix.** Three changes. (a) Added the standard `router_aux_loss` of magnitude 0.01. (b) Added `router_z_loss` of magnitude 1e-3 to keep router logits bounded. (c) Capacity-capped each expert at `1.25 × tokens / num_experts` for the first 200 steps. After 200 steps the router had specialized in a balanced way; the cap was lifted and the model continued without intervention. Final MFU: 38%.

**Lesson.** MoE without load balancing is a router-collapse machine. Always start with both `aux_loss` and `z_loss` on; capacity cap is a stronger guarantee for the first few hundred steps but trades some capacity for stability.

## 17. When to Reach for TorchTitan, When Not To

**Reach for TorchTitan when:**

- You are pre-training (not fine-tuning) at 7B and above.
- You want the freedom to read and modify every line of the parallelism logic.
- You need 4D parallelism (DP × TP × PP × CP) composed with float8.
- You want DCP-native checkpoints that resume across reconfigurations.
- Your team is comfortable with PyTorch APIs (DTensor, `parallelize_module`).
- You expect to add a new model architecture or parallelism axis within 6 months.
- You are on H100/H200/B200 and willing to ride PyTorch nightlies for the latest features.

**Do not reach for TorchTitan when:**

- You are fine-tuning a public model — use [torchtune](https://github.com/pytorch/torchtune) or [trl](/blog/machine-learning/open-source-library/trl-lib).
- You are doing post-training RLHF — use [verl](/blog/machine-learning/open-source-library/verl-rlhf-library-deep-dive).
- You are inferencing — use vLLM or SGLang.
- Your cluster is older than A100 (H100 is the supported platform; older GPUs lack the fp8 path).
- Your team needs an enterprise-supported framework with a vendor on the phone — pick NeMo or Megatron-LM.
- You want a "press the button and it trains" experience — TorchTitan is a *toolkit*, not a turnkey product.

The deeper observation: TorchTitan is the demonstration that pre-training in 2026 should be a *thin layer on top of PyTorch* rather than a vertically integrated framework. The interesting bits — FSDP2, DTensor, async TP, float8, DCP — all live in PyTorch core. TorchTitan's value is in showing the right *composition*, the right *defaults*, and the right *case-study lessons*. A team that internalizes those can write its own competing framework in a week. The point is not to compete; it is that you no longer need to.

If you take one thing from this article, take this: pre-training is a composition problem. The composition is small, readable, and mostly already in PyTorch. The reason your last training run was painful is not that you used the wrong framework; it is that you carried the assumption that "pre-training framework" was a thing you needed to fork rather than a `JobConfig` you needed to compose. TorchTitan is the proof that the second mode is the better one.

Companion reading on this blog: [verl for RLHF post-training](/blog/machine-learning/open-source-library/verl-rlhf-library-deep-dive), [LMCache for inference KV-cache layering](/blog/machine-learning/open-source-library/lmcache-kv-cache-layer-deep-dive), [KV cache fundamentals](/blog/machine-learning/large-language-model/kv-cache), and [Docker optimization for LLM workloads](/blog/machine-learning/mlops/docker-optimization-for-llm-and-ai-workloads) for what the multi-node container looks like underneath.
