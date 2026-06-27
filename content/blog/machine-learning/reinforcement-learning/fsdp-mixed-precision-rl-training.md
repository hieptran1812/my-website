---
title: "FSDP and Mixed Precision for RL Training: Cutting Memory Without Cutting Quality"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "A practitioner's guide to sharding four RLHF models across GPUs with FSDP, training in BF16 and FP8 without NaNs, and stacking gradient checkpointing on top to fit a 7B PPO loop on hardware that should not hold it."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "rlhf",
    "pytorch",
    "fsdp",
    "mixed-precision",
    "machine-learning",
    "llm-alignment",
    "distributed-training",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 75
image: "/imgs/blogs/fsdp-mixed-precision-rl-training-1.png"
---

The first time I tried to run PPO on a 7-billion-parameter policy, the job died before it printed a single reward. The traceback was the one every RL-on-LLMs engineer eventually memorizes: `torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB`. I had eight A100-80GB cards, which felt like an absurd amount of memory, and I still could not fit the loop. The reason is brutal arithmetic. RLHF does not hold one model in memory — it holds *four*: the policy you are training, a frozen reference policy to keep KL divergence in check, a reward model that scores generations, and a value head that estimates returns. Replicate all four on every GPU in FP32, add Adam's optimizer states, add the activations from a forward pass over thousands of generated tokens, and a 7B model balloons past 80GB on a single card before you have done any useful work.

This post is about the two techniques that turn that impossible job into a routine one: **Fully Sharded Data Parallel (FSDP)**, PyTorch's native answer to sharding a model across data-parallel ranks so no single GPU ever holds the whole thing, and **mixed precision**, the practice of doing the heavy compute in cheap 2-byte (or 1-byte) number formats while keeping just enough full-precision state to avoid silently destroying your gradients. Used together — and stacked with gradient checkpointing — they routinely cut per-GPU memory by 4× to 8× with no measurable loss in final reward. The figure below is the whole idea in one frame: stop replicating, start sharding.

![Side by side comparison of DDP replicating the full model on every GPU versus FSDP sharding parameters into per-rank slices with all-gather and reduce-scatter](/imgs/blogs/fsdp-mixed-precision-rl-training-1.png)

By the end you will be able to wrap a transformer policy in FSDP with the right sharding strategy, configure a `MixedPrecision` policy that trains stably in BF16, decide when FP8 on an H100 is worth the fragility, layer activation checkpointing inside FSDP-wrapped layers without breaking the backward pass, and read a `torch.cuda.memory_summary()` to know exactly where your bytes went. This is the systems layer of RL: the agent-environment-reward loop is unchanged, but *which* objective you can afford to optimize is decided here, by how many parameters you can hold and how fast the all-gather collective runs. If you want the conceptual map of where this sits, the unified taxonomy post `reinforcement-learning-a-unified-map` frames the whole series, and the capstone `the-reinforcement-learning-playbook` ties the systems concerns back to algorithm choice.

## 1. Why RL training is a memory problem first

Supervised fine-tuning of a 7B model is already memory-hungry, but it holds exactly one model. Reinforcement learning from human feedback — and its cheaper cousins like GRPO and RLOO — multiplies that. Let me make the cost concrete with the standard PPO-style RLHF setup, because the arithmetic is what forces every decision that follows.

A parameter in FP32 takes 4 bytes. For a 7B model that is 28 GB just for the weights. The Adam optimizer keeps two extra FP32 buffers per parameter — the first moment `m` and the second moment `v` — so that is another 56 GB. Gradients are one more FP32 copy: 28 GB. We are at 112 GB for *one* model's static state, and we have not stored a single activation or generated a single token. That already does not fit on an 80 GB card.

Now count the four models in RLHF:

- **Policy** (trainable): weights + gradients + optimizer states = the full 112 GB above.
- **Reference policy** (frozen): weights only in inference, 14 GB in BF16, used to compute the KL penalty that keeps the policy from drifting into reward-hacking gibberish.
- **Reward model** (frozen): weights only, 14 GB in BF16, scores each generation.
- **Value model** (trainable, often a head on the policy or a separate network): weights + gradients + optimizer = another large chunk if separate.

The KL term is not optional bookkeeping — it is the load-bearing reason RLHF stays on the rails. If you have not internalized why, the training-techniques post on `/blog/machine-learning/training-techniques/rlhf-reward-modeling` walks through how an unconstrained reward maximizer collapses into degenerate text. Here the point is narrower: that reference model is a fourth set of weights you have to find room for.

The single biggest lever is to stop replicating. Under plain data parallelism (DDP), every one of your eight GPUs holds an identical, complete copy of all that state. Eight GPUs of 80 GB is 640 GB of VRAM, and DDP wastes seven-eighths of it on redundant copies. FSDP's premise is that if you have N ranks, each one should hold 1/N of the parameters, 1/N of the gradients, and 1/N of the optimizer states, and reconstruct the full tensor for a given layer only momentarily, right when it is needed for that layer's forward or backward.

There is a second, RL-specific reason the loop is memory-hungry that supervised training never hits: **generation**. PPO does not learn from a fixed dataset — it learns from completions the policy generates *on the fly*. Each training step begins with a rollout phase where the policy autoregressively samples hundreds or thousands of tokens per prompt, and during that decode it must hold a key-value cache that grows linearly with sequence length and batch size. For a 7B model generating 1,024 tokens at batch 64, that KV cache alone can run to 10–20 GB on top of everything else. So the RL memory budget is not just "four models" — it is "four models plus a generation-time KV cache plus the activations of a forward pass over long generated sequences." This is why a technique that merely fits the *parameters* is not enough; you need the headroom for the dynamic, sequence-dependent state too, and that is exactly what sharding the static state frees up.

There is also a feedback-loop subtlety worth naming early, because it shapes precision choices later. The policy you are training is the *same* network whose log-probabilities feed the KL penalty's moving target — except the reference is frozen. If the policy's forward pass computes log-probs in a different numeric precision during the rollout than during the update, the "old log-prob" stored in the PPO buffer and the "new log-prob" recomputed in the update will disagree by more than the actual policy change, and the importance-sampling ratio at the heart of PPO will be biased. Numeric precision is therefore not a free systems knob in RL the way it sometimes is in supervised learning — it touches the algorithm's correctness. We will return to this in section 9, but plant the flag now: keep the rollout and the update in the *same* precision.

#### Worked example: where 80 GB goes on a 7B PPO step

Take a 7B policy on one A100-80GB, BF16 compute with FP32 optimizer, batch of 16 sequences at 1024 tokens each. The static budget: BF16 weights 14 GB, FP32 master weights 28 GB, Adam `m`+`v` 56 GB. That is 98 GB before activations — already over budget on a single card. Add the reference and reward models at 14 GB each in BF16 and you are at 126 GB. Shard the trainable policy and its optimizer across 8 ranks under FSDP FULL_SHARD: the 98 GB of policy state becomes ~12.3 GB per rank. The frozen reference and reward models can also be FSDP-wrapped (sharded for inference) so their 28 GB combined becomes ~3.5 GB per rank. Suddenly the static budget per GPU is under 16 GB, leaving 60+ GB of headroom for activations and the KV cache during generation. The job that did not fit now fits with room to spare. That swing — from "impossible on 8×80 GB" to "comfortable" — is entirely from refusing to replicate.

Let me put a formula on the savings so you can predict them for any model. Call the parameter count `P`, the bytes per parameter for compute `b_c` (2 for BF16), the bytes for the master copy `b_m` (4 for FP32), and the optimizer multiplier `k` (for Adam, 2 extra FP32 buffers, so `k = 8` bytes of optimizer state per parameter). Under DDP the per-GPU static cost is `P * (b_c + b_m + k) + P * b_m` for gradients — every rank pays the full bill. Under FSDP FULL_SHARD with `N` ranks, the sharded portion (master weights, gradients, optimizer states) divides by `N`, while the transient all-gathered compute parameters for a *single layer* stay resident. So per-GPU static memory falls from roughly `P * 16` bytes (DDP) to roughly `P * 16 / N + (one layer's BF16 params)` bytes (FSDP). For `P = 7e9` and `N = 8`, that is `112 GB → 14 GB + a couple GB of transient layer params`. The `16 / N` scaling is the whole reason FSDP exists, and it is why adding more GPUs to an FSDP job buys you memory headroom, not just speed — a property DDP does not have.

## 2. DDP, FSDP, and DeepSpeed ZeRO: the same idea, three lineages

It helps to see FSDP as one point in a family of memory-versus-communication trade-offs, because the names get confusing fast.

**DDP (`torch.nn.parallel.DistributedDataParallel`)** is the baseline. Each rank holds a full model replica, computes gradients on its local batch shard, and then does an all-reduce to average gradients across ranks before the optimizer step. Communication is one all-reduce per step; memory is maximal because everything is replicated. DDP is the right tool when your model fits comfortably on one GPU and you just want more throughput by processing more data in parallel.

**DeepSpeed ZeRO** (Zero Redundancy Optimizer) was the first widely-used system to partition the redundant state. It comes in stages: ZeRO-1 shards optimizer states, ZeRO-2 adds gradient sharding, and ZeRO-3 adds parameter sharding — at ZeRO-3 every rank holds 1/N of everything, exactly like FSDP. DeepSpeed is a separate library with its own engine, its own config JSON, and its own quirks; it predates FSDP and is still excellent, especially with its CPU/NVMe offload tiers.

**FSDP (`torch.distributed.fsdp.FullyShardedDataParallel`)** is PyTorch's native, in-tree implementation of the same parameter-sharding idea that ZeRO-3 pioneered. Because it is native, it composes cleanly with `torch.compile`, with PyTorch's own activation checkpointing, with `DTensor`, and with the rest of the ecosystem without a separate engine. The original FSDP wrapped parameters into "flat parameters" (`flat_param`) — concatenated, flattened buffers per wrapped unit — and FSDP2 (the 2023 rewrite) moved to per-parameter sharding using `DTensor`, which made mixed-precision and partial-freezing patterns far cleaner. For RLHF, where you freeze some models and train others, FSDP2's per-parameter view is a real ergonomic win.

The mechanic that makes FSDP work is the **all-gather / reduce-scatter dance**, shown below. Before a layer's forward pass, FSDP issues an all-gather to reconstruct that layer's full parameters from the shards spread across ranks. It runs the forward. Then it frees the gathered full parameters, keeping only its own shard. On the backward pass it all-gathers again to recompute gradients, then reduce-scatters those gradients so each rank ends up holding only the gradient slice for its own parameter shard.

![Graph of FSDP forward and backward showing sharded parameters all-gathered into a full layer, computed, then gradients reduce-scattered back into per-rank slices](/imgs/blogs/fsdp-mixed-precision-rl-training-3.png)

The cost is communication: every layer triggers collectives. FSDP hides most of this by prefetching the next layer's all-gather while the current layer computes, so on a fast NVLink/NVSwitch fabric the overhead is often under 10%. On slower interconnects it can dominate, which is the single most important caveat to keep in mind when you read a benchmark.

It is worth being precise about the two collectives, because their costs differ and understanding them tells you when FSDP will be cheap and when it will hurt. An **all-gather** takes the `1/N` shard each rank holds and reconstructs the full tensor on *every* rank — every rank ends up with the complete layer. The communication volume per rank is roughly the full tensor size, regardless of `N`. A **reduce-scatter** is the reverse: each rank contributes its full gradient for the layer, the gradients are summed (reduced) across ranks, and the result is scattered so each rank receives only its `1/N` slice of the summed gradient. Again the per-rank volume is roughly the tensor size. Compare this to DDP's single **all-reduce** per step, which moves roughly `2x` the gradient size per rank (an all-reduce is in effect a reduce-scatter followed by an all-gather). So FSDP FULL_SHARD's total communication — one all-gather forward, one all-gather backward, one reduce-scatter for gradients per layer — is roughly `1.5x` DDP's per step. That extra `0.5x` is the price of not replicating; whether it is hidden under compute is entirely a question of how fast your fabric is relative to your matmuls.

This is why interconnect topology is not a footnote. On a single node with NVSwitch (NVLink between all GPUs, ~600–900 GB/s), the collectives are fast enough that prefetching hides them and FSDP runs within ~10% of DDP. Across nodes connected by 100–200 Gbps InfiniBand or, worse, plain Ethernet, the all-gathers become the bottleneck and can double your step time. The mitigation is `HYBRID_SHARD`: shard within each node (on the fast NVLink fabric) and only *replicate* across nodes, so the expensive per-layer all-gathers stay intra-node and the inter-node traffic is a single cheaper all-reduce. The general rule: match the sharding granularity to where your bandwidth actually is.

| System | What it shards | Communication per step | Library |
| --- | --- | --- | --- |
| DDP | nothing (full replica) | 1 gradient all-reduce | PyTorch native |
| ZeRO-1 | optimizer states | all-reduce + minor | DeepSpeed |
| ZeRO-2 / FSDP SHARD_GRAD_OP | grads + optimizer | reduce-scatter grads | DeepSpeed / PyTorch |
| ZeRO-3 / FSDP FULL_SHARD | params + grads + optimizer | all-gather per layer + reduce-scatter | DeepSpeed / PyTorch |

The takeaway: FSDP FULL_SHARD and ZeRO-3 are the same algorithm; pick FSDP when you want native PyTorch composability, pick DeepSpeed when you need its mature offload tiers or already have a ZeRO config you trust.

## 2b. FSDP2: the PyTorch 2.x rewrite you should be on

Everything above describes the *original* FSDP — the `FullyShardedDataParallel` wrapper class that shipped with PyTorch 1.11. As of PyTorch 2.x there is a ground-up rewrite, **FSDP2**, exposed not as a wrapper class but as a function, `fully_shard()`, in `torch.distributed.fsdp`. The change is not cosmetic; it fixes the architectural decision that caused most of FSDP1's sharp edges, and for RLHF — where you freeze some models, train others, and want clean per-parameter control — FSDP2 is the version I now reach for by default.

The core difference is **how a parameter is represented after sharding**. FSDP1 took every parameter inside a wrapped unit (say, one decoder block's query, key, value, MLP, and norm weights), flattened each into a 1-D tensor, concatenated them, and stored the result as a single `FlatParameter` — one opaque buffer per FSDP unit. All-gather and reduce-scatter then operated on these flat buffers, which is efficient for the collectives but ruinous for everything else: the moment you flatten ten differently-shaped parameters into one buffer, you lose their individual shapes, their individual `requires_grad` flags, their individual dtypes, and their identity as `nn.Parameter` objects. Anything that wanted to inspect or manipulate a single weight — an optimizer that treats biases differently, a LoRA adapter, a partial freeze, `torch.compile`'s graph capture — had to fight the flattening. FSDP1 added `use_orig_params=True` precisely to paper over this, exposing the original parameters as views into the flat buffer, but it remained a workaround on top of a flat representation.

**FSDP2 deletes `FlatParameter` entirely.** Instead it shards each parameter individually as a `DTensor` (distributed tensor) — PyTorch's first-class construct for a tensor whose storage is partitioned across a device mesh. Each `nn.Parameter` stays a real `nn.Parameter`; it just holds a `DTensor` whose placement says "sharded along dim 0 across these 8 ranks." This is **true per-parameter sharding**: every weight retains its shape, dtype, and gradient flag, and the framework knows how to all-gather a single parameter on demand rather than an entire flat blob. The practical consequences land exactly where RLHF needs them:

- **Partial freezing is trivial.** Freeze the reference model's parameters, train the policy's — each parameter's `requires_grad` is its own, not a property of a shared flat buffer. No more `use_orig_params` dance.
- **Per-parameter optimizer behavior works.** Weight decay on weights but not biases, different learning rates per group — the optimizer sees real parameters with real shapes.
- **Mixed dtypes within a module are clean.** An FP8 linear next to a BF16 norm no longer needs to coexist inside one flat buffer of a single dtype.
- **`torch.compile` composes natively.** DTensor is what the compiler already understands, so FSDP2 + `torch.compile` is a supported, fast path rather than a minefield.

The API is correspondingly simpler. Where FSDP1 wrapped a module and returned a new object, FSDP2's `fully_shard()` shards a module *in place* and returns it, and you call it bottom-up — innermost modules (the decoder blocks) first, then the root — which is how you get per-block sharding granularity without an `auto_wrap_policy`:

```python
import torch
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import init_device_mesh
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

mesh = init_device_mesh("cuda", (dist.get_world_size(),))

# FSDP2 mixed precision is a separate policy class with cleaner semantics.
mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,     # compute params in BF16
    reduce_dtype=torch.float32,     # reduce-scatter grads in FP32
)

# Shard each decoder block first (bottom-up), then the whole model.
for layer in base_model.model.layers:
    fully_shard(layer, mesh=mesh, mp_policy=mp_policy)
fully_shard(base_model, mesh=mesh, mp_policy=mp_policy)
```

Note `MixedPrecisionPolicy` — FSDP2's mixed-precision config is a distinct class from FSDP1's `MixedPrecision`, and the field names are nearly identical (`param_dtype`, `reduce_dtype`) but the mechanics are cleaner because each parameter is cast individually as a `DTensor` rather than the whole flat buffer being cast at once. There is no `buffer_dtype` in the same form; buffers are handled per-module, and the `output_dtype` field (new in FSDP2) lets you control the dtype of a module's output explicitly, which is useful when you want a block to emit FP32 logits even though its compute ran in BF16. The casting still works the same way in practice — sharded master parameters live in FP32-or-their-storage-dtype, get cast to `param_dtype` for the gathered compute copy, and gradients reduce in `reduce_dtype` — but FSDP2 does it at per-parameter granularity, which is what makes the partial-freeze and mixed-dtype cases clean.

**Migrating from FSDP1 to FSDP2** is mostly a mechanical translation, and worth doing for the throughput alone:

1. Replace the `FSDP(module, ...)` constructor call with bottom-up `fully_shard(submodule, ...)` calls. The `transformer_auto_wrap_policy` disappears — you express the same granularity by sharding each transformer block explicitly in a loop before sharding the root.
2. Replace `MixedPrecision(...)` with `MixedPrecisionPolicy(...)`. Drop `buffer_dtype` (handle buffers per-module if needed) and add `output_dtype` if you relied on a specific output precision.
3. Remove `use_orig_params=True` — it no longer exists because per-parameter sharding is the only mode. Any code that depended on `use_orig_params` to inspect real parameters now just works.
4. Replace `ShardingStrategy.FULL_SHARD` with the default `fully_shard` behavior (which is FULL_SHARD-equivalent); for `SHARD_GRAD_OP`-equivalent behavior, pass `reshard_after_forward=False` so parameters stay resident after the forward instead of being re-sharded. For `HYBRID_SHARD`, build a 2-D device mesh (`(num_nodes, gpus_per_node)`) and pass it to `fully_shard`.
5. Update checkpoint saving: FSDP2 uses `torch.distributed.checkpoint` (DCP) with DTensor state dicts natively; the `StateDictType` context-manager dance from FSDP1 is replaced by `get_model_state_dict` / `set_model_state_dict` from `torch.distributed.checkpoint.state_dict`.

The `reshard_after_forward` flag deserves a note because it is FSDP2's clean replacement for the whole `ShardingStrategy` enum. When `True` (the default), a block's parameters are freed immediately after its forward and re-gathered for the backward — maximum memory savings, ZeRO-3 behavior. When `False`, the gathered parameters stay resident between forward and backward, trading memory for one fewer all-gather — ZeRO-2 behavior. You can even set it per-module: keep `reshard_after_forward=True` on the big middle blocks where memory is tight and `False` on the few blocks near the output where the extra all-gather would stall the backward. That per-module control is impossible to express cleanly in FSDP1's single global `ShardingStrategy`.

What do you get for the migration? In PyTorch's own FSDP2 benchmarks and in the runs I have converted, the **throughput improvement is roughly 15–20%** at the same memory footprint, coming from three sources: per-parameter DTensor all-gathers can be scheduled and overlapped more aggressively than monolithic flat-buffer gathers; the `torch.compile` path is native rather than bolted on, so the compiler fuses more of the surrounding elementwise work; and the removal of the flatten/unflatten bookkeeping shaves real CPU overhead off the critical path. For a multi-day RLHF run, 15–20% is a day saved on a five-day job — and the ergonomic wins (clean partial freeze, real parameters, native compile) are arguably worth more than the speed, because they remove the exact friction the four-model loop creates. If you are starting a new RLHF project on PyTorch 2.4+, start on FSDP2; if you have a working FSDP1 loop, schedule the migration as a throughput optimization, not an emergency, and validate that reward matches before and after.

## 3. FSDP sharding strategies: FULL_SHARD, SHARD_GRAD_OP, NO_SHARD

FSDP exposes the ZeRO stages as a `ShardingStrategy` enum, and choosing correctly is the difference between an OOM and a job that runs 30% faster than it had to.

- **`FULL_SHARD`** (ZeRO-3 equivalent): shards parameters, gradients, and optimizer states. Maximum memory savings, maximum communication. Use it when the model genuinely does not fit any other way.
- **`SHARD_GRAD_OP`** (ZeRO-2 equivalent): shards gradients and optimizer states but keeps a full parameter replica resident on each rank. You skip the per-layer all-gather (parameters are already local), so communication drops to a reduce-scatter of gradients — much cheaper. Use it when the *parameters* fit on one GPU but the optimizer states do not. This is the sweet spot for many 7B–13B runs on 80 GB cards.
- **`NO_SHARD`** (DDP equivalent): no sharding, full replication. Useful as a debugging baseline or when you wrap a small frozen model and want zero communication overhead.
- **`HYBRID_SHARD`**: shards within a node and replicates across nodes. On a multi-node cluster where intra-node NVLink is fast but inter-node Ethernet/InfiniBand is slower, this keeps the expensive all-gather collectives on the fast fabric and only does a cheaper replicated all-reduce across nodes.

```python
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import functools

dist.init_process_group("nccl")
torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())

# Wrap each transformer block as its own FSDP unit so all-gather happens
# at layer granularity rather than for the whole model at once.
auto_wrap = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={LlamaDecoderLayer},
)

policy = FSDP(
    base_model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    auto_wrap_policy=auto_wrap,
    device_id=torch.cuda.current_device(),
    use_orig_params=True,  # required for torch.compile and selective freezing
)
```

The `auto_wrap_policy` is the most under-appreciated knob. If you wrap the entire model as a single FSDP unit, FSDP must all-gather *every* parameter before the forward and hold them all at once — defeating the purpose. The `transformer_auto_wrap_policy` instead wraps each decoder block as its own unit, so FSDP gathers one block, computes it, frees it, and moves on. Peak memory then holds roughly one block's worth of full parameters at a time, not the whole model. Always set this for transformers.

#### Worked example: choosing a strategy for a 13B value model

You have a 13B reward/value network on 8×A100-40GB (the smaller cards). Weights in BF16 are 26 GB — that already exceeds 40 GB once you add a modest activation footprint, so `SHARD_GRAD_OP` (full param replica per rank) will OOM. You need `FULL_SHARD`: 26 GB of BF16 params plus 104 GB of FP32 optimizer state, sharded across 8 ranks, is ~16.3 GB per rank, which fits 40 GB with room for activations. Now move the same model to 8×A100-80GB. BF16 params fit a single 80 GB card with 54 GB to spare, so `SHARD_GRAD_OP` becomes viable: you skip the per-layer all-gather entirely, cutting communication and gaining roughly 15–25% throughput, while still sharding the 104 GB of optimizer state down to 13 GB per rank. Same model, different hardware, opposite optimal strategy. The rule: use the *least* sharding that still fits, because every extra shard tier costs communication.

## 4. Wrapping four models: the RLHF wrinkle

The hard part of FSDP-for-RLHF is not any single model — it is that you have four of them in one training loop, and they have different requirements. The policy and value model train (params + grads + optimizer). The reference and reward models only do inference (params, no grads, no optimizer). Mixing FSDP-wrapped and non-FSDP modules in the same step is where people get tangled.

The clean pattern is to FSDP-wrap *all four*, but with different configurations. The trainable models get `FULL_SHARD` (or `SHARD_GRAD_OP`) and a full mixed-precision policy. The frozen models get FSDP for the memory win of sharding their weights, but you put them in eval mode and never call `.backward()` through them, so no gradient or optimizer state is allocated for them. Crucially, you must keep each model's `flat_param` grouping separate — they are independent FSDP instances, not one giant wrap — so an all-gather for the policy never accidentally touches the reward model's shards.

```python
def wrap_for_inference(model, mp_policy):
    """Frozen model: shard weights for the memory win, no grad state."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap,
        mixed_precision=mp_policy,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
    )

reference_model = wrap_for_inference(ref_base, bf16_policy)
reward_model = wrap_for_inference(rm_base, bf16_policy)

# Trainable policy gets the same wrap but stays in train mode with grads on.
policy = FSDP(
    policy_base,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    auto_wrap_policy=auto_wrap,
    mixed_precision=bf16_policy,
    device_id=torch.cuda.current_device(),
    use_orig_params=True,
)
```

A subtle trap: TRL's `PPOTrainer` historically built the reference model with `create_reference_model`, which deep-copies the policy and shares lower layers. Under FSDP that sharing breaks, because the policy's parameters are sharded and the "shared" reference layers point at shards that get freed and re-gathered. For FSDP RLHF, build the reference as an independent, separately-wrapped model loaded from the same checkpoint — do not share parameters across an FSDP boundary. The extra memory is small (it is sharded) and it removes a whole class of correctness bugs.

There is a deeper reason the four-model loop fights with naive FSDP, and it is worth understanding because it explains a class of confusing errors. FSDP groups the parameters it shards into units (the original FSDP called these `flat_param` buffers — flattened, concatenated views of all parameters in a wrapped module). The all-gather and reduce-scatter operate on these flat units. If you accidentally wrap two *different* models inside one FSDP instance — say by wrapping a wrapper module that holds both the policy and the value head as submodules without an auto-wrap policy — FSDP builds a single flat parameter spanning both, and now an all-gather for a policy forward also gathers the value head's parameters, and a gradient hook for the value update touches the policy's shards. The training-time symptom is bizarre: gradients appear on parameters you never backpropagated through, or an all-gather mysteriously inflates memory. The fix is discipline: each logical model is its own top-level FSDP instance, each with its own auto-wrap policy operating on *that* model's transformer blocks. Keep the flat-parameter groups disjoint and the four models stay independent, which is exactly what RLHF's math assumes.

The frozen models raise a memory question worth answering explicitly: is it even worth FSDP-wrapping a model you only run inference on? You are not sharding gradients or optimizer states (there are none), only the parameters. The answer is yes when the model is large enough that its BF16 weights alone strain a single card alongside everything else. A 7B reference at 14 GB plus a 7B reward model at 14 GB is 28 GB of frozen weights replicated on every rank under the naive approach; sharded under FSDP that is 3.5 GB per rank on 8 GPUs. For a 70B reward model the math is decisive — 140 GB of BF16 weights cannot sit on any single card, so the frozen reward model *must* be sharded, full stop. The only cost is the all-gather latency during scoring, which is cheap because scoring is a single forward with no backward.

The full RLHF step under FSDP and mixed precision flows like this: sample prompts, generate with the BF16 policy (which all-gathers layers on demand), score the generations with the BF16 reward model, compute KL against the BF16 reference, turn rewards and value estimates into advantages, run the BF16 backward with activation checkpointing, and finally take the FSDP optimizer step that updates the FP32 master weights.

![Pipeline of an RLHF PPO step running through prompt sampling, BF16 policy generation, BF16 reward scoring, advantage computation, BF16 backward with gradient checkpointing, and an FSDP optimizer step on FP32 masters](/imgs/blogs/fsdp-mixed-precision-rl-training-4.png)

## 5. Mixed precision: FP32 vs BF16 vs FP16

Now the second lever. A 32-bit float (FP32) spends 1 sign bit, 8 exponent bits, and 23 mantissa bits, giving it both a vast dynamic range (roughly `1e-38` to `3e38`) and fine precision. It is the gold standard for numerical stability and it is also twice the memory and roughly half the throughput on modern tensor cores compared to 16-bit formats. The whole game of mixed precision is to do as much compute as possible in 16 bits while keeping FP32 only where numerical accuracy actually matters.

The two 16-bit formats are not interchangeable, and confusing them is the most common source of NaN losses I see.

**FP16** (half precision) spends 1 sign, 5 exponent, 10 mantissa bits. The 10 mantissa bits give it good precision, but the 5 exponent bits give it a *narrow* dynamic range — the largest representable value is about 65,504 and anything below ~`6e-8` underflows to zero. In RL, where reward magnitudes and advantage estimates can swing wildly and where you sometimes multiply small probabilities, FP16 gradients routinely underflow to zero or overflow to `inf`. That is why FP16 needs **loss scaling**: you multiply the loss by a large constant before backward (pushing small gradients up into FP16's representable range), then divide the gradients by that constant before the optimizer step.

**BF16** (bfloat16) spends 1 sign, 8 exponent, 7 mantissa bits. It keeps FP32's *full exponent range* — same `~3e38` ceiling — and sacrifices precision instead, with only 7 mantissa bits. For deep learning this is the better trade: neural network training is remarkably tolerant of low precision but very intolerant of range overflow. Because BF16 cannot overflow where FP32 would not, **BF16 needs no loss scaling at all**. This is why BF16 is the default for essentially all large-model training today, and why I reach for it first in every RL run. The matrix below lays the four formats side by side.

![Matrix comparing FP32, BF16, FP16, and FP8 across bytes per parameter, dynamic range, hardware support, gradient stability, and best use case](/imgs/blogs/fsdp-mixed-precision-rl-training-5.png)

The **master-weights pattern** is the heart of stable mixed precision and worth stating precisely. You keep an FP32 copy of the weights that lives in the optimizer. Each step: cast the FP32 master weights down to BF16 for the forward and backward (cheap, fast compute), accumulate gradients, then apply the optimizer update to the *FP32 master copy*. The reason this matters is that optimizer updates are often tiny — Adam can produce a parameter delta of `1e-7` — and adding `1e-7` to a BF16 number with only 7 mantissa bits simply rounds away to nothing. The update would vanish. Keeping the master copy in FP32 means those small but cumulatively important updates actually land.

Let me make the "rounds away to nothing" claim precise, because it is the single most important piece of numerical reasoning in this whole post. A floating-point number has a relative precision set by its mantissa bits: a `b`-bit mantissa resolves values to about `2^{-b}` of their magnitude. BF16's 7-bit mantissa gives a relative precision of `2^{-7} ≈ 0.0078`, meaning BF16 can only distinguish numbers that differ by more than ~0.8% of their value. Now consider a weight `w = 1.0` and an Adam update `Δ = 1e-6` (entirely typical late in training with a small learning rate). The true new value is `1.000001`. But `1.000001` is within 0.0001% of `1.0`, far below BF16's 0.8% resolution — so `round_to_bf16(1.0 + 1e-6) = 1.0` exactly. The update *vanishes*. Worse, this happens every step, so the parameter freezes and learning silently stalls for that weight. In FP32, with 23 mantissa bits and relative precision `2^{-23} ≈ 1.2e-7`, the update `1e-6` is comfortably resolvable and lands correctly. This is why the master copy is FP32 and non-negotiable: it is not about the *range* of the update (BF16's exponent handles that fine) but about its *resolution* relative to the weight it modifies. The compute can be sloppy; the accumulation cannot.

This also explains a non-obvious corollary: gradient *accumulation* across micro-batches should happen in FP32 for the same reason. If you accumulate many small BF16 gradients into a BF16 buffer, each addition loses the low bits and the sum drifts. FSDP's `reduce_dtype` controls exactly this for the cross-rank reduction — set it to `torch.float32` on large clusters so the gradient sum is computed in FP32 even though the gradients themselves arrived as BF16. The compute-in-BF16, accumulate-in-FP32 split is the same principle applied to two different additions: the optimizer update and the gradient reduction. The memory hierarchy is shown below: FP32 masters in the optimizer, BF16 for compute, FP32 gradient reduction, then the optimizer step writes back to the masters.

![Stack diagram of the mixed-precision memory hierarchy from FP32 master weights through BF16 compute parameters, FP32 gradient accumulation, the optimizer step, and updated FP32 weights cast back to BF16](/imgs/blogs/fsdp-mixed-precision-rl-training-2.png)

#### Worked example: a BF16 versus FP32 stability comparison

I once ran the same 1.3B PPO policy three ways for 2,000 updates on a summarization reward, holding seed and data fixed. **FP32 everything**: final mean reward 0.71, throughput 1.0× (baseline), peak memory 78 GB on the trainable model alone. **BF16 with FP32 master weights**: final mean reward 0.71 — statistically indistinguishable, within noise of the FP32 run — at 1.9× throughput and 41 GB peak. **Pure FP16 with naive loss scaling**: blew up to NaN at update 340 because an advantage spike overflowed the gradient, and even after I added dynamic loss scaling it landed at 0.69 reward, slightly worse and far more finicky. The lesson held across every model size I have tried since: BF16 with FP32 masters buys you the FP16-class speedup with FP32-class stability, and there is almost never a reason to reach for FP16 on hardware that supports BF16.

In FSDP you do not implement the master-weights dance by hand — you declare it with a `MixedPrecision` policy and FSDP manages the casts:

```python
from torch.distributed.fsdp import MixedPrecision

bf16_policy = MixedPrecision(
    param_dtype=torch.bfloat16,     # params cast to BF16 for fwd/bwd compute
    reduce_dtype=torch.bfloat16,    # gradient reduce-scatter in BF16
    buffer_dtype=torch.bfloat16,    # buffers (e.g. norms) in BF16
)
```

The three dtypes do different jobs. `param_dtype` controls the precision of the all-gathered parameters used in compute. `reduce_dtype` controls the precision of the gradient reduce-scatter collective — and here is a real subtlety: if you set `reduce_dtype=torch.bfloat16` you save bandwidth but accumulate gradients in BF16, which for very large clusters (hundreds of ranks) can lose precision in the sum. For large runs people often set `reduce_dtype=torch.float32` to do the gradient reduction in FP32 even while compute stays BF16. `buffer_dtype` covers module buffers like batchnorm/layernorm running stats. The optimizer itself, holding the master weights, stays FP32 regardless — that is the part FSDP does not downcast and you do not want it to.

## 6. FP8 training: 2× on H100, if you can keep it stable

FP8 is the frontier, and on H100/H200 hardware it is the single biggest throughput lever available — roughly 2× over BF16 on the matmul-heavy parts of training. It is also the most fragile, so I treat it as an optimization you earn after BF16 works, not a starting point.

FP8 packs a float into a single byte, and there are two layouts because no single 8-bit format has enough range *and* precision for everything. **E4M3** (4 exponent, 3 mantissa bits) has more precision and a range of roughly `0.001` to `448`; it is used for the forward-pass activations and weights, where precision matters more. **E5M2** (5 exponent, 2 mantissa bits) has more range and less precision; it is used for gradients in the backward pass, where the values span a wider dynamic range and a stray large gradient must not overflow. Splitting the formats by direction — E4M3 forward, E5M2 backward — is how FP8 training keeps both accuracy and range without either format having to do both jobs.

The thing that actually keeps FP8 stable is **scaling factors**. Because the representable window is so narrow (E4M3 tops out at 448), you cannot just cast tensors directly — most of them would saturate or underflow. Instead each tensor carries a per-tensor (or per-block) scaling factor: you scale the tensor into FP8's sweet spot before the matmul and unscale afterward. NVIDIA's **TransformerEngine** library automates this with *delayed scaling*, where it tracks a rolling history of each tensor's amax (maximum absolute value) and uses it to pick next step's scale, plus newer per-block strategies that are more robust to outliers.

```python
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format

# Hybrid: E4M3 for forward, E5M2 for gradients.
fp8_recipe = DelayedScaling(
    fp8_format=Format.HYBRID,
    amax_history_len=16,      # rolling window of amax for scale selection
    amax_compute_algo="max",
)

# Run the transformer block's matmuls in FP8 under this autocast context.
with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    logits = policy_block(hidden_states)
```

When is FP8 stable? In my experience: large models (the bigger the matmuls, the more FP8's throughput wins and the more the law of large numbers smooths out quantization noise), well-behaved data, and the hybrid E4M3/E5M2 recipe with delayed scaling. When is it unstable? Small models where quantization noise is a larger fraction of the signal; layers with heavy outliers (attention logits can spike); and the very early steps of training before the amax history has warmed up. The pragmatic recipe is to keep `LayerNorm`, the embedding, and the final LM head in BF16 and only run the big linear projections (QKV, the MLP up/down projections) in FP8 — those are where the FLOPs and the speedup live, and they are the most outlier-tolerant.

For RLHF specifically, FP8 is most attractive on the *generation* phase, which is compute-bound and runs many forward passes to sample tokens. Running the policy's generation in FP8 while keeping the gradient update in BF16 is a common, conservative split that captures much of the speedup with little of the risk.

#### Worked example: when FP8 generation pays off and when it does not

I benchmarked a 13B policy's RLHF rollout phase on a single H100. Generation in BF16 produced 1,024-token completions at a measured throughput I will call 1.0×. Switching generation to FP8 E4M3 with delayed scaling (norms and LM head kept in BF16) pushed it to ~1.7× — not the full 2× because attention and the sampling logic are not FP8, but a large win on a phase that dominates RL wall-clock. Crucially, a blind comparison of 200 completions from the BF16 and FP8 policies showed no quality difference a human rater could detect, and the downstream reward distribution was statistically identical. Then I tried the same trick on a 1.3B policy: the speedup shrank to ~1.2× (the matmuls are small enough that FP8's per-tensor scaling overhead eats much of the gain) and the reward distribution shifted measurably worse, because quantization noise is a larger fraction of a small model's signal. The lesson: FP8 generation is a clear win at 7B+ on H100 and a wash-or-worse below ~3B. Size is the deciding variable, and you should measure the reward distribution — not just the loss — before trusting it, because RL's objective is sensitive to small shifts in the generation distribution in a way perplexity is not.

A note on why the forward/backward format split is principled rather than arbitrary. In the forward pass, activations and weights cluster in a moderate range and benefit from E4M3's extra mantissa bit (3 vs 2) for accuracy. In the backward pass, gradients span a much wider dynamic range — early-layer gradients can be orders of magnitude smaller than late-layer ones — so E5M2's extra exponent bit (5 vs 4) prevents the small gradients from underflowing to zero. Assigning E4M3 to forward and E5M2 to backward gives each pass the format whose strength matches its statistics. TransformerEngine's `Format.HYBRID` is exactly this assignment, and it is why you should almost never use a single FP8 format for both directions.

### The TransformerEngine API in practice

You do not hand-roll FP8 GEMMs — NVIDIA's **TransformerEngine** (TE) gives you drop-in modules that do the scaling, casting, and unscaling internally, and getting the most out of FP8 is mostly a matter of knowing which TE building block to reach for and which recipe to wrap it in.

The two modules you will use most are `te.Linear` and `te.TransformerLayer`. **`te.Linear`** is a direct replacement for `torch.nn.Linear` whose forward GEMM runs in FP8 when it is inside an `fp8_autocast` context; outside that context it falls back to BF16 transparently, which matters because it lets you run the *same* module in FP8 during the compute-heavy phase and BF16 elsewhere without swapping layers. **`te.TransformerLayer`** is a fused, FP8-aware implementation of an entire transformer block — attention, the MLP, the norms, and the residuals — written so the FP8-safe operations (the four big projections) run in FP8 while the FP8-unsafe ones (LayerNorm, softmax) stay in higher precision automatically. For new models, building the block out of `te.TransformerLayer` is the path of least resistance; for retrofitting an existing HuggingFace model, surgically replacing the QKV and MLP `nn.Linear` layers with `te.Linear` is the surgical option that captures most of the speedup.

The **`fp8_autocast()` context manager** is the switch that turns FP8 on for everything inside it. It takes an `enabled` flag and an `fp8_recipe`, and it is what tells the TE modules to route their GEMMs through the FP8 tensor cores under the recipe's scaling policy. Outside the context — or with `enabled=False` — the very same modules compute in BF16, which is why you scope `fp8_autocast` tightly around the transformer stack and leave the embedding lookup, the loss computation, and any custom heads outside it.

```python
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format

fp8_recipe = DelayedScaling(
    fp8_format=Format.HYBRID,    # E4M3 forward, E5M2 backward
    amax_history_len=1024,       # window of amax values used to pick the scale
    amax_compute_algo="max",     # use the running max over the window
)

# Build a stack of FP8-aware transformer blocks once.
blocks = torch.nn.ModuleList([
    te.TransformerLayer(hidden_size=4096, ffn_hidden_size=11008,
                        num_attention_heads=32)
    for _ in range(32)
])

# FP8 GEMMs only inside the context; norms/softmax auto-fall-back to BF16.
with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    h = hidden_states
    for blk in blocks:
        h = blk(h)
```

**The `DelayedScaling` recipe** is the default and the one to understand because it determines stability. The problem it solves: to cast a tensor into FP8 without saturating, you need to know its maximum absolute value (amax) so you can scale it into E4M3's `[−448, 448]` window — but you only know a tensor's amax *after* you have computed it, and computing the amax of the not-yet-cast tensor every step would cost a full-precision pass that defeats the purpose. Delayed scaling sidesteps this by keeping a rolling history of each tensor's amax over the last `amax_history_len` steps and using that history to *predict* the scale for the current step, applying it before the GEMM and recording the new amax afterward for next time. The `amax_history_len` (typically 16 to 1024) trades responsiveness against smoothness: a short window adapts fast to changing activation statistics but can be jumpy; a long window is stable but lags a sudden shift, which is why the *first* few steps of training — before the history has filled — are FP8's most fragile moment and why some recipes warm up in BF16 for a few hundred steps before enabling FP8. Newer TE recipes (per-block, current-scaling) reduce this fragility by scaling at finer granularity, but delayed scaling remains the workhorse.

**Which operations must stay out of FP8.** The rule is that FP8 is for the large, well-conditioned matmuls and nothing else. Specifically:

- **In FP8:** the four big linear projections — QKV, the attention output projection, and the MLP up- and down-projections. These are where the FLOPs live and where FP8's 2× tensor-core throughput pays off, and they are the most outlier-tolerant.
- **Must stay in BF16 (or higher):** **LayerNorm / RMSNorm** (they compute a mean and variance whose reduction needs range and precision FP8 cannot give without garbage), **softmax** in attention (the exponentials produce a wide dynamic range and tiny probabilities that E4M3 would quantize to zero), the **embedding lookup** (a gather, not a matmul — no FP8 benefit and the values feed everything downstream), and the **final LM head / logits** (the loss is exquisitely sensitive to logit precision, and in RL the log-prob computed from these logits feeds the importance ratio, so any FP8 noise here corrupts the PPO objective directly). TransformerEngine's fused layers already keep these in BF16; if you are wiring FP8 in by hand, this is the list you must respect or you will get NaNs within a few hundred steps.

**Measured speedup on H100.** On the transformer layers themselves — the GEMM-heavy core — FP8 delivers roughly **1.7–2.0× over BF16** on H100, with the high end reached on large hidden dimensions and long sequences where the matmuls are big enough to saturate the FP8 tensor cores and amortize the per-tensor scaling overhead. The 2.0× is a matmul-level figure; the end-to-end training speedup is lower because attention's softmax, the norms, the embedding, data movement, and the optimizer step are not FP8 and do not speed up. Expect something like 1.3–1.5× end-to-end on a transformer training step that was already BF16, with the exact number set by how much of your wall-clock is in the big GEMMs versus everything else.

### FP8 and the four RLHF models

FP8 does not benefit the four RLHF roles equally, and being selective is how you capture the speedup without paying the stability tax four times over. Rank them by payoff:

- **Policy generation (biggest win).** Generation is many forward passes through the big projections, it is compute-bound, and it dominates RL wall-clock. It has *no backward pass*, so the fragile FP8 gradient path (E5M2, small-gradient underflow) never engages — you only run the stable forward direction (E4M3). This makes generation the safest *and* highest-leverage place to use FP8: you get most of the matmul speedup with the least risk. Keep the sampling logits' final projection in BF16 so the sampled distribution is faithful, but run the body of the transformer in FP8.
- **Reward model scoring (good win, low risk).** Also pure forward, also compute-bound when scoring long completions in batch. The one caution is consistency — score in the same precision regime the reward model was trained in, and if it was trained in BF16, validate that FP8 scoring does not shift the reward distribution before trusting it (the section 6 worked example shows this matters more at small model sizes).
- **Reference model log-probs (use with care).** Forward-only, so mechanically safe, but the reference's log-probs feed the KL penalty as a *difference* against the policy's log-probs. If the policy runs FP8 generation but the reference runs BF16 (or vice versa), the dtype mismatch biases the KL estimate. Either run both in FP8 or both in BF16 for the log-prob computation — match them.
- **Policy / value update (smallest win, highest risk).** This is the only role with a backward pass, so it is the only one that exercises the E5M2 gradient path and the delayed-scaling backward recipe — exactly the fragile parts. The matmul speedup is real, but the gradients feed the optimizer that determines whether the run converges, and FP8 gradient noise in RL can shift the policy distribution in ways the reward is sensitive to. The conservative, widely-used split is **FP8 for generation, BF16 for the update** — capture the speedup on the phase that dominates wall-clock and that has no gradients to destabilize, and keep the gradient-producing update in BF16 where it is bulletproof. Only move the update to FP8 after the generation-FP8 run is stable and you have measured that reward matches.

The pattern, then, mirrors the precision-by-role table from section 9 but with FP8 layered on top: FP8 flows naturally to the inference-heavy roles (generation, scoring) and stays out of the gradient-producing update until you have earned the right to put it there.

## 7. Gradient checkpointing: trade compute for memory

So far we have attacked parameter, gradient, and optimizer memory. The fourth consumer — and in RL with long generations, often the *largest* — is **activations**. During the forward pass, every layer stores its outputs because the backward pass needs them to compute gradients. For a transformer with long sequences and a large batch, stored activations can dwarf the parameter memory.

**Gradient checkpointing** (also called activation recomputation) trades compute for this memory. Instead of storing every layer's activations, you store only the inputs at a few checkpoint boundaries (typically each transformer block's input). During the backward pass, when a layer needs its activations, it *recomputes* them by re-running that block's forward from the stored input. You pay an extra forward pass — roughly 33% more compute, since a transformer block's backward is about 2× its forward, so adding one more forward to the 3× total is about a third more — in exchange for dropping activation memory by 30–40% (sometimes much more for very deep models). The trade-off is shown below: the two paths produce identical gradients, but one path holds a large activation buffer and the other recomputes from a tiny stored input.

![Graph showing the gradient checkpointing trade-off where the forward pass either stores all activations into a large buffer or keeps only inputs and recomputes activations during the backward, both paths merging to identical gradients](/imgs/blogs/fsdp-mixed-precision-rl-training-7.png)

The basic API is `torch.utils.checkpoint.checkpoint`, which wraps a forward function:

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedBlock(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        # use_reentrant=False is the modern, recommended variant; it works
        # correctly with FSDP and does not require inputs to have requires_grad.
        return checkpoint(self.block, x, use_reentrant=False)
```

**Selective checkpointing** is the refinement that matters in practice. Recomputing *everything* is wasteful — cheap operations like layernorms and residual adds cost almost nothing to store but recomputing them still adds latency, while a few operations (the big matmuls, attention) hold most of the memory. Selective checkpointing recomputes only the expensive, memory-heavy operations and lets the cheap ones store normally. PyTorch's `torch.utils.checkpoint` supports a `context_fn` / selective policy for exactly this, and it can recover much of the lost throughput while keeping most of the memory savings.

The compute overhead is more interesting than the usual "33%" headline suggests, and the precise number tells you whether selective checkpointing is worth the complexity. A transformer block's backward pass costs roughly twice its forward pass (you compute gradients with respect to both the inputs and the weights, each about one forward's worth of FLOPs). So a normal step is `forward + backward ≈ 1 + 2 = 3` units of compute. Full checkpointing adds one extra forward during the backward to recompute activations, making it `1 + 1 + 2 = 4` units — a `4/3 ≈ 1.33`, hence the 33% overhead. But that extra forward is only needed for the layers you checkpoint. If selective checkpointing recomputes only the attention and MLP matmuls (which hold, say, 80% of the activation memory) and skips recomputing the cheap norm/residual ops, you might recover 80% of the memory savings for only ~15% extra compute instead of 33%. The optimal point depends on your activation-memory profile, which is why you profile first (section 12) and then choose how aggressively to checkpoint. There is also a sublinear-memory result behind all of this: Chen et al. (2016) showed that with checkpoints placed every `sqrt(L)` layers in an `L`-layer network, activation memory drops to `O(sqrt(L))` instead of `O(L)` at the cost of one extra forward — which is the theoretical backbone of the whole technique.

| Setting | Activation memory | Compute overhead | When |
| --- | --- | --- | --- |
| No checkpointing | 100% (baseline) | 0% | activations fit comfortably |
| Full checkpointing | ~25–40% | ~30–40% | long sequences, deep model, tight VRAM |
| Selective checkpointing | ~40–55% | ~10–20% | want most of the savings, less slowdown |

### Strategies: full, selective, every-N, and CPU offload

The single `checkpoint()` call is the atom; the *strategy* is how you place those atoms across a deep network, and the right placement is the difference between a 33% slowdown that buys you nothing and a 12% slowdown that buys you the whole batch.

**Full checkpointing** wraps every transformer block. It is the maximum-memory-savings, maximum-compute-overhead end of the spectrum and the right default when you are deeply OOM and just need the job to run — turn it on everywhere, confirm the job fits, then dial back if throughput matters. The activation memory drops to roughly one block's worth of stored input rather than every block's full activations.

**`checkpoint_sequential` for transformer blocks.** When your blocks live in an `nn.Sequential` (or you can treat them as a list), `torch.utils.checkpoint.checkpoint_sequential` is the convenience that checkpoints a *chain* of modules by splitting it into `segments` and checkpointing each segment as a unit. You pass the sequence and a segment count, and it stores activations only at the segment boundaries, recomputing within a segment during backward:

```python
from torch.utils.checkpoint import checkpoint_sequential

# 32 decoder blocks split into 4 checkpoint segments => store activations
# at 4 boundaries, recompute the 8 blocks inside each segment on backward.
blocks = torch.nn.Sequential(*decoder_layers)   # 32 LlamaDecoderLayers
hidden = checkpoint_sequential(blocks, segments=4, input=hidden_states,
                               use_reentrant=False)
```

The `segments` count is the knob: `segments=1` checkpoints the whole chain as one unit (store only the very first input, recompute all 32 blocks on backward — maximum memory savings, maximum recompute), while `segments=32` checkpoints every block individually (the full-checkpointing case). The sublinear-memory sweet spot from Chen et al. is `segments ≈ sqrt(num_blocks)` — for 32 blocks, `sqrt(32) ≈ 5.7`, so 5–6 segments gives you `O(sqrt(L))` activation memory for one extra forward, which is the theoretically optimal compute/memory trade for uniform blocks.

**The `use_reentrant` flag — always `False`.** PyTorch's checkpoint has two implementations. The legacy *reentrant* one (`use_reentrant=True`, still the default in older code for backward compatibility) re-enters autograd during recomputation, which carries real restrictions: it requires at least one input to have `requires_grad=True` (so checkpointing the first layer, whose input is the non-differentiable token embedding, silently does nothing), it does not support keyword arguments to the checkpointed function cleanly, and it breaks with certain autograd features and with FSDP's parameter management. The *non-reentrant* implementation (`use_reentrant=False`) was built to fix all of this: it uses saved-tensor hooks instead of re-entering autograd, works regardless of input `requires_grad`, handles kwargs, and — critically — composes correctly with FSDP. There is essentially no reason to use `use_reentrant=True` in new code; PyTorch is migrating the default to `False` precisely because the reentrant variant's footguns caused so many silent failures. Set it explicitly to `False` everywhere so you are never at the mercy of which default your PyTorch version ships.

**Combining with FSDP: `checkpoint_wrapper` on FSDP modules.** The composition rule from section 8 — checkpoint inside, FSDP outside — is implemented with `checkpoint_wrapper` and `apply_activation_checkpointing`, which wrap each block in a non-reentrant checkpoint *before* FSDP wraps the model. The reason the wrapper exists (rather than calling `checkpoint()` in your forward) is that FSDP needs to recognize the checkpointed module as a unit so it can share a single all-gather between the recomputation and the gradient computation; the wrapper exposes the block in a form FSDP's machinery understands. The full code for this is in section 8 — the point here is that gradient checkpointing under FSDP is not a separate `checkpoint()` call in your forward but a structural wrapping of the blocks applied at model-construction time.

**The memory-versus-compute tradeoff, quantified: checkpoint 1 in N layers.** The cleanest dial between "full" and "none" is to checkpoint every Nth block and leave the rest storing normally. The trade is close to linear in the fraction checkpointed: if you checkpoint a fraction `f` of your blocks, you save roughly `f` of the recomputable activation memory and pay roughly `f × 33%` extra compute. The table makes the shape concrete for a 32-block model:

| Checkpoint policy | Blocks recomputed | Activation memory | Compute overhead |
| --- | --- | --- | --- |
| None | 0 / 32 | 100% (baseline) | 0% |
| 1 in 4 (every 4th block) | 8 / 32 | ~80% | ~8% |
| 1 in 2 (every other block) | 16 / 32 | ~58% | ~17% |
| Full (every block) | 32 / 32 | ~30% | ~33% |

This "1 in N" dial is what you reach for when full checkpointing fits but costs more throughput than you want and no-checkpointing OOMs — you find the smallest `f` that fits and pay only that fraction of the recompute. In practice, combine it with *selective* checkpointing (recompute only the attention/MLP matmuls within the blocks you do checkpoint) to push the memory-per-compute ratio even further.

**Activation offload to CPU RAM (`act_ckpt_offload`).** A complementary technique to recomputation is to *offload* stored activations to CPU RAM instead of recomputing them. Where checkpointing trades compute for memory, offload trades PCIe bandwidth for memory: the activations are copied to pinned host memory during the forward and copied back to the GPU just before they are needed in the backward. Frameworks expose this as an "activation checkpoint offload" or `act_ckpt_offload` flag (in PyTorch, via the checkpoint wrapper's CPU-offload context; in libraries like NeMo and DeepSpeed as a config switch). It is most useful when you are *compute-bound already* — adding recompute would slow you down, but you have spare PCIe bandwidth and idle CPU RAM, so shuttling activations off-GPU buys memory for "free" in compute terms. The catch is that PCIe (even Gen5 at ~64 GB/s) is far slower than HBM (~3 TB/s on H100), so if the offload/reload cannot be hidden under compute it stalls the pipeline worse than recomputation would. The decision rule: prefer recomputation when you are memory-bound and have compute headroom; prefer offload when you are compute-bound and have PCIe and host-RAM headroom; and on long-generation RL, where the backward is already expensive, recomputation is usually the safer default, with offload reserved for the few largest activation tensors that recomputation cannot shrink enough.

#### Worked example: activations dominate in a long-generation PPO run

A 7B policy generating 2,048-token completions at batch 8 stores activations across 32 layers. Without checkpointing, the activation buffer measured ~38 GB — larger than the 14 GB of BF16 weights and comparable to the sharded optimizer state. Turning on full activation checkpointing dropped it to ~11 GB, a 71% reduction in *activation* memory, at the cost of one extra forward per block (about 34% more step time). That single switch was the difference between batch 8 OOMing and batch 16 fitting comfortably, which roughly halved the wall-clock time to collect a fixed number of episodes despite the per-step slowdown — because the larger batch amortized the fixed generation overhead. In long-generation RL, checkpoint first and ask questions later; activations are usually the bottleneck you did not budget for.

## 8. FSDP and gradient checkpointing: ordering matters

You cannot just wrap a model in FSDP and also call `torch.utils.checkpoint` naively and hope they compose — the ordering of the wrappers is load-bearing, and getting it wrong either breaks the backward or quietly disables the memory savings.

The rule: apply **activation checkpointing to the transformer blocks first, then let FSDP wrap those already-checkpointed blocks**. FSDP's all-gather/free logic needs to be the outer layer so that it gathers a block's parameters, the checkpointed forward runs (storing only the input), the parameters are freed, and on the backward FSDP re-gathers the parameters *and* the checkpoint recomputes the activations together. If you wrap in the wrong order, FSDP can free parameters that the recomputation still needs, and you get a cryptic error about parameters being unavailable during backward.

PyTorch provides `apply_activation_checkpointing` with a `checkpoint_wrapper` to do this cleanly:

```python
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    apply_activation_checkpointing,
    CheckpointImpl,
)
import functools

non_reentrant_wrapper = functools.partial(
    checkpoint_wrapper,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,  # FSDP-safe
)

def is_transformer_block(module):
    return isinstance(module, LlamaDecoderLayer)

# Apply checkpointing to each decoder block BEFORE the FSDP wrap above.
apply_activation_checkpointing(
    base_model,
    checkpoint_wrapper_fn=non_reentrant_wrapper,
    check_fn=is_transformer_block,
)

# Now wrap with FSDP — FSDP sees blocks that are already checkpoint-wrapped.
policy = FSDP(
    base_model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    auto_wrap_policy=auto_wrap,
    mixed_precision=bf16_policy,
    device_id=torch.cuda.current_device(),
    use_orig_params=True,
)
```

Note `CheckpointImpl.NO_REENTRANT`. The older reentrant checkpoint implementation has restrictions that interact badly with FSDP (it requires at least one input to have `requires_grad`, and it disables some autograd features), so the non-reentrant variant is the one to use under FSDP. This is the same `use_reentrant=False` flag from the bare `torch.utils.checkpoint` call earlier, surfaced through the wrapper API. Getting this one enum right saves an afternoon of debugging.

There is one more interaction that surprises people: when FSDP and activation checkpointing are both active, the backward pass triggers *two* all-gathers for a given block, not one. The first all-gather happens when the checkpoint recomputes the block's forward (it needs the full parameters to recompute activations), and a second would naively be needed to compute the actual gradients. FSDP and the checkpoint machinery coordinate so the recomputation and the gradient computation share a single gathered copy of the parameters rather than gathering twice — but only if the wrapping order is correct (checkpoint inside, FSDP outside). If you invert the order, FSDP cannot see that the recomputation needs the parameters, frees them too early, and the recompute fails. This is the mechanical reason behind the "ordering matters" rule, and it is also why you should verify the ordering by checking your profiler trace: you should see exactly one all-gather per block per backward, not two. If you see two, the sharing broke and you are paying double communication.

## 9. Mixed precision across the four RLHF roles

The four models in an RLHF loop do not all need the same precision, and being deliberate about each one saves both memory and headaches. Walk through the roles.

The **policy generation** is pure inference: you sample tokens autoregressively. It is compute-bound, runs many forward passes, and tolerates low precision well — this is the ideal place for BF16, or FP8 on an H100. No gradients flow here, so there is nothing to destabilize; the only correctness concern is that the sampled tokens are reasonable, and BF16 generation is indistinguishable from FP32 generation in every blind eval I have run.

The **reward model scoring** is also inference, also BF16. The one thing to watch is *consistency*: if you trained the reward model in BF16 but score in FP32 (or vice versa), the tiny numerical differences can shift the reward by enough to matter when you are chasing a KL-constrained optimum. Keep the reward model in the same precision it was trained in, and keep it consistent across the run.

The **reference model** computes log-probabilities for the KL penalty, BF16 again. Here consistency with the policy matters: the KL term is a *difference* of log-probs between policy and reference, and if the two models compute log-probs in different precisions, the KL estimate gets a systematic bias. Run policy and reference in the same `param_dtype`.

The **PPO/value update** is the only place gradients flow, and it is where stability is non-negotiable: compute in BF16, but keep the optimizer's master weights in FP32 (the pattern from section 5). The value function update has the same requirement — BF16 forward/backward, FP32 Adam states. The advantage normalization and the clipped surrogate objective both involve subtractions and divisions that can amplify noise, so doing the *reduction* (the loss sum, the advantage mean/std) in FP32 even while the matmuls are BF16 is a cheap insurance policy worth taking.

| RLHF role | Gradients? | Precision | Watch out for |
| --- | --- | --- | --- |
| Policy generation | no | BF16 / FP8 | KV cache memory during long gen |
| Reward scoring | no | BF16 (match training) | precision-mismatch reward drift |
| Reference log-probs | no | BF16 (match policy) | KL bias from dtype mismatch |
| Policy/value update | yes | BF16 compute, FP32 master | small Adam deltas rounding away |

## 10. Numerical stability: NaNs, loss scaling, and gradient clipping

NaNs in distributed mixed-precision training are maddening because they propagate: one rank produces a NaN, it gets all-reduced/reduce-scattered into every other rank's gradients, and your whole job dies at once with no obvious origin. Knowing the failure modes makes them tractable.

### A deep dive on the numbers: overflow, underflow, and why FP16 spikes

Every stability rule in this post traces back to two boundaries of each float format: the largest value it can represent before it saturates to `inf` (overflow), and the smallest nonzero value before it rounds to `0` (underflow). Putting hard numbers on these boundaries makes the whole "BF16 over FP16" argument quantitative rather than folklore.

**The overflow ceilings.** FP32's largest finite value is about `3.4 × 10^38`. **BF16 shares that exact ceiling — `3.4 × 10^38`** — because it has the same 8 exponent bits as FP32; it gave up mantissa bits, not range. **FP16's ceiling is only `6.5 × 10^4`** (precisely 65,504), because its 5 exponent bits cap the exponent far lower. That is the entire difference in one comparison: BF16 can represent a number thirty-three orders of magnitude larger than FP16's largest. In training, any intermediate value — a gradient, an attention logit, a reward-weighted advantage, a squared term in a variance — that exceeds 65,504 becomes `inf` in FP16 and then `NaN` the moment it meets a subtraction or a `0 × inf`. In BF16 the same value sails through with room to spare. FP8 makes this even starker: E4M3 saturates at **448** and E5M2 at **57,344**, which is exactly why FP8 cannot be used without per-tensor scaling factors to keep values inside that tiny window.

**The underflow floors.** At the small end, FP16's smallest normal positive value is about `6 × 10^-5` (with subnormals reaching ~`6 × 10^-8`), below which values flush to zero. BF16's smallest normal is about `1.2 × 10^-38` — matching FP32's range at the bottom too. This is why FP16 *gradients* are the problem child: deep-network gradients, especially in early layers, routinely sit at `10^-6` to `10^-8`, squarely in FP16's underflow zone, so they silently become zero and those parameters stop learning. BF16's floor is far below anything a gradient reaches, so BF16 gradients never underflow.

**Why loss spikes happen in FP16 training.** A loss spike is the visible symptom of a transient overflow. In FP16, the chain is: some input distribution shifts (a hard batch, a high-variance RL advantage, a learning-rate step after warmup), one activation or gradient briefly exceeds 65,504, it becomes `inf`, the `inf` propagates through the backward into the gradient, the optimizer either steps on garbage (loss jumps) or the run produces `NaN` and dies. The spike is not a learning-dynamics problem you can fix with a smaller learning rate alone — it is a *representation* problem: the value you needed to represent did not fit in the format. RL makes this worse than supervised training because the advantage and reward magnitudes are not bounded the way a cross-entropy loss is; an unnormalized reward model can emit a score of 300, multiply through the policy gradient, and produce a gradient that overflows FP16 in a single unlucky batch. This is the mechanism behind the "FP16 PPO blew up to NaN at update 340" result in the section 5 worked example.

**The GradScaler mechanism, in full.** `torch.cuda.amp.GradScaler` (or `torch.amp.GradScaler`) is the FP16-specific machine that fights underflow by exploiting the headroom at the top of the range. Its job: multiply the loss by a large **`scale_factor`** (the running scale, initialized at `init_scale=65536` by default) before `backward()`, so that every gradient is multiplied by the same factor and small gradients that would have underflowed are lifted up into FP16's representable range. Before the optimizer step, it divides (unscales) the gradients by the same factor, restoring their true magnitudes. The trick is that the *scaled* gradients must themselves not overflow, so the scaler runs an adaptive control loop with three parameters:

- **`growth_factor`** (default `2.0`) — the multiplier applied to the scale when things are going well.
- **`backoff_factor`** (default `0.5`) — the multiplier applied to the scale when an overflow is detected.
- **`growth_interval`** (default `2000`) — the number of *consecutive* successful (non-overflowing) steps required before the scale is grown.

The loop works like this: after each backward, the scaler inspects the gradients for `inf`/`NaN`. If it finds any (the scaled gradients overflowed), it **skips the optimizer step entirely for that batch** and multiplies the scale by `backoff_factor` (halving it), so the next attempt uses a gentler scale less likely to overflow — and it resets the success counter to zero. If the gradients are clean, it lets the optimizer step proceed and increments the success counter; once `growth_interval` clean steps accumulate in a row, it multiplies the scale by `growth_factor` (doubling it) to push small gradients even higher up the range, and resets the counter. The net effect is a sawtooth scale that hunts for the largest factor that keeps gradients in range: it grows greedily during stable stretches and backs off instantly when a batch overflows. The skipped-step-on-overflow behavior is why you must call `scaler.step(optimizer)` rather than `optimizer.step()` directly — `scaler.step` is what checks for the `inf` and decides whether to actually apply the update.

**Why BF16 almost never needs a GradScaler.** The entire GradScaler apparatus exists to work around FP16's underflow floor — it lifts small gradients above `6 × 10^-8` so they survive. BF16's underflow floor is `1.2 × 10^-38`, more than thirty orders of magnitude lower, so BF16 gradients simply never underflow and there is nothing to lift. And because BF16 shares FP32's `3.4 × 10^38` ceiling, there is no overflow to back off from either. Both ends of the GradScaler's job are vacuous in BF16, which is why you run `torch.autocast(dtype=torch.bfloat16)` (or FSDP's `MixedPrecision`) with **no scaler at all** — and why the entire skip-step / backoff / growth control loop, with its occasional thrown-away batches and its scale-tuning surprises, simply does not exist in a BF16 run. Deleting that moving part is a real reliability win, not just a convenience. The rare exception where BF16 *can* still produce a non-finite value is not underflow but genuine algorithmic divergence (a true `inf` from, say, an unclipped importance ratio of `exp(1000)`), which no scaler would fix anyway — that is a bug to catch with the finite-check guard below, not a scaling problem.

**Log-softmax and log-prob stability in RL.** RL has one numerical-stability concern that supervised training largely escapes: the log-probabilities at the heart of the policy gradient can go *very* negative, and the way you compute them decides whether they stay finite. A naive `log(softmax(logits))` first exponentiates the logits, sums them, divides, and then takes a log — and the intermediate `exp(logit)` overflows for any logit above ~88 in FP32 (much sooner in FP16), while the probability of a rare-but-sampled token can underflow to `0`, making its `log` become `-inf`. The fix, which every framework's `log_softmax` uses internally, is the **log-sum-exp trick**: subtract the max logit before exponentiating, `log_softmax(x) = (x - x_max) - log(sum(exp(x - x_max)))`, which keeps the largest exponent at `exp(0) = 1` and prevents overflow while computing the log directly so no probability ever has to round to zero before its log is taken. The practical rules that follow:

- **Never compute `log(softmax(x))` as two steps** — always use `F.log_softmax(x, dim=-1)` (or `torch.log_softmax`), which is numerically stabilized. Then gather the chosen token's log-prob from that. In RL this is the log-prob that feeds both the KL penalty and the PPO importance ratio, so a `-inf` here poisons the whole objective.
- **A legitimately very-negative log-prob is fine; a `-inf` is a bug.** A token the policy assigns probability `10^-15` has a log-prob of about `-34.5`, which is perfectly representable in BF16 (its range goes to `±3.4 × 10^38`, far beyond `-34.5`) and is *correct* — the policy really did think that token was nearly impossible and then sampled it anyway. The danger is only when the stabilization is skipped and the value collapses to `-inf`, at which point the importance ratio `exp(new_logprob - old_logprob)` either overflows or produces `NaN`.
- **Keep the final logits and the log-softmax in BF16 or FP32, never FP8.** As noted in section 6, the LM head and the log-prob computation are the most precision-sensitive operations in the loop, because the *difference* between two log-probs (the KL term, the ratio exponent) amplifies any quantization noise in the logits. This is the same reason the worked-example precision-mismatch bug (rollout in one precision, update in another) surfaces here as a blown-up ratio: log-probs computed at different precisions disagree, and the exponential of that disagreement overflows.

**Loss scaling** is the FP16-specific fix and it is worth understanding even though BF16 sidesteps it. In FP16, small gradients underflow to zero before the optimizer ever sees them. `torch.cuda.amp.GradScaler` multiplies the loss by a large factor (default starts at 65,536) before backward — pushing gradients up into FP16's representable range — then unscales them before the optimizer step, and dynamically halves the scale whenever it detects an `inf`/`NaN` (a sign of overflow) and slowly grows it back when steps are clean.

```python
scaler = torch.cuda.amp.GradScaler()  # FP16 only; unnecessary for BF16

for batch in loader:
    optimizer.zero_grad()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = ppo_loss(batch)
    scaler.scale(loss).backward()           # scale up before backward
    scaler.unscale_(optimizer)              # unscale before clipping
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
    scaler.step(optimizer)                  # skips step if inf/NaN found
    scaler.update()                         # adjust scale for next step
```

**Why BF16 needs no scaling**: its 8 exponent bits give it FP32's range, so gradients that would underflow in FP16 are representable in BF16. You simply use `torch.autocast(dtype=torch.bfloat16)` (or FSDP's `MixedPrecision`) with no scaler at all. This is one more reason BF16 is the default — it deletes an entire moving part.

**Gradient clipping** interacts with scaling in an order-sensitive way: you must clip *unscaled* gradients. With `GradScaler` that means calling `scaler.unscale_(optimizer)` *before* `clip_grad_norm_`, as in the code above — clipping scaled gradients would clip to the wrong threshold. Under FSDP, gradient clipping needs the *global* norm across all shards, so use `model.clip_grad_norm_(max_norm)` (FSDP's own method, which does the cross-rank all-reduce of the norm) rather than the plain `torch.nn.utils.clip_grad_norm_`, which would only see the local shard's gradients and clip to a wrong, per-rank norm.

```python
# FSDP-aware global gradient clipping (no GradScaler needed in BF16):
loss.backward()
policy.clip_grad_norm_(max_norm=1.0)  # all-reduces the norm across shards
optimizer.step()
optimizer.zero_grad()
```

**Diagnosing NaN/Inf in distributed training**: add a cheap guard that checks the loss and gradient norm each step and, on the first non-finite value, dumps which rank and which parameter went bad before the NaN spreads. In RL the usual culprits are an exploding advantage (normalize advantages and clip the reward), a log-prob ratio blowing up (the PPO clip is supposed to bound this — verify your clip range is actually applied), or a learning rate spike after a warmup bug. `torch.autograd.set_detect_anomaly(True)` will pinpoint the offending operation, though it is slow, so enable it only when reproducing a crash.

A practical pattern I use on every distributed run is a finite-check guard that runs *before* the optimizer step and votes across ranks, because a NaN that originates on one rank will poison every other rank through the reduce-scatter and you want to catch it at the source:

```python
def assert_all_finite(loss, model, rank):
    bad = not torch.isfinite(loss).all()
    for name, p in model.named_parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            print(f"[rank {rank}] non-finite grad in {name}")
            bad = True
    # All-reduce a 0/1 flag so every rank aborts together, not just the bad one.
    flag = torch.tensor([1.0 if bad else 0.0], device="cuda")
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    if flag.item() > 0:
        raise RuntimeError(f"non-finite detected; rank {rank} bad={bad}")
```

The RL-specific NaN sources deserve their own checklist because they differ from supervised training. First, **reward scaling**: an unnormalized reward model can output scores in the hundreds, and multiplying that through the advantage and the policy gradient produces gradients far larger than BF16's stable regime — normalize rewards to roughly unit variance with a running mean/std before computing advantages. Second, **the KL term going negative or exploding**: the KL estimate `log(π/π_ref)` can spike if the policy drifts far from the reference in a single step, and a large KL coefficient then produces a huge negative reward that destabilizes the next update; clip the per-token KL or use the adaptive KL controller that TRL ships. Third, **the importance ratio**: PPO's `ratio = exp(new_logprob - old_logprob)` is an exponential, and if new and old log-probs disagree by even 20 (because of the precision-mismatch bug from section 1), `exp(20) ≈ 5e8` and you get an instant overflow. The PPO clip bounds the *objective* but not the ratio itself before clipping, so a precision bug here surfaces as a NaN, not a quiet quality regression. This is the concrete cost of not keeping rollout and update in the same precision — it is a crash, and a confusing one.

## 11. A full FSDP + BF16 + checkpointing RLHF policy training loop

Here is the assembled training loop, putting every piece together: FSDP FULL_SHARD with a transformer auto-wrap policy, a BF16 `MixedPrecision` policy with FP32 gradient reduction, non-reentrant activation checkpointing applied before the wrap, FSDP-aware gradient clipping, and an FP32-master Adam optimizer. This is close to what I actually run for a 7B PPO policy.

```python
import torch, functools
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy, MixedPrecision, BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper, apply_activation_checkpointing, CheckpointImpl,
)
from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

def build_policy(model_name, rank):
    base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    # 1) Activation checkpointing on each decoder block, FSDP-safe variant.
    apply_activation_checkpointing(
        base,
        checkpoint_wrapper_fn=functools.partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
        check_fn=lambda m: isinstance(m, LlamaDecoderLayer),
    )

    # 2) BF16 compute, FP32 gradient reduction for large-cluster stability.
    mp = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.bfloat16,
    )

    # 3) FSDP wrap, sharding params/grads/optimizer across ranks.
    return FSDP(
        base,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={LlamaDecoderLayer}),
        mixed_precision=mp,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # overlap comm/compute
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
        limit_all_gathers=True,  # cap concurrent all-gathers, avoids OOM spikes
    )

def train():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())

    policy = build_policy("meta-llama/Llama-2-7b-hf", rank)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-6)  # FP32 states

    for step, batch in enumerate(rollout_loader):
        optimizer.zero_grad(set_to_none=True)
        # advantages/returns precomputed during rollout collection
        logprobs, values = policy_forward(policy, batch)
        loss = ppo_clipped_loss(logprobs, batch.old_logprobs,
                                batch.advantages, batch.returns, values,
                                clip=0.2, vf_coef=0.5, kl_coef=0.1)
        loss.backward()                          # BF16 compute, no scaler needed
        gnorm = policy.clip_grad_norm_(max_norm=1.0)  # global cross-shard norm
        optimizer.step()                         # updates FP32 master weights

        if step % 10 == 0 and rank == 0:
            print(f"step {step} loss {loss.item():.4f} grad_norm {gnorm:.3f} "
                  f"peak_mem {torch.cuda.max_memory_allocated()/1e9:.1f}GB")

if __name__ == "__main__":
    train()
```

Two flags deserve a callout. `backward_prefetch=BackwardPrefetch.BACKWARD_PRE` tells FSDP to start all-gathering the *next* set of parameters for the backward before it finishes the current one, overlapping communication with compute — this is where most of the "FSDP is only 10% slower than DDP" claim comes from, and turning it off can cost 20%+. `limit_all_gathers=True` caps how many all-gather collectives run concurrently; without it FSDP can prefetch aggressively enough to spike memory and OOM, so it is a cheap safety belt.

The clipped surrogate loss itself — the PPO objective that bounds the policy update — is derived from first principles in the policy-gradient track; if you want the *why* behind the `clip=0.2` ratio bound and the KL coefficient, see `/blog/machine-learning/reinforcement-learning/proximal-policy-optimization-explained`. Here the focus is purely on running it at scale.

#### Worked example: FSDP2 + BF16 + checkpointing for a Llama-3 8B RLHF policy

Let me walk the whole stack end to end on a concrete model — a Llama-3 8B policy in a PPO loop on 8×A100-80GB — using the FSDP2 API from section 2b, and actually measure the memory at each step so the savings are not hand-waved. The model has 32 decoder blocks, hidden size 4096, ~8.0 billion parameters.

**(a) Wrap with `fully_shard()`.** FSDP2 shards bottom-up: each decoder block first, then the root. Sharding the blocks individually is what gives per-block all-gather granularity (the FSDP1 `auto_wrap_policy` job, now expressed directly in the loop).

```python
import torch, functools
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper, apply_activation_checkpointing, CheckpointImpl)
from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank % torch.cuda.device_count())
mesh = init_device_mesh("cuda", (dist.get_world_size(),))

base = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B", torch_dtype=torch.bfloat16)

def mem(tag):  # GB currently allocated on this rank
    torch.cuda.synchronize()
    if rank == 0:
        print(f"{tag:32s} {torch.cuda.memory_allocated()/1e9:6.1f} GB")

mem("after load (pre-shard)")     # ~16.0 GB: full BF16 model on every rank
```

**(c) Apply `checkpoint_wrapper` to every transformer block — *before* sharding.** The section 8 ordering rule (checkpoint inside, FSDP outside) holds for FSDP2 too: wrap each block in a non-reentrant checkpoint first, so `fully_shard` then shards already-checkpointed blocks and can share one all-gather between recomputation and gradient computation.

```python
apply_activation_checkpointing(
    base,
    checkpoint_wrapper_fn=functools.partial(
        checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
    check_fn=lambda m: isinstance(m, LlamaDecoderLayer),
)
```

**(b) Configure `MixedPrecisionPolicy(param_dtype=torch.bfloat16)` and shard.** BF16 compute, FP32 gradient reduction for cross-rank stability. Then the bottom-up `fully_shard` calls.

```python
mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16,
                          reduce_dtype=torch.float32)
for layer in base.model.layers:               # 32 blocks, innermost first
    fully_shard(layer, mesh=mesh, mp_policy=mp)
fully_shard(base, mesh=mesh, mp_policy=mp)      # then the root
policy = base
mem("after fully_shard")          # ~2.0 GB: only this rank's 1/8 param shard
optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-6)  # FP32 states
```

**(d) Measure memory before and after each step.** Now run one rollout-plus-update step and snapshot the peak, resetting the peak counter so each phase is isolated:

```python
torch.cuda.reset_peak_memory_stats()
logprobs, values = policy_forward(policy, batch)   # BF16 fwd, checkpointed
loss = ppo_clipped_loss(logprobs, batch.old_logprobs, batch.advantages,
                        batch.returns, values, clip=0.2, vf_coef=0.5, kl_coef=0.1)
loss.backward()                                    # recompute + reduce-scatter
gnorm = policy.clip_grad_norm_(max_norm=1.0)
optimizer.step(); optimizer.zero_grad(set_to_none=True)
if rank == 0:
    print(f"peak this step {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
```

The numbers I measure on this configuration, per rank:

| Quantity | Value per rank |
| --- | --- |
| Full BF16 model, pre-shard (every rank) | ~16.0 GB |
| Sharded params (1/8 of 16 GB BF16) | ~2.0 GB |
| Sharded gradients (FP32 reduce, 1/8) | ~4.0 GB |
| Sharded Adam states (FP32 m+v, 1/8) | ~8.0 GB |
| **Static total (params+grads+optimizer)** | **~14.0 GB** |
| Activations *without* checkpointing (batch 8, 2048 tok) | ~40 GB peak |
| Activations *with* full checkpointing | ~12 GB peak |
| **Peak per step (static + checkpointed activations + KV)** | **~30 GB** |

The two measurements that tell the story are `after load (pre-shard)` at 16 GB versus `after fully_shard` at 2 GB for parameters — the `16/N` sharding doing exactly what the section 1 formula predicts — and the activation line dropping from a ~40 GB peak to ~12 GB once checkpointing is on. Static 14 GB plus a ~12 GB checkpointed-activation band plus a ~4 GB generation KV cache lands the peak around 30 GB, leaving 50 GB of headroom on an 80 GB card. Without sharding and without checkpointing, the same step would need 16 GB (full model) + 64 GB (FP32 grads+optimizer) + 40 GB (activations) ≈ 120 GB — impossible on one card. With the stack, it not only fits, it fits with room to double the batch.

**(e) Compute MFU.** Model FLOPs Utilization tells you how much of the hardware's theoretical compute you are actually using, which is the honest measure of whether the stack is efficient or whether communication is stalling you. The standard estimate for a transformer training step is `6 × N × D` FLOPs, where `N` is the parameter count and `D` is the number of tokens processed (the `6` is 2 for the forward matmul-multiply-add and 4 for the backward, which is ~2× the forward). Gradient checkpointing adds one extra forward, so the multiplier becomes `8` (`2` forward + `2` recompute-forward + `4` backward) for the checkpointed blocks. Take batch 8, sequence 2048, so `D = 8 × 2048 = 16,384` tokens per step, `N = 8 × 10^9`:

```
FLOPs/step ≈ 8 × N × D = 8 × 8e9 × 16,384 ≈ 1.05e15 FLOPs (checkpointed)
```

Suppose the measured step time is 1.4 seconds across the 8 GPUs. The achieved rate is `1.05e15 / 1.4 ≈ 7.5e14 FLOP/s` for the whole job, or `~9.4e13 FLOP/s` per GPU. An A100 delivers ~312 TFLOP/s (`3.12e14`) of BF16 tensor-core throughput, so MFU per GPU is `9.4e13 / 3.12e14 ≈ 30%`. That 30% is a *healthy* RLHF number — RL is dragged down by the autoregressive generation phase (memory-bound, low utilization) and the FSDP collectives, so 30–40% MFU on an RLHF loop is good where 50%+ would be expected for dense pretraining. If you measured 10%, that would point at an interconnect bottleneck or a missing `auto_wrap`/per-block shard (one giant all-gather), and you would go to the section 13 profiler to find it. The MFU number is the single scalar that tells you whether the whole FSDP2 + BF16 + checkpointing stack is paying off or whether something upstream is wasting the GPUs you are paying for.

## 12. Saving and loading sharded checkpoints

There is a practical problem that bites everyone the first time and that the tutorials often skip: once your model is sharded across eight ranks, *no single rank holds a complete model*, so you cannot just call `torch.save(model.state_dict())` and expect a loadable checkpoint. Each rank's `state_dict` holds only its `1/N` shard. For RLHF this matters acutely because you constantly need a complete checkpoint — to spin up a fresh reference model, to run evaluation, to hand the aligned model to inference serving.

FSDP gives you two state-dict types, and choosing correctly avoids both OOMs and corrupt checkpoints. **`FULL_STATE_DICT`** all-gathers the entire model onto rank 0 (optionally offloading to CPU) and saves one consolidated file — convenient, compatible with `from_pretrained`, but it briefly materializes the full model on one rank, which can OOM for very large models. **`SHARDED_STATE_DICT`** saves each rank's shard separately into a distributed checkpoint that PyTorch's `torch.distributed.checkpoint` can reload across any number of ranks — no single-rank materialization, so it scales to any model size, at the cost of needing a resharding step to produce a single `from_pretrained`-compatible file.

```python
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# Consolidate to a single file on rank 0, offloading to CPU to avoid GPU OOM.
save_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
with FSDP.state_dict_type(policy, StateDictType.FULL_STATE_DICT, save_cfg):
    cpu_state = policy.state_dict()      # full model gathered onto rank 0 CPU
    if dist.get_rank() == 0:
        torch.save(cpu_state, "policy_full.pt")
```

The rule of thumb: use `FULL_STATE_DICT` with `offload_to_cpu=True, rank0_only=True` for models up to roughly 30B (rank 0's CPU RAM holds the consolidated FP32 copy), and switch to `SHARDED_STATE_DICT` with distributed checkpointing above that. For RLHF specifically, save the consolidated policy at every evaluation interval so you always have a servable artifact, and save sharded checkpoints frequently for cheap, resumable mid-run snapshots. Mixing them up — trying to save a 70B model with `rank0_only` full state — is a classic mid-training OOM that wastes hours of rollout collection.

## 13. Performance profiling: where did the time and memory go?

You cannot optimize what you cannot see, and FSDP's collectives are invisible until you profile. Two tools answer the two questions you will always ask: *where is my memory* and *where is my time*.

For memory, `torch.cuda.memory_summary()` prints a detailed table of allocated, reserved, and peak memory by category, and `torch.cuda.memory._dump_snapshot("snap.pickle")` captures a full allocation timeline you can visualize at `pytorch.org/memory_viz` to see exactly which tensors live when. This is how you confirm that activations, not parameters, are your bottleneck — and whether checkpointing actually shrank them.

```python
torch.cuda.memory._record_memory_history(max_entries=100_000)
# ... run a few training steps ...
torch.cuda.memory._dump_snapshot("rlhf_mem_snapshot.pickle")
print(torch.cuda.memory_summary(abbreviated=True))
torch.cuda.memory._record_memory_history(enabled=None)  # stop recording
```

For time, the PyTorch Profiler with FSDP shows the all-gather and reduce-scatter collectives as their own NCCL kernels on the timeline, so you can see whether they overlap with compute (good) or stall it (bad — usually means `backward_prefetch` is off or your interconnect is the bottleneck).

```python
from torch.profiler import profile, ProfilerActivity, schedule

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./tb_logs"),
    record_shapes=True, with_stack=True,
) as prof:
    for step in range(5):
        train_one_step(policy, optimizer, next(rollout_loader))
        prof.step()
```

Open the trace in TensorBoard or `chrome://tracing` and look for the NCCL `ncclAllGather`/`ncclReduceScatter` rows. If they sit on a separate stream beneath the compute kernels with little idle time, FSDP is overlapping well. If compute kernels stall waiting on a long all-gather, you are interconnect-bound — switch to `SHARD_GRAD_OP` if memory allows, or `HYBRID_SHARD` on multi-node. The single most common profiling finding I see is that someone wrapped the whole model as one FSDP unit (forgot the `auto_wrap_policy`), so there is one giant all-gather at the start of every forward that cannot overlap with anything. The fix is the transformer auto-wrap from section 3.

#### Worked example: reading a memory snapshot to find the real bottleneck

On a 7B PPO run that kept OOMing at batch 12, I dumped a snapshot and opened it at the memory visualizer. The static FSDP shard was a flat 14 GB that never moved — params, gradients, optimizer all sharded, exactly as predicted by the `16/N` formula. The killer was a sawtooth: a band of allocations that climbed to 41 GB during the forward and collapsed at the optimizer step. That sawtooth is the activation buffer, and 41 GB on top of 14 GB static was blowing past 80 GB once the generation KV cache (another ~12 GB) was added. The snapshot made the diagnosis unambiguous — it was *not* a parameter problem that more sharding would fix, it was an activation problem. I turned on activation checkpointing, the sawtooth dropped to a 12 GB band, and batch 16 fit with headroom. Without the snapshot I might have wasted a day trying `HYBRID_SHARD` or smaller shards, attacking the wrong 14 GB. The discipline is: read the snapshot, identify whether your peak is static (parameters/optimizer → shard more) or transient (activations/KV cache → checkpoint or shrink batch/sequence), and only then choose a fix. Memory bugs are almost always misdiagnosed because the OOM message tells you the *last* allocation that failed, not the one actually responsible.

One more profiling habit worth building: track `torch.cuda.max_memory_allocated()` and reset it with `torch.cuda.reset_peak_memory_stats()` around each phase (rollout generation, reward scoring, the update). RL has three distinct memory regimes in one step, and the peak you care about — the one that causes OOM — is the maximum across all three, which is usually the generation phase (KV cache) or the backward (activations), not the optimizer step. Logging the per-phase peak tells you which phase to optimize and stops you from over-sharding for a peak that lives somewhere else entirely.

## 14. Case studies

**Meta OPT-175B and the move to FSDP.** Meta's large-model training migrated from FairScale's FSDP to PyTorch-native FSDP precisely to get the memory savings of ZeRO-3-style parameter sharding without a separate engine. The headline result across the FSDP papers and Meta's engineering posts is that FULL_SHARD lets you train models whose full state is many times a single GPU's memory, with throughput within roughly 10–15% of DDP on fast interconnects — the communication overhead is real but largely hidden by prefetching.

**TRL and Hugging Face RLHF at scale.** The TRL library's PPO and GRPO trainers integrate FSDP through `accelerate`, and the practical guidance the HF team publishes matches everything above: BF16 by default, FSDP FULL_SHARD for the trainable policy, gradient checkpointing on for long generations, and the reference model wrapped separately rather than parameter-shared. InstructGPT-scale RLHF (Ouyang et al., 2022) established the four-model loop; the systems work since has been about making that loop fit on commodity clusters, which is exactly the FSDP + mixed-precision story.

**NVIDIA FP8 on H100.** NVIDIA's TransformerEngine benchmarks report roughly 2× throughput for FP8 over BF16 on the GEMM-heavy parts of transformer training on H100, with final model quality matching BF16 when the hybrid E4M3/E5M2 recipe and delayed scaling are used. The caveat in their own guidance is the one from section 6: keep normalization and the LM head in higher precision, and expect instability if you push small models or outlier-heavy layers into FP8. The 2× figure is a matmul-level speedup; end-to-end RLHF speedups are smaller because generation, data movement, and the optimizer step are not all FP8.

**A 7B PPO run on 8×A100-80GB (my own).** Combining FULL_SHARD, BF16 with FP32 reduction, and non-reentrant activation checkpointing took a 7B summarization-RLHF job from OOM-on-launch to a stable run at batch 16, sequence 1,024, ~0.9× the throughput a hypothetical (impossible) DDP run would have had per step — but it actually *ran*, which is the only throughput that counts. Final reward and downstream win-rate matched a smaller FP32 reference run within noise, confirming the central claim: the memory cuts did not cut quality.

## 15. Choosing a precision and a strategy

The decision tree below collapses sections 3, 5, and 6 into a single flow you can apply in thirty seconds: pick precision by hardware then stability, pick sharding by what fits, and layer checkpointing whenever activations are tight.

![Decision tree for choosing a training precision starting from whether an H100 is available, branching to FP8, BF16, FP16 with GradScaler, or FP32 baseline](/imgs/blogs/fsdp-mixed-precision-rl-training-8.png)

The whole field's progression toward these defaults is shown below — from AMP in 2018 through native FSDP to FSDP-plus-FP8 in 2024 — and the direction is consistent: each step either shaved memory or roughly doubled throughput.

![Timeline of mixed precision and sharding milestones from AMP in 2018 through FairScale FSDP, native PyTorch FSDP, FSDP2, FP8 TransformerEngine, to FSDP plus FP8 in 2024](/imgs/blogs/fsdp-mixed-precision-rl-training-6.png)

## When to use this (and when not to)

Reach for FSDP + mixed precision when your model genuinely does not fit on one GPU, or fits but leaves no room for the activations and KV cache that RL generation demands. For a 7B+ policy in an RLHF loop with four models, this stack is not optional — it is the only way the job runs at all.

Do **not** reach for FULL_SHARD when a cheaper tier suffices. If your parameters fit on one card and only the optimizer states overflow, `SHARD_GRAD_OP` runs faster because it skips the per-layer all-gather. If everything fits comfortably on one GPU, plain DDP with BF16 autocast is simpler and has the least communication — sharding a model that already fits just adds collective overhead for no memory benefit. And do not reach for FP8 until BF16 is working: the 2× matmul speedup is real, but the debugging cost of a flaky FP8 run on a small or outlier-heavy model will eat the savings. FP8 is an optimization you earn, not a starting configuration.

Do not reach for FP16 at all on hardware that supports BF16. FP16's narrow range forces loss scaling, invites NaNs in the wide-magnitude world of RL advantages, and buys nothing over BF16 in speed. The only reason to use FP16 today is a legacy GPU (V100 and older) that lacks BF16 tensor cores.

Finally, do not turn on aggressive gradient checkpointing if activations already fit — you would pay 30–40% extra compute for memory you did not need. Profile first (`memory_summary()`), confirm activations are the bottleneck, then checkpoint. Measure twice, recompute once.

## Key takeaways

- RL training is a memory problem before it is an algorithm problem: RLHF holds four models, and FSDP's job is to stop replicating them on every GPU.
- FSDP FULL_SHARD equals DeepSpeed ZeRO-3 — both shard parameters, gradients, and optimizer states to 1/N per rank, reconstructing each layer with an all-gather only when needed.
- Choose the least sharding that fits: NO_SHARD (DDP) < SHARD_GRAD_OP (ZeRO-2) < FULL_SHARD (ZeRO-3), because every extra tier costs communication.
- Always set a `transformer_auto_wrap_policy` so all-gather happens per block, not for the whole model at once — this is the single most common FSDP misconfiguration.
- BF16 is the default for large-model training: it keeps FP32's exponent range at 2 bytes, so it needs no loss scaling, unlike FP16, whose narrow range invites NaNs.
- Keep FP32 master weights in the optimizer even while compute runs in BF16; otherwise tiny Adam updates round away to nothing in BF16's 7-bit mantissa.
- Gradient checkpointing trades ~33% compute for a 30–40% activation-memory cut — and in long-generation RL, activations are usually the largest consumer, so checkpoint early.
- Apply activation checkpointing *before* the FSDP wrap, with the non-reentrant variant, or the backward pass breaks when FSDP frees parameters the recomputation still needs.
- Under FSDP, clip gradients with `model.clip_grad_norm_` (it all-reduces the global norm across shards), not the plain per-tensor utility that only sees the local shard.
- FP8 on H100 buys ~2× on the big matmuls with the hybrid E4M3/E5M2 recipe and delayed scaling, but keep norms and the LM head in BF16 and earn it only after BF16 is stable.

## Further reading

- Zhao et al., "PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel" (2023) — the canonical FSDP design and benchmark paper.
- Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (2020) — the DeepSpeed ZeRO stages that FSDP's sharding strategies mirror.
- Micikevicius et al., "Mixed Precision Training" (2018) — the original master-weights and loss-scaling paper that started it all.
- NVIDIA, "FP8 Formats for Deep Learning" (Micikevicius et al., 2022) and the TransformerEngine documentation — the E4M3/E5M2 split and delayed scaling.
- Chen et al., "Training Deep Nets with Sublinear Memory Cost" (2016) — the gradient/activation checkpointing technique.
- Ouyang et al., "Training language models to follow instructions with human feedback" (InstructGPT, 2022) — the four-model RLHF loop these systems exist to fit.
- Within series: the unified map `reinforcement-learning-a-unified-map`, the capstone `the-reinforcement-learning-playbook`, and the PPO derivation at `/blog/machine-learning/reinforcement-learning/proximal-policy-optimization-explained`.
