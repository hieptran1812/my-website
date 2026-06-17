---
title: "FSDP and Sharding Bugs: The Resume That Explodes the Loss"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Localize and fix the sharding-specific bugs FSDP introduces — the checkpoint resume that detonates the loss, the wrapping policy that saves no memory, and the grad-clip that silently does nothing."
tags:
  [
    "debugging",
    "model-training",
    "fsdp",
    "distributed-training",
    "checkpointing",
    "mixed-precision",
    "pytorch",
    "finetuning",
    "deep-learning",
    "llm",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/fsdp-and-sharding-bugs-1.png"
---

You finetuned a 7B model for two days across eight GPUs. The loss came down cleanly from 8.2 to 0.31, the eval numbers looked great, and you saved a checkpoint at step 5,000 feeling good about life. Then your spot instance got reclaimed. You relaunched, pointed the trainer at the checkpoint, and watched the loss come back as **8.0** on the very first logged step — almost exactly where it started two days ago. The model didn't *quite* start from scratch (the weights loaded), but something detonated, and within fifty steps you had `NaN`. Two days of compute, gone, and the run that "resumed" was lying to you about what it actually restored.

This is the single most expensive bug in distributed training, and it is almost never a weight bug. The weights round-tripped fine. What got dropped on the floor was the **sharded optimizer state** — the per-shard Adam moments that every rank holds for its slice of the parameters — and because each GPU under Fully Sharded Data Parallel (FSDP) owns only `1/N` of every tensor, the way you save and load is fundamentally different from the single-GPU case you debugged a hundred times. The sharding that lets you fit the model at all is the same sharding that creates an entire class of bugs that Distributed Data Parallel (DDP) simply does not have.

This post is about those bugs. We will start with the **science**: what FSDP actually shards (the ZeRO stages — optimizer state, then gradients, then parameters), the all-gather-on-forward / reduce-scatter-on-backward communication dance that makes a full-size compute possible from a `1/N` resident shard, and the memory math that explains why FSDP fits a model that DDP cannot. Then we go bug by bug, each with a mechanism, a runnable diagnostic, and before→after evidence: the checkpoint state-dict-type mismatch that explodes the resume, mixed precision under FSDP and the `MixedPrecision` policy, the wrapping policy that accidentally saves nothing, sharding-aware gradient clipping, frozen params and buffers, CPU offload, and meta-device init. The spine of this whole series holds here too: **a bug hides in one of six places — data, optimization, model code, numerics, systems, or evaluation — and you bisect to the right one before touching code.** FSDP bugs live squarely in **systems**, and the master tools still work: make-it-fail-small (run the resume test on 100 steps, not 5,000) and read the instruments (log the loss the instant you resume; if it jumps, the optimizer or RNG wasn't restored). Figure 1 shows the layered view of what FSDP shards versus what DDP copies — the source of both the memory savings and every bug we're about to hunt.

![Layered comparison showing DDP replicating full state per GPU versus FSDP sharding parameters, gradients, and optimizer moments into per-rank slices that all-gather only while in use](/imgs/blogs/fsdp-and-sharding-bugs-1.png)

By the end you will be able to take any FSDP or ZeRO run — PyTorch native FSDP, `accelerate` with FSDP, or DeepSpeed ZeRO — that explodes on resume, refuses to save memory, or silently degrades accuracy, and localize the bug in minutes. The definitive test, which we will build, is brutally simple: **a save → load → resume must produce a loss identical to a run that never stopped.** If it doesn't, you have a sharding bug, and this post tells you which one.

If you want the master decision tree this slots into, start with [the taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs); for the field-wide checklist, the [training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook) is the capstone.

## 1. The science: what FSDP shards and why DDP can't fit your model

Before any debugging, you need a precise account of where every byte lives, because every FSDP bug is a byte that ended up in the wrong place or got computed on the wrong slice. Let's build the memory math from first principles, then layer the sharding on top.

### 1.1 The memory budget of a training step

Train a model with `P` parameters in mixed precision with the Adam optimizer. At steady state, four things consume memory that scale with `P`:

- **Parameters.** If you keep an fp32 master copy plus a bf16/fp16 compute copy, that's `4P + 2P = 6P` bytes. (Many setups keep only the lower-precision params resident and an fp32 master inside the optimizer — accounting varies, but the order is the same.)
- **Gradients.** One gradient per parameter, typically in the compute dtype: `2P` bytes (or `4P` if you accumulate grads in fp32).
- **Optimizer state.** Adam keeps two fp32 moments per parameter — the first moment `m` and second moment `v` — so `4P + 4P = 8P` bytes. This is the big one, and it is the part most people forget when a resume explodes.
- **Activations.** These scale with batch size, sequence length, and depth, *not* directly with `P`, and they are addressed separately by gradient checkpointing. We cover activation memory in depth in [out-of-memory debugging](/blog/machine-learning/debugging-training/out-of-memory-debugging); here we focus on the model-state bytes.

Add the model-state terms (the classic Adam-mixed-precision accounting from the ZeRO paper, Rajbhandari et al. 2020): roughly $K \cdot P$ bytes where $K \approx 16$ — about 16 bytes of *state* per parameter for fp16 + fp32-master + Adam moments. For a 7B model:

$$
\text{state bytes} \approx 16 \times 7\times10^{9} = 1.12 \times 10^{11}\ \text{bytes} \approx 112\ \text{GB}.
$$

That doesn't fit on an 80 GB A100, let alone a 40 GB card — and we haven't counted a single activation yet. **This is why DDP fails on large models.** DDP replicates the *entire* `~16P` of state on *every* GPU; eight GPUs give you eight full copies and zero memory relief. Adding GPUs makes you faster, never bigger.

### 1.2 The ZeRO insight: shard the state, not just the data

ZeRO (Zero Redundancy Optimizer) — and FSDP, which is PyTorch's native implementation of the same idea — makes one observation: that `~16P` of state is *replicated* across `N` ranks for no reason. Each rank only ever applies an optimizer step to all of it, but it could instead own and update just `1/N` of it. ZeRO defines three progressively aggressive stages:

| Stage | What it shards | Per-GPU state (Adam, ~16P total) | DeepSpeed name | FSDP analog |
|---|---|---|---|---|
| ZeRO-1 | Optimizer state only | `~4P + 8P/N` | `stage=1` | (partial) |
| ZeRO-2 | + Gradients | `~2P + 14P/N` (approx) | `stage=2` | (partial) |
| ZeRO-3 | + Parameters | `~16P/N` | `stage=3` | **default FSDP** `FULL_SHARD` |

PyTorch FSDP with `ShardingStrategy.FULL_SHARD` is the equivalent of ZeRO-3: it shards **parameters, gradients, and optimizer state**. `SHARD_GRAD_OP` is the ZeRO-2 analog (shards grads + optimizer state, replicates params). For a 7B model on 8 GPUs under FULL_SHARD, per-GPU state drops to roughly `112 GB / 8 = 14 GB` — now it fits, with room for activations.

The trade-off is **communication**. Under DDP, the only collective is one all-reduce of gradients per step. Under FSDP FULL_SHARD, every shard has to be reassembled into the full parameter for compute and then re-split afterward. That is the dance in the next section, and it is the source of both the latency and several bugs.

It is worth being precise about *why* sharding the optimizer state is the highest-value move, because it explains the order of the ZeRO stages. Of the `~16P` bytes, the optimizer occupies `8P` (the two fp32 Adam moments) and the fp32 master copy another `4P` — together three-quarters of the budget — while the compute-precision parameters and gradients are only `~2P` each. So ZeRO-1 (shard the optimizer state alone) already removes the single biggest term, dropping per-GPU state from `16P` to roughly `4P + 12P/N`. On 8 GPUs that's `4P + 1.5P = 5.5P` versus `16P` — already a `~3×` reduction with the *least* communication added (you only need to gather the updated params after the step, not during forward/backward). ZeRO-2 adds gradient sharding for a bit more savings and the same comm profile. ZeRO-3 / FULL_SHARD is the aggressive end: it shards the parameters too, which forces the all-gather-during-compute dance and the most communication, but takes you all the way to `~16P/N`. The practical guidance that falls out of this: if a model fits under `SHARD_GRAD_OP` (ZeRO-2 analog — params replicated, grads+optimizer sharded), prefer it, because it avoids the parameter all-gather on every layer and is faster. Only step up to `FULL_SHARD` when ZeRO-2 still doesn't fit. Picking the wrong strategy isn't a *bug* exactly, but choosing `FULL_SHARD` for a model that would fit under `SHARD_GRAD_OP` leaves throughput on the table, and choosing `SHARD_GRAD_OP` for a model that needs `FULL_SHARD` will OOM — so it belongs in the same diagnostic frame.

### 1.3 The all-gather / reduce-scatter dance

A rank holds only `P/N` of any given parameter tensor — call it a *flat shard*. You cannot run a matrix multiply on `1/8` of a weight matrix. So FSDP does this, per FSDP **unit** (a wrapped module), on the way through the network:

1. **Forward, before the unit runs:** `all-gather` the parameter shards from all ranks so every rank momentarily holds the **full** parameter. Run the forward. Then **free** the full parameter, keeping only the local shard.
2. **Backward, before the unit's backward runs:** `all-gather` the full parameter again (it was freed). Compute the local gradient with respect to the full parameter.
3. **Backward, after:** `reduce-scatter` the gradient — this both averages the gradient across ranks (like all-reduce) *and* splits it so each rank keeps only its `1/N` slice. The full gradient is freed.
4. **Optimizer step:** each rank updates only its `P/N` shard of parameters using its `P/N` shard of gradients and its `P/N` shard of Adam moments. No communication needed.

Figure 2 traces this for one unit. The crucial consequence for debugging: **peak memory depends on the size of the largest single unit that is gathered at one time**, not on the whole model. Wrap the model as one giant unit and you gather the whole thing — no savings. Wrap each transformer block separately and you gather one block at a time. That is the wrapping-policy bug we hit in section 4.

![Dataflow graph of one FSDP unit all-gathering its full parameter for forward and backward compute then reduce-scattering the gradient back into a per-rank shard](/imgs/blogs/fsdp-and-sharding-bugs-2.png)

The second crucial consequence: **the optimizer state is sharded the same way the parameters are.** Rank 3's Adam moments correspond to rank 3's parameter shard and nothing else. When you save and restore, the moments must go back to the matching shard on the matching rank, or the optimizer applies a step computed from the wrong history. Hold that thought — it is the resume bug.

A third consequence shapes throughput and a class of subtle hangs: **communication can overlap compute, but only if the schedule is right.** While the GPU computes layer `L`'s forward, FSDP can prefetch (all-gather) layer `L+1`'s parameters in the background, so by the time compute reaches `L+1` its full parameter is already resident. The same overlap applies on the backward with the next-needed gather and the just-finished reduce-scatter. PyTorch exposes this through `forward_prefetch` and `backward_prefetch` (`BackwardPrefetch.BACKWARD_PRE` prefetches the next gather before the current backward, the more aggressive and usually faster setting). When the overlap works, comm hides behind compute and an 8-GPU FULL_SHARD run can approach DDP throughput; when it doesn't — small layers, a slow interconnect, or a prefetch setting that serializes comm behind compute — you get the "8 GPUs, barely faster than 1" disappointment. This is a throughput bug, not a correctness bug, but it lives in the same systems quadrant and you diagnose it the same way: read the instruments (a profiler trace showing comm not overlapping compute) rather than guessing.

#### Worked example: the memory math, DDP vs FSDP, on a 1.3B model

Take a 1.3B-parameter model, Adam, bf16 compute with an fp32 master copy. Model-state bytes ≈ `16 × 1.3e9 ≈ 20.8 GB`.

- **DDP on 4× 24 GB GPUs:** each GPU holds the full 20.8 GB of state. Add even a modest activation footprint (say 4 GB) and you are at ~25 GB — over budget. DDP **OOMs** on a 24 GB card. The error you'd see is `CUDA out of memory` at the optimizer step.
- **FSDP FULL_SHARD on the same 4 GPUs:** state per GPU ≈ `20.8 / 4 = 5.2 GB`. With the same 4 GB of activations you sit at ~9.2 GB, plus a transient all-gather buffer for the largest unit (one transformer block of, say, 0.3 GB). Comfortably under 24 GB. **FSDP fits.**

The difference is not subtle — it is the entire reason FSDP exists, and it tells you the *first* thing to check when someone says "I switched to FSDP and it still OOMs": did per-GPU memory actually drop? If it didn't, the sharding isn't happening, and that is a wrapping-policy bug, not a memory bug.

One more piece of the memory math deserves its own line, because it confuses people who *did* shard correctly and still OOM: **the transient all-gather buffer.** While a unit is in use, that unit's *full* parameter is materialized on every rank — that's the whole point of the gather. So the live memory at any instant is `(sharded state) + (full size of the currently-gathered unit) + (activations)`. If your single largest unit is enormous — say you wrapped the model so one unit contains a 20,000-dimensional embedding plus several blocks — the gather buffer for that one unit can dominate, and you OOM at the moment it gathers even though the *average* footprint looks fine. This is why the wrapping granularity controls peak memory: it sets the size of the largest thing ever gathered at once. The diagnostic is the same `max_memory_allocated` reading, but the *interpretation* is different — a spiky memory profile that peaks during one specific unit's forward points at an oversized unit, not at insufficient sharding overall. The fix is to wrap that unit more finely so its gather buffer shrinks.

#### Worked example: where ZeRO-2 saves you a GPU

Suppose you have a 2.7B model and 8× 24 GB GPUs, and you'd rather not pay the full-shard communication tax. Model-state bytes ≈ `16 × 2.7e9 ≈ 43 GB`. Under DDP, 43 GB of state per GPU — instant OOM on a 24 GB card. Under `SHARD_GRAD_OP` (ZeRO-2: shard grads + optimizer, replicate params), per-GPU state ≈ `params 2P + (grads+optim) 14P/N` ≈ `2×2.7e9 + 14×2.7e9/8` bytes ≈ `5.4 GB + 4.7 GB ≈ 10.1 GB`. That fits, with ~14 GB left for activations and the gather buffers — and you skip the parameter all-gather on every layer, so each step is faster than FULL_SHARD. Under `FULL_SHARD` the same model would sit at `~43/8 ≈ 5.4 GB` of state, leaving even more headroom but adding the per-layer parameter gather. The lesson: **measure first, then pick the least aggressive strategy that fits.** Reaching for FULL_SHARD reflexively when ZeRO-2 would do is a throughput bug hiding as a config choice — a few percent to 20% slower depending on model shape and interconnect.

## 2. The resume that explodes: checkpoint save/load under FSDP

This is the headline bug, the one in the intro, and the one that costs the most compute. Let's nail the mechanism, then build the diagnostic that catches it in 100 steps instead of after a two-day rerun.

### 2.1 The mechanism: state_dict types and the sharded optimizer

Under FSDP, there is no single canonical `state_dict`. There are three, and **the save and the load must agree on which one you used**:

- **`FULL_STATE_DICT`** — every rank's shard is all-gathered onto rank 0 (or all ranks) into the full, unsharded tensors. This is what a single-GPU `torch.load` expects. It is portable: you can reload it onto a different number of GPUs, or even onto one GPU for inference. The cost is a big all-gather and a memory spike on the gathering rank (it briefly holds the whole model).
- **`SHARDED_STATE_DICT`** — each rank saves its own `1/N` shard, with metadata describing how to reassemble. This is fast, has no memory spike, and is the recommended default for large models. It can be resharded onto a *different* world size on load (this is the modern distributed-checkpoint path).
- **`LOCAL_STATE_DICT`** — each rank saves its flat local shard with no resharding metadata. Fast, but you can only reload onto the **exact same** world size and wrapping. Brittle; mostly legacy.

The bug has three classic flavors:

1. **Type mismatch on weights.** You save with `SHARDED_STATE_DICT` but the resume code calls a plain `torch.load` expecting a full dict, or vice versa. Sometimes this throws a clear key error. The *dangerous* version is when it silently loads zeros or partial tensors and you don't notice until the loss is wrong.
2. **Optimizer state silently dropped.** The weight `state_dict` is the part everyone remembers. The optimizer `state_dict` under FSDP needs its own special handling — `FSDP.optim_state_dict()` to save and `FSDP.optim_state_dict_to_load()` to reload — because the moments are sharded just like the params. If you save the model but skip the optimizer (or save the optimizer with the wrong helper so it loads as empty), **Adam restarts with `m = 0, v = 0`.** This is the explosion.
3. **Per-shard misassignment.** Rank 2's moments get restored to rank 1's parameter shard because the resharding metadata was wrong or the world size changed without a resharding-capable format. The optimizer applies a history that belongs to a different slice of weights.

There is a fourth, sneakier flavor specific to **mixed precision plus checkpointing: the master weights.** When you train in bf16 with an fp32 master copy, the *master* weights are the ground truth — they accumulate the tiny updates that would round away in bf16, and the bf16 compute params are derived from them each step. The Adam moments are computed against the fp32 master. If your checkpoint saves only the bf16 compute params (the ones easy to grab from `model.parameters()`) and reconstructs an fp32 master from them on load, you have **quantized the master weights through bf16** — you've thrown away the sub-bf16 precision that the fp32 master existed to preserve. The resume won't *explode* (the weights are approximately right), but it won't be bit-exact, and over a long resume-heavy run the repeated bf16 round-tripping of the master can slowly degrade quality. The FSDP-aware state-dict APIs save the fp32 master correctly; the trap is hand-rolling a save that grabs only `model.parameters()`. This is the same master-weights subtlety that bites plain AMP, just harder to see because the master is also sharded.

To be concrete about *what* the optimizer state contains and why it's sharded the same way as the params: for a parameter shard of shape `[P/N]` on rank `r`, Adam stores `exp_avg` (the first moment `m`, same shape `[P/N]`) and `exp_avg_sq` (the second moment `v`, same shape `[P/N]`), plus a scalar `step` count. These per-element moments are meaningless without their matching parameter elements — `exp_avg_sq[i]` is the running average of `grad[i]²` for *this specific weight*. Restore them to a different shard and every per-parameter scale is wrong. This is precisely why a naive `torch.save(optimizer.state_dict())` followed by `torch.load` can silently corrupt: under FSDP the flat-parameter layout means the optimizer's state tensors are indexed by flat-shard position, and unless the resharding metadata travels with them, "position 0 on rank 2" after a reshape is not the same weight it was before.

### 2.2 Why a missing optimizer state makes the loss explode

Here is the *why*, with the math, because "the loss explodes" deserves a mechanism, not a shrug. Adam's update for parameter `θ` at step `t` is:

$$
\theta_{t} = \theta_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}, \quad \hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t},
$$

where `m` and `v` are the running first and second moments and `t` is the step count. Two things go wrong when `m`, `v`, and `t` are reset to zero on resume:

1. **The second moment `v` is what tames the step size.** At step 5,000, `v` has accumulated the running average of squared gradients, so `√v` is a sensible per-parameter scale and the effective update is well-conditioned. Reset `v = 0` and the very first update divides by `√(0) + ε = ε ≈ 1e-8`. The effective learning rate per parameter is momentarily enormous: $\eta \cdot \hat m_t / \epsilon$. The first step takes a giant, badly scaled jump.
2. **Bias correction with `t` reset.** If the step counter resets to `t = 1`, the bias-correction denominators `1 − β₁` and `1 − β₂` are at their smallest, amplifying the moments further. The intended bias correction (which is gentle by step 5,000) becomes aggressive again.

The result: one or a few catastrophic steps that throw the weights far from the converged region, the loss jumps from `0.31` back toward random (`~8`), and from there gradients can overflow into `NaN`. The weights *loaded correctly* — that is the cruel part. It's the **optimizer history** that vanished. Figure 3 contrasts the exploded resume against the clean one.

![Two-column before and after figure contrasting a resume that resets Adam moments to zero and jumps the loss from 0.3 to 8 against a resume that restores the sharded moments and continues the loss flat at 0.3](/imgs/blogs/fsdp-and-sharding-bugs-3.png)

### 2.3 The definitive diagnostic: the resume-equivalence test

Do not test resume by resuming your real two-day run. Test it on a tiny run you can iterate on in two minutes: train 100 steps, save, then compare a *continued* run against a *resumed* run at step 101. If the optimizer, RNG, scheduler, and weights all round-tripped, the two losses are identical to floating-point noise. If anything was dropped, the resumed loss diverges immediately. Figure 4 puts this signature into the broader symptom-test-fix table; figure 5 lays out the test itself as a timeline — two branches off a single saved checkpoint, compared at the same step.

![Timeline of the resume-equivalence test branching at step 100 into a continued run and a resumed run, then comparing the two losses at step 101 where a tiny delta proves an exact resume and a jump proves dropped state](/imgs/blogs/fsdp-and-sharding-bugs-5.png)

The logic is worth stating precisely because it is the foundation of every fix in this post. Training is a deterministic function of state: given the same weights, the same optimizer moments, the same RNG state, the same scheduler position, and the same next batch, step `t → t+1` produces a *bit-identical* result (modulo nondeterministic CUDA kernels, which is why we allow a small tolerance). The resume-equivalence test branches that function at step 100. Branch A carries the full live state forward in memory. Branch B serializes the state to disk, tears down the process, rebuilds it, deserializes, and steps. If branch B's step 101 matches branch A's step 101, then *everything that mattered* survived the round-trip — and if it doesn't, the size and sign of the gap localize the missing piece. A large positive jump (resumed loss higher) is dropped optimizer or weight state; a small persistent gap is RNG or dataloader position; an exact match is a clean resume. This single test, run on a toy model in CI, is worth more than any amount of staring at the long run's loss curve.

Here is the correct FSDP checkpoint save/load using the recommended `SHARDED_STATE_DICT` path with the distributed-checkpoint API:

```python
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, ShardedStateDictConfig, ShardedOptimStateDictConfig
from torch.distributed.checkpoint.state_dict import (
    get_state_dict, set_state_dict,
)

def save_checkpoint(model, optimizer, step, path):
    # get_state_dict returns FSDP-aware, resharding-capable state dicts.
    model_sd, optim_sd = get_state_dict(model, optimizer)
    state = {"model": model_sd, "optim": optim_sd, "step": step}
    # Distributed save: every rank writes its own shard in parallel.
    dcp.save(state, checkpoint_id=f"{path}/step_{step}")

def load_checkpoint(model, optimizer, path, step):
    # IMPORTANT: read into the SAME structure the save produced.
    model_sd, optim_sd = get_state_dict(model, optimizer)
    state = {"model": model_sd, "optim": optim_sd}
    dcp.load(state, checkpoint_id=f"{path}/step_{step}")
    # Push the loaded shards back into the live model + optimizer.
    set_state_dict(
        model, optimizer,
        model_state_dict=state["model"],
        optim_state_dict=state["optim"],
    )
```

The single most common mistake is loading the model but *not* threading the optimizer through `get_state_dict` / `set_state_dict` (or, on older code, `FSDP.optim_state_dict_to_load`). The model loads, the optimizer is silently fresh, and you get the explosion. Note that `get_state_dict` handles the optimizer's per-shard moments correctly — that is the whole point of using it instead of `optimizer.state_dict()` directly.

Now the test that proves it. Run this once before trusting any long run:

```python
import copy

def resume_equivalence_test(make_trainer, path, n=100):
    # make_trainer() returns (model, optimizer, data_iter, train_step_fn)
    # train_step_fn(model, optimizer, batch) -> float loss

    # --- Branch A: train n+1 steps without stopping ---
    model, opt, data, step_fn = make_trainer()
    losses_a = []
    for i in range(n + 1):
        batch = next(data)
        losses_a.append(step_fn(model, opt, batch))
        if i == n - 1:
            save_checkpoint(model, opt, step=n, path=path)

    # --- Branch B: fresh process loads the checkpoint and does 1 step ---
    model2, opt2, data2, step_fn2 = make_trainer()
    load_checkpoint(model2, opt2, path=path, step=n)
    # The data iterator must also be at the same position (see section 6).
    batch = next(data2)  # must equal branch A's step-n batch
    loss_b = step_fn2(model2, opt2, batch)

    delta = abs(losses_a[n] - loss_b)
    if dist.get_rank() == 0:
        print(f"continued loss @ {n}: {losses_a[n]:.6f}")
        print(f"resumed   loss @ {n}: {loss_b:.6f}")
        print(f"delta: {delta:.3e}  {'PASS' if delta < 1e-3 else 'FAIL'}")
    return delta
```

A `PASS` (delta below ~`1e-3`, ideally below `1e-5`) means your save/load round-trips every piece of state. A `FAIL` with the resumed loss *higher* than the continued loss is the classic optimizer-or-RNG drop. The threshold isn't zero because FSDP collectives and non-deterministic CUDA kernels add a little float noise; if you've set `torch.use_deterministic_algorithms(True)` you can tighten it toward `1e-6`.

![Field-guide matrix mapping four FSDP bugs — resume loss jump, no memory savings, wrong grad clip, and silent bf16 drift — each to its instrument signature, cheapest confirming test, and fix direction](/imgs/blogs/fsdp-and-sharding-bugs-4.png)

#### Worked example: catching the dropped optimizer in 100 steps

A real signature from a finetune. Branch A (continued) and branch B (resumed) on a 350M model, bf16, FSDP FULL_SHARD across 4 GPUs:

| Step | Branch A (continued) | Branch B v1 (optimizer dropped) | Branch B v2 (optimizer restored) |
|---|---|---|---|
| 100 | 0.412 | — | — |
| 101 | 0.409 | **2.87** | 0.409 |
| 102 | 0.407 | 5.41 | 0.407 |
| 105 | 0.401 | NaN | 0.401 |

In v1, the resumed loss at step 101 is `2.87` against the continued `0.409` — a `delta` of ~`2.46`, an instant `FAIL`, and it's `NaN` by step 105. We confirmed the cause by printing `optimizer.state_dict()["state"]` after load: every parameter's `exp_avg` and `exp_avg_sq` were zero tensors. The fix was a one-liner — switch the load path from a plain `torch.load` of only the model to the `get_state_dict` / `set_state_dict` pair above so the optimizer's sharded moments came back. v2 then matched the continued run to `delta < 5e-4`. **Before:** resume loss `0.41 → 2.87 → NaN`. **After:** resume loss `0.41 → 0.41`, indistinguishable from never stopping. Total debug time, because we tested on 100 steps: under ten minutes. The original symptom cost two days.

This sibling failure mode — the loss jump on resume from any dropped state, including the scheduler and RNG — is the whole subject of [debugging checkpoint and resume](/blog/machine-learning/debugging-training/debugging-checkpoint-and-resume); FSDP just adds the per-shard wrinkle on top.

## 3. Mixed precision under FSDP: the MixedPrecision policy

The second-largest source of FSDP-specific bugs is mixed precision, because FSDP gives you fine-grained control over the dtype of *three different things*, and the defaults are not always what you want. This is where an FSDP run silently loses two points of accuracy versus the same model on DDP+AMP.

### 3.1 The three dtypes FSDP controls

PyTorch's `MixedPrecision` policy has three independent dtype knobs:

```python
from torch.distributed.fsdp import MixedPrecision
import torch

bf16_policy = MixedPrecision(
    param_dtype=torch.bfloat16,    # dtype of the gathered params during compute
    reduce_dtype=torch.float32,    # dtype of the gradient reduce-scatter
    buffer_dtype=torch.bfloat16,   # dtype of buffers (e.g. norm running stats)
)
```

- **`param_dtype`** is the dtype the parameter is cast to when all-gathered for the forward/backward. Lower precision here saves the most communication and compute.
- **`reduce_dtype`** is the dtype of the gradient *reduction* (the reduce-scatter). **This is the subtle one.** If you reduce gradients in bf16, you accumulate the sum of `N` gradients in a format with only 8 bits of mantissa, and the rounding error compounds across ranks. Reducing in fp32 keeps the accumulation accurate at the cost of a bit more comm bandwidth.
- **`buffer_dtype`** is the dtype of non-parameter buffers — most importantly BatchNorm/LayerNorm running statistics. Cast these too aggressively and your normalization stats drift.

### 3.2 The science: bf16 vs fp16 under sharding

The numerics here are the same as in any mixed-precision setup — see [mixed precision debugging, fp16 vs bf16](/blog/machine-learning/debugging-training/mixed-precision-debugging-fp16-vs-bf16) for the full treatment — but sharding changes *where* the precision loss bites.

**fp16** has a 10-bit mantissa and a small exponent range; its smallest normal positive number is about $6.1\times10^{-5}$. Gradients in large models routinely fall below that and **underflow to zero**, which is why fp16 needs loss scaling (multiply the loss by a large factor before backward, unscale before the step). **bf16** has the same 8-bit exponent as fp32 — so its range is huge, $\sim 3.4\times10^{38}$ — but only a 7-bit mantissa, giving ~2-3 significant decimal digits. bf16 essentially never underflows, so it needs no loss scaling, which is why it's the default for large-model training.

Under FSDP the reduce-scatter is the danger point. Consider summing `N = 8` gradient contributions during the reduction. In bf16, each addition rounds to ~3 significant digits; summing eight numbers of varying magnitude accumulates relative error on the order of $N \cdot 2^{-8} \approx 0.03$ in the worst case — a few percent of error injected into *every* gradient, *every* step. That is enough to leave a couple of points of eval quality on the table across a long run. Setting `reduce_dtype=torch.float32` while keeping `param_dtype=torch.bfloat16` eliminates the reduction error for a small bandwidth cost — the standard production recipe.

A second trap is specific to **fp16 + FSDP**: the `GradScaler`. With plain AMP+DDP you use `torch.cuda.amp.GradScaler`. With FSDP you must use `torch.distributed.fsdp.sharded_grad_scaler.ShardedGradScaler`, because the scaler has to inspect gradients that are *sharded* across ranks to detect inf/NaN and synchronize the scale factor. Using the wrong scaler means each rank decides its own scale, the scales desync, and you get inconsistent skipping of steps across ranks — a subtle divergence. (With bf16 you skip the scaler entirely, which is one more reason bf16 is the path of least pain.)

### 3.3 Diagnostic: log the reduce dtype and compare a tiny run

The diagnostic is to make the silent dtype visible and to A/B a short run:

```python
# Print exactly what the policy is doing — never assume the default.
for module in model.modules():
    if isinstance(module, FSDP):
        mp = module.mixed_precision
        print(f"param={mp.param_dtype} reduce={mp.reduce_dtype} buffer={mp.buffer_dtype}")
        break

# A/B test: 500 steps each, identical seed, compare eval.
#   run 1: reduce_dtype=torch.bfloat16
#   run 2: reduce_dtype=torch.float32
# If run 2 is meaningfully better on eval, your reduction precision was the bug.
```

#### Worked example: two points of accuracy hiding in reduce_dtype

A vision-transformer finetune across 8 GPUs, FSDP FULL_SHARD, bf16. Two configs, identical seed, 3 epochs:

| Config | param_dtype | reduce_dtype | Top-1 val acc | Throughput |
|---|---|---|---|---|
| A | bf16 | **bf16** | 81.4% | 1.00× |
| B | bf16 | **fp32** | **83.1%** | 0.97× |

Reducing gradients in fp32 recovered **1.7 points** of top-1 accuracy for a 3% throughput cost. The bf16 reduction had been quietly injecting a few percent of relative error into every gradient, and over three epochs that compounded into a real quality gap. The confirming test was exactly the A/B above; the fix was one field in the `MixedPrecision` policy. Note this is *not* a bug in the usual sense — config A runs fine and converges — which is what makes it dangerous: nothing ever errors. You only catch it by suspecting `reduce_dtype` and measuring.

### 3.4 The same knobs in DeepSpeed ZeRO and accelerate

FSDP is PyTorch-native, but most people meet sharding through `accelerate` or DeepSpeed, and the same three dtype decisions live there under different names. It helps to know the translation so a bug you learned in one shows up recognizably in the other.

In **DeepSpeed ZeRO**, the precision is set in the JSON config: `"bf16": {"enabled": true}` or `"fp16": {"enabled": true, "loss_scale": 0, "initial_scale_power": 16}`. The `reduce_dtype` analog is `"communication_data_type"` (and the related `"reduce_scatter"`/`"allgather_partitions"` options) — set it to fp32 for the same accuracy reason. DeepSpeed handles loss scaling internally when `loss_scale: 0` (dynamic scaling), so you don't manage a `GradScaler` yourself. The cold-resume bug exists here too, but DeepSpeed's `model_engine.save_checkpoint()` / `load_checkpoint()` save the partitioned optimizer state by default — the trap is usually a tag/path mismatch or loading with a different ZeRO stage than you saved.

In **`accelerate`**, you configure FSDP through `accelerate config` (or an `FSDPPlugin`), and it sets `mixed_precision`, the auto-wrap policy, and the sharding strategy for you. Crucially, `accelerator.clip_grad_norm_`, `accelerator.save_state`, and `accelerator.load_state` are all sharding-aware — so the local-clip bug (section 5) and the cold-resume bug (section 2) are *avoided by construction* if you use the accelerate wrappers instead of raw PyTorch calls. The trap shifts: people mix raw `torch.save(model.state_dict())` into an otherwise-accelerate run and reintroduce the very bug accelerate was preventing. The rule is **pick one ownership model and stick with it** — either let accelerate/DeepSpeed own checkpointing and clipping, or own them yourself with the FSDP-aware APIs, but never half-and-half.

```yaml
# accelerate FSDP config (config.yaml) — the safe defaults
fsdp_config:
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_use_orig_params: true
mixed_precision: bf16
```

The single line that most often needs editing is `fsdp_transformer_layer_cls_to_wrap` — same trap as the raw `transformer_layer_cls`, same fix: it must be your model's exact block class.

## 4. The wrapping policy: shard at the wrong granularity and save nothing

This is the bug behind "I switched to FSDP and per-GPU memory didn't drop." FSDP shards at the granularity of **FSDP units** — wrapped modules — and the wrapping policy decides what becomes a unit. Get it wrong in either direction and you either save no memory or drown in communication.

### 4.1 The mechanism: a unit is the unit of gather and free

Recall from section 1.3 that a parameter is all-gathered to full size, used, and freed *per FSDP unit*. The peak transient memory for the gather is set by the **largest single unit**. So:

- **Wrap the whole model as one unit** (the accidental default if you just call `FSDP(model)` with no auto-wrap policy on some setups). There is exactly one gather, and it gathers the *entire* model to full size for the whole forward and backward. Peak memory is back to roughly the full model — you've paid for FSDP's complexity and gotten **no memory savings**. This is the single most common FSDP disappointment.
- **Wrap too finely** — every `Linear`, every `LayerNorm` as its own unit. Now you have thousands of tiny all-gathers and reduce-scatters, each with collective-launch overhead. Comm dominates, and an 8-GPU run can be slower than 1 GPU. (This is its own throughput trap — see [the GPU is idle, throughput debugging](/blog/machine-learning/debugging-training/out-of-memory-debugging) for the profiling angle on comm-bound runs.)
- **Wrap each transformer block as a unit** — the sweet spot. One block is gathered, used, freed; the next block is gathered, used, freed. Peak transient memory is one block's worth, the comm is one collective per block (well amortized against the block's compute), and the savings are real. Figure 6 contrasts the one-unit and per-block extremes.

![Two-column before and after figure contrasting wrapping the whole model as one FSDP unit with 60 GB peak memory and no savings against per-block wrapping that gathers one block at a time for 22 GB peak](/imgs/blogs/fsdp-and-sharding-bugs-6.png)

### 4.2 The right policy: transformer_auto_wrap_policy

For any transformer, use the built-in policy that wraps each block:

```python
import functools
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

auto_wrap = functools.partial(
    transformer_auto_wrap_policy,
    # The class of the repeating block — THIS is what becomes a unit.
    transformer_layer_cls={LlamaDecoderLayer},
)

model = FSDP(
    model,
    auto_wrap_policy=auto_wrap,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    mixed_precision=bf16_policy,
    device_id=torch.cuda.current_device(),
)
```

The one line that bites people: `transformer_layer_cls` must be the **exact class of your repeating block** (e.g. `LlamaDecoderLayer`, `GPT2Block`, `BertLayer`). If you pass the wrong class — or a base class that doesn't match — the policy matches *nothing*, no sub-modules get wrapped, the whole model becomes one root unit, and you're back to no savings. With a custom model, point it at your own block class. The alternative `size_based_auto_wrap_policy` (wrap any submodule above `min_num_params`) is a reasonable fallback for non-transformer models.

### 4.3 Diagnostic: confirm the sharding actually happened

Never assume the wrapping worked — **measure per-GPU memory and count the units.** Two cheap checks:

```python
# 1. Count FSDP units. One root + one per block is what you want.
n_units = sum(isinstance(m, FSDP) for m in model.modules())
print(f"FSDP units: {n_units}")   # e.g. 33 for a 32-layer model (32 blocks + root)
# n_units == 1 means NOTHING was wrapped beyond the root: no savings.

# 2. Measure peak memory after a real step, per rank.
torch.cuda.reset_peak_memory_stats()
loss = train_step(model, optimizer, batch)   # one full fwd/bwd/step
peak_gb = torch.cuda.max_memory_allocated() / 1e9
print(f"rank {dist.get_rank()} peak: {peak_gb:.1f} GB")
# Compare to the DDP estimate ~16P bytes. FULL_SHARD on N GPUs should be ~1/N of that.
```

If `n_units == 1`, your wrapping policy matched nothing — fix `transformer_layer_cls`. If `n_units` is in the thousands, you wrapped too finely — back off to block granularity. And the decisive instrument is `max_memory_allocated`: under FULL_SHARD on `N` GPUs it should land near `1/N` of the DDP figure. If per-GPU memory is the same as DDP, the sharding is not happening, full stop.

A deeper check, when you want to *see* the sharding rather than infer it from memory, is to inspect the flat parameter each rank actually holds. With `use_orig_params=False` FSDP replaces a unit's parameters with a single flattened `FlatParameter`; with `use_orig_params=True` (recommended) the original named parameters survive but their `.data` is a view into the shard. Either way you can confirm each rank holds roughly `1/N` of the elements:

```python
import torch.distributed as dist

def report_sharding(model):
    rank, world = dist.get_rank(), dist.get_world_size()
    local_numel = sum(p.numel() for p in model.parameters())
    # Sum the per-rank element counts across all ranks.
    t = torch.tensor([local_numel], device=torch.cuda.current_device())
    dist.all_reduce(t)
    total_numel = t.item()
    expected = total_numel / world
    if rank == 0:
        print(f"world size: {world}")
        print(f"total params across ranks: {total_numel:,}")
        print(f"per-rank expected ~{expected:,.0f}")
    # Each rank prints its own local count; they should each be ~total/world.
    print(f"rank {rank}: holds {local_numel:,} elements "
          f"({100*local_numel/expected:.0f}% of even share)")
```

On a correctly-sharded model every rank reports close to `100%` of the even share. If rank 0 reports `~100%` of the *total* (not the share) and the others report near zero, your params live entirely on rank 0 — a wrapping or initialization bug. This is the most direct possible answer to "is the sharding actually happening," and it costs one all-reduce of a single integer.

#### Worked example: 33 units that became 1

A team reported "FSDP OOMs on a 13B model on 8× 40 GB GPUs, same as DDP." We ran the unit count: `FSDP units: 1`. The model was a custom GPT variant, and the wrap policy had been copy-pasted with `transformer_layer_cls={GPT2Block}` while the actual block class was `CustomGPTBlock`. The policy matched nothing, the whole 13B model was one unit, and the all-gather tried to materialize the full ~26 GB of bf16 params plus an fp32 master and activations on one GPU — instant OOM. Changing the class to `CustomGPTBlock` produced `FSDP units: 41` (40 blocks + root), peak memory dropped from OOM to **18.6 GB** per GPU, and the run trained. **Before:** `n_units = 1`, OOM. **After:** `n_units = 41`, peak 18.6 GB. The fix was one identifier in a set literal — and the diagnostic that found it was a single `print`.

### 4.4 Wrapping interacts with activation checkpointing

There is one more wrapping subtlety that catches people combining FSDP with **activation (gradient) checkpointing** — the technique that frees activations during the forward and recomputes them in the backward to save memory. The order of wrapping matters. You want activation checkpointing applied to the *same* block granularity as the FSDP unit, and applied *under* the FSDP wrap, so that the recomputation in the backward sees the gathered parameters. PyTorch provides `apply_activation_checkpointing` with a checker that matches your block class:

```python
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing, checkpoint_wrapper, CheckpointImpl,
)
import functools

non_reentrant = functools.partial(
    checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)
apply_activation_checkpointing(
    model,
    checkpoint_wrapper_fn=non_reentrant,
    check_fn=lambda m: isinstance(m, LlamaDecoderLayer),  # same block class
)
```

Two traps. First, prefer `NO_REENTRANT` checkpointing with FSDP — the older reentrant implementation interacts badly with FSDP's backward hooks and can recompute with the wrong (un-gathered) parameters or double-trigger the all-gather, wasting comm. Second, if your activation-checkpoint granularity doesn't match your FSDP-unit granularity, you can end up gathering a parameter, freeing it, then re-gathering it during the recompute — paying the all-gather twice. Match the granularities (block = FSDP unit = checkpoint unit) and the recompute reuses the already-gathered parameters within the unit's backward window. The signature of getting this wrong is *memory savings that are smaller than expected combined with surprisingly low throughput* — the recompute is paying extra communication. Profile the run (the comm-vs-compute breakdown) if the numbers don't add up; the throughput angle is covered in [out-of-memory debugging](/blog/machine-learning/debugging-training/out-of-memory-debugging).

## 5. Gradient clipping under sharding: the clip that silently does nothing

Gradient clipping is a one-liner you've written a hundred times, and under FSDP the obvious version is wrong. The wrongness is silent: it doesn't error, it just clips to the wrong norm.

### 5.1 The mechanism: a local norm is not the global norm

Gradient clipping by global norm rescales all gradients so the total L2 norm doesn't exceed `max_norm`. The global norm is:

$$
\|g\|_2 = \sqrt{\sum_{i} g_i^2}\,,
$$

summed over *every* parameter. Under FSDP, each rank holds only `1/N` of the gradients. If you call the standard `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)`, each rank computes the norm of *its shard only*:

$$
\|g^{(r)}\|_2 = \sqrt{\sum_{i \in \text{shard } r} g_i^2}\,,
$$

which is roughly `√(1/N)` of the true global norm (if the gradient energy is spread evenly across shards). On 8 GPUs the local norm is about `1/√8 ≈ 0.35` of the true norm. So clipping to `max_norm=1.0` actually clips each shard to a global-equivalent of `~2.8` — the clip is effectively a no-op, and you lose the protection against the loss spikes clipping was supposed to prevent. Worse, the clip factor differs across ranks (shards aren't perfectly even), so different shards get scaled differently, which is itself incorrect.

The correct computation all-reduces the squared local norms across ranks *before* the square root, then applies one consistent scale to every shard. Figure 8 shows this flow.

![Dataflow graph showing each rank computing a local squared gradient norm, all-reducing the squares into a global sum, taking the square root for the true global norm, and scaling every shard by one consistent clip factor](/imgs/blogs/fsdp-and-sharding-bugs-8.png)

### 5.2 The fix: FSDP.clip_grad_norm_

PyTorch FSDP provides a **sharding-aware** clip as a method on the FSDP-wrapped module:

```python
# WRONG under FSDP — clips each shard's local norm independently.
# torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# RIGHT — all-reduces the squared norms, then scales every shard consistently.
total_norm = model.clip_grad_norm_(max_norm=1.0)
if dist.get_rank() == 0:
    print(f"global grad norm (pre-clip): {total_norm:.3f}")
```

`model.clip_grad_norm_` (where `model` is the FSDP-wrapped root) returns the **true global** pre-clip norm, which is exactly the instrument you want to log. If you're on `accelerate`, `accelerator.clip_grad_norm_` does the sharding-aware all-reduce for you — one more reason to let the framework own the clip. DeepSpeed handles clipping internally via its config (`gradient_clipping`), so you don't call it manually there.

### 5.3 Diagnostic: compare the logged norm to a single-GPU baseline

The tell is the magnitude of the logged grad norm. Run the same tiny batch on 1 GPU (no FSDP) and on `N` GPUs (FSDP), and log the grad norm:

```python
# Single-GPU reference:
norm_1gpu = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e9)  # huge max = measure only
# FSDP on N GPUs:
norm_fsdp = model.clip_grad_norm_(max_norm=1e9)
# These should AGREE. If norm_fsdp ~ norm_1gpu / sqrt(N), you used the wrong clip.
```

If the FSDP grad norm comes back about `1/√N` of the single-GPU norm, you're computing a per-shard local norm — switch to `FSDP.clip_grad_norm_`. If they agree (to float noise), you're clipping the true global norm.

#### Worked example: the clip that wasn't clipping

A 7B finetune on 8 GPUs had occasional loss spikes despite `clip_grad_norm_(..., 1.0)` in the loop. We logged the returned norm and saw values like `0.34`, `0.31`, `0.36` — suspiciously small and stable, never anywhere near `1.0`, so the clip never fired. A single-GPU repro on the same batch showed the true grad norm was `~0.95`, occasionally spiking to `~6`. The FSDP loop was reporting `~0.34 ≈ 0.95 / √8`: a per-shard local norm. We were using `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`, which on the FSDP shards clipped a local norm — so when the *true* norm spiked to 6, the local norm was only `~2.1`, still above 1 on some shards but inconsistently, and the spike got through. Switching to `model.clip_grad_norm_(1.0)` made the logged norm jump to the real `0.95`/`6` range, the clip started actually firing at the `6` spikes, and the spikes stopped reaching the loss. **Before:** logged norm `~0.34`, spikes leak through. **After:** logged norm `~0.95` (true), clip fires on the `6` spikes, no more loss spikes. The general spike phenomenon and the difference between a transient spike and true divergence is its own topic; here the FSDP-specific lesson is just: **the clip must be sharding-aware, and the returned norm is your proof it is.**

## 6. RNG, frozen params, buffers, and CPU offload

A few smaller — but still run-wrecking — sharding interactions round out the picture. Each is cheap to check once you know it exists.

### 6.1 RNG state on resume

The resume-equivalence test (section 2.3) will `FAIL` even with a perfect optimizer restore if the **RNG state** isn't restored. Dropout masks, data shuffling, and any stochastic augmentation depend on the random generators, and under multi-GPU you have *per-rank* RNG state. If you don't save and restore `torch.get_rng_state()`, `torch.cuda.get_rng_state_all()`, and your dataloader's generator state, the resumed run sees *different* dropout masks and *different* data order than the continued run. The loss won't explode — but it won't be bit-identical either, and your equivalence test won't pass cleanly. Save the RNG state in the checkpoint:

```python
state["rng"] = {
    "cpu": torch.get_rng_state(),
    "cuda": torch.cuda.get_rng_state_all(),
    "python": random.getstate(),
    "numpy": np.random.get_state(),
}
# On load:
torch.set_rng_state(state["rng"]["cpu"])
torch.cuda.set_rng_state_all(state["rng"]["cuda"])
```

Reproducibility is the precondition for *any* equivalence test — a run you can't reproduce is a run you can't bisect. If your resume test is noisy even with state restored, fix determinism first; the full treatment is in [reproducibility and determinism in training](/blog/machine-learning/debugging-training/reproducibility-and-determinism-in-training).

There is a companion to RNG state that's just as easy to drop: the **dataloader position**. Branch B in the equivalence test must consume the *same next batch* branch A consumed at step 100. If your dataloader is a plain `for batch in loader` with shuffling, resuming reshuffles from the top of the epoch and branch B trains on a *different* batch than branch A — the loss won't match, and it's not the optimizer's fault. Modern checkpointing saves the sampler/iterator state (PyTorch's `StatefulDataLoader` from `torchdata`, or `accelerate`'s `skip_first_batches`, or a manual record of `(epoch, step_within_epoch)`). The practical tell on the equivalence test: if the optimizer and RNG are restored and the delta is still moderate (say `~1e-2`) and *jittery* rather than a clean explosion, suspect the dataloader served a different batch. Restoring the data position is the last piece that takes the delta to float-noise.

Here is the symptom-to-piece mapping for resume bugs, which is the lookup you actually use at 2 a.m.:

| Resume signature | Missing piece | Confirming check |
|---|---|---|
| Loss `0.3 → 8 → NaN` | Optimizer moments | `exp_avg_sq` all zeros after load |
| Loss off by ~`1e-2`, jittery | Dataloader position | resumed batch ≠ continued batch |
| Loss off by ~`1e-2`, smooth | RNG state | dropout masks differ across branches |
| Loss off, LR wrong | Scheduler state | `scheduler.last_epoch` reset to 0 |
| KeyError / shape error on load | state_dict type mismatch | save type ≠ load type |
| Works on N GPUs, fails on M | No resharding metadata | used `LOCAL_STATE_DICT`, changed world size |

### 6.2 Frozen parameters under FSDP

Finetuning often freezes part of the model (`requires_grad=False`) — a frozen backbone, frozen embeddings, or LoRA's frozen base weights. Under FSDP this has two gotchas:

1. **Mixed `requires_grad` within one FSDP unit.** If a single wrapped unit contains both trainable and frozen parameters, older FSDP versions could error or behave inconsistently because the unit's flat parameter mixes the two. The safe pattern is to make the freeze granularity match the wrap granularity — freeze whole blocks, or use `use_orig_params=True` (now the recommended default), which lets FSDP handle per-original-parameter `requires_grad` correctly.
2. **A fully-frozen unit produces no gradient**, which interacts with the reduce-scatter the same way an unused parameter interacts with DDP's reducer. With `use_orig_params=True` and modern FSDP this is handled, but if you see a hang or a missing-gradient error after freezing, suspect the freeze-vs-wrap granularity mismatch first.

For LoRA specifically — frozen base, trainable adapters, all under FSDP — the interaction with gradient checkpointing and dtype is a recurring source of silent no-ops; that's covered in [debugging LoRA and PEFT](/blog/machine-learning/debugging-training/debugging-checkpoint-and-resume).

### 6.3 Buffers and non-persistent state

Buffers — BatchNorm running stats, rotary-embedding caches, attention masks registered as buffers — are *not* parameters and are *not* sharded; FSDP replicates them. Two consequences:

- Their dtype is governed by `buffer_dtype` in the `MixedPrecision` policy, not `param_dtype`. Casting BN running stats to bf16 can drift the statistics; keep `buffer_dtype=torch.float32` for normalization-heavy models if you see eval-mode degradation.
- **Non-persistent buffers** (registered with `persistent=False`) are intentionally excluded from the `state_dict`. That's usually correct (they're recomputable, like a causal mask), but if you registered something stateful as non-persistent by accident, it won't be in the checkpoint and your resume test will reveal a mismatch.

### 6.4 CPU offload correctness

`CPUOffload(offload_params=True)` moves the sharded params (and their grads/optimizer state) to CPU when not in use, trading PCIe bandwidth for even more GPU memory headroom — it lets you train a model that doesn't fit even sharded. The correctness traps:

- **It's slow**, often dramatically, because every gather now includes a host-to-device copy. Use it only when you genuinely can't fit otherwise, and confirm the speed cost is acceptable with a profiler.
- **The optimizer step runs on CPU** for offloaded params. Mixing a GPU optimizer with offloaded params, or assuming the params are on GPU when they're on CPU, causes device-mismatch errors. Let FSDP own the placement; don't manually `.cuda()` an offloaded parameter.
- **Checkpoint save with offload** must gather from CPU correctly — the distributed-checkpoint API handles this, but a hand-rolled save that assumes GPU tensors will grab the wrong device.

#### Worked example: the resume test that failed on RNG alone

After fixing the optimizer restore in section 2's worked example, the equivalence test *still* showed a small `delta` of `~3e-2` at step 101 — not an explosion, but not a clean pass either. The loss was `0.409` continued vs `0.431` resumed: close, but not float-noise close. The optimizer was now correct, so the suspect shifted to **RNG**: the resumed run drew different dropout masks because we hadn't restored the CUDA generator state. We added the RNG block from section 6.1 to the checkpoint. The `delta` dropped to `4e-6` — a clean `PASS`. **Before:** delta `3e-2` (RNG drift, dropout masks differ). **After:** delta `4e-6` (bit-reproducible resume). The lesson: a *small* persistent gap on the resume test is RNG or dataloader position; a *large* jump is the optimizer or weights. The magnitude of the delta tells you which.

## 7. Meta-device init and materialization

The last FSDP-specific flow worth getting right is **initialization**. For a model too large to instantiate on one device, you build it on the `meta` device — which allocates *no* real memory, just shapes and dtypes — then let FSDP shard and materialize it. Get this wrong and you either OOM at construction (the thing FSDP was supposed to prevent) or end up with uninitialized weights.

### 7.1 The mechanism

A 70B model has ~140 GB of bf16 parameters. You cannot call `Model(config)` on a single GPU or even on CPU comfortably. The meta-device flow:

1. Build the model on `meta`: `with torch.device("meta"): model = Model(config)`. No memory is allocated; the tensors are pure metadata.
2. Wrap with FSDP, passing a `param_init_fn` that tells FSDP how to materialize each module's parameters onto the real device *as it shards them*, so no rank ever holds the full model.
3. FSDP shards the meta tensors and calls the init function per unit, materializing only `1/N` of each parameter on each rank.

```python
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

with torch.device("meta"):
    model = build_model(config)   # zero memory; meta tensors only

def param_init_fn(module):
    # Materialize this module's params on the current GPU and (re)initialize.
    module.to_empty(device=torch.cuda.current_device(), recurse=False)
    # Then apply the real init (e.g. the model's reset_parameters).
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()

model = FSDP(
    model,
    auto_wrap_policy=auto_wrap,
    param_init_fn=param_init_fn,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    device_id=torch.cuda.current_device(),
)
```

### 7.2 The two bugs

1. **Forgetting `reset_parameters`** (or an equivalent real init). `to_empty` materializes the tensor but leaves it **uninitialized** — full of garbage. If you don't follow it with a real initialization, your model trains from random garbage with no sensible variance, and the loss either explodes immediately or never moves. The signature is a huge or `NaN` loss at step 0 from a freshly-built (not resumed) model.
2. **Loading pretrained weights over a meta model.** If you're finetuning, you build on meta and then load a checkpoint — in which case `reset_parameters` is moot because the load overwrites everything. But the load must happen *after* materialization, through the FSDP-aware state-dict path, or you'll try to copy into meta tensors and get a "meta tensor has no data" error. Build on meta, FSDP-wrap, *then* load the pretrained weights with the section-2 load path.

The init-vs-load order is a small thing that produces a confusing error, and it's worth a sanity print: after FSDP wrapping and any load, check that no parameter is still on the meta device with `assert not any(p.is_meta for p in model.parameters())`.

#### Worked example: a 70B finetune that started from garbage

A team building a 70B finetune on 16 GPUs reported "loss is `12.4` at step 0 and never moves." A pretrained 70B should start near its pretraining loss (single digits) and *fall*, not sit at a high plateau. The from-scratch-vs-resume cut (section 8, step 1) pointed at initialization: this was a fresh build that was *supposed* to load pretrained weights. We checked `any(p.is_meta for p in model.parameters())` after wrapping — `False`, so materialization happened — but then printed a slice of a known weight (`model.embed_tokens.weight[0, :5]`) and saw values like `[3.1e-2, -1.4e-1, 8.0e-2, ...]` that looked like a *random* init, not the pretrained embedding. The `param_init_fn` was calling `to_empty` and `reset_parameters` (random init), and the pretrained load was happening *before* the FSDP wrap, into meta tensors, so it silently no-op'd, and then `reset_parameters` overwrote with garbage at materialization. The fix was to reorder: build on meta, FSDP-wrap with materialization, *then* load the pretrained checkpoint through the FSDP-aware `set_state_dict`. After the reorder the same embedding slice matched the pretrained checkpoint, and the loss started at `2.1` and fell. **Before:** loss `12.4`, flat, embeddings random. **After:** loss `2.1`, falling, embeddings match pretrained. The lesson: a high *flat* loss on a model that should know something is an init/load-order bug; a high *exploding* loss is the optimizer-restore bug. Same "high loss" symptom, opposite root cause — which is exactly why the from-scratch-vs-resume bisection comes first.

## 8. Bisecting an FSDP failure: the full narrative

Let's put it together as one bisection, the way you'd actually debug in production. The spine: **localize to one of six places before touching code**, and FSDP bugs are in *systems* — but the symptom alone doesn't tell you which FSDP bug. Figure 7 is the decision you walk through for the checkpoint path specifically.

![Decision tree for choosing the FSDP state_dict type — sharded or local for same world size, full for portability — with the warning that a save and load mismatch produces a resume loss jump from 0.3 to 8](/imgs/blogs/fsdp-and-sharding-bugs-7.png)

**The report:** "Finetune trained fine, but after a checkpoint resume the loss exploded and then NaN'd."

**Step 1 — Is it actually a resume bug, or does it explode from scratch?** Run from scratch for 200 steps. Loss comes down cleanly. So the bug is in the resume path, not the model or data — that's a huge cut. We're now in the checkpoint/systems quadrant, not optimization or data.

**Step 2 — Run the resume-equivalence test on 100 steps.** Continued loss at step 101 is `0.41`; resumed loss is `2.9`. `FAIL`, and the resumed loss is *higher* — the classic dropped-state signature. We don't need the two-day run to debug anymore; we reproduce in two minutes.

**Step 3 — Is it the weights or the optimizer?** Print a few weight tensors before save and after load: they match. Print `optimizer.state_dict()["state"]` after load: `exp_avg` and `exp_avg_sq` are all zeros. **The optimizer state was dropped.** The weights are fine. This is the per-shard optimizer-state bug.

**Step 4 — Why was it dropped?** The load code called `torch.load` on a model-only checkpoint; the optimizer was never threaded through the FSDP-aware state-dict API. Under FSDP `optimizer.state_dict()` alone doesn't capture the sharded moments correctly, and the save had used the wrong helper.

**Step 5 — Fix and re-test.** Switch to `get_state_dict` / `set_state_dict` for both model and optimizer. Re-run the equivalence test: `delta = 3e-2`. Better — no explosion — but not a clean pass.

**Step 6 — Stress the remaining gap.** A `3e-2` delta that's *small and persistent* (not exploding) is RNG or dataloader position, per section 6. Add the RNG state to the checkpoint. Re-test: `delta = 4e-6`. **PASS.** Resume is now bit-reproducible.

**Step 7 — Confirm on the real run.** Resume the actual finetune from its step-5,000 checkpoint. Loss comes back at `0.31`, exactly where it left off, and continues down. Fixed.

Now the stress-test questions the kit demands — *what if it's not this bug?*

- **What if it OOMs instead of exploding?** Then it's not the optimizer-restore bug — it's wrapping (section 4). Count units (`n_units == 1` = no sharding) and measure peak memory.
- **What if the loss is *fine* on resume but eval is 2 points worse than expected the whole run?** Not a resume bug at all — suspect `reduce_dtype=bf16` (section 3). A/B with fp32 reduction.
- **What if it works on 8 GPUs but the resume fails when you reload on 4?** That's a resharding bug — you used `LOCAL_STATE_DICT` (no resharding metadata) and changed the world size. Use `SHARDED_STATE_DICT` with the distributed-checkpoint API, which reshards.
- **What if the grad norm you log is suspiciously tiny and stable?** Not a checkpoint bug — it's the local-vs-global clip (section 5). Switch to `FSDP.clip_grad_norm_`.
- **What if the loss is huge at step 0 of a *fresh* run (no resume involved)?** Meta-device init without `reset_parameters` (section 7) — uninitialized weights.

The discipline is the same every time: **make it fail small** (the 100-step equivalence test instead of the 5,000-step rerun) and **read the instruments** (the loss the instant you resume, the optimizer moments after load, the per-GPU peak memory, the returned global grad norm). The instruments tell you which of the FSDP bugs you have; the symptom alone never does.

Notice what made this bisection fast: every step *cut the space in half or pinned a quadrant*, and none of them required the expensive run. Step 1 (from-scratch vs resume) cut the six places down to one quadrant — systems/checkpoint — in two minutes. Step 2 (equivalence test) confirmed the quadrant and gave a reproducer. Step 3 (weights vs optimizer) split the checkpoint bug into its two halves and pinned the half. Steps 5–6 cleaned up the residual by reading the *magnitude* of the remaining delta. At no point did we change two things at once, and at no point did we guess. That is the whole method: a training run is a deterministic function with six input families, and a checkpoint resume is that function serialized and rebuilt — so any discrepancy is a serialization gap, and the gap's signature (explode vs drift, large vs small, fresh vs resume-only) names the family. You don't debug FSDP by reading the FSDP source; you debug it by reading the instruments it was kind enough to expose.

## 9. Case studies and real signatures

A few patterns that recur across real FSDP deployments, with the honest caveats on the numbers.

**The all-zeros optimizer after resume (the canonical bug).** This is so common it has a folk name: the "cold resume." The signature is unmistakable once you know it — weights load, loss explodes, `exp_avg_sq` is all zeros. The root cause is almost always calling `optimizer.state_dict()` / `load_state_dict()` directly under FSDP instead of the FSDP-aware `optim_state_dict` helpers (or, in modern code, `get_state_dict` / `set_state_dict`). PyTorch's own FSDP checkpoint tutorial exists largely to prevent this. The fix is mechanical; the cost of *not* knowing it is measured in GPU-days.

**The `transformer_layer_cls` typo (no savings).** Wrapping policies are copy-pasted across projects, and the layer class rarely matches your model out of the box. The result — one root unit, no memory savings, often an OOM that looks identical to "FSDP doesn't help" — is a documentation/config bug, not a framework bug. The unit count (`sum(isinstance(m, FSDP) for m in model.modules())`) is the one-line detector. A correctly-wrapped 32-layer model has ~33 units; one unit means the policy matched nothing.

**bf16 reduce-scatter quality gap.** The recommendation to set `reduce_dtype=torch.float32` while keeping `param_dtype=torch.bfloat16` comes directly from large-model training practice (the mixed-precision recipes in PyTorch's FSDP docs and the broader literature on bf16 training). The magnitude of the gap is model- and run-dependent — we measured ~1.7 points on one ViT finetune, but treat any specific number as illustrative, not universal. The robust claim is directional and provable: reducing in bf16 injects relative error of order $N \cdot 2^{-8}$ per reduction, and fp32 reduction removes it. When in doubt, reduce in fp32 — the bandwidth cost is small.

**The local grad-norm clip.** Reported in many issue trackers as "grad clipping doesn't seem to do anything under FSDP." It doesn't — `torch.nn.utils.clip_grad_norm_` on FSDP shards clips a `~1/√N`-scaled local norm. The fix (`FSDP.clip_grad_norm_` or `accelerate`'s wrapper) is documented but easy to miss when porting a single-GPU loop. The returned global norm is the instrument that proves the fix.

**The world-size-change reshard failure.** A team trained on 8 GPUs, saved with `LOCAL_STATE_DICT` for speed, then tried to resume on 4 GPUs after losing half their allocation. The load threw a shape error: a `LOCAL_STATE_DICT` records each rank's flat shard with *no* metadata for reslicing onto a different number of ranks, so the saved 8-way shards cannot be reassembled into 4-way shards. The fix was to re-save with `SHARDED_STATE_DICT` via the distributed-checkpoint API, which carries the reshape metadata and reshards transparently — you can save on 8 and load on 4, 16, or 1. The takeaway: **if there's any chance you'll change the world size between save and load, you must use a resharding-capable format (`SHARDED_STATE_DICT` through DCP, or `FULL_STATE_DICT`).** `LOCAL_STATE_DICT` is a same-topology-only optimization, and choosing it is a bet that your hardware never changes.

The honest meta-point across all five: **none of these throw a loud, obvious error** at the right place. The cold resume explodes (loud, but misleadingly looks like a learning-rate or data bug). The wrapping typo just quietly saves no memory. The bf16 reduction and the local clip degrade quality silently. The reshard failure errors, but at *load* time, long after the saving choice that doomed it. That silence — or that displacement between the cause and the symptom — is exactly why the resume-equivalence test, the unit count, the per-rank element report, and the logged global grad norm matter: they make the silent failures audible and pin the symptom back to its cause.

## 10. When this is (and isn't) your FSDP bug

A decisive section, because misattributing a symptom to FSDP wastes as much time as the bug itself.

- **A loss that explodes *only on resume*, with weights that load correctly, is an FSDP checkpoint bug** — specifically the optimizer (or RNG) state. If it explodes from scratch too, it is *not* a sharding bug; go look at the learning rate or numerics ([loss spikes and divergence](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs)).
- **No memory savings versus DDP is always a wrapping-policy bug.** If per-GPU memory genuinely dropped by `~1/N` and you still OOM, FSDP is working — your model is just bigger than `N`× your GPU, and you need CPU offload, more GPUs, or gradient checkpointing for the *activation* memory (a different axis — see [out-of-memory debugging](/blog/machine-learning/debugging-training/out-of-memory-debugging)).
- **An eval gap that's stable across the whole run, with no instability, is a precision or config issue, not a checkpoint bug.** Suspect `reduce_dtype`/`buffer_dtype` before you touch the save/load path.
- **A hang (not a crash, not an explosion) under FSDP** is usually a collective mismatch — uneven data across ranks so one rank skips a collective, or a frozen/unused parameter starving a reduce. That's a sync bug, sibling to the DDP `find_unused_parameters` hang in [debugging DDP and multi-GPU](/blog/machine-learning/debugging-training/debugging-ddp-and-multi-gpu).
- **If the overfit-one-batch test passes on a single GPU but the multi-GPU FSDP run won't learn, the bug is in the sharding/systems layer, not the model.** That's the cleanest cut there is: a model that overfits one batch on 1 GPU is not the problem.
- **If your resume-equivalence test passes**, stop blaming the checkpoint. A passing equivalence test rules out the entire checkpoint/resume class — go look elsewhere (data order, eval).

The single most useful FSDP habit: **run the resume-equivalence test as part of CI, on a 100-step toy run, before any long job.** It costs two minutes and it converts the most expensive bug in distributed training into an immediate `FAIL` you catch before you've spent a GPU-day.

## 11. Key takeaways

- **FSDP shards params, grads, and optimizer state (ZeRO-3); DDP replicates all of it.** That's the ~`16P/N` vs ~`16P` per-GPU memory difference — and the source of every FSDP-specific bug.
- **The resume that explodes is a dropped optimizer state, not a weight bug.** Weights load; the sharded Adam moments don't. Reset `v = 0` and the first step divides by `ε`, taking a giant jump (loss `0.3 → 8 → NaN`).
- **Save and load must use the same `state_dict` type.** `SHARDED_STATE_DICT` (fast, reshardable) for big models; `FULL_STATE_DICT` for portability. Thread the optimizer through `get_state_dict`/`set_state_dict`, never `optimizer.state_dict()` alone.
- **The definitive test: a save → load → resume must produce a loss identical to never stopping.** Test it on 100 steps, in CI, before any long run. A *large* delta is the optimizer/weights; a *small* persistent delta is RNG/dataloader position.
- **No memory savings = wrapping policy.** Use `transformer_auto_wrap_policy` with your *exact* block class. Confirm with the unit count (`==1` means nothing wrapped) and `max_memory_allocated` (`~1/N` of DDP).
- **`reduce_dtype=float32` with `param_dtype=bfloat16` is the safe recipe.** bf16 reduction injects ~`N·2⁻⁸` relative error per step; fp32 reduction removes it for a small bandwidth cost.
- **Grad clipping must be sharding-aware.** `torch.nn.utils.clip_grad_norm_` clips a `~1/√N` local norm and silently does nothing; use `FSDP.clip_grad_norm_` (it returns the true global norm — log it).
- **Restore RNG state too**, or your resume test won't pass cleanly even with a perfect optimizer restore — different dropout masks, different data order.
- **Meta-device init needs a real `reset_parameters` after `to_empty`**, or you train from uninitialized garbage (huge/`NaN` loss at step 0 of a fresh run).
- **Bisect before you touch code:** explodes only on resume → checkpoint; no savings → wrapping; stable eval gap → precision; hang → collective sync; tiny stable grad norm → local clip.

## 12. Further reading

- **Rajbhandari, Rajbhandari, Ruwase, He (2020), "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models."** The paper that introduced the optimizer/gradient/parameter sharding stages FSDP implements; the source of the `~16P` state accounting and the per-stage memory math.
- **Micikevicius et al. (2018), "Mixed Precision Training."** The foundational treatment of fp16 underflow, loss scaling, and the fp32 master-weights pattern — the numerics behind section 3.
- **PyTorch FSDP documentation and tutorials** (`torch.distributed.fsdp`): the `MixedPrecision` policy, `transformer_auto_wrap_policy`, `ShardingStrategy`, `FSDP.clip_grad_norm_`, and the distributed-checkpoint (`torch.distributed.checkpoint`) save/load APIs — the canonical reference for every API used here.
- **PyTorch Distributed Checkpoint (DCP) guide:** `dcp.save`/`dcp.load` and `get_state_dict`/`set_state_dict` — the resharding-capable checkpoint path that prevents the cold-resume bug.
- **Hugging Face `accelerate` FSDP guide:** how `accelerate` configures FSDP wrapping, mixed precision, and sharding-aware clipping for you — the easiest correct path for most finetunes.
- **DeepSpeed ZeRO documentation:** the ZeRO-1/2/3 stages, `gradient_clipping`, and checkpoint handling — the conceptual sibling to FSDP and the other production sharding implementation.
- Within this series: [a taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) (the master decision tree), [the training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook) (the capstone checklist), [debugging DDP and multi-GPU](/blog/machine-learning/debugging-training/debugging-ddp-and-multi-gpu), [out-of-memory debugging](/blog/machine-learning/debugging-training/out-of-memory-debugging), [debugging checkpoint and resume](/blog/machine-learning/debugging-training/debugging-checkpoint-and-resume), and [mixed precision debugging, fp16 vs bf16](/blog/machine-learning/debugging-training/mixed-precision-debugging-fp16-vs-bf16).
