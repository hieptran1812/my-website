---
title: "Inside Megatron-LM: How NVIDIA Slices a 462B-Parameter Model Across 6,144 GPUs"
date: "2026-08-10"
publishDate: "2026-08-10"
description: "A source-level tour of Megatron-Core — the conjugate autograd operators behind tensor parallelism, sequence parallelism, 1F1B and interleaved pipeline schedules, ring context parallelism, MoE token dispatchers, the distributed optimizer's flat gradient buffer, and the rank grid that ties all five axes together."
tags:
  [
    "megatron-lm",
    "megatron-core",
    "distributed-training",
    "tensor-parallelism",
    "pipeline-parallelism",
    "context-parallelism",
    "expert-parallelism",
    "mixture-of-experts",
    "fp8",
    "nvidia",
    "llm-training",
    "open-source-library",
  ]
category: "machine-learning"
subcategory: "Open Source Library"
author: "Hiep Tran"
featured: true
readTime: 54
---

The first thing to understand about Megatron-LM is that it is not a model.

People say "we trained it in Megatron" the way they say "we trained it in PyTorch", and that phrasing hides the entire point. PyTorch gives you autograd over a single device's memory. Megatron-LM gives you something much stranger: a set of rules for taking one logical transformer — one weight matrix, one attention head, one token embedding table — and cutting it into pieces that live on different GPUs, while arranging for the arithmetic to come out *bit-for-bit identical to what a single enormous GPU would have computed*.

That constraint is the whole game. Every technique in this post is a different answer to the same question: **given that this tensor no longer fits, where do I cut it, and what is the smallest amount of communication that restores correctness?**

Get that framing right and the codebase stops looking like a pile of NVIDIA-specific flags. `--tensor-model-parallel-size` and `--context-parallel-size` are not two features; they are two axes of the same coordinate system. `--sequence-parallel` is not an optimization bolted onto tensor parallelism; it is the observation that one collective was doing the work of two. The distributed optimizer is not "Megatron's ZeRO"; it is what happens when you notice that gradients already live in one flat buffer and a reduce-scatter is cheaper than an all-reduce.

![The Megatron-Core rank grid: five orthogonal axes over one fixed set of GPUs](/imgs/blogs/megatron-lm-parallelism-techniques-deep-dive-1.webp)

The diagram above is the mental model, and the rest of this article is a tour of it. Megatron-Core takes your GPUs and indexes them along five independent axes — tensor (TP), context (CP), expert (EP), data (DP), and pipeline (PP). Each axis splits a *different* thing: TP splits the matmul, CP splits the sequence, EP splits the experts, DP splits the batch, PP splits the layers. Every configuration decision you will ever make in this framework is a choice about how much of your GPU budget to spend on each axis, and every performance problem you will ever debug is a collective on one of those axes landing somewhere it shouldn't.

## Why Megatron is different

| Assumption | The naive view | What is actually true |
| --- | --- | --- |
| "It's NVIDIA's GPT implementation" | A reference model you fine-tune | A library of parallelism primitives; the GPT model is a ~400-line consumer of them |
| "Tensor parallelism is just splitting matrices" | Split every matmul the same way | Only *column-then-row* pairing avoids a second collective; the pairing is the trick |
| "Sequence parallelism is a separate feature" | Another axis to configure | It is a rewrite of tensor parallelism's collective; same bytes on the wire, less memory |
| "Pipelining wastes GPUs" | The bubble is unavoidable overhead | The bubble is $\frac{p-1}{m}$ and interleaving divides it by $v$ at the cost of more communication |
| "ZeRO and the distributed optimizer are the same" | Interchangeable memory savers | Megatron's is ZeRO stage 1 only, fused with the gradient buffer it already owned |
| "More parallelism is more scale" | Turn everything up | Each axis buys back a *different* scarcity; the wrong axis costs MFU with no memory gain |
| "EP is bounded by DP" | Experts ride on data parallel ranks | Parallelism folding gives expert layers their own grid; EP=64 with DP=8 is legal |

One more distinction worth nailing down before we go further, because the repository has two front doors. **Megatron-Core** (`megatron/core/`) is the library: GPU-optimized transformer building blocks, the five parallelism strategies, mixed precision down to FP8 and FP4, and the distributed checkpointing layer. **Megatron-LM** is the reference harness around it — `pretrain_gpt.py`, `pretrain_mamba.py`, `pretrain_vlm.py`, `train_rl.py`, the argument parser, the SLURM scripts. If you are building your own training framework you depend on `megatron-core` from PyPI and never touch the rest. If you are training a model on someone else's recipe, you live in the scripts and rarely open the core.

Everything below is about the core.

## 1. Tensor parallelism: two operators that are mirrors of each other

**The senior rule of thumb: tensor parallelism is not "split the weights". It is "choose the split so the nonlinearity stays element-wise."**

Start with the transformer MLP, which is two matmuls with a nonlinearity between them:

$$
Y = \text{GeLU}(XA)B
$$

where $X$ has shape $[s, b, h]$, $A$ has shape $[h, 4h]$, and $B$ has shape $[4h, h]$. You have two GPUs and $A$ does not fit. There are two ways to cut it, and only one of them works.

**Cut $A$ by rows.** Then $X$ must be cut by columns to match, each device computes a partial product, and you need an all-reduce *before* the GeLU, because $\text{GeLU}(x_1 + x_2) \neq \text{GeLU}(x_1) + \text{GeLU}(x_2)$. That is one collective in the middle of the MLP, and then a second one after $B$. Two collectives.

**Cut $A$ by columns**, so $A = [A_1, A_2]$. Now device $i$ computes $XA_i$, which is a complete slice of the output columns — not a partial sum. GeLU is element-wise, so $\text{GeLU}(XA_i)$ is exactly the corresponding slice of the true $\text{GeLU}(XA)$. No communication needed. Then cut $B$ by *rows*, so $B = [B_1; B_2]$, and device $i$ computes $\text{GeLU}(XA_i)B_i$ — a partial sum this time, which one all-reduce turns into $Y$.

![Column-then-row: the only pairing where GeLU stays local and the MLP needs one collective](/imgs/blogs/megatron-lm-parallelism-techniques-deep-dive-2.webp)

Column-then-row. One all-reduce per MLP forward pass. That is the entire idea, and it is why `ColumnParallelLinear` and `RowParallelLinear` always appear as a pair in Megatron code — you will never see two column-parallel layers stacked directly, because the second one would need its input gathered first.

If you want this derived from first principles with the arithmetic spelled out, I wrote that up separately in [tensor parallelism from first principles](/blog/machine-learning/distributed-training/tensor-parallelism-megatron). Here I want to look at how Megatron actually implements it, because the implementation contains a genuinely elegant idea that the math doesn't reveal.

### The conjugate pair

The communication above is asymmetric between forward and backward, and Megatron encodes that asymmetry as two custom autograd functions that are exact mirrors of each other. In `megatron/core/tensor_parallel/mappings.py`:

- `_CopyToModelParallelRegion` — **forward: identity. backward: all-reduce.** Conventionally called $f$.
- `_ReduceFromModelParallelRegion` — **forward: all-reduce. backward: identity.** Conventionally called $g$.

![f and g are conjugates: identity one way, all-reduce the other](/imgs/blogs/megatron-lm-parallelism-techniques-deep-dive-3.webp)

Why must it be this way? In the forward pass, $X$ is replicated on every TP rank, so $f$ has nothing to do. But in the backward pass, each rank computes $\frac{\partial L}{\partial X}$ from its own shard of $A$, and those are *partial* gradients that must be summed — so $f$'s backward is an all-reduce. Symmetrically, $g$'s forward sums the partial outputs, and its backward is a pure scatter of an already-correct gradient, which is a no-op.

Written out, the pair is about thirty lines:

```python
# megatron/core/tensor_parallel/mappings.py — reduced to the essentials
import torch

class _CopyToModelParallelRegion(torch.autograd.Function):
    """f: identity forward, all-reduce backward."""

    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        torch.distributed.all_reduce(
            grad_output, group=get_tensor_model_parallel_group()
        )
        return grad_output


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """g: all-reduce forward, identity backward."""

    @staticmethod
    def forward(ctx, input_):
        torch.distributed.all_reduce(
            input_, group=get_tensor_model_parallel_group()
        )
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def copy_to_tensor_model_parallel_region(input_):
    return _CopyToModelParallelRegion.apply(input_)


def reduce_from_tensor_model_parallel_region(input_):
    return _ReduceFromModelParallelRegion.apply(input_)
```

Once you have these two, a tensor-parallel linear layer is almost trivial. Here is the whole idea in raw `torch.distributed`, no Megatron imports, which is worth typing out once because it demystifies the real thing:

```python
import torch
import torch.distributed as dist
import torch.nn as nn


class ColumnParallelLinearMinimal(nn.Module):
    """Shard the output dimension. Input replicated, output sharded."""

    def __init__(self, in_features, out_features, tp_group):
        super().__init__()
        self.tp_group = tp_group
        world = dist.get_world_size(tp_group)
        assert out_features % world == 0, "out_features must divide TP size"
        self.out_per_partition = out_features // world
        # Each rank allocates only its slice: [out/tp, in]
        self.weight = nn.Parameter(
            torch.empty(self.out_per_partition, in_features)
        )
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        # f: identity forward, all-reduce backward
        x = _CopyToModelParallelRegion.apply(x)
        return torch.nn.functional.linear(x, self.weight)  # [.., out/tp]


class RowParallelLinearMinimal(nn.Module):
    """Shard the input dimension. Input sharded, output replicated."""

    def __init__(self, in_features, out_features, tp_group):
        super().__init__()
        self.tp_group = tp_group
        world = dist.get_world_size(tp_group)
        assert in_features % world == 0, "in_features must divide TP size"
        self.in_per_partition = in_features // world
        # Each rank allocates [out, in/tp]
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_per_partition)
        )
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        partial = torch.nn.functional.linear(x, self.weight)
        # g: all-reduce forward, identity backward
        return _ReduceFromModelParallelRegion.apply(partial)


class ParallelMLP(nn.Module):
    def __init__(self, hidden, tp_group):
        super().__init__()
        self.fc1 = ColumnParallelLinearMinimal(hidden, 4 * hidden, tp_group)
        self.fc2 = RowParallelLinearMinimal(4 * hidden, hidden, tp_group)

    def forward(self, x):
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))
```

That is a working tensor-parallel MLP. Megatron's real `ColumnParallelLinear` carries considerably more machinery — the constructor signature alone runs to sixteen keyword arguments:

```python
ColumnParallelLinear(
    input_size, output_size, *, config, init_method,
    bias=True, gather_output=False, stride=1,
    keep_master_weight_for_test=False, skip_bias_add=False,
    skip_weight_param_allocation=False, embedding_activation_buffer=None,
    grad_output_buffer=None, is_expert=False, tp_comm_buffer_name=None,
    disable_grad_reduce=False, tp_group=None, name=None, pg_collection=None,
)
```

Most of those exist for reasons you can now infer. `gather_output=False` is the default because in the column-then-row pairing you *want* the output left sharded — gathering it would be pure waste. `skip_bias_add=True` lets the bias be deferred and fused into a later kernel. `is_expert=True` routes the layer to the expert-tensor-parallel group instead of the dense one, which is what makes parallelism folding possible in §5. And `tp_comm_buffer_name` names a persistent communication buffer used by userbuffer-based overlap, which is what `--tp-comm-overlap` actually turns on.

The important piece of real machinery is `LinearWithGradAccumulationAndAsyncCommunication`, the autograd function that fuses the matmul with the collective. Rather than letting `f`'s backward all-reduce run as a separate op, it issues the gradient all-reduce **asynchronously** and computes the weight gradient underneath it:

```python
# Sketch of the overlap inside LinearWithGradAccumulationAndAsyncCommunication
grad_input = grad_output.matmul(weight)          # dgrad

if allreduce_dgrad:
    handle = torch.distributed.all_reduce(       # launched, not awaited
        grad_input, group=tp_group, async_op=True
    )

# wgrad computed while the dgrad all-reduce is in flight
grad_weight = grad_output.t().matmul(total_input)

if allreduce_dgrad:
    handle.wait()
```

With `sequence_parallel=True`, that `all_reduce` becomes a `reduce_scatter` — we will get to why in §2.

### Attention shards by heads, not by hidden dimension

Multi-head attention has a natural shard unit that the MLP doesn't: the head. Each head computes its own $\text{softmax}(QK^\top/\sqrt{d_k})V$ over its own slice of the hidden dimension, and heads do not interact until the output projection concatenates them.

![Each TP rank owns whole heads, so softmax never crosses a device](/imgs/blogs/megatron-lm-parallelism-techniques-deep-dive-4.webp)

So Megatron gives each TP rank a contiguous set of whole heads. The QKV projection is a `ColumnParallelLinear` (output sharded = heads sharded), attention runs entirely locally, and the output projection is a `RowParallelLinear` whose all-reduce reassembles the result. Same one-collective structure as the MLP, same $f$/$g$ pair.

The constraint this creates is the single most common configuration error in the framework: **`num_attention_heads` must be divisible by `tensor_model_parallel_size`.** A 40-head model cannot run at TP=16. With grouped-query attention there is a second, sharper constraint — `num_query_groups` must also divide TP, and since GQA models often have 8 KV heads, TP > 8 stops being expressible without replicating KV heads.

### The embedding table is the other thing that doesn't fit

At a vocabulary of 131,072 and a hidden size of 12,288, the embedding matrix is 1.6 billion parameters on its own. `VocabParallelEmbedding` shards it along the *vocabulary* dimension, which creates an interesting problem: rank $i$ only holds rows in $[\text{vocab\_start\_index}, \text{vocab\_end\_index})$, and most token IDs in any given batch belong to some other rank.

![Masking out-of-range IDs turns a 131,072-row table into TP shards with one all-reduce](/imgs/blogs/megatron-lm-parallelism-techniques-deep-dive-5.webp)

The solution is to look up everything, then zero out what you don't own, then sum:

```python
# megatron/core/tensor_parallel/layers.py — VocabParallelEmbedding.forward
def forward(self, input_):
    if self.tensor_model_parallel_size > 1:
        # Which IDs does this rank NOT own?
        input_mask = (input_ < self.vocab_start_index) | (
            input_ >= self.vocab_end_index
        )
        masked_input = input_.clone() - self.vocab_start_index
        masked_input[input_mask] = 0          # safe dummy index
    else:
        masked_input = input_

    output_parallel = self.weight[masked_input]   # local gather

    if self.tensor_model_parallel_size > 1:
        output_parallel[input_mask, :] = 0.0      # zero what we don't own

    # g: partial sums -> full embedding vectors
    output = reduce_from_tensor_model_parallel_region(output_parallel)
    return output
```

Every rank produces a tensor that is correct in the rows it owns and exactly zero everywhere else, so summing them across the TP group yields the true embedding. It is the same $g$ operator again.

The output side has a matching trick that is easy to miss and expensive to get wrong. The final logits layer produces a $[s, b, V]$ tensor, and at $V = 131{,}072$ with a 4096-token sequence that tensor is enormous — materializing it unsharded to compute cross-entropy would blow out memory on its own. So Megatron ships `VocabParallelCrossEntropy`, which computes the loss with the logits *left sharded*: each rank computes its local max for numerical stability, one all-reduce takes the global max, each rank computes its local `sum(exp)`, one more all-reduce gives the global denominator, and the target's logit is fetched from whichever rank owns it. Two small all-reduces of shape $[s, b]$ replace one gigantic gather of shape $[s, b, V]$.

### Second-order optimization: the flag that silently disables overlap

**Set `CUDA_DEVICE_MAX_CONNECTIONS=1` or the async overlap above does nothing.** Every Megatron launch script opens with it:

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
```

The reason is subtle. That variable controls how many hardware work queues the CUDA driver multiplexes streams onto. With the default (8), the communication kernel and the wgrad GEMM can be dispatched to different queues, and the GPU's scheduler is free to run the *communication* first — which serializes exactly the thing you were trying to overlap. Pinning to one connection forces issue-order execution, so the async all-reduce launched before the wgrad GEMM genuinely runs underneath it.

I have watched a team spend a week profiling a 175B run that showed no benefit at all from `--tp-comm-overlap`, on a cluster where the launch wrapper had stripped the environment. The flag was on. The overlap was not happening. One `export` recovered 9% of end-to-end throughput.

## 2. Sequence parallelism: the same bytes, a fifth of the activation memory

**The senior rule of thumb: sequence parallelism is free. If you have tensor parallelism on, you should have sequence parallelism on.**

Look again at what tensor parallelism leaves behind. Inside the attention block and inside the MLP, activations are sharded across TP ranks. But between them — at the LayerNorm, at the dropout, at the residual add — the tensor has been all-reduced back to full width, and every TP rank holds an identical $[s, b, h]$ copy. That replicated region is pure waste: TP=8 means eight copies of the same LayerNorm activations.

Megatron's fix, from Korthikanti et al. (2022), starts from an identity every distributed-systems engineer knows:

$$
\text{all-reduce} = \text{reduce-scatter} + \text{all-gather}
$$

An all-reduce is *already* implemented as those two phases internally. So instead of paying for both and landing on a replicated tensor, split them apart and put useful work in the middle. Replace $g$ with a reduce-scatter along the sequence dimension, do the LayerNorm and dropout on the resulting $[s/\text{tp}, b, h]$ shard, and replace $f$ with an all-gather that restores full sequence just in time for the next attention or MLP block.

![One all-reduce becomes reduce-scatter plus all-gather — same wire volume, sharded in between](/imgs/blogs/megatron-lm-parallelism-techniques-deep-dive-6.webp)

The wire volume is **identical**. You were already moving those bytes. What changes is that the region between the two collectives is now sharded by a factor of TP.

In `mappings.py` this is a third and fourth pair of conjugate operators, and they are structured exactly like $f$ and $g$:

| Operator | Forward | Backward |
| --- | --- | --- |
| `_ScatterToSequenceParallelRegion` | split along dim 0 | all-gather along dim 0 |
| `_GatherFromSequenceParallelRegion` | all-gather along dim 0 | reduce-scatter along dim 0 |
| `_ReduceScatterToSequenceParallelRegion` | reduce-scatter along dim 0 | all-gather along dim 0 |

Note that these operate on **dim 0**, the sequence dimension, whereas the tensor-parallel ones (`_ScatterToModelParallelRegion`, `_GatherFromModelParallelRegion`) operate on the **last** dimension, the hidden one. That is the entire structural difference between the two families, and it is worth memorizing because a surprising number of shape bugs come from confusing them.

The layer code reflects it directly. In `RowParallelLinear.forward`, the choice is one branch:

```python
if self.explicit_expert_comm:
    output_ = output_parallel
elif self.sequence_parallel:
    output_ = reduce_scatter_to_sequence_parallel_region(output_parallel)
else:
    output_ = reduce_from_tensor_model_parallel_region(output_parallel)
```

### What it actually saves

Per transformer layer, the activation memory that tensor parallelism alone leaves replicated is roughly

$$
s \cdot b \cdot h \cdot (10 + \frac{24}{t})
$$

bytes in the classic accounting, where $t$ is the TP size — the $10$ is the replicated part (LayerNorms, dropout masks, residuals) and the $24/t$ is the sharded part. Sequence parallelism moves that $10$ under the divisor too:

$$
s \cdot b \cdot h \cdot \frac{34}{t}
$$

At TP=8 that is a 2.6× reduction in per-layer activation memory, for zero additional communication.

![Per-layer activation memory: baseline, +TP, +TP&SP, +selective recompute](/imgs/blogs/megatron-lm-parallelism-techniques-deep-dive-7.webp)

Stack selective recomputation on top (§7) and the attention-score term goes away as well. The four bars above are the four regimes, and the reason the third bar is the big drop is that it is the first one to touch the replicated tail.

### Second-order optimization: sequence parallelism changes your divisibility constraints

`--sequence-parallel` requires `seq_length` to be divisible by `tensor_model_parallel_size`, and with context parallelism also on, by `tensor_model_parallel_size × context_parallel_size × 2` (the factor of 2 comes from the causal load-balancing scheme in §4). A sequence length of 4096 at TP=8, CP=4 needs to divide by 64 — fine. A sequence length of 8192 at TP=6 does not divide at all, and Megatron will refuse at startup rather than silently pad. That startup assertion has saved more runs than it has annoyed people.

## 3. Pipeline parallelism: the bubble and the two ways to shrink it

**The senior rule of thumb: pipeline parallelism is the axis you add when you have run out of intra-node bandwidth. It buys memory, not speed, and it charges you a bubble.**

Split the model's *layers* across GPUs — stage 0 gets layers 0–7, stage 1 gets 8–15, and so on. Now a microbatch flows forward through the stages and its gradient flows backward. The problem is immediate and structural: while stage 0 is computing the forward pass for microbatch 1, stages 1 through $p-1$ have nothing to do. They are waiting for data that does not exist yet.

That idle time is the **pipeline bubble**, and with $p$ stages and $m$ microbatches its fraction of total time is

$$
\text{bubble fraction} = \frac{p - 1}{m}
$$

At $p = 16$ and $m = 32$, that is 47% of your cluster idle. At $m = 256$, it is 5.9%. This is why global batch sizes in large runs are enormous — the 175B reference config uses `--global-batch-size 1536` with `--micro-batch-size 1`, giving 1536 microbatches to amortize a 16-stage pipeline.

![GPipe, 1F1B and interleaved on the same four stages and eight microbatches](/imgs/blogs/megatron-lm-parallelism-techniques-deep-dive-8.webp)

### From GPipe to 1F1B

The naive schedule (GPipe) runs all $m$ forward passes, then all $m$ backward passes. It works, but it holds the activations of *every* microbatch simultaneously — peak activation memory scales with $m$, which is exactly the number you wanted to make large.

**1F1B** (one-forward-one-backward) fixes the memory without touching the bubble. After a warmup phase, each stage alternates: one forward, one backward, forever. The key consequence is that a microbatch's activations are freed as soon as its backward runs, so peak memory scales with $p$ (the number of in-flight microbatches) rather than $m$.

The warmup count is the crux, and in `megatron/core/pipeline_parallel/schedules.py` it is one line:

```python
num_warmup_microbatches = (
    p2p_communicator.total_stages - p2p_communicator.current_stage - 1
)
num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
num_microbatches_remaining = num_microbatches - num_warmup_microbatches
```

Read it as: *stage $r$ must run $p - r - 1$ forward passes before its first backward can possibly arrive.* Stage 0 in a 16-stage pipeline warms up with 15 forwards and therefore holds 15 microbatches' worth of activations; the last stage warms up with zero and holds one. **The first stage is always your memory bottleneck in 1F1B**, which is why Megatron supports uneven layer distribution via `--decoder-first-pipeline-num-layers` — you give stage 0 fewer layers precisely because it holds more activations.

Reduced to its skeleton, the schedule is a three-phase loop:

```python
# forward_backward_pipelining_without_interleaving, structurally
input_tensors, output_tensors = [], []

# --- warmup: forward only ---
for i in range(num_warmup_microbatches):
    input_tensor = recv_forward(tensor_shape, config)
    output_tensor = forward_step(forward_step_func, data_iterator, model,
                                 num_microbatches, input_tensor, forward_data_store)
    send_forward(output_tensor, config)
    input_tensors.append(input_tensor)
    output_tensors.append(output_tensor)

# --- steady state: 1F1B ---
for i in range(num_microbatches_remaining):
    output_tensor = forward_step(...)
    output_tensor_grad = send_forward_recv_backward(output_tensor, ...)
    send_forward(output_tensor, config)

    input_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)
    input_tensor_grad = backward_step(input_tensor, output_tensor,
                                      output_tensor_grad, model_type, config)
    input_tensor = send_backward_recv_forward(input_tensor_grad, ...)

# --- cooldown: backward only ---
for i in range(num_warmup_microbatches):
    input_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)
    output_tensor_grad = recv_backward(tensor_shape, config)
    input_tensor_grad = backward_step(...)
    send_backward(input_tensor_grad, config)
```

Notice `send_forward_recv_backward` and `send_backward_recv_forward`. Those fused primitives are not cosmetic — they issue the send and the receive as a single batched P2P operation, which halves the number of synchronization points in the steady state and lets NCCL overlap the two directions on the link.

### Interleaving: dividing the bubble by $v$

The bubble is $(p-1)/m$ because each stage must wait $p-1$ microbatch-times for the pipeline to fill. The insight behind the **interleaved schedule** (virtual pipeline parallelism) is that you can make each "microbatch-time" smaller by giving each physical stage several *non-contiguous* chunks of the model.

With $v$ virtual chunks per rank, stage 1 might own layers 4–7 (chunk 0) *and* layers 20–23 (chunk 1). The pipeline now has $p \cdot v$ virtual stages, each holding $1/v$ as many layers, so each stage's compute time drops by $v$ and the fill time drops with it:

$$
\text{bubble fraction} = \frac{p - 1}{m \cdot v}
$$

The cost is communication volume: activations now cross the pipeline $v$ times as often. Interleaving is a bandwidth-for-idle-time trade, and it is only worth it when your pipeline links are fast relative to your bubble.

![The virtual-pipeline chunk table decouples which layers run from which microbatch runs](/imgs/blogs/megatron-lm-parallelism-techniques-deep-dive-9.webp)

Implementation-wise, this is where the scheduler stops being a counter and becomes a table lookup. `forward_backward_pipelining_with_interleaving` precomputes `model_chunk_id_table` and `microbatch_id_table` via `get_schedule_table()`, then indexes them by `virtual_microbatch_id`. The warmup formula grows accordingly:

```python
num_warmup_microbatches = (
    (pipeline_parallel_size - pipeline_parallel_rank - 1) * 2
    + (num_model_chunks - 1) * microbatch_group_size_per_vp_stage
)
```

That second term is the one that bites. More chunks means more warmup microbatches means more live activations on stage 0 — **turning on interleaving can OOM a job that was fitting comfortably**, and the memory goes up in the warmup phase, not the steady state. If you enable `--num-layers-per-virtual-pipeline-stage` and immediately OOM, do not go looking at your activation checkpointing config; look at this line.

Dispatch between the three schedules happens in `get_forward_backward_func()`, and it is exactly as simple as you'd hope: if `pp_size > 1` and a virtual pipeline size is set, use interleaved; else if `pp_size > 1`, use 1F1B; else use `forward_backward_no_pipelining`.

### Second-order optimization: `deallocate_output_tensor`

There is a small function in `schedules.py` called `deallocate_output_tensor` that does something that looks illegal. After a stage sends its output tensor downstream, that tensor's *data* is no longer needed locally — but autograd still holds a reference to it as a graph node. The function replaces the tensor's storage with a one-element allocation while keeping the graph node alive, then `custom_backward` supplies the gradient directly rather than letting autograd try to reconstruct it.

For a 16-stage pipeline at $[4096, 1, 12288]$ in bf16, each output tensor is 100 MB, and stage 0 holds 15 of them during warmup. Deallocating the payloads recovers about 1.5 GB per GPU. It is one of those optimizations that reads as a hack and is in fact the correct thing to do.

## 4. Context parallelism: when the sequence itself is the problem

**The senior rule of thumb: context parallelism is the axis for long sequences, and its hard part is not the attention math — it is the load balancing.**

Attention memory scales as $O(s^2)$ in the naive formulation and $O(s)$ with FlashAttention, but either way, at $s = 128{,}000$ the activations for a single sequence exceed one GPU. Tensor parallelism doesn't help — it splits heads and hidden dimensions, not sequence positions. So Megatron adds a fifth axis that splits the sequence itself.

The difficulty is that attention is the one operation that is *not* local in the sequence dimension. Query at position $i$ needs keys and values at every position $j \leq i$, and those live on other ranks.

The answer is ring attention. Each CP rank keeps its queries fixed and passes its K/V chunk around the ring; after $\text{cp\_size}$ steps every query has seen every key. Crucially, the send of chunk $k+1$ is issued *before* the attention computation on chunk $k$ begins, so the communication hides entirely behind the compute:

<figure class="blog-anim">
<svg viewBox="0 0 700 420" role="img" aria-label="A KV chunk rotates around a four-rank context-parallel ring while each rank's attention compute bar fills, showing the send overlapping the computation" style="width:100%;height:auto;max-width:720px">
<style>
.m1-node{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.m1-link{stroke:var(--border,#d1d5db);stroke-width:2;stroke-dasharray:6 6;fill:none}
.m1-lbl{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.m1-sub{font:13px ui-monospace,SFMono-Regular,monospace;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.m1-mid{font:13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.m1-track{fill:var(--border,#d1d5db)}
.m1-bar{fill:var(--accent,#6366f1);transform-box:fill-box;transform-origin:left center}
.m1-tokc{fill:var(--accent,#6366f1)}
.m1-tokt{font:600 11px ui-sans-serif,system-ui;fill:var(--background,#fff);text-anchor:middle}
@keyframes m1-orbit{0%{transform:translate(0,0)}25%{transform:translate(210px,120px)}50%{transform:translate(0,240px)}75%{transform:translate(-210px,120px)}100%{transform:translate(0,0)}}
@keyframes m1-fill{0%{transform:scaleX(0)}20%{transform:scaleX(1)}25%{transform:scaleX(1)}25.5%{transform:scaleX(0)}100%{transform:scaleX(0)}}
.m1-orbit{animation:m1-orbit 12s linear infinite}
.m1-bar{animation:m1-fill 12s linear infinite}
.m1-b1{animation-delay:-9s}
.m1-b2{animation-delay:-6s}
.m1-b3{animation-delay:-3s}
@media (prefers-reduced-motion:reduce){.m1-orbit,.m1-bar{animation:none}.m1-orbit{transform:translate(105px,180px)}.m1-bar{transform:scaleX(0)}.m1-b0{transform:scaleX(1)}.m1-b1{transform:scaleX(.55)}}
</style>
<path class="m1-link" d="M350 80 L560 200 L350 320 L140 200 Z"/>
<rect class="m1-node" x="255" y="25" width="190" height="110" rx="10"/>
<rect class="m1-node" x="465" y="145" width="190" height="110" rx="10"/>
<rect class="m1-node" x="255" y="265" width="190" height="110" rx="10"/>
<rect class="m1-node" x="45" y="145" width="190" height="110" rx="10"/>
<text class="m1-lbl" x="350" y="55">rank 0</text>
<text class="m1-sub" x="350" y="76">Q0 fixed</text>
<text class="m1-lbl" x="560" y="175">rank 1</text>
<text class="m1-sub" x="560" y="196">Q1 fixed</text>
<text class="m1-lbl" x="350" y="295">rank 2</text>
<text class="m1-sub" x="350" y="316">Q2 fixed</text>
<text class="m1-lbl" x="140" y="175">rank 3</text>
<text class="m1-sub" x="140" y="196">Q3 fixed</text>
<rect class="m1-track" x="280" y="82" width="140" height="12" rx="6"/>
<rect class="m1-track" x="490" y="202" width="140" height="12" rx="6"/>
<rect class="m1-track" x="280" y="322" width="140" height="12" rx="6"/>
<rect class="m1-track" x="70" y="202" width="140" height="12" rx="6"/>
<rect class="m1-bar m1-b0" x="280" y="82" width="140" height="12" rx="6"/>
<rect class="m1-bar m1-b1" x="490" y="202" width="140" height="12" rx="6"/>
<rect class="m1-bar m1-b2" x="280" y="322" width="140" height="12" rx="6"/>
<rect class="m1-bar m1-b3" x="70" y="202" width="140" height="12" rx="6"/>
<text class="m1-mid" x="350" y="190">attn compute bar fills while</text>
<text class="m1-mid" x="350" y="212">the next K,V chunk is in flight</text>
<g class="m1-orbit">
<circle class="m1-tokc" cx="350" cy="114" r="15"/>
<text class="m1-tokt" x="350" y="118">K,V</text>
</g>
<text class="m1-sub" x="350" y="405">context_parallel_size = 4</text>
</svg>
<figcaption>One full rotation of the CP ring: each step sends a K,V chunk to the neighbouring rank while the local attention for the previous chunk is still running, so the communication hides behind the compute.</figcaption>
</figure>

```python
# Ring attention step, structurally — one rank's view
def ring_attention(q, k, v, cp_group, causal=True):
    cp_size = torch.distributed.get_world_size(cp_group)
    rank = torch.distributed.get_rank(cp_group)
    out, lse = None, None                 # running output + logsumexp
    k_cur, v_cur = k, v

    for step in range(cp_size):
        if step + 1 < cp_size:
            # launch the rotation BEFORE computing on what we have
            k_next, v_next = torch.empty_like(k_cur), torch.empty_like(v_cur)
            reqs = ring_isend_irecv(k_cur, v_cur, k_next, v_next, cp_group)

        # which source chunk are we holding this step?
        src = (rank - step) % cp_size
        block_causal = causal and (src > rank)     # entirely masked out
        if not block_causal:
            out, lse = flash_attn_update(
                out, lse, q, k_cur, v_cur,
                causal=causal and (src == rank),
            )

        if step + 1 < cp_size:
            for r in reqs:
                r.wait()
            k_cur, v_cur = k_next, v_next

    return out
```

The `flash_attn_update` step is an online-softmax merge: each partial attention result carries its own logsumexp, and merging two partials is a numerically stable rescale. This is the same machinery FlashAttention uses to tile attention within a GPU, reused to tile it *across* GPUs.

### The causal mask makes naive chunking catastrophically unbalanced

Here is the part that trips people up. With a causal mask, query position $i$ attends to $i+1$ keys. If you split a sequence into contiguous chunks and give chunk $r$ to rank $r$, then rank 0 holds the earliest positions — which attend to almost nothing — and rank $p-1$ holds the latest positions, which attend to everything.

![Contiguous chunking gives rank 0 one-eighth of rank 7's work; striping equalizes it](/imgs/blogs/megatron-lm-parallelism-techniques-deep-dive-10.webp)

At CP=8 the last rank does roughly 8× the work of the first, and because every ring step is a synchronization point, **the whole ring runs at the speed of the slowest rank**. You have bought 8 GPUs of memory and roughly 1 GPU of throughput.

Megatron's fix is to split the sequence into $2 \cdot \text{cp\_size}$ chunks and give rank $r$ both chunk $r$ and chunk $2 \cdot \text{cp\_size} - 1 - r$. Rank 0 gets the very first chunk (nearly no work) and the very last chunk (nearly full work); rank 7 gets two middling chunks. Every rank ends up with the same number of unmasked blocks. That factor of 2 is exactly why `seq_length` must be divisible by `2 × cp_size`.

More recently the framework added **dynamic context parallelism**, which addresses the other imbalance: when a batch contains sequences of wildly different lengths, static chunking wastes work on padding. NVIDIA reports up to a 1.48× speedup for variable-length sequence training from that path.

### Second-order optimization: CP and the KV cache during inference

Context parallelism is a training-time axis, and the intuition does not transfer cleanly to inference. During generation there is one query position, so there is nothing to shard along the sequence dimension for the query — the KV cache is what's large, and sharding *it* across ranks means every decode step needs a ring pass. For serving long contexts, tensor parallelism plus a paged KV cache is almost always the better structure. I go through that in [multi-node LLM serving](/blog/machine-learning/model-serving/multi-node-llm-serving-100b-plus).

## 5. Expert parallelism: routing is a communication pattern

**The senior rule of thumb: for MoE layers, prefer expert parallelism over tensor parallelism. Splitting an expert's GEMM makes it small and inefficient; giving each GPU whole experts keeps the GEMM fat.**

A Mixture-of-Experts layer replaces the dense MLP with $E$ expert MLPs and a router that sends each token to its top-$k$ experts. The parameter count scales with $E$ while the FLOPs per token stay roughly constant — that is the entire appeal, and it is also what makes the layer a communication problem rather than a compute problem.

### The router, and the load-balancing problem it creates

```python
# megatron/core/transformer/moe/router.py — structurally
def routing(self, logits):
    # Router logits in fp32/fp64 — NOT bf16. See below.
    logits = logits.float()

    if self.config.moe_router_score_function == "sigmoid":
        scores = torch.sigmoid(logits)
    else:
        scores = torch.softmax(logits, dim=-1)

    # aux-loss-free load balancing: a learned per-expert bias that
    # steers routing WITHOUT contributing a gradient to the loss
    if self.config.moe_router_enable_expert_bias:
        scores_for_choice = scores + self.expert_bias
    else:
        scores_for_choice = scores

    probs, indices = torch.topk(scores_for_choice, k=self.topk, dim=-1)
    probs = scores.gather(-1, indices)      # weight with UNBIASED scores
    return probs, indices


@torch.no_grad()
def update_expert_bias(self, tokens_per_expert):
    """Called once per step. Pure bookkeeping, no autograd."""
    ideal = tokens_per_expert.mean()
    err = ideal - tokens_per_expert
    self.expert_bias += self.config.moe_router_bias_update_rate * torch.sign(err)
```

Two details in there are worth more than they look.

**Router precision.** `--moe-router-dtype` defaults to fp32 and the docs are explicit that fp32 or fp64 beats bf16. The reason is that expert outputs get multiplied by router scores, so score error propagates directly into the layer output — and with 256 experts, a softmax over 256 bf16 logits has genuinely poor resolution near the top-$k$ boundary. A run that routes slightly differently every step because of bf16 rounding will show up as unstable expert-utilization metrics and a loss curve that is noisier than it should be.

**Bias for balance, unbiased scores for weighting.** The bias steers *which* experts get chosen; the actual combination weights come from the unbiased scores. This is the DeepSeek-V3 aux-loss-free scheme, enabled with `--moe-router-enable-expert-bias --moe-router-bias-update-rate 1e-3`. It replaces the classic auxiliary loss (`--moe-router-load-balancing-type aux_loss`), whose gradient perturbs the model in service of a balance objective the model doesn't actually care about. The bias approach achieves balance with zero interference in the loss.

### Four dispatchers, three communication patterns

Once the router has decided, tokens have to physically reach their experts. Megatron offers several strategies via `--moe-token-dispatcher-type`:

![allgather, alltoall and flex/DeepEP are three answers to the same routing problem](/imgs/blogs/megatron-lm-parallelism-techniques-deep-dive-11.webp)

| Dispatcher | Communication | Volume | Use when |
| --- | --- | --- | --- |
| `allgather` | Every rank gathers all tokens | $O(\text{tokens} \times \text{EP})$ | TP-only setups, small EP, large top-$k$ |
| `alltoall` | Permutation exchange | $O(\text{tokens} \times k)$ | Standard EP > 1 |
| `flex` + `deepep` | Deduplicated cross-node, fused intra/inter | Much lower cross-node | Cross-node EP, fine-grained MoE (DeepSeek-V3 style) |
| `flex` + `hybridep` | NVIDIA's TMA/IBGDA path | Lowest on NVL domains | GB200 NVL72, multi-node NVLink |

The `allgather` default is the conservative choice and the wrong one for most modern configs. Its volume scales with EP size, so it is fine at EP=2 and disastrous at EP=64. The `alltoall` dispatcher moves each token exactly $k$ times regardless of EP, which is why it is the standard recommendation the moment EP > 1.

The `flex` dispatcher with the DeepEP backend addresses the specific pathology of fine-grained MoE: with 256 experts and top-8, a token's 8 targets may live on 8 different nodes, and a naive all-to-all sends the same token across the node boundary up to 8 times. DeepEP sends it across *once* per destination node and fans out with intra-node NVLink.

### Grouped GEMM: the other half of the story

Having gathered tokens for its experts, a rank must run $E_{\text{local}}$ separate MLPs of differing token counts. Doing that as a Python loop over `SequentialMLP` launches hundreds of small kernels and leaves the GPU mostly idle. `--moe-grouped-gemm` swaps in `TEGroupedMLP`, which batches all local experts into a single grouped-GEMM kernel:

```bash
--moe-grouped-gemm \
--moe-permute-fusion \        # fuse the permute/unpermute around the GEMM
--moe-router-fusion           # fuse router projection + topk + softmax
```

On a fine-grained MoE with 8 local experts, I have seen this alone move MoE-layer time by more than 2×, entirely from kernel-launch overhead and occupancy rather than from any change in FLOPs.

### Parallelism folding: EP is no longer bounded by DP

Here is the structural idea that I think is Megatron's most underrated contribution.

Classically, expert parallelism rides on the data-parallel group — experts are distributed across DP ranks, which imposes $\text{EP} \leq \text{DP}$. For a model where you want EP=64 but your DP is only 8 (because TP, CP and PP consumed the rest of the grid), that constraint is fatal.

Megatron decouples the two by giving attention layers and expert layers **different rank grids over the same GPUs**:

```
Attention layers:  TP × CP × DP  × PP
MoE layers:        ETP × EP × EDP × PP
```

![Fold the grid and the EP ≤ DP constraint disappears](/imgs/blogs/megatron-lm-parallelism-techniques-deep-dive-12.webp)

The same 256 GPUs can be indexed as `TP=4 × CP=2 × DP=8 × PP=4` for attention and `ETP=1 × EP=64 × EDP=1 × PP=4` for the experts. This is what `is_expert=True` on a linear layer selects, and what `--expert-tensor-parallel-size` configures independently of `--tensor-model-parallel-size`.

The guideline that comes with it: **keep $\text{EP} \times \text{TP}$ inside a single node.** Both are communication-intensive per token, and both want NVLink rather than InfiniBand. For Mixtral 8x7B, `EP8×TP1` beats `EP4×TP2` — same 8 GPUs, but the first keeps whole experts local and the second splits every expert GEMM in half for no memory benefit.

### Capacity factor: the memory-versus-quality dial

There is one more MoE decision that has no good default, because it is a genuine trade rather than an optimization.

Expert batch sizes are *data-dependent* — the router decides at runtime how many tokens each expert receives, and that number varies per step. That is awkward for a system that wants static shapes and predictable memory. `--moe-expert-capacity-factor` resolves it by capping each expert at

$$
\text{capacity} = \text{capacity\_factor} \times \frac{\text{tokens} \times k}{E}
$$

tokens. Anything above the cap is **dropped** — the token simply does not get processed by that expert, and its contribution comes only from its other top-$k$ choices (or from the shared expert, if there is one). `--moe-token-drop-policy` chooses *which* tokens get dropped: `probs` discards the lowest-confidence assignments, `position` discards by sequence position.

The dial runs between two failure modes. A capacity factor of 1.0 means perfect balance is assumed and any imbalance drops tokens — cheap and shape-stable, but you are discarding computation the router asked for, and early in training the router is very imbalanced. Dropless operation (no capacity factor set) processes everything, at the cost of dynamic shapes and a memory high-water mark set by the worst-case expert.

Combined with `--moe-pad-expert-input-to-capacity`, which pads every expert to exactly the capacity, you get fully static shapes — which is what makes the FP8 alignment in `--moe-router-padding-for-fp8` possible, since FP8 GEMMs want tile-aligned token counts.

My default is dropless with aux-loss-free bias balancing, moving to a capacity factor only when the memory high-water mark actually becomes the binding constraint. Dropping tokens to save memory is a quality decision disguised as a performance flag, and it deserves to be made deliberately rather than inherited from a config file.

### Second-order optimization: overlap the shared expert

Models like DeepSeek-V3 pair routed experts with an always-on shared expert. Since the shared expert processes *every* token, it does not participate in the all-to-all at all — which means it can run *during* the dispatch. `--moe-shared-expert-overlap` does exactly that, hiding a meaningful chunk of the shared expert's compute behind the routed experts' communication. Combined with `--overlap-moe-expert-parallel-comm --delay-wgrad-compute`, the EP all-to-all can be almost entirely hidden on a well-configured node.

## 6. The distributed optimizer: ZeRO-1, fused into a buffer that already existed

**The senior rule of thumb: turn on `--use-distributed-optimizer` first, before any model parallelism. It is the cheapest memory you will ever buy.**

For a model with $P$ parameters trained in mixed precision with Adam, the persistent state is roughly:

| Item | Bytes per parameter |
| --- | --- |
| bf16 parameters | 2 |
| bf16 gradients | 2 |
| fp32 master parameters | 4 |
| fp32 Adam `exp_avg` | 4 |
| fp32 Adam `exp_avg_sq` | 4 |
| **Total** | **16** |

A 7B model needs 112 GB of state before a single activation is allocated. The last 12 bytes — master params plus the two moments — are *identical work on every data-parallel rank*, which makes them pure redundancy at DP > 1.

Megatron's `DistributedOptimizer` shards those 12 bytes across the DP group. This is ZeRO stage 1: optimizer state is partitioned, parameters and gradients stay replicated. What makes Megatron's version distinctive is that it is built on top of a data structure the framework already had for a different reason.

### One flat buffer, sliced twice

`_ParamAndGradBuffer` packs all parameters of a given dtype into one contiguous allocation, with each parameter's `.data` remapped to a *view* into that buffer. Parameters are packed in **reverse order** — the order backward will touch them — and the buffer is then divided into buckets of roughly `bucket_size` elements.

![One flat buffer, sliced into buckets, then reduce-scattered into per-rank shards](/imgs/blogs/megatron-lm-parallelism-techniques-deep-dive-13.webp)

The consequence is that a gradient reduce-scatter is a single collective over one contiguous tensor, and the resulting shard *is* the slice of optimizer state this rank owns. There is no gather-and-scatter step, no per-parameter bookkeeping at collective time.

The bookkeeping happens once, at construction, and it is the fiddliest code in the file. `_build_model_gbuf_param_range_map` computes four ranges for every parameter:

| Range | Meaning |
| --- | --- |
| `gbuf_world` | the param's range within the entire grad buffer |
| `gbuf_local` | its range within this DP rank's local view of the buffer |
| `gbuf_world_in_bucket` | its range within the relevant bucket's buffer |
| `param` | its range within itself — i.e. which slice of the param this rank owns |

That last one exists because a parameter can *straddle* a shard boundary. A 4096×4096 weight in a buffer sharded 8 ways may have its first 30% on rank 3 and the rest on rank 4, and both ranks own a partial optimizer state for it. Nothing in the API surface exposes this, and everything downstream — checkpointing, gradient clipping, `load_state_dict` — has to respect it.

### Overlap: buckets fire as they fill

The second reason for the bucket structure is timing. Backward proceeds layer by layer from the output, and gradients become available progressively. Waiting for the entire backward pass before starting DP communication leaves the network idle for the whole pass and then the GPU idle for the whole reduce.

Instead, `_ParamAndGradBucketGroup` registers a hook per parameter:

<figure class="blog-anim">
<svg viewBox="0 0 720 300" role="img" aria-label="A backward-pass sweep moves right to left filling gradient slots; each bucket launches its reduce-scatter as soon as its last slot fills, so all three transfers are retired by the end of the pass" style="width:100%;height:auto;max-width:760px">
<style>
.m2-hd{font:13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.m2-lbl{font:600 12px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.m2-mono{font:12px ui-monospace,SFMono-Regular,monospace;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.m2-slot{fill:none;stroke:var(--border,#d1d5db);stroke-width:1.5}
.m2-full{fill:var(--accent,#6366f1);opacity:.85}
.m2-bkt{fill:none;stroke:var(--text-secondary,#6b7280);stroke-width:1.5;stroke-dasharray:5 4}
.m2-shroud{fill:var(--background,#fff)}
.m2-sweep{fill:var(--accent,#6366f1)}
.m2-lane{fill:none;stroke:var(--border,#d1d5db);stroke-width:1.5;stroke-dasharray:4 4}
.m2-tx{fill:var(--accent,#6366f1);opacity:.85;transform-box:fill-box;transform-origin:left center}
.m2-txt{font:600 12px ui-sans-serif,system-ui;fill:var(--background,#fff);text-anchor:middle}
@keyframes m2-sweep{0%{transform:translateX(0)}70%{transform:translateX(-644px)}100%{transform:translateX(-644px)}}
@keyframes m2-tx2{0%,21%{opacity:0;transform:scaleX(0)}30%,100%{opacity:.85;transform:scaleX(1)}}
@keyframes m2-tx1{0%,45%{opacity:0;transform:scaleX(0)}54%,100%{opacity:.85;transform:scaleX(1)}}
@keyframes m2-tx0{0%,69%{opacity:0;transform:scaleX(0)}78%,100%{opacity:.85;transform:scaleX(1)}}
.m2-mv{animation:m2-sweep 14s ease-in-out infinite}
.m2-tx2{animation:m2-tx2 14s ease-out infinite}
.m2-tx1{animation:m2-tx1 14s ease-out infinite}
.m2-tx0{animation:m2-tx0 14s ease-out infinite}
@media (prefers-reduced-motion:reduce){.m2-mv,.m2-tx0,.m2-tx1,.m2-tx2{animation:none}.m2-mv{transform:translateX(-370px)}.m2-tx2{opacity:.85;transform:scaleX(.55)}.m2-tx1{opacity:0}.m2-tx0{opacity:0}}
</style>
<text class="m2-hd" x="40" y="20">_ParamAndGradBuffer: one flat allocation, three buckets</text>
<text class="m2-hd" x="686" y="42" text-anchor="end">backward pass sweeps right to left</text>
<rect class="m2-full" x="48" y="70" width="42" height="44" rx="4"/>
<rect class="m2-full" x="96" y="70" width="42" height="44" rx="4"/>
<rect class="m2-full" x="144" y="70" width="42" height="44" rx="4"/>
<rect class="m2-full" x="192" y="70" width="42" height="44" rx="4"/>
<rect class="m2-full" x="270" y="70" width="42" height="44" rx="4"/>
<rect class="m2-full" x="318" y="70" width="42" height="44" rx="4"/>
<rect class="m2-full" x="366" y="70" width="42" height="44" rx="4"/>
<rect class="m2-full" x="414" y="70" width="42" height="44" rx="4"/>
<rect class="m2-full" x="492" y="70" width="42" height="44" rx="4"/>
<rect class="m2-full" x="540" y="70" width="42" height="44" rx="4"/>
<rect class="m2-full" x="588" y="70" width="42" height="44" rx="4"/>
<rect class="m2-full" x="636" y="70" width="42" height="44" rx="4"/>
<rect class="m2-shroud m2-mv" x="9" y="50" width="680" height="88"/>
<rect class="m2-slot" x="48" y="70" width="42" height="44" rx="4"/>
<rect class="m2-slot" x="96" y="70" width="42" height="44" rx="4"/>
<rect class="m2-slot" x="144" y="70" width="42" height="44" rx="4"/>
<rect class="m2-slot" x="192" y="70" width="42" height="44" rx="4"/>
<rect class="m2-slot" x="270" y="70" width="42" height="44" rx="4"/>
<rect class="m2-slot" x="318" y="70" width="42" height="44" rx="4"/>
<rect class="m2-slot" x="366" y="70" width="42" height="44" rx="4"/>
<rect class="m2-slot" x="414" y="70" width="42" height="44" rx="4"/>
<rect class="m2-slot" x="492" y="70" width="42" height="44" rx="4"/>
<rect class="m2-slot" x="540" y="70" width="42" height="44" rx="4"/>
<rect class="m2-slot" x="588" y="70" width="42" height="44" rx="4"/>
<rect class="m2-slot" x="636" y="70" width="42" height="44" rx="4"/>
<rect class="m2-bkt" x="40" y="58" width="202" height="68" rx="6"/>
<rect class="m2-bkt" x="262" y="58" width="202" height="68" rx="6"/>
<rect class="m2-bkt" x="484" y="58" width="202" height="68" rx="6"/>
<text class="m2-lbl" x="141" y="146">bucket 0</text>
<text class="m2-lbl" x="363" y="146">bucket 1</text>
<text class="m2-lbl" x="585" y="146">bucket 2</text>
<rect class="m2-sweep m2-mv" x="684" y="52" width="5" height="84" rx="2"/>
<rect class="m2-lane" x="40" y="170" width="646" height="64" rx="6"/>
<rect class="m2-tx m2-tx0" x="48" y="182" width="186" height="40" rx="5"/>
<rect class="m2-tx m2-tx1" x="270" y="182" width="186" height="40" rx="5"/>
<rect class="m2-tx m2-tx2" x="492" y="182" width="186" height="40" rx="5"/>
<text class="m2-lbl" x="363" y="252">DP reduce-scatter, overlapped with the backward pass</text>
<text class="m2-mono" x="363" y="276">register_grad_ready() -&gt; start_grad_sync()   --overlap-grad-reduce</text>
</svg>
<figcaption>Backward walks the buffer in reverse; each bucket fires its reduce-scatter the moment its last gradient lands, so by the end of the pass all three transfers are already retired.</figcaption>
</figure>

```python
def register_grad_ready(self, param):
    """Called from a backward hook the moment this param's grad lands."""
    assert self.ddp_config.overlap_grad_reduce
    self.params_with_grad.add(param)
    # Once every param in this bucket group has a gradient, fire immediately.
    if len(self.params_with_grad) == len(self.params):
        self.start_grad_sync()


def start_grad_sync(self):
    """Launch reduce-scatter (distributed optimizer) or all-reduce (plain DDP)."""
    if self.ddp_config.use_distributed_optimizer:
        local_data_view = shard_buffer(self.grad_data, self.data_parallel_world_size)[
            self.data_parallel_rank
        ]
        self.communication_handle = dist_reduce_scatter_func(
            local_data_view, self.grad_data,
            group=self.data_parallel_group, async_op=True,
        )
    else:
        self.communication_handle = torch.distributed.all_reduce(
            self.grad_data, group=self.data_parallel_group, async_op=True,
        )
```

By the time the backward pass reaches the embedding layer, the buckets from the top of the model have long since finished their reduce-scatter. `finish_grad_sync()` at the end of the step usually waits on nothing at all.

The mirror image happens in the forward pass. With `--overlap-param-gather`, `start_param_sync()` all-gathers the next bucket's parameters while the current bucket's layers are still computing, and `next_param_gather_bucket_group` chains them so each completion immediately dispatches the following one.

### Second-order optimization: bucket size is a real tuning knob

Too small and you pay per-collective latency (and NCCL never reaches peak bandwidth). Too large and the first bucket cannot fire until most of the backward pass has completed, which defeats the overlap. The default is tuned for typical dense models; for a model with unusual layer sizes — very wide embeddings, or MoE layers an order of magnitude larger than the attention blocks — it is worth sweeping `--ddp-bucket-size` explicitly. I have seen 15% of step time recovered on an MoE run purely by shrinking buckets so the expert gradients did not all land in one enormous final collective.

For the fuller comparison of this design against DeepSpeed's stages 2 and 3, see [DeepSpeed ZeRO and 3D parallelism](/blog/machine-learning/open-source-library/deepspeed-zero-3d-parallelism-deep-dive) — the short version is that Megatron deliberately stops at stage 1 and spends its memory budget on model parallelism instead of on gradient and parameter sharding.

## 7. Memory: recomputation, offloading, and low precision

Three levers remain once the parallelism axes are set, and they trade different resources.

### Recomputation

Activation checkpointing discards activations in the forward pass and recomputes them in the backward. Megatron exposes it along two dimensions.

**Granularity** (`--recompute-granularity`):
- `full` — checkpoint whole transformer layers. Saves the most memory, costs a full extra forward pass (~30% step time).
- `selective` — checkpoint only specific modules, chosen because they are cheap to recompute and expensive to store.

**Method** (`--recompute-method`, for `full`):
- `uniform` — divide all layers into chunks of `--recompute-num-layers` and checkpoint each chunk's input.
- `block` — checkpoint only a contiguous window of layers, skipping the first `recompute_skip_num_layers`. This is the one to use when you need *some* memory back but not all of it; it lets you recompute exactly as many layers as required to fit.

Selective is the one that earns its keep. Attention scores are $O(s^2)$ to store and cheap to regenerate, so recomputing just those buys most of the memory for a small fraction of the compute:

```bash
--recompute-granularity selective \
--recompute-modules core_attn moe_act layernorm
```

The module list is the modern refinement — `core_attn`, `moe_act`, `layernorm`, `mla_up_proj`, `mlp`, `moe` — letting you target exactly the memory hog in your architecture rather than accepting a preset.

Beyond that there is `--fine-grained-activation-offloading --offload-modules expert_fc1 moe_act`, which moves activations to host memory over PCIe rather than recomputing them, and `--optimizer-cpu-offload` for the optimizer state. Both trade PCIe bandwidth for GPU memory, and both are correct only when you are genuinely memory-bound rather than bandwidth-bound.

### Low precision

FP8 is where the current performance frontier sits, and the recipe matters more than the format:

| Recipe | Granularity | Format | Platform | Status |
| --- | --- | --- | --- | --- |
| Per-tensor | whole tensor | E4M3 / E5M2 | Hopper, Blackwell | Conservative, widely safe |
| Blockwise | 1×128 activations, 128×128 weights | E4M3 | Hopper | Production-proven |
| MXFP8 | 1×32 with E8M0 scales | E4M3 + E8M0 | Blackwell | Native hardware support |

Per-tensor scaling means one scale factor for an entire tensor, so a single outlier channel compresses everything else toward zero. Blockwise gives each 128-element block its own scale, which is why it survives contact with real activation distributions. On Hopper, blockwise is the recommendation:

```bash
--fp8-format e4m3 \
--fp8-recipe blockwise \
--fp8-param-gather \
--moe-router-padding-for-fp8
```

`--fp8-param-gather` is the flag people miss. It performs the distributed optimizer's parameter all-gather *in FP8* rather than bf16, halving that collective's volume. And `--moe-router-padding-for-fp8` pads the routing map so each expert's token count aligns to the FP8 GEMM's tile requirements — without it, ragged expert batches silently fall back to a slower path.

FP8 attacks three walls at once: 50% activation memory, 50% EP dispatch volume and parameter all-gather volume, and faster tensor-core GEMMs. That is the rare optimization with no obvious counter-argument, provided your recipe is granular enough to keep the numerics honest.

## 8. How the groups are actually built

Everything above assumes `get_tensor_model_parallel_group()` returns the right set of ranks. Constructing those groups correctly is `parallel_state.py`'s job, and at 114 KB it is the largest file in the core for a reason.

```python
initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    expert_tensor_parallel_size: Optional[int] = None,
    num_distributed_optimizer_instances: int = 1,
    nccl_communicator_config_path: Optional[str] = None,
    distributed_timeout_minutes: int = 30,
    order: str = "tp-cp-ep-dp-pp",
    ...
) -> None
```

The parameter that matters most, and that almost nobody touches, is `order`.

`RankGenerator` treats the global rank space as a multi-dimensional index. With `order="tp-cp-ep-dp-pp"`, the leftmost dimension has the **smallest stride** — TP ranks are adjacent global ranks, hence on the same node, hence talking over NVLink. Pipeline ranks have the largest stride, so they are furthest apart, which is correct because PP moves the least data (one activation tensor per microbatch boundary) and tolerates InfiniBand latency.

Concretely, the decomposition is:

$$
\text{global\_rank} = \text{tp\_rank} + \text{dp\_rank} \cdot t + \text{pp\_rank} \cdot t \cdot d
$$

for TP size $t$ and DP size $d$. Take 16 GPUs across two 8-GPU nodes at TP=2, DP=2, PP=4:

| Global rank | Node | tp | dp | pp | TP group | DP group | PP group |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 0 | 0 | {0,1} | {0,2} | {0,4,8,12} |
| 1 | 0 | 1 | 0 | 0 | {0,1} | {1,3} | {1,5,9,13} |
| 2 | 0 | 0 | 1 | 0 | {2,3} | {0,2} | {2,6,10,14} |
| 3 | 0 | 1 | 1 | 0 | {2,3} | {1,3} | {3,7,11,15} |
| 4 | 0 | 0 | 0 | 1 | {4,5} | {4,6} | {0,4,8,12} |
| 8 | 1 | 0 | 0 | 2 | {8,9} | {8,10} | {0,4,8,12} |
| 12 | 1 | 0 | 0 | 3 | {12,13} | {12,14} | {0,4,8,12} |

Read the columns. Every TP group is a pair of **adjacent** ranks, so it never leaves a node — that is the ordering doing its job. Every PP group is strided by 4 and spans both nodes, which is fine because pipeline traffic is one tensor per microbatch boundary. DP groups sit in between.

Now imagine `order="pp-dp-tp"` instead. TP groups would become `{0,4,8,12}` — four ranks strided across both nodes — and every one of the two all-reduces per transformer layer would cross InfiniBand. The model would train perfectly. It would simply do so at a fraction of the speed.

![The same 16 GPUs under two orderings: only one keeps tensor-parallel groups on NVLink](/imgs/blogs/megatron-lm-parallelism-techniques-deep-dive-15.webp)

Getting this backwards is catastrophic and completely silent. A TP group spread across four nodes will train correctly and produce identical loss curves — it will just do so at a third of the throughput, because the per-layer all-reduce that assumed NVLink is now crossing InfiniBand twice per transformer layer.

This is the mechanism behind the rule everyone repeats without explaining: **keep tensor parallelism inside a node**. At 8 GPUs per node with NVLink, TP=8 is the largest TP group that stays local. TP=16 necessarily crosses the node boundary, and the all-reduce on the critical path of every layer now runs at InfiniBand bandwidth.

## 9. Cross-cutting: checkpoints, conversion, and running the thing

### Distributed checkpointing

A checkpoint saved at TP=8, PP=16 is 128 shards of a model that no longer resembles its logical structure. Megatron's `dist_checkpointing` layer solves this by having every module describe its shards symbolically:

```python
from megatron.core import dist_checkpointing

def save_distributed_checkpoint(checkpoint_path, gpt_model):
    sharded_state_dict = gpt_model.sharded_state_dict(prefix='')
    dist_checkpointing.save(
        sharded_state_dict=sharded_state_dict, checkpoint_dir=checkpoint_path
    )

def load_distributed_checkpoint(checkpoint_path, gpt_model):
    sharded_state_dict = gpt_model.sharded_state_dict(prefix='')
    checkpoint = dist_checkpointing.load(
        sharded_state_dict=sharded_state_dict, checkpoint_dir=checkpoint_path
    )
    gpt_model.load_state_dict(checkpoint)
    return gpt_model
```

Because `sharded_state_dict()` records each tensor's *global* shape and this rank's offset within it, a checkpoint saved under one parallel configuration can be loaded under another. That is what makes it possible to pretrain at TP=8/PP=16 and fine-tune at TP=2/PP=1 without a conversion script.

### Megatron Bridge

The other conversion problem is ecosystem-shaped: the world's model weights live in Hugging Face format, and Megatron's do not. **Megatron Bridge** provides bidirectional HF ↔ Megatron checkpoint conversion with production recipes, which is what closes the loop between "pretrain in Megatron" and "serve in vLLM or SGLang".

### A complete, runnable Megatron-Core loop

Stripped of the training harness, using the library directly:

```python
import os
import torch
from functools import partial
from torch.optim import Adam
from torch.utils.data import DataLoader

from megatron.core import parallel_state
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.datasets.blended_megatron_dataset_builder import (
    BlendedMegatronDatasetBuilder,
)
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, MockGPTDataset
from megatron.core.datasets.utils import compile_helpers
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.training.tokenizer.tokenizer import _NullTokenizer

_SEQUENCE_LENGTH = 64


def initialize_distributed(tensor_model_parallel_size=1, pipeline_model_parallel_size=1):
    rank = int(os.environ['LOCAL_RANK'])
    world_size = torch.cuda.device_count()
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(world_size=world_size, rank=rank)
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size, pipeline_model_parallel_size
    )


def model_provider():
    transformer_config = TransformerConfig(
        num_layers=2,
        hidden_size=12,
        num_attention_heads=4,
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32,
    )
    return GPTModel(
        config=transformer_config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=100,
        max_sequence_length=_SEQUENCE_LENGTH,
    )


def forward_step_func(data_iterator, model):
    def loss_func(loss_mask, output_tensor):
        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
        return loss, {'lm loss': loss}

    device = torch.device("cuda")
    data = next(data_iterator)
    output_tensor = model(
        data['tokens'].to(device),
        data['position_ids'].to(device),
        data['attention_mask'].to(device),
        labels=data['labels'].to(device),
    )
    return output_tensor, partial(loss_func, data['loss_mask'].to(device))


def get_train_data_iterator():
    if torch.distributed.get_rank() == 0:
        compile_helpers()
    torch.distributed.barrier()
    config = GPTDatasetConfig(
        random_seed=0,
        sequence_length=_SEQUENCE_LENGTH,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        tokenizer=_NullTokenizer(vocab_size=_SEQUENCE_LENGTH),
    )
    datasets = BlendedMegatronDatasetBuilder(
        MockGPTDataset, [1000, None, None], lambda: True, config
    ).build()
    return iter(DataLoader(datasets[0], batch_size=8, shuffle=True))


if __name__ == "__main__":
    initialize_distributed(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)

    gpt_model = model_provider().to(torch.device("cuda"))
    optim = Adam(gpt_model.parameters())
    train_iterator = get_train_data_iterator()

    # THIS is the line that hides all of §3
    forward_backward_func = get_forward_backward_func()

    for _ in range(5):
        optim.zero_grad()
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=train_iterator,
            model=gpt_model,
            num_microbatches=1,
            seq_length=_SEQUENCE_LENGTH,
            micro_batch_size=8,
            decoder_seq_length=_SEQUENCE_LENGTH,
            forward_only=False,
        )
        optim.step()
        print(f'Losses reduced : {losses_reduced}')
```

Launch it with `torchrun --nproc-per-node 2 train_loop.py`. Note `model_parallel_cuda_manual_seed(123)` — it seeds two separate RNG states, one shared across TP ranks (for anything that must be identical, like dropout on replicated tensors) and one that differs per rank (for dropout on sharded tensors). Skip it and your tensor-parallel dropout masks will be correlated in ways that quietly degrade the model.

### The real launch scripts

The reference 175B config, verbatim in structure:

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

GPT_MODEL_ARGS=(
    --num-layers 96 --hidden-size 12288 --num-attention-heads 96
    --seq-length 2048 --max-position-embeddings 2048
    --attention-backend auto
)
TRAINING_ARGS=(
    --micro-batch-size 1 --global-batch-size 1536
    --train-iters 500000 --weight-decay 0.1
    --adam-beta1 0.9 --adam-beta2 0.95 --init-method-std 0.006
    --clip-grad 1.0 --fp16
    --lr 6.0e-5 --lr-decay-style cosine --min-lr 6.0e-6
    --lr-warmup-fraction .001 --lr-decay-iters 430000
)
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 8
    --pipeline-model-parallel-size 16
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} ${TRAINING_ARGS[@]} ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} ${EVAL_AND_LOGGING_ARGS[@]}
```

TP=8 (one node, NVLink) × PP=16 (across nodes) = 128 GPUs of model parallelism, with data parallelism filling whatever remains. A modern MoE config layers considerably more on top:

```bash
MOE_ARGS=(
    --num-experts 256 --moe-router-topk 8
    --moe-router-score-function sigmoid --moe-router-dtype fp32
    --moe-router-enable-expert-bias --moe-router-bias-update-rate 1e-3
    --moe-token-dispatcher-type flex --moe-flex-dispatcher-backend deepep
    --moe-grouped-gemm --moe-permute-fusion --moe-router-fusion
    --moe-shared-expert-intermediate-size 2048 --moe-shared-expert-overlap
)
PARALLEL_ARGS=(
    --tensor-model-parallel-size 4 --context-parallel-size 2
    --pipeline-model-parallel-size 4
    --expert-model-parallel-size 64 --expert-tensor-parallel-size 1
    --sequence-parallel --use-distributed-optimizer
    --overlap-grad-reduce --overlap-param-gather --tp-comm-overlap
    --overlap-moe-expert-parallel-comm --delay-wgrad-compute
)
FP8_ARGS=(
    --fp8-format e4m3 --fp8-recipe blockwise
    --fp8-param-gather --moe-router-padding-for-fp8
)
MEMORY_ARGS=(
    --recompute-granularity selective
    --recompute-modules core_attn moe_act
)
```

Every flag in that block maps to a section above. That is the payoff of understanding the axes rather than memorizing the flags.

## 10. Observability: the failures that do not raise

A distributed training job has a property that makes it uniquely hostile to debug: **almost every failure mode produces a loss curve that looks fine.** A misplaced TP group, a straggling GPU, a silently corrupted gradient, a router collapsing onto four experts — none of these throw. They cost you throughput, or quality, or both, while the dashboard stays green.

Megatron-Core ships a surprising amount of machinery aimed squarely at this, and most teams use none of it.

**Timers.** `megatron/core/timers.py` instruments the phases of a step — forward, backward, the optimizer, each collective — and `--timing-log-level` controls how much detail lands in your logs and TensorBoard. The single most useful diagnostic in the framework is the ratio of communication time to compute time per step, because every misconfiguration in this post shows up there first. If your TP all-reduce time jumps 3× after a cluster change, you have crossed a node boundary, and you will see it here weeks before you see it in a benchmark.

**Straggler detection.** `megatron/core/README_STRAGGLER.md` documents a detector for the classic distributed pathology: one GPU running slightly slower than its peers — thermal throttling, a degraded NVLink, a bad card — silently setting the pace for the entire job. With collectives at every layer boundary, a single rank at 90% speed makes the *whole cluster* run at 90%, and the cost is invisible in any per-rank metric because every rank reports the same step time. Only a cross-rank comparison finds it.

**Result validation and replay.** `megatron/core/rerun_state_machine.py` is, at 62 KB, one of the larger files in the core — a fact worth pausing on. It exists to re-execute work and compare results, which is the only reliable way to catch **silent data corruption**: a GPU that returns wrong arithmetic without raising, at a rate low enough that nothing crashes and high enough to poison a multi-week run. Its companion `fault_injector.py` exists to test that path deliberately. If you have ever wondered why a 30-day pretraining run diverged at day 19 with no bad input and no code change, this is the class of cause that machinery is built for.

**Configuration logging.** `config_logger.py` dumps the fully-resolved configuration, which matters more than it sounds. Between CLI flags, environment variables, config files, and the defaults each argument falls back to, the config the job *actually ran with* is frequently not the config anyone intended — case study 1 below is exactly this failure.

**Energy.** `energy_monitor.py` tracks power draw. At the scale where you are reading this post, energy is a first-order cost line, not a curiosity.

The practical minimum I would want on any run longer than a day: per-phase timers exported to TensorBoard, a cross-rank step-time comparison to catch stragglers, resolved-config logging archived with the checkpoint, and expert-utilization histograms if the model is MoE. All four catch problems that produce no error message.

> The loss curve is not a health check. It is the one metric that looks identical whether your cluster is running at 47% MFU or 31%.

## Case studies from production

### 1. The overlap flag that did nothing

A 175B run on 512 H100s showed `--tp-comm-overlap` producing no measurable improvement. The team profiled for a week, suspecting the userbuffer allocation. The actual cause: the cluster's job wrapper sanitized the environment and dropped `CUDA_DEVICE_MAX_CONNECTIONS=1`. With multiple hardware connections available, the GPU scheduler reordered the communication kernel ahead of the wgrad GEMM it was supposed to hide behind. Restoring one `export` recovered roughly 9% of step time. **Lesson: this variable is not a tuning knob, it is a correctness precondition for every overlap optimization in the framework.**

### 2. TP=16 and the vanishing MFU

A team scaling a 70B model needed more memory per GPU and moved from TP=8 to TP=16, reasoning that halving per-GPU weights was worth some extra communication. MFU fell from 47% to 31%. At 8 GPUs per node, TP=16 spans two nodes, so the per-layer all-reduce — which happens twice per transformer layer and sits directly on the critical path — moved from NVLink to InfiniBand. The fix was to keep TP=8 and add pipeline parallelism for the extra memory, since PP communicates once per microbatch boundary rather than twice per layer. **Lesson: TP is a bandwidth-hungry axis with a hard node boundary. Cross it and you are not slightly worse, you are structurally worse.**

### 3. Four experts eating sixty percent of the tokens

A 128-expert MoE showed severe routing collapse: expert utilization metrics were wildly skewed and the loss curve was noisy. The auxiliary loss was configured correctly and the coefficient was reasonable. The problem was `--moe-router-dtype bf16`, chosen to save a little memory. With 128 logits, bf16's ~3 decimal digits of precision made the top-8 boundary effectively arbitrary for tokens with similar scores, and small numerical biases compounded into a self-reinforcing preference. Switching to fp32 — plus `--moe-router-enable-expert-bias` for aux-loss-free balancing — flattened the distribution within a few hundred steps. **Lesson: the router is a discrete decision made from continuous scores. It is the one place in the model where a rounding error changes control flow.**

### 4. Interleaving that OOM'd a job with room to spare

A run comfortably using 62 GB of 80 GB enabled `--num-layers-per-virtual-pipeline-stage 2` to shave the bubble, and OOM'd immediately on pipeline stage 0. The engineer went looking at activation checkpointing. The real cause was the warmup formula: with $v = 4$ chunks, the `(num_model_chunks - 1) * microbatch_group_size_per_vp_stage` term added substantially more in-flight microbatches on the earliest stage, and each carries a full set of layer activations. The fix was `--decoder-first-pipeline-num-layers` to give stage 0 fewer layers. **Lesson: interleaving costs memory on the first stage specifically, and it costs it during warmup, not steady state.**

### 5. The context-parallel ring that ran at one-eighth speed

A 128k-context run at CP=8 was profiled showing every rank at low utilization with large synchronization gaps. Ranks were finishing their attention computation at wildly different times, and since each ring step is a barrier, everyone waited for the slowest. The sequence had been split into 8 contiguous chunks, so under the causal mask rank 7 was doing roughly 8× rank 0's work. The framework's `2 × cp_size` striped assignment fixed it. **Lesson: with a causal mask, contiguous sequence chunking is always load-imbalanced, and the imbalance is linear in CP size.**

### 6. FP8 param gather and the requantization stall

Enabling `--fp8-param-gather` alongside `--overlap-param-gather` produced a stall right at the start of each layer's forward pass. Gathering parameters in FP8 halves the collective's bytes, but the gathered values must be requantized into the layer's expected storage format afterward — `_post_param_sync()` doing `copy_tensors_to_quantized_params()`. That post-processing was landing on the critical path because bucket sizes were large enough that the gather completed only just in time. Shrinking `--ddp-bucket-size` gave the requantization room to hide behind the previous bucket's compute. **Lesson: an optimization that reduces bytes can add a compute step, and overlap budgets have to account for both.**

### 7. Resharding a checkpoint and the loss spike that followed

A model pretrained at TP=8 was loaded at TP=4 for fine-tuning. Distributed checkpointing resharded the weights correctly and the model's initial loss looked right. Two hundred steps in, the loss spiked. The weights had reshaped fine; the *optimizer state* had not been reshaped consistently for parameters straddling a shard boundary, so a subset of Adam's second-moment estimates were mismatched with their parameters — producing effective step sizes far larger than intended for those slices. **Lesson: `param` in the four-range map exists precisely because parameters straddle shards, and any code path that reshards has to honor it for optimizer state, not just for weights.**

### 8. EP×TP crossing the NVLink domain

An MoE config used `EP=8 × TP=2` on 8-GPU nodes, putting $\text{EP} \times \text{TP} = 16$ — two nodes — inside the innermost communication region. All-to-all throughput fell off a cliff, because every token dispatch now crossed InfiniBand. Reconfiguring to `EP=8 × TP=1` with the memory recovered via more pipeline stages kept the product at 8 and restored expected throughput. It also made each expert GEMM twice as large, which improved tensor-core efficiency independently. **Lesson: the $\text{EP} \times \text{TP} \leq \text{GPUs-per-node}$ guideline is about the NVLink domain, and violating it costs you twice — once in bandwidth, once in GEMM shape.**

### 9. Strong scaling GPT-3 from 96 to 4,608 GPUs

NVIDIA's own published scaling data is the cleanest illustration of where the ceiling is. Holding GPT-3 175B fixed and scaling from 96 to 4,608 H100s, MFU falls from 47% to 42%. Weak scaling in the same benchmark suite moves the other way — from 41% at the smallest models up to 47–48% at the largest, with a 462B model trained on 6,144 H100s. The gap is exposed communication: at fixed model size, adding GPUs means more data-parallel ranks reducing over the same gradients, and eventually the reduce stops hiding behind compute. **Lesson: strong scaling has a floor set by your collective bandwidth, and past it more GPUs buys latency, not throughput. Weak scaling — bigger model, more GPUs — is the regime this framework is built for.**

### 10. The one GPU that set the pace for four thousand

A large run showed step times about 11% worse than the same configuration had achieved a month earlier, with no code or config change. Every rank reported the same step time, so per-rank metrics showed nothing — which is exactly the signature of the problem. Because collectives synchronize at every layer boundary, a single slow rank does not appear as one slow rank; it appears as *uniform* slowness across the entire job, since everyone else spends the difference waiting inside NCCL. A cross-rank comparison of pre-collective compute time isolated one GPU that had begun thermal-throttling under sustained load. Draining that node restored the original step time. **Lesson: in a tightly-coupled job, the slowest rank sets the pace and then hides, because waiting inside a collective is indistinguishable from working. You cannot find stragglers without comparing ranks to each other.**

### 11. The divergence on day nineteen

A multi-week pretraining run diverged with no bad data batch, no code change, and no hardware alert. The checkpoint from the previous day resumed and trained fine for several days before diverging again — on a different step, from a different checkpoint. Nothing was reproducible, which ruled out a data or code bug and pointed at hardware. The eventual cause was a GPU producing occasional incorrect arithmetic without raising: rare enough that no step crashed, frequent enough that corrupted gradients accumulated into the optimizer state over days. This is the failure class `rerun_state_machine.py` exists for — re-executing work and comparing results is the only way to catch an error that never announces itself. **Lesson: at thousands of GPUs running for weeks, silent data corruption stops being a theoretical concern and becomes a scheduled event. Budget for detecting it, not just for checkpointing around it.**

## Megatron vs the alternatives

The honest framing is that these frameworks disagree about *where to spend memory*, and that disagreement determines who they suit.

| | Megatron-Core | DeepSpeed | FSDP2 / TorchTitan |
| --- | --- | --- | --- |
| Optimizer state sharding | ZeRO-1 only | ZeRO 1/2/3 + offload + Infinity | Full param/grad/optimizer sharding |
| Tensor parallelism | First-class, the core abstraction | Available, less central | First-class in TorchTitan |
| Pipeline parallelism | 1F1B + interleaved, deeply tuned | Supported | Supported, simpler schedules |
| Context parallelism | Ring + causal load balancing + dynamic | Limited | Supported |
| Expert parallelism | Strongest — folding, DeepEP, grouped GEMM | DeepSpeed-MoE | Emerging |
| FP8 / MXFP8 | Earliest and most granular | Available | Available via TE |
| API ergonomics | Framework-specific, large surface | Config-file driven | PyTorch-native, readable |
| Best at | Frontier pretraining on NVIDIA clusters | Memory-constrained training, offload | Clean composition, moderate scale |

Megatron deliberately stops at ZeRO stage 1 and spends the remaining memory budget on model parallelism instead. DeepSpeed goes the other way — shard everything, offload to host and NVMe, keep the model logically intact. Both are coherent positions. Megatron's wins when interconnect is fast and the model is enormous; DeepSpeed's wins when memory is the hard wall and you would rather trade PCIe bandwidth than restructure the model.

TorchTitan is the third position: accept slightly less peak MFU in exchange for code you can read and modify. For most teams not operating at the 462B-parameter frontier, that is the right trade, and I say so as someone who likes this codebase.

## When to reach for Megatron-LM

**Reach for it when:**

- You are **pretraining from scratch** at a scale where the model does not fit in one node's aggregate memory. This is the case the framework was designed for and where nothing else is close.
- You need **all five parallelism axes to compose**. Megatron is the most complete implementation of TP × CP × EP × DP × PP that exists, and parallelism folding for MoE has no real equivalent elsewhere.
- You are training a **large MoE** — 100+ experts, fine-grained routing, cross-node expert parallelism. The dispatcher work (DeepEP, hybridep, grouped GEMM, shared-expert overlap) is well ahead of the alternatives.
- You are on **NVIDIA hardware and want the newest numerics first**. Blockwise FP8 on Hopper and MXFP8 on Blackwell land here before they land anywhere else.
- You need **checkpoints that reshard** across different parallel configurations without a conversion script.

**Skip it when:**

- You are **fine-tuning**, especially with LoRA or QLoRA. The parallelism machinery is overhead you will never use; reach for a purpose-built stack instead.
- Your model **fits on one node** with FSDP or ZeRO. DDP plus a distributed optimizer will get you the same throughput with a fraction of the configuration surface.
- You want **PyTorch-native ergonomics**. [TorchTitan](/blog/machine-learning/open-source-library/torchtitan-pytorch-native-pretraining-deep-dive) composes FSDP2, TP, PP and CP with substantially less framework-specific API, and is the better starting point if you value readable code over the last few points of MFU.
- You are **serving, not training**. Almost none of this transfers — see [tensor-parallel inference](/blog/machine-learning/inference-engineering/tensor-parallel-inference-by-hand) for why the constraints invert.
- Your team **cannot own the configuration**. Megatron will happily run a misconfigured job at a third of achievable throughput and never warn you, because the loss curve looks fine.

![Which axis to turn on first, and what each one buys back](/imgs/blogs/megatron-lm-parallelism-techniques-deep-dive-14.webp)

The ordering in that tree is the practical summary of everything above. Distributed optimizer first, because it is nearly free. Then tensor parallelism up to the node boundary, with sequence parallelism turned on alongside it because it costs nothing. Then pipeline parallelism for more memory, interleaved only if the bubble justifies the bandwidth. Context parallelism only when the sequence itself is the problem. Expert parallelism first and foremost if the model is MoE, ahead of tensor parallelism for the expert layers.

Each axis buys back a different scarcity. Choosing the wrong one costs you MFU and gives you nothing, and the framework will not tell you.

## Further reading

- **Megatron-LM** (Shoeybi et al., 2019) — the original tensor-parallel formulation and the column-then-row derivation.
- **Efficient Large-Scale Language Model Training on GPU Clusters** (Narayanan et al., 2021) — where the interleaved pipeline schedule and the 3D-parallelism analysis come from.
- **Reducing Activation Recomputation in Large Transformer Models** (Korthikanti et al., 2022) — sequence parallelism and selective recomputation, with the activation-memory accounting used in §2.
- **Megatron-Core developer guide** — `docs.nvidia.com/megatron-core/developer-guide/latest/`
- [Tensor parallelism from first principles](/blog/machine-learning/distributed-training/tensor-parallelism-megatron) — the derivation this post assumes.
- [Pipeline parallelism and the bubble](/blog/machine-learning/distributed-training/pipeline-parallelism-and-the-bubble) — the scheduling analysis in more depth.
- [DeepSpeed ZeRO and 3D parallelism](/blog/machine-learning/open-source-library/deepspeed-zero-3d-parallelism-deep-dive) — the other major answer to the same problem.
- [TorchTitan](/blog/machine-learning/open-source-library/torchtitan-pytorch-native-pretraining-deep-dive) — the PyTorch-native alternative.
