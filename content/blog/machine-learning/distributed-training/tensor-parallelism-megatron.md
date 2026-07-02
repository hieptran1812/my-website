---
title: "Tensor Parallelism: Splitting the Matmul Across GPUs the Megatron Way"
date: "2026-07-02"
publishDate: "2026-07-02"
description: "When a single layer will not fit or per-step latency dominates, tensor parallelism splits the matmul itself across GPUs — here is the Megatron column-then-row trick, the all-reduce it costs, and why it must live on NVLink."
tags:
  [
    "distributed-training",
    "multi-gpu",
    "tensor-parallelism",
    "megatron",
    "model-parallelism",
    "nccl",
    "deep-learning",
    "ml-systems",
    "nvlink",
  ]
category: "machine-learning"
subcategory: "Distributed Training"
author: "Hiep Tran"
featured: true
readTime: 27
---

You are training a 20-billion-parameter transformer, and you have hit a wall that data parallelism cannot climb. It is not that the whole model does not fit — you already sharded the optimizer states with FSDP and the weights sit comfortably across your eight GPUs. The problem is narrower and more stubborn: one particular matmul, the up-projection in the MLP block, has a weight matrix that is two gigabytes on its own, and when you materialize it plus its activation plus the gradient, that single operation blows past what one 80GB card can hold at your batch size. Or maybe the model *does* fit, but each step takes 900 milliseconds and you need it under 300, and no amount of adding more data-parallel replicas helps because each replica still runs the full forward pass at full latency. Both of these are the same disease with the same cure: you need to split the *work of a single layer* across multiple GPUs. That is tensor parallelism, and the figure below is the shape of the trick.

![a dataflow graph where the input activation feeds two column shards of the first matmul that each run a local GeLU then two row shards of the second matmul whose partial outputs merge into a single all-reduce](/imgs/blogs/tensor-parallelism-megatron-1.webp)

Data parallelism, which we built up in [DDP from first principles](/blog/machine-learning/distributed-training/ddp-from-first-principles), splits the *batch* and replicates the model. Tensor parallelism does the opposite: it keeps the batch whole on every GPU and splits the *model's weight matrices* across them. Each GPU holds a slice of every layer and computes a slice of every activation, and the GPUs stitch their slices together with collective communication inside the layer. This is intra-layer parallelism, the third axis on [the map of parallelism](/blog/machine-learning/distributed-training/the-map-of-parallelism), and it was made practical at scale by NVIDIA's Megatron-LM. By the end of this post you will be able to derive the Megatron column-then-row split from scratch, know exactly where the all-reduce lands and why it sits on the critical path where it cannot hide, price the communication on NVLink versus PCIe versus an inter-node fabric and predict whether tensor parallelism will help or wreck your throughput, write the column-parallel and row-parallel linear layers in PyTorch, and decide with confidence when to reach for it and — just as important — when not to. This is the third post in the [Distributed Training in the Trenches](/blog/machine-learning/distributed-training/why-distributed-training) Track C on model parallelism.

## What tensor parallelism splits, and what it does not

Start with the invariant, because it is what every design decision protects. In data parallelism the invariant is *the replicas must stay identical* — every GPU holds the same weights and takes the same step. In tensor parallelism the invariant is different: *no single GPU ever holds the whole layer, but together they compute exactly what the unsharded layer would have computed.* The output of a tensor-parallel MLP block is bit-for-bit (up to floating-point reduction order) what a single giant GPU would have produced. Tensor parallelism is a way to run one big layer on many GPUs, not a way to run many copies of a small one.

The batch stays whole. Every GPU in a tensor-parallel group sees the same input tokens, the same sequence, the same batch. What differs is which *columns* or *rows* of the weight matrices each GPU owns, and therefore which slice of the intermediate activation each GPU computes. Because the batch is not split, tensor parallelism does not directly increase your throughput the way adding data-parallel replicas does — it increases the *size of model* you can run at a given latency, and it *reduces per-step latency* by spreading one layer's compute over more silicon. Those are the two things it buys: fit and speed-per-step. Everything else it costs.

And it costs communication, a lot of it, in a place that hurts. That is the tension the rest of this post unpacks. But first, the mechanism, because the beauty of Megatron is that the communication is as small as it can possibly be — exactly one all-reduce per block in the forward pass and one in the backward — and that is not an accident of engineering but a consequence of choosing the split directions correctly.

## The Megatron MLP trick, derived

Take the standard transformer MLP block. It is two linear layers with a nonlinearity between them: $Y = \text{GeLU}(X A) B$, where $X$ is the input activation of shape batch-by-sequence-by-hidden, $A$ is the up-projection weight (hidden by 4-hidden, say), and $B$ is the down-projection (4-hidden by hidden). We want to split $A$ and $B$ across two GPUs — call the tensor-parallel degree two for now — such that the total communication is minimal.

There are two ways to split a weight matrix: by columns or by rows. The whole Megatron insight is that you must choose the direction so that the nonlinearity in the middle does not force a communication. Consider splitting $A$ by **columns**: $A = [A_1, A_2]$, where $A_1$ is the left half of the columns and $A_2$ the right half. Then $XA = [XA_1, XA_2]$ — GPU 0 computes $XA_1$ and GPU 1 computes $XA_2$, each producing half the columns of the intermediate activation. Here is the key: because GeLU is applied elementwise, $\text{GeLU}([XA_1, XA_2]) = [\text{GeLU}(XA_1), \text{GeLU}(XA_2)]$. Each GPU can apply the nonlinearity to its own half **with no communication at all**. If we had split $A$ by rows instead, each GPU would compute a partial sum of the *full* activation, and we would need to all-reduce *before* the GeLU — a communication on every block just to apply a pointwise function. Column-splitting the first matrix makes the nonlinearity free.

Now the second matmul, $B$. The input to $B$ is the intermediate activation, which after the column split lives as two halves: GPU 0 holds $\text{GeLU}(XA_1)$, GPU 1 holds $\text{GeLU}(XA_2)$. To multiply this by $B$ without a communication, we split $B$ by **rows**: $B = [B_1; B_2]$ (stacked). Then $\text{GeLU}(XA_1) B_1$ is computed entirely on GPU 0 and $\text{GeLU}(XA_2) B_2$ entirely on GPU 1, and — this is the arithmetic that makes it work — the true output $Y = \text{GeLU}(XA_1)B_1 + \text{GeLU}(XA_2)B_2$ is the *sum* of the two GPUs' partial outputs. Each GPU has a partial $Y$ of the full shape; adding them gives the answer. That addition across GPUs is exactly an **all-reduce**. Column-parallel then row-parallel: the GeLU is local, and the whole block costs a single all-reduce at the end. The figure above traces this path — input fanning into two column shards, local GeLU, two row shards, and the merge into one all-reduce.

That is the entire trick, and it generalizes to any tensor-parallel degree $t$: split $A$ into $t$ column blocks, split $B$ into $t$ row blocks, and the block costs one all-reduce of the output activation. No communication touches the nonlinearity.

The backward pass deserves a moment because it is where the symmetry becomes clear and where a subtle communication hides. Gradients flow in the opposite direction. The gradient of the loss with respect to the block output $Y$ arrives already full (it is the same shape on every rank, since $Y$ was all-reduced to be identical). Backpropagating through the row-parallel $B$ is local — each GPU computes the gradient of its $B_i$ and the partial gradient of its intermediate activation — because the forward all-reduce's backward is the identity. But backpropagating through the column-parallel $A$ is where the second all-reduce lives: each GPU computes a partial gradient of the input $X$ (from its column slice of $A$), and those partials must be *summed* to reconstruct the true input gradient that flows to the previous block. That summation is the backward all-reduce. So the accounting holds exactly: the forward pass all-reduces the output (the $g$ operator), the backward pass all-reduces the input gradient (the $f$ operator), and a tensor-parallel MLP block costs two all-reduces total across the round trip — no more, no less. The attention block, split by heads, costs the same two. This is the minimum any exact sharded computation could achieve, and it is why Megatron's communication is so much lighter than the naive "all-reduce after every matmul" you would get from splitting in the wrong directions.

## Attention splits by heads

Multi-head attention has a natural parallel structure that fits tensor parallelism even more cleanly than the MLP: the heads are already independent. Each attention head computes its own queries, keys, values, and its own weighted sum, entirely independently of the other heads, right up until the output projection combines them. So you split by heads.

![a dataflow graph where the input fans to two GPUs each owning a disjoint subset of attention heads whose results merge through a row-parallel output projection into one all-reduce](/imgs/blogs/tensor-parallelism-megatron-5.webp)

Give GPU 0 heads 0 through 3 and GPU 1 heads 4 through 7. Each GPU has the query, key, and value projection weights for only its heads — those projections are column-parallel, exactly like $A$ above, so each GPU produces the Q, K, V for its own heads with no communication. Each GPU runs the full attention computation (the softmax over its heads' scores, the weighted sum over values) locally, because a head never needs another head's scores. The heads' outputs are then concatenated and passed through the output projection $W_O$, which is row-parallel: each GPU multiplies its heads' outputs by its slice of $W_O$ to get a partial result, and one all-reduce sums the partials into the final attention output. Same pattern, same cost: the QKV projections and attention are local, and the block ends in a single all-reduce. A transformer layer, then — attention block plus MLP block — costs two all-reduces in the forward pass and two in the backward, and nothing else crosses the wire inside the layer.

There is a subtlety worth stating because it caps the degree: you cannot have more tensor-parallel ranks than you have attention heads, since a head is the atomic unit of the split. A model with 32 heads can be tensor-parallel up to degree 32 in principle, but in practice you rarely go past 8 for a reason we come to shortly — the interconnect.

Modern models complicate this with grouped-query attention (GQA) and multi-query attention (MQA), where many query heads share a smaller number of key-value heads to shrink the KV cache. This interacts with the head split in a way that bites people: the *key-value* heads are now the scarce resource. If a model has 32 query heads but only 8 KV heads, and you set tensor-parallel degree to 8, each rank gets exactly one KV head — fine. But push the degree to 16 and you have fewer KV heads than ranks, so the KV projections must be *replicated* across some ranks rather than sharded, which both wastes memory and complicates the code. The practical rule is to keep the tensor-parallel degree at or below the number of KV heads for a GQA model, which is one more reason the degree stays small on modern architectures. When you see a recipe pin tensor-parallel degree to 8 for a model with 8 KV-head groups, this is why — it is the largest split that keeps one KV head per rank.

## The f and g conjugate operators

There is an elegant way to see the whole forward-and-backward communication pattern, and it is worth internalizing because it is how the Megatron code is actually structured. Define two conjugate operators, $f$ and $g$, that sit at the boundaries of each tensor-parallel region. In the **forward** pass, $f$ is the identity (it just passes the input through to all ranks) and $g$ is an all-reduce (it sums the partial outputs). In the **backward** pass, the roles swap by the chain rule: $f$ becomes an all-reduce (it sums the input gradients flowing back from all ranks) and $g$ becomes the identity. That is the whole communication contract — $f$ at the entry, $g$ at the exit, one an all-reduce and one an identity in each direction, swapping between forward and backward. Two operators, one all-reduce each per direction per block. When you read the Megatron source and see `f` and `g` autograd functions wrapping the column-parallel and row-parallel linears, this is what they are: the minimal communication that makes the sharded computation exact.

## The communication is on the critical path

Here is where tensor parallelism earns its reputation as the parallelism you deploy carefully. Recall from [DDP from first principles](/blog/machine-learning/distributed-training/ddp-from-first-principles) that data parallelism's gradient all-reduce is *overlappable*: gradients become ready during the backward pass, and DDP fires the all-reduce for early-ready buckets while the rest of the backward is still computing, so the communication hides under compute and costs almost nothing in wall-clock on a fast link. Tensor parallelism does not get this gift.

![a timeline showing layer compute followed by a blocking forward all-reduce then the next layer then a blocking backward all-reduce with the gradient ready and no opportunity to hide the communication](/imgs/blogs/tensor-parallelism-megatron-3.webp)

The reason is causal. The all-reduce in a tensor-parallel block produces the *output activation* of that block, and the next block cannot start until it has that output. The communication is not a side effect you can defer to the end of the step, the way a gradient is — it is a data dependency in the middle of the forward pass. Compute the block, all-reduce to get the real output, feed it to the next block, compute, all-reduce again. Each all-reduce blocks. There is nothing to overlap it with because the very next operation needs its result. As the timeline shows, the all-reduce sits squarely on the critical path in both directions, and the gradient it produces in the backward pass is likewise needed immediately. This single fact — blocking, unhideable communication on every layer — is why tensor parallelism is so sensitive to the speed of the interconnect, far more than data parallelism ever is.

## The size of the communication bill

Let us count the bytes, because the numbers are what decide everything. One all-reduce in a tensor-parallel block moves the output activation, whose size is batch times sequence times hidden dimension, times the bytes per element. Using the ring all-reduce cost we derived in [collectives from scratch](/blog/machine-learning/distributed-training/collectives-from-scratch), each GPU sends and receives roughly $2(t-1)/t$ times the activation size per all-reduce, where $t$ is the tensor-parallel degree. Now multiply: two all-reduces per layer in the forward pass region (attention plus MLP), two more in the backward, across every one of $L$ layers.

![a stacked breakdown showing one all-reduce moving a full batch-by-sequence-by-hidden activation, multiplied by two per layer, by the number of layers, all landing on the critical path](/imgs/blogs/tensor-parallelism-megatron-6.webp)

The activation is *large*. For a batch of 8 sequences of length 4096 with a hidden dimension of 6144 in bf16, one activation is about 8 × 4096 × 6144 × 2 bytes ≈ 400 MB. A 48-layer model does roughly four such all-reduces per layer across forward and backward, so on the order of 48 × 4 × 400 MB ≈ 77 GB of activation traffic must be all-reduced *per training step*, all of it blocking. Contrast that with data parallelism, whose per-step all-reduce moves the *gradients* — the same size as the weights, a fixed quantity independent of batch and sequence, and overlappable. Tensor parallelism's traffic scales with your batch and sequence length and lands on the critical path. That is a fundamentally heavier communication profile, and it is the reason the next question — which wire carries it — is the most important design decision you will make.

## Why tensor parallelism must stay inside a node

Put the communication bill on different interconnects and the verdict is stark.

![a matrix comparing NVLink, PCIe, and inter-node InfiniBand by all-reduce bandwidth, the share of step time spent communicating, and whether tensor parallelism is worth it on each](/imgs/blogs/tensor-parallelism-megatron-4.webp)

On **NVLink4**, with roughly 900 GB/s of aggregate bandwidth per GPU inside a DGX/HGX node (see [the interconnect physics](/blog/machine-learning/distributed-training/the-interconnect-physics)), that 77 GB of blocking traffic takes on the order of tens of milliseconds — a real cost, but small relative to the compute of a large model's forward and backward, so tensor parallelism is a net win. On **PCIe 4.0**, at roughly 32 GB/s, the same traffic takes 20-30× longer, and because it is blocking, it lands directly on your step time; the communication now dominates and throughput collapses. Across **nodes on InfiniBand**, at roughly 25 GB/s per direction, it is worse still, and you have the added latency of the fabric on every one of those blocking all-reduces. The rule that falls out is one of the sharpest in all of distributed training: **keep the tensor-parallel group inside a single NVLink node, and set the tensor-parallel degree no larger than the number of GPUs in that node** — typically 8. Tensor parallelism across nodes is almost always a mistake; the moment an all-reduce has to cross the inter-node fabric on every layer, the numbers stop working.

#### Worked example: TP within a node versus across two nodes

Consider a model whose per-layer forward compute on one shard is about 3 ms at tensor-parallel degree 8, and whose per-layer forward all-reduce moves 400 MB. On NVLink4 at 900 GB/s, the ring all-reduce moves about $2 \times 7/8 \times 400$ MB ≈ 700 MB per GPU, taking roughly 0.8 ms — about a 25% overhead on the 3 ms of compute, tolerable, and the fit-and-latency win pays for it. Now force that same degree-8 group to span two nodes over InfiniBand at 25 GB/s: the same 700 MB now takes about 28 ms, nearly *ten times* the compute of the layer it serves. Your step time is now dominated by blocking communication, and you would have been far better off with a smaller tensor-parallel degree that fit in one node and more data or pipeline parallelism across the nodes. The lesson is not subtle: the interconnect, not the arithmetic, decides whether tensor parallelism helps.

## The fit-and-latency win, quantified

So what do you get for the communication you pay? Two things, and the before-and-after makes them concrete.

![a before and after comparison showing a two gigabyte replicated layer with full matmul latency versus a tensor-parallel degree four split with half-gigabyte shards, quarter latency, and two all-reduces per layer](/imgs/blogs/tensor-parallelism-megatron-2.webp)

First, **fit**: a layer whose weights are 2 GB, replicated on every GPU under data parallelism, becomes a 0.5 GB shard per GPU at tensor-parallel degree 4. The activations for that layer shrink correspondingly, because each GPU only materializes its slice of the intermediate. A layer that would not fit at your batch size now fits, because the memory for weights and their gradients and the intermediate activations is divided by the tensor-parallel degree. Second, **latency**: the matmul that took some time $T$ on one GPU now takes roughly $T/4$ on each of four GPUs working in parallel, so per-step latency for that layer drops — minus the all-reduce overhead. This is why tensor parallelism shows up in low-latency *inference* serving too, not just training: it is the lever that cuts the time-to-first-token of a large model by spreading each layer across the GPUs of a node.

#### Worked example: a 20B model, one node

Take a 20B-parameter model on a single 8-GPU A100 node with NVLink. Under pure FSDP the weights and optimizer shard fine, but suppose the largest MLP up-projection plus its activation at your batch size peaks at 88 GB on one GPU — an OOM on the 80GB card. Apply tensor parallelism of degree 8 within the node: that layer's weight shard, gradient shard, and activation slice each divide by 8, dropping the peak to roughly 11 GB for that layer, comfortably within budget. The cost is the per-layer blocking all-reduce on NVLink, which our numbers above put at roughly a 25% step-time overhead — a price you happily pay to turn an impossible run into a possible one. Combine this with FSDP over a second dimension and data-parallel replicas over a third, and you have the 3D parallelism of the next post.

## Tensor parallelism is also the latency lever for inference

It is worth pausing on the second thing tensor parallelism buys, because it is the reason you will meet it outside of training too. When you *serve* a large model, the metric that matters is often time-to-first-token and per-token latency, and those are set by how fast a single forward pass runs — there is no batch to grow your way out of a latency requirement, and no gradient all-reduce at all. Tensor parallelism spreads each layer's matmuls across the GPUs of a node, so the forward pass of a 70B model that would take, say, 40 ms per token on one hypothetical enormous GPU instead runs in roughly a quarter of that on four GPUs (minus the all-reduce), because the matmuls are four times smaller and run in parallel. This is exactly why inference stacks like vLLM and TensorRT-LLM expose a `tensor_parallel_size` knob and default to keeping it within a node: the same NVLink-versus-everything-else physics applies, and the same rule holds — a tensor-parallel group that spans nodes will spend more time in the blocking all-reduce than it saves in matmul time. The mechanism you derived for training is the same mechanism that cuts serving latency; only the surrounding loop differs.

There is a further wrinkle that makes tensor parallelism *more* attractive for inference than for training in one respect: inference has no backward pass, so it pays one all-reduce per block instead of two, halving the communication overhead relative to a training step of the same shape. The flip side is that inference is often memory-bandwidth-bound rather than compute-bound at small batch sizes, so the matmul you are splitting may already be cheap, leaving the all-reduce a larger *fraction* of the whole. The net is that tensor parallelism helps inference latency most for large models at modest batch sizes on NVLink — precisely the regime where you reach for it in the first place.

## The activation memory tensor parallelism leaves behind

Tensor parallelism shrinks the weights, the gradients, and the *sharded* portions of the activations by the tensor-parallel degree. But it does not shrink everything, and the gap it leaves is worth understanding because it is what motivates sequence parallelism, the subject of a sibling post. Between the tensor-parallel regions — at the LayerNorm and dropout operations that sit at the block boundaries — the activation is *not* sharded; every GPU holds the full batch-by-sequence-by-hidden tensor there, because those operations were not part of the column-then-row split. For a long sequence, those replicated activation regions can dominate memory even after tensor parallelism has done its work on the matmuls.

This is the seam that [sequence and context parallelism](/blog/machine-learning/distributed-training/sequence-and-context-parallelism) closes: it shards those LayerNorm-region activations along the sequence dimension, converting the block-boundary all-reduce into an all-gather before the tensor-parallel region and a reduce-scatter after — which, as [collectives from scratch](/blog/machine-learning/distributed-training/collectives-from-scratch) showed, together move exactly the same bytes as the single all-reduce they replace. The upshot is that Megatron sequence parallelism composes with tensor parallelism at *no extra communication volume* and further cuts activation memory. When you deploy tensor parallelism for a long-context model and find the LayerNorm-region activations are your peak, sequence parallelism is the free companion that finishes the job. Treat the two as a pair: tensor parallelism for the matmul weights and their activations, sequence parallelism for the replicated regions in between.

#### Worked example: where the activation memory actually sits

Take the same 20B model at 8-way tensor parallelism, sequence length 8192, batch 4. The sharded MLP intermediate activation is divided by 8 across the group — small. But the LayerNorm-region activation, at the input and output of each block, is the full 4 × 8192 × 6144 in bf16 ≈ 400 MB *per occurrence*, replicated on every GPU, and there are two such regions per layer across 44 layers — roughly 35 GB of replicated activation memory that tensor parallelism did nothing for. Turn on sequence parallelism and that 35 GB divides by 8 to about 4.4 GB per GPU. The tensor-parallel split fixed the weights and the matmul activations; sequence parallelism fixed the seams. Neither alone is enough for a long-context run; together they fit.

## Debugging a tensor-parallel run

Tensor parallelism has its own family of bugs, distinct from the DDP gotchas in [DDP internals and gotchas](/blog/machine-learning/distributed-training/ddp-internals-and-gotchas), and they share a nasty property: the loss often still goes down, just to the wrong place, so the run *looks* healthy while quietly training a subtly incorrect model. Four failure modes account for most of the pain.

The first is a **missing or misplaced all-reduce**. If you forget the all-reduce in the row-parallel layer, each GPU trains on its own partial output as if it were the full output — the forward pass produces garbage that is nonetheless differentiable, so the optimizer happily minimizes a meaningless loss. The tell is that a tensor-parallel run diverges from a single-GPU reference run of the same config; always keep a small single-GPU baseline and assert the loss curves match for the first few steps.

The second is **unsynchronized initialization of the replicated parameters**. The LayerNorm weights, biases, and embeddings are *replicated*, not sharded, across the tensor-parallel group, so they must be identical on every rank at step zero — otherwise the group is computing with inconsistent parameters. Megatron handles this by initializing the sharded weights with rank-offset seeds (so the shards differ, as they must) but broadcasting the replicated weights from rank 0. Get the seeding backwards and you either desynchronize the replicated params or accidentally give every rank the same shard.

The third is **tensor-parallel degree exceeding the head count** for attention, which either errors out or silently gives some ranks zero heads — check that your head count is divisible by the tensor-parallel degree. The fourth is the **quiet interconnect fallback** we met in [the interconnect physics](/blog/machine-learning/distributed-training/the-interconnect-physics): if NCCL cannot use NVLink and falls back to PCIe or, worse, TCP sockets, a tensor-parallel run that should be fast becomes catastrophically slow, because every one of those blocking per-layer all-reduces now crawls. Run with `NCCL_DEBUG=INFO` once and confirm the transport is `NVLink`/`P2P`, not `NET/Socket`. The pattern across all four is the same: because tensor parallelism's correctness depends on collectives firing in exactly the right places with exactly the right group, a single misconfiguration produces a run that trains — just not the model you meant to train.

## The code

Here is a column-parallel and a row-parallel linear layer in PyTorch, stripped to the essentials so the communication is visible. These are the building blocks Megatron wraps in the `f` and `g` autograd functions.

```python
import torch
import torch.distributed as dist

class ColumnParallelLinear(torch.nn.Module):
    """Splits the weight by output columns across the TP group.
    Forward needs no comms; backward all-reduces the input gradient (the f operator)."""
    def __init__(self, in_features, out_features, tp_group):
        super().__init__()
        self.tp_group = tp_group
        world = dist.get_world_size(tp_group)
        assert out_features % world == 0
        self.weight = torch.nn.Parameter(
            torch.empty(out_features // world, in_features, device="cuda")
        )
        torch.nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        # x is the full activation, replicated on every rank; each rank
        # computes its slice of the output columns. No communication forward.
        return torch.nn.functional.linear(x, self.weight)


class RowParallelLinear(torch.nn.Module):
    """Splits the weight by input rows. Each rank produces a partial sum of the
    full output; forward all-reduces to combine (the g operator)."""
    def __init__(self, in_features, out_features, tp_group):
        super().__init__()
        self.tp_group = tp_group
        world = dist.get_world_size(tp_group)
        assert in_features % world == 0
        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_features // world, device="cuda")
        )
        torch.nn.init.normal_(self.weight, std=0.02)

    def forward(self, x_shard):
        # x_shard is this rank's slice of the intermediate activation.
        partial = torch.nn.functional.linear(x_shard, self.weight)
        # all-reduce sums the partial outputs into the full result.
        dist.all_reduce(partial, group=self.tp_group)
        return partial
```

The MLP block then composes them so the GeLU sits between a column-parallel and a row-parallel layer, exactly the derivation above:

```python
class TensorParallelMLP(torch.nn.Module):
    def __init__(self, hidden, tp_group):
        super().__init__()
        self.up = ColumnParallelLinear(hidden, 4 * hidden, tp_group)   # column split
        self.down = RowParallelLinear(4 * hidden, hidden, tp_group)    # row split
    def forward(self, x):
        h = torch.nn.functional.gelu(self.up(x))  # local GeLU, no comms
        return self.down(h)                        # one all-reduce inside .down
```

To wire up the tensor-parallel process group and place it correctly, use a device mesh. On a single 8-GPU node you make one tensor-parallel group of all 8 ranks:

```python
from torch.distributed.device_mesh import init_device_mesh

# One node, 8 GPUs, all in a single tensor-parallel group.
mesh = init_device_mesh("cuda", (8,), mesh_dim_names=("tp",))
tp_group = mesh.get_group("tp")
```

And you launch it with `torchrun`, one process per GPU, exactly as in [your first multi-GPU run](/blog/machine-learning/distributed-training/your-first-multi-gpu-run):

```bash
torchrun --standalone --nproc_per_node=8 train_tp.py
```

In Megatron-LM proper you do not hand-roll these; you set `--tensor-model-parallel-size 8` and the framework builds the column- and row-parallel layers, the `f`/`g` operators, and the process groups for you. But knowing what those flags expand to is the difference between using tensor parallelism and debugging it at 2 a.m.

To actually *measure* whether tensor parallelism is helping, time the all-reduce against the compute honestly — with a warm-up, a `torch.cuda.synchronize()` before you read the clock, and a steady-state average, exactly the timing discipline from [profiling a distributed run](/blog/machine-learning/distributed-training/why-distributed-training):

```python
import torch, torch.distributed as dist, time

def time_allreduce(numel, group, iters=50, warmup=10):
    x = torch.empty(numel, device="cuda", dtype=torch.bfloat16)
    for _ in range(warmup):           # warm up NCCL + caching allocator
        dist.all_reduce(x, group=group)
    torch.cuda.synchronize()          # never time without this
    t0 = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(x, group=group)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    gb = numel * 2 / 1e9              # bf16 = 2 bytes
    busbw = 2 * (dist.get_world_size(group) - 1) / dist.get_world_size(group) * gb / dt
    if dist.get_rank(group) == 0:
        print(f"{gb:.2f} GB  {dt*1e3:.2f} ms  bus {busbw:.0f} GB/s")
```

If the reported bus bandwidth is near your NVLink spec (hundreds of GB/s), tensor parallelism will pay off; if it is in the tens of GB/s, NCCL has fallen back to a slow transport and you must fix that before the rest of the tuning matters.

## Case studies and real numbers

A useful way to hold the empirical picture is to watch what happens to throughput as you sweep the tensor-parallel degree on a fixed model and node. From degree 1 to degree 8 within a single NVLink node, throughput per GPU stays high — you are trading a modest, well-hidden all-reduce for the ability to fit a larger model and cut per-step latency, and scaling efficiency remains in the acceptable range. The moment you push to degree 16, forcing the group across two nodes, the curve falls off a cliff: the blocking all-reduce now crosses InfiniBand on every layer, and per-GPU throughput can drop by more than half. This inverted-U — helpful up to the node boundary, harmful past it — is the single most reproducible observation about tensor parallelism, and it is why every production recipe you will read caps the tensor-parallel degree at the node's GPU count and reaches for pipeline or data parallelism to go wider.

Megatron-LM's original paper reported training an 8.3B-parameter model with 8-way tensor parallelism and sustained a high fraction of peak throughput on NVLink-connected V100s, precisely because the tensor-parallel group stayed within a node. NVIDIA's later work composing tensor, pipeline, and data parallelism to train GPT-scale models on thousands of A100s reported model-FLOP-utilization (MFU) in the roughly 50% range at the largest scales — a number that is only reachable because tensor parallelism is confined to intra-node NVLink while the thinner inter-node fabric carries only the cheaper pipeline and data-parallel traffic. The recurring empirical lesson across these reports is the same one the arithmetic predicts: tensor-parallel degree tracks the node's GPU count (8 on a DGX), and pushing it higher, across nodes, degrades throughput sharply. Treat any headline tensor-parallel MFU number as implicitly assuming NVLink; on PCIe the same configuration would look nothing like it. (Exact figures vary by model, sequence length, and cluster; take these as order-of-magnitude, and consult the primary papers for the precise setups.)

## When to reach for tensor parallelism, and when not

Tensor parallelism is a precise tool for two specific problems, and a liability everywhere else.

![a decision tree that routes a model whose layer fits and only needs throughput to data or FSDP parallelism, and a model with an oversized or latency-bound layer to tensor parallelism kept inside a node](/imgs/blogs/tensor-parallelism-megatron-7.webp)

Reach for it when a **single layer will not fit** even after FSDP has sharded the optimizer and weights across your data-parallel dimension — when the issue is one operation's peak memory, not the model's total size. Reach for it when **per-step latency dominates** and you need to spread one layer's compute over more GPUs, which is common in low-latency inference and in training when the batch cannot grow. In both cases, keep the tensor-parallel group inside one NVLink node and set the degree at or below the node's GPU count.

Do **not** reach for it when the model already fits with FSDP and DDP saturates your interconnect — you would be adding blocking, unhideable communication for no benefit. Do **not** stretch a tensor-parallel group across nodes; the per-layer all-reduce crossing InfiniBand will dominate your step time. Do **not** use a tensor-parallel degree larger than your attention head count, and do not use it as your *first* memory lever — FSDP is cheaper because its communication overlaps. The correct mental model is that tensor parallelism is the *innermost* dimension of a composed strategy: it lives inside a node, wrapped by pipeline parallelism across nodes and data parallelism across the rest, which is exactly the subject of [3D parallelism](/blog/machine-learning/distributed-training/3d-parallelism) and the decision framework in [picking a parallelism strategy](/blog/machine-learning/distributed-training/picking-a-parallelism-strategy).

## Key takeaways

- Tensor parallelism splits the **weight matrices** of a single layer across GPUs and keeps the batch whole; it buys **fit** and **lower per-step latency**, not throughput.
- The Megatron trick is **column-parallel then row-parallel**: split the first matmul by columns so the nonlinearity is local, the second by rows so one all-reduce combines the partial sums.
- A transformer layer costs **two all-reduces forward, two backward** — and nothing else crosses the wire inside the layer.
- That all-reduce is on the **critical path** and **cannot overlap** compute, unlike DDP's gradient all-reduce. This makes tensor parallelism acutely sensitive to interconnect speed.
- The communication scales with **batch × sequence × hidden**, so it is large; keep it on **NVLink inside one node**, degree ≤ GPUs-per-node (typically 8).
- Never span a tensor-parallel group across nodes; the inter-node all-reduce on every layer will dominate the step.
- Use FSDP first for pure memory pressure; reach for tensor parallelism when a **single layer** will not fit or **latency** dominates, and compose it as the innermost dimension of 3D parallelism.

## Further reading

- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) — Shoeybi et al. (2019). The original column-then-row derivation and the `f`/`g` operators.
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473) — Narayanan et al. (2021). Composing tensor, pipeline, and data parallelism and the MFU numbers at scale.
- [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198) — Korthikanti et al. (2022). Sequence parallelism as the companion that shrinks the activation memory tensor parallelism leaves behind.
- Within this series: [why distributed training](/blog/machine-learning/distributed-training/why-distributed-training) for the four walls, [the map of parallelism](/blog/machine-learning/distributed-training/the-map-of-parallelism) for where TP sits, [collectives from scratch](/blog/machine-learning/distributed-training/collectives-from-scratch) for the all-reduce cost, [the interconnect physics](/blog/machine-learning/distributed-training/the-interconnect-physics) for why placement matters, [3D parallelism](/blog/machine-learning/distributed-training/3d-parallelism) for composing it, and [the distributed training playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook) for the full decision checklist.
