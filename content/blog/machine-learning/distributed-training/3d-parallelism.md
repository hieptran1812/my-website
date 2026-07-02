---
title: "3D Parallelism: Composing Tensor, Pipeline, and Data Parallelism on a Cluster"
date: "2026-07-02"
publishDate: "2026-07-02"
description: "No single parallelism scales a frontier model. Here is how tensor, pipeline, and data parallelism compose onto a device mesh, why each axis goes on a different link tier, and how to size the three degrees for a real cluster."
tags:
  [
    "distributed-training",
    "multi-gpu",
    "multi-node",
    "3d-parallelism",
    "device-mesh",
    "megatron",
    "deepspeed",
    "deep-learning",
    "ml-systems",
  ]
category: "machine-learning"
subcategory: "Distributed Training"
author: "Hiep Tran"
featured: true
readTime: 27
---

You have a 175-billion-parameter model and 512 GPUs, and none of the three parallelisms you have learned so far can train it by itself. Data parallelism replicates the whole model, and 175B parameters in mixed precision with Adam is over a terabyte of state per GPU — it does not fit, not even close. Tensor parallelism, from [tensor parallelism the Megatron way](/blog/machine-learning/distributed-training/tensor-parallelism-megatron), splits each layer but must stay inside a node, capping at 8 GPUs. Pipeline parallelism, from [pipeline parallelism and the bubble](/blog/machine-learning/distributed-training/pipeline-parallelism-and-the-bubble), splits the layers across nodes but a single stage of a 175B model is still enormous. The answer that trained GPT-3, Megatron-Turing, PaLM, and every other frontier model is not to pick one — it is to **compose all three at once**. That composition is 3D parallelism, and the figure below is its skeleton.

![a graph where a node for 512 GPUs branches into tensor degree 8, pipeline degree 8, and data degree 8 which all merge into a single named three-dimensional device mesh](/imgs/blogs/3d-parallelism-1.webp)

The arithmetic is exact and unforgiving: the total number of GPUs equals the product of the three degrees, $N = t \times p \times d$, where $t$ is the tensor-parallel degree, $p$ the pipeline degree, and $d$ the data-parallel degree. For 512 GPUs you might choose $t = 8$, $p = 8$, $d = 8$, and every GPU belongs to exactly one tensor group, one pipeline stage, and one data replica simultaneously. By the end of this post you will be able to lay a model onto a device mesh, know which axis belongs on which interconnect and why, compute the per-GPU memory and communication of a composed layout, size the three degrees for a real cluster from first principles, and recognize when 3D parallelism is warranted versus when it is over-engineering. This is the capstone of the model-parallelism track in [Distributed Training in the Trenches](/blog/machine-learning/distributed-training/why-distributed-training), and it ties together every axis on [the map of parallelism](/blog/machine-learning/distributed-training/the-map-of-parallelism).

## The problem no single axis solves

Be precise about why each axis alone fails at frontier scale, because the failure modes are what dictate the composition. Data parallelism's problem is memory: it never reduces the per-GPU model state, so a model that does not fit on one GPU does not fit under any amount of data parallelism. FSDP and ZeRO, from [ZeRO and FSDP: the memory model](/blog/machine-learning/distributed-training/zero-and-fsdp-the-memory-model), soften this by sharding the optimizer, gradients, and parameters across the data-parallel dimension — and for many models FSDP alone is enough. But at the very largest scales, even FSDP's sharded parameters, when gathered just-in-time for a layer's forward, can exceed what one GPU can hold for a single enormous layer, and the all-gather traffic on every layer becomes its own bottleneck. You need to also shrink the *layer*.

Tensor parallelism shrinks the layer, but its blocking per-layer all-reduce confines it to one NVLink node — 8 GPUs. Eight-way tensor parallelism on a 175B model still leaves each tensor-parallel group holding roughly 175B / 8 ≈ 22B parameters' worth of state, which is still too much for the group's GPUs to hold across all layers at once. So you also need to split the *layers* across more devices — pipeline parallelism. And once you have split layers across pipeline stages and matmuls across tensor groups, you still want to process more data in parallel for throughput, which is data parallelism. Each axis solves exactly what the others cannot: tensor parallelism shrinks the layer, pipeline parallelism shrinks the number of layers per device, data parallelism multiplies throughput. Composed, they turn an impossible run into a routine one.

## The device mesh is the core abstraction

The tool that makes composition tractable is the **device mesh**: a logical arrangement of your $N$ GPUs into a multi-dimensional grid, where each dimension corresponds to one parallelism axis, and each GPU has a coordinate along every axis.

![a three by three grid of GPUs where each row is a tensor-parallel group sharing NVLink and each column is a data-parallel group whose gradients all-reduce across the fabric](/imgs/blogs/3d-parallelism-2.webp)

Think of it as a coordinate system. A GPU's position $(t_i, p_j, d_k)$ tells it exactly which group it belongs to for each collective: it all-reduces tensor-parallel activations with the GPUs that share its $p_j$ and $d_k$ but differ in $t_i$; it sends pipeline activations to the GPU one step along the $p$ axis; and it all-reduces gradients with the GPUs that share its $t_i$ and $p_j$ but differ in $d_k$. The mesh turns "which GPUs talk to each other for which operation" from a bookkeeping nightmare into a lookup along a named axis. PyTorch's `DeviceMesh` and Megatron's process-group construction both build exactly this structure, deriving the per-axis communication groups from the mesh coordinates. The grid figure shows a two-dimensional slice — tensor groups along the rows, data replicas along the columns — and the full mesh simply adds the pipeline axis as a third dimension.

The reason the mesh matters beyond bookkeeping is that it makes the *placement* decision explicit. Which physical GPUs should form a tensor-parallel group? Which should be pipeline neighbors? The mesh coordinates map directly to physical hardware, and getting that mapping right is the difference between a fast run and a slow one.

The mesh also generalizes cleanly to more than three dimensions, which is why "3D" is really shorthand rather than a hard limit. A Mixture-of-Experts model adds a fourth axis — **expert parallelism**, from [expert parallelism and the all-to-all bottleneck](/blog/machine-learning/distributed-training/expert-parallelism-moe) — as its own mesh dimension, with its all-to-all running along that axis; a long-context model adds **sequence/context parallelism** as another. The frontier is quietly moving to 4D and 5D meshes for MoE and long-context models, but the principle is unchanged: each axis is a mesh dimension, each has a communication profile, and each is placed on the link tier its collective can afford. Once you understand the three-dimensional case and the placement law that governs it, adding a fourth or fifth dimension is more of the same reasoning, not a new idea. The device mesh is the abstraction that makes that extensibility possible — it does not care how many dimensions you give it.

## The placement law

Here is the single most important design principle in 3D parallelism, and it comes straight from [the interconnect physics](/blog/machine-learning/distributed-training/the-interconnect-physics): **put the chattiest parallelism on the fastest link.**

![a stack showing tensor parallelism placed on intra-node NVLink, pipeline parallelism across a few nodes on InfiniBand, and data parallelism across all nodes, under the rule chattiest on fastest link](/imgs/blogs/3d-parallelism-3.webp)

Each axis has a distinct communication profile. Tensor parallelism all-reduces a full activation *twice per layer*, on the critical path, unhideable — the heaviest, most latency-sensitive traffic — so it goes on intra-node NVLink, and its degree is capped at the node's GPU count (8). Pipeline parallelism sends one activation between adjacent stages, small and point-to-point, so it tolerates the inter-node fabric and spans a few nodes. Data parallelism all-reduces gradients once per step, and that all-reduce *overlaps* with the backward pass (from [DDP from first principles](/blog/machine-learning/distributed-training/ddp-from-first-principles)), so it is the most forgiving of a slow link and spans the widest set of nodes. The stack captures the ordering: NVLink for tensor, InfiniBand for pipeline, the broad fabric for data.

The mapping to physical hardware follows directly. On a cluster of 8-GPU NVLink nodes: the 8 GPUs *within* a node form a tensor-parallel group; *adjacent nodes* form pipeline stages, passing activations over InfiniBand; and the *remaining nodes* form data-parallel replicas. Get this backwards — put tensor parallelism across nodes, or pipeline within a node while tensor parallelism crosses the fabric — and the blocking per-layer all-reduce crawls over InfiniBand and your throughput collapses, exactly as the tensor-parallelism post's worked example showed. The placement law is not a preference; it is a hard consequence of the communication physics, and every production 3D layout obeys it.

## Why one axis is never enough

Make the necessity concrete with the memory arithmetic.

![a before and after comparison showing a 175 billion parameter model needing over 1120 gigabytes per GPU under data parallelism versus about 35 gigabytes per GPU once split three ways](/imgs/blogs/3d-parallelism-4.webp)

#### Worked example: fitting a 175B model

A 175B-parameter model in mixed precision with Adam needs, by the $(2+2+12)\Psi$ accounting from [the ZeRO memory model](/blog/machine-learning/distributed-training/zero-and-fsdp-the-memory-model), about $16 \times 175\text{B} = 2800$ GB of state — call it roughly 1120 GB per GPU if you tried to replicate it (the exact figure depends on precision choices, but it is far past any single GPU). Now compose. Tensor parallelism of degree 8 divides the per-layer weights, gradients, and their activations by 8. Pipeline parallelism of degree 8 divides the *number of layers* each device holds by 8. Data parallelism of degree 8, with the optimizer states sharded across it à la ZeRO-1, divides the optimizer state by a further 8. The per-GPU state drops by roughly the product of the sharding factors, landing in the tens of gigabytes — around 35 GB per GPU in a typical configuration, comfortably within an 80GB card with room for activations. The 512-GPU layout does not just make the model *faster*; it makes it *possible*, and no two of the three axes alone would have sufficed. That is the point of the before-and-after: each axis contributes a division, and only their product brings a terabyte-scale model onto 80GB cards.

## What one training step actually does

To reason about performance, trace the collectives a single step fires across all three axes.

![a timeline showing tensor-parallel all-reduces every layer, pipeline point-to-point sends between stages, the backward pass in reverse, one overlapped data-parallel gradient all-reduce, and the synchronized optimizer step](/imgs/blogs/3d-parallelism-5.webp)

Within each pipeline stage, the forward pass runs its layers, and each layer fires a tensor-parallel all-reduce on NVLink (two, counting the attention and MLP blocks). As each micro-batch finishes a stage, its activation is sent point-to-point to the next pipeline stage over InfiniBand. The backward pass runs the same pattern in reverse — tensor all-reduces per layer, pipeline sends between stages — accumulating gradients across all micro-batches. Only after the last micro-batch's backward has drained does the data-parallel gradient all-reduce fire, once, across the data-parallel replicas, overlapped with the tail of the backward. Then every GPU applies its optimizer step together. The timeline shows the three communication rhythms coexisting: the fast, frequent tensor all-reduces inside the node; the modest pipeline sends between nodes; and the single, overlapped data-parallel all-reduce at the end. Understanding which collective lands where is what lets you predict — and debug — a 3D run's throughput.

## The three axes side by side

The comparison is worth having in one place, because sizing decisions come down to these three columns.

![a matrix comparing tensor, pipeline, and data parallelism by what each splits, the collective it fires, and the link tier it belongs on](/imgs/blogs/3d-parallelism-6.webp)

Tensor parallelism splits the layer's matmuls, fires an all-reduce per layer, and belongs on NVLink. Pipeline parallelism splits the layers, fires a point-to-point send per stage boundary, and spans a few nodes. Data parallelism splits the batch, fires one overlapped all-reduce per step, and spans the whole cluster. Read the matrix top to bottom as a gradient of communication intensity — heaviest and most local at the top, lightest and most global at the bottom — and the placement law falls out of it directly. This table is the mental model to carry into any sizing decision.

Counting the collectives per step drives the point home. For a model with $L$ layers, tensor parallelism fires roughly $4L$ all-reduces per step (two per layer, forward and backward) — hundreds or thousands of collectives, which is why they must be the cheapest per-collective, on NVLink. Pipeline parallelism fires on the order of $2mp$ point-to-point sends (one per micro-batch per stage boundary, both directions) — many, but each tiny. Data parallelism fires exactly *one* all-reduce per step (or a handful of bucketed ones), the largest single message but the rarest and the only one that overlaps. The counts explain the placement: thousands of tiny-latency-sensitive collectives on the fastest link, one big overlapped one on the slowest. Any layout that inverts this ordering is paying its heaviest collective traffic on its worst link.

## The memory and communication accounting

Put the two together. The per-GPU **memory** of a 3D layout is roughly the full model state divided by the product of the sharding factors: tensor parallelism divides the per-layer state by $t$, pipeline parallelism divides the layer count by $p$, and data-parallel optimizer sharding (ZeRO-1) divides the optimizer state by $d$. Activation memory is trickier — pipeline parallelism's 1F1B schedule caps stashed activations at $p$ micro-batches, and activation recomputation shrinks each, as covered in the pipeline post. The **communication** is the sum of the three rhythms, but only the tensor all-reduce is fully on the critical path; the pipeline sends mostly overlap with the next stage's compute, and the data-parallel all-reduce overlaps with the backward. This is why a well-tuned 3D run can hit 50–60% model-FLOP-utilization despite all that communication: most of it hides, and the placement law ensures the un-hideable part (tensor) runs on the fastest link.

To make the memory division concrete, walk the 175B case term by term. The full state, by the $(2+2+12)\Psi$ accounting, is 2 bytes of bf16 weights, 2 bytes of bf16 gradients, and 12 bytes of fp32 Adam state (master weight, momentum, variance) per parameter — 16 bytes total, so $16 \times 175\text{B} = 2800$ GB. Under tensor parallelism of degree 8, each GPU holds $1/8$ of every layer's weights and gradients, so the weight-plus-gradient portion ($4\Psi = 700$ GB) divides to about 87 GB across the tensor group. Under pipeline parallelism of degree 8, each GPU holds only $1/8$ of the *layers*, dividing that 87 GB by a further 8 to about 11 GB of weights and gradients per GPU. The optimizer state ($12\Psi = 2100$ GB) is sharded across the data-parallel dimension of degree 8 with a distributed optimizer, so it divides by all three factors — tensor, pipeline, and data — landing around 4 GB per GPU. Sum the pieces and you are at roughly 15–20 GB of model state per GPU, leaving 60+ GB on an 80GB card for activations and the framework's overhead. Each factor of division came from a different axis, and removing any one would push a term back over budget. That is the arithmetic behind the before-and-after figure, and it is worth doing by hand once so the composition stops feeling like magic.

#### Worked example: the MFU of a 512-GPU layout

Estimate the efficiency the way you would before committing a cluster-month of compute. Each axis contributes a multiplicative efficiency. Tensor parallelism of degree 8 on NVLink adds roughly a 20–25% per-layer communication overhead that does not fully hide, so call its efficiency about 0.8. Pipeline parallelism of degree 8 with, say, $m = 64$ micro-batches has a bubble of $7/71 \approx 9.9\%$, so about 0.90. Data parallelism of degree 8 with a well-overlapped, optimizer-sharded all-reduce costs a few percent, so about 0.95. The efficiencies roughly multiply: $0.8 \times 0.90 \times 0.95 \approx 0.68$. Fold in kernel efficiency and memory-bandwidth-bound operations (another ~0.75–0.85 factor) and you land in the 50–58% model-FLOP-utilization range that frontier runs report. The value of the estimate is not the exact number; it is that it tells you *which axis to fix first*. If your measured MFU is well below this, the multiplicative structure says to find the axis whose efficiency collapsed — a pipeline bubble from too few micro-batches, a tensor all-reduce that fell back off NVLink, a data-parallel all-reduce that is not overlapping — rather than blaming the whole system at once.

## Activation memory in a 3D layout

The memory arithmetic so far counted *model state* — weights, gradients, optimizer. But at frontier scale, *activation* memory is often the tighter constraint, and 3D parallelism shapes it in a way worth spelling out because it is where runs actually OOM. Each axis touches activations differently. Tensor parallelism shrinks the activation of the split matmuls by $t$ (each GPU holds its slice), but leaves the LayerNorm-region activations replicated — the seam that [sequence parallelism](/blog/machine-learning/distributed-training/sequence-and-context-parallelism) closes. Pipeline parallelism does *not* shrink per-layer activations, but its 1F1B schedule bounds the *number* of in-flight micro-batches whose activations are stashed to $p$ rather than $m$. Data parallelism does nothing for activations at all — each replica processes its own micro-batches with full activations.

So the activation budget on one GPU is roughly: (the activations of the layers in this pipeline stage, tensor-split by $t$) × (up to $p$ stashed micro-batches under 1F1B) — and then **activation recomputation** divides the per-micro-batch footprint further by only stashing each stage's input and recomputing the intermediates. The practical consequence is that a real 3D recipe almost always turns on recomputation, because it is the lever that keeps activation memory affordable once model state has been divided three ways. When a 3D run OOMs, the culprit is usually activations, not weights — the weight division is easy to compute and gets budgeted correctly, while activations, which scale with batch and sequence and stage depth, are the ones that surprise you. The first response to a 3D OOM is almost always "turn on (more aggressive) recomputation," and only after that "add another pipeline stage."

#### Worked example: the global-batch coupling

Make the batch coupling concrete, because it is the sizing mistake that quietly wastes clusters. Suppose 512 GPUs, a layout of $t = 8$, $p = 8$, $d = 8$, and a global batch of 1024 sequences. Data parallelism splits that batch across the 8 replicas, so each replica processes 128 sequences per step. Those 128 are what the pipeline splits into micro-batches. With micro-batch size 2, that is $m = 64$ micro-batches — comfortably more than $p = 8$, giving a bubble of $7/71 \approx 9.9\%$. Good. Now suppose someone, chasing throughput, raises the data-parallel degree to 32 (and drops pipeline to 2 to keep the product at 512). Each replica now sees $1024/32 = 32$ sequences; with the same micro-batch size 2, that is only $m = 16$ micro-batches. The pipeline bubble at $p = 2$ is fine ($1/17 \approx 6\%$), but the deeper point is the *coupling*: had they kept $p = 8$, the 32-sequence replica could supply only $m = 16$ micro-batches for 8 stages, a bubble of $7/23 \approx 30\%$ — a disaster. The global batch is a shared budget across the data-parallel split and the pipeline's micro-batch appetite, and you cannot size $d$ and $p$ independently. The fix when the pipeline is starved is to raise the global batch (if the learning dynamics allow) or lower the data-parallel degree — never to just accept the bubble.

## FSDP as the data-parallel dimension

A practical refinement that appears in almost every modern 3D layout: the data-parallel dimension is usually not plain DDP but **FSDP** (or ZeRO), sharding the optimizer, gradients, and often parameters across the data-parallel replicas. This is why the memory arithmetic above divided the optimizer state by $d$ — that division is FSDP's doing, not DDP's. Composed this way, the layout is sometimes called **HSDP** (hybrid sharded data parallel): shard within one group and replicate across groups, exactly the `HYBRID_SHARD` strategy from [FSDP in practice](/blog/machine-learning/distributed-training/fsdp-in-practice).

The composition has a subtlety worth internalizing. FSDP's parameter all-gather (to reconstruct a full layer's weights just in time) happens along the data-parallel dimension, while tensor parallelism's all-reduce happens along the tensor dimension, and the two must not collide. The device mesh keeps them separate: FSDP operates over the `dp` axis, tensor parallelism over the `tp` axis, and because a GPU has independent coordinates on each, the collectives run on disjoint process groups. In practice you wrap each tensor-parallel module and then apply FSDP over the data-parallel mesh dimension, and the framework composes the two shardings. When you see a recipe that says "FSDP + TP" or "2D parallelism," this is it — the data-parallel dimension is doing double duty as a memory-sharding dimension, and it is often the whole story for models in the tens of billions of parameters, with pipeline parallelism added only when the model grows past what FSDP + TP can hold on the available nodes.

## The order of the mesh dimensions

A detail that bites people the first time: the *order* in which you list the mesh dimensions determines which physical GPUs land in which group, and getting it wrong silently violates the placement law. GPUs are numbered globally, and typically the GPUs within a node are contiguous (global ranks 0–7 on node 0, 8–15 on node 1, and so on). The mesh's *last* dimension varies fastest across those global ranks. So if you want the tensor-parallel group to be the 8 GPUs of a node — which the placement law demands — the tensor dimension must be the *last* (fastest-varying) dimension of the mesh, so that ranks 0–7 form one tensor group on one node. List the dimensions as `(dp, pp, tp)` with `tp` last, and tensor groups fall on nodes; list them as `(tp, pp, dp)` with `dp` last, and your tensor-parallel groups get scattered across nodes, forcing the blocking per-layer all-reduce over InfiniBand and destroying throughput.

This is why the `init_device_mesh` call in the code below lists `("dp", "pp", "tp")` in that order — it is not arbitrary. The rule to remember: **the axis you want on the fastest link goes last in the mesh**, because last varies fastest and fastest-varying maps to physically adjacent GPUs. When a 3D run is inexplicably slow despite correct degrees, a mis-ordered mesh — tensor parallelism accidentally spanning nodes — is one of the first things to check, right after confirming NCCL is actually using NVLink.

#### Worked example: a 70B model on 64 GPUs

A more common scale: a 70B model on 64 GPUs (eight 8-GPU nodes). A good layout is $t = 8$ (the full node on NVLink), $p = 2$ (two pipeline stages spanning node groups), $d = 4$ (four data-parallel replicas). Check the arithmetic: $8 \times 2 \times 4 = 64$. Tensor parallelism puts each layer's matmuls across the 8 GPUs of a node; pipeline parallelism splits the 80 layers into two stages of 40, each stage occupying four nodes; data parallelism replicates that two-stage pipeline four times. Per-GPU weights land around a few gigabytes after the tensor and pipeline splits, optimizer states shard across the data-parallel dimension, and activations are bounded by 1F1B plus recomputation. The reasoning for each choice: $t = 8$ because that is the node size and the placement law caps tensor parallelism there; $p = 2$ because two stages suffice to fit the layer count and more would grow the bubble; $d = 4$ because it uses the remaining GPUs for throughput. This is the layout you would actually run, and it generalizes: max out tensor parallelism to the node, add just enough pipeline to fit, and give the rest to data parallelism.

## The code

PyTorch's `DeviceMesh` expresses the three axes and derives the process groups:

```python
from torch.distributed.device_mesh import init_device_mesh

# 64 GPUs = tp(8) x pp(2) x dp(4). Order matters: the last dim varies fastest,
# so tp GPUs are contiguous (same node), then pp, then dp.
mesh = init_device_mesh(
    "cuda", (4, 2, 8), mesh_dim_names=("dp", "pp", "tp")
)
tp_group = mesh["tp"].get_group()   # 8 GPUs in a node, on NVLink
pp_group = mesh["pp"].get_group()   # pipeline neighbors, across nodes
dp_group = mesh["dp"].get_group()   # data-parallel replicas
```

You then wrap the model with tensor parallelism along `tp`, split it into pipeline stages along `pp`, and shard/replicate along `dp`. The FSDP-over-the-data-parallel-dimension part looks like this — you hand FSDP the `dp` sub-mesh so its all-gather runs only along that axis, disjoint from the tensor all-reduce:

```python
from torch.distributed.fsdp import fully_shard   # FSDP2

# Apply tensor parallelism first (parallelize_module over the tp mesh),
# then shard the tensor-parallel model over the dp dimension only.
for block in model.layers:
    fully_shard(block, mesh=mesh["dp"])   # all-gather runs on the dp axis
fully_shard(model, mesh=mesh["dp"])
```

Because `mesh["dp"]` and `mesh["tp"]` are disjoint process groups, FSDP's parameter all-gather (on `dp`) and tensor parallelism's activation all-reduce (on `tp`) never contend — the mesh is what keeps the two shardings orthogonal. In Megatron-DeepSpeed the same layout is a set of flags, launched across nodes with `torchrun`:

```bash
torchrun --nnodes=8 --nproc_per_node=8 pretrain_gpt.py \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 2 \
    --num-layers 80 \
    --micro-batch-size 2 \
    --global-batch-size 1024 \
    --use-distributed-optimizer      # shards optimizer over the data-parallel dim
```

The framework reads these three degrees, builds the mesh and its process groups, wires the tensor-parallel `f`/`g` operators inside each node, sets up the 1F1B pipeline schedule across the pipeline groups, and installs the gradient all-reduce (or ZeRO shard) across the data-parallel groups. The whole composition is expressed in three numbers whose product must equal your GPU count.

## Case studies and real numbers

The 3D recipe is the empirical backbone of frontier training. **GPT-3** (175B) was trained with a combination of tensor and pipeline model parallelism plus data parallelism on a large A100 cluster. **Megatron-Turing NLG** (530B) reported an explicit $t \times p \times d$ layout on thousands of A100s and documented the tensor-inside-node, pipeline-across-node placement. **PaLM** (540B) used a related model-and-data-parallel composition on TPU pods. The **Megatron-LM scaling paper** (Narayanan et al., 2021) is the canonical reference: it lays out the 3D decomposition, the placement law, and the interleaved-1F1B refinement, and reports model-FLOP-utilization in the roughly 50% range at the largest scales — a number that is only reachable because tensor parallelism stays on NVLink, pipeline bubbles are kept small with many micro-batches, and the data-parallel all-reduce overlaps. The consistent lesson: the degrees are chosen by the placement law (tensor = node size, pipeline to span nodes, data for the rest), and the MFU you achieve is the product of each axis's efficiency. (Exact configurations and numbers vary; consult the papers, and treat percentages as order-of-magnitude.)

It is instructive to see how the choices shift with model shape. A model that is *wide but not deep* (large hidden dimension, moderate layer count) leans on tensor parallelism, because the expensive thing is the per-layer matmul, and may need little or no pipeline parallelism. A model that is *deep* (many layers) leans on pipeline parallelism, because the constraint is fitting the layer count. A model at *modest scale* (tens of billions) often needs neither tensor nor pipeline parallelism and runs on FSDP alone across a handful of nodes. This is why you cannot copy a frontier lab's exact $t \times p \times d$ and expect it to be optimal for your model — the degrees are a function of the model's shape, your node size, and your interconnect, not a universal constant. The Megatron paper's own ablations show throughput varying substantially with the split of a fixed GPU count between tensor and pipeline degrees, with a clear optimum that depends on the model and cluster. The transferable artifact is the *method* — max tensor to the node, pipeline to fit, data for the rest, respect the batch coupling — not any particular triple of numbers.

The other lesson these runs teach, quietly, is about reliability at scale. A 512- or 2048-GPU run touches every failure mode in this series at once: a single straggler GPU slows a tensor group, a slow node stalls a pipeline stage, a NCCL timeout on any axis can hang the whole job, and a hardware failure on one of thousands of GPUs is a near-certainty over a multi-week run. This is why the frontier reports devote as much text to checkpointing, elastic restart, and monitoring as to the parallelism itself — topics the reliability track of this series takes up. 3D parallelism is what makes the run *possible*; fault tolerance is what makes it *finish*.

## Debugging a 3D run

A 3D run has the failure modes of all three axes at once, plus a few that only emerge from their composition, and the discipline is to *localize which axis* is the problem before touching anything. The composition's own bugs cluster into three kinds.

The first is **a mis-ordered or mis-mapped mesh**, described above: the degrees are correct but the physical placement violates the placement law, so tensor parallelism crosses nodes. The symptom is throughput far below the MFU estimate with no obvious single culprit; the diagnosis is to run with `NCCL_DEBUG=INFO` and confirm the tensor-parallel group's transport is NVLink/P2P, not NET/Socket. The second is **a hang at the first step**, which in a 3D run is almost always a collective or point-to-point mismatch on one of the three axes — a pipeline send with no matching receive, a tensor all-reduce that one rank skipped, a process group built with the wrong ranks. Because there are three axes of collectives, the trick is to bisect: disable pipeline parallelism (set $p = 1$) and see if it hangs; if not, the bug is in the pipeline schedule; if so, test tensor and data parallelism similarly. The third is **an imbalance that stalls one axis** — a slow pipeline stage, a straggler GPU in a tensor group, an uneven data shard — which shows up as one rank consistently behind and the rest waiting on it.

The general method is the multiplicative-efficiency estimate turned into a diagnostic: compute the MFU each axis *should* contribute, measure the actual per-axis timing with the profiler (see [profiling a distributed run](/blog/machine-learning/distributed-training/why-distributed-training)), and find the axis whose measured efficiency falls short of its prediction. That axis is your bottleneck. This is far more effective than staring at an aggregate throughput number, because a 3D run's throughput is a product of three efficiencies and the aggregate cannot tell you which factor collapsed. Debugging 3D is, more than anything, the skill of attributing a slowdown to the right axis.

## Sizing the three degrees

Turn all of this into a procedure you can follow.

![a decision tree with three steps: set tensor parallelism to the node size, add pipeline degree until the model fits with many micro-batches, then give the remaining GPUs to data parallelism](/imgs/blogs/3d-parallelism-7.webp)

The sizing follows the placement law in order. **Step 1: set tensor parallelism to the node size.** Almost always $t$ equals the GPUs per node (8 on a DGX), because that is the largest tensor-parallel group that stays on NVLink, and tensor parallelism is your most powerful per-layer memory and latency lever. **Step 2: add pipeline degree until the model fits.** Increase $p$ just enough that the per-GPU layer count, after the tensor split, fits in memory — and no more, because every pipeline stage adds to the bubble, and you must be able to afford $m \gg p$ micro-batches. **Step 3: give the remaining GPUs to data parallelism.** Whatever GPUs are left after $t$ and $p$ are fixed become the data-parallel degree, $d = N / (t \times p)$, multiplying throughput and (with optimizer sharding) reducing per-GPU optimizer state. The tree captures this order: TP first, PP to fit, DP for the rest. It is a greedy procedure, and it is what the [picking a parallelism strategy](/blog/machine-learning/distributed-training/picking-a-parallelism-strategy) post generalizes into a full decision framework.

There is one coupling to respect, flagged in the pipeline post: the global batch size is a shared budget. Data parallelism splits the global batch across $d$ replicas, and each replica's share is what you split into pipeline micro-batches. If you push $d$ too high, each replica sees too few sequences to supply the $m \gg p$ micro-batches the pipeline needs, and the bubble grows. Sizing a 3D layout is therefore a joint optimization over $t$, $p$, $d$, and the global batch — not four independent choices.

## When 3D parallelism is overkill

The most important thing to say about 3D parallelism is that most people do not need it. If your model fits with **FSDP alone** — and models up to tens of billions of parameters often do on a single 8-GPU node or a handful of nodes — then FSDP is simpler, has fewer failure modes, and is easier to debug. Reach for 3D parallelism only when the model is large enough that FSDP's per-layer all-gather becomes a bottleneck or a single gathered layer will not fit, which in practice means the 100B+ regime or long-context training of large models. Adding tensor and pipeline parallelism to a model that fits with FSDP buys you nothing but blocking communication, more processes to coordinate, and a much harder debugging surface. The progression is a ladder: single GPU, then DDP, then FSDP, then FSDP plus tensor parallelism inside a node, and only at the top, when nothing else fits, the full 3D composition. Climb it only as far as your model forces you to.

A useful gut check: count the ways your run can break at each rung. Single-GPU has essentially none of the distributed failure modes. DDP adds gradient-sync correctness and sampler bugs. FSDP adds wrapping-policy and sharded-checkpoint subtleties. Tensor parallelism adds the placement law and the replicated-init requirement. Pipeline parallelism adds the bubble, stage balance, and send/recv deadlocks. Full 3D adds the mesh-ordering trap and the batch coupling on top of *all* of the above. Each rung multiplies your debugging surface, and the debugging is genuinely harder because a symptom on one axis can be caused by a misconfiguration on another. The engineering discipline is to stop climbing the moment the model fits with acceptable throughput — every additional axis you add is a permanent tax on every future debugging session, paid whether or not the run has a problem. The teams that ship large models are not the ones that use the most parallelism; they are the ones that use the least parallelism that works.

## Key takeaways

- No single parallelism scales a frontier model; **3D parallelism composes tensor, pipeline, and data** with $N = t \times p \times d$.
- The **device mesh** gives every GPU coordinates along each axis and derives the per-axis communication groups.
- The **placement law** is the core principle: tensor on intra-node NVLink (degree = node size), pipeline across a few nodes, data across the rest.
- Each axis contributes a **memory division** (tensor per-layer, pipeline layer-count, data optimizer-shard); only their product fits a 175B model on 80GB cards.
- A single step fires **tensor all-reduces per layer, pipeline sends per stage, and one overlapped data all-reduce**; most communication hides, which is how a 3D run reaches 50–60% MFU.
- **Size greedily**: max tensor parallelism to the node, add pipeline just enough to fit, give the rest to data parallelism — respecting the global-batch coupling.
- The mesh **generalizes past three dimensions**: expert and sequence parallelism are simply more axes, placed by the same law.
- **Do not reach for 3D if FSDP fits.** Climb the ladder only as far as the model forces; the complexity tax of 3D is real, and the teams that ship large models use the least parallelism that works.

## Further reading

- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473) — Narayanan et al. (2021). The canonical 3D decomposition, placement law, and interleaved 1F1B.
- [Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B](https://arxiv.org/abs/2201.11990) — Smith et al. (2022). An explicit $t \times p \times d$ layout at 530B scale.
- [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311) — Chowdhery et al. (2022). Model-and-data-parallel composition on TPU pods.
- Within this series: [tensor parallelism the Megatron way](/blog/machine-learning/distributed-training/tensor-parallelism-megatron) and [pipeline parallelism and the bubble](/blog/machine-learning/distributed-training/pipeline-parallelism-and-the-bubble) for the axes, [ZeRO and FSDP](/blog/machine-learning/distributed-training/zero-and-fsdp-the-memory-model) for the data-parallel memory model, [the interconnect physics](/blog/machine-learning/distributed-training/the-interconnect-physics) for the placement law, [picking a parallelism strategy](/blog/machine-learning/distributed-training/picking-a-parallelism-strategy) for the full framework, and [the distributed training playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook) for the capstone checklist.
