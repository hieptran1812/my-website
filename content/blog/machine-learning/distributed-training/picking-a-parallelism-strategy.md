---
title: "Picking a Parallelism Strategy: A Decision Framework by Model, Cluster, and Budget"
date: "2026-07-02"
publishDate: "2026-07-02"
description: "You know the parallelism axes — now how do you choose? A fit-first decision framework that turns your model size, cluster topology, and budget into a concrete tensor-pipeline-data layout, with four worked examples and the anti-patterns to avoid."
tags:
  [
    "distributed-training",
    "multi-gpu",
    "multi-node",
    "parallelism-strategy",
    "fsdp",
    "tensor-parallelism",
    "pipeline-parallelism",
    "deep-learning",
    "ml-systems",
  ]
category: "machine-learning"
subcategory: "Distributed Training"
author: "Hiep Tran"
featured: true
readTime: 27
---

You have read the whole model-parallelism track: data parallelism and DDP, ZeRO and FSDP, tensor parallelism, pipeline parallelism, sequence parallelism, expert parallelism, and how they compose into 3D. You understand each lever. And yet, standing in front of a specific model and a specific cluster, the practical question remains unanswered: *which ones do I actually use, and at what degree?* This post is the decision framework. It takes three inputs — your model, your cluster, and your budget — and produces a concrete layout, and it does so with a single organizing principle that makes the choice almost mechanical.

![a decision tree that asks whether the model fits on one GPU and routes to a single GPU or DDP if so, and otherwise to FSDP first, then tensor parallelism if a layer will not fit, then pipeline parallelism if it is too deep to span nodes](/imgs/blogs/picking-a-parallelism-strategy-1.webp)

The organizing principle is **fit first, then speed, and climb the ladder only as far as the model forces you to.** Every additional axis of parallelism buys a capability you may need — the ability to fit a bigger model, or run it faster — at the price of communication overhead, more processes to coordinate, and a dramatically larger debugging surface. So the correct strategy is never "use the most parallelism"; it is "use the *least* parallelism that fits and hits your throughput target." By the end of this post you will be able to take any model-and-cluster pair and derive a layout from first principles, apply the rules of thumb that constrain each degree, work through four representative examples end to end, and recognize the handful of anti-patterns that cause most real-world distributed-training disasters. This is the closing post of Track C in [Distributed Training in the Trenches](/blog/machine-learning/distributed-training/why-distributed-training), and it turns everything on [the map of parallelism](/blog/machine-learning/distributed-training/the-map-of-parallelism) into a procedure.

## The three inputs

Every parallelism decision is a function of three things, and naming them precisely is half the work.

The **model** is its parameter count, its shape (wide versus deep — hidden dimension and layer count), whether it is dense or Mixture-of-Experts, and its context length. Parameter count sets the memory floor; shape decides whether tensor or pipeline parallelism is the better model-parallel axis; MoE adds an expert axis; long context adds a sequence axis. A 7B dense model at 2K context is a completely different problem from a 7B model at 256K context, even though the parameter count is identical — the second is bounded by activation memory, the first by nothing much at all.

The **cluster** is the number of GPUs, the memory per GPU, and — critically — the interconnect topology: how many GPUs share NVLink within a node, and what fabric (InfiniBand, RoCE, Ethernet) connects the nodes. The topology is not a detail; it is what makes tensor parallelism viable only within a node and pipeline parallelism the axis that spans nodes, as [the interconnect physics](/blog/machine-learning/distributed-training/the-interconnect-physics) established. A layout that is optimal on an NVLink-and-InfiniBand DGX cluster can be a disaster on a PCIe-and-Ethernet box.

The **budget** is your throughput target (tokens per second, or a wall-clock deadline), your GPU-hour ceiling, and your tolerance for complexity. This is the input people forget, and it matters because it decides *how hard to push*. If the model fits and trains fast enough with FSDP, the budget-optimal choice is to stop there even if you could squeeze out more with 3D parallelism, because the engineering time and debugging risk of the extra axes is itself a cost. The budget is what turns "what is possible" into "what is worth it."

## How the cluster changes the answer

The same model demands a different strategy on a different cluster, and the reason is entirely the interconnect topology. Consider three clusters, each with 8 GPUs, training the same 13B model.

On a **DGX-class node** (8 GPUs fully connected by NVLink and NVSwitch), FSDP full-shard is excellent: the parameter all-gather runs at ~900 GB/s, overlaps compute, and the model trains near the roofline. If you needed tensor parallelism, it too would be fine on this fabric. On a **PCIe workstation** (8 GPUs sharing PCIe, no NVLink), the same FSDP all-gather crawls at ~32 GB/s, and it may no longer overlap — you might be better off with a smaller per-GPU footprint from gradient checkpointing and a lower data-parallel degree, or accepting that this box simply cannot train a 13B model efficiently. Tensor parallelism here would be a disaster, because its blocking all-reduce on PCIe dominates. On a **multi-node Ethernet cluster** (8 GPUs split across two nodes on slow Ethernet), even the data-parallel gradient all-reduce struggles to overlap, and you would lean on `HYBRID_SHARD` to keep the heavy sharding traffic within each node and only the lighter replication across the Ethernet link.

The lesson is that "what parallelism for a 13B model" has no answer without the topology. Before you choose degrees, run `nvidia-smi topo -m` to read the connectivity matrix — which GPUs share NVLink (marked `NV#`), which share a PCIe host bridge (`PIX`/`PXB`), which cross a socket (`SYS`) — and confirm with `NCCL_DEBUG=INFO` which transport NCCL actually selects. A layout designed for NVLink and run on PCIe is one of the most common causes of "why is my training so slow," and the topology check is a five-minute step that prevents days of confusion. The interconnect is not a footnote to the parallelism decision; it is a primary input, as central as the model size.

## The five levers

Before the procedure, the levers in one table, because the decision is really about matching each lever to the problem it uniquely solves.

![a matrix comparing DDP, FSDP, tensor, pipeline, and sequence parallelism by the problem each solves, its main collective, and when to add it](/imgs/blogs/picking-a-parallelism-strategy-2.webp)

**DDP** (data parallelism) solves *too slow*: the model fits, and you want more throughput, so you replicate it and all-reduce gradients. **FSDP / ZeRO** solves *will not fit*: it shards the optimizer, gradients, and parameters across the data-parallel dimension, and its parameter all-gather is the price. **Tensor parallelism** solves *a single layer is too big or latency-bound*: it splits the matmuls, at the cost of a blocking per-layer all-reduce that confines it to NVLink. **Pipeline parallelism** solves *too deep to fit even after sharding*: it splits the layers into stages across nodes, at the cost of the bubble. **Sequence / context parallelism** solves *activations OOM at long context*: it splits the sequence, at the cost of ring or all-to-all communication. Each row is a distinct problem; the art of choosing is diagnosing *which problem you actually have* before reaching for a lever. Most teams reach for tensor or pipeline parallelism when their real problem was "state too big," which FSDP solves more cheaply.

## The fit ladder

The procedure is a ladder, and you climb it one rung at a time, stopping at the first rung where the model fits and trains fast enough.

![a stack of five rungs from a single GPU that fits, to DDP for speed, to FSDP when it will not fit, to tensor parallelism when a layer is too big, to pipeline parallelism when it is too deep](/imgs/blogs/picking-a-parallelism-strategy-3.webp)

**Rung 1 — does it fit on one GPU with room to train?** If yes, use one GPU. No distribution, no communication, no coordination bugs. A surprising number of models — anything up to a few billion parameters at modest batch and sequence — fit on an 80GB card and train fine. Do not distribute a model that does not need it; the complexity is pure cost.

**Rung 2 — does it fit but you want it faster?** Add DDP. Replicate the model across $N$ GPUs, all-reduce gradients each step (overlapped with the backward), and get close to $N\times$ throughput as long as the interconnect can absorb the gradient all-reduce. Scale $N$ until the all-reduce stops overlapping (you have saturated the link) or the global batch grows past what the learning dynamics tolerate. This is the workhorse, and for many production models it is the whole story.

**Rung 3 — does it not fit, but a single layer does?** Shard with FSDP / ZeRO. This is the crucial rung, and the most common mistake is skipping it for model parallelism. FSDP divides the per-GPU model *state* by the data-parallel degree while keeping the simple data-parallel programming model, and its communication overlaps far better than tensor parallelism's. Models up to tens of billions of parameters routinely fit with FSDP alone on a single node or a handful of nodes. Reach for `HYBRID_SHARD` (shard within a node, replicate across) when your inter-node fabric is thin. Only when FSDP is not enough do you climb higher.

**Rung 4 — does a single layer not fit, or is per-step latency dominated by one huge layer?** Add tensor parallelism, *inside a node*, degree at or below the node's GPU count. This shrinks the per-layer weights and activations and cuts per-layer latency, at the cost of the blocking all-reduce that must stay on NVLink. Compose it with FSDP over the data-parallel dimension (2D parallelism, "FSDP + TP").

**Rung 5 — is the model so deep that even tensor parallelism plus FSDP will not fit it on the nodes you have?** Add pipeline parallelism to span more nodes, with enough micro-batches ($m \gg p$) to keep the bubble small and balanced stages. Now you are at full 3D parallelism. This is the top of the ladder, and most models never reach it.

The ladder's discipline is the whole point: each rung you climb adds capability and a debugging tax, and you stop climbing the moment the model fits and trains fast enough. The teams that ship large models are not the ones using the most parallelism; they are the ones using the least that works.

## The rules of thumb, and why they hold

A handful of rules constrain the degrees, and each is a consequence of the physics we derived earlier, not a heuristic to memorize blindly.

**Tensor-parallel degree ≤ GPUs per node.** Because tensor parallelism's per-layer all-reduce is blocking and on the critical path, it must run on NVLink; the moment it crosses to InfiniBand it dominates the step. So $t$ is capped at the node's GPU count, almost always 8. This is the hardest rule; violating it is the single most common cause of a mysteriously slow run.

**Use FSDP before tensor parallelism for pure memory.** FSDP's communication overlaps with compute; tensor parallelism's does not. For the same memory reduction, FSDP costs less throughput. So when the problem is "state too big" rather than "a single layer too big," FSDP is the cheaper lever. Only when a single layer will not fit even sharded, or latency demands splitting one layer's compute, does tensor parallelism win.

**Pipeline degree needs $m \gg p$ micro-batches.** The bubble fraction is $(p-1)/(m+p-1)$, so a pipeline with too few micro-batches wastes GPUs. Keep the micro-batch count at least 4× the stage count, ideally 8×+. This couples to the global batch, below.

**Global batch size is a shared budget.** The global batch is split across the data-parallel replicas, and each replica's share is what feeds the pipeline's micro-batches. You cannot size the data-parallel and pipeline degrees independently; push $d$ too high and you starve the pipeline of micro-batches, growing the bubble. And the global batch itself is bounded above by the learning dynamics — past some critical batch size, more data parallelism stops helping convergence, so throwing GPUs at data parallelism has a ceiling. Apply the linear learning-rate scaling rule with warmup as you grow the batch, from [DDP internals and gotchas](/blog/machine-learning/distributed-training/ddp-internals-and-gotchas).

**Measure MFU and stop when it stops helping.** Every axis contributes a multiplicative efficiency; the product is your model-FLOP-utilization. When adding an axis or raising a degree stops improving measured throughput, you have hit the point of diminishing returns, and further complexity is pure cost. The profiler, not the theory, is the final arbiter.

## Diagnosing the binding constraint

Before any layout, answer one question: *what is actually running out?* The whole framework hinges on this diagnosis, because each binding constraint points to a different lever, and reaching for the wrong lever is how most bad layouts happen. There are four constraints, and they are distinguishable by a quick memory calculation.

**Total model state** ($16\Psi$ bytes for mixed-precision Adam) exceeds one GPU but each individual layer fits comfortably — the constraint is aggregate size, and the lever is FSDP, which shards state across the data-parallel dimension. This is by far the most common constraint for models in the 10–70B range, and FSDP is almost always the right first response. **A single layer** — its weights plus the activation it produces at your batch and sequence — exceeds one GPU even after FSDP has sharded the aggregate state; the constraint is per-layer size, and the lever is tensor parallelism, which splits the layer itself. This is rarer, arising for very wide models or very large batches. **Depth** — the model has so many layers that even one tensor-parallel slice of all of them will not fit on the nodes you have; the lever is pipeline parallelism, splitting the layers across more devices. **Activations** — the weights fit easily but activation memory, which scales with batch × sequence × layers, is the peak; the lever is sequence/context parallelism (for the sequence dimension) or more aggressive recomputation (for the layer dimension).

The diagnostic habit is to compute, for your specific model, batch, and sequence: the state ($16\Psi$), the largest single layer's weight-plus-activation footprint, and the total activation memory. Whichever exceeds your per-GPU budget first is your binding constraint, and it names your lever. Skipping this diagnosis and reaching for a familiar axis — usually tensor or pipeline parallelism because they feel like "real" model parallelism — when the actual constraint was aggregate state (an FSDP problem) is the single most common strategic error in the field. Diagnose first; the lever follows.

## Worked example: 1.5B on 8 GPUs

A 1.5B-parameter model on a single 8-GPU node. Memory check: $16 \times 1.5\text{B} = 24$ GB of state, which fits on one 80GB GPU with plenty of room for activations. So rung 1 is satisfied — it fits on one GPU. But you have 8 GPUs and want to train faster, so climb to rung 2: **DDP**. Replicate the model on all 8, all-reduce the 3 GB of bf16 gradients each step (overlapped, on NVLink — trivial), and get roughly 7.5× throughput. No FSDP, no tensor parallelism, no pipeline parallelism — those would add communication and complexity for a model that does not need them. The layout is `DDP, degree 8`, and you are done. The most important lesson from this example is restraint: a 1.5B model does not need model parallelism, and adding it would be an anti-pattern that buys nothing but blocking communication and a harder debugging surface. If you later find the single-GPU footprint tight at a larger batch, the next step is not tensor parallelism but FSDP's cheaper cousin — gradient checkpointing or a shard-grad-op strategy — long before any model-parallel axis enters the picture.

## Worked example: 13B on 8 GPUs

A 13B model on the same 8-GPU node. Memory check: $16 \times 13\text{B} = 208$ GB of state — far past one 80GB GPU, so rung 1 fails. Does a single layer fit? Yes — a 13B model's individual layers are a few hundred megabytes, comfortably fitting. So the problem is "total state too big," not "a single layer too big," which means rung 3: **FSDP full-shard**, not tensor parallelism. FSDP shards the 208 GB across the 8 GPUs, landing about 26 GB per GPU, which fits with room for activations. The parameter all-gather runs on NVLink and overlaps well with compute via prefetching. The layout is `FSDP FULL_SHARD, degree 8`, and it will substantially outperform an 8-way tensor-parallel layout of the same model because FSDP's communication overlaps while tensor parallelism's blocks. This example is the one people get wrong most often — reaching for tensor parallelism when FSDP is both simpler and faster.

## Worked example: 70B on 64 GPUs

A 70B model on 64 GPUs (eight 8-GPU NVLink nodes). Memory check: $16 \times 70\text{B} = 1120$ GB of state — impossible on any single GPU, and even FSDP across 64 GPUs (17.5 GB of state per GPU) leaves a single layer's just-in-time-gathered weights plus activations tight at long sequence, and the per-layer all-gather traffic becomes heavy. This is the regime where model parallelism earns its place.

![a graph where 64 GPUs across 8 nodes fan out into tensor degree 8 per node, pipeline degree 2 across nodes, and data degree 4 with FSDP, merging into a layout that fits at about 55 percent MFU](/imgs/blogs/picking-a-parallelism-strategy-5.webp)

Apply the sizing procedure. **Step 1: tensor parallelism to the node size**, $t = 8$, filling each NVLink node and shrinking each layer's weights and activations by 8. **Step 2: add pipeline parallelism to fit**, $p = 2$ — two stages spanning groups of nodes, halving the layer count each device holds; two stages suffice for 80 layers and more would grow the bubble. **Step 3: data parallelism for the rest**, $d = 64/(8 \times 2) = 4$, with FSDP over the data-parallel dimension to shard the optimizer state. Check the arithmetic: $8 \times 2 \times 4 = 64$. Trace the per-GPU memory to see it fit: the full state is $16 \times 70\text{B} = 1120$ GB. Tensor parallelism of degree 8 divides the per-layer weights and gradients ($4\Psi = 280$ GB) to 35 GB across the tensor group; pipeline parallelism of degree 2 halves the layers per device, to ~17.5 GB of weights and gradients; and the optimizer state ($12\Psi = 840$ GB), sharded across the data-parallel degree 4 with a distributed optimizer and further divided by the tensor and pipeline splits, lands around 13 GB. Sum to roughly 30–35 GB of model state per GPU, leaving 45+ GB for activations — which 1F1B (bounding stashed micro-batches to $p = 2$) and recomputation keep in budget. The multiplicative-efficiency estimate — tensor ~0.8, pipeline ~0.9, data ~0.95 — puts MFU around 55% before kernel efficiency. The layout is `TP=8 × PP=2 × DP=4 (+FSDP)`. Every axis is justified: tensor parallelism because a layer's compute and memory need splitting, pipeline because the depth needs spanning nodes, data parallelism for throughput and optimizer sharding. This is 3D parallelism from [3D parallelism](/blog/machine-learning/distributed-training/3d-parallelism), sized by the greedy procedure.

## Worked example: a long-context 7B

A 7B model, but at 256K context for long-document training, on 8 GPUs. Memory check: the *weights* are only $16 \times 7\text{B} = 112$ GB of state, which FSDP shards to about 14 GB per GPU — easy. But the *activations* at 256K context are enormous, hundreds of gigabytes on one GPU, and none of DDP, FSDP, tensor, or pipeline parallelism shrinks per-GPU activation memory as sequence grows. This is the sequence-length problem, and it needs the one axis that splits the sequence: **context parallelism**. The layout is `FSDP (for the weights) + context parallelism degree 8 (for the activations)`, with Ring Attention streaming the key-value blocks around the 8 GPUs so no GPU ever holds the full attention. This example teaches that the binding constraint dictates the axis: here it is activations, not weights, so the answer is the sequence axis, which a parameter-count-only analysis would have missed entirely. Always ask *what is actually running out* — weights, optimizer state, or activations — because each points to a different lever.

## Four models, four strategies

The four examples in one view, because the contrast is the lesson.

![a matrix mapping a 1.5B model on 8 GPUs to DDP, a 13B model to FSDP full shard, a 70B model on 64 GPUs to tensor times pipeline times data parallelism, and a long-context 7B to FSDP plus context parallelism](/imgs/blogs/picking-a-parallelism-strategy-6.webp)

Notice that the four strategies are completely different despite two of the models being close in size — the 7B long-context model needs a different axis than the 13B model, and the 1.5B and 13B models on the same hardware land on different rungs. The strategy is a function of the *binding constraint*, not the parameter count alone. This is why you cannot copy a published recipe's layout and expect it to be right for your model: the recipe encodes someone else's binding constraint on someone else's cluster. What transfers is the procedure — diagnose the constraint, climb the ladder, size greedily — not the numbers.

## Budget: the GPU-hour arithmetic

The budget input deserves its own arithmetic, because "can I afford this?" is answered by MFU, not by GPU count. The wall-clock time of a training run is the total compute (in FLOPs) divided by the achieved throughput (in FLOP/s), and the achieved throughput is the peak hardware FLOP/s times your MFU. So two layouts that both *fit* can differ by 2× in cost if one achieves 55% MFU and the other 28% — the low-MFU layout burns twice the GPU-hours for the same result. This is why the fit-first principle is also a cost principle: a simpler layout that overlaps its communication (FSDP) often achieves higher MFU than a complex one that blocks (gratuitous tensor parallelism), and so is *cheaper* as well as simpler.

#### Worked example: pricing two layouts

Suppose training your model requires $C$ total FLOPs, and you have 64 A100s at roughly 312 bf16 TFLOP/s peak each, so 64 × 312 = ~20 PFLOP/s of peak. At 55% MFU, you sustain ~11 PFLOP/s and the run takes $C / 11\text{e}15$ seconds. At 28% MFU — say, because tensor parallelism accidentally fell back to PCIe on one axis — you sustain ~5.6 PFLOP/s and the run takes nearly *twice* as long, doubling the GPU-hours and the dollar cost. At \$2 per GPU-hour, a run that should cost \$40,000 now costs \$78,000 for an identical model. The MFU is not an academic metric; it is the dollar figure. This is why the last step of the sizing procedure is always to *measure* MFU on a short run before committing the full budget — a layout that looks fine on paper but achieves half the expected MFU is a layout that will cost you double, and the profiler catches it in minutes.

The budget also bounds the *data-parallel* degree from a direction people forget: the critical batch size. Beyond some global batch, adding more data-parallel replicas (which grows the batch) stops improving convergence per step, so you spend more GPUs for no wall-clock gain — the run does the same number of *useful* steps, just with a bigger, less efficient batch. Past the critical batch size, data parallelism has hit its ceiling, and if you need more speed you must find it in higher MFU (better kernels, better overlap) rather than more replicas. Knowing roughly where that ceiling sits for your model keeps you from over-provisioning the cheapest-looking axis.

## Validate the choice — do not guess

The framework produces a *candidate* layout; the profiler confirms it. This is not optional, and it is the step that separates engineers who ship from those who theorize. Before committing a cluster-week, run a short profile of a handful of steps and check three things.

First, **is each axis achieving its expected efficiency?** Compute the multiplicative-efficiency estimate (tensor × pipeline × data), then measure the actual per-axis timing with `torch.profiler` and Nsight Systems. If the measured MFU is far below the estimate, one axis's efficiency collapsed — a pipeline bubble from too few micro-batches, a tensor all-reduce off NVLink, a data all-reduce not overlapping — and the per-axis breakdown tells you which. Second, **is the GPU actually busy?** Watch `nvidia-smi` and the profiler timeline for gaps: utilization dipping to zero between steps means the data loader is starving the GPU (a confound that has nothing to do with parallelism, from [the data pipeline at scale](/blog/machine-learning/distributed-training/why-distributed-training)), and no parallelism tuning will fix a loader bottleneck. Third, **does the single-GPU reference match?** For correctness, run a tiny single-GPU baseline and confirm the distributed loss curve tracks it for the first steps; a divergence means a correctness bug (a missing all-reduce, a bad seed, a sampler error) that the throughput numbers would never reveal.

The discipline is to treat the framework's output as a hypothesis and the profiler as the experiment. A layout that fits and looks right on paper but profiles at half the expected MFU is telling you something is wrong, and finding it before the full run is the difference between a \$40,000 job and an \$80,000 one. Guessing at scale is how clusters get wasted; measuring is how they get used.

## MoE and inference change the calculus

Two variations shift the decision enough to note. A **Mixture-of-Experts** model adds expert parallelism as a distinct axis, and it changes the diagnosis: an MoE's parameter count is dominated by the experts, so the binding constraint is usually "the experts do not fit," and the lever is expert parallelism (splitting experts across GPUs with an all-to-all), from [expert parallelism and the all-to-all bottleneck](/blog/machine-learning/distributed-training/expert-parallelism-moe). Expert parallelism composes with data parallelism (for the non-expert params) and sometimes tensor parallelism (within an expert), and its all-to-all is sensitive to the interconnect much like tensor parallelism's all-reduce — so, like tensor parallelism, expert parallelism prefers to stay on fast links and is tuned around capacity and load balance rather than raw degree. When you see a sparse model, add the expert axis to your diagnosis before reaching for the dense-model ladder.

**Inference** flips several priorities. There is no backward pass and no gradient all-reduce, so tensor parallelism costs one all-reduce per layer instead of two, and there is no data-parallel gradient sync at all — replicas are independent request servers. The binding constraint for inference is usually latency (time-to-first-token, per-token latency) and the KV-cache memory, not training-state memory. So the inference layout leans on tensor parallelism (to cut per-token latency by splitting each layer across a node's GPUs) and treats "data parallelism" as simply running more independent replicas for throughput. The fit-first, placement-aware principles still hold — tensor parallelism inside a node on NVLink — but the *reason* you reach for each axis differs. If your task is serving rather than training, re-run the diagnosis with latency and KV-cache as the constraints, and the ladder rearranges accordingly.

## Fine-tuning and LoRA shift the memory math

A large fraction of real distributed-training jobs are not pretraining from scratch but fine-tuning, and fine-tuning changes the diagnosis because it changes the memory math. Full fine-tuning has the same $16\Psi$ state as pretraining — you update every parameter, so you still carry full gradients and optimizer state — and the same ladder applies unchanged. But **parameter-efficient fine-tuning** (LoRA, QLoRA, adapters) updates only a tiny fraction of the parameters, which collapses the optimizer and gradient memory: you keep the base weights frozen (2 bytes each, no gradient, no optimizer state) and carry the $16\Psi$ overhead only for the small adapter. This often moves a model *down* the ladder. A 70B model that needs full 3D parallelism to pretrain can frequently be LoRA-fine-tuned on a single node, because the optimizer state — the 12-bytes-per-parameter term that dominated the memory — now applies only to the adapter's few million parameters, not the 70 billion base parameters.

The practical consequence: re-run the binding-constraint diagnosis for the *fine-tuning* memory profile, not the pretraining one. With QLoRA, the base weights are further quantized to 4 bits, cutting the frozen-weight memory by 4×, and a 70B model's frozen weights drop to ~35 GB — fitting on a single 80GB card with room for the LoRA adapter's tiny optimizer state and activations. At that point the whole parallelism question dissolves: you do not need FSDP, tensor, or pipeline parallelism at all, just a single GPU or plain DDP for throughput across a few adapters. This is why "how do I fine-tune a 70B model" often has a much simpler answer than "how do I pretrain a 70B model" — the memory that forced the complex layout is mostly optimizer state, and parameter-efficient methods make most of it disappear. Always ask whether the job is pretraining or fine-tuning before you reach for the heavy machinery; the answer can move you several rungs down the ladder.

## The anti-patterns

Most distributed-training disasters are one of a small set of wrong turns, and naming them is the cheapest way to avoid them.

![a before and after comparison showing tensor parallelism across nodes, pipelines with few micro-batches, and 3D parallelism when FSDP would fit, corrected to tensor parallelism inside a node, many micro-batches, and FSDP until it no longer fits](/imgs/blogs/picking-a-parallelism-strategy-4.webp)

**Tensor parallelism across nodes.** The most damaging: a tensor-parallel group spanning the inter-node fabric puts the blocking per-layer all-reduce on InfiniBand, and throughput collapses. Fix: cap $t$ at the node's GPU count. **Pipeline with too few micro-batches.** Setting a large pipeline degree with a micro-batch count near the stage count gives a 40–50% bubble. Fix: raise micro-batches to $m \gg p$, or reduce the pipeline degree. **3D when FSDP would have fit.** Reaching for tensor and pipeline parallelism on a model that FSDP alone would hold, paying blocking communication and a huge debugging surface for no benefit. Fix: climb the ladder — FSDP first, model parallelism only when forced. **Ignoring the interconnect.** Choosing degrees without checking `nvidia-smi topo -m` and confirming NCCL uses NVLink, then wondering why the run is slow. Fix: verify the topology and transport before tuning anything. **Over-provisioning data parallelism.** Pushing the data-parallel degree so high that the global batch exceeds the critical batch size (convergence stops improving) or starves the pipeline of micro-batches. Fix: treat the global batch as a shared, bounded budget. Every one of these is a violation of either the placement law or the fit-first principle.

## The sizing procedure

Put it together as a repeatable procedure you run before every large training job.

![a timeline of the sizing procedure: assess the model, set tensor parallelism to the node size, add pipeline degree until it fits, give the rest to data parallelism, check the bubble and batch budget, then measure MFU and commit](/imgs/blogs/picking-a-parallelism-strategy-7.webp)

**Assess the model** — parameter count, shape, dense or MoE, context length — and identify the binding constraint (state, a single layer, depth, or activations). **Set tensor parallelism to the node size** if a layer needs splitting, else leave $t = 1$. **Add pipeline degree until it fits**, keeping it minimal. **Give the rest to data parallelism**, with FSDP for optimizer and parameter sharding. **Check the bubble and the batch budget** — confirm $m \gg p$ and the global batch is within the critical batch size and supplies enough micro-batches. **Measure MFU and commit** — run a short profile, confirm each axis contributes its expected efficiency, and if the measured MFU is far below the estimate, find the axis whose efficiency collapsed before scaling up. The procedure is greedy and fit-first, and it is deliberately biased toward *less* parallelism: start at $t = p = 1$, add only what the binding constraint forces, and stop the moment it fits and hits your target.

## When to stop optimizing

A closing word on the budget input, because it is the one that tells you when to *stop*. Distributed training has no natural end of optimization — there is always another axis to add, another degree to tune, another few points of MFU to chase. The budget is what bounds it. If your run fits and hits its throughput target and its GPU-hour cost is acceptable, you are done, even if a more elaborate layout could go 10% faster, because the engineering time to find and debug that layout has a cost too, and a more complex layout is more likely to fail mid-run. The right amount of parallelism is the least that meets the budget, and recognizing when you have reached it — rather than optimizing indefinitely — is itself a senior skill. The [capstone playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook) turns this whole framework into a checklist you can run against any job.

Step back and notice what the whole model-parallelism track has really been teaching. Every axis — data, tensor, pipeline, sequence, expert — is a way of trading one scarce resource (a GPU's memory, or its idle time) for a plentiful one (more GPUs, connected by some link), and the *price* of every trade is communication on that link. The placement law, the bubble formula, the memory divisions, the multiplicative efficiencies: these are all just accounting for that one trade, done carefully. Once you see distributed training this way — as resource trades priced in communication — the decision framework stops being a list of rules to memorize and becomes something you can re-derive from first principles for any model and cluster you meet, including ones with hardware that does not exist yet. The rules of thumb (tensor to the node, FSDP before tensor, $m \gg p$) are the current answers for current hardware; the *reasoning* behind them is what transfers when the hardware changes. That is the real deliverable of this track: not a lookup table, but the ability to reason about the trade. A new interconnect, a new memory hierarchy, a new model shape — you re-run the same accounting and the right layout falls out. The engineer who has internalized the trade will still be choosing good layouts a hardware generation from now, when today's specific numbers are obsolete but the physics of moving bytes between processors is exactly the same.

## Key takeaways

- The strategy is a function of three inputs — **model, cluster, budget** — and one principle: **fit first, then speed, climb only as far as forced.**
- Diagnose the **binding constraint** (total state, a single layer, depth, or activations) before reaching for a lever; each constraint points to a different axis.
- Climb the **fit ladder**: 1 GPU → DDP → FSDP → +TP (in node) → +PP (across nodes). Stop at the first rung that fits and hits the target.
- **FSDP before tensor parallelism** for pure memory — its communication overlaps, tensor parallelism's blocks.
- **Tensor-parallel degree ≤ GPUs per node**; **pipeline needs $m \gg p$ micro-batches**; the **global batch is a shared, bounded budget**.
- The four worked examples land on four different strategies despite similar sizes — **copy the procedure, not the numbers.**
- Avoid the anti-patterns: TP across nodes, starved pipelines, 3D when FSDP fits, ignoring the interconnect, over-provisioned data parallelism.
- **MFU is the dollar figure**: two layouts that both fit can differ 2× in GPU-hours; measure it on a short run before committing the full budget.
- **Fine-tuning is not pretraining**: parameter-efficient methods (LoRA, QLoRA) collapse the optimizer memory and can move a model several rungs down the ladder.
- **Stop when the budget is met.** The least parallelism that works is the right amount, and every axis is a resource trade priced in communication — learn the trade, not the lookup table.

## Further reading

- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473) — Narayanan et al. (2021). The trade-offs between tensor, pipeline, and data degrees at fixed GPU count.
- [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://arxiv.org/abs/2304.11277) — Zhao et al. (2023). When FSDP alone suffices and how it composes with model parallelism.
- [An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162) — McCandlish et al. (2018). The critical batch size that bounds data parallelism.
- Within this series: the axis deep-dives ([DDP](/blog/machine-learning/distributed-training/ddp-from-first-principles), [FSDP](/blog/machine-learning/distributed-training/fsdp-in-practice), [tensor](/blog/machine-learning/distributed-training/tensor-parallelism-megatron), [pipeline](/blog/machine-learning/distributed-training/pipeline-parallelism-and-the-bubble), [3D](/blog/machine-learning/distributed-training/3d-parallelism)), [the interconnect physics](/blog/machine-learning/distributed-training/the-interconnect-physics) for the placement law, and [the distributed training playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook) for the capstone checklist.
