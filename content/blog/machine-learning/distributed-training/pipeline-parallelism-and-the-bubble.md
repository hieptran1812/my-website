---
title: "Pipeline Parallelism and the Bubble: Splitting Layers Across GPUs Without Wasting Them"
date: "2026-07-02"
publishDate: "2026-07-02"
description: "Pipeline parallelism splits a model's layers into stages across GPUs, but the naive schedule leaves GPUs idle in a bubble. Here is where the bubble comes from, the (p-1)/(m+p-1) formula, and how micro-batches and 1F1B scheduling beat it."
tags:
  [
    "distributed-training",
    "multi-gpu",
    "pipeline-parallelism",
    "gpipe",
    "1f1b",
    "model-parallelism",
    "megatron",
    "deep-learning",
    "ml-systems",
  ]
category: "machine-learning"
subcategory: "Distributed Training"
author: "Hiep Tran"
featured: true
readTime: 27
---

You have a model too deep to fit on one GPU, and the two levers you already know do not quite solve it. Data parallelism replicates the whole model, so it does nothing for a model that will not fit. Tensor parallelism, which we built in [tensor parallelism the Megatron way](/blog/machine-learning/distributed-training/tensor-parallelism-megatron), splits each layer's matmuls but must stay inside a single NVLink node, capping you at 8 GPUs of intra-layer splitting. What if the model is so deep that even after tensor-parallelizing every layer across a node, you need *more* nodes to hold all the layers? You split the layers themselves: put the first quarter of the layers on the first group of GPUs, the next quarter on the next, and so on, streaming activations down the chain. That is pipeline parallelism, and the picture below is the starting point.

![a vertical stack of four pipeline stages each holding a contiguous block of a transformer's layers on its own GPU with the loss and backward gradients at the bottom](/imgs/blogs/pipeline-parallelism-and-the-bubble-1.webp)

Pipeline parallelism is inter-layer model parallelism: it partitions the model *depth-wise* into contiguous stages, one stage per GPU (or per group of GPUs). Its great virtue, and the reason it is the axis you stretch across nodes, is that the communication between stages is tiny — just the activation tensor handed from one stage to the next over a point-to-point send, not a full all-reduce on every layer. Its great vice, and the entire subject of this post, is the *bubble*: run it naively and most of your GPUs sit idle most of the time. By the end you will be able to derive the bubble fraction $(p-1)/(m+p-1)$ from first principles, know exactly how many micro-batches you need to make it negligible, understand why the 1F1B schedule is the modern default and what it saves, write a pipeline step with explicit sends and receives, and recognize the stage-imbalance failure that silently halves your throughput. This is the second model-parallelism post in the [Distributed Training in the Trenches](/blog/machine-learning/distributed-training/why-distributed-training) Track C.

## What pipeline parallelism splits

The map is worth fixing before the mechanics. On [the map of parallelism](/blog/machine-learning/distributed-training/the-map-of-parallelism), data parallelism splits the batch, tensor parallelism splits within a layer, and pipeline parallelism splits *across* layers. A model with 48 transformer layers and a pipeline degree of 4 becomes four stages of 12 layers each; stage 0 lives on GPU 0, stage 1 on GPU 1, and so on. A forward pass feeds the input to stage 0, which computes its 12 layers and *sends the resulting activation* to stage 1, which computes its 12 and sends to stage 2, and so on until the last stage produces the loss. The backward pass runs the same chain in reverse: the last stage computes its gradients and sends the gradient of its input activation back to the previous stage, which continues the backward.

One detail that matters for correctness: the optimizer step is synchronized across the pipeline. All stages accumulate gradients across all $m$ micro-batches of a step, and only after the last micro-batch's backward has drained does every stage apply its optimizer update together, so the whole pipeline advances one consistent step at a time. A stage does not update its weights the moment it finishes a micro-batch's backward; it waits, accumulating, so that the model the next step sees is coherent across stages. This is why the drain matters and why an interrupted pipeline must roll back to a clean step boundary.

The communication is a single point-to-point send of one activation tensor between adjacent stages — for a batch of $b$ sequences of length $s$ and hidden size $h$, that is $b \times s \times h$ elements per send, in each direction, per stage boundary. Compare that to tensor parallelism, which all-reduces a full activation *twice per layer*: pipeline parallelism's communication is smaller by a factor of the layers-per-stage and uses cheap point-to-point instead of a collective. This is precisely why pipeline parallelism tolerates the thin inter-node fabric while tensor parallelism demands NVLink — a fact we will lean on when we compose them in [3D parallelism](/blog/machine-learning/distributed-training/3d-parallelism). The catch is not the communication. The catch is keeping all those stages busy.

## The naive pipeline and the birth of the bubble

Run the pipeline the obvious way — push one whole batch through — and watch what happens to utilization. At the first time step, only stage 0 has work; stages 1 through 3 have no input yet, so they sit idle. At the second step, stage 0 hands its activation to stage 1, so now stages 0 and 1 are busy and 2 and 3 idle. Only at the fourth step, once the first activation has propagated all the way to stage 3, are all four stages simultaneously busy. The pipeline had to *fill*, and during the fill the downstream stages wasted their time.

![a three by three schedule grid where the top-left-to-bottom-right diagonal fills with active forward micro-batches while the lower-left triangle of cells sits idle as the bubble](/imgs/blogs/pipeline-parallelism-and-the-bubble-2.webp)

The grid makes the waste visible: time runs left to right, stages top to bottom, and the lower-left triangle of idle cells is the pipeline *filling*. The same thing happens in reverse at the end — as the last micro-batch drains out, upstream stages finish and go idle while the downstream ones complete. Fill plus drain is the **bubble**: the fraction of the total GPU-time that is spent idle because the pipeline is not yet full or is emptying. With a single batch and $p$ stages, the pipeline is only fully utilized for a vanishing fraction of the time, and you have bought four GPUs to get barely more throughput than one. This is why nobody runs a naive pipeline; the entire art is making the bubble small.

## Deriving the bubble fraction

Put numbers on it. The key idea that tames the bubble is to split the batch into $m$ **micro-batches** and feed them into the pipeline one after another, so that while stage 0 works on micro-batch 2, stage 1 can work on micro-batch 1, and the stages overlap in time. Let each stage take one unit of time to process one micro-batch's forward (and, for now, fold the backward into the same accounting). The pipeline needs $p-1$ time units to fill — that is how long it takes the first micro-batch to reach the last stage — then all $p$ stages run in lockstep for the $m$ micro-batches, then $p-1$ units to drain.

![a stacked breakdown showing p minus one fill steps and p minus one drain steps around m steady steps, yielding a bubble fraction of p minus one over m plus p minus one](/imgs/blogs/pipeline-parallelism-and-the-bubble-3.webp)

The total time, measured in these units, is the fill plus the steady state plus the drain. The *useful* work is $m$ steps (each of the $m$ micro-batches gets processed once per stage, and in steady state all stages are busy). The *wasted* work is the fill and drain, $p-1$ steps each. So the fraction of time the pipeline is idle — the bubble fraction — is:

$$\text{bubble} = \frac{(p-1)}{m + (p-1)}$$

Read that formula, because it dictates every pipeline decision you will make. The bubble grows with the number of stages $p$ (more stages, longer fill and drain) and shrinks as the number of micro-batches $m$ grows (more micro-batches to amortize the fixed fill/drain over). The single most important consequence: **you need $m$ to be much larger than $p$** for the pipeline to be efficient. If $m = p$, the bubble is $(p-1)/(2p-1) \approx 50\%$ — half your GPUs idle. If $m = 4p$, the bubble is $(p-1)/(5p-1) \approx 20\%$. If $m = 32$ and $p = 4$, the bubble is $3/35 \approx 8.6\%$. The rule of thumb that falls out is to keep the micro-batch count at least 4× the stage count, and ideally 8× or more.

#### Worked example: sizing the micro-batches

Suppose you have a 48-layer model, a pipeline degree of $p = 8$, and a global batch of 512 sequences. How many micro-batches should you use? With $m = 8$ (micro-batch size 64), the bubble is $7/15 \approx 47\%$ — unacceptable, you are wasting nearly half your cluster. With $m = 64$ (micro-batch size 8), the bubble is $7/71 \approx 9.9\%$ — tolerable. With $m = 128$ (micro-batch size 4), the bubble is $7/135 \approx 5.2\%$ — better still, but now each micro-batch is so small that the per-stage matmuls may be too tiny to saturate the GPU, and you lose efficiency to poor kernel utilization instead of to the bubble. The sweet spot balances the bubble against kernel efficiency; in practice, micro-batches large enough to keep the GPU busy (often 1–8 sequences) with as many of them as the global batch allows. The bubble and the kernel are the two forces you are trading between.

A subtlety that trips people reading the formula: the accounting above folded forward and backward into a single "step," but a real training iteration does both, and the backward pass typically takes about twice as long as the forward (it computes two gradients — with respect to the input and with respect to the weights — for every forward matmul). This does not change the *shape* of the bubble formula, because the fill and drain still scale with $p-1$ regardless of what a "step" costs; it just means the absolute wall-clock of both the useful work and the bubble grows proportionally. What it *does* affect is the schedule design, because the moment you allow forwards and backwards to interleave — which is exactly what 1F1B does — you can start reclaiming the drain time of the forwards with the fill time of the backwards. The naive GPipe schedule, which does all forwards then all backwards, effectively pays the fill-and-drain bubble *twice* (once for the forward sweep, once for the backward sweep); 1F1B's interleaving is part of why it is strictly better than GPipe on utilization as well as memory. When you see a pipeline utilization number, ask whether it accounts for the backward being heavier than the forward — many back-of-envelope estimates quietly assume they are equal and are optimistic as a result.

It is also worth being precise about what "one unit of time" hides. Each stage's per-micro-batch time is not just its matmuls; it includes the point-to-point send and receive of the activation to the next stage. On a fast intra-node link that send is negligible next to the compute, but across a slow inter-node fabric it is not, and it lengthens every step in the steady state, not just the fill. This is the one place pipeline parallelism's communication can bite: not in volume (the point-to-point sends are small) but in *latency* on a high-latency fabric, where the fixed cost of initiating each send adds up across the many micro-batches. It is a second reason, beyond the bubble, to prefer fewer, larger stages over many tiny ones when your inter-node latency is high.

## Micro-batches shrink the bubble, at a memory cost

There is no free lunch, and the cost of more micro-batches is memory. Here is why: in the naive schedule where each stage does *all* its forwards before any backwards — the original GPipe schedule — every stage must *stash* the activations of every micro-batch it has processed, because those activations are needed for the backward pass. With $m$ micro-batches in flight, each stage holds $m$ sets of activations. More micro-batches means less bubble but more stashed activation memory.

![a before and after comparison showing four micro-batches with a 43 percent bubble and few stashed activations versus thirty-two micro-batches with an 8.6 percent bubble and more stashed activations](/imgs/blogs/pipeline-parallelism-and-the-bubble-4.webp)

The before-and-after shows the tension for $p = 4$: at $m = 4$ the bubble is a wasteful 43% but each stage stashes only 4 micro-batches of activations; at $m = 32$ the bubble drops to a comfortable 8.6% but each stage now stashes 32 micro-batches of activations, which can be gigabytes. For a large model at long sequence length, this stashed-activation memory is often the binding constraint — you would happily use more micro-batches to shrink the bubble, but you run out of memory to hold their activations. This is the problem the 1F1B schedule solves, and it is the reason 1F1B, not GPipe, is what production frameworks run.

## GPipe versus 1F1B: bounding the memory

The insight behind **1F1B** (one-forward-one-backward) is that you do not need to finish all the forwards before starting the backwards. As soon as the first micro-batch has completed its forward pass through the last stage, you can start *its* backward pass immediately, freeing its activations, while later micro-batches are still doing their forwards. The schedule interleaves: after a warmup of $p$ forwards to fill the pipe, each stage alternates one forward and one backward in steady state, then a cooldown of $p$ backwards to drain.

![a timeline showing a warmup of p forwards, a steady state alternating one forward and one backward, a cooldown of p backwards, and activation memory capped at p micro-batches](/imgs/blogs/pipeline-parallelism-and-the-bubble-5.webp)

The magic is in what this does to memory. Because a micro-batch's backward runs as soon as possible after its forward, a stage never holds more than about $p$ micro-batches' worth of activations at once — bounded by the pipeline depth, not by the total micro-batch count $m$. So 1F1B achieves the *same bubble fraction* as GPipe (the fill and drain are unchanged) while capping activation memory at $p$ instead of $m$. That is a huge win: you can now crank $m$ up to 64 or 128 to shrink the bubble without the activation memory exploding, because memory is pinned to $p$. This decoupling — bubble set by $m$, memory set by $p$ — is why 1F1B is the default schedule in Megatron-LM, DeepSpeed, and PyTorch's native pipelining. GPipe is the conceptually simpler schedule you derive the bubble from; 1F1B is the one you actually run.

There is a further refinement, **interleaved 1F1B**, where each GPU owns several *non-contiguous* stages (say, layers 1–6 and 25–30 rather than a single contiguous block). This makes each stage smaller, which shortens the fill and drain and thus shrinks the bubble further — at the cost of more point-to-point sends, since activations now cross the stage boundaries more often. It is the schedule used at the largest scales, where even a 5% bubble on thousands of GPUs is expensive.

![a matrix comparing GPipe, 1F1B, and interleaved 1F1B across bubble size, activation memory, and communication cost](/imgs/blogs/pipeline-parallelism-and-the-bubble-6.webp)

The comparison matrix is the summary to keep: GPipe and 1F1B share the same bubble, but 1F1B holds far less activation memory; interleaving trims the bubble below both at the price of extra communication. For almost everyone, 1F1B is the right default, and interleaving is the optimization you reach for only when the bubble is your measured bottleneck at very large pipeline degree.

#### Worked example: the activation-memory difference

Make the GPipe-versus-1F1B memory gap concrete. Take a 32-layer model split into $p = 8$ stages of 4 layers each, sequence length 4096, micro-batch size 2, hidden 6144, in bf16. One micro-batch's activations for a 4-layer stage — the tensors that must be stashed for backward — are on the order of a few hundred megabytes (activations scale with layers × sequence × hidden × batch, and with recomputation you can shrink this, but take ~300 MB per stashed micro-batch here). Under **GPipe** with $m = 64$ micro-batches (chosen to push the bubble down to $7/71 \approx 9.9\%$), each stage must stash all 64 micro-batches' activations before the backward sweep begins: 64 × 300 MB ≈ 19 GB of stashed activations, on top of the weights, gradients, and optimizer states — often enough to OOM. Under **1F1B** with the same $m = 64$ and the same 9.9% bubble, each stage stashes at most about $p = 8$ micro-batches: 8 × 300 MB ≈ 2.4 GB. Same bubble, one-eighth the activation memory. That 8× reduction is what lets you keep $m$ high enough to shrink the bubble without running out of memory — and it is why every serious pipeline runs 1F1B. If you find yourself unable to raise the micro-batch count because of activation memory, check first that you are actually on a 1F1B schedule and not the naive GPipe one.

#### Worked example: interleaving at large degree

Now suppose you are at $p = 16$ stages and even at $m = 128$ your bubble is $15/143 \approx 10.5\%$ — on a 1024-GPU cluster, 10% is a lot of wasted silicon. Interleaved 1F1B assigns each GPU two non-contiguous "virtual" stages, so the model is effectively cut into 32 smaller pieces while still using 16 GPUs. The fill and drain now scale with the *virtual* stage count in a way that shrinks the bubble by roughly the interleaving factor — cutting that 10.5% to around 5% — because the smaller stages fill the pipe faster. The cost is that activations now cross a stage boundary twice as often, doubling the number of point-to-point sends. On NVLink-connected or fast-IB clusters that extra communication is cheap enough that the bubble reduction wins; on a high-latency fabric it may not. This is the calculation the largest training runs make explicitly, and it is why you see "virtual pipeline stages" or "interleaving factor" as a tuning knob in Megatron configs.

## The code

Here is the shape of a 1F1B pipeline step with explicit point-to-point communication, stripped to show the send/receive structure. In practice you would use a framework, but seeing the sends makes the schedule concrete.

```python
import torch
import torch.distributed as dist

def pipeline_stage_step(stage_module, micro_batches, stage_id, num_stages, group):
    """One pipeline stage's role in a 1F1B step. stage_id 0 is the first stage."""
    is_first = stage_id == 0
    is_last = stage_id == num_stages - 1
    prev, nxt = stage_id - 1, stage_id + 1
    activations = []  # stashed for backward; bounded to ~num_stages micro-batches

    # Warmup: fill the pipe with forwards.
    for mb in micro_batches[:num_stages - stage_id]:
        x = mb if is_first else recv_tensor(prev, group)
        y = stage_module(x)
        activations.append((x, y))
        if not is_last:
            dist.send(y, dst=nxt, group=group)

    # Steady state: alternate one forward and one backward (1F1B).
    # Each backward frees the oldest stashed activation, so memory stays ~num_stages.
    #   ... (forward of the next micro-batch, then backward of the oldest) ...

    # Cooldown: drain the remaining backwards.
    #   ... (backward of the remaining stashed micro-batches) ...
```

Notice what is *not* in the sketch and yet dominates the real implementation: the steady-state loop, where each stage, on each iteration, issues the forward of the next micro-batch and the backward of the oldest stashed one, then frees that oldest micro-batch's activations. That interleaving is the entire point of 1F1B, and getting the send/receive ordering right in it — so that a stage's forward-send to the next stage and backward-send to the previous stage do not deadlock against their matching receives — is where the framework earns its keep. The warmup issues exactly `num_stages - stage_id` forwards so that by the time the steady state begins, the pipe is full and every stage has a backward available to interleave; the cooldown drains the remaining backwards symmetrically. Get the warmup count wrong and either the pipe never fills (extra bubble) or a stage tries to run a backward before its forward has completed (a hang).

The `recv_tensor` helper is a blocking `dist.recv` that must know the shape in advance (a common gotcha: the receiver has to be told the activation shape, since NCCL point-to-point does not carry metadata). In real code you would use PyTorch's `torch.distributed.pipelining` (the `PipelineStage` and schedule classes), which implement 1F1B and interleaved schedules for you:

```python
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe, Schedule1F1B

stage = PipelineStage(stage_module, stage_id, num_stages, device)
schedule = Schedule1F1B(stage, n_microbatches=64, loss_fn=loss_fn)
# Drives the full 1F1B fill/steady/drain across all ranks:
losses = schedule.step(input_microbatches)
```

And in Megatron-LM you set the pipeline degree and micro-batch count with flags, launched with `torchrun` across nodes:

```bash
torchrun --nnodes=8 --nproc_per_node=8 pretrain_gpt.py \
    --pipeline-model-parallel-size 8 \
    --micro-batch-size 4 \
    --global-batch-size 2048 \
    --num-layers-per-virtual-pipeline-stage 2   # enables interleaving
```

## Debugging a pipeline: hangs, shapes, and deadlocks

Pipeline parallelism has a distinctive family of bugs, and nearly all of them stem from the same root: point-to-point communication is a rendezvous, and a `send` on one rank must be matched by a `recv` on exactly the right rank, with exactly the right tensor shape, in exactly the right order — or the job hangs. This is the same class of failure as the collective-mismatch deadlock from [collectives from scratch](/blog/machine-learning/distributed-training/collectives-from-scratch), sharpened by the fact that pipelines issue many sends and receives per step.

The most common hang is a **shape mismatch**. NCCL point-to-point does not carry metadata, so the receiving stage must know the exact shape of the activation it is about to receive and allocate a buffer for it. If the sender's activation shape does not match what the receiver expects — because the last micro-batch is a different size, or the sequence length varies, or someone changed the hidden dimension on one stage — the `recv` either errors or, worse, silently reads the wrong number of bytes and corrupts the activation. The fix is to make shapes static across micro-batches (pad the last one, or drop it) and to assert the expected shape on the receiver. The second common hang is a **send/recv ordering deadlock**: if two adjacent stages both try to `send` before either `recv`s, and the messages are large enough not to fit in the transport's buffer, both block forever waiting for the other to receive. Frameworks avoid this with a careful ordering (odd stages receive-then-send while even stages send-then-receive, or by using non-blocking `isend`/`irecv`), but hand-rolled pipelines fall into it constantly. The third is a **rank-mapping error** in the schedule — sending to `stage_id + 1` when the process-group rank layout does not match the stage layout, so activations go to the wrong GPU. The way you localize any of these is to run with `NCCL_DEBUG=INFO` and a short timeout; a pipeline hang shows up as one rank stuck in a `recv` waiting for a `send` that never comes, and the rank that is *not* stuck is usually the one that took the wrong branch. When a pipeline "just hangs at step 0," it is almost always one of these three, and almost never a compute bug.

## The stage-imbalance war story

Here is the failure that catches teams the first time they run a pipeline at scale, and it does not show up as an error — it shows up as throughput that is mysteriously half what you predicted. A pipeline runs only as fast as its *slowest* stage, because every other stage stalls waiting for it, exactly as a factory line moves at the pace of its slowest station. If your stages are unbalanced — if stage 0 has more expensive layers than stage 3, or if the first stage also carries the embedding table and the last stage carries the output projection and loss — then the fast stages spend part of every step idle, waiting on the slow one, and that idle time is a *second* bubble on top of the fill-and-drain bubble.

The classic offenders: the embedding layer on the first stage and the final LayerNorm plus output projection plus loss computation on the last stage, which make those two stages heavier than the middle ones. The fix is to balance the partition by *compute*, not by layer count — give the first and last stages fewer transformer layers to compensate for their extra work, so every stage takes the same wall-clock time per micro-batch. Megatron and other frameworks expose knobs for uneven stage assignment precisely for this. The way you catch it is to profile per-stage step time (see [profiling a distributed run](/blog/machine-learning/distributed-training/why-distributed-training)) and look for a stage whose compute time exceeds the others; that stage is your pipeline's speed limit. A balanced pipeline is a fast pipeline, and an unbalanced one wastes GPUs quietly.

#### Worked example: what imbalance costs

Suppose a 4-stage pipeline where stages 1 and 2 each take 10 ms per micro-batch, but stage 0 carries the embedding and takes 13 ms and stage 3 carries the output projection plus loss and takes 14 ms. The pipeline runs at the pace of the slowest stage, 14 ms, so every micro-batch effectively costs 14 ms even though the average stage work is 11.75 ms. That is a 19% throughput loss *on top of* the fill-and-drain bubble — a second, hidden bubble born purely of imbalance. Rebalance by moving two transformer layers off stage 3 and one off stage 0 onto the lighter middle stages, so all four land near 11.75 ms, and you recover almost all of that 19%. The lesson: when your measured throughput is well below what the bubble formula predicts, suspect imbalance before you suspect the network, and profile per-stage before you touch anything else.

## Recomputation composes with pipelining

Pipeline parallelism's memory pressure is the stashed activations, and there is a second lever against it beyond 1F1B: **activation recomputation** (gradient checkpointing), the subject of [activation checkpointing](/blog/machine-learning/distributed-training/activation-checkpointing). Instead of stashing every intermediate activation of a stage for the backward pass, you stash only the stage's *input* and recompute the intermediates during the backward by re-running the forward. This trades extra compute (one extra forward per stage per micro-batch) for a large reduction in stashed memory, and it composes cleanly with 1F1B: the $p$ micro-batches a stage holds now each cost only their input activation, not the full set of intermediates.

The reason this matters for pipelines specifically is that it lets you raise the micro-batch count $m$ even further to shrink the bubble, because the per-micro-batch memory footprint is smaller. The combined recipe at the largest scales is: 1F1B to bound the number of in-flight micro-batches to $p$, recomputation to shrink each one's footprint, and enough micro-batches on top to drive the bubble under 10%. The extra forward from recomputation adds compute, but in a pipeline that compute partly hides in what would otherwise be bubble time, so the net cost is often smaller than the naive "33% more forward FLOPs" estimate suggests. When you read that a large model trained with a 5% pipeline bubble at a manageable memory footprint, this stack — 1F1B plus recomputation plus many micro-batches — is almost always how.

## Case studies and real numbers

The original **GPipe** paper (Huang et al., 2019) introduced the micro-batching schedule and demonstrated near-linear scaling for training giant models, establishing the $(p-1)/(m+p-1)$ bubble analysis and the memory cost of stashing all micro-batches' activations. **PipeDream** (Narayanan et al., 2019) introduced the 1F1B idea to bound activation memory and keep the pipeline full, at the cost of a weight-versioning scheme to keep the asynchronous updates consistent. **Megatron-LM's** large-scale work (Narayanan et al., 2021) combined pipeline parallelism with tensor and data parallelism to train GPT-scale models across thousands of A100s and introduced interleaved 1F1B to shrink the bubble at very large pipeline degree; their reported model-FLOP-utilization in the roughly 50% range at the largest scales depends on keeping the pipeline bubble small through many micro-batches and balanced stages. The consistent empirical lesson across all three: the bubble is real, micro-batches are the primary lever against it, 1F1B is what makes the memory affordable, and stage balance is the difference between the throughput you predicted and the throughput you get. (Exact numbers vary by model and cluster; consult the papers for precise setups, and treat percentages as order-of-magnitude.)

## Pipeline parallelism is the middle dimension

Pipeline parallelism is rarely used alone; it is one dimension of a composed layout, and understanding *where* it sits clarifies both its role and its constraints. The physics comes straight from [the interconnect physics](/blog/machine-learning/distributed-training/the-interconnect-physics): each parallelism has a communication intensity, and you place the chattiest one on the fastest link. Tensor parallelism all-reduces a full activation twice per layer, so it goes on NVLink *inside* a node. Data parallelism all-reduces gradients once per step and overlaps that with the backward, so it tolerates the fabric and goes *across* the widest set of nodes. Pipeline parallelism sends one small activation between adjacent stages — cheaper than tensor parallelism's collectives but more frequent and more latency-sensitive than data parallelism's single overlapped all-reduce — so it sits *in the middle*, spanning a modest number of nodes.

Concretely, a common layout for a large model on 64 GPUs (eight 8-GPU nodes) is tensor-parallel degree 8 within each node, pipeline-parallel degree 4 across groups of nodes, and data-parallel degree 2 across the rest. Each of the 8 GPUs in a node forms a tensor-parallel group on NVLink; adjacent nodes form the pipeline stages, passing activations over InfiniBand; and the whole thing is replicated twice for data parallelism, with the gradient all-reduce spanning the data-parallel replicas. The pipeline's point-to-point sends cross the inter-node fabric, which is exactly why pipeline parallelism — and not tensor parallelism — is the axis chosen to span nodes. This three-way composition is the subject of [3D parallelism](/blog/machine-learning/distributed-training/3d-parallelism), and it is the layout behind essentially every frontier-scale training run.

#### Worked example: utilization of a 64-GPU 3D layout

Put the pieces together and estimate the effective utilization. Tensor-parallel degree 8 on NVLink adds, say, a 25% per-layer communication overhead (from the [tensor parallelism](/blog/machine-learning/distributed-training/tensor-parallelism-megatron) worked example), so the tensor-parallel efficiency is roughly 0.8. Pipeline-parallel degree 4 with $m = 32$ micro-batches has a bubble of $3/35 \approx 8.6\%$, so its efficiency is about 0.91. Data-parallel degree 2 with a well-overlapped gradient all-reduce on the fabric costs maybe 3%, so 0.97. The efficiencies roughly multiply: $0.8 \times 0.91 \times 0.97 \approx 0.71$. So this 64-GPU layout runs at around 70% of the ideal 64× speedup — a number consistent with the ~50-60% model-FLOP-utilization frontier runs report once you also fold in kernel efficiency and the memory-bandwidth-bound operations. The point of the arithmetic is not the exact figure but the structure: each axis contributes a multiplicative efficiency, the pipeline bubble is one factor among several, and pushing any single axis too hard (a 16-way pipeline with too few micro-batches, tensor parallelism forced across nodes) drags the whole product down. This is why tuning a large run is a joint optimization, not a sequence of independent choices, and it is the mindset [picking a parallelism strategy](/blog/machine-learning/distributed-training/picking-a-parallelism-strategy) builds.

There is one interaction worth flagging now, because it surprises people: the total number of micro-batches your global batch supports has to be *shared* across the pipeline's need for many micro-batches (to shrink the bubble) and data parallelism's splitting of the batch across replicas. If you data-parallelize by 8 and your global batch is 512, each replica sees 64 sequences, and *those* 64 are what you split into pipeline micro-batches. Push the data-parallel degree too high and you starve the pipeline of the micro-batches it needs to keep the bubble small; push the pipeline degree too high and you need a larger global batch to feed it. Balancing these two is a real part of tuning a 3D layout, and it is why the [picking a parallelism strategy](/blog/machine-learning/distributed-training/picking-a-parallelism-strategy) decision framework treats the global batch size as a shared budget rather than a free parameter.

## When to reach for pipeline parallelism, and when not

Pipeline parallelism is the axis you add when a model is too deep to fit even after data and tensor parallelism have done their work, and when you must therefore span nodes — because its cheap point-to-point communication tolerates the inter-node fabric that tensor parallelism cannot.

![a decision tree routing models that fit with data or tensor parallelism away from pipeline parallelism and routing very deep models that must span nodes toward it with the micro-batch and stage-balance conditions](/imgs/blogs/pipeline-parallelism-and-the-bubble-7.webp)

The single most common anti-pattern is worth naming so you can avoid it: a team reaches for a large pipeline degree because it seems the simplest way to fit a big model — "just put different layers on different GPUs" — sets $p = 8$ with a global batch that only supports $m = 8$ micro-batches, and then wonders why their eight-GPU pipeline is barely faster than two. The bubble is $7/15 \approx 47\%$; they are running at roughly half utilization before any other overhead. The correct response is almost never "add more pipeline stages"; it is either raise the micro-batch count (if the global batch allows), shrink the pipeline degree and use FSDP or tensor parallelism for the rest, or grow the global batch. Pipeline degree is the axis people over-provision because it looks easy, and the bubble is the tax that punishes it. A useful discipline: before you set a pipeline degree, compute the bubble at the micro-batch count your global batch actually supports, and if it exceeds about 15%, you have chosen too many stages for your batch. The formula is cheap to evaluate and will save you from a run that quietly wastes a third of a cluster. Treat $(p-1)/(m+p-1)$ as a design-time check, not a post-mortem discovery.

Reach for it when the model is **very deep and must span multiple nodes**, and only when you can afford **many micro-batches** ($m \gg p$) to keep the bubble small and can **balance the stages** by compute. Do *not* reach for it when the model fits with FSDP and tensor parallelism inside your nodes — you would be adding a bubble for no reason. Do *not* use a large pipeline degree with a small micro-batch count; the bubble will eat you, and you would have been better served by more data or tensor parallelism. And always watch for stage imbalance, which turns a well-designed pipeline into a slow one. In the composed strategy of [3D parallelism](/blog/machine-learning/distributed-training/3d-parallelism), pipeline parallelism is the *middle* dimension: tensor parallelism inside a node, pipeline parallelism across a few nodes, data parallelism across the rest — a layout we make precise in [picking a parallelism strategy](/blog/machine-learning/distributed-training/picking-a-parallelism-strategy).

## Key takeaways

- Pipeline parallelism splits the model's **layers** into stages across GPUs, communicating with cheap **point-to-point** sends, so it tolerates the inter-node fabric.
- The **bubble** is the idle time while the pipeline fills and drains; its fraction is $(p-1)/(m+p-1)$.
- **Micro-batches are the primary lever**: keep $m \gg p$ (at least 4×, ideally 8×+) to make the bubble negligible.
- More micro-batches cost **stashed-activation memory** in the naive GPipe schedule; **1F1B** bounds that memory to $p$ micro-batches while keeping the same bubble — which is why it is the default.
- **Interleaved 1F1B** shrinks the bubble further at the cost of more point-to-point sends; reach for it only at very large pipeline degree.
- A pipeline runs at the speed of its **slowest stage**; balance the partition by **compute**, not layer count, and watch the embedding and loss stages.
- Add pipeline parallelism only when the model must **span nodes** and you can afford many micro-batches; it is the middle dimension of 3D parallelism.

## Further reading

- [GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism](https://arxiv.org/abs/1811.06965) — Huang et al. (2019). The micro-batching schedule and the bubble analysis.
- [PipeDream: Generalized Pipeline Parallelism for DNN Training](https://arxiv.org/abs/1806.03377) — Narayanan et al. (2019). 1F1B and bounding activation memory.
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473) — Narayanan et al. (2021). Interleaved 1F1B and composing pipeline with tensor and data parallelism.
- Within this series: [why distributed training](/blog/machine-learning/distributed-training/why-distributed-training) for the four walls, [the map of parallelism](/blog/machine-learning/distributed-training/the-map-of-parallelism) for where PP sits, [tensor parallelism the Megatron way](/blog/machine-learning/distributed-training/tensor-parallelism-megatron) for the intra-layer companion, [3D parallelism](/blog/machine-learning/distributed-training/3d-parallelism) for composing it, and [the distributed training playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook) for the full decision checklist.
