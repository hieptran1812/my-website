---
title: "Overlapping Compute and Communication: The Trick That Makes Distributed Training Scale"
date: "2026-07-02"
publishDate: "2026-07-02"
description: "The single idea that decides whether adding GPUs helps: communication only costs you wall-clock if it sits on the critical path. Learn how CUDA streams hide the all-reduce under the backward pass, how DDP bucketing and FSDP prefetch actually schedule it, what breaks overlap, and how to read a profiler to prove it."
tags:
  [
    "distributed-training",
    "multi-gpu",
    "cuda-streams",
    "communication-overlap",
    "ddp",
    "fsdp",
    "nccl",
    "pytorch",
    "gpu",
    "ml-systems",
  ]
category: "machine-learning"
subcategory: "Distributed Training"
author: "Hiep Tran"
featured: true
readTime: 40
---

Two teams train the same 1.5-billion-parameter transformer on the same 8-GPU A100 node. Same model code, same batch size, same NCCL, same PyTorch. Team A reports 7.4x speedup over one GPU — 93% scaling efficiency, near textbook. Team B reports 3.1x — they added eight times the hardware and got barely three times the throughput, and their finance lead is asking why the GPU bill tripled for a 3x return. Nobody changed the math. Nobody changed the interconnect. The gradients are the same gradients, the all-reduce moves the same number of bytes over the same NVLink. The only thing that differs between the two runs is *when* the communication happens relative to the compute — whether the all-reduce runs **during** the backward pass, hidden underneath work the GPU was already doing, or **after** it, as a serial phase that adds straight to the step time.

That is the entire subject of this post. Distributed training moves an enormous amount of data between GPUs: a gradient all-reduce every step in data parallelism, two parameter all-gathers and a reduce-scatter in FSDP, activations across pipeline stages, blocking all-reduces inside every tensor-parallel layer. Every one of those transfers takes real milliseconds. But milliseconds of communication only turn into milliseconds of *wall-clock* if they land on the **critical path** — the sequence of operations the GPU must finish before it can take the next step. If the communication overlaps with compute the GPU has to do anyway, it costs you almost nothing. If it doesn't, you pay for every byte. The figure that opens this post draws exactly that difference: the same work, scheduled two ways, one adding 40% to every step and one adding 5%.

Overlap is not a minor optimization. It is the difference between distributed training that scales and distributed training that doesn't. The reason DDP gets 90%+ efficiency on a well-connected node is not that its communication is cheap — a 3 GB gradient all-reduce is never cheap — it is that virtually all of that communication happens while the backward pass is still computing. Take the overlap away and the same job collapses to the 50–60% efficiency that makes people conclude, wrongly, that "multi-GPU just doesn't scale for our model." It scales fine. The comms was on the critical path.

By the end of this post you will be able to write the one equation that predicts your step time from compute and communication; explain how a single GPU runs compute and NCCL at the same time using CUDA streams; deepen your understanding of exactly how DDP bucketing and FSDP prefetch schedule that overlap, and tune the knobs (`bucket_cap_mb`, `backward_prefetch`, `limit_all_gathers`) that control it; recognize the four situations that break overlap — a data dependency like tensor parallelism's blocking all-reduce, messages too small to be bandwidth-bound, a CPU too slow to launch kernels, and a comms bill simply larger than the compute available to hide it — and apply the right fix to each; and read a `torch.profiler` or Nsight Systems trace to *measure* how much of your communication is actually exposed. This is post 20 of the [Distributed Training in the Trenches](/blog/machine-learning/distributed-training/why-distributed-training) series, and it sits underneath almost everything else in it: DDP, FSDP, pipeline, and 3D parallelism all live or die by whether their comms overlaps.

## 1. The one idea: communication only costs you if it doesn't overlap

Start with the simplest possible model of a training step. The GPU does two kinds of work: **compute** (the matmuls of forward and backward, the optimizer update) and **communication** (moving bytes to and from other GPUs over NVLink or InfiniBand). Call the time they take $T_\text{compute}$ and $T_\text{comm}$. The question that decides your scaling efficiency is: do these two happen one after the other, or at the same time?

![a two-column comparison of a serial schedule where compute finishes before communication begins against an overlapped schedule where the all-reduce runs during the backward pass](/imgs/blogs/overlapping-compute-and-communication-1.webp)

If they are **serial** — the GPU finishes all its compute, then starts communicating, then waits for the transfer to finish before the next step — your step time is the sum:

$$T_\text{step}^\text{serial} = T_\text{compute} + T_\text{comm}$$

If they **overlap perfectly** — the communication runs concurrently with compute, using hardware that would otherwise sit idle — your step time is the *maximum* of the two, because the shorter one hides entirely inside the longer one:

$$T_\text{step}^\text{overlap} = \max(T_\text{compute}, T_\text{comm})$$

That gap between the sum and the max is the whole game. When $T_\text{compute} = 120$ ms and $T_\text{comm} = 80$ ms, the serial step is 200 ms and the overlapped step is 120 ms — a 40% reduction in step time, for free, from nothing but scheduling. In practice you never get *perfect* overlap; some fraction of the communication is unavoidably **exposed**, meaning it sits on the critical path with no compute to hide behind. So the honest formula is:

$$T_\text{step} = T_\text{compute} + T_\text{exposed}, \qquad T_\text{exposed} = \max(0,\; T_\text{comm} - T_\text{overlappable})$$

Your engineering goal is to drive $T_\text{exposed}$ toward zero — to arrange for as much of $T_\text{comm}$ as possible to fall inside a window where the GPU is busy computing something that does not depend on the bytes in flight.

### Why exposed comms destroys scaling efficiency

Scaling efficiency is the number your capacity planning actually cares about. Run on $N$ GPUs, measure the speedup $S = T_1 / T_N$ relative to one GPU, and the efficiency is $E = S / N$. In data-parallel training, each GPU does roughly the *same* per-GPU compute regardless of $N$ (it processes its own slice of the batch), so $T_\text{compute}$ is basically fixed as you scale. What grows with $N$ is the communication. If that communication is exposed, it adds directly to every step, and efficiency falls as:

$$E = \frac{T_\text{compute}}{T_\text{compute} + T_\text{exposed}}$$

Plug in the two schedules from the figure. With comms fully exposed ($T_\text{exposed} = 80$ ms on top of 120 ms compute), efficiency is $120 / 200 = 60\%$. With overlap driving the exposed part down to 8 ms, efficiency is $120 / 128 \approx 94\%$. Same hardware, same bytes — a 34-point swing in efficiency, purely from whether the all-reduce runs during backward or after it. On a 64-GPU run that 34 points is the difference between finishing a training run in nine days and finishing it in fourteen, and the difference translates almost linearly into the cloud bill. This is why overlap is not a tuning detail you get to the end of; it is the first-order term in whether distributed training was worth doing.

### Predicting overlap before you launch

You can estimate whether your comms will hide *before* you run anything, and it is worth doing because it tells you whether overlap is even the right lever. You need two numbers. $T_\text{compute}$ is the per-step compute time: roughly the model's FLOPs per step divided by the GPU's achievable throughput (a fraction of its peak — an A100 does about 312 dense bf16 TFLOP/s of peak, and you might sustain 40–50% of it). $T_\text{comm}$ is the collective time: the byte volume divided by the effective bandwidth. For a DDP all-reduce of a gradient of size $S$, the ring algorithm moves about ${2S}$ bytes of bus traffic, and the effective all-reduce bandwidth on NVLink is often 60–80% of the raw link bandwidth. Compute both, and the comparison decides your fate:

$$\text{overlap ceiling} = \frac{T_\text{compute}}{\max(T_\text{compute}, T_\text{comm})}$$

If $T_\text{comm} \ll T_\text{compute}$, overlap can hide essentially all of it and you should expect near-linear scaling — spend your effort on the bucket/prefetch tuning that gets you *close* to the ceiling. If $T_\text{comm} \approx T_\text{compute}$, overlap helps but you are on a knife's edge; a slightly slower link or a slightly smaller batch tips you into exposed comms. If $T_\text{comm} > T_\text{compute}$, no schedule saves you — the ceiling itself is below one, and you need to reduce comms or lengthen compute (section 5, cause 4). This one back-of-envelope, done before you burn a single GPU-hour, tells you which of those three worlds you live in.

The rest of this post is about the *mechanism* that makes overlap possible (CUDA streams), the two places you meet it most (DDP bucketing and FSDP prefetch), the four ways it breaks, and how to prove with a profiler that it is working.

## 2. CUDA streams: how one GPU runs compute and comms at the same time

The natural first question is: how can a single GPU compute and communicate *at the same time*? Isn't it one chip doing one thing? The answer is the **CUDA stream**, and it is the primitive that every overlap in distributed training is built on.

A CUDA stream is an ordered queue of GPU work. Operations placed on the *same* stream execute in the order you enqueued them, one strictly after the previous finishes — that is the sequential semantics your single-GPU PyTorch code relies on. But operations on *different* streams have **no ordering guarantee between them**; the GPU is free to run them concurrently, and it will, as long as there is hardware to do so and no data dependency forces one to wait for the other. A modern GPU has the hardware: multiple copy engines (DMA units) that move bytes over NVLink and PCIe independently of the compute cores, and enough streaming multiprocessors that a communication kernel and a compute kernel can occupy the chip at once. Put your matmuls on one stream and your NCCL all-reduce on another, and the GPU overlaps them.

![a dataflow graph of a compute stream and a communication stream running concurrently on one GPU with an event enforcing the single ordering dependency between them](/imgs/blogs/overlapping-compute-and-communication-2.webp)

The figure draws the model. There is a **compute stream** carrying the forward and backward kernels, and a separate **communication stream** carrying the NCCL collectives. NCCL, by design, runs its collectives on their own dedicated CUDA stream precisely so they can overlap with whatever your default (compute) stream is doing. The two streams proceed independently — except at the one point where correctness demands a dependency. You cannot all-reduce a gradient that has not been computed yet; the all-reduce of layer 24's gradient must wait until layer 24's backward has produced it. That single ordering constraint is enforced with a **CUDA event**: the backward kernel records an event when the gradient is ready, and the NCCL all-reduce on the comm stream waits on that event before it starts. Everything *else* — all the compute for layers 23, 22, 21 that does not depend on layer 24's reduced gradient — keeps running on the compute stream while the all-reduce proceeds on the comm stream. That is overlap, expressed exactly: two streams, concurrent, joined only where data actually flows between them.

Here is the mechanism made concrete, stripped to its essentials. You rarely write this by hand — DDP and FSDP do it for you inside their C++ engines — but seeing it once makes everything after it legible:

```python
import torch
import torch.distributed as dist

# A dedicated stream for communication, separate from the default compute stream.
comm_stream = torch.cuda.Stream()

def overlapped_step(grad_ready_tensor, more_compute):
    # 1) Compute happens on the default stream and records when the gradient is ready.
    ready = torch.cuda.Event()
    ready.record()  # marks the point in the default stream after grad is produced

    # 2) The all-reduce runs on comm_stream, but must not start until 'ready' fires.
    with torch.cuda.stream(comm_stream):
        comm_stream.wait_event(ready)          # the ONE dependency
        dist.all_reduce(grad_ready_tensor)      # NCCL kernel on the comm stream

    # 3) Meanwhile, unrelated compute keeps running on the default stream,
    #    concurrently with the all-reduce above. No dependency, no wait.
    out = more_compute()                        # overlaps with the all-reduce

    # 4) Only when we NEED the reduced gradient do we synchronize the streams.
    torch.cuda.current_stream().wait_stream(comm_stream)
    return out
```

Three things in that snippet are the whole story. `comm_stream.wait_event(ready)` is the data dependency — the all-reduce cannot start before the gradient exists. `more_compute()` on the default stream is the overlap — it runs *concurrently* with the all-reduce because nothing ties them together. And `wait_stream(comm_stream)` at the end is the **join**: the point where you finally need the reduced result, so the compute stream blocks until the comm stream catches up. If you place that join too early — right after launching the all-reduce, before doing any independent compute — you have serialized the two streams and thrown away all the overlap. Where the join sits is, quite literally, where overlap ends.

One honesty note, because vendors gloss over it. Compute and communication are not perfectly free of each other. NCCL's collective kernels run on the GPU's streaming multiprocessors too (for the reduction arithmetic and to drive the transfers), so a large all-reduce and a large matmul do contend for SMs, and you will see the compute kernel run slightly slower while the collective is in flight. The overlap is real and enormous, but it is "the shorter one mostly hides under the longer one," not "the shorter one becomes literally zero." That is why the exposed-comms term in the equation above is $\max(0, T_\text{comm} - T_\text{overlappable})$ and not simply zero — there is always a little residue. On NVLink, where a per-block or per-bucket collective is small relative to the compute, that residue is a few percent. On a thin inter-node link it can be the whole step. We will come back to this.

Two hardware details make the overlap sturdier than "two kernels sharing one chip" sounds. First, the actual byte movement over NVLink or PCIe is driven by dedicated **copy engines** (DMA units) that are physically separate from the SMs, so the data transport itself runs alongside compute without stealing math throughput; the SM contention is only for the small amount of reduction arithmetic and kernel bookkeeping NCCL does. Second, communication streams are usually created at **high priority**. CUDA streams have priorities, and NCCL's stream typically sits above the compute stream so that when a collective is ready to run, the scheduler admits its kernel promptly instead of leaving it queued behind a long compute kernel. That matters because a collective that fires late — after its overlap window has already closed — becomes exposed even though nothing structurally prevented hiding it. High-priority comm streams are how the framework keeps "ready to overlap" and "actually overlapping" the same thing. You do not set this yourself in normal DDP/FSDP use; the point is only that when you read a trace and see the NCCL kernel start the instant its dependency clears, this is why.

With the streams model in hand, the two workhorse strategies — DDP and FSDP — are just two different, very good answers to the same question: *what independent compute can we schedule on the default stream to hide the collective on the comm stream?*

## 3. DDP: firing all-reduce while backward is still running

Data-parallel training (DDP) has exactly one collective per step: an all-reduce that averages the gradients across all ranks so every replica applies the identical update. That all-reduce is the entire communication cost of DDP, and on a 1.5B model in bf16 it moves a 3 GB gradient — a ring all-reduce transfers about ${2S}$ bytes of bus traffic, so roughly 6 GB crosses the fabric per GPU. On NVLink that takes tens of milliseconds; on PCIe or across nodes, much more. If DDP did that all-reduce *after* the backward pass finished, it would be pure exposed comms, and DDP would scale terribly. It does not, because of one observation about how backpropagation works, drawn in the figure below.

![a left to right timeline of a backward pass where gradient buckets fill in reverse layer order and each full bucket fires its all-reduce while earlier layers keep computing](/imgs/blogs/overlapping-compute-and-communication-3.webp)

Backpropagation computes gradients in **reverse layer order**. The gradient for the last layer is ready first, then the second-to-last, and so on down to the first layer. The gradient of layer 24 exists long before the gradient of layer 1. So there is no reason to wait for *all* the gradients before you start communicating: the moment layer 24's gradient is ready, you can begin all-reducing it while the backward pass keeps churning through layers 23, 22, 21. By the time backward reaches layer 1, most of the network's gradients have already been all-reduced and are sitting averaged, ready for the optimizer step. The all-reduce has been overlapped with the backward compute — which is exactly the independent work on the default stream that hides the collective on the comm stream. In the timeline, each bucket's all-reduce (comm stream) runs directly beneath the backward compute of the *earlier* layers (compute stream).

### Bucketing: the granularity knob

DDP does not fire one all-reduce per parameter — that would be thousands of tiny latency-bound collectives, each too small to use the fabric's bandwidth, spending all its time in launch and handshake overhead. Instead DDP **buckets**: it groups consecutive parameters into buffers of a target size (default about 25 MB, set by `bucket_cap_mb`) and fires the all-reduce for a whole bucket at once, the instant every gradient in that bucket is ready. A 25 MB transfer is large enough to be bandwidth-bound — the fixed launch cost is amortized over a meaningful payload — but small enough that many buckets fill during a single backward pass, giving overlap plenty of chances to fire early. It is the Goldilocks size between "one giant all-reduce at the very end, zero overlap" and "a thousand tiny all-reduces, all latency, no bandwidth."

The full derivation of the byte volume and the autograd-hook machinery lives in [DDP From First Principles](/blog/machine-learning/distributed-training/ddp-from-first-principles); here the point is the *overlap*, and the key consequence is the **last bucket**. Whatever bucket finishes last has, by definition, no more backward compute to hide behind — the backward pass is over. Its all-reduce is therefore exposed: it sits on the critical path, and its duration is your irreducible $T_\text{exposed}$ for DDP. Everything before it overlapped; the last bucket did not. This immediately tells you how to think about tuning.

Here is the DDP wrap with the two knobs that matter for overlap, plus `static_graph` which lets DDP commit to a fixed bucket order:

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)  # bind THIS process to its own GPU before NCCL init

model = build_model().cuda(local_rank)
model = DDP(
    model,
    device_ids=[local_rank],
    bucket_cap_mb=25,              # the overlap granularity: 25 MB buckets
    gradient_as_bucket_view=True, # grads are written INTO the bucket, no extra copy
    static_graph=True,            # fixed graph -> stable bucket order, faster path
)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for batch in loader:
    optimizer.zero_grad(set_to_none=True)
    loss = model(batch).loss
    loss.backward()   # autograd hooks fire the overlapped, bucketed all-reduces
    optimizer.step()  # by here, all but the last bucket already finished reducing
```

`gradient_as_bucket_view=True` deserves a callout because it interacts with overlap. Normally each parameter has its own `.grad` tensor and DDP must *copy* each gradient into the bucket's contiguous buffer before the all-reduce, then copy the reduced result back. Those copies are extra work on the compute stream that can delay when a bucket becomes "ready" to fire. With `gradient_as_bucket_view=True`, each parameter's `.grad` is a *view* directly into the bucket buffer, so the gradient is written straight into the bucket during backward — no copy in, no copy out. The bucket is ready the instant its last gradient lands, so the all-reduce fires earlier and overlaps more. It also saves the memory of a second gradient copy. Turn it on for essentially every job.

There is one more interaction worth naming because it trips people up: **gradient accumulation**. When you accumulate gradients over several micro-batches before stepping the optimizer, you do *not* want DDP to all-reduce on every micro-batch — that would communicate the same gradients repeatedly and destroy your efficiency. DDP's `no_sync()` context manager suppresses the all-reduce for the accumulation micro-steps, letting gradients pile up locally, and fires the bucketed, overlapped all-reduce only on the final micro-step where you actually step. From the overlap perspective this is a gift: the accumulation micro-steps are pure compute with zero comms, and the single sync at the end still overlaps with that last backward pass. The net effect is that a job with accumulation factor four does its all-reduce once per four backwards, so the amortized comms-to-compute ratio drops by 4x and exposed comms nearly vanishes. If a small model is comms-bound because its per-step compute is too short to hide the all-reduce (cause 4, later), turning up gradient accumulation is often the simplest fix — it lengthens the compute between syncs.

### Tuning bucket size: the two failure modes

The bucket size is the single overlap knob for DDP, and it fails in both directions.

![a comparison table of small versus medium versus large gradient bucket sizes and how each affects the number of all-reduces and the quality of overlap](/imgs/blogs/overlapping-compute-and-communication-4.webp)

**Too small** (say `bucket_cap_mb=1`) and you fragment the gradient into thousands of tiny all-reduces. Each one carries a fixed cost — kernel launch, NCCL handshake, ramp-up before it reaches full bandwidth — that a 1 MB payload cannot amortize. The collectives become **latency-bound**: their total time is dominated by per-message overhead, not by bytes moved. Worse, launching thousands of them can saturate the CPU's ability to enqueue kernels, so the comm stream starves waiting for the CPU. Overlap looks like it should be great (lots of small pieces to interleave) but the aggregate comms time balloons and much of it ends up exposed anyway.

**Too big** (say `bucket_cap_mb=500` on a model with only 3 GB of gradient) and you have a handful of enormous buckets. The last one is now huge, and it cannot overlap because backward has finished — a giant exposed all-reduce on the critical path. You have also delayed the *first* all-reduce until a large chunk of backward has completed, wasting the early overlap window. The sweet spot — usually 25–50 MB on NVLink, sometimes larger on very fast fabric with very large models — is where buckets are big enough to be bandwidth-bound but small enough that the last one is a small fraction of the step.

| Bucket size | Number of all-reduces | Each collective | Overlap quality | Failure mode |
|---|---|---|---|---|
| Too small (1 MB) | Thousands | Latency-bound, sub-bandwidth | Fragmented, CPU-launch-bound | Comms balloons; much exposed |
| Sweet spot (25–50 MB) | Tens to low hundreds | Bandwidth-bound | Fills through backward, tiny last bucket | ~5% exposed on NVLink |
| Too big (500 MB) | A handful | Bandwidth-bound but huge | Late first fire, huge last bucket | Big exposed last all-reduce |

#### Worked example: from 40% comms to 5% exposed on NVLink

Take the 1.5B model, 8×A100 on NVLink, and suppose a modest per-GPU batch so the per-step compute is short: $T_\text{compute} = 120$ ms. The gradient is 3 GB; a ring all-reduce moves about 6 GB of bus traffic, and at an effective all-reduce bandwidth of roughly 75 GB/s on this configuration that is about $T_\text{comm} = 80$ ms. Run it **serially** — all-reduce after backward — and each step is 120 + 80 = 200 ms. The all-reduce is 40% of the step. Scaling efficiency versus a compute-only ideal is $120 / 200 = 60\%$, so eight GPUs deliver about a 4.8x speedup. This is Team B from the intro.

Now turn on proper bucketing (25 MB buckets, `gradient_as_bucket_view=True`). The gradient splits into about 120 buckets. As backward runs, buckets fill and fire, their all-reduces overlapping with the backward of earlier layers. By the time backward ends, 119 buckets have completed; only the last one — about 25 MB, roughly 8 ms of all-reduce — is exposed. Step time drops to 120 + 8 = 128 ms. Exposed comms is now 8/128 ≈ 6% of the step. Efficiency is $120 / 128 \approx 94\%$, a 7.5x speedup on 8 GPUs. This is Team A. Nothing changed but *when* the 80 ms of communication happened: 72 of those 80 ms moved off the critical path and under the backward pass. That is the 34-point efficiency swing from section 1, made concrete.

## 4. FSDP: prefetch the next layer's parameters under this layer's compute

FSDP (Fully Sharded Data Parallel, PyTorch's ZeRO) shards parameters, gradients, and optimizer states across ranks, so no rank holds the whole model. The price is that FSDP must *reconstruct* each layer's full parameters just before it runs, with an `all_gather`, then free them afterward — and it does a `reduce_scatter` of gradients in the backward pass. That is three collectives per step where DDP had one, and 1.5x the byte volume. On paper FSDP should be much slower than DDP. On NVLink it lands within a few percent, and the reason is the same as DDP's: **overlap**. But FSDP's overlap has a different shape, because its collectives are on the *forward* critical path, not just the backward one.

The key move is **prefetching**: while the current layer computes, FSDP all-gathers the *next* layer's parameters. When the current layer finishes, the next layer's parameters are already resident and it starts immediately — no stall waiting for the gather.

![a left to right timeline of two transformer blocks under sharded data parallelism where the next block parameters all-gather while the current block computes](/imgs/blogs/overlapping-compute-and-communication-5.webp)

The timeline shows it: while block $i$ computes on the default stream, the `all_gather` for block $i+1$'s parameters runs on the comm stream. When block $i$ finishes, block $i+1$ is ready and runs with no gap. In the backward pass the same overlap runs in reverse, and the gradient `reduce_scatter` of a finished block streams out on the comm stream while the previous block recomputes. The network is never idle and the GPU is never waiting — the collective for the *next* unit of work hides under the compute of the *current* one. This is the same "independent compute on the default stream hides the collective on the comm stream" pattern from section 2, just organized around the layer boundary instead of the bucket boundary.

The mechanism is captured in the same sum-versus-max equation from the FSDP post: without overlap a step costs $T_\text{compute} + T_\text{comm}$; with prefetch it costs $\max(T_\text{compute}, T_\text{comm})$. Prefetch turns the sum into the max. The full memory analysis is in [FSDP in Practice](/blog/machine-learning/distributed-training/fsdp-in-practice); here we care about the three knobs that control the overlap and how deep to prefetch:

```python
import functools
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy, MixedPrecision, BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={TransformerBlock},  # one FSDP unit per block -> per-block gathers
)

model = FSDP(
    model,
    auto_wrap_policy=wrap_policy,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,
                                   reduce_dtype=torch.float32),
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # prefetch next gather BEFORE current backward
    forward_prefetch=True,                            # overlap forward gathers too
    limit_all_gathers=True,                           # cap prefetch depth to protect memory
    device_id=torch.cuda.current_device(),
)
```

- **`backward_prefetch=BackwardPrefetch.BACKWARD_PRE`** kicks off the next-needed `all_gather` *before* the current block's backward computation, giving the deepest overlap at the cost of briefly holding two blocks' parameters resident. `BACKWARD_POST` prefetches after, using less memory but overlapping less. Default to `BACKWARD_PRE` unless you are memory-bound.
- **`forward_prefetch=True`** prefetches the next block's parameters during the forward pass. It matters most when the forward is **CPU-bound** — when kernel-launch latency would otherwise delay the gather — which is precisely the failure mode section 5 is about.
- **`limit_all_gathers=True`** (the "rate limiter") caps how far ahead FSDP prefetches. This is the **prefetch-depth** knob, and it is a genuine tension. Prefetch too shallow (depth 1) and a slow gather can still stall the next block. Prefetch too deep and you have several blocks' worth of gathered parameters resident at once — the transient memory balloons and you OOM. `limit_all_gathers=True` trades a little overlap for a memory ceiling; leave it on unless you have profiled and have memory to spare.

### Why prefetch depth is the FSDP analogue of bucket size

DDP's bucket size and FSDP's prefetch depth are the same knob wearing different clothes: both control *how far ahead of the compute the communication is allowed to run*. Too shallow and comms cannot get ahead enough to hide — you stall. Too deep and you either fragment (DDP) or blow the memory budget (FSDP). The sweet spot in both cases is "comms runs one unit of work ahead of compute, bandwidth-bound, never starving the GPU and never overflowing memory." When you internalize that both DDP and FSDP are solving the *same* scheduling problem — keep the comm stream exactly one step ahead of the compute stream — the two APIs stop looking like separate things to memorize.

| Strategy | Collective(s) per step | What overlaps them | What stays exposed |
|---|---|---|---|
| DDP | 1 all-reduce (gradients) | Backward compute of earlier layers | The last bucket |
| FSDP `FULL_SHARD` | 2 all-gathers + 1 reduce-scatter | Compute of the current block (prefetch next) | First forward gather, last backward reduce-scatter |
| Tensor parallel | 1 blocking all-reduce per layer, fwd and bwd | **Nothing** — it is on the critical path | All of it (see section 5) |
| Pipeline parallel | Point-to-point sends between stages | Compute of other micro-batches | The pipeline bubble |

That table previews the next section: three of the four rows overlap well, and one — tensor parallelism — structurally cannot. Understanding *why* is understanding the limits of overlap.

Pipeline parallelism deserves a word because its overlap has yet another shape. A pipeline stage sends its output activations to the next stage over a point-to-point link, and receives gradients back. Those sends and receives are small relative to a full all-reduce, and they overlap with the compute of *other micro-batches* flowing through the pipeline — while stage 0 sends micro-batch 3's activations forward, it is already computing micro-batch 4. So the point-to-point comms is rarely the bottleneck; the exposed cost of pipeline parallelism is the **bubble**, the idle time at the start and end of each batch when the pipeline is filling and draining and some stages have no micro-batch to work on. The bubble is not comms sitting on the critical path — it is *compute* that cannot happen yet because the data has not arrived. Overlap in the streams sense is already good in a well-scheduled pipeline (1F1B keeps every stage busy in steady state); the lever there is increasing the micro-batch count to shrink the bubble fraction, a different knob from anything in this post. The unifying thread across all four strategies is the same question asked four ways: *is there independent work to run while the bytes move?* DDP and FSDP and pipeline all find some; tensor parallelism, by construction, does not.

## 5. What breaks overlap (and what to do about each)

Overlap is not automatic. It is a scheduling opportunity that four distinct problems can take away from you. The diagnostic tree below is the one to run when a profiler shows the GPU sitting idle waiting on the network — start at the top and branch to the cause.

![a decision tree that starts from a GPU sitting idle waiting on the network and branches to the four distinct root causes of broken overlap and their fixes](/imgs/blogs/overlapping-compute-and-communication-6.webp)

### Cause 1: a data dependency forces a wait (tensor parallelism)

The most fundamental way to break overlap is a **data dependency**: the compute you would use to hide the collective *needs the result of that collective*. Then there is nothing independent to run, and the communication is on the critical path by construction.

The textbook case is tensor parallelism. When you split a linear layer's matmul across GPUs (column-parallel then row-parallel), the row-parallel half produces *partial sums* on each GPU that must be added together with an **all-reduce before the next layer can run** — the next layer's input *is* the reduced output. There is no independent compute to overlap it with, because everything downstream depends on the all-reduce result. This is a **blocking** all-reduce, and it happens in both the forward and the backward pass, on every tensor-parallel layer. It cannot be hidden. This is the core reason tensor parallelism is only worth its comms cost inside a single node where NVLink makes the blocking all-reduce cheap; stretch it across a slow inter-node link and the exposed all-reduces dominate. The full picture is in [Tensor Parallelism with Megatron](/blog/machine-learning/distributed-training/tensor-parallelism-megatron). The fix here is not "make overlap work" — it can't — it is "put tensor parallelism only where the link is fat enough that the exposed all-reduce is cheap, and keep the TP degree small."

### Cause 2: messages too small to be bandwidth-bound

If your collectives are tiny — from too-small DDP buckets, from a model with thousands of small parameters, from an unfused sequence of little all-reduces — they become **latency-bound**. Each transfer spends most of its time in fixed overhead (launch, handshake, bandwidth ramp) rather than moving bytes. Latency-bound comms overlaps badly for two reasons: the aggregate time is inflated far beyond what the byte count justifies, and launching a flood of tiny collectives can saturate the CPU's kernel-launch throughput so the comm stream starves. The fix is **fewer, larger messages**: raise `bucket_cap_mb`, fuse gradients, use a coarser FSDP wrap granularity so each `all_gather` moves more. Bandwidth-bound collectives are the ones that overlap cleanly.

### Cause 3: the CPU can't launch kernels fast enough

This one surprises people because it has nothing to do with the network. A GPU step is a *stream of kernels* that the CPU enqueues one at a time. Each launch costs a few microseconds of CPU time. If your model is made of many tiny kernels — lots of small elementwise ops, tiny matmuls, an un-fused attention — the CPU can spend more time launching kernels than the GPU spends running them. The GPU goes idle *between* kernels, waiting for the CPU to feed it the next one, and the comm stream starves right alongside the compute stream. The profiler signature is a compute stream full of little gaps and a CPU pinned at 100%. This is **CPU-bound / kernel-launch-bound** execution, and no amount of NCCL tuning fixes it because the bottleneck is upstream of the GPU.

The fix is to stop launching so many kernels. `torch.compile` fuses many small operations into a few large ones, cutting launch count. **CUDA graphs** go further: they *capture* the entire sequence of kernels for a step once and then *replay* the whole graph with a single launch, eliminating per-kernel CPU overhead almost entirely. With the launch bottleneck gone, the GPU stays busy, the comm stream gets fed, and overlap is restored:

```python
import torch

# Option A: torch.compile fuses small ops, cutting the number of kernel launches.
model = torch.compile(model, mode="max-autotune")

# Option B: capture the whole step as a CUDA graph and replay it with one launch.
# (Static shapes and static input buffers are required; NCCL collectives can be
#  captured too, so the overlapped all-reduce replays inside the graph.)
static_input = torch.empty_like(sample_batch, device="cuda")
g = torch.cuda.CUDAGraph()

# Warm up a few steps first so cuBLAS/NCCL allocate and autotune outside the graph.
for _ in range(3):
    optimizer.zero_grad(set_to_none=True)
    loss = model(static_input).loss
    loss.backward()
    optimizer.step()

with torch.cuda.graph(g):
    static_loss = model(static_input).loss
    static_loss.backward()          # bucketed all-reduces captured into the graph
    optimizer.step()

for batch in loader:
    static_input.copy_(batch)       # fill the static buffer in place
    g.replay()                      # ONE launch replays the whole overlapped step
```

CUDA graphs demand static shapes and a fixed control flow, which is why they fit LLM pretraining (fixed sequence length, fixed batch) far better than variable-length inference. But when they fit, they turn a kernel-launch-bound step back into a compute-bound one, and the overlap you designed for actually happens.

### Cause 4: there is simply more comms than compute to hide it under

The last cause is not a bug — it is physics. Overlap can only hide the *shorter* of compute and comms under the *longer*. If $T_\text{comm} > T_\text{compute}$, then even perfect overlap leaves $T_\text{comm} - T_\text{compute}$ exposed: the max is now the comms term, and no scheduling saves you. This happens with tiny models (little compute, so any collective sticks out), tiny per-GPU batches (same), and — most commonly — when you cross a slow link, where $T_\text{comm}$ inflates 5–40x. FSDP `FULL_SHARD` across nodes on InfiniBand is the canonical case: the per-block gathers that hid perfectly on NVLink now take longer than the block's compute, and throughput falls off a cliff.

When you are in this regime, overlap tuning is the wrong tool. The fixes all attack $T_\text{comm}$ or $T_\text{compute}$ directly: use a **fatter link** (keep the frequent collectives on NVLink — this is exactly what FSDP's `HYBRID_SHARD` does, sharding within the node and only replicating across it); use a **bigger per-GPU batch** to lengthen $T_\text{compute}$ so there is more to hide under; send **fewer, larger messages** so the comms is bandwidth-efficient; or reduce the comms volume outright with gradient compression or a lower-precision reduce. The decision of *which* is the subject of picking a parallelism strategy, but the trigger is always the same measurement: $T_\text{comm} > T_\text{compute}$ means stop tuning overlap and start reducing comms.

#### Worked example: too-small buckets, latency-bound, recovered

A team ports a model to 8×A100 and sees only 3.2x speedup. The profiler shows the comm stream busy the entire step but the *aggregate* all-reduce time absurdly high for a 3 GB gradient — a red flag for latency-bound comms. The cause: someone set `bucket_cap_mb=1` (copied from a debugging config), fragmenting the gradient into roughly 3,000 tiny 1 MB all-reduces. Each carries ~25 µs of fixed overhead, so 3,000 of them cost ~75 ms in overhead *alone*, on top of the bytes — and launching 3,000 collectives per step saturates the CPU, so the comm stream stalls waiting to be fed. Aggregate exposed comms: ~90 ms on a 120 ms compute step. Efficiency ~57%.

The fix is one line: `bucket_cap_mb=25`. Now the gradient is ~120 buckets of 25 MB each, every one bandwidth-bound, the CPU launches 120 collectives instead of 3,000, and overlap fills through the backward pass with only the last bucket exposed. Exposed comms drops to ~8 ms, efficiency to ~94%, speedup to 7.1x. Same bytes, same fabric — the messages were just too small to use the wire, and the CPU was drowning in launches. This is cause 2 and cause 3 hitting at once, and one knob fixed both.

## 6. Seeing it: reading a profiler for exposed communication

Everything above is a claim about *when* things happen on the timeline, and the only way to know for sure is to look at the timeline. Two tools show it: `torch.profiler` (per-rank, exports a Chrome trace and a TensorBoard view) and Nsight Systems (`nsys`, the system-wide view with NCCL and CUDA on separate rows). The skill of reading them is covered in depth in [Profiling GPU Workloads](/blog/machine-learning/high-performance-computing/profiling-gpu-workloads-finding-the-real-bottleneck); here is the specific thing to look for and how to quantify it.

```python
import torch
from torch.profiler import profile, ProfilerActivity, schedule

# Warm up first (the first steps do one-time setup), then capture a few steady steps.
prof = profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=5, warmup=3, active=5, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./trace"),
    record_shapes=True,
    with_stack=False,
)
prof.start()
for step, batch in enumerate(loader):
    optimizer.zero_grad(set_to_none=True)
    loss = model(batch).loss
    loss.backward()
    optimizer.step()
    prof.step()          # tells the profiler where step boundaries are
    if step >= 13:
        break
prof.stop()
```

Open the resulting trace and find two rows: the **compute stream** (your matmul and elementwise kernels) and the **NCCL stream** (`ncclAllReduce`, `ncclAllGather`, `ncclReduceScatter` kernels). Overlap is *visual and unmistakable*: when a NCCL kernel on the comm row sits directly beneath compute kernels on the compute row, that communication is hidden — it cost you nothing. **Exposed** communication is the opposite: a NCCL kernel with *nothing* running on the compute row above it, and — the smoking gun — a **gap in the compute row** where the GPU is idle, waiting on that collective. Those gaps are your lost throughput. Sum their duration and you have measured $T_\text{exposed}$ directly. Divide by step time and you have your exposed-comms fraction — the exact number that determined the 60% versus 94% efficiency in the worked examples.

You do not have to eyeball the trace to get a first-order number, either. The profiler's aggregated table gives you total NCCL device time, which — compared against total step time — tells you the *upper bound* on how much comms there is to worry about before you go hunting for exposed gaps:

```python
# After prof.stop(), print the heaviest CUDA ops and pull out the NCCL total.
key_avgs = prof.key_averages()
print(key_avgs.table(sort_by="cuda_time_total", row_limit=15))

nccl_us = sum(evt.cuda_time_total for evt in key_avgs
              if "nccl" in evt.key.lower())
print(f"total NCCL device time: {nccl_us / 1e3:.1f} ms over the captured steps")
# If NCCL device time is a large fraction of wall-clock AND the compute row has
# gaps, comms is exposed. If NCCL time is large but the compute row is gap-free,
# it is overlapping fine and the fabric is simply busy -- not your bottleneck.
```

The distinction in that comment is the one people get wrong: *large NCCL time is not itself a problem.* A perfectly overlapped run can have the comm stream busy the entire step — that is comms doing its job under compute. What costs you is NCCL time that coincides with an *idle compute row*. Total NCCL time bounds the problem; the compute-row gaps are the problem.

For the system view across ranks, Nsight Systems is the tool. Capture CUDA, NVTX ranges, and NCCL on every rank:

```bash
# Profile rank-by-rank; -t nccl puts the collectives on their own timeline row.
nsys profile \
  --trace=cuda,nvtx,nccl \
  --output=overlap_rank${RANK} \
  --capture-range=cudaProfilerApi \
  python train.py

# Then open overlap_rank0.nsys-rep in the Nsight Systems GUI and align the
# NCCL row against the CUDA kernel row. Idle gaps on the CUDA row that line up
# with NCCL kernels are exposed communication.
```

The stack below names where the wall-clock of one step actually goes once you have read the trace — it is the anatomy you are trying to reshape.

![a layered breakdown of where the wall clock time of one training step goes split into useful compute hidden communication exposed communication and kernel launch gaps](/imgs/blogs/overlapping-compute-and-communication-7.webp)

Useful compute is the floor you cannot go below (short of a better kernel). Hidden communication is free — it overlapped. Exposed communication and kernel-launch gaps are the two things overlap tuning removes: the first by getting the collective under compute (bucket/prefetch tuning), the second by launching fewer kernels (`torch.compile` / CUDA graphs). Your job, reduced to a picture, is to shrink the top two bands until only useful compute remains.

#### Worked example: the "GPU idle waiting on NCCL" signature

A run scales to only 4.4x on 8 GPUs and the team suspects the interconnect. The profiler tells a cleaner story: the compute row has a regular 55 ms gap at the *end* of every step, and directly under that gap sits a single large `ncclAllReduce` — the entire gradient in one collective. The signature is unambiguous: **all** the comms is exposed, none is overlapped. The cause turns out to be a hand-rolled gradient sync — a loop calling `dist.all_reduce(p.grad)` after `loss.backward()` returned — instead of DDP's bucketed, hook-driven overlap. Because the sync ran *after* backward completed, there was no compute left to hide it, and the whole 55 ms landed on the critical path.

The fix is to let DDP do it: wrap the model in `DistributedDataParallel` and delete the manual loop. Now the all-reduce is bucketed and fired from autograd hooks *during* backward. The 55 ms gap collapses to a ~5 ms gap (the last bucket), the compute row goes nearly gap-free, and the speedup climbs from 4.4x to 7.3x. The interconnect was never the problem; the *schedule* was. This is the most common overlap bug there is — comms that is correct but serial — and the profiler finds it in thirty seconds.

### Measuring honestly

A note on method, because it is easy to fool yourself. GPU kernels are asynchronous: `loss.backward()` returns before the work finishes. Any timing must bracket a `torch.cuda.synchronize()` (or CUDA events) or you are timing kernel *launches*, not kernel *execution*. Always **warm up** — the first several steps do one-time work (DDP rebuilds buckets on step one, cuBLAS and NCCL autotune and allocate) and are 2–5x slower than steady state; timing them makes overlap look worse than it is. Measure **steady-state** median step time over tens of steps, not the mean (which a single stall skews). And watch the **data-loader confound**: if the GPU is waiting on the loader, the compute row has gaps that look like exposed comms but are not — check that the loader (`num_workers`, `pin_memory`, `prefetch_factor`) is keeping the GPU fed before you blame the network. The profiler distinguishes them cleanly: a loader stall shows an idle GPU with *nothing* on either the compute or comm row; exposed comms shows an idle *compute* row with a live *comm* row.

## 7. Case studies and real numbers

A few results from the literature and from named hardware, to ground the mechanism. Where a number is approximate or configuration-dependent, it is flagged as such — do not quote these to four significant figures.

**DDP on NVLink lands near-linear precisely because of overlap.** PyTorch's own DDP design notes and the accompanying paper (Li et al., *PyTorch Distributed*, VLDB 2020) report that gradient bucketing with computation-communication overlap is what takes DDP from mediocre to near-linear scaling; without overlap, the all-reduce is a serial phase and efficiency drops sharply. The reported scaling on well-connected nodes is in the ~90%+ range for models where the gradient bucket collectives fit under the backward compute — exactly the regime of the worked examples above.

**FSDP within a few percent of DDP on NVLink, despite 1.5x the bytes.** Because prefetch hides the extra `all_gather` and `reduce_scatter` under compute, FSDP's larger communication volume mostly does not show up in wall-clock on a fast intra-node fabric. Meta's FSDP reports (Zhao et al., *PyTorch FSDP*, 2023) show throughput competitive with DDP on models that fit, with the gap widening only when communication can no longer be hidden — i.e., across a thin link, which is the motivation for `HYBRID_SHARD`. The lesson is the recurring one: on NVLink, overlap makes the byte-count difference between DDP and FSDP nearly invisible; across nodes, it does not, and topology-aware sharding is required.

**Megatron-LM's tensor parallelism is deliberately confined to a node for the overlap reason.** The Megatron-LM papers (Shoeybi et al., 2019; Narayanan et al., *Efficient Large-Scale Language Model Training*, SC 2021) keep the tensor-parallel degree at or below the number of GPUs in a node (typically 8) specifically because TP's all-reduces are *blocking* and cannot overlap. Inside a node, NVLink makes each exposed all-reduce cheap; across nodes it would dominate. Their high aggregate MFU (reported in the ~50% range on large clusters) depends on placing the un-overlappable comms on the fattest link and using pipeline and data parallelism — which *do* overlap — across the slower links. This is the whole overlap thesis expressed as a cluster-layout decision.

**Interconnect sets the ceiling on how much overlap can save you.** NVLink4 on H100 offers roughly 900 GB/s of aggregate bandwidth per GPU; InfiniBand HDR is about 200 Gb/s (~25 GB/s) and after protocol overhead often delivers an effective ~10–15 GB/s per GPU cross-node. That is a 40x-plus gap. Overlap can hide a collective that takes 8 ms on NVLink; the *same* collective taking 250 ms across InfiniBand exceeds any plausible per-step compute and cannot be hidden. The interconnect physics is covered in [The Interconnect Physics](/blog/machine-learning/distributed-training/the-interconnect-physics); the point for overlap is that the link sets $T_\text{comm}$, and once $T_\text{comm} > T_\text{compute}$, overlap has done all it can.

## 8. When to reach for overlap tuning (and when it can't help)

Overlap tuning is high-leverage when your comms is *overlappable in principle but currently exposed* — a serial all-reduce, mis-sized buckets, missing prefetch, a kernel-launch bottleneck. In those cases a one-line change can move you from 60% to 90%+ efficiency, and it is almost always the first thing to try when a multi-GPU job underperforms. Reach for it when the profiler shows the comm stream live and the compute stream idle above it: that gap is recoverable throughput.

It **cannot** help in two situations, and recognizing them saves you days of tuning the wrong thing. First, when the communication is on the critical path *by construction* — tensor parallelism's blocking all-reduce, a pipeline stage waiting on the previous stage's activation, any collective whose result the very next operation needs. There is no independent compute to hide it under, so the answer is not "overlap it" but "make it cheaper or rarer": shrink the TP degree, put it on the fattest link, reduce the message. Second, when there is simply more comms than compute — a tiny model, a tiny per-GPU batch, or a slow inter-node link where $T_\text{comm} > T_\text{compute}$. Then even perfect overlap leaves the excess exposed, and the fix is to attack the comms directly: a fatter link, a bigger batch to lengthen compute, fewer and larger messages, `HYBRID_SHARD` to keep gathers on NVLink, or lower-precision/compressed collectives.

The decision rule is one measurement. Profile a steady step, sum the compute-row idle time that lines up with live NCCL kernels, and compare $T_\text{comm}$ to $T_\text{compute}$. If comms is exposed but smaller than compute, tune overlap — it will work. If comms exceeds compute, stop tuning overlap and reduce comms — no schedule can hide a collective longer than all the compute you have. And if the compute row has gaps with *nothing* on the comm row, it is not comms at all — check the data loader and kernel-launch overhead first.

## Key takeaways

- **Communication only costs wall-clock if it's on the critical path.** The step-time law is $T_\text{step} = T_\text{compute} + T_\text{exposed}$ with $T_\text{exposed} = \max(0, T_\text{comm} - T_\text{overlappable})$. Overlap drives $T_\text{exposed}$ toward zero; that is the entire scaling game.
- **CUDA streams are the mechanism.** Compute runs on one stream, NCCL collectives on another, and the GPU executes them concurrently — joined only by a CUDA event where data actually flows between them. Every overlap in distributed training is two streams and one dependency.
- **DDP overlaps by bucketing gradients that become ready in reverse layer order**, firing each bucket's all-reduce during the backward of earlier layers. Only the last bucket is exposed. `bucket_cap_mb` is the knob: too small is latency-bound, too big leaves a huge exposed last bucket.
- **FSDP overlaps by prefetching the next layer's parameters under the current layer's compute.** `backward_prefetch=BACKWARD_PRE`, `forward_prefetch`, and `limit_all_gathers` control the depth. Prefetch depth is the FSDP analogue of DDP's bucket size — keep the comm stream one unit ahead of compute.
- **Four things break overlap:** a data dependency (tensor parallelism's blocking all-reduce), messages too small to be bandwidth-bound, a CPU too slow to launch kernels, and comms simply larger than the compute available to hide it. Each has a different fix; diagnose before you tune.
- **`torch.compile` and CUDA graphs fix the kernel-launch-bound case** by cutting or eliminating per-kernel CPU launch overhead, keeping the GPU fed so overlap can happen.
- **Read the profiler to prove it.** In a `torch.profiler` or `nsys` trace, a NCCL kernel with an idle compute row above it is exposed comms; sum those gaps to measure $T_\text{exposed}$ directly. A NCCL kernel under live compute is free.
- **Overlap can't beat physics.** When $T_\text{comm} > T_\text{compute}$ — tiny model, tiny batch, or slow inter-node link — stop tuning overlap and reduce comms: fatter link, bigger batch, fewer/larger messages, `HYBRID_SHARD`.
- **Measure honestly.** Warm up, `torch.cuda.synchronize()` before timing, use steady-state median step time, and rule out the data loader before blaming the network.

## Further reading

- [DDP From First Principles](/blog/machine-learning/distributed-training/ddp-from-first-principles) — the gradient all-reduce, bucketing, and the autograd-hook machinery this post builds on.
- [FSDP in Practice](/blog/machine-learning/distributed-training/fsdp-in-practice) — the sharding strategies, prefetch knobs, and the sum-versus-max overlap derivation in full.
- [Tensor Parallelism with Megatron](/blog/machine-learning/distributed-training/tensor-parallelism-megatron) — why TP's all-reduce is blocking and cannot overlap, and where to place it.
- [Why Distributed Training](/blog/machine-learning/distributed-training/why-distributed-training) — the four walls and the parallelism map that frame the whole series.
- [The Distributed Training Playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook) — the capstone decision-and-debugging checklist that ties overlap to every other lever.
- [Profiling GPU Workloads: Finding the Real Bottleneck](/blog/machine-learning/high-performance-computing/profiling-gpu-workloads-finding-the-real-bottleneck) — reading traces to separate compute, comms, kernel-launch, and loader stalls.
- Li et al., *PyTorch Distributed: Experiences on Accelerating Data Parallel Training* (VLDB 2020) — the DDP bucketing-and-overlap design paper.
- Zhao et al., *PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel* (2023); Narayanan et al., *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM* (SC 2021) — overlap at scale in FSDP and Megatron.
