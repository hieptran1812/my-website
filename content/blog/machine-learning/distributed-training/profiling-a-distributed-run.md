---
title: "Profiling a Distributed Run: Seeing Where the Time Actually Goes"
date: "2026-07-02"
publishDate: "2026-07-02"
description: "Aggregate tokens per second tells you a distributed run is slow; only the profiler tells you where the time went. Learn to read a multi-GPU timeline in torch.profiler and Nsight Systems, spot the five signatures of wasted time — exposed comms, the pipeline bubble, the straggler, kernel-launch stalls, and a starving data loader — and measure honestly enough to trust the number."
tags:
  [
    "distributed-training",
    "multi-gpu",
    "profiling",
    "torch-profiler",
    "nsight-systems",
    "nccl",
    "pytorch",
    "gpu",
    "ml-systems",
    "mfu",
  ]
category: "machine-learning"
subcategory: "Distributed Training"
author: "Hiep Tran"
featured: true
readTime: 41
---

Here is the situation you are actually in when you open a profiler. You have a 7-billion-parameter model training on 64 A100s. The training log prints a tokens-per-second number, you do the arithmetic, and it comes out to 31% Model FLOPs Utilization. You know that is bad — a well-tuned run of this shape should be in the mid-forties — so you know, with certainty, that roughly a third of a very expensive cluster is being wasted. That is *everything* the aggregate number can tell you. It says a problem exists. It does not say whether the GPUs are idling because the data loader can't keep up, because the all-reduce is running exposed on the critical path, because one node is a straggler dragging the other 63, because the CPU is too slow to launch kernels fast enough, or because your batch is simply too small to fill the tensor cores. Those five causes produce the *identical* aggregate MFU. They require five completely different fixes. And no scalar you can print — not tokens/s, not MFU, not GPU-utilization from `nvidia-smi` — can distinguish them, because a scalar has thrown away the one dimension that carries the answer: time.

The profiler is the instrument that gives that dimension back. It records, kernel by kernel and rank by rank, *when* every operation ran and how long the GPU sat between operations doing nothing. When you lay that timeline out, the thing you paid for and the thing that is stealing your money become visible as two lanes and the gap between them. The exposed all-reduce is a block of NCCL running with no compute beside it. The starving loader is a rhythmic band of dead air where nothing runs at all. The straggler is one rank's step stretched longer than its 63 peers, who are all parked in a spinning collective waiting for it. The kernel-launch stall is a lawn of tiny gaps between kernels, each one the GPU waiting on a CPU that fell behind. You cannot fix what you cannot see, and the profiler is how you see it.

![a two column comparison of a sick timeline where the all-reduce runs as an exposed gap after the backward pass against a healthy timeline where the same all-reduce is hidden underneath compute](/imgs/blogs/profiling-a-distributed-run-1.webp)

By the end of this post you will be able to stand up a `torch.profiler` block that records a few steady-state steps and exports a Chrome trace; run `nsys profile` to capture the whole-machine timeline including the kernel-launch gaps `torch.profiler` glosses over; annotate your training loop with NVTX ranges so the phases of a step are labeled instead of anonymous; read the five distributed signatures of wasted time straight off the timeline and route each to its fix; and — the part everyone skips and everyone should not — measure with enough discipline (warm-up, `torch.cuda.synchronize()`, steady state, the same step across ranks, an honest accounting of the observer effect) that you can trust the number you report. This is the profiling post of the [Distributed Training in the Trenches](/blog/machine-learning/distributed-training/why-distributed-training) series, and it is the instrument that every other post in the series implicitly assumes you can operate: [overlapping compute and communication](/blog/machine-learning/distributed-training/overlapping-compute-and-communication) is a claim you can only verify with a profile; [the straggler](/blog/machine-learning/distributed-training/the-straggler) is a rank you can only find with per-rank timing; [the data pipeline at scale](/blog/machine-learning/distributed-training/the-data-pipeline-at-scale) is a stall you can only confirm in the trace.

## 1. Why aggregate throughput can't tell you where the time went

Tokens per second is the right north-star metric for a training run, and MFU derived from it is the right score. But both are *scalars*, and a scalar is an average over the whole step. Averages are lossy on purpose — they compress a rich thing down to one number so you can track it on a dashboard. The compression is exactly what loses the diagnosis.

Write out a single training step as a sum of the wall-clock time spent in each phase:

$$T_\text{step} = T_\text{forward} + T_\text{backward} + T_\text{comm}^\text{exposed} + T_\text{idle}^\text{loader} + T_\text{optim}$$

Aggregate throughput is $\text{tokens per step} / T_\text{step}$. It is a function of the *sum* of those terms. Two runs with wildly different term breakdowns can land on the identical $T_\text{step}$: run A spends 40 ms in exposed all-reduce and zero idling on the loader; run B spends zero exposed comms and 40 ms starving for data. Same tokens/s, same MFU, opposite fixes. Run A needs bucketing and prefetch to overlap the comms; run B needs more dataloader workers and pinned memory. If you only have the scalar, you cannot tell which run you are looking at, so you cannot know which fix to apply — you are reduced to guessing, changing one thing, and re-measuring, which on a 64-GPU reservation is an expensive way to search.

The profiler dissolves the sum back into its terms. It measures each phase separately, per rank, in time order, so instead of one number you get the whole breakdown — and the breakdown *is* the diagnosis. This is the single most important mental shift in performance work at scale: **stop asking "how fast is it" and start asking "where did the time go."** The first question has a one-number answer that you already know is bad. The second has a timeline answer that tells you what to do.

A useful way to hold this: the aggregate number is the *symptom*, and the timeline is the *examination*. You would not prescribe a treatment from a fever reading alone, and you should not prescribe a performance fix from an MFU reading alone. Both tell you something is wrong; neither tells you what.

### The GPU-utilization trap

Before we get to real profilers, kill the metric that fools the most people: `nvidia-smi`'s GPU-Util. It is seductive because it is one command away and it is a percentage, so it *feels* like it measures how productively the GPU is working. It does not. GPU-Util is defined as the fraction of the last sampling interval during which at least one kernel was executing on the device — nothing more. A GPU spinning in a NCCL all-reduce, waiting for a straggler's data to arrive and accomplishing zero useful arithmetic, is running a kernel, so it reports 100% utilization. A GPU grinding a productive matmul also reports 100%. The number cannot tell the two apart, which is precisely why [the straggler war story](/blog/machine-learning/distributed-training/the-straggler) opens with 64 GPUs all reading 100% busy while the job ran at half speed. GPU-Util answers "is a kernel running," never "is the kernel useful." Only a profiler that names the kernels — and shows the gaps between them — answers the question you actually have.

## 2. The anatomy of a distributed timeline: three lanes and the gaps

Everything a profiler shows you, once you strip away the thousands of individual op rows, resolves into three lanes stacked in time. Learn to see these three and you can read any distributed trace.

![a stacked diagram of the compute lane the NCCL communication lane and the idle gaps that decide whether a step is overlapped or exposed](/imgs/blogs/profiling-a-distributed-run-2.webp)

The **compute lane** is the CUDA stream where your matmuls, layernorms, and elementwise kernels run — the forward pass, the backward pass, the optimizer update. This is the work you paid for; every millisecond here is arithmetic that moves the loss. The **communication lane** is a *separate* CUDA stream where NCCL runs its collectives: the gradient all-reduce in DDP, the parameter all-gather and reduce-scatter in FSDP, the blocking all-reduce inside every tensor-parallel layer. NCCL deliberately runs on its own stream so it *can* execute concurrently with compute — that concurrency is the entire mechanism of communication overlap. And the third thing, the one that is not a lane so much as an absence: **the gaps**, the intervals where neither lane has a kernel running and the GPU is genuinely idle. The gaps are where your wasted time lives. A profile is, at its heart, a tool for making the gaps visible and attributable.

Now the reading. Put the compute lane and the comms lane side by side in time and ask one question: *does the comms overlap the compute, or does it sit in a gap alone?* When the all-reduce runs during the backward pass — while the compute lane is still busy computing gradients for earlier layers — the communication is **hidden**. It costs almost no wall-clock, because it happens in a window the GPU was busy anyway. The step time is then the *maximum* of the two lanes, since the shorter one fits entirely inside the longer:

$$T_\text{step}^\text{overlap} = \max(T_\text{compute}, T_\text{comm})$$

When the all-reduce instead runs *after* the backward finishes, with the compute lane empty beside it, the communication is **exposed** — it sits on the critical path, adding straight to the step, and the step time is the *sum*:

$$T_\text{step}^\text{exposed} = T_\text{compute} + T_\text{comm}$$

The gap between the sum and the max is the prize, and the profiler is the only tool that shows you which regime you are in. This is the exact idea [overlapping compute and communication](/blog/machine-learning/distributed-training/overlapping-compute-and-communication) derives from first principles; profiling is how you *verify* that the overlap you think you have is actually happening. It is depressingly common to enable every overlap knob, believe the comms is hidden, and discover in the trace that a data dependency forced it exposed anyway.

### The exposed-comms number, straight from the trace

Here is the mechanism that turns "the trace looks bad" into a number you can put in a ticket. Let $C$ be the total time the compute stream is active in a step, and $K$ the total time the NCCL stream is active. If they overlapped *perfectly*, the step would take $\max(C, K)$. The **exposed communication** is the part of $K$ that did not find compute to hide behind:

$$T_\text{exposed} = T_\text{step} - C$$

because whatever the step took beyond the compute time must have been the GPU waiting on something that was not compute. And the *overlap efficiency* — the fraction of your communication you successfully hid — is

$$\eta_\text{overlap} = 1 - \frac{T_\text{exposed}}{K} = 1 - \frac{T_\text{step} - C}{K}.$$

If $\eta_\text{overlap}$ is near 1, essentially all your comms is hidden and you are done tuning overlap. If it is near 0, the comms is fully exposed and bucketing or prefetch is your highest-leverage fix. You can read $C$, $K$, and $T_\text{step}$ straight off a `torch.profiler` summary — I will show the exact code in §4 — which means you can compute $\eta_\text{overlap}$ per run and track it, instead of eyeballing a timeline and hoping. That is the whole point of profiling: it converts a picture into a number you can regress against.

### The barrier makes the step the max, not the mean

There is a second law hiding in the timeline, and it is the reason single-rank profiling can lie to you about a distributed run. A synchronous data-parallel step ends on a collective — a gradient all-reduce — that is a *barrier*: no rank leaves it until every rank has entered it. So the wall-clock of one step is not the average of the ranks' compute times and it is not any one rank's compute time. It is the *maximum* over all ranks, plus the fixed cost of the reduction itself once everyone has arrived:

$$T_\text{step} = \max_{i \in \text{ranks}} c_i \; + \; A$$

where $c_i$ is rank $i$'s per-step compute time and $A$ is the reduction cost. When every rank is identical, $\max_i c_i = c$ and the step is the healthy $c + A$; nobody waits more than the fixed $A$. But let one rank be a fraction $s$ slower — $c(1+s)$ while the rest stay at $c$ — and the max jumps to $c(1+s)$, so the step stretches by $c \cdot s$ *for all ranks*, and the whole-job throughput falls by roughly ${1/(1+s)}$. A rank 40% slower drops the job to $1/1.4 \approx 0.71$ of throughput; a rank 2x slow halves it. Notice what is *absent* from that ratio: the number of ranks. The tax is set entirely by the slowest participant's relative slowdown, not by how many fast ranks are waiting.

The profiling consequence is direct. If you profile only rank 0 and it happens to be a fast rank, you will see a clean step and conclude the run is healthy — while rank 41 grinds 65% slower and gates everyone. The max in that formula is exactly why the distributed-specific discipline of §5, *profile the same step across ranks and take the max*, is not optional: the single number that governs your throughput lives on whichever rank is slowest, and you cannot know which rank that is without measuring all of them.

## 3. Two profilers, and when to reach for each

There are two profilers worth knowing for distributed PyTorch, and they answer different questions. Reaching for the wrong one wastes an afternoon.

![a comparison matrix of torch dot profiler against Nsight Systems across scope granularity NCCL visibility launch gaps setup and best use](/imgs/blogs/profiling-a-distributed-run-3.webp)

**`torch.profiler`** is PyTorch-native. You wrap your training loop in a context manager, it records every PyTorch operator with its CUDA kernel timings, optionally the input shapes and the Python call stack, and it exports a Chrome trace you open in `chrome://tracing` or Perfetto. It sees NCCL collectives as operator rows (`nccl:all_reduce`, `c10d::allreduce_`), so it answers "which op dominates the step" and "is my comms overlapping compute" cleanly and with almost no setup. Its blind spot is the space *between* kernels: it is built around the PyTorch op stream, so a step that is slow because the CPU can't launch kernels fast enough — thousands of microscopic gaps — is harder to see, because `torch.profiler` centers the ops, not the idle.

**Nsight Systems** (`nsys`) is NVIDIA's system-wide timeline profiler. It traces the whole machine: every CUDA kernel, every memory copy, the CUDA runtime API calls the CPU makes to launch kernels, NCCL, and any NVTX ranges you add — all on one aligned timeline across all the streams. Because it sees the CPU-side launch calls and the GPU-side kernels together, it shows you the launch gaps that `torch.profiler` hides: if the GPU is idle because the CPU is 200 microseconds late issuing the next kernel, `nsys` draws that gap explicitly. The cost is friction: it is a command-line tool that produces a `.nsys-rep` file you open in the Nsight Systems GUI, and the traces are large. Reach for `nsys` when the question is about the *whole timeline* — cross-stream overlap, kernel-launch overhead, what the CPU is doing — and for `torch.profiler` when the question is *which PyTorch op is slow* or *is this collective overlapping*.

| Dimension | `torch.profiler` | Nsight Systems (`nsys`) |
|---|---|---|
| Scope | PyTorch operators + their kernels | Whole system: CUDA, NCCL, CPU, NVTX |
| Granularity | Per-op, optional shapes + stack | Per-kernel, per-API-call |
| Sees NCCL | Yes, as operator rows | Yes, as CUDA kernels + NVTX |
| Sees kernel-launch gaps | Partially | Yes, explicitly (CPU launch vs GPU exec) |
| Setup | `pip`, a few lines in the loop | CLI wrapper + GUI viewer |
| Output | Chrome trace, key-averages table | `.nsys-rep`, timeline GUI |
| Best question | "Which op is slow? Is comms overlapped?" | "What is the whole timeline doing between kernels?" |
| Overhead | Low–moderate (higher with `with_stack`) | Low with sampling; can be moderate |

The honest workflow is: start with `torch.profiler` because it is one context manager away and it answers the two most common questions. Escalate to `nsys` when `torch.profiler` says the ops are individually fine but the step is still slow — that "the ops are fine but the wall-clock isn't" gap is almost always launch overhead or cross-stream stalls, and that is `nsys`'s home turf.

## 4. torch.profiler in practice

The single most important thing about `torch.profiler` is the **schedule**. Beginners wrap the whole training loop and profile every step, which produces a gigantic trace dominated by cold-start artifacts — the first few steps pay for cuDNN autotuning, NCCL's first-collective handshake, and the caching allocator warming up, none of which represent steady state. The schedule fixes this by skipping cold steps, warming the caches without recording, recording only a handful of steady-state steps, then stopping.

![an ordered schedule showing the profiler skipping cold steps warming the caches recording a few steady state steps and exporting the trace](/imgs/blogs/profiling-a-distributed-run-4.webp)

```python
import torch
from torch.profiler import profile, schedule, ProfilerActivity, tensorboard_trace_handler

# Record ONLY on rank 0 by default; every rank writing a trace is a lot of I/O.
# Profile a few representative steady-state steps, not the whole run.
prof_schedule = schedule(
    wait=5,       # skip the first 5 steps entirely (cold: autotune, NCCL init, allocator)
    warmup=1,     # run 1 step "warm" — tracing on, but results discarded
    active=3,     # record 3 steady-state steps
    repeat=1,     # do this cycle once
)

def on_ready(prof):
    # Export a Chrome/Perfetto trace and print the top ops by CUDA time.
    prof.export_chrome_trace(f"trace_rank{torch.distributed.get_rank()}.json")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=prof_schedule,
    on_trace_ready=on_ready,
    record_shapes=True,     # record op input shapes — lets you spot tiny/underfed matmuls
    profile_memory=True,    # track allocator activity — catches fragmentation and OOM pressure
    with_stack=False,       # Python stacks are gold for attribution but add real overhead
) as prof:
    for step, batch in enumerate(loader):
        loss = train_step(model, batch)   # forward + backward + optimizer
        prof.step()                        # MUST call every iteration to advance the schedule
        if step >= 9:                      # wait(5)+warmup(1)+active(3) = 9 steps captured
            break
```

Three things earn their place here. `record_shapes=True` records the input shapes of every op, which is how you catch a matmul that is running but *underfed* — a batch so small the tensor cores never fill, the classic reason a run is slow while every op "looks fine." `profile_memory=True` tracks the caching allocator so you can see fragmentation and the memory high-water mark, which matters because the fix for an exposed collective (bigger buckets) and the fix for a starving loader (bigger prefetch) both cost memory, and you need to know your headroom. And `with_stack` is a genuine trade-off: the Python call stack tells you *which line of your model* a slow op came from, which is invaluable, but walking the stack on every op adds meaningful overhead — turn it on when you are hunting a specific op, off when you are measuring aggregate timing.

### Computing the exposed-comms number from the trace

The `key_averages()` table gives you per-op CUDA time, and that is enough to compute the overlap efficiency from §2 without ever opening the GUI. Sum the NCCL ops to get $K$, take the total step time from the profiler's recorded step duration, and read the compute time as the step minus the exposed part.

```python
def overlap_efficiency(prof):
    """Estimate what fraction of NCCL communication was hidden under compute."""
    events = prof.key_averages()
    # NCCL collectives show up with 'nccl' or 'c10d' in the op name.
    comm_us = sum(e.cuda_time_total for e in events
                  if "nccl" in e.key.lower() or "c10d" in e.key.lower())
    # Total device-active time across all ops (rough proxy; the trace has the exact step span).
    total_cuda_us = sum(e.self_cuda_time_total for e in events)
    # Compute-only time = everything on the compute stream that isn't a collective.
    compute_us = total_cuda_us - comm_us

    # Read the true wall-clock step time from your own timing (see the sync discipline in section 7).
    step_us = measured_step_time_us          # torch.cuda.Event-timed, steady state
    exposed_us = max(0.0, step_us - compute_us)
    eta = 1.0 - (exposed_us / comm_us if comm_us else 0.0)
    print(f"comms {comm_us/1e3:.1f} ms | compute {compute_us/1e3:.1f} ms | "
          f"exposed {exposed_us/1e3:.1f} ms | overlap η = {eta:.2f}")
    return eta
```

An $\eta$ of 0.95 means you hid 95% of your communication and there is nothing left to win from overlap; an $\eta$ of 0.2 means four-fifths of your comms is on the critical path and bucketing is your next move. The number is approximate — the exact step span from the trace is cleaner than summing op times — but it is directionally right and, crucially, *comparable across runs*, which is what lets you prove a fix worked instead of believing it did.

### Reading the memory timeline

`profile_memory=True` records the caching allocator's activity over the step, and it earns its keep for two distributed-specific reasons. First, a mid-step `cudaMalloc` — the allocator running out of cached blocks and going to the driver for more — is a *synchronizing* call that stalls the whole stream and shows up as a sudden gap in the compute lane. If you see an unexplained gap that is not comms and not the loader, check the memory trace: an allocator spike right before it means fragmentation, and the fix is `expandable_segments` or a steadier batch shape, not anything to do with the model. Second, the memory high-water mark is the budget you are spending the two most common fixes *against*: bigger gradient buckets to overlap comms, and a deeper prefetch queue to feed the loader, both cost memory. The trace tells you whether you have the headroom to apply them before you OOM trying. Read the allocated-vs-reserved curve: a large gap between them is fragmentation you can reclaim; a reserved curve pinned near the card's capacity means you are out of room and the exposed-comms fix will have to come from elsewhere.

### Reading it in the GUI

Export the Chrome trace, open it in Perfetto (`ui.perfetto.dev`) or `chrome://tracing`, and you get the timeline directly. The compute stream and the NCCL stream are separate rows. Zoom into one step and look for the signature: if the NCCL row's blocks sit *underneath* the compute row's blocks, overlapped in time, your comms is hidden and healthy. If the NCCL row has a fat block with *nothing* in the compute row above it, that block is exposed communication sitting on the critical path — the sick pattern from figure 1, now visible as a literal empty space above a NCCL kernel. This is the fastest way to answer "is my overlap working" for one rank; the number from `overlap_efficiency` is how you track it over time.

## 5. Nsight Systems, NVTX, and profiling across ranks

`torch.profiler` centers PyTorch ops. Nsight Systems centers *time* — it shows the whole machine's timeline, including the CPU launching kernels and the gaps where the GPU waited on it. That is exactly what you need for the two signatures `torch.profiler` is weakest at: kernel-launch-bound steps and cross-stream stalls. The invocation wraps your normal launch.

```bash
# Profile one rank's process for a bounded window. -t selects the trace domains:
# cuda (kernels + memcpy), nvtx (your phase labels), nccl (collectives), osrt (CPU/OS runtime).
nsys profile \
  -t cuda,nvtx,nccl,osrt \
  -o dist_run_rank0 \
  --capture-range=cudaProfilerApi \
  --force-overwrite=true \
  python train.py --steps 30

# For a torchrun / multi-rank job, wrap the whole launcher and let nsys tag each process,
# or profile a single rank by gating on the env RANK inside a small wrapper script.
nsys profile -t cuda,nvtx,nccl -o dist_run \
  torchrun --nproc_per_node=8 train.py --steps 30
```

Two flags matter for distributed runs. `--capture-range=cudaProfilerApi` lets you bound the capture to a window you mark in code with `torch.cuda.profiler.start()` / `stop()`, so you record a few steady-state steps instead of the whole cold-started run — the same discipline as `torch.profiler`'s schedule. And keeping the trace domains tight (`-t cuda,nvtx,nccl`) keeps the `.nsys-rep` from ballooning; add `osrt` only when you specifically suspect CPU-side stalls, because OS-runtime tracing is voluminous.

### NVTX: give your phases names

Raw `nsys` output is a wall of anonymous kernels. The fix is **NVTX ranges** — you annotate your code with named spans, and they show up as labeled bands across the top of the timeline, so instead of squinting at kernel names you see "forward", "backward", "optimizer", "all\_reduce" laid out in time. This one change turns an unreadable trace into a legible one.

```python
import torch

def train_step(model, batch):
    # NVTX ranges become labeled bands in the Nsight timeline (and torch.profiler too).
    with torch.autograd.profiler.emit_nvtx():           # emit NVTX for every autograd op
        pass  # (usually set once around the whole run, not per step)

    torch.cuda.nvtx.range_push("forward")
    out = model(batch["input_ids"])
    loss = out.loss
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("backward")               # the all-reduce lives inside here in DDP
    loss.backward()
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("optimizer")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.nvtx.range_pop()
    return loss

# torch.profiler has a portable equivalent that also renders in Chrome traces:
from torch.profiler import record_function
def train_step_portable(model, batch):
    with record_function("forward"):
        out = model(batch["input_ids"]); loss = out.loss
    with record_function("backward"):
        loss.backward()
    with record_function("optimizer"):
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
    return loss
```

With the `backward` range labeled, the overlap question becomes trivial to answer in the GUI: if the NCCL all-reduce kernels sit *inside* the `backward` band, comms is overlapping the backward pass, which is exactly what DDP's bucketing is designed to make happen. If the NCCL kernels sit in a bare region *after* the `backward` band ends, the all-reduce is exposed.

### Profiling the same step across ranks

The distributed-specific move — the one thing single-GPU profiling never teaches you — is to profile the **same step on multiple ranks and compare**. In a synchronous data-parallel job, every rank runs the identical step and they all meet at the all-reduce barrier; the step's wall-clock is set by the *slowest* rank, because the barrier can't fire until the last one arrives. So the way to find a straggler is not to stare at one rank's trace — it is to overlay the same step number from several ranks and look for the one whose compute band is longer than the others'.

![a merge diagram where three ranks feed a gradient all-reduce barrier and the slowest rank at 290 milliseconds gates the whole step](/imgs/blogs/profiling-a-distributed-run-5.webp)

```python
# Cheap per-rank timing you can run ALWAYS, not just under a profiler.
# CUDA events time the device without the overhead of a full trace.
import torch, torch.distributed as dist

def timed_step(model, batch):
    start, after_bwd, end = (torch.cuda.Event(enable_timing=True) for _ in range(3))
    start.record()
    loss = forward_backward(model, batch)     # compute + the collective inside backward
    after_bwd.record()
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    end.record()
    torch.cuda.synchronize()                  # events are only valid after a sync

    compute_ms = start.elapsed_time(after_bwd)
    step_ms    = start.elapsed_time(end)
    # Gather every rank's compute time to rank 0 to find the outlier.
    t = torch.tensor([compute_ms], device="cuda")
    gathered = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, t)
    if dist.get_rank() == 0:
        times = [g.item() for g in gathered]
        slowest = max(range(len(times)), key=lambda i: times[i])
        print(f"step compute ms: min {min(times):.0f} max {max(times):.0f} "
              f"-> straggler is rank {slowest} at {times[slowest]:.0f} ms")
    return loss
```

When 63 ranks report 175 ms of compute and rank 41 reports 290 ms, you have found the straggler in one line of output — no GUI, no guesswork. That is the technique behind [the straggler war story](/blog/machine-learning/distributed-training/the-straggler): the profiler (or even just CUDA-event timing) makes per-rank timing visible, and per-rank timing is the *only* thing that fingers a straggler, because `nvidia-smi` shows all 64 ranks at 100% while 63 of them spin in the barrier waiting for the 64th.

### Aligning timelines across nodes, and cross-referencing NCCL

Two practical wrinkles show up the moment you go multi-node. First, aligning traces from different nodes on a common clock is harder than aligning ranks on one node, because each node's CPU clock drifts independently and `nsys` timestamps are host-relative. Do not try to overlay two nodes' timelines to the microsecond; instead align on *step number* (tag traces `trace_rank{rank}.json` and compare the same step index) and compare the *durations* of the compute bands, not their absolute positions. The straggler shows up as a longer band on one rank regardless of clock offset, so duration comparison is robust where absolute alignment is fragile.

Second, when the multi-node timeline shows exposed comms that was hidden on a single node, the profiler tells you *that* the collective is slow but not *why* — and the why is almost always the interconnect. Run `NCCL_DEBUG=INFO` alongside the profile and read which algorithm (Ring vs Tree) and which transport (NVLink, `P2P`, `SHM`, `NET`/InfiniBand) NCCL actually chose. A collective that fell back from InfiniBand to TCP sockets — because `NCCL_SOCKET_IFNAME` pointed at the wrong NIC, or GPUDirect RDMA was unavailable — will be an order of magnitude slower per byte, and that slowdown is what turns a hidden intra-node all-reduce into an exposed multi-node one. The timeline shows the exposed block; the NCCL log names the transport that made it slow. You need both, and this pairing is the bridge to the deeper NCCL log-reading covered in the series' NCCL debugging post.

```bash
# Profile AND capture NCCL's transport/algorithm choice in the same run.
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET \
  nsys profile -t cuda,nvtx,nccl -o dist_run \
  torchrun --nnodes=8 --nproc_per_node=8 --rdzv_backend=c10d \
           --rdzv_endpoint=$HEAD:29500 train.py --steps 30 \
  2> nccl_rank_${RANK}.log
# Then grep the log for the transport actually used per peer:
#   grep -E "via NET|via P2P|via SHM|NCCL INFO Channel" nccl_rank_0.log
```

## 6. The five distributed signatures and how to read them

A distributed run wastes time in a small number of characteristic ways, and each one leaves a distinct fingerprint in the trace. Learn the five and reading a profile becomes pattern-matching instead of archaeology. Every one of them is a *gap* — the GPU idle — and the shape of the gap tells you the cause.

![a decision tree routing an idle gap in the trace through its signature to a cause and then to the first fix to try](/imgs/blogs/profiling-a-distributed-run-6.webp)

**1. Exposed all-reduce (a fat NCCL block, alone).** In the timeline you see a large NCCL kernel on the comms stream with *nothing* on the compute stream above it — the backward pass has already finished and the all-reduce is running by itself on the critical path. Cause: the gradient bucket is too large to overlap, or an operation depends on the all-reduce's result so it can't be hidden (tensor parallelism's blocking all-reduce is exposed *by construction*). Fix: smaller gradient buckets (`bucket_cap_mb`) and backward prefetch so the collective starts while later layers are still computing; this is the entire subject of [overlapping compute and communication](/blog/machine-learning/distributed-training/overlapping-compute-and-communication).

**2. Data-loader stall (rhythmic dead air).** A periodic band where *neither* stream has a kernel — the GPU finished the batch and is waiting for the loader to hand it the next one. Cause: too few dataloader workers, no prefetching, unpinned host memory, or heavy per-sample transforms on the CPU. Fix: more `num_workers`, `prefetch_factor`, `pin_memory=True`, `persistent_workers=True` — the toolkit from [the data pipeline at scale](/blog/machine-learning/distributed-training/the-data-pipeline-at-scale).

**3. The straggler (one rank's step longer than its peers').** Only visible when you compare ranks: 63 ranks finish compute at 175 ms and spin in the barrier; one rank grinds to 290 ms and gates them all. Cause: usually thermal throttling on one physically-hot card, sometimes a bad NIC or a slow disk on one node. Fix: find it with per-rank timing, then cool it, evict it, or rebalance — the menu from [the straggler](/blog/machine-learning/distributed-training/the-straggler).

**4. Kernel-launch-bound (a lawn of tiny gaps).** Dozens of microscopic gaps between kernels, each one the GPU idle while the CPU issues the next launch. Cause: many small ops (small model, small batch, un-fused elementwise) so the per-op launch overhead — a few microseconds each — dominates, and the CPU can't queue kernels fast enough to keep the GPU fed. This is the signature `nsys` shows best. Fix: `torch.compile` to fuse ops and cut launch count, or CUDA graphs to replay a captured launch sequence with near-zero CPU cost.

**5. The pipeline bubble (idle stages at the edges).** In pipeline parallelism, a triangular region at the start (fill) and end (drain) of each step where some stages have no micro-batch to work on. Cause: the fundamental $(p-1)/(m+p-1)$ bubble fraction for $p$ stages and $m$ micro-batches. Fix: more micro-batches to amortize the bubble, or an interleaved 1F1B schedule — the mechanism from [pipeline parallelism and the bubble](/blog/machine-learning/distributed-training/pipeline-parallelism-and-the-bubble).

| Signature in the trace | What you see | Most likely cause | First fix |
|---|---|---|---|
| Exposed all-reduce | Fat NCCL block, compute stream empty above it | Bucket too big / data dependency | Smaller buckets, backward prefetch |
| Data-loader stall | Rhythmic gap, *both* streams idle | Too few workers, no prefetch | `num_workers`, `prefetch_factor`, `pin_memory` |
| Straggler | One rank's compute band longer than peers' | Thermal throttle, bad NIC/disk | Per-rank timing, then cool/evict/rebalance |
| Kernel-launch-bound | Lawn of tiny gaps between kernels | Many small ops, CPU launch overhead | `torch.compile`, CUDA graphs |
| Pipeline bubble | Triangular idle at step edges | $(p-1)/(m+p-1)$ fill/drain | More micro-batches, 1F1B |

The discipline this table encodes: **name the signature before you name the fix.** The most expensive mistake in performance work is to see a slow run, assume it's the comms because comms is the fashionable culprit, spend a day tuning buckets, and discover the real problem was a starving loader all along. The trace tells you the signature for free; read it first.

## 7. The measurement discipline (or: how to not lie to yourself)

A profile from a badly-measured run is worse than no profile, because it looks authoritative while being wrong. Five disciplines separate a number you can trust from a number that will send you tuning the wrong thing.

**Warm up first.** The first several steps of any run are unrepresentative. The caching allocator is still carving up memory and will `cudaMalloc` mid-step (a synchronizing stall that shows up as a giant gap). cuDNN is autotuning convolution and matmul algorithms, trying variants to pick the fastest. NCCL is doing its first-collective handshake, establishing rings and buffers. And on the very first step, lazy CUDA context initialization fires. Profile any of that and you are measuring cold-start, not steady state. Skip at least 5–10 steps — the `wait` in the profiler schedule — before you record anything.

**Synchronize before you read the clock.** CUDA kernels are *asynchronous*: `loss.backward()` returns to Python almost immediately, having only *queued* the GPU work, not waited for it. A naive `time.time()` around a training step therefore measures how long it took the CPU to *launch* the kernels, which can be a small fraction of how long the GPU takes to *run* them. You must call `torch.cuda.synchronize()` before reading the clock, or time with `torch.cuda.Event` (which records on the stream and is measured after a sync, as in the per-rank code above). Forget this and your "step time" is fiction — usually far too optimistic — and every downstream number is poisoned.

**Time steady state, not step 0, and average a few steps.** Even after warm-up, a single step is noisy — a checkpoint save, a learning-rate-schedule boundary, a garbage-collection pause can land on it. Record 3–10 representative steps and take the median or mean. The profiler's `active=3` records exactly this window; do not report the max or the first, report the steady-state central tendency.

**Respect the observer effect.** Profiling is not free. Tracing every op, and especially walking the Python stack with `with_stack=True`, adds overhead — sometimes 10–30% to the step time, more with stacks on. This means two things. First, do not report the *profiled* step time as your throughput; profile to find *where* the time goes, then measure throughput with the profiler *off*. Second, the overhead is not uniform — it falls harder on many-small-op (launch-bound) workloads than on a few-big-matmul workload — so a profile can slightly distort the very ratio you are trying to measure. Use it to locate the bottleneck, not to quote the final tokens/s. That final number comes from a clean, un-profiled, warmed-up, synchronized steady-state measurement.

**Profile the same step across ranks.** As §5 showed, the distributed-specific discipline is to capture the *identical* step number on multiple ranks so you can compare like with like. Comparing rank 0's step 100 to rank 41's step 137 tells you nothing; comparing everyone's step 100 tells you who the straggler is. When you export per-rank traces, tag them by rank (`trace_rank{rank}.json`) and align on step number, not wall-clock.

Get these five right and the profile means what it says. Skip them and you will confidently fix a problem you don't have.

## 8. Two reads from real traces

Signatures and disciplines are abstract until you walk a real one end to end. Here are two, with numbers, the way they actually go.

#### Worked example: the exposed all-reduce, before and after

An 8-GPU A100 run of a 1.5B transformer was scaling at only 60% efficiency — 4.9x over one GPU where the interconnect should have delivered better than 7x. Aggregate MFU said "slow"; it did not say why. The `torch.profiler` trace, opened in Perfetto and zoomed to one step, said everything: the backward pass finished at roughly 120 ms, and then a single fat `nccl:all_reduce` block ran *alone* from 120 ms to 200 ms with the compute stream empty above it. The all-reduce was fully exposed — 80 ms of communication sitting on the critical path, adding straight to every step.

![a before and after comparison of one exposed 80 millisecond all-reduce becoming a set of small overlapped buckets that lift efficiency from 60 to 91 percent](/imgs/blogs/profiling-a-distributed-run-7.webp)

Running `overlap_efficiency` on the trace confirmed the eye: $K = 80$ ms of NCCL, $T_\text{step} = 200$ ms, $C = 120$ ms, so $T_\text{exposed} = 200 - 120 = 80$ ms and $\eta_\text{overlap} = 1 - 80/80 = 0$. Zero percent of the communication was hidden. The cause was a single oversized gradient bucket: DDP was accumulating *all* gradients into one bucket and firing one giant all-reduce after the backward completed, so there was no earlier-layer compute left to overlap it with. The fix was to shrink the bucket so the all-reduce fires in pieces *during* the backward, each chunk overlapping the compute of the layers still to come:

```python
model = torch.nn.parallel.DistributedDataParallel(
    model,
    bucket_cap_mb=25,               # split gradients into ~25 MB buckets that overlap backward
    gradient_as_bucket_view=True,   # avoid an extra copy into the bucket
    static_graph=True,              # let DDP plan the overlap when the graph is fixed
)
```

Re-profiled, the trace showed the NCCL blocks now nested *inside* the backward band — overlapped — with only a 5 ms tail exposed at the end. The measured result:

| Metric | Before (1 bucket) | After (25 MB buckets) |
|---|---|---|
| Exposed comms per step | 80 ms | 5 ms |
| Step time | 200 ms | 128 ms |
| Overlap efficiency η | 0.00 | 0.94 |
| Scaling efficiency (8 GPU) | 60% | 91% |
| Tokens/s (aggregate) | ~48,000 | ~75,000 |

Same hardware, same bytes, same gradients — a 31-point swing in scaling efficiency, found in the trace in five minutes and fixed in one line. This is the exact mechanism [scaling a 7B LLM from 1 to 64 GPUs](/blog/machine-learning/distributed-training/scaling-a-7b-llm-1-to-64-gpus) leans on when it lists communication overlap as one of the biggest single levers on the road from 28% to 44% MFU.

#### Worked example: the data-loader stall

A different 8-GPU run, a fit and stable 7B FSDP job, sat at 33% MFU when the single-node ceiling should have been the low forties. The tell was in `nvidia-smi dmon` before we ever opened a profiler: GPU utilization sawtoothing between 95% and 30% on a steady rhythm, roughly once per step. That periodicity is the data-loader signature — the GPU burns through a batch, then waits for the loader to assemble the next one, then burns through it, over and over.

```log
# nvidia-smi dmon -s u  (utilization %, one line per second)
# gpu    sm   mem
    0    94    61
    0    31    18     <- stall: loader assembling next batch
    0    95    62
    0    29    17     <- stall again, same rhythm
    0    93    60
```

The profiler confirmed it and located the gap precisely. In the `torch.profiler` trace, each step ended with a band where *both* the compute stream and the NCCL stream were empty — not an exposed collective (the comms stream was idle too), but pure dead air, the GPU waiting on the host. `torch.profiler`'s own data-loading annotation (`enumerate(DataLoader)` shows up as time spent outside any GPU op) put a number on it: about 22 ms of the 175 ms step was the GPU idle waiting for data. The fix was pure plumbing, not model surgery:

```python
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=micro_bsz,
    num_workers=8,             # 8 CPU processes assembling batches in parallel
    prefetch_factor=4,         # each worker stages 4 batches ahead -> 32-batch pipeline
    pin_memory=True,           # page-locked host memory -> async H2D copy over DMA
    persistent_workers=True,   # don't respawn workers every epoch
    sampler=torch.utils.data.distributed.DistributedSampler(dataset),
)
```

Eight workers with a four-deep prefetch queue kept a full pipeline of 32 batches always staged, so the GPU never waited. The sawtooth flattened to a solid 95% in `dmon`, the dead-air band vanished from the trace, and MFU climbed from 33% to 38% with *zero* change to the model. Five points of MFU that were never a compute problem — they were a data-pipeline problem, and the only reason we knew to fix the loader instead of the model is that the trace showed both streams idle, not the comms stream running alone. That distinction — dead air versus an exposed collective — is the whole diagnostic value of the profile.

#### Worked example: the kernel-launch-bound step

The third signature is the sneakiest because the ops all look healthy in isolation. A 350M model — small by 2026 standards — trained on 8 GPUs was stuck at a step time that made no arithmetic sense: sum the kernel times in the `torch.profiler` table and they came to about 9 ms, yet the measured step was 15 ms. Six milliseconds a step, roughly 40%, were unaccounted for. `torch.profiler` could not show where; the ops were individually fine. This is exactly the escalation trigger from §3 — ops fine, wall-clock isn't — so we ran `nsys`, and the system timeline answered instantly: between almost every pair of kernels sat a tiny gap of a few microseconds where the GPU was idle, waiting for the CPU to issue the next `cudaLaunchKernel`. A lawn of hundreds of these per step. The model was so small and its ops so numerous that the per-launch overhead — a fixed few microseconds of CPU work per kernel — could not keep the fast GPU fed. The GPU was starving not for data but for *instructions*.

The fix is to cut the number of launches or eliminate the per-launch CPU cost. `torch.compile` fuses adjacent elementwise and normalization ops into single kernels, cutting the launch count; CUDA graphs go further, capturing a whole step's launch sequence once and replaying it with a single call, so the CPU issues one graph launch instead of hundreds of individual ones.

```python
# Two levers against launch overhead. Compile fuses ops; CUDA graphs replay a captured launch sequence.
model = torch.compile(model, mode="reduce-overhead")   # 'reduce-overhead' enables CUDA graphs internally

# For a hand-rolled CUDA-graph capture of a static step (advanced, but the biggest launch-overhead win):
g = torch.cuda.CUDAGraph()
optimizer.zero_grad(set_to_none=True)
# ... run a few warmup iters first so cuDNN/allocator settle ...
with torch.cuda.graph(g):
    static_loss = train_step(model, static_batch)      # captured once
# Thereafter, one replay issues the entire step's kernels with ~zero CPU launch cost:
for batch in loader:
    copy_into(static_batch, batch)
    g.replay()
```

Re-profiled with `nsys`, the lawn of gaps collapsed: `torch.compile` alone cut the launch count enough to drop the step from 15 ms to about 11 ms (a fused kernel does the work of several with one launch), and the CUDA-graph replay took it to roughly 9.5 ms — within a hair of the pure kernel time, meaning the launch overhead was essentially gone. The lesson generalizes: **launch-bound is a small-model and small-batch disease**, and it is invisible to a per-op profiler because no single op is slow — the waste is in the spaces between them, which only a system-wide timeline like `nsys` renders.

| Metric | Baseline | `torch.compile` | + CUDA graphs |
|---|---|---|---|
| Step time | 15 ms | 11 ms | 9.5 ms |
| Kernel time (sum) | 9 ms | 8.5 ms | 8.5 ms |
| Launch-gap overhead | ~6 ms | ~2.5 ms | ~1 ms |
| Effective speedup | 1.0x | 1.36x | 1.58x |

## 9. Case studies and real numbers

**Megatron-LM's MFU as a profiling target.** NVIDIA's Megatron-LM papers report Model FLOPs Utilization in the 40–52% range for large GPT models on A100 clusters with tuned tensor, pipeline, and data parallelism. Those numbers are not luck; they are the *result* of exactly the profiling loop in this post — capture the timeline, find the exposed collective or the bubble, tune the parallelism degrees and micro-batch count until the gaps close, re-measure. The published MFU is the score after the profiler has done its job. Treat mid-40s MFU on A100 (and low-to-mid 40s or better on H100) as the band a well-profiled dense-transformer run should reach; if you are well below it, the trace will show you which of the five signatures is eating the gap.

**The straggler that cost \$90,000.** In [the straggler war story](/blog/machine-learning/distributed-training/the-straggler), a single H100 with a failing fan throttled from 1900 MHz to 1200 MHz, stretching its step from 175 ms to about 290 ms and dragging a 64-GPU job from roughly 680,000 to 410,000 tokens/s — a 40% loss, invisible on every dashboard because all 64 GPUs read 100% utilization. Per-rank timing (the CUDA-event technique from §5) fingered the culprit in one line; a facilities fix to the fan restored full throughput. The lesson for profiling specifically: the aggregate throughput told them there was a 40% problem, and *only* per-rank timing told them it was one physical card in one slot, not a software bug in the training code they spent the first hour suspecting.

**The 7B MFU journey.** In [scaling a 7B LLM from 1 to 64 GPUs](/blog/machine-learning/distributed-training/scaling-a-7b-llm-1-to-64-gpus), the run climbs from 28% MFU on eight GPUs to 44% by fixing signatures in order: the data-loader stall (28% → 33%, the plumbing fix above), bf16 (33% → 39%), a bigger batch via activation checkpointing (39% → 42%), and communication overlap (42% → 44%). Every one of those steps was *found* by measuring where the time went, not by guessing. The post's own summary — "the biggest single lever was the pair that kept the GPU busy: feeding it and hiding its comms" — is a profiling conclusion: you cannot know which lever is biggest without a trace that attributes the wasted time.

**Reading the pipeline bubble in a trace.** When a run uses pipeline parallelism, the profiler shows the bubble directly as triangular idle at the edges of each step: at the start, the later stages have no micro-batch yet (fill); at the end, the earlier stages have nothing left to do (drain). The idle fraction is $(p-1)/(m+p-1)$ for $p$ stages and $m$ micro-batches, so for 4 stages and 8 micro-batches the bubble is $3/11 \approx 27\%$ — over a quarter of the timeline is stage-idle, and you will see it as clear triangular gaps in the per-stage NVTX bands. Push to 16 micro-batches and the fraction drops to $3/19 \approx 16\%$; the trace's triangles shrink correspondingly. The profiler is how you *confirm* the schedule is filling the pipe: if you switched to an interleaved 1F1B schedule and expected the bubble to shrink, the trace is where you verify the idle triangles actually got smaller rather than trusting the config flag. This is the mechanism from [pipeline parallelism and the bubble](/blog/machine-learning/distributed-training/pipeline-parallelism-and-the-bubble), now read off a timeline instead of a formula.

**What the numbers should look like.** For calibration when you profile your own run: on a well-connected 8-GPU A100 node, DDP or FSDP on a dense transformer should hit 90%+ scaling efficiency and low-to-mid 40s MFU. NVLink4 delivers on the order of 900 GB/s aggregate per H100, InfiniBand HDR about 200 Gb/s per link — so a multi-node all-reduce is roughly an order of magnitude slower per byte than an intra-node one, which is why exposed comms bites hardest the moment you cross the node boundary. If your intra-node run is at 60% efficiency, the trace will almost always show an exposed collective or a starving loader; if your *multi-node* run drops from a healthy single-node number, suspect the interconnect fell back to a slow path, and profile with `NCCL_DEBUG=INFO` alongside the timeline to see which algorithm and transport NCCL actually chose. (These interconnect figures are approximate vendor peaks; your achieved bandwidth is always lower, and the profiler measures what you actually got.)

## 10. When to reach for the profiler (and when not to)

Profiling is cheap relative to the compute it saves, but it is not free of your time, and there is a right moment.

**Reach for `torch.profiler` first, always, when a distributed run is slower than it should be.** It is one context manager, it answers the two most common questions (which op dominates, is comms overlapped), and it costs you five minutes. Any time your MFU is more than a few points below the band for your hardware, this is the first move — before you change a single knob. The trace tells you which of the five signatures you have, which tells you which knob to turn.

**Escalate to Nsight Systems when the ops look individually fine but the step is still slow.** That specific situation — every op's kernel time is reasonable, yet the wall-clock step doesn't add up — is the fingerprint of launch overhead or a cross-stream stall, and `nsys` is the tool that draws those gaps. It is also the right tool when you suspect the CPU is the bottleneck (launch-bound), because it traces the CPU-side launch calls that `torch.profiler` doesn't center.

**Use per-rank CUDA-event timing continuously, even without a profiler.** The straggler check in §5 is cheap enough to leave running in every job — a few CUDA events and an all-gather per step. It is the standing early-warning for the single most expensive and most invisible failure mode at scale, and it costs almost nothing.

**Don't profile before you have a symptom.** If your run is already at the MFU band for your hardware and the loss is descending on schedule, there is nothing to find; profiling a healthy run is a way to feel productive without being productive. **Don't profile the whole run** — a schedule that skips cold steps and records a few steady-state ones tells you everything a 10,000-step trace would, at a thousandth the size. **Don't quote profiled throughput as your real throughput** — the observer effect inflates step time; profile to locate, measure clean to report. And **don't tune from a cold, un-synced, single-step measurement** — that number is fiction, and a fix "validated" against fiction is a coin flip. If you only remember one rule: the profiler tells you *where*, and you only earn the right to trust its answer by measuring with the discipline of §7.

For the broader debugging context — where to even look first when a distributed job misbehaves, and how the profiler fits into the rest of the toolkit — see [debugging distributed jobs](/blog/machine-learning/distributed-training/debugging-distributed-jobs) and, for the throughput-as-north-star framing and its confounds, [throughput regressions](/blog/machine-learning/distributed-training/throughput-regressions). The whole decision-and-debugging checklist ties together in [the distributed training playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook).

## Key takeaways

- **The aggregate number is the symptom; the timeline is the diagnosis.** Tokens/s and MFU tell you a run is slow. Only a profiler, which keeps the time dimension, tells you *where* the time went — and five different causes produce the identical MFU.
- **Kill GPU-Util as a health metric.** It measures "is a kernel running," never "is the kernel useful." A spinning NCCL wait reads 100%. Use a profiler that names kernels and shows gaps.
- **Read every trace as three lanes:** compute, NCCL comms, and the gaps. The gaps are where the waste lives; the shape of the gap names the cause.
- **`torch.profiler` for "which op / is comms overlapped"; Nsight Systems for "the whole timeline including launch gaps."** Start with the former, escalate to the latter when the ops look fine but the step doesn't.
- **Compute the overlap efficiency, don't eyeball it:** $\eta_\text{overlap} = 1 - (T_\text{step} - C)/K$. Near 1 means comms is hidden; near 0 means bucketing is your next move.
- **Learn the five signatures cold:** exposed all-reduce (fat NCCL block alone), loader stall (rhythmic dead air, both streams idle), straggler (one rank's band longer), launch-bound (lawn of tiny gaps), pipeline bubble (triangular idle at edges). Name the signature *before* the fix.
- **Profile the same step across ranks.** The distributed-specific move: overlay the identical step from several ranks to find the straggler, because the barrier gates on the slowest and per-rank timing is the only thing that shows it.
- **Measure with discipline or don't measure:** warm up, `torch.cuda.synchronize()` before the clock, average steady-state steps, respect the observer effect (profile to locate, measure clean to report), align ranks by step number.

## Further reading

- **PyTorch Profiler documentation** — the `schedule`, `record_shapes`, `profile_memory`, `with_stack`, and `on_trace_ready` API, plus the TensorBoard and Chrome-trace exporters.
- **NVIDIA Nsight Systems User Guide** — `nsys profile`, trace domains (`-t cuda,nvtx,nccl,osrt`), `--capture-range`, and reading the system-wide timeline.
- **NVTX documentation** — `torch.cuda.nvtx.range_push/pop` and `torch.profiler.record_function` for labeling phases.
- **Megatron-LM** (Shoeybi et al.; Narayanan et al., "Efficient Large-Scale Language Model Training on GPU Clusters") — reported MFU targets and the profiling-driven tuning behind them.
- [Overlapping compute and communication](/blog/machine-learning/distributed-training/overlapping-compute-and-communication) — the mechanism the profiler verifies.
- [The data pipeline at scale](/blog/machine-learning/distributed-training/the-data-pipeline-at-scale) and [the straggler](/blog/machine-learning/distributed-training/the-straggler) — the two signatures with full war stories.
- [Scaling a 7B LLM from 1 to 64 GPUs](/blog/machine-learning/distributed-training/scaling-a-7b-llm-1-to-64-gpus) and [why distributed training](/blog/machine-learning/distributed-training/why-distributed-training) — the run this profiling loop is applied to, and the four-walls frame it sits in.
- [The distributed training playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook) — the capstone checklist that ties profiling into the whole decision-and-debugging flow.
