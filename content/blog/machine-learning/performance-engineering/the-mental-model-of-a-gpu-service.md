---
title: "The Mental Model of a GPU Service: How a Request Really Spends Its Time"
date: "2026-07-14"
publishDate: "2026-07-14"
description: "You called model(x) and it returned in 40 ms. This is where those milliseconds actually went, why timing without a synchronize measures nothing, and how to see the idle GPU the profiler is hiding from you."
tags:
  [
    "performance-engineering",
    "gpu-optimization",
    "profiling",
    "pytorch",
    "cuda",
    "latency",
    "throughput",
    "inference",
    "ml-systems",
  ]
category: "machine-learning"
subcategory: "Performance Engineering"
author: "Hiep Tran"
featured: true
readTime: 30
---

You called `model(x)`. Forty milliseconds later it returned a tensor. Where did those 40 ms go?

Almost nobody on an ML team can answer that question, and that ignorance is expensive. It is the reason a service runs at 30% GPU utilization while the bill reads \$40 per hour. It is the reason a "GPU-Util 100%" line in `nvidia-smi` coexists with a GPU that is idle most of the time. It is the reason a team spends a sprint buying a bigger GPU when the actual bottleneck was a Python `for` loop on the host. You cannot fix waste you cannot see, and a request's 40 ms is invisible by default. This post makes it visible.

By the end you will have one durable mental model of what happens when you call a model on a GPU: your CPU is a *producer* that enqueues kernels, your GPU is a *consumer* that drains them off a stream asynchronously, and `model(x)` mostly just *launches* work and returns before any of it finishes. From that single model, everything else in this series falls out — why timing without `torch.cuda.synchronize()` measures nothing, why "util 100%" lies, why a small model at batch 1 can leave an A100 two-thirds idle, and which of the four wastes (idle GPU, low occupancy, the bandwidth wall, redundant work) is eating your particular request. We will break one concrete 40 ms request into four buckets, derive the launch-overhead law that predicts host-boundness, and then reach for the three cheapest tools that show it: a corrected manual timer, `nvidia-smi dmon`, and a first `torch.profiler` trace.

![a host cpu enqueues launches onto a cuda stream that fans out to a compute engine and a copy engine which merge back into a host result](/imgs/blogs/the-mental-model-of-a-gpu-service-1.webp)

Look at the figure above before we go further, because it is the whole post in one picture. The CPU does not *run* your matrix multiply. It writes a small command — "run kernel K with these arguments" — into a queue and moves on. Two engines on the GPU drain that queue: the streaming multiprocessors (SMs) that do the arithmetic, and a separate copy engine that moves bytes across the PCIe bus. They finish independently and their results merge into the tensor you eventually read back on the host. The call returned when the *last command was written*, not when the *last result was computed*. Hold onto that gap between "enqueued" and "finished" — it is where your 40 ms hides.

This is the second post in the **Profiling & Optimizing AI Services** series. The [intro post](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu) laid out the four wastes and the profile → hypothesize → fix → measure loop. This post builds the substrate that loop runs on: a correct picture of where a request's time goes, so that when the profiler shows you a gap, you know what it means.

## Your CPU produces work; your GPU consumes it

Here is the one sentence that explains most GPU performance mysteries: **the host and the device run on two different clocks, connected by a queue.**

When you write `y = model(x)` in eager PyTorch, the Python interpreter walks the model's `forward`, and for every tensor operation it hits — a `conv2d`, a `linear`, a `layernorm`, an `add` — it does *not* compute the result on the spot. It records a **kernel launch**: a request that the GPU run a specific compiled function (a *kernel*) with specific arguments. That request is placed on a **CUDA stream**, which is nothing more exotic than an ordered FIFO queue of work for the GPU. The GPU pulls launches off the stream in order and executes them on its SMs. Crucially, the two sides are **asynchronous**: the CPU does not wait for a kernel to finish before enqueuing the next one. It fires and forgets, racing ahead to launch the next op while the GPU is still chewing on the last.

Two pieces of vocabulary you will use for the rest of the series, defined once:

- **Host** = the CPU, the Python process, system RAM. This is where your code, the framework, and the CUDA driver's launch machinery live.
- **Device** = the GPU, its SMs, and its high-bandwidth memory (HBM). This is where the kernels actually run.
- A **kernel** is a function compiled to run on the GPU (e.g. cuBLAS's matmul, a cuDNN convolution, or a fused elementwise op).
- A **kernel launch** is the host-side act of telling the GPU to run one kernel. It costs CPU time — roughly 5–10 µs of driver and dispatch work per launch — even though the kernel itself runs on the device.
- A **stream** is an ordered queue of launches. Work on one stream runs strictly in order; the *default stream* is the single stream everything lands on unless you say otherwise.

The reason this matters is that the host has a job that has nothing to do with arithmetic: it has to keep the stream *fed*. If the queue ever runs dry — if the GPU finishes the last enqueued kernel before the CPU has managed to enqueue the next — the GPU has nothing to do and sits idle. That idle time is invisible in your Python code, which is happily off launching the next op, and it is invisible in a naive timer. But it is real, and on small workloads it is often the *majority* of your wall-clock.

![a vertical stack showing a python op descending through aten dispatch then the cuda driver then the stream queue and finally reaching sm execution](/imgs/blogs/the-mental-model-of-a-gpu-service-2.webp)

The figure traces what one `model(x)` line actually costs on the host before the GPU does anything. Your Python op descends the framework's dispatch stack: PyTorch's ATen dispatcher resolves the operator and dtype (a few microseconds of pure host work), the CUDA driver packages a launch and pushes it onto the stream queue, and only at the bottom does an SM pick it up and compute. The top four layers are all **host cost**. They are paid on the CPU, per op, whether or not the GPU is busy. A model is not "one launch"; a ResNet-50 forward is on the order of a thousand kernels, and a Transformer block is dozens per layer. Multiply the per-op host cost by the number of ops and you get a number that can easily exceed the GPU's compute time — which is exactly when the GPU starves.

### Why timing without a synchronize measures nothing

Now we can explain the single most common mistake in GPU benchmarking. Because launches are asynchronous, `model(x)` returns as soon as the last kernel has been *enqueued*. If you wrap it in a wall-clock timer, you are timing the enqueue, not the execution:

```python
import torch, time

model = build_model().cuda().eval()
x = torch.randn(1, 3, 224, 224, device="cuda")

# WRONG: no synchronize
t0 = time.perf_counter()
with torch.no_grad():
    y = model(x)
t1 = time.perf_counter()
print(f"{(t1 - t0) * 1e3:.2f} ms")   # prints ~0.30 ms — a lie
```

That `0.30 ms` is not the time to run the model. It is the time for the CPU to walk the `forward` and shove ~2000 launches onto the stream. The GPU has barely started. If you believe this number, you will conclude your model is 100x faster than it is, ship it, and then be baffled when production latency is 40 ms. To measure the truth, you must block the CPU until the GPU has actually drained the stream:

```python
import torch, time

# RIGHT: synchronize before and after
torch.cuda.synchronize()          # make sure nothing is in flight
t0 = time.perf_counter()
with torch.no_grad():
    y = model(x)
torch.cuda.synchronize()          # WAIT for the GPU to finish the stream
t1 = time.perf_counter()
print(f"{(t1 - t0) * 1e3:.2f} ms")   # prints ~40 ms — the truth
```

`torch.cuda.synchronize()` blocks the host until every kernel on the device has completed. Now the interval spans enqueue *plus* execution, and you get the real 40 ms. Everything downstream in this series — every before/after number, every profiler reading — depends on you never forgetting this call. The [reproducible-benchmark post](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark) makes it a discipline; here it is just the direct consequence of the async model. We will draw the wrong-versus-right timer as a figure once we have the numbers to put on it.

## Where a request's 40 ms actually goes

Let's take that honest 40 ms and split it into buckets you can attack independently. A single inference request on one GPU spends its wall-clock in four places:

1. **H2D copy** — moving the input from host RAM to device HBM across the PCIe bus (host-to-device).
2. **Compute kernels** — the SMs doing the actual arithmetic. This is the only bucket that is "real work."
3. **D2H copy** — moving the output back from device to host (device-to-host).
4. **Sync / idle wait** — time the host spends blocked, plus time the GPU spends idle between kernels waiting for the next launch.

![a matrix mapping the four time buckets to their typical share of wall clock and the specific waste that attacks each one](/imgs/blogs/the-mental-model-of-a-gpu-service-3.webp)

The matrix pins concrete numbers to a plausible host-bound request and names the waste that owns each row. Notice the shape of it: the compute bucket — the only bucket that produces your answer — is 30% of the wall-clock. The *idle-wait* bucket is 55%. More than half of what you are paying for is the GPU standing around. That is not a pathological case I invented to make a point; it is the default state of a small model at batch 1, and it is exactly the case people misdiagnose as "I need a faster GPU." A faster GPU would shrink the 30% compute bucket and leave the 55% idle bucket untouched — you would pay more per hour to make the *smaller* half of the problem slightly smaller.

Each bucket has its own fix, and matching the fix to the bucket is the entire skill:

| Bucket | What it is | The waste | The fix (later in this series) |
|---|---|---|---|
| H2D copy | input host → device | copy tax | pinned memory + `non_blocking=True`, overlap with compute |
| Compute kernels | SM arithmetic | low occupancy / memory-bound | better kernels, fusion, `channels_last`, the roofline |
| D2H copy | output device → host | copy tax | `non_blocking=True`, keep results on device |
| Sync / idle wait | GPU idle between launches | host-bound (idle GPU) | CUDA graphs, `torch.compile`, bigger batches |

The rest of this post is about that last row, because it is the one most people have never looked at and the one that most often dominates. If your idle-wait bucket is small — if the GPU is already back-to-back busy — then you are GPU-bound, and you should be reading the [roofline post](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) about the compute bucket instead. Knowing *which bucket* you are in is the fork in the road; everything after it is a different set of tools.

![a timeline of one request showing an input copy then two short kernels separated by a wide idle gap then an output copy and a final wait](/imgs/blogs/the-mental-model-of-a-gpu-service-4.webp)

The timeline shows the idle-wait bucket in motion. The GPU runs `kernel 1` for six microseconds, then *nothing* while the CPU scrambles up the launch path to enqueue `kernel 2`, then six more microseconds of compute, then another gap. The kernels are the thin slivers; the gaps between them are the fat bars. Stack ten thousand of these and the GPU's own timeline is mostly whitespace even though the CPU is pinned at 100%. This is the picture the profiler will draw for you later — dense CPU-op lane on top, sparse CUDA-kernel lane below with gaps between the slivers — and once you have seen it here you will recognize it on sight.

#### Worked example: splitting a real ResNet-50 request

Take a ResNet-50 inference service, batch 1, on an A100 80GB SXM (312 dense bf16 TFLOP/s, 2.0 TB/s HBM2e — NVIDIA's published peak figures). One request, measured honestly with `synchronize`, comes back at 40 ms. We instrument the four buckets (we'll see the exact tools in a moment):

- **H2D copy**: the input is a `1×3×224×224` float image, about 600 KB. Across a PCIe Gen4 link (~25 GB/s effective for a pageable transfer) that is well under a millisecond of transfer, but with an unpinned buffer and a synchronizing copy it lands around 3 ms including the stall. Bucket: ~3 ms.
- **Compute kernels**: ResNet-50 at batch 1 is roughly 4 GFLOPs of actual math. On a 312 TFLOP/s GPU that is arithmetically ~13 µs of pure FLOPs, but batch 1 leaves the SMs badly underfilled, so the real SM-busy time is ~12 ms. Bucket: ~12 ms.
- **D2H copy**: a 1000-class logit vector is 4 KB. Trivial transfer, dominated by the launch and sync overhead of the copy. Bucket: ~1 ms.
- **Sync / idle**: the remainder, ~24 ms, is the GPU sitting idle between the model's ~2000 kernels while the CPU launches them, plus the final blocking wait. Bucket: ~24 ms.

Add it up: 3 + 12 + 1 + 24 = 40 ms, and the GPU was *busy* for only 12 of them. Utilization by the honest definition (SM-busy / wall) is 30%. That 24 ms idle bucket is not a mystery once you have the mental model: the CPU cannot launch 2000 kernels fast enough to keep a GPU that drains each one in six microseconds fed. Which brings us to the law that predicts exactly this.

## The launch-overhead law: predicting host-boundness before you profile

You can predict whether a workload will be host-bound with arithmetic, before you touch a profiler. The host has to pay a fixed cost per kernel launch. Call it $t_\text{launch}$ — the CUDA driver and dispatch work to enqueue one kernel, empirically 5–10 µs on a typical server CPU. A model with $N_\text{kernels}$ kernels therefore incurs a total host-side launch cost:

$$t_\text{overhead} = N_\text{kernels} \cdot t_\text{launch}$$

That is the **launch-overhead law**, and it is the most useful back-of-the-envelope in this series. Compare it to the total GPU compute time $t_\text{compute}$ (the sum of SM-busy time across all kernels). Two regimes fall out:

$$\text{host-bound} \iff t_\text{overhead} \gt t_\text{compute}, \qquad \text{GPU-bound} \iff t_\text{overhead} \lt t_\text{compute}$$

When $t_\text{overhead} \gt t_\text{compute}$, the CPU is the slower producer: it cannot enqueue launches as fast as the GPU drains them, so the stream empties, the GPU idles, and the wall-clock is set by the host. The utilization you'll observe is approximately:

$$\text{util} \approx \frac{t_\text{compute}}{\max(t_\text{overhead},\, t_\text{compute})}$$

This is worth internalizing because it says something non-obvious: **on a host-bound workload, a faster GPU does not help.** It shrinks $t_\text{compute}$, which is already the smaller term, and leaves $t_\text{overhead}$ — the term that sets your wall-clock — completely unchanged. You would pay for an H100 and get almost nothing, because the bottleneck was never the arithmetic. The only levers that help a host-bound service are the ones that shrink $N_\text{kernels}$ (fusion, CUDA graphs) or $t_\text{launch}$ (fewer, cheaper launches via graph replay).

One honest caveat: $t_\text{launch}$ in the law above is the pure CUDA-launch cost. In eager PyTorch each op also pays Python interpreter time and ATen dispatch on top, so the *full* per-op host cost is often 15–20 µs, not 7. The law with $t_\text{launch} \approx 7\,\mu s$ gives you the CUDA-side floor; the real host cost is that plus the framework tax above it (the top layers of the launch-path figure). Use the floor as a lower bound and remember the real number is worse.

#### Worked example: does ResNet-50 at batch 1 starve an A100?

Plug the numbers in. ResNet-50 forward is roughly $N_\text{kernels} \approx 2000$ kernels (convolutions, batchnorms, ReLUs, the final matmul — many are tiny). Take $t_\text{launch} \approx 7\,\mu s$:

$$t_\text{overhead} = 2000 \times 7\,\mu s = 14{,}000\,\mu s = 14\text{ ms}$$

That is just the CUDA-launch floor. Layer the Python and dispatch cost on top (say ~17 µs all-in per op) and the real host time is closer to:

$$t_\text{host} = 2000 \times 17\,\mu s = 34\text{ ms}$$

Against $t_\text{compute} \approx 12\text{ ms}$, we have $t_\text{host} = 34 \gt 12 = t_\text{compute}$. Host-bound, decisively. Predicted utilization:

$$\text{util} \approx \frac{12}{34} \approx 35\%$$

Which is exactly the 30–35% we measured. The law told us the answer before we opened a profiler. Now the fix is obvious in shape: collapse those 2000 launches into one. That is what CUDA graphs do — capture the whole kernel sequence once and *replay* it with a single launch, so $N_\text{kernels}$ on the host effectively becomes 1. The [kernel-launch-overhead post](/blog/machine-learning/performance-engineering/the-kernel-launch-overhead-problem) goes deep on measuring $t_\text{launch}$ directly; here the point is that you could see this coming with multiplication.

![a before and after contrast of a host-bound service at thirty-five percent utilization versus a gpu-bound service at eighty-five percent](/imgs/blogs/the-mental-model-of-a-gpu-service-5.webp)

The before/after makes the punchline concrete, and it is the most important figure in this post: **the compute did not change.** Twelve milliseconds of arithmetic on the left, twelve milliseconds on the right. The only thing that changed is the host overhead wrapped around it — 34 ms of launching in the "before," ~2 ms of graph replay in the "after." Utilization climbs from 35% to 85% not because the GPU got faster but because we stopped making it wait. This is the single most common win in production ML inference, and it is worth restating as a principle: *when you are host-bound, you optimize the host, not the device.* People reach for a bigger GPU here roughly nine times out of ten, and roughly nine times out of ten it is the wrong move.

## Host-bound or GPU-bound? Read the race

Everything so far reduces to a race between two rates. Model it as a producer and a consumer sharing one queue: the CPU *produces* launches at some rate (launches per second it can issue), and the GPU *consumes* them at some rate (kernels per second it can execute). The stream queue between them is the shared buffer.

![a producer consumer graph where a cpu refill rate and a gpu drain rate both meet at a stream queue that branches to either a full queue or an empty queue](/imgs/blogs/the-mental-model-of-a-gpu-service-6.webp)

The graph shows both rates meeting at the queue and the two outcomes that branch from it. If the CPU refills faster than the GPU drains, the queue stays full, the GPU always has a next kernel waiting, and it runs back-to-back — GPU-bound, high utilization. If the GPU drains faster than the CPU refills — which happens whenever kernels are tiny and numerous — the queue empties, the GPU hits the bottom and idles, and you are host-bound. In our ResNet example the GPU drains a kernel every ~6 µs (≈170k kernels/s) while the CPU refills one every ~17 µs (≈60k launches/s). The consumer is three times faster than the producer, so the queue is empty most of the time and the GPU idles. Same queue, opposite conclusion, entirely determined by which rate is slower.

This gives you a **diagnostic signature** you can read off two numbers without any deep profiling:

| Symptom | Host-bound (queue empties) | GPU-bound (queue full) |
|---|---|---|
| CPU utilization | pinned near 100% on one core | moderate, waiting on the GPU |
| GPU util (`nvidia-smi`) | low-to-moderate, jumpy (35%) | high and steady (85–95%) |
| GPU timeline (trace) | short kernels with wide gaps | kernels packed back-to-back |
| Effect of a faster GPU | ~none | proportional speedup |
| Effect of a bigger batch | often fixes it | little help, may OOM |
| The right fix | fewer/cheaper launches | better kernels, roofline work |

The trap in that table is the GPU-util row, and it deserves a warning because it is the single most misleading number in ML infrastructure. `nvidia-smi` reporting "GPU-Util: 100%" does **not** mean the GPU is doing useful work 100% of the time. That counter reports the fraction of the last sampling window during which *at least one kernel was running* — it goes to 100% if a single tiny kernel was active at any point in the sample. A host-bound service that launches one 6 µs kernel every 60 µs can show high util because the sampler keeps catching a kernel mid-flight, while the GPU is genuinely idle 90% of the time. The honest utilization metric is *SM-busy time over wall-clock*, which you get from the profiler, not from `nvidia-smi`'s headline number. The [metrics post](/blog/machine-learning/performance-engineering/metrics-that-actually-matter) takes this apart in detail — util versus occupancy versus MFU — because getting fooled by the util counter is the most common way teams waste a quarter chasing the wrong bottleneck.

## Seeing it with the cheapest tools first

You do not need Nsight to diagnose host-boundness. Reach for the cheapest tool that answers the question and only escalate when it doesn't. Here is the ladder, from a thirty-second check to a full trace.

### Tool 1: a corrected manual timer (and CUDA events)

The wrong-versus-right timer from earlier already tells you a lot. If your no-sync time is sub-millisecond and your with-sync time is 40 ms, the ratio *is* the diagnosis: the host spent 0.3 ms enqueuing work that took the device 40 ms to run, so the device is doing something and the question is only whether it did it back-to-back or with gaps.

![a before and after contrast of a timer with no synchronize reading a fraction of a millisecond versus a timer with a synchronize reading forty milliseconds](/imgs/blogs/the-mental-model-of-a-gpu-service-7.webp)

The figure is the two timers side by side: the no-sync path reads 0.3 ms because `model(x)` returned right after the last enqueue, while the with-sync path reads 40 ms because it waited for the GPU to actually drain the stream. Same code, same request, a 130x difference — and the wrong one is the one people paste into a benchmark spreadsheet. For steady-state numbers, though, even the synced wall-clock timer includes Python and OS jitter. The right tool for GPU-side timing is **CUDA events**, which are recorded *on the stream* and measure device time directly:

```python
import torch

model = build_model().cuda().eval()
x = torch.randn(1, 3, 224, 224, device="cuda")

start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)

# Warm up: first calls pay cuDNN autotune + allocator + JIT costs.
with torch.no_grad():
    for _ in range(10):
        model(x)
torch.cuda.synchronize()

# Time 50 steady-state iterations on the stream.
start.record()
with torch.no_grad():
    for _ in range(50):
        y = model(x)
end.record()
torch.cuda.synchronize()          # events are only valid after sync
print(f"{start.elapsed_time(end) / 50:.2f} ms/iter")   # ~40.0 ms/iter
```

`start.record()` and `end.record()` drop timestamps onto the stream itself, so `elapsed_time` returns the wall time the *GPU* spent between them, excluding host jitter. The warm-up loop matters: the first few iterations pay one-time costs (cuDNN algorithm autotuning, allocator warm-up, lazy CUDA context init) that would otherwise pollute your average. This event-based, warmed-up, synced pattern is the backbone of every honest measurement in this series.

### Tool 2: nvidia-smi dmon, a live pulse

Before any code, you can watch the GPU breathe. `nvidia-smi dmon` prints a one-line-per-second monitor of the device:

```bash
nvidia-smi dmon -s um -d 1
```

`-s um` selects utilization and memory columns, `-d 1` samples every second. Running our host-bound service, it prints:

```console
# gpu   pwr gtemp  sm   mem   enc   dec  mclk  pclk
# Idx     W     C   %     %     %     %   MHz   MHz
    0    98    54  34     8     0     0  1593  1410
    0   101    55  36     9     0     0  1593  1410
    0    97    54  33     8     0     0  1593  1410
```

The `sm` column hovering at 34% is the tell. On a genuinely GPU-bound workload it would sit at 90%+ and the power draw would be near the card's limit (400 W for an A100 SXM, not 98 W). A 98 W draw on a 400 W card is a GPU that is mostly asleep. Notice `dmon`'s `sm` here is closer to the truth than `nvidia-smi`'s headline util because it is a coarser SM-activity sample, but it is still a sample — treat it as a pulse, not a stopwatch. If it reads low and steady, you are host-bound and you can stop here and go fix launches.

### Tool 3: the first torch.profiler trace

When you want to *see* the gaps rather than infer them, `torch.profiler` is the right first real profiler. It records both the CPU-op lane and the CUDA-kernel lane and lets you export a Chrome trace you can scroll through:

```python
import torch
from torch.profiler import profile, schedule, ProfilerActivity, tensorboard_trace_handler

model = build_model().cuda().eval()
x = torch.randn(1, 3, 224, 224, device="cuda")

# wait 1 step, warm up 1, record 3, once.
sched = schedule(wait=1, warmup=1, active=3, repeat=1)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=sched,
    on_trace_ready=tensorboard_trace_handler("./tb_trace"),
    record_shapes=True,
    with_stack=False,
) as prof:
    with torch.no_grad():
        for _ in range(5):
            y = model(x)
            prof.step()             # advances the wait/warmup/active schedule

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

The `schedule(wait, warmup, active, repeat)` argument is what makes this trustworthy: it skips the first step, warms up on the second, and only records steps 3–5, so your trace is steady-state and not polluted by cold-start costs. The printed summary is the first thing to read:

```console
-------------------------  ------------  ------------  ------------  ------------
                     Name    Self CPU %    Self CPU     Self CUDA    Self CUDA %
-------------------------  ------------  ------------  ------------  ------------
             aten::conv2d        18.2%      6.180 ms      4.021 ms         33.5%
             aten::addmm         11.7%      3.970 ms      2.110 ms         17.6%
        aten::batch_norm         14.9%      5.055 ms      1.240 ms         10.3%
              aten::relu_         12.1%      4.110 ms      0.902 ms          7.5%
             aten::add_          9.8%      3.320 ms      0.640 ms          5.3%
-------------------------  ------------  ------------  ------------  ------------
Self CPU time total: 33.9 ms
Self CUDA time total: 12.0 ms
```

Read the two totals at the bottom, not the rows. **Self CPU time total: 33.9 ms** against **Self CUDA time total: 12.0 ms** is the host-bound signature in two numbers: the host spent nearly three times as long dispatching as the device spent computing. That ratio is the launch-overhead law showing up in a real table. Then open the Chrome trace (`chrome://tracing` or [ui.perfetto.dev](https://ui.perfetto.dev), load `tb_trace/*.json`) and you will see the timeline from figure four made real: a dense, unbroken CPU-op lane on top, and a sparse CUDA-kernel lane below where thin kernel slivers float in a sea of gaps. The gaps *are* the idle bucket. You can measure one: hover the whitespace between two kernels and read ~10–15 µs — the time the CPU took to climb the launch path and enqueue the next op.

Here is the tool ladder as a decision table, because knowing which to reach for is half the craft:

| Tool | What it sees | Overhead | Reach for it when |
|---|---|---|---|
| Corrected timer + CUDA events | end-to-end latency, steady-state | ~none | you need one honest number |
| `nvidia-smi dmon` | live SM% + power pulse | ~none | quick "is it idle?" check |
| `torch.profiler` + Chrome trace | CPU vs CUDA lanes, per-op time, the gaps | ~5–30% | you need to *see* where time goes |
| Nsight Systems (`nsys`) | system-wide timeline, memcpy, NVTX | moderate | the gap crosses CPU/GPU/copy |
| Nsight Compute (`ncu`) | one kernel to the metal, occupancy | very high | a single kernel is the bottleneck |

Start at the top and stop the moment a tool answers your question. Most host-bound diagnoses never need to descend past `torch.profiler`. When the gap is more tangled — spanning copies, kernels, and CPU threads at once — you graduate to [Nsight Systems](/blog/machine-learning/model-serving/profiling-llm-serving-with-nsight), which shows the whole system on one timeline. But do not start there; a `dmon` pulse and a synced timer answer the host-bound question in under a minute.

## Streams, the default stream, and the concurrency you're leaving on the table

One last piece completes the mental model. Everything above happened on a *single* stream — the default stream that every op lands on unless you say otherwise. A single stream is a single ordered queue: kernel B cannot start until kernel A finishes, even if they are completely independent. That is often fine (your model's layers genuinely depend on each other in order), but it means two things worth naming.

First, on a single stream the H2D copy of the *next* request cannot overlap the compute of the *current* one — they serialize, and the copy tax is paid in full on the critical path. Pinned memory plus `non_blocking=True` plus a second stream lets the copy engine move the next input while the SMs are still busy on this one, hiding the copy behind compute. That is a real, separate win from the launch-overhead fix, and it lives in the [copy-elimination post](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark).

Second, if you run two small independent models on one GPU on the same default stream, they cannot overlap even though neither fills the SMs — each waits behind the other's kernels in the one queue. Give each its own stream and the GPU can interleave their kernels, filling the SMs that a single small model leaves empty. This is the concurrency the single-stream default leaves on the table, and it is the subject of the dedicated [overlapping-streams post](/blog/machine-learning/performance-engineering/overlapping-streams-and-concurrency) later in the series. For now, just carry the picture: the default stream is *one* lane; the GPU has room for several; whether you use them is a choice.

The reason streams matter *here*, in the mental-model post, is that they explain why "the GPU is a consumer" is the right frame. The GPU is not a single serial engine you feed one thing at a time — it is a pool of SMs and copy engines that can drain multiple queues at once. Host-boundness is the failure to keep even *one* queue full. Concurrency is the opportunity, once you can, to keep *several* full. Both come from the same producer/consumer picture; you just add more consumers.

## Case studies: the launch tax in the wild

These are published results where removing host overhead — not faster arithmetic — produced the win. I've kept the numbers to what the primary sources report and flagged where I'm rounding.

**PyTorch CUDA Graphs on small-batch inference.** NVIDIA and Meta's PyTorch engineering write-ups on CUDA Graphs describe exactly the pattern in this post: workloads dominated by many small kernel launches, where the CPU cannot keep the GPU fed, see substantial end-to-end speedups from capturing and replaying the kernel sequence as a single launch. The reported wins are largest precisely where the launch-overhead law predicts — small batches, many tiny kernels, models where per-launch host cost rivals per-kernel compute. On genuinely GPU-bound workloads the same technique buys almost nothing, which is the law running in reverse. (See the PyTorch blog "Accelerating PyTorch with CUDA Graphs.")

**`torch.compile(mode="reduce-overhead")`.** PyTorch 2.x's compile stack has a mode named, literally, `reduce-overhead`, which composes Inductor's kernel fusion with CUDA graphs. The naming is a tell: the PyTorch team built a first-class mode specifically for the host-bound case, because it is that common. Fusion shrinks $N_\text{kernels}$ (fewer, larger kernels) and the CUDA-graph capture drives the residual launch cost toward a single replay. The documented caveat is equally instructive — `reduce-overhead` needs static-ish shapes and extra memory for the graph pool, so it helps host-bound services and can hurt shape-varying ones. That trade-off is the whole [compile-plus-graphs post](/blog/machine-learning/model-serving/kernel-fusion-cuda-graphs-torch-compile) in the model-serving series.

**LLM decode is host-bound by construction.** Autoregressive decoding generates one token at a time; at batch 1 each step is a sequence of tiny matmuls and elementwise ops — a textbook many-small-kernels workload. The `gpt-fast` project and similar minimal LLM inference stacks report large per-token latency reductions from CUDA graphs, because decode is launch-bound: the GPU finishes each tiny kernel long before the CPU can launch the next. The arithmetic is trivial; the launches are the wall. This is why every serious LLM serving stack graphs its decode loop, and why [continuous batching](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention) — which raises the effective batch and thus the work-per-launch — is the other half of the fix.

The common thread: in all three, the GPU's arithmetic was never the bottleneck. The host's ability to *dispatch* was. That is the failure mode you now have a model for.

## When this mental model earns its keep (and when to ignore it)

The producer/consumer frame is not a fix; it is a *diagnosis*. It tells you which half of the problem you have. Use it, but do not over-apply it.

**Reach for the host-bound lens when:** your GPU util (measured honestly, SM-busy / wall) is well below 80%; your batch size is small (1–8); your model has many small ops (an LLM decode step, a small CNN, anything with lots of pointwise/norm layers); your `torch.profiler` Self-CPU total exceeds your Self-CUDA total. These are the signatures that the queue is emptying and the CPU is the wall.

**Do not reach for it when** the profiler shows the GPU already packed back-to-back at 90% util. Then you are GPU-bound and the launch-overhead law says fixing launches buys you nothing — every microsecond you save on the host is spent waiting behind a full queue. Your problem is inside the compute bucket: low occupancy, a memory-bound kernel hitting the bandwidth wall, or redundant work that fusion would remove. That is the [roofline](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) and [SM-internals](/blog/machine-learning/high-performance-computing/inside-the-gpu-sms-warps-and-the-simt-execution-model) territory, a different set of tools (`ncu`, occupancy, warp-stall reasons) and a different post.

**And do not chase host overhead before you've confirmed it's host overhead.** The number-one way to waste a day is to CUDA-graph a service that was actually memory-bound, ship a change that does nothing, and lose faith in the tool. Run the cheap check first: synced timer, `dmon` pulse, `torch.profiler` CPU-vs-CUDA totals. Two numbers tell you which bucket you're in. Only then pick the fix.

### Stress-testing the model: what changes at the edges?

A mental model you can't break is a model you don't understand. Push on it:

- **Batch 1 versus batch 64.** Host-boundness is a small-batch disease. At batch 64 each kernel does 64x the arithmetic — the compute bucket grows — while the number of launches is *unchanged* (still ~2000 for ResNet-50; the launch count depends on the op graph, not the batch). So $t_\text{compute}$ rises past $t_\text{overhead}$ and the workload flips to GPU-bound. The same code, same GPU, larger batch, and suddenly util climbs to 90% with no other change. This is why "just increase the batch size" is the first thing a serving engineer tries — it directly moves you along the producer/consumer race by making each launch worth more.
- **L4 versus A100.** The L4 has far less compute (~121 fp16 TFLOP/s dense, ~300 GB/s memory — an inference-class card, figures approximate) than an A100. Slower compute means $t_\text{compute}$ is *larger* on the L4, which paradoxically makes the *same* launch overhead a smaller fraction — so a workload that's host-bound on an A100 can be closer to GPU-bound on an L4. But L4 instances usually pair with weaker host CPUs, which raises $t_\text{launch}$, pushing the other way. The lesson is that host-boundness is a property of the *host-and-device pair*, not the GPU alone. Always re-measure on the target hardware; do not port a diagnosis across cards.
- **Small model versus large.** A 70B-parameter LLM prefill on a long prompt is GPU-bound (huge matmuls, few relative launches). The *decode* step of the same model at batch 1 is host-bound (tiny matmuls, one per layer per token). Same model, opposite regime, depending on the phase. You can be both, in the same request, in different phases — which is why you profile the phase, not the model.
- **Shapes that vary every request.** The CUDA-graph fix assumes static shapes: a captured graph replays the *exact* kernel sequence with the *exact* tensor sizes. If every request has a different sequence length or image size, naive capture breaks or you thrash re-capturing. The fix is shape bucketing (pad to a few fixed sizes and keep one graph per bucket), covered in the CUDA-graphs-in-serving post. The mental model still holds; the *implementation* of the fix gets more careful.

## Key takeaways

- **The CPU produces launches; the GPU consumes them off a stream, asynchronously.** `model(x)` returns after the *last launch is enqueued*, not after the *last result is computed*. That gap is where your wall-clock hides.
- **Timing without `torch.cuda.synchronize()` measures the enqueue, not the execution.** Always synchronize before and after, or use CUDA events on the stream. A no-sync timer can under-report by 100x.
- **A request's wall-clock splits into four buckets: H2D copy, compute kernels, D2H copy, and sync/idle wait.** On a small-batch service the idle-wait bucket is often the largest — and a faster GPU shrinks the wrong one.
- **The launch-overhead law, $t_\text{overhead} = N_\text{kernels} \cdot t_\text{launch}$, predicts host-boundness before you profile.** If it exceeds compute time, the queue empties and the GPU idles; util ≈ compute / max(overhead, compute).
- **`nvidia-smi` "GPU-Util 100%" lies.** It reports "a kernel was running at some point in the sample," not useful-work fraction. Trust SM-busy / wall from the profiler.
- **Host-bound is fixed on the host, not the device.** Fewer, cheaper launches — CUDA graphs, `torch.compile`, bigger batches — not a bigger GPU.
- **Diagnose before you fix.** Synced timer, `dmon` pulse, `torch.profiler` CPU-vs-CUDA totals. Two numbers tell you the bucket. Reach for the cheapest tool that answers the question and stop there.
- **Host-boundness is a property of the host-and-device pair and the batch size**, not of the GPU alone. Re-measure on the target hardware and the target batch; a diagnosis does not port.

## Further reading

- [PyTorch Profiler tutorial](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) — the `schedule`, `key_averages`, and Chrome-trace export used above.
- [PyTorch: Accelerating PyTorch with CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/) — the primary source for the launch-overhead win and the `reduce-overhead` mode.
- [CUDA C++ Programming Guide: asynchronous concurrent execution](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution) — streams, the default stream, and async launch semantics from the source.
- [Why your AI service wastes CPU and GPU](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu) — the series intro: the four wastes and the profile → fix → measure loop this post feeds.
- [Metrics that actually matter](/blog/machine-learning/performance-engineering/metrics-that-actually-matter) — util versus occupancy versus MFU, and what each metric lies about.
- [The kernel-launch-overhead problem](/blog/machine-learning/performance-engineering/the-kernel-launch-overhead-problem) — measuring $t_\text{launch}$ directly and the CPU-bound signature in depth.
- [Inside the GPU: SMs, warps, and the SIMT execution model](/blog/machine-learning/high-performance-computing/inside-the-gpu-sms-warps-and-the-simt-execution-model) — what the consumer side actually is, once you go below the stream.
- [The performance engineering playbook](/blog/machine-learning/performance-engineering/the-performance-engineering-playbook) — the capstone decision tree this post is one branch of.
