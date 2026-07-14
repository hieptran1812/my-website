---
title: "Setting Up a Reproducible Benchmark: How Not to Fool Yourself Measuring GPU Code"
date: "2026-07-14"
publishDate: "2026-07-14"
description: "Every before-and-after number in this series rests on one skill: measuring GPU code honestly. Learn why an unsynchronized timer lies, how CUDA events tell the truth, why the first iterations don't count, and how to lock the machine down so your speedup is real and not noise."
tags:
  [
    "performance-engineering",
    "gpu-optimization",
    "benchmarking",
    "profiling",
    "pytorch",
    "cuda",
    "latency",
    "throughput",
    "ml-systems",
  ]
category: "machine-learning"
subcategory: "Performance Engineering"
author: "Hiep Tran"
featured: true
readTime: 35
---

An engineer on your team pings the channel: "Swapped the attention implementation, it's **3x faster**." The graph looks great. The change ships. A week later the p99 latency dashboard for the service has not moved a millimeter, the GPU bill is identical, and nobody can reproduce the 3x on a fresh box. What happened is the most common failure in all of performance work, and it has nothing to do with attention. They timed the wrong thing. Specifically: they timed the very first (cold) iteration of the baseline, which paid for cuDNN autotuning and CUDA context creation; they never called `torch.cuda.synchronize()`, so their timer measured how long Python took to *launch* the kernels, not how long the GPU took to *run* them; and the baseline happened to run while the card was thermally throttling from the previous job. Three separate measurement bugs stacked into one beautiful, fake 3x.

This post is the foundation the rest of the [Profiling & Optimizing AI Services](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu) series stands on. Every later post makes a claim of the form "this fix took GPU utilization from 30% to 85%" or "CUDA graphs cut p99 from 180 ms to 40 ms." Those numbers are worthless — worse than worthless, actively misleading — if the measurement underneath them is wrong. So before we touch a single optimization, we are going to build the one thing that makes all the others trustworthy: a reproducible benchmark. By the end you will be able to time a GPU workload so that the number you report is the number the service actually feels, know exactly how many iterations to throw away and how many to keep, report a distribution instead of a lucky single reading, lock the machine down so it stops lying to you, and tell — with a real threshold, not a vibe — whether a change is a genuine win or noise.

![a two column comparison showing an unsynchronized wall clock reading a fake 0.2 milliseconds against a cuda event timer reading the true 11 milliseconds for the same forward pass](/imgs/blogs/setting-up-a-reproducible-benchmark-1.webp)

The figure above is the whole problem in one frame, and we will earn every part of it. On the left, a `time.time()` call wrapped around `model(x)` returns in 0.2 ms and someone excitedly reports the model "runs in a fraction of a millisecond." On the right, the same forward pass measured with CUDA events reports 11 ms — the truth. The 50x gap between them is not a speedup and not a slowdown; it is the difference between measuring the launch and measuring the compute. If you internalize only one thing from this post, make it this: **on a GPU, the call returning is not the work finishing.**

## The one mistake that fools everyone: GPU work is asynchronous

Start with the mechanism, because once you see it you will never write the naive timer again. When you call `y = model(x)` in PyTorch on a CUDA device, the Python thread does *not* wait for the matrix multiplies to complete. It enqueues each kernel — a small packet of "run this function with these arguments on this data" — onto a CUDA **stream**, which is an ordered queue the GPU drains on its own schedule. Enqueuing is cheap: a few microseconds of CPU work per kernel to build the launch and hand it to the driver. The moment the last kernel is queued, Python's `model(x)` call *returns*, and your next line of code runs — while the GPU is very possibly still chewing through the first layer.

This is the host/device split, the single most important idea in GPU performance. The **host** is the CPU running your Python. The **device** is the GPU. They run concurrently and communicate through the stream. A kernel **launch** is the host-side act of queuing work; kernel **execution** is the device-side act of doing it. They are separated in time, sometimes by a lot. Now look at what a naive timer does:

```python
import time, torch

model = build_model().cuda().eval()
x = torch.randn(32, 3, 224, 224, device="cuda")

# WRONG: this measures the launch, not the compute.
t0 = time.time()
with torch.no_grad():
    y = model(x)
dt = time.time() - t0
print(f"forward: {dt*1e3:.3f} ms")   # prints something like 0.187 ms
```

The `t0 = time.time()` records the host clock. `model(x)` enqueues, say, 220 kernels onto the stream and returns. `time.time() - t0` is the wall-clock time it took the *host* to push 220 launches into the queue — a couple hundred microseconds — and has almost nothing to do with the 11 ms the GPU will spend actually computing `y`. The printed 0.187 ms is real, but it answers a question nobody asked ("how long did Python take to dispatch?") while looking exactly like the answer to the question everyone cares about ("how long does inference take?").

![a branching dataflow where the cpu path launches a kernel and races ahead to read the clock and print a wrong number while the gpu path is still running until a synchronize point rejoins them](/imgs/blogs/setting-up-a-reproducible-benchmark-3.webp)

The branch in the figure is the async trap made visual. After the launch, two paths run at once. The CPU path returns immediately, reads `t1`, and prints 0.2 ms — a number that is *correct for the CPU* and *wrong for the workload*. The GPU path is still grinding through 11 ms of matmuls, obliviously. The only thing that reunites the two timelines is a **synchronize** — an explicit "host, wait here until the device has finished everything queued so far." Until you insert that barrier, any host-side clock you read is measuring the launch.

### The two correct fixes

There are exactly two right ways to time GPU work, and you should know both. The first is to force the host to wait, then use a host clock:

```python
import time, torch

# RIGHT (option 1): synchronize before BOTH timestamps, use a wall clock.
torch.cuda.synchronize()          # make sure nothing earlier is still running
t0 = time.perf_counter()
with torch.no_grad():
    y = model(x)
torch.cuda.synchronize()          # WAIT for the GPU to actually finish
dt = time.perf_counter() - t0
print(f"forward: {dt*1e3:.3f} ms")   # now prints ~11.4 ms
```

Two things changed and both matter. There is a `synchronize()` *before* `t0` so that any leftover work from a previous line is not counted inside your window, and a `synchronize()` *before* reading the end time so the host blocks until the GPU has truly finished. Note also `time.perf_counter()` instead of `time.time()`: `perf_counter` is a monotonic high-resolution timer that never jumps backward when NTP adjusts the system clock, which is what you want for measuring short intervals. This version is correct. Its only cost is that the number includes the `synchronize()` overhead (roughly 5 to 20 µs of host-side latency) plus whatever Python overhead sat between the timestamps — usually negligible against an 11 ms kernel, but it matters for tiny workloads, as we will see.

The second fix is better for isolating pure GPU time, and it is what you should reach for by default: **CUDA events**. An event is a marker you insert *into the stream itself*. The GPU timestamps the marker as it drains past it, on the device clock, so the measurement never includes host-side Python or dispatch latency at all.

```python
import torch

# RIGHT (option 2): CUDA events time the device region, excluding host overhead.
start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)

torch.cuda.synchronize()
start.record()                    # queue a timestamp marker onto the stream
with torch.no_grad():
    y = model(x)
end.record()                      # queue the closing marker
torch.cuda.synchronize()          # wait until the GPU has passed 'end'
dt_ms = start.elapsed_time(end)   # milliseconds, measured on the device clock
print(f"forward: {dt_ms:.3f} ms")   # ~11.2 ms, the truest number
```

`start.record()` and `end.record()` do not block — they queue markers, cheaply, just like kernel launches. The single `synchronize()` after `end.record()` makes the host wait until the GPU has processed the closing marker, and only then is `elapsed_time` valid. The value it returns is the device-clock time between the two markers: pure GPU region time, with the Python and launch overhead squeezed out. This is the gold standard for timing a kernel or a bounded region, and it is what our harness will use.

Here is the mechanism stated as a small law, because it explains *why* the wrong timer produces the number it does. If a forward pass launches $N$ kernels and each launch costs the host $t_\text{launch}$ of dispatch time, then an unsynchronized wall-clock timer measures approximately

$$t_\text{wrong} \approx N \cdot t_\text{launch},$$

whereas the true device time is $t_\text{true} = \sum_i t_{\text{kernel},i}$, the sum of how long each kernel actually runs. With $N = 220$ kernels and $t_\text{launch} \approx 6\ \mu\text{s}$, the wrong timer reads about 1.3 ms of pure launch — or even less, because the launches pipeline and the driver batches them, which is how you get numbers as small as 0.2 ms. Meanwhile $t_\text{true} = 11\ \text{ms}$. The two quantities are unrelated; there is no fixed ratio, so you cannot "correct" a wrong measurement after the fact. You have to measure right from the start. (There is one exception that makes the wrong timer *sometimes* accidentally close: if the launch queue fills up, the driver blocks the host inside a launch until the GPU catches up, so a very deep, long-running model can back-pressure the CPU and make the wall clock partly reflect GPU time. Relying on that is a coin flip. Never do it.)

## Wall time versus device time: which clock to trust

Now that we have three ways to read a clock — unsynchronized wall, synchronized wall, and CUDA events — plus the two profilers we will lean on later, it is worth laying them side by side. They do not measure the same thing, and choosing the wrong one silently changes your answer.

![a five row grid classifying timing methods by what they measure by their overhead and by when to use each with the unsynchronized wall clock marked as never valid](/imgs/blogs/setting-up-a-reproducible-benchmark-4.webp)

The matrix above is the cheat sheet. Read it as: what does this method actually measure, what does it cost, and when should you reach for it. The written-out version, with the reasoning:

| Method | Measures | Overhead | Use when |
|---|---|---|---|
| Wall clock, no sync | Launch/dispatch time only | Near zero | Never — this is the trap |
| Wall clock + `synchronize()` | Region wall time incl. host + sync | One sync per read (~10 µs) | Quick smoke test, coarse end-to-end |
| CUDA events | Pure device region time | Under 1 µs per marker | Default for timing kernels or a bounded GPU region |
| `torch.profiler` | Per-op CPU + CUDA time, shapes, memory | 5–30% while active | Finding *which* op dominates, not the headline number |
| `nsys` (Nsight Systems) | System-wide timeline: API, kernels, copies, NVTX | Low, sampling-based | Seeing the whole picture across CPU, GPU, and copies |

The distinction between the two profilers and the two timers is worth stating plainly, because people confuse them. A **timer** answers "how long did this take?" with one number you can put in a table. A **profiler** answers "where did the time go?" with a breakdown. You benchmark with a timer and you *diagnose* with a profiler. Reaching for `torch.profiler` to get a headline latency number is a mistake — the profiler's own instrumentation overhead (5% to 30% depending on `record_shapes` and `with_stack`) inflates the very number you are trying to measure. Use events for the number, use the profiler to explain it. The next post in the series, [Profiling PyTorch with torch.profiler](/blog/machine-learning/performance-engineering/profiling-pytorch-with-torch-profiler), lives in that second world; this post is about getting the number itself right first.

#### Worked example: the tiny kernel where launch dominates

Consider two workloads on the same A100 80GB SXM. Workload A is one big matmul, `(8192, 8192) @ (8192, 8192)` in bf16 — about 1.1 TFLOP of work. Workload B is an elementwise add on a 1024-element vector — about 1024 FLOP, effectively nothing. Time both three ways:

```console
workload A: big matmul (8192^3 bf16)
  wall, no sync :   0.21 ms   <- launch time, meaningless
  wall + sync   :   3.83 ms
  cuda events   :   3.79 ms   <- true device time

workload B: tiny elementwise add (1024 elems)
  wall, no sync :   0.008 ms  <- launch time
  wall + sync   :   0.019 ms  <- dominated by the sync + launch
  cuda events   :   0.006 ms  <- true kernel time: 6 microseconds
```

The big matmul is compute-bound: wall-plus-sync and events agree to within 1%, because the 3.8 ms of real work dwarfs the ~10 µs of sync overhead. The tiny add tells the opposite story. Its *kernel* runs in 6 µs, but wall-plus-sync reads 19 µs — three times too high — because the sync barrier and launch dispatch are now the same size as the work. For workload B, only CUDA events give you the truth, and even then the honest conclusion is "this kernel is launch-bound; the 6 µs of compute is drowned by the ~6 µs it costs the CPU just to launch it." That is the whole thesis of the [kernel-launch-overhead](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu) problem in miniature, and you cannot even *see* it without a timer that measures device time. The lesson: the smaller the kernel, the more the timing method matters, and the more you must use events.

## Warmup: the first iterations are lying too

Suppose you have fixed the async bug and you are timing with events. You run the model once, get 47 ms, run it again, get 11 ms, and run it a third time and get 11 ms. Which is right? The 11 ms — the 47 ms was the *cold* iteration, and it is lying for a completely different reason than the async timer did. The first few iterations of any GPU workload pay a set of one-time costs that never recur, and if you include them in your measurement you are averaging in taxes the steady-state service will never pay.

![an ordered protocol timeline that discards a cold initialization and a fifty iteration warmup then times a thousand steady state iterations and reports the percentiles](/imgs/blogs/setting-up-a-reproducible-benchmark-2.webp)

The timeline above is the protocol every honest benchmark follows, and warmup is its first act. Here is what the discarded early iterations are actually paying for:

- **Lazy CUDA context creation.** The first CUDA call in a process initializes the driver context — allocating the primary context, loading the module, setting up the memory pools. This is hundreds of milliseconds, once per process.
- **cuDNN / cuBLAS autotuning.** With `torch.backends.cudnn.benchmark = True`, cuDNN runs a mini-benchmark of several convolution algorithms the first time it sees a given input shape and caches the winner. That search happens on iteration one for each new shape. cuBLAS similarly picks a GEMM algorithm on first use.
- **Allocator growth.** The CUDA caching allocator grows its pool by calling `cudaMalloc` as your tensors demand memory. Those `cudaMalloc` calls are slow and happen mostly in the first few iterations, until the pool is large enough to satisfy every allocation from cache.
- **`torch.compile` / JIT compilation.** If you compiled the model, the first call through each code path triggers Dynamo tracing and Inductor codegen — often *seconds*. It happens once per traced shape.

You can model the whole thing as a decay. Let $t_\infty$ be the steady-state per-iteration latency and $c_k$ be the one-time cost still being paid at iteration $k$. Then the measured latency of iteration $k$ is

$$t_k = t_\infty + c_k, \qquad c_k \to 0 \text{ as } k \text{ grows}.$$

The $c_k$ term is large at $k = 0$ (context init, first-shape autotune) and decays over the next handful of iterations (allocator growth settling). **Warmup** is simply: run $W$ iterations without timing them, until $c_k$ has decayed into the noise, then start measuring. How big should $W$ be? Enough to get past the last one-time cost. For a plain eager model, 10 to 50 warmup iterations is plenty. For a `torch.compile`d model, the *first* iteration alone is the compile, so you need at least one full pass through every shape you will serve, and often 50+ to let the allocator and any lazy paths settle. When in doubt, plot it:

```python
import torch

start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)

lat = []
for i in range(80):
    torch.cuda.synchronize()
    start.record()
    with torch.no_grad():
        y = model(x)
    end.record()
    torch.cuda.synchronize()
    lat.append(start.elapsed_time(end))

for i in (0, 1, 2, 3, 5, 10, 20, 40, 79):
    print(f"iter {i:3d}: {lat[i]:6.2f} ms")
```

```console
iter   0:  312.44 ms   <- context init + cudnn autotune + first cudaMalloc
iter   1:   18.90 ms   <- allocator still growing
iter   2:   13.71 ms
iter   3:   12.05 ms
iter   5:   11.38 ms
iter  10:   11.22 ms   <- steady state reached here
iter  20:   11.19 ms
iter  40:   11.21 ms
iter  79:   11.20 ms
```

The curve settles. Iteration 0 is 28x the steady-state value; by iteration 10 the workload is flat at 11.2 ms and stays there. If you had averaged all 80 iterations you would report `(312 + 19 + 14 + ... ) / 80 ≈ 15.9 ms` — a number 40% too high, dominated by a single cold iteration that a warm service never repeats. Discard the warmup, and you measure the 11.2 ms the service actually delivers. The rule is: **find where the curve goes flat, discard everything before it, measure everything after.**

## One number is not a measurement: report the distribution

You have warmed up and you are timing with events. You run once and get 11.2 ms. Ship it? No — because 11.2 ms is a *sample*, not the latency. Run the timed loop 1000 times and you will not get 11.2 ms a thousand times; you will get a *distribution*, and the shape of that distribution is where all the interesting behavior of a real service hides. A single number throws that away.

The two seductive summaries to avoid are the **minimum** and the **mean**. The minimum ("the model can do 10.9 ms") describes a best case the service almost never hits — it is the number vendors quote and the number that never survives contact with production. The mean is worse for latency work: latency distributions are **right-skewed** (there is a floor set by the compute, but the tail can stretch far when a GC pause or a co-tenant or a periodic sync lands), and a single fat outlier drags the mean up while telling you nothing about the typical case. Report **percentiles** instead:

- **p50** (the median) — the typical latency. Half of requests are faster, half slower. This is your honest "usual" number.
- **p90 / p95** — the mild tail. Where the distribution starts to spread.
- **p99** — the tail your users actually feel. In a service that fans one page-load into 100 backend calls, the *slowest of the 100* is what the user waits for, and that is a p99-shaped event. p99 is not paranoia; it is the number that shows up in your SLO.

There is a clean way to think about the tail. Write

$$p99 = p50 + \text{stall},$$

where "stall" is whatever intermittent cost lands on roughly 1% of iterations. If your p99 sits right on top of your p50, the service is smooth and you should optimize the *median* (make the common case faster). If your p99 towers over your p50, the service has a **stall** — a periodic sync, a garbage-collection pause, a recompilation, a memory-allocator hiccup — and chasing the median is a waste; you hunt the stall. Two services with identical p50 and wildly different p99 need completely different fixes, and a benchmark that reports only a mean cannot tell them apart. This distinction drives an entire later post, [the p99 latency tail](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu), but it starts here, at measurement.

How many iterations do you need? Enough that the percentile you care about is stable. A useful rule of thumb for estimating the $p$-th percentile: you want the tail to contain enough samples that it does not swing on a single draw, so you need roughly $n \cdot (1 - p) \gtrsim 10$ samples above the percentile. For p99 that means $n \cdot 0.01 \gtrsim 10$, i.e. $n \gtrsim 1000$. For a stable p50 you need far fewer — 100 is plenty — but if you want to *quote* a p99 you should be timing at least 1000 iterations, and ideally a few thousand. The standard error of the median also shrinks like $1/\sqrt{n}$, so more iterations tighten every estimate; there is rarely a reason to time fewer than a few hundred once warmup is cheap. Report it like this:

```python
import numpy as np

lat = np.array(lat_ms)            # the timed (post-warmup) samples
print(f"n      = {len(lat)}")
print(f"mean   = {lat.mean():.2f} ms   (do not headline this)")
print(f"stddev = {lat.std():.2f} ms")
print(f"min    = {lat.min():.2f} ms   (best case, misleading)")
print(f"p50    = {np.percentile(lat, 50):.2f} ms")
print(f"p90    = {np.percentile(lat, 90):.2f} ms")
print(f"p99    = {np.percentile(lat, 99):.2f} ms")
```

```console
n      = 1000
mean   = 11.63 ms   (do not headline this)
stddev = 1.44 ms
min    = 11.02 ms   (best case, misleading)
p50    = 11.21 ms
p90    = 11.44 ms
p99    = 17.90 ms   <- the tail: ~1% of iters hit a stall
```

Look at what the single number would have hidden. The mean (11.63) sits above the median (11.21) — the signature of a right skew. The min (11.02) is a fantasy nobody experiences. And the p99 (17.9 ms) is *60% higher* than the p50, which is a giant flashing sign that this workload has a stall on about 1% of iterations. If you had reported "11.2 ms" you would have shipped a service whose users routinely wait 18 ms and never known why. The distribution is the measurement; the single number is a rumor about it.

## Controlling the machine: lock the clocks, watch the thermals

You can time perfectly and still get garbage if the machine underneath you is moving. A modern datacenter GPU does not run at one fixed speed. It **boosts** its clock when it is cool and has power headroom, and it **throttles** — drops the clock — when it gets hot or hits a power cap. That means the exact same kernel can take 11 ms on a cool card and 14 ms on the same card twenty minutes into a benchmark, purely because the SM clock sagged from 1410 MHz to 1200 MHz under thermal load. If you measure your baseline on a hot card and your optimized version on a cool one, you can manufacture a speedup — or erase a real one — without changing any code.

![a two state comparison of an unlocked gpu whose clock sags and whose latency drifts to fourteen milliseconds against a clock locked gpu that stays flat at eleven milliseconds and cooler](/imgs/blogs/setting-up-a-reproducible-benchmark-7.webp)

The fix is to take the clock out of the equation. Before a benchmark, pin the GPU to a fixed clock and disable auto-boost, then watch the thermals and power to confirm it is holding:

```bash
# Requires root/admin on the box. Persistence mode keeps settings across idle.
sudo nvidia-smi -pm 1                    # enable persistence mode

# Lock the graphics (SM) clock to a fixed value, e.g. 1410 MHz on an A100.
# Query supported clocks first:
nvidia-smi -q -d SUPPORTED_CLOCKS | head -40
sudo nvidia-smi -lgc 1410,1410           # lock SM clock: min=max=1410 MHz
# Optionally lock memory clock too on cards that allow it:
sudo nvidia-smi -lmc 1215                 # lock memory clock (A100 HBM2e)

# ... run your benchmark here ...

# Afterward, release the locks so the card can boost normally again:
sudo nvidia-smi -rgc
sudo nvidia-smi -rmc
```

While the benchmark runs, keep an eye on the card with `dmon`, which prints a one-line-per-second dump of the metrics that reveal throttling:

```bash
nvidia-smi dmon -s pucm -d 1
# -s pucm = power, utilization, clocks, memory;  -d 1 = every 1 second
```

```console
# gpu    pwr  gtemp  mtemp     sm    mem    enc    dec   mclk   pclk
# Idx      W      C      C      %      %      %      %    MHz    MHz
    0    398     64      -     99     71      0      0   1215   1410
    0    401     66      -    100     72      0      0   1215   1410
    0    399     67      -    100     71      0      0   1215   1410
    0    402     68      -    100     72      0      0   1215   1410
```

That is what a healthy, clock-locked run looks like: `pclk` (the processor/SM clock) pinned at 1410, `gtemp` climbing gently but under control, `sm` utilization at 100%. Now compare with what an unlocked, throttling run looks like — the `pclk` column drifts *down* as the temperature climbs, and your latency drifts *up* in lockstep. That drift is the difference between figure seven's left panel and its right panel, and it is entirely an artifact of not controlling the machine.

Beyond clocks, a controlled benchmark environment means: pin the process to specific CPU cores and the right NUMA node (`numactl --cpunodebind=0 --membind=0 python bench.py`) so the host thread is not migrating across sockets mid-measurement; set `torch.set_num_threads()` and `OMP_NUM_THREADS` to a fixed value so thread-pool contention is constant; make sure no other tenant is on the GPU (`nvidia-smi` should show only your process); kill background jobs on the box; and fix everything about the workload — batch size, sequence length, dtype — and seed every RNG (`torch.manual_seed`, and set `torch.backends.cudnn.deterministic` if the kernel choice itself is a variable you want to freeze). The goal is that the *only* thing changing between your A run and your B run is the one line of code you are testing.

![a vertical stack ranking measurement noise sources from cold start and thermal throttle at the top down through unlocked clocks and dataloader starvation to garbage collection and rng variance at the bottom](/imgs/blogs/setting-up-a-reproducible-benchmark-5.webp)

The stack above ranks the noise sources roughly worst to least, and it is a useful triage order. A cold start or a throttling GPU can move a number by tens of percent; unlocked clocks and a starving dataloader by ten to twenty; GC pauses and RNG variance mostly just fatten the tail. Fix them top-down: warmup kills the cold start, clock-locking kills the throttle and the boost drift, isolation kills the neighbor, and only once those are handled does chasing the last few percent of variance make sense.

#### Worked example: the thermal throttle on a long run

Here is the throttle, measured. Take a ResNet-50 forward at batch 32 on an A100 80GB SXM, bf16, and run it 10,000 timed iterations back to back with **auto-boost left on** (clocks unlocked). Record the CUDA-event latency and, from `dmon`, the SM clock and temperature at intervals:

| Iteration | GPU temp | SM clock | p50 latency (window) |
|---|---|---|---|
| 0–500 | 58 °C | 1410 MHz | 11.2 ms |
| 2000–2500 | 71 °C | 1350 MHz | 11.8 ms |
| 5000–5500 | 80 °C | 1275 MHz | 12.9 ms |
| 9000–9500 | 84 °C | 1200 MHz | 14.1 ms |

Nothing about the code changed across those 10,000 iterations. The workload got 26% slower purely because the card heated up and dropped its clock 15% to stay inside its thermal envelope. If your benchmark ran the baseline in the first 500 iterations and the "optimized" version after 9000 iterations of warmup, the baseline would look 26% faster than it is — enough to erase a genuine 20% win or invent a fake one. Now re-run with `nvidia-smi -lgc 1410,1410`: the SM clock holds at 1410 the whole way, the temperature stabilizes around 65 °C because the pinned clock draws less peak power, and p50 stays flat at 11.2 ms from iteration 0 to iteration 10,000. That flat line is a *reproducible* measurement. The drifting one is a trap.

#### Worked example: the noisy cloud neighbor

You benchmark the same code twice on a cloud instance and get p50 = 11.2 ms both times, but p99 = 12.0 ms on Monday and p99 = 24 ms on Tuesday. Nothing you control changed. The culprit is almost always a **noisy neighbor**: on shared infrastructure, another tenant's job can contend for the same physical resources — PCIe bandwidth, host memory bandwidth, CPU cores, even the same GPU under time-slicing or MIG misconfiguration. Your p50 is stable because most of the time you have the machine to yourself; your p99 explodes because 1% of the time the neighbor is hammering the shared bus. How do you *detect* it, since you cannot see the neighbor? Two signals. First, run the same benchmark several times and watch the **run-to-run variance of the tail**: a well-controlled bare-metal box gives you the same p99 every run to within a percent or two; a noisy shared box gives you a p99 that jumps around by 2x. Second, correlate with `dmon`: if your SM utilization dips below 100% during the stalls even though your workload is unchanged and GPU-bound, something outside your process is stealing cycles. The defensive move is to benchmark on a dedicated, non-shared instance (or bare metal) for any number you are going to make a decision on, and to treat cloud p99s from a shared box as an upper bound, not a measurement.

## The dataloader confound: isolate the GPU step

There is one more way to fool yourself that deserves its own section because it wastes so many engineer-days: benchmarking end-to-end when the bottleneck is the **input pipeline**, then "optimizing the model" and seeing nothing move. Picture the timeline of one training or inference step: the CPU reads a sample from disk, decodes the JPEG or tokenizes the text, augments or pads it, copies it to the GPU, and only *then* does the GPU compute. If the CPU-side preprocessing takes 20 ms and the GPU compute takes 11 ms, the GPU sits idle waiting for input almost half the time, and your end-to-end latency is 20 ms — set by the loader, not the model. Now you make the model 2x faster. End-to-end latency: still 20 ms, because the GPU was never the bottleneck. You "optimized" the model and the benchmark did not move, and if you were not careful you would conclude your optimization did not work — when in fact it worked perfectly and you were measuring the wrong thing.

The discipline is to **benchmark the GPU step in isolation first**, on cached or synthetic tensors that are already on the device, so the input pipeline is entirely out of the picture:

```python
import torch

# Isolate the GPU step: a fixed synthetic input that lives on the device.
# No disk, no decode, no host->device copy inside the timed region.
x = torch.randn(32, 3, 224, 224, device="cuda")   # allocate ONCE, up front

# warmup
for _ in range(50):
    with torch.no_grad():
        model(x)
torch.cuda.synchronize()

# timed, events, steady state -- this measures the MODEL, nothing else
lat = []
start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
for _ in range(1000):
    start.record()
    with torch.no_grad():
        model(x)
    end.record()
    torch.cuda.synchronize()
    lat.append(start.elapsed_time(end))
```

Because `x` is allocated once and already on the GPU, there is no disk read, no decode, and no host-to-device copy inside the timed loop — the number you get is the model's device time and nothing else. If that number is 11 ms but your production end-to-end latency is 20 ms, you have just proven the other 9 ms lives in the input pipeline, and *that* is where to point your optimization effort. The full treatment of loader starvation — `num_workers`, `prefetch_factor`, pinned memory, overlapping the copy with compute — is its own post, [the dataloader and preprocessing wall](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu); the benchmarking rule that unlocks it is simply *isolate the GPU step before you trust any end-to-end number*. Measure the parts, then measure the whole, and never confuse the two.

## A/B honesty: is the difference real, or is it noise?

Everything so far has been about measuring *one* configuration honestly. But the whole point of a benchmark in this series is to compare two: baseline A versus optimized B. That comparison has its own set of ways to lie, and they are subtle because both numbers can be individually correct and the *comparison* still be wrong.

![a decision tree that gates a claimed speedup through correct timing then a warm and clock locked machine then a gap larger than twice the standard deviation before calling it a real win](/imgs/blogs/setting-up-a-reproducible-benchmark-6.webp)

The tree above is the gate every claimed speedup has to pass, and it has three levels because there are three independent ways an A/B can be fake. First gate: **is the timing correct?** If either number came from an unsynchronized wall clock, throw both out and remeasure with events — you are comparing launch times, not compute. Second gate: **was the machine warm and locked for both?** If A ran cold and B ran warm, or A ran throttled and B ran cool, the difference is an artifact of the machine, not the code. Third gate, the one people skip: **is the gap bigger than the noise?** If A is 11.2 ms and B is 11.0 ms but the run-to-run standard deviation is 0.5 ms, you have measured nothing — the "improvement" is inside the noise band and will flip sign on the next run.

Three practices make an A/B trustworthy. **Change exactly one thing.** Same seed, same shapes, same batch, same clocks, same dtype — only the line under test differs. If you change the attention kernel *and* the batch size, you cannot attribute the result. **Interleave A and B.** Do not run all of A and then all of B, because slow drift in the machine (warming up, a background job starting, the neighbor waking) then hits one condition and not the other. Run A, B, A, B, A, B, … so any drift affects both equally and cancels:

```python
import numpy as np, torch

def timed_run(fn, x, iters=500):
    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    out = []
    for _ in range(iters):
        start.record(); fn(x); end.record()
        torch.cuda.synchronize()
        out.append(start.elapsed_time(end))
    return np.array(out)

# warmup both paths first (compile, autotune, allocator)
for _ in range(50):
    a_fn(x); b_fn(x)
torch.cuda.synchronize()

# interleave A and B in short blocks so machine drift cancels
A, B = [], []
for _ in range(20):                     # 20 blocks
    A.append(timed_run(a_fn, x, iters=100))
    B.append(timed_run(b_fn, x, iters=100))
A = np.concatenate(A); B = np.concatenate(B)

dp50 = np.percentile(A,50) - np.percentile(B,50)
pooled_std = np.sqrt(A.std()**2 + B.std()**2)
print(f"A p50={np.percentile(A,50):.2f}  B p50={np.percentile(B,50):.2f}")
print(f"delta p50 = {dp50:.2f} ms   pooled stddev = {pooled_std:.2f} ms")
print("REAL" if abs(dp50) > 2*pooled_std else "WITHIN NOISE")
```

```console
A p50=11.21  B p50=9.86  delta p50 = 1.35 ms   pooled stddev = 0.41 ms
REAL
```

That last line is the discipline made mechanical: a delta of 1.35 ms against a pooled standard deviation of 0.41 ms clears the "bigger than twice the noise" bar ($1.35 \gt 2 \times 0.41$), so this is a real 12% win, not a lucky draw. Had the delta been 0.3 ms against the same noise, the honest output would be "within noise" and the correct conclusion would be *we cannot yet tell*, which is a perfectly good scientific answer and a far better one than shipping a fake speedup. **Report the distribution and the noise band, not a point estimate.** Confidence, not vibes.

## A reusable benchmark harness

Everything in this post composes into one small piece of code you will use in every later post: a harness that warms up, times with CUDA events, computes percentiles, runs a fixed workload, and can optionally drop into `torch.profiler` when you need to explain a number rather than just report it. Here it is, complete and copy-adaptable:

```python
import numpy as np
import torch
from contextlib import nullcontext

class GPUBench:
    """Reproducible GPU timing: warmup + CUDA events + percentiles."""

    def __init__(self, warmup=50, iters=1000, seed=0):
        self.warmup = warmup
        self.iters = iters
        torch.manual_seed(seed)
        # freeze the machine as much as the process can:
        torch.backends.cudnn.benchmark = True      # autotune once, then cache
        torch.backends.cuda.matmul.allow_tf32 = True

    @torch.no_grad()
    def run(self, fn, *args, label=""):
        # --- warmup: pay every one-time cost, measure nothing ---
        for _ in range(self.warmup):
            fn(*args)
        torch.cuda.synchronize()

        # --- steady-state timing with CUDA events ---
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        lat = np.empty(self.iters, dtype=np.float64)
        for i in range(self.iters):
            start.record()
            fn(*args)
            end.record()
            torch.cuda.synchronize()
            lat[i] = start.elapsed_time(end)      # ms, device clock

        return self._report(lat, label)

    def _report(self, lat, label):
        stats = {
            "label":  label,
            "n":      len(lat),
            "p50":    float(np.percentile(lat, 50)),
            "p90":    float(np.percentile(lat, 90)),
            "p99":    float(np.percentile(lat, 99)),
            "mean":   float(lat.mean()),
            "std":    float(lat.std()),
            "min":    float(lat.min()),
        }
        print(f"[{label}]  p50={stats['p50']:.2f}  p90={stats['p90']:.2f}  "
              f"p99={stats['p99']:.2f}  std={stats['std']:.2f} ms  (n={stats['n']})")
        return stats

    @torch.no_grad()
    def profile(self, fn, *args, trace="trace.json"):
        """Drop into torch.profiler to explain WHERE the time goes."""
        from torch.profiler import profile, schedule, ProfilerActivity
        sched = schedule(wait=1, warmup=5, active=10, repeat=1)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=sched,
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            for _ in range(1 + 5 + 10):
                fn(*args)
                prof.step()
        print(prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=10))
        prof.export_chrome_trace(trace)
```

Using it is three lines, and comparing two implementations is one more:

```python
bench = GPUBench(warmup=50, iters=1000)
x = torch.randn(32, 3, 224, 224, device="cuda")

a = bench.run(lambda t: model_eager(t), x, label="eager")
b = bench.run(lambda t: model_compiled(t), x, label="compiled")

print(f"speedup p50: {a['p50'] / b['p50']:.2f}x")
bench.profile(lambda t: model_compiled(t), x)   # only when you need the WHY
```

```console
[eager]     p50=11.21  p90=11.44  p99=17.90  std=1.44 ms  (n=1000)
[compiled]  p50=8.03   p90=8.19   p99=9.41   std=0.42 ms  (n=1000)
speedup p50: 1.40x
-------------------------------  ------------  ------------  ----------
Name                               CUDA total    CUDA %       # of Calls
-------------------------------  ------------  ------------  ----------
triton_fused_conv_bn_relu_0           41.２ms       38.4%           640
ampere_bf16_gemm                      28.7ms        26.8%           320
triton_fused_add_layernorm            12.1ms        11.3%           480
...
```

Note two things the harness does that matter. The `profile` method uses the `schedule(wait, warmup, active)` API so the profiler *itself* skips warmup iterations before it starts recording — you never want your profiler capturing the cold-start iteration either. And the `run` method sorts nothing and hides nothing: it hands back p50, p90, p99, and the standard deviation, so the caller can apply the "bigger than twice the noise" test rather than eyeballing a mean. This little class is the measurement backbone the rest of the series leans on; when a later post says a fix took p99 from 17.9 ms to 9.4 ms, this is the harness that produced both numbers, under warmup, with events, on a clock-locked card.

## Case studies and real numbers

**The 3x that was a measurement bug (the hook, resolved).** Return to the engineer who reported a 3x attention speedup. Reconstruct their measurement honestly and it falls apart in three places. Their timer was `time.time()` around `model(x)` with no synchronize — so on the *baseline* they actually captured the first cold iteration (autotune + context, ~300 ms of one-time cost folded into the number) while on the *new* code they measured a warm iteration. That alone can look like a 3x. Layer on that the baseline ran while the box was still hot from a previous job (clock throttled to 1200 MHz) and the new code ran on a cooled card (1410 MHz), adding another ~15%. Re-measure the *same two kernels* with the `GPUBench` harness — warmup 50, 1000 iters, CUDA events, clocks locked at 1410 — and the honest result is: baseline p50 = 11.8 ms, new p50 = 11.2 ms. A 5% improvement, well inside what a slightly better attention kernel plausibly delivers, and *below* the threshold where you would rush to ship it as a headline. The 3x was never real. Nothing was wrong with the attention code; everything was wrong with the measurement.

**MLPerf-style methodology (cited as approximate).** The industry's answer to "how do we benchmark honestly enough to publish and compare across vendors" is MLPerf, and its rules encode exactly the disciplines above. Roughly: a load generator issues a fixed, reproducible query stream; there is a mandated warmup period before timing begins; results are reported as latency percentiles and throughput under a latency bound, not as a single mean; and submissions run for a minimum number of queries so the tail is statistically meaningful. Different scenarios (single-stream, server, offline) measure different things on purpose, because "latency" and "throughput" are not the same number. The precise thresholds evolve version to version, so treat these as the shape of the methodology rather than exact figures — but the shape is the point: fixed workload, warmup, percentiles, enough samples, controlled environment. Every serious benchmark converges on the same protocol because the failure modes are universal.

**A `torch.compile` "2x" that was really 1.1x.** A common and instructive pattern: someone benchmarks `torch.compile(model)` against eager and reports a 2x speedup, then it evaporates in production. Two measurement bugs usually explain it. First, they timed the *first* compiled call, which includes seconds of Dynamo tracing and Inductor codegen — but they timed it against a *warm* eager baseline, so the comparison was compile-time-included versus compile-time-excluded (nonsensical). Second, the reverse: after fixing warmup, the compiled version genuinely was faster, but only 1.1x on their actual, shape-varying workload — because production requests changed shape every call, triggering *recompilation* each time, which the fixed-shape benchmark never exercised. The honest number came from warming up the compiled model on the real distribution of shapes and *then* timing steady state, at which point the 2x became a 1.1x and the team correctly decided the compile complexity was not worth it for that service. The benchmark did not just measure the speedup; it changed the decision. (The recompilation-storm failure mode is its own post later in the series; the point here is that a dishonest benchmark would have shipped a 2x that production would have quietly refunded.)

## When to reach for this (and when not to)

This is the one post in the series with no "when not to." Reproducible measurement is not an optimization you selectively apply; it is the substrate every optimization stands on, and skipping it does not save you time — it costs you the far larger time of chasing a fake win or dismissing a real one. That said, there is a sensible *amount* of rigor for the situation:

- **A quick sanity check** ("did this obviously help or obviously hurt?") justifies wall-clock-plus-sync, warmup of 10, and 100 iterations. You are looking for a factor of two, and this is enough to see it.
- **A decision you will ship** ("do we replace kernel A with kernel B in production?") demands the full protocol: CUDA events, 50+ warmup, 1000+ iterations, locked clocks, isolated box, interleaved A/B, and the "bigger than twice the noise" test. The cost of the rigor is minutes; the cost of shipping a fake speedup is a sprint.
- **A number you will publish or put in an SLO** demands everything above plus multiple runs on multiple boxes to bound run-to-run variance, and honesty about the environment (bare metal vs shared cloud, which GPU SKU, which clocks).

The only genuine trap is *over-controlling in a way that hides the thing you care about*. If your production service runs with auto-boost on and shared tenancy, a perfectly clock-locked, isolated benchmark tells you the *ceiling* but not the *lived* latency — you may also want a "realistic" measurement that leaves the machine as production runs it, precisely to capture the throttle and the neighbor. Measure the controlled number to compare code fairly, and the realistic number to set expectations. They answer different questions; report both.

## Key takeaways

- **The call returning is not the work finishing.** GPU launches are asynchronous; an unsynchronized wall clock measures dispatch time, not compute time, and the two are unrelated. Always `synchronize()` around a wall-clock region, or use CUDA events.
- **CUDA events are the default timer.** `Event(enable_timing=True)`, `record()` / `record()` / `synchronize()` / `elapsed_time()` gives you pure device region time with host and launch overhead squeezed out. Reach for it first.
- **Throw away the warmup.** The first iterations pay context init, cuDNN/cuBLAS autotuning, allocator growth, and compile — one-time costs a steady-state service never repeats. Discard until the latency curve goes flat, then measure.
- **Report a distribution, not a number.** p50 is the typical case, p99 is the tail your users feel, and $p99 = p50 + \text{stall}$ tells you whether to optimize the median or hunt a stall. Never headline the mean or the min. For a stable p99, time at least ~1000 iterations.
- **Lock the machine.** Pin GPU clocks with `nvidia-smi -lgc`, disable auto-boost, watch `dmon` for thermal throttling, pin CPU threads and NUMA, isolate the box. A throttling GPU can fake or erase a 20% win.
- **Isolate the GPU step.** Benchmark on cached/synthetic device tensors before you trust any end-to-end number, or you will "optimize" a model whose bottleneck was the dataloader and see nothing move.
- **Change one thing, interleave A/B, beat the noise.** Same seed, same shapes, same clocks; run A/B/A/B so drift cancels; and only call it a win if the delta exceeds about twice the run-to-run standard deviation. Confidence, not vibes.
- **A dishonest benchmark changes the decision.** The 3x that was 1.05x, the compile 2x that was 1.1x — bad measurement does not just report the wrong number, it ships the wrong code. Measurement is the cheapest insurance in performance engineering.

## Further reading

- [Why your AI service wastes CPU and GPU](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu) — the series intro and the four-wastes / profile-hypothesize-fix-measure loop this benchmark serves.
- [The performance engineering playbook](/blog/machine-learning/performance-engineering/the-performance-engineering-playbook) — the capstone decision tree that ties every fix in the series back to a measured symptom.
- [The mental model of a GPU service](/blog/machine-learning/performance-engineering/the-mental-model-of-a-gpu-service) — host enqueues, device drains: the async model that makes the timing rules here inevitable.
- [Metrics that actually matter](/blog/machine-learning/performance-engineering/metrics-that-actually-matter) — util vs occupancy, p50 vs p99, allocated vs reserved, and which numbers lie.
- [Profiling PyTorch with torch.profiler](/blog/machine-learning/performance-engineering/profiling-pytorch-with-torch-profiler) — once your timer is honest, the profiler explains *where* the time goes.
- [LLM GPU benchmark](/blog/machine-learning/mlops/llm-gpu-benchmark) — the same measurement discipline applied to serving throughput and tokens-per-second.
- [PyTorch Profiler tutorial](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) and the `torch.cuda.Event` / `torch.cuda.synchronize` API docs — the primary sources for the timing APIs used here.
- The MLPerf Inference rules (mlcommons.org) — the industry reference for warmup, load generation, and percentile reporting; treat specific thresholds as version-dependent.
