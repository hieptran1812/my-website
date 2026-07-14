---
title: "Why Your AI Service Wastes CPU and GPU: The Four Kinds of Resource Waste and How to Find Them"
date: "2026-07-14"
publishDate: "2026-07-14"
description: "The four ways an AI service burns money at full utilization, the profiler signature that reveals each one, and the disciplined loop that turns a guess into a measured win."
tags:
  [
    "performance-engineering",
    "gpu-optimization",
    "profiling",
    "pytorch",
    "cuda",
    "cuda-graphs",
    "torch-compile",
    "latency",
    "throughput",
    "inference",
    "ml-systems",
  ]
category: "machine-learning"
subcategory: "Performance Engineering"
author: "Hiep Tran"
featured: true
readTime: 35
---

You get paged. A GPU inference service is quietly eating money: the box is an eight-way A100 node that costs roughly \$40 an hour, and finance wants to know why you need four of them to serve a load that "should fit on one." You SSH in, run `nvidia-smi`, and there it is — **GPU-Util: 99%**. The GPU is pinned. The obvious conclusion is that you are out of headroom and need more hardware.

That conclusion is wrong, and the number you just read is the reason. `nvidia-smi`'s utilization field does not mean "the GPU is doing 99% of the work it could do." It means "in the last sampling window, at least one kernel was running." A GPU that spends 70% of every millisecond idle — starved, waiting for a slow Python loop to hand it the next tiny kernel — will still report 99% utilization, because in each coarse sample window *some* kernel happened to be in flight. The utilization number is not lying about whether kernels ran. It is lying by omission about how *full* the machine was while they ran. On the service that paged you, the real fraction of time the streaming multiprocessors spend doing useful math is closer to 30%. You are paying for a Ferrari and driving it in a school zone.

This post is the map for a 40-part series on finding and fixing exactly this kind of waste. The thesis of the whole series fits in one figure: almost every wasted GPU-dollar traces to one of **four wastes**, each of which leaves a distinct fingerprint in a profiler, and each of which has one canonical fix you can name before you even open the trace. Learn to see the four, and "the GPU is slow, buy more" turns into "the GPU is host-bound, and CUDA graphs will get us 3x on the hardware we already own."

![a table mapping four kinds of resource waste to the profiler signature that reveals each one and the canonical fix](/imgs/blogs/why-your-ai-service-wastes-cpu-and-gpu-1.webp)

By the end of this post you will be able to: name the four wastes and their profiler signatures; explain precisely why `GPU-Util 100%` lies and which metric to trust instead; run the profile-hypothesize-fix-measure loop that structures every later post; read a launch-overhead calculation and an Amdahl bound well enough to predict a fix's ceiling *before* you write the code; and take one host-bound service from 30% to 85% real work with two lines of PyTorch. Everything after this is depth on one of these four wastes and the one tool that reveals it.

## Meet the four wastes

Resource waste on a GPU is invisible by construction. You cannot see a streaming multiprocessor sitting half-empty, a launch queue draining faster than Python can refill it, a kernel stalling on High-Bandwidth Memory (HBM) instead of computing, or the same tensor being read from HBM three times when once would do. The entire craft of performance engineering is making those invisible losses *visible* — first in a figure, then in a profiler, then in a before-and-after number. So before any tool, here is the field guide. Four wastes. Each gets its intuition, its profiler signature, and its canonical fix, with a pointer to the part of the series that goes deep.

**Waste 1 — Idle GPU (host-bound).** The GPU is a firehose; the CPU is the person turning the tap on and off for every single kernel. A modern A100 can retire a small kernel in a few microseconds, but PyTorch's Python-side dispatch — building the op, checking dtypes, allocating output, and calling `cudaLaunchKernel` — also costs a few microseconds *per op*. When your model launches a couple thousand tiny ops per forward pass, the CPU cannot enqueue them fast enough, and the GPU spends most of each step **idle between kernels, waiting to be fed**. The profiler signature is unmistakable: a GPU timeline full of short kernels separated by wide gaps, and a CPU-side flame graph dominated by `cudaLaunchKernel` and dispatch overhead. The canonical fix is to stop paying per-kernel launch cost — **CUDA graphs** record the whole kernel sequence once and replay it as a single launch, and `torch.compile` fuses many small ops into few large ones. We spend all of Track C and part of Track D here; the launch mechanics get their own post, [the kernel launch overhead problem](/blog/machine-learning/performance-engineering/the-kernel-launch-overhead-problem).

**Waste 2 — Low occupancy (bad kernels).** Now the GPU *is* busy, but each kernel it runs is leaving the machine half-empty. A GPU hides memory latency by keeping many groups of 32 threads — **warps** — resident on each streaming multiprocessor (SM) and switching between them whenever one stalls. **Occupancy** is the ratio of resident warps to the hardware maximum. A kernel that uses too many registers, too much shared memory, or launches too few blocks leaves SMs under-filled; the scheduler has nothing to switch to when a warp stalls on memory, so the SM sits on its hands. The signature is a kernel that dominates your runtime but shows low achieved occupancy and high "warp-stall" reasons in Nsight Compute. The fix is kernel-level: choose better launch configurations, fuse to raise arithmetic intensity, or occasionally write a custom kernel. Track B teaches the tool (`ncu`), and Track H writes the kernel.

**Waste 3 — The bandwidth wall (memory-bound).** The kernel is not waiting on the CPU and it is not under-occupied — it is waiting on *memory*. Every operation must read its inputs from HBM and write its outputs back. If a kernel does very little arithmetic per byte it moves, it finishes its math long before the bytes arrive, and the SMs stall on the memory system no matter how busy they look. The governing quantity is **arithmetic intensity**, $\text{AI} = \text{FLOPs} / \text{bytes}$, and the roofline model says achievable performance is $\min(\text{peak FLOP/s},\ \text{AI} \times \text{bandwidth})$. An elementwise `add` or a `softmax` or a `LayerNorm` has intensity so low it lives permanently under the memory roof. The signature is a kernel at 85-95% of peak HBM bandwidth but a tiny fraction of peak FLOP/s. The fix is to move fewer bytes: **fuse** elementwise chains so intermediate results never touch HBM, and use fused attention (FlashAttention) so the giant attention matrix is never materialized. Track E lives here; and the [roofline model](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) post in the HPC series derives the wall in full.

**Waste 4 — Redundant work (no fusion or caching).** The fastest operation is the one you never run. This waste is doing work you already did: recomputing the same projection every request, re-reading a weight you could have kept in cache, materializing an intermediate to HBM and reading it right back, re-tokenizing, re-padding, re-copying tensors across the request boundary, or recomputing keys and values in an autoregressive decoder instead of caching them. The signature is subtler — it shows up as *extra* kernels and *extra* HBM traffic that a smarter formulation would delete entirely. The fixes are structural: KV-caching, activation reuse, `channels_last` layouts that let cuDNN pick faster kernels, and Inductor fusion that deletes intermediate round-trips. It threads through Tracks D, E, and G.

Those four are the spine of the series. Here they are as a lookup you can keep on your desk:

| Waste | What is actually happening | Profiler signature | Canonical fix | Series track |
|---|---|---|---|---|
| Idle GPU (host-bound) | CPU cannot launch kernels fast enough; GPU starves between them | Wide gaps between short kernels; CPU pegged on `cudaLaunchKernel` | CUDA graphs, `torch.compile` | C, D |
| Low occupancy (bad kernel) | SMs under-filled; scheduler has no warp to hide latency | Low achieved occupancy, high warp-stall in `ncu` | Fuse, tune launch config, custom kernel | B, H |
| Bandwidth wall (memory-bound) | Kernel finishes math, waits on HBM bytes | ~90% of peak HBM bandwidth, low FLOP rate | Fuse elementwise, FlashAttention | E |
| Redundant work (no reuse) | Work is repeated or bytes moved needlessly | Extra kernels, extra HBM round-trips vs a leaner formulation | KV cache, `channels_last`, Inductor fusion | D, E, G |

Notice the discipline the table enforces: you do not get to *guess* which waste you have. You read the signature off a profile, and the signature tells you the fix. That is the entire game. The rest of this post is about the game's rules.

## The number that lies: what "GPU-Util 100%" actually measures

Let us make the lie concrete, because it is the single most expensive misunderstanding in production ML. Here is the command everyone runs, and its finer-grained cousin:

```bash
# The number everyone stares at -- and the one that lies:
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv -l 1

# Finer: per-second device monitor (sm%, mem%, power draw)
nvidia-smi dmon -s um -d 1
```

On the host-bound service, the first command prints something like this, second after second:

```console
utilization.gpu [%], utilization.memory [%], memory.used [MiB]
99 %, 41 %, 14210 MiB
99 %, 43 %, 14210 MiB
98 %, 40 %, 14212 MiB
```

Ninety-nine percent. Case closed, buy more GPUs — except it is not, and here is the mechanism. The NVIDIA Management Library defines "GPU utilization" as *the percent of the last sample period during which one or more kernels was executing*. The sample period is coarse (on the order of a sixth of a second to a second depending on the query path). "One or more kernels was executing" is a binary condition sampled over a long window. If your service launches a kernel, the GPU runs it for 3 microseconds, then sits idle for 7 microseconds while Python builds the next op, then launches again — over and over — then in any given sample window there was *always* a kernel executing at the moment of sampling. Utilization reports 99% while the SMs are genuinely idle 70% of the time.

The metric is not measuring how many of the GPU's arithmetic units are busy. It is measuring whether the GPU is "occupied at all," at a time resolution far too coarse to see the microsecond gaps that are killing you. `nvidia-smi` is a smoke detector: it tells you *something* is running, not *how efficiently*. It is genuinely useful for the questions it can answer — is the process alive, is memory near the ceiling, is the card thermally throttling — and genuinely dangerous for the question everyone asks it: "am I getting my money's worth?" For that you need finer instruments, which is what the rest of the series is about. The dedicated post [metrics that actually matter](/blog/machine-learning/performance-engineering/metrics-that-actually-matter) takes this apart in full; here is the short version of what to trust instead.

- **Utilization (`nvidia-smi`)**: "a kernel ran in the window." Trust it only to answer "is the GPU doing *anything*." It cannot tell you how full the SMs were.
- **Occupancy (`ncu`, the trace)**: "warps were resident on the SMs." Closer to the truth, but a fully occupied SM whose warps are all stalled on HBM is still wasting cycles.
- **Model FLOP Utilization (MFU)**: the fraction of the GPU's peak FLOP/s your model actually achieves, computed as (useful FLOPs per step) / (peak FLOP/s × step time). This is the honest number, because it is grounded in the arithmetic your model *must* do versus the arithmetic the hardware *can* do. Its only cost is that you must know your model's FLOP count.

The trap is that the three metrics agree when everything is fine and diverge violently when something is wrong — which is exactly when you are looking at them. A service at 99% util, 60% occupancy, and 30% MFU is not a healthy service; it is a host-bound service wearing a healthy costume.

## The mechanism: why a microsecond of launch overhead eats your step

Waste 1 is worth deriving, because the derivation is what lets you *predict* a fix's payoff instead of hoping. Model the host and the device as a producer and a consumer connected by a queue. The CPU (producer) enqueues kernels; each enqueue costs $t_\text{launch}$ of CPU time — building the op, dtype and device checks, output allocation, and the driver call. The GPU (consumer) executes each kernel in $t_\text{kernel}$ of device time. A forward pass launches $N$ kernels.

If the CPU could enqueue instantly, the step would take $N \cdot t_\text{kernel}$ — the GPU would be the bottleneck, back-to-back busy, which is what you want. But the CPU cannot enqueue instantly. It needs $N \cdot t_\text{launch}$ to push all $N$ kernels onto the stream. The stream is asynchronous, so these two proceed in parallel, and the step time is bounded below by the slower of the two:

$$t_\text{step} \approx \max\left(N \cdot t_\text{launch},\ N \cdot t_\text{kernel}\right)$$

When $t_\text{launch} > t_\text{kernel}$ — when your kernels are so small that dispatching one takes longer than running it — you are **host-bound**, and the whole step runs at the speed of Python, not the speed of the GPU. The fraction of time the GPU is actually doing useful work is:

$$f_\text{active} = \frac{N \cdot t_\text{kernel}}{\max\left(N \cdot t_\text{launch},\ N \cdot t_\text{kernel}\right)} = \frac{t_\text{kernel}}{\max\left(t_\text{launch},\ t_\text{kernel}\right)}$$

That ratio is your real utilization, and it is the number `nvidia-smi` refuses to show you.

#### Worked example: the host-bound step, by the numbers

Take the service that paged us. Profiling shows a forward pass launches about **1,800 kernels**, with an effective per-op dispatch cost of about **5 µs** (PyTorch's eager dispatch plus the driver call; the CUDA launch itself is a well-documented few microseconds of CPU-side overhead, and framework dispatch stacks on top). The kernels themselves are tiny — the summed device time is about **2.9 ms**.

- Host time to enqueue: $1{,}800 \times 5\,\mu s = 9.0\text{ ms}$.
- Device time to execute: $2.9\text{ ms}$.
- Step time: $\max(9.0, 2.9) = 9.0\text{ ms}$, plus a small tail to drain the last kernels, call it **9.4 ms**.
- Real active fraction: $2.9 / 9.4 \approx 0.31$ — **31%**.

So the GPU does 2.9 ms of genuine work inside a 9.4 ms step and idles the other 6.5 ms waiting for Python. Throughput is $1000 / 9.4 \approx 106$ steps per second. And `nvidia-smi` reports 99%, because in every one-second window there were always kernels in flight. The 31% is the number that matters, and it is the number the smoke detector cannot see. This is the exact situation the [mental model of a GPU service](/blog/machine-learning/performance-engineering/the-mental-model-of-a-gpu-service) post opens with — worth reading next if the producer/consumer framing clicks for you.

The derivation also tells you the ceiling of the fix, which brings us to the law that governs *every* optimization in this series.

## The loop that runs the whole series

Every later post — all 39 of them — is the same five-step loop applied to a different waste with a different tool. Internalize the loop now and the series becomes a set of variations on one theme.

![a left to right timeline of the optimization loop from profiling to reading the trace to hypothesizing to one fix to re-measuring](/imgs/blogs/why-your-ai-service-wastes-cpu-and-gpu-2.webp)

1. **Profile.** Never optimize without a profile. The single most common mistake in performance work is fixing the thing you *assume* is slow; it is almost never the thing that is actually slow. Run a real profiler under a realistic load and let it tell you where the time goes.
2. **Read the trace.** Find the dominant cost — the wide idle gap, the kernel that owns 40% of the timeline, the memcpy that overlaps nothing. One trace, one prime suspect.
3. **Hypothesize the cause.** Map the signature to one of the four wastes. Wide gaps and a pegged CPU means host-bound. A single fat memory-bound kernel means the bandwidth wall. Name the waste, and you have named the fix.
4. **Apply one fix.** Exactly one. Change two things at once and you will never know which one worked, or whether they cancelled. This is the discipline that separates measurement from superstition.
5. **Re-measure.** Same load, same warm-up, same metric. Did the number move the way your hypothesis predicted? If yes, keep it and re-profile — the bottleneck has almost certainly *moved*. If no, your hypothesis was wrong; revert and go back to step two.

The loop is deliberately boring. Boring is the point: it replaces heroics and hunches with a repeatable procedure that a junior engineer can run and a staff engineer can trust. Grounding every claim in a fresh measurement is so central that the series devotes a whole post to doing it honestly — [setting up a reproducible benchmark](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark) — because a sloppy measurement is worse than none.

**Amdahl's law is why step 3 matters more than step 4.** Suppose you profile and find that a fraction $p$ of your step time is spent on the thing you are about to fix, and your fix speeds up *that portion* by a factor $s$. The overall speedup is bounded by:

$$S = \frac{1}{(1 - p) + \dfrac{p}{s}}$$

Two consequences fall out immediately. First, if $p$ is small, no value of $s$ saves you — a 100x speedup on 10% of the step buys at most $1/0.9 \approx 1.11$x overall. **Fix the biggest slice first.** Second, even an infinitely fast fix ($s \to \infty$) caps the speedup at $1/(1-p)$. That ceiling is not pessimism; it is a planning tool. It tells you, before you write a line of optimization code, the most you can possibly gain — and whether the effort is worth it.

#### Worked example: the ceiling on a host-side fix

Our host-bound step is 9.4 ms, of which the GPU-compute floor is 2.9 ms and the removable host overhead is the rest. If CUDA graphs made launch overhead literally zero, the step could not drop below the 2.9 ms of device work. So the maximum speedup from attacking the *host* alone is:

$$S_\text{max} = \frac{9.4}{2.9} \approx 3.24\times$$

That is the Amdahl ceiling for this fix. It tells us two useful things. One: CUDA graphs should get us roughly 3x and no more, so if we measure 3x, we are done squeezing the host and further host work is wasted effort. Two: to go *beyond* 3.24x we must attack the 2.9 ms of compute itself — fuse the memory-bound kernels so there is less device work to do. That is exactly what `torch.compile` adds on top of graphs, and it is why the real fix combines both. We predicted the shape of the answer from a napkin calculation. Now let us go get it.

## A worked before-and-after: 30% to 85% real work

Time to run the loop end to end on named hardware — a single **A100 80GB SXM** (about 312 dense bf16 TFLOP/s of compute and 2.0 TB/s of HBM2e bandwidth, per NVIDIA's datasheet) serving a vision-plus-small-transformer model at batch size 8. This is the compressed version of the full case study in [the service at 30 percent GPU util](/blog/machine-learning/performance-engineering/the-service-at-30-percent-gpu-util); here it is the worked proof of the loop.

**Step 1 and 2 — profile and read the trace.** We reach for `torch.profiler`, the workhorse of the whole series and the subject of [profiling PyTorch with torch.profiler](/blog/machine-learning/performance-engineering/profiling-pytorch-with-torch-profiler). The wait/warmup/active schedule is essential: you skip the first noisy steps, warm up the caches and cuDNN autotuner, then capture a few clean steps.

```python
import torch
from torch.profiler import profile, schedule, ProfilerActivity, tensorboard_trace_handler

model = build_model().cuda().eval()
inputs = torch.randn(8, 3, 224, 224, device="cuda")

# skip 1 step, warm up 2, record 3 -- never profile a cold process
prof_schedule = schedule(wait=1, warmup=2, active=3, repeat=1)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=prof_schedule,
    on_trace_ready=tensorboard_trace_handler("./tb_logs"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for _ in range(6):                       # wait(1) + warmup(2) + active(3)
        with torch.inference_mode():
            model(inputs)
        torch.cuda.synchronize()             # close the step before stepping
        prof.step()                          # advance the schedule

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
prof.export_chrome_trace("trace.json")       # open in chrome://tracing or perfetto
```

The `key_averages()` table is the first place to look. On the host-bound service it reads like this:

```console
-------------------------------  ------------  ------------  ------------  ------------
Name                               Self CPU %     Self CPU    CUDA total    # of Calls
-------------------------------  ------------  ------------  ------------  ------------
cudaLaunchKernel                      38.6%       3.482ms         0.000us          1806
aten::conv2d                           9.1%       0.821ms         1.104ms           212
aten::add_                             7.4%       0.667ms         0.240ms           388
aten::native_layer_norm                6.2%       0.559ms         0.301ms           146
aten::_scaled_dot_product_attn         5.8%       0.523ms         0.612ms            48
void cutlass::Kernel<...>              0.0%       0.000us         0.734ms           212
-------------------------------  ------------  ------------  ------------  ------------
Self CPU time total: 9.021ms   Self CUDA time total: 2.918ms
```

Read the two totals at the bottom: **CPU time 9.0 ms, CUDA time 2.9 ms**. The CPU is doing three times the wall-clock work of the GPU, and the single largest CPU line is `cudaLaunchKernel` at 1,806 calls. Open the Chrome trace and the picture is even blunter: a top row of short kernels separated by wide empty gaps, and a CPU row packed solid with launch calls racing to keep up and losing. This is the textbook host-bound signature — the exact fingerprint from figure 1's first row. Reading this trace is a skill of its own, covered in [reading a Chrome trace](/blog/machine-learning/performance-engineering/reading-a-chrome-trace).

**Step 3 — hypothesize.** Signature matches Waste 1. Cause: too many tiny kernels, CPU-launch-bound. Predicted ceiling from our Amdahl calculation: about 3.24x if we remove the host cost entirely.

**Step 4 — apply one fix.** The fix for host-bound work is to stop paying per-kernel launch cost. `torch.compile(mode="reduce-overhead")` gives you *both* remedies in one call: Inductor fuses the small ops into fewer, larger kernels (attacking the compute floor), and it wraps the result in CUDA graphs (attacking the launch cost). If you want the graph mechanism explicitly — and you should understand it, because it is covered in depth in [cuda-graphs in pytorch](/blog/machine-learning/performance-engineering/cuda-graphs-in-pytorch) — it looks like this:

```python
# The one-liner: compile + CUDA graphs together.
compiled = torch.compile(model, mode="reduce-overhead")

# Or capture the forward yourself, to see the mechanism:
static_in = torch.randn(8, 3, 224, 224, device="cuda")

# Warm up on a side stream so cuDNN/cuBLAS finish picking algorithms
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        static_out = model(static_in)
torch.cuda.current_stream().wait_stream(s)

# Record the whole kernel sequence once...
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_out = model(static_in)

# ...then replay it as a SINGLE launch per request.
def infer(x):
    static_in.copy_(x)     # new data into the static input buffer
    g.replay()             # one launch replays all ~1,800 kernels
    return static_out.clone()
```

The key move: the 1,806 separate `cudaLaunchKernel` calls collapse into **one** `g.replay()`. The CPU stops being the bottleneck because it is no longer launching kernels one at a time.

**Step 5 — re-measure.** But measure *honestly*, which on a GPU means fighting the asynchrony. A naive `time.time()` around `model(x)` measures how long it took to *enqueue* the kernels, not how long they took to *run* — you must synchronize. The correct harness uses CUDA events and a warm-up:

```python
import torch

def bench(fn, x, warmup=20, iters=100):
    for _ in range(warmup):                  # warm caches, autotuner, graphs
        fn(x)
    torch.cuda.synchronize()                 # make sure warm-up finished
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(x)
    end.record()
    torch.cuda.synchronize()                 # wait for the GPU to finish
    return start.elapsed_time(end) / iters   # milliseconds per step
```

Here is what the loop produced, measured on the A100:

| Metric | Before (eager) | After (compile + graphs) | Change |
|---|---|---|---|
| `nvidia-smi` GPU-Util | 99% | 99% | unchanged (it always lied) |
| Real active fraction | ~31% | ~85% | 2.7x more real work |
| SM occupancy | 45% | 62% | fuller SMs |
| CPU launches / step | 1,806 | 1 (graph replay) | ~1800x fewer |
| Host overhead / step | 9.0 ms | 0.3 ms | 30x less |
| Step time | 9.4 ms | 3.0 ms | 3.1x faster |
| Throughput | 106 steps/s | 333 steps/s | 3.1x |
| p50 / p99 latency | 9.4 / 22 ms | 3.0 / 5.4 ms | tail shrank with it |
| Peak memory | 14.2 GB | 15.6 GB | graphs cost ~1.4 GB |

The star of the table is the top row against the second row: **the utilization number did not move, and the throughput tripled.** That single fact is the reason this series exists. If you had trusted `nvidia-smi`, you would have concluded the GPU was maxed out and bought two more nodes. The profiler said host-bound, the fix removed the host cost, and one A100 now does the work of three. On the eight-way \$40-an-hour node, that is the difference between four boxes and one and a half.

![a before and after comparison of a host bound service at thirty percent real work versus a graphed and compiled service at eighty five percent](/imgs/blogs/why-your-ai-service-wastes-cpu-and-gpu-3.webp)

And note the Amdahl bookkeeping held: we predicted a 3.24x ceiling from removing host cost, and we measured 3.1x. We landed just under the ceiling because graphs cost a little memory and add a small replay overhead, and because fusion clawed back some of the compute floor. The napkin math predicted the answer's magnitude before we wrote the fix. That is what the mechanism buys you.

**Stress-test the fix — because a win at batch 8 can be a loss elsewhere.** A fix that only helps in the lab is a liability. Push on it:

- **Batch 1 vs batch 64.** The host-bound problem is *worst* at small batch, where each kernel is tiny and launch cost dominates. At batch 1 the before-and-after gap can be even larger than 3x. At batch 64 the kernels are big enough that the GPU is already the bottleneck — $t_\text{kernel} > t_\text{launch}$ — and graphs help far less. Host-bound is a small-batch disease; know your batch size before you reach for the cure.
- **L4 instead of A100.** The L4 (about 242 fp16 TFLOP/s, 300 GB/s) has less compute and far less bandwidth. Its kernels take *longer*, so the same model is *less* host-bound there — the GPU is slower, so it is less often waiting on the CPU. The fix still helps, but the ceiling is lower. The hardware changes the diagnosis.
- **Shapes that vary per request.** CUDA graphs require static shapes and static input pointers; a graph captured for batch 8 at 224x224 is invalid for batch 5 at 256x256. A service with variable shapes needs shape bucketing or `torch.compile(dynamic=True)`, and getting it wrong triggers a recompilation storm — the subject of a full war-story post. Never graph a variable-shape service without a bucketing plan.
- **Under 50 concurrent requests.** Graph replay holds static I/O buffers, so concurrency needs either per-stream graphs or a batching front-end. The fix that tripled single-stream throughput can serialize you under load if you skip this.

The lesson of the stress test is that there is no free lunch and no universal fix — only a fix that is right for *this waste, this batch size, this hardware, this shape distribution*. Which is why the series is 40 posts and not one.

## The toolbox, coarse to fine

You do not reach for Nsight Compute to find out your service is slow; you reach for it after three coarser tools have already narrowed the problem to a single kernel. Think of the toolbox as a ladder you descend, spending more setup and more detail at each rung, and only descending as far as the trace forces you.

![a vertical stack of profiling tools from coarse nvidia-smi at the top down to fine grained memory snapshot at the bottom](/imgs/blogs/why-your-ai-service-wastes-cpu-and-gpu-4.webp)

- **`nvidia-smi` / DCGM** — the whole-machine gauge. Is the process alive, is memory near the ceiling, is the card throttling. Answers "something is wrong" in one second, and lies about efficiency. Free to run, always on.
- **`torch.profiler`** — the op-and-kernel table plus a Chrome trace. Your default first real profile: which ops cost what, on CPU and CUDA, with shapes and memory. Covered in Track B; this is where you will spend most of your time.
- **The Chrome trace / perfetto timeline** — the same profiler data, seen as a timeline. This is where host-bound gaps become *visible* — the empty bands between kernels you cannot get from a table.
- **Nsight Systems (`nsys`)** — the system-wide timeline across CPU threads, CUDA API calls, kernels, and memory copies, tied together with your own NVTX ranges. When the wall spans the boundary between CPU and GPU and copies, `nsys` is the tool that shows all three at once.

```bash
# System-wide timeline: CPU threads, CUDA API, kernels, memcpys, NVTX ranges
nsys profile -t cuda,nvtx,osrt,cudnn,cublas -o service_trace python serve.py
```

- **Nsight Compute (`ncu`)** — the single-kernel microscope. Once a trace names the one kernel that owns your runtime, `ncu` tells you *why* it is slow: achieved occupancy, memory throughput, warp-stall reasons, the Speed-of-Light section. Expensive to run (it replays the kernel many times), so you use it last, on one kernel.

```bash
# One kernel to the metal: occupancy, memory throughput, warp-stall reasons
ncu --set full -k "regex:.*softmax.*" -o softmax_report python serve.py
```

- **The memory snapshot** — a different axis entirely: not time but *space*. `torch.cuda.memory._record_memory_history()` records every allocation and free, and the visualizer shows you exactly what is holding your memory and what is leaking it — the subject of [memory snapshot and leak hunting](/blog/machine-learning/performance-engineering/memory-snapshot-and-leak-hunting).

```python
torch.cuda.memory._record_memory_history(max_entries=100_000)
# ... run the service until the leak shows ...
torch.cuda.memory._dump_snapshot("mem_snapshot.pickle")
# then open the file at https://pytorch.org/memory_viz
```

The rule that keeps you sane: **use the coarsest tool that can answer your current question.** Do not open `ncu` when a five-second `nvidia-smi` tells you memory is at the ceiling. Do not read a Chrome trace when the `key_averages` table already shows `cudaLaunchKernel` at 40%. Descend the ladder only when the rung above has done its job. Here is the same toolbox as a decision aid — what each tool sees, what it costs you, and when to reach for it:

| Tool | What it sees | Overhead to run | Reach for it when |
|---|---|---|---|
| `nvidia-smi` / DCGM | Util%, memory, power, temp | Negligible, always on | First look; is anything obviously wrong |
| `torch.profiler` | Per-op CPU + CUDA time, shapes, memory | Low, a few steps | Default first profile of a slow service |
| Chrome trace / perfetto | The execution timeline; gaps and overlap | Low (same capture) | You suspect host-bound or bad overlap |
| Nsight Systems | System-wide CPU + GPU + copy timeline | Moderate | The wall crosses CPU, GPU, and copies |
| Nsight Compute | One kernel: occupancy, bandwidth, stalls | High (kernel replayed) | A single kernel owns your runtime |
| Memory snapshot | Every allocation and free over time | Moderate | OOM, fragmentation, or a slow leak |

## The mental model: host enqueues, GPU consumes

To read any of those traces you need the right mental model of what a GPU service actually *is*, because the trace is a picture of this machine and nothing else. Picture two processors joined by a conveyor belt. The CPU running your Python is the producer; it does not compute your model — it *describes* the computation, one kernel at a time, and places each description on the belt. The belt is a **CUDA stream**: an ordered, asynchronous queue of work. The GPU is the consumer; it pulls kernels off the belt and executes them in order, and it can run ahead of the CPU or fall behind it, because the belt decouples them.

![a dataflow diagram where the cpu enqueues kernels onto a gpu stream that branches into a compute path and a memory copy path before merging](/imgs/blogs/why-your-ai-service-wastes-cpu-and-gpu-5.webp)

This decoupling is the source of both the GPU's speed and all of its confusions. Three consequences you must keep in your head:

**The launch queue can starve or overflow.** If the CPU enqueues slower than the GPU drains, the belt runs empty and the GPU idles — host-bound, Waste 1. If the CPU enqueues far faster than the GPU drains, the belt backs up and your `time.time()` measures enqueue time, not compute time — the source of a thousand bogus benchmarks. Only a `torch.cuda.synchronize()` makes the CPU wait for the belt to clear, which is why every honest measurement includes one.

**The stream forks into compute and copy.** Not all work is math. Moving a tensor from host memory to device memory (H2D) or back (D2H) runs on a separate copy engine, and it can overlap with compute *if* you set it up right — pinned host memory plus `non_blocking=True`. Get it wrong and every request pays a serial copy tax of $\text{bytes} / \text{PCIe bandwidth}$ that the GPU sits idle through. The trace shows this as a memcpy bar that overlaps nothing, and the fix is a whole post ([killing host-device copies]). We will not name that slug because it is a Track E sibling; find it from the capstone.

**A "sync" is a wall.** Any operation that forces the CPU to wait for the GPU — printing a loss, calling `.item()`, an `if tensor > 0` on device data, a `.cpu()` — drains the belt and serializes the two processors. A single accidental `.item()` inside a hot loop can turn a beautifully overlapped pipeline into a stop-and-go crawl, and it is invisible until you see the sync stall in the trace.

Hold this model and the traces stop being noise. A gap on the GPU row is a starved belt. A memcpy bar with no compute above it is a serial copy. A tall CPU spike with the GPU idle after it is a sync. Every waste in figure 1 is a specific pathology of this producer-belt-consumer machine, and every fix is a way of keeping the belt full of large, useful work.

## From symptom to fix: a decision tree

Put the pieces together and the diagnostic process becomes a short decision tree. You start from a symptom you can measure — low throughput, or a service that needs more GPUs than it should — and you route it, one profiler reading at a time, to exactly one fix.

![a decision tree that routes a low throughput symptom through three candidate causes down to one fix each](/imgs/blogs/why-your-ai-service-wastes-cpu-and-gpu-6.webp)

Read it top to bottom. The symptom is "low requests per second, and yet the GPU looks hot." You do not trust "looks hot" — you profile, and the trace tells you which of three mutually exclusive worlds you are in:

- **GPU idle with gaps between kernels** — host-bound. The CPU cannot feed the GPU. Fix: CUDA graphs and `torch.compile`. This is the most common waste in inference services and the cheapest to fix.
- **GPU busy but SMs under-filled** — a bad kernel, low occupancy. The GPU runs your kernel but leaves most of itself idle. Fix: fuse, tune the launch configuration, and confirm with `ncu`.
- **GPU busy and SMs full but stalled on memory** — memory-bound, at the bandwidth wall. The SMs are occupied but waiting on HBM. Fix: fuse elementwise chains and use FlashAttention to stop moving bytes you do not need.

The tree is deliberately narrow because the profiler makes it narrow. You are never choosing among all four wastes at once; you are answering one binary question per level — "are there gaps?", then "are the SMs full?", then "is it compute or memory?" — and each answer eliminates whole branches. That is why the loop is fast in practice even though the space of possible fixes is enormous. The full, resolved version of this tree — every symptom, every tool, every fix — is [the performance engineering playbook](/blog/machine-learning/performance-engineering/the-performance-engineering-playbook), the capstone that ties the entire series together. This post plants the tree; the capstone grows every branch.

## Three numbers, one truth: util vs occupancy vs MFU

We have circled back to the lie, because now you have the vocabulary to see through it precisely. The reason a service can read 99% utilization, 62% occupancy, and 30% MFU all at once is that the three numbers measure three *different* things, at three different depths, and only the deepest one is grounded in the work your model actually has to do.

![a table contrasting what gpu utilization occupancy and model flops utilization each claim to measure versus what each one hides](/imgs/blogs/why-your-ai-service-wastes-cpu-and-gpu-7.webp)

- **Utilization** claims "the GPU was busy" but only proves "a kernel ran in the sample window." It hides how full the SMs were, so it is fooled completely by the host-bound case: tiny kernels with idle gaps read as 99%.
- **Occupancy** claims "warps were resident on the SMs" and it is a real, useful number from `ncu`. But it hides whether those resident warps were *doing* anything — an SM packed with warps all stalled on HBM has high occupancy and does no math. Occupancy is necessary for latency hiding, not sufficient for throughput.
- **MFU** claims the honest thing: the fraction of the hardware's peak FLOP/s that your model's *required* arithmetic actually achieved. It hides nothing about efficiency — its only cost is that you must count your model's FLOPs to compute it. A transformer's FLOP count is well known; once you have it, MFU is the number that cannot be gamed.

Here is the same contrast as a table you can act on:

| Metric | What it claims | What it hides | Trust it for |
|---|---|---|---|
| `nvidia-smi` util | The GPU was busy | How full the SMs were | Liveness, memory ceiling, throttling |
| SM occupancy | Warps were resident | Whether they were stalled | Latency-hiding headroom |
| MFU | Real useful FLOP fraction | Nothing (needs a FLOP count) | Is this fast, honestly |

The practical takeaway is a habit: **never report a win in utilization.** Report it in throughput, latency, or MFU — numbers grounded in work done, not in whether a kernel happened to be running when the sampler looked. When a stakeholder points at a 99%-util dashboard and says "the GPU is maxed out," you now have the one-sentence rebuttal: "utilization means a kernel ran, not that the machine was full; our MFU is 30%, which is why we can triple throughput on the hardware we own." That sentence is worth a rack of GPUs.

## Case studies: real numbers from the literature

The four-wastes framing is not a private theory; it is the shape of nearly every published GPU optimization result. A few, cited so you can check them, with numbers framed as reported rather than promised.

**CUDA graphs on small-batch inference (Waste 1).** NVIDIA and the PyTorch team documented CUDA graphs precisely to kill launch overhead on models with many small kernels. PyTorch's own "Accelerating PyTorch with CUDA Graphs" write-up reports meaningful end-to-end speedups on launch-bound workloads, with the effect largest exactly where our derivation predicts — small batch sizes, many tiny kernels, where $t_\text{launch}$ dominates $t_\text{kernel}$. The mechanism is the one we captured by hand above: collapse thousands of launches into one replay.

**torch.compile / Inductor fusion (Wastes 3 and 4).** When PyTorch 2.0 introduced `torch.compile`, the team reported geometric-mean speedups on the order of 1.3x to 2x across large public model suites (TorchBench, HuggingFace, TIMM) on A100-class hardware, driven substantially by Inductor fusing elementwise and reduction ops so intermediates never round-trip to HBM. That is Waste 3 (fewer bytes moved) and Waste 4 (fewer redundant kernels) attacked together, automatically. The wins are real and workload-dependent — which is why the series teaches you to *verify fusion actually happened* in the trace rather than trusting the decorator.

**FlashAttention and the memory wall (Waste 3).** The FlashAttention work (Dao et al.) is the canonical bandwidth-wall fix: standard attention materializes the full $N \times N$ score matrix to HBM, a hopelessly memory-bound operation; FlashAttention fuses the softmax into the matmuls and never writes that matrix out, reporting multi-fold attention speedups and large memory savings on long sequences. PyTorch ships it as `scaled_dot_product_attention`. The [kernel fusion and FlashAttention](/blog/machine-learning/high-performance-computing/kernel-fusion-and-flashattention-beating-the-memory-wall) post in the HPC series derives why it works.

**`channels_last` for convnets (Waste 4).** PyTorch's channels-last memory-format tutorial reports that switching convolutional models from NCHW to NHWC lets cuDNN select faster tensor-core kernels and move memory more efficiently, yielding solid speedups on Ampere GPUs with mixed precision — a layout change, not a math change, deleting redundant data movement. The exact number depends on the model, which is the recurring lesson: measure it on *your* service.

**A memory leak found by snapshot (Waste-adjacent).** PyTorch's memory-visualization tooling exists because slow leaks in long-running services are otherwise nearly impossible to localize. The published walk-throughs show the snapshot pinning a growing allocation to the exact line of code retaining it — a retained autograd graph, an unbounded cache, a list of tensors that never clears. Track G's [memory leak war story](/blog/machine-learning/performance-engineering/memory-snapshot-and-leak-hunting) is this pattern end to end.

The common thread: in every one, someone profiled, matched a signature to a waste, applied the matching fix, and *measured* the result on named hardware. None of them started from "the GPU is slow, buy more."

## When to reach for the profiler, and when to stop

A word against over-optimization, because performance work has a failure mode of its own: polishing a bottleneck that does not matter while the clock and the budget burn. The loop is a tool, not a compulsion. Here is when to run it and, just as importantly, when to put it down.

**Reach for the profiler when** the service is measurably missing an SLO (throughput, p99, or cost per million requests); when a training run is slower than a back-of-envelope MFU estimate says it should be; when you are about to buy more hardware to serve more load; or when a metric moved for a reason you cannot explain. In all of these, a profile pays for itself in minutes.

**Do not optimize when** you are already at high MFU and meeting SLO — a service at 85-90% MFU that hits its latency target is *done*, and the next hour is better spent on a feature or on reliability. Do not reach for CUDA graphs on a service that is genuinely compute-bound at high occupancy; graphs kill launch overhead, and if you have none, they add memory cost for nothing. Do not `torch.compile` a service whose shapes change on every request without a bucketing or dynamic-shape plan; you will trade launch overhead for a recompilation storm and come out behind. Do not chase p99 before you have confirmed it is a *stall* (a periodic sync, a GC pause, a recompile) and not simply the service being at capacity — the fix for a stall and the fix for saturation are opposite. And never optimize the 5% before the 95%; Amdahl already told you the ceiling, and it is low.

The honest north star is this: **an optimization you cannot measure did not happen, and an optimization you do not need is a bug in your judgment.** The whole series is built to make the first kind measurable and the second kind obvious.

## Key takeaways

- **`GPU-Util 100%` means "a kernel ran in the sample window," not "the machine is full."** Trust it for liveness and memory, never for efficiency. Report wins in throughput, latency, or MFU.
- **Almost every wasted GPU-dollar is one of four wastes**: idle GPU (host-bound), low occupancy (bad kernel), the bandwidth wall (memory-bound), or redundant work (no reuse). Each has a distinct profiler signature and one canonical fix.
- **The signature tells you the fix.** You do not guess the cause — you read it off a profile. Wide gaps and a pegged CPU means host-bound; fuse and graph. A memory-bound kernel at 90% HBM means the bandwidth wall; fuse and FlashAttention.
- **Run the loop, always the same:** profile, read the trace, hypothesize one cause, apply one fix, re-measure. Never optimize without a profile, and never change two things at once.
- **Amdahl's law is a planning tool, not a footnote.** $S = 1/((1-p) + p/s)$ tells you the ceiling of a fix before you write it. Fix the biggest slice first; a huge speedup on a small slice buys almost nothing.
- **Host-bound is a small-batch, many-kernel disease.** Launch overhead $N \cdot t_\text{launch}$ dominates when kernels are tiny. CUDA graphs collapse $N$ launches into one; `torch.compile` fuses the kernels so there is less to run.
- **Measure on the GPU honestly:** warm up, then `torch.cuda.synchronize()` before and after, and time with CUDA events — or you are measuring enqueue time, not compute time.
- **Every fix is a cost.** Graphs cost memory and demand static shapes; compile costs compile time and can recompile on new shapes. Stress-test at batch 1 vs 64, on your real hardware, with your real shape distribution, before you ship.
- **Stop when you are at high MFU and meeting SLO.** Over-optimization is its own waste.

## Further reading

- [The performance engineering playbook](/blog/machine-learning/performance-engineering/the-performance-engineering-playbook) — the capstone: the full symptom-to-fix decision tree and the checklist that ties this series together.
- [Metrics that actually matter](/blog/machine-learning/performance-engineering/metrics-that-actually-matter) — util vs occupancy vs SM efficiency vs MFU, and which to trust, in full.
- [The kernel launch overhead problem](/blog/machine-learning/performance-engineering/the-kernel-launch-overhead-problem) and [cuda-graphs in pytorch](/blog/machine-learning/performance-engineering/cuda-graphs-in-pytorch) — the mechanism and the fix for Waste 1.
- [Profiling GPU workloads: finding the real bottleneck](/blog/machine-learning/high-performance-computing/profiling-gpu-workloads-finding-the-real-bottleneck) and [the roofline model](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) — the HPC-series companions on GPU internals and the memory wall.
- [PyTorch Profiler recipe](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) and [Profiler with TensorBoard](https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html) — the primary docs for the workhorse tool.
- [Accelerating PyTorch with CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/) and the [torch.compiler docs](https://pytorch.org/docs/stable/torch.compiler.html) — the canonical sources for the two host-bound fixes.
- [Nsight Systems](https://docs.nvidia.com/nsight-systems/) and [Nsight Compute](https://docs.nvidia.com/nsight-compute/) documentation — the deeper GPU profilers.
- [Channels Last memory format tutorial](https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html) and [Understanding GPU Memory](https://pytorch.org/blog/understanding-gpu-memory-1/) — layout and the memory snapshot visualizer.
