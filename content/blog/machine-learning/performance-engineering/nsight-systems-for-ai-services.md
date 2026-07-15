---
title: "Nsight Systems for AI services: finding the wall across CPU, GPU, and copies"
date: "2026-07-14"
publishDate: "2026-07-14"
description: "torch.profiler shows PyTorch's view of a service; Nsight Systems shows the whole machine on one aligned timeline. Learn the method for finding the wall when it lives outside PyTorch — a copy, a sync, a CPU thread."
tags:
  [
    "performance-engineering",
    "gpu-optimization",
    "nsight",
    "profiling",
    "pytorch",
    "cuda",
    "nvtx",
    "latency",
    "ml-systems",
  ]
category: "machine-learning"
subcategory: "Performance Engineering"
author: "Hiep Tran"
featured: true
readTime: 40
---

The kernels were fine. That was the maddening part. We had an embedding service on a single A100 80GB running at p50 55 ms, p99 92 ms, throughput stuck at 180 req/s, and every profile we pulled with `torch.profiler` said the same reassuring thing: the GEMMs were fast, self-CUDA time was near 100%, nothing obviously fusable, no giant kernel to blame. By the PyTorch view of the world, this service was compute-bound and healthy. The dashboard agreed — `nvidia-smi` showed GPU utilization bouncing between 70% and 100%. The previous owner had written "GPU-bound, needs a bigger box" in the runbook, and on paper he was right.

He was wrong. The wall was a 24 MB device-to-host copy of the output tensor and a CPU thread parked in `cudaMemcpy` for three milliseconds per request, unable to launch the next batch while it waited. `torch.profiler` never showed it as a wall — it recorded the copy as a modest `aten::copy_` line and the sync as nothing at all, because a Python thread blocked in the CUDA driver is invisible to a profiler that only instruments PyTorch operators. The tool that saw it was Nsight Systems. One `nsys profile` run put the GPU kernels, the CUDA API calls, the memory-copy engine, and the OS thread states onto a single clock-aligned timeline, and the wall was suddenly obvious: a 40-microsecond kernel, then three milliseconds of dead GPU while a pageable copy dribbled bytes back to the host over PCIe, then the main thread waking up to do it all again.

![a stacked set of timeline rows for operating system runtime, cpu threads, cuda api, gpu kernels, memory copies, and multi gpu communication all aligned on one clock](/imgs/blogs/nsight-systems-for-ai-services-1.webp)

This post is about the second rung of the profiling ladder in the *[Profiling & Optimizing AI Services](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu)* series: **Nsight Systems (`nsys`), the system-level profiler you reach for when the bottleneck lives outside PyTorch.** The series intro names the four wastes — idle GPU, low occupancy, the bandwidth wall, redundant work — and the profile → hypothesize → fix → measure loop. `torch.profiler`, covered in the [sibling post on the PyTorch profiler](/blog/machine-learning/performance-engineering/profiling-pytorch-with-torch-profiler), is the right tool for three of those four when the waste is inside a PyTorch program. But the first waste — idle GPU — is very often caused by something PyTorch cannot see: a copy, a synchronization, a CPU thread starved on the GIL, a dataloader worker that fell behind, an NCCL collective waiting on a straggler. For those you need to see the whole machine. That is exactly what `nsys` draws. By the end of this post you will be able to run `nsys` against a live service, read every row of the timeline, apply a repeatable method for finding the single longest wall, and know precisely which finer tool to reach for once you have named it.

## The map and the microscopes: where nsys sits

Before any command, get the scopes straight, because reaching for the wrong profiler wastes an afternoon. There are three profilers you will use constantly, and they answer three different questions at three different scopes.

![a three by three comparison table of torch profiler, nsight systems, and nsight compute across their scope, what they see, and what they are blind to](/imgs/blogs/nsight-systems-for-ai-services-2.webp)

`torch.profiler` sees **one process, at the granularity of PyTorch operators and the CUDA kernels they launch.** It knows that `aten::mm` ran, how long the kernel took, what the input shapes were, how much memory it allocated. It is the fastest way to localize a hot region to a specific line of your model. What it cannot see is anything outside the Python process it is attached to: the OS scheduler parking your thread, a second process sharing the GPU, the memory-copy engine, the NCCL library's own threads, the driver's polling. It is a microscope pointed at your PyTorch code.

Nsight Compute (`ncu`) is the other microscope, pointed the other direction — **down into a single kernel.** It replays one kernel dozens of times with hardware counters enabled and tells you its occupancy, its memory throughput, why its warps stalled, where it sits against the Speed-of-Light roofline. Covered in the [Nsight Compute deep-dive sibling](/blog/machine-learning/performance-engineering/nsight-compute-kernel-deep-dive), `ncu` is the tool for "this one kernel is the wall and I need to make it faster." It is useless for finding *which* kernel, or for anything that is not a kernel at all.

Nsight Systems is neither microscope. It is **the map.** It traces the whole machine — every CPU thread, every CUDA API call, every kernel on every stream, every memory copy, every NCCL collective, and the OS runtime underneath all of it — and lays them on one timeline whose clocks are aligned to within microseconds. It does not tell you why one kernel is slow (that is `ncu`) or which PyTorch line launched it (that is `torch.profiler`). It tells you *where the time goes across the entire system*, so you can see that the GPU is idle, and then see that it is idle because a copy is running, and then see that the copy is running because a thread called `.cpu()`. The map first; the microscope second. That ordering is the whole discipline, and getting it backwards — pulling `ncu` on a kernel that was never the bottleneck — is the single most common way engineers burn a day.

The division of labor is worth stating as a table, because you will make this choice on every investigation:

| Tool | Scope | Overhead | Reach for it when |
|---|---|---|---|
| `torch.profiler` | one PyTorch process, ops + kernels | low (2–20%) | you suspect the waste is inside your model code |
| Nsight Systems (`nsys`) | whole machine, all threads + streams + copies | low (5–15%) | the GPU is idle and you don't yet know why |
| Nsight Compute (`ncu`) | one kernel, replayed with counters | very high (10–100×) | you already know which kernel is the wall |

Notice the overhead column. `nsys` is cheap enough to run against a real service under real load — it samples and traces, it does not replay — which is why it belongs early in an investigation. `ncu` is so expensive it can only profile one kernel at a time, deliberately serialized and replayed, which is why it belongs last. This series' [companion Nsight post for LLM serving](/blog/machine-learning/model-serving/profiling-llm-serving-with-nsight) walks the same ladder for a token-generation workload; here we stay on the general method of finding the wall, which is identical whether you serve embeddings, run diffusion, or train a transformer.

## The nsys command, decoded

Here is the command that produced the trace at the top of this post. Every flag earns its place, so it is worth taking apart.

```bash
nsys profile \
  -t cuda,nvtx,osrt,cudnn,cublas \
  -o embed_service \
  --force-overwrite true \
  -c cudaProfilerApi \
  --capture-range-end stop \
  python serve.py
```

The `-t` (trace) flag chooses which **domains** `nsys` instruments, and each domain is one class of the rows in figure 1. This is the flag you will tune most, so here is what each domain buys you:

| Domain (`-t`) | What it captures | The row it draws |
|---|---|---|
| `cuda` | CUDA runtime/driver API calls **and** the kernels + memcpys they launch | the CUDA API row and the per-stream kernel/memcpy rows |
| `nvtx` | your own `nvtx.range_push/pop` annotations | a custom lane labeling your handler phases |
| `osrt` | OS runtime: thread state, `poll`, `select`, `pthread_cond_wait`, mutexes | the CPU thread rows and their blocked/running states |
| `cudnn` | cuDNN library calls (convolutions, RNN primitives) | named cuDNN spans on the API row |
| `cublas` | cuBLAS library calls (GEMMs) | named cuBLAS spans on the API row |
| `nccl` | NCCL collectives (all-reduce, all-gather) | the NCCL row, essential for multi-GPU |

The `cuda` domain is non-negotiable — it draws both the API calls and the actual GPU work. The `osrt` domain is the one people forget, and it is precisely the domain that made our copy visible: without `osrt`, you see the GPU go idle but you cannot see the CPU thread sitting blocked in `cudaMemcpy`, so you cannot prove *who* caused the idle. Add `nvtx` and you get your own code's phase labels on the timeline (more on that later). Add `cublas`/`cudnn` and the anonymous `ampere_sgemm_...` kernels get their library context. On a multi-GPU service, add `nccl`.

The `-o embed_service` names the output; `nsys` writes `embed_service.nsys-rep`, the report file the GUI and the CLI both read. `--force-overwrite true` lets you re-run without renaming.

The `-c cudaProfilerApi` and `--capture-range-end stop` pair is the part that separates a useful trace from a useless one. **Never profile the cold start.** The first few iterations of any service are a lie: cuDNN is running its autotuner, the allocator is growing its pools, `torch.compile` is tracing, weights are still landing in HBM. If you profile from process start, ninety percent of your trace is warmup you will never see again in production, and the steady-state wall you actually care about is a thin sliver buried in the noise. `-c cudaProfilerApi` tells `nsys` to record *nothing* until your program calls `cudaProfilerStart()`, and to stop when it calls `cudaProfilerStop()`. From PyTorch you drive that with two lines around the region you care about:

```python
import torch

# ... warm up: run 20 steady-state requests so cuDNN autotuning,
# allocator growth, and any torch.compile tracing are already done ...

torch.cuda.synchronize()
torch.cuda.profiler.start()          # nsys begins recording here
for _ in range(30):                  # capture 30 steady requests
    handle_one_request(batch)
torch.cuda.synchronize()
torch.cuda.profiler.stop()           # nsys stops; everything else is dropped
```

Now the trace contains thirty clean steady-state requests and nothing else. If you cannot edit the code, the alternatives are a fixed duration (`-d 20` to profile twenty seconds) or a delay-then-duration (`--delay 60 -d 20` to skip a minute of warmup then grab twenty seconds). The `cudaProfilerApi` range is strictly better when you can reach the code, because it aligns the capture to *exactly* the requests you want rather than a wall-clock guess.

### The mechanism: how nsys puts everything on one clock

The reason `nsys` can say "the CPU thread was blocked *while* the copy was running *while* the GPU was idle" is that all three observations share a single timebase, and building that shared timebase is the whole engineering trick of a system profiler. `nsys` collects data two different ways and reconciles them onto one clock. For CPU-side work it **samples**: on a timer interrupt (typically around 1 kHz) it walks the call stack of each thread and records what it was doing and whether it was running or blocked. For CUDA work it **traces**: it hooks the CUDA runtime so that every `cudaLaunchKernel`, `cudaMemcpyAsync`, and `cudaStreamSynchronize` is timestamped on entry and exit, and it uses CUPTI (the CUDA Profiling Tools Interface) to get the *device-side* timestamps of when each kernel and copy actually began and ended on the GPU.

Those device timestamps come off the GPU's own clock, not the CPU's, so `nsys` measures the offset between the two clocks at profile start and corrects for drift, aligning them to within a microsecond or two. That correction is why a kernel's bar on the GPU row lines up correctly under the `cudaLaunchKernel` bar on the API row that launched it — you can see the launch latency, the gap between "the CPU asked for the kernel" and "the GPU started running it," measured directly off the aligned timeline. Get that alignment wrong and the whole picture is useless; getting it right is what you are paying for when you run a system profiler instead of wrapping your code in `time.time()`.

### Trusting the trace: overhead, sampling, and the observer effect

`nsys` is cheap, but it is not free, and knowing where it perturbs your measurement keeps you from chasing artifacts. Three things to keep honest.

First, **the profiler adds overhead, so read ratios, not absolutes.** Tracing every CUDA API call and sampling every thread costs a few percent of wall-clock, and that cost is not uniform — it lands most heavily on the host-side API row, because that is what `nsys` hooks. A service that is already launch-bound (thousands of tiny API calls) will look slightly *more* launch-bound under `nsys` than in production, because each hooked call got a hair slower. This does not change your diagnosis — a copy that is 63% of host time is the wall whether the trace inflated it to 65% or not — but it does mean you should trust the *shape* (which row dominates, by roughly how much) over the exact nanosecond totals. If you need absolute latency numbers, measure those separately with CUDA events on an un-profiled run, as the [reproducible-benchmark post](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark) describes, and use `nsys` only to explain *why* the number is what it is.

Second, **CPU sampling needs permission, and without it the thread rows go dark.** The `osrt` backtrace sampling that reveals a blocked thread relies on kernel performance counters, and on many Linux hosts those are gated behind `perf_event_paranoid`. If your thread rows show CUDA calls but no call-stack samples — no way to see *what* the thread was blocked in — the fix is usually to lower the paranoia level (`echo 1 | sudo tee /proc/sys/kernel/perf_event_paranoid`) or run `nsys` with the appropriate capability. On a container or a locked-down cloud box you may not be able to, in which case you still get the CUDA API and kernel rows (which are enough to see the *gap*), just not the OS-runtime detail that names the blocking call. Know which you have before you conclude "the thread wasn't blocked" — it may just be unsampled.

Third, **add GPU metrics when you want SM activity on the same timeline.** By default `nsys` shows you *what ran*, not *how full the GPU was while it ran*. The `--gpu-metrics-device=all` flag samples hardware counters (SM active, tensor-pipe active, DRAM throughput) and draws them as a graph lane on the timeline, so you can see a kernel that is running but only 6% occupied — the "utilization lies" problem from the [metrics post](/blog/machine-learning/performance-engineering/metrics-that-actually-matter) — right next to the kernel bar itself:

```bash
nsys profile -t cuda,nvtx,osrt --gpu-metrics-device=all \
  -o embed_service -c cudaProfilerApi python serve.py
```

This lane is the bridge between `nsys` (which kernel, when) and `ncu` (why that kernel is slow): if the GPU-metrics lane shows a fat kernel bar sitting at 8% SM-active, you have your `ncu` target *and* a strong hint it is latency- or launch-limited rather than compute-limited. Use it sparingly — GPU-metrics sampling adds its own overhead — but reach for it when "the GPU is busy" needs to become "the GPU is busy doing how much."

## Reading the system timeline: the rows and who waits on whom

Open the report in the Nsight Systems GUI and you get a stack of horizontal rows, one group per process, time flowing left to right. From top to bottom the rows you care about, and what each one is telling you:

- **The CPU thread rows (osrt + sampling).** One row per thread. Color or state tells you whether the thread is *running* (executing your Python, dispatching ops) or *blocked* (parked in a `poll`, a `pthread_cond_wait`, a `cudaStreamSynchronize`). This is where you catch a host-bound service: if the main thread is running flat-out launching tiny kernels, you are launch-bound; if it is blocked in a sync while the GPU sits idle, you are copy-bound or sync-bound. The GIL shows up here too — if you have several Python threads but only one is ever *running* at a time, they are fighting over the interpreter lock.
- **The CUDA API row.** Every `cudaLaunchKernel`, `cudaMemcpyAsync`, `cudaMemcpy`, `cudaStreamSynchronize` call, timestamped on the *host* side. A dense wall of back-to-back `cudaLaunchKernel` bars with no gaps is the signature of a service drowning in launch overhead. A single fat `cudaMemcpy` bar that the thread sits inside for milliseconds is a blocking copy.
- **The per-stream kernel rows.** One row per CUDA stream, showing the actual kernels executing on the GPU. Gaps here are the four wastes made visible: a gap means the GPU is idle. The entire game of `nsys` is explaining every gap on this row by pointing at something on the rows above or below it.
- **The memcpy rows.** Host-to-device (H2D) and device-to-host (D2H) copies on the copy engines. Two things to read here: the *width* (how long the copy took) and whether the copy overlapped with kernels on another stream or serialized against them. A copy that runs while the kernel rows are empty is pure dead time on the GPU.
- **The NCCL row.** On multi-GPU, the collectives. A wide `ncl AllReduce` bar with the compute rows idle beside it is a communication bubble — the thing the [distributed-training profiling post](/blog/machine-learning/distributed-training/profiling-a-distributed-run) and the [compute/communication overlap post](/blog/machine-learning/distributed-training/overlapping-compute-and-communication) exist to kill.

Reading a timeline is the art of **alignment**: you find a gap on the kernel row, drop a vertical line, and read straight up and down to see what every other row was doing at that instant. In our embedding service, the gap on the kernel row lines up with a fat `cudaMemcpy` on the API row, a busy D2H bar on the memcpy row, and a *blocked* state on the main-thread row. Four rows, one instant, one story: the thread called `.cpu()`, that issued a blocking pageable copy, the copy took three milliseconds, and the GPU had nothing to do the entire time because the thread that would have launched the next batch was parked inside the copy.

![a dependency graph showing the main thread branching to a kernel launch and a blocking copy, with the copy waiting on both the kernel and the thread before the gpu goes idle](/imgs/blogs/nsight-systems-for-ai-services-3.webp)

The dependency figure above is the shape of the stall, and it explains why a *fast* kernel does not save you. Follow the arrows. The main thread does two things: it enqueues the matmul kernel (a cheap 8-microsecond launch, the kernel then runs 40 microseconds on the GPU), and then it calls `tensor.cpu()`. That `.cpu()` issues a blocking device-to-host copy — and here is the crux, the copy has **two parents.** It cannot start until the kernel has produced the data (the edge from the kernel), and it holds the main thread hostage until it finishes (the edge from the thread). So the copy is a merge point where the CPU and the GPU dependency chains meet, and while it runs, *both* are stuck: the GPU is idle because no new kernel can be launched, and the CPU is idle because it is blocked inside the driver. The kernel being fast is irrelevant; the wall is the copy the kernel feeds, and the sync the copy forces.

### The mechanism: why a pageable copy is the wall, and the copy tax

The copy is slow for a specific, fixable reason: it is **pageable**, not **pinned**. A tensor you get from `torch.empty(...)` on the CPU lives in ordinary, pageable host memory — memory the OS is free to swap or move. The GPU's copy engine (DMA) cannot read pageable memory directly, because the physical address might change mid-copy. So the CUDA driver does it in two hops: it copies your data into a hidden, pinned "bounce buffer" it controls, then DMAs from there to the device (or the reverse for D2H). That staging halves your effective bandwidth and, worse, forces the whole thing to be synchronous — `cudaMemcpy` on pageable memory cannot be made asynchronous, so it blocks the calling thread.

Pin the host buffer — allocate it with `pin_memory=True` or `torch.empty(..., pin_memory=True)` — and the DMA engine reads it directly, at full PCIe bandwidth, and the copy can be issued with `cudaMemcpyAsync` on a stream so the thread does not block. The difference is a law you can compute. The copy tax is just bytes over bandwidth:

$$t_\text{copy} = \frac{\text{bytes}}{\text{BW}_\text{effective}}$$

On our A100 80GB with a PCIe Gen4 x16 host link, effective bandwidth is roughly 8 GB/s for a pageable copy (staged through the bounce buffer) and roughly 24 GB/s for a pinned copy (direct DMA); these are approximate and depend on the platform, but the ratio is real. For the 24 MB output tensor:

$$t_\text{pageable} = \frac{24\text{ MB}}{8\text{ GB/s}} \approx 3.0\text{ ms} \qquad t_\text{pinned} = \frac{24\text{ MB}}{24\text{ GB/s}} \approx 1.0\text{ ms}$$

Pinning alone cuts the copy from 3 ms to 1 ms. But the bigger win is that the pinned copy can be *asynchronous and overlapped*: issue it on a side stream with `non_blocking=True` and it runs concurrently with the next request's kernels, so its one millisecond hides behind compute and leaves the critical path entirely. That is the difference between a copy that costs 3 ms of dead GPU time per request and one that costs zero. Here is the comparison that drives the fix:

| Property | Pageable D2H | Pinned D2H |
|---|---|---|
| Effective PCIe Gen4 BW | ~8 GB/s (staged) | ~24 GB/s (direct DMA) |
| Blocks the calling thread? | yes (synchronous) | no (`cudaMemcpyAsync`) |
| Can overlap with compute? | no | yes, on a side stream |
| Cost of a 24 MB copy | ~3.0 ms, all on critical path | ~1.0 ms, hidden behind compute |

## Same run, two views

The most convincing way to internalize what `nsys` adds is to look at the exact same thirty-request capture through both profilers and watch one of them miss the wall entirely.

![a two column comparison contrasting the torch profiler verdict of healthy gpu bound kernels against the nsight systems verdict that a pageable copy is the real wall](/imgs/blogs/nsight-systems-for-ai-services-4.webp)

Start with what `torch.profiler` printed for the same run. This is the table that sent the previous owner shopping for a bigger box:

```console
-------------------------------  ------------  ------------  ------------  ------------  ------------
                           Name    Self CUDA   Self CUDA %    CPU total     CUDA total    # of Calls
-------------------------------  ------------  ------------  ------------  ------------  ------------
              ampere_sgemm_128     18.240ms        52.1%       19.010ms      18.240ms           480
      ampere_fp16_gemm_64x64_tn      9.980ms        28.5%       10.410ms       9.980ms           480
                    aten::copy_      4.010ms        11.5%        6.220ms       4.010ms            30
              elementwise_kernel      2.760ms         7.9%        3.020ms       2.760ms          1440
-------------------------------  ------------  ------------  ------------  ------------  ------------
Self CUDA time total: 34.99ms
```

Read it the way most engineers do and you conclude the service is healthy. Self-CUDA time is dominated by GEMMs, 80% of it in two matmul kernels, exactly what a compute-bound transformer should look like. `aten::copy_` is a modest 11.5%, easy to dismiss as bookkeeping. There is no idle time in this table *because the table has no concept of idle time* — `torch.profiler`'s `key_averages()` sums the busy time of each operator; it does not, by default, show you the gaps *between* operators, and it certainly does not show you a thread blocked in the driver. The verdict "GPU-bound" is the only verdict this view can produce.

Now the `nsys` view of the identical run. The `cudaapisum` report — the host-side cost of each CUDA API call — tells a different story:

```console
 ** CUDA API Summary (cudaapisum):

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)    Med (ns)     Name
 --------  ---------------  ---------  ----------  ----------  ---------------------------
    62.8        91,140,000         30  3,038,000   3,041,000   cudaMemcpy
    19.1        27,720,000        960     28,875      9,200     cudaLaunchKernel
    14.3        20,760,000         30    692,000     690,500   cudaStreamSynchronize
     3.1         4,500,000         60     75,000      9,400     cudaMalloc
     0.7         1,020,000        480      2,125      2,050     cudaMemcpyAsync
```

There it is. `cudaMemcpy` — the *blocking* copy, note it is not `cudaMemcpyAsync` — is 62.8% of all host-side CUDA time, three milliseconds per call, thirty calls, one per request. The GEMMs that dominated the `torch.profiler` table are hiding inside those 960 `cudaLaunchKernel` calls that account for a fifth of the host time. The thing `torch.profiler` reported as a benign 11.5% `aten::copy_` is, on the system timeline, the largest single consumer of wall-clock time in the service. Same run, same thirty requests, opposite conclusions. One profiler measures operator busy-time; the other measures where the wall-clock actually went, and only the second one could see that the wall was a copy.

#### Worked example: catching the copy that torch.profiler dismissed

You inherit the embedding service: A100 80GB, p50 55 ms, p99 92 ms, 180 req/s, spending about \$2.20 per GPU-hour on-demand. The runbook says buy a bigger box. You do not buy the box. You warm up the service, wrap thirty steady requests in `cudaProfilerStart/Stop`, and run `nsys profile -t cuda,nvtx,osrt,cublas -c cudaProfilerApi python serve.py`.

You open the timeline and zoom to a single request. The kernel row shows about 35 ms of dense GEMM activity — genuinely healthy compute — and then a 3 ms gap where the kernel row is empty. You drop a vertical cursor in the gap. Directly above, the CUDA API row shows a fat `cudaMemcpy` bar. Below, the D2H memcpy row shows a copy in flight, and the properties pane says **"Memory kind: Pageable."** On the main-thread row, the thread state for those 3 ms is *blocked*. The story assembles itself: the handler's last line is `return embeddings.cpu().tolist()`, `.cpu()` copies a 24 MB fp32 tensor to a pageable host buffer, the copy is synchronous and pageable so it blocks the thread for 3 ms, and during those 3 ms the GPU is idle because the thread that would launch the next batch is stuck.

Under 50 concurrent requests, that per-request 3 ms bubble is not just 3 ms — because the blocking copy serializes launches, the bubbles stack into a queueing tail, which is why p99 (92 ms) is so much worse than p50 (55 ms). The fix is two changes: copy into a pre-allocated pinned buffer, and issue it on a side stream with `non_blocking=True` so it overlaps the next request's compute.

```python
# Before: pageable, synchronous, blocks the thread for ~3 ms
result = embeddings.cpu().tolist()

# After: pinned destination + async copy on a side stream, overlaps compute
copy_stream = torch.cuda.Stream()
pinned = torch.empty_like(embeddings, device="cpu", pin_memory=True)
with torch.cuda.stream(copy_stream):
    pinned.copy_(embeddings, non_blocking=True)   # cudaMemcpyAsync, no thread block
# ... launch the NEXT request's kernels here; the copy runs concurrently ...
copy_stream.synchronize()                          # wait only when you truly need the bytes
result = pinned.tolist()
```

Re-profile and the 3 ms gap on the kernel row is gone: the copy now shows on the D2H row *overlapping* the next request's GEMMs, the main thread is no longer blocked, and the kernel row is continuous. The measured result on the A100:

| Metric | Before (pageable, sync) | After (pinned, overlapped) |
|---|---|---|
| GPU util (nvidia-smi) | 72% | 91% |
| p50 latency | 55 ms | 34 ms |
| p99 latency | 92 ms | 38 ms |
| Throughput | 180 req/s | 290 req/s |
| Per-request D2H cost on critical path | ~3.0 ms | ~0 ms (hidden) |
| Host-side `cudaMemcpy` time | 62.8% of API | 0.7% (now async) |
| \$ per million requests | ~\$3.40 | ~\$2.10 |

No bigger box. A pinned buffer and a side stream, found because one profiler could see a copy and a blocked thread that the other could not. This is the general shape of a copy-bound service, and it recurs anywhere a handler pulls a large tensor back to the host — one of the four wastes the series is organized around.

## Finding the wall: the method

The embedding-service story was not luck; it was a method, and the method is the transferable skill. It is the same four moves every time, whether the wall turns out to be a copy, a launch storm, a sync, or a CPU thread.

![a left to right timeline of the find the wall method, from zooming out on one request through naming the owner of the longest span to picking a fine tool and re-measuring](/imgs/blogs/nsight-systems-for-ai-services-5.webp)

**Move one: zoom out to one steady-state request.** Not the whole trace, not one kernel — one representative request, start to finish. You are looking for the shape of a single unit of work. If your capture range was set up right, every request looks about the same, and you pick any one in the middle.

**Move two: find the longest span that is either idle or dominated by one thing.** Scan the kernel row for the widest gap (idle GPU), or the widest single bar (one kernel eating everything), or a wide bar on the memcpy or NCCL row. You are not trying to understand the whole request yet — you are looking for the single biggest contiguous chunk of time, because Amdahl's law says that is the only thing worth fixing first. A span that is 3% of the request is not the wall no matter how ugly it looks.

**Move three: name the owner of that span.** Drop a vertical cursor at the span and read every row at that instant. Idle kernel row + fat `cudaMemcpy` + blocked thread = a blocking copy. Idle kernel row + a dense wall of `cudaLaunchKernel` + a running, flat-out CPU thread = launch overhead. Idle kernel row + a wide NCCL bar = a comms bubble. One fat kernel bar with everything else quiet = a slow kernel. The owner is whichever row explains the span.

**Move four: switch to the right fine tool.** `nsys` found *where* and *what*; now you need *why*, and `nsys` is deliberately not the tool for that. If the owner is a kernel, you now know exactly which kernel to hand to `ncu`. If it is a Python hot path, you hand it to `py-spy`. If it is a copy or a sync, you do not need a finer tool at all — you already know the fix. This is the map-then-microscope discipline made concrete: `nsys` is the map that tells you which microscope to pick up, and pointing a microscope at the wrong spot is how investigations die.

The reason this method works is that it is ruthlessly Amdahl-bounded. The speedup you can get from fixing one span is capped at $\frac{1}{1-p}$ where $p$ is the fraction of the request that span occupies. If the copy is 3 ms of a 55 ms request, fixing it perfectly caps your win at about 5%… except that under concurrency the *serialization* the copy causes multiplies its impact, which is why the p99 win (92 to 38 ms) dwarfed the p50 win. The method forces you to quantify $p$ before you spend a day on a fix, so you never optimize a span that cannot pay you back.

## The wall is a, reach for

Move four deserves its own map, because "pick the right fine tool" is not a judgment call — it is a lookup table keyed on what `nsys` showed you. Once you have named the owner of the longest span, the fine tool and the class of fix are nearly mechanical.

![a decision tree routing each kind of wall, a hot kernel, a copy, a launch bound cpu, or a needless sync, to its matching fine tool and fix](/imgs/blogs/nsight-systems-for-ai-services-6.webp)

Read the tree from the root. `nsys` told you the longest span's owner; each owner routes to one tool and one remedy:

- **The wall is a hot kernel** (one fat bar on the kernel row, everything else quiet). Reach for `ncu --set full -k <kernel_name>`. It will tell you whether the kernel is memory-bound, compute-bound, or latency-bound, and what to do about it — the [Nsight Compute sibling](/blog/machine-learning/performance-engineering/nsight-compute-kernel-deep-dive) is the whole story. This is the only branch where you need another profiler.
- **The wall is a copy** (idle kernel row, fat memcpy bar, blocked or copying thread). You do not need `ncu`. Pin the host buffer and issue the copy with `non_blocking=True` on a side stream so it overlaps compute. If the copy is unavoidable and large, question whether it needs to happen at all — can the post-processing stay on the GPU?
- **The wall is launch overhead** (idle kernel row, dense `cudaLaunchKernel` wall, a CPU thread running flat-out). Reach for `py-spy` to confirm the Python dispatch path is the hot loop, then attack it with CUDA graphs or `torch.compile(mode="reduce-overhead")` to collapse thousands of launches into one replay. The [model-serving post on fusion, graphs, and compile](/blog/machine-learning/model-serving/kernel-fusion-cuda-graphs-torch-compile) is the fix.
- **The wall is a needless sync** (a `cudaStreamSynchronize` or a `.item()`/`.cpu()` that forces the GPU to drain mid-pipeline). Often the fix is simply to *delete the sync* — a stray `print(loss.item())` inside the hot loop, a `.cpu()` you did not need until the end, a `torch.cuda.synchronize()` left over from debugging. Each of these drains the pipeline and serializes host and device. Move the sync out of the loop, or remove it.

The point of the tree is that `nsys` collapses an open-ended question ("why is my service slow?") into a closed one ("which of these four rows owns the longest span?"), and the closed question has a lookup-table answer. That is the entire value of a system profiler: it turns debugging into routing.

## nsys stats: reading the trace headless

The GUI is where you learn to read timelines, but you will spend most of your professional life on a locked-down inference host with no display, an SSH session, and a `.nsys-rep` file. The `nsys stats` command prints the same information the GUI's summary panes show, as text tables you can read over SSH or diff between two runs. This is not a lesser mode — for the "which span is biggest" question, the CLI is often *faster* than the GUI, because it hands you a sorted table directly.

![a two column grid of the four nsys stats reports, gpukernsum, cudaapisum, gpumemtimesum, and osrtsum, each paired with the question it answers](/imgs/blogs/nsight-systems-for-ai-services-7.webp)

Four reports answer four questions, as the grid above lays out. You ask for them by name:

```bash
nsys stats \
  --report gpukernsum,cudaapisum,gpumemtimesum,osrtsum \
  embed_service.nsys-rep
```

The first, `gpukernsum` (GPU kernel summary), answers *which kernel dominates the GPU*. It is the CLI equivalent of scanning the kernel row for the widest bars:

```console
 ** CUDA GPU Kernel Summary (gpukernsum):

 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)   Max (ns)               Name
 --------  ---------------  ---------  --------  ---------  ---------  ---------  ----------------------------
    52.1        18,240,000        480    38,000     37,900     36,200     41,100   ampere_sgemm_128x64_nn
    28.5         9,980,000        480    20,791     20,700     19,900     22,400   ampere_fp16_gemm_64x64_tn
     7.9         2,760,000       1440     1,916      1,880      1,700      2,300   elementwise_kernel
     6.2         2,170,000        480     4,520      4,480      4,100      5,000   layernorm_kernel
     5.3         1,860,000        480     3,875      3,800      3,500      4,300   softmax_warp_kernel
```

The second, `cudaapisum` (CUDA API summary), answers *where the host-side time went* — launch overhead versus copy versus sync. You already saw ours: `cudaMemcpy` at 62.8% is the smoking gun. In a launch-bound service this report is dominated instead by `cudaLaunchKernel` with tens of thousands of calls, each a few microseconds, which is the fingerprint of the kernel-launch-overhead problem that CUDA graphs and [`torch.compile` with fusion](/blog/machine-learning/model-serving/kernel-fusion-cuda-graphs-torch-compile) exist to fix.

The third, `gpumemtimesum` (GPU memory-operations summary, by time), answers *the copy tax* — how much wall-clock the copies cost and, crucially, whether they were pageable:

```console
 ** CUDA GPU MemOps Summary (by Time) (gpumemtimesum):

 Time (%)  Total Time (ns)  Count  Avg (ns)   Med (ns)  Min (ns)  Max (ns)          Operation
 --------  ---------------  -----  --------  ---------  --------  --------  --------------------------------
    94.6        90,720,000     30  3,024,000  3,020,000  2,910,000  3,180,000  [CUDA memcpy Device-to-Host]
     5.4         5,160,000     30    172,000    170,500    160,000    190,000  [CUDA memcpy Host-to-Device]
```

Thirty D2H copies, three milliseconds each, 90.7 ms of copy time in a thirty-request trace — that is three milliseconds of copy on the critical path of *every single request*. The H2D copies (loading inputs) are a twentieth the cost. If you cross-reference this with `gpukernsum`, whose total kernel time was about 35 ms, you see the copies cost nearly three times what the compute cost. No GUI needed to conclude the wall is D2H.

The fourth, `osrtsum` (OS runtime summary), answers *what the CPU threads were blocked on*. This is the report that catches a host-bound service the other three miss:

```console
 ** OS Runtime Summary (osrtsum):

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)      Name
 --------  ---------------  ---------  -----------  -------------------
    71.4       106,200,000        128    829,687     pthread_cond_wait
    18.9        28,110,000       2400     11,712     poll
     6.1         9,070,000        960      9,448     ioctl
     3.6         5,350,000        480     11,145     futex
```

Seventy-one percent of OS-runtime time in `pthread_cond_wait` — threads parked, waiting. On a healthy compute-bound service you *want* to see the launching thread mostly running, not blocked; a fat `pthread_cond_wait` next to an idle GPU means your host is waiting on something (here, the blocking copy) instead of feeding the device.

#### Worked example: sizing the copy tax before you touch a line of code

A teammate on a vision service claims their preprocessing is fine and the model is the bottleneck. L4 GPU, 300 GB/s HBM, PCIe Gen4 host link, serving a ResNet at p99 41 ms, GPU util reported at 60%. Before arguing, you profile: warm up, capture 40 steady requests, `nsys stats --report gpukernsum,gpumemtimesum,osrtsum` over SSH because the box has no display.

`gpumemtimesum` shows H2D copies (the input images) totaling 240 ms across 40 requests, 6 ms each, and every one tagged pageable. `gpukernsum` shows the conv kernels totaling 320 ms, 8 ms each. So the input copy is 6 ms of a request whose compute is 8 ms — the copy is 43% of the GPU-visible work, and it is on the critical path because the DataLoader hands over pageable numpy arrays. `osrtsum` confirms the worker threads spend most of their time in `pthread_cond_wait`, blocked on the pinned-buffer-less copy. The arithmetic settles the argument without a debate: the report measured 6 ms of pageable H2D transfer per batch, and the copy-tax law says a pinned, directly-DMA'd transfer moves at roughly three times the effective bandwidth — so pinning the DataLoader (`pin_memory=True`) plus `non_blocking=True` on the `.to(device)` call should cut that copy to about 2 ms and, issued on a side stream, overlap it with the previous batch's compute. You did not open the GUI or guess; three CLI tables sized the tax and named the fix. This headless workflow is why `nsys stats` belongs in your muscle memory, not just the GUI.

## NVTX: labeling the timeline with your code

By default the timeline speaks CUDA's language — `ampere_sgemm_128x64_nn`, `cudaLaunchKernel`, `[CUDA memcpy Device-to-Host]` — not yours. It does not know that kernels 1 through 40 are your tokenizer, 41 through 300 are the encoder, and 301 through 320 are the pooling head. NVTX (NVIDIA Tools Extension) fixes that: you push named ranges around your code's phases, and `nsys` draws them as a labeled lane at the top of the timeline, so the profiler rows map back to your handler.

```python
import torch

def handle_request(batch):
    with torch.autograd.profiler.emit_nvtx():  # optional: also tag every aten op
        torch.cuda.nvtx.range_push("tokenize")
        tokens = tokenizer(batch)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("encode")
        hidden = model.encoder(tokens.to("cuda", non_blocking=True))
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("pool_and_copy")
        emb = pool(hidden)
        result = emb.cpu()          # the copy we are hunting shows up inside this range
        torch.cuda.nvtx.range_pop()
    return result
```

Better than bare `range_push`/`range_pop` is a context manager, so an exception cannot leak an unclosed range:

```python
from contextlib import contextmanager
import torch

@contextmanager
def nvtx_range(name):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()   # always closes, even on exception

def handle_request(batch):
    with nvtx_range("tokenize"):
        tokens = tokenizer(batch)
    with nvtx_range("encode"):
        hidden = model.encoder(tokens.to("cuda", non_blocking=True))
    with nvtx_range("pool_and_copy"):
        return pool(hidden).cpu()
```

Now the trace has a top lane reading `tokenize | encode | pool_and_copy`, and the 3 ms copy bubble sits visibly inside `pool_and_copy` rather than floating anonymously among CUDA memcpys. When you find a gap, you read up to the NVTX lane and instantly know which phase of *your* handler owns it — no cross-referencing line numbers against kernel names. NVTX is the single highest-leverage change you can make to a trace's readability, and it costs three lines. The deeper treatment — annotating CUDA graphs, custom domains and categories, colorizing lanes, and building semantic traces that survive `torch.compile` — is the subject of the [sibling post on NVTX and semantic profiling traces](/blog/machine-learning/performance-engineering/nvtx-and-semantic-profiling-traces); here the takeaway is just: always add NVTX ranges around your handler phases before you profile a service, because an unlabeled system timeline is far harder to read than a labeled one.

## Bottleneck triage: inside PyTorch versus outside it

The deepest reason to own `nsys` is that it draws the boundary `torch.profiler` cannot: the line between a bottleneck *inside* your PyTorch program and one *outside* it. Get on the wrong side of that line and you will optimize code that was never the problem.

**Inside PyTorch** means the wall is an operator or a kernel your model launched: a slow GEMM, an unfused elementwise chain, an attention kernel that spills to HBM. For these, `torch.profiler` is the better first tool — it ties the kernel straight to the Python line and the input shapes, and its Chrome-trace export (see the [chrome-trace sibling](/blog/machine-learning/performance-engineering/profiling-pytorch-with-torch-profiler)) shows the op timeline in perfetto. `nsys` will also show these, but with less PyTorch context.

**Outside PyTorch** means the wall is something PyTorch never launched and cannot instrument, and this is `nsys`'s home turf. The catalogue of things that live outside PyTorch's view is exactly the catalogue of things that make a "healthy" service slow:

- **A blocking copy** — our embedding service. `torch.profiler` sees `aten::copy_` but not the thread it blocks.
- **A synchronization** — a `.item()`, a `.cpu()`, a `torch.cuda.synchronize()` that drains the pipeline. `torch.profiler` records the op that syncs but not the GPU-idle gap it creates.
- **A CPU thread on the GIL** — several request threads but only one running at a time. `torch.profiler` cannot see the Python interpreter lock; `nsys`'s osrt rows show it directly as threads blocked in `futex`/`pthread_cond_wait`.
- **A dataloader worker** — the input pipeline falling behind, GPU starved between batches. `nsys` shows the worker processes and their copies, which live in separate processes PyTorch's profiler never attaches to.
- **A second process** — a sidecar, a logging daemon, another model sharing the GPU, contending for SMs or PCIe. Only a whole-machine profiler sees the neighbor.
- **NCCL** — on multi-GPU, a collective waiting on a straggler rank. `torch.profiler` sees the `nccl:all_reduce` op start; only `nsys` (with `-t nccl`) shows the collective's actual bar and whether compute overlapped it.

### Stress-testing the diagnosis

A wall found at one operating point is not a wall found. Before you commit a fix, push on it:

**Batch 1 versus batch 64.** Our copy scaled with output size, so at batch 1 the 24 MB copy shrinks to ~0.4 MB and the wall vanishes — the service really is compute-bound at batch 1 and copy-bound at batch 64. If you only ever profiled batch 1, you would never see the wall. Always profile at the batch size production actually runs. Conversely, a launch-overhead wall is *worst* at small batch (many tiny kernels, little compute to hide the launches) and hides at large batch — the opposite scaling. Which way the wall scales with batch size is itself a diagnostic.

**L4 versus A100.** The same copy that costs 3 ms on the A100's Gen4 link costs about the same on an L4's Gen4 link (PCIe bandwidth is a platform property, not a GPU-compute property), but because the L4's compute is slower, the *fraction* of the request the copy represents is smaller — so the identical bug is a bigger deal on the faster GPU. Fixes that matter on an A100 can be noise on an L4, and vice-versa. Name the hardware when you report a win.

**Shapes that vary each request.** If your service takes variable sequence lengths, the copy size and the kernel times both vary, and the "one steady request" you zoomed into may not be representative. Capture more requests, and sort `gpukernsum` and `gpumemtimesum` by max as well as average to catch the tail request that dominates p99.

**Fifty concurrent requests.** The single biggest lesson of the embedding service: a 3 ms per-request bubble that looks trivial in isolation becomes a 54 ms p99 tail under concurrency, because the blocking copy *serializes* the launch loop. You must profile under representative concurrency, because serialization effects are invisible at concurrency 1. The [reproducible-benchmark post](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark) covers how to generate that load honestly.

**Memory-bound versus compute-bound.** If the wall is a single fat kernel, `nsys` cannot tell you whether it is starved on HBM bandwidth or genuinely maxing the math pipes — that is an `ncu` question, and the answer changes the fix completely (fuse to cut memory traffic versus increase occupancy). `nsys` routes you to `ncu`; it does not replace it.

## Case studies and real numbers

Three investigations where the wall was outside PyTorch and `nsys` was the tool that found it. Numbers are representative of what the shown reports produce; treat the exact figures as illustrative of the shape, not universal constants.

**The pageable D2H copy (A100 80GB).** The embedding service of this whole post. Symptom: p99 92 ms, "GPU-bound" per `torch.profiler`. `nsys` `cudaapisum` showed `cudaMemcpy` at 62.8% of host time, 3 ms per request, pageable. Fix: pinned destination buffer plus `non_blocking=True` on a side stream. Result: p99 92 to 38 ms, GPU util 72% to 91%, throughput 180 to 290 req/s. The fix touched five lines; finding it required seeing the thread block, which only a whole-machine profiler shows.

**The launch-bound decode loop (L4).** A small-model service on an L4 sat at 40% GPU util with `torch.profiler` reporting healthy kernels. `nsys` `cudaapisum` was dominated by `cudaLaunchKernel` — tens of thousands of 4-microsecond launches — with the main thread running flat-out and the kernel row full of gaps: the classic host-bound, launch-overhead signature. The Python dispatch loop could not refill the GPU's launch queue fast enough. Fix: `torch.compile(mode="reduce-overhead")`, which wraps the model in CUDA graphs and collapses the launches into replays, as the [model-serving fusion/graphs/compile post](/blog/machine-learning/model-serving/kernel-fusion-cuda-graphs-torch-compile) details. Util climbed from roughly 40% toward 85%, and the gaps on the kernel row closed. NVIDIA's own CUDA Graphs material documents the launch-overhead mechanism and the graph-replay remedy; the effect is real and large for many-small-kernel workloads.

**The NCCL comms bubble (2× A100).** A data-parallel training step showed the GPUs busy on `nvidia-smi` but throughput below the roofline estimate. Single-GPU `torch.profiler` cannot see cross-GPU waits. `nsys -t cuda,nvtx,osrt,nccl` on the multi-GPU run drew the NCCL row and revealed a wide `AllReduce` bar with the compute rows idle beside it — the gradient all-reduce was not overlapping the backward pass, so every step paid full communication time serially. This is the exact problem the [compute/communication overlap post](/blog/machine-learning/distributed-training/overlapping-compute-and-communication) and the [distributed-run profiling post](/blog/machine-learning/distributed-training/profiling-a-distributed-run) address: bucket the gradients and overlap the all-reduce with backward compute so the NCCL bar hides under the kernel row. Only a system profiler that traces NCCL could have shown the bubble at all.

The through-line: in all three, the headline dashboard metric (`nvidia-smi` utilization) and the PyTorch-scoped profiler both said "fine," and in all three the wall was a copy, a launch pattern, or a collective — something on a row that only `nsys` draws.

## When to reach for nsys (and when not to)

`nsys` is the right first profiler for a specific class of problem, and the wrong tool for others. Being decisive about this saves the most time.

**Reach for `nsys` when** the GPU is idle and you do not know why; when the symptom is a latency tail (p99 much worse than p50) that smells like a periodic stall; when you suspect a copy, a sync, or the host feeding the device; when more than one process shares the GPU; when you are on multi-GPU and need to see NCCL; or, honestly, as the *default* second step after `torch.profiler` on any "it's slow and I can't see why" investigation. It is cheap enough to run against a live service, and it is the only tool that draws the whole machine.

**Do not reach for `nsys` when** you already know the wall is one specific kernel and you need to know *why that kernel* is slow — that is `ncu`, and `nsys` will only frustrate you by showing the kernel without explaining it. Do not reach for it when the hot path is clearly Python-side business logic (JSON serialization, tokenization in pure Python, a slow feature transform) — `py-spy` gives you a flame graph of the Python stack far faster than reading a system timeline. Do not reach for it to compare two model architectures' operator costs — `torch.profiler`'s `key_averages()` table is the direct answer. And do not chase a p99 tail with `nsys` before you have confirmed the tail is a *stall* and not simply *load*: if p99 is bad because the service is at capacity, the fix is more capacity or better batching, not a timeline read.

The one-line rule: **`nsys` is the map; `ncu`, `torch.profiler`, and `py-spy` are the microscopes.** Reach for the map when you do not know where you are, and for a microscope only once the map has pointed at a spot. The full symptom-to-tool routing is collected in the [capstone playbook](/blog/machine-learning/performance-engineering/the-performance-engineering-playbook).

## Key takeaways

- **`torch.profiler` shows PyTorch's view; `nsys` shows the whole machine.** When the wall is a copy, a sync, a CPU thread, a dataloader, or NCCL, it lives outside PyTorch, and only a system profiler draws it.
- **Profile the steady state, not the cold start.** Warm up, then bound the capture with `cudaProfilerStart/Stop` and `-c cudaProfilerApi`, so the trace is the requests you care about and nothing else.
- **The `-t` domains are the rows.** `cuda` draws kernels and API; `osrt` draws the CPU thread states that reveal blocking; `nvtx` labels your phases; `nccl` draws collectives. Forgetting `osrt` is why people see idle GPU but can't prove the cause.
- **Read the timeline by alignment.** Find the longest idle-or-dominant span on the kernel row, drop a cursor, read every row at that instant, name the owner.
- **A pageable copy is synchronous and half-speed; a pinned copy is async and overlappable.** The copy tax is bytes over bandwidth, and pinning plus a side stream can move it off the critical path entirely.
- **The method is four moves:** zoom to one request, find the longest span, name its owner, switch to the right fine tool. It is Amdahl-bounded, so you never fix a span too small to pay you back.
- **`nsys stats` reads traces headless.** `gpukernsum`, `cudaapisum`, `gpumemtimesum`, `osrtsum` answer which kernel, which API cost, the copy tax, and what the threads blocked on — no GUI required.
- **Add NVTX ranges before you profile a service.** Three lines turn an anonymous CUDA timeline into a labeled map of your handler phases.
- **Stress-test every diagnosis** across batch size, GPU model, variable shapes, and concurrency — a 3 ms bubble at concurrency 1 can be a 54 ms p99 tail under load.
- **`nsys` is the map, not the microscope.** It tells you which finer tool to pick up; pointing `ncu` at the wrong kernel is how a day disappears.

## Further reading

- [NVIDIA Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) — the authoritative reference for `nsys profile`, the `-t` domains, `nsys stats` reports, and the timeline.
- [NVIDIA CUDA C++ Programming Guide — pinned (page-locked) host memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#page-locked-host-memory) — why pageable copies stage through a bounce buffer and pinned copies do not.
- [PyTorch: `torch.cuda.nvtx` and profiler NVTX](https://pytorch.org/docs/stable/generated/torch.cuda.nvtx.range_push.html) — the API for annotating your timeline.
- [Profiling LLM serving with Nsight](/blog/machine-learning/model-serving/profiling-llm-serving-with-nsight) — the same ladder applied to token generation, with prefill/decode timelines.
- [Profiling PyTorch with torch.profiler](/blog/machine-learning/performance-engineering/profiling-pytorch-with-torch-profiler) — the PyTorch-scoped profiler that is the first rung of the ladder.
- [Nsight Compute kernel deep-dive](/blog/machine-learning/performance-engineering/nsight-compute-kernel-deep-dive) — the microscope for when `nsys` names one kernel as the wall.
- [NVTX and semantic profiling traces](/blog/machine-learning/performance-engineering/nvtx-and-semantic-profiling-traces) — deeper NVTX: custom domains, CUDA-graph annotations, traces that survive compile.
- [Why your AI service wastes CPU and GPU](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu) and [the performance-engineering playbook](/blog/machine-learning/performance-engineering/the-performance-engineering-playbook) — the series intro and the capstone decision tree.
