---
title: "Profiling PyTorch with torch.profiler: your first real look at where time goes"
date: "2026-07-14"
publishDate: "2026-07-14"
description: "A hands-on tour of torch.profiler — the schedule, the key_averages table, record_function, and Chrome-trace export — so you can stop guessing which part of your AI service is slow and read the answer off a profile."
tags:
  [
    "performance-engineering",
    "gpu-optimization",
    "profiling",
    "pytorch",
    "torch-profiler",
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
readTime: 41
---

You have a service that feels slow. A request that should take single-digit milliseconds takes twenty. The GPU dashboard says utilization is high, so someone in the standup suggests a bigger box. Someone else suspects the model. A third person is sure it's the tokenizer. Everyone has a theory, and every theory is a guess, because *slow where?* is not a question you can answer by staring at code or at a green line on a Grafana panel. It is a measurement. Until you take that measurement, every optimization you ship is a bet placed in the dark, and half the time you will "fix" the part that was already fast and leave the real bottleneck untouched.

`torch.profiler` is the cheapest instrument that ends the guessing, and it is already installed — it ships inside PyTorch, needs no separate download, no root, no special driver flags. In about ten lines of wrapper code it will tell you, per operator, how much time was spent on the CPU launching work versus on the GPU doing it, how many kernels ran, what shapes flowed through them, and how many bytes each allocated. It is the first tool you reach for in the profile → hypothesize → fix → measure loop that runs through this whole series, and it is the one you'll reach for most often, because it answers the first and most important question — *where does the time actually go?* — before you ever escalate to heavier machinery like Nsight.

![a horizontal schedule of profiler steps starting with a skipped wait step then a warmup step then three recorded active steps ending in a ready trace](/imgs/blogs/profiling-pytorch-with-torch-profiler-1.webp)

The figure above is the shape of a profiling run, and it is the first thing to understand because it governs *what* you measure. You do not profile a whole training run or an entire request flood; you profile a handful of steady-state steps, chosen deliberately: skip the first (the `wait` step), pay one-time costs on the next (the `warmup` step), then record a few clean iterations (the `active` steps). By the end of this post you'll be able to write that schedule, read the table it produces, name the bottleneck from the shape of that table, and export a timeline you can scrub through kernel by kernel. This is the [second track](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu) of the playbook — the tools — and `torch.profiler` is where it starts. Later posts assume you can drive it cold.

## Slow where? The guess you cannot afford

Let me make the cost of guessing concrete, because it is the entire justification for spending an afternoon learning a profiler. Suppose the service is an inference endpoint on a single A100 80GB, serving a small transformer, and p50 latency is 18 ms. You have three plausible stories. Story one: the GPU is compute-bound and you need an H100. Story two: the model does too much redundant work and needs fusion. Story three: Python can't launch kernels fast enough and the GPU spends most of each request idle, waiting to be fed. Each story points to a completely different fix — a hardware purchase, a `torch.compile` pass, or CUDA graphs — and at most one of them is right. Acting on the wrong one costs a sprint and, in the hardware case, real money on a machine that will sit 90% empty.

![a two panel comparison contrasting a vague slow somewhere guess with a bigger gpu bet against a profiled fact naming softmax at forty-two percent of device time](/imgs/blogs/profiling-pytorch-with-torch-profiler-2.webp)

The two panels above are the before and after of taking one measurement. On the left is the state everyone starts in: *slow somewhere*, zero measurements, and the tempting, expensive reflex to buy a bigger card. On the right is what a five-minute profile gives you: a named operator with a number attached — *softmax is 42% of device time*, say, or *the CPU spends five times longer launching kernels than the GPU spends running them*. One of those is a fact you can attack with one targeted change. The gap between the panels is not intelligence or seniority; it is whether you profiled. This series' recurring frame — the [four wastes](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu): idle GPU, low occupancy, the bandwidth wall, redundant work — only becomes *actionable* once a profile tells you which waste you actually have. The profiler is how the invisible becomes a row in a table.

## The instrument: torch.profiler in one block

Here is the full invocation, every knob turned, wrapped around a normal inference loop. Read it once end to end; we'll then take it apart argument by argument.

```python
import torch
from torch.profiler import (
    profile, schedule, ProfilerActivity, record_function,
    tensorboard_trace_handler,
)

model = build_model().cuda().eval()          # your model
inputs = make_batch().cuda()                  # a representative batch

# The schedule decides WHICH steps get recorded.
prof_schedule = schedule(wait=1, warmup=1, active=3, repeat=1)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=prof_schedule,
    on_trace_ready=tensorboard_trace_handler("./log/inference"),
    record_shapes=True,      # capture input shapes per op (some overhead)
    profile_memory=True,     # capture per-op allocations
    with_stack=True,         # capture Python call stack (expensive!)
) as prof:
    for step in range(1 + 1 + 3):            # wait + warmup + active
        with torch.no_grad():
            with record_function("inference_step"):
                out = model(inputs)
        torch.cuda.synchronize()             # finish device work this step
        prof.step()                          # advance the schedule

# Read the aggregated table, sorted by total device time.
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

That is the whole thing. Nine of those lines are your loop; the profiling is the `with profile(...)` context, the `prof.step()` call, and the `print` at the end. Every argument earns its place, so let's go through them.

**`activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]`** tells the profiler which two worlds to watch. `CPU` records the host-side operator dispatch — the Python-driven `aten` calls, the `cudaLaunchKernel` calls, the time your CPU spends preparing and enqueuing work. `CUDA` records the device-side kernels — the actual arithmetic running on the GPU's streaming multiprocessors. You almost always want both, because the single most common finding in an AI service is a *mismatch* between them: the CPU busy for 25 ms launching work the GPU finishes in 6 ms. If you record only one world you cannot see that gap, and the gap is usually the whole story.

**`schedule=schedule(wait=1, warmup=1, active=3, repeat=1)`** is the state machine from figure 1, and it is the argument people most often get wrong by omitting it. Without a schedule, the profiler records *every* iteration from the moment the context opens, including the cold ones — the first forward pass that triggers cuDNN autotuning, the allocator growing its pool, lazy CUDA context initialization. Those one-time costs can be ten times a steady-state step, and they will dominate and distort your averages. The schedule fixes this by moving through phases, which we'll dissect next.

**`on_trace_ready=tensorboard_trace_handler("./log/inference")`** is a callback fired once each `active`+`repeat` cycle completes. This particular handler writes a trace file that the TensorBoard PyTorch Profiler plugin can render. You can pass your own function instead — a common pattern is a lambda that calls `prof.export_chrome_trace(...)`, which we cover in the export section. If you only want the printed table and no trace file, you can omit this argument entirely.

**`record_shapes=True`** makes the profiler record the input tensor shapes for every operator. This is enormously useful — it's how you discover that `aten::addmm` is being called with two different shapes (the attention projection and the MLP projection) that you'd otherwise see merged into one row — but it adds overhead, because the profiler has to introspect and store shape metadata on every dispatch. **`profile_memory=True`** records per-operator allocations and frees, so the table gains memory columns; indispensable for a leak hunt, modest overhead. **`with_stack=True`** captures the full Python call stack for each operator so you can map a kernel back to the exact line of your code — and it is the *expensive* one, capable of doubling your measured runtime, so you turn it on for a targeted question and off for routine measurement. We'll quantify that cost later, with a figure.

**`prof.step()`** inside the loop is not optional — it is the heartbeat that advances the schedule. Every call moves the state machine forward one step (wait → warmup → active → …). Forget it and the schedule never advances; the profiler sits in `wait` forever and records nothing. It goes at the *end* of each iteration, after the work for that step is done.

One more line deserves attention: **`torch.cuda.synchronize()`**. CUDA kernels launch asynchronously — the Python call returns immediately, and the kernel runs later on the device. If you don't synchronize, the profiler still captures the kernel (that's what the CUDA activity tracing is for), but your *loop boundary* becomes fuzzy: step N's kernels may still be running when step N+1 begins enqueuing. For a clean per-step attribution during profiling, a synchronize at the step boundary keeps the phases crisp. In production you would *never* add gratuitous syncs — they serialize CPU and GPU and destroy overlap — but inside a profiling harness, a per-step sync buys you clean, comparable steps. This is exactly the kind of measurement hygiene the [reproducible-benchmark post](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark) argues for at length; the profiler inherits all of it.

### What the activities can and cannot see

`ProfilerActivity.CPU` and `ProfilerActivity.CUDA` are the two you'll use on an NVIDIA box, and it's worth being precise about their reach. `CPU` traces PyTorch's operator dispatch on the host — it sees `aten` ops, `cudaLaunchKernel`, memory allocations, and any `record_function` span. It does *not* see arbitrary Python that isn't a PyTorch op: a slow pure-Python loop, a regex in your tokenizer, JSON parsing at the request boundary — those are invisible unless you wrap them in `record_function`, and even then you get a span duration, not a line-by-line breakdown. For that you drop to py-spy or cProfile, which the CPU-side track of this series covers. `CUDA` traces device-side kernel execution and memory copies through CUPTI; it sees every kernel, every H2D and D2H `memcpy`, and their true device durations. What neither activity sees is anything *outside* the process — other processes contending for the GPU, driver-level scheduling, NCCL traffic across ranks. When your bottleneck lives there, the profiler's clean table is a false all-clear, and you escalate to Nsight Systems, which watches the whole node. Knowing the boundary of what an instrument can observe is half of using it well: a tool that can't see your problem will always report that your problem isn't there.

### The schedule state machine, and why warmup exists

The schedule is worth deriving rather than memorizing, because once you see *why* each phase is there you'll never misconfigure it. A cycle is `wait` steps, then `warmup` steps, then `active` steps, and the whole cycle repeats `repeat` times. With `wait=1, warmup=1, active=3, repeat=1` you get a single cycle: skip one step, warm up one step, record three steps, done.

The `wait` phase does nothing — no recording, no tracing. Its purpose is to let you skip the genuinely cold iterations of a fresh process: the first CUDA call that initializes the context (which can cost hundreds of milliseconds), the first allocation that grows the caching allocator's pool. You don't want those in your trace at all, so you don't even record them.

The `warmup` phase is where the subtle cost lives, and it's the one people delete to "save time" and then wonder why their measurements are unstable. During warmup, the profiler is active but *discards* what it records. Why record-then-discard instead of just skipping? Because the act of profiling itself has a first-time cost — the tracer's own buffers get allocated, its callbacks get registered on the dispatcher — and you want that cost paid *before* the steps you keep. More importantly, warmup is when the real steady-state costs get paid on the device side: cuDNN and cuBLAS run their heuristics or autotuning to pick a kernel for each new shape; `torch.compile`, if present, triggers its compilation; the allocator settles into a stable set of blocks. A single warmup step amortizes all of that so the `active` steps measure the service as it runs *in production after warm-up*, not as it stutters through its first breath.

We can make the distortion precise. Say a steady-state step costs $t_\text{steady}$ and the first step additionally pays a one-time cost $c$ (autotuning, allocator growth, context init). If you profile $n$ steps with no warmup and naively average, you measure

$$\bar{t} = t_\text{steady} + \frac{c}{n}.$$

With $c$ on the order of tens of milliseconds and $t_\text{steady}$ a few milliseconds, even $n = 5$ leaves you reporting a per-step time inflated by several milliseconds — a bias larger than most optimizations you're trying to detect. The warmup phase drives the $c/n$ term to zero by ensuring $c$ is paid outside the measured window. That is the entire justification for the phase, and it's why the state machine exists at all rather than "record for a while and average."

The `active` count is a bias–variance trade. One active step is fast but noisy; a single unlucky step (a stray GC pause, a clock blip) skews everything. Twenty active steps average out the noise but bloat the trace and slow the run. Three to ten is the usual sweet spot for inference; for training, where a step is heavier, three to five is plenty. `repeat` lets you capture several cycles across a longer run — useful when you suspect periodic behavior, like a stall every N steps, which is exactly the [p99-tail](/blog/machine-learning/performance-engineering/metrics-that-actually-matter) pattern a single cycle might miss.

## How the profiler sees your program

Before we read a table, it helps to know where the numbers come from, because the two-worlds structure of the data explains every column you're about to see.

![a dataflow where a host thread feeds a launch queue that runs kernels on a device stream and both the host times and device times fan into the profiler which emits one merged table](/imgs/blogs/profiling-pytorch-with-torch-profiler-3.webp)

The diagram shows the two timing sources the profiler merges. On the host side, a CPU thread runs your Python, which dispatches `aten` operators, each of which (for a CUDA tensor) eventually calls `cudaLaunchKernel` to *enqueue* a kernel onto a CUDA stream. That enqueue is cheap-ish but not free — roughly 5–10 µs of CPU-side work per launch — and it returns immediately, before the kernel runs. The profiler timestamps these host events. Separately, on the device side, the CUDA activity API (CUPTI, the CUDA Profiling Tools Interface, which the profiler uses internally) reports when each kernel actually *executed* on the GPU, which happens asynchronously, some time after the launch. The profiler collects both streams of timestamps and merges them by time into one table and one timeline. That merge is why a single row can show both a CPU time and a CUDA time for the "same" operator: the CPU time is how long the host spent dispatching and launching it, and the CUDA time is how long the device spent running the resulting kernel. They are two different clocks measuring two different machines, stitched together.

This is also why the `~8 µs each` on the launch-queue node matters so much for AI services. That per-launch host cost is fixed regardless of how big the kernel is. A kernel that does 8 µs of device work and costs 8 µs to launch is 50% overhead. Fire ten thousand such tiny kernels per step — very common in an un-fused, un-graphed model — and the host simply cannot enqueue them fast enough to keep the device busy. The GPU drains its queue and idles, waiting for Python. That's the launch-bound waste, and the profiler is about to show it to you as a stark asymmetry between the CPU and CUDA totals.

### The launch-overhead law

We can make the launch-bound condition exact, because it explains why "add more GPU" so often fails. Let a step launch $N$ kernels, each costing $t_\text{launch}$ of host time to enqueue and running for an average $t_\text{kernel}$ on the device. The host must spend $t_\text{host} = N \cdot t_\text{launch}$ enqueuing, and the device has $t_\text{device} = N \cdot t_\text{kernel}$ of work. The step is launch-bound whenever the host cannot keep the queue full:

$$t_\text{host} > t_\text{device} \quad\Longleftrightarrow\quad N \cdot t_\text{launch} > N \cdot t_\text{kernel} \quad\Longleftrightarrow\quad t_\text{launch} > t_\text{kernel}.$$

The $N$ cancels — which is the surprising and important part. Whether you're launch-bound depends not on how many kernels you have but on whether the average kernel runs longer than it costs to launch. With a launch cost around 8 µs, any kernel that does less than 8 µs of device work is pure overhead: the GPU finishes it before the CPU can enqueue the next one, and the queue drains. The fraction of each step the device sits idle is

$$\text{idle fraction} = \frac{t_\text{host} - t_\text{device}}{t_\text{host}} = 1 - \frac{t_\text{kernel}}{t_\text{launch}} \quad (\text{valid when } t_\text{launch} > t_\text{kernel}).$$

Put the launch-bound table's numbers in: 1120 launches at ~8 µs is roughly 9 ms of pure host enqueue time, against 6.4 ms of device work, so the device idles a substantial slice of every step — which is exactly the 34% utilization the dashboard showed. This is why the fix for launch-bound services is always *fewer, bigger launches*: raise $t_\text{kernel}$ per launch by fusing several ops into one, or drop $N$ toward one with a CUDA graph. Both push $t_\text{launch}$ back below $t_\text{kernel}$ and let the queue refill faster than the device drains it. The profiler doesn't just tell you *that* you're launch-bound; the call count and the CPU/CUDA totals tell you *how far* below the break-even line each kernel sits.

### From your span down to a kernel

There's a second structural fact to absorb: the profiler's data is *hierarchical*. When you wrap code in `record_function("name")`, or when an `aten` operator calls sub-operators, the profiler nests them. Understanding this nesting is the key to reading the difference between "self" time and "total" time, which is the single most misread thing in the whole table.

![a vertical stack showing a named record function span expanding into aten operators then a dispatch to the cuda backend then the launched kernels then the measured device time](/imgs/blogs/profiling-pytorch-with-torch-profiler-4.webp)

Read the stack top to bottom. Your `record_function("inference_step")` span sits at the top — it is a label you placed. Inside it, PyTorch executes `aten` operators like `aten::linear`, which itself calls `aten::addmm`. Each of those dispatches to the CUDA backend, which launches one or more kernels. Those kernels, running on the device, are where the CUDA time is actually spent. So a single high-level span *contains* everything beneath it. The `total` CUDA time of your span is the time of every kernel launched anywhere inside it. The `self` CUDA time of the span is only the kernels launched *directly* by that node and not by its children — for a pure label span, that's usually zero, because the span itself launches nothing; its children do.

Formally, for any node in the tree,

$$t_\text{self}(op) = t_\text{total}(op) - \sum_{c \,\in\, \text{children}(op)} t_\text{total}(c).$$

Self time is total time minus the total time of your children. This has a beautiful consequence: while *total* times overlap (a parent's total includes its children's, so summing totals double-counts), *self* times partition the timeline cleanly. Summing self time over every operator gives you the real wall-clock device time with no double counting:

$$\sum_{op} t_\text{self}^\text{CUDA}(op) = t_\text{wall}^\text{device}.$$

That identity is why the profiler prints a `Self CUDA time total` footer and why you sort by self time when you want to find *which operator is actually eating the device*, but sort by total time when you want to find *which high-level region* to attack. Confuse the two and you'll "discover" that `inference_step` is 100% of your time — true, and useless, because it contains everything.

## Reading key_averages().table()

Now the payoff. `prof.key_averages()` aggregates every recorded event by operator (and, with `group_by_input_shape=True`, by shape), and `.table()` renders it. Here is a realistic printout from the toy transformer above — six layers, batch 1, sequence length 128, on an A100 — sorted by total CUDA time:

```console
-------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                                 Name    Self CPU %      Self CPU    Self CUDA %     Self CUDA    CUDA total    # of Calls
-------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                          aten::addmm        14.22%       3.544ms        21.11%       1.354ms       1.354ms           192
                       aten::_softmax         4.10%       1.022ms        18.63%       1.195ms       1.195ms            36
                            aten::gelu         3.02%       0.753ms         8.44%       0.541ms       0.541ms            36
              aten::native_layer_norm         6.85%       1.707ms         9.98%       0.640ms       1.021ms            72
                             aten::mul         9.71%       2.420ms         7.02%       0.450ms       0.450ms           260
                             aten::add         8.33%       2.077ms         6.55%       0.420ms       0.420ms           220
                           aten::copy_         7.44%       1.855ms         5.10%       0.327ms       0.327ms           148
                     cudaLaunchKernel        28.90%       7.204ms         0.00%       0.000us       0.000us          1120
-------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 24.918ms
Self CUDA time total: 6.412ms
```

Every number here is telling you something. Let me name the columns and what each reveals.

![a five row matrix mapping table columns like self cuda and cpu total and call count and shapes and memory to the bottleneck signature each one reveals](/imgs/blogs/profiling-pytorch-with-torch-profiler-5.webp)

- **Self CUDA / Self CUDA %** — device time spent *in this operator's own kernels*. This is your "which kernel is eating the GPU" column. Sort by it (`sort_by="self_cuda_time_total"`) and the top row is the kernel to optimize first. In the table, `aten::addmm` (the matrix multiplies) and `aten::_softmax` together are ~40% of device time.
- **CPU total / Self CPU** — host time. The `Self CPU %` column tells you where the CPU is spending its dispatch effort. The screaming signal here is `cudaLaunchKernel` at **28.90% of self CPU with 1120 calls and zero CUDA time**. That row is pure launch overhead — the CPU cost of firing kernels — and it is the largest single consumer of CPU time in the table.
- **# of Calls** — how many times the operator ran. 1120 launches for a single forward pass of a six-layer model is a lot of small kernels. High call counts on cheap ops are the fingerprint of a launch-bound service.
- **Input Shapes** — present only with `record_shapes=True`; the shapes that flowed through each op (shown in the next table). It's how you split one merged row into its real callers.
- **Self CUDA Mem / Self CPU Mem** — present only with `profile_memory=True`; bytes allocated. The column you live in during a [memory leak hunt](/blog/machine-learning/performance-engineering/metrics-that-actually-matter).

Now read the *footer*, because it's the fastest diagnosis in the whole tool. `Self CPU time total: 24.918ms` versus `Self CUDA time total: 6.412ms`. The host spent nearly four times as long dispatching work as the device spent doing it. On a well-fed GPU those numbers are close, or CUDA exceeds CPU. Here CPU dwarfs CUDA by 4x. That single ratio says: **this service is launch-bound.** The GPU is idle most of each step, waiting for Python to hand it the next kernel. No amount of a faster GPU fixes this; you need fewer, bigger launches — CUDA graphs, fusion, `torch.compile` — which is where the [next track](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu) of the series goes.

### Splitting a merged row by shape

The table above merged all 192 `aten::addmm` calls into one row, but those calls aren't the same operation — some are the attention query, key, and value projections, some are the wider MLP projections, and they have very different shapes and costs. `record_shapes=True` plus `group_by_input_shape=True` splits them apart:

```python
print(prof.key_averages(group_by_input_shape=True).table(
    sort_by="self_cuda_time_total", row_limit=10))
```

```console
-------------------------  ------------  ------------  -------------------------------------
                     Name     Self CUDA    # of Calls                             Input Shapes
-------------------------  ------------  ------------  -------------------------------------
             aten::addmm       0.902ms            36         [[128, 768], [768, 3072], [3072]]
             aten::addmm       0.452ms           144           [[128, 768], [768, 768], [768]]
          aten::_softmax       1.195ms            36                       [[1, 12, 128, 128]]
-------------------------  ------------  ------------  -------------------------------------
```

The one `addmm` row is now two: the 36 MLP projections at shape `[768, 3072]` cost more per call than the 144 attention projections at `[768, 768]`, even though there are fewer of them. That distinction matters when you decide what to fuse or which matmul to hand to a faster kernel. Shape splitting is also your early warning for a [recompilation storm](/blog/machine-learning/performance-engineering/metrics-that-actually-matter): if the *same* operator shows up under a dozen slightly different shapes, a downstream `torch.compile` will recompile for each one, and the shape column shows you that diversity before it becomes a wall.

### Driving key_averages: grouping and sorting

The one method takes a few arguments worth knowing, because each changes the question the table answers. `sort_by` accepts `"cuda_time_total"` (the hot region), `"self_cuda_time_total"` (the hot kernel), `"cpu_time_total"` (host-bound triage), `"self_cpu_time_total"`, and — with `profile_memory=True` — `"self_cuda_memory_usage"` (the biggest allocator). `group_by_input_shape=True` splits rows by shape as above. `group_by_stack_n=5`, combined with `with_stack=True`, groups by the top five Python stack frames, so you can attribute time to the exact *call site* in your handler rather than to a bare `aten` op. And `row_limit` caps the rows; start at 10 to 20 and raise it only when you're hunting something specific. The habit to build is reaching for a different `sort_by` per question instead of reading one sort and trying to answer everything from it.

### The three signatures

The reason the table is worth learning is that a small number of *shapes* in it correspond to a small number of *root causes*. Learn the three signatures and you can diagnose most services at a glance.

| Signature | What the table shows | Root cause | Where to fix it |
|---|---|---|---|
| **Launch-bound** (host-bound) | Self CPU total ≫ Self CUDA total; a fat `cudaLaunchKernel` row; huge `# of Calls` on cheap ops | Python can't enqueue kernels fast enough; GPU idles between tiny launches | CUDA graphs, `torch.compile(mode="reduce-overhead")`, fusion, bigger batch |
| **Kernel-bound** (compute-bound) | One or two operators dominate Self CUDA; CPU total ≈ or < CUDA total | A genuine hot kernel — a big matmul, an attention softmax — is the work | Optimize/replace the kernel; better algorithm; Nsight Compute on that kernel |
| **Memory-bound** (bandwidth wall) | Many elementwise ops (`mul`, `add`, `gelu`) with high Self CUDA relative to their FLOPs; high per-op memory traffic | Kernels stalled on HBM bandwidth, re-reading tensors that never fuse | Fusion (`torch.compile`), FlashAttention-style kernels, `channels_last` |

The launch-bound table above is signature one. Here's what signature two — a genuinely kernel-bound service — looks like. Same model architecture, but now serving long context (sequence length 4096), sorted by self CUDA time:

```console
-------------------------------  ------------  ------------  ------------  ------------
                           Name     Self CUDA    Self CUDA %    CUDA total    # of Calls
-------------------------------  ------------  ------------  ------------  ------------
                 aten::_softmax       18.42ms         42.05%       18.42ms            36
                       aten::bmm        9.88ms         22.56%        9.88ms            72
                     aten::addmm        7.31ms         16.69%        7.31ms           192
         aten::native_layer_norm        2.90ms          6.62%        4.10ms            72
                       aten::gelu        2.04ms          4.66%        2.04ms            36
                        aten::mul        1.55ms          3.54%        1.55ms           260
-------------------------------  ------------  ------------  ------------  ------------
Self CUDA time total: 43.80ms
```

Here the footer would show CPU total *below* CUDA total, and one operator — `aten::_softmax` — is **42% of all device time**. That is the "softmax 42%" fact from figure 2, and it is a completely different problem than the launch-bound table. Buying a bigger GPU actually might help here (it's real device work), but the smarter move is a fused attention kernel that never materializes the full attention matrix, which is precisely the [memory-wall fix](/blog/machine-learning/high-performance-computing/kernel-fusion-and-flashattention-beating-the-memory-wall) — because at 4096 tokens the softmax is memory-bound on that giant score matrix, not compute-bound. The profiler pointed at the operator; the roofline (from the [roofline post](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu)) tells you whether to attack it with a faster kernel or a fused one.

### From the table to the next step

You don't have to hold all this in your head. The table drives a short decision.

![a decision tree rooted at reading the table branching into three questions about host time a hot kernel and memory each leading to a named fix](/imgs/blogs/profiling-pytorch-with-torch-profiler-6.webp)

Read the tree as three questions asked of the table, and take the first yes. Is CPU total roughly a small multiple larger than CUDA total (say ~5x), with a fat launch row? You're launch-bound; the fix is CUDA graphs. Does a single kernel own more than ~40% of Self CUDA time? You're kernel-bound; profile that one kernel to the metal with Nsight Compute and either replace it or fuse it. Is memory traffic high while the arithmetic is trivial — lots of elementwise ops moving lots of bytes? You're memory-bound; fusion and better memory formats cut the HBM round-trips. This is the same symptom→cause→fix logic the [capstone playbook](/blog/machine-learning/performance-engineering/the-performance-engineering-playbook) generalizes across every tool in the series; the profiler table is where it starts, because it's the cheapest place to ask the question.

## record_function: making the trace speak your language

So far the table speaks `aten`. That's fine for spotting a hot kernel, but a real service has *phases* — preprocess, forward, postprocess, or embed, prefill, decode — and you want to know which phase owns the time, in your vocabulary, not PyTorch's. `record_function` names a span, and every operator that runs inside it gets attributed to that name in the trace and the table.

```python
from torch.profiler import record_function

@torch.no_grad()
def handle_request(model, raw_text):
    with record_function("preprocess"):
        tokens = tokenizer(raw_text, return_tensors="pt")
        tokens = {k: v.cuda(non_blocking=True) for k, v in tokens.items()}

    with record_function("forward"):
        logits = model(**tokens).logits

    with record_function("postprocess"):
        probs = logits.softmax(dim=-1)
        top = probs.topk(5, dim=-1)
        result = decode(top)          # CPU-side, back to Python objects

    return result
```

Now when you profile `handle_request`, the table and the timeline carry three labeled spans, and `key_averages()` will show `preprocess`, `forward`, and `postprocess` as top-level rows with their own CPU and CUDA totals. The first time you do this on a real handler the result is often a shock: the "model" you were sure was slow is 60% of the time, but `preprocess` — tokenization and an H2D copy you never thought about — is 30%, and `postprocess` (a `.topk` plus a Python decode loop that syncs the GPU) is the other 10%. You cannot see any of that from the raw `aten` table, because tokenization is CPU Python and the sync is hidden inside `.topk().tolist()`. `record_function` maps the trace back onto *your* code, and it's the difference between "the forward is 42% softmax" and "the request is 30% tokenizer, and we're paying for a bigger GPU to speed up a part that isn't even on the GPU."

A practical note: `record_function` has real but small overhead (it inserts an event pair on the dispatcher), so put spans around *phases*, not around every line. Half a dozen well-placed spans turn an inscrutable trace into a readable one; a hundred spans just add noise and overhead.

## Profiling training, not just inference

Everything so far profiled a forward pass, but the same instrument covers the whole training step, and the backward pass is where a lot of hidden time lives. Wrap forward, backward, and the optimizer in their own spans:

```python
prof_schedule = schedule(wait=1, warmup=1, active=3, repeat=1)
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=prof_schedule,
    record_shapes=True,
) as prof:
    for step, (x, y) in zip(range(5), loader):
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        with record_function("forward"):
            loss = criterion(model(x), y)
        with record_function("backward"):
            loss.backward()
        with record_function("optimizer"):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        prof.step()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
```

In the resulting table the backward pass shows up as operators suffixed `Backward` — `aten::AddmmBackward0`, `aten::NativeLayerNormBackward0`, and so on — alongside `autograd::engine::evaluate_function` rows that are the autograd engine's own dispatch overhead. Two things routinely surprise people the first time. First, the backward pass is usually *more* expensive than the forward, often close to 2x, because each forward operator generates two backward operators — a gradient with respect to its inputs and one with respect to its weights; if your forward is 6 ms of device work, budget roughly 12 ms for backward. Second, the `optimizer` span can be shockingly launch-bound: a plain Adam step launches a handful of tiny elementwise kernels *per parameter tensor*, so a model with hundreds of parameters fires hundreds of microscopic kernels every step — a textbook launch-bound signature that the fused and `foreach` optimizer implementations exist to kill. Profiling training is how you discover that the optimizer, not the model, is a third of your step time.

The `zero_grad(set_to_none=True)` above is a small, real optimization the profiler will reward you for noticing: setting gradients to `None` instead of zeroing them skips a per-parameter memset kernel, dropping the launch count. It's the kind of change you'd only think to make after the profiler showed you the memsets in the first place.

## Watching memory per operator

Flip on `profile_memory=True` and the table gains memory columns — `CPU Mem`, `Self CPU Mem`, `CUDA Mem`, `Self CUDA Mem`. Sort by `self_cuda_memory_usage` and you get a ranking of which operators *allocated* the most device memory, which is the first thing you want when a service creeps toward OOM or you're trying to shrink a batch's footprint.

```python
print(prof.key_averages().table(
    sort_by="self_cuda_memory_usage", row_limit=10))
```

```console
-------------------------  ------------  --------------  ------------  ------------
                     Name      CUDA Mem   Self CUDA Mem     Self CUDA    # of Calls
-------------------------  ------------  --------------  ------------  ------------
             aten::empty      412.00 Mb       412.00 Mb       0.000us           248
          aten::_softmax      288.00 Mb       288.00 Mb       1.195ms            36
               aten::bmm      196.00 Mb       196.00 Mb       9.880ms            72
             aten::addmm       96.00 Mb        96.00 Mb       1.354ms           192
-------------------------  ------------  --------------  ------------  ------------
```

Read the softmax row against the earlier table: 288 MB of device memory for the attention scores, and 42% of device time. That co-location — big memory *and* big time on the same operator — is the memory-bound fingerprint. The kernel is slow *because* it's shuttling that 288 MB to and from HBM, not because it's doing hard arithmetic, which is exactly what a fused attention kernel fixes by never materializing the full score matrix. The `aten::empty` row at the top is worth understanding too: it's allocation with zero CUDA time, so a large `empty` footprint points at big transient buffers — activations, workspace — that activation checkpointing or a smarter allocator could shrink. One caveat: these columns report allocator *activity* per operator, not a live snapshot of what's resident. For the slow-growing-leak hunt you graduate to `torch.cuda.memory._record_memory_history()` and the snapshot viewer, which the memory track of this series covers in depth.

## Exporting a Chrome trace

The table tells you *what* is slow. It cannot show you *when* — the gaps, the overlap, the exact moment the GPU goes idle waiting for a copy. For that you need the timeline, and the profiler exports one in the Chrome trace format that `chrome://tracing` and [Perfetto](https://ui.perfetto.dev) render as a scrubable, zoomable set of lanes.

```python
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
    record_shapes=True,
) as prof:
    for step in range(5):
        with record_function("inference_step"):
            model(inputs)
        torch.cuda.synchronize()
        prof.step()

# Write a Chrome/Perfetto-readable timeline.
prof.export_chrome_trace("trace.json")
```

Open `trace.json` in `chrome://tracing` (or drag it into Perfetto), and you get lanes: a CPU thread lane showing `aten` ops and `cudaLaunchKernel` calls, and a CUDA stream lane showing the kernels themselves, laid out on a shared time axis. This is where the launch-bound signature stops being a ratio and becomes *visible*: you'll see short kernels on the GPU lane with wide empty gaps between them, and above each gap, the CPU lane busy launching the next op. The GPU is waiting for Python. Reading that timeline well — spotting bubbles, tiny kernels, the launch-to-execute lag, sync stalls — deserves its own treatment, which is the very next post, [reading a Chrome trace](/blog/machine-learning/performance-engineering/reading-a-chrome-trace). For now, know that `export_chrome_trace` is the bridge from the table's *what* to the timeline's *when*.

There are two ways to get the trace file, and they compose with the schedule differently. `export_chrome_trace` called after the context closes writes whatever the profiler currently holds — simple, good for a quick look. The `on_trace_ready=tensorboard_trace_handler("./log")` callback, by contrast, fires automatically at the end of *each* schedule cycle and writes a file the TensorBoard PyTorch Profiler plugin ingests. The TensorBoard view is worth knowing because it does analysis for you: it renders the same timeline, but also computes a per-step breakdown (time in kernels vs. memcpy vs. runtime vs. idle), flags the "GPU Summary" with an estimated utilization and occupancy, and even suggests likely bottlenecks. For a first pass, TensorBoard's automated breakdown is the fastest way to see the CPU-vs-GPU split; for surgical timeline reading, the raw Chrome trace in Perfetto gives you full control.

## Overhead and honesty

A profiler is an observer, and observing perturbs. This is not a footnote — it's the difference between a measurement you can trust and a number that sends you chasing a ghost. The profiler adds work: it timestamps every operator, and with the richer knobs it introspects shapes, records allocations, and unwinds the Python stack on every dispatch. That work shows up as slowdown, and the slowdown is not uniform across the knobs.

![a three row matrix comparing record shapes and with stack and profile memory by what each adds and its overhead and when to enable it](/imgs/blogs/profiling-pytorch-with-torch-profiler-7.webp)

The matrix ranks the optional knobs by their cost. `record_shapes` is low-to-moderate overhead and usually worth leaving on — shapes are how you split merged rows and catch [recompilation](/blog/machine-learning/performance-engineering/metrics-that-actually-matter) risks. `profile_memory` is moderate and you turn it on for a memory question. `with_stack` is the expensive one: capturing and unwinding the Python call stack on every operator can *double* your runtime for a Python-heavy model, because the cost is proportional to how often you dispatch, and a launch-bound service dispatches constantly. The trap is subtle: `with_stack` inflates CPU time specifically, which can make a service look *more* launch-bound than it really is — the very signature you're trying to measure. So the rule is: turn `with_stack` on for one targeted deep-dive when you need to map a kernel to a source line, read your answer, and turn it back off before you measure steady-state times or compare before/after.

We can bound the distortion. Let $t_0$ be the true per-step time and let the profiler add a fixed overhead $\delta$ per recorded operator. If a step dispatches $N$ operators, the measured step time is

$$t_\text{measured} = t_0 + N\cdot\delta.$$

For a service that is *already* launch-bound — large $N$, small per-op work — the $N\cdot\delta$ term is proportionally largest, which is exactly the case where you must be most careful. The defenses are the ones baked into the schedule and into good benchmarking hygiene: profile a handful of steady steps, not the whole run; keep `with_stack` off unless you need it; measure your *actual* before/after latency with lightweight [CUDA events or wall-clock timing](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark) *outside* the profiler, and use the profiler only to find *what* to fix, not to report the final speedup number. The profiler tells you where the time goes; a clean benchmark tells you how much you saved. Don't cross the wires.

There's a related honesty point about *cold* iterations. Never profile the first few steps of a fresh process and report them as representative — that's what the `wait` and `warmup` phases exist to exclude. And never profile a single step: one step catches a stray GC pause or a clock transition and you'll chase noise. Three to five steady active steps, warmup paid, `with_stack` off for the timing run — that's an honest profile.

### Common ways the profiler misleads you

A handful of traps recur, and every one of them comes from forgetting that CUDA is asynchronous. **A kernel's device time can look attributed to the wrong CPU op** if you read the raw timeline naively — the launch and the execution sit on different clocks, and a kernel may run long after the `aten` call that launched it already returned. The table merges them correctly, but on the timeline you have to follow the launch-to-execute arrows, which is the [Chrome-trace reading](/blog/machine-learning/performance-engineering/reading-a-chrome-trace) skill. **A `synchronize` or a `.item()` / `.cpu()` / `.tolist()` call shows up as a huge CPU time** on whatever op happens to trigger it, because the CPU blocks there waiting for all prior async work to finish — the time isn't really *in* that op, it's the bill for everything launched before it coming due. When you see one innocuous-looking operator with an enormous Self CPU time, suspect a hidden sync. **Averaging across steps hides a periodic stall**: `key_averages` collapses your active steps into means, so a once-every-Nth-step GC pause or allocator resize gets smeared thin. When you're chasing a [p99 tail](/blog/machine-learning/performance-engineering/metrics-that-actually-matter), read the per-step timeline, not the averaged table. And **the profiler measures the process, not the pipeline**: if your dataloader starves the GPU, a profiler wrapped around the model shows a fast, healthy forward, because the stall is *outside* the profiled region, upstream in a worker process. A green profiler table next to a slow service means the bottleneck lives somewhere you aren't profiling — widen the window until it appears.

## Worked example: the service at 34% GPU utilization

Let me run the full loop end to end on a concrete case, because a profiler is only useful in the context of a decision.

#### Worked example: diagnosing a launch-bound inference endpoint

**Symptom.** A six-layer transformer classifier serves on a single A100 80GB. `nvidia-smi` shows GPU utilization bouncing around 34%, p50 latency 18.2 ms, throughput 440 req/s at batch 8. The previous owner's note says "GPU-bound, needs H100." On-demand A100 runs about \$2.20 per GPU-hour, so at 440 req/s that's roughly \$1.39 per million requests, and an H100 would nearly double the hourly rate.

**Profile.** Wrap the forward in the schedule from the top of this post, `with_stack=False` for a clean timing view, and print `key_averages().table(sort_by="cpu_time_total")`. The footer reads `Self CPU time total: 24.918ms` and `Self CUDA time total: 6.412ms`. The `cudaLaunchKernel` row is 28.9% of self CPU across 1120 calls. That is the launch-bound signature, unambiguously — the footer ratio is ~4x CPU over CUDA, and utilization at 34% is the GPU idling two-thirds of each step waiting for the host.

**Hypothesis.** The device does only 6.4 ms of real work per step, but the step takes ~18 ms because the CPU needs ~25 ms to enqueue 1120 tiny kernels and the GPU stalls in the gaps. This is not a hardware problem; it's a launch-count problem. An H100 would run the same 6.4 ms of work slightly faster and then idle *even more*, at *twice* the price — a strictly worse dollar-per-request.

**Fix.** Collapse the launches. `torch.compile(model, mode="reduce-overhead")` fuses elementwise ops and wraps the forward in CUDA graphs, turning ~1120 per-step launches into a single graph replay. (The mechanics of that fix are the [next track](/blog/machine-learning/performance-engineering/the-performance-engineering-playbook) of this series; here we only need to know the profiler pointed straight at it.)

**Re-measure**, on the same A100, with an honest out-of-profiler benchmark:

| Metric | Before (eager) | After (compile + CUDA graphs) |
|---|---|---|
| GPU utilization | 34% | 82% |
| Self CPU time / step | 24.9 ms | 3.1 ms |
| Self CUDA time / step | 6.4 ms | 5.9 ms |
| Kernel launches / step | 1120 | 1 (graph replay) |
| p50 latency | 18.2 ms | 7.4 ms |
| p99 latency | 41.0 ms | 9.1 ms |
| Throughput @ batch 8 | 440 req/s | 1080 req/s |
| Cost per million req | \$1.39 | \$0.57 |

The most important row is `Self CUDA time / step`: it barely moved, from 6.4 ms to 5.9 ms. The GPU was never the problem — the device work was almost unchanged. Everything we won came from the host side: CPU time collapsed from 24.9 ms to 3.1 ms, launches from 1120 to 1, and with the host no longer the bottleneck, utilization rose to 82%, latency more than halved, and throughput 2.4x'd — on the *same* card, at 40% of the cost per request. The H100 would have been a \$1.39-per-request answer to a \$0.57 software fix.

**Stress test.** Does the diagnosis hold as conditions change? At **batch 1**, the launch-bound problem is *worse* — device work per step shrinks while launch count stays constant, so the CPU-to-CUDA ratio widens and the fix helps even more. At **batch 64**, the device finally has enough work per launch that some steps become compute-bound; the graph still helps, but the marginal win shrinks — the profiler footer moves toward CPU ≈ CUDA. On an **L4** (much weaker device, ~242 fp16 TFLOP/s, 300 GB/s) the *device* work takes longer, so the same launch overhead is a smaller fraction of the step; still worth graphing, but the utilization jump is more modest. And if the model's shapes **vary per request**, CUDA graphs and `torch.compile` can thrash (recompilation, capture invalidation) — the fix that was free here becomes a [recompilation-storm](/blog/machine-learning/performance-engineering/metrics-that-actually-matter) risk, and you'd bucket shapes or use `dynamic=True`. The profiler is what tells you which regime you're in each time — you re-profile after every change, because the bottleneck *moves*.

#### Worked example: catching a phase you never suspected

**Symptom.** A text-classification service on an L4 has p50 latency of 11 ms and everyone blames "the model." A `torch.compile` pass on the model barely moves p50.

**Profile with `record_function`.** Wrap the handler in `preprocess` / `forward` / `postprocess` spans and read the table:

```console
------------------  ------------  ------------  ------------  ------------
              Name    Self CPU %     CPU total    CUDA total    # of Calls
------------------  ------------  ------------  ------------  ------------
       preprocess        31.4%        3.45ms        0.28ms             5
          forward        44.1%        4.85ms        4.60ms            96
      postprocess        20.2%        2.22ms        0.05ms             8
------------------  ------------  ------------  ------------  ------------
```

**Read it.** `forward` is indeed the largest single span, but it's only 44% — and `preprocess` is 31% with almost *no* CUDA time (0.28 ms), meaning it's nearly all CPU: tokenization plus a blocking H2D copy. `postprocess` is 20%, also CPU-bound — a `.topk().tolist()` that synchronizes the device to pull results back to Python. Compiling the model optimized the 44% that was already fast per byte and left 51% of the request untouched. **The fix** isn't a better model; it's overlapping tokenization with the previous request's forward, making the H2D copy `non_blocking=True` on pinned memory, and deferring the CPU-side decode. The lesson: without `record_function`, "the request is slow" collapses onto "the model," and you optimize the wrong half. The named spans are what revealed that most of the latency wasn't on the GPU at all — a pattern the [CPU-bottleneck track](/blog/machine-learning/performance-engineering/the-performance-engineering-playbook) of the series is built around.

## Case studies and real numbers

A few published, checkable results that show the profiler-then-fix loop paying off, so you can calibrate what a win looks like.

**CUDA graphs / `reduce-overhead` on launch-bound models.** PyTorch's own documentation and the CUDA Graphs blog post report that wrapping a launch-bound forward in CUDA graphs (which `torch.compile(mode="reduce-overhead")` does automatically) removes essentially all per-kernel launch overhead, with the largest wins on models built from many small kernels — exactly the launch-bound signature the profiler footer reveals. The mechanism is precisely what the table predicts: when Self CPU ≫ Self CUDA, collapsing launches recovers the idle device time. Verify the magnitude for *your* model with the before/after table above rather than trusting a headline multiplier; the win scales with how launch-bound you started.

**`torch.compile` Inductor fusion.** The Inductor backend fuses elementwise chains (the `mul`/`add`/`gelu` rows that clutter the memory-bound table) into single kernels, cutting both launch count and HBM round-trips. The PyTorch team's compile benchmarks report meaningful speedups across large model suites; the profiler is how you *confirm* fusion happened — you re-profile and watch the elementwise rows collapse and the kernel count drop. If the count doesn't drop, a graph break (from the [graph-break debugging](/blog/machine-learning/performance-engineering/the-performance-engineering-playbook) work) prevented the fusion, and the trace will show it.

**Nsight-guided kernel work.** When the profiler says "kernel-bound — one operator owns the device," the next tool is Nsight Compute, which reports occupancy, memory throughput, and warp-stall reasons for that single kernel. The [model-serving Nsight post](/blog/machine-learning/model-serving/profiling-llm-serving-with-nsight) and the [HPC bottleneck-finding post](/blog/machine-learning/high-performance-computing/profiling-gpu-workloads-finding-the-real-bottleneck) walk a hot kernel from a Speed-of-Light readout to a concrete fix. The point for *this* post is that `torch.profiler` is what tells you *which* kernel to hand to Nsight — you never `ncu` a whole model, you `ncu` the one operator the averages table indicted.

**Channels-last for vision.** For convolutional services, switching to `channels_last` memory format lets cuDNN pick faster kernels and cuts bandwidth; PyTorch's channels-last tutorial documents real speedups on ResNet-style models. The profiler shows the win as the conv rows' Self CUDA time dropping and the kernel *names* changing (to the NHWC-optimized variants) — a change you'd never notice without reading the per-operator table before and after.

## When to reach for torch.profiler (and when not)

The profiler is the right first instrument almost always, but it is not the *only* instrument, and knowing its edges keeps you from misusing it.

**Reach for it first** whenever you don't yet know *what* is slow. It's cheap, it's already installed, and its CPU-vs-CUDA split answers the single most important triage question — host-bound or device-bound — in one footer. Any performance investigation on a PyTorch service should start here before you touch anything heavier.

**Reach past it to Nsight Systems** when the bottleneck is *system-wide* rather than inside one process — when you need to see NCCL communication overlapping (or not) with compute across ranks, or CPU threads and CUDA streams and memcpy engines all on one timeline, or activity outside PyTorch entirely. `torch.profiler` sees PyTorch's view of the world; [Nsight Systems](/blog/machine-learning/performance-engineering/nsight-systems-for-ai-services) sees the whole machine. **Reach past it to Nsight Compute** when the profiler has already told you *which* kernel is hot and you need to know *why* that specific kernel is slow at the warp level — occupancy, memory throughput, stall reasons. The profiler finds the kernel; Nsight Compute dissects it.

**Don't reach for `with_stack` casually.** It can double your measured time and, worse, distort the very CPU/CUDA ratio you're diagnosing. Turn it on for a single "map this kernel to a source line" question, then off. **Don't report the profiler's own timings as your speedup** — the profiler perturbs; measure the final before/after with a clean, out-of-profiler benchmark. **Don't profile cold steps** and call them representative; the schedule's `wait`/`warmup` phases exist precisely to stop you. And **don't profile the whole run** — a few steady steps tell you everything a million steps would, at a fraction of the trace size and the perturbation.

## The profiling loop in one page

Pulling it together, here's the routine you run every time a service feels slow, and it's the same four beats no matter the model. **One: instrument.** Wrap the hot path in `profile(...)` with the `wait`/`warmup`/`active` schedule, `record_shapes=True`, `with_stack=False` for now, and drop a few `record_function` spans around your real phases. **Two: read the footer, then the table.** Self CPU total versus Self CUDA total tells you host-bound or device-bound in one glance; then sort by self CUDA time for the hot kernel, or by CPU time for the launch overhead. **Three: name the signature.** A fat `cudaLaunchKernel` row and huge call counts mean launch-bound; one operator owning the device means kernel-bound; elementwise ops moving big buffers mean memory-bound. **Four: form one hypothesis, apply one fix, and re-profile.** Not three fixes — one, so you can attribute the change to it. Then run it again, because killing one bottleneck exposes the next, and the table you read after the fix is a different table than the one before it.

Everything downstream in this series — CUDA graphs, `torch.compile`, memory tuning, CPU work, custom kernels — is a *fix* selected by this loop. The profiler is not one technique among many; it's the instrument that decides which technique you need, and the discipline of always measuring before and after is what separates engineering from guessing. When you can drive `torch.profiler` cold — schedule, table, spans, export — you can start any performance investigation from a fact instead of a theory, which is the whole premise of the [playbook](/blog/machine-learning/performance-engineering/the-performance-engineering-playbook).

## Key takeaways

- The profiler exists to replace *slow somewhere* with a named operator and a number. Take the measurement before you place the bet.
- The `schedule(wait, warmup, active, repeat)` is not optional ceremony. `warmup` pays autotuning, allocator growth, and context init so your `active` steps measure steady state; skipping it biases every average by the one-time cost divided by your step count.
- Read the footer first: `Self CPU time total` versus `Self CUDA time total`. CPU ≫ CUDA means launch-bound; the fix is fewer launches, not a bigger GPU.
- Sort by **self** CUDA time to find the hot *kernel*; sort by **total** time to find the hot *region*. Self times partition the timeline; total times overlap. Confusing them "proves" your top span is 100% of the time.
- Three table signatures cover most services: launch-bound (fat `cudaLaunchKernel`, huge call counts), kernel-bound (one operator owns the device), memory-bound (elementwise ops moving many bytes). The shape of the table names the waste.
- `record_function` maps the trace onto *your* phases. It's how you discover that 30% of a "model" latency is actually the tokenizer and an H2D copy.
- `export_chrome_trace` turns the *what* into the *when* — open it in Perfetto to see the idle gaps the table only implies.
- `with_stack` is the expensive knob and it inflates CPU time specifically. Use it for one targeted question, then turn it off before you measure.
- Re-profile after every fix. The bottleneck moves — kill the launch overhead and the next step may be memory-bound. The loop is profile → hypothesize → fix → **re-profile**.

## Further reading

- [PyTorch Profiler tutorial](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) — the primary source for `torch.profiler`, the schedule, and `key_averages`.
- [PyTorch Profiler with TensorBoard](https://docs.pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html) — the automated GPU-summary and per-step breakdown view.
- [Perfetto UI](https://ui.perfetto.dev) and `chrome://tracing` — where an exported Chrome trace becomes a scrubable timeline.
- [Why your AI service wastes CPU and GPU](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu) — the series intro: the four wastes and the profile → hypothesize → fix → measure loop.
- [Reading a Chrome trace](/blog/machine-learning/performance-engineering/reading-a-chrome-trace) — the next post: turning the exported timeline into a diagnosis of gaps, bubbles, and sync stalls.
- [Setting up a reproducible benchmark](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark) — how to measure the before/after honestly, outside the profiler.
- [Metrics that actually matter](/blog/machine-learning/performance-engineering/metrics-that-actually-matter) — why utilization lies and which numbers to trust when scoring a fix.
- [Nsight Systems for AI services](/blog/machine-learning/performance-engineering/nsight-systems-for-ai-services) — the system-wide timeline for when the profiler's PyTorch-only view isn't enough.
- [Profiling GPU workloads: finding the real bottleneck](/blog/machine-learning/high-performance-computing/profiling-gpu-workloads-finding-the-real-bottleneck) — the HPC companion on taking one indicted kernel down to the metal.
- [The performance engineering playbook](/blog/machine-learning/performance-engineering/the-performance-engineering-playbook) — the capstone: the full symptom → tool → cause → fix decision tree this table feeds into.
