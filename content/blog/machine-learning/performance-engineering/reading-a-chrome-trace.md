---
title: "Reading a Chrome Trace: Spotting Bubbles, Tiny Kernels, and Sync Stalls on the Timeline"
date: "2026-07-14"
publishDate: "2026-07-14"
description: "Your torch.profiler export is 50,000 events on a timeline. This teaches you to read it — to see host-bound gaps, launch-bound picket-fences, and forced-sync stalls at a glance, and to point each shape at its fix."
tags:
  [
    "performance-engineering",
    "gpu-optimization",
    "profiling",
    "pytorch",
    "cuda",
    "cuda-graphs",
    "latency",
    "inference",
    "ml-systems",
  ]
category: "machine-learning"
subcategory: "Performance Engineering"
author: "Hiep Tran"
featured: true
readTime: 41
---

You captured a profile. You ran the service under `torch.profiler`, called `prof.export_chrome_trace("trace.json")`, dragged the file into the Perfetto UI, and now you are staring at a wall of colored rectangles. There are 50,000 of them. Some rows are dense with slivers too thin to read; others have long solid bars; the whole thing scrolls for what feels like a kilometer of timeline. Somewhere in there is the reason your service runs at 45% GPU activity while `nvidia-smi` cheerfully reports "GPU-Util 100%". The capture was the easy part. The skill — the thing that separates an engineer who *has* a trace from one who can *fix* the service — is knowing where to look.

Here is the good news that this entire post rests on: you do not have to read all 50,000 events. Almost every performance problem you will ever chase shows up as one of **three shapes** on the timeline, and once you can recognize the three shapes on sight, a trace stops being a wall of rectangles and becomes a diagnosis. **Gaps** on the GPU row mean the GPU is idle and you are host-bound. A **picket-fence** of tiny back-to-back kernels means launch overhead is eating you even though the row looks busy. A **long solid bar** where the CPU sits blocked means a forced synchronization or a device-to-host copy stalled the whole pipeline. Three shapes, three causes, three families of fix. That is the reading skill, and it is a visual skill — the whole point of this post is to teach your eye.

![a gpu timeline where an input copy overlaps the first kernel and three kernels run back to back with no idle gaps](/imgs/blogs/reading-a-chrome-trace-1.webp)

Start by looking at the figure above, because it is what *healthy* looks like — the baseline your eye needs so it can recognize sick. The GPU row is packed solid: an input copy on the copy engine overlaps the first kernel, then three compute kernels run back-to-back with no whitespace between them. No gaps, no picket-fence, no long stall. When your trace looks like this, the GPU is the bottleneck and you are getting your money's worth. Every diagnostic below is really a way of measuring *how far your trace is from this picture* — and every fix is a way of pushing it closer.

This post is the reading companion to its sibling, [profiling PyTorch with torch.profiler](/blog/machine-learning/performance-engineering/profiling-pytorch-with-torch-profiler), which covers *capturing* the trace — the schedule, `record_function`, `key_averages()`, and the export call. Assume you have a `trace.json` in hand. Here we cover *interpreting* it. If you have not yet met the producer/consumer model of a GPU service — CPU enqueues, GPU drains, asynchronously — read [the mental model of a GPU service](/blog/machine-learning/performance-engineering/the-mental-model-of-a-gpu-service) first; this post assumes it and shows you what that model looks like when it is drawn out in 50,000 events. And it sits inside the [Profiling & Optimizing AI Services](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu) series, whose whole loop is: profile, read the trace, hypothesize the cause, apply one fix, re-measure. This is the *read the trace* step, drawn.

## The shape of waste: sick versus healthy

Before we name the rows and the events, calibrate your eye on the contrast, because the contrast is the entire diagnostic. Below are the same model — a small Transformer encoder, batch 1, sequence length 128, fp16, on an A100 80GB — profiled two ways. On the left, the service as first shipped: eager PyTorch, one kernel launched per operation. On the right, the same forward pass after we collapsed the launches into a single CUDA-graph replay. Same math, same output, same hardware.

![two gpu timelines side by side where the left row is broken by wide idle gaps and the right row is packed solid with kernels](/imgs/blogs/reading-a-chrome-trace-2.webp)

The left timeline is the sick one. The GPU row is more whitespace than color — the kernels are real, but between every kernel there is a gap where the GPU has finished its work and is waiting for the host to launch the next thing. Add up the gaps and the GPU is *idle 55% of the step*. The right timeline is the same work with the gaps closed: the kernels are packed shoulder-to-shoulder, idle time falls to 8%, and the step that took 9.0 ms now takes 4.3 ms. The wall-clock roughly halved and the only thing that changed was *how the work was launched*, not the work itself.

This is the single most important thing to internalize about trace reading: **the waste is in the whitespace, not the color.** A beginner looks at the sick timeline and sees kernels running and concludes the GPU is busy. A trace reader looks at the same timeline and sees the *gaps* — and knows the GPU spent more of the step doing nothing than doing math. That inversion of attention, from the boxes to the spaces between them, is the skill. Everything below is a way of putting numbers on the whitespace.

One caution before we go on, because it trips up everyone once: `nvidia-smi` reported "GPU-Util 100%" for *both* of these runs. The util counter increments whenever *at least one* kernel was resident during the sampling window — it says nothing about how densely the row is packed. The sick run at 45% real activity and the healthy run at 92% activity both read as "100%" on `nvidia-smi`. The trace is the only tool that tells you the truth about the whitespace. (The [metrics-that-actually-matter post](/blog/machine-learning/performance-engineering/metrics-that-actually-matter) dissects why the util counter lies; here we just take it as given and go to the trace.)

## The timeline anatomy: what the rows mean

Now let us name what you are looking at. A `torch.profiler` Chrome trace (and an Nsight Systems timeline, which is the same idea with more rows) is organized into horizontal **lanes** stacked top to bottom, with time flowing left to right. Each lane is a thread or a device engine, and each colored rectangle is one **event** with a start time and a duration. The lanes you care about, from top to bottom, are these four.

![a vertical stack of four profiler rows from the python annotation lane down through the launch api lane the gpu kernel lane and the copy engine lane](/imgs/blogs/reading-a-chrome-trace-3.webp)

- **The Python / annotation lane (top).** This is your CPU thread as it walks the model's `forward`. If you wrapped regions in `record_function("attention")` or pushed NVTX ranges, they show up here as labeled bars — your map back to your own code. This is the lane that tells you *which part of your handler* a chunk of the timeline belongs to.
- **The CUDA runtime API lane.** Still on the CPU, but one level down: this is where each `cudaLaunchKernel`, `cudaMemcpyAsync`, and `cudaStreamSynchronize` call appears. Every launch is a small bar here — this lane is where *launch overhead* becomes visible, because if it is a dense picket-fence of `cudaLaunchKernel` bars, your CPU is spending all its time launching.
- **The GPU stream lane.** This is the device. The bars here are the *actual kernels executing* on the streaming multiprocessors (SMs). This is the lane you read for gaps: whitespace here is the GPU idle. This is the row that matters most.
- **The copy engine lane.** The GPU's DMA engines moving bytes host-to-device (H2D) and device-to-host (D2H) over PCIe. Whether these bars *overlap* the kernel bars or *serialize* before them is the difference between a hidden copy and a copy tax.

The mental model to carry — and this figure is drawn precisely so you can hold it in your head — is that **cause lives on the top rows and effect lives on the bottom rows.** The GPU row shows you the symptom (a gap). The CPU rows above it show you the cause (a slow launch, a blocking sync). You diagnose by reading *down* from a symptom on the GPU row to the CPU event that is directly above the gap that caused it.

### The launch-to-execute lag

There is one piece of timeline geometry that confuses everyone the first time, and understanding it is what makes the top-row-to-bottom-row reading work: a `cudaLaunchKernel` on the CPU row does **not** line up vertically with its kernel on the GPU row. The launch happens *earlier* in time; the kernel executes *later*. There is a horizontal lag between them.

![a branching diagram where a single launch call forks into a stream queue and a return to python then the queue either feeds a running kernel or drains to an idle gap](/imgs/blogs/reading-a-chrome-trace-4.webp)

The figure traces why. When the CPU calls `cudaLaunchKernel`, two things happen and then the call returns. The launch is *enqueued* onto the CUDA stream — an ordered FIFO of pending work — and the CPU immediately returns to Python to launch the *next* op. It does not wait. Meanwhile the GPU is draining that same queue from the front, one kernel at a time. So at any instant the stream holds a backlog of launched-but-not-yet-executed kernels, and the horizontal distance on the trace between a launch and its kernel is exactly *how deep that backlog is in time*.

This lag is not a bug — it is the async pipeline working as designed, and it is *good*. A healthy backlog is what keeps the GPU fed: as long as the queue always has the next kernel ready, the GPU never waits. The trouble starts when the backlog runs dry. If the CPU cannot enqueue fast enough — because each op costs microseconds of Python and dispatch and driver work — the GPU drains the queue to empty, finishes the last kernel, and then *sits idle* until the CPU finally gets the next launch in. That idle stretch is the gap. So the launch-to-execute lag has a diagnostic reading:

- **Lag large and stable** → the queue stays deep → the GPU always has work → you are GPU-bound (healthy).
- **Lag shrinks to zero right before a gap** → the queue drained empty → the CPU fell behind → you are host-bound (the gap is the proof).

Hold that in mind, because it is the mechanism behind shape number one.

## The three diagnostic shapes

Here is the framework, and it is almost the whole post. Nearly every trace you will read reduces to one (or a mix) of three shapes on the GPU row. Learn to name them and you can diagnose a service in the first ten seconds of opening its trace.

![a table matching three timeline shapes to what you see the cause and the fix for each one](/imgs/blogs/reading-a-chrome-trace-5.webp)

| Shape on the GPU row | What you see | The cause | The fix |
|---|---|---|---|
| **Gaps** | Wide whitespace between kernels; row is >30% empty | Host-bound: the CPU can't refill the launch queue fast enough | Fewer, bigger launches: CUDA graphs, `torch.compile`, bigger batch |
| **Picket-fence** | Row looks busy but is hundreds of 3–8 µs kernels back-to-back | Launch-bound: per-launch overhead dominates real compute | Fuse the tiny ops: `torch.compile` (Inductor fusion), CUDA graphs |
| **Long sync bar** | One long solid bar; CPU thread blocked next to it | A forced `cudaStreamSynchronize` / `.item()` / D2H copy | Remove the sync, defer `.item()`, pin memory + `non_blocking=True` |

Each shape gets its own section below, with the numeric read that confirms it and the fix pointer. But keep the table in view — it is the decision procedure, and the rest is elaboration.

### Shape 1: gaps mean host-bound

A gap is whitespace on the GPU row: an interval where no kernel is executing. One gap is nothing — there is always a little slack. The diagnosis is the *fraction* of the step that is gap. So the first thing you compute on any trace is the **gap fraction**: of the total wall span, how much was the GPU idle?

The mechanism is exactly the queue-draining story from the launch-to-execute lag. Let a step have $N$ kernels, each taking on average $\bar{t}_k$ on the GPU, so the real GPU work is $T_\text{gpu} = N \cdot \bar{t}_k$. Each of those $N$ kernels cost the CPU roughly $t_\text{launch}$ to enqueue (dispatch through ATen, package the launch, hand it to the driver — call it 5–8 µs on a modern host). So the CPU needs $T_\text{cpu} = N \cdot t_\text{launch}$ to launch the whole step. When these pipeline, the wall-clock is approximately the *larger* of the two:

$$T_\text{step} \approx \max\left(N \cdot t_\text{launch},\; N \cdot \bar{t}_k\right)$$

You are **host-bound whenever $N \cdot t_\text{launch} > N \cdot \bar{t}_k$**, i.e. whenever the average launch cost exceeds the average kernel duration: $t_\text{launch} > \bar{t}_k$. And the gap fraction is what is left over after the GPU finishes its share of the wall:

$$\text{gap fraction} = 1 - \frac{\sum_i d_i}{T_\text{span}} = 1 - \frac{T_\text{gpu}}{T_\text{step}}$$

where $d_i$ is kernel $i$'s duration on the GPU row and $T_\text{span}$ is the wall span of the step. This is not a metaphor — it is a number you read directly off the trace by summing the kernel durations on the GPU stream lane and dividing by the step's span. If it comes out above ~0.3, you are host-bound and the gaps are your problem.

You do not have to sum it by eye. The Chrome trace is JSON, and every event carries a `dur`. Before the reader, look at what the raw events actually are — three of them, pulled straight from a `trace.json`, showing a launch, its kernel, and a copy:

```json
[
  { "name": "cudaLaunchKernel", "ph": "X", "ts": 1024300, "dur": 6,
    "pid": 1234, "tid": 1234, "cat": "cuda_runtime",
    "args": { "correlation": 55123 } },

  { "name": "sm80_xmma_gemm_f16f16_128x128_ldg8_f2f_tn", "ph": "X",
    "ts": 1024341, "dur": 41, "pid": 0, "tid": 7, "cat": "kernel",
    "args": { "stream": 7, "correlation": 55123, "grid": [64,1,1] } },

  { "name": "Memcpy HtoD (Pinned -> Device)", "ph": "X", "ts": 1024260,
    "dur": 1602, "pid": 0, "tid": 8, "cat": "gpu_memcpy",
    "args": { "bytes": 19267584 } }
]
```

Three things to read off this. `ts` and `dur` are in microseconds — the kernel starts at 1024341 and runs 41 µs. The `pid`/`tid` place the event on a lane: `tid: 1234` is a CPU thread (the launch), `tid: 7` is a GPU stream (the kernel), `tid: 8` is the copy engine. And the shared `"correlation": 55123` is the thread that stitches the launch to its kernel — the launch at ts 1024300 and the kernel at ts 1024341 are the *same op*, and the 41 µs between them is the launch-to-execute lag, quantified. Now the reader is trivial:

```python
import json

# torch.profiler's export_chrome_trace writes the Chrome Trace Event format:
# a list of events, each {name, ph, ts (microseconds), dur (microseconds),
# pid, tid, args:{...}}. GPU-stream kernels live on a device tid; CPU ops
# and cudaLaunchKernel live on host tids.
with open("trace.json") as f:
    trace = json.load(f)
events = trace["traceEvents"] if isinstance(trace, dict) else trace

# 1) find the GPU stream track: torch.profiler tags device work with a
#    "stream" arg or a "cat" of "kernel"/"gpu_memcpy". Adjust the predicate
#    to your export; here we take complete ("ph":"X") events categorized
#    as kernels.
def is_gpu_kernel(e):
    return e.get("ph") == "X" and e.get("cat") in ("kernel", "Kernel")

kernels = [e for e in events if is_gpu_kernel(e)]
kernels.sort(key=lambda e: e["ts"])

busy_us = sum(e["dur"] for e in kernels)                 # sum of kernel durations
span_us = (kernels[-1]["ts"] + kernels[-1]["dur"]) - kernels[0]["ts"]
gap_fraction = 1.0 - busy_us / span_us

print(f"kernels on GPU row : {len(kernels)}")
print(f"GPU busy           : {busy_us/1e3:.2f} ms")
print(f"wall span          : {span_us/1e3:.2f} ms")
print(f"gap fraction       : {gap_fraction:.1%}")
```

Run it on the sick trace from the contrast figure and it prints something like this:

```console
kernels on GPU row : 1500
GPU busy           : 4.02 ms
wall span          : 9.01 ms
gap fraction       : 55.4%
```

There it is, quantified: 1,500 kernels, 4 ms of actual GPU work, spread across a 9 ms step. The GPU was idle 55% of the time. That is the definition of host-bound, and no amount of "GPU-Util 100%" from `nvidia-smi` changes it. The fix for gaps is always the same family — **make fewer, larger launches** so the CPU has less per-step work to do — which in practice means CUDA graphs (collapse the whole step to one replay) or `torch.compile` (fuse and reduce Python overhead). We link the mechanism of that fix to the [kernel-launch-overhead problem](/blog/machine-learning/performance-engineering/the-kernel-launch-overhead-problem); here the trace's job was just to *prove* you are host-bound before you spend a day on graphs.

#### Worked example: reading the gap fraction on an A100

Take the running Transformer encoder, batch 1, seq 128, fp16, A100 80GB. You open the trace, zoom the GPU stream lane to one step, and read:

- The GPU row has ~1,500 kernels, average duration ~2.7 µs, longest ~40 µs (a couple of matmuls).
- Summed kernel duration: 4.0 ms. Step wall span: 9.0 ms. Gap fraction: **55%**.
- On the CUDA runtime API lane directly above, the `cudaLaunchKernel` bars are a nearly-solid picket-fence — the CPU is launching almost continuously. Mean launch cost ~6 µs; 1,500 × 6 µs = 9.0 ms of CPU launch work.

Now the arithmetic clicks together: the CPU needs 9.0 ms to launch, the GPU needs 4.0 ms to run, they pipeline, and $\max(9.0, 4.0) = 9.0$ ms is the wall. The CPU is the pacemaker. The GPU finishes its 4 ms of real work and spends the other 5 ms waiting for launches. The gap fraction (55%) and the launch-bound test ($t_\text{launch} = 6\ \mu s > \bar{t}_k = 2.7\ \mu s$) agree: **host-bound.** Reach for CUDA graphs. When we do, the after-trace reads 1 launch, 4.3 ms wall, 8% gap — the right half of the contrast figure.

### Shape 2: a picket-fence means launch overhead

Shape one is easy because the whitespace is obvious. Shape two is the sneaky one, because the GPU row looks *busy* — there is barely any whitespace — and yet the service is slow. The tell is that the kernels are *tiny and identical-looking*, hundreds of them, each 3–8 µs, packed into a fence of thin slivers. This is the **launch-bound** case, and it is different from pure host-bound in an important way: here the GPU is not idle waiting, it is genuinely running kernels back-to-back — but each kernel does so little work that the *fixed cost* around it (launch, scheduling, the tiny prologue/epilogue) dominates the useful compute inside it.

The mechanism is a ratio. A kernel has a fixed overhead $t_\text{launch}$ and a useful compute time $\bar{t}_k$. The fraction of the row that is *real work* is:

$$\text{useful fraction} = \frac{\bar{t}_k}{\bar{t}_k + t_\text{launch}}$$

For a fat matmul where $\bar{t}_k = 400\ \mu s$ and $t_\text{launch} = 6\ \mu s$, the useful fraction is 98% — overhead is negligible. But for an elementwise `add` or a decomposed `LayerNorm` sub-op where $\bar{t}_k = 4\ \mu s$, the useful fraction is $4 / (4 + 6) = 40\%$ — you are paying more to *launch* the kernel than to *run* it. Six hundred such kernels and you have burned a third of your step on launch overhead for kernels that touch a few megabytes of memory each.

The other read of the same phenomenon is **kernels per second**. There is a hard ceiling on how many kernels a single CPU thread can launch, set by $1 / t_\text{launch}$. At $t_\text{launch} = 8\ \mu s$ that ceiling is about 125,000 kernels per second. If your trace shows a step running kernels at anywhere near that rate — say you count 1,500 kernels in a 12 ms step, which is 125,000/s — you have hit the launch ceiling and the CPU literally cannot go faster no matter how fast the GPU is. That is the picket-fence signature in one number.

The fix for a picket-fence is **fusion**: collapse the many tiny kernels into few big ones so the overhead is amortized. `torch.compile`'s Inductor backend does exactly this — it fuses chains of elementwise ops (a `LayerNorm` + `GELU` + residual `add` becomes one or two generated Triton kernels instead of a dozen library calls). CUDA graphs help too, but differently: graphs remove the *launch* cost without changing the *kernel count*, so a graph turns a picket-fence of 600 tiny kernels into 600 tiny kernels launched for free — good, but fusion is better because it also cuts the memory round-trips. In practice `torch.compile(mode="reduce-overhead")` does both (compile-time fusion plus CUDA graphs). The discipline the [torch.profiler post](/blog/machine-learning/performance-engineering/profiling-pytorch-with-torch-profiler) sets up applies directly here: after you compile, re-open the trace and *count the kernels* — if the fence is gone, the fusion happened; if it is still there, a graph break stopped it.

#### Worked example: the LayerNorm picket-fence

In the same Transformer, zoom into one encoder layer's normalization region using the `record_function("layernorm")` bar on the top lane as your anchor. Below it on the GPU row you find not one kernel but a fence of eight: a mean reduction, a subtract, a square, a variance reduction, an rsqrt, a multiply, a scale, a shift — the decomposed `LayerNorm`, each 3–4 µs. Across 12 layers that is ~96 tiny kernels just for normalization, plus the GELUs and residual adds: call it 300 elementwise kernels in the step, ~1.1 ms of GPU time but ~1.8 ms of CPU launch time.

Apply `torch.compile`. Re-capture, re-open, re-count: the eight-kernel fence per LayerNorm collapses to **one fused Triton kernel**; the 300 elementwise kernels become ~40 fused kernels. The GPU-time for that region drops from 1.1 ms to 0.7 ms (fewer HBM round-trips) and the launch time from 1.8 ms to 0.25 ms. The picket-fence is visibly gone from the trace — the region that was a comb of slivers is now a handful of solid bars. That before/after is the proof the fusion fired; the number is the win.

### Shape 3: a long sync bar means a forced stall

The third shape is the most dramatic and the easiest to miss the *cause* of. You see it as a single **long solid bar** — often on the CPU thread, sometimes as a suspiciously long "kernel-shaped" region — where everything just stops. The GPU row goes quiet, the Python lane shows one fat bar named `cudaStreamSynchronize` or `cudaDeviceSynchronize` or `aten::item` or `aten::copy_`, and the pipeline is frozen for the duration.

This is a **forced synchronization**: something in your code asked the CPU to *wait for the GPU to finish*, which throws away the whole async pipeline. The classic culprits, all of which you can spot by the name on the long bar:

- **`.item()` or `.cpu()` or `.tolist()`** — reading a scalar or tensor back to the host mid-forward (a stopping criterion, a logged metric, a branch on a computed value). Each one forces a `cudaStreamSynchronize` and a tiny D2H copy, and the CPU blocks until every kernel launched so far has drained.
- **`print(loss)` or `if tensor > threshold:`** — the same thing in disguise; any Python that needs a *value* from a GPU tensor forces the sync.
- **A pageable (non-pinned) D2H copy of the output** — `cudaMemcpy` from device to unpinned host memory is *synchronous by definition*; it blocks the stream. Even `non_blocking=True` silently falls back to blocking if the host buffer is not pinned.
- **`torch.cuda.synchronize()` you forgot to remove** — sometimes it is your own benchmarking scaffold left in the hot path.

The mechanism of why a sync is so expensive is the launch-to-execute lag again, run in reverse. Normally the CPU is *ahead* of the GPU by a full queue's worth of backlog. A sync forces the CPU to *stop and wait for that entire backlog to drain* before it can continue. So the cost of the sync is not the sync call itself (microseconds) — it is the depth of the queue you throw away, which can be milliseconds. And if the sync is *inside your per-step loop*, you pay it every step: the pipeline never gets to run more than one step ahead, so you lose all the overlap the async model was giving you. A per-step `.item()` can silently double your latency.

The fix is to **remove the sync from the hot path**: defer the `.item()` to the end of the batch, keep results on-device, accumulate metrics on the GPU and read them once at the end, and *pin* any host buffer you copy into so the D2H can be truly asynchronous and overlap. Here is the copy version of the fix, which is the most common one in inference services:

```python
import torch

# WRONG: pageable host buffer -> the D2H copy is synchronous and stalls
# the stream on every request, showing as a long cudaMemcpy bar.
def infer_stalling(model, x):
    y = model(x)                      # runs on GPU
    return y.cpu()                    # blocking D2H: CPU waits for the whole
                                      # queue to drain -> long sync bar

# RIGHT: pin the destination once, copy non_blocking into it, and only
# synchronize when you actually need the bytes on the host.
class Copier:
    def __init__(self, shape, dtype=torch.float16):
        self.host = torch.empty(shape, dtype=dtype, pin_memory=True)  # pinned
        self.stream = torch.cuda.Stream()

    def infer_overlapped(self, model, x):
        y = model(x)                                    # compute stream
        with torch.cuda.stream(self.stream):
            self.host.copy_(y, non_blocking=True)       # true async D2H,
                                                        # overlaps next compute
        return self.host, self.stream                   # sync later, once
```

The `non_blocking=True` flag is *only* honored if `self.host` is pinned memory — that is the whole reason for the `pin_memory=True`. With a pageable buffer the flag is silently ignored and you are back to a stalling copy. This is the single most common "I added `non_blocking` and nothing changed" bug, and the trace shows it immediately: a pinned copy appears as a bar on the *copy engine* lane overlapping compute; a pageable copy appears as a long *synchronous* bar that blocks everything. We go deeper on this in the memory track; here the point is that the trace tells you which one you have by *where the bar sits* and *whether anything runs next to it*.

#### Worked example: the .item() that doubled p50

A text-generation service reads a stopping token each step: `if next_id.item() == eos: break`. Innocuous Python — but `next_id` is a GPU tensor, so `.item()` forces a `cudaStreamSynchronize` every decode step. Open the trace and the `postprocess`-adjacent region shows a long bar named `aten::item` / `cudaStreamSynchronize` on the CPU thread, and — the tell — the GPU row goes *quiet* right after it, because the sync drained the queue and the CPU has to refill it from empty before the GPU restarts.

Put numbers on it. The decode step's real GPU work is 2.2 ms. The sync itself is a few microseconds, but it forces the CPU to wait for the ~2.2 ms backlog to drain *and* forbids the CPU from running the next step ahead — so the async overlap that normally hides the next step's launches is gone. Measured p50 per decode step: 4.6 ms with the per-step `.item()`, 2.4 ms after moving the stop check to a batched, on-device comparison read once every 16 tokens. The trace before shows a sync bar between every pair of decode steps like rungs on a ladder; after, the rungs vanish and the decode steps butt up against each other. Same model, same tokens, 1.9× on p50 — from deleting one `.item()` from the hot path.

## The overlap you want to see

That last point deserves its own picture, because the difference between a copy that costs you and a copy that is free is entirely about *overlap*, and overlap is a visual property of the trace — two bars side by side in time versus one after the other.

![two timelines where the top serializes a memcpy before the kernel and the bottom hides the memcpy underneath the kernel](/imgs/blogs/reading-a-chrome-trace-6.webp)

The top timeline is the serialized case: the H2D copy runs, *finishes*, and only then does the kernel start. The copy time is pure tax — it is added to the step, on the critical path, wasted. The bottom timeline is the overlapped case: the copy runs on the copy-engine lane *underneath* the kernel on the compute lane. The bytes for the next request are moving across PCIe while the GPU is busy computing the current one. The copy still takes the same wall-clock to complete, but it is *hidden* — it is off the critical path, so it costs the step nothing.

The arithmetic makes the stakes concrete. Move a batch of 64 images, 3×224×224, fp16, from host to device. That is $64 \times 3 \times 224 \times 224 \times 2$ bytes ≈ 19.3 MB. Over PCIe Gen4 at an effective ~12 GB/s with pinned memory, the copy takes ~1.6 ms. If your compute step is 6 ms, then serialized you pay ${6 + 1.6 = 7.6}$ ms; overlapped you pay $\max(6, 1.6) = 6$ ms. That is a 21% latency cut from *nothing but scheduling the copy differently* — no faster GPU, no smaller model. And it is invisible unless you read the copy-engine lane against the kernel lane and ask: *are these two bars stacked in time, or strung out?*

| | Serialized copy | Overlapped copy |
|---|---|---|
| Host buffer | pageable | pinned (`pin_memory=True`) |
| Copy call | blocking | `non_blocking=True` on its own stream |
| Trace signature | copy bar *then* kernel bar | copy bar *under* kernel bar |
| Step cost | compute + copy | max(compute, copy) |
| The 64-image example | 7.6 ms | 6.0 ms |

The healthy trace from figure 1 — the one your eye calibrated on at the top — has exactly this overlap: the input copy tucked under the first kernel. Now you know what you were looking at.

## Zooming and reading: the mechanics in the UI

Enough theory; here is how you actually drive the tool. You have `trace.json`; open it one of two ways.

```bash
# Option A: Perfetto UI (recommended for large traces; runs locally in-browser).
#   Go to https://ui.perfetto.dev and drag trace.json onto the page.
#   Nothing is uploaded to a server for the Chrome-trace format; it parses
#   client-side.

# Option B: TensorBoard, if you exported with tensorboard_trace_handler.
pip install torch-tb-profiler
tensorboard --logdir=./log        # then open the PYTORCH_PROFILER tab

# Option C: legacy chrome://tracing (fine for small traces).
#   Open chrome://tracing in Chrome, click Load, pick trace.json.
```

Once it is open, the reading loop is four moves:

1. **Find one step.** Use the `record_function`/NVTX bars on the top lane, or the repeating pattern, to isolate a single forward pass. Everything you measure should be *per step*, not across the whole capture (which includes warmup).
2. **Select the step's span and read its duration.** In Perfetto, drag-select a region and it shows you the total duration in the selection — that is your $T_\text{span}$. Or press `m` to mark a range.
3. **Read the GPU row's gap fraction.** Eyeball the whitespace first; if it is clearly host-bound, you are done diagnosing. For a number, select the GPU stream track and Perfetto's "slice" summary gives you the summed duration — divide by the span. (Or run the JSON reader from earlier.)
4. **Read down from any symptom.** See a gap? Look at the CPU lane directly above its left edge — the launch that was late is right there. See a long bar? Read its name. See a fence of slivers? Count them and check the per-kernel duration in the tooltip.

The single most useful habit: **hover a kernel and read its `dur`.** A trace reader always knows whether the kernels are 4 µs (launch-bound territory) or 400 µs (compute-bound territory), because that one number tells you which of the three shapes you are even allowed to be in. Four-microsecond kernels can *only* be a gap or a picket-fence problem; four-hundred-microsecond kernels are compute-bound and you should stop looking at the timeline and go read the [roofline](/blog/machine-learning/performance-engineering/the-roofline-for-your-service) and Nsight Compute instead.

### Reading at three zoom levels

A trace answers different questions at different magnifications, and part of the skill is knowing which zoom to be at for which question. There are three levels, and you move between them deliberately.

- **The whole-capture zoom (milliseconds per pixel).** Zoomed all the way out, one step is a few pixels wide and you cannot see individual kernels — but you *can* see the macro-rhythm: are the steps evenly spaced, or is every seventh one twice as tall? This is where you catch periodic stalls — a garbage-collection pause, a recompilation, a checkpoint write — that never show up if you only ever look at one step. If the steps are ragged at this zoom, your p99 problem is a *periodic* event and you should hunt for what happens on the tall steps. This is also the zoom at which you confirm you are looking at *steady state* and not warmup: the first few steps are always slower and you exclude them.
- **The one-step zoom (microseconds per pixel).** This is the working zoom for the three shapes. One forward pass fills the screen, the GPU row's gaps and fences are legible, and the gap-fraction arithmetic applies. Ninety percent of trace reading happens here. Anchor on a `record_function` bar to make sure you have exactly one step selected.
- **The one-kernel zoom (nanoseconds per pixel).** All the way in on a single kernel and its launch: this is where you *see* the launch-to-execute lag as an actual horizontal distance, where you read the exact `dur`, and where you confirm whether the copy bar truly overlaps the kernel bar or just *looks* close at coarser zoom. When two people disagree about whether a copy is overlapped, the argument is settled at this zoom.

The mistake beginners make is diagnosing at the wrong zoom — declaring a service healthy from the whole-capture view (where gaps are invisible) or hunting for a periodic tail from the one-step view (where you only see one step, so periodicity is definitionally invisible). Match the zoom to the question: macro-rhythm and tails at the top, the three shapes in the middle, the lag and the overlap at the bottom.

There is a direct line from that top zoom to the p99 tail, worth stating because it is where trace reading pays for itself on the metrics that page you. If the steady-state step is $p50$ and one step in a hundred hits a stall of duration $s$ — a sync, a GC pause, a recompile — then $p99 \approx p50 + s$. The trace at the whole-capture zoom shows you $s$ as the height of the tall steps, and reading down into one tall step at the one-step zoom shows you *what* $s$ is (a long sync bar, a compile event on the CPU lane). That is the entire method for a stall-driven tail: find the tall steps, zoom in, name the bar.

### Annotating so the trace is readable

The reason step 1 and step 4 work at all is annotation. Out of the box, the trace shows kernel names like `sm80_xmma_gemm_f16f16_...` and `void at::native::vectorized_elementwise_kernel<...>` — accurate but unmappable to your code. Fix that at capture time by wrapping your handler's phases in ranges, so the top lane becomes a legend:

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity, schedule

def handler(model, batch):
    with record_function("preprocess"):
        x = batch.pin_memory().to("cuda", non_blocking=True)
    with record_function("encoder"):
        h = model.encoder(x)
    with record_function("head"):
        y = model.head(h)
    with record_function("postprocess"):
        return y.argmax(-1).cpu()          # note: this .cpu() is a sync — watch it

sched = schedule(wait=1, warmup=1, active=3, repeat=1)
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=sched,
    record_shapes=True,
    with_stack=False,
) as prof:
    for _ in range(5):
        handler(model, next_batch())
        prof.step()

prof.export_chrome_trace("trace.json")
```

Now, when you open the trace, the top lane reads `preprocess | encoder | head | postprocess`, and any gap or fence or stall on the GPU row sits *under* the named phase that produced it. The `postprocess` phase, with its `.cpu()`, will show the tell-tale sync bar — which you left in on purpose so the trace teaches you where your own stalls are. This is the same idea NVTX ranges give you in Nsight Systems, which the [nsight-systems sibling post](/blog/machine-learning/performance-engineering/nsight-systems-for-ai-services) covers for the system-wide view; the annotation discipline is identical.

## A guided read of a real-looking trace

Let us put it all together the way you would in an incident: a service is slow, you have the trace, and you have thirty seconds before someone asks what is wrong. Here is the decision procedure drawn as a tree — you route from the GPU row's shape to the cause to the fix.

![a decision tree that routes from a question about the gpu row shape through three shapes down to the cause and fix for each](/imgs/blogs/reading-a-chrome-trace-7.webp)

Now walk it on a concrete trace. Our Transformer service is at p50 = 9 ms, and product wants 5 ms. Open the trace, isolate one step (the `encoder` range on the top lane spans it), and read top to bottom.

**First read — the GPU row.** Whitespace everywhere between kernels. Eyeball says host-bound. Confirm with the number: summed kernel duration 4.0 ms, span 9.0 ms, gap fraction 55%. That is shape one. Look up at the CUDA runtime API lane: a solid picket-fence of `cudaLaunchKernel` bars, the CPU launching nonstop. Hypothesis: host-bound from too many tiny launches. Fix pointer: CUDA graphs. This alone predicts a step near the 4 ms GPU floor.

**Second read — zoom the LayerNorm region.** Anchor on a normalization `record_function` bar; below it, a fence of eight 3–4 µs slivers per norm. That is shape two nested inside shape one: even the kernels that *do* run are launch-dominated. Fix pointer: `torch.compile` fusion, which will also cut HBM round-trips. This tightens the 4 ms GPU floor itself toward ~3.4 ms.

**Third read — the postprocess phase.** The `postprocess` range on the top lane sits over a long solid bar named `aten::copy_` / `cudaStreamSynchronize` — the `.cpu()` in the handler. That is shape three: a forced D2H stall, ~0.9 ms every step, and because it is a pageable copy it blocks the pipeline from running the next request ahead. Fix pointer: pin the output buffer, copy `non_blocking`, defer the `.cpu()`.

Three shapes, three fixes, all read off one trace in under a minute. Apply all three — `torch.compile(mode="reduce-overhead")` collapses launches *and* fuses (shapes one and two), and the pinned-copy change removes the stall (shape three) — re-capture, and the trace transforms into the right half of the contrast figure: a packed GPU row, 8% gap, no fence, no sync bar. Here is the honest before/after on named hardware.

| Metric (A100 80GB, batch 1, seq 128, fp16) | Before (eager) | After (compile + graphs + pinned copy) |
|---|---|---|
| `nvidia-smi` GPU-Util | ~100% (lies) | ~100% |
| GPU active fraction (from trace) | 45% | 85% |
| Gap fraction (idle) | 55% | 8% |
| Kernels launched per step | ~1,500 | 1 (graph replay) |
| Host launch time per step | 9.0 ms | 0.2 ms |
| D2H stall per step | 0.9 ms | ~0 (overlapped) |
| p50 latency | 9.0 ms | 4.3 ms |
| p99 latency | 11.5 ms | 4.8 ms |
| Throughput | ~111 req/s | ~230 req/s |
| Cost per million requests (≈\$1.8/GPU-hr, approx) | ~\$4.5 | ~\$2.2 |

Every number in the "after" column is one the trace *predicted* — the 4 ms GPU floor became the wall, the fence became solid bars, the sync bar vanished. That is the payoff of reading the trace before touching the code: you knew the answer would be ~4.3 ms before you wrote the fix, because you had measured the whitespace.

### The stress test: does the diagnosis hold at other operating points?

A diagnosis you cannot stress-test is a guess. Re-read the trace under different conditions and watch the shapes change — that is how you know you understood the cause, not just this one capture.

- **Batch 1 vs batch 64.** At batch 1 the kernels are tiny (2–4 µs) and the gap fraction is 55% — deeply host-bound. Re-capture at batch 64 and the *same model* has kernels that are now 40–120 µs (they do 64× the work), the launch cost is unchanged, and the gap fraction collapses to ~10%. The trace visibly heals as you raise the batch: the picket-fence fattens into solid bars. This is why "just increase the batch size" is the first thing to try for a host-bound service — and why the graphs win is *largest* at batch 1, where the launch overhead is the biggest share.
- **A100 vs L4.** Move the batch-1 service to an L4 (≈242 fp16 TFLOP/s, 300 GB/s — a much smaller GPU). The kernels get *slower* (less compute), so $\bar{t}_k$ rises, which means the launch cost is a *smaller* fraction of each kernel — the L4 is *less* host-bound than the A100 for the same model, counterintuitively. The gap fraction shrinks not because the L4 is better but because a slower GPU hides launch overhead. Read the trace, not your intuition about "bigger GPU = better."
- **With vs without the pinned-copy overlap.** Toggle the pinning off and re-capture: the copy bar jumps from the copy-engine lane (overlapping compute) to a long synchronous bar on the CPU thread, and the step grows by the full copy time. Toggle it on and the bar slides back under the compute. Watching that bar move between two captures is the cleanest way to *prove* the overlap is working.
- **Fixed shapes vs a shape that changes every request.** The CUDA-graphs fix for shape one assumes the step is *replayable* — same tensor shapes, same pointers, every time. If your service takes a different sequence length per request, capture two traces: one where you padded every request to a fixed bucket, and one where shapes float. The floating-shape trace will show the graph *failing to replay* (you fell back to eager, the gaps are back) or, worse, a `torch.compile` recompilation event — a giant CPU bar where Inductor re-generates kernels for the new shape. The trace tells you *before* you ship whether your graphs will hold: if the kernel counts and durations are identical across two consecutive steps, the shape is stable and graphs are safe; if they wander, you need shape bucketing first. That recompilation-storm signature is its own war story later in the series.

## The same three shapes in Nsight Systems

The Chrome trace is the fastest way in, but it has a ceiling: it only sees what `torch.profiler` instruments (PyTorch ops, CUDA runtime, kernels, copies), and it captures a few steps at a time. When you need the *system-wide* picture — the OS scheduler, other processes, a longer window — you reach for Nsight Systems, and the good news is that **the three shapes read identically**. The rows are the same idea with more of them: a CUDA API row (the `cudaLaunchKernel` picket-fence lives here), one row per GPU stream (the kernels and the gaps), a memcpy row, and OS/runtime rows below. Capture it like this:

```bash
# System-wide timeline: CUDA API + kernels + NVTX ranges + OS runtime + cuBLAS/cuDNN.
nsys profile -t cuda,nvtx,osrt,cublas,cudnn \
    --capture-range=cudaProfilerApi \
    -o transformer_service \
    python serve.py --steps 200

# Then, instead of eyeballing, get the numbers straight from the CLI:
nsys stats --report cuda_gpu_kern_sum transformer_service.nsys-rep
nsys stats --report cuda_api_sum       transformer_service.nsys-rep
```

The `cuda_gpu_kern_sum` report is the picket-fence and the gap fraction in a table. It ranks kernels by total GPU time and shows the count and average duration of each — a wall of thousands of tiny kernels *is* shape two, printed:

```console
 Time(%)  Total Time(ns)   Instances   Avg(ns)   Name
 -------  ---------------  ----------  --------  ------------------------------------
    41.2       1,648,000        300      5,493   vectorized_elementwise_kernel<...>
    28.7       1,148,000         24     47,833   sm80_xmma_gemm_f16f16_128x128_tn
    12.1         484,000         96      5,041   layer_norm_kernel<float, float>
     9.4         376,000         48      7,833   softmax_warp_forward<...>
     8.6         344,000         12     28,667   sm80_xmma_gemm_f16f16_64x64_nt
 -------  ---------------  ----------  --------  ------------------------------------
```

Read it the way you read the timeline. The top line is 300 elementwise kernels averaging 5.5 µs each — a picket-fence, and 41% of GPU time spent on kernels so small they are launch-dominated (5.5 µs of GPU work behind a ~6 µs launch). The two `xmma_gemm` lines are the real matmuls: 36 instances, 28–48 µs each — those are the fat kernels doing actual work. The diagnosis writes itself: fuse the 300 elementwise kernels (they collapse under `torch.compile`) and the matmuls stay. And the companion `cuda_api_sum` report confirms host-boundness directly:

```console
 Time(%)  Total Time(ns)   Num Calls   Avg(ns)   Name
 -------  ---------------  ----------  --------  --------------------------
    72.4       9,010,000       1,500      6,006   cudaLaunchKernel
    14.1       1,756,000           4    439,000   cudaStreamSynchronize
     8.9       1,108,000         200      5,540   cudaMemcpyAsync
 -------  ---------------  ----------  --------  --------------------------
```

There it is in one number: 1,500 `cudaLaunchKernel` calls totaling 9.01 ms of CPU time per step — the exact 9 ms that paces the wall-clock and starves the GPU. The `cudaStreamSynchronize` line (1.76 ms across 4 calls, 439 µs each) is shape three, quantified: four forced syncs eating the pipeline. You did not even open the timeline and the CLI already told you all three shapes are present. The [Nsight Systems sibling post](/blog/machine-learning/performance-engineering/nsight-systems-for-ai-services) drives this workflow end to end; the point here is that the *reading* transfers — gaps, fences, and sync bars are the same three shapes whether you see them in Perfetto's rectangles or Nsight's summary tables.

| Question you are asking | Best tool | What it shows the three shapes as |
|---|---|---|
| Where is the whitespace between kernels? | Chrome trace (Perfetto) | Visual gaps on the GPU row |
| How many tiny kernels, and how small? | `nsys stats cuda_gpu_kern_sum` | Ranked kernel table, count + avg µs |
| Is the CPU launch-bound? | `nsys stats cuda_api_sum` | `cudaLaunchKernel` total time |
| Why is *one* kernel slow inside? | Nsight Compute (`ncu`) | Occupancy, warp stalls, Speed-of-Light |
| Is the stall in Python before the GPU? | py-spy flamegraph | CPU time with no GPU work under it |

## Case studies and real numbers

These are the kinds of results the reading skill produces, drawn from published sources and the mechanisms above. Where a number is approximate I say so.

- **CUDA graphs / `reduce-overhead` on launch-bound models.** PyTorch's own CUDA-graphs documentation and the `torch.compile(mode="reduce-overhead")` design notes report that collapsing per-op launches into a single graph replay gives the largest wins precisely on small-batch, many-tiny-kernel workloads — the host-bound regime this post's shape-one diagnosis identifies. The mechanism (removing $N \cdot t_\text{launch}$) is exactly the gap-fraction math; the reported speedups on such workloads are commonly in the ~1.5–2× range at small batch, shrinking toward 1× as the batch grows and the model becomes compute-bound. That "shrinks with batch" behavior is the stress test above, confirmed.
- **Inductor fusion killing a picket-fence.** The `torch.compile` Inductor backend fuses chains of pointwise and reduction ops into generated Triton kernels; the published pattern is that a decomposed normalization or a GELU + residual chain — a dozen library kernels — becomes one or two fused kernels. The visible proof is exactly the shape-two before/after: a comb of 3–4 µs slivers on the GPU row becomes a handful of solid bars, with both the kernel *count* and the HBM traffic dropping. Verifying fusion *by re-reading the trace and counting kernels* is the recommended discipline, not trusting that it happened.
- **Pinned-memory copy overlap.** The PyTorch performance guide documents that `non_blocking=True` is only asynchronous with pinned host memory, and that overlapping H2D/D2H with compute removes the copy from the critical path. On a data-heavy vision service the trace signature — copy bar sliding from a serial position to under the kernel — corresponds to the ~15–25% step-time reduction the arithmetic predicts for a copy that was a meaningful fraction of the step.
- **The util-counter lie, quantified.** The gap between `nvidia-smi`'s "100% util" and a trace's true active fraction is not a rounding error; the host-bound Transformer above reads 100% on the counter and 45% on the trace. NVIDIA's own documentation notes the util metric counts "at least one kernel resident in the sampling window," which is why the trace, not the counter, is the source of truth for whitespace.

For the deeper system-wide version of this reading — across CPU, GPU, and copies at once, with the NVTX-annotated timeline — see the model-serving series' [profiling LLM serving with Nsight](/blog/machine-learning/model-serving/profiling-llm-serving-with-nsight), which applies the same three-shapes eye to an Nsight Systems capture of a real LLM server.

## When to reach for the trace (and when not to)

The Chrome trace is the right tool for a specific question and the wrong tool for others. Reach for it when:

- **You suspect host-bound or a stall** — anything where *time is being lost between kernels* rather than inside them. The timeline is the only view that shows whitespace, launch overhead, and sync bars. This is its home turf.
- **You need to map GPU work back to your code** — the annotated top lane is how you connect a slow kernel to the `attention` block in your handler.
- **You want to verify a fix visually** — did the fence collapse, did the gap close, did the copy slide under compute? The trace *shows* the change, which is more convincing than a single latency number that could have moved for other reasons.

Do **not** reach for the Chrome trace when:

- **Your kernels are already big and back-to-back** — a packed GPU row with 400 µs kernels means you are compute- or memory-bound *inside* a kernel, and the timeline has nothing more to tell you. Switch to Nsight Compute (`ncu`) for occupancy, warp-stall reasons, and the Speed-of-Light section, and read the [roofline](/blog/machine-learning/performance-engineering/the-roofline-for-your-service) to know whether it is compute or bandwidth. The timeline shows *that* a kernel is slow; only `ncu` shows *why*.
- **The problem is a rare tail, not the steady state** — a p99 spike that happens once every thousand requests will not show up in a 3-step capture. You need continuous, low-overhead system tracing (Nsight Systems over a longer window, or a sampling profiler) to catch a periodic GC pause or a recompilation storm.
- **The bottleneck is pure Python far from the GPU** — a slow tokenizer or a JSON serialization in the request path shows as CPU time with *no* corresponding GPU work, and a Python sampling profiler (py-spy) gives you the flamegraph faster than squinting at CPU-op bars.

The rule of thumb: **the Chrome trace is for the gap between kernels; Nsight Compute is for the inside of a kernel; py-spy is for the Python before the kernel.** Pick by where you suspect the time is going — and if you do not know, the trace is the right *first* look precisely because it shows you which of the three regions to zoom into next.

## Key takeaways

- **Read the whitespace, not the color.** The waste on a GPU timeline is the gaps between kernels, not the kernels themselves. Train your eye to see the spaces.
- **Three shapes diagnose almost everything.** Gaps → host-bound (fix: CUDA graphs). Picket-fence of tiny kernels → launch-bound (fix: `torch.compile` fusion). Long sync bar → forced sync/D2H (fix: remove the sync, pin the buffer).
- **Compute the gap fraction.** $1 - \sum d_i / T_\text{span}$ off the GPU stream lane. Above ~0.3 and you are host-bound, no matter what `nvidia-smi` says.
- **`nvidia-smi` util lies; the trace tells the truth.** "100% util" and "45% active" describe the same host-bound run. Only the timeline shows the difference.
- **Cause is on the top rows, effect on the bottom.** Read *down* from a gap on the GPU row to the late launch or blocking sync on the CPU rows above it.
- **The launch-to-execute lag is the mechanism.** A deep, stable lag means a full queue and a fed GPU; a lag that collapses to zero right before a gap is the CPU falling behind, drawn.
- **`non_blocking=True` needs pinned memory.** A pageable D2H copy is synchronous and shows as a long stall bar; a pinned copy slides under compute and costs the step nothing.
- **Verify the fix by re-reading the trace.** Count kernels to confirm fusion, measure the gap to confirm graphs, watch the copy bar move to confirm overlap. The picture is the proof.
- **Kernel duration tells you which shape you can even be in.** 4 µs kernels can only be gaps or fences; 400 µs kernels are compute-bound — go to Nsight Compute.

## Further reading

- [Profiling PyTorch with torch.profiler](/blog/machine-learning/performance-engineering/profiling-pytorch-with-torch-profiler) — the sibling that captures the trace you just learned to read (the schedule, `record_function`, `key_averages()`, the export call).
- [The GPU-service execution model](/blog/machine-learning/performance-engineering/the-mental-model-of-a-gpu-service) — the producer/consumer, host/device, async-stream substrate this whole reading rests on.
- [The kernel-launch-overhead problem](/blog/machine-learning/performance-engineering/the-kernel-launch-overhead-problem) — the mechanism behind shape one and shape two, and the CUDA-graphs fix in depth.
- [Nsight Systems for AI services](/blog/machine-learning/performance-engineering/nsight-systems-for-ai-services) — the same three-shapes eye applied to a system-wide timeline across CPU, GPU, and copies.
- [Profiling LLM serving with Nsight](/blog/machine-learning/model-serving/profiling-llm-serving-with-nsight) — a real LLM server read on the timeline, end to end.
- [Why your AI service wastes CPU and GPU](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu) — the series intro: the four wastes and the profile → hypothesize → fix → measure loop.
- [The performance-engineering playbook](/blog/machine-learning/performance-engineering/the-performance-engineering-playbook) — the capstone decision tree: symptom → which profiler → likely cause → fix.
- The PyTorch Profiler tutorial, the PyTorch CUDA Graphs documentation, the `torch.compile` / Inductor docs, and the Perfetto UI documentation — the primary sources for the tools and APIs used above.
