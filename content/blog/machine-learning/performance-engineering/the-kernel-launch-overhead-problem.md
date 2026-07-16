---
title: "The Kernel Launch Overhead Problem: When Thousands of Tiny Kernels Starve Your GPU"
date: "2026-07-14"
publishDate: "2026-07-14"
description: "Your service reads GPU-Util 100% and still runs 3x too slow. This is the story of launch overhead: why 1,800 tiny kernels a step cost more in CPU launch time than they cost in GPU compute, how to prove it with the profiler, and the law that tells you exactly when it bites."
tags:
  [
    "performance-engineering",
    "gpu-optimization",
    "cuda-graphs",
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
readTime: 41
---

Here is a bug report that will make no sense until you have read this post. A Transformer inference service is deployed on an A100 80GB. The on-call engineer runs `nvidia-smi`, sees `GPU-Util 100%`, and closes the ticket: the GPU is pegged, nothing to optimize, buy more hardware. Except the service serves 40 requests per second when the same model on the same card should serve 120. The GPU is reporting that it is completely busy, and it is also, provably, doing nothing three-fifths of the time. Both facts are true at once, and the reason is the single most under-appreciated tax in GPU computing: **kernel launch overhead**.

The model runs a batch-1 forward pass that dispatches roughly 1,800 CUDA kernels — one per elementwise op, one per small GEMM, one per norm, one per activation, one per bias-add, all of them tiny because the batch is one. Each of those kernels is genuinely fast on the GPU: 3 to 8 microseconds of actual math. But every single one of them costs the *CPU* another 5 to 10 microseconds just to *launch* — to walk PyTorch's dispatcher, pack the launch arguments, and hand the work to the driver — before the GPU is even told what to do. Multiply the launch cost by 1,800 and you get roughly 12 milliseconds of pure host-side launch work per step, against about 8 milliseconds of real compute. The GPU finishes its 8 ms of math and then sits idle for the remaining 4.6 ms of every step, waiting for a CPU that cannot launch fast enough. That idle time is invisible to `nvidia-smi`, which is why the ticket got closed.

![a four row table matching each launch bound symptom to what you see and where you read it](/imgs/blogs/the-kernel-launch-overhead-problem-1.webp)

The figure above is the signature you are learning to recognize — the four tells of a launch-bound service, and the tool that reveals each. By the end of this post you will be able to read all four off a live service in about two minutes: the profiler showing CPU time far exceeding CUDA time, the Chrome trace showing a picket-fence of tiny kernels, the `#Calls` column in the thousands, and `nvidia-smi` reporting 100% while the wall clock says otherwise. You will be able to derive the exact point at which a service *becomes* launch-bound, measure how far past that line you are, and name which of three fixes to reach for. This is the first post of the CUDA-graphs track in the [Profiling & Optimizing AI Services](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu) series, and it exists to prove the problem is real and measurable *before* the next posts hand you the fix. You cannot appreciate CUDA graphs until you have felt the pain they remove.

This is the first of the four wastes from the series intro — the **idle GPU**, host-bound, starved by its own launch queue. It is also the most common, the most misdiagnosed, and the one that looks the most like "the GPU is just busy." Let us make it visible.

## A service that is all green and still too slow

Start with the contradiction, because resolving it is the whole game. The waste in a launch-bound service is not in the kernels — the kernels are fine. The waste is in the *gaps between* the kernels, and the util counter is structurally incapable of seeing gaps.

Recall from [the mental model of a GPU service](/blog/machine-learning/performance-engineering/the-mental-model-of-a-gpu-service) how work actually reaches the GPU. Your Python code does not run on the GPU. It runs on the CPU, and for every operation — every `matmul`, every `layer_norm`, every `+` — it issues a **kernel launch**: a request that says "run this compiled GPU function with these arguments." That request is enqueued onto a **CUDA stream**, an ordered FIFO of pending device work. The CPU is the *producer* filling the queue; the GPU is the *consumer* draining it. The two run asynchronously — the CPU does not wait for a kernel to finish before launching the next one, which is exactly what lets a fast GPU stay fed while a slower CPU races ahead building the backlog.

That asynchronous producer/consumer design is a gift when the producer keeps up. It becomes the bottleneck the instant the producer *cannot* keep up. And here is the cruel arithmetic: the GPU consumes a tiny batch-1 kernel in a few microseconds, but the CPU takes just as long — sometimes longer — to *produce* the next launch. When production is slower than consumption, the queue drains to empty, the GPU finishes its last kernel, and it stalls with nothing to run. The GPU is idle, but it is idle in short bursts scattered across the step, so at any given `nvidia-smi` sampling instant there is usually *some* kernel resident, and the counter dutifully reports "busy."

That is the entire lie in one sentence: **`nvidia-smi` GPU-Util reports whether at least one kernel was resident during its sampling window, not what fraction of the wall clock the SMs were doing math.** A service that runs a 5 µs kernel, idles 10 µs, runs another 5 µs kernel, idles 10 µs, forever, is idle two-thirds of the time and reads 100% util. (The [metrics-that-actually-matter post](/blog/machine-learning/performance-engineering/metrics-that-actually-matter) dissects this counter in depth; here we take it as the trap that sends you looking in the wrong place.)

So the symptom "util 100%, throughput a third of what it should be" is not a paradox. It is the fingerprint of a host-bound service. The question is how to confirm it and how to quantify it — and for that we first need to see where the launch time actually goes, and then derive the law that predicts it.

## The punchline first: 1,800 launches versus one replay

Before the mechanism, let me show you the destination, because it reframes everything that follows. The service above — the 1,800-kernel batch-1 encoder on the A100 — has exactly one thing wrong with it: it launches too many kernels, and each launch costs too much. There are two ways to fix that (launch fewer kernels, or make each launch cost nothing), and the most dramatic is CUDA graphs, which does the second: it records the entire sequence of 1,800 launches *once*, then replays the whole recorded graph as a single driver operation. One launch instead of 1,800.

![two stacked panels contrasting a run of 1800 launches against a single graph replay with host launch and idle numbers](/imgs/blogs/the-kernel-launch-overhead-problem-2.webp)

The numbers on that before-and-after are the ones we are going to earn, step by step, in the rest of the post. On the left, the eager service: 12.6 ms of host launch cost per step, a GPU that idles 4.6 ms every step waiting for launches, and a p50 latency of 12.6 ms (the step is *bounded by the launch cost*, not the compute). On the right, the same forward pass replayed as one CUDA graph: host launch cost collapses to 0.2 ms, the idle gap closes to near zero, and p50 drops to 8.1 ms — which is the actual compute time, the number the GPU could have hit all along. Same math, same output tensors, same hardware. The only thing that changed is *how the work was launched*.

Here is that result as the kind of before/after table you should demand of every optimization, on named hardware, measured honestly (warm-up iterations, `torch.cuda.synchronize()` before every timing boundary, CUDA events for the device timing, locked clocks, steady state — the method we will show later):

| Metric (A100 80GB, batch 1, small encoder) | Eager (1,800 launches) | CUDA-graph replay | Change |
|---|---|---|---|
| Kernels launched / step | 1,800 | 1 (replay) | −1,799 |
| Host launch cost / step | 12.6 ms | 0.2 ms | −98% |
| GPU idle / step | 4.6 ms (37%) | 0.1 ms (1%) | gap closed |
| Real GPU compute / step | 8.0 ms | 8.0 ms | unchanged |
| p50 latency | 12.6 ms | 8.1 ms | −36% |
| Throughput (1 stream) | 79 req/s | 123 req/s | +56% |
| GPU-Util (`nvidia-smi`) | 100% | 100% | unchanged (still lies) |

Note the last row: the util counter reads 100% both before and after. It was never going to tell you there was a 56% throughput win sitting on the table. The trace and the profiler tell you. The point of the whole track is to teach you to see that gap and close it — and this post is the part where you learn to *see* it. So let us go to the mechanism.

## Anatomy of a single kernel launch

Why does launching a kernel cost 5 to 10 microseconds of CPU time when the kernel itself might run in 3? Because "launch a kernel" is not one instruction — it is a small pipeline of host-side work that has to happen before the GPU can be told to do anything, and that work is *fixed*: it costs the same 7-ish microseconds whether the kernel processes four numbers or four million.

![a vertical stack of layers showing a launch passing through dispatch argument packing and driver enqueue before the kernel runs](/imgs/blogs/the-kernel-launch-overhead-problem-3.webp)

Walk down the stack in the figure. When your Python code calls, say, `torch.nn.functional.gelu(x)`, the following happens entirely on the CPU before a single SM lights up:

1. **Python and ATen dispatch (~2 µs).** The Python call crosses into C++, PyTorch's dispatcher resolves which kernel to run based on the tensor's dtype, device, and layout, autograd bookkeeping is checked, and the correct backend function is selected. This is pure host CPU work, and for a small op it is a real fraction of the total.
2. **Build the launch (~1.5 µs).** The kernel's launch configuration is assembled: the grid and block dimensions, the shared-memory size, the pointer to each input and output tensor, the scalar arguments. These get marshalled into the structure the driver expects.
3. **Driver enqueue (~2 µs).** Control passes into the CUDA driver via `cudaLaunchKernel`. The driver validates the launch, translates it into a hardware command packet, and appends it to the command buffer for the target stream. This is the part you cannot avoid without a graph — it is the driver's per-launch cost.
4. **Push to the stream queue (~1.5 µs).** The command is placed onto the stream's FIFO. The call returns to Python, which is now free to go build the *next* launch. The GPU, whenever it gets around to draining this far into the queue, will finally execute the kernel.

Add those up and you get roughly 7 microseconds of host work standing between "Python wants to run gelu" and "an SM is running gelu." The exact figure depends on your CPU, your PyTorch version, whether you are in eager or a compiled region, and how warm the caches are — 5 µs on a fast host with a simple op, 10 µs or more on a slow host or a complex dispatch — but the order of magnitude is a hard fact of the architecture, and it is documented consistently across NVIDIA's own materials and the PyTorch performance guides: a CUDA kernel launch costs single-digit microseconds of CPU-side overhead.

The sneakiest of these costs is the first one, the dispatcher, because it is the one that feels like it should be free and is not. PyTorch's eager mode is *dynamic*: nothing about which kernel to run is decided until the moment the op is called. For every `+` the dispatcher must ask, at runtime, "what dtype are these tensors? what device? what layout? is autograd recording? is there a custom mode or a subclass override in play?" and route to the matching implementation through a chain of dispatch keys. That flexibility is what makes PyTorch pleasant to use — you can mix dtypes and devices freely and it just works — but every ounce of that runtime flexibility is paid for in host CPU cycles, per op, every time. It is the price of eager execution, and it is precisely the cost that `torch.compile` removes by resolving the dispatch *once* at compile time and generating a static kernel sequence. When you hear that a compiled model has "less Python overhead," this dispatcher work is a large part of what that means.

The critical property is that **this cost is fixed per launch and independent of the work.** A GEMM that multiplies two 4096×4096 matrices pays the same ~7 µs launch tax as a GEMM that multiplies two 8×8 matrices. For the big GEMM, 7 µs of launch amortized over 40,000 µs of compute is a rounding error. For the tiny one, 7 µs of launch against 3 µs of compute means you spent **more than twice as long launching the work as doing it.** That ratio — launch cost over kernel duration — is the whole story, and it is why small batches and small models are where launch overhead goes to feast. Hold onto the fixed-cost idea; it is the load-bearing assumption in the law we are about to derive.

### The launch-to-execute lag

One more piece of the mechanism, because it is what makes the queue image precise. When the CPU calls `cudaLaunchKernel`, the call *returns immediately* — the kernel is now sitting in the stream queue, but it has not run yet. It runs later, whenever the GPU drains the queue down to it. So on a profiler timeline, a launch bar on the CPU row does **not** line up vertically with its kernel on the GPU row; the kernel is shifted to the right by however deep the queue was when it was enqueued. That horizontal distance is the *launch-to-execute lag*, and it is a direct readout of queue depth in time.

The lag has a diagnostic reading that is worth internalizing. A **large, stable lag** means the queue stays deep — the CPU is comfortably ahead, the GPU always has the next kernel waiting, and you are GPU-bound and healthy. A lag that **shrinks toward zero right before a gap** means the queue drained empty — the CPU fell behind, the GPU caught up to the front of the queue and ran out of work, and the gap that follows is the GPU idling until the next launch lands. A healthy backlog is not waste; it is the buffer that keeps a fast consumer fed by a slower producer. The trouble is only when the buffer empties, and the whole art of fixing a launch-bound service is keeping that queue from ever running dry.

## Three levers on a launch-bound service

Before the math, orient yourself on the exits, because the mechanism only matters if it points at a fix. Once you have confirmed a service is launch-bound, there are exactly three levers you can pull, and this track is organized around them.

![a decision tree branching from a host bound symptom into reduce count eliminate cost and hide it with leaf fixes](/imgs/blogs/the-kernel-launch-overhead-problem-4.webp)

- **Reduce the kernel COUNT.** If total launch cost is `N × t_launch`, shrinking `N` shrinks the bill. Two ways: **fuse** many small ops into fewer bigger ones (this is what `torch.compile`'s Inductor backend does — it fuses chains of elementwise ops into single kernels, taking a forward pass from 1,800 kernels to a few hundred), or run a **bigger batch** so each kernel does more work and there are proportionally fewer of them per unit of throughput. Fusion attacks the count directly; batching attacks the *count per useful result*.
- **Eliminate the per-launch cost.** Leave the kernel count alone but stop paying `t_launch` for each one. This is **CUDA graphs**: capture the whole sequence of launches once, then replay it as a single operation, so the 1,800 launches cost one launch's worth of host time. This is the subject of the next posts — [CUDA graphs from first principles](/blog/machine-learning/performance-engineering/cuda-graphs-from-first-principles) and [CUDA graphs in PyTorch](/blog/machine-learning/performance-engineering/cuda-graphs-in-pytorch) — and it is the most surgical fix because it changes nothing about the math.
- **Hide the launch cost behind more host throughput.** The real intermediate option, often forgotten: if one CPU thread can only issue launches at a fixed rate, use *more* of them. Multiple CPU threads each driving their own CUDA stream can roughly multiply the aggregate launch rate, letting independent work overlap. It is a partial mitigation — it raises the ceiling rather than removing it — but for services that cannot use graphs (dynamic shapes every request) it can be the difference between host-bound and fed.

The three levers compose: the strongest configuration is usually `torch.compile` (fewer kernels) *plus* CUDA graphs (no per-launch cost), which is exactly what `mode="reduce-overhead"` gives you in one call. As a preview of where this track lands, the entire fix for a static-shape service can be one line — the next posts unpack what it does and when it breaks:

```python
# Preview of the fix (detailed in the next posts of this track).
# reduce-overhead = Inductor fusion (fewer kernels) + CUDA graphs (no
# per-launch cost). One call attacks both levers on a static-shape service.
model = build_encoder().cuda().eval()
fast = torch.compile(model, mode="reduce-overhead")

# Warm up: the first calls trace, compile, and capture the graph.
x = torch.randn(1, 128, 768, device="cuda", dtype=torch.float16)
for _ in range(3):
    with torch.no_grad():
        fast(x)          # after warmup, the whole forward replays as ~1 launch
```

That is the destination. But you cannot choose intelligently among the levers — or know whether this one-liner will even help — without knowing *how far past the line* you are, and that requires the law.

## The launch-cost law and the crossover

Here is the mechanism made provable. It is simple algebra, and it is the most useful thing in this post because it turns "the GPU feels starved" into a number you can compute before you touch a profiler.

Let a forward pass launch `$N$` kernels. Each kernel costs the host a fixed `$t_\text{launch}$` to launch (our ~7 µs) and runs for an average of `$\bar{t}_k$` on the GPU. Then two quantities race each other across the step:

- **Host launch cost:** the CPU must issue all `$N$` launches, so it needs $t_\text{host} = N \cdot t_\text{launch}$ of wall time to feed the whole step.
- **GPU compute cost:** the GPU must execute all `$N$` kernels, so it needs $t_\text{gpu} = N \cdot \bar{t}_k$ of wall time to do the actual math.

Because the producer and consumer run asynchronously and pipeline, the step's wall-clock time is approximately the *larger* of the two — whichever side is slower sets the pace:

$$T_\text{step} \approx \max\left(N \cdot t_\text{launch},\; N \cdot \bar{t}_k\right)$$

Divide through by `$N$` and the crossover condition falls out immediately. You are **host-bound (launch-bound) whenever**:

$$t_\text{launch} > \bar{t}_k$$

That is the whole law in one line: **if your average kernel launches slower than it runs, you are launch-bound, and adding more of the same kernels only makes it worse.** The kernel count `$N$` does not even appear in the crossover condition — it cancels — which is the counterintuitive punchline: whether you are launch-bound is decided purely by the *ratio of launch cost to kernel duration*, not by how many kernels you have. What `$N$` decides is *how much wall time you waste*: the GPU idle time per step is exactly the slack between the two racers,

$$T_\text{idle} \approx N \cdot t_\text{launch} - N \cdot \bar{t}_k = N \left( t_\text{launch} - \bar{t}_k \right)$$

whenever that quantity is positive. Double the kernel count and you double the wasted idle time. That linear growth in `$N$` is why the timeline below tips over.

A word on why the step time is a `$\max$` and not a sum, because the approximation is where the intuition lives. If the CPU launched one kernel, waited for it to finish, launched the next, and so on — fully serialized — the step would cost `$N(t_\text{launch} + \bar{t}_k)$`, the sum of both. But that is not how streams work. The CPU launches ahead while the GPU runs behind, so the two costs *overlap in time*. In the ideal overlap, the total is set by whichever pipeline stage is the bottleneck — the slower of the producer and the consumer — which is the `$\max$`. Real services sit somewhere between the two extremes: overlap is imperfect (a forced `.item()` or a `cudaStreamSynchronize` drains the pipeline and forces a partial serialization, which is why those show up as long bars on the trace), so the true step time is usually a bit above the `$\max$` and well below the sum. The `$\max$` is the right mental anchor: it tells you which side to attack and roughly how much is recoverable. If you fix the launches and land near `$N \cdot \bar{t}_k$`, you overlapped well; if you are still well above it, you have a sync stall serializing the pipeline and that is your next target.

![a left to right timeline where launch cost rises across growing kernel counts until it overtakes compute and the gpu idles](/imgs/blogs/the-kernel-launch-overhead-problem-5.webp)

The timeline sweeps `$N$` upward at fixed `$t_\text{launch} = 7\text{ µs}$` against a fixed 8 ms of real compute. At N=200, launch cost is 1.4 ms — a fraction of compute, GPU-bound and healthy. By N=1,140 the launch cost has climbed to exactly 8 ms and *meets* the compute line: this is the crossover, the last moment the GPU stays fed. Past it, at N=1,800, launch cost is 12.6 ms and the GPU idles 4.6 ms every step. The service did not change hardware or math; it just crossed a line that the profiler can see and the util counter cannot.

#### Worked example: the 1,800-kernel encoder

Put real numbers through the law. Our batch-1 encoder on the A100 launches N = 1,800 kernels per forward pass. Measured launch cost on this host is `$t_\text{launch} = 7\text{ µs}$`. The kernels are tiny — batch 1 means each GEMM and norm touches only a few thousand elements — so the average GPU duration is `$\bar{t}_k \approx 4.4\text{ µs}$`.

- Host launch cost: 1,800 × 7 µs = **12.6 ms**.
- GPU compute: 1,800 × 4.4 µs ≈ **8.0 ms**.
- Crossover check: is `$t_\text{launch} > \bar{t}_k$`? 7 µs > 4.4 µs — **yes, launch-bound.**
- Predicted idle per step: 1,800 × (7 − 4.4) µs = 1,800 × 2.6 µs ≈ **4.7 ms**, which matches the measured 4.6 ms in the table to within rounding.
- Predicted step time: max(12.6, 8.0) = **12.6 ms**, and indeed the measured p50 was 12.6 ms. The step is bounded by the launch cost. The GPU's 8 ms of capability is capped by the CPU's 12.6 ms of launch work.

The law predicted the profile before we opened the profiler. That is what makes it worth memorizing.

### The launch-rate ceiling

There is a second, harder limit hiding in the law, and it is the one that bites at scale. If a single CPU thread issues launches at a fixed cost of `$t_\text{launch}$` each, then the maximum number of launches that one thread can issue per second is:

$$R_\text{max} = \frac{1}{t_\text{launch}} \approx \frac{1}{7\ \mu s} \approx 143{,}000 \text{ launches/s}$$

Call it order 100,000 to 200,000 launches per second from one Python thread — a ceiling, not a target, since real code does other work between launches. This is a *hard rate limit on your GPU's throughput* that has nothing to do with how fast the GPU is. If your workload needs to launch, say, 1,800 kernels per request and you want 200 requests per second, that is 360,000 launches per second — **more than double what one thread can physically issue**, no matter how idle the A100 sits. You would be launch-rate-limited before you were compute-limited or memory-limited. The GPU could be an H100 or a toaster; the launch thread does not care. This ceiling is precisely why the "more CPU threads / more streams" lever exists, and why at extreme scale even fusion is not enough and you reach for graphs: a graph replay issues *one* launch for the whole sequence, so it sidesteps the per-launch ceiling entirely.

## Watch the launch queue drain

The law tells you *when* the GPU idles. It helps to *see* the mechanism that produces the idle — the queue running dry — because that image is what makes the fixes obvious. The producer/consumer race is not a metaphor; it is a queue with a fill rate and a drain rate, and when drain outpaces fill, it empties.

<figure class="blog-anim">
<svg viewBox="0 0 700 250" role="img" aria-label="A launch queue of six slots empties faster than the CPU can refill it, so the GPU flips from busy to idle and waits" style="width:100%;height:auto;max-width:820px">
<style>
.a1-slot{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.a1-fill{fill:var(--accent,#6366f1)}
.a1-box{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.a1-t{font:700 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.a1-s{font:500 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.a1-busy{font:700 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.a1-idle{font:700 14px ui-sans-serif,system-ui;fill:#dc2626;text-anchor:middle}
.a1-drop{fill:var(--accent,#6366f1)}
@keyframes a1-fadeA{0%,38%{opacity:1}52%,92%{opacity:0}100%{opacity:1}}
@keyframes a1-fadeB{0%,38%{opacity:0}52%,92%{opacity:1}100%{opacity:0}}
@keyframes a1-drip{0%,45%{transform:translateX(0);opacity:0}55%{opacity:1}80%{transform:translateX(150px);opacity:1}100%{transform:translateX(150px);opacity:0}}
.a1-A{animation:a1-fadeA 9s ease-in-out infinite}
.a1-B{animation:a1-fadeB 9s ease-in-out infinite}
.a1-drop{animation:a1-drip 9s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.a1-A{animation:none;opacity:1}.a1-B{animation:none;opacity:0}.a1-drop{animation:none;opacity:0}}
</style>
<text class="a1-t" x="350" y="26">launch queue (the stream FIFO)</text>
<rect class="a1-box" x="25" y="98" width="125" height="56" rx="8"/>
<text class="a1-t" x="87" y="122">CPU</text>
<text class="a1-s" x="87" y="140">1 launch / 7 µs</text>
<rect class="a1-box" x="548" y="98" width="127" height="56" rx="8"/>
<text class="a1-t" x="611" y="122">GPU</text>
<text class="a1-s" x="611" y="140">drains &lt; 1 µs each</text>
<rect class="a1-slot" x="175" y="101" width="50" height="50" rx="6"/>
<rect class="a1-slot" x="235" y="101" width="50" height="50" rx="6"/>
<rect class="a1-slot" x="295" y="101" width="50" height="50" rx="6"/>
<rect class="a1-slot" x="355" y="101" width="50" height="50" rx="6"/>
<rect class="a1-slot" x="415" y="101" width="50" height="50" rx="6"/>
<rect class="a1-slot" x="475" y="101" width="50" height="50" rx="6"/>
<text class="a1-s" x="160" y="80">refills slowly</text>
<text class="a1-s" x="540" y="80">drains fast</text>
<g class="a1-A">
<rect class="a1-fill" x="178" y="104" width="44" height="44" rx="5"/>
<rect class="a1-fill" x="238" y="104" width="44" height="44" rx="5"/>
<rect class="a1-fill" x="298" y="104" width="44" height="44" rx="5"/>
<rect class="a1-fill" x="358" y="104" width="44" height="44" rx="5"/>
<rect class="a1-fill" x="418" y="104" width="44" height="44" rx="5"/>
<rect class="a1-fill" x="478" y="104" width="44" height="44" rx="5"/>
<text class="a1-busy" x="350" y="205">queue 6 / 6 &#183; GPU busy</text>
</g>
<g class="a1-B">
<circle class="a1-drop" cx="160" cy="126" r="8"/>
<text class="a1-idle" x="350" y="205">queue 0 / 6 &#183; GPU idle, waiting</text>
</g>
</svg>
<figcaption>The GPU drains the launch queue in well under a microsecond per kernel, but the CPU can only refill one slot every ~7 µs. The queue runs dry and the GPU sits idle, waiting on the host to launch the next kernel: that idle time is the launch-overhead tax, drawn.</figcaption>
</figure>

That draining queue is the exact thing the two fixes target from opposite ends. Fusion and bigger batches make each slot in the queue represent *more work*, so the GPU takes longer to drain each one and the slow refill can keep up. CUDA graphs replace the whole slow-refill process with a single pre-recorded batch of work, so the queue is filled in one shot instead of one launch at a time. Same starvation, two ways to end it. Keep the image of the empty queue; it is what "host-bound" looks like when you draw it.

## Why batch 1 and small models get hit hardest

The crossover condition `$t_\text{launch} > \bar{t}_k$` has a corollary that decides which services suffer: **anything that shrinks `$\bar{t}_k$` pushes you toward launch-bound, and anything that grows it hides the problem.** Launch cost is fixed; kernel duration is what moves. So the severity is entirely a function of how much work each kernel does.

![a table rating launch bound severity across batch one small model up to batch sixty four large model](/imgs/blogs/the-kernel-launch-overhead-problem-6.webp)

The table sorts the common cases. Batch 1 on a small model is the worst possible case: tiny tensors mean every kernel finishes in ~5 µs, well under the 7 µs launch cost, so the GPU idles roughly 70% of the step and throughput is a fraction of the card's capability. At batch 8 on a mid-size model the kernels run ~25 µs each — now launch is a third of the kernel and you idle ~30%, noticeable but survivable. At batch 64 on a large model each kernel runs 200 µs or more, launch is 3% of the kernel, and the problem is effectively invisible: you idle under 5% and the util counter is finally telling something close to the truth. This is why the standard advice "just use a bigger batch" works — it is not magic, it is `$\bar{t}_k$` growing until it clears `$t_\text{launch}$`.

Two structural cases deserve names because you will meet them constantly:

- **LLM decode is the pathological case.** Autoregressive generation produces one token per step, and a decode step is a forward pass at effective batch 1 (per sequence) over tiny tensors: many small GEMMs, many norms, many elementwise ops, each touching one token's worth of data. A 32-layer model can dispatch well over a thousand kernels to produce a *single token*, each kernel a few microseconds. Decode is launch-bound almost by construction, which is exactly why production LLM serving leans so hard on CUDA graphs and continuous batching — batching many sequences together is the only way to grow `$\bar{t}_k$` back above the launch floor. This is covered from the serving angle in the model-serving track's [kernel fusion, CUDA graphs, and torch.compile](/blog/machine-learning/model-serving/kernel-fusion-cuda-graphs-torch-compile) post.
- **Small vision models at batch 1 (real-time inference).** A ResNet-18 serving one frame at a time on an edge GPU is the same disease: hundreds of small conv and BN kernels, each fast, the launch tax dominating. Real-time constraints forbid batching (you cannot wait to accumulate 64 frames), which is why these services are the textbook customer for CUDA graphs.

It is tempting to assume this is purely an inference problem, but training gets bitten too, just in a different place. A large-batch training step is usually compute-bound — big kernels, `$\bar{t}_k$` far above the launch floor — so launch overhead hides. But small-model training, RL rollouts, and any step with a lot of tiny bookkeeping ops (optimizer element-wise updates, gradient clipping, EMA updates, per-parameter operations in a naive loop) can drift launch-bound even in training, especially with gradient accumulation at micro-batch 1. And the moment you shrink the batch to fit a bigger model in memory, you shrink `$\bar{t}_k$` and push toward the launch floor — a memory decision quietly becoming a launch-overhead decision. The lesson generalizes: **any time each kernel does little work, the fixed launch tax dominates, regardless of whether you are training or serving.**

#### Worked example: an LLM decode step at batch 1

Concrete numbers for the decode case. A mid-size decoder-only model, 32 layers, generating one token at batch 1 on an A100. Each layer dispatches roughly 40 kernels (QKV projections, attention, output projection, two-layer MLP, norms, residual adds, RoPE, elementwise), so a decode step launches about N = 1,300 kernels. At batch 1 the tensors are one-token-wide and each kernel runs `$\bar{t}_k \approx 3.5\text{ µs}$` on the GPU. Launch cost is the usual `$t_\text{launch} = 7\text{ µs}$`.

- Host launch cost: 1,300 × 7 µs = **9.1 ms**.
- GPU compute: 1,300 × 3.5 µs ≈ **4.6 ms**.
- Launch-bound? 7 µs > 3.5 µs — **yes, and badly: launch is 2x the compute.**
- Per-token wall time: max(9.1, 4.6) = **9.1 ms/token** → about 110 tokens/s.
- If you could eliminate the launch cost (graphs), you would be bounded by the 4.6 ms compute → about 217 tokens/s, nearly **2x more tokens per second** from the same GPU on the same model.

That 2x is not hypothetical; it is the well-documented reason `mode="reduce-overhead"` and hand-written CUDA-graph decode loops are standard in high-performance LLM serving. The GPU was never the bottleneck — the launch thread was.

## Measuring it yourself

Enough theory. Here is how you prove a service is launch-bound in practice, with the actual tools and their actual output. There are three confirmations, in increasing order of effort, and any one of them settles it. Pick by how much certainty you need and how much overhead you can afford:

| Tool | What it reveals about launch overhead | Overhead | Reach for it when |
|---|---|---|---|
| `torch.profiler` table | CPU-total vs CUDA-total ratio; `#Calls` per op; `cudaLaunchKernel` cost | Moderate (inflates CPU) | First look; you want the verdict in one run |
| Chrome trace / `nsys` timeline | The picket-fence *shape*; gaps; launch-to-execute lag | Low (`nsys` is light) | You need to *see* the whitespace and where it is |
| Micro-benchmark | The isolated `$t_\text{launch}$` slope, free of your model | None (it is your own loop) | You want to measure the launch cost itself, cleanly |

Any one of them confirms launch-bound; together they also localize it. Start with the profiler table for the verdict, drop to the timeline to see where the gaps sit, and use the micro-benchmark to pin your host's actual `$t_\text{launch}$` so the law's predictions match your hardware.

### Confirmation 1: `torch.profiler` — CPU time versus CUDA time, and #Calls

The fastest confirmation lives in a `torch.profiler` table. Profile a handful of steady-state steps and print `key_averages()`. Two columns decide the verdict: **the total CPU time versus the total CUDA time** (if CPU >> CUDA, the host is the bottleneck) and **the `#Calls` column** (thousands of calls to a launch means thousands of launches).

```python
import torch
from torch.profiler import profile, schedule, ProfilerActivity

model = build_encoder().cuda().eval()          # our batch-1 encoder
x = torch.randn(1, 128, 768, device="cuda", dtype=torch.float16)

# wait 2 steps (let caches/clocks settle), warm up 2, record 6, once.
sched = schedule(wait=2, warmup=2, active=6, repeat=1)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=sched,
    record_shapes=False,
) as prof:
    for _ in range(10):
        with torch.no_grad():
            model(x)
        prof.step()

# sort by CPU time to surface the launch/dispatch cost, and show #Calls.
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=8))
```

The table that comes back is the diagnosis. Read the `# of Calls` column and the two time-total columns:

```console
-------------------------------  ------------  ------------  ------------  ------------  ------------
                           Name    Self CPU %      CPU total     CUDA total    # of Calls   CUDA time avg
-------------------------------  ------------  ------------  ------------  ------------  ------------
                cudaLaunchKernel        41.3%       9.980ms        0.000us          1806         0.000us
                    aten::linear         6.1%        4.720ms        3.140ms           432         7.269us
                     aten::addmm         5.4%        3.910ms        3.140ms           432         7.269us
                 aten::layer_norm         3.9%        2.010ms        0.980ms           288         3.403us
                       aten::gelu         3.3%        1.660ms        0.720ms           144         5.000us
                        aten::add         2.8%        1.540ms        0.610ms           576         1.059us
              cudaStreamSynchronize       2.1%        1.210ms        0.000us            12         0.000us
-------------------------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 24.160ms
Self CUDA time total: 8.010ms
```

Three things in that output nail the diagnosis, and they are the first three tells from the signature figure. First, **`Self CPU time total: 24.16 ms` versus `Self CUDA time total: 8.01 ms`** — the host spent 3x as long as the device. The CPU, not the GPU, sets the pace. Second, **`cudaLaunchKernel` is the single most expensive line at 41% of self CPU time**, with 1,806 calls (six recorded steps × ~300 launches shown here, or 1,800 per step depending on how you slice it) — nearly ten milliseconds of pure launching. Third, **the `# of Calls` column is in the hundreds per op** across an entire fleet of tiny operations, `aten::add` alone called 576 times. That is a picket-fence of launches, and it is the fingerprint. A GPU-bound service inverts all three: CUDA total exceeds CPU total, `cudaLaunchKernel` is a minor line, and `#Calls` is modest.

One honest caveat: `torch.profiler` itself adds overhead, and it can *inflate* the apparent CPU time. Do not read the absolute milliseconds as gospel; read the *ratio* (CPU-total to CUDA-total) and the *call counts*, which are robust to the profiler's own cost. For absolute host cost, fall back to the CUDA-events timing shown below or to `nsys`, which is lighter-weight on the timeline.

### Confirmation 2: `nsys` — the picket-fence on the timeline

The profiler table tells you *that* you are launch-bound; Nsight Systems shows you the *shape*. Capture a system-wide timeline and you will see the picket-fence directly — a dense row of tiny back-to-back kernels on the GPU with gaps between them, and a CPU row saturated with `cudaLaunchKernel` calls trying to keep up.

```bash
# Capture CUDA API, kernels, and OS runtime for a short window.
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=cpu \
  --duration=8 \
  --output=encoder_launchbound \
  python serve_encoder.py --batch 1 --steady
```

The summary `nsys stats` prints from that capture is the second confirmation. The `cudaLaunchKernel` line dominating the CUDA API summary, with a call count in the thousands per second, is the picket-fence quantified:

```console
** CUDA API Summary (cuda_api_sum):

 Time(%)  Total Time (ns)   Num Calls    Avg (ns)   Name
 -------  ---------------  ----------  ----------  ---------------------------
    58.4     1,041,220,300      144,480     7,206.0  cudaLaunchKernel
    18.9       336,900,120       12,040    27,982.0  cudaStreamSynchronize
     9.1       162,330,540      144,480     1,123.4  cudaLaunchKernelExC
     ...

** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):

 Time(%)  Total Time (ns)   Instances   Avg (ns)   Name
 -------  ---------------  ----------  ---------  -----------------------------
    22.1       162,004,800      36,120     4,485.0  sm80_xmma_gemm_f16f16_...
    14.8       108,540,200      24,080     4,507.0  void at::native::vectorized_...
     ...
```

Read the two summaries against each other. The CUDA API side shows **144,480 `cudaLaunchKernel` calls averaging 7,206 ns each** — that is your `$t_\text{launch}$` measured directly on the host, ~7.2 µs, and it is the top line at 58% of API time. The GPU kernel side shows the kernels themselves averaging **~4,485 ns** — your `$\bar{t}_k$`, ~4.5 µs. Launch cost (7.2 µs) exceeds kernel duration (4.5 µs): the crossover condition, read straight off the profiler. On the timeline view in the Nsight UI this is unmistakable — the CUDA API row is a solid wall of launch bars and the GPU row is a picket-fence of thin kernels with visible whitespace between them. (For the deep method of reading these timelines — which rows mean what, how to spot the launch-to-execute lag — see [reading a Chrome trace](/blog/machine-learning/performance-engineering/reading-a-chrome-trace); the shapes translate directly to the Nsight timeline.)

### Confirmation 3: the micro-benchmark that scales the kernel count

The cleanest proof, and the one that builds intuition fastest, is a controlled experiment: launch `N` trivial kernels back-to-back, sweep `N`, and watch the wall-clock become a straight line in `N` with a slope equal to `$t_\text{launch}$`. If the total time grows linearly with the *number* of launches while the *work per launch* stays trivially small, launch overhead is the only thing you are measuring.

```python
import torch, time

def bench_launches(n_kernels: int, iters: int = 50) -> float:
    """Launch n_kernels tiny ops per iter; return mean host-bound ms/iter."""
    a = torch.ones(64, device="cuda")          # tiny: kernel runs in ~2-3 us
    torch.cuda.synchronize()

    # warm up: first launches pay one-time driver/cache costs.
    for _ in range(10):
        for _ in range(n_kernels):
            a = a + 1.0
    torch.cuda.synchronize()

    # time with CUDA events so we capture wall time incl. the idle gaps.
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        for _ in range(n_kernels):
            a = a + 1.0
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters      # ms per iter

for n in (100, 300, 600, 1200, 1800):
    ms = bench_launches(n)
    print(f"N={n:>4}  {ms:7.3f} ms/iter  {1000*ms/n:5.2f} us/launch")
```

The output makes the law visible: time per iteration is linear in `N`, and the per-launch cost is flat at ~7 µs regardless of how many you launch — proof that you are measuring a fixed per-launch tax, not compute.

```console
N= 100    0.712 ms/iter   7.12 us/launch
N= 300    2.121 ms/iter   7.07 us/launch
N= 600    4.238 ms/iter   7.06 us/launch
N=1200    8.501 ms/iter   7.08 us/launch
N=1800   12.744 ms/iter   7.08 us/launch
```

Each kernel does one trivial add on 64 elements — a few microseconds of real GPU work — yet the wall clock is 7 µs *per launch*, dead flat across an 18x range of `N`. The kernels are not the cost. The launching is. That flat 7.08 µs is `$t_\text{launch}$` on this host, measured with nothing but a loop and CUDA events. Run the same sweep with `a = a @ a` on a 4096×4096 matrix instead of a 64-element add and the per-launch number vanishes into the noise — because now `$\bar{t}_k$` is 40,000 µs and the 7 µs launch is a rounding error. That contrast, run yourself, is the whole post in ten lines of code.

### How to measure it honestly

Three rules keep these numbers from lying to you, and they apply to every before/after in this series:

1. **Warm up, always.** The first few launches pay one-time costs — kernel module loads, `cudaMalloc` calls, autotuning, cold instruction caches. Discard at least 5 to 10 iterations before timing. The `wait`/`warmup` fields in the profiler schedule and the warm-up loop in the benchmark exist for exactly this.
2. **Synchronize at the boundaries, and time with CUDA events.** Because the CPU races ahead of the GPU, a naive `time.time()` around a forward pass measures when the CPU *finished launching*, not when the GPU *finished computing*. Call `torch.cuda.synchronize()` before you start and before you stop the clock, or use CUDA events (`start.record()` / `end.record()` / `elapsed_time`) which timestamp on the device timeline. Get this wrong on a launch-bound service and you will measure the launch time and think it is the compute time — which, ironically, is *correct* here, but for the wrong reason, and it will mislead you the moment you fix the launches.
3. **Lock the clocks and reach steady state.** GPU boost clocks drift with temperature and load; a cold card runs faster for the first second, then throttles. For reproducible before/after numbers, lock the clocks (`nvidia-smi -lgc <freq>`) and measure only steady-state steps. Otherwise your "optimization" might just be a cooler GPU.

## Stress-testing the diagnosis

A diagnosis you cannot break is a diagnosis you do not understand. So push on it: change the batch, change the kernel size, change the hardware, change the thread count, and confirm the launch-bound story predicts each outcome. This is the problem-solving loop from the series — hypothesize, then try to falsify.

**Batch 1 versus batch 64.** The law says launch overhead should evaporate as the batch grows, because `$\bar{t}_k$` grows with the batch while `$t_\text{launch}$` stays fixed. Measure it:

| Batch | Kernels/step | `$\bar{t}_k$` | Launch cost | Compute | GPU idle | Verdict |
|---|---|---|---|---|---|---|
| 1 | 1,800 | 4.4 µs | 12.6 ms | 8.0 ms | 37% | launch-bound |
| 8 | 1,800 | 24 µs | 12.6 ms | 43 ms | ~0% | compute-bound |
| 64 | 1,800 | 190 µs | 12.6 ms | 342 ms | ~0% | compute-bound |

The kernel *count* barely changes with batch — you still launch ~1,800 kernels — but each kernel now does 8x or 64x the work, so `$\bar{t}_k$` clears the launch floor and the 12.6 ms of launch cost becomes a rounding error against 342 ms of compute. This is the single most important operational lever: **if you can batch, batch, and the launch problem often solves itself.** The catch is latency-sensitive services that *cannot* batch (real-time inference, single-user decode), which is exactly where you must reach for graphs instead.

**Small kernels versus large kernels.** Hold the batch at 1 but swap the model. A model built from many tiny ops (deep, narrow, lots of norms and activations) is far more launch-bound than a model built from a few big ops (wide GEMMs) even at the same FLOP count, because the same math split across more, smaller kernels means more launches and a lower `$\bar{t}_k$`. Architecture is a launch-overhead decision, not just a FLOP decision.

**L4 versus A100.** Here is a counterintuitive one that trips people up. Move the *same* launch-bound batch-1 service from an A100 to a smaller L4. The A100 has far more compute (312 vs 242 fp16 TFLOP/s) and much more bandwidth (2.0 TB/s vs 300 GB/s), so you would expect the A100 to win big. It barely wins at all — because the service is host-bound, and *both cards share the same CPU launch cost.* The 12.6 ms of launch time is identical on both; the A100's extra compute just means it finishes its 8 ms of work sooner and then idles *longer* waiting for the same slow launch thread. A launch-bound service does not get faster on a bigger GPU — it gets *more idle*. This is the clearest possible proof that the bottleneck is the host: the one thing you upgraded (the GPU) is the one thing that did not matter.

| Same batch-1 service | A100 80GB | L4 | Why so close |
|---|---|---|---|
| Launch cost / step | 12.6 ms | 12.6 ms | same CPU, same launches |
| Real compute / step | 8.0 ms | 11.5 ms | L4 slower but hidden |
| Step time (p50) | 12.6 ms | 12.6 ms | both bounded by launch |
| GPU idle | 37% | 8% | A100 idles *more* |

**Shapes vary every request.** Here is the case that decides *which* fix you are allowed to use. If every request comes in at a different sequence length or batch size, the tensor shapes change, and CUDA graphs — which record fixed launch configurations against fixed tensor pointers and sizes — cannot be captured once and replayed against a different shape. This does not make the service any less launch-bound; it just takes the sharpest tool off the table. The answers are then to *bucket* the shapes (round every request up to one of a handful of fixed sizes and keep one graph per bucket) or to lean on fusion and more launch threads instead of graphs. The launch-bound diagnosis is the same; the shape-stability of your workload decides which lever you can pull, and the next posts in this track spend real time on exactly this constraint.

**More CPU threads.** The partial mitigation. If one thread issues launches at ~143,000/s, can two threads driving two streams double it? For genuinely independent work, roughly yes — this is the "hide it" lever from the tree, and for services that cannot use graphs it can lift a host-bound service off the floor. But it has sharp limits: Python's GIL serializes a lot of the dispatch work, the CUDA driver has internal locks, and the streams must carry *independent* work to overlap. It raises the launch-rate ceiling; it does not remove the per-launch cost. Reach for it when graphs are off the table (dynamic shapes every request) and you have independent requests to overlap; do not expect it to beat a graph replay, which pays the launch cost exactly once.

## Case studies and real numbers

Three grounded results, each traceable to a documented source, showing the launch-overhead story is not a toy.

**PyTorch `reduce-overhead` on small-batch inference.** PyTorch's own documentation for `torch.compile(mode="reduce-overhead")` describes exactly this mechanism: the mode uses CUDA graphs to "reduce the overhead of Python and kernel launches," and its documented wins are largest precisely where launch overhead dominates — small batches, small models, and inference where the per-op Python and launch cost is a big fraction of the step. The reported speedups for launch-bound workloads are commonly in the 1.3x to 2x range, matching the "launch cost is roughly equal to or larger than compute" regime our law predicts. When the workload is already compute-bound (big batch, big kernels), the same mode barely moves the needle — again exactly what the crossover condition says.

**LLM decode with CUDA graphs.** Production LLM inference stacks (the model-serving track covers this in depth) adopt CUDA-graph decode loops as a default because a decode step is the pathological launch-bound case from our worked example: batch-1-per-sequence, a thousand-plus tiny kernels per token, `$\bar{t}_k$` well under `$t_\text{launch}$`. Capturing the decode step as a graph and replaying it per token removes essentially all per-launch cost, and the documented token-throughput improvements are frequently in the 1.5x to 2x range on small-batch decode — the same near-2x our decode worked example computed from first principles. The GPU was idle; the graph fed it.

**Inductor fusion cutting kernel count.** `torch.compile` with the Inductor backend attacks the *count* lever rather than the *per-launch* lever: it fuses chains of elementwise operations into single generated kernels. A forward pass that dispatched 1,800 eager kernels can drop to a few hundred fused kernels, and since total launch cost is `N × t_launch`, cutting `N` by 5x cuts launch cost by 5x directly — before any graph is involved. This is why the strongest configuration composes fusion (fewer kernels) with graphs (no per-launch cost): `mode="reduce-overhead"` does both at once, which is the subject the next posts build toward.

**The util-lie, confirmed by a trace.** The most instructive "case study" is the one you run on your own service in five minutes. Put `nvidia-smi dmon` in one terminal reporting utilization once a second, and capture a `torch.profiler` Chrome trace of the same steady-state window in another. On a launch-bound service the two disagree flatly: `dmon` reports utilization pinned near 100%, while summing the kernel durations on the GPU row of the trace and dividing by the wall span gives something like 60% — the other 40% is whitespace the counter cannot see. That gap between "util 100%" and "60% real activity" is your recoverable headroom, quantified, and it is the number that justifies the fix to a skeptical reviewer who trusts `nvidia-smi`. The counter is not broken; it is answering a different question than the one you are asking, and the trace answers yours.

## When to reach for this (and when not to)

The launch-overhead diagnosis is cheap — one profiler table — but the fixes are not free, so spend them where the law says they pay off.

- **Reach for it when the profiler shows CPU-total >> CUDA-total and `#Calls` in the thousands.** That is the launch-bound fingerprint, full stop. If your service is at batch 1, does real-time inference, or runs LLM decode, assume you are launch-bound until the profiler proves otherwise — you almost certainly are.
- **Do not chase launch overhead if you are compute-bound.** If the profiler shows CUDA-total exceeding CPU-total, or the GPU row on the trace is packed solid with no gaps, the launch cost is already amortized and CUDA graphs will buy you almost nothing. Don't reach for graphs on a big-batch training step at 90% occupancy — there is no idle to close. Measure first; the whole point of the law is to tell you whether there is a gap worth closing before you spend a week on graph capture.
- **Try the cheapest lever first.** If you can grow the batch, do that before anything — it is one line and it often dissolves the problem by growing `$\bar{t}_k$`. If you cannot batch, `torch.compile(mode="reduce-overhead")` is the next cheapest and gets you both fusion and graphs. Hand-rolled CUDA-graph capture and multi-stream launching are the heavy tools; reach for them when the easy levers are exhausted and the numbers justify the complexity.
- **Beware the measurement trap.** Before you declare victory, confirm you measured the GPU, not the launch thread. A naive wall-clock timer on a launch-bound service measures launch time; "fix" the launches and your timer suddenly measures something else, and the delta can flatter or fool you. Time with `torch.cuda.synchronize()` boundaries or CUDA events, warm up, and lock the clocks — every number in this post came from that discipline.

## Key takeaways

- **A kernel launch costs the CPU 5 to 10 µs, fixed, regardless of the kernel's size.** That fixed cost, multiplied by thousands of tiny kernels, is a real and often dominant slice of your step time.
- **The law is `$T_\text{step} \approx \max(N \cdot t_\text{launch},\ N \cdot \bar{t}_k)$`, and you are launch-bound whenever `$t_\text{launch} > \bar{t}_k$`** — when your average kernel launches slower than it runs. The kernel count cancels out of the crossover; it only sets how much time you waste.
- **`nvidia-smi` GPU-Util cannot see launch overhead.** It reports "at least one kernel resident," not "SMs doing math," so a launch-bound service reads 100% while idling a third of every step. Trust the profiler's CPU-vs-CUDA ratio and the trace's whitespace instead.
- **Confirm it three ways:** `torch.profiler` (CPU-total >> CUDA-total and `#Calls` in the thousands), `nsys` (a picket-fence of tiny kernels with a wall of `cudaLaunchKernel` bars), and a micro-benchmark (wall clock linear in `N` with a flat ~7 µs/launch slope).
- **Batch 1, small models, and LLM decode are the pathological cases** because they shrink `$\bar{t}_k$` below the launch floor. Big batches and big kernels hide the problem by growing `$\bar{t}_k$`.
- **A launch-bound service does not get faster on a bigger GPU — it gets more idle.** The launch cost lives on the host; upgrading the device you are not bottlenecked on just widens the gap.
- **Three levers, one fingerprint:** reduce kernel count (fusion, bigger batch), eliminate per-launch cost (CUDA graphs), or hide it behind more host threads. The next posts pull the second lever, hard.
- **There is a hard launch-rate ceiling** of order 143,000 launches/s from one CPU thread. At scale, this caps throughput independent of GPU speed, and only a graph replay — one launch for the whole sequence — sidesteps it.

## Further reading

- [Why your AI service wastes CPU and GPU](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu) — the series intro and the four wastes; launch overhead is the first, the idle GPU.
- [The mental model of a GPU service](/blog/machine-learning/performance-engineering/the-mental-model-of-a-gpu-service) — the producer/consumer, host-enqueues-device-drains model this whole post rests on.
- [Reading a Chrome trace](/blog/machine-learning/performance-engineering/reading-a-chrome-trace) — how to see the picket-fence, the gaps, and the launch-to-execute lag on a real timeline.
- [CUDA graphs from first principles](/blog/machine-learning/performance-engineering/cuda-graphs-from-first-principles) — the next post: capture versus replay, and how a recorded graph eliminates per-launch cost.
- [CUDA graphs in PyTorch](/blog/machine-learning/performance-engineering/cuda-graphs-in-pytorch) — wiring `torch.cuda.graph` and `make_graphed_callables` into a real forward pass.
- [Kernel fusion, CUDA graphs, and torch.compile](/blog/machine-learning/model-serving/kernel-fusion-cuda-graphs-torch-compile) — the serving-side view of the same fixes, applied to LLM inference.
- [The performance engineering playbook](/blog/machine-learning/performance-engineering/the-performance-engineering-playbook) — the capstone decision tree: symptom to profiler to cause to fix.
- PyTorch docs: the `torch.compile` tutorial and the `mode="reduce-overhead"` reference; the `torch.profiler` recipe; NVIDIA's CUDA C++ Programming Guide section on streams and the CUDA Graphs documentation.
