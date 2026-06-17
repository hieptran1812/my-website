---
title: "Profiling GPU Workloads: Finding the Real Bottleneck"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Stop guessing why your training run is slow. Learn to read nsys, ncu, and torch.profiler like an instrument panel, find the actual wall — a starving data loader, an exposed all-reduce, an fp32 fallback, a memory-bound kernel — and score every fix by the one metric that matters, MFU."
tags:
  [
    "high-performance-computing",
    "gpu",
    "profiling",
    "nsight",
    "torch-profiler",
    "mfu",
    "roofline",
    "a100",
    "h100",
    "deep-learning",
    "ml-systems",
  ]
category: "machine-learning"
subcategory: "High Performance Computing"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/profiling-gpu-workloads-finding-the-real-bottleneck-1.png"
---

Here is the most expensive mistake I see AI engineers make, and I have made it myself more than once. A training run is slow. The obvious move feels like *optimize the model*: rewrite the attention, try a bigger batch, switch to a fancier optimizer, maybe buy a faster GPU. So you spend a week doing exactly that, ship the change, and the step time barely moves. The reason it barely moved is that you optimized the part of the run that was never the problem. The GPU was spending a third of every step *idle*, sitting on its hands waiting for the CPU to hand it the next batch of data — and no amount of faster math fixes a chip that is waiting for groceries.

The discipline that prevents that week of wasted work has a name, and it is the single highest-leverage skill in GPU performance: **profiling**. Profiling means measuring where the time actually goes, in real units, on real hardware, instead of reasoning about where you *think* it goes. A profiler is a tool that records what the GPU and CPU were doing, microsecond by microsecond, and lets you read it back. The first law of performance work is *measure, don't guess*, and the reason it is a law and not a slogan is that human intuition about where a program spends its time is wrong far more often than it is right. The bottleneck is almost never where it feels like it should be. I have watched a senior engineer spend three days hand-tuning a matmul that turned out to account for four percent of the step.

This post teaches you to find the real bottleneck with three tools — **Nsight Systems** (`nsys`) for the whole-system timeline, **Nsight Compute** (`ncu`) for the detail inside a single kernel, and PyTorch's built-in **`torch.profiler`** for the framework-level view of which operation cost what. We will define every piece of jargon the first time it shows up, build the one north-star metric that scores every change (**MFU**, model FLOPs utilization), and then walk the four findings that account for the overwhelming majority of slow training runs: a starving data loader, an exposed all-reduce, an fp32 fallback, and a memory-bound kernel. Figure 1 is the loop the whole post lives inside — you do not optimize once, you *loop*: profile, find the wall, fix one thing, re-measure MFU, repeat until you are near the hardware's real ceiling.

![diagram of the optimization loop showing profile then find the wall then fix one thing then re-measure MFU then repeat](/imgs/blogs/profiling-gpu-workloads-finding-the-real-bottleneck-1.png)

Why does this skill matter so much that it gets its own post? Because GPU time is the single largest line item in modern AI, and most of it is wasted by default. An A100 rents for a few dollars an hour (call it \$2–\$4 per GPU-hour) and a large training run burns tens of thousands of GPU-hours; an H100 cluster for a frontier model burns millions of dollars of compute. If your run sits at 18% MFU instead of 45%, you are paying 2.5× more for the same result — not because the hardware is slow, but because four-fifths of it is idle in the way that counts, and nobody measured. The profiler is the instrument that converts "the run feels slow" into "the data loader stalls the GPU for five milliseconds every step, here is the bar, here is the fix, MFU went from 18 to 31." That conversion — from vague to measured to fixed to re-measured — is the most valuable habit an AI engineer can build, and it generalizes: the same loop that fixes a training run fixes an inference server, an embedding pipeline, or a data-preprocessing job. Learn it once on a Transformer and you own it everywhere.

This is a post in the High-Performance-Computing series, and it sits right at the hinge. The earlier posts gave you the physics: the [intro on why HPC is the bottleneck for modern AI](/blog/machine-learning/high-performance-computing/why-hpc-is-the-bottleneck-for-modern-ai) laid out the three walls — compute, memory bandwidth, and communication — that every workload eventually hits; the [roofline post](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) gave you the chart that says which wall a given operation is stuck against; and the [memory hierarchy post](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm) explained where the bytes actually live and why moving them is the expensive part. Profiling is how you connect that theory to *your* run. The roofline tells you a LayerNorm is memory-bound in principle; the profiler tells you it is the third-biggest line in *your* step and worth fusing today. By the end you will be able to take a real, slow training job, profile it, name the wall in measured units, fix it, and prove the fix worked with a number — going, in the worked case we carry throughout, from 18% MFU to 45% MFU on an A100 by fixing a data loader, enabling bf16, and overlapping communication.

## Why "the GPU is at 100%" tells you almost nothing

Let us kill the most dangerous false signal first, because it sends more people down the wrong path than anything else. You run `nvidia-smi`, you see `GPU-Util: 100%`, and you conclude the GPU is fully used and the bottleneck must be elsewhere. That conclusion is wrong, and understanding *why* it is wrong is the gateway to everything else.

`nvidia-smi`'s utilization number answers a very narrow question: *over the last sampling window, was at least one kernel running on the GPU at any point?* A **kernel** is a single function the GPU executes — one matmul, one LayerNorm, one elementwise add is each a separate kernel launch. The utilization counter goes to 100% if a kernel was active during the sample, full stop. It says nothing about whether that kernel was doing useful arithmetic or stalled waiting on memory, nothing about whether the kernel was occupying 100% or 5% of the chip's parallel capacity, and nothing about whether the *gaps between* kernels are eating your step. A GPU running one tiny memory-bound kernel that uses three percent of its math units, back to back, will happily report 100% utilization while delivering a few percent of its real throughput.

So we need a metric that measures *useful work*, not mere busyness. That metric is **MFU**, model FLOPs utilization. It is the fraction of the GPU's advertised peak math throughput that your model's actual, necessary arithmetic achieves:

$$\text{MFU} = \frac{\text{model FLOP/s achieved}}{\text{peak FLOP/s of the hardware}}$$

The numerator is the floating-point operations your model *must* do per second to make progress — the matmuls in attention and the MLP, the operations the math of a Transformer demands regardless of how you implement them. The denominator is the datasheet number for the chip, for example 312 teraFLOP per second of bf16 matmul on an A100, or about 989 teraFLOP per second on an H100 SXM. (One quick definition we will lean on constantly: **FLOPs** with a lowercase "s" is a *count* of operations; **FLOP/s** is a *rate*, operations per second. Intensity and totals use the count; throughput and the MFU denominator use the rate.) If your model needs to do 56 trillion useful floating-point operations per second to keep up with your step time, and the chip is rated for 312 trillion, your MFU is 18%. Four-fifths of the silicon is idle in the sense that matters, even while `nvidia-smi` shows a confident green 100%.

MFU is the north-star metric for this entire post because it is honest in a way utilization is not. It cannot be gamed by running a tiny kernel in a busy loop. It does not care how the GPU spends its time — it only rewards time spent on the math the model actually needs. Every fix in this post is scored by the same question: *did MFU go up?* If it did, the fix was real. If it did not, you optimized the wrong thing, and the profiler will tell you what the right thing was.

There is one honest subtlety worth stating up front. MFU compares against *model* FLOPs — the arithmetic the model definitionally requires — not against every FLOP your implementation happens to execute. A naive attention that recomputes a giant intermediate does more *hardware* FLOPs than it needs, and a metric that counted those would flatter a wasteful kernel. By scoring against the necessary work, MFU correctly punishes waste: a wasteful kernel takes longer, so the *necessary* FLOPs per second drop, so MFU drops. This is exactly the behavior you want from a north-star number, and it is why the field — Google's PaLM paper, the Megatron-LM scaling work, every serious training report — quotes MFU rather than utilization.

To feel how far apart utilization and MFU can drift, take a deliberately pathological example. Suppose your training loop launches, for every useful matmul, a chain of twenty tiny elementwise kernels — a reshape, a couple of adds, a cast, a mask, each its own launch — and each tiny kernel uses about three percent of the chip's math capacity but keeps a kernel "active" on the GPU. `nvidia-smi` samples the GPU, sees a kernel running, and reports utilization near 100% the whole time. Meanwhile those twenty tiny kernels collectively burn most of the step doing almost no useful arithmetic, so the necessary matmul FLOP/s — the MFU numerator — is small and your MFU sits at 15%. Same wall-clock, same green bar, two metrics telling opposite stories. The gap between them *is* the optimization opportunity: utilization measures occupancy of the clock, MFU measures occupancy of the *math units*, and the difference is precisely the glue and idle you are hunting. This is also why "my GPU is at 100%, the problem must be elsewhere" sends so many engineers to optimize the wrong system entirely — they trust the metric that cannot see the waste.

## The three-tool ladder: nsys, torch.profiler, ncu

You do not reach for one profiler; you reach for three, each at a different zoom level, and you use them in order from coarse to fine. Skipping straight to the finest tool is the second-most-common mistake after trusting `nvidia-smi` — you end up obsessing over a single kernel's occupancy while a five-millisecond idle gap, invisible at that zoom, eats half your step. Figure 2 is the ladder.

![diagram of the three-tool profiling ladder from nsys at the system level down through torch profiler to ncu at the kernel level](/imgs/blogs/profiling-gpu-workloads-finding-the-real-bottleneck-2.png)

The top of the ladder is **Nsight Systems**, invoked as `nsys`. It is a *system-level* profiler: it records a timeline of everything — every CUDA kernel, every host-to-device memory copy, every CPU thread, every synchronization point — laid out left to right in time. A **timeline** is exactly what it sounds like: a horizontal track per resource (the GPU, the CUDA streams, each CPU thread) with colored bars showing what ran when. The thing `nsys` is uniquely good at is showing you the *gaps* — the stretches where the GPU has nothing to do because the CPU has not handed it the next batch, or because it is blocked on a synchronization. Those gaps are invisible to any tool that only looks inside kernels, and they are, in my experience, the single most common cause of a low MFU. When MFU is bad and you have no idea why, you start here.

The middle of the ladder is **`torch.profiler`**, PyTorch's own profiler. It sees the world the way you wrote it — in terms of PyTorch operations (`aten::mm`, `aten::layer_norm`, `aten::add`) rather than raw CUDA kernels. Its great strength is the split between **CPU time** and **CUDA time** for each operation: it tells you that `aten::mm` cost 3.1 milliseconds of GPU time but only 40 microseconds of CPU time, while some Python-side operation cost 2 milliseconds of CPU time and launched no GPU work at all. That split is how you find a **bubble** — a stretch where one side (usually the GPU) is idle while the other side does serial work it could have overlapped. `torch.profiler` also emits a Chrome/Perfetto **trace**, a JSON file you open in a browser to scrub the timeline visually, the same way `nsys` shows it but annotated with your PyTorch op names. (A **trace** is just the recorded log of timed events; "the trace" and "the timeline" refer to the same data viewed two ways.)

The bottom of the ladder is **Nsight Compute**, invoked as `ncu`. It is a *single-kernel* microscope. You point it at one kernel and it tells you everything about that kernel's execution: its arithmetic intensity, where it lands on the roofline, its **occupancy** (the fraction of the GPU's parallel slots that were actually filled — we will define this carefully in its own section), its achieved memory bandwidth in gigabytes per second versus the chip's peak, and a breakdown of what stalled the warps cycle by cycle. `ncu` is expensive — it replays a kernel many times to gather all its counters, so it can slow a kernel by 100× or more — which is exactly why you do not run it on your whole program. You use `nsys` and `torch.profiler` to find the *one* kernel that matters, then aim `ncu` at that single kernel.

There is a fourth, coarsest rung worth naming so you know its place: `nvidia-smi` and the data-center monitoring tool **DCGM** (`dcgmi`). These are for *monitoring*, not *profiling* — they sample coarse aggregates like utilization, memory used, power draw, and clock speed once a second or so. They are how you notice a GPU is thermally throttling its clocks or running out of memory across a whole cluster. They are useless for finding why one step is slow. Keep them on the dashboard; do not debug a kernel with them.

| Tool | Zoom level | The question it answers | When to reach for it |
| --- | --- | --- | --- |
| `nsys` (Nsight Systems) | Whole system, all streams + CPU | Where are the idle gaps and sync stalls? | First, whenever MFU is low and you don't know why |
| `torch.profiler` | PyTorch ops, CPU vs CUDA time | Which op costs what, and is the CPU the bubble? | Second, to attribute time to named ops |
| `ncu` (Nsight Compute) | One kernel, cycle-level counters | Is this kernel memory- or compute-bound, and why? | Last, on the one kernel that matters |
| `nvidia-smi` / `dcgmi` | Coarse aggregates, 1 Hz | Is the box healthy — throttling, OOM, power? | For monitoring, never for finding a bottleneck |

The discipline is *always* top-down. Run `nsys` on the whole step first to see the shape of the time — is the GPU mostly idle, mostly in comms, or mostly busy? That answer routes you. If the GPU is idle, you have a feeding problem (data loader or sync) and you stay at the `nsys` level. If the GPU is busy but MFU is still low, you have a kernel-efficiency problem, and *now* you drop to `torch.profiler` to find which op dominates, then to `ncu` to dissect that op. Going the other way — starting at `ncu` — is like inspecting one brick when the question is whether the building is on fire.

## The science of MFU: turning a step time into a percentage

Before we profile anything, you need to be able to *compute* MFU from quantities you can measure, because that number is how you score every fix. The whole calculation is two pieces: how many FLOPs your model does per step (a property of the model and batch you can compute on paper), and how long the step takes (a number the profiler hands you).

The model-FLOPs side has a famous and genuinely useful approximation. For a dense Transformer with $N$ parameters, processing one token in the forward pass costs about $2N$ FLOPs — the factor of 2 because each parameter participates in one multiply and one add (a multiply-accumulate). The backward pass costs about twice the forward, because you compute gradients with respect to both the activations and the weights. So forward plus backward is about $6N$ FLOPs per token. Multiply by the number of tokens in a step and you have the model FLOPs:

$$\text{model FLOPs per step} \approx 6 \, N \, \times \, (\text{tokens per step})$$

This $6N$ rule is the workhorse of every MFU calculation in the literature; it appears in the Kaplan scaling-laws paper, the Chinchilla paper, and the PaLM report, all of which use it to report training efficiency. It is worth seeing *why* the constant is 6 and not some other number, because the derivation tells you exactly what it counts and what it leaves out. Take a single linear layer with a weight matrix of $P$ parameters. Multiplying an input vector by that matrix is $P$ multiplies and $P$ adds — $2P$ FLOPs — for one token, by the multiply-accumulate argument. Sum over every weight matrix in the model and the forward pass is $2N$ FLOPs per token, where $N$ is the total parameter count. The backward pass computes two gradients per layer: the gradient with respect to the layer's *input* (so the signal can flow further back) and the gradient with respect to the layer's *weights* (so the optimizer can update them). Each of those is another matrix multiply of the same size as the forward, so backward is $2 \times 2N = 4N$ FLOPs per token. Forward plus backward: $2N + 4N = 6N$. The 6 is "one forward, two backward," nothing more mysterious.

What the $6N$ rule deliberately ignores is the attention score computation — the $\mathcal{O}(s^2)$ term for sequence length $s$, where every token attends to every other token. That cost is *not* in the parameter count (the score matrix is data, not weights), so $6N$ misses it. For a model where the per-layer attention FLOPs $\approx 2 \cdot s \cdot d$ per token (with $d$ the model dimension) are small next to the MLP's $6N/L$ per layer, the omission is a few percent and you ignore it. But for long context — sequence length in the tens of thousands — the quadratic term can rival or exceed the linear term, and a clean MFU report adds the attention FLOPs explicitly: $\text{FLOPs} \approx 6N \cdot T + 12 \cdot L \cdot s^2 \cdot d \cdot (\text{batch})$, with $L$ layers. For the running example we keep the $6N$ approximation and flag that at very long context it under-counts the necessary work — which makes your *measured* MFU look artificially low because you divided real achieved FLOP/s by an under-estimate of the work. Knowing which term you dropped keeps the number honest.

Now divide. If a step does $6N \cdot T$ FLOPs (with $T$ tokens) in $t$ seconds, the achieved rate is $6NT/t$ FLOP per second, and MFU is that over the peak:

$$\text{MFU} = \frac{6 \, N \, T}{t \times P_\text{peak}}$$

That is the entire formula. Everything on the right is measurable: $N$ is your parameter count, $T$ is batch size times sequence length, $t$ is the median step time from the profiler, and $P_\text{peak}$ is the datasheet number for your GPU at your precision. Let us put real numbers on it.

#### Worked example: MFU of a 7B model on one A100

Take a 7-billion-parameter Transformer ($N = 7 \times 10^9$) training on a single A100 80GB SXM, whose peak bf16 matmul throughput is $P_\text{peak} = 312 \times 10^{12}$ FLOP/s. Suppose your batch is 8 sequences of length 2048, so $T = 8 \times 2048 = 16{,}384$ tokens per step, and the profiler reports a median step time of $t = 3.0$ seconds.

Model FLOPs per step: $6 \times 7\times10^9 \times 16{,}384 \approx 6.88 \times 10^{14}$ FLOPs.

Achieved rate: $6.88\times10^{14} / 3.0 \approx 2.29 \times 10^{14} = 229$ TFLOP/s.

$$\text{MFU} = \frac{229}{312} \approx 0.73 = 73\%$$

Seventy-three percent MFU would be an *excellent* result — large, well-tuned dense training runs report MFU in the 40–55% range, and getting past 50% on a single GPU usually means your matmuls dominate and everything else is overlapped. Now flip it: if that same step took $t = 12.4$ seconds instead of 3.0, the achieved rate is $55$ TFLOP/s and MFU is $55/312 \approx 18\%$. Same model, same hardware, same necessary FLOPs — the *only* difference is wall-clock step time, and step time is exactly what the profiler dissects. The whole game is finding which slice of those 12.4 seconds is not matmul, and moving it.

Here is the Python you would actually run to compute this in your training loop. Notice it times the step with CUDA events and synchronizes correctly — measuring GPU time from the CPU clock without a synchronize is the single most common timing bug, because kernel launches are *asynchronous*: the CPU queues the work and races ahead, so the CPU-side timer stops before the GPU has finished.

```python
import torch

def mfu(num_params, tokens_per_step, step_time_s, peak_flops):
    """Model FLOPs utilization from the 6N rule."""
    model_flops = 6 * num_params * tokens_per_step
    achieved = model_flops / step_time_s
    return achieved / peak_flops

# Time a step correctly: events on the GPU stream, synchronize before reading.
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

torch.cuda.synchronize()          # drain anything still queued
start.record()
loss = model(batch).loss          # forward
loss.backward()                   # backward
optimizer.step(); optimizer.zero_grad()
end.record()
torch.cuda.synchronize()          # wait for the GPU to actually finish
step_ms = start.elapsed_time(end) # milliseconds, GPU-side

A100_PEAK = 312e12               # bf16 TFLOP/s, NVIDIA A100 datasheet
print(f"step {step_ms:.1f} ms  "
      f"MFU {mfu(7e9, 8*2048, step_ms/1e3, A100_PEAK):.1%}")
```

Two measurement disciplines are baked into that snippet and you must keep both. First, **warm up** before you trust any number: the first several steps include CUDA context creation, cuDNN autotuning, and `torch.compile` graph capture, all one-time costs that have nothing to do with steady-state speed. Throw away the first 10–20 steps and report the *median* of the next 50, not the mean (the mean is dragged around by the occasional slow step from a checkpoint write or a Python garbage-collection pause). Second, **synchronize** around the region you time, as shown, or your timer measures launch latency instead of execution. Get these two wrong and every MFU number you compute is fiction.

### Why honest measurement is harder than it looks

The reason GPU timing is a minefield deserves a full paragraph, because nearly every wrong performance number I have seen traces to one of three timing bugs. The root cause is *asynchrony*: a CUDA kernel launch is non-blocking. When you write `loss = model(batch).loss`, the CPU does not wait for the GPU to finish — it queues the kernels onto a CUDA stream and immediately returns, racing ahead to launch the next ones. This is by design and it is *good*: it lets the CPU keep the GPU's queue full so the GPU never starves. But it means the CPU clock and the GPU clock are decoupled, and naively timing with `time.time()` around a GPU operation measures how long it took to *queue* the work, not how long the work *ran*. A matmul that takes 3 milliseconds on the GPU might take 40 microseconds to queue; if you time the launch, you will report 40 microseconds and conclude your matmul is blazing fast, which is nonsense.

The three timing bugs, then. **Bug one: no synchronize.** Timing with the CPU clock and forgetting `torch.cuda.synchronize()` measures launch latency. Fix: use CUDA events (`Event.record()` and `Event.elapsed_time()`), which are placed *into the stream* and measure GPU-side wall time, or call `synchronize()` before stopping a CPU timer. **Bug two: no warm-up.** The first step pays for context init and autotuning; including it inflates your average by tens of milliseconds. Fix: discard the first 10–20 iterations. **Bug three: an accidental synchronize *inside* the timed region.** A `loss.item()`, a `.cpu()`, a `print(tensor)`, or an `assert` on a tensor value forces a sync mid-step, which serializes the CPU and GPU and *changes the thing you are measuring* — the profiled run is slower than the real run because your measurement code added a stall. Fix: keep scalar reads out of the hot loop, or accept that the profiled number is a pessimistic bound. These three are not exotic; they are the default mistakes, and a senior engineer's habit is to assume any too-good or too-noisy number is one of them until proven otherwise.

There is a fourth confound that is not a timing bug but a *physics* one: **thermal and power throttling**. A GPU under sustained load heats up, and when it crosses a temperature or power threshold it lowers its clock speed to stay within its limit. Your first 30 steps might run at the boosted clock and your steady-state at a 10–15% lower clock, so a short benchmark over-reports throughput. `nvidia-smi -q -d CLOCK` shows the current versus maximum clocks; `dcgmi dmon` streams clock, temperature, and power over time. If your measured MFU drifts down over a few minutes of running, you are watching the clock throttle, and the *honest* number is the steady-state one, not the first warm-and-cool burst. Always benchmark for long enough to reach thermal steady state, and always report the clock you measured at.

## Nsight Systems: reading the system timeline

Now we profile for real, top-down, starting with `nsys`. The invocation is one line. You wrap your normal training script and tell `nsys` which categories of events to capture:

```bash
nsys profile -t cuda,nvtx,osrt --capture-range=cudaProfilerApi \
  -o train_profile --force-overwrite true \
  python train.py --steps 20
```

The `-t cuda,nvtx,osrt` flag is the important part. `cuda` captures kernel launches, memory copies, and stream activity; `nvtx` captures the named ranges you annotate in your code (more on NVTX in a moment); `osrt` captures OS runtime calls — `pthread` waits, file reads, the things the *CPU* blocks on, which is exactly what you need to see a data-loader stall. The output is a `.nsys-rep` file you open in the Nsight Systems GUI, where you get the timeline. Profile only a handful of steps (`--steps 20`) — the trace file grows fast, and ten steady-state steps tell you everything.

What you are reading on that timeline is the relationship between the CPU rows and the GPU rows over time. Figure 3 shows the canonical shape and the one pattern you are hunting for.

![diagram of an nsys system timeline showing CPU batch prep feeding an H2D copy then CUDA kernels with a five millisecond GPU idle gap caused by the data loader](/imgs/blogs/profiling-gpu-workloads-finding-the-real-bottleneck-3.png)

Read the figure as time flowing left to right. The CPU thread spends 9 milliseconds preparing a batch — decoding, tokenizing, collating. Then a **host-to-device copy** (H2D, moving the batch from CPU RAM into GPU memory) takes 0.4 milliseconds. Then the CUDA kernels run — the matmuls, 3.1 milliseconds of real work. And then, the tell: a 5-millisecond **idle gap** on the GPU row, a stretch where the GPU has *no kernel running at all*, lined up exactly with the CPU thread starting to prepare the *next* batch. That gap is the signature of a starving data loader. The GPU finished its work, asked for the next batch, and the CPU was not ready, so the GPU sat idle for 5 milliseconds out of an 8.5-millisecond effective step. That is 5/8.5 ≈ 59% of the step *thrown away* — and it caps MFU at 41% before you have done anything else, no matter how fast your kernels are.

This is the science worth internalizing, and it is why profiling beats kernel-tuning so often: **a fraction $f$ of GPU-idle time caps your MFU at $(1 - f)$ regardless of how fast the busy part runs.** If 30% of every step is idle, the very best MFU you can reach — even with kernels running at the hardware's absolute peak — is 70%. You cannot tune your way past idle time by speeding up the work that is *not* the idle time. The only fix for a gap is to fill the gap, and you fill it by overlapping the work that caused it. A faster matmul shrinks the 3.1-millisecond busy bar; it does nothing to the 5-millisecond hole. This single inequality is why "the GPU is at 100%" is such a trap and why you look at the timeline first.

The other gap shape `nsys` reveals is the **sync bubble**: a place where the GPU is idle not because data is missing but because the code forced a synchronization. Every `.item()`, every `.cpu()`, every `print(loss)` that reads a tensor value, every `torch.cuda.synchronize()` forces the CPU to stop and wait for the GPU to drain — and worse, it stops the CPU from *racing ahead to launch the next batch of kernels*, so the GPU's launch queue runs dry and a bubble opens. On the `nsys` timeline these show up as periodic small gaps locked to a host-side blocking call. The fix is usually to stop reading scalars off the GPU every step (log the loss every 50 steps, or accumulate it on-device and read once).

### NVTX: putting your own labels on the timeline

A raw `nsys` timeline is a sea of anonymous kernel names. To make it readable, you annotate your code with **NVTX ranges** — named markers that show up as labeled bars on the timeline, so you can see "data loading," "forward," "backward," and "optimizer" as distinct colored regions instead of guessing which kernels belong to which phase. NVTX (NVIDIA Tools Extension) is a tiny API; PyTorch exposes it through `torch.cuda.nvtx`. Annotating one training step takes four lines:

```python
import torch

for step, batch in enumerate(loader):
    torch.cuda.nvtx.range_push("data_to_gpu")
    batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("forward")
    loss = model(**batch).loss
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("backward")
    loss.backward()
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("optimizer")
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    torch.cuda.nvtx.range_pop()
```

Run `nsys` on that and the timeline now has four labeled bands per step. The instant you see the `data_to_gpu` band stretch wide while the GPU rows go quiet, you have *seen* the data-loader stall rather than inferred it. This is the difference between profiling and guessing: the bottleneck stops being a hypothesis and becomes a measured, labeled bar you can point at.

## torch.profiler: which op, CPU time vs CUDA time

`nsys` told you the *shape* of the time — idle here, busy there. `torch.profiler` tells you *which PyTorch operation* owns each slice, and critically, splits each op's cost into CPU time and CUDA time. Here is the canonical block. The `schedule` controls when it records (skip warm-up, then capture a few steady steps); the `tensorboard_trace_handler` writes a trace you can open in TensorBoard or drag into Perfetto; `record_function` lets you name custom regions just like NVTX:

```python
import torch
from torch.profiler import profile, schedule, ProfilerActivity, record_function, tensorboard_trace_handler

prof_schedule = schedule(wait=5, warmup=3, active=5, repeat=1)  # skip 5, warm 3, record 5

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=prof_schedule,
    on_trace_ready=tensorboard_trace_handler("./tb_traces"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step, batch in enumerate(loader):
        with record_function("forward"):
            loss = model(**batch).loss
        with record_function("backward"):
            loss.backward()
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        prof.step()  # advance the schedule each iteration

# Print the heaviest ops by GPU time.
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
```

The line that earns its keep is the last one. `prof.key_averages().table(sort_by="cuda_time_total")` prints a ranked table of every operation, sorted by how much *GPU* time it consumed in total. Reading that table is a skill, so let us read a real-shaped one. Suppose your slow 7B step prints something like this (rounded, representative of a real fp32-fallback run):

```bash
-------------------------------  ------------  ------------  ------------  ------------
Name                               Self CPU     CPU total    CUDA total    # of Calls
-------------------------------  ------------  ------------  ------------  ------------
aten::mm                              1.2 ms       4.1 ms       6.80 s          512
aten::layer_norm                      0.4 ms       1.1 ms       1.90 s          128
aten::_softmax                        0.3 ms       0.9 ms       1.40 s           64
aten::add                             0.2 ms       0.6 ms       0.95 s          448
aten::gelu                            0.2 ms       0.5 ms       0.70 s          128
ProfilerStep*                         8.9 ms      12.4 s        ----            ---
-------------------------------  ------------  ------------  ------------  ------------
```

Three things jump off that table. First, `aten::mm` (matrix multiply) dominates CUDA time at 6.8 seconds — that is your useful matmul work, and it being the top line is healthy. Second, look at `ProfilerStep`: its **CPU total** is 12.4 seconds while the sum of CUDA time across all ops is roughly 11.75 seconds. The gap — about 0.65 seconds of CPU time with no overlapping GPU work — is a bubble: the CPU was doing serial work (or blocked on the loader) while the GPU idled. Third, and this is the subtle, expensive finding: `aten::mm` is taking 6.8 seconds for 512 calls. If you know the matmul sizes (you do — `record_shapes=True` captured them), you can compute the FLOPs and discover the matmuls are running at ~110 TFLOP/s, far below the A100's 312. That under-speed is the fingerprint of an **fp32 fallback** — the matmuls are running in 32-bit because autocast was not enabled or some op forced a dtype promotion, so they cannot use the Tensor Cores' bf16 path and run at a fraction of peak. The table did not say "fp32 fallback" in words; you *read* it from the time-per-call against the known FLOPs.

That is the whole art of the profiler table: it ranks the time, and you bring the model knowledge that turns a ranking into a diagnosis. A few rules for reading it fast. The op at the top is where to spend your attention — optimizing anything below the top three is rarely worth it (Amdahl's law: a 2× speedup of a line that is 5% of the step buys you 2.5%). A large CPU total with a small CUDA total on the same op means the op is launch-bound or running on the CPU — a candidate for `torch.compile` to fuse away the launch overhead. And a matmul whose time-per-call implies far-below-peak FLOP/s is your precision smell, which we chase next.

### Reading the trace, not just the table

The table ranks ops by total time, but it flattens the *timeline* — it cannot show you a bubble, because a bubble is about *when* things ran, not how long in aggregate. For that you open the trace the profiler emitted. `tensorboard_trace_handler` writes a `.json` (or `.pt.trace.json`) file that you load either in TensorBoard's profiler plugin or, more directly, by dragging it into [Perfetto](https://ui.perfetto.dev) (or `chrome://tracing` in older Chrome). What you see is the same kind of timeline `nsys` gives you, but with PyTorch's own op names on the bars: a row for the CPU thread showing `aten::mm`, `aten::layer_norm` as they were *dispatched*, and a row for the CUDA stream showing the corresponding kernels as they *executed*, with the two offset in time by exactly the launch-ahead distance.

Reading the trace is a different skill from reading the table, and it catches things the table cannot. You look for three shapes. First, a **gap in the CUDA row** — the GPU stream goes empty while the CPU row is still busy: that is a CPU-bound bubble, the CPU could not keep the launch queue full, often because of Python overhead or a synchronizing call. Second, the **launch-ahead distance**: a healthy run has the CPU row running well *ahead* of the CUDA row (the CPU has already queued the next several ops by the time the GPU starts the current one); when that distance collapses to near zero, the CPU is barely keeping up and you are one hiccup away from a bubble. Third, **a kernel that is wider than you expect** — you click it, Perfetto shows its name and duration, and you cross-reference it to the table to confirm it is the op you think it is. The table tells you *what* costs time; the trace tells you *whether it overlaps*, and overlap is half of performance. When in doubt, open the trace.

There is a cost to all this measurement, and you should know it so you do not mis-attribute it. **`torch.profiler` adds overhead** — recording every op's start and stop, capturing shapes, and walking the Python stack (`with_stack=True`) is not free, and a heavily-profiled step can run 10–50% slower than the unprofiled one. That overhead is *fine for finding the bottleneck* (the ranking is still correct; the relative shares still hold) but *wrong for reporting an MFU number* (the absolute step time is inflated). So the discipline is two-mode: profile *with* the profiler to find the wall, then turn the profiler *off* and re-time with bare CUDA events to report the honest MFU. Never quote an MFU number measured with the profiler attached — it is a diagnostic tool, not a benchmark.

## Reading a kernel roofline in ncu

When `torch.profiler` has named the one kernel that dominates, you drop to the bottom of the ladder and put it under `ncu`. The invocation aims at a single kernel and gathers the full set of counters:

```bash
ncu --set full --launch-count 1 \
  --kernel-name regex:"sgemm|gemm" \
  -o kernel_report \
  python train.py --steps 1
```

`--set full` collects every metric section, including the roofline and the memory-workload analysis; `--launch-count 1` profiles just one launch of the matched kernel (remember, `ncu` replays each kernel many times to read all its counters, so one launch is plenty and anything more is slow); `--kernel-name regex:"gemm"` filters to the matmul kernels by name so you do not profile the entire program. Open `kernel_report.ncu-rep` in the Nsight Compute GUI and the two sections that decide everything are the **roofline** and the **memory workload analysis**.

The roofline section is the chart from the [roofline post](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound), but now with *your measured kernel* plotted on it as a dot. `ncu` measures the kernel's actual FLOPs and actual bytes moved to and from HBM, computes its arithmetic intensity (FLOPs per byte), and places the dot. If the dot sits under the slanted bandwidth ceiling, the kernel is **memory-bound** — it is moving bytes as fast as the memory system allows and the math units are partly idle. If the dot sits under the flat compute ceiling, it is **compute-bound** — the math units are saturated. Figure 5 is how that placement decision flows.

![diagram showing how a kernel from ncu is placed on the roofline by its arithmetic intensity to determine whether it is memory bound or compute bound](/imgs/blogs/profiling-gpu-workloads-finding-the-real-bottleneck-5.png)

The figure routes a kernel through the placement: `ncu` hands you FLOPs and bytes, you form the intensity $I$ in FLOP per byte, you compare it to the A100 ridge of 156 FLOP per byte, and the comparison names the wall. A LayerNorm with $I \approx 3$ lands far left — memory-bound, and no faster math will help it. A big GEMM with $I \approx 250$ lands right of the ridge — compute-bound, and the only levers are better precision (bf16 over fp32) or a better matmul algorithm. This is the moment the roofline theory and the live profiler fuse into one decision.

#### Worked example: a memory-bound LayerNorm seen in ncu

Suppose `ncu` profiles your LayerNorm kernel and reports: 0.5 GFLOPs of arithmetic, 6.0 GB moved to and from HBM, and a duration of 3.4 milliseconds on an A100. Let us check the diagnosis against the hardware.

Arithmetic intensity: $I = 0.5\times10^9 / 6.0\times10^9 = 0.083$ FLOP per byte. That is *far* below the A100 ridge of 156 — wildly memory-bound, as theory predicts a normalization op should be.

Achieved bandwidth: $6.0 \text{ GB} / 3.4\text{ ms} = 1.76 \text{ TB/s}$. The A100's peak HBM bandwidth is 2.0 TB/s, so this kernel is running at $1.76/2.0 \approx 88\%$ of peak bandwidth.

That second number is the verdict. The kernel is at 88% of the memory ceiling — it is doing about as well as a memory-bound kernel *can* do on this chip. There is no point hand-tuning its inner loop; it is already nearly saturating the only resource it is limited by. The *only* way to make this LayerNorm faster is to make it move fewer bytes — which means **fusion**: folding the LayerNorm into the adjacent operations so the intermediate tensors never round-trip to HBM at all. That is precisely the insight FlashAttention and `torch.compile` exploit, and it is covered in depth in the kernel-fusion post of this series. `ncu` did not tell you "go fuse this"; it told you the kernel is memory-bound and near the bandwidth ceiling, and *you* concluded that fewer bytes — fusion — is the only remaining lever. Reading the achieved-versus-peak bandwidth is how you know whether a memory-bound kernel is worth touching at all.

The other `ncu` section you will read constantly is **occupancy**, important enough for its own treatment.

### Occupancy, defined carefully

**Occupancy** is the ratio of active warps to the maximum warps a streaming multiprocessor (SM) can hold. Unpacking that: an SM is one of the GPU's parallel processor cores (an A100 has 108), a **warp** is a group of 32 threads that execute together in lockstep, and each SM can keep a fixed maximum number of warps "resident" at once — 64 on an A100's architecture, so 2048 threads. Occupancy is how many of those slots your kernel actually fills. If your kernel launches enough work to keep 48 of the 64 warp slots busy, occupancy is 75%.

Why it matters: the GPU hides memory latency by *switching between warps*. When one warp stalls waiting on a memory load, the SM instantly runs another resident warp, so the math units stay fed. High occupancy means many warps to switch among, so latency stays hidden. Low occupancy — too few warps resident — means that when a warp stalls, there may be nothing else to run, and the SM idles. Occupancy is limited by resources: a kernel that uses too many registers per thread, or too much shared memory per block, can only fit a few warps per SM, starving the latency-hiding mechanism.

The critical nuance, which trips up everyone the first time: **high occupancy is necessary for memory-bound kernels but does not help compute-bound ones, and 100% occupancy is rarely the goal.** A compute-bound matmul that already saturates the Tensor Cores does not need more warps to hide latency — there is no latency to hide, the math units are the bottleneck. Pushing its occupancy higher by shrinking its tile can *hurt*, because a smaller tile reuses data less and lowers arithmetic intensity. So when `ncu` reports low occupancy, the right question is not "how do I raise it" but "is this kernel memory-bound?" If yes, low occupancy may genuinely be starving it and you raise it (smaller blocks, fewer registers). If it is compute-bound and already near peak FLOP/s, the low occupancy is fine and chasing it is wasted effort. `ncu`'s roofline section and occupancy section are read *together*, never alone.

`ncu` even tells you *what* is capping occupancy, in its "occupancy" section, which lists the limiter: registers per thread, shared memory per block, or block size. A concrete pattern: a kernel uses 128 registers per thread, and because each SM has a fixed register file (65,536 32-bit registers on an A100), $65{,}536 / 128 = 512$ threads can be resident, which is only 16 of the 64 warp slots — 25% occupancy, register-limited. If that kernel is memory-bound, that low occupancy is genuinely starving its latency-hiding, and the fix is to lower register pressure (simpler arithmetic, smaller unrolling, `__launch_bounds__` to cap the compiler). `ncu` hands you the limiter by name so you do not guess which resource to free. But the discipline holds: you only chase it after the roofline section confirms the kernel is memory-bound and would *benefit* from more resident warps. For a compute-bound kernel, 25% occupancy with saturated Tensor Cores is a finished kernel, and the register "limit" is a red herring.

| Kernel type | Roofline placement | Occupancy goal | The real lever |
| --- | --- | --- | --- |
| Big GEMM (matmul) | Right of ridge, compute-bound | Whatever the tuned tile gives; ~50% is fine | bf16/Tensor Cores; better matmul algo |
| LayerNorm / softmax | Far left, memory-bound | High — need warps to hide HBM latency | Fusion: move fewer bytes |
| Elementwise add / GELU | Far left, memory-bound | High | Fusion into the adjacent op |
| Attention (naive) | Memory-bound on the big score matrix | High | FlashAttention: never materialize the matrix |

## The four findings: what the profiler almost always reveals

If you profile enough real training runs, you start to see the same four villains over and over. Knowing them is not a substitute for profiling — you still measure to confirm which one you have — but it lets you read the timeline faster, because you know the shapes. Figure 7 breaks a single slow step into where the time actually goes, and the four non-matmul slices map exactly to the four findings.

![diagram of an MFU breakdown matrix showing the step split into matmul glue comms and idle with the share counts as MFU and the fix lever for each](/imgs/blogs/profiling-gpu-workloads-finding-the-real-bottleneck-7.png)

Read the matrix as the anatomy of a bad step. Only the matmul slice — 38% here — is *useful* work that counts toward MFU. The other 62% is the four findings: glue (memory-bound norms, activations, copies, 22%), comms (an exposed all-reduce, 18%), and idle (the data-loader stall, 22%). Each slice has a different fix lever, and each lever is a different post in this series. The point of the figure is the brutal arithmetic: if matmul is 38% of the step, your MFU is capped near 38% no matter how fast the matmul runs, because the other 62% is not matmul. You raise MFU by *shrinking the non-matmul slices*, and the profiler is how you find which slice to shrink first.

**Finding one: the starving DataLoader.** The most common, and the one we have been carrying. The GPU finishes a step and waits for the CPU to produce the next batch. On the `nsys` timeline it is a GPU idle gap aligned with a `data_to_gpu` NVTX band. The fix is to overlap data preparation with GPU compute, so the next batch is ready the instant the GPU asks for it. Figure 4 is the before-and-after.

![diagram comparing a starving data loader with num_workers zero against an overlapped prefetched loader with workers and pinned memory showing MFU rising from 18 to 31 percent](/imgs/blogs/profiling-gpu-workloads-finding-the-real-bottleneck-4.png)

The fix is three `DataLoader` arguments. `num_workers` spawns background processes that prepare batches in parallel while the GPU trains on the current one. `pin_memory=True` allocates the batch in page-locked host memory, which makes the host-to-device copy faster and, crucially, lets it run *asynchronously* via `non_blocking=True` so the copy overlaps compute instead of blocking it. `prefetch_factor` controls how many batches each worker stages ahead. Together they turn the serial "load then compute" into an overlapped pipeline:

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=8,
    num_workers=8,          # background processes prep batches in parallel
    pin_memory=True,        # page-locked host buffers -> faster async H2D
    prefetch_factor=4,      # each worker stages 4 batches ahead
    persistent_workers=True,# don't respawn workers every epoch
    drop_last=True,
)

for batch in loader:
    # non_blocking=True lets the copy overlap with the previous step's compute
    batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
    loss = model(**batch).loss
    loss.backward()
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
```

In the running example, that change alone took the idle gap to near zero and lifted MFU from 18% to 31%. Re-profile to confirm: the `data_to_gpu` band should now sit *inside* the previous step's compute, not after it.

**Finding two: the exposed all-reduce.** On multiple GPUs, after each backward pass the gradients must be averaged across all GPUs — a collective operation called an **all-reduce**. If that all-reduce runs *after* the backward pass finishes, the GPUs sit idle during the communication: a comms bubble. On the `nsys` timeline it is a gap where the NCCL kernels run and the compute kernels do not. Figure 6 is the before-and-after.

![diagram comparing an exposed all-reduce where compute waits four milliseconds against an overlapped reduction hidden inside the backward pass lifting eight GPU MFU from 22 to 45 percent](/imgs/blogs/profiling-gpu-workloads-finding-the-real-bottleneck-6.png)

To know whether overlap is even *possible*, you compute how much time the all-reduce costs, and that is a derivation worth doing because it tells you when comms is hideable and when it is the wall. The standard collective NCCL uses is the **ring all-reduce**, and its cost is not the naive "send everything to everyone." Lay the $N$ GPUs in a ring and split each GPU's gradient buffer of $S$ bytes into $N$ chunks. The reduce-scatter phase passes chunks around the ring $N-1$ times, each GPU sending one chunk per step; the all-gather phase does another $N-1$ passes. Each GPU therefore sends $2(N-1)/N \cdot S$ bytes total over the whole collective — and crucially, that volume is *nearly independent of $N$* for large rings (it approaches $2S$ as $N$ grows), which is why ring all-reduce scales. The time is that byte volume over the interconnect bandwidth $B$:

$$T_\text{all-reduce} = \frac{2(N-1)}{N} \cdot \frac{S}{B}$$

Put numbers on it: a 7B model in bf16 has $S = 14$ GB of gradients. On 8 GPUs connected by NVLink at, say, 300 GB/s effective, $T \approx (2 \cdot 7/8) \cdot 14\text{ GB} / 300\text{ GB/s} \approx 82$ ms per all-reduce. If your backward pass takes 200 ms, that 82 ms fits *inside* it and can be fully hidden by overlap — the comms bubble should be zero. But move that same job onto 8 GPUs connected by PCIe at ~25 GB/s, and the all-reduce balloons to nearly a second, far longer than the backward pass; now comms is *exposed by physics*, not by a bug, and no overlap can hide it. The interconnect matters as much as the GPU, which is the subject of this series' interconnects post. The derivation is how you tell "I have an overlap bug" (comms is smaller than backward but still showing up exposed) from "I am interconnect-bound" (comms is genuinely larger than the compute it could hide behind).

The fix, when it *is* an overlap bug, is to overlap the communication with the backward pass: as soon as a layer's gradients are computed, start reducing them while the backward pass continues computing the *next* layer's gradients. PyTorch's `DistributedDataParallel` does this automatically by bucketing gradients and kicking off the all-reduce per bucket during backward — you mostly need to *not break* the overlap (don't call `.grad` mid-backward, don't use gradient accumulation without `no_sync()` on the intermediate steps). When it is working, the NCCL kernels on the timeline sit *underneath* the backward compute, not after it, and the comms bubble vanishes. In the running example, fixing this took the 8-GPU MFU from 22% to 45%.

```python
import contextlib
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

model = DDP(model, device_ids=[local_rank],
            gradient_as_bucket_view=True)  # less copy, cleaner overlap

# With gradient accumulation, suppress the all-reduce on the inner steps
# so only the final micro-step triggers the (overlapped) reduction.
for i, micro_batch in enumerate(micro_batches):
    is_last = (i == len(micro_batches) - 1)
    ctx = model.no_sync() if not is_last else contextlib.nullcontext()
    with ctx:
        loss = model(**micro_batch).loss / len(micro_batches)
        loss.backward()  # on the last micro-step, DDP overlaps all-reduce here
optimizer.step(); optimizer.zero_grad(set_to_none=True)
```

**Finding three: the fp32 fallback.** Your matmuls are running in 32-bit when they should be in bf16, so they cannot use the Tensor Cores' fast 16-bit path and run at a fraction of peak. The smell, from the `torch.profiler` table, is a matmul whose time-per-call implies far-below-peak FLOP/s (the 110-of-312 we read earlier). The cause is usually that autocast was never enabled, or one op in the forward pass promotes the dtype back to fp32 and poisons the chain. The fix is to wrap the forward pass in `torch.autocast`, which runs the matmul-heavy ops in bf16 while keeping numerically sensitive reductions in fp32:

```python
import torch

# bf16 autocast: matmuls run on Tensor Cores at 312 TFLOP/s instead of the
# fp32 path. bf16 needs no GradScaler (its exponent range matches fp32);
# fp16 would require torch.cuda.amp.GradScaler for loss scaling.
for batch in loader:
    batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = model(**batch).loss
    loss.backward()
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
```

Re-profile and the `aten::mm` line should drop dramatically — in the running example, enabling bf16 cut matmul time roughly in half and was the largest single MFU jump. (Whether you reach for bf16 or fp16, and why bf16 trains stably where fp16 needs loss scaling, is the subject of this series' mixed-precision post; the short version is bf16's 8-bit exponent matches fp32's dynamic range, so gradients do not underflow.)

**Finding four: the memory-bound kernel.** A non-matmul kernel — a LayerNorm, a softmax, a chain of elementwise ops — dominates more of the step than it should, because each one round-trips its tensors to HBM. `ncu` confirms it: low arithmetic intensity, high achieved bandwidth, the dot sitting on the slanted ceiling (the LayerNorm worked example above). The fix is fusion — `torch.compile` to fuse elementwise chains automatically, FlashAttention for the attention block, or a hand-written Triton kernel for a specific hot path — anything that keeps intermediates in registers or shared memory instead of HBM. `torch.compile` is the cheapest first move:

```python
import torch

# torch.compile fuses elementwise + normalization chains, cutting HBM
# round-trips on the memory-bound glue. Often a free 10-30% on a step
# whose profiler table shows fat layer_norm / softmax / add lines.
model = torch.compile(model, mode="max-autotune")
```

## Case studies / real numbers

Profiling is most convincing with real numbers from real systems, so here are three grounded reference points. Mark them as reported figures; where I am approximating, I say so.

**The starving-loader run (the carried example).** The 18%-to-45% arc in this post is representative of a class of bug I have watched play out many times and that is well documented in PyTorch's own profiler tutorials and the TensorBoard profiler guide. A run trains a 7B model on a single A100, the profiler shows a multi-millisecond GPU idle gap per step aligned with the data loader, and the cumulative fix sequence is: add `DataLoader` workers and pinned memory with `non_blocking` copies (18% → roughly 31% MFU as the idle gap closes), enable bf16 autocast so the matmuls leave the fp32 path (roughly 31% → 42% as matmul time halves), and on the 8-GPU version ensure the all-reduce overlaps backward (lifting the multi-GPU MFU into the mid-40s). The exact percentages depend on model and batch, but the *shape* — three independent fixes found by three readings of the profiler, each worth roughly 10–15 MFU points — is the typical story. None of the three would have been found by staring at the model code; all three were sitting in plain sight on the timeline.

**Megatron-LM and PaLM reported MFU.** The Megatron-LM tensor-parallel training work reported model FLOPs utilization in the 40–52% range for large GPT-style models on thousands of A100s, which the field treats as the benchmark for "well-optimized large-scale dense training" — see the Megatron-LM scaling paper. Google's PaLM report quotes 46.2% MFU for the 540B-parameter model and explicitly uses the $6N$-per-token convention this post derived, which is why that number is directly comparable to the MFU you compute on your own run. The takeaway for calibration: if your single-GPU MFU is in the teens, you have a finding to chase; if it is in the 40s, you are in the same neighborhood as the best published large-scale runs and further gains get hard. These are reported figures from the respective papers.

**A memory-bound kernel found in ncu.** The general result that normalization, activation, and attention-softmax kernels are memory-bound — pinned to the slanted roofline ceiling rather than the compute ceiling — is exactly the observation that motivated FlashAttention. The FlashAttention paper measured that standard attention spends most of its time moving the large intermediate score matrix to and from HBM, and by fusing the whole attention into one kernel that never materializes that matrix, it reported roughly 2–4× wall-clock speedups and large memory savings, with no change to the math. That is the LayerNorm worked example from this post writ large: `ncu` shows a kernel near the bandwidth ceiling, the diagnosis is memory-bound, and the only lever is moving fewer bytes via fusion. The 2–4× figure is from the FlashAttention paper; the exact speedup depends on sequence length and head dimension.

**`torch.compile` on the glue.** A useful calibration point for the fusion lever: PyTorch's own benchmarks for `torch.compile` report that fusing the elementwise and normalization "glue" around the matmuls — the exact memory-bound kernels `ncu` flags — commonly yields a 10–40% end-to-end speedup on training workloads, with the gain largest on models whose profiler table shows fat `layer_norm`, `gelu`, and `add` lines and smallest on models already dominated by big matmuls. This is the cheapest of the four fixes (one line, `model = torch.compile(model)`) and the right first move whenever finding four is the diagnosis. The range is approximate and model-dependent; a matmul-bound model sees little, a glue-heavy one sees a lot — which is, again, why you profile before you decide.

### Profiling memory, not just time

Time is not the only thing the profiler measures, and on large models the second resource — **memory** — is often the wall you hit first. A run that *would* be fast cannot run at all if it OOMs (runs out of GPU memory) on the activations. `torch.profiler` with `profile_memory=True` records each op's memory allocation, and the table gains a `Self CUDA Mem` column showing which ops allocate the most. But the faster diagnostic is `torch.cuda.max_memory_allocated()`, which reports the peak bytes the program held at any point — the number that decides whether you fit. The standard memory budget for training has four parts you can compute on paper, exactly as the [memory hierarchy post](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm) lays out: the parameters ($2N$ bytes in bf16), the gradients (another $2N$), the optimizer state (Adam keeps two moments in fp32, $8N$, plus an fp32 master copy of the weights, $4N$), and the activations saved for backward (the term that scales with batch and sequence and is the one you actually control). For a 7B model that is already $2 \cdot 7 + 2 \cdot 7 + 12 \cdot 7 = 112$ GB of static state before a single activation — which is why a 7B model does not fit on one 80GB A100 without sharding the optimizer state, and why ZeRO/FSDP exist. When the wall is memory rather than time, you profile with `max_memory_allocated()` and `memory_summary()`, find which of the four terms dominates, and reach for activation checkpointing (recompute activations in backward instead of saving them) or optimizer-state sharding — a different toolbox from the time-profiling one, but the same measure-first discipline. The peak-memory number is also load-bearing for the cost question: a model that needs 2 GPUs to fit costs twice as much per training-run as one that needs 1, so shrinking peak memory is sometimes a \$ decision as much as a fits-or-not one.

## The bottleneck decision tree

You now have the tools and the four findings. The remaining skill is *routing* — given a symptom, knowing which tool to reach for and which finding is likely. Figure 8 is the decision tree that ties the whole post together.

![diagram of a bottleneck decision tree routing a low MFU symptom through whether the GPU is idle or busy to the most likely cause and first fix](/imgs/blogs/profiling-gpu-workloads-finding-the-real-bottleneck-8.png)

Read it top-down. The symptom is always the same — MFU is low. The first fork is the question `nsys` answers: *is the GPU idle, or is it busy but unproductive?* If the timeline shows idle gaps, you have a feeding problem, and it is one of two things: a data-loader stall (idle gap aligned with the loader) or an exposed all-reduce (idle gap aligned with NCCL kernels). If the GPU is busy but MFU is still low, you have a kernel-efficiency problem, and `torch.profiler` plus `ncu` tell you which: a memory-bound glue kernel eating the step (fuse it) or an fp32 fallback making the matmuls crawl (enable bf16). Four leaves, four fixes, each reached by a measured signal rather than a guess. This tree is the post in one picture, and it is the routine I run on every slow job: timeline first to split idle from busy, then drill into whichever branch the measurement points at.

#### Worked example: routing a real slow run end to end

Concrete walk-through. A colleague's 1.3B model trains at what feels slow; they want to "rewrite the attention." We resist and profile instead. `nsys` first: the timeline shows the GPU idle for about 30% of each step, gaps aligned with the `data_to_gpu` NVTX band — that is the idle branch, data-loader sub-branch. We add 8 workers, `pin_memory`, and `non_blocking` copies; re-measure, MFU rises from 19% to 27% and the gaps shrink but don't vanish. Re-profile: a smaller residual gap now aligns with NCCL kernels — the exposed-all-reduce sub-branch. We confirm DDP bucketing is intact and remove a stray `loss.item()` that was forcing a sync mid-step; MFU to 33%. Now the GPU is busy, so we cross to the busy branch: `torch.profiler` shows `aten::mm` running at ~120 TFLOP/s on an A100, the fp32-fallback smell. We add bf16 autocast; matmul time halves, MFU to 44%. Finally `ncu` on the top remaining non-matmul kernel shows a fat fused-norm chain at 85% of peak bandwidth — memory-bound and near-saturated, so we `torch.compile` the model to fuse it, gaining a last few points to 47%. Total: 19% to 47% MFU, four fixes, every one located by the tree and confirmed by re-measurement. The attention rewrite they wanted to do first would have touched the part that was already fine.

## When to reach for the profiler (and when not to)

Profiling is cheap relative to the time it saves, but it is not free, and a few honest boundaries keep you from over-using it.

Reach for the profiler the moment a run is slower than your MFU expectation and you cannot name *why* in measured units. If you are below ~30% MFU on a single GPU, there is almost certainly a finding waiting, and the timeline will show it in ten minutes — far faster than the days you would burn guessing. Reach for `ncu` only after `nsys` and `torch.profiler` have narrowed the field to one kernel; running `ncu --set full` on a whole program is so slow it is its own bottleneck.

Do *not* profile before you have a baseline MFU number — you need to know whether you have a problem before you go hunting for one. A run already at 45% MFU on a single GPU is in the same band as the best published large-scale runs; the marginal hour spent chasing it to 48% is usually better spent elsewhere unless you are at a scale where 3 MFU points is real money. Do not micro-optimize a kernel that `ncu` shows is already near its ceiling (88% of peak bandwidth for a memory-bound kernel, or near peak FLOP/s for a compute-bound one) — it is done; the only lever left is changing the algorithm so it moves fewer bytes or does less work, which is a fusion or architecture decision, not a tuning one. And do not trust a single profiled step: warm up, take the median of many, and synchronize your timers, or every conclusion rests on noise.

One more boundary, on *how much* to invest in the loop. Profiling has diminishing returns, and a mature engineer knows where the knee is. The first pass — running `nsys` once and finding the data-loader stall — costs ten minutes and might double your throughput; that is the best ROI in all of software performance and you should never skip it. The second and third passes — bf16, then overlapping comms — cost an hour each and each buys a real double-digit MFU jump. But the fourth, fifth, and sixth passes, where you are chasing the last few points by hand-fusing a specific kernel or tuning a tile size, can cost days for a couple of MFU points. Whether that is worth it depends entirely on scale: at one GPU for a week, a 3-point gain is rounding error; at a thousand GPUs for a month, a 3-point gain is a six-figure savings and absolutely worth a senior engineer's week. The profiler tells you *where* the time is; your judgment about scale and deadline tells you *when to stop*. Stopping at "good enough MFU for this job's scale" is a skill too, and over-optimizing a small run is its own kind of waste.

The deeper rule is the one we opened with. The profiler exists to defeat intuition, because intuition about where time goes is reliably wrong. Every fix in this post — the data loader, the all-reduce, the fp32 fallback, the memory-bound kernel — was invisible to reasoning about the model and obvious on the timeline. The skill is not knowing the four findings; it is having the discipline to measure first, every time, and to score every change by whether MFU actually moved. (For the embedded and on-device flavor of this same discipline — profiling a model on a phone or an edge accelerator, where the tools differ but the loop is identical — see the [edge-AI profiling and benchmarking post](/blog/machine-learning/edge-ai/profiling-and-benchmarking-on-device); for picking and benchmarking a serving GPU by measured throughput and latency rather than datasheet peaks, see the [LLM GPU benchmark post](/blog/machine-learning/mlops/llm-gpu-benchmark).)

## Key takeaways

- **Measure, don't guess.** Human intuition about where a program spends time is wrong more often than right. The bottleneck is almost never where it feels like it should be. Profile first, every time.
- **`nvidia-smi` at 100% means busy, not productive.** It cannot tell math from stall or full-chip from one-tiny-kernel. Use MFU — model FLOPs achieved over peak FLOP/s — as the honest north-star metric.
- **MFU = $6NT / (t \cdot P_\text{peak})$.** Parameters, tokens, step time, datasheet peak. Everything is measurable; the $6N$-per-token rule is the workhorse the whole field uses. Warm up and synchronize your timers or the number is fiction.
- **Use the three-tool ladder top-down.** `nsys` for the system timeline and idle gaps, `torch.profiler` for which op and CPU-vs-CUDA time, `ncu` for one kernel's roofline and occupancy. Never start at `ncu`.
- **A fraction $f$ of idle time caps MFU at $1 - f$**, no matter how fast the busy part runs. You cannot tune past a gap; you can only fill it by overlapping the work that caused it.
- **The four findings cover most slow runs**: a starving DataLoader (idle gap on the loader), an exposed all-reduce (idle gap on NCCL), an fp32 fallback (matmuls below peak FLOP/s), and a memory-bound kernel (near the bandwidth ceiling in `ncu`). Each has a different lever.
- **Read `ncu`'s achieved-vs-peak bandwidth before tuning a memory-bound kernel.** At 88% of peak it is done; the only lever left is fusion — moving fewer bytes — not inner-loop tuning.
- **Occupancy helps memory-bound kernels, not compute-bound ones.** 100% occupancy is rarely the goal; read the roofline and occupancy sections together, never alone.
- **Score every fix by re-measuring MFU.** If MFU went up, the fix was real. If it didn't, you optimized the wrong thing — and the profiler already told you what the right thing was.

## Further reading

- [Nsight Systems user guide](https://docs.nvidia.com/nsight-systems/) — the system-timeline profiler: capture flags, NVTX, reading idle gaps and sync stalls.
- [Nsight Compute documentation](https://docs.nvidia.com/nsight-compute/) — the per-kernel profiler: the roofline, occupancy, and memory-workload analysis sections.
- [PyTorch Profiler documentation](https://pytorch.org/docs/stable/profiler.html) and the [PyTorch Profiler with TensorBoard tutorial](https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html) — the `schedule`, `record_function`, trace handler, and the data-loader case study this post mirrors.
- [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311) — the 46.2% MFU report and the $6N$-per-token convention.
- [Megatron-LM scaling](https://arxiv.org/abs/2104.04473) — tensor-parallel MFU benchmarks for large-scale dense training.
- [FlashAttention](https://arxiv.org/abs/2205.14135) — the canonical memory-bound-kernel-to-fused-kernel result, and why attention is bandwidth-limited.
- Within this series: the [intro on why HPC is the bottleneck](/blog/machine-learning/high-performance-computing/why-hpc-is-the-bottleneck-for-modern-ai), the [roofline model](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound), the [memory hierarchy](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm), and [inside the GPU: SMs, warps, and SIMT](/blog/machine-learning/high-performance-computing/inside-the-gpu-sms-warps-and-the-simt-execution-model). The kernel-fusion post (the fix for memory-bound kernels) and the capstone HPC playbook complete the loop.
