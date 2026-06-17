---
title: "Profiling and benchmarking on-device: measuring latency, memory, and energy honestly"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Build a timing harness that does not lie — warm-up, device sync, median and p99, fixed clocks, thermal soak — then profile per-layer, measure peak memory and joules per inference, and ship numbers other people can actually reproduce."
tags:
  [
    "edge-ai",
    "model-optimization",
    "profiling",
    "benchmarking",
    "latency",
    "energy",
    "inference",
    "efficient-ml",
    "torch-profiler",
    "mlperf",
  ]
category: "machine-learning"
subcategory: "Edge AI"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/profiling-and-benchmarking-on-device-1.png"
---

Here is a true story, lightly anonymized, that has happened to almost everyone who has shipped a model onto a device.

A team had two candidate models for a vision feature on a Jetson Orin Nano: a compact MobileNet-style network and a slightly larger EfficientNet-style network. They wrote a quick script: load the model, run it once, print the elapsed time. The compact model clocked **6 ms**. The bigger one clocked **14 ms**. Obvious winner, right? They shipped the compact one. Two weeks later the field telemetry came back: the "fast" model was the *slower* one in production, by a wide margin, and it was missing the 30 ms frame budget under sustained load while the "slow" one was comfortably inside it. The benchmark had lied to them — not by a little, but by **inverting the ranking of the two models they were choosing between.**

How does a measurement get a binary A-vs-B decision exactly backwards? Three classic traps, all present in that one-line script. They timed the **first run**, which paid for just-in-time kernel compilation, a cold instruction cache, and lazy memory allocation — costs you never pay again. They timed **on the GPU without synchronizing**, so the host clock stopped the instant the kernel was *launched*, not when it *finished* — they were timing an asynchronous dispatch, not the compute. And they took a **single sample on a cold chip**, so they never saw what happens once the silicon heats up and the governor claws the clocks back. Each trap individually distorts the number. Together, they manufacture a fiction.

This post is the measurement spine of the whole series. Every other article here makes a claim of the form "this technique made the model X% smaller, Y% faster, Z points less accurate." Quantization, pruning, distillation, the efficient architectures, the compilers — [the entire taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) is a set of bets, and a bet is only as good as the scale you weigh it on. The companion post [the metrics that actually matter on-device](/blog/machine-learning/edge-ai/the-metrics-that-actually-matter-on-device) argued *which* numbers to care about — batch-1 tail latency, peak working set, joules per inference. This post is about *how to obtain those numbers without fooling yourself*. The figure below shows the same two models measured the wrong way and the right way; the ranking flips.

![Side-by-side comparison of a model measured the wrong way with a cold single run and no device sync against the same model measured with warm-up sync and a median, where the ranking flips](/imgs/blogs/profiling-and-benchmarking-on-device-1.png)

By the end you will be able to: write a timing harness that produces a number you would stake a release on; profile a model layer-by-layer to find the one operator eating most of your latency; measure peak memory and energy with real tools (`torch.profiler`, `tegrastats`, `powermetrics`, the TFLite and ONNX Runtime benchmark binaries); and publish a benchmark that another engineer can reproduce — which, as we will see, is shockingly rare in this field.

## Why on-device measurement is uniquely treacherous

Cloud benchmarking is forgiving. You run on a server with fixed clocks, ample cooling, a steady power supply, and you usually care about throughput at large batch sizes, which averages away a lot of noise. Edge measurement throws all of that out.

On a phone or a Jetson or a microcontroller, the clock frequency is *not constant*. A dynamic voltage and frequency scaling (DVFS) governor — the kernel subsystem that trades clock speed for power and heat — moves the CPU and GPU frequencies up and down on a millisecond timescale based on load, temperature, and battery state. A device that boots cold at its maximum clock will, under a sustained inference loop, heat past its thermal limit and get **throttled**: the governor drops the frequency to keep the junction temperature in bounds. Your model did not change; the hardware running it did. So "the latency" is not one number — it is a function of thermal state, power mode, what else is running, and how long you have been measuring.

On top of that, edge accelerators are **asynchronous**. When you call a GPU or NPU (neural processing unit — a dedicated matrix-math accelerator on the SoC) op from Python, you are not running it; you are *queuing* it. Control returns to the host immediately. If you stop a host-side timer right after the call, you measured the time to *enqueue work*, which can be a hundred times smaller than the time to *do* the work. This single mistake — forgetting to synchronize — is the most common reason a published edge latency number is physically impossible.

And edge inference is almost always **batch-1 and latency-bound**, not throughput-bound. A camera delivers one frame at a time; a voice assistant transcribes one utterance. You cannot hide kernel-launch overhead or memory-bound stalls behind a big batch the way a datacenter does. The batch-1 number is the number that matters, and batch-1 is exactly where overhead, variance, and the long tail dominate. If you have not read it yet, [the roofline model post](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives) explains *why* a low-FLOP model can still be slow — it is memory-bound — and profiling is how you find out which side of the roofline you are actually on.

So the rest of this post builds, piece by piece, a measurement protocol that survives all of this: warm-up, synchronization, many runs reported as percentiles, pinned clocks, an explicit thermal state, batch-1 reality, and clean separation of data-loading from compute. Then we profile per-layer, measure memory and energy, and write it all down so it is reproducible.

## The anatomy of an honest timing harness

Let me state the protocol up front, then justify each piece. A trustworthy latency measurement runs through exactly five stages, in order. The figure below is the harness as a timeline.

![Timeline of the honest timing harness moving through warm-up then device sync then a thousand timed runs then sample collection then reporting the median and p99](/imgs/blogs/profiling-and-benchmarking-on-device-2.png)

1. **Warm-up.** Run the model some number of times (often 20–100) and throw those numbers away. This pays off all the one-time costs so they do not contaminate your samples.
2. **Synchronize the device.** Before you start *and* stop the timer, force the host to wait until the device is actually idle, so you bracket real compute, not async dispatch.
3. **Many timed runs.** Run N times (often 200–2000) and record each individual latency. One number is not a measurement; a distribution is.
4. **Collect the samples.** Keep the per-iteration times so you can compute percentiles, not just a running average.
5. **Report robust statistics.** Report the **median (p50)** as the typical latency and a **tail percentile (p90/p99)** as the worst-case the user feels. Never report the bare mean.

Each stage exists to kill a specific lie. Let us take them one at a time, because the *why* is where the engineering judgment lives.

### Warm-up: why the first run is a different model

The first time you run a model, you pay costs you will never pay again:

- **Just-in-time (JIT) compilation.** PyTorch eager mode, TorchScript, ONNX Runtime, TensorRT, and TFLite all compile or specialize kernels on first use. cuDNN runs autotuning — it benchmarks several algorithms for your specific tensor shapes and caches the winner. The first convolution might try five algorithms before picking one.
- **Cold caches.** The instruction cache, the data cache, and the kernel's own lookup tables are all empty. The first pass is full of cache misses.
- **Lazy allocation.** Memory pools, workspace buffers, and the CUDA context are allocated on demand. The first run triggers `cudaMalloc` calls; later runs reuse the pool.
- **Clock spin-up.** A device sitting idle is at a low clock. The first inference may run partly before DVFS ramps the GPU to its inference clock.

The result: the first run is routinely **2–10x slower** than steady state, and it is *not representative of anything the user will experience* after the model is loaded and warm. Timing it is like timing a sprinter's first step out of the blocks and reporting it as their top speed. You warm up, discard, and only then measure.

How many warm-up runs? Enough that the autotuner has settled and the clocks are stable. For a small CNN on a GPU, 20–50 is plenty. For a large model with many distinct kernel shapes, or a runtime that autotunes aggressively, use more. The cheap, robust rule: warm up until consecutive runs stop trending downward, then add a margin.

### Device synchronization: you are timing the launch, not the kernel

This is the subtle one, and it is worth slowing down on because it is the single most common way to publish a number that is off by 50–100x.

GPUs and NPUs execute asynchronously with respect to the host CPU. When your Python code calls `y = model(x)` on a CUDA tensor, the framework enqueues the kernels into a stream and returns *immediately* — before a single multiply has happened. The host is now free to do other work while the device chews through the queue. This asynchrony is a feature: it lets you overlap host and device work. But it wrecks naive timing.

Consider the wrong way:

```python
import time
import torch

x = torch.randn(1, 3, 224, 224, device="cuda")

t0 = time.perf_counter()
y = model(x)              # returns immediately — kernels are merely QUEUED
t1 = time.perf_counter()
print((t1 - t0) * 1000)   # prints the LAUNCH time, e.g. 0.15 ms — a fiction
```

That `0.15 ms` is the time to *enqueue* the work, not to *compute* it. The actual forward pass might take 14 ms. You measured a number 90x too small. If you then `print(y)` or call `.cpu()` somewhere, the print *implicitly synchronizes* (it has to, to read the data), so the numbers in your script become an unpredictable mix of synced and unsynced — which is how people get "latency" numbers that change when they add a debug print.

The fix is to force the host to wait for the device to finish before you stop the timer:

```python
import time
import torch

x = torch.randn(1, 3, 224, 224, device="cuda")

torch.cuda.synchronize()              # ensure device is idle before we start
t0 = time.perf_counter()
y = model(x)                          # queue the kernels
torch.cuda.synchronize()              # WAIT until they actually finish
t1 = time.perf_counter()
print((t1 - t0) * 1000)               # now this is real compute time
```

The equivalents on other backends:

- **CUDA:** `torch.cuda.synchronize()`, or use CUDA events (`torch.cuda.Event(enable_timing=True)`) which measure on the device's own clock and avoid host-scheduling jitter entirely.
- **Apple Metal (MPS):** `torch.mps.synchronize()`.
- **TFLite / ONNX Runtime on CPU:** these are synchronous — the call returns when the work is done — so an explicit sync is unnecessary, but you still warm up. On a GPU/NNAPI/Core ML delegate, the delegate handles the sync internally for `Invoke()`, but you must ensure the output is materialized.
- **TensorRT:** `context.execute_async_v3(stream)` followed by `stream.synchronize()`.

CUDA events deserve a special mention because they are the gold standard for GPU timing. Instead of timing on the noisy host clock, you record two events *into the stream* and ask the device how much time elapsed between them on its own hardware timer:

```python
import torch

start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)

torch.cuda.synchronize()
start.record()
y = model(x)
end.record()
torch.cuda.synchronize()              # events are only valid after sync
elapsed_ms = start.elapsed_time(end)  # device-measured milliseconds
```

This removes host-side scheduling jitter and Python overhead from the measurement, which matters most for small, fast models where a few microseconds of Python is a meaningful fraction of the kernel time.

### Many runs and the right statistic: why the median is robust

You have warmed up and you synchronize correctly. Now: how many times do you run, and what do you report?

Run many times — at least a few hundred, ideally a thousand or more for a fast model — and keep **every** sample. The reason is that on-device latency is a *random variable*, not a constant. The DVFS governor nudges clocks; the OS scheduler preempts you for a background task; a thermal event drops a clock for a few milliseconds; memory pressure triggers a page fault. Your latency has a distribution with a body and a tail.

Now the central question: **mean or median?** This is not a stylistic preference. It is a statistics decision with a provable answer, and getting it wrong is how the "mean-over-throttle" benchmark lies.

The mean (arithmetic average) of $n$ samples $x_1, \dots, x_n$ is

$$
\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i .
$$

The mean is **not robust**: a single large outlier moves it by an amount proportional to the outlier's size. Concretely, the influence of one sample on the mean is $1/n$ of that sample's value, with no bound. So if 999 of your runs are 12.0 ms and one run hits a thermal spike at 90 ms, the mean is

$$
\bar{x} = \frac{999 \times 12.0 + 1 \times 90}{1000} = 12.078 \text{ ms}.
$$

That is only a 0.6% shift here — but make the spike rarer-but-larger (a GC pause, a 500 ms page-in on a memory-starved device) and the mean drifts visibly above the latency *every single user actually experiences on a typical run*. The mean answers "what is the average if I sum everything," which is the right question for **throughput** (total work over total time) but the wrong question for **what a user feels on a given inference.**

The median (p50) is the middle value of the sorted samples. Its key property is a **breakdown point of 50%**: you can corrupt up to half the data with arbitrarily large outliers before the median becomes meaningless. One spike — or even a hundred spikes out of a thousand — does not move it at all, because the middle of the sorted list is unchanged. For the example above, the median is exactly 12.0 ms: the 90 ms spike is just the last element after sorting and has zero effect on the middle. The figure below shows this contrast directly.

![Comparison of reporting the mean versus the median for the same thousand runs where one ninety millisecond throttle spike inflates the mean but leaves the robust median unchanged](/imgs/blogs/profiling-and-benchmarking-on-device-4.png)

So the rule is: **report the median (p50) as the typical latency.** It is the robust estimate of where the body of the distribution sits, and it is exactly the number a user experiences on a normal run.

But the median alone hides the tail — and on the edge, the tail *is* the product. A voice assistant that responds in 200 ms on the median but 2 seconds at p99 feels broken once in a hundred times, which is once a minute of heavy use. So you also report a **tail percentile.** The p90, p95, or p99 latency is the value below which that fraction of runs fall:

$$
p_q = \inf \{ t : F(t) \ge q \}, \qquad F(t) = \Pr[X \le t],
$$

where $F$ is the cumulative distribution of latency. In practice with $n$ sorted samples $x_{(1)} \le \dots \le x_{(n)}$, the p99 is approximately $x_{(\lceil 0.99\, n \rceil)}$ — the value at the 99th-percentile rank. For $n = 1000$, the p99 is roughly the 990th-smallest sample. (Use enough samples that the tail is actually populated: to estimate p99 you want hundreds of samples *above* the body, so a few thousand total runs.)

So the honest report for a single configuration is a triple: **p50 / p90 / p99**, plus the device, clocks, and thermal state we will get to. Three numbers, not one. The median says "typically this fast," the p99 says "and even on a bad run, no worse than this." That is a number you can write a service-level objective against.

### How many runs do you actually need?

"Run a lot" is good advice but imprecise, and the right answer is different for the median than for the tail — a distinction that trips up a lot of otherwise-careful benchmarks. The reason is that the *uncertainty* of a percentile estimate depends on how densely your samples populate that part of the distribution.

For the **median**, you have samples on both sides, so it stabilizes quickly. The standard error of the sample median for a distribution with density $f$ at the median $m$ is approximately

$$
\text{SE}(\hat{m}) \approx \frac{1}{2 f(m) \sqrt{n}},
$$

which shrinks like $1/\sqrt{n}$. In practice a few hundred samples pin the median to well within the run-to-run noise of the hardware itself. There is little benefit past about 500–1000 runs *for the median.*

For a **tail percentile** like p99, the situation is harsher: by definition only 1% of your samples land at or above it, so with $n$ samples you have only about $0.01\,n$ data points to estimate it from. At $n = 200$ that is *two* samples above p99 — your p99 estimate is essentially one or two lucky-or-unlucky draws, and it will swing wildly between repeated benchmarks. To estimate p99 with the same relative confidence as the median, you need roughly two orders of magnitude more runs, because the effective sample size for the tail is $0.01\,n$, not $n$. A defensible rule of thumb: for a stable p99 you want at least a few hundred samples *above* the body, which means a few thousand total runs; for p99.9 you want tens of thousands. This is not pedantry — it is why a p99 quoted from a 100-run benchmark is meaningless, and why people who report only "we ran it 100 times" are usually quoting a noisy mean dressed up as a tail.

The practical consequence: pick your run count from the *most extreme percentile you intend to report.* If you only need p50, 500 runs is plenty. If you are writing a p99 SLO, budget for a few thousand. If you genuinely need p99.9 (a safety-critical real-time system), you are doing a much longer soak, and at that point you should also be watching for the thermal drift that a multi-minute run introduces — which is the next concern.

#### Worked example: the same model, measured wrong vs right

Let me make the opening story concrete with numbers you can reproduce in spirit. We have a MobileNetV3-Small classifier exported and running on a Jetson Orin Nano (15 W power mode), batch 1, 224x224 input.

**Measured wrong** — cold, no sync, one run, host timer stopped right after the call:

```python
import time, torch
x = torch.randn(1, 3, 224, 224, device="cuda")
t0 = time.perf_counter()
y = model(x)                 # async launch, not awaited
t1 = time.perf_counter()
print((t1 - t0) * 1000)      # ~0.4 ms  <-- physically impossible
```

Reported latency: **0.4 ms**. That implies 2,500 inferences per second on a 15 W board for a real CNN — a number that should immediately fail your sniff test. It is the launch time.

**Measured right** — warm up 50, synchronize, 1000 runs, report percentiles:

```python
import torch
import numpy as np

x = torch.randn(1, 3, 224, 224, device="cuda")

# warm-up
for _ in range(50):
    _ = model(x)
torch.cuda.synchronize()

# timed runs with CUDA events
times = []
start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)
for _ in range(1000):
    start.record()
    y = model(x)
    end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end))   # ms, device-measured

t = np.array(times)
print(f"p50 {np.percentile(t,50):.2f} ms  "
      f"p90 {np.percentile(t,90):.2f} ms  "
      f"p99 {np.percentile(t,99):.2f} ms")
# p50 5.9 ms   p90 6.4 ms   p99 8.1 ms
```

Reported latency: **p50 5.9 ms, p99 8.1 ms.** Now the number is believable (~170 inferences/s) and *complete* — it tells you both the typical case and the bad case. The wrong measurement said 0.4 ms; the right one says ~6 ms typical. That is a **15x** distortion from a single missing `synchronize()` plus a missing warm-up, and it is exactly the kind of error that, applied unevenly across two candidate models, flips a ranking.

### Fixed clocks: stop measuring the governor

Even a perfect harness produces noisy, non-reproducible numbers if the clock frequency is wandering. Between two runs of your benchmark, the DVFS governor might have settled at a different operating point because the room was warmer or a background sync ran. You are then measuring the *governor's mood*, not the model.

The fix is to pin the clocks for the duration of the benchmark so every run sees the same hardware:

- **NVIDIA Jetson:** set the power mode with `nvpmodel`, then lock the CPU/GPU/EMC clocks to their max for that mode with `jetson_clocks`. This is the single most important step for reproducible Jetson numbers.

```bash
# choose a named power budget (e.g. mode 0 = MAXN, mode 1 = 15W on Orin Nano)
sudo nvpmodel -m 1
# lock CPU, GPU, and memory clocks to the max for that mode
sudo jetson_clocks
# verify
sudo jetson_clocks --show
```

- **Linux CPU (Raspberry Pi, x86):** set the CPUfreq governor to `performance` so the cores stay at max instead of scaling down.

```bash
# pin every CPU core to the performance governor
for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  echo performance | sudo tee "$c" > /dev/null
done
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq   # confirm it is at max
```

- **Android (rooted, for engineering benchmarks):** lock the little/big cluster frequencies and disable hotplug. On unrooted devices you cannot pin clocks, which is exactly why MLPerf Mobile reports both a peak and a sustained run.
- **Desktop GPU:** `nvidia-smi -lgc <freq>` locks the graphics clock; `--lock-memory-clocks` pins memory.

Pinning clocks does two things. It removes the largest source of run-to-run variance, and it makes your number *mean something specific*: "at this power mode, at these clocks." Without it, two engineers benchmarking "the same model on the same Jetson" can legitimately get numbers 30% apart and waste a day arguing.

### Thermal state: cold burst vs sustained reality

Here is the trap that bit the team in the opening story, and the one most likely to bite you. Even with pinned clocks and a perfect harness, **the number you get depends on how long you have been running.**

A device starts cold. Its first few seconds of inference run at full clocks with thermal headroom to spare — the *cold burst*. As you keep inferring, the silicon heats up. Once the junction temperature crosses the thermal limit, the hardware's own thermal management kicks in and **throttles**: it drops the clock (sometimes below the level you pinned, because thermal protection overrides your pin) to keep the chip from cooking. Now you are in the *sustained* regime, which can be 1.5–2x slower than the cold burst. The figure shows the two regimes side by side.

![Before and after comparison showing a device running fast during the cold burst then throttling to a much slower sustained latency once junction temperature crosses the thermal limit](/imgs/blogs/profiling-and-benchmarking-on-device-3.png)

A benchmark that runs for 2 seconds reports the cold burst. A product that runs continuously — a security camera, a wearable doing always-on wake-word detection, a drone doing real-time vision — lives in the sustained regime forever. If you benchmark the burst and ship to a continuous workload, you will miss your frame budget in the field and not understand why your lab numbers were fine.

So you must decide *which* number you are reporting and say so:

- **Cold/peak latency:** warm up just enough to settle JIT, then measure quickly before the chip heats. This is the best-case, the "spec sheet" number, and it is the right one for bursty workloads (snap a photo, classify, sleep).
- **Sustained latency:** run a continuous load for a *soak period* — typically 3–10 minutes — until the temperature plateaus, then measure. This is the right one for continuous workloads, and it is the number that predicts field behavior.

Measuring the throttle curve is itself useful: run a long loop, log the per-iteration latency *and* the junction temperature, and plot latency over time. You will see a flat fast region, a knee where throttling begins, and a higher plateau. The knee tells you how long your "fast" budget lasts; the plateau tells you the steady state.

#### Worked example: the throttle curve costs you 1.75x

On the same Orin Nano (15 W mode, clocks pinned), a continuous int8 inference loop on a small detector:

| Phase | Time into run | Junction temp | p50 latency | Throughput |
|---|---|---|---|---|
| Cold burst | 0–5 s | 45 °C | 12.0 ms | 83 inf/s |
| Warming | 30–60 s | 68 °C | 14.5 ms | 69 inf/s |
| Throttle knee | ~90 s | 80 °C | 18.0 ms | 56 inf/s |
| Sustained | 5 min+ | 84 °C | 21.0 ms | 48 inf/s |

The cold benchmark says 12 ms; the device a user holds says 21 ms — a **1.75x** gap, entirely from thermal throttling, with the model and the clocks held fixed. If your frame budget is 16 ms, the cold number says "pass" and the sustained number says "fail." Only one of them is true in the field. (The fix, beyond a better model: better thermal design — a heatsink or fan raises the throttle plateau; a lower power mode trades peak speed for a *higher sustained* speed because it never overheats. Measuring both modes tells you which wins for your duty cycle.)

## Separating data-loading and preprocessing from compute

One more way to lie to yourself, this time by accident: timing the wrong thing. End-to-end "inference latency" in a real pipeline is

$$
t_{\text{e2e}} = t_{\text{load}} + t_{\text{preprocess}} + t_{\text{compute}} + t_{\text{postprocess}},
$$

and on the edge these terms are often the same order of magnitude. Decoding a JPEG, resizing, normalizing, and laying out a tensor (the preprocess) can cost as much as the network forward pass for a small model. Copying the input from CPU to GPU and the output back (the host-device transfer) is pure overhead that a roofline analysis will never show you because it is not FLOPs.

If you measure $t_{\text{compute}}$ alone and report it as "inference latency," you understate what the user feels. If you measure $t_{\text{e2e}}$ but call it "model latency," you cannot tell whether to optimize the model or the data path. **Measure both, separately, and label them.** A common and painful discovery is that 40% of "inference time" is actually image decode and resize on the CPU — and the right fix is a faster decoder or GPU preprocessing, not a smaller model. You will not find that by timing only the forward pass.

A clean harness times each stage independently:

```python
import time, torch

def stage_time(fn, x, n=200, sync=True):
    for _ in range(20):           # warm-up
        fn(x)
    if sync: torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn(x)
    if sync: torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000   # ms per call

t_pre  = stage_time(preprocess, raw_image, sync=False)   # CPU work
t_h2d  = stage_time(lambda t: t.cuda(), cpu_tensor)      # host->device copy
t_fwd  = stage_time(model, gpu_tensor)                   # pure compute
print(f"preprocess {t_pre:.2f} ms | h2d {t_h2d:.2f} ms | forward {t_fwd:.2f} ms")
```

Now you know where the time goes *before* you decide what to optimize. This is the difference between engineering and guessing.

## Per-layer profiling: finding the one op that eats your latency

Median latency tells you *how slow* the model is. It does not tell you *why*. For that you need a per-operator profile: a breakdown of where the milliseconds actually go, op by op. This is where profiling earns its keep, because the answer is almost always surprising.

The recurring lesson — and the reason [the roofline post](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives) exists — is that **time is not proportional to FLOPs.** A layer with a tiny multiply-add count can dominate runtime if it is memory-bound (it moves a lot of data per FLOP) or if it falls back to an unoptimized kernel. Depthwise convolutions are the canonical example: they have very few FLOPs but low arithmetic intensity, so on many accelerators they are memory-bound and slow relative to their FLOP count. You will never see this from a FLOP table; you only see it in a profile. The figure shows a real-shaped per-layer breakdown where a depthwise conv eats 60% of the time despite a small FLOP share.

![Stacked breakdown of per-layer inference time showing a single depthwise convolution consuming sixty percent of latency while the FLOP-heavy pointwise layers take far less](/imgs/blogs/profiling-and-benchmarking-on-device-5.png)

### Profiling in PyTorch with `torch.profiler`

PyTorch's built-in profiler records every operator's CPU and CUDA time, memory allocations, and can emit a Chrome trace you open in a flame-graph viewer. The minimal idiomatic usage:

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity, schedule

x = torch.randn(1, 3, 224, 224, device="cuda")

# warm up first so JIT/autotune costs don't show up in the profile
for _ in range(20):
    model(x)
torch.cuda.synchronize()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=False,
) as prof:
    for _ in range(20):                 # profile a handful of steady-state runs
        with record_function("forward"):
            model(x)
        torch.cuda.synchronize()

# the most useful one-liner: rank ops by device time
print(prof.key_averages().table(
    sort_by="cuda_time_total", row_limit=15))

# write a Chrome trace for the flame-graph view (chrome://tracing or Perfetto)
prof.export_chrome_trace("trace.json")
```

The `key_averages().table()` output is a ranked list of operators by total device time, with call counts, average time, and shapes. The top few rows are your bottleneck. Reading it correctly takes a little practice:

- **Sort by `cuda_time_total` (or `self_cuda_time`)**, not CPU time, when you are GPU-bound. `self` time excludes child ops, which is what you want to attribute time to a specific kernel.
- **Watch the call count.** An op called 200 times when you expected 50 means a layer is being re-run — often a sign of a missing fusion or a control-flow bug.
- **Look for the gap between CPU time and CUDA time.** If CPU time on an op is large but CUDA time is tiny, you are **launch-bound**: Python/dispatch overhead, not compute, dominates. The fix is fusion, CUDA graphs, or a compiled runtime — not a smaller op.
- **Spot the CPU-fallback op.** This is the big one for edge. If your model runs on an NPU/GPU delegate but one operator is unsupported, the runtime silently moves that tensor back to the CPU, runs the op there, and copies the result back. In the profile this shows up as an op with surprisingly high time *plus* two memory-copy ops bracketing it. We will hunt that explicitly below.

For the Chrome trace, open `trace.json` in `chrome://tracing` or [Perfetto](https://ui.perfetto.dev) and you get a timeline flame graph: each kernel as a bar, host and device on separate tracks. The visual instantly reveals gaps (the device idle, waiting on the host — launch-bound) and the widest bars (your hot kernels).

#### Worked example: the depthwise conv eating 60% of the time

Profiling our MobileNetV3-Small on the Orin Nano, sorted by device time, the top of the table (abridged) looks like this:

| Operator | Calls | Self CUDA time | % of forward | FLOP share |
|---|---|---|---|---|
| `conv2d` (depthwise 3x3) | 11 | 9.1 ms | 60% | 6% |
| `conv2d` (stem 3x3) | 1 | 2.6 ms | 17% | 22% |
| `conv2d` (pointwise 1x1) | 27 | 1.8 ms | 12% | 55% |
| `hardswish` + `add` | 38 | 1.0 ms | 7% | 4% |
| `softmax` + classifier | 2 | 0.6 ms | 4% | 13% |

Read the first and last columns together and the lesson jumps out: the **depthwise convolutions are 6% of the FLOPs but 60% of the time**, while the pointwise 1x1 convolutions are 55% of the FLOPs but only 12% of the time. FLOPs predicted exactly the wrong priority. The depthwise ops are memory-bound — they stream a lot of activation data for very little arithmetic — so on this accelerator they are starved for bandwidth, not compute. If you had spent a week shrinking the pointwise layers (where the FLOPs are), you would have optimized 12% of the runtime. The profile tells you the leverage is in the depthwise path: fuse the conv-bn-activation, pick a layout that improves memory coalescing, or use a kernel tuned for depthwise. That is a profile-driven decision, and it is the opposite of what the FLOP count would have told you.

### Profiling with the runtime's own benchmark tools

You do not always have a Python harness on the target. Both major mobile runtimes ship a standalone benchmark binary that does warm-up, runs, and a per-op profile *correctly* — they have already solved the harness problem, and you should use them rather than reinventing a worse one.

**TFLite `benchmark_model`** runs a `.tflite` file on the device with proper warm-up, configurable runs, and an optional per-op profile:

```bash
# build/obtain benchmark_model, push to device (adb push for Android), then:
./benchmark_model \
  --graph=model_int8.tflite \
  --num_threads=4 \
  --warmup_runs=50 \
  --num_runs=1000 \
  --enable_op_profiling=true \
  --use_xnnpack=true        # or --use_gpu=true / --use_nnapi=true

# output reports: inference avg/min/max/std, plus a per-op time table
# the per-op table is sorted and shows which ops dominate and which
# fell back to the CPU reference kernel
```

The op-profiling table is the gold here: it lists each node, its time, and crucially whether it ran on the delegate (GPU/NNAPI/XNNPACK) or fell back. A row that ran on the CPU reference kernel inside a GPU-delegated model is your fallback culprit.

**ONNX Runtime** has both a Python profiler and the `onnxruntime_perf_test` binary. The Python path:

```python
import onnxruntime as ort

opts = ort.SessionOptions()
opts.enable_profiling = True            # emit a per-op profile JSON

sess = ort.InferenceSession(
    "model.onnx", opts,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

for _ in range(50):                      # warm up
    sess.run(None, {"input": x})
for _ in range(1000):                    # timed runs
    sess.run(None, {"input": x})

prof_file = sess.end_profiling()         # path to a Chrome-trace JSON
print("profile written to", prof_file)
```

The profile JSON lists each node's provider and duration. The thing to look for on the edge is **provider assignment**: ONNX Runtime partitions the graph across execution providers, and any node assigned to `CPUExecutionProvider` inside an otherwise-GPU model is a fallback. You can dump the partitioning by setting the session log severity to verbose, which prints the placement decision for every node.

### Hunting the CPU-fallback op explicitly

This deserves its own treatment because it is the most expensive surprise in edge deployment and the hardest to see without a profiler. The story: you quantized your model to int8 and targeted the NPU. The NPU supports 95% of your ops. The other 5% — maybe a custom activation, a `gather`, an unusual `reshape`, a `LayerNorm` variant — are not supported. The runtime does not error. It *silently* splits the graph: the supported ops run on the NPU, the unsupported op runs on the CPU, and the runtime inserts memory copies to shuttle the tensor between them.

Each fallback costs you three things: the slow CPU execution of the op itself, two device↔host memory copies, and — the killer — it *breaks the NPU pipeline*, forcing a synchronization point that destroys overlap. A single unsupported op in the middle of the network can cost more than all the ops it sits between. I have seen a model where 4% CPU-fallback ops accounted for 70% of the latency.

How to find it:

- In `torch.profiler`, look for ops whose device time is zero but CPU time is high, bracketed by `aten::copy_` or `to` ops (the host transfers).
- In the TFLite op-profiling table, look for nodes running on the reference (CPU) kernel inside a delegated run.
- In ONNX Runtime, dump the provider assignment and find nodes on `CPUExecutionProvider`.
- In TensorRT, check the build log for layers that could not be assigned to a tactic and fell back, and use `trtexec --dumpProfile` for the per-layer timing.

The fixes, in order of preference: replace the unsupported op with a supported equivalent (e.g., swap an exotic activation for a `relu6` the NPU has a kernel for); restructure the graph so the op is computed elsewhere; or, if you must keep it, *group* fallback ops so you pay the host round-trip once rather than ping-ponging. None of these are findable without a profile that shows you the op placement. This is why profiling is not optional on the edge: the FLOP count and the model summary look identical whether or not you have a 70%-latency fallback hiding in the graph.

#### Worked example: the fallback that cost more than the whole backbone

A team deployed a quantized segmentation model to a phone NPU and benchmarked it at 48 ms — four times their 12 ms target — despite a model whose FLOPs predicted comfortably under budget. The model summary was unremarkable. The op-placement profile told the real story: a single `resize_bilinear` upsampling op in the decoder was not in the NPU's supported set, so the runtime ran it on the CPU. That one op, plus its two device-to-host-and-back copies of a large feature map, accounted for 33 ms — more than the entire NPU-resident backbone. The fix was a two-line change: replace `resize_bilinear` with a `transpose_conv` (a learned upsample) the NPU *did* support, fine-tune for one epoch to recover the half-point of accuracy the swap cost, and re-export. Latency dropped from 48 ms to 11 ms — a 4.4x improvement — with no change to the backbone, no new quantization, and no architecture search. The leverage was entirely in the op-placement profile, and it was invisible to every other view of the model. This is the single highest-return profiling habit on the edge: before optimizing anything, dump the op placement and confirm the whole graph actually ran where you think it did.

### Reading a flame-graph trace

The ranked table tells you *which* ops are slow; the timeline trace tells you *why the device is idle*, which is a different and equally important question. When you open the exported `trace.json` in Perfetto or `chrome://tracing`, you get two kinds of track: a host (CPU) track showing Python dispatch and kernel launches, and a device track showing the actual kernels executing. Three patterns are worth recognizing on sight:

- **Wide gaps on the device track** mean the GPU/NPU is *waiting* — it has finished its queued work and the host has not enqueued the next kernel yet. This is the signature of a launch-bound model: the bottleneck is CPU-side dispatch overhead, not compute. The cure is to reduce the number of kernel launches (operator fusion), capture the launch sequence once (CUDA graphs), or move to a compiled runtime that does the launching ahead of time. Shrinking the kernels themselves does nothing here, because they are not the bottleneck — the gaps between them are.
- **A few very wide bars** on the device track are your hot kernels — the ones the ranked table also flagged. The trace adds the spatial context: are they back-to-back (good, the device is saturated) or separated by gaps (the host cannot feed them fast enough)?
- **A repeating host-then-device-then-host sawtooth** around a single op is the fallback pattern from the previous section, drawn out in time: a host copy, a CPU op, a host copy back, then the device resumes. Once you have seen it on a timeline, you will spot it instantly.

The table and the trace are complementary. Use the table to rank where the time goes; use the trace to decide whether the fix is a smaller op (compute-bound), fewer launches (launch-bound), or a removed fallback (placement bug).

## Measuring memory honestly: peak working set, not parameter count

Latency is one axis; memory is the other hard constraint, and it is the one that decides whether the model *runs at all*. As [memory is the real constraint](/blog/machine-learning/edge-ai/memory-is-the-real-constraint) argues at length, the number that matters is not the parameter count or the file size — it is the **peak working set**: the maximum bytes resident at any instant during inference. A model can have 3 MB of weights and need 40 MB of RAM at its peak because of a single fat activation tensor. On a microcontroller with 256 KB of SRAM, the peak activation, not the weights, is what fails the build.

Peak memory has three components:

$$
M_{\text{peak}} = M_{\text{weights}} + M_{\text{activations}}^{\max} + M_{\text{workspace}} + M_{\text{runtime}},
$$

where $M_{\text{weights}}$ is the parameters (fixed, the easy part), $M_{\text{activations}}^{\max}$ is the largest set of activation tensors that must be live simultaneously (depends on the graph and the execution order, *not* the total of all activations), $M_{\text{workspace}}$ is scratch space the kernels need (convolution algorithms can demand large workspaces), and $M_{\text{runtime}}$ is the framework/CUDA-context overhead. People budget for $M_{\text{weights}}$ and get killed by $M_{\text{activations}}^{\max}$.

The key subtlety is that activation memory depends on **liveness**, not total size. A tensor is "live" from when it is produced until its last consumer reads it. A good runtime reuses buffers for tensors whose lifetimes do not overlap, so the peak is the maximum *overlapping* liveness, which can be far below the sum of all activations. This is why the same model can have different peak memory under different runtimes — and why you must *measure* it on your actual runtime rather than computing it from the architecture.

### Measuring it in PyTorch

For CUDA, PyTorch tracks the peak allocation precisely:

```python
import torch

torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()

y = model(x)                                 # run one inference
torch.cuda.synchronize()

peak_bytes = torch.cuda.max_memory_allocated()
reserved   = torch.cuda.max_memory_reserved()   # includes the caching pool
print(f"peak allocated {peak_bytes/1e6:.1f} MB | reserved {reserved/1e6:.1f} MB")
```

`max_memory_allocated` is the high-water mark of *live* tensors — the number that matters. `max_memory_reserved` includes the caching allocator's pool, which is what the process actually holds from the OS. Report the allocated peak for the model's intrinsic need and the reserved figure for the process footprint. `torch.profiler` with `profile_memory=True` goes further: it attributes allocations to specific ops, so you can find *which* layer's activation is the fat one driving the peak — exactly the tensor to target with a smaller resolution, a tiling scheme, or an earlier downsample.

For CPU and for the whole-process footprint (which is what an OOM killer actually sees), measure resident set size (RSS):

```python
import os, psutil
proc = psutil.Process(os.getpid())
rss_before = proc.memory_info().rss
y = model(x)
rss_after = proc.memory_info().rss
print(f"RSS delta {(rss_after - rss_before)/1e6:.1f} MB, "
      f"total {rss_after/1e6:.1f} MB")
```

On a microcontroller there is no `psutil`; the relevant numbers are the **tensor arena size** (the static buffer TFLite-Micro pre-allocates for all activations — you size it once, with no `malloc` at runtime) and the **Flash footprint** (weights + code). You measure the arena by asking the interpreter for `arena_used_bytes()` after allocation and the Flash from the linker map. Those two numbers, against the chip's SRAM and Flash budget, decide whether the model fits — a topic the TinyML posts in this series go into in depth.

## Measuring energy: joules per inference, the number batteries care about

The third axis, and the one most often ignored because it is the hardest to instrument, is **energy**. For a battery-powered device — a wearable, a phone, a drone, a sensor — the question is not "how fast" but "how many inferences per charge," and that is governed by energy per inference, measured in millijoules (mJ) or, equivalently, milliwatt-hours.

The fundamental relationship: power is energy per unit time, so the energy of one inference is the integral of power over its duration,

$$
E = \int_{0}^{T} P(t)\, dt \approx \bar{P} \cdot T,
$$

where $\bar{P}$ is the average power during the inference and $T$ is its duration. The crucial and counterintuitive consequence: **faster is not always lower-energy.** If a higher clock doubles power to halve the time, energy is unchanged ($2\bar{P} \cdot \tfrac{T}{2} = \bar{P}T$). But "race to idle" often *does* save energy, because finishing fast lets the chip drop to a low-power idle state sooner, and idle power dominates the duty cycle of a bursty workload. Whether to optimize for speed or for energy depends on the duty cycle, and you cannot reason about it without measuring power, not just time. The figure shows the three measurement axes and the instruments each one needs.

![Stacked view of the three measurement tap points where latency comes from a synced timer memory from the peak working set and energy from a battery delta or power monitor](/imgs/blogs/profiling-and-benchmarking-on-device-7.png)

There are three ways to measure energy, in increasing order of fidelity.

**1. On-chip power telemetry.** Modern SoCs expose power rails through OS counters. This is the easiest and usually good enough.

- **NVIDIA Jetson — `tegrastats`:** prints CPU/GPU/SoC power on the VDD rails, in milliwatts, at a configurable interval. Log it during a benchmark and integrate.

```bash
# log power every 100 ms to a file while the benchmark runs
tegrastats --interval 100 --logfile power.log &
TEGRA_PID=$!
python run_benchmark.py            # your warm-up + N-run harness
kill $TEGRA_PID
# power.log lines include e.g. "VDD_GPU_SOC 3200mW/3100mW VDD_CPU_CV 800mW/..."
# parse the GPU+SoC columns, multiply mean power (W) by mean latency (s)
# E_per_inference = mean_power_W * latency_s   ->  joules per inference
```

- **Apple Silicon (macOS) — `powermetrics`:** reports CPU, GPU, and ANE (Apple Neural Engine) power in milliwatts.

```bash
# sample combined power every 200 ms; needs sudo
sudo powermetrics --samplers cpu_power,gpu_power -i 200 > power.txt &
PM_PID=$!
python run_benchmark.py
sudo kill $PM_PID
# look for "CPU Power", "GPU Power", and on ANE workloads "ANE Power" (mW)
```

- **Android — battery stats / Power Profiler:** `dumpsys batterystats`, the on-device Power Profiler in Android Studio, or per-rail counters on dev boards via `/sys/class/power_supply/`.

**2. Battery delta.** The crudest but most honest whole-system method: fully charge, run a fixed number of inferences in a tight loop for a long time (long enough that the battery percentage moves measurably), and divide the energy consumed by the count. This captures *everything* — compute, memory, the display if it is on, radios — which is exactly what the user experiences, but it cannot isolate the model from the rest of the system.

$$
E_{\text{per inf}} = \frac{C_{\text{battery}} \cdot V \cdot \Delta\text{SoC}}{N_{\text{inferences}}},
$$

where $C_{\text{battery}}$ is capacity (Ah), $V$ the nominal voltage, $\Delta\text{SoC}$ the fraction of charge consumed, and $N$ the inference count. Run with the screen off and radios quiet to attribute most of the delta to compute.

**3. External power monitor.** The gold standard: an inline USB power meter or a bench supply with logging (or a sense-resistor rig) measures the *actual* current the whole board draws from the wall, sampled at high rate. This is what MLPerf uses for its power-measured submissions because it cannot be gamed by the OS counters, which sometimes report modeled rather than measured power. For a microcontroller, a current probe on the supply rail gives you per-inference energy in microjoules — essential when your battery is a coin cell and the duty cycle is "wake, infer, sleep."

#### Worked example: joules per inference and inferences per charge

On the Orin Nano (15 W mode, sustained), our int8 detector runs at p50 = 21 ms, and `tegrastats` reports a mean total board power of about 6.8 W during the loop (the "15 W mode" is a cap, not the draw). The energy per inference:

$$
E = \bar{P} \cdot T = 6.8 \text{ W} \times 0.021 \text{ s} = 0.143 \text{ J} = 143 \text{ mJ}.
$$

Now suppose the same detector runs on a battery-powered companion device with a 20 Wh battery, doing always-on inference at one frame per 21 ms (sustained). Energy budget per hour is the power draw, 6.8 W, so the 20 Wh battery lasts about $20 / 6.8 \approx 2.9$ hours of continuous inference — and in that time it does roughly $2.9 \times 3600 / 0.021 \approx 497{,}000$ inferences. If a quantization or distillation step cuts the per-inference energy from 143 mJ to 95 mJ (by both speeding up the model and letting it idle more between frames at a lower duty cycle), battery life rises proportionally. *That* is the number a product manager for a wearable actually cares about, and you can only produce it by measuring power, not latency alone. Notice also: if cutting latency required *raising* the clock (more power), the energy might not improve at all — which is why you measure $E$, not assume $E \propto T$.

## The measurement-mistake catalog

We have covered a lot of failure modes. Here they are collected as a single reference: the mistake, the distortion it produces, and the one-line fix. The figure renders this as a matrix; the table below adds detail.

![Matrix mapping each common measurement mistake to its predictable distortion and a concrete one-step fix for honest benchmarking](/imgs/blogs/profiling-and-benchmarking-on-device-6.png)

| Mistake | What it does to your number | The fix |
|---|---|---|
| Timing the first (cold) run | 2–10x too slow; measures JIT/autotune/cold cache, not steady state | Warm up 20–100 runs, discard them, then measure |
| No device synchronization | Up to ~100x too fast on GPU/NPU; times the async launch, not the kernel | `cuda.synchronize()` / CUDA events / `mps.synchronize()` before stopping the timer |
| Reporting the mean | Inflated by spikes; not what a typical run feels | Report median (p50); add p90/p99 for the tail |
| Too few runs | Noisy, unrepeatable; tail invisible | Run hundreds–thousands; populate the tail |
| Free-running clocks | 10–30% run-to-run variance; numbers not comparable | Pin clocks: `nvpmodel` + `jetson_clocks`, `performance` governor |
| Cold-only (no soak) | Misses sustained throttling; 1.5–2x optimistic for continuous loads | Soak 3–10 min, report sustained as well as peak |
| Timing compute only | Hides preprocess + host-device copy that the user feels | Time each stage (load/preprocess/copy/compute) separately |
| Counting params/FLOPs as "memory/speed" | Ignores peak activations and memory-bound ops | Measure peak working set and per-op time on the real runtime |
| Ignoring CPU fallback | A 4%-of-ops fallback can be 70% of latency, invisible in summaries | Profile op placement; replace/group unsupported ops |
| Assuming faster = lower energy | Energy can be flat or worse if the clock rose to gain speed | Measure power and integrate to joules per inference |

This table is, honestly, the whole post in one screen. Print it; tape it to the wall next to whoever signs off on "the model got 2x faster."

## Why published edge numbers are usually not comparable

Now the uncomfortable part. Go read three blog posts or papers claiming latency for the same model on the same chip. You will frequently find numbers that differ by 2–3x. Almost never because anyone lied — because they measured *different things and called them the same thing.* One reported the cold burst; another the sustained. One had clocks pinned to MAXN; another ran the default governor. One used batch 8 and divided; another used batch 1. One reported the mean of a noisy run; another the p50. One included preprocessing; another timed only the forward pass. One used the GPU delegate; another fell back to CPU on one op without noticing.

Every one of those is a legitimate measurement. None of them are *comparable* unless the conditions are stated. This is why a latency number with no metadata is nearly worthless, and why the most valuable thing you can do for the next engineer (often future-you) is publish the full configuration alongside the number.

The fix is a reproducible-benchmark checklist: a fixed set of facts that must travel with every latency figure. The figure shows it as a stack; treat it as a required header on any number you report or trust.

![Stacked reproducible benchmark checklist listing the device runtime pinned clocks thermal state warm-up count run count batch precision and reported percentile](/imgs/blogs/profiling-and-benchmarking-on-device-8.png)

A number is comparable only if it states:

1. **Device + runtime + version** — e.g., "Jetson Orin Nano 8 GB, JetPack 6, TensorRT 8.6." Chip *and* software, with versions, because a runtime update can change latency 20%.
2. **Power mode + clocks** — e.g., "15 W mode (`nvpmodel -m 1`), `jetson_clocks` locked." Or "unpinned, default governor" if you genuinely could not pin them (state it).
3. **Thermal state** — "cold burst" or "sustained after 5 min soak," with the junction temperature if you have it.
4. **Warm-up count + run count** — "50 warm-up, 1000 timed runs." This lets someone reproduce your protocol exactly.
5. **Batch size + precision** — "batch 1, int8." Batch 1 is the edge reality; if you used a larger batch, say so, because per-image latency at batch 8 is not the same as latency at batch 1.
6. **Statistic reported** — "p50 = 5.9 ms, p99 = 8.1 ms." Median plus a tail, never a bare mean.
7. **What is included** — "forward pass only" or "end-to-end including JPEG decode and resize." Two very different numbers.

If a published number is missing these, you cannot use it to make a decision; at best it is a vibe. If your own number is missing these, you cannot defend it in a design review. Make the checklist a literal template in your benchmark script's output.

## Standardized benchmarks: MLPerf Mobile and MLPerf Tiny

You do not have to invent all of this protocol yourself. The ML community built standardized, audited benchmarks precisely to make edge numbers comparable, and their rules are a master class in honest measurement.

**MLPerf Mobile** (from MLCommons) benchmarks phones and edge SoCs on standard vision and language tasks. Its rules encode everything in this post: it requires a warm-up, mandates a *minimum run duration* so you cannot report a lucky cold burst, reports a sustained-performance figure to expose throttling, and has a separate **power-measured** category using an external meter (not OS counters) for energy. It also fixes the model, the dataset, and the accuracy target, so a faster number is only valid if accuracy is preserved — you cannot win by quietly degrading quality. That last rule is the link to the next post: speed only counts at fixed accuracy, which is the [accuracy–latency Pareto frontier](/blog/machine-learning/edge-ai/the-accuracy-latency-pareto-frontier) we explore next.

**MLPerf Tiny** does the same for microcontrollers: tasks like keyword spotting, visual wake words, and anomaly detection on Cortex-M-class hardware, with measured latency *and* energy via a standardized energy-measurement harness (the EEMBC EnergyRunner framework). For TinyML, where the whole point is fitting into kilobytes and microjoules, a standardized energy measurement is not a nicety — it is the only way to compare submissions at all.

Two things to take from MLPerf even if you never submit. First, its rules are a ready-made checklist: warm up, run long enough, report sustained, measure power externally, hold accuracy fixed. Adopt them in your internal benchmarks. Second, when you read a vendor's "Nx faster" claim, check whether it followed MLPerf-style rules or whether it is an unqualified marketing number. The former is comparable; the latter is, at best, a starting point for your own measurement on your own device.

## Stress-testing the protocol: when measurement gets harder

The clean protocol above assumes a cooperative device. The real world is messier. Let me walk through the cases that break the simple harness and how to handle them, because this is where junior and senior measurement diverge.

**The model is non-deterministic in shape.** An LLM generating text has a latency that depends on the prompt length and the number of tokens generated; a detector's postprocessing depends on how many objects it finds. There is no single "latency." For autoregressive models, report **time-to-first-token** (the prefill latency, which dominates the felt responsiveness) and **per-token latency / tokens-per-second** (the decode rate) separately — they have completely different bottlenecks, prefill being compute-bound and decode being memory-bandwidth-bound. Measure decode rate over a fixed, representative generation length and report the median across many prompts. (The on-device LLM posts in this series go deeper here.)

**The device cannot have its clocks pinned.** On a stock consumer phone you cannot root and lock frequencies. Then you embrace the variance instead of pretending it is gone: run many times across the thermal range, report the *distribution* (p50/p90/p99) and the sustained number after a soak, and disclose that clocks were not pinned. MLPerf Mobile does exactly this. An unpinned number with a full distribution and a soak is far more honest than a pinned number that nobody can reproduce on a retail device.

**The calibration of the measurement itself drifts.** Over a long soak the room warms, the battery depletes, background apps wake. Log a timestamp, temperature, and clock with *every* sample, not just the latency. Then if a benchmark looks weird, you can see whether a thermal event or a background task caused it, rather than re-running blind. Measurement metadata is cheap; debugging a mysterious result without it is expensive.

**The op is memory-bound, so the bottleneck is invisible to a timer.** A timer tells you a layer is slow; it does not tell you *why*. Cross-reference the per-op profile with the [roofline model](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives): compute the op's arithmetic intensity (FLOPs per byte moved) and compare it to the hardware's ridge point. If the op sits on the memory-bound side, no amount of FLOP reduction will speed it up — you need to move less data (lower precision, fusion to avoid round-trips to DRAM, a better layout). The profiler finds the slow op; the roofline tells you which knob actually moves it.

**The first measurement contradicts a second.** Two runs, same config, 30% apart — almost always a thermal or clock difference. Cool the device to a known state between runs (or soak both to steady state), pin clocks, and the variance collapses. If it does not, you have a real source of nondeterminism (a background process, a frequency that races, a runtime that re-autotunes) and the metadata log will point at it.

The throughline: every "hard case" is solved by *measuring more, labeling more, and reporting the distribution rather than a point.* When in doubt, collect another column of metadata.

## A complete, reusable benchmark harness

Let me assemble everything into one harness you can adapt. It warms up, synchronizes, runs N times with CUDA events, separates a preprocessing stage, measures peak memory, and prints a full reproducible report with percentiles. This is the thing to keep in your toolbox.

```python
import time, json, platform
import numpy as np
import torch

def benchmark(model, make_input, *, device="cuda",
              warmup=50, runs=1000, soak_seconds=0, label="model"):
    """Honest latency + memory benchmark. Returns a reproducible report dict."""
    model.eval()
    x = make_input().to(device)

    # --- warm-up (discarded): pays JIT / autotune / cold-cache costs ---
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # --- optional thermal soak to reach sustained state ---
    if soak_seconds > 0:
        t_end = time.perf_counter() + soak_seconds
        with torch.no_grad():
            while time.perf_counter() < t_end:
                model(x)
        if device == "cuda":
            torch.cuda.synchronize()

    # --- timed runs (device-clock timing via CUDA events on GPU) ---
    times = []
    with torch.no_grad():
        if device == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            for _ in range(runs):
                start.record()
                model(x)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))    # ms
        else:
            for _ in range(runs):
                t0 = time.perf_counter()
                model(x)
                times.append((time.perf_counter() - t0) * 1000)

    t = np.array(times)
    report = {
        "label": label,
        "device": torch.cuda.get_device_name() if device == "cuda" else platform.processor(),
        "torch": torch.__version__,
        "batch": x.shape[0],
        "warmup": warmup, "runs": runs, "soak_seconds": soak_seconds,
        "thermal": "sustained" if soak_seconds else "cold-ish",
        "p50_ms": round(float(np.percentile(t, 50)), 3),
        "p90_ms": round(float(np.percentile(t, 90)), 3),
        "p99_ms": round(float(np.percentile(t, 99)), 3),
        "mean_ms": round(float(t.mean()), 3),     # printed only to contrast with p50
    }
    if device == "cuda":
        report["peak_mem_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
    return report

# usage
rep = benchmark(model, lambda: torch.randn(1, 3, 224, 224),
                warmup=50, runs=1000, soak_seconds=0, label="mobilenetv3_int8")
print(json.dumps(rep, indent=2))
```

The report it prints is self-documenting: it carries the device, the runtime version, the batch, the warm-up and run counts, the thermal state, the percentiles, and the peak memory — most of the reproducibility checklist, emitted automatically. Set `soak_seconds=300` and run it again to get the sustained number. Diff the two reports and you have your throttle gap. Wrap it to also log `tegrastats`/`powermetrics` during the timed loop and you have energy too. The point is that an honest measurement is not more work once the harness is written — it is the *same* work, done correctly, every time.

## Putting it together: a measured before→after

Everything in this series claims a before→after improvement. Here is what a *defensible* one looks like, using our running MobileNetV3-Small on the Orin Nano, measured with the harness above. We compare the fp16 baseline against an int8 (post-training quantized) version, reporting the full checklist.

| Metric | fp16 baseline | int8 (PTQ) | Change |
|---|---|---|---|
| Device / runtime | Orin Nano 15 W, TRT 8.6 | Orin Nano 15 W, TRT 8.6 | same |
| Clocks / thermal | pinned, sustained 5 min | pinned, sustained 5 min | same |
| Batch | 1 | 1 | same |
| p50 latency | 9.4 ms | 5.9 ms | **1.6x faster** |
| p99 latency | 12.1 ms | 8.1 ms | 1.5x faster |
| Peak working set | 41 MB | 26 MB | 1.6x smaller |
| Model size on disk | 4.0 MB | 1.1 MB | 3.6x smaller |
| Energy / inference | 0.20 J | 0.14 J | 1.4x less |
| Top-1 accuracy | 67.4% | 66.9% | −0.5 pts |

Notice what makes this *defensible* rather than marketing: both numbers were measured the same way (same clocks, same thermal soak, batch 1, p50/p99, sustained), the accuracy delta is reported so the speedup is honest (a 1.6x speedup that cost 5 points of accuracy would be a different story), and energy is measured, not assumed from latency. This is one Pareto point you can act on — and reading several such points is exactly the [accuracy–latency frontier](/blog/machine-learning/edge-ai/the-accuracy-latency-pareto-frontier) the next post is built from. A row that did *not* hold accuracy, or that compared a cold int8 run against a sustained fp16 run, would be the kind of "win" that evaporates in the field.

## When to invest in rigorous measurement (and when a quick number is fine)

Measurement has a cost, and not every situation deserves the full protocol. Be deliberate.

**Always do the full protocol when:** you are choosing between models or techniques for a release (a wrong ranking ships the wrong model); you are publishing a number others will rely on; you are debugging a "fast in the lab, slow in the field" gap (almost always thermal or fallback); or you are validating that an optimization actually paid off (many "optimizations" are within the noise floor — if you cannot beat your own p50 variance, you did not improve anything).

**A quick, less-rigorous number is fine when:** you are doing a coarse feasibility check ("is this model in the ballpark of the budget, or off by 10x?") where a warm, synced single measurement on a cold chip answers the question; or you are iterating rapidly on architecture in a regime where the differences are large and obvious. Even then, *warm up and synchronize* — those two are non-negotiable because they fix order-of-magnitude lies, not just noise. The thermal soak, the 1000-run distribution, and the energy measurement are the parts you can skip for a quick look and must restore for a decision.

The trap to avoid is the false economy of skipping measurement to "save time" and then making a decision on a number that is off by 2x. The hour you spend on a proper harness is repaid the first time it stops you from shipping the slower model. And once the harness exists, the marginal cost of an honest measurement is essentially zero — which is the whole argument for building it once and reusing it everywhere, as the [capstone playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook) recommends making standard practice across a team.

## Key takeaways

- **A latency number without a protocol is a rumor.** Warm up, synchronize the device, run hundreds-to-thousands of times, and report the median plus a tail percentile. One cold, unsynced run can be off by 10–100x and can invert an A-vs-B ranking.
- **Synchronize or you are timing the launch, not the kernel.** On any async accelerator (GPU/NPU), the call returns before the work is done. `cuda.synchronize()`, CUDA events, or `mps.synchronize()` before you stop the timer. This is the single most common impossible-number bug.
- **Report the median, not the mean.** The median has a 50% breakdown point — spikes do not move it — so it is the robust estimate of the typical run. The mean is for throughput; the median is for what a user feels. Always add p90/p99 because on the edge the tail is the product.
- **Pin the clocks and declare the thermal state.** `nvpmodel` + `jetson_clocks`, or the `performance` governor, removes the largest source of variance. Then say whether your number is a cold burst or a sustained soak — the gap is routinely 1.5–2x and decides whether you make your frame budget in the field.
- **Profile per-op; FLOPs lie.** Time is not proportional to multiply-adds. A depthwise conv can be 6% of FLOPs and 60% of time because it is memory-bound. Use `torch.profiler`, the TFLite/ONNX Runtime benchmark tools, and the per-op table to find the real bottleneck.
- **Hunt the CPU-fallback op.** An unsupported op on an NPU silently falls back to CPU with two memory copies and a broken pipeline; 4% of ops can be 70% of latency. It is invisible in model summaries and only shows up in an op-placement profile.
- **Measure peak working set, not parameter count.** The bytes that decide whether the model runs are the maximum simultaneously-live activations plus workspace, measured on the real runtime — not the file size.
- **Measure joules, do not assume them.** Energy is power integrated over time; faster is not automatically lower-energy. Use `tegrastats`/`powermetrics`, a battery delta, or an external meter, and report joules (or mJ) per inference.
- **Publish the reproducibility checklist with every number** — device, runtime version, clocks, thermal state, warm-up, N, batch, precision, percentile, and what is included. Numbers without it are not comparable, which is why most published edge numbers are not.

## Further reading

- **Within this series:** [A taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) for the four-lever frame every measurement validates; [The metrics that actually matter on-device](/blog/machine-learning/edge-ai/the-metrics-that-actually-matter-on-device) for *which* numbers to chase; [Memory is the real constraint](/blog/machine-learning/edge-ai/memory-is-the-real-constraint) for peak-working-set reasoning; [The roofline model: where your bottleneck lives](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives) for *why* a low-FLOP op is slow; and next, [The accuracy–latency Pareto frontier](/blog/machine-learning/edge-ai/the-accuracy-latency-pareto-frontier) and the [capstone edge-optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook).
- **MLPerf Mobile / MLPerf Tiny** (MLCommons): the benchmark rules and result tables — the canonical reference for warm-up, minimum-duration, sustained-performance, and externally-measured-power protocols. See the MLPerf Inference: Mobile and MLPerf Tiny benchmark papers (Reddi et al. and Banbury et al.).
- **`torch.profiler` documentation** (PyTorch) and the PyTorch Profiler with TensorBoard / Perfetto tutorial — the per-op profiling and trace workflow shown here.
- **NVIDIA Nsight Systems and Nsight Compute** — the system-level timeline and kernel-level profilers for Jetson and CUDA, the deeper tools when `torch.profiler` is not enough.
- **TensorFlow Lite `benchmark_model`** and **ONNX Runtime performance tuning** docs — the standalone on-device benchmark binaries with correct warm-up and per-op profiling built in.
- **"Roofline: An Insightful Visual Performance Model for Multicore Architectures"** (Williams, Waterman, Patterson, 2009) — the foundation for reading a per-op profile against memory- vs compute-bound limits.
