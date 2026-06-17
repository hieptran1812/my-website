---
title: "The GPU Is Idle: Throughput Debugging from DataLoader to MFU"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Find out why your expensive GPU sits at 31 percent utilization and turn a starved training run into a saturated one, measured by util and MFU."
tags:
  [
    "debugging",
    "model-training",
    "throughput",
    "dataloader",
    "mfu",
    "profiling",
    "pytorch",
    "finetuning",
    "deep-learning",
    "torch-compile",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/the-gpu-is-idle-throughput-debugging-1.png"
---

Here is a number that should make you angry. You rented an 8×H100 node at roughly \$30 per hour, you launched a finetune that was supposed to take 18 hours, and `nvidia-smi` says your GPUs are sitting at 31 percent utilization. The card costs you the same whether it runs at 31 percent or 95 percent — the meter does not care. So you are not paying for compute, you are paying for *time*, and two-thirds of that time the most expensive silicon you have ever touched is doing nothing but waiting. The run that should have cost \$540 is going to cost \$1,600, finish a day and a half late, and you will tell yourself "training is just slow" and move on.

It is not slow. It is *idle*, and idle is a bug. A slow training run is a debugging problem with exactly the same shape as a NaN or a silent data leak: there is a symptom (low utilization, gaps between steps), there is a mechanism that produces it (one stage of the pipeline is starving the others), there is a diagnostic that localizes it (profile the run, time the dataloader alone, compute the model FLOPs utilization), and there is a fix that you can confirm with the instruments afterward. This post is about turning "training is slow" into "the dataloader is the bottleneck because `num_workers=2` and each sample does a 38 ms JPEG decode on the main process, here is the trace, here is the fix, util went 31 percent to 92 percent and wall-clock halved."

![A vertical stack of five training-step stages from dataloader through host-to-device copy, forward, backward, and optimizer step, each labeled with a time in milliseconds, showing that the step is capped by the slowest stage which is a 40 ms dataloader.](/imgs/blogs/the-gpu-is-idle-throughput-debugging-1.png)

I want to be precise about what we are debugging, because "throughput" is a vague word. The thing you actually control is the wall-clock time of one training step, and the thing that sets that time is the simple fact in figure 1: **a training step is a pipeline of serial stages — load a batch, copy it to the device, run the forward, run the backward, take an optimizer step — and the step time is the time of the slowest stage**, not the sum, if the stages overlap, and the sum if they do not. Every throughput bug in this post is a story about one stage being slow, or about the stages failing to overlap when they should. That framing — the step as a pipeline, the bottleneck as its slowest stage — is the spine of everything that follows, and it maps directly onto the six places a bug hides that this series keeps returning to: a throughput bug lives in **systems** (the dataloader, the transfer, the launch overhead, multi-GPU sync) far more often than in your model code. By the end you will be able to take any under-utilized run, find the slow stage in about ten minutes with `nvidia-smi` and the PyTorch profiler, fix it, and prove the fix with utilization and MFU. Let us start by reading the meter honestly, because the first lie a slow run tells you is on the dashboard you already have open.

## 1. The symptom: what "the GPU is idle" actually looks like

The crudest instrument is `nvidia-smi`, and it is the right place to start because it is already installed and it does not lie about the headline. Run it in a loop while your training is going:

```bash
# Sample GPU utilization and memory every 500 ms, one row per sample
nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,power.draw \
           --format=csv,noheader,nounits -l 1
```

You will see something like `0, 31, 18, 41203, 198`. Read that as: GPU 0 is 31 percent utilized, memory controller 18 percent busy, 41 GB used, drawing 198 watts. An H100 will draw 600–700 watts when it is genuinely working; 198 watts is a card that is mostly asleep. If you watch the utilization number for thirty seconds and it *flickers* — 95, 0, 0, 92, 0, 0, 94 — that flicker is the smoking gun. The card does a burst of work, then waits, then bursts again. That is the sawtooth of a starved GPU, and figure 2 shows the timeline that produces it.

![A horizontal timeline of one training run showing alternating blocks where the GPU is idle waiting for the next batch for forty milliseconds and then busy for forty-two milliseconds, ending with a summary that utilization reads thirty-one percent at eighty-two milliseconds per step.](/imgs/blogs/the-gpu-is-idle-throughput-debugging-2.png)

There is one critical caveat about the `utilization.gpu` number, and missing it sends people down wrong paths for hours: it is the fraction of the last sample window in which **at least one kernel was running**, not how *efficiently* those kernels used the hardware. A run that launches one tiny kernel that occupies a single SM for the whole window will report 100 percent "utilization" while using under one percent of the GPU's FLOPs. So `nvidia-smi` utilization is a great detector of *starvation* (the gaps) and a terrible detector of *inefficiency* (low arithmetic intensity, no tensor cores). For starvation it is enough; for everything past section 6 we will need MFU. For richer fleet-wide monitoring you would reach for DCGM (`dcgm-exporter` feeding Prometheus), which exposes SM-activity and tensor-core-activity counters that distinguish "a kernel is running" from "the math units are busy" — but for one run on one box, the smi loop plus the profiler covers it.

The second symptom worth naming is *step time variance*. Log the wall-clock time of each step and look at the distribution, not the mean:

```python
import time, torch

step_times = []
for step, batch in enumerate(loader):
    t0 = time.perf_counter()
    batch = {k: v.to("cuda", non_blocking=True) for k, v in batch.items()}
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()          # so the timer measures real work, see section 4
    step_times.append(time.perf_counter() - t0)
    if step == 200:
        break

import numpy as np
a = np.array(step_times[20:])         # drop warmup
print(f"median {np.median(a)*1e3:.1f} ms  p90 {np.percentile(a,90)*1e3:.1f} ms  "
      f"max {a.max()*1e3:.1f} ms  cv {a.std()/a.mean():.2f}")
```

A healthy compute-bound run has a tight distribution: median 39 ms, p90 41 ms, max 44 ms, coefficient of variation under 0.1. A dataloader-bound run has a long right tail — median 42 ms, p90 78 ms, max 210 ms — because every so often the worker pool falls behind and a step has to *wait* for a batch that is not ready. That long tail is starvation made into a number. When you see it, you already know the suspect is the input side; the rest is confirmation.

There is a particular shape worth naming because it is so diagnostic: **periodic spikes at the epoch boundary**. If your step times are tight for thousands of steps and then, at the start of each epoch, you get a cluster of slow steps before settling back down, that is the worker pool *refilling its prefetch buffer* from empty at epoch start. With `persistent_workers=False` (the default), PyTorch tears the worker processes down at the end of each epoch and respawns them at the start of the next — paying the process-fork cost and the cold-cache cost every epoch. Setting `persistent_workers=True` keeps the workers alive across epochs and erases that periodic spike. It is a one-line fix for a signature that people often misread as "the data is harder at the start of the epoch" (it is not; the data is shuffled). When you see a sawtooth whose period equals your epoch length, the suspect is worker lifecycle, not the data.

A second diagnostic shape is the **slow first few steps of the entire run** that then settle. That is normal and you must exclude it from every measurement: it is workers spinning up, the prefetch buffer filling from cold, CUDA context initialization, cuDNN autotuning its convolution algorithms (the `torch.backends.cudnn.benchmark = True` warmup, which pays a one-time cost to find the fastest conv kernel for each input shape), and — if you use it — `torch.compile` tracing. Every harness in this post drops a warmup window for exactly this reason. The mistake of including warmup in a benchmark is so common that it deserves a rule: **never report a throughput number that includes the first ~20 steps.** Those steps are measuring your framework's startup, not your training's steady state.

## 2. The science: the step as a pipeline, and why overlap is everything

Let me make the pipeline framing rigorous, because the fix for every throughput bug falls out of it directly. Call the five stage times $t_{\text{load}}$, $t_{\text{h2d}}$, $t_{\text{fwd}}$, $t_{\text{bwd}}$, $t_{\text{step}}$. The forward, backward, and optimizer step happen on the GPU; the load happens on the CPU (workers); the host-to-device copy happens on the PCIe/NVLink bus. These run on *different hardware*, which is the whole point — they can overlap.

If nothing overlaps, the step time is the sum:

$$
t_{\text{naive}} = t_{\text{load}} + t_{\text{h2d}} + t_{\text{fwd}} + t_{\text{bwd}} + t_{\text{step}}.
$$

But a `DataLoader` with workers prefetches: while the GPU is busy with step $n$, the worker processes are already decoding and collating batch $n+1$ on the CPU, and the prefetch buffer hands it over the instant the GPU is ready. When prefetch works, the load stage is *hidden* behind the GPU compute, and the step time collapses to

$$
t_{\text{pipelined}} = \max\!\big(t_{\text{load}},\; t_{\text{h2d}} + t_{\text{fwd}} + t_{\text{bwd}} + t_{\text{step}}\big).
$$

This is the single most important equation in the post. It says the dataloader is *free* — completely hidden — as long as $t_{\text{load}} \le t_{\text{gpu}}$, where $t_{\text{gpu}} = t_{\text{h2d}} + t_{\text{fwd}} + t_{\text{bwd}} + t_{\text{step}}$. The moment the load takes *longer* than the GPU work, the GPU finishes step $n$ and the next batch is not ready, so it stalls. The excess, $t_{\text{load}} - t_{\text{gpu}}$, is pure idle time per step, and it shows up exactly as the sawtooth in figure 2.

Now you can predict utilization. If the GPU is busy for $t_{\text{gpu}}$ out of a step that lasts $\max(t_{\text{load}}, t_{\text{gpu}})$, then

$$
\text{GPU util} \approx \frac{t_{\text{gpu}}}{\max(t_{\text{load}},\, t_{\text{gpu}})}.
$$

Plug in the figure-1 numbers: GPU work is $6 + 12 + 24 + 6 = 48$ ms... but wait, that includes the H2D copy which overlaps too on a separate stream. Take the pure compute as $t_{\text{gpu}} \approx 42$ ms and the load as $t_{\text{load}} = 40$ ms when workers keep up. Then util $\approx 42 / 42 = 100$ percent — fine. But if the workers *cannot* keep up, because there are too few of them or the per-sample work is too heavy, the effective load time balloons. Suppose each sample takes 38 ms of CPU work and your batch is 32 samples: that is $32 \times 38 = 1216$ ms of CPU work per batch. Split across `num_workers=2` that is 608 ms per batch — fifteen times the GPU's 42 ms. Util collapses to $42/608 \approx 7$ percent. The fix is not subtle: you need enough workers that $t_{\text{load}}/n_{\text{workers}} \le t_{\text{gpu}}$. With 32 samples at 38 ms each and a 42 ms GPU step, you need at least $\lceil 1216/42 \rceil = 29$ workers, which is unrealistic — so the *real* fix is to make each sample cheaper (precompute, faster decode), and *then* add workers. We will do both.

This equation also tells you the ceiling. No amount of dataloader tuning gets you below $t_{\text{gpu}}$, because the GPU work is irreducible at a fixed batch and model. Once $t_{\text{load}} \le t_{\text{gpu}}$, the input side is solved and you must move to the device side — small-op overhead, dtype, tensor cores, MFU. That is the branching structure in figure 6, and it is why "add more workers" is the right first move and the wrong only move.

One refinement that matters for finetuning specifically: **gradient accumulation changes the arithmetic of this equation in your favor.** If you accumulate gradients over $k$ micro-batches before stepping the optimizer, the GPU does $k$ forward+backward passes per optimizer step, so $t_{\text{gpu}}$ per *optimizer step* grows roughly $k\times$ while the dataloader still has to produce $k$ batches in that window. The ratio $t_{\text{load}}/t_{\text{gpu}}$ is unchanged per micro-batch, so accumulation does not by itself fix a starved loader — but it does mean that a fixed dataloader latency is amortized over more compute, which slightly relaxes the worker requirement. The trap is the opposite case: people use a *tiny* micro-batch (to fit memory) without accumulation, which shrinks $t_{\text{gpu}}$ per step toward the launch-and-sync overhead floor, making the run launch-bound *and* easier to starve. If your micro-batch is small for memory reasons, accumulation is often a throughput win on top of a correctness one, because larger effective work per step hides both the loader and the per-step fixed overhead better. The interaction between accumulation and effective batch is a correctness topic in its own right; here the throughput point is simply that bigger per-step GPU work makes everything else easier to hide.

There is also a *prefetch depth* subtlety hiding in the word "buffer." The `DataLoader`'s `prefetch_factor` controls how many batches *per worker* are buffered ahead. The default is 2, so with 8 workers you have up to 16 batches queued. If your per-batch latency is *bursty* — most batches are fast but occasionally one sample triggers a slow path (a large image, a cache miss, a garbage-collection pause in a worker) — a deeper buffer absorbs the burst so the GPU never sees it. Raising `prefetch_factor` to 4 trades a little host memory for a smoother feed, and it is the right knob when your step-time distribution has a long tail despite enough workers on average. It does not raise *average* throughput (that is set by worker throughput), but it converts a spiky feed into a steady one, which is what keeps util pinned near 100 percent rather than sawtoothing around it.

#### Worked example: predicting the stall from the numbers

You profile a ResNet-50 finetune on a single A100. The pure GPU compute is $t_{\text{gpu}} = 55$ ms per step at batch 256. You time the dataloader alone (section 3) and find it produces a batch every 140 ms with `num_workers=4`. Predict the outcome before you change anything.

Since $t_{\text{load}} = 140 > t_{\text{gpu}} = 55$, the GPU stalls $140 - 55 = 85$ ms every step. Predicted util $= 55/140 = 39$ percent. Predicted step time $= 140$ ms. Now: the per-sample CPU work (JPEG decode + resize + normalize) is the cost, and four workers give $140$ ms/batch, so one worker gives $\approx 560$ ms/batch and the per-sample cost is $560/256 \approx 2.2$ ms. To hide the load you need $t_{\text{load}} = 560/n \le 55$, i.e. $n \ge 11$ workers. You set `num_workers=12`, and indeed util jumps to 91 percent and step time drops to 60 ms (a touch above $t_{\text{gpu}}$ because of scheduling jitter). You did not guess; you computed the worker count from the per-sample cost and the GPU step time, and the meter confirmed it. That is the whole method: measure the two stage times, compare them, fix the slower one.

## 3. Diagnostic: time the dataloader *alone* (the highest-leverage test)

The make-it-fail-small move for throughput is to **remove the model and iterate the dataloader by itself**. If the dataloader alone cannot feed batches faster than your GPU consumes them, no GPU-side optimization will help, and you have localized the bug to the input pipeline in thirty seconds. This is the throughput analogue of the [overfit-a-single-batch test](/blog/machine-learning/debugging-training/the-overfit-a-single-batch-test): a cheap experiment that rules an entire region in or out before you touch anything.

```python
import time, torch
from torch.utils.data import DataLoader

def time_dataloader(loader, n_batches=100, warmup=10):
    """Iterate the loader with NO model. Returns batches/sec and ms/batch."""
    it = iter(loader)
    for _ in range(warmup):          # let workers spin up + fill prefetch buffer
        next(it)
    t0 = time.perf_counter()
    seen = 0
    for _ in range(n_batches):
        batch = next(it)             # this is the ONLY work: produce a batch
        seen += 1
    dt = time.perf_counter() - t0
    bps = seen / dt
    print(f"{bps:6.1f} batches/s   {1e3*dt/seen:6.1f} ms/batch   "
          f"(workers={loader.num_workers}, pin={loader.pin_memory})")
    return bps

loader = DataLoader(train_ds, batch_size=256, shuffle=True,
                    num_workers=4, pin_memory=True)
time_dataloader(loader)
```

Compare the `ms/batch` this prints against your measured GPU step time. Three outcomes, three conclusions:

- **Loader ms/batch ≪ GPU ms/step.** The input side is fast enough; your bottleneck is on the device. Go to section 6 (small ops, dtype, MFU). Do not touch the dataloader.
- **Loader ms/batch ≈ GPU ms/step.** You are on the knife's edge; small variance will cause intermittent stalls. Add a little headroom (more workers, `prefetch_factor`) and move on.
- **Loader ms/batch ≫ GPU ms/step.** Confirmed dataloader bottleneck. Now bisect *within* the loader: is it the worker count, the per-sample transform, the disk I/O, or the collate?

To bisect within the loader, sweep the knobs and watch the number move:

```python
for nw in [0, 2, 4, 8, 16]:
    ld = DataLoader(train_ds, batch_size=256, shuffle=True,
                    num_workers=nw, pin_memory=True,
                    persistent_workers=(nw > 0),
                    prefetch_factor=(4 if nw > 0 else None))
    time_dataloader(ld, n_batches=50)
```

If throughput climbs with `num_workers` and then plateaus, you were worker-starved and the plateau is where the *per-sample work* (or disk, or the GIL on the main process collate) becomes the limit. If `num_workers=0` is already close to the multi-worker number, the bottleneck is *not* parallelizable per-sample work — it is probably one shared resource (a single slow disk, a network mount, a global lock) and adding workers will not help. If throughput rises then *falls* past some worker count, you have oversubscribed CPU cores and the workers are fighting each other; back off. This sweep, by itself, has ended more "my GPU is slow" investigations than any profiler trace.

One more sub-test: separate the *transform* cost from the *I/O* cost. Wrap your dataset's `__getitem__` to time itself, or just replace the transform with a no-op and re-time:

```python
class TimedDataset(torch.utils.data.Dataset):
    def __init__(self, base): self.base = base; self.t = 0.0; self.n = 0
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        t0 = time.perf_counter()
        x = self.base[i]
        self.t += time.perf_counter() - t0; self.n += 1
        return x
# After iterating: print(ds.t / ds.n * 1e3, "ms/sample")  (run with num_workers=0)
```

If `__getitem__` is 38 ms/sample and 35 of those are the JPEG decode, you fix the decode; if 30 are a synchronous read from a network filesystem, you fix the storage (local SSD, sharded `webdataset`, an LMDB/`FFCV` cache). Knowing *which* is the difference between a five-minute fix and a five-hour one.

### The storage-bound case and streaming formats

The nastiest input bottleneck is not CPU work at all — it is **synchronous random reads from slow storage**. If your dataset is a million small files on a network filesystem (NFS, S3 mounted via FUSE, a cloud bucket), then `__getitem__` does a tiny read that costs a full network round-trip — often 5–50 ms of latency that is *pure wait*, not compute, and that no number of CPU workers can parallelize away once you saturate the connection's IOPS. The signature is the one from section 3's worker sweep: throughput barely improves with more workers, because the bottleneck is one shared, latency-bound resource. The math units are idle, the CPUs are idle, and everyone is waiting on the network.

The fix is to stop doing random small reads. Pack the data into **large sequential shards** and stream them. The `webdataset` format stores samples in tar files of, say, 1 GB each; a worker opens a shard once and reads it sequentially, turning a million tiny random reads into a few hundred big sequential ones — which is what spinning disks, SSDs, and object stores are all fast at. Streaming formats also let you avoid downloading the whole dataset before training starts:

```python
import webdataset as wds

# Sequential reads from shards; each worker streams its own shards.
dataset = (
    wds.WebDataset("s3://bucket/imagenet-train-{000000..001281}.tar",
                   shardshuffle=True, nodesplitter=wds.split_by_node)
    .shuffle(1000)                       # buffer shuffle, not random-access shuffle
    .decode("pil")                       # decode happens in the worker, parallelized
    .to_tuple("jpg", "cls")
    .map_tuple(transform, lambda y: y)
)
loader = wds.WebLoader(dataset, batch_size=256, num_workers=8,
                       pin_memory=True, prefetch_factor=4)
```

Two things to understand about streaming. First, you give up *exact* global shuffling — `webdataset` shuffles shards and then shuffles a buffer within them, which is statistically fine for training but is not a true permutation; if your training is sensitive to that (rare), bucket carefully. Second, the decode and transform still run in the workers, so a streaming format fixes the *I/O* bottleneck but not a *CPU-decode* bottleneck — you may still need a fast decoder on top. The two fixes compose: shard for I/O, fast-decode for CPU, enough workers to hide both behind the GPU step.

The other heavy hammer is to **precompute and cache** the expensive part. If your per-sample transform is deterministic (resize to a fixed resolution, tokenize text, extract mel-spectrograms), do it once, write the result to disk in a compact format, and have training read the precomputed tensors. This trades disk space and a one-time preprocessing pass for a near-zero per-sample cost at train time. For text finetuning this is standard: tokenize and pack the corpus into fixed-length `uint16` token arrays once, and each epoch is then a sequential read of integers with no tokenization at all. The rule: **any deterministic transform on the hot path is a candidate to precompute.** Only the *random* augmentations must run per-epoch; the deterministic preprocessing should run zero times during training.

#### Worked example: the network filesystem that capped the run at 18 percent

A team trains a CLIP-style model on 50 million image-text pairs stored as individual files on a cloud bucket mounted via FUSE. Util sits at 18 percent regardless of `num_workers` — they tried 4, 8, 16, 32, and the number barely moved past 8. They blamed the GPU, then the model, then PyTorch. The dataloader-alone test told the truth in thirty seconds: 230 ms/batch at every worker count above 8, with the GPU step at 41 ms. Worker count not helping is the storage-bound signature. They timed `__getitem__`: 1.2 ms of decode, 26 ms of *read latency* per sample — the FUSE round-trip dominated, and 8 workers already saturated the mount's IOPS. The fix was to repack the 50M pairs into 1 GB `webdataset` shards on local NVMe-backed storage and stream sequentially. New dataloader-alone: 33 ms/batch (now under the 41 ms GPU step). Util 18 → 90 percent. Wall-clock per epoch dropped from 31 hours to 6.5 hours — a 4.8× speedup with **zero change to the model, the optimizer, or the hyperparameters**. The bug was never in the training; it was in the storage layout, and the worker sweep pointed straight at it.

For quick reference, here are the `DataLoader` knobs that actually move throughput, what each does, and when it helps:

| Knob | What it does | When it helps | Default trap |
|---|---|---|---|
| `num_workers` | parallel processes producing batches | per-sample CPU work (decode/augment) | default 0 = serial, on main process |
| `pin_memory` | page-locks host buffers for async copy | any GPU run (required for non_blocking) | default False = blocking copies |
| `prefetch_factor` | batches buffered ahead per worker | bursty per-sample latency | default 2 may be too shallow |
| `persistent_workers` | keep workers alive across epochs | erases per-epoch spawn spike | default False respawns each epoch |
| `non_blocking` (on `.to`) | copy on a side stream, return now | overlapping H2D with compute | no-op on unpinned memory |
| batch size | work per step, amortizes overhead | launch-bound / small-step runs | too small = launch-bound floor |

None of these is a silver bullet; each maps to a specific bottleneck the diagnostics in this post identify. The discipline is to *measure first* (dataloader-alone, the worker sweep, the per-sample timer) and then turn the one knob that addresses the bottleneck you found — not to flip them all and hope.

## 4. The trap that fools every timing harness: asynchronous CUDA

Before we go further, I have to stop and warn you about the single most common mistake in throughput debugging, because if you get this wrong every number you measure is fiction. **CUDA kernels launch asynchronously.** When you write `loss = model(x); loss.backward()`, the Python returns almost immediately — it has only *enqueued* the kernels onto the GPU's stream. The GPU runs them later, in the background. So if you time like this:

```python
t0 = time.perf_counter()
loss = model(x)          # enqueues forward kernels, returns instantly
loss.backward()          # enqueues backward kernels, returns instantly
dt = time.perf_counter() - t0   # measures Python enqueue time, NOT GPU work!
```

you measure how long it took Python to *launch* the work, which might be 2 ms for 40 ms of actual GPU work. Your timings will be absurdly fast and completely wrong. To measure real GPU time you must either call `torch.cuda.synchronize()` before reading the clock (cheap, correct, what the harness in section 1 does) or use CUDA events, which are more precise because they timestamp on the device:

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
loss = model(x); loss.backward(); optimizer.step()
end.record()
torch.cuda.synchronize()             # wait for the events to be reached
print(f"{start.elapsed_time(end):.2f} ms")   # device-measured, accurate
```

Here is the subtle, important part — and it connects directly to a *real* bug, not just a measurement artifact. The reason `nvidia-smi` can show low utilization is that the async queue runs *ahead* of the Python loop, keeping the GPU fed even though Python is off doing other things. That is the design and it is good. But the instant your loop reads a value *off* the GPU — `loss.item()`, `tensor.cpu()`, `print(loss)`, `if loss > threshold` — Python must **block until that specific value is computed**, which means draining the queue and idling the GPU until the CPU catches up. That is not just a measurement problem; it is a *throughput* problem, and it is section 5. The connection is exact: the same synchronize that makes your timing correct is the thing that, if it happens every step inside your loop, serializes your training and craters your throughput.

## 5. The sync-point bug: how `loss.item()` halves your throughput

This is my favorite throughput bug because it is invisible, it is in almost everyone's training loop, and the fix is to delete code. Figure 4 shows the mechanism: any operation that reads a GPU value into Python forces a `cudaDeviceSynchronize`, which drains the asynchronous queue and stalls the pipeline.

![A graph showing four sources of GPU synchronization — calling item on the loss, calling cpu or numpy, printing the loss, and a conditional that reads a GPU value — all flowing into a single synchronize node that drains the asynchronous queue, which then flows into a hot-loop stall of three milliseconds per step.](/imgs/blogs/the-gpu-is-idle-throughput-debugging-4.png)

The classic offender is logging the loss every step:

```python
# BUG: forces a sync every single step
for step, batch in enumerate(loader):
    loss = model(**batch).loss
    loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
    running_loss += loss.item()          # <-- .item() blocks until loss is ready
    if step % 1 == 0:
        wandb.log({"loss": loss.item()}) # <-- and again
```

Each `.item()` says to the GPU: "stop the line, I need *this exact number* in Python *right now*." The async queue, which was happily running three steps ahead, drains; the GPU sits idle while the CPU catches up; then it refills. On a small model where the per-step compute is short relative to the launch and sync overhead, this can be 5–15 percent of step time. I have measured a transformer finetune go from 1,180 tokens/sec to 1,310 tokens/sec — an 11 percent speedup — by deleting one `.item()` from the hot loop. The fix is to **accumulate the loss as a GPU tensor and only sync occasionally**:

```python
# FIX: keep the running loss on-device, sync only every `log_every` steps
running = torch.zeros((), device="cuda")
for step, batch in enumerate(loader):
    loss = model(**batch).loss
    loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
    running += loss.detach()             # stays on GPU, no sync
    if step % log_every == 0:
        # one sync per `log_every` steps instead of every step
        wandb.log({"loss": (running / log_every).item()})
        running.zero_()
```

The same discipline applies to gradient-clipping if you read the returned norm (`clip_grad_norm_` returns a tensor — do not `.item()` it every step unless you log it), to early-stopping checks (`if loss.item() < best` every step is a sync; compare on-device or every N steps), and to any metric you compute inside the loop. The rule: **nothing leaves the GPU on the hot path except on a logging cadence.** This connects to the logging discipline in [instrumenting a training run](/blog/machine-learning/debugging-training/instrumenting-a-training-run-what-to-log) — the same `.item()` that makes a log line cost you a sync is the one to batch behind a cadence.

How do you *find* the sync points you did not know about? The PyTorch profiler will surface them, but there is a brutal-force debugging mode that makes any accidental sync throw an exception so you can catch it by stack trace:

```python
# Raises an error at the exact line that triggers a host-device sync.
torch.cuda.set_sync_debug_mode("error")   # or "warn"
# ... run a few steps; the traceback points at the offending .item()/.cpu()
torch.cuda.set_sync_debug_mode("default")  # turn it off after debugging
```

Run a handful of steps with this on, read the traceback, and you will find every place your code accidentally reaches into the GPU. It is the throughput equivalent of `set_detect_anomaly` for NaNs: noisy, slow, and exactly what you want when you are hunting.

#### Worked example: the print statement that cost \$400

A team is finetuning a 7B model on 4×A100 for 40 hours, budgeted at roughly \$8 per GPU-hour, so \$1,280. Someone left a `print(f"step {step}: loss {loss.item():.4f}")` in the loop "for debugging" and never removed it. With 4 GPUs and DDP, that `.item()` does not just sync one device — it sits on the critical path of every step on rank 0, and because the ranks must stay in lockstep for gradient all-reduce, the stall propagates. Measured step time with the print: 0.93 s. Without it: 0.81 s. That is a 13 percent slowdown, which on a 40-hour run is 5.2 hours of wall-clock, and at 4 GPUs × \$8 that is **\$166 of pure waste from one print statement** — and if the run is repeated across a hyperparameter sweep of three configs, \$400+. The fix was to delete the line. The confirming test: step time 0.93 s → 0.81 s, util 71 percent → 84 percent, and `set_sync_debug_mode("warn")` went from screaming every step to silent. No model change, no data change — a systems bug, found by reading the loop for sync points.

## 6. Host-device transfer stalls and `non_blocking`

The host-to-device copy is the second pipeline stage, and it has its own failure mode. The copy from CPU memory to GPU memory happens over PCIe (or NVLink on newer boxes), and by default a `.to("cuda")` call on pageable host memory is *synchronous* — it blocks the CPU until the copy finishes, which means it cannot overlap with GPU compute. Two things make the copy overlap-friendly:

```python
# 1. pin_memory=True in the DataLoader makes the host buffer page-locked,
#    which is required for an async copy.
loader = DataLoader(ds, batch_size=256, num_workers=8,
                    pin_memory=True, prefetch_factor=4, persistent_workers=True)

# 2. non_blocking=True lets the copy run on a side stream and return immediately,
#    so the next thing the CPU enqueues (the forward) overlaps the copy.
for batch in loader:
    x = batch["input_ids"].to("cuda", non_blocking=True)
    y = batch["labels"].to("cuda", non_blocking=True)
    out = model(x)        # enqueued; overlaps the copy on the compute stream
```

The catch — and this is a real correctness trap, not just performance — is that `non_blocking=True` only actually overlaps when the *source* is pinned memory. On pageable memory it silently falls back to a blocking copy, so you get the API but not the benefit. That is why `pin_memory=True` and `non_blocking=True` are a *pair*: one without the other is half a fix. The DataLoader's `pin_memory=True` pins the batch in a background thread so the copy in your loop can be truly async.

There is a third subtlety that bites at scale. If the per-batch tensor is large (a big batch of high-res images, or long sequences) and your PCIe bandwidth is the limit, the H2D copy itself can become the bottleneck — $t_{\text{h2d}}$ exceeds the time the GPU has to hide it. The fix there is to move work *across* the boundary differently: do the cheap CPU-side transforms (decode, crop) on the CPU but the expensive normalization on the GPU after a small transfer, or transfer `uint8` images and convert/normalize on-device (a uint8 image is 4× smaller than the float32 version, so you copy a quarter of the bytes and do the float conversion where it is free). Profiling will tell you whether H2D is your problem; if the `Memcpy HtoD` rows in the trace are a meaningful fraction of step time, it is.

Let me put a number on the uint8 trick because it is one of the highest-leverage transfer fixes for vision. A batch of 256 RGB images at 224×224 in float32 is $256 \times 3 \times 224 \times 224 \times 4 = 154$ MB. The same batch as uint8 is 38 MB. Over a 16 GB/s effective PCIe link, that is 9.6 ms versus 2.4 ms per batch — a 7 ms saving every step, which on a 40 ms step is meaningful, and it also frees the CPU from doing the float conversion. You transfer the small `uint8` tensor and do the `.float()` and the mean/std normalization in the first line of your forward, where it runs on the GPU at memory speed and overlaps with nothing because it is on the compute stream anyway. The pattern:

```python
# Worker returns uint8 HWC; convert + normalize on the GPU, not the CPU.
mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
std  = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

for batch in loader:                                # batch["img"] is uint8 NCHW
    x = batch["img"].to("cuda", non_blocking=True)  # 38 MB copy, not 154 MB
    x = x.float().div_(255).sub_(mean).div_(std)    # normalize on-device
    out = model(x)
```

There is a related *layout* lever worth knowing: `channels_last` memory format. Convolutions on tensor cores are fastest when the tensor is laid out NHWC (channels last) rather than the PyTorch default NCHW, because that is the layout the tensor-core conv kernels want. Converting the model and inputs to `channels_last` (`model.to(memory_format=torch.channels_last)` and `x = x.to(memory_format=torch.channels_last)`) can give a real convolution speedup on Ampere+ GPUs in mixed precision, with no accuracy change — it is purely a memory-layout match to what the hardware prefers. Like the multiple-of-8 rule, it is a case of giving the tensor cores the shape they want; the profiler will show the conv kernels switching from the slow NCHW variant to the fast NHWC one.

## 7. The PyTorch profiler: seeing the gaps in the timeline

`nvidia-smi` tells you *that* the GPU is idle. The PyTorch profiler tells you *why* — it captures a timeline of every operator on CPU and every kernel on GPU, and the gaps in the GPU track are your idle time, annotated with what the CPU was doing during them. This is the instrument that turns a guess into a diagnosis.

```python
from torch.profiler import profile, schedule, ProfilerActivity, tensorboard_trace_handler

prof = profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=3, repeat=1),  # skip 1, warm 1, record 3
    on_trace_ready=tensorboard_trace_handler("./prof_logs"),
    record_shapes=True,
    with_stack=True,
    profile_memory=True,
)
prof.start()
for step, batch in enumerate(loader):
    batch = {k: v.to("cuda", non_blocking=True) for k, v in batch.items()}
    loss = model(**batch).loss
    loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
    prof.step()                      # tell the profiler a step boundary passed
    if step >= 6:                    # wait+warmup+active+slack
        break
prof.stop()
```

Then open the trace. The fastest read is the text summary:

```python
print(prof.key_averages().table(
    sort_by="cuda_time_total", row_limit=15))
# Columns: Name | Self CPU | CPU total | Self CUDA | CUDA total | # of Calls
```

But the *real* value is the timeline. Point a Chrome tab at `chrome://tracing` (or use the TensorBoard profiler plugin / Perfetto) and load the trace JSON. You are looking for one thing: **gaps in the GPU stream.** If the GPU track is solid with kernels back-to-back, you are compute-bound and the dataloader is hidden — good, go optimize the kernels (section 8). If the GPU track has big empty stretches and the CPU track during those stretches shows `enumerate(DataLoader)#_MultiProcessingDataLoaderIter.__next__` (the loader waiting for a worker) or a long `aten::item` (a sync), the trace has handed you the bug with a label on it.

Here is what the three diagnoses look like in a trace, concretely:

- **Dataloader-bound:** the GPU stream has a 40 ms gap before each compute burst; the CPU stream during the gap is parked in `DataLoaderIter.__next__`. Fix: workers/prefetch/cheaper samples.
- **Sync-bound:** the GPU stream has a small gap right after the backward, and the CPU stream shows `cudaDeviceSynchronize` triggered by `aten::item`. Fix: remove the `.item()` from the hot path.
- **Launch-bound:** the GPU stream is full of *tiny* kernels with little gaps between them, but each kernel is a few microseconds and the CPU is busy launching them faster than they complete — you are spending more time launching than computing. Fix: fuse ops, `torch.compile`. This is the next section.

The profiler is also where you confirm a fix: capture a trace before, capture one after, and watch the gaps close. "Util went from 31 to 92 percent" is the headline; "the 40 ms `DataLoaderIter` gaps before each step are gone in the after-trace" is the proof.

A practical note on profiler overhead and scope. The profiler is *not* free — capturing every operator with stacks and shapes adds overhead, so you do not run it for the whole training; you use the `schedule` (as above) to record a handful of steps in steady state and then stop. The numbers inside a profiled window are slightly inflated by the profiling itself, so use the profiler to find *what* is slow (the relative breakdown, the gaps, the kernel names) and use the plain synchronized timer to measure *how much faster* your fix made it. They answer different questions: the profiler localizes, the timer quantifies. A common mistake is to compare absolute step times *inside* the profiler before and after a fix; that mixes the profiling overhead into your delta. Localize with the profiler, quantify with the timer.

For the kernel-level view — once you are device-bound and want to know which specific operation dominates — the `key_averages()` table sorted by `Self CUDA` time is the fastest read. It tells you, for instance, that 60 percent of GPU time is in `aten::mm`/`aten::addmm` (the matmuls, which is healthy — that is where the FLOPs should be) or that a surprising fraction is in `aten::native_layer_norm` and `aten::copy_` (memory-bound elementwise work that fusion should absorb). If the matmuls dominate and util is high, you are compute-bound and near the ceiling; if elementwise/normalization/copy ops dominate, you have fusion headroom and `torch.compile` will help. This single table — what fraction of GPU time is matmul versus everything else — is one of the most informative numbers you can read about a device-bound run, because it tells you immediately whether you are spending time on the math you wanted or on the plumbing around it.

## 8. Small-op overhead, kernel launch cost, and `torch.compile`

Suppose you have solved the input side — the dataloader is hidden, no sync points, transfers overlap — and the GPU stream is *full*, yet throughput is still mediocre and MFU (section 9) is low. Now the problem is on the device, and the most common device-side throughput bug after dtype is **kernel-launch overhead from many tiny operations**.

Every PyTorch operation in eager mode launches at least one CUDA kernel, and each launch costs a few microseconds of fixed overhead on the CPU side plus scheduling on the GPU. For a big matmul that runs for milliseconds, a few microseconds of launch is nothing. But a transformer block in eager mode is *dozens* of small ops — a LayerNorm is a mean, a subtract, a square, a mean, a sqrt, a divide, a multiply, an add, each its own kernel — and on small tensors each kernel runs for *less time than it took to launch*. You become **launch-bound**: the GPU finishes each tiny kernel and waits for the CPU to launch the next one. The GPU stream looks busy (lots of kernels) but it is mostly gaps-between-tiny-kernels, and the math units are idle most of the time.

The structural fix is **operator fusion** — combine many small ops into one kernel so you launch once and the GPU does all the elementwise work in a single pass, also saving memory traffic (it reads the input once instead of once per op). You can fuse by hand with custom kernels, but the practical answer in modern PyTorch is `torch.compile`:

```python
# Eager: dozens of tiny kernels per block, launch-bound on small models.
model = MyTransformer().cuda()

# Compiled: TorchInductor fuses elementwise chains into few kernels,
# generates Triton kernels, and cuts launch overhead dramatically.
model = torch.compile(model, mode="max-autotune")

# First few steps are SLOW (compilation/autotuning); measure AFTER warmup.
for step, batch in enumerate(loader):
    ...                              # same loop; the speedup is automatic
```

Two warnings that save you confusion. First, the **first call is slow** — `torch.compile` traces and compiles on the first invocation (and `max-autotune` benchmarks kernel variants), which can take tens of seconds to a few minutes. Always discard the warmup steps before measuring, or you will "measure" a 30-second step and conclude compile made things worse. Second, **recompilation on changing shapes**: if your batch or sequence length varies every step, the compiler may recompile for each new shape, and you spend all your time compiling. The fix is to pad to fixed shapes (bucket sequence lengths) or use `dynamic=True` to compile a shape-generic kernel. On a small model that was launch-bound, I have seen `torch.compile` give 1.4–2.0× end-to-end; on a large model that was already compute-bound, the gain is smaller (10–30 percent) because there was less launch overhead to remove. The win scales with how launch-bound you were.

#### Worked example: a small model that compile saved

A team trains a 30M-parameter sequence model (an encoder for a retrieval task) on a single A100 and is frustrated that the "small, cheap" model trains slower per-token than a 10× larger one they ran last month. The meter is confusing: util is 96 percent, so it does not look starved. They compute MFU: 9 percent. High util, low MFU — that is the inefficiency signature, and for a *small* model the usual culprit is launch overhead, not dtype. The profiler confirms it: the GPU stream is a dense field of 3–8 microsecond kernels (the LayerNorms, GELUs, residual adds, attention sub-ops of a small model), and the CPU is launching them flat out — the GPU finishes each tiny kernel before the CPU can launch the next, so the math units are idle between launches even though "a kernel is running" almost always (hence 96 percent util). This is exactly the case `torch.compile` is built for. They wrap the model in `torch.compile(model, mode="max-autotune")`, discard the first 40 steps (compilation), and re-measure: tokens/sec 1.9× higher, MFU 9 → 19 percent, util still 96 percent but now the kernels are fused and each does real work. The lesson: **small models are launch-bound, not compute-bound, and util cannot see it — only MFU can.** Bigger models amortize launch over more compute, which is why the larger model had felt more efficient.

There is a lower-level tool for the *purely* launch-bound case where even fused kernels are being launched one at a time from Python: **CUDA graphs**. A CUDA graph records a sequence of kernel launches once and then replays the entire sequence with a single CPU-side call, eliminating per-kernel launch overhead almost entirely. `torch.compile` with `mode="reduce-overhead"` uses CUDA graphs internally, which is the easiest way to get them:

```python
# reduce-overhead captures a CUDA graph to kill per-launch CPU overhead;
# best when the model is small and you were launch-bound, not compute-bound.
model = torch.compile(model, mode="reduce-overhead")
```

The caveat with CUDA graphs is that they require **static shapes and static memory addresses** — the captured launches assume the tensors live at the same addresses every replay, so dynamic shapes, data-dependent control flow, and anything that allocates fresh tensors of varying size will break the capture or force re-capture. For a fixed-shape training step this is fine and the overhead reduction is real; for a model with heavy Python-side branching it is more trouble than it is worth. The decision rule: reach for `reduce-overhead`/CUDA graphs only after the profiler confirms you are CPU-launch-bound (the GPU stream is a dense field of microsecond kernels with the CPU launching at its limit), and only if your shapes are static. If you are compute-bound, CUDA graphs do nothing — there was no launch overhead to remove.

One more often-missed knob in the same family: **fuse the optimizer**. PyTorch optimizers default to a Python loop over parameter groups, launching a few kernels per parameter tensor — for a model with hundreds of weight tensors, that is hundreds of tiny launches per optimizer step. Passing `fused=True` (or `foreach=True`, the default on CUDA in recent versions) collapses the update into a handful of fused multi-tensor kernels:

```python
# Multi-tensor / fused optimizer: one kernel updates many params at once.
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, fused=True)
```

On a model with many small parameter tensors the optimizer step can be a surprisingly large slice of step time in eager mode, and `fused=True` can shrink it several-fold. It is a free win for any launch-bound run and costs nothing for a compute-bound one.

## 9. MFU: the honest efficiency metric

GPU utilization, as we said in section 1, only tells you whether *a* kernel is running. To know whether the GPU is doing *useful work efficiently*, you need **Model FLOPs Utilization (MFU)**: the ratio of the floating-point operations your model actually needed to the peak floating-point operations the hardware could have done in the same wall-clock time.

$$
\text{MFU} = \frac{\text{model FLOPs per second achieved}}{\text{hardware peak FLOPs per second}}
= \frac{C_{\text{model}} \cdot (\text{tokens/sec})}{F_{\text{peak}}}.
$$

The numerator needs $C_{\text{model}}$, the FLOPs per token of forward+backward. For a dense transformer there is a clean, well-known approximation from the scaling-laws literature: **forward+backward costs about $6N$ FLOPs per token**, where $N$ is the number of non-embedding parameters. The "6" decomposes as 2 FLOPs per parameter for the forward (a multiply and an add per weight) and roughly 4 for the backward (the backward does about twice the work of the forward — one pass for the input gradient, one for the weight gradient). So a 7B-parameter model processes each token at roughly $6 \times 7\times10^9 = 4.2\times10^{10}$ FLOPs. (If you want to include attention's quadratic term precisely, add $6 \cdot L \cdot s \cdot d$ terms, but for $s \ll$ model width the $6N$ rule is within a few percent and is what most reported MFU numbers use.)

The denominator $F_{\text{peak}}$ is the spec-sheet peak for your dtype: an A100 does about 312 TFLOP/s in bf16/fp16 with tensor cores (and only ~19.5 TFLOP/s in fp32 *without* tensor cores — a 16× difference, which is why dtype matters so much, section 10). An H100 (SXM) does roughly 990 TFLOP/s bf16 dense. Use the *dense* tensor-core number for your dtype, not the sparsity-inflated marketing number.

Here is a calculator you can drop into any training loop:

```python
def mfu(num_params_non_embed, tokens_per_sec, peak_flops_per_sec):
    """Model FLOPs Utilization for a dense transformer (6N rule)."""
    model_flops_per_token = 6 * num_params_non_embed
    achieved = model_flops_per_token * tokens_per_sec
    return achieved / peak_flops_per_sec

# Example: 7B model, 4.2e10 FLOPs/token (6N), measured 3,000 tokens/sec, A100 bf16.
N = 7e9
tps = 3000
A100_bf16 = 312e12
print(f"MFU = {mfu(N, tps, A100_bf16):.1%}")   # -> 40.4%
```

To turn your batch into tokens/sec: `tokens_per_sec = batch_size * seq_len / step_time_seconds`. Measure `step_time_seconds` with the synchronized timer from section 1 (over the steady state, not warmup), and you have a number you can track across every change you make.

**What is good?** For dense transformer pretraining/finetuning on modern GPUs, **40–55 percent MFU is solid**, and well-tuned large runs reach the mid-50s; the original PaLM report famously hit ~46 percent and described pushing toward the high 50s as a significant systems achievement. Below ~25 percent, something is wrong (you are starved, launch-bound, or not using tensor cores), and above ~60 percent on a real model with real data loading is excellent and rare. The number is not the goal — it is the instrument. If you make a change and MFU goes up, the GPU is doing more of your useful work per second; if util is high but MFU is low, you are running *something* but not your model's math efficiently, which points straight at dtype, shapes, and fusion.

One precision point that trips people up when they compare their MFU to a published one: **MFU is not HFU (Hardware FLOPs Utilization).** HFU counts *every* FLOP the hardware executes, including the recomputed forward passes that gradient checkpointing (activation recomputation) introduces — checkpointing throws away activations in the forward and recomputes them in the backward to save memory, which adds roughly one extra forward's worth of FLOPs. Those recomputed FLOPs are real work the hardware does, so they raise HFU, but they are not work the *model* needed — they are a memory-saving overhead. MFU counts only the model's necessary FLOPs ($6N$ per token), so a checkpointed run will show HFU noticeably higher than MFU. When you read "we achieved X percent utilization," ask which one it is: HFU flatters by 10+ points when checkpointing is on, and MFU is the honest cross-system comparison. The practical implication: if you turn on gradient checkpointing to fit a bigger batch, your *MFU per token* will look slightly lower (you added FLOPs the model did not need) even though total throughput may rise because the bigger batch hides overhead better — which is why you track tokens/sec *and* MFU, not either alone.

A second subtlety: the $6N$ rule undercounts for short-context models where attention's quadratic cost is non-trivial and overcounts nothing — it is a *lower bound* on the model's true FLOPs, so your real MFU is, if anything, slightly higher than the $6N$ estimate gives. For most finetuning at moderate sequence lengths the error is a few percent and not worth the extra terms. The point of MFU is not four-decimal precision; it is a *consistent* instrument you compute the same way before and after every change so the delta is meaningful. A number that is off by a constant factor but computed identically each time still tells you exactly whether your last change helped.

#### Worked example: reading util and MFU together to localize the bug

Two runs of the same 1.3B model on one A100, both reporting 88 percent `nvidia-smi` utilization, but run B is 35 percent slower in tokens/sec. Util alone says "both fine." Compute MFU. Run A: 12,000 tokens/sec, MFU $= 6 \times 1.3\text{e}9 \times 12000 / 312\text{e}12 = 30$ percent. Run B: 7,800 tokens/sec, MFU $= 20$ percent. Both have a full GPU stream (util 88 percent) but B does far less *useful* math per second — so B is running inefficient kernels, not starving. The profiler confirms: run B was accidentally in fp32 (no tensor cores) for the matmuls because someone disabled autocast for a "numerical stability" experiment and forgot to re-enable it. Util could not see it; MFU did. Fix: re-enable bf16 autocast, MFU 20 → 44 percent, tokens/sec 7,800 → 17,100. The lesson: **util detects starvation, MFU detects inefficiency, and you need both** — high util with low MFU is the signature of a dtype/shape/fusion bug, which is exactly section 10.

## 10. Mixed precision, tensor cores, and the multiple-of-8 rule

Tensor cores are the matrix-multiply units that give modern GPUs the bulk of their FLOPs, and they only engage under specific conditions. Miss the conditions and you fall back to the general-purpose cores at a fraction of the throughput — the 16× fp32-vs-bf16 gap from section 9. Figure 8 contrasts the two states.

![A before-and-after comparison where the before column shows fp32 dtype with a hidden dimension of 4095 yielding eighteen percent MFU, and the after column shows bf16 autocast with a hidden dimension of 4096 that is a multiple of eight yielding forty-eight percent MFU.](/imgs/blogs/the-gpu-is-idle-throughput-debugging-8.png)

The two conditions, both of which people miss:

**1. Use a tensor-core dtype.** fp32 matmuls do not use tensor cores at full rate; bf16 and fp16 do. Wrap your forward in autocast so the matmuls run in low precision while accumulations stay fp32:

```python
scaler = torch.amp.GradScaler("cuda")     # fp16 needs loss scaling; bf16 does not
for batch in loader:
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        loss = model(**batch).loss
    scaler.scale(loss).backward()          # (for bf16 you can skip the scaler)
    scaler.step(optimizer); scaler.update()
    optimizer.zero_grad(set_to_none=True)
```

A cheaper partial win, if you cannot move to bf16/fp16, is to enable TF32 for fp32 matmuls, which lets the tensor cores handle fp32 inputs with reduced-precision mantissas: `torch.set_float32_matmul_precision("high")` (or `torch.backends.cuda.matmul.allow_tf32 = True`). On Ampere+ this alone can multiply fp32 matmul throughput several-fold for a tiny, usually-irrelevant precision cost. The precision/throughput trade-off here is exactly the kind of thing the [roofline model](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives) makes concrete — moving to a lower-precision dtype both raises peak FLOPs and shrinks bytes moved, shifting you rightward toward the compute-bound regime.

**2. Make the shapes tensor-core-friendly.** Tensor cores process matrices in tiles, and they are most efficient when the matrix dimensions are **multiples of 8** (for fp16/bf16) — some kernels prefer 16 or 64. If your hidden size is 4095, or your vocab is 50001, or your batch is 30, the GPU pads to the next tile and wastes the padded lanes, or falls to a slower kernel. This is why model widths are 4096 not 4095, why vocab sizes get rounded up to multiples of 64 (a padded-vocab trick from the megatron/scaling literature), and why you should make batch and sequence dimensions multiples of 8. The fix is trivial and the win is real: rounding a hidden dimension from 4095 to 4096, or padding a vocab from 50,257 to 50,304, can move MFU several points with zero accuracy cost. You can confirm tensor cores are actually firing with the profiler — the kernel names will contain markers like `s16816gemm` or `tensor` for tensor-core GEMMs, versus plain `gemm`/`sgemm` for the fallback.

## 11. Eval and checkpoint stalls hiding in the training loop

The last family of throughput bugs is not in the step at all — it is the *stuff between steps* that you forgot counts toward wall-clock. Two big ones:

**Evaluation in the hot loop.** If you run a full validation pass every 500 steps and it takes 90 seconds, and your steps are 40 ms, then every 500 steps you spend 20 seconds training and 90 seconds evaluating — you are spending **69 percent of wall-clock on eval**. The GPU is busy during eval (so util looks fine) but it is not making training progress. Three fixes: evaluate less often, evaluate on a *subset* (a fixed 2,000-example slice tracks the trend without a full pass), and make eval itself efficient (no-grad, bigger batch, no per-example Python). The diagnostic is to log per-phase time separately — train-step time vs eval time vs checkpoint time — so the eval cost is visible instead of smeared into "training is slow."

**Synchronous checkpointing.** Saving a 13 GB optimizer+model checkpoint to a network filesystem can block the training loop for 30–120 seconds while it serializes and writes. If you checkpoint every 1,000 steps and each save blocks for 60 seconds, that is another silent tax. The fix is asynchronous checkpointing — copy the state to CPU (or a staging buffer) quickly, then write to disk in a background thread while training continues — which PyTorch's `torch.distributed.checkpoint` supports via async save, or you can roll a simple version that snapshots to pinned CPU memory and writes off-thread. The diagnostic, again, is to *time the checkpoint phase separately*; if "every 1000th step takes 60 seconds," you have found it.

The general principle for this section: **measure wall-clock end to end, not just the train step.** A run can have a perfectly tuned 40 ms step and still waste half its wall-clock on eval and checkpoint stalls that never show up if you only profile the inner loop. The matrix in figure 3 puts these symptoms next to the dataloader and sync symptoms so you can route any of them to its test and fix.

![A matrix mapping four throughput symptoms — sawtooth low utilization, high utilization but slow steps, periodic step-time spikes, and eight GPUs running at single-GPU speed — to their most likely bottleneck, the cheapest confirming test, and the fix direction.](/imgs/blogs/the-gpu-is-idle-throughput-debugging-3.png)

## 12. Bisecting a real starved run, end to end

Let me put it all together on a realistic run, the way you would actually do it, narrating the bisection. The setup: a ViT-B/16 image classifier finetune on a single A100, batch 256, reported by the team as "way slower than it should be." `nvidia-smi` shows 31 percent utilization, sawtoothing. Figure 6 is the decision tree we are about to walk.

![A decision tree that starts from low utilization and slow steps, branches into an input side where the loader alone is slow and a device side where the loader alone is fast, then splits the input side into dataloader-bound and host-to-device-bound and the device side into launch-bound and compute-bound.](/imgs/blogs/the-gpu-is-idle-throughput-debugging-6.png)

**Step 1 — read the meter.** Util 31 percent, sawtooth, power 210 W. Symptom: starvation. The card is waiting.

**Step 2 — is it the input side or the device side?** Run the dataloader-alone harness from section 3. It reports 140 ms/batch with `num_workers=4`. The GPU step (timed with synchronize) is 55 ms. Since 140 ≫ 55, this is **input-bound**. We have ruled out the entire device side (kernels, dtype, fusion) in one thirty-second test. We are on the left branch of figure 6.

**Step 3 — within the input side, dataloader or transfer?** Check the trace: the GPU gaps line up with `DataLoaderIter.__next__`, not with `Memcpy HtoD`. So it is the **dataloader**, not the H2D copy. Now bisect within the loader with the worker sweep: `num_workers` 0/2/4/8/16 gives 560/280/140/95/92 ms/batch. It plateaus around 8 workers at ~95 ms — still above the 55 ms GPU step, and adding workers past 8 stops helping, which means the *per-sample work* is now the limit, not the worker count.

**Step 4 — what is the per-sample work?** Time `__getitem__` with the wrapper from section 3: 1.9 ms/sample, of which 1.5 ms is the JPEG decode (PIL) and 0.3 ms is the resize+normalize. The decode dominates. So the real fix is a faster decode path *and* enough workers and a precompute where possible.

**Step 5 — fix and confirm.** Three changes: (a) switch the JPEG decode to a faster backend (`torchvision.io.decode_jpeg` on GPU, or `Pillow-SIMD`, or pre-resize the dataset to the training resolution once so each epoch decodes smaller files); (b) set `num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4`; (c) remove a stray `loss.item()` that was logging every step. Re-measure. Dataloader-alone: 48 ms/batch (now *below* the 55 ms GPU step — input side solved). Full run: util 31 → 92 percent, step time 178 → 60 ms, tokens/sec roughly tripled, wall-clock for the finetune 14 hours → 4.8 hours. MFU went from 17 percent (starved) to 44 percent (compute-bound, healthy). The before→after table:

| Instrument | Before (starved) | After (saturated) | What it confirms |
|---|---|---|---|
| `nvidia-smi` util | 31% sawtooth | 92% steady | starvation gone |
| Step time (synced) | 178 ms | 60 ms | ~3× faster |
| Dataloader alone | 140 ms/batch | 48 ms/batch | input hidden under 55 ms GPU |
| MFU (6N rule) | 17% | 44% | GPU now doing useful math |
| Wall-clock / finetune | 14.0 h | 4.8 h | the bill you actually pay |
| GPU-hours cost @ \$8 | \$112 | \$38 | \$74 saved per run |

Figure 5 is that result as a before-after: the starved run on the left, the saturated run on the right, with the three numbers that matter.

![A before-and-after comparison showing a starved run at thirty-one percent utilization with eighty-two milliseconds per step and seventeen percent MFU on the left, and a saturated run at ninety-two percent utilization with thirty-nine milliseconds per step and forty-six percent MFU on the right.](/imgs/blogs/the-gpu-is-idle-throughput-debugging-5.png)

The discipline that made this fast was not knowing the fix in advance — it was the *bisection*. Each test cut the search space in half: input vs device, dataloader vs transfer, worker-count vs per-sample-cost, decode vs resize. Four tests, each one cheap, and we never touched the model. That is the throughput debugging method.

## 13. Memory-bound vs compute-bound: the roofline, briefly

When you have saturated util and decent MFU but want to understand the *ceiling*, you need one more concept: **arithmetic intensity**, the ratio of FLOPs to bytes moved for an operation. Figure 7 lays it out.

![A two-row grid where the top row shows an elementwise add at one FLOP per byte, a LayerNorm at four FLOPs per byte, and a big matmul at eighty FLOPs per byte, and the bottom row labels them memory bound, near the ridge, and compute bound respectively.](/imgs/blogs/the-gpu-is-idle-throughput-debugging-7.png)

A GPU has two limits: how fast it can do math (peak FLOP/s) and how fast it can move data to and from memory (peak bytes/s). An operation is **compute-bound** if it does enough math per byte to keep the math units fed, and **memory-bound** if it is starved for data — the math units idle while waiting for memory. The crossover is the *ridge point*, the arithmetic intensity where the two limits cross, roughly `peak_FLOPs / peak_bandwidth`. For an A100 that is about $312\text{e}12 / 2\text{e}12 \approx 156$ FLOPs/byte.

The consequence for throughput debugging: **elementwise ops are memory-bound** (a vector add does 1 FLOP per ~12 bytes moved — far below the ridge — so it runs at memory speed, not compute speed), while **large matmuls are compute-bound** (high reuse, lots of FLOPs per byte). This is *why* fusion helps so much: fusing a chain of elementwise ops means you read the data from memory once and do all the math, instead of reading and writing it once per op — you raise the arithmetic intensity and move the op rightward toward compute-bound. And it is why your MFU has a real ceiling below 100 percent: a transformer is matmuls (compute-bound, efficient) *plus* LayerNorms, activations, dropout, and residual adds (memory-bound, capped by bandwidth), and the memory-bound fraction drags the whole-model MFU below the matmul's peak. A deeper treatment lives in the [roofline model post](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives); for our purposes the takeaway is: if you are launch-bound or memory-bound, fuse (`torch.compile`); if you are bandwidth-bound on transfers, move fewer bytes (uint8, lower precision); and accept that the memory-bound ops put a real ceiling on MFU.

## 14. Case studies and known signatures

A few real, well-documented patterns that match what you will see in the wild. I give numbers where they are reliable and flag where they are order-of-magnitude.

**The PaLM MFU benchmark.** Google's PaLM paper (Chowdhery et al., 2022) reported around 46 percent model FLOPs utilization for training a 540B-parameter model across thousands of TPU v4 chips, and framed pushing efficiency higher as a serious systems achievement. They also introduced the cleaner MFU metric precisely because hardware-FLOPs-utilization (which counts rematerialized/recomputation FLOPs from gradient checkpointing) flatters the number; MFU counts only the FLOPs the model *needs*. The practical takeaway: when a vendor or paper quotes a utilization number, ask whether it is HFU or MFU — they can differ by 10+ points, and MFU is the honest one for comparing systems.

**The `num_workers=0` default.** PyTorch's `DataLoader` defaults to `num_workers=0`, meaning the *main process* does all data loading inline, serially, with zero overlap with GPU compute. Countless first training runs are slow for exactly this reason and exactly this fix. It is the single most common throughput bug in tutorials-to-production transitions, and the fix is one keyword argument. The corollary signature: if a run is slow and `num_workers` is 0 or 1, you have probably found it before you even profile.

**The pin-memory-without-non-blocking half-fix.** A frequent partial optimization: people set `pin_memory=True` but keep `.to("cuda")` (blocking), or set `non_blocking=True` on pageable memory. Either way the copy does not overlap, util stays a few points lower than it should, and the trace shows the H2D copy on the critical path instead of hidden. The fix is the *pair*, as in section 6. The signature in a trace: a `Memcpy HtoD` block that sits in a GPU-stream gap rather than overlapping a compute kernel.

**`torch.compile` recompilation thrash.** A real and common surprise: a run with variable sequence lengths under `torch.compile` that spends most of its time *recompiling* because every new shape triggers a fresh compile. The signature is step times that are wildly variable and a profiler/log showing repeated TorchInductor compilation, and the fix is fixed/bucketed shapes or `dynamic=True`. People frequently conclude "compile made it slower" when in fact compile was thrashing on shapes; bucket the lengths and the speedup appears.

**The "8 GPUs, same speed as 1" trap.** When you scale to DDP and throughput does not improve, the usual cause is that the *per-GPU* data pipeline is the bottleneck (each rank still feeds itself from a too-slow loader), or a per-step sync (`.item()`, an unguarded print on rank 0) serializes the ranks across the all-reduce barrier, or the gradient all-reduce is bandwidth-bound on a slow interconnect. The diagnostic is the same: profile one rank, time its dataloader alone, grep its loop for syncs. The multi-GPU specifics — overlapping all-reduce with backward, `find_unused_parameters`, gradient bucketing — are their own topic in the DDP and multi-GPU debugging post; the *throughput* lens is identical to single-GPU, applied per rank.

## 15. When this is (and isn't) your bug

Throughput debugging has clean boundaries, and knowing them keeps you from optimizing the wrong thing.

**It IS a throughput bug when:** `nvidia-smi` util is low and sawtoothing; step time has a long right tail; the dataloader-alone test is slower than the GPU step; the profiler shows GPU-stream gaps; util is high but MFU is low (inefficient kernels); or scaling to more GPUs does not speed things up. These are systems problems — fix the pipeline, not the model.

**It is NOT your bug — look elsewhere — when:** the *loss* is wrong (diverging, NaN, plateaued at chance). Throughput is about *speed*, not *correctness*; a perfectly utilized GPU can train a broken model very efficiently. If your loss is bad, this whole post is a distraction — go to the [taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) and bisect the *correctness* failure first. Make it *correct*, then make it *fast*, never the other way around — there is no point speeding up a run that learns nothing.

**It is a different systems bug when:** memory, not speed, is the wall. If you are getting CUDA out-of-memory rather than low utilization, that is a separate investigation (where the memory goes, activation memory math, gradient checkpointing) covered in the out-of-memory debugging post — and note the *trade-off*: gradient checkpointing buys memory by recomputing activations, which *costs* throughput (it adds forward FLOPs), so the two are linked. Sometimes the right move is to spend a little MFU to fit a bigger batch that raises overall throughput; that is a roofline decision, not a free lunch.

**A subtle one:** if util is genuinely high (90 percent+) and MFU is genuinely good (45 percent+) and the run is *still* "too slow" for your deadline, you do not have a bug — you have a *capacity* problem, and the answer is more/bigger GPUs, a bigger batch, or a smaller model, not more profiling. Knowing when to stop optimizing and start provisioning is part of the skill. A 50 percent MFU run is near the practical ceiling; chasing it to 55 percent is rarely worth the engineering when a second GPU doubles throughput linearly.

## 16. The throughput checklist

When a run is slow, walk this in order; each step is cheap and rules out a region.

1. **`nvidia-smi -l 1`** — is util low and sawtoothing? If yes, suspect starvation. If util is high, suspect inefficiency (skip to MFU).
2. **Time the dataloader alone** — is it slower than the GPU step? If yes, you are input-bound; bisect workers vs per-sample cost vs I/O. Done in thirty seconds.
3. **Set `num_workers` to a real number** (8–16), `pin_memory=True`, `persistent_workers=True`, `prefetch_factor=4`. Re-measure.
4. **Grep the loop for sync points** (`.item()`, `.cpu()`, `print`, conditionals on GPU values); use `set_sync_debug_mode("warn")` to find the ones you missed. Move logging behind a cadence.
5. **Use `non_blocking=True` with pinned memory** for transfers; transfer `uint8` and convert on-device for big images.
6. **Profile** with `torch.profiler`; read the GPU-stream gaps and what the CPU is doing during them.
7. **Compute MFU** with the `6N` rule; if util is high but MFU is low, you are inefficient — check dtype and shapes.
8. **Enable bf16/fp16 autocast** (or at least TF32) and make dimensions multiples of 8 so tensor cores fire.
9. **`torch.compile`** if you are launch-bound (many tiny ops); discard warmup; bucket shapes to avoid recompiles.
10. **Time eval and checkpoint phases separately**; move eval off the hot cadence and checkpoint asynchronously.

## Key takeaways

- **A slow run is a debugging problem, not a fact of life.** The GPU costs the same idle or saturated, so 31 percent util is wasted money — treat it like a NaN: symptom, mechanism, diagnostic, fix, confirmation.
- **The train step is a pipeline; throughput is set by the slowest stage.** $t_{\text{step}} = \max(t_{\text{load}}, t_{\text{gpu}})$ when prefetch works. The dataloader is *free* until it is slower than the GPU step, then it is pure idle.
- **Time the dataloader alone first.** Iterate the loader with no model; if it cannot beat the GPU step time, the bug is on the input side and no kernel optimization will help.
- **CUDA is asynchronous, so synchronize before you time** — and the same sync that fixes your timer is the `.item()`/`print`/`.cpu()` that, in the hot loop, serializes your training. Delete sync points from the hot path.
- **`pin_memory` and `non_blocking` are a pair;** one without the other is half a fix and the copy will not overlap.
- **`nvidia-smi` util detects starvation; MFU detects inefficiency.** High util with low MFU means a dtype/shape/fusion bug. Compute MFU with the `6N`-FLOPs-per-token rule; 40–55 percent is solid for transformers.
- **Tensor cores need a low-precision dtype and multiple-of-8 shapes.** bf16 autocast plus aligned dimensions can multiply matmul throughput; fp32 leaves most of the GPU's FLOPs unused.
- **`torch.compile` fixes launch-bound runs** (many tiny eager ops) by fusing kernels — but discard the compile-warmup steps and bucket shapes or it thrashes on recompilation.
- **Profile the whole wall-clock, not just the step.** Eval-in-the-loop and synchronous checkpointing can eat half your run while the per-step time looks perfect; time those phases separately.
- **Fix correctness before speed.** A fully utilized GPU happily trains a broken model. Make it learn, then make it fast.

## Further reading

- **"PaLM: Scaling Language Modeling with Pathways"** — Chowdhery et al., 2022. Introduces and reports Model FLOPs Utilization; the canonical reference for MFU and the ~46 percent figure for a very large run.
- **"Scaling Laws for Neural Language Models"** — Kaplan et al., 2020. Source of the $6N$ FLOPs-per-token approximation (2 forward, 4 backward) used in every MFU calculation.
- **PyTorch Profiler and `torch.profiler` documentation** — the recipe for capturing CPU+CUDA traces, the TensorBoard/Perfetto trace viewer, and reading GPU-stream gaps. The primary diagnostic instrument for this post.
- **PyTorch `DataLoader` documentation** — `num_workers`, `pin_memory`, `prefetch_factor`, `persistent_workers`, and the warnings about worker semantics; the knobs section 3 sweeps.
- **`torch.compile` / TorchInductor documentation** — fusion, `mode="max-autotune"`, dynamic shapes, and the recompilation caveats from section 8.
- **NVIDIA Deep Learning Performance / Tensor Core guides** — the multiple-of-8 requirement, dtype throughput tables, and how to confirm tensor-core kernels in a profile.
- **The taxonomy: [A Taxonomy of Training and Finetuning Bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs)** — the master symptom-to-suspect-to-test decision tree this throughput investigation plugs into (throughput lives in the "systems" branch).
- **The capstone: [The Training Debugging Playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook)** — the full bisection method and printable checklist, of which this throughput checklist is one page.
- **Sibling posts: [The Input Pipeline Is Lying to You](/blog/machine-learning/debugging-training/the-input-pipeline-is-lying-to-you)** (dataloader *correctness* bugs, the companion to this post's *speed* bugs), **[Instrumenting a Training Run](/blog/machine-learning/debugging-training/instrumenting-a-training-run-what-to-log)** (logging throughput without paying a sync), and the cross-cutting **[Roofline Model](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives)** for arithmetic intensity and the memory-bound vs compute-bound regimes.
