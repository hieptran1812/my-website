---
title: "The Roofline for Your Service: Are You Compute-Bound or Memory-Bound?"
date: "2026-07-14"
publishDate: "2026-07-14"
description: "How to take one live kernel out of a profile, place it on the roofline, and read off the single fix that will actually make it faster — compute, bandwidth, fusion, or none of the above."
tags:
  [
    "performance-engineering",
    "gpu-optimization",
    "roofline",
    "arithmetic-intensity",
    "profiling",
    "pytorch",
    "nsight",
    "memory",
    "inference",
    "ml-systems",
  ]
category: "machine-learning"
subcategory: "Performance Engineering"
author: "Hiep Tran"
featured: true
readTime: 35
---

Here is a bill I have watched a team pay twice. Their inference service ran on A100s, the p50 latency was fine, the p99 was not, and someone with budget authority decided the fix was newer silicon. They moved the fleet to H100s. On paper this is a 3.2× jump in bf16 math throughput — 312 TFLOP/s becomes 989 TFLOP/s — and a 1.7× jump in memory bandwidth, 2.0 TB/s becomes 3.35 TB/s. The service got about 5% faster. The invoice roughly doubled. Somebody asked me, not unreasonably, where the speed went.

The speed went nowhere, because it was never there to get. The dominant kernel in that service — a single-token decode step — was reading the entire weight matrix of the model from memory to produce one token of output, and doing almost no arithmetic with each byte it read. It was **memory-bound**: limited by how fast the chip can move bytes, not by how fast it can multiply. Tripling the multiply rate does exactly nothing for a kernel that is standing around waiting for bytes to arrive. The 1.7× bandwidth bump is the only lever that touched it, and 1.7× on the one part of the chip that mattered, diluted by everything else in the request path, is how you get 5%. They paid for compute they physically could not use.

There is one diagram that predicts this before you sign the purchase order, and one procedure that tells you, for any kernel in your service, whether you are about to make that mistake. The diagram is the **roofline**. The procedure is: pull the kernel's achieved arithmetic intensity out of a profiler, place the dot on the roofline for the exact GPU you run on, and read off which of a small number of fixes — more FLOP per second, more bandwidth, fusion, bigger batches, or "this isn't a roofline problem at all" — is the one that will move it. Figure 1 is that reading applied to the four kernels you will meet most often in a live AI service; by the end of this post you will be able to produce that table for your own service from a profile, not from a datasheet.

![a comparison grid of four common service kernels each with its arithmetic intensity its bound and the single fix that will move it on an A100](/imgs/blogs/the-roofline-for-your-service-1.webp)

This is the fourth post in the [Profiling & Optimizing AI Services](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu) series, and it is deliberately *not* a from-scratch derivation of the roofline model — that already exists, worked out line by line with the algebra and the log-log geometry, in the [HPC roofline post](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound), and I will lean on it rather than repeat it. This post is the applied half: you have a real service, you have a profiler, and you want to know what to *do*. The roofline stops being a chart you nod at and becomes a decision procedure you run on Tuesday afternoon when the GPU bill lands on your desk. We keep the series' running spine — a Transformer inference and training service on one GPU — and score everything the way the series always does: achieved TFLOP/s, achieved GB/s, percent of peak, and the fix that changed it.

## The roofline in one comparison: the min of two roofs

Strip the roofline to its load-bearing sentence and it is this. Any kernel has an attainable performance ceiling equal to the smaller of two limits:

$$P_\text{attainable} = \min\big(P_\text{peak},\; \text{AI} \times B\big)$$

The first term, $P_\text{peak}$, is the chip's peak math rate in FLOP per second — a flat ceiling that no kernel can punch through no matter how much data it has. The second term is the memory ceiling: the kernel's **arithmetic intensity** $\text{AI}$ — the floating-point operations it performs per byte it moves to and from HBM (the high-bandwidth memory next to the die) — multiplied by the chip's peak bandwidth $B$ in bytes per second. Arithmetic intensity is defined once and used forever:

$$\text{AI} = \frac{\text{FLOPs}}{\text{bytes moved}} \quad \left[\frac{\text{FLOP}}{\text{byte}}\right]$$

The whole model is that $\min$. If your intensity is low — few FLOPs per byte — then $\text{AI} \times B$ is the smaller number and it caps you; you are **memory-bound**, and more math throughput is wasted on you. If your intensity is high, the flat $P_\text{peak}$ caps you; you are **compute-bound**, and more bandwidth is wasted on you. The two ceilings cross at exactly one intensity, the **ridge point**, where neither limit dominates:

$$\text{AI}^* = \frac{P_\text{peak}}{B}$$

For an A100 80GB SXM — 312 TFLOP/s of bf16 Tensor Core math, 2.0 TB/s of HBM2e, both straight off the [NVIDIA datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet.pdf) — the ridge sits at $312 / 2.0 = 156$ FLOP per byte. That number is the line in the sand. A kernel that does at least 156 FLOPs for every byte it drags across HBM can, in principle, saturate the A100's math units. A kernel that does fewer is memory-bound and cannot, no matter how perfect the kernel, no matter how new the tensor cores. One hundred and fifty-six is a brutal bar, and almost nothing in a neural network clears it except a large, well-tiled matrix multiply. Figure 2 draws the decision as it actually runs: your profiled intensity feeds the slanted memory roof, the flat compute roof stands on its own, the $\min$ picks the lower of the two, and that pick forks straight into your diagnosis.

![a dataflow diagram in which arithmetic intensity feeds the slanted bandwidth roof while the flat compute roof stands separate and the two merge into a minimum that forks into memory-bound or compute-bound](/imgs/blogs/the-roofline-for-your-service-2.webp)

The reason this is worth internalizing as a *procedure* and not a fact is that the fix is completely determined by which side of that fork you land on, and the fork is cheap to evaluate: it is one division and one comparison. Everything expensive — buying hardware, rewriting kernels, restructuring the batch scheduler — should happen only after you have done the one division. The rest of this post is how to get the two numbers the division needs (your kernel's real FLOPs and real bytes) out of a live service, and what to do with the answer.

One clarification the series insists on, because it is the difference between a useful number and a lie. "GPU utilization" as reported by `nvidia-smi` is not where you are on the roofline. Utilization only says the GPU had *a* kernel resident during the sampling window; a batch-1 decode kernel that is 99% stalled waiting on HBM still shows 100% utilization because *something* was scheduled. The roofline is about **useful** throughput, and the metric that tracks it is achieved FLOP/s against the ceiling (or achieved GB/s against the bandwidth ceiling for memory-bound ops). The [metrics post](/blog/machine-learning/performance-engineering/metrics-that-actually-matter) takes that lie apart in full; here it is enough to say that the green bar is not your dot on the roofline, and if you optimize toward the green bar you will optimize toward nothing.

## Why almost every inference kernel is memory-bound

Start with the kernel that pays most of the bill in a generative service: the single-token decode step. This is where the H100 upgrade died, and the reason is pure roofline.

When an LLM generates text, it produces one token at a time, and to produce each token it must pass the current hidden state through every layer of the model. Passing one vector through a linear layer of weight matrix $W$ is a matrix-vector product: you read the entire weight matrix out of HBM, and you multiply it by a single vector. For a 7-billion-parameter model in bf16, the weights are about 14 GB. To emit *one* token, the decode step reads on the order of 14 GB from memory. And the arithmetic it does with those 14 GB is: two FLOPs — one multiply, one add — per weight. That is the definition of a matrix-vector product. Two FLOPs per weight, one weight per two bytes (bf16), so:

$$\text{AI}_\text{decode} \approx \frac{2 \cdot P}{2 \cdot P} = 1 \;\frac{\text{FLOP}}{\text{byte}}$$

where $P$ is the parameter count. The dimensions cancel; the model size does not matter. A single-token decode step has an arithmetic intensity of about **1 FLOP per byte**, which on the A100 sits 156× below the ridge. It is not slightly memory-bound. It is memory-bound by two orders of magnitude. Figure 3 accounts for where those bytes actually go in one decode step and why the intensity collapses to one.

![a vertical stack showing that weight reads dominate the byte budget of a decode token with kv cache and activations far smaller and a fusion and precision fix at the bottom](/imgs/blogs/the-roofline-for-your-service-3.webp)

The consequence is a hard ceiling you can compute without a profiler. If the decode step must move 14 GB per token and the A100 delivers 2.0 TB/s, then even a *perfect* kernel — one that hits 100% of bandwidth — takes at least $14 / 2000 = 7$ ms per token, for a ceiling of about 142 tokens per second at batch 1. That is the physics. No amount of tensor-core throughput changes it, because the tensor cores are not the constraint; they are idle 99% of the time, waiting for the next slab of weights to arrive. The attainable performance is $\text{AI} \times B = 1 \times 2.0 = 2.0$ TFLOP/s, which is 0.6% of the 312 TFLOP/s the datasheet advertises. A profiler will confirm the decode kernel runs at roughly 1% "MFU" (model FLOPs utilization, the fraction of peak math you actually use) — and that 1% is not a bug to fix, it is the roofline telling you the truth: for this kernel, at this batch size, math throughput is the wrong resource to measure.

This is the single most important reallocation of attention the roofline buys you. For a memory-bound kernel, **MFU is the wrong metric and achieved bandwidth is the right one.** The decode step at 1% MFU might be running at 95% of the 2.0 TB/s bandwidth ceiling, in which case it is *already optimal for what it is* and no kernel rewrite will help. Or it might be at 40% of bandwidth, in which case there is a real, fixable inefficiency — poor memory coalescing, a bad access pattern, launch overhead between too-small kernels — and you should go find it. The two situations look identical on `nvidia-smi` (both show 100% util) and identical on an MFU dashboard (both show ~1%). Only the bandwidth number distinguishes "physically maxed out" from "leaving half the memory system on the floor." We will pull that number out of `ncu` in a moment.

### Attention is memory-bound for the same reason, in a different disguise

Decode is the loudest memory-bound kernel, but attention is the one that launched a thousand papers, and it is memory-bound by the same mechanism: it moves a lot of bytes relative to the math it does. Naive attention computes $\text{softmax}(QK^\top/\sqrt{d})\,V$ by materializing the full $N \times N$ score matrix in HBM — writing it, reading it back for the softmax, writing the probabilities, reading them back for the second matmul. For a sequence of 8192 tokens that is a 67-million-entry matrix shoved to memory and dragged back four times, and it drags the effective intensity of the whole attention block down to roughly 63 FLOP/byte on the numbers worked in the [HPC roofline post](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) — still below the A100 ridge of 156, still memory-bound, despite the block containing two perfectly compute-bound matmuls.

FlashAttention is famous not because it is a clever math trick — it computes the exact same attention, the exact same FLOPs — but because it is a *roofline move*. It tiles $Q$, $K$, $V$ so the scores never leave on-chip SRAM, which deletes the four $N^2$ HBM passes and collapses the byte count. Same numerator, tiny denominator, intensity rockets past the ridge, and the kernel physically moves from the memory-bound slope to the compute-bound ceiling. That is why the win grows with sequence length — the deleted traffic scaled as $N^2$ — and it is the canonical example of the only two levers a memory-bound kernel has: move fewer bytes (fusion), or move them faster (bandwidth). The [kernel-fusion post](/blog/machine-learning/high-performance-computing/kernel-fusion-and-flashattention-beating-the-memory-wall) and this series' own [bandwidth-bound-and-fusion](/blog/machine-learning/performance-engineering/bandwidth-bound-and-fusion) post work that fix end to end; the point here is just that the roofline *predicted the existence of the fix* before anyone wrote the kernel, purely from the intensity being below the ridge.

The elementwise and normalization ops that pepper every model are the same story in even starker form. A LayerNorm does about 8 FLOPs per element over a read-and-write, an intensity near 2 to 4 FLOP/byte. A GELU, less. An elementwise residual add reads two arrays and writes one, three memory touches for a single FLOP, an intensity around 0.08. Every one of these sits hundreds of times below the ridge, structurally, on every GPU ever made, and there is no shape or size that lifts them off the memory slope because they are *defined* by touching each byte once and doing almost nothing with it. When your profiler shows a LayerNorm eating 8% of step time while contributing 0.1% of the FLOPs, that inversion is not a mystery — it is the roofline. The cheap-in-math ops are expensive-in-time precisely because time is set by bytes, and they are all bytes.

## Batching is a roofline move (this is *why* it works)

If a single decode token is pinned at AI ≈ 1, how does any serving system ever reach useful throughput? By batching — and the roofline explains exactly why batching is the master lever of inference, not as folklore but as arithmetic.

Take the same linear layer, weight $W$ of shape $K \times N$, but now push $M$ tokens through it at once instead of one. The weight is read from HBM exactly once — it is the same 14 GB — but now it feeds $M$ separate token-vectors, doing $2 \cdot M \cdot K \cdot N$ FLOPs instead of $2 \cdot K \cdot N$. The bytes are almost unchanged (one weight read); the FLOPs multiply by $M$. So:

$$\text{AI}_\text{batch} \approx \frac{2 M K N}{K N \cdot s} = \frac{2M}{s}$$

For bf16 ($s = 2$ bytes per element), that is simply $\text{AI} \approx M$. **Every token you add to the batch reuses the loaded weights one more time, and intensity climbs linearly with batch size.** Batching does not make the memory faster and it does not make the math faster; it changes the *ratio*, and the ratio is the whole game. Figure 4 places the two extremes on the same A100.

![a before and after comparison of a decode kernel at batch one reading all weights per token versus batch sixty four reusing each weight across many tokens and climbing toward the ridge](/imgs/blogs/the-roofline-for-your-service-4.webp)

At batch 1, AI ≈ 1, attainable ≈ 2 TFLOP/s, about 1% of peak — memory-bound, wasting the tensor cores. At batch 64, AI ≈ 64, attainable ≈ $64 \times 2.0 = 128$ TFLOP/s, about 41% of the A100's peak — the *same kernel*, the *same weights*, dragged rightward across the roofline into respectable territory purely by stacking requests. To fully clear the ridge you need $\text{AI} \geq 156$, i.e. $M \geq 156$ tokens flowing through the weight-bound GEMM before it tips onto the compute ceiling. That is the quantitative reason continuous batching, and the whole edifice of request-batching schedulers, dominates modern LLM serving: each request you stack onto a weight read pushes $M$ toward 156, and below 156 your tensor cores are structurally starved. The [continuous-batching-and-pagedattention post](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention) is the systems engineering of doing this safely under real traffic; the roofline is *why* it is worth the engineering.

There is a caveat worth stating because it bites in production: the KV cache. The clean $\text{AI} \approx M$ assumes weight traffic dominates, which is true when the context is short. As context grows, each decode step must also read the KV cache — the stored keys and values for every prior token — and that traffic grows with sequence length and does *not* amortize across the batch the way weights do (each sequence has its own KV). So at long context, the KV reads can become a second memory wall that batching does not knock down, and the effective intensity plateaus below what the weight-only formula predicts. The [KV-cache optimization post](/blog/machine-learning/model-serving/kv-cache-optimization) is the fix menu there. The roofline still applies verbatim; you just have to count the KV bytes in the denominator, which the profiler does for you automatically because it counts *all* HBM traffic, not the traffic you remembered to model.

## Placing your own kernel: from a profile to a dot

Everything so far was arithmetic you can do on a napkin. The reason you still need a profiler is that the napkin computes the *ideal* intensity — the byte count assuming every matrix is read exactly once — and real kernels miss the ideal. A poorly tiled GEMM re-reads its operands from HBM many times, moving far more bytes than the formula says, landing to the *left* of where you predicted, memory-bound when you expected compute-bound. The only way to know where your kernel *actually* sits is to measure the bytes it *actually* moved and the FLOPs it *actually* executed. That is what Nsight Compute (`ncu`) does: it reads hardware counters and computes the real, achieved intensity, then places your kernel on the real roofline. Here is the invocation.

```bash
# Profile the decode kernel with the roofline + memory + compute sections.
# --launch-skip 20 skips warm-up iterations (autotuning, cuDNN algo pick,
# allocator warm-up) so you profile steady state, not the first slow call.
# --launch-count 5 profiles five kernel launches and reports the median.
ncu --set roofline \
    --section SpeedOfLight_RooflineChart \
    --section MemoryWorkloadAnalysis \
    --launch-skip 20 --launch-count 5 \
    -k "regex:.*gemm.*|.*attention.*" \
    -o decode_roofline \
    python serve_one_request.py
```

The two counters you actually read out of the report — whether in the `ncu` UI's "GPU Speed Of Light" section or the terminal summary — are **Compute (SM) Throughput %** and **Memory Throughput %**. They are the whole diagnosis in two lines. Here is what a memory-bound decode GEMM looks like in the terminal output:

```console
  gemm_bf16_nt_decode (M=1, N=4096, K=4096)  Duration: 41.28 usecond
  ----------------------------------------------------------------------
  Section: GPU Speed Of Light Throughput
    DRAM Frequency            cycle/nsecond         1.51
    SM Frequency              cycle/nsecond         1.28
    Memory Throughput                    %         93.412   <-- the wall
    Compute (SM) Throughput              %          4.117   <-- idle math
    Achieved Occupancy                   %         71.9
  ----------------------------------------------------------------------
  Section: Memory Workload Analysis
    Memory Throughput               Gbyte/s      1868.2      <-- 93% of 2.0 TB/s
    DRAM Bandwidth Utilization           %         93.4
```

Read those two numbers — Memory 93%, Compute 4% — and you are done diagnosing. The kernel is riding the memory roof (93% of the 2.0 TB/s ceiling) and barely touching the compute roof (4%). That is a dot hard against the slanted line, far left of the ridge, unambiguously memory-bound, and *already near-optimal for a memory-bound op* because it is at 93% of the bandwidth it is allowed. No kernel rewrite will help this one; the only levers are the roofline's memory levers — cut the bytes (batch, fuse, lower precision) or buy more bandwidth. Contrast it with a healthy compute-bound GEMM, where the numbers invert: Compute Throughput 88%, Memory Throughput 30%, a dot up against the flat roof, and the lever becomes precision (fp8) or "you are near peak, stop."

The rule that turns those two counters into an action is one line, and it is worth memorizing:

- **Memory Throughput high, Compute low** → memory-bound. Fuse, batch, or lower precision to cut bytes. Bandwidth is your resource; measure GB/s.
- **Compute high, Memory low** → compute-bound. Better precision or tensor-core utilization. FLOP/s is your resource; measure MFU.
- **Both low** → you are not on either roof. This is not a roofline problem — it is occupancy, launch overhead, or a sync stall. Stop looking at the roofline and go read the [launch-overhead post](/blog/machine-learning/performance-engineering/the-kernel-launch-overhead-problem).

That third case is the one people miss, and it is the reason the fix-selection tree in Figure 5 has three branches, not two.

![a decision tree that reads the kernel placement relative to the ridge and both roofs and routes memory-bound compute-bound and host-bound kernels each to their matching fix](/imgs/blogs/the-roofline-for-your-service-5.webp)

The "both low" branch is where the roofline hands off to the rest of the profiling craft. If Compute is 6% *and* Memory is 8%, your kernel is not near either ceiling — it is not doing enough work to be limited by the hardware at all. Something else is throttling it: too few threads to fill the SMs (low occupancy), thousands of tiny kernels each paying microseconds of CPU-side launch latency (host-bound), or a `torch.cuda.synchronize()` stalling the pipeline every step. Those are real, common, and *not* fixable by anything the roofline recommends — buying bandwidth or FLOP/s for a host-bound service is the exact same mistake as the H100 upgrade, one level up. The [nsight-compute deep dive](/blog/machine-learning/performance-engineering/nsight-compute-kernel-deep-dive) drills into the occupancy and warp-stall side; the launch-overhead post handles the tiny-kernel case. The roofline's job is to tell you *whether* you are on a roof at all, and if both throughputs are low, the honest answer is no.

### The analytical companion: a roofline calculator

Before you profile, it pays to compute the *ceiling* analytically so you know what "good" looks like when the profiler answers. This small calculator takes an op's FLOP count, byte count, and a GPU spec and returns the intensity, the ridge, the verdict, and the attainable performance — the napkin math, made repeatable.

```python
from dataclasses import dataclass

@dataclass
class GPU:
    name: str
    peak_flops: float   # FLOP/s  (e.g. 312e12 for A100 bf16 tensor cores)
    bandwidth: float    # bytes/s (e.g. 2.0e12 for A100 HBM2e)

A100 = GPU("A100-80GB-SXM", peak_flops=312e12, bandwidth=2.0e12)
H100 = GPU("H100-SXM",      peak_flops=989e12, bandwidth=3.35e12)
L4   = GPU("L4",            peak_flops=242e12, bandwidth=0.30e12)

def roofline(flops, bytes_moved, gpu):
    ai         = flops / bytes_moved
    ridge      = gpu.peak_flops / gpu.bandwidth
    attainable = min(gpu.peak_flops, ai * gpu.bandwidth)
    return {
        "intensity_flop_per_byte": round(ai, 2),
        "ridge": round(ridge, 1),
        "attainable_TFLOPs": round(attainable / 1e12, 1),
        "pct_of_peak": round(100 * attainable / gpu.peak_flops, 2),
        "bound": "compute-bound" if ai >= ridge else "memory-bound",
    }

# One decode token through a 7B model, bf16: ~2*P FLOPs over ~2*P bytes.
P = 7e9
print("decode  ", roofline(2 * P, 2 * P, A100))
# -> {'intensity...': 1.0, 'ridge': 156.0, 'attainable_TFLOPs': 2.0,
#     'pct_of_peak': 0.64, 'bound': 'memory-bound'}

# The same layer batched to 64 tokens: FLOPs x64, bytes ~unchanged.
print("batch64 ", roofline(2 * P * 64, 2 * P, A100))
# -> {'intensity...': 64.0, ..., 'attainable_TFLOPs': 128.0,
#     'pct_of_peak': 41.03, 'bound': 'memory-bound'}
```

The batch-64 line is instructive precisely because it is *still labeled memory-bound* — AI 64 is below the ridge of 156 — yet it attains 41% of peak. Memory-bound is not a binary of "0% or 100%"; it is a slope, and 41% is where AI 64 lands on it. The calculator gives you the ceiling; the profiler tells you how close the real kernel got. Use them together: if the calculator says the ceiling is 128 TFLOP/s and `ncu` says you achieved 118, you are done — you are at 92% of the attainable and the only way up is to change the intensity (batch more) or the hardware. If the calculator says 128 and `ncu` says 40, you have a kernel bug to hunt, not a hardware limit to accept.

### Measuring achieved intensity yourself, without ncu

You do not always have `ncu` — it needs elevated permissions and a kernel you can isolate. You can get the *achieved* FLOP/s and GB/s of any op with `torch.cuda.Event` timing, the FLOP count you compute, and the byte count you compute. The one rule that matters: **CUDA is asynchronous, so you must synchronize and you must warm up**, or you will time launch overhead instead of the kernel.

```python
import torch

def timed(fn, *args, warmup=25, iters=100):
    for _ in range(warmup):           # trigger autotune, allocator, clocks
        fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()          # wait for the GPU before reading the clock
    return (start.elapsed_time(end) / iters) / 1e3   # seconds per call

dev = "cuda"
# A decode-shaped matmul: (1 x 4096) @ (4096 x 4096) -> a fat weight read,
# one token of work. This is the shape that pins AI near 1.
W = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
x = torch.randn(1,    4096, device=dev, dtype=torch.bfloat16)
t = timed(lambda: x @ W)
flops = 2 * 1 * 4096 * 4096
bytes_moved = (1*4096 + 4096*4096 + 1*4096) * 2   # x, W, out; W dominates
print(f"decode : {flops/t/1e12:7.3f} TFLOP/s  {bytes_moved/t/1e9:7.0f} GB/s  "
      f"AI={flops/bytes_moved:5.2f}")
# decode :   0.79 TFLOP/s    1642 GB/s  AI= 0.50   <- 82% of BW, 0.25% of FLOPs

# Now batch the same weight to 64 tokens: (64 x 4096) @ (4096 x 4096).
xb = torch.randn(64, 4096, device=dev, dtype=torch.bfloat16)
t = timed(lambda: xb @ W)
flops = 2 * 64 * 4096 * 4096
bytes_moved = (64*4096 + 4096*4096 + 64*4096) * 2
print(f"batch64: {flops/t/1e12:7.3f} TFLOP/s  {bytes_moved/t/1e9:7.0f} GB/s  "
      f"AI={flops/bytes_moved:5.2f}")
# batch64:  118.4  TFLOP/s     920 GB/s  AI=57.9    <- 38% of FLOPs, 46% of BW
```

Read those two lines against each other and the roofline is *in the numbers*. The decode row is at 82% of the 2.0 TB/s bandwidth ceiling and a rounding error of the FLOP ceiling — memory-bound, near-optimal for what it is. The batch-64 row flipped: it is now at 38% of the FLOP ceiling and only 46% of bandwidth — it walked toward the compute roof. Notice the achieved AI (0.50, then 57.9) differs a little from the napkin (1.0, 64) because the byte count now honestly includes the input and output activations, not just the weights. That honesty is the point: measure, do not assume. And notice the decode row moving *bytes* at 1642 GB/s — that is the number to optimize for a memory-bound op, and it is nowhere on an MFU dashboard.

## Walking the roof: worked examples with real numbers

The roofline earns its keep when you carry it into a specific decision. Here are the placements worked out, with the fix each one dictates, for the three kernels that show up in nearly every service.

#### Worked example: an LLM decode step on an A100 (memory-bound)

A 7B model, bf16, batch 1, on an A100 80GB. Weights are 14 GB; one token reads them and does 2 FLOPs each. FLOPs ≈ $1.4 \times 10^{10}$, bytes ≈ $1.4 \times 10^{10}$, so AI ≈ 1. The A100 ridge is 156, so this sits 156× to the left of it: deeply memory-bound. Attainable ceiling: $1 \times 2.0 = 2.0$ TFLOP/s, or 0.6% of the 312 peak. Time floor: $14\,\text{GB} / 2.0\,\text{TB/s} = 7$ ms per token, 142 tokens/s ceiling. `ncu` confirms Memory Throughput ~93%, Compute ~4%.

**The fix the placement dictates:** raise the intensity by batching (each of 64 requests reuses the weight read, AI → ~64, ceiling → 128 TFLOP/s), and/or move fewer bytes by quantizing weights to fp8 or int4 (halve or quarter the 14 GB, doubling or quadrupling AI). What the placement forbids: buying an H100 for its 3.2× math. The H100's *bandwidth* (3.35 vs 2.0 TB/s) would help this kernel by 1.7×; its extra FLOP/s would help by 0×. That is the arithmetic behind the 5% upgrade.

#### Worked example: a big training GEMM on an A100 (compute-bound)

A feed-forward GEMM in training, $M = N = K = 4096$, bf16. FLOPs $= 2 \cdot 4096^3 = 1.37 \times 10^{11}$; bytes $= 3 \cdot 4096^2 \cdot 2 = 1.01 \times 10^8$ (each matrix touched once, assuming good tiling). AI $= 1.37\times10^{11} / 1.01\times10^8 \approx 1365$ FLOP/byte, nearly 9× past the ridge of 156: solidly compute-bound. Attainable ceiling: the full 312 TFLOP/s. A well-tuned cuBLAS kernel of this shape hits ~285 TFLOP/s, about 91% of peak; `ncu` shows Compute Throughput ~88%, Memory ~30%.

**The fix the placement dictates:** faster math. Drop to fp8 on a chip that supports it (roughly doubling peak on Hopper), or confirm you are already near peak and spend your time elsewhere. What the placement forbids: fusion (there is no memory round-trip to eliminate — the op is not waiting on bytes) and more bandwidth (30% of the memory system is idle). Here the H100 upgrade *would* have paid off, because this kernel actually uses the FLOP/s you would be buying. The lesson is not "H100s are bad" — it is "match the purchase to the bound."

#### Worked example: a LayerNorm on an A100 (deeply memory-bound)

A LayerNorm over a $2048 \times 4096$ activation, bf16. It reads the input and writes the output — 2 × 2048 × 4096 × 2 ≈ $6.7 \times 10^7$ bytes — and does ~8 FLOPs per element, ≈ $6.7 \times 10^7$ FLOPs. AI ≈ 2, which is 78× below the ridge: the far-left of the memory slope. Attainable ceiling: $2 \times 2.0 = 4$ TFLOP/s, 1.3% of peak. If this LayerNorm eats 6% of your step's wall time, that is not a bug — it is a $6.7\times10^7$-byte round-trip to HBM that the roofline says *must* cost at least $6.7\times10^7 / 2\times10^{12} = 33$ µs, no matter how good the kernel.

**The fix the placement dictates:** fusion. The only way to make a LayerNorm faster is to make it stop being a separate kernel — fuse it into the epilogue of the preceding GEMM (which `torch.compile`'s Inductor does automatically), so the activation is produced, normalized, and consumed on-chip and never round-trips to HBM. That deletes the 33 µs entirely. Buying bandwidth would help linearly; buying FLOP/s would help not at all. The [bandwidth-bound-and-fusion post](/blog/machine-learning/performance-engineering/bandwidth-bound-and-fusion) is the whole playbook for this class.

Put those three side by side and the table writes itself. This is the deliverable the roofline produces for a service — one row per dominant kernel, and the last column is a work order.

| Kernel | AI (FLOP/byte) | Bound on A100 | Attainable | `ncu` signature | Right fix |
|---|---|---|---|---|---|
| Decode step, batch 1 | ≈ 1 | memory-bound | 2 TFLOP/s (0.6%) | Mem 93% / Comp 4% | batch, fp8 weights |
| Decode step, batch 64 | ≈ 64 | memory-bound | 128 TFLOP/s (41%) | Mem 70% / Comp 40% | keep batching to 156 |
| Big GEMM 4096³ | ≈ 1365 | compute-bound | 312 TFLOP/s (91%) | Mem 30% / Comp 88% | fp8, else stop |
| Naive attention, N=8192 | ≈ 63 | memory-bound | 126 TFLOP/s (40%) | Mem 85% / Comp 25% | FlashAttention |
| LayerNorm | ≈ 2 | memory-bound | 4 TFLOP/s (1.3%) | Mem 95% / Comp 3% | fuse into GEMM |
| Host-bound tiny kernels | n/a | neither roof | n/a | Mem 8% / Comp 6% | CUDA graphs |

The last row is the one the roofline itself cannot fix, and putting it in the table on purpose is the discipline: if both throughputs are low, the answer is "this is not a roofline problem," and the table tells you to stop reasoning about intensity and go look at launch overhead.

### The batch sweep: watching a kernel walk the roof

The single most useful experiment for building intuition — and for choosing a serving batch size — is to sweep batch size and watch the same kernel walk up the slope. Here is the sweep, and Figure 6 is its trajectory.

```python
import torch

A100_PEAK, A100_BW = 312e12, 2.0e12
RIDGE = A100_PEAK / A100_BW  # 156.0
W = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)

def sweep(M):
    x = torch.randn(M, 4096, device="cuda", dtype=torch.bfloat16)
    for _ in range(25): x @ W
    torch.cuda.synchronize()
    s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    s.record()
    for _ in range(100): x @ W
    e.record(); torch.cuda.synchronize()
    t = (s.elapsed_time(e) / 100) / 1e3
    flops = 2 * M * 4096 * 4096
    tflops = flops / t / 1e12
    return M, tflops, 100 * tflops / (A100_PEAK / 1e12)

for M in (1, 8, 32, 64, 128, 256):
    m, tf, pct = sweep(M)
    print(f"batch {m:4d}:  {tf:6.1f} TFLOP/s  {pct:5.1f}% of peak")
# batch    1:     0.8 TFLOP/s    0.3% of peak   <- memory-bound, wasted silicon
# batch    8:    12.9 TFLOP/s    4.1% of peak
# batch   32:    58.0 TFLOP/s   18.6% of peak
# batch   64:   118.4 TFLOP/s   38.0% of peak   <- climbing the slope
# batch  128:   201.7 TFLOP/s   64.6% of peak   <- approaching the ridge
# batch  256:   268.3 TFLOP/s   86.0% of peak   <- past the ridge, compute-bound
```

![a left to right timeline showing arithmetic intensity and percent of peak rising as batch size increases from one up to the compute ridge](/imgs/blogs/the-roofline-for-your-service-6.webp)

This is the roofline as a *dial*. Below the ridge (batch under ~156) you are memory-bound and every added request is nearly free in latency because the weights are already being read — you are filling idle tensor cores. Past the ridge you are compute-bound and added requests start costing real latency because the math units are now the constraint. The knee is the ridge, and it is the single most important operating point for a serving system: batch *up to* the ridge to convert wasted bandwidth into free throughput, and be cautious about batching past it, where you trade latency for throughput you may not need. This one sweep, run on your real model and hardware, tells you the batch size to configure — not a guessed one, a measured one that lands you at the knee.

## Different GPUs, different ridges

The ridge is $P_\text{peak}/B$, and both terms are properties of the specific chip, so **the ridge moves as you change hardware — and a kernel that is compute-bound on one GPU can be memory-bound on another.** This is the fact that makes "compute-bound vs memory-bound" a question you must re-answer for every deployment target, not once. Figure 7 puts three common serving GPUs side by side.

![a comparison grid of A100 H100 and L4 listing each chip's peak throughput its bandwidth and the resulting ridge intensity](/imgs/blogs/the-roofline-for-your-service-7.webp)

| GPU | Peak (bf16/fp16) | HBM bandwidth | Ridge AI | What it means |
|---|---|---|---|---|
| A100 80GB SXM | 312 TFLOP/s | 2.0 TB/s | 156 | the baseline |
| H100 SXM | 989 TFLOP/s | 3.35 TB/s | 295 | higher bar to be compute-bound |
| L4 | 242 TFLOP/s | 300 GB/s | 807 | almost everything is memory-bound |

Look at the L4. It has a respectable 242 fp16 TFLOP/s of tensor throughput but only 300 GB/s of memory bandwidth, so its ridge is 807 FLOP per byte — more than five times the A100's. On an L4, a kernel needs to do *807* FLOPs per byte before it is compute-bound, which means almost every kernel that was already memory-bound on the A100 is *even more* memory-bound on the L4, and some kernels that were compute-bound on the A100 (say AI ≈ 300) flip to memory-bound on the L4. The L4 is a cost-efficient inference chip precisely for workloads where you never expected to be compute-bound anyway — and a disastrous choice if you were counting on its FLOP/s, because its bandwidth will strangle you first. The H100 is the opposite lesson: raising peak FLOP/s from 312 to 989 while bandwidth only rises from 2.0 to 3.35 pushes the ridge *up* to 295, which means the H100 needs *more* intensity than the A100 before it pays off — exactly why the memory-bound decode service saw almost none of its 3.2× math.

The operational takeaway is a checklist item: recompute the ridge for every GPU you deploy to, and re-place your dominant kernels on it. A serving config tuned for the A100 ridge (batch to 156) is mistuned for the H100 (batch to 295) and wildly mistuned for the L4 (batch to 807, if you can afford the latency). The roofline is not one chart; it is one chart *per chip*, and the kernel does not move — the roof does.

#### Worked example: the same decode service, three chips

Take the 7B decode service, batch 32, and place it on all three. At batch 32, AI ≈ 32 (weight-dominated). On the A100 (ridge 156): memory-bound, attainable $32 \times 2.0 = 64$ TFLOP/s. On the H100 (ridge 295): still memory-bound, attainable $32 \times 3.35 = 107$ TFLOP/s — the H100 helps here, but through its *bandwidth*, and the win is 107/64 ≈ 1.7×, not 3.2×. On the L4 (ridge 807): deeply memory-bound, attainable $32 \times 0.30 = 9.6$ TFLOP/s — the L4 is bandwidth-starved for this batch and would need batch ~807 to use its tensor cores, which the latency budget of an interactive service will never allow. Same kernel, same batch, three completely different verdicts, and the only thing that changed is the roof. If you had budgeted the H100 upgrade expecting 3.2× and got 1.7×, the roofline told you why before you bought it: the decode kernel lives on the bandwidth roof, and the bandwidth only went up 1.7×.

## Case studies and real numbers

The roofline's predictions are not hypothetical; they are the load-bearing analysis behind several of the best-known GPU optimizations of the last few years. A few, with the numbers framed honestly.

**FlashAttention as a memory-wall fix.** The original FlashAttention work (Dao et al., 2022) is the cleanest published case of "the roofline predicted the fix." Standard attention's $O(N^2)$ HBM traffic from materializing the score matrix pins it below the ridge; FlashAttention tiles the computation to keep scores in SRAM, deleting the $N^2$ passes, and reports 2–4× wall-clock speedups on attention that *grow with sequence length* — exactly the signature the roofline predicts, because the deleted traffic scaled as $N^2$ while the retained traffic scales as $N$. The FLOPs never changed; only the denominator did. This is the canonical demonstration that a "slow" kernel can be memory-bound rather than compute-bound, and that the fix is fewer bytes, not more math. The [HPC FlashAttention post](/blog/machine-learning/high-performance-computing/kernel-fusion-and-flashattention-beating-the-memory-wall) works the intensities in detail.

**Batching to move the LLM-serving dot.** Every production LLM serving system — vLLM, TensorRT-LLM, TGI — is fundamentally a machine for raising the arithmetic intensity of decode by batching, and the throughput curves they publish are roofline curves: near-flat, low absolute throughput at batch 1 (memory-bound, wasted tensor cores), rising steeply with batch size as intensity climbs the slope, then flattening as the workload approaches the compute ridge or hits a KV-cache memory limit. Continuous batching wins because it keeps the effective batch high under bursty traffic, holding the dot as far right on the roofline as the latency budget allows. The exact numbers depend on model and hardware, but the *shape* is always the same, and it is the shape this post derived.

**channels_last as a bandwidth win.** For vision models, PyTorch's `channels_last` memory format is a roofline move hiding in a layout change: NHWC layout lets convolution kernels access memory contiguously, cutting effective HBM traffic and letting the kernel select faster tensor-core paths. PyTorch's own tutorial reports meaningful speedups on ResNet-family inference from the format change alone — no math changed, the bytes moved more efficiently, and a memory-bound conv moved up its bandwidth roof. It is the same principle as fusion (reduce or streamline bytes for a memory-bound op), applied through layout rather than kernel merging.

**The `ncu` roofline chart as ground truth.** NVIDIA's Nsight Compute draws the roofline for you — the "GPU Speed Of Light Roofline Chart" section places your kernel as a dot on the exact plot, computed from real hardware counters. This matters because the *achieved* intensity of a real kernel is often below the ideal you compute by hand: a naive GEMM that re-reads its operands from HBM moves more bytes than the formula assumes and lands left of where you predicted. `ncu` counts the actual bytes and actual FLOPs, so it captures the effective intensity your kernel really achieves. Use the calculator for the ceiling; use `ncu` for the truth; the gap between them is your kernel-optimization headroom.

## When to reach for the roofline (and when not to)

The roofline is the first analysis to run on any "why is this slow" question, because it is cheap — two numbers and a division — and it eliminates whole categories of wasted effort. But it is not the only analysis, and reaching for it reflexively has its own failure modes.

**Reach for it when** you are choosing hardware (the entire point — do not buy FLOP/s for a memory-bound service or bandwidth for a compute-bound one), when you are deciding a serving batch size (batch to the ridge), when a kernel is slow and you want to know whether the ceiling is math or memory before you invest in a rewrite, and when a "cheap" op is eating surprising wall time (the roofline explains the inversion). In all of these the roofline replaces a guess with a computed answer.

**Do not reach for it when both throughputs are low** — that is the "not on either roof" case, and the roofline has nothing to say about occupancy, launch overhead, or sync stalls. If `ncu` shows Compute 6% and Memory 8%, the roofline's advice ("cut bytes" or "add FLOP/s") is actively wrong; you have a host-bound or latency-bound kernel and the fix is CUDA graphs, larger kernels, or removing a synchronization, none of which the roofline recommends. **Do not use it to chase a memory-bound kernel that is already at 90%+ of bandwidth** — it is optimal for what it is, and the only remaining lever is to change the algorithm's byte count (fusion, quantization) or the hardware, not to tune the kernel. And **do not confuse the ideal roofline with the hierarchical one** for heavily-reused kernels: a well-tiled GEMM can clear the HBM ridge yet be bottlenecked by L2 bandwidth on its re-reads, which the single-line HBM roofline cannot see. For those, `ncu`'s hierarchical roofline (HBM *and* L2 lines) is the right tool, and the HPC roofline post covers that extension.

The meta-rule: the roofline tells you *which resource* bounds you, and it is authoritative about that. It does not tell you whether you are *near* the bound you are on — that is what the achieved-throughput number from the profiler tells you — and it says nothing at all when you are on neither roof. Run it first, believe its verdict on the bound, then use the profiler to see how much room is left under whichever roof it named.

## Key takeaways

- **Every kernel is memory-bound or compute-bound, decided by one number.** Arithmetic intensity — FLOPs per byte of HBM traffic — compared to the ridge $P_\text{peak}/B$ tells you which, and the answer dictates the fix. Compute it before you spend money.
- **Most inference is memory-bound.** Single-token decode reads all weights to do one tiny matmul, AI ≈ 1, ~1% of peak. This is physics, not a bug; the tensor cores are idle because there is nothing for them to do.
- **Batching is a roofline move.** It raises AI linearly (AI ≈ batch size for bf16) by reusing each weight read across more tokens, dragging the kernel from the memory slope toward the ridge. Batch *to* the ridge; it is the cheapest throughput you will ever get.
- **For a memory-bound op, measure GB/s, not MFU.** A decode kernel at 1% MFU may be at 93% of bandwidth — already optimal — or at 40% — a real bug. Only the bandwidth number distinguishes them; `nvidia-smi` and MFU cannot.
- **The fix follows the placement, mechanically.** Memory-bound → fuse, batch, lower precision (cut bytes). Compute-bound → precision, tensor cores (more FLOP/s). Under both roofs → not a roofline problem; go find the launch overhead or occupancy bug.
- **Read two `ncu` counters.** Memory Throughput % and Compute (SM) Throughput %. The higher one names your bound; if both are low, you are on neither roof and the roofline does not apply.
- **The ridge moves with the hardware.** Recompute $P_\text{peak}/B$ for every GPU — 156 on A100, 295 on H100, 807 on L4 — and re-place your kernels. A kernel that is compute-bound on one chip is memory-bound on another, and the L4's tiny bandwidth makes almost everything memory-bound.
- **Match the purchase to the bound.** The H100's 3.2× math helps compute-bound kernels and does nearly nothing for memory-bound ones; its 1.7× bandwidth is the only lever that touches decode. Buying the wrong axis is the most expensive mistake the roofline prevents.

## Further reading

- [The Roofline Model: Compute-Bound vs Memory-Bound](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) — the full from-scratch derivation, the log-log geometry, the hierarchical (HBM + L2) roofline, and the by-hand intensity of GEMM, LayerNorm, and attention. This post builds directly on it.
- [Kernel Fusion and FlashAttention: Beating the Memory Wall](/blog/machine-learning/high-performance-computing/kernel-fusion-and-flashattention-beating-the-memory-wall) — the canonical memory-bound fix worked end to end.
- [Metrics That Actually Matter](/blog/machine-learning/performance-engineering/metrics-that-actually-matter) — why GPU-util and MFU lie about where you are on the roofline, and which number to trust for a memory-bound op.
- [The Nsight Compute Kernel Deep Dive](/blog/machine-learning/performance-engineering/nsight-compute-kernel-deep-dive) — reading the Speed-of-Light roofline chart, occupancy, and warp-stall reasons for the "both throughputs low" case.
- [Bandwidth-Bound and Fusion](/blog/machine-learning/performance-engineering/bandwidth-bound-and-fusion) — the fix menu for the left side of the ridge: fusion, layout, and precision.
- [The Kernel-Launch-Overhead Problem](/blog/machine-learning/performance-engineering/the-kernel-launch-overhead-problem) — what to do when the roofline says "neither roof": the host-bound signature and CUDA graphs.
- [Why Your AI Service Wastes CPU and GPU](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu) — the series intro and the four-wastes frame.
- [The Performance Engineering Playbook](/blog/machine-learning/performance-engineering/the-performance-engineering-playbook) — the capstone decision tree that ties the roofline into the full symptom → tool → fix flow.
- NVIDIA Nsight Compute documentation — the "GPU Speed Of Light Roofline Chart" and Memory Workload Analysis sections, for pulling achieved intensity from hardware counters.
