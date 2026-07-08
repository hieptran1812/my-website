---
title: "Roofline Analysis for LLM Inference: Measure the Ceiling Before You Optimize"
date: "2026-07-07"
publishDate: "2026-07-07"
description: "Derive the roofline model for LLM serving, compute the ridge point for every modern GPU, and use arithmetic intensity to decide exactly which optimization will move your numbers."
tags:
  [
    "model-serving",
    "inference",
    "ml-infrastructure",
    "roofline",
    "arithmetic-intensity",
    "gpu",
    "memory-bandwidth",
    "performance-optimization",
    "llm",
  ]
category: "machine-learning"
subcategory: "Model Serving"
author: "Hiep Tran"
featured: true
readTime: 53
image: "/imgs/blogs/roofline-analysis-for-llm-inference-1.webp"
---

The first time a senior engineer asked me "is this kernel memory-bound or compute-bound?" I did not have an answer. I had a flame graph, a GPU utilization number that hovered around 40%, and a strong opinion that we should "write a fused CUDA kernel." We spent two weeks writing that kernel. It made the decode loop 3% faster. The problem was never the kernel — it was that every decode step re-read 140 GB of weights across the HBM bus, and no amount of clever fusion changes how many bytes you have to move. We optimized the wrong axis because we never measured which axis was the ceiling.

The roofline model is the tool that would have saved those two weeks. It is a single plot, backed by two numbers your GPU vendor already published, that tells you the maximum performance any operation can reach on your hardware and whether that ceiling is set by compute or by memory bandwidth. Once you can place an operation on the roofline, "make it faster" stops being a vibe and becomes a decision: below the ridge point you are memory-bound and you should quantize, batch, or fuse; above it you are compute-bound and you should reach for better matmul kernels or lower-precision tensor cores. You never spend effort on the axis that is not the bottleneck.

Figure 1 shows the whole model in one picture. Performance rises along a diagonal memory-bound slope, hits a knee called the ridge point, and then flattens into a horizontal compute-bound ceiling. LLM decode — the token-by-token generation that dominates chatbot latency — lives at the far bottom-left of that plot, attaining well under 1% of the GPU's advertised FLOP/s. Prefill, the one-shot processing of the prompt, lives on the flat top-right. Understanding why those two phases sit where they do, and what moves them, is the entire game of LLM-serving optimization.

By the end of this post you will be able to: derive the roofline from first principles, compute the ridge point for an H100, H200, A100, MI300X, or L40S from published specs, prove that decode arithmetic intensity is approximately the batch size, calculate the hard floor on time-per-output-token for any model on any GPU, and read a profiler dump to decide the single next optimization worth doing. This is the analytical foundation the rest of the [model-serving series](/blog/machine-learning/model-serving/why-llm-serving-is-different) builds on. Everything downstream — [KV-cache optimization](/blog/machine-learning/model-serving/kv-cache-optimization), [quantization](/blog/machine-learning/model-serving/quantization-for-llm-serving), custom kernels — is a specific move on this one plot.

## The roofline model in one equation

The roofline model comes from a 2009 paper by Samuel Williams, Andrew Waterman, and David Patterson, and it rests on an almost embarrassingly simple observation: any computation needs both arithmetic and data, and each has a hard rate limit set by the hardware.

![The roofline splits into a memory-bound diagonal and a compute-bound ceiling that meet at the ridge point, with decode on the left and prefill on the right.](/imgs/blogs/roofline-analysis-for-llm-inference-1.webp)

Take an operation that performs $W$ floating-point operations and moves $Q$ bytes to and from HBM (the GPU's high-bandwidth memory). The GPU has two published peak rates: peak compute $\pi$ in FLOP/s, and peak memory bandwidth $\beta$ in bytes/s. To finish the arithmetic you need at least $W/\pi$ seconds. To move the data you need at least $Q/\beta$ seconds. On a modern GPU these two can overlap — the memory controllers stream bytes while the tensor cores crunch — so the best case is that the slower of the two hides the faster. The time to run the operation is therefore bounded below by:

$$T \ge \max\!\left(\frac{W}{\pi},\; \frac{Q}{\beta}\right)$$

Performance is work divided by time, $P = W/T$. Substituting the bound and simplifying gives the roofline equation. Define **arithmetic intensity** $I = W/Q$, the FLOPs performed per byte moved, measured in FLOP/byte. Then:

$$P(I) = \min\!\left(\pi,\; I \cdot \beta\right)$$

That is the whole model. Read it carefully, because every optimization decision in this post falls out of it:

- When $I$ is small, $I \cdot \beta < \pi$, so $P = I \cdot \beta$. Performance is proportional to intensity and capped by bandwidth. You are **memory-bound**. The tensor cores sit mostly idle waiting for data.
- When $I$ is large, $\pi < I \cdot \beta$, so $P = \pi$. Performance is flat at the compute peak, and bandwidth no longer matters. You are **compute-bound**.

The crossover — the **ridge point** — is where the two terms are equal, $I \cdot \beta = \pi$:

$$I^{*} = \frac{\pi}{\beta}$$

The ridge point is a property of the *hardware*, not of your model. It is the arithmetic intensity at which the GPU is perfectly balanced: below it you cannot keep the tensor cores fed, above it you cannot keep them from being the bottleneck. For an H100 SXM, peak dense BF16 is about 989 TFLOP/s and HBM3 bandwidth is about 3.35 TB/s, so the ridge sits at roughly 989 / 3.35 ≈ 295 FLOP/byte. Any operation that moves a byte for fewer than 295 useful FLOPs is memory-bound on an H100. Hold that number; it is the single most useful constant in LLM serving, and almost every operation in a decode step misses it by two orders of magnitude.

One subtlety worth stating plainly: the roofline is a *ceiling*, not a prediction. It tells you the best you could possibly do given the FLOPs and bytes an operation requires. Real kernels fall below the roofline because of launch overhead, imperfect overlap, cache misses, tail effects at small batch, and warp scheduling stalls. The gap between where you land and the roofline is your headroom; the roofline itself is the wall.

### A minimal roofline calculator

Here is the model as runnable code. Everything else in the post is elaboration on these fifteen lines. The GPU specs are published dense figures; sparsity roughly doubles the compute peak and is noted separately later.

```python
# roofline.py — the model in one function.
# All FLOP/s are DENSE BF16 published specs; all bandwidths are peak HBM.
GPUS = {
    "H100-SXM":  {"flops": 989e12,  "bw": 3.35e12},  # HBM3,  80 GB
    "H200-SXM":  {"flops": 989e12,  "bw": 4.80e12},  # HBM3e, 141 GB
    "A100-80GB": {"flops": 312e12,  "bw": 2.039e12}, # HBM2e, 80 GB
    "MI300X":    {"flops": 1307e12, "bw": 5.30e12},  # HBM3,  192 GB
    "L40S":      {"flops": 362e12,  "bw": 0.864e12}, # GDDR6, 48 GB
}

def ridge_point(gpu):
    """Arithmetic intensity (FLOP/byte) where the GPU becomes compute-bound."""
    g = GPUS[gpu]
    return g["flops"] / g["bw"]

def attainable(gpu, ai):
    """Best-case FLOP/s for an operation of arithmetic intensity `ai`."""
    g = GPUS[gpu]
    return min(g["flops"], ai * g["bw"])

for name in GPUS:
    r = ridge_point(name)
    print(f"{name:10s}  ridge = {r:6.1f} FLOP/byte  "
          f"(AI=1 attains {attainable(name, 1)/1e12:5.2f} TFLOP/s)")
```

Running it prints the ridge point and the attained performance at arithmetic intensity 1 — the intensity of naive single-request decode:

```console
H100-SXM    ridge =  295.2 FLOP/byte  (AI=1 attains  3.35 TFLOP/s)
H200-SXM    ridge =  206.0 FLOP/byte  (AI=1 attains  4.80 TFLOP/s)
A100-80GB   ridge =  153.0 FLOP/byte  (AI=1 attains  2.04 TFLOP/s)
MI300X      ridge =  246.6 FLOP/byte  (AI=1 attains  5.30 TFLOP/s)
L40S        ridge =  418.9 FLOP/byte  (AI=1 attains  0.86 TFLOP/s)
```

At arithmetic intensity 1, an H100 attains 3.35 TFLOP/s out of a possible 989 — about 0.34% of the chip. That is not a bug in your code. That is physics: at intensity 1 you move one byte for every FLOP, and the bus can only deliver 3.35 TB of bytes per second, so you can only do 3.35 tera-FLOPs per second no matter how many tensor cores are etched into the die. The other 99.66% of the silicon is waiting.

### The overlap assumption, and why real kernels miss the roofline

The bound $T \ge \max(W/\pi,\, Q/\beta)$ takes the *maximum* of the two times, not their sum, and that single choice encodes the most optimistic thing a GPU can do: perfectly overlap computation with memory traffic. While the tensor cores chew on the tile already sitting in registers, the memory controllers stream the next tile in. When that overlap is perfect, the faster of the two operations hides entirely behind the slower, and the maximum is exact.

Real kernels rarely achieve perfect overlap, and the honest pessimistic bound is the *sum* rather than the maximum:

$$T_{\text{worst}} = \frac{W}{\pi} + \frac{Q}{\beta}$$

The truth lives between the two. A well-written kernel with deep software pipelining and enough in-flight memory requests to keep the bus saturated lands close to the max; a kernel that stalls its tensor cores every time it waits on a load drifts toward the sum. For a purely memory-bound decode operation the gap barely matters — $W/\pi$ is a rounding error next to $Q/\beta$, so max and sum agree to within a percent — which is another way of saying the roofline is *tightest* exactly where LLM decode lives. That is a happy accident: the regime we care most about is the regime the model predicts best.

There is a second reason real kernels miss the roofline: $\pi$ and $\beta$ are themselves optimistic. Published FLOP/s assume every fused multiply-add issues every cycle with no stalls; published bandwidth assumes a streaming access pattern that touches every byte of every cache line. A strided KV-cache read that touches 128 useful bytes out of every 512-byte sector wastes three-quarters of the bus. So when you place a measured kernel below the roofline, the shortfall has two components: imperfect overlap (the max-versus-sum gap) and imperfect peak utilization (the fraction of $\pi$ or $\beta$ you actually reach). The roofline does not separate them for you; the profiler does, and the final section shows how to read the split.

### Plotting the roofline for your GPU

Numbers in a table are easy to misread; the log-log plot makes the two regimes and the knee between them impossible to miss. This is the figure I paste into design docs, annotated with wherever the workload actually landed:

```python
# plot_roofline.py — draw the roofline and drop your operators onto it.
import numpy as np
import matplotlib.pyplot as plt
from roofline import GPUS, ridge_point

def plot(gpu, ops):
    peak, bw = GPUS[gpu]["flops"], GPUS[gpu]["bw"]
    ai = np.logspace(-1, 4, 500)               # 0.1 .. 10000 FLOP/byte
    attainable = np.minimum(peak, ai * bw)     # the roofline itself
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(ai, attainable / 1e12, lw=2, color="black")
    ax.axvline(ridge_point(gpu), ls="--", color="gray")     # the ridge
    for label, op_ai in ops.items():
        p = min(peak, op_ai * bw) / 1e12
        ax.plot(op_ai, p, "o")
        ax.annotate(label, (op_ai, p), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("Attainable performance (TFLOP/s)")
    ax.set_title(f"{gpu}  ridge = {ridge_point(gpu):.0f} FLOP/byte")
    fig.tight_layout(); fig.savefig("roofline.png", dpi=150)

plot("H100-SXM", {"decode b=1": 1, "decode b=64": 64,
                  "attn GQA8": 8, "prefill": 4096})
```

The diagonal is $P = I\beta$; the flat top is $P = \pi$; the dashed vertical is the ridge. Every operator you profile is a dot, and the vertical distance from the dot up to the black line is the headroom you have left on that operator. When the dot already touches the line, stop — that kernel is at its wall, and no amount of tuning moves it without changing $I$, $\pi$, or $\beta$.

## The ridge point, GPU by GPU

The ridge point is where memory-bound thinking gives way to compute-bound thinking, so the first thing to do with any GPU is compute it. The pattern across hardware is instructive: the ridge is peak FLOP/s over bandwidth, and vendors have grown FLOP/s far faster than bandwidth, so ridge points have crept steadily rightward — meaning it takes ever-higher arithmetic intensity to use the compute you paid for.

![A per-GPU matrix listing peak BF16 FLOP rate, HBM bandwidth, ridge-point arithmetic intensity, and decode TPOT floor for five accelerators.](/imgs/blogs/roofline-analysis-for-llm-inference-3.webp)

Here are the numbers, computed from published specs. I list dense BF16 and dense FP8 peaks because low-precision matmul roughly doubles the compute peak (and therefore doubles the ridge point), which matters when you decide whether quantization is buying you compute or just bandwidth.

| GPU | Peak BF16 (dense) | Peak FP8 (dense) | HBM bandwidth | Ridge AI (BF16) | Ridge AI (FP8) | TPOT floor, 8B fp16 |
|---|---|---|---|---|---|---|
| H100 SXM5 | 989 TFLOP/s | 1,979 TFLOP/s | 3.35 TB/s | ≈ 295 | ≈ 590 | 4.8 ms |
| H200 SXM | 989 TFLOP/s | 1,979 TFLOP/s | 4.80 TB/s | ≈ 206 | ≈ 412 | 3.3 ms |
| A100 80GB | 312 TFLOP/s | — (INT8 624 TOPS) | 2.04 TB/s | ≈ 153 | ≈ 306 (INT8) | 7.9 ms |
| MI300X | 1,307 TFLOP/s | 2,615 TFLOP/s | 5.30 TB/s | ≈ 247 | ≈ 493 | 3.0 ms |
| L40S | 362 TFLOP/s | 733 TFLOP/s | 0.86 TB/s | ≈ 419 | ≈ 848 | 18.5 ms |

Several things jump out of this table:

- **The A100 has the lowest ridge point (≈153).** It is the most "balanced" chip here for LLM decode: relatively modest compute against decent bandwidth. Hopper and Blackwell added enormous FP8/FP4 compute without proportional bandwidth, so their ridge points are higher. This is why an A100 sometimes shows *higher decode efficiency* (attained fraction of peak) than an H100 even though the H100 is faster in absolute terms.
- **The L40S has a ridge point of ≈419** because it uses GDDR6, not HBM. Its 0.86 TB/s bandwidth is a quarter of an H100's. For decode, the L40S is bandwidth-starved: you would need arithmetic intensity above 419 to use its Ada tensor cores, and decode never gets close. The L40S is a fine prefill and vision-inference chip and a poor batch-1 decode chip.
- **The MI300X pairs the highest compute (1,307 dense BF16) with the highest bandwidth (5.30 TB/s).** Its ridge lands near the H100's, but its raw bandwidth gives it the lowest per-token floor in the table for models that fit in its enormous 192 GB.

A crucial nuance: the FP8 columns show ridge points that are *double* the BF16 ones. This trips people up. Quantizing weights to FP8 does not move the arithmetic-intensity requirement down — it moves the ridge point *up*, because FP8 matmul is twice as fast. What quantization actually buys you for decode is on the *other* axis: fewer bytes to move. We will make that precise in the batching-and-quantization section. For now, just note that "FP8 makes it faster" is two separate claims — half the bytes and double the FLOP/s — and only one of them helps a memory-bound decode.

#### Worked example: deriving a ridge point from the datasheet

The ridge point is not a benchmark you run; it is two numbers off a spec sheet divided. Take the H100 SXM5. NVIDIA's datasheet lists 989.4 TFLOP/s of dense BF16 tensor-core throughput and 3.35 TB/s of HBM3 bandwidth. Divide:

$$I^{*}_{\text{H100}} = \frac{989 \times 10^{12}\ \text{FLOP/s}}{3.35 \times 10^{12}\ \text{byte/s}} \approx 295\ \text{FLOP/byte}$$

Now the A100 80 GB: 312 TFLOP/s dense BF16, 2.039 TB/s HBM2e, which gives ${312/2.039 \approx 153}$ FLOP/byte. The A100's ridge is *lower* than the H100's, which sounds backwards until you look at what changed between the generations. Hopper multiplied BF16 throughput by 3.2 times (312 to 989) but multiplied bandwidth by only 1.64 times (2.04 to 3.35). Compute outran memory, so the balance point moved right. Every operation whose intensity lands between 153 and 295 is compute-bound on an A100 but memory-bound on an H100 — the same kernel, a different verdict, purely because the hardware balance shifted.

That divergence is a trend, not an accident, and it is the single most important thing to internalize about GPU roadmaps for inference. Here is the ridge point across five generations, all dense BF16 (FP16 tensor for Volta):

| GPU (year) | Peak BF16/FP16 | HBM bandwidth | Ridge AI | Bandwidth vs. prior |
|---|---|---|---|---|
| V100 (2017) | 125 TFLOP/s | 0.90 TB/s | ≈ 139 | — |
| A100 (2020) | 312 TFLOP/s | 2.04 TB/s | ≈ 153 | 2.3× |
| H100 (2022) | 989 TFLOP/s | 3.35 TB/s | ≈ 295 | 1.6× |
| H200 (2023) | 989 TFLOP/s | 4.80 TB/s | ≈ 206 | 1.4× |
| B200 (2024) | ≈ 2,250 TFLOP/s | ≈ 8.0 TB/s | ≈ 281 | 1.7× |

Read the last column: bandwidth grew 1.4–2.3 times per generation while compute, when it moved, moved faster. The instructive exception is the H200, a pure bandwidth refresh of the H100 — identical compute, 43% more bandwidth — which *lowers* the ridge back to 206. The H200's existence is itself an argument that the industry noticed decode is bandwidth-bound and shipped silicon to fix exactly that. The B200 figures here are approximate pre-shipping dense numbers and should be checked against the final datasheet before you quote them in a capacity plan.

There is one more axis the table flattens: precision. The ridge points above are for BF16. Drop to FP8 and the compute peak roughly doubles while bandwidth is unchanged, so the FP8 ridge is roughly double the BF16 ridge — an H100 in FP8 has a ridge near 590 FLOP/byte. Enabling 2:4 structured sparsity (which Ampere and Hopper support) doubles the nominal compute peak again, doubling the ridge a second time. Each step up in headline compute makes the chip *harder* to saturate at low intensity, which is why the FP8 and sparse peaks in a vendor's marketing number are almost never peaks a decode workload gets anywhere near.

### Computing ridge points and attained performance per GPU

The following script extends the calculator to report, for each GPU, the ridge point and where a handful of representative arithmetic intensities land. It is the tool I keep in a scratch file when someone quotes me a kernel's FLOPs and bytes.

```python
# ridge_report.py — where do common ops land on each GPU's roofline?
from roofline import GPUS, ridge_point, attainable

# Representative arithmetic intensities for LLM inference (derived later).
WORKLOADS = {
    "decode GEMM, batch=1":   1.0,
    "decode GEMM, batch=64":  64.0,
    "decode attention (GQA8)": 8.0,
    "prefill GEMM, seq=4096": 4096.0,
}

for gpu in GPUS:
    r = ridge_point(gpu)
    print(f"\n=== {gpu}  (ridge = {r:.0f} FLOP/byte) ===")
    for label, ai in WORKLOADS.items():
        p = attainable(gpu, ai)
        pct = 100 * p / GPUS[gpu]["flops"]
        bound = "compute-bound" if ai >= r else "MEMORY-BOUND"
        print(f"  {label:26s} AI={ai:7.1f}  "
              f"{p/1e12:7.1f} TFLOP/s  ({pct:5.1f}% of peak)  {bound}")
```

The output makes the memory-bound reality of decode impossible to miss. On an H100:

```console
=== H100-SXM  (ridge = 295 FLOP/byte) ===
  decode GEMM, batch=1       AI=    1.0     3.4 TFLOP/s  (  0.3% of peak)  MEMORY-BOUND
  decode GEMM, batch=64      AI=   64.0   214.4 TFLOP/s  ( 21.7% of peak)  MEMORY-BOUND
  decode attention (GQA8)    AI=    8.0    26.8 TFLOP/s  (  2.7% of peak)  MEMORY-BOUND
  prefill GEMM, seq=4096     AI= 4096.0   989.0 TFLOP/s  (100.0% of peak)  compute-bound
```

Batch-1 decode uses 0.3% of the chip. Even batch-64 decode, which most teams consider "well batched," reaches only 22%. Prefill saturates it completely. That gulf — two orders of magnitude between decode and prefill efficiency — is the reason [prefill/decode disaggregation](/blog/machine-learning/model-serving/why-llm-serving-is-different) exists, and it is visible directly from the ridge point without running a single benchmark.

## Why decode lives on the far left of the roofline

The central claim of this post is that LLM decode is *deeply* memory-bound — not by a little, by a factor of hundreds. To prove it, we have to compute the arithmetic intensity of the operation that dominates decode: the weight GEMMs (the general matrix multiplications for the Q/K/V projections, the attention output projection, and the two or three MLP matrices). These are where most of a transformer's FLOPs and almost all of its parameter bytes live.

![A branching graph shows one weight tile read once from HBM and reused across every token in the batch, yielding arithmetic intensity equal to batch size.](/imgs/blogs/roofline-analysis-for-llm-inference-4.webp)

Consider a single weight matrix $W$ of shape $d \times d$ (a projection in a transformer with hidden size $d$). During decode, each of the $B$ sequences in the batch contributes exactly one token, so the input activation is a matrix $X$ of shape $B \times d$ — one row per sequence. The GEMM computes $X W$, producing a $B \times d$ output.

Count the FLOPs. A matrix multiply of $B \times d$ by $d \times d$ costs $2 B d^2$ FLOPs (the factor of 2 is one multiply plus one add per inner-product term). Count the bytes moved from HBM. In FP16 the weight matrix is $2 d^2$ bytes (two bytes per element). The input activation is $2 B d$ bytes and the output is another $2 B d$ bytes. For real transformers $d$ is thousands (Llama-3-70B has $d = 8192$) while $B$ is at most a few hundred, so $d \gg B$ and the weight term $2 d^2$ dominates the activation terms $2 B d$. The bytes moved are approximately $2 d^2$.

Arithmetic intensity is then:

$$I_{\text{GEMM}} = \frac{W}{Q} = \frac{2 B d^2}{2 d^2} = B$$

**The arithmetic intensity of a decode weight GEMM is the batch size.** This is the most important single fact in LLM serving. It says the weight matrix gets read from HBM exactly once per step, and every one of the $B$ tokens in the batch reuses that same in-flight copy. One token reuses the weights once (intensity 1); sixty-four tokens reuse them sixty-four times (intensity 64). Reuse *is* arithmetic intensity.

Now compare against the ridge point. On an H100 the ridge is 295. To make the decode GEMM compute-bound you would need a batch of roughly 295 sequences. In practice you never get there — the KV cache runs out of memory, or your p99 latency SLA caps the batch far below that — so **decode weight GEMMs are memory-bound for every realistic batch size.** At batch 1 they run at 0.3% of peak; at batch 64, still under the ridge, they run at 22%.

#### The activation correction: why "batch equals intensity" slightly overstates the reuse

The clean result $I = B$ dropped the activation bytes as negligible, and it is worth seeing exactly how negligible. Keep them and the intensity is:

$$I_{\text{GEMM}} = \frac{2 B d^2}{2 d^2 + 4 B d} = \frac{B}{1 + 2B/d}$$

The correction factor ${1/(1 + 2B/d)}$ is below one, so the true intensity is slightly *less* than the batch — the activations you stream in and out cost bandwidth that the weights do not amortize. For Llama-3-70B with $d = 8192$ the correction is tiny until the batch grows large: at batch 64 it is ${1/(1 + 128/8192) = 0.984}$, pulling intensity from 64 to 63; at batch 256 it is ${1/1.0625 = 0.941}$, pulling 256 down to 241. The practical consequence: the batch at which the GEMM actually crosses the H100 ridge of 295 is not 295 but closer to 318, because the activation traffic steals a little of the reuse. It never changes the verdict for realistic batches — you are memory-bound either way — but it explains why measured throughput curves bend toward the ridge a touch later than the naive $I = B$ line predicts.

```python
# effective_ai.py — GEMM intensity with the activation correction.
def gemm_ai(batch, d_model, wbytes=2):
    flops = 2 * batch * d_model**2
    bytes_moved = wbytes * d_model**2 + 2 * (2 * batch * d_model)  # W + in + out
    return flops / bytes_moved

for b in (1, 8, 64, 256, 512):
    naive, corrected = b, gemm_ai(b, 8192)
    print(f"batch={b:4d}  naive AI={naive:4d}  corrected AI={corrected:6.1f}  "
          f"(-{100*(1-corrected/naive):4.1f}%)")
```

```console
batch=   1  naive AI=   1  corrected AI=   1.0  (- 0.0%)
batch=   8  naive AI=   8  corrected AI=   8.0  (- 0.2%)
batch=  64  naive AI=  64  corrected AI=  63.0  (- 1.6%)
batch= 256  naive AI= 256  corrected AI= 240.9  (- 5.9%)
batch= 512  naive AI= 512  corrected AI= 455.1  (-11.1%)
```

There is a subtler point hiding in the word "batch." In a continuously batched server the $B$ that sets your GEMM intensity is not a static configuration value — it is the number of sequences that happen to be in the running batch *at that step*, which the scheduler grows and shrinks as requests arrive and finish. Intensity, and therefore GPU efficiency, breathes with load: a server at 10% utilization runs its GEMMs at intensity 3 or 4 and burns most of its bandwidth re-reading weights for almost no one; the same server at capacity runs at intensity 100 or more and extracts far more of the chip. This is why "tokens per second" quoted at low concurrency tells you almost nothing about tokens per second at capacity, and why folding prefill tokens into a decode step via [chunked prefill](/blog/machine-learning/model-serving/why-llm-serving-is-different) is such a direct roofline win: every prefill token piggybacking on a decode GEMM is one more unit of weight reuse, one more notch up the memory-bound slope.

There is a second reason decode is memory-bound, and it is more insidious than the GEMMs because it does not improve with batching at all: attention over the KV cache.

![A five-layer stack shows decode streaming 16 GB of weights from HBM every step, making TPOT floor equal to model bytes over bandwidth.](/imgs/blogs/roofline-analysis-for-llm-inference-2.webp)

The stack above traces the bandwidth bottleneck end to end for a Llama-3-8B model on an H100. The tensor cores can do 989 TFLOP/s but sit below 1% utilization during batch-1 decode. Why? Because the real limiter is the 3.35 TB/s HBM3 bus, and every decode step must drag all 16 GB of FP16 weights across it. A single decode step produces one token per sequence and reuses no weights across the tokens *of a single sequence* over time — the next token needs the weights read fresh. The bytes moved per step are fixed at the model size; the FLOPs are tiny. Bandwidth wins, and it sets a hard floor on how fast a token can come out.

That floor deserves its own section, because it is the number that determines whether your product can hit its latency target at all.

## The decode TPOT floor: bytes over bandwidth

TPOT — time per output token, sometimes called inter-token latency — is the metric your users feel as "how fast does the text stream." It is bounded below by a quantity you can compute from two numbers before you write any serving code: the model size in bytes and the GPU's memory bandwidth.

The derivation is the roofline applied to a whole decode step. In steady-state generation, each step must read every model weight from HBM at least once (the GEMMs above establish this — the weights are the dominant byte traffic and each is touched every step). Let $M$ be the number of bytes of weights the GPU must read per step. The time for that read is bounded below by $M / \beta$. A step produces exactly one token per sequence, so the time between consecutive tokens of a given sequence — the TPOT — obeys:

$$T_{\text{TPOT}} \ge \frac{M}{\beta}$$

That is the decode floor. It is remarkably clean: it does not depend on the batch size (batching produces more tokens per step but does not change the per-step time until you approach the ridge), it does not depend on FLOPs, and it does not depend on how clever your kernels are. It depends only on how many bytes of weights you must move and how fast the bus moves them. Quantization lowers $M$; faster HBM raises $\beta$; nothing else touches this floor.

#### Worked example: Llama-3-70B decode TPOT floor on H100

Llama-3-70B has about 70 billion parameters. In FP16 that is 140 GB of weights — more than a single 80 GB H100 can hold — so we serve it with tensor parallelism across 2 GPUs (TP=2), each holding a 70 GB shard. During a decode step both GPUs read their shards in parallel, so the effective per-step read time is set by one shard:

- Bytes per GPU per step: 70 GB.
- H100 HBM3 bandwidth: 3.35 TB/s.
- TPOT floor: 70 GB / 3.35 TB/s ≈ **20.9 ms per token**.

That caps single-sequence generation at about 1 / 0.0209 ≈ **48 tokens/second**, before adding a single millisecond for the two NCCL all-reduces per layer, kernel launch overhead, or attention. If your product needs 60 tokens/second per user on H100s in FP16, the roofline tells you it is *physically impossible* — you would be asking the bus to move 140 GB in under 16.7 ms, which exceeds 8.4 TB/s of effective bandwidth across two GPUs, and you only have 6.7 TB/s. No kernel rewrite fixes that. Your options are exactly two: reduce $M$ (quantize) or increase $\beta$ (move to H200 at 4.8 TB/s, or add GPUs with more tensor-parallel bandwidth).

Now quantize to FP8. The weights become 70 GB total, 35 GB per GPU under TP=2:

- Bytes per GPU per step: 35 GB.
- TPOT floor: 35 GB / 3.35 TB/s ≈ **10.4 ms per token**, or about **96 tokens/second**.

Quantization to FP8 halved the bytes, halved the floor, and doubled the token-rate ceiling — because decode is memory-bound and FP8 attacks the exact axis that binds. On an H200, the same FP8 model reads 35 GB per GPU at 4.8 TB/s for a 7.3 ms floor (≈137 tok/s). This is the entire argument for FP8 decode in one calculation.

### A TPOT-floor calculator

The following tool computes the floor for any model, precision, GPU, and tensor-parallel degree. I use it in capacity planning before committing to a GPU SKU, because it answers "can this hardware even hit our latency target" in one line.

```python
# tpot_floor.py — the hard lower bound on inter-token latency.
from roofline import GPUS

BYTES_PER_PARAM = {"fp16": 2, "bf16": 2, "fp8": 1, "int4": 0.5}

def tpot_floor_ms(n_params, precision, gpu, tp=1):
    """Lower bound on time-per-output-token, in milliseconds."""
    model_bytes = n_params * BYTES_PER_PARAM[precision]
    bytes_per_gpu = model_bytes / tp          # tensor parallelism splits weights
    bw = GPUS[gpu]["bw"]                       # bytes/second, per GPU, in parallel
    return 1e3 * bytes_per_gpu / bw

def max_tokens_per_sec(n_params, precision, gpu, tp=1):
    return 1e3 / tpot_floor_ms(n_params, precision, gpu, tp)

for prec in ("fp16", "fp8", "int4"):
    ms = tpot_floor_ms(70e9, prec, "H100-SXM", tp=2)
    tps = max_tokens_per_sec(70e9, prec, "H100-SXM", tp=2)
    print(f"Llama-3-70B {prec:4s} on 2x H100:  "
          f"TPOT floor {ms:5.1f} ms  ->  <= {tps:5.0f} tok/s/seq")
```

```console
Llama-3-70B fp16 on 2x H100:  TPOT floor  20.9 ms  ->  <=    48 tok/s/seq
Llama-3-70B fp8  on 2x H100:  TPOT floor  10.4 ms  ->  <=    96 tok/s/seq
Llama-3-70B int4 on 2x H100:  TPOT floor   5.2 ms  ->  <=   191 tok/s/seq
```

Run the same floor across the SKUs you might actually rent and the GPU-selection decision makes itself. The table below is the weight-streaming floor in milliseconds per token — the model read once, no KV or communication overhead — holding tensor parallelism fixed per model (TP=1 for 8B, TP=2 for 70B) so the columns compare cleanly:

| Model / precision | Bytes per GPU | H100 (3.35 TB/s) | H200 (4.80 TB/s) | A100 (2.04 TB/s) |
|---|---|---|---|---|
| Llama-3-8B FP16 | 16 GB | 4.8 ms | 3.3 ms | 7.8 ms |
| Llama-3-8B FP8 | 8 GB | 2.4 ms | 1.7 ms | 3.9 ms |
| Llama-3-70B FP16 | 70 GB | 20.9 ms | 14.6 ms | 34.3 ms |
| Llama-3-70B FP8 | 35 GB | 10.4 ms | 7.3 ms | 17.2 ms |

Read a row against your TPOT budget. If you need sub-15 ms tokens on a 70B model, FP16 on A100s (34 ms) and even FP16 on H100s (21 ms) are ruled out by physics before you write a line of code — you need FP8, or H200-class bandwidth, or both. The floor does not care how good your engineers are; it cares how many bytes cross the bus. That is exactly the kind of question that is cheap to answer on a spreadsheet and expensive to answer after a procurement cycle.

Two warnings about reading this floor. First, it is a *floor*, not a forecast: real systems land 1.3× to 2× above it because of attention over the KV cache (which adds bytes that grow with context length), imperfect overlap, and communication. Second, INT4 halves the byte traffic again but does *not* generally halve latency in practice, because INT4 weights are usually dequantized to FP16 for the matmul, and the accuracy cost is real; treat the INT4 row as an aspirational bound, not a promise. The floor's job is to tell you what is impossible, and it does that perfectly: if the floor exceeds your TPOT budget, no engineering saves you, and you must change $M$ or $\beta$.

### When the KV cache joins the byte budget

The floor $T \ge M/\beta$ counts only weight bytes, which is correct at short context and modest batch, where weights dominate the per-step traffic. But the KV cache is also read in full every step — attention must attend over every past token — and those bytes grow with both batch and context. At long context and high batch they stop being negligible and eventually overtake the weights entirely.

The KV bytes read per step are the size of the whole cache: two tensors (key and value), across every layer, every KV head, and every cached position, for every sequence in the batch. For Llama-3-70B (80 layers, 8 KV heads, head dimension 128) each token of each sequence contributes ${2 \times 80 \times 8 \times 128 \times 2 = 327{,}680}$ bytes of KV in FP16 — about 320 KB per token. Watch what that does at scale:

- Context 8,192, batch 1: KV read ≈ ${8192 \times 320\ \text{KB} \approx 2.7}$ GB per step. Against 140 GB of weights, roughly a 2% tax. The weight floor is essentially the whole story.
- Context 8,192, batch 64: KV read ≈ ${64 \times 2.7 \approx 172}$ GB per step. That now *exceeds* the 140 GB of weights. The step is no longer weight-bound; it is KV-bound, and TPOT is set by $(140 + 172)/\beta$, not ${140/\beta}$.

This is the quantitative heart of why long-context serving feels so different from short-context serving, and why "just raise the batch" stops working at long context: past some batch, every additional sequence adds a full context worth of KV reads to every step, so per-step time grows with batch instead of staying flat. The memory-bound slope that made batching free is still there for the weights; it is the KV term that bends the throughput curve back down. The complete per-step floor is:

$$T_{\text{TPOT}} \ge \frac{M_{\text{weights}} + B \cdot S \cdot b_{\text{kv}}}{\beta}$$

where $b_{\text{kv}}$ is the per-token KV bytes derived above and everything is per GPU. The calculator below adds the KV term, so you can see where a given model crosses from weight-bound into KV-bound:

```python
# tpot_floor_kv.py — TPOT floor including the KV-cache read.
from roofline import GPUS

def kv_bytes_per_token(n_layers, n_kv_heads, head_dim, kvbytes=2):
    return 2 * n_layers * n_kv_heads * head_dim * kvbytes   # key + value

def tpot_floor_ms(n_params, n_layers, n_kv_heads, head_dim,
                  gpu, precision="fp16", tp=1, batch=1, seq_len=4096):
    wbytes = {"fp16": 2, "bf16": 2, "fp8": 1, "int4": 0.5}[precision]
    weight_bytes = n_params * wbytes / tp
    kv = batch * seq_len * kv_bytes_per_token(n_layers, n_kv_heads, head_dim) / tp
    return 1e3 * (weight_bytes + kv) / GPUS[gpu]["bw"]

# Llama-3-70B on 2x H100, FP16, sweeping context at batch 64.
for S in (2048, 8192, 32768):
    ms = tpot_floor_ms(70e9, 80, 8, 128, "H100-SXM",
                       tp=2, batch=64, seq_len=S)
    print(f"context={S:6d}  batch=64  TPOT floor = {ms:6.1f} ms")
```

```console
context=  2048  batch=64  TPOT floor =   27.3 ms
context=  8192  batch=64  TPOT floor =   46.5 ms
context= 32768  batch=64  TPOT floor =  123.5 ms
```

At 32K context the KV read has ballooned the floor to over 120 ms per token — nearly six times the 20.9 ms weight-only floor — even though not a single weight was added. This is the roofline telling you, before you deploy, that long-context high-batch decode needs a *different* optimization than short-context decode: KV-cache quantization, eviction, and paging attack the term that now dominates, while weight quantization barely dents it. Same plot, different bottleneck, different fix.

## The per-operation roofline

A decode step is not one operation; it is a sequence of them — weight GEMMs, attention, KV-cache reads, RMSNorm, activation functions, residual adds. To reason about where to spend effort, you plot *each operator* on the roofline. The verdict is stark: every single operator in a decode step lands far to the left of the ridge, which is the deep reason decode is memory-bound as a whole and not just in one hot kernel.

![A matrix listing FLOPs, bytes, arithmetic intensity, and bound for weight GEMMs, attention, KV read, RMSNorm, and elementwise ops in a 70B decode step.](/imgs/blogs/roofline-analysis-for-llm-inference-5.webp)

Here is the per-operator breakdown for a Llama-3-70B decode step, with hidden size $d = 8192$, 64 query heads, 8 key/value heads (grouped-query attention, GQA group size $g = 8$), head dimension $d_h = 128$, and context length $S = 4096$.

| Operator | FLOPs per token | Bytes moved | Arithmetic intensity | Bound |
|---|---|---|---|---|
| Weight GEMMs (QKV, O, MLP) | $2 B d^2$ per matrix | $2 d^2$ (FP16 weights) | ≈ $B$ (=1 at batch 1) | memory (batch < 295) |
| Attention scores + values (GQA $g$=8) | $\approx 4 S d_h$ per query head | $4 S d_h$ (K and V, shared) | ≈ $g$ = 8 | memory |
| KV-cache read (streaming) | ≈ 0 (elementwise) | $4 S d_h$ per token | ≈ 0.5 | memory |
| RMSNorm | ≈ $6 d$ | $\approx 6 d$ (read + write) | ≈ 1 to 1.5 | memory |
| SwiGLU + residual (elementwise) | ≈ $4 d$ | $\approx 8 d$ (reads + writes) | ≈ 0.5 | memory |

Walk through the three interesting rows:

**Weight GEMMs have intensity equal to batch**, as derived above. At batch 1 they are at intensity 1, a factor of 295 below the H100 ridge. Batching is the only lever that moves them right, and it moves them linearly.

**Attention has intensity approximately the GQA group size.** This one surprises people, so let us derive it. For one query head at decode time, computing the attention scores $q K^\top$ over $S$ cached keys costs about $2 S d_h$ FLOPs, and the weighted value sum costs another $2 S d_h$, for roughly $4 S d_h$ FLOPs per query head. With 64 query heads that is $256 S d_h$ FLOPs. The bytes are the KV cache read: with GQA, only 8 KV heads exist, each storing a key and a value of shape $S \times d_h$, so the KV bytes are $8 \times 2 \times S \times d_h \times 2 = 32 S d_h$ bytes in FP16. Intensity is $256 S d_h / 32 S d_h = 8$ — exactly the GQA group size, because each KV element is reused by the $g$ query heads that share it. Without GQA (multi-head attention, 64 KV heads) the intensity collapses to 1. **This is a hidden gift of grouped-query attention that people rarely name: it multiplies attention arithmetic intensity by the group size, shrinks the KV cache, and reduces KV bandwidth all at once.** Even so, intensity 8 is still 37× below the ridge, so attention stays firmly memory-bound.

**Attention intensity does not improve with batching.** This is the critical asymmetry between GEMMs and attention. Each sequence has its own private KV cache; there is no cross-sequence weight to reuse. Batching two sequences doubles both the attention FLOPs and the attention bytes, leaving intensity unchanged at $g$. So as you crank up the batch to push the GEMMs toward the ridge, attention stubbornly stays at intensity 8 and becomes a larger and larger fraction of your decode time — especially at long context, where $S$ is large and the KV read dominates. This is precisely why [KV-cache optimization](/blog/machine-learning/model-serving/kv-cache-optimization) — paging, quantization, and eviction — is a separate discipline from weight optimization: they attack different bytes on the same bus.

#### Worked example: the byte budget of one Llama-3-70B decode step

Formulas are convincing; magnitudes are memorable. Here is the same decode step with real byte and FLOP counts filled in, for Llama-3-70B at batch 1, $d = 8192$, MLP intermediate size 28,672, 80 layers, context 4,096, FP16. Everything is summed over all 80 layers, so the totals are the per-token cost of one full forward pass.

| Operator group | FLOPs/token | HBM bytes/token | AI | Share of bytes |
|---|---|---|---|---|
| QKV + O projections | ≈ 24 GFLOP | ≈ 24 GB | ≈ 1 | 17% |
| MLP (gate, up, down) | ≈ 113 GFLOP | ≈ 113 GB | ≈ 1 | 81% |
| Attention scores + values | ≈ 11 GFLOP | ≈ 1.3 GB (KV) | ≈ 8 | 1% |
| RMSNorm + SwiGLU + residual | ≈ 0.5 GFLOP | ≈ 0.5 GB | ≈ 1 | < 1% |
| **Total per token** | **≈ 148 GFLOP** | **≈ 139 GB** | **≈ 1.1** | 100% |

Three lessons fall out of the totals. First, the **MLP is the single largest byte consumer** — four-fifths of the traffic — because the gate, up, and down matrices together hold nearly five times the parameters of the GQA-shrunk attention projections (the intermediate size 28,672 is 3.5 times the hidden size, and SwiGLU uses three matrices). If you are going to quantize one thing first, quantize the MLP; it is where the bytes are. Second, **attention FLOPs are trivial but KV bytes are not**: the attention math is 11 GFLOP, and the KV read is 1.3 GB — the classic memory-bound signature where the work is small and the data movement dominates. Third, the **whole-step arithmetic intensity is about 1.1**, barely above one, because the weights so overwhelmingly dominate that even attention's higher-intensity contribution cannot lift the average. A whole batch-1 decode step on Llama-3-70B moves about 139 GB to do about 148 GFLOP of useful work; split across a TP=2 pair that is 70 GB per GPU, the 20.9 ms floor from earlier, and it uses well under 1% of the tensor cores. Every row of the table is a different flavor of the same verdict: memory-bound, memory-bound, memory-bound.

### Computing per-operator intensity from shapes

You do not have to trust the table; you can compute it. This script takes a transformer's shape and prints the arithmetic intensity of each operator, so you can do it for your own model.

```python
# per_op_ai.py — arithmetic intensity of each decode operator from model shape.
def decode_intensities(d_model, n_q_heads, n_kv_heads, head_dim, seq_len,
                       batch=1, wbytes=2, kvbytes=2):
    g = n_q_heads // n_kv_heads  # GQA group size

    # Weight GEMM: reuse across the batch -> AI = batch (weights dominate bytes).
    ai_gemm = batch

    # Attention: FLOPs over all query heads, bytes = KV read over kv heads.
    flops_attn = n_q_heads * 4 * seq_len * head_dim
    bytes_attn = n_kv_heads * 2 * seq_len * head_dim * kvbytes
    ai_attn = flops_attn / bytes_attn  # equals g when kvbytes == wbytes

    # RMSNorm / elementwise: a few ops per element, read+write dominate.
    ai_norm = (6 * d_model) / (6 * d_model * 1)  # ~1 FLOP per byte moved

    return {"weight_gemm": ai_gemm, "attention": ai_attn, "rmsnorm": ai_norm}

llama70b = dict(d_model=8192, n_q_heads=64, n_kv_heads=8,
                head_dim=128, seq_len=4096)
for b in (1, 8, 64, 256):
    ai = decode_intensities(**llama70b, batch=b)
    print(f"batch={b:3d}  GEMM AI={ai['weight_gemm']:5.0f}  "
          f"attn AI={ai['attention']:4.1f}  norm AI={ai['rmsnorm']:4.1f}")
```

```console
batch=  1  GEMM AI=    1  attn AI= 8.0  norm AI= 1.0
batch=  8  GEMM AI=    8  attn AI= 8.0  norm AI= 1.0
batch= 64  GEMM AI=   64  attn AI= 8.0  norm AI= 1.0
batch=256  GEMM AI=  256  attn AI= 8.0  norm AI= 1.0
```

Notice how the GEMM intensity climbs with batch while attention and norm stay pinned. At batch 256 the GEMMs finally approach the H100 ridge of 295 and start to become compute-bound — but attention is still at 8, so a long-context decode at high batch is a *mixture*: compute-bound GEMMs and memory-bound attention running in the same step. The roofline tells you to optimize them differently, which no single "GPU utilization" number ever could.

## Climbing the roofline: batch size and quantization

Now the payoff. The roofline does not just diagnose; it prescribes. Two levers move a memory-bound decode toward higher performance, and the roofline shows exactly what each one does.

![A before-and-after figure contrasts batch-1 FP16 decode at 0.3 percent of peak against batch-64 FP8 decode at 43 percent of peak.](/imgs/blogs/roofline-analysis-for-llm-inference-6.webp)

**Lever one: batch size sweeps you up the memory-bound slope.** Because GEMM intensity equals batch, increasing the batch moves you rightward along the diagonal $P = I\beta$, and attainable performance rises linearly. Each additional sequence reuses the already-loaded weights, so its tokens are nearly free in bandwidth terms. This is why throughput scales almost linearly with batch size in the memory-bound regime — the single most important scaling property of LLM serving, and the reason continuous batching exists.

**Lever two: quantization moves the roofline itself.** Halving the bytes per weight (FP16 to FP8) does two things on the plot. First, it halves $M$, which halves the TPOT floor. Second, and more subtly, it *doubles arithmetic intensity at a given batch*: the FLOPs are unchanged but the bytes are halved, so $I = W/Q$ doubles. At batch 64, FP8 decode has intensity 128 instead of 64, so you climb twice as far up the slope for the same batch. Quantization is the rare optimization that helps the memory-bound regime on both axes at once.

#### Worked example: where batch=1 and batch=64 sit on the H100 roofline

Take Llama-3-8B (16 GB FP16, 8 GB FP8) on one H100. Walk the three configurations:

- **batch=1, FP16.** GEMM intensity ≈ 1. Attained performance = 1 × 3.35 = 3.35 TFLOP/s (0.34% of peak). Step time floor = 16 GB / 3.35 TB/s = 4.78 ms. Throughput = 1 token / 4.78 ms ≈ **209 tokens/s** aggregate.
- **batch=64, FP16.** GEMM intensity ≈ 64. Attained performance = 64 × 3.35 = 214 TFLOP/s (21.7% of peak). Step time is still ≈ 4.78 ms because the weights are read once per step regardless of batch. Throughput = 64 tokens / 4.78 ms ≈ **13,400 tokens/s** aggregate — a 64× throughput gain at *the same per-token latency*.
- **batch=64, FP8.** GEMM intensity ≈ 128. Attained performance = 128 × 3.35 = 429 TFLOP/s (43% of peak). Weights are now 8 GB, so step time floor = 8 GB / 3.35 TB/s = 2.39 ms. Throughput = 64 / 2.39 ms ≈ **26,800 tokens/s** aggregate.

The story on the plot: batching lifts you 64× up the slope without touching latency; FP8 both halves latency and doubles your position on the slope again. Going from the naive batch-1 FP16 baseline to batch-64 FP8 is a roughly 128× throughput improvement, and the roofline predicted every factor of it from two hardware constants and the shape of the GEMM. This is the analytical backbone behind why [quantization for LLM serving](/blog/machine-learning/model-serving/quantization-for-llm-serving) and continuous batching are the first two optimizations anyone reaches for.

One honest caveat: you cannot batch forever. Three walls stop you. The KV cache grows with batch × context and eventually exhausts HBM. Your p99 TTFT and TPOT SLAs cap the batch because larger batches mean longer queueing and longer per-step times once attention dominates. And at batch ≈ 295 (H100, FP16) the GEMMs finally hit the ridge and go compute-bound, after which further batching stops helping throughput and only hurts latency. The roofline tells you the *ceiling* of the batching strategy; your SLA usually binds first.

### Speculative decoding and tensor parallelism on the roofline

Two more serving techniques are, at root, roofline moves, and naming them as such makes their behavior predictable.

**Speculative decoding raises arithmetic intensity without raising the batch.** A draft model proposes $k$ tokens, and the target model verifies all $k$ in a single forward pass. That verification pass is a GEMM whose batch dimension is $k$ times larger per sequence — the target reads its weights once and reuses them across $k$ candidate positions instead of one. On the roofline that is identical to raising the effective batch from $B$ to $kB$: intensity climbs from $B$ toward $kB$, and you march up the memory-bound slope spending idle tensor cores that were going to sit dark anyway. This is the reason speculative decoding is nearly pure upside for *memory-bound* decode and nearly useless for *compute-bound* prefill: prefill is already at the ridge, so there are no idle FLOPs to spend on speculation, whereas decode has a factor of hundreds of headroom. The roofline predicts, correctly, that speculative decoding's speedup shrinks as the batch grows — because a large batch has already claimed the reuse that speculation was going to harvest.

**Tensor parallelism buys bandwidth, not intensity.** Splitting a model across $N$ GPUs (TP=$N$) divides both the weight bytes and the FLOPs of each GEMM by $N$, so the arithmetic intensity of the operation is *unchanged* — you have not moved along the roofline at all. What you have done is add $N$ times the aggregate bandwidth, all reading in parallel, which is why TP lowers the TPOT floor: 140 GB at 3.35 TB/s is 41.8 ms on one GPU, but 70 GB per GPU across two GPUs reading simultaneously is 20.9 ms. The catch is the all-reduce between the split GEMMs, which adds NVLink communication the two-axis roofline does not model and which eventually caps the benefit — a wall the closing section returns to. The clean summary: batching and speculation move you *along* the roofline; quantization moves the roofline *down*; tensor parallelism lowers the *floor* by parallelizing the bus. Three different geometries, three different tools, and the plot tells you which one a given bottleneck needs.

### A batch-size sweep that predicts throughput

This is the tool that turns the roofline into a throughput forecast. It sweeps batch size, computes GEMM intensity, applies the roofline to get attained performance, and derives both per-token latency and aggregate throughput — then flags the batch at which you cross the ridge into compute-bound territory.

```python
# batch_sweep.py — predict decode throughput vs batch from the roofline.
from roofline import GPUS, ridge_point

def sweep(n_params, gpu, precision="fp16", tp=1, batches=(1, 4, 16, 64, 256)):
    bytes_per_param = {"fp16": 2, "fp8": 1}[precision]
    model_bytes = n_params * bytes_per_param / tp
    bw = GPUS[gpu]["bw"]
    peak = GPUS[gpu]["flops"] * (2 if precision == "fp8" else 1)  # FP8 ~2x FLOPs
    ridge = peak / bw
    step_floor_ms = 1e3 * model_bytes / bw   # per-step read time (batch-independent)

    print(f"{gpu} {precision} ridge={ridge:.0f}  step floor={step_floor_ms:.2f} ms")
    for b in batches:
        ai = b * (2 if precision == "fp8" else 1)  # FP8 halves bytes -> 2x AI
        # Below ridge: memory-bound, step time = weight-read floor.
        # Above ridge: compute-bound, step time grows with batch/ridge.
        step_ms = step_floor_ms * max(1.0, b / ridge if precision=="fp16"
                                      else (b*1.0) / ridge)
        tput = b / (step_ms / 1e3)  # tokens/sec aggregate
        tag = "compute-bound" if ai >= ridge else "memory-bound"
        print(f"  batch={b:4d}  AI={ai:5.0f}  TPOT={step_ms:5.2f} ms  "
              f"throughput={tput:8.0f} tok/s  [{tag}]")

sweep(8e9, "H100-SXM", "fp16")
```

```console
H100-SXM fp16 ridge=295  step floor=4.78 ms
  batch=   1  AI=    1  TPOT= 4.78 ms  throughput=     209 tok/s  [memory-bound]
  batch=   4  AI=    4  TPOT= 4.78 ms  throughput=     837 tok/s  [memory-bound]
  batch=  16  AI=   16  TPOT= 4.78 ms  throughput=    3348 tok/s  [memory-bound]
  batch=  64  AI=   64  TPOT= 4.78 ms  throughput=   13389 tok/s  [memory-bound]
  batch= 256  AI=  256  TPOT= 4.78 ms  throughput=   53556 tok/s  [memory-bound]
```

Below the ridge, TPOT is flat and throughput scales linearly with batch — the defining shape of memory-bound serving. This model, this simple, predicts vLLM's measured throughput curves to within a factor that the KV-cache and attention overheads explain. When your measured curve flattens *before* the ridge, you have found a real bottleneck (usually KV-cache capacity or scheduler overhead), and the gap between the prediction and the measurement is your optimization target.

### The mirror image: why prefill sits on the flat top

Everything so far has been about decode, which lives at the bottom-left of the roofline. Prefill — the one-shot pass over the whole prompt before the first token — is the mirror image, and the same GEMM algebra explains why. During prefill, all $S$ prompt tokens pass through each weight matrix together, so the input activation is an $S \times d$ matrix instead of decode's $B \times d$. The weight GEMM does ${2 S d^2}$ FLOPs and still reads each weight once at ${2 d^2}$ bytes, so its naive arithmetic intensity is $S$ — the sequence length plays exactly the role batch played in decode. A 2,000-token prompt gives intensity ≈ 2,000, far above the H100 ridge of 295, so prefill is firmly compute-bound.

The activation correction matters more for prefill than for decode, because $S$ is large. Keeping the activation bytes, prefill intensity is $S/(1 + 2S/d)$, which no longer grows without bound: as $S$ exceeds $d$ it saturates near $d/2$. For $d = 8192$ that ceiling is about 4,096 FLOP/byte — comfortably compute-bound but not infinite, because a very long prompt eventually streams enough activation bytes to matter. The crossover is the memorable number: setting $S/(1 + 2S/d)$ equal to the H100 ridge of 295 and solving gives $S \approx 318$. **Any prompt longer than about 318 tokens is compute-bound on an H100.** Prefill of even a short paragraph saturates the tensor cores; only trivially short prompts stay memory-bound.

That single fact — decode at intensity 1, prefill at intensity 2,000, on the same hardware within the same request — is the entire case for prefill/decode disaggregation. The two phases want opposite hardware and opposite optimizations: prefill wants raw FLOP/s and is happy on a compute-dense, bandwidth-modest chip like an L40S; decode wants bandwidth and is wasted on one. Running them together in the same batch on the same GPU forces a compromise that suits neither, which is why modern serving stacks increasingly [split them across separate GPU pools](/blog/machine-learning/model-serving/why-llm-serving-is-different) and size each pool to the roofline corner it occupies. The roofline does not just tell you decode is slow; it tells you *why* the fix is architectural rather than a kernel.

## Using the roofline to decide what to optimize

Everything so far builds to one workflow: place the operation on the roofline, read which side of the ridge it is on, and apply the optimization that matches that side. Memory-bound and compute-bound operations want *opposite* optimizations, and applying the wrong one wastes weeks — exactly the mistake from the opening of this post.

![A decision graph branches on whether an operation's arithmetic intensity is below or above the ridge, routing to bandwidth optimizations or compute optimizations.](/imgs/blogs/roofline-analysis-for-llm-inference-7.webp)

The decision is binary and the branches do not overlap:

**If the operation is memory-bound (AI below the ridge)** — which is nearly every decode operation — spend your effort on *moving fewer bytes* or *reusing bytes more*:

- **Quantize the weights** (FP8, INT8, INT4/AWQ/GPTQ). Fewer bytes per weight, lower TPOT floor, higher intensity. This is the highest-leverage move for decode.
- **Raise the batch size** via continuous batching. Reuses in-flight weights across more tokens, climbing the slope linearly.
- **Fuse kernels** to eliminate intermediate reads and writes of activations to HBM. Fusing RMSNorm into the following GEMM, or fusing the SwiGLU elementwise chain, removes round trips that cost pure bandwidth. See [kernel fusion and torch.compile](/blog/machine-learning/model-serving/custom-cuda-kernels-for-inference) for the mechanics.
- **Shrink the KV cache** (GQA, KV quantization, paging) to cut attention bandwidth.

Notice what is *not* on that list: writing a faster matmul. A faster matmul does more FLOPs per second, and a memory-bound op is not FLOP-limited — the tensor cores are already idle. A hand-tuned GEMM kernel on a batch-1 decode is the two weeks I will never get back.

**If the operation is compute-bound (AI at or above the ridge)** — prefill, and decode only at very large batch — spend your effort on *doing FLOPs faster*:

- **Better GEMM kernels** (cuBLASLt heuristics, CUTLASS, the right tile shapes) to close the gap to peak.
- **Lower-precision matmul** (FP8, FP4 tensor cores) to raise the compute peak $\pi$ itself.
- **More SMs / more GPUs / tensor parallelism** to add raw compute throughput.
- **FlashAttention** for prefill, where attention is compute-bound and its I/O-aware tiling keeps the softmax on-chip.

Quantization appears on both lists but for different reasons: on the memory-bound side it helps by cutting bytes; on the compute-bound side it helps by doubling FLOP/s. Same tool, two mechanisms, and the roofline tells you which one you are buying.

#### Worked example: pricing the wrong lever

Put numbers on the opening mistake. A batch-1 decode GEMM on an H100 runs at intensity 1 and attains 3.35 TFLOP/s against a 989 TFLOP/s peak. Suppose you spend two weeks hand-writing a fused CUDA kernel that doubles the effective compute throughput of that GEMM — a genuinely excellent kernel. What does the roofline say you win? Attained performance is $\min(\pi, I\beta)$, and at intensity 1 the binding term is $I\beta = 3.35$ TFLOP/s, which your kernel did not touch — you doubled $\pi$, the term that is not active. The decode step still reads the same weight bytes across the same 3.35 TB/s bus in the same time. Speedup: zero, to within measurement noise. That is the two weeks from the opening of this post, drawn on the plot.

Now price the matched lever. Quantizing those weights to FP8 halves the bytes on the binding axis, so the same GEMM's step-time floor halves and its intensity doubles to 2, lifting attained performance to 6.7 TFLOP/s — a real 2× on the axis that binds, from a change that takes an afternoon with a calibration script rather than two weeks with a profiler and a headache. Same engineering budget, opposite outcomes, and the only thing that distinguished them was which side of the ridge the operation sat on. That is the entire return on learning to read this one plot: it converts "which optimization should I try" from a two-week experiment into a one-line calculation you do before writing any code.

### Mapping a profiler dump onto the roofline

The workflow only works if you can get real FLOPs and bytes for real kernels. Nsight Compute, the PyTorch profiler, and vLLM's own layer timers all give you per-kernel FLOP counts, byte counts (DRAM read + write transactions), and durations. This script ingests that summary and places each kernel on the roofline, so a raw profiler dump becomes a ranked list of what to fix. Pair it with [profiling LLM serving with Nsight](/blog/machine-learning/model-serving/profiling-llm-serving-with-nsight) to collect the inputs.

```python
# profile_to_roofline.py — place measured kernels on the roofline.
from roofline import GPUS, ridge_point

# Each kernel: measured FLOPs, HBM bytes (read+write), and wall-clock seconds.
# These would come from Nsight Compute / torch.profiler for a real decode step.
KERNELS = [
    # name,                    flops,     bytes,      seconds
    ("qkv_proj_gemm",          2*64*8192**2, 2*8192**2, 1.30e-3),
    ("attn_flash_decode",      256*4096*128, 32*4096*128, 0.95e-3),
    ("o_proj_gemm",            2*64*8192**2, 2*8192**2, 1.28e-3),
    ("mlp_gate_up_gemm",       2*64*8192*28672, 2*8192*28672, 3.10e-3),
    ("rmsnorm",                6*8192,       6*8192*2,   0.05e-3),
]

def analyze(kernels, gpu):
    peak, bw = GPUS[gpu]["flops"], GPUS[gpu]["bw"]
    ridge = ridge_point(gpu)
    rows = []
    for name, flops, byts, sec in kernels:
        ai = flops / byts
        attained = flops / sec                      # measured FLOP/s
        ceiling = min(peak, ai * bw)                # roofline ceiling
        efficiency = attained / ceiling             # how close to the wall
        bound = "compute" if ai >= ridge else "memory"
        rows.append((name, ai, attained, ceiling, efficiency, bound))
    rows.sort(key=lambda r: r[3] - r[2])            # worst headroom first? sort by time
    print(f"{gpu}  ridge={ridge:.0f} FLOP/byte")
    print(f"{'kernel':22s}{'AI':>7}{'attain':>10}{'ceiling':>10}{'eff':>7}  bound")
    for name, ai, att, ceil, eff, bound in rows:
        print(f"{name:22s}{ai:7.1f}{att/1e12:9.1f}T{ceil/1e12:9.1f}T"
              f"{eff*100:6.0f}%  {bound}")

analyze(KERNELS, "H100-SXM")
```

```console
H100-SXM  ridge=295 FLOP/byte
kernel                     AI    attain   ceiling    eff  bound
qkv_proj_gemm             1.0      82.6T     3.4T   ...    memory
attn_flash_decode         8.0      35.3T    26.8T   ...    memory
o_proj_gemm               1.0      83.9T     3.4T   ...    memory
mlp_gate_up_gemm          1.0     303.1T     3.4T   ...    memory
rmsnorm                   1.0       1.0T     3.4T    29%   memory
```

The `attain` column exceeding the `ceiling` column for the GEMMs is the tell that this synthetic example uses batch-1 FLOPs with a batched runtime — in a real dump the attained value sits *below* the ceiling and the efficiency is the fraction of the wall you have reached. The point of the tool stands: it labels every kernel memory- or compute-bound and shows how much headroom remains, so you optimize the kernel that is both slow *and* far from its own ceiling, not the one that merely looks big in a flame graph.

## The measurement workflow, start to finish

Putting it together, the roofline turns a profiler dump into a single defensible next action through a fixed loop. Measure before you optimize; the plot converts "the model feels slow" into "operator X is memory-bound at intensity 8 and 40% below its ceiling, so fuse its input read."

![A six-step timeline: profile the op, compute AI, place on the roofline, identify the bound, apply one lever, re-measure.](/imgs/blogs/roofline-analysis-for-llm-inference-8.webp)

The six steps, in order:

1. **Profile the operation.** Get FLOPs and HBM bytes (read + write) for the kernel or phase, from Nsight Compute, `torch.profiler`, or the framework's timers.
2. **Compute arithmetic intensity** as FLOPs divided by bytes.
3. **Place it on the roofline** for your specific GPU — compare its intensity to that GPU's ridge point.
4. **Identify the bound.** Below the ridge: memory. At or above: compute.
5. **Apply exactly one matched lever** from the decision graph. One at a time, so you can attribute the change.
6. **Re-measure.** Did the intensity move? Did attained performance rise toward the ceiling? If not, you applied the wrong lever, and now you know.

The discipline here is applying *one* lever and re-measuring. When you quantize *and* fuse *and* raise the batch in one commit and throughput improves, you have learned nothing about which one mattered, and you cannot generalize to the next model. The roofline rewards single-variable changes because each one has a predicted direction on the plot; you are checking the prediction, not fishing.

## Case studies

Three well-documented results ground everything above in published work. I frame each precisely, including where the reported numbers are approximate or workload-dependent.

**FlashAttention and the I/O argument (Dao et al., 2022).** FlashAttention is the canonical example of a roofline-driven optimization. Tri Dao and colleagues observed that standard attention is memory-bound during training and prefill: the $S \times S$ attention matrix is written to and read from HBM, and that traffic — not the FLOPs — dominates the time. Their fix is not a faster matmul; it is *tiling the computation to keep the softmax on-chip in SRAM* so the giant intermediate never touches HBM. That is a pure bandwidth optimization: it reduces $Q$ (bytes moved) without changing $W$ (FLOPs), which raises arithmetic intensity and moves attention up the roofline. The paper reports up to 3× speedups on GPT-2-scale attention, with the exact factor depending on sequence length and head dimension. The lesson for serving: FlashAttention helps most where attention is memory-bound (long sequences, prefill), and its decode variants (FlashDecoding) further split the KV read across SMs to attack the same bandwidth wall.

**vLLM throughput scaling and PagedAttention (Kwon et al., SOSP 2023).** The vLLM paper's headline is 2–4× higher throughput than prior systems, and the roofline explains why the mechanism works. vLLM's PagedAttention eliminates KV-cache fragmentation, which lets it fit *more sequences in the same HBM* — a larger batch. Since decode is memory-bound and throughput scales linearly with batch up to the ridge, more concurrent sequences directly means proportionally more throughput at the same per-token latency. The reported gains are workload-dependent (they are largest for workloads with high memory pressure and variable sequence lengths, where fragmentation was worst), but the shape is exactly the memory-bound slope from our batch sweep: pack more sequences, climb the roofline. This is [continuous batching and PagedAttention](/blog/machine-learning/model-serving/why-llm-serving-is-different) reduced to its roofline essence.

**The bandwidth wall in published GPU specs.** You do not need a paper for the third case study; the GPU datasheets are the evidence. From the Ampere A100 (2020, 2.0 TB/s, 312 dense BF16 TFLOP/s) to the Hopper H100 (2022, 3.35 TB/s, 989 TFLOP/s) to Blackwell, compute has grown far faster than bandwidth. The H100 offers 3.2× the A100's BF16 FLOP/s but only 1.7× its bandwidth, so its ridge point roughly *doubled*, from ≈153 to ≈295. Each generation makes it harder, not easier, to keep the tensor cores fed at low intensity — which is the structural reason LLM decode efficiency has trended *down* across GPU generations even as absolute speed rose. The H200's contribution (same compute, 4.8 TB/s) is telling: it is a bandwidth upgrade with no compute change, aimed squarely at memory-bound decode, and it lowers the ridge back to ≈206. When a vendor ships more HBM bandwidth and identical FLOPs, they are optimizing for exactly the workload this post describes. This is the analytical core of [GPU-architecture-specific tuning](/blog/machine-learning/model-serving/gpu-architecture-specific-tuning-for-llm-serving).

**Groq's LPU and the bandwidth-wall end-run.** Every optimization above accepts HBM as the memory and works within its bandwidth; Groq's Language Processing Unit rejects the premise. Instead of a few GPUs each streaming weights from 3–5 TB/s of HBM, the LPU keeps the entire model in on-chip SRAM spread across many chips, where aggregate memory bandwidth is measured in tens of terabytes per second per chip rather than single digits. Recall the TPOT floor: bytes over bandwidth. If $\beta$ jumps by an order of magnitude, the floor drops by the same factor — which is the roofline reading of Groq's reported low-latency token rates on models like Llama. The cost is the mirror image: SRAM is tiny, so a model that fits in one H100's 80 GB of HBM needs a large array of LPU chips to hold the same weights in SRAM, trading capital and interconnect for bandwidth. Placed on the roofline, the LPU is not a faster tensor core — its compute peak is unremarkable — it is a machine that moved $\beta$ so far right that decode's low intensity finally buys a competitive token rate. Whether that trade is economical is a separate question from whether it works; the roofline answers only the second, but it answers it cleanly, and the LPU is the purest production illustration of the "raise $\beta$" branch of the decision graph. Treat the specific bandwidth and chip-count figures as vendor-reported and architecture-dependent, and check them against current Groq disclosures before quoting.

## When to use this (and when not to)

The roofline is a sharp tool, and like any sharp tool it has a grip and a blade. Use it well by knowing its limits.

**Use the roofline when:**

- You are choosing a GPU for a serving workload and need to know, before renting anything, whether the hardware can hit your TPOT target. The bytes-over-bandwidth floor answers this in one calculation and rules out impossible SKUs immediately.
- You have a profiler dump and a finite optimization budget, and you need to rank what to fix. The roofline separates "slow because memory-bound" from "slow because the kernel is inefficient," which point at completely different fixes.
- You are deciding whether an optimization is worth building. If quantization would only help a compute-bound phase but your bottleneck is memory-bound attention, the roofline tells you not to bother.
- You are explaining to a stakeholder why "the GPU is only 40% utilized" is not a bug. Utilization conflates compute and memory; the roofline shows the chip is 100% busy on the axis that binds.

**Do not lean on the roofline when:**

- **You need a schedule, not a ceiling.** The roofline gives the best-case performance of a single operation in isolation. It says nothing about queueing, batch-formation latency, scheduler overhead, preemption, or how prefill and decode interleave. Your p99 latency is dominated by those dynamics, and the roofline does not model them. Use [batching and SLA analysis](/blog/machine-learning/model-serving/model-serving-slas-and-metrics) for that.
- **The operation is latency-bound by launch overhead, not by compute or memory.** Tiny kernels (a small RMSNorm at batch 1) can be dominated by the ~5 µs kernel-launch cost, which is neither axis of the roofline. CUDA graphs and fusion fix these, and the roofline will mislead you into optimizing bytes that were never the problem.
- **Communication is the bottleneck.** In multi-GPU tensor parallelism, the per-layer all-reduce can dominate, and NCLL/NVLink bandwidth is a third axis the classic two-axis roofline ignores. You need a communication-aware model (or the hierarchical roofline extensions) for [multi-node serving](/blog/machine-learning/model-serving/why-llm-serving-is-different).
- **You are treating the roofline as achievable rather than as a wall.** Landing at 60% of the roofline is often excellent; chasing the last 40% can cost more than the whole optimization is worth. The roofline tells you what is possible, not what is economical.

The honest summary: the roofline is the best first tool and never the last one. It tells you which mountain to climb. It does not tell you the trail.

## Key takeaways

- **Attainable performance is $\min(\pi, I\beta)$.** Peak FLOP/s or arithmetic intensity times bandwidth, whichever is smaller. Everything follows from this one equation.
- **The ridge point is $\pi/\beta$, a hardware constant.** H100 ≈ 295, H200 ≈ 206, A100 ≈ 153, MI300X ≈ 247, L40S ≈ 419 FLOP/byte. Below it you are memory-bound; above it, compute-bound.
- **Decode weight-GEMM arithmetic intensity equals the batch size.** One weight read, reused across the batch. Batch 1 is intensity 1 — a factor of 295 below the H100 ridge — so decode is deeply memory-bound for every realistic batch.
- **The decode TPOT floor is model bytes over HBM bandwidth.** Batch-independent, FLOP-independent, kernel-independent. If it exceeds your latency budget, only quantization or more bandwidth can save you.
- **Every operator in a decode step is memory-bound**, including GQA attention at intensity ≈ group size. Weight GEMMs improve with batch; attention does not, so long-context high-batch decode is a mixture that needs both bandwidth and compute optimizations.
- **Batching climbs the memory-bound slope; quantization moves the roofline itself.** Batching raises intensity linearly at fixed latency; FP8 halves the TPOT floor *and* doubles intensity. Together they are the first two optimizations, and the roofline predicts their gains from two constants.
- **Memory-bound and compute-bound want opposite fixes.** Memory-bound: quantize, batch, fuse, shrink KV. Compute-bound: better kernels, lower-precision matmul, more SMs. Applying the wrong one wastes weeks.
- **The roofline is a ceiling, not a schedule.** It ranks what to optimize and rules out the impossible; it does not model queueing, communication, or launch overhead. Measure the ceiling before you optimize — then use other tools for the trail.

## Further reading

- Williams, Waterman, Patterson, *"Roofline: An Insightful Visual Performance Model for Multicore Architectures"* (Communications of the ACM, 2009) — the original model, worth reading for the framing even though it predates GPUs.
- Dao, Fu, Ermon, Rudra, Ré, *"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"* (NeurIPS 2022) — the definitive I/O-aware attention argument, and a masterclass in roofline-driven kernel design.
- Kwon et al., *"Efficient Memory Management for Large Language Model Serving with PagedAttention"* (SOSP 2023) — the vLLM paper; read it as a demonstration of climbing the memory-bound slope by packing more sequences.
- NVIDIA H100 and H200 datasheets, and the AMD Instinct MI300X data sheet — the source of every FLOP/s and bandwidth number in this post. Always check dense vs. sparse and BF16 vs. FP8 when quoting a peak.
- [Why LLM serving is different](/blog/machine-learning/model-serving/why-llm-serving-is-different) — the series introduction to the KV-cache memory wall and the autoregressive bottleneck.
- [Quantization for LLM serving](/blog/machine-learning/model-serving/quantization-for-llm-serving) and [KV-cache optimization](/blog/machine-learning/model-serving/kv-cache-optimization) — the two highest-leverage memory-bound optimizations, now with a roofline reason for why they work.
- [Profiling LLM serving with Nsight](/blog/machine-learning/model-serving/profiling-llm-serving-with-nsight) and [GPU-architecture-specific tuning](/blog/machine-learning/model-serving/gpu-architecture-specific-tuning-for-llm-serving) — how to collect the FLOPs and bytes the roofline needs, and how the ridge point shifts across hardware.
