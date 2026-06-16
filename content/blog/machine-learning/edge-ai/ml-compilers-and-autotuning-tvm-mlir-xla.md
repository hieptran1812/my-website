---
title: "ML compilers and autotuning: TVM, MLIR, XLA, and the schedule search"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Learn how ML compilers generate and autotune kernels for your exact op shapes on your exact chip — sometimes beating hand-written vendor libraries — and how to decide when that build-time cost is worth paying on the edge."
tags:
  [
    "edge-ai",
    "model-optimization",
    "ml-compilers",
    "tvm",
    "mlir",
    "xla",
    "autotuning",
    "inference",
    "efficient-ml",
    "triton",
  ]
category: "machine-learning"
subcategory: "Edge AI"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/ml-compilers-and-autotuning-tvm-mlir-xla-1.png"
---

A few years ago I was shipping a depthwise-separable backbone onto an ARM Mali GPU in a phone, and the single layer that dominated the latency was a perfectly ordinary 1x1 convolution at an unusual channel count. The vendor's hand-written GPU kernel library had a fast path for the channel counts the vendor cared about, and a slow, generic fallback for everything else — and our channel count landed squarely in the fallback. The fallback was about 3x slower than it had any right to be. I could not edit the vendor's closed kernel. I could not change my model's channel count without retraining and losing accuracy. I was stuck between a kernel I could not touch and a model I could not change.

What got me unstuck was not a faster runtime. It was a **compiler** — specifically TVM with its auto-scheduler. Instead of calling a pre-written kernel, the compiler *generated* a kernel for my exact op shape on my exact GPU, searched a few thousand candidate implementations overnight, measured the fast ones on the actual device, and produced a kernel that beat the vendor fallback by roughly 2.4x and landed within a hair of the vendor's hand-tuned fast path. No retraining. No hand-written assembly. Just a search over how the loops should be arranged, run once at build time, baked into a deployable module.

That is the difference this whole post is about. In [the inference runtimes comparison](/blog/machine-learning/edge-ai/inference-runtimes-compared) we treated the runtime as a library of pre-written kernels: ONNX Runtime, TensorRT, LiteRT, llama.cpp each ship hand-optimized kernels for the common ops on the hardware they support, and they are excellent when your op and your chip are on the supported list. An **ML compiler is a different beast**: it *generates and autotunes* kernels for your op shapes on your chip. Sometimes it merely matches the vendor library; sometimes it beats it; and for novel ops or weird hardware where no vendor kernel exists, it is the only thing that works at all.

![A before-and-after figure contrasting a naive algorithm-only matmul that thrashes the cache at about four gigaflops with the same algorithm plus a tuned schedule that tiles and vectorizes to reach roughly sixty gigaflops, a fifteen-times win.](/imgs/blogs/ml-compilers-and-autotuning-tvm-mlir-xla-1.png)

By the end of this post you will understand the one idea that makes all of this work — Halide's separation of *algorithm* from *schedule* — and you will be able to: read what TVM, MLIR, XLA, IREE, and Triton actually do and how they differ; reason about *why* searching the schedule space beats fixed heuristics; do the tiling-and-arithmetic-intensity math that explains the latency wins; write a TVM/Ansor autotune sketch, a `torch.compile` before/after, and a Triton kernel skeleton; and decide, on a real project, when a compiler earns its build-time cost and when a runtime's built-in kernels are the right call. This sits on the "compilers/runtimes" layer of the [four-lever Pareto frame](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression): quantization, pruning, distillation, and efficient architecture all change *what* you compute, but a compiler changes *how* it runs on the metal, and that last-mile "how" is frequently where the remaining 2-3x of edge latency is hiding.

## What a compiler does that a runtime does not

Let me draw the line sharply, because the words "runtime" and "compiler" get used loosely and the distinction is the whole game.

A **runtime** executes a model graph by dispatching each operator to a pre-written kernel. When ONNX Runtime hits a convolution, it calls into a kernel that some human (or some vendor's autotuning system that ran long ago) already wrote and optimized — oneDNN on x86, cuDNN on NVIDIA, XNNPACK on ARM. The runtime's job is graph management, memory planning, and dispatch. The kernels are fixed assets it ships with. This is fast to develop against, robust, and excellent *for the cases the kernel authors anticipated*. The catch is right there in that clause: the kernels are optimized for the shapes and hardware the authors anticipated.

A **compiler** does not ship the kernel. It ships the *ability to generate* a kernel. Given your operator and your target, it emits candidate loop nests, lowers them to machine code, and — in the autotuning compilers — measures them and searches for the fastest one. The output is a kernel specialized to *your* `(op, shape, dtype, target)` tuple. There is no generic fallback because there is no generic kernel; every kernel is bespoke.

Why does this matter so much on the edge specifically? Three reasons, and they compound.

First, **edge hardware is diverse and weird**. Cloud inference is dominated by a handful of NVIDIA GPUs with superb vendor libraries. The edge is a zoo: dozens of mobile SoCs, each with its own NPU and GPU and DSP; Jetsons; Coral Edge TPUs; Hexagon DSPs; RISC-V accelerators; bare Cortex-M microcontrollers. No vendor writes a hand-tuned kernel for every op on every one of these. For the long tail, a compiler that *generates* a kernel is not a nice-to-have, it is the only path.

Second, **edge ops are diverse in shape**. The four levers of compression deliberately produce *unusual* tensors: depthwise-separable convs with channel multipliers, grouped convs, pruned channel counts that are not nice powers of two, quantized int8 and int4 GEMMs, fused attention blocks. These are exactly the shapes that fall off the vendor's fast path. A compiler tuned for *your* shape does not have a fast path and a slow path; it has one path, tuned for you.

Third, **the edge is where the last 2x matters most**. In the cloud you can add a GPU. On a battery-powered device you cannot, and a 2x latency or energy win is the difference between shipping and not shipping. Compilers harvest exactly the last-mile gains that vendor libraries leave on the table for non-mainstream shapes.

Here is the honest counterweight, which I will keep returning to: a compiler is not free. Autotuning costs *build time* — minutes to many hours of measurement on the target device. The generated kernel can be brittle if your shapes change. And for a common op on supported hardware, the vendor library is often already at or near the hardware roofline, so a compiler's search converges to roughly the same number after burning an hour. The art is knowing which situation you are in, and the [decision tree](#when-to-reach-for-a-compiler-and-when-not-to) at the end of this post is the summary I actually use.

## The Halide insight: separate the algorithm from the schedule

Everything modern in this space — TVM, Halide, the loop-level dialects in MLIR, Triton's autotuner — descends from one idea published by Jonathan Ragan-Kelley and colleagues in the 2013 Halide paper. The idea is so clean it is worth stating in one sentence and then unpacking it for the rest of the section.

> Separate **what** you compute (the algorithm) from **how** you compute it (the schedule), so that one algorithm can be paired with many schedules, and you can search over schedules without ever risking the correctness of the result.

Consider a 3x3 blur, or a matrix multiply, or a convolution. The **algorithm** says, for each output element, what arithmetic produces it: this output pixel is the average of its nine neighbors; this output element of `C` is the dot product of a row of `A` and a column of `B`. The algorithm is a pure mathematical specification. It says nothing about loop order, nothing about memory, nothing about parallelism.

The **schedule** is every decision about *how* to actually run that specification on hardware:

- **Loop order** — do you iterate rows-then-columns or columns-then-rows? For a matmul, which of the three loops (i, j, k) is outermost?
- **Tiling (blocking)** — do you process the whole row at once, or break it into 64x64 tiles that fit in cache?
- **Vectorization** — does the inner loop emit scalar instructions or SIMD instructions that do 8 or 16 lanes at once?
- **Parallelization** — which loop is split across CPU cores or GPU thread blocks?
- **Caching / compute-vs-store** — do you recompute an intermediate, or compute it once and store it for reuse? Where in the loop nest does that storage live?
- **Unrolling** — do you unroll the inner loop to expose more instruction-level parallelism and amortize loop overhead?

The crucial, non-obvious fact is that **the schedule does not change the output**. Every legal schedule for a given algorithm produces bit-identical results (modulo floating-point reassociation, which good compilers control). So the schedule is *pure performance*: it cannot break correctness, only change speed. That is what makes searching over schedules safe. You are not searching over programs that might be wrong; you are searching over equivalent programs that differ only in latency.

And the latency differences are enormous. For a single matmul on a CPU, the gap between the naive schedule (default loop order, scalar, no tiling) and a well-tuned schedule (tiled to fit L1, vectorized, multithreaded, the k-loop unrolled) is routinely **10x to 30x** on the *exact same arithmetic*. The figure above shows this gap concretely: about 4 GFLOP/s naive versus roughly 60 GFLOP/s tuned, a 15x win, with not one floating-point operation changed. The numbers come from nothing but rearranging loops and choosing where data lives.

This is the deep reason ML compilers exist. If the schedule were a minor 5% knob, nobody would build a search engine for it. It is an order-of-magnitude knob, and the *optimal* schedule depends on the op, the exact shape, the dtype, the cache sizes, the SIMD width, the number of cores, and the memory bandwidth — none of which a single hand-written heuristic can possibly nail across the edge zoo. So instead of one expert hand-writing one schedule, you let a machine search the schedule space for *your* combination.

### Why a fixed heuristic cannot win across shapes

You might object: surely a smart engineer can write a good schedule that works well enough everywhere? In practice, no, and the reason is that the optimal tile size and loop order are *discontinuous* functions of the problem. A tile that fits L1 for a 256-channel layer overflows it for a 512-channel layer. A loop order that is bandwidth-optimal for a tall-skinny matmul is terrible for a short-fat one. A vectorization width that is perfect for fp32 is half-utilized for fp16. The vendor library handles this by shipping *many* hand-written kernels and dispatching among them by shape — which is exactly why it has a fast path and a slow fallback. The compiler handles it by *generating* the right kernel for the shape in front of it. One approach pre-pays human effort for a finite set of shapes; the other pays machine search time for whatever shape shows up.

## The science: tiling, arithmetic intensity, and the schedule's effect on the roofline

The single most important schedule transformation — the one that produces the biggest wins and is easiest to reason about — is **tiling for cache reuse**. To see *why* it works, we need the roofline vocabulary. If the roofline is new to you, read [the roofline post](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives) first; here I will use just enough of it to make the tiling math land.

Define **arithmetic intensity** $I$ as the ratio of useful arithmetic to bytes moved from main memory:

$$
I \;=\; \frac{\text{FLOPs}}{\text{Bytes moved from DRAM}} \quad\left[\frac{\text{FLOP}}{\text{byte}}\right].
$$

The roofline says attainable performance is $\min(P_{\text{peak}},\, I \cdot B_{\text{peak}})$, where $P_{\text{peak}}$ is peak compute and $B_{\text{peak}}$ is peak memory bandwidth. A kernel is memory-bound when $I < P_{\text{peak}} / B_{\text{peak}}$ (the *ridge point*) and compute-bound otherwise. The whole point of tiling is to **raise $I$** so the kernel crosses from the memory-bound side to the compute-bound side, where the fast compute units are actually the limit.

### Why naive matmul is memory-bound

Take $C = A B$ with $A$ of size $N \times N$ and $B$ of size $N \times N$, accumulating into $C$ of size $N \times N$. The total arithmetic is fixed: $2N^3$ FLOPs (one multiply and one add per inner-loop step, $N^3$ steps). The arithmetic is the same for *every* schedule — that is the algorithm. What changes is bytes moved.

The naive triple loop computes one output element fully before moving on:

```python
# Naive matmul: for each output C[i,j], stream a full row of A and column of B
for i in range(N):
    for j in range(N):
        acc = 0.0
        for k in range(N):
            acc += A[i, k] * B[k, j]   # B[:, j] is re-read for every i
        C[i, j] = acc
```

The killer is `B[k, j]`. The entire column `B[:, j]` is read from memory for every single `i`. Across the full computation, `B` is read $N$ times (once per row of `A`), and `A` is read $N$ times (once per column of `B`). With 4-byte floats, the bytes moved are on the order of:

$$
\text{Bytes}_{\text{naive}} \;\approx\; 4 \cdot (N \cdot N^2 + N \cdot N^2 + N^2) \;\approx\; 8 N^3 \text{ bytes}.
$$

So the arithmetic intensity of the naive schedule is

$$
I_{\text{naive}} \;=\; \frac{2 N^3}{8 N^3} \;=\; \frac{1}{4} \;\approx\; 0.25 \ \frac{\text{FLOP}}{\text{byte}}.
$$

A quarter of a FLOP per byte. On any modern chip whose ridge point sits at tens or hundreds of FLOP/byte, that is *deeply* memory-bound. The multiply-accumulate units sit nearly idle, waiting on DRAM. This is the "out of gas" situation — and it is why the naive schedule in the intro figure clocks ~4 GFLOP/s on a CPU capable of 50+.

### Why tiling raises arithmetic intensity

Now tile the loops. Choose a tile size $T$ (say 64) and partition each matrix into $T \times T$ blocks. Load one $T \times T$ block of `A` and one $T \times T$ block of `B` into cache, multiply them, accumulate into a $T \times T$ block of `C`, and *reuse those resident blocks for all $T^2$ output elements they touch* before evicting them.

```python
# Tiled matmul: a T-by-T block of A and B stays cache-resident and is reused T times
T = 64
for i0 in range(0, N, T):
    for j0 in range(0, N, T):
        # C-block accumulator stays in registers/L1
        for k0 in range(0, N, T):
            # load A[i0:i0+T, k0:k0+T] and B[k0:k0+T, j0:j0+T] once
            for i in range(i0, i0 + T):
                for j in range(j0, j0 + T):
                    acc = C[i, j]
                    for k in range(k0, k0 + T):
                        acc += A[i, k] * B[k, j]   # both operands are cache-resident
                    C[i, j] = acc
```

Inside one tile, each loaded element of `A` and `B` participates in $T$ multiply-accumulates before it is evicted. The bytes moved from DRAM drop by a factor of roughly $T$. The arithmetic is unchanged ($2N^3$ FLOPs). So:

$$
I_{\text{tiled}} \;\approx\; \frac{2 N^3}{8 N^3 / T} \;=\; \frac{T}{4} \ \frac{\text{FLOP}}{\text{byte}}.
$$

With $T = 64$, $I_{\text{tiled}} \approx 16$ FLOP/byte — a **64x improvement in arithmetic intensity** over the naive 0.25. More carefully, with three levels of tiling matched to L1/L2/registers, real kernels push the effective reuse much higher; the figure for this section uses $T=64$ and a clean register-blocked inner loop to reach an intensity near 64. The exact constant depends on how you count and on the cache hierarchy, but the *scaling* is the point: **tiling multiplies arithmetic intensity by roughly the tile dimension**, and that is what walks the kernel up the slanted memory roof and onto the flat compute roof.

There is a ceiling on $T$: the working set must fit. For square tiles you need three $T \times T$ blocks (one each of A, B, C) resident, i.e. $3 T^2$ elements. For a 32 KB L1 cache holding 4-byte floats, $3 T^2 \cdot 4 \le 32{,}768$ gives $T \le 52$ or so — which is why real CPU matmul kernels tile around 32-64 at the L1 level and use bigger tiles for L2. Choosing $T$ too small wastes reuse; too large spills the cache and you are memory-bound again. The optimal $T$ is hardware-specific, which is *exactly* why you search rather than hardcode.

#### Worked example: naive versus tiled matmul, the cache-reuse latency win

Let me make the intensity numbers concrete on a named target. Take $N = 1024$ single-precision matmul ($2 \cdot 1024^3 \approx 2.15$ GFLOP of work) on an M2 MacBook performance core, which has roughly $P_{\text{peak}} \approx 50$ GFLOP/s of single-thread fp32 SIMD throughput and per-core L1 bandwidth far above DRAM's ~100 GB/s.

- **Naive schedule.** $I_{\text{naive}} = 0.25$ FLOP/byte. Attainable performance is bounded by the memory roof: $I \cdot B_{\text{peak}} = 0.25 \times 100 = 25$ GFLOP/s in the *best* case, but the naive code's poor access pattern and cache misses drag the *achieved* rate down to roughly **4 GFLOP/s** in practice. Time $\approx 2.15 / 4 \approx 540$ ms.
- **Tiled + vectorized schedule ($T = 64$).** $I_{\text{tiled}} \approx 16$-$64$ FLOP/byte, well past the ridge point, so the kernel is now compute-bound. With SIMD and register blocking it reaches roughly **50 GFLOP/s**. Time $\approx 2.15 / 50 \approx 43$ ms.

That is a **~12.5x latency win** — 540 ms down to 43 ms — from rearranging loops and choosing a tile size. Zero change to the math, zero change to accuracy. This is the single most important number to internalize: the schedule, not the algorithm, is where an order of magnitude of edge latency frequently lives.

![A before-and-after figure showing a naive triple loop that streams all of B per output row at arithmetic intensity one and four gigaflops, versus a sixty-four by sixty-four tiled loop that keeps a block cache-resident at intensity sixty-four and fifty gigaflops.](/imgs/blogs/ml-compilers-and-autotuning-tvm-mlir-xla-5.png)

The reason a compiler beats a fixed heuristic here is now precise: the optimal $T$, the loop order, the unroll factor, and the vectorization width are all functions of the cache sizes, SIMD width, and exact shape. A hand-written heuristic picks one and hopes. A search tries thousands and measures.

### How big is the schedule space, really?

It is worth quantifying *why* nobody enumerates the schedule space, because the size is the whole justification for guided search. Take a single 2D convolution and count the independent schedule decisions. The spatial output loops, the channel loops, and the reduction loops can each be tiled at multiple cache levels — say three levels (register, L1, L2) — and each tile size is a divisor of the loop extent. A 256-extent loop has on the order of a dozen sensible tile factors; with, say, six loops each tiled at three levels, the tiling choices alone are roughly $12^{6 \times 3} \approx 12^{18}$, an astronomically large number before you have even chosen a loop order. Then multiply by the loop-order permutations (the number of ways to interleave the tiled loops, which for a dozen resulting loops is a large factorial), the unroll factors, the vectorization choices (which loop and what width), and the parallelization decisions (which loop maps to threads or thread blocks). The legal, *distinct* schedule count for one realistic conv routinely exceeds $10^9$ and can reach $10^{12}$ or more.

You cannot measure $10^9$ schedules. If each on-device measurement takes 0.3 seconds, measuring a billion of them would take about ten years per op. Even the cost model, at microseconds per evaluation, cannot score $10^{12}$ candidates exhaustively. So the search must be *both* guided (the cost model prunes the obviously bad) *and* sampled (evolutionary mutation and random restarts explore the good neighborhoods without enumerating). The combinatorial explosion is precisely why a generated-and-searched kernel can find a configuration a human would never hand-pick: the optimum frequently sits at a non-obvious combination — an unusual tile size paired with a counterintuitive loop order — that no heuristic's priors would suggest but that the hardware happens to love. The search does not need to understand *why* that combination is fast; it only needs to measure that it is.

### The cost model, made concrete

The cost model is the component that makes searching $10^9$ schedules tractable, so it is worth stating what it actually predicts. For a candidate schedule $\sigma$ it predicts a runtime (or, in practice, a normalized throughput score) $\hat{c}(\sigma)$ from a feature vector $\phi(\sigma)$ extracted statically from the loop nest — without running it. The features are exactly the things our tiling analysis showed matter: estimated bytes moved at each cache level, arithmetic intensity, vectorization utilization, loop-unroll counts, estimated occupancy on the target. The model is trained to minimize a ranking-aware loss over measured pairs, because the search only needs the *relative order* of candidates right, not the absolute milliseconds:

$$
\mathcal{L} \;=\; \sum_{(\sigma_i,\,\sigma_j)\,\in\,\mathcal{P}} \ell\!\big(\operatorname{sign}(c_i - c_j),\; \hat{c}(\sigma_i) - \hat{c}(\sigma_j)\big),
$$

where $c_i, c_j$ are the *measured* runtimes of two candidates and $\ell$ is a pairwise ranking loss. Getting the ranking right is enough to pick the top-k to measure; the absolute prediction can be off by a constant and the search still works. Ansor uses a gradient-boosted tree for this by default — cheap to train, robust on small measurement sets, and fast to evaluate. The reason the loop in the earlier figure *retrains* the model from new measurements is that the model starts ignorant of *this* target's quirks; each round of true measurements teaches it where its predictions were wrong, and after a few rounds its top-k picks are almost always genuinely among the fastest. This is the precise mechanism by which "score $10^4$ candidates for the cost of measuring one" turns into a kernel that beats a vendor library.

### Why convolutions are special: im2col versus direct schedules

One more piece of science makes the TVM section concrete: convolutions have *fundamentally different* schedule families, and choosing among them is part of what the search does. The classic approach is **im2col + GEMM**: unfold every sliding window of the input into a column of a big matrix, then run a dense matmul against the reshaped kernel. This reuses all the matmul tiling machinery and is why convs and matmuls share so much, but it inflates memory by the kernel area (a 3x3 conv's im2col matrix is ~9x the input size) — a memory-bound cost on a bandwidth-poor edge chip. The alternative is a **direct conv** schedule that loops over the windows in place without materializing the unfolded matrix, trading the GEMM's clean tiling for lower memory traffic. Which family wins depends on the shape: large channel counts favor im2col-GEMM (the matmul is fat and tiles beautifully), while small channel counts on a bandwidth-poor target favor direct convs (avoiding the im2col blowup). A fixed heuristic must commit to one family; an auto-scheduler explores *both* families' schedule spaces and measures which one the actual hardware prefers for the actual shape. This is a concrete, common case where search finds a win a single hand-written kernel cannot, because the right *algorithmic strategy*, not just the right tile size, depends on the shape and target.

## TVM and the auto-scheduler: searching the schedule space

TVM, introduced by Tianqi Chen and colleagues in 2018, was the system that made "search the schedule space for ML ops" practical and general. Its model has two layers that map directly onto the Halide split.

**Tensor Expression (TE)** is the algorithm language. You declare *what* a tensor op computes as a mathematical expression over index ranges:

```python
import tvm
from tvm import te

# Algorithm: C[i, j] = sum_k A[i, k] * B[k, j]  — the WHAT, no schedule yet
N = 1024
k = te.reduce_axis((0, N), name="k")
A = te.placeholder((N, N), name="A", dtype="float32")
B = te.placeholder((N, N), name="B", dtype="float32")
C = te.compute(
    (N, N),
    lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
    name="C",
)
```

That `te.compute` declares the algorithm and *nothing else*. There is no loop order, no tiling, no parallelism — exactly the Halide separation. The schedule is a separate object you build by applying transformations:

```python
# Schedule: the HOW — split into tiles, reorder, vectorize, parallelize
s = te.create_schedule(C.op)
xo, xi = s[C].split(C.op.axis[0], factor=64)   # tile rows
yo, yi = s[C].split(C.op.axis[1], factor=64)   # tile cols
ko, ki = s[C].split(k, factor=8)               # tile the reduction
s[C].reorder(xo, yo, ko, xi, ki, yi)           # loop order matters
s[C].vectorize(yi)                              # SIMD the inner col loop
s[C].parallel(xo)                               # one tile-row per core
func = tvm.build(s, [A, B, C], target="llvm -mcpu=apple-m1")
```

This is the manual version. You *could* hand-write schedules forever — that is the AutoTVM era, where you wrote a parameterized schedule "template" with knobs (tile sizes, unroll factors) and let TVM search over the knob values. AutoTVM works, but it still requires a human to write the template, and the template constrains what the search can find.

**Ansor** (Zheng et al., 2020), TVM's auto-scheduler, removed the template. Instead of a human writing a parameterized schedule, Ansor *generates* the search space automatically from the algorithm by composing a set of derivation rules (tiling, fusion, unrolling, parallelization) into a hierarchical space of complete schedules — then searches that space. This is the modern default, and it is what the rest of this section means by "autotuning."

### How the search actually works

You cannot enumerate the schedule space. For a single conv it can contain $10^9$ or more distinct schedules; measuring each one on hardware would take centuries. So the search is *guided*, and the guidance has three moving parts that loop, shown in the figure below.

![A timeline figure of the cost-model search loop sampling about ten thousand candidate schedules, ranking them with a fast learned cost model, measuring the top few on the real device, and retraining the model from those measurements until latency converges.](/imgs/blogs/ml-compilers-and-autotuning-tvm-mlir-xla-2.png)

1. **Sample candidates.** The auto-scheduler samples many complete schedules from the generated space — thousands at a time — using evolutionary mutation (mutate tile sizes, swap loop orders) plus random restarts to maintain diversity.
2. **Rank with a learned cost model.** Measuring on hardware is slow (each measurement is a compile + run, tens of milliseconds to seconds). So a *cost model* — a small gradient-boosted tree or neural net trained on features of the loop nest (tile sizes, vectorization, estimated cache behavior) — predicts the runtime of each candidate in microseconds, with no hardware needed. This lets the search *score* $10^4$ candidates for the cost of *measuring* one.
3. **Measure the top-k on the real device, then retrain.** The few hundred most promising candidates per round are actually compiled and timed on the target. Those true measurements are added to the cost model's training set, the model is retrained, and the loop repeats. Over rounds, the cost model gets sharper, the samples get better, and the best measured latency converges.

The reason this beats a fixed heuristic is structural, not incidental. A heuristic encodes one expert's priors about good schedules. The search *measures* the actual hardware, so it captures effects no human models exactly: bank conflicts, the precise cache replacement policy, how the prefetcher behaves on this stride, how the specific NPU's tiling units like to be fed. The cost model lets it explore breadth (millions of schedules considered) while on-device measurement gives it ground truth (hundreds actually timed). That breadth-with-grounding is the thing a static heuristic cannot match, and it is why an autotuned kernel can beat a vendor library on a shape the vendor did not specifically tune.

The cost is right there in step 3: **on-device measurement is the bottleneck**, and it is why autotuning takes minutes to hours. Each task (a unique op shape) needs hundreds to thousands of trials to converge; a full model can have dozens of unique tasks. This is build-time cost, paid once. The output is a tuning log you can save and reuse.

#### Worked example: an autotuned conv beating the default on a specific shape

Here is the kind of result that justifies the build-time cost. Take a 1x1 convolution with 384 input channels, 96 output channels, on a 28x28 feature map, int8, targeting an ARM Mali-G78 mobile GPU — a representative "off the vendor fast path" shape produced by a pruned, quantized MobileNet variant.

- **Vendor library fallback (default runtime kernel):** the generic int8 1x1 conv kernel runs at roughly **2.1 ms** per call. The vendor's tuned fast path exists only for channel counts that are multiples of 128; 384-to-96 is not on it.
- **TVM/Ansor autotuned kernel** for this exact shape and target, after ~800 trials (~25 minutes of on-device tuning): roughly **0.9 ms** per call.

That is a **~2.3x latency win on a single layer**, achieved with no retraining and no hand-written GPU code — just a search over schedules for the specific shape the model actually has. Multiply that across the handful of off-fast-path layers in the network and it is the difference between 18 ms and 11 ms end-to-end, which on a real-time camera pipeline is the difference between 30 fps and 60 fps. The build-time cost was one overnight tuning run, amortized over every future inference. *This* is when a compiler is worth it: a non-mainstream shape on hardware where the vendor left the fast path on the table.

### The practical TVM/Ansor autotune flow, end to end

The figure below is the lifecycle; the code after it is the runnable sketch.

![A timeline figure of the TVM and Ansor autotune flow importing a model from ONNX, extracting tunable tasks per op shape, tuning N trials with Ansor, picking the best schedule from the log, building a deployable module, then benchmarking it at one point three times over cuDNN.](/imgs/blogs/ml-compilers-and-autotuning-tvm-mlir-xla-4.png)

```python
import tvm
from tvm import relay, auto_scheduler
from tvm.contrib import graph_executor
import onnx, numpy as np

# 1. Import the model into Relay (TVM's high-level graph IR)
onnx_model = onnx.load("mobilenet_pruned_int8.onnx")
shape_dict = {"input": (1, 3, 224, 224)}
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

target = tvm.target.Target("opencl -device=mali")   # the actual edge GPU

# 2. Extract the tunable tasks — each unique op+shape becomes one task
tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
print(f"{len(tasks)} tunable tasks extracted")

# 3. Tune: a fixed TRIAL BUDGET is the build-time cost knob
log_file = "mobilenet_mali.json"
tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=20000,                 # total on-device measurements
    runner=auto_scheduler.RPCRunner(          # measure on the REAL device over RPC
        "mali-board", "127.0.0.1", 9190,
        number=3, repeat=1, timeout=10,
    ),
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
)
tuner.tune(tune_option)                        # this is the minutes-to-hours step

# 4. Compile with the best schedules found in the log
with auto_scheduler.ApplyHistoryBest(log_file):
    with tvm.transform.PassContext(opt_level=3,
                                   config={"relay.backend.use_auto_scheduler": True}):
        lib = relay.build(mod, target=target, params=params)

# 5. Deploy + benchmark (batch=1, the edge reality)
dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](dev))
module.set_input("input", np.random.randn(1, 3, 224, 224).astype("float32"))
print(module.benchmark(dev, number=100, repeat=10))   # report p50, warm runs
```

Two things deserve emphasis. The `num_measure_trials` budget is your build-time-cost dial: more trials means better kernels but longer tuning. And `RPCRunner` measures on the *actual device* over RPC — you tune for a Mali board by running candidate kernels on a Mali board, because the cost model alone cannot capture every quirk. The `TaskScheduler` is smart about budget allocation: it spends more trials on the tasks (op shapes) that dominate the model's latency, which is why you give it a *total* budget rather than per-task.

## MLIR: a reusable, multi-level compiler infrastructure

TVM is a compiler. MLIR (Lattner et al., 2021) is something subtler and arguably more consequential: it is *infrastructure for building compilers*. The name stands for Multi-Level Intermediate Representation, and the key word is **multi-level**.

The problem MLIR solves is that historically every ML compiler reinvented the same plumbing. Each one had its own IR, its own pass manager, its own way of doing constant folding and dead-code elimination, its own lowering from "high-level model op" all the way down to "machine instruction" in essentially one giant leap. That single leap is brutal: a high-level op like `conv2d` and a low-level concept like an LLVM `fmul` live so far apart that the translation between them is a sprawling, hard-to-maintain, hard-to-optimize monolith.

MLIR's answer is to make the IR a stack of **dialects**, each a small, self-contained set of operations at one level of abstraction, with **progressive lowering** moving the program down the stack one verifiable step at a time. The figure below shows the canonical path.

![A stack figure of MLIR progressive lowering from framework ops in TOSA or StableHLO, down to the linalg dialect of structured tensor ops, then affine and scf for explicit loops and tiling, then memref and vector for buffers and SIMD, down to LLVM or SPIR-V backends, with every step verified.](/imgs/blogs/ml-compilers-and-autotuning-tvm-mlir-xla-3.png)

Read it top to bottom:

- **Framework-level dialects** (TOSA, StableHLO) represent the model the way a framework thinks about it: `conv`, `matmul`, `softmax` as whole-tensor ops. This is where a model from TensorFlow, JAX, or PyTorch lands after import.
- **`linalg`** is the structured-ops dialect: still tensor-level, but expressed as generic structured loops (a `linalg.matmul`, a `linalg.generic`) that the compiler knows how to tile and fuse mechanically. This is where most of the interesting transformation happens.
- **`affine` / `scf`** make the loops explicit and analyzable. `affine` loops have a precise mathematical structure that the compiler can reason about for tiling, fusion, and dependence analysis; `scf` (structured control flow) handles the more general cases. This is where a tile-size decision becomes concrete loop bounds.
- **`memref` / `vector`** lower tensors to explicit memory buffers (`memref`) and SIMD operations (`vector`). Now there is no more "tensor" level — there are buffers, loads, stores, and vector registers.
- **`LLVM` / `SPIR-V`** are the bottom: standard LLVM IR for CPU, or SPIR-V for GPU/Vulkan, handed off to a mature backend that emits actual machine code.

Why does this layered design matter so much? Three reasons.

**Verifiability.** Each lowering step is small and has a verifier. After lowering `linalg` to `affine`, MLIR *checks* that the result is well-formed. Bugs are localized to one small pass instead of buried in a monolithic translation. This is the same engineering reason microservices beat a monolith: small, independently verifiable steps.

**Reuse.** A tiling pass written at the `linalg` level works for *every* model and *every* backend, because it operates on the structured-ops representation, not on a specific op or target. Write the optimization once, reuse it everywhere. Before MLIR, every compiler reimplemented tiling; with MLIR, it is a shared pass. This is the leverage that lets a small team build a competitive compiler — they stand on the shared dialect infrastructure instead of rebuilding it.

**Extensibility.** Your new accelerator does not match any existing dialect? Define your own dialect for it and write lowering rules from `linalg` (or `vector`) into your dialect. The rest of the stack — the import, the high-level optimizations, the verifiers — you get for free. This is precisely why MLIR underpins so many modern compilers: it is the substrate, not the product.

That last point is the practical takeaway. You rarely use "MLIR" directly. You use compilers *built on* MLIR. **Torch-MLIR** lowers PyTorch into MLIR so PyTorch models can target the whole MLIR ecosystem. **IREE** (more below) is an end-to-end MLIR-based compiler and runtime aimed at edge and heterogeneous deployment. Even XLA has been migrating its internals toward MLIR via the StableHLO dialect. When you read that a new accelerator "has MLIR support," it means someone wrote a dialect and lowering rules, and now the shared infrastructure can target it. That network effect — every new frontend and backend makes the shared middle more valuable — is why MLIR has become the de facto IR substrate for ML compilation.

### IREE and Triton, briefly

Two MLIR-adjacent tools deserve a mention because they show up constantly in edge work.

**IREE** (Intermediate Representation Execution Environment) is the most edge-relevant MLIR-based compiler. It takes a model (from TF, JAX, or PyTorch via the StableHLO/Torch-MLIR frontends), lowers it through the MLIR dialect stack, and produces a compact, ahead-of-time-compiled artifact plus a tiny runtime that can target CPU, GPU (via Vulkan/SPIR-V or CUDA/ROCm), and embedded targets. Its design priorities — ahead-of-time compilation, a small runtime footprint, explicit scheduling — are exactly what edge deployment wants: you compile on a build machine and ship a self-contained, low-overhead binary. For heterogeneous or unusual edge targets where you want one toolchain across CPU and GPU, IREE is increasingly the answer.

The IREE flow is a two-step build-then-run, and seeing the commands makes the ahead-of-time edge story concrete. You compile the model to a `.vmfb` flatbuffer once, on a build machine, targeting the device's backend, then run it with the small runtime on the device:

```bash
# 1. Compile (on a build server) — lower StableHLO/Torch-MLIR through the dialect
#    stack to a self-contained flatbuffer for the chosen backend.
iree-compile model.mlir \
  --iree-hal-target-backends=vulkan-spirv \   # mobile GPU via Vulkan/SPIR-V
  --iree-vulkan-target-triple=adreno-unknown-android \
  -o model_adreno.vmfb

# For a CPU edge target instead (e.g. a Cortex-A or RISC-V board):
iree-compile model.mlir \
  --iree-hal-target-backends=llvm-cpu \
  --iree-llvmcpu-target-triple=aarch64-none-linux-gnu \
  --iree-llvmcpu-target-cpu-features=+dotprod \  # use int8 dot-product ISA
  -o model_arm.vmfb

# 2. Run (on the device) — the tiny runtime loads the flatbuffer and executes.
iree-run-module --module=model_arm.vmfb \
  --device=local-task \
  --function=main \
  --input="1x3x224x224xf32=@input.npy"
```

The point of the snippet is the *separation*: all the heavy MLIR lowering and optimization happens at compile time on a capable machine, and the device only carries a small runtime plus the flatbuffer. That ahead-of-time model is the opposite of a just-in-time framework that drags a multi-hundred-megabyte runtime onto the device, and it is exactly why IREE fits microcontroller-class and mobile targets where binary size and cold-start latency are first-class constraints. The `--iree-llvmcpu-target-cpu-features` flag is the kind of last-mile control the edge needs — telling the compiler to emit the int8 dot-product instructions a specific ARM core has, which a generic build would miss.

**Triton** (Tillet et al., 2019) is a different and complementary tool: a Python-embedded language for writing GPU kernels at the *block* level. You write the kernel logic — load a block of data, do the math, store the result — in Python with Triton's primitives, and Triton's compiler handles the within-block scheduling (memory coalescing, shared-memory management, the register-level details) and autotunes over configurations like block size and number of warps. It hits a sweet spot: more control than a black-box library, far less pain than raw CUDA. It is the kernel layer underneath PyTorch's `torch.compile` (via TorchInductor), and it is how a lot of custom edge/server GPU kernels — fused attention, fused norms, custom quantized GEMMs — actually get written today. We will write a Triton skeleton in the practical section.

## XLA: whole-graph compilation and the fusion approach

XLA (Accelerated Linear Algebra) is the compiler behind JAX and TensorFlow, and its defining idea is different from TVM's per-op schedule search. XLA's superpower is **whole-graph compilation with aggressive operator fusion**.

XLA takes the entire computation graph, represents it in **HLO** (High-Level Operations, XLA's IR), and runs whole-graph optimizations across it. The headline optimization is **fusion**: combining many small operators into a single compiled kernel.

To see why fusion is such a large win, count the memory traffic of an unfused elementwise chain. Suppose you compute `y = tanh(a * x + b)` where each tensor has $M$ elements in fp32. Unfused — the eager-execution default — this is three separate kernels:

1. `t1 = a * x` — read `a` and `x` ($2 \cdot 4M$ bytes), write `t1` ($4M$ bytes).
2. `t2 = t1 + b` — read `t1` and `b` ($2 \cdot 4M$), write `t2` ($4M$).
3. `y = tanh(t2)` — read `t2` ($4M$), write `y` ($4M$).

Total DRAM traffic: roughly $24M$ bytes, *plus three kernel launches*, for an operation that does only ~$3M$ FLOPs. The arithmetic intensity is about $3M / 24M = 0.125$ FLOP/byte — catastrophically memory-bound. The intermediates `t1` and `t2` are written to DRAM and immediately read back, pure waste.

Now **fuse** all three into one kernel. The fused kernel reads `a`, `x`, `b` once, keeps `t1` and `t2` in registers (never touching DRAM), and writes `y` once:

$$
\text{Bytes}_{\text{fused}} \;\approx\; 4M \cdot (3 \text{ reads} + 1 \text{ write}) \;=\; 16M, \quad \text{vs} \quad 24M \text{ unfused},
$$

and — more importantly for a longer chain — the intermediates *never hit memory at all*, so the savings grow with chain length. A 10-op pointwise chain unfused moves ~$80M$+ bytes through DRAM in nine round-trips; fused, it moves the input and output once, roughly $8M$ bytes, a **~10x reduction in memory traffic** and *one* kernel launch instead of ten. The figure below contrasts the two.

![A before-and-after figure showing an unfused eager chain of five separate kernels with four DRAM round-trips and five launch overheads, versus an XLA-fused HLO version that is one kernel reading and writing memory once with intermediates held in registers and roughly three times less DRAM traffic.](/imgs/blogs/ml-compilers-and-autotuning-tvm-mlir-xla-6.png)

This is why XLA shines on the memory-bound, launch-overhead-heavy parts of a model: long chains of elementwise ops (activations, normalizations, bias adds, the glue around the big matmuls). The big matmuls and convs are usually already at the compute roof, so XLA leans on the vendor library (cuDNN, the TPU's MXU) for those and concentrates its own effort on *fusing everything around them*. This is a deliberate division of labor: vendor kernels for the compute-bound heavy hitters, compiler fusion for the memory-bound glue.

The practical face of XLA is one decorator in JAX:

```python
import jax, jax.numpy as jnp

def block(x, a, b, w):
    # A chain of pointwise ops + a matmul — exactly what fusion loves
    h = jnp.tanh(a * x + b)        # pointwise chain -> fused into one kernel
    return h @ w                   # matmul -> vendor kernel, fusion around it

# jit triggers XLA: trace the graph, fuse, compile to one (or few) kernels
fast_block = jax.jit(block)

# First call compiles (slow); subsequent calls run the compiled, fused kernel
y = fast_block(x, a, b, w)         # whole-graph compiled, fused, launched
```

`jax.jit` traces the function into HLO, hands it to XLA, which fuses the pointwise chain into a single kernel and compiles the whole thing. The first call pays the compile cost; every call after runs the fused artifact. In TensorFlow the equivalent is `tf.function(jit_compile=True)`. The mental model to carry: **XLA optimizes the graph; TVM/Ansor optimize the kernel**. They are complementary — XLA decides *which kernels exist and how the graph flows*, schedule-search compilers decide *how each kernel runs*. Modern stacks increasingly do both.

The trade-off to know honestly: XLA's fusion is heuristic-driven (rule-based), not an exhaustive schedule search, so it is fast to compile and excellent at the fusion job, but it does not autotune individual kernels the way Ansor does. And `jit` recompiles when input *shapes* change, so dynamic-shape workloads can thrash the compile cache — you control this with shape padding or `donate_argnums` and by keeping shapes static. On the edge, XLA is most relevant via TF Lite / LiteRT's use of XLA-derived kernels and via JAX-on-edge experiments; for bare-metal edge targets, IREE and TVM tend to be the more natural fit.

#### Worked example: fusion on a transformer MLP block, the launch-and-bandwidth win

Put fusion numbers on a real block. Take a transformer feed-forward block at hidden size $d = 4096$, batch-times-sequence $M = 64 \times 512 = 32768$ tokens, fp16, on a single mobile-class GPU. The block is two matmuls with a GELU and a residual-plus-LayerNorm around them. The matmuls dominate the FLOPs and are already compute-bound, so they ride the vendor kernel. The interesting part is the *glue*: the GELU on the $M \times 4d$ intermediate (about $32768 \times 16384 = 5.4 \times 10^8$ elements), the two bias adds, the residual add, and the LayerNorm.

- **Unfused (eager).** Each of those glue ops is its own kernel that reads and writes the big $M \times 4d$ tensor. At 2 bytes/element in fp16, one full pass over that intermediate is about $1.07$ GB of traffic. The GELU alone is a read-plus-write, ~$2.1$ GB; across the handful of glue ops you move on the order of $6$-$8$ GB through DRAM and pay ~5 separate kernel launches, all for arithmetic that is a tiny fraction of the matmuls' FLOPs. On a ~200 GB/s mobile GPU that glue alone costs roughly $7\,\text{GB} / 200\,\text{GB/s} \approx 35$ ms — and it is pure overhead around the real work.
- **Fused (XLA / Inductor).** The GELU and bias adds fuse into the first matmul's *epilogue* (computed on the matmul output while it is still in registers, never written to DRAM separately), and the residual+LayerNorm fuse into the second matmul's epilogue. The big intermediate is read and written *once* as part of the matmul kernels, not five times by five glue kernels. The glue's standalone DRAM traffic and its five launches collapse toward zero, cutting that ~35 ms of overhead to a few ms.

The end-to-end effect is the ~1.3x-2x `torch.compile`/`jit` speedup quoted earlier, and now you can see exactly where it comes from: it is not faster matmuls, it is *deleting the memory-bound glue* that surrounded them. On the edge, where bandwidth is the scarce resource, this is frequently the single largest compiler win on a transformer — and it is why the first thing to try on any transformer-on-edge is the one-line fusion compiler before reaching for anything heavier.

## The practical section: torch.compile and a Triton kernel

The most accessible entry point to ML compilers for a PyTorch user is `torch.compile`, introduced in PyTorch 2.0. One line wraps your model and, behind the scenes, captures the graph (TorchDynamo), optimizes it, and lowers it through **TorchInductor**, which generates fused **Triton** kernels for GPU (and C++/OpenMP for CPU). It is the "compiler benefits with zero ceremony" option.

### torch.compile before/after

```python
import torch, torch.nn as nn, time

class MLPBlock(nn.Module):
    def __init__(self, d=4096):
        super().__init__()
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)
        self.norm = nn.LayerNorm(d)
    def forward(self, x):
        # GELU + the norm + residual = lots of fusible pointwise glue
        h = torch.nn.functional.gelu(self.fc1(x))
        h = self.fc2(h)
        return self.norm(x + h)

model = MLPBlock().cuda().eval()
x = torch.randn(64, 512, 4096, device="cuda")

def bench(fn, iters=100):
    for _ in range(10):              # warm-up: JIT/compile + cache fill
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3   # ms

with torch.no_grad():
    eager_ms = bench(lambda: model(x))

    # ONE LINE: capture graph, fuse, lower to Triton kernels
    compiled = torch.compile(model, mode="max-autotune")
    # first compiled call(s) pay the compile + autotune cost inside warm-up
    comp_ms = bench(lambda: compiled(x))

print(f"eager:        {eager_ms:.2f} ms")
print(f"torch.compile {comp_ms:.2f} ms   ({eager_ms / comp_ms:.2f}x)")
```

On this kind of block — a matmul sandwiched in pointwise glue (GELU, the bias adds, the LayerNorm, the residual) — `torch.compile` typically lands a **1.3x to 2x** speedup on a single GPU, almost entirely by fusing the glue into the matmul kernels' epilogue and prologue and cutting launch overhead, exactly the XLA-style fusion win analyzed above. The `mode="max-autotune"` flag tells Inductor to autotune its Triton kernel configs (block sizes, number of warps) by measuring on the device — a small dose of schedule search — which costs more compile time for a better kernel. The measurement harness is the part people get wrong: **always warm up** (the first call pays compile + autotune, which would poison the average), **always `torch.cuda.synchronize()`** before timing (GPU calls are async), and report **batch=1** numbers separately if batch=1 is your edge reality, because fusion wins shrink at small batch where you are launch-bound differently.

### A Triton kernel skeleton

When `torch.compile` is not enough — a genuinely novel op, or a fused pattern Inductor will not discover — you drop to Triton and write the kernel yourself, in Python, at the block level. Here is the canonical skeleton: a fused `y = relu(a * x + b)` that does in one kernel what eager does in three.

```python
import triton
import triton.language as tl
import torch

@triton.autotune(                       # search a few configs, measure on device
    configs=[
        triton.Config({"BLOCK": 256},  num_warps=2),
        triton.Config({"BLOCK": 512},  num_warps=4),
        triton.Config({"BLOCK": 1024}, num_warps=8),
    ],
    key=["n_elements"],                  # re-tune when the size class changes
)
@triton.jit
def fused_affine_relu(x_ptr, a_ptr, b_ptr, y_ptr, n_elements,
                      BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)          # which block am I
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements             # guard the ragged last block
    x = tl.load(x_ptr + offs, mask=mask)
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    y = tl.maximum(a * x + b, 0.0)       # the whole chain, intermediates in regs
    tl.store(y_ptr + offs, y, mask=mask) # one read pass, one write pass

def fused_op(x, a, b):
    y = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    fused_affine_relu[grid](x, a, b, y, n)
    return y
```

Three things to notice. The kernel operates on a *block* of elements (`BLOCK` of them per program instance), and Triton handles the within-block memory coalescing and register allocation — you never write a thread index or manage shared memory by hand the way raw CUDA forces you to. The `@triton.autotune` decorator runs a tiny schedule search over `BLOCK` and `num_warps`, measuring on the actual GPU and caching the winner per size class — the same search-and-measure principle as Ansor, scoped to a hand-written kernel. And the whole `a * x + b` then `relu` chain executes with the intermediates living in registers, never touching DRAM — the fusion win, made explicit by your own hand. This is the layer at which fused attention kernels (FlashAttention is, at heart, a hand-fused Triton-style kernel), custom quantized GEMMs, and novel edge ops get written when no library and no auto-compiler will do it for you.

## Results: comparing the compilers and the autotune cost curve

Time to put the four stacks side by side. The decision is not "which is best" — they target different problems — but "which fits this situation." The matrix below is the comparison I keep in my head.

![A matrix figure comparing TVM Ansor, XLA HLO, MLIR IREE, and torch.compile across core approach, whether they search schedules, supported targets, and when to reach for each one.](/imgs/blogs/ml-compilers-and-autotuning-tvm-mlir-xla-7.png)

| Compiler | Core approach | Searches schedules? | Targets | Reach for it when |
|---|---|---|---|---|
| **TVM / Ansor** | Tensor expression + auto-scheduled kernels | Yes — cost-model-guided search, on-device measurement | CPU, GPU, NPU, DSP, MCU (broadest) | Novel op or unusual hardware; you need to beat a vendor fallback on a specific shape |
| **XLA (HLO)** | Whole-graph compilation + heuristic fusion | No (heuristic fusion; relies on vendor kernels for heavy matmuls) | TPU, GPU, CPU | You are in JAX/TF and want a fast, hands-off graph-level win |
| **MLIR / IREE** | Dialect-based progressive lowering, AOT compile | Optional (pluggable; IREE adds autotuning paths) | CPU, GPU, Vulkan/SPIR-V, embedded | Portable, ahead-of-time edge deploy across heterogeneous targets |
| **torch.compile** | TorchDynamo capture + Inductor + Triton | Partial — autotunes Triton configs in `max-autotune` | CUDA GPU, CPU | One-line PyTorch speedup with no toolchain change |

A few reads of this table. If you live in PyTorch and want a quick win, `torch.compile` is the no-brainer first move — it is one line and it composes with everything. If you live in JAX/TF, XLA via `jit` is the equivalent default. If you are deploying to *the edge zoo* — a Mali GPU, a Hexagon DSP, an obscure NPU, a microcontroller — and you need a kernel that does not exist in any vendor library, **TVM/Ansor** is the workhorse, because it generates and autotunes kernels for arbitrary targets. And if you want one portable, ahead-of-time toolchain that compiles once on a build server and ships a small self-contained artifact across CPU and GPU edge targets, **IREE** (on MLIR) is the modern choice. These are not mutually exclusive: a real pipeline might use `torch.compile` in development, export to a graph, and hand the export to TVM or IREE for the final edge build.

### The autotune-trials-versus-latency curve

The build-time cost of autotuning is not linear in value — it follows a sharply diminishing-returns curve, and knowing its shape tells you when to stop tuning. The pattern, across TVM/Ansor runs I have done and what the literature reports, looks like this:

| Trials (per task) | Best latency found | Notes |
|---|---|---|
| 0 (default heuristic kernel) | 100% (baseline) | The fixed-schedule starting point |
| 64 | ~78% | Big early wins — the cost model is finding obvious tiling |
| 256 | ~68% | Diminishing but real; most of the win is here |
| 1000 | ~62% | The knee of the curve; near convergence |
| 4000 | ~60% | Marginal; mostly chasing the last 1-2% |
| 20000 | ~59% | Essentially flat — not worth the wall-clock |

The shape is the lesson: roughly **70-80% of the achievable speedup arrives in the first few hundred trials**, and the curve flattens hard after ~1000 trials per task. So a pragmatic default is to budget on the order of 800-1500 trials per dominant task, watch the best-latency-so-far plateau, and stop when it has been flat for a couple hundred trials. Spending 20,000 trials per task to shave the last 1% is almost never worth the overnight wall-clock unless that 1% is genuinely the difference between hitting a hard latency SLA and missing it. The `TaskScheduler` helps here by concentrating the total budget on the tasks that actually move end-to-end latency, so you set a *total* budget and let it allocate.

#### Worked example: the build-time-cost decision on a real deployment

Make the trade-off concrete. Suppose a quantized detection model has 14 unique op tasks, and you have a single Mali board for measurement. Tuning at 1000 trials/task, with each measurement taking ~0.3 s (compile + 3 runs) and the cost model overhead folded in, a full run is roughly $14 \times 1000 \times 0.3\,\text{s} \approx 70$ minutes of wall-clock — call it an overnight job with margin. The payoff, from the curve and the per-layer worked example earlier, is an end-to-end drop from ~18 ms to ~11 ms per frame (~1.6x), which lifts a real-time pipeline from 30 fps to 60 fps. That trade — one overnight build-machine job, amortized over millions of inferences in the field — is clearly worth it. Now flip it: if the model is all standard 3x3 convs at power-of-two channel counts on an NVIDIA Jetson with excellent cuDNN coverage, the same 70-minute job might buy you 5%, because the vendor kernels are already at the roofline. *That* is when you skip the compiler and ship the runtime's kernels. The decision is not ideological; it is whether the vendor left latency on the table for your specific shapes and target.

## When to reach for a compiler (and when not to)

Here is the decision distilled. The figure below is the tree; the prose after it is how I actually reason through it.

![A decision tree figure for when to reach for a compiler, branching on whether the op and target are well supported into using a runtime kernel or autotuning for last-mile latency, and on whether the op is novel or the hardware unusual into the compiler generating the kernel at the cost of hours of tuning.](/imgs/blogs/ml-compilers-and-autotuning-tvm-mlir-xla-8.png)

**Reach for a compiler when:**

- **The op is novel.** A fused attention variant, a custom quantized GEMM, a new normalization — anything no vendor library implements. The compiler (or hand-written Triton) is the only way to get a fast kernel without writing assembly. This is the clearest, least-ambiguous case.
- **The hardware is unusual.** A Mali GPU, a Hexagon DSP, a RISC-V accelerator, a microcontroller — the long tail of edge silicon where no vendor ships hand-tuned kernels for your op. TVM/Ansor generating a kernel for the target is frequently the *only* path that produces a fast kernel at all.
- **Your shapes fall off the vendor fast path.** Pruned channel counts, depthwise/grouped convs, odd sequence lengths — shapes the vendor library handles with a slow generic fallback. Autotuning for *your* exact shape recovers the 2-3x the fallback wastes.
- **You need the last-mile latency.** You are already on supported hardware with decent kernels, but you are 20% short of a hard SLA and have exhausted the four compression levers. Autotuning the dominant kernels can find the remaining margin that a one-size-fits-all kernel leaves behind.

**Stick with the runtime's built-in kernels when:**

- **Common ops on well-supported hardware.** Standard 3x3 convs, plain GEMMs, attention on an NVIDIA GPU with cuDNN/cuBLAS, or on a CPU with oneDNN/XNNPACK. The vendor kernels are at or near the roofline; a compiler's search converges to roughly the same number after burning an hour. Don't pay build time to match what you already have.
- **Shapes change constantly.** Autotuning specializes to a shape. If your input shapes are highly dynamic and you cannot pad to a few buckets, the autotuned kernel's advantage evaporates and you re-pay compile cost on every shape — a runtime with shape-agnostic kernels is more robust.
- **The build-time budget does not exist.** Autotuning needs the *target device* available for measurement during the build, and hours of it. If you cannot put the device in your CI, or you ship dozens of target variants, the operational cost of per-target autotuning can outweigh the latency win. Be honest about this; it is a real and common blocker.
- **`torch.compile` / `jit` already hit the target.** Always try the one-line option first. If `torch.compile` or `jax.jit` gets you to SLA, you are done — the full TVM/IREE autotuning pipeline is a much heavier lift you only take on when the easy win falls short.

The meta-rule, which echoes the rest of this series: **measure first, optimize second.** Profile the model, find the kernels that actually dominate latency (use the [roofline](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives) to know whether they are memory- or compute-bound and therefore whether fusion or tiling is the lever), and only autotune those. Autotuning every op uniformly wastes most of the build budget on kernels that contribute nothing to end-to-end latency. The compiler is a precision tool, not a blanket — and it composes with, rather than replaces, the [graph-level optimizations](/blog/machine-learning/edge-ai/graph-level-optimization) (constant folding, layout transforms, op fusion at the graph level) that you should already have applied.

## Stress-testing the decision

Let me poke at the edges, because the clean rules above hide real failure modes.

**What happens when the NPU does not support the op and it falls back to CPU?** This is the most common edge surprise. Your autotuned-or-not kernel is fast, but the *previous* op in the graph runs on the NPU and the next runs on CPU, so a tensor round-trips across the SoC's memory boundary with a layout conversion in between — and that transfer can cost more than the kernel you optimized. The fix is not a faster kernel; it is keeping the op *on the accelerator*, which means choosing a compiler/runtime path that supports the op natively or rewriting the model to avoid the unsupported op. A compiler helps here precisely because it can *generate* an NPU kernel for the op the vendor library skipped, eliminating the fallback. Always profile the *whole graph's* device placement, not just the hot kernel.

**What happens when the model is memory-bound, not compute-bound?** Then tiling for compute does nothing — you were already not compute-limited. The lever shifts to *fusion* (cut DRAM round-trips, the XLA win) and to *quantization* (fewer bytes per element, which directly raises the memory roof). This is why the roofline diagnosis comes before the compiler choice: a memory-bound kernel wants fusion and fewer bytes, not a cleverer tile size. Reaching for Ansor's schedule search on a memory-bound elementwise chain is using the wrong tool — XLA-style fusion or quantization is the right one.

**What happens when shapes are dynamic?** Autotuning specializes to a static shape, so dynamic shapes are the autotuner's natural enemy. The pragmatic answer is *bucketing*: pad inputs to a small set of shape buckets (e.g., sequence lengths rounded up to 128, 256, 512), autotune each bucket once, and dispatch by bucket at runtime. You pay a little wasted compute on the padding and a few autotune runs instead of one, in exchange for keeping the autotuned speed. If even bucketing is infeasible — truly arbitrary shapes — lean on a shape-agnostic runtime kernel and accept the fallback performance.

**What happens when the autotuned kernel regresses after a driver or library update?** It can. The kernel was tuned against a specific hardware/driver behavior, and a vendor update can shift the optimum. The discipline is to treat the tuning log as a build artifact under version control, re-tune on toolchain bumps as part of CI, and keep a non-autotuned fallback path so a regression degrades gracefully instead of breaking. This operational tax is real and is part of the honest cost of choosing a compiler over a runtime library.

**What happens when the cost model is wrong about your hardware?** The cost model is trained on features and measurements, and a brand-new accelerator with no measurement history is a domain it has never seen. Early in a tuning run on novel silicon, the model's top-k picks can be no better than random, which is why the first few rounds spend their measurement budget seemingly inefficiently — they are *bootstrapping* the model from zero knowledge of this target. The practical consequences are two. First, budget more trials for a never-before-tuned target than for a familiar one, because the model needs more measurements to become useful. Second, if you have a pre-trained cost model from a *similar* target (same SIMD width, similar cache hierarchy), transfer it as a warm start — Ansor and similar systems support seeding the model, and a related-target warm start can cut the trials-to-convergence substantially. The failure mode to avoid is concluding "autotuning does not help on this chip" after a too-small budget on a cold model; the curve has not had a chance to descend.

**What happens at the boundary between two compilers?** Real pipelines mix tools — `torch.compile` in training, an export handed to TVM or IREE for the edge build — and the seams are where surprises live. An op that one compiler fuses, the other may not; a layout one prefers (NCHW versus NHWC), the other fights. The discipline is to *measure the handoff*, not assume it: export, compile with the edge tool, and benchmark the full graph end to end on the device, watching specifically for unexpected layout-transform ops or device round-trips inserted at the boundary. More than once I have seen a clean per-kernel win erased by a layout transpose the export silently introduced, which only the whole-graph profile revealed. The kernel is never the whole story; the graph around it is.

## Case studies: real numbers from the literature and the field

Concrete results, with sources, so the claims above are not just my anecdotes.

**TVM beating vendor libraries on the edge (Chen et al., 2018).** The original TVM paper reported that autotuned kernels matched or exceeded hand-tuned vendor libraries across a range of targets, with the largest wins exactly where you would predict: on the ARM Mali GPU and on server GPUs for *non-standard* operator shapes, where the vendor's hand-tuned coverage was thin. For several workloads on the Mali GPU, TVM's generated kernels delivered multi-x speedups over the available baselines, validating the core thesis that search beats fixed heuristics on the long tail of shapes and targets.

**Ansor's auto-scheduler (Zheng et al., 2020).** Ansor reported outperforming the best existing systems — including hand-written templates (AutoTVM) and vendor libraries — by meaningful margins (often 1.2x to 3.8x depending on the workload and target) *and* doing it with less engineering effort, because no human had to write the schedule templates. The win came from a larger, automatically generated search space than templates could express, plus the cost-model-guided search exploring it efficiently. This is the paper that made template-free autotuning the default.

**torch.compile in the field (PyTorch 2.0).** PyTorch's own benchmarks across a large suite of models reported geometric-mean inference speedups in the ~1.3x-2x range on common GPUs from `torch.compile` alone, with no model changes — the fusion-plus-Triton-codegen win, available behind one line of code. The wins are largest on models with lots of pointwise glue around the matmuls (transformers, models with heavy normalization), exactly as the fusion analysis predicts, and smallest on models that are already a single dominant compute-bound op.

**XLA fusion on TPU/GPU.** XLA's whole-graph fusion is the reason JAX and TF-on-TPU achieve high hardware utilization on transformer training and inference: by fusing the elementwise glue (activations, layer norms, bias adds) into the matmul kernels' epilogues, it keeps the memory-bound work from bottlenecking the compute-bound matmuls. The documented effect is fewer, larger kernels with far less DRAM traffic — the $24M \to 16M$-bytes-and-shrinking analysis above, scaled to a whole model.

**FlashAttention as a hand-fused kernel.** Though not a "compiler" output, FlashAttention is the canonical proof of the fusion thesis at the kernel level: by fusing the attention softmax and the two matmuls into a single kernel that never materializes the $O(n^2)$ attention matrix in DRAM, it turns a memory-bound op into a compute-bound one and delivers large speedups and memory savings. It is exactly the kind of fused kernel you would write in Triton, and it shows the ceiling of what hand-fusion (and, increasingly, what compilers that can discover such fusions) can achieve.

## Key takeaways

- **A runtime ships kernels; a compiler generates and autotunes them.** Use the runtime's hand-written kernels for common ops on supported hardware; reach for a compiler for novel ops, unusual hardware, off-fast-path shapes, or last-mile latency.
- **The Halide insight is the whole field: separate algorithm from schedule.** The algorithm fixes the result; the schedule (loop order, tiling, vectorization, parallelism, caching) fixes the speed, and the schedule space is where an order of magnitude of latency lives — safely searchable because every schedule gives identical results.
- **Tiling wins by raising arithmetic intensity by roughly the tile dimension**, walking a memory-bound kernel up the slanted roof onto the flat compute roof — a 64-wide tile lifts intensity from ~0.25 to ~16+ FLOP/byte and turns a ~540 ms naive matmul into ~43 ms.
- **TVM/Ansor searches the schedule space** with a learned cost model plus on-device measurement, which beats fixed heuristics because it measures the real hardware across millions of candidates while timing only hundreds.
- **MLIR is infrastructure, not a product:** a stack of dialects with progressive, verifiable lowering, reused across frontends (Torch-MLIR) and backends (IREE), which is why it underpins the modern ecosystem.
- **XLA optimizes the graph (fusion); schedule-search compilers optimize the kernel.** They are complementary; fusion cuts DRAM round-trips and launches, autotuning cuts per-kernel time.
- **`torch.compile` is the one-line first move** for PyTorch (Inductor + Triton), `jax.jit` for JAX; try the easy win before the heavy TVM/IREE pipeline.
- **Autotuning is build-time cost with diminishing returns:** ~70-80% of the win arrives in the first few hundred trials; budget ~800-1500 trials on the dominant tasks and stop at the plateau, and only autotune the kernels that actually dominate latency.

## Further reading

- **Ragan-Kelley, Adams, Paris, Levoy, Amarasinghe, Durand (2013), "Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation in Image Processing Pipelines"** — the algorithm/schedule separation that started it all.
- **Chen, Moreau, Jiang, et al. (2018), "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning"** (OSDI) — the system that made ML schedule search general and practical.
- **Zheng, Jia, Sun, et al. (2020), "Ansor: Generating High-Performance Tensor Programs for Deep Learning"** (OSDI) — template-free auto-scheduling via an automatically generated search space.
- **Lattner, Amini, Bondhugula, et al. (2021), "MLIR: Scaling Compiler Infrastructure for Domain Specific Computation"** (CGO) — the multi-level IR and dialect design.
- **Tillet, Kung, Cox (2019), "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations"** — Python-authored, block-level GPU kernels.
- **Official docs:** the TVM documentation (auto-scheduling and `tvm.relay`), the XLA / OpenXLA documentation (HLO and fusion), the IREE documentation (MLIR-based edge deployment), and the PyTorch `torch.compile` documentation (TorchInductor and `max-autotune`).
- **Within this series:** [the taxonomy of model compression and the four-lever Pareto frame](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression), [inference runtimes compared](/blog/machine-learning/edge-ai/inference-runtimes-compared), [graph-level optimization](/blog/machine-learning/edge-ai/graph-level-optimization), [the roofline model](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives), and the capstone [the edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook).
