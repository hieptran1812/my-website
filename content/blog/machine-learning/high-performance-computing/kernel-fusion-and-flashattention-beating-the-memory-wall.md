---
title: "Kernel Fusion and FlashAttention: Beating the Memory Wall"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Most deep-learning ops are memory-bound, so the win is fewer trips to HBM. Learn how operator fusion and FlashAttention cut HBM traffic and run two to four times faster."
tags:
  [
    "high-performance-computing",
    "gpu",
    "kernel-fusion",
    "flashattention",
    "torch-compile",
    "triton",
    "attention",
    "deep-learning",
    "ml-systems",
    "memory-bandwidth",
  ]
category: "machine-learning"
subcategory: "High Performance Computing"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/kernel-fusion-and-flashattention-beating-the-memory-wall-1.png"
---

Here is a moment that converts skeptics. You profile a Transformer training step on an A100 and you find that the big matrix multiplies — the projections, the feed-forward, the attention `QK^T` — are humming along near peak. They are not the problem. The problem is the *glue*: the LayerNorms, the GELUs, the dropouts, the residual adds, the bias adds. Individually each one looks trivial. Together they eat 40% of your step time. You stare at the trace and ask the obvious question: how can a handful of elementwise operations, which do almost no arithmetic, cost more than the matmuls that do the actual work?

The answer is the memory wall, and once you see it you cannot un-see it. A GELU on an A100 does roughly one floating-point operation per element. To do that one operation, the GPU has to read the element from off-chip memory and write the result back. The arithmetic takes a few picoseconds. The memory round-trip takes hundreds of nanoseconds. The chip is not compute-starved; it is *bandwidth-starved*. It spends almost all of its time waiting for bytes to arrive from a slow pool of memory called HBM, and almost none of its time computing. We call such an operation **memory-bound**: its speed is set by how fast you can move bytes, not by how fast you can do math.

This post is about the single most important lever for memory-bound deep learning: **moving fewer bytes**. We will do it two ways. First, **operator fusion** — taking a chain of small elementwise ops and running them as one kernel, so the intermediate results live in fast on-chip registers instead of being shipped back and forth to HBM. We will see how `torch.compile` does this automatically and read the Triton kernel it generates. Second, the crown jewel of IO-aware deep learning: **FlashAttention**, which computes exact attention without ever writing the giant `N×N` score matrix to memory, dropping HBM traffic from quadratic to roughly quadratic-divided-by-on-chip-memory and running two to four times faster while using *linear* instead of quadratic memory.

The discipline throughout is the same, and it is the discipline this whole series is built on: **count the bytes**. We will count HBM bytes for a fused versus unfused LayerNorm-plus-GELU chain and derive the speedup. We will count HBM bytes for naive versus FlashAttention and derive why one is quadratic and the other is not. Numbers first, then the formulas that explain them, then runnable code, then measured before-and-after results on named hardware. By the end you will know exactly what `torch.compile` fuses, why FlashAttention works, and when neither is worth your time. If you want the broader frame — the three walls of compute, memory bandwidth, and communication — start with [why HPC is the bottleneck for modern AI](/blog/machine-learning/high-performance-computing/why-hpc-is-the-bottleneck-for-modern-ai) and the [roofline model](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound).

![Diagram contrasting an unfused pointwise chain with five HBM round-trips against a fused version with a single round-trip](/imgs/blogs/kernel-fusion-and-flashattention-beating-the-memory-wall-1.png)

Look at the figure above. On the left, an unfused chain of `add`, `mul`, and `gelu` reads from and writes to HBM at every step — five round-trips for three operations, because each kernel must load its input and store its output. On the right, the fused version reads the input once, does all three operations in registers, and writes the output once. Same math, a fifth of the memory traffic. That picture is the entire post in miniature. Now let us make it rigorous.

## Why most deep-learning ops are memory-bound

Let me define the terms precisely, because the whole argument rests on them. **HBM** (High Bandwidth Memory) is the GPU's main memory — the 40 or 80 gigabytes on an A100, the pool your tensors live in. It is fast by CPU standards (an A100 delivers about 2.0 TB/s, an H100 SXM about 3.35 TB/s) but it is *off-chip*, and getting a byte from it takes hundreds of nanoseconds of latency and consumes precious bandwidth. **SRAM** (the on-chip shared memory and registers) is tiny — tens of kilobytes to a few hundred kilobytes per streaming multiprocessor — but it is roughly an order of magnitude faster in bandwidth and far lower latency. The whole game of GPU performance is keeping data in SRAM and registers as long as possible and touching HBM as little as possible. If that hierarchy is new to you, the [memory hierarchy post](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm) walks through every level.

Now the key quantity. **Arithmetic intensity** is the ratio of floating-point operations performed to bytes moved from HBM:

$$I = \frac{\text{FLOPs}}{\text{bytes moved from HBM}}$$

An operation is **compute-bound** when its arithmetic intensity is high enough that the math, not the memory, is the limit — there is so much work per byte that the compute units stay busy while bytes trickle in. It is **memory-bound** when intensity is low — the chip finishes the math instantly and sits idle waiting for the next bytes. The crossover point is a property of the hardware: it is the ratio of the chip's peak FLOP/s to its peak HBM bandwidth, the "ridge point" of the roofline. For an A100 in bf16 that is roughly $312 \times 10^{12} / (2.0 \times 10^{12}) \approx 156$ FLOPs per byte. Any kernel below that intensity is memory-bound on an A100.

Where do common ops fall? A matmul of two $N \times N$ matrices does $2N^3$ FLOPs and moves about $3N^2$ elements (two inputs, one output). Its intensity is $\sim 2N^3 / (3N^2 \cdot 2) = N/3$ in bf16 — it grows with $N$, so for any reasonably large matmul it sails past the ridge and is comfortably compute-bound. That is why your matmuls run near peak.

A GELU, by contrast, reads one element (2 bytes in bf16) and writes one element (2 bytes), and does maybe a handful of FLOPs per element. Its intensity is on the order of $\sim 8 / 4 = 2$ FLOPs per byte — about *eighty times* below the A100 ridge. It is hopelessly memory-bound. So is `add`, so is `mul`, so is the bias add, so is dropout, so is the residual add. Every pointwise op in your network is memory-bound by a wide margin, and every reduction (LayerNorm, softmax) is only slightly better because it still reads and writes the whole tensor for a few FLOPs per element.

This is the structural fact that motivates everything else. Your network is a sandwich: compute-bound matmuls separated by layers of memory-bound elementwise glue. The matmuls are near peak and you cannot speed them up much. But the glue is wasting bandwidth, and bandwidth is the bottleneck. The way to make the glue faster is not to do less math — there is barely any math — it is to *move fewer bytes*. And the way to move fewer bytes is fusion.

#### Worked example: is your LayerNorm memory-bound on an A100?

Take a LayerNorm over a tensor of shape `[batch=8, seq=2048, hidden=4096]` in bf16. That is $8 \times 2048 \times 4096 = 6.7 \times 10^7$ elements. LayerNorm does roughly 10 FLOPs per element (subtract mean, divide by std, scale, shift, plus the reduction arithmetic), so about $6.7 \times 10^8$ FLOPs total. Bytes moved: it reads the input (2 bytes/element) and writes the output (2 bytes/element), so $\sim 4 \times 6.7 \times 10^7 = 2.7 \times 10^8$ bytes, ignoring the tiny mean/var passes. Arithmetic intensity: $6.7 \times 10^8 / 2.7 \times 10^8 \approx 2.5$ FLOPs/byte. The A100 ridge is ~156. So LayerNorm runs at roughly $2.5/156 \approx 1.6\%$ of the chip's compute capability. It is bandwidth-limited. Its runtime is set entirely by `bytes / 2.0 TB/s` $= 2.7 \times 10^8 / 2.0 \times 10^{12} \approx 135$ microseconds. The compute is free; the memory is the whole cost.

### The running example: one Transformer block, op by op

It helps to anchor everything in a single concrete workload and return to it the whole way through, so let me set up the spine for this post: **one Transformer block** running a forward pass on an A100, with a batch of 8, sequence length 2048, hidden size 4096, 32 heads of head dimension 128, in bf16. A block is two sub-layers — attention and the feed-forward network (FFN) — each wrapped in a LayerNorm and a residual add. Walk through the ops and sort each into compute-bound or memory-bound:

- **The QKV projection** (`x @ W_qkv`): a matmul, `[8·2048, 4096] @ [4096, 3·4096]`. Compute-bound, runs near peak. *Fast, not our problem.*
- **The attention itself** (`QK^T`, softmax, `PV`): two matmuls (compute-bound) bracketing a softmax over the `N×N` scores. The matmuls are fine; the softmax and the score-matrix traffic are the memory wall — this is where FlashAttention earns its keep.
- **The output projection** (`@ W_o`): another matmul. Compute-bound.
- **The two LayerNorms**: memory-bound reductions, ~135 µs each by the count above.
- **The FFN** (`x @ W_1`, GELU, `@ W_2`): two big compute-bound matmuls with a memory-bound GELU sandwiched between them. The GELU operates on a `[8, 2048, 16384]` tensor (the FFN expands 4×), which is `2.1 × 10^8` elements — a *huge* memory-bound op, moving ~0.9 GB just to read and write.
- **The residual adds, the dropouts, the bias adds**: all memory-bound pointwise ops, each a full tensor pass.

Add up the memory-bound pieces and you find that a meaningful fraction of the block's wall-clock — often 30–45% on real traces — is spent in operations that do essentially no arithmetic. That is the budget fusion and FlashAttention are going after. Keep this block in your head; every section below is a different way of clawing back the bandwidth those memory-bound ops are wasting. The matmuls we leave alone — they are already near peak, and as the [roofline model](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) shows, there is nothing to recover from a kernel that already sits on the compute roof.

There is a second-order reason the glue matters even more than its raw byte count suggests: **kernel launch overhead**. Every separate kernel costs a few microseconds of CPU-side launch latency, and a Transformer block unfused can be dozens of tiny kernels. At a few microseconds each, the launches alone can add tens of microseconds of pure overhead per block — and across dozens of layers and thousands of steps that is real time. Fusion collapses dozens of launches into a handful, which cuts both the HBM traffic *and* the launch overhead. On small batch sizes, where each kernel does little work, launch overhead can actually dominate, and fusion (or CUDA graphs, a related technique) is the only way to claw it back. We will keep both effects in view: fewer bytes *and* fewer launches.

## Operator fusion: load once, compute, store once

Here is the definition we have been circling. **Operator fusion** is the technique of combining several operations into a single GPU kernel so that intermediate results never leave the chip. Instead of one kernel per op — each loading its input from HBM and storing its output back — a fused kernel loads the input *once*, performs the whole chain of operations on values held in registers (the fastest, closest storage there is), and stores the final output *once*. The intermediate tensors are never **materialized**, which is the term of art for actually writing a tensor out to memory so that it physically exists as bytes in HBM.

Why does this help so much? Because for memory-bound ops, runtime is dominated by HBM traffic, and fusion attacks the traffic directly. Consider a chain of `k` pointwise operations on a tensor with `E` elements at `b` bytes each. Unfused, each op reads `E·b` and writes `E·b`, so the chain moves `k · 2 · E · b` bytes — every intermediate is written and then read back by the next op. Fused, the chain reads the input once and writes the output once: `2 · E · b` bytes. The traffic ratio is exactly `k`. A chain of 5 pointwise ops moves 5× less data when fused, and since these ops are memory-bound, ~5× less data means ~5× faster. That is the whole magic, and it is just byte counting.

Let me make the byte count concrete for the canonical glue in a Transformer block: a **LayerNorm followed by a GELU** (and let us throw in a residual add and a scale to make it a realistic chain). Take the same tensor as before, `[8, 2048, 4096]` bf16, so `E = 6.7 × 10⁷` elements, `b = 2` bytes, `E·b ≈ 134 MB` for a single read or write of the tensor.

![Matrix comparing HBM bytes moved by an unfused versus fused LayerNorm and GELU block with a three times reduction](/imgs/blogs/kernel-fusion-and-flashattention-beating-the-memory-wall-2.png)

The figure quantifies it. Conceptually, the unfused path is the failure mode: LayerNorm reads the tensor and writes a normalized tensor; GELU reads that normalized tensor and writes an activated tensor; the residual add reads two tensors and writes one. Each materialized intermediate is a full pass through HBM. Count it as round-trips of the tensor: LayerNorm = 2 (read in, write norm), GELU = 2 (read norm, write act), residual add ≈ 3 (read act, read residual, write out). That is 7 tensor-passes, about `7 × 134 MB ≈ 940 MB` of HBM traffic. The fused kernel reads the input and the residual once and writes the output once: about 3 passes, `3 × 134 MB ≈ 400 MB`. (The simplified figure uses a cleaner 24-vs-8 framing of a three-op chain to keep the picture readable; the LayerNorm-plus-GELU-plus-residual chain here is the same story with more terms.) The ratio is the win: roughly **2.3× less HBM traffic**, hence roughly 2.3× faster for this memory-bound block. Add more pointwise ops to the chain — bias, dropout, another scale — and the ratio climbs toward the number of ops you fuse.

![Stack diagram showing a fused kernel that loads from HBM once, runs the elementwise chain in registers, and stores to HBM once](/imgs/blogs/kernel-fusion-and-flashattention-beating-the-memory-wall-3.png)

The stack above is the mechanism. Think of the fused kernel as a single pass: it streams a chunk of the input from HBM into registers, applies `add`, then `mul`, then `gelu` while the values sit on-chip with essentially zero access latency, and only then writes the finished chunk back to HBM. The intermediate `add` result and `mul` result are never tensors in memory — they are transient values in a register, alive for a few instructions and then overwritten. Under the hood, the GPU's massive thread parallelism means thousands of these chunks are in flight at once, so the chip stays busy issuing loads and stores; the point is just that it issues *one* load and *one* store per element instead of `k` of each.

#### Worked example: the fusion speedup, end to end

Suppose your Transformer block's elementwise glue — across all the LayerNorms, GELUs, biases, dropouts, residuals — moves 2.0 GB of HBM traffic per forward pass when unfused. On an A100 at 2.0 TB/s effective bandwidth, that is `2.0 GB / 2.0 TB/s = 1.0 ms` just for the glue, assuming you hit peak bandwidth (you will hit maybe 80%, so call it ~1.25 ms). Now fuse the chains. Say fusion collapses the traffic by 2.5× to 0.8 GB. New glue time: `0.8 GB / 2.0 TB/s = 0.4 ms` (~0.5 ms at 80% peak). You just cut roughly 0.75 ms off every step. If your matmuls take 2.0 ms, your step went from ~3.25 ms to ~2.5 ms — a ~1.3× end-to-end speedup *for free*, no accuracy change, no new hardware. That is why fusion is the first lever you reach for after mixed precision.

### Why registers, and not shared memory or L2

It is worth being precise about *where* the intermediates live in a fused kernel, because the GPU memory hierarchy has several levels and the win depends on which one you hit. From fastest to slowest: **registers** (per-thread, single-cycle access, but a scarce resource — a streaming multiprocessor has a fixed register file split across all its threads), **shared memory / SRAM** (per-block, on-chip, low-latency, used for thread cooperation), **L2 cache** (a few megabytes, shared across the chip), and finally **HBM** (the off-chip pool). A pointwise fusion keeps each element's intermediate values in *registers* — the fastest tier — because each thread owns its element and never needs a neighbor's value. There is no cross-thread sharing, so shared memory is not even needed. The intermediates are born, used, and discarded entirely within the register file, which is why a fused pointwise chain has essentially zero extra memory cost beyond the single load and store. A reduction like softmax is different: threads must combine their partial sums, so the fused kernel stages data through *shared memory* for the cross-thread reduction. That extra cooperation is exactly why reductions fuse only partially — there is real data movement on-chip, even if it never reaches HBM.

A subtlety the byte count alone hides: fusion also reduces pressure on the **L2 cache** and the memory controllers. When the unfused chain writes an intermediate and the next op reads it back, a chunk of that intermediate may still be sitting in L2, so some "round-trips" are served from L2 rather than HBM and are cheaper than the worst case. This is why a measured fusion speedup is sometimes a bit *less* than the naive round-trip ratio predicts — L2 already recovered part of the traffic. But for large tensors that overflow the few-megabyte L2 (and a `[8, 2048, 16384]` FFN activation is hundreds of megabytes, far bigger than L2), the intermediates spill to HBM and you pay the full traffic, so the byte count is the right first-order model. The takeaway: count HBM bytes as the upper bound on the win, expect the measured speedup to land somewhere between that and 1× depending on how much L2 was already helping, and trust the CUDA-event measurement over the back-of-envelope number when they disagree.

## How torch.compile fuses pointwise chains

You do not have to write fused kernels by hand. Since PyTorch 2.0, `torch.compile` does it for you, and the component responsible is **TorchInductor**, the default backend. The flow has three stages, and it is worth knowing each because it tells you what *will* and *will not* fuse.

![Timeline of the torch.compile flow from Dynamo graph capture through Inductor fusion to generated Triton kernel](/imgs/blogs/kernel-fusion-and-flashattention-beating-the-memory-wall-6.png)

As the timeline shows, compilation runs as a small pipeline. First, **TorchDynamo** traces your Python function and captures it as an FX graph — a clean dataflow graph of the operations, with the Python control flow resolved. Second, **TorchInductor** lowers that graph to a loop-level intermediate representation and runs a *scheduler* that groups operations into fusion regions: it walks the graph and merges adjacent pointwise (and compatible reduction) ops that share the same iteration space, so they can become one kernel. Third, Inductor performs **code generation** — for the GPU it emits **Triton**, a Python-embedded language for writing GPU kernels, and compiles that to a single fused kernel. The net result is fewer, bigger kernels and far fewer HBM round-trips. (If you want to go deeper on writing GPU kernels yourself, the [CUDA programming post](/blog/machine-learning/high-performance-computing/cuda-programming-for-ai-engineers-threads-blocks-and-a-first-kernel) builds a first kernel from scratch.)

Let us actually see it. Here is an eager-mode function and its compiled version. We will turn on `TORCH_LOGS=output_code` to dump the kernel Inductor generates.

```python
import torch

def glue(x, w, b):
    # a realistic pointwise chain: scale, bias, gelu, residual
    y = x * w
    y = y + b
    y = torch.nn.functional.gelu(y)
    return y + x

x = torch.randn(8, 2048, 4096, device="cuda", dtype=torch.bfloat16)
w = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
b = torch.randn(4096, device="cuda", dtype=torch.bfloat16)

# eager: four separate kernels, four sets of HBM round-trips
y_eager = glue(x, w, b)

# compiled: Inductor fuses the whole chain into one Triton kernel
glue_c = torch.compile(glue)
y_comp = glue_c(x, w, b)   # first call traces + compiles
torch.testing.assert_close(y_eager, y_comp, rtol=1e-2, atol=1e-2)
```

Run it with the logging env var to see the generated code:

```bash
TORCH_LOGS=output_code python glue.py
```

You will see Inductor print a Triton kernel that looks roughly like this (trimmed and lightly annotated — the exact names and tiling vary by version and shape):

```python
import triton
import triton.language as tl

@triton.jit
def triton_fused_mul_add_gelu_add(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
                                  xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = xindex < xnumel
    # ---- ONE load of x from HBM ----
    x0 = xindex % 4096
    tmp0 = tl.load(in_ptr0 + xindex, xmask)          # x
    tmp1 = tl.load(in_ptr1 + x0, xmask)              # w (broadcast)
    tmp2 = tl.load(in_ptr2 + x0, xmask)              # b (broadcast)
    # ---- the whole chain, in registers ----
    tmp3 = tmp0 * tmp1                                # x * w
    tmp4 = tmp3 + tmp2                                # + b
    tmp5 = 0.5 * tmp4 * (1.0 + tl.math.erf(tmp4 * 0.70710678))  # gelu
    tmp6 = tmp5 + tmp0                                # + x (residual)
    # ---- ONE store to HBM ----
    tl.store(out_ptr0 + xindex, tmp6, xmask)
```

This is the payoff made visible. Four PyTorch operations — `mul`, `add`, `gelu`, `add` — became **one** Triton kernel with **one** `tl.load` of the big tensor and **one** `tl.store`. The intermediate `tmp3`, `tmp4`, `tmp5` are registers, never HBM tensors. Eager mode would have launched four kernels and moved the full tensor through HBM roughly eight times (four reads, four writes). The fused kernel moves it through about twice. That is the ~4× traffic reduction you can read straight off the source.

A few practical notes that save you grief. The first compiled call is slow because it traces and compiles; benchmark the *steady state*, not the first iteration. Recompilation triggers on changing input shapes unless you compile with `dynamic=True`, so a data loader that yields ragged batch sizes can thrash the compile cache. And `torch.compile` does not fuse everything — it cannot fuse across a matmul (more on why in the last section), and it fuses reductions only partially. But for the long pointwise chains that dominate the glue in a Transformer, it is close to the hand-written ideal, and it is one line of code.

### What the Inductor scheduler is actually deciding

It is worth understanding the *decision* the scheduler makes, because it explains both the wins and the limits. Inductor models each operation as a set of loop nests over an iteration space (think: "for each of the 67 million elements, do this"). Two ops can fuse if their iteration spaces are compatible — the same shape, or one broadcastable into the other — and if the data dependency is *local*, meaning the consumer's element only needs the producer's corresponding element (the pointwise case) rather than a whole row (the reduction case). The scheduler greedily merges compatible adjacent ops into fusion groups, subject to a budget: too many ops in one kernel and you run out of registers (register *spilling* to local memory, which is secretly HBM, would undo the win), so the scheduler caps group size. It also has to respect *memory aliasing* — if an op writes in place over a tensor another op still needs, they cannot freely reorder. The upshot for you as a user: long *chains* of pointwise ops fuse beautifully; a pointwise op separated from the next by a matmul or a shape change starts a new group; and very wide fusions can hit the register ceiling and fuse less than you would hope. None of this needs your intervention in the common case, but when a compiled model speeds up less than expected, `torch._dynamo.explain(fn)(*args)` shows you the graph breaks and `TORCH_LOGS=fusion` shows you which groups formed and why.

### Graph breaks: the silent fusion killer

The single most common reason `torch.compile` underperforms is a **graph break** — a point where Dynamo cannot trace through your Python and splits the graph in two, compiling each half separately and falling back to eager at the seam. Anything Dynamo cannot symbolically reason about causes one: a `.item()` or `.tolist()` that forces a device-to-host sync, data-dependent control flow (`if x.sum() > 0:`), an unsupported custom op, printing a tensor's value, or calling into a library Dynamo does not understand. Each break is a fusion boundary — ops on opposite sides cannot fuse, and you eat the launch overhead and HBM round-trip at the seam. A model riddled with graph breaks compiles "successfully" but barely speeds up. The fix is to find them with `torch._dynamo.explain` (or `fullgraph=True`, which *errors* on the first break instead of silently degrading) and rewrite the offending Python — move the `.item()` out of the hot path, replace data-dependent branches with masked arithmetic, register custom ops properly. The discipline is the same as everywhere in this series: measure, find the seam, fix it, re-measure.

## Measuring fusion honestly with CUDA events

Before we trust any speedup, we have to measure it correctly, because GPU timing is full of traps. The CPU launches kernels *asynchronously* — `y = f(x)` returns before the GPU has finished — so timing with `time.time()` measures launch overhead, not execution. You must use **CUDA events**, which are timestamps recorded *on the GPU's own timeline*, and you must synchronize before reading them. You must also warm up (the first iterations pay one-time costs: caching allocators, autotuning, compilation) and time the steady state. Here is the pattern I use for every kernel comparison; the [profiling post](/blog/machine-learning/high-performance-computing/profiling-gpu-workloads-finding-the-real-bottleneck) goes deeper on doing this at scale.

```python
import torch

def bench(fn, *args, warmup=20, iters=100):
    # warm up: pay compilation / autotune / allocator costs once
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()                 # finish all pending work
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()                 # wait for the GPU to finish
    return start.elapsed_time(end) / iters   # milliseconds per call

x = torch.randn(8, 2048, 4096, device="cuda", dtype=torch.bfloat16)
w = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
b = torch.randn(4096, device="cuda", dtype=torch.bfloat16)

def eager(x, w, b):
    y = torch.nn.functional.gelu(x * w + b)
    return y + x

compiled = torch.compile(eager)

t_eager = bench(eager, x, w, b)
t_comp  = bench(compiled, x, w, b)
print(f"eager   : {t_eager:.3f} ms")
print(f"compiled: {t_comp:.3f} ms   ({t_eager/t_comp:.2f}x)")
```

On an A100 for this shape you should see something like `eager: ~0.62 ms` and `compiled: ~0.17 ms`, roughly a 3.5× speedup on the glue — close to the traffic ratio you would predict from counting round-trips, which is the sign you got the measurement right. (Exact numbers depend on driver, PyTorch version, and clocks; treat them as approximate and re-measure on your box.) Note the two `synchronize()` calls bracketing the timed region — without them you are timing the Python loop, not the GPU. This is the single most common GPU-benchmarking mistake, and it makes fusion look either magical or useless depending on which way the async lie cuts.

### A comparison table for the elementwise glue

| Path | Kernels launched | Tensor passes through HBM | A100 time (approx) | Speedup |
|---|---|---|---|---|
| Eager, unfused | 4 | ~8 | ~0.62 ms | 1.0× |
| `torch.compile` fused | 1 | ~3 | ~0.17 ms | ~3.5× |
| Hand-written Triton | 1 | ~3 | ~0.16 ms | ~3.9× |

The hand-written kernel barely beats Inductor here, which is the point: for pointwise chains, `torch.compile` gets you ~95% of the hand-tuned win for one line of code. Hand-writing Triton pays off when you need fusion patterns the compiler does not find — and the most important of those is attention, which is where we go next.

## A profiler-driven story: finding the fusable glue

Before attention, let me walk the realistic workflow, because "turn on `torch.compile`" is not how you should actually operate — you profile first, you find the memory-bound glue, *then* you fuse, and you measure the before-and-after. This is the loop the whole series preaches: profile, find the wall, fix, re-measure. Here it is end to end on the running Transformer block.

Start by profiling the eager forward pass with `torch.profiler` to see where time goes and which kernels are tiny memory-bound stragglers:

```python
import torch
from torch.profiler import profile, ProfilerActivity, schedule

model = build_transformer_block().cuda().to(torch.bfloat16)
x = torch.randn(8, 2048, 4096, device="cuda", dtype=torch.bfloat16)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=3, active=5),   # skip warmup iters
    record_shapes=True,
) as prof:
    for _ in range(10):
        out = model(x)
        prof.step()

# rank CUDA kernels by total device time
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
prof.export_chrome_trace("block_eager.json")   # open in chrome://tracing or Perfetto
```

The table that prints is the diagnostic. In the eager trace you will typically see the matmuls at the top by total time (they do the most work) but then a long tail of `elementwise_kernel`, `gelu`, `native_layer_norm`, `add`, and `dropout` entries — each individually small but collectively a large slice, and each a separate kernel launch with its own HBM round-trip. That long tail is your fusable glue. The Chrome/Perfetto trace makes it visual: zoom in between two matmuls and you see a picket fence of tiny kernels with gaps between them, the gaps being launch overhead and bandwidth stalls. (For the full profiling workflow — `nsys`, Nsight Compute, reading occupancy and the memory chart — see [profiling GPU workloads](/blog/machine-learning/high-performance-computing/profiling-gpu-workloads-finding-the-real-bottleneck).)

Now apply the fix — compile the block — and re-profile the same way:

```python
model_c = torch.compile(model)   # Inductor fuses the pointwise tail

# ... identical profiler block, writing "block_compiled.json" ...
```

In the compiled trace the picket fence collapses: the long tail of tiny elementwise kernels is replaced by a few fused Triton kernels, the gaps shrink, and the total device time for the non-matmul portion drops sharply. *That* before-and-after — two profiler tables side by side, the eager one with twenty memory-bound kernels and the compiled one with three — is the evidence that fusion worked, and it is exactly the artifact you want to attach to a performance PR. Do not trust a speedup you have not seen in a trace; a number without a trace is a number you cannot debug when it regresses.

#### Worked example: reading the glue fraction off a trace

Suppose the eager profiler table shows total CUDA time per step of 3.4 ms, of which the matmuls (QKV, output proj, two FFN matmuls) account for 2.0 ms and the elementwise-plus-LayerNorm-plus-softmax tail accounts for 1.4 ms — so 41% of the step is memory-bound glue. You compile. The new table shows 2.6 ms total: matmuls still 2.0 ms (unchanged, as expected — fusion does not touch them), the glue down to 0.6 ms. You recovered 0.8 ms of the 1.4 ms of glue (the rest is irreducible — the LayerNorm and softmax reductions still have to read and write their tensors, and the attention matmuls' epilogues). End-to-end: 3.4 → 2.6 ms, a 1.3× speedup, entirely from collapsing memory-bound kernels. That 1.3× is typical for a Transformer block where the matmuls already dominate; on a more elementwise-heavy model (lots of normalization, gating, activations) the fraction of fusable glue is higher and the speedup is larger.

## The attention bottleneck: materializing the N×N matrix

Now to the marquee example. **Attention** is the heart of the Transformer, and in its standard form it is a textbook case of the memory wall — except the wasted tensor is not a small intermediate, it is an enormous `N×N` matrix, where `N` is the sequence length. Let us write out what naive attention does, then count the bytes, then watch the memory explode.

Attention takes queries `Q`, keys `K`, and values `V`, each of shape `[N, d]` where `N` is sequence length and `d` is the head dimension. The computation is:

$$S = QK^\top \in \mathbb{R}^{N \times N}, \quad P = \text{softmax}(S), \quad O = PV \in \mathbb{R}^{N \times d}$$

The trouble is `S` and `P`. They are `N×N`. The two matmuls (`QK^T` and `PV`) are compute-bound and fine. But a naive implementation **materializes** `S` to HBM, then reads it back to compute the softmax, writes `P` to HBM, then reads `P` back to multiply by `V`. For `N = 8192`, the matrix `S` has $8192^2 = 6.7 \times 10^7$ elements — about 134 MB per head in bf16, and with many heads and a batch dimension you are writing and reading *gigabytes* of intermediate that exist only to be immediately consumed.

![Before and after diagram showing naive attention writing the N by N scores to HBM versus FlashAttention streaming tiles through SRAM](/imgs/blogs/kernel-fusion-and-flashattention-beating-the-memory-wall-4.png)

The figure draws the contrast. On the left, naive attention's three big HBM tensors: the scores `QK^T` written out as `N×N`, the softmax reading that `N×N` back, the multiply by `V` reading it yet again. On the right, the FlashAttention approach we are about to build: tile `Q`, `K`, `V`, stream them through SRAM, run an online softmax, and write only the final output `O`. The naive path's defining sin is that it materializes the score matrix; the flash path's defining trick is that it never does.

Let us count the HBM traffic for naive attention, because the count is the whole argument. Reading `Q`, `K`, `V` is `3Nd` elements. Writing `S` is `N²`. Reading `S` for softmax is `N²`, writing `P` is `N²`, reading `P` for the second matmul is `N²`. Writing `O` is `Nd`. The dominant terms are the four `N²` passes:

$$\text{HBM bytes (naive)} \approx (4N^2 + 4Nd) \cdot b = O(N^2 d^0)\text{-dominated} \approx O(N^2)$$

The `N²` term swamps everything for long sequences. And worse than the traffic is the **memory**: you have to *hold* the `N×N` matrix, so peak memory is $O(N^2)$. That is the cliff. At `N = 2048` the score matrix is ~16 MB per head — annoying. At `N = 8192` it is ~256 MB per head — painful. At `N = 32768` it is ~4 GB per head — and you OOM before you even reach long context. The quadratic memory of naive attention is the single biggest reason long-context models were hard before 2022.

#### Worked example: how big does the score matrix get?

Take a single attention head, bf16, and walk `N` up. The score matrix `S` is `N² × 2` bytes. At `N = 1024`: `1024² × 2 = 2.1 MB`. At `N = 4096`: `4096² × 2 = 34 MB`. At `N = 16384`: `16384² × 2 = 537 MB`. *Per head.* A model with 32 heads and a batch of 8 at `N = 16384` would need, naively, `537 MB × 32 × 8 ≈ 137 GB` just for the score matrices of one layer — more than an entire H100's 80 GB. That is not a slowdown; that is a hard wall. Naive attention simply cannot run long context. FlashAttention's whole reason for existing is to make that 137 GB disappear by never storing `S` at all.

## FlashAttention: tiling, online softmax, never materialize

The breakthrough — Dao, Fu, Ermon, Rudra, and Ré, *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (2022) — is to compute *exact* attention without ever materializing the `N×N` matrix. It is **IO-aware**, meaning the algorithm is designed around the cost of moving data between HBM and SRAM rather than around the FLOP count. The trick has two parts: **tiling** (process `Q`, `K`, `V` in blocks small enough to fit in SRAM) and an **online softmax** (compute the softmax incrementally as the blocks stream by, so you never need the whole row of scores at once).

![Grid diagram of FlashAttention tiling with Q tiles streaming across K and V tiles in SRAM while a running max and sum are maintained](/imgs/blogs/kernel-fusion-and-flashattention-beating-the-memory-wall-5.png)

The grid figure is the algorithm's shape. Outer loop: tiles of `Q` (block of `Br` rows). Inner loop: tiles of `K` and `V` (blocks of `Bc` columns). For each `(Q-tile, K-tile)` pair, load the tiles into SRAM, compute the small `Br × Bc` block of scores `S = QK^T` *in SRAM*, and use it to update a running output `O` and two small running statistics per query row — the running max `m` and the running normalizer `l`. The score block lives in SRAM for a few instructions and is discarded. The `N×N` matrix never exists. The way this works is that softmax, which looks like it needs the whole row, can in fact be computed incrementally if you carry the right running state.

### The online softmax recurrence — the heart of it

Softmax over a vector $x_1, \dots, x_n$ is $\text{softmax}(x)_i = e^{x_i - m} / \sum_j e^{x_j - m}$ where $m = \max_j x_j$ (we subtract the max for numerical stability — without it, $e^{x_j}$ overflows for large logits). The problem: the max and the sum need *all* the values, but tiling gives them to us a block at a time. The **online softmax** solves this with a running recurrence. Process the values in blocks. Maintain a running max $m$ and a running sum $l$. When a new block arrives with its own local max $m^{\text{new}}$ and local exponential sum, update:

$$m' = \max(m, m^{\text{new}})$$
$$l' = e^{m - m'} \, l + e^{m^{\text{new}} - m'} \, l^{\text{new}}$$

The key is the **rescaling factor** $e^{m - m'}$. When a new block raises the running max, every previously accumulated term was exponentiated against the *old* max, so we multiply the old running sum (and the old partial output) by $e^{m - m'}$ to re-base it to the new max. The output accumulator gets the same treatment:

$$O' = e^{m - m'} \, O + e^{m^{\text{new}} - m'} \, (P^{\text{new}} V^{\text{block}})$$

At the end, divide the accumulated output by the final $l$. The result is *bit-for-bit the same softmax* as the naive version (up to floating-point reordering) — FlashAttention computes exact attention, not an approximation. The genius is that the running state is just two scalars per query row (`m` and `l`) plus the output accumulator, all of which fit in SRAM. You never hold a full row of `N` scores.

#### Worked example: the online max-and-sum on three numbers

Make the recurrence concrete with $x = [1, 3, 2]$ processed one at a time. Start $m=-\infty$, $l=0$. See $x_1=1$: $m'=\max(-\infty,1)=1$, $l' = e^{-\infty}\cdot 0 + e^{1-1}\cdot 1 = 1$. See $x_2=3$: $m'=\max(1,3)=3$, rescale old sum by $e^{1-3}=e^{-2}\approx 0.135$, so $l' = 0.135 \times 1 + e^{3-3}\times 1 = 0.135 + 1 = 1.135$. See $x_3=2$: $m'=\max(3,2)=3$ (unchanged), $l' = e^{3-3}\times 1.135 + e^{2-3}\times 1 = 1.135 + 0.368 = 1.503$. Compare to computing it all at once: $m=3$, $\sum e^{x_i-3} = e^{-2}+e^{0}+e^{-1} = 0.135+1+0.368 = 1.503$. Identical. The recurrence reproduces the exact softmax denominator while only ever holding two running scalars — that is what lets the score matrix stay tiled in SRAM and never touch HBM.

### Deriving FlashAttention's HBM traffic

Now the payoff derivation — why the traffic drops from $O(N^2)$ to roughly $O(N^2 d^2 / M)$, where `M` is the SRAM size in elements. FlashAttention loads `Q`, `K`, `V` from HBM and writes `O`. The inner loop reloads `K` and `V` blocks for each `Q` block. The block sizes are chosen so a `Q`-block, a `K`-block, a `V`-block, and the working state all fit in SRAM of size `M`; concretely the block columns are set to about `Bc ≈ M / (4d)` so the tiles fit. The number of times `K` and `V` get reloaded scales with the number of `Q` blocks times the SRAM budget, and the careful accounting in the paper gives:

$$\text{HBM bytes (flash)} = O\!\left(\frac{N^2 d^2}{M}\right)$$

Compare to naive's $O(N^2 d)$ of *score-matrix* traffic (the dominant term; we wrote it as $O(N^2)$ for fixed `d` above). The ratio of naive to flash traffic is $\sim N^2 d / (N^2 d^2 / M) = M / d$. For an A100, `M` (usable SRAM per block) is on the order of $10^5$ elements and `d` is typically 64 or 128, so $M/d$ is in the tens — FlashAttention moves roughly an *order of magnitude* fewer bytes than naive attention. Since attention's elementwise/softmax part is memory-bound, an order of magnitude less traffic translates directly into the measured 2–4× wall-clock speedup (it is not the full $M/d$ because the two matmuls, which are compute-bound, are shared by both methods and set a floor).

And the memory result is even cleaner. FlashAttention never stores `S` or `P`, so its peak memory for the attention computation is just the inputs, the output, and the tiny running statistics:

$$\text{peak memory (flash)} = O(N d + N) = O(N)$$

linear in sequence length, versus $O(N^2)$ for naive. *That* is what unlocks long context — the 137 GB of score matrices from our earlier worked example simply never exist.

![Before and after diagram showing naive attention memory growing as N squared versus FlashAttention staying linear in N](/imgs/blogs/kernel-fusion-and-flashattention-beating-the-memory-wall-7.png)

The figure makes the asymptotics visceral. As `N` climbs from 2k to 8k to 32k, naive attention's memory goes 16 MB to 256 MB to 4 GB per head and eventually OOMs; FlashAttention's stays linear and keeps fitting. This is not a constant-factor win, it is a change in the *shape* of the curve — and that is why FlashAttention is effectively the enabling technology for every long-context model you use today.

#### Worked example: HBM traffic at seq 8192 on an A100

Let me put real bytes on the derivation. One attention head, `N = 8192`, `d = 128`, bf16 (2 bytes). **Naive** writes the score matrix `S` (`N² = 6.7×10⁷` elements, 134 MB), reads it back for softmax (134 MB), writes `P` (134 MB), reads `P` for the `PV` matmul (134 MB) — that is `~536 MB` of score-matrix traffic alone, on top of the small `Q`/`K`/`V`/`O` reads and writes (`4Nd = 4.2×10⁶` elements ≈ 8 MB). Call it ~544 MB per head. **FlashAttention** reads `Q`, `K`, `V` once and writes `O` once, plus reloads `K` and `V` blocks across the `Q` blocks. With an SRAM block budget that makes `K`/`V` reloaded roughly `N / Bc` times where `Bc` is the column-block size, the total works out to a few tens of megabytes — call it ~24 MB per head for these dimensions. The ratio is ~544 / 24 ≈ **23× less HBM traffic**. At the A100's 2.0 TB/s, naive's 544 MB takes ~270 µs of pure memory time for the score traffic; FlashAttention's 24 MB takes ~12 µs. The matmuls cost the same for both and set the floor, which is why the *end-to-end* speedup is the famous 2–4× rather than the full 23× — but the bandwidth saving is real and it is enormous.

### The backward pass and recomputation

There is a wrinkle the forward-pass story glosses over: the backward pass. Normally, to compute gradients you need the activations from the forward pass — and for attention that would mean keeping the `N×N` probability matrix `P` around, which would reintroduce the `O(N²)` memory you just eliminated. FlashAttention's answer is **recomputation** (a form of activation checkpointing): it does *not* store `P`; instead, during the backward pass it *recomputes* the score blocks on the fly from `Q`, `K`, `V`, using the stored running statistics (the per-row max and normalizer it saved from the forward pass) to reconstruct exactly the same softmax. This trades a bit of extra compute (the score blocks get computed twice, once forward and once backward) for keeping the backward pass at `O(N)` memory too. It is a beautiful trade: attention's matmuls are compute-bound and the GPU has spare FLOP headroom, so spending a few extra FLOPs to avoid storing and re-reading a giant matrix is exactly the right side of the memory wall to be on. The net result is that *both* directions of attention run at linear memory, which is what makes training long-context models — not just running inference on them — feasible.

### Stress-testing the result: when does FlashAttention not win?

Reason through the edges, because every optimization has them. **Very short sequences** (`N` in the low hundreds): the `N²` matrix is tiny, the matmuls dominate, and the tiling bookkeeping is overhead — FlashAttention is roughly a wash, neither helping nor hurting much. **Tiny head dimension or huge head dimension**: the `M/d` traffic ratio shrinks as `d` grows, so a very large head dim erodes the win (and head dims past the backend's supported maximum fall off the fast path entirely). **Exotic attention masks**: if your mask is not causal or one of the supported patterns, the fused kernel may not express it and you fall back to a slower path — arbitrary additive masks can cost you. **Non-contiguous or oddly-strided tensors**: the kernel wants specific memory layouts; a transpose or a weird view can force a copy or a fallback. **Tiny batch in inference (decode)**: at decode time with a batch of one and a single query token, attention is over a long KV cache but the query side is trivial, so the regime shifts to KV-cache bandwidth — which is why serving stacks pair FlashAttention with paged KV management (see [KV-cache optimization and management](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management)). The honest summary: FlashAttention is a near-universal win for training and prefill on long sequences, a wash on short ones, and one ingredient (not the whole answer) for the decode-time serving problem.

## Using FlashAttention from PyTorch

You almost never write FlashAttention yourself — it ships inside PyTorch. The function `torch.nn.functional.scaled_dot_product_attention` (SDPA) dispatches to a fused, IO-aware backend, and on supported hardware that backend *is* FlashAttention. Here is the call:

```python
import torch
import torch.nn.functional as F

# q, k, v: [batch, heads, seq_len, head_dim]
B, H, N, d = 8, 32, 8192, 128
q = torch.randn(B, H, N, d, device="cuda", dtype=torch.bfloat16)
k = torch.randn(B, H, N, d, device="cuda", dtype=torch.bfloat16)
v = torch.randn(B, H, N, d, device="cuda", dtype=torch.bfloat16)

# one call — fused, tiled, online-softmax attention. is_causal masks the future.
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
print(out.shape)   # [8, 32, 8192, 128]
```

That single call replaces the entire `QK^T` / softmax / `PV` sequence and never materializes the score matrix. You can inspect and constrain which backend it picks. PyTorch exposes a context manager to force or forbid specific kernels — useful when you want to *guarantee* the FlashAttention path or compare against the math fallback:

```python
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

# Force the FlashAttention backend (errors if unavailable for this shape/dtype)
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    out_flash = F.scaled_dot_product_attention(q, k, v, is_causal=True)

# The naive, materializing reference path for an apples-to-apples comparison
with sdpa_kernel(SDPBackend.MATH):
    out_math = F.scaled_dot_product_attention(q, k, v, is_causal=True)

torch.testing.assert_close(out_flash, out_math, rtol=2e-2, atol=2e-2)
```

The `MATH` backend is the naive implementation that materializes `S` — it is your baseline for measuring the speedup and for confirming FlashAttention gives the same answer (within bf16 tolerance). Note the constraints under which the flash backend engages: head dimension within supported bounds (typically ≤ 256), a dtype it supports (fp16/bf16), contiguous-enough layouts, and a mask it can express (causal and a few others, via `is_causal` or `attn_mask`). If your masking is exotic, SDPA silently falls back to the slower `MATH` or `EFFICIENT` path — so when performance matters, force the backend and let it error loudly rather than degrade silently. FlashAttention also composes with `torch.compile`: Inductor recognizes the SDPA call and keeps it as the fused flash kernel while fusing the surrounding pointwise glue around it.

### A Triton-style sketch of the attention tile

To cement how the tiling and online softmax fit together, here is a stripped-down sketch of the inner loop in Triton-flavored pseudocode (the real FlashAttention-2 kernel is far more tuned, but the skeleton is exactly this — outer loop over `Q` blocks, inner loop over `K`/`V` blocks, running `m`/`l`/`O`):

```python
import triton
import triton.language as tl

@triton.jit
def flash_attn_inner(Q_ptr, K_ptr, V_ptr, O_ptr, scale,
                     N, d, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # one program handles one block of BLOCK_M query rows
    q = tl.load(Q_ptr + ...)              # [BLOCK_M, d] into SRAM
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)   # running max
    l_i = tl.zeros([BLOCK_M], tl.float32)                 # running sum
    acc = tl.zeros([BLOCK_M, d], tl.float32)              # output accumulator

    for start_n in range(0, N, BLOCK_N):                  # stream K,V blocks
        k = tl.load(K_ptr + ...)          # [BLOCK_N, d] into SRAM
        v = tl.load(V_ptr + ...)          # [BLOCK_N, d] into SRAM
        s = tl.dot(q, tl.trans(k)) * scale                # [BLOCK_M, BLOCK_N] in SRAM
        m_new = tl.maximum(m_i, tl.max(s, axis=1))        # new running max
        p = tl.exp(s - m_new[:, None])                    # rescaled probs
        alpha = tl.exp(m_i - m_new)                       # rescale factor
        l_i = l_i * alpha + tl.sum(p, axis=1)             # update sum
        acc = acc * alpha[:, None] + tl.dot(p, v)         # update output
        m_i = m_new
    acc = acc / l_i[:, None]                              # final normalize
    tl.store(O_ptr + ..., acc)            # ONE write of [BLOCK_M, d]
```

Read the loop body against the online-softmax recurrence from earlier and you will see them line up term for term: `m_new` is $m'$, `alpha` is the rescaling factor $e^{m - m'}$, `l_i` is the running normalizer $l$, and `acc` is the running output `O`. The score block `s` is computed and consumed entirely inside SRAM, then thrown away. The only HBM writes are the final `acc` — one store of the `[BLOCK_M, d]` output tile. No `N×N` matrix, ever.

## Case studies / real numbers

Now the measured results — what these techniques actually buy on named hardware. I will flag every figure as approximate where I am reconstructing from the literature rather than citing an exact line, and I will not invent precise numbers.

**FlashAttention on A100 — the original paper.** Dao et al. (2022) report that FlashAttention trains BERT-large ~15% faster than the then-fastest baseline and gives ~3× end-to-end speedup on GPT-2 versus a standard HuggingFace implementation, with attention itself running several times faster and using *linear* instead of quadratic memory. The headline the community remembers is the 2–4× attention speedup with a large memory reduction — and crucially, it enabled training on longer sequences (they demonstrated context lengths that were previously infeasible) because the $O(N)$ memory removed the quadratic cliff.

**FlashAttention-2 — better GPU utilization.** Dao (2023), *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*, reworks the algorithm to reduce non-matmul FLOPs and parallelize across the sequence and warps. It roughly *doubles* FlashAttention-1's throughput, reaching on the order of 50–73% of the A100's theoretical max FLOP/s for attention — meaning attention, historically the memory-bound straggler, now runs at a respectable fraction of peak. On an A100 at long sequence lengths, FA-2 attention forward passes are commonly ~2× the speed of FA-1 and many times the speed of a naive materializing implementation.

**FlashAttention-3 — Hopper and FP8.** Shah, Bikshandi, Zhang, Dao et al. (2024), *FlashAttention-3*, targets the H100's Hopper architecture, overlapping the matmul (Tensor Core) and softmax (multifunction unit) work and exploiting the new asynchronous instructions and FP8 support. It reaches roughly 1.5–2× over FA-2 on H100, with reported utilization up to ~75% of the H100's bf16 peak (and higher absolute throughput in FP8). The trajectory is the story: each version squeezes attention closer to the compute roofline by being more IO-aware and more hardware-aware.

**Rough before-and-after on A100/H100.** The table below collects approximate numbers for attention at two sequence lengths to show the *shape* of the win — same arithmetic, an order of magnitude less HBM traffic, and linear memory. Treat the millisecond figures as illustrative order-of-magnitude estimates; your exact numbers depend on head count, batch, dtype, masking, and software version, so always re-measure with the CUDA-event harness above.

| Setup (one attention layer, bf16) | Backend | Peak memory | Time (approx) |
|---|---|---|---|
| seq 2k, 32 heads, A100 | naive (materializes S) | high, ~O(N²) | baseline |
| seq 2k, 32 heads, A100 | FlashAttention-2 | low, ~O(N) | ~2–3× faster |
| seq 8k, 32 heads, A100 | naive (materializes S) | very high, often OOM | (often will not fit) |
| seq 8k, 32 heads, A100 | FlashAttention-2 | low, ~O(N) | runs comfortably |
| seq 8k, 32 heads, H100 | FlashAttention-3 | low, ~O(N) | ~1.5–2× over FA-2 |

The most important cell in that table is "often will not fit." At seq 8k the naive score matrices blow past memory and the run is impossible, not just slow — FlashAttention does not merely speed up long context, it *enables* it. This is also why FlashAttention is the backbone of efficient LLM serving: combined with paged storage of the keys and values, it is what makes long-context inference tractable. For that serving side of the story, see [KV-cache optimization and management](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management) and the survey of [inference runtimes](/blog/machine-learning/edge-ai/inference-runtimes-compared).

**torch.compile fusion in production.** PyTorch's own benchmarks report `torch.compile` delivering meaningful end-to-end training speedups across a broad model suite — frequently in the 1.3–2× range for Transformer training, with the gains coming substantially from fusing the pointwise glue (and from picking better matmul kernels). For inference of memory-bound, pointwise-heavy models the speedup can be larger. The exact number is workload-dependent; the mechanism is the one we counted by hand — fewer kernels, fewer HBM round-trips.

**Long-context enablement — the qualitative win.** The most important "result" is not a speedup number at all; it is a capability that did not exist before. Pre-FlashAttention, training a model at sequence length 16k or 32k was effectively impossible on a single GPU because the score matrices blew the memory budget — you had to shard, approximate, or give up. FlashAttention's linear memory turned long context from a research stunt into a default. Essentially every modern long-context model — the 100k+ token context windows you take for granted — relies on FlashAttention-style IO-aware attention (or a close descendant) to fit the computation in memory. When you read that a model supports a 128k context, the unsung enabler is that its attention does not materialize a `128000 × 128000` matrix. That is the kind of win byte-counting buys you: not a faster version of the old thing, but the old thing becoming possible at all.

#### Worked example: the dollar cost of leaving fusion off

Put money on it. Suppose a training run takes 30 days on a 64-GPU A100 cluster and you are renting at roughly \$2 per GPU-hour. That is `64 × 24 × 30 × \$2 = \$92,160` for the run. Now suppose `torch.compile` plus FlashAttention together give a conservative 1.4× end-to-end speedup (fusion on the glue, FlashAttention on the attention) — entirely plausible for a Transformer with non-trivial sequence length. The run now finishes in `30 / 1.4 ≈ 21.4` days, costing `64 × 24 × 21.4 × \$2 ≈ \$65,700`. You saved roughly \$26,000 and a week and a half of wall-clock, by adding one `torch.compile` line and using the built-in SDPA backend. (Numbers are illustrative — rental rates and speedups vary — but the order of magnitude is real, and it is why fusion and IO-aware attention are not optional niceties at scale.)

## Complementary techniques: where fusion fits

Fusion and FlashAttention are two tools in a kit, and they compose with the rest of the single-GPU optimization stack — it is worth knowing the neighbors so you do not reach for fusion when a different lever is the real fix.

**Mixed precision (bf16/fp16)** is the lever you pull *before* fusion. It halves the bytes of every tensor, which directly halves HBM traffic for memory-bound ops — a 2× win on exactly the kernels fusion also targets. Fusion and mixed precision multiply: bf16 halves the bytes per pass, fusion cuts the number of passes, and you get both factors. Always do mixed precision first; it is one line (`autocast`) and it benefits everything.

**CUDA graphs** attack the *launch-overhead* half of the problem that fusion only partially addresses. A CUDA graph captures a sequence of kernel launches once and replays them as a single submission, eliminating the per-kernel CPU launch latency. For small-batch or short-sequence workloads where launch overhead dominates, CUDA graphs (exposed via `torch.compile(mode="reduce-overhead")` or `torch.cuda.CUDAGraph`) can matter as much as the byte savings. Fusion reduces the *number* of launches; CUDA graphs reduce the *cost* of the ones that remain. They stack.

**Activation checkpointing** is the memory-side cousin of FlashAttention's recomputation. When activations do not fit, you recompute them in the backward pass instead of storing them — trading compute for memory. FlashAttention's backward pass *is* a specialized, baked-in version of this idea for the attention block; you can apply the general form (`torch.utils.checkpoint`) to other memory-heavy sub-layers.

The decision rule that ties them together: profile to find the wall, then pick the lever that moves *that* wall. Compute-bound and near peak? None of these help much — you need better matmul kernels or more FLOP/s. Memory-bound on pointwise glue? Mixed precision, then fusion. Launch-overhead-bound (tiny kernels, small batch)? Fusion plus CUDA graphs. Out of memory on long context? FlashAttention plus checkpointing. Each lever has a wall it is for, and using the wrong one is wasted effort.

## Which ops fuse and which do not

To use fusion well, you need a mental model of *what the compiler can and cannot do*. The rule follows directly from data dependencies, and it sorts every op into one of three buckets.

![Matrix classifying pointwise ops as fully fusible, reductions as partially fusible, and matmul as its own kernel](/imgs/blogs/kernel-fusion-and-flashattention-beating-the-memory-wall-8.png)

The figure lays out the taxonomy. **Pointwise ops** — `add`, `mul`, `gelu`, `sigmoid`, bias, dropout, residual — fuse fully and freely. Each output element depends only on the corresponding input element(s), so a single thread can compute the whole chain for its element without ever needing data from a neighbor. There is no cross-element dependency, so there is nothing to stop the compiler from chaining them in registers. This is the easy, high-value case, and it is exactly what `torch.compile` excels at.

**Reductions** — `sum`, `mean`, `max`, `softmax`, `LayerNorm` — fuse *partially*. A reduction's output depends on a whole row (or the whole tensor), so the threads computing it must cooperate and share data, which constrains fusion: you can usually fuse a reduction with the pointwise ops that *feed* it (fuse the `mul` into the `sum`) and with pointwise ops that *consume* its result, but two reductions over different axes generally cannot collapse into one pass. The online-softmax trick in FlashAttention is precisely a hand-crafted way to make a reduction (softmax) fuse with its neighboring matmuls — that is why it took a research paper and not just a compiler pass.

**Matmul** — `QK^T`, the linear layers, the projections — does *not* fuse into the surrounding pointwise chain. Matmul is compute-bound and runs on specialized **Tensor Cores** with a carefully tiled, register-blocked schedule (the [memory hierarchy post](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm) covers why the tiling is what makes it fast). Fusing arbitrary elementwise work into that schedule would wreck the tiling that gives the matmul its speed, so compilers keep matmul as its own kernel and at most fuse a trailing pointwise op (a bias, an activation) into the *epilogue* — the final write-out stage of the matmul kernel. This epilogue fusion is real and valuable (it saves the bias/activation from being a separate memory-bound pass), but the core matmul stays separate. The practical upshot: do not expect the compiler to fuse across your matmuls, and do not be disappointed when it doesn't — it is keeping the compute-bound part fast.

| Op category | Examples | Fuses? | Reason |
|---|---|---|---|
| Pointwise | add, mul, gelu, dropout, residual | Fully | output element depends only on input element |
| Reduction | sum, mean, softmax, LayerNorm | Partially | output depends on a whole row; threads must cooperate |
| Matmul | linear, QK^T, projections | No (epilogue only) | Tensor-Core tiling; fusion would break the schedule |

Epilogue fusion deserves a second look, because it is where the matmul boundary is softest and where a lot of real-world wins hide. The classic pattern is `linear → bias → activation`: the bias add and the GELU are both memory-bound pointwise ops that, unfused, would each read the matmul's output from HBM and write it back. Epilogue fusion folds them into the matmul kernel's final write stage, so the matmul computes a tile of its result in registers, applies the bias and activation to that tile *before* it ever leaves the chip, and writes the activated result once. You get the matmul's compute-bound speed *and* you eliminate two memory-bound passes — the best of both. This is why a fused `Linear + GELU` (sometimes exposed as a single op, sometimes found automatically by Inductor) beats the three-kernel version even though the matmul itself is unchanged. The asymmetry to remember: you can fuse pointwise work *after* a matmul (the epilogue) far more easily than *before* it, because the matmul reads its inputs in a tiled pattern that arbitrary pre-fused pointwise work would disrupt. So structure your model to put cheap pointwise ops on the *output* side of a matmul where the compiler can absorb them, and do not expect miracles on the input side.

FlashAttention, viewed through this taxonomy, is the masterstroke: it takes the one pattern that should be *impossible* to fuse — two matmuls with a reduction (softmax) sandwiched between them — and fuses the whole thing by inventing the online-softmax recurrence that lets the reduction proceed incrementally inside the tiled matmul loop. It is fusion across a reduction, achieved not by a compiler pass but by a change of *algorithm*. That is the deeper lesson of this entire post: when the compiler cannot fuse something, sometimes the answer is not a cleverer compiler but a re-derived algorithm that has fusion built into its math. The compiler gives you the easy 80%; the research-paper-grade algorithm gives you the hard 20% that changes what is possible.

## When to reach for fusion and FlashAttention, and when not to

Every optimization is a cost, so let me be decisive about where these two pay and where they don't.

**Use FlashAttention essentially always for attention.** There is no downside: it is exact, it is faster, and it uses far less memory. If you are calling `F.scaled_dot_product_attention` you are likely already getting it; the only thing to watch is that exotic masking or an unsupported head dimension can silently drop you onto the slow `MATH` path, so when performance matters, force the backend with `sdpa_kernel` and let it error rather than degrade. The one place it does *not* help is very short sequences (say `N` in the low hundreds) where the `N²` matrix is small anyway and the matmuls dominate — there the tiling overhead can make it a wash. But you lose nothing by leaving it on.

**Use `torch.compile` when memory-bound glue is a real fraction of your step.** Profile first ([how to profile](/blog/machine-learning/high-performance-computing/profiling-gpu-workloads-finding-the-real-bottleneck)). If your trace shows lots of small elementwise kernels eating wall-clock between the matmuls, fusion will help a lot — that is the textbook win. If your trace shows you are already compute-bound (matmuls near peak, glue is a sliver), fusion buys little, because there is barely any memory-bound time to recover. Spending a day hand-writing a fused kernel to shave a memory-bound op that is 2% of your step is the classic mistake of optimizing the wrong wall — the [roofline](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) tells you which wall you are on, and you should let it.

**Watch the `torch.compile` failure modes.** Dynamic shapes cause recompilation thrash — compile with `dynamic=True` or pad to fixed shapes if your batch sizes vary. The first iteration is slow (compilation); never include it in your timing or your throughput math. And graph breaks (from data-dependent Python control flow, unsupported ops, or `.item()` calls that force a sync) fragment the graph and limit fusion — check `torch._dynamo.explain` if a model compiles but barely speeds up.

**Don't hand-write a kernel the compiler already fuses.** For a vanilla pointwise chain, `torch.compile` gets ~95% of the hand-tuned Triton win, as our table showed. Reach for hand-written Triton/CUDA only when you need a fusion the compiler can't find — a custom attention variant, a fused optimizer step, a domain-specific op — and when profiling proves it is worth the engineering and maintenance cost.

## Key takeaways

- **Most deep-learning ops are memory-bound.** Pointwise ops (GELU, add, bias, dropout) and even reductions (LayerNorm, softmax) do almost no math per byte, so their speed is set by HBM bandwidth, not compute. Count arithmetic intensity to confirm.
- **The win is fewer HBM round-trips, not less math.** Fusion runs a chain of `k` pointwise ops as one kernel, reading the input once and writing the output once — roughly `k×` less HBM traffic and, for memory-bound ops, roughly `k×` faster.
- **`torch.compile` fuses pointwise chains automatically.** TorchDynamo captures the graph, TorchInductor groups and fuses, and it codegens one Triton kernel. Read it with `TORCH_LOGS=output_code`; expect 1.3–2× end-to-end on Transformer training, more on memory-bound inference.
- **Measure with CUDA events and synchronize.** Warm up, `torch.cuda.synchronize()` before and after the timed region, and benchmark steady state. Wall-clock timing of async GPU launches is the most common benchmarking lie.
- **Naive attention materializes an `N×N` matrix** — `O(N²)` HBM traffic and, worse, `O(N²)` peak memory. That quadratic memory is the wall that made long context infeasible.
- **FlashAttention never materializes it.** Tiling plus an online-softmax recurrence (running max `m` and sum `l`, rescaled by `e^(m−m')`) compute exact attention from SRAM-sized tiles, dropping traffic from `O(N²d)` to `O(N²d²/M)` and peak memory from `O(N²)` to `O(N)` — 2–4× faster and linear memory.
- **Get FlashAttention via `F.scaled_dot_product_attention`.** Force the backend with `sdpa_kernel(SDPBackend.FLASH_ATTENTION)` when performance is critical so it errors instead of silently falling back to the materializing `MATH` path.
- **Pointwise fuses fully, reductions partially, matmul not at all.** The boundary follows data dependencies; matmul stays its own Tensor-Core kernel with at most an epilogue fusion. Don't fight it.
- **Profile before you fuse.** If you are already compute-bound, fusion buys little. Optimize the wall you are actually on.

## Further reading

- Dao, Fu, Ermon, Rudra, Ré — *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (2022). The original tiling + online-softmax IO-aware algorithm.
- Dao — *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning* (2023). Better warp/sequence parallelism, roughly 2× over FA-1.
- Shah, Bikshandi, Zhang, Dao et al. — *FlashAttention-3* (2024). Hopper-specific overlap of matmul and softmax, FP8 support.
- Milakov, Gimelshein — *Online normalizer calculation for softmax* (2018). The running-max-and-sum recurrence FlashAttention builds on.
- PyTorch docs — `torch.compile`, TorchInductor, and `torch.nn.functional.scaled_dot_product_attention` with `sdpa_kernel`.
- The OpenAI Triton documentation — the fused-softmax and matmul tutorials, the language `torch.compile` generates.
- Within this series: [why HPC is the bottleneck for modern AI](/blog/machine-learning/high-performance-computing/why-hpc-is-the-bottleneck-for-modern-ai), [the memory hierarchy](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm), [the roofline model](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound), [profiling GPU workloads](/blog/machine-learning/high-performance-computing/profiling-gpu-workloads-finding-the-real-bottleneck), and the capstone [HPC playbook for AI engineers](/blog/machine-learning/high-performance-computing/the-hpc-playbook-for-ai-engineers).
