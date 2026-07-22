---
title: "Bandwidth-Bound and Fusion: Why FlashAttention Is the Canonical Memory-Wall Fix"
date: "2026-07-14"
publishDate: "2026-07-14"
description: "Half the kernels in your service are slow because they wait on HBM, not because the GPU is weak — and the fix is to stop moving the bytes. Here is how fusion works, why FlashAttention is the textbook case, and how the SDPA API picks the fused path for you."
tags:
  [
    "performance-engineering",
    "gpu-optimization",
    "flashattention",
    "kernel-fusion",
    "memory",
    "profiling",
    "pytorch",
    "cuda",
    "inference",
    "ml-systems",
  ]
category: "machine-learning"
subcategory: "Performance Engineering"
author: "Hiep Tran"
featured: true
readTime: 43
---

Here is a profile that stops people the first time they read it. An attention kernel, running on an A100 80GB SXM — a chip rated at 312 bf16 TFLOP/s — is doing its work at roughly 8% of that peak. Not 80%. Eight. The tensor cores, the whole reason you rented the A100, are sitting idle more than nine-tenths of the time. And when you pull the kernel apart in Nsight Compute, the Speed-of-Light section tells you exactly why: **Memory Throughput 89%, Compute Throughput 11%.** The GPU is not weak and it is not broken. It is doing almost nothing but reading and writing a giant intermediate matrix to and from HBM, and it spends its life waiting on that ~2 TB/s pipe while the arithmetic units twiddle their thumbs.

This is the third of the four wastes this series keeps returning to: not the idle GPU of a host-bound service, not the low occupancy of a badly-shaped kernel, but the **bandwidth wall** — a kernel that is *memory-bound*, limited by how fast the chip can move bytes rather than how fast it can multiply. And the frustrating thing about a memory-bound kernel is that the obvious levers do nothing. A faster GPU with more FLOP/s? Wasted — you were not compute-limited. More threads, higher occupancy? The bytes still have to travel. The only thing that helps a kernel that is drowning in HBM traffic is to **stop moving the bytes** — to merge the operations so their intermediate results never leave the chip. That merge is called **fusion**, and its most famous instance, the one that turned "attention is quadratic in memory" from a hard limit into a footnote, is FlashAttention.

Figure 1 is the whole post in one picture. On the left, naive attention: it builds the full N-by-N score matrix in HBM, streams it back and forth four times, and lands at 8% of peak — memory-bound. On the right, the fused version — what you get when you call `torch.nn.functional.scaled_dot_product_attention` and it dispatches FlashAttention: the same math, the same output, but the N-by-N matrix is never written to HBM. It is computed tile by tile in SRAM (the small, fast on-chip memory), the HBM traffic drops from quadratic to linear in sequence length, and the kernel crosses over to *compute-bound* at north of 60% of peak. Nothing about the arithmetic changed. Only the bytes did.

![a two panel comparison of naive attention at eight percent of peak against fused attention at sixty two percent of peak with the score matrix removed from HBM](/imgs/blogs/bandwidth-bound-and-fusion-1.webp)

This is the twenty-fifth post in the [Profiling & Optimizing AI Services](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu) series, and it closes the memory-and-data-movement track. By the end you will be able to look at a kernel in a profile, decide in one glance whether it is bandwidth-bound, prove it with two Nsight Compute counters, and know whether fusion is the fix or a waste of your afternoon. We will derive the HBM-traffic reduction that makes FlashAttention work — the honest arithmetic, not the hand-wave — walk through the online-softmax trick that makes tiling possible, and then wire it into a real service through the SDPA API, including the subtle traps (an `fp32` tensor, an unusual mask) that silently kick you off the fused path and back onto the slow one. The running example stays what it has been all series: a Transformer inference service on one GPU. This time the enemy is the memory wall.

## The memory-bound diagnosis: the GPU is waiting on HBM

Before we fix anything, we have to be able to name the disease, and the name is *arithmetic intensity*. The [roofline post](/blog/machine-learning/performance-engineering/the-roofline-for-your-service) develops this in full; here is the load-bearing sentence. Every kernel does some number of floating-point operations and moves some number of bytes to and from HBM, and the ratio of the two is its arithmetic intensity:

$$\text{AI} = \frac{\text{FLOPs}}{\text{bytes moved to/from HBM}} \quad \left[\frac{\text{FLOP}}{\text{byte}}\right]$$

This single number decides your fate, because the attainable performance of any kernel is the smaller of two ceilings — the chip's peak math rate, and its arithmetic intensity multiplied by its peak bandwidth:

$$P_\text{attainable} = \min\big(P_\text{peak},\; \text{AI} \times B\big)$$

The two ceilings cross at the **ridge point**, $\text{AI}^* = P_\text{peak} / B$. For an A100 that is 312/2.0 = 156 FLOP per byte. A kernel with intensity above 156 is capped by the flat compute roof — it is **compute-bound**, and only more FLOP/s helps it. A kernel below 156 is capped by the sloped memory roof — it is **memory-bound** (equivalently *bandwidth-bound*), and only fewer bytes or more bandwidth helps it. That is the whole diagnosis, and it is one division and one comparison.

The reason this matters for fusion is that almost nothing in a neural network clears the 156 bar. A large, well-tiled matrix multiply does. Practically everything else — the elementwise chains, the normalizations, the activation functions, and, crucially, naive attention — sits far to the left of the ridge, memory-bound, running at a single-digit or low-double-digit percentage of the chip's advertised peak. When you look at your service and see the GPU "busy" but the work crawling, this is usually why: the kernels are busy *moving bytes*, not computing.

### Why naive attention is the worst offender

Attention is the canonical case because its memory cost is not just high, it is *quadratic in the sequence length*, and the quadratic term is pure HBM traffic that produces no lasting output. Walk through what a textbook implementation does for a single attention head with sequence length $N$ and head dimension $d$. It computes the score matrix $S = QK^\top$, which is $N \times N$. It applies a softmax over each row of $S$. Then it multiplies the normalized scores by $V$ to get the output. Written as three separate kernels — which is exactly what eager PyTorch does — the sequence of HBM operations is brutal.

![a four layer stack of the naive attention HBM passes writing and reading the score matrix four times totalling roughly four N squared bytes](/imgs/blogs/bandwidth-bound-and-fusion-2.webp)

Figure 2 counts the passes. First, the $QK^\top$ kernel **writes** the $N \times N$ score matrix $S$ to HBM. Then the softmax kernel **reads** all $N^2$ elements back to find each row's max and sum, and **writes** the $N^2$ normalized probabilities back out. Then the $P V$ kernel **reads** those $N^2$ probabilities one more time. That is four full passes over an $N \times N$ matrix — two writes and two reads — before the output is produced. The score matrix itself is throwaway scratch; it exists only because the three kernels cannot see each other's registers, so each one has to spill its result to HBM for the next one to pick up.

Now put numbers on it. Take a realistic decode-time attention: batch 8, 16 heads, head dimension 64, and let the sequence length be 8192, in fp16 (2 bytes per element). The score matrix for a single (batch, head) pair is $8192^2 = 67$ million elements, or about 134 MB. Across all $8 \times 16 = 128$ (batch, head) pairs, one copy of the score matrix is roughly 17 GB. And naive attention touches it four times. That is on the order of 68 GB of HBM traffic, per attention layer, moving a matrix that is thrown away the moment the output is computed. On a chip with 80 GB of memory total, you can see the second problem coming: at long sequence length the score matrix does not just make attention *slow*, it makes it *not fit*.

The FLOPs, meanwhile, are modest. The two matmuls in attention cost about $4 N^2 d$ FLOPs per head. Divide the FLOPs by the bytes and the arithmetic intensity of naive attention lands around $d/2$ — for $d = 64$, roughly 30 FLOP per byte. Thirty is far below the A100's ridge of 156, so naive attention is unambiguously memory-bound, and the 30-out-of-156 headroom is why a well-tuned naive kernel still only reaches the single-digit-to-low-teens percentage of peak we opened with. The tensor cores are not the bottleneck. The 17 GB round-trips are.

### The signature in the profiler

You do not have to take my word for any of this; the profiler will tell you. The one tool that reads a single kernel down to the metal is Nsight Compute (`ncu`), and its Speed-of-Light section reports exactly the two numbers that settle the memory-vs-compute question. Run it against the naive attention kernel:

```bash
# Profile just the attention kernels, full metric set, one launch.
ncu --set full \
    --launch-count 1 \
    -k "regex:.*attention.*|.*bmm.*|.*softmax.*" \
    -o naive_attn_report \
    python bench_attention.py --impl naive --seq 8192
```

The Speed-of-Light table for the score-matrix kernels reads like a confession:

```console
  Section: GPU Speed Of Light Throughput
  ----------------------------------------------------------------------
  Metric Name                          Metric Unit      Metric Value
  ----------------------------------  -------------  ----------------
  Compute (SM) Throughput                    %                 11.24
  Memory Throughput                          %                 89.71
  DRAM Throughput                            %                 89.71
  L2 Cache Throughput                        %                 61.30
  Duration                                  usecond          2140.0
  Achieved Occupancy                         %                 71.80
  ----------------------------------------------------------------------
  OPT   Memory is more heavily utilized than Compute: look at the
        Memory Workload Analysis section to identify the DRAM
        bottleneck. Check whether accesses can be reduced or fused.
```

Memory Throughput 89.7%, Compute Throughput 11.2%. The kernel is pinned against the memory roof and nowhere near the compute roof. Note the last line — Nsight Compute's own recommendation is literally "check whether accesses can be reduced or **fused**." When memory throughput dwarfs compute throughput like this, you are memory-bound, full stop, and the [Nsight Compute deep-dive](/blog/machine-learning/performance-engineering/nsight-compute-kernel-deep-dive) walks through reading the rest of this report. The occupancy is a healthy 72%, which is the tell that this is *not* an occupancy problem — you have plenty of warps resident, they are all just waiting on the same slow pipe. Throwing more threads at a bandwidth wall does not move it.

### Why HBM is the slow part

It helps to be precise about *what* the kernel is waiting on, because "memory is slow" is two different facts wearing one coat. HBM — the high-bandwidth memory stacked next to the GPU die — is enormous (80 GB on an A100) and, by the standards of DRAM, genuinely fast: 2.0 TB/s of bandwidth is a firehose. But it is off-die, and reaching it costs both *bandwidth* (how many bytes per second the pipe can sustain) and *latency* (how long any single access takes before the first byte arrives — hundreds of clock cycles). SRAM — the on-chip shared memory and registers each streaming multiprocessor owns — is one to two orders of magnitude faster on both axes, but it is measured in tens of kilobytes per SM, not gigabytes. The [GPU memory hierarchy post](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm) lays out the full ladder; the number that matters here is the ratio, and the ratio is why fusion pays. Every byte you keep in SRAM instead of spilling to HBM is a byte you fetch in a handful of cycles instead of a few hundred, from a pipe that is a hundred times wider.

A bandwidth-bound kernel like naive attention is losing on the *bandwidth* axis: it has so many bytes to move that even at 90% of the 2.0 TB/s firehose it takes tens of milliseconds, and the tensor cores starve waiting for data that is physically in transit. This is distinct from a *latency-bound* kernel — one that stalls on the round-trip time of individual scattered accesses rather than the aggregate volume — which is a different disease with a different fix (better access patterns, more warps to hide the latency). The profiler tells them apart: a bandwidth-bound kernel shows DRAM throughput near 100% (it is saturating the pipe), while a latency-bound one shows DRAM throughput low but memory *stalls* high (the pipe is idle, waiting). Naive attention is squarely the former — DRAM throughput 89.7% in the report above — which is exactly the disease fusion cures, because fusion attacks the byte *volume*, not the access pattern.

That is the diagnosis. Now the cure.

## Fusion: the fix is to stop moving the bytes

If the disease is HBM round-trips, the cure is to have fewer of them, and the way you get fewer of them is **fusion**: merge a chain of operations into a single kernel so their intermediate results live in registers or shared memory (SRAM) and never spill to HBM. The score matrix in naive attention exists *only* because three separate kernels had to hand data to each other through the one memory they both can see, which is HBM. Fuse them into one kernel and the intermediate can stay on-chip, where a round-trip costs a few nanoseconds instead of the hundreds of nanoseconds an HBM access costs. The [GPU memory-hierarchy post](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm) covers why SRAM is one to two orders of magnitude faster and smaller than HBM; the practical consequence is that anything you can keep in SRAM, you should.

Start with the simplest possible case, because it makes the mechanism obvious: a chain of elementwise operations. Suppose your model does something like a scaled, biased activation — common in the residual and gating paths of a Transformer:

```python
import torch

def bias_gelu_scale(x, bias, scale):
    # Four elementwise ops. In eager mode, four separate CUDA kernels.
    y = x + bias          # kernel 1: read x, read bias, write y  -> HBM
    y = torch.nn.functional.gelu(y)  # kernel 2: read y, write y  -> HBM
    y = y * scale         # kernel 3: read y, read scale, write y -> HBM
    return y * y          # kernel 4: read y, write y             -> HBM
```

In eager mode each line is its own CUDA kernel launch. Each kernel reads its input tensor from HBM and writes its output tensor back to HBM. So a four-op chain over a tensor of $M$ bytes moves roughly $8\,M$ bytes of HBM traffic (four reads plus four writes), and computes only a handful of FLOPs per element. The arithmetic intensity is a fraction of one FLOP per byte — this is as memory-bound as it gets. Every one of those kernels is spending its life waiting on HBM to serve up the same data the previous kernel just wrote.

Fuse the four ops into one kernel and the picture inverts. A single fused kernel reads $x$, $bias$, and $scale$ once, does all four operations while the data sits in registers, and writes the final result once. HBM traffic collapses from $8\,M$ to roughly $2\,M$ (one read of the inputs, one write of the output) — a 4x reduction — and because the FLOPs are unchanged while the bytes dropped 4x, the arithmetic intensity rose 4x. You moved the dot to the right on the roofline. That is the entire principle in one sentence: **fewer HBM round-trips means higher arithmetic intensity means you climb the memory roof toward the ridge.**

You do not write this fused kernel by hand. `torch.compile` does it for you — its Inductor backend is, among other things, an elementwise-fusion engine, and [what torch.compile actually does](/blog/machine-learning/performance-engineering/what-torch-compile-actually-does) walks through the whole stack. Wrapping the function is one line:

```python
fused = torch.compile(bias_gelu_scale)
# First call traces and codegens a single fused Triton kernel; later calls replay it.
out = fused(x, bias, scale)
```

To *prove* the fusion happened rather than trust that it did, dump the generated code. Inductor names its fused elementwise kernels `triton_poi_fused_*`, and the name lists every op it swallowed:

```bash
TORCH_LOGS="output_code" python fuse_demo.py 2>&1 | grep "def triton_poi_fused"
```

```console
def triton_poi_fused_add_gelu_mul_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, ...):
```

One kernel — `add`, `gelu`, `mul` all fused into it (the final `y * y` folds into the same `mul` graph). Four launches became one, and eight HBM passes became two. If you profile before and after with `torch.profiler`, the eager version shows four separate kernel rows in the trace and the compiled version shows one; the [profiling-compiled-code post](/blog/machine-learning/performance-engineering/profiling-compiled-code) shows how to read that difference in a Chrome trace. Figure 3 draws the general shape of the merge, using attention's three kernels as the example we are about to build up to.

![a dataflow diagram in which three separate HBM bound kernels for the score product softmax and value product merge into one fused kernel that sweeps SRAM once](/imgs/blogs/bandwidth-bound-and-fusion-3.webp)

### What limits how much you can fuse

Fusion is not infinite, and knowing its ceiling keeps you from expecting miracles. The intermediate results a fused kernel keeps on-chip have to *fit* on-chip — in registers and shared memory, which are tiny. An SM has on the order of 256 KB of register file and up to ~228 KB of shared memory (on an A100), split across all the warps resident on it. Fuse too many operations, or operations over too large a tile, and the intermediates spill: the compiler runs out of registers and starts staging values back to HBM (called *register spilling*), or the kernel's shared-memory request drops its occupancy so low that it can no longer hide latency. Either way the fusion stops paying. This is why `torch.compile` fuses *elementwise and reduction* chains aggressively but does not try to fuse two large matmuls into one kernel — the working set of a big GEMM does not fit in SRAM, so there is nothing to keep on-chip.

FlashAttention is impressive precisely because it fuses *matmuls* — normally too big to fuse — and it can only do so because tiling shrinks the working set to one block of $Q$ and one block of $K,V$ at a time, each sized to fit in SRAM. The tiling is what makes the fusion legal; without it, the score matrix's working set would blow the on-chip budget and force exactly the spill to HBM you were trying to avoid. So the general principle has a corollary: **fuse until the working set stops fitting on-chip, then tile so it fits, then fuse the tiles.** That two-step — tile, then fuse — is the pattern behind essentially every high-performance fused kernel, and it is what separates a toy fusion (four elementwise ops) from a hard one (all of attention).

#### Worked example: an elementwise chain on an L4

Take the `bias_gelu_scale` chain above, applied to a hidden-state tensor of shape (batch 32, sequence 512, hidden 4096) in fp16 — about 134 MB per tensor — on an L4 (242 fp16 TFLOP/s, ~300 GB/s HBM). Eager mode launches four kernels; measured with CUDA events after a warm-up and a `torch.cuda.synchronize()`, each kernel runs about 1.0 ms — nearly all of it bandwidth-limited, because $8 \times 134$ MB $= 1.07$ GB at 300 GB/s is about 3.6 ms of pure transfer, and the four launches also carry roughly 4 × 7 µs of host-side launch overhead. Fused with `torch.compile`, the single kernel moves about 268 MB (one read of the three inputs, one write) — roughly 0.9 ms of transfer — and measures around 1.0 ms total. The chain went from ~4.0 ms to ~1.0 ms, a 4x win, and every bit of it came from deleting HBM round-trips, not from computing faster. The FLOPs were identical in both runs.

## Which kernels fusion actually helps

Fusion is not free and it is not universal, so before we spend real effort on the hard case — attention — it is worth being precise about *which* kernels are candidates. The answer is exactly the roofline answer: fusion helps kernels that are memory-bound, and it does nothing for kernels that are compute-bound, because a compute-bound kernel is not waiting on HBM in the first place. Figure 4 sorts the four kernel families you meet most often.

![a comparison grid of four kernel families with their arithmetic intensity their bound and whether fusion is the right fix](/imgs/blogs/bandwidth-bound-and-fusion-4.webp)

Read the rows. **Naive attention** sits at AI ≈ 30, memory-bound, and the fix is to fuse it into FlashAttention — the whole subject of the next section. The **elementwise chain** sits at AI ≈ 0.25, deeply memory-bound, and `torch.compile` fuses it automatically. A **large GEMM** — say a $2048^3$ matrix multiply — sits at AI ≈ 500, well above the ridge, firmly compute-bound; there is nothing to fuse and nothing fusion could buy you, because the kernel is already spending its time on arithmetic, not on HBM. And **fused attention** — the after-state, once FlashAttention has done its work — has an arithmetic intensity that grows with sequence length (we will derive the $N/2$ figure shortly), which pushes it *across* the ridge to compute-bound, which is precisely why it is fast and precisely why there is nothing left to fuse.

The general rule the table encodes is worth stating plainly, because it saves you from a common failure mode: **do not try to fuse a compute-bound kernel.** If `ncu` shows a kernel at 85% Compute Throughput and 20% Memory Throughput, it is doing its job; fusion would add complexity and buy nothing. The single big GEMM in your model is not where your bandwidth problem lives. Your bandwidth problem lives in the long tail of small, memory-bound kernels — the norms, the activations, the elementwise glue, and above all the attention — that surround the GEMMs and, in aggregate, often eat more wall-clock time than the GEMMs themselves.

## Deciding whether fusion is your fix

Put the diagnosis and the rule together and you get a procedure you can run on any slow kernel in about a minute. It starts from the symptom — a kernel that is slow, running far below the chip's peak — and routes to a fix using the two Speed-of-Light counters you already know how to read.

![a decision tree that routes a slow kernel through the memory and compute throughput readings toward fusion or a different fix](/imgs/blogs/bandwidth-bound-and-fusion-5.webp)

Figure 5 is that procedure. The kernel is slow — below, say, 15% of peak. Read the Speed-of-Light section. If **Memory Throughput is high** and compute is low, you are memory-bound, and now the shape of the kernel tells you which fusion: an elementwise chain gets `torch.compile`; an attention kernel that is materializing a score matrix gets SDPA/FlashAttention. If instead **Compute Throughput is high**, you are compute-bound, and fusion is the wrong tool — a single big GEMM is already optimal, and if compute is high but the kernel is still slow you are looking at a low-occupancy or launch-overhead problem, which the other posts in this series handle. The value of drawing it as a tree is that it stops you from reaching for fusion reflexively. Fusion is the answer to exactly one question — "am I memory-bound because of avoidable HBM round-trips?" — and the tree is how you confirm the question applies before you invest in the answer.

There is one more branch the tree does not draw but you should keep in mind: a kernel can be memory-bound and *already* moving the minimum possible bytes. A single elementwise op that reads one tensor and writes one tensor is memory-bound, but there is nothing to fuse it *with* in isolation — the win comes from fusing it into its neighbors, which requires that it have neighbors in the graph. This is why `torch.compile` works at the graph level, not the op level: fusion is a property of adjacent operations, not of a single op.

## FlashAttention: fusing the whole of attention

Now the main event. Everything so far was to earn the right to state, precisely, what FlashAttention does and why it works. The claim is simple and the mechanism is beautiful: FlashAttention computes exactly the same attention output as the naive three-kernel version — bit-for-bit equivalent up to floating-point reordering — but it never writes the $N \times N$ score matrix to HBM. It fuses $QK^\top$, softmax, and $PV$ into a single kernel that processes the score matrix in tiles, keeping each tile in SRAM, and carries just enough running state to stitch the tiles together. The result is that HBM traffic drops from quadratic in $N$ to linear in $N$, which is the entire ballgame.

### The traffic derivation

Let us do the arithmetic honestly, because the honest version is more convincing than the slogan. Attention on a single head with sequence length $N$ and head dimension $d$ does about $4 N^2 d$ FLOPs — that number is fixed, it is a property of the math, and no implementation changes it. What changes is the bytes.

**Naive attention** moves the score matrix through HBM four times, as we counted: two writes and two reads of an $N \times N$ matrix, which in fp16 is $4 \times 2 N^2 = 8 N^2$ bytes, plus the $O(N d)$ bytes for loading $Q$, $K$, $V$ and writing $O$. For large $N$ the $8 N^2$ term dominates completely. So the arithmetic intensity of naive attention is

$$\text{AI}_\text{naive} \approx \frac{4 N^2 d}{8 N^2} = \frac{d}{2}.$$

For $d = 64$ that is 32 FLOP per byte — a constant, independent of sequence length, and well below the ridge of 156. Naive attention is memory-bound at *every* sequence length, and it gets absolutely no better as $N$ grows; it just gets bigger and slower in lockstep.

**FlashAttention** never materializes the $N \times N$ matrix, so the $8 N^2$ term vanishes. What remains is the traffic for $Q$, $K$, $V$, and $O$, each of which is $N \times d$ — a total of $O(N d)$ bytes. So its arithmetic intensity is

$$\text{AI}_\text{flash} \approx \frac{4 N^2 d}{O(N d)} = O(N).$$

The arithmetic intensity now *grows linearly with sequence length*. For $N = 8192$ that is thousands of FLOP per byte, far to the right of the ridge — FlashAttention is compute-bound, which is exactly where you want a matmul-heavy kernel to live. The FLOPs never changed; the denominator went from $N^2$ to $N$, and that single change moved attention from the worst memory-bound kernel in the model to a healthy compute-bound one.

Two honest caveats, because I promised the honest version. First, the "$O(Nd)$" hides a constant: FlashAttention re-reads blocks of $K$ and $V$ once per block of $Q$, so its true HBM traffic is $\Theta(N^2 d^2 / M)$ where $M$ is the SRAM size available to the kernel — sub-quadratic and, for the SRAM sizes of real GPUs, a large constant-factor reduction rather than a literal linear count. The original [FlashAttention paper](https://arxiv.org/abs/2205.14135) (Dao et al., 2022) proves this bound rigorously; the intuition that survives is that the dominant $\Theta(N^2)$ score-matrix traffic is *gone*. Second, all of this assumes you are in the regime where the score matrix dominates, which is exactly the long-sequence, small-head-dimension regime that modern Transformers live in. At very short sequence lengths the score matrix is tiny and none of this matters — a point we will stress-test shortly.

#### Worked example: HBM traffic, naive versus flash, at seq 8192

Put the same numbers through both. Batch 8, 16 heads, head dim 64, sequence 8192, fp16, one attention layer, on an A100. Naive: the score matrix is $8 \times 16 \times 8192^2 \times 2$ bytes ≈ 17 GB for one copy, touched four times, so ≈ 68 GB of HBM traffic for the score matrix alone, plus a rounding error of $QKVO$ traffic. At the A100's 2.0 TB/s that is a floor of about 34 ms just to move the bytes — and that is if bandwidth were perfectly utilized, which it never is. Flash: the score matrix traffic is *zero*; the kernel moves $Q$, $K$, $V$, $O$, which is $8 \times 16 \times 8192 \times 64 \times 2 \times 4$ bytes ≈ 0.54 GB, a factor of ~125 less traffic, plus the bounded $K,V$ re-reads. The transfer floor drops from ~34 ms to well under 1 ms, and the kernel becomes limited by the $4N^2d$ FLOPs instead — which on an A100's tensor cores is the fast path. Same output. One hundred and twenty-five times less HBM traffic. That ratio is why FlashAttention exists.

### Online softmax: the trick that makes tiling legal

There is a genuine obstacle to tiling attention, and it is softmax. To normalize a row of the score matrix you need that row's maximum (for numerical stability) and its sum of exponentials — and both are reductions over the *entire* row. If you only ever hold a tile of the row in SRAM, you do not know the global max or the global sum yet, so how can you normalize? The answer is **online softmax** (also called streaming or running softmax), and it is the mathematical heart of FlashAttention. Figure 6 shows the sweep it enables.

![a left to right timeline of FlashAttention loading key and value blocks into SRAM one at a time while carrying a running maximum and sum to accumulate the output](/imgs/blogs/bandwidth-bound-and-fusion-6.webp)

The idea is to maintain three running quantities as you stream key-value blocks through SRAM: the running row-maximum $m$, the running sum of exponentials $\ell$, and the running weighted output accumulator $O$. When a new block $j$ arrives with its own local scores, you update all three, correcting the previous partial results for the fact that the max may have just increased. Concretely, for each new block:

$$m_j = \max(m_{j-1},\ \text{rowmax}(S_j))$$
$$\ell_j = e^{\,m_{j-1} - m_j}\,\ell_{j-1} + \text{rowsum}\!\big(e^{\,S_j - m_j}\big)$$
$$O_j = e^{\,m_{j-1} - m_j}\,O_{j-1} + e^{\,S_j - m_j}\,V_j$$

The correction factor $e^{\,m_{j-1} - m_j}$ is the whole trick: whenever a later block reveals a larger maximum, it rescales everything you accumulated from earlier blocks so that all terms are expressed relative to the *current* running max. At the end, the true softmax-weighted output is simply $O = O_T / \ell_T$. Every quantity here is a small vector — the size of one tile, not the size of the full row — so the entire computation fits in SRAM. You have computed an exact softmax over a row you never held in full. This is why FlashAttention can fuse the three kernels: softmax stopped being a barrier that needs the whole row at once and became a running update that consumes the row block by block, in lockstep with the $QK^\top$ and $PV$ matmuls that produce and consume it.

The tiling loop, then, is: load a block of $Q$ into SRAM; loop over blocks of $K$ and $V$, computing that tile's partial scores, updating $m$, $\ell$, and the output accumulator; when the $K,V$ blocks are exhausted, divide by $\ell$ and write the output block to HBM. The $N \times N$ score matrix is produced and consumed one tile at a time and never exists in HBM as a whole object. That is fusion at its most complete — not merging two elementwise ops, but merging two matmuls and a reduction into a single kernel with a hand-carried numerical invariant.

One subtlety worth naming, because it is where naive tiling attempts go wrong: the max subtraction is not optional. Softmax is computed as $e^{S - m} / \sum e^{S - m}$ rather than $e^S / \sum e^S$ precisely because $e^S$ overflows fp16 for scores of any real magnitude — an attention score of 20 already gives $e^{20} \approx 5 \times 10^8$, and fp16 tops out around 65504. Subtracting the row max keeps every exponent at or below zero, so the largest term is $e^0 = 1$ and nothing overflows. The reason online softmax needs the *running* max and the rescaling factor $e^{m_{j-1} - m_j}$ is that it must preserve this stability guarantee even though it never sees the whole row at once — each block normalizes against the best max seen so far, and the correction retroactively fixes the earlier blocks when a bigger max shows up. Get the rescaling wrong and the kernel produces NaNs on long sequences with large scores; get it right and FlashAttention is numerically *identical* to the stable naive softmax, not an approximation.

### The backward pass: recompute rather than store

Training raises a question the inference story skips: the backward pass needs the attention probabilities to compute gradients, and FlashAttention deliberately never stored them. The naive answer would be to keep the $N \times N$ probability matrix around for the backward pass — but that reintroduces exactly the $O(N^2)$ memory we just deleted, and for training the memory pressure is worse than inference because you are also holding activations for every layer. FlashAttention's answer is **recomputation**: in the backward pass it re-derives the needed score and probability tiles on the fly from $Q$, $K$, $V$ and the stored per-row softmax statistics (the max and sum it saved, which are only $O(N)$), tiling the backward pass the same way it tiled the forward pass. This trades a little extra compute — you compute the scores twice, once forward and once backward — for keeping memory linear in $N$ instead of quadratic. It is the same bargain fusion always offers, seen from the memory side: spend redundant FLOPs, which you have to spare because you are memory-bound, to avoid moving or storing bytes, which you do not. The extra matmul in the backward pass is nearly free precisely because attention was never compute-bound to begin with.

## The SDPA API: how to actually call the fused path

You will almost never write a FlashAttention kernel yourself, and you should not — the CUDA is genuinely hard, the numerics are fiddly, and the good implementations are already in PyTorch. The API is `torch.nn.functional.scaled_dot_product_attention` (SDPA), and it is a dispatcher: you hand it $Q$, $K$, $V$, and it picks the best available fused backend for your dtype, shape, mask, and hardware. Replacing hand-written attention with it is usually a few lines:

```python
import torch
import torch.nn.functional as F

# q, k, v: (batch, heads, seq, head_dim). fp16 or bf16 for the fast path.
def attention(q, k, v, is_causal=True):
    # One fused call. No explicit N-by-N score matrix anywhere in your code.
    return F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=None,     # None + is_causal=True is the fast, mask-free path
        dropout_p=0.0,
        is_causal=is_causal,
    )
```

That single call replaces the `q @ k.transpose(-2, -1)`, the `softmax`, the masking, and the `@ v` — and, critically, it dispatches to FlashAttention when it can, so the $N \times N$ matrix never appears. There are four backends the dispatcher chooses among, and knowing them is the difference between assuming you got the fast path and knowing you did.

![a comparison grid of the four SDPA backends showing when each is picked and its requirement with the math fallback rebuilding the full matrix](/imgs/blogs/bandwidth-bound-and-fusion-7.webp)

Figure 7 lays them out. The **FlashAttention backend** is the one you want: it needs fp16 or bf16, a head dimension within supported bounds, and no arbitrary dense mask (a causal flag is fine). The **memory-efficient backend** is a slightly more permissive fused path (it tolerates more mask and dtype variety, including some fp32) that still avoids materializing the full matrix. The **cuDNN backend** is NVIDIA's fused attention on recent hardware. And then there is the **math backend** — the fallback, a straight PyTorch transcription of the naive three-step computation that *does* materialize the $N \times N$ score matrix in HBM. The math backend is correct and it is a safety net, but it is the slow, memory-bound path you were trying to escape. The trap is that SDPA falls back to it *silently*: if your inputs do not qualify for a fused backend, you still get correct answers, just at 8%-of-peak speed, with no error and no warning.

### Checking which backend actually ran

Because the fallback is silent, you must verify. The cleanest way is to force a backend with the `sdpa_kernel` context manager and let PyTorch raise if the requirements are not met:

```python
from torch.nn.attention import SDPBackend, sdpa_kernel

# Assert you are on the flash path — this raises if flash cannot run here.
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

If `q`, `k`, `v` qualify, this runs FlashAttention. If they do not — wrong dtype, unsupported head dim, an incompatible mask — it raises a `RuntimeError` telling you why, instead of quietly falling back. That turns a silent performance bug into a loud correctness-style failure you cannot miss. In production you typically leave SDPA on automatic dispatch and use this context manager only in a test that asserts the fast path is reachable for your real shapes.

The other way to check is to look at the trace and read the kernel names, which never lie. Profile both dtypes and compare:

```python
import torch
from torch.profiler import profile, ProfilerActivity

def run(dtype):
    q = torch.randn(8, 16, 4096, 64, device="cuda", dtype=dtype)
    k = torch.randn_like(q); v = torch.randn_like(q)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            F.scaled_dot_product_attention(q, k, v, is_causal=True)
        torch.cuda.synchronize()
    print(f"--- dtype={dtype} ---")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=4))

run(torch.float16)   # expect a single flash kernel
run(torch.float32)   # expect bmm + softmax + bmm (the math fallback)
```

The fp16 run shows one fused kernel dominating the CUDA time:

```console
--- dtype=torch.float16 ---
--------------------------------------------------  ------------  ----------
Name                                                  CUDA total    # Calls
--------------------------------------------------  ------------  ----------
void flash_fwd_kernel<...>(Flash_fwd_params, ...)        7.412ms          10
--------------------------------------------------  ------------  ----------
```

The fp32 run shows the naive three-kernel decomposition — the score matmul, the softmax, and the value matmul, all separate, exactly the memory-bound pattern we started with:

```console
--- dtype=torch.float32 ---
--------------------------------------------------  ------------  ----------
Name                                                  CUDA total    # Calls
--------------------------------------------------  ------------  ----------
aten::bmm  (QK^T score matmul)                          19.883ms          10
aten::_softmax                                          11.204ms          10
aten::bmm  (P·V value matmul)                           14.061ms          10
--------------------------------------------------  ------------  ----------
```

One `flash_fwd_kernel` versus three separate `bmm`/`softmax` kernels. That is the whole story in a trace: fp16 fuses, fp32 does not. Which brings us to the single most common way people accidentally give up FlashAttention.

#### Worked example: the fp32 trap that silently disables flash

A team ships a Transformer inference service and, for reasons of "numerical safety," runs attention in fp32. The service works. The p99 latency is bad and the memory footprint is enormous, and nobody can see why, because the code calls `scaled_dot_product_attention` — surely that is FlashAttention? It is not. FlashAttention's kernels are written for fp16 and bf16; an fp32 tensor does not qualify, so SDPA silently dispatches to the **math backend**, which materializes the full $N \times N$ score matrix in HBM. At sequence length 8192 that is the 17 GB score matrix and its four round-trips, back in full. The fix is a one-line dtype change — run attention in bf16, which is both faster and, for attention, numerically fine because the online-softmax rescaling handles the dynamic range. Measured on an A100, switching that service's attention from fp32-math to bf16-flash cut attention latency by roughly 4x and dropped peak memory enough that the service could suddenly handle sequences it had been OOMing on. The single most expensive line in the service was the dtype nobody thought to question.

## The measured win, on named hardware

Here is the before-and-after the whole post has been building toward, on an A100 80GB SXM, for one attention layer at two sequence lengths, batch 8, 16 heads, head dim 64. The naive column is the fp32 math backend (materializing the score matrix); the flash column is bf16 SDPA on the FlashAttention backend. Numbers are representative of what the profiler above produces and are rounded; treat them as order-of-magnitude, not spec-sheet.

| Metric (A100 80GB) | Naive, seq 2048 | Flash, seq 2048 | Naive, seq 8192 | Flash, seq 8192 |
|---|---|---|---|---|
| Attention time (ms) | ~2.1 | ~0.6 | ~34 | ~4.2 |
| Score-matrix HBM traffic | ~4.3 GB | 0 | ~68 GB | 0 |
| Peak memory (score scratch) | ~2.1 GB | ~0.05 GB | ~34 GB | ~0.2 GB |
| Compute throughput (% SM) | ~11% | ~64% | ~9% | ~62% |
| Memory throughput (%) | ~90% | ~38% | ~91% | ~35% |
| Longest seq that fits (80GB) | — | — | ~9k (OOMs above) | 100k+ |

Read the columns. At **seq 2048** naive attention is already memory-bound and about 3.5x slower than flash, but it fits in memory and the service works — this is the regime where a lazy team gets away with it. At **seq 8192** the story turns brutal: naive attention is ~8x slower and needs ~34 GB of scratch just for the score matrix, which crowds out the model weights and KV cache and pushes the service toward OOM. Flash uses ~0.2 GB of scratch at the same sequence length, which is why the bottom row is the real headline: **FlashAttention is what lets you have long context at all.** The naive path OOMs somewhere around 9k tokens on an 80 GB card; the fused path keeps going into the hundreds of thousands because its memory grows linearly, not quadratically. Long-context models are not possible without this fusion. That is not a speed optimization; it is a capability that does not exist otherwise.

### How to measure this honestly

Every number above is easy to fake by accident, so the measurement discipline matters as much as the fix. Warm up first — the first call to SDPA triggers backend selection, autotuning, and lazy CUDA context work, and if you time it you will measure the compiler, not the kernel. Call `torch.cuda.synchronize()` before you read the clock, because CUDA is asynchronous and a naive `time.time()` around an un-synchronized kernel measures the *launch*, not the *execution* — a classic way to "prove" a kernel is instant when it is actually the bottleneck. Prefer CUDA events (`torch.cuda.Event(enable_timing=True)`) over wall-clock for kernel timing; they timestamp on the device and are immune to host-side jitter. Lock the clocks if you can (`nvidia-smi -lgc`) so thermal throttling does not masquerade as a regression. And for the memory numbers, read `torch.cuda.max_memory_allocated()` around the call rather than eyeballing `nvidia-smi`, which reports the reserved pool, not the live allocation. The [reproducible-benchmark discipline](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark) is what separates a real 8x from a measurement artifact.

## Beyond attention: the same pattern, everywhere

FlashAttention is the famous case, but "kill the HBM round-trip" is a pattern, and once you see it you see it everywhere in a model. The normalization layers are the next-biggest offenders. A naive **LayerNorm** or **RMSNorm** reads the input, computes the mean and variance (a reduction), reads the input *again* to subtract and scale, and writes the output — multiple passes over the same tensor, each a full HBM trip, for an operation that does almost no arithmetic. Fused into one kernel, the tensor is read once, the statistics and the normalization are computed while it sits in SRAM, and it is written once. `torch.compile` does this automatically, and hand-written fused RMSNorm kernels (the kind you find in high-performance inference stacks) do it explicitly; either way the win is the same shape as attention's — a memory-bound multi-pass op becomes a single-pass one.

Put a number on it so the scale is concrete. RMSNorm over a hidden state of shape (batch 32, sequence 2048, hidden 4096) in bf16 is about 537 MB per tensor. A naive decomposition — square the elements, reduce to a per-row sum, then a second pass to divide and scale — reads the input roughly twice and writes twice, so on the order of $4 \times 537$ MB ≈ 2.1 GB of HBM traffic for an operation whose actual arithmetic is a multiply and a reciprocal-square-root per element, arithmetic intensity well below one. Fused, it reads once and writes once — ~1.1 GB, a 2x traffic cut — and because a Transformer has two norms per layer times dozens of layers, that 2x compounds into real wall-clock. It never shows up as a single dramatic kernel the way attention does; it hides as a long tail of small memory-bound kernels that collectively rival the matmuls. This is the quiet half of the memory wall, and it is why turning on `torch.compile` often buys more than any single hand-optimization: it fuses the whole tail at once.

**Fused optimizers** apply the identical logic to the training step. A naive optimizer update walks every parameter tensor several times — read the gradient, read the momentum buffer, read the variance buffer, compute, write them all back — and for a model with thousands of parameter tensors that is a storm of small memory-bound kernels. A fused optimizer (`torch.optim.AdamW(..., fused=True)`, or the `foreach` variants) merges the elementwise math across the whole parameter list into a handful of kernels, collapsing thousands of tiny HBM-bound launches. It is the elementwise-chain win from earlier, applied to the optimizer's per-parameter arithmetic. **Activation fusion** — folding a bias-add and a GELU into the matmul epilogue so the activation happens while the GEMM's output is still in registers — is the same idea pushed into the matmul itself.

It is worth collecting the whole fusion menu in one place, because in practice you reach for a different tool depending on which memory-bound pattern you are looking at:

| Memory-bound pattern | Arithmetic intensity | Fusion tool | Typical win |
|---|---|---|---|
| Naive attention (score matrix) | ~30 FLOP/B | `F.scaled_dot_product_attention` | 2–8x time, O(N²)→O(N) memory |
| Elementwise chain (bias/act/scale) | ~0.25 FLOP/B | `torch.compile` (Inductor) | ~2–4x on the chain |
| LayerNorm / RMSNorm | below 1 FLOP/B | `torch.compile` or fused kernel | ~2x traffic per norm |
| Per-parameter optimizer step | below 1 FLOP/B | `fused=True` / `foreach` optimizer | thousands of launches → a few |
| Large GEMM | ~500 FLOP/B | none (already compute-bound) | none — do not fuse |

The first four rows are the memory wall; the last row is the reminder that not everything belongs on this table. And the pattern has a hard boundary, which is the discipline of knowing when to stop. Fusion does nothing for a kernel that is already compute-bound. If your profile shows a large GEMM at 85% compute throughput, there is no HBM round-trip to kill — the kernel is spending its time multiplying, which is what you wanted. Fusing around it adds engineering risk for zero benefit. The value of the roofline framing is that it tells you, before you write a line of kernel code, whether fusion is even the right category of fix. Memory-bound and multi-pass: fuse. Compute-bound: leave it alone and go find your bottleneck elsewhere — the launch overhead, the occupancy, the dataloader.

### The stress test: where naive attention is fine

Honesty requires the counter-case, because "always use flash" is not quite true and the roofline tells you why. At **very short sequence lengths**, naive attention is fine. If your sequence is 128 tokens, the score matrix is $128 \times 128$ — 32 KB per head in fp16, a rounding error — and it may well fit in cache and never stress HBM at all. The quadratic term that makes naive attention catastrophic at 8192 is negligible at 128, so the fusion win shrinks to nothing and the naive path is perfectly serviceable. This is the general truth about memory-bound problems: the pain scales with the size of the intermediate, so a small intermediate is not worth fusing. FlashAttention's advantage *grows with sequence length* — it is roughly break-even with a good naive kernel at short sequences and a runaway win at long ones — which is exactly the signature the traffic derivation predicts, since the deleted traffic scaled as $N^2$ and what remains scales as $N$.

The other stress test is **the compute-bound fusion that helps nothing.** Suppose someone profiles a service, sees it is slow, and reaches for fusion on the model's big feed-forward GEMMs. The GEMMs are at AI ≈ 500, compute-bound, 85% SM throughput. Fusing the bias-add into the GEMM epilogue saves one small elementwise kernel — real, but a rounding error against the GEMM's own runtime. The service does not get meaningfully faster, because the GEMM was never waiting on HBM. The lesson is the tree from Figure 5: read the two throughput counters *first*, and only reach for fusion when memory throughput is the high one. Reaching for it because the kernel is "slow" without checking the bound is how you spend a week for a 2% win.

## Case studies and real numbers

A few published results, cited as approximate — the exact figures depend on model, shapes, and hardware, but the *shapes* of the wins are robust and repeatedly reproduced.

**FlashAttention (Dao et al., 2022).** The original [FlashAttention paper](https://arxiv.org/abs/2205.14135) is the canonical demonstration that a "slow" attention kernel is memory-bound rather than compute-bound. It reports attention speedups of roughly 2–4x over a well-optimized standard implementation, and — the number that matters more — a memory reduction from quadratic to linear in sequence length, on the order of 10–20x less memory, which is what made training and serving long-context models practical. The BERT and GPT-2 end-to-end training speedups (roughly 15% and up to ~3x on the attention portion) are smaller than the kernel speedup because attention is only part of the model, which is Amdahl's law doing its usual work — but the *attention-only* win is exactly what the traffic derivation predicts, and it grows with sequence length exactly as predicted.

**FlashAttention-2 and beyond.** The follow-up work (Dao, 2023) roughly doubled FlashAttention's throughput again by improving the work partitioning across warps and reducing non-matmul FLOPs, reaching a substantial fraction of the A100's theoretical peak on attention — the kind of 50–70%-of-peak figures we used in the tables. The trajectory is instructive: once you have deleted the HBM traffic and become compute-bound, the *next* round of optimization is about occupancy and warp scheduling, not memory — the bound moved, so the fix moved with it. That is the series' whole method in miniature.

**torch.compile elementwise fusion.** PyTorch's own documentation and the Inductor design notes report that automatic fusion of elementwise and reduction chains is one of the largest single sources of `torch.compile`'s speedup on real models, precisely because those chains are memory-bound and fusion converts many kernels into few. The magnitude varies by model, but the mechanism is the one we measured in the worked example: fewer kernels, fewer HBM passes, higher arithmetic intensity.

**channels_last as a bandwidth win.** A cousin of fusion: PyTorch's `channels_last` memory format speeds up vision-model convolutions not by changing the math but by making memory access contiguous, cutting effective HBM traffic and enabling faster tensor-core kernel paths. PyTorch's own tutorial reports meaningful ResNet-family speedups from the layout change alone. It is the same principle as fusion — help a memory-bound op by reducing or streamlining its bytes — applied through layout rather than kernel merging.

## When to reach for fusion (and when not to)

Fusion is one of the highest-leverage optimizations in the book when it applies and a waste of effort when it does not, so be decisive about the boundary.

**Reach for fusion when** the profiler says memory-bound — high Memory Throughput, low Compute Throughput — and the kernel is one of a chain of small ops or is materializing a large intermediate it does not need to keep. Elementwise chains, norms, activations, and above all attention are the prime candidates. For elementwise and norm chains, `torch.compile` gets it for free; reach for it first. For attention, use `scaled_dot_product_attention` and *verify* you are on a fused backend. For the training step, turn on the fused optimizer.

**Do not reach for fusion when** the profiler says compute-bound. A kernel at 85% SM throughput is not waiting on HBM, and fusing around it buys nothing but risk. A single large GEMM is already optimal; leave it alone. **Do not hand-write a fused kernel** when `torch.compile` or a library kernel already fuses the pattern — the hand-written CUDA is a maintenance liability you rarely need. **Do not assume SDPA gave you FlashAttention** — an fp32 dtype, an exotic mask, or an unsupported head dimension silently drops you to the math backend, so assert the backend in a test. And **do not fuse a tiny intermediate** — at short sequence lengths naive attention is fine, and the fusion win is proportional to the size of the intermediate you are eliminating. The meta-rule is the one this whole series runs on: read the profile, name the bound, and apply the fix that matches the bound. Fusion is the fix for exactly one bound — memory — and applying it to any other bound is motion without progress.

## Key takeaways

- **Many kernels are slow because they wait on HBM, not because the GPU is weak.** A memory-bound kernel runs at single-digit percent of peak with the tensor cores idle, waiting on the ~2 TB/s pipe. More FLOP/s does nothing for it; fewer bytes is the only lever.
- **Arithmetic intensity decides the bound.** FLOPs per byte, compared to the ridge $P_\text{peak}/B$ (156 on an A100). Below the ridge you are memory-bound and fusion helps; above it you are compute-bound and fusion does not.
- **Fusion means: keep intermediates in SRAM, touch HBM once.** Merging a chain of ops so their intermediate results never spill to HBM cuts round-trips, raises arithmetic intensity, and climbs the memory roof toward the ridge. `torch.compile` does this for elementwise and norm chains automatically.
- **Naive attention materializes the N-by-N score matrix in HBM and touches it four times.** That is $O(N^2)$ throwaway traffic that dominates everything, makes attention memory-bound at every sequence length, and OOMs at long context.
- **FlashAttention fuses QKᵀ, softmax, and PV into one tiled kernel, so the score matrix never touches HBM.** HBM traffic drops from $O(N^2)$ to $O(Nd)$, arithmetic intensity rises from $O(1)$ to $O(N)$, and attention crosses over to compute-bound. Online softmax — a running max and sum with a rescaling correction — is what makes tiling exact.
- **The win grows with sequence length and enables long context.** Memory goes from quadratic to linear, which is why long-context models are possible at all — it is a capability, not just a speedup.
- **Call `F.scaled_dot_product_attention` and verify the backend.** The dispatcher picks FlashAttention when it can, but silently falls back to the materializing math backend for fp32, exotic masks, or unsupported dims. Assert with the `sdpa_kernel` context manager or read the kernel name in the trace.
- **fp32 silently disables FlashAttention.** The most common self-inflicted memory wall: run attention in bf16, not fp32, to stay on the fused path.
- **Do not fuse a compute-bound kernel.** Read the two Speed-of-Light counters first; fusion is the answer only when memory throughput is the high one.

## Further reading

- [Kernel Fusion and FlashAttention: Beating the Memory Wall](/blog/machine-learning/high-performance-computing/kernel-fusion-and-flashattention-beating-the-memory-wall) — the from-scratch derivation of the memory-wall argument and the tiling math, worked end to end.
- [The Memory Hierarchy: Registers, Shared Memory, and HBM](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm) — why SRAM is orders of magnitude faster than HBM, which is the physical reason fusion is worth doing.
- [The Roofline for Your Service](/blog/machine-learning/performance-engineering/the-roofline-for-your-service) — how to place a live kernel on the roofline and read off whether it is compute- or memory-bound.
- [The Nsight Compute Kernel Deep Dive](/blog/machine-learning/performance-engineering/nsight-compute-kernel-deep-dive) — reading the Speed-of-Light section, memory throughput, and warp-stall reasons for the two counters this post relies on.
- [What torch.compile Actually Does](/blog/machine-learning/performance-engineering/what-torch-compile-actually-does) — the Dynamo/Inductor stack that performs elementwise and norm fusion for free.
- [Why Your AI Service Wastes CPU and GPU](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu) — the series intro and the four-wastes frame that names the bandwidth wall.
- [The Performance Engineering Playbook](/blog/machine-learning/performance-engineering/the-performance-engineering-playbook) — the capstone decision tree that routes a symptom to a profiler to a fix.
- FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Dao et al., 2022), [arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135) — the primary source for the traffic bound and the online-softmax formulation.
- PyTorch, "Scaled Dot Product Attention" tutorial and the `torch.nn.functional.scaled_dot_product_attention` / `torch.nn.attention.sdpa_kernel` documentation — the API, the backends, and how to select among them.
