---
title: "The Roofline Model: Compute-Bound vs Memory-Bound"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "The single most useful mental model in GPU performance — arithmetic intensity, the roofline, and the ridge point — so you can look at any op, place it on the chart, and read off whether more FLOP per second, more bandwidth, or fusion is the fix."
tags:
  [
    "high-performance-computing",
    "gpu",
    "roofline",
    "arithmetic-intensity",
    "memory-bandwidth",
    "flashattention",
    "kernel-fusion",
    "a100",
    "h100",
    "deep-learning",
    "ml-systems",
  ]
category: "machine-learning"
subcategory: "High Performance Computing"
author: "Hiep Tran"
featured: true
readTime: 59
image: "/imgs/blogs/the-roofline-model-compute-bound-vs-memory-bound-1.png"
---

Here is a scene that has happened to almost every AI engineer who has ever tried to make a model faster. You profile a training step. You see that the GPU is, according to `nvidia-smi`, pegged at 100% utilization. You feel good. Then you compute your model FLOPs utilization — MFU, the fraction of the GPU's advertised math throughput your model actually uses — and it is 19%. The chip that the datasheet says does 312 trillion floating-point operations per second is, in your hands, doing about 59 trillion. Four-fifths of the silicon you are paying roughly \$30,000 a card for is idle, and the green bar in `nvidia-smi` was lying to you the whole time, because it only tells you the GPU is *busy*, not that it is busy doing *useful math*.

So you start optimizing. You upgrade from an A100 to an H100, which has three times the peak FLOP per second. Your training step barely speeds up. You are baffled and a little angry. You spent the budget; where did the speed go? The answer — the answer to almost every "why is my GPU slow" question — is one idea: **most of your kernels are not waiting on math. They are waiting on memory.** A faster math unit does nothing for a kernel that is starving for bytes. The H100's extra FLOP per second was irrelevant because the bottleneck was never FLOP per second. It was bandwidth.

There is exactly one diagram that makes this obvious, predicts it before you ever buy the hardware, and tells you which fix will actually help. It is called the **roofline model**, and it is the single most useful mental model in all of GPU performance. Once you can draw it and place an operation on it, you stop guessing. You look at a LayerNorm and know, before profiling, that buying more FLOP per second is a waste. You look at a big matrix multiply and know that fusion will not help it but a better numerical format will. You look at attention and understand precisely *why* FlashAttention is a 2-to-4× win and a memory kernel rewrite, not a math trick. Figure 1 is the whole model in one picture, and the rest of this post is how to read it, derive it, and use it on real Transformer ops.

![diagram of the roofline model showing a bandwidth-limited memory-bound slope rising to a ridge point and then a flat compute-bound ceiling at peak FLOP per second with example operations placed on each side](/imgs/blogs/the-roofline-model-compute-bound-vs-memory-bound-1.png)

This is the fourth post in the series, and it is the analytical spine the whole thing keeps returning to. The [intro](/blog/machine-learning/high-performance-computing/why-hpc-is-the-bottleneck-for-modern-ai) laid out the three walls — compute, memory bandwidth, and communication — and the [memory hierarchy post](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm) explained where the bytes actually live. The roofline is what ties those two together into a single number you can act on. By the end you will be able to take any operation in your model, compute its arithmetic intensity by hand, place it on the roofline for a named GPU, and state — with a number, not a vibe — whether the right lever is more FLOP per second, more bandwidth, or fusion. We keep one running example throughout: the GEMMs, LayerNorms, softmaxes, and attention of a Transformer, the same spine the whole series uses.

## The one number that decides everything: arithmetic intensity

Start with the only quantity you really need. Every operation a GPU runs does two things: it performs some number of floating-point operations (FLOPs), and it moves some number of bytes between the chip's compute units and its main memory (HBM, the high-bandwidth memory that sits next to the die). The ratio of those two is the operation's **arithmetic intensity**:

$$I = \frac{\text{FLOPs}}{\text{bytes moved}} \quad \left[\frac{\text{FLOP}}{\text{byte}}\right]$$

That is it. Arithmetic intensity is how much math you do per byte you fetch from memory. It is a property of the *algorithm and its data layout*, not of the hardware. A big matrix multiply does hundreds of multiply-adds for every byte it reads, because it reuses each loaded value many times. An elementwise add does one addition for every twelve bytes it touches — read two operands, write one result — and never reuses anything. Those two numbers, hundreds versus one-twelfth, are the entire difference between a kernel that loves a GPU and a kernel that starves on it.

Here is the intuition before any formula. Picture the GPU as a wildly fast kitchen with one slow delivery truck. The chefs (the math units) can chop, sear, and plate astonishingly quickly. The truck (the HBM bus) brings ingredients from the warehouse. If a dish requires a lot of cooking per ingredient — a long braise on one cut of meat — the chefs stay busy and the truck has plenty of time to restock. That dish is **compute-bound**: limited by how fast the chefs work. But if a dish is just "unwrap an ingredient and put it on a plate," the chefs finish instantly and stand around waiting for the next delivery. That dish is **memory-bound**: limited by the truck, not the chefs. Buying faster chefs (more FLOP per second) does nothing for the unwrap-and-plate dish. You need a faster truck (more bandwidth) or a recipe that does more cooking per delivery (higher intensity, via fusion).

Arithmetic intensity is the number that tells you which kind of dish you are cooking. And the roofline is the chart that, given a specific GPU's chefs and truck, tells you exactly where the crossover is.

Two terms we will use constantly, defined once. **Compute-bound** means the operation is limited by the rate at which the GPU can do arithmetic — its peak FLOP per second. **Memory-bound** means it is limited by the rate at which the GPU can move bytes to and from HBM — its peak bandwidth in bytes per second. Every operation is one or the other (or, rarely, sitting right at the boundary). The roofline's job is to tell you which, from a single number.

One more distinction that trips up newcomers, because the words look identical. **FLOPs** with a lowercase "s" means a *count* of operations — "this matmul is 137 GigaFLOPs." **FLOP/s** (or TFLOP/s) means a *rate* — "the A100 does 312 TeraFLOPs per second." Intensity uses the count in the numerator (FLOPs); the roofline ceiling uses the rate (FLOP/s). Keep them straight and the algebra never confuses you.

## Building the roofline: two ceilings and where they cross

Now we derive the model. It is genuinely two lines on a log-log plot, and the derivation is three sentences of algebra.

A GPU has two hard physical limits. The first is its peak math throughput, $P_\text{peak}$, in FLOP per second — for an A100 doing bf16 (the 16-bit "brain float" format) matmul on its Tensor Cores, that is 312 TFLOP/s, straight from the [NVIDIA A100 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet.pdf). The second is its peak memory bandwidth, $B$, in bytes per second — for the A100 80GB SXM, that is 2.0 TB/s of HBM2e, also from the datasheet. No kernel can ever exceed either of these. They are the laws of physics for this chip.

Now ask: given an operation with arithmetic intensity $I$, how fast can it possibly run? There are two upper bounds, and the real answer is whichever is smaller.

**Bound one — the compute ceiling.** No matter what, you cannot do math faster than the chip's peak rate. So your attainable performance is at most $P_\text{peak}$. This is a flat horizontal line on the plot: the roof.

**Bound two — the memory ceiling.** The operation must move $\text{FLOPs}/I$ bytes (by definition of $I$). At best the memory system delivers those bytes at bandwidth $B$ bytes per second. So the time to move the bytes is at least $(\text{FLOPs}/I)/B$ seconds, and the attainable performance — FLOPs divided by that time — is at most:

$$P_\text{mem} = \frac{\text{FLOPs}}{(\text{FLOPs}/I)/B} = I \cdot B$$

That is the crucial line. **Attainable performance is at most $I \cdot B$**, the arithmetic intensity times the bandwidth. On a log-log plot of performance versus intensity, $P = I \cdot B$ is a straight line with slope 1 (its slope *is* the bandwidth $B$). Low intensity, low ceiling; double the intensity, double the ceiling. This is the slanted part of the roof.

Put both bounds together and you get the roofline equation:

$$P_\text{attainable}(I) = \min\big(P_\text{peak},\; I \cdot B\big)$$

For small $I$, the $I \cdot B$ term is smaller and wins — you are **memory-bound**, climbing the slanted bandwidth line. For large $I$, the flat $P_\text{peak}$ term is smaller and wins — you are **compute-bound**, pinned under the horizontal ceiling. The two lines cross at exactly one intensity, and that crossing point is the most important number on the chart.

### The ridge point: where memory-bound flips to compute-bound

Set the two ceilings equal and solve for the intensity where they meet. This crossing is the **ridge point**, written $I^*$:

$$P_\text{peak} = I^* \cdot B \quad\Longrightarrow\quad I^* = \frac{P_\text{peak}}{B}$$

The ridge point is the peak FLOP per second divided by the peak bandwidth. It is the arithmetic intensity at which a workload flips from memory-bound to compute-bound. To the left of $I^*$, you are bandwidth-limited and more FLOP per second buys you nothing; to the right, you are compute-limited and more bandwidth buys you nothing. It is the single dividing line that decides which lever helps.

#### Worked example: the A100 ridge point

Let us put real numbers on it. The A100 80GB SXM has $P_\text{peak} = 312 \times 10^{12}$ bf16 FLOP/s and $B = 2.0 \times 10^{12}$ bytes/s. So:

$$I^* = \frac{312 \times 10^{12}}{2.0 \times 10^{12}} = 156 \;\frac{\text{FLOP}}{\text{byte}}$$

**An operation on an A100 must do at least 156 floating-point operations for every byte it reads from HBM, or it is memory-bound.** That is a brutal threshold. One hundred and fifty-six FLOPs per byte. Almost nothing in a neural network clears it except large matrix multiplies. A LayerNorm does about 3 FLOPs per byte. A softmax, about 4. An elementwise GELU, less than 1. Every one of those operations sits far to the left of the ridge and is, structurally, bandwidth-bound on an A100 — and no amount of extra Tensor Core throughput will ever speed them up. That is the whole reason your H100 upgrade barely helped: most of your kernels were already on the slanted part of the roof, and the H100 raised the *flat* part.

Figure 1 above is this picture: the slanted memory-bound region on the left where performance equals $I \cdot B$, the ridge point at $I^* = 156$ where the lines cross, the flat compute-bound ceiling at 312 TFLOP/s on the right, and example ops placed on each side. Notice the LayerNorm and softmax markers sit far left, the big GEMM far right. That spatial placement *is* the optimization decision, and we will make it precise op by op.

A note on why this is a log-log plot, since it matters for intuition. Intensities in real workloads span four orders of magnitude — from 0.08 for an elementwise add to over 400 for a big GEMM. Performance spans three. On a linear plot you would see a tiny cluster near the origin and one dot way off in the corner. Log-log compresses both axes so the slope-1 bandwidth line and the flat compute ceiling are both visible across the whole range, and a 10× change in intensity is the same visual distance everywhere. DSL diagrams cannot draw a true log-log scatter cleanly, so throughout this post the rooflines are drawn as labeled regions and before-after placements; the algebra is what you internalize, and the algebra is exact.

### The hierarchical roofline: one chart per memory level

So far we have treated "the bytes" as a single number — bytes moved to and from HBM — and "the bandwidth" as a single ceiling, the 2.0 TB/s of the A100's HBM. That is the textbook roofline, and it is the right first model. But a real GPU has a *memory hierarchy*, not a single memory, and each level of that hierarchy has its own bandwidth, which is far higher than HBM's. The L2 cache on an A100 delivers data at roughly 5–7 TB/s; the per-SM shared memory (SRAM) delivers at well over 19 TB/s aggregated across the chip; the register file is faster still. When you account for that, a single operation does not have one arithmetic intensity — it has *several*, one per memory level, because the byte count depends on which level you are counting traffic against.

This is the **hierarchical roofline**, and it is the version that NVIDIA's Nsight Compute actually draws. Instead of one slanted memory line, the chart has several: a slanted line for HBM (the lowest bandwidth, the lowest ceiling), a steeper-rising slanted line for L2 (higher bandwidth, higher ceiling), and another for shared memory/SRAM. Each level gets its own ridge point. The flat compute roof is shared across all of them. Your kernel is then placed against *each* line, because the same kernel moves a different number of bytes at each level — and the level it is closest to its ceiling on is the one that bottlenecks it.

The intuition is the same kitchen, but now with a pantry between the warehouse and the chefs. HBM is the warehouse across town (slow truck, the 2.0 TB/s line). L2 is a pantry in the back of the restaurant (fast, the 5–7 TB/s line). SRAM is the cutting board right next to the chef (blazing, the 19+ TB/s line). A datum that lives in the pantry is delivered far faster than one that has to come from the warehouse. So an op that re-reads the same values many times, where those values fit in L2 or SRAM, pays the *fast* bandwidth on the re-reads and the *slow* HBM bandwidth only on the first load. Its intensity measured against HBM traffic is high; its intensity measured against L2 traffic is lower. The two numbers describe the same op seen through two different lenses.

Here is why this matters in practice, and it is a subtlety that bites people who only know the one-line roofline. An op can be **HBM-bound but L2-OK** — meaning it is *not* limited by the HBM line (it has enough arithmetic intensity against HBM traffic to clear the HBM ridge), yet it is pinned against the *L2* line because it pounds the L2 cache with re-reads faster than L2 can serve them. A moderately-tiled GEMM is the classic case: a good tile size cuts HBM traffic so the op clears the HBM ridge, but if the tile is too large for shared memory it spills to L2 and the L2 bandwidth, not HBM bandwidth, becomes the wall. The fix is not "more HBM bandwidth" and not "more FLOP per second" — it is a *better tile size* that keeps the working set in SRAM. The one-line roofline cannot see that distinction; the hierarchical roofline names it exactly.

The way `ncu` surfaces this is the "GPU Speed Of Light Roofline Chart" section, which by default plots two rooflines — one for the DRAM (HBM) level and one for the L2 (or L1/shared, depending on the chart variant) level — and drops your kernel as a dot on each. The reading rule extends naturally: find the line your dot is *closest to its ceiling on*, and that names your true bottleneck level. If the dot is far below the HBM line but right up against the L2 line, the kernel is L2-bandwidth-bound, and the lever is to reduce L2 traffic (better tiling, smaller working set, more reuse out of registers) — buying a GPU with faster HBM would do nothing. Conversely a dot riding the HBM line and comfortably below the L2 line is genuinely HBM-bound, and there the HBM bandwidth, or fusion to cut HBM trips, is the real lever. You request both lines explicitly with the roofline section:

```bash
# The roofline section draws the HBM (DRAM) and L2 lines together,
# placing each kernel against both so you can see which level binds.
ncu --set roofline \
    --section SpeedOfLight_HierarchicalDoubleRooflineChart \
    -o hier_roofline python train_step.py
```

For the rest of this post we mostly work the single HBM roofline, because for the *memory-bound elementwise and normalization ops that dominate the optimization conversation* the HBM line is the binding one — those ops touch each datum once, there is no reuse for L2 to capture, and the HBM intensity is the whole story. But keep the hierarchical picture in your back pocket: the moment you are tuning a GEMM or a fused attention kernel that *does* reuse data heavily, the L2 and SRAM lines are where the real fight happens, and a dot that looks healthy against HBM can be quietly strangled by L2. The single number becomes a small vector of numbers, one per level, and the slowest level wins.

## Computing intensity for real ops: the math that matters

The roofline is only useful if you can place *your* operation on it, which means computing its arithmetic intensity by hand. This is where most of the rigor lives, so let us derive it carefully for the three operations that dominate a Transformer: a matrix multiply, a LayerNorm, and attention.

### A matrix multiply (GEMM)

Take a general matrix-matrix multiply — a **GEMM** (general matrix multiply) — of an $M \times K$ matrix by a $K \times N$ matrix, producing an $M \times N$ result. This is the workhorse of deep learning: every linear layer, every attention projection ($Q$, $K$, $V$, output), every feed-forward layer, and the final logit projection is a GEMM. For a 7B-parameter LLM, well over 95% of all FLOPs live in a handful of GEMM shapes.

**The FLOP count.** Each of the $M \times N$ output elements is a dot product of length $K$: $K$ multiplies and $K$ adds, so $2K$ FLOPs per output element. Total:

$$\text{FLOPs} = 2 \cdot M \cdot N \cdot K$$

**The byte count.** In the ideal case where every matrix is read or written exactly once, you touch the $A$ matrix ($M \cdot K$ elements), the $B$ matrix ($K \cdot N$ elements), and the $C$ output ($M \cdot N$ elements). At $s$ bytes per element (2 for bf16, 4 for fp32):

$$\text{bytes} = (M K + K N + M N) \cdot s$$

So the arithmetic intensity of a GEMM is:

$$I_\text{GEMM} = \frac{2 M N K}{(MK + KN + MN)\, s}$$

This formula contains the most important fact about GPU performance in the entire post, so we are going to stare at it. Notice that the numerator grows like the *product* of all three dimensions, while the denominator grows like the *sum* of pairwise products. When the matrices are large and roughly square, the numerator dominates massively, and intensity grows roughly linearly with the matrix dimension. **Bigger GEMMs have higher arithmetic intensity.** That is the central reason GPUs love big batches and big models: scaling up the matrices pushes you rightward across the roofline, off the memory-bound slope and onto the compute ceiling where the silicon is fully fed.

#### Worked example: intensity of a 4096-cube GEMM on an A100

Let $M = N = K = 4096$, bf16, so $s = 2$ bytes. This is a typical feed-forward GEMM shape in a mid-size LLM.

$$\text{FLOPs} = 2 \cdot 4096^3 = 1.37 \times 10^{11} \text{ FLOPs}$$

$$\text{bytes} = 3 \cdot 4096^2 \cdot 2 = 1.01 \times 10^{8} \text{ bytes}$$

$$I = \frac{1.37 \times 10^{11}}{1.01 \times 10^{8}} \approx 1365 \;\frac{\text{FLOP}}{\text{byte}}$$

That is nearly 1365 FLOP/byte — almost nine times the A100 ridge of 156. This GEMM is solidly compute-bound, and that is exactly why a well-tuned cuBLAS GEMM of this shape achieves around 280–290 TFLOP/s on an A100, roughly 90% of peak. (The figures use a more conservative $I \approx 410$ for a smaller, more typical attention-projection shape; the headline point — GEMMs clear the ridge with room to spare — is identical. The exact value depends on the shape; what matters is that it is comfortably right of 156.) For this op, the lever that helps is *faster math*: better precision (bf16 over fp32, fp8 over bf16), Tensor Cores instead of CUDA cores. Bandwidth and fusion do almost nothing, because the op is not waiting on memory.

The caveat that makes this honest: that byte count assumes each matrix is read exactly once, which only happens if the GEMM kernel is well-tiled and reuses operands out of fast on-chip SRAM. A naive GEMM that re-reads $A$ and $B$ from HBM for every output tile moves far more bytes and has *lower* effective intensity. The whole art of a fast matmul kernel — which the [memory hierarchy post](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm) covers — is tiling to hit that ideal byte count so the op stays compute-bound. The roofline tells you the ceiling; tiling is how you reach it.

### Why batching raises GEMM intensity

The same formula explains the most important practical lever in training: batch size. Take a linear layer with weight $W$ of shape $K \times N$, applied to a batch of $M$ token-vectors. The weight is fixed; only $M$ (the batch-times-sequence dimension) grows as you batch more.

With one token ($M = 1$), you read the whole $K \times N$ weight matrix to do a single $1 \times N$ output — you move $KN$ bytes of weight to do $2KN$ FLOPs, an intensity of roughly $2KN / (KN \cdot s) = 2/s = 1$ FLOP/byte for bf16. That is catastrophically memory-bound: you paid to haul the entire weight matrix across HBM and used each weight exactly once. This is precisely why **single-token decode in LLM inference is memory-bound** — you re-read every weight to generate one token, and the [efficient inference techniques](/blog/machine-learning/large-language-model/efficient-llm-inference-techniques) post is largely about amortizing that.

Now batch $M$ tokens through the same weight. The weight is still read once ($KN$ bytes), but it now feeds $2MKN$ FLOPs. Intensity climbs toward $2MKN / (KN \cdot s) = 2M/s$. **Every token you add to the batch reuses the same loaded weight one more time, so intensity grows linearly with batch size.** At $M = 256$, bf16, that is $2 \cdot 256 / 2 = 256$ FLOP/byte — comfortably right of the A100 ridge. Batching is the single cheapest way to drag a GEMM from the memory-bound slope onto the compute ceiling. It is also why training (huge effective batch) hits high MFU while single-stream inference (batch of 1) does not, and why continuous batching is the dominant serving optimization.

The same reasoning shows precisely *how far* you have to batch to clear the ridge, which is a number worth carrying around. To be compute-bound you need $2M/s \geq I^*$, so $M \geq I^* s / 2$. On an A100 bf16 ($I^* = 156$, $s = 2$) that is $M \geq 156$ — you need at least **156 tokens** flowing through a weight-bound GEMM before it tips onto the compute ceiling. Below 156 effective batch, the linear layer is memory-bound and your Tensor Cores are starved no matter how big the weights are. This is the quantitative reason a single inference request (one token, $M = 1$) is hopelessly memory-bound while a training micro-batch of 8 sequences × 2048 tokens ($M = 16384$) is deep in compute-bound territory — and it is why serving systems fight so hard to *batch requests together*, because every request they stack onto a weight read pushes $M$ up toward and past that 156 threshold.

#### Worked example: a 1×1 convolution is a GEMM in disguise

A 1×1 convolution — the pointwise convolution that dominates the channel-mixing layers of MobileNets, ResNet bottlenecks, and most modern vision backbones — is worth working through, because it looks like a "conv" and people assume convs are cheap and compute-light, when in fact a 1×1 conv is *exactly a GEMM* and obeys the same intensity law. Take an input feature map of $H \times W$ spatial positions and $C_\text{in}$ channels, a 1×1 filter producing $C_\text{out}$ channels. Each output position is a dot product of length $C_\text{in}$, computed at every one of the $H \cdot W$ positions, for every output channel. Unrolled, this is a GEMM of an $(H W) \times C_\text{in}$ matrix by a $C_\text{in} \times C_\text{out}$ weight: $M = HW$, $K = C_\text{in}$, $N = C_\text{out}$.

Put numbers on a typical mid-network layer: $H = W = 14$ (so $HW = 196$ spatial positions), $C_\text{in} = 512$, $C_\text{out} = 512$, bf16. The FLOPs are $2 \cdot 196 \cdot 512 \cdot 512 \approx 1.03 \times 10^{8}$. The bytes, reading input ($196 \cdot 512$), weight ($512 \cdot 512$), and output ($196 \cdot 512$) once each at 2 bytes, are $(196 \cdot 512 + 512 \cdot 512 + 196 \cdot 512) \cdot 2 \approx (1.0 \times 10^5 + 2.6 \times 10^5 + 1.0 \times 10^5) \cdot 2 \approx 9.3 \times 10^{5}$ bytes. So:

$$I = \frac{1.03 \times 10^{8}}{9.3 \times 10^{5}} \approx 111 \;\frac{\text{FLOP}}{\text{byte}}$$

That is **below the A100 ridge of 156** — this particular 1×1 conv is *memory-bound*, even though it is "a matmul," because the spatial dimension $HW = 196$ plays the role of the batch dimension $M$ and 196 is barely above the 156 threshold once you account for the weight and output traffic, which the simple $2M/s$ estimate ignored. The roofline's predicted ceiling is $111 \times 2.0\,\text{TB/s} \approx 222$ TFLOP/s, about 71% of peak — respectable but not saturating, and a real kernel will fall short of even that. The lever the roofline points to is *raising $M$*: process more images per batch (larger $HW$ effectively, via batch), or fuse the 1×1 conv with its neighboring batch-norm and activation so the activation traffic disappears. Drop the batch to a single image at inference and the spatial dimension alone may not feed the weight read, and the same layer slides further left into deeply memory-bound territory. The lesson is the one the GEMM formula already told us, now in vision clothing: it is the *shape*, specifically the dimension that gets reused against the weight, that decides which side of the ridge you land on — not whether the op is called a "convolution" or a "linear layer."

#### Worked example: a fused MLP block versus its unfused parts

The feed-forward block of a Transformer — the MLP that follows attention in every layer — is the best place to see fusion as a roofline move on a *mixed* op, because it interleaves compute-bound GEMMs with memory-bound elementwise work. A standard MLP is: an up-projection GEMM ($K \times 4K$), a GELU activation, and a down-projection GEMM ($4K \times K$), often with a bias add and a residual add bolted on. Consider it at $M = 2048$ tokens, $K = 4096$, bf16.

The two GEMMs are large and compute-bound: the up-projection is $2 \cdot 2048 \cdot 16384 \cdot 4096 \approx 2.7 \times 10^{11}$ FLOPs over $(2048 \cdot 4096 + 4096 \cdot 16384 + 2048 \cdot 16384) \cdot 2 \approx 2.3 \times 10^{8}$ bytes, an intensity around $1200$ — deep compute-bound, riding the ceiling. The GELU and the bias/residual adds in between are the opposite: GELU on the $2048 \times 16384$ intermediate is $\sim 8$ FLOPs per element over a read-and-write, intensity $\approx 2$, deeply memory-bound, and it must haul that whole $2048 \times 16384$ intermediate (67 million elements, 134 MB at bf16) out to HBM and back if run as a separate kernel. Run unfused, the block makes a *separate* HBM round-trip for the GELU and another for each elementwise add, and those memory-bound round-trips can easily eat 20–30% of the block's wall time despite contributing under 1% of its FLOPs — the classic "the cheap ops cost the most time" inversion the roofline predicts.

Fuse the GELU and the bias/residual adds *into the epilogue of the GEMM* — which is exactly what a fused-MLP kernel or `torch.compile` does — and the intermediate never leaves the chip: it is produced by the GEMM in registers/SRAM, the GELU is applied there, and only the final result is written to HBM. The memory-bound elementwise traffic vanishes. The block's overall intensity, which the unfused version dragged down with the GELU round-trip, climbs back toward the GEMM's, and the predicted-vs-achievable gap closes: the unfused block might achieve, say, 70% of the GEMM's compute ceiling because the GELU round-trips stall it, while the fused block achieves the GEMM's full ceiling because nothing stalls between the GEMMs. The roofline names both the disease (a memory-bound intermediate dragging a compute-bound block down) and the cure (fuse the intermediate away), and it does so *before you profile*, just from the two intensities.

### A LayerNorm

Now the other side. A **LayerNorm** normalizes each row (each token's hidden vector) of an $M \times N$ activation: subtract the mean, divide by the standard deviation, scale and shift by learned parameters. Per element it does a small constant number of FLOPs — roughly counting the mean, variance, normalization, and affine, you land around 8 FLOPs per element (the exact constant varies with implementation; call it order 5–10).

**The FLOP count:** about $8 \cdot M \cdot N$ FLOPs.

**The byte count:** you read the $M \times N$ input and write the $M \times N$ output, so at $s$ bytes per element you move at least $2 \cdot M \cdot N \cdot s$ bytes (ignoring the tiny parameter vectors of length $N$).

$$I_\text{LN} = \frac{8 M N}{2 M N \cdot s} = \frac{8}{2 s} = \frac{4}{s}$$

#### Worked example: intensity of a LayerNorm on an A100

For bf16, $s = 2$:

$$I_\text{LN} = \frac{4}{2} = 2{-}3 \;\frac{\text{FLOP}}{\text{byte}}$$

Notice the dimensions cancelled. **LayerNorm's intensity is a small constant — around 2 to 4 FLOP/byte — independent of how big the tensor is.** It does not matter whether you normalize a tiny activation or a huge one; the intensity is the same handful of FLOPs per byte, because each element is read once, lightly processed, and written once with no reuse. That constant is roughly *fifty times below* the A100 ridge of 156. LayerNorm is structurally, unavoidably memory-bound on every current GPU. Buying an H100 — three times the FLOP per second — does *not* speed up a LayerNorm at all, because the op never touches the compute ceiling. The only levers that help are (1) a higher-bandwidth GPU, or (2) fusion — fusing the LayerNorm into an adjacent op so the activation never makes a separate round-trip to HBM.

Reductions and elementwise ops are the same story in even starker form. A **reduction** — summing a tensor, computing a norm, a mean — does one FLOP per element and reads every element once, for an intensity of roughly $1/s$, about 0.25–0.5 FLOP/byte. An **elementwise** op like an add reads two inputs and writes one output, three array touches for one FLOP, intensity around 0.08 FLOP/byte. Both are *hundreds* of times below the A100 ridge. There is no shape, no size, no precision trick that lifts them off the memory-bound slope, because they are defined by touching each byte once and doing almost nothing with it. The roofline's verdict on a reduction is identical to its verdict on the elementwise add: this op is bandwidth-bound, measure it in GB/s, and the only way to make it faster is to fuse it away or move fewer bytes. This is why a "reduction" kernel that looks trivially cheap in FLOPs can nonetheless be a real fraction of your step time — it is paying full freight on bandwidth while doing no math, and bandwidth is the scarce resource.

This is figure 2: a side-by-side of the intensities of the common Transformer ops, with the GEMM clearing the ridge and everything else — LayerNorm, softmax, elementwise add, reduction — sitting far below it, color-coded by their roofline verdict.

![comparison matrix of arithmetic intensity for common operations showing big GEMM far above the ridge and LayerNorm softmax elementwise and reduction all far below it on an A100](/imgs/blogs/the-roofline-model-compute-bound-vs-memory-bound-2.png)

### Attention

Attention is the interesting case because it is a *sequence* of ops with very different intensities, and that is the whole reason FlashAttention exists. Standard attention computes $\text{softmax}(QK^\top / \sqrt{d})\,V$. Done naively, it is three steps:

1. $S = QK^\top$ — a GEMM producing the $N \times N$ score matrix (for sequence length $N$, head dim $d$). High intensity *as a GEMM*, but the output is enormous: $N^2$ elements.
2. $P = \text{softmax}(S)$ — a row-wise softmax over the $N \times N$ matrix. Low intensity (a few FLOPs per element), and it must read all $N^2$ scores and write all $N^2$ probabilities.
3. $O = PV$ — another GEMM, $N \times N$ by $N \times d$.

The killer is step 2 and the round-trips around it. The naive implementation **materializes the full $N \times N$ score matrix in HBM**, writes it, reads it back for softmax, writes the probabilities, reads them back for the second GEMM. For a sequence of 8192 tokens, $N^2 = 67$ million entries *per head per layer*, and at bf16 that is 134 MB shoved to HBM and dragged back, several times. The softmax and the round-trips dominate, and the whole attention block becomes **memory-bound** even though it contains two perfectly compute-bound GEMMs. Its effective intensity — total FLOPs divided by total HBM bytes including the $O(N^2)$ score traffic — collapses far below the ridge.

That is the precise diagnosis FlashAttention fixes, and we will return to it as a case study. The roofline does not just say "attention is slow"; it says *why*: the $O(N^2)$ HBM traffic from materializing the scores drags the intensity below the ridge. Fix the traffic, fix the intensity, fix the speed.

#### Worked example: intensity of naive attention on an A100

Let us put numbers on the attention diagnosis so it is not a hand-wave. Take one attention head with sequence length $N = 8192$ and head dimension $d = 128$, bf16 ($s = 2$ bytes), and count both the math and the bytes for the naive path.

The two GEMMs do the bulk of the FLOPs. $S = QK^\top$ is $N \times d$ times $d \times N$, which is $2 N^2 d$ FLOPs. $O = PV$ is $N \times N$ times $N \times d$, another $2 N^2 d$ FLOPs. The softmax adds a few more FLOPs per score element, on the order of $5 N^2$, which is small next to the GEMMs. So total math is roughly:

$$\text{FLOPs} \approx 4 N^2 d + 5 N^2 = 4 \cdot 8192^2 \cdot 128 + 5 \cdot 8192^2 \approx 3.4 \times 10^{10}$$

Now the bytes, and this is where the naive path bleeds. It reads $Q$, $K$, $V$ (each $N \times d$, so $3 N d \cdot s$ bytes — small), but it also **writes the full $N \times N$ score matrix $S$ to HBM, reads it back for the softmax, writes the $N \times N$ probabilities $P$, and reads them back for the second GEMM.** That is four passes over an $N^2$ matrix: $4 N^2 \cdot s$ bytes. With $N = 8192$:

$$\text{bytes} \approx 4 N^2 s + 3 N d s = 4 \cdot 8192^2 \cdot 2 + \text{(small)} \approx 5.4 \times 10^{8}$$

So the effective intensity of naive attention is:

$$I_\text{naive} = \frac{3.4 \times 10^{10}}{5.4 \times 10^{8}} \approx 63 \;\frac{\text{FLOP}}{\text{byte}}$$

That is **below the A100 ridge of 156** — naive attention is memory-bound, dragged there entirely by the four $N^2$ HBM passes. Strip those passes out (keep the scores in SRAM, never write $P$ or $S$ to HBM) and the byte count collapses to roughly the $3 N d \cdot s$ of reading $Q$, $K$, $V$ plus writing the $N \times d$ output — orders of magnitude fewer bytes. The same $3.4 \times 10^{10}$ FLOPs now divide by a tiny byte count, and the intensity rockets *past* the ridge into compute-bound territory. That is FlashAttention in one calculation: the FLOPs never changed; the bytes did. Note also that intensity here is *independent of $d$ in the dominant term* — the $N^2$ traffic, not the head dimension, is what pins naive attention to the memory-bound slope, which is exactly why the win grows with sequence length.

#### Worked example: intensity of FlashAttention on an A100

Now let us finish the calculation we just gestured at and put a real number on the FlashAttention side, so the before-and-after is two intensities you can place on the same A100 roofline rather than one number and a hand-wave. Same head: $N = 8192$, $d = 128$, bf16 ($s = 2$). The FLOPs are *identical* to the naive path — FlashAttention computes the exact same attention, the same two GEMMs and the same softmax, so the math is still $\approx 3.4 \times 10^{10}$ FLOPs. Nothing about the arithmetic changed. What changed is the denominator.

FlashAttention tiles $Q$, $K$, $V$ into blocks that fit in SRAM, streams them through, and **never materializes $S$ or $P$ in HBM at all.** So the only HBM traffic is: read $Q$, $K$, $V$ once each ($3 N d \cdot s$ bytes) and write the output $O$ once ($N d \cdot s$ bytes). That is four passes over an $N \times d$ tensor, not four passes over an $N \times N$ tensor — the $N^2$ term is gone entirely:

$$\text{bytes}_\text{flash} \approx 4 N d \cdot s = 4 \cdot 8192 \cdot 128 \cdot 2 \approx 8.4 \times 10^{6} \text{ bytes}$$

Compare that to the naive path's $5.4 \times 10^{8}$ bytes — FlashAttention moves roughly **64× fewer bytes** through HBM for the exact same computation. The intensity follows directly:

$$I_\text{flash} = \frac{3.4 \times 10^{10}}{8.4 \times 10^{6}} \approx 4050 \;\frac{\text{FLOP}}{\text{byte}}$$

Read those two numbers against the A100 ridge of 156 and the whole story is in the placement. Naive attention sits at $I \approx 63$, *left* of the ridge, on the memory-bound slope — its attainable performance is capped at $63 \times 2.0\,\text{TB/s} \approx 126$ TFLOP/s, about 40% of the chip's math peak, no matter how good the kernel is. FlashAttention sits at $I \approx 4050$, *far right* of the ridge, deep in compute-bound territory — its attainable performance is the full 312 TFLOP/s compute ceiling, because the bytes are no longer the constraint. The op physically moved from one side of the ridge to the other, and it did so by changing the denominator, not the numerator. That ratio of attainable ceilings — roughly 312 versus 126, about 2.5× — is exactly the order of the reported 2–4× wall-clock speedup, and the gap *widens with $N$*, because the naive byte count grows as $N^2$ while FlashAttention's grows as $N$. At $N = 32768$ the naive intensity is even lower and the flash intensity is even higher; the longer the sequence, the more dramatic the move across the ridge. This is the single cleanest illustration in the post of the roofline's core claim: speed is set by the side of the ridge you land on, and you choose that side by choosing how many bytes you move.

## Placing your ops: the four-quadrant decision

Now the payoff. You have an op, you compute its intensity, you compare to the ridge, and the lever is *decided for you*. Figure 3 makes the contrast concrete with the two extreme cases on the same A100.

![before and after comparison showing an elementwise add stuck on the memory-bound slope at low achieved throughput versus a large GEMM pinned at the compute ceiling near peak throughput on an A100](/imgs/blogs/the-roofline-model-compute-bound-vs-memory-bound-3.png)

On the left, an elementwise add: intensity 0.08, far down the memory-bound slope. It achieves maybe 0.16 TFLOP/s — a rounding error against the 312 TFLOP/s ceiling, but that is *fine and expected*, because at intensity 0.08 the roofline says the maximum attainable is $0.08 \times 2.0\,\text{TB/s} = 0.16$ TFLOP/s and the add is hitting it. The add is at 100% of *its* roofline even though it is at 0.05% MFU. The metric that matters for a memory-bound op is not MFU — it is achieved bandwidth, and the add is moving bytes at nearly the full 2.0 TB/s. More FLOP per second is useless; the op is already saturating the only resource it depends on.

On the right, the big GEMM: intensity in the hundreds, pinned just under the compute ceiling at ~290 TFLOP/s, 93% MFU. This op *is* what MFU measures, and it is near peak. Fusion does nothing for it (it is not memory-bound), more bandwidth does nothing (it is not bandwidth-bound). The only lever is faster math — drop to fp8 on a chip that supports it, or accept you are near peak and move on.

This is the four-quadrant logic, and it is worth stating as a flat rule:

- **Far left of the ridge (deeply memory-bound):** the op saturates bandwidth, not FLOP per second. Measure GB/s, not MFU. To go faster: fuse to cut HBM trips, or move to a higher-bandwidth GPU. Do *not* buy FLOP per second.
- **Far right of the ridge (deeply compute-bound):** the op saturates FLOP per second. Measure MFU. To go faster: better precision/Tensor Cores. Do *not* fuse or chase bandwidth.
- **Near the ridge:** balanced; both levers help a little. This is the rare comfortable case.
- **Near a ceiling but below it (low achieved vs the relevant roof):** you have a *bug* — poor occupancy, bad memory coalescing, launch overhead, a sync bubble. Profile and fix the kernel before touching hardware. The [profiling post](/blog/machine-learning/high-performance-computing/profiling-gpu-workloads-finding-the-real-bottleneck) is the tool here.

Figure 4 nails down the ridge derivation for the A100 specifically — peak math over peak bandwidth — so you can recompute it for any chip you are handed.

![before and after diagram deriving the A100 ridge point by dividing 312 teraflops per second of peak math by 2.0 terabytes per second of bandwidth to get 156 flop per byte and splitting ops into memory-bound and compute-bound sides](/imgs/blogs/the-roofline-model-compute-bound-vs-memory-bound-4.png)

## Code: a roofline calculator and an op profiler

Enough math; let us make it runnable. The roofline is only as good as your ability to compute it for your own ops and *measure* where they actually land. Here is the toolkit, in four snippets you can copy and adapt.

### A Python arithmetic-intensity and roofline calculator

This is the analytical half — given an op's FLOP count, byte count, and a GPU's specs, it tells you the intensity, the ridge, the verdict, and the attainable performance.

```python
from dataclasses import dataclass

@dataclass
class GPU:
    name: str
    peak_flops: float   # FLOP/s, e.g. 312e12 for A100 bf16
    bandwidth: float    # bytes/s, e.g. 2.0e12 for A100 HBM2e

A100 = GPU("A100-80GB-SXM", peak_flops=312e12, bandwidth=2.0e12)
H100 = GPU("H100-SXM",       peak_flops=989e12, bandwidth=3.35e12)

def roofline(flops, bytes_moved, gpu):
    I = flops / bytes_moved                      # arithmetic intensity, FLOP/byte
    ridge = gpu.peak_flops / gpu.bandwidth       # ridge point I*
    attainable = min(gpu.peak_flops, I * gpu.bandwidth)
    bound = "compute-bound" if I >= ridge else "memory-bound"
    return {
        "intensity": I,
        "ridge": ridge,
        "attainable_TFLOPs": attainable / 1e12,
        "bound": bound,
        "pct_of_peak": 100 * attainable / gpu.peak_flops,
    }

def gemm_intensity(M, N, K, dtype_bytes=2):
    flops = 2 * M * N * K
    bytes_moved = (M*K + K*N + M*N) * dtype_bytes
    return flops, bytes_moved

# A 4096-cube bf16 GEMM on an A100
f, b = gemm_intensity(4096, 4096, 4096, dtype_bytes=2)
print(roofline(f, b, A100))
# -> intensity ~1365, ridge 156.0, bound 'compute-bound', pct_of_peak ~100

# A LayerNorm over a 4096 x 4096 bf16 activation: ~8 FLOP/elem, read+write
N_elem = 4096 * 4096
print(roofline(8 * N_elem, 2 * N_elem * 2, A100))
# -> intensity ~2.0, ridge 156.0, bound 'memory-bound', pct_of_peak ~1.3%
```

Run it on every op in your model and you have a static roofline analysis before you touch a profiler. The GEMM comes back compute-bound at ~100% of peak; the LayerNorm comes back memory-bound at ~1.3% of peak. That 1.3% is not a bug — it is the roofline telling you the LayerNorm *cannot* exceed it without more bandwidth or fusion.

### Timing a PyTorch op to get achieved FLOP/s and GB/s

The analytical calculator gives the ceiling. To see where the op *actually* lands, you measure. The right way to time a CUDA kernel is with `torch.cuda.Event`, never Python's `time.time()` — CUDA launches are asynchronous, so the CPU clock measures launch overhead, not kernel time. You must synchronize and you must warm up.

```python
import torch

def time_op(fn, *args, warmup=20, iters=100):
    # Warm up: trigger autotuning, cuDNN algo selection, allocator, clocks.
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()          # wait for the GPU to finish
    ms = start.elapsed_time(end) / iters
    return ms / 1e3                   # seconds per call

device = "cuda"
M = N = K = 4096
A = torch.randn(M, K, device=device, dtype=torch.bfloat16)
B = torch.randn(K, N, device=device, dtype=torch.bfloat16)

# GEMM: 2*M*N*K FLOPs, (MK+KN+MN)*2 bytes
t = time_op(lambda: A @ B)
flops = 2 * M * N * K
bytes_moved = (M*K + K*N + M*N) * 2
print(f"GEMM: {flops/t/1e12:6.1f} TFLOP/s   {bytes_moved/t/1e9:6.1f} GB/s")
# e.g. GEMM:  ~285 TFLOP/s    ~210 GB/s   -> 91% of the 312 ceiling

# Elementwise add: 1 FLOP/elem, 3 arrays * 2 bytes touched
x = torch.randn(M, N, device=device, dtype=torch.bfloat16)
y = torch.randn(M, N, device=device, dtype=torch.bfloat16)
t = time_op(lambda: x + y)
flops = M * N
bytes_moved = 3 * M * N * 2
print(f"ADD:  {flops/t/1e12:6.3f} TFLOP/s   {bytes_moved/t/1e9:6.1f} GB/s")
# e.g. ADD:   ~0.15 TFLOP/s   ~1850 GB/s  -> 0.05% of FLOP ceiling, 93% of BW ceiling
```

Read the two lines side by side and the roofline jumps out of the numbers. The GEMM is at 91% of the *FLOP* ceiling and a trivial fraction of the bandwidth ceiling — compute-bound, exactly as predicted. The add is at 0.05% of the FLOP ceiling but **93% of the bandwidth ceiling** (1850 of 2000 GB/s) — memory-bound, also exactly as predicted, and already near-optimal *for what it is*. If someone hands you that add kernel and asks you to make it 10× faster with a better algorithm, the roofline says: you can't, not without changing the data movement. It is already riding its roof.

### Reading the roofline straight out of Nsight Compute

You do not have to hand-derive intensity for every kernel. NVIDIA's profiler, Nsight Compute (`ncu`), computes the roofline for you from hardware counters and places each kernel on the chart.

```bash
# Profile a single kernel (or a few) with the roofline section enabled.
# --launch-skip/--launch-count avoid profiling warm-up iterations.
ncu --set roofline \
    --launch-skip 20 --launch-count 5 \
    -o gemm_roofline \
    python train_step.py

# Or, the broader sweep that also includes occupancy + memory workload:
ncu --set full --section SpeedOfLight_RooflineChart \
    -o full_report python train_step.py
```

Open the `.ncu-rep` in the Nsight Compute UI and the "GPU Speed Of Light Roofline Chart" section draws your kernel as a dot on the exact plot from figure 1: the slanted memory line, the flat compute roof, the ridge, and your kernel's achieved performance versus its intensity. How to read it:

- **Dot under the slanted line, far left:** memory-bound. The "Memory Throughput %" will be high, "Compute (SM) Throughput %" low. Fuse or cut bytes.
- **Dot under the flat roof, far right:** compute-bound. SM throughput high, memory low. The op is healthy; improve precision or stop.
- **Dot well *below* whichever line it is nearest:** the kernel is leaving performance on the table — bad occupancy, uncoalesced access, launch overhead. This is the most actionable signal: it means there is a fixable inefficiency, not a hardware limit. The two summary numbers to check are **"Compute (SM) Throughput"** and **"Memory Throughput"**; whichever is higher names your bound, and if *both* are low you have a latency or occupancy problem, not a roofline problem.

`ncu` is the ground truth because it counts *actual* bytes moved (including cache misses and replays) and *actual* FLOPs executed, so it captures the effective intensity your kernel really achieves — which, for a poorly-tiled GEMM, can be far below the ideal we computed by hand. Use the analytical calculator to know the ceiling; use `ncu` to see how close you got.

### A tiny matmul-vs-add benchmark you can run right now

To feel the difference in your own hands, here is a self-contained sweep over GEMM sizes, showing intensity and achieved throughput climbing as the matrices grow — batching/sizing dragging the op rightward across the roofline.

```python
import torch

A100_PEAK, A100_BW = 312e12, 2.0e12
RIDGE = A100_PEAK / A100_BW   # 156

def bench_gemm(n):
    A = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(20): A @ B
    torch.cuda.synchronize()
    s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    s.record()
    for _ in range(50): A @ B
    e.record(); torch.cuda.synchronize()
    t = s.elapsed_time(e) / 50 / 1e3
    flops, byts = 2*n**3, 3*n*n*2
    return flops/byts, flops/t/1e12, 100*(flops/t)/A100_PEAK

print(f"ridge I* = {RIDGE:.0f} FLOP/byte")
for n in [256, 512, 1024, 2048, 4096, 8192]:
    I, tfps, mfu = bench_gemm(n)
    print(f"n={n:5d}  I={I:8.1f}  {tfps:6.1f} TFLOP/s  MFU={mfu:5.1f}%")
# n= 256  I=  85.3   ~ 40 TFLOP/s  MFU ~13%   <- below ridge, memory/launch bound
# n=4096  I=1365.3   ~285 TFLOP/s  MFU ~91%   <- well right of ridge, compute-bound
```

Watch what happens: at $n = 256$ the GEMM's intensity (85) is *below* the ridge (156) and MFU is low; as $n$ grows the intensity blows past the ridge and MFU climbs to ~90%. The size of the matmul literally walks the op across the roofline. That single sweep is the most convincing demonstration of the model — you can see the ridge point in your own measured numbers.

## Results on named hardware: A100 vs H100

Now the proof on real silicon. The roofline makes a sharp, falsifiable prediction: when you upgrade from A100 to H100, **compute-bound ops speed up a lot; memory-bound ops speed up only a little.** Let us put the numbers on it.

| GPU | Peak bf16 FLOP/s | Peak HBM bandwidth | Ridge point I* |
|---|---|---|---|
| A100 80GB SXM | 312 TFLOP/s | 2.0 TB/s (HBM2e) | **156 FLOP/byte** |
| H100 SXM | ~989 TFLOP/s | 3.35 TB/s (HBM3) | **~295 FLOP/byte** |

Sources: NVIDIA A100 and H100 datasheets; figures are the dense (non-sparse) bf16 Tensor Core peaks. The H100's peak math grew 3.2×, but its bandwidth only grew 1.67×. **Because math outran bandwidth, the ridge point moved rightward, from 156 to 295.** That is a profound and underappreciated fact: each GPU generation makes it *harder*, not easier, to be compute-bound. More of your ops fall on the memory-bound side of the ridge with every generation, which is exactly why kernel fusion and IO-aware algorithms (FlashAttention) have become more important over time, not less. The hardware is sprinting away from memory bandwidth.

Now the before→after on actual ops. The roofline predicts the speedups; here is what you measure (approximate, representative of well-tuned kernels — exact numbers depend on shape and software version):

| Operation | Intensity | A100 achieved | H100 achieved | Speedup | Why |
|---|---|---|---|---|---|
| 4096-cube GEMM (bf16) | ~1365 (compute-bound) | ~285 TFLOP/s | ~830 TFLOP/s | **~2.9×** | rides the FLOP ceiling, which grew 3.2× |
| LayerNorm (4096×4096) | ~2 (memory-bound) | ~1.85 TB/s | ~3.1 TB/s | **~1.67×** | rides the BW ceiling, which grew 1.67× |
| Elementwise GELU | ~0.5 (memory-bound) | ~1.85 TB/s | ~3.1 TB/s | **~1.67×** | bandwidth-limited, tracks BW exactly |

The pattern is unmistakable and it is *predicted by the slopes, not measured into existence*. The compute-bound GEMM gained ~2.9×, tracking the 3.2× FLOP increase. The two memory-bound ops gained ~1.67×, tracking the 1.67× bandwidth increase *exactly*. **An op's speedup across a hardware generation is set by which ceiling it rides.** If you had only ever looked at `nvidia-smi`, the LayerNorm's modest 1.67× on a "3× faster" GPU would be a mystery. With the roofline, it is arithmetic.

#### Worked example: why the H100 barely helped your training step

Here is the scenario from the intro, now solvable. Your training step is 60% GEMM time and 40% memory-bound ops (LayerNorm, softmax, residual adds, dropout, activation functions). You upgrade A100 → H100. Naive expectation: 3× faster. Roofline reality:

- The 60% that is GEMM speeds up ~2.9× → that portion's time shrinks from 60 to ~21 units.
- The 40% that is memory-bound speeds up ~1.67× → that portion shrinks from 40 to ~24 units.
- New step time: $21 + 24 = 45$ units, versus 100 before. That is **~2.2× faster, not 3×.**

And if your step were the *other* way around — 40% GEMM, 60% memory-bound, which is common for small models or short sequences — you would get $(40/2.9) + (60/1.67) = 14 + 36 = 50$ units, only **2.0×**. The more memory-bound your workload, the less a compute-heavy GPU upgrade buys you. This is Amdahl's law wearing a roofline costume: your speedup is capped by the part of the workload that does *not* benefit. The fix is not more FLOP per second — it is fusing the memory-bound 40-60% so it makes fewer HBM trips, which is the next section.

### The operational intensity of a whole model predicts MFU

Everything so far has placed *individual ops* on the roofline. But there is a beautiful and underused move: you can compute an arithmetic intensity for an *entire training step* — sum the FLOPs of every op, sum the HBM bytes of every op, and take the ratio — and that single "step intensity" predicts the MFU you can hope to achieve. This is the bridge between the per-kernel roofline and the one number leadership actually asks about, which is "what fraction of the GPU are we using."

The construction is exactly the per-op one, scaled up. The whole step's intensity is:

$$I_\text{step} = \frac{\sum_\text{ops} \text{FLOPs}}{\sum_\text{ops} \text{bytes moved to/from HBM}}$$

The numerator is the total math of the step — for a Transformer, dominated by the GEMMs, roughly $6 N_\text{params}$ FLOPs per token for a dense forward-plus-backward pass (the familiar $6ND$ training-compute rule). The denominator is every byte that crosses the HBM bus during the step: weight reads, activation reads and writes, the memory-bound elementwise round-trips, gradient and optimizer-state traffic, and — critically — all the intermediates that the unfused memory-bound ops shove out to HBM and drag back. A step composed mostly of big compute-bound GEMMs has a high $I_\text{step}$, lands right of the ridge, and can hit high MFU. A step whose GEMMs are surrounded by a thick layer of unfused LayerNorms, softmaxes, dropouts, and elementwise activations — each making its own HBM round-trip — has those round-trips piled into the denominator, which drags $I_\text{step}$ down toward the ridge and caps the achievable MFU well below peak.

This is the precise, quantitative reason MFU is hard to push past 50–60% on real models even with perfect GEMM kernels: the memory-bound "glue" ops contribute almost no FLOPs to the numerator but a large pile of bytes to the denominator, so they pull the *whole step's* intensity left even though each GEMM in isolation is far right of the ridge. You cannot reason about MFU by looking at the GEMMs alone; you have to account for the bytes the cheap ops move. And this is why the headline optimizations that raise MFU — fusing the glue ops (cutting their HBM bytes from the denominator), activation checkpointing tradeoffs, keeping intermediates on-chip — are all *roofline moves on the whole step*. They raise $I_\text{step}$ by shrinking the byte total, which slides the entire step rightward toward the compute ceiling, which is what "higher MFU" physically means.

The discipline this gives you is concrete: when your measured MFU is, say, 38% and you want to know whether that is a *fixable inefficiency* or a *structural ceiling*, compute $I_\text{step}$ and place it on the roofline. If $I_\text{step}$ is well right of the ridge but your achieved MFU is far below the compute ceiling, you have a kernel-level bug (poor tiling, occupancy, launch overhead) and the profiler is your tool. But if $I_\text{step}$ itself lands near or left of the ridge — because your step is drowning in unfused memory-bound traffic — then 38% MFU is *roughly your structural ceiling*, and the only way up is to raise $I_\text{step}$ by cutting HBM bytes, i.e. fusion and on-chip reuse, not better GEMM kernels and not a faster-math GPU. The roofline turns "is our MFU bad or is it as good as it gets" from a debate into an arithmetic check.

## Moving an op rightward: fusion as the universal memory-bound fix

If an op is memory-bound, the roofline says exactly one thing helps short of buying bandwidth: **raise its arithmetic intensity by moving fewer bytes.** And the way you move fewer bytes is fusion — combine a chain of small ops into a single kernel so the intermediate results never make a round-trip to HBM.

Here is the mechanism. A typical Transformer has chains like `add bias → scale → GELU → dropout → residual add`. Run as five separate PyTorch ops, each one reads its input from HBM, computes one cheap thing, and writes its output back to HBM — five reads and five writes, ten HBM passes over the activation, with almost no math in between. Intensity is rock-bottom. **Fuse them into one kernel** and you read the input once, keep all the intermediates in registers (chefs passing the dish hand to hand, never sending it back to the warehouse), and write the result once — two HBM passes instead of ten. You did the same FLOPs but moved one-fifth the bytes, so the intensity went up 5×, and you slid the op rightward across the roofline toward the ridge.

Figure 5 shows this move directly: five unfused kernels making ten HBM passes on the left, one fused kernel making two passes on the right, with the intensity climbing from 0.5 to 2.5.

![before and after diagram showing five unfused elementwise kernels making ten HBM round trips versus one fused kernel making two round trips raising arithmetic intensity fivefold](/imgs/blogs/the-roofline-model-compute-bound-vs-memory-bound-5.png)

In PyTorch the cheapest version of this is one line:

```python
import torch

@torch.compile  # fuses pointwise chains into single Triton kernels
def ffn_chain(x, b, w):
    h = x + b
    h = h * 0.5
    h = torch.nn.functional.gelu(h)
    h = torch.nn.functional.dropout(h, p=0.1, training=True)
    return h + w   # residual add

# Eager runs ~5 kernels (10 HBM passes); compiled fuses to ~1 (2 passes).
# On an A100 the fused version is typically 2-4x faster for these chains,
# and the speedup is bandwidth, not FLOPs: you moved fewer bytes.
```

`torch.compile`'s inductor backend generates a single fused Triton kernel for that pointwise chain. The measured win is typically 2–4× *for the chain*, and — crucially — the roofline tells you the win is a *bandwidth* win: you did not do less math, you moved fewer bytes. That is why fusion helps memory-bound ops and does *nothing* for the already-compute-bound GEMM. If you tried to "fuse" a giant standalone matmul, there is nothing to save — it was never making redundant HBM trips. **Fusion is a memory-bound-only lever, and the roofline is what tells you, before you try, whether an op has room to be fused.** This is the central idea of the [kernel fusion and FlashAttention post](/blog/machine-learning/high-performance-computing/kernel-fusion-and-flashattention-beating-the-memory-wall), which goes deep on the mechanics; here we just place it on the roofline.

Figure 6 makes the A100-vs-H100 ridge shift visual — both ceilings rise, but the math ceiling rises faster, so the ridge marches right and more of your ops fall into the memory-bound region that fusion targets.

![comparison matrix of A100 and H100 rooflines showing peak FLOP per second bandwidth and ridge point with the ridge climbing from 156 to 295 because compute grew faster than bandwidth](/imgs/blogs/the-roofline-model-compute-bound-vs-memory-bound-6.png)

## Case studies and real numbers

Theory placed; now the canonical real-world stories the roofline explains. Three of them, with citations, because the roofline is most convincing when it predicts a result someone already published.

### FlashAttention: turning a memory-bound op compute-bound

This is the textbook roofline success story. As we derived, naive attention is memory-bound because it materializes the $N \times N$ score matrix in HBM and reads it back several times for the softmax, generating $O(N^2)$ HBM traffic that drags the effective intensity far below the ridge. The two GEMMs inside attention are compute-bound; the *plumbing* around them is not, and the plumbing dominates.

[FlashAttention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135) is, at its core, a roofline intervention. It tiles $Q$, $K$, and $V$ into blocks that fit in on-chip SRAM (the fast shared memory next to the compute units), computes the softmax *online* (an incremental running-max-and-sum trick so you never need the whole row at once), and **never writes the $N \times N$ score matrix to HBM at all.** The HBM traffic drops from $O(N^2)$ to $O(N)$. With the dominant memory traffic gone, the effective arithmetic intensity of the attention block jumps, the op crosses toward the ridge, and it becomes compute-bound — riding the GEMM ceiling instead of the bandwidth slope. The reported result is a **2–4× wall-clock speedup** and, just as important, a memory footprint that drops from $O(N^2)$ to $O(N)$, which is what makes long context lengths feasible at all. FlashAttention-2 and FlashAttention-3 pushed this further with better work partitioning and, on Hopper, fp8 and asynchrony.

Figure 7 is this story on the roofline: naive attention pinned to the memory-bound slope because it materializes the scores, FlashAttention lifted toward the compute ceiling because it keeps the scores in SRAM.

![before and after diagram of naive attention materializing the N by N score matrix in HBM as memory-bound versus FlashAttention tiling in SRAM with online softmax as compute-bound and two to four times faster](/imgs/blogs/the-roofline-model-compute-bound-vs-memory-bound-7.png)

The general lesson is bigger than attention: **any op whose intensity is dragged down by an intermediate that does not need to exist in HBM can be rescued by keeping that intermediate on-chip.** FlashAttention is the most famous instance, but the same move powers fused LayerNorm, fused softmax, fused optimizers (`apex`'s fused Adam), and the entire `torch.compile` fusion engine. The roofline is the lens that tells you the move is *possible* — if the op is left of the ridge and there is an intermediate making redundant HBM trips, there is a fusion win waiting.

### The memory-bound kernel no FLOP/s can save

The flip side, and a humbling one. Consider a pure elementwise kernel — a large scalar-times-tensor multiply, or a bias add, or a copy — with intensity well under 1 FLOP/byte. The roofline ceiling for an intensity-0.25 op on an A100 is $0.25 \times 2.0\,\text{TB/s} = 0.5$ TFLOP/s, which is 0.16% of the 312 TFLOP/s peak. **There is no algorithm, no precision change, no Tensor Core trick that makes this kernel exceed 0.5 TFLOP/s on an A100, because it is bounded by bandwidth and it already saturates bandwidth.** A team chasing higher MFU on such a kernel is chasing a number that is *structurally* impossible to raise; the only honest moves are (1) fuse it into a neighbor so it stops being a separate kernel, (2) move to a higher-bandwidth GPU, or (3) reduce the data — quantize the tensor to fewer bytes (which, note, *raises* intensity by shrinking the denominator). This is the case the [edge-AI inference runtimes comparison](/blog/machine-learning/edge-ai/inference-runtimes-compared) hits constantly on memory-starved devices, where bandwidth, not FLOP per second, is the binding constraint and a "faster" NPU does nothing.

The discipline here is to *not measure memory-bound ops in MFU*. MFU is the right north-star for the compute-bound parts of a workload (the GEMMs), but for memory-bound ops the right metric is **achieved bandwidth as a fraction of peak**. An add at 93% of peak bandwidth is a *triumph*; reporting it as "0.05% MFU" and panicking is a category error the roofline cures.

To make this concrete with measured numbers rather than asserted ones: a well-written elementwise or copy kernel on an A100 routinely reaches 1.85–1.95 TB/s of achieved HBM bandwidth against the 2.0 TB/s spec — roughly 92–97% of peak (approximate, representative of a coalesced bandwidth-bound kernel; exact numbers depend on access pattern and driver). NVIDIA's own `bandwidthTest` and the widely-used STREAM-style GPU benchmarks land in the same band on the A100, and that is the practical ceiling such a kernel can hit. The point is not the exact percentage; it is that the kernel is *already at its roof*. When a vendor's saxpy or a framework's add kernel reports ~1.9 TB/s, there is no software change that improves it, because the roof is 2.0 TB/s and you are at 95% of it. The only honest next steps are the three the roofline names — fuse, upgrade bandwidth, or shrink the data — and a team that keeps "optimizing the add kernel" past that point is sanding a surface that is already flat.

### LLM decode: the whole serving problem is one ridge point

The single most consequential roofline fact in production today: **autoregressive LLM decode is memory-bound.** Generating one token re-reads every weight in the model to produce a single new vector — batch size 1 through a weight matrix, which (as we derived) gives an intensity of about $2/s$, roughly 1 FLOP/byte in bf16, catastrophically left of the ridge. A 70B-param model in bf16 is 140 GB of weights; reading all of it from HBM at 2.0 TB/s takes ~70 ms *just to move the bytes*, and that 70 ms is the floor on your per-token latency no matter how fast the math is. The compute is nearly free; the bandwidth is everything.

This is why every serious LLM serving optimization is a roofline move to raise decode intensity: **continuous batching** stacks many requests through the same weight read (raising $M$, raising intensity, exactly like training batch size); **paged attention** (vLLM) manages the KV-cache so you can fit a bigger batch; **speculative decoding** verifies several tokens per weight read; **quantization** to int8/int4 shrinks the bytes moved, directly raising intensity. The [serving-at-scale post](/blog/machine-learning/large-language-model/serving-llms-at-scale-production-systems) and the [KV-cache post](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management) are, fundamentally, applied roofline. Knowing that decode lives on the memory-bound slope tells you, before you write a line of serving code, that the levers are batching and byte-reduction — never a faster math unit.

#### Worked example: a decode step bound by the KV-cache, not the weights

There is a second memory-bound term in decode that grows with context length, and it is worth a number because it dominates long-context serving. Beyond re-reading the weights, each decode step must read the *entire KV-cache* — the stored keys and values for every previous token, every layer, every head — to attend over the context. The KV-cache size for a single sequence is roughly $2 \cdot L \cdot N_\text{ctx} \cdot n_\text{heads} \cdot d_\text{head} \cdot s$ bytes (the factor 2 for K and V, across $L$ layers and $N_\text{ctx}$ context tokens). For a 7B-class model — say $L = 32$ layers, hidden 4096 ($n_\text{heads} \cdot d_\text{head} = 4096$), bf16 — and a context of $N_\text{ctx} = 32{,}000$ tokens, that is $2 \cdot 32 \cdot 32000 \cdot 4096 \cdot 2 \approx 3.4 \times 10^{10}$ bytes, about **34 GB of KV-cache for one sequence** (approximate; the exact figure depends on grouped-query-attention head sharing, which shrinks it).

Now place it on the roofline. To generate one token at that context, decode must stream those ~34 GB of KV through HBM (plus the ~14 GB of weights for a bf16 7B, roughly), and it does only a few FLOPs per KV byte — the attention dot products against the cached keys and values are themselves low-intensity, because each cached key is read once and multiplied into a single query. So the *KV-attention portion of decode is memory-bound on the cache itself*, and at long context the cache traffic can exceed the weight traffic. Streaming 34 GB at the A100's 2.0 TB/s takes ~17 ms; add the ~7 ms to read 14 GB of weights and the per-token floor is ~24 ms *purely from moving bytes*, with the math contributing almost nothing. This is why long-context decode gets slower per token as the conversation grows even though the model is unchanged — the KV-cache, not the weights, becomes the dominant byte source, and the roofline says plainly that the levers are (1) shrink the KV-cache (grouped-query / multi-query attention, KV quantization to int8 or int4, which directly halves or quarters those bytes), and (2) batch sequences so the *weight* read is amortized, though note the KV read is *per-sequence* and does not amortize across a batch the way weights do. That asymmetry — weights amortize across a batch, KV does not — is itself a roofline insight, and it is why KV-cache compression became one of the hottest areas in serving: it attacks the one memory-bound term that batching cannot fix.

## When the roofline lies (and how to keep it honest)

The roofline is a model, and every model has a domain of validity. A staff engineer knows where it breaks, so here are the four caveats that keep you from over-trusting it.

**It assumes you reach the relevant ceiling.** The roofline gives the *upper bound*. A kernel can land well below both the memory line and the compute roof because of poor occupancy (not enough warps to hide latency), uncoalesced memory access (each thread fetching scattered bytes, so effective bandwidth is a fraction of peak), launch overhead (a kernel so small the launch dominates), or a sync bubble. When `ncu` shows a dot floating below *both* lines, the roofline is not your problem — your kernel is leaving performance on the table and you should profile the occupancy and memory-access patterns. The roofline tells you the speed limit; it does not tell you you are speeding.

**Effective intensity depends on caching.** Our hand-computed byte counts assume each datum is read from HBM exactly once. Real kernels hit the L2 cache, so a value re-read soon after may not cost an HBM trip. This *raises* effective intensity above the naive estimate. It is why a tiled GEMM beats a naive one: same FLOPs, but tiling keeps operands in SRAM/L2 and cuts HBM bytes, pushing intensity up. Use `ncu`'s measured byte counts for the truth; use the hand calc for the order of magnitude.

**The peak you divide by must match the op.** The A100's 312 TFLOP/s is the *bf16 Tensor Core* peak. If your op runs in fp32 on the regular CUDA cores, the relevant peak is ~19.5 TFLOP/s, and the ridge point is $19.5\text{e}12 / 2.0\text{e}12 \approx 10$ FLOP/byte — a *completely different* chart on which far more ops are compute-bound. Always use the peak for the *precision and units* your op actually runs in. This is also why mixed precision is a roofline lever: dropping fp32→bf16 raises the FLOP ceiling 16× *and* halves the bytes moved, a double win that the [mixed-precision post](/blog/machine-learning/high-performance-computing/numerical-formats-and-mixed-precision-fp32-tf32-bf16-fp16-fp8) covers in full.

**It is a single-GPU model.** The classic roofline says nothing about the network. Once you scale to many GPUs, a *third* ceiling appears — communication bandwidth (NVLink, InfiniBand) — and an all-reduce can become the bottleneck even when every GPU's local compute is perfectly compute-bound. There are "hierarchical" and "communication" rooflines that add this axis, and the [collectives post](/blog/machine-learning/high-performance-computing/parallelism-strategies-data-tensor-pipeline-and-expert) takes it up. For a single op on a single GPU, though, the two-ceiling roofline is exact and complete.

## The decision, as a flowchart

Step back and the roofline collapses into a procedure you can run in your head on any kernel. Figure 8 is that procedure as a decision tree: plot the op, compare its intensity to the ridge, and the lever is read off for you.

![decision tree for reading the optimization off the roofline branching from plotting an op into a memory-bound left side fixed by fusion or bandwidth and a compute-bound right side fixed by Tensor Cores and lower precision](/imgs/blogs/the-roofline-model-compute-bound-vs-memory-bound-8.png)

In words:

1. **Compute the op's arithmetic intensity** $I = \text{FLOPs} / \text{bytes}$ — by hand with the calculator, or read it from `ncu --set roofline`.
2. **Compute the GPU's ridge point** $I^* = P_\text{peak}/B$ for the *precision your op runs in* (156 for A100 bf16; 295 for H100 bf16; ~10 for A100 fp32).
3. **If $I < I^*$ (memory-bound):** the lever is fewer bytes. Fuse the op into its neighbors (`torch.compile`, a Triton kernel, FlashAttention for attention), quantize to shrink the data, or move to a higher-bandwidth GPU. Measure success in **GB/s vs peak**, not MFU. Do not buy FLOP per second.
4. **If $I > I^*$ (compute-bound):** the lever is faster math. Use Tensor Cores, drop to bf16/fp8, ensure good tiling. Measure success in **MFU**. Do not chase fusion or bandwidth.
5. **If the op is below *both* lines:** it is not a roofline problem — it is an efficiency bug. Profile occupancy, memory coalescing, and launch overhead, and fix the kernel.

That is the whole model. Five steps, one number, and you never guess again.

## When to reach for the roofline (and when not to)

The roofline is the *first* thing to reach for whenever you ask "why is this slow" or "what should I optimize." It is cheap — a back-of-envelope calculation gives you the bound and the direction before you spend an hour in a profiler. Reach for it:

- **Before buying hardware.** If your workload is 50% memory-bound, a GPU with 3× the FLOP per second but 1.5× the bandwidth gives you nowhere near 3×. The roofline says so in one line of arithmetic and can save a six-figure procurement mistake.
- **Before optimizing a kernel.** It tells you whether fusion or precision is the right lever, so you do not spend a week fusing an op that was already compute-bound (and gain nothing).
- **To set realistic targets.** "This LayerNorm is at 1.3% MFU" sounds like failure until the roofline shows 1.3% is its ceiling. The roofline turns a panic into a correct "this is optimal, move on."

When *not* to lean on it:

- **When the op is below both ceilings.** Then it is an occupancy/coalescing/launch bug, and the roofline only tells you the unreachable limit. Go to the profiler.
- **When the bottleneck is not on the GPU at all.** A data-loader stall, a CPU preprocessing bottleneck, or a sync barrier between ranks will pin your GPUs idle, and no roofline analysis of the kernels will find it. The timeline view in Nsight Systems will. The roofline is a *per-kernel* model; it cannot see a host-side stall.
- **At multi-GPU scale, by itself.** Add the communication ceiling. A compute-bound forward pass can still be wall-clock-dominated by an all-reduce; the single-GPU roofline is blind to that.

## Key takeaways

- **Arithmetic intensity $I = \text{FLOPs}/\text{bytes}$ is the one number that decides everything.** It is a property of the algorithm and data layout, not the hardware.
- **The roofline is $P_\text{attainable} = \min(P_\text{peak}, I \cdot B)$:** a flat compute ceiling and a slanted memory line that cross at the ridge point $I^* = P_\text{peak}/B$.
- **The ridge point is the dividing line.** Left of it you are memory-bound (more FLOP/s is useless); right of it you are compute-bound (more bandwidth is useless). On an A100 bf16, $I^* = 156$; on an H100 bf16, ~295.
- **Big GEMMs are compute-bound; LayerNorm, softmax, elementwise ops, and reductions are memory-bound.** Their intensity is a small constant independent of tensor size, structurally far below the ridge.
- **Batching raises GEMM intensity linearly** — each token reuses the loaded weight once more — which is why training hits high MFU and single-token decode does not.
- **Fusion is the universal memory-bound fix:** it raises intensity by moving fewer bytes (fewer HBM trips), sliding the op rightward. It does nothing for compute-bound ops.
- **Each GPU generation moves the ridge rightward** because peak math grows faster than bandwidth — so fusion and IO-aware kernels matter *more* over time, not less.
- **Measure memory-bound ops in GB/s-vs-peak, compute-bound ops in MFU.** An add at 93% of peak bandwidth is optimal even at 0.05% MFU; do not panic at the wrong metric.
- **The roofline gives the ceiling, not the achievement.** A kernel below both lines is an efficiency bug — profile it.

## Further reading

- [Williams, Waterman, and Patterson, "Roofline: An Insightful Visual Performance Model" (2009)](https://dl.acm.org/doi/10.1145/1498765.1498785) — the original paper that introduced the model.
- [NVIDIA A100 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet.pdf) and the H100 datasheet — the peak FLOP/s and HBM bandwidth specs used throughout.
- [Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (2022)](https://arxiv.org/abs/2205.14135) — the canonical memory-bound-to-compute-bound rewrite.
- [NVIDIA Nsight Compute Roofline documentation](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) — how `ncu --set roofline` builds the chart from hardware counters.
- Series intro: [Why HPC Is the Bottleneck for Modern AI](/blog/machine-learning/high-performance-computing/why-hpc-is-the-bottleneck-for-modern-ai) and the memory-side companion [The Memory Hierarchy: Registers, Shared Memory, and HBM](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm).
- Where the roofline becomes action: [Kernel Fusion and FlashAttention](/blog/machine-learning/high-performance-computing/kernel-fusion-and-flashattention-beating-the-memory-wall), [Profiling GPU Workloads](/blog/machine-learning/high-performance-computing/profiling-gpu-workloads-finding-the-real-bottleneck), and the [capstone HPC playbook](/blog/machine-learning/high-performance-computing/the-hpc-playbook-for-ai-engineers).
- For the compute-optimal angle on *how much* compute to spend in the first place: [Chinchilla Compute-Optimal Scaling](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling).
